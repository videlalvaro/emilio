"""qwen36_hf_golden_capture.py — Phase 0 HF golden capture for Qwen3.6.

Captures, for a small prompt suite:

- prompt ids
- prompt logits
- per-layer hidden states
- MoE router top-k indices and weights (best-effort via hooks)
- greedy next-token ids and per-step logits
- one cache snapshot for the selected prompt, to preserve DeltaNet / KV state

Writes:
  python/moe/out/qwen36/qwen36_golden__<suiteHash>__<shaShort>.npz
  python/moe/out/qwen36/qwen36_golden.npz
  python/moe/out/qwen36/.qwen36_golden_PASS

Usage:
  .venv313/bin/python python/moe/qwen36_hf_golden_capture.py
  .venv313/bin/python python/moe/qwen36_hf_golden_capture.py --prompt-key two_plus_two
    .venv313/bin/python python/moe/qwen36_hf_golden_capture.py --offload --limit 1 --n-new 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

assert "venv313" in sys.executable, (
    f"Wrong interpreter: {sys.executable!r}. Use .venv313/bin/python."
)

import numpy as np
import torch
import transformers

from qwen36_phase0_spec import (
    DEFAULT_N_NEW,
    GOLDEN_LATEST_LINK,
    GOLDEN_SENTINEL,
    MODEL_DIR,
    OUT_DIR,
    PROMPT_SUITE,
)

LAYER_NAME_RE = re.compile(r"layers\.(\d+)")


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def _require_disk_free(paths: list[Path], min_free_gb: int) -> None:
    checked: dict[str, float] = {}
    for path in paths:
        probe = path if path.exists() else path.parent
        resolved = str(probe.resolve())
        if resolved in checked:
            continue
        free_bytes = shutil.disk_usage(probe).free
        free_gb = free_bytes / (1024 ** 3)
        checked[resolved] = free_gb
        print(f"  disk free @ {probe}: {free_gb:.1f} GiB")
        if free_gb < min_free_gb:
            raise SystemExit(
                f"refusing to run: {probe} has {free_gb:.1f} GiB free, "
                f"needs at least {min_free_gb} GiB"
            )


def _atomic_write_npz(target: Path, payload: dict[str, np.ndarray]) -> None:
    tmp_base = target.with_suffix(target.suffix + ".tmp")
    tmp_written = Path(str(tmp_base) + ".npz")
    np.savez(str(tmp_base), **payload)
    assert tmp_written.exists(), f"savez did not produce {tmp_written}"
    with open(tmp_written, "rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp_written, target)


def _atomic_symlink(link: Path, target_name: str) -> None:
    tmp_link = link.with_suffix(link.suffix + ".tmp")
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    tmp_link.symlink_to(target_name)
    os.replace(tmp_link, link)


def _load_json(path: Path) -> dict:
    with open(path) as handle:
        return json.load(handle)


def _select_prompts(args: argparse.Namespace) -> list[tuple[str, str]]:
    selected = PROMPT_SUITE
    if args.prompt_key is not None:
        selected = [entry for entry in PROMPT_SUITE if entry[0] == args.prompt_key]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise SystemExit("no prompts selected")
    return selected


def _prompt_suite_hash(selected: list[tuple[str, str]]) -> str:
    blob = json.dumps(selected, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


def _model_sha8(model_dir: Path) -> str:
    index_path = model_dir / "model.safetensors.index.json"
    config_path = model_dir / "config.json"
    if index_path.exists():
        return hashlib.sha256(index_path.read_bytes()).hexdigest()[:8]
    return hashlib.sha256(config_path.read_bytes()).hexdigest()[:8]


def _decoder_layers(model: torch.nn.Module) -> Any:
    text_model = getattr(model, "model", None) or model
    for attr in ("layers", "decoder_layers"):
        if hasattr(text_model, attr):
            return getattr(text_model, attr)
        if hasattr(text_model, "language_model") and hasattr(text_model.language_model, attr):
            return getattr(text_model.language_model, attr)
    raise SystemExit("could not resolve decoder layers on the HF model")


def _parse_gate_output(output: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if isinstance(output, tuple) and len(output) >= 3 and all(torch.is_tensor(x) for x in output[:3]):
        return output[0], output[1], output[2]
    attr_sets = (
        ("logits", "weights", "indices"),
        ("router_logits", "routing_weights", "selected_experts"),
        ("router_logits", "weights", "indices"),
    )
    for attrs in attr_sets:
        if all(hasattr(output, attr) for attr in attrs):
            values = tuple(getattr(output, attr) for attr in attrs)
            if all(torch.is_tensor(x) for x in values):
                return values  # type: ignore[return-value]
    return None


def _capture_router_hooks(model: torch.nn.Module, state: dict[str, Any]) -> list[Any]:
    handles = []
    for name, module in model.named_modules():
        if not hasattr(module, "gate") or not hasattr(module, "shared_expert"):
            continue
        match = LAYER_NAME_RE.search(name)
        layer_index = int(match.group(1)) if match else None
        if layer_index is None:
            continue

        def _hook(current_module: torch.nn.Module, inputs: tuple[Any, ...], layer_index: int = layer_index) -> None:
            if not state.get("enabled"):
                return
            hidden_states = inputs[0]
            if not torch.is_tensor(hidden_states):
                state["captures"][layer_index] = {"error": "non-tensor input to router hook"}
                return
            flat_hidden = hidden_states.detach().reshape(-1, hidden_states.shape[-1])
            try:
                parsed = _parse_gate_output(current_module.gate(flat_hidden))
            except Exception as exc:  # pragma: no cover - defensive for HF drift
                state["captures"][layer_index] = {"error": repr(exc)}
                return
            if parsed is None:
                state["captures"][layer_index] = {"error": "unrecognized gate output schema"}
                return
            router_logits, topk_weights, topk_indices = parsed
            tokens = hidden_states.shape[1] if hidden_states.ndim == 3 else flat_hidden.shape[0]
            state["captures"][layer_index] = {
                "logits": router_logits.reshape(tokens, -1).float().cpu().numpy().astype(np.float32),
                "topk_weights": topk_weights.reshape(tokens, -1).float().cpu().numpy().astype(np.float32),
                "topk_indices": topk_indices.reshape(tokens, -1).cpu().numpy().astype(np.int32),
            }

        handles.append(module.register_forward_pre_hook(_hook))
    return handles


def _contains_tensor(obj: Any, depth: int = 0) -> bool:
    if depth > 6:
        return False
    if torch.is_tensor(obj):
        return True
    if isinstance(obj, (list, tuple)):
        return any(_contains_tensor(item, depth + 1) for item in obj)
    if isinstance(obj, dict):
        return any(_contains_tensor(item, depth + 1) for item in obj.values())
    if hasattr(obj, "to_legacy_cache"):
        return True
    if hasattr(obj, "__dict__"):
        return any(_contains_tensor(item, depth + 1) for item in vars(obj).values())
    return False


def _flatten_tensor_tree(
    obj: Any,
    prefix: str,
    payload: dict[str, np.ndarray],
    meta: dict[str, Any],
    visited: set[int],
) -> None:
    obj_id = id(obj)
    if obj_id in visited:
        return
    visited.add(obj_id)

    if torch.is_tensor(obj):
        tensor = obj.detach().cpu()
        if tensor.is_floating_point():
            tensor = tensor.to(torch.float32)
        payload[prefix] = tensor.numpy()
        meta[prefix] = {"shape": list(payload[prefix].shape), "dtype": str(obj.dtype)}
        return

    if hasattr(obj, "to_legacy_cache"):
        _flatten_tensor_tree(obj.to_legacy_cache(), prefix, payload, meta, visited)
        return

    if isinstance(obj, (list, tuple)):
        for index, item in enumerate(obj):
            _flatten_tensor_tree(item, f"{prefix}__{index}", payload, meta, visited)
        return

    if isinstance(obj, dict):
        for key, item in sorted(obj.items()):
            safe_key = str(key).replace(".", "_").replace("/", "_")
            _flatten_tensor_tree(item, f"{prefix}__{safe_key}", payload, meta, visited)
        return

    if hasattr(obj, "__dict__"):
        public_items = {
            name: value
            for name, value in vars(obj).items()
            if not name.startswith("_") and _contains_tensor(value)
        }
        if public_items:
            _flatten_tensor_tree(public_items, prefix, payload, meta, visited)
            return

    meta[prefix] = {"repr": repr(obj), "type": type(obj).__name__}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--n-new", type=int, default=DEFAULT_N_NEW)
    parser.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--offload-folder", type=Path, default=OUT_DIR / ".offload_qwen36")
    parser.add_argument("--max-cpu-mem", default="40GiB")
    parser.add_argument("--max-disk-mem", default="200GiB")
    parser.add_argument("--disk-free-min-gb", type=int, default=250)
    parser.add_argument("--prompt-key", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--capture-cache-key", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if GOLDEN_SENTINEL.exists() and not args.force:
        print(f"sentinel exists: {GOLDEN_SENTINEL} (delete it or pass --force)")
        sys.exit(0)

    if not args.model_dir.exists():
        raise SystemExit(f"missing model dir: {args.model_dir}")
    index_path = args.model_dir / "model.safetensors.index.json"
    config_path = args.model_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"missing config file: {config_path}")

    selected = _select_prompts(args)
    capture_cache_key = args.capture_cache_key or selected[0][0]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.offload:
        args.offload_folder.mkdir(parents=True, exist_ok=True)

    disk_paths = [args.model_dir, args.out_dir]
    if args.offload:
        disk_paths.append(args.offload_folder)
    _require_disk_free(disk_paths, args.disk_free_min_gb)

    torch.set_num_threads(os.cpu_count() or 1)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    dtype = _dtype_from_name(args.dtype)

    print("=== Qwen3.6 HF golden capture ===")
    print(f"  executable: {sys.executable}")
    print(f"  model_dir: {args.model_dir}")
    print(f"  selected prompts: {[key for key, _ in selected]}")
    print(f"  capture-cache-key: {capture_cache_key}")
    print(f"  dtype: {args.dtype}")
    print(f"  offload: {args.offload}")
    if args.offload:
        print(f"  offload-folder: {args.offload_folder}")
        print(f"  max-cpu-mem: {args.max_cpu_mem}")
        print(f"  max-disk-mem: {args.max_disk_mem}")
    print(f"  transformers={transformers.__version__}  torch={torch.__version__}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    t0 = time.perf_counter()
    if args.offload:
        model = AutoModelForCausalLM.from_pretrained(
            str(args.model_dir),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
            max_memory={"cpu": args.max_cpu_mem, "disk": args.max_disk_mem},
            offload_folder=str(args.offload_folder),
            offload_state_dict=True,
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(args.model_dir),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="cpu",
        ).eval()
    print(f"  load wall: {time.perf_counter() - t0:.1f}s")

    decoder_layers = _decoder_layers(model)
    config = _load_json(config_path)
    model_sha8 = _model_sha8(args.model_dir)
    suite_hash = _prompt_suite_hash(selected)
    output_path = args.out_dir / f"qwen36_golden__{suite_hash}__{model_sha8}.npz"

    router_state: dict[str, Any] = {"enabled": False, "captures": {}}
    hook_handles = _capture_router_hooks(model, router_state)

    payload: dict[str, np.ndarray] = {
        "n_prompts": np.array(len(selected), dtype=np.int64),
        "n_new": np.array(args.n_new, dtype=np.int64),
        "model_dir": np.array(str(args.model_dir)),
        "model_sha8": np.array(model_sha8),
        "suite_hash": np.array(suite_hash),
        "transformers_version": np.array(transformers.__version__),
        "torch_version": np.array(torch.__version__),
        "n_layers": np.array(len(decoder_layers), dtype=np.int64),
        "layer_types_json": np.array(json.dumps(config.get("layer_types", []))),
    }

    try:
        for prompt_index, (prompt_key, prompt_text) in enumerate(selected):
            print(f"  prompt[{prompt_index}] {prompt_key}: {prompt_text!r}")
            encoded = tokenizer(prompt_text, return_tensors="pt")
            input_ids = encoded["input_ids"]
            router_state["enabled"] = True
            router_state["captures"] = {}
            with torch.no_grad():
                prompt_out = model(
                    input_ids=input_ids,
                    use_cache=True,
                    output_hidden_states=True,
                )
            router_state["enabled"] = False

            prompt_logits = prompt_out.logits[0].float().cpu().numpy().astype(np.float32)
            hidden_stack = np.stack(
                [hidden[0].float().cpu().numpy().astype(np.float32) for hidden in prompt_out.hidden_states],
                axis=0,
            )

            payload[f"p{prompt_index}_key"] = np.array(prompt_key)
            payload[f"p{prompt_index}_prompt"] = np.array(prompt_text)
            payload[f"p{prompt_index}_prompt_ids"] = input_ids[0].cpu().numpy().astype(np.int32)
            payload[f"p{prompt_index}_prompt_logits"] = prompt_logits
            payload[f"p{prompt_index}_hidden_states"] = hidden_stack

            for layer_index, capture in sorted(router_state["captures"].items()):
                if "error" in capture:
                    payload[f"p{prompt_index}_L{layer_index:02d}_router_error"] = np.array(capture["error"])
                    continue
                payload[f"p{prompt_index}_L{layer_index:02d}_router_logits"] = capture["logits"]
                payload[f"p{prompt_index}_L{layer_index:02d}_topk_weights"] = capture["topk_weights"]
                payload[f"p{prompt_index}_L{layer_index:02d}_topk_indices"] = capture["topk_indices"]

            past_key_values = prompt_out.past_key_values
            generated_ids: list[int] = []
            step_logits: list[np.ndarray] = []
            cached_payload: dict[str, np.ndarray] = {}
            cached_meta: dict[str, Any] = {}

            next_token = int(prompt_logits[-1].argmax())
            current = torch.tensor([[next_token]], dtype=torch.long)
            for step_index in range(args.n_new):
                with torch.no_grad():
                    step_out = model(input_ids=current, past_key_values=past_key_values, use_cache=True)
                past_key_values = step_out.past_key_values
                logits_row = step_out.logits[0, -1].float().cpu().numpy().astype(np.float32)
                step_logits.append(logits_row)
                generated_ids.append(int(current.item()))
                if step_index == 0 and prompt_key == capture_cache_key:
                    _flatten_tensor_tree(
                        past_key_values,
                        f"p{prompt_index}_decode_cache",
                        cached_payload,
                        cached_meta,
                        set(),
                    )
                current = torch.tensor([[int(logits_row.argmax())]], dtype=torch.long)

            payload[f"p{prompt_index}_gen_ids"] = np.array(generated_ids, dtype=np.int32)
            payload[f"p{prompt_index}_step_logits"] = np.stack(step_logits, axis=0).astype(np.float32)

            if prompt_key == capture_cache_key:
                prompt_cache_meta: dict[str, Any] = {}
                _flatten_tensor_tree(
                    prompt_out.past_key_values,
                    f"p{prompt_index}_prompt_cache",
                    payload,
                    prompt_cache_meta,
                    set(),
                )
                payload[f"p{prompt_index}_cache_meta_json"] = np.array(
                    json.dumps({**prompt_cache_meta, **cached_meta}, sort_keys=True)
                )
                payload.update(cached_payload)

    finally:
        for handle in hook_handles:
            handle.remove()

    _atomic_write_npz(output_path, payload)
    if GOLDEN_LATEST_LINK.exists() or GOLDEN_LATEST_LINK.is_symlink():
        GOLDEN_LATEST_LINK.unlink()
    _atomic_symlink(GOLDEN_LATEST_LINK, output_path.name)
    GOLDEN_SENTINEL.write_text(
        (
            f"suite_hash={suite_hash} model_sha8={model_sha8} "
            f"prompts={[key for key, _ in selected]} n_new={args.n_new} "
            f"transformers={transformers.__version__} torch={torch.__version__}\n"
        )
    )

    print(f"  saved -> {output_path}")
    print(f"  latest -> {GOLDEN_LATEST_LINK}")
    print(f"  sentinel -> {GOLDEN_SENTINEL}")


if __name__ == "__main__":
    main()