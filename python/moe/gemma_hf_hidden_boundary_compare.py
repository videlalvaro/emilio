"""Compare Swift hidden-boundary dumps against REAP-aware HF hidden states.

This tool loads a Swift hidden-boundary dump produced by
`gemma_ane.swift --dump-hidden-boundary-prefix ... --dump-hidden-boundary-steps ...`,
replays the same token prefix through the HF Gemma model with the REAP router
mask applied, and reports stage-wise cosine at shard boundaries and after final
norm projection.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path("python/moe/out")
MODEL_DIR = Path("models/gemma-4-26b-a4b")
REAP_MASK = OUT_DIR / "gemma_reap_mask.npz"
HEAD_NPZ = OUT_DIR / "gemma_logit_head.npz"
GOLDEN_NPZ = OUT_DIR / "gemma_golden.npz"
DEFAULT_PREFIX = Path("/tmp/gemma_swift_t415_decode_full_trace_run2")
INTERNAL_STAGE_HOOKS = {
    "post_attn": "post_attention_layernorm",
    "ffn_out": "post_feedforward_layernorm",
}
FINAL_DECODER_OUTPUT_KEY = ("__final_decoder_output__", None)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.astype(np.float64).ravel()
    b64 = b.astype(np.float64).ravel()
    return float(np.dot(a64, b64) / (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-12))


def _final_norm_projected_hidden(hidden: np.ndarray,
                                 gamma: np.ndarray,
                                 eps: float,
                                 softcap: float) -> np.ndarray:
    hidden32 = hidden.astype(np.float32).reshape(-1)
    mean_square = float(np.mean(hidden32 * hidden32))
    rms = float(np.sqrt(mean_square + eps))
    scale = rms * (softcap if softcap > 0 else 1.0)
    return hidden32 * (gamma.astype(np.float32) / scale)


def _resolve_decoder_layers(model):
    text_model = getattr(model, "model", None) or model
    decoder_layers = None
    for cand in ("layers", "decoder_layers"):
        if hasattr(text_model, cand):
            decoder_layers = getattr(text_model, cand)
            break
        if hasattr(text_model, "language_model"):
            lm = text_model.language_model
            if hasattr(lm, cand):
                decoder_layers = getattr(lm, cand)
                break
        if hasattr(text_model, "text_model"):
            tm = text_model.text_model
            if hasattr(tm, cand):
                decoder_layers = getattr(tm, cand)
                break
    assert decoder_layers is not None and len(decoder_layers) == 30
    return decoder_layers


def _parse_hidden_stage_name(stage_name: str) -> tuple[int, str | None] | None:
    if not stage_name.startswith("hidden_l"):
        return None
    body = stage_name.removeprefix("hidden_l")
    digits = []
    for char in body:
        if char.isdigit():
            digits.append(char)
        else:
            break
    if not digits:
        return None
    boundary = int("".join(digits))
    suffix = body[len(digits):]
    if suffix.startswith("_"):
        suffix = suffix[1:]
    elif suffix:
        raise SystemExit(f"Unsupported hidden stage name: {stage_name}")
    return boundary, (suffix or None)


def _register_internal_stage_hooks(decoder_layers, stage_names):
    captured = {}
    handles = []
    registered = set()
    needs_final_decoder_output = "projected_hidden" in stage_names

    def make_hook(key):
        def hook(_module, _inputs, output):
            captured[key] = output.detach().cpu()
        return hook

    for stage_name in stage_names:
        parsed = _parse_hidden_stage_name(stage_name)
        if parsed is None:
            continue
        layer_index, suffix = parsed
        if layer_index == len(decoder_layers):
            needs_final_decoder_output = True
            if suffix is not None:
                raise SystemExit(
                    f"Internal hidden stage {stage_name!r} cannot target boundary {layer_index}; "
                    "use the preceding decoder layer for internal taps")
            continue
        if suffix is None:
            continue
        if suffix not in INTERNAL_STAGE_HOOKS:
            raise SystemExit(
                f"Unsupported internal hidden stage suffix {suffix!r}; "
                f"expected one of {sorted(INTERNAL_STAGE_HOOKS)}")
        if not (0 <= layer_index < len(decoder_layers)):
            raise SystemExit(
                f"Internal hidden stage {stage_name!r} must target an existing layer")
        hook_key = (layer_index, suffix)
        if hook_key in registered:
            continue
        module = getattr(decoder_layers[layer_index], INTERNAL_STAGE_HOOKS[suffix])
        handles.append(module.register_forward_hook(make_hook(hook_key)))
        registered.add(hook_key)

    if needs_final_decoder_output and FINAL_DECODER_OUTPUT_KEY not in registered:
        handles.append(decoder_layers[-1].register_forward_hook(make_hook(FINAL_DECODER_OUTPUT_KEY)))
        registered.add(FINAL_DECODER_OUTPUT_KEY)

    return handles, captured


def _hf_stage_value(stage_name, position, hidden_states, decoder_layers,
                    internal_stage_tensors, gamma, eps, softcap):
    if stage_name == "projected_hidden":
        if FINAL_DECODER_OUTPUT_KEY not in internal_stage_tensors:
            raise SystemExit(
                "Missing captured HF final decoder output for projected_hidden; "
                "ensure the final decoder-layer hook was registered before the forward pass")
        return _final_norm_projected_hidden(
            internal_stage_tensors[FINAL_DECODER_OUTPUT_KEY][0, position].float().cpu().numpy(),
            gamma,
            eps,
            softcap,
        )

    parsed = _parse_hidden_stage_name(stage_name)
    if parsed is None:
        raise SystemExit(f"Unsupported stage name {stage_name!r}")
    boundary, suffix = parsed
    if suffix is None:
        if boundary == len(decoder_layers):
            if FINAL_DECODER_OUTPUT_KEY not in internal_stage_tensors:
                raise SystemExit(
                    f"Missing captured HF final decoder output for {stage_name!r}; "
                    "ensure the final decoder-layer hook was registered before the forward pass")
            return internal_stage_tensors[FINAL_DECODER_OUTPUT_KEY][0, position].float().cpu().numpy()
        return hidden_states[boundary][0, position].float().cpu().numpy()

    hook_key = (boundary, suffix)
    if hook_key not in internal_stage_tensors:
        raise SystemExit(
            f"Missing captured HF internal stage for {stage_name!r}; "
            "ensure hooks were registered before the forward pass")
    hook_tensor = internal_stage_tensors[hook_key]
    hook_value = hook_tensor[0, position].float().cpu().numpy()
    if suffix == "post_attn":
        base_hidden = hidden_states[boundary][0, position].float().cpu().numpy()
        return base_hidden + hook_value
    if suffix == "ffn_out":
        return hook_value
    raise SystemExit(f"Unsupported internal stage suffix {suffix!r}")


def _apply_reap_mask(decoder_layers, keep_idx, torch):
    masks = []
    for layer_index in range(30):
        mask = torch.zeros(128, dtype=torch.bool)
        mask[torch.from_numpy(keep_idx[layer_index])] = True
        masks.append(mask)

    def make_patched(orig_router, keep_mask):
        not_kept = ~keep_mask

        def patched(hidden_states):
            x = orig_router.norm(hidden_states)
            x = x * orig_router.scale * orig_router.scalar_root_size
            scores = orig_router.proj(x)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(not_kept.to(scores.device), min_value)
            probs = torch.nn.functional.softmax(scores, dim=-1)
            top_weights, top_indices = torch.topk(
                probs,
                k=orig_router.config.top_k_experts,
                dim=-1,
            )
            top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
            top_weights = top_weights * orig_router.per_expert_scale[top_indices]
            return probs, top_weights, top_indices

        return patched

    for layer_index, decoder_layer in enumerate(decoder_layers):
        decoder_layer.router.forward = make_patched(decoder_layer.router, masks[layer_index])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--swift-prefix", default=str(DEFAULT_PREFIX),
                    help="prefix used by --dump-hidden-boundary-prefix")
    ap.add_argument("--golden", default=str(GOLDEN_NPZ),
                    help="golden decode NPZ used only for same-prefix checks")
    args = ap.parse_args()

    if "venv313" not in sys.executable:
        raise SystemExit(
            f"Wrong interpreter: {sys.executable!r}. "
            "Use .venv313/bin/python for HF hidden-boundary comparison."
        )

    import torch
    from transformers import AutoModelForCausalLM

    prefix = Path(args.swift_prefix)
    meta_path = Path(str(prefix) + "_hidden_boundaries_meta.json")
    bin_path = Path(str(prefix) + "_hidden_boundaries_f32.bin")
    golden_path = Path(args.golden)
    for path in [meta_path, bin_path, REAP_MASK, HEAD_NPZ, MODEL_DIR, golden_path]:
        if not path.exists():
            raise SystemExit(f"FATAL missing artifact: {path}")

    meta = json.loads(meta_path.read_text())
    prompt_ids = [int(x) for x in meta["prompt_ids"]]
    stage_names = [str(x) for x in meta["stage_names"]]
    decode_steps = [int(x) for x in meta["decode_steps"]]
    absolute_positions = [int(x) for x in meta["absolute_positions"]]
    emitted_token_ids = [int(x) for x in meta["emitted_token_ids"]]
    generated_ids_prefixes = [[int(y) for y in x] for x in meta["generated_ids_prefixes"]]
    dim = int(meta["dim"])
    n_steps = int(meta["n_steps"])
    n_stages = len(stage_names)
    swift_hidden = np.fromfile(bin_path, dtype=np.float32).reshape(n_steps, n_stages, dim)

    head = np.load(HEAD_NPZ, allow_pickle=False)
    gamma = head["final_norm_gamma"].astype(np.float32)
    eps = float(head["rms_norm_eps"])
    softcap = float(head["softcap"])
    keep_idx = np.load(REAP_MASK, allow_pickle=False)["keep_idx"]
    gold = np.load(golden_path, allow_pickle=False)
    golden_next_ids = gold["next_token_ids"].astype(np.int64)

    longest_prefix = max(generated_ids_prefixes, key=len) if generated_ids_prefixes else []
    sequence_ids = prompt_ids + longest_prefix
    input_ids = torch.tensor([sequence_ids], dtype=torch.long)

    torch.set_num_threads(os.cpu_count() or 1)
    torch.set_grad_enabled(False)

    print("=== Gemma HF hidden-boundary compare ===")
    print(f"  prompt ids: {prompt_ids}")
    print(f"  replay ids: {sequence_ids}")
    print(f"  decode steps: {decode_steps}")
    print(f"  stage names: {stage_names}")

    print("  loading model...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    ).eval()
    print(f"  load wall: {time.perf_counter() - t0:.1f}s")

    decoder_layers = _resolve_decoder_layers(model)
    _apply_reap_mask(decoder_layers, keep_idx, torch)
    hook_handles, internal_stage_tensors = _register_internal_stage_hooks(
        decoder_layers, stage_names)

    try:
        print("  forward (with hidden states)...")
        t0 = time.perf_counter()
        out = model(input_ids=input_ids, use_cache=False, output_hidden_states=True)
        print(f"  forward wall: {time.perf_counter() - t0:.1f}s")
        hidden_states = out.hidden_states

        for step_index, decode_step in enumerate(decode_steps):
            position = absolute_positions[step_index]
            prefix_ids = generated_ids_prefixes[step_index]
            golden_prefix = golden_next_ids[:len(prefix_ids)].tolist()
            same_prefix = prefix_ids == golden_prefix
            print(f"\n=== decode step {decode_step} ===")
            print(f"  absolute_pos: {position}")
            print(f"  emitted_token_swift: {emitted_token_ids[step_index]}")
            print(f"  emitted_token_hf: {int(golden_next_ids[decode_step])}")
            print(f"  same_prefix: {same_prefix}")
            for stage_offset, stage_name in enumerate(stage_names):
                hf_stage = _hf_stage_value(
                    stage_name,
                    position,
                    hidden_states,
                    decoder_layers,
                    internal_stage_tensors,
                    gamma,
                    eps,
                    softcap,
                )
                swift_stage = swift_hidden[step_index, stage_offset]
                cosine = _cos(swift_stage, hf_stage)
                max_abs = float(np.max(np.abs(swift_stage - hf_stage)))
                swift_norm = float(np.linalg.norm(swift_stage))
                hf_norm = float(np.linalg.norm(hf_stage))
                print(
                    f"  {stage_name}: cos={cosine:.6f} "
                    f"swift_norm={swift_norm:.3f} hf_norm={hf_norm:.3f} max_abs={max_abs:.6f}"
                )
    finally:
        for handle in hook_handles:
            handle.remove()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())