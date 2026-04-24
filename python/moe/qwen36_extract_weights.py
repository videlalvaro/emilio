"""qwen36_extract_weights.py — Phase 0 per-layer weight extraction scaffold.

Writes one `.npz` per requested layer plus a manifest describing the raw HF
tensor names, source safetensor files, and saved array names.

By default this extracts layer 0 only and skips huge global tensors like the
embedding table and lm_head. Use `--all-layers` and `--include-large-globals`
only after verifying the schema and disk budget.

Usage:
  .venv313/bin/python python/moe/qwen36_extract_weights.py
  .venv313/bin/python python/moe/qwen36_extract_weights.py --layer 3 --dtype float32
  .venv313/bin/python python/moe/qwen36_extract_weights.py --all-layers
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from qwen36_phase0_spec import GLOBAL_NPZ_PATH, MANIFEST_PATH, MODEL_DIR, WEIGHTS_OUT_DIR

LAYER_KEY_RE = re.compile(r"^(?P<root>.*layers\.)(?P<index>\d+)\.(?P<leaf>.+)$")


def _atomic_write_npz(target: Path, payload: dict[str, np.ndarray]) -> None:
    tmp_base = target.with_suffix(target.suffix + ".tmp")
    tmp_written = Path(str(tmp_base) + ".npz")
    np.savez(str(tmp_base), **payload)
    assert tmp_written.exists(), f"savez did not produce {tmp_written}"
    with open(tmp_written, "rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp_written, target)


def _load_json(path: Path) -> dict:
    with open(path) as handle:
        return json.load(handle)


def _detect_layer_root(weight_map: dict[str, str], config: dict) -> str:
    roots: dict[str, set[int]] = defaultdict(set)
    for key in weight_map:
        match = LAYER_KEY_RE.match(key)
        if match:
            roots[match.group("root")].add(int(match.group("index")))
    if not roots:
        raise SystemExit("could not detect a layer root from model.safetensors.index.json")

    text_config = config.get("text_config", {})
    expected_main_layers = text_config.get("num_hidden_layers")
    if expected_main_layers is not None:
        exact_matches = [
            root for root, indices in roots.items() if len(indices) == expected_main_layers
        ]
        if exact_matches:
            for preferred_root in (
                "model.language_model.layers.",
                "model.layers.",
                "language_model.layers.",
            ):
                if preferred_root in exact_matches:
                    return preferred_root
            return sorted(exact_matches)[0]

    return max(
        roots.items(),
        key=lambda item: (len(item[1]), item[0].startswith("model.language_model.layers."), item[0]),
    )[0]


def _group_layer_keys(weight_map: dict[str, str], layer_root: str) -> dict[int, list[str]]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for key in weight_map:
        match = LAYER_KEY_RE.match(key)
        if match and match.group("root") == layer_root:
            grouped[int(match.group("index"))].append(key)
    return {index: sorted(keys) for index, keys in grouped.items()}


def _sanitize_name(raw_name: str, prefix: str | None = None) -> str:
    name = raw_name[len(prefix):] if prefix and raw_name.startswith(prefix) else raw_name
    name = name.replace(".", "__")
    name = name.replace("/", "__")
    name = name.replace("-", "_")
    return name


def _is_large_global(key: str, shape: tuple[int, ...]) -> bool:
    lower = key.lower()
    if any(token in lower for token in ("embed_tokens.weight", "lm_head.weight")):
        return True
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel >= 50_000_000


def _should_include_global_key(key: str, args: argparse.Namespace) -> bool:
    if key.startswith("model.visual.") and not args.include_visual_globals:
        return False
    if key.startswith("mtp.") and not args.include_mtp_globals:
        return False
    return True


def _convert_tensor(tensor: torch.Tensor, target_dtype: str) -> np.ndarray:
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)
    if tensor.is_floating_point():
        if target_dtype == "float16":
            tensor = tensor.to(torch.float16)
        elif target_dtype == "float32":
            tensor = tensor.to(torch.float32)
        else:
            raise ValueError(f"unsupported target dtype: {target_dtype}")
    return tensor.contiguous().cpu().numpy()


def _materialize_tensors(
    model_dir: Path,
    keys: list[str],
    weight_map: dict[str, str],
    target_dtype: str,
) -> tuple[dict[str, np.ndarray], dict[str, list[int]]]:
    arrays: dict[str, np.ndarray] = {}
    shapes: dict[str, list[int]] = {}
    by_file: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        by_file[weight_map[key]].append(key)
    for filename, file_keys in sorted(by_file.items()):
        with safe_open(model_dir / filename, framework="pt") as handle:
            for raw_key in sorted(file_keys):
                tensor = handle.get_tensor(raw_key)
                shapes[raw_key] = list(tensor.shape)
                arrays[raw_key] = _convert_tensor(tensor, target_dtype)
    return arrays, shapes


def _selected_layers(args: argparse.Namespace, grouped: dict[int, list[str]]) -> list[int]:
    available = sorted(grouped)
    if args.all_layers:
        return available
    if args.layer:
        missing = [layer for layer in args.layer if layer not in grouped]
        if missing:
            raise SystemExit(f"requested layers missing from weights: {missing}")
        return sorted(set(args.layer))
    if 0 not in grouped:
        raise SystemExit("layer 0 not found; pass --layer explicitly")
    return [0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--weights-out-dir", type=Path, default=WEIGHTS_OUT_DIR)
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--global-npz-path", type=Path, default=GLOBAL_NPZ_PATH)
    parser.add_argument("--layer", type=int, action="append", default=[])
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--include-large-globals", action="store_true")
    parser.add_argument("--include-visual-globals", action="store_true")
    parser.add_argument("--include-mtp-globals", action="store_true")
    args = parser.parse_args()

    if not args.model_dir.exists():
        raise SystemExit(f"missing model dir: {args.model_dir}")

    args.weights_out_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.model_dir / "model.safetensors.index.json"
    config_path = args.model_dir / "config.json"
    if not index_path.exists():
        raise SystemExit(f"missing index file: {index_path}")
    if not config_path.exists():
        raise SystemExit(f"missing config file: {config_path}")

    print("=== Qwen3.6 Phase 0 weight extraction ===")
    print(f"  executable: {sys.executable}")
    print(f"  model_dir: {args.model_dir}")
    print(f"  target dtype: {args.dtype}")

    config = _load_json(config_path)
    weight_map = _load_json(index_path)["weight_map"]
    layer_root = _detect_layer_root(weight_map, config)
    grouped = _group_layer_keys(weight_map, layer_root)
    chosen_layers = _selected_layers(args, grouped)
    print(f"  detected layer root: {layer_root}")
    print(f"  available layers: {min(grouped)}..{max(grouped)} ({len(grouped)} total)")
    print(f"  extracting layers: {chosen_layers}")

    all_layer_keys = {key for keys in grouped.values() for key in keys}
    global_keys = sorted(set(weight_map) - all_layer_keys)

    manifest: dict[str, object] = {
        "model_dir": str(args.model_dir),
        "config_path": str(config_path),
        "index_path": str(index_path),
        "layer_root": layer_root,
        "dtype": args.dtype,
        "n_layers_detected": len(grouped),
        "requested_layers": chosen_layers,
        "config_model_type": config.get("model_type"),
        "include_visual_globals": args.include_visual_globals,
        "include_mtp_globals": args.include_mtp_globals,
        "layers": {},
        "globals": {},
        "skipped_filtered_globals": {},
        "skipped_large_globals": {},
    }

    t0 = time.perf_counter()
    for layer_index in chosen_layers:
        prefix = f"{layer_root}{layer_index}."
        raw_keys = grouped[layer_index]
        arrays, shapes = _materialize_tensors(args.model_dir, raw_keys, weight_map, args.dtype)
        payload: dict[str, np.ndarray] = {
            "meta__layer_index": np.array(layer_index, dtype=np.int32),
            "meta__layer_prefix": np.array(prefix),
            "meta__dtype": np.array(args.dtype),
        }
        name_map: dict[str, str] = {}
        for raw_key in raw_keys:
            saved_key = _sanitize_name(raw_key, prefix)
            payload[saved_key] = arrays[raw_key]
            name_map[saved_key] = raw_key
        payload["meta__tensor_name_map_json"] = np.array(json.dumps(name_map, sort_keys=True))

        out_path = args.weights_out_dir / f"qwen36_layer{layer_index:02d}.npz"
        _atomic_write_npz(out_path, payload)
        manifest["layers"][str(layer_index)] = {
            "npz_path": str(out_path),
            "n_tensors": len(raw_keys),
            "source_files": {raw_key: weight_map[raw_key] for raw_key in raw_keys},
            "shapes": shapes,
        }
        print(f"  layer {layer_index:02d}: {len(raw_keys)} tensors -> {out_path.name}")

    global_small_keys: list[str] = []
    skipped_filtered_globals: dict[str, dict[str, object]] = {}
    skipped_globals: dict[str, dict[str, object]] = {}
    if global_keys:
        by_file: dict[str, list[str]] = defaultdict(list)
        for key in global_keys:
            by_file[weight_map[key]].append(key)
        for filename, file_keys in sorted(by_file.items()):
            with safe_open(args.model_dir / filename, framework="pt") as handle:
                for raw_key in sorted(file_keys):
                    shape = tuple(int(dim) for dim in handle.get_tensor(raw_key).shape)
                    if not _should_include_global_key(raw_key, args):
                        skipped_filtered_globals[raw_key] = {
                            "shape": list(shape),
                            "source_file": filename,
                        }
                        continue
                    if _is_large_global(raw_key, shape) and not args.include_large_globals:
                        skipped_globals[raw_key] = {"shape": list(shape), "source_file": filename}
                    else:
                        global_small_keys.append(raw_key)

    if global_small_keys:
        arrays, shapes = _materialize_tensors(args.model_dir, global_small_keys, weight_map, args.dtype)
        payload = {
            "meta__dtype": np.array(args.dtype),
        }
        name_map: dict[str, str] = {}
        for raw_key in global_small_keys:
            saved_key = _sanitize_name(raw_key)
            payload[saved_key] = arrays[raw_key]
            name_map[saved_key] = raw_key
        payload["meta__tensor_name_map_json"] = np.array(json.dumps(name_map, sort_keys=True))
        _atomic_write_npz(args.global_npz_path, payload)
        manifest["globals"] = {
            "npz_path": str(args.global_npz_path),
            "n_tensors": len(global_small_keys),
            "source_files": {raw_key: weight_map[raw_key] for raw_key in global_small_keys},
            "shapes": shapes,
        }
        print(f"  globals: {len(global_small_keys)} tensors -> {args.global_npz_path.name}")

    manifest["skipped_filtered_globals"] = skipped_filtered_globals
    manifest["skipped_large_globals"] = skipped_globals
    args.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"  manifest -> {args.manifest_path}")
    if skipped_filtered_globals:
        print(f"  skipped filtered globals: {len(skipped_filtered_globals)}")
    if skipped_globals:
        print(f"  skipped large globals: {len(skipped_globals)}")
    print(f"  wall: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
