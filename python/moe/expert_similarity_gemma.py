"""Pairwise expert weight similarity for Gemma 4 MoE layers.

Gemma 4 stores experts as stacked tensors:
  model.language_model.layers.{L}.experts.gate_up_proj  -> [n_experts, 2*I, H]
  model.language_model.layers.{L}.experts.down_proj     -> [n_experts, H, I]

where the first half of `gate_up_proj` along dim 1 is the gate proj and the
second half is the up proj (fused SwiGLU). H = hidden_size, I = moe_intermediate_size.

Usage:
    python -m python.moe.expert_similarity_gemma \
        --model models/gemma-4-26b-a4b \
        --out python/moe/out/gemma_similarity
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


PROJS = ("gate_proj", "up_proj", "down_proj")


def _norm_rows_inplace(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.clip(n, 1e-12, None)
    np.divide(x, n, out=x)
    return x


def _stats(S: np.ndarray, n: int) -> dict:
    iu = np.triu_indices(n, k=1)
    offdiag = S[iu]
    evals = np.sort(np.linalg.eigvalsh(S.astype(np.float64)))[::-1]
    return {
        "off_mean": float(offdiag.mean()),
        "off_std":  float(offdiag.std()),
        "off_max":  float(offdiag.max()),
        "off_min":  float(offdiag.min()),
        "off_p95":  float(np.quantile(offdiag, 0.95)),
        "top_eig_frac": float(evals[0] / evals.sum()),
        "evals_top5":   evals[:5].astype(np.float32).tolist(),
    }


def load_layer_experts(model_dir: Path, weight_map: dict, layer: int,
                       intermediate: int) -> dict[str, np.ndarray]:
    """Returns {proj: [n_experts, flat_dim] float32}.

    Splits the fused `gate_up_proj` into separate gate and up matrices.
    """
    base = f"model.language_model.layers.{layer}"
    keys = {
        "gate_up": f"{base}.experts.gate_up_proj",
        "down":    f"{base}.experts.down_proj",
    }
    tensors: dict[str, np.ndarray] = {}
    for name, k in keys.items():
        shard = weight_map[k]
        with safe_open(model_dir / shard, framework="pt", device="cpu") as f:
            t = f.get_tensor(k).to(torch.float32).numpy()
        tensors[name] = t
    gate_up = tensors["gate_up"]   # [E, 2I, H]
    down    = tensors["down"]      # [E, H, I]
    n_experts = gate_up.shape[0]
    # Split fused gate/up
    gate = gate_up[:, :intermediate, :]
    up   = gate_up[:, intermediate:, :]
    out = {
        "gate_proj": gate.reshape(n_experts, -1).copy(),
        "up_proj":   up.reshape(n_experts, -1).copy(),
        "down_proj": down.reshape(n_experts, -1).copy(),
    }
    del tensors, gate_up, down, gate, up
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out",   required=True,
                   help="output stem; writes <out>.json (and <out>.npz if --save-matrices)")
    p.add_argument("--save-matrices", action="store_true")
    args = p.parse_args()

    model_dir = Path(args.model)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((model_dir / "config.json").read_text())
    text_cfg = cfg["text_config"]
    n_layers = text_cfg["num_hidden_layers"]
    n_routed = text_cfg["num_experts"]
    intermediate = text_cfg["moe_intermediate_size"]
    moe_layers = list(range(n_layers))   # all 30 layers have an MoE block
    weight_map = json.loads(
        (model_dir / "model.safetensors.index.json").read_text())["weight_map"]

    print(f"[cfg] {len(moe_layers)} MoE layers, {n_routed} routed experts, "
          f"moe_intermediate={intermediate}")

    summary: dict[str, list[dict]] = {p: [] for p in PROJS}
    summary["concat"] = []
    matrices: dict[str, list[np.ndarray]] = {p: [] for p in PROJS}
    matrices["concat"] = []

    for layer in moe_layers:
        ex = load_layer_experts(model_dir, weight_map, layer, intermediate)
        # Per-proj similarity
        for proj in PROJS:
            X = ex[proj]
            _norm_rows_inplace(X)
            S = (X @ X.T).astype(np.float32)
            stats = _stats(S, n_routed); stats["layer"] = layer
            summary[proj].append(stats)
            if args.save_matrices:
                matrices[proj].append(S.astype(np.float16))
        del ex; gc.collect()
        # Concatenated triple - reload (we destroyed normalization above)
        ex2 = load_layer_experts(model_dir, weight_map, layer, intermediate)
        Xcat = np.concatenate([ex2[p] for p in PROJS], axis=1)
        del ex2; gc.collect()
        _norm_rows_inplace(Xcat)
        Scat = (Xcat @ Xcat.T).astype(np.float32)
        stats = _stats(Scat, n_routed); stats["layer"] = layer
        summary["concat"].append(stats)
        if args.save_matrices:
            matrices["concat"].append(Scat.astype(np.float16))
        del Xcat, Scat; gc.collect()
        print(f"  layer {layer:2d}  "
              + "  ".join(f"{p[:4]}={summary[p][-1]['off_mean']:+.3f}"
                          for p in (*PROJS, "concat")))

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"[save] {json_path}")
    if args.save_matrices:
        npz_path = out_path.with_suffix(".npz")
        np.savez_compressed(
            npz_path,
            **{f"{p}_S": np.stack(matrices[p]) for p in (*PROJS, "concat")},
            moe_layers=np.array(moe_layers, dtype=np.int32),
        )
        print(f"[save] {npz_path}")

    print("\n=== summary (mean off-diagonal cosine across MoE layers) ===")
    for proj in (*PROJS, "concat"):
        m   = float(np.mean([d["off_mean"]    for d in summary[proj]]))
        mx  = float(np.max ([d["off_max"]     for d in summary[proj]]))
        eig = float(np.mean([d["top_eig_frac"] for d in summary[proj]]))
        print(f"  {proj:>10s}: mean={m:+.3f}  max={mx:+.3f}  "
              f"top_eig_frac={eig:.3f}  (uniform = {1.0/n_routed:.4f})")


if __name__ == "__main__":
    main()
