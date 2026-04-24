"""Smoke test: load layer 5 (global) into GemmaGlobalLayer, run a forward
pass, check shapes and finiteness. Cheap (no CoreML, no quant).

Run with the torch-only env:
  /Users/alvarovidela/Code/em2/.venv/bin/python python/moe/_smoke_global.py
"""
from pathlib import Path
import numpy as np
import torch

from gemma_to_ane import (  # noqa: E402
    D_MODEL, GLB_N_KV, GLB_D_HEAD, GLB_ROT_DIM,
    GemmaGlobalLayer, _load_layer_weights,
)

OUT_DIR = Path("python/moe/out")
MAX_CTX = 256


def main():
    print("=== smoke: GemmaGlobalLayer (layer 5) ===")
    layer = GemmaGlobalLayer(MAX_CTX).half().eval()
    npz = OUT_DIR / "gemma_layer5_packed.npz"
    print(f"  load {npz.name}")
    _load_layer_weights(layer, npz)

    n_params = sum(p.numel() for p in layer.parameters())
    print(f"  parameters: {n_params:,} (fp16) "
          f"= {n_params * 2 / 1e6:.1f} MB pre-quant")

    x   = torch.randn(1, 1, D_MODEL,    dtype=torch.float16) * 0.1
    cos = torch.randn(1, 1, GLB_ROT_DIM, dtype=torch.float16)
    sin = torch.randn(1, 1, GLB_ROT_DIM, dtype=torch.float16)
    kc  = torch.zeros(1, GLB_N_KV, MAX_CTX, GLB_D_HEAD, dtype=torch.float16)
    vc  = torch.zeros(1, GLB_N_KV, MAX_CTX, GLB_D_HEAD, dtype=torch.float16)
    am  = torch.full((1, 1, 1, MAX_CTX), -1e4, dtype=torch.float16)
    am[..., 0] = 0.0
    wm  = torch.zeros(1, 1, MAX_CTX, 1, dtype=torch.float16)
    wm[0, 0, 0, 0] = 1.0

    print("  forward...")
    with torch.no_grad():
        out, k_new, v_new = layer(x, cos, sin, kc, vc, am, wm)

    print(f"  out      : {tuple(out.shape)} {out.dtype} "
          f"finite={torch.isfinite(out).all().item()}")
    print(f"  k_new    : {tuple(k_new.shape)} (expect 1,{GLB_N_KV},{MAX_CTX},{GLB_D_HEAD})")
    print(f"  v_new    : {tuple(v_new.shape)} (expect same)")
    print(f"  k_new[0,0,0,:6] = {k_new[0,0,0,:6].tolist()}")
    print(f"  v_new[0,0,0,:6] = {v_new[0,0,0,:6].tolist()}")
    print(f"  out norm = {out.float().norm().item():.4f}")
    assert torch.isfinite(out).all(), "non-finite output"
    assert tuple(out.shape) == (1, 1, D_MODEL)
    assert tuple(k_new.shape) == (1, GLB_N_KV, MAX_CTX, GLB_D_HEAD)
    print("\n# SMOKE GLOBAL: PASS")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.resolve()))
    main()
