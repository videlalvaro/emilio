"""Extract all weights + golden test data needed for the pure-Swift PF driver.

Outputs one directory `emilio/conv-ane/PF_swift/` with fp16/int32 .bin files.
Accompanied by a small JSON manifest describing shapes and dtypes.

Run with .venv313 (has opf installed):
    .venv313/bin/python python/privacy/extract_pf_swift_weights.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = REPO_ROOT / "python" / "privacy" / "_vendor_src" / "weights"
GOLDEN = REPO_ROOT / "python" / "privacy" / "out" / "pf_golden.npz"
OUT = REPO_ROOT / "emilio" / "conv-ane" / "PF_swift"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "python" / "privacy" / "_vendor_src" / "opf_src"))


def write_bin(name: str, arr: np.ndarray, manifest: dict, dtype: str = "f16"):
    """Write `arr` as raw bytes; record shape+dtype in manifest."""
    if dtype == "f16":
        a = arr.astype(np.float16, copy=False)
    elif dtype == "i32":
        a = arr.astype(np.int32, copy=False)
    elif dtype == "f32":
        a = arr.astype(np.float32, copy=False)
    else:
        raise ValueError(dtype)
    p = OUT / f"{name}.bin"
    p.write_bytes(a.tobytes(order="C"))
    manifest[name] = {"shape": list(a.shape), "dtype": dtype, "bytes": a.nbytes}
    print(f"  {name:30s} {str(a.shape):20s} {dtype}  ({a.nbytes/1024:.1f} KB)")


def main():
    from opf._model.model import Transformer
    print("[extract] loading opf Transformer (CPU)")
    model = Transformer.from_checkpoint(str(WEIGHTS_DIR), device=torch.device("cpu"))
    model.eval()

    n_layers = len(model.block)
    cfg = {
        "n_layers": n_layers,
        "d_model": int(model.embedding.weight.shape[1]),
        "vocab_size": int(model.embedding.weight.shape[0]),
        "num_labels": int(model.unembedding.weight.shape[0]),
        "n_experts": int(model.block[0].mlp.gate.weight.shape[0]),
        "topk": 4,
        "rms_eps": float(model.norm.eps),
        "T": 128,
    }
    print(f"[extract] config: {cfg}")
    manifest = {"config": cfg, "tensors": {}}
    tens = manifest["tensors"]

    # ---- Embedding (large) -------------------------------------------------
    emb = model.embedding.weight.detach().to(torch.float32).cpu().numpy()
    write_bin("embedding", emb, tens, "f16")

    # ---- Final norm + unembedding ----------------------------------------
    write_bin("final_norm_scale",
              model.norm.scale.detach().to(torch.float32).cpu().numpy(), tens, "f32")
    write_bin("unembedding",
              model.unembedding.weight.detach().to(torch.float32).cpu().numpy(),
              tens, "f16")

    # ---- Per-layer MoE side-data: pre-MoE norm scale + gate w/b ---------
    for n in range(n_layers):
        mlp = model.block[n].mlp
        write_bin(f"L{n}_mlp_norm_scale",
                  mlp.norm.scale.detach().to(torch.float32).cpu().numpy(), tens, "f32")
        write_bin(f"L{n}_mlp_gate_w",
                  mlp.gate.weight.detach().to(torch.float32).cpu().numpy(), tens, "f32")
        write_bin(f"L{n}_mlp_gate_b",
                  mlp.gate.bias.detach().to(torch.float32).cpu().numpy(), tens, "f32")

    # ---- Golden test data (input_ids, attention_mask, logits) ----------
    z = np.load(GOLDEN, allow_pickle=False)
    write_bin("input_ids",      z["input_ids"].astype(np.int32),      tens, "i32")
    write_bin("attention_mask", z["attention_mask"].astype(np.int32), tens, "i32")
    write_bin("golden_logits",  z["logits"].astype(np.float32),       tens, "f32")
    # also pad_token_id and sentence count
    manifest["pad_token_id"] = int(z["pad_token_id"])
    manifest["n_sentences"]  = int(z["input_ids"].shape[0])

    # Manifest
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[extract] wrote manifest + {len(tens)} tensors to {OUT}")


if __name__ == "__main__":
    main()
