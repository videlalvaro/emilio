"""T5b: capture attn-output goldens for all 8 layers (residual stream after each
AttentionBlock). Plus block-input (pre-attn residual) for each layer so we can
feed PF_attn{N}_T128 with the right input.

Output: python/privacy/out/pf_attn_alllayers.npz
Keys per layer N in 0..7:
  L{N}_attn_in   [B, T, 640]  fp32 -- input to block N (pre-attn residual)
  L{N}_attn_out  [B, T, 640]  fp32 -- output of block N's attention (post-residual-add)
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = REPO_ROOT / "python" / "privacy" / "_vendor_src" / "weights"
GOLDEN_DIR = REPO_ROOT / "python" / "privacy" / "out"
REF_GOLDEN = GOLDEN_DIR / "pf_golden.npz"
OUT_GOLDEN = GOLDEN_DIR / "pf_attn_alllayers.npz"
N_LAYERS = 8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not REF_GOLDEN.exists():
        raise SystemExit(f"missing {REF_GOLDEN}")
    if OUT_GOLDEN.exists() and not args.force:
        raise SystemExit(f"{OUT_GOLDEN} exists. Use --force.")

    z = np.load(REF_GOLDEN, allow_pickle=False)
    input_ids = torch.from_numpy(z["input_ids"].astype(np.int64))
    attn = torch.from_numpy(z["attention_mask"].astype(np.int64))

    torch.manual_seed(0)
    torch.set_num_threads(min(8, os.cpu_count() or 8))
    from opf._model.model import Transformer
    print(f"[t5b-goldens] loading model from {WEIGHTS_DIR}")
    model = Transformer.from_checkpoint(str(WEIGHTS_DIR), device=torch.device("cpu"))
    model.eval()

    cap: dict[int, dict[str, torch.Tensor]] = {n: {} for n in range(N_LAYERS)}
    handles = []
    for n in range(N_LAYERS):
        a = model.block[n].attn
        # block input = pre_hook input[0]; attn output = forward_hook output
        handles.append(a.register_forward_pre_hook(
            lambda m, ins, li=n: cap[li].__setitem__(
                "attn_in", ins[0].detach().to(torch.float32).cpu())))
        handles.append(a.register_forward_hook(
            lambda m, ins, out, li=n: cap[li].__setitem__(
                "attn_out", out.detach().to(torch.float32).cpu())))

    print(f"[t5b-goldens] forward B={input_ids.shape[0]} T={input_ids.shape[1]}")
    with torch.inference_mode():
        _ = model(input_ids, attention_mask=attn)
    for h in handles:
        h.remove()

    out: dict[str, np.ndarray] = {}
    for n in range(N_LAYERS):
        ai = cap[n]["attn_in"]; ao = cap[n]["attn_out"]
        out[f"L{n}_attn_in"] = ai.numpy()
        out[f"L{n}_attn_out"] = ao.numpy()
        delta = (ao - ai).abs().max().item()
        print(f"  L{n}: in={tuple(ai.shape)}  out={tuple(ao.shape)}  |Δresid|max={delta:.3f}")

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_GOLDEN, **out)
    print(f"[t5b-goldens] wrote {OUT_GOLDEN} ({OUT_GOLDEN.stat().st_size/1024**2:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
