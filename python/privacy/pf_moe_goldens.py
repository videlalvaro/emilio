"""T3a: capture MoE routing + intermediate tensors for layer 0.

For the same 8 sentences from pf_ref.py, captures:
  - mlp_norm_out   [B, T, 640] fp32  — input to the gate (post-RMSNorm)
  - gate_logits    [B, T, 128] fp32  — pre-softmax router output
  - topk_indices   [B, T, 4]   int32 — chosen experts (sorted desc by logit)
  - topk_weights   [B, T, 4]   fp32  — softmax(topk_logits)/k

This gives T3 (multifunction expert pack) a per-token routing trace to
replay, plus the exact normed input each expert was called with.

Note: opf MLPBlock multiplies the per-token combine by experts_per_token (4) at
the end. We store the FINAL weights (softmax/k) — the *4 happens later inside
the block. This keeps our schema closer to the on-device dispatch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = REPO_ROOT / "python" / "privacy" / "_vendor_src" / "weights"
GOLDEN_DIR = REPO_ROOT / "python" / "privacy" / "out"
REF_GOLDEN = GOLDEN_DIR / "pf_golden.npz"
MOE_GOLDEN = GOLDEN_DIR / "pf_layer0_moe.npz"

EXPERTS_PER_TOK = 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not REF_GOLDEN.exists():
        raise SystemExit(f"Missing {REF_GOLDEN}. Run pf_ref.py first.")
    if MOE_GOLDEN.exists() and not args.force:
        raise SystemExit(f"{MOE_GOLDEN} exists. Use --force.")

    z = np.load(REF_GOLDEN, allow_pickle=False)
    input_ids_np = z["input_ids"]
    attn_np = z["attention_mask"]

    torch.manual_seed(0)
    torch.set_num_threads(8)

    from opf._model.model import Transformer
    print(f"[t3a] loading model")
    model = Transformer.from_checkpoint(str(WEIGHTS_DIR), device=torch.device("cpu"))
    model.eval()

    captures: dict[str, torch.Tensor] = {}
    mlp = model.block[0].mlp

    def norm_hook(module, inputs, output):
        captures["mlp_norm_out"] = output.detach().to(torch.float32).cpu()

    h1 = mlp.norm.register_forward_hook(norm_hook)

    input_ids = torch.from_numpy(input_ids_np.astype(np.int64))
    attn = torch.from_numpy(attn_np.astype(np.int64))

    print(f"[t3a] forward B={input_ids.shape[0]} T={input_ids.shape[1]}")
    with torch.inference_mode():
        _ = model(input_ids, attention_mask=attn)
    h1.remove()

    norm_out = captures["mlp_norm_out"]   # [B, T, 640]
    B, T, D = norm_out.shape

    # opf MLPBlock calls F.linear directly on gate.weight (not self.gate(t)),
    # so a forward hook on the gate Linear doesn't fire. Recompute manually.
    gate_w = mlp.gate.weight.detach().to(torch.float32).cpu()  # [128, 640]
    gate_b = mlp.gate.bias.detach().to(torch.float32).cpu()    # [128]
    gate = torch.nn.functional.linear(norm_out, gate_w, gate_b)  # [B, T, 128]

    # Replay opf top-k selection
    topk = torch.topk(gate, k=EXPERTS_PER_TOK, dim=-1, sorted=True)
    weights = torch.nn.functional.softmax(topk.values, dim=-1) / EXPERTS_PER_TOK

    print(f"  mlp_norm_out: {tuple(norm_out.shape)} "
          f"min={norm_out.min().item():.3f} max={norm_out.max().item():.3f}")
    print(f"  gate_logits:  {tuple(gate.shape)} "
          f"min={gate.min().item():.3f} max={gate.max().item():.3f}")

    # Stats: how often each expert is selected (over valid tokens only)
    valid = torch.from_numpy(attn_np.astype(bool))    # [B, T]
    used = topk.indices[valid]                        # [N_valid, 4]
    counts = torch.bincount(used.flatten(), minlength=128)
    n_active = (counts > 0).sum().item()
    n_valid_tokens = int(valid.sum().item())
    print(f"  valid tokens: {n_valid_tokens}, expert calls: {n_valid_tokens*4}")
    print(f"  unique experts hit: {n_active} / 128")
    print(f"  top-10 most-hit experts: {sorted(counts.tolist(), reverse=True)[:10]}")

    np.savez_compressed(
        MOE_GOLDEN,
        mlp_norm_out=norm_out.numpy(),
        gate_logits=gate.numpy(),
        topk_indices=topk.indices.to(torch.int32).numpy(),
        topk_weights=weights.numpy(),
        expert_call_counts=counts.numpy().astype(np.int32),
    )
    print(f"[t3a] wrote {MOE_GOLDEN} ({MOE_GOLDEN.stat().st_size/1024**2:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
