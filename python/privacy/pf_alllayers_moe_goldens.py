"""T4 multi-layer MoE goldens for privacy-filter (all 8 layers).

Mirrors pf_moe_goldens.py but captures every layer's MoE input + routing.
Outputs `python/privacy/out/pf_alllayers_moe.npz` with keys per layer:
  L{n}_mlp_norm_out   [B, T, 640]   fp32
  L{n}_topk_indices   [B, T, 4]     int32
  L{n}_topk_weights   [B, T, 4]     fp32
  L{n}_mlp_output     [B, T, 640]   fp32  -- residual delta from MoE block
  L{n}_block_input    [B, T, 640]   fp32  -- pre-residual input to layer
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = REPO_ROOT / "python" / "privacy" / "_vendor_src" / "weights"
GOLDEN_DIR = REPO_ROOT / "python" / "privacy" / "out"
REF_GOLDEN = GOLDEN_DIR / "pf_golden.npz"
ALL_GOLDEN = GOLDEN_DIR / "pf_alllayers_moe.npz"
EXPERTS_PER_TOK = 4
N_LAYERS = 8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not REF_GOLDEN.exists():
        raise SystemExit(f"Missing {REF_GOLDEN}. Run pf_ref.py first.")
    if ALL_GOLDEN.exists() and not args.force:
        raise SystemExit(f"{ALL_GOLDEN} exists. Use --force.")

    z = np.load(REF_GOLDEN, allow_pickle=False)
    input_ids = torch.from_numpy(z["input_ids"].astype(np.int64))
    attn = torch.from_numpy(z["attention_mask"].astype(np.int64))

    torch.manual_seed(0)
    torch.set_num_threads(8)
    from opf._model.model import Transformer
    print("[t4-goldens] loading model")
    model = Transformer.from_checkpoint(str(WEIGHTS_DIR), device=torch.device("cpu"))
    model.eval()

    captures: dict[int, dict[str, torch.Tensor]] = {n: {} for n in range(N_LAYERS)}
    handles = []
    for n in range(N_LAYERS):
        mlp = model.block[n].mlp

        def make_norm_hook(layer_idx):
            def h(module, inputs, output):
                captures[layer_idx]["norm_out"] = output.detach().to(torch.float32).cpu()
            return h

        def make_block_in_hook(layer_idx):
            def h(module, inputs, output):
                # inputs[0] is x going into MLPBlock.forward
                captures[layer_idx]["block_in"] = inputs[0].detach().to(torch.float32).cpu()
            return h

        def make_block_out_hook(layer_idx):
            def h(module, inputs, output):
                # MLPBlock returns the residual delta (not added back -- caller adds)
                # Actually opf MLPBlock returns x + delta; we want delta.
                # Compute delta = output - inputs[0]
                delta = (output - inputs[0]).detach().to(torch.float32).cpu()
                captures[layer_idx]["mlp_delta"] = delta
            return h

        handles.append(mlp.norm.register_forward_hook(make_norm_hook(n)))
        handles.append(mlp.register_forward_pre_hook(
            lambda m, ins, li=n: captures[li].__setitem__(
                "block_in", ins[0].detach().to(torch.float32).cpu())))
        handles.append(mlp.register_forward_hook(make_block_out_hook(n)))

    print(f"[t4-goldens] forward B={input_ids.shape[0]} T={input_ids.shape[1]}")
    with torch.inference_mode():
        _ = model(input_ids, attention_mask=attn)
    for h in handles:
        h.remove()

    out: dict[str, np.ndarray] = {}
    for n in range(N_LAYERS):
        norm_out = captures[n]["norm_out"]                       # [B, T, 640]
        block_in = captures[n]["block_in"]
        mlp_delta = captures[n]["mlp_delta"]
        # Recompute gate logits manually (opf uses F.linear directly, no nn-module hook)
        gate_w = model.block[n].mlp.gate.weight.detach().to(torch.float32).cpu()
        gate_b = model.block[n].mlp.gate.bias.detach().to(torch.float32).cpu()
        gate = torch.nn.functional.linear(norm_out, gate_w, gate_b)
        topk = torch.topk(gate, k=EXPERTS_PER_TOK, dim=-1, sorted=True)
        weights = torch.nn.functional.softmax(topk.values, dim=-1) / EXPERTS_PER_TOK

        out[f"L{n}_mlp_norm_out"] = norm_out.numpy()
        out[f"L{n}_topk_indices"] = topk.indices.to(torch.int32).numpy()
        out[f"L{n}_topk_weights"] = weights.numpy()
        out[f"L{n}_mlp_delta"] = mlp_delta.numpy()
        out[f"L{n}_block_in"] = block_in.numpy()
        print(f"  L{n}: norm_out shape {tuple(norm_out.shape)}  "
              f"|delta|max={mlp_delta.abs().max().item():.3f}")

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ALL_GOLDEN, **out)
    print(f"[t4-goldens] wrote {ALL_GOLDEN} ({ALL_GOLDEN.stat().st_size/1024**2:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
