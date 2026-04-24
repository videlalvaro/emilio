"""Per-expert utilization on Gemma 4 26B-A4B over wikitext-2 calibration text.

Hooks every Gemma4TextRouter and accumulates top-k expert selections.
Mirrors expert_utilization.py (DeepSeek) but for Gemma 4's router signature
``(router_probs, topk_weights, topk_idx)``.

Usage:
    python -m python.moe.expert_utilization_gemma \
        --model models/gemma-4-26b-a4b \
        --dtype bf16 --device cpu \
        --max-tokens 4096 --seq-len 1024 \
        --out python/moe/out/gemma_utilization.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device", default="cpu",
                   help="ignored when --offload set")
    p.add_argument("--offload", action="store_true",
                   help="use accelerate device_map=auto with disk offload")
    p.add_argument("--offload-folder", default=".offload_gemma4")
    p.add_argument("--max-cpu-mem", default="40GiB")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32}[args.dtype]

    print(f"[load] tokenizer + model dtype={args.dtype} "
          f"offload={args.offload} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.offload:
        Path(args.offload_folder).mkdir(parents=True, exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, low_cpu_mem_usage=True,
            device_map="auto",
            max_memory={"cpu": args.max_cpu_mem, "disk": "200GiB"},
            offload_folder=args.offload_folder,
            offload_state_dict=True,
        ).eval()
    else:
        device = torch.device(args.device)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, low_cpu_mem_usage=True,
        ).to(device).eval()
    print(f"[load] done in {time.time() - t0:.1f}s")

    # Identify routers and locate them by layer index
    counts: dict[int, torch.Tensor] = {}
    n_experts = model.config.text_config.num_experts
    top_k = model.config.text_config.top_k_experts
    n_layers = model.config.text_config.num_hidden_layers

    handles = []
    for layer_idx, layer in enumerate(model.model.language_model.layers):
        router = getattr(layer, "router", None)
        if router is None:
            continue
        counts[layer_idx] = torch.zeros(n_experts, dtype=torch.long)

        def make_hook(li: int):
            def hook(_module, _inputs, output):
                # output: (router_probs, topk_weights, topk_idx)
                topk_idx = output[2].detach().to("cpu").reshape(-1)
                counts[li] += torch.bincount(topk_idx, minlength=n_experts)
            return hook
        handles.append(router.register_forward_hook(make_hook(layer_idx)))
    print(f"[hook] registered on {len(counts)} layers; "
          f"experts={n_experts}, top_k={top_k}")

    print("[data] loading wikitext-2 ...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(x for x in ds["text"] if x.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    print(f"[data] {ids.shape[0]:,} tokens; using {args.max_tokens:,}")
    ids = ids[: args.max_tokens]

    seq_len = args.seq_len
    n_chunks = math.ceil(ids.shape[0] / seq_len)
    print(f"[run] {n_chunks} chunks of {seq_len} tokens ...")
    with torch.no_grad():
        for i in range(n_chunks):
            chunk = ids[i * seq_len:(i + 1) * seq_len].unsqueeze(0)
            tt = time.time()
            model(chunk)
            print(f"  chunk {i+1}/{n_chunks}  tokens={chunk.shape[1]}  "
                  f"sec={time.time()-tt:.1f}")

    for h in handles:
        h.remove()

    # Aggregate stats
    total_layers = len(counts)
    print(f"\n=== utilization summary (Gemma 4, {total_layers} MoE layers, "
          f"{n_experts} experts, top_{top_k}) ===")
    layer_stats = {}
    overall = torch.zeros(n_experts, dtype=torch.long)
    uniform_p = 1.0 / n_experts
    for li in sorted(counts):
        c = counts[li].float()
        total = c.sum().item()
        if total == 0:
            continue
        p = c / total
        H = -(p[p > 0] * p[p > 0].log()).sum().item()
        H_norm = H / math.log(n_experts)
        # Coefficient of variation as load-imbalance metric
        cv = (c.float().std() / c.float().mean()).item()
        max_load = c.max().item() / total
        min_load = c.min().item() / total
        n_dead = int((c == 0).sum().item())
        layer_stats[str(li)] = {
            "tokens_routed": int(total),
            "entropy_nats": H,
            "entropy_norm": H_norm,
            "cv": cv,
            "max_load": max_load,
            "min_load": min_load,
            "n_dead_experts": n_dead,
            "counts": c.long().tolist(),
        }
        overall += c.long()
        print(f"  L{li:2d}: H/lnN={H_norm:.4f}  CV={cv:.3f}  "
              f"max={max_load*100:5.2f}%  min={min_load*100:5.2f}%  "
              f"dead={n_dead}")

    p = overall.float() / overall.sum()
    H = -(p[p > 0] * p[p > 0].log()).sum().item()
    H_norm = H / math.log(n_experts)
    cv = (overall.float().std() / overall.float().mean()).item()
    print(f"\n  overall: H/lnN={H_norm:.4f}  CV={cv:.3f}  "
          f"uniform_p={uniform_p:.4f}")

    out = {
        "model": args.model,
        "n_experts": n_experts,
        "top_k": top_k,
        "max_tokens": args.max_tokens,
        "seq_len": args.seq_len,
        "uniform_prob": uniform_p,
        "overall_entropy_norm": H_norm,
        "overall_cv": cv,
        "layers": layer_stats,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
