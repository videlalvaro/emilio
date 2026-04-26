"""Expert utilization histogram.

Runs DeepSeek-V2-Lite-Chat over a calibration corpus and records the top-k
routed-expert indices at every MoE layer for every token. Output:
  - per-layer expert hit count (n_layers x n_routed_experts)
  - per-layer entropy of the empirical expert distribution
  - cumulative-mass curve: what fraction of routing decisions go to the top-N
    experts (tells you the prunable tail directly)

Usage:
    python -m python.moe.expert_utilization \
        --model models/deepseek-v2-lite-chat \
        --calib-tokens 200000 \
        --out python/moe/out/utilization.npz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from hf_full_model_safety import (
    DEFAULT_DISK_FREE_MIN_GIB,
    DEFAULT_MAX_CPU_MEMORY_GIB,
    DEFAULT_MAX_DISK_MEMORY_GIB,
    parse_gib_value,
    prepare_model_load_kwargs,
    require_disk_free,
    validate_full_model_load_policy,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--calib-tokens", type=int, default=200_000)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    p.add_argument("--split", default="train")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")  # CPU is fine; we only need router outputs
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--offload", dest="offload", action="store_true", default=True)
    p.add_argument("--no-offload", dest="offload", action="store_false")
    p.add_argument("--offload-folder", type=Path, default=Path(".offload_expert_utilization"))
    p.add_argument("--max-cpu-mem", default=f"{DEFAULT_MAX_CPU_MEMORY_GIB}GiB")
    p.add_argument("--max-disk-mem", default=f"{DEFAULT_MAX_DISK_MEMORY_GIB}GiB")
    p.add_argument("--disk-free-min-gb", type=int, default=DEFAULT_DISK_FREE_MIN_GIB)
    p.add_argument("--allow-unsafe-cpu-memory", action="store_true")
    p.add_argument("--allow-no-disk-offload", action="store_true")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    max_cpu_memory_gib = parse_gib_value(args.max_cpu_mem)
    max_disk_memory_gib = parse_gib_value(args.max_disk_mem)

    validate_full_model_load_policy(
        "expert_utilization",
        offload_enabled=args.offload,
        max_cpu_memory_gib=max_cpu_memory_gib,
        max_disk_memory_gib=max_disk_memory_gib,
        allow_unsafe_cpu_memory=args.allow_unsafe_cpu_memory,
        allow_no_disk_offload=args.allow_no_disk_offload,
    )

    disk_paths = [Path(args.model), out_path.parent]
    if args.offload:
        disk_paths.append(args.offload_folder)
    require_disk_free(disk_paths, args.disk_free_min_gb)

    print(f"[load] tokenizer + model from {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = getattr(torch, args.dtype)
    print(f"[load] memory policy offload={args.offload} cpu={max_cpu_memory_gib}GiB disk={max_disk_memory_gib}GiB")
    if args.offload:
        model_kwargs, actual_offload_folder = prepare_model_load_kwargs(
            torch_dtype=dtype,
            offload_enabled=True,
            offload_folder=args.offload_folder,
            max_cpu_memory_gib=max_cpu_memory_gib,
            max_disk_memory_gib=max_disk_memory_gib,
            local_files_only=False,
        )
        print(f"[load] offload folder: {actual_offload_folder}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            **model_kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(torch.device(args.device))
    model.eval()

    cfg = model.config
    n_routed = cfg.n_routed_experts
    n_topk = cfg.num_experts_per_tok
    first_dense = getattr(cfg, "first_k_dense_replace", 0)
    moe_layer_ids = list(range(first_dense, cfg.num_hidden_layers))
    print(f"[cfg] {len(moe_layer_ids)} MoE layers, {n_routed} routed, top-{n_topk}")

    # Hit-count tensor: rows = MoE layer index, cols = expert id.
    hits = np.zeros((len(moe_layer_ids), n_routed), dtype=np.int64)
    tokens_routed = 0  # one "routing decision" = one (token, layer) pair

    # Hook the MoE gate of every MoE layer.
    # In modeling_deepseek, MoEGate.forward returns (topk_idx, topk_weight, aux_loss).
    layer_idx_lookup = {id(model.model.layers[i].mlp.gate): k
                        for k, i in enumerate(moe_layer_ids)
                        if hasattr(model.model.layers[i].mlp, "gate")}

    def make_hook(slot: int):
        def hook(_module, _inp, out):
            # topk_idx shape: (B*T, top_k)
            if isinstance(out, tuple):
                topk_idx = out[0]
            else:
                topk_idx = out
            idx = topk_idx.detach().to("cpu").numpy().reshape(-1)
            np.add.at(hits[slot], idx, 1)
        return hook

    handles = []
    for i in moe_layer_ids:
        gate = getattr(model.model.layers[i].mlp, "gate", None)
        if gate is None:
            continue
        slot = moe_layer_ids.index(i)
        handles.append(gate.register_forward_hook(make_hook(slot)))

    # Stream a calibration corpus.
    print(f"[calib] streaming {args.dataset}/{args.dataset_config} :: {args.split}")
    ds = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=True)

    buf: list[int] = []
    t0 = time.time()
    with torch.inference_mode():
        for row in ds:
            text = row.get("text") or ""
            if not text.strip():
                continue
            buf.extend(tok(text, add_special_tokens=False)["input_ids"])
            while len(buf) >= args.seq_len and tokens_routed < args.calib_tokens:
                chunk = buf[: args.seq_len]
                buf = buf[args.seq_len:]
                ids = torch.tensor([chunk], dtype=torch.long, device=args.device)
                model(ids, use_cache=False)
                tokens_routed += args.seq_len
                if tokens_routed % (args.seq_len * 4) == 0:
                    rate = tokens_routed / max(time.time() - t0, 1e-6)
                    print(f"  [{tokens_routed}/{args.calib_tokens}] {rate:.0f} tok/s")
            if tokens_routed >= args.calib_tokens:
                break

    for h in handles:
        h.remove()

    # Each (token, layer) routing decision contributes top_k expert hits.
    # Total hits per layer should equal tokens_routed * top_k.
    expected = tokens_routed * n_topk
    actual = hits.sum(axis=1)
    if not np.allclose(actual, expected):
        print(f"[warn] hits/layer mismatch: expected {expected}, got {actual.tolist()[:3]}...")

    # Per-layer entropy and cumulative-mass curves.
    probs = hits / hits.sum(axis=1, keepdims=True).clip(min=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        layer_entropy = -np.nansum(probs * np.log2(probs + 1e-12), axis=1)
    uniform_entropy = np.log2(n_routed)

    # Cumulative mass when experts sorted by usage (per layer).
    sorted_probs = -np.sort(-probs, axis=1)
    cum_mass = np.cumsum(sorted_probs, axis=1)

    np.savez_compressed(
        out_path,
        hits=hits,
        layer_entropy=layer_entropy,
        uniform_entropy=uniform_entropy,
        cum_mass=cum_mass,
        moe_layer_ids=np.array(moe_layer_ids, dtype=np.int32),
        tokens_routed=tokens_routed,
        n_topk=n_topk,
    )
    print(f"[save] {out_path}")

    # Quick text summary.
    print("\n=== summary ===")
    print(f"tokens routed: {tokens_routed}, top-k: {n_topk}, experts: {n_routed}")
    print(f"uniform-routing entropy: {uniform_entropy:.3f} bits")
    print(f"mean per-layer entropy:  {layer_entropy.mean():.3f} bits "
          f"({100 * layer_entropy.mean() / uniform_entropy:.1f}% of uniform)")
    # For each "fraction kept" threshold, how many experts on average?
    for thresh in (0.50, 0.80, 0.90, 0.95, 0.99):
        # number of experts needed per layer to cover `thresh` of mass
        ks = (cum_mass >= thresh).argmax(axis=1) + 1
        print(f"  experts to cover {thresh:.0%} of routing: "
              f"mean={ks.mean():.1f}  median={int(np.median(ks))}  max={int(ks.max())}")


if __name__ == "__main__":
    main()
