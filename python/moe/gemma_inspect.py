"""gemma_inspect.py — dump real Gemma-4-26B-A4B architecture and run a small
calibration pass to measure two things we need before committing to an ANE
conversion strategy:

  1. Architecture truth (config + safetensors shapes).
  2. Router behavior — for each MoE layer, what fraction of the time do the
     top-8 chosen experts cluster vs. spread? Specifically, given pack size G,
     how many distinct packs do the chosen experts span on average?

Run with:  .venv313/bin/python python/moe/gemma_inspect.py
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = Path("models/gemma-4-26b-a4b")
OUT_DIR   = Path("python/moe/out"); OUT_DIR.mkdir(parents=True, exist_ok=True)


def section(t): print(f"\n{'='*8} {t} {'='*8}")


def dump_arch():
    section("CONFIG")
    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    tcfg = cfg.text_config
    fields = [
        "model_type", "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "head_dim", "num_global_key_value_heads",
        "global_head_dim", "intermediate_size", "moe_intermediate_size",
        "num_experts", "top_k_experts", "enable_moe_block",
        "hidden_activation", "attention_k_eq_v", "sliding_window",
        "max_position_embeddings", "vocab_size", "tie_word_embeddings",
        "final_logit_softcapping", "rms_norm_eps",
    ]
    for f in fields:
        v = getattr(tcfg, f, "<missing>")
        print(f"  {f:32s} = {v}")
    print(f"  layer_types histogram: {Counter(tcfg.layer_types)}")
    print(f"  rope_parameters: {json.dumps(tcfg.rope_parameters, indent=2)}")


def dump_tensor_shapes():
    section("SAFETENSORS — first MoE layer tensors")
    idx_path = MODEL_DIR / "model.safetensors.index.json"
    idx = json.loads(idx_path.read_text())
    weight_map = idx["weight_map"]
    # Find layer 0 keys
    layer0_keys = sorted(k for k in weight_map if "layers.0." in k)
    print(f"  {len(layer0_keys)} tensors in layer 0:")
    for k in layer0_keys:
        # open the right shard, peek shape + dtype
        shard = weight_map[k]
        with safe_open(MODEL_DIR / shard, framework="pt") as f:
            t = f.get_slice(k)
            print(f"    {k:80s}  {tuple(t.get_shape())}  {t.get_dtype()}")


def routing_entropy(n_calib_tokens: int = 512):
    """Run a tiny forward pass and capture router decisions per MoE layer.

    For pack size G ∈ {4, 8, 16, 32}, count distinct packs spanned by the
    top-k chosen experts at each token. Lower mean = better locality =
    fewer ANE calls per layer.
    """
    section(f"ROUTING — {n_calib_tokens} calibration tokens")
    print("  loading model in bf16 on CPU (this may take a couple of minutes)...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Hook every MoE router. The Gemma4 MoE block routes via a Linear layer;
    # we capture its output (logits → top-k indices).
    routings = {}  # layer_idx -> list of (top_k_indices) per token
    handles = []
    text_model = model.model if hasattr(model, "model") else model
    # walk submodules to find router-like linear layers in MoE blocks
    n_experts = model.config.text_config.num_experts
    top_k = model.config.text_config.top_k_experts

    def make_hook(layer_idx):
        def hook(_mod, _inp, out):
            # out: (..., n_experts) router logits
            logits = out
            if isinstance(logits, tuple):
                logits = logits[0]
            flat = logits.reshape(-1, logits.shape[-1])
            topk = torch.topk(flat, k=top_k, dim=-1).indices.cpu().numpy()
            routings.setdefault(layer_idx, []).append(topk)
        return hook

    n_hooked = 0
    for name, mod in model.named_modules():
        # Common naming: "...layers.{i}.mlp.router" or "...moe.router"
        if name.endswith(".router") and isinstance(mod, torch.nn.Linear):
            try:
                layer_idx = int(name.split(".layers.")[1].split(".")[0])
            except Exception:
                layer_idx = n_hooked
            handles.append(mod.register_forward_hook(make_hook(layer_idx)))
            n_hooked += 1
    print(f"  hooked {n_hooked} routers (n_experts={n_experts}, top_k={top_k})")
    if n_hooked == 0:
        # fall back: hook anything that outputs (..., n_experts) feature dim
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and mod.out_features == n_experts:
                try:
                    layer_idx = int(name.split(".layers.")[1].split(".")[0])
                except Exception:
                    layer_idx = n_hooked
                handles.append(mod.register_forward_hook(make_hook(layer_idx)))
                n_hooked += 1
        print(f"  fallback hooked {n_hooked} layers via out_features=={n_experts}")

    text = (
        "The Apple Neural Engine is a 16-core specialized accelerator that excels "
        "at int8 and int4 matrix multiplications. In this study we benchmark how "
        "well a sparse mixture-of-experts model maps onto it, focusing on routing "
        "locality, weight-streaming bandwidth, and end-to-end token throughput.\n\n"
        "We find that the dominant cost at long context is attention KV streaming, "
        "while expert MLPs remain cache-resident when packed in groups of eight or "
        "more INT4-quantized experts per program."
    )
    ids = tok(text, return_tensors="pt").input_ids[:, :n_calib_tokens]
    print(f"  prompt = {ids.shape[1]} tokens")
    with torch.no_grad():
        model(ids)
    for h in handles:
        h.remove()

    # Concatenate per layer
    section("ROUTING LOCALITY by pack size")
    print(f"  {'layer':>5s}  " + "  ".join(f"G={g:>2d}" for g in (4, 8, 16, 32)))
    summary = {}
    for li, chunks in sorted(routings.items()):
        topk_all = np.concatenate(chunks, axis=0)  # (n_tokens, top_k)
        row = []
        for G in (4, 8, 16, 32):
            packs_per_token = [len(set((topk_all[t] // G).tolist()))
                               for t in range(topk_all.shape[0])]
            row.append(np.mean(packs_per_token))
        summary[li] = row
        print(f"  {li:5d}  " + "  ".join(f"{v:5.2f}" for v in row))

    # Aggregate
    print("\n  MEAN over layers:")
    arr = np.array(list(summary.values()))
    print(f"  {'':5s}  " + "  ".join(f"{v:5.2f}" for v in arr.mean(axis=0)))

    np.savez(OUT_DIR / "gemma_routing.npz", layers=np.array(sorted(summary.keys())),
             distinct_packs=np.array(list(summary.values())),
             pack_sizes=np.array([4, 8, 16, 32]))
    print(f"\n  saved → {OUT_DIR/'gemma_routing.npz'}")


if __name__ == "__main__":
    dump_arch()
    dump_tensor_shapes()
    routing_entropy(n_calib_tokens=512)
