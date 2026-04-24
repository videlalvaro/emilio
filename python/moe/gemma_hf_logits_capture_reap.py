"""gemma_hf_logits_capture_reap.py — REAP-aware HF golden capture.

Runs the HF Gemma-4-26B-A4B model with each layer's router masked so that only
the layer's `keep_idx` experts can be selected (non-kept logits → -inf BEFORE
softmax). This produces the apples-to-apples reference for our REAP-pruned
64-expert deployment.

Writes:
  python/moe/out/gemma_hf_golden_logits_reap__<promptHash>__<shaShort>.npz
  python/moe/out/gemma_hf_golden_logits_reap.npz   (atomic symlink swap)
  python/moe/out/.gemma_hf_golden_reap_PASS        (sentinel)

Existing 128-expert golden (gemma_hf_golden_logits.npz) is NOT touched.

Gatekeeper-approved 2026-04-23 with 3 required additions (interpreter check,
atomic write, mask-effectiveness assertion). All present below.

Usage (must be .venv313):
  .venv313/bin/python python/moe/gemma_hf_logits_capture_reap.py
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

# --- REQUIRED #2: hard interpreter assertion ----------------------------
assert "venv313" in sys.executable, (
    f"Wrong interpreter: {sys.executable!r}. "
    "This script needs HF transformers — use .venv313/bin/python.")

import numpy as np
import torch
import transformers

assert transformers.__version__.split(".")[0] in ("4", "5"), \
    f"Unexpected transformers major: {transformers.__version__}"

PROMPT = "The capital of France is"
MODEL_DIR = Path("models/gemma-4-26b-a4b")
OUT_DIR = Path("python/moe/out")
SENTINEL = OUT_DIR / ".gemma_hf_golden_reap_PASS"
LATEST_LINK = OUT_DIR / "gemma_hf_golden_logits_reap.npz"
REAP_MASK = OUT_DIR / "gemma_reap_mask.npz"
SIDE_JSON_SUFFIX = "_router_audit.json"


def _atomic_write_npz(target: Path, **arrays):
    """Write npz to <target>.tmp, fsync, then rename. Atomic on POSIX."""
    # np.savez appends '.npz' if missing — pass a base path so the actual
    # written file matches the path we fsync/rename.
    tmp_base = target.with_suffix(target.suffix + ".tmp")  # e.g. foo.npz.tmp
    tmp_written = Path(str(tmp_base) + ".npz")             # np.savez result
    np.savez(str(tmp_base), **arrays)
    assert tmp_written.exists(), f"savez did not produce {tmp_written}"
    with open(tmp_written, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp_written, target)


def _atomic_symlink(link: Path, target_name: str):
    """Replace symlink atomically (write to .tmp link then os.replace)."""
    tmp = link.with_suffix(link.suffix + ".tmp")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(target_name)
    os.replace(tmp, link)


def main():
    if SENTINEL.exists():
        print(f"sentinel already exists: {SENTINEL}")
        print("Delete it to recapture.")
        sys.exit(0)

    # --- pre-flight ----------------------------------------------------
    if not MODEL_DIR.exists():
        print(f"FATAL: model dir missing: {MODEL_DIR}", file=sys.stderr); sys.exit(2)
    if not REAP_MASK.exists():
        print(f"FATAL: REAP mask missing: {REAP_MASK}", file=sys.stderr); sys.exit(2)

    mask_npz = np.load(REAP_MASK, allow_pickle=False)
    keep_idx = mask_npz["keep_idx"]                        # (30, 64) int64
    assert keep_idx.shape == (30, 64), \
        f"unexpected keep_idx shape {keep_idx.shape}, want (30, 64)"
    print(f"  REAP mask: keep_idx shape={keep_idx.shape}  "
          f"unique experts kept per layer = {np.unique(keep_idx[0]).size}")

    # --- recommended #5: max threads for CPU fp16 ----------------------
    nth = os.cpu_count() or 1
    torch.set_num_threads(nth)
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    print(f"  torch threads = {nth}")

    # --- recommended #6: free-RAM gate ---------------------------------
    try:
        import psutil
        free_gb = psutil.virtual_memory().available / 1e9
        print(f"  free RAM: {free_gb:.1f} GB")
        if free_gb < 8.0:
            print(f"FATAL: only {free_gb:.1f} GB RAM free, want >= 8 GB", file=sys.stderr)
            sys.exit(2)
    except ImportError:
        print("  (psutil not installed; skipping RAM check)")

    print(f"=== HF gemma-4-26b-a4b REAP-aware logit capture ===")
    print(f"  prompt: {PROMPT!r}")
    print(f"  model_dir: {MODEL_DIR}")
    print(f"  transformers={transformers.__version__}  torch={torch.__version__}")

    print("  loading tokenizer...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4TextDecoderLayer, Gemma4TextRouter,
    )
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    enc = tok(PROMPT, return_tensors="pt")
    input_ids = enc["input_ids"]
    print(f"  tokens ({input_ids.shape[1]}): {input_ids[0].tolist()}")
    print(f"  decoded: {[tok.decode([i]) for i in input_ids[0].tolist()]}")

    print("  loading model fp16 on CPU (this takes a while; 26 GB)...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    ).eval()
    print(f"  load wall: {time.perf_counter()-t0:.1f}s")

    # --- monkey-patch routers per layer --------------------------------
    # HF text_model attr name varies; resolve robustly.
    text_model = getattr(model, "model", None)
    if text_model is None:
        text_model = model
    decoder_layers = None
    for cand in ("layers", "decoder_layers"):
        if hasattr(text_model, cand):
            decoder_layers = getattr(text_model, cand); break
        if hasattr(text_model, "language_model"):
            lm = text_model.language_model
            if hasattr(lm, cand):
                decoder_layers = getattr(lm, cand); break
        if hasattr(text_model, "text_model"):
            tm = text_model.text_model
            if hasattr(tm, cand):
                decoder_layers = getattr(tm, cand); break
    assert decoder_layers is not None and len(decoder_layers) == 30, \
        f"could not resolve 30 decoder layers; got {decoder_layers!r}"
    print(f"  resolved {len(decoder_layers)} decoder layers")

    # Build per-layer keep masks (bool, length=128) on CPU.
    n_experts = 128
    layer_keep_masks: list[torch.Tensor] = []
    for li in range(30):
        m = torch.zeros(n_experts, dtype=torch.bool)
        m[torch.from_numpy(keep_idx[li])] = True
        assert int(m.sum()) == 64, f"L{li}: kept={int(m.sum())} != 64"
        layer_keep_masks.append(m)

    # --- REQUIRED #3: patch + assert effectiveness ---------------------
    audit_log = {"per_layer": [], "model_sha8": "", "prompt_sha8": ""}
    patch_count = 0

    def make_patched_forward(orig_router: Gemma4TextRouter, keep_mask: torch.Tensor,
                             layer_idx: int, audit_target: list):
        # Bind: router, mask, layer_idx, audit_target.
        # Replicates HF's Gemma4TextRouter.forward but with -inf masking
        # before softmax. See modeling_gemma4.py Gemma4TextRouter.forward.
        not_kept = ~keep_mask  # (128,)
        first_call_done = {"flag": False}

        def patched(hidden_states: torch.Tensor):
            x = orig_router.norm(hidden_states)
            x = x * orig_router.scale * orig_router.scalar_root_size
            expert_scores = orig_router.proj(x)        # [B*S, 128]

            # Mask non-kept to -inf BEFORE softmax (log-sum-exp safe).
            ns = expert_scores.dtype
            mask_value = torch.finfo(ns).min
            expert_scores = expert_scores.masked_fill(not_kept.to(expert_scores.device),
                                                       mask_value)

            # First-call assertions (REQUIRED #3).
            if not first_call_done["flag"]:
                assert (expert_scores[..., not_kept] <= mask_value + 1).all(), \
                    f"L{layer_idx}: mask not applied (non-kept not min-valued)"
                assert torch.isfinite(expert_scores[..., keep_mask]).all(), \
                    f"L{layer_idx}: kept-expert scores not finite"
                first_call_done["flag"] = True

            router_probabilities = torch.nn.functional.softmax(
                expert_scores, dim=-1)

            # Verify mask actually zeroed non-kept probabilities.
            if len(audit_target) == 0:  # only first layer to keep cheap
                kept_prob_sum = float(router_probabilities[..., keep_mask].sum(-1).mean())
                assert kept_prob_sum > 0.999, \
                    f"L{layer_idx}: kept prob sum {kept_prob_sum:.6f} < 0.999"

            top_k_weights, top_k_index = torch.topk(
                router_probabilities,
                k=orig_router.config.top_k_experts,
                dim=-1,
            )
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
            top_k_weights = top_k_weights * orig_router.per_expert_scale[top_k_index]

            # Audit: log first-token, last-position routing.
            audit_target.append({
                "layer": layer_idx,
                "kept_prob_sum_first_pos": float(
                    router_probabilities[0, keep_mask].sum().item()
                    if router_probabilities.dim() == 2
                    else router_probabilities.flatten()[keep_mask].sum().item()
                ),
                "top1_expert_first_pos": int(top_k_index.flatten()[0].item()),
                "top1_in_keep": bool(keep_mask[int(top_k_index.flatten()[0].item())].item()),
            })

            return router_probabilities, top_k_weights, top_k_index

        return patched

    for li, dec in enumerate(decoder_layers):
        if not hasattr(dec, "router"):
            print(f"FATAL: L{li} has no .router attr (enable_moe_block=False?)",
                  file=sys.stderr); sys.exit(2)
        per_layer_audit: list = []
        dec.router.forward = make_patched_forward(
            dec.router, layer_keep_masks[li], li, per_layer_audit)
        audit_log["per_layer"].append(per_layer_audit)
        patch_count += 1
    assert patch_count == 30, f"patched only {patch_count}/30 routers"
    print(f"  patched {patch_count} routers with -inf mask before softmax")

    # --- forward -------------------------------------------------------
    print("  forward (CPU fp16, T=6, no_cache)...")
    t0 = time.perf_counter()
    out = model(input_ids=input_ids, use_cache=False)
    fwd_t = time.perf_counter() - t0
    print(f"  forward wall: {fwd_t:.1f}s")

    logits = out.logits[0].float().cpu().numpy()  # (T, vocab)
    print(f"  logits shape: {logits.shape}  dtype={logits.dtype}")
    last_top1 = int(np.argmax(logits[-1]))
    print(f"  last-pos top-1: token_id={last_top1}  text={tok.decode([last_top1])!r}")

    # Verify mask was actually used: sum of audit per_layer entries.
    total_audit_entries = sum(len(p) for p in audit_log["per_layer"])
    assert total_audit_entries >= 30, \
        f"audit captured {total_audit_entries} router calls; expected >= 30"
    bad = [e for layer_entries in audit_log["per_layer"] for e in layer_entries
           if not e["top1_in_keep"]]
    assert not bad, f"top-1 expert outside keep_idx in {len(bad)} entries: {bad[:3]}"
    print(f"  audit: {total_audit_entries} router calls captured, "
          f"all top-1 experts inside keep_idx")

    # --- save (atomic) -------------------------------------------------
    prompt_hash = hashlib.sha256(PROMPT.encode()).hexdigest()[:8]
    idx_text = (MODEL_DIR / "model.safetensors.index.json").read_bytes()
    model_sha8 = hashlib.sha256(idx_text).hexdigest()[:8]
    audit_log["prompt_sha8"] = prompt_hash
    audit_log["model_sha8"] = model_sha8
    out_name = f"gemma_hf_golden_logits_reap__{prompt_hash}__{model_sha8}.npz"
    out_path = OUT_DIR / out_name

    _atomic_write_npz(
        out_path,
        logits=logits.astype(np.float32),
        input_ids=input_ids[0].numpy(),
        prompt=np.array(PROMPT),
        tokenizer=np.array(str(MODEL_DIR)),
        model_sha8=np.array(model_sha8),
        prompt_sha8=np.array(prompt_hash),
        transformers_version=np.array(transformers.__version__),
        torch_version=np.array(torch.__version__),
        keep_idx=keep_idx,
        reap_aware=np.array(True),
    )
    sz_mb = out_path.stat().st_size / 1e6
    print(f"  saved -> {out_path}  ({sz_mb:.1f} MB)")
    assert sz_mb > 1.0, f"suspiciously small npz: {sz_mb:.1f} MB"

    # Atomic symlink swap.
    if LATEST_LINK.exists() or LATEST_LINK.is_symlink():
        LATEST_LINK.unlink()
    _atomic_symlink(LATEST_LINK, out_name)
    print(f"  symlink -> {LATEST_LINK} -> {out_name}")

    # Sidecar router audit JSON.
    import json
    audit_path = out_path.with_suffix(".npz" + SIDE_JSON_SUFFIX)
    audit_path.write_text(json.dumps(audit_log, indent=2))
    print(f"  audit -> {audit_path}")

    SENTINEL.write_text(
        f"prompt_sha8={prompt_hash} model_sha8={model_sha8} "
        f"forward_s={fwd_t:.1f} last_top1={last_top1}\n")
    print(f"  sentinel -> {SENTINEL}")
    print("\n# REAP-aware HF golden: PASS")


if __name__ == "__main__":
    main()
