"""T1: CPU reference forward + golden capture for openai/privacy-filter.

Mirrors the golden-npz contract documented in `python/moe/GEMMA_ANE_RESEARCH.md`
(keys: input_ids, attention_mask, logits, argmax_labels, seq_len, model_sha).
Adds `opf_src_sha` for traceability of the modeling code (gatekeeper request).

Pinned versions:
  - opf source:   git SHA 2e8c95b9771eec29ef61012f6e5e836f9bad7635
  - HF weights:   revision 7ffa9a043d54d1be65afb281eddf0ffbe629385b

Run with: .venv313/bin/python -m python.privacy.pf_ref
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# --- Pins (stamped into the npz; not used to drive any download dynamically) ---
HF_REPO = "openai/privacy-filter"
HF_REVISION = "7ffa9a043d54d1be65afb281eddf0ffbe629385b"
OPF_SRC_SHA = "2e8c95b9771eec29ef61012f6e5e836f9bad7635"

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_DIR = REPO_ROOT / "python" / "privacy" / "_vendor_src"
WEIGHTS_DIR = VENDOR_DIR / "weights"  # opf-native checkpoint dir lives here
OUT_DIR = REPO_ROOT / "python" / "privacy" / "out"
GOLDEN_PATH = OUT_DIR / "pf_golden.npz"

# --- Fixed sentences exercising every category at least once ---
SENTENCES = (
    "Alice Smith was born on 1990-01-02 in Springfield.",
    "Email me at alice.smith@example.com if you have questions.",
    "Call Bob at +1-415-555-0142 between 9 AM and 5 PM.",
    "His home address is 742 Evergreen Terrace, Springfield, IL.",
    "Wire the funds to account 0001234567 at First National Bank.",
    "Visit https://example.com/dashboard?token=abc for details.",
    "The API key is sk-proj-FAKE0000000000000000000000000000000 for staging.",
    "Carol's appointment is scheduled for March 15, 2026 at 10:30 AM.",
)


def _config_first_guard() -> dict:
    """Read the HF config (already on disk) and abort BEFORE any 2.8 GB pull."""
    hf_cfg_path = VENDOR_DIR / "config.json"
    if not hf_cfg_path.exists():
        raise FileNotFoundError(
            f"Expected HF config at {hf_cfg_path}. Run the vendor download first."
        )
    cfg = json.loads(hf_cfg_path.read_text())
    expected = {
        "model_type": "openai_privacy_filter",
        "num_hidden_layers": 8,
        "hidden_size": 640,
        "num_local_experts": 128,
        "num_experts_per_tok": 4,
        "vocab_size": 200064,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "head_dim": 64,
    }
    for k, want in expected.items():
        got = cfg.get(k)
        if got != want:
            raise SystemExit(
                f"Config guard failed: {k}={got!r}, expected {want!r}. "
                f"Refusing to download 2.8 GB until config matches."
            )
    return cfg


def _ensure_weights() -> Path:
    """Ensure original/model.safetensors + opf config exist locally."""
    from huggingface_hub import snapshot_download

    if (WEIGHTS_DIR / "model.safetensors").exists() and (WEIGHTS_DIR / "config.json").exists():
        return WEIGHTS_DIR

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[pf_ref] downloading opf-native checkpoint at revision {HF_REVISION}")
    snapshot_download(
        repo_id=HF_REPO,
        revision=HF_REVISION,
        allow_patterns=[
            "original/model.safetensors",
            "original/config.json",
            "original/dtypes.json",
            "original/viterbi_calibration.json",
        ],
        local_dir=str(VENDOR_DIR / "_hf_snapshot"),
    )
    src = VENDOR_DIR / "_hf_snapshot" / "original"
    for fname in ("model.safetensors", "config.json", "dtypes.json", "viterbi_calibration.json"):
        s = src / fname
        d = WEIGHTS_DIR / fname
        if s.exists() and not d.exists():
            d.symlink_to(s.resolve())
    return WEIGHTS_DIR


def _tokenize(encoding, sentences, *, T: int, pad_id: int) -> tuple[np.ndarray, np.ndarray]:
    B = len(sentences)
    ids = np.full((B, T), pad_id, dtype=np.int64)
    mask = np.zeros((B, T), dtype=np.int64)
    for i, s in enumerate(sentences):
        toks = encoding.encode(s, allowed_special="all")
        if len(toks) > T:
            raise ValueError(f"Sentence {i} exceeds T={T} ({len(toks)} tokens): {s!r}")
        ids[i, : len(toks)] = toks
        mask[i, : len(toks)] = 1
    return ids, mask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing pf_golden.npz")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if GOLDEN_PATH.exists() and not args.force:
        raise SystemExit(
            f"{GOLDEN_PATH} already exists. Re-run with --force to overwrite."
        )

    # 1. Config-first guard (cheap, no download).
    hf_cfg = _config_first_guard()
    print("[pf_ref] config guard passed.")

    # 2. Ensure weights (one-off 2.8 GB download).
    ckpt_dir = _ensure_weights()
    print(f"[pf_ref] checkpoint at {ckpt_dir}")

    # 3. Determinism knobs.
    torch.manual_seed(0)
    torch.set_num_threads(min(8, os.cpu_count() or 8))

    # 4. Build model on CPU using the vendored opf package.
    from opf._model.model import Transformer  # noqa: WPS433  (vendored, audited)

    device = torch.device("cpu")
    print("[pf_ref] loading model (CPU, native bfloat16)...")
    model = Transformer.from_checkpoint(str(ckpt_dir), device=device)
    model.eval()
    assert model.training is False
    for m in model.modules():
        assert m.training is False, f"Module {type(m).__name__} still in training mode"

    # 5. Tokenize fixed sentences.
    import tiktoken
    opf_cfg = json.loads((ckpt_dir / "config.json").read_text())
    encoding = tiktoken.get_encoding(opf_cfg["encoding"])
    pad_id = int(encoding.eot_token)
    input_ids_np, attn_np = _tokenize(encoding, SENTENCES, T=args.seq_len, pad_id=pad_id)

    input_ids = torch.from_numpy(input_ids_np).to(device)
    attn = torch.from_numpy(attn_np).to(device)

    # 6. Forward.
    print(f"[pf_ref] forward: B={input_ids.shape[0]}, T={input_ids.shape[1]}")
    with torch.inference_mode():
        logits = model(input_ids, attention_mask=attn)
    logits_fp32 = logits.detach().to(torch.float32).cpu().numpy()
    argmax = logits_fp32.argmax(axis=-1).astype(np.int32)
    print(f"[pf_ref] logits shape: {logits_fp32.shape}, dtype: {logits_fp32.dtype}")

    # 7. Save golden.
    id2label = {int(k): v for k, v in hf_cfg["id2label"].items()}
    np.savez_compressed(
        GOLDEN_PATH,
        input_ids=input_ids_np.astype(np.int32),
        attention_mask=attn_np.astype(np.int32),
        logits=logits_fp32,
        argmax_labels=argmax,
        seq_len=np.int32(args.seq_len),
        model_sha=np.str_(HF_REVISION),
        opf_src_sha=np.str_(OPF_SRC_SHA),
        id2label_json=np.str_(json.dumps(id2label)),
        sentences_json=np.str_(json.dumps(list(SENTENCES))),
        encoding=np.str_(opf_cfg["encoding"]),
        pad_token_id=np.int32(pad_id),
    )
    print(f"[pf_ref] wrote {GOLDEN_PATH} ({GOLDEN_PATH.stat().st_size/1024:.1f} KB)")

    # 8. Self-determinism check (forward again, expect cosine == 1.0).
    print("[pf_ref] determinism re-check...")
    with torch.inference_mode():
        logits2 = model(input_ids, attention_mask=attn)
    l1 = logits.detach().to(torch.float32).flatten()
    l2 = logits2.detach().to(torch.float32).flatten()
    cos = float(torch.nn.functional.cosine_similarity(l1, l2, dim=0))
    print(f"[pf_ref] self-cosine = {cos:.10f} (expect 1.0)")
    if cos < 1.0 - 1e-6:
        raise SystemExit(f"Non-deterministic forward: cos={cos}")

    # 9. Brief preview.
    for i, s in enumerate(SENTENCES):
        labels = [id2label[int(x)] for x in argmax[i, : int(attn_np[i].sum())]]
        nontriv = [l for l in labels if l != "O"]
        print(f"  [{i}] {len(nontriv)} non-O tokens | {s[:60]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
