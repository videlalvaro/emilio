"""Compare Swift-dumped Gemma prompt logits against the saved HF golden.

This is the Swift-side companion to gemma_t414_logit_gate.py. The Swift runtime
can dump prompt-position logits with:

  /tmp/gemma_ane_smoke \
      --meta python/moe/out/gemma_swift_head_meta.json \
      --prompt-ids 2,818,5279,529,7001,563 \
      --n-new 0 \
      --dump-prompt-logits-prefix python/moe/out/gemma_swift_t414

That produces:
  python/moe/out/gemma_swift_t414_meta.json
  python/moe/out/gemma_swift_t414_logits_f32.bin

This script loads those files, checks prompt-ID alignment against the REAP-aware
HF golden, then reports per-position cosine and top-1 agreement.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

OUT_DIR = Path("python/moe/out")
DEFAULT_PREFIX = OUT_DIR / "gemma_swift_t414"
HF_GOLDEN = OUT_DIR / "gemma_hf_golden_logits_reap.npz"
SENTINEL = OUT_DIR / ".gemma_swift_t414_logit_gate_PASS"

PASS_COS = 0.96
MIN_TOP1_AGREE = 5


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.astype(np.float64).ravel()
    b64 = b.astype(np.float64).ravel()
    return float(np.dot(a64, b64) / (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=str(DEFAULT_PREFIX),
                    help="prefix used by --dump-prompt-logits-prefix")
    ap.add_argument("--golden", default=str(HF_GOLDEN),
                    help="HF golden logits NPZ")
    args = ap.parse_args()

    prefix = Path(args.prefix)
    meta_path = Path(str(prefix) + "_meta.json")
    logits_path = Path(str(prefix) + "_logits_f32.bin")
    golden_path = Path(args.golden)

    print("=== Gemma Swift logit gate ===")
    for path in [meta_path, logits_path, golden_path]:
        if not path.exists():
            print(f"FATAL missing artifact: {path}", file=sys.stderr)
            return 2

    meta = json.loads(meta_path.read_text())
    prompt_ids = [int(x) for x in meta["prompt_ids"]]
    rows = int(meta["rows"])
    cols = int(meta["cols"])
    dtype = str(meta["dtype"])
    if dtype != "float32":
        print(f"FATAL unsupported dtype in dump meta: {dtype}", file=sys.stderr)
        return 2

    swift_logits = np.fromfile(logits_path, dtype=np.float32)
    expected_count = rows * cols
    if swift_logits.size != expected_count:
        print(f"FATAL logits size mismatch: {swift_logits.size} vs {expected_count}",
              file=sys.stderr)
        return 2
    swift_logits = swift_logits.reshape(rows, cols)

    gold = np.load(golden_path, allow_pickle=False)
    if "logits" in gold and "input_ids" in gold:
        hf_logits = gold["logits"].astype(np.float32)
        hf_input_ids = gold["input_ids"].astype(np.int64)
        golden_schema = "input_ids/logits"
    elif "logits_full" in gold and "prompt_ids" in gold:
        hf_logits = gold["logits_full"].astype(np.float32)
        hf_input_ids = gold["prompt_ids"].astype(np.int64)
        golden_schema = "prompt_ids/logits_full"
    else:
        print(
            "FATAL unsupported golden schema: expected "
            "input_ids/logits or prompt_ids/logits_full",
            file=sys.stderr,
        )
        return 2
    if hf_logits.shape != swift_logits.shape:
        print(f"FATAL shape mismatch: swift={swift_logits.shape} hf={hf_logits.shape}",
              file=sys.stderr)
        return 2
    if prompt_ids != hf_input_ids.tolist():
        print(f"FATAL prompt ids mismatch: swift={prompt_ids} hf={hf_input_ids.tolist()}",
              file=sys.stderr)
        return 2

    print(f"  prompt ids: {prompt_ids}")
    print(f"  golden schema: {golden_schema}")
    if "prompt_text" in meta:
        print(f"  prompt text: {meta['prompt_text']!r}")
    print(f"  logits shape: {swift_logits.shape}")

    cos_per_pos: list[float] = []
    swift_top1: list[int] = []
    hf_top1: list[int] = []

    for pos in range(rows):
        cosine = _cos(swift_logits[pos], hf_logits[pos])
        swift_argmax = int(np.argmax(swift_logits[pos]))
        hf_argmax = int(np.argmax(hf_logits[pos]))
        cos_per_pos.append(cosine)
        swift_top1.append(swift_argmax)
        hf_top1.append(hf_argmax)
        print(f"  pos={pos}  cos={cosine:.4f}  swift_top1={swift_argmax}  hf_top1={hf_argmax}")

    min_cos = min(cos_per_pos)
    n_top1_agree = sum(int(a == b) for a, b in zip(swift_top1, hf_top1))
    print(f"\n  min cos     = {min_cos:.4f}  (floor {PASS_COS})")
    print(f"  top-1 agree = {n_top1_agree}/{rows}  (need >= {MIN_TOP1_AGREE})")

    if min_cos >= PASS_COS and n_top1_agree >= MIN_TOP1_AGREE:
        # Only write/delete sentinel when using defaults (avoids cross-contamination).
        using_defaults = (prefix == DEFAULT_PREFIX and golden_path == HF_GOLDEN)
        if using_defaults:
            SENTINEL.write_text(
                f"min_cos={min_cos:.6f} cos_per_pos={cos_per_pos} top1_agree={n_top1_agree}/{rows}\n"
            )
            print(f"\n# GEMMA SWIFT LOGIT GATE: PASS")
            print(f"  sentinel: {SENTINEL}")
        else:
            print(f"\n# GEMMA SWIFT LOGIT GATE: PASS  (non-default golden; no sentinel written)")
        return 0

    # Only delete sentinel on FAIL when using defaults.
    using_defaults = (prefix == DEFAULT_PREFIX and golden_path == HF_GOLDEN)
    if using_defaults and SENTINEL.exists():
        SENTINEL.unlink()
    print("\n# GEMMA SWIFT LOGIT GATE: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())