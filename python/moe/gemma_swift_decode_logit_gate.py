"""Compare Swift-dumped Gemma decode logits against gemma_golden.npz.

The Swift runtime can dump one full-vocab post-token decode row per greedy step
with:

  /tmp/gemma_ane_smoke \
      --meta python/moe/out/gemma_swift_head_meta.json \
      --prompt-ids 2,3689,563,506,5279,529,7001,236881 \
      --n-new 2 \
      --dump-decode-logits-prefix /tmp/gemma_swift_t415_decode_smoke

That produces:
  /tmp/gemma_swift_t415_decode_smoke_meta.json
  /tmp/gemma_swift_t415_decode_smoke_logits_f32.bin

This script checks the dump schema, prompt-ID alignment, generated-ID prefix
alignment, then reports per-step cosine and top-1 agreement against the decode
rows saved in python/moe/out/gemma_golden.npz.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

OUT_DIR = Path("python/moe/out")
DEFAULT_PREFIX = Path("/tmp/gemma_swift_t415_decode_smoke")
GOLDEN_NPZ = OUT_DIR / "gemma_golden.npz"

# This is a short decode-smoke floor, not the full T4.3 ship gate.
PASS_COS = 0.96


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.astype(np.float64).ravel()
    b64 = b.astype(np.float64).ravel()
    return float(np.dot(a64, b64) / (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-12))


def _topk(row: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
    idx = np.argpartition(row, -k)[-k:]
    idx = idx[np.argsort(row[idx])[::-1]]
    return [(int(token_id), float(row[token_id])) for token_id in idx]


def _sentinel_for_prefix(prefix: Path) -> Path:
    safe_name = prefix.name.replace(".", "_")
    return OUT_DIR / f".gemma_swift_decode_logit_gate_{safe_name}_PASS"


def _fail(sentinel: Path, message: str, code: int = 2) -> int:
    if sentinel.exists():
        sentinel.unlink()
    print(message, file=sys.stderr)
    return code


def _print_row_diagnostic(
    step: int,
    swift_row: np.ndarray,
    hf_row: np.ndarray,
    swift_generated_id: int,
    hf_generated_id: int,
) -> None:
    swift_top5 = _topk(swift_row)
    hf_top5 = _topk(hf_row)
    swift_top1 = swift_top5[0][0]
    hf_top1 = hf_top5[0][0]
    swift_margin = swift_top5[0][1] - swift_top5[1][1]
    hf_margin = hf_top5[0][1] - hf_top5[1][1]
    hf_rank_of_swift_top1 = int(np.sum(hf_row > hf_row[swift_top1])) + 1
    swift_rank_of_hf_top1 = int(np.sum(swift_row > swift_row[hf_top1])) + 1

    print(f"  diag step={step}  cos={_cos(swift_row, hf_row):.4f}")
    print(
        f"    generated ids: swift={swift_generated_id} hf={hf_generated_id}"
    )
    print(f"    swift_top5={swift_top5}")
    print(f"    hf_top5={hf_top5}")
    print(f"    swift_margin={swift_margin:.6f}  hf_margin={hf_margin:.6f}")
    print(
        "    ranks: "
        f"hf_rank_of_swift_top1={hf_rank_of_swift_top1}  "
        f"swift_rank_of_hf_top1={swift_rank_of_hf_top1}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=str(DEFAULT_PREFIX),
                    help="prefix used by --dump-decode-logits-prefix")
    ap.add_argument("--golden", default=str(GOLDEN_NPZ),
                    help="golden decode NPZ")
    args = ap.parse_args()

    prefix = Path(args.prefix)
    meta_path = Path(str(prefix) + "_meta.json")
    logits_path = Path(str(prefix) + "_logits_f32.bin")
    golden_path = Path(args.golden)
    sentinel = _sentinel_for_prefix(prefix)

    print("=== Gemma Swift decode-logit gate ===")
    for path in [meta_path, logits_path, golden_path]:
        if not path.exists():
            return _fail(sentinel, f"FATAL missing artifact: {path}")

    meta = json.loads(meta_path.read_text())
    kind = str(meta.get("kind", ""))
    prompt_ids = [int(x) for x in meta["prompt_ids"]]
    generated_ids = [int(x) for x in meta.get("generated_ids", [])]
    rows = int(meta["rows"])
    cols = int(meta["cols"])
    dtype = str(meta["dtype"])
    row_semantics = str(meta.get("logit_row_semantics", ""))

    if kind != "decode":
        return _fail(sentinel, f"FATAL expected kind=decode, got {kind!r}")
    if dtype != "float32":
        return _fail(sentinel, f"FATAL unsupported dtype in dump meta: {dtype}")
    if row_semantics != "post_token":
        return _fail(sentinel,
                     f"FATAL expected logit_row_semantics=post_token, got {row_semantics!r}")
    if rows != len(generated_ids):
        return _fail(sentinel,
                     f"FATAL rows/generated mismatch: rows={rows} generated={len(generated_ids)}")

    swift_logits = np.fromfile(logits_path, dtype=np.float32)
    expected_count = rows * cols
    if swift_logits.size != expected_count:
        return _fail(sentinel,
                     f"FATAL logits size mismatch: {swift_logits.size} vs {expected_count}")
    swift_logits = swift_logits.reshape(rows, cols)

    gold = np.load(golden_path, allow_pickle=False)
    golden_prompt_ids = gold["prompt_ids"].astype(np.int64)
    golden_next_ids = gold["next_token_ids"].astype(np.int64)
    golden_next_logits = gold["next_token_logits"].astype(np.float32)

    if prompt_ids != golden_prompt_ids.tolist():
        return _fail(
            sentinel,
            f"FATAL prompt ids mismatch: swift={prompt_ids} hf={golden_prompt_ids.tolist()}"
        )
    if rows > golden_next_logits.shape[0]:
        return _fail(
            sentinel,
            f"FATAL decode rows exceed golden: rows={rows} golden={golden_next_logits.shape[0]}"
        )
    if cols != golden_next_logits.shape[1]:
        return _fail(
            sentinel,
            f"FATAL vocab mismatch: swift={cols} golden={golden_next_logits.shape[1]}"
        )

    golden_ids_prefix = golden_next_ids[:rows].tolist()
    generated_ids_mismatch: str | None = None
    first_generated_mismatch: int | None = None
    if generated_ids != golden_ids_prefix:
        generated_ids_mismatch = (
            f"FATAL generated ids mismatch: swift={generated_ids} hf={golden_ids_prefix}"
        )
        first_generated_mismatch = next(
            index
            for index, (swift_id, hf_id) in enumerate(zip(generated_ids, golden_ids_prefix))
            if swift_id != hf_id
        )

    hf_logits = golden_next_logits[:rows]

    print(f"  prompt ids: {prompt_ids}")
    if "prompt_text" in meta:
        print(f"  prompt text: {meta['prompt_text']!r}")
    print(f"  generated ids: {generated_ids}")
    print(f"  logits shape: {swift_logits.shape}")

    cos_per_step: list[float] = []
    swift_top1: list[int] = []
    hf_top1: list[int] = []

    for step in range(rows):
        cosine = _cos(swift_logits[step], hf_logits[step])
        swift_argmax = int(np.argmax(swift_logits[step]))
        hf_argmax = int(np.argmax(hf_logits[step]))
        cos_per_step.append(cosine)
        swift_top1.append(swift_argmax)
        hf_top1.append(hf_argmax)
        print(f"  step={step}  cos={cosine:.4f}  swift_top1={swift_argmax}  hf_top1={hf_argmax}")

    min_cos = min(cos_per_step) if cos_per_step else 1.0
    n_top1_agree = sum(int(a == b) for a, b in zip(swift_top1, hf_top1))
    print(f"\n  min cos     = {min_cos:.4f}  (floor {PASS_COS})")
    print(f"  top-1 agree = {n_top1_agree}/{rows}")

    diagnostic_steps: list[int] = []
    if first_generated_mismatch is not None:
        print(f"  first generated-id mismatch at index {first_generated_mismatch}")
        if row_semantics == "post_token" and first_generated_mismatch > 0:
            print(
                "  post_token note: the likely causal logits row is the previous step; "
                "the mismatch index itself is already downstream of the wrong token."
            )
            diagnostic_steps.extend(
                [first_generated_mismatch - 1, first_generated_mismatch]
            )
        else:
            diagnostic_steps.append(first_generated_mismatch)

    first_top1_mismatch = next(
        (index for index, (swift_id, hf_id) in enumerate(zip(swift_top1, hf_top1)) if swift_id != hf_id),
        None,
    )
    if first_top1_mismatch is not None:
        diagnostic_steps.append(first_top1_mismatch)

    first_cos_below_floor = next(
        (index for index, cosine in enumerate(cos_per_step) if cosine < PASS_COS),
        None,
    )
    if first_cos_below_floor is not None:
        diagnostic_steps.append(first_cos_below_floor)

    diagnostic_steps = sorted({step for step in diagnostic_steps if 0 <= step < rows})
    if diagnostic_steps:
        print("\n  diagnostics:")
        for step in diagnostic_steps:
            _print_row_diagnostic(
                step,
                swift_logits[step],
                hf_logits[step],
                generated_ids[step],
                golden_ids_prefix[step],
            )

    if generated_ids_mismatch is None and min_cos >= PASS_COS and n_top1_agree == rows:
        sentinel.write_text(
            " ".join([
                f"rows={rows}",
                f"min_cos={min_cos:.6f}",
                f"cos_per_step={cos_per_step}",
                f"top1_agree={n_top1_agree}/{rows}",
                f"generated_ids={generated_ids}",
            ]) + "\n"
        )
        print("\n# GEMMA SWIFT DECODE-LOGIT GATE: PASS")
        print(f"  sentinel: {sentinel}")
        return 0

    if sentinel.exists():
        sentinel.unlink()
    if generated_ids_mismatch is not None:
        print(f"\n  {generated_ids_mismatch}")
    print("\n# GEMMA SWIFT DECODE-LOGIT GATE: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())