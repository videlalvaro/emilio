"""Inspect decode-boundary rows for a Swift Gemma decode dump.

This helper compares selected post-token decode rows from a Swift dump against
the saved golden rows in `python/moe/out/gemma_golden.npz`. It is meant for
localizing where a long decode first degrades: whether a row is still on the
same generated-token prefix, whether the next-token choice is a narrow tie, and
which logits contribute most to the row-wise drift.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path("python/moe/out")
DEFAULT_PREFIX = Path("/tmp/gemma_swift_t415_decode_full_trace_run2")
GOLDEN_NPZ = OUT_DIR / "gemma_golden.npz"


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.astype(np.float64).ravel()
    b64 = b.astype(np.float64).ravel()
    return float(np.dot(a64, b64) / (np.linalg.norm(a64) * np.linalg.norm(b64) + 1e-12))


def _topk(arr: np.ndarray, k: int) -> list[tuple[int, float]]:
    idx = np.argpartition(arr, -k)[-k:]
    idx = idx[np.argsort(arr[idx])[::-1]]
    return [(int(i), float(arr[i])) for i in idx]


def _margin(arr: np.ndarray) -> float:
    top2 = np.partition(arr, -2)[-2:]
    return float(top2.max() - top2.min())


def _parse_steps(raw: str) -> list[int]:
    return [int(piece.strip()) for piece in raw.split(",") if piece.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=str(DEFAULT_PREFIX),
                    help="prefix used by --dump-decode-logits-prefix")
    ap.add_argument("--golden", default=str(GOLDEN_NPZ),
                    help="golden decode NPZ")
    ap.add_argument("--steps", default="4,5,6",
                    help="comma-separated zero-based decode rows to inspect")
    ap.add_argument("--top-k", type=int, default=10,
                    help="number of top logits to print per row")
    ap.add_argument("--delta-k", type=int, default=10,
                    help="number of largest absolute logit deltas to print per row")
    args = ap.parse_args()

    prefix = Path(args.prefix)
    meta_path = Path(str(prefix) + "_meta.json")
    logits_path = Path(str(prefix) + "_logits_f32.bin")
    golden_path = Path(args.golden)

    meta = json.loads(meta_path.read_text())
    rows = int(meta["rows"])
    cols = int(meta["cols"])
    generated_ids = [int(x) for x in meta.get("generated_ids", [])]

    swift_logits = np.fromfile(logits_path, dtype=np.float32).reshape(rows, cols)
    gold = np.load(golden_path, allow_pickle=False)
    golden_next_ids = gold["next_token_ids"].astype(np.int64)
    golden_next_logits = gold["next_token_logits"].astype(np.float32)

    steps = _parse_steps(args.steps)
    top_k = args.top_k
    delta_k = args.delta_k

    for step in steps:
        if step >= rows or step >= golden_next_logits.shape[0]:
            print(f"=== step {step} ===")
            print("out of range")
            print()
            continue

        swift_row = swift_logits[step]
        golden_row = golden_next_logits[step]
        swift_argmax = int(np.argmax(swift_row))
        golden_argmax = int(np.argmax(golden_row))
        same_prefix = generated_ids[:step + 1] == golden_next_ids[:step + 1].tolist()
        delta = np.abs(swift_row - golden_row)
        delta_idx = np.argpartition(delta, -delta_k)[-delta_k:]
        delta_idx = delta_idx[np.argsort(delta[delta_idx])[::-1]]

        print(f"=== step {step} ===")
        print("same_prefix", same_prefix)
        print("emitted_token_swift", generated_ids[step] if step < len(generated_ids) else None)
        print("emitted_token_hf", int(golden_next_ids[step]))
        print("cos", round(_cos(swift_row, golden_row), 6))
        print("swift_argmax", swift_argmax, "swift_logit", float(swift_row[swift_argmax]))
        print("hf_argmax", golden_argmax, "hf_logit", float(golden_row[golden_argmax]))
        print("swift@hf_argmax", float(swift_row[golden_argmax]))
        print("hf@swift_argmax", float(golden_row[swift_argmax]))
        print("swift margin top1-top2", _margin(swift_row))
        print("hf margin top1-top2", _margin(golden_row))
        print("top10_overlap", len({idx for idx, _ in _topk(swift_row, top_k)}
                                   & {idx for idx, _ in _topk(golden_row, top_k)}))
        print("swift_topk", _topk(swift_row, top_k))
        print("hf_topk", _topk(golden_row, top_k))
        print(
            "largest_abs_deltas",
            [
                (int(i), float(delta[i]), float(swift_row[i]), float(golden_row[i]))
                for i in delta_idx
            ],
        )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())