"""Interactive inference wrapper for the ANE privacy-filter pipeline.

Tokenizes arbitrary text with tiktoken (o200k_base), writes temporary .bin
fixtures, invokes the Swift pf_e2e driver, reads back argmax labels, and
decodes BIOES tags into entity spans with character offsets.

Usage:
    # Single sentence:
    .venv313/bin/python python/privacy/pf_infer.py "My SSN is 123-45-6789."

    # Multiple sentences (one per line from stdin):
    echo -e "Call Alice at 555-1234.\\nEmail bob@test.com" | \
        .venv313/bin/python python/privacy/pf_infer.py -

    # From a file:
    .venv313/bin/python python/privacy/pf_infer.py -f docs.txt

    # Pipe a document (splits on newlines, skips blanks):
    cat my_doc.txt | .venv313/bin/python python/privacy/pf_infer.py -

Requires: tiktoken (installed in .venv313).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SWIFT_BIN = REPO / "emilio" / "conv-ane" / "pf_e2e"
SDIR = REPO / "emilio" / "conv-ane" / "PF_swift"
LABELS_OUT = REPO / "python" / "privacy" / "out" / "swift_dump" / "swift_argmax_labels.bin"
CFG_PATH = REPO / "python" / "privacy" / "_vendor_src" / "weights" / "config.json"
GOLDEN_PATH = REPO / "python" / "privacy" / "out" / "pf_golden.npz"

T_SEQ = 128
N_LABELS = 33


def _load_id2label() -> dict[int, str]:
    g = np.load(GOLDEN_PATH, allow_pickle=False)
    raw = json.loads(str(g["id2label_json"]))
    return {int(k): v for k, v in raw.items()}


def _tokenize(encoding, sentences: list[str], pad_id: int) -> tuple[np.ndarray, np.ndarray]:
    B = len(sentences)
    ids = np.full((B, T_SEQ), pad_id, dtype=np.int32)
    mask = np.zeros((B, T_SEQ), dtype=np.int32)
    for i, s in enumerate(sentences):
        toks = encoding.encode(s, allowed_special="all")
        n = min(len(toks), T_SEQ)
        if len(toks) > T_SEQ:
            print(f"  [warn] sentence {i} truncated: {len(toks)} -> {T_SEQ} tokens",
                  file=sys.stderr)
        ids[i, :n] = toks[:n]
        mask[i, :n] = 1
    return ids, mask


def _decode_spans(labels: list[str]) -> list[tuple[int, int, str]]:
    """BIOES label list -> [(start_tok, end_tok_inclusive, entity_type)]."""
    spans: list[tuple[int, int, str]] = []
    i, n = 0, len(labels)
    while i < n:
        tag = labels[i]
        if tag == "O" or tag == "<pad>" or "-" not in tag:
            i += 1; continue
        prefix, etype = tag.split("-", 1)
        if prefix == "S":
            spans.append((i, i, etype)); i += 1; continue
        if prefix == "B":
            j = i + 1
            while j < n and labels[j].endswith("-" + etype) and labels[j][0] in ("I", "E"):
                if labels[j].startswith("E-"):
                    j += 1; break
                j += 1
            spans.append((i, j - 1, etype)); i = j; continue
        i += 1
    return spans


def _annotate_text(text: str, encoding, tokens: list[int], spans, id2label) -> str:
    """Rebuild text with [TYPE: ...] brackets around detected entities."""
    # Map token indices to byte offsets in the original text.
    byte_ranges: list[tuple[int, int]] = []
    cursor = 0
    text_bytes = text.encode("utf-8")
    for tid in tokens:
        tok_bytes = encoding.decode_bytes([tid])
        start = text_bytes.find(tok_bytes, cursor)
        if start == -1:
            # Fallback: greedy advance by decoded length
            start = cursor
        end = start + len(tok_bytes)
        byte_ranges.append((start, end))
        cursor = end

    # Sort spans by start position, build annotated string.
    result = []
    last_end = 0
    for s_tok, e_tok, etype in sorted(spans, key=lambda x: x[0]):
        char_start = len(text_bytes[:byte_ranges[s_tok][0]].decode("utf-8", errors="replace"))
        char_end = len(text_bytes[:byte_ranges[e_tok][1]].decode("utf-8", errors="replace"))
        result.append(text[last_end:char_start])
        result.append(f"[{etype.upper()}]")
        last_end = char_end
    result.append(text[last_end:])
    return "".join(result)


def main() -> int:
    ap = argparse.ArgumentParser(description="ANE privacy-filter inference on arbitrary text")
    ap.add_argument("text", nargs="?", default=None,
                    help='Text to scan (use "-" for stdin, or -f for file)')
    ap.add_argument("-f", "--file", type=Path, default=None,
                    help="Read sentences from file (one per line)")
    ap.add_argument("--raw", action="store_true",
                    help="Print raw BIOES tags instead of annotated text")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Show Swift driver output and entity details")
    ap.add_argument("--batch", type=int, default=8,
                    help="Max sentences per batch (default: 8, must match N_SENT)")
    args = ap.parse_args()

    # Gather input sentences.
    if args.file:
        lines = [l.strip() for l in args.file.read_text().splitlines() if l.strip()]
    elif args.text == "-":
        lines = [l.strip() for l in sys.stdin.readlines() if l.strip()]
    elif args.text:
        lines = [args.text]
    else:
        ap.print_help()
        return 1

    if not lines:
        print("No input sentences.", file=sys.stderr)
        return 1

    import tiktoken
    cfg = json.loads(CFG_PATH.read_text())
    encoding = tiktoken.get_encoding(cfg["encoding"])
    pad_id = int(encoding.eot_token)
    id2label = _load_id2label()

    batch_size = args.batch
    all_labels: list[list[str]] = []

    for batch_start in range(0, len(lines), batch_size):
        batch = lines[batch_start:batch_start + batch_size]
        # Pad batch to exactly batch_size (Swift driver expects N_SENT).
        padded = batch + [""] * (batch_size - len(batch))

        ids, mask = _tokenize(encoding, padded, pad_id)

        # Write temp .bin files over the fixture dir.
        (SDIR / "input_ids.bin").write_bytes(ids.tobytes())
        (SDIR / "attention_mask.bin").write_bytes(mask.tobytes())
        # Golden logits: dummy (not used for inference, only for validation printout).
        dummy_golden = np.zeros((batch_size, T_SEQ, N_LABELS), dtype=np.float32)
        (SDIR / "golden_logits.bin").write_bytes(dummy_golden.tobytes())

        # Run Swift driver.
        result = subprocess.run(
            [str(SWIFT_BIN)],
            capture_output=True, text=True,
            env={**__import__("os").environ, "PF_INFER_MODE": "1"},
        )
        if result.returncode != 0:
            print(f"Swift driver failed:\n{result.stderr}", file=sys.stderr)
            return 1

        # Print Swift driver output (load times, cos, etc.) to stderr if verbose.
        if args.verbose:
            for line in result.stdout.splitlines():
                if line.startswith("["):
                    print(line, file=sys.stderr)

        # Read argmax labels.
        labels_raw = np.fromfile(str(LABELS_OUT), dtype=np.int32).reshape(batch_size, T_SEQ)
        for i, sent in enumerate(batch):
            n_tok = int(mask[i].sum())
            lab = [id2label[int(labels_raw[i, t])] for t in range(n_tok)]
            all_labels.append(lab)

    # Output results.
    for i, (sent, lab) in enumerate(zip(lines, all_labels)):
        toks = encoding.encode(sent, allowed_special="all")
        spans = _decode_spans(lab)
        if args.raw:
            print(f"--- sentence {i} ---")
            for t, l in enumerate(lab):
                tok_text = encoding.decode([toks[t]]) if t < len(toks) else "<pad>"
                print(f"  {t:3d}  {l:20s}  {tok_text!r}")
        else:
            if spans:
                annotated = _annotate_text(sent, encoding, toks[:len(lab)], spans, id2label)
                print(annotated)
            else:
                print(sent)
        if spans and not args.raw and args.verbose:
            for s, e, etype in spans:
                tok_text = encoding.decode(toks[s:e + 1])
                print(f"  -> {etype}: {tok_text!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
