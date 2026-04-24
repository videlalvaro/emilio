from __future__ import annotations

import os
from pathlib import Path

MODEL_DIR = Path(os.environ.get("QWEN36_MODEL_DIR", "models/qwen3.6-35b-a3b"))
OUT_DIR = Path("python/moe/out/qwen36")
WEIGHTS_OUT_DIR = OUT_DIR / "weights"
MANIFEST_PATH = WEIGHTS_OUT_DIR / "qwen36_weights_manifest.json"
GLOBAL_NPZ_PATH = WEIGHTS_OUT_DIR / "qwen36_global.npz"
GOLDEN_SENTINEL = OUT_DIR / ".qwen36_golden_PASS"
GOLDEN_LATEST_LINK = OUT_DIR / "qwen36_golden.npz"

DEFAULT_N_NEW = 8

# Keep this short and low-entropy: one arithmetic prompt, two factual prompts,
# and one short code/text prompt.
PROMPT_SUITE: list[tuple[str, str]] = [
    ("capital_france", "The capital of France is"),
    ("two_plus_two", "2 + 2 ="),
    ("silver_symbol", "The chemical symbol for silver is"),
    ("python_hello", "Write Python code that prints hello world:"),
]
