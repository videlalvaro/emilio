"""Shared prompt spec for the low-entropy factual regression battery.

These prompts are chosen so the first 1-2 generated tokens should contain the
entire factual answer. This avoids the high-entropy continuation problem in the
original `gemma_golden.npz` oracle, where a single near-tie caused the rest of
the decode path to diverge while staying semantically coherent.
"""
from __future__ import annotations

from pathlib import Path

OUT_DIR = Path("python/moe/out")
HF_OUT_NPZ = OUT_DIR / "gemma_hf_closedform_battery.npz"
HF_SENTINEL = OUT_DIR / ".gemma_hf_closedform_battery_PASS"
GATE_SENTINEL = OUT_DIR / ".gemma_t415_closedform_battery_PASS"

# Capture a short answer prefix plus one or two trailing tokens for context.
N_NEW = 4
MIN_REQUIRED_MARGIN = 1.0

# (stable key, prompt, required HF-prefix tokens that CML must match)
BATTERY = (
    ("gold_symbol", "The chemical symbol for gold is", 1),
    ("silver_symbol", "The chemical symbol for silver is", 1),
    ("iron_symbol", "The chemical symbol for iron is", 1),
)

TOTAL_REQUIRED_PREFIX = sum(required for _, _, required in BATTERY)