"""Generate per-layer hidden-state trace from HF Gemma-4 for BOS token.

Runs the HF model on a single BOS token and prints the hidden state
after attention and after FFN for each layer (L2 norm + first 4 values).
This provides a reference to compare against the Swift driver's --trace-layers.

Usage:
    .venv313/bin/python python/moe/gemma_hf_layer_trace.py
"""
from __future__ import annotations

import sys
import math
import torch
import numpy as np
from pathlib import Path

MODEL_DIR = Path("models/gemma-4-26b-a4b")
D_MODEL = 2816


def main():
    from transformers import Gemma4ForConditionalGeneration, AutoConfig

    print("Loading model config...")
    config = AutoConfig.from_pretrained(str(MODEL_DIR))
    tc = config.text_config
    print(f"  layers: {tc.num_hidden_layers}")
    print(f"  hidden_size: {tc.hidden_size}")

    print("Loading model (bf16)...")
    full_model = Gemma4ForConditionalGeneration.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.bfloat16, device_map="cpu")
    full_model.eval()
    # Navigate: full_model.model.language_model is the text decoder
    lm = full_model.model.language_model

    # Single BOS token
    input_ids = torch.tensor([[2]], dtype=torch.long)

    print("\nRunning forward with hooks to capture per-layer states...")

    with torch.no_grad():
        outputs = lm(input_ids, output_hidden_states=True)

    # output_hidden_states gives hidden state AFTER each layer (and initial embedding)
    # hidden_states[0] = embedding output
    # hidden_states[i+1] = output of layer i (after FFN + residual + layer_scalar)
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        # Try accessing from a different attribute
        print("ERROR: output_hidden_states not returned, trying base_model_output")
        sys.exit(1)

    print(f"\nGot {len(hidden_states)} hidden states (embed + {len(hidden_states)-1} layers)")

    # Embedding
    emb = hidden_states[0][0, 0].float().numpy()  # (D_MODEL,)
    norm = np.linalg.norm(emb)
    print(f"  embed    x[0:4]=[{emb[0]:.4f},{emb[1]:.4f},{emb[2]:.4f},{emb[3]:.4f}] L2={norm:.5f}")

    for i in range(1, len(hidden_states)):
        h = hidden_states[i][0, 0].float().numpy()
        norm = np.linalg.norm(h)
        print(f"  L{i-1:2d} ffn   x[0:4]=[{h[0]:.4f},{h[1]:.4f},{h[2]:.4f},{h[3]:.4f}] L2={norm:.5f}")

    # Also get final logits via lm_head
    with torch.no_grad():
        final_h = hidden_states[-1]  # last hidden state after final norm
        logit_out = full_model.lm_head(final_h)
        logits = logit_out[0, 0].float().numpy()
    top1 = int(np.argmax(logits))
    print(f"\n  final logit top1={top1}, max={logits[top1]:.4f}")

    # Save per-layer hidden states for comparison with Swift trace
    save_path = Path("python/moe/out/gemma_hf_layer_hidden_bos.npz")
    save_dict = {}
    for i in range(len(hidden_states)):
        h = hidden_states[i][0, 0].float().numpy()
        save_dict[f"layer_{i}"] = h
    np.savez(str(save_path), **save_dict)
    print(f"\n  saved per-layer hidden states to {save_path}")

    # Now compare per-layer norms
    print("\n=== Summary: per-layer L2 norms ===")
    for i in range(1, len(hidden_states)):
        h = hidden_states[i][0, 0].float().numpy()
        norm = np.linalg.norm(h)
        print(f"  L{i-1:2d}: {norm:.2f}")


if __name__ == "__main__":
    main()
