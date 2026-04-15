#!/usr/bin/env python3
"""Generate a tiny synthetic .eml v2 model file for testing.

Creates a valid EML v2 binary with:
  - Tiny dimensions (d_model=16, 1 layer, vocab=259)
  - Full tokenizer (all 256 byte mappings + special tokens)
  - Random weights (output will be gibberish, but exercises full pipeline)
  - File size: ~300 KB

Usage:
    python3 make_tiny_model.py [output_path]
    # Default: tiny_model.eml
"""

import struct
import math
import random
import sys

random.seed(42)

OUTPUT = sys.argv[1] if len(sys.argv) > 1 else "tiny_model.eml"

# --- Model config (very small) ---
vocab_size = 259  # 256 byte tokens + 3 special
n_layers = 1
n_heads = 2
n_kv_heads = 2
d_model = 16
d_ff = 32
d_head = 8  # n_heads * d_head = 16 = d_model
max_seq_len = 256
rope_freq_base = 10000.0
rms_norm_eps = 1e-6

# Derived
q_dim = n_heads * d_head  # 16
kv_dim = n_kv_heads * d_head  # 16
qkv_dim = q_dim + 2 * kv_dim  # 48

# --- Writer helpers ---


def write_u32(f, v):
    f.write(struct.pack("<I", v))


def write_u64(f, v):
    f.write(struct.pack("<Q", v))


def write_f64(f, v):
    f.write(struct.pack("<d", v))


def write_string(f, s):
    b = s.encode("utf-8")
    write_u32(f, len(b))
    f.write(b)


def write_f64_array(f, arr):
    write_u64(f, len(arr))
    for v in arr:
        write_f64(f, v)


def write_sm_tensor(f, n, magnitudes, signs):
    """Write SmTensor: u64 len, magnitudes (f64s), packed sign bitmap."""
    assert len(magnitudes) == n
    assert len(signs) == n
    write_u64(f, n)
    for m in magnitudes:
        write_f64(f, m)
    packed_len = (n + 7) // 8
    packed = bytearray(packed_len)
    for i in range(n):
        if signs[i] < 0:
            packed[i // 8] |= 1 << (i % 8)
    write_u32(f, packed_len)
    f.write(bytes(packed))


def random_f64_array(n, scale=0.1):
    return [random.gauss(0, scale) for _ in range(n)]


def random_sm(n, scale=0.1):
    mags = [abs(random.gauss(0, scale)) for _ in range(n)]
    signs = [random.choice([-1.0, 1.0]) for _ in range(n)]
    return mags, signs


# --- Build vocab ---
# GPT-2 byte_to_unicode mapping
def byte_to_unicode():
    mapping = {}
    n = 256
    for b in range(256):
        if (33 <= b <= 126) or (161 <= b <= 172) or (174 <= b <= 255):
            mapping[b] = b
        else:
            mapping[b] = n
            n += 1
    return mapping


b2u = byte_to_unicode()

vocab = []
for byte_val in range(256):
    cp = b2u[byte_val]
    # Encode codepoint as UTF-8 string
    vocab.append(chr(cp))

# Special tokens at indices 256, 257, 258
vocab.append("<|im_start|>")
vocab.append("<|im_end|>")
vocab.append("<|endoftext|>")

assert len(vocab) == vocab_size

# BPE merges: none (each byte is its own token)
merges = []

# Special token IDs
bos_id = 258  # <|endoftext|>
eos_id = 258  # <|endoftext|>

# --- Exec graph (empty -- C loader skips it) ---
exec_graph_ops = []

# --- Write the file ---

with open(OUTPUT, "wb") as f:
    # Header
    f.write(b"EML2")
    write_u32(f, 2)  # version

    # Config
    write_u32(f, vocab_size)
    write_u32(f, n_layers)
    write_u32(f, n_heads)
    write_u32(f, n_kv_heads)
    write_u32(f, d_model)
    write_u32(f, d_ff)
    write_f64(f, rope_freq_base)
    write_f64(f, rms_norm_eps)
    write_u32(f, max_seq_len)
    write_u32(f, d_head)

    # Sparsity stats
    write_u64(f, 0)  # total_params
    write_u64(f, 0)  # pruned_params
    write_f64(f, 0.0)  # threshold

    # Tokenizer
    write_u32(f, vocab_size)
    write_u32(f, len(merges))
    write_u32(f, bos_id)
    write_u32(f, eos_id)

    # Vocab strings
    for tok in vocab:
        write_string(f, tok)

    # Merges (none)

    # Exec graph (empty)
    write_u32(f, 0)

    # --- Global weights ---

    # token_embd: (vocab_size * d_model)
    write_f64_array(f, random_f64_array(vocab_size * d_model, scale=0.02))

    # output_norm: (d_model)
    write_f64_array(f, [1.0] * d_model)  # unit gamma

    # sm_output: LM head (vocab_size * d_model)
    n_out = vocab_size * d_model
    mags, signs = random_sm(n_out, scale=0.1)
    write_sm_tensor(f, n_out, mags, signs)

    # --- Per-layer weights ---
    for layer_idx in range(n_layers):
        # sm_qkv: (qkv_dim * d_model) = 48 * 16 = 768
        n = qkv_dim * d_model
        m, s = random_sm(n, scale=0.1)
        write_sm_tensor(f, n, m, s)

        # sm_o: (d_model * q_dim) = 16 * 16 = 256
        n = d_model * q_dim
        m, s = random_sm(n, scale=0.1)
        write_sm_tensor(f, n, m, s)

        # sm_gate_up: (2 * d_ff * d_model) = 64 * 16 = 1024
        n = 2 * d_ff * d_model
        m, s = random_sm(n, scale=0.1)
        write_sm_tensor(f, n, m, s)

        # sm_down: (d_model * d_ff) = 16 * 32 = 512
        n = d_model * d_ff
        m, s = random_sm(n, scale=0.1)
        write_sm_tensor(f, n, m, s)

        # q_bias: q_dim = 16
        write_f64_array(f, [0.0] * q_dim)

        # k_bias: kv_dim = 16
        write_f64_array(f, [0.0] * kv_dim)

        # v_bias: kv_dim = 16
        write_f64_array(f, [0.0] * kv_dim)

        # attn_norm: d_model = 16
        write_f64_array(f, [1.0] * d_model)

        # ffn_norm: d_model = 16
        write_f64_array(f, [1.0] * d_model)

import os

size = os.path.getsize(OUTPUT)
print(f"Created {OUTPUT} ({size:,} bytes)")
print(f"  vocab_size={vocab_size}, n_layers={n_layers}, d_model={d_model}")
print(f"  n_heads={n_heads}, n_kv_heads={n_kv_heads}, d_head={d_head}, d_ff={d_ff}")
