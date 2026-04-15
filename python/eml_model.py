"""
Tiny EML Transformer - forward pass using only eml(x,y) = exp(x) - ln(y)

Architecture: 1 layer, 2 heads, d_model=16, d_ff=32, vocab=64
Every mathematical operation traced back to EML primitives.

This is a POC: weights are random (no training), but the forward pass
is architecturally complete and token generation works.
"""

import numpy as np
from eml_core import (
    eml, eml_exp, eml_ln, eml_softmax, eml_layer_norm,
    eml_matmul, eml_gelu, eml_sqrt, eml_mul, eml_add, eml_div,
    eml_sub, eml_neg, eml_inv
)

# ─── Config ───────────────────────────────────────────────────────────────────

VOCAB    = 64      # small vocab
D_MODEL  = 16      # embedding dim
N_HEADS  = 2       # attention heads
D_HEAD   = D_MODEL // N_HEADS   # 8
D_FF     = 32      # feedforward dim
MAX_SEQ  = 16      # max context length


# ─── Weight init ──────────────────────────────────────────────────────────────

def init_weights(rng: np.random.Generator):
    """Random weights. In a real model these would be loaded from a checkpoint."""
    w = {}
    
    # Token embedding
    w['wte'] = rng.standard_normal((VOCAB, D_MODEL)).astype(np.float64) * 0.02
    
    # Positional embedding  
    w['wpe'] = rng.standard_normal((MAX_SEQ, D_MODEL)).astype(np.float64) * 0.02
    
    # Attention: Q, K, V projections + output
    w['attn_qkv'] = rng.standard_normal((D_MODEL, 3 * D_MODEL)).astype(np.float64) * 0.02
    w['attn_out']  = rng.standard_normal((D_MODEL, D_MODEL)).astype(np.float64) * 0.02
    
    # Layer norm 1 (pre-attention)
    w['ln1_g'] = np.ones(D_MODEL,  dtype=np.float64)
    w['ln1_b'] = np.zeros(D_MODEL, dtype=np.float64)
    
    # FFN
    w['ff1'] = rng.standard_normal((D_MODEL, D_FF)).astype(np.float64) * 0.02
    w['ff2'] = rng.standard_normal((D_FF, D_MODEL)).astype(np.float64) * 0.02
    
    # Layer norm 2 (pre-FFN)
    w['ln2_g'] = np.ones(D_MODEL,  dtype=np.float64)
    w['ln2_b'] = np.zeros(D_MODEL, dtype=np.float64)
    
    # Final layer norm
    w['lnf_g'] = np.ones(D_MODEL,  dtype=np.float64)
    w['lnf_b'] = np.zeros(D_MODEL, dtype=np.float64)
    
    # LM head (tied to wte for simplicity)
    w['lm_head'] = w['wte'].T.copy()  # (D_MODEL, VOCAB)
    
    return w


# ─── EML Attention ────────────────────────────────────────────────────────────

def eml_attention(x, w, mask=None):
    """
    Multi-head self-attention — all ops EML-derived.
    
    x: (T, D_MODEL)
    
    Steps:
    1. QKV projection: matmul (sum of EML-mul)
    2. Split heads, reshape
    3. Scaled dot-product: matmul + scale by 1/sqrt(d_head)
    4. Causal mask (addition of -inf — EML: add large negative)
    5. Softmax: all-EML (shown in eml_core.py)
    6. Weighted sum: matmul
    7. Output projection: matmul
    """
    T, D = x.shape
    
    # 1. QKV projection: (T, D) @ (D, 3D) → (T, 3D)
    #    Each element = sum_k eml_mul(x[i,k], w[k,j])
    #    = sum_k exp(ln(x[i,k]) + ln(w[k,j]))  [for positive values]
    qkv = eml_matmul(x, w['attn_qkv'])  # (T, 3*D_MODEL)
    
    Q = qkv[:, :D_MODEL]           # (T, D_MODEL)
    K = qkv[:, D_MODEL:2*D_MODEL]  # (T, D_MODEL)
    V = qkv[:, 2*D_MODEL:]         # (T, D_MODEL)
    
    # 2. Split into heads: (T, D_MODEL) → (N_HEADS, T, D_HEAD)
    Q = Q.reshape(T, N_HEADS, D_HEAD).transpose(1, 0, 2)
    K = K.reshape(T, N_HEADS, D_HEAD).transpose(1, 0, 2)
    V = V.reshape(T, N_HEADS, D_HEAD).transpose(1, 0, 2)
    
    # 3. Scaled dot-product attention
    #    scale = 1/sqrt(D_HEAD) = exp(-0.5 * ln(D_HEAD))
    #    In EML: eml_exp(eml_mul(-0.5, eml_ln(D_HEAD)))
    scale = eml_exp(eml_mul(eml_neg(0.5), eml_ln(float(D_HEAD))))
    
    # scores[h,i,j] = sum_k Q[h,i,k] * K[h,j,k] * scale
    # Pure EML: dot product per (h,i,j) then mul by scale
    scores = np.empty((N_HEADS, T, T), dtype=np.float64)
    for h in range(N_HEADS):
        for i in range(T):
            for j in range(T):
                # dot(Q[h,i,:], K[h,j,:]) * scale
                acc = eml_mul(Q[h, i, 0], K[h, j, 0])
                for d in range(1, D_HEAD):
                    acc = eml_add(acc, eml_mul(Q[h, i, d], K[h, j, d]))
                scores[h, i, j] = np.real(eml_mul(acc, scale))
    
    # 4. Causal mask: add large negative to future positions
    #    -1e9 = eml_neg(1e9) — EML constructible
    large_neg = np.real(eml_neg(1e9))
    if mask is not None:
        scores = np.vectorize(lambda a, b: np.real(eml_add(a, b)))(scores, mask)
    else:
        causal = np.triu(np.full((T, T), large_neg), k=1)  # (T, T)
        scores = np.vectorize(lambda a, b: np.real(eml_add(a, b)))(scores, causal[np.newaxis, :, :])
    
    # 5. Softmax over last dim: all-EML
    #    attn[h,i,:] = softmax(scores[h,i,:])
    attn = np.stack([
        np.stack([eml_softmax(scores[h, i, :]) for i in range(T)])
        for h in range(N_HEADS)
    ])  # (N_HEADS, T, T)
    
    # 6. Weighted sum: (N_HEADS, T, T) @ (N_HEADS, T, D_HEAD) → (N_HEADS, T, D_HEAD)
    #    Pure EML: weighted sum per element
    out = np.empty((N_HEADS, T, D_HEAD), dtype=np.float64)
    for h in range(N_HEADS):
        for i in range(T):
            for d in range(D_HEAD):
                acc = eml_mul(attn[h, i, 0], V[h, 0, d])
                for j in range(1, T):
                    acc = eml_add(acc, eml_mul(attn[h, i, j], V[h, j, d]))
                out[h, i, d] = np.real(acc)
    
    # 7. Merge heads: (N_HEADS, T, D_HEAD) → (T, D_MODEL)
    out = out.transpose(1, 0, 2).reshape(T, D_MODEL)
    
    # 8. Output projection
    out = eml_matmul(out, w['attn_out'])  # (T, D_MODEL)
    
    return out


# ─── EML FFN ──────────────────────────────────────────────────────────────────

def eml_ffn(x, w):
    """
    Feed-forward: x → Linear → GELU → Linear
    
    GELU = x * sigmoid(1.702x) — fully EML-derived (see eml_core.py)
    Linear = matmul + bias (no bias here for simplicity)
    """
    h = eml_matmul(x, w['ff1'])   # (T, D_FF)
    h = np.vectorize(lambda v: np.real(eml_gelu(v)))(h)  # elementwise GELU — EML
    h = eml_matmul(h, w['ff2'])   # (T, D_MODEL)
    return h


# ─── EML Transformer Layer ────────────────────────────────────────────────────

def eml_transformer_layer(x, w):
    """
    Pre-norm transformer block:
    x = x + Attention(LayerNorm(x))
    x = x + FFN(LayerNorm(x))
    
    All residual additions are EML add.
    LayerNorm uses eml_sqrt internally.
    """
    # Pre-attention layer norm
    x_norm = eml_layer_norm(x, w['ln1_g'], w['ln1_b'])
    
    # Attention + residual
    attn_out = eml_attention(x_norm, w)
    x = np.vectorize(lambda a, b: np.real(eml_add(a, b)))(x, attn_out)   # residual add — pure EML
    
    # Pre-FFN layer norm  
    x_norm = eml_layer_norm(x, w['ln2_g'], w['ln2_b'])
    
    # FFN + residual
    ffn_out = eml_ffn(x_norm, w)
    x = np.vectorize(lambda a, b: np.real(eml_add(a, b)))(x, ffn_out)    # residual add — pure EML
    
    return x


# ─── Full Forward Pass ────────────────────────────────────────────────────────

def eml_forward(token_ids: list[int], w: dict) -> np.ndarray:
    """
    Full transformer forward pass.
    Returns logits of shape (T, VOCAB).
    
    token_ids: list of ints in [0, VOCAB)
    
    EML trace for the embedding lookup:
    - Embedding = table lookup (discrete, not EML) → float vectors
    - Positional embedding: add (EML)
    - All subsequent ops: EML-derived
    """
    T = len(token_ids)
    assert T <= MAX_SEQ, f"Sequence too long: {T} > {MAX_SEQ}"
    
    # Token embeddings (discrete lookup, not EML — unavoidable for discrete tokens)
    tok_emb = w['wte'][token_ids]   # (T, D_MODEL)
    
    # Positional embeddings + add (pure EML)
    pos_emb = w['wpe'][:T]           # (T, D_MODEL)
    x = np.vectorize(lambda a, b: np.real(eml_add(a, b)))(tok_emb, pos_emb)  # elementwise add — pure EML
    
    # Transformer layer
    x = eml_transformer_layer(x, w)
    
    # Final layer norm
    x = eml_layer_norm(x, w['lnf_g'], w['lnf_b'])
    
    # LM head: (T, D_MODEL) @ (D_MODEL, VOCAB) → (T, VOCAB)
    logits = eml_matmul(x, w['lm_head'])
    
    return logits


# ─── Token Generation ─────────────────────────────────────────────────────────

def eml_generate(prompt_ids: list[int], w: dict, max_new: int = 8,
                 temperature: float = 1.0) -> list[int]:
    """
    Autoregressive generation.
    
    Sampling via softmax + argmax:
    - logits / temperature: div (EML)
    - softmax: EML (see eml_core.py)
    - argmax: comparison (discrete, not EML — unavoidable for token selection)
    """
    ids = list(prompt_ids)
    
    for _ in range(max_new):
        if len(ids) >= MAX_SEQ:
            ids = ids[-MAX_SEQ:]
        
        # Forward pass — all EML inside
        logits = eml_forward(ids, w)
        
        # Next token logits (last position)
        next_logits = logits[-1]  # (VOCAB,)
        
        # Temperature scaling: logits / T = mul(logits, 1/T)
        # 1/T = exp(-ln(T)) = eml_exp(eml_mul(-1, eml_ln(T)))
        if temperature != 1.0:
            inv_temp = eml_inv(temperature)
            next_logits = np.array([np.real(eml_mul(l, inv_temp)) for l in next_logits])  # pure EML mul
        
        # Softmax → probability distribution (all EML)
        probs = eml_softmax(next_logits)
        
        # Sample: argmax (greedy) — discrete selection, not EML
        next_token = int(np.argmax(probs))
        ids.append(next_token)
    
    return ids
