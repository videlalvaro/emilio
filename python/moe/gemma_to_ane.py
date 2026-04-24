"""gemma_to_ane.py — T4.1 of GEMMA_ANE_RESEARCH.md.

CoreML conversion of one Gemma-4-26B-A4B sliding layer for the ANE.

Status:
  T4.1.0 (skeleton with random weights, 1L)             — PASS, JOURNAL 2026-04-22
  T4.1.1 (real REAP-permuted weights + real top-k)      — THIS FILE

Layer-0 architecture (sliding, k_eq_v=False at sliding layers):
  residual = x
  h = input_ln(x);   a, k_new, v_new = self_attn(h, …);  x = residual + post_attn_ln(a)
  residual = x
  pre = pre_ffn_ln(x)
  dense = mlp_down(GELU(mlp_gate(pre)) * mlp_up(pre))
  h1 = post_ffn_ln_1(dense)
  pre2 = pre_ffn_ln_2(residual)
  rscores = router_proj( router_norm(residual) * router_scale * D_MODEL**-0.5 )
  rprob = softmax(rscores)
  topw, topi = topk(rprob, k=8); topw = topw / topw.sum * per_expert_scale[topi]
  dense_w = scatter(topi, topw, dim=64) → split into 4 packs of 16
  moe = sum_p combine(pack_p(pre2), pack_w_p)
  h2 = post_ffn_ln_2(moe)
  x = residual + post_ffn_ln(h1 + h2)
  x = x * layer_scalar

Run with the only python that has coremltools 9:
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
      python/moe/gemma_to_ane.py layer1 --max-ctx 1024
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEBUG_INTERNAL_TAP_ORDER = ("post_attn", "ffn_out")
DEBUG_INTERNAL_TAP_SET = set(DEBUG_INTERNAL_TAP_ORDER)
import coremltools as ct
import coremltools.optimize.coreml as cto

OUT_DIR = Path("python/moe/out"); OUT_DIR.mkdir(parents=True, exist_ok=True)

D_MODEL          = 2816
D_FFN            = 704
D_DENSE          = 2112
N_HEADS          = 16
N_EXPERTS_KEEP   = 64
TOP_K            = 8
PACK_G           = 16
N_PACKS          = N_EXPERTS_KEEP // PACK_G
RMS_EPS          = 1e-6

SLD_N_KV   = 8
SLD_D_HEAD = 256
SLD_WIN    = 1024

GLB_N_KV     = 2
GLB_D_HEAD   = 512
GLB_ROT_DIM  = 512   # full rotation — HF full_attention_inv_freq has 256 entries → cos/sin dim 512 = head_dim

# INT4 quant block size. bs=16 was the T4.1.2 baseline (PASS on layer1
# alone), but the global-layer probe (T4.1.3c attribution) showed each
# global layer drops cos by ~4× more than a sliding layer (0.964 vs 0.992).
# Sensitivity sweep on layers [4,5]: bs=8 gains 0.013 cos for +20% pkg size.
# Locked at bs=8 going forward; sliding layers also benefit slightly.
INT4_BLOCK_SIZE = 8


def _state_shape(is_global: bool, max_ctx: int):
    return (1, GLB_N_KV if is_global else SLD_N_KV,
            max_ctx,
            GLB_D_HEAD if is_global else SLD_D_HEAD)


class RMSNorm(nn.Module):
    """Gemma-style RMSNorm: weight stored as (γ-1) so we apply (1+γ)·x."""
    def __init__(self, dim: int, with_scale: bool = True, eps: float = RMS_EPS):
        super().__init__()
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.zeros(dim, dtype=torch.float16))
        self.eps = eps

    def forward(self, x):
        v = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(v + self.eps).to(x.dtype)
        if self.with_scale:
            x = x * (1.0 + self.weight)
        return x


def apply_rope(x, cos, sin):
    """Standard rotary embedding over full head_dim.
    x: (1, T, H, D); cos/sin: (1, T, D)."""
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin


def apply_rope_partial(x, cos, sin, rot_dim: int):
    """Rotate only the first `rot_dim` features per head; pass-through for the rest.
    x: (1, T, H, D); cos/sin: (1, T, rot_dim)."""
    x_rot = x[..., :rot_dim]
    x_pass = x[..., rot_dim:]
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    half = rot_dim // 2
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    rot = torch.cat([-x2, x1], dim=-1)
    rotated = x_rot * cos + rot * sin
    return torch.cat([rotated, x_pass], dim=-1)


class SlidingAttention(nn.Module):
    def __init__(self, max_ctx: int):
        super().__init__()
        self.n_kv  = SLD_N_KV
        self.dh    = SLD_D_HEAD
        self.q_dim = N_HEADS * self.dh
        self.kv_dim = self.n_kv * self.dh
        self.max_ctx = max_ctx
        self.q_proj = nn.Linear(D_MODEL, self.q_dim,  bias=False)
        self.k_proj = nn.Linear(D_MODEL, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(D_MODEL, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(self.q_dim, D_MODEL,  bias=False)
        self.q_norm = RMSNorm(self.dh, with_scale=True)
        self.k_norm = RMSNorm(self.dh, with_scale=True)
        self.v_norm = RMSNorm(self.dh, with_scale=False)

    def forward(self, x, cos, sin, k_cache, v_cache, attn_mask, kv_write_mask):
        q = self.q_proj(x).view(1, 1, N_HEADS, self.dh)
        k = self.k_proj(x).view(1, 1, self.n_kv, self.dh)
        v = self.v_proj(x).view(1, 1, self.n_kv, self.dh)
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        wm = kv_write_mask.to(k.dtype)
        k_new = k_cache + (k * wm).to(k_cache.dtype)
        v_new = v_cache + (v * wm).to(v_cache.dtype)

        rep = N_HEADS // self.n_kv
        K = k_new.repeat_interleave(rep, dim=1)
        V = v_new.repeat_interleave(rep, dim=1)
        # HF Gemma-4: self.scaling=1.0 (q_norm/k_norm provide magnitude control,
        # no 1/sqrt(d_head) division). See modeling_gemma4.py Gemma4TextAttention.
        scores = (q @ K.transpose(-2, -1)) + attn_mask
        w = F.softmax(scores.float(), dim=-1).to(q.dtype)
        ctx = w @ V
        ctx = ctx.transpose(1, 2).contiguous().view(1, 1, self.q_dim)
        return self.o_proj(ctx), k_new, v_new


class GlobalAttention(nn.Module):
    """Full-attention layer (k_eq_v=True, full RoPE 512/512, no v_proj).

    V is derived from the same projection as K (raw k_proj output), then runs
    through the unscaled RMSNorm (v_norm with γ=1) and is NOT rotated.
    K runs through k_norm (γ=γ_k) and full RoPE (same as HF full_attention).
    """
    def __init__(self, max_ctx: int):
        super().__init__()
        self.n_kv   = GLB_N_KV
        self.dh     = GLB_D_HEAD
        self.q_dim  = N_HEADS * self.dh
        self.kv_dim = self.n_kv * self.dh
        self.max_ctx = max_ctx
        self.q_proj = nn.Linear(D_MODEL, self.q_dim,  bias=False)
        self.k_proj = nn.Linear(D_MODEL, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(self.q_dim, D_MODEL,  bias=False)
        self.q_norm = RMSNorm(self.dh, with_scale=True)
        self.k_norm = RMSNorm(self.dh, with_scale=True)
        self.v_norm = RMSNorm(self.dh, with_scale=False)

    def forward(self, x, cos, sin, k_cache, v_cache, attn_mask, kv_write_mask):
        q_raw = self.q_proj(x).view(1, 1, N_HEADS, self.dh)
        k_raw = self.k_proj(x).view(1, 1, self.n_kv, self.dh)
        q = self.q_norm(q_raw)
        k = self.k_norm(k_raw)
        v = self.v_norm(k_raw)                                # k_eq_v: V from raw K
        q = apply_rope(q, cos, sin)           # full 512-dim rotation (same as HF)
        k = apply_rope(k, cos, sin)           # full 512-dim rotation (same as HF)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        wm = kv_write_mask.to(k.dtype)
        k_new = k_cache + (k * wm).to(k_cache.dtype)
        v_new = v_cache + (v * wm).to(v_cache.dtype)

        rep = N_HEADS // self.n_kv
        K = k_new.repeat_interleave(rep, dim=1)
        V = v_new.repeat_interleave(rep, dim=1)
        # HF Gemma-4: self.scaling=1.0 (q_norm/k_norm provide magnitude control,
        # no 1/sqrt(d_head) division). See modeling_gemma4.py Gemma4TextAttention.
        scores = (q @ K.transpose(-2, -1)) + attn_mask
        w = F.softmax(scores.float(), dim=-1).to(q.dtype)
        ctx = w @ V
        ctx = ctx.transpose(1, 2).contiguous().view(1, 1, self.q_dim)
        return self.o_proj(ctx), k_new, v_new


class DenseMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(D_MODEL, D_DENSE, bias=False)
        self.up   = nn.Linear(D_MODEL, D_DENSE, bias=False)
        self.down = nn.Linear(D_DENSE, D_MODEL, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.gate(x), approximate="tanh") * self.up(x))


class PackedExpertMLP(nn.Module):
    """16 fused experts. Returns per-expert outputs (1, 1, G, D_MODEL)."""
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(D_MODEL, PACK_G * D_FFN,   bias=False)
        self.up   = nn.Linear(D_MODEL, PACK_G * D_FFN,   bias=False)
        self.down = nn.Linear(D_FFN,   PACK_G * D_MODEL, bias=False)

    def forward(self, x):
        g = F.gelu(self.gate(x), approximate="tanh")
        u = self.up(x)
        h = (g * u).view(1, 1, PACK_G, D_FFN).reshape(PACK_G, D_FFN)
        out = self.down(h).view(PACK_G, PACK_G, D_MODEL)
        idx = torch.arange(PACK_G)
        return out[idx, idx].view(1, 1, PACK_G, D_MODEL)


class Router(nn.Module):
    """Top-K=8 router over 64 experts. Returns dense (1, 1, 64) routing vector."""
    def __init__(self):
        super().__init__()
        self.norm = RMSNorm(D_MODEL, with_scale=False)
        self.proj = nn.Linear(D_MODEL, N_EXPERTS_KEEP, bias=False)
        self.scale = nn.Parameter(torch.ones(D_MODEL, dtype=torch.float16))
        self.per_expert_scale = nn.Parameter(
            torch.ones(N_EXPERTS_KEEP, dtype=torch.float16))
        self._scalar_root = D_MODEL ** -0.5

    def forward(self, x):
        h = self.norm(x) * self.scale * self._scalar_root
        scores = self.proj(h)
        prob = F.softmax(scores, dim=-1)
        topw, topi = torch.topk(prob, k=TOP_K, dim=-1)
        topw = topw / topw.sum(dim=-1, keepdim=True)
        topw = topw * self.per_expert_scale[topi]
        oh = F.one_hot(topi, N_EXPERTS_KEEP).to(topw.dtype)
        return (topw.unsqueeze(-1) * oh).sum(dim=-2)


class GemmaSlidingLayer(nn.Module):
    is_global = False

    def __init__(self, max_ctx: int):
        super().__init__()
        self.max_ctx = max_ctx
        self.input_ln       = RMSNorm(D_MODEL)
        self.attn           = SlidingAttention(max_ctx)
        self.post_attn_ln   = RMSNorm(D_MODEL)
        self.pre_ffn_ln     = RMSNorm(D_MODEL)
        self.mlp_dense      = DenseMLP()
        self.post_ffn_ln_1  = RMSNorm(D_MODEL)
        self.pre_ffn_ln_2   = RMSNorm(D_MODEL)
        self.router         = Router()
        self.packs          = nn.ModuleList([PackedExpertMLP() for _ in range(N_PACKS)])
        self.post_ffn_ln_2  = RMSNorm(D_MODEL)
        self.post_ffn_ln    = RMSNorm(D_MODEL)
        self.register_buffer("layer_scalar", torch.ones(1, dtype=torch.float16))

    def forward(self, x, cos, sin, k_cache, v_cache, attn_mask, kv_write_mask):
        residual = x
        h = self.input_ln(x)
        a, k_new, v_new = self.attn(h, cos, sin, k_cache, v_cache,
                                    attn_mask, kv_write_mask)
        x = residual + self.post_attn_ln(a)

        residual = x
        pre = self.pre_ffn_ln(x)
        dense = self.mlp_dense(pre)
        h1 = self.post_ffn_ln_1(dense)

        rw = self.router(residual)                                # (1, 1, 64)
        rw = rw.view(1, 1, N_PACKS, PACK_G)                       # (1, 1, P, G)
        pre2 = self.pre_ffn_ln_2(residual)
        moe = torch.zeros(1, 1, D_MODEL, dtype=x.dtype)
        for p, pack in enumerate(self.packs):
            per_e = pack(pre2)                                    # (1, 1, G, D_MODEL)
            w_p = rw[:, :, p, :].unsqueeze(-1)                    # (1, 1, G, 1)
            moe = moe + (per_e * w_p).sum(dim=-2)                 # (1, 1, D_MODEL)
        h2 = self.post_ffn_ln_2(moe)

        x_ffn = h1 + h2
        x_ffn = self.post_ffn_ln(x_ffn)
        x = residual + x_ffn
        x = x * self.layer_scalar
        return x, k_new, v_new

    def forward_internal_taps(self, x, cos, sin, k_cache, v_cache,
                              attn_mask, kv_write_mask,
                              tap_names: tuple[str, ...]):
        tap_name_set = set(tap_names)
        tap_values = {}

        residual = x
        h = self.input_ln(x)
        a, k_new, v_new = self.attn(h, cos, sin, k_cache, v_cache,
                                    attn_mask, kv_write_mask)
        x = residual + self.post_attn_ln(a)
        if "post_attn" in tap_name_set:
            tap_values["post_attn"] = x.view(1, D_MODEL)

        residual = x
        pre = self.pre_ffn_ln(x)
        dense = self.mlp_dense(pre)
        h1 = self.post_ffn_ln_1(dense)

        rw = self.router(residual)
        rw = rw.view(1, 1, N_PACKS, PACK_G)
        pre2 = self.pre_ffn_ln_2(residual)
        moe = torch.zeros(1, 1, D_MODEL, dtype=x.dtype)
        for p, pack in enumerate(self.packs):
            per_e = pack(pre2)
            w_p = rw[:, :, p, :].unsqueeze(-1)
            moe = moe + (per_e * w_p).sum(dim=-2)
        h2 = self.post_ffn_ln_2(moe)

        x_ffn = h1 + h2
        x_ffn = self.post_ffn_ln(x_ffn)
        if "ffn_out" in tap_name_set:
            tap_values["ffn_out"] = x_ffn.view(1, D_MODEL)
        x = residual + x_ffn
        x = x * self.layer_scalar
        return x, k_new, v_new, tuple(tap_values[name] for name in tap_names)


class GemmaGlobalLayer(GemmaSlidingLayer):
    """Same FFN/router/norms as sliding; only attention differs."""
    is_global = True

    def __init__(self, max_ctx: int):
        super().__init__(max_ctx)
        # Replace sliding attention with global attention.
        self.attn = GlobalAttention(max_ctx)


class GemmaLayer1Wrap(nn.Module):
    def __init__(self, max_ctx: int):
        super().__init__()
        self.max_ctx = max_ctx
        self.layer0 = GemmaSlidingLayer(max_ctx)
        self.register_buffer(
            "k_cache_0",
            torch.zeros(1, SLD_N_KV, max_ctx, SLD_D_HEAD, dtype=torch.float16))
        self.register_buffer(
            "v_cache_0",
            torch.zeros(1, SLD_N_KV, max_ctx, SLD_D_HEAD, dtype=torch.float16))

    def forward(self, x, cos, sin, attn_mask, kv_write_mask):
        x, k_new, v_new = self.layer0(
            x, cos, sin, self.k_cache_0, self.v_cache_0,
            attn_mask, kv_write_mask)
        # CoreML state write-back (lowers to coreml_update_state).
        self.k_cache_0[:] = k_new
        self.v_cache_0[:] = v_new
        return x.view(1, D_MODEL), k_new, v_new


class GemmaStackWrap(nn.Module):
    """N stacked sliding layers sharing position inputs.

    Each layer has its own KV cache (k_cache_i / v_cache_i state buffers).
    cos/sin/attn_mask/kv_write_mask are the same for all layers (single token,
    same pos). Outputs only the final hidden + last layer's k_new/v_new for
    sanity. Per-layer K/V can be read from state buffers post-call if needed.
    """
    def __init__(self, max_ctx: int, n_layers: int):
        super().__init__()
        self.max_ctx = max_ctx
        self.n_layers = n_layers
        self.layers = nn.ModuleList(
            [GemmaSlidingLayer(max_ctx) for _ in range(n_layers)])
        for i in range(n_layers):
            self.register_buffer(
                f"k_cache_{i}",
                torch.zeros(1, SLD_N_KV, max_ctx, SLD_D_HEAD,
                            dtype=torch.float16))
            self.register_buffer(
                f"v_cache_{i}",
                torch.zeros(1, SLD_N_KV, max_ctx, SLD_D_HEAD,
                            dtype=torch.float16))

    def forward(self, x, cos, sin, attn_mask, kv_write_mask):
        k_last = v_last = None
        for i, layer in enumerate(self.layers):
            kc = getattr(self, f"k_cache_{i}")
            vc = getattr(self, f"v_cache_{i}")
            x, k_last, v_last = layer(x, cos, sin, kc, vc,
                                       attn_mask, kv_write_mask)
            kc[:] = k_last
            vc[:] = v_last
        return x.view(1, D_MODEL), k_last, v_last


class GemmaMixedStackWrap(nn.Module):
    """Heterogeneous stack: each position is sliding or global per layer_types.

    Forward inputs:
      x, cos_s, sin_s, cos_g, sin_g, attn_mask, kv_write_mask
    Sliding layers consume (cos_s, sin_s) [dim=256], global layers consume
    (cos_g, sin_g) [dim=512, full rotation over head_dim=512].
    Per-layer KV state buffers have type-appropriate shapes.
    """
    def __init__(self, max_ctx: int, layer_types: list[str]):
        super().__init__()
        self.max_ctx = max_ctx
        self.layer_types = list(layer_types)
        self.n_layers = len(layer_types)
        layers = []
        for i, t in enumerate(layer_types):
            if t == "global":
                layers.append(GemmaGlobalLayer(max_ctx))
            elif t == "sliding":
                layers.append(GemmaSlidingLayer(max_ctx))
            else:
                raise ValueError(f"unknown layer type {t!r} at {i}")
        self.layers = nn.ModuleList(layers)
        for i, t in enumerate(layer_types):
            shp = _state_shape(t == "global", max_ctx)
            self.register_buffer(
                f"k_cache_{i}", torch.zeros(*shp, dtype=torch.float16))
            self.register_buffer(
                f"v_cache_{i}", torch.zeros(*shp, dtype=torch.float16))

    def forward(self, x, cos_s, sin_s, cos_g, sin_g, attn_mask, kv_write_mask):
        hidden, k_last, v_last, _ = self._run_layers(
            x, cos_s, sin_s, cos_g, sin_g, attn_mask, kv_write_mask)
        return hidden, k_last, v_last

    def _run_layers(self, x, cos_s, sin_s, cos_g, sin_g, attn_mask,
                    kv_write_mask, tap_local_ends: tuple[int, ...] = (),
                    internal_taps_by_layer: dict[int, tuple[str, ...]] | None = None):
        k_last = v_last = None
        tap_end_set = set(tap_local_ends)
        internal_taps_by_layer = internal_taps_by_layer or {}
        taps = []
        for i, layer in enumerate(self.layers):
            kc = getattr(self, f"k_cache_{i}")
            vc = getattr(self, f"v_cache_{i}")
            layer_tap_names = internal_taps_by_layer.get(i, ())
            if self.layer_types[i] == "global":
                if layer_tap_names:
                    x, k_last, v_last, layer_taps = layer.forward_internal_taps(
                        x, cos_g, sin_g, kc, vc, attn_mask, kv_write_mask,
                        tap_names=layer_tap_names)
                    taps.extend(layer_taps)
                else:
                    x, k_last, v_last = layer(x, cos_g, sin_g, kc, vc,
                                               attn_mask, kv_write_mask)
            else:
                if layer_tap_names:
                    x, k_last, v_last, layer_taps = layer.forward_internal_taps(
                        x, cos_s, sin_s, kc, vc, attn_mask, kv_write_mask,
                        tap_names=layer_tap_names)
                    taps.extend(layer_taps)
                else:
                    x, k_last, v_last = layer(x, cos_s, sin_s, kc, vc,
                                               attn_mask, kv_write_mask)
            # CoreML state write-back (lowers to coreml_update_state).
            kc[:] = k_last
            vc[:] = v_last
            if (i + 1) in tap_end_set:
                taps.append(x.view(1, D_MODEL))
        return x.view(1, D_MODEL), k_last, v_last, taps

    def forward_hidden_taps(self, x, cos_s, sin_s, cos_g, sin_g, attn_mask,
                            kv_write_mask, tap_local_ends: tuple[int, ...],
                            internal_taps_by_layer: dict[int, tuple[str, ...]] | None = None):
        hidden, _k_last, _v_last, taps = self._run_layers(
            x, cos_s, sin_s, cos_g, sin_g, attn_mask, kv_write_mask,
            tap_local_ends=tap_local_ends,
            internal_taps_by_layer=internal_taps_by_layer)
        return (hidden, *taps)


# ------------------- weight loader ----------------------------------------

def _load_layer_weights(layer, npz_path: Path):
    """Load packed REAP weights into a single Gemma layer (sliding or global)."""
    data = np.load(npz_path)
    is_global_npz = bool(data["is_global"]) if "is_global" in data.files else False
    is_global_mod = bool(getattr(layer, "is_global", False))
    assert is_global_npz == is_global_mod, (
        f"layer/npz type mismatch: module.is_global={is_global_mod} "
        f"npz.is_global={is_global_npz} ({npz_path.name})")

    def _set(t: nn.Parameter, arr: np.ndarray):
        a = torch.from_numpy(arr.astype(np.float16))
        assert tuple(a.shape) == tuple(t.shape), \
            f"shape mismatch: dest={tuple(t.shape)} src={tuple(a.shape)}"
        t.data.copy_(a)

    def _set_ln(mod: RMSNorm, arr: np.ndarray):
        # Stored γ; module applies (1.0 + weight). Weight = γ - 1.
        _set(mod.weight, (arr.astype(np.float32) - 1.0).astype(np.float16))

    _set_ln(layer.input_ln,      data["input_ln"])
    _set_ln(layer.post_attn_ln,  data["post_attn_ln"])
    _set_ln(layer.pre_ffn_ln,    data["pre_ffn_ln"])
    _set_ln(layer.post_ffn_ln,   data["post_ffn_ln"])
    _set_ln(layer.pre_ffn_ln_2,  data["pre_ffn_ln_2"])
    _set_ln(layer.post_ffn_ln_1, data["post_ffn_ln_1"])
    _set_ln(layer.post_ffn_ln_2, data["post_ffn_ln_2"])

    _set(layer.attn.q_proj.weight, data["q_proj"])
    _set(layer.attn.k_proj.weight, data["k_proj"])
    if not is_global_mod:
        _set(layer.attn.v_proj.weight, data["v_proj"])
    _set(layer.attn.o_proj.weight, data["o_proj"])
    _set_ln(layer.attn.q_norm, data["q_norm"])
    _set_ln(layer.attn.k_norm, data["k_norm"])

    _set(layer.mlp_dense.gate.weight, data["mlp_gate"])
    _set(layer.mlp_dense.up.weight,   data["mlp_up"])
    _set(layer.mlp_dense.down.weight, data["mlp_down"])

    for p in range(N_PACKS):
        _set(layer.packs[p].gate.weight, data["gate_packs"][p])
        _set(layer.packs[p].up.weight,   data["up_packs"][p])
        _set(layer.packs[p].down.weight, data["down_packs"][p])

    _set(layer.router.proj.weight,      data["router_proj"])
    _set(layer.router.scale,            data["router_scale"])
    _set(layer.router.per_expert_scale, data["router_per_expert_scale"])

    layer.layer_scalar.data.copy_(
        torch.from_numpy(data["layer_scalar"].astype(np.float16)))


def _ordered_internal_tap_names(raw_names: list[str]) -> tuple[str, ...]:
    unique_names = set(raw_names)
    unknown = sorted(unique_names - DEBUG_INTERNAL_TAP_SET)
    if unknown:
        raise ValueError(
            f"unknown debug internal tap(s) {unknown}; "
            f"expected subset of {list(DEBUG_INTERNAL_TAP_ORDER)}")
    return tuple(name for name in DEBUG_INTERNAL_TAP_ORDER if name in unique_names)


def _load_weights(model: GemmaLayer1Wrap, npz_path: Path):
    _load_layer_weights(model.layer0, npz_path)


def _load_stack_weights(model: GemmaStackWrap, npz_paths: list[Path]):
    assert len(npz_paths) == model.n_layers, \
        f"need {model.n_layers} npz paths, got {len(npz_paths)}"
    for i, p in enumerate(npz_paths):
        print(f"  layer {i}: loading {p.name}")
        _load_layer_weights(model.layers[i], p)


# ------------------- driver -----------------------------------------------

def cmd_layer1(args):
    max_ctx = args.max_ctx
    print(f"=== T4.1.1 layer1 (real REAP weights, max_ctx={max_ctx}) ===")
    print(f"  PACK_G={PACK_G}, N_PACKS={N_PACKS}, KEEP={N_EXPERTS_KEEP}")

    model = GemmaLayer1Wrap(max_ctx)
    model.half().eval()

    npz_path = OUT_DIR / "gemma_layer0_packed.npz"
    print(f"  loading weights from {npz_path}")
    _load_weights(model, npz_path)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,} (fp16) "
          f"= {n_params * 2 / 1e6:.1f} MB pre-quant")

    dh = SLD_D_HEAD
    x_ex = torch.randn(1, 1, D_MODEL, dtype=torch.float16) * 0.1
    cos_ex = torch.randn(1, 1, dh, dtype=torch.float16)
    sin_ex = torch.randn(1, 1, dh, dtype=torch.float16)
    mask_ex = torch.full((1, 1, 1, max_ctx), -1e4, dtype=torch.float16)
    mask_ex[..., 0] = 0.0
    wmask_ex = torch.zeros(1, 1, max_ctx, 1, dtype=torch.float16)
    wmask_ex[0, 0, 0, 0] = 1.0

    print("  forward smoke...")
    with torch.no_grad():
        out, k_new, v_new = model(x_ex, cos_ex, sin_ex, mask_ex, wmask_ex)
    print(f"  out: {out.shape} {out.dtype}, "
          f"finite={torch.isfinite(out).all().item()}")

    print("  tracing...")
    traced = torch.jit.trace(model, (x_ex, cos_ex, sin_ex, mask_ex, wmask_ex))

    with torch.no_grad():
        model.k_cache_0.zero_()
        model.v_cache_0.zero_()

    print("  converting to CoreML (fp16, CPU+ANE)...")
    t0 = time.perf_counter()
    ct_inputs = [
        ct.TensorType(name="x",             shape=(1, 1, D_MODEL),    dtype=np.float16),
        ct.TensorType(name="cos",           shape=(1, 1, dh),         dtype=np.float16),
        ct.TensorType(name="sin",           shape=(1, 1, dh),         dtype=np.float16),
        ct.TensorType(name="attn_mask",     shape=(1, 1, 1, max_ctx), dtype=np.float16),
        ct.TensorType(name="kv_write_mask", shape=(1, 1, max_ctx, 1), dtype=np.float16),
    ]
    ct_outputs = [
        ct.TensorType(name="hidden", dtype=np.float16),
        ct.TensorType(name="k_new",  dtype=np.float16),
        ct.TensorType(name="v_new",  dtype=np.float16),
    ]
    ct_states = [
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(1, SLD_N_KV, max_ctx, SLD_D_HEAD), dtype=np.float16),
            name="k_cache_0"),
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(1, SLD_N_KV, max_ctx, SLD_D_HEAD), dtype=np.float16),
            name="v_cache_0"),
    ]
    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=ct_outputs,
        states=ct_states,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    print(f"  convert wall: {time.perf_counter() - t0:.1f}s")

    print(f"  INT4 weight quantize (per_block, block_size={INT4_BLOCK_SIZE})...")
    t0 = time.perf_counter()
    mlmodel = cto.linear_quantize_weights(
        mlmodel, config=cto.OptimizationConfig(
            global_config=cto.OpLinearQuantizerConfig(
                mode="linear_symmetric", dtype="int4",
                granularity="per_block", block_size=INT4_BLOCK_SIZE,
                weight_threshold=0)))
    print(f"  quant wall: {time.perf_counter() - t0:.1f}s")

    out_pkg = OUT_DIR / "gemma4_layer1_real.mlpackage"
    if out_pkg.exists():
        shutil.rmtree(out_pkg)
    mlmodel.save(str(out_pkg))
    pkg_mb = sum(f.stat().st_size for f in out_pkg.rglob("*") if f.is_file()) / 1e6
    print(f"  saved → {out_pkg}  ({pkg_mb:.1f} MB)")

    compiled = OUT_DIR / "gemma4_layer1_real.mlmodelc"
    if compiled.exists():
        shutil.rmtree(compiled)
    print("  coremlcompiler...")
    t0 = time.perf_counter()
    compiled = Path(ct.utils.compile_model(str(out_pkg), str(compiled)))
    print(f"  compile wall: {time.perf_counter() - t0:.1f}s")
    print(f"  compiled → {compiled}")
    print("\n# T4.1.1 LAYER1: PASS (compiled)")


def cmd_stack(args):
    max_ctx  = args.max_ctx
    n_layers = args.n_layers
    clone    = args.clone
    print(f"=== T4.1.3 stack N={n_layers} (max_ctx={max_ctx}, clone={clone}) ===")

    model = GemmaStackWrap(max_ctx, n_layers)
    model.half().eval()

    if clone:
        npz_paths = [OUT_DIR / "gemma_layer0_packed.npz"] * n_layers
    else:
        npz_paths = [OUT_DIR / f"gemma_layer{i}_packed.npz"
                     for i in range(n_layers)]
        for p in npz_paths:
            assert p.exists(), f"missing pack: {p}"

    print(f"  loading {n_layers} layer(s) of weights...")
    _load_stack_weights(model, npz_paths)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,} (fp16) "
          f"= {n_params * 2 / 1e6:.1f} MB pre-quant")

    dh = SLD_D_HEAD
    x_ex = torch.randn(1, 1, D_MODEL, dtype=torch.float16) * 0.1
    cos_ex = torch.randn(1, 1, dh, dtype=torch.float16)
    sin_ex = torch.randn(1, 1, dh, dtype=torch.float16)
    mask_ex = torch.full((1, 1, 1, max_ctx), -1e4, dtype=torch.float16)
    mask_ex[..., 0] = 0.0
    wmask_ex = torch.zeros(1, 1, max_ctx, 1, dtype=torch.float16)
    wmask_ex[0, 0, 0, 0] = 1.0

    print("  forward smoke...")
    with torch.no_grad():
        out, k_new, v_new = model(x_ex, cos_ex, sin_ex, mask_ex, wmask_ex)
    print(f"  out: {out.shape} {out.dtype}, "
          f"finite={torch.isfinite(out).all().item()}")

    print("  tracing...")
    traced = torch.jit.trace(model, (x_ex, cos_ex, sin_ex, mask_ex, wmask_ex))
    with torch.no_grad():
        for i in range(n_layers):
            getattr(model, f"k_cache_{i}").zero_()
            getattr(model, f"v_cache_{i}").zero_()

    print("  converting to CoreML (fp16, CPU+ANE)...")
    t0 = time.perf_counter()
    ct_inputs = [
        ct.TensorType(name="x",             shape=(1, 1, D_MODEL),    dtype=np.float16),
        ct.TensorType(name="cos",           shape=(1, 1, dh),         dtype=np.float16),
        ct.TensorType(name="sin",           shape=(1, 1, dh),         dtype=np.float16),
        ct.TensorType(name="attn_mask",     shape=(1, 1, 1, max_ctx), dtype=np.float16),
        ct.TensorType(name="kv_write_mask", shape=(1, 1, max_ctx, 1), dtype=np.float16),
    ]
    ct_outputs = [
        ct.TensorType(name="hidden", dtype=np.float16),
        ct.TensorType(name="k_new",  dtype=np.float16),
        ct.TensorType(name="v_new",  dtype=np.float16),
    ]
    ct_states = []
    for i in range(n_layers):
        ct_states.append(ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(1, SLD_N_KV, max_ctx, SLD_D_HEAD), dtype=np.float16),
            name=f"k_cache_{i}"))
        ct_states.append(ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(1, SLD_N_KV, max_ctx, SLD_D_HEAD), dtype=np.float16),
            name=f"v_cache_{i}"))

    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=ct_outputs,
        states=ct_states,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    print(f"  convert wall: {time.perf_counter() - t0:.1f}s")

    print(f"  INT4 weight quantize (per_block, block_size={INT4_BLOCK_SIZE})...")
    t0 = time.perf_counter()
    mlmodel = cto.linear_quantize_weights(
        mlmodel, config=cto.OptimizationConfig(
            global_config=cto.OpLinearQuantizerConfig(
                mode="linear_symmetric", dtype="int4",
                granularity="per_block", block_size=INT4_BLOCK_SIZE,
                weight_threshold=0)))
    print(f"  quant wall: {time.perf_counter() - t0:.1f}s")

    tag = "clone" if clone else "real"
    out_pkg = OUT_DIR / f"gemma4_stack{n_layers}_{tag}.mlpackage"
    if out_pkg.exists():
        shutil.rmtree(out_pkg)
    mlmodel.save(str(out_pkg))
    pkg_mb = sum(f.stat().st_size for f in out_pkg.rglob("*") if f.is_file()) / 1e6
    print(f"  saved → {out_pkg}  ({pkg_mb:.1f} MB)")

    compiled = OUT_DIR / f"gemma4_stack{n_layers}_{tag}.mlmodelc"
    if compiled.exists():
        shutil.rmtree(compiled)
    print("  coremlcompiler...")
    t0 = time.perf_counter()
    compiled = Path(ct.utils.compile_model(str(out_pkg), str(compiled)))
    print(f"  compile wall: {time.perf_counter() - t0:.1f}s")
    print(f"  compiled → {compiled}")
    print(f"\n# T4.1.3 STACK{n_layers}_{tag.upper()}: PASS (compiled)")


def _layer_types_from_config(n: int) -> list[str]:
    """Return canonical 'sliding'/'global' types for the first n layers
    derived from models/gemma-4-26b-a4b/config.json."""
    import json
    cfg = json.loads((Path("models/gemma-4-26b-a4b") / "config.json").read_text())
    raw = cfg["text_config"]["layer_types"][:n]
    out = []
    for t in raw:
        if t == "full_attention":
            out.append("global")
        elif t == "sliding_attention":
            out.append("sliding")
        else:
            raise ValueError(f"unexpected layer_type {t!r}")
    return out


def cmd_mixed(args):
    max_ctx  = args.max_ctx
    n_layers = args.n_layers
    tag_suffix = getattr(args, "tag_suffix", None)
    no_quant = getattr(args, "no_quant", False)
    # Optional shard range [layer_start, layer_end). If unset, full stack.
    layer_start = getattr(args, "layer_start", 0) or 0
    layer_end   = getattr(args, "layer_end", None) or n_layers
    is_shard = (layer_start != 0) or (layer_end != n_layers)
    raw_debug_hidden_boundaries = getattr(args, "debug_hidden_boundaries", None)
    raw_debug_internal_taps = getattr(args, "debug_internal_taps", None)
    full_layer_types = _layer_types_from_config(n_layers)
    layer_types = full_layer_types[layer_start:layer_end]
    shard_n = len(layer_types)
    debug_hidden_boundaries = []
    internal_taps_by_layer = {}
    if raw_debug_hidden_boundaries:
        debug_hidden_boundaries = sorted({
            int(part.strip())
            for part in raw_debug_hidden_boundaries.split(",")
            if part.strip()
        })
    if raw_debug_internal_taps:
        raw_internal_taps_by_layer = {}
        for raw_spec in raw_debug_internal_taps.split(","):
            spec = raw_spec.strip()
            if not spec:
                continue
            layer_text, sep, tap_name = spec.partition(":")
            if not sep:
                raise ValueError(
                    "debug internal tap specs must look like LAYER:TAP, "
                    f"got {spec!r}")
            layer_index = int(layer_text.strip())
            tap_name = tap_name.strip()
            if not (layer_start <= layer_index < layer_end):
                raise ValueError(
                    f"debug internal tap layer {layer_index} must satisfy "
                    f"{layer_start} <= layer < {layer_end}")
            local_index = layer_index - layer_start
            raw_internal_taps_by_layer.setdefault(local_index, []).append(tap_name)
        internal_taps_by_layer = {
            local_index: _ordered_internal_tap_names(tap_names)
            for local_index, tap_names in raw_internal_taps_by_layer.items()
        }
    for boundary in debug_hidden_boundaries:
        if not (layer_start < boundary < layer_end):
            raise ValueError(
                f"debug hidden boundary {boundary} must satisfy "
                f"{layer_start} < boundary < {layer_end}")
    tap_local_ends = tuple(boundary - layer_start
                           for boundary in debug_hidden_boundaries)
    tap_end_set = set(tap_local_ends)
    tap_output_names = []
    for local_index in range(shard_n):
        global_layer = layer_start + local_index
        for tap_name in internal_taps_by_layer.get(local_index, ()): 
            tap_output_names.append(f"hidden_l{global_layer}_{tap_name}")
        if (local_index + 1) in tap_end_set:
            tap_output_names.append(f"hidden_l{global_layer + 1}")
    tag_base = (f"shard{layer_start}_{layer_end}" if is_shard
                else f"mixed{n_layers}")
    tag = f"{tag_base}_{tag_suffix}" if tag_suffix else tag_base
    print(f"=== T4.1.3c {tag} N={shard_n}/{n_layers} (max_ctx={max_ctx}) ===")
    print(f"  global_layer_range: [{layer_start}, {layer_end})")
    print(f"  layer_types (shard): {layer_types}")
    n_global = sum(1 for t in layer_types if t == "global")
    print(f"  sliding={shard_n-n_global}  global={n_global}")
    if tap_output_names:
        print(f"  debug hidden taps: {tap_output_names}")

    model = GemmaMixedStackWrap(max_ctx, layer_types)
    model.half().eval()

    npz_paths = [OUT_DIR / f"gemma_layer{i}_packed.npz"
                 for i in range(layer_start, layer_end)]
    for p in npz_paths:
        assert p.exists(), f"missing pack: {p}"
    print(f"  loading {shard_n} layer packs (global {layer_start}..{layer_end-1})...")
    for local_i, pth in enumerate(npz_paths):
        gi = layer_start + local_i
        print(f"  layer {gi} ({layer_types[local_i]}): {pth.name}")
        _load_layer_weights(model.layers[local_i], pth)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,} (fp16) "
          f"= {n_params * 2 / 1e6:.1f} MB pre-quant")

    dh_s = SLD_D_HEAD
    dh_g_rot = GLB_ROT_DIM
    x_ex = torch.randn(1, 1, D_MODEL, dtype=torch.float16) * 0.1
    cos_s = torch.randn(1, 1, dh_s,    dtype=torch.float16)
    sin_s = torch.randn(1, 1, dh_s,    dtype=torch.float16)
    cos_g = torch.randn(1, 1, dh_g_rot, dtype=torch.float16)
    sin_g = torch.randn(1, 1, dh_g_rot, dtype=torch.float16)
    mask_ex = torch.full((1, 1, 1, max_ctx), -1e4, dtype=torch.float16)
    mask_ex[..., 0] = 0.0
    wmask_ex = torch.zeros(1, 1, max_ctx, 1, dtype=torch.float16)
    wmask_ex[0, 0, 0, 0] = 1.0

    print("  forward smoke...")
    with torch.no_grad():
        out, k_new, v_new = model(x_ex, cos_s, sin_s, cos_g, sin_g,
                                  mask_ex, wmask_ex)
    print(f"  out: {out.shape} {out.dtype}, "
          f"finite={torch.isfinite(out).all().item()}")

    print("  tracing (hidden output for shard wrap)...")
    # Wrap to drop redundant k_new/v_new outputs: with state-write enabled the
    # cache lives in state buffers, so emitting them as outputs creates two
    # paths to the same tensor and trips MIL→proto serialization.
    class _HiddenOnly(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x, cs, ss, cg, sg, am, wm):
            h, _k, _v = self.m(x, cs, ss, cg, sg, am, wm)
            return h

    class _HiddenWithTaps(nn.Module):
        def __init__(self, m, tap_local_ends, internal_taps_by_layer):
            super().__init__()
            self.m = m
            self.tap_local_ends = tuple(tap_local_ends)
            self.internal_taps_by_layer = dict(internal_taps_by_layer)

        def forward(self, x, cs, ss, cg, sg, am, wm):
            return self.m.forward_hidden_taps(
                x, cs, ss, cg, sg, am, wm,
                tap_local_ends=self.tap_local_ends,
                internal_taps_by_layer=self.internal_taps_by_layer)

    wrap = (_HiddenWithTaps(model, tap_local_ends, internal_taps_by_layer) if tap_output_names
            else _HiddenOnly(model)).eval()
    traced = torch.jit.trace(
        wrap,
        (x_ex, cos_s, sin_s, cos_g, sin_g, mask_ex, wmask_ex))
    with torch.no_grad():
        for i in range(shard_n):
            getattr(model, f"k_cache_{i}").zero_()
            getattr(model, f"v_cache_{i}").zero_()

    print("  converting to CoreML (fp16, CPU+ANE)...")
    t0 = time.perf_counter()
    ct_inputs = [
        ct.TensorType(name="x",             shape=(1, 1, D_MODEL),     dtype=np.float16),
        ct.TensorType(name="cos_s",         shape=(1, 1, dh_s),        dtype=np.float16),
        ct.TensorType(name="sin_s",         shape=(1, 1, dh_s),        dtype=np.float16),
        ct.TensorType(name="cos_g",         shape=(1, 1, dh_g_rot),    dtype=np.float16),
        ct.TensorType(name="sin_g",         shape=(1, 1, dh_g_rot),    dtype=np.float16),
        ct.TensorType(name="attn_mask",     shape=(1, 1, 1, max_ctx),  dtype=np.float16),
        ct.TensorType(name="kv_write_mask", shape=(1, 1, max_ctx, 1),  dtype=np.float16),
    ]
    ct_outputs = [
        ct.TensorType(name="hidden", dtype=np.float16),
    ]
    ct_outputs.extend(
        ct.TensorType(name=name, dtype=np.float16)
        for name in tap_output_names)
    ct_states = []
    for i, t in enumerate(layer_types):
        shp = _state_shape(t == "global", max_ctx)
        ct_states.append(ct.StateType(
            wrapped_type=ct.TensorType(shape=shp, dtype=np.float16),
            name=f"m.k_cache_{i}"))
        ct_states.append(ct.StateType(
            wrapped_type=ct.TensorType(shape=shp, dtype=np.float16),
            name=f"m.v_cache_{i}"))

    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=ct_outputs,
        states=ct_states,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    print(f"  convert wall: {time.perf_counter() - t0:.1f}s")

    if no_quant:
        print("  SKIPPING INT4 quantization (--no-quant)")
    else:
        print(f"  INT4 weight quantize (per_block, block_size={INT4_BLOCK_SIZE})...")
        t0 = time.perf_counter()
        mlmodel = cto.linear_quantize_weights(
            mlmodel, config=cto.OptimizationConfig(
                global_config=cto.OpLinearQuantizerConfig(
                    mode="linear_symmetric", dtype="int4",
                    granularity="per_block", block_size=INT4_BLOCK_SIZE,
                    weight_threshold=0)))
        print(f"  quant wall: {time.perf_counter() - t0:.1f}s")

    out_pkg = OUT_DIR / f"gemma4_{tag}_real.mlpackage"
    if out_pkg.exists():
        shutil.rmtree(out_pkg)
    mlmodel.save(str(out_pkg))
    pkg_mb = sum(f.stat().st_size for f in out_pkg.rglob("*") if f.is_file()) / 1e6
    print(f"  saved \u2192 {out_pkg}  ({pkg_mb:.1f} MB)")

    compiled = OUT_DIR / f"gemma4_{tag}_real.mlmodelc"
    if compiled.exists():
        shutil.rmtree(compiled)
    print("  coremlcompiler...")
    t0 = time.perf_counter()
    compiled = Path(ct.utils.compile_model(str(out_pkg), str(compiled)))
    print(f"  compile wall: {time.perf_counter() - t0:.1f}s")
    print(f"  compiled \u2192 {compiled}")
    print(f"\n# T4.1.3c {tag.upper()}: PASS (compiled)")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("layer1")
    pl.add_argument("--max-ctx", type=int, default=1024)
    pl.set_defaults(func=cmd_layer1)

    ps = sub.add_parser("stack")
    ps.add_argument("--n-layers", type=int, required=True)
    ps.add_argument("--max-ctx", type=int, default=1024)
    ps.add_argument("--clone", action="store_true",
                    help="reuse layer0 npz for all N layers (cheap probe)")
    ps.set_defaults(func=cmd_stack)

    pm = sub.add_parser("mixed",
        help="heterogeneous stack (sliding+global) using config.json layer_types")
    pm.add_argument("--n-layers", type=int, required=True)
    pm.add_argument("--max-ctx", type=int, default=1024)
    pm.add_argument("--layer-start", type=int, default=0,
                    help="shard start (inclusive); default 0 = full stack")
    pm.add_argument("--layer-end", type=int, default=None,
                    help="shard end (exclusive); default = n-layers")
    pm.add_argument("--debug-hidden-boundaries", type=str, default=None,
                    help=("comma-separated global hidden boundary ends to emit "
                          "as extra outputs, e.g. 23,24,25 for shard [22,30)"))
    pm.add_argument("--debug-internal-taps", type=str, default=None,
                    help=("comma-separated LAYER:TAP specs for extra internal taps, "
                          "e.g. 29:post_attn,29:ffn_out"))
    pm.add_argument("--tag-suffix", type=str, default=None,
                    help="optional suffix appended to the saved artifact tag")
    pm.add_argument("--no-quant", action="store_true",
                    help="skip INT4 quantization (keep FP16 weights)")
    pm.set_defaults(func=cmd_mixed)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
