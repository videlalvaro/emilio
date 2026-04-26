"""gemma_to_ane.py — T4.1 of GEMMA_ANE_RESEARCH.md.

CoreML conversion of one Gemma-4-26B-A4B sliding layer for the ANE.

Status:
  T4.1.0 (skeleton with random weights, 1L)             — PASS, JOURNAL 2026-04-22
  T4.1.1 (real weights, all 128 experts, proportional RoPE)  — THIS FILE

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
  dense_w = scatter(topi, topw, dim=128) → split into 8 packs of 16
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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEBUG_INTERNAL_TAP_ORDER = ("attn_delta", "post_attn", "ffn_out")
DEBUG_INTERNAL_TAP_SET = set(DEBUG_INTERNAL_TAP_ORDER)
import coremltools as ct
import coremltools.optimize.coreml as cto

OUT_DIR = Path("python/moe/out"); OUT_DIR.mkdir(parents=True, exist_ok=True)

D_MODEL          = 2816
D_FFN            = 704
D_DENSE          = 2112
N_HEADS          = 16
N_EXPERTS_KEEP   = 128
TOP_K            = 8
PACK_G           = 16
N_PACKS          = N_EXPERTS_KEEP // PACK_G
RMS_EPS          = 1e-6

SLD_N_KV   = 8
SLD_D_HEAD = 256
SLD_WIN    = 1024

GLB_N_KV     = 2
GLB_D_HEAD   = 512
GLB_ROT_DIM  = 128   # partial_rotary_factor=0.25 × head_dim=512 → 128 rotated dims

# Gemma's tuned default remains INT4 grouped with block size 8. Keep that
# default, but expose the quantization shape so other model families can reuse
# this converter with different bit-width and grouping choices.
DEFAULT_QUANT_BITS = 4
DEFAULT_INT4_GROUP_SIZE = 8


@dataclass(frozen=True)
class QuantConfig:
    bits: int
    group_size: int
    granularity: str = "per_tensor"   # per_tensor | per_channel | per_block
    weight_threshold: int = 0         # skip linear ops with fewer elements

    @property
    def enabled(self) -> bool:
        return self.bits != 0

    def describe(self) -> str:
        if self.bits == 0:
            return "fp16 weights"
        if self.bits == 4:
            return f"int4 per_block (block_size={self.group_size})"
        if self.granularity == "per_block":
            return f"int8 per_block (block_size={self.group_size})"
        return f"int8 {self.granularity}"


def _quant_config_from_args(args) -> QuantConfig:
    bits = getattr(args, "quant_bits", DEFAULT_QUANT_BITS)
    group_size = getattr(args, "group_size", DEFAULT_INT4_GROUP_SIZE)
    granularity = getattr(args, "granularity", "per_tensor")
    if bits not in (0, 4, 8):
        raise ValueError(f"quant_bits must be 0, 4, or 8 (got {bits})")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive (got {group_size})")
    return QuantConfig(bits=bits, group_size=group_size, granularity=granularity)


def _add_quant_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--quant-bits",
        type=int,
        choices=[0, 4, 8],
        default=DEFAULT_QUANT_BITS,
        dest="quant_bits",
        help=(
            "Weight quantization bits (0=fp16, 4=int4 grouped, 8=int8). "
            f"Default {DEFAULT_QUANT_BITS}."
        ),
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=DEFAULT_INT4_GROUP_SIZE,
        dest="group_size",
        help=(
            "Block/group size for grouped quantization. Used for int4. "
            f"Default {DEFAULT_INT4_GROUP_SIZE}."
        ),
    )
    parser.add_argument(
        "--granularity",
        type=str,
        choices=["per_tensor", "per_channel", "per_block"],
        default="per_tensor",
        help=(
            "INT8 quantization granularity. per_channel gives better accuracy "
            "than per_tensor at minimal size cost. Default per_tensor."
        ),
    )
    parser.add_argument(
        "--no-quant",
        action="store_const",
        const=0,
        dest="quant_bits",
        help="Deprecated alias for --quant-bits 0 (keep fp16 weights)",
    )


def _quant_artifact_suffix(quant_config: QuantConfig) -> str:
    if (quant_config.bits == DEFAULT_QUANT_BITS
            and quant_config.group_size == DEFAULT_INT4_GROUP_SIZE):
        return ""
    if quant_config.bits == 0:
        return "_fp16"
    suffix = f"_q{quant_config.bits}"
    if quant_config.bits == 8 and quant_config.granularity == "per_channel":
        suffix += "c"  # q8c = INT8 per-channel
    if quant_config.bits == 4 and quant_config.group_size != DEFAULT_INT4_GROUP_SIZE:
        suffix += f"g{quant_config.group_size}"
    return suffix


def _apply_weight_quantization(mlmodel, quant_config: QuantConfig):
    if quant_config.bits == 0:
        print("  SKIPPING weight quantization (--quant-bits 0 / --no-quant)")
        return mlmodel

    wt = quant_config.weight_threshold

    if quant_config.bits == 8 and quant_config.granularity == "per_block":
        print(f"  INT8 weight quantize (per_block, block_size={quant_config.group_size})...")
        op_config = cto.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int8",
            granularity="per_block",
            block_size=quant_config.group_size,
            weight_threshold=wt,
        )
    elif quant_config.bits == 8:
        gran = quant_config.granularity
        print(f"  INT8 weight quantize ({gran} symmetric)...")
        op_config = cto.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int8",
            granularity=gran,
            weight_threshold=wt,
        )
    else:
        print(
            "  INT4 weight quantize "
            f"(per_block, block_size={quant_config.group_size})..."
        )
        op_config = cto.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int4",
            granularity="per_block",
            block_size=quant_config.group_size,
            weight_threshold=wt,
        )

    t0 = time.perf_counter()
    # Use op_type_configs (not global_config) to restrict quantization to
    # linear/matmul/conv ops only.  global_config also quantizes standalone
    # constant tensors (e.g. the MoE routing range [0..63] feeding 'equal'),
    # which corrupts non-weight constants.
    mlmodel = cto.linear_quantize_weights(
        mlmodel,
        config=cto.OptimizationConfig(op_type_configs={
            "linear": op_config,
            "matmul": op_config,
            "conv": op_config,
        }),
    )
    print(f"  quant wall: {time.perf_counter() - t0:.1f}s")
    return mlmodel


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
        orig_dtype = x.dtype
        h = x.float()
        v = h.pow(2).mean(dim=-1, keepdim=True)
        h = h * torch.rsqrt(v + self.eps)
        if self.with_scale:
            h = h * (1.0 + self.weight.float())
        return h.to(orig_dtype)


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
    """Full-attention layer (k_eq_v=True, partial RoPE 128/512, no v_proj).

    V is derived from the same projection as K (raw k_proj output), then runs
    through the unscaled RMSNorm (v_norm with γ=1) and is NOT rotated.
    K runs through k_norm (γ=γ_k) and partial RoPE (first 128 of 512 dims).
    Q runs through q_norm (γ=γ_q) and partial RoPE (first 128 of 512 dims).
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
        q = apply_rope_partial(q, cos, sin, GLB_ROT_DIM)  # 128/512 dims rotated
        k = apply_rope_partial(k, cos, sin, GLB_ROT_DIM)  # 128/512 dims rotated
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
        # Post-activation scale for gate output. Set by fuse_norm_scales_for_ane()
        # to prevent gate*up fp16 overflow on ANE.  Default 1.0 = no-op.
        self.register_buffer(
            "_gate_scale", torch.ones(1, dtype=torch.float16))

    def forward(self, x):
        g = F.gelu(self.gate(x), approximate="tanh") * self._gate_scale
        u = self.up(x)
        h = (g * u).view(1, 1, PACK_G, D_FFN).reshape(PACK_G, D_FFN)
        out = self.down(h).view(PACK_G, PACK_G, D_MODEL)
        idx = torch.arange(PACK_G)
        return out[idx, idx].view(1, 1, PACK_G, D_MODEL)


class Router(nn.Module):
    """Top-K=8 router over 128 experts. Returns dense (1, 1, 128) routing vector."""
    def __init__(self):
        super().__init__()
        self.norm = RMSNorm(D_MODEL, with_scale=False)
        self.proj = nn.Linear(D_MODEL, N_EXPERTS_KEEP, bias=False)
        self.scale = nn.Parameter(torch.ones(D_MODEL, dtype=torch.float16))
        self.per_expert_scale = nn.Parameter(
            torch.ones(N_EXPERTS_KEEP, dtype=torch.float16))
        self._scalar_root = D_MODEL ** -0.5
        # Constant range [0..127] for ANE-friendly one-hot via == comparison.
        # one_hot and scatter_along_axis both lack ANE kernels; == (equal) has one
        # but only on fp16 — int32 equal falls back to CPU.
        # Integers 0..127 are exact in fp16 (mantissa covers up to 2048).
        # Stored as int16 so the weight quantizer won't touch it — INT8
        # quantization of a [0..127] range corrupts the exact integer values
        # needed for equality comparison (discovered the hard way).
        self.register_buffer(
            "_expert_range",
            torch.arange(N_EXPERTS_KEEP, dtype=torch.int16).view(1, 1, 1, -1),
        )

    def forward(self, x):
        h = self.norm(x) * self.scale * self._scalar_root
        scores = self.proj(h)
        prob = F.softmax(scores, dim=-1)
        topw, topi = torch.topk(prob, k=TOP_K, dim=-1)
        topw = topw / topw.sum(dim=-1, keepdim=True)
        topw = topw * self.per_expert_scale[topi]
        # Build dense routing vector via broadcast == (ANE-native 'equal' op).
        # Cast both to fp16 at forward time: indices from topk + range buffer.
        # The int16→fp16 cast keeps exact integers (0..63 fit in fp16 mantissa).
        expert_range_f = self._expert_range.to(torch.float16)
        mask = (topi.to(torch.float16).unsqueeze(-1) == expert_range_f).to(topw.dtype)
        return (topw.unsqueeze(-1) * mask).sum(dim=-2)


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

    @torch.no_grad()
    def fuse_norm_scales_for_ane(self):
        """Absorb RMSNorm scales into subsequent linear weights (exact).

        Prevents fp16 overflow on ANE for layers with large norm scales
        (e.g. pre_ffn_ln_2 max=175 on layer 5 vs 2.75 on layer 0).

        Fusions performed:
          1) pre_ffn_ln_2 scale → expert pack gate+up weights
          2) router.scale * scalar_root → router.proj weight
          3) Rescale up/down weights to prevent gate*up fp16 overflow
          4) pre_ffn_ln scale → dense MLP gate+up weights (INT8 friendliness)
        """
        import math

        # --- 1) pre_ffn_ln_2 scale → pack gate + up weights ---------------
        # pre2 = rmsnorm(x) * (1 + w)   →   pack.gate(pre2)
        # Fused: pack.gate.weight *= diag(1+w),  norm becomes scale-free
        s_pre2 = (1.0 + self.pre_ffn_ln_2.weight).float()       # (D_MODEL,)
        for pack in self.packs:
            # gate.weight: (PACK_G*D_FFN, D_MODEL)
            pack.gate.weight.mul_(s_pre2.unsqueeze(0).to(pack.gate.weight.dtype))
            pack.up.weight.mul_(s_pre2.unsqueeze(0).to(pack.up.weight.dtype))
        self.pre_ffn_ln_2.weight.zero_()                         # 1+0 = 1 → identity

        # --- 2) router scale → router.proj weight -------------------------
        # h = rmsnorm(x) * scale * root  →  proj(h)
        # Fused: proj.weight *= diag(scale * root), scale=1, root=1
        s_rtr = (self.router.scale * self.router._scalar_root).float()  # (D_MODEL,)
        self.router.proj.weight.mul_(s_rtr.unsqueeze(0).to(self.router.proj.weight.dtype))
        self.router.scale.fill_(1.0)
        self.router._scalar_root = 1.0

        # --- 3) Rescale gate-output / up / down to prevent fp16 overflow ----
        # out = down @ (gelu(gate(x)) * _gate_scale * up(x))
        # _gate_scale absorbs 1/S_g (exact, post-activation), up /= S_u,
        # down *= S_g * S_u.  Keeps gate*up AND down@h within fp16 range.
        FP16_SAFE = 8000.0    # target max for gate*up (well under 65504)
        fan_in = D_MODEL
        for pack in self.packs:
            g_est = pack.gate.weight.float().abs().max().item() * fan_in ** 0.5
            u_est = pack.up.weight.float().abs().max().item() * fan_in ** 0.5
            max_prod = g_est * u_est
            if max_prod > FP16_SAFE:
                S_total = max_prod / FP16_SAFE
                S_total = 2.0 ** math.ceil(math.log2(S_total))
                # Split evenly: S_g ≈ S_u ≈ sqrt(S_total), powers of 2
                half_exp = math.ceil(math.log2(S_total) / 2)
                S_g = 2.0 ** half_exp
                S_u = S_total / S_g
                if S_u < 1.0:
                    S_u, S_g = 1.0, S_total
                pack._gate_scale.fill_(1.0 / S_g)
                pack.up.weight.div_(S_u)
                pack.down.weight.mul_(S_g * S_u)

        # --- 4) pre_ffn_ln scale → dense MLP gate + up weights ------------
        # pre = rmsnorm(x) * (1 + w)  →  mlp_dense.gate(pre), mlp_dense.up(pre)
        # Fused: gate.weight *= diag(1+w), up.weight *= diag(1+w), norm = identity
        # This makes dense MLP weights larger → more INT8 quantization levels.
        s_pre = (1.0 + self.pre_ffn_ln.weight).float()           # (D_MODEL,)
        self.mlp_dense.gate.weight.mul_(s_pre.unsqueeze(0).to(self.mlp_dense.gate.weight.dtype))
        self.mlp_dense.up.weight.mul_(s_pre.unsqueeze(0).to(self.mlp_dense.up.weight.dtype))
        self.pre_ffn_ln.weight.zero_()                           # 1+0 = 1 → identity

    def forward_attn(self, x, cos, sin, k_cache, v_cache, attn_mask, kv_write_mask):
        residual = x
        h = self.input_ln(x)
        a, k_new, v_new = self.attn(h, cos, sin, k_cache, v_cache,
                                    attn_mask, kv_write_mask)
        x = residual + self.post_attn_ln(a)
        return x, k_new, v_new

    def forward_ffn(self, x):
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
        return x

    def forward_ffn_partial(self, x, pack_start: int, pack_end: int):
        """Run router + expert packs [pack_start, pack_end) only.

        Returns partial weighted MoE contribution (1, 1, D_MODEL).
        Router and pre_ffn_ln_2 are duplicated across sub-shards (cheap,
        <5 MB each) so each sub-shard is self-contained.
        """
        rw = self.router(x)                                       # (1, 1, 64)
        rw = rw.view(1, 1, N_PACKS, PACK_G)                       # (1, 1, P, G)
        pre2 = self.pre_ffn_ln_2(x)
        moe = torch.zeros(1, 1, D_MODEL, dtype=x.dtype)
        for p in range(pack_start, pack_end):
            per_e = self.packs[p](pre2)                           # (1, 1, G, D_MODEL)
            w_p = rw[:, :, p, :].unsqueeze(-1)                    # (1, 1, G, 1)
            moe = moe + (per_e * w_p).sum(dim=-2)                 # (1, 1, D_MODEL)
        return moe

    def forward_ffn_combine(self, x, partial_moe):
        """Combine pre-summed partial MoE output with dense path + norms.

        Takes x (residual) and partial_moe (sum of all sub-shard outputs).
        Runs the dense MLP, norms, residual connection, and layer scalar.
        """
        residual = x
        pre = self.pre_ffn_ln(x)
        dense = self.mlp_dense(pre)
        h1 = self.post_ffn_ln_1(dense)
        h2 = self.post_ffn_ln_2(partial_moe)
        x_ffn = h1 + h2
        x_ffn = self.post_ffn_ln(x_ffn)
        x = residual + x_ffn
        x = x * self.layer_scalar
        return x

    def forward_ffn_last_partial_with_combine(
            self, x, prior_partial_moe, pack_start: int, pack_end: int):
        """Last FFN partial fused with the combiner.

        Runs expert packs [pack_start, pack_end), adds prior_partial_moe
        from earlier shards, then runs the full combine path (dense MLP +
        norms + residual + layer_scalar).  This eliminates the separate
        combiner shard which is too small to land on ANE.
        """
        rw = self.router(x)
        rw = rw.view(1, 1, N_PACKS, PACK_G)
        pre2 = self.pre_ffn_ln_2(x)
        moe = prior_partial_moe
        for p in range(pack_start, pack_end):
            per_e = self.packs[p](pre2)
            w_p = rw[:, :, p, :].unsqueeze(-1)
            moe = moe + (per_e * w_p).sum(dim=-2)
        return self.forward_ffn_combine(x, moe)

    def forward(self, x, cos, sin, k_cache, v_cache, attn_mask, kv_write_mask):
        x, k_new, v_new = self.forward_attn(
            x, cos, sin, k_cache, v_cache, attn_mask, kv_write_mask)
        x = self.forward_ffn(x)
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
        attn_delta = self.post_attn_ln(a)
        x = residual + attn_delta
        if "attn_delta" in tap_name_set:
            tap_values["attn_delta"] = attn_delta.view(1, D_MODEL)
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


def _make_layer_for_type(layer_type: str, max_ctx: int):
    if layer_type == "global":
        return GemmaGlobalLayer(max_ctx)
    if layer_type == "sliding":
        return GemmaSlidingLayer(max_ctx)
    raise ValueError(f"unknown layer type {layer_type!r}")


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


class GemmaLayer1AttnWrap(nn.Module):
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
        x, k_new, v_new = self.layer0.forward_attn(
            x, cos, sin, self.k_cache_0, self.v_cache_0,
            attn_mask, kv_write_mask)
        self.k_cache_0[:] = k_new
        self.v_cache_0[:] = v_new
        return x.view(1, D_MODEL), k_new, v_new


class GemmaLayer1FfnWrap(nn.Module):
    def __init__(self, max_ctx: int):
        super().__init__()
        self.max_ctx = max_ctx
        self.layer0 = GemmaSlidingLayer(max_ctx)

    def forward(self, x):
        return self.layer0.forward_ffn(x).view(1, D_MODEL)


class GemmaMixedSingleLayerStatefulWrap(nn.Module):
    def __init__(self, max_ctx: int, layer_type: str, split_mode: str = "full"):
        super().__init__()
        if split_mode not in ("full", "attn"):
            raise ValueError(
                f"stateful mixed single-layer split_mode must be 'full' or 'attn', got {split_mode!r}")
        self.max_ctx = max_ctx
        self.layer_type = layer_type
        self.split_mode = split_mode
        self.layer0 = _make_layer_for_type(layer_type, max_ctx)
        shp = _state_shape(layer_type == "global", max_ctx)
        self.register_buffer("k_cache_0", torch.zeros(*shp, dtype=torch.float16))
        self.register_buffer("v_cache_0", torch.zeros(*shp, dtype=torch.float16))

    def _rope_inputs(self, cos_s, sin_s, cos_g, sin_g):
        if self.layer_type == "global":
            return cos_g, sin_g
        return cos_s, sin_s

    def forward(self, x, cos_s, sin_s, cos_g, sin_g, attn_mask, kv_write_mask):
        cos, sin = self._rope_inputs(cos_s, sin_s, cos_g, sin_g)
        if self.split_mode == "attn":
            x, k_new, v_new = self.layer0.forward_attn(
                x, cos, sin, self.k_cache_0, self.v_cache_0,
                attn_mask, kv_write_mask)
        else:
            x, k_new, v_new = self.layer0(
                x, cos, sin, self.k_cache_0, self.v_cache_0,
                attn_mask, kv_write_mask)
        self.k_cache_0[:] = k_new
        self.v_cache_0[:] = v_new
        return x.view(1, D_MODEL), k_new, v_new


class GemmaMixedSingleLayerFfnWrap(nn.Module):
    def __init__(self, max_ctx: int, layer_type: str):
        super().__init__()
        self.max_ctx = max_ctx
        self.layer_type = layer_type
        self.layer0 = _make_layer_for_type(layer_type, max_ctx)

    def forward(self, x):
        return self.layer0.forward_ffn(x).view(1, D_MODEL)


class GemmaMixedSingleLayerFfnPartialWrap(nn.Module):
    """FFN sub-shard: router + expert packs [pack_start, pack_end).

    Each sub-shard is self-contained (duplicates router + pre_ffn_ln_2,
    which are tiny). Returns the partial weighted MoE contribution.
    The n_packs parameter is kept generic so other model families with
    different expert counts can reuse this wrapper.
    """
    def __init__(self, max_ctx: int, layer_type: str,
                 pack_start: int, pack_end: int):
        super().__init__()
        self.max_ctx = max_ctx
        self.layer_type = layer_type
        self.layer0 = _make_layer_for_type(layer_type, max_ctx)
        self.pack_start = pack_start
        self.pack_end = pack_end

    def forward(self, x):
        return self.layer0.forward_ffn_partial(
            x, self.pack_start, self.pack_end).view(1, D_MODEL)


class GemmaMixedSingleLayerFfnCombineWrap(nn.Module):
    """FFN combiner: dense MLP + final norms + residual + layer_scalar.

    Takes hidden state x and the pre-summed partial MoE output from all
    FFN sub-shards.  Runs the dense MLP path, combines with MoE, applies
    norms, residual connection, and per-layer scalar.
    """
    def __init__(self, max_ctx: int, layer_type: str):
        super().__init__()
        self.max_ctx = max_ctx
        self.layer_type = layer_type
        self.layer0 = _make_layer_for_type(layer_type, max_ctx)

    def forward(self, x, partial_moe):
        return self.layer0.forward_ffn_combine(
            x, partial_moe.view(1, 1, D_MODEL)).view(1, D_MODEL)


class GemmaMixedSingleLayerFfnLastPartialWrap(nn.Module):
    """Last FFN partial fused with combiner (dense MLP + norms + residual).

    Combines the final expert-pack shard with the combine path so that the
    separate combiner shard (which is too small to land on ANE) is
    eliminated.  Takes x (post-attn hidden), prior_partial_moe (sum of
    partial MoE outputs from earlier shards), and returns the final hidden.
    """
    def __init__(self, max_ctx: int, layer_type: str,
                 pack_start: int, pack_end: int):
        super().__init__()
        self.max_ctx = max_ctx
        self.layer_type = layer_type
        self.layer0 = _make_layer_for_type(layer_type, max_ctx)
        self.pack_start = pack_start
        self.pack_end = pack_end

    def forward(self, x, prior_partial_moe):
        return self.layer0.forward_ffn_last_partial_with_combine(
            x, prior_partial_moe,
            self.pack_start, self.pack_end)


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
    (cos_g, sin_g) [dim=128, partial rotation of first 128/512 head dims].
    Per-layer KV state buffers have type-appropriate shapes.
    """
    def __init__(self, max_ctx: int, layer_types: list[str]):
        super().__init__()
        self.max_ctx = max_ctx
        self.layer_types = list(layer_types)
        self.n_layers = len(layer_types)
        self.layers = nn.ModuleList(
            [_make_layer_for_type(t, max_ctx) for t in layer_types])
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
    """Load packed ts into a single Gemma layer (sliding or global)."""
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
    split_mode = getattr(args, "split_mode", "full")
    quant_config = _quant_config_from_args(args)
    quant_suffix = _quant_artifact_suffix(quant_config)
    mode_suffix = "" if split_mode == "full" else f"_{split_mode}"
    print(f"=== T4.1.1 layer1{mode_suffix} (all 128 experts, max_ctx={max_ctx}) ===")
    print(f"  split_mode: {split_mode}")
    print(f"  quant: {quant_config.describe()}")
    print(f"  PACK_G={PACK_G}, N_PACKS={N_PACKS}, N_EXPERTSPERTS={N_EXPERTS_KEEP}")

    if split_mode == "full":
        model = GemmaLayer1Wrap(max_ctx)
    elif split_mode == "attn":
        model = GemmaLayer1AttnWrap(max_ctx)
    elif split_mode == "ffn":
        model = GemmaLayer1FfnWrap(max_ctx)
    else:
        raise ValueError(f"unknown split_mode {split_mode!r}")
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
        if split_mode == "ffn":
            out = model(x_ex)
        else:
            out, k_new, v_new = model(x_ex, cos_ex, sin_ex, mask_ex, wmask_ex)
    print(f"  out: {out.shape} {out.dtype}, "
          f"finite={torch.isfinite(out).all().item()}")

    print("  tracing...")
    if split_mode == "ffn":
        traced = torch.jit.trace(model, (x_ex,))
    else:
        traced = torch.jit.trace(model, (x_ex, cos_ex, sin_ex, mask_ex, wmask_ex))

    if split_mode != "ffn":
        with torch.no_grad():
            model.k_cache_0.zero_()
            model.v_cache_0.zero_()

    print("  converting to CoreML (fp16, CPU+ANE)...")
    t0 = time.perf_counter()
    if split_mode == "ffn":
        ct_inputs = [
            ct.TensorType(name="x", shape=(1, 1, D_MODEL), dtype=np.float16),
        ]
        ct_outputs = [
            ct.TensorType(name="hidden", dtype=np.float16),
        ]
        ct_states = []
    else:
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

    mlmodel = _apply_weight_quantization(mlmodel, quant_config)

    out_pkg = OUT_DIR / f"gemma4_layer1_real{mode_suffix}{quant_suffix}.mlpackage"
    if out_pkg.exists():
        shutil.rmtree(out_pkg)
    mlmodel.save(str(out_pkg))
    pkg_mb = sum(f.stat().st_size for f in out_pkg.rglob("*") if f.is_file()) / 1e6
    print(f"  saved → {out_pkg}  ({pkg_mb:.1f} MB)")

    compiled = OUT_DIR / f"gemma4_layer1_real{mode_suffix}{quant_suffix}.mlmodelc"
    if compiled.exists():
        shutil.rmtree(compiled)
    print("  coremlcompiler...")
    t0 = time.perf_counter()
    compiled = Path(ct.utils.compile_model(str(out_pkg), str(compiled)))
    print(f"  compile wall: {time.perf_counter() - t0:.1f}s")
    print(f"  compiled → {compiled}")
    print(f"\n# T4.1.1 LAYER1{mode_suffix.upper()}: PASS (compiled)")


def cmd_stack(args):
    max_ctx  = args.max_ctx
    n_layers = args.n_layers
    clone    = args.clone
    quant_config = _quant_config_from_args(args)
    quant_suffix = _quant_artifact_suffix(quant_config)
    print(f"=== T4.1.3 stack N={n_layers} (max_ctx={max_ctx}, clone={clone}) ===")
    print(f"  quant: {quant_config.describe()}")

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

    mlmodel = _apply_weight_quantization(mlmodel, quant_config)

    tag = "clone" if clone else "real"
    out_pkg = OUT_DIR / f"gemma4_stack{n_layers}_{tag}{quant_suffix}.mlpackage"
    if out_pkg.exists():
        shutil.rmtree(out_pkg)
    mlmodel.save(str(out_pkg))
    pkg_mb = sum(f.stat().st_size for f in out_pkg.rglob("*") if f.is_file()) / 1e6
    print(f"  saved → {out_pkg}  ({pkg_mb:.1f} MB)")

    compiled = OUT_DIR / f"gemma4_stack{n_layers}_{tag}{quant_suffix}.mlmodelc"
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
    _repo_root = Path(__file__).resolve().parent.parent.parent
    cfg = json.loads((_repo_root / "models" / "gemma-4-26b-a4b" / "config.json").read_text())
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


def _convert_save_compile(traced, ct_inputs, ct_outputs, ct_states,
                          quant_config: QuantConfig,
                          out_pkg: Path, compiled: Path, label: str):
    """Common convert → quantize → save → compile cycle.

    Returns the compiled Path.  Caller is responsible for creating the traced
    model and defining I/O specs.  Reusable across model families.
    """
    print(f"  converting {label} to CoreML (fp16, CPU+ANE)...")
    t0 = time.perf_counter()
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

    mlmodel = _apply_weight_quantization(mlmodel, quant_config)

    if out_pkg.exists():
        shutil.rmtree(out_pkg)
    mlmodel.save(str(out_pkg))
    pkg_mb = sum(f.stat().st_size for f in out_pkg.rglob("*") if f.is_file()) / 1e6
    print(f"  saved \u2192 {out_pkg}  ({pkg_mb:.1f} MB)")

    if compiled.exists():
        shutil.rmtree(compiled)
    print("  coremlcompiler...")
    t0 = time.perf_counter()
    compiled = Path(ct.utils.compile_model(str(out_pkg), str(compiled)))
    print(f"  compile wall: {time.perf_counter() - t0:.1f}s")
    print(f"  compiled \u2192 {compiled}")
    del mlmodel
    return compiled


def _cmd_mixed_ffn_subsplit(*, max_ctx: int, layer_type: str, tag: str,
                            quant_config: QuantConfig, quant_suffix: str,
                            npz_path: Path, ffn_shards: int,
                            layer_start: int, n_layers: int, shard_n: int):
    """Produce (N-1) FFN partial shards + 1 merged last-partial-with-combiner.

    Each partial shard contains the router + a slice of expert packs.
    The last shard also includes the dense MLP + norms + residual (combiner)
    so that all ops land on ANE — a standalone combiner is too small for the
    ANE scheduler.
    """
    assert npz_path.exists(), f"missing pack: {npz_path}"
    packs_per_shard = N_PACKS // ffn_shards

    ct_x_input = [
        ct.TensorType(name="x", shape=(1, 1, D_MODEL), dtype=np.float16),
    ]
    ct_partial_output = [
        ct.TensorType(name="partial_moe", dtype=np.float16),
    ]

    print(f"\n{'='*60}")
    if ffn_shards > 1:
        print(f"FFN sub-split: {ffn_shards - 1} partial + 1 merged last")
    else:
        print(f"FFN sub-split: 1 merged shard (partial + combiner)")
    print(f"  packs_per_shard: {packs_per_shard}  (total {N_PACKS})")
    print(f"  quant: {quant_config.describe()}")
    print(f"{'='*60}")

    x_ex = torch.randn(1, 1, D_MODEL, dtype=torch.float16) * 0.1

    # --- Regular partial shards (all except the last) ---------------------
    for k in range(ffn_shards - 1):
        pack_start = k * packs_per_shard
        pack_end = (k + 1) * packs_per_shard
        suffix = f"_ffn_p{k}of{ffn_shards}"
        print(f"\n--- partial {k}/{ffn_shards} "
              f"(packs {pack_start}..{pack_end - 1}) ---")

        model = GemmaMixedSingleLayerFfnPartialWrap(
            max_ctx, layer_type, pack_start, pack_end)
        model.half().eval()
        print(f"  loading weights from {npz_path.name}...")
        _load_layer_weights(model.layer0, npz_path)
        model.layer0.fuse_norm_scales_for_ane()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  parameters (full layer): {n_params:,} "
              f"= {n_params * 2 / 1e6:.1f} MB pre-quant")

        print("  forward smoke...")
        with torch.no_grad():
            out = model(x_ex)
        print(f"  out: {out.shape} {out.dtype}, "
              f"finite={torch.isfinite(out).all().item()}")

        print(f"  tracing (ffn partial {k})...")
        traced = torch.jit.trace(model, (x_ex,))

        out_pkg = OUT_DIR / f"gemma4_{tag}_real{suffix}{quant_suffix}.mlpackage"
        compiled = OUT_DIR / f"gemma4_{tag}_real{suffix}{quant_suffix}.mlmodelc"
        # Keep router projection (128×2816 = 360K params) in fp16 to avoid
        # routing instability from INT8 quantization.  Expert packs (≥31M)
        # are still INT8.  Adds ~720 KB per shard.
        partial_quant = QuantConfig(
            bits=quant_config.bits,
            group_size=quant_config.group_size,
            granularity=quant_config.granularity,
            weight_threshold=500_000,
        ) if quant_config.bits == 8 else quant_config
        _convert_save_compile(
            traced, ct_x_input, ct_partial_output, [],
            partial_quant, out_pkg, compiled,
            label=f"partial {k}/{ffn_shards}")
        print(f"  # PARTIAL {k}: PASS")
        del model, traced

    # --- Last partial + combiner (merged) ---------------------------------
    last_k = ffn_shards - 1
    pack_start = last_k * packs_per_shard
    pack_end = N_PACKS
    suffix_last = f"_ffn_p{last_k}of{ffn_shards}"
    print(f"\n--- last partial {last_k}/{ffn_shards} + combiner "
          f"(packs {pack_start}..{pack_end - 1} + dense MLP + norms) ---")

    model = GemmaMixedSingleLayerFfnLastPartialWrap(
        max_ctx, layer_type, pack_start, pack_end)
    model.half().eval()
    print(f"  loading weights from {npz_path.name}...")
    _load_layer_weights(model.layer0, npz_path)
    model.layer0.fuse_norm_scales_for_ane()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters (full layer): {n_params:,} "
          f"= {n_params * 2 / 1e6:.1f} MB pre-quant")

    prior_ex = torch.randn(1, 1, D_MODEL, dtype=torch.float16) * 0.1
    print("  forward smoke...")
    with torch.no_grad():
        out = model(x_ex, prior_ex)
    print(f"  out: {out.shape} {out.dtype}, "
          f"finite={torch.isfinite(out).all().item()}")

    print(f"  tracing (last partial + combiner)...")
    traced = torch.jit.trace(model, (x_ex, prior_ex))

    ct_last_inputs = [
        ct.TensorType(name="x",                shape=(1, 1, D_MODEL), dtype=np.float16),
        ct.TensorType(name="prior_partial_moe", shape=(1, 1, D_MODEL), dtype=np.float16),
    ]
    ct_last_output = [
        ct.TensorType(name="hidden", dtype=np.float16),
    ]
    out_pkg = OUT_DIR / f"gemma4_{tag}_real{suffix_last}{quant_suffix}.mlpackage"
    compiled = OUT_DIR / f"gemma4_{tag}_real{suffix_last}{quant_suffix}.mlmodelc"
    # The merged shard contains both expert pack weights (~31M params each,
    # INT8-friendly) and dense MLP weights (~6M params each, INT8-hostile
    # because of outlier channels after norm fusion → only 3 INT8 levels at
    # mean value).  Use weight_threshold to keep dense MLP in fp16 while
    # INT8-quantizing the expert packs.  Adds ~15 MB vs all-INT8.
    last_quant = QuantConfig(
        bits=quant_config.bits,
        group_size=quant_config.group_size,
        granularity=quant_config.granularity,
        weight_threshold=10_000_000,
    ) if quant_config.bits == 8 else quant_config
    _convert_save_compile(
        traced, ct_last_inputs, ct_last_output, [],
        last_quant, out_pkg, compiled,
        label=f"last partial + combiner")
    print(f"  # LAST PARTIAL + COMBINER: PASS")
    del model, traced

    print(f"\n# T4.1.3c {tag.upper()}_FFN_SUBSPLIT: "
          f"ALL {ffn_shards} SHARDS PASS (compiled)")


def cmd_mixed(args):
    max_ctx  = args.max_ctx
    n_layers = args.n_layers
    tag_suffix = getattr(args, "tag_suffix", None)
    split_mode = getattr(args, "split_mode", "full")
    mode_suffix = "" if split_mode == "full" else f"_{split_mode}"
    quant_config = _quant_config_from_args(args)
    quant_suffix = _quant_artifact_suffix(quant_config)
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
    if split_mode != "full" and shard_n != 1:
        raise ValueError(
            "mixed --split-mode attn/ffn is only supported when the selected "
            "layer range contains exactly one layer")
    print(f"=== T4.1.3c {tag} N={shard_n}/{n_layers} (max_ctx={max_ctx}) ===")
    print(f"  split_mode: {split_mode}")
    print(f"  quant: {quant_config.describe()}")
    print(f"  global_layer_range: [{layer_start}, {layer_end})")
    print(f"  layer_types (shard): {layer_types}")
    n_global = sum(1 for t in layer_types if t == "global")
    print(f"  sliding={shard_n-n_global}  global={n_global}")
    if tap_output_names:
        print(f"  debug hidden taps: {tap_output_names}")
    if split_mode != "full" and tap_output_names:
        raise ValueError(
            "debug hidden taps are only supported for mixed --split-mode full")

    ffn_shards = getattr(args, "ffn_shards", 1)
    if ffn_shards > 1 and split_mode != "ffn":
        raise ValueError(
            "--ffn-shards > 1 only makes sense with --split-mode ffn")
    if ffn_shards > 1 and N_PACKS % ffn_shards != 0:
        raise ValueError(
            f"--ffn-shards={ffn_shards} does not evenly divide "
            f"N_PACKS={N_PACKS}")

    # --- FFN sub-shard path (produces N partials + 1 combiner) -----------
    if split_mode == "ffn" and ffn_shards > 1:
        _cmd_mixed_ffn_subsplit(
            max_ctx=max_ctx, layer_type=layer_types[0], tag=tag,
            quant_config=quant_config, quant_suffix=quant_suffix,
            npz_path=OUT_DIR / f"gemma_layer{layer_start}_packed.npz",
            ffn_shards=ffn_shards, layer_start=layer_start,
            n_layers=n_layers, shard_n=shard_n)
        return

    if split_mode == "full":
        model = GemmaMixedStackWrap(max_ctx, layer_types)
    elif split_mode == "ffn":
        model = GemmaMixedSingleLayerFfnWrap(max_ctx, layer_types[0])
    else:
        model = GemmaMixedSingleLayerStatefulWrap(
            max_ctx, layer_types[0], split_mode=split_mode)
    model.half().eval()

    npz_paths = [OUT_DIR / f"gemma_layer{i}_packed.npz"
                 for i in range(layer_start, layer_end)]
    for p in npz_paths:
        assert p.exists(), f"missing pack: {p}"
    print(f"  loading {shard_n} layer packs (global {layer_start}..{layer_end-1})...")
    if split_mode == "full":
        for local_i, pth in enumerate(npz_paths):
            gi = layer_start + local_i
            print(f"  layer {gi} ({layer_types[local_i]}): {pth.name}")
            _load_layer_weights(model.layers[local_i], pth)
    else:
        print(f"  layer {layer_start} ({layer_types[0]}): {npz_paths[0].name}")
        _load_layer_weights(model.layer0, npz_paths[0])

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
        if split_mode == "ffn":
            out = model(x_ex)
        else:
            out, k_new, v_new = model(x_ex, cos_s, sin_s, cos_g, sin_g,
                                      mask_ex, wmask_ex)
    print(f"  out: {out.shape} {out.dtype}, "
          f"finite={torch.isfinite(out).all().item()}")

    if split_mode == "full":
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
    elif split_mode == "ffn":
        print("  tracing (ffn-only output for mixed shard)...")
        traced = torch.jit.trace(model, (x_ex,))
    else:
        print("  tracing (attn-only output for mixed shard)...")
        traced = torch.jit.trace(
            model,
            (x_ex, cos_s, sin_s, cos_g, sin_g, mask_ex, wmask_ex))
        with torch.no_grad():
            model.k_cache_0.zero_()
            model.v_cache_0.zero_()

    if split_mode == "full":
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
    elif split_mode == "ffn":
        ct_inputs = [
            ct.TensorType(name="x", shape=(1, 1, D_MODEL), dtype=np.float16),
        ]
        ct_outputs = [
            ct.TensorType(name="hidden", dtype=np.float16),
        ]
        ct_states = []
    else:
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
            ct.TensorType(name="k_new",  dtype=np.float16),
            ct.TensorType(name="v_new",  dtype=np.float16),
        ]
        shp = _state_shape(layer_types[0] == "global", max_ctx)
        ct_states = [
            ct.StateType(
                wrapped_type=ct.TensorType(shape=shp, dtype=np.float16),
                name="k_cache_0"),
            ct.StateType(
                wrapped_type=ct.TensorType(shape=shp, dtype=np.float16),
                name="v_cache_0"),
        ]

    out_pkg = OUT_DIR / f"gemma4_{tag}_real{mode_suffix}{quant_suffix}.mlpackage"
    compiled = OUT_DIR / f"gemma4_{tag}_real{mode_suffix}{quant_suffix}.mlmodelc"
    _convert_save_compile(
        traced, ct_inputs, ct_outputs, ct_states,
        quant_config, out_pkg, compiled,
        label=f"{tag}{mode_suffix}")
    print(f"\n# T4.1.3c {tag.upper()}{mode_suffix.upper()}: PASS (compiled)")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("layer1")
    pl.add_argument("--max-ctx", type=int, default=1024)
    pl.add_argument("--split-mode", choices=["full", "attn", "ffn"],
                    default="full",
                    help=("Build the full layer, only the attention/stateful half, "
                          "or only the FFN/router half. Useful for large-layer "
                          "models that need intra-layer splits."))
    _add_quant_args(pl)
    pl.set_defaults(func=cmd_layer1)

    ps = sub.add_parser("stack")
    ps.add_argument("--n-layers", type=int, required=True)
    ps.add_argument("--max-ctx", type=int, default=1024)
    ps.add_argument("--clone", action="store_true",
                    help="reuse layer0 npz for all N layers (cheap probe)")
    _add_quant_args(ps)
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
    pm.add_argument("--split-mode", choices=["full", "attn", "ffn"],
                    default="full",
                    help=("Build the full shard, only the attention/stateful half, "
                          "or only the FFN/router half. Split mode is only "
                          "supported when the selected mixed shard contains exactly "
                          "one layer."))
    pm.add_argument("--ffn-shards", type=int, default=1, dest="ffn_shards",
                    help=("Number of FFN sub-shards when --split-mode=ffn. "
                          "Default 1 (single shard). When >1, produces N partial "
                          "expert-pack shards + 1 combiner (dense MLP + norms). "
                          "Must evenly divide the number of expert packs "
                          f"(currently {N_PACKS})."))
    _add_quant_args(pm)
    pm.set_defaults(func=cmd_mixed)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
