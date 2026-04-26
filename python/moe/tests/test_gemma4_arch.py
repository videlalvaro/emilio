"""test_gemma4_arch.py — architectural conformance battery vs HF Gemma-4.

Reference sources (independently confirm the same architecture):
  HF transformers:
    .venv313/lib/python3.13/site-packages/transformers/models/gemma4/modeling_gemma4.py
    classes: Gemma4TextDecoderLayer, Gemma4TextAttention, Gemma4TextMLP,
             Gemma4TextRouter, Gemma4TextExperts, Gemma4TextRotaryEmbedding
  llama.cpp:
    /Users/alvarovidela/Code/llama.cpp/src/models/gemma4-iswa.cpp
    function: llm_build_gemma4_iswa::llm_build_gemma4_iswa()

This battery does NOT load the full HF Gemma-4 26B model (50 GB). It uses
structural inspection + functional ablation on real packed-weight layers.

Run:
  /Applications/Xcode.app/Contents/Developer/usr/bin/python3 \
      python/moe/tests/test_gemma4_arch.py

Each test prints PASS/FAIL with a one-line reason. Goal: most fail at first,
get fixed one by one.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gemma_to_ane import (  # noqa: E402
    GemmaSlidingLayer, GemmaGlobalLayer, DenseMLP,
    SlidingAttention, GlobalAttention,
    _load_layer_weights, _layer_types_from_config,
    D_MODEL, SLD_D_HEAD, GLB_ROT_DIM,
)

OUT_DIR = Path("python/moe/out")
MAX_CTX = 1024
LAYER_FOR_TEST = 29   # the late attractor layer
SLIDING_LAYER_FOR_TEST = 28  # last sliding layer


def _build_layer(layer_idx: int):
    types = _layer_types_from_config(30)
    t = types[layer_idx]
    if t == "global":
        m = GemmaGlobalLayer(MAX_CTX)
    else:
        m = GemmaSlidingLayer(MAX_CTX)
    m.half().eval()
    _load_layer_weights(m, OUT_DIR / f"gemma_layer{layer_idx}_packed.npz")
    return m, t


# ===========================================================================
# A. Structural conformance — module attributes / submodules
# ===========================================================================

class TestStructuralConformance(unittest.TestCase):

    def test_01_layer_has_layer_scalar_buffer(self):
        """HF: Gemma4TextDecoderLayer.register_buffer('layer_scalar', ...)
        + multiplied at end of forward: hidden_states *= self.layer_scalar."""
        layer, _ = _build_layer(LAYER_FOR_TEST)
        has = hasattr(layer, "layer_scalar")
        self.assertTrue(has,
            "FAIL: layer.layer_scalar buffer missing — "
            "npz contains it but module never registers it.")

    def test_02_layer_has_moe_branch(self):
        """HF: enable_moe_block=True → layer must have router, experts (or REAP
        packs equivalent), post_feedforward_layernorm_1, post_feedforward_layernorm_2,
        pre_feedforward_layernorm_2 modules."""
        layer, _ = _build_layer(LAYER_FOR_TEST)
        has_router  = any(hasattr(layer, n) for n in ("router", "moe_router"))
        has_experts = any(hasattr(layer, n) for n in ("experts", "moe_experts", "packs"))
        has_norms   = all(hasattr(layer, n)
                          for n in ("post_ffn_ln_1", "post_ffn_ln_2", "pre_ffn_ln_2"))
        self.assertTrue(has_router and has_experts and has_norms,
            f"FAIL: MoE branch incomplete. router={has_router} "
            f"experts={has_experts} 3-norms={has_norms}")

    def test_03_global_layer_uses_k_eq_v(self):
        """HF: attention_k_eq_v=True for non-sliding layers.
        v_proj must be None / absent on global layers."""
        layer, t = _build_layer(LAYER_FOR_TEST)
        self.assertEqual(t, "global", "L29 must be a global layer")
        attn = layer.attn
        # k_eq_v means no separate v_proj
        has_separate_v = hasattr(attn, "v_proj") and attn.v_proj is not None
        self.assertFalse(has_separate_v,
            "FAIL: global attention has separate v_proj; should derive V from K (k_eq_v).")

    def test_04_sliding_layer_has_v_proj(self):
        """HF: sliding layers (is_sliding=True) skip k_eq_v alt-attention,
        so they DO have a separate v_proj."""
        layer, t = _build_layer(SLIDING_LAYER_FOR_TEST)
        self.assertEqual(t, "sliding", "L28 must be sliding")
        attn = layer.attn
        self.assertTrue(hasattr(attn, "v_proj") and attn.v_proj is not None,
            "FAIL: sliding attention missing v_proj.")

    def test_05_v_norm_is_scale_free(self):
        """HF: v_norm = Gemma4RMSNorm(head_dim, eps, with_scale=False)."""
        layer, _ = _build_layer(LAYER_FOR_TEST)
        v_norm = layer.attn.v_norm
        self.assertFalse(getattr(v_norm, "with_scale", True),
            "FAIL: v_norm should be scale-free RMS (γ=1).")

    def test_06_q_k_norm_with_scale(self):
        """HF: q_norm + k_norm = Gemma4RMSNorm(head_dim, eps, with_scale=True)."""
        layer, _ = _build_layer(LAYER_FOR_TEST)
        for name in ("q_norm", "k_norm"):
            n = getattr(layer.attn, name)
            self.assertTrue(getattr(n, "with_scale", False),
                f"FAIL: {name} must be RMSNorm with γ.")


# ===========================================================================
# B. Functional / ablation — what the forward actually does
# ===========================================================================

class TestForwardSemantics(unittest.TestCase):

    def _make_inputs(self, layer_t: str, pos: int = 0):
        from gemma_mixedN_golden import _real_rope
        x = (torch.randn(1, 1, D_MODEL) * 0.5).half()
        if layer_t == "global":
            cs, sn = _real_rope(theta=1_000_000.0, dh=GLB_ROT_DIM, pos=pos)
        else:
            cs, sn = _real_rope(theta=10_000.0, dh=SLD_D_HEAD, pos=pos)
        cs = torch.from_numpy(cs.astype(np.float16).reshape(1, 1, -1))
        sn = torch.from_numpy(sn.astype(np.float16).reshape(1, 1, -1))
        am = torch.full((1, 1, 1, MAX_CTX), -1e4, dtype=torch.float16)
        am[..., :pos+1] = 0.0
        wm = torch.zeros(1, 1, MAX_CTX, 1, dtype=torch.float16)
        wm[0, 0, pos, 0] = 1.0
        return x, cs, sn, am, wm

    def test_07_layer_scalar_actually_scales_output(self):
        """If layer_scalar is wired in forward, halving its value should
        scale the output's deviation-from-input correspondingly."""
        layer, t = _build_layer(LAYER_FOR_TEST)
        x, cs, sn, am, wm = self._make_inputs(t)
        if not hasattr(layer, "layer_scalar"):
            self.skipTest("layer_scalar buffer missing (see test_01)")

        # Synthetic KV caches (caches live on wrap, not layer).
        from gemma_to_ane import GLB_N_KV, GLB_D_HEAD
        kc = torch.zeros(1, GLB_N_KV, MAX_CTX, GLB_D_HEAD, dtype=torch.float16)
        vc = torch.zeros(1, GLB_N_KV, MAX_CTX, GLB_D_HEAD, dtype=torch.float16)

        with torch.no_grad():
            ls_orig = layer.layer_scalar.clone()
            layer.layer_scalar.fill_(1.0)
            h_a, _, _ = layer(x.clone(), cs, sn, kc.clone(), vc.clone(), am, wm)
            layer.layer_scalar.fill_(0.5)
            h_b, _, _ = layer(x.clone(), cs, sn, kc.clone(), vc.clone(), am, wm)
            layer.layer_scalar.copy_(ls_orig)

        ratio = h_b.float().norm() / (h_a.float().norm() + 1e-9)
        self.assertAlmostEqual(ratio.item(), 0.5, delta=0.01,
            msg=f"FAIL: ‖out(0.5)‖/‖out(1.0)‖={ratio.item():.4f}, expected ~0.5. "
                "layer_scalar is not applied in forward.")

    def test_08_attention_scaling_is_one(self):
        """HF: Gemma4TextAttention.scaling = 1.0 (q_norm/k_norm provide
        magnitude control). Our impl should not divide by sqrt(d_head)."""
        # Probe the attention forward by zeroing all weights except q_proj=I,
        # k_proj=I, v_proj=I (or k_eq_v), and norms=identity. Then attention
        # output magnitude depends on whether scale=1 or scale=1/sqrt(d).
        # Easier: introspect the source to find any "** -0.5" scale factor.
        import inspect
        src_s = inspect.getsource(SlidingAttention.forward)
        src_g = inspect.getsource(GlobalAttention.forward)
        bad = [s for s in (src_s, src_g)
               if ("** -0.5" in s) or ("**-0.5" in s)
                  or ("/ math.sqrt" in s) or ("rsqrt" in s)]
        self.assertEqual(len(bad), 0,
            "FAIL: attention forward divides by sqrt(d_head). "
            "HF Gemma-4 uses scaling=1.0.")

    def test_09_dense_mlp_uses_gelu_tanh(self):
        """HF: hidden_activation='gelu_pytorch_tanh' → F.gelu(x, 'tanh')."""
        import inspect
        src = inspect.getsource(DenseMLP.forward)
        self.assertIn("approximate=\"tanh\"", src.replace("'", "\""),
            "FAIL: DenseMLP must use F.gelu(x, approximate='tanh').")


# ===========================================================================
# C. End-to-end smoke (uses real HF golden via T4.1.4)
# ===========================================================================

class TestEndToEnd(unittest.TestCase):

    # Threshold ratchets up as fixes land. Edit when re-baselining.
    MIN_COS_REQUIRED = 0.97   # ship gate target

    def test_10_t414_min_cos_above_threshold(self):
        """End-to-end ship gate: T4.1.4 logit gate min_cos >= MIN_COS_REQUIRED."""
        sentinel = OUT_DIR / ".gemma_t414_logit_gate_PASS"
        if not sentinel.exists():
            self.fail(f"FAIL: T4.1.4 sentinel missing → ship gate not yet passed. "
                      f"Run gemma_t414_logit_gate.py and confirm cos ≥ {self.MIN_COS_REQUIRED}.")
        txt = sentinel.read_text()
        # parse "min_cos=0.xxxxxx" from sentinel
        for tok in txt.split():
            if tok.startswith("min_cos="):
                v = float(tok.split("=", 1)[1].rstrip(","))
                self.assertGreaterEqual(v, self.MIN_COS_REQUIRED,
                    f"FAIL: T4.1.4 min_cos={v:.4f} < {self.MIN_COS_REQUIRED}")
                return
        self.fail("FAIL: could not parse min_cos from sentinel")


if __name__ == "__main__":
    # Run with descriptive output. Use -v for full traceback.
    unittest.main(verbosity=2)
