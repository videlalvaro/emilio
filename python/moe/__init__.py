"""MoE-on-K2.6-proxy experiments.

Target: DeepSeek-V2-Lite-Chat (15.7B / 2.4B active, 64 routed experts + 2 shared,
27 layers, MLA attention) - same architecture family as Kimi K2.6 (1T / 32B,
384 routed experts + 1 shared, 61 layers).

Experiments:
  1. expert_utilization.py  - which experts actually fire on a calibration corpus
  2. expert_similarity.py   - pairwise weight similarity within each MoE layer
  3. expert_bwt_entropy.py  - is the expert bank BWT-compressible beyond LZ77?
"""
