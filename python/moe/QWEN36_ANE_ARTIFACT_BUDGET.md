# Qwen3.6 ANE Artifact Budget Sheet

**Purpose**: compare the three realistic routed-expert packaging strategies for `Qwen/Qwen3.6-35B-A3B` on ANE:

- separate per-expert packs
- 32-expert banks
- 64-expert banks

This sheet is a planning aid for **Phase 0** and **Phase 1** of [QWEN36_ANE_PORT_PLAN.md](/Users/alvarovidela/Code/em2/python/moe/QWEN36_ANE_PORT_PLAN.md). The first artifact measurements now exist and they materially changed the picture: package size is no longer the main blocker, executed-op geometry is.

---

## 0. Measured Artifact Anchors (2026-04-23)

| Artifact | Shape | `.mlpackage` | `.mlmodelc` | Residency result | Interpretation |
|---|---|---:|---:|---|---|
| Single expert | layer 0, expert 0 | 2.37 MB | 2.37 MB | not representative | Too small to say anything about ANE landing |
| MFD bank 32 | layer 0, experts 0..31 | 75.59 MB | 75.60 MB | CPU fallback | Below cliff, but separate-function banking is not ANE-viable in current shape |
| Packed executed-op G=24 | layer 0, experts 0..23 | 56.75 MB | 56.75 MB | CPU/CPU/CPU | In the expected size window, still wrong executed geometry |
| Packed executed-op G=32 | layer 0, experts 0..31 | 75.67 MB | 75.67 MB | CPU/CPU/CPU | Last nearby probe also failed; stop widening this branch |
| Conv-authored single expert B=1 | layer 0, expert 0 | 2.37 MB | 2.37 MB | CPU/CPU/CPU | PF operator family alone does not rescue Qwen decode geometry |
| Conv-authored single expert B=64 | layer 0, expert 0 | 2.37 MB | 2.37 MB | CPU/CPU/CPU | Even PF-like batch does not change placement |

These four numbers replace the earlier assumption that a 32-expert public-CoreML bank would be roughly 48 MB in compiled form and likely land on ANE.

---

## 1. Fixed Assumptions

### Model-side assumptions

- 40 layers
- 256 routed experts per layer
- routed expert weight budget: **~1.5 MB INT4 per expert**
- routed expert raw weight total: `40 × 256 × 1.5 MB = 15,360 MB = 15.36 GB`

### Fixed non-routed ANE artifact estimate

These do not change across packaging variants.

| Artifact class | Count | Planning estimate |
|---|---:|---:|
| DeltaNet / full-attention packs | 40 | ~340 MB total |
| Shared expert packs | 40 | ~160 MB total |
| **Fixed non-routed ANE total** | **80** | **~500 MB** |

So the total ANE-resident weight footprint is roughly:

- routed experts: **15.36 GB**
- fixed non-routed artifacts: **~0.5 GB**
- **whole ANE-side planning total: ~15.9 GB**

This fits in 48 GB unified memory. The open question is not RAM capacity but **driver cache behavior at this total byte count**.

### Proven PF baseline to compare against

From privacy-filter on M4 Max:

- **1024 artifacts** proven resident with no measurable eviction
- **~2.4 GB total weights** proven resident

That means Qwen3.6 is beyond the PF proof in **bytes** no matter which packaging strategy is used. Banking only changes artifact count and package size; it does **not** reduce total routed weight bytes.

---

## 2. Formulas

For a bank size `B` experts per routed artifact:

- packs per layer = `256 / B`
- routed packs total = `40 × (256 / B)`
- raw MB per routed artifact = `1.5 × B`
- raw routed MB per layer = `256 × 1.5 = 384 MB`
- raw routed GB per model = `15.36 GB` (constant)
- total ANE artifact count = `80 + routed_packs_total`

---

## 3. Variant Comparison

| Variant | Experts / routed artifact | Routed artifacts / layer | Routed artifacts / model | Raw MB / routed artifact | Total ANE artifacts incl. fixed 80 | Cliff risk | Cache/load risk | Verdict |
|---|---:|---:|---:|---:|---:|---|---|---|
| Separate packs | 1 | 256 | 10,240 | 1.5 MB | 10,320 | Safest per artifact | Extreme artifact-count risk; 10x PF-proven artifact count | Use only for one-layer proof and correctness bring-up |
| 32-expert banks | 32 | 8 | 320 | 48 MB raw, **75.60 MB measured compiled in current MFD shape** | 400 | Still below cliff | Artifact count is comfortable, but the measured MFD-32 bank fell back to CPU | **Blocked in current bank shape** |
| 64-expert banks | 64 | 4 | 160 | 96 MB raw | 240 | At the cliff before metadata/scales | Artifact count is best, but package-size risk is worst | Blocked unless a different smaller packed shape lands on ANE first |

---

## 4. Same Bytes, Different Failure Modes

### Separate per-expert packs

- Solves the package-size problem completely.
- Maximizes the number of XPC/model objects, file opens, load operations, and cache entries.
- Best for proving one expert and one layer because the mechanics are already validated by privacy-filter.
- Worst candidate for full-model deployment unless the cache law scales by an order of magnitude in artifact count.

### 32-expert banks

- Keeps the **raw** routed artifact size at ~48 MB, but the first real compiled Qwen MFD bank came out at **75.60 MB**.
- Cuts routed artifacts from `10,240` to `320`.
- Keeps the public CoreML surface.
- The current measured MFD shape is not ANE-viable because the executed expert path still fell back to CPU.

### Packed executed-op probes

- These were the next local attempt after the MFD bank failure.
- `G=24` compiled at **56.75 MB** and still ran `CPU/CPU/CPU`.
- `G=32` compiled at **75.67 MB** and still ran `CPU/CPU/CPU`.
- Conv-as-linear single-expert probes at both `B=1` and `B=64` also ran `CPU/CPU/CPU`.
- This means current executed geometry, not package size alone, is the blocker.

### Single-expert low-rank split is not a valid ANE residency gate

- A rank-`r` factorization of one Qwen expert matrix (`2048 x 512` or `512 x 2048`) creates factors `2048 x r` and `r x 512`
- The largest factor has raw INT4 bytes `2048 * r * 0.5 = 1024r`
- To reach the empirical lower ANE landing band of about `12 MB`, we would need `r ≈ 11,700`
- That is impossible because the matrix max rank here is only `512`
- So a single-expert low-rank chain may still be numerically interesting, but it cannot be the right residency probe under the current ANE size law
- If low-rank is used, it has to appear inside a **shared-basis or packed factor-bank architecture** where the executed base path is large enough to land on ANE

### Shared expert is a bad residual anchor for plain routed-expert adapters

- The layer-0 / expert-0 offline probe tested the cheapest version of this idea before any new CoreML artifact build
- Shared-only output cosine was only `min=-0.046877, mean=0.022067`
- Residual Fro norms were larger than the routed expert itself: `gate 2.18x`, `up 2.02x`, `down 1.91x`
- Direct raw-low-rank beat shared+low-rank-residual at every tested rank `8, 16, 32, 64, 128, 256`
- So a plain shared-expert base does not actually reduce the approximation pressure on the routed path, even before ANE packaging enters the picture
- If basis sharing stays in play, it should move to a **cluster-shared or packed factor-bank base**, not the one shared expert that HF adds as a separately gated path

### Cluster-shared routed bases are the first positive signal

- A cheap layer-0 / expert-0 cluster-mean proxy produced the first measurable improvement over direct raw-low-rank
- Tight clusters help most: cluster `4` cut residual norm ratios below `1.0` on all three matrices (`gate 0.823x`, `up 0.831x`, `down 0.834x`)
- That same cluster `4` beat raw low-rank at every tested rank `8..256`; for example rank `8` improved from `0.137576` to `0.352795` mean cosine, and rank `64` improved from `0.492317` to `0.646179`
- Clusters `8` and `16` still beat raw low-rank, but the gain shrinks as the cluster becomes more diffuse
- Widened offline gate: cluster `24` remained slightly positive at ranks `64`, `128`, and `256`, which made it the first admissible packed choice under the ANE size law
- First packed-base CoreML probe at cluster `24`, rank `128`, batch `1` measured `.mlpackage 74.58 MB` and `.mlmodelc 74.59 MB`, so the artifact stayed below the `~96 MB` cliff while putting the packed base in the intended byte band
- But all linears still placed on CPU, so simply widening the base is not enough if the graph still carries many tiny per-expert residual linears around it
- Shared-basis bank follow-up closed the remaining size-floor excuse: exact three-bank probes at cluster24/basis24 (`12.58 MB` per family, `.mlpackage 56.84 MB`) and cluster32/basis32 (`16.78 MB` per family, `.mlpackage 75.78 MB`) both still placed `gate/up/down` on CPU
- Conv-authored full-bank follow-up at cluster32/basis32 also stayed on CPU: the three-bank `1x1` conv graph compiled at `.mlpackage 75.81 MB` / `.mlmodelc 75.82 MB` and still placed all three hot banks on CPU
- Minimal three-conv isolator at the same cluster32/basis32 size also stayed on CPU: raw total `50.33 MB`, `.mlpackage 75.67 MB`, `.mlmodelc 75.67 MB`, and all three hot conv banks still placed on CPU even after removing coefficient mixing and per-slot recombine glue
- Fused Gate+Up follow-up also stayed on CPU: a two-bank `1x1` conv graph at the same cluster32/basis32 total raw bytes compiled at `.mlpackage 75.70 MB` / `.mlmodelc 75.70 MB`, with fused `gate_up=33.55 MB` raw and `down=16.78 MB` raw, and still placed both conv banks on CPU
- Residual-spectrum follow-up also failed to justify a cheap compressed-residual story: for layer `0` / expert `0`, the cluster32 residual stayed roughly as complex as raw or slightly worse (`gate stable/r95 6.0/182 -> 7.2/196`, `up 47.4/194 -> 49.0/205`, `down 36.8/195 -> 38.6/206`), so cluster subtraction does not by itself expose an obviously lower-rank routed path to exploit
- Exact packed-array follow-up also stayed on CPU: a clustered Iverson-style expert-axis conv rewrite at cluster32 with `pack1 gate_up=33.55 MB` raw and `pack2=16.78 MB` raw compiled at `.mlpackage 75.50 MB` / `.mlmodelc 75.50 MB`, validated exactly offline, and still placed both hot convs on CPU
- Public-CoreML isolated-expert dispatch smoke is also not ANE evidence for Qwen yet: the existing single-expert artifact still sits at only `2.37 MB`, gated well numerically (`cos mean=0.992939` vs exact slice), and a same-model fanout-4 Swift smoke was cheap (`0.133 ms` concurrent), but the isolated artifact's compute plan still placed all three hot linears on CPU. So the low wall time here cannot be promoted as residency; it is only dispatch-overhead evidence on a CPU-placed expert graph
- First direct private-runtime admissibility probe also narrows the substrate options: [emilio/conv-ane/ane_virtual_client_probe.m](/Users/alvarovidela/Code/em2/emilio/conv-ane/ane_virtual_client_probe.m) found that `_ANEVirtualClient` exists, but the only discovered constructor `sharedConnection` returned `nil` from an unsigned/dev binary ([tmp/ane_virtual_client_probe/summary.json](/Users/alvarovidela/Code/em2/tmp/ane_virtual_client_probe/summary.json)). So if Qwen leaves public CoreML, the next credible substrate is daemon-backed `_ANEClient` / in-memory / chaining, not the direct virtual-client path as tested here
- Packaging implication now: bank size alone is no longer the missing variable for this Qwen expert family, and neither a `linear` bank, a three-bank `1x1 conv`, a fused-Gate+Up `1x1 conv`, a naive cluster-residual compression story, nor an exact clustered packed-array conv rewrite rescued residency at the same in-band sizes. The next H1 geometry has to change **execution substrate or decomposition family**, not just keep stretching or simplifying the same public-CoreML bank shape

### 64-expert banks

- Cuts routed artifacts even further to `160`.
- But `64 × 1.5 MB = 96 MB` leaves effectively no headroom for:
  - CoreML metadata
  - quantization scales / auxiliary tensors
  - compiled-format inflation
- This is the highest-risk option even though it looks best on paper by artifact count.

---

## 5. Relative Pressure vs PF-Proven Baseline

| Variant | Artifact count vs PF-proven 1024 | Byte total vs PF-proven ~2.4 GB | Main concern |
|---|---:|---:|---|
| Separate packs | ~10.1x | ~6.6x | Too many artifacts |
| 32-expert banks | ~0.39x | ~6.6x | Total resident bytes |
| 64-expert banks | ~0.23x | ~6.6x | Package cliff |

This is the key planning insight:

**32-bank packaging solves the artifact-count problem but not the total-byte problem.**

That is still the right trade because the byte-total problem exists for every strategy, while the artifact-count problem is self-inflicted if we keep separate packs for the whole model.

---

## 6. What To Measure First

These measurements should replace all estimates above before any model-wide conversion run.

1. One compiled routed expert pack: actual `.mlpackage` and `.mlmodelc` sizes.
2. One compiled 32-expert MFD bank: actual `.mlpackage` and `.mlmodelc` sizes.
3. One compiled 64-expert MFD bank: size only; do not promote unless clearly below the cliff after compilation.
4. One-layer cold load time for:
   - 256 separate packs
   - 8 banks of 32
5. One-layer warm latency for:
   - separate packs
   - 32-banks
6. ANE residency check on 32-bank and packed executed-op variants.

---

## 7. Recommended Decision Rule

Use this order:

1. **Separate packs** for first expert and one-layer correctness proof.
2. **32-expert MFD bank** was the first deployment candidate, but the measured artifact fell back to CPU and is now rejected in this exact shape.
3. **Packed executed-op probes** at `G=24` and `G=32` are also rejected in their current form because both remained `CPU/CPU/CPU`.
4. The next candidate must change executed geometry or decompose the expert linears before any larger bank is attempted.
5. If low-rank is the chosen decomposition, do not start with a single-expert split chain; start with a shared-basis or packed factor bank that keeps at least one executed path inside the proven ANE window.
6. Do not spend more cycles on direct PF-style operator-family transfer (`linear` vs `conv`) without changing the Qwen expert decomposition itself.

Operationally, the first full-model attempt should be:

**banked 32-expert MFD + CPU router + public CoreML dispatch**

not

**10,240 separate expert artifacts**.

---

## 8. Bottom Line

All three variants pay the same routed-expert byte budget: **15.36 GB**.

What changes is where the risk lands:

- separate packs risk **too many artifacts**
- 32-banks risk **byte-total only**
- 64-banks risk **the package cliff itself**

That means the original 32-expert-bank plan was directionally right on package budget and wrong on executed geometry. The active blocker is no longer “can we fit under the package cliff?” but “what expert linear geometry actually lands on ANE?”