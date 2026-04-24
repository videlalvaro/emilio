# Qwen3.6-35B-A3B → Apple Neural Engine Execution Plan

**Date**: 2026-04-23  
**Target**: `Qwen/Qwen3.6-35B-A3B` on M4 Max 48 GB, ANE-first, public CoreML mainline  
**Prerequisite already proven**: privacy-filter ANE chain on public CoreML (`8L × 128 experts`, per-expert dispatch, safe-norm, 1024 cached artifacts, task metric preserved)  
**Headline hypothesis**: Qwen3.6 is plausible on ANE because it is a `35B / 3B-active` MoE; the real blockers are not total parameter count but (1) DeltaNet on ANE, (2) expert artifact count, and (3) package-size cliffs.

---

## 1. Goal, Non-Goals, Success Criteria

### Goal

Make **greedy decode** on Qwen3.6 possible on ANE using a path that can actually ship:

- public CoreML first
- CPU router + ANE experts
- ANE for both full-attention and DeltaNet if feasible
- exact or near-exact greedy agreement with HuggingFace on a small prompt suite

### Non-goals for v0

- 32K context
- private `_ANEClient` / chaining SPI as a required dependency
- single-graph dynamic expert routing
- perfect logits everywhere before task-level quality is acceptable

### Success criteria

| Stage | Gate |
|---|---|
| Component proof | expert/shared/router cos ≥ 0.99 vs PyTorch golden |
| Attention proof | full-attention layer cos ≥ 0.99, ANE residency |
| DeltaNet proof | decode-step cos ≥ 0.99, prefill cos ≥ 0.97, ANE residency |
| 4-layer repeat unit | hidden-state cos ≥ 0.97 through layers 0..3 |
| Full model | final top-1 ≥ 95% vs HF on prompt suite |
| Performance | greedy decode ≥ 20 tok/s at 1K ctx on M4 Max |
| Productability | public CoreML path works without private entitlements |

---

## 2. Hard Facts We Must Respect

### Model facts

| Property | Value |
|---|---|
| Layers | 40 |
| Hidden dim | 2048 |
| Pattern | `[linear, linear, linear, full] × 10` |
| Linear-attention layers | 30 × GatedDeltaNet |
| Full-attention layers | 10 × GatedAttention |
| MoE | 256 routed experts (top-8) + 1 shared per layer |
| Active params/token | ~3B |
| Quant target | INT4 for experts; fp16 activations; attention quant TBD |

### ANE laws from verified work

- **No in-graph dynamic routing.** Router must stay on host.
- **Do not exceed the ~96 MB compiled-package cliff.** This is per package, not total cached weights.
- **Gather-on-weights is dead.** PF proved real MoE gather falls to CPU and loses badly.
- **Safe-norm is mandatory anywhere we square large fp16 activations.** This applies to RMSNorm and likely L2-normalization in DeltaNet.
- **Public CoreML per-expert dispatch already works.** PF reached `2.15 ms/layer` under real routing, `35.2×` vs dense.
- **Public CoreML stateful decode has a backend landmine.** In the dense stateful Qwen driver, `MLState` plus `.cpuAndNeuralEngine` produced NaN or garbled output across the tested seq-len range, while `.all` was the only correct public-CoreML option observed. Treat any stateful public-CoreML throughput claim as compute-unit-specific, and do not assume `.cpuAndNeuralEngine` is a valid decode path for Qwen attention / DeltaNet state until disproven.
- **Cache law proven only to 1024 artifacts / ~2.4 GB.** Qwen3.6 will exceed that unless we reduce artifact count.

### Book-driven tactics to keep

| Tactic | Book source | Why it matters here |
|---|---|---|
| Static chunk unrolling for DeltaNet prefill | Dragon Book, optimization / loop unrolling | Turns sequential chunk logic into fixed-shape ANE graphs |
| Treat expert dispatch as an array program, not scalar branching | Iverson, *A Programming Language* | Bank experts, route by index arrays, reduce on host |
| Reduce top-8 expert outputs as an associative combine | Stepanov/McJones, semigroups | Use tree-style weighted accumulation, not ad hoc serial glue |

---

## 3. Recommended Architecture

### 3.1 Mainline path: public CoreML only

```text
embed + lm_head          → CPU
router                   → CPU
shared expert            → ANE
routed experts           → ANE (banked or separate)
GatedAttention           → ANE
GatedDeltaNet            → ANE if phase gates pass
residual adds / scatter  → CPU
```

This is the only path that counts as “possible” for the main result.

### 3.2 Expert artifact strategy: do not start with 10,240 separate packs

We need an explicit artifact-count plan before any full-model conversion.

See the companion budget sheet: [QWEN36_ANE_ARTIFACT_BUDGET.md](/Users/alvarovidela/Code/em2/python/moe/QWEN36_ANE_ARTIFACT_BUDGET.md).

#### Option A — Separate per-expert `.mlmodelc`

- **Pros**: lowest engineering risk; PF already validated this pattern end-to-end
- **Cons**: `40 × 256 = 10,240` expert artifacts, which is far beyond the currently proven cache law
- **Use**: baseline for correctness and first one-layer MoE proof

#### Option B — Banked `MLMultiFunctionDescriptor` packs

- Group **32 experts per pack**
- Raw weight estimate is `32 × 1.5 MB ≈ 48 MB INT4`, but the first real compiled Qwen bank measured **75.60 MB**
- Per layer: `256 / 32 = 8` expert banks
- Full model: `40 × 8 = 320` expert-bank artifacts instead of `10,240`
- The first real Qwen MFD bank stayed below the package cliff but its executed expert function still fell back to CPU, so this exact bank shape is **not** currently a promotable ANE path

#### Decision rule

- Start with **separate packs** for one-layer proof
- Immediately test **32-expert banked MFD**
- Promote banked MFD to mainline only if it stays on ANE and preserves cosine
- Current measured result: the first Qwen MFD-32 bank compiled at **75.60 MB** but failed ANE residency, so do **not** promote this exact shape
- Current measured result: packed executed-op probes at **G=24** (`56.75 MB`) and **G=32** (`75.67 MB`) also fell back to CPU, so the present expert-packing branch is blocked pending an executed-geometry change
- Do **not** try 64 experts/bank until a different executed geometry lands on ANE at smaller `G`

---

## 4. Phase Plan

## Phase 0 — Ground Truth and Planner

**Purpose**: produce exact goldens and replace all hand-wave estimates with measured numbers.

- [x] Extract all weights from HuggingFace to per-layer `.npz`
  - Script: `python/moe/qwen36_extract_weights.py`
- [x] Capture `qwen36_golden.npz` with:
  - Script: `python/moe/qwen36_hf_golden_capture.py`
  - test prompts
  - sampled hidden states at layer boundaries
  - router top-k indices / weights for selected tokens
  - DeltaNet decode states
  - final logits and greedy continuation
  - Prompt / path defaults: `python/moe/qwen36_phase0_spec.py`
- [x] Measure one actual expert INT4 package size after conversion
- [x] Measure one actual 32-expert MFD bank size after conversion
- [ ] Freeze prompt suite for gates: arithmetic, factual recall, short code/text prompt

**Measured artifact anchors so far**

- Single routed expert (`layer 0`, `expert 0`): `2.37 MB` `.mlpackage`, `2.37 MB` `.mlmodelc`, local cosine about `0.993`
- 32-expert MFD bank (`layer 0`, experts `0..31`): `75.59 MB` `.mlpackage`, `75.60 MB` `.mlmodelc`, executed expert path fell back to CPU
- Packed executed-op probe `G=24` (`layer 0`, experts `0..23`): `56.75 MB` compiled, `CPU/CPU/CPU`
- Packed executed-op probe `G=32` (`layer 0`, experts `0..31`): `75.67 MB` compiled, `CPU/CPU/CPU`
- Conv-authored single expert (`layer 0`, `expert 0`, `B=1`): `2.37 MB` compiled, `CPU/CPU/CPU`
- Conv-authored single expert (`layer 0`, `expert 0`, `B=64`): `2.37 MB` compiled, `CPU/CPU/CPU`

**Gate**

- HF greedy is reproducible from the saved golden file on 3 prompts
- measured package sizes replace all estimated sizes in the plan
- expert-packing branch stays blocked until an executed expert geometry reaches `ANE/ANE/ANE` on a one-layer probe

## Phase 1 — MoE Block First (highest leverage, lowest novelty)

**Purpose**: retire the part we already know how to do from PF, but at Qwen scale.

- [ ] Build one routed expert pack (`2048 → 512 → 2048`, SwiGLU, INT4)
- [ ] Build one shared expert pack + gate
- [ ] Build exact CPU router (`Linear(2048→256) → softmax → topk(8) → renormalize`)
- [ ] Validate one expert vs PyTorch golden
- [ ] Validate one full MoE block (shared + routed top-8) vs golden `mlp_delta`
- [ ] Compare two execution variants on one layer:
  - separate expert packs
  - 32-expert MFD banks

**Current status**

- Separate-function MFD banking is measured but rejected for ANE rollout: the `32`-expert bank stayed below the package cliff and still ran on CPU
- Packed executed-op probes at `G=24` and `G=32` are also rejected for ANE rollout: both remained `CPU/CPU/CPU`
- PF-style conv-as-linear authoring is also rejected for Qwen expert decode geometry: both `B=1` and `B=64` single-expert probes remained `CPU/CPU/CPU`
- The next MoE-block step is **not** more packing fan-out; it is a change in executed geometry or decomposition of the expert linears

### Architecture shortlist from books + recent MoE papers (2026-04-23)

**H1 — Cluster-shared base + routed low-rank residuals**

- Source ideas: MoBE basis sharing, MoE-I2 intra-expert low-rank decomposition, and dynamic expert clustering with hierarchical routing
- Qwen-specific form: cluster the `256` experts in a layer into small groups (`8..16`) by parameter similarity plus routing frequency, then represent each expert matrix as `W_base_group + A_i B_i^T`
- Why this is the strongest remaining branch: it changes **both** the executed kernels and the routing locality. REAP-style pruning and cache-aware routing help fanout, but they keep the failing raw expert geometry intact. After the layer-0 / expert-0 offline check below, this is now the default next branch.
- First positive signal: a cheap cluster-mean proxy on layer `0` / expert `0` beat direct raw-low-rank at every tested rank for cluster sizes `4`, `8`, and `16`, with cluster `4` clearly best
- Best measured slice so far: cluster `4` gave base-only cosine `mean=0.270200`, residual norm ratios below `1.0` on all three matrices (`gate 0.823`, `up 0.831`, `down 0.834`), and cluster+residual low-rank beat raw low-rank from rank `8` (`0.352795` vs `0.137576`) through rank `128` (`0.835300` vs `0.763524`)
- Widened offline gate: cluster `24` was still mildly positive over raw low-rank at ranks `64/128/256`, while cluster `32` was weaker but still not negative
- First packed-base CoreML probe result: cluster `24`, rank `128`, batch `1` produced packed base shapes `gate/up=(12288,2048)` and `down=(49152,512)`, raw INT4 bytes `49.55 MB`, `.mlpackage 74.58 MB`, `.mlmodelc 74.59 MB`, but **all linears still placed on CPU**
- Shared-basis routed-op rewrite: cluster `24` with basis `24` is exact offline and gives per-family raw bank size `12.58 MB`, while cluster `32` with basis `32` gives per-family raw bank size `16.78 MB`; both were converted as three-bank CoreML probes and **both still placed gate/up/down on CPU**
- Conv-authored rewrite of the same exact shared-basis banks at cluster `32` / basis `32` also failed residency: the full three-bank `1x1` conv graph compiled at `.mlpackage 75.81 MB` / `.mlmodelc 75.82 MB`, but **all three hot conv banks still placed on CPU**
- Minimal three-conv isolator of that same `cluster32/basis32` geometry also failed residency: keeping only the three hot `1x1` conv banks and removing coefficient mixing plus per-slot recombine still compiled at `.mlpackage 75.67 MB` / `.mlmodelc 75.67 MB`, and **all three conv banks still placed on CPU**
- Fused Gate+Up rewrite also failed residency on the same geometry: a two-bank `1x1` conv graph with fused `gate_up` (`33.55 MB` raw) plus `down` (`16.78 MB` raw) compiled at `.mlpackage 75.70 MB` / `.mlmodelc 75.70 MB`, and **both conv banks still placed on CPU**
- Standalone per-expert dispatch is also blocked as the next branch gate: the existing one-expert package is only `2.37 MB` total, and each expert linear is only about `0.52 MB` raw INT4, so a one-expert residency probe is already below the empirical ANE landing band and cannot be a discriminating check for PF-style per-expert dispatch
- Bounded public-CoreML per-expert dispatch smoke also closes as **CPU-only evidence** for Qwen: a new isolated-artifact gate on `qwen36_L00_expert000_int4` validated the math (`cos mean=0.992939`, `min=0.991351`, `max_abs=8.57e-05`) and a Swift same-model fanout-4 smoke was cheap (`single 0.088 ms`, `seq 0.364 ms`, `conc 0.133 ms`), but the isolated expert `MLComputePlan` still placed all three hot linears on `CPU/CPU/CPU`. So low dispatch wall time at this size does **not** clear ANE residency, and distinct-expert scale-out remains blocked as an ANE-first branch on the current public-CoreML substrate
- Residual-spectrum follow-up weakens the easy compressed-residual story: for layer `0` / expert `0`, subtracting a cluster-mean base improved output approximation earlier, but the residual spectra did **not** become materially lower-rank. At cluster `32`, raw vs residual was `gate stable/r95 6.0/182 -> 7.2/196`, `up 47.4/194 -> 49.0/205`, `down 36.8/195 -> 38.6/206`
- Exact packed-array expert-axis rewrite also failed on the first bounded slice: the new clustered Iverson conv probe at layer `0` / anchor expert `0` / cluster `32` packed real expert weights directly over the expert axis, kept the hot ops in-band (`pack1 gate_up 33.55 MB`, `pack2 16.78 MB` raw INT4), validated exactly offline (`cos_mean ~ 1.0`), compiled at `.mlpackage 75.50 MB` / `.mlmodelc 75.50 MB`, and still placed both hot convs on CPU
- First private-runtime substrate admissibility probe also narrowed the path: [emilio/conv-ane/ane_virtual_client_probe.m](/Users/alvarovidela/Code/em2/emilio/conv-ane/ane_virtual_client_probe.m) loaded `AppleNeuralEngine.framework` and resolved `_ANEVirtualClient`, but the only discovered constructor, `sharedConnection`, returned `nil` from an unsigned/dev binary ([tmp/ane_virtual_client_probe/summary.json](/Users/alvarovidela/Code/em2/tmp/ane_virtual_client_probe/summary.json)). So direct daemon-bypass IOKit is not the immediate Qwen rescue on this machine; any private-runtime follow-up should pivot to daemon-backed `_ANEClient` / `_ANEInMemoryModel` / chaining surfaces instead of `_ANEVirtualClient`
- First daemon-backed `_ANEInMemoryModelDescriptor` probe also narrowed the private path: [emilio/conv-ane/ane_inmemory_model_probe.m](/Users/alvarovidela/Code/em2/emilio/conv-ane/ane_inmemory_model_probe.m) resolved `_ANEInMemoryModelDescriptor`, `_ANEInMemoryModel`, and `_ANEWeight`; the inline-empty MIL case created descriptor + model and reached `compileWithQoS:`, but failed in the private compiler with `com.apple.appleneuralengine.compiler` code `1` / `_ANECompiler : ANECCompile() FAILED` ([tmp/ane_inmemory_model_probe/summary.json](/Users/alvarovidela/Code/em2/tmp/ane_inmemory_model_probe/summary.json)). Weighted cases also ruled out the naive `_weights` payloads: flat `NSData` values, flat `_ANEWeight` values, and one-element array wrappers all fail inside the descriptor factory, which is sending `count` / `allValues` to entries and therefore appears to expect a more nested dictionary-like weight map. A follow-up replay of a **known-good small Qwen conv artifact** through the same path did not change that boundary: real `model.mil` (`4777` bytes) plus real `weights/weight.bin` (`2364992` bytes) still created descriptor + model and still failed at `_ANECompiler : ANECCompile() FAILED`. Feeding the packaged `coremldata.bin` and `analytics/coremldata.bin` blobs into the descriptor `optionsPlist` slot also left the boundary unchanged, and switching to `modelWithNetworkDescription:weights:optionsPlist:` with packaged `coremldata.bin` still produced the same compile failure. The next compile-options probe made one important point: `compileWithQoS:options:error:` is not ignored. `_ANEInMemoryModel` synthesizes an internal compiler-options dictionary, caller-supplied keys are merged into it, and a dummy probe key survives into that derived dictionary. But forcing the observed internal `kANEFModelType` key to `kANEFModelANECIR` on the real MIL replay is normalized back to `kANEFModelMIL` and still fails at the same compile wall. A follow-up probe using the other obvious real internal key, `kANEFCompilerOptionsFilenameKey`, also failed to move the wall: the caller-supplied alternate filename was normalized back to `compiler_options.plist`, and compile still failed at the same `ANECCompile()` boundary. A final setter-based probe bypassed that normalization and still did not help: directly calling `_ANEInMemoryModel`'s `setCompilerOptionsFileName:` changed both `compilerOptionsFileName` and the derived `kANEFCompilerOptionsFilenameKey` value to `copilot_missing_compiler_options.plist`, yet compile still failed at the same `ANECCompile()` boundary. A further compiler-owned probe using `maxModelMemorySize = 4096` also survived into the derived compiler-options dictionary unchanged and still failed at the same `ANECCompile()` boundary. Subsequent structural probes also closed the remaining obvious path branches: canonicalizing the nested weights map to one outer key plus one inner `w` payload changed the weights hash but not the failure, changing the inner key to `weights/weight.bin` normalized away entirely, and rewriting every BLOBFILE path in the replay MIL from `@model_path/weights/weight.bin` to `@model_path/w` changed the network hash but still failed at `_ANECompiler : ANECCompile() FAILED`. The new lower-surface control is also negative: direct `_ANEModel + _ANEClient` compile of both the original real `.mlmodelc` and the rewritten staged in-memory directory fails identically in `com.apple.appleneuralengine.espresso` code `-1` with `_ANEEspressoIRTranslator : error Cannot load network '.../model.espresso.net'`. So the daemon-backed in-memory path is real but not yet usable for Qwen as currently authored, and the remaining unknown is no longer toy MIL syntax, sidecar blobs, the obvious alternate descriptor entrypoint, the obvious compile-option overrides, weight-key aliasing, or `weight.bin` vs `w`; it is the hidden MIL-to-Espresso translation / staging contract upstream of `_ANEClient compileModel:options:qos:error:`.
- Readout from the full H1 bank sweep so far: enlarging only the shared base was not enough, exact shared-basis banks in the nominal ANE size window were not enough, the reduced three-conv isolate was not enough, fused Gate+Up was not enough, cluster subtraction did not make the routed residual obviously low-rank, and even an exact packed-array expert-axis rewrite over a tight cluster still fell back to CPU on the first bounded slice. Tight routed clusters are still the right H1 direction numerically, but the current public-CoreML shared-basis `linear` bank, three-bank `1x1 conv`, fused-Gate+Up `1x1 conv`, naive compressed-residual interpretations, and cluster-packed Iverson-style conv rewrite are now exhausted for this Qwen expert family; the next implementation target must change **execution substrate or decomposition family** more materially, not keep sanding the same bank graph
- Book tie:
  - Dragon Book: treat this as a legal rewrite of one bad kernel family into a different factored kernel family rather than another operator-family swap
  - Iverson: route over expert groups as an array program, not `256` scalar branches
  - Stepanov/McJones: combine the routed outputs with an associative weighted reduction, not ad hoc serial glue

**H2 — Shared expert as the dense base, routed experts as residual adapters**

- Use the already-required shared expert as the common dense path
- Approximate each routed expert as a residual adapter around that base instead of another full `2048 -> 512 -> 2048` expert
- This is more radical than H1, but it aligns best with the ANE evidence because the routed path becomes much smaller than the raw expert that keeps falling back to CPU
- Layer-0 / expert-0 offline numerics were a **weak result** for this branch: shared-only output cosine was `min=-0.046877, mean=0.022067`, residual Fro norms were larger than the expert itself (`gate 2.18x`, `up 2.02x`, `down 1.91x`), and shared+low-rank-residual underperformed direct raw-low-rank at every tested rank from `8` through `256`
- Current decision: keep H2 as a negative result unless a later **cluster-shared** base changes the residual norms materially; do not spend the next CoreML residency probe on a plain shared-expert base

**H3 — Prune and hierarchy after geometry is fixed**

- Source ideas: REAP, EAC-MoE/PESF, and cache-conditional experts
- Use these only after an ANE-resident geometry exists
- Reason: they reduce expert count, routing fanout, and cache pressure, but they do not by themselves rescue the failing single-expert kernel shape

**Explicit no-go from the gatekeeper**

- Do **not** use a single-expert two-stage low-rank split as the next ANE gate
- For the Qwen expert shape, the largest factor from a rank-`r` split has raw INT4 bytes `2048 * r * 0.5 = 1024r`
- Reaching the empirical lower landing band of about `12 MB` would require `r ≈ 11,700`, but the matrix max rank is only `512`
- So single-expert low-rank factorization can still be useful for numerics work, but it cannot be the right residency probe
- If low-rank is the next branch, the first admissible probe is a **shared-basis or packed factor bank** whose executed base path stays in the proven ANE weight window

**Gate**

- single expert cos ≥ 0.99
- full MoE block cos ≥ 0.99
- chosen artifact strategy is justified by measured ANE residency + load time

## Phase 2 — Full-Attention Pack (known shape, PF-derived)

**Purpose**: prove the simpler attention path before touching DeltaNet.

- [ ] Build one GatedAttention pack for a full-attention layer
- [ ] Include safe RMSNorm from day one
- [ ] Implement partial RoPE exactly as HF does it
- [ ] Validate both prefill and decode-mode KV behavior

**Gate**

- cos ≥ 0.99 vs golden
- ANE residency confirmed on representative shape

## Phase 3 — DeltaNet Decode Step (highest risk, smallest representative shape)

**Purpose**: prove the one operation family that makes or breaks Qwen3.6-on-ANE.

Start with **decode**, not prefill. The decode step is the smallest shape, the most reusable for generation, and the cheapest ANE-residency gate.

- [ ] Build one decode-step DeltaNet pack for `T=1`
- [ ] Inputs: hidden state, conv state, recurrent state
- [ ] Outputs: next hidden state, next conv state, next recurrent state
- [ ] Apply safe scaling to every squared-reduce / L2-normalization path
- [ ] If a monolithic pack misses ANE residency, split by head groups or state shards before attempting full prefill

**Gate**

- single-step cos ≥ 0.99
- 10-step decode drift stays within cos ≥ 0.97
- ANE residency holds on the chosen decomposition

**Kill switch**

- If DeltaNet decode cannot be made ANE-resident with a reasonable pack split, stop here and pivot the project claim to **“Qwen3.6 MoE-on-ANE with DeltaNet on CPU”** as an interim result.
- Do **not** start 40-layer artifact generation before this gate passes.

## Phase 4 — DeltaNet Prefill

**Purpose**: convert the recurrent/chunked formulation into a fixed-shape graph.

- [ ] Build one prefill DeltaNet pack for `T=128`
- [ ] Use static chunk unrolling (`chunk_size=64`) rather than any dynamic loop
- [ ] Validate chunk outputs and final layer output vs golden

**Gate**

- prefill cos ≥ 0.97
- ANE residency confirmed

## Phase 5 — 4-Layer Repeat Unit

**Purpose**: prove the smallest slice that exercises the real pattern of the model.

Use layers `0..3`, because they cover:

- DeltaNet
- DeltaNet
- DeltaNet
- full attention
- MoE after every attention block

That 4-layer unit is the true first end-to-end milestone, not a single layer.

- [ ] Build a Swift orchestrator for layers `0..3`
- [ ] Run real routing, not synthetic top-k
- [ ] Track hidden-state cosine at each layer boundary
- [ ] Measure cold load, warm load, and steady-state decode

**Gate**

- hidden-state cos ≥ 0.97 through the 4-layer block
- top-1 agreement remains acceptable on the prompt suite
- load/warm behavior is operationally sane

## Phase 6 — Scale 4 Layers → 40 Layers

**Purpose**: replicate the proven unit without reopening architectural questions.

- [ ] Generate artifacts for all 10 repeat units
- [ ] Load all required expert banks / packs
- [ ] Manage 30 DeltaNet states + 10 KV caches
- [ ] Validate full forward pass on the golden suite
- [ ] Validate greedy decode on the same prompts used in Phase 0

**Gate**

- final top-1 ≥ 95% vs HF
- greedy output stays coherent on the prompt suite
- memory high-water mark fits comfortably on 48 GB

## Phase 7 — Throughput and Energy

**Purpose**: measure the result only after correctness is proven.

- [ ] Prefill latency at `T=128`
- [ ] Decode tok/s at 1K context
- [ ] Energy vs CPU / Metal baseline

**Gate**

- decode ≥ 20 tok/s at 1K context
- ANE path uses less energy than CPU path

---

## 5. Immediate Next 10 Tasks

1. Extract one full layer of Qwen3.6 weights into a stable `.npz` schema.
2. Capture one golden prompt with saved router top-k and DeltaNet state.
3. Convert one routed expert to INT4 and compile it.
4. Validate that single expert against PyTorch.
5. Convert one 32-expert MFD bank and check compiled size.
6. Run ANE residency on that bank.
7. Build shared expert + gate.
8. Build exact CPU router and one-layer MoE harness.
9. Build one full-attention pack and validate it.
10. Build one DeltaNet decode-step pack and treat that as the first true go/no-go test.

If step 10 fails, do not start a model-wide conversion campaign.

---

## 6. Main Risks and Mitigations

| Risk | Why it matters | Mitigation |
|---|---|---|
| DeltaNet decode misses ANE | This kills the “all-ANE core” claim | Prove decode first; split by head group/state shard before giving up |
| 10k artifact cache does not scale | PF only proved 1024 artifacts | Use 32-expert banked MFD as the default compaction path |
| fp16 reductions overflow | PF already hit this exact class of bug | Apply safe scaling before every squared reduce from day one |
| CPU lm_head becomes the bottleneck | Generation may become output-head bound | Accept CPU head for v0; optimize only after full-model correctness |
| Load time becomes absurd | Cold start can bury the headline | Measure cold and warm separately; parallel load; bank experts |

---

## 7. Agent Workflow and Gating Discipline

Use this exact order for every substantial phase:

1. `historian` — record intent.
2. `optimality-gatekeeper` — approve any long conversion, calibration, or benchmark.
3. `compiler` — build packs.
4. `ane-validator` — smallest representative shape first.
5. `golden-validator` — compare against saved reference.
6. `tester` — run focused regression checks.
7. `energy-bencher` — only after correctness passes.
8. `doc-writer` + `historian` — record what actually happened.

No full-model conversion should begin until **Phases 1, 2, 3, and 5** are green.

---

## 8. Bottom Line

Qwen3.6 on ANE is **plausible**, but only if we treat it as three separate problems in this order:

1. **MoE dispatch at Qwen scale** — likely solvable, because PF already proved the pattern.
2. **GatedAttention on ANE** — likely solvable, because PF attention is already close in structure.
3. **DeltaNet decode on ANE** — the real frontier and the only phase that deserves a hard kill switch.

The right headline path is therefore:

**public CoreML + CPU router + banked INT4 experts + ANE attention + ANE DeltaNet (if Phase 3 passes).**

That is the shortest path to making Qwen3.6 on ANE real rather than merely imaginable.
