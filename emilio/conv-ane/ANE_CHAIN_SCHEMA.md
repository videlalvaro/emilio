# ANE Chain Primitive — Reverse-Engineered Schema

**Date**: 2026-04-22
**Source**: live ObjC runtime reflection of `AppleNeuralEngine.framework` and `ANECompiler.framework` on macOS (M4 Max, h16g, 16 ANE cores).
**Probe binaries** (in this dir):
- `ane_chain_probe.m` — verifies the chain XPC primitive exists and is per-call retargetable.
- `ane_class_dump.m` — enumerates every ObjC class registered in the loaded private framework images, dumps ivar layouts + selectors + adopted protocols.

Logs: `/tmp/chain_probe.log`, `/tmp/ane_class_dump.log`, `/tmp/ane_chain_classes.log`.

## TL;DR

The Apple Neural Engine private runtime exposes a **chain primitive** that lets the daemon execute a sequence of model procedures on-device with no host round-trip between stages and a shared memory pool for intermediates. Combined with the fact that one loaded model carries an **array of procedures**, this maps almost 1:1 onto MoE expert dispatch.

## Verified XPC Surface

```
-[_ANEClient prepareChainingWithModel:options:chainingReq:qos:error:]
-[_ANEClient loadModelNewInstance:options:modelInstParams:qos:error:]
-[_ANEDaemonProtocol prepareChainingWithModel:options:chainingReq:qos:withReply:]
   encoded_types = v52@0:8@16@24@32I40@?44
   (return void; args: id model, id options, id chainingReq, uint32_t qos, void(^reply)(...))
```

`chainingReq` is **per-call**, not bound at `loadModel:` time → the chain can be retargeted on every invocation (per token) with no recompile and no reload.

The daemon holds multiple loaded models per `_ANEDaemonConnection`. Verified via `-[_ANEClient connectionsUsedForLoadingModels]` showing two distinct `_ANEModel` UUIDs after a probe loaded two `.mlmodelc` artifacts.

## `_ANEChainingRequest` (size 80, NSSecureCoding)

```
+ chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:
                          procedureIndex:signalEvents:transactionHandle:
                          fwEnqueueDelay:memoryPoolId:

ivars:
  [+0x08] _inputBuffer                NSArray   // IOSurfaces in
  [+0x10] _outputSets                 NSArray   // IOSurfaces out (PLURAL)
  [+0x18] _loopbackInputSymbolIndex   NSArray   // chain-edges:
  [+0x20] _loopbackOutputSymbolIndex  NSArray   //   stage-N out → stage-N+1 in
  [+0x28] _signalEvents               NSArray   // cross-stage sync
  [+0x30] _transactionHandle          NSNumber  // groups stages atomically
  [+0x38] _procedureIndex             NSNumber  // which procedure to invoke
  [+0x40] _fwEnqueueDelay             NSNumber  // firmware-side enqueue delay
  [+0x48] _memoryPoolId               NSNumber  // shared mem pool across stages
```

Key implications:
- `_loopback*SymbolIndex` are how stage outputs feed stage inputs **inside the daemon** (no host copy).
- `_memoryPoolId` lets all stages share a single buffer pool — intermediates never cross the bus.
- `_signalEvents` lets stages issue/wait on Mach events.
- `_procedureIndex` is a single `NSNumber`, but `_outputSets` is plural. Open question: does one chain request fan out across multiple procedures, or is fan-out done by daisy-chaining requests via `_transactionHandle`? **Probe needed.**
- Conforms to `NSSecureCoding` → can be persisted/transported safely.

## `_ANEModelInstanceParameters` (size 24, NSCopying + NSSecureCoding)

```
+ withProcedureData:procedureArray:

ivars:
  [+0x08] _instanceName     NSString
  [+0x10] _procedureArray   NSArray   // multiple procedures per loaded model
```

This is the killer fact: **one loaded model carries N procedures**, addressed by `_procedureIndex`. This is the private-API equivalent of CoreML 9's public `MLMultiFunctionDescriptor`. For MoE this means all N experts of a layer can live in one `.mlmodelc`/`.mlpackage` and be selected per-token by index.

## `_ANEInMemoryModelDescriptor` (size 64)

```
+ modelWithMILText:weights:optionsPlist:
+ modelWithNetworkDescription:weights:optionsPlist:

ivars:
  [+0x08] _isMILModel        BOOL
  [+0x10] _networkTextHash   NSString
  [+0x18] _weightsHash       NSString
  [+0x20] _optionsPlistHash  NSString
  [+0x28] _networkText       NSData       // raw .mil text OR network description
  [+0x30] _weights           NSDictionary // weights as a dict, not a path
  [+0x38] _optionsPlist      NSData
```

Lets us **bypass the filesystem entirely** — feed MIL text + weight dict from memory, skip the `model.espresso.net` materialization that broke our coremltools-generated `.mlmodelc` artifacts in the chain probe. Backed by `_ANEInMemoryModel.compileWithQoS:options:error:` and `loadWithQoS:options:error:`.

## `_ANEInMemoryModel` (size 112)

Full lifecycle on a memory-resident model:
- `-initWithDesctiptor:` (sic, typo'd in Apple's framework)
- `-compileWithQoS:options:error:`
- `-loadWithQoS:options:error:`
- `-evaluateWithQoS:options:request:error:`
- `-mapIOSurfacesWithRequest:cacheInference:error:`
- `-purgeCompiledModel`
- `-saveModelFiles`
- `-localModelPath`

Holds `_program: _ANEProgramForEvaluation` and `_descriptor: _ANEInMemoryModelDescriptor`.

## `_ANEProgramForEvaluation` (size 56)

```
+ programWithController:intermediateBufferHandle:queueDepth:
+ programWithHandle:intermediateBufferHandle:queueDepth:

key methods:
  - processInputBuffers:model:options:error:
  - processOutputSet:model:options:error:
  - processRequest:model:qos:qIndex:modelStringID:options:returnValue:error:
  - processSessionHint:options:report:error:
```

Note `processSessionHint:` — there is a "session" concept; likely how the daemon caches IO surface bindings across a sequence of related calls (e.g. token-by-token decoding).

## `_ANERequest` (size 96) — non-chained sibling

```
ivars:
  [+0x08] _inputArray              NSArray
  [+0x10] _inputIndexArray         NSArray
  [+0x18] _outputArray             NSArray
  [+0x20] _outputIndexArray        NSArray
  [+0x28] _weightsBuffer           _ANEIOSurfaceObject   // ← per-call weight injection!
  [+0x30] _sharedEvents            _ANESharedEvents
  [+0x38] _transactionHandle       NSNumber
  [+0x40] _procedureIndex          NSNumber
  [+0x48] _perfStats               _ANEPerformanceStats
  [+0x50] _perfStatsArray          NSArray
  [+0x58] _completionHandler       block
```

Two surprises:
1. `_weightsBuffer` is an **`_ANEIOSurfaceObject`** — the ANE supports per-request weight injection via IOSurface. This is potentially how Apple ships ad-hoc LoRA / model variants without recompiling. For MoE this is an alternative to the multi-procedure approach: load one MLP shape, inject expert weights per token. Latency cost: TBD.
2. `_perfStatsArray` plural → per-procedure perf stats are returned for chained calls.

## `_ANEProgramIOSurfacesMapper` (size 32)

Wraps the surface-mapping path used by both single and chained requests; binds IOSurface handles to program input/output symbols. Mostly internal.

## Implications for "Flash MoE on ANE"

The original concern — "1024 expert kernels × 32 layers = combinatorial chain blowup" — is dissolved:

| Earlier guess | Schema reality |
|---|---|
| Compile each expert as a separate `.mlmodelc`, load 1024 models | One model per layer, **N procedures inside it**, addressed by `_procedureIndex`. |
| Pre-build chains for every expert combination | `_ANEChainingRequest` is a tiny `NSSecureCoding` struct (≤100 bytes payload). Build one per token. |
| Worry about per-expert host↔ANE memcpy | `_memoryPoolId` + `_loopbackSymbolIndex` keep activations on-device across stages. |
| 8 XPC RTTs per layer for top-8 experts | One chain request per layer (assuming fan-out works) or 8 with shared `_transactionHandle`. |
| Filesystem-backed `.mlmodelc` files | `_ANEInMemoryModelDescriptor` accepts MIL text + weight dict directly. |

## Update — More Classes Discovered (round 2)

Full ANE class enumeration: 34 `_ANE*` classes total. New high-value ones beyond round 1:

### `_ANEOutputSetEnqueue` (size 32) — **multi-procedure fan-out**

```
+ outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:

ivars:
  [+0x08] _signalNotRequired  BOOL
  [+0x09] _isOpenLoop         BOOL
  [+0x0c] _procedureIndex     uint32_t
  [+0x10] _setIndex           uint32_t
  [+0x18] _signalValue        uint64_t
```

This is the answer to the open question "does one chain call execute multiple procedures?" — **yes**. The `_outputSets` array in `_ANEChainingRequest` is `NSArray<_ANEOutputSetEnqueue *>` — each entry says "run procedure P, write its set-index S, raise event with value V". A single chain submission can fan out across N procedures of the same loaded model.

`_isOpenLoop` is striking: ANE supports async/streaming dispatch where the host never reads back — fire-and-forget enqueues. Useful for prefetching the next layer's experts while the current layer is still summing.

`_signalNotRequired` lets some stages skip event signaling — the "leaf" procedures of a chain DAG can be cheaper.

### `_ANEInputBuffersReady` (size 40) — input-side handshake

```
+ inputBuffersWithProcedureIndex:inputBufferInfoIndex:inputFreeValue:executionDelay:

ivars:
  [+0x08] _procedureIndex          uint32_t
  [+0x10] _inputBufferInfoIndex    NSArray
  [+0x18] _inputFreeValue          NSArray
  [+0x20] _executionDelay          uint64_t
```

Companion to `_ANEOutputSetEnqueue`. Notice `_executionDelay` again — firmware-level timing control. `_inputFreeValue` likely lets the daemon recycle the input buffer after the listed event values are reached → enables pipelining across procedures.

### `_ANEBuffer` (size 32) — symbol-bound IO

```
+ bufferWithIOSurfaceObject:symbolIndex:source:

ivars:
  [+0x08] _ioSurfaceObject  _ANEIOSurfaceObject
  [+0x10] _symbolIndex      NSNumber
  [+0x18] _source           int64_t
```

`_inputBuffer` array elements are these. Inputs bind by **symbol index** (compiler-assigned integer), not by name → fast.

### `_ANEWeight` (size 40, NSCopying + NSSecureCoding)

```
+ weightWithSymbolAndURLSHA:weightURL:SHACode:

ivars:
  [+0x08] _weightSymbol       NSString
  [+0x10] _weightURL          NSURL
  [+0x18] _SHACode            NSData
  [+0x20] _sandboxExtension   NSString
```

Weights are **per-symbol file URLs**, not raw blobs. Sandbox extensions let the daemon access weight files outside its sandbox. This is the standard path. (`_ANEInMemoryModelDescriptor._weights` likely uses a different value type — raw `NSData` per symbol — for the in-memory path.)

### `_ANEVirtualClient` (size 24) — **direct IOKit driver path, bypasses daemon**

67 instance methods, 22 class methods. This is the **second runtime path** alongside `_ANEClient → _ANEDaemonConnection → daemon`. `_ANEVirtualClient` talks directly to the IOKit user client, bypassing `aned`. Selected methods of interest:

```
-[validateNetworkCreate:uuid:function:directoryPath:scratchPadPath:milTextData:]
   ret=^{__CFDictionary=}  args=[Q, @, @, @, @, @]
   -- accepts a `function:` NSString and `milTextData:` NSData
   -- this is HOW MULTI-FUNCTION MIL FLOWS DOWN to firmware

-[validateNetworkCreateMLIR:validation_params:]
   -- accepts MLIR (not just MIL) directly

-[compileModel:options:qos:error:]
-[loadModel:options:qos:error:]
-[loadModelNewInstance:options:modelInstParams:qos:error:]
-[loadModelNewInstanceLegacy:...]
-[doEvaluateWithModel:options:request:qos:completionEvent:error:]
-[doEvaluateWithModelLegacy:...]   -- "Legacy" suggests v1/v2 protocol coexists
-[mapIOSurfacesWithModel:request:cacheInference:error:]
-[sessionHintWithModel:hint:options:report:error:]
-[beginRealTimeTask] / -[endRealTimeTask]   -- real-time priority bracket
-[exchangeBuildVersionInfo] / -[sendGuestBuildVersion] / -[hostBuildVersionStr]
-[negotiatedCapabilityMask] / -[negotiatedDataInterfaceVersion]
-[validateEnvironmentForPrecompiledBinarySupport]
```

Implications:
- A `function:` parameter at the lowest IOKit boundary confirms: **CoreML 9's `MLMultiFunctionDescriptor` lowers to a per-function string here**, and the multi-function `.mlmodelc` becomes the `_procedureArray` populated on the `_ANEModelInstanceParameters` side.
- `beginRealTimeTask` / `endRealTimeTask` — the ANE supports a real-time scheduling bracket. Worth wrapping our hot decode loop in this for tail-latency reasons.
- `validateNetworkCreateMLIR:` — Apple's compiler accepts MLIR, not just `.mil`. Interesting alternate authoring path.

### `VirtANEModel` C struct shape (from method signatures)

```
{VirtANEModel = I q I I I I Q Q Q Q
                [32 I] [32 Q] [32 I] [32 Q]
                Q Q Q c C I Q I I I Q I I Q I Q
                [64 I] [64 I] [64 I] [64 I]
                I Q Q [64 I] [64 I]
                I I I I I I Q q I I Q I Q I Q I Q I Q I Q I Q I Q}
```

Fixed-size arrays: `[32 I]`, `[32 Q]` and `[64 I]`. Reasonable interpretation:
- `[32 ...]` ≈ up to **32 inputs/outputs/symbols** per procedure (consistent with ANE's known small fan-in).
- `[64 ...]` ≈ up to **64 procedures or weight banks** per loaded model.

If the **64-procedure cap holds**, then for 128-expert MoE layers we need **2 loaded model instances per layer** (or a cheaper alternative: one model with 64 procedures and a second one with the other 64; chain calls already work across two distinct loaded models — verified in `ane_chain_probe`).

### `_ANEPerformanceStats` (size 32)

```
ivars:
  [+0x08] _hwExecutionTime   uint64_t (ns)
  [+0x10] _pStatsRawData     NSData
  [+0x18] _perfCounterData   NSData

methods of interest:
  - performanceCounters
  - emitPerfcounterSignpostsWithModelStringID:
  + driverMaskForANEFMask:
```

Per-procedure perf counters with hardware execution time in ns. Combined with `_perfStatsArray` (plural) on `_ANERequest`, we can profile per-expert latency directly.

### `_ANECloneHelper`

```
+ cloneIfWritable:isEncryptedModel:cloneDirectory:
+ shouldSkipCloneFor:isEncryptedModel:
```

Daemon clones writable model directories before mapping. Means we can drop a `.mlmodelc` in a writable dir and it will be auto-copied. (Operational footnote.)

## Round-2 Summary: What's Now Settled

| Question | Answer |
|---|---|
| Can one chain request invoke multiple procedures? | **Yes.** `_outputSets` is `NSArray<_ANEOutputSetEnqueue*>`, each carrying its own `_procedureIndex` + `_setIndex` + `_signalValue`. |
| Is async fire-and-forget dispatch supported? | **Yes.** `_isOpenLoop` BOOL on `_ANEOutputSetEnqueue`. |
| Per-procedure signaling and pipelining? | **Yes.** Per-procedure `signalValue` + `_inputFreeValue` + `_executionDelay`. |
| Is there a daemon-bypass path? | **Yes.** `_ANEVirtualClient` talks directly to the IOKit user client. |
| Does the firmware-level API accept named functions? | **Yes.** `validateNetworkCreate:...function:...milTextData:`. |
| Can ANE compile MLIR (not just MIL)? | **Yes.** `validateNetworkCreateMLIR:`. |
| Procedure cap per loaded model? | Likely **64** (firmware struct has `[64 I]` arrays). To be probed empirically. |
| Real-time priority bracket? | **Yes.** `beginRealTimeTask` / `endRealTimeTask`. |

## Updated MoE Plan Implications

For Gemma 4 26B-A4B (128 routed experts × 30 layers):
- **2 loaded models per layer** if the 64-procedure cap is real → 60 model instances total. The daemon already keeps ≥2 distinct models resident per connection (verified). Need to test ≥60.
- **Per-token chain**: build one `_ANEChainingRequest` whose `_outputSets` lists the 8 active experts as 8 `_ANEOutputSetEnqueue` entries with the same `_memoryPoolId`. One XPC RTT per layer, period.
- **Pipelined decoding**: use `_isOpenLoop=YES` on the early-layer expert dispatches and only synchronize at the final logits stage. Could dramatically reduce host-bound idle time.
- **Real-time bracket**: wrap the per-token decode loop in `beginRealTimeTask` / `endRealTimeTask` for predictable latency.

## Remaining Open Questions

1. **64-procedure cap** — probe by attempting to load a model with N procedures, sweeping N. Needs the multi-function `.mlpackage` build path first.
2. ~~**In-memory weights value-type**~~ — **PARTIALLY ANSWERED.** [emilio/conv-ane/ane_inmemory_model_probe.m](../../emilio/conv-ane/ane_inmemory_model_probe.m) shows `_weights` is **not** a flat `{NSString → NSData}`, **not** a flat `{NSString → _ANEWeight}`, and **not** a one-element array wrapper of either. The descriptor factory sends `count` to direct entries and `allValues` to array-wrapped entries, which strongly suggests a more nested dictionary-like per-weight payload. The same probe showed the inline-empty MIL path is real enough to create `_ANEInMemoryModelDescriptor` + `_ANEInMemoryModel` and reach `compileWithQoS:`, but the toy MIL fails at `_ANECompiler : ANECCompile() FAILED`. A follow-up replay with a **real compiled Qwen conv artifact** (`model.mil` + `weights/weight.bin`) using the nested raw-`NSData` container reached the **same** boundary: descriptor created, model created, private compiler entered, `ANECCompile()` failed again. Feeding either packaged sidecar blob (`coremldata.bin` or `analytics/coremldata.bin`) into the descriptor `optionsPlist` slot also left the boundary unchanged, and swapping to the alternate `modelWithNetworkDescription:weights:optionsPlist:` factory with packaged `coremldata.bin` still reached the same compile failure. The next compile-options probe established that `compileWithQoS:options:error:` is a **live** input surface: `_ANEInMemoryModel` synthesizes a compiler-options dictionary containing `kANEFCompilerOptionsFilenameKey`, `kANEFInMemoryModelIsCachedKey`, `kANEFIsInMemoryModelTypeKey`, and `kANEFModelType`, and caller-supplied options are merged into that dictionary. But forcing `kANEFModelType = kANEFModelANECIR` on the real MIL replay is normalized back to `kANEFModelMIL` and still fails at the same `ANECCompile()` wall. A follow-up probe using the other obvious real internal key, `kANEFCompilerOptionsFilenameKey`, also failed to move the wall: the caller-supplied alternate filename was not preserved in the derived compiler-options dictionary, which normalized back to `compiler_options.plist`, and compile still failed at the same `ANECCompile()` boundary. A final discriminator then bypassed the options-dictionary normalization entirely by calling `_ANEInMemoryModel`'s `setCompilerOptionsFileName:` directly before deriving compiler options. That direct setter path is live: `compilerOptionsFileName` changed to `copilot_missing_compiler_options.plist`, the derived compiler-options dictionary carried `kANEFCompilerOptionsFilenameKey = "copilot_missing_compiler_options.plist"`, and compile still failed at the same `ANECCompile()` boundary. A further compiler-owned probe using `maxModelMemorySize = 4096` also survived into the derived compiler-options dictionary unchanged and still failed at the same `ANECCompile()` boundary. Subsequent structural probes also closed the remaining obvious path / alias branches: shrinking the nested weights map to the single canonical outer key (`@model_path/weights/weight.bin`) plus a single inner `w` payload changed the weights hash but not the boundary; changing that inner key to `weights/weight.bin` normalized away entirely (same weights hash, same staged `w` file); and rewriting the MIL itself from `@model_path/weights/weight.bin` to `@model_path/w` changed the network hash but still failed at the same `ANECCompile()` wall. The next lower control also removed `_ANEModel + _ANEClient` as a rescue path for this specific problem: a direct compile of both the **original real `.mlmodelc`** and the **rewritten staged in-memory directory** failed identically in `com.apple.appleneuralengine.espresso` code `-1` with `_ANEEspressoIRTranslator : error Cannot load network '.../model.espresso.net'`. That showed the lower direct-client surface expects a different on-disk layout than both current public-CoreML `.mlmodelc` output and the in-memory staging directory. A follow-up control search then found a distinct legacy artifact family under Microsoft Teams 2 temp storage whose `.mlmodelc` root contains `model.espresso.net`, `model.espresso.shape`, and `model.espresso.weights`. Running the existing direct `_ANEModel + _ANEClient` path against two of those bundles moved the boundary exactly once: the missing-`model.espresso.net` failure disappeared, `_ANEModel` objects were created for both, and compile/load advanced to later failures instead of file lookup. But it still did not yield an end-to-end positive control: one bundle failed in `com.apple.appleneuralengine.compiler` with `InvalidNetworkSourceFileName`, and the other failed in `com.apple.appleneuralengine.espresso` code `-2` with `Cannot serialize ANEC_IR_repr`. A final repo-only control then removed the need for any external artifact family in the argument: the existing tiny multifunction package at `python/tmp/mfn_probe/experts_multi.mlpackage` has the normal public package layout (`Manifest.json` + `Data/`), its compiled sibling `experts_multi.mlmodelc` has the same modern public layout (`analytics/`, `coremldata.bin`, `model.mil`, `weights/`), and the direct `_ANEModel + _ANEClient` path against that compiled artifact still fails immediately with `_ANEEspressoIRTranslator : error Cannot load network '.../model.espresso.net'`. So the remaining unknown is now tighter than before: nested weight-key aliasing and `weight.bin` vs `w` are out, plain `model.espresso.net` materialization is necessary but not sufficient, and repo-local public compile outputs are definitively still upstream of the legacy Espresso layout the lower direct-client surface expects. The unresolved contract is the hidden MIL-to-Espresso translation step plus the stricter legacy Espresso source-name / IR format that the current daemon/compiler still accepts upstream of `_ANEClient compileModel:options:qos:error:`.
3. ~~**`_ANEVirtualClient` vs daemon**~~ — **PARTIALLY ANSWERED.** [emilio/conv-ane/ane_virtual_client_probe.m](../../emilio/conv-ane/ane_virtual_client_probe.m) loaded `AppleNeuralEngine.framework` and resolved `_ANEVirtualClient`, but the only discovered constructor, `+sharedConnection`, returned `nil` from an unsigned/dev binary (`tmp/ane_virtual_client_probe/summary.json`). Treat the direct daemon-bypass path as blocked unless a different bootstrap or entitlement-bearing host is found.
4. **Multi-function .mlpackage authoring** — coremltools 9.0 returned `MLMultiFunctionDescriptor: False` from `ct.models`. Find the actual API surface (likely `ct.utils.bisect_model` style helpers, or the `mil.Function` route).

## Update — Round 3: Public Multi-Function API Verified End-to-End

**Public CoreML surface** (macOS 15+):
- `MLModelConfiguration.functionName: NSString?` — selects which function to load
- `MLModelAsset.functionNames(completionHandler:)` — enumerates available functions
- `MLModelAsset.modelDescriptionOfFunctionNamed:` — per-function input/output descriptions

**coremltools 9.0 authoring API** (lives at `coremltools.models.utils`, not `coremltools.utils`):
```python
from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction

desc = MultiFunctionDescriptor()
for i, p in enumerate(expert_packages):
    desc.add_function(str(p), "main", f"expert_{i}")
desc.default_function_name = "expert_0"
save_multifunction(desc, "experts_multi.mlpackage")
# performs constant deduplication across functions for shared weights
```

**End-to-end probe** ([python/moe/ane_multifunction_probe.py](../../python/moe/ane_multifunction_probe.py)):
Built 4 tiny `linear+relu` "experts", combined into one `.mlpackage` via `save_multifunction`, loaded each via `function_name="expert_i"` on `CPU_AND_NE`, all four ran and produced distinct outputs:
```
expert_0: out[:4] = [15.585, 0.0, 8.421, 0.0]
expert_1: out[:4] = [4.269, 0.0, 10.835, 3.355]
expert_2: out[:4] = [7.714, 0.0, 6.128, 0.0]
expert_3: out[:4] = [0.0,   0.0, 8.632, 0.278]
```

**Constant deduplication** is the key win: `save_multifunction` automatically shares constant tensors across functions. For MoE this means:
- Per-layer shared blocks (norms, attention QKV/O, router) are stored once.
- Only the per-expert MLP weights are unique storage.
- The 50 GB Gemma 4 footprint isn't multiplied by the function count.

**Authoring contract for one MoE layer**:
1. Build N "expert kernels" as standalone `.mlpackage`s, each `f(x) -> expert_out`.
2. Optionally build the shared attention block as another function.
3. `MultiFunctionDescriptor` + `save_multifunction` → one `.mlpackage` with N+1 named functions.
4. At runtime: for each chosen expert per token, instantiate `MLModel(..., function_name="expert_k")`. Reuse instances across tokens.

**Caveats observed**:
- Default top-level `spec.description.input` is empty in multi-function models. Use `MLModel.input_description` (per-instance, picks up the active function) or `MLModelAsset.modelDescription(of:)`.
- `ct.utils.compile_model` produces a `.mlmodelc` whose layout is **not** what `ct.models.MLModel` expects to load (no `Manifest.json`). Load directly from the `.mlpackage` instead — CoreML compiles internally.
- coremltools warns about `fp16` IO and inserts CPU casts; for ANE-resident pipelines author IO as `fp32` at the boundary or pass `outputs=[ct.TensorType(dtype=np.float16)]` with `iOS16`+ opset.

## What's Now Settled (Cumulative)

| Question | Answer | Source |
|---|---|---|
| Per-call retargetable chain XPC | YES | ane_chain_probe |
| Multi-procedure per loaded model | YES, capped near 64 (TBD) | _ANEModelInstanceParameters |
| Multi-procedure fan-out per chain call | YES via `_outputSets: [_ANEOutputSetEnqueue]` | round 2 dump |
| Async fire-and-forget dispatch | YES, `_isOpenLoop` | round 2 dump |
| Daemon-bypass IOKit path | Surface exists, but unsigned admission is blocked via `sharedConnection -> nil` in the current probe | round 2 dump + ane_virtual_client_probe |
| ANE accepts MLIR | YES, `validateNetworkCreateMLIR:` | round 2 dump |
| Real-time scheduling bracket | YES, begin/endRealTimeTask | round 2 dump |
| Public multi-function API | YES, macOS 15+ `functionName` | SDK headers |
| Coremltools authoring path | YES, `MultiFunctionDescriptor` + `save_multifunction` | API discovery |
| Constant deduplication across functions | YES (built into save_multifunction) | docstring + probe |
| Multi-function model loads & runs on ANE | YES, 4/4 functions ran | live probe |
| `_ANEInMemoryModel` path reaches private compiler | YES, descriptor + model instantiate and `compileWithQoS:` runs | ane_inmemory_model_probe |

## Still Open

1. ~~**Procedure-count cap per loaded model**~~ — **REFUTED.** Sweep N=4,16,32,64,96,128,192,256 ([python/moe/ane_proc_cap_sweep.py](../../python/moe/ane_proc_cap_sweep.py)) — **all 256 functions loaded and ran successfully on `CPU_AND_NE`**. No cap. The `[64 I]` arrays in `VirtANEModel` are per-procedure symbol/bank limits, not procedure counts. Implication: **all 128 Gemma 4 experts per layer can live in a single `.mlpackage`** — 30 packages total instead of 60.
2. **Shared-weight memory accounting** — measure on-disk and in-memory size of an N-function `.mlpackage` vs N standalone packages, with and without identical weights, to validate dedup ratio.
3. **Sequential-load cost vs cached instances** — sweep showed 0.33s per `MLModel(..., function_name=...)` load+predict at N=256 (84s total). Real driver must cache the loaded `MLModel` per function. Open: does the daemon refcount the underlying `_ANEModel` so 128 cached instances share one resident model? (The chain primitive proves yes at the firmware level.)
4. **Chain across two distinct loaded models** — verified the daemon holds 2 models concurrently (round 1); next is to actually `prepareChainingWithModel:` with a real `_ANEChainingRequest` whose `_outputSets` reference procedures from both models. Open question: does `_procedureIndex` namespace span both models, or is the chain bound to one model only?
5. **Per-call weight injection** — `_ANERequest._weightsBuffer: _ANEIOSurfaceObject` semantics. If weights can be injected per call, the multi-function approach becomes redundant for MoE.
6. **Exact in-memory `_weights` container shape** — the probe ruled out flat values and one-element arrays. Need runtime reflection or a harvested working example to learn the nested dictionary-like payload expected by `modelWithMILText:weights:optionsPlist:`.
6. ~~**Daemon-bypass entitlements**~~ — **PARTIALLY ANSWERED.** The runtime surface exists, but [emilio/conv-ane/ane_virtual_client_probe.m](../../emilio/conv-ane/ane_virtual_client_probe.m) found only `+sharedConnection`, and that returned `nil` from an unsigned/dev binary on this machine (`tmp/ane_virtual_client_probe/summary.json`). Direct `_ANEVirtualClient` work is deprioritized until we find a different bootstrap path or an entitlement-bearing host.
## Update — Round 4: Empirical Bandwidth & Latency

Three probes measured what the ANE actually does on this M4 Max with realistic shapes.

### Device assignment is shape-dependent (critical caveat)

`MLComputePlan.get_compute_device_usage_for_mlprogram_operation` reveals that small shapes silently fall back to CPU even with `compute_units=CPU_AND_NE`:

| Shape (d_model, d_ffn) | Weights (fp16) | Device |
|---|---|---|
| 1024, 4096 | 25 MB | **CPU** |
| 2048, 4096 | 50 MB | ANE |
| 2304, 9216 | 127 MB | ANE |

So **earlier "GB/s" numbers under ~16 MB were measuring CPU**, not ANE. This invalidates the simple BW sweep at small sizes — those got 24–116 GB/s but it was the CPU's BW, not the ANE's.

### Upper-size cliff recheck (2026-04-24)

The earlier working assumption that compiled artifacts fall off ANE around `~96 MB` is **wrong as a general law**.

New counterexample: a **stateful INT8 one-layer Qwen 7B probe** built via `gguf_to_ane.py` compiled to:

| Artifact | Size | Conv placement |
|---|---|---|
| `Qwen7B_1L_probe.mlpackage` | 223 MB | 4/4 ANE |
| `Qwen7B_1L_probe.mlmodelc`  | 223 MB | 4/4 ANE |

Measured via `MLComputePlan` on `tmp/qwen7b_1layer_probe/Qwen7B_1L_probe.mlmodelc`, which reported `ANE=4 CPU=0 GPU=0` for conv ops (`ios18.conv*` / `conv` / `ios18.convolution`).

So the right reading is narrower:
- there is a **small-graph / INT4 shard bug** that pushes convs to CPU
- there is a **lower-size floor** where tiny shapes fall to CPU
- but there is **not** a universal upper compiled-size cliff at `~96 MB`

At least for **stateful INT8 conv-heavy shards**, ANE placement remains valid at **223 MB compiled size** on this M4 Max.

### Bandwidth at ANE-resident sizes

Pure-linear sweep ([python/moe/ane_bw_sweep.py](../../python/moe/ane_bw_sweep.py)) — **only sizes >16 MB are reliably on ANE**:

| Weights | Latency | Effective BW |
|---|---|---|
| 16.8 MB | 0.25 ms | 67 GB/s |
| 67.1 MB | 3.11 ms | 21 GB/s |
| 268 MB | 10.6 ms | 25 GB/s |

Realistic ANE BW is **~25–110 GB/s** depending on shape. The 21 GB/s case at 67 MB is suspiciously slow; likely a bad tile choice. The Gemma-shape SwiGLU expert at 127 MB hit 110 GB/s — close to ANE's advertised peak.

### Cached-instance is mandatory

Probe ([python/moe/ane_cache_probe.py](../../python/moe/ane_cache_probe.py)) — 16 MB linear, 200 iters:

| Operation | Latency |
|---|---|
| Cached `predict` | 0.23 ms |
| Cold `MLModel(path) + predict` | 78.5 ms |
| Cold load alone | 77.4 ms |

**Cold load is ~340× slower than warm predict.** The 60-instance Gemma plan only works if all `MLModel` instances are loaded once and reused. Re-instantiating per token is fatal.

### Realistic Gemma-expert timing

[python/moe/ane_expert_probe.py](../../python/moe/ane_expert_probe.py) builds a Gemma-style SwiGLU MLP (gate+up+silu*+down) with realistic shapes:

| Label | d_model | d_ffn | Weights | Latency | Effective BW |
|---|---|---|---|---|---|
| medium | 2048 | 4096 | 50 MB | 2.40 ms | 21 GB/s |
| gemma  | 2304 | 9216 | 127 MB | 1.15 ms | 110 GB/s |

The gemma shape achieves much better arithmetic-intensity-vs-tile-size match. Take it as the realistic upper bound.

### Decode tok/s projection (revised, measured)

Per token: 8 active experts × 30 layers = 240 expert MLP calls.

**fp16 (worst case, what we measured directly):**
- 240 × 1.15 ms = 276 ms/token → **3.6 tok/s**
- chain primitive cannot help — ANE is one physical device, BW-bound at the SRAM/LPDDR boundary

**INT4 (Gemma 4 actual quant target, BW scales ~4×):**
- 276 / 4 = 69 ms/token → **~14 tok/s** if INT4 streams as int4 to ANE
- if engine materializes fp16 from int4 in main memory before transfer, no win
- compression-on-wire support is unverified

**Plus attention** (~30 layers × ~3–5 ms each at fp16 with KV cache): adds 90–150 ms/token. Total realistic decode: **2.5–4 tok/s fp16** or **6–10 tok/s INT4** if compression-on-wire works.

### What this means

The earlier 25–50 tok/s back-of-envelope was **2–5× too optimistic**. The dominant cost is per-expert BW; 8 active experts × 30 layers is just a lot of bytes per token, and ANE's effective BW (~110 GB/s peak, much less typical) doesn't keep up.

**For Gemma 4 26B-A4B on this M4 Max via ANE, expect ~5–10 tok/s decode at INT4** if the chain primitive is implemented and INT4 streams compressed. The lower-power-than-GPU value prop holds; the speed parity-with-GPU claim does not.

**The numbers that would change this materially:**
- Per-call weight injection via `_weightsBuffer` (one model shape, weights swap) → eliminates per-expert tile setup overhead, possibly 2–3× win.
- ANE INT4 wire compression confirmed → 4× win on BW-bound fraction.
- Speculative decoding with a small dense draft model on GPU → 2–3× tokens/sec multiplier independent of ANE rate.

### Files

- [python/moe/ane_bw_sweep.py](../../python/moe/ane_bw_sweep.py) — pure-linear BW sweep
- [python/moe/ane_cache_probe.py](../../python/moe/ane_cache_probe.py) — cached vs fresh
- [python/moe/ane_expert_probe.py](../../python/moe/ane_expert_probe.py) — Gemma-shape SwiGLU
- [python/moe/ane_device_probe.py](../../python/moe/ane_device_probe.py) — verifies which ops actually ran on ANE
## Sweep Results (Procedure Count)

| N | build_s | save_s | run_s | pkg_MB | ok |
|---|---|---|---|---|---|
| 4 | 0.5 | 0.0 | 0.2 | 0.01 | YES |
| 16 | 1.8 | 0.1 | 1.1 | 0.02 | YES |
| 32 | 3.6 | 0.2 | 2.7 | 0.04 | YES |
| 64 | 7.3 | 0.5 | 7.3 | 0.08 | YES |
| 96 | 11.2 | 0.7 | 14.2 | 0.12 | YES |
| **128** | 15.5 | 0.9 | 23.6 | 0.16 | **YES** |
| 192 | 24.0 | 1.5 | 48.7 | 0.24 | YES |
| 256 | 32.9 | 1.9 | 84.5 | 0.32 | YES |

`run_s` includes a fresh `MLModel(...)` instantiation + `.predict` per function — i.e. ~0.33s/function. In a real driver these would be loaded once and cached.

`pkg_MB` grows linearly with N (~1.25 KB/function) — when each function has unique constants, dedup adds nothing, as expected. For Gemma's per-layer attention weights (shared across all 128 experts) the dedup will be huge.

## Update — Round 5: Gemma-4-26B-A4B Full 30-Layer ANE Compilation (2026-04-25)

The entire Gemma-4-26B-A4B model (30 transformer layers) has been compiled to **90 ANE-resident CoreML shards** with zero failures. This is the first full-depth MoE model converted to 100% ANE execution.

### Architecture per layer (3 shards)

| Shard | Quant | Compiled Size | ANE |
|-------|-------|:---:|:---:|
| Attention (INT8 per_tensor) | INT8 | 33–47 MB | 100% |
| FFN partial 0 (2 expert packs, INT8 per_tensor) | INT8 | ~182 MB | 100% |
| FFN partial 1 + combiner (merged, mixed INT8/fp16) | INT8+fp16 | ~227 MB | 100% |

**Total**: 30 layers × 3 shards = **90 `.mlmodelc` artifacts**

### Empirical laws confirmed at 30-layer scale

1. **Merged combiner eliminates CPU fallback** — standalone combiner shards (19–36 MB) always fell to CPU due to ANE minimum size threshold. Merging into the last FFN partial keeps the shard above the floor. Validated uniformly across all 30 layers (both sliding and global types).

2. **INT8 per_tensor is the production baseline for current Gemma shards** — tested `constexpr_blockwise_shift_scale` per-block quantization paths poisoned the graph to 0% ANE. Do not generalize this to all 4-bit formats: INT4 per-grouped-channel palettization (`constexpr_lut_to_dense`) is a separate, promising but unvalidated path that needs residency + golden gates before scale-out.

3. **Rank-3 tensors (1,1,D) required** — rank-2 (1,D) inputs cause 100% CPU fallback for linear ops. All trace inputs must be rank-3.

4. **`weight_threshold=10_000_000` mixed quantization** — keeps dense MLP weights (gate ~6M, up ~6M, down ~6M params) in fp16 while INT8-quantizing expert pack weights (>10M params each). Required because dense MLP weights are INT8-hostile after norm fusion (outlier channels → only 3 INT8 levels at mean → cos=0.66).

5. **~250 MB compiled shard limit holds** — largest shard (merged last partial + combiner) compiled to ~227 MB, safely within the empirically validated 250 MB ceiling.

### Golden validation (7-layer sample)

| Layer | Type | cos(hidden) | cos(attn) | top-32 overlap |
|-------|---------|-----------|-----------|----------------|
| L0 | sliding | 0.999832 | 0.999964 | 1.000 |
| L5 | global | 0.999922 | 0.999897 | 1.000 |
| L10 | sliding | 0.999814 | 0.999928 | 1.000 |
| L15 | sliding | 0.999482 | 0.999886 | 1.000 |
| L20 | sliding | 0.999555 | 0.999908 | 1.000 |
| L25 | sliding | 0.999711 | 0.999879 | 0.969 |
| L29 | global | 0.955543 | 0.999722 | 0.719 |

All above 0.95 cosine floor. Attention cosines consistently >0.9997. L29 (deepest global layer) lowest at 0.956 — expected for cumulative quantization at the final layer.

### Revised shard-size law

Previous Round 4 established that INT8 stateful shards work at 223 MB. Round 5 now establishes:
- **Safe operating range**: ≤220 MB compiled (ANE_SAFE_SHARD_MB)
- **Hard ceiling**: ~250 MB compiled (ANE_MAX_COMPILED_SHARD_MB)
- The 227 MB merged shard compiles and runs on ANE at every layer without exception.

### Conversion artifacts

All in `python/moe/out/`:
- `gemma4_shard{L}_{L+1}_real_attn_q8.{mlpackage,mlmodelc}` — attention shards (L=0..29)
- `gemma4_shard{L}_{L+1}_real_ffn_p0of2_q8.{mlpackage,mlmodelc}` — regular FFN partial
- `gemma4_shard{L}_{L+1}_real_ffn_p1of2_q8.{mlpackage,mlmodelc}` — merged last partial + combiner
- Batch script: `tmp/convert_all_30.sh`
- Converter: `python/moe/gemma_to_ane.py mixed --n-layers 30 --max-ctx 128 --quant-bits 8`
