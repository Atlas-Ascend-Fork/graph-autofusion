# Scalar Source for `afir.broadcast`: Design for the MLIR Side

> 2026-11-13 · English companion of
> [afir.broadcast 的标量源](2026-11-13-afir-broadcast-scalar-source-design.md), which remains the
> source of record. Driven by the online fuzz failure of seed `EDL4VTLLGBGY5RIZOUBMFG1GMO`.
> Completes the branch that
> [AFIR Broadcast Integration](2026-09-23-afir-broadcast-design_en.md) left unmodelled.

# Introduction

## Purpose

Model the **scalar source** of `afir.broadcast` — an ASCIR `Scalar` / `IndexExpr` / `ScalarData` node
feeding a `Broadcast` node — so that it matches the first branch of legacy's `BroadcastScalar`. At the
same time, remove the MLIR verifier hard failure this shape triggers today, so that any combination
still unsupported is rejected in the capability layer and the graph falls back to legacy cleanly.

Audience: whoever implements and reviews the Autofuse MLIR takeover path (bridge / AFIR dialect /
AFIRToAscendC).

Expected output: coordinated changes in four places — the ODS type constraint, the shape verifier, the
capability decision and the lowering — plus lit / UT / E2E coverage.

## Scope

**In scope:**

- the ODS type constraint on `afir.broadcast` operand #0 and the scalar-source branch of
  `verifyBroadcastShapeImpl`;
- the capability decision and code emission of the `Duplicate` tier, including how it answers
  `queryScalarInput`;
- how a failure of this shape is classified on the bridge side (capability shortfall rather than a
  compilation error);
- lit positive and negative cases, a capability UT snapshot row, and a CPU E2E case.

**Out of scope:**

- changing legacy's `ScalarBroadcastOptimizationPass` so that it eliminates one more class of scalar
  broadcast (we do not touch legacy for MLIR's sake; legacy's current behavior is correct);
- rewriting the graph in the bridge to feed the scalar straight into the downstream `afir.truediv`
  (that departs from "take over what legacy's scheduler produced", and the broadcast may have several
  consumers);
- materializing the scalar into a single-element tile in the bridge (one more UB block, one more write
  and one more synchronization event than legacy's own output);
- modelling the second branch of `BroadcastScalar` (`is_ub_scalar`);
- completing capability for reduce / concat / transpose / matmul.

Components involved: Autofuse (bridge, AFIR dialect, AFIRToAscendC, codegen orchestration).
SuperKernel is not involved.

# Overview

## Software Overview

### Project Context

An online fuzz record (seed `EDL4VTLLGBGY5RIZOUBMFG1GMO`, attempt
`53076cc0-201a-4807-b413-bcf1fffe878b`, backend `af-mlir`, CANN 9.3.0) ended in a hard failure,
diagnosed as:

```
error: 'afir.broadcast' op operand #0 must be Tensor or MemRef of ... 32-bit float ... or target dialect shaped type, but got 'f32'
AF_MLIR_STRICT gate=lowering shortfall=no graph=autofused_mean_...
```

This diagnostic is **not** an unsupported dtype: F32 is present in the ODS element-type list
(`dialect/AFIR/IR/Ops/Broadcast.td:29`), in the AFIR dtype bit table
(`support/AfirElementTypes.h:61-68`) and in the element-type set of all four implementations. What the
constraint demands is a **shaped type** whose element type is in that list, while operand #0 is a
**bare scalar** `f32`.

The root cause chain, verified segment by segment:

1. The ASCIR graph decomposes `mean` into `sum` + `truediv`; the divisor (the number of reduced
   elements) is a `Scalar` or `IndexExpr` node, expanded by a `Broadcast` and then fed to `truediv`
   (online dump: `truediv1.x2 = broadcast.y`).
2. The bridge builds **no op, only a value** for these two node kinds:
   `bridge/AfirBuilder/AfirBuilder.cpp:1183-1203` (`handleMarkerNode`, whose comment literally reads
   "`Scalar` does not produce an op, only a value").
3. The value's type comes from `parseDtypeImpl` — the **element type itself**, not through
   `makeTensorType`: `AfirBuilder.cpp:1056`, ending in an `arith::ConstantOp` of bare scalar type
   (`:1100` float / `:1129` integer). The symbolic path of `IndexExpr` likewise ends in a bare scalar
   (`:1169` `index_cast`, `:1179-1180` `sitofp`).
4. The downstream `Broadcast` goes through the common `buildComputeNode`, taking its operand straight
   out of `valueMap` without rebuilding the type (`AfirBuilder.cpp:1428-1436`).
5. `afir.broadcast` constrains operand #0 to `AFIR_BufferizableShapedType`
   (`Ops/Broadcast.td:34-41` → `IR/AFIROpBase.td:36-38`, i.e.
   `AnyTypeOf<[TensorOrMemRef<elementTypes>, AFIR_TargetShapedType], "", "::mlir::ShapedType">`), with
   no `scalarOperand` escape hatch, so the module verification at `AfirBuilder.cpp:2110` fails.
6. That failure carries no capability-shortfall marker, so the `classify` at
   `CodegenOrchestration.cpp:446-451` returns `kFailed` because `sawShortfall() == false` — a hard
   failure with no fallback, matching the log's `gate=lowering shortfall=no` term for term.

The gap against legacy is definite. Legacy's `BroadcastScalar` has three branches
(`codegen/api_call/broadcast/broadcast_api_call.cpp:242-273`); the first one, `in.is_constant`, splices
the constant's name straight into `Duplicate(out[off], in.name, out.actual_size)` (`:249-251`) and
needs **neither a tile nor a `GetValue(0)`**. The entry predicate `IsBroadcastConstantTensor`
(`:227-240`) is unconditionally true for `Constant` / `IndexExpr` / `Scalar` / `ScalarData` nodes
(`graph_metadef/graph/ascendc_ir/utils/asc_tensor_utils.cc:15-20`). MLIR implemented only the third
branch (a real tile, the event triple, and `GetValue(0)`; see
`conversion/AFIRToAscendC/Implementations/Broadcast.cpp:892-905`).

This gap is unrelated to the inline-broadcast work on the current branch (`d1b8cd0e4`, `77c390732`,
`aca71764f`, `2e0249dbf`): those four commits only touch
`AFIRToAscendC/Implementations/*`, `Analysis/TransferPlan.cpp` and `Transforms/ExecConditionPass.cpp`,
and within `Implementations/Broadcast.cpp` they add two comment lines only — whereas this failure
happens during module verification, before any of that lowering code is reached.

### Product Environment

- Upstream: the ASCIR graph produced by legacy's scheduler (containing `Scalar` / `IndexExpr` nodes and
  `Broadcast` nodes), converted to AFIR by `bridge/ScheduleToAfir` plus `bridge/AfirBuilder`.
- Downstream: `AFIRToAscendC` emits AscendC calls; `Duplicate` maps to `AscendC::Duplicate`, or to the
  mem-side `DuplicateU8` wrapper for ui8.
- External interface boundary: no exported C/C++ symbol, Python interface, script argument or artifact
  layout is added or changed.

## Software Functions

1. AFIR can express "a broadcast whose source is a scalar" and it passes verification in buffer form;
   the logical (tensor) form keeps rejecting scalars.
2. With a scalar source, the `Duplicate` tier emits `Duplicate(dst, <scalar>, count)` without reading
   an on-chip tensor and without a synchronization event, matching legacy's first branch item for item.
3. Combinations still unsupported with a scalar source (the two segmented tiers, element types outside
   the `Duplicate` set, a multi-segment destination, …) answer unsupported in the capability layer, and
   the whole graph falls back to legacy cleanly.
4. No failure of this shape may appear again as an unmarked compilation error.

## Design Constraints

- CANN Toolkit: verified against 9.3.0 as deployed; the signature of `AscendC::Duplicate` and the state
  of both headers are unchanged.
- ABI/API: no public interface is touched, so there is no ABI impact.
- Element-type domain: the scalar source's dtype set **must not be wider** than the existing
  `Duplicate` tier's — `kDuplicateElementTypes = kF16|kF32|kI16|kI32|kUI16`, plus `|kUI8` on the mem
  side (`Implementations/Broadcast.cpp:101, 1105-1111`). The single source of truth for dtypes remains
  `support/AfirElementTypes.h`; no second vocabulary is created.
- The tmp-buffer contract keeps its registered values: `needsTmpBuffer=true` for the mem-side
  `broadcast.tensor.duplicate` and `false` for the reg-side `broadcast.tensor.duplicate_reg_base`
  (`Implementations/Broadcast.cpp:1129-1136`); `checkTmpBuffer` keeps comparing them strictly.
- Rejection may only happen in the capability layer: a verifier failure is a hard failure and cannot
  serve as a means of rejection.
- No hard-coded chip model, SoC string or framework name; the established convention of
  `Implementations/Broadcast.cpp` is that "neither the stable ids nor the comments mention a chip name".

## Assumptions and Dependencies

- **To be confirmed 1**: whether the divisor node in the online graph is a `Scalar` or an `IndexExpr`.
  The platform's dump is truncated; `ScalarData` can be ruled out (the nnsmith graph has a single tensor
  input and no by-value scalar parameter). Both end identically on the MLIR side (the same
  `buildTypedConstant`, the same bare scalar), so the design does not depend on telling them apart; to
  decide it, read the producer node kind of `broadcast.x` from a complete dump.
- **To be confirmed 2**: how many consumers that `Broadcast` has. Legacy's
  `ScalarBroadcastOptimizationPass` requires both in-degree and out-degree to be 1
  (`optimize/schedule_utils.cpp:1100-1111`, decided at `:1107`) before it eliminates a scalar
  broadcast. It did not fire for this graph, most likely because two divisions share the divisor. If a
  measurement shows a single consumer, the next thing to check is what `IsNodeSupportsScalarInput`
  returns for TrueDiv's x2 position — but either way the MLIR side should carry this shape on its own
  rather than depend on legacy eliminating it.
- **~~To be confirmed 3~~ (resolved 2026-11-13)**: bare scalar operands already have defined handling in
  the descriptors and in `TransferPlan`; no new mechanism is needed:
  - `verifyCanonicalBufferDescriptors` explicitly excludes scalar operands from the "half a buffer
    form" predicate (`IR/AFIRVerifierUtils.cpp:208-218`, whose comment uses `afir.add %tile, %scalar`
    as the example);
  - `verifyDescriptorList` accepts `operand_descs` with `allowsScalars=true`
    (`AFIRVerifierUtils.cpp:237`): a scalar **still occupies one slot** while its content is skipped
    entirely (`:105-110`), the slot being there so that descriptor indices do not shift out of step
    with operand indices;
  - `buildTransferPlan(op)` takes `result_descs[0]` by default
    (`Analysis/TransferPlan.cpp:87-100`), which is independent of whether the source is a tile or a
    scalar, so the count computation of the `Duplicate` tier needs no change at all.

# Requirements Analysis and Design

## Overall

The data flow is unchanged; two places are relaxed and one emission branch is added:

```
ASCIR  Scalar/IndexExpr ──┐
                          ├─► Broadcast ─► TrueDiv ─► Store
       (reduced element count)

bridge  bare-scalar arith.constant ─► afir.broadcast(operand#0 = bare scalar, destination = tile)
                                     │
ODS/verifier   (1) operand#0 relaxed to "shaped or bare scalar"; shape check takes a scalar branch
                                     │
capability     (2) Duplicate tier accepts a scalar source; the segmented tiers keep rejecting it
                   ─► unsupported means the whole graph falls back to legacy
                                     │
lowering       (3) emit Duplicate(dst, <scalar>, count): no tile read, no event
```

The key point about control flow: the right to reject belongs to the capability layer alone. After the
ODS relaxation, being expressible is not the same as being generatable; any combination the capability
layer answers unsupported is marked as a capability shortfall through the `unsupported` branch at
`CodegenOrchestration.cpp:469-477`, and `classify` returns `kNotSupported` and falls back to legacy.

## Functional Requirements

### Requirement 1: `afir.broadcast` accepts a scalar source and verifies it correctly

1. **Behavior**

   For a buffer-form `afir.broadcast`, operand #0 may be a bare scalar whose element type is in
   `AFIR_BroadcastElementTypes`; the destination and the result must still be shaped types. The logical
   (tensor) form is unchanged and keeps rejecting non-tensor operands.

2. **Input**

   - Legal: operand #0 is a bare scalar of `f16/f32/i16/ui16/i32/ui32/ui64/i8/ui8/bf16`, the
     destination is a `LocalTensorType`/memref, and the two element types are equal.
   - Illegal and required to be diagnosed clearly: a bare scalar operand #0 whose element type differs
     from the result's; a bare scalar operand in the logical form; a missing destination.
   - Boundary: the ODS element-type list is wider than the AFIR dtype bit table (`I8`/`UI32`/`UI64`
     have no bit), so graphs using those types can be built but select no implementation and fall back
     cleanly, exactly as before.

3. **Processing**

   - ODS: the type constraint of operand #0 becomes
     `AnyTypeOf<[AFIR_BufferizableShapedType<AFIR_BroadcastElementTypes>] # AFIR_BroadcastElementTypes>`,
     **reusing** the `scalarOperand` formulation already present in `Elementwise.td:216-244` rather
     than inventing a mechanism. The destination and result slots are untouched.
   - `verifyBroadcastShapeImpl` (`IR/Ops/Broadcast.cpp:26-61`) gains a scalar-source branch: when
     operand #0 is not a `ShapedType`, compare `getElementTypeOrSelf` against the result's element type
     and return success, skipping the rank and broadcastable loop (a scalar has no shape to compare).
     The result side must still be a `ShapedType`.
   - `BroadcastOp::verify()` (`:91-106`) needs no change: its logical branch (`:96-98`) already keeps
     bare scalars out of the tensor form, and the bridge gives the tensor form its own diagnostic for
     `Scalar` nodes at `AfirBuilder.cpp:1190-1193`.

4. **Output**

   The IR can be built and verified; illegal combinations produce a clear verifier diagnostic pointing
   at the broadcast op. Module verification (`AfirBuilder.cpp:2110`) no longer fails for this shape.

### Requirement 2: the `Duplicate` tier emits a scalar-source broadcast

1. **Behavior**

   The scalar source is registered as **two more implementations** (one mem, one reg) whose predicates
   are mutually exclusive with the two tile-source ones.

   > Revised 2026-11-13: the first draft said "branch inside the existing `Duplicate` tier, add no new
   > id", on the grounds of touching fewer hand-maintained registry lists. The implementation follows
   > the established precedent instead — the tensor form and the scalar form of `afir.add` are **two
   > implementations of the same op**, their predicates mutually exclusive by operand shape rather than
   > ordered by priority (see the header of `test/lit/conversion/afir-scalar-binary-api.mlir`, and the
   > `registerScalarBinary` family at `Implementations/BinaryElementwise.cpp:980-1008` for the
   > registration form). "Where the scalar comes from" determines what gets emitted (whether a tile is
   > read, whether an event is issued), which is exactly what an implementation is meant to
   > distinguish. The cost is one more snapshot row in `ElementTypeMatrixTest`'s `kKnownNarrower` for
   > the reg-side scalar tier (the same ui8 reason as the tile-side row).

2. **Input**

   An `afir.broadcast` whose operand #0 is a bare scalar, whose destination is a single-segment tile,
   and whose element type is in that implementation's dtype set.

3. **Processing**

   - `BroadcastCapabilityProvider::queryTensorContract` (`Implementations/Broadcast.cpp:780-831`):
     - the current `:786-790` answers `UnsupportedType` for any source that is not a `LocalTensorType`;
       it now splits first — a bare scalar source with `tier == Duplicate` takes the scalar path, and
       everything else keeps the original judgment (the segmented tiers must keep rejecting a scalar
       source, since they depend on the source's geometry);
     - the scalar path's predicate: the element type is in `mapping_->elementTypes` and equals the
       destination's; "the source has repeat 1 on every vectorized axis" (`:799`,
       `isConstantOnVectorizedAxes`) holds trivially for a scalar, so no operand descriptor is read;
       `scalarReadEvent` (`:802`) is meaningful only for a tile source and is not consulted;
     - the gate at `:813` **must be kept**: under a `Scheduled` plan it requires
       `localTileIsSingleSegment`. Its reason has nothing to do with the source — it constrains "filling
       by the span would write into the inter-row padding", and its evidence comes from the destination
       side.
   - `queryScalarInput` (`:838-841`), today unconditionally `unsupported`, now answers supported when
     `tier == Duplicate` and the query is `scalarOperandIndices == {0}` with `!allScalar`, and keeps
     answering unsupported otherwise. This query has no caller in production code today (only
     `test/ut/conversion/BinaryElementwiseL2CapabilityTest.cpp` uses it); filling it in keeps the op's
     self-description consistent with the ODS relaxation, and adds no caller.
   - The lowering `BroadcastDuplicateConversionPattern::matchAndRewrite` (`:860-909`): with a bare
     scalar source, skip the source-side `LocalTensorSubIndexOp` at `:885`, the tile type check at
     `:892-895` and the `materializeScalarRead` at `:896`, and pass that SSA value directly as the
     scalar argument of `DuplicateL2Op` / `DuplicateU8Op`; the destination's zero offset and the `count`
     computation (`:877-890`) stay as they are.
   - The dtype branch keeps its existing form: `ui8` goes to `DuplicateU8Op` (`:901-902`), everything
     else to `DuplicateL2Op` (`:904`).

4. **Output**

   In the generated kernel the node becomes a single `Duplicate(dst[off], <scalar>, count)`, the same
   shape as legacy's first branch (`broadcast_api_call.cpp:249-251`), with one synchronization event
   group fewer than a tile source.

### Requirement 3 (conditional, **not implemented**)

2026-11-13: requirements 1 and 2 shipped together, so per the design's own rule ("do not introduce it
if both ship together") this one is **not done** and no temporary patch is left behind. The original
text is kept below as a record of the reasoning.

Introduce it only if the fuzz gate has to be unblocked first while requirements 1/2 cannot ship in the
same batch: before the bridge builds the `Broadcast` op, detect "the input is a bare scalar", report it
through `emitCapabilityShortfall` (`support/CapabilityShortfall.h`) and return failure, so that
`classify` goes to `kNotSupported` and falls back to legacy. The formulation follows the three existing
precedents at `AfirBuilder.cpp:794-796`, `:1058-1060` and `:1151-1153`.

That branch is a temporary patch and **must be deleted** when requirements 1/2 land; if the two ship
together it is never introduced, so that no structural patch is left behind.

## Non-functional Requirements

### Maintainability

- Reuses the `scalarOperand` formulation of `Elementwise.td` and the existing structure of the
  `Duplicate` tier; no new tier, dtype vocabulary or selection algorithm.
- The scalar path and the tile path split at exactly one point in the capability layer and one in the
  pattern, on a single criterion (whether operand #0 is a `ShapedType`), so the conditions do not get
  scattered.
- Comments explain only the non-obvious part — why the scalar path reads no descriptor and issues no
  event — and point at the corresponding legacy branch by line.

### Testability

- lit: a positive case (a scalar-source broadcast parses and verifies) and negative cases (element
  types that differ, a bare scalar in the logical form). Each negative block violates exactly one
  predicate, uses `--split-input-file`, and its CHECKs are counted.
- Literal sizes get folded away, so alignment-style predicates are expressed with symbolic sizes and
  whatever cannot be pinned there goes to a UT.
- capability UT: following the existing broadcast capability tests, add supported / unsupported cases
  for a scalar source, covering an element type outside the set, a multi-segment destination, and the
  segmented tiers rejecting a scalar source.
- E2E: a new `Scalar → Broadcast(several consumers) → TrueDiv` CPU case, with a shape that makes
  `a0Tb_size > 1` so it does not land in the "one row per tile" blind spot.
- Instrumentation: `af-capability-probe` must report FAILED / DRIFT of 0; an all-green lit run is not
  sufficient as a conclusion.

### Portability

Mem and reg sides are covered separately, each with the dtype set its header actually supports (the
reg-side `Duplicate` tier has no `UI8`, a recorded narrowing that this change does not relax). Nothing
depends on a chip model.

### Reliability

- A scalar source allocates no UB and holds no new resource, so no release path changes.
- No new `rt*` / `aclrt*` call.
- Risk: once the ODS is relaxed, a missing capability predicate turns the failure mode from "hard
  failure" into "silently wrong code". Therefore the capability predicate and the pattern split must
  land in the same batch, and the E2E must check numbers (fill the output with a sentinel rather than
  0); "it compiled" is not evidence of passing.

### Cross-feature Impact

Item by item, per `docs/guidelines/cross_feature_check.md`:

| Scenario | Applicable | Analysis |
|----------|------------|----------|
| SuperKernel Python interface | No | `super_kernel/`, option parsing and wheel contents are untouched |
| SuperKernel C++/AOT interface | No | `libascendsk.so`, AOT and device-side RDV are untouched |
| Autofuse graph optimization | No | No ASCIR pass added or reordered, no graph rewrite; legacy's `ScalarBroadcastOptimizationPass` explicitly untouched |
| Autofuse Codegen/Backend | Yes | Changes the AFIR ODS/verifier, the `Duplicate` tier's capability predicate and its emission; generated code and kernel execution both need coverage |
| AscendC API / Runtime interaction | Yes | Reuses the existing `AscendC::Duplicate` / `DuplicateU8`; signature and tmp-buffer contract unchanged; the scalar path issues one `SetFlag`/`WaitFlag` group fewer, adding no lifetime constraint |
| Python/C++ bindings | No | `autofuse/compiler/py_module/` interfaces are untouched |
| Build and packaging | No | CMake, `build.sh`, dependencies and install paths are untouched |
| Tests and coverage | Yes | New lit positive/negative cases, a capability UT and a CPU E2E case; see Test Design |
| Performance and logging | Yes | One extra type check at compile time, negligible; one synchronization event group fewer at run time; no new logging on by default |
| Compatibility | Yes (non-breaking) | Implementation ids and tmp-buffer contracts preserved; existing tile-source artifacts must stay byte-identical, to be shown by a baseline comparison |

## Performance

### Compile Time

What is added is one `isa<ShapedType>` test per broadcast node and one split in the capability layer —
constant work, introducing no new graph traversal or string processing, with no measurable effect on
`autofuse/codegen` compile time.

### Execution Performance

Compared with the alternative of "materialize into a single-element tile and read it back", the scalar
source saves a UB block, a write-back and a `SetFlag`/`WaitFlag` pair; it produces the same instruction
sequence as legacy's first branch, so there is no regression relative to legacy. This shape emits a
single `Duplicate` per node and adds no overhead on a hot path.

### Memory and Artifact Size

No resident structure or cache is added; no implementation registration is added beyond two ids, so the
registry's size is effectively unchanged; dynamic library and run package sizes show no measurable
change.

## Interface Design

### New / Changed Interfaces

All internal; no public signature changes:

1. `AFIR_BroadcastOp` operand #0 type constraint: `AFIR_BufferizableShapedType<T>` →
   `AnyTypeOf<[AFIR_BufferizableShapedType<T>] # T>` (`Ops/Broadcast.td:34-41`).
2. `verifyBroadcastShapeImpl(Operation *)`: a scalar-source branch (`IR/Ops/Broadcast.cpp:26-61`).
3. `BroadcastCapabilityProvider::queryTensorContract` / `queryScalarInput`: scalar-source predicates
   (`Implementations/Broadcast.cpp:780-841`).
4. `BroadcastDuplicateConversionPattern::matchAndRewrite`: a scalar-source emission branch
   (`Implementations/Broadcast.cpp:860-909`).

### Interface Checklist

| Item | Sub-item | Applicable | Notes |
|------|----------|------------|-------|
| Description | Needs an interface review | No | All internal ODS / conversion implementation |
| Description | Needs documentation | Yes | Update the "source is a tile" wording in `docs/development/2026-09-23-afir-broadcast-design.md`, and mark the first legacy branch as wired up in the four-branch table in the header of `Implementations/Broadcast.cpp` |
| Compatibility | Behavior compatible | Yes | A relaxation; the existing tile-source path must keep its behavior and its artifacts byte for byte |
| Compatibility | ABI/API compatible | No | No public symbol, Python interface, script argument or artifact layout changes |
| Constraints | Clear error when a constraint is unmet | Yes | An element-type mismatch gives a verifier diagnostic; a capability shortfall goes through `emitCapabilityShortfall` and falls back, no longer appearing as an unmarked `kFailed` |
| Testing | Needs dedicated interface cases | Yes | lit positive/negative cases plus a capability UT |

## Software Design

### Key Data Structures

- `BroadcastMapping` (`Implementations/Broadcast.cpp`): one more field,
  `BroadcastSource source` (`Tile` / `Scalar`), defaulting to `Tile` and placed after `needsTmpBuffer`
  so that the three existing aggregate initializations keep their positional arguments. Both the
  capability layer and the pattern read it: the capability layer compares it against the actual form of
  operand #0 in the IR and answers unsupported on a mismatch (which is what makes the two mutually
  exclusive); the pattern uses it to decide whether to take the scalar-read path.
- `ScalarInputQuery` (`target/Ascend/Capability/Capability.h:139-142`,
  `{scalarOperandIndices, allScalar}`): used as is, structure unchanged.

### Key Technique / Algorithm

The scalar path needs no geometric analysis: the semantics of `Duplicate` are "fill one contiguous
span of the destination with a single value", and the length still comes from
`materializeScheduledLocalSpan` (the physical span between the first and last element, not the element
count), which is why the single-segment gate at `:813` must be kept. The source-side predicate "repeat
1 on every vectorized axis" degenerates to trivially true for a scalar source — it is not a check that
was given up. Complexity O(1).

### Flow

Main flow: the bridge builds a bare scalar → builds the broadcast op → module verification (scalar
branch) → capability query (the `Duplicate` scalar path) → plan selection → lowering emits `Duplicate`
→ artifact assembly.

Exception flow: the capability layer answers unsupported → the `unsupported` marker →
`emitCapabilityShortfall` → `classify` returns `kNotSupported` → the whole graph falls back to legacy
(`CodegenOrchestration.cpp:446-451, 469-477`).

Rollback / cleanup: no persistent resource; a failure simply falls back, so nothing needs cleaning up.

### Changes per Submodule

| Directory / file | Change |
|------------------|--------|
| `autofuse/mlir/dialect/AFIR/IR/Ops/Broadcast.td` | Relax the operand #0 type constraint to "shaped or bare scalar"; add a comment pointing at legacy's first branch |
| `autofuse/mlir/dialect/AFIR/IR/Ops/Broadcast.cpp` | A scalar-source branch in `verifyBroadcastShapeImpl` |
| `autofuse/mlir/conversion/AFIRToAscendC/Implementations/Broadcast.cpp` | The capability layer's source-form split, the `queryScalarInput` answer, the lowering's scalar path; update the legacy-branch table in the file header |
| `autofuse/mlir/test/lit/conversion/afir-broadcast-duplicate-scalar.mlir` | New: the scalar-source positive case (including the rejection tier for a strided destination) |
| `autofuse/mlir/test/lit/conversion/afir-broadcast-scalar-invalid.mlir` | New: three negative cases (element types differ / dtype outside the ODS list / logical form) |
| `autofuse/mlir/test/lit/conversion/afir-broadcast-duplicate.mlir` | Add `--implicit-check-not=broadcast.scalar.duplicate` to pin mutual exclusion |
| `autofuse/mlir/test/ut/conversion/ElementTypeMatrixTest.cpp` | One more `kKnownNarrower` row for the reg-side scalar tier |
| `autofuse/tests/framework/easy_asc_graph/asc_graph_builder.{h,cpp}` | New `BroadcastFromScalar`: the output axes of a scalar-source broadcast come from the graph's loop axes (the source's output view is empty and there is no other input to follow). Purely additive; the three existing `Broadcast` overloads are unchanged |
| `autofuse/mlir/test/e2e/cpu/cases/load_scalar_broadcast_store/` | New: `Scalar → Broadcast(two consumers) → Add → Mul → Store`, two shape tiers |
| `autofuse/mlir/test/e2e/cpu/CMakeLists.txt` | Register that case directory and hang the `_131` tier on `check-e2e-cpu-nonreduce-addressing` |
| `autofuse/mlir/docs/development/2026-09-23-afir-broadcast-design.md` | Pending: revise the "source is a tile" wording |

The bridge (`AfirBuilder.cpp`) is **not changed** when requirements 1/2 ship together; the temporary
rejection is added only if the conditional requirement 3 is taken, and is removed afterwards.

### Error Handling

#### System Errors

No new path allocates memory, touches files or calls the runtime. Any failure inside the pattern goes
through `notifyMatchFailure` and is rolled back by the conversion framework, with existing behavior
unchanged.

#### Interface Errors

- An element-type mismatch, or a bare scalar in the logical form: a verifier diagnostic pointing at the
  broadcast op.
- A capability shortfall: the `emitCapabilityShortfall` marker plus the existing diagnostic text; the
  normal mode falls back and strict mode returns `kFailed` while printing the `gate=` label. The
  diagnostic text is identical in both modes.

## Security Review

Item by item against `docs/guidelines/编码红线.md`:

- No hard-coded secrets, public addresses, accounts, or chip / framework type decisions (general rule
  1, Graph-autofusion rule 2).
- External input used as an index or length: `count` still comes from the existing
  `materializeScheduledLocalSpan` / `LocalTensorGetSizeOp`; no new path lets a graph attribute act
  directly as a length or an offset, and the single-segment gate is kept so nothing is written into the
  padding by span (rules 2, 3).
- Integer arithmetic: scalar literal parsing keeps the existing overflow / invalid-text rejection in
  `buildTypedConstant` (`AfirBuilder.cpp:1095-1128`) and relaxes nothing (rule 3).
- Resource release and allocation: no resource is held or allocated (rules 4, 5).
- No reliance on unspecified evaluation order; no logic added that lets `unordered_map` iteration order
  or pointer addresses determine the artifact, so graph-rewrite determinism is unaffected (rule 6,
  Graph-autofusion rule 5).
- Scope not widened: legacy is untouched, adjacent code is not refactored, and no existing dead code is
  removed (rule 7, Graph-autofusion rule 3).
- ABI/API: no public interface changes (Graph-autofusion rule 1).
- Pass ordering: no pass added or reordered (Graph-autofusion rule 4).
- Logging: nothing new on by default (Graph-autofusion rule 6).
- AscendC lifetime: the scalar path involves no asynchronous copy, and the `Duplicate` destination is
  valid within the current tile; the event that is no longer issued existed only to serve "read a scalar
  out of a tile", which the scalar path does not do (Graph-autofusion rule 7).
- No non-public runtime interface is used; no Python binding reference counting is involved; build
  scripts are untouched; test artifacts land in ignored directories (Graph-autofusion rules 8-12).

## Compatibility Review

- Existing tile-source `Duplicate` artifacts must stay byte-identical: take a kernel baseline by
  running `check-e2e-cpu` with the pre-change code (the build directory does not keep `kernel.cpp`
  around), then compare the post-change artifacts against it.
- Stable implementation ids, dtype sets and tmp-buffer contracts are unchanged, so
  `ElementTypeMatrixTest`'s existing snapshots should show no added or altered row; if one appears, the
  change has gone out of bounds.
- No effect on script arguments, configuration files, install directories, dynamic libraries, the
  Python package, or historical artifacts.

## Test Design

### Test Boundary

- Entry points: `af-opt` for AFIR verification and AFIRToAscendC (already built, so a few-second loop);
  capability UTs construct the op and query it directly; E2E goes through the
  `check-e2e-cpu-<case>` targets.
- Exit points: verifier diagnostics, capability query results, the shape of the emitted AscendC call,
  and the CPU numerical result.
- Stubs: the existing capability UT construction is reused; no new stub.
- Verification data: the E2E output buffer is pre-filled with a sentinel so that "did not compute" is
  distinguishable from "computed correctly".

### Test Cases

| Category | Key item | Method | Type |
|----------|----------|--------|------|
| Function | A scalar-source broadcast verifies and lowers to `Duplicate(dst, scalar, count)` | lit (symbolic sizes, counted CHECKs) | UT |
| Function | `Scalar → Broadcast(several consumers) → TrueDiv` is numerically correct end to end | `check-e2e-cpu-<case>`, shape satisfying `a0Tb_size > 1` | ST |
| Exception | Scalar and result element types differ; a bare scalar in the logical form; a missing destination | lit negatives, `--split-input-file`, one predicate per block | UT |
| Exception | A dtype outside the `Duplicate` set; a multi-segment destination; the segmented tiers meeting a scalar source | capability UT asserting unsupported with the right reason code | UT |
| Performance | The scalar path issues one synchronization event group fewer than the tile path, and no extra instruction relative to legacy | Manual inspection of generated code plus a baseline diff | Manual |
| Compatibility | Existing tile-source kernels stay byte-identical | Kernel baseline diff before and after | ST |
| Cross-feature | No new hard failures appear in the broadcast rows of the takeover table | `af-capability-probe`, FAILED/DRIFT of 0 | E2E |

### Test Commands

```bash
bash ./mlir-test.sh
```

```bash
make -C build check-e2e-cpu-<case>x128 -j 6
```

```bash
cmake --build build --target af-opt -j 6
```

Expected results: the verdict line of `mlir-test.sh` is all green (only the last line counts; do not
count `[ FAILED ]` occurrences); the new E2E case is numerically correct; `af-capability-probe` reports
FAILED / DRIFT of 0; a retest of the online seed goes from a hard failure to being taken over by MLIR.

## Acceptance Criteria

1. A retest of the online seed `EDL4VTLLGBGY5RIZOUBMFG1GMO` no longer shows
   `operand #0 must be ... but got 'f32'`, and the graph is taken over by MLIR (if it falls back for
   some other predicate, the diagnostic must carry the capability-shortfall marker and must not be an
   unmarked `kFailed`).
2. The call emitted for a scalar-source broadcast has the same shape as legacy's
   `broadcast_api_call.cpp:249-251`.
3. Existing tile-source artifacts are byte-identical under a baseline comparison, and
   `ElementTypeMatrixTest`'s snapshots show no altered row.
4. The new lit positive/negative cases, the capability UT and the CPU E2E all pass; the verdict line of
   `mlir-test.sh` is all green.
5. `af-capability-probe` reports FAILED / DRIFT of 0.
6. Mutation check: reversing either the element-type decision or the single-segment gate of the scalar
   path must turn the corresponding negative case red (distinguishing the three cases of blind spot /
   redundancy / conservative boundary).
7. Documentation updated: the "source is a tile" wording in `2026-09-23-afir-broadcast-design.md` and
   the legacy-branch table in the header of `Implementations/Broadcast.cpp`.

## Implementation Record (2026-11-13)

The implementation covers requirements 1 and 2; requirement 3 was not introduced. Measured locally
(x86_64 plus the CPU simulator):

| Item | Result |
|------|--------|
| Full MLIR suite `bash ./mlir-test.sh` | Verdict line `Autofuse MLIR targets succeeded: check-ut check-lit check-st check-e2e`, 14m42s |
| lit | 299 / 299 pass (including the two new files) |
| `AFIRToAscendCTest` | 88 / 88 pass (including the new `ElementTypeMatrixTest` snapshot row) |
| New E2E `load_scalar_broadcast_store` | Both the `_131` and `_40x128` tiers pass, with an element-wise assertion of `(input + 2.5) * 2.5` |
| Emitted shape | `AscendC::Duplicate(v80, c2_5_f32, v79);` — the constant goes straight into the call, with no `.GetValue(` and no `MTE2_S` event pair |
| The two consumers share one broadcast tile | The `v93` of `AscendC::Add(v94, v92, v93, v76);` and of `Mul(v97, v94, v93, v76);` is the same value |
| clang-format | `--dry-run -Werror` prints nothing for the seven C++ files involved |

Mutation check (acceptance criterion 6): after making the capability layer's source-form exclusion
predicate never reject (`if (false && …)`) and rebuilding `af-opt`, **both**
`afir-broadcast-duplicate.mlir` and `afir-broadcast-duplicate-scalar.mlir` turned red, confirming that
this predicate is genuinely covered rather than written and never checked; it was then restored and the
tests were re-run green.

Not done / not covered:

- **The online seed has not been retested**: `EDL4VTLLGBGY5RIZOUBMFG1GMO` needs the fuzz platform's
  environment (aarch64, a real NPU, torch), which this machine does not have.
- **The takeover-table probe has not been re-measured**: it lives in the remote NPU machine's
  environment and does not exist here, so there is no data for "how the hard-failure count changed".
- **No separate case for the `IndexExpr` form**: the new cases use a `Scalar` node. Both go through the
  same `buildTypedConstant` in the bridge and end in a bare scalar the same way, and the lowering
  treats that operand as an opaque SSA value, so the mechanism is shared; but which of the two the
  online graph's divisor actually is remains undecided (see "To be confirmed 1").
- The second branch of legacy's `BroadcastScalar` (`is_ub_scalar`) is still not modelled; the capability
  layer rejects it and the graph falls back.

### Design Document Checklist

- [x] Cross-feature impact: filled in item by item per `cross_feature_check.md`; the applicable ones are
      Autofuse Codegen/Backend, AscendC API interaction, tests and coverage, performance and logging,
      and compatibility, with reasons given for the rest
- [x] Coding red lines: checked item by item against the 7 general and 12 Graph-autofusion rules of
      `编码红线.md`, with no violation; particular attention to not touching legacy, not widening scope,
      graph-rewrite determinism, and AscendC lifetimes
- [x] Test design: lit positive/negative cases, a capability UT, a CPU E2E case and
      `af-capability-probe`, plus the mutation check and the baseline comparison requirement
- [x] Performance assessment: constant extra work at compile time; one UB block and one synchronization
      event group fewer than the "materialize into a tile" alternative, and no regression relative to
      legacy
