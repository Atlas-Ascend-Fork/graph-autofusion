# Inline Broadcast (brc_inline): Design for the MLIR Side

> 2026-11-10 · Upstream: requirement 5 (batch 3) of
> [Broadcast cache and inline broadcast: issue list and fix design](2026-11-09-brc-cache-and-brc-inline-issues-design_en.md).
> This is the separate design document that requirement asked for.

# Introduction

## Purpose

Give MLIR the capability that legacy's `brc_inline` template has: when one side of a binary elementwise
op is a row broadcast, do not require a fully materialized broadcast tile; fold the broadcast into the
elementwise call itself by taking a repeat stride of 0 on that source.

The audience is whoever implements and reviews this batch. Acceptance means: a plan for which legacy
chose the brc_inline template is taken over by MLIR, with numbers matching legacy.

## Scope

- In scope: wiring up the `BroadcastInline` capability query in `target/Ascend/Capability`; one new
  operand form in the binary elementwise capability decision and lowering; one new extension-API op in
  the Asc dialect plus its emitter; the corresponding lit and numeric cases.
- Out of scope: any change to the three `afir.broadcast` lowering tiers; unary and ternary elementwise;
  reduce; any change to legacy.
- Component: Autofuse (MLIR codegen path). SuperKernel is not involved.

# Overview

## Software Overview

### Project Background

Legacy's `BrcInlineTemplate` (`optimize/platform/v1/template/brc_inline_template.cpp`) **removes the
Broadcast node** during optimize and tags the consumer with the `kInline` semantic role; codegen's
`BinaryApiCall` then emits `BinaryBrcInlineApiWithTwoVectorizedAxisV2`
(`codegen/api_call/elewise/binary_api_call.cpp:196`), a wrapper that blocks over two vectorized axes and
multiplies the broadcast side's repeat stride by 0.

MLIR has no such path today: `CapabilityKind::BroadcastInline` has no production caller anywhere in the
repository, all nine `queryBroadcastInline` overrides answer unsupported, and the fields of
`BroadcastInlineQuery` are never populated. On the lowering side only `isSingletonTensorInput`
(`ElementwiseTransferPlan.cpp:25`) exists, i.e. "all repeats are 1 → read as a scalar". MLIR also never
reads the upstream template semantic role, so a graph that legacy judged to be brc_inline looks to MLIR
like an elementwise op whose two inputs have incompatible local views, and
`checkLocalViewCompatibility` rejects it.

The 2026-11-10 NPU run quantifies the cost: of seven broadcast e2e cases, five still carry a known
rejection diagnostic, and every one of them is
`'afir.add' / 'afir.mul' op has no supported Ascend implementation` (4 + 1).

### Product Environment

Compile-time capability. The emitted function comes from `autofuse/ascendc/api/brc_inline_api.h`, which
already ships with the run package (`autofuse/ascendc/api/CMakeLists.txt:39`); legacy inlines it into the
generated kernel as `brc_inline_api_str.h`. The MLIR side uses the existing extension-header mechanism
(`ApiExtensionHeader`) and adds no deliverable.

## Software Functions

1. Recognise the "one side of a binary elementwise op is a row broadcast" form, with a predicate
   equivalent to legacy's `IsGeneralizeBrcInlineScene`.
2. Have the `BroadcastInline` capability query answer that form, giving the interface its first real
   caller.
3. Lower it to `BinaryBrcInlineApiWithTwoVectorizedAxisV2` with arguments matching legacy item by item.
4. Keep rejecting every other form as a capability shortfall with a wholesale fallback; never compute
   silently wrong results.

## Design Constraints

- Rejection may only happen in the capability layer; silently passing is not allowed.
- The new implementation's dtype set **must not be narrower than legacy's for the same form** (see the
  two hand-maintained registry gates).
- The last emitted argument is the function pointer `&AscendC::<Api><T, false>`. That `false` is
  `isSetMask` and **does not contradict AGENTS experience 3.3**: the wrapper sets the mask itself in
  counter mode (`SetMaskCount` / `SetVectorMask` / `SetMaskNorm` in `brc_inline_api.h`), so the caller
  must hand over mask control — this is exactly what `false` is for. The reason must be in a comment so
  nobody "fixes" it back to `true`.
- No new hardcoded SoC or platform checks.

## Assumptions and Dependencies

All three originally open premises were measured before implementation; the conclusions are at the end of
this document and only the results are kept here:

- The upstream does hand brc_inline plans to MLIR and the bridge does not reject them earlier: the inline
  template arrives intact in AFIR, and the only failure is that `afir.add` has no usable implementation.
- V2 of `brc_inline_api.h` is what legacy actually emits today (V1 is still in the header).
- The wrapper's "two vectorized axes" assumption holds for this form; whether MLIR's axis collapsing
  agrees with legacy's grouping for three or more vectorized axes was not verified, so the
  implementation follows legacy's grouping.
- The reg side has no equivalent wrapper, so this batch covers the mem side only.

# Requirement Analysis and Design

## Overall Approach

Three steps, each separately verifiable:

1. **The form predicate** — a "row broadcast input" predicate in
   `conversion/AFIRToAscendC/Implementations/`, equivalent to legacy's `IsGeneralizeBrcInlineScene`,
   answered by the binary elementwise capability provider in `queryBroadcastInline`. This step changes
   only predicates and queries and is pinned by UT.
2. **The API op** — a new `AscendC_BinaryBrcInlineOp` in the Asc dialect plus its emitter, producing
   `BinaryBrcInlineApiWithTwoVectorizedAxisV2<T>(dst, src0, src1, shape0, shape1,
   is_input0_block_brc, is_input1_block_brc, first_axis_v_stride, &AscendC::<Api><T, false>)`. Pinned by
   lit on the emitted text.
3. **Wiring** — one new binary elementwise implementation whose capability uses step 1 and whose pattern
   emits step 2's op; plus lit and CPU E2E numeric cases, and a fresh NPU run over the five cases that
   carry the diagnostic today.

## Functional Requirements

### Requirement 1: the row-broadcast form predicate

1. **Description** — decide whether a binary elementwise op matches legacy's inline broadcast form.
2. **Input** — the op's two operand descriptors and its destination descriptor.
3. **Processing** — aligned item by item with `IsGeneralizeBrcInlineScene`
   (`common/common_utils.cpp:856`): both inputs are tensors with the same number of vectorized axes;
   comparing repeats per vectorized axis, **exactly one** side has a position that is 1 while the other
   is not (both or neither is rejected); after merging consecutive broadcast and non-broadcast positions
   each side has 2 groups; and the broadcast side's **outer** group repeat is 1, i.e. the form is
   `(1, A) op (B, A)`. The predicate reads repeats only; strides are used solely to compute
   `first_axis_v_stride`.
4. **Output** — which side is the broadcast side, and the two merged repeats; otherwise unsupported.

### Requirement 2: wiring up the `BroadcastInline` query

1. **Description** — give the interface a real caller and a meaningful answer from the binary elementwise
   provider.
2. **Input** — `BroadcastInlineQuery`: `sourceOperandIndex` (the broadcast side),
   `destinationOperandIndex`, `axis`, both `sourceStrides` / `destinationStrides`, `hasDynamicShape`.
   These fields have never been populated, so their meaning is fixed in this batch rather than
   reinvented per call site.
3. **Processing** — `queryBroadcastInline` answers supported (with constraints) when the form matches and
   the dtype is in that implementation's set, otherwise unsupported. The other eight implementations stay
   unsupported — they genuinely are — but their placeholder comments get a real reason.
4. **Output** — a `CapabilitySupport` the caller uses to select an implementation.

### Requirement 3: `BinaryBrcInlineOp` and its emitter

1. **Description** — a new extension-API op in the Asc dialect that emits the V2 wrapper.
2. **Input** — the three tiles `dst`, `src0`, `src1`; the five scalars `shape0`, `shape1`,
   `isInput0BlockBrc`, `isInput1BlockBrc`, `firstAxisVStride`; and the wrapped API name as an attribute.
3. **Processing** — the ODS follows `AscendC_BroadcastDim3Op`: `VectorOp<"binary_brc_inline", ...>` plus
   `ApiExtensionHeader<["brc_inline_api.h"]>`. The emitter lives in
   `api/ascendc/lib/Target/AscendC/Autofuse/`, is registered in `Translation.cpp`, and renders the last
   argument as `&AscendC::<Api><T, false>`. **The parameter list is copied verbatim from the C++
   signature**, with no constant arguments dropped — the same reason `BroadcastDim3Op` keeps its
   always-zero `src_z`: dropping one forces a literal into the emitter and splits "what is emitted"
   across two files.
4. **Output** — one call line; the extension and prelude headers reach the generated kernel through the
   existing mechanism.

### Requirement 4: wiring the binary elementwise path

1. **Description** — one new implementation so matching graphs take the inline path instead of falling
   back.
2. **Input** — one more bit in `BinaryElementwiseMapping` (or `ScalarOperand` generalised into an operand
   form enum); the registration site supplies the dtype set and `ExecutionSpec`.
3. **Processing**
   - capability: the existing feature / dtype / tmp-buffer checks first, then requirement 1's predicate;
   - pattern: derive both `is_inputN_block_brc` from the predicate, compute `first_axis_v_stride` the way
     legacy does — walk the vectorized axes, and while the broadcast side's repeat is 1 record the
     **destination's** `vectorized_strides` for that axis (overwriting only when non-zero), stopping at
     the first axis whose repeat is not 1; it is the destination's row pitch, not the source's
     (`binary_api_call.cpp:177`) — materialise the two merged repeats and build requirement 3's op.
   - relation to the cache guard: legacy allows an inline-broadcast consumer to carry an
     `exec_condition` at the same time (`DoesInlineNodeNeedCache`, UT
     `CodegenKernel.BroadcastInlineWithExecCondition`). On the MLIR side `ExecConditionPass` treats every
     AFIR op alike, so the combination works without extra code — but it needs a case, otherwise "works"
     is only an inference.
4. **Output** — matching graphs emit the inline call; everything else still falls back.

## Non-Functional Requirements

### Maintainability

The predicate is written once and shared by capability and pattern (the same organisation as
`analyzeSegmentedBroadcast`). The dtype set lives at the registration site, not inside the predicate.

### Testability

Every step has something that can turn red: requirements 1 and 2 by UT (predicate truth table and query
answers), requirement 3 by lit on the emitted text, requirement 4 by lit plus a CPU E2E numeric case, and
finally a NPU re-run of the five affected cases.

### Portability

The mem and reg sides are answered separately: `brc_inline_api.h` wraps the mem-side (Level-0
`BinaryRepeatParams`) form, and the reg side has no equivalent (measured, see the end). This batch
registers the mem side only, so no implementation exists that could emit a call it cannot form.

### Reliability

No new resources. The wrapper pairs `SetMaskCount` with `SetMaskNorm` internally, so the emitter must not
insert other mask operations around the call.

### Cross-Feature Impact

Per [cross_feature_check.md](../../../../docs/guidelines/cross_feature_check.md):

| Scenario | Applicable | Analysis |
|------|--------|----------|
| SuperKernel Python interface | No | `super_kernel/` untouched |
| SuperKernel C++/AOT interface | No | No exported symbols |
| Autofuse graph optimization | No | Legacy optimize and templates unchanged; only their plans are consumed |
| Autofuse codegen/backend | Yes | A new implementation and API op change the generated kernel directly; needs both lit and E2E validation |
| AscendC API / runtime interaction | Yes | A new AscendC wrapper call with internal mask state and repeat parameters; mask pairing and the repeat-stride limit (the wrapper handles `> 255` itself) must be checked |
| Python/C++ bindings | No | Not involved |
| Build and packaging | No | The header is already in the delivery list; the new op follows the existing tablegen flow |
| Test and coverage | Yes | New UT, lit and E2E; five NPU cases re-run |
| Performance and logging | Yes | See Performance. No new default-on logging |
| Compatibility | Yes | Only forms that fall back today become owned; generated text for already-owned graphs must be byte-identical, and that is what verifies it |

## Performance

### Compile Time

One extra predicate (one pass over the descriptors along the vectorized axes) and one more implementation
to query; negligible against the existing candidate walk.

### Execution Performance

The direction is favourable: one broadcast materialization (a full UB tile write) and one buffer are
saved. **No gain may be claimed up front**: when
`cut_quotient >= block || repeat_stride_in_blocks > 255` the wrapper degenerates into a per-block counter
mode loop, and whether that beats "broadcast first, then elementwise" depends on the shape. This batch
draws no performance conclusion and only requires that it is not slower than falling back; a real number
means measuring per shape band on the NPU.

### Memory and Artifact Size

One UB buffer saved; one more inlined header in the generated kernel.

## Interface Design

### New or Modified Interfaces

No external interface. Internally: `AscendC_BinaryBrcInlineOp` (Asc dialect), one internal predicate
function, one bit in `BinaryElementwiseMapping`. The fields of `BroadcastInlineQuery` get a real meaning
for the first time, so `Capability.h` needs matching field comments.

### Interface Checklist

| Item | Sub-item | Applicable | Notes |
|--------|----------|----------|------|
| Description | Interface review needed | No | No external interface |
| Description | Documentation needed | Yes | This document plus `Capability.h` comments |
| Compatibility | Behaviour compatible | Yes | Forms that fall back today become owned; generated text for owned graphs must not change |
| Compatibility | ABI/API compatible | No | No exported symbols |
| Constraints | Clear error when unmet | Yes | Always `emitCapabilityShortfall` |
| Testing | Dedicated interface case | Yes | One per step |

## Software Design

### Key Data Structures

- The form result (broadcast-side index, the two merged repeats, the `first_axis_v_stride` expression),
  shared by capability and pattern.
- The new bit in `BinaryElementwiseMapping`; the registration site decides it and the pattern's template
  argument together, preserving the existing "single source of truth" rule.

### Key Algorithm

Merging consecutive broadcast / non-broadcast positions must agree with legacy's `MergeBrcAxisRepeats`.
**This is the likeliest place to drift**: MLIR's existing `collapseVectorizedAxes` merges by two-sided
stride contiguity, which is not the same criterion as grouping by "is this a broadcast position". On the
form measured here (two vectorized axes) the two coincide; for three or more they are unverified, so the
implementation writes legacy's grouping and says in a comment why the existing collapsing is not reused.

### Flow

`queryLowering` → form predicate → implementation selected → pattern materialises the two repeats and the
stride → build the op → emitter renders the call. Any failure becomes a capability shortfall.

### Per-Module Changes

| Directory/file | Change |
|---|---|
| `conversion/AFIRToAscendC/Implementations/BinaryElementwise.cpp` | new form, capability predicate, pattern |
| `conversion/AFIRToAscendC/Implementations/ElementwiseTransferPlan.{h,cpp}` | the row-broadcast predicate, beside `isSingletonTensorInput` |
| `target/Ascend/Capability/Capability.h` | `BroadcastInlineQuery` field comments |
| `api/ascendc/include/ascir/Dialect/Asc/IR/Autofuse/OpVecBinary*.td` | the new op |
| `api/ascendc/lib/Target/AscendC/Autofuse/` + `Translation.cpp` | emitter and registration |
| `test/ut/`, `test/lit/conversion/`, `test/e2e/cpu/cases/` | cases |

### Error Handling

#### System Errors

No allocation, no file access, no runtime calls. Missing attributes are rejected as missing information.

#### Interface Errors

Everything is reported through `emitCapabilityShortfall`, and the caller falls the whole graph back to
legacy.

## Security Review

Per [编码红线.md](../../../../docs/guidelines/编码红线.md):

- Secrets: not applicable.
- External input as index or length: vectorized axis indices come from descriptors and their lengths are
  checked for agreement first; both `is_inputN_block_brc` are 0/1 only.
- Integer overflow and division by zero: `first_axis_v_stride` and the two repeats are symbolic
  expressions materialised through the existing `materializeStrideValue` /
  `materializeSegmentExtent`; the wrapper's internal divisors are constants derived from `sizeof(T)`.
- Resource release: no new resources.
- ABI/API: no exported symbols.
- Graph rewrite equivalence: this design rewrites no graph.
- AscendC lifetime and mask: the wrapper's `SetMaskCount` / `SetMaskNorm` must stay paired and the
  emitter must not insert mask operations around the call; the reason for `isSetMask=false` goes in a
  comment (see Design Constraints).

## Compatibility

Only forms that fall back today become owned. Generated text for already-owned graphs must be
byte-identical, verified against the existing CPU E2E output. Takeover-table cells will move from
fallback to takeover; re-measure them with the existing method and record the result.

## Test Design

### Test Boundary

- Entry: UT (predicate and capability query), `af-opt` (lit), CPU E2E (numbers), NPU e2e (device numbers
  and route).
- Exit: the predicate's truth table; the emitted text; numeric comparison; `AF_MLIR_STRICT`'s `gate=` and
  `shortfall=`.
- No stubs.

### Test Cases

| Category | Key item | Method | Type |
|----------|------------|----------|------|
| Functional | Predicate truth table, including the three negatives: both sides have broadcast axes, more than 2 groups after merging, outer group repeat not 1 | new UT | UT |
| Functional | The query answers supported for this form and unsupported elsewhere | extend `BinaryElementwiseL2CapabilityTest` | UT |
| Functional | Emitted text matches legacy item by item, including `&AscendC::Add<float, false>` | new lit | lit |
| Functional | End-to-end numbers for `(1,A) + (B,A)` | new CPU E2E case | E2E |
| Cross-feature | Inline broadcast together with a cache condition | new lit | lit |
| Negative | A non-matching form reports a shortfall and falls back wholesale | new lit (`not af-opt`) | lit |
| Compatibility | Existing E2E generated text byte-identical | full suite plus text diff | E2E |
| Compatibility | Whether the diagnostic disappears for the five NPU cases | NPU run | E2E |

### Test Commands

```bash
# Fast local regression
/opt/llvm/bin/llvm-lit -q build/autofuse/mlir/test/lit

# Full local suite
bash ./mlir-test.sh

# Device box (a separate repository under ~/niu, -j 40)
bash autofuse/mlir/test/run_npu_e2e.sh --mode mlir -j 40 --case load_broadcast_add_direct
```

## Acceptance Criteria

1. The predicate's UT truth table passes and every negative can be made red on its own;
2. the lit on the emitted text matches legacy item by item;
3. the new CPU E2E numbers pass and existing E2E generated text is byte-identical;
4. the five NPU cases that today carry
   `'afir.add' / 'afir.mul' op has no supported Ascend implementation` lose that diagnostic while their
   numbers stay correct;
5. affected takeover-table cells are re-measured and recorded;
6. if the three measured conclusions at the end of this document are overturned during implementation,
   they are corrected in place rather than restated elsewhere.

# Three Measured Conclusions Before Implementation (2026-11-10)

All three were measured with a temporary ST probe (`AscGraphBuilder` building `(1, N) + (M, N)`, run
through the real optimizer and codegen with the AFIR dumped to a working directory). The probe has been
removed from the source.

**1. The bridge does not reject it earlier; the form arrives intact.** This graph's plan has four
templates, with `afir.original_graph_name` of `B0Y0`, `B0Y1`, `B0Y0_unaligned` and `B0Y0_inline`. Feeding
the dump to `af-opt --afir-plan-candidate-lowering-pipeline`, **only `_inline` fails**, with
`'afir.add' op has no supported Ascend implementation`; the other three lower today. The gap is therefore
exactly in lowering/capability and the bridge needs no change.

This also fixes the descriptor form to accept (the three descriptors of `afir.add` in the `_inline`
template):

| Side | repeats | vectorized_axis | vectorized_strides |
|---|---|---|---|
| broadcast input | `(1, 1, 1, s1)` | `[3, 1]` | `(0, 1)` |
| other input | `(s0/(z0Tb*z0t), z0Tb, z0t, s1)` | `[3, 1]` | `(8*Ceiling(s1/8), 1)` |
| destination | same as the other input | `[3, 1]` | same as the other input |

The predicate can therefore be tight: on some vectorized axis the broadcast side has repeat 1 **and**
`vectorized_strides` 0, while the other side and the destination carry a real row pitch. Note the pitch is
`8*Ceiling(s1/8)`, padded to 32B, not `s1` — `first_axis_v_stride` must take it.

**2. The merging criteria coincide on this graph; three or more axes are unverified.** This form has
exactly two vectorized axes (`[3, 1]`), where legacy's grouping by broadcast / non-broadcast position and
MLIR's collapsed segments give the same result. Whether they still agree with three or more vectorized
axes **was not verified**, so the implementation follows legacy's grouping and does not reuse
`collapseVectorizedAxes`; merging them can be reconsidered once a three-axis corpus exists.

**3. The reg side has no equivalent wrapper, so this batch is mem-side only.**
`autofuse/v35/ascendc/api_regbase/` has no brc inline wrapper and
`autofuse/v35/codegen/reg_api_call/` has no special emission for it (`BinaryApiCallV2` only builds generic
parameters). Yet `autofuse/v35/optimize/template/brc_inline_template_v2.cpp` exists, so the reg-side
optimizer does produce inline plans — how the reg side should emit them has to be established separately,
and this batch registers nothing there to avoid an implementation that cannot form its own call.

**Reproduction**: add the probe ST back to `test/st/bridge/LegacyGraphGuardSt.cpp` (described in the first
paragraph of this section), run
`GTEST_FILTER=... ctest --test-dir build/autofuse/mlir/test/st -R '^MlirCodegenSt$' -V`, then feed the
dumped `afir_0_RunMlirCodegen_*_plan0.mlir` to `af-opt`'s candidate lowering pipeline.
