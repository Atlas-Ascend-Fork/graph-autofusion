# Broadcast Cache (brc_cache) and Inline Broadcast (brc_inline): Issue List and Fix Design

> 2026-11-09 · Code baseline `d28990bd6` (develop, clean tree)
> Upstream material: BR-1 in [elementwise remaining issues](2026-09-22-afir-elementwise-remaining-issues.md),
> and the "T4 lacks a cache guard" entry for `LegacyGraphGuardRegSt.ContinuesBrc` in
> [test adjustments and re-enable conditions](2026-11-08-test-adjustments-and-reenable_en.md).

# Introduction

## Purpose

This document records the result of a full code review of the MLIR-side broadcast cache (the
`exec_condition` field of `afir.sched`) and inline broadcast (the MLIR counterpart of the legacy
`brc_inline` template): five reproduced issues, a fix design for each, and a batched fix plan. The
audience is whoever implements and reviews these five changes.

Every issue comes with a standalone reproducer; locations and commands are in the evidence appendix.
Nothing unreproduced is recorded as an issue. Two hypotheses that the review disproved are listed in
their own section so they are not investigated again.

## Scope

- In scope: within `autofuse/mlir`, the lowering of `exec_condition` (`ExecConditionPass`), guard
  construction (`materializeExecConditionGuard`), schedule axis resolution (`buildScheduledPlan`), and
  the inline-broadcast capability decision and lowering for binary elementwise ops.
- Out of scope: the three `afir.broadcast` lowering tiers (Duplicate / Segmented / Extend) — the review
  found no defect in their predicates; no change to the legacy `brc_inline_template` or
  `node_cache_marker`.
- Component: Autofuse (MLIR codegen path). SuperKernel is not involved.

# Overview

## Software Overview

### Project Background

`exec_condition` is the upstream autoschedule annotation for cacheable operations: the annotated
operation's output is invariant within one broadcast reuse cycle, so it is computed at the start of the
cycle and every other iteration reads the previous result left in UB. On the MLIR side,
`ExecConditionPass` turns that statement into an `scf.if` inside the loop body.

`brc_inline` is a legacy optimize-stage template: it removes the Broadcast node, tags the consumer with
the `kInline` semantic role, and codegen then emits an elementwise call with `src_repeat_stride = 0`,
saving one broadcast materialization and one UB buffer. MLIR has no equivalent today.

### Product Environment

Both capabilities are compile-time only: `ExecConditionPass` produces an `scf.if` that ends up as an
`if` statement in the generated AscendC kernel; inline broadcast, once implemented, emits the existing
`BinaryBrcInlineApiWithTwoVectorizedAxis` from `autofuse/ascendc/api/brc_inline_api.h`. No external
C/C++/Python interface is added and the run-package layout is unchanged.

## Software Functions

1. The cache guard predicate must match the operation's own schedule axes, never reuse another
   operation's cycle within the same loop.
2. Only operations writing a single-slot buffer may be wrapped in a cache guard; multi-slot queues must
   be rejected.
3. Operations writing GM must not be wrapped in a cache guard.
4. The fused cache condition must be constructible on multi-dimensional graphs instead of always
   rejected.
5. Inline broadcast must reach parity with the legacy `brc_inline` template.

Items 1-3 tighten behaviour (paths that today are silently wrong or order-dependent become rejections or
correct predicates), item 4 relaxes a gate, item 5 is a new capability.

## Design Constraints

- Rejection may only happen in the capability layer (`emitCapabilityShortfall`), with the caller falling
  the whole graph back to legacy; silently passing is not allowed. This is existing discipline: a
  verifier failure is a hard failure and must not be treated as a capability boundary.
- Relaxing any gate requires numeric proof; inspecting generated text is not enough.
- Legacy must not be modified for MLIR's acceptance. Where legacy does not support a shape, the answer
  is "unsupported" plus fallback.
- No new hardcoded SoC or platform checks.

## Assumptions and Dependencies

- The upstream gives cacheable tensors `buf_num = 1`. This assumption **does not hold**:
  `buf_que_allocator.cpp:700` only gives 1 for VecOut when `is_node_cached && !is_reusable`, so cached
  multi-slot tensors exist. Issue 2 addresses exactly this.
- The split axis always lands in segment 0 after axis collapsing (`getTileInnerAxis` takes `front()`).
  This design does not relax that assumption.
- Issue 3 involves one deliberate divergence from legacy behaviour, confirmed with the user as
  "reject rather than guess".

# Requirement Analysis and Design

## Overall Approach

The first three changes are confined to `dialect/AFIR/Transforms/ExecConditionPass.cpp`: each adds a
predicate before the guard is inserted, or corrects the condition under which a predicate is reused. Data
flow and control flow are unchanged. The fourth change touches axis resolution and guard construction in
`dialect/AFIR/Analysis/TransferPlan.cpp` and is a behavioural relaxation. The fifth spans
`target/Ascend/Capability` and `conversion/AFIRToAscendC` and needs its own design document.

## Functional Requirements

### Requirement 1: a guard predicate must not be reused across schedule axes (P3, batch 1)

1. **Description**

   `GuardCache` keys on `(loop, condition)` but takes the predicate value from the first operation
   processed in that loop (`ExecConditionPass.cpp:59`, `:61`). The fused predicate depends on
   `blockOuterAxis`, `loopAxis` and `tileOuterAxis`; `tileOuterAxis` comes from the operation's own
   outermost vectorized tile axis, which can differ between operations in one loop. The consequence is
   that for the same graph and the same operation the answer depends on traversal order:

   - `brc_cache_3a`: the `load` carries the condition first and the `broadcast`'s axes do not resolve →
     two `scf.if`, no diagnostic, and the `broadcast` runs on the `load`'s cycle;
   - `brc_cache_3b`: the same graph with only the `broadcast` carrying the condition → compile-time
     rejection.

2. **Input**

   Several AFIR operations in one function carrying a non-`NoCache` condition and sharing one `scf.for`.

3. **Processing**

   Before `GuardCache::get` returns a cache hit, resolve the three axes from the current operation's own
   plan: on failure return `failure()` (translated into a capability shortfall by `guardOperation`); on
   success but with axes different from the cached entry, build a separate predicate. An equivalent
   implementation extends the key to `(loop, condition, blockOuterAxis, loopAxis, tileOuterAxis)` and
   records the three axes when inserting. The `origin` condition does not depend on these axes, so they
   stay -1 in the key and reuse behaves exactly as today.

4. **Output**

   Operations with the same cycle still share one predicate (the existing `CNT-COUNT-2` assertion stays
   valid); different cycles get their own; an operation whose axes do not resolve is always rejected,
   regardless of its position in the loop.

### Requirement 2: reject cache guards on multi-slot buffers (P2, batch 1)

1. **Description**

   The guard skips only the operation itself; `AllocTensor` / `EnQue` / `DeQue` / `FreeTensor` are issued
   every iteration. Only with a single slot is the address identical each time, which is what makes the
   skipped iteration read the previous result. A multi-slot queue rotates every iteration, so a skipped
   iteration reads data from two iterations ago. `ExecConditionPass` never inspects the slot count
   (`brc_cache_2`: changing `slot_count` from 1 to 2 alone still produces the guards, with `init_queue`
   initialising two slots and no diagnostic at all).

2. **Input**

   The `afir.buffer_desc` of every destination operand of the guarded operation. After resource
   materialization the attribute still hangs on the `reinterpret_cast` and is read with
   `getBufferDescAttr(Value)` (`dialect/AFIR/IR/AFIR.h:250`).

3. **Processing**

   Walk the destinations before inserting the `scf.if`: a missing buffer_desc is reported as missing
   information; `slot_count > 1` emits a capability shortfall stating that the cache guard requires a
   single-slot buffer. `slot_count == -1` (TBuf and other buffers that do not set the field) is treated
   as 1 — such buffers do not rotate.

4. **Output**

   Single-slot graphs are unchanged; multi-slot graphs have the whole plan rejected with a locatable
   diagnostic and fall back to legacy.

### Requirement 3: reject cache guards on operations writing GM (P4, batch 1)

1. **Description**

   `runOnFunction` collects condition-carrying operations without distinguishing their kind
   (`ExecConditionPass.cpp:140`), so an `afir.store` carrying a cache condition is wrapped and the output
   is only written at cycle starts (`brc_cache_4`: three `scf.if`, with the store inside one). Legacy
   behaves the same way (`enable_cache = IsValidCacheCondition(exec_condition)`, no per-kind check) and
   the upstream never produces such an annotation, so this is not a parity gap but an unvalidated input
   invariant. Handled as "reject rather than guess".

2. **Input**

   Operations carrying a non-`NoCache` condition whose destination lives in GM space.

3. **Processing**

   During collection, inspect the destination's `buffer_desc` space: GM emits a capability shortfall
   stating that the cache condition only applies to on-chip results. The predicate is written on the
   memory space rather than the operation kind, which avoids enumerating every GM-writing operation
   besides `afir.store`.

4. **Output**

   A store carrying a cache condition has the whole plan rejected with a diagnostic instead of silently
   writing only the first iteration.

### Requirement 4: construct the fused cache predicate on multi-dimensional graphs (P1, batch 2)

1. **Description**

   `collectTransferOuterAxes` resolves `blockOuterAxis` only when
   `BlockOuter.split_pair_other_id == plan.loopAxis` (`TransferPlan.cpp:464`), i.e. it requires
   `loop_axis` to be the paired BlockInner. That holds for one-dimensional graphs; on multi-dimensional
   graphs `loop_axis` is usually the innermost dimension's TileOuter, so neither `blockIndex` nor
   `tileLoopSize` can be found and the fused condition is always rejected (`brc_cache_1`: diagnostic
   `cannot build the cache guard`). The same graph with the `origin` condition lowers fine, which places
   the gap precisely in the fused branch. This is the source of BR-1's diagnostic and the blocker behind
   T4 of `LegacyGraphGuardRegSt.ContinuesBrc`.

2. **Input**

   A multi-dimensional schedule where the block split is on one original axis and the cached operation's
   outermost vectorized tile axis may not share an origin with `loop_axis`.

3. **Processing**

   Decouple axis resolution from `loop_axis`:

   - `tileOuterAxis` is still the TileOuter paired with the cached operation's outermost vectorized tile
     axis; it defines the reuse cycle length;
   - the block axes are taken from the BlockOuter / BlockInner pair that **shares an origin** with
     `tileOuterAxis` (walk the `from` chain up to the same Original axis), instead of requiring a pairing
     with `loop_axis`;
   - when `loop_axis` shares an origin with `tileOuterAxis`, the predicate stays
     `iv == 0 || (blockIndex * blockInnerSize + iv) % tileOuterLoopSize == 0`;
   - otherwise that dimension's tile index is invariant inside this loop, so the cached value is
     invariant for the whole loop and the predicate collapses to `iv == 0`. This is tighter than legacy's
     cross-dimension modulo: in this shape legacy's formula only recomputes more often, it never
     under-computes, so tightening is the safe direction;
   - if no same-origin block axis resolves, or the axis values are not materialized, keep rejecting.

4. **Output**

   Broadcast caching on multi-dimensional graphs no longer always falls back; the predicate shape is
   decided by axis origin relationships, not by the chip.

### Requirement 5: inline broadcast (P5, batch 3, needs its own design document)

1. **Description**

   MLIR has no inline broadcast: no production code anywhere in the repository issues a
   `CapabilityKind::BroadcastInline` query, all nine `queryBroadcastInline` overrides return unsupported,
   and the `axis` / `sourceStrides` / `destinationStrides` / `hasDynamicShape` fields of
   `BroadcastInlineQuery` are never populated. On the lowering side only
   `isSingletonTensorInput` (`ElementwiseTransferPlan.cpp:25`) exists, i.e. "all repeats are 1 → read as
   a scalar". MLIR also never reads the upstream template semantic role (`template_semantic` and
   `kInline` have zero hits inside `autofuse/mlir`), so a plan for which legacy chose `brc_inline` looks
   to MLIR like an elementwise graph with incompatible shapes (`brc_inline_1`:
   `'afir.add' op has no supported Ascend implementation`).

2. **Input**

   The accepted shape follows legacy `IsGeneralizeBrcInlineScene` (`common/common_utils.cpp:856`): two
   tensor inputs, broadcast axes on exactly one side, two groups per side after merging consecutive axes,
   and the broadcast side's outer group repeat equal to 1 — that is, `(1, A) op (B, A)`.

3. **Processing**

   Two steps, both to be detailed in a separate design document:

   - capability: make "one side is a row broadcast" a form of binary elementwise (structurally like the
     existing `ScalarOperand`), which issues the `BroadcastInline` query and populates its fields, giving
     that interface a real caller;
   - lowering: emit the `src_repeat_stride = 0` form by reusing
     `BinaryBrcInlineApiWithTwoVectorizedAxis` from `brc_inline_api.h`, which requires a matching op in
     the Asc dialect and in the extension-header registry.

4. **Output**

   `(1, A) op (B, A)` no longer falls back, and the legacy ST case `brc_inline_test` can enter the
   takeover table.

## Two Disproved Hypotheses

Recorded so they are not re-investigated:

- **Conditions on operations inside a `vector_func` region being skipped** is not a defect. Legacy's
  `VectorFuncPartitioner` makes the vf node inherit the `sched` of the first output producer and demotes
  the whole node to `kNoCache` when the outputs disagree (`vector_func_partitioner.cpp:1404`, `:1419`),
  which is equivalent to MLIR's "no guard inside the region, the vf's own condition governs".
- **Taking the block index from `AxisIndex` rather than `GetBlockIdx()`** is not a defect. With a single
  block axis the two are equal; with several, per-axis coordinates are more accurate than legacy's raw
  `GetBlockIdx()`.

## Non-Functional Requirements

### Maintainability

The first three changes land in two existing functions of `ExecConditionPass.cpp` (`GuardCache::get` and
`guardOperation` / `runOnFunction`), add no types and do not change pass ordering. Every predicate is
expressed through `emitCapabilityShortfall`, structurally identical to the three diagnostics already in
that file.

### Testability

Each of the three changes has a reproduced lit case that becomes a passing assertion after the fix, via
either the rejection diagnostic or the predicate count. Requirement 4 needs two more cases (same-origin
and cross-origin) plus numeric validation.

### Portability

No chip capability queries and no new platform branches. The lit cases keep the existing
`--target=Ascend910B1`.

### Reliability

The three changes only add rejection paths: no new allocations, no change to resource lifetimes. The
predicate-reuse fix only affects how many `arith` constants and comparisons are created.

### Cross-Feature Impact

Per [cross_feature_check.md](../../../../docs/guidelines/cross_feature_check.md):

| Scenario | Applicable | Analysis |
|------|--------|----------|
| SuperKernel Python interface | No | Changes are confined to `autofuse/mlir`; `super_kernel/` untouched |
| SuperKernel C++/AOT interface | No | No exported symbols, `libascendsk.so` unchanged |
| Autofuse graph optimization | No | ASCIR graph structure and legacy passes unchanged; `exec_condition` is only read |
| Autofuse codegen/backend | Yes | `ExecConditionPass` and `TransferPlan` are MLIR codegen; the first three changes only add rejections, affected graphs fall back wholesale, and generated text for single-slot same-cycle graphs is byte-identical |
| AscendC API / runtime interaction | No | No new `rt*` / `aclrt*` / AscendC calls; synchronization and lifetime order around the guard unchanged |
| Python/C++ bindings | No | `py_module` untouched |
| Build and packaging | No | No CMake or install path changes; new lit cases are picked up by the existing directory |
| Test and coverage | Yes | One lit case per change; requirement 4 adds two cases plus numeric validation |
| Performance and logging | Yes | See the performance section; no new default-on logging |
| Compatibility | Yes | The first three changes turn two classes of graph that generate kernels today into fallbacks; see the compatibility section |

## Performance

### Compile Time

The new predicates in batch 1 are a constant number of attribute reads (one destination walk per guarded
operation), negligible against the plan construction the pass already performs. The predicate-reuse fix
builds one extra predicate (three `arith` ops plus a comparison) only for graphs that should be rejected
today.

### Execution Performance

Generated text for single-slot same-cycle graphs is byte-identical, so kernel performance is unchanged.
After requirement 4, broadcasts on multi-dimensional graphs move from recomputing every iteration to
recomputing at cycle starts, which is a gain; the cross-origin predicate is tighter than legacy's, so it
recomputes less often.

### Memory and Artifact Size

No new data structures or resident memory, no impact on run-package or shared-library size.

## Interface Design

### New or Modified Interfaces

No external interface changes. One internal signature may change in batch 2: the axis resolution behind
`materializeExecConditionGuard` moves into a new `TransferPlan` field (candidate: add `blockInnerAxis`) or
is resolved in place by walking the `from` chain. The choice is made when batch 2 starts, together with
the field comments in `TransferPlan.h`.

### Interface Checklist

| Item | Sub-item | Applicable | Notes |
|--------|----------|----------|------|
| Description | Interface review needed | No | No external interface |
| Description | Documentation needed | Yes | This document; batch 2 also updates `TransferPlan.h` comments |
| Compatibility | Behaviour compatible | Yes | The first three changes turn two graph classes from generated into fallback |
| Compatibility | ABI/API compatible | No | No exported symbols |
| Constraints | Clear error when unmet | Yes | Always via `emitCapabilityShortfall`, pointing at the operation |
| Testing | Dedicated interface case | Yes | One lit case per item |

## Software Design

### Key Data Structures

- `GuardCache` (`ExecConditionPass.cpp:59`): the cached value grows from `Value` to "`Value` plus three
  axes", or the three axes join the key. The two forms are equivalent; keeping them in the value is
  needed only if a diagnostic should be emitted on mismatch.
- `TransferPlan` (`TransferPlan.h`): batch 2 may add `blockInnerAxis`. The field comments already state
  that -1 means unresolved and consumers must handle the absence, and the new field follows that
  convention.

### Key Algorithm

The same-origin test for requirement 4 walks `AxisAttr::from` up to the Original axis and compares the
two axes' roots. The axis table comes from `collectGraphAxes`; the `from` chain is finite and acyclic (the
bridge guarantees this when building the graph attribute), and the walk is bounded by the axis table size
so a malformed table cannot cause an infinite loop.

### Flow

The main flow is unchanged: `runOnFunction` collects → `guardOperation` per operation → build the
predicate → `detachResults` → create the `scf.if` → move the operation in. All new predicates sit before
"build the predicate"; any failure returns `failure()` for the whole function and the pass calls
`signalPassFailure`, leaving no half-built IR.

### Per-Module Changes

| Directory/file | Change | Batch |
|---|---|---|
| `dialect/AFIR/Transforms/ExecConditionPass.cpp` | predicate reuse condition, multi-slot rejection, GM destination rejection | 1 |
| `dialect/AFIR/Analysis/TransferPlan.cpp` | same-origin block axis resolution, two fused predicate shapes | 2 |
| `dialect/AFIR/Analysis/TransferPlan.h` | fields and comments | 2 |
| `test/lit/conversion/` | three new cases (batch 1), two (batch 2), one (batch 3) | each |
| `target/Ascend/Capability/`, `conversion/AFIRToAscendC/Implementations/BinaryElementwise.cpp` | inline broadcast capability and lowering | 3 |

### Error Handling

#### System Errors

These changes allocate no memory, touch no files and call no runtime. Attribute lookup failures (no
buffer_desc, no axis table) are rejected as missing information.

#### Interface Errors

All errors are reported through `emitCapabilityShortfall`, on which the caller falls the graph back to
legacy. No new error codes and no change to user-visible messages.

## Security Review

Per [编码红线.md](../../../../docs/guidelines/编码红线.md):

- Hardcoded secrets or public addresses: not applicable.
- External input used as index/length/offset: `slot_count` and axis ids come from graph attributes and are
  range-checked (including the -1 sentinel) before use; the `from` walk is bounded by the axis table size.
- Integer overflow and division by zero: the `%` divisor is `tileOuterLoopSize`, taken from the axis
  `loop_size` exactly as today; no new multiplications are introduced.
- Resource release on exception paths: no new resources.
- ABI/API compatibility: no exported symbols.
- Graph rewrite equivalence and determinism: the new predicates do not rewrite the graph; the number of
  predicates depends only on axis id comparisons, never on hash container iteration order or pointer
  addresses (an extended cache key still compares axis ids).
- Runtime/AscendC lifetime constraints: the order of `AllocTensor` / `EnQue` / `DeQue` / barriers around
  the guard is unchanged.

## Compatibility

The first three changes turn two classes of graph that generate kernels today into compile-time
rejections with a wholesale fallback to legacy:

1. graphs whose cached tensor lives in a multi-slot queue — today's kernel reads data from two iterations
   ago, which is a silent wrong result, so rejection is a correction;
2. graphs whose `afir.store` carries a cache condition — today's kernel writes only at cycle starts,
   equally a silent wrong result.

Neither affects the legacy route, and generated text for single-slot same-cycle graphs is unchanged.
Whether any takeover-table cell moves from "MLIR pass" to "fallback" must be checked: if so, those cells
were wrong to begin with and the reason must be recorded in the table; the predicates must not be
loosened to preserve a count.

## Test Design

### Test Boundary

- Entry: `af-opt` with pass combinations (lit) and the full `--afir-plan-candidate-lowering-pipeline`.
- Exit: diagnostic text for rejections; counts and structure of `scf.if` and `arith.cmpi` for predicates.
- No stubs: existing passes and hand-written AFIR input only.

### Test Cases

| Category | Key item | Method | Type |
|----------|------------|----------|------|
| Functional | Same-cycle operations share one predicate (existing count assertion holds) | `afir-exec-condition-cache-guard.mlir` keeps passing | lit |
| Negative | Different cycle / unresolved axes always rejected, independent of traversal order | new pair (from `brc_cache_3a/3b`) | lit |
| Negative | Cache guard on a multi-slot queue rejected | new case (from `brc_cache_2`) | lit |
| Negative | GM destination with a cache condition rejected | new case (from `brc_cache_4`) | lit |
| Functional | Both fused predicate shapes on multi-dimensional graphs | two new cases in batch 2 | lit |
| Cross-feature | Numeric correctness of multi-dimensional broadcast caching | batch 2: CPU e2e and NPU comparison | E2E |
| Functional | `(1,A) op (B,A)` inline broadcast | batch 3 | lit + E2E |
| Compatibility | Takeover-table cell changes | re-measure affected rows after batch 1 lands | probe |

### Test Commands

```bash
# Fast regression (local, prebuilt af-opt)
/opt/llvm/bin/llvm-lit -q build/autofuse/mlir/test/lit

# Full suite
bash ./mlir-test.sh
```

Expected: all lit cases pass; after batch 1 the three existing `afir-exec-condition-*.mlir` cases are
unchanged and the three new cases pass.

## Acceptance Criteria

- Batch 1: the three new lit cases pass; the predicate count assertion in
  `afir-exec-condition-cache-guard.mlir` does not regress; the full lit suite passes; changes to affected
  takeover-table cells are recorded.
- Batch 2: the fused condition generates kernels on multi-dimensional graphs with a case per predicate
  shape; numbers match legacy on CPU e2e and NPU; the T4 blocker of
  `LegacyGraphGuardRegSt.ContinuesBrc` is gone.
- Batch 3: per its own design document.

# Fix Plan

| Batch | Content | Direction | Acceptance depends on |
|---|---|---|---|
| 1 | Requirements 1, 2, 3 (P3 → P2 → P4) | tightening | local lit |
| 2 | Requirement 4 (P1) | relaxation | remote e2e / NPU numbers |
| 3 | Requirement 5 (P5) | new capability | separate design document |

All three items in batch 1 live in `ExecConditionPass.cpp`, do not conflict, and ship as one commit.
Batch 2 depends on batch 1 having cleaned up the "cannot build it, so reject" path, otherwise a relaxation
cannot be told apart from a reused predicate. Batch 3 touches a disjoint set of files and can proceed in
parallel once its design document exists.

# Evidence Appendix

Reproducers live in this review's working directory (not under version control):
`/tmp/claude-1000/-home-zhe-code-autofuse/c0aed98b-c37a-484a-b928-c7bff9d59139/scratchpad/brc/cases/`,
where `README.md` lists each case's current behaviour and the location of the predicate, and each `.mlir`
file header records the reproduction command and the expected behaviour.

| Case | Requirement | Current behaviour |
|---|---|---|
| `brc_cache_1_two_dim_loop_axis.mlir` | 4 | rejected: `cannot build the cache guard` |
| `brc_cache_2_multi_slot_queue.mlir` | 2 | two `scf.if`, queue with two slots, no diagnostic |
| `brc_cache_3a_guard_reused_across_plans.mlir` | 1 | two `scf.if`, no diagnostic |
| `brc_cache_3b_guard_rejected_alone.mlir` | 1 | rejected (same graph and operation as 3a) |
| `brc_cache_4_store_guarded.mlir` | 3 | three `scf.if`, store wrapped in the conditional |
| `brc_inline_1_row_broadcast.mlir` | 5 | rejected: `'afir.add' op has no supported Ascend implementation` |

Tooling and baseline: `build/autofuse/mlir/tools/af-opt/af-opt` (baseline `d28990bd6`);
`/opt/llvm/bin/llvm-lit -q build/autofuse/mlir/test/lit` had all 290 cases passing when the review started.

# Batch 1 Verification Record (2026-11-09)

Change: `dialect/AFIR/Transforms/ExecConditionPass.cpp` (+70 −3), plus three new lit cases
`afir-exec-condition-guard-reuse-unsupported.mlir`, `afir-exec-condition-multi-slot-unsupported.mlir`
and `afir-exec-condition-store-unsupported.mlir`.

- lit: 293 cases pass (the previous 290 plus 3 new); assertions in the three existing
  `afir-exec-condition-*.mlir` cases were not touched.
- Mutation check: with the production change reverted and rebuilt, exactly the three new cases fail and
  the three existing ones stay green — the cases pin this change and nothing else.
- Full suite: `bash ./mlir-test.sh` ends with
  `Autofuse MLIR targets succeeded: check-ut check-lit check-st check-e2e`, 3m31s.
  So no graph in the current UT/ST/E2E-CPU corpus has a cached tensor in a multi-slot queue or a cache
  condition on a GM destination; this tightening turned no existing case from generated into fallback.
- **Not covered**: re-measuring takeover-table cells needs the remote probe and was not done in this
  round. If a later measurement shows a cell moving from "MLIR pass" to "fallback", handle it per the
  compatibility section; the predicates must not be loosened to preserve a count.

# Batch 2 Verification Record (2026-11-10)

Change: `dialect/AFIR/Analysis/TransferPlan.{h,cpp}`, plus the lit cases
`afir-exec-condition-two-dim-cross-origin.mlir` and `afir-exec-condition-two-dim-same-origin.mlir`.

One correction to the original design: it said the block axes are taken from the pair that "shares an
origin (walk the `from` chain to the same Original axis)" with the tile axis. Implemented against a
single origin axis, `load_broadcast_segmented_store_8x1` went from lowering to rejected: that graph
merges axes before splitting blocks (`z0z1T` derives from both z0 and z1), and a single-origin test has
no answer for a merged axis. The implementation now compares **origin sets**: the loop axis's set
covering the tile axis's set counts as the same dimension (merged forms land here), disjoint sets count
as another dimension, and partial overlap is treated as unresolved.

- lit: 295 cases pass (293 after batch 1, plus 2 new).
- Mutation check: reverting this batch turns only
  `afir-exec-condition-two-dim-cross-origin.mlir` red. The same-origin case passes before and after, so
  it is a regression guard rather than a mutation witness; its header says so.
- Local full suite: `Autofuse MLIR targets succeeded: check-ut check-lit check-st check-e2e`.
- Remote device box (`npussh mlir`, `~/niu/brc-verify`, cloned from `cann/graph-autofusion` `develop`
  whose HEAD is the baseline `d28990bd6`, `-j 40`): `check-ut`, `check-lit` (295/295) and `check-st`
  pass; `check-e2e` fails at link time for three transpose cases because CANN's `libtiling_api.a` needs
  `dlopen` and the link line lacks `-ldl`. With this batch stashed, the same target fails identically on
  the baseline, so it is an environment issue on that box, unrelated to this change.
- Remote NPU run (`run_npu_e2e.sh --mode mlir`, seven broadcast cases): `All NPU E2E cases passed.`,
  all numbers correct, `backend_route=mlir`. `load_broadcast_to_abs` and
  `load_broadcast_inner_brc_cascade` are `PASS`; the five previously recorded as `PASS_FALLBACK` still
  pass numerically with a known rejection diagnostic, and every remaining diagnostic is
  `'afir.add' / 'afir.mul' op has no supported Ascend implementation`, i.e. the requirement-5
  (brc_inline) gap rather than the cache guard.

## Evidence this batch did not obtain

**The "other dimension" branch has no witness in the corpus.** The `cache guard` diagnostic appears zero
times in the local full-suite logs before and after the change, in the remote full-suite log, and in the
NPU logs; a pre-change baseline run of `load_broadcast_to_cascade` also shows zero (its blocker was
always the elementwise one). No graph in the current corpus is held back by this gate, so this batch
lowers **no additional graph**, and the new branch is pinned only by a hand-written lit case.

The conclusion is therefore limited to two statements:

1. No regression — both full suites and the NPU numbers pass, and the rewritten axis resolution is
   genuinely exercised by real graphs (the merged-axis `load_broadcast_segmented_store_8x1`,
   `load_broadcast_duplicate_store` and others); they are what caught the first implementation's error.
2. Numeric validation of the "other dimension" branch is still missing. Per the 2026-11-08 test
   adjustment record, the removed ST `LegacyGraphGuardRegSt.ContinuesBrc` has exactly "missing cache
   guard" as its T4 blocker, so restoring that ST (or adding an e2e case of that shape) is the next step
   towards this witness; this batch being green is not a substitute.

# Attempt to Obtain the "Other Dimension" Witness (2026-11-10)

The goal was a real optimizer-produced graph that exercises requirement 4's "other dimension" branch.
None of the three routes produced a witness, and the attempt uncovered a new silent-wrong-result defect,
recorded in the next section.

**1. Restoring `LegacyGraphGuardRegSt.ContinuesBrc` as a probe.** Restored verbatim from the 2026-11-08
record and run on its own, the first blocker on the reg side is not the cache predicate but
`UnsupportedOperation: VectorFunc subgraph node 'brc2' carries a cache exec_condition, which the
register-form lowering does not model yet` (`bridge/ScheduleToAfir/ScheduleToAfir.cpp:1454`): a
subgraph node carrying a cache condition is rejected as "not modeled" in the bridge, long before
predicate construction. Running the same graph on the mem side, the blocker is
`'afir.broadcast' op has no supported Ascend implementation` (five-dimensional chained broadcast with
multiple consumers). Neither side is a witness. The probe has been removed from the source.

**2. Scanning the generated kernels of the existing CPU E2E corpus.** Classifying the 70 generated
kernels that contain an `if`: all 30 broadcast-related cases use the **phase form**
(`bool g = iv == 0 | phase == 0`, i.e. the same-dimension branch); not one uses the first-iteration-only
form. The corpus genuinely does not cover "other dimension".

**3. Constructing a shape that forces the inner axis to be split.** Registered
`[1, 65536] → [33, 65536]` under `load_broadcast_abs_store`: 65536 f32 on the inner axis exceeds UB, so
the schedule must split it and loop over it, while the broadcast axis sits outside and lands on TILE_IN
— exactly the "other dimension" geometry. In practice the schedule emits no cache condition at all for
this shape, and the generated kernel contains no `Broadcast` / `Duplicate` call, only one `DataCopy`,
with wrong numbers (see the next section). Both the registration and the temporarily relaxed shape
assertion used to obtain the kernel text have been reverted.

**Conclusion**: the "other dimension" branch is still pinned only by
`afir-exec-condition-two-dim-cross-origin.mlir`. A corpus witness needs a graph that satisfies two
conditions at once: TILE_IN is a broadcast axis (otherwise `NodeCacheMarker` emits no fused condition),
and another dimension is also split with the loop running on it. The third route's shape satisfies the
geometry but runs into the defect below.

# New Issue 6: a two-dimensional broadcast with a large inner axis lowers to a kernel without any broadcast call, and most of the output stays zero

**Symptom**: after registering `OUTPUT_SHAPE "[33, 65536]"` with `INPUT_SHAPES "[1, 65536]"` (f32) under
`load_broadcast_abs_store`, `AF_MLIR_STRICT gate=none shortfall=no` — MLIR claims the graph — yet the
generated kernel contains no `Broadcast` / `Duplicate` call, only one `AscendC::DataCopy`, and the CPU
simulation comparison fails over most of the output (expected 3, got 0; expected 2.75, got 0).

**Nature**: claiming the graph and then computing silently wrong results, which is worse than falling
back. **Not introduced by this round's two batches**: reverting `TransferPlan.{h,cpp}` and
`ExecConditionPass.cpp` to the baseline `d28990bd6` and rebuilding reproduces the same wrong output for
the same shape.

**Reproduction**: add that shape to
`test/e2e/cpu/cases/load_broadcast_abs_store/CMakeLists.txt`, reconfigure cmake, then
`cmake --build build --target check-e2e-cpu-load_broadcast_abs_store_1x65536`. The case's own
`CheckGeneratedKernel` fails first on the missing `Broadcast(` and therefore no kernel text is written,
so investigating it requires temporarily relaxing that assertion.

**Next step**: first establish at which layer the broadcast disappears (whether the schedule removed the
Broadcast node, or the lowering folded it into a transfer), then decide between adding a capability
predicate that rejects the shape and fixing that layer. This issue is independent of this document's
five items and should rank above the "other dimension" witness: it is a silent wrong result, whereas the
missing witness is only a coverage gap.
