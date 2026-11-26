# Remaining Elementwise Issues

English | [简体中文](./2026-09-22-afir-elementwise-remaining-issues.md)

> English companion summary, not a full translation. The [Chinese source](2026-09-22-afir-elementwise-remaining-issues.md)
> retains the detailed design, tables, commands, dated evidence and open decisions.
> Historical results below are source records, not newly executed validation.

## Introduction

### Purpose

Track the remaining elementwise and related lowering gaps and the evidence needed to close each item.

### Scope

This page summarizes the linked Autofuse MLIR record. It introduces no new runtime
behavior, API, supported platform or implementation decision.

## General Overview

### Software and Product Environment

The subject belongs to the Autofuse MLIR compilation and validation workflow.
CANN, AscendC, ATT and LLVM/MLIR requirements are those documented in the source;
this summary does not change their versions or supported configurations.

### Functionality, Constraints, Assumptions and Dependencies

Use the source item identifiers, reproduction cases and follow-up notes. Later probes may invalidate earlier assumptions; do not treat an old prerequisite list as a current implementation contract.

## Requirements and Design

### Overall Approach and Functional Requirements

Follow the source for inputs, processing rules, outputs, failure behavior and
module-level responsibilities. Existing identifiers and historical commands must
be interpreted with the source migration notes; this summary adds no alternatives.

### Nonfunctional Requirements and Cross-Feature Impact

This documentation-only addition does not change SuperKernel Python or C++/AOT
interfaces, Autofuse graph rewriting, code generation, AscendC/runtime calls,
Python/C++ bindings, packaging or compatibility. Implementation requirements and
cross-feature decisions remain those of the linked source.

### Performance

No compilation, execution, memory or package-size change is introduced by this
companion page. Performance conclusions about the implementation require the
source measurement conditions and cannot be inferred from this summary.

### Interfaces and Software Design

No interface is added or modified here; no API/ABI review or caller migration is
introduced by this page. Data structures, ownership, algorithms, control flow,
module changes, system errors and interface errors are described in the source.

### Safety and Compatibility

No new code or resource lifecycle is introduced. This page does not relax input
validation, arithmetic bounds, deterministic graph rewriting, runtime lifetimes,
license obligations, or the source compatibility constraints.

### Test Design

The validation boundary for this change is documentation: check the companion
link, preserve the source file, and run the PR Markdown and pre-commit checks.
Implementation UT, ST, lit, CPU/NPU E2E, performance and compatibility coverage
must follow the linked record; creating this page does not establish new passes.

## Acceptance Criteria

- The English companion links to the authoritative Chinese record.
- The summary does not claim full translation or fresh implementation validation.
- Source implementation requirements, limitations and evidence remain unchanged.
- The companion is included with its source in the PR bilingual file check.


## 2026-11-19 Addendum: remaining gaps after the second dtype wave (i8 / ui32 / ui64)

Upstream: the second dtype wave PR (i8 / ui32 / ui64 promoted into `AFIR_ELEMENT_TYPE_TABLE`,
bits 8/9/10). This section lists only the combinations still NOT wired into MLIR and why;
the accepted parts are in that PR's description. Criteria follow `ElementTypeMatrixTest`:
"deferred" = ODS accepts, legacy supplies, registration does not, selection falls back cleanly.

### Capability gaps (CANN call / wrapper does not support; needs a mechanism)

| Combination | Reason | Path to closure |
|---|---|---|
| Mul i8/ui8 (tensor), i8/ui8/ui32 (scalar) | contiguous `Mul` / `Muls` lack the SupportType entries (`kernel_operator_vec_binary_*.h`) | add a `MulExtendRegBaseOp` wrapper (Sub precedent) or wait for CANN type widening |
| Adds / Maxs / Mins ui16/ui32 (scalar) | the three static_assert tables in the same header | same, or route scalar forms through micro/SIMT (verify what legacy emits) |
| Fma i8/i16/ui8/bf16 | `AscendC::Fma` accepts half/float only; legacy casts via `GetConversionDtype` first | build an input-cast mechanism for elementwise |
| FloorDiv i8/ui8/i64 | wrapper dispatch has no int8/uint8 branch; i64 SIMT lacks INT64_MIN/-1 handling | extend the wrapper / overflow handling (needs review) |
| Broadcast duplicate tier i8 (both sides), mem ui64/i64 | duplicate.h 1-byte overload not wired; mem four-arg tail form not built | `DuplicateI8Op` (DuplicateU8Op precedent); new tmp-tail dialect form |
| Transpose mem i32/ui32, reg i64 | v1 layout vs formula mismatch (legacy broken in production too); Gather lacks 8-byte | layout/formula alignment (standalone bug fix) |

### Split-row gaps (Any/Shared single row can only take the two-column intersection; pre-existing)

exp / ln / sqrt reg bf16; relu reg ui8/i64; ub2ub reg i64; axpy reg bf16/i64/ui64 (axpy is
additionally blocked by the family rule: reg rows require a `_reg_base` header, `axpy.h` is
neutral). Uniform closure path: split into mem/reg rows; axpy needs `headersMatchFamily` widened.

### Mechanism-level

- reduce (Max/Min/Sum) reg-side types incl. i8: MLIR reduce is mem-contract only; taking the
  reg side means a whole new reg reduce implementation (dialect op + ApiCall + pattern), a
  standalone task on par with Ub2ub/Concat.
- compare mem scalar i64: compare.h i64 scalar overloads support EQ/NE only and need extra scratch.
