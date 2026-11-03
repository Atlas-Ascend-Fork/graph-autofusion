# AFIR VectorFunc Implementation

> English companion summary, not a full translation. The [Chinese source](2026-10-18-afir-vector-func-implementation-design.md)
> retains the detailed design, milestones, measurements, commands and dated evidence.
> Historical results below are source records, not newly executed validation.

## Introduction

### Purpose

Implement and validate the bridge and AFIR lowering required for scheduled `VectorFunc` nodes and their nested subgraphs on the reg-base path, based on the completed 2026-10-17 measurement and follow-up re-screening.

### Scope

This page summarizes the linked Autofuse MLIR record. It introduces no new runtime
behavior, API, supported platform or implementation decision.

## General Overview

### Software and Product Environment

The subject belongs to the Autofuse MLIR compilation and validation workflow.
CANN, AscendC, ATT, LLVM/MLIR and tikicpulib requirements are those documented in
the source; this summary does not change their versions or supported configurations.

### Functionality, Constraints, Assumptions and Dependencies

The source covers bridge translation of `VectorFunc`, region boundaries, register-level
lowering, capability handling, staged milestones, acceptance evidence and regression
measurement. SuperKernel, package delivery and Python interfaces are out of scope.
The original small-cell plan was superseded by the 2026-10-27 re-screening; milestone
status, including the recorded M1 through M4 completion, is authoritative only in the
source.

## Requirements and Design

### Overall Approach and Functional Requirements

Follow the source for inputs, processing rules, outputs, failure behavior,
module-level responsibilities and milestone sequencing. Existing identifiers and
historical commands must be interpreted with the source migration notes; this
summary adds no alternatives.

### Nonfunctional Requirements and Cross-Feature Impact

This documentation-only addition does not change SuperKernel Python or C++/AOT
interfaces, Autofuse graph rewriting, code generation, AscendC/runtime calls,
Python/C++ bindings, packaging or compatibility. Implementation requirements and
cross-feature decisions remain those of the linked source.

### Performance

No compilation, execution, memory or package-size change is introduced by this
companion page. Performance conclusions about the implementation require the source
measurement conditions and cannot be inferred from this summary.

### Interfaces and Software Design

No interface is added or modified here; no API/ABI review or caller migration is
introduced by this page. Data structures, ownership, algorithms, control flow,
module changes, system errors and interface errors are described in the source.

### Safety and Compatibility

No new code or resource lifecycle is introduced. This page does not relax input
validation, arithmetic bounds, deterministic graph rewriting, execution ordering,
runtime lifetimes, license obligations, or the source compatibility constraints.

### Test Design

The validation boundary for this change is documentation: check the companion
link, preserve the source file, and run the PR Markdown and pre-commit checks.
Implementation UT, ST, lit, CPU E2E, takeover-table, probe, mutation and
compatibility coverage must follow the linked record; creating this page does not
establish new implementation passes.

## Acceptance Criteria

- The English companion links to the authoritative Chinese record.
- The summary does not claim full translation or fresh implementation validation.
- Source implementation requirements, limitations and evidence remain unchanged.
- The companion is included with its source in the PR bilingual file check.
