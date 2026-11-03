# Elementwise / Broadcast Closeout

> English companion summary, not a full translation. The [Chinese source](2026-10-16-afir-elementwise-broadcast-closeout-design.md)
> retains the detailed work items, constraints, tables, commands, dated evidence and status updates.
> Historical results below are source records, not newly executed validation.

## Introduction

### Purpose

Close out the Autofuse MLIR elementwise and broadcast workstream by requiring graphs that legacy codegen handles on the supported mem-based and reg-base profiles to be handled by the MLIR path without fallback.

### Scope

This page summarizes the linked Autofuse MLIR record. It introduces no new runtime
behavior, API, supported platform or implementation decision.

## General Overview

### Software and Product Environment

The subject belongs to the Autofuse MLIR compilation and validation workflow.
CANN, AscendC, ATT, LLVM/MLIR, tikicpulib and profile requirements are those
documented in the source; this summary does not change their versions or
supported configurations.

### Functionality, Constraints, Assumptions and Dependencies

The source covers the AFIR elementwise family, `afir.broadcast`, their load, store,
data movement and padding dependencies, strict takeover observability, element-type
completion, VectorFunc assessment, operator completion, remaining shape/layout
boundaries, and migration of legacy E2E cases. Cube fusion, unrelated graph families,
deferred IR-shape optimization and the TensorFlow entry are out of scope. Work-item
status, including the completed N1 and strict-mode milestones, is authoritative only
in the source.

## Requirements and Design

### Overall Approach and Functional Requirements

Follow the source for inputs, processing rules, outputs, failure behavior,
module-level responsibilities and required ordering. Existing identifiers and
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
validation, arithmetic bounds, deterministic graph rewriting, verifier behavior,
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
