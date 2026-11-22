# FP16 Fused-Chain Rounding

> English companion summary, not a full translation. The [Chinese source](2026-10-22-fp16-fused-chain-rounding-issue.md)
> retains the detailed design, tables, commands, dated evidence and open decisions.
> Historical results below are source records, not newly executed validation.

## Introduction

### Purpose

Record the difference between eager FP16 rounding after every operator and fused execution with FP32 intermediates followed by a final FP16 conversion.

### Scope

This page summarizes the linked Autofuse MLIR record. It introduces no new runtime
behavior, API, supported platform or implementation decision.

## General Overview

### Software and Product Environment

The subject belongs to the Autofuse MLIR compilation and validation workflow.
CANN, AscendC, ATT and LLVM/MLIR requirements are those documented in the source;
this summary does not change their versions or supported configurations.

### Functionality, Constraints, Assumptions and Dependencies

The source reports 89 of 96 cases passing and seven numerical failures under the original tolerances, unchanged across the compared scratch-related commits. Nine additional probes matched the FP32-chain oracle exactly. The decision on semantics versus oracle/tolerance remains deferred; failures must not be relabeled as passes.

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
