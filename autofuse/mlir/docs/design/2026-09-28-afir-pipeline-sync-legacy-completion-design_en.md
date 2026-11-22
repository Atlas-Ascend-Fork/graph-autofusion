# MLIR Pipeline Synchronization Completion

> English companion summary, not a full translation. The [Chinese source](2026-09-28-afir-pipeline-sync-legacy-completion-design.md)
> retains the detailed design, tables, commands, dated evidence and open decisions.
> Historical results below are source records, not newly executed validation.

## Introduction

### Purpose

Complete the documented S1-S4 and M1-M3 synchronization cases in AFIR resource modeling and AscendC lowering, including load-to-store queue binding.

### Scope

This page summarizes the linked Autofuse MLIR record. It introduces no new runtime
behavior, API, supported platform or implementation decision.

## General Overview

### Software and Product Environment

The subject belongs to the Autofuse MLIR compilation and validation workflow.
CANN, AscendC, ATT and LLVM/MLIR requirements are those documented in the source;
this summary does not change their versions or supported configurations.

### Functionality, Constraints, Assumptions and Dependencies

A direct load-to-store path needs the appropriate dual-ended queue semantics. Preserve producer/consumer ordering and resource lifetimes; the source separates lit completion from the detailed UT and E2E validation record.

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
