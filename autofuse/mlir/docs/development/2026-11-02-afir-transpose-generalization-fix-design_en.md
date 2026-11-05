# AFIR Transpose Generalization Combination Qualification and Candidate Fix Design

> English companion summary, not a full translation. The [Chinese source](2026-11-02-afir-transpose-generalization-fix-design.md)
> retains the detailed design, tables, commands, dated evidence and open decisions.
> Historical results below are source records, not newly executed validation.

## Introduction

### Purpose

Centrally record the known evidence, pending hypotheses, fix boundaries and
acceptance methods for the five generalization problem classes, for use by
implementation and review. The source proposes only preconditioned candidate
solutions; exploration failures are not directly classified as MLIR defects,
and no claim is made that the generalization problems are already fixed.

### Scope

Covers the MLIR layout investigation and fix design for combinations of
Cast/chain/broadcast/Sum/GEWhere with Transpose. The legacy migration targets
Inductor only; TensorFlow is out of scope. Framework-agnostic optimizer graphs
borrowed for supplementary generalization exploration do not change the original
migration acceptance. Excludes legacy ATT/v35 common implementations,
SuperKernel, and new scheduling strategies.

## General Overview

### Software and Product Environment

The AFIR/AscendC lowering in graph-autofusion reuses existing scheduling
candidates and resource information. A single-graph generation or basic
transpose numerical pass does not prove layout consistency when transpose
combines with data movement, broadcast, predicates and reductions. The original
requirements design and test acceptance are maintained separately; follow-up gap
fixes are maintained in the source document. Environment: CANN 9.2.0,
same-source production codegen/pyautofuse overlay, CPU E2E with the strict MLIR
entry. Existing 910B3 numerical evidence does not cover the failing combinations
here; a3 and performance are not verified.

### Functionality, Constraints, Assumptions and Dependencies

Status as of 2026-11-02: the five exploration combinations have not passed the
legacy CPU numerical gate; the fixes in the source are candidates without
implementation preconditions yet. The verification order is graph/oracle audit →
legacy Inductor generate/compile/numerical pass → same-graph same-platform MLIR
comparison → confirm differences before fixing. Legacy optimizer success or
legacy codegen success must not substitute for end-to-end passing; no MLIR fix
was implemented in this round.

## Requirements and Design

### Overall Approach and Functional Requirements

Follow the source for the root-cause analysis with evidence-strength grading
and the per-combination functional requirements (Cast, chain, broadcast, Sum,
GEWhere interactions with Transpose). This summary introduces no alternative
behavior or additional fix commitments.

### Nonfunctional Requirements and Cross-Feature Impact

Documentation-only companion. It changes no SuperKernel Python or C++/AOT
interfaces, no packaging, and no runtime lifecycle.

## Testing and Evidence

Acceptance methods, per-combination fixed loops and evidence-strength notes
remain as recorded in the Chinese source.

## Performance and Memory

No performance or memory claims are made by this summary; the source governs.

## Open Decisions

Candidate-fix preconditions, the strict verification order, and the deferred
a3/performance validation remain as stated in the Chinese source.
