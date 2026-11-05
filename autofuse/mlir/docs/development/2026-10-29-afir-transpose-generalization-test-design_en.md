# Transpose Generalization Test Design (P1–P5, Legacy-Scenario Alignment)

> English companion summary, not a full translation. The [Chinese source](2026-10-29-afir-transpose-generalization-test-design.md)
> retains the detailed design, tables, commands, dated evidence and open decisions.
> Historical results below are source records, not newly executed validation.

## Introduction

### Purpose

Provide executable case design and acceptance criteria for the generalization
routes P1–P5: eliminating route-less uncompensated shapes, elementwise chain
fusion, multiple transposes, interaction shapes, and on-device validation. The
core principle is **aligning cases with legacy test scenarios** — shapes where
legacy has cases are anchored directly to its shapes and assertion levels;
shapes without legacy cases are anchored by comparing against the same-graph
legacy artifacts.

### Scope

Includes: P1–P5 CPU E2E cases (mainly mem tier), lit synthetic-matrix additions,
reg-tier alignment criteria, a2/a3 on-device scripts with verdicts, fixed loops
per graph and acceptance standards. Excludes: legacy ATT/v35 common-generator
modifications; MLIR adaptation boundary defects exposed by validation were fixed
alongside this round. The nddma broadcast folding case family belongs to the
baseline broadcast feature; reg-tier E2E numerical acceptance is closed per the
upstream design (legacy has no numerical anchor at that tier).

## General Overview

### Software and Product Environment

Transpose/Pad migration (upstream design parts one and two) is complete:
`afir.transpose`/`afir.pad` dialect and bridge, strided load/store segment
geometry, `TransposeExtend`/`ConfusionTranspose` lowering families,
RemovePad/Pad, and allocator fixes. Validation uses the MLIR E2E CPU framework
(`test/e2e/cpu` with the tikicpulib simulator), lit (`test/lit`), and the E2E NPU
scripts (`test/e2e/npu`, npussh plus torch comparison).

### Functionality, Constraints, Assumptions and Dependencies

This round delivers only the legacy Inductor migration per user decision;
TensorFlow is excluded. The migration checklist has passed verification; the
P1–P5 historical extensions are not the mandatory delivery list of this round.
The 2026-10-31 migration inventory and execution revision in the source governs
the latest state.

## Requirements and Design

### Overall Approach and Functional Requirements

Follow the source for the five functional requirements (P1 route-less
uncompensated shapes scenario 2, P2 elementwise chain fusion, P3 multiple
transpose cascades, P4 interaction shapes, P5 on-device) with their fixed loops,
assertion levels and acceptance criteria. This summary introduces no alternative
behavior.

### Nonfunctional Requirements and Cross-Feature Impact

Documentation-only companion. It changes no SuperKernel Python or C++/AOT
interfaces, no packaging, and no runtime lifecycle.

## Testing and Evidence

Legacy line numbers in the source are 2026-10-29 snapshots; symbol names
govern. The Inductor migration acceptance section in the source holds the
latest results; older P1–P5 records are historical context.

## Performance and Memory

Performance-related expectations remain governed by the ATT/perf-model
statements in the upstream design; this summary adds no claims.

## Open Decisions

Deferred items and anchor-selection methodology notes remain as stated in the
Chinese source.
