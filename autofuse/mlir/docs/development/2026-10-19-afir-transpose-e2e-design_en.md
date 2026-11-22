# `afir.transpose` End-to-End Landing and Generalization Design

> English companion summary, not a full translation. The [Chinese source](2026-10-19-afir-transpose-e2e-design.md)
> retains the detailed design, tables, commands, dated evidence and open decisions.
> Historical results below are source records, not newly executed validation.

## Introduction

### Purpose

Describe the end-to-end design of transpose-family ops (`afir.transpose` / `afir.pad`)
on the MLIR codegen path — op bodies (dialect definition, bridge integration, emitters),
whole-graph enablement of strided data movement, scheduling/ATT-side support, and the
test and acceptance criteria — plus the follow-up generalization roadmap.

### Scope

The 2026-11-02 acceptance scope correction limits this round to legacy Inductor
migration; TensorFlow is out of the delivery scope. Seven migrated standalone
behaviors and one native integration test passed; full UT 14/14 and lit 284/284;
on 910B3 the transpose→Add graph passed 4/4 numerical checks across two shapes and
both legacy/MLIR routes.

## General Overview

### Software and Product Environment

The subject belongs to the Autofuse MLIR compilation and validation workflow.
CANN, AscendC, ATT and LLVM/MLIR requirements are those documented in the source;
this summary does not change their versions or supported configurations.

### Functionality, Constraints, Assumptions and Dependencies

The 2026-11-03 reg-simulation recheck reports 14 CPU figures passing element-wise,
with mem/reg variants matched by shape across seven graphs. Compact transfers for
the reg tier reuse the existing `datacopy_reg_base.h` Compact wrapper; suffix
strides and axis merging follow the actual tail-block size, without solver changes
or forced tile alignment. Two new `dynamic=True` NPU checklist entries passed
legacy/MLIR × two shapes, eight numerical verifications in total. Python ST and
native Inductor checks (stage contracts, artifact determinism, illegal-schedule
rejection) were added. TF fixed-source calls to `Generate`, PGO and the v35
IndirectLoad E2E are not counted as mandatory native-Inductor migration items.

## Requirements and Design

### Overall Approach and Functional Requirements

Follow the source for the op-本体 sections (ODS and verifier, bridge integration,
official header declaration, two AscendC wrapper ops with hand-written emitters,
reg-tier implementation `transpose.ub_ub_reg_base`, mem-tier implementation
`transpose.ub_ub_extend`, and the Nddma descriptor baseline recheck dated
2026-10-31), plus the whole-graph enablement sections (segment geometry and
unified emission for strided load/store, `afir.pad` dialect definition and bridge
integration). This summary introduces no alternative behavior.

### Nonfunctional Requirements and Cross-Feature Impact

Documentation-only companion. It changes no SuperKernel Python or C++/AOT
interfaces, no packaging, and no runtime lifecycle. The source governs any
cross-feature impact conclusions.

## Testing and Evidence

Testing and acceptance follow the README MLIR suite and the NPU checklist recorded
in the source: UT, ST, lit, CPU E2E with simulator execution, and 910B3 numerical
validation. Old-baseline regression numbers must not be reused as evidence for the
current patch; the review document's latest records govern. P1–P4 generalization
routes have tests; P5 still awaits on-device verification on the current version.

## Performance and Memory

The source records the ATT/perf-model alignment expectations and tiling
implications for transpose-family ops; no additional performance claim is made
by this summary.

## Open Decisions

Line numbers in the source are writing-time snapshots; symbol names govern.
Deferred items, upstream design links, and the design-template compliance note
remain as stated in the Chinese source.
