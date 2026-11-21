# Introduction

## Purpose

Record the v35 estimate additions for Load, Store, NDDMA, Transpose, and VF, and provide reviewable descriptor conditions, coefficients, and collection cases.

## Scope

The changes affect Autofuse ATT performance expressions and default tiling ordering. This directory records independent data and applicability for each model.

# Overall Overview

## Software Overview

### Project Introduction

ATT computes pipeline costs from the physical codegen transfer and instruction descriptors, and compares legal tilings with the model objective.

### Product Environment

Calibration uses existing Ascend 950/v35 collectors and the CANN Toolkit. Hardware conditions are expressed by the existing v35 model registration and platform parameters.

## Software Functions

Apply the corresponding calibration on descriptors covered by measurements. Symbolic conditions are evaluated when tiling is instantiated; descriptors outside the fitted domains keep the original expressions.

## Design Constraints

PMU cycles, task cycles, and device wall time are recorded separately. Training, holdout, known candidates, and out-of-domain controls are marked independently. Cases, point-level results, and external evidence indexes are archived in the repository.

## Assumptions and Dependencies

The calibration assumes the generated API, address progression, synchronization cadence, and cache policy. Data under other cache policies or platforms is validated separately.

# Requirements Analysis and Design

## Overall Description

Codegen descriptors enter the ATT model factory. The factory generates estimates and optional guards. Host tiling evaluates them with the real shape, tile, and active core count.

## Functional Requirements

### Transfer Estimates

Inputs include dtype, physical axes, GM/UB strides, effective bytes, and active core count. Load/Store reuse the bandwidth model, while NDDMA selects either the bandwidth or residual branch from the normalized descriptor. The output is an MTE2/MTE3 cycle expression.

### Vector Estimates and Shared Cost

Transpose computes VEC cost from the actual Gather bank period and masked lane count. VF is modeled from instruction dependencies and issue cost, and the shared startup slope enters the v35 objective.

## Non-Functional Requirements

### Maintainability

Descriptor conditions and formulas are stored in each model directory. Coefficients are traceable to the frozen version of each experiment.

### Testability

Case lists can be replayed through the corresponding collection entry. UT verifies product estimates, dynamic guards, and out-of-domain fallback.

### Portability

The v35 registry carries these estimates. Bit width, alignment, and core count are read from the model and platform configuration.

### Reliability

Existing descriptor validation runs before modeling. The original formula is kept when the applicable domain cannot be proven.

### Cross-Feature Impact

| Scenario | Conclusion |
| --- | --- |
| SuperKernel Python, C++, and AOT | Not involved; its artifacts are separate from estimate registration |
| Autofuse graph optimization | This archive covers estimates; graph-pass changes have separate functional tests |
| Autofuse Codegen/Backend | Host estimates and default selection may change; verify the tiling actually emitted |
| AscendC API, Runtime, Python/C++ bindings | Reuse existing device descriptors and external entries |
| Build and packaging | Use the existing build and delivery entries |
| Test and coverage | Model UT, descriptor experiments, and device regression with PGO disabled |
| Performance and logging | Focus on Host expressions, cold compile, and device time; model-branch information is logged at debug level |
| Compatibility | Out-of-domain fallback and existing rules for V1 platforms must be preserved |

## Performance

### Compile Time

Symbolic domain checks add estimate expressions. Host cold solve, cache hit, and cold compile are compared with Base separately. Repeated solves from the default search and model fitting error are measured separately.

### Execution Performance

New estimates may change tiling. PGO must be disabled, and the actual emitted kernels, accuracy, and device time must be checked. Component cycles are only used to validate the corresponding model.

### Memory and Artifact Size

The added data is lightweight CSV, reports, and evidence indexes. Raw profiles, plog, binaries, and large tensors stay in the experiment environment.

## Interface Design

### New/Modified Interfaces

The model factory and tiling-expression interfaces are reused. Estimate inputs come from existing dtype, shape, stride, and core-count fields. The NDDMA debug model name can represent a conditional expression.

### Interface Checks

| Check | Description |
| --- | --- |
| External interfaces, ABI | Model archives add no public entry |
| Parameter domain and errors | Out-of-domain inputs use the original formula; invalid descriptors keep the original error handling |
| Interface tests | Model-factory UT covers static and symbolic conditions |

## Software Design

### Key Data Structures

Existing DMA descriptors and `PerfOutputInfo` hold estimates and condition expressions. Archives align `cases.csv` and `samples.csv` by case ID.

### Key Techniques/Algorithms

Reuse existing model forms. New parameters come from matching independent collections. The raw descriptor distinguishes contiguous, cross-pitch, LoopMode, NG2, and sparse multi-axis layouts.

### Flow Design

Check the actual API and measurement conditions, freeze fitting data, verify independent holdout, generate product estimates, validate candidate ordering, and validate defaults with PGO disabled before regression.

### Submodule Changes

- [Load](mte2/load/design_en.md)
- [NDDMA](mte2/nddma/design_en.md)
- [Store](mte3/store/design_en.md)
- [Transpose](vector/transpose/design_en.md)
- [VF](vector/vf/design_en.md) and [shared startup cost](common/op_head_cost/design_en.md)

### Error Handling

#### System Errors

Results with missing collection, accuracy, profile, or evidence summaries keep their original state and are diagnosed and re-measured under the original protocol.

#### Interface Errors

Invalid descriptors use the original model-entry fallback or error codes. Debug logs for symbolic models explicitly record possible paths.

## Safety Check

Reuse the original shape/stride validation, symbolic length computation, graph execution, and resource lifetime. New estimates do not call device APIs. Archives contain descriptors and summaries.

## Compatibility Check

The case index is placed beside the old Transpose archive. Original reports and old collections remain readable. Platforms other than v35 keep the existing scheduling configuration.

## Test Design

### Test Boundaries

Validate fixed descriptors, dynamic conditions, and device defaults separately. Cold-read or address-reuse controls keep their own cache policies.

### Test Case Design

| Category | Coverage |
| --- | --- |
| UT | Domain boundaries, other dtypes, symbolic guards, model fallback |
| Collection | Independent training, independent holdout, known candidates, out-of-domain controls |
| Device regression | Actual emission with PGO disabled, accuracy, business gate, and per-case fallback |
| Engineering performance | Uncached Host solve, cache hit, and cold compile are accepted separately |

### Test Commands

Use `sh build.sh -u --module=autofuse_framework -j 8` for build verification. Each model design page lists targeted UT, and collection entries are referenced by the corresponding result manifest.

## Acceptance Criteria

Check the descriptor and measurement semantics item by item, and validate fitting error and candidate ordering. Product defaults must meet accuracy, performance, regression, and Host engineering gates. Archived historical component data is interpreted under its original measurement semantics.
