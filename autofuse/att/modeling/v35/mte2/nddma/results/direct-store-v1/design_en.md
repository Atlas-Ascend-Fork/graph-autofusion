# Introduction

## Purpose

Calibrate the effect of full Store writeback on effective MTE2 throughput in packed 2D NDDMA-to-Store pipelines. Isolated NDDMA API parameters remain unchanged.

## Scope

v35 ATT costs and collection entry points only. Codegen, device synchronization, kernels, search, configuration meanings and SuperKernel are unchanged.

# Overview

## Software overview

### Project

Isolated NDDMA slopes underestimate MTE2 costs in a full direct-store pipeline. Keeping APIs, descriptors, queues and fresh offsets fixed while changing writeback length reproduces the difference; see report_en.md.

### Product environment

CANN 9.3.0, dav-3510, measured on 56 available AIVs. The model reads candidate block_dim, not device total core count.

## Functions

Keep isolated API costs and add a bounded read/write-concurrency residual for verified direct-store paths. Two-inner-axis transpose index stores use their actual int16/int32 types in the existing instruction table.

## Constraints

Public API/ABI, package layout, runtime calls and device source remain unchanged. Build parallelism is at most eight; performance collection owns the device exclusively. No case, absolute-address or total-device-core special cases, and no cross-pipe compensation.

## Assumptions and dependencies

One output with a known allocated ID, double-buffered VECIN queue, and a unique Store consumer, matching the production direct queue path. Calibration covers one full-writeback consumer, buf_num=2 and a GM contiguous source no wider than one 32-byte DMA block. Multiple Stores, vector consumers and CV/UB fusion keep their existing costs. Active core counts above 56 are extrapolations, not measured device validation.

# Requirements and design

## Overview

Apply existing NDDMA cost first. Add a concurrency residual only when graph, queue, dtype and descriptor gates all match. Otherwise keep existing costs.

## Functional requirements

### Requirement 1: effective direct-store MTE2 cost

1. Introduction: distinguish isolated APIs from full read/write pipelines without enlarging all NDDMA coefficients.
2. Input: existing NodeInfo.node_ptr, validated descriptor, dtype and candidate block_dim. Missing structural evidence means no match.
3. Processing: add `k*max(B*C-U,0)`, where B is payload bytes per core/API and C is active candidate cores. Dimensions `[M,N]`, GM strides `[1,M]`, UB strides `[N,1]`, FP32, M=2–8, N=512–16384 divisible by eight, B=8192–131072. Static out-of-domain inputs do not emit the branch; symbolic inputs emit a guard. Training-only fit: k=0.0038940645164791633, U=4456448.
4. Output: effective AIV_MTE2 cost and NDDMA_ND_MULTICORE_DIRECT_STORE_V1 attribution; symbolic selection retains runtime-dispatch attribution.

### Requirement 2: index Store dtype

1. Introduction: GenTwoInnerDimTransposeIndex stores RangeType<T>, not uint8.
2. Input: original element byte size.
3. Processing: size≤2 uses int16; otherwise int32. Query existing kStore parameters for initialization and tail loops.
4. Output: corrected VEC cost for this index generator; existing semantic template classification is retained.

## Nonfunctional requirements

### Maintainability

Reuse existing NDDMA equations and guards; the residual models pipeline contention, not a refit of isolated APIs. Missing structural evidence disables it.

### Testability

Training, holdout and known roles are frozen in cases.csv. UT covers static/symbolic boundaries, dtype, layout and low-core fallback.

### Portability

Only v35; V1 does not call this model. Higher active core counts require device acceptance rather than Host-only equations.

### Reliability

No new device buffers, threads or resources. Existing TernaryOp and descriptor validation are reused.

### Cross-feature effects

| Feature | Conclusion |
| --- | --- |
| SuperKernel Python/C++/AOT | No interface or implementation change |
| Autofuse optimization | No graph or pass change |
| Codegen/Backend | ATT costs and selection can change; kernel APIs do not |
| AscendC/Runtime | Calls, cache hints, synchronization and lifetime are unchanged |
| Python/C++ bindings | Public interfaces unchanged |
| Build/package | No new targets, dependencies or installation paths |
| Tests | Targeted UT, independent collection, target/control regression |
| Performance/logging | One adjacency check and bounded expressions; no device logic |
| Compatibility | Existing CLI preserved; collector adds nddma_full_store |

## Performance

### Compilation

Check direct consumers once per NDDMA node. The model adds one linear hinge and at most one domain guard. Thirty paired rounds show +1.42% first Host tiling; repeat cache hits remain essentially unchanged (report_en.md).

### Execution

No added device instructions/runtime calls. Default timing is in the report; full regression was not completed at this design stage.

### Memory/artifacts

No new persistent cache or shared state. Keep compact evidence rather than element-wise tensors.

## Interfaces

### Changes

Internal EvaluateNddmaModel adds a default-false direct_store parameter, populated from existing graph relations. Existing callers retain isolated costs. Public exported interfaces are unchanged.

### Interface checks

| Check | Affected | Conclusion |
| --- | --- | --- |
| Review | Internal | ATT helper only |
| Documentation | Yes | Design and tool README |
| Behavior | Yes | Structurally gated measured domain |
| Public ABI/API | No | Unchanged |
| Invalid inputs | Yes | Existing descriptor validation/fallback |
| Tests | Yes | NddmaModelV2 / UTestAscirPerfV2 |

## Software design

### Structures

Reuse NddmaDescriptorInfo, NddmaModelResult and NodeInfo; no shared state.

### Algorithm

For K=1/2/4/10, fit per-core MTE2 median cycles against K. Fit a positive residual coefficient and one threshold on training rows only, then freeze and validate holdout/known points.

### Flow

Verify collection, compute slopes, freeze fit, validate holdouts, build private products, verify PGO-off defaults, run incremental regression. Preserve compact failures; never overwrite samples or change isolated-API coefficients to compensate.

### Submodules

nddma_model: separate pipeline residual; ascir_api_perf_v2: structural evidence; ascendc_regbase_perf: actual index Store types; tools/modeling/v35/dma: existing full-writeback collector mode.

### Errors

#### System errors

Keep build/collection exit codes. Fit output uses exclusive creation rather than overwriting historical evidence.

#### Interface errors

Keep descriptor fallback. Missing graph/queue evidence cannot enable the residual.

## Safety

Encoding redlines checked: no credentials, accounts, chipset special cases, new indexing/allocation/runtime lifetime or graph rewrites. Existing descriptor length checks remain.

## Compatibility

Accuracy-level meanings, defaults, Codegen/device ABI, Python interfaces and installation paths are unchanged.

## Tests

### Boundaries

API/structure gates, constant and symbolic formulas, independent collection and real default/PGO execution. Forcing a fast template does not verify default selection.

### Cases

| Category | Coverage | Type |
| --- | --- | --- |
| Functional | Direct cost, index type | UT |
| Invalid/domain | Padding, dtype, low core, out-of-domain | UT |
| Performance | Train/holdout/known, default/PGO; fresh GM and full Store | Benchmark |
| Compatibility | Static/symbolic and controls; execution identity | UT/regression |
| Cross-feature | V1 and CV/UB fusion retain gates | UT/regression |

### Commands

See the DMA README for build, collection and refit commands. Build att_ut with bounded parallelism, then run `--gtest_filter='NddmaModelV2.*:UTestAscirPerfV2.*Transpose*:UTestAscirPerfV2.*Nddma*'`.

## Acceptance

Separate training/holdout/known sets; maximum holdout relative error≤30%; PGO-off default latency≤1.05 times independently profiled same-package PGO. Precision, boundary UT, affected regression and Host overhead must pass before claiming completion. See report_en.md for actual scope and limits.
