# Introduction

## Purpose

Archive the measurement boundary and offline recomputation method of the existing v35 ATT `GetOpHeadCost` coefficient.

## Scope

Applies to v35 candidates that reference this shared term. The head costs of individual APIs such as VF, MTE2, and MTE3 still follow their own formulas.

# Overall Overview

## Software Overview

Two independent AscendC benches measure task time and AIV PMU cycles. The original experiment used Ascend 950 / CANN 9.3.0.

## Software Functions

The tools check launch and profiling, extract the median per group, and then fit the core-count slope and compute the holdout error.

## Design Constraints

Training, holdout, and control points are partitioned in advance. The measured core counts are experiment sample points; production still uses the core count computed from the candidate.

## Assumptions and Dependencies

Collection requires a matching CANN, msprof, and at least 50 available AIV cores. Offline recomputation requires Python 3.9+.

# Requirements Analysis and Design

## Overall Description

`PerfParamTableV2::GetOpHeadCost` returns the shared cost per active core. The existing objective adds
`22.059531283 x modeled core count` on top of the pipe costs. First establish the conversion between task time
and PMU, then fit the core-count slope with an independent empty kernel.

## Functional Requirements

| Experiment | Training points | Holdout and control | Processing |
| --- | --- | --- | --- |
| Scalar-loop conversion | 3, 24, 48 cores; 128, 512, 2048 loops | 256, 1024 loops as holdout; 0 loops as control | Fit the slope of task us and AIV PMU cycles/core versus loop count per round and core count, and take the median ratio |
| Empty-kernel core slope | 3, 6, 12, 24, 36, 48 cores | 5, 15, 18, 30, 42, 50 cores as holdout; a kernel writing one marker per core is control only | Multiply task us by the conversion factor and fit an ordinary least-squares model with intercept versus core count |

## Non-Functional Requirements

The sampling plan is fixed in the [case list](cases.csv). The [tool guide](../../../../tools/modeling/op_head_cost/README_en.md)
provides device collection and offline recomputation commands. Measurement files, run results, and tools are stored
in separate directories. Invalid output, count mismatch, missing samples, or an invalid fit terminates the run.

### Cross-Feature Impact

| Scenario | Impact of this archive |
| --- | --- |
| SuperKernel Python, C++ / AOT | Not involved |
| Autofuse graph optimization, Codegen / Backend | No change to graphs, generated code, or tiling behavior; only solver documentation changes |
| AscendC / Runtime | ACL and AscendC are used only by the independent bench |
| Python / C++ bindings | Not involved |
| Build and packaging | The benches have separate CMake and are not part of the product build |
| Test and coverage | Check the entry after moving the tools, and recompute from frozen measurements |
| Performance and logging | No new execution or log overhead in the product |
| Compatibility | Only the in-repo modeling tool entry path changes |

## Performance

This archive and the comments do not change product compile, tiling, or kernel execution performance, and add no
delivery package content or resident memory. A single empty-task calibration cannot prove that all v35 candidate
orderings improve.

## Interface Design

`collect.sh <build-dir> <new-run-dir>` writes collection results to the specified new directory.
`calibrate.py fit <samples.csv>` writes fitting results to stdout. Invalid input returns non-zero.
Only in-repo experiment entries change; there is no product API/ABI change.

## Software Design

The data flow is `msprof + launch receipt -> grouped-median CSV -> independent conversion -> empty-task slope and holdout error`.
Both groups run four rounds in the order forward, reverse, reverse, forward. Each group runs three warmups and then takes
the median of 31 formal samples. Check the kernel name, core count, order, single stream, and output/protection-region
receipts. Regression uses `Task Duration(us)`; ACL events are for diagnosis only. AIV PMU cycles are divided by the
block count according to the experiment convention. Candidate, default, and PGO measurements are not used for fitting.

Each CSV row stores the bench, round, core count, workload, training role, sample count, and both measurements.
The experiment intercept is only used to evaluate fitting error; the product keeps the existing constant ownership
and uses only the slope. The PMU and task-minus-PMU slopes are reported separately to explain the difference between
the two measurement semantics; they cannot be attributed to a specific exit instruction. Keep the artifacts after a
failed collection or audit and retry in a new directory.

## Safety Check

Both independent benches use fixed-size buffers, and they verify and release them after the task completes. The tools
do not overwrite an existing output directory. Full profiles are not committed; only grouped results and source
summaries are archived.

## Compatibility Check

Offline recomputation does not need an NPU. Re-collection needs a v35 device that supports the listed core counts
and counters. V1 does not use this coefficient.

## Test Design

Run the offline recomputation command from the tool guide and check it against the
[frozen measurements](results/v35/pr2921-d68/report_en.md).
`UTestAscirPerfV2.TestGetOpHeadCostValid` covers reading the product constant. Default and PGO ordering, accuracy,
and performance must be validated separately with real candidates.

## Acceptance Criteria

The training group reproduces the archived coefficient and holdout error, and the control group is kept
independently. Empty-task results are not used as a product regression conclusion.
