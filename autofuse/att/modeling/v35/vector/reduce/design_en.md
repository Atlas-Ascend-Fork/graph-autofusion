# Introduction

## Purpose

Align v35 ATT normal aligned RA Reduce cost dispatch with production `ReduceRAImpl` when the inner width is symbolic.

## Scope

Reuse existing concat, less-than-VL and over-VL formulas and coefficients. Device implementation, templates, VF accounting and shared head cost are unchanged.

# Overview

## Software overview

### Project

A static input can still have a symbolic tile width. Previously such a width always used the over-VL cost, even if the executed API selected concat.

### Product environment

Autofuse v35 ATT and existing CANN AscendC Reduce APIs; no public API, ABI or package changes.

## Functions

Dispatch by resolved inner width A, matching constant-width behavior.

## Constraints

No network/case/device-core-count special cases, additional searches or cross-pipe compensation.

## Assumptions and dependencies

Production ReduceRAImpl uses REGULAR_FOLD_NUM=2 and the dtype VL. Model boundaries and tests must follow that implementation.

# Requirements and design

## Overview

BuildRaNormalAlignedCost reuses SelectByLessEqual and three existing formulas.

## Functional requirements

### Requirement 1: symbolic dispatch

1. Introduction: concat for A≤VL/2 or A>65535, less-than-VL for VL/2<A≤VL, otherwise over-VL.
2. Input: existing positive RA dimensions, dtype and reuse flag.
3. Processing: construct three conditions, propagate existing status failures.
4. Output: AIV_VEC estimate and ternary expressions; constant input behavior is retained.

## Nonfunctional requirements

### Maintainability

No new cost vocabulary or template names; production APIs and strategies are unchanged.

### Testability

[Cases](cases.csv) check F32/F16/BF16, reuse and threshold neighbors, including symbolic R and A.

### Portability

Only v35; existing C++ and expression interfaces are reused.

### Reliability

No new resources or shared mutable state; construction is bounded.

### Cross-feature effects

SuperKernel Python/C++/AOT, graph optimization, runtime, bindings and packaging are unchanged. Host formula generation, model UT and default ordering are affected. Device implementation, ABI and logging are unchanged.

## Performance

### Compilation

Three conditions per symbolic RA, no extra solves or traversal. Native Host timing and regression identity evidence are in the [validation report](../../mte2/load/results/dense-bulk-v1/report_en.md).

### Execution

No added device instructions. The chain selects PGO cases1/1/4 with the accompanying independently calibrated ordinary Load model.

### Memory and artifacts

Only bounded Host expressions; no new device layout, cache or logs.

## Interfaces

### Changes

No public changes; BuildRaNormalAlignedCost keeps its signature and status propagation.

### Interface checks

| Check | Affected | Conclusion |
| --- | --- | --- |
| Interface review | No | Existing interface |
| Documentation | Yes | This design and cases |
| Behavior | Yes | Corrected symbolic costs |
| ABI/API | No | Unchanged |
| Error contract | No | Existing statuses |
| Tests | Yes | Numeric branch-equivalence UT |

## Software design

### Structures

Existing PerfOutputInfo.ternary_ops owns the expressions.

### Algorithm

A>65535 selects concat; otherwise A≤VL/2 selects concat, then A≤VL selects less-than-VL, else over-VL. See the [flow diagram](design.md#关键技术算法).

### Flow

Build costs, dispatch by A, solve and select. Constants and unrelated API paths retain their behavior.

### Submodules

reduce_api_perf_v2.cpp dispatch and test_reduce_min_max_api_perf_v2.cpp coverage. BF16 assertions compare actual numeric results with FP16 because valid concat folds can simplify to 1/64.

### Errors

#### System errors

No new allocations, files or runtime calls; existing propagation remains.

#### Interface errors

No new error codes or input contracts.

## Safety

Encoding redlines checked: no secrets, external indices, pointer offsets, divisions, new lifecycle, graph rewrite or ABI changes.

## Compatibility

Script arguments, configuration, installation and device APIs are unchanged.

## Tests

### Boundaries

Upper-level ReducePerf inputs; numeric evaluation of symbolic/constant paths across dtype and reuse boundaries.

### Cases

| Category | Coverage | Type |
| --- | --- | --- |
| Functional | Symbolic A vs constant A | UT |
| Functional | Symbolic R/A dispatch | UT |
| Boundary | VL/2, VL, 65536; reuse and dtype | UT |
| Performance | Default vs frozen PGO independent replay | Device |
| Cross-feature | 17/48 complete identity | Incremental regression |

### Commands

Use the bounded build (-j8 maximum) and gtest filter in the [Chinese design](design.md#测试命令). Related tests: 195 passed; the new branch test fails on the original implementation.

## Acceptance

Branch-equivalence UT, precision and fast-template default selection pass; changed regression execution identities require device runs. No claim that every absolute Reduce cost is calibrated, and no VF coefficients are fitted from aggregate kernel data.
