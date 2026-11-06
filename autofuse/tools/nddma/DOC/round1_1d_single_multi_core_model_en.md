<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# NDDMA2 Round1: contiguous 1D multicore modeling

[简体中文](round1_1d_single_multi_core_model.md)

## 1. Goal

Round1 reuses the A+C dataset from HW_GE_ATT Round2 to build a piecewise multicore baseline for contiguous 1D NDDMA.

## 2. Entry points

Run from `autofuse/tools/nddma/`. Scripts are under `Modeling/Data_collection/round1/scripts/`:
`e2e.py`, `collect.py`, `fit.py`, and `draw.py`. Outputs default to `Modeling/Ana/round1/`.
Before collection, configure `NPU_ARCH` for the actual device or pass `--npu-arch` to the collection/e2e entry point.
See the [tool guide](../README_en.md) for environment setup and commands.

## 3. Scope

Each configuration uses:

```text
dim = 1
input_stride = 1
output_stride = 1
GM = contiguous
UB = contiguous
MTE2 only; no GM store
src_offset_elem = 0
dst_offset_elem = 0
```

Dtypes are `int8_t`, `int16_t`, `int32_t`, and `int64_t`; `block_dim` is the core count.

## 4. Dataset

The A+C fit grid matches `HW_GE_ATT/NDDMA/Modeling/Data_collection/round2/scripts/plan_common.py`.

| Group | Source of bytes_per_core | block_dim | Purpose |
| --- | --- | --- | --- |
| A | STAGE1_X_BY_DTYPE | 1,2,4,8,16,32,56 | Representative core counts |
| C | C_X_BY_DTYPE | 1..56 | All core counts |

There are 483 A configurations and 1904 C configurations, totaling 2387.
`bytes_per_core` is the transfer size handled by one core;
`logical_total_bytes = bytes_per_core * block_dim`.
The factor filename remains `round1_1d_single_core_factor.csv` for workflow compatibility,
although it contains the multicore A+C grid.

## 5. Formula and fitting

```text
block_dim <= 2: cycles = bytes_per_core / T_1(dtype) + H_1(dtype)
block_dim > 2:  cycles = bytes_per_core / T_2(dtype) + H_2(dtype)
```

`cycles` measures MTE2 cycles per NDDMA call. T is in bytes/cycle; H is in cycles.
For each dtype, fitting follows the HW_GE_ATT Round2 stages:

1. Select contiguous 1D A/C samples with `fit_role=fit`.
2. Average cycles for identical dtype, core count, and bytes_per_core, including overlapping A/C points and repeated measurements.
3. Independently fit `cycles = beta*B + H` by ordinary least squares for each core count, with `T=1/beta`.
4. Average T and H with equal weight across core counts within each branch (<=2 and >2).

The JSON `per_core_fits` field records individual fits. Aggregation reduces the point count below the 2387 raw configurations.
Grid parity does not imply identical measurements across runs.

The checked-in `Modeling/Ana/round*/*.json` files are historical snapshots, not newly fitted results.
Raw measurement CSVs are not included; old coefficients cannot reconstruct a new fit.
After collecting measurements, refit Round1, then Round2 and the dependent rounds.
The output JSONs already use the paths consumed by ad-hoc predictors; no copy into DOC is required.
