<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Round2: non-contiguous 1D modeling aligned with HW_GE_ATT Round4

[简体中文](round2_1d_noncontiguous_model.md)

Round2 inherits the Round1 contiguous baseline and fits only N_G, N_GU, and multicore rho.
It matches the final HW_GE_ATT Round4 1D model, excluding historical candidate-model and feature-search diagnostics.
Python and NumPy are required for least-squares fitting.

## Dataset

The default factor file is `Modeling/Ana/round2/round2_1d_noncontiguous_factor.csv`.
Safety filtering limits GM span to 4 MiB and UB span to 256 KiB, including a 64-element guard.
Dtypes are int8_t/int16_t/int32_t/int64_t. Kernel repeat defaults to 200 and execution repeats to 3.

| Group | Configurations | Purpose |
| --- | ---: | --- |
| A | 480 | Single-core GM gaps; low byte strides fit N_G |
| B | 350 | Single-core UB-gap validation, excluded from final N_GU estimation |
| C | 4075 | Joint gaps; os=2 subset fits N_GU |
| F | 440 | Multicore contiguous/output-gap reference validation |
| G | 840 | Multicore GM-gap calibration |
| H | 840 | Multicore joint-gap calibration |
| I | 120 | Output-gap validation at independent core counts |
| J | 1 | int32_t, 55 cores, 9656 elements/core, is=3/os=1 validation |
| Total | 7146 | Excludes repeated executions |

Single-core dimensions: `8,16,32,64,128,512,1024,2048,4196,10001,20001,30001,42302`.
Group A input strides: `2,4,8,16,32,64,128,256,512,1024,2048,4096`.
C additionally includes input stride 3. B/C output strides: `2,4,6,8,10,12,14,16,32`.

G/H core counts: `1,2,4,8,15,32,56`; bytes/core: `64,128,512,1024,4096,16384`;
input strides: `2,8,32,128,256`. G uses os=1 and H uses os=2.
F core counts: `1,2,4,8,16,32,56,6,24,40`; stride pairs: `(1,1),(1,2),(1,16)`.
I core counts: `6,24,40`; stride pairs: `(2,4),(32,16),(128,32)`.
F/I bytes/core: `64,512,4096,16384`.

## Formula

```text
B = bytes_per_core = logical_total_bytes / block_dim
s = min(input_stride*dtype_size, 128)
g = min(1, output_stride-1)
N_base = B/T_1+H_1 (k<=2), B/T_2+H_2 (k>2)
N_G = (a1+a2*B)*s
N_GU = ((b1+b2*s)+(b3+b4*s)*B)*g
rho = (c1+c2*s)+g*(c3+c4*s)
N_1 = N_base+N_G+N_GU (k<=2)
N_1 = N_base+rho*(N_G+N_GU) (k>2)
```

## Fitting

Aggregate repeated measurements by config_id/token using the median.
Divide raw cycles by kernel repeat; `actual_y` is already normalized.

1. Freeze the Round1 baseline; do not refit it with F.
2. N_G: use single-core A samples with B<50000 and input_stride*dtype_size<128.
   For each B, regress actual-base against byte stride through the origin to obtain g_prime;
   then fit `g_prime=a1+a2*B`.
3. N_GU: use single-core C, os=2, B<50000. Subtract Base and N_G.
   For each fixed input stride, take median residuals by B and fit `q0+q1*B`.
   Fit q0 and q1 against byte stride using only nodes below 128 bytes to obtain b1..b4.
   Higher-stride nodes retain first-stage fits for completeness checks but do not enter the second-stage estimation.
4. rho: freeze single-core coefficients and use G/H with k>2 and input_stride*dtype_size<=128.
   Regress actual-base on `[delta,s*delta,g*delta,g*s*delta]`, where delta=N_G+N_GU.
   The B<50000 single-core filter does not apply here.

Single-core fits use column scaling and `numpy.linalg.lstsq(rcond=1e-12)`;
rho uses an unscaled matrix and `rcond=None`. All safe configurations are evaluated,
including points outside the fitting domain as validation points.

## Running

Run from `autofuse/tools/nddma/`; configure NPU_ARCH before collection as described in the [tool guide](../README_en.md).

```bash
python3 Modeling/Data_collection/round2/scripts/e2e.py all --msprof-bin "$(command -v msprof)"
python3 Modeling/Data_collection/round2/scripts/e2e.py fit --measurement-csv /path/to/measurements.csv
```

The default input model is `Modeling/Ana/round1/round1_1d_single_multi_core_model.json`.
Models, per-point predictions and figures are written under `Modeling/Ana/round2`;
`fit_scope` records fitting conditions.

Checked-in JSONs under Modeling/Ana are historical snapshots. The old 7221-configuration dataset
lacks points in the current grid and is not a complete replacement for it.
Collect the aligned grid, refit Round1/2, and update dependent Round3/4 models.
Ad-hoc predictors read the resulting Ana paths directly.

## Reference validation

```bash
HW_NDDMA_MODELING=/path/to/HW_GE_ATT/NDDMA/Modeling \
  python3 -m unittest discover -s Modeling/tests -v
```

Tests compare complete collection configurations, including repeats, and N_G/N_GU/rho coefficients
using identical perturbed inputs. Additional perturbations of validation points verify that they do not affect fitted parameters.
