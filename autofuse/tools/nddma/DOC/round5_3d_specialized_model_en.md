<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Round5: 3D specialized model

[简体中文](round5_3d_specialized_model.md)

Round5 uses the same multidimensional formula as Round3 but refits corrections using only 3D measurements.
Copy the Round2 baseline unchanged; do not fit d_0 or use the 2D N_G2 specialization.
Round3/4 and their predictors retain their respective parameter sets.

## Formula and parameters

API dimensions and strides are outermost-first. Reverse them into innermost-first
ls_j, is_j and os_j when evaluating per-axis terms.

```text
B = product(output_dims) * dtype_size               # per-core bytes; no division by core count
N_base = B/T_1 + H_1, k<=2; B/T_2 + H_2, k>2       # fixed Round2 baseline
B_j = B / product(ls_t for t<j)
is_hat_0 = is_0; os_hat_0 = os_0
is_hat_j = abs(is_j - sum(ls_t*is_t for t<j)) + 1   # j>0
os_hat_j = abs(os_j - sum(ls_t*os_t for t<j)) + 1
s_j = min(is_hat_j * dtype_size, 128)
g_j = min(1, max(0, os_hat_j - 1))
N_G(j) = (a1 + a2*B_j)*s_j
N_GU(j) = (b1 + b2*s_j + (b3 + b4*s_j)*B_j)*g_j
rho(j) = c1 + c2*s_j + g_j*(c3 + c4*s_j)
N_3 = N_base + sum(N_G(j) + N_GU(j)), k<=2
N_3 = N_base + sum((N_G(j) + N_GU(j))*rho(j)), k>2
```

Fit ten coefficients independently for each dtype: a1,a2,b1,b2,b3,b4,c1,c2,c3,c4.

1. With k<=2 3D data, subtract the fixed baseline and jointly fit the six axis-summed
   columns `[s_j,s_j*B_j,g_j,g_j*s_j,g_j*B_j,g_j*s_j*B_j]` to obtain a/b.
2. With k>2 data, freeze a/b, define q_j=N_G(j)+N_GU(j), and fit the four axis-summed
   columns `[q_j,q_j*s_j,q_j*g_j,q_j*g_j*s_j]` to obtain c.

Both stages scale columns numerically, solve with `numpy.linalg.lstsq(rcond=1e-12)`,
and restore coefficients. There is no ridge penalty, clipping or extra intercept.
This is staged OLS, not joint nonlinear optimization of ten parameters.
All four dtypes need both core-count branches, with design ranks 6 and 4 respectively;
otherwise fitting fails rather than retaining old corrections. Single-core HW Round5 J / Round6 data alone cannot fit rho.

## Dataset

Reuse V2 Round3 3D configurations: 18743 from HW Round5 J and 6780 from HW Round6,
retaining layouts, strides and capacity filtering. They include 3D transpose,
the original `continuous` layout (including its original strides), and GM gaps.

Expand each configuration to core counts `(1,2,4,8,16,32,56)`: 178661 configurations,
with int8_t/int16_t/int32_t/int64_t, repeat=200 and three executions by default.
bytes_per_core and total_bytes describe one core; logical_total_bytes=bytes_per_core*block_dim.
Limits are 4 MiB GM and 256 KiB UB with 64 guard elements per core.

Only dim=3 rows enter fitting. Aggregate by dtype, core count, shape and both stride vectors.
Normalize positive cycles per execution and take the median, also merging configurations repeated across source grids.
actual_y is normalized per call; other supported cycle columns are divided by repeat.
All input 3D rows are training samples, including rows reused from Round3 validation.
Reported errors are training errors, not independent validation results.

## Running

Run from `autofuse/tools/nddma/` with NumPy installed. Collection requires Ascend/CANN and msprof.
Configure NPU_ARCH for the actual platform, or pass `--npu-arch` to e2e/collect.

```bash
python3 Modeling/Data_collection/round5/scripts/e2e.py all --msprof-bin "$(command -v msprof)"
```

Alternatively, execute each stage separately:

```bash
python3 Modeling/Data_collection/round5/scripts/e2e.py collect --msprof-bin "$(command -v msprof)"
python3 Modeling/Data_collection/round5/scripts/e2e.py fit
python3 Modeling/Data_collection/round5/scripts/e2e.py draw
```

Use existing measurements with repeatable --measurement-csv options:

```bash
python3 Modeling/Data_collection/round5/scripts/e2e.py fit \
  --round2-model Modeling/Ana/round2/round2_1d_noncontiguous_model.json \
  --measurement-csv /path/to/3d_single_dual_core.csv \
  --measurement-csv /path/to/3d_multicore.csv
```

e2e supports --run-output-dir, --data-dir, --model-output-dir, --factor-csv and
--dry-run (print commands without running subprocesses). Pass additional collection options directly to collect.py.

Default outputs under `Modeling/Ana/round5/`:

- round5_3d_factor.csv: collection configurations.
- collection/: build, execution, profiling and measurements.
- round5_3d_specialized_model.json: fixed baseline, fitted corrections, sample counts and errors.
- round5_3d_specialized_predictions.csv: per-configuration predictions and inherited Round2 comparison, including core counts.
- figures/: relative-error plots for fitted and inherited parameters by dtype and core count.

Predict with the newly fitted model:

```bash
python3 Modeling/Data_collection/round5/scripts/predict.py \
  4x8x16 1x4x64 128x16x1 8 int32_t \
  --model Modeling/Ana/round5/round5_3d_specialized_model.json
```

The JSON must be fitted from real measurements; no fabricated 3D coefficients are included.
