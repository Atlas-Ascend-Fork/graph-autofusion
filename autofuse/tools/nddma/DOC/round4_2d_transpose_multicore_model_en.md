<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Round4: 2D UB-contiguous N_G2 specialization aligned with HW Round5

[简体中文](round4_2d_transpose_multicore_model.md)

## Training set

Use the single-core 2D D/F/G data used by HW Round5 fitting, rather than G alone.

| Group | Contents | Configurations |
| --- | --- | ---: |
| D | is2=1, 2D transpose with ordinary input strides | 2586 |
| F | is2=1, additional high-stride configurations | 1930 |
| G | Outer strides is2=2/4/6/8/16 | 8858 |
| Total | Single-core training configurations | 13374 |

D/F require is1>=M; G requires is2*M<=is1. GM/UB safety bounds match HW.
Aggregate repeats by config_id/token using the median. Other groups and multicore samples
are excluded from parameter estimation. Missing D/F/G groups for any dtype cause an error.

HW A/E measured 1D anchors support paired diagnostics; the final N_G2 regression denominator
uses frozen 1D predictions. This version inherits Round2 directly without collecting those anchors again.
HW I/K/L/M extrapolation validation is outside this training set and fitting process.

## Formula and fitting

```text
output_dims=[M,N]; input_stride=[is2,is1]; output_stride=[N,1]
B=M*N*dtype_size         # bytes per core
B1=N*dtype_size
N_base = B/T_1+H_1 (k<=2), B/T_2+H_2 (k>2)
N_G1 = N_1_prime(B1,is1,1,k)  # use 1D rho for k>2
observed_N_G2 = (actual_cycles-N_base)/N_G1
N_G2 = g10*is2 + g11_M*M*is2 + g00 + g01_M*M
N_2 = N_base + N_G1*N_G2
```

Freeze Round2 coefficients. For each dtype, regress observed_N_G2 on `[is2,M*is2,1,M]`
using `numpy.linalg.lstsq(..., rcond=None)`. This is one joint OLS fit, not a two-stage fit,
matching current HW code: no regularization, column scaling, prediction clipping or d_0.
Fit N_G2 with single-core data only. Multicore predictions reuse 1D rho without extra 2D multicore coefficients.

## Running

Run from `autofuse/tools/nddma/`; configure NPU_ARCH before collection as described in the [tool guide](../README_en.md).

```bash
python3 Modeling/Data_collection/round4/scripts/e2e.py all --msprof-bin "$(command -v msprof)"
python3 Modeling/Data_collection/round4/scripts/e2e.py fit --measurement-csv /path/to/measurements.csv
```

Inherit `Modeling/Ana/round2/round2_1d_noncontiguous_model.json` and write to `Modeling/Ana/round4`.
NumPy is required. The model records N_G2, frozen 1D parameters and fit_scope.
Checked-in JSONs under Modeling/Ana are historical snapshots. After updating Round1/2,
refit from new D/F/G measurements and validate the results. Predictors read Ana directly;
no historical measurement coefficients are fabricated.
