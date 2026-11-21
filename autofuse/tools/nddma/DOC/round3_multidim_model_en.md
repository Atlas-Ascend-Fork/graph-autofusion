<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Round3: unified multidimensional model without d_0

[简体中文](round3_multidim_model.md)

Inherit Round2 base, N_G, N_GU and rho without fitting new multidimensional parameters.
Match unscaled HW Round7 predictions and the unified predictor; do not fit or apply d_0.
HW Round7 results after d_0 fitting are not the parity target.

## Dataset

| Source | Contents | Safe configurations |
| --- | --- | ---: |
| HW Round4 2D D/E | Complete GM-padding and UB-gap grid | 8532 |
| HW Round5 J | 3D inherited-parameter configurations, also used in HW Round7 joint evaluation | 18743 |
| HW Round6 | Four basic layouts and 11 GM-stride nodes | 6780 |
| HW Round7 | Three layouts for 4D and 5D | 14313 |
| Total | Evaluation only; no new parameter estimation | 48368 |

Safety limits are 4 MiB GM span and 256 KiB UB span, including 64 guard elements.
Defaults are kernel repeat=200 and execution repeat=3. Distinct strides/sources retain distinct IDs.
The Round6 `continuous` layout preserves the reference input and output strides `(n*k,n,1)`.
This is not always conventionally contiguous, but the reference descriptor is intentionally retained.

## Calculation

Reverse API output_dims/input_stride/output_stride into innermost-first axis order.

```text
B = product(output_dims)*dtype_size  # per-core descriptor bytes; do not divide by core count
N_base = B/T_1+H_1 (k<=2), B/T_2+H_2 (k>2)
B_j = B / product(ls_t for t<j)
is_hat_0=is_0; os_hat_0=os_0
is_hat_j=abs(is_j-sum(ls_t*is_t for t<j))+1
os_hat_j=abs(os_j-sum(ls_t*os_t for t<j))+1
N_D = N_base + sum_j N_1_prime(B_j,is_hat_j,os_hat_j,k)
```

N_1_prime is the Round2 N_G+N_GU correction, multiplied by 1D rho for k>2.
Compute the baseline once, with neither legacy h coefficients nor d_0 scaling.
Measurements determine evaluation errors only, not inherited coefficients or predictions.

## Running

Run from `autofuse/tools/nddma/`; configure NPU_ARCH before collection as described in the [tool guide](../README_en.md).

```bash
python3 Modeling/Data_collection/round3/scripts/e2e.py all --msprof-bin "$(command -v msprof)"
python3 Modeling/Data_collection/round3/scripts/e2e.py fit --measurement-csv /path/to/measurements.csv
```

Read `Modeling/Ana/round2/round2_1d_noncontiguous_model.json` by default;
write outputs to `Modeling/Ana/round3`. Checked-in JSONs under Modeling/Ana are historical snapshots;
old measurements do not constitute the complete aligned evaluation grid.
