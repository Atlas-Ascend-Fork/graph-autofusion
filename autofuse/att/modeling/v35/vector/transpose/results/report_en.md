# Transpose Model Validation Report

> This report is based on measured data from [samples.csv](samples.csv) (46 cases). model_predict_cycles
> come from the closed-form formulas in [design.md](../../../design.md) under the assumptions
> stride_count=1, gather_count=2, repeat_elm=64, multiplied by outer_loop.
> Error definition: `error = (predict - measured) / measured * 100%`; positive means overestimation.

## Environment and Version

See [manifest.json](manifest.json).

## Prediction Basis

- The `repeat` column is repeat_time (cal_count = 64 * repeat), `tail_repeat = repeat - 1`.
- `inner_stride = 1` -> stride_count = 1 -> gather_count = 2 (for inner_dim 1/2/3 alike).
- Per-branch instruction cost formulas (od = outer_dim, rt = repeat):

| Branch (total_dim, inner_dim) | Index gen | Movement | Latency | Formula |
| --- | --- | --- | --- | --- |
| (2,1) / (3,1) / (4,1) | GenOne = 3·rt | 3·od·rt | 69 | (3+3·od)·rt+89 |
| (3,2) / (4,2) | GenTwo = 17+17·(rt-1) | 3·od·rt | 167 | (3·od+17)·rt+187 |
| (4,3) | GenThree = 17+17·(rt-1) | 3·od·rt | 386 | (3·od+17)·rt+406 |
| (2,2) (generic) | GenTwo = 17+17·(rt-1) | 3·rt | 167 | 20·rt+187 |
| (3,3) (generic) | GenThree = 17+17·(rt-1) | 3·rt | 386 | 20·rt+406 |

- Final prediction = branch formula x outer_loop (25 for the (4,1) branch, 22 otherwise).

## Error Analysis

### Branch Summary

| Branch | Cases | Mean error | Max absolute error |
| --- | --- | --- | --- |
| (2,1) | 7 | -0.25% | 0.86% |
| (2,2) | 4 | +0.64% | 1.21% |
| (3,1) | 7 | +0.86% | 7.58% |
| (3,2) | 7 | +0.32% | 1.08% |
| (3,3) | 4 | +0.47% | 1.17% |
| (4,1) | 3 | -0.01% | 0.44% |
| (4,2) | 7 | +0.25% | 1.05% |
| (4,3) | 7 | -0.01% | 1.48% |
| All | 46 | +0.27% | 7.58% |

Overall MAE 0.69%, RMSE 1.30%; 45 of 46 cases fall within ±1.5%.

### Case Details

| Branch | outer_loop | outer_dim | cal_count | repeat | Predicted (cycles) | Measured (cycles) | Error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (2,1) | 22 | 1 | 64 | 1 | 2090 | 2091.93 | -0.09% |
| (2,1) | 22 | 1 | 128 | 2 | 2222 | 2218.48 | +0.16% |
| (2,1) | 22 | 1 | 256 | 4 | 2486 | 2484.03 | +0.08% |
| (2,1) | 22 | 1 | 512 | 8 | 3014 | 3013.83 | +0.01% |
| (2,1) | 22 | 2 | 64 | 1 | 2156 | 2174.62 | -0.86% |
| (2,1) | 22 | 4 | 64 | 1 | 2288 | 2300.83 | -0.56% |
| (2,1) | 22 | 8 | 64 | 1 | 2552 | 2564.97 | -0.51% |
| (2,2) | 22 | 1 | 64 | 1 | 4554 | 4528.79 | +0.56% |
| (2,2) | 22 | 1 | 128 | 2 | 4994 | 4985.86 | +0.16% |
| (2,2) | 22 | 1 | 256 | 4 | 5874 | 5836.30 | +0.65% |
| (2,2) | 22 | 1 | 512 | 8 | 7634 | 7542.57 | +1.21% |
| (3,1) | 22 | 1 | 64 | 1 | 2090 | 2081.28 | +0.42% |
| (3,1) | 22 | 1 | 128 | 2 | 2222 | 2065.46 | +7.58% |
| (3,1) | 22 | 1 | 256 | 4 | 2486 | 2487.08 | -0.04% |
| (3,1) | 22 | 1 | 512 | 8 | 3014 | 3009.47 | +0.15% |
| (3,1) | 22 | 2 | 64 | 1 | 2156 | 2174.01 | -0.83% |
| (3,1) | 22 | 4 | 64 | 1 | 2288 | 2304.39 | -0.71% |
| (3,1) | 22 | 8 | 64 | 1 | 2552 | 2566.66 | -0.57% |
| (3,2) | 22 | 1 | 64 | 1 | 4554 | 4528.56 | +0.56% |
| (3,2) | 22 | 1 | 128 | 2 | 4994 | 4992.19 | +0.04% |
| (3,2) | 22 | 1 | 256 | 4 | 5874 | 5842.74 | +0.54% |
| (3,2) | 22 | 1 | 512 | 8 | 7634 | 7552.35 | +1.08% |
| (3,2) | 22 | 2 | 64 | 1 | 4620 | 4612.43 | +0.16% |
| (3,2) | 22 | 4 | 64 | 1 | 4752 | 4755.69 | -0.08% |
| (3,2) | 22 | 8 | 64 | 1 | 5016 | 5018.66 | -0.05% |
| (3,3) | 22 | 1 | 64 | 1 | 9372 | 9445.49 | -0.78% |
| (3,3) | 22 | 1 | 128 | 2 | 9812 | 9730.02 | +0.84% |
| (3,3) | 22 | 1 | 256 | 4 | 10692 | 10568.86 | +1.17% |
| (3,3) | 22 | 1 | 512 | 8 | 12452 | 12372.54 | +0.64% |
| (4,1) | 25 | 2 | 64 | 1 | 2450 | 2457.92 | -0.32% |
| (4,1) | 25 | 4 | 64 | 1 | 2600 | 2603.58 | -0.14% |
| (4,1) | 25 | 8 | 64 | 1 | 2900 | 2887.42 | +0.44% |
| (4,2) | 22 | 1 | 64 | 1 | 4554 | 4532.79 | +0.47% |
| (4,2) | 22 | 1 | 128 | 2 | 4994 | 4978.29 | +0.32% |
| (4,2) | 22 | 1 | 256 | 4 | 5874 | 5853.28 | +0.35% |
| (4,2) | 22 | 1 | 512 | 8 | 7634 | 7554.60 | +1.05% |
| (4,2) | 22 | 2 | 64 | 1 | 4620 | 4615.80 | +0.09% |
| (4,2) | 22 | 4 | 64 | 1 | 4752 | 4764.33 | -0.26% |
| (4,2) | 22 | 8 | 64 | 1 | 5016 | 5027.94 | -0.24% |
| (4,3) | 22 | 1 | 64 | 1 | 9372 | 9440.77 | -0.73% |
| (4,3) | 22 | 1 | 128 | 2 | 9812 | 9756.42 | +0.57% |
| (4,3) | 22 | 1 | 256 | 4 | 10692 | 10543.56 | +1.41% |
| (4,3) | 22 | 1 | 512 | 8 | 12452 | 12270.84 | +1.48% |
| (4,3) | 22 | 2 | 64 | 1 | 9438 | 9534.91 | -1.02% |
| (4,3) | 22 | 4 | 64 | 1 | 9570 | 9656.07 | -0.89% |
| (4,3) | 22 | 8 | 64 | 1 | 9834 | 9924.05 | -0.91% |

### Key Findings

1. **Overall accuracy is good**: 45 of 46 cases fall within ±1.5%; overall MAE 0.69%. The branch
   latency constants (69/167/386) and the MicroAPI throughput table calibration match the
   measurements.
2. **Outlier**: (3,1) cal_count=128 (rt=2) predicts 2222 vs measured 2065.46 (+7.58%). The same
   parameter combination behaves normally in other branches ((2,1) +0.16%, (4,2) +0.32%);
   likely a single-run measurement fluctuation. Re-measurement recommended.
3. **Slight overestimation at long repeats**: at cal_count=512 (rt=8), (2,2)/(3,2)/(4,2) all
   overestimate by +1.0%~+1.2%; the movement term 3·rt (kGather2/kStore throughput = 1) is
   slightly conservative at large repeats.
4. **(4,3) od>=2 slightly underestimates**: at rt=1, od ∈ {2,4,8} the model underestimates by
   -0.9%~-1.0%; the GenThree index segment or the 386 latency constant is slightly low.

## Conclusion

- Except for one outlier pending re-measurement, all cases are within ±1.5%; the model is usable
  for ATT performance evaluation.
- The branch structure and calibrated constants (head=20, 69/167/386, GenTwo/GenThree 17+17·tr)
  are correct overall.

## Limitations

- Only stride_count=1 is covered (inner_stride=1, not divisible by the alignment 64 with
  remainder 1); aligned scenarios (stride_count=64, gather_count=128/65) are not covered.
- The repeat basis is repeat_elm=64 (cal_count = 64·repeat); other repeat_elm values (e.g. 128
  for fp16) are not covered.
- outer_loop only covers 22 / 25; outer_dim covers {1,2,4,8}, where (2,2)/(3,3) take the generic
  branch (outer_dim does not participate), and (4,1) does not cover od=1.
- The contribution of unmodeled items (UpdateMask, GenThree's Sub/Adds/CompareScalarGE, etc.) to
  the error is not quantified separately.
- The cycles basis is the AIV_VEC pipe expression value; the conversion to the profiling basis
  (e.g. us/cycle frequency) depends on the measurement environment.
