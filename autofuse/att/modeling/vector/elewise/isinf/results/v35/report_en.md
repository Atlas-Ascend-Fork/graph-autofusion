# IsInf Model Validation Report

> This report is based on measured data from [samples.csv](samples.csv) (5 cases, the
> float32->bool main path, reusing the same parameter gradient as IsNan). model_predict_cycles
> are derived from the formulas in [design.md](../../../design.md); all 5 match the closed-form
> formulas. Error definition: `error = (predict - measured) / measured * 100%`; positive means
> overestimation.

## Environment and Version

See [manifest.json](manifest.json).

## Prediction Basis

- All cases are the float32 -> bool branch (codegen output is always bool, MaskPack always
  executes), `outer_loop = 100`, `rt = Ceil(cal_count / 64)`.
- Prediction formula: `(48+7·rt)·cc` (head 20 + calibrated latency 28 + CompareScalarEQ 2rt +
  MaskPack x2 4rt + Store rt).

## Error Analysis

### Case Details

| input->output | outer_loop | cal_count | rt | Predicted (cycles) | Measured (cycles) | Error | Per-block conversion (model 7) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| float32->bool | 100 | 64 | 1 | 5500 | 5906.92 | -6.89% | 11.07 |
| float32->bool | 100 | 128 | 2 | 6200 | 6045.45 | +2.56% | 6.23 |
| float32->bool | 100 | 256 | 4 | 7600 | 7216.82 | +5.31% | 6.04 |
| float32->bool | 100 | 512 | 8 | 10400 | 10080.17 | +3.17% | 6.60 |
| float32->bool | 100 | 1024 | 16 | 16000 | 16004.03 | -0.03% | 7.00 |

Overall: mean +0.82%, MAE 3.59%, maximum absolute error 6.89% (rt=1).

### Key Findings

1. **rt=16 matches exactly** (error -0.03%, per-block conversion 7.00 vs model 7), validating
   the calibrated constant 28 and the count of 7 cycles/block
   (CompareScalarEQ x2 + MaskPack x2 + Store).
2. **rt=1 underestimates by -6.9%**: per-block conversion 11.07 vs model 7; the single-block
   scenario carries about +4 cycles of unmodeled fixed overhead, consistent with the same
   parameter points of IsNan (+3.6) / IsFinite (+4.4); the three sibling APIs in the family
   share the same order of single-block overhead (+3~4 cycles).
3. **rt=2/4/8 slightly overestimate** (+2.6%/+5.3%/+3.2%, per-block conversion 6.0~6.6 vs 7);
   the error is non-monotonic in rt, similar to the mid-range fluctuation of IsNan/IsFinite.
4. The sample size is small (5 cases on a single branch); the above patterns need more gradient
   cases to confirm.

## Conclusion

- The float32->bool branch has an overall MAE of 3.59% (at the same level as IsNan 3.06% and
  IsFinite 4.61%); the calibrated constant and instruction count are basically correct and the
  model is usable for performance evaluation.
- Improvement direction: add about +4 cycles of fixed overhead for the single-block (rt=1)
  scenario (consistent across the three sibling APIs).

## Limitations

- Only the single float32->bool branch is covered (5 cases); float16->bool, bf16 input (handled
  as fp32 after conversion), and the `valid=false` fallback path are not measured (the model
  does not distinguish output dtype, MaskPack always executes).
- outer_loop covers only 100; full / tail blocks (cal_count not a multiple of 64) are not
  sampled separately.
- The error is non-monotonic in rt (rt=2/4/8 overestimate); the pattern awaits more samples.
- cases.csv and samples.csv are isomorphic (5 rows); sync if the measured case set is expanded
  later.
- The cycles basis is the AIV_VEC pipe expression value; the conversion to the profiling basis
  (e.g. us/cycle frequency) depends on the measurement environment.
