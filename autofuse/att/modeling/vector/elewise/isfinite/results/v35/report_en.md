# IsFinite Model Validation Report

> This report is based on measured data from [samples.csv](samples.csv) (5 cases, the
> float32->bool main path, reusing the same parameter gradient as IsNan). model_predict_cycles
> are derived from the formulas in [design.md](../../../design.md); all 5 match the closed-form
> formulas. Error definition: `error = (predict - measured) / measured * 100%`; positive means
> overestimation.

## Environment and Version

See [manifest.json](manifest.json).

## Prediction Basis

- All cases are the float32 -> bool branch (the codegen main path, output
  `ReinterpretCast<bool>`), `outer_loop = 100`, `rt = Ceil(cal_count / 64)`.
- Prediction formula: `(36+8·rt)·cc` (head 20 + calibrated latency 16 + Ne/Store 2rt +
  MaskOr 2rt + MaskPack x2 4rt).

## Error Analysis

### Case Details

| input->output | outer_loop | cal_count | rt | Predicted (cycles) | Measured (cycles) | Error | Per-block conversion (model 8) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| float32->bool | 100 | 64 | 1 | 4400 | 4841.14 | -9.11% | 12.41 |
| float32->bool | 100 | 128 | 2 | 5200 | 5238.04 | -0.73% | 8.19 |
| float32->bool | 100 | 256 | 4 | 6800 | 6170.92 | +10.19% | 6.43 |
| float32->bool | 100 | 512 | 8 | 10000 | 9764.63 | +2.41% | 7.71 |
| float32->bool | 100 | 1024 | 16 | 16400 | 16301.96 | +0.60% | 7.94 |

Overall: mean +0.67%, MAE 4.61%, maximum absolute error 10.19% (rt=4).

### Key Findings

1. **rt=2/8/16 fit well** (errors -0.7%/+2.4%/+0.6%, per-block conversion 7.7~8.2 vs model 8),
   validating the calibrated constant 16 and the count of 8 cycles/block
   (Ne + Store + MaskOr x2 + MaskPack x2).
2. **rt=1 underestimates by -9.1%**: per-block conversion 12.41 vs model 8; the single-block
   scenario carries about +4.4 cycles of unmodeled fixed overhead, consistent with the same
   parameter point of IsNan (+3.6 cycles) and the "row-entry overhead" trend of the small-scale
   Compare/Where scenarios.
3. **rt=4 overestimates by +10.2%**: per-block conversion 6.43, clearly below the neighboring
   rt levels (8.19/7.71 at rt=2/8); the error is non-monotonic in rt, similar to the rt=2/4
   fluctuation of IsNan; suspected measurement fluctuation, re-measurement recommended.
4. The sample size is small (5 cases on a single branch); the above patterns need more gradient
   cases to confirm.

## Conclusion

- The float32->bool branch has an overall MAE of 4.61% (slightly worse than IsNan 3.06%, better
  than Compare 12.6%); the calibrated constant and instruction count are basically correct and
  the model is usable for performance evaluation.
- Improvement directions: add about +4 cycles of fixed overhead for the single-block (rt=1)
  scenario; re-measure the rt=4 anomaly.

## Limitations

- Only the single float32->bool branch is covered (5 cases); float16->bool, uint8 output (SIMT
  path), and bf16 input (handled as fp32 after conversion) are not measured.
- outer_loop covers only 100; full / tail blocks (cal_count not a multiple of 64) are not
  sampled separately.
- The error is non-monotonic in rt (rt=4 overestimates); the pattern awaits more samples.
- cases.csv and samples.csv are isomorphic (5 rows); sync if the measured case set is expanded
  later.
- The cycles basis is the AIV_VEC pipe expression value; the conversion to the profiling basis
  (e.g. us/cycle frequency) depends on the measurement environment.
