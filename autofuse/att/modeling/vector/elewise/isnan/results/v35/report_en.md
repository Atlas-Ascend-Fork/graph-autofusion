# IsNan Model Validation Report

> This report is based on measured data from [samples.csv](samples.csv) (5 cases, the
> float32->bool main path). model_predict_cycles are derived from the formulas in
> [design.md](../../../design.md); all 5 match the closed-form formulas.
> Error definition: `error = (predict - measured) / measured * 100%`; positive means
> overestimation.

## Environment and Version

See [manifest.json](manifest.json).

## Prediction Basis

- All cases are the float32 -> bool branch (the codegen main path, output
  `ReinterpretCast<bool>`), `outer_loop = 100`, `rt = Ceil(cal_count / 64)`.
- Prediction formula: `(46+6·rt)·cc` (head 20 + calibrated latency 26 + Ne/Store/MaskPack x2
  throughput 6rt).

## Error Analysis

### Case Details

| input->output | outer_loop | cal_count | rt | Predicted (cycles) | Measured (cycles) | Error | Per-block conversion (model 6) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| float32->bool | 100 | 64 | 1 | 5200 | 5561.70 | -6.50% | 9.62 |
| float32->bool | 100 | 128 | 2 | 5800 | 5615.12 | +3.29% | 5.08 |
| float32->bool | 100 | 256 | 4 | 7000 | 6819.56 | +2.65% | 5.55 |
| float32->bool | 100 | 512 | 8 | 9400 | 9399.63 | +0.00% | 6.00 |
| float32->bool | 100 | 1024 | 16 | 14200 | 14620.55 | -2.88% | 6.26 |

Overall: mean -0.69%, MAE 3.06%, maximum absolute error 6.50% (rt=1).

### Key Findings

1. **rt=8 matches exactly** (error +0.00%, per-block conversion 6.00 vs model 6), validating
   the calibrated constant 26 and the count of 6 cycles/block (Ne + Store + MaskPack x2).
2. **rt=1 underestimates by -6.5%**: per-block conversion 9.62 vs model 6; the single-block
   scenario carries about +3.6 cycles of unmodeled fixed overhead (consistent with the trend of
   row-entry overhead in the small-scale Compare/Where scenarios).
3. **rt=2/4 slightly overestimate** (+3.3%/+2.7%, per-block 5.08~5.55), rt=16 slightly
   underestimates (-2.9%, per-block 6.26); the error is non-monotonic in rt and all within
   ±7%, suspected to contain measurement fluctuation.
4. The sample size is small (5 cases on a single branch); the above patterns need more gradient
   cases to confirm.

## Conclusion

- The float32->bool branch has an overall MAE of 3.06% (better than Compare 12.6% and Where
  4.1%); the calibrated constant and instruction count are basically correct and the model is
  usable for performance evaluation.
- Improvement direction: add about +3~4 cycles of fixed overhead for the single-block (rt=1)
  scenario.

## Limitations

- Only the single float32->bool branch is covered (5 cases); float16->bool, uint8 output (SIMT
  path), and bf16 input (handled as fp32 after conversion) are not measured.
- outer_loop covers only 100; full / tail blocks (cal_count not a multiple of 64) are not
  sampled separately.
- The error is non-monotonic in rt (rt=2/4 overestimate, rt=1/16 underestimate); the pattern
  awaits more samples.
- cases.csv remains the early mock case set (28 rows with output_dtype/repeat columns),
  inconsistent with the measured case set (5 rows); to be synced.
- The cycles basis is the AIV_VEC pipe expression value; the conversion to the profiling basis
  (e.g. us/cycle frequency) depends on the measurement environment.
