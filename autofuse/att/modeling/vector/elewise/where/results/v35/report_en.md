# Where Model Validation Report

> This report is based on measured data from [samples.csv](samples.csv) (26 cases, all 2-D).
> model_predict_cycles are derived from the closed-form formulas in [design.md](../../../design.md);
> all 26 match the closed-form formulas. Error definition:
> `error = (predict - measured) / measured * 100%`; positive means overestimation.
>
> **Modeling correction note**: verification of the measured predictions found an error in the
> original B8 2-D formula derivation in design.md -- the trigger condition of
> UpdateMask/CompareScalarNE in `AddWhereComputePerf` is `!IsWhereB16B32Dtype` (B8 also executes
> it on the 2-D path), so B8 2-D should be 4 cycles per block instead of 2; design.md has been
> corrected to `(46+4·cf·rt)·cc` (consistent with the predictions in samples.csv). The 1-D path
> condition is `!IsWhereB8B16B32Dtype` (B8 does not execute it); `(46+2·rt)·cc` is unchanged.

## Environment and Version

See [manifest.json](manifest.json).

## Prediction Basis

- All cases are 2-D: `rt = Ceil(output_dim1 / repeat_elm)`, `cf = output_dim0`,
  `cc = outer_loop` (1 for int8/int64, 10 for float16/float32).
- Prediction formulas:

| Branch | repeat_elm | 2-D formula |
| --- | --- | --- |
| int8 (B8) | 256 | (46+4·cf·rt)·cc |
| float16 (B16) | 128 | (45+(3·rt+4)·cf)·cc |
| float32 (B32) | 64 | (49+(4·rt+3)·cf)·cc |
| int64 (B64) | 64 | (65+7·cf·rt)·cc |

- 2-D B16/B32 take the per-row path: `row_max_latency` is discarded (max_latency=0) and the row
  overhead (B16=4 / B32=3) x cf is added; B8/B64 count the whole block_count (= cf·rt) with
  max_latency accumulated normally.

## Error Analysis

### Branch Summary

| Branch | Cases | Mean error | Error range |
| --- | --- | --- | --- |
| int8 (B8) | 6 | +4.73% | [-2.91%, +22.14%] |
| float16 (B16) | 7 | -4.51% | [-9.70%, +0.52%] |
| float32 (B32) | 7 | -2.90% | [-11.09%, +0.61%] |
| int64 (B64) | 6 | +0.20% | [-4.37%, +1.84%] |
| All | 26 | -0.86% | [-11.09%, +22.14%] |

Overall MAE 4.09%; 23 of 26 cases within ±10%.

### Case Details

| dtype | outer_loop | dims | rt | bc | Predicted (cycles) | Measured (cycles) | Error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| int64 | 1 | [2,64] | 1 | 2 | 79 | 82.61 | -4.37% |
| int64 | 1 | [2,256] | 4 | 8 | 121 | 121.15 | -0.13% |
| int64 | 1 | [2,512] | 8 | 16 | 177 | 175.66 | +0.77% |
| int64 | 1 | [2,1024] | 16 | 32 | 289 | 285.12 | +1.36% |
| int64 | 1 | [4,512] | 8 | 32 | 289 | 284.08 | +1.73% |
| int64 | 1 | [8,512] | 8 | 64 | 513 | 503.74 | +1.84% |
| int8 | 1 | [2,64] | 1 | 2 | 54 | 55.62 | -2.91% |
| int8 | 1 | [2,256] | 1 | 2 | 54 | 52.72 | +2.43% |
| int8 | 1 | [2,512] | 2 | 4 | 62 | 56.97 | +8.83% |
| int8 | 1 | [2,1024] | 4 | 8 | 78 | 63.86 | +22.14% |
| int8 | 1 | [4,512] | 2 | 8 | 78 | 77.50 | +0.65% |
| int8 | 1 | [8,512] | 2 | 16 | 110 | 113.15 | -2.78% |
| float32 | 10 | [10,3] | 1 | 10 | 1190 | 1255.49 | -5.22% |
| float32 | 10 | [100,3] | 1 | 100 | 7490 | 7551.20 | -0.81% |
| float32 | 10 | [200,3] | 1 | 200 | 14490 | 14443.35 | +0.32% |
| float32 | 10 | [2,64] | 1 | 2 | 630 | 708.61 | -11.09% |
| float32 | 10 | [2,256] | 4 | 8 | 870 | 913.03 | -4.71% |
| float32 | 10 | [2,512] | 8 | 16 | 1190 | 1183.21 | +0.57% |
| float32 | 10 | [2,1024] | 16 | 32 | 1830 | 1818.87 | +0.61% |
| float16 | 10 | [10,3] | 1 | 10 | 1150 | 1267.40 | -9.26% |
| float16 | 10 | [100,3] | 1 | 100 | 7450 | 7430.87 | +0.26% |
| float16 | 10 | [200,3] | 1 | 200 | 14450 | 14375.82 | +0.52% |
| float16 | 10 | [2,64] | 1 | 2 | 590 | 645.70 | -8.63% |
| float16 | 10 | [2,256] | 2 | 4 | 650 | 650.55 | -0.08% |
| float16 | 10 | [2,512] | 4 | 8 | 770 | 852.76 | -9.70% |
| float16 | 10 | [2,1024] | 8 | 16 | 1010 | 1059.49 | -4.67% |

### Key Findings

1. **int64 (B64) performs best across the range**: -4.4% ~ +1.8%; a stable slight
   overestimation of +0.8%~+1.8% when bc>=16, validating the B64 count of 7 cycles per block
   (UpdateMask+CompareScalarNE+MaskUnPack x2+Select x2+Store).
2. **int8 overestimates at large rt**: [2,1024] (rt=4) +22.1%, [2,512] (rt=2) +8.8%; while
   [4,512] (rt=2, cf=4) with the same bc is only +0.65%. Converted per-block actual costs:
   ~2.2 at rt=4, ~2.7 at rt=2 with cf=2, ~3.9~4.2 when cf>=4 (model 4) -- the longer the inner
   repeat, the fuller the inter-block pipeline masking, which the model does not include.
3. **B16/B32 underestimate at small scale**: fp32 [2,64] -11.1%, fp16 [2,64] -8.6%,
   fp16 [10,3] -9.3%, fp32 [10,3] -5.2%. Converted per-row cost (constant deducted, / cf):
   fp32 ~10.9 vs model 7 at rt=1, ~34.7 vs model 35 at rt>=8 -- small rt rows carry about +4
   cycles of unmodeled row-entry overhead; it converges as cf grows ([100,3]/[200,3] error
   within ±0.8%).
4. **fp16 data points fluctuate**: [2,256] (rt=2) matches at -0.08% but [2,512] (rt=4)
   underestimates at -9.7%; non-monotonic, suspected measurement fluctuation; re-measurement
   recommended.
5. **2-D per-row path modeling validated**: fp32/fp16 are within ±0.8% at cf=100/200, rt=1
   (tail blocks); the "discard row_max_latency + row overhead 4/3" modeling matches the
   measurements.

## Conclusion

- Overall MAE 4.09%, better than Compare (12.63%) and Cast (74%), worse than Transpose (±1.5%);
  23 of 26 cases within ±10%.
- Improvement directions:
  - introduce inter-block pipeline masking modeling for the int8 large-rt scenario (per-block
    cost decays with rt);
  - add about 2~4 cycles of row-entry overhead per row for B16/B32 small rt (<=4);
  - the +1%~+2% systematic overestimation of B64 at large bc can be fine-tuned via the
    per-block constant.

## Limitations

- Only the 2-D scenario is covered; the 1-D formulas (including the difference that B8 1-D does
  not execute UpdateMask/CompareScalarNE) are not measured.
- The legacy path (`valid=false`), bcast inputs (no effect on predictions), bool/bf16/int16/
  uint32 same-width dtypes, and uint8/uint64 are not covered.
- int8 has only 6 cases and the fp16 rt gradient is incomplete (rt=4 has only 1 case with an
  anomalous deviation); the small-rt row overhead and pipeline effect conversions need more
  gradient cases.
- cases.csv remains the early mock case set (28 rows with scene/repeat columns), inconsistent
  with the measured case set (26 rows); to be synced.
- The cycles basis is the AIV_VEC pipe expression value; the conversion to the profiling basis
  (e.g. us/cycle frequency) depends on the measurement environment.
