# Compare Model Validation Report

> This report is based on measured data from [samples.csv](samples.csv) (18 cases, all 2-D,
> covering the tensor/scalar input forms). model_predict_cycles are derived from the closed-form
> formulas in [design.md](../../../design.md); all 18 match the closed-form formulas.
> Error definition: `error = (predict - measured) / measured * 100%`; positive means overestimation.
>
> **Data repair note**: the original samples.csv had two format issues that were fixed --
> (1) the first 10 rows used a full-width comma (U+FF0C) between dtype and is_scalar, causing
> column misalignment; replaced with the ASCII comma; (2) the dims/prediction of the large-scale
> `int64` row were misaligned; corrected by the case owner ([1,6960], prediction 62016).

## Environment and Version

See [manifest.json](manifest.json).

## Prediction Basis

- 2-D scenario: `output_dims = [output_dim0, output_dim1]`,
  `bc = output_dim0 x Ceil(output_dim1 / vl_size)`, `cc = outer_loop`.
- Prediction formulas (bc = block_count, cc = outer_loop):

| Branch | vl_size | tensor formula | scalar formula |
| --- | --- | --- | --- |
| int8 (8-bit) | 256 | (42+3·bc)·cc | (43+3·bc)·cc |
| float16 (16-bit) | 128 | (56+4·bc)·cc | (57+4·bc)·cc |
| float32 (32-bit) | 64 | (61+5·bc)·cc | (62+5·bc)·cc |
| int64 (64-bit) | 64 | (63+5·bc)·cc | not covered |

## Error Analysis

### Branch Summary

| Branch | Cases | Mean error | Error range |
| --- | --- | --- | --- |
| float32 scalar | 5 | +12.61% | [-4.85%, +39.19%] |
| float16 scalar | 5 | +0.03% | [-25.83%, +28.68%] |
| int64 tensor | 4 | +3.23% | [-2.48%, +8.00%] |
| int8 tensor | 4 | +0.32% | [-6.87%, +17.64%] |
| All | 18 | +4.30% | [-25.83%, +39.19%] |

Overall MAE 12.63%, maximum absolute error 39.19%.

### Case Details

| input | is_scalar | outer_loop | dims | bc | Predicted (cycles) | Measured (cycles) | Error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| float32 | true | 100 | [4,256] | 16 | 14200 | 11058.35 | +28.41% |
| float32 | true | 100 | [4,512] | 32 | 22200 | 15948.92 | +39.19% |
| float32 | true | 100 | [4,3] | 4 | 8200 | 7843.44 | +4.55% |
| float32 | true | 100 | [100,3] | 100 | 56200 | 58697.74 | -4.26% |
| float32 | true | 100 | [200,3] | 200 | 106200 | 111616.43 | -4.85% |
| float16 | true | 100 | [4,256] | 8 | 8900 | 8105.84 | +9.80% |
| float16 | true | 100 | [4,512] | 16 | 12100 | 9403.19 | +28.68% |
| float16 | true | 100 | [4,3] | 4 | 7300 | 6712.86 | +8.75% |
| float16 | true | 100 | [100,3] | 100 | 45700 | 58017.45 | -21.23% |
| float16 | true | 100 | [200,3] | 200 | 85700 | 115541.38 | -25.83% |
| int64 | false | 102 | [1,3] | 1 | 6936 | 7112.33 | -2.48% |
| int64 | false | 102 | [1,256] | 4 | 8466 | 8138.34 | +4.03% |
| int64 | false | 102 | [1,512] | 8 | 10506 | 10164.79 | +3.36% |
| int64 | false | 102 | [1,6960] | 109 | 62016 | 57420.63 | +8.00% |
| int8 | false | 18 | [1,3] | 1 | 810 | 868.00 | -6.68% |
| int8 | false | 18 | [1,256] | 1 | 810 | 869.72 | -6.87% |
| int8 | false | 18 | [1,512] | 2 | 864 | 888.97 | -2.81% |
| int8 | false | 18 | [1,2560] | 10 | 1296 | 1101.65 | +17.64% |

### Key Findings

1. **Full-block overestimation**: when the inner dim is an integer multiple of vl (no tail
   block), the model systematically overestimates -- fp32 scalar [4,256]/[4,512]
   (+28.4%/+39.2%), fp16 scalar [4,512] (+28.7%), int8 [1,2560] (+17.6%). Converted per-block
   actual costs: fp32 ~= 3.0 cycles/block (model 5), fp16 ~= 2.3~3.0 cycles/block (model 4),
   int8 ~= 1.9 cycles/block (model 3); the hardware has inter-block pipelining / parallel
   masking that the model's linear term does not include.
2. **Masked-tail blocks cost more than full blocks**: when dim1=3 each block has only 3 valid
   elements but is counted as a full block; fp16 [100,3]/[200,3] underestimate by
   -21.2%/-25.8% (per-block ~= 5.2~5.5 cycles, model 4), fp32 [100,3]/[200,3] slightly
   underestimate by -4.3%/-4.9% (per-block ~= 5.3 cycles, model 5). The model does not
   distinguish full blocks from tail blocks, and the two scenarios err in opposite directions.
3. **int64 performs best across the range**: [1,3] ~-2.5%, [1,256]/[1,512] +3.4%/+4.0%, and the
   large-scale [1,6960] (bc=109, including 1 tail block) +8.0%; full blocks convert to ~=
   4.2~4.6 cycles/block (model 5), the tail block [1,3] converts to ~= 6.7 cycles/block,
   showing the same "full-block slight overestimate, tail-block underestimate" trend.
4. **int8 slight underestimate at small scale**: -2.8%~-6.9% when bc<=2; the constant term (42)
   is slightly small; at bc=10 it flips to +17.6% overestimation, consistent with the
   full-block overestimation trend.
5. **is_scalar path**: fp32/fp16 scalar shares the same dimensional basis with the tensor path;
   the scalar +1 Duplicate constant introduces no extra systematic bias (fp32 [4,3] is only
   +4.6%).

## Conclusion

- Overall MAE 12.63%, between Transpose (±1.5%) and Cast (MAE 74%); the directional bias is
  regular (full-block overestimate, tail-block underestimate) and there is room for correction.
- Improvement directions:
  - distinguish per-block costs of full blocks and tail blocks (measured conversions: fp32
    full ~=3.0 / tail ~=5.3, fp16 full ~=2.3~3.0 / tail ~=5.3, int64 full ~=4.4 / tail ~=6.7);
  - introduce inter-block pipeline masking modeling for large bc;
  - raise the int8 constant term (42) by about 5~7 cycles to improve small-bc scenarios.

## Limitations

- bool/bf16/int16/uint32, 64-bit scalar, and the compare_mode dimension are not covered (the
  current model does not distinguish mode, and this measurement did not sample by mode either).
- The per-block cost conversion for full/tail blocks assumes "constants per formula, linear term
  per measurement"; the sample size is limited (2-4 points per class) and needs more gradient
  cases to confirm.
- cases.csv remains the early mock case set (49 rows with compare_mode/block_count columns),
  inconsistent with the measured case set (18 rows with output_dim0/output_dim1 columns); to be
  synced.
- The cycles basis is the AIV_VEC pipe expression value; the conversion to the profiling basis
  (e.g. us/cycle frequency) depends on the measurement environment.
