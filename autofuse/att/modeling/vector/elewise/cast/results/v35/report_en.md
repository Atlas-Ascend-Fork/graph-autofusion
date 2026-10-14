# Cast Model Validation Report

> This report is based on measured data from [samples.csv](samples.csv) (17 cases, covering the 8
> dtype dispatch branches of `CastPerf`). model_predict_cycles are the model outputs of the
> `CastPerf` implementation (`autofuse/v35/att/api_perf_register/ascendc_regbase_perf.cpp`).
> Error definition: `error = (predict - measured) / measured * 100%`; positive means overestimation.

## Environment and Version

See [manifest.json](manifest.json).

## Case and Branch Coverage

| Dispatch branch | dtype pairs | Cases |
| --- | --- | --- |
| Same-bit integer (`AddSameBitIntegerCastPerf`) | int8->uint8, uint64->int64 | 2 |
| Generic float (single default `kCast`) | fp16->fp32, fp32->fp16 | 2 |
| Two-step conversion (int32->fp32->fp16) | int32->float16 | 1 |
| B64 (`AddB64CastPerf`) | int64->fp32, fp32->int64, int32->int64 | 3 |
| B64 transfer (`AddB64TransferCastPerf`) | fp16->int64, int64->fp16 | 2 |
| B8 specific (`AddB8SpecificCastPerf`) | uint8->fp32/int32/int16, int16->uint8/int8 | 5 |
| Int64->UInt8 (`AddInt64ToUInt8CastPerf`) | int64->uint8 | 1 |
| UInt8->Int64 (`AddUInt8ToInt64CastPerf`) | uint8->int64 | 1 |

## Error Analysis

### Branch Summary

| Dispatch branch | Cases | Mean error | Error range |
| --- | --- | --- | --- |
| Same-bit integer | 2 | -28.25% | [-37.32%, -19.19%] |
| Generic float | 2 | +40.86% | [+34.84%, +46.88%] |
| Two-step conversion | 1 | +95.49% | [+95.49%, +95.49%] |
| B64 | 3 | +69.88% | [+63.17%, +73.87%] |
| B64 transfer | 2 | +41.76% | [+41.71%, +41.81%] |
| B8 specific | 5 | +128.01% | [+97.94%, +147.92%] |
| Int64->UInt8 | 1 | +64.30% | [+64.30%, +64.30%] |
| UInt8->Int64 | 1 | +27.58% | [+27.58%, +27.58%] |
| All | 17 | +67.40% | [-37.32%, +147.92%] |

Overall MAE 74.05%, maximum absolute error 147.92%.

### Case Details

| input | output | outer_loop | cal_count | Predicted (cycles) | Measured (cycles) | Error | Dispatch branch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| int8 | uint8 | 7 | 57264 | 2518 | 3116 | -19.19% | Same-bit integer |
| float16 | int64 | 32 | 12528 | 64423 | 45462 | +41.71% | B64 transfer |
| int64 | float16 | 32 | 12528 | 64423 | 45428 | +41.81% | B64 transfer |
| uint8 | float32 | 16 | 25056 | 32583 | 16461 | +97.94% | B8 specific |
| uint8 | int32 | 16 | 25056 | 38855 | 16554 | +134.72% | B8 specific |
| uint8 | int16 | 10 | 40080 | 16725 | 6746 | +147.92% | B8 specific |
| int16 | uint8 | 10 | 40080 | 10445 | 4927 | +112.00% | B8 specific |
| int16 | int8 | 10 | 40080 | 16725 | 6758 | +147.48% | B8 specific |
| int64 | float32 | 38 | 10552 | 64539 | 39554 | +63.17% | B64 |
| float32 | int64 | 38 | 10552 | 39345 | 22629 | +73.87% | B64 |
| int64 | uint8 | 29 | 13824 | 64227 | 39092 | +64.30% | Int64->UInt8 |
| uint8 | int64 | 29 | 13824 | 89283 | 69981 | +27.58% | UInt8->Int64 |
| int32 | int64 | 38 | 10552 | 39345 | 22797 | +72.59% | B64 |
| uint64 | int64 | 51 | 7864 | 14341 | 22879 | -37.32% | Same-bit integer |
| int32 | float16 | 19 | 21096 | 26330 | 13469 | +95.49% | Two-step conversion |
| float16 | float32 | 19 | 21096 | 13771 | 10213 | +34.84% | Generic float |
| float32 | float16 | 19 | 21096 | 13771 | 9376 | +46.88% | Generic float |

### Key Findings

1. **Systematic overestimation**: 15 of 17 cases are overestimated; the overall mean is +67.40%,
   significantly worse than the Transpose model in this repository (±1.5%, see
   `vector/transpose/results/v35/report.md`). The consistent direction indicates that the
   throughput summation does not model instruction-level parallelism / pipeline masking (noted as
   a future modeling item in `vf_perf_utils.cpp`), and that the `kCast` dtype-mapping default
   entry throughput of 2 is too large.
2. **Same-bit integer branches are the only underestimates** (-19% ~ -37%): this branch only
   counts UpdateMask x2 + Load + Store; the measured cost is notably higher, and the actual
   execution path may contain extra instructions or non-aligned DataCopy overhead.
3. **B8 specific branches have the largest errors** (+98% ~ +148%): the two `kCast` default
   entries (5, 2) in `AddB8SpecificCastPerf` plus the UnPack / Pack counts overestimate clearly;
   uint8->fp32 (+97.94%) is better than int16->int8 / uint8->int16 (about +148%).
4. **UInt8->Int64 is relatively the best** (+27.58%): the store in this branch uses the
   dtype-mapping special case (uint8->int64 throughput 5), whose calibration is relatively close
   to the measurement.
5. **B64 / B64 transfer / two-step conversion** overestimate by +42% ~ +96%, in between.

## Conclusion

- The model's trend direction is mostly consistent (overestimation everywhere except the
  same-bit branches), but the numeric deviation is large (MAE 74.05%); it cannot currently be
  used directly for precise performance evaluation.
- Re-calibration needed: `kCast` / `kStore` dtype-mapping throughputs, instruction counts of the
  B8 specific and same-bit branches, or introduce instruction concurrency / pipeline masking
  modeling before re-measuring.

## Limitations

- Only 1-5 cases per branch; gradients of cal_count / outer_loop are not covered, so constant
  and linear error terms cannot be separated.
- B4 (fp16<->int4), fp32->bool, fp16<->bf16, the uint8->int4 special case (one_rep_size=128),
  and int64<->uint64 same-bit branches are not covered.
- The predicted values have a constant offset from the closed-form instruction formulas in
  [design.md](../../../design.md) (+775 cycles in 16 of 17 cases, +520 for uint64->int64); the
  origin of the offset is pending confirmation and should be noted in error attribution.
- cases.csv remains the early mock case set (49 rows with a repeat column), inconsistent with
  the measured case set (17 rows); to be synced.
- The cycles basis is the AIV_VEC pipe expression value; the conversion to the profiling basis
  (e.g. us/cycle frequency) depends on the measurement environment.
