# IsInf V2 Performance Model

## Scope

- Targeting the v35 (V2) ASCIR `IsInf` node, corresponding to the API `IsInf` (header
  `adv_api/math/is_inf.h`, codegen generates calls via `UnaryBitWidthChangeApiCallV2`).
- Covers fp16/fp32 input; bf16 input is handled as fp32 after the dtype conversion
  (BF16->FLOAT) (`GetConversionDtype`).
- `IsNan` / `IsFinite` / `IsInf` share the `UnaryBitWidthChange` framework; this archive covers
  `IsInf` only.
- Output pipe is `AIV_VEC`; the copy overhead between Reg and UB is not modeled.

## Source References

- codegen impl: `autofuse/v35/ascir/generator/v2_ascir_codegen_impl.h`
  `IsInfAscIrCodegenImplV2` (`GetApiName()=IsInf`,
  `GetApiCallName()=UnaryBitWidthChangeApiCallV2`, output via `ReinterpretCast<bool>`).
- API source: `adv_api/math/is_inf.h` (asc-devkit, external dependency).
- API call generation: `autofuse/v35/codegen/reg_api_call/unary_bitwidth_change_api_call_v2.cpp`
  `UnaryBitWidthChangeApiCallV2::Generate` / `FillUnaryBitWidthChangeNodeParams`.
- Performance table: `autofuse/v35/att/api_perf_register/perf_param_v2.cpp` `kVfInstructPerfTable`.
- Current perf implementation: `autofuse/v35/att/api_perf_register/ascendc_regbase_perf.cpp`
  `IsInfPerf` (finish `FinishUnaryBitWidthChangePerf`).
- ATT registration entry: `autofuse/v35/att/api_perf_register/ascir_api_perf_v2.cpp` `IsInfApi`
  (`kIsInf` -> `IsInfV2`).

## Parameter Sources and Chain

`UnaryBitWidthChangeNodeParams` (`autofuse/common/ascir_node_param/ascir_node_param.h:114`)
fields:

| Field | codegen source | Description |
| --- | --- | --- |
| `valid` | set by `FillUnaryBitWidthChangeNodeParams` | when false, cal_count is the total input element count and call_count=1 |
| `cal_count` | `param.cal_count` | innermost per-call element count |
| `outer_repeats` | `merge_repeats[0, size-1)` (last one removed) | call_count = Π outer_repeats |
| `output_strides` / `input_strides` | corresponding strides | participate in the model but do not affect current instruction counting |

Chain: `FillUnaryBitWidthChangeNodeParams` (written into `AscirNodeParams::specific_params`) ->
`FillSpecificParams` (`autofuse/att/gen_model_info/parser/specific_params_builder.cpp:65`
extracts into `NodeInfo`) -> `IsInfApi` -> `IsInfPerf`.

## Shared Modeling Quantities

- `repeat_elm = kRptEleMap[input_dtype]` (fp16=128, fp32=64, default `kRptSizeFloat=64`).
- `repeat_time = Ceil(cal_count / repeat_elm)`.
- `call_count = Π outer_repeats` (1 when params are invalid).
- **Output dtype is not distinguished**: the codegen `ReinterpretCast<bool>` makes the output
  always bool, and MaskPack always executes (`IsInfPerf` has no `output_is_bool` branch).

## Instruction Sequence and Branch Analysis

IsInf-specific sequence (rt = repeat_time, does not reuse
`AddUnaryBitWidthChangeCommonPerf`):

- `kLoad(input_dtype) x rt` ((0,0), not counted);
- `kCompareScalarEQ(input_dtype) x 2rt` ((3,1), comparing +Inf and -Inf respectively);
- `kMaskPack(uint8) x 2 calls` ((8,1), packing cmpMaskReg and mask; fp32 2rt each, fp16 rt
  each);
- `kStore(kFloat32) x rt` ((0,1)).

Per-branch formulas (rt = repeat_time, cc = call_count; calibrated latency constant 28; the
measurement comment says "Actual 7 repeat: Duplicate(2), UpdateMask(1 repeat), Select(1 repeat),
MaskOr(1 repeat) are not recorded"):

| Branch (input->output) | repeat_elm | Throughput sum | Formula |
| --- | --- | --- | --- |
| float16 -> bool | 128 | 5·rt (CompareScalarEQ 2rt + MaskPack 2rt + Store rt) | (48+5·rt)·cc |
| float32 -> bool | 64 | 7·rt (CompareScalarEQ 2rt + MaskPack 4rt + Store rt) | (48+7·rt)·cc |

Differences from IsNan/IsFinite: IsInf has no `kNe`/`kMaskOr` common segment and uses
`kCompareScalarEQ x 2rt` instead (two scalar comparisons for +Inf/-Inf); MaskPack has no
`output_is_bool` condition and always executes.

## MicroAPI Counting

Key performance table entries (latency, throughput):

| perf type | latency | throughput | remarks |
| --- | --- | --- | --- |
| kLoad | 0 | 0 | |
| kCompareScalarEQ (uint8~fp32/bf16) | 3 | 1 | once each for +Inf and -Inf, x 2rt |
| kStore (kFloat32) | 0 | 1 | |
| kMaskPack (uint8) | 8 | 1 | x 2 calls; fp32 2rt each, fp16 rt each |

## Formula

Final cost structure (calibrated constant `kIsInfMaxLatency = 28` replaces the accumulated
max_latency, head = 20):

```cpp
// kLoad / kCompareScalarEQ / kMaskPack / kStore accumulate all_vf_instruct_cost
// (throughput x count) via AddVfInstructPerf; the accumulated max_latency is discarded and
// replaced by the calibrated constant 28;
Expr res = (VfPerfUtils::GetVFHeadCost() + CreateExpr(28) + all_vf_instruct_cost) * call_count;
res.Simplify();
perf.pipe_res[PipeType::AIV_VEC] = res;
```
