# IsFinite V2 Performance Model

## Scope

- Targeting the v35 (V2) ASCIR `IsFinite` node, corresponding to the API `IsFinite` (header
  `adv_api/math/is_finite.h`, codegen generates calls via `UnaryBitWidthChangeApiCallV2`).
- Covers fp16/fp32 input x bool/uint8 output combinations; bf16 input is handled as fp32 after
  the dtype conversion (BF16->FLOAT) (`GetConversionDtype`).
- `IsNan` / `IsFinite` / `IsInf` share the `UnaryBitWidthChange` framework; this archive covers
  `IsFinite` only.
- Output pipe is `AIV_VEC`; the copy overhead between Reg and UB is not modeled.

## Source References

- codegen impl: `autofuse/v35/ascir/generator/v2_ascir_codegen_impl.h`
  `IsFiniteAscIrCodegenImplV2` (`GetApiName()=IsFinite`,
  `GetApiCallName()=UnaryBitWidthChangeApiCallV2`, output via `ReinterpretCast<bool>`).
- API source: `adv_api/math/is_finite.h` (asc-devkit, external dependency).
- API call generation: `autofuse/v35/codegen/reg_api_call/unary_bitwidth_change_api_call_v2.cpp`
  `UnaryBitWidthChangeApiCallV2::Generate` / `FillUnaryBitWidthChangeNodeParams`.
- Performance table: `autofuse/v35/att/api_perf_register/perf_param_v2.cpp` `kVfInstructPerfTable`.
- Current perf implementation: `autofuse/v35/att/api_perf_register/ascendc_regbase_perf.cpp`
  `IsFinitePerf` (common segment `AddUnaryBitWidthChangeCommonPerf` / finish
  `FinishUnaryBitWidthChangePerf`).
- ATT registration entry: `autofuse/v35/att/api_perf_register/ascir_api_perf_v2.cpp`
  `IsFiniteApi` (`kIsFinite` -> `IsFiniteV2`).

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
extracts into `NodeInfo`) -> `IsFiniteApi` -> `IsFinitePerf`.

## Shared Modeling Quantities

- `repeat_elm = kRptEleMap[input_dtype]` (fp16=128, fp32=64, default `kRptSizeFloat=64`).
- `repeat_time = Ceil(cal_count / repeat_elm)`.
- `call_count = Π outer_repeats` (1 when params are invalid).
- `output_is_bool = (output_dtype == bool)`; the codegen output is always bool
  (`ReinterpretCast<bool>`); the uint8 output is only supported on the SIMT scalar path
  (`IsSimtScalarSupported`).

## Instruction Sequence and Branch Analysis

Common segment (`AddUnaryBitWidthChangeCommonPerf`, rt = repeat_time):

- `kLoad(input_dtype) x rt` ((0,0), not counted);
- `kNe(uint8) x rt` ((3,1));
- `kStore(kFloat32) x rt` ((0,1)).

IsFinite-specific segment:

- `kMaskOr(input_dtype) x rt` ((4,2) -> throughput 2rt, bitwise-OR of the NaN and Inf verdicts);
- when `output_is_bool`, two MaskPack calls are appended (packing cmpMaskReg and mask):
  - fp32 input: `kMaskPack(uint8) x 2rt` twice (fp32 masks need packing twice);
  - fp16 input: `kMaskPack(uint8) x rt` twice.

Per-branch formulas (rt = repeat_time, cc = call_count; calibrated latency constant 16; the
measurement comment says "IsFinite: Actual 8 repeat, Duplicate(2), UpdateMask(1 repeat),
Select(1 repeat), CompareScalarEQ(2 repeat) are not recorded"):

| Branch (input->output) | repeat_elm | Throughput sum | Formula |
| --- | --- | --- | --- |
| float16 -> bool | 128 | 6·rt (Ne rt + Store rt + MaskOr 2rt + MaskPack 2rt) | (36+6·rt)·cc |
| float32 -> bool | 64 | 8·rt (Ne rt + Store rt + MaskOr 2rt + MaskPack 4rt) | (36+8·rt)·cc |
| float16 -> uint8 | 128 | 4·rt (Ne rt + Store rt + MaskOr 2rt) | (36+4·rt)·cc |
| float32 -> uint8 | 64 | 4·rt (Ne rt + Store rt + MaskOr 2rt) | (36+4·rt)·cc |

## MicroAPI Counting

Key performance table entries (latency, throughput):

| perf type | latency | throughput | remarks |
| --- | --- | --- | --- |
| kLoad | 0 | 0 | |
| kNe (uint8) | 3 | 1 | |
| kMaskOr (uint8/i8/u16/i16/u32/i32/fp16/fp32/bf16) | 4 | 2 | IsFinite specific |
| kStore (kFloat32) | 0 | 1 | |
| kMaskPack (uint8) | 8 | 1 | 2 calls for bool output; fp32 2rt each, fp16 rt each |

## Formula

Final cost structure (calibrated constant `kIsFiniteMaxLatency = 16` replaces the accumulated
max_latency, head = 20):

```cpp
// AddUnaryBitWidthChangeCommonPerf / kMaskOr / MaskPack accumulate all_vf_instruct_cost
// (throughput x count) via AddVfInstructPerf; the accumulated max_latency is discarded and
// replaced by the calibrated constant 16;
Expr res = (VfPerfUtils::GetVFHeadCost() + CreateExpr(16) + all_vf_instruct_cost) * call_count;
res.Simplify();
perf.pipe_res[PipeType::AIV_VEC] = res;
```
