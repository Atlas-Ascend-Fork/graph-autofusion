# Cast (CastExtend) V2 Performance Model

## Scope

- Targeting the v35 (V2) ASCIR `Cast` node, corresponding to the regbase API `CastExtend`;
  APIs such as `Ceil2Int` reuse the same model.
- Covers 7 typical branches: generic float conversions (fp16<->fp32), same-bit integer
  conversions (int32<->uint32), two-step conversion (int32->fp16), float32->bool, B64 narrowing
  (int64->fp32), B64 transfer (int64->fp16).
- B4 (fp16<->int4) and B8 specific branches (uint8->fp32 etc.) are not covered in this archive;
  see Limitations.
- Output pipe is `AIV_VEC`; the copy overhead between Reg and UB is not modeled.

## Source References

- codegen impl: `autofuse/v35/ascir/generator/v2_ascir_codegen_impl.h` `CastAscIrCodegenImplV2`
  (`GetApiName()=CastExtend`, `GetApiCallName()=CastV2ApiCall`).
- API source (regbase): `autofuse/v35/ascendc/api_regbase/cast.h`.
- API call generation: `autofuse/v35/codegen/reg_api_call/cast_v2_api_call.cpp`
  `CastV2ApiCall::BuildApiParam` / `FillCastNodeParams`.
- Performance table: `autofuse/v35/att/api_perf_register/perf_param_v2.cpp` `kVfInstructPerfTable`
  and `kVfInstructDtypeMappingPerfTable`.
- Current perf implementation: `autofuse/v35/att/api_perf_register/ascendc_regbase_perf.cpp` `CastPerf`.
- ATT registration entry: `autofuse/v35/att/api_perf_register/ascir_api_perf_v2.cpp` `CastApi`
  (`kCast` -> `CastV2`).

## Parameter Sources and Chain

`CastNodeParams` (`autofuse/common/ascir_node_param/ascir_node_param.h:77`) fields:

| Field | codegen source | Description |
| --- | --- | --- |
| `valid` | set by `BuildApiParam` | when false, model by total input element count |
| `output_dims` | output vectorized axes | `back()` is the innermost count; `front()` is api_outer |
| `output_strides` | output vectorized strides | decides contiguous vs strided together with count/input_strides |
| `input_strides` | input vectorized strides | ditto |

Chain: `BuildApiParam` -> `FillCastNodeParams` (written into `AscirNodeParams::specific_params`)
-> `FillSpecificParams` (`autofuse/att/gen_model_info/parser/specific_params_builder.cpp:37`
extracts into `NodeInfo`) -> `CastApi` -> `CastPerf`.

## Shared Modeling Quantities

- `one_rep_size = min(kRptEleMap[input_dtype], kRptEleMap[output_dtype])`, defaulting to
  `kRptSizeFloat = 64`; special case `uint8 -> int4` takes `kRptEleMap[float16] = 128`.
  (fp16<->fp32, int32<->uint32, int32->fp16, fp32->bool are 64; int64->fp32, int64->fp16 are 32.)
- `repeat_time`:
  - same-bit integer cast: `Ceil(output_dims.back() / one_rep_size)`, or by total input element
    count when params are invalid;
  - non-same-bit: 1-D `Ceil(count / one_rep_size)`; multi-D via the ternary
    `output_strides.front() == input_strides.front()` and `== count` (inner contiguous) taking
    `Ceil(api_outer * count / one_rep_size)`, otherwise `Ceil(count / one_rep_size)`.
- `call_count`: same-bit takes `output_dims.front()` when `output_dims.size() > 1`, else 1;
  non-same-bit via ternary (1 when inner contiguous, `output_dims.front()` when strided).
- The cases in this archive uniformly take the strided scenario: `rt = Ceil(count / one_rep_size)`,
  `cc = outer_loop`.

## Branch Analysis

Each branch consists of a DataCopy (Load+Store) segment and a compute segment;
`rt` = repeat_time, `cc` = call_count:

| Branch (input->output) | Instruction sequence (throughput count) | max_latency | Formula |
| --- | --- | --- | --- |
| fp16<->fp32 / fp16<->bf16 / fp32<->bf16 (generic float) | Load x rt(0) + Store x rt(1) + Cast x rt(1) | 4 | (24+2·rt)·cc |
| int32<->uint32 and other same-bit integers | UpdateMask x 2(2) + Load x rt(0) + Store x rt(1) | 3 | (25+rt)·cc |
| int32->float16 (two-step) | Load x rt(0) + Store x rt(1) + Cast(int32->fp32) x rt(2) + Cast(fp32->fp16) x rt(1) | 5 | (25+4·rt)·cc |
| float32->bool | Load x rt(0) + Store x rt(1) + Duplicate x 1(1) + And x rt + Adds x rt(2) + Or x rt + Not x rt + Vshrs x rt | 8 | (29+7·rt)·cc |
| int64->float32 (B64 narrowing) | Load x rt(0) + Store x rt(1) + Cast x rt(2) + Duplicate x rt(1) + DeInterleave x rt(1) | 8 | (28+5·rt)·cc |
| int64->float16 (B64 transfer) | Load x rt(0) + Store x rt(1) + Duplicate x 1(1) + MaskPack x rt(0, no table entry for u32) + DeInterleave x rt(1) + Cast(in->fp32) x rt(2) + Cast(fp32->fp16) x rt(1) | 8 | (29+5·rt)·cc |
| B4 / B8 specific branches | see `AddB4CastPerf` / `AddB8SpecificCastPerf` | - | not covered |

## MicroAPI Counting

Key performance table entries (latency, throughput):

| perf type | latency | throughput | remarks |
| --- | --- | --- | --- |
| kLoad | 0 | 0 | DataCopy load |
| kStore (dtype-mapping default entry) | 0 | 1 | DataCopy store; only the uint8->int64 special case is 5 |
| kCast (fp16/bf16/fp32 conversions) | 4 | 1 | dtype-mapping table |
| kCast (default entry, incl. integers) | 5 | 2 | dtype-mapping table |
| kUpdateMask | 3 | 1 | same-bit cast header x2 |
| kDuplicate (uint32) | 8 | 1 | |
| kAnd / kOr / kNot / kVshrs (uint32) | 3 | 1 | |
| kAdds (uint32) | 4 | 2 | |
| kDeInterleave / kInterleave (uint32) | 8 | 1 | |
| kMaskPack | 8 | 1 | only u8/i8 have a table entry; u32 calls count as 0 |

## Formula

Final cost structure (unlike branches with calibrated constants, `max_latency` participates in
the sum):

```cpp
// Load/Store/Compute MicroAPIs accumulate all_vf_instruct_cost (throughput x count)
// via AddVfInstructPerf / AddVfInstructDtypeMappingPerf, and take max_latency;
Expr res = (VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost) * call_count;
res.Simplify();
perf.pipe_res[PipeType::AIV_VEC] = res;
```

where `GetVFHeadCost() = 20`. When `valid = false` or params are missing, `repeat_time` is
derived from the total input element count and `call_count = 1`.
