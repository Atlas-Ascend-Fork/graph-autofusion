# Compare (GE/EQ/NE/GT/LE/LT) V2 Performance Model

## Scope

- Targeting the v35 (V2) ASCIR Compare node, covering the six modes Eq/Ne/Gt/Ge/Lt/Le,
  corresponding to the regbase APIs `GE`/`EQ`/`NE`/`GT`/`LE`/`LT` (`CompareExtend` /
  `CompareScalarExtend` in `compare.h`).
- Covers 8/16/32/64-bit input dtypes and the two input forms tensor/tensor and tensor/scalar
  (is_scalar).
- bool input is executed as uint8 at the lower level and modeled uniformly as uint8.
- Output pipe is `AIV_VEC`; the copy overhead between Reg and UB is not modeled.

## Source References

- codegen impl: `autofuse/v35/ascir/generator/v2_ascir_codegen_impl.h` `CompareAscIrCodegenImplV2`
  (`GetApiCallName()=CompareV2ApiCall`).
- API source (regbase): `autofuse/v35/ascendc/api_regbase/compare.h`.
- API call generation: `autofuse/v35/codegen/reg_api_call/compare_v2_api_call.cpp`
  `CompareV2ApiCall::Generate` / `BuildCompareLoopParams` / `FillCompareNodeParams`.
- Performance table: `autofuse/v35/att/api_perf_register/perf_param_v2.cpp` `kVfInstructPerfTable`
  and `kVfInstructDtypeMappingPerfTable`.
- Current perf implementation: `autofuse/v35/att/api_perf_register/ascendc_regbase_perf.cpp`
  `CompareSpecificPerf` (shared by the six entries `CompareGEPerf` / `CompareEQPerf` /
  `CompareNEPerf` / `CompareGTPerf` / `CompareLEPerf` / `CompareLTPerf`).
- ATT registration entry: `autofuse/v35/att/api_perf_register/ascir_api_perf_v2.cpp`
  `CompareGeApi` etc. (`kGe`/`kEq`/`kNe`/`kGt`/`kLe`/`kLt`).

## Parameter Sources and Chain

`CompareNodeParams` (`autofuse/common/ascir_node_param/ascir_node_param.h:94`) fields:

| Field | codegen source | Description |
| --- | --- | --- |
| `valid` | set by `FillCompareNodeParams` | when false, `output_dims` falls back to `node_info.input_dims` |
| `is_scalar` | `x2.IsAnyScalar()` | scalar input saves one Load and adds one Duplicate |
| `outer_call_count` | `Π merge_repeats[0, size-1)` | multiplier of the outermost loop over the whole |
| `output_dims` | `[merge_repeats.back(), cal_count]` (dim=2) or `[merge_repeats.back()]` (dim=1) | `front()` is the innermost outer axis, `back()` is the per-call count |
| `output_strides` / `input_strides` | corresponding strides | participate in the model but do not affect current instruction counting |

Chain: `BuildCompareLoopParams` -> `FillCompareNodeParams` (written into
`AscirNodeParams::specific_params`) -> `FillSpecificParams`
(`autofuse/att/gen_model_info/parser/specific_params_builder.cpp` extracts into `NodeInfo`) ->
`CompareXxApi` -> `CompareSpecificPerf`.

## Shared Modeling Quantities

- `input_dtype`: bool is mapped to uint8.
- `repeat_elm = kRptEleMap[input_dtype]` (fp16=128, fp32=64, u8/i8=256, i64/u64=32).
- `vl_size`: 64-bit is `repeat_elm x 2` (=64, corresponding to `vlSize = 2 * GetVecLen() /
  sizeof(EffT)` when `sizeof(EffT)==8` in `compare.h`), otherwise `repeat_elm`.
- `block_count = output_dims.front() (when dim>1) x Ceil(output_dims.back() / vl_size)`.
- `compare_vf_head_cost`: u8/i8 is 30; fp16/bf16/u16/i16 is 44; u64/i64 is 51; the rest
  (fp32/u32/i32) is 49.
- The cases in this archive take the 1-D scenario (`output_dims = [cal_count]`):
  `bc = Ceil(cal_count / vl_size)`, `cc = outer_loop`.

## Instruction Sequence and Branch Analysis

Fixed header (independent of block_count):

- `kPlaceholder(u8) x1` (CreateMask, no table entry, counted as 0);
- `kDuplicate(u8) x2` (all-ones / all-zeros registers);
- an extra `kDuplicate(input_dtype) x1` when is_scalar;
- `kUpdateMask(u8) x2` (mainBlockCount + counterTail).

Loop body (x block_count):

- `kLoad(input) x bc` (two paths for tensor / one for scalar, both (0,0), not counted);
- `k{mode}(input) x bc` (Eq/Ne/Gt/Ge/Lt/Le all have table entry (3,1);
  **the current model does not distinguish mode**);
- `kMaskPack(u8) x bc` (16-bit) / `x 2bc` (32/64-bit) / not executed (8-bit);
- `kSelect(u8) x bc`;
- `kStore(input->uint8) dtype-mapping x bc` (default entry (0,1)).

`max_latency = 8` (kDuplicate(u8) / kMaskPack). Per-branch formulas (bc = block_count,
cc = outer_call_count):

| Branch | vl_size | head | tensor formula | scalar formula |
| --- | --- | --- | --- | --- |
| 8-bit (int8/uint8/bool) | 256 | 30 | (42+3·bc)·cc | (43+3·bc)·cc |
| 16-bit (fp16/bf16/int16/uint16) | 128 | 44 | (56+4·bc)·cc | (57+4·bc)·cc |
| 32-bit (fp32/int32/uint32) | 64 | 49 | (61+5·bc)·cc | (62+5·bc)·cc |
| 64-bit (int64/uint64) | 64 | 51 | (63+5·bc)·cc | not covered |

## MicroAPI Counting

Key performance table entries (latency, throughput):

| perf type | latency | throughput | remarks |
| --- | --- | --- | --- |
| kLoad | 0 | 0 | DataCopy load src0/src1 |
| kDuplicate (uint8 / input) | 8 | 1 | header x2, scalar adds x1 |
| kUpdateMask | 3 | 1 | header x2 |
| kEq / kNe / kLt / kGt / kGe / kLe | 3 | 1 | mode not distinguished |
| kMaskPack (uint8) | 8 | 1 | 16-bit x bc, 32/64-bit x 2bc |
| kSelect (uint8) | 3 | 1 | |
| kStore (dtype-mapping default entry) | 0 | 1 | input->uint8 |
| kPlaceholder | 0 | 0 | CreateMask placeholder |

## Formula

Final cost structure (`compare_vf_head_cost` replaces the generic `GetVFHeadCost()`):

```cpp
// Header + loop body MicroAPIs accumulate all_vf_instruct_cost (throughput x count)
// via AddVfInstructPerf / AddVfInstructDtypeMappingPerf, and take max_latency;
Expr res = compare_vf_head_cost + max_latency + all_vf_instruct_cost;
res = res * node_info.compare_node_params.outer_call_count;
res.Simplify();
perf.pipe_res[PipeType::AIV_VEC] = res;
```
