# Where (WhereExtend) V2 Performance Model

## Scope

- Targeting the v35 (V2) ASCIR `Where` / `Select` node, corresponding to the regbase API
  `WhereExtend` (`where.h`).
- Covers the two loop structures 1-D (`output_dims = [cal]`) and 2-D
  (`output_dims = [counter_first, cal]`), as well as the four data bit-widths B8/B16/B32/B64.
- Falls back to the legacy model when `where_node_params.valid = false` (not covered in this
  archive).
- Output pipe is `AIV_VEC`; the copy overhead between Reg and UB is not modeled.

## Source References

- codegen impl: `autofuse/v35/ascir/generator/v2_ascir_codegen_impl.h` `WhereAscIrCodegenImplV2`
  (`GetApiCallName()=WhereRegApiCall`, `GetApiName()=WhereExtend`).
- API source (regbase): `autofuse/v35/ascendc/api_regbase/where.h`.
- API call generation: `autofuse/v35/codegen/reg_api_call/reg_where_api_call.cpp`
  `BuildWhereLoopParams` / `FillWhereNodeParams` (`is_bcast_src0/src1` are taken from
  `x2/x3.IsAnyScalar()`).
- Performance table: `autofuse/v35/att/api_perf_register/perf_param_v2.cpp` `kVfInstructPerfTable`.
- Current perf implementation: `autofuse/v35/att/api_perf_register/ascendc_regbase_perf.cpp` `WherePerf`.
- ATT registration entry: `autofuse/v35/att/api_perf_register/ascir_api_perf_v2.cpp` `WhereApi`
  (`kWhere` -> `WhereV2`; `kSelect` is also mapped to `WhereV2`).

## Parameter Sources and Chain

`WhereNodeParams` (`autofuse/common/ascir_node_param/ascir_node_param.h:103`) fields:

| Field | codegen source | Description |
| --- | --- | --- |
| `valid` | set by `FillWhereNodeParams` | when false, ATT takes the legacy model |
| `is_bcast_src0` / `is_bcast_src1` | `IsAnyScalar()` of `x2` / `x3` | only removes one kLoad (table entry (0,0)); **does not affect predictions** |
| `outer_call_count` | `Π merge_repeats[0, size-1)` | outermost loop multiplier; always 1 for 1-D (empty outer_repeats) |
| `output_dims` | `[cal]` (1-D) or `[merge_repeats.back(), cal]` (2-D) | `front()` is counter_first, `back()` is the per-call count |
| `output_strides` / `mask_strides` / `input_strides` | corresponding strides | participate in the model but do not affect current instruction counting |

Chain: `BuildWhereLoopParams` -> `FillWhereNodeParams` (written into
`AscirNodeParams::specific_params`) -> `FillSpecificParams`
(`autofuse/att/gen_model_info/parser/specific_params_builder.cpp:58` extracts into `NodeInfo`) ->
`WhereApi` -> `WherePerf`.

## Shared Modeling Quantities

- `data_dtype`: the output dtype (`GetWhereDataDtype`).
- `repeat_elm = kRptEleMap[data_dtype]`, x2 again for B64 (int64/uint64) (=64);
  fp16=128, fp32=64, u8/i8=256.
- `repeat_time = Ceil(count / repeat_elm)`: 1-D takes `count = output_dims.front()`;
  2-D takes `count = output_dims.back()` (counter_last).
- `where_vf_head_cost`: B8 (u8/i8) = 43, B16 (fp16/bf16/i16/u16) = 45,
  B32 (fp32/i32/u32) = 49, B64 (i64/u64) = 57.
- 2-D row overhead (only the B16/B32 per-row path): B16=4, B32=3.
- Select count x2 for B64 output (`kWhereSelectOutputFactorB64`).

## Instruction Sequence and Branch Analysis

`AddWhereComputePerf(data_dtype, block_count=bc)` (used by the 2-D path):

- `kPlaceholder(u8) x1` (CreateMask, no table entry, counted as 0);
- when not B16/B32 (i.e. B8 and B64): `kUpdateMask(i32) x bc` + `kCompareScalarNE(u8) x bc`
  (each (3,1));
- `kLoad(u8) x bc` (mask load, (0,0));
- `kMaskUnPack(u8)`: B16 x bc, B32/B64 x 2bc, not executed for B8 ((8,1));
- non-bcast src0/src1 each `kLoad(dtype) x bc` ((0,0), so the bcast flags do not affect
  predictions);
- `kSelect(i32) x bc x output_factor` ((3,1), B64 output factor=2);
- `kStore(i32) x bc` ((0,1)).

The `AddWhereImplPerf` (1-D path) instruction sequence is identical to the above, except that
the trigger condition for UpdateMask/CompareScalarNE is "not B8/B16/B32" (i.e. only B64 executes
them; B8 does not in 1-D).

Path dispatch:

- **1-D** (`AddWhereImplPerf`): `rt = Ceil(cal / repeat_elm)`, counted with `bc = rt` per the
  table above, `max_latency` accumulated normally (B16/B32/B64 = 8, B8 = 3).
- **2-D B8/B64** (generic path of `AddWhereExtend2DPerf`): `bc = counter_first x rt`, a single
  `AddWhereComputePerf`, `max_latency` accumulated normally.
- **2-D B16/B32** (per-row path): after calling `AddWhereComputePerf(bc = rt)` per row, only
  `row_vf_instruct_cost` is taken, **`row_max_latency` is discarded (max_latency=0)**, and the
  row overhead (B16=4 / B32=3) is added before multiplying by counter_first.

Per-branch formulas (rt = repeat_time, cf = counter_first, cc = outer_call_count):

| Branch | repeat_elm | 1-D formula | 2-D formula |
| --- | --- | --- | --- |
| int8/uint8 (B8) | 256 | (46+2·rt)·cc | (46+4·cf·rt)·cc |
| float16/bf16/int16/uint16 (B16) | 128 | (53+3·rt)·cc | (45+(3·rt+4)·cf)·cc |
| float32/int32/uint32 (B32) | 64 | (57+4·rt)·cc | (49+(4·rt+3)·cf)·cc |
| int64/uint64 (B64) | 64 | (65+7·rt)·cc | (65+7·cf·rt)·cc |

## MicroAPI Counting

Key performance table entries (latency, throughput):

| perf type | latency | throughput | remarks |
| --- | --- | --- | --- |
| kLoad | 0 | 0 | mask / src0 / src1 load |
| kMaskUnPack (uint8) | 8 | 1 | B16 x bc, B32/B64 x 2bc |
| kUpdateMask | 3 | 1 | only B64 (and 2-D B8), x bc |
| kCompareScalarNE | 3 | 1 | only B64 (and 2-D B8), x bc |
| kSelect (int32) | 3 | 1 | B64 output x 2bc |
| kStore (int32) | 0 | 1 | |
| kPlaceholder | 0 | 0 | CreateMask placeholder |

## Formula

Final cost structure (`where_vf_head_cost` replaces the generic `GetVFHeadCost()`):

```cpp
// Each MicroAPI accumulates all_vf_instruct_cost (throughput x count) via AddVfInstructPerf
// and takes max_latency;
Expr res = (where_vf_head_cost + max_latency + all_vf_instruct_cost) * outer_call_count;
res.Simplify();
perf.pipe_res[PipeType::AIV_VEC] = res;
```

The legacy path when `valid = false`: `kDuplicate x n + kCompareScalarNE x rt + kSelect x rt`,
head takes the generic `GetVFHeadCost()=20`, not multiplied by outer_call_count.
