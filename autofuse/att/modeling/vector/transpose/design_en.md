# Transpose V2 Performance Model

## Scope

- Targeting the v35 (V2) ASCIR `Transpose` node, corresponding to the regbase API `TransposeExtend`.
- Covers the six explicit template branches (`total_dim <= 4`, `inner_dim <= 3`) plus the generic
  branch (`(2,2)` / `(3,3)` fall back to the generic index-generation + generic extend path).
- `valid = false` or missing `TransposeNodeParams` falls back to modeling by the total input
  element count.
- Output pipe is `AIV_VEC`; the copy overhead between Reg and UB is not modeled.

## Source References

- codegen impl: `autofuse/v35/ascir/generator/v2_ascir_codegen_impl.h` `TransposeAscIrCodegenImplV2`
  (`GetApiCallName()=TransposeV2ApiCall`).
- API source (regbase): `autofuse/v35/ascendc/api_regbase/transpose.h`.
- API call generation: `autofuse/v35/codegen/reg_api_call/transpose_v2_api_call.cpp`
  `TransposeV2ApiCall::Generate` / `FillTransposeNodeParams`.
- Performance table: `autofuse/v35/att/api_perf_register/perf_param_v2.cpp` `kVfInstructPerfTable`.
- Current perf implementation: `autofuse/v35/att/api_perf_register/ascendc_regbase_perf.cpp` `TransposePerf`.
- ATT registration entry: `autofuse/v35/att/api_perf_register/ascir_api_perf_v2.cpp`
  (`kTranspose` -> `TransposeV2`).

## Parameter Sources and Chain

`TransposeNodeParams` (`autofuse/common/ascir_node_param/ascir_node_param.h:122`) fields:

| Field | codegen source | Description |
| --- | --- | --- |
| `valid` | set by `FillTransposeNodeParams` | when false, ATT builds params from input dims |
| `total_dim` / `inner_dim` | vectorized axis merge result | number of template axes / trailing continuous inner axes |
| `outer_loop_axes` | product of outer repeats (outside the template) | multiplies the whole cost |
| `output_dims` | vectorized output dims | inner axes product = cal_count; template outer dims drive movement_count |
| `input_strides` / `output_strides` | corresponding strides | the innermost input stride decides stride_count |

Chain: `BuildApiParam` -> `FillTransposeNodeParams` (written into `AscirNodeParams::specific_params`)
-> `FillSpecificParams` (`autofuse/att/gen_model_info/parser/specific_params_builder.cpp` extracts
into `NodeInfo`) -> `TransposeApi` -> `TransposePerf`.

## Shared Modeling Quantities

- `repeat_elm = kRptEleMap[input_dtype]` (fp16 = 128; fall back to `kRptSizeFloat = 64` when absent).
- `repeat_time = Ceil(cal_count / repeat_elm)`, where `cal_count` is the product of the
  `inner_dim` inner axes of `output_dims`.
- `tail_repeat = Ceil(Max(cal_count - repeat_elm, 0) / repeat_elm)`.
- `alignment = kRptSizeHalf / data_size` (fp16 = 64).
- `stride_count = Mod(input_strides.back(), alignment)`; when 0, take `alignment`.
- `gather_count`: `inner_dim ∈ {1, 2}` -> `2 * stride_count`; `inner_dim = 3` -> `stride_count + 1`.
- `outer_count = Π outer_loop_axes`.
- `GetVFHeadCost() = 20`.

## Branch Analysis

Six explicit branches plus the generic branch. `rt` = repeat_time, `tr` = tail_repeat:

| Branch (total_dim, inner_dim) | Index gen | Movement | Latency |
| --- | --- | --- | --- |
| (2,1) | GenOne | OneOuter | 69 |
| (3,1) | GenOne | TwoOuter | 69 |
| (3,2) | GenTwo | OneOuter | 167 |
| (4,1) | GenOne | ThreeOuter | 69 |
| (4,2) | GenTwo | TwoOuter | 167 |
| (4,3) | GenThree | OneOuter | 386 |
| generic | by inner_dim | generic extend | per inner_dim |

- GenOne instruction count: kPlaceholder x1, kMuls x rt, kAdds x rt, kPlaceholder x rt.
- GenTwo: init kArange / kDuplicate / kCopy / kDiv / kMul / kSub / kMuls x2 / kAdd / kStore x1,
  tail x tr: kUpdateMask / kAdds / kCompareScalarGE / kSelect / kMuls / kSub / kAdds / kAdd /
  kMuls x2 / kStore.
- GenThree: init kPlaceholder / kDuplicate x2 / kCopy x2 / kDiv x3 / kMul x3 / kPlaceholder,
  tail x tr: kUpdateMask / kSelect x2 / kMuls x5 / kAdd x4 / kMuls x3 / kPlaceholder.
- Movement: kLoad x rt (0-cost), kGather2 x movement_count * gather_count,
  kStore(kFloat32) x movement_count.

## MicroAPI Counting

Only the throughput of each instruction participates in the final result (the accumulated
max_latency is discarded; the calibrated branch latency constant is used instead):

| perf type | latency | throughput | remarks |
| --- | --- | --- | --- |
| kLoad | 0 | 0 | DataCopy load |
| kGather2 | 0 | 1 | |
| kStore (kFloat32) | 0 | 1 | movement store |
| kPlaceholder | 0 | 0 | Arange / store-index placeholder |
| kMuls | 5 | 1 | |
| kAdds | 4 | 2 | |
| kAdd / kSub | 4 | 1 | |
| kMul | 5 | 1 | |
| kDiv (kFloat32) | 14 | 4 | GenTwo/GenThree init |
| kDuplicate | 8 | 1 | |
| kArange (kInt32) | 4 | 2 | |
| kCompareScalarGE / kSelect / kUpdateMask | 3 | 1 | |
| kCopy / kUnitVector | no entry | 0 | counted as 0 |

## Formula

Final cost structure (calibrated latency constants 69 / 167 / 386; head 20):

```cpp
// Index generation + movement: each MicroAPI accumulates all_vf_instruct_cost
// (throughput x count) via AddVfInstructPerf; the max_latency accumulation is discarded.
Expr res = (GetVFHeadCost() + CreateExpr(latency) + all_vf_instruct_cost) * outer_count;
res.Simplify();
perf.pipe_res[PipeType::AIV_VEC] = res;
```
