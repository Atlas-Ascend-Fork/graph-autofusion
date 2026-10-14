# Compare（GE/EQ/NE/GT/LE/LT）V2 性能模型

## 适用范围

- 面向 v35 形态（V2）ASCIR Compare 节点，覆盖 Eq/Ne/Gt/Ge/Lt/Le 六种 mode，对应 regbase API
  `GE`/`EQ`/`NE`/`GT`/`LE`/`LT`（`compare.h` 的 `CompareExtend` / `CompareScalarExtend`）。
- 覆盖 8/16/32/64 bit 输入 dtype 与 tensor/tensor、tensor/scalar（is_scalar）两类输入形态。
- bool 输入底层按 uint8 执行，统一按 uint8 建模。
- 输出 Pipe 为 `AIV_VEC`；不建模 Reg 与 UB 间的搬运开销。

## 源码依据

- codegen impl：`autofuse/v35/ascir/generator/v2_ascir_codegen_impl.h` `CompareAscIrCodegenImplV2`
  （`GetApiCallName()=CompareV2ApiCall`）。
- API 源码（regbase）：`autofuse/v35/ascendc/api_regbase/compare.h`。
- API 调用生成：`autofuse/v35/codegen/reg_api_call/compare_v2_api_call.cpp`
  `CompareV2ApiCall::Generate` / `BuildCompareLoopParams` / `FillCompareNodeParams`。
- 性能表：`autofuse/v35/att/api_perf_register/perf_param_v2.cpp` `kVfInstructPerfTable` 与
  `kVfInstructDtypeMappingPerfTable`。
- 当前 perf 实现：`autofuse/v35/att/api_perf_register/ascendc_regbase_perf.cpp` `CompareSpecificPerf`
  （`CompareGEPerf` / `CompareEQPerf` / `CompareNEPerf` / `CompareGTPerf` / `CompareLEPerf` /
  `CompareLTPerf` 六个入口共用）。
- ATT 注册入口：`autofuse/v35/att/api_perf_register/ascir_api_perf_v2.cpp` `CompareGeApi` 等
  （`kGe`/`kEq`/`kNe`/`kGt`/`kLe`/`kLt`）。

## 参数来源与链路

`CompareNodeParams`（`autofuse/common/ascir_node_param/ascir_node_param.h:94`）字段与来源：

| 字段 | codegen 传参来源 | 说明 |
| --- | --- | --- |
| `valid` | `FillCompareNodeParams` 置位 | false 时 `output_dims` 回退 `node_info.input_dims` |
| `is_scalar` | `x2.IsAnyScalar()` | scalar 输入少一路 Load、多一次 Duplicate |
| `outer_call_count` | `Π merge_repeats[0, size-1)` | 最外层循环对整体的乘子 |
| `output_dims` | `[merge_repeats.back(), cal_count]`（dim=2）或 `[merge_repeats.back()]`（dim=1） | `front()` 为最内层外轴，`back()` 为单次处理 count |
| `output_strides` / `input_strides` | 对应 stride | 参与模型但不影响当前指令计数 |

链路：`BuildCompareLoopParams` -> `FillCompareNodeParams`（写入 `AscirNodeParams::specific_params`）->
`FillSpecificParams`（`autofuse/att/gen_model_info/parser/specific_params_builder.cpp` 提取到 `NodeInfo`）->
`CompareXxApi` -> `CompareSpecificPerf`。

## 公共建模量

- `input_dtype`：bool 映射为 uint8。
- `repeat_elm = kRptEleMap[input_dtype]`（fp16=128、fp32=64、u8/i8=256、i64/u64=32）。
- `vl_size`：64 bit 为 `repeat_elm × 2`（=64，对应 `compare.h` 中 `sizeof(EffT)==8` 时
  `vlSize = 2 * GetVecLen() / sizeof(EffT)`），否则为 `repeat_elm`。
- `block_count = output_dims.front()（dim>1 时）× Ceil(output_dims.back() / vl_size)`。
- `compare_vf_head_cost`：u8/i8 为 30；fp16/bf16/u16/i16 为 44；u64/i64 为 51；其余（fp32/u32/i32）为 49。
- 本归档用例取 1-D 场景（`output_dims = [cal_count]`）：`bc = Ceil(cal_count / vl_size)`，
  `cc = outer_loop`。

## 指令序列与分支分析

固定头部（与 block_count 无关）：

- `kPlaceholder(u8) ×1`（CreateMask，无表项计 0）；
- `kDuplicate(u8) ×2`（全 1 / 全 0 寄存器）；
- is_scalar 时追加 `kDuplicate(input_dtype) ×1`；
- `kUpdateMask(u8) ×2`（mainBlockCount + counterTail）。

循环体（× block_count）：

- `kLoad(input) ×bc`（tensor 两路 / scalar 一路，均 (0,0) 不计）；
- `k{mode}(input) ×bc`（Eq/Ne/Gt/Ge/Lt/Le 表项均为 (3,1)，**当前模型不区分 mode**）；
- `kMaskPack(u8) ×bc`（16 bit）/ `×2bc`（32/64 bit）/ 不执行（8 bit）；
- `kSelect(u8) ×bc`；
- `kStore(input→uint8) dtype-mapping ×bc`（默认项 (0,1)）。

`max_latency = 8`（kDuplicate(u8) / kMaskPack）。各分支公式（bc = block_count，cc = outer_call_count）：

| 分支 | vl_size | head | tensor 公式 | scalar 公式 |
| --- | --- | --- | --- | --- |
| 8 bit（int8/uint8/bool） | 256 | 30 | (42+3·bc)·cc | (43+3·bc)·cc |
| 16 bit（fp16/bf16/int16/uint16） | 128 | 44 | (56+4·bc)·cc | (57+4·bc)·cc |
| 32 bit（fp32/int32/uint32） | 64 | 49 | (61+5·bc)·cc | (62+5·bc)·cc |
| 64 bit（int64/uint64） | 64 | 51 | (63+5·bc)·cc | 未覆盖 |

## MicroAPI 计数

性能表关键项（latency, throughput）：

| perf 类型 | latency | throughput | 备注 |
| --- | --- | --- | --- |
| kLoad | 0 | 0 | DataCopy load src0/src1 |
| kDuplicate (uint8 / input) | 8 | 1 | 头部 ×2，scalar 追加 ×1 |
| kUpdateMask | 3 | 1 | 头部 ×2 |
| kEq / kNe / kLt / kGt / kGe / kLe | 3 | 1 | mode 不区分 |
| kMaskPack (uint8) | 8 | 1 | 16 bit ×bc，32/64 bit ×2bc |
| kSelect (uint8) | 3 | 1 | |
| kStore（dtype-mapping 默认项） | 0 | 1 | input→uint8 |
| kPlaceholder | 0 | 0 | CreateMask 占位 |

## 公式

最终 cost 结构（`compare_vf_head_cost` 替代通用 `GetVFHeadCost()`）：

```cpp
// 头部 + 循环体各 MicroAPI 通过 AddVfInstructPerf / AddVfInstructDtypeMappingPerf
// 累加 all_vf_instruct_cost（吞吐 × 次数），并取 max_latency；
Expr res = compare_vf_head_cost + max_latency + all_vf_instruct_cost;
res = res * node_info.compare_node_params.outer_call_count;
res.Simplify();
perf.pipe_res[PipeType::AIV_VEC] = res;
```
