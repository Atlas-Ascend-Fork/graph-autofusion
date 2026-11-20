# Where（WhereExtend）V2 性能模型

## 适用范围

- 面向 v35 形态（V2）ASCIR `Where` / `Select` 节点，对应 regbase API `WhereExtend`
  （`where.h`）。
- 覆盖 1-D（`output_dims = [cal]`）与 2-D（`output_dims = [counter_first, cal]`）两类循环
  结构，以及 B8 / B16 / B32 / B64 四档数据位宽。
- `where_node_params.valid = false` 时回退 legacy 模型（本归档未覆盖）。
- 输出 Pipe 为 `AIV_VEC`；不建模 Reg 与 UB 间的搬运开销。

## 源码依据

- codegen impl：`autofuse/v35/ascir/generator/v2_ascir_codegen_impl.h` `WhereAscIrCodegenImplV2`
  （`GetApiCallName()=WhereRegApiCall`，`GetApiName()=WhereExtend`）。
- API 源码（regbase）：`autofuse/v35/ascendc/api_regbase/where.h`。
- API 调用生成：`autofuse/v35/codegen/reg_api_call/reg_where_api_call.cpp`
  `BuildWhereLoopParams` / `FillWhereNodeParams`（`is_bcast_src0/src1` 取自
  `x2/x3.IsAnyScalar()`）。
- 性能表：`autofuse/v35/att/api_perf_register/perf_param_v2.cpp` `kVfInstructPerfTable`。
- 当前 perf 实现：`autofuse/v35/att/api_perf_register/ascendc_regbase_perf.cpp` `WherePerf`。
- ATT 注册入口：`autofuse/v35/att/api_perf_register/ascir_api_perf_v2.cpp` `WhereApi`
  （`kWhere` -> `WhereV2`；`kSelect` 同样映射 `WhereV2`）。

## 参数来源与链路

`WhereNodeParams`（`autofuse/common/ascir_node_param/ascir_node_param.h:103`）字段与来源：

| 字段 | codegen 传参来源 | 说明 |
| --- | --- | --- |
| `valid` | `FillWhereNodeParams` 置位 | false 时 ATT 走 legacy 模型 |
| `is_bcast_src0` / `is_bcast_src1` | `x2` / `x3` 的 `IsAnyScalar()` | 仅减少一路 kLoad（表项 (0,0)），**不影响预测值** |
| `outer_call_count` | `Π merge_repeats[0, size-1)` | 最外层循环乘子；1-D（outer_repeats 为空）时恒为 1 |
| `output_dims` | `[cal]`（1-D）或 `[merge_repeats.back(), cal]`（2-D） | `front()` 为 counter_first，`back()` 为单次处理 count |
| `output_strides` / `mask_strides` / `input_strides` | 对应 stride | 参与模型但不影响当前指令计数 |

链路：`BuildWhereLoopParams` -> `FillWhereNodeParams`（写入 `AscirNodeParams::specific_params`）->
`FillSpecificParams`（`autofuse/att/gen_model_info/parser/specific_params_builder.cpp:58` 提取到
`NodeInfo`）-> `WhereApi` -> `WherePerf`。

## 公共建模量

- `data_dtype`：取输出 dtype（`GetWhereDataDtype`）。
- `repeat_elm = kRptEleMap[data_dtype]`，B64（int64/uint64）再 ×2（=64）；
  fp16=128、fp32=64、u8/i8=256。
- `repeat_time = Ceil(count / repeat_elm)`：1-D 取 `count = output_dims.front()`；
  2-D 取 `count = output_dims.back()`（counter_last）。
- `where_vf_head_cost`：B8（u8/i8）=43、B16（fp16/bf16/i16/u16）=45、B32（fp32/i32/u32）=49、
  B64（i64/u64）=57。
- 2-D 行开销（仅 B16/B32 走按行路径）：B16=4、B32=3。
- B64 输出的 Select 次数 ×2（`kWhereSelectOutputFactorB64`）。

## 指令序列与分支分析

`AddWhereComputePerf(data_dtype, block_count=bc)`（2-D 路径使用）：

- `kPlaceholder(u8) ×1`（CreateMask，无表项计 0）；
- 非 B16/B32（即 B8 和 B64）时：`kUpdateMask(i32) ×bc` + `kCompareScalarNE(u8) ×bc`
  （各 (3,1)）；
- `kLoad(u8) ×bc`（mask load，(0,0)）；
- `kMaskUnPack(u8)`：B16 ×bc、B32/B64 ×2bc、B8 不执行（(8,1)）；
- 非 bcast 的 src0/src1 各 `kLoad(dtype) ×bc`（(0,0)，bcast 标志因此不影响预测）；
- `kSelect(i32) ×bc×output_factor`（(3,1)，B64 输出 factor=2）；
- `kStore(i32) ×bc`（(0,1)）。

`AddWhereImplPerf`（1-D 路径）指令序列与上表一致，仅 UpdateMask/CompareScalarNE 的
触发条件为非 B8/B16/B32（即仅 B64 执行，B8 在 1-D 不执行）。

路径分派：

- **1-D**（`AddWhereImplPerf`）：`rt = Ceil(cal / repeat_elm)`，按上表 `bc = rt` 计数，
  `max_latency` 正常累加（B16/B32/B64 = 8，B8 = 3）。
- **2-D B8/B64**（`AddWhereExtend2DPerf` 通用路径）：`bc = counter_first × rt`，单次
  `AddWhereComputePerf`，`max_latency` 正常累加。
- **2-D B16/B32**（按行路径）：每行调用 `AddWhereComputePerf(bc = rt)` 后仅取
  `row_vf_instruct_cost`，**`row_max_latency` 被丢弃（max_latency=0）**，并叠加行开销
  （B16=4 / B32=3）后 × counter_first。

各分支公式（rt = repeat_time，cf = counter_first，cc = outer_call_count）：

| 分支 | repeat_elm | 1-D 公式 | 2-D 公式 |
| --- | --- | --- | --- |
| int8/uint8（B8） | 256 | (46+2·rt)·cc | (46+4·cf·rt)·cc |
| float16/bf16/int16/uint16（B16） | 128 | (53+3·rt)·cc | (45+(3·rt+4)·cf)·cc |
| float32/int32/uint32（B32） | 64 | (57+4·rt)·cc | (49+(4·rt+3)·cf)·cc |
| int64/uint64（B64） | 64 | (65+7·rt)·cc | (65+7·cf·rt)·cc |

## MicroAPI 计数

性能表关键项（latency, throughput）：

| perf 类型 | latency | throughput | 备注 |
| --- | --- | --- | --- |
| kLoad | 0 | 0 | mask / src0 / src1 load |
| kMaskUnPack (uint8) | 8 | 1 | B16 ×bc，B32/B64 ×2bc |
| kUpdateMask | 3 | 1 | 仅 B64，×bc |
| kCompareScalarNE | 3 | 1 | 仅 B64，×bc |
| kSelect (int32) | 3 | 1 | B64 输出 ×2bc |
| kStore (int32) | 0 | 1 | |
| kPlaceholder | 0 | 0 | CreateMask 占位 |

## 公式

最终 cost 结构（`where_vf_head_cost` 替代通用 `GetVFHeadCost()`）：

```cpp
// 各 MicroAPI 通过 AddVfInstructPerf 累加 all_vf_instruct_cost（吞吐 × 次数）并取 max_latency；
Expr res = (where_vf_head_cost + max_latency + all_vf_instruct_cost) * outer_call_count;
res.Simplify();
perf.pipe_res[PipeType::AIV_VEC] = res;
```

`valid = false` 时 legacy 路径：`kDuplicate ×n + kCompareScalarNE ×rt + kSelect ×rt`，
head 取通用 `GetVFHeadCost()=20`，不乘 outer_call_count。
