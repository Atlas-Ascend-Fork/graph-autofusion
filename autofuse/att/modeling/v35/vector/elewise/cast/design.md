# Cast（CastExtend）V2 性能模型

## 适用范围

- 面向 v35 形态（V2）ASCIR `Cast` 节点，对应 regbase API `CastExtend`；`Ceil2Int` 等 API 复用同一模型。
- 覆盖 7 类典型分支：通用浮点互转（fp16↔fp32）、同位宽整数互转（int32↔uint32）、
  两段转换（int32→fp16）、float32→bool、B64 收窄（int64→fp32）、B64 transfer（int64→fp16）。
- B4（fp16↔int4）、B8 特殊分支（uint8→fp32 等）本归档未覆盖，见限制。
- 输出 Pipe 为 `AIV_VEC`；不建模 Reg 与 UB 间的搬运开销。

## 源码依据

- codegen impl：`autofuse/v35/ascir/generator/v2_ascir_codegen_impl.h` `CastAscIrCodegenImplV2`
  （`GetApiName()=CastExtend`，`GetApiCallName()=CastV2ApiCall`）。
- API 源码（regbase）：`autofuse/v35/ascendc/api_regbase/cast.h`。
- API 调用生成：`autofuse/v35/codegen/reg_api_call/cast_v2_api_call.cpp` `CastV2ApiCall::BuildApiParam` /
  `FillCastNodeParams`。
- 性能表：`autofuse/v35/att/api_perf_register/perf_param_v2.cpp` `kVfInstructPerfTable` 与
  `kVfInstructDtypeMappingPerfTable`。
- 当前 perf 实现：`autofuse/v35/att/api_perf_register/ascendc_regbase_perf.cpp` `CastPerf`。
- ATT 注册入口：`autofuse/v35/att/api_perf_register/ascir_api_perf_v2.cpp` `CastApi`
  （`kCast` -> `CastV2`）。

## 参数来源与链路

`CastNodeParams`（`autofuse/common/ascir_node_param/ascir_node_param.h:77`）字段与来源：

| 字段 | codegen 传参来源 | 说明 |
| --- | --- | --- |
| `valid` | `BuildApiParam` 置位 | false 时按输入元素总数建模 |
| `output_dims` | 输出向量化轴 repeat | `back()` 为最内处理 count，`front()` 为外层 api_outer |
| `output_strides` | 输出向量化 stride | 与 count / input_strides 共同决定连续或跨 stride |
| `input_strides` | 输入向量化 stride | |

链路：`BuildApiParam` -> `FillCastNodeParams`（写入节点 `AscirNodeParams::specific_params`）->
`FillSpecificParams`（`autofuse/att/gen_model_info/parser/specific_params_builder.cpp:37` 提取到
`NodeInfo`）-> `CastApi` -> `CastPerf`。

## 公共建模量

- `one_rep_size = min(kRptEleMap[input_dtype], kRptEleMap[output_dtype])`，无表项时取
  `kRptSizeFloat = 64`；特判 `uint8 -> int4` 取 `kRptEleMap[float16] = 128`。
  （fp16↔fp32、int32↔uint32、int32→fp16、fp32→bool 均为 64；int64→fp32、int64→fp16 为 32。）
- `repeat_time`：
  - 同位宽整数 cast：`Ceil(output_dims.back() / one_rep_size)`，参数无效时按输入元素总数；
  - 非同位宽：一维 `Ceil(count / one_rep_size)`；多维按三元
    `output_strides.front() == input_strides.front()` 且 `== count`（内层连续）取
    `Ceil(api_outer * count / one_rep_size)`，否则取 `Ceil(count / one_rep_size)`。
- `call_count`：同位宽在 `output_dims.size() > 1` 时取 `output_dims.front()`，否则 1；
  非同位宽按三元（内层连续取 1，跨 stride 取 `output_dims.front()`）。
- 本归档用例统一取跨 stride 场景：`rt = Ceil(count / one_rep_size)`，`cc = outer_loop`。

## 分支分析

每个分支由「DataCopy（Load+Store）」与「计算段」组成，`rt` = repeat_time，`cc` = call_count：

| 分支（input→output） | 指令序列（吞吐计数） | max_latency | 公式 |
| --- | --- | --- | --- |
| fp16↔fp32 / fp16↔bf16 / fp32↔bf16（通用浮点） | Load×rt(0) + Store×rt(1) + Cast×rt(1) | 4 | (24+2·rt)·cc |
| int32↔uint32 等同位宽整数 | UpdateMask×2(2) + Load×rt(0) + Store×rt(1) | 3 | (25+rt)·cc |
| int32→float16（两段） | Load×rt(0) + Store×rt(1) + Cast(int32→fp32)×rt(2) + Cast(fp32→fp16)×rt(1) | 5 | (25+4·rt)·cc |
| float32→bool | Load×rt(0) + Store×rt(1) + Duplicate×1(1) + And×rt + Adds×rt(2) + Or×rt + Not×rt + Vshrs×rt | 8 | (29+7·rt)·cc |
| int64→float32（B64 收窄） | Load×rt(0) + Store×rt(1) + Cast×rt(2) + Duplicate×rt(1) + DeInterleave×rt(1) | 8 | (28+5·rt)·cc |
| int64→float16（B64 transfer） | Load×rt(0) + Store×rt(1) + Duplicate×1(1) + MaskPack×rt(0，u32 无表项) + DeInterleave×rt(1) + Cast(in→fp32)×rt(2) + Cast(fp32→fp16)×rt(1) | 8 | (29+5·rt)·cc |
| B4 / B8 特殊分支 | 见 `AddB4CastPerf` / `AddB8SpecificCastPerf` | - | 未覆盖 |

## MicroAPI 计数

性能表关键项（latency, throughput）：

| perf 类型 | latency | throughput | 备注 |
| --- | --- | --- | --- |
| kLoad | 0 | 0 | DataCopy load |
| kStore（dtype-mapping 默认项） | 0 | 1 | DataCopy store，仅 uint8→int64 特例为 5 |
| kCast（fp16/bf16/fp32 互转） | 4 | 1 | dtype-mapping 表 |
| kCast（默认项，含整型） | 5 | 2 | dtype-mapping 表 |
| kUpdateMask | 3 | 1 | 同位宽 cast 头部 ×2 |
| kDuplicate (uint32) | 8 | 1 | |
| kAnd / kOr / kNot / kVshrs (uint32) | 3 | 1 | |
| kAdds (uint32) | 4 | 2 | |
| kDeInterleave / kInterleave (uint32) | 8 | 1 | |
| kMaskPack | 8 | 1 | 仅 u8/i8 有表项，u32 调用无表项计 0 |

## 公式

最终 cost 结构（区别于标定常数分支，`max_latency` 参与求和）：

```cpp
// Load/Store/Compute 各 MicroAPI 通过 AddVfInstructPerf / AddVfInstructDtypeMappingPerf
// 累加 all_vf_instruct_cost（吞吐 × 次数），并取 max_latency；
Expr res = (VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost) * call_count;
res.Simplify();
perf.pipe_res[PipeType::AIV_VEC] = res;
```

其中 `GetVFHeadCost() = 20`。`valid = false` 或参数缺失时 `repeat_time` 按输入元素总数推导，
`call_count = 1`。
