# IsNan V2 性能模型

## 适用范围

- 面向 v35 形态（V2）ASCIR `Isnan` 节点，对应 API `IsNan`（头文件 `adv_api/math/is_nan.h`，
  codegen 经 `UnaryBitWidthChangeApiCallV2` 生成调用）。
- 覆盖 fp16/fp32 输入 × bool/uint8 输出组合；bf16 输入经 dtype 转换（BF16→FLOAT）后按
  fp32 处理（`GetConversionDtype`）。
- `IsNan` / `IsFinite` / `IsInf` 共用 `UnaryBitWidthChange` 框架，本归档仅覆盖 `IsNan`。
- 输出 Pipe 为 `AIV_VEC`；不建模 Reg 与 UB 间的搬运开销。

## 源码依据

- codegen impl：`autofuse/v35/ascir/generator/v2_ascir_codegen_impl.h` `IsnanAscIrCodegenImplV2`
  （`GetApiName()=IsNan`，`GetApiCallName()=UnaryBitWidthChangeApiCallV2`，输出经
  `ReinterpretCast<bool>`）。
- API 源码：`adv_api/math/is_nan.h`（asc-devkit，仓库外依赖）。
- API 调用生成：`autofuse/v35/codegen/reg_api_call/unary_bitwidth_change_api_call_v2.cpp`
  `UnaryBitWidthChangeApiCallV2::Generate` / `FillUnaryBitWidthChangeNodeParams`。
- 性能表：`autofuse/v35/att/api_perf_register/perf_param_v2.cpp` `kVfInstructPerfTable`。
- 当前 perf 实现：`autofuse/v35/att/api_perf_register/ascendc_regbase_perf.cpp` `IsNanPerf`
  （公共段 `AddUnaryBitWidthChangeCommonPerf` / 收尾 `FinishUnaryBitWidthChangePerf`）。
- ATT 注册入口：`autofuse/v35/att/api_perf_register/ascir_api_perf_v2.cpp` `IsNanApi`
  （`kIsnan` -> `IsNanV2`）。

## 参数来源与链路

`UnaryBitWidthChangeNodeParams`（`autofuse/common/ascir_node_param/ascir_node_param.h:114`）
字段与来源：

| 字段 | codegen 传参来源 | 说明 |
| --- | --- | --- |
| `valid` | `FillUnaryBitWidthChangeNodeParams` 置位 | false 时 cal_count 按输入元素总数、call_count=1 |
| `cal_count` | `param.cal_count` | 最内层单次处理元素数 |
| `outer_repeats` | `merge_repeats[0, size-1)`（去掉最后一个） | call_count = Π outer_repeats |
| `output_strides` / `input_strides` | 对应 stride | 参与模型但不影响当前指令计数 |

链路：`FillUnaryBitWidthChangeNodeParams`（写入 `AscirNodeParams::specific_params`）->
`FillSpecificParams`（`autofuse/att/gen_model_info/parser/specific_params_builder.cpp:65` 提取到
`NodeInfo`）-> `IsNanApi` -> `IsNanPerf`。

## 公共建模量

- `repeat_elm = kRptEleMap[input_dtype]`（fp16=128、fp32=64，缺省 `kRptSizeFloat=64`）。
- `repeat_time = Ceil(cal_count / repeat_elm)`。
- `call_count = Π outer_repeats`（参数无效时为 1）。
- `output_is_bool = (output_dtype == bool)`；codegen 侧输出恒为 bool（`ReinterpretCast<bool>`），
  uint8 输出仅 SIMT 标量路径支持（`IsSimtScalarSupported`）。

## 指令序列与分支分析

公共段（`AddUnaryBitWidthChangeCommonPerf`，rt = repeat_time）：

- `kLoad(input_dtype) ×rt`（(0,0) 不计）；
- `kNe(uint8) ×rt`（(3,1)）；
- `kStore(kFloat32) ×rt`（(0,1)）。

`output_is_bool` 时追加 MaskPack（两次调用，打包 cmpMaskReg 与 mask）：

- fp32 输入：`kMaskPack(uint8) ×2rt` ×2 次（fp32 mask 需打包两次）；
- fp16 输入：`kMaskPack(uint8) ×rt` ×2 次。

各分支公式（rt = repeat_time，cc = call_count；标定 latency 常数 26，实测注释
"Actual 6 repeat, Duplicate(2)、UpdateMask(1 repeat)、Select(1 repeat) are not recorded"）：

| 分支（input→output） | repeat_elm | 吞吐合计 | 公式 |
| --- | --- | --- | --- |
| float16 → bool | 128 | 4·rt（Ne rt + Store rt + MaskPack 2rt） | (46+4·rt)·cc |
| float32 → bool | 64 | 6·rt（Ne rt + Store rt + MaskPack 4rt） | (46+6·rt)·cc |
| float16 → uint8 | 128 | 2·rt（Ne rt + Store rt） | (46+2·rt)·cc |
| float32 → uint8 | 64 | 2·rt（Ne rt + Store rt） | (46+2·rt)·cc |

## MicroAPI 计数

性能表关键项（latency, throughput）：

| perf 类型 | latency | throughput | 备注 |
| --- | --- | --- | --- |
| kLoad | 0 | 0 | |
| kNe (uint8) | 3 | 1 | |
| kStore (kFloat32) | 0 | 1 | |
| kMaskPack (uint8) | 8 | 1 | bool 输出时 ×2 次调用，fp32 每次 2rt、fp16 每次 rt |

## 公式

最终 cost 结构（标定常数 `kIsNanMaxLatency = 26` 替代累加的 max_latency，head = 20）：

```cpp
// AddUnaryBitWidthChangeCommonPerf / MaskPack 通过 AddVfInstructPerf 累加
// all_vf_instruct_cost（吞吐 × 次数）；累加的 max_latency 被丢弃，以标定常数 26 替代；
Expr res = (VfPerfUtils::GetVFHeadCost() + CreateExpr(26) + all_vf_instruct_cost) * call_count;
res.Simplify();
perf.pipe_res[PipeType::AIV_VEC] = res;
```
