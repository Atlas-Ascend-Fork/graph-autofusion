# Transpose（TransposeExtend）V2 性能模型

## 适用范围

- 面向 v35 形态（V2）ASCIR `Transpose` 节点，对应 regbase API `TransposeExtend`。
- 覆盖 `total_dim <= 4` 且 `inner_dim <= 3` 的六个分支，以及 `total_dim > 4` 或 `inner_dim >= 4` 的通用回退分支。
- 节点参数（`TransposeNodeParams`）无效时回退 `UnitVector` 通用模型。
- 输出 Pipe 为 `AIV_VEC`；不建模 Reg 与 UB 间的搬运开销。

## 源码依据

- codegen impl：`autofuse/v35/ascir/generator/v2_ascir_codegen_impl.h` `TransposeAscIrCodegenImplV2`（`GetApiName()=TransposeExtend`，`GetApiCallName()=TransposeRegApiCall`）。
- API 源码（regbase）：`autofuse/v35/ascendc/api_regbase/transpose.h`。
- API 调用生成：`autofuse/v35/codegen/reg_api_call/reg_transpose_api_call.cpp` `TransposeRegApiCall::BuildApiParam` / `FillTransposeNodeParams` / `GenDimensionParam`。
- 性能表：`autofuse/v35/att/api_perf_register/perf_param_v2.cpp` `kVfInstructPerfTable`。
- 当前 perf 实现：`autofuse/v35/att/api_perf_register/ascendc_regbase_perf.cpp` `TransposePerf`。
- ATT 注册入口：`autofuse/v35/att/api_perf_register/ascir_api_perf_v2.cpp` `TransposeApi`（`kTranspose` -> `TransposeV2`）。

## 参数来源与链路

`TransposeNodeParams`（`autofuse/common/ascir_node_param/ascir_node_param.h:122`）字段与来源：

| 字段 | codegen 传参来源 | 说明 |
| --- | --- | --- |
| `valid` | `BuildApiParam` 必置 true | false 时 ATT 回退 UnitVector |
| `inner_dim` | `GetContinuousInnerAxisNum(y, ...)` | 输出向量化轴尾部连续内轴数，上限 3 |
| `total_dim` | `min(y.vectorized_axis.size(), 4)` | 参与模板的向量化轴总数，上限 4 |
| `outer_loop_axes` | `out_vectorized_repeats[0, size - total_dim)` | total_dim 之外的输出向量化轴 repeat |
| `output_dims` | `out_vectorized_repeats[size - total_dim, size)` | 按输出轴序的 repeat，原始符号表达式 |
| `input_strides` | `ReorderInputStrideByOutputAxisOrder` 结果 | 按输出轴序重排的输入 stride，原始符号表达式 |
| `output_strides` | `y.vectorized_strides[size - total_dim, size)` | 输出 stride，原始符号表达式 |

链路：`BuildApiParam`（merge 后原始符号填充）-> `FillTransposeNodeParams`（写入节点 `AscirNodeParams::specific_params`）-> `FillSpecificParams`（`autofuse/att/gen_model_info/parser/specific_params_builder.cpp:70` 提取到 `NodeInfo`）-> `TransposeApi` -> `TransposePerf`。

生成 C++ 调用字符串使用 `tpipe.tiler` 展开（`GenTransposeDimParam`），传给 ATT 的节点参数保持 `merge_info`/向量化轴原始符号，两者解耦。

## 公共建模量

设 `data_dtype` 为输入 dtype，`repeat_elm` 取自 `kRptEleMap`（fp16=128，fp32=64 等）：

- `cal_count`：内轴元素数，`cal_count = Π output_dims[total_dim - inner_dim, total_dim)`。
- `repeat_time = Ceil(cal_count / repeat_elm)`。
- `tail_repeat = Ceil(max(cal_count - repeat_elm, 0) / repeat_elm)`（索引生成尾部循环次数）。
- `outer_count`：`Π outer_loop_axes`，为空时取 1。
- `transpose_stride_count`（符号变量）+ 三元定义：
  `alignment = 128 / sizeof(data_dtype)`（fp16 时 64），
  `stride_count = (Mod(input_strides.back(), alignment) == 0) ? alignment : Mod(input_strides.back(), alignment)`。
- `gather_count`：
  - `inner_dim ∈ {1, 2}`：`2 * stride_count`；
  - `inner_dim = 3`：`stride_count + 1`。

## 分支分析

`transpose.h` 的 `TransposeExtend` 按 `(dim, inner_dim)` 模板分派，每个分支由「索引生成」与「数据搬运」两段组成：

| 分支 (total_dim, inner_dim) | 索引生成 | 搬运实现 | 标定 latency 常数 |
| --- | --- | --- | --- |
| (2, 1) | `GenOneInnerDimTransposeIndex` | `TransposeOneOuterDimExtendImpl` | 69 |
| (3, 1) | `GenOneInnerDimTransposeIndex` | `TransposeTwoOuterDimExtendImpl` | 69 |
| (3, 2) | `GenTwoInnerDimTransposeIndex` | `TransposeOneOuterDimExtendImpl` | 167 |
| (4, 1) | `GenOneInnerDimTransposeIndex` | `TransposeThreeOuterDimExtendImpl` | 69 |
| (4, 2) | `GenTwoInnerDimTransposeIndex` | `TransposeTwoOuterDimExtendImpl` | 167 |
| (4, 3) | `GenThreeInnerDimTransposeIndex` | `TransposeOneOuterDimExtendImpl` | 386 |
| 其他（total_dim > 4 或 inner_dim >= 4） | inner_dim ∈ {1,2,3} 按对应 Gen 成本；inner_dim >= 4 不建模索引 | `TransposeExtendImpl` 通用搬运 | inner_dim ∈ {1,2,3} 沿用 69/167/386，否则 0 |
| `valid = false` | 无 | UnitVector 回退 | 0 |

搬运阶段循环层级（决定 Gather/Store 次数）：

- `TransposeOneOuterDimExtendImpl`：`movement_count = repeat_time * output_dims[0]`。
- `TransposeTwoOuterDimExtendImpl`：`movement_count = repeat_time * output_dims[0] * output_dims[1]`。
- `TransposeThreeOuterDimExtendImpl`：`movement_count = repeat_time * output_dims[0] * output_dims[1] * output_dims[2]`。
- `TransposeExtendImpl`（通用）：`movement_count = repeat_time`。

## MicroAPI 计数

性能表关键项（latency, throughput；fp16 除非注明）：

| perf 类型 | latency | throughput | 备注 |
| --- | --- | --- | --- |
| kLoad | 0 | 0 | LoadAlign |
| kGather2 | 0 | 1 | Gather 按索引搬运 |
| kStore (fp16/u8) | 0 | 5 | StoreAlign / 索引 DataCopy |
| kStore (fp32) | 0 | 1 | 搬运阶段 store 使用 |
| kPlaceholder | 0 | 0 | 无精确表项的占位 |
| kMuls / kAdds | 5 / 4 | 1 / 2 | |
| kAdd / kSub / kMul | 4 / 4 / 5 | 1 / 1 / 1 | |
| kDiv (fp32) | 14 | 4 | 索引生成除法按 fp32 建模 |
| kDuplicate / kArange (int32) | 8 / 4 | 1 / 2 | |
| kCompareScalarGE / kSelect / kUpdateMask | 3 | 1 | |
| kCopy / kUnitVector | 无表项 | 无表项 | 计 0, 0 |

各段 `AddVfInstructPerf` 计数（`repeat_time` 简写 rt，`tail_repeat` 简写 tr）：

- GenOne（`src_stride0` 单内轴索引）：kPlaceholder(Arange, fp16) x1；kMuls x rt；kAdds x rt；kPlaceholder(DataCopy store index) x rt。
- GenTwo（双内轴索引）：
  - 初始化：kArange(int32) x1；kDuplicate x1；kCopy x1；kDiv(fp32) x1；kMul x1；kSub x1；kMuls x2；kAdd x1；kStore(u8, DataCopy store index) x1。
  - 尾部循环：kUpdateMask x tr；kAdds x2tr；kCompareScalarGE x tr；kSelect x tr；kMuls x tr；kSub x tr；kAdd x tr；kMuls x2tr；kStore(u8) x tr。
- GenThree（三内轴索引）：
  - 初始化：kPlaceholder(Arange, fp16) x1；kDuplicate x2；kCopy x2；kDiv(fp32) x3；kMul x3；kPlaceholder(DataCopy store index) x1。
  - 尾部循环：kUpdateMask x tr；kSelect x2tr；kMuls x5tr；kAdd x4tr；kMuls x3tr；kPlaceholder(DataCopy store index) x tr。
- 搬运段（One/Two/Three Outer 及通用 ExtendImpl 结构一致，仅 movement_count 不同）：
  - kLoad x rt；
  - kGather2 x `movement_count * gather_count`；
  - kStore(fp32) x movement_count。

## 公式

最终 cost 结构（区别于通用 `GetVFHeadCost() + max_latency + all_vf_instruct_cost` 形式）：

```cpp
// 索引生成 + 搬运各 MicroAPI 通过 AddVfInstructPerf 累加 all_vf_instruct_cost；
// max_latency 计算后未使用，以分支标定常数 latency (69/167/386/0) 替代。
Expr res = (VfPerfUtils::GetVFHeadCost() + CreateExpr(latency) + all_vf_instruct_cost) * outer_count;
res.Simplify();
perf.pipe_res[PipeType::AIV_VEC] = res;
```

其中 `GetVFHeadCost() = 20`。`valid = false` 时回退 `RegVfPerf(kUnitVector, ...)`：`20 + 0 + 0`（kUnitVector 无表项）。
