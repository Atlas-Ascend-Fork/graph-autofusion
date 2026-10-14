# Concat 自动融合算子 — 综合技术文档

## 1. 特性背景

### 1.1 Concat 算子简介

Concat 算子是深度学习和科学计算中最基础、最常用的张量操作之一，核心功能是将多个张量沿某个指定的维度拼接在一起。

示例（以 PyTorch 为例）：

```python
import torch

a = torch.tensor([[1, 2], [3, 4]])   # shape: (2, 2)
b = torch.tensor([[5, 6], [7, 8]])   # shape: (2, 2)

# 沿 dim=0 拼接（按行堆叠）
c = torch.cat([a, b], dim=0)
# shape: (4, 2)
# [[1, 2],
#  [3, 4],
#  [5, 6],
#  [7, 8]]

# 沿 dim=1 拼接（按列堆叠）
d = torch.cat([a, b], dim=1)
# shape: (2, 4)
# [[1, 2, 5, 6],
#  [3, 4, 7, 8]]
```

### 1.2 融合场景与设计目标

Concat 融合是自动融合子系统的关键特性之一，通过将 Concat 算子与前后的算子融合，减少中间数据的搬运开销，提升整体执行性能。

Concat 支持以下前融合场景：

- ElementWise + Concat
- Gather + Concat
- Slice + Concat
- Split + Concat

其中，Split + Concat 的融合存在限制：Split 在首轴和/或 Concat 在首轴（split/concat 轴之前的轴的大小全为 1 也被视作首轴），详见 2.2。

Concat 支持以下后融合场景：

- Concat + ElementWise

## 2. 特殊背景及限制

### 2.1 平台与 dtype 支持差异

- **A2/A3**（v1 注册）：Concat 不支持 `DT_BOOL` 与 `DT_INT64`（源码注释：DT_INT64 后续根据需要放开）；
- **A5**（v2 注册）：支持 `DT_BOOL` 与 `DT_INT64`。

### 2.2 Split-Concat 首轴互斥约束

Split 与 Concat 中，至少需要有一个操作首轴时才允许融合，该约束由前端保证。不满足约束的图不进入本特性处理流程。

### 2.3 Concat 轴不可切分

当前实现中，concat 轴以及其后的轴不支持切分。这些轴在 TilingGroup 分组时被分入 N-group（见 4.1），也是融合模板生成需要处理"UB 不全载"问题的根源。

### 2.4 inductor 场景的输入个数限制

该场景使用 `<<<>>>` 的方式进行 kernel 调用，不支持 ListTensorDesc 特性，过多的输入会引入较高的 Scalar 开销；另外该场景也不支持并发编译，过多的输入也会导致编译时间较长。建议不要对超过阈值的 concat 进行 lowering。

### 2.5 Concat 后融合限制

Concat 后融合的算子不应对 Concat 轴再进行 Broadcast，否则会导致可用于 Concat 的 UB buffer 大小过小，从而导致性能劣化。

### 2.6 动态 shape 场景限制

1. 如果 concat 轴以及其后轴存在动态 shape，则无法在编译期计算出其大小，无法对 concat 进行分组，此时只能将其单独分为一组。group 总数过多会导致编译时间过长、函数栈过大等问题，性能也大概率不好，因此需要前端将 Concat 的输入个数限制得更小，如 32；
2. 如果子图内涉及的动态 shape 符号个数过多，会导致 tiling data 过大等问题，所以需要前端限制最大的符号个数，超过则不 lowering Concat。当前 TF 前端的限制为 16 个。

### 2.7 融合阈值的限制

基于 AF codegen 的一些限制，codegen 出的 Concat 的性能在一些场景会差于单算子，所以如果产生的融合算子中 cat 的融合比例较低，则会导致劣化。为了缓解该问题，需要由前端计算融合的比例，对于低于阈值的场景不要进行 Concat 融合。

- 如果存在后融合，则可认为比例足够，可以进行融合；
- 否则，计算输入的比例：`sum(能融合的输入的concat_dim) / 输出concat_dim`。

对于不同的 cat 场景，比例的 threshold 可以不同。如 TF 中，区分首轴 concat、no task 的首轴 concat、全对齐的 concat、一般 Concat 等：

```c++
std::map<ConcatCase, ge::float64_t> threshold = {
    {ConcatCase::kFirstDim, 0.0},
    {ConcatCase::kFirstDimNoTaskLarge, 1.0},
    {ConcatCase::kFirstDimNoTaskMid, 0.8},
    {ConcatCase::kFirstDimNoTaskSmall, 0.5},
    {ConcatCase::kFirstDimNoTaskTiny, 0.3333},
    {ConcatCase::kAllAligned, 0.1666},
    {ConcatCase::kOther, 0.1666},
};
```

## 3. 对外接口

### 3.1 IR 注册

AF 只定义了 Concat 的 AscIR。对于一些与 Concat 类似的算子（如 tf.stack），没有单独定义 IR，需要由前端转换为 Concat 的形式。

v1 注册位于 `autofuse/ascir/generator/ascir_builtin_ops_v1.cpp`，v2 注册位于 `autofuse/v35/ascir/generator/ascir_builtin_ops_v2.cpp`：

```c++
// A2/A3（v1，不支持 DT_BOOL/DT_INT64）
REG_ASC_IR(Concat)
    .DynamicInput("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeConcat)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ConcatAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::ConcatAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_UINT64,
                                              DT_BF16, DT_FLOAT16, DT_FLOAT}}}});

// A5（v2，支持 DT_BOOL/DT_INT64）
REG_ASC_IR(Concat).Impl(v2_soc_versions,
                        {af::ascir::AscIrImplCreator<af::ascir::ConcatAscIrAttImplV2>(),
                         af::ascir::AscIrImplCreator<af::ascir::ConcatAscIrCodegenImplV2>(),
                         {{"T", TensorType{DT_BOOL, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32,
                                           DT_INT64, DT_UINT64, DT_BF16, DT_FLOAT16, DT_FLOAT}}}});
```

### 3.2 设备侧 AscendC API

**A2/A3（`autofuse/ascendc/api/concat.h`）：**

| API | 实现方式 | 触发条件 |
| --- | --- | --- |
| `ConcatAllAligned` | DataCopy | 所有输入对齐到 32B |
| `ConcatExtendV2` / `ConcatExtendV2Dyn` | TransDataTo5HD | 小尾轴场景特化，性能优于 ConcatExtend；`V2Dyn` 用于动态 shape |
| `ConcatExtend` | TransDataTo5HD | 默认实现，无法使用以上特化时 |

**A5（`autofuse/v35/ascendc/api_regbase/concat.h`）：**

| API | 实现方式 | 触发条件 |
| --- | --- | --- |
| `ConcatOneAxis` | 单维首轴拼接展开 | 单维首轴 Concat，尾轴不超过一个向量长度 |
| `ConcatAllAligned` | DataCopy | 所有输入对齐到 32B |
| `ConcatExtendByGather` | DataCopyGather | 输入 shape 相同 |
| `ConcatExtendDyn` | 运行期分派 | 动态 shape 场景，运行时在 Gather/Extend 之间选择 |
| `ConcatExtend` | DataCopyScatter | 默认实现，无法使用以上特化时 |

各 API 由 Codegen 按节点信息选择，具体触发条件见 4.4。

## 4. 核心实现

### 4.1 TilingGroup 轴分组（tiling_group.cpp）

`autofuse/optimize/autoschedule/tiling_group.cpp` 根据节点的 `compute_type` 把 loop 轴分入四组：

| 组 | 含义 | 包含的轴 |
| --- | --- | --- |
| X-group | Transpose 输入排他轴 | Transpose 中输入有但输出没有的轴 |
| Y-group | 主计算轴 | ElementWise/Broadcast 的所有 loop 轴；Reduce 的非 Reduce 轴 |
| R-group | Reduce 轴 | 被 Reduce 规约的轴 |
| N-group | 不可切分轴 | size=1 的轴、Concat/Split 的操作轴 |

如 2.3 所述，concat 轴及其后的轴不支持切分，会被分入 N-group。

### 4.2 Split-Concat 融合的处理（split_concat_optimization_pass.cpp）

`autofuse/v35/optimize/graph_pass/split_concat_optimization_pass.cpp`。

Split 与 Concat 中至少需要有一个操作首轴时才允许融合（该约束由前端保证，见 2.2）。该 Pass 进行以下改图，确保图中只存在 Split 或只存在 Concat：

1. 如果 Split 在首轴，则将 Split 转为多个非连续 Load；
2. 否则，此时 Concat 在首轴，将 Concat 转为多个非连续 Store。

### 4.3 融合模板生成（concat_schedule_case_generator.cpp）

`autofuse/optimize/task_generator/concat_schedule_case_generator.cpp`。

由于 N-group 的存在，完整的 Concat 可能无法在 UB 内全载，所以需要生成拆分 Concat 的模板以应对该场景。

为便于描述，可将原始轴以 concat 轴为界合并为 2 个轴：concat 轴之前的轴合并为 m 轴，concat 轴及后续轴合并为 n 轴。

#### 4.3.1 非首轴 Concat：输入分组与拆分

对于**非首轴** Concat，会对 Concat 按输入进行分组（`autofuse/optimize/task_generator/concat_group_partitioner.cpp`），确保每个组内的融合子图能够在 UB 内全载。

分组逻辑如下：

1. 组内 n 轴大小之和超过上限（如果 n 轴包含动态 shape，则需要单独分为一组）；
2. 组内输入个数超过上限；
3. 组类型不匹配。

不同的组可能存在共同祖先节点，后续无法拆开会导致仍然无法在 UB 内全载，所以还需要对跨组共享的节点进行重计算，确保后续各组可以完全被拆开。

分组后，按分组的结果拆分原 Concat：

- 如果组内只有一个输入，则将该输入与备份出的 Concat 的输出相连；
- 如果组内有多个输入，则构造出新的较小的 Concat，将输入改连到新的 Concat 上，Concat 的输出与备份出的 Concat 的输出相连；
- 重新连接节点后，更新关联的轴信息，刷新关联的 Load/Store 节点的 offset 属性。

#### 4.3.2 首轴 Concat

对于**首轴** Concat，如果所有轴都在 N-group，无法进行后续处理，所以按每个输入都各自为一组的方式进行拆分。

但是如果 n 轴过小，此时搬运效率下降可能会导致性能劣化，此时也会生成拆分为子 Concat 的模板：

- 如果 Concat 轴之前不包含大小为 1 的轴，则补一个大小为 1 的轴，确保不是所有轴都位于 N-group 中；
- 标记该轴不被 Merge。

#### 4.3.3 模板清单

综上：

首轴 Concat 会生成 2 个模板：

1. 拆分为子 Concat；
2. 完全消除 Concat。

非首轴 Concat 会生成 2 个模板：

1. 保留 Concat（输入个数超过上限时不生成）；
2. 拆分 Concat，应对 UB 不全载场景。

#### 4.3.4 平台特殊场景

1. [A5] 如果只有一根轴，且大小足够 UB 全载，则在前面补一个大小为 1 的轴，保留原始 Concat；
2. [A3] 如果匹配小尾轴场景，则只生成保留 Concat 的模板；
3. [A3] 如果是动态 shape，则多生成一个保留 Concat 的模板，标记为在运行时选择是否使用小尾轴。

#### 4.3.5 打分函数

提供了打分函数缓解 ATT 选择模板不准确的问题：

- **A3**：优先选择小尾轴模板；不满足时，如果对齐的输入的占比高，则选择保留 Concat 的模板，否则选择拆分 Concat 的模板；
- **A5**：如果不存在重计算，且分组后每个组的元素个数都为 1，且输出大小较小，则优先选择保留 Concat 的模板。

### 4.4 Codegen

根据 Concat 节点信息，生成执行 Concat 所需的代码。能够在编译期简化的计算需要在此进行，不要遗留到运行期。

#### 4.4.1 A2/A3：ConcatApiCall

`autofuse/codegen/api_call/concat/concat_api_call.cpp`，按以下优先级生成三种调用形态：

1. **小尾轴路径**：节点带有 `_concat_small_tail` 属性时走此路径（对应 4.3.4 的小尾轴特化）：
   - concat 轴大小为静态：编译期 `CalcTiling` 预计算全部 tiling 参数，生成 `ConcatExtendV2(concat_context, tiling, y, tmp_buf)`；
   - concat 轴大小为动态：生成运行期使用的 `ConcatShape`，生成 `ConcatExtendV2Dyn(concat_context, concat_shape, y, tmp_buf)`；
   - `concat_context` 按 `SameDim/DiffDim`（各输入 concat 段大小是否全为 1）与 `Padded`（是否存在 padding）分型。
2. **全对齐路径**：`IsAllAligned` 成立（每个输入 concat 段实际大小 × dtype size 对 32B 取模为 0）时，生成 `ConcatAllAligned<dtype, N>(rows, concat_tiling, y, concat_src_tensors)`；
3. **默认路径**：生成 `ConcatExtend<dtype, 向量化轴数, 输入数>(dst_params, srcs_params, concat_dim, tmp_buf)`，输出与各输入以 `ConcatParams`（shape/stride/tensor）结构描述。

补充说明：

- **concat 轴识别**：沿向量化轴比较首个输入与输出的轴大小，第一个不相等处即为 concat 轴；
- **tmp buffer**：除 `ConcatAllAligned` 外，其余路径均需要全生命周期 TBuf（`life_time_axis_id = -1`）复用 id；
- **padding**：当前 padding 只允许出现在最后一维，与 concat 轴无关，存在 padding 时会在 tiling/shape 中额外生成 row stride 信息。

生成调用形态示例：

```cpp
ConcatAllAligned<float, 3>(concat_rows, concat_tiling, y, concat_src_tensors);
ConcatExtend<float, 2, 3>(dst_params, srcs_params, concat_dim, tmp_buf_0);
ConcatExtendV2(concat_context, tiling, y, tmp_buf_0);
```

#### 4.4.2 A5：ConcatRegApiCall

`autofuse/v35/codegen/reg_api_call/concat_reg_api_call.cpp`，按以下优先级选择路径：

1. **`ConcatOneAxis`**：单维首轴 Concat（输出仅有两根向量化轴且首 stride 为 0，各输入尾轴为静态且元素数不超过 `256 / dtype_size`），生成 constexpr tiling（各输入尾轴大小与输出偏移）并调用 `concat::ConcatOneAxis`；
2. **`ConcatAllAligned`**：全对齐场景（判定方式与 A2/A3 一致）；
3. **`ConcatExtendByGather`**：满足 `CanUseGather` 条件时生成——无 padding、输入 shape 相同（或所有输入来自同一源节点的 tile 场景）、单个输入 concat 段字节数不超过 128B。gather 路径偏好连续的输入 buffer；
4. **`ConcatExtend` / `ConcatExtendDyn`**：默认 scatter 路径；若满足 gather 条件但 shape 为动态，则生成 `ConcatExtendDyn`，由运行期在 Gather/Extend 之间分派。

dtype 规约（在 tiling 计算前完成）：

- 8 字节 dtype（`DT_INT64/DT_UINT64`）→ 拆为 `uint32_t` 计算（B64ToB32）；
- 1 字节 dtype 且各段大小可被 2 整除 → 转 `uint16_t` 计算（B8ToB16）；
- `int8_t`/`bool` → `uint8_t`（pack 不支持 int8_t）。

生成调用形态示例：

```cpp
concat::ConcatOneAxis<float, 3>((float *)y.GetPhyAddr(), concat_src_addrs, concat_tiling);
concat::ConcatExtendByGather<float, 3>((float *)y.GetPhyAddr(), concat_src_addrs, tmp_buf_0, concat_tiling);
concat::ConcatExtend<float, 3>((float *)y.GetPhyAddr(), concat_src_addrs, tmp_buf_0, concat_tiling);
```

## 5. 常见问题定位

### 5.1 问题1：编译失败——没有可用的 tiling

查看 AfterSplitConcat 的 dump 图，检查拆分 Concat 后，是否存在 repeat 更新错误，导致 Concat 轴识别错误。

### 5.2 问题2：编译失败——存在重定义

查看 AfterSplitConcat 的 dump 图，检查是否存在不同 group 存在共同祖先问题。如果是，则为重计算逻辑存在 bug，导致 group 无法完全拆开。

### 5.3 问题3：精度问题

1. 确认实际调用的模板；
2. 如果是拆分 Concat 的模板，可通过出现精度问题的数据的位置，推算出对应的 group；
3. 检查对应的 dump 图，查看是否存在轴大小错误等问题；
4. 如果还是定位不出来，可通过 replace_kernel 的方式，增加 **printf** 打印进行定位。

### 5.4 问题4：性能问题

1. 排除其他算子的性能问题；
2. 检查 ATT 选中的模板，可以强制指定模板测试是否有更好的选择。

## 6. 关键文件索引

| 文件路径 | 职责 |
| --- | --- |
| `autofuse/ascir/generator/ascir_builtin_ops_v1.cpp` | Concat IR v1 注册（A2/A3） |
| `autofuse/v35/ascir/generator/ascir_builtin_ops_v2.cpp` | Concat IR v2 注册（A5，支持 BOOL/INT64） |
| `autofuse/optimize/autoschedule/tiling_group.cpp` | loop 轴分组（N-group） |
| `autofuse/v35/optimize/graph_pass/split_concat_optimization_pass.cpp` | Split-Concat 改图 Pass |
| `autofuse/optimize/task_generator/concat_schedule_case_generator.cpp` | 融合模板生成与打分函数 |
| `autofuse/optimize/task_generator/concat_group_partitioner.cpp` | Concat 输入分组 |
| `autofuse/codegen/api_call/concat/concat_api_call.cpp` | A2/A3 Codegen |
| `autofuse/v35/codegen/reg_api_call/concat_reg_api_call.cpp` | A5 Codegen |
| `autofuse/ascendc/api/concat.h` | A2/A3 设备侧 API 实现 |
| `autofuse/v35/ascendc/api_regbase/concat.h` | A5 设备侧 API 实现 |

## 相关链接

- 返回 [AutoFuse 架构介绍](../../introduction/architecture.md)
