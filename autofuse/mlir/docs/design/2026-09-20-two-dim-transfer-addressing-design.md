# 搬运的二维 tile 寻址设计

> **接口迁移说明（2026-10-22）**：正文提及的 `AF_MLIR_PINNED_*` / `AF_MLIR_ENABLE_PLAN_PINNING` 是历史接口，现已移除；当前使用 ATT `AUTOFUSE_DFX_FLAGS` 的 `--force_schedule_result` / `--force_tiling_case`。候选编号须核对当前 shape 的 schedule dump。

> 状态：实施中（S1～S5 已完成，S5 余两条 E2E 矩阵项待 910B 实机 / RCore / AllLoad 关闭，
> 见 8.5；RA 下降已于 2026-09-23 放开，不再是其中的阻塞）。2026-09-21 按 ARAR 调度探针的
> 实测结果做过一次重大修订，
> 推翻了「调度器会把 reduce 归一到二维」这个前提（实测 4 / 3 / 3 / 2 根），
> 并据此把「连续性合并」与「> 2 根套外层循环」从已知债务提进本期。
>
> 覆盖目标：reduce 的 **RCommon / RCore / AllLoad** 三个调度模板 × **AR / RA** 两种
> 归约形态，共六个组合。**每个组合的全部 tiling case 都必须可下降**，不是"跑通
> 一个即可"。
>
> 相关：[910B 被误判为不支持 DataCopyPadExtend](../development/2026-09-20-datacopy-pad-extend-wrongly-disabled-on-910b.md)
> ——**已修复**，本方案不再有外部阻塞项。

# 1 简介

## 1.1 目的

让 AFIR 的搬运算子（`afir.load` / `afir.store`）的寻址覆盖 **reduce 的三个调度
模板 × 两种归约形态**：

| | RCommon（`kCommon`） | RCore（`kRCore`） | AllLoad（`kAllLoad`） |
|---|---|---|---|
| **AR** | 必须支持 | 必须支持 | 必须支持 |
| **RA** | 必须支持 | 必须支持 | 必须支持 |

现有的 `TransferPlan` 只认单根向量化轴、单层循环、`loop_axis` 是 block-inner 轴，
六个组合**一个都进不来**。失败形态有两种，危险的是前一种：

- `getVectorizedAxis()` 取 `.front()` **无 size 检查**，多根向量化轴静默只取一根 →
  分配按 N 维、寻址按 1 维，**静默搬错地址**
- `buildScheduledPlan` 的 `splitPairOtherId` 反查落空 → `failure()`（可诊断）

本文给出定位公式、有效视图的归约规则、数据结构、失败形态与分期，
并顺带清理寻址链路上四项已确认的债务。

**目标读者**：接手 `dialect/AFIR/Analysis/TransferPlan.*` 与
`conversion/AFIRToAscendC/Implementations/` 的开发者。

**期望输出**：一份可以直接按 S1～S5 提交的实施依据。

### 1.1.1 AR / RA 对寻址是中立的

`TransferPlan` 只按 `vectorized_axis` 的顺序建表，**不关心哪根是 A、哪根是 R**——
那个区分属于 `ReducePlan`。所以"支持 RA"在寻址侧不产生额外工作量；真正的 RA 工作
在下降侧（`reduce_init` 的行主序假设、累加长度取物理第一维），不在本方案范围。

本方案对 RA 的义务只有一条：**不得引入任何依赖"末轴是被归约轴"的假设**。

### 1.1.2 三个模板各自提出什么

依据 `optimize/optimize.h:33-37` 的 `ReduceTemplateType`、`schedule.h:25-42` 的
`TilingCase`、`axis_group.h:19-24` 的 `AxisGroup`：

| 模板 | 分核轴 | tile 轴来源 | 实测 `vectorized_axis` 根数 |
|---|---|---|---|
| **RCommon** | 1 根（A 轴） | `y_group` + `r_group` + `n_group`，经调度器合轴后归一 | **4 / 3 / 3 / 2**（四个 tiling case，见 4.2.2） |
| **RCore** | **2 根**（A 轴 + R 轴，`reduce_is_block = true`） | 同上 | **4 / 3 / 3 / 2**；分核轴是 A 与 R 的 TileOuter **先合并再 block 切**，不是两根独立块轴 |
| **AllLoad** | 1 根 | `y_group` + `r_group`（**不含 `n_group`**，见 `schedule.cpp:714`） | 轴数更少，落进同一套逻辑，无额外要求。**2026-10-01 实测证实**：二维 AR 是 `[A_inner, R]` 2 根；三维 ARA 是 `[R, A_inner, A_outer]` 3 根（因此判成 **RA**）。见 [AllLoad 模板设计](2026-10-01-afir-reduce-allload-template-design.md) 2.3 |

`TilingCase` 还有 `ub_tiling_id_x` / `_y` / `_r` **三个** UB 切分位，所以
"tile 轴最多两根"从一开始就不成立。

**四个 tiling case 必须全部可下降**，这是本方案的硬约束（不是"能跑通一个就行"）。
它直接决定了 `> 2 根` 不能报不支持——四个候选里只有一个是 2 根。这一条把
「在二维 DMA 外套外层循环」从已知债务提到了本期范围内（S4）。

## 1.2 范围

### 包含

- `TransferPlan` 的计划结构由四个标量轴改为 **`TransferDim` 数组**（无轴 id，
  与 legacy `DataCopyParams` 同构，见 4.1）
- GM 偏移改为 **`axis \ vectorized_axis` 上的求和**，对循环嵌套层数不作假设，
  也不做 `splitPairOtherId` 反查（见 4.2.1）
- **有效视图**：零轴剔除、extent-1 剔除、**连续性合并**、不跨 TileInner 边界
  （四条规则照搬 legacy，见 4.2.2）
- **> 2 根有效轴时在 DMA 外套 `scf.for`**（legacy `CreateOuterFor` 的等价物，见 4.2.6）
- `ScheduledTileIndices` 扩展为 `DataCopyPadExtend` 的四个参数，**按空间命名**
- **符号闭合门禁**：stride 的自由符号必须 ⊆ `asc_graph.size_var`（见 4.2.5）
- 计算算子（elementwise）在二维 tile 上显式拒绝
- 归约算子可显式选择"局部侧"取哪张 descriptor
- 四项寻址债务清理

### 不包含

| 项 | 理由 |
|---|---|
| legacy 的运行期 `need_swap` 二选一 | `CreateDmaCall` 对 > 2 根发**两份** DMA，套在 `if (ActualSize(倒数第2) < ActualSize(倒数第3))` 里选 blockCount。那是纯性能选择，不影响正确性；本期只发 `need_swap = false` 一版，代码体积减半。记为 L7 |
| 二维 elementwise | 见 3.2.3；接通方式已定（线性跨度），不在本期 |
| `afir.sum` 的 RA 下降（`reduce_init` 与累加长度） | AR 的多轴合并已为关闭 S4 数值验收补齐；RA 仍是独立需求，与寻址正交 |
| 910B 的 `DataCopyPadExtend` 判定 | 已修复并由 910B tikicpulib API UT 验证，见文首依赖 |
| bridge 改动 | **一行都不需要**，见 2.4 |
| TilingData schema / `size_var` 的扩充 | **不需要**。符号闭合已实测成立，见 4.2.5 |

### 涉及组件

Autofuse Codegen/Backend（MLIR 子树）。不涉及 SuperKernel、Python 绑定、打包交付。

# 2 总体概述

## 2.1 软件概述

### 2.1.1 项目介绍

`TransferPlan` 是 `dialect/AFIR/Analysis/` 下的纯分析组件，回答两个问题：

1. **这个搬运算子的形态支不支持**（`buildTransferPlan`，只读属性、不建 IR）
2. **这一轮搬哪一段**（`materializeScheduledTileIndices`，建 `arith` IR。本方案
   改名为 `emitScheduledTransfer` 并改成发射器，见 3.5.1）

两相分离是因为第一个函数每个算子被调用两次：一次在 `queryLowering`（能力查询，在
任何 IR 之前），一次在改写。能力查询阶段失败的含义是"**这个形态本期不支持**"，
而不是"算错了"。

当前实现隐含**六条前提**，elementwise 全部满足，reduce 切 R 轴后一条都不成立。
下表的 reduce 实况取自 2026-09-21 的 ARAR 调度探针（`Construct_Reduce_ARAR`，
RCommon case[0]，完整数据见 4.2.2）：

| 前提 | 代码位置 | reduce 的实况 | 失败形态 |
|---|---|---|---|
| 只有一根向量化轴 | `getVectorizedAxis()` 取 `.front()`，**全仓无 size 检查** | `vectorized_axis = [z0t, z2, z1t, z3]` 四根 | **静默**取走最外一根，丢掉三根 |
| 只有一层循环 | `findLoop(op, plan.loopAxis)` | 四根轴挂在**两棵**循环子树下（`z0T` 与 `z1T`） | 另一棵树的归纳变量进不了偏移 |
| `loop_axis` 恰好是 block-inner 轴 | `splitPairOtherId == loopAxis && type == BlockOuter` | RCore 的块轴是 `z0Tz1TTB`（A 与 R 的 TileOuter **合并后**再切），`AxisType` 是 `Merged` 派生，不与任何 tile 轴构成 split 对 | `failure()` |
| 外层轴靠 `splitPairOtherId` 反查得到 | `TransferPlan.cpp:123-130` | 同上，一对一反查在合并轴上不成立 | `failure()` |
| GM 偏移是单项，且不乘 stride | `globalOffset = tileIndex × tileSize` | `z0t` 的 GM stride 是 `s1*s2*s3`（不是 1），且还缺 `z1T` 一整项 | **静默**少乘一个因子 |
| 局部侧＝有结果取 result / 无结果取 operand | `TransferPlan.cpp:88` | 归约的输入与 destination **都在 UB 上**；`b0_max` 结果侧 `vect_strides = [s2, 1, 0, 0]`，Load 侧不是 | 按结果侧建计划会丢掉两根零轴 |

**两种失败形态要分开看。** `failure()` 是可诊断的；`.front()` 与"偏移不乘 stride"
是**静默错值**。S1 的第一件事就是把前者补成检查，让 reduce 从"静默搬错"变成
"明确拒绝"。

另有一条不属于"前提"、但同样没有表达处的事实：**UB 轴序与 GM 轴序不一致**
（case[0] 的 UB 序是 A A R R，GM 排布是 A R A R），见 4.2.6。

### 2.1.2 产品环境介绍

- **AscendC**：二维搬运通过 `autofuse/ascendc/api/datacopy.h` 的
  `DataCopyPadExtend(dst, src, block_count, block_len, src_stride, dst_stride)`
  发射，落到 `AscendC::DataCopyPad`。该 wrapper 在 mem-based 与 reg-base 两张
  供给表上同名同签名。
- **目标能力**：由 `TargetFeature::DataCopyPadExtend` 控制。V2201
  （`Ascend910B1`）与 V3510（`Ascend950DT_950x`）均为 `Available`；910B 的历史误判
  已修复，见文首依赖。
- **不新增**任何 `rt*` / `aclrt*` 调用，不改变 kernel 入口 ABI。

## 2.2 软件功能

1. 从局部侧 `TensorDescAttr` 的两张步长表归约出**有效视图**（N 项
   `{extent, globalStride, localStride}`）
2. 按 `Σ_{a ∈ axis \ vectorized_axis} 轴索引(a) × strides[a]` 物化 GM 偏移
3. 最内 ≤2 项按"缝隙 = 外层 stride − 本次实际长度"物化 `blockCount` /
   `blockLen` / 两侧 stride；其余每项发一层 `scf.for`
4. 形态不支持时在能力查询阶段报 `UnsupportedShape`，不进入改写
5. 计算算子遇到二维 tile 时拒绝而非只算第一行

## 2.3 设计约束

| 约束 | 来源 | 影响 |
|---|---|---|
| `buildTransferPlan` 必须是纯的 | 能力查询在 IR 之前调用它 | 不得建任何 op |
| **descriptor 的 `axis` 长度必须等于 memref rank** | `AFIRVerifierUtils.cpp:113`，bridge 侧是 `MemRefType::get(v.tensor.shape, ...)`（`AfirBuilder.cpp:154`） | 多轴 descriptor 只能挂在同 rank 的 memref 上。构造测试夹具时**必须同时**给对 memref rank 与 `indexing_maps` 的维数，否则在 op verifier 就被拦住，到不了 `TransferPlan` |
| **向量化轴没有 `actual_size` 绑定** | `LoopGenerate` 只对它**生成循环的那根轴**发 `ActualSize`（`:543`、`:594`，用作循环上界）。2026-09-21 实测一个 block+tile 切分的图，活到相位 C 的是 `axis=4(循环轴) actual_size` 与 `axis=2(向量化轴) tail_size`——**向量化轴只有 tail，没有 actual** | tile 长度必须按 `{axis_size, tail_size, 外层序号}` **现算**，没有现成绑定可查。block 的 `axis_index` 则是有的，偏移可以直接用 |
| stride 可能是符号表达式 | UB 行距形如 `8 × Ceiling(z1t_size / 8)` | 转换里要带符号下降；`MemoryAllocate` 之后流水线不再有符号下降，`afir.symbolic_eval` 必须在本次转换内消掉 |
| 计划拿不到搬运方向 | Load / Store 共用同一个函数 | DMA 参数**必须按空间命名**，方向映射放发射侧 |
| CPU E2E 只有 910B1 | `AF_MLIR_E2E_CPU_SOC_VERSION`、`tikicpulib_ascend910B1` | 910B 判定修复后此约束从"阻塞"变成"**910B 是唯一能验数值的目标**"。二维与 > 2 维的正确性最终靠它，950 一路只有文本比对 |
| 禁止硬编码芯片类型 | `编码红线.md` Graph-autofusion 规则 2 | 能力判断只能走 `TargetProfile`，不得在 conversion 里写 soc 字面量 |

## 2.4 假设和依赖关系

### 已验证的前提

- **bridge 不需要任何改动。** `AfirBuilder::makeTensorDescAttr`
  （`AfirBuilder.cpp:214-222`）没有任何 `node.type` 分支，四张寻址表
  （`axis`/`strides`、`vectorized_axis`/`vectorized_strides`）**原样抄自调度器
  DTO**，对所有节点一视同仁。reduce 的两项 `vectorized_axis` 早已在 IR 里，
  今天没用上纯粹是因为 `TransferPlan` 只读了 `.front()`。
- **调度循环的 tile 起点不需要 `localOffset`。** `8fcd2ebe` 把它从 `ScheduledTileIndices` 删除，
  理由是队列按 slot 自己轮转、TBuf 按 `b{id}_size` 只装一个 tile。二维下 tile 的
  起点仍是 buffer 基址，**UB 偏移仍为 0**。
- **lit 可以钉目标。** `af-opt --target=<soc>` 已在多条既有 lit 中使用，二维的 IR 与
  发射文本可以按目标分别验证。**两个目标都要验，不是二选一**：V2201（910B）走
  mem-based 供给表、V3510（950）走 reg-base，`DataCopyPadExtend` 在两张表上同名
  同签名但不是同一份实现（2.1.2）。只验 V3510 会把 910B 的 mem-based 二维发射
  整条路留空。
- **线性跨度公式已存在。** `CanonicalMaterialization.cpp:330` 的
  `materializeQueueMemberElementCount` 就是 `Σ(repeat−1)×stride + 1`，且用的是
  checked 算术。二维 elementwise 接通时复用它，**不要重写**。
- **真实 ARAR 的调度产物已实测**（2026-09-21，`optimize_ut` +
  `Construct_Reduce_ARAR`）：RCommon 与 RCore 各 4 个 tiling case，
  `vectorized_axis` 分别是 4 / 3 / 3 / 2 根。数据见 4.2.2。
- **符号闭合已实测成立**（同一次探针）：八张图的 `strides` 与
  `vectorized_strides` 的自由符号**全部**落在 `asc_graph.size_var` 里，
  一个都不缺。机制与后果见 4.2.5。

### 待确认

- **pad 语义差异是否造成实际错值**：`DataCopyPad`（`isPad = true`）与
  `DataCopy` 在非 32B 对齐尾巴上的行为差异未实测。记在前置依赖文档 §4.2。

### 外部依赖

- 910B 的 `DataCopyPadExtend` 判定**已修复并验证**（见文首链接），本方案不再有外部阻塞项。

# 3 需求分析与设计

## 3.1 整体介绍

数据流不变，仍是两相：

```
buildTransferPlan(op, side)                 纯分析，只读属性
   ├─ getMaps            indexing_maps 不存在，或恰好两条且维数/符号数一致
   ├─ getLocalDescriptor 按 side 取局部侧 TensorDescAttr
   ├─ checkSymbolClosure  两张 stride 表的自由符号 ⊆ asc_graph.size_var
   └─ reduceEffectiveView 四条规则归约出 N 项 {extent, gmStride, ubStride}
          ↓ TransferPlan{kind, dims[1..N]}
emitScheduledTransfer(op, builder, plan, emitBody)      建 IR
   ├─ globalOffset = Σ_{a ∈ axis \ vectorized_axis} index(a) × strides[a]
   ├─ dims.size() > 2 → 外层每项发一层 scf.for，两侧各自加 Σ i × stride
   └─ 最内 ≤2 项 → blockCount / blockLen / globalStride / localStride
          ↓ ScheduledTileIndices（在最内层）
DataMovement          按方向映射两个 stride，发 DataCopyPadExtend
Unary/BinaryElementwise   dims.size() != 1 时拒绝
```

两处关键改动：

1. **偏移不再靠配对反查，而是集合差求和**——`axis` 表里不在 `vectorized_axis`
   里的轴，就是 tile 外面还剩的循环层，每根乘自己的 stride。合并块轴自动被覆盖。
2. **物化函数从"返回值"变成"发射器"**——>2 根要建 `scf.for` 嵌套，循环体内的
   偏移依赖归纳变量，必须在循环内算，扁平返回值撑不住。

## 3.2 功能需求

### 3.2.1 功能需求 1：有效视图取代标量轴

**1）介绍**

把计划从"一根向量化轴 + 一层循环 + loop_axis 是 block-inner"这组前提中解开，
改为从局部侧 descriptor 的两张步长表归约出一个**有效视图**。

**关键取舍：有效视图里不带轴 id。** 因为连续性合并之后，一个表项的 extent 是
若干根轴的乘积，它不对应任何一根调度轴，没有 id 可填。legacy 早就承认了这一点
（`api_call_utils.h:20` 的 `DataCopyParams` 是三条并行的 `SizeExpr` 数组，
一个轴 id 都没有）。本方案与之同构。

**2）输入**

| 来源 | 字段 | 用途 |
|---|---|---|
| 局部侧 `TensorDescAttr` | `vectorized_axis` / `vectorized_strides` | tile 的轴序与 UB 步长 |
| 同一张 descriptor | `axis` / `repeats` / `strides` | 同一根轴的范围与 GM 步长 |
| `func` 的 `afir.asc_graph` | `size_var` | 符号闭合门禁（4.2.5） |
| 算子的 `afir.sched` | `loop_axis` | 仅作「这个算子被调度过」的门（`>= 0`），不进计划 |

**不再需要** `asc_graph.axes` 的 `splitPairOtherId` / `from`——配对反查整段删除。

非法输入：
- `vectorized_axis` 与 `vectorized_strides` 长度不等 → `failure()`
- 某根轴不在 `axis` 表里（查不到 GM stride / repeat）→ `failure()`（契约破损）
- stride 的自由符号不在 `size_var` 里 → `failure()`（见 4.2.5）

**3）处理**

**从内向外**扫一趟（顺序与 legacy `CalculateDmaParams` 一致，四条归约规则
见 4.2.2）：

```
for axisId, ubStride in reversed(zip(vectorized_axis, vectorized_strides)):
    pos      = indexOf(descriptor.axis, axisId)        # 缺失 → failure()
    gmStride = descriptor.strides[pos]
    extent   = descriptor.repeats[pos]

    # 规则 1：extent == 1 的轴不占维度
    if isOne(extent): continue

    # 规则 2：两侧 stride 均为 0 且满足 ShouldIgnoreDataCopyZeroAxis 时跳过
    if isZero(ubStride) and isZero(gmStride) and shouldIgnoreZeroAxis(...): continue

    # 规则 3：与已积累的内层连续则合并（三值判定，判不出就不合并）
    if canMerge(dims.back(), gmStride, ubStride): dims.back().extent *= extent; continue

    # 规则 4：不跨 TileInner 边界合并
    dims.push({extent, gmStride, ubStride})

reverse(dims)                    # 回到外 -> 内
if dims.empty(): return failure()
```

**没有上限判据。** `dims.size() > 2` 不再是失败条件——超出的部分由 4.2.6 的
外层循环承接。这是"四个 tiling case 必须全部可下降"（1.1.2）的直接后果。

**4）输出**

`TransferPlan{kind = Scheduled, dims[1..N]}`。一维输入退化成
`dims.size() == 1`，与改动前语义等价。

---

### 3.2.2 功能需求 2：运行期偏移与 DMA 参数

**1）介绍**

把 GM 偏移从单项式换成求和式，并补齐 `DataCopyPadExtend` 的四个参数。

**2）输入**

`TransferPlan`，以及 IR 中由 `LoopGenerate` 留下的 `#afir.axis_binding` 标记和
函数入口的 `afir.tiling_param`（符号绑定来源）。

**3）处理**

**第一步：GM 偏移 = 集合差求和。**

```
globalOffset = Σ_{a ∈ descriptor.axis, a ∉ descriptor.vectorized_axis}
                   axisIndex(a) × materializeRuntimeSize(descriptor.strides[a])
```

`axis \ vectorized_axis` 恰好就是「tile 外面还剩几层循环」。case[0] 的
`axis` 是 7 根、`vectorized_axis` 是 4 根，差集是 `[z0TB, z0Tb, z1T]`——
block outer、block inner、R 的 TileOuter，一根不多一根不少。

`axisIndex(a)` 从 `#afir.axis_binding` 取，只有两种：

| 轴形态 | 取法 | `LoopGenerate` 的产生点 |
|---|---|---|
| 有 `scf.for` | 沿 parent chain 找 `AxisBinding<Loop, a>` 的循环，取归纳变量 | `:554` |
| `BlockOuter`（不产生 `scf.for`） | `findAxisValue(a, AxisIndex)` | `:525-531` 的 `RemSIOp(blockIndex, loopSize)` |

**这个式子的四条性质，每一条都对应上面破掉的一条前提：**

- **不做反查**，因此合并块轴（RCore 的 `z0Tz1TTB`）天然被包含——它在 `axis` 里、
  不在 `vectorized_axis` 里，自动贡献一项，用它自己的 stride
- **不假设嵌套层数**，几层就求几项
- **不假设 tile 连续**，每根轴乘自己的 stride
- 与 legacy `tiler.Offset(current_axis, axis, axis_strides)` 同构——legacy 的
  `current_axis` 就是这个集合差

**一维下它与现有式子数值相等，但发射的 op 序列不同。** 2026-09-21 用
`afir-to-ascendc-scheduled-transfer.mlir` 实测的当前产物是：

```mlir
%60 = arith.muli %54, %5     // blockIdx × zTb_size
%61 = arith.addi %60, %arg4  // + 归纳变量
%62 = arith.muli %61, %3     // × zt_size        —— 3 个 op
```

现有实现做的是 **Horner 因式分解**：它利用了 `stride(zTb) = zt_size` 且
`stride(zTB) = zTb_size × zt_size = stride(zTb) × axis_size(zTb)` 这个嵌套关系，
把两项合并成一次外提的乘法。

新公式不能这么提——**能提恰恰是因为假设了 stride 嵌套，而丢掉这个假设正是本方案
的目的**（reduce 的 GM stride 在 vectorized 序下非单调，见 4.2.6）。所以它发：

```mlir
%s  = afir.symbolic_eval %5, %3 {expr = a*b}   // stride(zTB)，循环不变量，外提
%a  = arith.muli %54, %s
%b  = arith.muli %arg4, %3
%o  = arith.addi %a, %b                         //  循环体内 3 个 op + 1 个外提
```

循环体内多一次乘法。**因此 S2 的验收不能是逐字节比对**，判据见 8.2。

**第二步：每项有效轴的运行期范围。**

```
extent_rt = materializeActualSize(builder, func, op, extent)
```

一根 tile 轴的运行期范围是它的 `ActualSize`（整块取 `axis_size`、尾块取
`tail_size`）。这段逻辑今天有**两份**——`LoopGenerate.cpp:541-543` / `:591-594`，
和 `TransferPlan.cpp:220-227` 手搓的那份。本方案提成公共 helper：

> **本段已被 S2 的实测推翻，保留原文供对照，结论见 8.2.1 偏差三。**
>
> ~~**必须现算，没有绑定可查。** `LoopGenerate` 只对**生成循环的那根轴**发
> `AxisValueRole::ActualSize`（用作循环上界）；向量化轴是 tile 内部，不产生循环，
> 所以它只有 `tail_size` 没有 `actual_size`（2.3 的约束表，已实测）。~~
>
> ~~因此 `TransferPlan.cpp:220-227` 那段尾块 select **不是冗余**，它是必要的。~~

**订正**：`LoopGenerate` 的 `generatePairedInnerActualSize` **确实**为配对 inner 发
`ActualSize`，向量化轴有这个绑定。此前之所以「查不到」，是因为它在
memory-allocate 的一次全函数贪心重写里被当作死代码删掉了——那时还没有消费者，而
`#afir.axis_binding` 不构成使用。堵住那个 DCE 点之后，尾块 select 整段可以删掉，
换成一次绑定查询。两者逐值相等：`loop_size = axis_size + (tail != 0)`，所以
「序号 == loop_size − 1 且有尾巴」与「序号 >= axis_size」是同一个条件。
`materializeActualSize` 的职责是把这段逻辑收成一处并支持多项，而不是替换成查表：

```
extent_rt = select(是最后一轮 ∧ tail != 0, tail_size, axis_size)
```

「是最后一轮」需要该轴的外层序号与 loop_size，两者都有绑定。

**第三步：DMA 四参（最内 ≤2 项）与外层循环（其余项）。**

```
blockLen     = dims.back().extent_rt
blockCount   = dims[-2].extent_rt   若存在，否则 1
globalStride = dims[-2].globalStride - blockLen   若存在，否则 0
localStride  = dims[-2].localStride  - blockLen   若存在，否则 0
```

`dims.size() > 2` 时，外层每项发一层 `scf.for`，两侧各自累加
`Σ outer_i × stride_i`，见 4.2.6。

缝隙单位是元素个数，32B 换算由 `DataCopyPadExtend` 自己做，与 legacy
`SetDmaParams` 同一套量纲。

**stride 的物化走 `materializeRuntimeSize`**（`SymbolicBuilder.cpp:106`），
不经 `SymbolBindingMap` 参数。它自己抽自由符号、对每个符号
`getOrCreateTilingParam`，合法性由 4.2.5 的闭合性保证。附带好处：它把值插在
`sizePrologueEnd(entry)`，stride 是循环不变量，天然被提到函数入口。

**4）输出**

```cpp
struct ScheduledTileIndices {
  Value globalOffset;
  Value blockCount;
  Value blockLen;
  Value globalStride;   // GM 侧行间缝隙
  Value localStride;    // UB 侧行间缝隙
};
```

**按空间命名而非 `src`/`dst` 是硬要求。** 理由见 4.2.3。

这是 S4 前的一/二维接口：UB tile 起点为 0。S4 已在结构体恢复 `localOffset`，只承载 tile 内额外行循环的偏移，见 8.4.1。

---

### 3.2.3 功能需求 3：计算侧显式拒绝二维

**1）介绍**

`UnaryElementwise` 与 `BinaryElementwise` 也是 `emitScheduledTransfer` 的消费者，
把返回的长度直接当向量长度用。二维下一次调用覆盖不到整块。

**2）输入**

同 3.2.1。

**3）处理**

```cpp
// 二维 tile 的计算算子要按行展开（外层 stride ≠ 内层长度时一次 count 盖不住），
// 本期只有搬运侧支持二维。不拦住就会只算第一行。
if (plan->dims.size() != 1) {
  return failure();
}
```

**4）输出**

能力查询返回 `UnsupportedShape`。

**这一条必须与 3.2.1 同一次提交**——单独放开计划结构会让二维 tile 流进计算侧，
只算第一行且无诊断。这是本方案里唯一一个"中间状态比两端都危险"的位置。

**已知后果**：融合了 elementwise 的 reduce 图本期会被拒
（`notepad/reduce.md` 里的 `x = elewise_inplace(x)` 就是这个形态）。这是正确的
失败形态：报不支持，不算错值。

---

### 3.2.4 功能需求 4：局部侧的显式选择

**1）介绍**

"哪一侧是局部侧"今天按"有结果取 result_descs、无结果取 operand_descs"推断。
那是**搬运**的规则，归约不适用。

**2）输入**

调用方传入的 `TransferPlanSide`。

**3）处理**

```cpp
enum class TransferPlanSide { Automatic, Operand };
FailureOr<TransferPlan> buildTransferPlan(Operation *op,
                                          TransferPlanSide side = TransferPlanSide::Automatic);
```

`Automatic` 维持原规则，搬运侧零行为变化。归约显式传 `Operand`：它的输入与
destination **都在 UB 上**，而高阶 API 看到的二维形状是**输入**那一侧的 `[A, R]`；
destination 在被归约轴上 stride 为 0，按结果侧建计划会被 3.2.1 的零轴规则丢掉。

探针实测佐证（RCommon case[0]）：同一张图里 Load 侧
`vect_strides = [8⌈s3/8⌉·s2·z1t, 8⌈s3/8⌉·z1t, 8⌈s3/8⌉, 1]` 四项全非零，
而归约节点 `b0_max` 的结果侧是 `[s2, 1, 0, 0]`——`z1t` / `z3` 在结果侧**是**零轴、
在 Load 侧**不是**。两侧不是同一个视图，所以"哪边是局部侧"必须由调用方说。

**4）输出**

不进 IR。"哪边是局部侧"取决于谁在问，不是张量的固有属性，归属在调用点。

顺带把 `findAxisValue` 提升为公开接口：归约要用同一套取法拿被归约轴的
`axis_size`，不必另写一份。

---

### 3.2.5 功能需求 5：寻址债务清理

| 项 | 位置 | 处理 | 理由 |
|---|---|---|---|
| `isQueueBuffer` | `TransferPlan.cpp:231`、`.h:48-58` | **删除** | 零调用者。且头文件注释写着"多 tile 的本地侧接通时它是第一个要回来的取用点"——**这个预期是错的**：多 tile 本地侧需要的是 stride 表，不是容器判别。留着会让下一个接二维的人照它再按容器分叉一次 |
| `StorageView::indexingMaps` | `CanonicalBindings.cpp:147` | **已删除**（S5） | 写了从不读 |
| `getValueMemorySpace` 只读 descriptor | `AscendCAttributes.cpp:44-50` | **已对齐**（S5）：先读 type 再回落 descriptor | 与同仓的 `resolveValueSpace`（`CanonicalBindings.cpp:108-113`）是两套判据 |
| map 与 descriptor 无一致性校验 | `TransferPlan.cpp:31-43` | **记为 L4**（S5 查证后的结论，见 8.5） | `isDirectProjection` 接受 rank-2 置换。但 S5 查清 `indexing_maps` 是 bridge 从两侧 descriptor 的 `axis` 表**派生**出来的，不是独立事实源——补 verifier 等于把那段派生复刻成两份。真正的修法是把派生收进 dialect 由 bridge 调用 |

`.front()` 静默截断与"GM 偏移不乘 stride"两条**随 3.2.1 消失**，不单列。

> `.front()` 那条值得单独说明危害：`materializeQueueMemberElementCount` 按
> `getVectorizedAxis().size()` 循环算 buffer 大小，而寻址只取第一根。二维
> descriptor 进来就是**分配按 N 维、寻址按 1 维**，静默取错地址。全仓没有任何
> size 检查。

## 3.3 非功能需求

### 3.3.1 可维护性

- **一个公式，换一张表**：GM 与 UB 的差别只体现在取 `globalStride` 还是
  `localStride`，不再有两套偏移逻辑。
- **删多于加**：本方案净删掉四个计划字段、一整段 `splitPairOtherId` 反查、
  一份手搓的 `ActualSize` 重算，新增的只有 `TransferDim` 三字段与一个发射器。
- **按空间命名贯穿到底**：`globalOffset` / `globalStride` / `localStride`。发射侧
  是唯一知道方向的地方。
- **失败形态单一**：形态不支持一律在 `buildTransferPlan` 返回 `failure()`，由
  能力查询映射成 `UnsupportedShape`。不新增"静默降级"路径。
- **不提前实现**：不为 transpose / concat / gather 预留钩子。没有使用者的抽象
  会固化一次猜测。

### 3.3.2 可测试性

`TransferPlan` 是**纯分析函数，只读属性、无目标依赖**，可直接单元测试。

**今天 `test/ut/dialect/` 下九个 UT 没有一个覆盖 `TransferPlan`**，它全靠 lit
间接覆盖——而它正是本方案改动量最大的地方。因此：

- **新增 `test/ut/dialect/TransferPlanTest.cpp`**，照 `ReducePlanTest.cpp` 的形式
- 这是 S1/S2 最重要的产出：序号来源与剔除规则都是纯符号判断，UT 可以完全覆盖，而它们写错的表现是静默搬错地址或静默少搬一维

详见第 7 章。

### 3.3.3 可移植性

- 能力判断全部经由 `TargetProfile`，conversion 层不出现 soc 字面量或
  `NpuArch::`（`编码红线.md` Graph-autofusion 规则 2）。
- 二维发射依赖 `TargetFeature::DataCopyPadExtend`。该 feature 在 V2201 与 V3510 上
  **都可用**，V2201 的历史误判已修复（见前置依赖）。
- **两个目标都是一等目标，reduce 不是 950 专属能力。** 910B 还有一层特殊性：
  它是**唯一**能跑 CPU E2E 的目标（`tikicpulib_ascend910B1`），所以二维搬运的
  **数值**正确性只能在 910B 上验——950 一路只有 IR 与文本比对，验不了值。
- `DataCopyPadExtend` 在 mem-based（V2201）与 reg-base（V3510）两张供给表上同名
  同签名，但**不是同一份实现**。API 层两条路的双向 / 多 block / 非零 stride 已由
  910B tikicpulib UT 覆盖（见前置依赖 §6）；本方案要补的是 MLIR 下降侧在两个
  目标上各发一遍。
- 不依赖特定 CANN 版本的新 API：`DataCopyPadExtend` wrapper 已在仓库内，
  legacy 一直在用。

### 3.3.4 可靠性

- **整数安全**：`globalOffset` 的乘加、以及缝隙的减法都发生在 `index` 域。
  缝隙理论上非负（外层 stride ≥ 本次实际长度），但这依赖调度器的正确性，
  不是本层能证明的——**不在运行期插断言**，与既有 GM 偏移的处理保持一致；
  若后续要收紧，应整体改用 `arith::*Extended` + `cf.assert`，与
  `CanonicalResources.cpp` 同一写法，而不是在这里单点加。
- **资源生命周期**：不新增任何资源申请/释放，不触碰队列与 TBuf 的生命周期。
- **确定性**：归约的遍历顺序取自 `vectorized_axis` 的数组序（内 → 外扫，
  再翻回外 → 内），偏移的求和顺序取自 `axis` 的数组序。两者都是属性里的固定数组，
  同一输入产出同一计划。

### 3.3.5 特性交叉影响

已读取 `docs/guidelines/cross_feature_check.md`，逐项结论：

| 场景 | 适用性 | 分析说明 |
|---|---|---|
| SuperKernel Python 接口 | 不适用 | 改动全在 `autofuse/mlir/`，不触及 `super_kernel/` |
| SuperKernel C++/AOT 接口 | 不适用 | 不改对外符号、结构体、动态库 |
| Autofuse 图优化 | 不适用 | 不改 ASCIR 图结构与融合 pass；本方案只读调度器已写入的属性 |
| **Autofuse Codegen/Backend** | **适用** | 核心改动区。`TransferPlan` 与三个 implementation。一维产物要求逐字节不变（见 8.1），二维为新增形态 |
| **AscendC API / Runtime 交互** | **适用** | 发射 `DataCopyPadExtend`（落到 `AscendC::DataCopyPad`）。不新增 `rt*` / `aclrt*`。参数量纲与 legacy `SetDmaParams` 对齐；方向映射见 4.2.3 |
| Python/C++ 混合绑定 | 不适用 | 不触及 `autofuse/compiler/py_module/` |
| 构建与打包 | 不适用 | 910B 判定修复不改 `AddCpuCase.cmake`；CPU E2E 沿用扩展头内联机制，本方案不改构建 |
| **测试与覆盖率** | **适用** | 新增 `TransferPlanTest.cpp`、二维 Store lit 与外层循环 lit（**两个目标各一条 RUN 行**）、910B CPU E2E 数值验证、变异验证。详见第 7 章 |
| 性能与日志 | **部分适用** | 见 3.4。不新增任何日志 |
| 兼容性 | **适用** | `TransferPlan.h` 是内部头，无对外 ABI；但结构体字段改名会影响所有调用点，见第 6 章 |

## 3.4 性能

### 3.4.1 编译时长

`buildTransferPlan` 对每个搬运/计算算子调用两次，改动后每次多做的事是：

- 按 `vectorized_axis` 长度（实测 1～4）循环，每项在 `descriptor.axis` 上做一次
  线性查找取 stride / repeat
- 每项两次 `StaticCheckEq`（三值符号判定）用于零轴与连续性
- 一次符号闭合判断：两张 stride 表的自由符号 vs `size_var`，都是十几个元素的小集合

**净减，不是净增。** 换掉的是「在 `asc_graph.axes` 上遍历全表找 `splitPairOtherId`
配偶」——那是对**整张轴表**的线性扫描，而新实现只在 `descriptor.axis`（更短）上
按 id 查找。`asc_graph.axes` 的遍历整段删除。

### 3.4.2 执行性能

- **一维**：产物要求逐字节不变，执行性能无变化。
- **二维**：新增形态，此前无法表达。与 legacy 对比，发射的是同一个
  `DataCopyPadExtend` 调用、同一套参数量纲，预期性能等同。
- **> 2 项**：每多一项多一层 `scf.for`，与 legacy `CreateOuterFor` 同形。
  相对 legacy 少了运行期 `need_swap` 的二选一（L7），在"倒数第三根远大于倒数
  第二根"的形状上会比 legacy 慢；实测再补。
- 新增的 `arith` 运算全部在地址计算上，相对一次 DMA 可忽略；stride 为 1 时省略
  乘法这一条能消掉一维路径上全部新增算术。

### 3.4.3 内存和产物大小

`TransferPlan` 由 4 个 `int64_t` 变为两个动态序列：tile 外层的 `outerAxes` 与 tile
内部有效视图的 `dims`。两者均使用 `SmallVector<..., 0>`，不把当前测试 corpus 的
轴数固化成内联容量；项数由 scheduler 产物决定，没有长度上限。计划生命周期是单次分析调用。

生成 kernel 的源码体积：一维不变；二维每次搬运多几条地址运算；> 2 根每多一根
多一层 `scf.for`。**不发 legacy 的 `need_swap` 双份版本**（1.2），这一条让
> 2 根路径的源码体积相对 legacy 减半。

## 3.5 接口设计

### 3.5.1 新增/修改接口描述

全部位于 `dialect/AFIR/Analysis/TransferPlan.h`，**内部头，无对外 ABI**。

| 接口 | 变更 | 说明 |
|---|---|---|
| `enum class TransferPlanKind` | `ScheduledContiguous` → `Scheduled` | "Contiguous" 是一维的性质，二维下是错名 |
| `struct TransferDim` | **新增** | `{extent, globalStride, localStride}`，三个 `ExprAttr`。**没有轴 id**——合并后的项不对应任何一根调度轴（见 4.1） |
| `struct TransferPlan` | 四个标量轴 → `SmallVector<TransferOuterAxis, 0> outerAxes` + `SmallVector<TransferDim, 0> dims` | **四个字段全删**：`vectorizedAxis` / `loopAxis` / `blockOuterAxis` / `tileOuterAxis` 在新偏移式下都没有读者。两个序列均无内联容量和长度上限 |
| `struct ScheduledTileIndices` | `{globalOffset, count}` → `{globalOffset, localOffset, blockCount, blockLen, globalStride, localStride}` | `count` 更名 `blockLen`；`localOffset` 是 S4 加的，装外层循环的 UB 侧行内偏移（见 4.1 与 8.4.1） |
| `materializeScheduledTileIndices` | → `emitScheduledTransfer(op, builder, plan, emitBody)` | 从"返回值"变"发射器"：> 2 根要建 `scf.for` 嵌套并在最内层调 `emitBody`（见 3.1） |
| ~~`materializeActualSize`~~ | **未实施**（S4 偏差，见 8.4.1） | 草案要的是「按轴查实际尺寸」，而合并之后一项不对应任何一根轴，这个签名写不出来。实际用的是 S3 已有的按符号替换（`collectExtentBindings`），它对合并项同样成立 |
| `enum class TransferPlanSide` | **新增** | `{Automatic, Operand}` |
| `buildTransferPlan` | 增加默认参数 `side = Automatic` | 既有调用点不变 |
| `findAxisValue` | 由匿名命名空间提升为公开 | 供归约与 `materializeActualSize` 复用 |
| `isQueueBuffer` | **删除** | 零调用者，见 3.2.5 |

**`emitScheduledTransfer` 的签名变化是本方案唯一的结构性改动**，三个消费者都要
改调用形式。`TransferDim` 反而比它替换掉的 `TransferTileAxis` 草案更小：后者
的 `tileOuterAxis` / `blockOuterAxis` / `blockInnerAxis` 三个字段只为逐轴算偏移
服务，而偏移不再那么算。

### 3.5.2 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|---|---|---|---|
| 接口说明 | 是否需要接口评审 | 否 | 内部分析头，不对外 |
| 接口说明 | 是否需要补充文档 | 是 | 本文档 + 头文件注释 |
| 接口兼容 | 行为是否兼容 | 是 | 一维行为要求逐字节等价，由 8.1 守住 |
| 接口兼容 | ABI/API 是否兼容 | 否 | 非导出接口，随库一起编译 |
| 接口约束 | 约束不满足时是否清晰报错 | 是 | 一律 `failure()` → `UnsupportedShape`，不静默降级 |
| 接口测试 | 是否需要独立接口用例 | 是 | 新增 `TransferPlanTest.cpp` |

# 4 软件设计

## 4.1 关键数据结构

```cpp
enum class TransferPlanKind { Identity, Scheduled };

// 有效视图的一项。**没有轴 id**：连续性合并之后一个表项的 extent 是若干根轴的
// 乘积，它不对应任何一根调度轴，没有 id 可填。与 legacy `DataCopyParams`
// （`api_call_utils.h:20` 的三条并行 `SizeExpr` 数组）同构。
//
// 两侧 stride 相等时这一项与更内层连续（已被规则 3 合并掉），不等时要靠 DMA 的
// blockCount / stride，或靠外层 `scf.for` 跨过缝隙。
struct TransferDim {
  ExprAttr extent;
  ExprAttr globalStride;
  ExprAttr localStride;
};

struct TransferPlan {
  TransferPlanKind kind = TransferPlanKind::Identity;
  // 外 -> 内，长度 >= 1、无上限。最内 1~2 项进 DataCopyPadExtend，其余每项
  // 一层 scf.for（4.2.6）。不把当前 corpus 的轴数固化成内联容量。
  SmallVector<TransferDim, 0> dims;
};

// `DataCopyPadExtend` 的四个参数加两侧偏移。一维搬运时 blockCount 为 1、两个
// stride 为 0，与二维走同一条发射路径。
//
// **两个偏移的来历不同，不要按对称去读。** `globalOffset` 是 tile 在 GM 里的起点，
// 即 `axis \ vectorized_axis` 上的求和；局部侧对应的量恒为 0（队列按 slot 自己轮转、
// TBuf 按 `b{id}_size` 只装一个 tile，tile 的起点就是 buffer 基址）。`localOffset`
// 装的是另一件事：> 2 项时外层 `scf.for` 走的是 tile **内部**的行，那几层的
// `Σ 归纳变量 × 本项 stride` 两侧各有一份，GM 侧那份加进 `globalOffset`，UB 侧那份
// 就是 `localOffset`。<= 2 项时它是常量 0。
//
// （S4 实施偏差：草案里写的是「inner offset 是循环体内的局部量，不进本结构体」。
//   实际上它必须进——发射体要拿它去建 `LocalTensorSubIndexOp`，而回调只收到这一个
//   结构体。见 8.4.1。）
struct ScheduledTileIndices {
  Value globalOffset;
  Value localOffset;
  Value blockCount;
  Value blockLen;
  Value globalStride;
  Value localStride;
};

enum class TransferPlanSide { Automatic, Operand };
```


**所有权**：全部是值类型，生命周期在单次分析调用内。`ExprAttr` 是 MLIR 的
uniqued attribute，由 context 持有。

## 4.2 关键技术/算法

### 4.2.1 统一偏移公式：集合差求和

```
offset(space) = Σ_{a ∈ axis 表 \ vectorized_axis}  轴索引(a) × stride[space](a)
```

**求和集合是一个集合差，不是一次配对反查。** `axis` 是完整轴表（GM 视图），
`vectorized_axis` 是 tile 内轴（UB 视图），差集就是"tile 外面还剩几层循环"。

现在的代码用 `splitPairOtherId == X` 一对一反查外层轴
（`TransferPlan.cpp:123-130`），遇到 `Merged` 轴就断——而 RCore 的块轴恰好是
合并出来的。集合差没有这个问题：合并轴在 `axis` 里、不在 `vectorized_axis` 里，
自动贡献一项。

一个公式，换一张步长表就换一个空间。GM 侧取 `strides`；UB 侧取
`vectorized_strides`，而由于 tile 起点恒在 buffer 基址，UB 侧的和恒为 0
（见 4.5 备选方案一）。

与 legacy `tiler.Offset(current_axis, axis, axis_strides)` 同构——legacy 的
`current_axis` 就是这个集合差。

**边界**：
- `stride == 1` 时省略乘法（消掉一维路径上的全部新增算术）
- `stride` 为符号表达式时走 `materializeRuntimeSize`（4.2.5 保证可解）

**复杂度**：O(|axis|)，一个 kernel 的轴数是个位数到十几。

**一维等价性**见 3.2.2 的手算：逐项相同，连乘法次数都一样。

### 4.2.2 有效视图：四条归约规则

`vectorized_axis` 到达 `TransferPlan` 时已经被调度器归一过一轮，但**归一不到
两根**。本节用实测数据确定本期在它之上还要做什么。

#### 调度器已经做了什么

`optimize.cpp:959-964`，在任务生成**之前**：

```cpp
// cube拆分后再做合轴
if (!ScheduleUtils::HasComputeType(optimize_graph, af::ComputeType::kComputeCube)) {
  // 这里concat已经打破了一套轴的约束
  GE_ASSERT_SUCCESS(RemoveAllZeroStrideLoopAxis(optimize_graph), ...);
  GE_ASSERT_SUCCESS(MergeContinuousAxis(optimize_graph), ...);
}
```

调度**之后**还有一次重新合并，`CanDoReMergeAxis`（`optimize.cpp:365-371`）排除三类图：

```cpp
// 如果包含Gather、Cube类型节点，则不能重新合并轴
// RCore类型中包含Reduce的图依赖固定的二维调度轴，不能重新合并轴
const bool is_rcore_reduce = (reduce_type == ReduceTemplateType::kRCore) && cache.HasReduce();
return !is_rcore_reduce && !cache.HasGather() && !cache.HasCube();
```

注释里那句「**RCore 类型中包含 Reduce 的图依赖固定的二维调度轴**」曾被本文当作
"调度器会把 reduce 归一到二维"的信号。**实测证明这个读法是错的**——它说的是
RCore+Reduce 的图**不能再合轴**，不是它已经是二维。

#### 实测：真实 ARAR 的四个 tiling case

2026-09-21，`optimize_ut` + `Construct_Reduce_ARAR`（`test_reduce.cpp:129`，
四根轴 z0..z3，归约 z1 与 z3）。RCommon（默认）产出 4 个并行 tiling case：

| case | `vectorized_axis` | 根数 |
|---|---|---|
| [0] | `[z0t(TileInner) z2(Original) z1t(TileInner) z3(Original)]` | **4** |
| [1] | `[z0t, z2, z3t]`（含 `z1z3T(Merged)`） | **3** |
| [2] | `[z2t, z1t, z3]`（含 `z0z2TB(BlockOuter)`） | **3** |
| [3] | `[z2t, z3t]`（两次合并都发生） | **2** |

RCore（`AutoSchedule(graph, impl_graphs, true)`）同样是 4 / 3 / 3 / 2，但块轴是
`z0Tz1TTB(BlockOuter)` + `z1Ttb(BlockInner)`——**A 的 TileOuter 与 R 的 TileOuter
先合并成一根再 block 切分**，不是两根独立块轴。

case[0] 的 Load descriptor 全貌：

```
axis      = [z0TB(BlockOuter) z0Tb(BlockInner) z0t(TileInner) z1T(TileOuter) z1t(TileInner) z2 z3]
strides   = [s1*s2*s3*z0Tb*z0t   s1*s2*s3*z0t   s1*s2*s3   s2*s3*z1t   s2*s3   s3   1]
vectorized_axis    = [z0t  z2  z1t  z3]                                  << 4 根
vectorized_strides = [8⌈s3/8⌉*s2*z1t  8⌈s3/8⌉*z1t  8⌈s3/8⌉  1]
```

**连续性合并救不了 case[0]。** 从内向外扫：`z3`（gm=1, ub=1）之后轮到 `z1t`，
累积 GM stride 是 `s3` 而实际是 `s2*s3`（不等），累积 UB 是 `s3` 而实际是
`8⌈s3/8⌉`（符号上判不出）。**第一步就断**，四根全部留下。

#### 因此本期做什么

调度器那两个 pass 是**全图一套轴**的（`optimize.cpp:960` 注释：「这里 concat 已经
打破了一套轴的约束」）——只有对图里**每个**张量都成立才能删或合。据此拆开：

| 规则 | 调度器覆盖了吗 | 本期 | legacy 出处 |
|---|---|---|---|
| **零轴剔除**（两侧 stride 均为 0） | 做了，但全图一套轴，**删不掉单张量的零轴** | **做** | `ShouldIgnoreDataCopyZeroAxis` |
| **extent == 1 剔除** | 未确认 | **做**（便宜） | `CalculateDmaParams` |
| **连续性合并** | 做了，但**归一不到两根**（上表） | **做** | `IsDataCopyAxisContinuous` |
| **不跨 TileInner 边界合并** | — | **做**（随合并） | `CalculateDmaParams` |

**四条全做。** 上一版本文写的是"剔除，不合并"，依据是 `load_sum_store 16x32` 的
`vectorized_axis` 只有两根——那个例子**只有两根原始轴，永远不可能超过 2**，是退化
样本。ARAR 探针推翻了它。

**零轴剔除为什么调度器删不掉**：归约的 destination 在被归约轴上 stride 为 0，而
输入不是（3.2.4 的实测数据）。这是**单张量**事实，graph-wide 的
`RemoveAllZeroStrideLoopAxis` 看不见。这也正是 `TransferPlanSide::Operand` 存在的
同一个原因。

零轴剔除照搬 legacy 的条件（`common_utils.cpp:236-241`），**不能写成无条件跳过**：

```cpp
has_non_zero_axis || axis_pos == 0 ||
StaticCheckEq(ub_strides[axis_pos - 1], One)  == kTrue ||
StaticCheckEq(ub_strides[axis_pos - 1], Zero) == kTrue
```

注释写明了意图：「如果当前轴 gm 和 ub 上对应的 stride 均为 0，**如果前序轴的 stride
不为 1，则保留当前轴**」。

连续性合并同样照搬（`common_utils.cpp:243-248`），注意它是**三值**逻辑：

```cpp
StaticCheckEq(cur_gm_stride, gm_stride) == kTrue &&
StaticCheckEq(cur_ub_stride, ub_stride) == kTrue
```

**判不出来就不合并。** 这是安全方向——合错的表现是把不连续的两根并成一根，即
静默搬错地址；不合的代价只是多一层外层循环。

#### 归约之后仍 > 2 根

**不报不支持，套外层循环**，见 4.2.6。这是 1.1.2「四个 tiling case 必须全部可
下降」的直接后果：四个候选里只有 case[3] 是 2 根。

### 4.2.3 为什么 DMA 参数必须按空间命名

这不是风格问题，有三层理由：

1. **计划拿不到方向。** `emitScheduledTransfer` 是 Load 和 Store 共用的一个函数。
   按 `src` / `dst` 命名字段，等于逼这个函数去猜它不知道的事。
2. **接反了量纲也跟着错。** `DataCopyPadExtend` 的两个重载**单位换算不同**
   （`autofuse/ascendc/api/datacopy.h`）：`LocalTensor ← GlobalTensor` 支把
   `src_stride` 按字节、`dst_stride` 按 32B 块处理；`GlobalTensor ← LocalTensor`
   支正好相反。选错一侧不只是位置错，量纲也错。
3. **既有字段已经是这个做法。** `globalOffset` 从来没出过这个问题，正因为它按
   空间命名、方向映射在发射侧。

legacy 的 `SetDmaParams`（`codegen/api_call/utils/api_call_utils.cpp`）用
`copy_in` 做的正是这个映射：`src = copy_in ? gm : ub`，`dst = copy_in ? ub : gm`。

### 4.2.4 缝隙的定义

```
缝隙 = 外层 stride − 本次实际搬的长度
```

单位是**元素个数**，32B 块的换算由 `DataCopyPadExtend` 自己做。与 legacy
`SetDmaParams` 传的是同一套量。

### 4.2.5 符号闭合：为什么 stride 可以直接物化

新偏移式要**物化 stride**，而现有实现从不读 `strides`，这条路一次都没被执行过。
风险是具体的：`materializeSymbolicExpr` 对每个自由符号查 `bindings`，**查不到就
`failure()`**（`SymbolicBuilder.cpp:52-55`），而 `bindings` 只来自
`asc_graph.size_var`（`AscendCKernelABI.cpp:196-201`）。

#### 实测结论

同一次 ARAR 探针，八张图（RCommon ×4、RCore ×4）的
`strides` 与 `vectorized_strides` 的自由符号**全部落在 `size_var` 里**：

```
===== RCommon case[0] =====
  size_var        (7): s0 s1 s2 s3 z0Tb_size z0t_size z1t_size
  gm strides syms (6): s1 s2 s3 z0Tb_size z0t_size z1t_size
  ub strides syms (3): s2 s3 z1t_size
  MISSING from size_var [gm strides]: <none>
  MISSING from size_var [ub strides]: <none>
```

八张图全是 `<none>`。还拿到一条更强的事实：**`size_var` 与「所有 `axis.size` 的
自由符号并集」逐项相等**，八张图都是 7 个。

#### 为什么必然如此

`ascendc_ir.cc:411-418`，每次切分只新建**一个**符号：

```cpp
inner_size = CreateSizeVar(actual_inner_axis_name + "_size");   // 只有 inner 建符号
outer_size = sym::Ceiling(single_axis.size / inner_size);       // outer 是派生式
```

stride 是轴尺寸的乘积，所以它的符号集天然
⊆ {原始维度} ∪ {每次切分的 inner}，而这两拨恰好就是 `size_var`
（`optimize.cpp:1118-1121` 先 `ClearAllSizeVar` 再加原子符号，切分过程中各自补回
inner）。不是巧合，是结构性的。

而 TilingData 的字段表又正是由 `size_var` 生成的（`AscendCKernelABI.cpp:146-150`），
于是三者闭环：

```
axis.size 的符号  ==  size_var  ==  TilingData 字段表  ⊇  strides 的符号
```

#### 两个后果

1. **`TilingParams.cpp` 不需要改。** 它今天只 walk `size_var` 与
   `buffer_desc.size_bytes_expr`，不 walk `TensorDescAttr`——闭合成立就不必扩。
   同样不需要新增 `size_var`、不需要扩 TilingData schema。
2. **stride 物化走 `materializeRuntimeSize`，不改函数签名。** 它无条件
   `getOrCreateTilingParam`，之前不敢用正是怕造出 schema 里没有的字段；闭合成立
   后安全，于是不必把 `SymbolBindingMap` 从 ABI 层一路穿到所有改写 pattern。

#### 仍然要加一道门禁

闭合是**当前调度器的性质**，不是 IR 层的不变量。哪天 stride 引入一个 `size_var`
之外的符号，`materializeRuntimeSize` 会造出一个 TilingData 里没有的字段，失败点
掉到 lowering，诊断离现场很远。

所以在 `buildTransferPlan`（纯属性阶段）就查一次集合包含，不满足直接
`failure()` → `UnsupportedShape`。成本是一次集合判断，收益是把失败点拉回可诊断的
位置。

### 4.2.6 超过两根：外层循环，以及轴序不一致

#### 轴序不一致是常态

case[0] 的 UB 轴序是 `[z0t, z2, z1t, z3]` = **A A R R**，而它们在 GM 里的排布是
**A R A R**。按 vectorized 顺序取出的 GM stride 是
`[s1*s2*s3, s3, s2*s3, 1]`——**非单调递减**。

现有实现里没有任何地方表达"顺序"：`isDirectProjection`
（`TransferPlan.cpp:31-43`）只检查各 result 是互不相同的 dim，它**允许置换**，
但下游没有一处读这个置换。

#### legacy 的答案：不重排，用循环走

legacy 对此没有任何特殊处理。合并按 `vectorized_axis` 顺序扫，GM stride 靠
`std::find` 在 `axis` 表里查位置取；顺序不一致时连续性判据自然不成立，合并不触发
——然后**外层循环兜住**：每根外层轴各自贡献 `outer_for_i × 自己的 stride`，
任意轴序都正确。

```cpp
// api_call_utils.cpp:102-114 CreateOuterFor + CalcInnerOffset
for (int outer_for_0 = 0; outer_for_0 < ActualSize(r0); outer_for_0++) {
  for (int outer_for_1 = 0; outer_for_1 < ActualSize(r1); outer_for_1++) {
    DataCopyPadExtend(ub[Σ outer_for_i × ub_stride_i],
                      gm[gm_offset + Σ outer_for_i × gm_stride_i],
                      blockCount, blockLen, srcStride, dstStride);
  }
}
```

**DMA 只负责最内两根，其余全是显式 `for`。** 本方案照此办理，用 `scf.for` 表达。

#### 与本方案的对应

| ARAR case | 有效轴数 | 发射形态 |
|---|---|---|
| [3] | 2 | 一条 `DataCopyPadExtend` |
| [1] [2] | 3 | 一层 `scf.for` + 一条 DMA |
| [0] | 4 | 两层 `scf.for` + 一条 DMA |

**不实现 legacy 的 `need_swap` 二选一。** `CreateDmaCall` 对 > 2 根发**两份**
DMA，包在运行期 `if (ActualSize(倒数第2) < ActualSize(倒数第3))` 里，在"段多段短"
与"段少段长"之间选。那是纯性能选择，不影响正确性；本期只发 `need_swap = false`
一版，源码体积减半。记为 L7。

## 4.3 流程设计

### 主流程

见 3.1。

### 异常流程

全部收敛到 `failure()`，由能力查询映射成诊断：

| 情形 | 阶段 | 结果 |
|---|---|---|
| 两张步长表长度不等 | `buildTransferPlan` | `failure()` |
| 轴查不到 GM stride / repeat | 同上 | `failure()`（契约破损） |
| stride 的自由符号不在 `size_var` 里 | 同上 | `failure()` → `MissingInformation`（见 4.2.5） |
| 归约后有效轴为 0 根 | 同上 | `failure()` → `MissingInformation` |
| 计算算子遇到 > 1 根 | `materializeOperandTiles` | `failure()` → `UnsupportedShape` |
| 运行期取不到 axis binding | `emitScheduledTransfer` | `failure()` → match 失败 |
| 符号 stride 物化失败 | 同上 | `failure()` |

**归约后 > 2 根不再是异常**，走 4.2.6 的外层循环。

S4 之后**寻址侧已经没有形状天花板**，于是 `TransferPlanFailure::UnsupportedShape`
在 `buildTransferPlan` 里没有产生点了：上表前两行按实测口径记的都是
`MissingInformation`（草案原先写的是 `UnsupportedShape`，与实现不符，这里按实现回写）。
剩下的天花板全在计算侧——`isLoweringSupportedForCompute` 的 elementwise 多维限制，以及
`ReducePlan` 对非连续 A/R 分段和 RA 的拒绝。它们直接报 `CapabilityReason::UnsupportedShape`，
不经过本枚举。枚举值保留，理由见
`TransferPlan.h` 的注释。

**没有回滚需求**：`buildTransferPlan` 不建 IR；`emitScheduledTransfer`
在 `ConversionPatternRewriter` 下建 IR，失败由 rewriter 自身回滚。

## 4.4 对子模块的修改

| 目录 / 文件 | 修改点 |
|---|---|
| `dialect/AFIR/Analysis/TransferPlan.h` | 见 3.5.1 的接口表 |
| `dialect/AFIR/Analysis/TransferPlan.cpp` | `getLocalDescriptor` 支持 `side`；`buildScheduledPlan` 改为有效视图归约（四条规则 + 符号闭合门禁）；**删除** `splitPairOtherId` 反查整段；新增 `collectOuterAxisIndices` / `materializeActualSize` / `emitOuterLoops`；`materializeScheduledTileIndices` → `emitScheduledTransfer`；删除 `isQueueBuffer` |
| `conversion/AFIRToAscendC/Implementations/DataMovement.cpp` | 改为向 `emitScheduledTransfer` 传发射回调；消费新字段；**新增方向映射**（`src = load ? global : local`）；`ScheduledContiguous` → `Scheduled` |
| `conversion/AFIRToAscendC/Implementations/UnaryElementwise.cpp` | `dims.size() != 1` 拒绝；`count` → `blockLen`；枚举改名 |
| `conversion/AFIRToAscendC/Implementations/BinaryElementwise.cpp` | 同上 |
| `dialect/AFIR/Transforms/LoopGenerate.cpp` | `ActualSize` 的计算（`:541-543`、`:591-594`）提取为共用 helper，与 `materializeActualSize` 同一实现 |
| `dialect/AFIR/Analysis/CanonicalBindings.cpp` | 删除 `StorageView::indexingMaps` |
| `conversion/AFIRToAscendC/AscendCAttributes.cpp` | `getValueMemorySpace` 改为先读 type |
| `test/ut/dialect/TransferPlanTest.cpp` | **新增** |
| `test/ut/dialect/CMakeLists.txt` | 注册新 UT |
| `test/lit/conversion/afir-to-ascendc-two-dim-store.mlir` | **新增** |

## 4.5 备选方案

### 备选一：UB 侧也保留偏移字段

**不采纳。** `8fcd2ebe` 已经论证并删除了局部侧偏移：队列由 DeQue 定位、TBuf 每轮
整块重写。二维不改变这一点——二维 tile 仍然是**一个** tile，它在 UB 里的起点仍是
buffer 基址，行与行的间距由 `localStride` 表达。

保留它的代价是真实的：此前那份实现里，TBuf 分支会算出一个**从未分配过的**位置，
这正是 `8fcd2ebe` 修掉的越界。

### 备选二：引入独立的 `AddressView{axes, strides, unit}` 抽象

**不采纳。** 曾考虑按空间分两张视图表。但 DMA 参数（缝隙 = 外层 stride − 实际
长度）本来就要**同时**读两侧，按项交错存两侧 stride 的 `TransferDim` 更贴合
使用方式。额外的抽象层没有第二个使用者。

### 备选三：让 `indexing_maps` 成为寻址的语义源

**不采纳。** 三条理由：
1. `AFIRIndexingMapInterface` 是 op 的公开契约，删不掉
2. `isIdentity()` 那道门没有替代品
3. `Data.td` 文档化了"形状可以不同"的语义

且真正的权威源是第三者——`asc_graph` 的轴切分树。本方案的做法是**让
descriptor 的两张步长表成为寻址事实源**，`indexing_maps` 退回"是不是直接投影"
这一道门的角色，并由 3.2.5 的 verifier 保证两者不打架。

### 备选四：不在 codegen 里做连续性合并，> 2 根报不支持

**不采纳——本方案对这一条反复过两次，过程值得留档。**

1. 最初**采纳合并**，依据是"RCommon 的 tile 是 `[a0t, r0t, a1, r1]` 四根"。那个
   依据错了：它读的是 `notepad/reduce.md` 对**原始张量维度**的记法，不是调度后的
   轴表。
2. 于是改成**不合并**，依据是 `load_sum_store 16x32` 实测 `vectorized_axis` 只有
   两根。这个依据也错了：那个例子**只有两根原始轴**，永远不可能超过 2，是退化
   样本，不能反推一般情况。
3. 2026-09-21 的 ARAR 探针给出真实数据：四个 tiling case 是 **4 / 3 / 3 / 2** 根
   （4.2.2）。**合并必须做**，而且合并之后 case[0] 仍是 4 根——外层循环也必须做。

**教训**：两次错误都是从单个样本外推。第三次用的是"跑一遍真实调度，四个 case
全打出来"，这才是能支撑设计的证据强度。

至于"> 2 根报 `UnsupportedShape`"这个曾经的备选：它与 1.1.2「四个 tiling case
必须全部可下降」直接冲突——四个候选里只有一个是 2 根。而且它还额外依赖一条
**未经验证**的假设：plan 选择器遇到候选下降失败时会回落到别的候选，而不是整体
失败。两条理由任取其一都足以否掉它。

## 4.6 通用性边界：其他搬运类 op 能不能复用

评审必然会问：`TransferDim` 这套表达，对 transpose / concat / gather 还成立吗？

**结论：它是「调度驱动的、单张量的、DMA 类」搬运的通用方案，不是全部搬运类 op 的
通用方案。但三个扩展方向都没有被堵死。**

分四层看，通用性各不相同。

### 4.6.1 第 1 层：偏移公式 —— 真通用

```
offset = Σ_{a ∈ 轴表 ∩ 当前循环轴}  轴变量(a) × stride(a)
```

legacy 的 `tiler.Offset(current_axis, axis, axis_strides)`
（`codegen/codegen_kernel.h:263`）就是这个函数，**gather 自己也在调它**：

```cpp
// autofuse/codegen/api_call/gather/gather_api_call_base.cpp:86
ss << "(" << tpipe.tiler.Offset(gather_outer_axis_id, gather_outer_axis_id,
                                gather_outer_axis_strides) << ")";
```

这一层是整个 legacy 寻址的公共底座，把它搬进 MLIR 对所有搬运类 op 都有效。

### 4.6.2 第 2 层：轴的表达 —— 部分通用

| op | 够不够 | 依据 |
|---|---|---|
| **transpose** | **够** | 置换体现在「按 `globalStride` 排序的项序 ≠ 按 `localStride` 排序的项序」。一项带两侧 stride，恰好把置换编码进去了。reduce 的 A A R R vs A R A R（4.2.6）就是这个形态的一个实例 |
| **concat** | **不够** | `TransferPlan` 从**一张** local descriptor 建。concat 是 N 个源各有自己的 stride 表与 buffer 偏移（`concat_api_call.cpp:221-291` 的 `src_buffer_offsets` / `src_loop_strides` 按 `input_index` 循环填） |
| **gather** | **不够** | 见 4.6.3 |

### 4.6.3 第 3 层：序号从哪来 —— 真正的分水岭

本方案假定序号来自**循环结构**（4.2.1 的集合差 + `AxisBinding`）。gather 破的就是
这条：

- 偏移里有一项是 `indices[...]` 的**运行期数据值**
  （`CalGatherParamOffset(param_axis, indices_value, ...)`）
- 它**自己发循环**，并且要把线性偏移**除模拆回每根轴**
  （`gather_api_call_base.cpp:45`：`axis = outer_axis_offset / stride`）——
  那是 Σ 公式的**逆运算**

所以 gather 不是「多一种序号来源」，它是另一类寻址。

### 4.6.4 第 4 层：发射 —— 不通用，本来也不该通用

- 搬运 → `DataCopyPadExtend`
- transpose → legacy 按置换模式分派到五种以上不同 API
  （`transpose_api_call.cpp` 的 `IsTranposeType10` / `102` / `021` / `210` / `0213`），
  **根本不是 DMA**
- concat → 打包成一个 tiling 结构体传给 API

transpose 的情况最微妙：**表达够用，发射不够用**。`TransferDim[]` 能描述一个
转置，但发射侧要在它之上再做一次「这是哪种置换」的分类，而那属于 transpose 自己的
分析，不该进 `TransferPlan`。

### 4.6.5 三个扩展方向，都不堵

| 需求 | 扩展方式 | 是否需要改动本方案已定的结构 |
|---|---|---|
| 多源（concat） | `TransferPlanSide` 已经是「取哪张 descriptor」的口子，推广成 operand index | 否。三个调用点都有默认值 |
| 数据依赖序号（gather） | 给 `emitScheduledTransfer` 加 `extraAxisValues: ArrayRef<pair<axisId, Value>>`，让调用方覆盖某根轴的索引 | 否。纯新增参数 |
| 置换分类（transpose） | 独立分析，读同一个 `TransferPlan` | 否 |

### 4.6.6 一个曾经的耦合，已随结构变化自动消失

本方案的 `TransferTileAxis` 草案里有 `tileOuterAxis` / `blockOuterAxis` /
`blockInnerAxis` 三个字段。它们**不是轴的固有属性，是「序号怎么算」的配方**，
对 gather 这类序号来自数据的 op 完全无意义。当时的结论是"本期不拆，但要写明
拆点"。

改成 `TransferDim` 之后，这件事**自动解决了**：

```
TransferDim{extent, globalStride, localStride}   ← 纯粹是「搬多少、两侧间距多大」
索引从哪来                                            ← 在 emitScheduledTransfer 里现查，不进计划
```

两件事天然分开，不需要额外的解耦动作。下一个做 gather 的人走 4.6.5 的
`extraAxisValues` 即可，没有"配方字段"可以被误扩。原 L5 条目因此销号。

## 4.7 错误处理

### 系统错误

不涉及内存申请、文件、runtime 调用。唯一的"系统"资源是 MLIR 的
`OpBuilder` 建 op，失败由 MLIR 自身处理。

### 接口错误

MLIR 内部接口，不返回错误码。错误表达为：

- `FailureOr<T>` 的 `failure()`
- 能力查询的 `CapabilitySupport::unsupported(CapabilityReason::UnsupportedShape)`
- 改写阶段的 `rewriter.notifyMatchFailure(op, "...")`

**判据统一**：凡是"漏判的后果是静默错值"的检查，一律返回 `FailureOr` 而不是
带默认值的 `bool`——调用方要么拿到答案，要么拿到一次失败。

# 5 安全检查

已读取 `docs/guidelines/编码红线.md`。逐项：

| 红线 | 结论 |
|---|---|
| 1 硬编码敏感信息 | 不涉及 |
| 2 外部输入作索引/长度/偏移前校验 | **涉及且已处理**。`vectorized_axis` / `strides` 来自调度器，是外部输入：两表长度必须相等（否则 `failure()`）、轴必须在 `axis` 表里查得到（否则 `failure()`）、归约后至少一项（否则 `failure()`）、**stride 的自由符号必须 ⊆ `size_var`**（否则 `failure()`，见 4.2.5）。**不再有 `.front()` 这种无检查取用** |
| 3 整数溢出/反转/除 0 | **涉及**。偏移的乘加在 `index` 域，与既有 GM 偏移同一处理方式（裸算术）。本方案**不单点加断言**，理由见 3.3.4；缝隙的减法理论非负，依赖调度器正确性，不在本层证明。无除法。**有效视图的四条归约规则（4.2.2）全部在编译期符号域做，用三值 `StaticCheckEq`，判不出即不剔除/不合并，不产生运行期算术** |
| 4 资源释放覆盖异常分支 | 不涉及，无资源申请 |
| 5 内存申请前判大小 | 不涉及。本方案不改分配路径 |
| 6 未指定求值顺序 | **需注意**。`builder.create<A>(loc, builder.create<B>(...))` 这种嵌套在既有代码里存在，参数求值顺序未指定但各 `create` 之间无副作用依赖，行为确定。新代码按单条语句拆开写 |
| 7 禁止无理由扩大修改范围 | **涉及**。3.2.5 的四项债务都在寻址链路上且与本方案同文件/同调用链；`getValueMemorySpace` 一项最远，若评审认为超范围可拆出 |
| G1 API/ABI 兼容 | 不涉及，内部头 |
| G2 禁止硬编码芯片/平台类型 | **涉及且已处理**。能力判断全部经 `TargetProfile`，conversion 层不出现 soc 字面量或 `NpuArch::` |
| G3 图改写保持数据边/控制边等价 | **涉及**。本方案只改地址计算，不增删数据边；新增的 `arith` op 是纯计算，无副作用 |
| G4 新增时序依赖的 pass | 不涉及，不新增 pass |
| G5 图改写结果确定 | **涉及且已处理**，见 3.3.4 确定性一段 |
| G6 高频路径海量日志 | 不涉及，不新增日志 |
| G7 `rt*`/`aclrt*`/AscendC 生命周期 | **涉及**。发射 `DataCopyPadExtend` → `AscendC::DataCopyPad`。同步由既有的 `afir-pipeline-sync` 负责，本方案不改同步语义；tensor 生命周期由 `MemoryAllocate` 负责，本方案不改 |
| G8 非开放 runtime 接口 | 不涉及 |
| G9 Python/C++ 绑定引用计数 | 不涉及 |
| G10 构建脚本增量/离线 | 不涉及 |
| G11 测试临时产物进版本库 | 不涉及 |
| G12 全局/静态对象析构跨 SO | 不涉及 |

# 6 兼容性检查

| 项 | 影响 |
|---|---|
| 对外 API/ABI | **无**。`TransferPlan.h` 是内部头 |
| 脚本参数 / 配置文件 | **无** |
| 安装目录 / 产物布局 | **无** |
| 生成 kernel 的源码 | **一维要求逐字节不变**（见 8.1）；二维为新增形态 |
| 既有 lit 期望 | **一维不变**。枚举改名与字段改名是 C++ 侧，不影响 IR 文本 |
| 既有调用点 | `buildTransferPlan` 新增的是**默认参数**，调用形式不变。但 `materializeScheduledTileIndices` → `emitScheduledTransfer` 从返回值变发射器，**三个消费者都要改调用形式**（S4），这是本方案唯一的破坏性接口变更；字段改名（`count` → `blockLen`）要同步 |
| `isQueueBuffer` 的删除 | 零调用者，无影响 |

# 7 测试设计

## 7.1 测试边界

| 层 | 入口 | 目标依赖 | 验什么 |
|---|---|---|---|
| **UT** | `ctest` → `TransferPlanTest` | **无** | 计划结构、四条归约规则、集合差偏移、符号闭合门禁、失败形态 |
| **lit** | `af-opt --target=Ascend910B1` | V2201（mem-based） | 一维不回退 **+ 二维 / 外层循环的 IR 与发射文本** |
| **lit** | `af-opt --target=Ascend950DT_950x` | V3510（reg-base） | 同上，另一张供给表 |
| **CPU E2E** | `check-e2e-cpu` | **仅 910B1** | **数值**——二维与 > 2 维的唯一数值验证入口 |
| **调度探针** | `optimize_ut` + `Construct_Reduce_ARAR` | 无 | **不是回归测试**，是设计依据的取数手段。四个 tiling case 的轴表与符号集合由它得出（4.2.2 / 4.2.5）。结论变化时需重跑 |

**两个目标都要发一遍，不是二选一。** `DataCopyPadExtend` 在两张供给表上同名同
签名，但走的是不同实现；只钉 V3510 会让 910B 的 mem-based 二维发射整条路没有
用例。这条在 910B 被误判为不支持的那段时间里被写成了"V3510 一路不受影响"，
判定修复后不再成立。

**910B 的角色反而更重**：它是唯一能跑 CPU E2E 的目标，二维的**数值**正确性只能
在它上面验。950 一路只有 IR 与文本比对——比对通过不等于搬对了。

**UT 的构造数据直接取自探针实测**，不要手编轴表——手编的轴表会同时编掉
"UB 轴序 ≠ GM 轴序"和"符号 stride 判不出相等"这两个恰恰要验的性质。

**打桩**：无。`TransferPlan` 是纯分析，UT 直接用 `parseSourceString` 构造带属性的
IR 片段。

## 7.2 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|---|---|---|---|
| 功能 | 一项有效轴 → `dims.size() == 1`，字段与改动前等价 | 构造 elementwise 形态 IR，断言计划 | UT |
| 功能 | 两项有效轴 → 外/内顺序、两侧 stride 取值正确 | 构造 reduce 形态 IR | UT |
| 功能 | **偏移集合差**：`axis \ vectorized_axis` 取对，每项乘自己的 stride | 用 ARAR case[0] 的真实轴表构造 | UT |
| 功能 | **合并块轴**：`Merged` 派生的 BlockOuter 在差集里，贡献一项 | 用 RCore 的 `z0Tz1TTB` 形态构造 | UT |
| 功能 | 索引来源两种：`scf.for` 归纳变量 / `BlockOuter` 的 `AxisIndex` | 两条用例 | UT |
| 功能 | **归约 · extent == 1** 的轴不占维度 | 构造含 extent-1 轴的 tile | UT |
| 功能 | **归约 · 零轴**按 `ShouldIgnoreDataCopyZeroAxis` 条件跳过 | 正例一条 | UT |
| 功能 | **归约 · 零轴的保留分支**：前序轴 stride 非 0/1 时**保留**该轴 | 反例一条，防写成无条件跳过 | UT |
| 功能 | **归约 · 连续性合并**：两侧 stride 都相等时合并，extent 相乘 | 正例一条 | UT |
| 功能 | **归约 · 合并的不确定分支**：`StaticCheckEq` 判不出时**不合并** | 反例一条（符号 stride），防写成 `!= kFalse` | UT |
| 功能 | **归约 · 不跨 TileInner 边界** | 反例一条 | UT |
| 功能 | **ARAR 四个 case 的有效轴数**：4 / 3 / 3 / 2 归约后分别得几项 | 四条用例，用探针实测的轴表构造 | UT |
| 功能 | **符号闭合门禁**：stride 用了 `size_var` 外的符号 → `failure()` | 反例一条 | UT |
| 功能 | **AllLoad**：`vectorized_axis` 不含 `n_group`，轴数更少仍成立 | 构造 AllLoad 形态 | UT |
| 功能 | **AR / RA 中立**：同一组轴、A/R 顺序互换，计划结构对称 | 两条对照用例 | UT |
| 功能 | 二维 Load 的 `blockCount`/`blockLen`/两侧 stride | lit + FileCheck，**V2201 与 V3510 各一条 RUN 行** | lit |
| 功能 | **二维 Store 的方向映射**（GM/UB 行距不等） | 新增 `afir-to-ascendc-two-dim-store.mlir`，**两个目标** | lit |
| 功能 | **三项 → 一层 `scf.for`；四项 → 两层**，两侧 inner offset 各自正确 | 新增 `afir-to-ascendc-outer-loop-transfer.mlir`，**两个目标** | lit |
| 功能 | **910B（mem-based）二维发射的数值正确性** | `check-e2e-cpu`，唯一能验值的入口 | CPU E2E |
| 功能 | **轴序不一致**（UB A A R R vs GM A R A R）时偏移仍正确 | 用 case[0] 的 stride 表 | UT + E2E |
| 异常 | 两张步长表长度不等 → `failure()` | 断言 `failed()` | UT |
| 异常 | 轴不在 `axis` 表里 → `failure()` | 断言 `failed()` | UT |
| 异常 | 归约后 0 项 → `failure()` | 断言 `failed()` | UT |
| 异常 | 计算算子遇到 > 1 项 → `UnsupportedShape` | 构造二维 elementwise，断言能力查询 | UT/lit |
| 兼容性 | 生成 kernel 的 diff **逐处人工确认**为同一种变化（不是逐字节不变）。语料 13 个 kernel / 26 份源文件（44d7cb85 合并用例之前是 24 / 48） | S1/S2/S3 前后各生成一遍 | 手工验证 |
| 兼容性 | 一维 lit 全绿、CPU E2E 数值全绿 | `llvm-lit`、`check-e2e-cpu` | lit/E2E |
| 兼容性 | **每个 E2E 用例断言自己的切分形态**（`block_dim` / `block_inner` / 尾块） | `TilingShape` + `Covers*` 判据 | CPU E2E |
| 特性交叉 | **ARAR 四个 tiling case 全部下降成功**（不是"至少一个"） | 端到端跑一遍，断言四份 kernel 都生成 | E2E |
| 特性交叉 | RCommon / RCore / AllLoad × AR / RA 六个组合的数值正确性（AllLoad 两格已于 2026-10-01 在 CPU 上完成，且「AllLoad × 二维 RA」改为由 ≥3 维 ARA 覆盖） | 依赖 `afir.sum` 下降落地 | E2E |

**归约规则的用例必须成对写。** 四条规则里有三条都是"条件成立才动"，而写成无条件
的表现全是**静默少搬/搬错**，没有运行期征兆：

| 规则 | 写错的方向 | 后果 |
|---|---|---|
| 零轴剔除 | 写成无条件跳过 | 丢掉一根该搬的轴，静默少搬一维 |
| 连续性合并 | 把三值判定写成 `!= kFalse` | 把不连续的两根并成一根，静默搬错地址 |
| 不跨 TileInner | 漏掉边界检查 | 同上 |

第二条尤其要小心：`StaticCheckEq` 返回三值，**只有 `== kTrue` 才能合并**。
实测 case[0] 的 UB stride 是 `8⌈s3/8⌉` 这类符号式，落在"判不出"的那一档——
如果把判不出当成可合并，四根轴会被并成错的形状。

### 基线覆盖实测（已落地）

改动前必须知道现有 E2E 到底守住了什么。2026-09-21 给 `add_mlir_e2e_cpu_case` 加了
`TILE_ACCESSOR` / `BLOCK_INNER_ACCESSOR` / `COVERS` 三个**必填**参数，测试在运行期
打印并断言自己的切分形态（`test/e2e/cpu/support/E2eTilingShape.h`）。Ascend910B1、
`aiv_num = 48`、`ub_size = 196608` 下的实测：

| 注册 | 元素数 | tile | 每核轮数 | 核数 | blockIdx | 轮次 | 尾块 |
|---|---:|---:|---:|---:|:---:|:---:|:---:|
| `load_add_store_128x128x128` | 2097152 | 7288 | 6 | 48 | ✅ | ✅ | ✅ |
| `load_add_store_128x23x5` | 14720 | 4912 | 1 | 3 | ✅ | ❌ | ✅ |
| 9 个 `[128]` 用例 | 128 | 64 | 2 | 1 | ❌ | ✅ | ❌ |
| `load_add3_storex128` | 128 | 128 | 1 | 1 | ❌ | ❌ | ❌ |
| `load_add_diamond_storex128` | 128 | 128 | 1 | 1 | ❌ | ❌ | ❌ |

三条结论直接决定了 S1/S2 的风险面：

1. **`blockIdx` 项在 13 个注册里有 11 个是死的。** 10 个 `[128]` 用例 `blockDim = 1`，
   从来没验过多核寻址。
2. **`load_add3_storex128` 与 `load_add_diamond_storex128` 三项全零**——偏移恒为 0，
   对寻址零覆盖，它们验的是 op 下降。两者的 `COVERS` 记为 `none`，是显式声明而非遗漏。
3. **整套 E2E 里只有 `load_add_store_128x128x128` 一个注册压住了完整偏移式。**
   ATT 换个启发式让它的切分退化，寻址覆盖就整体归零，而数值断言照样全绿。

第 3 条是把 `COVERS` 做成必填的理由：覆盖面不该靠一个没人声明的 ATT 决策撑着。
已做变异验证——把 `load_add_storex128` 的声明改成 `multi_block multi_round tail`，
两条假声明各自失败并给出原因，真声明静默通过。

### 变异验证（必做）

`c2bed123` 的经验：本仓库在二维 Store 出现之前**所有 Store 都是
`blockCount = 1`、两个 stride = 0**，方向接反在全套测试下完全不可见。

因此 S3 必须做一次变异验证：**把方向映射退回无条件赋值，重跑 lit，确认只有
新增的二维 Store 那一条失败。** 既证明用例能捕获该缺陷，也反证旧用例确实覆盖不到。

## 7.3 测试命令

```bash
# 容器入口见 docs/guides/；以下都在容器内执行。

# 0) 配置。MLIR 测试树要求 RUN_TEST 处于 unset 状态，在 /workspace/build 里直接跑
#    `cmake .` 会把 check-ut / check-e2e-cpu 等整棵测试树从构建图里撤掉，而子目录的
#    Makefile 和 CMakeFiles/check-e2e-cpu.dir/ 会留在原地不被删除——于是 `make help`
#    照样列出这些 target，跑起来才报 "No rule to make target"。
cmake -U RUN_TEST -S /workspace -B /workspace/build \
      -D ENABLE_AUTOFUSE_MLIR_TESTS=ON -D ENABLE_AUTOFUSE_MLIR=ON
grep -c check-e2e-cpu /workspace/build/Makefile   # 必须非 0，否则上一步没生效

# 1) 单元测试（无目标依赖，二维在此验证）。ctest 必须指到 autofuse/mlir 子目录：
#    顶层 build 没有 CTestTestfile.cmake，ctest 会打印 "No tests were found!!!"
#    但退出码为 0。
ctest --test-dir /workspace/build/autofuse/mlir -R TransferPlanTest --output-on-failure

# 2) lit（含二维与外层循环；每条二维用例的 RUN 行同时钉 V2201 与 V3510）。
#    没有 check-lit-mlir 这个 target，直接调 llvm-lit。
/opt/llvm/bin/llvm-lit -sv /workspace/build/autofuse/mlir/test/lit

# 3) CPU E2E（一维回归 + 二维数值）。910B1 上唯一能验数值的入口。
cmake --build /workspace/build --target check-e2e-cpu -j 8
```

产物比对**不做逐字节 diff**：新公式丢掉了现有实现依赖的 stride 嵌套假设，一维产物的
op 序列本来就会变（3.2.2 有实测对照）。验收依据是 8.2 的三条判据。

# 8 验收标准

## 8.1 S1（把静默错误变成诊断）

- [x] `getVectorizedAxis` 加 `size() > 1 → failure()`（2026-09-21）
- [x] 符号闭合门禁落地，反例 UT 通过
- [x] `TransferPlanTest.cpp` 新增（此前零测试），7 条用例，正反成对
- [x] **行为零变化**：lit 117/117、`check-ut` 0 失败、`check-e2e-cpu` 26/26
- [x] reduce 形态从"静默取 `.front()`"变成 `UnsupportedShape`，有诊断；
      新增 `TransferPlanFailure` 出参，三处能力查询经
      `Implementations/TransferPlanCapability.h` 统一映射

**实现时发现的一条约束**（此前设计没有写到）：**空 `vectorized_axis` 必须保持回落
identity，不能算作明确拒绝。**

带 `afir.sched` 但 descriptor 上没有 `vectorized_axis` 是既有的合法形态——
`AFIRToAscendCPipelineTest` 的 plan candidate 夹具就是这样。第一版把"空表"和"多于一根"
一起按明确拒绝处理，整条 plan candidate 下降链路立刻报
`'afir.load' op has no supported Ascend implementation`。

因此不回落分支只认**两个新门禁**写进去的原因（轴数超限、步长符号越出 `size_var`），
其余失败保持原有回落行为。这条边界由 `MissingVectorizedAxisFallsBackToIdentityPlan`
钉住。

**不回落这件事本身也必须有用例。** 多维 tile 的 `indexing_maps` 完全可能就是 identity，
所以"调度路径拒绝 → 继续试 identity"会把一次明确拒绝换成一个搬整块的 identity 计划。
只有 `TwoVectorizedAxesAreRejectedAsUnsupportedShape` 时，把不回落分支删掉它照样绿——
`UnsupportedShapeDoesNotFallBackToIdentityPlan` 是补上的那一半。

**这一档可以独立合入。** 它不改任何正确行为，只把一个静默错误变成诊断，并把测试
地基铺上。

## 8.2 S2（偏移改集合差求和）

- [x] 删除 `loopAxis` / `blockOuterAxis` / `tileOuterAxis` 与 `splitPairOtherId`
      反查整段，全仓编译通过（2026-09-21）
- [x] 两种索引来源 UT 通过（`OffsetSumsBlockIndexAndLoopInduction`：块轴取
      `axis_index` 绑定、循环轴取归纳变量）
- [x] 合并块轴能贡献偏移项的 UT 通过（`MergedBlockAxisContributesAnOffsetTerm`：
      `from = [1, 5]` 且无 `split_pair_other_id`，旧反查在这种轴上落空）
- [x] 一维产物**数值不变，op 序列变**，三条判据：
      - `afir-to-ascendc-scheduled-transfer.mlir` 按新形态更新，两个目标都通过；
        lit 117/117
      - **910B CPU E2E 13/13 全绿**，含唯一压住完整偏移式的
        `load_add_store_128x128x128`（块号、轮次、尾块三项全覆盖）
      - 生成 kernel 的 diff 逐处确认，方法见下

`TransferPlanTest.cpp` 从 7 条增至 15 条。

### 生成 kernel 的 diff：机械穷举而非抽样

把 SSA 编号（`v[0-9]+` → `V`）与 `constexpr` 声明行归一后，对全部 13 个 CPU
kernel 做并集比对，**全仓只出现一种差异**，每个 kernel 各一份：

```
删  V ? V : V      V - c1_idx      V + V      V == V      V & V
加  V = V * V（提到循环外）   V + V ×2   V * V   V ? V : V   V < V
```

前五行是旧的「尾块 select」，后六行是新的「集合差求和 + `actual_size` 绑定」。
`constexpr` 行单独按集合比对，13 个 kernel 全部逐项相同（只是声明位置从函数头
散到首次使用处，是下一条偏差的副作用）。**没有第二种形态的差异**，判据满足。

**实测推翻了 3.2.2 的一个预测**：循环体内**不多**一次乘法，净持平。偏移从 3 个
op 变成 4 个（多一次 stride 乘法），但搬运长度从 4 个 op 变成 3 个（尾块 select
改成查 `actual_size` 绑定），一增一减抵消。另外 `块号 × zTb_size` 这一项被提到了
循环外——它是 `LoopGenerate` 算好的块偏移，旧式子每轮重算一次。

## 8.2.1 S2 实施中发现的三条设计偏差

**偏差一：`materializeRuntimeSize` 不能在下降阶段调用。**

4.2.5 写的是「stride 物化走 `materializeRuntimeSize`」。它内部是
`getOrCreateTilingParam`，缺符号时用裸 `OpBuilder` 往入口块插一个新的
`afir.tiling_param`。而调用点在 `applyFullConversion` 里（afir 方言整体非法），
转换驱动看不见裸 builder 建的 op，留下的就是一个没人下降、也没人填值的参数。

改为 `collectTilingParamBindings`：只认入口块里**已经在**的那些 `afir.tiling_param`，
缺了就失败。合法性仍由 S1 的符号闭合门禁保证（符号 ⊆ `size_var`，而相位 A 的
`--afir-materialize-tiling-params` 把 `size_var` 全部物化过）。

**偏差二：`afir.symbolic_eval` 此前在下降阶段没有下降者。**

复合 stride（`"(zTb_size * zt_size)"`）物化的产物就是一个 `afir.symbolic_eval`，
而它过去只在 memory-allocate 里被下降，那一步跑在转换之前。于是转换阶段新产的这
一个无人认领，整条搬运以「`afir.load` 无可用实现」失败——失败点离原因很远，而且
只在 stride 非字面量时出现（字面 stride 走 `arith.constant` 快路径，看不出问题）。

修法：`AFIRToAscendC.cpp` 的 pattern 集合里也 `populateSymbolicLoweringPatterns`。
模式本身是普通 `OpRewritePattern`，产物 arith / index / math 在该目标里都合法。

**偏差三（最重要）：轴绑定值的存活此前取决于巧合。**

3.2.2 写过「向量化轴只有 `tail_size` 没有 `actual_size`」。**这是错的**：
`LoopGenerate` 的 `generatePairedInnerActualSize` 确实为配对 inner 发了
`actual_size`（`--afir-loop-generate` 之后 IR 里看得到）。真正的情况是它**被删了**：
memory-allocate 的 `lowerSymbolicExpressions` 用 `applyPatternsGreedily(func, ...)`
扫全函数，顺手消除无使用者的纯 op，而 `#afir.axis_binding` 是「从轴找回值」的唯一
通路，**标记本身不构成使用**。

于是「一个轴绑定能不能活到下降阶段」取决于此前碰巧有没有别的消费者：block-inner
轴的 `actual_size` 当了循环上界所以活着，向量化轴的那份没人用，被删。

修法：工作表改成「全函数**除了**带 `#afir.axis_binding` 的 op」，配
`GreedyRewriteStrictness::ExistingAndNewOps`。**不能**改成「只扫
`afir.symbolic_eval`」——这一轮同时在做全函数的 fold，偏移为 0 的
`local_tensor.subindex` 之类要靠它折掉，`afir_pipeline_sync_canonical` 等四个用例
钉着折叠后的形态（第一版就是这么挂的）。

副作用：`constexpr` 常量的声明位置从函数头散到首次使用处（常量集合不变，见上）。

**这条偏差的份量超出 S2。** 它说明 `#afir.axis_binding` 目前是个**弱引用**，任何
新增消费者都可能撞上「我要的那个绑定恰好没有别的使用者，已经被删了」。S2 只堵住了
已知的那一个 DCE 点。S5 应当把它变成不变量——要么让标记本身算使用，要么让轴绑定值
由结构承载而不是由悬空的 SSA 值承载。

> **S5 已处理**，但走的是第三条路：弱引用保留，改为**可验证**。`LoopGenerate` 把承诺过
> 的 `(轴, 角色)` 登记在 func 的 `afir.axis_bindings` 上，MemoryAllocate 结束时
> 逐条核对，缺失在该 pass 内报出是哪根轴的哪个角色，而不是拖到搬运下降
> 时报成「缺 actual_size 绑定」。两条原方案的代价见 8.5。
>
> 另需记一笔：全仓当时只有两个全函数贪心驱动，`afir-lower-symbolic` 那个**不在生产
> 流水线**里（`addPlanCandidateLoweringPipeline` 只有 loop-generate → memory-allocate →
> pipeline-sync → select → convert），所以这条债在 S5 时还不是活 bug，守的是将来。

**`materializeActualSize` 未提取。** 3.5.1 计划把尾块 select 收成一个公共 helper。
S2 走的是更彻底的一步：直接查 `actual_size` 绑定，一行取代整段，`LoopGenerate` 的
那份成了唯一实现。`materializeActualSize` 因此不再需要，除非将来出现没有绑定、又
确实可能有尾块的轴形态。

**`collectOuterAxisIndices` 未作为独立接口存在。** 两种索引来源合在
`findOuterAxisIndex` 里（先找 `scf.for` 的归纳变量，再查 `axis_index` 绑定）。
顺序不能反：block-inner 的循环体里还会为**源轴**打 `axis_index` 标记，先查绑定会把
循环轴的序号取成源轴的，两者差一个块偏移。

**空集合差回落 identity。** `axis` 表整张都在 tile 内时，tile 外面一层循环都没有，
偏移恒为 0——这就是 identity 搬运。返回 failure 让调用方回落，而不是建一个「偏移为
0 的调度计划」：两者的搬运长度来源不同（identity 取本地张量整体大小，调度路径取轴
尺寸），改道会换掉未切 tile 的单轴图的产物（`CodegenOrchestrationTest` 的夹具正是
这个形态，第一版就是这么挂的）。由 `TileCoveringEveryAxisFallsBackToIdentityPlan`
钉住。

## 8.3 S3（有效视图 + 二维 DMA）

S3 原始全量自检：lit 119/119、六个 UT 二进制 246 条全绿、`ctest` 24/24、CPU E2E
13 个 kernel 全绿。2026-09-21 补齐 reduce 下降与数值验收后，增量回归为：lit 123/123、
`AFIRDialectTest` 85/85、`ScheduleToAfirTest` 70/70；下列四个 `load_sum_store` CPU E2E
均完成生成、编译和数值校验：`16x32`、`16x37`、`17x32`、`arar_case3`。

- [x] 四条归约规则的 UT 通过，**每条都含反例**（7.2 的成对表）。
      `TransferPlanTest.cpp` 从 15 条增至 24 条，其中归约规则 8 条：
      | 规则 | 正例 | 反例 |
      | --- | --- | --- |
      | 1 范围为 1 | `UnitExtentAxisIsDropped` | `NonUnitExtentAxisIsKept` |
      | 2 零轴 | `ZeroStrideAxisIsDropped` | `ZeroStrideAxisIsKeptWhenPreviousLocalStrideIsNotUnit` |
      | 3 连续合并 | `ContiguousAxesAreMerged` | `AxesContiguousOnlyInGlobalSpaceAreNotMerged`、`UnprovableContiguityDoesNotMerge` |
      | 4 不跨 TileInner | —— | `MergeDoesNotCrossTileInnerBoundary` |
      规则 4 只有反例是对的：它本身就是一条「不许做」的规则，正例即规则 3 的正例。
- [x] 二维 Load / Store 的 lit 通过，**V2201 与 V3510 两个目标各一条 RUN 行**
      （`afir-to-ascendc-two-dim-transfer.mlir`）
- [x] **变异验证**：`emitTransfer` 的方向映射退回 `src = global / dst = local`
      之后，全仓只有二维 Store 的 `data_copy_pad_extend` 一条 CHECK 失败，
      与 8.3 的预测一致
- [x] 一维发射路径退化正确（`blockCount = 1`、两 stride = 0），由
      `afir-to-ascendc-scheduled-transfer.mlir` 与
      `OffsetSumsBlockIndexAndLoopInduction` 的 `m_One()` / `m_Zero()` 钉住
- [x] 计算算子对 > 1 项返回 `UnsupportedShape`。判据钉在**有没有报出来**上：
      `afir-two-dim-compute-rejected.mlir` 与二维搬运用例共用同一套轴与描述符，只在
      Load 与 Store 之间多一个 `afir.add`，断言 `'afir.add' op has no supported
      Ascend implementation`。
      把 `isLoweringSupportedForCompute` 改成恒真之后**全仓只有这一条失败**——说明
      它确实钉的是这条门禁，而不是撞上了别的既有检查（既有的秩/形状检查走的是类型化
      张量那条路，对 memref 形态的调度搬运不生效）
- [x] **生成 kernel 逐处比对**（语料是 13 个 kernel / 26 份源文件，不是正文原先写的
      48 个——用例在 44d7cb85 合并过一轮，48 是那之前的份数）。结论见下
- [x] ARAR case[3]（2 项）端到端跑通：`load_sum_store_arar_case3` 构造真实四维
      `[A, R, A, R]` 图，固定 plan 0 / template 3，生成、编译与 CPU 数值校验均通过。
- [x] **二维 CPU E2E 数值正确（910B）**——`load_sum_store_16x32`、`16x37`、
      `17x32` 分别覆盖基本 AR、归约轴尾块、保留轴尾块；真实 ARAR case[3] 额外覆盖
      合并归约外轴的跨轮累加。实现直接基于当前分支的 `TransferPlan::dims` 与
      `AscGraph` 轴谱系，不依赖旧分支的 `tiles` 结构。

### 生成 kernel 的比对：差异恰好三种

方法同 8.2（SSA 编号归一 + `constexpr` 集合单独比），基线是 `HEAD`，因为 S2 与 S3
在工作区里交织、没有提交边界可切。把全部 26 份的差异取并集，**只出现三种**：

| # | 形态 | 出处 | 判定 |
| --- | --- | --- | --- |
| A | 删 5 行（`V+V`、`V-c1_idx`、`V==V`、`V&V`、`V?V:V`）／加 6 行（`V*V` 提到循环外、`V+V`×2、`V*V`、`V<V`、`V?V:V`） | S2 的集合差求和 | 与 8.2 记录的形态**逐字相同**，说明 S3 对这部分零贡献 |
| B | `AscendC::ONE_BLK_SIZE` → `ONE_BLK_SIZE` | `ascendc/api/datacopy.h`，910B 上启用 `DataCopyPadExtend` 时改的 | 早于 S3，与寻址无关 |
| C | DMA 四参里的 `c1_i32` / `c0_i32` → `c1_idx` / `c0_idx`，且 `constexpr int32_t c0_i32 / c1_i32` 两行消失 | **S3 唯一的一维产物变化** | 见下 |

A 逐字复现了 8.2 的形态，等价于「S2 → S3 之间一维产物只差 C 这一处」。

**C 的判定：有意，且不是缺陷显形。** 退化路径此前在 `emitTransfer` 里现场造 i32 常量，
现在这四个参数一律由计划物化，而符号表达式物化出来的是 `index`。值完全相同（1 / 0 /
0），两边都套在 `static_cast<uint32_t>` 里，且 `c0_idx` / `c1_idx` 本来就已经在同一个
kernel 里（循环用着），所以产物净减两行 `constexpr`。

反方向（为保持逐字节不变而把它们转回 i32）要在真正的二维路径上多插 index → i32 的转换，
为一个常量名去给热路径加 op，不划算。**接受。**

数值侧由 13 个 CPU E2E 用例全绿背书（26 个断言），含唯一压住完整偏移式的
`load_add_store_128x128x128`。

## 8.3.1 S3 实施中发现的一条设计偏差

**`Expression::Parse` 的规范性检查，把「可证相等」翻译成了「证不出来」。**

4.2.2 把连续性判据定为 `Prove(ExpectEq(a, b)) == True`，对照 legacy 的
`StaticCheckEq(...) == kTrue`。照着写出来是这三步：

```cpp
Expression::Parse("ExpectEq(" + a + ", " + b + ")")  // ← 这里
    .Simplify()
    -> ExpressionAccess::Prove()
```

第一步就废了整条链。`Parse` 除了解析还校验**规范性**：它把解析结果重新序列化，与
入参逐字节比，不同就报错。而 `ExpectEq(s1, s1)` 的规范形式是 `True`——两边相等时
这段文本**必然**不规范。于是：

| a 与 b | `Parse` | 判据给出 |
| --- | --- | --- |
| 可证相等 | **失败**（规范形式是 `True`） | `Unknown` |
| 可证不等 | 失败（规范形式是 `False`） | `Unknown` |
| 证不出 | 成功 | `Unknown` |

三种情形全归 `Unknown`。同一个坑还有第二处：`multiplyExpressions` 拼的
`"(zt_size) * (s1)"`，规范形式是 `"(s1 * zt_size)"`，因子顺序不对也被 `Parse` 拒收。

**表现是「全绿但什么都没做」。** 四条归约规则一条都没触发过：正例（该剔该合）和反例
（该留该分）返回的都是「不合并、不剔除」，而反例全部通过。是写正例时才暴露的——这正是
7.2 要求每条规则成对的理由，只有反例的规则集与「规则整个不存在」无法区分。

**修法**：`Expression` 增开 `ParseRelaxed`，走同一个解析器但跳过规范性校验。
规范性是 `#afir.expr` **存储形态**的约束，对程序拼出来的中间文本没有意义。

仓里此前有两份绕法，都是从 `Parse` 的错误消息里把 `canonical form is '...'` 抠出来
重新解析（`Data.cpp` 与 `AfirBuilder.cpp` 各一份 `parseGeneratedExpression`）。两份
都改成调 `ParseRelaxed`，行为不变。

**顺带发现一处同源缺陷，本期不修**：`AfirBuilder.cpp` 的 `proveExpressionsEqual`
自己没用那份绕法，直接 `Parse`，于是在两边字面相同时返回 false——它唯一该返回 true
的情形。后果是动态形状 alloc 的 `memref.dim` 快路径从不命中，一律落到符号序言分支。
**数值不受影响**，只是多一段 IR。改它会动到每个动态形状 alloc 的 IR 形态，需要自己的
A/B，记入 S5 债务。

> **S5 已修**（只需把那一行 `Parse` 换成 `ParseRelaxed`；前两处 `Parse` 读的是存储形态的
> `#afir.expr`，本来就规范，不动）。A/B 的结论值得单记：CPU E2E 语料形状全是静态的，
> 这条路一次都没走到，产物 44/44 逐字节不变——**那个 A/B 是空的**。真正的观测点是
> `ScheduleToAfirTest` 里那条动态形状用例，它当时断言 `dimCount == 0`，而 walk 体里的
> 两句断言在计数为 0 时根本执行不到：它钉的是坏行为。修复后计数变 2，两句首次真正执行。
> 详见 8.5。

## 8.4 S4（外层循环，> 2 项）

最终自检：lit 124/124；`ReducePlanTest` + `TransferPlanTest` 37/37；CPU E2E 16/16，
其中 ARAR case[0..3] 四条均完成生成、编译和数值校验。S3 已完成的 `ctest` 24/24
未因 S4 增加测试项，本阶段以直接覆盖改动面的两组 UT、lit 与 CPU E2E 为验收入口。

- [x] `afir-to-ascendc-outer-loop-transfer.mlir` 通过：三项一层循环、四项两层，
      **V2201 与 V3510 两个目标各一条 RUN 行**。两段用例放在一个文件里，靠
      `--split-input-file` 分开，各自 `CHECK-LABEL` 起一段
- [x] **嵌套层数按 N − 2 钉死，不是「有循环就行」**。两段的 CHECK 都以
      `CHECK-NEXT: }` 收口（四项那段连着两个），少一层或多一层都会红。搬错块这件事
      文本上看不出来，但层数看得出来，所以判据落在层数上
- [x] 两侧 inner offset 各自正确。UT 的 `FourExtentsEmitTwoOuterLoops` 用的形态里
      **两侧步长逐层都不一样、顺序也对不上**（GM 是 s0/s1/zTb_size，UB 是
      y1t_size/zt_size/s1），正是 4.2.6 说的轴序不一致；断言直接比两侧步长不相等，
      任何一处串了空间都会相等
- [x] **变异验证**：把 DMA 内层项数从 2 临时改成 3，lit 结果为 123/124，且唯一失败
      是 `afir-to-ascendc-outer-loop-transfer.mlir`；三项与四项用例都在外层层数/stride
      断言处失败。恢复为 2 后重新跑回 124/124
- [x] 不发 `need_swap` 双份版本（L7）：发射器对每条搬运只建一条
      `DataCopyPadExtend`，没有运行期二选一的分支
- [x] 一维与二维路径保持原判据：≤ 2 项不发新增 `scf.for`；单轴 Reduce 的
      `first_actual` 与 `last_actual` 仍走原值/对齐口径。既有 lit 与 13 条 CPU E2E 全绿
- [x] ARAR case[0] / [1] / [2] 端到端跑通
- [x] **四个 tiling case 全部下降成功**
- [x] **> 2 项的 CPU E2E 数值正确（910B）**：case[0] / [1] / [2] 分别覆盖 4 / 3 / 3
      项 transfer effective view，输出逐项与主机端双重循环求和比较

## 8.4.1 S4 实施中发现的四条设计偏差

**一、行内偏移必须进 `ScheduledTileIndices`。** 4.1 的草案写的是「> 2 根时外层循环会
给两侧各加一个 inner offset，那是循环体内的局部量，不进本结构体」。实际上它必须进：
发射体要拿 UB 侧那一份去建 `LocalTensorSubIndexOp`，而发射体收到的就只有这一个结构体。
草案把它当成「循环体里的局部量」而排除，但发射体本身就在循环体里——局部量恰恰是它唯一
需要的东西。GM 侧那一份则直接叠进 `globalOffset`，因为那一侧本来就是一个偏移。

**二、`materializeActualSize` 写不出来，没有实施。** 3.5.1 的草案要的签名是「按轴查
实际尺寸：先查 `AxisValueRole::ActualSize`，查不到按 `{axis_size, tail_size, 归纳变量}`
现算」。但归约之后一项不对应任何一根轴——这正是 `TransferDim` 不带轴 id 的理由
（4.1）——「这一项的实际尺寸」问不到某一根轴头上。

实际用的是 S3 已经建好的那条路：**按符号替换**（`collectExtentBindings`）。extent 是
若干根轴尺寸的乘积，把其中每个会缩水的符号换成它的运行期实际值再乘，对合并项同样成立，
而且不需要新的公开接口。连带地，「`LoopGenerate` 的 ActualSize 计算提取为共用 helper」
也失去了对象——两边算的不是同一件事。

**三、`TransferPlanFailure::UnsupportedShape` 在寻址侧失去了产生点。** 项数上限是它
最后一个产生点，删掉之后 `buildTransferPlan` 只会报 `MissingInformation`。剩下的形状
天花板全在计算侧（elementwise 的多维限制、Reduce 的非连续 A/R 与 RA 限制），那边
直接报 `CapabilityReason::UnsupportedShape`，不经过本枚举。枚举值保留，理由写在
`TransferPlan.h`：判据本身没变，下一条落在寻址侧的形状限制要报的仍然是它。

连带两条 UT 失去了对象：
- `ThreeEffectiveExtentsAreRejectedAsUnsupportedShape` 删除，由
  `ThreeExtentsEmitOneOuterLoop` 取代——同一个形状，结论从「拒绝」变成「发一层循环」。
- `UnsupportedShapeDoesNotFallBackToIdentityPlan` 删除，它钉的判据（**明确拒绝不得回落
  identity**）改挂到符号闭合门禁的反例上。那是调度路径上仅剩的一处明确拒绝，而那条用例
  本来就用默认的 identity `indexing_maps`，判据一字不改地成立。

**四、Load 的替换值改成整块缓冲。** 此前 `replaceOp` 用的是带偏移的那个子视图；外层
循环接通后它落在循环体内，而 `afir.load` 的结果要给循环外面的使用者用——直接违反支配
关系。改成用未加偏移的缓冲本身：Load 的结果代表**整个** tile，子视图只是本轮 DMA 的
落点。

≤ 2 项时两者取值相同（`LocalTensorSubIndexOp` 在下标为 0 时折叠回张量自身），所以这条
改动**没有改变任何一维或二维产物**，由下面的逐处比对背书。

## 8.4.2 Reduce 对 > 2 项有效视图的闭环

最初把问题判断成「Reduce 也要按 transfer effective view 发 N − 2 层循环」，实施时用
case[0] 的真实 descriptor 验证后发现这个边界划错了：transfer 的 4 / 3 / 3 / 2 项是按
**GM 与 UB 两侧同时连续**归约的；数据搬完以后，Reduce 只消费 UB 局部布局。四个模板的
局部向量化轴都已经排成 `A...A,R...R`，因此应把连续 A 轴相乘成 `first`、连续 R 轴相乘
成 `last`，仍然只发一次二维 `ReduceSum`，不应再复制 DMA 外循环。

实现因此落在 `ReducePlan`：destination 的 `vectorized_strides` 将轴分成连续 A/R 两段；
`A/R/A` 或 `R/A/R` 这类无法保持局部内存顺序的形态明确拒绝。下降侧按轴绑定物化每根轴
的 nominal/actual extent，再合并成二维参数。R 侧只对齐最内轴：`[r0,r1]` 的 UB footprint
是 `r0 * align(r1)`，不是 `align(r0 * r1)`；`ReduceInit` 的 `inner_r` 继续传最内轴的原始
actual extent。该口径与 legacy `ReduceMergedSizeCodeGen` 一致，但实现不依赖旧 `tiles`
结构。

验收结果：case[0..3] 全部使用同一真实 ARAR 图和固定 template id 生成 kernel，910B CPU
逐项比较输出与主机端对轴 1、3 的双重循环求和，四条均通过。case[0]/[2] 还覆盖了多 R 轴
的 `isTailLast=false` 路径。

## 8.5 S5（债务清理与全矩阵）

- [x] **`#afir.axis_binding` 从弱引用变成不变量**（S2 偏差三留下的债）：标记本身
      应当让值免于死代码消除，或者轴绑定改由结构承载。S2 只堵住了
      `lowerSymbolicExpressions` 这一个 DCE 点，别处再来一个全函数贪心重写就会
      复现同一个静默失效

      **做法：func 级清单 + MemoryAllocate 结束校验**，不改值的 IR 形态。`LoopGenerate` 把这一趟打过
      的全部 `(轴, 角色)` 按序写进 func 的 `afir.axis_bindings`（见
      `AFIRAttributes.h`），在 `afir-memory-allocate` 最后一次符号改写之后逐条确认它们还在。
      2026-10-23 将原独立 pass 收入 allocator 的后置校验，移除独立注册；绑定保护与诊断保持。
      **只查存在，不查支配**：同一根轴在不同循环层各有一份，支配是取用点自己的问题。

      为什么不选另外两条：让标记算作 SSA 使用在 MLIR 里做不到（属性不建立 use），只能
      靠每个驱动各自过滤工作表，仍是约定；改成 `afir.axis_value` 这类非 Pure 的承载 op
      是真不变量，但会改动每个过 loop-generate 的 IR 形态、lit CHECK 大面积要动，且要
      在后端翻译前擦掉——代价与收益不成比例。

      副作用：清单是 func 属性，撞上了 AFIR 的 func 属性白名单
      （`AFIRDialect.cpp::isKnownFunctionAttribute`），34 个 lit 一度报
      `unknown AFIR function attribute`；加进白名单后恢复。
- [x] **`AfirBuilder::proveExpressionsEqual` 的反转**（S3 偏差留下的债，见 8.3.1）：
      改用 `Expression::ParseRelaxed` 后 `memref.dim` 快路径才会命中，动态形状 alloc
      的 IR 形态会变，需要自己的 A/B

      **A/B 的结论与预期不同，记在这里**：CPU E2E 的 22 个用例形状全是静态的，动态形状
      alloc 那条路一次都没走到，产物 44/44 逐字节不变——**这个 A/B 是空的，不构成验证**。
      真正的观测点在 `ScheduleToAfirTest.DoesNotMaterializeDynamicDimForBufferForm`：它
      断言 `dimCount == 0`，而 walk 体里那两句（源是 kernel 参数、结果有使用者）在计数为
      0 时一次都执行不到——写它的人预期有 `DimOp`，那个 0 是后来按当时的实际行为改的，
      即这条用例**钉的是坏行为**。改动把它翻成 2，两句断言首次真正执行且通过。用例改名为
      `MaterializesDynamicDimFromTheMatchingInput` 并补上来由。
- [x] `buildTransferPlan` 的默认参数不改变三个既有消费者的行为
      —— 查证：`DataMovement.cpp` 两处用默认的 `Automatic`（Load 取 `result_descs`、
      Store 取 `operand_descs`，按有没有结果分），`UnaryElementwise.cpp` 与
      `BinaryElementwise.cpp` 显式传 `Operand`（要输入那块 tile 的几何，与 legacy 以
      `x.actual_size` 为准同一条）。这条规则此前**零覆盖**，补
      `TransferPlanTest.PlanSideSelectsWhichDescriptorIsLocal`：把 GM 侧调成未切 tile，
      同一个 Load 上 `Automatic` 给 Scheduled、`Operand` 给 Identity，读错一侧立刻现形。
- [x] `StorageView::indexingMaps` 删除后全仓编译通过。**2026-10-02 复核**：该字段已不在
      `dialect/AFIR/Analysis/CanonicalBindings.h` 的 `StorageView` 里（删除本身在 S5 完成，
      见 3.2.5 的清理表），全仓构建 0 个 `error:`、ctest 94/94、lit 191/191。
      「全仓」指容器里 `scripts/test/run_autofuse_test.sh -u -m mlir --enable-mlir` 配出的
      422 个目标，含非 MLIR 的 `libaihac_codegen.so` 与 `pyautofuse.so`，不是只编 MLIR 子树。**`isQueueBuffer` 已于 2026-09-22 删除**：
      S4 收尾时它确实短暂有过一个调用方（`Implementations/Reduce.cpp` 要求输入与
      destination 都是队列形态），但 elementwise-reduce 融合接通时那条门禁被放宽成「两端都在
      UB」——TBuf 的取用点同样是下标 0，容器判别不是这里要问的问题。判据换成
      `getValueMemorySpace` 之后它重新变成零调用者，按 3.2.5 的原判删除。详见
      `2026-09-22-elementwise-multidim-calcount-design.md` 的「超出原范围的一处修改」
- [x] `getValueMemorySpace` 与 `resolveValueSpace` 判据一致
      —— 前者改成「先读 type 再回落 descriptor」。`memref` 的空间由类型自带，是一手事实；
      AscendC 的 local/global tensor 类型不带空间属性，那时才问 descriptor，既有行为不变。
      补 `AscendCAttributesTest.PrefersTheMemorySpaceCarriedByTheType`：没有 descriptor 时
      光凭类型就能答，descriptor 说的是别的空间时以类型为准。
- [x] map/descriptor 一致性：**记为 L4**，理由如下
      —— 实施时查清了一件 3.2.5 没写的事：`indexing_maps` 不是独立事实源，它由 bridge
      **从两侧 descriptor 的 `axis` 表派生**（`AfirBuilder.cpp::makeTransferIndexingMapsAttr`：
      迭代空间是两侧 axis 的并集，每一维的结果表达式按 axis 在并集里的下标取）。两者因此
      不可能在 bridge 产出的 IR 里打架；能打架的只有手写 IR。
      要在 dialect 里补 verifier，就得把那段派生逻辑复刻一遍去比对——把一处实现变成两处，
      正是本方案反复要避免的漂移。真正的修法是把派生本身收进 dialect、由 bridge 调用，
      那是接口迁移，不属于债务清理的范围。
      期间寻址侧一直只把 map 当「是不是直接投影」的门用（`buildScheduledPlan` 只调
      `isDirectProjection`，从不读它的置换），descriptor 是布局的唯一真源。
- [x] RCommon / RCore / AllLoad × AR / RA 六个组合的 E2E 数值正确（910B）
      —— **RA 下降与 AllLoad 都已不再是阻塞**：
      ① RA：`2026-09-21-afir-reduce-sum-ra-lowering-design.md` 的 S1～S3 已完成（2026-09-23），
      `Implementations/Reduce.cpp` 对 R 组单轴的 RA 放行，RCommon × RA 有三条 CPU E2E 数值用例。
      ② AllLoad：2026-10-01 实测**本来就能编**，五条钉住 AllLoad 候选的 CPU E2E 数值全对
      （`[16,32]`、`[16,37]`、`[17,32]`、`[8,1024]`、`[4,32,8]`），形态另由两条 lit 钉住；
      见 [AllLoad 模板设计](2026-10-01-afir-reduce-allload-template-design.md)。
      注意该文 2.2 同时改了**矩阵本身**：「AllLoad × 二维 RA」这一格**不存在**
      （二维 RA 根本不生成 AllLoad 候选），只能由 ≥3 维 ARA 覆盖。
      **2026-10-02 在 910B 实机跑了四个可达格子，10/10 数值通过**
      （`test/e2e/npu/reduce_template_matrix.py`，模板由 ATT `AUTOFUSE_DFX_FLAGS` 的 `--force_schedule_result` 钉，
      plan 身份由 dump 的调度图命名复核）：
      | 格子 | 钉的 plan | 形状 | 结果 |
      |---|---|---|---|
      | RCommon × AR | 0 | `(16,32)` / `(16,37)` / `(17,32)` | 3/3 |
      | RCommon × RA | 0 | `(37,16)` / `(32,17)` / `(37,17)` | 3/3 |
      | AllLoad × AR | 2 (`B0Y0_S2G0C0`) | 同 AR 三条 | 3/3 |
      | AllLoad × RA | 3 (`B0Y0_S3G0C0`) | `(4,32,8)` ARA | 1/1 |
      **2026-10-02 补完 RCore 两格，六个组合全部通过。** 那两条阻塞
      （`2026-09-21-afir-reduce-sum-ra-lowering-design.md` 3.10 的 `Workspace` 节点与 3.9 的
      tiling_key 命名分歧）都已被先前提交拆除，见该文两节的复测补注。
      | 格子 | 钉的 plan | 形状 | 结果 |
      |---|---|---|---|
      | RCore × AR | 1 | `(8,131072)` / `(16,65536)` / `(20,131072)` | 3/3 |
      | RCore × RA | 1 | `(131072,8)` / `(65536,16)` / `(131072,20)` | 3/3 |
      RCore 的形状必须**大**：它的 block 轴是 `MergeAxis({A_outer, R_block})`，nA 喂满核时
      `Ceiling(nR/a1Tt) = 1`，phase 2 只合并一份 partial，**跨核归约根本不触发**。上面
      RCommon / AllLoad 用的小形状 block_dim = 1，对 RCore 零区分度；这里取 nA = 8/16/20 <
      corenum(40)，核数必须从 R 方向来，跨核合并才真的走到。CPU E2E 同样测不到跨核合并，
      所以 `load_sum_store_rcore_ar_8x1024` / `_ra_1024x8` 两条只钉下降与两相管道。
      **判据同时从写死的 `atol=1e-5` 改成随归约长度走**（`margin × sqrt(n) × eps_fp32`）：
      131072 项 fp32 求和的典型量级是 362，但个别输出列会相消到 1 附近，此时累加舍入噪声
      相对该元素就是 1e-4 量级。`(131072,20)` 第一次就是这样挂的——20 个元素只错 1 个、
      偏差 4.58e-5、反推真值 ≈ 0.77（比典型值小 470 倍），是相消不是算错。真缺陷会整行错
      且偏差量级与真值相当，放宽到这个量级不会把它们放过。
- [x] 六个组合的 lit 在 V3510 上也各发一遍（950 无 E2E，只能比对文本）
      —— **2026-10-02 完成。** 目标串是 `Ascend950DT_950x`（`Ascend950DT` 不是内建目标，
      `af-opt` 报 `unknown builtin target platform`）。
      | 格子 | lit |
      |---|---|
      | RCommon × AR | `conversion/afir-to-ascendc-reduce-sum.mlir` |
      | RCommon × RA | `conversion/afir-to-ascendc-reduce-sum-ra.mlir` |
      | AllLoad × AR | `conversion/afir-to-ascendc-reduce-allload-ar.mlir` |
      | AllLoad × RA | `conversion/afir-to-ascendc-reduce-allload-ra.mlir` |
      | RCore × AR / RA | `conversion/afir-to-ascendc-reduce-rcore.mlir`（**新增**，两个 stage 同在一份模块里）|
      前四条此前只跑 910B1，各补一条 950DT_950x 的 RUN 行；RCore 两格原先**根本没有 lit**，
      新增的那条取 `load_sum_store_rcore_ar_8x1024` 钉 plan 1 的真实 dump，一份输入同时钉住
      两格——RCore 的两格不是两张独立的图，stage 0 是 phase 1（`[A, Rm_t]` 上 AR，写 GM
      workspace），stage 1 是 phase 2（workspace 的 `[Rm, A]` 上 RA，写 kernel 输出），
      另外钉住共用的 `workspace_tensor_ids`、跨 stage 的 `var_relations`，以及 stage 1
      **不发** `ReduceInit`。
      **五条在两个目标上的输出逐字节相同**（实测 `diff` 全为 0），所以共用同一组 CHECK ——
      reduce 的下降没有任何目标相关分支，这本身就是这条验收项的结论。

### 8.5.1 S5 验证记录（2026-09-23）

| 套件 | 结果 |
|------|------|
| lit | 125 / 125（V2201 与 V3510 双目标） |
| `make check-ut` | 全绿；`ScheduleToAfirTest` 70 / 70 |
| `ctest --test-dir build/autofuse/mlir` | 33 / 33 |
| `make check-e2e-cpu` | MAKE=0 |
| 产物 A/B | 22 个用例的 `kernel.cpp` + `autofuse_tiling_data.h` 共 **44 / 44 逐字节不变** |

产物零变化是 S5 的预期结果（9 节的分期表把 S5 标为"否"），但要注意它对**条目 2 不成立地
成立**：不是那条改动没效果，而是 E2E 语料里没有动态形状的图，见上面条目 2 的记录。

## 8.6 文档

- [x] 本文档按实施结果回写（接口表、用例表、偏差记录）
- [x] `docs/README.md` 索引状态更新

# 9 实施分期

```
S1 ──→ S2 ──→ S3 ──→ S4 ──→ S5
 │      │      │      │
 │      │      │      └─ 四个 tiling case 全部可下降（1.1.2 的硬约束在此闭合）
 │      │      └─ 第一步放开能力，也是唯一可能改变一维产物的一步
 │      └─ 逐字节可证不变
 └─ 行为零变化，可独立合入
```

| 期 | 内容 | 是否改变产物 | 验收 |
|---|---|---|---|
| **S1** | `.front()` 加 size 检查；符号闭合门禁；新增 `TransferPlanTest.cpp` | 否（零变化） | 8.1 |
| **S2** | 偏移改集合差求和；删四个轴字段与反查整段；`collectOuterAxisIndices` | **数值不变，op 序列变** | 8.2 |
| **S3** | 有效视图四条规则；`TransferDim`；DMA 四参 + 方向映射；计算侧拒绝 > 1 | 是 | 8.3 |
| **S4** | 外层循环（> 2 项）；`emitScheduledTransfer` 变发射器；`ScheduledTileIndices` 增 `localOffset`（`materializeActualSize` 未实施，见 8.4.1） | **否**（一维与二维产物逐字节不变，只有 > 2 项是新形态） | 8.4 |
| **S5** | 轴绑定清单 + MemoryAllocate 结束校验；`proveExpressionsEqual` 改 `ParseRelaxed`；`StorageView::indexingMaps` 与 `getValueMemorySpace` 两项清理；`TransferPlanSide` 补覆盖；map/descriptor 记为 L4 | **否**（44/44 逐字节不变） | 8.5 |

**为什么 S1 单列**：它是唯一一个"行为零变化"的档，可以独立评审、独立合入，
而它解决的是本方案里最危险的一条——`.front()` 的静默搬错。先把失败形态从
"静默"改成"诊断"，后面几档就都有一条安全底线。

**为什么 S2 单列**：偏移式的改动面很大（删四个字段 + 一整段反查），而它改变的是
**每一个**搬运与计算算子的地址计算。单列一步是为了让这一批 diff 能被干净地逐处
确认——它们应当全部是同一种形态（stride 从 `strides` 表物化），混进 S3 的归约规则
之后就分不清了。

**"计算侧拒绝 > 1 项"必须落在 S3**（而非更早）：它要挡的是"有效轴数 > 1 流进
elementwise"，而 S1/S2 阶段这条路被 `failure()` 堵着。S3 放开能力的同一次提交里
必须把它补上，否则二维 tile 会流进计算侧**只算第一行**。

**S3 与 S4 不能合并**：S3 放开 ≤2 项、S4 放开任意项。分开是为了让"二维 DMA 参数
算对了"和"外层循环建对了"两个结论各自可验——它们的失败表现不同（前者是
blockCount/stride 错，后者是漏搬整块）。

## 9.1 已知债务（不在本期）

| 编号 | 内容 |
|---|---|
| L2 | 二维 elementwise（接通方式已定：复用 `materializeQueueMemberElementCount` 的线性跨度，不要重写） |
| L3 | ~~`indexing_maps` 与 descriptor 的一致性 verifier~~ → S5 查证后归入 L4 口径，见 8.5：map 由 descriptor 派生，补 verifier 等于复刻派生逻辑；正解是把派生收进 dialect |
| L4 | 偏移算术全面改 checked（应与 `CanonicalResources.cpp` 同一写法整体推进，不在本方案单点加） |
| L7 | legacy 的运行期 `need_swap` 二选一（`CreateDmaCall` 对 > 2 根发两份 DMA 选 blockCount）。纯性能优化，触发条件是 > 2 根路径出现性能问题 |

**已销号**：
- ~~L1（> 2 根套外层循环）~~ → 提到本期 S4，见 1.1.2
- ~~L6（codegen 侧连续性合并）~~ → 提到本期 S3，见 4.2.2
- ~~L5（`TransferTileAxis` 的序号配方解耦）~~ → 结构改成 `TransferDim` 后
  自动消失，见 4.6.6

### 设计文档检查结果

- [x] 跨特性交叉影响：已按 `cross_feature_check.md` 逐项检查，影响集中在 Autofuse
      Codegen/Backend、AscendC API 交互与测试；SuperKernel、Python 绑定、打包交付
      不受影响（见 3.3.5）
- [x] 编码红线：已按 `编码红线.md` 逐条核对，重点处理了红线 2（外部输入作索引前
      校验——消除 `.front()` 无检查取用）与 G2（禁止硬编码芯片类型）；红线 3 的
      处理边界与理由见 3.3.4 与第 5 章
- [x] 测试设计：新增 `TransferPlanTest.cpp`（UT，无目标依赖）、二维 Store lit
      与外层循环 lit（**V2201 与 V3510 两个目标都发**）、910B CPU E2E 数值验证、
      一维产物逐字节比对，并要求一次变异验证（见 7.2）
- [x] 性能评估：编译时长**净减**（删掉整张 `asc_graph.axes` 的遍历）；执行性能
      一维不变（stride 为 1 时省略乘法）、二维与 > 2 维为新增形态（见 3.4）
- [x] 关键前提已实测，不靠推断：ARAR 四个 tiling case 的 `vectorized_axis` 根数
      （4.2.2）、stride 的符号闭合（4.2.5）、归约结果侧与 Load 侧的零轴差异
      （3.2.4），均来自 2026-09-21 的 `optimize_ut` 探针
