> 多维 tile 上 elementwise 的一维 calCount 设计
>
> 状态：**已完成**（2026-09-22 按实施结果回写）。验收标准九条全部达成，验证记录见文末。
>
> 实施中有一处超出原范围的修改（放宽 `Implementations/Reduce.cpp` 的取用形态门禁），
> 理由与影响记在「对子模块的修改」一节。
>
> 前置：[搬运的二维 tile 寻址设计](2026-09-20-two-dim-transfer-addressing-design.md) 的 S4——
> 没有它 `TransferPlan::extents` 不会多于一项，本方案无从谈起。

# 简介

## 目的

本文档定义 AFIR 多维 tile 上 elementwise 算子的最终下降方案：复现 legacy
`Tiler::TensorActualSize` 的语义，把多维局部视图折算成一个一维 `calCount`，继续调用现有一维
AscendC elementwise API，并移除 `isLoweringSupportedForCompute` 对多维计划的临时拒绝。

目标读者是 Autofuse MLIR 的 TransferPlan、AFIRToAscendC conversion 和 E2E 测试维护者。期望输出是可直接
实施、可按数值结果验收的 S4 收尾方案。

## 范围

本次包含：

- scheduled unary/binary elementwise 的多维局部跨度计算；
- capability 查询与实际 conversion 使用同一组布局判据；
- elementwise-reduce 融合图的结构和数值验证；
- 删除多维 compute 的临时拒绝门禁。

本次不包含：

- 新增二维 elementwise API；
- 修改 scheduler、ASCIR 图结构或 legacy codegen；
- 引入 `origin/af-mlir` 的旧 `tiles` 数据结构；
- 放开新的 broadcast-inline、dtype 或 AscendC API 家族能力；
- 修改 reduce 的 AR/RA 规划与 DMA 多维发射规则。

# 总体概述

## 软件概述

### 项目介绍

S3 已允许 `TransferPlan::extents` 表达多维有效视图。DMA 可把最内两项映射为
`blockCount/blockLen`，更多项由外层 `scf.for` 发射；reduce 则单独把语义 A/R 轴合成二维形状。

当前 unary/binary elementwise 复用了 DMA 发射器并把 `blockLen` 直接作为 API 长度，因此只能正确处理一维
有效视图。`TransferPlanCapability.h::isLoweringSupportedForCompute` 暂时拒绝 `extents.size() > 1`，避免只计算
最内一段的静默数值错误。这也导致 legacy 已支持的 elementwise-reduce 融合计划在 MLIR 后端被拒绝。

legacy 并不调用二维 elementwise API，而是通过 `Tiler::TensorActualSize` 计算当前局部 tile 从首个有效元素到
最后一个有效元素的物理跨度，再调用一次一维 API。

### 产品环境介绍

本功能只影响 Autofuse MLIR 生成的 AscendC kernel body。它不改变 CANN Toolkit、runtime、host launch、Python
包或 run 包接口，也不新增 AscendC API。生成代码继续调用目标 profile 已选择的一维 unary/binary API。

## 软件功能

- 对任意非空 `TransferPlan::extents` 计算当前 tile 的局部物理跨度；
- 对尾 tile 使用 actual extent，对 UB 行距使用 nominal local stride；
- unary、tensor-tensor binary 和 tensor-scalar binary 共用同一套跨度物化逻辑；
- 对布局不兼容或运行期绑定不完整的输入继续明确拒绝；
- 支持 elementwise 后接 reduce 的融合计划完整下降和数值执行。

## 设计约束

- `TransferPlan` 是当前分支唯一的 tile 寻址数据结构，不回引旧 `tiles`。
- AscendC elementwise API 签名不变，仍只接收一个一维 `calCount`。
- capability 查询是只读阶段，不得创建 SSA 或修改 IR。
- `LoopGenerate` 只负责产生 loop/index/actual-size 绑定，不承载 API 参数语义。
- UB 的物理行距由标称 tile 布局决定，尾块只缩 actual extent，不能缩 local stride。
- 只放开与 legacy 相同的局部跨度语义；布局不一致、inline broadcast 等既有不支持项不随本需求放开。
- 所有构建命令最多使用 `-j 8`。
- **一次一维调用会连同行间的对齐缝隙一起算**。`localSpan` 覆盖的是 `[base, base + span)`
  这一整段连续地址，其中包含多维布局里行与行之间的填充。方案成立依赖三条前置，缺一不可：
  1. 不越界——由 `actual <= nominal` 加「buffer 按 nominal 分配」保证；
  2. 缝隙里被改写的值不会被后续读取当成数据——归约路径由 `ReduceInit` 在归约前把 R 方向
     的填充重置成单位元；搬出路径由 `DataCopyPadExtend` 的 `blockCount/blockLen` 跳过缝隙；
  3. A 方向走不到 nominal 之外——`span` 用的是 actual extent。
  这三条都是**别的模块**的性质，不是本算法自证的。将来若新增一个「按 `LocalTensorGetSizeOp`
  长度读整块 UB」的下游算子，第 2 条即告破，表现是读到被 elementwise 改写过的填充值——
  数值错，无报错。届时要么该算子自己先重置填充，要么本方案退回按段发射。

## 假设和依赖关系

- `LoopGenerate` 已为会缩小的 TileInner 轴产生 `actual_size` 绑定。
- `TransferPlan::extents` 保持外到内顺序，且每项保存规范化后的 extent 与 local stride。
- actual extent 满足 `1 <= actual <= nominal`；该约束由现有 tiling/loop 生成契约保证。
- MemoryAllocate 按 nominal 局部布局分配完整 tile，故 actual local span 不超过 LocalTensor 容量。
- elementwise 的主 tensor 输入、destination，以及 tensor RHS（若存在）具有兼容的局部有效视图。不能证明时按
  `UnsupportedShape` 拒绝，不猜测。

# 需求分析与设计

## 整体介绍

方案不增加新 pass。相位 B 的 `LoopGenerate` 保持现状；相位 C 的实现选择只做结构判定；真正的 `calCount`
SSA 值在 `ConvertAFIRToAscendC` 的 unary/binary pattern 中物化。

```text
canonical TensorDesc + axis actual-size bindings
                    |
                    v
          buildTransferPlan(Operand)
                    |
          compatible local views?
              /             \
            no               yes
            |                 |
   UnsupportedShape    materialize local span
                              |
                              v
             AscendC::<Elementwise>(..., calCount)
```

## 功能需求

### 功能需求 1：物化 scheduled tile 的局部物理跨度

1. 介绍

   为 `TransferPlan` 增加只描述局部地址几何的物化接口。它不选择 API，也不创建额外循环。

2. 输入

   - 已成功构建且 `kind == Scheduled` 的 `TransferPlan`；
   - `plan.dims` 非空，顺序为外到内；
   - 当前 op 所在函数中的 tiling 参数和 actual-size 绑定；
   - 每项的 `extent` 与 `localStride`。

3. 处理

   新增接口：

   ```cpp
   FailureOr<Value> materializeScheduledLocalSpan(
       Operation *op, OpBuilder &builder, const TransferPlan &plan);
   ```

   计算公式与 legacy `Tiler::TensorActualSize` 等价：

   ```text
   localSpan = 1 + sum((actualExtent[i] - 1) * nominalLocalStride[i])
   ```

   - `actualExtent[i]` 使用 `collectExtentBindings` 后的 actual-size 替换结果；
   - `nominalLocalStride[i]` 使用 tiling 参数的标称绑定，不做 actual-size 替换；
   - local stride 为 0 的项贡献 0；
   - 连续轴被 effective-view 合并后，公式保持不变；
   - 一维退化为 `actualExtent`；
   - 任一表达式、符号绑定或 plan 前置条件缺失时返回 failure。

   例：二维局部布局的 actual extent 为 `[3, 5]`，标称行距为 8，最内 stride 为 1：

   ```text
   localSpan = 1 + (3 - 1) * 8 + (5 - 1) * 1 = 21
   ```

4. 输出

   返回 index 类型的 SSA `Value`，表示从 LocalTensor 基址开始、一次一维 elementwise 调用需要覆盖的元素数。

### 功能需求 2：elementwise 使用一维跨度下降

1. 介绍

   unary/binary lowering 不再把 DMA 的 `blockLen` 当作 elementwise `calCount`，也不再通过
   `emitScheduledTransfer` 为 compute 解释 `blockCount/blockLen`。

2. 输入

   - unary：主 tensor input 与 destination；
   - binary：lhs、destination，以及 tensor RHS 或 scalar RHS；
   - capability 阶段已经选择的 AscendC implementation。

3. 处理

   - Identity plan 保持现状，使用 `LocalTensorGetSizeOp`；
   - Scheduled plan 按主 tensor operand 构建计划，与 legacy 以输入 `x.actual_size` 为准的行为一致；
   - 对所有参与的一维 tensor 参数检查局部视图兼容性；scalar RHS 不参与布局检查；
   - 调用 `materializeScheduledLocalSpan` 得到 `calCount`；
   - 所有 LocalTensor 仍从当前 tile 的基址传入，不为有效视图的维度生成 compute 内层循环或子视图；
   - 创建既有 AscendC unary/binary op，参数列表和 API family 选择保持不变。

   “兼容局部视图”至少要求：有效 extent 数量相同，各项 extent 和 local stride 可证明相等。不能证明视为不支持，
   不能退化为字符串猜测或只比较 ranked tensor shape。

4. 输出

   产生一次一维 AscendC elementwise 调用。多维只体现在 `calCount` 的计算表达式中，不新增二维 API 或 kernel
   内层 elementwise 循环。

### 功能需求 3：能力查询与 conversion 保持一致

1. 介绍

   删除 `isLoweringSupportedForCompute(plan)` 的 `extents.size() <= 1` 临时门禁，但不能出现 capability 宣称支持、
   conversion 随后因同一结构失败的分裂。

2. 输入

   `buildTransferPlan` 结果、canonical descriptors、目标 profile 和既有 elementwise mapping。

3. 处理

   - capability 继续调用 `buildTransferPlan`；
   - 使用与 conversion 相同的纯只读局部视图兼容判据；
   - 多维本身不再是 `UnsupportedShape`；
   - 布局不兼容仍返回 `UnsupportedShape`；
   - descriptor 或符号闭包缺失仍通过 `toCapabilityReason` 返回 `MissingInformation`；
   - dtype、API family、tmp buffer 和 target feature 判据保持不变。

4. 输出

   可按 legacy 一维跨度语义下降的多维 elementwise 被选中；无法安全折算的计划得到明确 capability 诊断。

### 功能需求 4：elementwise-reduce 融合数值闭环

1. 介绍

   用非对齐尾轴和多维有效视图验证 elementwise、ReduceInit、ReduceSum 的组合语义。

2. 输入

   至少一个 `Load -> unary elementwise -> ReduceSum -> Store` 图，输入包含正负数，R 轴不是 32B 对齐整数倍，
   并选择能产生两项及以上 effective view 的 tiling candidate。

3. 处理

   - elementwise 先对 local span 调用一次一维 API；
   - ReduceInit 随后把 R 轴 padding 重置为 sum 单位元 0；
   - ReduceSum 按现有 AR/RA 规则归约；
   - CPU E2E 与 host reference 逐元素比较结果。

4. 输出

   生成 kernel 不再因多维 elementwise 被拒绝，且尾块、padding 和跨轮累加结果正确。

## 非功能需求

### 可维护性

- 跨 unary/binary 的局部跨度算法只实现一次；
- 不把 compute 语义继续塞入 `emitScheduledTransfer`，避免 DMA 与 elementwise 的参数模型互相污染；
- `TransferPlanCapability.h` 保留失败原因转换，删除临时 compute shape 门禁及过期注释；
- 注释同时写明 actual extent 与 nominal local stride 的口径差异。

### 可测试性

- 公式层使用 gtest 覆盖一维、二维、三维、zero-stride、连续合并和尾块；
- conversion 使用 lit 固定生成的 index 算术和一维 AscendC 调用；
- CPU E2E 验证 elementwise-reduce 的实际数值，不只检查源码文本。

### 可移植性

算法只使用 AFIR descriptor、index/arith op 和现有 AscendC elementwise op，不依赖芯片型号分支。两套 API family
沿用 target profile 选择，910B 与 950 的 capability 行为保持一致。

### 可靠性

- 不可证明布局兼容时保守拒绝；
- 不产生越过 LocalTensor nominal 容量的额外偏移；
- 不用 `min(span, capacity)` 掩盖坏 descriptor；
- 物化失败通过 pattern match failure 终止 conversion，并由已有 pipeline 诊断报告。

### 特性交叉影响

| 场景 | 适用性 | 分析说明 |
|------|--------|----------|
| SuperKernel Python 接口 | 不适用 | 不修改 Python 包、参数或 wheel |
| SuperKernel C++/AOT 接口 | 不适用 | 不修改 `libascendsk.so`、AOT 接口或符号 |
| Autofuse 图优化 | 不适用 | 不修改 ASCIR 图和融合 pass，只承接已有融合结果 |
| Autofuse Codegen/Backend | 适用 | 修改 MLIR TransferPlan 物化和 unary/binary lowering，新增 backend E2E |
| AscendC API / Runtime 交互 | 适用 | 不新增 API，只修正既有一维 API 的 `calCount`；无新增 runtime 生命周期 |
| Python/C++ 混合绑定 | 不适用 | 不修改 pyautofuse 或 CPython 扩展 |
| 构建与打包 | 不适用 | 预计不新增目标、依赖或安装文件；若新增测试目录，仅更新测试 CMake |
| 测试与覆盖率 | 适用 | 新增 gtest、lit 和 CPU E2E，覆盖正向与拒绝路径 |
| 性能与日志 | 适用 | kernel 每个 elementwise 多 O(rank) 个整数算术；调用次数与 legacy 相同，不新增日志 |
| 兼容性 | 适用 | 对外 API/ABI 不变；此前被拒绝的融合计划变为可下降，既有一维生成结果应逐字或等价不变 |

## 性能

### 编译时长

物化本身对最多数项 extent 做一次线性遍历，复杂度 O(rank)，无全图重复扫描。

但**能力查询这一侧的开销要大一个量级**，实施时才看清：`checkElementwiseLocalViews` 为参与的
每份 descriptor 各建一个 `TransferPlan`（unary 两份、binary tensor-tensor 三份），再对每一项
extent 做两次 `proveEqual` 符号证明；而能力查询对每个候选 implementation 各调一次，二元 op 有
tensor-tensor 与 tensor-scalar 两个候选。于是一个 `afir.add` 实际会建 6 个计划、跑 O(rank) 次
表达式解析与化简。

这仍在可接受范围（实测 lit 全集 125 个用例 2.1 秒），但它是**符号层**开销，不是整数算术开销；
将来若候选实现数量或 descriptor 份数增长，这里是第一个要加缓存的地方，不要按"O(rank) 一次遍历"
去估。

### 执行性能

采用 legacy 的单次一维 API 调用，不按行增加调用次数，避免额外 `scf.for` 和 vector API 启动开销。动态
`calCount` 增加 O(rank) 个 index 加减乘，位于每个 tile 的 elementwise 调用前，与 legacy 字符串表达式的运行期
计算量同阶。

### 内存和产物大小

不新增 buffer、workspace 或常驻对象。生成 kernel 只增加少量 index 算术，产物大小影响可忽略。

## 接口设计

### 新增/修改接口描述

新增内部接口：

```cpp
FailureOr<Value> materializeScheduledLocalSpan(
    Operation *op, OpBuilder &builder, const TransferPlan &plan);
```

- 参数：当前 AFIR op、插入 SSA 的 builder、基于主 tensor operand 构建的 scheduled plan；
- 返回：成功时为 index 类型的 local span，失败时为 `failure()`；**identity 计划一律 `failure()`**，
  不给兜底长度——兜底出来的长度算多算少都不报错，调用方在 identity 分支本来就走
  `LocalTensorGetSizeOp`，不该调到这里；
- 生命周期：返回 Value 由当前 MLIR region 持有，不新增所有权；
- 可见性：仅 Autofuse MLIR 内部使用，不导出动态库 ABI。

另新增一个纯只读的 elementwise local-view compatibility helper（`ElementwiseTransferPlan.{h,cpp}`），
capability 和 conversion 调用同一实现。它需要按**显式给定的 local descriptor** 建计划（而不是按
op 的方向反查），因此同时新增：

```cpp
FailureOr<TransferPlan> buildTransferPlanForDescriptor(
    Operation *op, TensorDescAttr localDescriptor, TransferPlanFailure *reason = nullptr);
detail::TruthValue proveEqual(ExprAttr lhs, ExprAttr rhs);   // 原为 TransferPlan.cpp 内部函数，上移为接口
```

`buildTransferPlan` 与 `buildTransferPlanForDescriptor` 共用同一个主体，差别只有 descriptor 的来源；
不得各写一份，两份会分开漂。

helper 的兼容性判定返回 `LocalViewCompatibility`（三值枚举），**不包在 `std::optional` 里**：
没有"问不出结果"这第四种状态，包起来只会让调用方以为有。判定同时可选地输出参与比较的那组计划的
`TransferPlanKind`——秩的门禁只对 identity 成立，而这个区分只有计划知道。

### 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|--------|----------|----------|------|
| 接口说明 | 是否需要接口评审 | 否 | 仅内部 C++ 接口 |
| 接口说明 | 是否需要补充文档 | 是 | 本设计文档及头文件注释 |
| 接口兼容 | 行为是否兼容 | 是 | 一维保持不变，多维由拒绝变为支持 |
| 接口兼容 | ABI/API 是否兼容 | 是 | 不修改导出接口和结构体布局 |
| 接口约束 | 约束不满足时是否清晰报错 | 是 | MissingInformation 与 UnsupportedShape 分流 |
| 接口测试 | 是否需要独立接口用例 | 是 | TransferPlan gtest + conversion lit |

## 软件设计

### 关键数据结构

不新增持久数据结构。继续使用：

- `TransferPlan::extents`：有效视图，外到内；
- `TransferDim::{extent, localStride}`：跨度公式输入；
- LoopGenerate 产生的 `AxisValueRole::ActualSize` SSA 绑定；
- nominal tiling parameter binding：物化稳定的 UB 行距。

### 关键技术/算法

伪代码如下：

```cpp
FailureOr<Value> materializeScheduledLocalSpan(Operation *op, OpBuilder &builder,
                                               const TransferPlan &plan) {
  if (plan.kind != TransferPlanKind::Scheduled || plan.dims.empty()) {
    return failure();
  }

  auto actualBindings = collectExtentBindings(func, op, graph, plan.dims);
  auto nominalBindings = collectTilingParamBindings(func);
  Value span = constantIndex(1);
  Value one = constantIndex(1);

  for (const TransferDim &item : plan.dims) {
    Value extent = materialize(item.extent, actualBindings);
    Value stride = materialize(item.localStride, nominalBindings);
    Value distance = muli(subi(extent, one), stride);
    span = addi(span, distance);
  }
  return span;
}
```

不使用 `emitScheduledTransfer`，因为它把最内两项解释为 DMA 的二维参数；compute 需要的是整个局部地址区间的
一维长度。公式不会生成 compute 内层循环，也不会使用 GM offset/stride。

### 流程设计

正常流程：

1. capability 从主 tensor operand 构建计划；
2. 检查参与 tensor 的局部视图兼容；
3. 选择既有 unary/binary implementation；
4. conversion 再执行同一结构检查；
5. Identity 使用 LocalTensor size；Scheduled 物化 local span；
6. 创建一个 AscendC elementwise op；
7. reduce 按现有顺序执行 ReduceInit 和 ReduceSum。

异常流程：

- descriptor/绑定缺失：`MissingInformation`；
- tensor 局部布局不兼容：`UnsupportedShape`；
- conversion 物化失败：match failure，pipeline 失败，不生成部分正确的 kernel；
- 不采用回退到 `blockLen`、默认 stride 或截断 count 的策略。

### 对子模块的修改

- `dialect/AFIR/Analysis/TransferPlan.{h,cpp}`
  - 新增 `materializeScheduledLocalSpan`；
  - 新增 `buildTransferPlanForDescriptor`，与 `buildTransferPlan` 共用同一主体；
  - `proveEqual` 由匿名 namespace 上移为接口；
  - 复用现有 actual/nominal 符号绑定和表达式物化；
  - 新增公式级 UT。
- `conversion/AFIRToAscendC/Implementations/UnaryElementwise.cpp`
  - scheduled path 改用 operand plan 和 local span；
  - 删除对 `emitScheduledTransfer` 的 compute 复用。
- `conversion/AFIRToAscendC/Implementations/BinaryElementwise.cpp`
  - 与 unary 同步；scalar RHS 不参与布局检查；
  - 「右操作数是不是 tile」取 `mapping_->rhsForm`，不取实际值的类型——conversion 侧只有模板
    参数这一个来源，两处取同一个来源才不会出现"能力说支持、改写匹配失败"的分裂。
- `conversion/AFIRToAscendC/Implementations/TransferPlanCapability.h`
  - 保留 `toCapabilityReason`；
  - 删除 `isLoweringSupportedForCompute` 和临时限制说明。
- `conversion/AFIRToAscendC/Implementations/ElementwiseTransferPlan.{h,cpp}`（新增）
  - 承载 local-view compatibility 判据，避免 unary/binary 与 capability/conversion 四处分叉；
  - 不并入只负责 API family 的 `ElementwiseApiFamily`，避免职责混杂。
- `conversion/AFIRToAscendC/Implementations/Reduce.cpp`（**超出原范围声明**，见下）
  - 输入/destination 的取用形态门禁由「必须是队列」放宽为「必须在 UB」。
- `test/lit/conversion`
  - 将 `afir-two-dim-compute-rejected.mlir` 改为正向下降用例 `afir-two-dim-compute.mlir`；
  - 新增布局不兼容负例 `afir-two-dim-compute-mismatch.mlir`。
- `test/e2e/cpu/cases/load_neg_sum_store`（新增）
  - `Load -> Neg -> ReduceSum -> Store`，二维非对齐 R 轴与四维 ARAR candidate 两个注册。

#### 超出原范围的一处修改：reduce 的取用形态门禁

§范围写明「本次不包含：修改 reduce 的 AR/RA 规划与 DMA 多维发射规则」。实施时发现**不改它，
功能需求 4 无法达成**：融合之后 `afir.sum` 的输入是上一个 elementwise 的输出，那是 TBuf 而不是
队列，被 `Reduce.cpp` 的 queue-only 前置条件挡住，表现为 `failed to legalize operation 'afir.sum'`。

放宽的依据：两类缓冲的取用点都是下标 0——队列的 tile 偏移由 `DeQue` 吃掉，TBuf 按 ATT 的
`b{id}_size` 只装一个 tile、块内每轮整块重写。这与 elementwise 侧对同一批缓冲用的是同一条判据，
且 `load_exp_sqrt_store` 这类链式用例早就在这个前提下跑通。判据改成「两端都在 UB」而不是「不是
队列」，是因为后者对 GM 值也成立，会给一个 GM 值发 UB 的取用形态。

副作用：`afir::isQueueBuffer` 失去唯一调用方，一并删除（它的注释原本就写着「调用方是
`Implementations/Reduce.cpp`」，留着会是一句错的话）。

**仍存在的不对称**：reduce 的能力查询不检查 buffer space，所以"能力说支持、改写匹配失败"的分裂
在 reduce 上依然存在，只是触发面变小了。功能需求 3 只对 elementwise 要求了两侧一致，reduce 没要求。
要消除它，得把这条 space 判据也提到 `ReduceCapabilityProvider::queryLowering` 里。

### 错误处理

#### 系统错误

本功能不申请 host/device 资源，不新增文件或 runtime 调用。MLIR op 创建失败沿用 pass failure 机制。

#### 接口错误

错误分类保持稳定：信息不完整报告 `MissingInformation`，已知布局不能由当前一维调用安全覆盖报告
`UnsupportedShape`。不得把不兼容布局伪装成缺少属性。

## 安全检查

- [x] 不新增敏感信息、公网地址或芯片型号硬编码。
- [x] extent/stride 来自 canonical descriptor 和 tiling SSA，不直接使用未经校验的用户字符串。
- [x] actual extent 由既有 tiling 契约限制在 nominal extent 内，local span 与 allocation 使用同一 nominal
  布局；坏 descriptor 不通过截断掩盖。
- [x] 不新增除法，不引入除 0；加减乘使用 MLIR index 类型，常量表达式继续由符号安全检查处理。
- [x] 不新增资源所有权和异常释放路径。
- [x] 不修改 ABI/API、runtime 生命周期或非开放接口。
- [x] 图结构与控制边不变，只修改一个 op 的下降参数；生成顺序确定，不依赖无序容器遍历。
- [x] 不新增高频日志。

实现时必须额外确认：动态 actual extent 的非零契约已由 LoopGenerate/调度器验证；若现有入口允许零维 tensor，
则本需求不得用 `extent - 1` 直接下降，需先为零维建立统一语义并补充 guard，该问题不能用无符号反转掩盖。

## 兼容性检查

- 对外 C/C++、Python、AscendC API 和生成 artifact 文件名均不变；
- 一维 elementwise 的 IR 与生成调用保持不变；
- 多维 elementwise 从明确拒绝变为支持，是预期能力扩展；
- layout mismatch 仍拒绝，不扩大 broadcast 或跨布局能力；
- legacy codegen 不修改，作为行为基线保留。

## 测试设计

### 测试边界

测试输入从 canonical AFIR descriptor 开始，出口分别为 TransferPlan 物化 IR、AscendC dialect IR、生成 kernel
和 CPU 数值结果。不 mock scheduler 的旧 `tiles`；lit/UT 直接构造当前 `AscGraphAttrGroupsAttr`、TensorDesc 和
axis binding。

### 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|----------|------------|----------|----------|
| 功能 | 一维 scheduled span 等于 actual inner | 检查物化常量/SSA | gtest |
| 功能 | 多维逐项：extent 取 actual 绑定、stride 取 nominal | 拆解左结合加法链，逐项比对 SSA | gtest |
| 功能 | 三维及连续轴合并前后跨度等价 | 比较物化结果 | gtest |
| 功能 | zero-stride 轴不增加 span | 正向用例 | gtest |
| 功能 | unary 多维 plan 生成一次一维 API，count 为 local span | FileCheck AscendC IR | lit |
| 功能 | binary tensor/scalar 两类共享多维 count | FileCheck 两种目标 op | lit |
| 异常 | operand/destination 局部布局不兼容 | 检查 UnsupportedShape 诊断 | lit |
| 异常 | actual-size 绑定缺失 | 检查 MissingInformation/转换失败 | gtest/lit |
| 兼容性 | 既有一维 elementwise 输出不变 | 运行现有 conversion corpus | lit |
| 特性交叉 | 非对齐 R 轴的 elementwise-reduce 融合 | 与 host reference 比较 | CPU E2E |
| 特性交叉 | ARAR 至少一个多维 candidate | 固定候选并检查数值 | CPU E2E |
| 性能 | 单次一维调用，无新增逐行循环 | 检查生成源码无新增 compute `for` | lit/手工 |

建议 E2E 采用 `Load -> Neg -> ReduceSum -> Store`：输入使用正负混合值，reference 为 `sum(-x)`；R 轴选
37 等非 32B 对齐长度，确保 elementwise 覆盖 padding 后，ReduceInit 仍能在 reduce 前恢复 sum 单位元。

实施时的两点偏差：

- 「二维 `[3,5]`、行距 8 得到 21」这条写不成值级断言。物化出来的是 tiling 符号上的 `arith` 算术，
  UT 夹具里没有常量可折，fold 不到字面量 21。改为**逐项结构断言**：把左结合加法链拆回
  `1 + Σ (actual_i - 1) * stride_i`，按 `plan.dims` 的顺序逐项比对两个因子各自取自哪个 SSA
  （actual 绑定 vs. 标称 tiling 参数）。这比数一数有几个乘法强，也正好钉住尾块口径。
- 布局不兼容负例要**同时**换掉 `destination_descs`、`result_descs` 和下游 Store 的
  `operand_descs`。只换第一处会先撞上 verifier 的
  `InvalidBinding: tied result and destination descriptors differ`，只换前两处会撞上
  `conflicting canonical descriptors for one SSA value`——两种情况钉住的都不是这条判据。

增加两项 mutation 验证：

1. 把 `calCount` 临时改回 `blockLen`，融合 E2E 必须失败；
2. 恢复 `extents.size() <= 1` 门禁，正向多维 conversion lit 必须单点失败。

### 测试命令

按仓库实际的目标名回写（均在 dev 容器内，`/workspace/build`）：

```bash
make check-ut -j 4                                            # 全部 gtest，含 AFIRDialectTest
/opt/llvm/bin/llvm-lit -s /workspace/build/autofuse/mlir/test/lit
make check-e2e-cpu -j 2                                       # 生成并运行 CPU E2E kernel
ctest --test-dir /workspace/build/autofuse/mlir               # UT + ST + E2E 汇总
```

两处容易踩空、都静默返回 0：

- **lit 必须指向 build 树**。指向源码树会以
  `AttributeError: 'TestingConfig' object has no attribute 'autofuse_mlir_test_exec_root'` 失败。
- **lit 不在 ctest 里**。只跑 ctest 不覆盖 125 个 lit 用例。

另：容器可能在长构建中途被重建，留下 0 字节的 `.o`/`.a` 而 make 认为它们是最新的（表现为跨模块
undefined reference）。每轮构建前先
`find /workspace/build \( -name "*.o" -o -name "*.a" \) -size 0 -delete`。

## 验收标准

- [x] `materializeScheduledLocalSpan` 对一维、二维、三维、zero-stride、合并轴和尾块用例结果正确。
  —— `TransferPlanTest` 的 `LocalSpanOfOneExtentIsTheInnermostActualExtent` /
  `LocalSpanUsesActualExtentAndNominalStride`（三项，同时钉住 actual 与 nominal 的口径差）/
  `LocalSpanOfMergedAxesHasOneTerm` / `LocalSpanOfZeroStrideExtentContributesNothing` /
  `LocalSpanOfIdentityPlanFails`。尾块由第二条覆盖：它断言 extent 取 actual 绑定、stride 不取。
- [x] unary/binary scheduled lowering 不再调用 DMA 语义的 `emitScheduledTransfer` 来取得 elementwise count。
- [x] 多维 elementwise 仍只生成一次一维 AscendC API 调用，`calCount` 与 legacy
  `TensorActualSize` 等价。—— `afir-two-dim-compute.mlir` 钉住 IR；生成的 kernel 里
  `NegExtend<float>(dst, src, 1 + (A-1)*行距 + (R-1)*1)`，无新增 compute `for`。
- [x] `isLoweringSupportedForCompute` 临时门禁删除；布局不兼容仍明确报告 `UnsupportedShape`。
  —— 负例 `afir-two-dim-compute-mismatch.mlir`，观测点是
  `'afir.add' op has no supported Ascend implementation`。
- [x] 一维既有 lit、UT 和 CPU E2E 全部通过，生成行为无非预期变化。
- [x] elementwise-reduce 融合的非对齐尾轴 CPU E2E 数值通过。—— `load_neg_sum_store_16x37`。
- [x] 至少一个 ARAR 多维 candidate 的 elementwise-reduce 数值通过。—— `load_neg_sum_store_arar_case3`，
  搬运有效视图两项。**只覆盖了 candidate 3**：candidate 0/1/2 的 `TILE_ACCESSOR` 切的是另一根轴，
  而 `KernelTest.cpp` 里报告覆盖用的轴长是按注册的维数写死的（二维取第 0 维、四维取第 2 维），
  再加一个 candidate 需要把"切哪根轴"也做成 per-case 的编译期定义。没做，因为验收只要求一个。
- [x] 两项 mutation check 均只打破对应新增验收用例。见下。
- [x] 不修改 legacy codegen、旧 `tiles`、对外 ABI/API、打包内容或 runtime 生命周期。

### 验证记录（2026-09-22，Ascend910B1 + Ascend950DT_950x）

| 套件 | 结果 |
|------|------|
| lit | 125 / 125 |
| `make check-ut` | 全绿，0 处 FAILED；`TransferPlanTest` 30 / 30 |
| `ctest --test-dir build/autofuse/mlir` | 33 / 33（含 22 个 CPU E2E） |
| `make check-e2e-cpu` | MAKE=0，17 个 kernel 重新生成 |

变异验证（两条都跑完并回退）：

| 变异 | 打破的用例 | 是否只打破对应用例 |
|------|------------|--------------------|
| `localSpan` 只累计最内一项（等价于退回 `blockLen`） | lit `afir-two-dim-compute.mlir`；UT `LocalSpanUsesActualExtentAndNominalStride`、`LocalSpanOfZeroStrideExtentContributesNothing`；E2E `load_neg_sum_store_16x37`、`load_neg_sum_store_arar_case3` | 是。其余 124 个 lit、20 个 E2E 全部仍通过——一维计划的 span 恰好就是最内项，所以既有用例本就不该动 |
| 恢复 `extents.size() <= 1` 门禁 | lit `afir-two-dim-compute.mlir` | 是，单点 |

### 设计文档检查结果

- [x] 跨特性交叉影响：已按 `cross_feature_check.md` 检查；仅 Autofuse MLIR codegen、测试和性能相关。
- [x] 编码红线：已检查整数范围、图改写确定性、ABI/API、资源和 runtime 生命周期约束。
- [x] 测试设计：覆盖 TransferPlan gtest、conversion lit、CPU E2E、负例与 mutation check。
- [x] 性能评估：保持一次一维 API 调用，仅新增 O(rank) index 算术，无新增 buffer 和日志。
