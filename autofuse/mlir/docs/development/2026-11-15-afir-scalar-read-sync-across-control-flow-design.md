# 简介

## 目的

修复 MLIR codegen 中标量读取的同步来源查询。写入者与读取者位于不同条件块或循环体时，旧查询可能找不到写入者，导致实现选择失败。本设计在 AFIR 计算算子尚未下降时分析写入流水线，将结果记录为属性，供能力查询和下降共用。

目标读者：Autofuse MLIR 的 AFIR、AFIRToAscendC 和同步模块开发、检视人员。

期望输出：明确存储匹配规则、写入流水线的分析范围、同步属性和多事件生成方式，以及验证边界。

## 范围

包含 `Synchronization.{h,cpp}` 中的存储分析与标量读取同步，新增的 `AFIRScalarReadSync` pass，以及 `Broadcast.cpp`、`BinaryElementwise.cpp`、`TernaryElementwise.cpp` 三个调用方。

不调整 `AFIRExecConditionPass` 的守卫粒度，不移动生产者，不修改 legacy 的同步判据；跨核同步、队列深度和双缓冲事件配对仍由原有模块负责。同一 PR 中的广播源行距和内段填充属于布局支持，不属于本同步方案。

涉及组件为 Autofuse 的 AFIR 和 AFIRToAscendC，不涉及 SuperKernel。

# 总体概述

## 软件概述

### 项目介绍

2026-11-14 的历史 fuzz 复测中，45 个失败 seed 有 22 个首先在 `afir.broadcast` 报错，其中 6 个涉及跨条件块的标量读取。典型结构如下，省略了 descriptor：

```mlir
%storage = ascendc.que_bind.alloc_tensor %queue
%scalar_buffer = ascendc.local_tensor.reinterpret_cast %storage
scf.if %guard {
  afir.truediv %input, %divisor outs(%scalar_buffer)
}
afir.pipe_barrier vector
scf.if %guard {
  afir.broadcast %scalar_buffer outs(%output)
}
```

`AFIRExecConditionPass` 为各算子分别建立 `scf.if`。旧 `findWriterEvent` 仅扫描当前 block 的前驱，无法进入前一个条件块查找写入者。此外，写入侧的 `alloc_tensor` 和读取侧的 `deque_tensor` 返回不同的 SSA 值，直接比较张量值也无法识别同一存储。

下降期间还有第二个问题：已下降的写入者不再携带 AFIR trait，重新分析可能遗漏其流水线。能力查询和下降即使调用同一个函数，也可能得到不同结果。

已有的 `afir.pipe_barrier vector` 只保证 V 流水线内部的顺序，不能代替 V 到 S 的同步，因为 `GetValue(0)` 在 S 流水线上执行。二元、三元 elementwise 的标量输入使用相同的同步 helper，也受此问题影响。

### 产品环境介绍

方案影响 `libaihac_codegen.so` 生成的 AscendC 源码。标量读取前仍生成 `FetchEventID`、`SetFlag`、`WaitFlag`，随后执行 `GetValue(0)`。不新增 runtime 或 aclrt 接口，不改变 Python 包、run 包和安装路径。

## 软件功能

1. 按存储容器查找写入者，覆盖兄弟 `scf.if`、循环体及程序序上位于读取者之后的写入者。
2. 同一存储存在多种写入流水线时，记录去重后的集合并逐条同步。
3. 在计算算子尚未下降时保存分析结果，避免下降顺序影响同步选择。
4. 无法确定同步来源时继续判为不支持；函数内无写入者时，仅按受限的容器规则兜底。

## 设计约束

- **排空语义**：在读取处连续生成 `SetFlag` 和 `WaitFlag`，等待指定流水线在该同步点之前的任务完成。因此，分析需要收集可能写入该存储的流水线，无须证明写入者和读取者的控制流条件相同。同步不负责保证输入数据已初始化，也不能使未来的写入提前完成。
- **保守同步**：遗漏必要事件可能造成数值错误；额外事件可能增加等待开销。
- **事件槽**：`FetchEventID` 不占用槽位，每次 `SetFlag` 后立即 `WaitFlag`，不交叠同类型事件的等待区间。
- **确定性**：流水线集合按 `afir::Pipe` 枚举值排序，不依赖哈希容器或指针地址。
- **执行顺序**：分析必须在资源物化之后、实现选择之前完成，下降阶段只读取属性。
- 不新增默认开启的高频日志，不改变对外 ABI/API。

## 假设和依赖关系

- 函数内的 AFIR 写入者可以通过 `AFIRDestinationStyleInterface` 和 `getOperandPartition` 枚举目的地。
- 沿用既有流水线分类：`AFIRTransferIn` 对应 MTE2，`AFIRTransferOut` 对应 MTE3，其余支持的写入者对应 V。
- 存储由队列或 TBuf 表示；队列的入队、出队操作可能切断张量层面的 SSA 关联。
- 容器兜底用于兼容外部写入或手工构造的 IR 片段，不是跨函数写入分析。其正确性依赖既有位置约定。

# 需求分析与设计

## 整体介绍

```text
LoopGenerate → MemoryAllocate → PipelineSync → ScalarReadSync → ExecCondition → 实现选择 → conversion
                                                分析并记录属性                     读取属性
```

这里的“尚未下降”指 AFIR 计算算子仍保留 trait；资源物化后，IR 中已经存在 AscendC 队列和 TBuf 操作，并非完全不含目标方言。

## 功能需求

### 功能需求 1：按存储收集写入流水线

1. 介绍

   给定算子及输入下标，收集该输入所在存储的写入流水线，不按控制流位置过滤。

2. 输入

   `Operation *anchor` 和 `unsigned inputIndex`。输入须属于算子的输入分区，且类型为 `ascendc::LocalTensorType`。

3. 处理

   沿 reinterpret cast、单目的地 AFIR 算子和队列/TBuf 的取张量操作回溯到存储容器。遍历所在 `func.func`，比较各写入目的地的存储根，按 trait 分类、去重并排序。

   同一队列的多个成员可能归为同一存储，因而额外收集其他成员的写入流水线。这是保守处理，不做成员级别的精确别名分析。

   匹配到 `AFIRTransferOut` 写入者时返回失败，沿用既有不支持行为。函数内无写入者时，仅对通过 `deque_tensor` 或 `tbuf.get_tensor` 获取的张量兜底：VECIN 返回 `TransferIn`，TBuf 返回 `Vector`。仅有 `alloc_tensor` 且无写入者时不兜底。

4. 输出

   `FailureOr<SmallVector<afir::Pipe>>`。失败表示输入或写入类型不支持；成功但集合为空表示无法确定同步来源。

### 功能需求 2：在分析和下降之间传递结果

1. 介绍

   在实现选择前记录结果，下降阶段不重新扫描混合 IR。

2. 输入

   不在 `afir.vector_func` 内部的 AFIR 算子。可能按标量读取的输入须为片上张量，且所有向量化轴的 repeat 均可证明为 1。

3. 处理

   将结果写为 `afir.scalar_read_sync`，数组按输入下标排列，每项为该输入的 `#afir<pipe ...>` 列表，例如 `[[], [#afir<pipe vector>]]`。所有项均为空时省略属性。

   能力查询优先读取属性；独立执行 `--afir-select-implementation`、属性缺失时，直接分析尚未下降的计算算子。下降仅读取属性，缺失或无效均视为无可用同步来源。

4. 输出

   算子属性及其对应的事件集合，可通过 IR dump 和 lit 检查。

### 功能需求 3：生成多事件同步

1. 介绍

   在标量读取前等待所有相关流水线。

2. 输入

   标量来源张量、事件集合和插入点。

3. 处理

   复用函数已有的 `TPipe`，按集合顺序依次生成 `FetchEventID`、`SetFlag`、`WaitFlag`，最后执行一次 `GetValue(0)`。空集合或缺少 `TPipe` 时返回失败。

4. 输出

   N 组同步操作和一次标量读取。N 为 1 时保持原有调用形式。

## 非功能需求

### 可维护性

存储分析和事件生成集中在 `Synchronization.cpp`，三个调用方复用同一接口。流水线到 `HardEvent` 的转换使用既有 `kHardEventTable`，不增加另一套映射。

### 可测试性

lit 检查跨条件块的分析、实现选择及下降结果，检查多流水线集合和顺序，并复用既有 singleton 正负例。当前提交没有新增独立分析 UT。CPU 模拟器可验证数值，但顺序执行不能证明真实跨流水线同步正确；该项需要 NPU 验证。

### 可移植性

不新增芯片型号判断，事件是否可用仍由 `kHardEventTable` 判断。

### 可靠性

属性读取前检查长度和类型。无法确定来源或无法映射事件时，整体判为不支持，不生成不完整的同步序列。

### 特性交叉影响

| 场景 | 适用性 | 分析说明 |
|------|--------|----------|
| SuperKernel Python 接口 | 不适用 | 不修改 Python 选项和 wheel。 |
| SuperKernel C++/AOT 接口 | 不适用 | 不修改接口、设备代码和 `libascendsk.so`。 |
| Autofuse 图优化 | 不适用 | 不修改 ASCIR 节点和依赖边；AFIR 只新增分析属性。 |
| Autofuse Codegen/Backend | 适用 | 改变标量读取的同步来源分析，需检查实现选择、生成代码和数值。 |
| AscendC API / Runtime 交互 | 适用 | 使用既有事件 API，立即配对，不跨区域持有槽位。 |
| Python/C++ 混合绑定 | 不适用 | 不修改绑定接口。 |
| 构建与打包 | 适用 | 在 AFIRToAscendC 中注册新增 pass 和源文件，不改安装布局。 |
| 测试与覆盖率 | 适用 | 新增 lit，复用既有回归；真实同步需 NPU 验证。 |
| 性能与日志 | 适用 | 新增函数遍历和可能的额外等待，不新增日志。 |
| 兼容性 | 适用 | 对外接口不变；扩大可支持的标量读取形态，保持未知来源的拒绝规则。 |

## 性能

### 编译时长

每个满足标注条件的输入执行一次函数遍历。设函数内算子数为 N，候选输入数为 S，总开销约为 O(S × N)，另含存储根回溯和表达式证明成本。当前未缓存结果，也没有专门的编译耗时测量，不能据此断言开销可忽略。

### 执行性能

单事件形态保持原有同步形式。多流水线写入同一存储时增加等待事件，容器级匹配也可能扩大等待范围。方案不新增数据搬运；实际同步开销需要在 NPU 上测量。

### 内存和产物大小

新增 pass、临时流水线集合和算子属性，属性规模随候选输入数增长。当前无独立的内存或产物大小测量。

## 接口设计

### 新增/修改接口描述

以下均为内部接口，声明位于 `Implementations/Synchronization.h`：

- `analyzeScalarWriterPipes(Operation *, unsigned)`：返回 `FailureOr<SmallVector<afir::Pipe>>`，分析存储的写入流水线。
- `scalarReadEvents(Operation *, unsigned)`：返回 `SmallVector<ascendc::HardEvent>`，供能力查询使用，允许在属性缺失时分析。
- `recordedScalarReadEvents(Operation *, unsigned)`：返回相同事件类型，供下降使用，只读取属性。
- `materializeScalarRead(Operation *, Value, ArrayRef<ascendc::HardEvent>, RewriterBase &)`：返回 `FailureOr<Value>`，生成同步及标量读取。
- `createAFIRScalarReadSyncPass()`、`registerAFIRScalarReadSyncPass()`：创建和注册 pass。

### 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|--------|----------|----------|------|
| 接口说明 | 是否需要接口评审 | 否 | 内部 helper，无对外暴露。 |
| 接口说明 | 是否需要补充文档 | 是 | 本文和接口注释。 |
| 接口兼容 | 行为是否兼容 | 是 | 单事件调用形式不变，新增多事件支持。 |
| 接口兼容 | ABI/API 是否兼容 | 否 | 不修改对外接口。 |
| 接口约束 | 约束不满足时是否清晰报错 | 是 | 能力层报告 MissingInformation，下降报告匹配失败。 |
| 接口测试 | 是否需要独立接口用例 | 是 | 当前由 lit 覆盖分析接口，未新增独立 UT。 |

## 软件设计

### 关键数据结构

`StorageRoot` 包含存储容器值和 `handedOver` 标记，后者仅用于受限兜底。分析返回 `afir::Pipe` 集合；转换为 `HardEvent` 后用于生成同步。`afir.scalar_read_sync` 为按输入下标对齐的 `ArrayAttr`，在分析后到下降前随算子保留。

### 关键技术/算法

采用提前分析并保存属性的方案。若只在下降期间扩大查询范围，就需要为目标方言另建流水线分类规则，且两个阶段仍可能得到不同结果。当前方案复用 AFIR trait，在信息完整的阶段分析一次。

pass 位于 `conversion/AFIRToAscendC/`，因为存储回溯依赖 AscendC 的队列、TBuf 和 reinterpret cast；不能将该依赖引入 AFIR transforms。属性名加入 `AFIR.h` 的已知属性列表，避免被未知属性校验拒绝。

### 流程设计

分析 pass 遍历函数，为可能按标量读取的输入收集流水线并记录属性；实现选择查询事件是否可用；下降读取记录，依次同步后读取标量。

无写入者且不能兜底、遇到 TransferOut、属性不可用或事件不能映射时，沿用既有的能力不足或下降失败路径。调用方是否回退 legacy 取决于 `mlir_codegen_config`，不是该 pass 的保证。pass 不移动算子，不需要结构回滚。

### 对子模块的修改

| 位置 | 修改点 |
|------|--------|
| `AFIRScalarReadSync.cpp` | 新增分析 pass、候选输入判断和属性记录。 |
| `AFIRToAscendCPipeline.cpp`、`Passes.cpp`、pass 头文件及 CMake | 注册 pass 并接入资源物化流程。 |
| `AFIRAttributes.h`、`AFIR.h` | 定义属性名并加入已知属性列表。 |
| `Implementations/Synchronization.{h,cpp}` | 按容器匹配存储、分析流水线、读取属性和生成多事件同步。 |
| `Broadcast.cpp`、`BinaryElementwise.cpp`、`TernaryElementwise.cpp` | 使用统一的集合接口。 |
| `test/lit/conversion/` | 新增跨控制流和多流水线分析用例。 |

### 错误处理

#### 系统错误

不直接申请设备内存，不打开文件，不新增 runtime 调用。临时集合由现有容器管理。

#### 接口错误

输入分区、类型和属性不满足要求时返回失败或空集合。能力查询报告 `CapabilityReason::MissingInformation`；下降通过 `notifyMatchFailure` 报告无法同步标量读取，不承诺额外输出存储根诊断。

# 安全检查

已对照 `docs/guidelines/编码红线.md` 检查：不新增敏感信息、平台判断和高频日志；属性访问检查下标与类型；不引入新的长度计算或设备内存申请；事件立即配对，不跨区域持有；不修改对外 ABI/API 或 ASCIR 依赖边；集合顺序固定。

容器位置只用于受限兜底。函数内存在写入者时，必须以实际写入流水线为准，避免将 Vector 写入的 VECIN 存储错误地映射为 `MTE2_S`。

# 兼容性检查

不改变脚本参数、配置格式、安装布局和对外接口。既有单事件路径保持调用形式；新增多流水线同步可能增加等待。无法确定来源的输入继续判为不支持，具体错误和回退行为由既有调用链决定。

# 测试设计

## 测试边界

lit 入口包括 `--afir-scalar-read-sync`、`--afir-select-implementation` 和 `--afir-to-ascendc-pipeline`，使用手写 AFIR，无须 runtime 桩。CPU E2E 验证生成代码及数值；NPU 验证真实异步流水线同步。`load_broadcast_scalar_read_sync.py` 已加入 NPU 清单，复用统一 runner；未新增独立分析 UT。

## 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|----------|------------|----------|----------|
| 功能 | 兄弟 scf.if 中的写入者 | 检查属性、实现选择及 V_S 生成。 | lit |
| 功能 | 程序序上位于读取者之后的写入者 | 检查收集范围，不验证数据初始化。 | lit |
| 功能 | MTE2 和 V 写入同一存储 | 检查记录的集合及顺序，不覆盖多事件完整下降。 | lit |
| 回归 | 队列/TBuf 的不同张量 SSA 值 | 复用既有 singleton 用例。 | lit |
| 异常 | 无法确定同步来源 | 复用 UNKNOWN 等拒绝用例。 | lit |
| 兼容性 | 既有广播和 elementwise | 运行相关 lit 和 CPU E2E。 | lit/E2E |
| 特性交叉 | 跨条件块的真实同步 | `abs(8×1) * (8×16384)`：CPU/NPU eager 与 MLIR 数值对比，并检查本轮 kernel 中兄弟条件块内的 Abs、V_S、GetValue/Duplicate 及块外 Mul。未接管、缺少事件或路径变化均报错。 | NPU E2E |
| 性能 | 函数遍历及额外等待成本 | 尚无专门测量，不作为已验证结论。 | 待测 |

## 测试命令

已按 MLIR README 配置构建并加载 CANN 环境后执行：

```bash
cmake --build build --target check-lit-mlir -j 8
bash scripts/test/run_autofuse_test.sh -u -s -m mlir -j 8
```

前者覆盖 lit，后者覆盖 MLIR UT、lit、ST 和 CPU E2E，不包含真实 NPU 执行。

在已配置匹配源码、构建产物及 CANN 的 NPU 环境中执行：

```bash
python3 autofuse/mlir/test/e2e/npu/run.py --id load_broadcast_scalar_read_sync --mode both
```

用例默认关闭 profiling，逐元素精确比较；`both` 依次运行 legacy 和 MLIR。生成代码检查仅用于 MLIR 路线，拒绝旧缓存或未融合执行；离线自测覆盖错误流水线、缺失等待、同一条件块及错误标量等反例。

2026-11-16 在 CANN 9.3.0 的 NPU 环境中完成上述用例验证：NPU eager、legacy、MLIR 与 CPU 参考值逐元素完全一致，MLIR 生成路径检查通过，`gate=none shortfall=no`，本轮新生成一个融合 kernel。该记录仅覆盖这一固定形状的回归用例。

# 验收标准

1. 跨条件块用例能记录正确流水线、选择标量实现并生成对应同步。
2. 多流水线用例记录完整、有序的集合；无法确定来源的既有负例仍被拒绝。
3. 相关 lit 和 CPU E2E 通过；全量结果须记录实际命令及对应提交，历史结果与本轮结果分开。
4. 真实 NPU 上的跨条件块标量读取通过数值对比；CPU 通过不能替代此项。
5. 文档与接口、兜底规则和实际测试范围一致，保留本轮未验证项。
