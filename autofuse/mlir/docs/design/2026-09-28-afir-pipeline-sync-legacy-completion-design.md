# MLIR 流水线同步场景补全设计

- 日期：2026-09-28（2026-09-29 随 origin/af-mlir `84a9a00a` 重新核对全部代码锚点）
- 状态：已实现（2026-10-09，实现记录见文末；lit 全量通过，UT/E2E 见验证记录）
- 组件：Autofuse / MLIR（`autofuse/mlir/`）
- 关联设计：[AFIR 对外 DSL 架构设计](2026-09-09-afir-public-dsl-architecture-design.md)（ADR-11/12/19/20、3.6 同步）、[AFIR 分层重构](2026-09-14-afir-layering-refactor.md)、[AFIR→AscendC pass 管线设计](2026-08-05-afir-to-ascendc-pass-pipeline-design.md)

# 简介

## 目的

MLIR 路径目前不支持若干 legacy 路径已支持的同步场景，典型如 load 直连 store：该场景需要双端队列 `TQueBind<VECIN, VECOUT>`，当前编译直接报错。本文档：

1. 给出缺失场景的完整清单及代码定位；
2. 给出补全方案：各场景的目标产物、修改点、验证方式。

目标读者：Autofuse 开发者。

## 范围

**包含**：下文缺失清单 7 项（S1–S4、M1–M3）在 AFIR / 资源模型 / 下降层的实现，及配套 UT / lit / E2E 测试。

**不包含**：

- 标量、Cube、多核算子的同步分支（B 类）：这些算子尚未迁移到 MLIR 路径，前端不产生其 AFIR，同步缺口无法暴露。本次落地的机制（显式同步下降、双端队列物化）与算子类型无关，B 类迁移时按 S1–S4 模式扩展分支即可；
- 循环多缓冲轮转的 WAR 形式化检测（RaceCheck 既有课题）；
- ascendc 方言层既有 `InsertQueSync` / `VerifySync` / `EraseSync` transforms；
- SuperKernel 组件、子图间运行时依赖调度。

# 总体概述

## 软件概述

### 术语

| 术语 | 定义 |
|------|------|
| AFIR | Autofuse MLIR 中间表示 |
| pass | 编译流水线中的单个变换步骤 |
| `AFIRPipelineSync` | 资源物化后自动插入引擎间同步的 pass |
| MTE2 / MTE3 / Vector | 搬入引擎 / 搬出引擎 / 矢量计算引擎 |
| 队列同步 | 物化 `ascendc` 的 TQueBind 入队/出队操作，由队列隐式完成引擎间同步 |
| 显式同步 | `afir.event_record` / `afir.event_wait` 操作对（token），下降为 `SetFlag` / `WaitFlag` |
| 下降（lowering） | AFIR 转换为 ascendc 方言并打印为 AscendC C++ 的过程 |
| WAR | Write-After-Read 反依赖 |

### 项目介绍

AFIR 将片上存储与引擎间同步解耦（ADR-12）：存储以 memref 表示，同步由编译器插入显式操作表达。生产管线中，`AFIRMemoryAllocate` 完成资源物化后，`AFIRPipelineSync` 按生产者与消费者所在引擎逐条转移插入同步，当前支持两种方式：

- **队列同步**（MTE2→Vector、Vector→MTE3）：物化 `ascendc` 的 `TQueBind{Alloc,Enque,Deque,Free}TensorOp`；
- **显式同步 / 单管屏障**（Vector→Vector）：产出 `afir.pipe_barrier`，由同步族下降为 `ascendc.pipe_barrier`。

转移分派位于 `synchronizeTransition`（`AFIRPipelineSync.cpp:407-427`），当前仅覆盖上述三种转移，其余组合一律报错（`:425`）：

```
error: unsupported pipeline synchronization from MTE2 to MTE3
```

legacy 路径支持的多个同步场景因此无法通过 MLIR 路径编译，或缺少竞争保护。

**缺口成因**：上述限制源于 [2026-08-05 pass 管线设计](2026-08-05-afir-to-ascendc-pass-pipeline-design.md)的范围裁剪，而非技术不可实现：

- 当时前提是队列等资源事实由 legacy Schedule 决定，管线仅负责物化；队列方向定义为按 buffer 所属算子的 `AFIR_TransferIn` / `AFIR_TransferOut` trait 单值推导，仅覆盖当时迁移所需的 load→计算、计算→store 路径；
- 范围外场景按该设计的 "deterministic unsupported" 原则显式报错，并以负向 lit 用例固化（`afir_pipeline_sync_unsupported.mlir`、`afir_memory_allocate_queue_direction_conflict.mlir`，与功能代码同提交 `49e46f35` 引入）；
- 2026-09-09 DSL 架构设计已明确记录"自动同步的生产者只覆盖 V→V"，完整同步能力列为后续工作；
- ascendc 方言层的 TQueBind 操作集与打印按 legacy API 全量存在，双端形态仅缺失 AFIR 资源模型支持。

本设计是该范围裁剪的自然扩展：将 MTE2→MTE3、MTE3→MTE2 等转移纳入支持范围，对应报错与负向用例翻转为正向产物，与既有设计原则一致。

### 产品环境介绍

```
PyTorch 算子图
  → AFIR
  → 生产管线：AFIRLoopGenerate → AFIRMemoryAllocate → AFIRPipelineSync → ConvertAFIRToAscendC
  → ascendc 方言 → CodeEmitter 打印 AscendC kernel.cpp
  → bisheng/clang 编译 → 上板执行
```

本设计修改 `AFIRPipelineSync` 及其上下游配合点（资源模型的双端队列支持、AFIR 同步操作集、下降规则），不新增 pass，不调整管线时序。legacy codegen（`autofuse/codegen/`）零修改，仅作为产物对拍基准。

## 缺失场景清单

**自动同步分支缺失（编译报错或无保护）**

| 编号 | 场景 | 引擎方向 | 现状 |
|------|------|----------|------|
| S1 | load 直连 store（数据不经计算直接搬出） | MTE2→MTE3 | 编译报错；`test/lit/demo/afir_pipeline_sync_unsupported.mlir` 已将该报错固化为预期行为 |
| S2 | 同一 load 结果同时供矢量计算与 store 消费 | MTE2→MTE3（旁路） | 矢量侧可走既有队列；store 侧编译报错 |
| S3 | store 写 workspace 后 load 回读 | MTE3→MTE2 | 编译报错；且依赖收集跳过 GM 目的地（`AFIRPipelineSync.cpp:194-196`），该依赖未被追踪 |
| S4 | 末读的 UB buffer 被复用写入（WAR，store 或矢量算子末读） | MTE3→MTE2 / MTE3→Vector / Vector→MTE2 | 队列路径已有释放屏障（`materializeQueueFree`，`:292-316`）；非队列路径无保护 |

**机制缺失（AFIR 层无法表达）**

| 编号 | 机制 | legacy 形式 | 现状 |
|------|------|-------------|------|
| M1 | TQueBind 双端队列 | `TQueBind<TPosition::VECIN, TPosition::VECOUT, 1>`（产物基准：`test_codegen_kernel.cpp:4420`） | ascendc 方言操作与打印已存在；资源模型限定队列资源仅允许单一传输方向（`CanonicalResources.cpp:82-84`），队列类型仅携带单一 position（`:386`），双端形态无法生成 |
| M2 | 全引擎屏障 | `AscendC::PipeBarrier<PIPE_ALL>()` | `afir.pipe_barrier` 仅接受单一 pipe 且不跨 pipe（`Synchronization.td:78-95`）；ascendc 层 `PIPE_ALL` 已存在（`Core/Attributes.td:441`） |
| M3 | 跨核栅栏 | `AscendC::SyncAll()`（缺省 `isAIVOnly = true`，非 cube） | AFIR 无对应操作；ascendc 层 `SyncAllHardOp` / `SyncAllSoftOp` 已存在（`Basic/OpBlockSync.td:80/90`）。legacy 的 `SyncAll<false>` 是 AIC+AIV 混模协议（`GenCubeCommonFunc*`，或按 `mix_mode` 分支），MLIR codegen 只覆盖非 cube（纯 AIV）图，此处缺省 true 即唯一正确取值 |

M1–M3 的下降落点在 ascendc 方言均已存在，缺失仅在 AFIR 层。

## 软件功能

| 功能 | 内容 | 覆盖 |
|------|------|------|
| F1 | S1 编译通过，队列物化为双端 TQueBind，产物与 legacy 一致 | S1、M1 |
| F2 | S2 编译通过：矢量侧走既有队列，store 侧插入显式同步对 | S2 |
| F3 | S3 编译通过：回读 load 前插入 MTE3→MTE2 显式同步对 | S3 |
| F4 | S4 中 buffer 复用写之前插入释放方向显式同步对（任意最后读者） | S4 |
| F5 | AFIR 新增全引擎屏障与跨核栅栏操作及下降规则 | M2、M3 |

## 设计约束

- **产物对齐 legacy**：S1 产物与 legacy codegen 逐字符一致，且不得额外生成 SetFlag/WaitFlag（队列已含同步，叠加即为双重同步）；S2–S4 产物为 legacy `TQueSync`（SetFlag/WaitFlag）的等价形式。显式同步的下降机制已实现（`Implementations/Synchronization.cpp:144/178`），复用不重建；
- **ADR-12**：存储与同步解耦，不引入队列抽象；S1 的队列物化是下降期的实现选择，不改变该架构；
- **ADR-19**：token 携带 from/to pipe 对且两端不同，方向错误由 verifier 拒绝；
- **Pipe 枚举单引擎纪律**（`AFIREnums.td`）：M2 不为 `afir.Pipe` 增加"全部"取值，新增独立操作；
- **手写跳过语义**：函数内已存在任何同步操作时 `AFIRPipelineSync` 整体跳过（`LoweringStage.cpp:26`），新操作必须加入该判定集合；
- **图改写确定性**（编码红线 5）：同步插入位置由程序序决定，不依赖哈希表遍历顺序；
- **无对外 ABI/API 变化**。

## 假设和依赖关系

已验证（2026-09-29 随 `84a9a00a` 重新核对）：

- 显式同步下降 pattern 存在且被 broadcast 特性复用（`SetFlag`/`WaitFlag` 事件号走 `TPipe::FetchEventID`）；
- `PIPE_ALL`、`SyncAllHard/Soft`、TQueBind 操作集及 CodeEmitter 打印在 ascendc 层均存在；
- S1 产物对拍基准存在（`test_codegen_kernel.cpp:4420`）；
- 两个负向 lit 用例（`afir_pipeline_sync_unsupported.mlir`、`afir_memory_allocate_queue_direction_conflict.mlir`）已定位，落地时翻转为正向。

待实现阶段确认：

- 双端队列的实现形态：扩展 `QueueType` 携带位置对，或复用带 `srcUserPos`/`dstUserPos` 的 `TQueBind{Enque,Deque}TensorPosOp` 变体，实现评审定稿；
- 双端 TQueBind 运行时行为与 legacy 一致性，由 legacy 既有用例与 E2E 数值验证。

# 需求分析与设计

## 整体介绍

扩展后的转移分派策略：

```
依赖转移（生产者引擎 → 消费者引擎，同一 buffer）
├─ MTE2→Vector、Vector→MTE3      ：队列同步（现状，不变）
├─ Vector→Vector                 ：单管屏障（现状，不变）
├─ MTE2→MTE3，buffer 仅有 store 消费且有队列：双端队列（S1）
├─ MTE2→MTE3，buffer 另有矢量消费          ：显式同步对 transfer_in→transfer_out（S2）
├─ MTE2→MTE3，buffer 无队列（TBuf）        ：确定性报错（范围外，见下）
├─ MTE3→MTE2（workspace 回读）             ：显式同步对 transfer_out→transfer_in（S3）
└─ 末读者→复用写者（buffer 复用 WAR）  ：显式同步对 末读者 pipe → 写者 pipe（S4）
```

S1 采用队列的理由：依赖形状与队列语义一致（单生产单消费、先进先出、两端引擎），硬件队列隐式同步且产物与 legacy 一致。S2–S4 采用显式同步的理由：S2 的同一 buffer 存在多方向消费，S3 的载体为 GM（无队列资源），S4 保护的是释放时序，均不符合队列形状。

无队列的 MTE2→MTE3 报错而非回落显式同步对：PipeBarrier 只在同一条流水内排序，MTE2 与 MTE3 之间没有 barrier 可用；且 legacy 的资源分配对搬运缓冲一律给队列，该形状在 legacy 中不出现，属「范围外场景按 deterministic unsupported 原则显式报错」。

设计原则：

1. 仅新增同步操作与队列物化，不移动、不改写数据流，保证改写等价；
2. 现有三种转移产物保持不变，由既有测试回归锁定；
3. S2–S4 复用既有显式同步下降机制，无新增下降框架。

## 功能需求

### 功能需求 1：S1 load 直连 store（含 M1 双端队列）

**场景**：load 搬入的数据未经矢量计算直接由 store 搬出（GM → load → UB → store → GM）。

**现状**：`synchronizeTransition` 无 MTE2→MTE3 分支，编译报错（`AFIRPipelineSync.cpp:425`）；同时资源模型限定一条队列资源仅允许单一传输方向（`CanonicalResources.cpp:82-84`），双端队列无法生成。

**目标**：编译通过；队列物化为双端 TQueBind，产物与 legacy 一致，不生成 SetFlag/WaitFlag：

```cpp
TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> q0;
q0.EnQue(q0_buf);
q0_buf = q0.DeQue<uint8_t>();
```

**设计**：

1. 资源模型（`CanonicalResources.cpp`）：
   - 队列方向由单值 `queueDirection` 改为方向集合；同时包含 TransferIn 与 TransferOut 成员的队列判定为双端队列，合法，删除对应报错；
   - `getPosition`（`:378-381`）与队列类型构造（`QueueType::get`，`:386`）扩展支持位置对 `(VECIN, VECOUT)`，单端行为不变；
   - 校验：双端队列成员必须同时具备两类方向，缺失即报 `kInvalidBindingDiagnostic`，禁止生成不完整队列；
   - 适配近期字段演进（`size_from_resource` 尺寸二选一、`canonicalResourceFactsAgree` 统一字段比较、`slot_count` 聚合取 max），不绕开统一校验。
2. 同步 pass（`AFIRPipelineSync.cpp`）：`synchronizeTransition` 新增 MTE2→MTE3 分支，复用 `insertQueueSync`（其入队/出队逻辑与方向无关，`:260-266`）；producer 为 MTE2 时的前置 `PIPE_MTE2` 屏障（`:336-338`）与释放屏障（`:347-352`，MTE3 时 `releasePipe = PIPE_MTE3`）既有逻辑直接适用。
3. 下降与打印：`TQueBind` 操作集与 CodeEmitter 打印已存在，仅做串联；物化复用队列成员共用一次 `AllocTensor` 的既有机制（`e0f8e849`）。

**验证**：`afir_pipeline_sync_unsupported.mlir` 翻转为正向（断言 TQueBind 产物）；codegen 对拍逐字符一致；E2E 数值正确。

### 功能需求 2：S2 混合消费（store 侧显式同步）

**场景**：同一 load 结果的消费者同时包含矢量算子与 store。

**现状**：矢量侧走既有 MTE2→Vector 队列路径；store 侧触发 MTE2→MTE3 报错。

**目标**：矢量侧维持队列同步；store 侧由 pass 在 load 之后插入 `afir.event_record`（`from = transfer_in, to = transfer_out`）、store 之前插入对应 `afir.event_wait`，下降为 SetFlag/WaitFlag，与 legacy `TQueSync` 等价。

**设计**：

- MTE2→MTE3 分支内按 buffer 消费者形状分派：仅 store 消费走双端队列（S1），另有矢量消费走显式同步对；判据在分派入口统一判定一次；
- 操作数分区遵循 `OperandPartition` 三段式（inputs / destinations / tmp buffers，`AFIRPipelineSync.cpp:379-389`）：tmp buffer 段由被调方独占读写、**不构成生产者→消费者的依赖边**，同步判定仅针对 inputs 段。注意这只限定同步视角；竞争检查视角下 tmp 按写记账——两个共享同一块 tmp buffer 的 Op 之间是真实的复用竞争（`RaceCheck.cpp` 的 `collectAccesses`），两个视角的前提不同，不矛盾。

**验证**：lit 断言队列同步与显式同步对并存、方向正确；E2E 数值正确。

### 功能需求 3：S3 workspace 回读

**场景**：store 将中间结果写入 GM workspace，后续 load 从同一 workspace 读回，依赖方向为 MTE3→MTE2。

**现状**：无 MTE3→MTE2 分支，编译报错；依赖收集跳过 GM 目的地（`:194-196`），该依赖未被追踪。

**目标**：回读 load 之前插入显式同步对（`from = transfer_out, to = transfer_in`）。

**设计**：依赖收集新增 GM 读写配对——store 的 GM 写集与 load 的 GM 读集按同一 memref 配对（SSA 同值，或经 `BufferAlias` 判定别名），命中即在 load 前插入同步对。复用现有别名分析，不新增。

**验证**：lit 覆盖同值与别名两类 workspace 夹具；E2E 数值正确。

### 功能需求 4：S4 buffer 复用 WAR 保护（任意最后读者）

**场景**：某 UB buffer 的最后读者之后，该 buffer 被复用写入（下一迭代 load 或其他算子）。若复用写先于末次读完成，构成 WAR 竞争，无编译诊断，仅表现为输出数据错误。store 末读为典型场景；矢量算子末读后 buffer 由 load 复用（Vector→MTE2 方向）为同型场景，一并覆盖。

**现状**：队列路径在 buffer 释放时已插入释放屏障（`materializeQueueFree`，`:292-316`，已覆盖 V 与 MTE3 两类消费者）；非队列路径与显式同步形态无系统保护。

**目标**：复用写之前插入释放方向显式同步对（`from = 末读者所在 pipe, to = 复用写者所在 pipe`）。

**设计**：最后消费者追踪（当前仅限队列路径）扩展为不限路径、不限读者类型（store / 矢量算子）；命中"存在程序序在后的复用写者"即按末读者 pipe 插入对应方向的同步对，覆盖 MTE3→MTE2、MTE3→Vector、Vector→MTE2 等全部释放边。跨迭代多缓冲轮转的 WAR 检测不在本期范围。

**验证**：lit 含循环内 buffer 复用夹具（store 末读与矢量算子末读各一）；E2E 循环场景数值正确。

### 功能需求 5：M2/M3 全引擎屏障与跨核栅栏

**场景**：两类 legacy 同步原语在 AFIR 中无对应表达。全引擎屏障 `PipeBarrier<PIPE_ALL>`：`afir.pipe_barrier` 仅接受单一 pipe 且不跨 pipe。跨核栅栏 `SyncAll`：AFIR 无对应操作。二者在 ascendc 方言的下降落点均已存在。

**目标**：AFIR 新增两个同步操作及下降规则，支持手写 IR 表达与后续 pass 使用；不进入 `AFIRPipelineSync` 自动插入（全引擎屏障为兜底原语，自动插入与最小同步原则冲突；跨核同步属调度层职责，超出单核函数体管辖）。

**设计**：

1. IR（`Synchronization.td`）：新增 `afir.barrier_all`（暂名，无操作数）与 `afir.sync_all`（暂名）；verifier 约束仅允许出现在 `KernelBody` 函数；命名实现评审定稿；
2. 下降（`Implementations/Synchronization.cpp`）：`afir.barrier_all` → `ascendc.pipe_barrier`（`PIPE_ALL`）；`afir.sync_all` → `ascendc.sync_all_hard`。本期仅实现 hard 变体（无参数）；soft 变体带 `gmWorkspace` / `ubWorkspace` / `usedCores`，为 superkernel feed-sync 语义，登记为后续扩展；
3. 一致性：`hasPipelineSynchronization`（`LoweringStage.cpp:26`）判定集合加入新操作；
4. 检查器：`RaceCheck` 认定两个新操作为全序点，终止穿越该点的冒险查询。

**验证**：lit 断言下降产物为 `ascendc.pipe_barrier pipe_all` 与 `ascendc.sync_all_hard {isAIVOnly}`；UT（`SynchronizationTranslationTest`）钉翻译文本 `AscendC::SyncAll()`；上板（910B3）验证该协议在真实 kernel 中可正常完成（`<false>` 在纯 AIV 发射下死锁，见实现记录第 16/17 条）。

## 覆盖评估（Inductor 非 CV 场景）

本设计落地后，单核函数体内 {MTE2, Vector, MTE3} 三引擎的全部转移均有同步方案：

| 转移 | 状态 | Inductor 非 CV 对应场景 |
|------|------|------------------------|
| MTE2→Vector | 现有 | 常规输入搬运 |
| Vector→Vector | 现有 | 算子链 |
| Vector→MTE3 | 现有 | 常规输出搬运 |
| MTE2→MTE3 | S1/S2 新增 | pass-through 输出（多输出子图中直通输出）、`to_copy`/clone 类节点 |
| MTE3→MTE2 | S3 新增 | UB 不足时的 GM 暂存、多 stage 中间结果回读 |
| 释放方向（末读者→复用写者） | S4 新增 | tiling 循环 buffer 轮转复用（Inductor 融合 kernel 均为循环结构） |

不在本设计覆盖内、但与 Inductor 非 CV 相关的边界：

1. **MM/BMM epilogue 融合**：非 CV 但走 Cube 引擎，卡点在 Cube 算子迁移未完成（Gen* Pass 显式拒绝），不在同步侧。同步基础设施已就绪：token 词汇与下降已支持 Cube 管道对（`mte2_mte1` / `mte1_m` / `m_fix` / `fix_mte3`，见 `afir-synchronization-lowering.mlir`），Cube 迁移落地时按 S1–S4 模式扩展分支即可；
2. **跨核 reduction combine**：M3 仅补词汇，不自动插入（跨核同步属调度层职责）；逐 tile 独立的标准融合不涉及；
3. **跨子图依赖调度**：多子图聚合后的运行时调度，非本 pass 职责。

## 非功能需求

### 可维护性

新增分支收敛于 `synchronizeTransition` 分派结构；S1/S2 形状判据集中一处判定；资源模型注释说明单端/双端两类队列形态的判定规则；新操作定义延续现有注释风格。

### 可测试性

每场景一个 lit 夹具（正向断言产物、负向断言诊断）；S1 与 legacy codegen 对拍；E2E 上板数值验证（含循环复用 buffer 用例）。

### 可移植性

AFIR `Pipe` 为抽象引擎（ADR-20），显式同步下降按后端实现注册表映射，不影响参照后端；双端队列物化封装于 conversion 层，不进入 AFIR。

### 可靠性

双端队列成员方向不齐时编译期诊断，禁止生成不完整队列；无法识别的形状保持显式报错，不静默降级；pass 失败走既有物化回滚机制，异常分支不泄漏资源。

### 特性交叉影响

按 [cross_feature_check.md](../../../../docs/guidelines/cross_feature_check.md) 逐项：

| 场景 | 适用性 | 分析说明 |
|------|--------|----------|
| SuperKernel Python 接口 | 不适用 | 不修改 `super_kernel/` 任何代码 |
| SuperKernel C++/AOT 接口 | 不适用 | SK 使用独立 JIT 文本生成路径；无 `libascendsk.so` 行为变化 |
| Autofuse 图优化 | 适用（弱） | 不新增 optimize pass、不改 ASCIR 图；仅新增同步边，数据边不变 |
| Autofuse Codegen/Backend | 适用 | MLIR 路径产物变化：四个场景从编译报错变为正确产物；legacy codegen 零修改，仅作对拍基准 |
| AscendC API / Runtime 交互 | 适用 | 产物仅使用既有开放 AscendC API；无新增 `rt*`/`aclrt*` 调用；`TQueBindFreeTensorOp` 释放路径覆盖异常分支 |
| Python/C++ 混合绑定 | 不适用 | 无绑定变化 |
| 构建与打包 | 适用（弱） | 仅向 CMakeLists 增列源文件，无构建逻辑与产物内容变化 |
| 测试与覆盖率 | 适用 | 新增 lit/UT/E2E，按 TDD 先写失败用例 |
| 性能与日志 | 适用 | 编译期遍历量级不变；不新增运行时日志 |
| 兼容性 | 适用 | 现有产物不变；"报错→通过"为本需求目标行为；无 ABI/API 变化 |

## 性能

### 编译时长

`AFIRPipelineSync` 为 O(ops × operands) 单遍扫描，新增分支仅改分派不增遍历；S3 的 GM 配对复用现有 `BufferAlias` 分析结果。编译时长量级不变。

### 执行性能

S1 产物与 legacy TQueBind 一致，零额外同步指令；S2–S4 产物为 legacy SetFlag/WaitFlag 等价形式。相对 legacy 无新增运行时开销。

### 内存和产物大小

双端队列复用同一 buffer 组成员（既有聚合机制），不新增 UB 占用；新增两个操作定义与下降规则对编译产物大小影响可忽略。

## 接口设计

### 新增/修改接口描述

均为编译器内部接口，无对外 C++/Python API、无 ABI：

| 接口 | 类型 | 说明 |
|------|------|------|
| `afir.barrier_all`（暂名） | 新 IR 操作 | 无操作数；下降为 `ascendc.pipe_barrier`（`PIPE_ALL`） |
| `afir.sync_all`（暂名） | 新 IR 操作 | 本期实现 hard 变体；下降为 `ascendc.sync_all_hard` |
| 队列方向模型 | 内部数据结构 | 方向单值改为方向集合；队列类型支持位置对 |
| `synchronizeTransition` 分派 | 内部函数 | 新增 MTE2→MTE3 / MTE3→MTE2 / WAR 分支及 S1/S2 形状判据 |

### 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|--------|----------|----------|------|
| 接口说明 | 是否需要接口评审 | 是 | 新操作命名与语义随设计评审定稿 |
| 接口说明 | 是否需要补充文档 | 是 | 操作定义即文档（`-gen-dialect-doc`） |
| 接口兼容 | 行为是否兼容 | 是 | 既有产物不变，仅新增 |
| 接口兼容 | ABI/API 是否兼容 | 不涉及 | 无对外 ABI/API |
| 接口约束 | 约束不满足时是否清晰报错 | 是 | verifier 与编译诊断，沿用既有前缀 |
| 接口测试 | 是否需要独立接口用例 | 是 | lit 逐操作断言下降产物 |

## 软件设计

### 关键数据结构

- **队列方向集合**：`CanonicalResourceInfo` 的队列方向由 `TransferDirection` 单值改为 `{In, Out}` 集合；单端/双端形态判定与位置推导集中于 `getPosition`；
- **消费者形状**：按 buffer 聚合的消费者引擎集合，作为 S1/S2 分派输入（复用 `lastQueueConsumers` 聚合方式并扩展为不限队列路径）；
- **token**：`!afir.token<from, to>`，既有类型，无修改。

### 关键技术/算法

- **依赖配对**：既有 `operationOrder` / `destinationProducers` / `lastQueueConsumers` 覆盖 S1/S2/S4 的配对输入；S3 新增 GM 读写配对（SSA 同值优先，别名走 `BufferAlias`）；
- **确定性**：插入位置由程序序（`operationOrder`）决定，`DenseMap` 仅用于查找；
- **事件号分配**：显式同步下降时的事件号分配走既有 DAG 着色 / `FetchEventID` 机制，pass 层不感知。

### 流程设计

```
AFIRMemoryAllocate：资源规划 → 队列方向集合（新增双端判定）→ 物化队列/Buffer
AFIRPipelineSync：  依赖收集（新增 GM 配对、不限路径的最后消费者）
                    → 逐转移分派（新增三分支 + S1/S2 形状判据）
ConvertAFIRToAscendC：队列 → TQueBind 物化（既有）；
                      显式同步 → SetFlag/WaitFlag（既有）；
                      新操作 → PIPE_ALL / SyncAll（新增两条下降规则）
```

异常流程：无法识别的形状报错终止（现状语义）；资源物化失败走既有 `createdOps` 回滚。

### 对子模块的修改

（实现后的实际修改面，与原稿的差异见文末实现记录）

| 文件 / 目录 | 修改点 |
|-------------|--------|
| `conversion/AFIRToAscendC/AFIRPipelineSync.cpp` | `synchronizeTransition` 新增 MTE2→MTE3 分支（S1/S2 形状判据一次判定）；S3 GM 读写配对与 S4 最后读者追踪（视图值粒度、同迭代程序序）；S2 释放屏障按读者 pipe 聚合；双端队列延迟替换 `ascendc.queue` → `ascendc.que_bind`；遵循 `OperandPartition` 三段式 |
| `conversion/AFIRToAscendC/MemoryAllocate/` | 零修改（双端物化移入同步层，见实现记录第 1 条） |
| `dialect/AFIR/IR/Ops/Synchronization.td` / `.cpp` | 新增 `afir.barrier_all` / `afir.sync_all` 及 KernelBody 驻地 verifier |
| `conversion/AFIRToAscendC/Implementations/Synchronization.cpp` | 新操作的下降 pattern（`PIPE_ALL` / `sync_all_hard`）与映射注册 |
| `conversion/AFIRToAscendC/LoweringStage.cpp` | `hasPipelineSynchronization` 判定集合加入新操作 |
| `dialect/AFIR/Transforms/LoopGenerate.cpp` | `isSynchronizationOperation` 清单加入新操作（原稿遗漏） |
| `dialect/AFIR/Analysis/RaceCheck.{h,cpp}` | 认定新操作为全序点（程序序位置判定） |
| `api/ascendc/`（ascendc 方言） | 上游 `OpBlockSync.td` 的 `SyncAllHardOp` 置 `genEmitter = 0`（登记 README §4 例外清单）；手写打印与注册在子树 `Autofuse/BlockSync.{h,cpp}` + `Autofuse/Ops.h` |
| `test/`（lit / ut / e2e） | 新增 S1/S2/S3×2/S4/FR5 正负 lit、RaceCheck 正负对照、gtest 驻地用例、CPU 与 NPU E2E `load_store_direct` |

### 错误处理

#### 系统错误

pass 失败走既有 `signalPassFailure`；资源物化失败走既有回滚清单；不新增系统级错误路径。

#### 接口错误

- IR verifier：新操作的函数角色约束与变体合法性；token 单次 wait（既有）；
- 编译诊断：双端队列成员方向不齐、S3 别名无法判定等场景输出带 `kInvalidBindingDiagnostic` / `kUnsupportedOperationDiagnostic` 前缀的定位信息，风格与现有一致。

## 安全检查

按 [编码红线.md](../../../../docs/guidelines/编码红线.md) 逐条核对：无敏感信息硬编码；资源尺寸与偏移沿用既有溢出、非负检查（`CanonicalResources.cpp` 的 `cf.assert` 系列），新路径复用不绕开；`TQueBindFreeTensorOp` 释放覆盖异常分支（既有机制）；无对外 ABI/API 变化；仅新增同步边、数据边不变，插入位置由程序序决定（图改写等价性与确定性）；不涉及运行时调用；不新增日志。结论：无红线触碰。

## 兼容性检查

- 现有三种转移产物逐字符不变，由既有 lit 用例回归锁定（`afir-pipeline-sync-produces-afir.mlir` 等）；
- `hasPipelineSynchronization` 手写跳过语义不变（新操作入集为规则一致性的必然）；
- 四个场景从编译报错变为编译通过为本需求目标行为；`afir_pipeline_sync_unsupported.mlir`
  翻转为正向（更名 `afir_pipeline_sync_load_store_direct.mlir`，夹具即 S1 的测试输入）；
  `afir_memory_allocate_queue_direction_conflict.mlir` 维持负向——其成员是并排的两块存储、
  store 读未写入偏移，不是合法双端形状（见文末实现记录第 3 条）；
- 不影响安装目录、脚本参数、Python 包与历史产物。

## 测试设计

### 测试边界

入口：手写 AFIR 夹具与 E2E 前端图。出口：PipelineSync 与 MemoryAllocate 后的 IR 断言（lit）、生成 cpp 与 legacy 对拍（UT）、上板数值（E2E）。无运行时桩。

### 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|----------|------------|----------|----------|
| 功能 | S1 产出双端 TQueBind、无 SetFlag/WaitFlag | lit（翻转 `afir_pipeline_sync_unsupported.mlir` 夹具） | UT(lit) |
| 功能 | S1 生成 cpp 与 legacy 对拍 | codegen UT，期望文本 `TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> q0;` | UT |
| 功能 | S2 队列同步与显式同步对并存、pipe 对正确 | lit | UT(lit) |
| 功能 | S3 回读 load 前显式同步对（`transfer_out→transfer_in`） | lit（同值与别名两类 workspace 夹具） | UT(lit) |
| 功能 | S4 复用写前释放方向显式同步对（store 末读与矢量算子末读两类夹具） | lit（含循环内复用夹具） | UT(lit) |
| 功能 | 新操作下降为 `PIPE_ALL` / `sync_all_hard` | lit | UT(lit) |
| 异常 | 双端队列成员方向不齐 → 诊断 | lit 负向 | UT(lit) |
| 异常 | 新操作出现在非 KernelBody → verifier 拒绝 | UT（`AFIRSynchronizationTest.cpp` 扩展） | UT |
| 异常 | 既有负向用例回归（token 同 pipe、跨块同步） | 既有用例不动 | UT(lit) |
| 性能 | 新场景编译时长不劣化 | 手工对比（可选） | 手工验证 |
| 兼容性 | 现有三转移产物不变；两个负向 lit 翻转 | 既有 lit 全量回归 + 改写两用例 | UT(lit) |
| 特性交叉 | S1/S4 E2E 数值（含 NPU） | E2E 脚本 | ST/E2E |

### 测试命令

```bash
# MLIR UT/ST（lit + gtest）
bash scripts/test/run_autofuse_test.sh -u -s -m mlir -j 10

# codegen 对拍 UT
sh build.sh -u --module=autofuse_framework -j 8

# E2E
sh build.sh -s --module=autofuse_e2e -j 8
python3 autofuse/mlir/test/e2e/npu/load_add_store.py        # 既有回归
python3 autofuse/mlir/test/e2e/npu/load_store_direct.py     # 新增：S1 纯搬运
```

预期：全绿；`load_store_direct.py` 上板数值与 eager 对齐。

## 实现记录（2026-10-09）

按本文档实现完成。四处与设计正文偏离的实现决策，均为实现期发现的正确性问题所迫，逐条记录：

1. **双端队列的物化位置（M1）**：未采用"资源模型方向集合 + `getPosition` 位置对"方案，改为
   `AFIRPipelineSync` 在 S1 分派命中后延迟把 `ascendc.queue` 替换为 `ascendc.que_bind`
   （`QueBindType<VECIN, VECOUT, 1>`，深度模板与 legacy 对齐取 1）。原因：S1 与 S2 在方向
   聚合层面不可区分（都是 load 输出绑到 store 操作数上，方向集合同为 {In, Out}），双端判定
   必须依赖"store 是否为唯一消费者"这一形状判据，而该判据属同步层职责；提前在资源层定型会把
   S2 的队列错误物化成双端、破坏矢量侧的 VECIN 流程。延迟替换保住了各映射按旧队列值记的键。
   `CanonicalResources.cpp` 因此零修改，S1/S2 形状判据集中在 `synchronizeTransition` 的
   MTE2→MTE3 分支一次判定，与 FR2 的要求一致。
2. **S2 形状判据必须在改写前定案**：队列同步会替换消费者操作数，扫描进行到直连 store 时矢量
   消费者已从 user 列表消失。收集阶段（`collectProducersWithNonStoreConsumers`）对原始 IR
   预计算并缓存结论。
3. **`afir_memory_allocate_queue_direction_conflict.mlir` 维持负向**（与兼容性检查一节的
   预期不同）：该夹具的两个成员是同一队列资源上**并排**的两块存储（In 侧 `%tmp0` 与 Out 侧
   `%tmp1` 各占一个偏移），store 读的是从未被写入的偏移——这不是 legacy 的双端直连形状
   （legacy 的 S1 是 store 输入直连 load 输出、同一存储，见 `test_codegen_kernel.cpp:5788`
   的 `store_op.x = load_op.y`）。翻转它会生成读未初始化存储的错误产物，保留确定性报错。
   S1 能力由 SSA 直连形态（翻转后的 `afir_pipeline_sync_load_store_direct.mlir`）覆盖。
4. **同步族 op 清单不止 `LoweringStage.cpp` 一处**：`LoopGenerate.cpp` 的
   `isSynchronizationOperation`（无 `afir.sched` 的同步原语不进循环树）同样需要加入
   `barrier_all` / `sync_all`，设计文档原清单遗漏，已补。
5. **ascendc 层非零修改**：`SyncAllHardOp` 的 cpp 打印原先依赖 tablegen 自动生成，而
   `SyncAll` 是模板函数（`SyncAll<isAIVOnly>()`），UnitAttr 的 bool 模板实参超出
   `paramTypeLists` 表达力。已在 `OpBlockSync.td` 置 `genEmitter = 0` 并补手写打印
   （手写打印与注册位于 `Autofuse/` 子树，见实现记录第 10 条）。
6. **释放屏障按读者 pipe 聚合**：S2 使同一队列可能出现矢量与搬出两类读者，队列释放屏障从
   单条（末位消费者所在 pipe）改为按读者 pipe 各插一条（`queueConsumerPipes` 位聚合）；
   单一引擎读者时产物与既有逐字相同。

实现记录（2026-10-10 检视修复，PR #80 review 五项 findings）：

7. **事件槽按 legacy 纪律分配（高）**：初版下降用 `FetchEventID` 逐条取号不占位，
   交叠的同型同步对会共享同一事件号（硬件上要么挂死要么丢失排序），且 pass 侧注释
   「各自占一个事件槽」与事实矛盾。改为 record 侧 `AllocEventID` 占位、wait 侧
   `WaitFlag` 后 `ReleaseEventID` 归还，与 legacy `SyncMte3ToMte2`/`SyncMte2ToMte3`
   （`codegen_kernel.cpp`）的 Alloc/Release 纪律一致：交叠对各自持有不同槽，顺序化
   完成后归还复用。
8. **显式同步对增加同块校验（中高）**：S3/S4 按 buffer 归并建对，没有 SSA 数据边
   保证 def 支配 use；跨循环嵌套（store 在 axis 0 循环、load 在嵌套内循环）会生成
   token use 不被 def 支配的非法 IR，且 Alloc/Release 执行次数不再配平。`insertExplicitPairSync`
   入口统一拒绝跨块配对（与 `collectDestinationDependency` 既有判据同风格同文案），
   新增负向夹具 `afir_pipeline_sync_workspace_readback_across_nests.mlir`。
9. **S4 收敛为每 pipe 末读者 + 成员视图归并（低）**：初版对全部前序读者逐个建边
   （方向安全的超集），与本设计「按末读者 pipe 插入」不符且放大事件槽冲突面。改为
   按 (存储, 读者 pipe) 记末读者、写者只与各 pipe 末读者建边；存储归并键停在
   **成员字节视图（SubIndex）粒度**——物化规范链 `ReinterpretCast(SubIndex(
   TBufGetTensor(tbuf), offset))` 追过 ReinterpretCast 即停，同成员命中、同 TBuf 不同
   offset 的不相交成员（含同 buffer_id 不同 reuseId 的复用组）不误并（本就无需 WAR
   边）。新增多读者夹具断言单边。
10. **api/ascendc 定制收回 `Autofuse/` 子树（中）**：初版把 `SyncAllHardOp` 的手写
    打印声明/实现写进上游 `Basic/BlockSync.{h,cpp}`、注册进上游 `PrintableOpTypes`，
    违反 README §4「定制内容一律进子树」。三处外移：声明 `Autofuse/BlockSync.h`、
    实现 `Autofuse/BlockSync.cpp`（CMake GLOB_RECURSE 免改）、注册进
    `Autofuse/Ops.h` 的 `autofuse::PrintableOpTypes`；上游仅保留 `OpBlockSync.td` 的
    `genEmitter = 0`（TableGen 无法 reopen，与 `VecScalarL2Op::isSetMask` 同类），
    已登记进 README §4 例外清单。
11. **运行时验证补齐（中）**：初版仅 S1 有运行时用例。新增 CPU E2E
    `workspace_readback`（store→GM workspace→load 回读；CPU debug 下 MTE3_MTE2 的
    `WaitFlag` 非 no-op（`kernel_event.h` 仅 3510 架构豁免），回读时序被真实执行，
    断言产物含 MTE3_MTE2 的 Alloc/Wait/Release 三件套且数值 = 4x）；新增 NPU 脚本
    `mixed_consumption.py`（S2：`x + 1` 与 `x.clone()` 双输出）。局限如实记录：
    CPU debug 对 V 参与的事件 WaitFlag 是 no-op，S2/S4 的跨引擎时序与多对交叠形状
    只能上板验证；S3 的 GM WAR 方向（load 读后 store 覆写）legacy 同样不覆盖
    （`WaitInputMte` 仅按 SSA 关系建边），维持本期范围。

实现记录（2026-10-10 第二轮检视修复）：

12. **S3/S2 归并收敛并发事件槽（中）**：Alloc/Release 修复解决了槽位**复用**，但逐
    写者/逐 store 建对仍会使交叠对数随成员数增长：同一 workspace 根 N 个 store 加一次
    回读（多 stage 各写子区是典型形态）产生 N 个并发 MTE3_MTE2 槽，S2 一个 load 喂
    M 个直连 store 同理——超过 QUE_MAX_EVENT（2201 等 8、其余架构 4，
    `kernel_macros.h`）时 `AllocEventID` 的 sff0 返回越界事件号（debug 断言、release
    静默坏 id）。S3 改为每 load 只与**该根上程序序末位写者**建对（GM 写者全在 MTE3，
    同 pipe 程序序使末写者的 record 覆盖全部前序 store）；S2 改为每 load 只与**首个**
    直连 store 建对（wait 挡住首 store 即按程序序覆盖后续 store）。两者的并发槽位需求
    均收敛为 1。与 S4 的「每 pipe 末读者」构成同一公理的三个方向。
13. **悬空 record 拒绝（verifier，弱化版）**：与既有「token 只能等一次」对称的另一
    半——没人消费的 record 在下降层 `AllocEventID` 占槽后无人 `ReleaseEventID`，槽
    泄漏的失败形态是后续同型 record 拿到越界事件号。不做「恰好等一次」的强校验：
    跨迭代携带（token 经 scf.for 的 iter_args / yield 传给下一迭代的 wait，
    `AFIRRaceCheckTest.AcceptsLoopCarriedWriteAfterReadGuardedByACarriedEvent`）是
    合法习语，其末迭代的 record 按构造不被等、泄漏一个槽——强校验会拒掉它。落地
    为「token 至少被消费一次」：拦手写 IR 里悬空的 record。pass 生成的对总是成对。

实现记录（2026-10-12 合并 af-mlir 与测试冗余检视）：

14. **合并 af-mlir 的平行 S1 实现，无队列语义收敛为报错**：af-mlir 侧（5995fe88 /
    cf238413）与本文档平行实现了 MTE2→MTE3。合并取舍：分派保留本支
    `synchronizeTransferInToTransferOut`（涵盖 S1/S2，是上游内联版的超集）；
    `getPipeline` 采用上游的 AFIR_TransferIn/Out trait 版（op 名单会把新增的
    `afir.nddma` 误判成 Vector 算子）；**无队列（TBuf）的直连从「回落显式同步对」
    改为确定性报错**——本支初版的 `!origin.queue` 事件对回落超出本设计范围，按
    「deterministic unsupported」原则（简介一节）收敛，与上游
    `afir_pipeline_sync_unsupported.mlir`（TBuf 报错版）对齐；`Autofuse/Ops.h` 与
    e2e `CMakeLists.txt` 双方新增并存。
15. **测试冗余检视修复**：`multi_writer_readback` 夹具补首 store 的 `MULTI-NEXT`
    断言（store1 与 reloaded 之间无 record），与末 store→record 的 NEXT 一起钉住
    第 12 条的 S3 末写者归并，回归逐写者形态即失败；删除
    `AFIRSynchronizationTest` 中与 lit 重复的两条 verifier 极性 UT（负向
    `afir_synchronization_invalid.mlir` 已覆盖且钉住错误文本，正向
    `afir-synchronization-lowering.mlir` 已覆盖），保留 MemoryEffect/CSE 两条
    lit 做不了的。

实现记录（2026-10-15 上板验证修复）：

16. **M3 跨核栅栏改落 `SyncAll<true>`（高，910B3 实测死锁）**：初版把 legacy 的
    `SyncAll<false>` 原样搬作 M3 落点，漏了它的前置条件——`<false>` 是 AIC+AIV
    混模协议（AIV 侧 `wait_flag_dev(SYNC_AIC_AIV_FLAG)` 等一个**只有 AIC 会置位**
    的旗），legacy 仅在混模上下文（`GenCubeCommonFuncForAIV`）或按 `mix_mode`
    分支时使用。而 MLIR codegen 对 `IsCubeFusedScheduled` 直接回落 legacy，自身
    只产生纯 AIV kernel——`<false>` 在本路线**永远**死锁。判别与修复按
    `autofuse/mlir/AGENTS.md` §3.4/§3.5 链路做了产物级 A/B（同 kernel 同位置只换
    模板实参）：`SyncAll<false>()` 120s 零输出挂死（exit 124），`SyncAll<true>()`
    与 `SyncAll()`（缺省即 true）数值全对不挂。修复：下降显式置 `isAIVOnly`
    （`rewriter.getUnitAttr()`），不靠 `UnitAttr` 缺省（不写 = false = 封死侧）。
    lit 补 `{isAIVOnly}` 断言、UT 补 `AscendC::SyncAll()` 文本钉。此条与
    AGENTS §3.3 的「缺省极性」同源：跨语言镜像模板参数时，`UnitAttr` 缺省落在
    哪一侧必须单独核。

实现记录（2026-10-15 二次修正：收回 api/ascendc 打印定制）：

17. **收回 `SyncAllHardOp` 的手写打印包，回归 tablegen 自动 emitter（中）**：第 5/10
    条的 `genEmitter = 0` + `Autofuse/BlockSync.{h,cpp}` 手写打印包，动机是「把
    `isAIVOnly` 如实打成模板实参」。第 16 条修复后重新审视：非 cube 范围内该属性
    恒为 true，auto emitter 的输出 `AscendC::SyncAll()`（缺省 `isAIVOnly = true`）
    与手写版**同一个模板实例、行为逐字等价**（910B3 双向实测）；且 auto 版结构上
    打不出 `<false>`，反而封死了死锁取值出现的可能。手写保真打印唯一的增益是
    「将来混模能表达 `<false>`」——而混模目前回落 legacy，属未立项的后续设计。
    按「上游同步文件（`Basic/OpBlockSync.td` 来自 PyAsc 同步）能不改就不改」原则，
    全部收回：`OpBlockSync.td` / `Autofuse/Ops.h` / README §4 例外登记回退到基线，
    `Autofuse/BlockSync.{h,cpp}` 删除；下降处的 `isAIVOnly` 属性保留作 IR 层意图
    声明（与 `GenKernelEntry` 一致，也是将来混模选择协议时的旋钮）。将来引入混模
    时须同时做两件事：按发射模式取 `isAIVOnly`，以及恢复能表达模板实参的打印
    （届时按第 16 条的教训单独上板验证）。

验证记录（2026-10-12 合并 af-mlir 后全量复测，容器内 `check-lit-mlir` / `check-ut` /
`check-e2e`）：lit 199/199（+23 为上游新增用例，含其 TBuf 报错版 unsupported 与
正例 afir-transfer-load-store-mte2-mte3）；gtest 全部通过（`AFIRSynchronizationTest`
余 2 条）；CPU E2E 全绿。

验证记录（2026-10-10 检视修复后复测）：lit 176/176（新增 across_nests 负向夹具）；gtest 58/58；CPU E2E 全绿，新增 `workspace_readback` 两形状（回读时序在 CPU debug 真实执行：断言生成产物含 `AllocEventID/WaitFlag/ReleaseEventID<MTE3_MTE2>` 三件套，输出 = 2x / 4x 两路数值）。

验证记录（2026-10-09，容器内 `check-ut` / `check-e2e` 全量）：

- lit 175/175 通过（含两个翻转/新增 S1、S2 夹具，S3 同值/别名、S4 两类末读、
  FR5 下降/verifier/RaceCheck 正负用例）。
- gtest 58/58 通过，其中 `AFIRSynchronizationTest` 4 条：同步原语副作用声明
  （不可 DCE/CSE）、同型 record 不可互换、全序原语驻地约束正负两向。
  （负向断言采用解析失败极性 + `ScopedDiagnosticHandler` 吞诊断；正向用例的
  kernel body 函数需 `private` 可见性。）
- CPU E2E 124 例全绿，含新增 `load_store_direct` 两形状（`128x128x128` 常规与
  `1x67x209` 非规则），断言 `TQueBind<VECIN, VECOUT>` 声明且无 SetFlag/WaitFlag、
  数值等于输入。
- NPU E2E 脚本 `load_store_direct.py` 已就位待上板。
- 构建备注：`ENABLE_AUTOFUSE_MLIR=ON` + `ENABLE_AUTOFUSE_MLIR_TESTS=ON` 时需
  `CANN_3RD_LIB_PATH` 指向含 boost 源码与 lib_cache 的完整三方目录（本仓
  `output/third_party`）；缺省回退 `third_party/` 会触发 boost 源码引导失败。

## 验收标准

1. 缺失清单 7 项全部闭环：S1–S4 编译通过且 E2E 数值正确；M1 双端队列可生成；M2/M3 新操作可表达、可下降、可检查；
2. S1 生成 cpp 与 legacy 逐字符一致（`TQueBind<VECIN, VECOUT, 1>` 声明与 EnQue/DeQue，无多余同步指令）；
3. 现有三种转移产物不变，既有 lit/UT/E2E 全量回归通过；
4. 两个负向 lit 用例翻转为正向，新增用例覆盖功能与异常两类；
5. 无新增默认日志、无编码红线触碰；
6. B 类后续依赖在分层重构设计的依赖清单中登记；本设计随实现更新状态。
