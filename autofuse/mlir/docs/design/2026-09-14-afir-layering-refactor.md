# AFIR 分层重构设计：conversion 目录归属与流水线相位

> 状态：六期已落地
> 日期：2026-09-14
> 文档类型：目录分层与 pass 流水线重构设计

处理 [AFIR DSL 架构重构设计](2026-09-09-afir-public-dsl-architecture-design.md) 5.7 记下的分层问题：
`conversion/AFIRToAscendC/` 今天 60 个文件 8895 行，住着四类职责不同的东西。

不含退出 `TQue`、Op 补齐、Schedule / ATT 迁移、性能基线（见前置文档 5.7）。

# 1 现状

## 1.1 判据

一个 pass 跑完之后，函数体里有没有 `ascendc` / `emitasc` / `emitc` 算子。没有则属 AFIR 层，
有则属 conversion 层。产物混着两种的，里面是分析与发射两件事，拆开：分析归 AFIR 层，发射留下。
所以这次重构以拆为主，移为辅。

按后端符号数 / `afir::` 符号数统计：

| 文件 | 后端 | AFIR | 判定 |
|---|---:|---:|---|
| `Implementations/TransferPlan.cpp` | 0 | 24 | AFIR 调度分析，名字挂错地方 |
| `MemoryAllocate/CanonicalBindings.cpp` | 0 | 55 | AFIR descriptor 收集 |
| `MemoryAllocate/CanonicalRewrite.cpp` | 0 | 2 | 纯改写 |
| `AFIRVerifyKernelComplete.cpp` | 0 | 0 | C1–C3 纯 AFIR；C4 依赖 `target/Ascend/Capability` |
| `AFIRKernelBodySignature.cpp` | 1 | 24 | 算法通用，类型来自 `AscendCKernelABI.h` |
| `MemoryAllocate/Canonical{Resources,Materialization}.cpp` | 12 | 15–39 | 分析通用，发射后端 |
| `AFIRPipelineSync.cpp` | 24 | 22 | 队列路径必然发 `TQue`，见 2.4 |
| `Implementations/{BinaryElementwise,DataMovement,Synchronization}.cpp` | 12–54 | 3–13 | 本来就该在这儿 |

## 1.2 四个族

| 族 | 内容 | 规模 | conversion？ |
|---|---|---:|---|
| A 下降本体 | `AFIRToAscendC.cpp`、`Implementations/*`、`SelectImplementation`、`AscendC{KernelABI,Attributes}`、`TilingSchema` | ~2200 行 | 是 |
| B AFIR 层的 pass 穿后端外衣 | `KernelBodySignature`、`MemoryAllocate` 族、`PipelineSync`、`VerifyKernelComplete` | ~3400 行 | 部分 |
| C host 侧单候选产物 | `GenTilingData`、`GenKernelEntry` | ~820 行 | 否，紧跟 conversion |
| D 多候选聚合与文本发射 | `GenAggregate*`、`AggregateLaunchDriver`、`Emit/*`、`AddKernelIncludes` | 1697 行 / 15 文件 | 否 |

D 族占 19%，后端符号数 0，做的是把多个 plan module 合并成 entry module 再吐三段文本，
调用方是 `bridge/CodegenOrchestration`。它**单向**依赖 A 族（`AscendC{Attributes,KernelABI}`、
`TilingSchema`），A/B/C 不依赖它，所以可以整块抽出。

去处是新建的 `artifact/`，与 `conversion/` 平级且在它下游。不是 `target/Ascend/Launch/`——
`target/` 排在 `conversion/` **上游**（`TargetProfile` / `Capability` 都被 conversion 依赖），
把一个依赖 conversion 的库放进去正是这次重构要消灭的反向依赖。也不叫 `codegen/`：
`autofuse/codegen/` 已经存在，正是这次迁移要取代的 legacy ATT codegen。

`artifact/` 的定位是**交付物装配**——收下降完的 plan modules，产出交给外部编译器的文本；
这是 `mlir/` 里唯一不产出 IR 的一层。两个成员按「产出哪一份源码」分，这也是消费方
（`CodegenOrchestration` 的 `result.kernel` 与 `assembleTilingSources`）自己的说法：

| 成员 | 产物 | 来处 |
|---|---|---|
| `KernelSource` | kernel 源码、TilingData header、AutofuseLaunch host wrapper | `conversion/AFIRToAscendC/` |
| `TilingSource` | Inductor host tiling ABI | `bridge/` |

两个名字必须落在同一条轴上。原来的 `AscendLaunch` / `HostTiling` 一个说目标、一个说
side，读的人看不出它们是兄弟；换成 `KernelSource` / `TilingSource` 之后轴只有一条，
而且 `Source` 正好点出 `artifact/` 的定位——出口是源码文本，不是 IR。

`TilingSource` 一并搬过来，因为它与 `KernelSource` 是同一层的两半：同样收
`ArrayRef<ModuleOp>`、同样吐 `std::string`，并在 `bridge/CodegenOrchestration` 的
`FinalizeMlirCodegenResult` 里被先后调用。此前分开住是按调用方归类，不是按职责。
只搬一个就是为一个成员立一层，那是投机性通用化（2.4 第 2 条）。

`CodegenOrchestration` 留在 `bridge/`：它是编排与外部入口，不是发射器。

`AFIRToAscendC.cpp` 本身 184 行，占以它命名的目录的 2%。

## 1.3 流水线相位

`addPlanCandidateLoweringPipeline`（`AFIRToAscendCPipeline.cpp:86`）九步，相位四段：

```
A 准备      1 KernelBodySignature      补 workspace/TilingData 参数、打 kernel_arg 角色
            2 MaterializeTilingParams  运行期尺寸符号 → SSA 值
B AFIR 物化  3 LoopGenerate             轴 → scf.for
            4 MemoryAllocate           缓冲 → 资源
            5 PipelineSync             顺序 → 同步边
C 下降      6 SelectImplementation
            7 ConvertAFIRToAscendC
D host 产物  8 GenTilingData
            9 GenKernelEntry
```

代码切的是 `1 │ 2..7 │ 8 9`，四条相位边界一条没对上；中间层 `addAFIRToAscendCPipeline`
的名字只覆盖它六步中的两步，且除 UT 外无第二个调用方。

顺序本身对：逐对核过，每条相邻边都有真实约束（3→4 查 `hasGeneratedLoops`，
4→5 查 `hasPendingResourceMaterialization`，5→6 是 ADR-19 要求同步算子先选实现，
8 必须在最后一个 `TilingDataType` 消费者之后）。唯一例外是 2→3：
`materializeAscendCTilingDataBindings` 与 `CanonicalResources` 都调 `getOrCreateTilingParam`，
miss 时自建，`test/lit/demo/afir_memory_allocate_buffer_form.mlir` 的 RUN 行不跑第 2 步也能
CHECK 出 `afir.tiling_param "s0"`。第 2 步实为可选的归一化步骤（统一尺寸序言顺序、坏表达式
一次报错），但名字说的是生产者，且同一个 op 有两个活的产生点（D2）。

## 1.4 两处缺陷

1. TilingData 的身份 A3 只迁了一半：`getCanonicalTilingArgument` 按 `#afir.kernel_arg<role = tiling>`
   找参数，但 `AscendCKernelABI.cpp:162` 的 `TilingDataType::get` 仍要求结构体字面叫 `"TilingData"`。
   第 8 步改名后失配，于是整条流水线的一条排序约束挂在一次字符串比较上。
2. `AFIRToAscendCPipeline.cpp:119` 的 dependent dialects 是手工并集，注释漏了后加的
   `MaterializeTilingParams` 与 `PipelineSync`；`:64` 的 session 校验 `return failure()` 不发诊断。

# 2 方案

## 2.1 三档搬运

**档 1 直接移**（430 行）：`Implementations/TransferPlan` → `dialect/AFIR/Analysis/TransferPlan`；
`LoweringStage.h` 的 `hasPendingResourceMaterialization` → `dialect/AFIR/Transforms/`
（`hasMaterializedAscendCResources` 是后端判据，留下）。

**档 2 一个谓词**（293 行）：`AFIRVerifyKernelComplete` 的 C4 不再收注册表指针，
改收 `ImplementationQuery{targetName, contains}`——查询由调用方注入，
`dialect/AFIR/` 不再需要链接 `target/Ascend/`。

结构体而不是裸的 `function_ref` 是因为还差一个 `targetName`：C4 的诊断要说「缺的是**哪个后端**
的实现」，而把 `"Ascend"` 写死在 AFIR 层就等于承认这条判据只对一个后端成立。两个字段一起进来，
诊断逐字不变（`no Ascend implementation is registered for ...`），AFIR 层却一个目标都不认识。

**档 3 拆类**：`KernelBodySignature`（470 行）、`MemoryAllocate`（1681 行）。

落地时两个都**没有**上 provider seam，理由是逐函数量出来的比例：

- `KernelBodySignature` 里与后端无关的只有资源 id 收集那 ~80 行；其余六个函数（ABI 校验、
  tensor → GM memref、workspace 类型、TilingData 结构体、地址空间改写、kernel 属性）是
  AscendC ABI 本身，通用化需要 **7 个回调**，而底下没有算法——参数顺序是 AscendC 生成的
  C++ 签名的要求。对照 `LoopGenerate`：**一个**回调，通用的那半是真算法（轴 → `scf.for`
  嵌套），壳 46 行。比例完全不同。做的是提出那 80 行（`Analysis/ResourceIds`）。
- `MemoryAllocate` 的分界由状态读写量出来：Bindings 相只碰 `values` / `valueOrder` /
  `nextOrdinal`，完全不碰 `resourceBuilder` / `createdOps`。整相搬进
  `Analysis/CanonicalBindings`，其余三相留下。不需要回调——收集的产物是一张表，不是一次
  回调；物化器读表即可。

**判据补一条**：seam 的收益来自「通用的那半是个算法」。通用的那半只是一串目标决策时，
seam 只是把直调换成 `std::function`，读的人多一跳、编译器少一次内联，分层没有变对。

## 2.2 LIT 不变

CLI pass 名从构造出的 pass 上读（`registerPass([]{ return createAscendCLoopGeneratePass(); })`
注册出来仍叫 `afir-loop-generate`），所以这套拆分对 91 条 LIT 一条不改，测试全程是等价性夹具。
推论：不改任何已注册的 pass 名。

## 2.3 `MemoryAllocate` 排最后

`CanonicalBindings.cpp` 后端符号 0 看着最好搬，但四个 TU 共用的
`MemoryAllocate/CanonicalMemoryAllocator.h` 在类声明里带 `ascendc::TPosition`（`:163`）与
`ascendc::LocalTensorType`（`:185`、`:189`）。搬任何一相都要先把
`CanonicalFunctionMemoryAllocator` 拆成 AFIR 侧状态 + 后端侧物化器。

## 2.4 不动的

1. `AFIRPipelineSync` 不整体搬。队列路径产物就是 `TQueBind{Enque,Deque}TensorOp`，
   搬进 `dialect/AFIR/` 即反向依赖后端方言。等退出 `TQue`。
2. 不给每个 pass 都上 seam。只在确有分层违规时才把直调换成 `std::function`。
3. `ConvertAFIRToAscendC` 名实不符不修。真实方言边界在第 4 步：缓冲要先落成 AscendC 资源，
   引用它的算子才能转换。边界回到第 7 步要等 ADR-11/12。

   **债的边界可以量出来，不必靠推测。** `AscendC_LocalTensor` 的全部类型参数就是 `shape` + `elementType`
   （`api/ascendc/include/ascir/Dialect/Asc/IR/Core/Base.td:33`，`Types.td:71` 无额外参数），而 AFIR 的
   `AFIR_SpaceEnum` 覆盖 GM/UB/L1/L2/L0A/L0B/L0C（`AFIREnums.td:56`）且 Bridge 建类型时已挂在 memref 上
   （`AfirBuilder.cpp:200`）。即
   `!ascendc.local_tensor<?xf16>` 与 `memref<?xf16, #afir.space<ub>>` **信息量相等**。

   两条推论：(a) ADR-1 的推翻条件「memref 无法表达某类必需语义且无扩展点」**未被触发**，第 4 步的类型替换
   不是 ADR-1 的反例，而是一笔有理由的技术债；(b) 把物化推后到第 7 步不会让中间三个 pass 少看到任何事实——
   位置信息始终在 `#afir.space` 里。因此解除该债的真实阻塞只有 ADR-11/12（退出 `TQue`），而非类型表达力。
   归属于 AFIR 层的 pass 不应绑定底层 API，这是分层原则，第 4 步的现状是该原则的已知欠账，不是对它的修订。

## 2.5 流水线相位

```cpp
LogicalResult addPlanCandidateLoweringPipeline(OpPassManager &pm, const Session &s) {
  if (!isUsableSession(s)) {
    return failure();
  }
  addKernelABIPreparation(pm);      // 1, 2
  addAFIRMaterialization(pm);       // 3, 4, 5
  addAscendCLowering(pm, s);        // 6, 7
  addHostArtifacts(pm);             // 8, 9
  return success();
}
```

四个 `add*` 返回 void，不是各自返回 `LogicalResult` 串起来：它们只往 `OpPassManager` 里塞 pass，
没有一步会失败。唯一能失败的是 session 不完整，那是**入口**的判据，所以提成 `isUsableSession`
在最前面问一次。原先它藏在 `addAFIRToAscendCPipeline` 里，于是 `addPlanCandidateLoweringPipeline`
失败返回时 pm 里已经躺了一个 `KernelBodySignature`。

配套四条：

1. `afir-to-ascendc-pipeline` 注册名保留（4 个 LIT 在用），重定义为 B+C。第 2 步挪进 A
   之后 4 条 LIT 的 CHECK 一条没动——尺寸序言本来就由消费者 create-on-miss 建出来，
   顺序不取决于谁先跑（见配套 4）。生产路径走的是 `addPlanCandidateLoweringPipeline`，
   这个注册名今天只有 LIT 与 UT 在用。
2. `TilingDataType::get` 的字面名匹配换成 `#afir.kernel_arg<role = tiling>` + schema 校验。
   落地时发现去掉名字判据之后这个类只剩 `getTilingSchema` 的一层包装，于是整个删掉。
3. `getStaticDependentDialects` 不再手写并集：每段 `add*` 函数同时提供自己的方言集，父 pass 求并。
4. `MaterializeTilingParams` 写明它可选、消费者 create-on-miss。**不改名**：它确实在创建
   op，只是不是唯一的创建者，改成「归一化」会更不准确；改 CLI 名还要动 2 条 LIT。

# 3 门禁

`tools/af-mlir-tools/check_afir_layering.py` 今天只禁 `dialect/AFIR/` 出现 `afplan`，扩两条：

- `dialect/AFIR/` 源码不出现 `ascendc` / `emitasc` / `emitc` / `ascir/`；
- `AFIRTransforms` / `AFIRAnalysis` 的 `LINK_LIBS` 不出现 `MLIRAsc` / `MLIREmitAsc` /
  `MLIREmitCDialect` / `MLIRTargetAsc`。

`ascir/` 单独列一条，因为它是后端方言头文件的路径前缀，而
`#include "ascir/Dialect/Asc/IR/Asc.h"` 这一行里没有 `ascendc` 这个词。同理，`#include` 的路径
**就是**字符串字面量，会被「抹掉注释与字符串」那一步一起抹掉，所以 include 指令要在抹之前
单独扫一遍——这是原来那条 `afplan` 规则也拦不住的口子，而它正是依赖进门的主要方式。

排在搬运之前。今天的保护是隐式的——`AFIRTransforms` 本来就不链接后端库，改一行 CMake 就没了，
而链接期不报错。门禁要能被验证挡得住：人为往 `dialect/AFIR/` 注入一行后端 `#include`，
脚本返回 1 并指名道姓。

# 4 分期

| 期 | 内容 | 验收 |
|---|---|---|
| 1 | 门禁扩两条 | 对人为注入的违规返回 1；已接 pre-commit |
| 2 | D 族迁出，与 `HostTiling` 合成 `artifact/` | 四套测试全绿、无 LIT 改动；目录减 19% |
| 3 | 档 1 + 档 2 | 同上；`AFIRVerifyKernelComplete` 不再链接 `AscendInstructionCapability` |
| 4 | 流水线相位 + 2.5 的四条 | 九步顺序逐字不变；两个 pipeline pass 的 11 条 LIT 全绿 |
| 5 | `KernelBodySignature` 拆出 `Analysis/ResourceIds` | 同 2 |
| 6 | `MemoryAllocate` 拆类，Bindings 相归 `Analysis/CanonicalBindings` | 同 2 |

六期均已落地，每期四套测试全绿且 **LIT 零改动**。`conversion/AFIRToAscendC/`
8895 行 60 文件 → 6237 行 40 文件。

期 4 与期 6 各抓出一处同型的既有缺陷：按「倒数第二个是 memref、倒数第一个是 tiling
结构体」反推 ABI 参数，而不是问 `#afir.kernel_arg`。期 4 新加的方言对账用例
（`DeclaredDependentDialectsCoverTheRealPipeline`）第一次跑就抓到两处既有漏声明。

纪律同前置文档 5.3：先确认 LIT 不需要改，再动生产代码。某一期里出现"顺手改一条 CHECK"，
说明这次搬运不等价，停下来解释。

# 5 与前置文档的关系

本文取代前置文档 5.7 末尾那段分层说明。五项剩余工作不受影响。退出 `TQue` 与本文 2.4 的
第 1、3 条互为前置——它落地后 `PipelineSync` 的队列路径与 `MemoryAllocate` 的资源发射才可能
回到 AFIR 层。本文六期不等它，也不为它预留结构。
