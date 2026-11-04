# elementwise / broadcast 收尾：开发设计

> 2026-10-16 · 上游：
> [elementwise 遗留问题清单](2026-09-22-afir-elementwise-remaining-issues.md)、
> [广播设计](2026-09-23-afir-broadcast-design.md)、
> [向量化轴分段统一](2026-09-28-afir-vectorized-axis-collapse-design.md)、
> [泛化用例测试设计](2026-10-12-afir-elementwise-broadcast-generalization-test-design.md)
>
> 本文按 `docs/guidelines/design_document_template.md` 的章节写，**取代上述四份文档的「下一步」
> 各节**作为 elementwise / broadcast 这条线收尾阶段的唯一工作清单。各份文档仍是详情来源。

# 简介

## 目的

把 elementwise / broadcast 从「若干形状已验证」收到一个可验收的终点：
**在 A2/A3 与 A5 两档平台上，legacy 跑得通的 elementwise / broadcast 图，MLIR 路径也必须跑通，
且不依赖回落。**

目标读者：继续做 AFIR codegen 的开发与检视者。

## 范围

**包含**：

- `afir` 的 elementwise 族与 `afir.broadcast`（含其搬运侧依赖 `Load` / `Store` / `Nddma` / `RemovePad`）。
- 两档平台：mem-based（NpuArch 2201，A2 ≡ A3）与 reg-base（NpuArch 3510，A5）。
- 入口只有 inductor（`Codegen::GenerateForInductor`）。
- legacy 自有 e2e 用例向本仓 MLIR 用例集的搬运。

**不包含**：

- CV（Cube）融合。代码层面已经挡住：`IsCubeFusedScheduled` 的图在 `RunMlirCodegen`
  入口即返回 `kNotSupported`（`CodegenOrchestration.cpp:351`），本设计不改这一条。
- reduce / concat / gather / transpose / split / topn / argmax / indirect-load 等非 elementwise 族。
- IR 形态优化（ADR-21 / ADR-22、`ViewLikeOpInterface`）。按 2026-10-16 决定**搁置**，
  不在本设计内，也不作为任何一项的前置。
- TensorFlow 入口（`Codegen::Generate`）。

涉及组件：Autofuse Codegen/Backend（MLIR 侧）、测试框架。**不涉及 Autofuse 图优化**——
N1 原以为要改调度侧，实测那是一条有意的约定，缺陷全在 MLIR 侧（见功能需求 1）。
不涉及 SuperKernel、打包、Python 接口。

# 总体概述

## 软件概述

### 项目介绍

MLIR codegen 路径接在 `Codegen::GenerateForInductor` 上，由 `ENABLE_AUTOFUSE_MLIR` 与
`AF_ENABLE_MLIR_CODEGEN` 双开关控制。elementwise / broadcast 是这条路径上第一个被完整接通的
算子族，前序四份文档已经把 mem 侧的单算子面、多轴形状面与广播级联面做到可量化。

本阶段要回答的不再是「这个 op 通不通」，而是「**对着 legacy 的全集，还差哪些**」。

### 产品环境介绍

- 依赖 CANN Toolkit 提供的 AscendC 头（`autofuse/ascendc` 与 `autofuse/v35/ascendc`
  两套 API，分别对应 mem / reg 两侧）与 tikicpulib（CPU 模拟器）。
- 对外接口无变化：MLIR 路径是 `Codegen` 内部的一条分支，不新增导出符号、不改 Python 绑定。
- `AF_ENABLE_MLIR_CODEGEN` 的语义本阶段要补一位（见功能需求 1），它是**测试用环境变量**，
  不是产品配置项。

## 软件功能

1. 修 N1（MLIR 侧两处：撤掉 `afir.nddma` 的自相一致判据、收 Store 的 GM 偏移回落），
   打开 reg 侧被它整体挡住的那一面。**已落地。**
2. 把「MLIR 是否真的接管了这张图」变成可观测、可断言的事实（严格模式 + 未支持原因标签）。
3. 按**宽度优先**补齐元素类型：先让 dtype 的真源可机读，再把 legacy 收而 AFIR 不建模的
   元素类型补进来（bf16 为首）。
4. 评估并排期 **VectorFunc**：A5 调度阶段的二次融合，今天 MLIR 完全没有涉及。
5. 补齐算子：A5 还差 83 个 elementwise 节点类型。
6. 收掉形状泛化的剩余边界（尾轴不对齐、最内轴是广播轴、一处能力/pattern 漂移）。
7. 把 legacy 自有 e2e 用例搬进 MLIR 用例集，并顺带收掉用例样板成本。

## 设计约束

- **CANN / AscendC**：新增 dtype 与新增 op 的供给表必须与 AscendC 头实际收的集合一致，
  不得按 legacy 注册表的并集放宽——两者不同时以头文件为准。
- **不得放宽 verifier**：拒收只能拒在 capability 层，verifier 失败是硬失败
  （`CapabilityShortfall.h` 的两类划分）。
- **ABI/API**：无对外接口变化。`AutofuseTilingData` 的字段布局不因本设计改变。
- **构建**：并行度一律 `-j 6`（本机 `-j 8` 会被 OOM killer 杀掉）。
- **日志**：不新增默认开启的高频日志；严格模式的诊断只在环境变量打开时输出。

## 假设和依赖关系

已核（2026-10-16 实测或读码确认）：

- `run_npu_e2e.sh` 的 10 个 NPU E2E 用例已上板全部通过（用户实测）。**真机验证不再是欠账。**
- A2 与 A3 等价：`Ascend910B1.ini` 与 `Ascend910_9381.ini` 只差 `SoC_version` /
  `Short_SoC_version` 两个名串与 `cube_freq`（1850 → 1800）；`NpuArch` 同为 2201，
  `ub_size` 同为 196608，核数同为 24/48。**A3 不单独入表。**
- 回落确实存在，但只对「能力不足」这一类：`codegen.cpp:403` 对 `kNotSupported` **不返回**、
  继续跑 legacy；`kFailed` 直接返回失败。分类点在 `CodegenOrchestration.cpp:378`。
- 未建模的节点类型走 `emitCapabilityShortfall`（`ScheduleToAfir.cpp:1087`），即**静默回落**。
- VF 融合只在 A5：`PlatformV1::PartitionSubFunctions` 是空实现
  （`autofuse/optimize/platform/v1/platformv1.cpp:30`），V2 才真正分区。

待确认（各自在对应功能需求里给出确认方法）：

- ~~**VF 实际吞掉多大一片**~~：**已实测（2026-10-17）**，reg 侧 200 格里 110 格的调度产物
  带 `VectorFunc`（55%），其中只有 17 格是「只差 VF」。先后结论见功能需求 4 第 4 条：
  功能需求 5 与 VF 正交，可以先做。
- 新增 dtype 后 ATT 侧 tiling 的 dtype 分流是否全部跟得上（`b{id}_size` 的字节数计算）。
- `Frexp`（1 入 2 出）与 `Arange`（0 入 1 出）是否纳入本阶段，见功能需求 5。

# 需求分析与设计

## 整体介绍

差距按**四条相互独立的轴**分解，这四条轴的交点就是「legacy 能跑而我们不能」的全部来源：

```
轴 1  节点类型      A2/A3 差 5（均非简单 vector op）；A5 差 83
轴 2  元素类型      AFIR 只建模 6 种：f16 f32 i16 i32 ui8 ui16
                    缺 bf16 / i8 / i64 / ui32 / ui64 / bool
轴 3  图结构        A5 调度把多节点子图融成 VectorFunc —— MLIR 零覆盖
轴 4  形状 / 布局   尾轴不对齐、最内轴是广播轴、nddma N1（已收）
```

**四条轴不可互相替代，也不可互相代偿**：补了 op 不解决 dtype，补了 dtype 不解决 VF，
而 VF 吞掉的恰好是**已经接通的那批 op 组成的多节点图**——所以它不是「以后再说」的一项。

数据流不变：Schedule（legacy）→ Bridge（`ScheduleToAfir`）→ AFIR → 下降 → AscendC →
kernel + tiling。本设计只在 Bridge、conversion 与 AFIR 分析三层加东西，**Schedule 一处不动**。

### 各项先后（2026-10-17 更新）

四条轴彼此独立，但**各项的开工顺序不独立**，因为排期依据来自度量：

```
功能 2（接管表）──┬─→ 功能 4 的度量 ──→ 由 VF 那个数定「功能 4 实现」与「功能 5」的先后
                  │                      （文档原话：这个数决定两者先后）
                  └─→ 功能 3 的收益也落在 A5 那半边，要同一张表才看得见

功能 6 的 F-4 ────────────────────────→ 功能 7（用例搬运）的前置
```

两条硬依赖，不要跳：

1. **功能 2 的接管表是功能 3 / 4 / 5 共同的排期依据。** 它没出全（见功能需求 2 的状态），
   动功能 3 或功能 5 就是在猜顺序。
2. **F-4 是功能 7 的前置。** 今天把非 elementwise 的 legacy 用例搬进来，不会「回落后变绿」
   而是当场红（mem 侧实测 14 格）。

当前位置：功能 1、功能 2 已闭环（两侧接管表已出），**功能 4 的度量已闭环**（2026-10-17）。
那个数不是 17 而是 110 / 200：17 只是「只差 VF」的格数，而「含 VF、不接 VF 就进不来」的是
110 格（55%）。由它定下的先后是——**功能 5（补 op）与 VF 正交，可以先做**；VF 的大头
（48 格）要与功能 6 的 F-4 一起才解得开。功能 4 的**实现**另立文档，功能 3、5、6、7 未开工。

*更新（2026-10-26）*：功能 3 与功能 5 已推进并有全量度量——dtype 真源已收且带门禁、bf16
已开位，算子补齐做到第 7 批（bridge 白名单 47 → 99）。全量 200 格 reg 侧接管表重量结果见
`2026-10-26-legacy-takeover-table-reg-full200.txt`：takeover 15 → 63、fallback 117 → 69、
hard-failure 66 不变、0 格变差。功能 6、7 仍未开工。

## 功能需求 1：N1——`afir.nddma` 接通与 Store 的 GM 偏移（**已落地，2026-10-17**）

1. 介绍

   > **本节的原方案（「调度侧 `GenNddmaNode` 补抄 `strides`」）是错的，已订正。**
   > 原文把搬运节点上的 `strides` 当成「漏抄」，而它是一条**有意**的约定：搬运节点的
   > descriptor 把两侧合在一份里——`strides` 记它自己那一侧的 GM 布局，`vectorized_strides`
   > 记片上那一侧。legacy 的 `NddmaApiCall` 正是从 `strides` 这一列取 `input_stride` 与 GM
   > 偏移（`v35/codegen/reg_api_call/reg_nddma_api_call.cpp:74/76` →
   > `codegen/api_call/utils/api_call_utils.cpp:116` 的 `gm_tensor.axis_strides`；
   > `codegen_kernel.cpp:428` 把 `attr.strides` 装进 `Tensor::axis_strides`），该约定写在
   > `v35/optimize/template/nddma_template.cpp:858` 的注释里，`IsTailBroadcastNddmaNode` 的
   > 打分判据也建在它上面。按原方案抄过去，`[8,1] → [8,128]` 这一格 legacy 的 `input_stride`
   > 会从 `{1,0}` 变成 `{64,1}`、GM 偏移从 `z0×1` 变成 `z0×64+z1`：**把 legacy 自己的 nddma
   > 搬运搬错地址**。另一条来路 `GenLoadToGenNddmaNode`（`v35/optimize/un_alignment_strategy.cpp:261`）
   > 只改节点类型、什么都不抄，它的两列本来就不相等——不相等正是它要转 nddma 的原因。
   >
   > 所以**调度侧一行不动**，缺陷全在 MLIR 侧两处（见下）。原方案要求的「独立提交 + v35 ST
   > 单独验」随之作废：legacy 产物逐字不变。

2. 输入

   reg 侧（NpuArch 3510）任何带 `afir.nddma` 的图：`Load → Broadcast` 被 `NddmaTemplate`
   融掉的，以及尾轴不连续被 `LoadToNddmaTemplate` 转过来的。

3. 处理（两处，都在 MLIR 侧）

   - **撤掉 `Nddma.cpp` 的 `resultDescriptorIsSelfConsistent`**。它要求向量化轴上
     `strides == vectorized_strides`，而这一族的定义就是两列说两侧，于是把整族判成
     「上游给错了」。它答错的是**归属**。本档要的 GM 步长逐轴问 `globalAxisUnitStride`、
     片上步长从 `collapseVectorizedAxes` 的段里拿，谁也不需要那条等式。
   - **收 Store 的 GM 偏移回落**（`TransferPlan.cpp`）。真受害者是下游 `afir.store`：它的
     局部 descriptor 是**借上游的**，上游是 nddma 时借到的是输入张量的行距，而
     `OffsetContext::strideOf` 在 GM 答不出的调度轴上会回落到这一列。新增
     `localStridesAreLayoutProxy`：向量化轴上两列说同一件事才把这一列填进 `localStrides`，
     不同就整张留空；留空后 `axisUnitStride` 在「往下没有子支」时沿 `from` 往**上**问 GM
     descriptor（`z0T` 从 `stride(z0) = 128` 乘回 `128 × z0t_size`）。
     判据按 descriptor 判而**不按「上游是不是搬运 op」判**：下降到那一刻 `afir.nddma` 可能
     已经被换成 AscendC 那条调用，操作数定义点不再带 `AFIRTransferIn` trait——实测那样写的
     判据答「能用」，GM 偏移照旧少乘一个 128。

4. 输出（实测）

   - `test/e2e/cpu/CMakeLists.txt` 那行 `add_subdirectory` 放开，
     `load_broadcast_nddma_store` 四格（`[8,1]` / `[1,8]` / `[1024,1]` / `[1,2,1,8]`）
     数值全绿，**用例文件一个字没改**。会变红的埋点就是 `[1024,1] → [1024,128]`：69 个
     分块，偏移少乘一个 128 时 130982 / 131072 个元素错；`[8,128]` 只有一个分块、碰巧全绿，
     所以这一格不可省。
   - 探针（`af-capability-probe`，改动前后各跑一遍）：
     **reg 侧 68 → 165 行 `ok`**（逐行比对 95 格 `unsupported → ok`、0 格变差），
     mem 侧两份表**逐字相同**。两侧 `DRIFT` 恒为 0；mem 侧那 2 条 `FAILED`（T3 的
     `[1,8] → [128,8]` 与 `[1,1,8] → [4,4,8]`）改动前后都在，是既有项，不属本次。
   - `bash ./mlir-test.sh` 四目标全绿（check-ut / check-lit / check-st / check-e2e）。

## 功能需求 2：严格模式——让「谁接管了这张图」可断言（**已落地，2026-10-17**）

> **状态**：严格位、闸门标签、量具脚本与 6 条埋点已落地；接管表两侧都已出
> （mem 38 行 + reg 200 行），**未知 0、缺二进制 0**。本节验收标准达成。

1. 介绍

   今天 inductor 路径对「能力不足」是静默回落，于是**拿 legacy 套件跑一遍全绿，证明不了
   任何事**：回落了也是绿的。本需求补一个只在测试里打开的严格位，把 shortfall 也判成失败，
   并把未支持的原因打成可聚合的标签。

2. 输入

   `AF_ENABLE_MLIR_CODEGEN` 的取值扩一档（如 `strict`），或新增一个只读一次的环境变量；
   取值解析与现有 `ReadMlirCodegenOptions` 同一处，不另立配置通道。

   **落地取值表**（`CodegenOrchestration.h`）：`"1"` 开、缺省语义不变；`"strict"` 开且严格；
   其余拼写（含未设置、`"STRICT"`、`"strict "`、`"strict1"`）一律关。**不做前缀 / 大小写 /
   布尔近似匹配**——猜错的后果是把回落变成失败、整条编译挂掉。

3. 处理

   - 严格模式下 `classify()` 一律返回 `kFailed`，于是 `GenerateForInductor` 直接失败，
     用例变红而不是悄悄走 legacy。
   - 原因标签复用探针已有的分桶（`UnsupportedOperation` / 能力不足 / `DRIFT` / `kFailed`），
     桶按**是哪道闸门拦下的**分，不按 op 分。
   - **缺省行为一个字不改**：不设严格位时，回落语义与今天完全一致。

   **落地与原方案的两处差别**：

   - **标签走结构化的闸门枚举，不走诊断文本的分桶。** 原方案说「复用探针已有的分桶」，而探针
     那套桶是**解析诊断文本**得来的，`support/CapabilityShortfall.h` 明确写了产品代码不得按
     文本分类（`UnsupportedOperation` 那个前缀今天两种失败都在用）。所以产品侧新增
     `MlirCodegenGate` 枚举，严格模式下在每个出口打一行

     ```text
     AF_MLIR_STRICT gate=<哪道闸门> shortfall=<yes|no|unknown> graph=<图名>
     ```

     闸门取值：`none`（接管）/ `cube-fusion` / `soc-version` / `candidate-preparation` /
     `pipeline-session` / `lowering` / `artifact-assembly` / `exception`。口径与探针的
     `Bucket()` 相同（按闸门分不按 op 分），但判据是结构而不是字符串，所以**探针那份
     `Bucket()` 不必抽出来共用**，聚合直接按 `gate` 做。
   - **`shortfall` 单列一位，且是三态。** 严格模式把结论一律压成 `kFailed`，于是「能力不足
     回落」与「真坏了」在状态上分不开——而那正是表里三态中的两态。这一位记的是观察者看到的
     事实（缺省那一档会不会回落），与 `gate` 不能互推：同一道 `lowering` 闸门两种都可能。
     抛异常那一支里观察者已出作用域（它与 `MLIRContext` 一起建在 `try` 内），答不出来就打
     `unknown`，**不许顺手填 `no`**——填了就是把「不知道」说成「没有」。

   **两道闸门严格模式故意不收紧**：开关没开（严格模式本身就是它的一档取值），以及 Cube 融合
   （明确不在本设计范围，收紧只会让 legacy 的 Cube 用例整片变红）。于是 `kNotSupported` 在
   严格模式下只剩这两个来路，「回落」与「不在范围」因此分得开，而 `gate=cube-fusion` 这个标签
   让接管表**不必按用例名猜哪些是 Cube**。

4. 输出

   一张以 legacy 用例为行的表：`用例 → {接管 / 回落(第一条诊断) / 硬失败}`。
   这张表是本设计后续每一项的验收手段，也是功能需求 4 待确认项的测量工具。

   **量具**：`autofuse/mlir/test/probe/legacy_takeover_table.sh`（与 `af-capability-probe`
   同目录、同性质——不断言任何事，不接任何 `check-*` 目标）。它在严格模式下逐个跑 legacy 的
   `*_codegen` 那一阶段，按 `AF_MLIR_STRICT` 标签把每个用例收成一行，状态四态：
   `takeover` / `fallback` / `hard-failure` / `out-of-scope`，另有 `no-mlir-call` 一桶专门装
   **状态未知**的行（用例根本没调 codegen，或二进制没编出来）——验收标准要求这一桶为 0，
   所以它必须单列而不是混进 `fallback`。

   一个用例里多张图时取**最差**那一格：有一张没接管，这个用例就不算接管。

   **前置条件与两条已知口径限制**（都写在脚本头部，读表前先看）：

   - 跑它要先把 build 配成 legacy ST 那一档
     （`bash scripts/test/run_autofuse_test.sh -s -m backend --enable-mlir -j 6`），
     **这一步会覆盖 `mlir-test.sh` 用的那一档配置**（MLIR 自己的测试目标用 `cmake -U RUN_TEST`
     配，两档互斥）；量完删 `build/CMakeCache.txt` 复原。脚本不替你改配置，只在配好的目录上
     做「编 + 跑 + 收表」。
   - **入口是 TF 而不是 inductor。** legacy 的 `*_codegen` 用例走 `Codegen::Generate`
     （`MlirCallerScene::kTensorFlow`）。场景这一位会传进 ATT（`is_inductor_scene`），而 ATT 在
     候选准备那道闸门里面——所以 **inductor 与 TF 在 `gate=candidate-preparation` 这一道上的
     差异，这张表量不到**；其余各道（bridge、能力层、pattern、产物装配）与场景无关。对齐这条
     口径要给 legacy 用例样板加一个 inductor 入口开关，归功能需求 7。
   - **reg 侧曾经量不了，已解锁（2026-10-17）**：`autofuse/tests/v35/st/optimize/runtime_stub.h`
     的 `RuntimeStubV2` 覆写了 `rtGetSocVersion`（`"Ascend910_9591"`）与 `rtGetSocSpec`
     （NpuArch 3510），但**没覆写 `aclrtGetSocName`**，于是继承基类那个 `"Ascend910B1"`。
     legacy 只读 NpuArch 所以一直无所谓，而 MLIR 的 `ResolveSocVersion` 读的正是
     `aclrtGetSocName`——用例给的 `CodegenOptions{}` soc_version 为空，MLIR 就按 **mem 侧**的
     Ascend910B1 查 TargetProfile，对着一份 reg-base 的调度产物。**这样量出来的 169 行不报错、
     看起来像一张正常的表，但每一行都按错的 target 出的结果**，所以本轮只量 mem 侧。
     **解锁做法**：给 `RuntimeStubV2` 补 `aclrtGetSocName` 覆写，返回 `Ascend950PR_9599`
     （该 ini 的 `NpuArch=3510`，与同一个 stub 里 `rtGetSocSpec` 答的 arch 对得上；取值与
     `mlir/test/e2e/cpu/AddCpuCase.cmake` 的 `reg_base` 那一支一致）。**没有动
     `rtGetSocVersion`**——那个 `"Ascend910_9591"` 在 `platform_config/` 里确实没有对应 ini
     （那边是 `Ascend950DT_9591`），但 legacy 读的是它，改了会动 legacy 的平台分流。
     **爆炸半径已实测**：这个 stub 有 178 个文件在用，其中 v35 的两个 matmul 用例会经
     `codegen/codegen_tiling_cube.cpp:461` 把它装进 `compile_info.soc_version` 再进生成的
     tiling 文本。在 MLIR 关闭的情况下跑了一遍全套 v35 backend ST：**147 + 146 全绿**，
     两个 matmul 用例的 codegen 与 e2e 两阶段都 Passed。

   **mem 侧首次实测（2026-10-17，38 个 `*_codegen` 用例）**：

   | 状态 | 数 | 说明 |
   |---|---|---|
   | `takeover` | 13 | MLIR 发的产物 |
   | `hard-failure` | 21 | 被拦下，且 `shortfall=no` |
   | `out-of-scope` | 1 | `inductor_matmul_elemwise`，`gate=cube-fusion` |
   | `no-hook-call` | 3 | 挂钩点没被调到，已逐个归因（见下） |

   闸门分布：`candidate-preparation` 14、`lowering` 7、`cube-fusion` 1、`none` 13。

   **本次量出来的两个实质结论**：

   1. **那 21 行在缺省那一档（`AF_ENABLE_MLIR_CODEGEN=1`）今天也是红的，不是静默回落。**
      已实测三格（`compare_test` / `concat_all_aligned_test` / `sub_abs_test`）确认，
      与严格模式无关——`shortfall=no` 时 `classify()` 两档都返回 `kFailed`。
      也就是说**「MLIR 这一侧收不了的图会静默回落」这个前提对 mem 侧 legacy 套件不成立**：
      它们是硬失败。
      其中 14 行拦在 `candidate-preparation`，而**这道闸门没有「能力不足」这个词汇**——
      `PlanSelector.cpp:99` 把候选枚举失败一律当成「坏了」。可是 concat / topn / reduce /
      int64 比较这些图的节点类型 MLIR 本来就没建模，它们该走的是能力不足那一路。
      于是这条仓库约定（**拒收只能拒在 capability 层**）在非 elementwise 族上是破的：
      **第一道闸门就是 plan selection，而它答不出「不支持」，只答得出「坏了」。**
      这与 F-1 那条 `DRIFT` 是同一类问题的另一半，但比 F-1 大：F-1 是 4 格，这是 14 格。
      **本节不修它**（修法要动 PlanSelector 的失败分类，范围超出严格模式），入册。
   2. **`no-hook-call` 那 3 行已归因，都不是坏了**：`conv2d_elemwise_test` 与
      `matmul_elemwise_test` 调的是 `TilingLib::Generate`，`pgo_add_abs_test` 直接调
      `GenerateKernel` / `GenerateTilingData` / `GenerateTiling`——挂钩只在 `Codegen::Generate`
      与 `Codegen::GenerateForInductor` 两处，这正是 2026-08-26 launch 设计记下的已知限制
      （`pyautofuse.cpp` 的四个 Handle 函数走的也是这条绕过）。所以这一桶不是「状态未知」，
      但量具分不开它与「二进制没编出来」，**所以每一行都要人工归因一次**，脚注如此要求。

   **量错过一次，记在这里**：第一版用 `ctest --output-on-failure`，而它**只在用例失败时**打
   输出——「接管了因此变绿」的用例一个标签都收不到，17 行被记成没进过 MLIR，连 `add_abs_test`
   （UT 自己那张 fixture）都在里面。判据是：接管数为 0 的表一定是量错了。已改成 `-V`。

   **reg 侧首次实测（2026-10-17，200 个 `*_codegen_v2` 用例）**：

   | 状态 | 数 | 说明 |
   |---|---|---|
   | `takeover` | **3** | `load_leaky_relu` / `load_tanh` / `remainder_int32` |
   | `fallback` | **116** | 干净回落：107 条「bridge 不认识这个节点类型」+ 9 条「能力层没有实现」 |
   | `hard-failure` | 79 | 55 拦在 `candidate-preparation`、24 拦在 `lowering` |
   | `out-of-scope` | 2 | `gate=cube-fusion` |

   未知 0、缺二进制 0。闸门分布：`lowering` 140、`candidate-preparation` 55、
   `cube-fusion` 2、`none` 3。

   **与 mem 侧一比，两侧的失败形态完全不同**：mem 侧 0 格干净回落、21 格硬失败；
   reg 侧 116 格干净回落、79 格硬失败。原因是 reg 侧绝大多数图在 **bridge 那一层**就被
   「这个节点类型没建模」挡住（带能力不足标记，所以是回落），而 mem 侧的节点类型大多认识、
   于是走到更深的闸门才失败。**这条差别说明 F-4 不是「所有图都硬失败」，它只在
   「bridge 认识节点类型、但 plan selection 枚举不出候选」这一簇上出现。**

   **功能需求 4 要的那个数（VF 占比）**：

   `fallback` 的 107 条按节点类型分，**`VectorFunc` 占 17 格**，是最大的单一桶
   （其后是 `IndirectLoad` 8、`Gather` 3、`Transpose` / `Floor` 各 2，余下都是 1）。
   **17 / 200 = 8.5%，而且这是下界**——它数的是「VF 是第一个拦下这张图的东西」，含 VF 但
   更早被别的缺口挡住的图不在其中。

   按本设计功能需求 4 的原话（「如果 VF 只命中极少数图，A 作为过渡是可接受的」）：
   8.5% 不算极少数，但也远没到「A5 的多节点图基本都没走 MLIR」那种量级——那个判断是按
   「两个以上相邻常见 elementwise 节点就会被融」推出来的，实测没有兑现成用例数。
   **原因候选有两个，选型前要先分开**：这 200 个用例多是单算子图（够不上
   `kMinVfNodesNum = 2`），还是多节点图被更早的缺口挡住、于是 VF 还没轮到？
   前者说明 VF 真的只是个中等缺口，后者说明这个数还会长。**判法**：对 169 个用例目录数一遍
   调度产物里带不带 `VectorFunc` 节点，与这 17 格比——差值就是「含 VF 但被更早挡住」的那些。
   这一步属功能需求 4 的度量，尚未做。

   **会变红的埋点**（`test/ut/bridge/CodegenOrchestrationTest.cpp`）：

   | 用例 | 钉的是 |
   |---|---|
   | `StrictValueIsTheOnlySpellingThatEnablesStrictMode` | 取值白名单，一半是反例 |
   | `StrictModeTurnsCapabilityShortfallIntoFailure` | 同一张图、同一处埋点，缺省回落而严格变红 |
   | `StrictModeStillFallsBackForCubeFusion` | Cube 那道闸门严格模式也不收紧 |
   | `StrictModeKeepsHardFailureFailing` | 严格模式只升级回落，不降级失败 |
   | `StrictModeLabelsTheGateThatStoppedTheGraph` | 四格逐格钉 `gate=` 与 `shortfall=`，并要求标签**恰好一行** |
   | `DefaultModeEmitsNoGateLabel` | 缺省那一档一个字节都不多输出 |

   变异验证：把 `classify()` 里那个 `if (strict)` 短路掉重编，
   **只有 `StrictModeTurnsCapabilityShortfallIntoFailure` 变红**，其余三格仍绿——说明那几格是
   彼此独立的判据，不是同一件事写了四遍。闸门归属那一格的「恰好一行」是专门防
   「插错位置不会让任何既有用例变红，只会让接管表把一行归到别的桶里」的。

## 功能需求 3：元素类型补齐（宽度优先）

1. 介绍

   按 2026-10-16 的决定走**宽度优先**：先补 dtype，再补 op。

   今天 AFIR 的供给表只有 6 位——`kF16` / `kF32` / `kI16` / `kI32` / `kUI8` / `kUI16`
   （`Implementations/ElementwiseApiFamily.h:72`）。**legacy 收而我们不建模的有
   bf16 / i8 / i64 / ui32 / ui64 / bool。**

   要订正一处散在注释里的说法：「AFIR 整个不建模 bf16」**不准确**。Bridge 建得出来
   （`ScheduleToAfir.cpp:52` 发 `"bfloat16"`、`AfirBuilder.cpp:171` 解析成 `BFloat16Type`、
   `ElementByteWidth` 也有 bf16 一行）。真正挡住的是两层供给表：ODS 的 `defvar`
   （如 `Elementwise.td:628` 的 `AFIR_AddElementTypes = [I16, I32, F16, F32]`）与上面那 6 位。

2. 输入

   每个 op 两侧 AscendC 头实际收的元素类型集合。**以头文件为准，不以 legacy 注册表为准**；
   两者不一致时按头文件收窄，并把差异写在注册处。

3. 处理

   **先收真源，再加位**，顺序不能反：

   - **第一步（L-18 第 1 条）**：今天一个 op 的 dtype 事实要写 3~4 遍——ODS 的 `defvar`、
     ODS 的 `logicalResultElementType`、conversion 注册处的位掩码（每侧各一份）、
     `Ops/Cast.cpp` 的对表。其中 **ODS 那一份可以由 conversion 那两份机械求并集**得到
     （「AFIR 能不能表达」= 两侧「供不供得起」的并集），现在是手写第三遍。
     收成一份可机读的真源 + 一条门禁，否则下一步是 47 个 op × 手改 3 处，而且写错不会红。
   - **第二步**：加 `kBF16`，按 op 逐个核头文件后开位；随后按同样方式处理
     i8 / i64 / ui32 / ui64 / bool。
   - bool 单独说明：legacy 的 `DT_BOOL` 在 AscendC 侧多以 `uint8_t` 承载，
     接之前先确认它与 `kUI8` 是不是同一位，**不是就不要合**。

4. 输出

   - 83 个缺口 op 里 **35 个要 bf16、43 个要至少一种今天没建模的 dtype**；补完这一步，
     功能需求 5 的可做面从 31 个（只要 f32）扩到全部 83 个。
   - A5 的 legacy 套件里 **40 / 169 个用例目录名字带 bf16**，它们从「一张图都编不出」
     变成「可以开始接」。

## 功能需求 4：VectorFunc——A5 的二次融合，当前零覆盖

1. 介绍

   **这是本设计新识别出的最大结构性缺口**，四份上游文档都没有涉及。

   A5 的调度在 buffer 分配阶段会做一次**二次融合**：把融合子图里支持 reg-base 的相邻节点
   再聚成一个小子图，替换成一个 `VectorFunc` 节点，节点体是一张**嵌套的 AscGraph**
   （自带 Data / Load / Store / Output），codegen 为它发一个独立的
   `inline __simd_vf__ void <name>(__local_mem__ T *...)` 函数，函数内按 `GetVecLen()`
   做逐元素循环，kernel 体只调用它。

   落点：

   - 分区器 `autofuse/v35/optimize/partition/vector_func_partitioner.cpp`（1711 行），
     由 `PlatformV2::PartitionSubFunctions` 驱动，调用点在
     `autofuse/optimize/buffer_allocate/buf_que_allocator.cpp:912`——**对每张 impl graph 无条件调用**；
   - 发射 `autofuse/v35/codegen/vec_func_call/vec_func_call.cpp`（734 行），
     产物带 `#if defined(__DAV_C310__) || ...` 的架构门；
   - 节点类型在 `ascir_builtin_ops_v2.cpp` 注册为 `VectorFunc`。

   **触发条件（读码确认）**：相邻且全部 VF-capable 的节点数 ≥ `kMinVfNodesNum = 2`
   （`vector_func_partitioner.cpp:31`；`Arange` 单点有例外）。VF-capable 由
   `AscIrCodegen::IsVectorFunctionSupported` 决定，基类缺省 `false`
   （`ascir_registry.h:145`），v2 里 22 个 impl 覆写了它，其中**无条件 true 的恰好是最常见的
   那批**：`Abs` `Exp` `Ln` `Sqrt` `Rsqrt` `Neg` `Relu` `Add` `Sub` `Mul` `Div` `TrueDiv`
   `Maximum` `Minimum` `Where` `LeakyRelu`，另有 `Cast` / `Compare` / `Broadcast` /
   `Arange` / `BitwiseNot` / `IndirectLoad` 带条件。

   **于是今天的实际后果是**：A5 上凡是两个以上相邻常见 elementwise 节点的图，调度产物里
   就有 `VectorFunc`；Bridge 不认识这个类型，在 `ScheduleToAfir.cpp:1087` 报
   `UnsupportedOperation` 并整图回落 legacy。也就是说——**A5 上我们已经接通的那 43 个 op
   所组成的多节点图，基本都没有走 MLIR**，而单算子图才走。

   一个重要的正面结论：**34 个 SIMT 超越函数不参与 VF**（除 `Rsqrt` 外均未覆写
   `IsVectorFunctionSupported`，缺省 false），所以功能需求 5 的收益**不会被 VF 吃掉**。
   两件事正交。

2. 输入

   带 `VectorFunc` 节点的 `FusedScheduledResult`：节点自身的输入 / 输出锚点、以及挂在它
   下面的那张子图（子图内每个节点都有自己的 `sched` 与 view 属性）。

3. 处理

   本阶段**只做评估与度量，不做实现**。实现设计见
   `docs/development/2026-10-18-afir-vector-func-implementation-design.md`（2026-10-18 已出）。
   那份文档把这里对规模的估计**往下修了一档**：寄存器级的算子、类型、发射与 `vec_scope` /
   `vf_for` 在 vendored 的 PyAsc（`api/ascendc`）里**已经有了**，tile 级 → 寄存器级的下降在
   上游 `experimental/asctile` 的 `AscVF` 里也有，所以新增的是「一个 AFIR region 算子 + 四个
   物化 pass 的边界 + 一次机械替换 + 一套移植与能力判定」，不是从零建一层。

   本阶段必须产出的两件事：

   - **度量（已完成，2026-10-17，结论见本节第 4 条）**：原计划是「用严格模式数出因
     `VectorFunc` 回落的用例数」，**实测这个口径答不了这个问题**——严格模式只报第一道拦下
     图的闸门，含 VF 但更早被别的缺口挡住的图里 VF 一个字都不出现。所以改成数调度产物本身：
     把 MLIR 关掉让 200 格全走 legacy，数每格生成出来的 `__simd_vf__` 函数
     （`test/probe/vf_node_census.sh`）。探针侧按原计划加了 `vf=<a>/<b>` 一列。
   - **两条候选路线的边界**，供实现阶段选型：

     | 路线 | 做法 | 代价 | 风险 |
     |---|---|---|---|
     | A　展平 | Bridge 把 VF 子图展平成普通 afir 节点，逐节点发 reg API | 不新增 IR 结构；复用全部已有 implementation | **放弃 VF 的收益**（中间结果留在寄存器而不是 UB），与 legacy 产物不逐字一致，性能可能回退；UB 占用变大，可能撞 tiling |
     | B　建模 | 新建带 region 的 `afir.vector_func`，下降成 `__simd_vf__` 函数 | 新 op + 嵌套 bridge + 新发射器 + tiling 对接 | 工作量大；`__DAV_C310__` 架构门与 CPU 模拟器的关系要先验 |

     **初步倾向 B**，理由是 A 改变了性能特征却仍然答不出「与 legacy 等价」——而本设计的
     验收标准是「legacy 跑得通的我们也跑得通」，一旦 A 让某些图因 UB 变大而 tiling 失败，
     它连功能目标都保不住。**但在度量数出来之前不定选型**：如果 VF 只命中极少数图，
     A 作为过渡是可接受的。

     **2026-10-18 已定选型：B。** 度量给出 55% 的命中面，A 的那个前提（只命中极少数图）
     不成立；而且实现设计读码后发现 A 不止是性能问题——**VF 的中间结果没有任何 UB 分配**，
     展平后每个中间结果都要一块 UB，而 UB 预算由 legacy 的 buffer 分配与 ATT 定、MLIR 只读
     `tiling.q*_size`，所以 A 要么溢出 UB 要么得重做分配，后者越过「Schedule 一处不动」。
     理由与 B 的两种发射形态（就地 `__VEC_SCOPE__` / 外联 `__simd_vf__`）见实现设计。

4. 输出（**度量已完成，2026-10-17**）

   两台量具，口径互补，都不断言任何事：

   - `test/probe/vf_node_census.sh`：把 MLIR 关掉、让 reg 侧 200 格全走 legacy，数每一格
     生成出来的 `__simd_vf__` 函数。它答的是「**调度到底融不融 VF**」，与 MLIR 这一侧的
     任何闸门无关——这正是要的性质。产物
     `docs/development/2026-10-17-vf-node-census-reg.txt`。
   - `test/probe/af-capability-probe` 新增一列 `vf=<带 VF 的 impl graph 数>/<impl graph 总数>`，
     用在级联切片那 112 格上。

   **reg 侧 200 格里 110 格（55%）的调度产物带 `VectorFunc`**，而接管表上只有 17 格把 VF
   记成首因。与接管表交叉（量具自带 `-t` 那张交叉表）：

   | 接管表状态 | 含 VF | 不含 VF |
   |---|---|---|
   | `takeover` | 0 | 3 |
   | `fallback`，首因就是 VF | 17 | — |
   | `fallback`，首因是别的 | 34 | 65 |
   | `hard-failure` | 58 | 21 |
   | `out-of-scope`（Cube） | 1 | 1 |
   | 小计 | **110** | 90 |

   `takeover` 那一栏的 0 / 3 是个自洽性检查：bridge 见到 `VectorFunc` 必拒，所以接管的格子
   不可能含 VF。量到非 0 说明两张表不是同一轮量的。

   **两个候选原因里 (a) 被推翻**。不是「这 200 个用例多是单算子图、够不上
   `kMinVfNodesNum = 2`」——110 / 200 过了那道门槛。成立的是 (b)：VF 在很大一片上是必要条件，
   但它上游还压着别的闸门，于是接管表把那些格记在别的原因上。

   **于是 17 这个数的方向要反过来读。** 它不是「VF 挡住多少」的下界，而是「**只差 VF**」的
   格数，也就是单独接 VF 当期能解锁的**上界**（这 17 格后面还可能露出下一个缺口）。
   「VF 是必要条件」的格数是 110。两个数要分开用：

   - **排期看 110**：不接 VF 这 55% 永远进不来，补 op、补 dtype 一格也换不回来；
   - **兑现看 17**：单独接 VF 当期只解锁 8.5%。

   与另外几项的重叠（同一张表交叉出来，按闸门分）：

   | 闸门 | 含 VF | 不含 VF |
   |---|---|---|
   | `candidate-preparation`（即 F-4） | 48 | 7 |
   | `lowering`（节点类型 / dtype / 没有实现） | 61 | 79 |
   | `none`（已接管） | 0 | 3 |
   | `cube-fusion` | 1 | 1 |

   **48 格要 F-4 与 VF 两件都修才动**，单修 F-4 最多解锁 7 格。这 48 格里 **44 格是
   `indirect_load_*` 那一族**（`IndirectLoad` 是条件 VF-capable 的那几个之一），余下 4 格是
   `binary_api_scalar` / `floordiv_mul_le_select` / `load_compare_cast_sum_store` /
   `ub_scalar_brc_abs_add`——也就是说 F-4 与 VF 的交集**不是普通 elementwise，而是 IndirectLoad
   这一族**，而那一族本身还缺 `IndirectLoad` 的 bridge 建模（接管表上另有 8 格以它为首因）。
   这就是功能 4 与功能 5 先后的答案：**功能 5（补 op）的收益落在不含 VF 的那 90 格里，与 VF 正交，可以先做**；VF 自己
   不解锁大头，除非 F-4 一起解。

   选型（路线 A / B）的两条新证据。**本阶段不定选型；选型已在实现设计里定为 B**
   （`2026-10-18-afir-vector-func-implementation-design.md`，那里还多了一条否决 A 的硬理由：
   VF 的中间结果没有 UB 分配，展平要重做分配与 tiling）：

   - 命中面 55%，不是「极少数图」。按上面第 3 条自己的判据（「如果 VF 只命中极少数图，
     A 作为过渡是可接受的」），**A 的那个前提不成立**。
   - 路线 B 那条列为待验的风险（`__DAV_C310__` 架构门与 CPU 模拟器的关系）**实测不成立**：
     reg 侧 CPU E2E 目标编译时就带 `-D__DAV_C310__ -D__NPU_ARCH__=3510`（见
     `backend_e2e_v2/<case>/CMakeFiles/<case>_e2e_v2.dir/flags.make`），而一张含 VF 的
     legacy kernel（`add_abs_test_e2e_v2`，三个形状）在 CPU 模拟器上数值全对。
     也就是说 `__simd_vf__` 这个形态在既有 CPU E2E 框架里验得出来，不必上板。

   级联切片那 112 格的 VF 标签（`--reg-base --topo`）：

   | 格 | 结论 |
   |---|---|
   | T5 / T5c / T5d / T5e 各 10 格，共 40 格 | 全部 `vf=a/a`（没有不带 VF 的候选模板），拒收原因就是 `VectorFunc`——菱形那三个算子（`abs` / `neg` / `add`）够门槛 |
   | T1 / T4 / T7 共 10 格 | `vf=0`，拒收原因是 `afir.nddma`——**VF 不在它们更上游，那条原因是真的** |
   | 4 格（T3 在 `[8,1] -> [8,128]` 等） | `vf>0` 但判决仍是 ok：有候选模板带 VF，而 plan selection 选中了不带 VF 的那个 |
   | 其余 58 格 | `vf=0` 且 ok |

   mem 侧同样 112 格全部 `vf=0/b`，与「V1 不产生 `VectorFunc`」一致——那是个自带的对照组。

   **这改写了级联切片的读法**：T5c / T5d / T5e 三个对照「十格全过」是**mem 侧**的事实
   （见功能需求 6 与探针里那段注释），reg 侧它们全被 VF 拦下。所以「菱形本身没问题、判据在
   分叉点是不是单输入单输出」那条结论只在 mem 侧成立；reg 侧今天量不到菱形的判据，
   因为 VF 在它之前就拦了。

## 功能需求 5：算子补齐（A5 差 83 个）

1. 介绍

   把 legacy 两份 `REG_ASC_IR` 注册表里 `ComputeType::kComputeElewise` 的算子与 bridge
   白名单那 47 个（`BridgeOperationRegistryTest.cpp:79`）做差，得到的就是算子轴的全部差距。

2. 输入

   `autofuse/ascir/generator/ascir_builtin_ops_v1.cpp`（mem 侧，48 个 elementwise）与
   `autofuse/v35/ascir/generator/ascir_builtin_ops_v2.cpp`（reg 侧另外声明 86 个）。

3. 处理

   **A2 / A3：48 个里覆盖 43，差 5**，且 5 个都不是简单 vector op——

   | op | 为什么不补 |
   |---|---|
   | `MaskedFill` | 已故意删除：`MaskedFillInputReorderPass` 在调度前就把它改写成 `Select`，bridge 看不到这个类型 |
   | `Pad` | 布局变换族（与 `RemovePad` 同族），不属 elementwise，另线 |
   | `FlashSoftmax` | 复合算子，含 reduce 语义 |
   | `Nop` | 空操作 |
   | `VectorFunc` | 见功能需求 4；V1 平台不产生它 |

   **结论：A2/A3 的算子面已经齐了。那一侧剩下的全是 dtype、形状与拓扑，不是接算子。**

   **A5：差 83 个**，按形态分解（机读统计，2026-10-16）：

   | 切法 | 分布 |
   |---|---|
   | 入 / 出 | 1入1出 **58**、2入1出 **21**、3入1出 1（`Fma`）、0入1出 1（`Arange`）、1入2出 1（`Frexp`）、特殊 1（`VectorFunc`） |
   | 机制 | 走 `SimtFloatUnaryAscIrCodegenImplV2` **34**；其余 48 走通用 `AscIrCodegenV2` |
   | dtype | 只要 f32 **31**；要 bf16 35；要至少一种未建模 dtype 43 |

   **34 个 SIMT 一元算子是「没有新机制」那一类**：1 入 1 出、`ComputeType::kComputeElewise`、
   **不要 tmp buffer**（`CalcTmpBufSize` 返回 `CalcVoidTmpSizeV2`），复杂度全在 AscendC 头里
   （`bessel_j0.h` 170 行、`zeta.h` 645 行），MLIR 侧要发的只是一条 `XxxExtend` 调用加
   2~3 个前导头。而且这条机制**已经验证过**——`Erf` / `Tanh` / `Rsqrt` 就在同一个基类上，
   三个都已接通。单个成本是 L-19 实测的量级：产品代码 40~130 行，测试约 330 行。

   排期建议：31 个 f32-only 的可以在 dtype 之前就接（不碰公共机制）；其余在功能需求 3
   落地后接。三个形态特殊的单独裁决——

   - `Frexp`（1 入 **2 出**）：AFIR 的 destination-style 今天一个 result，要多输出支持。**本阶段不做**，入册。
   - `Arange`（0 入 1 出）：无输入的生成类，没有源 tile；且它与 VF 的 `Arange` 单点例外纠缠。**本阶段不做**，入册。
   - `VectorFunc`：见功能需求 4。

4. 输出

   bridge 白名单从 47 增长到 47 + N，每一次增长都要同步改
   `BridgeOperationRegistryTest` 里写死的计数与名字列表——**那两处是故意要求手改的**
   （增长必须是一次经过评审的改动），保留。

## 功能需求 6：形状 / 布局的剩余边界

1. 介绍

   轴 4 的三条，都已经有最小复现，不需要再探。

2. 输入 / 处理 / 输出（逐条）

   **F-1　一处能力 / pattern 漂移（优先，它是被破的不变量）**

   - 现象：4 次 `DRIFT` 全部命中 `[8,1] → [8,128]`（形态 B、尾轴对齐），f32 / f16 各两次：
     能力层答了支持，pattern 才失败。仓库的约定是拒收只能拒在 capability 层。
   - 落点：elementwise / `Cast` 两处 pattern 怎么看待「源是两段窗口分解出来的片上张量」。
   - 判据：`af-capability-probe` 两侧 `DRIFT` 与 `kFailed` 两桶恒为 0。

   *进展（2026-10-26）*：**不复现，本项划掉。** `DRIFT` 桶 0（`failed to legalize` 零命中），
   而那一桶在探针里仍然实现着（`CapabilityBoundaryProbe.cpp:214`），所以 0 是真的 0 而不是
   探不到；文档点名的那四格（`[8,1] → [8,128]` 的 f32 / f16）在 T0~T7 每个场景下全是 `ok`。
   具体是哪次改动收掉的没有逐一归因——`9453c9eda` 修过一处 `failed to legalize`，但它自述的
   形态是「动态形状的一维图」，与这四格不是同一个，要定位得 bisect，不值当。

   **判据的另一半（`kFailed` 恒为 0）没达成，而且它与 F-1 无关**：实测 126 格里
   **120 格是 `afir.broadcast` 的 ODS 类型域不含 BF16**（`Broadcast.td:23` 的
   `AFIR_BroadcastElementTypes` 少 BF16 一位），5 格是 `afir.cast` 不支持 bf16 → f16，
   其余 1 格。这是 dtype 轴（功能需求 3）的事：bf16 开闸只开了 elementwise 那一侧。
   这条判据把 F-1 与整个 dtype 轴混成一个数，应当拆开——F-1 只看 `DRIFT`。

   **那 120 格不要直接往 defvar 里加一位**：ODS 的类型域按约定是 conversion 两侧「供不供得
   起」的并集，要先核 broadcast 的 implementation 在 bf16 上到底供不供。

   *进展（2026-10-26）*：**核过了，供，已开位；120 格硬失败清零。** 逐档核头文件（**以头
   文件为准，不以 legacy 注册表为准**），这个 op 的四个 implementation 里三档供得起 bf16：

   | implementation | 发的调用 | bf16 | 依据 |
   |---|---|---|---|
   | `broadcast.tensor.extend_reg_base` | `BroadcastExtend` | **供** | `v35/ascendc/api_regbase/broadcast_extend_impl.h:260` 的 `static_assert(SupportBytes<T, 1, 2, 4, 8>())` 只看字节宽 |
   | `broadcast.tensor.duplicate` / `_reg_base` | `AscendC::Duplicate` | **供** | `dav_c220` 与 `dav_3510` 两份 `kernel_operator_vec_duplicate_impl.h` 的 `SupportType` 都列着 `bfloat16_t` |
   | `broadcast.tensor.segmented` | `broadcast.h` 的包装 `Broadcast` | **不供** | `ascendc/api/broadcast.h:511` 的分发链三支都不收 bf16，落在 `else` 的 `ASSERT(false)` 上 |

   所以 ODS 的 `AFIR_BroadcastElementTypes` 与 `kExtendElementTypes` 各开一位；mem 分段那一
   档不开，**它的 bf16 广播于是在能力层答不支持、整图干净回落**，而不是失败在 verifier 上。
   常量源那两档本次也不开——那一档收哪几种的判据是「legacy 那句不带限定的 `Duplicate` 绑到
   谁」（`ui8` 已经因此被收窄一次，见 `kKnownNarrower`），要单独核一遍，**入册**。

   **要订正这张表原先的两处说法。** 第一，`Broadcast.td` 上「没有 BF16，而且不是漏的」那句
   话读的是 `REG_ASC_IR` 的**声明**，不是 AscendC 头；两者在这个 op 上不一致。第二，门禁
   `ElementTypeMatrixTest` 的判据二（**ODS ⊆ legacy 两列的并集**）隐含「legacy 的声明 = 头
   文件供得起的集合」这个前提，这个 op 上不成立。判据二因此多了一张例外快照
   `kOdsWiderThanLegacy`：钉死多出来的那几位，**并且多查一条——钉在上面的每一位必须真有
   implementation 注册着它**。判据二要挡的「ODS 收、没人供、白回落」于是仍然挡着，两条都
   做过变异验证（删掉快照行 / 撤掉注册位，各自按预期变红）。

   *实测（探针，2026-10-26，本地两侧各一遍）*：

   | 侧 | FAILED | ok | unsupported | DRIFT |
   |---|---|---|---|---|
   | mem（Ascend910B1） | 126 → **5** | 358 → 358 | 191 → 312 | 0 → 0 |
   | reg（Ascend950PR_9599） | 118 → **75** | 200 → **210** | 355 → 388 | 0 → 0 |

   mem 侧剩的 5 格全是 `afir.cast` 不支持 bf16 → f16，另一条判据。reg 侧剩的 70 格**是另一
   个缺口**：全是 bf16、**一条诊断都没有**，而且改动前后都是 70 格——与 ODS 这一条无关，
   形态上是 `candidate-preparation` 那一类（功能需求 6 的 F-4 同族）。reg 侧新通的 6 格广播
   都是 `inner=normal`，`inner=brc` 那一簇仍在那 70 格里。

   **收益只在探针这一侧，接管表上是 0 格——已实测，不是推断。** 两张 2026-10-26 的接管表里
   带这条诊断的行本来就是 0（`grep 'afir.broadcast'` 两张表各 0 命中），而这一条改动真正要
   验的是另一面：**放开一位之后，图会建到更深的一层，可能撞上别的硬失败**（VF 那道闸门放开
   时 reg 侧硬失败实测从 79 涨到 89，见功能需求 4）。所以接管表改动前后都要量。

   reg 侧全量 200 格实测（2026-10-26，远端 AF_MLIR_0814，`-j 40`，在
   `/mnt/workspace/agent/claude/brc-bf16-wt` 这个隔离 worktree 里量，基准取 **6a525700**——
   与 `2026-10-26-legacy-takeover-table-reg-full200.txt` 同一个基准，所以只差这一个改动，
   不必重跑基线）：

   | | takeover | fallback | hard-failure | out-of-scope |
   |---|---|---|---|---|
   | 基准 6a525700 | 63 | 69 | 66 | 2 |
   | + 本改动 | 63 | 69 | 66 | 2 |

   gate 分布同样逐项相同（`candidate-preparation` 55、`cube-fusion` 2、`lowering` 80、
   `none` 63）。**逐行对比 200 格，连首因诊断列都完全一致**（只差行尾空白）。

   mem 侧 38 格也量了，但**不拿已发布的那张 mem 表对**——
   `2026-10-26-legacy-takeover-table-mem-f4.txt` 的基准是 `cc7d56b2`（F-4 之后），与这个
   worktree 的 `6a525700` 不是同一个。改成在同一个 worktree 里前后各量一遍，两遍之间只差
   这一个补丁：

   | | takeover | fallback | hard-failure | no-hook-call | out-of-scope |
   |---|---|---|---|---|---|
   | 基准 6a525700 | 13 | 1 | 20 | 3 | 1 |
   | + 本改动 | 13 | 1 | 20 | 3 | 1 |

   同样 38 格逐行一致（含首因诊断列）。顺带是一次口径交叉验证：这五个数与那张 mem 表上
   **「F-4 前」那一列逐个相等**，也就确认了 `6a525700` 正是 F-4 之前的那个基准，量具这一侧
   没有接错。

   所以这一条既没有收益也没有代价，也没有新的输入图可以搬进 `LegacyGraphGuardSt`。
   数值看护另立一个用例：`test/e2e/cpu/cases/load_broadcast_bf16_reg_base_store`，reg 平台、
   两个形状（尾轴 256B 对齐与 10B 带填充），按位比对、输出填哨兵，生成码里钉
   `BroadcastExtend<bfloat16_t,`。

   **F-2　尾轴不对齐（mem 侧最大单一卡点）**

   - 现象：尾轴字节数不足 32B 的广播形状报不支持（f16 的 `[1,8] → [128,8]` 尾轴 16B、
     `[1,5] → [33,5]` 尾轴 20B）。
   - 处理：要换 API 形态，不是调判据。放开判据不会多编出一张图（C-2 那一轮实测过）。
   - 注意：`Broadcast` 尾轴 32B 对齐是**正确性不变量**，违反则静默算错。
     放开这道闸之后必须验数值，且输出缓冲填哨兵而不是填 0。

   **F-3　最内轴是广播轴（对齐形状上唯一的卡点）**

   - 现象：28 / 28 全部报不支持。
   - 与 F-2 的关系：F-2 收掉之后 F-3 的剖面会变，**先做 F-2 再重量 F-3**。

   *进展（2026-10-26）*：**F-2 与 F-3 都要订正，而且它们不是两条，是同一条。** 重量了 mem
   侧广播 sweep（4 个非 bf16 dtype，475 格里的那 380 格），按真判据切——判据是**行距是否等于
   行长**（即尾轴字节数是不是 32B 的整数倍），不是文档原先写的「尾轴不足 32B」：

   | | 最内是广播轴 | 最内不是广播轴 |
   |---|---|---|
   | 行距 ≠ 行长 | 119 unsup / 17 ok | **164 ok / 0 unsup** |
   | 行距 == 行长 | 28 unsup / 52 ok | **50 ok / 0 unsup** |

   **形态 A（最内不是广播轴）214 格全通，不分对不对齐**——`afir-broadcast-tail-pitch.mlir`
   那一轮把行距收进来之后，「尾轴不对齐」作为独立卡点**已经不存在了**。「28 / 28 全部报不
   支持」同样不成立：最内是广播轴且行距等于行长的格子有 52 格是 ok 的。

   147 格 unsupported 按首因逐格归因（临时给 `classifySegments` 九个出口与能力层十五个出口
   各插一个计数点，出口和守恒：`S0_entry 740 = S1_constant_source 222 + E0_entry 518`，
   `E0_entry 518 = E8 42 + 支持 476`，`Q14 264 = S1 222 + E8 42`；`S2_collapse_failed` 与
   `Q09`/`Q10` 一次都没走到），再把 E8 那条打成带 `error:` 前缀的首因，于是逐格可读：

   | 簇 | 格数 | 真判据 | 归属 |
   |---|---|---|---|
   | **E8**：`行距 ≠ 行长 && 最内段是广播段` | **30** | `classifySegments` 里 `if (shape.broadcastIsInner) return failure();` | **F-2 与 F-3 唯一的真卡点** |
   | ui8 卡 `afir.broadcast` | 51 | `kDuplicateElementTypes` 不含 ui8 | 已知 dtype 窄差（`kKnownNarrower` 快照），不是形状 |
   | 卡 `afir.load` | 66 | Load 侧下降（源尾轴为 1 的 tile） | **新缺口**，不属 F-2 / F-3 |

   30 + 51 + 66 = 147，三簇互不重叠。所以这条线上**要做的只有 E8 那一支 30 格**，另外 117 格
   动广播一个字都不会变。

   *落点（2026-10-26，待实施）*：形态 A 带填充的做法是**把 `dst_k` 换成行距**——包装的
   `BroadcastFirstDim` 里 `dst_k` 身兼行长与行距两职，所以一个数换掉就对了
   （`SegmentedBroadcastEmitter` 里 `if (shape_.pitch) inner = *pitch;` 那一段）。形态 B 走的
   是 `BroadcastCommon` 的另一支（`ascendc/api/broadcast.h:304`），那里 `dst_k` 不兼这个职，
   同一招不成立——这就是那段注释说的「两侧要对上的是另一对数」。

   但那一支里**行距恒是一整档 32B**（填充的定义），所以
   `AscendC::Broadcast<T, 2, 1>(dst, src, {dst_m, pitch}, {src_m, 1})` 的对齐前提天然满足，而
   包装对两侧 shape 的要求只落在非广播那一位（`m`）上，换 `pitch` 不碰它。**所以这一支很可能
   不需要新 API，只需要把 pitch 也喂给形态 B**——`emitWindowCall` 换掉 `inner` 的那段代码形态
   上已经覆盖得到。要订正文档原先「要换 API 形态而不是调判据」那句话的适用范围：它对的是
   **旧那条稠密判据**（C-2 那轮实测放开它不会多编出一张图），而 E8 不是那条，E8 拦的是
   「pitch 算得出来但没人验过形态 B」。

   **下一步不是直接放开 E8**：按那条正确性不变量，要先在 CPU 模拟器上拿一个形态 B 带填充的
   形状验数值（输出填哨兵不填 0），确认 `AscendC::Broadcast<T,2,1>` 在 `dst_k = pitch` 下写
   出来的布局与 `afir.store` 按行距读的那一侧对得上；对不上才谈换 API。

   *进展（2026-10-26）*：**验过了，对得上，E8 已放开；不需要新 API。**

   落点比预想的小：形态 A 要「行距两边一起换」，是因为 `emitWindowCall` 的 `sourceK` 与
   `destinationK` 发的是同一个 `inner`；而形态 B 的 `sourceK` 发的是**字面量 `one_`**，
   所以换 `inner` 只动得了目的地那一边——也只需要动那一边。发出去的是

       Broadcast(dst, src, M, 1, 0, M, pitch, 0, tmp)   →   AscendC::Broadcast<T, 2, 1>

   包装那一支（`ascendc/api/broadcast.h:304` 的 `src_k == 1 && src_m == dst_m`）对两侧 shape
   的要求只落在非广播那一位（`m`）上，换 `dst_k` 不碰它。所以代码改动就是去掉
   `classifySegments` 里那个早退，`emitWindowCall` 一行没动。

   **数值证据**（CPU 模拟器，哨兵填充、逐元素比对）：`load_broadcast_segmented_store` 新增
   两格，`_33x1`（`[33,1] → [33,5]`，尾轴 20B、行距 32B）与 `_2x3x5x7x1`
   （`[2,3,5,7,1] → [2,3,5,7,11]`，尾轴 44B、行距 48B，广播段是最内段而外面还有三段普通段，
   覆盖「带填充的窗口调用嵌在多层循环里」）。两格都过，生成码里 `dst_k` 发的确实是行距
   （`_33x1` 发的是 `c8_idx`，行长是 5）。

   **探针**（mem 侧，改动前后）：`ok` 358 → **400**（+42，正好等于 E8 的命中数）、
   `unsupported` 312 → 269、硬失败 5 → 5 **不变**；逐格迁移只有一个方向，**42 格
   `unsupported` → `ok`，0 格变差**。reg 侧三个数一个没动（它走 `Extend` 档，不经过这条判据）。

   **接管表**（远端 AF_MLIR_0814，`-j 40`，同一个隔离 worktree，对照是贴 E8 之前那一遍）：
   mem 侧 38 格**逐行完全一致，含首因诊断列**——既没有一格变差（放开闸门最该防的就是这个，
   VF 那轮实测硬失败从 79 涨到 89），也没有一格翻成接管。reg 侧不必量：探针三个数一个没动，
   这条判据不在 `Extend` 档的路上。所以这一项的收益同样只在探针那一侧。

   **用例搬动**：`afir-broadcast-unsupported.mlir` 的第二片（`broadcast_padded_tail_inner_is_broadcast`）
   是为这条判据写的专属负例，放开后当场红——按设计。它搬成正例
   `afir-broadcast-tail-pitch-inner-brc.mlir`，与形态 A 那一格配成一对，两格的区别就是**换哪
   一边**；原文件 `CHECK-COUNT-3` 改成 `CHECK-COUNT-2`。那一片当初的理由「两侧要对上的是另
   一对数，不是发同一个行距能解决的」要订正：前半句对（确实不是发同一个行距），后半句不必要
   ——只换目的地一边就对。变异验证两条，都实测过：加回早退 → SELECT / IR / CPP 三条全红；
   去掉 pitch 替换 → IR / CPP 红而 **SELECT 仍绿**（能力层查不出发错的数）。

   **一条要记下的敞口**：这一支靠「行距恒是 32B 的整数倍」这条调度侧不变量，
   `AscendC::Broadcast<T,2,1>` 的对齐要求才天然满足，而 **codegen 这一侧没有检查它**——与形态
   A 那一支同一个敞口，不是本次新引入的。真被破了是静默算错。要堵它得在能力层加一条「行距可
   被 32B 整除」的判据，而符号行距的整除只能走表达式树的结构判据，范围超出本项，**入册**。

   **剩下的**：这条线上原本 147 格里，E8 那 30 格已收；另外 117 格与广播无关——66 格卡在
   `afir.load`（源尾轴为 1 的 tile，Load 侧下降，**新缺口**），51 格是 ui8 那条已知 dtype 窄差。

   **F-4　plan selection 答不出「不支持」（新识别，2026-10-17，入册不在本阶段修）**

   - 现象：mem 侧 38 个 legacy 用例里 **14 格拦在 `candidate-preparation`**，全部
     `shortfall=no`——也就是硬失败。被拦的是 concat / topn / reduce / int64 比较这些
     **节点类型 MLIR 本来就没建模**的图，它们该走能力不足那一路，而不是坏了那一路。
   - 落点：`bridge/PlanSelector/PlanSelector.cpp:99` 把候选枚举失败一律归成「坏了」。
     结构上的原因是**第一道闸门就是 plan selection**（`PrepareMlirCandidates` 在任何下降
     之前跑），而这道闸门的词汇表里没有「能力不足」。于是仓库那条约定
     （**拒收只能拒在 capability 层**）在非 elementwise 族上整片是破的。
   - 与 F-1 的关系：同一类问题的另一半，但量级不同——F-1 是 4 格 `DRIFT`（能力层说支持而
     pattern 失败），F-4 是 14 格（能力层根本没被问到）。
   - 与功能需求 7 的关系：**它是搬运的前置**。今天把这些用例搬进 MLIR 用例集，它们不会
     「回落后变绿」，而是当场红。
   - 为什么不在本阶段修：修法要动 PlanSelector 的失败分类，范围超出严格模式这一位；而且
     要先判定哪些枚举失败是「这张图我们收不了」、哪些是真的坏了——那是一次独立的归因。
   - 判据（修的时候用）：这 14 格从 `hard-failure` 翻成 `fallback`，`shortfall` 变 `yes`，
     且 `af-capability-probe` 两侧 `kFailed` 桶不变大。

## 功能需求 7：legacy e2e 用例搬运与样板收敛

1. 介绍

   把 legacy 自有的 e2e 用例搬进 MLIR 用例集。注意用户已说明：测试人员仓里还有更多用例，
   **本阶段只搬本仓的**。

2. 输入

   - mem 侧：`autofuse/tests/st/backend_e2e`，44 个用例目录，按名字排掉 cube / reduce /
     concat / transpose / topn / argmax 之后**在范围内约 28 个**；
   - reg 侧：`autofuse/tests/v35/st/backend_e2e_v2`，169 个目录，同样排掉之后**约 155 个**。

   这两个数是按目录名估的，**精确的在范围内集合由功能需求 2 那张表给**（按图里的节点
   类型判定，而不是按名字猜）。

3. 处理

   先收样板再搬，否则搬运成本按 L-19 实测会被放大：

   - **L-19 的实测**：四个「没有新机制」的 op 提交里，产品代码 79~135 行而测试 323~415 行。
     一个 CPU E2E 用例 4 个文件 215 行，其中 `CMakeLists.txt`（14 行）与 `Case.h`（30 行）
     是逐字样板，`Case.cpp` 建图段高度雷同，真正一个 op 独有的只有 `KernelTest.cpp` 的数值参考。
   - **收敛目标**：E2E 样板收成一个 CMake 函数 + 一个头文件模板，新增用例只写建图段与
     数值参考。**原先这里写的是「单用例从 215 行降到 100 行以内」，那个数已订正删除**
     （理由见下面两条进展：建图段与数值参考不是样板，压它们是压错了对象）。

     *进展（2026-10-26）*：**样板已收，但 100 行这个数达不到，要订正。** 实测 73 个用例，
     一个用例 201 行（中位）里 `Case.cpp` 70 + `KernelTest.cpp` 80 是建图段与数值参考，
     不是样板——真正逐字重复的只有 `CMakeLists.txt`（14 → **12**）与 `Case.h`
     （30 → **19**，带自有常量的 38 → 28）。收法：CMake 两个源码列表缺省成用例目录下的
     固定名字；`Case.h` 收成 `AF_MLIR_E2E_CPU_CASE_BEGIN/END` 一对宏（顺带收掉那两个
     **每个用例各抄一份 73 遍**的 stringify 宏）。所以新增用例的实际规模是 180 行上下，
     要再往下只能动 `Case.cpp` / `KernelTest.cpp` 里的理由注释，那与仓里的注释密度要求相悖。
     已改两个用例当样例，其余 71 个未动。

     *进展（2026-10-27）*：**71 个全部改完，这一节闭环。** 74 个用例现在一致：`Case.h` 走
     `AF_MLIR_E2E_CPU_CASE_BEGIN/END`、`CMakeLists.txt` 不再写 `GENERATOR_SOURCES` /
     `TEST_SOURCES`（走 `add_mlir_e2e_cpu_case` 的缺省值）。142 个文件 +145 / -1082 行，
     `Case.h` 合计 2144 -> 1391 行，单用例中位 190 行。
     **转换里值得记一条**：第一版把「样板行」按行正则删，误删了 `load_broadcast_duplicate_store`
     自己那条 `static_assert(kInputShapes.begin()->element_count == 1U, ...)` 的首行——它
     以 `static_assert(kInput` 开头，和要删的那两条同形。改成按**语句**删（只删
     `kInputShapes.size()` / `kOutputShapes.size()` 那两条，吃到配平的 `);`），保留内容的用例
     数从 7 升到 12，多出的 5 个正是被误删的用例自有 `static_assert` 与 `using`。
   - **不做的**：不要为省行数去掉 lit 的两个 `--target` 正例——那一对是两侧「差异」的唯一
     证据，E5 / E2 两轮都靠它抓到过写反的家族。

4. 输出

   搬运后的用例进 `check-e2e-cpu`，并在严格模式下复跑一遍（否则绿色可能来自回落）。

   *进展（2026-10-26）*：**已落地，但形态与本节原先设想的不同，要订正两处。**

   第一，**搬的是输入图与形状，不是用例本体**。legacy 用例给的是**调度前**的 ComputeGraph
   （`ShareGraph::Xxx` 工厂 → 内部 `CreateXxxAscGraph` + `ConstructVVAscGraphAxisInfo` 设符号
   轴 s0..sN），由 Optimizer 去调度；而 `check-e2e-cpu` 要的是**调度后**、具体形状的 impl
   graph。两个 harness 不在同一层，原样搬等于逐个重写 + 手写数值参考（按 L-19 的量级每格约
   330 行）。实际做法是把输入图译成 `AscGraphBuilder` 链，落在**我们 ST 档**
   （`test/st/bridge/LegacyGraphGuardSt.cpp`）：真 Optimizer 调度 → `GenerateForInductor`。

   第二，**断言面只有「能生成 kernel 代码，且必须是 MLIR 生成的」**，不含数值比对（符号形状
   下也算不了）。严格模式下回落算失败，所以某个 op 的下降没了、图悄悄落回 legacy 时会红。

   范围与剩余：legacy 两套 e2e 去重后 **183 个输入图**（分母是 `ShareGraph::Xxx` 工厂，不是
   用例目录——按目录名去重是 204），按两张接管表的 takeover 筛出 **74 个**，已搬
   **73 个 / 76 条用例**（`LoadWhereReduceStore` 未搬，理由写在该文件里）。**其余 109 个
   图 MLIR 今天生成不出 kernel**，就是缺口清单本身——**每补上一个缺口，就能把对应的图以约
   10 行的成本搬进来**，不必再逐个读 legacy 的建图代码。

   *重筛（2026-10-26，接管表走到 reg 67 之后）*：**74 -> 78，剩余 109 -> 105**；按节点类型
   归类也补上了（原文这一项记的是「尚未做」）——105 里**在范围内的只有 82**，另 22 个是
   reduce / matmul / gather / transpose / concat / indirect_load / rand 族，1 个归不出类。
   *定界（2026-10-27）*：**legacy 自己不跑的用例不算缺口。** 183 里有 26 个工厂只被已禁用的
   用例引用（`backend_e2e_v2/CMakeLists.txt` 第 4 行「要修复的目标列表」那批 + 5 个没有注册
   行的），它们没有可运行的 legacy 用例可对照，补完也验不了。去掉后分母 **157**、剩余
   **79**、其中在范围内 **60**。这 26 个是补量过一遍才划出界的（1 takeover / 3 fallback /
   4 hard-failure / 18 generator 编不出来），不是直接划掉。

   筛法、逐个清单与两个口径的对照见 `2026-10-26-feature7-remaining-graphs-rescreen.txt`。
   原文「对应 fallback 132 格 + 硬失败 17 格」数的是**格**不是图，两者不是一个单位，已不再引用。

## 非功能需求

### 可维护性

本阶段有两笔样板债在关键路径上，都已在上面展开：**L-18**（dtype 真源不可机读，写 3~4 遍）
在功能需求 3 第一步解，**L-19**（测试样板占新增 op 成本的七成）在功能需求 7 解。
另有 L-20（三处中文注释段落未逐段重写）与 L-5（常量 block 的预留在 ATT、分配在 MLIR，
今天不会错但失效形态是运行期 UB 越界且无用例变红）维持现状，不在本阶段动。

### 可测试性

- 入口：`bash ./mlir-test.sh`（固定跑 `-u -s -m mlir -j 6`，一次只起一个）、
  `check-e2e-cpu-<用例名>`、`run_npu_e2e.sh`、`af-capability-probe`（量具，不挂 CI）。
- 新增的严格模式是本阶段最重要的测试能力：它让「接管」可断言。
- 覆盖目标：每一条功能需求都要有一个**会变红的东西**，不接受「读过代码觉得对」。

### 可移植性

两档平台两套 AscendC API。新增 dtype / op 必须同时说明两侧各收什么，不一致是常态
（`floor_div` 两侧集合就不同），**不得为了一致把某一侧悄悄放宽**。

### 可靠性

- 严格模式只影响测试路径，缺省行为不变。
- `kFailed` 缺省语义保留：新增一条没打标记的失败路径只会多报一次失败，不会把真 bug 藏起来。
- N1 涉及数值正确性（静默算错），必须有多分块形状的数值用例，不能只看编译通过：
  `[1024,1] → [1024,128]` 那一格是唯一测得出来的（见功能需求 1）。

### 特性交叉影响

按 `docs/guidelines/cross_feature_check.md` 逐项：

| 场景 | 适用性 | 分析说明 |
|------|--------|----------|
| SuperKernel Python 接口 | 不适用 | 本设计不碰 `super_kernel/`，不改 Python 包与 wheel 内容 |
| SuperKernel C++/AOT 接口 | 不适用 | 不改 `libascendsk.so` 的任何导出符号与结构体 |
| Autofuse 图优化 | 不适用 | N1 原列在这里（`nddma_template.cpp`），实测那是一条有意的约定、改了会把 legacy 的 nddma 搬错地址；最终一处未动，legacy 产物逐字不变 |
| Autofuse Codegen/Backend | **适用** | 主体全在这里：dtype 供给表、新 op 的 implementation、形状判据、严格模式分类 |
| AscendC API / Runtime 交互 | **适用** | 新增 op 会发新的 `*Extend` 调用与前导头；前导头清单必须编过（不是读过），`ApiExtensionHeader` 写的是**生成名**（如 `x_reg_base.h`）而非 `x.h` |
| Python/C++ 混合绑定 | 不适用 | 不改 `autofuse/compiler/py_module/` |
| 构建与打包 | **部分适用** | 只增 CMake 用例目标与一个样板函数，不改安装路径与 run 包内容。注意 `build.sh --pkg` 会弄坏测试配置（之后 `mlir-test.sh` 报 no rule for check-ut），删 `build/CMakeCache.txt` 复原 |
| 测试与覆盖率 | **适用** | 新增 UT / lit / CPU E2E，并把 legacy e2e 搬过来；严格模式本身要有正负用例 |
| 性能与日志 | **适用** | 见「性能」一节；不新增默认开启的日志 |
| 兼容性 | **部分适用** | `AF_ENABLE_MLIR_CODEGEN` 增一档取值，向后兼容（旧取值语义不变）；无产物与安装路径变化 |

## 性能

### 编译时长

- 新增 dtype 位与新增 op 的注册都是查表，`O(1)`，不引入新的图遍历。
- 功能需求 2 的严格模式只改分类返回值，不增加遍历。
- 功能需求 5 若按 34 个 SIMT op 全量接入，`ImplementationRegistry` 的表变大约 70%，
  注册是启动期一次性行为，查询按 `OperationName` 哈希，影响可忽略。**但要实测一次**，
  不要推。

### 执行性能

- N1 不改 kernel 结构，只修正 Store 的偏移来源，无性能影响；修完后 reg 侧多出的那些图
  （探针实测 95 格）从 legacy 换成 MLIR 产物，**仍需对拍一次 kernel 文本与一次上板实测**。
- VF 路线 A（展平）会改变性能特征（中间结果从寄存器挪回 UB），这是倾向路线 B 的主要理由；
  路线选定前不下结论。
- 新增 op 的 kernel 与 legacy 发的是同一条 AscendC API，执行性能应当一致；
  逐字对比要先用改动前代码生成基线（build 目录平时不留 `kernel.cpp`）。

### 内存和产物大小

- 新增 op 只增加编译期注册表与生成代码中的头包含，不增加常驻内存。
- 新增 dtype 不改 `b{id}_size` 的计算口径，但 bf16 的 2 字节步长要确认 ATT 侧分流正确
  （见假设与依赖的待确认项）。

## 接口设计

### 新增/修改接口描述

- 无对外 C/C++ / Python 接口变化。
- `AF_ENABLE_MLIR_CODEGEN`：取值扩一档用于严格模式。它是测试用环境变量，解析仍在
  `ReadMlirCodegenOptions` 一处，不新增配置通道。
- AFIR 内部：新增 op 的 ODS 行、bridge 映射行、implementation 注册行各一；
  新增 dtype 的位与 `defvar` 真源。这些是方言内部契约，不是对外接口。

### 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|--------|----------|----------|------|
| 接口说明 | 是否需要接口评审 | 否 | 无对外接口变化 |
| 接口说明 | 是否需要补充文档 | 是 | 本文档 + 各 op 的 ODS 描述 |
| 接口兼容 | 行为是否兼容 | 是 | 缺省（非严格）行为一个字不改 |
| 接口兼容 | ABI/API 是否兼容 | 是 | 无导出符号与结构体布局变化 |
| 接口约束 | 约束不满足时是否清晰报错 | 是 | 能力不足走 `emitCapabilityShortfall`，坏了走 `kFailed`，两类不混 |
| 接口测试 | 是否需要独立接口用例 | 是 | 严格模式的正负用例（ST） |

## 软件设计

### 关键数据结构

- `ElementTypes` 位掩码（`ElementwiseApiFamily.h:72`）：本阶段扩位的那一处。
  它**放的是拼写不是集合**，各 implementation 收哪几种仍写在各自注册处，不要合并。
- dtype 真源（新增）：一份可机读的表，ODS 的 `defvar` 由它生成或校验。
- `AFIRBridgeOpMapping`（bridge 白名单行）：新增 op 的落点。
- VF 相关数据结构不在本阶段引入。

### 关键技术/算法

- dtype 并集推导：ODS 的 `elementTypes` = 两侧 implementation 位掩码的并集。
  这是机械规则，正是它可机读的理由。
- 新增 SIMT 一元 op 的下降：复用既有 `*Extend` 调用 + 前导头声明，无新算法。

### 流程设计

主流程不变。严格模式只在 `classify()` 这一个分支点插入；异常流程（`kFailed`）与
清理流程（MLIRContext 生命周期）均不变。

### 对子模块的修改

| 目录 | 修改点 |
|---|---|
| `autofuse/mlir/dialect/AFIR/Analysis/` | N1：`localStridesAreLayoutProxy` 与 `axisUnitStride` 往上问 GM 那一支 |
| `autofuse/mlir/bridge/CodegenOrchestration/` | 严格模式取值解析与分类 |
| `autofuse/mlir/conversion/AFIRToAscendC/Implementations/` | dtype 位、新 op 的 implementation 与注册 |
| `autofuse/mlir/dialect/AFIR/IR/Ops/` | 新 op 的 ODS 行、dtype `defvar` 改为由真源来 |
| `autofuse/mlir/bridge/AfirBuilder/` | 新 op 的 bridge 映射行 |
| `autofuse/mlir/test/` | lit / UT / CPU E2E / 探针标签 / 样板收敛 / legacy 用例搬运 |

### 错误处理

#### 系统错误

MLIR 侧的异常已被 `RunMlirCodegen` 的 try 包住，失败一律落到 `classify()`；
不新增内存申请路径。

#### 接口错误

对 `Codegen` 调用方只有两种结论：成功，或失败并带诊断。严格模式下「能力不足」也算失败，
诊断文本保留原标签以便聚合。

## 安全检查

按 `docs/guidelines/编码红线.md`：

- **1 敏感信息硬编码**：无。不新增密钥、公网地址；不新增芯片类型或框架类型的硬编码判断
  （平台差异走 TargetProfile 与 NpuArch，不写 if soc == "..."）。
- **2 外部输入作索引/长度**：N1 用到的 `strides` 来自图属性，用于计算 GM 偏移——
  这正是它出错的地方。`localStridesAreLayoutProxy` 先查 `axis` 与 `strides`、
  `vectorized_axis` 与 `vectorized_strides` 两对长度一致，不一致一律答「不能用」
  （保守方向），不按下标硬取、不截断。
- **3 整数溢出**：新增 dtype 改变元素字节宽度，`count * sizeof(T)` 一类计算要复核。
  已知一处同形态的风险记录在案：比较族 reg 侧发 `static_cast<uint16_t>(count)`，
  count > 65535 会静默截断；今天 UB 预算到不了，**按判断不改**，但新增 dtype 若改变
  单 tile 元素数上限，要重新核这一条。
- **4 资源释放**：无新增资源。
- **5 内存申请**：无新增申请路径。
- **6 未定义行为**：无。
- **7 禁止无理由扩大修改范围**：本设计把 ADR-21 / ADR-22、L-5、L-20 明确排除在外。
- **Graph-autofusion / ABI 兼容**：无导出符号变化。
- **图改写确定性**：本设计不改图改写，N1 最终没有动调度侧。

## 兼容性检查

- 不改脚本参数（`build.sh` / `mlir-test.sh` 不变）、不改安装目录与动态库名、不改 Python 包。
- `AF_ENABLE_MLIR_CODEGEN` 旧取值语义不变。
- 新增 op 会让 `BridgeOperationRegistryTest` 的计数断言失败——**这是设计意图**，
  随改动一起更新。

## 测试设计

### 测试边界

- 入口：`mlir-test.sh`（UT / lit / ST / CPU E2E 四目标）、`check-e2e-cpu-<name>`、
  `run_npu_e2e.sh`、`af-capability-probe`。
- 出口：四目标全绿且退出码 0；探针的 `kFailed` / `DRIFT` 两桶恒为 0。
- 打桩：CPU E2E 用 runtime stub 给默认平台；reg 侧用例按 `PLATFORM reg_base` 切到
  `Ascend950PR_9599` / NpuArch 3510。
- 验证数据：形状不要用一根轴、也不要只含 2 的幂（那样每趟 count 恒对齐，尾块那一类判据
  探不到）；数值用例优先取在目标 dtype 里可精确表示的值，比相等而不是比容差。

### 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|----------|------------|----------|----------|
| 功能 | N1 修复后 nddma 四格数值正确 | `check-e2e-cpu-load_broadcast_nddma_store*` | E2E |
| 功能 | N1 对两侧拒收剖面的影响 | 改动前后各跑一遍 `af-capability-probe`，逐行比对 | 探针 |
| 功能 | 严格模式下未支持图变红 | 构造一张含未建模节点类型的图 | ST |
| 功能 | 新增 dtype 的下降与数值 | 每个 dtype 至少一个 E2E（bf16 首批） | E2E + lit |
| 功能 | 新增 op 的两侧差异 | 每个 op 一对 `--target` lit 正例 | lit |
| 异常 | 能力不足与硬失败不混 | 负例：verifier 失败不得被当成回落 | UT + lit |
| 异常 | dtype 真源与 ODS 不一致 | 门禁脚本 | UT/脚本 |
| 性能 | 新增 op 的 kernel 与 legacy 逐字对比 | 先用改动前代码生成基线再对比 | 手工验证 |
| 兼容性 | 缺省（非严格）行为不变 | 既有全量回归 | UT/ST/E2E |
| 特性交叉 | reg / mem 两侧各自的供给表 | 两档平台各跑一遍探针 | 探针 + E2E |

**每条功能需求的完成判据都要求一个会变红的埋点**，并在落地时记录「改了什么、哪条用例当场红」。

### 测试命令

```bash
bash ./mlir-test.sh
```

```bash
make -C build af-capability-probe -j 6
```

```bash
bash autofuse/mlir/test/run_npu_e2e.sh --mode mlir
```

```bash
sh build.sh -s --module=autofuse_e2e -j 8
```

## 验收标准

1. **N1 闭环**（已达成）：`load_broadcast_nddma_store` 四格在 `check-e2e-cpu` 里数值正确；
   reg 侧探针 68 → 165 行 `ok`（95 格翻转、0 格变差），mem 侧逐字不变。
2. **可度量**：严格模式落地，并产出 legacy 两套用例（44 + 169）的接管表；
   表里不存在「状态未知」的行。
   *进展（2026-10-17）*：**已达成**。机制 + 量具 + 两侧表（mem 38 行、reg 200 行），
   未知 0、缺二进制 0。表存在 `2026-10-17-legacy-takeover-table-mem.txt` 与
   `-reg.txt`。
3. **dtype**：dtype 真源可机读且有门禁；bf16 在两侧各有 E2E 数值证据；
   A5 的 40 个 bf16 用例目录从「编不出」变为有明确状态（接管 / 入册带解锁条件）。
   *进展（2026-10-26）*：真源与门禁已达成（`support/AfirElementTypes.h`），bf16 已开位。
   全量表里 28 个 bf16 用例全部有明确状态：接管 18 格、干净回落 10 格、硬失败 0 格
   （回落首因细分为 VF 不建模 bf16 与个别 op 无 implementation 两类）。**i8 / i64 / ui32 / ui64 仍停在 named-only 表**，
   对应回落 int64 11 格、int8 4 格、bool（dtype value 12）4 格。
   *补（2026-10-26，同日第二轮）*：**bf16 开闸此前只开在 elementwise 那一侧**，`afir.broadcast`
   的 ODS 类型域一直少 BF16 一位，于是 bf16 的广播图失败在 verifier 上——探针 mem 侧 126 格
   硬失败里 120 格、reg 侧 118 格里 43 格都是它。已逐档核头文件后开位（证据与剩余缺口见
   功能需求 6 的 F-1 那一节），mem 侧硬失败 126 → 5、reg 侧 118 → 75，两侧 `ok` 都不降。
   **接管表上这一条的收益是 0 格**，两张 2026-10-26 的表里带这条诊断的行都是 0。
4. **VF**：产出被 VF 挡住的用例占比，并据此定下路线 A / B；实现另立文档。
   *进展（2026-10-26，重量）*：**占比 110 / 200 没变，但「只差 VF」17 → 8 格**，而且
   「VF 是那 55% 的必要条件」被否证——本期换回来的 48 格里 24 格是含 VF 的（普查是关掉
   MLIR 量的，量的是 legacy 调度融不融 VF，不是 MLIR 非接不可）。实现侧 M1 / M2 已落地
   （`7530b4de` / `c561127a`），所以功能 4 不再是「零覆盖」。剩下 8 格里 6 格卡 VF 的
   dtype 白名单、2 格卡 VF 子图里的 `Scalar`。细节与新表见
   `2026-10-26-vf-node-census-reg.txt` 与 VF 实现设计的「重量」那一段。
5. **算子**：A2/A3 的 5 个缺口各有「不补」的书面裁决（本文档已给）；
   A5 的 83 个全部入册，每条带形态、dtype 需求与机制归属；已接的每个都有两侧 lit + 一个 E2E。
   *进展（2026-10-26）*：七批接完，bridge 白名单 47 → 99。全量表里 48 格从 `fallback`
   翻成 `takeover`，0 格变差。剩余 fallback 里仍由「bridge 不收这个节点类型」挡住的，
   主要是多项式族（Chebyshev / ShiftedChebyshev / Hermite / Laguerre / Legendre）、
   Gather / Bitwise / Transpose / Rand 族，以及 Frexp / Fma / IndirectLoad 等单点。
   *进展（2026-10-27）*：第 8 批接通 `log10` / `hypot`，bridge 白名单 99 → 101。两个都是
   **标准调用**：`log10` 是第 6 批注释里点名留下的那个空组合（`MathLibraryOp` 却不要 tmp，
   撞 `UnaryElementwiseConversionPattern` 的 `static_assert`），解法不是放开那条
   `static_assert` 而是换供给——`Adv/Math.td` 本来就没有同名的 `UnaryMathOp`，在 Autofuse
   子树里按 `UnaryL2Op` 新建一个即可；`hypot` 的坑是反向的，fork 里有 `api_regbase/hypot.h`
   的 `HypotExtend` 但 legacy 一处都不加载它，照文件名挑供给会发出一个 legacy 从不发的调用。
   接管表 69 → 71，逐格对完只有该变的两行变、198 行逐字未变、0 格变差，见
   `2026-10-27-legacy-takeover-table-reg-b8.txt`。
   **口径订正**：「bridge 不收节点类型」在范围内是 **26 格**而不是总表原先写的 27
   （单点那一族逐项相加是 13 不是 14）。本批吃掉 2 格，剩 24。
   **新识别的阻塞项**：第 9 批（多项式族八个）在 aarch64 上把 `autofuse/tests/libcodegen.so`
   的 `.text` 顶过了 `R_AARCH64_CALL26` 的 ±128 MiB，legacy ST 档链不过、接管表量不出来。
   干净树那份 `.text` 就已占 98.1%。**这是后面每一批补 op 的前置**，数与候选方向记在剩余
   工作总表的「阻塞项」那一节。
   *进展（2026-10-28）*：上面那条阻塞项**已解**（`21f9611a`，`libcodegen.so` 的 `.text`
   126 → 101 MiB、余量 2.1 → 28.4 MB，判据是接管表逐格不变、实测 0 行差异）。第 9 批随后
   接通多项式族八个，bridge 白名单 101 → 109，接管表 71 → 79、硬失败不变、192 行逐字未变、
   0 格变差，见 `2026-10-28-legacy-takeover-table-reg-b9.txt`。这一批引入的是「整数
   `ir_attr` 落在模板实参位」那根轴（`ChebyshevPolynomialTExtend<float, 3>(...)`），
   不是八个新算子。
   *进展（2026-10-28，step 2a）*：接通 `ceil2int` / `floor_to_int`，bridge 白名单 109 → 111，
   接管表 79 → 81、硬失败不变、198 行逐字未变、0 格变差，见
   `2026-10-28-legacy-takeover-table-reg-s2a.txt`。这一批引入 `ResultElementRule::Int32`
   （结果恒 signless i32）与一条「固定 RoundMode 的标准 `AscendC::Cast`」pattern。
   **订正一处会发出不存在调用的错**：对照表的 `reg_api_name` 列对这一族给的是 legacy 的
   `GetApiName()`，不是发出去的函数名——CANN 里没有 `AscendC::FloorToInt`，实际发的是
   `AscendC::Cast(dst, src, RoundMode::CAST_FLOOR, count)`。
   **`round_to_int` / `trunc_to_int` 写过一遍后退回**：它们卡在 `afir.store` 的「两端 dtype
   必须相同」（legacy 调度出的图存 i32 而 GM 侧是 f32），与 uint8 那一族同一道闸门，
   接它的前置是先给这种 store 定语义，那是独立一件活。
   「bridge 不收节点类型」在范围内从 16 格降到 14 格。
   *进展（2026-10-28，step 3）*：**`fma` 写过一遍后整体退回，接管表仍是 81。**
   实现本身跑通了（三元那条现成 pattern 直接复用——`TernaryOperandTiles` 的 `condition`
   只是「三个源里的第一个」这个命名，`clip_by_value` 早就以 `conditionIsMask = false`
   在用它；`FmaL2Op` 按 pattern 的顺序声明 `arguments`，由 emitter 打成
   `AscendC::Fma(dst, s0, s1, s2, tmp, count)`，所以 tmp 在 count 之前这件事不需要新
   pattern）。退回的理由是实测 **2 格里 1 格变差**：`abs_fma_test` 回落 → 接管，而
   `abs_fma_bf16_test` 回落 → **硬失败**（takeover 81 → 82、hard-failure 11 → 12，
   198 行逐字未变）。
   卡点与 `round_to_int` / `trunc_to_int` 是同一道闸门：legacy 自己调度出的签名
   `(%data0: bf16[s0], ..., %output: f32[s0])` 配上 `Store(Cast(Fma) -> bf16)`，
   `afir.store` 两端一个 bf16 一个 f32。**这根轴上没有 dtype 旋钮**——收窄注册集合也拦不住，
   那条 `afir.fma` 在这张图里本来就是 f32（`DtypeConsistency` 按
   `FmaAscIrCodegenImplV2::GetConversionDtype` 的 `bf16->f32` 抬过），硬失败发生在
   bridge 建 store 的 verifier 上，比 implementation 选择更早。
   于是那个「给两端 dtype 不同的 store 定语义」的前置现在挡着 2b 的 2 格 + `fma` 的 1 格，
   并且是接管表 11 格硬失败里 9 格的首因，细节与量法在工作总表 step 3 那一节。
   *进展（2026-10-28，前置已接）*：**那条前置的风险最低的一半已落地**——`AfirBuilder.cpp`
   的 `buildBufferNodeOp` 在建搬运节点（Load/Store/Nddma）之前先比一次两端元素类型，
   不等就走 `CapabilityShortfall` 回落，不碰 `afir.store` 本身「两端必须同 dtype」这条
   不变量（没有按字节宽度悄悄放开）。接管表逐格核过：81 格 takeover 不变、硬失败
   11 → **2**（s2a 那 9 格全部翻成 fallback，198 行里其余 191 行逐字未变）；production
   侧抽查 `sign_uint8_test_codegen_v2` 在缺省配置下从崩改为 `OK`。剩 2 格硬失败是另外两件
   与这条无关的事（`afir.cast` 操作数类型约束、`Scalar inf` 字面量解析）。
   *进展（2026-10-28，step 3 重接）*：前置清完后把 `fma` 重新接上，代码与退回那版逐字
   相同。接管表 81 → **82**（`abs_fma_test` 回落 → 接管，0 格变差）。`abs_fma_bf16_test`
   仍留在回落——不是硬失败了，首因诊断换成了「Store 两端 bf16/f32 不同，拒收」，这正是
   前置在接；它卡的不是 `fma` 的 dtype 注册，是「两端 dtype 不同但字节宽度相同的 store」
   还没定语义的那一半，前置只堵住了硬失败、没有放开接管。所以 step 3 最终交付 1 格，
   `Fma` 这个节点类型从 `kRegOnlyElementwiseOps` 摘掉。
   *订正（2026-10-28）*：**「两端 dtype 不同但字节宽度相同的 store」这半前置根本不需要
   存在**——`abs_fma_bf16_test` 的 bf16(2B)/f32(4B) 本来就不等宽，「reinterpret」从一开始
   框架就错了。真卡点是 legacy 用例样板漏了一行：`AbsFmaBf16FusedGraph` 的 `Output` 节点
   没补 `y.y.dtype = af::DT_BF16`，落到 ODS 缺省 f32——与 `6a525700d`（补 12 个 bf16 用例
   漏设的 Output dtype）症状逐字相同。查过另外 9 个仍卡着的（`afir.bitwise_not/or/xor`
   三个、`load_logicaland/or_store`、`sign_uint8`、`float/half/int16/int32/
   uint8_logical_not`），全部漏了同一行。`share_graph.cc` 逐个补上（`round_to_int` /
   `trunc_to_int` 还没注册实现，顺手一起补，补了不白补）之后：接管表 82 → **95**（+13），
   两端套件 succeeded、lit 220/220，187 行逐字未变，硬失败仍是 2、out-of-scope 仍是 2。
   `d5930fcd` 那条 capability 干净拒收原样保留，没有碰 `afir.store` 本身的语义。
   **这一批量了两次，第一次是变差**：ODS 写 `SI64Attr` 而 bridge 建 `getI64IntegerAttr`，
   verifier 拒收、八格从干净回落变成硬失败；lit 按原理抓不到它（lit 从「属性已经写好」
   开始，bridge 的构造路径一行都没走到），所以补了一条走 `BuildAfirModule` 并断言
   `mlir::verify` 的 UT，并做了变异检查。「bridge 不收节点类型」在范围内从 26 格降到 16 格。
6. **形状**：探针 `kFailed` / `DRIFT` 恒为 0；F-2 收掉后重量 F-3 并更新剖面；
   F-4（plan selection 答不出「不支持」，2026-10-17 新识别）那 14 格从 `hard-failure`
   翻成 `fallback`。
   *进展（2026-10-26）*：**F-4 已落地，两侧都量了**，见
   `2026-10-26-f4-plan-selection-shortfall.txt` 与 `2026-10-26-legacy-takeover-table-mem-f4.txt`。
   那个数不是 14：reg 侧 39 格（hard-failure 66 → 27、fallback 69 → 108），mem 侧 3 格
   （hard-failure 20 → 17）。两侧 takeover 都逐格未变、0 格变差；
   `af-capability-probe` 的 `FAILED` 128 → 126、`DRIFT` 0 → 0。
   **范围与本文档原先设想的不同**：标记只覆盖「模板判据拒掉了某个 stage 的全部模板」那一类
   （26 queue-resources + 11 elementwise-broadcast + 2 data-movement）；余下 16 格
   （`gate=candidate-preparation` 但死在 ATT 之后的字段校验，`AttSchemaAdapter.cpp:130`）
   是 ATT 与 `TilingFieldSchema` 两侧 tiling_key 拼法对不上，属缺陷而非能力边界，**不标**，
   仍是硬失败并入册待判。F-4 是功能 7 的前置，这一条因此解开。
7. **用例**：E2E 样板收敛，逐字重复的部分收掉；搬运后的用例在严格模式下复跑一遍。
   **F-4 是搬运的前置**：今天搬进来的非 elementwise 用例不会「回落后变绿」而是当场红。
   *订正（2026-10-27）*：**原先写的「单用例 100 行以内」是个错的指标，已改掉。** 实测一个
   用例 190 行（中位）里，`Case.cpp` 的建图段与 `KernelTest.cpp` 的数值参考占 150 行上下，
   它们**不是样板**——每个用例的图和期望值本来就不一样，压它们等于压掉那条「注释要写出理由」
   的要求。真正逐字重复的只有 `CMakeLists.txt` 与 `Case.h`，所以指标换成「逐字重复的部分
   收掉」，可验、也不诱导错的优化。
   *进展（2026-10-27）*：**已达成。** 74 个用例全部收完（原先只改了 3 个当样例）：
   `Case.h` 收成 `AF_MLIR_E2E_CPU_CASE_BEGIN/END` 一对宏（含收掉**每个用例各抄一份**的两个
   stringify 宏）、`CMakeLists.txt` 的两个源码列表走 `add_mlir_e2e_cpu_case` 的缺省值
   （`<用例目录>/Case.cpp` 与 `/KernelTest.cpp`）。142 个文件 +145 / -1082 行，
   `Case.h` 合计 2144 -> 1391 行。用例自有的内容逐个保住了：12 个用例在两个宏之间留着自己的
   常量、`static_assert` 或 `using`（例如 `load_broadcast_duplicate_store` 那条
   `element_count == 1U`、`load_axpy_store` 的 `kAlpha`）。
8. **文档**：本文档随各项落地更新状态；被它取代的四份文档「下一步」各节标注指向本文档。

---

### 设计文档检查结果

- [x] **跨特性交叉影响**：已按 `cross_feature_check.md` 逐项填写（见「特性交叉影响」）。
  适用项为 Codegen/Backend、AscendC API 交互、测试与覆盖率、性能与日志；构建打包与兼容性
  为部分适用；SuperKernel 两项、Python 绑定与 Autofuse 图优化不适用。
- [x] **编码红线**：已按 `编码红线.md` 逐条检查（见「安全检查」）。重点三处——N1 的
  偏移来源（规则 2）、新增 dtype 引起的字节宽度与 count 截断复核（规则 3）、
  范围边界明确排除 ADR-21/22 与 L-5/L-20（规则 7）。无违反。
- [x] **测试设计**：按变更类型分 UT / lit / ST / CPU E2E / NPU E2E / 探针六类，
  每条功能需求要求一个会变红的埋点；见「测试用例设计」。
- [x] **性能评估**：涉及编译期注册表增长、N1 修复后 reg 侧产物切换、VF 路线 A 的
  UB 占用变化三处，已分别给出评估与「要实测不要推」的标注。
