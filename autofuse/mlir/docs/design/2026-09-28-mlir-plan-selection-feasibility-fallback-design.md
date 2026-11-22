# MLIR 计划选择的可行性与回退

> **接口迁移说明（2026-10-22）**：正文提及的 `AF_MLIR_PINNED_*` / `AF_MLIR_ENABLE_PLAN_PINNING` 是历史接口，现已移除；当前使用 ATT `AUTOFUSE_DFX_FLAGS` 的 `--force_schedule_result` / `--force_tiling_case`。候选编号须核对当前 shape 的 schedule dump。

> **当前实现核对（2026-10-22）**：本文下方保留原设计及历史验收记录，不能据其中的“当前/今天”判断现状。
> 静态形状仍先调用 `SelectStaticWinner`；代价无法求值时保留全部候选，动态形状保留全部候选。
> 当前不做逐候选下降容错：一个候选失败会拒绝整份 MLIR 产物，不按幸存者重建 tiling。
> 诊断带能力不足标记时返回 `kNotSupported` 并交给 legacy；其他失败返回 `kFailed`，不会自动回落。
> 真源：`bridge/CodegenOrchestration/CodegenOrchestration.cpp`、`bridge/PlanSelector/PlanSelector.cpp`。
> 原 MLIR pinning 开关已移除，当前测试使用 `AUTOFUSE_DFX_FLAGS="--force_schedule_result=N;--force_tiling_case=M"`。


# 简介

## 目的

让 MLIR 后端在「某个计划（plan）在当前形状上不可用」时换一个计划，而不是让整张图失败。

原方案提出时，MLIR 在**静态形状**下于编译期按代价模型选定**唯一**一个计划，随后只为它生成 kernel
与 host tiling。该计划在运行期解不出 tiling 时，kernel 初始化即失败，没有第二个候选可退。
legacy 不是这样：它把全部 schedule result 都生成出来，host tiling 逐个试、解不出的静默跳过，
按 perf 选最优。本方案把这条语义补齐到 MLIR 侧。

## 范围

**在范围内**：`bridge/PlanSelector` 的静态选择策略、`bridge/CodegenOrchestration` 的候选下降
与产物装配次序、`artifact/KernelSource` 分发默认分支的失败形态。

**不在范围内**：ATT 求解器本身、RCore 的切分方案（见下「假设和依赖关系」）、动态形状路径的
选择策略（它已经保留全部候选，本方案只是让静态路径与它汇合）。

# 总体概述

## 软件概述

### 项目介绍

Ascend NPU 编译器后端的 MLIR 通路：承接上层框架导出的计算图，经 ASCIR / ATT 自动调度产出若干
**计划**（同一张子图的不同调度方案，如归约的 RCommon 与 RCore），再由 AFIR 下降到 AscendC。

### 产品环境介绍

主要调用方是 torch inductor（`MlirCallerScene::kInductor`）。该场景下形状通常是常量，
`ascgen_utils::IsStaticSchedResult` 为真，走的正是本方案要改的静态分支。编译发生在模型预热期，
每个形状一次。

## 软件功能

对一张融合子图，MLIR 通路要从 ATT 给出的计划集合里产出**至少一个**可用的 kernel 与配套 host
tiling；可用的定义是：MLIR 能把它下降到 AscendC，且它在目标形状上解得出 tiling。

## 设计约束

1. **不改 ATT。** ATT 的求解器与 `optiling::GetTiling` 是既有稳定件，`autofuse/att/` 有稳定性
   约束，本方案只改调用方给它什么输入。
2. **下降出来的计划集必须与 tiling 能选中的计划集逐一对应。** 二者由同一个
   `narrowed_result` 的计划编号联系起来（`NarrowFusedResultToSelections` 会把编号重排成从 0
   连续），任何一侧多一个或少一个都会让 `graph<N>_tiling_key` 指到不存在的分支。
3. **失败必须是拒绝，不能是静默。** MLIR 通路失败时整图回落 legacy，这是安全的；而分发落到
   空分支、或者下降出来的计划与 tiling 不匹配，都是**算错**，不是失败。
4. 编译时长不能显著劣化——预热期编译在用户程序的关键路径上。

## 假设和依赖关系

- 依赖 ATT 生成的 host tiling 在多计划输入下沿用 `GenGetAllSchedulesResults` 的语义：逐个试、
  返回值丢弃、`GetResultSummary` 仅在全部失败时才失败。**这一条要在实施第一步用实机产物证伪
  或证实**（见测试设计 T1），它是整个方案的地基。
- 假设「计划不可用」有且只有两种成因：MLIR 下降不了（能力缺口，编译期可见）、tiling 解不出
  （运行期可见）。前者由本方案的容错下降处理，后者交给 ATT 的多计划求解器。
- RCore 在大 A 上无解不是本方案要修的缺陷（它是切分方案的适用域，见
  `2026-09-27-mlir-att-schedule-group-tiling-naming-alignment-design.md` 的归因更正）。本方案
  要做的是让这种无解**不再致命**。

# 需求分析与设计

## 整体介绍

关键事实是：**多计划这条路端到端已经存在，并且动态形状路径每天在走它。**

- `SelectDynamicCandidates` → `BuildAllCandidateSelections` 保留全部计划；
- `GenAggregateKernelEntry` 已按 `graph<N>_tiling_key` 发 `scf.IndexSwitchOp` 分发；
- TilingData schema 的 `graph<i>_result<j>_g<k>_tiling_data` 嵌套本来就是按多计划设计的；
- `AggregateLaunchDriverTest.ProducesArtifactsForMultiplePlans` 已经钉住多计划装配。

所以本方案**不新建回退机制**，而是：静态路径停止收窄，并把这条路上两处「只有单计划时才成立」
的假设补上。

## 当前实施状态（2026-10-22 核对）

| 原设计需求 | 当前状态 |
|---|---|
| 静态形状保留全部计划候选 | 有条件保留：静态代价无法求值时保留全部；可定选时仍收窄 |
| 逐候选下降容错 | 当前不保留；任一候选失败即拒绝整份 MLIR 产物 |
| 按幸存候选重建 tiling | 当前不采用；ATT 在候选下降之前运行，不剔除失败候选 |
| 分发默认分支必须失败 | 已实施 |
| 静态代价模型只作诊断 | 当前未采用；仍参与静态定选 |

下面至“需求分析与设计”之前保留 2026-10-07 的历史取证，描述的是当时修改，
其中“现在/已实施”均不表示当前 checkout。当前失败分类见文首，历史 REFUSED 统计不能用于解释现有 PASS_FALLBACK。

### 需求 1 的状态更正（2026-10-07）

本表曾记「已实施（`f2ff1c66`）」。核对 `f2ff1c66` 的实际内容：它把「编译期定选选不出」从
**错误**改成**回退到运行期分派**，但**没有**停止调用 `SelectStaticWinner`——静态分支仍然先试
编译期定选，定出来就只下降那一个计划。落地的只是失败路径那一半。

剩下那一半与需求 5 是同一处改动，已于本次一起补齐：`SelectPlanCandidates` 的非钉选路径统一走
`SelectDynamicCandidates`。

### 需求 5 的核对与实施（2026-10-07）

`SelectStaticWinner` 有两条路：

1. `HasUniqueCandidatePerSubgraph` 为真 → 原样返回全部候选，**不收窄**。代价模型根本没被问到，
   这条路无害。
2. 否则 → `GetModelInfoMap` + `SelectWinnersFromModels`，**每个子图挑一个赢家、其余丢弃**。

所以代价模型今天仍然在**决策**而不是仅仅诊断，需求 5 未实施。而且收窄比需求 5 原文描述的更彻底：
`SelectTemplatesForPlan` 对每个 stage 也只 `push_back` 一个 `template_id`，而
`BuildAllCandidateSelections` 走的是 `MakeContiguousIds(impl_graphs.size())`（全部模板）——
**计划和模板两层都被收到一个**。

门禁在 `ScoreModel`：`model.objects` 的每一项与 `head_cost` 代入具体 shape 后都必须
`IsConstExpr()`。`f2ff1c66` 的归因是 `ModelInfo::objects` 是 ATT 求解器的目标函数、自由变量是
切分轴的 size 而不是原始 shape，所以代进去仍然不是常量，门禁基本不成立。

**但「基本不成立」不等于「恒不成立」**，有一条具体的通路会让它成立：`ScoreModel` 里
`model.objects` 为空时那个 for 循环一次都不进，`has_pipe_cost` 保持 false，于是只要
`head_cost` 是常量就**评分成功**。也就是说某个计划的目标项集合为空，代价模型就会开始决策。

这件事在需求 2、3 落地之后更要紧：逐候选容错的退路长度由交给它的 selections 决定，
一旦被收到一个就没有退路了——与
[钉 plan / 钉 template 的总闸](2026-10-03-mlir-plan-pinning-switch-design.md) 记录的
实测失败形态（钉 RCore 候选时 multicore tiling 无解，失败发生在运行期且只报
`NPU kernel <path> init failed`）是同一个机制。

**实施**：代价模型保留，但搬进 `LogStaticCostModelOpinion`，只 `GELOGD` 一句「静态模型认为最优
的是哪个计划」，不参与决策。它与运行期实际选中的 `graph<N>_tiling_key` 之间的分歧，是排查
选择质量问题的唯一线索。本次**不引入** top-K 配置项。

诊断前置了 `IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)` 判据：`SelectStaticWinner` 会调
`GetModelInfoMap` 做 ATT 建模，为一行默认关掉的 debug 日志付这笔钱会劣化编译时长，而预热期
编译在用户程序的关键路径上（非功能需求第 4 条）。这是仓里既有写法（`shape_refiner.cc`、
`vector_func_partitioner.cpp` 都这样守）。

**实测**：行为无变化，符合预期——门禁今天几乎不成立，所以收窄本来就没在发生。
910B `norm_fuzz.py` 46 格 **15/46**，与实施前逐格相同（通过的是同一批编号）；
`reduce_op_fuzz`、`reduce_template_matrix` 两条回归 rc=0；CPU 三条 norm 语料全绿且耗时持平
（无编译时长劣化）。需求 5 的实质是**拆掉潜伏通路**，不是修正在发作的缺陷。

UT `StaticShapeKeepsEveryCandidate` 守的是不变量而**不区分本次改动**：两候选时旧代码的门禁同样
失败、同样回落到全候选，所以改动前后它都过。要让门禁成立需要 ATT 产出 `objects` 为空的 model，
UT 里造不出来。它的价值是：谁在静态路径上重新引入收窄（例如加 top-K），这条用例会红。

需求 4 早于本次就由 `db5af3be` 落了：`buildGraphDispatch` 给
`scf.IndexSwitchOp` 的缺省区域发 `ascendc.trap` 而不是空 `scf.yield`，两侧都有测试守着——
`IsolatedPlanEntryTest` 断言缺省区域里恰有一个 `TrapOp`，
`test/lit/conversion/afir-to-ascendc-gen-kernel-entry.mlir` 另有 `CHECK-NEXT: ascendc.trap`。
本表此前写「未实施」是 2026-10-02 那次核对之后没跟上，**不是**真实状态。

### 实测收益（910B，`norm_fuzz.py` 46 格，seed 20261004）

| 阶段 | 通过 |
|---|---|
| 需求 2、3 之前 | 11/46 |
| 需求 2、3 之后 | **15/46** |

翻过来的四格是 rmsnorm / meansub 的 innermost（`truediv` 与全量形状结果共用 `buf_id` 判
`BindingConflict`）：这些格子**仍然各报一次 `BindingConflict`**，但冲突的候选被丢弃、另一个
候选顶上，整图成功。这就是需求 2 要的形态。

另一个不在预期内的收益：**「某格失败」的含义变强了**。此前一个候选失败即整图失败，所以
REFUSED 说不清缺口是普遍的还是只差一个候选；现在 REFUSED 意味着该子图**所有**候选都失败。
证据是有的格子（softmax innermost、layernorm innermost）现在同时出现 `InvalidWorkspace` 与
`BindingConflict` 两种成因——不同候选死于不同原因，逐个试过了。

反例也要记下：maxsub 的 outermost / middle / multi 共 5 格仍挂在 queue 域的
`BindingConflict`（菱形两条路径各 load 同一输入，`load`/`load1` 共用
`queue buffer_id=0 reuse_id=0` 而形状不同）。曾按单一形状 `(256, 128)` 的 dump 推断「同图的
另一个候选干净、容错落地后应当恢复」，实测没有——那个结论是**形状相关**的，不能外推到整格。

## 功能需求

### 功能需求 1：静态形状保留全部计划候选

`PrepareMlirCandidates` 的静态分支不再调 `SelectStaticWinner` 收窄到一个赢家，改为与动态分支
同源，保留该子图的全部计划。选哪个交给运行期的 ATT 求解器——它按 perf 选，且天然跳过解不出的。

`AF_MLIR_PINNED_PLAN_ID` / `AF_MLIR_PINNED_TEMPLATE_ID` 的**裁剪行为**不变，但它们已于 2026-10-02
收到总闸 `AF_MLIR_ENABLE_PLAN_PINNING` 之后、**默认关闭**（见
[钉 plan / 钉 template 的总闸](2026-10-03-mlir-plan-pinning-switch-design.md)）。

理由正是本节这条「选哪个交给运行期的 ATT 求解器」：钉 plan 会把候选集裁剪到一个，而 ATT 那条
「全试、取优、只有全部候选都无解才失败」的退路，长度正是由交给它的 selections 决定的——裁到一个
就没有退路了。实测 `(41, 7, 2)` 归约 dim=1 钉 RCore 候选时 multicore tiling 无解，失败发生在
**运行期**且只报 `NPU kernel <path> init failed`；同一形状不钉，ATT 换一个 `tiling_key` 直接过。
所以钉选是调试专用能力，不该由现网路径碰到；闸关着却设了钉，按配置自相矛盾报 `option_error`
而不是静默忽略（静默忽略会让一条本以为钉住了某模板的用例变成测 ATT 挑中的那个，用例照样绿）。

### 功能需求 2：逐候选下降容错

`BuildCandidateAfirModules` 改为：某个候选 `PlanToAfir` 或下降流水线失败时，**记录并丢弃该候选**，
继续下一个；一个子图的候选全军覆没时才判整图失败（回落 legacy）。

这一条是需求 1 的**前置条件**，不是锦上添花。今天 MLIR 的逐计划下降覆盖并不完整（例如 A=1 的
纯全归约在 `LoopGenerate` 就报 `tensor SSA value crosses sibling/outer schedule scopes`）。不做
容错就放开计划集，等于把「某个计划下降不了」从「选另一个」变成「整图回落 legacy」——对今天能
跑的图是**净劣化**。

丢弃必须留痕：`GELOGW` 给出 `subgraph/plan` 与失败阶段，否则「为什么这张图只剩一个计划」无从查起。

### 功能需求 3：下降与 tiling 的计划集保持一致

今天的次序是「先 `PrepareMlirCandidates`（选择 + 收窄 + ATT tiling 生成）→ 后
`BuildCandidateAfirModules`（下降）」。需求 2 会在下降阶段丢弃候选，而此时 tiling 已按更大的
集合生成完毕——`graph<N>_tiling_key` 于是可能选中一个 kernel 里不存在的计划。

把次序倒过来：**先下降、后按幸存者生成 tiling。**

可行性已核对：下降只消费 `narrowed_result` 与 `PlanLoweringSession`，不消费 `bundle`；
`bundle` 只在 `FinalizeMlirCodegenResult` 才被用到。

编号问题：`NarrowFusedResultToSelections` 会把计划编号重排成从 0 连续，所以幸存者集合一旦小于
初始集合，编号就变了，已下降出来的模块带的 `#afplan.candidate` 会对不上。处理办法是**只在发生
丢弃时重新收窄并重下降一遍**：常规路径（无丢弃）零额外成本，退化路径多一次下降，换取编号在
两侧逐字一致。不采用「留空洞」的编号方案——那要求 ATT 为不存在的计划也生成 tiling。

### 功能需求 4：分发的默认分支必须是失败，不是空操作

`GenAggregateKernelEntry` 给 `scf.IndexSwitchOp` 的默认区域放的是一个空 `scf.yield`。单计划时
`tiling_key` 恒为 0，走不到；多计划时它变成一条**静默算错**的路：kernel 什么都不做就返回，
输出 GM 保留调用前的内容，数值错而无任何报错。

默认分支改为发一条可观测的失败（`trap` / `assert` / 打印后置错误码，取设备侧可用的那一种），
让「tiling 选了一个 kernel 没有的计划」当场暴露。这是需求 3 的兜底：需求 3 保证它不发生，
需求 4 保证它发生时不是静默的。

### 功能需求 5：静态代价模型降级为诊断

`SelectStaticWinner` / `SelectWinnersFromModels` / `AttCostEvaluator` 不再决定生成哪个计划。
保留它们，职责改为：在 `GELOGD` 里给出「静态模型认为最优的是哪个计划」。它与运行期实际选中的
`tiling_key` 之间的分歧，是后续排查选择质量问题的唯一线索。

不删除的另一个理由：若将来计划数膨胀到编译时长不可接受，需要一个 top-K 上限，这套代价模型就是
现成的排序键。本次**不引入** K 这个配置项——没有数据支撑的旋钮只会变成第二处要维护的判据。

## 非功能需求

### 可维护性

静态与动态两条选择路径合并成一条，`PrepareMlirCandidates` 里少一个分支。代价模型从「决策」
退到「观测」，判据数量净减少。

### 可测试性

需求 2、3 的失败形态在 UT 层就能构造：给一个子图两个计划、让其中一个下降失败，断言产物里只剩
另一个、且 tiling 的计划集与之一致。需求 4 可由 lit 检查默认分支不再是空 yield。需求 1 的收益
需要实机（RCore 大 A 形状）。

### 可移植性

不涉及芯片型号或框架类型判断，不新增硬编码。

### 可靠性

核心收益就在这里：单点故障（唯一计划解不出 tiling）变成可退化。同时需求 3、4 消除了本方案自身
引入的两条静默算错路径。

### 特性交叉影响

| 场景 | 适用性 | 分析说明 |
|------|--------|----------|
| SuperKernel Python 接口 | 不适用 | 不涉及 Python 包与选项解析 |
| SuperKernel C++/AOT 接口 | 不适用 | 不涉及 `libascendsk.so` 与 ABI |
| Autofuse 图优化 | 不适用 | 不改 ASCIR 图结构与融合 pass；输入的 `FusedScheduledResult` 原样使用 |
| Autofuse Codegen/Backend | **适用** | 改选择策略、候选下降次序与产物装配；kernel 由单计划变为多计划，host tiling 由单计划求解变为多计划求解 |
| AscendC API / Runtime 交互 | **适用** | 需求 4 要在设备侧默认分支发失败，需选一种不违反 AscendC 生命周期约束的手段 |
| Python/C++ 混合绑定 | 不适用 | 不改 pyautofuse 导出面 |
| 构建与打包 | 不适用 | 不改 CMake 与安装路径 |
| 测试与覆盖率 | **适用** | 新增 UT 与 lit，见测试设计 |
| 性能与日志 | **适用** | 编译时长与产物大小上升，见性能一节；新增的是 `GELOGW`/`GELOGD` 级别日志，不在高频路径 |
| 兼容性 | **适用** | 生成的 kernel 与 TilingData 结构变化，见兼容性检查 |

## 性能

### 编译时长

计划数从 1 变为 N（归约今天 N=2，elementwise N=1）。需要注意的是**成本是次线性的**：聚合阶段
把 N 个计划的 main 装进**同一份** kernel 源码，bisheng 只编一次，只是文件更大；host tiling 同理。
增量主要在 MLIR 侧的 N 遍下降流水线。

需求 3 的重下降只在发生丢弃时触发，常规路径不付这个成本。

实测要求见验收标准第 4 条：对 `load_sum_store` 这类 N=2 的图，端到端编译时长劣化不超过既有
单计划路径的 1.6 倍。

### 执行性能

运行期多一次 `tiling_key` 分发（一个 switch），相对 kernel 本体可忽略。计划选择质量由 ATT 的
perf 模型决定，与 legacy 同源，预期不劣于今天的静态代价模型。

### 内存和产物大小

kernel 二进制与 TilingData 结构随计划数增大。TilingData 是每个 (subgraph, plan, stage) 一个嵌套
结构体，N=2 时约翻倍；它驻留在 GM，量级为 KB。

## 接口设计

### 新增/修改接口描述

内部接口，不对外导出：

- `SelectStaticWinner` 保留原型，调用点从 `PrepareMlirCandidates` 移除，改由诊断路径调用。
- `BuildCandidateAfirModules` 返回值增加「被丢弃的候选」信息（或改为返回幸存者的 selections），
  使调用方能判断是否需要重新收窄。
- `PrepareMlirCandidates` 拆成两段：选择 + 收窄（下降前）、ATT tiling 生成（下降后）。

### 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|--------|----------|----------|------|
| 接口说明 | 是否需要接口评审 | 否 | 均为 `mlir::bridge` 内部接口 |
| 接口说明 | 是否需要补充文档 | 是 | 本文档 + `AGENTS.md` 中「为什么产物里有多个计划」的说明 |
| 接口兼容 | 行为是否兼容 | 是 | 同一张图的产物由单计划变多计划，数值结果不变 |
| 接口兼容 | ABI/API 是否兼容 | 否 | 无对外符号变更 |
| 接口约束 | 约束不满足时是否清晰报错 | 是 | 需求 4 即为此项 |
| 接口测试 | 是否需要独立接口用例 | 是 | 见测试设计 |

## 软件设计

### 关键数据结构

- `PlanSelection{subgraph_id, plan_id, template_ids_per_stage}`：不变。静态路径下 `selections`
  的长度从「子图数」变为「全部 (子图, 计划) 数」，与动态路径一致。
- `MlirCandidatePreparation`：拆分后 `bundle` 不再与 `narrowed_result` 同时产出，需要重新界定
  这个结构体的边界，或拆成两个。
- `BuiltPlanModules`：增加幸存者的 `selections`，供需求 3 判断是否重新收窄。

### 关键技术/算法

不涉及新算法。选择由「编译期代价最小化」改为「运行期可行性优先 + perf 最小化」，后者是 ATT 既有实现。

### 流程设计

```
BuildAllCandidateSelections（全部计划）
  → NarrowFusedResultToSelections（编号重排 N1）
  → 逐候选下降（容错，收集幸存者）
  → 幸存者为空 ⇒ 整图失败，回落 legacy
  → 幸存者 == 全体 ⇒ 直接进入下一步
     幸存者 ⊂ 全体 ⇒ 按幸存者重新收窄（编号 N2）并重下降
  → 按最终计划集生成 ATT host tiling
  → 聚合装配（plan_entries / tiling_key 分发 / 默认分支失败）
```

### 对子模块的修改

| 模块 | 修改 |
|------|------|
| `bridge/PlanSelector/PlanSelector.cpp` | 静态分支改为保留全部候选；`SelectStaticWinner` 转为诊断用途 |
| `bridge/CodegenOrchestration/CodegenOrchestration.cpp` | 拆分 `PrepareMlirCandidates`；`BuildCandidateAfirModules` 容错并回报幸存者；插入重新收窄分支 |
| `artifact/KernelSource/GenAggregateKernelEntry.cpp` | 分发默认分支由空 `scf.yield` 改为可观测失败 |

### 错误处理

#### 系统错误

原方案设想逐候选容错；当前任一候选下降失败即结束 MLIR 编排。仅能力不足返回 `kNotSupported` 并回落 legacy；`kFailed` 是硬失败，不回落。

#### 接口错误

- 单个候选下降失败：`GELOGW` 记 `subgraph/plan` 与失败阶段，丢弃，不中断。
- 运行期 `tiling_key` 指向不存在的计划：设备侧默认分支发失败（需求 4）。此路径在需求 3 成立时
  不可达，属于防御性断言。

## 安全检查

已读 `docs/guidelines/编码红线.md`。逐条：

- **敏感信息 / 公网地址 / 芯片或框架类型硬编码**：不涉及，本方案不新增任何判型分支。
- **外部输入作索引、长度、偏移**：`tiling_key` 来自 host tiling 写入的 TilingData，是设备侧读到
  的外部输入，被用作 switch 选择子。需求 4 正是这一条的落实——越界必须可观测，不能落到空分支。
- **整数溢出 / 反转 / 除 0**：不涉及新的算术。
- **资源生命周期与异常分支释放**：`BuiltPlanModules::owned` 持有 `OwningOpRef`，丢弃候选时必须
  连同 `owned` 与 `views` 一起丢，不能只从 `views` 摘掉——否则 `views` 里留下悬垂 `ModuleOp`。
  这是容错改造里唯一的内存安全点。
- **ABI/API 兼容**：无对外符号变更。
- **图改写等价性与确定性**：候选遍历与丢弃必须按固定次序（按 `subgraph_id`、`plan_id` 升序），
  否则同一输入两次编译可能产出不同的计划编号。
- **runtime/AscendC 生命周期约束**：需求 4 选用的失败手段要在 kernel 上下文内合法。

## 兼容性检查

- 生成的 kernel 源码与 TilingData 结构变化（多计划），但二者都是每次编译重新生成的中间产物，
  没有持久化的旧产物需要兼容。
- `AF_MLIR_PINNED_PLAN_ID` 的裁剪行为不变。它自 2026-10-02 起需要 ``
  才被采信（默认关），仓内的调试脚本与 e2e 注册已随之同步，见
  [钉 plan / 钉 template 的总闸](2026-10-03-mlir-plan-pinning-switch-design.md)。
- 依赖固定 TilingData 布局的既有用例（`afir-to-ascendc-gen-tiling-data.mlir` 等）是单计划输入，
  不受影响。

## 测试设计

### 测试边界

**入口**：`RunMlirCodegen`（UT 经 `CodegenOrchestrationTest` 的既有夹具）、`af-opt` 的聚合 pass
（lit）、实机 e2e。**出口**：生成的 kernel 源码、TilingData 结构、`MlirCodegenStatus`。
**打桩**：让某个计划下降失败，用一个 MLIR 侧确定不支持的形态构造（不改产品码打桩）。

### 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|----------|------------|----------|----------|
| 功能 | T1：ATT 在多计划输入下生成的 host tiling 确实逐个试、跳过失败 | 实机跑 RCore 大 A 形状，dump host tiling 源码确认 `GetTiling` 的多候选循环 | ST |
| 功能 | T2：静态形状下产物包含全部计划 | `CodegenOrchestrationTest` 断言 `plan_entries` 长度 | UT |
| 功能 | T3：`tiling_key` 分发的 case 数与 `plan_entries` 一致 | lit：两计划输入，检查 `scf.index_switch` 的 case 数 | ST |
| 异常 | T4：一个候选下降失败 ⇒ 丢弃它，产物只含另一个，且 tiling 计划集与之一致 | UT | UT |
| 异常 | T5：候选全部下降失败 ⇒ `kFailed`，不产出半成品 | UT | UT |
| 异常 | T6：分发默认分支不是空操作 | lit：`CHECK-NOT` 空 yield，`CHECK` 失败指令 | ST |
| 性能 | T7：N=2 时端到端编译时长 | 手工计时对比 | Benchmark |
| 兼容性 | T8：单计划图的产物与变更前逐字一致 | 既有 lit 全量 | ST |
| 特性交叉 | T9：RCore 大 A 形状从「整图失败」变为「跑通（走 RCommon）」 | 实机 e2e，新增形状 | E2E |
| 特性交叉 | T10：`` + `AF_MLIR_PINNED_PLAN_ID` 仍然只产出被钉住的那个计划 | 既有 e2e 注册 | E2E |

### 测试命令

```bash
# 容器内
cmake --build build -j 8
ctest --test-dir build/autofuse/mlir --output-on-failure -j 4
cmake --build build --target check-lit -j 8

# 实机（910B）
bash /mnt/workspace/run_npu.sh "" load_sum_store_rcore.py rcore-fallback
```

预期：容器内 ctest 与 lit 全绿；实机上不指定 `AF_MLIR_PINNED_PLAN_ID` 时，大 A 形状由
RCommon 接管并通过，小 A 形状仍由 RCore 接管并通过。

## 验收标准

1. T1 证实 ATT 多计划求解器的容错语义；若证伪，本方案地基不成立，需回到「编译期跑 tiling
   判可行性」的备选方案重新设计。
2. 容器内 ctest 与 lit 全量通过；新增 T2～T6 五条用例。
3. 实机上**不指定** `AF_MLIR_PINNED_PLAN_ID`，`load_sum_store` 与 `load_sum_store_rcore` 全部
   形状通过；此前整图失败的 RCore 大 A 形状转为通过。
4. N=2 的图端到端编译时长不超过变更前的 1.6 倍。
5. 单计划图的生成产物与变更前逐字一致。
6. 所有改动的 C++ 文件通过 `clang-format -i`。
7. 本文档随代码合入；`AGENTS.md` 补一条「产物里为什么有多个计划、怎么看运行期选了谁」。

# 备选方案（未采纳）

**B：编译期跑一遍 host tiling 判可行性，再收窄。** 要先把每个计划的 tiling 代码编出来才能调用，
与「选完再生成」互为鸡蛋；且即便实现，也只换来编译期知道可行性，仍要丢掉 ATT 按 perf 选最优的
能力。成本高于方案 A 而收益更少。

**C：保留收窄，运行期失败后重编另一个计划。** 失败发生在 kernel init，此时已越过 inductor 的
编译缓存，重编译路径不存在；等于把编译期问题推成运行期重试。
