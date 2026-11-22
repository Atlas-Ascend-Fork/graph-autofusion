# Inductor 生产入口多子图聚合汇编设计

- 状态：设计中
- 作者：niuyuhu
- 关联分支：feat/mlir-inductor-production-entry

## 设计文档检查结果

### 特性交叉影响自查（参见 `docs/guidelines/cross_feature_check.md`）

| 维度 | 是否涉及 | 说明 |
| --- | --- | --- |
| 调度策略 | 是 | `StaticPlanSelector` 的多子图拒绝逻辑需要放开，见 5.4。 |
| Tiling 数据 | 是 | 复用现有 `GenAggregateTilingData` 聚合 Tiling Schema，不新增字段布局规则，见 5.2。 |
| 内存复用/工作区 | 否 | 工作区字段去重逻辑已由 `GenAggregateTilingData::addWorkspaceFields` 处理，本设计不修改。 |
| 同步指令 | 否 | `ascendc.need_sync_all` 的处理逻辑保持不变，本设计只搬运函数体，不改写函数体内容。 |
| Cube/CV 场景 | 否 | 现有 Gen* Pass 均已拒绝 `afir.cube_type != -1`，本设计不解除该限制。 |
| ABI/符号命名 | 是 | 聚合入口函数需要按现网命名规则重命名，见 5.3、7。 |
| 编译流水线接入 | 是 | 新增 Pass 需要接入 `addAFIRToAscendCPipeline`，见 5.3。 |
| 测试基础设施 | 是 | 需要新增多子图场景的 UT，见 9。 |
| 性能 | 是 | 见 4。 |
| 兼容性 | 是 | 见 8。 |

### 编码红线自查（参见 `docs/guidelines/编码红线.md`）

19 条红线逐项确认：本设计新增/修改的代码均为 Pass 内部逻辑（函数体克隆、符号改名、`StaticPlanSelector` 内部循环化），不引入全局可变状态、不引入裸指针生命周期悬挂、不新增未受控递归、不引入线程不安全的静态变量、不绕过现有的属性契约校验（`afir.module_kind`/`ascendc.plan_entries`/`afir.cube_type` 等仍在克隆前完整校验）、错误路径统一走 `LogicalResult`/`FailureOr` 并 `emitError`，不使用 `assert` 替代业务校验，不引入新的第三方依赖，命名与现有 Pass 保持一致风格。第 9 章列出具体测试用例覆盖失败路径，满足红线中"新增功能必须有对应失败路径测试"的要求。

### 测试设计自查

见第 9 章，覆盖单子图（回归）、双子图、子图间参数去重冲突、非法聚合契约、`StaticPlanSelector` 多子图选优五类场景。

### 性能评估自查

见第 4 章：本设计不引入运行时开销（聚合分发逻辑复用现有 `scf::IndexSwitchOp` 机制），编译期开销为线性于子图数量的函数体克隆操作。

## 1 简介

### 1.1 背景

Autofuse 编译链正在从 ASCIR（legacy 图算融合 IR 与代码生成体系）迁移到 MLIR（Autofuse 编译链迁移到 MLIR，以下简称"MLIR 迁移"）。MLIR 迁移中，AFIR Dialect 负责表达一份具体调度方案到 AscendC 的转换。当前 AFIR→AscendC 转换链路（`conversion/AFIRToAscendC/` 下的 Pass 集合）已经具备将**单个子图（subgraph）的单个调度方案**转换为一个可独立编译的 AscendC 核函数入口的能力，但尚未具备将 **N（N≥1）个子图各自选定的调度方案聚合为一个统一核函数入口**的能力，而 Inductor 生产场景中一次 Codegen 请求可能同时包含多个融合子图。

### 1.2 目的

本设计描述如何在现有 AFIR→AscendC Pass 体系基础上，新增最少必要的组件，使 N 个子图的调度方案能够被聚合为一个可独立编译、可被 `AutofuseLaunch` 直接调用的单一 AscendC 核函数源码，同时保持 N=1（单子图）场景与现网行为一致。

### 1.3 范围

- 覆盖范围：AFIR ModuleOp 级别的聚合、聚合入口函数体的生成、聚合入口函数到 AscendC 文本的汇编、`StaticPlanSelector` 对多子图输入的选优。
- 不覆盖范围：Cube/CV 融合场景（现有 Gen* Pass 已显式拒绝，本设计不解除该限制）；子图间的运行时依赖调度（每个子图的调度方案相互独立，聚合只做符号级合并，不引入子图间的数据依赖分析）；`AFIRKernelBodySignaturePass` 及 4-Pass 生产管线（`AFIRLoopGenerate`/`AFIRPipelineSync`/`AFIRMemoryAllocate`/`ConvertAFIRToAscendC`）本身尚未接入 `Codegen::GenerateForInductor`，此接入工作不在本设计范围内，本设计只保证聚合入口生成阶段本身是正确、可独立验证的。

### 1.4 术语

| 术语 | 含义 |
| --- | --- |
| 计划（Plan） | 调度器为一个子图选出的一种具体调度方案，对应一个 AFIR ModuleOp。当前代码使用 `afir.module_kind = "plan"` 标识计划模块，使用 `PlanInfo`/`plan_main` 等命名；聚合模块使用 `ascendc.plan_entries` 保存子图、计划和 Tiling 字段的映射。 |
| 子图（subgraph） | Inductor 侧一次 Codegen 请求中参与融合的一个独立计算图分区，对应属性 `afir.subgraph_id`。 |
| 聚合入口模块（Entry Module） | 本设计生成的、包含 N 个子图对应计划的聚合 Tiling Schema 与聚合 `main` 函数的单一 AFIR ModuleOp，当前代码中对应属性 `afir.module_kind = "entry"`。 |
| KernelEntry / KernelBody | `afir::FunctionRole` 枚举的两个取值，分别表示"可被外部 `<<<>>>` 调用的核函数入口"与"入口内部调用的普通函数"。 |
| Ascend Target Profile | 描述具体运行硬件的目标配置（如 AI Core 数量、内存层次参数），本设计不涉及其定义，仅引用其已有的读取方式。 |

## 2 总体概述

### 2.1 现状（当前代码事实）

- `GenAggregateTilingData.cpp` 已经能够正确地将 N 个计划的 Tiling Schema 聚合为一个 `AutofuseTilingData` 结构体：当子图数或计划数大于 1 时为每个子图生成一个 `graph<N>_tiling_key` 选择字段，去重聚合所有子图的 `workspace<id>` 字段，并将每个计划自身的（已经包含逐 Stage 嵌套结构的）Tiling 结构体以 `graph<subgraph_id>_result<plan_id>_tiling_data` 为字段名整体嵌入。逐 Stage 粒度的 Tiling 数据由每个计划自身的 Tiling 结构体内部嵌套保留，聚合层不需要、也没有丢失这一粒度。该 Pass 同时把子图与计划的坐标关系写入聚合模块的 `ascendc.plan_entries` 属性，供后续消费。
- `GenAggregateKernelEntry.cpp` 已经能够正确地为 N 个子图生成分发骨架：为每个 `ascendc.plan_entries` 条目在聚合模块内创建一个与对应计划的 `<计划符号>_main` 同签名的**无函数体私有声明**，按参数 key 去重合并为聚合 `main` 函数的参数列表，并在聚合 `main` 函数体内按子图分组生成 `scf::IndexSwitchOp`（多子图场景下依次生成 N 个独立的 `IndexSwitchOp`，只有唯一声明时退化为直接调用）。该分发骨架已经是正确的 N 子图实现，不需要重新设计。
- 上述两个 Pass 生成的聚合模块中，`<计划符号>_main` 均为**无函数体的私有声明**（`declaration.setPrivate()`，未克隆函数体）。以当前编译 pipeline 是否加载了对应 conversion pattern 为准，`externals/pyasc/lib/Target/AscendC/External/Func.cpp` 的 `func::FuncOp` 打印逻辑不区分"有函数体"与"无函数体"两种情况：只要该 Op 存在，就会打印出完整的 `{ ... }` 函数框架；对于零 Block 的声明，打印结果是一个**空函数体的桩函数**，而不是前置声明或链接错误。这意味着如果直接对当前 `GenAggregateKernelEntry.cpp` 产出的聚合模块调用 `translateToAscendC`，会静默生成语义错误（调用了空函数体）而非编译报错的 AscendC 源码——这是当前链路唯一的功能缺口。
- `StaticPlanSelector.cpp` 的 `ValidateStaticSelectionInput`/`ValidatePlanSelection`/`SelectDynamicCandidates`/`SelectStaticWinner` 四个函数目前均以 `fused_result.node_idx_to_scheduled_results.size() != 1U` 或 `.front()`/`model_infos.find(0U)` 的方式硬编码只处理下标为 0 的单个子图，遇到多子图输入时前三者返回错误信息 `"multi-subgraph fusion is not supported by the MLIR candidate selector"`，`SelectStaticWinner` 只会为该单个子图填充 `selections` 出参。

### 2.2 设计方向（本次设计内容）

保留 `GenAggregateTilingData`/`GenAggregateKernelEntry` 的现有实现不做重写，仅新增一个组件，并对 `StaticPlanSelector` 做等价的内部泛化：

1. 新增 `GenAggregateKernelBody` Pass：在 `GenAggregateKernelEntry` 完成分发骨架构建之后运行，将聚合模块中每一个 `<计划符号>_main` 私有声明替换为其在原计划模块中的**完整函数体克隆**，使聚合模块成为不依赖外部符号、可独立编译的翻译单元。
2. 在函数体克隆完成后，将聚合 `main` 函数提升为 `afir::FunctionRole::KernelEntry`（复用 `setAscendCKernelAttributes`），并按现网规则重命名为 `graph_name = CamelToLowerSneak(GenValidName(fused_graph_name))`——与现有单子图链路对单个计划 `main` 的改名机制完全一致，只是作用对象从"单个计划的 main"变为"聚合 Entry 模块的 main"。
3. `StaticPlanSelector` 的四个函数保持函数签名不变（`SelectStaticWinner` 已经以 `std::vector<PlanSelection> &` 出参设计，天然支持多个子图），仅将内部的"拒绝多子图"与"硬编码子图 0"逻辑改为对 `node_idx_to_scheduled_results` 的循环，每个子图独立选优后追加一个 `PlanSelection` 到出参。

单子图（N=1）场景下，`GenAggregateTilingData` 因 `graphCount==1 && infos.size()==1` 跳过选择字段生成，`GenAggregateKernelEntry` 因唯一声明退化为直接调用，`StaticPlanSelector` 的循环体只执行一次迭代——三者均与现网行为逐位一致，不存在行为回归风险。

### 2.3 设计约束

- C1：不修改 `GenAggregateTilingData.cpp`/`GenAggregateKernelEntry.cpp` 的现有实现，只在其输出之上新增一个后置 Pass。
- C2：新增 Pass 不得引入子图间的数据依赖分析——每个子图的计划在运行时仍是相互独立调用的普通函数体拷贝，聚合只是符号与调用点的合并。
- C3：Cube/CV 融合场景维持现状不支持，本设计不解除 `afir.cube_type` 检查。
- C4：聚合入口函数的 ABI（参数顺序：input → output → workspace → TilingData）与命名规则必须与现网单子图链路保持一致，使 `AutofuseLaunch` 侧的调用点生成逻辑不需要区分单/多子图。
- C5：计划模块统一使用 `afir.module_kind="plan"`，聚合契约统一使用 `ascendc.plan_entries`，并在 C++ 内部使用 `PlanInfo`、`PlanDeclaration` 等与四级概念一致的命名。

### 2.4 假设与依赖

- A1：聚合契约（`ascendc.plan_entries`：`subgraph_id`/`plan_id`/`tiling_field` 三元组数组）与各计划模块的 `afir.subgraph_id`/`afir.plan_id`/`afir.module_kind` 属性已经由 `AfirBuilder.cpp` 中的 `PlanToAfir` 正确产出。
- A2：每个计划模块的 `<计划符号>_main` 函数体在聚合发生前已经完整生成（由现有 `GenKernelEntry.cpp` 产出），本设计只做搬运，不生成新的函数体逻辑。
- A3：聚合发生时，所有参与聚合的计划 ModuleOp 与 Entry ModuleOp 处于同一个 `MLIRContext` 下（与 `GenAggregateTilingData::runGenAggregateTilingData` 现有的同 Context 校验假设一致）。

## 3 需求分析与设计

### 3.1 功能需求

| 编号 | 需求描述 | 优先级 |
| --- | --- | --- |
| F1 | 支持将 N（N≥1）个子图各自选定的计划聚合为一个可独立编译的 AscendC 核函数源码。 | P0 |
| F2 | 聚合后的核函数入口函数必须被标记为 `FunctionRole::KernelEntry` 并按现网规则命名。 | P0 |
| F3 | N=1 场景下聚合结果与现网单子图链路产出逐位一致。 | P0 |
| F4 | `StaticPlanSelector` 能够为 N 个子图分别独立选优，产出 N 个 `PlanSelection`。 | P0 |
| F5 | 聚合过程中发现的契约错误（缺失计划、聚合契约与计划模块不匹配等）必须以 `LogicalResult` 失败并携带可定位的诊断信息返回，不得静默产出错误源码。 | P0 |

### 3.2 关键设计点

#### 3.2.1 函数体克隆而非重新生成

`GenAggregateKernelEntry.cpp` 已经建好了每个 `<计划符号>_main` 声明到聚合参数的映射关系（`PlanDeclaration::argumentMapping`）。新增的 `GenAggregateKernelBody` 只需要：对聚合模块中每一个带有聚合契约对应关系的私有声明，找到其在原计划模块中的实际定义（`planModule.lookupSymbol<func::FuncOp>(name)`），将该定义的 Region 通过 `IRMapping`/`clone` 整体克隆进声明对应的 FuncOp 中，替换掉零 Block 的声明体。由于参数签名在 `createPlanDeclaration` 阶段已经保证与原函数一致，克隆不需要重写函数体内部的任何 Value 引用。

#### 3.2.2 KernelEntry 提升与命名

克隆完成后，聚合模块只有一个函数需要对外可见：聚合 `main`。其余被克隆函数体的 `<计划符号>_main` 声明保持 `private`、保持 `FunctionRole` 未设置（或显式设为 `KernelBody`，与现有 4-Pass 生产管线对 `KernelBody` 的处理保持一致，即使当前该管线尚未接入生产也不产生冲突）。聚合 `main` 调用 `setAscendCKernelAttributes(main, FunctionRole::KernelEntry, inputCount)`，并重命名为 `graph_name`，命名规则沿用现有单子图链路：`CamelToLowerSneak(GenValidName(fused_graph_name))`。

#### 3.2.3 StaticPlanSelector 的循环化

四个函数当前对 `node_idx_to_scheduled_results` 的处理方式统一改造为：

```cpp
for (size_t subgraph_id = 0; subgraph_id < fused_result.node_idx_to_scheduled_results.size(); ++subgraph_id) {
  const auto &plans = fused_result.node_idx_to_scheduled_results[subgraph_id];
  // 原先针对 .front() / node_idx_to_scheduled_results[0] 的逻辑，替换为 subgraph_id 参数化
  ...
  selections.push_back(std::move(selection));  // SelectStaticWinner 场景
}
```

`model_infos.find(0U)` 相应改为 `model_infos.find(subgraph_id)`；校验失败时的诊断信息去掉"multi-subgraph fusion is not supported"，改为定位到具体子图下标的错误信息。函数签名（包括 `selections` 出参类型 `std::vector<PlanSelection> &`）保持不变。

## 4 非功能需求

### 4.1 可维护性

新增逻辑集中在一个新 Pass（`GenAggregateKernelBody`）与 `StaticPlanSelector` 内部循环体中，不修改已验证正确的 `GenAggregateTilingData`/`GenAggregateKernelEntry` 实现，维护面最小。

### 4.2 可测试性

新增 Pass 与 `StaticPlanSelector` 泛化后的循环体均可独立于生产管线接入进度进行单元测试（延续现有 `af-opt`/UT 驱动的测试方式）。

### 4.3 可移植性

不引入平台相关代码，函数体克隆与符号重命名均为标准 MLIR API 操作。

### 4.4 可靠性

克隆前对聚合契约与各计划模块的匹配关系做完整性校验（数量一致、坐标一致、符号存在），任一校验失败即返回 `failure()` 并 `emitError`，不产出半成品源码。

## 5 性能

### 5.1 编译时长

新增的函数体克隆操作与子图数量 N 成线性关系，单次克隆的代价与被克隆函数体规模相同（不涉及重新分析或重新生成 IR），预期对整体编译时长的增量可忽略。

### 5.2 执行性能

聚合后的核函数在运行时的分发路径（`scf::IndexSwitchOp` + 直接调用）与克隆前的每子图独立核函数在语义与开销上完全一致，本设计不引入额外的运行时分支或数据拷贝。

### 5.3 内存与产物大小

聚合后单个核函数源码文件包含 N 个子图的全部函数体，产物大小随 N 线性增长，与"每子图单独出一个核函数文件"的方案相比总大小相同，只是合并进单一翻译单元。

## 6 接口设计

### 6.1 新增接口

| 接口 | 描述 |
| --- | --- |
| `mlir::afir::createGenAggregateKernelBodyPass()` | 新增 Pass 工厂函数，返回 `GenAggregateKernelBody` Pass 实例。 |
| `mlir::afir::registerGenAggregateKernelBodyPass()` | 新增 Pass 注册函数，供 `af-opt` 与生产管线构建器调用。 |

### 6.2 修改接口

| 接口 | 修改内容 |
| --- | --- |
| `StaticPlanSelector::ValidateStaticSelectionInput` | 移除多子图拒绝分支，改为按子图循环校验。 |
| `StaticPlanSelector::ValidatePlanSelection` | 同上。 |
| `StaticPlanSelector::SelectDynamicCandidates` | 同上。 |
| `StaticPlanSelector::SelectStaticWinner` | 移除 `.front()`/`find(0U)` 硬编码，按子图循环填充 `selections` 出参，函数签名不变。 |

### 6.3 接口检查项

| 检查项 | 结论 |
| --- | --- |
| 是否为 ABI 破坏性修改 | 否，`StaticPlanSelector` 四个函数签名均不变。 |
| 是否需要新增依赖 | 否。 |
| 是否影响现网单子图行为 | 否，N=1 时行为逐位一致（见 3.2、8）。 |

## 7 软件设计

### 7.1 关键数据结构

沿用现有 `PlanDeclaration`（`GenAggregateKernelEntry.cpp`）与 `AggregateContract`，新增 Pass 不引入新的持久化数据结构，仅在 Pass 内部使用局部的"声明符号 → 原始定义 FuncOp"映射表完成克隆定位。

### 7.2 关键流程

```
GenAggregateTilingData(计划模块集合) -> EntryModule(tiling schema + 聚合契约)
GenAggregateKernelEntry(计划模块集合, EntryModule) -> EntryModule(聚合 main + N 个空函数体 <计划符号>_main 声明)
GenAggregateKernelBody(计划模块集合, EntryModule)   [新增]
  for each <计划符号>_main declaration in EntryModule:
    locate original definition in matching 计划模块 via 聚合契约
    clone its body into the declaration
  promote EntryModule.main to FunctionRole::KernelEntry, rename to graph_name
translateToAscendC(EntryModule) -> 可独立编译的 AscendC 源码
```

### 7.3 对子模块的修改

- `conversion/AFIRToAscendC/`：新增 `GenAggregateKernelBody.h`/`.cpp`，接入 `addAFIRToAscendCPipeline` 构建器，位置在 `GenAggregateKernelEntry` 之后。
- `bridge/StaticPlanSelector/`：`StaticPlanSelector.cpp` 内部循环化改造，`StaticPlanSelector.h` 不变。

### 7.4 错误处理

- 克隆阶段找不到匹配的原始定义（符号不存在、聚合契约坐标与实际计划模块不匹配）：返回 `failure()`，在声明位置 `emitError`，携带缺失的 `subgraph_id`/`plan_id` 信息。
- `StaticPlanSelector` 某个子图下标在 `model_infos` 中找不到对应条目：定位到具体子图下标返回错误，不再使用统一的"多子图不支持"提示。

## 8 兼容性检查

- 单子图输入（N=1）：`GenAggregateTilingData`/`GenAggregateKernelEntry` 现有的 N=1 简化路径不变；新增 `GenAggregateKernelBody` 对唯一声明做同样的克隆操作，行为与"直接对单个计划做 KernelEntry 提升"完全一致；`StaticPlanSelector` 循环体只执行一次迭代，选优结果与现网一致。
- 现有 4-Pass 生产管线（`AFIRLoopGenerate`/`AFIRPipelineSync`/`AFIRMemoryAllocate`）仅处理 `FunctionRole::KernelBody`，本设计克隆产生的每个子图函数体默认不设置或设置为 `KernelBody`，与该管线的现有处理方式兼容，不产生冲突。

## 9 测试设计

### 9.1 测试边界

- 覆盖：聚合 Tiling Schema 正确性（N=1、N=2）、聚合分发骨架正确性（延续现有 `GenAggregateKernelEntry` UT）、函数体克隆正确性、KernelEntry 提升与命名正确性、`StaticPlanSelector` 多子图选优正确性、契约错误路径。
- 不覆盖：Cube/CV 场景（现有校验已拒绝，无需新增用例）、生产管线接入后的端到端性能测试（管线本身尚未接入生产，不在本设计范围）。

### 9.2 测试用例

| 用例 | 输入 | 期望结果 |
| --- | --- | --- |
| 单子图聚合回归 | 1 个计划模块 | 聚合结果与现网单子图链路逐位一致 |
| 双子图聚合 | 2 个计划模块，各自不同 `subgraph_id` | 聚合 `main` 函数体不再包含空函数体声明，可通过 `translateToAscendC` 生成完整可编译源码 |
| 参数 key 冲突 | 2 个计划模块存在同名不同类型的参数 key | 返回 `failure()` |
| 聚合契约与计划模块不匹配 | 聚合契约中的坐标在计划模块集合中找不到对应模块 | 返回 `failure()`，诊断信息包含缺失坐标 |
| `StaticPlanSelector` 多子图选优 | `node_idx_to_scheduled_results.size() == 2` | `selections` 出参包含 2 个 `PlanSelection`，各自对应正确的子图与最优模板 |

### 9.3 测试命令

```bash
cmake --build build --target af-mlir-unittests
./build/af-mlir-unittests --gtest_filter='*AggregateKernelBody*:*StaticPlanSelector*'
```

## 10 验收标准

- N=1 与 N≥2 场景下，聚合入口模块经 `translateToAscendC` 均能生成不含空函数体桩函数的完整 AscendC 源码。
- N=1 场景下产出与现网单子图链路逐位一致（无行为回归）。
- `StaticPlanSelector` 四个函数在多子图输入下不再返回"multi-subgraph fusion is not supported"错误，能够为每个子图独立选优。
- 第 9 章所列测试用例全部通过。
