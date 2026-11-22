# Inductor 生产入口接入 MLIR Codegen 与 bisheng 编译执行链路设计

- 状态：已实现（完整 NPU/E2E 验收按第 10 节执行）
- 作者：niuyuhu
- 关联分支：af-mlir

## 1 简介

### 1.1 背景

Autofuse 面向 Inductor 的生产编译入口 `Codegen::GenerateForInductor`（`autofuse/codegen/codegen.cpp:394`）目前完全走 legacy ASCIR 路径生成 kernel/tiling 文本。MLIR 迁移已经分阶段落地了以下能力，但彼此之间、以及与 `GenerateForInductor` 之间尚未串联：

- 调度候选选优：`StaticPlanSelector`（`mlir/bridge/StaticPlanSelector/StaticPlanSelector.h`）已支持静态/动态 shape 的多子图候选选优。
- ATT 接入：`AttSchemaAdapter`（`mlir/bridge/AttSchemaAdapter/AttSchemaAdapter.h`）已支持对选优后的候选批量调用 ATT 生成 tiling 源码。
- 单 Plan 到 AFIR：`ScheduleToAfir::PlanToAfir`（`mlir/bridge/ScheduleToAfir/ScheduleToAfir.h`）已支持把一个 `ScheduledResult` 转换为一个 AFIR `ModuleOp`。
- AFIR→AscendC 四段生产 Pass：`addAFIRToAscendCPipeline`（`mlir/conversion/AFIRToAscendC/AFIRToAscendCPipeline.h`）已实现 `AFIRLoopGenerate`→`AFIRMemoryAllocate`→`SelectImplementation`→`ConvertAFIRToAscendC`。
- 多子图聚合：`GenAggregateTilingData`/`GenAggregateKernelEntry`/`GenAggregateKernelBody`（`mlir/conversion/AFIRToAscendC/`）已实现把 N 个子图各自选定的 Plan 模块聚合为一个可独立编译的单一核函数入口（`2026-08-24-inductor-production-entry-multi-subgraph-aggregate-design.md`，已合入 `30685413`）。
- AscendC 文本发射：`translateToAscendC`（`mlir/api/ascendc/include/ascir/Target/Asc/Translation.h`，签名为 `LogicalResult translateToAscendC(Operation *op, llvm::raw_ostream &os)`——写入流而非直接返回字符串，调用方需用 `llvm::raw_string_ostream` 收集文本）已支持把聚合后的 Entry Module 翻译为完整可编译的 AscendC 设备源码。该组件原为对第三方 PyAsc 仓库的外部依赖（旧路径 `mlir/externals/pyasc/...`），现已随 `8485c128` 提交整体拷贝进本仓库 `autofuse/mlir/api/ascendc/` 并独立演进，不再是外部依赖；对应 CMake 目标为 `MLIRTargetAsc`（`autofuse/mlir/api/ascendc/lib/Target/AscendC/CMakeLists.txt:11`）。本文其余各处对 `externals/pyasc` 路径的引用均已按此更正（见 3.5/7.4/6.1 节）。

`2026-08-18-schedule-space-att-mlir-inductor-architecture.md` 第 3.2.2 节明确指出"生产入口如何组装候选"这一段仍未解决，并将其显式委托给"本文（或新开一份设计文档）"。本文即为该委托的落地设计：把上述已实现组件串联进 `GenerateForInductor`，补齐 TargetProfile 构造、`AutofuseLaunch` host wrapper 生成两个缺失环节，并最终打通 bisheng 编译与 Inductor 调用执行。

### 1.2 目的

在不改变 `ENABLE_AUTOFUSE_MLIR=OFF` 时行为、不改变现网单子图/CV 场景行为的前提下，新增一条可显式开启的 MLIR 编排路径，使：

1. `Codegen::GenerateForInductor`（Inductor 场景）与 `Codegen::Generate`（TensorFlow 场景入口；该方法当前在生产代码中无实际调用方，见 3.2 节说明与已知限制）均能够选择走同一条统一 MLIR 编排入口 `Codegen::CodegenViaMlir` 生成 `CodegenResult.kernel`/`tiling`/`tiling_data`；
2. 生成的 kernel 文本可直接被 `autofuse/compiler/python/ascendc_compile.py` 现有 bisheng 编译驱动（`compile_device_obj`/`build_device_so`/`link_shared`）编译为 `.so`；
3. 生成的 `AutofuseLaunch` 符号与现网签名/调用约定完全一致，使 Inductor/torchair 侧调用点无需任何修改即可执行该 kernel。

初期验证范围可以是单子图 + 静态 shape，但设计本身覆盖 N 子图（N≥1）与静态/动态 shape 两种候选选优模式，不做场景定制化分支。

### 1.3 范围

- 覆盖范围：Codegen 编排入口新增、`ascend::TargetProfile` 构造与线程、`AutofuseLaunch` host wrapper 文本生成、ATT tiling 产物与聚合 Tiling Schema 的拼装、`CodegenOptions`/`AutofuserOptions` 新增可选字段、CMake 链接补齐。新增的编排入口统一服务 `GenerateForInductor`（Inductor）与 `Codegen::Generate`（TensorFlow）两处调用点，不新增第三条调用路径（见 3.7 节）。**已知限制**：`Codegen::Generate` 当前在生产代码中没有实际调用方（`pyautofuse.cpp` 的 `HandleDeviceCodeGenForNonCVFusion`/`HandleHostCodeGenForNonCVFusion` 直接调用 `GenerateTilingData`/`GenerateKernel`/`GenerateTiling`，不经过 `Generate`，见 3.2 节），因此在 `Generate` 内挂钩对现网真实 TensorFlow 流量暂不生效；本文按用户明确指示只在 `Codegen::Generate`/`GenerateForInductor` 两处挂钩，是否打通 `pyautofuse.cpp` 到 `Generate` 的调用链路是另一个独立问题，不在本文范围内解决。
- 不覆盖范围：`StaticPlanSelector`/`AttSchemaAdapter`/`GenAggregateTilingData`/`GenAggregateKernelEntry`/`GenAggregateKernelBody`/四段生产 Pass 本身的实现（均已完成，本文只描述其调用方式）；Cube/CV 融合场景（现有 Gen* Pass 已拒绝，本文不解除该限制）；`ascendc_compile.py`/`compile_adapter.py` 内部实现（复用不修改）；torchair/inductor-npu-ext 仓库代码；`pyautofuse.cpp` 内 `HandleDeviceCodeGenForNonCVFusion`/`HandleHostCodeGenForNonCVFusion` 到 `Codegen::Generate` 的调用链路打通（见上，另一个独立问题）。

### 1.4 术语

沿用 `2026-08-18-schedule-space-att-mlir-inductor-architecture.md`、`2026-08-24-inductor-production-entry-multi-subgraph-aggregate-design.md` 已定义的术语（Plan/Subgraph/聚合入口模块/KernelEntry/KernelBody/Target Profile）。新增：

| 术语 | 含义 |
| --- | --- |
| MLIR 编排入口 | 本文新增的 `Codegen::CodegenViaMlir`（原方案名 `GenerateViaMlir`，因返回三态状态码而更名，见 3.7/6.1 节），统一服务 `GenerateForInductor` 与 `Codegen::Generate` 两个调用点，按 `CallerScene`（`kInductor`/`kTensorFlow`）区分场景专属细节（如向 ATT/`PlanToAfir` 转发的 `is_inductor` 语义），但选优/转换/Pass/聚合主流程不按场景分叉。返回值为专用三态枚举 `MlirCodegenStatus{kNotSupported, kFailed, kSucceeded}`：`kNotSupported`（未开启开关、命中 Cube 融合场景、或其他"本次不适用"情形）时调用方继续执行各自 legacy 路径；`kFailed`/`kSucceeded` 时调用方直接以对应的 `af::Status` 返回，不再回退（见 3.7 节）。原诊断钩子 `MaybeDumpAfirFromSchedule` 已删除，其 dump 能力收编进本入口内部实现。 |
| AutofuseLaunch host wrapper | 本文新增组件生成的 `extern "C" int64_t AutofuseLaunch(...)` 函数文本，逐字节复用 `codegen_kernel.cpp:4515` 起的现网 ABI。其形参布局从聚合 `main`（`ascendc.fused_main` 标注）按 `mlir::afir::KernelArgumentIdentity`（`input_index`/`output_index`/`workspace`）与末位 tiling 结构体参数推导，见 3.6 节。 |

## 2 总体概述

### 2.1 软件概述

#### 2.1.1 项目介绍

本需求是 Autofuse MLIR 迁移中"让 MLIR Codegen 真正产出生产可用的 Inductor kernel"这一目标的收口设计。此前的设计文档（07-27/08-05/08-13/08-18/08-24）已经分别完成了调度候选选优、Tiling Schema、AFIR→AscendC Pass、多子图聚合的独立设计与实现，但都止步于"生成一个可独立编译的 AscendC 设备源码"，尚未接入生产入口 `GenerateForInductor`，也没有 host 侧 launch wrapper 和编译驱动的接入设计。本文是这条链路的最后一段拼装。

#### 2.1.2 产品环境介绍

- 依赖 CANN Toolkit 提供的 bisheng 编译器与 `<<<>>>` kernel launch 机制，本文不改变该 ABI（见 08-18 doc 2.1.2 已有结论，本文直接复用）。
- 编译驱动复用 `autofuse/compiler/python/ascendc_compile.py`（`get_soc_type`/`compile_device_obj`/`build_device_so`/`link_shared`）与 `compiler/python/compile_adapter.py`（`jit_compile`/`kernel_compile`/`host_compile`），两者均已支撑现网 Inductor 编译，本文不修改其实现，只保证 `CodegenResult` 三个字符串字段（`tiling_data`/`tiling`/`kernel`）的内容契约与现网一致。
- `ENABLE_AUTOFUSE_MLIR=OFF` 时不引入 MLIR/LLVM 依赖，行为与当前 legacy 完全一致；`ENABLE_AUTOFUSE_MLIR=ON` 但未显式开启本文新增开关时，行为同样与当前 legacy 完全一致（新路径默认关闭）。

### 2.2 软件功能

1. 在 `Codegen::GenerateForInductor` 内新增一条可选的 MLIR 编排路径，串联候选选优→AFIR 转换→四段 Pass→多子图聚合→AscendC 文本发射→`AutofuseLaunch` host wrapper 生成，产出与 legacy 路径等价契约的 `CodegenResult`。
2. 新增 `ascend::TargetProfile` 的构造与传递：从既有 `soc_version` 字符串（新增到 `CodegenOptions`/`AutofuserOptions`）解析出 MLIR 侧 Target Profile，供四段 Pass 使用。
3. 新增 `AutofuseLaunch` host wrapper 文本生成组件，输出与现网 `codegen_kernel.cpp` 生成文本 ABI 一致的 launch 函数。
4. 保证上述产物无需修改即可被现有 `ascendc_compile.py`/`compile_adapter.py` 编译驱动消费，直至生成可被 Inductor 调用执行的 `.so`。

### 2.3 设计约束

- C2：新路径默认关闭（`ENABLE_AUTOFUSE_MLIR=OFF` 或未显式开启开关时），不得改变任何现网行为、产物内容或性能特征。
- C3：`AutofuseLaunch` 的参数顺序（`blockDim → stream → input... → output... → workspace → tiling_data`）、返回值语义（`int64_t`，成功 `0`，失败非 `0`）、`extern "C"` 链接方式与现网逐字节一致，调用点（Inductor/torchair 侧）不需要修改。
- C4：设计必须覆盖 N（N≥1）子图与静态/动态 shape 两种候选选优模式，不允许出现"只支持单子图"或"只支持静态 shape"的架构性限制；初期验证范围可以窄（单子图+静态 shape），但由调用方式（选择跑哪个测试用例）决定，不由代码分支决定。
- C5：`CodegenOptions`/`AutofuserOptions` 新增字段必须是可选、有安全默认值的增量修改，不改变现有构造函数/绑定的调用方式。

### 2.4 假设和依赖关系

- A1：`StaticPlanSelector::SelectStaticWinner`/`SelectDynamicCandidates`、`AttSchemaAdapter::RunAttForCandidates`、`ScheduleToAfir::PlanToAfir`、`addAFIRToAscendCPipeline`/`createAFIRToAscendCPipelineSession`、`GenAggregateTilingData`/`GenAggregateKernelEntry`/`GenAggregateKernelBody`、`translateToAscendC` 均已实现且行为符合各自设计文档描述，本文直接调用，不重新验证其内部正确性（各自的 UT 已覆盖）。其中 `RunAttForCandidates`/`PlanToAfir` 目前只有 UT 调用方，尚无生产调用方，本文是它们的首个生产编排者（见 3.3 节）——这意味着本文的编排代码同时承担了"这两个组件首次接受生产输入"的验证职责，9.2 节测试用例需覆盖比其自身 UT 更贴近真实生产输入分布的场景，不能只依赖其既有 UT 已通过来判定生产可用性。
- A2：`ascend::TargetProfile` 的构造入口 `TargetSelection::builtinPlatform(soc_version)` + `TargetProfileLoader::load` 已支持解析现有 `soc_version` 字符串格式（`Profiles/manifest.txt` 别名表已覆盖 Ascend910B*/Ascend910_93*/Ascend950* 全系列），无需新增映射表。
- A3：`ascendc_compile.py`/`compile_adapter.py` 的 `sources["kernel_impl_code"]`/`sources["tiling_struct_code"]`/`host_impl_code` 只做文本写入与编译驱动，不解析内容语义，因此 `CodegenResult` 三个字符串字段只要保持文本上可独立编译即满足契约，无需与 legacy 文本逐字节一致（`AutofuseLaunch` 的 ABI 一致性除外，见 C3）。
- A4：`GenAggregateKernelEntry.cpp` 内部的 `collectAggregateContract`/`KernelArgumentIdentity` 计算逻辑本身正确（有既有 UT 覆盖），3.4 节的接口扩展（导出为 `validateAggregateContract` 公开函数、`runGenAggregateKernelEntry` 返回类型扩展为 `FailureOr<AggregateArgumentLayout>`）只改变对外可见性与返回值组装方式，不改变内部计算过程，因此该扩展本身不需要重新验证聚合算法正确性，只需为新增的返回值组装路径补充针对性 UT（见 9.2 节）。
- A5：`Codegen` 类目前不保留 `CodegenOptions` 整体作为成员（构造函数把各字段拆分进独立私有成员），本文新增的 `soc_version` 字段需要在 `Codegen` 类内新增对应私有成员并在构造函数中初始化才能被 `GenerateForInductor`/`Generate` 读取到，不能假设"字段加进 `CodegenOptions` 结构体就自动可用"（见 6.1.2 节修正）。MLIR 编排路径的显式开关经与作者确认**不**采用 `CodegenOptions` 字段/`Codegen` 私有成员方案，而是改为进程环境变量 `AF_ENABLE_MLIR_CODEGEN`（见 6.1.2/3.7 节），因此不涉及该开关的成员存取问题，`CodegenViaMlir` 内部直接调用 `codegen.cpp` 内的 `IsMlirCodegenEnabled()` 读取环境变量。`soc_version` 这一挂钩点（`GenerateForInductor`/`Codegen::Generate`）均是 `Codegen` 自身的成员方法，可以直接读取新增私有成员，不涉及跨对象访问问题；`Codegen::Generate` 内挂钩对 TensorFlow 现网流量暂不生效是调用链路层面的已知限制（见 3.2/3.7 节），与本假设（成员变量可读性）无关。

## 3 需求分析与设计

### 3.1 整体介绍

链路按以下阶段串行执行，每个阶段直接调用一个已实现组件，本文只新增阶段之间的编排代码与两个缺失组件（TargetProfile 构造、AutofuseLaunch 生成）：

```
FusedScheduledResult（既有，legacy schedule 产出）
   │
   ▼
① 候选选优         StaticPlanSelector::SelectStaticWinner / SelectDynamicCandidates
   │  产出：std::vector<PlanSelection>（out 参数，每个元素含 subgraph_id/plan_id/template_ids_per_stage；
   │        静态 shape 每子图一个元素，动态 shape 每 (子图,plan) 一个元素）
   ▼
①′（本文新增）Tiling Schema 构造   common::BuildTilingFieldSchema(fusedResult)（或按 ScheduledResult 重载）
   │  产出：common::TilingFieldSchema（fields 列表 + version），作为②的输入参数
   ▼
② ATT tiling 生成  AttSchemaAdapter::RunAttForCandidates(op_name, fusedResult, selections, schema, options,
   │                                                       is_inductor_scene, /*out*/bundle, diagnostic)
   │  产出：TilingCodeBundle { tiling_functions: map<string,string>, selections, schema_version, schema_validated }
   │        （不含 TilingFieldSchema 本身，只回填其 version 字符串用于一致性校验，见 3.3 节修正）
   ▼
③ 单 Plan 转 AFIR    ScheduleToAfir::PlanToAfir(context, fusedResult, scheduledResult, subgraphId, planId, isInductor)
   │  产出：每个候选生成一个 OwningOpRef<ModuleOp>，标注 afir.subgraph_id / afir.plan_id
   ▼
④（本文新增）TargetProfile + Pipeline Session 构造（一次性，跨候选复用，见 3.5 节修正）
   │  ascend::TargetSelection::builtinPlatform(soc_version) → ascend::TargetProfileLoader::load
   │  → createAFIRToAscendCPipelineSession(context, targetProfile)（内部冻结 ImplementationRegistry）
   ▼
⑤ AFIR→AscendC 四段 Pass   对每个候选模块新建一个 PassManager，addAFIRToAscendCPipeline(pm, session) 后 pm.run(module)
   │  产出：每个候选模块内部完成 AFIRLoopGenerate→AFIRMemoryAllocate→SelectImplementation→ConvertAFIRToAscendC；
   │        Session（含冻结后的 ImplementationRegistry）在所有候选间共享，只需构造一次
   ▼
⑥ 多子图聚合       runGenAggregateTilingData → runGenAggregateKernelEntry → runGenAggregateKernelBody
   │  产出：单一 Entry ModuleOp；聚合 main 的实参通过 afir.kernel_arg 标注
   │  （勘误 2026-09-14：原为 afir.input_index/afir.output_index，已收敛成一个强类型
   │   属性 #afir.kernel_arg<role = ...>，见公开面重构设计文档 3.6）
   │        （workspace 按类型识别，无专属属性；tiling 为末位参数，按位置识别，见 3.4 节修正）；
   │        runGenAggregateKernelEntry 返回类型扩展为 FailureOr<AggregateArgumentLayout>，
   │        直接携带 (KernelArgumentIdentity 列表, tilingArgIndex)，供 ⑨ 消费，无需重新解析属性
   ▼
⑦ AscendC 文本发射  translateToAscendC(entryModule, ostream)（写入 llvm::raw_string_ostream，非直接返回字符串）
   │  产出：设备端 kernel 实现文本（KernelBody，含 tiling_key 分支）
   ▼
⑧（本文新增）AutofuseLaunch host wrapper 生成  基于 ⑥ 返回的 AggregateArgumentLayout 直接拼接文本
   │  产出：extern "C" int64_t AutofuseLaunch(...) 函数文本
   ▼
CodegenResult { tiling_data = ⑥ 聚合 PyStructType 经 `mlir::afir::emitTilingDataHeader` 渲染的结构体文本（见下方说明）,
                tiling      = ② TilingCodeBundle.tiling_functions 文本,
                kernel      = ⑦ 输出 + ⑧ 输出拼接 }
```

`CodegenResult.tiling_data` 复用既有 header 生成入口：`runGenAggregateTilingData` 在返回的 Entry module 中记录名为 `AutofuseTilingData` 的聚合 schema，编排层通过 `mlir::afir::TilingSchema` 读取，编排层在 ⑥ 之后调用 `mlir::afir::emitTilingDataHeader(ArrayRef<mlir::afir::TilingSchema>, llvm::raw_ostream &)`，将其渲染为 C++ 结构体文本。该函数声明于 `mlir/conversion/AFIRToAscendC/Emit/TilingDataHeaderEmitter.h`，实现位于同目录 `.cpp`，可独立消费 schema，无需挂载到 `Operation`。这与 `translateToAscendC` 经 `DeclarePyStructOp` 生成文本的路径不同；相关输出由 `IsolatedPlanEntryTest.cpp` 覆盖。ATT 的 `TilingCodeBundle` 只提供 tiling 计算源码，结构体字段文本由上述 schema 入口生成。

阶段 ①-⑦ 全部复用既有已实现、已测试组件（`RunAttForCandidates`/`PlanToAfir` 目前实际只有 UT 调用，尚无生产调用方，本文是它们的首个生产编排者，见 3.3 节），本文只新增：一个串联上述阶段的编排函数 `Codegen::CodegenViaMlir`（新增于 `autofuse/codegen/codegen.cpp`，与既有 `GenerateForInductor`/`Codegen::Generate` 同级，统一服务两者，见 3.7 节）、Tiling Schema 构造、TargetProfile+Session 构造代码、AutofuseLaunch host wrapper 生成组件；此外对 `GenAggregateKernelEntry.h`/`.cpp` 做小幅接口扩展（导出契约校验、扩展返回类型，见 3.4/7.4 节，原 C1 约束已放宽，见 2.3 节）。

两点已知的临时状态，本设计不消除，仅记录并在 ATT 完成自身 MLIR 迁移后重新评估：

- **② 先于 ③⑤ 执行是临时的依赖倒置**：旧 ATT 模块尚未迁移到消费 mlir-codegen 结果 IR，因此当前只能先跑②让 ATT 消费 `FusedScheduledResult`，再跑③⑤把同一份候选转成 AFIR 并执行四段 Pass；ATT 完成迁移后，预期顺序应调整为 ①→③⑤→②′（新 ATT 消费⑤产出的 IR）→⑥⑦，届时 ATT 不再需要独立于⑤重新计算循环层数/Load-Store buffer 大小等信息。
- **`tiling_data` 的产出机制经本次核查确认为两条完全独立、互不依赖的路径，而非"预期等价但未交叉校验"**：ATT 侧 `TilingCodeBundle` 只回填调用方传入 `TilingFieldSchema` 的 `version` 字符串，不产出字段级结构体文本；真正携带字段定义的聚合 Schema 是 `runGenAggregateTilingData` 从 AFIR 模块的 `afir.stage_id`/`afir.template_id` 等属性独立推导出的 `PyStructType`，与 ATT 完全无关。因此"二者预期内容等价"这一假设本身需要重新论证——两条路径的输入完全不同（一个来自 ATT 生成的 tiling 源码字符串集合，一个来自 MLIR 属性），没有共同的推导基础，不存在自然等价的理由。本设计确定：`CodegenResult.tiling_data` 只由⑥的聚合 `PyStructType` 驱动产出，ATT 侧的 `TilingCodeBundle.schema_version` 仅用于第②步内部一致性自检（见 3.3 节），不参与 `tiling_data` 拼装。ATT 完成迁移后，②本身可能整体收敛进⑤⑥，该问题自然消失。

### 3.2 关于编排代码放置位置的决策

`2026-08-18-schedule-space-att-mlir-inductor-architecture.md` 3.2.2 节遗留了"编排代码放在 C++ Codegen 层还是 Python orchestration 层"的开放问题。本文决策：**放在 C++ 内**，由 `Codegen::GenerateForInductor`（`codegen.cpp:394`，Inductor 生产路径）与 `Codegen::Generate`（`codegen.cpp:387`/`437`，TensorFlow 场景入口）在开关开启且非 Cube 融合场景时优先调用，失败或不满足条件时回退各自现有 legacy 实现（统一入口与调用点的具体设计见 3.7 节）。

> **后续修订（2026-08-31 追加）**：本节及以下 3.3-3.7、6.1、7.4 节最初把该编排入口设计为 `Codegen` 类的私有方法 `Codegen::CodegenViaMlir`，定义在 `autofuse/codegen/codegen.cpp`/`codegen.h` 内。落地过程中，考虑到该方法体量较大（编排候选选优、逐候选 Pass、聚合三个阶段），且其内部除入口守卫的两个判断外，其余逻辑均只依赖 `ascir::FusedScheduledResult`、MLIR 组件与 `soc_version` 字符串，不依赖 `Codegen` 类的其他私有成员，遂改为**自由函数** `mlir::bridge::RunMlirCodegen`，定义在 `autofuse/mlir/bridge/CodegenOrchestration/CodegenOrchestration.{h,cpp}`（与 `ScheduleToAfir`/`StaticPlanSelector`/`AttSchemaAdapter` 同级、同风格的 `mlir::bridge` 组件），随 `78fa250c`（"将 Inductor MLIR codegen 编排逻辑提取到 mlir/ 目录"）落地，是对本节"放在 C++ 内"决策的具体化，不是对决策本身的推翻——`Codegen::GenerateForInductor`/`Codegen::Generate` 仍是唯二调用点，调用时序、三态返回码语义（见 3.7 节）与本节的核心结论完全不变。伴随此次提取，原方法名与场景枚举也一并调整：`CodegenViaMlir` → `RunMlirCodegen`；`CallerScene` → `MlirCallerScene`（避免与其他潜在的通用 `CallerScene` 命名冲突，明确其只服务于 MLIR 编排入口）；原本通过 `this->soc_version_` 隐式读取的 `soc_version`，因编排逻辑不再是成员方法，改为显式形参传入。当前实际签名见 6.1 节更新后的接口描述；本节及 3.3-3.7 节正文中出现的 `Codegen::CodegenViaMlir`/`CallerScene` 均为历史设计快照，读者应以 6.1 节为准。

理由：

1. `GenerateForInductor` 是 Python（`AutofuserOptions`/`fuser.codegen(...)`）与 C++ 之间真实存在的生产调用边界，`Generate` 是 TensorFlow 场景在代码层面标识的对应入口；两者均是 `Codegen` 类自身的公开成员方法，Python 侧调用方式（对 `GenerateForInductor` 一侧）不需要感知内部走 legacy 还是 MLIR 路径，符合 C3/C5 的"调用点无需修改"约束。

   **需要记录的一个已知限制**：经核查，`Codegen::Generate`（`codegen.cpp:387`/`437`）在全仓库范围内只被测试文件调用；`pyautofuse.cpp` 的 `HandleDeviceCodeGenForNonCVFusion`（551 行起）/`HandleHostCodeGenForNonCVFusion`（638 行起）——这两个函数是 TensorFlow 场景现网真实生产入口——直接调用 `Codegen::GenerateTilingData`/`GenerateKernel`/`GenerateTiling` 这三个更底层的方法，并不经过 `Codegen::Generate`。也就是说，若只在 `Codegen::Generate` 内部插入 `CodegenViaMlir` 调用，该挂钩点在当前调用链路下不会被真实 TensorFlow 生产流量触发，环境变量 `AF_ENABLE_MLIR_CODEGEN` 开关对 TensorFlow 路线暂时形同虚设，直到 `pyautofuse.cpp` 的这两个 Handle 函数改为经过 `Codegen::Generate`（或另行在其内部单独挂钩）为止。**本文经与作者确认，按 Q7 原始指示只在 `Codegen::Generate`/`GenerateForInductor` 两处挂钩**，`pyautofuse.cpp` 到 `Codegen::Generate` 的调用链路打通是否需要、如何实施是另一个独立问题，明确不在本文范围内（见 1.3 节不覆盖范围）；本文只保证机制本身（三态返回码、内部开关判断）在 `Codegen::Generate` 内部是正确、可用的，一旦调用链路打通即可直接生效，不需要重新设计。
2. 候选选优、ATT 调用、AFIR 转换、四段 Pass、聚合、文本发射均是 C++ 组件，若编排逻辑放在 Python 层，需要把这些 C++ 组件逐一暴露到 Python 绑定，属于不必要的接口面扩大，违反编码红线通用规则 7（禁止无理由扩大修改范围）。
3. C++ 内编排可以直接复用 `FusedScheduledResult`（C++ 内部类型），不需要跨语言边界传递中间产物，减少序列化开销（对应第 4 章性能评估）。
4. `GenerateForInductor`（Inductor 场景）与 `Generate`（TensorFlow 场景入口）的现有 legacy 实现已经共用 `ascgen_utils::IsCubeFusedScheduled` 判断 Cube/CV 融合场景（`codegen.cpp:403`；TensorFlow 侧该判断在 `pyautofuse.cpp:698/748` 上游的场景分支中，属于第 1 点所述"暂不生效"范围的一部分，见上），说明"是否可以走新 pass 路径"本身与调用来源（Inductor/TensorFlow）无关，只与融合场景有关；因此新增编排入口不按调用方场景重复实现一套判断/编排逻辑，而是做成统一入口 `Codegen::CodegenViaMlir`，仅在必须区分场景的少数细节（如 3.3 节 `PlanToAfir` 的 `is_inductor` 语义）上通过参数区分，最大化复用、最小化对现有两条 legacy 路径的侵入式修改。

### 3.3 功能需求 1：候选选优结果到 AFIR 模块的转换编排

1. 介绍

   `Codegen::CodegenViaMlir` 接收调用方（`GenerateForInductor` 或 `Codegen::Generate`，见 3.2/3.7 节）已产出的 `FusedScheduledResult`（与各自 legacy 路径完全共享该输入，不重新跑 schedule），对其中每个子图调用 `StaticPlanSelector` 选出候选（静态 shape 场景为唯一 winner，动态 shape 场景为候选集合），再对每个候选依次调用 `AttSchemaAdapter::RunAttForCandidates`（需先构造 `TilingFieldSchema` 作为其输入，见 3.1 节①′）批量生成 tiling 源码，然后对每个候选调用 `ScheduleToAfir::PlanToAfir` 转换为独立 AFIR `ModuleOp`。

2. 输入

   - `FusedScheduledResult`：来自 legacy schedule 阶段，`node_idx_to_scheduled_results` 字段包含每个子图的候选 `ScheduledResult` 列表，非空性由既有 schedule 阶段保证，本编排代码不重复校验。
   - `ascend::TargetProfile`：由 3.5 节描述的构造流程产出。
   - `CodegenOptions.soc_version`（新增可选字段，见 6.2）。

3. 处理

   - 对 `node_idx_to_scheduled_results` 中每个 `(subgraph_id, candidates)`：调用 `StaticPlanSelector::SelectStaticWinner`（静态 shape）或 `SelectDynamicCandidates`（动态 shape，由 `ScheduledResult` 内部 shape 信息自动判定，不由本编排代码分支硬编码场景）。
   - 对选出的每个 `(subgraph_id, plan_id, ScheduledResult)`：先调用 `common::BuildTilingFieldSchema` 构造 `TilingFieldSchema`，再以其作为输入调用 `AttSchemaAdapter::RunAttForCandidates` 得到 `TilingCodeBundle`（含 `tiling_functions` 文本；不含 `TilingFieldSchema` 本身，只回填其 `schema_version` 用于一致性自检，见 3.1 节修正）。
   - 对每个候选：调用 `ScheduleToAfir::PlanToAfir(context, fusedResult, scheduledResult, subgraphId, planId, /*isInductor=*/scene == CallerScene::kInductor)` 得到 `OwningOpRef<ModuleOp>`（`scene` 即 3.7 节定义的调用方场景参数）。
   - 对所有候选共享的 `ascend::TargetProfile` + `AFIRToAscendCPipelineSession` 只构造一次（见 3.5 节），对每个候选模块新建一个 `PassManager`，调用 `addAFIRToAscendCPipeline(pm, session)` 后 `pm.run(module)` 跑四段 Pass。
   - 任一阶段失败（`LogicalResult` 为 failure，或抛出异常）：记录日志（等级 warning，说明失败阶段与 subgraph_id/plan_id），整个 `CodegenViaMlir` 返回 `MlirCodegenStatus::kFailed`（见 3.7 节），由调用方直接以对应 `af::Status` 返回，不回退 legacy（这是 Q7 三态设计与此前"failure 即回退"设计的关键差异，见 3.7 节）。异常统一以 `catch (...)` 兜底，不吞日志。

4. 输出

   - 成功：每个子图产出一个已完成四段 Pass 的候选 `ModuleOp` 列表（每个子图恰好一个，因为 `SelectStaticWinner`/`SelectDynamicCandidates` 最终都收敛到每子图一个 winner 用于聚合；动态 shape 多候选场景下，聚合阶段消费的是"每子图已选定的那一个" plan，多候选之间的运行时选择属于 `StaticPlanSelector` 既有职责范围，不属于本设计新增内容）。
   - 失败：`MlirCodegenStatus::kFailed` + 失败日志（见 3.7 节，调用方不再回退 legacy，直接返回失败）。
   - 未开启开关或命中 Cube 融合场景：本节流程不会被进入（守卫在 `CodegenViaMlir` 入口处，见 3.7 节），对应 `MlirCodegenStatus::kNotSupported`。

### 3.4 功能需求 2：多子图聚合与 AscendC 文本发射编排

1. 介绍

   把 3.3 节产出的每子图候选模块，通过既有 `GenAggregateTilingData`/`GenAggregateKernelEntry`/`GenAggregateKernelBody` 三个已实现组件聚合为单一 Entry Module，再通过 `translateToAscendC` 发射为设备端文本。

   **本节的接口扩展方案（落实 Q5：移除原 C1"不修改 GenAggregate\* 组件"约束）**：`GenAggregateKernelEntry.cpp` 内部已经实现了聚合契约校验（`collectAggregateContract`，77-90 行）与实参身份推导（`KernelArgumentIdentity`），但目前均为文件内部细节，不对外暴露；`runGenAggregateKernelEntry` 当前只返回 `LogicalResult`，调用方拿不到已经计算好的实参身份列表，如果编排层想知道"聚合后第几个形参对应哪个 input/output/workspace"，唯一途径是重新解析聚合后 `main` FuncOp 的属性——这是对已有计算结果的重复解析，且两处解析逻辑未来存在漂移风险（一处改了属性命名规则，另一处忘了同步）。相比"在编排层外部重新实现一遍解析逻辑"，让 `GenAggregateKernelEntry` 直接把已经算好的结果作为返回值的一部分导出，是更高效、更不易出错的方案，因此本设计选择修改该组件的对外接口（保持其内部聚合算法/校验逻辑不变，只扩展导出面）：

   - 新增公开类型 `AggregateArgumentLayout`（`GenAggregateKernelEntry.h`）：
     ```cpp
     struct AggregateArgumentLayout {
       std::vector<mlir::afir::KernelArgumentIdentity> arguments;  // 与聚合 main 形参顺序一一对应
       size_t tiling_arg_index;                                    // 末位 tiling 结构体形参下标
     };
     ```
   - 将 `collectAggregateContract` 内部已使用的契约校验逻辑提炼为公开函数 `validateAggregateContract(ModuleOp, Diagnostic &)`，供编排层在聚合前单独调用做前置校验（避免聚合失败时才发现契约问题，日志更早、更定位准确）。
   - 把 `runGenAggregateKernelEntry` 的返回类型从 `LogicalResult` 扩展为 `FailureOr<AggregateArgumentLayout>`：成功时同时完成聚合 `main` 构造与实参身份导出，失败时仍是纯 `failure()`，不引入调用方需要区分的第三种状态（与既有 `FailureOr` 惯例一致，未新增设计）。
   - 上述改动只新增/扩展接口，不改变 `collectAggregateContract`/`KernelArgumentIdentity` 内部已有的计算与校验算法本身，符合"改动最小化"原则，同时消除编排层重复解析属性的必要性。

2. 输入

   - 3.3 节产出的候选模块列表（每子图一个，均已完成四段 Pass）。
   - 各候选对应的聚合所需元数据（`afir.subgraph_id`/`afir.plan_id` 等属性，已由 3.3 节各阶段标注在模块/操作上，编排层不重复传递）。

3. 处理

   - 调用 `runGenAggregateTilingData`：把各候选模块中已由四段 Pass 标注的 tiling 相关属性聚合为单一 `AutofuseTilingData`（`PyStructType`，命名逐字节复用现网 struct 名，见 `GenAggregateTilingData.cpp:142`）。
   - 调用（扩展后的）`runGenAggregateKernelEntry`：先内部执行 `validateAggregateContract` 校验（`ascendc.plan_entries` 与候选模块数一致等，见 7.5 节错误处理），通过后生成聚合 `main` FuncOp——聚合后每个形参标注 `mlir::afir::KernelArgumentIdentity`（`kind: Input/Output/Workspace` + `index`），tiling 结构体形参始终位于形参列表末位、按位置识别（不再依赖 `autofuse.argument_name` 字符串属性，该属性随 `0e9f0486` 已废弃，见 1.4 术语表/6.1 节）——返回 `FailureOr<AggregateArgumentLayout>`。
   - 调用 `runGenAggregateKernelBody`：为聚合 `main` 生成函数体（按 `plan_entries` 顺序分发到各子图对应的候选实现，tiling_key 驱动分支已在四段 Pass 的 `SelectImplementation`/`ConvertAFIRToAscendC` 阶段处理，聚合阶段不重新引入分支逻辑）。
   - 调用 `translateToAscendC(entryModule, ostream)`（写入 `llvm::raw_string_ostream`，见 1.1 节修正）得到完整 KernelBody 设备端文本。
   - 调用 `mlir::afir::emitTilingDataHeader({autofuseTilingData}, ostream)`（`TilingDataHeaderEmitter.h:13`，写入 `llvm::raw_string_ostream`）把上一步得到的 `AutofuseTilingData` 渲染为 C++ 结构体文本，作为 `CodegenResult.tiling_data`（见 3.1 节说明，该函数已存在，不属于本文新增组件）。

4. 输出

   - `KernelBody` 文本（完整可编译的 AscendC 设备源码，尚不含 host wrapper）。
   - `AggregateArgumentLayout`（上述扩展接口的返回值），直接供 3.6 节 AutofuseLaunch 生成消费，无需重新解析属性。
   - `tiling_data` 结构体文本（`emitTilingDataHeader` 输出），直接填入 `CodegenResult.tiling_data`。

### 3.5 功能需求 3：TargetProfile 与 Pipeline Session 构造

1. 介绍

   从既有的、已在 bisheng 编译驱动侧使用的 `soc_version` 字符串（如 `"Ascend910B2"`），构造四段 Pass 所需的 `ascend::TargetProfile`，并在其基础上构造 `AFIRToAscendCPipelineSession`（3.3 节修正已引入的步骤）。二者均只需要在一次 `CodegenViaMlir` 调用内构造一次，跨该次调用涉及的全部候选模块复用，不随候选数量重复构造。

2. 输入

   - `soc_version`：字符串，来源为本文新增的 `CodegenOptions.soc_version`/`AutofuserOptions` kwarg（见 6.2），与 `ascendc_compile.py --soc_version` 参数是同一份调用方传入的字符串，不与 Cube-tiling 运行时查询的 `aclrtGetSocName()` 混用（三者的区分见 1.4 术语表后续说明与 6.3 节）。

3. 处理

   - `auto selection = ascend::TargetSelection::builtinPlatform(soc_version);`
   - `auto profile = ascend::TargetProfileLoader::load(selection, diagnostic);`
   - 若 `soc_version` 未在 `Profiles/manifest.txt` 别名表中命中（`TargetProfileLoader::loadBuiltin` 内部 `HasPlatformAlias` 校验失败），`load` 返回失败，`diagnostic` 携带错误信息；本编排代码将其视为 `MlirCodegenStatus::kFailed`（见 3.7 节），不新增独立的错误码体系。
   - `auto session = createAFIRToAscendCPipelineSession(context, profile);`：内部构造并冻结 `shared_ptr<const ImplementationRegistry>`，供 3.3 节所有候选模块的 `PassManager` 共享同一个 `session`（`session` 本身按值持有 `shared_ptr`，多个 `PassManager` 并发或顺序调用 `addAFIRToAscendCPipeline(pm, session)` 均安全，因为注册表构造完成后只读）。

4. 输出

   - 成功：`ascend::TargetProfile` + `AFIRToAscendCPipelineSession`，前者不直接参与后续调用（已被 Session 封装），后者传给 3.3 节每个候选的 `addAFIRToAscendCPipeline`。
   - 失败：failure + diagnostic 文本，`CodegenViaMlir` 返回 `MlirCodegenStatus::kFailed`。

### 3.6 功能需求 4：AutofuseLaunch host wrapper 生成

1. 介绍

   本文新增的唯一"生成全新文本"的组件（其余组件均是既有能力的编排调用）。基于 3.4 节聚合返回的 `AggregateArgumentLayout`，生成与现网 `codegen_kernel.cpp:4438-4919` 逐字节兼容的 `extern "C" int64_t AutofuseLaunch(...)` 函数文本。

   本节同时给出对用户在设计讨论阶段提出的"launch 机制选型"问题的结论：CANN 对外只提供两种 kernel launch 原语——`<<<>>>` 三重尖括号语法与 `ACLRT_LAUNCH_KERNEL(kernel_name)(...)` 宏，两者都要求目标 kernel 名称在编译期已知，不存在运行时/句柄/字符串动态分发的公开 API（`2026-08-18-schedule-space-att-mlir-inductor-architecture.md` 第 2.1.2/2.2 节已完成该项分析）。因此"是否有比现网 `<<<>>>` 更灵活的方案"这一问题的答案是否定的：灵活性（多结构候选共享一次 launch）已经在更底层通过 tiling_key 分支（`scf.index_switch`，`GenKernelEntry.cpp`/`SelectImplementation` 已实现）解决，与 launch 原语选择无关。据此，本设计的 `AutofuseLaunch` 生成组件选择逐字节复用现网 ABI，而非设计新的 descriptor/LaunchFrame 机制。

2. 输入

   - 3.4 节输出的 `AggregateArgumentLayout { arguments: vector<KernelArgumentIdentity>, tiling_arg_index }`：`KernelArgumentIdentity` 携带 `kind ∈ {Input, Output, Workspace}` + `index`（同 kind 内的顺序号），按聚合 `main` 形参顺序排列；`tiling_arg_index` 指向末位 tiling 结构体形参（`AscendCKernelABI.h`/`.cpp`，见 1.4 术语表，随 `0e9f0486` 引入，替代原 `getArgumentKey`/`autofuse.argument_name` 字符串方案）。
   - 聚合后 kernel 函数名（图名，与 legacy 路径的 `graph_name` 语义一致）。

3. 处理

   - 遍历 `AggregateArgumentLayout.arguments`：按 `kind == Input` 的 `index` 升序得到输入形参顺序，按 `kind == Output` 的 `index` 升序得到输出形参顺序，`kind == Workspace` 精确匹配唯一的 workspace 形参（数量恒为 1，由聚合阶段保证，见 `validateAggregateContract`）；`tiling_arg_index` 直接给出 tiling 结构体形参在整体形参列表中的位置，不需要额外匹配。
   - 按 `codegen_kernel.cpp:4515` 起的现网模板拼接文本：形参列表顺序为 `blockDim → stream → input0..inputN-1 → output0..outputM-1 → workspace → tiling_data`，函数体为单一 `graph_name<<<blockDim, nullptr, stream>>>((uint8_t*)input0, ..., (uint8_t*)output0, ..., (uint8_t*)workspace, *tiling_data); return 0;`（非 CV 场景；本设计范围不含 CV/Cube 融合场景，见 1.3 不覆盖范围）。
   - 若 `AggregateArgumentLayout` 与预期模式不匹配（如 workspace 不为恰好 1 个，或 `tiling_arg_index` 越界），视为聚合契约违反，返回 `MlirCodegenStatus::kFailed`（该情形理论上不应发生，因为 `validateAggregateContract` 已在 3.4 节聚合阶段前置校验过；此处的检查是防御性收尾，不是本设计的主校验点）。

4. 输出

   - `AutofuseLaunch` 函数文本（string），与 3.4 节 `KernelBody` 文本拼接后作为 `CodegenResult.kernel`。

### 3.7 统一编排入口、三态返回码与 Cube 场景守卫

1. 背景与决策（回应 Q7："先尝试调用 CodegenViaMlir，NotSupported 则回退旧流程，否则直接以其返回码退出"是否可行）

   **结论：机制本身可行，按此设计。**

   三态返回码本身（`NotSupported` 继续走 legacy；`Failed`/`Succeeded` 直接以该状态返回，不回退）在架构上是合理的、比"任何失败都静默回退 legacy"的设计更清晰：一旦 MLIR 路径已经尝试过并明确失败（而不是"本次不适用"），继续跑一遍 legacy 路径掩盖了失败原因，也造成两倍的计算浪费；`NotSupported` 与 `Failed` 分开，使调用方可以准确区分"这条路径这次没有资格跑"和"跑了但出了问题"。这与 `af::Status` 现有的 `SUCCESS`/`FAILED` 二元惯例不冲突，是在其之上新增一个仅供本编排入口内部判断使用的三态枚举，不改变 `af::Status` 本身，也不要求其余 `Codegen` 方法采用三态。

   挂钩位置按 Q7 原始指示：直接在 `Codegen::Generate`（`codegen.cpp:387`/`437`，TensorFlow 场景入口）与 `Codegen::GenerateForInductor`（`codegen.cpp:394`，Inductor 场景入口）这两个函数内部调用 `CodegenViaMlir`，不下沉到调用方之外的其他文件/函数。

   ```cpp
   enum class MlirCodegenStatus { kNotSupported, kFailed, kSucceeded };
   enum class CallerScene { kInductor, kTensorFlow };

   // 调用方约定：
   //   status = CodegenViaMlir(...)
   //   若 status == kNotSupported：调用方继续执行原有 legacy 流程（本次不适用，非错误）
   //   否则（kFailed 或 kSucceeded）：调用方直接以 status 对应的 af::Status 返回，不再回退
   // 不抛出异常穿透到调用方（内部 catch (...) 后转换为 kFailed）。
   MlirCodegenStatus CodegenViaMlir(const ascir::FusedScheduledResult &fusedResult,
                                    const CodegenOptions &options, CallerScene scene,
                                    CodegenResult &result);
   ```

   `scene` 只用于 3.3 节 `PlanToAfir`/`AttSchemaAdapter::RunAttForCandidates` 等下游既有接口仍要求的 `is_inductor`/`is_inductor_scene` 语义转发，不影响本入口自身的选优/转换/Pass/聚合主流程分支（该主流程对两个场景完全一致）。

   **已知限制（与 3.2 节一致，此处不重复论证只记录结论）**：`Codegen::Generate` 目前在生产代码中没有实际调用方——`pyautofuse.cpp` 的 `HandleDeviceCodeGenForNonCVFusion`（551 行起）/`HandleHostCodeGenForNonCVFusion`（638 行起）是 TensorFlow 场景现网真实生产入口，但它们绕过 `Codegen::Generate`，直接调用 `Codegen::GenerateTilingData`/`GenerateKernel`/`GenerateTiling`。因此在 `Codegen::Generate` 内部挂钩 `CodegenViaMlir`，在 `pyautofuse.cpp` 的调用链路打通之前，对真实 TensorFlow 生产流量不会生效；环境变量 `AF_ENABLE_MLIR_CODEGEN` 开关目前只能在 Inductor 路径（`GenerateForInductor`）真正生效。是否打通 `pyautofuse.cpp` 到 `Codegen::Generate` 的调用链路，是否需要新增单测直接覆盖 `Codegen::Generate`+`CodegenViaMlir` 组合，均按用户指示明确列为本文范围之外的独立问题（见 1.3 节）。

2. `CodegenViaMlir` 内部的开关与 Cube 场景守卫（均映射为 `kNotSupported`）

   `CodegenViaMlir` 入口按顺序做两个判断，均不满足时才进入 3.3-3.6 节的实际编排流程：

   - 环境变量 `AF_ENABLE_MLIR_CODEGEN` 未设置为 `"1"`（即 `IsMlirCodegenEnabled()` 返回 `false`）：直接返回 `kNotSupported`（新路径默认关闭，见 C2）。
   - `ascgen_utils::IsCubeFusedScheduled(fusedResult)` 为真：直接返回 `kNotSupported`（现有 `GenAggregate*`/四段 Pass 均未支持 Cube/CV 融合场景，见 1.3 不覆盖范围）。两条调用方各自既有的 Cube 专属 legacy 分支不受影响，继续按原逻辑执行；本条守卫只影响"是否尝试走 MLIR 路径"这一新增判断本身。

   3.3-3.6 节任一阶段失败时返回 `kFailed`（不触发回退）；全部成功时返回 `kSucceeded` 并填充 `result`。

3. 调用点改动

   - `GenerateForInductor`（`codegen.cpp:394`）：在现有 `IsCubeFusedScheduled` 判断与 `GenerateKernel`/`GenerateTilingForInductor` 调用之前，调用 `auto status = CodegenViaMlir(fusedResult, options, CallerScene::kInductor, result);`；`status == kNotSupported` 时继续执行现有 legacy 流程（无改动）；否则直接 `return status == kSucceeded ? af::SUCCESS : af::FAILED;`。
   - `Codegen::Generate`（`codegen.cpp:387`/`437`）：在其现有实现的最前面，同样调用 `CodegenViaMlir(fused_schedule_result, options, CallerScene::kTensorFlow, result)`，`kNotSupported` 时继续执行现有逻辑（调用 `GenerateKernel`/`GenerateTilingData`/`GenerateTiling` 等），否则直接返回对应状态。按上方"已知限制"所述，此挂钩点在 `pyautofuse.cpp` 调用链路打通前对真实 TensorFlow 流量不生效，但机制本身在函数内部是完整、正确、可独立测试的（见 9.1 节测试入口）。
   - `pyautofuse.cpp` 的 `HandleDeviceCodeGenForNonCVFusion`/`HandleHostCodeGenForNonCVFusion`/`HandleDeviceCodeGenForCVFusion`/`HandleHostCodeGenForCVFusion`：**不修改**。这些函数是否改为经过 `Codegen::Generate`、或是否需要单独挂钩，是本文范围之外的独立问题（见 1.3/3.2 节）。

4. 删除 `MaybeDumpAfirFromSchedule`

   `2026-07-27-schedule-att-mlir-migration-design.md` 已明确不把 `MaybeDumpAfirFromSchedule` 扩展为新路径的路由枢纽（"新路径放在 compiler 编排层"）。本设计进一步：既然统一编排入口本身已经具备完整的选优/转换/Pass/聚合能力，`MaybeDumpAfirFromSchedule`（`ScheduleToAfir.h:70`/`ScheduleToAfir.cpp:1060`，唯一调用点 `codegen.cpp:432`）作为诊断性 dump 钩子已无存在必要，予以删除；其原有的 `旧的 统一 dump 前缀（`ascir::utils::GetDumpFilePrefix()`）/统一 dump 前缀（`ascir::utils::GetDumpFilePrefix()`） 环境变量控制的 dump 能力（现已统一由 `ascir::utils::GetDumpFilePrefix()` 驱动），等价迁移进 `CodegenViaMlir` 内部实现（在 ③ 单 Plan 转 AFIR 之后按相同环境变量输出，行为对使用者透明）。此项修改超出原 C1 约束列出的"不修改 `ScheduleToAfir`"范围，但该约束已确认不必遵守（`ScheduleToAfir` 本身可以修改，见 2.3 节 C1 已同步调整）。

## 4 性能

### 4.1 编译时长

- `ENABLE_AUTOFUSE_MLIR=OFF`：不链接 MLIR/LLVM，无编译时长影响（现状不变）。
- 开关关闭（默认）：`CodegenViaMlir` 内部第一步判断 `IsMlirCodegenEnabled()`（读取环境变量 `AF_ENABLE_MLIR_CODEGEN`）即返回 `kNotSupported`，`GenerateForInductor`/TensorFlow 侧 Handle 函数行为与现网逐行相同，零额外编译期开销。
- 开关开启：新增编译期开销来源为 (a) 每候选一次 `PlanToAfir` + 四段 Pass，随子图数/候选数线性增长，与既有 07-27/08-13 设计文档已评估的 Pass 本身开销一致，本文不重复评估其内部实现开销；(b) 聚合三步与 `translateToAscendC`，已在 08-24 doc 评估过，随子图数线性；(c) 本文新增的 TargetProfile+Session 构造（每次调用一次，不随候选数重复）与 AutofuseLaunch 文本生成，均为 O(实参数量) 的字符串拼接，量级远小于 (a)(b)，可忽略。

### 4.2 执行性能

新路径生成的 kernel 在运行时通过与 legacy 完全一致的 `<<<blockDim, nullptr, stream>>>` 直接调用，不引入额外调度层、不增加 host-device 同步点、不新增 runtime API 调用，运行时开销为零增量。

### 4.3 内存和产物大小

- `AutofuseLaunch` host wrapper 文本体量与 legacy 生成文本同量级（相同模板，仅参数个数随实参列表变化）。
- 不新增运行包/wheel 内容；新链接的 `AFIRToAscendC`、`AscendTargetProfile`、`MLIRTargetAsc` 静态库仅在 `ENABLE_AUTOFUSE_MLIR=ON` 时参与链接，不影响 `OFF` 场景产物大小。

## 5 非功能需求

### 5.1 可维护性

`Codegen::CodegenViaMlir` 及其子步骤（3.3-3.6 节）作为独立私有方法拆分，每个方法对应一个既有组件的调用或本文新增的一个生成动作，不与 legacy `GenerateForInductor`/`GenerateKernel`/TensorFlow 侧 Handle 函数的实现交织，两条路径在代码层面完全并列、互不干扰，便于后续任一路径独立演进或下线。

### 5.2 可测试性

- 每个子步骤（TargetProfile 构造、AutofuseLaunch 文本生成）具备独立可测的输入输出契约（字符串输入→`TargetProfile`/字符串输出），可单独编写 UT 不依赖完整 pipeline。
- 完整 pipeline 的 UT/ST 复用现有 `autofuse/tests/ut/ascendc/compile/` 目录下的图构造模式（如 `abs.py`）与 `compile_adapter.jit_compile` 调用方式，见第 9 章。

### 5.3 可移植性

不新增平台相关分支；`soc_version` 到 `TargetProfile` 的映射完全依赖既有 `Profiles/manifest.txt` 数据文件，新增芯片型号只需更新该数据文件，不需要改动本文新增代码（呼应编码红线 Graph-autofusion 规则 2：禁止硬编码芯片类型）。

### 5.4 可靠性

- 任一阶段失败（Pass 失败、TargetProfile 解析失败、聚合契约缺失、AutofuseLaunch 实参模式不匹配）均归一化为 `MlirCodegenStatus::kFailed` + warning 日志，由调用方直接以 `af::FAILED` 返回（不回退 legacy，见 3.7 节机制变更），不存在"部分产出"的中间状态被返回给调用方。
- 所有新增 MLIR 上下文对象（`MLIRContext`、`OwningOpRef<ModuleOp>`）均为 `CodegenViaMlir` 调用栈内的函数局部对象，函数返回时自动释放，不引入需要手动管理生命周期的资源。

### 5.5 特性交叉影响

已在文档头部"设计文档检查结果"章节逐项填写，结论：Autofuse Codegen/Backend、AscendC API/Runtime 交互、Python/C++ 混合绑定、构建与打包、测试与覆盖率、性能与日志、兼容性共 7 项适用，均已在本文对应章节给出方案；SuperKernel Python/C++ 接口、Autofuse 图优化 3 项不适用。

## 6 接口设计

### 6.1 新增/修改接口描述

1. `mlir::bridge::RunMlirCodegen`（自由函数，定义于 `autofuse/mlir/bridge/CodegenOrchestration/CodegenOrchestration.h`/`.cpp`），统一服务 `Codegen::GenerateForInductor` 与 `Codegen::Generate`（详细设计见 3.7 节，含其对 TensorFlow 现网流量暂不生效的已知限制）

   **本节为 2026-08-31 更正后的实际接口**（原方案在此处描述的是 `Codegen::CodegenViaMlir` 私有方法，见 3.2 节"后续修订"说明；该方法体落地时被提取为下述自由函数，签名与放置位置均已调整，语义——三态返回码、`kNotSupported` 继续走 legacy、`kFailed`/`kSucceeded` 直接返回——不变）：

   ```cpp
   namespace mlir::bridge {

   enum class MlirCodegenStatus { kNotSupported, kFailed, kSucceeded };
   enum class MlirCallerScene { kInductor, kTensorFlow };

   // kNotSupported：调用方继续执行原有 legacy 流程；kFailed/kSucceeded：调用方直接以该状态返回，不回退。
   // 不抛出异常穿透到调用方（内部 catch (...) 后转换为 kFailed）。
   MlirCodegenStatus RunMlirCodegen(const ascir::FusedScheduledResult &fused_schedule_result, MlirCallerScene scene,
                                    const std::string &soc_version, codegen::CodegenResult &result);

   }  // namespace mlir::bridge
   ```

   与原方案的具体差异：
   - `CodegenViaMlir` → `RunMlirCodegen`，且不再是 `Codegen` 的私有成员方法，改为 `mlir::bridge` 命名空间下的自由函数——与该命名空间下 `StaticPlanSelector`/`AttSchemaAdapter` 等既有组件风格一致。
   - `CallerScene` → `MlirCallerScene`（同一 `mlir::bridge` 命名空间下的枚举改用更具体的名字，避免与其他潜在的通用场景枚举混淆）。
   - 入参不再是整个 `const CodegenOptions &options`，而是直接传入其中唯一被用到的 `soc_version` 字符串——因为编排逻辑已不是成员方法，无法再通过 `this->soc_version_` 隐式读取，且函数本身只需要这一个字段，没有理由透传整个 `CodegenOptions`。
   - 调用方 `codegen.cpp` 侧的调用时序与三态返回码语义完全不变：`GenerateForInductor`/`Codegen::Generate` 在各自现有 legacy 调用之前调用 `mlir::bridge::RunMlirCodegen(...)` 并传入对应的 `MlirCallerScene`；`kNotSupported` 时继续执行各自现有 legacy 逻辑（无任何改动），否则直接以 `kSucceeded → af::SUCCESS`/`kFailed → af::FAILED` 返回（具体调用点代码见 `codegen.cpp:401-403`/`446-448`）。

2. `CodegenOptions` 新增字段与 `Codegen` 类成员存取修正（`autofuse/codegen/codegen.h`）

   ```cpp
   struct CodegenOptions {
     // ...既有字段不变...
     std::string soc_version;  // 新增，默认空字符串；空值时视为未提供 target 信息
   };
   ```

   MLIR 编排路径的显式开关**不**作为 `CodegenOptions`/`AutofuserOptions` 字段引入（经与作者确认调整，理由与实现见下方说明）；本节只新增 `soc_version` 一个字段。

   **本节需要更正原方案的一处隐含假设**：原方案设想 `GenerateForInductor`/`Generate` 可以直接读取 `options.soc_version`。经核查，`Codegen` 类的构造函数（`codegen.cpp:383-385` 附近）并不保留传入的 `CodegenOptions` 整体，而是把其字段逐一拆分进独立私有成员（现有 `tiling_lib_`、`using_att_calc_qbt_size_`），构造完成后原始 `options` 对象即被丢弃；`GenerateForInductor`/`CodegenViaMlir` 都是 `Codegen` 的成员方法，只能访问 `Codegen` 自己的私有成员，访问不到调用方当年传入构造函数的那个 `CodegenOptions` 实例。因此 `soc_version` 必须同样在 `Codegen` 类内新增对应私有成员并在构造函数中初始化：

   ```cpp
   class Codegen {
     // ...
    private:
     TilingLib tiling_lib_;
     bool using_att_calc_qbt_size_;
     std::string soc_version_;  // 新增
   };
   ```

   **MLIR 编排路径开关改为环境变量**：经与作者确认（不在 `pyautofuse.cpp`/`CodegenOptions` 内新增 `enable_mlir_codegen` 参数，尽量小改动旧代码），`CodegenViaMlir` 入口不读取任何 `CodegenOptions` 字段或 `Codegen` 私有成员来判断是否启用，而是调用 `codegen.cpp` 内匿名命名空间中的一个 helper：

   ```cpp
   bool IsMlirCodegenEnabled() {
     const char *flag = std::getenv("AF_ENABLE_MLIR_CODEGEN");
     return flag != nullptr && std::string(flag) == "1";
   }
   ```

   即只有进程环境变量 `AF_ENABLE_MLIR_CODEGEN` 取值为 `"1"` 时才视为开启，其余情形（未设置、设为其他值）均视为关闭。该 helper 与既有 `ScheduleToAfir.cpp` 内 统一 dump 前缀（`ascir::utils::GetDumpFilePrefix()`）/统一 dump 前缀（`ascir::utils::GetDumpFilePrefix()`） 环境变量开关（非空即启用）是同一模式，不是新引入的机制。这样 `CodegenOptions`/`AutofuserOptions`/`Codegen` 均不新增任何与"是否走 MLIR 路径"相关的字段或成员，符合"尽量小的改动旧代码"的约束；`soc_version_` 成员的读取方式（构造时经 `CodegenOptions` 传入→拆分进成员变量→方法内读成员变量）与既有 `tiling_lib_`/`using_att_calc_qbt_size_` 的存取方式一致，不是新模式。

   兼容性：`soc_version` 字段有安全默认值（空字符串），不改变现有构造方式与既有调用点（`codegen.cpp` 内部现有构造处不需要修改）；环境变量默认未设置，等价于开关默认关闭。

3. `AutofuserOptions::Init` 新增可选 kwarg（`autofuse/compiler/py_module/pyautofuse.cpp`）

   新增 `soc_version`（str，默认空字符串）一个可选 kwarg，解析方式沿用现有 `PyDict_GetItemString`/`PyUnicode_Check` 模式，透传到 `CodegenOptions` 构造参数（进而拆分进 `Codegen` 的新增私有成员 `soc_version_`，见上一条）。**不**新增 `enable_mlir_codegen` kwarg（经与作者确认，见 6.1.2 节说明）——MLIR 编排路径完全由进程环境变量 `AF_ENABLE_MLIR_CODEGEN` 控制，`AutofuserOptions::Init` 不解析、不透传该开关。未传入 `soc_version` 时行为与现网完全一致。

4. AutofuseLaunch host wrapper 文本（生成产物，非 C++ 接口，但作为对外 ABI 契约列出）

   ```cpp
   extern "C" int64_t AutofuseLaunch(uint32_t blockDim, void* stream,
                                      /* per input: void* inputN */,
                                      /* per output: void* outputN */,
                                      void* workspace, AutofuseTilingData* tiling_data);
   ```

   逐字节复用 `codegen_kernel.cpp:4438-4919` 现网约定：参数顺序、`extern "C"` 链接方式、返回值语义（`0` 成功，非 `0` 失败）不变。调用方（Inductor/torchair 侧）无需任何修改。形参的输入/输出/workspace/tiling 划分依据来自 3.6 节 `AggregateArgumentLayout`（`KernelArgumentIdentity`），不再依赖已废弃的 `autofuse.argument_name` 字符串属性。

### 6.2 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|--------|----------|----------|------|
| 接口说明 | 是否需要接口评审 | 是 | `CodegenOptions`/`AutofuserOptions` 新增字段属于新增可选能力，建议随本设计文档一并评审。 |
| 接口说明 | 是否需要补充文档 | 是 | 本文即为该补充文档；后续需在 `AutofuserOptions` 使用说明中补充 `soc_version` kwarg 及 `AF_ENABLE_MLIR_CODEGEN` 环境变量说明。 |
| 接口兼容 | 行为是否兼容 | 是 | 新增字段均有默认值，未显式传入时行为与现网完全一致，见 C2/C5。 |
| 接口兼容 | ABI/API 是否兼容 | 是 | `AutofuseLaunch` 逐字节复用现网 ABI，见 6.1.4/C3；`CodegenOptions` 为增量字段，不改变既有内存布局中已存在字段的位置语义（新增字段追加在结构体末尾）。 |
| 接口约束 | 约束不满足时是否清晰报错 | 是 | TargetProfile 解析失败、聚合契约缺失、AutofuseLaunch 实参模式不匹配均有明确日志且返回 `kFailed`，见 3.3-3.6 节处理逻辑。 |
| 接口测试 | 是否需要独立接口用例 | 是 | 见第 9 章：TargetProfile 解析失败用例、AutofuseLaunch ABI 快照比对用例。 |

### 6.3 关于三种 soc_version 输入的区分说明

本设计涉及的 `soc_version` 字符串与另外两个已存在、职责不同的 soc-version 来源不得混用：

1. `ascendc_compile.py --soc_version` / `get_soc_type`：编译驱动侧参数，映射到 bisheng `--npu-arch=` 编译选项，本设计新增的 `CodegenOptions.soc_version` 与调用方传入 `ascendc_compile.py` 的值应为同一份字符串（由 Inductor/torchair 侧统一提供），但两者是两个独立的调用参数，本设计不做自动同步或校验，由调用方保证一致性传参（不新增校验代码，因为该一致性保证属于既有调用约定，不属于本文范围）。
2. 本设计新增的 `CodegenOptions.soc_version` → `TargetSelection::builtinPlatform` → MLIR `TargetProfile`：仅供四段 Pass 使用，见 3.5 节。
3. `codegen_tiling_cube.cpp::GetFullSocVersionForCubeTiling` 运行时查询的 `aclrtGetSocName()`：Cube/CV tiling 专用，本设计范围不含 CV 融合场景（见 1.3），不涉及、不修改该路径。

## 7 软件设计

### 7.1 关键数据结构

- 新增 `AggregateArgumentLayout`（`GenAggregateKernelEntry.h`，见 3.4 节）：`{ arguments: vector<KernelArgumentIdentity>, tiling_arg_index: size_t }`，作为 `runGenAggregateKernelEntry` 扩展返回值的一部分，生命周期与其构造函数的调用栈一致（值语义，随 `FailureOr` 按值返回）。
- 新增 `MlirCodegenStatus{kNotSupported, kFailed, kSucceeded}` 枚举（实际位于 `mlir/bridge/CodegenOrchestration/CodegenOrchestration.h`，`mlir::bridge` 命名空间下，非原方案所述 `codegen.h`——见 3.2/6.1/7.4 节 2026-08-31 更正），仅在 `RunMlirCodegen` 的返回值上使用，不改变 `af::Status` 本身。
- `ascend::TargetProfile`、`AFIRToAscendCPipelineSession`、`mlir::afir::KernelArgumentIdentity` 均为既有类型（后者随 `0e9f0486` 引入），本文只新增其构造/消费代码。
- 新增的两个 `CodegenOptions` 字段及对应 `Codegen` 私有成员均为值类型（`std::string`、`bool`），生命周期与各自持有者一致，无需额外管理（见 6.1.2 节）。

### 7.2 关键技术/算法

- 无新增算法。核心技术点是"`AggregateArgumentLayout` → AutofuseLaunch 形参列表"的确定性映射：按 `KernelArgumentIdentity.kind` 分组（`Input`/`Output`/`Workspace`），组内按 `index` 升序排列，`tiling_arg_index` 直接定位末位 tiling 结构体形参，不需要字符串前缀解析（`getArgumentKey`/`autofuse.argument_name` 方案已随 `0e9f0486` 废弃，见 1.4 术语表）。该映射是 O(实参数量) 的一次分组 + 排序，不存在遍历顺序不确定性风险（呼应编码红线 Graph-autofusion 规则 5）。

### 7.3 流程设计

主流程见 3.1 节数据流图。异常流程：3.3-3.6 节任一阶段失败 → 记录 warning 日志（含失败阶段、subgraph_id/plan_id 或 diagnostic 文本）→ `CodegenViaMlir` 返回 `MlirCodegenStatus::kFailed` → 调用方直接以 `af::FAILED` 返回（不回退 legacy，见 3.7 节机制变更），不做部分结果的清理（因为失败前不存在任何跨函数边界暴露的部分状态，局部对象随函数返回自动释放）。

### 7.4 对子模块的修改

- `autofuse/codegen/codegen.h`/`codegen.cpp`：`GenerateForInductor`（`codegen.cpp:394`）与 `Codegen::Generate`（`codegen.cpp:387`/`437`）各自新增前置调用分支，调用 `mlir::bridge::RunMlirCodegen(...)`（`kNotSupported` 时代码路径与现状逐行相同）；`CodegenOptions` 新增 `soc_version` 字段，`Codegen` 类新增对应私有成员 `soc_version_` 并在构造函数（`codegen.cpp:383-385`）中初始化（见 6.1.2 节修正）。**本条已随后续提取更正（见 3.2 节"后续修订"与下一条新增条目）**：`MlirCodegenStatus`/`MlirCallerScene` 枚举、`RunMlirCodegen` 实现本体及其入口读取的 `IsMlirCodegenEnabled()` 环境变量开关，均不在 `codegen.h`/`codegen.cpp` 内，而是随 `78fa250c` 一并提取到了下一条新增的 `mlir/bridge/CodegenOrchestration/` 目录；`codegen.h`/`codegen.cpp` 现在只保留 `soc_version_` 成员与两处调用点这一薄层。
- **（新增子模块，2026-08-31 追加）** `autofuse/mlir/bridge/CodegenOrchestration/CodegenOrchestration.h`/`.cpp`：随 `78fa250c` 新增，承载原计划放在 `codegen.cpp` 内的全部 MLIR 编排逻辑——`MlirCodegenStatus`/`MlirCallerScene` 枚举、`IsMlirCodegenEnabled()` 开关判断、逐子图候选选优→ATT tiling→`PlanToAfir`→四段 Pass 的编排循环、聚合 + `AutofuseLaunch` 发射的收尾，对外只暴露 `mlir::bridge::RunMlirCodegen` 一个入口（见 6.1 节更正后的接口描述）。构建方式复用既有 `autofuse_enable_mlir_bridge` CMake 函数（`autofuse/CMakeLists.txt:133` 起）把该 `.cpp` 作为 `target_sources` 直接编译进 `aihac_codegen`，与 `ScheduleToAfir.cpp` 走同一条路径（因为二者都依赖后端专属头文件如 `ge_common/debug/log.h`，不能进 `AFIRToAscendC` 这个纯 MLIR library），不新建 CMake target。
- `autofuse/mlir/bridge/ScheduleToAfir/ScheduleToAfir.h`/`.cpp`：删除诊断钩子 `MaybeDumpAfirFromSchedule` 及其在 `codegen.cpp:432` 的唯一调用点；dump 能力迁移进 `CodegenViaMlir`（见 3.7 第 4 点）。此项修改在 C1 约束中已单独列出（`ScheduleToAfir` 不在"不修改"范围内）。
- `autofuse/compiler/py_module/pyautofuse.cpp`：仅 `AutofuserOptions::Init` 新增一个可选 kwarg 解析（`soc_version`），透传到 `CodegenOptions`，供 Inductor 侧经 `GenerateForInductor` 使用；不新增 `enable_mlir_codegen` kwarg，MLIR 编排路径的开关完全由环境变量 `AF_ENABLE_MLIR_CODEGEN` 控制，与 Python 层配置无关（见 6.1.2 节）；**本文按 Q7 原始指示及与作者确认的最终结论（见 1.3/3.2/3.7 节），不修改 `HandleDeviceCodeGenForNonCVFusion`（551 行起）/`HandleHostCodeGenForNonCVFusion`（638 行起）/`HandleDeviceCodeGenForCVFusion`/`HandleHostCodeGenForCVFusion` 这四个函数**——它们直接调用 `Codegen::GenerateTilingData`/`GenerateKernel`/`GenerateTiling`，不经过 `Codegen::Generate`，因此 `Codegen::Generate` 内新增的挂钩对经由这条链路的现网真实 TensorFlow 流量暂不生效，这是本文已知且接受的限制，不在本文范围内解决。
- `autofuse/mlir/conversion/AFIRToAscendC/GenAggregateKernelEntry.h`/`.cpp`：新增公开类型 `AggregateArgumentLayout`、新增公开函数 `validateAggregateContract`，把 `runGenAggregateKernelEntry` 返回类型从 `LogicalResult` 扩展为 `FailureOr<AggregateArgumentLayout>`（详见 3.4 节）。**这是相对原 C1 约束的修正**：原 C1 列出该文件为"不修改"范围，本设计为落地 Q5（移除该约束、采用最高效方案）改为允许修改，且已论证改动范围最小化（不改变内部聚合算法，只扩展导出面），见 2.3 节 C1 已同步调整。
- `autofuse/mlir/conversion/AFIRToAscendC/Emit/`：`AutofuseLaunchEmitter.h`/`.cpp`（当前已实现），职责为 3.6 节描述的文本生成，输入为 `AggregateArgumentLayout` 与 graph 名，输出为 host wrapper 字符串。
- `autofuse/CMakeLists.txt`：`autofuse_enable_mlir_bridge` 函数（133-163 行）或 `aihac_codegen` 的 `target_link_libraries(aihac_codegen PUBLIC ...)` 块（232-254 行）追加链接 `AFIRToAscendC`（`autofuse/mlir/conversion/AFIRToAscendC/CMakeLists.txt:1` 定义的 `add_mlir_library`）、`AscendTargetProfile`（`autofuse/mlir/target/Ascend/CMakeLists.txt:28`）、`MLIRTargetAsc`（`autofuse/mlir/api/ascendc/lib/Target/AscendC/CMakeLists.txt:11`，供 `translateToAscendC` 使用，原方案曾遗漏该项）三个目标；三者均只在 `ENABLE_AUTOFUSE_MLIR=ON` 时参与链接（通过既有 `if(ENABLE_AUTOFUSE_MLIR)` 分支，见该文件 171/200/171 行现有结构）。**原方案在此处引用的行号（133-162 行"现有链接列表"）有误**：133-162 行是 `autofuse_enable_mlir_bridge` 函数定义本身，真正的 `aihac_codegen` PUBLIC 链接列表在 232-254 行，两处均可承载新增链接，本文选择放入 `autofuse_enable_mlir_bridge` 函数体内（与 `AfirBuilder`/`AFIRDialect` 现有两项一并追加，162 行处），因为该函数本身就是"仅在 `ENABLE_AUTOFUSE_MLIR=ON` 时被调用"的既定入口，不需要再包一层条件判断。
- 不修改：`autofuse/mlir/bridge/StaticPlanSelector/`、`autofuse/mlir/bridge/AttSchemaAdapter/`、`autofuse/mlir/conversion/AFIRToAscendC/AFIRToAscendCPipeline.*`、`GenAggregateTilingData.cpp`、`GenAggregateKernelBody.cpp`（`GenAggregateKernelEntry.h`/`.cpp` 已移出"不修改"列表，见上）、`autofuse/mlir/target/Ascend/`、`autofuse/compiler/python/ascendc_compile.py`、`compile_adapter.py`、`autofuse/codegen/codegen.cpp` 内 `Codegen::Generate` 方法本身（见上）。

### 7.5 错误处理

#### 7.5.1 系统错误

- TargetProfile 解析失败（`TargetProfileLoader::load` 返回 failure）：归一化为 `MlirCodegenStatus::kFailed`，不终止进程、不抛异常穿透。
- 四段 Pass 执行失败（`PassManager::run` 返回 failure）：同上。
- 聚合契约缺失（`validateAggregateContract` 校验失败，如 `ascendc.plan_entries` 与 `planModules.size()` 不一致）：同上。

#### 7.5.2 接口错误

- `CodegenViaMlir` 对调用方以 `MlirCodegenStatus` 三态返回值 + 已填充（`kSucceeded`）/未填充（`kNotSupported`/`kFailed`）的 `result` 表达状态，不新增异常类型，与既有 `Codegen` 类其余方法使用 `af::Status` 的错误处理风格保持同构（三态是在此基础上新增的一层，语义上是"`af::Status` 之外插入一个仅供本入口内部判断使用的中间态"，不改变 `af::Status` 本身，见 3.7 节第 1 点）。
- 失败时的日志经由既有日志组件输出（等级 warning），内容包含阶段名与 subgraph_id/plan_id 或 diagnostic 文本，供离线排查，不向 Python 调用方传播额外异常（保持 C3 约定的调用点零改动）。

## 8 兼容性检查

- 不影响已有脚本参数：`ascendc_compile.py`/`compile_adapter.py` 命令行参数不变。
- 不影响配置文件：`Profiles/manifest.txt` 只读，不新增格式。
- 不影响安装目录/动态库产物布局：环境变量 `AF_ENABLE_MLIR_CODEGEN` 未设置为 `"1"`（默认）时产物与现网完全一致；设置为 `"1"` 时产物内容变化但产物形态（.so 名称、tiling/kernel 文本三段式契约）不变。
- 不影响 Python 包对外接口：`AutofuserOptions` 新增两个有默认值的可选 kwarg，不改变现有正例调用（如 `AutofuserOptions(tiling_lib_path="", tiling_lib_codegen_symbol="")`）的行为。
- 不影响历史模型或用户代码：`AutofuseLaunch` ABI 逐字节不变，Inductor/torchair 侧调用点无需修改（C3）。

## 9 测试设计

### 9.1 测试边界

- 测试入口：C++ UT 通过公开的 `GenerateForInductor` + 环境变量 `AF_ENABLE_MLIR_CODEGEN` 组合调用间接覆盖 `CodegenViaMlir`（UT 内用 `ScopedMlirCodegenEnv` 一类 RAII helper 在测试体内 `setenv`/`unsetenv`，避免污染其他用例，见 `test_codegen_tiling.cpp`）；`Codegen::Generate` 侧同样通过公开方法 + 环境变量组合直接调用覆盖（两者均是 `Codegen` 的公开成员方法，无需 friend/protected 访问）。Python ST 通过设置环境变量 `AF_ENABLE_MLIR_CODEGEN=1` 后构造 `AutofuserOptions(soc_version="Ascend910B2", ...)` + `fuser.codegen(...)` 触发 `GenerateForInductor` 一侧；`Codegen::Generate` 一侧因当前无生产调用方（见 1.3/3.2/3.7 节已知限制），只在 C++ UT 中直接构造 `Codegen` 实例调用覆盖，不经由 Python 侧或 `pyautofuse.cpp` 触发。
- 测试出口：`CodegenResult` 三个字符串字段内容校验；`AutofuseLaunch` 文本片段的 ABI 快照比对；调用 `ascendc_compile.py`/`compile_adapter.py` 编译产出的 `.so` 通过既有 e2e runner 执行结果比对。
- 打桩模块：TargetProfile 解析失败场景通过传入不在 `Profiles/manifest.txt` 别名表中的非法 `soc_version` 字符串触发，不需要 mock。
- 验证数据：复用 `autofuse/tests/ut/ascendc/compile/abs.py` 等既有图构造模式，新增单子图+静态 shape 的最小正向用例。

### 9.2 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|----------|------------|----------|----------|
| 功能 | 单子图+静态 shape 全链路生成并编译执行 | 构造最小图 → 设置环境变量 `AF_ENABLE_MLIR_CODEGEN=1` → `codegen` → `compile_adapter.jit_compile` → 执行并比对结果 | UT/ST |
| 功能 | N 子图（N≥2）聚合场景全链路 | 复用 08-24 doc 已有的多子图聚合 UT 图构造，接入本文新增编排入口 | UT |
| 功能 | 动态 shape 候选选优场景 | 构造含动态 shape 轴的图，验证 `SelectDynamicCandidates` 路径可达且产出可编译 kernel | UT |
| 异常 | `ENABLE_AUTOFUSE_MLIR=OFF` 或环境变量 `AF_ENABLE_MLIR_CODEGEN` 未设置为 `"1"` 时行为与 legacy 逐字节等价 | 对同一图分别跑 legacy 与"开关关闭"两条路径，比对 `CodegenResult` 三字段文本完全一致 | UT |
| 异常 | TensorFlow 场景入口（`Codegen::Generate`）+ Cube 融合场景 + 环境变量 `AF_ENABLE_MLIR_CODEGEN=1`：仍跳过 MLIR 路径，回退原有实现 | 直接构造 `Codegen` 实例并复用 08-24 doc 已有的 Cube 融合图构造，设置环境变量 `AF_ENABLE_MLIR_CODEGEN=1` 后直接调用 `Codegen::Generate`，断言 `CodegenViaMlir` 返回 `kNotSupported`（`IsCubeFusedScheduled` 守卫命中）且最终结果与 legacy 路径一致；**注**：本用例仅验证 `Codegen::Generate` 内部机制本身的正确性，不代表覆盖 `pyautofuse.cpp` 的现网 TensorFlow 调用链路（该链路当前不经过 `Codegen::Generate`，见 1.3/3.2/3.7 节已知限制） | UT |
| 异常 | TargetProfile 解析失败（非法 `soc_version`） | 传入不在 manifest 别名表中的字符串，断言返回失败并回退 legacy，且回退后结果与纯 legacy 路径一致 | UT |
| 异常 | 聚合契约缺失（`plan_entries` 与 `planModules` 不一致） | 复用 08-24 doc 已有的契约缺失构造方式 | UT |
| 异常 | AutofuseLaunch 实参模式不匹配 | 构造缺少 `workspace`/`tiling` 实参的聚合 `main`，断言生成失败 | UT |
| 兼容性 | `AutofuseLaunch` 文本 ABI 快照比对 | 对固定输入图，比对生成的 `AutofuseLaunch` 函数签名文本与预先记录的 legacy 签名模板逐字节一致 | UT |
| 兼容性 | `CodegenOptions`/`AutofuserOptions` 新增字段默认值不改变现有调用 | 现有测试文件（如 `abs.py`）不修改，直接重跑确认结果不变 | UT |
| 性能 | 开关关闭时零编译期开销 | 对同一图分别计时"开关开启前/后"的 `GenerateForInductor` 调用（开关关闭状态），确认耗时无统计显著差异 | 手工验证 |
| 特性交叉 | bisheng 编译产物可执行验证 | 生成的 `.so` 通过 `autofuse/tests/st/codegen/e2e/` 既有 e2e runner 执行，比对计算结果与参考实现 | ST/E2E |

### 9.3 测试命令

```bash
sh build.sh -u --module=autofuse_framework -j 8
sh build.sh -s --module=autofuse_e2e -j 8
```

预期结果：新增 UT 全部通过；既有 UT/ST（尤其是 legacy `GenerateForInductor` 相关）保持全部通过，不出现回归。

## 10 验收标准

1. 功能：单子图+静态 shape 场景下，设置环境变量 `AF_ENABLE_MLIR_CODEGEN=1` 后能生成可通过 `ascendc_compile.py`/`compile_adapter.py` 编译的 `.so`，且执行结果与 legacy 路径生成的 `.so` 计算结果一致。
2. 功能：设计覆盖 N 子图（N≥1）与静态/动态 shape 两种模式，对应 UT 用例均通过，不存在只支持单一场景的架构性限制（对应 C4）。
3. 兼容性：`ENABLE_AUTOFUSE_MLIR=OFF` 或环境变量 `AF_ENABLE_MLIR_CODEGEN` 未设置为 `"1"`（默认）时，`GenerateForInductor` 产出与本设计实施前逐字节一致；`AutofuseLaunch` ABI 快照比对用例通过。
4. 测试：第 9 章列出的全部 UT/ST 用例通过，且现有回归测试（`build.sh -u --module=autofuse_framework -j 8`、`build.sh -s --module=autofuse_e2e -j 8`）无新增失败。
5. 性能：开关关闭时的编译耗时与本设计实施前无统计显著差异。
6. 文档：本设计文档完成评审；`AutofuserOptions` 新增 kwarg 说明补充到相应使用文档（若存在）；已知限制（`Codegen::Generate` 挂钩对经由 `pyautofuse.cpp` 的现网 TensorFlow 流量暂不生效，见 1.3/3.2/3.7 节）在文档中已明确记录，且验收范围本身不要求打通该链路。
