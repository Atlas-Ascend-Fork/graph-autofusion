# Inductor MLIR Codegen 全链路流程图与阶段说明

> 状态：现状说明（非新设计，不引入任何行为变更）
> 日期：2026-08-31
> 文档类型：现状导览

## 目的

`2026-08-26-inductor-mlir-codegen-launch-pipeline-design.md` 是本链路的设计文档。

本文则按本 PR 落地后的实际代码，画出从 `FusedScheduledResult` 到 `CodegenResult` 的完整调用链路图，并逐阶段说明输入、输出和职责。不重复设计文档中的决策背景、备选 方案或历史演进，需要这些内容时请查阅上述设计文档。

## 总体流程图

```mermaid
flowchart TD
    A["FusedScheduledResult<br/>（legacy schedule 产出）"] --> B{"RunMlirCodegen 入口守卫<br/>AF_ENABLE_MLIR_CODEGEN=1？<br/>非 Cube 融合场景？"}
    B -- "否 → kNotSupported" --> Z["调用方回退 legacy 路径<br/>GenerateKernel/GenerateTiling..."]
    B -- "是" --> C["① PrepareMlirCandidates"]

    subgraph C1["① 候选选优 + ATT tiling（均依赖 legacy ATT）"]
        C1a["StaticPlanSelector::SelectStaticWinner<br/>/ SelectDynamicCandidates<br/>⚠️ 内部调用 att::GetModelInfoMap 读取 legacy ATT 模型信息"] --> C1b["NarrowFusedResultToSelections<br/>只保留选中的模板"]
        C1b --> C1c["common::BuildTilingFieldSchema"]
        C1c --> C1d["AttSchemaAdapter::RunAttForCandidates<br/>⚠️ 调用 legacy ATT C++ 生成器<br/>→ TilingCodeBundle"]
    end
    C --> C1

    C1 --> D["② BuildPipelineSession<br/>soc_version → TargetProfile<br/>→ AFIRToAscendCPipelineSession<br/>（每次调用只构造一次，跨候选复用）"]
    D --> E["③ BuildCandidateAfirModules"]

    subgraph E1["③ 逐候选 AFIR 转换 + 四段 Pass"]
        E1a["ScheduleToAfir::PlanToAfir<br/>ScheduledResult → AFIR ModuleOp"] --> E1b["addPlanCandidateLoweringPipeline"]
        E1b --> E1b1["AFIRKernelBodySignature"]
        E1b1 --> E1b2["AFIRLoopGenerate → AFIRMemoryAllocate<br/>→ SelectImplementation → ConvertAFIRToAscendC"]
        E1b2 --> E1c["GenTilingData → GenKernelEntry<br/>（单候选 tiling schema + 单候选 dispatcher）"]
    end
    E --> E1
    E1 -->|"每子图一个已完成 Pass 的候选模块"| F["④ FinalizeMlirCodegenResult<br/>runAggregateLaunchDriver"]

    subgraph F1["④ 多子图聚合 + 文本发射"]
        F1a["runGenAggregateTilingData<br/>聚合 PyStructType"] --> F1b["validateAggregateContract"]
        F1b --> F1c["runGenAggregateKernelEntry<br/>→ AggregateArgumentLayout"]
        F1c --> F1d["runGenAggregateKernelBody<br/>拼装聚合 main 分派体"]
        F1d --> F1e["translateToAscendC<br/>→ KernelBody 文本"]
        F1e --> F1f["emitTilingDataHeader<br/>→ tiling_data 结构体文本"]
        F1f --> F1g["emitAutofuseLaunch<br/>→ AutofuseLaunch host wrapper 文本"]
    end
    F --> F1

    F1 --> G["CodegenResult<br/>kernel = KernelBody + AutofuseLaunch<br/>tiling = TilingCodeBundle.tiling_functions 拼接<br/>tiling_data = emitTilingDataHeader 输出"]
    G --> H["af::SUCCESS，调用方直接返回<br/>（kFailed 同理直接返回，不回退 legacy）"]
```

## 调用入口与三态返回码

链路由 `mlir::bridge::RunMlirCodegen`（`autofuse/mlir/bridge/CodegenOrchestration/CodegenOrchestration.h`）
统一承载，`Codegen::GenerateForInductor` 与 `Codegen::Generate`（`autofuse/codegen/codegen.cpp:397`/`442`）
在各自现有 legacy 调用之前调用它：

```cpp
enum class MlirCodegenStatus { kNotSupported, kFailed, kSucceeded };
enum class MlirCallerScene { kInductor, kTensorFlow };

MlirCodegenStatus RunMlirCodegen(const ascir::FusedScheduledResult &fused_schedule_result, MlirCallerScene scene,
                                 const std::string &soc_version, codegen::CodegenResult &result);
```

- `kNotSupported`：环境变量 `AF_ENABLE_MLIR_CODEGEN` 未设为 `"1"`，或命中 Cube 融合场景。调用方继续执行
  原有 legacy 流程，等价于本次未调用过这条新路径。
- `kFailed` / `kSucceeded`：调用方直接以对应 `af::Status`（`af::FAILED`/`af::SUCCESS`）返回，**不**回退
  legacy——一旦真正尝试过 MLIR 路径，其失败被视为真实错误而非"本次不适用"。
- 内部任意阶段抛出异常均被 `catch (...)` 兜底为 `kFailed`，不会穿透到调用方。

## 各阶段详细说明

### 阶段 0：入口守卫（`RunMlirCodegen` 开头两行判断）

- 读取进程环境变量 `AF_ENABLE_MLIR_CODEGEN`（非 `"1"` 即关闭，默认关闭，零行为变化）。
- 调用 `ascgen_utils::IsCubeFusedScheduled` 排除 Cube/CV 融合场景（现有 `GenAggregate*`/四段 Pass 均未
  支持该场景）。
- 两者任一命中即返回 `kNotSupported`，不进入下面任何阶段。

### 阶段 ①：`PrepareMlirCandidates`（候选选优 + ATT tiling）

输入：`FusedScheduledResult`（每子图含若干候选 `ScheduledResult`）、`MlirCallerScene`。

1. `SelectStaticWinner`（静态 shape）尝试按 ATT 代价模型选出每子图的 winner；无法确定完整
   winner 时保留全部候选。`SelectDynamicCandidates`（动态 shape）直接保留全部候选，产出
   `PlanSelection` 列表。静态尺寸已是常量，传入空的 `ConcreteShape`，不把常量文本当作符号名。
   ⚠️ **依赖 legacy ATT**：多候选静态择优通过 `att::GetModelInfoMap` 获取代价模型；唯一候选
   无需生成模型。结构非法、ATT 模型生成失败仍然报错，不等同于“无法确定 winner”。
2. `NarrowFusedResultToSelections`：把 `FusedScheduledResult` 收窄为只含已选中的 plan/template，并把
   `plan_id`、`template_ids_per_stage` 重新从 0 连续编号。**这一步是后续所有阶段共享的基准**——ATT
   tiling 生成与 AFIR 转换看到的都是这份收窄+重编号后的结果，不是原始的 `fused_schedule_result`。
3. `common::BuildTilingFieldSchema`：从收窄结果构造 `TilingFieldSchema`（字段列表 + version）。
4. `AttSchemaAdapter::RunAttForCandidates`：调用既有 ATT C++ 生成器，为每个候选批量产出 tiling 计算
   函数源码，装进 `TilingCodeBundle{ tiling_functions, selections, schema_version, schema_validated }`。
   ⚠️ 同样**依赖 legacy ATT**（ATT 移到 MLIR 后应替换为 MLIR-native 的 tiling/schema 契约）。

阶段①因此整体是一个 legacy-ATT 依赖环节：候选打分（步骤1）和 tiling 代码生成（步骤4）分别调用了
ATT 的两套不同 API（`att::GetModelInfoMap` 与 ATT C++ 生成器），迁移完成前这两处都不能删除。

输出：`MlirCandidatePreparation{ narrowed_result, selections, op_name, bundle }`。任一步失败即返回
`mlir::failure()`，`RunMlirCodegen` 记录 warning 日志后返回 `kFailed`。

#### 当前候选策略（2026-10-20 确认）

- **不按 MLIR 能力预筛选模板。** 已删除的搬运连续性、逐元素广播和 queue/reuse 视图筛选，
  是为了规避当时后端无法处理的场景，不是性能择优。选择层不再复制这些支持条件，不因它们
  删除模板或整个 plan；对应的合法性和能力判断由 bridge、资源校验及 lowering 负责。
- **保留代价择优。** 静态能够选出 winner 时通过 Narrow 收窄；静态无法选出 winner 或动态
  shape 时保留全部模板，交给生成的 ATT tiling 在运行时选择。两条路径都经过 Narrow，
  ATT 和 AFIR 使用相同的候选集合及重编号结果。
- **保留旧执行顺序。** Prepare 内完成选择、Narrow 和 ATT 生成，随后 lowering。ATT 是 legacy
  与 MLIR 共用的能力；ATT 失败按错误处理。ATT 成功后，lowering 的能力不足诊断才触发统一
  legacy 回落，其他错误仍返回失败。候选 lowering 失败时不删除该候选后重试。
- **接受完整候选集的覆盖要求。** 保留多个候选时，其中一个无法下降就不能产出成功的 MLIR
  结果；不以剩余候选成功掩盖覆盖缺口。静态已选出 winner 时只要求选中的候选能够下降。
- **测试区分层次。** 改写节点类型的合成输入只验证 bridge 能力诊断；端到端回落使用真实
  ArgMax 图经 optimizer 生成的合法 schedule，验证 ATT 成功后返回 `kNotSupported`。
  保留 legacy ATT 模板选择测试能力，不恢复 MLIR 专用 pinning。

验证：桥接 110 项、schedule 51 项、选择策略 19 项单测通过；此前静态择优恢复后的 29 项
CPU 回归通过。此次策略确认不新增 NPU 验证，不改动 legacy 实现或 AGENTS.md。

### 阶段 ②：`BuildPipelineSession`（TargetProfile + Pass Session 构造）

输入：`soc_version`（如 `"Ascend910B2"`，来自 `Codegen` 的 `soc_version_` 成员，由 `CodegenOptions`
构造时传入）。

1. `soc_version` 为空直接失败（清晰报错，不静默用默认值）。
2. `TargetSelection::builtinPlatform(soc_version)` → `TargetProfileLoader::load`，解析出
   `ascend::TargetProfile`。
3. `createAFIRToAscendCPipelineSession`：内部构造并冻结 `ImplementationRegistry`/
   `SelectionPolicyRegistry`，产出 `AFIRToAscendCPipelineSession`。

该 Session **只构造一次**，供本次调用涉及的全部候选模块共享（冻结后只读，多个 `PassManager` 复用同一
份安全）。

### 阶段 ③：`BuildCandidateAfirModules`（逐候选 AFIR 转换 + 四段 Pass）

对阶段①产出的每个 `(subgraph_id, plan_id)` 候选：

1. `ScheduleToAfir::PlanToAfir`：把一个 `ScheduledResult` 转换为一个独立的 AFIR `ModuleOp`
   （标注 `#afplan.candidate`，即候选身份 `(subgraph, plan)`）。
2. 为该模块新建一个 `PassManager`，调用 `addPlanCandidateLoweringPipeline`（
   `mlir/conversion/AFIRToAscendC/AFIRToAscendCPipeline.cpp:67`），依次跑：
   - `AFIRKernelBodySignature`：补齐规范的 `TilingData` 形参，构造 kernel body 函数边界。
   - `addAFIRToAscendCPipeline` 四段正式 Pass：
     - `AFIRLoopGenerate`：物化调度后的 `scf` 循环树（只生成轴值和循环结构，不做张量/内存 lowering）。
     - `AFIRMemoryAllocate`：完成内存分配。
     - `SelectImplementation`：按 `TargetProfile` + 冻结后的 `ImplementationRegistry`/
       `SelectionPolicyRegistry` 为每个 AFIR 算子选定具体实现。
     - `ConvertAFIRToAscendC`：把已选定实现的 AFIR 算子转换为 AscendC/EmitAsc dialect。
   - `GenTilingData`：把规范 `TilingData` 重命名为该候选的 per-stage schema。
   - `GenKernelEntry`：为该候选合成 plan 级 dispatcher。
3. 任一子步骤失败：记录 `subgraph_id`/`plan_id` 的 warning 日志，整体返回 `kFailed`。

输出：每子图一个已完成上述全部 Pass 的候选 `ModuleOp`（`BuiltPlanModules`，`owned` 持有生命周期，
`views` 是传给下一阶段的非拥有视图）。

### 阶段 ④：`FinalizeMlirCodegenResult`（多子图聚合 + 文本发射）

调用 `runAggregateLaunchDriver`（`mlir/conversion/AFIRToAscendC/AggregateLaunchDriver.cpp`），内部两段：

**4a. `BuildAggregateEntryModule`（聚合）**

1. `runGenAggregateTilingData`：把各候选模块中已由四段 Pass 标注的 tiling 相关属性聚合为单一
   `AutofuseTilingData`（`PyStructType`）。
2. `validateAggregateContract`：校验 `ascendc.plan_entries` 与候选模块数一致等聚合前置契约，
   校验失败时比聚合本身失败给出更早、更精确的诊断。
3. `runGenAggregateKernelEntry`：生成聚合 `main` FuncOp，每个形参标注
   `mlir::afir::KernelArgumentIdentity{kind, index}`（`Input`/`Output`/`Workspace`），返回
   `FailureOr<AggregateArgumentLayout>`（`arguments` + `tiling_arg_index`）。
4. `runGenAggregateKernelBody`：把各候选真实的 `<symbol>_main` 函数体克隆进聚合模块内对应的空函数声明，
   并把聚合 `main` 提升为 `FunctionRole::KernelEntry`。

**4b. `EmitArtifactsFromEntryModule`（文本发射）**

1. 在聚合模块内找到 `FunctionRole::KernelEntry` 函数，取其名字作为 `graphName`。
2. `translateToAscendC`：把聚合 Entry Module 翻译为完整可编译的 AscendC 设备源码文本（`kernel_body`，
   含 tiling_key 分支）。
3. 从 `ascendc.tiling_schema` 属性取出聚合 `PyStructType`，调用 `emitTilingDataHeader` 渲染为 C++
   结构体文本（`tiling_data`）。
4. `emitAutofuseLaunch`：基于 4a 产出的 `AggregateArgumentLayout`，按
   `blockDim → stream → input... → output... → workspace → tiling_data` 的现网 ABI 顺序拼接
   `extern "C" int64_t AutofuseLaunch(...)` 函数文本（`autofuse_launch`），逐字节兼容
   `codegen_kernel.cpp` 现网生成文本，调用点（Inductor/torchair 侧）无需修改。

### 收尾：装配 `CodegenResult`

```text
result.kernel      = artifacts.kernel_body + artifacts.autofuse_launch
result.tiling      = bundle.tiling_functions 逐项拼接（阶段①的 ATT 产物，与阶段④的聚合 tiling_data 是两条独立路径）
result.tiling_data = artifacts.tiling_data（阶段④的聚合 PyStructType 渲染结果）
```

全部成功后 `RunMlirCodegen` 返回 `kSucceeded`，调用方以 `af::SUCCESS` 返回；产出的 `.so` 可直接被
`autofuse/compiler/python/ascendc_compile.py`/`compile_adapter.py` 现有 bisheng 编译驱动消费。

## 涉及的源码位置速查

| 阶段 | 组件 | 路径 |
|---|---|---|
| 入口/编排 | `RunMlirCodegen` 及其四个子步骤 | `autofuse/mlir/bridge/CodegenOrchestration/CodegenOrchestration.{h,cpp}` |
| ① 候选选优 ⚠️legacy ATT | `StaticPlanSelector`（内部调用 `att::GetModelInfoMap`） | `autofuse/mlir/bridge/StaticPlanSelector/StaticPlanSelector.{h,cpp}` |
| ① ATT tiling ⚠️legacy ATT | `AttSchemaAdapter`（调用 ATT C++ 生成器） | `autofuse/mlir/bridge/AttSchemaAdapter/AttSchemaAdapter.{h,cpp}` |
| ② TargetProfile/Session | `AFIRToAscendCPipelineSession` | `autofuse/mlir/conversion/AFIRToAscendC/AFIRToAscendCPipeline.{h,cpp}` |
| ③ 单 Plan 转 AFIR | `PlanToAfir` | `autofuse/mlir/bridge/ScheduleToAfir/ScheduleToAfir.{h,cpp}` |
| ③ 四段 Pass + 单候选收尾 | `addPlanCandidateLoweringPipeline` | `autofuse/mlir/conversion/AFIRToAscendC/AFIRToAscendCPipeline.cpp` |
| ④ 聚合 + 文本发射 | `runAggregateLaunchDriver` | `autofuse/mlir/conversion/AFIRToAscendC/AggregateLaunchDriver.{h,cpp}` |
| ④ AutofuseLaunch 生成 | `emitAutofuseLaunch` | `autofuse/mlir/conversion/AFIRToAscendC/Emit/AutofuseLaunchEmitter.{h,cpp}` |
| 调用点 | `GenerateForInductor`/`Codegen::Generate` | `autofuse/codegen/codegen.cpp:397`/`442` |
| 命令行等价入口 | `af-opt --afir-plan-candidate-lowering-pipeline` / `--afir-aggregate-launch` | `autofuse/mlir/tools/af-opt/main.cpp` |

`af-opt` 提供了与上述③④阶段等价的独立命令行入口，可脱离完整 Codegen 调用单独调试单个候选的四段 Pass
（`--afir-plan-candidate-lowering-pipeline`）或聚合发射（`--afir-aggregate-launch`），对应 lit 测试见
`autofuse/mlir/test/lit/conversion/afir-plan-candidate-lowering-pipeline.mlir`/`afir-aggregate-launch.mlir`。

## 已知限制（不在本文解决，详见设计文档 1.3/3.2/3.7 节）

- `Codegen::Generate` 内的挂钩点对经由 `pyautofuse.cpp`（`HandleDeviceCodeGenForNonCVFusion`/
  `HandleHostCodeGenForNonCVFusion`）的现网 TensorFlow 生产流量暂不生效，因为这两个函数直接调用
  `GenerateTilingData`/`GenerateKernel`/`GenerateTiling`，不经过 `Codegen::Generate`。
- 阶段①先于阶段③执行是临时的依赖倒置：旧 ATT 尚未迁移到消费 MLIR 产出的 IR，因此当前只能先让 ATT
  消费 `FusedScheduledResult`，再单独跑 AFIR 转换与四段 Pass；ATT 完成自身 MLIR 迁移后预期顺序会调整。
