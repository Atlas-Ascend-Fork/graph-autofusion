# 1 Kernel Entry 生成设计

> 状态：已实现
> 日期：2026-08-08
> 现状复核日期：2026-08-26
> 对应实现：`GenKernelEntry.cpp`、`GenAggregateKernelEntry.cpp`、`AscendCKernelABI.cpp`

## 1.1 文档结构

本文先说明目标和边界，再说明四个相关 pass 的调用关系、参数身份契约、单 Plan entry、aggregate entry、错误处理和验证方法。阅读第 4 章可以直接了解一次完整编译从 Bridge 输出到 aggregate `main` 的数据流。

# 2 简介

## 2.1 目的

本文描述 Autofuse MLIR 如何把一个 Plan module 中的 Template 函数组织成 Plan main，以及如何把多个 Plan 的 main 组织成 Entry module 的 aggregate main。重点说明参数 union、参数 mapping、TilingData 转发和 selector dispatch 的实际规则。

本文面向 Bridge、AFIRToAscendC conversion、AscendC codegen 和相关测试维护人员。本文中的“Plan”是 Bridge 为一个 subgraph/plan 候选生成的独立 `ModuleOp`；“Template”是该 Plan 中带有 stage/template 坐标的私有 `func.func`。

## 2.2 范围

本文包含：

- `AFIRKernelBodySignature` 对 Template ABI 的补全；
- `GenTilingData` 生成 stage/Plan TilingData 后，`GenKernelEntry` 生成 Plan main；
- `GenAggregateTilingData` 生成 Entry schema 后，`GenAggregateKernelEntry` 生成 Plan declaration 和 aggregate main；
- 输入、输出、workspace、stage TilingData 的身份和转发规则；
- stage/template selector、graph/plan selector、非法输入和测试链路。

本文不包含：

- TilingData 字段如何收集以及 C++ header 如何输出，见 `2026-08-13-tiling-data-schema-and-header-generation-design.md`；
- Template 函数体的 Bridge 构造、AFIR 图优化和 AscendC lowering；
- Cube/CV 正向支持；
- Host 侧 tiling value 填充、runtime 调用、Python/C++ binding、SuperKernel 和打包流程。

# 3 总体概述

## 3.1 软件概述

一个完整的入口生成流程处理两种 module：

```text
每个 Plan module
  Template functions
    -> AFIRKernelBodySignature
    -> GenTilingData
    -> GenKernelEntry
    -> <plan_symbol>_main definition

全部 Plan modules
  -> GenAggregateTilingData
  -> Entry module + AutofuseTilingData + ascendc.plan_entries
  -> GenAggregateKernelEntry
  -> Plan main declarations + aggregate main definition
```

生成后的调用层级是：

```text
aggregate main
  -> 读取 graph<N>_tiling_key，选择 subgraph 内的 plan ordinal
  -> 调用 <plan_symbol>_main
     -> 每个 stage 读取 g<N>_tiling_data.tiling_key
     -> 选择一个 Template
     -> 转发 union 参数和对应 stage TilingData
```

## 3.2 产品环境介绍

- 代码依赖项目的 LLVM/MLIR、Func、SCF、Arith、EmitC 和 EmitAsc dialect。
- 四个 pass 只构造和校验 MLIR，不调用 `rt*`、`aclrt*` 或设备接口。
- Plan main 的 definition 保留在 Plan module；aggregate 阶段只在 Entry module 创建同名 private declaration。
- `af-opt --gen-kernel-entry` 是开发和 LIT 入口，不是产品对外 API。
- 后续 codegen/链接阶段负责消费这些 MLIR definition、declaration 和 schema；本文不定义其链接实现。

## 3.3 软件功能

1. 校验 Plan/Entry module、schema、Template 坐标、tiling key 和函数 ABI。
2. 为每个 stage 生成按实际 `afir.asc_graph.tiling_key` 分派的 `scf.index_switch`。
3. 将不同 Template 的输入/输出参数合并成一个稳定的 Plan main 参数 union。
4. 按参数身份生成每个 Template 的调用 mapping，允许 Template 参数数量和顺序不同。
5. 将多个 Plan main 的参数合并成 aggregate main 参数 union，并替换各 Plan 的 nested TilingData。
6. 按 aggregate schema 中的 graph selector 选择 Plan；非法 selector 不调用任何 Plan。

## 3.4 设计约束

> **勘误（2026-09-12，AFIR 公开面重构 5.1 步 7）**：候选身份与分派表已迁出 `afir` 命名空间——
> `afir.subgraph_id` + `afir.plan_id` 合为 `#afplan.candidate<subgraph, plan>`，
> `afir.stage_id` + `afir.template_id` 合为 `#afplan.variant<stage, template>`，
> `afir.template_counts_per_stage` 改为 `#afplan.stages<template_counts>`，`afir.stage_count`
> 已删除（其值恒等于 `template_counts` 的长度且无读点）。「非负」不再由各读点分别校验，
> 而由属性 verifier 一次钉死。判别与理由见
> `docs/design/2026-09-09-afir-public-dsl-architecture-design.md` 2.3 与 5.1 步 7。

> **勘误（2026-09-14，AFIR 公开面重构 3.6「Kernel 程序容器」）**：kernel 参数身份已收敛成
> **一个**强类型 arg attr `#afir.kernel_arg<role = input|output|workspace|tiling (, index = N)?>`。
> 本节下文中这三条描述的机制**均已删除**：
>
> | 原文 | 现状 |
> |---|---|
> | 「输入参数使用 `afir.input_index`，输出参数使用 `afir.output_index`；两者不能同时存在」 | 两个属性都已删除。「不能同时存在」不再靠约定——一个参数只有一个角色，冲突在解析期即失败 |
> | 「workspace 不使用参数属性，通过规范的 workspace `MemRefType` 识别」 | 改为 `role = workspace` 显式声明。类型相等降级为对声明的**校验**，不再是识别依据 |
> | 「TilingData 不参与普通参数身份收集，通过类型和参数位置单独替换」 | 改为 `role = tiling` 显式声明。位置约定（workspace、tiling 在最后两位）仍是 AscendC C++ 签名的要求，但同样降级为被检查的约定 |
>
> 同时删除的还有函数属性 `afir.num_inputs`——它恒等于 `afir.arg_descs` 的长度，是同一事实的
> 第二份编码；数据参数个数现在由角色数出来（`getCanonicalDataArgumentCount`）。3.5 中
> 「Bridge 已经把 scheduler 的输入输出索引写成 `afir.input_index`/`afir.output_index`」一条
> 同此：bridge 现在写的是 `#afir.kernel_arg`。
>
> 改动的理由正是本节这几条约束暴露的问题——**反推规则不在任何对外文档里**，而每个消费端都要
> 把它重写一遍。判别与理由见
> `docs/design/2026-09-09-afir-public-dsl-architecture-design.md` 3.6 的实施结论。

- 当前实现只接受 `afir.module_kind = "plan"` 的 Plan module 和 `"entry"` 的 Entry module。
- Plan 必须有非空 symbol、`ascendc.tiling_schema` 和非空 `afir.template_counts_per_stage`。
- 每个 `(stage_id, template_id)` 必须唯一对应一个 private Template。
- 每个 Template 必须恰好有一个与该 stage schema 相等的 TilingData 参数。
- Template 必须提供 `ascendc.num_inputs`，其类型为 signless i32，值非负且不大于 TilingData 参数前的参数数量。
- 输入参数使用 `afir.input_index`，输出参数使用 `afir.output_index`；两者不能同时存在。
- workspace 不使用参数属性，通过规范的 workspace `MemRefType` 识别。
- TilingData 不参与普通参数身份收集，通过类型和参数位置单独替换。
- 同一 Plan 内相同参数身份的类型必须一致；不同 Template 可以缺少 union 中未使用的参数。
- Plan main 和 aggregate main 的普通参数按 `input -> output -> workspace` 的 kind/index 顺序稳定排列。
- Template 的 tiling key 必须非负且在同一 stage 内唯一；case 值使用 tiling key，不使用 template ordinal。
- aggregate Plan ID 必须由 `GenAggregateTilingData` 校验为每个 subgraph 内从 0 连续的序列。
- 构建命令必须限制并行度，使用 `-j 8`。

## 3.5 假设和依赖关系

- Bridge 已经把 scheduler 的输入输出索引写成 `afir.input_index`/`afir.output_index`。
- `AFIRKernelBodySignature` 已经把输出、workspace 和通用 TilingData 参数补入 Template ABI。
- `GenTilingData` 已经把通用 TilingData 参数替换成 stage schema，并写入 Plan schema。
- `GenAggregateTilingData` 已经生成 Entry 的 `ascendc.tiling_schema` 和 `ascendc.plan_entries`。
- aggregate selector 当前表示 subgraph 内的 plan ordinal；Template selector 则表示真实 tiling key。这两个 selector 不是同一个编号空间。

# 4 需求分析与设计

## 4.1 整体数据流

以两个 Template 参数不同的一个 Plan 为例：

```text
Template 0: input:0, input:1, output:0
Template 1: input:0, output:0
```

经过 `AFIRKernelBodySignature` 后分别变为：

```text
Template 0: input:0, input:1, output:0, workspace, TilingData
Template 1: input:0, output:0, workspace, TilingData
```

`GenKernelEntry` 收集身份后生成 Plan main union：

```text
Plan main: input:0, input:1, output:0, workspace, PlanTilingData
```

调用 mapping 为：

```text
Template 0 call: %arg0, %arg1, %arg2, %arg3, stage_tiling
Template 1 call: %arg0, %arg2, %arg3, stage_tiling
```

其中 `%arg1` 是 union 中的 `input:1`，Template 1 不使用它，因此不会出现在 Template 1 的 call 中。

## 4.2 四个 pass 的调用顺序

对每个 Plan，调用方必须按以下顺序运行：

```text
AFIRKernelBodySignature
  -> 为 Template 补全输出/workspace/TilingData ABI
GenTilingData
  -> 生成 stage/Plan schema，并物化 stage TilingData 参数
GenKernelEntry
  -> 生成 Plan main definition
```

全部 Plan 完成后：

```text
GenAggregateTilingData
  -> 创建 Entry module，生成 AutofuseTilingData 和 plan_entries
GenAggregateKernelEntry
  -> 创建 Plan main declaration 和 aggregate main
```

如果跳过前置 pass，后续 pass 会因为缺少 schema、stage TilingData 或 ABI metadata 失败，不会使用旧字符串属性猜测参数身份。

## 4.3 功能需求 1：识别 Template 参数

### 4.3.1 输入

`GenKernelEntry` 遍历当前 Template `func::FuncOp` 的函数参数。参数身份由 `getKernelArgumentIdentity()` 统一解析：

| 参数类别 | 识别方式 | 身份值 |
|---|---|---|
| 输入 | `afir.input_index` | `Input(index)` |
| 输出 | `afir.output_index` | `Output(index)` |
| workspace | 参数类型等于 `getAscendCWorkspaceType(context)` | `Workspace` |
| stage TilingData | 参数类型等于当前 stage type | 不进入普通参数 union |

旧的 `autofuse.argument_key` 和 `autofuse.argument_name` 不再作为生产参数身份来源。

### 4.3.2 workspace 类型

`getAscendCWorkspaceType(MLIRContext *)` 在 `AscendCKernelABI.cpp` 中构造规范类型：

```text
memref<ui8, 22>
```

其中 `22` 是 AscendC global memory space。这个函数只根据传入的 `MLIRContext` 构造类型，不查询 module，也不读取 func 属性。调用方用当前 Template 的第 `index` 个参数类型与该规范类型比较。

### 4.3.3 校验

- 同一参数同时有 input/output 属性时失败。
- input index 必须非负，并且小于 `ascendc.num_inputs`。
- output index 必须非负。
- 没有身份属性且又不是 workspace 类型的普通参数失败。
- 同一 Template 内重复身份失败。
- 作为 vector 下标使用前，所有 index 都已经完成非负和范围校验。

## 4.4 功能需求 2：生成 Plan main

### 4.4.1 收集 Template

`collectTemplate()` 按 `afir.stage_id` 和 `afir.template_id` 查找函数：

1. 遍历当前 Plan module 的顶层 `func::FuncOp`。
2. 只接受坐标完全匹配的函数。
3. 找到第二个相同坐标时失败。
4. 找不到函数或函数不是 private 时失败。

随后 `collectStageTilingKeys()` 从每个 Template 的 `afir.asc_graph` 读取 tiling key，并检查非负和 stage 内唯一。

### 4.4.2 推导 Plan main 类型

对每个 stage/template，`collectTemplateAbi()` 完成以下操作：

1. 在 Template 输入类型中定位唯一的 stage TilingData 参数。
2. 读取并校验 `ascendc.num_inputs`。
3. 遍历除 stage TilingData 外的所有参数，得到参数身份。
4. 首次发现身份时加入 `MainSignature`。
5. 再次发现相同身份时比较参数类型；类型不同则失败。
6. 保存当前 Template 到 Plan main 参数的 mapping。

所有 stage 完成后，`reorderMainArguments()` 按以下顺序重排 union：

```text
KernelArgumentKind::Input，按 index 升序
KernelArgumentKind::Output，按 index 升序
KernelArgumentKind::Workspace
```

重排同时更新所有 Template mapping。最终 Plan main 类型为：

```text
(union ordinary arguments..., plan TilingData) -> ()
```

### 4.4.3 生成 dispatch

`buildStage()` 对每个 stage 递归生成：

1. 从 Plan TilingData 读取 `g<N>_tiling_data`。
2. 从 stage TilingData 读取 `tiling_key`。
3. 将 selector 转换为 `index`。signless integer 使用 `arith.index_cast`，unsigned integer 使用 `emitc.cast`。
4. 创建无结果 `scf.index_switch`。
5. 每个 case 使用 Template 的真实 `afir.asc_graph.tiling_key`。
6. 按保存的 mapping 构造 Template call 参数。
7. Template 有 `ascendc.need_sync_all` 时，紧跟 call 生成 `ascendc.sync_all_hard`。
8. default 只生成 `scf.yield`，不调用 Template。

输出示例：

```mlir
func.func @plan_main(%arg0, %arg1, %arg2, %workspace, %plan_tiling) {
  %stage = emitasc.member %plan_tiling "g0_tiling_data"
  %key = emitasc.member %stage "tiling_key"
  scf.index_switch %key {
  case 0 {
    func.call @stage0_template0(%arg0, %arg1, %arg2, %workspace, %stage)
    scf.yield
  }
  case 100 {
    func.call @stage0_template1(%arg0, %arg2, %workspace, %stage)
    scf.yield
  }
  default { scf.yield }
  }
  func.return
}
```

## 4.5 功能需求 3：生成 aggregate main

### 4.5.1 Entry contract

`GenAggregateKernelEntry` 只消费 Entry module metadata，不重新遍历 Template body。Entry 必须包含：

```text
afir.module_kind = "entry"
ascendc.tiling_schema = TypeAttr<PyStructType>
ascendc.plan_entries = ArrayAttr<DictionaryAttr>
```

每个 `plan_entries` 项包含：

```text
subgraph_id
plan_id
tiling_field
```

`tiling_field` 指向 aggregate TilingData 中的 Plan schema。

### 4.5.2 收集 Plan declaration

每个 contract 项按以下顺序处理：

1. 按 `subgraph_id` 和 `plan_id` 找到 Plan module。
2. 找到 `<plan_symbol>_main` definition。
3. 在 Plan main 参数中唯一定位 contract 指向的 Plan TilingData 类型。
4. 对其余参数使用相同的 `KernelArgumentIdentity` 规则。
5. 将相同身份合并到 aggregate union，并校验类型一致。
6. 记录当前 Plan declaration 的参数 mapping；TilingData 位置用特殊值表示，后续替换为 aggregate schema member。

收集完成后，`reorderAggregateArguments()` 使用同一 canonical 顺序重排 aggregate union，并同步修改每个 Plan declaration 的 mapping。

### 4.5.3 生成 aggregate main

aggregate main 的参数为：

```text
(所有 Plan 的 ordinary argument union..., AutofuseTilingData) -> ()
```

普通输入输出参数写回 `afir.input_index`/`afir.output_index`；workspace 仍通过类型识别，不写参数属性。

当只有一个 Plan 时，aggregate main 直接调用该 Plan declaration。当有多个 Plan 时：

1. 按 subgraph 分组 contract 项。
2. 读取 `graph<N>_tiling_key`。
3. 将 selector 转换为 `index`。
4. 在当前 subgraph 内按 declaration 数组顺序生成 case `0..N-1`。
5. case 中通过 `tiling_field` 读取 Plan nested TilingData。
6. 使用 Plan mapping 转发 aggregate main 普通参数，并把特殊 mapping 替换成 nested TilingData。
7. default 只生成 `scf.yield`。

aggregate selector 是 plan ordinal，不是 Plan 内 Template 的 tiling key。Plan 内 Template selector 在 Plan main 中单独处理。

## 4.6 非功能需求

### 4.6.1 可维护性

- 参数身份、参数 key 字符串和 canonical 排序集中在 `AscendCKernelABI`。
- 单 Plan 和 aggregate 层分别保存自己的 mapping，但遵循相同身份模型。
- pass 不复制 Template body，不通过全局状态传递 contract。
- 所有输出顺序由显式 stage/template、subgraph/plan ID 和稳定的参数排序决定。

### 4.6.2 可测试性

- conversion LIT 使用 `af-opt` 验证正向和异常诊断。
- Bridge gtest 运行真实 `PlanToAfir -> AFIRKernelBodySignature -> GenTilingData -> GenKernelEntry`。
- aggregate gtest 再运行 `GenAggregateTilingData -> GenAggregateKernelEntry`，并通过 `mlir::verify` 和 call walk 检查结果。
- two_graph fixture 中 Template 0 使用二输入一输出，Template 1 使用一输入一输出，验证 union/mapping。

### 4.6.3 可移植性

不硬编码芯片型号、操作系统或框架类型。workspace memory space 来自当前 AscendC ABI 常量和类型构造函数。

### 4.6.4 可靠性

- module 属性、ID、数组索引和参数位置在使用前校验。
- 失败通过 `FailureOr`、`LogicalResult` 和 pass failure 传播。
- 不在失败时把部分生成的入口标记为可用结果。
- 不申请 runtime 资源、不访问文件、不增加全局可变状态。

### 4.6.5 特性交叉影响

| 场景 | 适用性 | 分析说明 |
|---|---|---|
| SuperKernel Python 接口 | 不适用 | 不修改 Python 包、选项或 wheel。 |
| SuperKernel C++/AOT 接口 | 不适用 | 不修改 SuperKernel 动态库和 AOT ABI。 |
| Autofuse 图优化 | 适用 | 只消费 Bridge facts 和 schema，不改写 Template 图。 |
| Autofuse Codegen/Backend | 适用 | Plan/aggregate entry 是后续 codegen 的入口 ABI。 |
| AscendC API / Runtime 交互 | 适用 | 只生成既有同步 IR，不新增 runtime 调用；workspace/TilingData 类型必须保持 ABI 一致。 |
| Python/C++ 混合绑定 | 不适用 | 不修改 pyautofuse 或 CPython binding。 |
| 构建与打包 | 适用 | 影响 `AFIRToAscendC` target 的源文件和 include 根路径，不改变打包内容。 |
| 测试与覆盖率 | 适用 | 需要 LIT、Bridge gtest 和不同参数 mapping 覆盖。 |
| 性能与日志 | 适用 | 增加编译期参数收集和少量 switch IR；不新增默认日志。 |
| 兼容性 | 适用 | 内部 MLIR entry ABI 和 contract 需与 schema/codegen 同步；不改变产品外部 API。 |

## 4.7 性能

### 4.7.1 编译时长

`GenKernelEntry` 的参数收集复杂度与 stage 数、Template 数和参数数线性相关；canonical reorder 复杂度与 union 参数数线性相关。aggregate pass 只遍历 Plan main declaration 和 Entry contract，不重新遍历 Template body。

### 4.7.2 执行性能

每个 Plan 增加 stage selector，每个多 Plan Entry 增加 graph selector。selector 只决定控制流，不复制或搬运普通参数。只有存在 `ascendc.need_sync_all` 时才生成硬同步。

### 4.7.3 内存和产物大小

Entry module 只保留 declaration、aggregate main 和 schema metadata，不复制 Plan body。新增内容主要是少量函数参数、switch IR 和 mapping 对应的 call operands，不增加 runtime 常驻内存。

## 4.8 接口设计

### 4.8.1 当前已实现接口

```cpp
std::unique_ptr<mlir::Pass> createGenKernelEntryPass();
void registerGenKernelEntryPass();

mlir::LogicalResult runGenAggregateKernelEntry(
    llvm::ArrayRef<mlir::ModuleOp> planModules,
    mlir::ModuleOp entryModule);
```

调用示例（当前已实现）：

```text
输入：已完成 signature/schema 的 Plan modules 和 GenAggregateTilingData 生成的 Entry module
调用：runGenAggregateKernelEntry(planModules, entryModule)
成功：Entry module 得到 Plan declarations 和 aggregate main
失败：返回 failure，调用方负责终止后续 codegen
```

### 4.8.2 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|---|---|---|---|
| 接口说明 | 是否需要接口评审 | 是 | 涉及内部 pass 和 aggregate contract。 |
| 接口说明 | 是否需要补充文档 | 是 | 本文与 TilingData 文档共同定义入口 ABI。 |
| 接口兼容 | 行为是否兼容 | 是 | 旧字符串身份 fallback 已移除，输入必须满足新 contract。 |
| 接口兼容 | ABI/API 是否兼容 | 是 | 不修改产品外部 API；内部 MLIR 参数 ABI 有明确迁移规则。 |
| 接口约束 | 约束不满足时是否清晰报错 | 是 | 缺失 metadata、类型冲突、ID 和 selector 错误均失败。 |
| 接口测试 | 是否需要独立接口用例 | 是 | 正向、不同参数数量和异常 LIT/gtest 已覆盖。 |

## 4.9 软件设计

### 4.9.1 关键数据结构

- `KernelArgumentKind`：区分 Input、Output、Workspace。
- `KernelArgumentIdentity`：保存参数 kind 和 input/output index；workspace 的 index 为 `-1`。
- `MainSignature`：保存 Plan main 的身份 union、参数类型、TilingData 类型和位置。
- `PlanInfo`：保存 Template、tiling key、每个 Template 的 mapping 和最终 main 类型。
- `PlanDeclaration`：保存 Entry 中 declaration、nested TilingData 字段和 Plan 参数 mapping。
- `DeclarationCollection`：保存 aggregate 参数 union、类型和所有 Plan declaration。

### 4.9.2 关键算法

1. 用参数 metadata/type 解析身份，不依赖参数名字或发现顺序。
2. 用身份相等关系合并 union，用参数类型检查同身份的一致性。
3. 用 canonical kind/index 排序，更新旧 mapping 到新位置。
4. 用真实 tiling key 生成 Plan 内 Template case。
5. 用 contract 顺序生成 aggregate plan case，并通过 nested schema member 转发 TilingData。

### 4.9.3 主流程和异常流程

```text
Plan
  -> deriveMainSignature
  -> validate Plan/schema/template
  -> locate unique stage TilingData
  -> collect identity/type/mapping
  -> collect unique tiling keys
  -> canonical reorder
  -> create Plan main

Entry
  -> validate entry/schema/plan_entries
  -> locate each Plan main and unique Plan TilingData
  -> collect identity/type/mapping
  -> canonical reorder
  -> create declarations and aggregate main
  -> create direct calls or graph switches
```

任何步骤失败都返回 failure；调用方不得继续使用未完成的 entry。

### 4.9.4 对子模块的修改

| 模块 | 当前职责 |
|---|---|
| `AFIRKernelBodySignature.cpp` | 补全输出、workspace 和 TilingData ABI。 |
| `GenTilingData.cpp` | 生成 Plan/stage schema 和 stage TilingData 类型。 |
| `GenKernelEntry.cpp` | 生成单 Plan main、参数 union/mapping 和 Template dispatch。 |
| `GenAggregateTilingData.cpp` | 生成 Entry schema 和 `ascendc.plan_entries`。 |
| `GenAggregateKernelEntry.cpp` | 生成 Plan declarations、aggregate union/mapping 和 aggregate dispatch。 |
| `AscendCKernelABI.cpp/.h` | 提供 workspace 类型和参数身份公共规则。 |
| `mlir/test/ut/bridge` | 运行真实 Bridge 到 aggregate 的端到端验证。 |

### 4.9.5 错误处理

#### 系统错误

这些 pass 不申请文件、runtime 或设备资源。MLIR 对象创建失败由 MLIR/LLVM 框架处理，失败结果通过 `FailureOr` 或 `LogicalResult` 返回。

#### 接口错误

以下输入必须失败：module kind 或 symbol 缺失；schema 缺失或类型错误；Template 坐标缺失/重复/越界；stage TilingData 不唯一；`ascendc.num_inputs` 缺失或越界；参数身份缺失、冲突、重复或类型不一致；tiling key 负数或重复；aggregate contract、Plan module、Plan main 或 nested TilingData 不匹配。

## 4.10 安全检查

- 未引入敏感信息、公网地址、账号、令牌或芯片型号硬编码。
- 所有外部属性 ID 在作为 vector 下标、循环边界或 case 值前检查非负和范围。
- `ascendc.num_inputs` 在转换为 unsigned 前检查 signless i32、非负和相对 TilingData 位置的范围。
- 参数 mapping 只使用经过校验的 index；TilingData 特殊 mapping 在消费时单独识别。
- 输出顺序不依赖 unordered 容器或指针地址。
- 不修改图数据边/控制边，不新增默认高频日志，不涉及 runtime 生命周期。

## 4.11 兼容性检查

- 不修改 SuperKernel、Python/C++ 对外接口、安装目录和打包内容。
- `autofuse.argument_key` 和 `autofuse.argument_name` 不再是生产 contract；旧输入需要在 Bridge/fixture 层迁移到新 metadata。
- Plan main definition 和 Entry declaration 使用相同 symbol 和函数类型。
- aggregate main 继续接收一个顶层 `AutofuseTilingData` 参数。
- workspace 参数保持 AscendC 规范类型；它不要求 input/output 参数属性。
- 当前只支持普通非 Cube/CV Plan，Cube/CV 不静默降级。

## 4.12 测试设计

### 4.12.1 测试边界

- Bridge UT：验证真实 scheduler graph 生成输入输出 index 和 Template 函数。
- Plan entry LIT：验证 stage/template dispatch、tiling key、参数 ABI 和异常诊断。
- Aggregate entry LIT/gtest：验证多 Plan contract、声明、union/mapping、graph dispatch 和 verifier。
- 端到端 gtest：验证两个 subgraph、两个 Plan，以及每个 Plan 内参数数量不同的两个 Template。

### 4.12.2 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|---|---|---|---|
| 功能 | 真实 Bridge 到 Plan main | 运行四个 pass 并 `mlir::verify` | gtest |
| 功能 | Template 参数数量不同 | Template 0 为 3 个业务参数，Template 1 为 2 个业务参数 | gtest |
| 功能 | 参数 union/mapping | 检查 Template call 是否跳过未使用的 `input:1` | gtest + IR walk |
| 功能 | tiling key dispatch | 检查 case 使用真实 key | LIT + gtest |
| 功能 | aggregate graph/plan dispatch | 检查 Plan declaration 和调用数量 | gtest |
| 功能 | workspace/TilingData 转发 | 检查类型和 call operand 顺序 | gtest |
| 异常 | 缺失/非法 `ascendc.num_inputs` | `--verify-diagnostics` | LIT |
| 异常 | 身份冲突、类型冲突、重复 key | `FailureOr`/diagnostic | LIT + gtest |
| 异常 | schema、module、contract 不完整 | `--verify-diagnostics` | LIT |
| 兼容性 | 生成 module 可 verify | `mlir::verify` | gtest |

### 4.12.3 测试命令

PR !45/!46 规则下，构建目录使用 `build/mlir-ut`；以下命令中的 `build/mlir-ut` 需要替换为实际宿主机构建目录：

```bash
source /usr/local/Ascend/cann-9.2.0/set_env.sh
cmake --build build/mlir-ut --target AFIRToAscendC af-opt -j 8
cmake --build build/mlir-ut --target ScheduleToAfirTest -j 8
/opt/llvm/bin/llvm-lit -sv \
  build/mlir-ut/autofuse/mlir/test/lit/conversion/afir-to-ascendc-gen-aggregate-entry.mlir \
  build/mlir-ut/autofuse/mlir/test/lit/conversion/afir-to-ascendc-gen-tiling-data.mlir \
  build/mlir-ut/autofuse/mlir/test/lit/conversion/afir-to-ascendc-gen-tiling-data-invalid.mlir \
  build/mlir-ut/autofuse/mlir/test/lit/conversion/afir-to-ascendc-gen-kernel-entry.mlir \
  build/mlir-ut/autofuse/mlir/test/lit/conversion/afir-to-ascendc-gen-kernel-entry-invalid.mlir
export LD_LIBRARY_PATH="$(pwd)/build/mlir-ut/autofuse:$(pwd)/build/mlir-ut/autofuse/graph_metadef/graph/ascendc_ir/generator:$(pwd)/build/mlir-ut/autofuse/graph_metadef/graph/ascendc_ir:$(pwd)/build/mlir-ut/autofuse/graph_metadef/graph/expression:$(pwd)/build/mlir-ut/autofuse/graph_metadef/graph:$LD_LIBRARY_PATH"
build/mlir-ut/autofuse/mlir/test/ut/bridge/ScheduleToAfirTest \
  --gtest_filter='ScheduleToAfirTest.TwoGraphsRunBridgeAndAllAggregateEntryPasses:IsolatedPlanEntryTest.*'
```

预期：5 个 conversion LIT 全部通过，端到端和独立 entry gtest 全部通过。测试生成的 MLIR/header/dump 只保存到临时目录，不进入版本库。

## 4.13 验收标准

- 四个 pass 的输入、输出和调用顺序与本文一致。
- 合法 Plan 能生成可 verify 的 Plan main；Template selector 使用真实 tiling key。
- 不同 Template 的参数数量和顺序可以不同，union 和 mapping 保持身份正确。
- aggregate main 能按 graph/plan contract 调用对应 Plan，并正确替换 nested TilingData。
- workspace 通过规范类型识别，不依赖旧字符串属性。
- 非法 metadata、类型、ID、schema 和 selector 明确失败。
- 5 个 conversion LIT 和端到端参数差异 gtest 通过。

### 4.13.1 设计文档检查结果

- [x] 跨特性交叉影响：已检查 SuperKernel、Autofuse 图优化、Codegen/Backend、AscendC/runtime、Python/C++ 绑定、构建打包、测试、性能和兼容性影响。
- [x] 编码红线：已检查外部索引、整数边界、ABI/API、图改写确定性、资源生命周期和 runtime 约束。
- [x] 测试设计：已覆盖四个 pass 的正向、异常、不同参数数量、union/mapping、aggregate 和 verifier 场景。
- [x] 性能评估：已说明编译期遍历、selector IR、参数 mapping、同步和产物大小影响。
