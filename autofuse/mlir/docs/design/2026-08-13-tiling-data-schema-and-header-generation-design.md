# 1 TilingData Schema 与 Header 生成设计

> 状态：已实现
> 日期：2026-08-13
> 现状复核日期：2026-08-26
> 对应实现：`GenTilingData.cpp`、`GenAggregateTilingData.cpp`、`TilingDataHeaderEmitter.cpp`、`AscendCKernelABI.cpp`

## 1.1 文档结构

本文依次说明 schema 的层级、四个相关 pass 的时序、字段收集规则、Plan/Entry contract、C++ header 输出、错误处理、接口和验证方法。第 4 章给出从一个 Plan 到多个 Plan aggregate schema 的完整过程。

# 2 简介

## 2.1 目的

本文描述 Autofuse MLIR 如何从 Plan module 的 Template facts 生成 stage TilingData、Plan TilingData 和 Entry aggregate TilingData，并将递归 `PyStructType` 输出为 AscendC 可使用的 C++ struct header。

本文面向 Bridge、AFIRToAscendC conversion、AscendC codegen 和测试维护人员。本文中的“schema”是 MLIR 中描述字段名称与字段类型的 `emitasc::PyStructType`；“header emitter”只输出 C++ 文本，不负责计算 tiling value 或写入设备。

## 2.2 范围

本文包含：

- `GenTilingData`：单个 Plan 的 stage schema、Plan schema 和 Template stage 参数物化；
- `GenAggregateTilingData`：多个 Plan 的 Entry schema 和 `ascendc.plan_entries` contract；
- `TilingDataHeaderEmitter`：递归 schema 到 C++ struct 的输出；
- 字段来源、顺序、去重、命名、ID 校验、类型校验和失败行为；
- schema 与 `GenKernelEntry`、`GenAggregateKernelEntry` 的消费关系。

本文不包含：

- Host 侧计算、填充或上传 tiling value；
- Template 函数体、Plan main 和 aggregate main 的生成，见 `2026-08-08-kernel-entry-generation-design.md`；
- Cube/CV 正向 schema/header 支持；
- runtime、Python/C++ binding、SuperKernel、默认 pipeline 和打包流程。

# 3 总体概述

## 3.1 schema 层级

每个 Plan 独立生成一套带 Plan symbol 前缀的递归类型：

```text
<plan_symbol>_Stage0TilingData
<plan_symbol>_Stage1TilingData
...
<plan_symbol>TilingData
  ├── g0_tiling_data: <plan_symbol>_Stage0TilingData
  └── g1_tiling_data: <plan_symbol>_Stage1TilingData
```

多个 Plan 再组合成 Entry schema：

```text
AutofuseTilingData
  ├── graph0_tiling_key: ui32       # 多 Plan/多 graph 场景
  ├── graph1_tiling_key: ui32
  ├── graph0_result0_tiling_data: <plan0>TilingData
  ├── graph0_result1_tiling_data: <plan1>TilingData
  ├── graph1_result0_tiling_data: <plan2>TilingData
  └── graph1_result1_tiling_data: <plan3>TilingData
```

Entry schema 的 Plan 字段由 `GenAggregateTilingData` 生成，Plan entry pass 通过 `g<N>_tiling_data` 读取 stage 字段，aggregate entry pass 通过 `tiling_field` 读取 Plan 字段。

## 3.2 四个 pass 的完整时序

每个 Plan 的前置 ABI 和 schema 处理顺序是：

```text
Bridge::PlanToAfir
  -> Template 函数带普通输入/输出和 scheduler facts
AFIRKernelBodySignature
  -> 追加 output、workspace 和通用 TilingData 参数
GenTilingData
  -> 收集 stage facts
  -> 生成 stage schema 和 Plan schema
  -> 将 Template 的 TilingData 参数替换为 stage schema
GenKernelEntry
  -> 消费 Plan schema，生成 Plan main
```

全部 Plan 完成后：

```text
GenAggregateTilingData
  -> 校验所有 Plan
  -> 生成 AutofuseTilingData
  -> 写入 ascendc.plan_entries
GenAggregateKernelEntry
  -> 消费 aggregate schema 和 plan_entries
  -> 生成 Plan declaration 和 aggregate main
TilingDataHeaderEmitter
  -> 消费一个或多个顶层 PyStructType
  -> 输出嵌套 C++ struct
```

`GenAggregateTilingData` 不复制 Plan body，`TilingDataHeaderEmitter` 不重新读取 scheduler facts；两者都只消费已经确定的 schema。

## 3.3 产品环境介绍

- 依赖项目 LLVM/MLIR、EmitAsc dialect 和标准 C++ 类型输出。
- schema pass 和 header emitter 只在编译期运行，不调用 `rt*`、`aclrt*` 或设备接口。
- header emitter 接受调用方传入的 `llvm::raw_ostream`，文件创建、打开和关闭由调用方负责。
- 生成的 header 描述 AscendC tiling struct 布局；实际 kernel/host 使用不在本文实现范围内。

## 3.4 软件功能

1. 校验 Plan module、Template 坐标、graph facts、ID 和 Cube/CV 状态。
2. 按 stage/template 收集通用字段、workspace 字段、tiling key、size variable、API tiling 和 queue/buffer 字段。
3. 以稳定顺序构造每个 stage 的 `PyStructType`。
4. 为每个 stage 构造带 Plan symbol 前缀的类型，并替换 Template stage TilingData 参数。
5. 将 stage 类型组合为 Plan 顶层 schema，并写回 `ascendc.tiling_schema`。
6. 将多个 Plan schema 组合为 `AutofuseTilingData`，并写入 `ascendc.plan_entries`。
7. 递归输出 nested C++ struct，检测同名不同布局冲突。

## 3.5 设计约束

- Plan module 必须有非空 symbol，且 `afir.module_kind = "plan"`。
- 当前只支持普通非 Cube/CV Plan；缺失 `afir.cube_type` 或值为 `-1` 表示普通路径。
- stage/template、subgraph/plan ID 使用前必须校验非负、范围和连续性。
- 每个 stage 至少有一个 Template；Template 的 `afir.asc_graph` 必须提供非负 tiling key。
- 同一 stage 内 tiling key 必须唯一；Template 按 `(stage_id, template_id)` 稳定排序。
- stage schema 字段顺序由固定收集流程决定，不依赖无序容器遍历。
- Plan schema 名称由 `makeCxxIdentifier(module_symbol) + "TilingData"` 生成；普通 `_` 保留，其他非法字符按当前标识符规则编码。
- header emitter 当前支持递归 `PyStructType`、整数类型和嵌套 struct；其他字段类型失败。
- 构建命令使用 `-j 8`。

## 3.6 假设和依赖关系

- Bridge 已写入 `afir.asc_graph`、workspace/queue/buffer/tmp buffer、size variable 和 API tiling facts。
- `AFIRKernelBodySignature` 已补全 Template ABI；workspace 类型是 `memref<ui8, 22>`。
- `GenKernelEntry` 在 `GenTilingData` 之后运行并消费 `g<N>_tiling_data`。
- `GenAggregateKernelEntry` 在 `GenAggregateTilingData` 之后运行并消费 `ascendc.plan_entries`。
- C++ header 的整数布局使用 `uint<N>_t`，这与当前 emitter 实现一致。

# 4 需求分析与设计

## 4.1 功能需求 1：生成 Plan stage schema

### 4.1.1 输入

> **勘误（2026-09-12，AFIR 公开面重构 5.1 步 7）**：候选身份与分派表已迁出 `afir` 命名空间——
> `afir.subgraph_id` + `afir.plan_id` 合为 `#afplan.candidate<subgraph, plan>`，
> `afir.stage_id` + `afir.template_id` 合为 `#afplan.variant<stage, template>`，
> `afir.template_counts_per_stage` 改为 `#afplan.stages<template_counts>`，`afir.stage_count`
> 已删除（其值恒等于 `template_counts` 的长度且无读点）。「非负」不再由各读点分别校验，
> 而由属性 verifier 一次钉死。判别与理由见
> `docs/design/2026-09-09-afir-public-dsl-architecture-design.md` 2.3 与 5.1 步 7。

`GenTilingData` 读取一个 Plan module：

```text
afir.module_kind
afir.plan_id
afir.template_counts_per_stage
ascendc.tiling_schema（由本 pass 最终写入）
```

并从带 `afir.stage_id`、`afir.template_id` 的 Template 读取：

```text
afir.asc_graph
afir.workspace_tensor_ids
afir.queue_ids
afir.buffer_ids
afir.tmp_buffer_ids
Template size variables and API tiling metadata
```

具体字段以当前 Bridge facts 和 pass 支持的属性为准。

### 4.1.2 处理流程

1. `validatePlanModule()` 检查 symbol、module kind 和 Cube/CV。
2. 收集带 stage/template 坐标的 Template；没有坐标的普通函数不作为 Template。
3. 校验坐标非负，并按 `(stage_id, template_id)` 排序。
4. 根据 `template_counts_per_stage` 为每个 stage 收集完整 Template 集合。
5. 从每个 Template 的 `afir.asc_graph` 读取 tiling key，校验非负和 stage 内唯一。
6. 收集并合并本 stage 所有 Template 的字段。第一次出现的字段按收集顺序加入，重复字段只保留一次。
7. 使用 Plan symbol 构造类型名：

   ```text
   <makeCxxIdentifier(module_symbol)>_Stage<N>TilingData
   ```

8. 查找每个 Template 的通用 TilingData 参数；如果参数已有该 stage type，则保留；如果没有，则在函数参数末尾追加 stage type。
9. 以 `g<N>_tiling_data` 为字段名，把每个 stage type 组合成 Plan 顶层 schema。
10. 以 `makeCxxIdentifier(module_symbol) + "TilingData"` 命名 Plan 顶层 schema，并写入 `ascendc.tiling_schema`。

### 4.1.3 字段顺序

当前实现按照以下类别追加字段，具体是否出现取决于该 Plan 的 facts：

```text
block_dim
corenum
ub_size
hbm_size
workspace<ID>
size variable
API tiling fields
q<ID>_size
b<ID>_size
```

workspace、queue 和 buffer ID 使用显式排序后的集合；API tiling 字段要求 schema、成员数量、名称、顺序和类型一致，否则失败。

### 4.1.4 输出

例如一个 Plan 的输出是：

```text
ascendc.tiling_schema = !emitasc.py_struct<"two_graphs_subgraph0_plan0TilingData", [
  !emitasc.py_struct<"two_graphs_subgraph0_plan0_Stage0TilingData", [ui32, ui32, ui32, ui32, ui32],
  ["block_dim", "corenum", "ub_size", "hbm_size", "tiling_key"]>
], ["g0_tiling_data"]>
```

## 4.2 功能需求 2：生成 Entry aggregate schema

### 4.2.1 输入

```cpp
FailureOr<OwningOpRef<ModuleOp>> runGenAggregateTilingData(
    llvm::ArrayRef<mlir::ModuleOp> planModules);
```

每个 Plan 必须提供：

```text
afir.module_kind = "plan"
afir.subgraph_id >= 0
afir.plan_id >= 0
ascendc.tiling_schema = TypeAttr<PyStructType>
```

### 4.2.2 处理流程

1. 空输入失败。
2. 以第一个 Plan 的 MLIRContext 为基准，校验所有 Plan 使用相同 context。
3. 校验 module kind、subgraph/plan ID、schema 和 Cube/CV。
4. 按 `(subgraph_id, plan_id)` 排序。
5. 要求 subgraph ID 连续；每个 subgraph 的 plan ID 从 0 连续。
6. 多 Plan 或多 graph 场景添加 `graph<N>_tiling_key : ui32`；单 graph 单 Plan 不添加 graph selector。
7. 对每个 Plan 添加 `graph<N>_result<M>_tiling_data`，类型直接引用该 Plan 的顶层 schema。
8. 创建名为 `AutofuseEntry` 的 Entry module，设置 `afir.module_kind = "entry"`。
9. 将 aggregate schema 写入 `ascendc.tiling_schema`。
10. 写入最小 contract `ascendc.plan_entries`。

### 4.2.3 contract

每项 contract 只有：

```text
subgraph_id
plan_id
```

例如：

```text
{subgraph_id = 0, plan_id = 1, tiling_field = "graph0_result1_tiling_data"}
```

contract 不重复存储 TilingData 类型；类型唯一从 aggregate schema 的 `tiling_field` 查询。

## 4.3 功能需求 3：输出 C++ header

### 4.3.1 输入和调用方

```cpp
LogicalResult emitTilingDataHeader(
    ArrayRef<emitasc::PyStructType> schemas,
    llvm::raw_ostream &output);
```

调用方负责准备顶层 schema 和输出流。正常调用传入 aggregate schema；单 Plan 测试也可以直接传入 Plan schema。

### 4.3.2 处理流程

1. 输出：

   ```cpp
   #pragma once
   #include <cstdint>
   ```

2. 对每个顶层 schema 深度优先递归。
3. 检查 schema name 合法且字段 name 是合法 C++ 标识符。
4. 通过 layout map 记录已经处理的 schema。
5. 同名 schema 再次出现时，如果字段布局不同，返回 failure；布局相同则跳过重复输出。
6. 先输出 nested struct，再输出外层 struct。
7. `IntegerType` 输出为 `uint<N>_t`；`PyStructType` 输出为其名称。
8. 遇到 malformed `TypeAttr`、不支持的字段类型或非法标识符时返回 failure。

### 4.3.3 使用示例

当前已实现的调用方使用 `raw_string_ostream` 接收输出：

```cpp
std::string header;
llvm::raw_string_ostream output(header);
LogicalResult result = mlir::afir::emitTilingDataHeader({aggregateSchema}, output);
if (failed(result)) {
  // 调用方终止后续 codegen，并报告 schema/header 错误。
}
```

生成结果示例：

```cpp
struct two_graphs_subgraph0_plan0_Stage0TilingData {
    uint32_t tiling_key;
};

struct two_graphs_subgraph0_plan0TilingData {
    two_graphs_subgraph0_plan0_Stage0TilingData g0_tiling_data;
};

struct AutofuseTilingData {
    uint32_t graph0_tiling_key;
    two_graphs_subgraph0_plan0TilingData graph0_result0_tiling_data;
};
```

## 4.4 非功能需求

### 4.4.1 可维护性

- schema builder 负责解释 Bridge facts 和确定字段顺序；emitter 不重新解释 scheduler。
- Plan schema、aggregate schema 和 C++ emitter 分层，contract 只保存 aggregate entry 所需的最小映射。
- 类型命名统一通过 `makeCxxIdentifier`；普通 `_` 保留，非法字符仍使用可区分编码。
- 临时 `AggregatePlanInfo` 等结构只在调用期间存在，不进入产品 ABI。

### 4.4.2 可测试性

- `GenTilingData` 使用 af-opt LIT 验证正向 schema 和 invalid diagnostics。
- aggregate 使用多个独立 Plan root module 验证排序、连续 ID 和 nested schema。
- emitter 使用 `raw_string_ostream` 验证字段顺序、递归输出、重复布局和非法类型。
- 端到端 gtest 验证四个 Plan 的 aggregate schema 被后续 entry pass 消费。

### 4.4.3 可移植性

不硬编码操作系统、芯片型号或 CANN 安装路径；header 只输出标准 `<cstdint>` 整数类型和已校验的 C++ 标识符。

### 4.4.4 可靠性

- 所有 module attribute、ID、字段和 TypeAttr 在使用前校验。
- 字段和 Plan 输出顺序由显式排序和固定收集顺序决定。
- emitter 不打开文件、不申请 runtime 资源，输出流所有权属于调用方。
- malformed schema、同名冲突和不支持类型返回 failure，不返回部分成功状态。

### 4.4.5 特性交叉影响

| 场景 | 适用性 | 分析说明 |
|---|---|---|
| SuperKernel Python 接口 | 不适用 | 不修改 Python 包、选项和 wheel。 |
| SuperKernel C++/AOT 接口 | 不适用 | 不修改 SuperKernel 动态库、AOT ABI 和设备测试。 |
| Autofuse 图优化 | 适用 | 只读取 facts，不改写 ASCIR 数据边和控制边。 |
| Autofuse Codegen/Backend | 适用 | schema/header 是后续 tiling 和 kernel codegen 的输入。 |
| AscendC API / Runtime 交互 | 适用 | 不新增 runtime 调用；字段类型和顺序构成后续 host/device tiling ABI。 |
| Python/C++ 混合绑定 | 不适用 | 不修改 pyautofuse 和 CPython 扩展。 |
| 构建与打包 | 适用 | 影响既有 `AFIRToAscendC` target；遵循 PR !45 include 和 PR !46 `build/mlir-ut` 目录规则。 |
| 测试与覆盖率 | 适用 | 需要 schema、aggregate、header、端到端和异常测试。 |
| 性能与日志 | 适用 | 评估编译期字段遍历、递归输出和 header 大小；不新增默认日志。 |
| 兼容性 | 适用 | 类型名、字段名、字段顺序和整数宽度是内部 tiling ABI contract。 |

## 4.5 性能

### 4.5.1 编译时长

单 Plan schema 构造遍历 Template 和 facts，复杂度与 Template 数、stage 数、字段数和 ID 数量线性相关。aggregate 只遍历 Plan metadata，不遍历 Template body。header emitter 对每个唯一 schema 和字段递归一次。

### 4.5.2 执行性能

schema 和 header 生成只发生在编译期，不增加 kernel launch、runtime 调度或内存拷贝。字段布局变化可能改变 host/device tiling ABI，因此通过 schema/header 测试约束布局。

### 4.5.3 内存和产物大小

临时 schema 和 contract 只存在于编译期。Entry 不复制 Plan body；header 大小与唯一 struct 和字段数量线性相关，不增加常驻 runtime 内存。

## 4.6 接口设计

### 4.6.1 当前已实现接口

```cpp
std::unique_ptr<mlir::Pass> createGenTilingDataPass();
void registerGenTilingDataPass();

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> runGenAggregateTilingData(
    llvm::ArrayRef<mlir::ModuleOp> planModules);

mlir::LogicalResult emitTilingDataHeader(
    llvm::ArrayRef<mlir::emitasc::PyStructType> schemas,
    llvm::raw_ostream &output);
```

调用时序：

```text
每个 Plan：AFIRKernelBodySignature -> GenTilingData
全部 Plan：runGenAggregateTilingData
Entry 消费：GenAggregateKernelEntry
header 输出：emitTilingDataHeader
```

失败时由调用方停止后续 entry/codegen；不把不完整 schema 或 header 作为成功结果。

### 4.6.2 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|---|---|---|---|
| 接口说明 | 是否需要接口评审 | 是 | 涉及 schema 属性、aggregate contract 和 header 输出。 |
| 接口说明 | 是否需要补充文档 | 是 | 本文定义字段、类型、顺序、时序和失败行为。 |
| 接口兼容 | 行为是否兼容 | 是 | 旧 module kind/contract 命名和旧 fallback 不再作为当前输入。 |
| 接口兼容 | ABI/API 是否兼容 | 是 | 不修改产品外部 API；内部 TilingData schema contract 有明确规则。 |
| 接口约束 | 约束不满足时是否清晰报错 | 是 | module、ID、字段、schema、context 和类型错误均失败。 |
| 接口测试 | 是否需要独立接口用例 | 是 | 单 Plan、aggregate、nested header 和异常测试均需要。 |

## 4.7 软件设计

### 4.7.1 关键数据结构

- `emitasc::PyStructType`：保存 schema 名称、字段类型数组和字段名称数组。
- `AggregatePlanInfo`：aggregate 阶段暂存 subgraph/plan ID、Plan module、Plan schema 和字段名。
- `AggregatePlanInfo` 集合：按 `(subgraph_id, plan_id)` 排序后生成 Entry schema。
- header emitter 的 layout map：记录同名 schema 的字段布局，防止冲突定义。

### 4.7.2 关键算法

1. 校验 Plan module，再按坐标收集 Template。
2. 按 stage 合并字段，使用显式排序保证 workspace/queue/buffer 字段稳定。
3. 用 `makeCxxIdentifier` 生成合法且可区分的 C++ 类型名。
4. 将 stage type 组合成 Plan schema，再按 subgraph/plan 组合 Entry schema。
5. 用 `plan_entries.tiling_field` 建立 aggregate schema 与 Plan declaration 的映射。
6. header emitter 深度优先输出 nested struct，并校验同名布局一致性。

### 4.7.3 流程设计

```text
Plan module
  -> validatePlanModule
  -> collect/sort Template
  -> collect stage facts and tiling keys
  -> build <plan>_Stage<N>TilingData
  -> materialize Template stage argument
  -> build <plan>TilingData
  -> set ascendc.tiling_schema

all Plan modules
  -> validate context/kind/IDs/schema
  -> sort by subgraph/plan
  -> build AutofuseTilingData
  -> set ascendc.plan_entries

top-level schema
  -> validate names/types/layout
  -> recursively emit nested structs
  -> emit outer struct
```

### 4.7.4 对子模块的修改

| 模块 | 当前职责 |
|---|---|
| `conversion/AFIRToAscendC/GenTilingData.cpp` | 单 Plan stage/Plan schema和 Template stage 参数物化。 |
| `conversion/AFIRToAscendC/GenAggregateTilingData.cpp` | 多 Plan aggregate schema、Entry module 和 `plan_entries`。 |
| `conversion/AFIRToAscendC/Emit/TilingDataHeaderEmitter.cpp/.h` | 递归 schema 到 C++ header。 |
| `conversion/AFIRToAscendC/AscendCKernelABI.cpp/.h` | workspace 类型和 ABI 相关类型约定。 |
| `mlir/test/ut/bridge` | 真实 Bridge、schema、aggregate 和 header 验证。 |

### 4.7.5 错误处理

#### 系统错误

实现不直接操作 runtime 或文件资源。输出流写入由调用方管理；MLIR 对象构造失败沿用框架失败传播。

#### 接口错误

以下情况失败：Plan symbol/kind 缺失；Cube/CV 不支持；Template、graph facts、schema 或 ID 非法；stage/template 坐标重复或越界；tiling key 负数或重复；API tiling 布局冲突；Plan context 不一致；aggregate Plan ID 不连续；header 遇到非法 C++ 标识符、malformed TypeAttr、未知字段类型或同名不同布局。

## 4.8 安全检查

- 不包含敏感信息、公网地址、账号、令牌或芯片类型硬编码。
- 所有外部 ID 在数组访问、字段生成和循环前校验非负、范围和连续性。
- schema 字段构造不进行未检查的 size、offset、shape 或内存申请运算。
- 输出顺序不依赖 `unordered_map`、指针地址或未指定遍历顺序。
- emitter 不持有全局可变状态，不申请 runtime 资源，不修改图语义。

## 4.9 兼容性检查

- 不修改 SuperKernel、Python/C++ 对外 API、安装目录和打包内容。
- 当前 contract 使用 `afir.module_kind = "plan"`、`ascendc.tiling_schema` 和 `ascendc.plan_entries`。
- Plan schema 类型名由 Plan module symbol 派生；普通下划线保留，非法字符按当前规则编码。
- aggregate 顶层类型固定为 `AutofuseTilingData`，Plan 字段固定为 `graph<N>_result<M>_tiling_data`。
- header emitter 当前将整数输出为 `uint<N>_t`，不提供 Cube/CV 或未知类型降级。

## 4.10 测试设计

### 4.10.1 测试边界

- conversion LIT：覆盖 `GenTilingData` 正向和异常 schema。
- isolated gtest：覆盖 Plan schema、aggregate schema、nested header、非法类型和同名冲突。
- Bridge 端到端 gtest：覆盖两个 subgraph、两个 Plan，以及每个 Plan 内两个参数数量不同的 Template。
- 最终产物验证：检查每个阶段 MLIR 和 `TilingData.h` 的类型名、字段名、字段顺序和嵌套关系。

### 4.10.2 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|---|---|---|---|
| 功能 | stage 字段收集和稳定顺序 | schema walk | LIT/gtest |
| 功能 | Plan schema 命名和嵌套 stage | TypeAttr 检查 | LIT/gtest |
| 功能 | 多 subgraph/plan aggregate schema | contract/schema walk | gtest |
| 功能 | 递归 C++ header | `raw_string_ostream` 字符串检查 | gtest |
| 功能 | header 普通下划线命名 | `makeCxxIdentifier` 和最终 header 检查 | gtest/E2E |
| 异常 | 缺少 symbol、module kind、schema 或 graph facts | `--verify-diagnostics` | LIT |
| 异常 | 非连续 ID、重复 key、API 布局冲突 | failure/diagnostic | LIT/gtest |
| 异常 | unsupported header type、同名不同布局 | `LogicalResult` 检查 | gtest |
| 特性交叉 | schema 被 Plan/aggregate entry 消费 | 四 pass 端到端 | gtest |

### 4.10.3 测试命令

PR !45/!46 规则下，MLIR UT 使用 `build/mlir-ut`：

```bash
source /usr/local/Ascend/cann-9.2.0/set_env.sh
cmake --build build/mlir-ut --target AFIRToAscendC af-opt -j 8
cmake --build build/mlir-ut --target ScheduleToAfirTest -j 8
/opt/llvm/bin/llvm-lit -sv \
  build/mlir-ut/autofuse/mlir/test/lit/conversion/afir-to-ascendc-gen-tiling-data.mlir \
  build/mlir-ut/autofuse/mlir/test/lit/conversion/afir-to-ascendc-gen-tiling-data-invalid.mlir \
  build/mlir-ut/autofuse/mlir/test/lit/conversion/afir-to-ascendc-gen-aggregate-entry.mlir
export LD_LIBRARY_PATH="$(pwd)/build/mlir-ut/autofuse:$(pwd)/build/mlir-ut/autofuse/graph_metadef/graph/ascendc_ir/generator:$(pwd)/build/mlir-ut/autofuse/graph_metadef/graph/ascendc_ir:$(pwd)/build/mlir-ut/autofuse/graph_metadef/graph/expression:$(pwd)/build/mlir-ut/autofuse/graph_metadef/graph:$LD_LIBRARY_PATH"
build/mlir-ut/autofuse/mlir/test/ut/bridge/ScheduleToAfirTest \
  --gtest_filter='ScheduleToAfirTest.TwoGraphsRunBridgeAndAllAggregateEntryPasses:IsolatedPlanEntryTest.*'
```

预期：指定 LIT 全部通过，two_graph 端到端和 isolated schema/header gtest 全部通过。dump 和 header 只保存到临时目录，不提交到版本库。

## 4.11 验收标准

- Plan stage/Plan schema 和 Entry aggregate schema 与当前字段、类型、命名及排序规则一致。
- `ascendc.plan_entries` 是当前 aggregate contract 名称，且不重复保存 schema 类型。
- 普通 module 名中的下划线在 C++ 类型名中保留；非法字符仍生成合法且不冲突的标识符。
- header emitter 能递归输出合法 schema，并拒绝 malformed schema、unsupported type 和布局冲突。
- schema 被 `GenKernelEntry` 和 `GenAggregateKernelEntry` 正确消费。
- 正向、异常、端到端和 header 测试均通过。

### 4.11.1 设计文档检查结果

- [x] 跨特性交叉影响：已按 SuperKernel、Autofuse、AscendC/runtime、绑定、构建打包、测试、性能和兼容性逐项检查。
- [x] 编码红线：已检查 ID/索引范围、整数边界、资源生命周期、ABI/API、图确定性和 runtime 约束。
- [x] 测试设计：已覆盖 schema、aggregate、header、异常输入、命名规则和四 pass 端到端链路。
- [x] 性能评估：已说明 schema/aggregate/header 的编译期复杂度、运行时无额外开销和产物大小影响。
