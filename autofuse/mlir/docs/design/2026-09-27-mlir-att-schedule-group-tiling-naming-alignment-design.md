# 简介

## 目的

把 MLIR 路径生成的 TilingData 布局与成员命名，对齐到 ATT 生成的 host tiling 程序**已经写死**的 schedule group 命名上，
使多 stage（一个 plan 含多个 schedule group，例如多核 reduce 的 RCore 两段式）的算子能够通过 host 编译并在真机跑通。

## 范围

涉及：

- `conversion/AFIRToAscendC`：`GenTilingData` 的 per-stage schema 命名、`GenKernelEntry` 的 tiling 形参 ABI
- `artifact/KernelSource`：`GenAggregateTilingData` 的最外层 TilingData 布局、`GenAggregateKernelEntry` 的聚合调用
- `artifact/TilingSource`：`TilingProgramSpec` 的 stage 字段路径、`TilingFuncEmitter` 的 `optiling::` 作用域
- `common/tiling_schema`：新增命名单点 `schedule_group_names.h`，修正 `tiling_field_schema` 的 stage path
- `dialect/AFIR`：`#afir.kernel_arg` 中 `tiling` 角色的 `index` 语义
- `bridge/AttSchemaAdapter`：生成代码的字段引用交叉校验增加 group 成员维度

不涉及：ATT 侧命名（不可配置，见下）、tiling 求解算法、device kernel 代码生成、plan 选择策略。

# 总体概述

## 软件概述

### 项目介绍

本仓 `autofuse/mlir` 是 Ascend NPU 编译器后端的 MLIR 路径：承接 Inductor/TF 导出的计算图，做图优化、自动融合、地址计算与代码生成，
产出 device kernel 与 host tiling 两份源码，交由 `ascendc_compile.py` 编译链接成 `kernel.so` / `wrapper.so`。

### 产品环境介绍

- 硬件：Ascend 910B（`dav-2201`），`aiv_num=40`，`ub_size=196608`
- CANN：容器内 `cann-9.2.0`（真机 toolkit 亦为 `cann-9.2.0` 安装路径）
- host tiling 编译器：`$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/bisheng -x c++`
- 上层入口：`torch_npu._inductor` + 本仓 `autofuse` python 包

## 软件功能

MLIR 路径为每个被选中的 plan 生成：

1. 一份最外层 `AutofuseTilingData` 结构体（`autofuse_tiling_data.h`）
2. 一份 host tiling 程序（`AutofuseTiling` / `CalculateWorkspaceSize`），其 schedule group 求解部分由 ATT 生成
3. 一份 device kernel（`AutofuseLaunch` + `__global__` 入口）

本次变更只改 1、2 中「名字与布局」这一层，不改任何求解或计算逻辑。

## 设计约束

1. **ATT 的 group 命名不可配置，只能由 MLIR 侧对齐。**
   `autofuse/att/base/model_info.h:115` 的 `ScheduleGroupIdent` 从 `(asc_graph_id, impl_graph_id, group_id)` 硬算出两个拼法：

   ```cpp
   GetGroupPrefix() = "AscGraph<g>ScheduleResult<p>G<i>"   // 结构体 / 命名空间
   GetItemPrefix()  = "graph<g>_result<p>_g<i>"            // 最外层 TilingData 的成员（+ "_tiling_data"）
   ```

   两者都不经过 `TilingCodeGenConfig`；该配置里只有 `tiling_data_type_name` 一项，管的是最外层那一个类型名。
   ATT 生成的 host tiling 直接按上面的名字访问成员：

   ```cpp
   AscGraph0ScheduleResult0G0::GetTiling(graph0_result0_g0_tiling_data, workspace_map, tiling_case_id);
   ```

   `impl_graph_id` 在 ATT 那边是 ScheduleResult（= 本仓的 plan）序号，不是 impl graph / template 序号；名字是历史遗留。

2. **最外层 TilingData 里没有「plan」这一层。** ATT 把所有 group 平铺在 root 上，legacy `ProcessMultiGroup`
   （`autofuse/codegen/codegen_tiling_data.cpp`）产出的形状被 `autofuse/tests/ut/codegen/test_codegen_tiling_data.cpp`
   钉住：`block_dim, corenum, ub_size, hbm_size, graph<g>_tiling_key, + 每个 group 一个成员`。

3. **单 stage（flat）场景的字节布局必须保持不变。** flat = 1 子图 && 1 plan && 1 stage，与
   `common/tiling_schema/tiling_field_schema.cpp` 的 `is_flat` 判据一致。flat 时 ATT 用裸 `tiling_key`，
   root 不能有自己的 `block_dim/corenum/ub_size/hbm_size`——这些以 forwarding accessor 的形式转发到唯一的 group。
   在 flat 场景给 root 加上真实标量会**静默**出错：ATT 写 root 的那一份，device 读 group 里的那一份。

4. `CalculateWorkspaceSize` / `GetWorkspaceSize` 必须留在全局作用域：ATT 在 `namespace optiling {` 打开之前就全局前向声明了
   `uint32_t GetWorkspaceSize(const AutofuseTilingData &);`。

5. 不新增硬编码芯片类型、框架类型判断；不扩大修改范围。

## 假设和依赖关系

- 假设一个 plan 的 stage 序号连续、从 0 起算，且与 ATT 的 `group_id` 同序。当前 `ScheduleToAfir` 即按此产出。
- 依赖 `afplan.candidate` 属性在 `GenTilingData` 运行时已挂在 module 上（承载 `subgraph` / `plan`）。
- 依赖 ATT `ScheduleGroupIdent` 的拼法不变。若 ATT 改名，`schedule_group_names.h` 是唯一需要跟改的地方。

# 需求分析与设计

## 整体介绍

变更前，MLIR 路径把最外层 TilingData 生成成**三层**：

```
AutofuseTilingData
└── plan0TilingData graph0_result0_tiling_data      // plan 这一层是 MLIR 自己加的
    ├── mlir_<module>_Stage0TilingData g0_tiling_data
    └── mlir_<module>_Stage1TilingData g1_tiling_data
```

而 ATT 生成的 host tiling 按**两层**、平铺访问：

```
AutofuseTilingData
├── AscGraph0ScheduleResult0G0TilingData graph0_result0_g0_tiling_data
└── AscGraph0ScheduleResult0G1TilingData graph0_result0_g1_tiling_data
```

单 stage 时靠 `emitRootForwardingAccessors`（`artifact/KernelSource/Emit/TilingDataHeaderEmitter.cpp:58`）在 root 上补一层
转发访问器，勉强兜住；该函数在 `owners.size() != 1` 时直接返回，所以多 stage 没有任何桥接，症状是 host 编译报

```
member access into incomplete type 'AscGraph0ScheduleResult0G0TilingData'
```

——MLIR 各 pass 全绿、device kernel 也编得过，只在真机 host 编译期才炸。

本次变更的方案是**把聚合层压平**：删掉 MLIR 自造的 plan 一层，stage 结构体直接按 ATT 的公式命名并挂在 root 上；
kernel ABI 相应从「一个 tiling 形参」改成「一 stage 一个 tiling 形参」。

数据流（变更后）：

```
CandidateAttr(subgraph, plan)
   │
   ├─ GenTilingData ──► per-stage TilingSchema
   │     类型名 = MakeScheduleGroupTilingTypeName(subgraph, plan, stage)
   │     字段名 = MakeScheduleGroupTilingFieldName(subgraph, plan, stage)
   │     （plan 级 schema 退化为**载体**，只按 stage 次序装 names/types，不发射结构体）
   │
   ├─ GenKernelEntry ──► plan `_main` 的 tiling 形参按 stage 展开
   │     #afir.kernel_arg<role = tiling, index = <stage>>
   │
   ├─ GenAggregateTilingData ──► 最外层 root 布局
   │     [block_dim, corenum, ub_size, hbm_size]（非 flat 才有）
   │     + graph<g>_tiling_key（非 flat 才有）
   │     + workspace<id>...
   │     + 每 stage 一个 group 成员（平铺）
   │
   ├─ GenAggregateKernelEntry ──► 聚合 main 按 stage 次序把 group 成员传进 plan `_main`
   │
   └─ TilingProgramSpec ──► stage 的 tiling_field_name 为一级：graph<g>_result<p>_g<i>_tiling_data
```

## 功能需求

### 功能需求 1：schedule group 命名单点

1. 介绍

   新增 `common/tiling_schema/schedule_group_names.h`，提供 ATT 两条命名公式的唯一实现，供 conversion / artifact / bridge 三处共用。

2. 输入

   `(subgraphId, planId, stageId)`，均为非负 `int64_t`。序号来自 `afplan.candidate` 与 stage 循环下标，不来自外部输入。

3. 处理

   ```cpp
   MakeScheduleGroupTilingFieldName(g, p, i) -> "graph<g>_result<p>_g<i>_tiling_data"
   MakeScheduleGroupTilingTypeName (g, p, i) -> "AscGraph<g>ScheduleResult<p>G<i>TilingData"
   ```

4. 输出

   `std::string`。头文件只依赖 `<string>` / `<cstdint>`，以便被只有 MLIR 少量 include 路径的 artifact / conversion 包含。

5. 异常

   无。非法（负）序号在上游属性校验处拦住，不在此处重复判断。

### 功能需求 2：最外层 TilingData 平铺

1. 介绍

   `GenAggregateTilingData` 取消 plan 一层，把每个 stage 的 group 结构体直接作为 root 成员。

2. 输入

   `AggregatePlanInfo` 列表（每个 plan 的 candidate + 载体 schema）。

3. 处理

   - `getStageCount(info)` = 载体 schema 的 `names` 个数
   - `isFlatLayout(infos)` = 只有一个 plan && `subgraph == 0` && stage 数为 1
   - `addCommonScalarFields`：非 flat 才在 root 加 `block_dim/corenum/ub_size/hbm_size`（`ui32`）
   - `addGraphSelectorFields`：非 flat 才加 `graph<g>_tiling_key`
   - `addWorkspaceFields`：不变
   - `addPlanTilingFields`：按 plan、再按 stage 次序平铺 group 成员

   root 字段次序固定为：标量 → selector → workspace → group 成员。

4. 输出

   最外层 `TilingSchema`；`attachPlanContracts` 在 plan entry 上写入 `tiling_fields`（ArrayAttr，一 stage 一个名字）。

5. 异常

   载体 schema 的 `names` / `types` 个数不一致，或 group 成员名与共享公式算出的不一致时报错。

### 功能需求 3：kernel ABI 按 stage 展开 tiling 形参

1. 介绍

   plan `_main` 由「收一个 tiling 结构体」改为「一 stage 收一个 group 结构体」。

2. 输入

   载体 schema 的 `types`（按 stage 次序）。

3. 处理

   - `signature.tilingArgument` 记录第一个 tiling 形参下标，其后连续 `stageCount` 个形参依次为 stage 0..N-1
   - 读 selector 时用 `ctx.arguments[ctx.tilingArgument + stage]` 取对应 stage，再读其 `tiling_key`
   - 每个 tiling 形参挂 `#afir.kernel_arg<role = tiling, index = <stage>>`
   - `GenAggregateKernelEntry` 的 `findTilingArguments` 校验：tiling 形参个数 == stage 数、下标连续、`index` 与 stage 一致

4. 输出

   带完整 `kernel_arg` 标注的 `_main`；聚合 main 按 `argumentMapping` 走到哨兵槽位时按 stage 计数取 group 成员。

5. 异常

   个数不符、下标不连续、`index` 与 stage 不符，均报错并终止 pass。

### 功能需求 4：`#afir.kernel_arg` 中 tiling 的 `index` 语义

1. 介绍

   `tiling` 角色的 `index` 变为**可选**，带号时号即 stage id。

2. 处理

   `KernelArgAttr::verify` 三分支：

   - 数据参数（input/output/workspace_offset）：`index < 0` 报 `requires a non-negative index`
   - `workspace`：`index != -1` 报 `cannot carry an index`（全 kernel 至多一块 workspace）
   - `tiling`：`index < -1` 报 `index must be a stage id or absent`

3. 异常

   判重仍按 `(role, index)`；两个不带号的 tiling 共用 `-1` 这个键，仍然互斥。

### 功能需求 5：stage 字段路径降为一级

1. 介绍

   `tiling_field_schema.cpp` 的 `NestedStagePath` 此前拼两级（plan + stage），与该函数注释声明的契约相反；
   `TilingProgramSpec::BuildStageSpecs` 也据此拼两级路径。现统一为一级。

2. 处理

   `BuildStageSpecs` 由 stage 循环下标推出 stage id，用共享公式算出期望名并与载体 schema 里的名字比对，
   不一致即报错；`tiling_field_name` 取该名字。

### 功能需求 6：`optiling::` 作用域限定

1. 介绍

   `CalculateWorkspaceSize` 在全局作用域发射，其中引用的 ATT 辅助符号需要区别对待。

2. 处理

   ATT 把 `Max/Min/Abs/Ceiling/Floor/Mod` 发射成 `namespace optiling` 内的**函数模板**，
   把 `Pow/Rational/Log/ExpectEq/ExpectNe/ExpectLe/ExpectLt/LogicAnd/LogicOr` 发射成**宏**。
   因此 `TilingFuncEmitter` 只给前六个加 `optiling::` 前缀（`kOptilingScope`），宏一律不加。

3. 异常

   若 ATT 未来把某个宏改成模板，症状是 `'X' was not declared in this scope; did you mean 'optiling::X'?`，
   修法是把该符号加入 `kOptilingScope` 限定列表。

### 功能需求 7：生成代码字段引用的 group 成员交叉校验

1. 介绍

   `ValidateGeneratedFieldReferences` 此前只扫标量访问器，正是这一点让命名不匹配一路漏到 device 编译器。

2. 处理

   新增 `detail::ExtractReferencedGroupMemberNames(const std::string &) -> std::set<std::string>`：
   匹配前面紧跟 `.` 或 `->` 的 `<ident>_tiling_data`。校验时用 schema 中非空、非 `"fusion"` 的 `field.path`
   构造 `known_group_members`，逐个核对生成代码里引用到的 group 成员。

3. 异常

   引用到 schema 中不存在的 group 成员时返回失败并写 `diagnostic`。

## 非功能需求

### 可维护性

命名从三处散落的字符串拼接收敛到一个头文件的两个函数，并在头注释里写明「名字的产地在 ATT」、失败症状与排查路径。
ATT 改名时只需改一处。

### 可测试性

- 平铺后的 root 字段名可在 UT 中逐个断言（`FlattensTwoStagePlanIntoAggregateRoot` 断言 8 个 root 字段名的精确次序）
- group 成员引用的交叉校验可用手写源码片段直接驱动（`detail::` 暴露的目的）
- lit 用例可直接 `CHECK-NOT` 旧的 plan 一层（`CHECK-NOT: struct plan0TilingData`、`CHECK-NOT: emitasc.member {{.*}} "graph0_result0_tiling_data"`）

### 可移植性

不涉及芯片型号、指令集或框架判断。命名公式与 CANN 版本无关，只与 ATT 源码一致。

### 可靠性

三道新增的早失败关卡：属性校验（stage id 合法性、判重）、`findTilingArguments`（ABI 形状）、
`ValidateGeneratedFieldReferences`（生成代码 ↔ schema 一致性）。原先这三处都是静默通过、到 host 编译期才炸。

### 特性交叉影响

| 特性 | 影响 | 处理 |
|------|------|------|
| 单 stage（RCommon）elementwise / reduce | root 成员名由 `graph0_result0_tiling_data` 改为 `graph0_result0_g0_tiling_data`，类型名由 `plan0TilingData` 改为 `AscGraph0ScheduleResult0G0TilingData` | **字节布局不变**；forwarding accessor 保留；flat 场景不给 root 加标量。真机 plan 0 全量回归通过 |
| 多 stage（RCore 两段式 reduce） | 从「host 编译失败」变为可用 | 真机小 shape 与 `(512,1024)` 通过 |
| workspace 偏移链 | root 的 workspace 字段位置不变（在 group 成员之前） | lit `afir-to-ascendc-workspace-offset*.mlir` 覆盖 |
| 多子图聚合 | `isFlatLayout` 要求 `subgraph == 0` 且单 plan 单 stage，多子图自动走非 flat 分支 | `IsolatedPlanEntryTest` / `ScheduleToAfirTest` 覆盖 |
| ATT schema 交叉校验 | `RunAttForCandidates` 传入的 `TilingFieldSchema` 仍仅作交叉校验，ATT 名字仍由 legacy `ScheduledResult` 产出 | 不变，本次只是把校验做严 |
| PGO / tiling cache | 只读 root 成员名，随生成头文件一起变 | 无需单独适配 |

## 性能

### 编译时长

新增的两处校验是对生成源码做一次正则扫描与字符串比较，量级为单个 kernel 的 host 源码（数千行），
相对 `RunCodegen`（本次真机日志为 274 ms）可忽略。

### 执行性能

device 侧无变化：单 stage 场景 TilingData 字节布局与访问路径完全不变；多 stage 场景此前不可用，无退化基线。

### 内存和产物大小

单 stage：`AutofuseTilingData` 大小不变。
多 stage：root 由「一个 plan 结构体」变为「N 个 group 结构体平铺」，并在非 flat 时新增 4 个 `uint32_t` 标量与一个 selector，
增量为 `5 * sizeof(uint32_t) = 20` 字节（本次真机两 stage 实测 root 为 6 个标量 + 2 个 group 结构体）。

已知的、**本次不处理**的问题：per-stage workspace 在 `CalculateWorkspaceSize` 中按 stage 累加，
对共享同一块 workspace 的两个 stage 存在超额分配。已记录，不在本变更范围内。

## 接口设计

### 新增/修改接口描述

新增（仓内头文件，非对外 ABI）：

```cpp
// common/tiling_schema/schedule_group_names.h
namespace mlir::common {
std::string MakeScheduleGroupTilingFieldName(int64_t subgraphId, int64_t planId, int64_t stageId);
std::string MakeScheduleGroupTilingTypeName (int64_t subgraphId, int64_t planId, int64_t stageId);
}

// bridge/AttSchemaAdapter/AttSchemaAdapter.h
namespace mlir::bridge::detail {
std::set<std::string> ExtractReferencedGroupMemberNames(const std::string &source);
}
```

修改：

| 接口 | 变更 | 兼容性 |
|------|------|--------|
| `buildStageSchema(functions, candidate, stage)` | 新增 `candidate` 形参 | 仓内，随调用点同步改 |
| `kAscendCPlanEntryTilingFieldKey` | 值由 `"tiling_field"` 改为 `"tiling_fields"`，类型由 StringAttr 改为 ArrayAttr | 中间 IR 属性，不跨版本持久化 |
| `#afir.kernel_arg<role = tiling>` | `index` 变为可选，语义为 stage id | 旧 IR（不带 index）仍合法 |
| `AggregatePlanInfo` | 删除 `fieldName` 成员 | 仓内结构体 |

### 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|--------|----------|----------|------|
| 接口说明 | 是否需要接口评审 | 否 | 均为仓内接口，无对外 API/ABI |
| 接口说明 | 是否需要补充文档 | 是 | 本文档 + `schedule_group_names.h` 头注释 + `AFIRAttrs.td` 中 `kernel_arg` 说明 |
| 接口兼容 | 行为是否兼容 | 是 | 单 stage 产物字节布局不变；生成的成员名变化，仅影响同批生成的 host/device 源码 |
| 接口兼容 | ABI/API 是否兼容 | 否 | 不涉及对外动态库导出符号 |
| 接口约束 | 约束不满足时是否清晰报错 | 是 | 缺 `afplan.candidate`、tiling 形参形状不符、group 成员名不符，均带定位信息报错 |
| 接口测试 | 是否需要独立接口用例 | 是 | `AttSchemaAdapterTest` 新增 4 个 group 成员交叉校验用例；`afir_kernel_arg_invalid.mlir` 新增 3 个负例 |

## 软件设计

### 关键数据结构

- `AggregatePlanInfo`：由 `{candidate, tilingType, fieldName, ...}` 变为不含 `fieldName`；stage 名字从 `tilingType.getNamesAttr()` 取。
- plan 级 `TilingSchema`：语义由「一个会被发射的 C++ 结构体」退化为**载体**——只按 stage 次序装 `names`/`types`，
  其自身的类型名不再进入生成代码。该点在 `GenTilingData` 的 pass body 中以注释显式说明。
- `PlanDeclaration`：`tilingField`（单个 StringAttr）改为 `tilingFields`（`SmallVector<StringAttr>`，按 stage 次序）。

### 关键技术/算法

平铺方案的取舍：

| 方案 | 结论 |
|------|------|
| 匿名 union 打洞，让 plan 层与 group 层共享存储 | 否决。类型双关是未定义行为，触编码红线 6 |
| 让 plan 结构体继承 group 结构体，靠切片访问 | 否决。多继承后不是 standard-layout，而这是一个 host↔device 传递的 POD |
| **删掉 plan 一层，group 直接挂 root** | 采纳。与 ATT/legacy 的形状一致，单 stage 场景可保持字节兼容 |

flat 判据必须与 `tiling_field_schema.cpp` 的 `is_flat` 严格一致，否则会出现「ATT 写 root 的标量、device 读 group 里的标量」
这种静默错位。这是本方案里唯一需要两处同步的判断，已在两处互相引用注释。

### 流程设计

见「整体介绍」的数据流图。关键次序约束：

1. root 字段次序：标量 → `graph<g>_tiling_key` → `workspace<id>` → group 成员。
   workspace 必须在 group 成员之前，与 legacy 一致。
2. kernel ABI 次序：`KernelArgumentKind { Input, Output, Workspace, WorkspaceOffset, Tiling }`，
   tiling 一族内部按 stage 升序且下标连续。

### 对子模块的修改

| 子模块 | 修改 |
|--------|------|
| `common/tiling_schema` | 新增 `schedule_group_names.h`；`NestedStagePath` 降为一级并订正注释 |
| `conversion/AFIRToAscendC` | `GenTilingData` 按公式命名并校验 candidate；`GenKernelEntry` 按 stage 展开 tiling 形参；`AscendCAttributes.h` 改 key 名 |
| `artifact/KernelSource` | `GenAggregateTilingData` 平铺 root；`GenAggregateKernelEntry` 按 stage 走哨兵槽位 |
| `artifact/TilingSource` | `TilingProgramSpec` 校验并使用一级 path；`TilingFuncEmitter` 加 `optiling::` 限定 |
| `dialect/AFIR` | `KernelArgAttr::verify` 三分支；`AFIRAttrs.td` 文档订正 |
| `bridge/AttSchemaAdapter` | 新增 group 成员引用提取与校验 |

### 错误处理

| 场景 | 处理 |
|------|------|
| module 上缺 `afplan.candidate` | `GenTilingData: missing afplan.candidate` + `signalPassFailure()` |
| 载体 schema 的 `types` 下标越界 | `return failure()` |
| tiling 形参个数 ≠ stage 数 / 下标不连续 / `index` 与 stage 不符 | `findTilingArguments` 报错 |
| group 成员名与共享公式不一致 | `BuildStageSpecs` / `createPlanDeclaration` `emitError` |
| 生成代码引用了 schema 中不存在的 group 成员 | `ValidateGeneratedFieldReferences` 返回失败并填 `diagnostic` |
| `tiling` 的 `index < -1` | 属性构造期 `verify` 失败 |

## 安全检查

已读 `docs/guidelines/编码红线.md`、`docs/guidelines/cross_feature_check.md`、`CONTRIBUTING.md`、`.clang-format`、`.pre-commit-config.yaml`。

| 检查项 | 结论 |
|--------|------|
| 敏感信息 / 公网地址 / 芯片型号 / 框架类型硬编码 | 无新增。命名公式中的字面量是 ATT 的标识符前缀，非配置项 |
| 外部输入作为索引 / 长度 / 偏移 / 申请大小 | 无。stage / plan / subgraph 序号来自 IR 属性与循环下标，且经 `verify` 与 `findTilingArguments` 双重校验 |
| 整数溢出 / 反转 / 除 0 | 不涉及新算术。`CalculateWorkspaceSize` 中原有的 `kMaxWorkspaceSize` 上限与进位前溢出检查保持不变 |
| 资源生命周期与异常分支释放 | 不涉及。全部为 MLIR 属性构造与字符串拼接 |
| ABI/API 兼容 | 见「接口检查项」。不涉及对外导出符号 |
| 图改写等价性与确定性 | root 字段次序完全由 plan 次序 + stage 次序决定，无容器遍历序依赖；`ExtractReferencedGroupMemberNames` 返回 `std::set`，有序 |
| runtime/aclrt/AscendC 生命周期约束 | 不涉及 |
| 类型双关 / UB | 显式否决了 union 打洞与基类切片两个方案 |
| 红线 7（不扩大修改范围） | 真机上发现的 `VecReduceND.cpp` 窄化告警为既有问题、非本变更引入，**未改动源码**，仅记录 |

## 兼容性检查

- 脚本参数、配置文件、安装目录、动态库、Python 包：无变化。
- 旧产物：生成的 `autofuse_tiling_data.h` 成员名变化。由于 host 与 device 源码在同一次编译中成对生成，
  不存在新 host + 旧 device 的混搭场景；kernel cache 以源码哈希为键，命名变化会自然 miss 并重编。
- 用户代码：不涉及。

# 测试设计

## 测试边界

- 入口：`af-opt` 的各 pass（lit）、`GenAggregateTilingData` / `GenAggregateKernelEntry`（UT）、
  `detail::ValidateGeneratedFieldReferences`（UT，手写源码片段打桩，绕开 `att::GenTilingImplAutoFuseV3`）。
- 出口：生成的 `emitasc` IR、最外层 TilingSchema 字段名列表、host tiling 源码文本。
- 打桩：`AttSchemaAdapterTest` 用手写源码片段代替真实 ATT 调用；`IsolatedPlanEntryTest` 用
  `makeTwoStagePlan` / `stageTilingType` / `markPlanMainTilingArguments` 构造两 stage 场景。
- 验证数据：真机 910B 上的 `load_sum_store.py` 与 `load_sum_store_large.py`，plan 0 / plan 1 各一轮。

## 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|----------|------------|----------|----------|
| 功能 | 两 stage plan 平铺后 root 的 8 个字段名与次序 | `IsolatedPlanEntryTest.FlattensTwoStagePlanIntoAggregateRoot` 逐个断言 | UT |
| 功能 | 聚合 main 恰好读取两个 group 成员 | 同上，遍历 `emitasc::MemberOp` 的 `getField()` | UT |
| 功能 | plan entry 上的 `tiling_fields` ArrayAttr 内容 | 同上 | UT |
| 功能 | ATT 命名（subgraph 0 / plan 3）出现在生成的 schema 里 | `afir-to-ascendc-gen-tiling-data.mlir` | lit |
| 功能 | `_main` 按 stage 收 tiling 形参，且 root 不再有 plan 一层 | `afir-to-ascendc-gen-kernel-entry.mlir`（含 `CHECK-NOT`） | lit |
| 功能 | stage 的 `tiling_field_name` 为一级 | `ScheduleToAfirTest` | UT |
| 功能 | `CalculateWorkspaceSize` 中六个模板加 `optiling::`、宏不加 | `ScheduleToAfirTest`（`optiling::Max(` / `optiling::Min(`） | UT |
| 异常 | module 缺 `afplan.candidate` | `afir-to-ascendc-gen-tiling-data-invalid.mlir` 的 `missing_candidate` | lit |
| 异常 | `tiling` 的 `index = -2` | `afir_kernel_arg_invalid.mlir` | lit |
| 异常 | 两个不带号的 tiling / 两个同号 stage 的 tiling | 同上 | lit |
| 异常 | `workspace` 带 index | 同上 | lit |
| 异常 | 生成代码引用了 schema 中不存在的 group 成员 | `AttSchemaAdapterTest` 新增 4 例 | UT |
| 兼容性 | 单 stage 场景 root 布局与 forwarding accessor 不变 | `afir-aggregate.mlir`、`afir-aggregate-launch.mlir`（`CHECK-NOT: struct plan0TilingData`） | lit |
| 兼容性 | workspace 字段位置不变 | `afir-to-ascendc-workspace-offset*.mlir` | lit |
| 特性交叉 | plan 0（RCommon）真机 e2e 不回退 | 910B 实机，`AUTOFUSE_DFX_FLAGS="--force_schedule_result=0"` | E2E |
| 特性交叉 | plan 1（RCore 两 stage）真机 e2e 可用 | 910B 实机，`AUTOFUSE_DFX_FLAGS="--force_schedule_result=1"` | E2E |
| 性能 | device 侧无退化 | 单 stage 字节布局不变，免测；多 stage 无基线 | 手工确认 |

## 测试命令

本机（容器内，`-j 8`）：

```bash
AF_MLIR_DEV_IMAGE=autofuse-mlir-dev:arm64 AF_CANN_VOLUME=af-cann-910 \
AF_CANN_INSTALL_PREFIX=/usr/local/Ascend/cann-9.2.0 \
./autofuse/mlir/scripts/enter_dev_container.sh -- bash -lc \
  'cmake --build /workspace/build --target check-ut check-e2e-cpu -j 8'

AF_MLIR_DEV_IMAGE=autofuse-mlir-dev:arm64 AF_CANN_VOLUME=af-cann-910 \
AF_CANN_INSTALL_PREFIX=/usr/local/Ascend/cann-9.2.0 \
./autofuse/mlir/scripts/enter_dev_container.sh -- bash -lc \
  '/opt/llvm/bin/llvm-lit -sv /workspace/build/autofuse/mlir/test/lit'
```

预期：`check-ut` / `check-e2e-cpu` 返回 0 且 0 failures；lit 132/132 通过。

真机（910B，`-j 40`）：

```bash
AUTOFUSE_DFX_FLAGS="--force_schedule_result=0" <runner> load_sum_store.py
AUTOFUSE_DFX_FLAGS="--force_schedule_result=1" <runner> load_sum_store.py
AUTOFUSE_DFX_FLAGS="--force_schedule_result=0" <runner> load_sum_store_large.py
AUTOFUSE_DFX_FLAGS="--force_schedule_result=1" <runner> load_sum_store_large.py
```

预期：前三条全部 `LOAD_SUM_STORE_NPU_E2E_PASS`。第四条见下节的已知问题。

## 实测结果

本机：`check-ut` + `check-e2e-cpu` rc=0 / 0 failures；lit 132/132。

真机 910B（`cann-9.2.0`，`aiv_num=40`，`ub_size=196608`）：

| 用例 | plan | 结果 |
|------|------|------|
| `load_sum_store.py`（小 shape ×3） | 0 | 全部 PASS |
| `load_sum_store.py`（小 shape ×3） | 1 | 全部 PASS（此前在 host 编译期即失败） |
| `load_sum_store_large.py` | 0 | 三个大 shape 全部 PASS |
| `load_sum_store_large.py` | 1 | `(512,1024)` PASS；`(1024,4096)` 失败 |

多 stage 场景生成的 root 布局已与 ATT 期望逐字段一致：

```c
struct AutofuseTilingData {
    uint32_t block_dim, corenum, ub_size, hbm_size;
    uint32_t graph0_tiling_key;
    uint32_t workspace2;
    AscGraph0ScheduleResult0G0TilingData graph0_result0_g0_tiling_data;
    AscGraph0ScheduleResult0G1TilingData graph0_result0_g1_tiling_data;
};
```

## 已知问题（不在本变更范围）

**plan 1 在 `(1024, 4096)` 上 tiling 求解失败。** 表象：

```
autofused_sum_... kernel tiling failed
RuntimeError: NPU kernel ... init failed   # torch_npu/_inductor/ascendc/compiler/_kernel.py:99
```

已在真机上用独立 harness 把生成的 host tiling 单独编译并逐级调用，定位结论：

```
AutofuseTiling rc=-1
G0::GetTiling -> 0  key=0 a0t=0 a1t=0 a1Tt=0 ...
=> stage 0 (RCore first stage) has no tiling solution
```

即 **ATT 生成的 `AscGraph0ScheduleResult0G0::GetTiling` 在该 shape 上解不出 tiling**，
`CalculateWorkspaceSize` 根本没有被执行到（`AutofuseTiling` 在 `optiling::GetTiling` 处就返回 -1）。
与本次命名/布局变更无关：plan 0 在同 shape 通过，plan 1 在 `(512,1024)` 通过。

> **归因更正（2026-09-28）。** 上面最后一句原先写的是「失败发生在完全由 ATT 生成的
> 求解器内部」，读起来像是求解器有缺陷。后续定位推翻了这个归因：**求解器没有错，是
> 这条路上的切分方案在大 A 上本来就无解。**
>
> `Scheduler::ReduceBlockTiling`（`autofuse/optimize/autoschedule/schedule.cpp:219-254`）
> 只对 R 外轴做 `TileTiling`，A 外轴走 `MergeAxis` 而**不做** `BlockSplit`。于是
>
> ```
> block_dim = Ceiling(A / a0t) × Ceiling(Ceiling(R / a1t) / a1Tt)
> ```
>
> 里只有第二个因子可调，而 `InitMulticoreVars` 把 `a1Tt` 直接顶到上界 `nR`，让它恒为 1。
> 所以 RCore phase 1 有解的充要条件就是 `Ceiling(A / a0t) ≤ corenum`。`(1024, 4096)`
> 的 A=1024 远超 40 核，无论怎么搜都超核——可行域其实有五万多个点，两阶段贪心不是搜漏了，
> 是这些点全都不满足核数约束。
>
> 这也不是缺陷：大 A 本来就该走 RCommon，RCore 是为**大 R 小 A** 那一档准备的。真正的
> 缺陷在选择侧——`SelectWinnersFromModels`（`bridge/PlanSelector/AttCostEvaluator.cpp:145`）
> 在编译期按 cost 选**一个**赢家、选不出就整条失败，而 legacy 的
> `GenGetAllSchedulesResults` 是全部试一遍、解不出的静默跳过。同一张图上 legacy 会换一个
> 模板，MLIR 会让整个 kernel 挂掉。要跟踪的是这一条，不是求解器。

# 验收标准

1. 本机 `check-ut`、`check-e2e-cpu` 返回 0 且无失败用例；lit 全量通过。
2. 单 stage 场景生成的 `AutofuseTilingData` 字节布局与变更前一致，forwarding accessor 保留。
3. 多 stage 场景生成的 root 字段名与次序与 ATT `ScheduleGroupIdent` 的公式逐字段一致。
4. 真机 plan 0 全量 e2e 不回退；plan 1 小 shape 与 `(512,1024)` 通过。
5. 三道新增关卡对命名/ABI 不一致给出带定位信息的报错，不再静默漏到 host 编译期。
6. 所有改动的 C++ 文件通过 `clang-format -i`。
7. 本文档随代码合入。
