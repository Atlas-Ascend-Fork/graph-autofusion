# MLIR 路径 workspace 偏移链路设计

> 状态：设计中（2026-09-23）
> 关联：[2026-09-21-afir-reduce-sum-ra-lowering-design.md](2026-09-21-afir-reduce-sum-ra-lowering-design.md) §3.8 / §3.10、
> [2026-08-13-tiling-data-schema-and-header-generation-design.md](2026-08-13-tiling-data-schema-and-header-generation-design.md)

# 简介

## 目的

RCore（reduce 多核两阶段）模板要求 phase 2 读取 phase 1 写在 GM workspace 里的中间结果。
当前 MLIR 下降路径在 `ScheduleToAfir` 就把 `Workspace` 节点顶了回去，整条模板链路走不通。

本文档确定 MLIR 路径上 workspace 张量从「调度器节点」到「kernel 里一个带偏移的 `GlobalTensor`」
的完整表示与下降方案。目标读者是 autofuse MLIR 后端的开发者。期望输出是可实施的改动清单、
验收口径和测试设计。

## 范围

**包含**：

- `Workspace` 调度节点在 AFIR 中的表示（bridge：`ScheduleToAfir` / `AfirBuilder`）
- AFIRToAscendC 侧把该表示物化成 `GlobalTensor` + 带字节偏移的 `SetGlobalBuffer`
- 上述表示与既有 tiling 侧 workspace 协议（ATT 生成的 Reset/Update/Finalize）的对接，
  以及「偏移确实会被赋值」的校验

**不包含**：

- workspace 的分配策略本身（复用、对齐、生命周期）——沿用 ATT/调度器既有口径，本次不改
- RCore 模板的 reduce 语义下降（归 RA 设计文档）
- legacy `autofuse/codegen` 路径——本次不动它，仅作为行为基准引用

## 涉及组件

Autofuse Codegen/Backend（`autofuse/mlir/bridge`、`autofuse/mlir/conversion`、`autofuse/mlir/common`）。
不涉及 SuperKernel、Python 绑定、打包。

# 总体概述

## 软件概述

### 项目介绍

workspace 是一块由 host 侧按 tiling 结果申请、通过 kernel 最后一个 `GM_ADDR` 参数传进来的
GM 空间。多块 workspace 张量共用这一个参数，各自的起始位置由 TilingData 里的
`workspace<ID>` 字段给出的**字节偏移**确定。

### legacy 行为基准（本次调查结论）

legacy 路径由三段协议构成，全部由 **ATT 生成的 tiling func** 完成：

| 段 | 生成位置 | 生成内容 |
|----|----------|----------|
| Reset | `tiling_code_gen_impl.cpp:3532` `ResetWorkspaceSizes` | 每个 id `set_workspace<ID>(0)` |
| Update | `tiling_code_gen_impl.cpp:3564` `GenUpdateWorkspace` | `set_workspace<ID>(max(get_workspace<ID>(), workspace_map[ID]))` |
| Finalize | `tiling_code_gen_impl.cpp:3538` `FinalizeWorkspaceOffsets` | 按固定 id 序做前缀和，把**尺寸**改写成**偏移** |

也就是说 `workspace<ID>` 字段是双语义的：tiling 过程中存的是尺寸，`FinalizeWorkspaceOffsets`
之后存的是偏移。ATT 源码里的注释写明了这一点（`tiling_code_gen_impl.cpp:3530`）。

`workspace_map` 的来源：`AscendGraphParser::ParseWorkspaceNode`（`ascend_graph_parser.cpp:612`）
按 `tensor_id` 对每个 `Workspace` 节点取 `CalculateOneWorkspaceSize`，同 id 取 `Max`，存进
`tuning_space_->workspace_size_map`，再经 `model_info.workspace_size_map` 由
`GenWorkspaceRelatedVars`（`base_types_printer.cpp:104`）生成 `SetWorkspaceSize` 的函数体。

kernel 侧读取在 `codegen_kernel.cpp:2087` `AppendWorkspaceTensorInit`：

```cpp
tensor->second.SetGlobalBuffer(workspace_buffer_arg, "workspace" + to_string(tId), local_result);
```

最终落成指针算术（`codegen_kernel.cpp:521`）：

```cpp
name.SetGlobalBuffer((__gm__ T*)((__gm__ uint8_t*)(workspace) + (workspace2)));
```

总量（传给 host 的 `workspaceSize`）由 `TilingLib::GenGetWorkspaceSizeFunc`
（`codegen_tiling.cpp:1017`）生成：按 impl graph 逐个 `ws_size += <本图 workspace 节点尺寸和>`，
由该图的 `tiling_key` 守卫，最后 512 对齐。

### MLIR 现状与前期记录的更正

调查后有两条此前记录需要更正。

**更正 1：`workspace<ID>` 的赋值链路在 MLIR 路径里已经在跑，不是缺失。**

RA 设计文档 §3.8 第 4 条写「`workspace<ID>` 字段没人赋值」。实际情况是：MLIR 把 ATT 产出的
tiling 源码**原样拼接**（`TilingFuncEmitter.cpp:378` `AssembleSources(bundle.tiling_functions)`），
而 `AutofuseTiling` 调用的正是 `optiling::GetTiling`（`TilingFuncEmitter.cpp:368`）——
Reset/Update/Finalize 都在它里面。字段名两侧一致（都是 `workspace<ID>`），作用域也一致
（`AutofuseTilingData` 根层：MLIR 侧见 `GenAggregateTilingData.cpp:116` / `GenTilingData.cpp:50`，
schema 侧见 `tiling_field_schema.cpp:211` 的 `"fusion"` 作用域）。

当前观察到「字段没被赋值」只是**语料结论**：RCommon 语料没有 `Workspace` 节点，
`workspace_size_map` 为空，ATT 生成的 `SetWorkspaceSize` 就是
`(void)tiling_data; (void)workspace_map;`（`tiling_code_gen_impl.cpp:2264`），
`GenWorkspaceOffsetHelpers` 也因 `workspace_tensor_id_set_` 为空而整段跳过。

**更正 2：「跨 stage 求和而不取并集」不是 MLIR 引入的缺陷。**

§3.8 第 3 条把 `AppendStageWorkspace` 的逐 stage 累加记成风险。核对后，
`TilingFuncEmitter::EmitWorkspaceHelper`（`TilingFuncEmitter.cpp:278`）是 legacy
`GenGetWorkspaceSizeFunc` 的忠实移植：同样逐 impl graph 累加、同样由 `tiling_key` 守卫、
同样 512 对齐；连尺寸表达式的来源都相同（两边都是
`ascgen_utils::CalculateWorkspaceSize(workspaceNodes)`，MLIR 侧见 `ScheduleToAfir.cpp:712`）。

两条路径在「多个 impl graph 共用同一 workspace tensor」时都会把总量算重——总量偏大，
偏移正确，属于**共有的保守超配**，不是越界风险。本次不改，只在文档里钉住这个口径。

**真正缺失的只有一段：kernel 侧的读取。** legacy 有 `AppendWorkspaceTensorInit`，MLIR 侧
没有任何对应物；而且缺口比这更靠前——`Workspace` 节点在 `ScheduleToAfir` 里根本没有映射成
AFIR 值，编译在 bridge 阶段就停了。

### 为什么不能把 Workspace 直接映射成 `Data`

`Data` 在 bridge 里是**入参**的同义词，不只是「一块 GM」：

- `NodeOutputRole`（`ScheduleToAfir.cpp:792`）把 `Data` 判为 `Role::Input`；
- `AppendNodeIo`（`:834`）据此把它收进 `info.inputs`；
- `AfirBuilder` 再按这份列表 `setKernelArgAttr(func, idx, KernelArgRole::Input, input.index)`（`:1242`），
  它决定 kernel 的形参表和 launch 侧实参的下标。

workspace 张量不是 host 传进来的实参，它的地址是「已有的 workspace 形参 + tiling 偏移」。
按 `Data` 映射会凭空多出一个入参、打乱既有入参编号，并让 launch 侧去找一个不存在的 tensor。

正确的类比不是「入参」，而是「**已有 GM 形参上的一个带偏移的视图**」。

### 产品环境介绍

不新增对 CANN Toolkit、runtime 的依赖。生成代码使用的 `SetGlobalBuffer` 与 legacy 完全相同，
AscendC API 面无变化。对外接口（`AutofuseTiling` 签名、kernel ABI 形参表）不变。

## 软件功能

1. `Workspace` 调度节点在 AFIR 中获得表示，bridge 不再报 `UnsupportedOperation`。
2. kernel body 每块 workspace 多一个 `ui32` 偏移形参，entry 从根 tiling 的 `workspace<ID>` 填。
3. 该表示在 AFIRToAscendC 下降为 `GlobalTensor` + 「按 `uint8_t*` 加偏移再转回元素指针」。
4. 绑定偏移前确认该偏移真的会被 ATT 赋值，确认不了即报错，不静默退化为 0。

## 设计约束

- **kernel body ABI 要扩，且只能按 legacy 那个形状扩。** 偏移**不在** kernel body 拿得到的
  TilingData 里：body 的 tiling 形参是 stage 级的 `Stage<N>TilingData`
  （`AFIRKernelBodySignature.cpp:239` 传入的 `tilingDataType` 来自 `getStageTilingType`），
  而 `workspace<ID>` 在根层 `AutofuseTilingData` 上。legacy 的做法是**每块 workspace 一个
  独立的 `uint32_t` 形参**：签名 `(..., GM_ADDR workspace, uint32_t workspace2, ..., const T *t)`
  （`codegen_kernel.cpp:1935-1938`），由 entry 从根 tiling 读出后传入
  （`codegen_kernel.cpp:2631-2634`）。本设计照此形状办，不另发明第二种。
- **不改 tiling 侧协议**：Reset/Update/Finalize 由 ATT 生成，MLIR 侧只消费 `workspace<ID>`，
  不自行赋值，避免出现第二个写点。
- **`workspace<ID>` 的读取时机**：只能在 `FinalizeWorkspaceOffsets` 之后读，即 kernel 执行期。
  kernel 侧读到的一律是偏移语义。
- 偏移字段类型为 `uint32_t`（`GenAggregateTilingData.cpp:132`），与 ATT 的
  `static_cast<uint32_t>` 口径一致。
- 遵循 `编码红线.md` 第 2/3 条：偏移进入指针算术前须校验来源，且不得产生整数溢出。

## 假设和依赖关系

- 假设 RCore 的 phase 1 / phase 2 impl graph 中各含一个 `Workspace` 节点，且二者
  `outputs[0].attr.mem.tensor_id` 相同。**待确认**——需在 S1 用真实 RCore plan 的图 dump 钉住。
- 假设一个 impl graph 内同一 `tensor_id` 的 `Workspace` 节点至多一个。
  `ToWorkspaceDescriptor` + `AppendWorkspaceDescriptor`（`ScheduleToAfir.cpp:343/370`）
  已按 `map<int64_t, BufferDescDTO>` 去重，本设计沿用。
- 依赖 ATT `workspace_tensor_id_set_` 与 `fused_result.workspace_nodes` 同源。**待确认**——
  它经 `ext_params` 传入（`tiling_code_gen_impl.cpp:5149`），调用侧是否与 MLIR 同一份
  `fused_result` 需在 S1 核对。校验项（FR5）正是为这条假设兜底。

# 需求分析与设计

## 整体介绍

数据流（自上而下）：

```
调度器 ImplGraph
  ├─ Workspace 节点 ──┬─► [bridge] BufferDescDTO{space=GM, bufferId=tensor_id, sizeBytesExpr}
  │                   │        （已有，plan 级 workspace_plan，ToWorkspaceDescriptor:343）
  │                   │
  │                   └─► [bridge, 新增] 模板体内一个 GM 值
  │                            memref.alloc() {afir.buffer_desc = #afir.buffer_desc<space=gm,
  │                                                                buffer_id=<tensor_id>, ...>}
  │
  └─ fused_result.workspace_nodes ─► common::BuildTilingFieldSchema ─► schema 里的 workspace<ID>
                                   └─► ATT workspace_size_map ─► Reset/Update/Finalize（已有）

AFIRToAscendC
  afir-kernel-body-signature
      └─ [新增] 每块被绑定的 workspace 追加一个 ui32 形参，
                role = KernelArgRole::WorkspaceOffset，argIndex = buffer_id
  CanonicalFunctionMemoryAllocator::materializeGlobalValue（已有，space==GM 即触发）
      └─ [新增分支] buffer_id ∈ workspace_tensor_ids 时：
             %p = emitasc.ptr_offset %wsArg[%off]        // __gm__ uint8_t* + 偏移
             %t = emitasc.reinterpret_cast %p            // 转回元素指针
             SetGlobalBuffer(global, %t)
         其余 GM 值维持现状：SetGlobalBuffer(global, info.value)

GenKernelEntry
  └─ [新增] KernelArgumentKind::WorkspaceOffset：实参为根 tiling 的 workspace<ID> 成员
```

控制流上，本设计只在两处插入：bridge 的节点分类、materialization 的 `SetGlobalBuffer` 构造。
其余环节（tiling 生成、字段声明、kernel 形参）全部复用既有链路。

## 功能需求

### 功能需求 1：`Workspace` 节点映射为模板体内的 GM 存储

1. **介绍**

   `ScheduleToAfir` 对 `type == "Workspace"` 的节点产出一个带 `#afir.buffer_desc` 的 GM 值，
   而不是报 `UnsupportedOperation`，也不把它记入 `info.inputs` / `info.outputs`。

2. **输入**

   `ImplGraph` 中的 `Workspace` 节点。合法形态：恰好一个输出张量，
   `outputs[0].attr.mem.position == kPositionGM`，`tensor_id >= 0`，dtype 可映射。
   这三条 `ToWorkspaceDescriptor`（`ScheduleToAfir.cpp:343`）已在校验，本需求复用该校验。

   非法输入：多输出、非 GM position、`tensor_id < 0`、dtype 不可映射 —— 沿用既有
   `InvalidWorkspace:` 诊断。

3. **处理**

   - `NodeOutputRole`（`:792`）：`Workspace` 保持返回 `Role::Compute`，**不改**。它决定的是
     张量描述子的 role，而 workspace 在写入模板里是被 Store 落盘的目标、在读取模板里是被 Load
     取用的源，二者都不是 kernel 入出参的 role。
   - `AppendNodeIo`（`:834`）：维持现状（只收 `Data` / `Output`），**不改**。这是本设计与
     「映射成 Data」方案的分水岭。
   - `AppendTemplateNode`（`:873`）的 buffer-form 豁免名单增加 `"Workspace"`，并在该分支内
     为其产出一条 `NodeInfo`，携带从 `ToWorkspaceDescriptor` 得到的
     `{space=GM, bufferId=tensor_id, sizeBytesExpr}`。
   - `AfirBuilder` 在模板体内为它建 `memref.alloc()` 并挂 `afir.buffer_desc`，空间为 GM。

   异常：若模板内出现 `Workspace` 节点但该 plan 的 `workspace_plan` 里没有同 `bufferId` 的
   条目，报 `InvalidWorkspace: workspace node <id> is not declared in workspace_plan`。

4. **输出**

   模板 func 体内多出一个 GM `memref`，被 Store 的 destination 或 Load 的 source 引用。
   `afir.workspace_plan` / `afir.workspace_tensor_ids` 不变（它们已由既有链路产出）。

### 功能需求 2：kernel body 追加 workspace 偏移形参

1. **介绍**

   `afir-kernel-body-signature` 为每块被本 body 绑定的 workspace 张量追加一个 `ui32` 形参，
   携带该块的字节偏移。这是 legacy 形状的移植，不是新发明——偏移在 body 拿得到的
   `Stage<N>TilingData` 里不存在，必须从外面传进来。

2. **输入**

   函数体内所有带 GM `#afir.buffer_desc` 且 `buffer_id` 命中 module 上
   `afir.workspace_tensor_ids` 的值，按 `buffer_id` 去重后升序。

3. **处理**

   - 形参插在既有的 workspace 形参**之后**、tiling 形参**之前**，与 legacy 的
     `(..., GM_ADDR workspace, uint32_t workspace<ID>..., const T *t)` 一致。
   - 每个形参挂 `KernelArgRole::WorkspaceOffset`，`argIndex` 取 `buffer_id`。
     这要求放宽两条既有约束（见「接口设计」）：`KernelArgAttr::verify` 今天只允许
     input / output 携带 `argIndex`；`verifyKernelArgumentRoles` 今天要求 workspace 至多一个
     ——新角色与 `Workspace` 是两个角色，后者仍然至多一个，但新角色可以有多个，
     以 `argIndex` 在角色内不重复为准。
   - 没有 workspace 绑定时一个形参都不加，既有 kernel 的签名逐字节不变。

4. **输出**

   ```mlir
   func.func @..._stage0_template0(%in: memref<...>, %out: memref<...>,
                                   %ws: memref<ui8, #gm> {afir.kernel_arg = <workspace>},
                                   %ws2_off: ui32 {afir.kernel_arg = <workspace_offset, 2>},
                                   %tiling: !ascendc.tiling<...>)
   ```

### 功能需求 3：GM workspace 值下降为带偏移的 `SetGlobalBuffer`

1. **介绍**

   `CanonicalFunctionMemoryAllocator::materializeGlobalValue` 对 workspace 值生成
   「按 `uint8_t*` 加偏移、再转回元素指针」的 `SetGlobalBuffer`，其余 GM 值行为不变。

2. **输入**

   `CanonicalValueInfo`，其 `view.resource.space == afir::Space::GM`
   （`CanonicalMaterialization.cpp:540` 已按此筛选），且 `info.buffer.getBufferId()` 命中
   module 上的 `afir.workspace_tensor_ids`。

   边界：`bufferId < 0`（普通 GM 入出参）走原分支；`workspace_tensor_ids` 属性缺失时
   一律走原分支。

3. **处理**

   - 查 workspace 形参：`afir::findKernelArgument(func, KernelArgRole::Workspace)`
     （`AFIR.h:279`，找不到返回 `nullopt` 由调用点诊断）。取不到则报
     `InvalidBinding: workspace tensor requires a workspace kernel argument`。
   - 查偏移形参：角色为 `WorkspaceOffset` 且 `argIndex == buffer_id` 的那个。取不到即报错，
     **不退化为 0**——退化成 0 会让两块 workspace 重叠写，是静默的数据损坏。
   - 建三条 op：

     ```mlir
     %i = arith.index_cast %off : ui32 to index
     %p = emitasc.ptr_offset %ws[%i] : memref<ui8, #gm>, memref<ui8, #gm>
     %t = emitasc.reinterpret_cast %p : memref<ui8, #gm> to memref<...x f32, #gm>
     ascendc.global_tensor.set_global_buffer %g, %t
     ```

     `emitasc.ptr_offset`（`EmitAsc/IR/Ops.td:146`）印成 `base + off`，
     `emitasc.reinterpret_cast`（`:160`）印成 `reinterpret_cast<T*>(...)`，
     两者都已在 `Translation.cpp:131` 的可印列表里，不需要新增 op 或新增印法。

4. **输出**

   生成的 kernel 中出现与 legacy 等价的三行：

   ```cpp
   __gm__ uint8_t* vA = v4 + vOff;
   __gm__ float*   vB = reinterpret_cast<__gm__ float*>(vA);
   vC.SetGlobalBuffer(vB);
   ```

### 功能需求 4：entry 侧把 stage tiling 的 `workspace<ID>` 传给 body

> **实现期更正**：本节初稿写的是「取自**根** tiling」，依据是 legacy 的 entry
> （`codegen_kernel.cpp:2631`）和 ATT 的字段作用域（`tiling_field_schema.cpp:225` 把
> `workspace<ID>` 放在 fusion 作用域）。MLIR 侧的布局与之不同且是既定的：
> `GenTilingData.cpp` 的 `collectStageFields` 把 `block_dim` / `corenum` / `ub_size` /
> `hbm_size` / `workspace<ID>` / `tiling_key` 一起复制进**每个** `Stage<N>TilingData`，
> plan 级结构体里只有各 stage 的嵌套成员（这一形态由
> `test/lit/conversion/afir-to-ascendc-gen-tiling-data.mlir:68` 逐字钉住）。entry 里所有
> tiling 成员读取（`tiling_key`、size_var）本来就走 stage 结构体，偏移也照此办理。按根
> tiling 读会直接失败：plan 级结构体上没有 `workspace<ID>` 字段。

1. **介绍**

   `GenKernelEntry` 为 body 的 `WorkspaceOffset` 形参提供实参，取自该 stage 的
   `Stage<N>TilingData` 的 `workspace<ID>` 成员。

2. **输入**

   body 的形参身份表（FR2 写入的 `afir.kernel_arg`），以及 entry 为本 stage 取出的
   stage tiling 值。

3. **处理**

   - `KernelArgumentIdentity` 增加 `KernelArgumentKind::WorkspaceOffset`，以 `index = buffer_id`
     区分同类形参；`getKernelArgumentPermutation` 把它排在 `Workspace` 之后、`Tiling` 之前。
   - `collectTemplateAbi` 今天用 `mapping[i] == UINT_MAX` 这一个哨兵表示「传 stage tiling」。
     再加一个哨兵会把这张表变成两处要同步的约定，所以把 mapping 的元素从 `unsigned` 改成
     一个三态小结构（主参下标 / stage tiling / stage tiling 的某个 workspace 字段）。
   - `buildTemplateArguments` 对第三态发 `readMember(stageTiling, "workspace<ID>")`。
     `readMember` 已有（`GenKernelEntry.cpp:328`），读的就是 `TilingSchema` 成员。

4. **输出**

   entry 中对 body 的调用多出实参，与 legacy 的
   `impl_graph(... , workspace, t.workspace2, ... , &tiling_data)` 同形；差别只在
   `t` 是 stage 结构体而不是根结构体，理由见本节开头的更正。

### 功能需求 5：workspace 偏移可用性校验（按消费端驱动）

1. **介绍**

   FR3 为一块 GM 张量绑定 `workspace<ID>` 偏移时，必须先确认这个偏移**真的会被赋值**。
   否则偏移恒为 0，多块 workspace 重叠写——静默的数据损坏。新增一处定向校验。

2. **输入**

   - A：`common::BuildTilingFieldSchema` 从 `fused_result.workspace_nodes` 生成的
     `workspace<ID>` 字段名集合（`tiling_field_schema.cpp:211`）
   - B：ATT 生成源码中实际引用的 `workspace<ID>` 名字集合
   - C：**本次真正被绑定**的 workspace id 集合，即 FR2 / FR3 在各 template func 里命中的
     `buffer_desc.buffer_id`

3. **处理**

   **不能按 schema 驱动校验。** A 与 `afir.workspace_tensor_ids` 都取自 **fused 级**的
   `fusedResult.workspace_nodes`（`ScheduleToAfir.cpp:387` / `tiling_field_schema.cpp:199`），
   对一个不使用 workspace 的 plan（例如 RCommon）它们照样非空，而 ATT 那侧不会引用任何
   `workspace<ID>`。「schema 声明了就该有人写」会在今天每一条 reduce 语料上误报——
   这正是 RA 设计文档 3.8 第 2 条更正后留下的结论。

   校验按**消费端**驱动，方向只有一个：

   - **C ⊆ B**：凡是 FR3 真正绑定了偏移的 id，ATT 生成的源码里必须出现对应的
     `get_workspace<ID>` / `set_workspace<ID>`。不出现即说明 ATT 那侧的
     `workspace_tensor_id_set_` 不含该 id，`FinalizeWorkspaceOffsets` 不会为它算偏移，
     读到的恒为 0。报错文案须直指该结论，而不是泛泛的「字段缺失」。
   - C ⊆ A 由既有的 `detail::ValidateGeneratedFieldReferences`
     （`AttSchemaAdapter.cpp:114`，调用点 `:172`）间接覆盖：ATT 引用了 schema 没有的字段会
     先被它拦下。本需求不重复这一侧。

位置上有一处顺序问题：`RunAttForCandidates` 跑在 AFIR 下降之前，拿不到 C。
   因此校验落在 **entry 侧**——FR4 为 `WorkspaceOffset` 形参找实参时，查根 `TilingSchema` 上
   有无 `workspace<ID>` 成员即可（该成员只有在 ATT 与 schema 双方都认这个 id 时才存在），
   不需要把 ATT 源码文本带到下降阶段。这也是 FR4「根 tiling 缺 `workspace<ID>` 成员即报错」
   那条的设计理由。

4. **输出**

   诊断经 MLIR diagnostic 返回，该候选 plan 被丢弃，由 `PrepareMlirCandidates` 的既有
   fallback 兜底。不新增日志级别。

## 非功能需求

### 可维护性

- 三处改动分别落在既有的单一职责点上（节点分类 / 值物化 / 适配层校验），不新增模块。
- 不引入第二个 `workspace<ID>` 写点：MLIR 侧**只读不写**，写点唯一且在 ATT。
  这条是本设计的核心不变式，需在代码注释里写明，否则后来者很容易在 `TilingFuncEmitter`
  里再补一个赋值。
- `EmitWorkspaceHelper` 的逐 stage 累加口径与 legacy 一致，本次不动；在其注释处补一行
  说明「与 `GenGetWorkspaceSizeFunc` 同源，共用超配口径」，避免被当成 bug 反复翻出来。

### 可测试性

- bridge 侧：`ScheduleToAfirTest` 已有 workspace 用例（`:347` 读 `afir.workspace_plan`），
  在同一夹具里加「模板体内出现 GM alloc 且不进入 kernel 形参表」的断言。
- 下降侧：lit 用例，输入一段带 GM `buffer_desc` 的 AFIR，`CHECK` 生成的
  `SetGlobalBuffer` 带偏移实参。
- 校验侧：与下降侧同一个 lit 套件——构造一份缺 `workspace<ID>` 成员的 TilingData 形参，
  断言报错文案指向「该 id 的偏移不会被赋值」而非泛泛的字段缺失。
- E2E：CPU 侧 `MlirE2eCpu` 无法覆盖多核 workspace 语义（单核模拟），只能覆盖到「生成的
  kernel 能编译且偏移表达式存在」。数值正确性须留到 910B 上验证。

### 可移植性

不依赖芯片型号、平台或框架类型判断（红线 Graph-autofusion 第 2 条）。偏移字段名
`workspace<ID>` 由 ATT 与 schema 共同约定，不在 MLIR 侧硬编码芯片相关常量。

### 可靠性

- 所有「取不到」的分支一律报错返回，不静默退化。偏移退化为 0 会造成多块 workspace
  重叠写，属静默数据损坏，必须拒绝。
- 不新增动态内存或资源持有，无异常分支释放问题。
- 图改写确定性（红线 Graph-autofusion 第 5 条）：workspace id 的遍历顺序沿用既有的
  `llvm::sort` + `unique`（`GenAggregateTilingData.cpp:130`）与 ATT 的 `std::set` 序，
  两侧都是 id 升序，生成结果确定。

### 特性交叉影响

按 `cross_feature_check.md` 逐项：

| 场景 | 适用性 | 分析说明 |
|------|--------|----------|
| SuperKernel Python 接口 | 不适用 | 不涉及 `super_kernel/` 下任何文件、选项解析或 wheel 内容 |
| SuperKernel C++/AOT 接口 | 不适用 | 不改 `libascendsk.so` 导出符号或结构体 |
| Autofuse 图优化 | 不适用 | 不改 ASCIR 图结构与融合 pass；只读调度器已产出的 `ImplGraph` |
| Autofuse Codegen/Backend | **适用** | 本设计的全部内容。改动点：`bridge/ScheduleToAfir`、`bridge/AfirBuilder`、`conversion/AFIRToAscendC/MemoryAllocate`。legacy `autofuse/codegen` 与 `bridge/AttSchemaAdapter` 均不动 |
| AscendC API / Runtime 交互 | **适用** | 新增 `SetGlobalBuffer` 调用一处，形态与 legacy 完全相同。workspace 内存由 host 按 `AutofuseTiling` 回填的 `workspaceSize` 申请，生命周期覆盖整个 kernel，满足红线 Graph-autofusion 第 7 条 |
| Python/C++ 混合绑定 | 不适用 | 不涉及 `autofuse/compiler/py_module/` |
| 构建与打包 | 不适用 | 无新增源文件目录、第三方依赖或安装路径变化 |
| 测试与覆盖率 | **适用** | 新增 bridge UT、下降 lit、适配层 UT；见「测试设计」 |
| 性能与日志 | **适用（影响可忽略）** | 见「性能」。不新增默认开启日志 |
| 兼容性 | 不适用 | kernel ABI 形参表、`AutofuseTiling` 签名、TilingData 字段名与布局均不变 |

## 性能

### 编译时长

三处改动都是 O(workspace 节点数) 的常数级遍历，workspace 节点数是个位数量级。
FR5 的校验只查一次根 `TilingSchema` 成员，不新增源码扫描。
编译时长影响不可测量。

### 执行性能

生成代码新增的是 kernel 初始化段的一次指针加法，与 legacy 完全一致。无执行性能变化。

### 内存和产物大小

`workspace<ID>` 字段已在 TilingData 中存在（当前是死字段），本设计只是让它被写和被读，
TilingData 大小不变。生成的 kernel 源码每块 workspace 多一行，可忽略。

设备侧内存：总量口径不变（沿用现有的逐 stage 累加），因此不引入新的超配。

## 接口设计

### 新增/修改接口描述

无对外接口变更。模块内部新增/修改的函数签名：

| 位置 | 变更 | 说明 |
|------|------|------|
| `ScheduleToAfir.cpp` `AppendTemplateNode` | 豁免名单增加 `"Workspace"` | 内部静态函数，无外部调用者 |
| `AFIREnums.td` `KernelArgRole` | 新增 `WorkspaceOffset = 4` | 枚举尾部追加，既有取值不变；该枚举只在 IR 属性里出现，不跨 SO 边界 |
| `AFIR.h` `setKernelArgAttr` / `verifyKernelArgumentRoles` | 允许新角色携带 `argIndex`，且可有多个 | 签名不变，只放宽校验 |
| `AFIRKernelBodySignature.cpp` `appendKernelArguments` | 追加偏移形参 | 静态成员函数，签名不变 |
| `CanonicalMaterialization.cpp` `materializeGlobalValue` | 新增 workspace 分支 | 成员函数签名不变 |
| `GenKernelEntry.cpp` `collectTemplateAbi` / `buildTemplateArguments` | mapping 元素由 `unsigned` 改成三态结构 | 内部静态函数，无外部调用者。FR5 的校验并入 FR4 的实参解析，不改 `AttSchemaAdapter` |

### 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|--------|----------|----------|------|
| 接口说明 | 是否需要接口评审 | 否 | 无对外接口 |
| 接口说明 | 是否需要补充文档 | 是 | 本文档 + RA 设计文档 §3.8 更正 |
| 接口兼容 | 行为是否兼容 | 是 | 无 workspace 节点的图行为逐字节不变（新分支不触发） |
| 接口兼容 | ABI/API 是否兼容 | 是 | TilingData 布局不变；kernel body 形参表**仅在有 workspace 绑定时**增加 ui32 偏移形参，与 legacy 同形。body 与 entry 同批生成，不存在跨版本混用 |
| 接口约束 | 约束不满足时是否清晰报错 | 是 | 各类失败各有定向诊断，见 FR1/FR3/FR4/FR5 |
| 接口测试 | 是否需要独立接口用例 | 否 | 覆盖在既有 UT 套件内 |

## 软件设计

### 关键数据结构

- `BufferDescDTO{space, bufferId, sizeBytesExpr, lifeTimeAxisId}`（既有）——
  workspace 用 `space = GM`、`bufferId = tensor_id`、`lifeTimeAxisId = -1`。
- `#afir.buffer_desc`（既有）——挂在模板体内的 GM `memref` 上。
  `CanonicalFunctionMemoryAllocator::validateResource`（`CanonicalResources.cpp:115`）
  对 GM 只禁止队列字段，允许携带 `buffer_id`，无需放宽校验。
- `afir.workspace_tensor_ids`（既有，module 级 `DenseI64ArrayAttr`）——
  FR2 / FR3 的判定依据，FR5 的校验对象之一。

所有权：AFIR 属性由 MLIRContext 持有，无手工生命周期管理。

### 关键技术/算法

核心只有一条判定：**一个 GM 值是不是 workspace**。判据是
`buffer_desc.buffer_id ∈ afir.workspace_tensor_ids`。

选它而不选「是不是 kernel 形参」，理由是后者要靠反推（形参 vs 非形参），而 AFIR 的既有
约定是「身份靠标记，不靠位置反推」（`AFIREnums.td:208` 已把「靠类型相等反推 workspace 形参」
记为待偿技术债）。用 id 集合判定是显式标记，与该方向一致。

复杂度 O(1)（id 集合规模个位数，线性查找即可，无需引入哈希集合）。

### 流程设计

主流程见「整体介绍」的数据流图。

异常流程：

| 失败点 | 诊断前缀 | 结果 |
|--------|----------|------|
| Workspace 节点形态非法 | `InvalidWorkspace:` | bridge 失败，整个 plan 被丢弃 |
| 节点未在 `workspace_plan` 中声明 | `InvalidWorkspace:` | 同上 |
| 找不到 workspace 形参 | `InvalidBinding:` | 下降失败 |
| 找不到匹配 `buffer_id` 的偏移形参 | `InvalidBinding:` | 下降失败 |
| stage tiling 缺 `workspace<ID>` 成员（FR5） | `GenKernelEntry:` | entry 生成失败 |


无回滚/清理流程——所有失败都在纯函数式的图构造阶段，失败即整体丢弃该候选，
由 `PrepareMlirCandidates` 的既有 fallback 兜底。

### 对子模块的修改

| 目录/文件 | 修改点 |
|-----------|--------|
| `autofuse/mlir/bridge/ScheduleToAfir/ScheduleToAfir.cpp` | `AppendTemplateNode` 豁免名单 + Workspace 节点的 `NodeInfo` 产出 |
| `autofuse/mlir/bridge/AfirBuilder/AfirBuilder.cpp` | 为 Workspace `NodeInfo` 建 GM `memref.alloc` 并挂 `afir.buffer_desc` |
| `autofuse/mlir/dialect/AFIR/IR/AFIREnums.td`、`AFIRDialect.cpp` | 新增 `KernelArgRole::WorkspaceOffset`；放宽 `argIndex` 与「至多一个」两条校验 |
| `autofuse/mlir/conversion/AFIRToAscendC/AFIRKernelBodySignature.cpp` | 追加 per-workspace 的 ui32 偏移形参 |
| `autofuse/mlir/conversion/AFIRToAscendC/MemoryAllocate/CanonicalMaterialization.cpp` | `materializeGlobalValue` 新增 workspace 偏移分支（ptr_offset + reinterpret_cast） |
| `autofuse/mlir/conversion/AFIRToAscendC/AFIRMemoryAllocate.cpp` | `getDependentDialects` 补声明 `emitasc` / `emitc`——上一行新建的 op 出自这两个方言，整条流水线里它们被别的 pass 拉起来了，单独跑本 pass 时缺声明会 `report_fatal_error` |
| `autofuse/mlir/conversion/AFIRToAscendC/GenKernelEntry.cpp` | 新增 `WorkspaceOffset` 实参种类；mapping 元素由 `unsigned` 改成三态 |
| `autofuse/mlir/dialect/AFIR/Transforms/LoopGenerate.cpp` | 见下「实现期发现的两处 LoopGenerate 缺陷」 |
| `autofuse/mlir/artifact/TilingSource/TilingFuncEmitter.cpp` | 仅补注释（说明与 legacy 同源的超配口径），无逻辑改动 |
| `autofuse/mlir/docs/design/2026-09-21-...-ra-lowering-design.md` | §3.8 第 2/3/4 条按本文「更正 1/2」修订 |

### 顺带闭环的 RCore 前置第 3 条：`need_sync_all`

本设计把 workspace 的**地址**接通了，但多核 reduce 还差一道**时序**：phase 1 各核写完自己
那一行之后、phase 2 开始读整块之前，必须全核同步。`ascendc.need_sync_all` 此前只有消费者
（`GenKernelEntry` 紧跟 call 发 `ascendc.sync_all_hard`），没有任何地方产出它，于是真实
RCore kernel 里 `SyncAll` 为 0 —— 而这是**单核上必然看不出来**的一类错，CPU E2E 会侥幸通过。

生产者补在 `AFIRKernelBodySignature` 的 `markNeedSyncAll`，判据与 legacy 同源
（`codegen_kernel.cpp:2748`）：

```
need_sync_all = kernel.has_workspace_node && j != schedule_groups.size() - 1
```

两边一一对上：`has_workspace_node`（这张 impl 图里有 `Workspace` 节点）在 MLIR 侧就是
`collectBoundWorkspaceIds` 非空——与偏移形参共用同一条判据，不另立；`j` 是 stage 序号，取自
`#afplan.variant`，总数取自 module 的 `afplan.stages`。最后一个 stage 之后没有读方，不发。

绑了 workspace 却取不到 stage 坐标时**报错**，不静默跳过：跳过的后果正是上面那种「单核看不
出来」的数据竞争。存量语料一个都不受影响——它们绑定数为 0，第一道判据就返回了。

### 实现期发现的一处发射侧约束

`reinterpret_cast` 结果类型的**地址空间**必须用 AscendC 那一种
（`ascendc::AddressSpace`，打印成 `22 : i32`），不能沿用 `info.value` 上的
`#afir.space<gm>`。这条 memref 会被 `emitasc.reinterpret_cast` 当成变量声明发成 C++，而
`CodeEmitter::emitBaseMemRefType` 把 memory space 按 `IntegerAttr` 取——给它一个方言属性会
在**发射期**直接 assert 掉。

普通 GM 入出参绑的是 kernel 形参，类型在 `afir-kernel-body-signature` 里就已经换成 AscendC
那一种，所以只有 workspace 这条新路会手工造 memref、也只有它会撞上。

这条约束值得单独记一笔，因为 MLIR 侧一路全绿也查不出来：`af-opt` 只跑到 MLIR，C++ 发射是
另一段。lit 用例因此把结果类型逐字钉住，而不只是断言 `reinterpret_cast` 出现过。

### 实现期发现的两处 LoopGenerate 缺陷

本设计只谈 workspace 偏移链路，但把 RCore 一阶段图真正下降一遍以后，撞上了两处与本设计
无关、却把这条链路整个挡死的既有缺陷。两处都是 RCore 一阶段图的形状**首次**触发，存量
语料的循环树形状碰不到，因此此前一直潜伏。

**缺陷 1：`emitTree` 用了会失效的插入点锚点。**

`OpBuilder` 的插入点存的是一个**指向后继 Op 的迭代器**，本层的锚点一开始正是函数里第一条
AFIR Op。子循环把那条 Op 搬进自己的体内之后，锚点迭代器就指向了另一个块里的节点；再拿它
去 `moveBefore(本层块, 该迭代器)`，`splice` 按迭代器的物理位置把 Op 接进了子循环体，却按
目标链表把它的 parent 记成本层块。链表自此自相矛盾，随后任意一次 `isBeforeInBlock` 都会
踩到哨兵节点直接 abort（表现为 `af-opt` 在支配关系校验中 `rc=134`）。

只有「循环节点后面还跟着同层兄弟 Op」的形状才会暴露。RCore 一阶段图正是这个形状：
`reduce` 在 `loop_axis = 7`，写 workspace 的 `store` 在 `loop_axis = 8`（块轴），于是 store
是循环之后的同层兄弟。修法是把循环项的锚点改到 `emitLoop` **返回之后**才重取——那时子树
该搬的都搬完了，循环后面剩下的第一条 Op 才是稳定的后继；`emitLoop` 因此改成返回它新建的
`scf.for`（块轴不建循环，返回空）。

**缺陷 2：由多条源轴合并而来的 `BlockOuter` 轴没有拆出源轴序号。**

`generateSplitOuterAxisValue` 已经认得「`BlockOuter` 可以由多条源轴合并而来」这一形态，并
按源轴 `loop_size` 连乘算它的尺寸；但序号一侧的 `generateMergedAxisSourceIndices` 只认
`AxisType::Merged`，于是这几根源轴的运行期序号根本不存在。下游 `findOuterAxisIndex` 查不
到，搬运就无从发射（`failed to legalize operation 'afir.load'`，内层诊断
`cannot emit the scheduled transfer`）。

RCore 一阶段图把 `z1T` 与 `z0TT` 合并绑到块轴 `z1Tz0TTB` 上，正是这个形态。修法是把「要不
要按多条源轴拆开」收成一个判据 `isMergedLike`（`Merged`，或源轴多于一条的 `BlockOuter`），
尺寸预计算、序号拆分与递归三处共用它，并在 `emitLoop` 的块轴分支上调用一次拆分。

两处都补在 `LoopGenerate.cpp` 内，不外溢到别的模块；对存量语料的循环树形状无影响（判据都
只在 RCore 那种形状上为真），由 `check-ut` / `check-e2e-cpu` / lit 全量回归确认。

### 错误处理

#### 系统错误

本设计不申请内存、不读写文件、不调用 runtime。编译失败沿用既有路径。

#### 接口错误

全部通过 MLIR diagnostic 返回，最终体现为该候选 plan 被丢弃 + `GELOGW`。不新增对外错误码。
注意 `PrepareMlirCandidates` 的失败只进 `GELOGW`，排查时须开
`ASCEND_SLOG_PRINT_TO_STDOUT=1 ASCEND_GLOBAL_LOG_LEVEL=2` 才看得到。

## 安全检查

按 `编码红线.md` 逐条：

- **1 硬编码敏感信息**：无。不新增密钥、账号、公网地址。
- **2 外部输入作索引/长度/偏移**：**直接相关**。`workspace<ID>` 的值来自 tiling 计算，
  进入指针算术。防线在 tiling 侧已有：`FinalizeWorkspaceOffsets` 的前缀和以 `uint32_t`
  累加，`CalculateWorkspaceSize`（`TilingFuncEmitter.cpp:304`）在 `ws_size > UINT32_MAX - 511`
  时返回 false 使整次 tiling 失败。kernel 侧不重复校验（设备端无法报错），依赖 host 侧
  这道闸。此依赖关系须写进代码注释。
- **3 整数溢出/反转/除 0**：偏移与尺寸全程 `uint32_t`，累加点在 `FinalizeWorkspaceOffsets`
  内，其上界由第 2 条的闸保证。本设计不新增算术。
- **4 资源释放覆盖异常分支**：无新增资源。
- **5 内存申请**：不新增申请点。host 侧按 `*workspaceSize` 申请的既有逻辑不变。
- **6 未定义行为/求值顺序**：无新增表达式副作用。
- **7 禁止扩大修改范围**：本设计明确不改 `EmitWorkspaceHelper` 的累加口径、不改
  `NodeOutputRole` / `AppendNodeIo`、不动 legacy `autofuse/codegen`。RA 代码检视中记录的
  6 条发现不在本次范围内。
- **Graph-autofusion 1 ABI/API 兼容**：TilingData 布局不变。kernel body 形参表在有 workspace
  绑定时会变，但 body 与 entry 由同一次 codegen 一起产出、一起编译，不存在跨版本混用；
  对外的 `AutofuseTiling` 签名与 launch 侧实参表不变。`KernelArgRole` 新增的枚举值排在尾部，
  且该枚举只出现在 IR 属性里，不跨 SO 边界。见「接口检查项」。
- **Graph-autofusion 2 硬编码芯片/平台/框架类型**：无。
- **Graph-autofusion 3 图改写等价性**：本设计不改写图，只补一类节点的表示。
- **Graph-autofusion 5 图改写确定性**：workspace id 两侧均按升序，见「可靠性」。
- **Graph-autofusion 6 高频路径日志**：不新增日志。
- **Graph-autofusion 7 AscendC 接口生命周期**：`SetGlobalBuffer` 绑定的 GM 地址来自 kernel
  形参，生命周期覆盖整个 kernel 执行，满足约束。

## 兼容性检查

- 不影响任何脚本参数、配置文件、安装目录、动态库名或 Python 包。
- 不影响历史产物：无 `Workspace` 节点绑定的图（当前全部 RCommon 语料）生成结果逐字节不变——
  偏移形参只在本 body 真的绑定了 workspace 时才追加，RCommon 一个都不加。
  注意判据是「**绑定了**」而不是「`afir.workspace_tensor_ids` 非空」：后者是 fused 级的，
  RCommon 上照样非空（见 RA 设计文档 3.8 第 2 条）。此性质作为一条回归断言。

## 测试设计

### 测试边界

- **入口**：`ScheduleToAfir` 的 `TemplateInfo` 构造（UT 直接喂 `ImplGraph`）；
  AFIRToAscendC 的 materialization pass（lit 直接喂 AFIR 文本，含 TilingData 形参）。
- **出口**：AFIR 文本（`FileCheck`）、生成的 AscendC 源码文本（`FileCheck`）、
  诊断字符串（gtest `EXPECT_*`）。
- **打桩**：不需要打桩——下降侧的输入是 AFIR 文本，TilingData 形参由用例自己写死。
- **验证数据**：S1 产出的真实 RCore plan 的图 dump（`AUTOFUSE_DFX_FLAGS=codegen_compile_debug=true`）。

### 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|----------|------------|----------|----------|
| 功能 | Workspace 节点产出 GM `buffer_desc` 且不进入 kernel 形参表 | `ScheduleToAfirTest` 断言形参 role 序列 + 体内 alloc | UT |
| 功能 | body 签名按 `buffer_id` 升序追加 ui32 偏移形参 | `demo/afir_memory_allocate_workspace_offset.mlir` | UT |
| 功能 | GM workspace 值下降出 `ptr_offset` + `reinterpret_cast` + `SetGlobalBuffer` | 同一 lit 用例 | UT |
| 功能 | 非 workspace 的 GM 入出参仍走无偏移分支 | 同一 lit 用例内的 `%input` 对照 | UT |
| 功能 | entry 把 stage tiling 的 `workspace<ID>` 作为实参传给 body | `afir-to-ascendc-workspace-offset.mlir` | UT |
| 异常 | 节点未在 `workspace_plan` 声明 → `InvalidWorkspace:` | `ScheduleToAfirTest` | UT |
| 异常 | 缺 workspace 形参 → `InvalidBinding:` | lit `expected-error` | UT |
| 异常 | 绑定的 id 没有对应偏移形参 → `InvalidBinding:` | `demo/afir_memory_allocate_workspace_offset_missing.mlir` | UT |
| 异常 | stage tiling 缺 `workspace<ID>` → 文案指向「偏移不会被赋值」 | `afir-to-ascendc-workspace-offset-invalid.mlir` | UT |
| 异常 | `workspace_offset` 缺 `index` → 属性 verifier 挡下 | `smoke/afir_kernel_arg_invalid.mlir` | UT |
| 兼容性 | 无 workspace 的语料生成结果不变 | 既有 `load_sum_store_*` 全套回归 | E2E |
| 特性交叉 | RCore plan 能走完 bridge → 下降 → 生成 kernel | 以 S1 语料钉住的 e2e 冒烟 | E2E |
| 性能 | 不涉及 | 编译时长与执行耗时影响不可测量，不单独立项 | — |

### 测试命令

```bash
find /workspace/build \( -name "*.o" -o -name "*.a" \) -size 0 -delete
cmake --build /workspace/build --target check-ut check-e2e-cpu -j 8
llvm-lit -sv /workspace/build/autofuse/mlir/test/lit
```

预期：全绿。lit 用例不在 ctest 集合内，须单独执行。

910B 数值验证暂无设备，作为遗留项记录，不阻塞本期合入。

## 验收标准

1. **功能**：真实 RCore plan 能走完 `ScheduleToAfir` → `AfirBuilder` → AFIRToAscendC，
   生成的 kernel 中 phase 1 的 Store 目标与 phase 2 的 Load 源指向同一
   `workspace + tiling_data.workspace<ID>`。（已达成：`af-opt --afir-plan-candidate-lowering-pipeline`
   对真实 RCore plan `rc=0`，两个 stage 各自发出 `emitc.cast` + `emitasc.ptr_offset` +
   `emitasc.reinterpret_cast` + `set_global_buffer`，entry 侧两处 `emitasc.member ... "workspace2"`。）
2. **正确性**：生成的偏移表达式与 legacy 对同一 plan 生成的 `SetGlobalBuffer` 逐字符可比对
   （允许变量名差异，不允许偏移来源差异）。
3. **测试**：上表所有 UT/E2E 通过；`check-ut`、`check-e2e-cpu`、lit 全绿。
4. **兼容性**：全部既有 CPU E2E 语料的生成产物与改动前逐字节一致。
5. **文档**：RA 设计文档 §3.8 第 2/3/4 条按本文「更正 1/2」修订完成（已于 `1de8715e` 完成）。
6. **遗留**：910B 上的数值验证、以及 workspace 总量的逐 stage 超配口径，均作为已知项记录，
   不在本期验收范围。

## 分期

按「每一期自己能验证」切，不按文件切：

| 期 | 内容 | 自验方式 |
|----|------|----------|
| W0 | 钉语料：跑通 RCore 探针，dump 出 phase 1 / phase 2 两张真实图，确认 `Workspace` 节点在 phase 1 当汇（`reduce → store → workspace`）、phase 2 当源（`workspace → load → reduce`），两侧 `tensor_id` 相同 | 图 dump 比对，落进本文「假设和依赖关系」 |
| W1 | FR1：bridge 产出 GM 值 | `ScheduleToAfirTest` + RCore plan 能产出 AFIR module（下降仍会失败） |
| W2 | FR2 + FR4：ABI 形参与 entry 实参 | lit 断言签名与调用实参；此时还没接上 `SetGlobalBuffer` |
| W3 | FR3 + FR5：物化与校验 | lit 断言 `ptr_offset` / `reinterpret_cast`；RCore plan 产出可编译 kernel |
| W4 | 回归与验收 | `check-ut`、`check-e2e-cpu`、lit 全绿；既有语料产物逐字节不变 |

W0 的结论可能推翻 W1–W3 的形状（例如两个 phase 的 `tensor_id` 若不同，绑定就不是一块而是两块），
所以它必须先做，不能与 W1 并行。

### 设计文档检查结果

- [x] 跨特性交叉影响：已按 `cross_feature_check.md` 逐项检查，适用项为 Autofuse Codegen/Backend、
      AscendC/Runtime 交互、测试与覆盖率、性能与日志，各自方案见正文
- [x] 编码红线：已按 `编码红线.md` 逐条检查。重点是第 2/3 条（偏移进入指针算术）——
      防线在 host 侧 tiling，kernel 侧依赖该闸且已在正文写明；第 7 条（不扩大范围）已明确
      列出本次不改的内容
- [x] 测试设计：bridge UT + 下降 lit + 适配层 UT + E2E 回归，正向与四类异常输入均覆盖
- [x] 性能评估：编译期为常数级遍历，执行期新增一次指针加法，内存总量口径不变，
      三项影响均不可测量
