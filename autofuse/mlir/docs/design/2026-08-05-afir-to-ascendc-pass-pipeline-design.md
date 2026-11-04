# AFIR → AscendC Pass Pipeline 设计

> 状态：已接入生产；当前顺序于 2026-10-22 对照 `AFIRToAscendCPipeline.cpp` 更新。
> 历史设计中的“本期新增”和资源扩展清单仅代表当时范围；workspace 和 Reduce scratch 链路已落地。
> AFIR 属性与 verifier 以当前 ODS/实现为准，后续专题见 docs 索引。

# 文档结构

第 1～2 章说明目标和当前代码事实；第 3～5 章定义最终 pipeline、函数角色、Pass 契约和 ABI；第 6～9 章说明实现范围、错误处理、
测试和验收。文中“当前代码”表示本 checkout 已存在的实现；“本期新增”表示必须在本次重构中落地的接口或改造。

# 1 简介

## 1.1 目的

本设计定义 `ScheduledResult -> AFIR buffer-form -> AscendC` 的正式 Pass pipeline。Schedule 已经决定了 buffer、queue、复用和
生命周期事实，因此 Bridge 直接生成 `memref` buffer-form AFIR；pipeline 不再把 tensor-form AFIR 先转换成 buffer-form，也不引入
One-Shot Bufferize。

`SelectImplementationPass` 只查询 Capability 并选择一个可用的 AscendC implementation key；`ConvertAFIRToAscendCPass` 才负责把
选定的 AFIR Op 真正改写为 AscendC dialect。二者职责不能合并。

## 1.2 范围

本期包含：

- buffer-form AFIR 的 dialect verifier 和 loop prerequisite 校验；
- `AFIRLoopGeneratePass` 对 Schedule axis/loop SSA 的实体化；
- 合并后的 `AFIRMemoryAllocatePass`：绑定应用和 Schedule 资源物化两个内部阶段；
- Capability selection 和 implementation key 写入；
- 选定 implementation 的 AFIR → AscendC conversion；
- kernel ABI 校验、源码生成和 pipeline 端到端测试；
- 删除 `AFIRToAscendCQueue` demo 的正式构建、注册和文档入口。

本期不包含：

- One-Shot Bufferize 或 AFIR BufferizableOpInterface；
- 新的 reuse/lifetime planner；
- SuperKernel、Python/C++ 对外 API、runtime ABI 的行为修改；
- CCE Intrinsic/PTO-ISA 直接 lowering；未注册的 Gather、Split、TransData、MatMul、Conv 等能力只返回 deterministic `unsupported`。

# 2 总体概述

## 2.1 软件概述

### 2.1.1 项目介绍

正式入口是 `addAFIRToAscendCPipeline(OpPassManager &, session)`。它替换当前仅包含 selection/conversion 的
`addAFIRToAscendCSelectionAndConversionPipeline`。输入来自 `ScheduleToAfir` Bridge，输出是不含 AFIR Op、可由
现有 AscendC translator 继续处理的后端 IR。

### 2.1.2 产品环境介绍

- 依赖固定版本的 LLVM/MLIR、PyAsc 和 CANN Toolkit；
- `af-opt` 用于单 Pass 和 LIT 验证，不是对外 API；
- 普通 CPU 环境可以完成 verifier、Bridge、selection 和 AscendC IR 生成；源码编译和设备 RDV 需要 CANN/Ascend 环境；
- pipeline 不新增 `rt*`、`aclrt*` 或 host runtime 调用。

## 2.2 软件功能

pipeline 按固定顺序完成：

1. 校验 `KernelBody` 中的 canonical buffer-form AFIR 和 Schedule loop 前置条件；
2. 应用 Schedule binding，解析 `BufferDescAttr.size_bytes_expr`，物化 GM/UB/L1/L0 和 AscendC view；
3. 查询 Capability，选择一个有 conversion pattern 的 AscendC implementation；
4. 由 conversion pattern 执行 AFIR → AscendC 改写，并处理 provider 返回的 implementation-specific scratch/workspace；
5. 校验输出不再包含 AFIR Op、旧属性或未解析的选择属性。

## 2.3 设计约束

- 新路径只接受 canonical buffer-form；遇到 tensor-form、`outputs`、`AscTensorGroupsAttr` 或 `tmp_buffers` 直接失败；
- Pass 顺序固定，不依赖无序容器、指针地址或非确定性遍历；
- `BufferDescAttr` 是资源事实唯一来源；`TilingData` 只承载动态表达式解析后的运行时值，allocator 不再按 `qN_size`/
  `bN_size` 字段名直接决定 queue/buffer 大小；
- `Role` 枚举只保留 `Input/Output/Compute/Constant`；queue 的搬运方向由 `AFIR_TransferIn`/`AFIR_TransferOut` trait
  推导，`TensorDescAttr.role` 只表达 value 语义；
- 历史 IR 中的 `role = transfer_in/transfer_out` 不保留兼容别名：load 的片上 destination/result 改为 `Compute`，store 的
  GM destination 改为 `Output`，旧文本 IR 必须迁移后才能进入 canonical pipeline；
- `slot_count` 和 `queue_depth` 分别校验并传递，不用一个字段推导另一个；
- AFIR Core 不依赖 AscendC 类型，AscendC 类型只在 `AFIRMemoryAllocatePass` 的物化阶段出现；
- 任何 shape、offset、stride、size、alignment、queue depth 或 buffer ID 运算都必须 checked；
- `SelectImplementationPass` 只写选择结果和资源需求，不执行 Op 改写；
- `ConvertAFIRToAscendCPass` 只转换已选择且已通过 preflight 的实现，不 fallback 到 demo。
- `AFIRPipelineSync` 是 conversion 的内部前置步骤，不作为正式四阶段 pipeline 的独立阶段；
- conversion target 将 AFIR dialect 视为非法方言，任何未转换 AFIR Op（包括 symbolic evaluation）都必须导致 failure。

## 2.4 假设和依赖关系

- Bridge 已为每个 operand/result/destination 生成按 SSA index 对齐的 `TensorDescAttr` 和 `BufferDescAttr`；
- `KernelBody` 已带 `afir.asc_graph`、`AxisAttr` 和可供 `AFIRLoopGeneratePass` 使用的 Schedule 轴事实；
- `KernelEntry` 只承担 ABI，不包含 AFIR Op；
- 旧 `Template` 函数角色在本期重命名为 `KernelBody`：`FunctionRole::KernelBody=0`，文本值为 `kernel_body`，不保留 `template` alias。
  `KernelBody` 是私有 kernel 代码本体，`KernelEntry` 是对外函数入口；
- `KernelBody` 的函数类型统一为 `() -> ()`，外部输出由 `afir.store` destination 写回 GM；`KernelEntry` 的 GM 输入、GM 输出、workspace
  和 TilingData 只由 ABI 参数描述，不使用 `func.return` descriptor。
- CANN/设备测试属于环境依赖，未运行的设备测试不能标记为通过。

# 3 目标 pipeline 和函数角色

## 3.1 最终数据流

```text
ScheduledResult → ScheduleToAfir → buffer-form AFIR / verifier
  A: AFIRKernelBodySignature → AFIRMaterializeTilingParams
  B: LoopGenerate → ScopeReduceScratch → MemoryAllocate
     → PipelineSync → ExecCondition
  C: SelectImplementation → ConvertAFIRToAscendC
  D: GenTilingData → GenKernelEntry
  → 产物装配与 AscendC/EmitAsc 文本发射
```

`addPlanCandidateLoweringPipeline` 包含 A～D；`addAFIRToAscendCPipeline` 仅包含 B、C。
ScopeReduceScratch 必须看到物化后的循环树；它为共享 scratch 求共同使用作用域并检查尺寸支配关系，
只移动逻辑 alloc 和标注作用域，不改变 buffer ID、大小或物理 lifetime。MemoryAllocate 保留该标注。
MemoryAllocate 在最后一次符号重写完成后对照 `afir.axis_bindings` 检查 `(axis, role)` 绑定存活，
缺失则本 pass 失败；无清单时跳过。此结束校验不代替使用点的支配关系验证。
2026-10-23 已移除独立校验 pass 的注册、命令行选项和流水线步骤，单独运行 MemoryAllocate 也会检查。
PipelineSync 必须先于实现选择，ExecCondition 在资源与同步物化后包裹执行守卫。
Reduce API 及 round_result 合并完成后的 PIPE_V 由 Reduce 下降实现保证。

`KernelEntry` 如果由 Bridge 同时生成，只允许保留 ABI/调用边界，不参与 AFIR Op lowering。入口函数中出现 AFIR Op 是 verifier failure，
不能被 Pass 静默跳过。

## 3.2 Pass 总表

> **勘误（2026-09-14，AFIR 公开面重构验收 8）**：下表「产出」列里的四个阶段标记属性
> （`afir.loop_generated` / `afir.memory_allocated` / `afir.ascendc_materialized` /
> `afir.pipeline_synchronized`）**均已删除**。它们是同一事实的第二份编码（设计文档 D5）：
> 每个阶段的产物就摆在 IR 里——循环生成产出 `#afir.axis_binding`，资源物化产出 `ascendc`
> 方言的算子，流水同步产出 enque / deque / pipe_barrier。「跑过没有」现在直接问 IR，判据见
> `conversion/AFIRToAscendC/LoweringStage.h` 与 `dialect/AFIR/Transforms/LoopGenerate.h`。
> 那两个头文件也记了这套结构判据的**边界**：资源为空时物化不产出任何 `ascendc` 算子。


| 相位 | Pass/阶段 | 当前输入与输出 |
|---|---|---|
| A | KernelBodySignature / MaterializeTilingParams | 补 ABI 参数，将运行期尺寸符号物化为 SSA |
| B | LoopGenerate | 调度轴变为实际循环及 axis_binding |
| B | ScopeReduceScratch | 验证归约累加轴、scratch 角色与共享作用域；物理分配不变 |
| B | MemoryAllocate | canonical buffer 变为 AscendC 资源和局部视图，完成符号改写后校验轴绑定存活 |
| B | PipelineSync / ExecCondition | 建立同步边，再包裹执行条件 |
| C | SelectImplementation / ConvertAFIRToAscendC | 选择实现并下降 AFIR 算子 |
| D | GenTilingData / GenKernelEntry | 生成 tiling schema 和入口分派 |

不新增 `VerifyAFIRForAscendCPass`、`PrepareAFIRKernelABIPass` 或 `MaterializeAFIRSchedulePass`。前者由 dialect verifier 承担；后两者的
职责分别由 ABI helper/Bridge 和已有 `AFIRLoopGeneratePass`、`AFIRMemoryAllocatePass` 承担。

## 3.3 `AFIRMemoryAllocatePass` 的两个内部阶段

把 4.7 的绑定应用和既有 MemoryAllocate 合并成一个 Pass 是合理的，但两个阶段必须保持边界：

### 3.3.1 `applyCanonicalStorageBindings`

- 读取每个 operand/result/destination 的 `TensorDescAttr` 和 `BufferDescAttr`；
- 用 `(func.func, 推导资源域, buffer_id)` 构造稳定 resource key；
- 将 `repeats`、`strides`、`vectorized_strides` 和 view layout 转换为 `StorageView`；
- 原样校验并保存 `reuse_id`、`slot_count`、`queue_depth`、`life_time_axis_id`；
- 不创建 AscendC Op，不改变 dtype、shape、layout、alias 或资源 ID；
- 所有 descriptor 校验成功后才进入下一阶段。

### 3.3.2 `resolveSizeExpressionsAndMaterializeAscendCResources`

- `BufferDescAttr.size_bytes_expr` 是唯一的资源大小来源；
- 表达式中的符号解析为 `TilingData` 成员、axis SSA 或静态常量；`q<id>_size`、`b<id>_size` 仅是可能的表达式符号，
  不能成为 descriptor 之外的第二份大小事实；
- 生成 checked index/byte-size SSA，拒绝未解析符号、0/负值、除零和溢出；
- 使用 `space` 和推导出的 Transfer/Compute role 创建 queue/buffer/LocalTensor/GlobalTensor；
- 将 `slot_count` 传给 buffer count，将 `queue_depth` 传给 queue type/init；
- 全部生命周期、大小和资源兼容性检查通过后，事务性创建 AscendC 资源；失败时删除本阶段已创建的临时 Op。

本阶段不调用 Capability provider。provider 的 `ResourceRequirements` 只有在未来 Schedule 迁移到 MLIR、并由 selection 构造
`ResourceOwnerContext` 后才查询。selection 将有 SSA owner 的 temporary buffer 写入对应 Op 的 `BufferDescAttr`，将无 SSA owner 的
temporary/workspace 按稳定 owner ordinal 写入 `func.func` 的 `afir.workspace_plan`，再交给 conversion 物化；provider 不负责推断
`life_time_axis_id`。KernelBody/函数 owner 使用 `-1`，loop owner 使用明确的 axis id，缺失时 failure。

该 Pass 只处理 `KernelBody`。`KernelEntry` 的 ABI 由 Bridge 和 ABI helper 生成/校验，不包含 AFIR Op。

## 3.4 Select 与 Convert 的职责边界

`SelectImplementationPass` 的输入是已物化的 AFIR Op 和 canonical descriptor。它调用 `CapabilityQuery::queryLowering()`，从有完整
pattern/mapping/provider 的候选中选择一个 `ImplementationKey`，并把选择属性和 provider 返回的 `ResourceRequirements` 写入 IR。
它不创建 AscendC API Op，不生成 DataCopy，不重写 AFIR operands/results。

`ConvertAFIRToAscendCPass` 先执行幂等的 pipeline synchronization，再校验选择属性与 registry/pattern 一致，然后执行 DialectConversion：

- 将 AFIR load/store/compute Op 改写为 AscendC Op；
- 将已选择的 mapping/execution spec 写入 AscendC 属性；
- 根据 selection 产生的 `BufferDescAttr`/`workspace_plan` 物化 implementation-specific scratch；
- 未选择、选择过期、无 pattern、unsupported 或 undetermined 时整体 failure。
- `load/store` 只接受已注册且 capability 可证明的搬运 mapping；conversion 成功后 IR 中不得残留任何 AFIR Op。

# 4 ABI 和 Pass 接线

## 4.1 现有 ABI 工具

现有 [`AscendCKernelABI.h`](../../conversion/AFIRToAscendC/AscendCKernelABI.h) / `.cpp` 是 ABI 的事实工具，提供：

- `checkAscendCKernelABI(func, role)`：检查 `workspace`、`TilingData`、输入数量和 role/`ascendc.global` 一致性；
- `setAscendCKernelAttributes(func, role, inputCount)`：设置 `ascendc.aicore`、`ascendc.num_inputs` 和 global 属性；
- `getAscendCGlobalMemRefType()`、`getAscendCWorkspaceType()`、`getAscendCTilingDataType()`：提供 ABI 类型。

本期不新增 `PrepareAFIRKernelABIPass`。Bridge 负责生成最终 buffer-form 函数边界，ABI helper 负责检查和补齐必要属性；如果仍需
函数签名改写，重构 `AFIRTemplateFunctionSignaturePass` 的内部逻辑为 `AFIRKernelFunctionSignature` helper，但该旧 tensor-form
Pass 不进入新 pipeline。函数名 `Template` 同步改为 `KernelBody`，避免把 kernel 本体误认为函数入口。

## 4.2 `addAFIRToAscendCPipeline`

本期必须实现并注册：

```cpp
LogicalResult addAFIRToAscendCPipeline(OpPassManager &pm,
                                       const AFIRToAscendCPipelineSession &session);
```

实现只负责按固定顺序 `pm.addPass(...)`：

```cpp
pm.addPass(createAscendCLoopGeneratePass());
pm.addPass(createAFIRMemoryAllocatePass());
pm.addPass(createSelectImplementationPass(session.implementationRegistry,
                                           session.targetProfile));
pm.addPass(createConvertAFIRToAscendCPass(session.implementationRegistry,
                                           session.targetProfile));
```

Target Profile 和 ImplementationRegistry 必须在 pipeline 创建前完成加载、注册和 freeze。pipeline 入口发现 session 不完整时直接
返回 `failure()`。每个 Pass 可以单独通过 `af-opt` 调试，但产品入口只使用上述组合。

## 4.3 Queue demo 边界

`mlir/demo/AFIRToAscendCQueue` 只是早期队列忠实演示，不是正式 pipeline、兼容入口或 fallback。它不参与 capability selection、
MemoryAllocate、conversion 和产品构建。本期删除其 Pass 注册、CMake、测试和文档入口；正式实现只保留
`mlir/conversion/AFIRToAscendC`。

# 5 软件设计和错误处理

## 5.1 关键数据结构

- `AFIRToAscendCPipelineSession`：每次编译构造一份，持有只读 `TargetProfile` 和 frozen `ImplementationRegistry`；
- `StorageView`：MemoryAllocate 内部临时结构，记录 resource key、offset、size、stride 和生命周期组，不写入全局状态；
- `BufferDescAttr`：唯一资源事实；Op 级 descriptor 与 SSA index 对齐，`workspace_plan` 承载无 SSA owner 的资源；
- `ascend.implementation`：selection 输出的实现 kind/id，不包含资源大小或生命周期事实。

## 5.2 错误处理

| 错误 | 处理 |
|---|---|
| `KernelEntry` 包含 AFIR Op | dialect verifier failure |
| `KernelBody` 缺少 descriptor、loop/axis 或混入旧 attrs | verifier/LoopGenerate/MemoryAllocate failure |
| `size_bytes_expr` 缺失、符号无法解析、0/负值、除零或溢出 | MemoryAllocate 在创建资源前 failure |
| 同一 resource key 的 space/layout/size/alignment/slot/depth 冲突 | binding apply failure |
| queue 缺 `slot_count` 或 `queue_depth` | failure，不写默认值 |
| provider 返回 unsupported/undetermined 或没有 registered pattern | Select failure，不进入 conversion |
| selection 属性与 descriptor/registry 不一致 | Convert preflight failure |
| 任一 conversion pattern 未完全转换 AFIR Op | `applyFullConversion` failure，不生成部分成功结果 |

## 5.3 对子模块的修改

| 模块 | 本期改动 |
|---|---|
| `dialect/AFIR` | canonical Attr、role Trait/Interface、descriptor 对齐规则和 verifier |
| `bridge/ScheduleToAfir`、`bridge/AfirBuilder` | 直接生成 buffer-form；生成 operand/result/destination BufferDesc；删除旧 `outputs` 生成 |
| `dialect/AFIR/Transforms/LoopGenerate.*` | 通用 AFIR loop 生成；AscendC 的 TilingData binding 由 `conversion/AFIRToAscendC/AscendCLoopGenerate.*` 注入 |
| `conversion/AFIRToAscendC/AFIRMemoryAllocate.*` | 删除 tensor→buffer 旧路径，实现两个 canonical 内部阶段和 size expression resolver |
| `conversion/AFIRToAscendC/AscendCKernelABI.*` | 复用并扩展 ABI 校验/属性和 buffer-form 边界工具；KernelEntry 只接收 ABI 参数，不接收函数返回 descriptor |
| `target/Ascend/Capability` | provider 读取 canonical descriptor；selection 只写 implementation key 和 resource requirements |
| `conversion/AFIRToAscendC/AFIRToAscendCPipeline.*` | 接入四个正式 Pass，提供 build pipeline 入口 |
| `demo/AFIRToAscendCQueue` | 删除正式构建、注册、测试和文档入口 |

# 6 非功能需求与跨特性交叉影响

## 6.1 可维护性和可测试性

每个 Pass 有独立 LIT；组合入口有一条真实 `ScheduledResult -> AFIR -> AscendC` E2E。Pass 内部状态只在一次 function/session 生命周期内存在，
不保存跨编译任务的静态对象。

## 6.2 特性交叉影响

| 场景 | 适用性 | 结论 |
|---|---|---|
| SuperKernel Python/C++/AOT | 不适用 | 不修改接口、ABI、wheel 或设备 API |
| Autofuse 图优化 | 适用 | 只消费已调度图；保持数据边和控制边，固定 Pass 顺序 |
| Autofuse Codegen/Backend | 适用 | 新增 buffer-form 物化、Capability selection 和 AscendC conversion |
| AscendC API/Runtime | 适用 | 校验 queue depth、slot count、size 和资源生命周期；不新增 host runtime API |
| Python/C++ 混合绑定 | 不适用 | DTO 是内部 C++ 接口，不改变 pyautofuse |
| 构建与打包 | 适用 | 更新 CMake、`af-opt` 注册并删除 Queue demo 入口；验证增量和 `--no-autofuse` |
| 测试与覆盖率 | 适用 | LIT、gtest、Capability、源码生成和设备 RDV 全覆盖 |
| 性能与日志 | 适用 | 资源解析和 descriptor 遍历线性；禁止默认逐 Op 海量日志 |
| 兼容性 | 适用 | 旧 AFIR attrs 不再接受；SuperKernel/Legacy Codegen 对外接口保持不变 |

## 6.3 性能和安全

Bridge、verifier、binding apply 和 size resolver 按稳定 IR 顺序线性遍历。所有大小、偏移、stride、axis 和 ID 在索引、循环或资源申请
前完成范围校验和 checked arithmetic。失败路径不写完成 marker，不留下 AscendC 资源。生成的 kernel 不应增加无必要的 DataCopy、queue 或 loop。

# 7 测试设计

## 7.1 测试矩阵

| 类别 | 关键用例 | 方法 |
|---|---|---|
| verifier | descriptor 数组长度、旧 attrs、KernelEntry 含 AFIR、role/alias 冲突 | LIT/gtest |
| LoopGenerate | KernelBody axis、动态 axis、缺 graph、重复运行 | LIT/gtest |
| MemoryAllocate | 两阶段顺序、size expression、TilingData symbol、queue depth/slot、lifetime/reuse、回滚 | LIT/gtest |
| Bridge | operand/result/destination 对齐、多结果、Gather/Split/TransData/MatMul unsupported | Bridge gtest/LIT |
| selection | supported/unsupported/undetermined、stale selection、resource requirement 合并 | Capability UT/LIT |
| conversion | selected key 与 pattern、一元/二元/load/store、未转换 AFIR | conversion LIT |
| E2E | ScheduledResult → buffer-form → AscendC IR → 源码 | ST/E2E |
| 设备 | load/compute/store、单双缓冲、queue depth、动态 size | RDV |

## 7.2 测试命令

```bash
sh build.sh -u --module=autofuse_framework -j 8
sh build.sh -s --module=autofuse_e2e -j 8
AF_CANN_VOLUME=cann-volume bash mlir/scripts/enter_dev_container.sh -- \
  cmake --build /workspace/build --target check-autofuse-mlir -j 8
git diff --check
```

无 CANN 环境时至少完成 dialect、Bridge、LoopGenerate、MemoryAllocate、selection、conversion 和失败诊断测试；不能把未运行的源码编译或 RDV 标记为通过。

# 8 验收标准

- `addAFIRToAscendCPipeline` 按本文顺序串联四个正式 Pass，并替换旧的 selection/conversion-only 接口；
- ScheduleToAfir 直接生成 buffer-form AFIR，不经过 One-Shot Bufferize；
- `KernelBody`/`KernelEntry` 的职责和 Pass 范围明确，`KernelEntry` 不含 AFIR Op；
- `FunctionRole::Template` 已替换为 `FunctionRole::KernelBody`，旧 `template` textual attribute 确定性拒绝；KernelBody 和
  `afir.store` 的 `()` 返回边界不维护 `return_descs`/`return_buffer_descs`；
- `AFIRMemoryAllocatePass` 的两个内部阶段完整实现，`size_bytes_expr` 是资源大小唯一事实来源；
- Select 只写 implementation key 和资源需求，Convert 才执行 AFIR → AscendC lowering；
- 旧 `outputs`、`AscTensorGroupsAttr`、`tmp_buffers` 和 Queue demo 不进入正式 pipeline；
- 所有未注册特殊 Op 返回稳定 `unsupported`，不生成半成品 IR；
- LIT/UT/ST/RDV 和 `git diff --check` 通过，未运行的环境测试明确标注。

## 8.1 设计文档检查结果

- [x] 跨特性交叉影响：已检查 Autofuse 图优化、AscendC/backend、构建交付、测试、性能和兼容性；SuperKernel/Python/C++ 公共接口不受影响。
- [x] 编码红线：已检查 checked arithmetic、资源生命周期、图改写确定性、ABI/API 和 runtime 边界。
- [x] 测试设计：已覆盖 verifier、Bridge、两个 MemoryAllocate 阶段、selection、conversion、E2E 和 RDV。
- [x] 性能评估：已说明稳定线性遍历、size expression 解析开销和设备端搬运/队列数量约束。
- [x] 贡献规范：属于 MLIR 内部新流程，实施前应关联 Requirement Issue，commit 使用 `<type>: <简短描述>`。
