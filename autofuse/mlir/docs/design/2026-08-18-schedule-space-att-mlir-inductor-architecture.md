# Schedule 到 MLIR-Codegen 的 Bridge 组织形式、低耗时编译执行方案与 ATT 交互设计

> 状态：功能需求 1（`TilingFieldSchema`）、2（`AttSchemaAdapter`）、3（`StaticPlanSelector`）、4（动态/静态候选
> 筛选策略）、6（结构 fingerprint）已实现并有 UT 覆盖；功能需求 5（编译驱动改直接调用 bisheng）未实现，暂不
> 开发；"生产入口如何组装候选"（原功能需求 2）仍是开放依赖项；AFIR 的 `node_id`→`subgraph_id`、
> variant→template 命名和 `PyStructType` 校验收敛已完成，见第八版说明。
> 日期：2026-08-19（本版）：
>   1. `node_id` 概念重命名为 `subgraph_id`（更准确：它对应一个原始 `AscBackend` 子图，不是一个 IR 节点）；
>   2. 去掉过度设计的 `ScheduleResultView` 类型，plan/stage/template 编号仍是 vector 下标，不需要专门包装；
>   3. 所有新增代码统一放到 `autofuse/mlir/` 目录下，legacy 代码只新增少量调用点，不重构；
>   4. 原 "att-bridge" 拆成两个职责清晰的组件：`AttSchemaAdapter`（对接 ATT 输出与统一 schema）和
>      `StaticPlanSelector`（静态 shape 编译前选 winner）；
>   5. "预编译共享 runtime" 从"待研究项"改为"已根据 CANN 公开文档确认不可行"，见"假设和依赖关系"。
> 第四版日期：2026-08-20（review 修正，不改变设计结论）：
>   1. 修正功能需求 1 对 `codegen::TilingData::Generate` 字段生成逻辑的描述（实际是
>      `ProcessSingleGroup`/`ProcessMultiGroup` 遍历 `GetAllSizeVar()`，不是 `impl_graphs[0].GetAllAxis()`）；
>   2. 修正对 07-27 文档"第 4.2 节"的虚构章节引用，改为具体小节名；
>   3. 修正 08-12 文档 6.1 节引用的措辞，"实测依据"改为"源码走查依据"，避免高估证据强度；
>   4. 原"功能需求 8"（`TilingDataType` 收敛重复校验）压缩为一段后续任务说明，不再占用完整四段式格式；
>   5. 安全检查补充 Python/C++ 绑定异常/GIL、增量与离线构建两条红线的显式结论。
> 第五版日期：2026-08-20（因 PR `feat/mlir-tiling-data-header`（#19）调整范围，不等其合入即可独立推进）：
>   1. 发现 PR#19 已实现 `GenTilingDataPass`/`runGenAggregateTilingData`/`runGenAggregateKernelEntry`/
>      `TilingDataHeaderEmitter`，和原功能需求 1 的 `ToPyStructType`、原功能需求 2 的 `BuildScheduleBundle`
>      高度重叠，且已改动 `GenKernelEntry.cpp`（与本文"不修改"的假设冲突），见"假设和依赖关系"；
>   2. 原功能需求 1 拆分：只保留服务 ATT 和 legacy `codegen_tiling_data.cpp` 的非 MLIR 字段 schema 部分，
>      去掉 `ToPyStructType`/`mlir/bridge/TilingSchemaToMlir/`（该产出等价物已由 PR#19 的
>      `GenTilingDataPass` 提供）；
>   3. 原功能需求 2（Bridge 生产入口与 Bundle 组装）从详细设计降级为一条"开放依赖项"：只声明
>      `PlanSelection` 列表的接口边界，具体如何组装成可编译产物不在本文设计，待 PR#19 最终形态（尤其是
>      它明确留白的 SR/Entry module 最终 link/finalize 方案）确定后再补；
>   4. 原功能需求 3/4/5/6/7（`AttSchemaAdapter`、`StaticPlanSelector`、动态/静态 shape 编译单元策略、
>      编译驱动切 bisheng、结构 fingerprint 接口）与 PR#19 不冲突，重新编号为功能需求 2-6，可独立设计和
>      交付，不依赖 PR#19 合入节奏；
>   5. `node_id`→`subgraph_id` 重命名范围标注为"待 PR#19 落地后重新核实"，因为 PR#19 新增文件
>      （`GenAggregateKernelEntry.cpp`/`GenAggregateTilingData.cpp`）也在用 `node_id`，覆盖面比本文原先
>      列出的两个文件更大。
> 第六版日期：2026-08-20（本分支开发修正）：补充文档导航、坐标映射、ATT/runtime 分层、版本化缓存 key 和生产 MLIR pipeline 接缝。
> 第七版日期：2026-08-21（开发完成后复核，不改变设计结论）：
>   1. PR `feat/mlir-tiling-data-header`（#19）已合入（`f63608a2`），全文把"在途 PR/尚未合入"改为"已合入"；
>   2. 功能需求 1（`TilingFieldSchema`，`mlir/common/tiling_schema/`）、2（`AttSchemaAdapter`，
>      `mlir/bridge/AttSchemaAdapter/`）、3（`StaticPlanSelector::SelectStaticWinner`）、4
>      （`SelectDynamicCandidates`）、6（`mlir/common/fingerprint/`）均已按本文接口实现，并有对应 UT
>      （`mlir/test/ut/schedule/`），全文相应章节补充"已实现"标注和文件路径；
>   3. 功能需求 5（编译驱动改直接调用 bisheng）确认未实现、暂不开发，标题已注明"暂时先不做"，本版不改变
>      这一结论；
>   4. **开放依赖项（原功能需求 2，生产入口如何组装候选）复核后确认仍未解决**：`Codegen::GenerateForInductor`
>      仍只调用 `MaybeDumpAfirFromSchedule` 诊断 dump 钩子；PR#19 的 `runGenAggregateKernelEntry` 只在
>      Entry module 里为每个 SR module 生成私有前向声明（`declaration.setPrivate()`，空函数体），并未把
>      SR 函数体搬入 Entry module 的符号表——即"SR module 与 Entry module 最终 link/finalize"这一步在
>      PR#19 合入后依然空缺，本文继续把它列为开放依赖项，不因 PR#19 已合入而视为解决；
>   5. `node_id`→`subgraph_id` 重命名复核后确认覆盖范围与本文原预测一致（`AfirBuilder.cpp`、
>      `ScheduleToAfir.cpp/.h`，以及 PR#19 新增的 `GenAggregateKernelEntry.cpp`/`GenAggregateTilingData.cpp`
>      均仍在用 `node_id`/`autofuse.node_id`），但重命名本身仍未执行，继续作为后续任务，不在本阶段完成。
> 第八版日期：2026-08-21（命名和校验收敛）：
>   1. AFIR Module、bridge DTO、aggregate contract 和测试中的 `node_id` 已统一为 `subgraph_id`；legacy
>      `node_idx_to_scheduled_results` 字段名保持不变；
>   2. `impl_graphs` 在 AFIR 中统一称为 template，相关属性、DTO、selector、函数名和测试由 variant 改为
>      template；
>   3. `emitasc::PyStructType` 和 ABI `TilingData` 的重复校验已收敛到 `AscendCKernelABI` helper；AFIR 核心不
>      新增同构结构类型，EmitAsc 类型仍只属于 `AFIRToAscendC` 后端边界。
> 迁移顺序：与 07-27 文档一致，Codegen → ATT 交互层 → Schedule；本版只覆盖 Codegen 阶段和 ATT 交互层。

阅读导航：先阅读第 1～2 章了解问题、范围和现状，再阅读第 3 章的功能边界与数据流；第 4～6 章分别说明性能、
接口和软件结构，第 7～10 章给出安全、兼容性、测试和验收条件。文中标记“当前实现”的内容来自现有源码走查，
标记“规划”的内容尚未实现，标记“开放依赖项”的内容不作为本次实现承诺。

**设计文档检查结果**

- [x] 跨特性交叉影响：已按 `cross_feature_check.md` 逐项检查 SuperKernel、Autofuse 图优化/Codegen、AscendC/Runtime、Python/C++ 绑定、构建打包、测试、性能和兼容性，见第 3.3.5 节。
- [x] 编码红线：已按 `编码红线.md` 检查外部输入索引校验、整数溢出、资源生命周期、ABI/API 兼容、图改写确定性和 runtime 接口生命周期，见第 7 章。
- [x] 测试设计：覆盖 schema 统一、ATT 适配、静态特化裁剪、编译驱动切换和 legacy fallback；schema/ATT
  适配/候选筛选部分已落地 UT（`mlir/test/ut/schedule/`），生产入口组装部分待其详细设计确定后补充。
- [x] 性能评估：分阶段给出编译时长和执行性能的评估口径，不用设计推测代替实测数据。

# 1 简介

## 1.1 目的

Autofuse 正在把 Codegen 模块从 ascir 迁移到 MLIR（AFIR 方言），总体链路仍是 Inductor + Autofuse（Schedule + ATT） +
MLIR（Codegen）。本文回答三个具体问题：

1. `Schedule` 产生的 `FusedScheduledResult` 通过 bridge 转成 AFIR 时应该用什么组织形式，才能既支持 MLIR
   Codegen，又尽量不改动 ATT？
2. Codegen 最终生成的各种模板如何编译和执行，才能显著降低 Inductor 场景下的编译耗时，同时不改变现有一次
   launch 的执行语义？
3. 旧 ATT 模块如何与新 MLIR-Codegen 模块交互，即 `TilingData` 在两边如何对齐？

结论已与 owner 对齐，本文把结论落成可实施的设计。

## 1.2 范围

本设计包含：

- `TilingData` 字段布局的统一生成方案，供 legacy Codegen 和 ATT 共用（非 MLIR 部分）；
- ATT 与新 MLIR-Codegen 之间的接口（不改 ATT 核心 solver）；
- 动态 shape 和静态 shape 两种场景下候选筛选策略（编译前选哪些候选，不涉及"怎么组装成可编译产物"）；
- 编译驱动从"每 kernel 一次 CMake configure"切换为直接调用 bisheng 的方案；
- 面向 Inductor/torchair 的结构 fingerprint 前置方案（接口级，不修改 torchair 仓库）；
- 新增代码的目录归属和依赖边界（尽量都放 `autofuse/mlir/` 下，legacy 代码只做最小侵入调用）。

本设计不包含：

- 重新设计 ATT 的求解器、评分公式或候选枚举算法；
- Codegen 内部 VF、CV 场景的 MLIR 迁移（按 owner 要求单独迁移）；
- Inductor + Autofuse（Schedule+ATT）+ MLIR（Codegen）的端到端联调（单独任务，依赖本文组件落地）；
- `optimize::Optimizer` 内部的调度算法改动；
- 立即删除 `FusedScheduledResult`、`codegen::TilingData` 等 legacy C++ 类型；
- 预编译共享 runtime（已确认在公开接口范围内不可行，见"假设和依赖关系"）；
- `node_idx_to_scheduled_results.size() > 1`（多 `AscBackend` 子图）的支持，继续 legacy fallback，与 07-27 文档结论一致；
- **`FusedScheduledResult` → AFIR bridge 的生产入口组装方式**（即原功能需求 2）：与 PR
  `feat/mlir-tiling-data-header`（#19，已合入）高度重叠且部分架构冲突，本文不预先绑定实现，只声明接口
  边界；PR#19 合入后复核，它自己留白的"SR module 与 Entry module 最终 link/finalize"仍未解决，具体方案
  待另行设计，见"假设和依赖关系"。

# 2 总体概述

## 2.1 软件概述

### 2.1.1 项目介绍

当前主链路：

```text
OriginGraph → optimize::Optimizer → FusedScheduledResult → ATT 生成 host tiling 源码
  → legacy Codegen 生成一个 AscendC kernel 入口 → bisheng 编译 → kernel.so + wrapper.so
  → Inductor 侧 dlopen，调用 AutofuseTiling + AutofuseLaunch，一次 runtime launch
```

`FusedScheduledResult` 的四层结构（`common/schedule_result.h:23-63`）：

```text
FusedScheduledResult
  node_idx_to_scheduled_results[node_index]     # legacy 字段名不变；本文把这一维的 ID 概念叫 subgraph_id
    ScheduledResult[result_index]                # 一个完整执行方案（下文称 "plan"）
      ScheduleGroup[group_index]                 # 方案里的一个执行 stage
        impl_graphs[template_index]                # 该 stage 的一个实现 template（af::AscGraph）
```

命名说明：`node_idx_to_scheduled_results` 这个 legacy 字段名本身不改。但它的下标对应的是一个原始
`AscBackend` 子图（融合前的一个独立子图），不是一个算子/IR 节点，本文和新增代码里统一用 **`subgraph_id`**
指代这个概念，不用 `node_id`，避免和"节点"混淆。`plan_id`/`stage_id`/`template_id` 三个名字保留（对应
`ScheduledResult`/`ScheduleGroup`/`impl_graph`），已核对过 `GenMulGroupKernelWithRegTilingKey` 的真实执行
语义，映射准确。当前实现已把 `AfirBuilder.cpp`、`ScheduleToAfir.cpp/.h`、
`GenAggregateKernelEntry.cpp` 和 `GenAggregateTilingData.cpp` 中的 AFIR 属性、contract key、参数和局部变量统一为
`subgraph_id`；这只是语义命名调整，不改变 legacy 容器或执行行为。

坐标是所有候选、tiling 字段和 dispatch 的唯一关联键，不使用指针地址、图名称或 `tiling_key` 代替：

| 坐标 | legacy 来源 | 语义 | 规划中的消费方 |
|---|---|---|---|
| `subgraph_id` | `node_idx_to_scheduled_results` 的外层下标 | 原始 `AscBackend` 子图在融合结果中的稳定位置；第一阶段只接受一个子图 | AFIR Module 属性、错误诊断、fingerprint |
| `plan_id` | 外层 `std::vector<ScheduledResult>` 下标 | 一个完整执行方案；ATT 可以在运行时返回它的选择 | `PlanSelection`、Entry dispatch |
| `stage_id` | `ScheduledResult::schedule_groups` 下标 | 方案中的执行阶段 | `func.func` 的 `afir.stage_id` |
| `template_id` | `ScheduleGroup::impl_graphs` 下标 | 阶段内的实现图 | `func.func` 的 `afir.template_id` |

以下字段必须在 bridge 和 ATT adapter 中保持同一份语义，不允许只复制 ID 而丢失调度约束：

| legacy 字段 | 目标表示 | 缺失时处理 |
|---|---|---|
| `ScheduleGroup::double_buffer` | AFIR 图函数的 `afir.double_buffer`，同时进入 tiling/schema 输入 | 字段无法编码时失败并回退 legacy |
| `ScheduleGroup::graph_name_to_score_funcs` | 按 `impl_graph` 名称查询后写入 `GraphInfo::score_func` | 找不到匹配项按当前 legacy 默认处理，但必须记录诊断 |
| `ScheduledResult::var_relations` | `afir.var_relations`，保留 dst/src stage、变量名和表达式 | 任一关系无法序列化时拒绝该 plan |
| `score_func`、`is_reduce_mem_reuse`、`enable_group_parallel`、`cube_type` | `afir.*` Module 属性，并传给 ATT model | 不支持的 `cube_type` 走明确 Unsupported，不静默降级为默认值 |
| `origin_vars`、shape guards/asserts、workspace plan | AFIR Module/函数属性和 `BufferDescDTO`，供 tiling 与资源检查使用 | workspace/表达式溢出或非法时整个 fusion 回退 |
| `input_nodes`、`output_nodes`、`workspace_nodes`、`gm_tensor_sizes`、`frontend_shape_vars` | tensor/buffer ABI、workspace 描述和 shape 环境的输入；不能从 impl graph 重新推导 | 当前 pipeline 尚未消费的字段保留版本化 opaque payload，并在生产接线前完成结构化映射；禁止静默丢弃 |

无法马上结构化的 legacy 字段使用带 schema 版本的 opaque payload 保存原始表达式和顺序。opaque payload 不能
参与 device 端索引或内存偏移计算；下游只有在完成字段校验后才能消费它。

`node_idx_to_scheduled_results.size() > 1` 时不得把多个子图压成同一个 `subgraph_id`；第一阶段返回
`Unsupported` 并保留 legacy 路径。这样可以在后续支持多子图时扩展坐标，而不改变已缓存 artifact 的含义。

MLIR 侧已有的、和本文直接相关的实现：

- `mlir/bridge/ScheduleToAfir/ScheduleToAfir.cpp`：已实现"一个 `ScheduledResult` → 一个 AFIR `ModuleOp`"，
  Module/func 属性已带 `subgraph_id`、`plan_id`、`stage_id`、
  `template_id`、`double_buffer`、workspace 等信息，但只在 统一 dump 前缀（`ascir::utils::GetDumpFilePrefix()`） 设置时触发，是调试
  dump 钩子，不是生产入口（`ScheduleToAfir.h:11-21`）。
- `mlir/conversion/AFIRToAscendC/GenKernelEntry.cpp`：已实现多 plan bundle → 统一 `kernel_entry` 的两级
  dispatch（外层按 `afir.plan_id` 选 plan，plan 内按 `afir.stage_id`/`afir.template_id` 选
  template，用嵌套 `scf.index_switch` 而非笛卡尔积展开），并已经把 tiling data 布局表达成类型化的
  `ascendc.tiling_schema`（`emitasc::PyStructType`）Module 属性。
- 缺口：`ScheduleToAfir` 产出的一组独立 Module，从未被组装成 `GenKernelEntry` 期望的 bundle Module（带
  `autofuse.fused_entry_name`/`ascendc.tiling_schema`），且 `ScheduleToAfir` 本身没有生产入口。
- **PR#19（已合入）**：`feat/mlir-tiling-data-header`（#19，作者 zqmin，直接基于上述代码状态，已合入
  `af-mlir`/本分支，共 15 个提交，`f63608a2`）实现了这个缺口的一种解法：`GenTilingDataPass` 从 bridge 完的
  AFIR IR 推导单 SR schema（产出 `ascendc.tiling_schema`）；`runGenAggregateTilingData`/
  `runGenAggregateKernelEntry` 把多个独立 SR Module 聚合成一个 Entry module（`autofuse.sr_entries`
  contract + 跨 module 声明/dispatch）；`TilingDataHeaderEmitter` 从 schema 递归生成 C++ header。代价是它
  把 `GenKernelEntry.cpp` 改了 287 行（dispatch 从"单 module 内 plan/stage/template 三级"变成"SR module 内
  stage/template + 独立聚合层"）。**合入后复核确认**：它的多 module 聚合方案仍然留白"SR module 与 Entry
  module 的最终 link/finalize"——`runGenAggregateKernelEntry` 目前只在 Entry module 里为每个 SR module
  生成私有前向声明（空函数体），并未把 SR 函数体搬入 Entry module 的符号表，"多个独立 module 怎么变成一个
  bisheng 能编译的单一 `__global__` 符号"这一步依然没有解决；同时它完全没有覆盖 ATT 和 legacy
  `codegen/codegen_tiling_data.cpp` 的字段统一问题（schema 只在 MLIR 世界内部流转）。本文因此继续把
  "生产入口组装"（原功能需求 2）列为开放依赖项，不因 PR#19 已合入而视为解决，不预先设计与它冲突的实现，
  见"假设和依赖关系"。

`AutofuseLaunch` 的真实调用约定（函数签名见 `autofuse/codegen/codegen_kernel.cpp:4515`，`<<<>>>` 调用点见
`:4911-4914`，torchair `inductor_npu_ext` 的 `_asc_codegen.py`/`revert_ascir.py` 与之一致）：

```cpp
extern "C" int64_t AutofuseLaunch(uint32_t blockDim, void* stream, /* tensor/workspace 参数 */,
                                   AutofuseTilingData* tiling_data) {
  graph_name<<<blockDim, nullptr, stream>>>(/* 参数展开 */);
  return 0;
}
```

上面的省略号不是可变参数接口：实际生成的声明按融合图固定展开为输入指针/数量、输出指针/数量、`workspace`
和 `AutofuseTilingData*`，顺序由 `GetScheduledResultInputOutput` 决定。MLIR 第一阶段必须逐字节保持该
host C ABI（包括 `uint32_t blockDim`、`void* stream`、workspace 指针和 tiling data 指针），新增的
descriptor/`LaunchFrame` ABI 不在本设计范围内。

同样，现有 Inductor `AutofuseTiling` 生成函数返回 `int64_t`，通过 `uint32_t *workspaceSize` 和
`uint32_t *blockDim` 写回结果，并按生成的 shape 参数读取输入；本文不把这些字段改成 `uint64_t` 或 opaque
payload。若未来要演进 descriptor ABI，必须先增加 adapter、旧新 ABI 双路测试和 manifest 版本校验。

`<<<blockDim, nullptr, stream>>>` 是 bisheng 对 AscendC `__global__ __aicore__` 函数的编译期确定符号的三尖括号
调用。CANN 官方"Kernel Launch"算子开发文档（[hiascend.com](https://www.hiascend.com/document/detail/en/canncommercial/800/opdevg/Ascendcopdevg/atlas_ascendc_10_0056.html)）
显示，除了 `<<<>>>`，CANN 还提供 `ACLRT_LAUNCH_KERNEL(kernel_name)(...)` 这个宏形式的调用方式，但它同样依赖
项目构建时自动生成的 `aclrtlaunch_{kernel_name}.h` 头文件，调用点在**编译期**就要看到目标 kernel 的声明，
不是运行时按 handle/字符串动态解析。也就是说，CANN 公开的两种 kernel 调用方式都要求"一次 launch 对应一个
编译期已知名字的 `__global__` 函数"，这是硬约束，不是 Autofuse 自己的实现选择。因此"多个结构候选要不要挤
进同一个 device kernel"不是自由选择的实现细节：要么把所有候选分支放进同一个 `graph_name` 函数体内部、由
device 端按 `tiling_key` 内部跳转（legacy `AppendFuncCall` 和 `GenKernelEntry.cpp` 的 `scf.index_switch`
都是这个模式），要么在 host wrapper 里对多个不同名字的 kernel 各写一次 `<<<>>>`/`ACLRT_LAUNCH_KERNEL`
调用点、由 host 侧分支只真正执行其中一个。两种方式都满足"一次 launch"的约束。本文采用前者（复用
`GenKernelEntry.cpp` 已实现的模式）。

`AutofuseLaunchV2` 一词在仓库里没有独立定义，是指上述 `<<<>>>` + `AutofuseLaunch` 这套 bisheng 编译期支持的
调用方案，下文统一用 `AutofuseLaunch` 指代。

`TilingData` 现状是两处独立生成、靠命名约定对齐，不是同一份 schema：

- `codegen::TilingData::Generate`（`codegen/codegen_tiling_data.cpp:161`）在 `ProcessSingleGroup`/
  `ProcessMultiGroup`（`:521`/`:546`）里遍历每个 `schedule_group` 下**全部** `impl_graphs[i]` 的
  `GetAllSizeVar()`（按 size 表达式字符串去重）生成 C++ struct 声明，`impl_graphs[0].GetAllAxis()`
  只是 `:231` 附近 CV 场景下查找单个 `kAxisTypeTileInner` 轴名的窄分支，不是字段生成主逻辑；
- ATT 的 `att/generator/tiling_data_gen/tiling_data_generator.cpp` + `tilingdata_gen_utils.cpp` 按
  `AttAxis`/`ModelInfo::arg_list` 独立推导同名字段，生成调用 `set_X`/`get_X` 的 C++ 源码文本
  （`GenTilingImplAutoFuseV3` 里显式设置 `generator_config.gen_tiling_data = false`，即信任 Codegen 那份已经
  生成了同名 struct）。

`att/gen_model_info/gen_model_info.cpp` 中有明确注释："ATT 必须与 Codegen 使用相同的 schedule result 路径,
否则 raw 1D 节点按 1D 模型评分,但 Codegen 最终生成 kUBFuse 固定 2D 描述"——说明 ATT 对
`FusedScheduledResult` 四层结构和 `cube_type`/`var_relations`/`score_func` 等字段是强耦合、且要求和
Codegen 走同一条遍历路径的。这是"尽量不改 ATT"的具体依据：任何重构都不能让 ATT 和 Codegen 走上不同的
遍历顺序或字段推导逻辑。

### 2.1.2 产品环境介绍

- 依赖 CANN Toolkit 提供的 bisheng 编译器和 `<<<>>>`/`ACLRT_LAUNCH_KERNEL` kernel launch 机制，本文不改变
  这个 ABI；
- 编译驱动依赖 `autofuse/compiler/python/ascendc_compile.py`（直接调用 bisheng，Inductor 现有路径已在用）
  和 `compiler/python/compile_adapter.py`（host/device 两段编译、PCH cache）；
- 不涉及 torchair/inductor-npu-ext 仓库代码修改；本文第 3.2.7 节的 fingerprint 只是给 torchair 侧暴露的
  接口建议，具体接入由 torchair 侧决定；
- `ENABLE_AUTOFUSE_MLIR=OFF` 时不引入 MLIR/LLVM 依赖，行为与当前 legacy 完全一致。

## 2.2 软件功能

1. **（已实现）** 新增统一的 `TilingData` 字段 schema 生成器（非 MLIR 部分，`mlir/common/tiling_schema/`），
   替换"C++ struct 独立生成 + ATT 独立按轴名推导"的双份实现，供 legacy Codegen 和 ATT 共用；MLIR 侧的
   等价产出（`ascendc.tiling_schema`）由已合入 PR#19 的 `GenTilingDataPass` 提供，本文不重复实现，见
   "假设和依赖关系"。
2. **（已实现）** 新增 `AttSchemaAdapter`（`mlir/bridge/AttSchemaAdapter/`）：调用现有 ATT、把结果和统一
   schema 对齐，不改 ATT 核心 API。
3. **（已实现）** 新增 `StaticPlanSelector`（`mlir/bridge/StaticPlanSelector/`）：静态 shape 场景下，在
   生成设备代码之前用 ATT 现有的代数成本模型选出唯一 winner（或 PGO 场景下 Top-N），只编译选中的候选。
4. **（已实现）** 明确动态 shape 和静态 shape 两种场景下的候选筛选策略：动态 shape 保留全部候选（受现有
   结构性剪枝之后的集合）；静态 shape 走 `StaticPlanSelector` 裁剪成单个（或 PGO 场景下 Top-N）候选。
   两种场景下候选如何最终组装成可编译产物仍是开放依赖项，不在本文设计，见"假设和依赖关系"。
5. **（未实现，暂不开发）** MLIR 产物的编译统一走直接调用 bisheng 的驱动方式（复用 `ascendc_compile.py`
   已验证的模式，并接入 CANN 官方文档提到的 `CMAKE_CXX_COMPILER_LAUNCHER`/ccache 加速方式），废弃
   per-kernel CMake configure + make 的路径。
6. **（已实现）** 定义一个可以在 Schedule 运行前计算的结构 fingerprint 接口（`mlir/common/fingerprint/`），
   供 Inductor/torchair 适配层在生成源码之前判断缓存是否命中。
7. **（已实现）** 新增代码统一放在 `autofuse/mlir/` 目录下，legacy 代码（`att/`、`codegen/`）只新增少量
   调用点，不重构既有函数体；本文新增组件均不依赖 MLIR，不链接 LLVM/MLIR（`mlir/common/CMakeLists.txt`、
   `mlir/bridge/legacy/CMakeLists.txt` 已核实只链接 `intf_pub`/`metadef_protos_af`，不引入 LLVM/MLIR）。

## 2.3 设计约束

- ATT 核心 solver（`GetModelInfoMap`、`GenTilingImplAutoFuseV3` 内部算法、评分公式）不改；
- ATT 对外的函数签名不改，新增内容只通过新增的适配组件接入；
- `AutofuseTiling`/`AutofuseLaunch` 导出符号、参数顺序和职责边界不变；
- `<<<blockDim, nullptr, stream>>>` 调用约定不变，一次 fused kernel 仍然只有一次 runtime launch；
- 新增组件（schema 生成、fingerprint、`AttSchemaAdapter`、`StaticPlanSelector`）全部编译成独立的、不链接
  LLVM/MLIR 的 CMake target，供 legacy `att/`、`codegen/` 直接链接，不因此给它们引入 MLIR 依赖；本文不新增
  任何依赖 MLIR 类型构造的组件（原 `TilingSchemaToMlir`/`ScheduleBundle` 已移出本文范围，见"假设和依赖
  关系"）；
- legacy `att/`、`codegen/` 现有函数体不重构，只新增至多几行调用新组件的代码；
- 候选筛选（`StaticPlanSelector`）本身不做笛卡尔积展开；候选最终如何组装编译不在本文约束范围内；
- 图改写和候选编号必须确定，不依赖指针地址或 `unordered_map` 遍历顺序；
- 所有 shape/tiling/workspace/block 数值在使用前检查范围和溢出；
- `node_idx_to_scheduled_results.size() > 1` 第一阶段返回明确 Unsupported 并回退 legacy。

## 2.4 假设和依赖关系

- **本文明确依赖 PR `feat/mlir-tiling-data-header`（#19，已合入 `af-mlir`/本分支，`f63608a2`）的最终
  形态**：该 PR 把 `GenTilingDataPass`/`runGenAggregateTilingData`/`runGenAggregateKernelEntry`/
  `TilingDataHeaderEmitter` 实现完（共 15 个提交，见"项目介绍"），和原功能需求 1 的 `ToPyStructType`、
  原功能需求 2 的 `BuildScheduleBundle` 高度重叠，并已改动 `GenKernelEntry.cpp` 287 行。本文因此：
  (a) 不再自行设计 `ToPyStructType`/`BuildScheduleBundle`，避免与其重复或冲突；(b) 把"生产入口如何组装
  候选、接入 `Codegen::GenerateForInductor`"列为开放依赖项——PR#19 合入后复核确认，`Codegen::
  GenerateForInductor` 仍只调用诊断性的 `MaybeDumpAfirFromSchedule`，PR#19 明确留白的"SR module 与
  Entry module 最终 link/finalize"方案仍未确定，本项因此继续保持开放，待其确定后再设计；(c) `node_id`→
  `subgraph_id` 重命名已覆盖 `AfirBuilder.cpp`、`ScheduleToAfir.cpp/.h`、
  `GenAggregateKernelEntry.cpp` 和 `GenAggregateTilingData.cpp`，legacy 容器名保持不变。本文其余功能需求（`AttSchemaAdapter`、`StaticPlanSelector`、
  动态/静态 shape 候选筛选策略、fingerprint）和 PR#19 无重叠，均已按接口实现，不受其合入节奏影响；
  编译驱动（功能需求 5）未实现，暂不开发；
- CANN 工具链要求设备端分支选择用 `<<<>>>`/`ACLRT_LAUNCH_KERNEL` 编译期确定符号 + 内部 `tiling_key` 跳转，
  不要求为每个候选生成独立可调用的符号；这一点已由现有 `AppendFuncCall`/`GenKernelEntry.cpp` 的实现方式
  和 CANN 官方"Kernel Launch"文档印证，不再单独验证；
- 是否需要为设备编译器生成编译期常量 `TILING_KEY_IS` 分支（而非运行时 `scf.index_switch`/if 分支）待实机
  测量确认，若需要，07-27 文档已预留的 `PlanDispatchSpecializePass` 思路可以兜底，不阻塞本文其余部分；
- **预编译共享 runtime 已确认在公开接口范围内不可行**：CANN 官方文档中的两种公开 kernel 调用方式
  （`<<<>>>` 和 `ACLRT_LAUNCH_KERNEL`）都要求调用点在编译期可见目标 kernel 的声明（`aclrtlaunch_{name}.h`
  是构建时生成、编译期 include 的），不是运行时按 handle/字符串动态解析，这与 CUDA driver API 的
  `cuModuleLoad`+`cuLaunchKernel`（调用方代码不需要重新编译）不同。`编码红线.md` 第 8 条又要求只能用公开
  接口。因此"一份预编译 runtime、launch 任意未来生成的 kernel"在当前公开接口下做不到，本文不再作为待研究
  项，直接排除在范围外。如果 CANN 更底层存在类似 `cuModule` 的公开 handle 式 launch API（未在本次调研的
  文档范围内确认到），可以在后续单独立项调研，但不阻塞本文；
- `StaticPlanSelector` 假设 `att::GetModelInfoMap`（`att/gen_model_info/gen_model_info.h:32`，返回结构化
  `ModelInfo` 而非生成代码文本）可以在不生成/编译 C++ 源码的前提下，直接对返回的代数成本表达式代入常量
  shape 求值。这依赖 ATT 内部 `af::Expression` 是否暴露"给定变量替换表直接求值"的接口（类似 AFIR 侧
  `mlir::afir::Expression::Evaluate`）。本文按此假设设计，**具体接口需要在实现前对照 ATT 源码确认**，如果
  `af::Expression` 不支持直接求值，`StaticPlanSelector` 需要改为调用 `GenTilingImplAutoFuseV3` 生成一段
  极小的求值代码、编译后在 host 侧执行一次，代价比现在设计的"纯内存求值"更高，但仍然远小于编译一次
  device kernel，不影响功能需求 3 的整体结论，只影响其"零编译开销"这一具体量化预期；
- torchair 侧的持久化 artifact cache（`inductor_npu_ext/compiler/__init__.py`）当前以生成源码 MD5 为 key，
  本文第 3.2.7 节的 fingerprint 建议需要 torchair 侧配合才能生效，本文只负责暴露可计算的接口；最终
  artifact key 还必须包含本节前文列出的 target、toolchain、pipeline、schema 和编译选项版本；
- `emitasc::PyStructType`/后端无关 struct-type IR 表示的讨论（AFIR dialect 是否该新增公共抽象、
  PTO-ISA 预留枚举值等）随 `ToPyStructType` 移出本文范围一并移出，相关约束由 PR#19 和
  `2026-08-05-afir-op-capability-architecture-design.md` 自行维护，不在本文重复。

# 3 需求分析与设计

## 3.1 整体介绍

```text
FusedScheduledResult（不变，legacy 结构不改）
  │
  ├─ mlir/common/tiling_schema/（新增，不链接 LLVM/MLIR）
  │    TilingFieldSchema：C++ TilingData struct 声明 + ATT 字段名查询接口
  │    （替换 codegen::TilingData::Generate 和 ATT 独立按轴名推导两处实现）
  │
  ├─ mlir/common/fingerprint/（新增，不链接 LLVM/MLIR）
  │    结构 fingerprint，供 StaticPlanSelector 内部去重和功能需求 6 的前置 cache key 共用同一套算法
  │
  ├─ mlir/bridge/AttSchemaAdapter/（新增，迁移期组件；依赖 ascir + tiling_schema，不需要链接 LLVM/MLIR）
  │    调用现有 att::GenTilingImplAutoFuseV3；ATT 完成 MLIR 化后删除，由 MLIR-native schema/tiling 接口替换
  │
  └─ mlir/bridge/StaticPlanSelector/（新增，迁移期组件；依赖 ascir + tiling_schema，不需要链接 LLVM/MLIR）
       仅静态 shape 场景调用：编译前用 ATT 现有代数成本模型选 winner；ATT 完成 MLIR 化后删除或替换

────────────────────────── 以下为开放依赖项，本文不设计 ──────────────────────────
StaticPlanSelector/SelectDynamicCandidates 产出的 PlanSelection 列表
  → ？？？（生产入口如何把候选组装成可编译产物，`feat/mlir-tiling-data-header`（#19）已落地但未覆盖此步，待另行设计）
      现状候选：PR#19 的 GenTilingDataPass → GenKernelEntryPass → runGenAggregateTilingData →
      runGenAggregateKernelEntry 四段管线（已实现，但 SR/Entry module 最终 link/finalize 未覆盖）
  → AFIRToAscendC pipeline + bisheng 直接编译（编译驱动改为复用 ascendc_compile.py + ccache，本文功能
    需求 5，和上面组装成什么 Module 无关，谁产出 host/device 源码都能接）
  → kernel.so + wrapper.so，AutofuseTiling/AutofuseLaunch 符号不变
──────────────────────────────────────────────────────────────────────────
```

生产 pipeline 的接缝必须和当前 dump 钩子区分开。**当前实现**只允许下列诊断路径：
`ScheduleResultToAfir` 构造 `ModuleOp` → 按需写入 统一 dump 前缀（`ascir::utils::GetDumpFilePrefix()`）；它不会自动调用
`PassManager`，也不会生成 `kernel.so`。**规划路径**固定为：

```text
ScheduleResultToAfir
  → PassManager.run（GenTilingData/GenKernelEntry/AFIRToAscendC 及校验 pass）
  → 可选 AFIR dump
  → 确定性的 aggregate/link/finalize（形成一个可编译的 fused entry）
  → host/device emitter
  → ascendc_compile.py 直接调用 bisheng
  → kernel.so + wrapper.so
```

其中，`MLIRContext`/`ModuleOp` 的所有权由单次 fusion 编译任务持有；候选之间不得共享可变的
`MLIRContext` 或跨线程写同一个 module。聚合阶段必须把被调用函数放入同一个符号表，或通过显式
proxy/custom symbol reference 建立合法引用，不能把一个 module 当作另一个 module 的 `func.call` 符号表。
PR#19 负责的 aggregate pass 与最终 link/finalize 之间的契约尚未冻结，因此具体 pass 名称和 module
合并 API 仍标记为**开放依赖项**；在契约冻结前不得把 dump 结果宣称为可执行产物。

legacy 侧改动：`codegen/codegen_tiling_data.cpp` 新增一次 `mlir/common/tiling_schema` 统一入口调用（诊断性，
不改变原有返回值，见 3.2.1"输出"）；`att/generator/generator_utils/tilingdata_gen_utils.cpp` 不修改——该文件
的工具函数只持有已派生好的字段名字符串，没有 `ScheduledResult` 上下文，接入 schema 校验需要改签名、牵动
`tiling_code_gen_impl.cpp` 内多处调用点，不是"一两行"能做到的；ATT 侧的字段一致性校验改在
`AttSchemaAdapter`（3.2.1"输出"、3.2.3）里对生成产物做文本级核对。`att/` 其余文件、`optimize/` 不改。

## 3.2 功能需求

### 3.2.1 功能需求 1：`TilingFieldSchema` —— 统一 TilingData 字段生成

**实现状态：已实现。** `mlir/common/tiling_schema/tiling_field_schema.h/.cpp` 提供了
`BuildTilingFieldSchema`/`EmitTilingDataStructCpp`/`LookupFieldName`（与"接口设计"节声明一致），并已在
`codegen/codegen_tiling_data.cpp` 的 `TilingData::Generate` 中接入诊断性调用（不一致只记 `GELOGW`，不改变
返回值），配套 UT 见 `mlir/test/ut/schedule/TilingSchemaTest.cpp`。

1. 介绍

   把"Codegen 独立生成 C++ struct"和"ATT 独立按 `AttAxis` 推导字段名文本"合并成一个共享生成器，消除两边
   靠命名约定对齐的隐患。不改变字段名、字段类型、对齐方式和现有 ABI（`block_dim`、`tiling_key`、
   `{axis}_size`/`{axis}_loop_num`/`{axis}_tail_size` 等命名规则原样保留）。

2. 输入

   一个 `ScheduledResult`（及其 `plan_id`/`stage_id`，作为普通函数参数传入，不需要专门的包装类型）。

3. 处理

   - 新增 `TilingFieldSchema`：字段的有序列表，每个字段有 `name`、`type`（纯 C++ 枚举，不是 `mlir::Type`，
     也不引用 `emitasc`/PyAsc 任何类型，保持与 MLIR、AscendC 后端无关）、`path`（对应
     `graph0_result<N>_tiling_data.g<S>_tiling_data.<field>` 这样的层级路径，兼作去重 scope key 和
     `EmitTilingDataStructCpp` 的嵌套 key）。**已实现**（`mlir/common/tiling_schema/tiling_field_schema.h`）
     的 `TilingFieldType` 枚举为 `kUInt32`/`kUInt64`/`kOpaqueStruct` 三值（不含 `int64`，与"输入非负计数"
     的现有字段语义一致），另有一个仅 opaque 字段使用的 `device_type_name`（API tiling 节点的已知设备端
     类型名，未知时留空、按注释占位符输出）；
   - 生成规则从 `codegen::TilingData::Generate` 现有的 `ProcessSingleGroup`/`ProcessMultiGroup` 遍历逻辑
     （对每个 `impl_graph` 的 `GetAllSizeVar()` 按表达式字符串去重 + 类型映射，另含 Tque/Tbuf/API tiling
     字段）原样迁移过来，不改变产出的字段集合；**注意不是**遍历 `impl_graphs[0].GetAllAxis()`（那是
     CV 场景查找单个轴名的窄分支，与字段生成无关，见"项目介绍"）；
   - 提供两个消费方式：
     1. `EmitTilingDataStructCpp(schema)` — 生成 `BEGIN_TILING_DATA_DEF_T/TILING_DATA_FIELD_DEF_T` 风格的
        C++ struct 声明，替换 `codegen::TilingData::Generate` 当前的直接字符串拼接实现（对外行为不变）；
     2. `LookupFieldName(schema, semantic_key)` — 供 ATT 的 `att/generator/tiling_code_gen_impl.cpp` 等
        文件在拼接 `set_X`/`get_X` 调用文本时查询字段名，替换目前从 `AttAxis` 独立重新推导命名的逻辑；
   - **不提供** MLIR 转换方法（原第三种消费方式 `ToPyStructType`）：PR#19 的 `GenTilingDataPass` 已经从
     AFIR IR 直接推导出等价的 `ascendc.tiling_schema`（`emitasc::PyStructType`），本文不重复实现，也不
     依赖它，`TilingFieldSchema` 和 PR#19 的 MLIR schema 是两条独立的生成路径，服务不同消费方（前者服务
     ATT/legacy codegen，后者服务 MLIR pipeline），二者字段集合语义上应一致但本文不强制保证工程上共用
     同一份代码——这是本文和 PR#19 之间明确的边界，避免互相依赖导致谁先合入都要等对方；
   - 这是机械重构：把"从图轴推导字段名"这段逻辑收敛到一处，不改 ATT 的评分/求解算法，不改 Codegen 生成的
     struct 布局；实现前需对照 `ProcessSingleGroup`/`ProcessMultiGroup` 完整迁移去重和字段排序细节，不能
     只按"轴遍历"简化理解。
   - Codegen 侧和 ATT 侧接入统一入口的方式不同，原因是两侧在"是否拥有 `ScheduledResult`/
     `FusedScheduledResult` 上下文"这件事上不对称（细节见"输出"）：Codegen 侧 `TilingData::Generate` 本身
     就拿着 `FusedScheduledResult`，可以直接调用 `BuildTilingFieldSchema`；ATT 侧字段名是在
     `tiling_code_gen_impl.cpp` 内部逐步派生后才传入 `tilingdata_gen_utils.cpp` 的工具函数（如
     `WriteTilingDataElement`），到那一层时手里只有已经算好的字符串集合，没有 `ScheduledResult`，要在那里
     接入 schema 校验就得给这些工具函数改签名、牵动 `tiling_code_gen_impl.cpp` 内多处调用点——这已经不是
     "一两行、其余代码不动"，在当前没有可用编译环境验证的前提下改这么深风险太高。因此 ATT 侧改为在
     `AttSchemaAdapter`（功能需求 2）这个已经同时拿到 schema 和 ATT 生成产物的边界上做交叉校验，校验的是
     ATT 最终生成的源码文本里实际引用了哪些字段，而不是拦截某个内部生成步骤——覆盖面不小于内部 hook，
     且不需要改动 ATT 内部任何调用签名。

4. 输出

   放置位置：`autofuse/mlir/common/tiling_schema/`（不依赖 MLIR，独立 CMake target）。当前实现是 ATT/legacy
   Codegen 迁移期间的过渡层，`BuildTilingFieldSchema` 提供两个重载：一个接收 `FusedScheduledResult`
   （fusion 级输入，workspace 字段和图级 tiling key 是 fusion 级数据，不是单个 impl graph 的属性）；另一个
   接收 `ScheduledResult`/`subgraph_id`/`plan_id`（服务已经按 `ScheduledResult` 遍历的 legacy 调用点）。
   这不是最终 MLIR schema 的输入边界。ATT 开始迁移时，应把字段模型与 legacy schedule 遍历拆开，改为
   AFIR 或后端无关 DTO 输入；在此之前不提前重构这条边界。
   - Codegen 侧：`codegen/codegen_tiling_data.cpp` 的 `TilingData::Generate` 在生成完
     `elemwise_schedule_result` 之后新增一次 `BuildTilingFieldSchema` 调用，失败只记 `GELOGW`，不改变
     `Generate` 原有返回值，即"新增一两行调用统一入口，其余代码不动"按字面实现。
   - ATT 侧：不修改 `att/generator/generator_utils/tilingdata_gen_utils.cpp`（原计划的接入点，见上面
     "处理"一节的说明）。字段名一致性校验改为在 `AttSchemaAdapter::RunAttForCandidates`
     （功能需求 2，`mlir/bridge/AttSchemaAdapter/AttSchemaAdapter.cpp`）里实现：扫描
     `att::GenTilingImplAutoFuseV3` 生成的 `tiling_func` 源码文本中的 `get_X`/`set_X`/`get_addr_X` 访问器
     调用，逐个比对 `TilingFieldSchema`，任一字段缺失即判定失败、不产出 `TilingCodeBundle`。

### 3.2.2 开放依赖项（原功能需求 2，非本文详细设计范围）：生产入口如何组装候选

**实现状态：仍未解决（PR#19 合入后复核确认）。** `FusedScheduledResult` → 可编译 AscendC 产物这条链路，
"多个候选怎么组装成一个 Module、怎么接入 `Codegen::GenerateForInductor`"这一步，本文不设计具体实现，
理由和现状见"假设和依赖关系"：PR#19 `feat/mlir-tiling-data-header` 已合入，实现了 `GenTilingDataPass →
GenKernelEntryPass → runGenAggregateTilingData → runGenAggregateKernelEntry` 四段管线，但没有覆盖"SR
module 与 Entry module 最终 link/finalize"（PR#19 自己的设计文档明确排除；合入后复核，
`runGenAggregateKernelEntry` 仍只生成私有前向声明，未做符号表合并），也没有接入生产入口——
`Codegen::GenerateForInductor` 目前仍只调用诊断性的 `MaybeDumpAfirFromSchedule`，不触发
`PassManager`，不产出 `.so`（同样明确排除"默认 pipeline 修改"）。因此：

- 本文功能需求 3（`StaticPlanSelector`）/功能需求 4（动态静态候选策略）产出的 `PlanSelection` 列表，
  消费方是谁、以什么接口消费，留空，作为后续接线任务的输入契约，不在本文提前绑定；
- `ScheduleToAfir` 是否需要 `subgraph_id` 重命名、以及提升为生产入口的具体方式，仍待另行设计（覆盖文件
  范围已在 PR#19 落地后重新核实，见"假设和依赖关系"）；
- PR#19 已合入，其设计文档明确排除的"SR/Entry module 最终 link/finalize"仍未由任何后续改动补上；本文
  （或新开一份设计文档）需要再补上这一段的详细设计，包括：候选组装成何种 Module 组织形式、
  `GenKernelEntry.cpp` 变化后的输入契约、SR/Entry module 最终 link/finalize 方案、ATT 生成的 tiling
  源码如何接入。

### 3.2.3 功能需求 2：`AttSchemaAdapter` —— ATT 输出与统一 Schema 的对接

**实现状态：已实现。** `mlir/bridge/AttSchemaAdapter/AttSchemaAdapter.h/.cpp` 提供 `RunAttForCandidates`，
内部调用 `att::GenTilingImplAutoFuseV3` 并做 `get_X`/`set_X` 字段引用扫描校验，配套 UT 见
`mlir/test/ut/schedule/AttSchemaAdapterTest.cpp`。

1. 介绍

   新增一个薄适配组件，职责是"调用现有 ATT、把结果和统一 schema 对齐"，不承担 ATT 的算法职责，不要求 ATT
   输出 MLIR，不依赖 MLIR（只依赖 `ascir::` 和功能需求 1 的 `TilingFieldSchema`）。

   适配结果分成三个层次，不能把它们合并成一个“ATT 结果”概念：

   - **host tiling program**：ATT 生成的 `GetTiling`/模型求值 host C++ 源码，负责读取运行时 shape、计算
     `tiling_key`、`block_dim`、workspace 和 tiling 字段；
   - **runtime tiling result**：一次 `AutofuseTiling` 调用写入的 `TilingData` 实例和选择结果，生命周期由
     当前 wrapper/调用方管理；
   - **device dispatch contract**：MLIR/AscendC entry 消费的 `(plan_id, stage_id, template_id)` 和字段 schema，
     只描述如何分派，不在 host adapter 中执行 device 选择。

2. 输入

   `FusedScheduledResult`、候选集合（`PlanSelection` 列表，来自功能需求 4 的动态候选或功能需求 3 的
   静态 winner）、对应的 `TilingFieldSchema`、`AttOptions`。

3. 处理

   - 调用现有 `att::GenTilingImplAutoFuseV3(op_name, fused_schedule_result, options, tiling_func,
     is_inductor)`，签名和内部实现不变；
   - 生成的 `tiling_func`（C++ 源码文本 map）中引用的字段名，由功能需求 1 的 `TilingFieldSchema::
     LookupFieldName` 提供，保证 ATT 侧和 legacy Codegen 侧引用同一份字段名来源；MLIR 侧
     `ascendc.tiling_schema`（PR#19 产出）字段是否与此一致，属于"开放依赖项"里提到的对接工作，本文
     只保证非 MLIR 的两个消费方（ATT、legacy codegen）一致；
   - ATT 生成的 tiling C++ 源码仍按现状单独编译（host 阶段），不要求编译进 MLIR module；动态 shape 时
     保留 host program 和 runtime `tiling_key` 选择，静态 shape 时 selector 只在编译前裁剪候选，不能把
     host program 错当成已经执行的 runtime result；
   - `TilingCodeBundle` 必须携带 schema 校验结果和候选坐标，供下游在生成 device entry 前校验
     `(subgraph_id, plan_id, stage_id, template_id)` 是否完整。

4. 输出

   `TilingCodeBundle`：ATT 生成的 host tiling C++ 源码 map（沿用现状）、候选坐标到字段 schema 的映射，
   以及一份和 `TilingFieldSchema` 的字段一致性校验结果。它不包含 device 端运行时已选的 tiling result；
   后者仍由 `AutofuseTiling` 在执行时产生。

### 3.2.4 功能需求 3：`StaticPlanSelector` —— 静态 shape 编译前选 winner

**实现状态：已实现。** `mlir/bridge/StaticPlanSelector/StaticPlanSelector.h/.cpp` 提供 `SelectStaticWinner`，
内部调用 `att::GetModelInfoMap` 对代数成本表达式求值选 winner，`PlanSelection` 结构与本节输出一致，配套
UT 见 `mlir/test/ut/schedule/PlanSelectorTest.cpp`。

1. 介绍

   **只在静态 shape 场景下调用**。Inductor 默认按具体 shape 特化，此时这个编译产物这辈子只服务一个已知
   shape，运行时 dispatch 是纯浪费。`StaticPlanSelector` 把 ATT 现有的代数成本评分逻辑从"运行时对每次
   调用求值一遍"提前搬到"编译期对已知常量 shape 求值一次"，只把选中的 winner 交给后续候选组装和设备
   编译（具体组装方式是开放依赖项，见"假设和依赖关系"），其余候选和分支不生成、不编译。这是全新能力，
   legacy 没有对应实现（legacy 不区分静态/动态 shape，
   一律把全部候选编译进同一个 kernel），但**复用**的是 ATT 已有的评分算法/表达式求值代码，不是新写一套
   算法。

2. 输入

   `FusedScheduledResult` 的候选集合（一个 subgraph 下所有 plan 及其 stage/template）、已知的常量
   `ConcreteShape`、`AttOptions`。

3. 处理

   - 调用 `att::GetModelInfoMap`（不生成代码文本，返回结构化 `ModelInfo`，含各候选的代数成本表达式）；
   - 对返回的成本表达式代入已知常量 shape 求值（host 侧纯算术，不涉及 device 编译），比较各候选，选出
     唯一 winner `(plan_id, [template_id...])`；此步骤的具体求值接口依赖"假设和依赖关系"节标注的待确认
     项，若 `af::Expression` 不支持直接求值，退化为编译一段极小求值代码执行一次，代价仍远小于编译一次
     device kernel；
   - 需要 PGO 实测复核时，允许选 Top-N（而不是 1 个）候选参与后续设备编译和实测，实测逻辑复用现有 PGO
     callback 机制，选中 winner 后其余候选和分支一起被裁剪，不进入最终产物；
   - 所有候选都不合法（无法满足资源约束）时返回失败，上层回退 legacy。

4. 输出

   ```cpp
   struct PlanSelection {
     size_t subgraph_id;
     size_t plan_id;
     std::vector<std::vector<size_t>> template_ids_per_stage;
   };

   bool SelectStaticWinner(const ascir::FusedScheduledResult &fused_result, const ConcreteShape &shape,
                           const AttOptions &options, std::vector<PlanSelection> &selections,
                           std::string *diagnostic = nullptr);
   ```

   已实现签名用 `bool` 返回值 + 输出参数 + 可选 `diagnostic`（而非 `af::StatusOr`），与"接口设计"5.1 节
   声明一致，见 `mlir/bridge/StaticPlanSelector/StaticPlanSelector.h`。

   `template_ids_per_stage[stage_id]` 的下标就是 `stage_id`；缺少某个 stage、出现重复或越界 template
   时，`PlanSelection` 无效，不能由下游猜测默认值。

   放置位置：`mlir/bridge/StaticPlanSelector/`。

### 3.2.5 功能需求 4：动态 shape 与静态 shape 的候选筛选策略

**实现状态：已实现。** `SelectDynamicCandidates` 与 `SelectStaticWinner` 同在
`mlir/bridge/StaticPlanSelector/StaticPlanSelector.h/.cpp`，配套 UT 见
`mlir/test/ut/schedule/PlanSelectorTest.cpp`。

1. 介绍

   明确"多少候选应该进入 device 编译"这件事在两种场景下的不同处理，只产出候选集合
   （`PlanSelection` 列表），不涉及这些候选最终如何组装编译（见"开放依赖项"）。两条路径的差异只是候选
   子集大小和是否调用功能需求 3。

2. 输入

   全部候选（一个 subgraph 下所有 plan 及其 stage/template），以及 shape 是否为 Inductor 动态 shape 的
   标记（由 Inductor adapter 传入，Autofuse 不自行判断）。

3. 处理

   **动态 shape**（shape 符号未在编译期解析为常量，或 Inductor `mark_dynamic`）：

   - 这个编译产物要服务一批将来才出现的具体 shape，编译期不知道该选哪个候选，不调用
     `StaticPlanSelector`；
   - 保留全部候选（受现有 `StaticUbTemplateFilter`/`FilterComplexTilingDataScoreFuncs` 结构性剪枝之后的
     集合），预期由下游（见"开放依赖项"）一次编译覆盖，运行时由 ATT 生成的 `AutofuseTiling` 通过
     `tiling_key` 在 device 端内部选择分支；
   - 需要澄清一个曾经写错的判断：legacy `AppendFuncCall`（`codegen_kernel.cpp:3540-3564`）确实按全部
     stage/template 组合展开成扁平 `TILING_KEY_IS` 分支链（分支数是各 stage template 数量之积），但每个分支
     只是若干条调用已生成函数的语句，不是重新展开函数体；实际的 AscendC 计算代码本身
     （`kernel.Generate(...)`，`codegen_kernel.cpp:2735-2737`）在 legacy 里就已经是每个 `(stage, template)`
     只生成一次（数量之和）。分发胶水代码的组织形式（扁平分支链 vs `scf.index_switch` 嵌套）只影响文本
     量，预期不是编译耗时的主要来源（编译耗时大头预期在 O(Σ) 的计算代码本身）；真正影响编译耗时的是
     本节后半段（静态 shape 裁剪候选数）和功能需求 5（编译驱动）。

   **静态 shape**（Inductor 默认按具体 shape 特化，编译产物只服务这一个已知 shape）：

   - 调用功能需求 3 的 `StaticPlanSelector`，得到唯一 winner（或 PGO 场景下 Top-N）；
   - 只保留这个（些）`PlanSelection`，其余候选和分支不进入下游编译；
   - 效果：静态 shape 场景下的 device 编译规模从"全部候选之和"降到"1 个（或 PGO 场景下 Top-N 个）"，
     直接减少 bisheng 编译时间和产物体积；这是本文对"编译耗时长"的主要修复点之一，另一个是功能需求 5。

4. 输出

   ```cpp
   bool SelectDynamicCandidates(const ascir::FusedScheduledResult &fused_result,
                                std::vector<PlanSelection> &selections, std::string *diagnostic = nullptr);
   ```

   已实现签名同样用 `bool` 返回值 + 输出参数（而非 `af::StatusOr`），见
   `mlir/bridge/StaticPlanSelector/StaticPlanSelector.h`。

   （`SelectStaticWinner` 见功能需求 3。）两者的输出都是 `std::vector<PlanSelection>`，下游消费方待
   "开放依赖项"确定，下游代码不需要区分调用方走的是哪条路径。

### 3.2.6 功能需求 5：编译驱动改为直接调用 bisheng（需要修改旧代码，暂时先不做）

**实现状态：未实现，暂不开发。** `compiler/python/asc_codegen_compile.py` 仍是每次 `cmake -S . -B .` +
`make`、主动清空 `CMakeCache.txt`/`CMakeFiles` 的路径，`mlir/` 下未新增任何直接调用 bisheng 的编译驱动。
本节内容保留为后续任务，验收范围不含本项，见"验收标准"。

1. 介绍

   MLIR Codegen 产出的 host/device 源码统一走 Inductor 现有路径已验证的直接编译器调用方式，不再使用
   `asc_codegen_compile.py` 那条每次 `cmake -S . -B .` + `make` 的路径（该路径会主动清空
   `CMakeCache.txt`/`CMakeFiles` 强制每次重新 configure，`asc_codegen_compile.py:1202-1209`）。

2. 输入

   MLIR Codegen 生成的 host tiling 源码、device AscendC 源码。

3. 处理

   - 复用 `autofuse/compiler/python/compile_adapter.py` 的 `host_compile`/`kernel_compile` 和
     `ascendc_compile.py` 的直接 bisheng 调用（`build_host_compile_cmd`/`compile_device_obj`/
     `link_shared`），不新增第二套编译驱动；
   - 继续复用已有的 host PCH cache（`get_host_pch_key`/`global_pch_build`）和 CV wrapper 共享 cache
     （`get_shared_cv_wrapper_so_path`），这两个是已验证的减少重复编译开销的机制；
   - 对仍经 CMake 调度的 host 编译步骤使用 CMake 标准机制 `CMAKE_CXX_COMPILER_LAUNCHER=ccache`；
     对直接调用 bisheng 的步骤使用等价的 ccache compiler wrapper。两条路径都必须在 ccache 不可用时回退
     到原始编译器，不把 ccache 误当成必需依赖；
   - 不修改 legacy TBE 单算子路径（`asc_codegen_compile.py`），只是 MLIR 产物不再新增对它的依赖。

4. 输出

   与 legacy 一致的 `kernel.so`/`wrapper.so`（或 `tiling.so`，命名沿用现状），编译耗时去掉 CMake
   configure/生成 CMakeLists 的固定开销，并叠加 ccache 命中收益。

### 3.2.7 功能需求 6：结构 Fingerprint 前置（接口级）

**实现状态：已实现（接口级）。** `mlir/common/fingerprint/schedule_fingerprint.h/.cpp` 提供
`ComputeScheduleFingerprint(FusedScheduledResult)`/`BuildArtifactCacheKey`，配套 UT 见
`mlir/test/ut/schedule/FingerprintTest.cpp`；torchair 侧接入本文不承诺，见"处理"一节。

1. 介绍

   当前 torchair `inductor_npu_ext` 的 binary cache 以生成源码全文 MD5 为 key（`compiler/__init__.py`），
   即使 cache 命中，也无法跳过 Schedule 和 Codegen 的源码生成过程（08-12 文档 6.1 节已有源码走查依据：
   `NPUKernel.tracing_asc()`/`Autofuser.schedule()`/`Codegen::GenerateForInductor()` 均在主进程同步执行，
   完整 artifacts 生成后才提交异步编译，尚无实测数据，验收仍以性能章节的实测结果为准）。本节
   给出 Autofuse 侧可以提供、供 torchair 侧选择性接入的前置 fingerprint 接口，不修改 torchair 仓库代码。
   fingerprint 算法复用 `mlir/common/fingerprint/`（和 `StaticPlanSelector` 内部去重用的是同一套实现）。

2. 输入

   Inductor fusion 对应的图结构（拓扑、shape 符号等价关系、target profile、编译选项），不含具体 shape
   数值。理想情况下输入是 Schedule 运行前的 `FusionGraph`；第一阶段如果 Schedule 尚不能提供稳定的
   pre-Schedule 结构快照，允许退化为对 `FusedScheduledResult`（Schedule 输出）计算 fingerprint，此时
   只能跳过 Codegen 阶段，跳不过 Schedule 阶段，需要在实现时如实标注这个退化，不能对外宣称已跳过
   Schedule。

3. 处理

   - 暴露一个可以尽量早计算的结构 fingerprint 函数；
   - fingerprint 的输入不含运行时才知道的具体 shape 数值，使同一结构、不同 shape 的多次调用得到相同
     fingerprint（配合功能需求 4 的动态 shape 路径：同一 fingerprint 只需编译一次）；
   - 这是一个接口建议，Autofuse 侧负责实现计算逻辑，torchair 侧决定是否、以及何时调用它来提前判断
     cache 是否命中，本设计不承诺 torchair 侧的改动。

   fingerprint 只是结构部分；最终 artifact cache 必须把它与以下版本化字段拼成有版本的 key，不能只使用
   图结构哈希：`target_profile_version`、CANN/bisheng 工具链版本、bridge/AFIR pass pipeline 版本、ATT
   schema/策略版本、Codegen/emitter 版本、编译选项、静态 shape 的常量集合、动态/静态 shape 模式，以及
   `TilingFieldSchema`/Entry ABI schema 版本。任一字段缺失或无法稳定序列化时，缓存视为未命中并重新生成。

4. 输出

   一个稳定、可序列化、不含指针的结构 fingerprint 值，可用作上述版本化 artifact cache key 的结构部分；
   返回值不承诺单独足以复用 `kernel.so`。

### 3.2.8 后续任务（非本次必须，PR#19 已落地，总数已确认但具体行号待启动本任务时重新核实）：收敛 `emitasc::PyStructType` 重复校验

`emitasc::PyStructType` + `getNameAttr() == "TilingData"` 的裸校验在 `AFIRToAscendC/` 下同样的 5 个文件里
重复实现，行号基于 `GenKernelEntry.cpp` 改动前的版本统计为 8 处（`AscendCKernelABI.cpp:76-77,112-115`、
`GenKernelEntry.cpp:44-55,63-64,141,172`、`AFIRKernelBodySignature.cpp:97,135,313`、
`AFIRMemoryAllocate.cpp:322,551`、`AFIRLoopGenerate.cpp:218`）。**PR#19 合入后复核**：`GenKernelEntry.cpp`
改动 287 行后，5 个文件里的裸校验合计增至 17 处，具体行号需要在启动这项后续任务时重新走查，`GenKernelEntry.cpp`
的 `resolveField` 也独立实现了一遍嵌套字段导航。建议在 `AscendCKernelABI.h/.cpp`
新增校验过的包装类型 `mlir::afir::TilingDataType`（`get(Type)` 做唯一一次 `dyn_cast` + 名字校验，
`resolveField(StringRef path)` 收敛字段导航逻辑），5 处调用点改为使用它，属于不改变 pass 输出语义的纯内部
重构。与功能需求 1 的关系：`TilingFieldSchema` 管字段"从哪来"（服务 ATT/legacy codegen，生成期），
`TilingDataType` 管"IR 里已有这个类型时怎么安全地读它"（消费期，MLIR 侧），两者不重叠，且都不依赖
`ToPyStructType`——`TilingDataType::get` 消费的是 PR#19 `GenTilingDataPass` 产出的
`ascendc.tiling_schema`，与本文的 `TilingFieldSchema` 是两条独立路径。
这是同一后端内部的重复代码问题，与"假设和依赖关系"中讨论的 AFIR 后端解耦问题相互独立，不阻塞功能需求
1-6，可作为独立后续任务排期，不在本文验收范围内。

## 3.3 非功能需求

### 3.3.1 可维护性

- 新增内容集中在 `autofuse/mlir/common/`、`autofuse/mlir/bridge/` 下的新子目录，不改动 `optimize/`
  和 ATT 核心算法文件；不涉及 `GenKernelEntry.cpp`（该文件已按 PR#19 的设计改动，但其后续 SR/Entry
  module link/finalize 走向仍未确定，见开放依赖项）；
- `TilingFieldSchema` 是纯函数生成器，不持有需要显式生命周期管理的资源；
- 依赖方向单向：legacy `codegen/` 新增一次诊断性调用指向 `mlir/common/`；legacy `att/` 不新增任何调用，
  ATT 侧的字段一致性校验在 `mlir/bridge/AttSchemaAdapter/` 里对 ATT 产物做只读核对；`mlir/` 内部组件不
  反向修改 legacy 代码逻辑，只读取（`ascir::`、`att::GenTilingImplAutoFuseV3`/`GetModelInfoMap`）。

### 3.3.2 可测试性

- `TilingFieldSchema`：UT 对比新生成器和 `codegen::TilingData::Generate`/ATT 现有推导逻辑在相同输入下的
  字段名、类型、顺序完全一致（回归基线用现有测试夹具，如
  `tests/ut/optimize/test_buf_que_allocator.cpp` 的 `MakeFusedScheduledResultWithGraphs`）；
- `StaticPlanSelector`：UT 用固定评分/固定 shape 验证选出的 winner 和现状（PGO 或 model-score 选择
  逻辑）一致；
- 编译驱动切换：ST 验证走新驱动产出的 `kernel.so`/`wrapper.so` 导出符号、tiling 行为和 legacy 编译产物
  一致；ccache 命中/未命中两种场景都需要验证正确性（不只是验证性能）；
- 本文各组件均可独立于"开放依赖项"单独做 UT（`TilingFieldSchema`/`AttSchemaAdapter`/
  `StaticPlanSelector`/候选筛选策略只依赖 `FusedScheduledResult`，编译驱动只依赖已生成的 host/device
  源码），不需要等生产入口组装方案确定；`TilingFieldSchema`/`AttSchemaAdapter`/`StaticPlanSelector`/候选
  筛选策略的 UT 已落地（`mlir/test/ut/schedule/TilingSchemaTest.cpp`、`AttSchemaAdapterTest.cpp`、
  `PlanSelectorTest.cpp`、`FingerprintTest.cpp`），编译驱动的 UT/ST 待功能需求 5 启动后再补。
- 本设计不新增候选级并行编译协议；在最终 aggregate/link 契约冻结前，单个 fusion 的 MLIR module 构造和
  pass 运行按任务串行。跨 fusion 的异步编译继续复用 Inductor 现有 artifact process pool，避免共享
  `MLIRContext` 或未定义的跨 module 符号引用。

### 3.3.3 可移植性

- target 能力（UB/core/对齐等）通过现有 Target Profile 接口获取，不硬编码芯片型号；
- `ENABLE_AUTOFUSE_MLIR=OFF` 时不引入本文新增代码路径的 MLIR 依赖（本文新增组件均不链接 MLIR）；
- fingerprint 接口不假设特定 torchair 版本，只是一个可选调用点。

### 3.3.4 可靠性

- `SelectStaticWinner` 在输入非法（空层级、越界）时返回明确错误，不生成半成品 tiling 源码；
- 静态 shape winner 选择失败（所有候选都不合法）时返回错误，回退 legacy，不生成半合法产物；
- 新编译驱动失败时清理临时目录，不留下可被误加载的半成品 `.so`。

### 3.3.5 特性交叉影响

| 场景 | 适用性 | 分析说明 |
|---|---|---|
| SuperKernel Python 接口 | 不适用 | 不修改 `super_kernel/` 任何代码、选项解析或 wheel 内容 |
| SuperKernel C++/AOT 接口 | 不适用 | 不修改 `libascendsk.so` 或 SuperKernel AOT ABI |
| Autofuse 图优化 | 适用 | 不改 `optimize/` 的图改写逻辑；只把 AFIR bridge 和 aggregate contract 的 `node_id` 语义命名统一为 `subgraph_id`，legacy 容器名保持不变 |
| Autofuse Codegen/Backend | 适用 | 核心范围：`TilingFieldSchema` 替换 `codegen_tiling_data.cpp` 内部实现、ATT 适配、候选筛选策略、编译驱动切换 |
| AscendC API/Runtime | 适用 | `<<<>>>`/`AutofuseTiling`/`AutofuseLaunch` 调用约定和参数顺序不变，需 ST 验证 |
| Python/C++ 混合绑定 | 适用 | `compiler/python/compile_adapter.py` 复用现状，需验证异常传播和 GIL 边界不变 |
| 构建与打包 | 适用 | 新增目录需要接入 CMake，MLIR 依赖仍受 `ENABLE_AUTOFUSE_MLIR` 开关隔离，需验证 legacy 链接的轻量 target 确实不拉入 LLVM；不新增第三方下载依赖，不影响增量构建、离线构建和 `--no-autofuse` 场景，见"安全检查"节 |
| 测试与覆盖率 | 适用 | 见"可测试性"节 |
| 性能与日志 | 适用 | 见"性能"节；新增组件不引入默认开启的高频日志 |
| 兼容性 | 适用 | 对外 `CodegenResult`、`AutofuseTiling`/`AutofuseLaunch` 符号、`kernel.so`/`wrapper.so` 产物边界不变 |

# 4 性能

## 4.1 编译时长

需要分别测量并对比 legacy 基线：

- `TilingFieldSchema` 构建耗时（预期是对已有数据的一次遍历，量级远小于 Schedule 本身）；
- 动态 shape 场景下候选分发胶水代码的耗时对比（`scf.index_switch` 嵌套 vs legacy `AppendFuncCall` 扁平
  展开）：如"项目介绍"节所述，两者实际编译的 AscendC 计算代码量相同（均为 O(Σ)），差异只在分发胶水代码
  的文本量，预期是编译耗时的次要因素；具体由谁生成 `scf.index_switch`（PR#19 的管线或后续方案）属于
  开放依赖项，测量目的是确认"胶水代码差异不是主要因素"这个预期，不预判具体实现；
- `StaticPlanSelector` 的耗时（预期是纯算术求值，量级远小于一次 device 编译；如"假设和依赖关系"节所述，
  若退化为编译极小求值代码，需要单独测量这条退化路径的实际耗时），以及裁剪到单候选后的 bisheng 编译耗时
  相比"编译全部候选"的下降幅度；
- 编译驱动从 CMake 切到直接 bisheng 调用、叠加 ccache 后，单 kernel 编译耗时的下降幅度（冷 cache 和热
  cache 分别测量）；
- 不承诺具体数字，以上均为待实测项，验收以实测数据为准，不以设计推测替代。

## 4.2 执行性能

- 动态 shape 路径的运行时行为和现状等价：一次 `AutofuseTiling` + 一次 `AutofuseLaunch`，device 端
  `scf.index_switch` 分支选择开销预期与现有 `tiling_key` 分支选择相当，需要 A/B 测量确认，不得引入
  额外 launch；
- 静态 shape 路径裁剪掉未选中分支后，device 端执行路径比动态 shape 路径更短（少了 dispatch 判断），
  预期执行开销不高于现状，需要实测验证；
- 不改变 `<<<>>>` 调用约定和 current stream 语义，不引入隐式同步。

## 4.3 内存和产物大小

- 静态 shape 场景下产物体积因为只保留 winner 分支而下降；动态 shape 场景下产物体积与 legacy 全候选编译
  同阶（`scf.index_switch` 相比笛卡尔积展开体积更小，但仍随候选数增长）；
- ccache 会新增本地缓存目录的磁盘占用，需要评估容量治理策略（沿用现有 PCH cache 目录管理方式）；
- 编译过程中的临时目录、CMakeCache 等文件按现有规则位于已忽略目录。

# 5 接口设计

## 5.1 新增/修改接口描述

功能需求 1、2、3、4、6 的声明已按下述签名实现，与当前 checkout 的公共 ABI 一致（见各函数后的实现文件
路径）；只有 `FusionGraphSnapshot`、`ArtifactVersionSet` 和 `CompileOptions` 仍是规划接口，尚未实现，需要在
实现任务中先冻结序列化字段，再决定是否暴露给 torchair。

```cpp
// mlir/common/tiling_schema/tiling_field_schema.h —— 不依赖 MLIR，已实现
bool BuildTilingFieldSchema(const ascir::FusedScheduledResult &fused_result, TilingFieldSchema &schema,
                            std::string *diagnostic = nullptr);
bool BuildTilingFieldSchema(const ascir::ScheduledResult &scheduled_result, size_t subgraph_id, size_t plan_id,
                            TilingFieldSchema &schema, std::string *diagnostic = nullptr);
std::string EmitTilingDataStructCpp(const TilingFieldSchema &schema,
                                    std::string_view struct_name = "AutofuseTilingData");
std::optional<std::string> LookupFieldName(const TilingFieldSchema &schema, std::string_view semantic_key);

// mlir/common/fingerprint/schedule_fingerprint.h —— 不依赖 MLIR，已实现
ScheduleFingerprint ComputeScheduleFingerprint(const ascir::FusedScheduledResult &fused_result);
// 规划接口：有稳定 pre-Schedule 快照后使用，尚未实现
ScheduleFingerprint ComputeScheduleFingerprint(const FusionGraphSnapshot &snapshot);
ArtifactCacheKey BuildArtifactCacheKey(const ScheduleFingerprint &structure,
                                        const ArtifactVersionSet &versions,
                                        const CompileOptions &options);  // 已实现

// mlir/bridge/AttSchemaAdapter —— 依赖 ascir + tiling_schema，调用现有 att::GenTilingImplAutoFuseV3，不改其签名，已实现
af::Status RunAttForCandidates(const std::string &op_name, const ascir::FusedScheduledResult &fused_result,
                                const std::vector<PlanSelection> &selections, const TilingFieldSchema &schema,
                                AttOptions options, bool is_inductor_scene, TilingCodeBundle &out,
                                std::string *diagnostic = nullptr);

// mlir/bridge/StaticPlanSelector —— 依赖 ascir，调用现有 att::GetModelInfoMap，不改其签名，已实现
bool SelectDynamicCandidates(const ascir::FusedScheduledResult &fused_result,
                             std::vector<PlanSelection> &selections, std::string *diagnostic = nullptr);
bool SelectStaticWinner(const ascir::FusedScheduledResult &fused_result, const ConcreteShape &shape,
                        const AttOptions &options, std::vector<PlanSelection> &selections,
                        std::string *diagnostic = nullptr);
```

已实现签名相比本文早期草案有若干细节调整（不改变行为意图）：`RunAttForCandidates` 额外带
`op_name`/`is_inductor_scene`/`diagnostic` 参数、`AttOptions` 按值传递；`LookupFieldName` 第二个参数是
`std::string_view semantic_key`，不是专门的 `SemanticKey` 类型；`EmitTilingDataStructCpp` 多一个带默认值的
`struct_name` 参数。均为签名细化，不影响本文描述的行为契约。

调用顺序（本文范围内）：`SelectDynamicCandidates`/`SelectStaticWinner`（按 shape 模式二选一）→
`BuildTilingFieldSchema` → `RunAttForCandidates` → **开放依赖项**（`PlanSelection` 列表 + `TilingFieldSchema`
如何组装成可编译产物，PR#19 已合入但未覆盖此步，待另行设计）→ AscendC translation → 功能需求 5 的编译驱动
（未实现，暂不开发）。任一步失败，
整个 fusion region 回退 legacy Codegen，不生成半成品产物。

`ComputeScheduleFingerprint(FusedScheduledResult)` 是第一阶段的退化入口，只能跳过 Codegen；只有
`FusionGraphSnapshot` 在 Schedule 前稳定可序列化时，才允许宣称可以跳过 Schedule。`ArtifactCacheKey` 必须由
结构 fingerprint、版本集合和编译选项共同构成，不能把结构 fingerprint 单独当作动态库缓存 key。

## 5.2 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|---|---|---|---|
| 接口说明 | 是否需要接口评审 | 是 | 新增 schema/`AttSchemaAdapter`/`StaticPlanSelector` 边界需要评审确认字段和错误语义 |
| 接口说明 | 是否需要补充文档 | 是 | 需要说明 `mlir/common/` 下"目录在 mlir/ 但不链接 LLVM"的依赖边界，避免后续误加 MLIR 依赖 |
| 接口兼容 | 行为是否兼容 | 是 | legacy 路径行为不变；本文组件产出的 `PlanSelection`/`TilingFieldSchema` 是否兼容下游（开放依赖项）待其设计确定后复核 |
| 接口兼容 | ABI/API 是否兼容 | 是 | `AutofuseTiling`/`AutofuseLaunch`/`CodegenResult` 不变；`att::GenTilingImplAutoFuseV3`/`GetModelInfoMap` 签名不变 |
| 接口约束 | 约束不满足时是否清晰报错 | 是 | 每个新增函数都有失败返回值，不吞错误 |
| 接口测试 | 是否需要独立接口用例 | 是 | 见"测试设计" |

# 6 软件设计

## 6.1 关键数据结构

- `TilingFieldSchema`：值类型，编译期生成后不变，可在 C++ struct 生成、ATT 字段查询两处复用（不构造
  MLIR 属性，本文不依赖 MLIR）；
- `PlanSelection`：值类型，包含 `subgraph_id`、`plan_id` 和按 `stage_id` 排列的 template 下标；动态 shape
  场景是结构性剪枝后的候选全集，静态 shape 场景是单个（或 PGO 场景下 Top-N 个）winner；
- `ScheduleFingerprint`：值类型，不含指针，可比较、可序列化，只表示结构部分；最终 artifact key 另有
  target/toolchain/pipeline/schema/编译选项等版本字段。

## 6.2 关键技术/算法

- fingerprint 计算：对图拓扑和 shape 符号等价类做确定性内容哈希，不含指针地址；
- `StaticPlanSelector` 选择：复用 ATT 现有代数成本表达式，在常量 shape 下求值，不涉及图遍历新算法（
  具体求值接口待确认，见"假设和依赖关系"）。

## 6.3 流程设计

见"整体介绍"的目录/依赖图和功能需求 1-6 的处理步骤。异常路径统一在适配组件失败时返回错误，由调用方
回退 legacy Codegen，不尝试修复或生成部分产物；候选组装成可编译产物之后的异常路径（含
`Codegen::GenerateForInductor` 的路由层）属于开放依赖项，待其设计确定后补充。

## 6.4 对子模块的修改

| 目录 | 修改 |
|---|---|
| `autofuse/mlir/common/`（新增） | `fingerprint/` 和 tiling schema 的纯字段模型；当前 schema builder 仍是 ATT/legacy Codegen 过渡实现，ATT 迁移时再替换 `af::AscGraph` 输入边界；独立 CMake target，不链接 LLVM/MLIR |
| `autofuse/mlir/bridge/`（新增） | 当前包含迁移期的 `AttSchemaAdapter/`、`StaticPlanSelector/`；ATT 完成 MLIR 化后删除这两个 legacy 适配目录，由 MLIR-native 组件接替。`bridge/legacy/CMakeLists.txt` 在 `ENABLE_AUTOFUSE_MLIR` 之外加入轻量 bridge target；`AfirBuilder` 仍由 MLIR 开关控制 |
| `autofuse/mlir/bridge/AfirBuilder/AfirBuilder.cpp` | AFIR Module 属性由 `autofuse.node_id` 改为 `afir.subgraph_id`，并同步 template 术语 |
| `autofuse/mlir/bridge/ScheduleToAfir` | 参数、DTO、生成函数名和诊断中的 node/variant 术语统一为 subgraph/template |
| `autofuse/mlir/conversion/AFIRToAscendC/GenKernelEntry.cpp` | 不在本文范围内；PR#19 已合入并按其设计改动该文件 287 行，但"SR module 与 Entry module 最终 link/finalize"仍未解决，见开放依赖项 |
| `autofuse/codegen/codegen_tiling_data.cpp` | `TilingData::Generate` 新增一次诊断性调用 `mlir/common/tiling_schema` 统一入口（不一致只记 `GELOGW`），对外返回值不变 |
| `autofuse/mlir/bridge/AttSchemaAdapter/AttSchemaAdapter.cpp` | 新增字段一致性校验：对 ATT 生成的 `tiling_func` 源码文本做 `get_X`/`set_X` 引用扫描，和 `TilingFieldSchema` 比对，不一致则拒绝产出 `TilingCodeBundle` |
| `autofuse/att/generator/generator_utils/tilingdata_gen_utils.cpp` | 不修改；该文件的工具函数只持有已派生字段名字符串、没有 `ScheduledResult` 上下文，接入需要改签名并牵动 `tiling_code_gen_impl.cpp` 多处调用点，风险和收益不成比例，一致性校验改在 `AttSchemaAdapter` 做 |
| `autofuse/att/`（其余文件） | 不修改 |
| `autofuse/compiler/python/` | MLIR 产物编译接入现有 `compile_adapter.py`/`ascendc_compile.py`，新增 ccache 配置，不新增编译驱动 |
| `autofuse/optimize/` | 不修改 |

## 6.5 错误处理

### 6.5.1 系统错误

- bisheng 子进程编译失败/超时：清理临时目录，回退 legacy 或返回明确错误；
- 动态库加载失败：不缓存 handle，报告加载失败原因；
- ccache 不可用（未安装/配置错误）：回退到无 cache 的直接编译，不阻断构建。

### 6.5.2 接口错误

- `TilingFieldSchema` 输入非法：返回 `af::Status` 错误并标注来源 `(subgraph_id, plan_id, stage_id,
  template_id)`；
- `SelectStaticWinner` 所有候选都不合法：返回失败，上层回退 legacy；
- `AttSchemaAdapter` 生成的字段名与 `TilingFieldSchema` 不一致：返回明确错误，不产出 `TilingCodeBundle`。

# 7 安全检查

| 红线 | 设计措施 |
|---|---|
| 外部输入作索引 | `subgraph_id`/`plan_id`/`stage_id`/`template_id`/`tiling_key` 使用前检查范围 |
| 整数溢出/除 0 | fingerprint 哈希、字段偏移、workspace/block 计算使用 checked arithmetic |
| 资源生命周期 | bisheng 子进程、临时目录、动态库句柄、ccache 目录有显式清理路径 |
| ABI/API 兼容 | `AutofuseTiling`/`AutofuseLaunch`/`CodegenResult`/`att::GenTilingImplAutoFuseV3`/`GetModelInfoMap` 签名不变 |
| 图改写确定性 | plan/stage/template 编号按 vector 下标而非指针/哈希表遍历顺序生成 |
| 非开放接口 | 编译驱动只调用 CANN 公开的 bisheng/ccache 命令行接口；kernel launch 只使用 `<<<>>>`/`ACLRT_LAUNCH_KERNEL` 两种公开机制，不绕过封装调用更底层接口 |
| 日志 | 新增组件不默认开启逐候选/逐字段的高频日志 |
| 硬编码 | 不硬编码芯片型号，target 能力从 Target Profile 查询 |
| Python/C++ 绑定异常与 GIL | 本文新增组件均为纯 C++ 层，不新增 `compiler/py_module/` 绑定代码；复用现状的 `compile_adapter.py` 调用路径不改变异常传播和 GIL 边界，不适用新增风险 |
| 增量/离线构建 | 新增 `autofuse/mlir/common/`、`autofuse/mlir/bridge/` 下的 CMake target 复用现有 MLIR 子目录的 CMake 组织方式，不引入新的第三方下载依赖，不改变 `build.sh --pkg`/`--no-autofuse`/增量构建行为；ccache 为可选加速手段，不可用时按"错误处理"节回退直接编译，不阻断离线构建 |

# 8 兼容性检查

- `AutofuseTiling`/`AutofuseLaunch` 导出符号、参数顺序和职责边界不变；
- `kernel.so`/`wrapper.so` 产物边界不变；
- `att::GenTilingImplAutoFuseV3`、`GetModelInfoMap` 等 ATT 对外签名不变；
- `ENABLE_AUTOFUSE_MLIR=OFF` 时行为和产物与当前完全一致；
- legacy TBE 单算子编译路径（`asc_codegen_compile.py`）不受影响，只是 MLIR 产物不再依赖它；
- `node_idx_to_scheduled_results.size() > 1` 继续回退 legacy，不改变现有行为；
- `autofuse.node_id`→`afir.subgraph_id` 的 MLIR 属性和 aggregate contract key 已同步修改；该内部 IR
  契约不改变产品对外 ABI。

# 9 测试设计

## 9.1 测试边界

- 输入边界：单/多 plan、单/多 stage、单/多 template、空层级、schema 字段不一致、静态/动态 shape、
  常量求值失败、所有候选都不合法；
- 输出验证：`TilingFieldSchema` 生成的字段与 legacy 两处独立生成结果逐字段比对一致；静态 shape winner
  与现有 model-score/PGO 选择结果一致；新旧编译驱动产物的导出符号、tiling 数值、执行结果一致。候选最终
  组装成可编译产物之后的验证（如 bundle/Entry Module 校验）属于开放依赖项范围，不在本文测试边界内。

## 9.2 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|---|---|---|---|
| 功能 | `TilingFieldSchema` 与 legacy 两处生成结果逐字段比对 | 复用现有 golden `AutofuseTilingData` 布局对比 | UT |
| 功能 | 四级坐标和 legacy 字段映射完整 | 构造多 stage/template 结果，逐项比对 AFIR 属性、ATT 输入和 opaque payload | UT |
| 功能 | `AttSchemaAdapter` 生成字段名与 `TilingFieldSchema` 一致 | 构造候选后比对 `tiling_func` 引用的字段名 | UT |
| 功能 | `StaticPlanSelector` 选择与现状一致 | 固定评分/固定 shape，比较选中的 `(plan_id, template_id)` | UT |
| 异常 | 空层级、越界索引、schema 不一致、全候选不合法 | 断言明确错误和 legacy fallback | UT |
| 兼容 | 版本化 artifact key 防止跨 target/toolchain/schema 误命中 | 修改任一版本字段，断言 key 改变；缺字段视为 miss | UT |
| 兼容 | legacy 与新驱动编译产物数值/符号/tiling 一致 | 双路 A/B | ST |
| 兼容 | 诊断 dump 不被误认为生产产物 | 未设置生产接线时断言只生成 dump，不生成 `.so` | ST |
| 性能 | 静态 shape 裁剪前后候选数量差异（编译耗时/产物体积的实际下降需接入开放依赖项后测量） | benchmark | Benchmark |
| 性能 | CMake 驱动 vs 直接 bisheng 驱动（含 ccache 冷/热）编译耗时 | benchmark | Benchmark |
| 特性交叉 | Inductor E2E 数值和 launch 次数与 legacy 一致 | 端到端对比（依赖开放依赖项和后续联调任务） | E2E |

## 9.3 测试命令

```bash
sh build.sh -u --module=autofuse_framework -j 8
sh build.sh -s --module=autofuse_e2e -j 8
bash scripts/test/run_autofuse_test.sh -u -m mlir -j 8
```

无有效 NPU 环境时，只验证结构、代码和缓存契约，不虚构设备性能结果；E2E 用例需要真实 CANN/NPU 环境，
无环境时不得标记为通过。

# 10 验收标准

功能需求 1、2、3、4、6 已按下述标准实现并有 UT 覆盖（`mlir/test/ut/schedule/`）；功能需求 5 未实现、
暂不开发；AFIR 命名和 `PyStructType` 校验收敛已完成。生产入口组装仍是开放依赖项，继续按下文标注为
不在本阶段验收范围。

- `TilingFieldSchema` 生成的字段集合与 legacy 两处独立实现完全一致，且成为 ATT 和 legacy codegen 两边
  唯一的字段名来源；
- `(subgraph_id, plan_id, stage_id, template_id)` 坐标和 `FusedScheduledResult` 的调度/资源字段逐项映射，
  无法结构化的字段有版本化 opaque payload，不能静默丢弃；
- `AttSchemaAdapter` 生成的 `tiling_func` 引用的字段名和 `TilingFieldSchema` 完全一致；
- `StaticPlanSelector`/`SelectDynamicCandidates` 产出的 `PlanSelection` 列表符合本文定义的策略（动态
  shape 全量、静态 shape 单个/Top-N），具体交给下游怎么编译是开放依赖项，不在本阶段验收范围；
- MLIR 产物编译不再依赖 per-kernel CMake configure，接入 ccache 后编译耗时相比 legacy CMake 路径有可
  测量的下降；
- `AutofuseTiling`/`AutofuseLaunch` 导出符号、`<<<>>>` 调用约定、一次 launch 语义、`kernel.so`/`wrapper.so`
  产物边界全部保持不变，legacy 编译产物数值不变；
- 生产接线明确区分 `ScheduleResultToAfir → PassManager → aggregate/link/finalize → emitter`，诊断 dump
  不被当成可执行产物；aggregate/link/finalize 未冻结前保持开放依赖项；
- artifact cache key 包含结构 fingerprint、target/toolchain/pipeline/schema/编译选项版本，任何版本字段
  不可稳定序列化时按未命中处理；
- 所有新增组件的 UT/ST 通过，`ENABLE_AUTOFUSE_MLIR=OFF` 时行为与当前完全一致；`att/`、`codegen/` 链接
  `mlir/common/` 的轻量 target 后确认不传递引入 LLVM/MLIR 依赖；
- 预编译共享 runtime 已确认在公开接口下不可行，不作为本阶段或后续阶段的验收内容；
- 编译驱动改直接调用 bisheng（功能需求 5）未实现，暂不开发，不在本阶段验收范围；
- 开放依赖项（生产入口如何组装候选、静态 shape 场景实际编译耗时/产物体积下降）继续不在本阶段验收范围，
  待其设计确定后另行验收。
