# Autofuse 从 ScheduledResult 到 MLIR 的编译流程迁移设计

> **2026-10-22 状态更新**：生产 MLIR Codegen 路由、AFIR→AscendC、legacy/MLIR 对照入口已落地，
> 开关为 `AF_ENABLE_MLIR_CODEGEN`；ATT/Schedule 完整迁移和 legacy 退役仍是长期目标。
> 以下旧日期的状态及代码盘点是历史基线，不能据此判断生产链路尚未接入。当前流程见
> [Pass pipeline](2026-08-05-afir-to-ascendc-pass-pipeline-design.md) 和 [文档索引](../README.md)。

> 状态：草稿（ScheduledResult → AFIR 调试桥已实现，Codegen/ATT/Schedule 迁移尚未完成）
> 日期：2026-07-27
> 现状复核日期：2026-08-06
> 迁移顺序：Codegen → ATT → Schedule
> 核心决策：一个 ScheduledResult 对应一个 MLIR Module；AF payload dialect 不定义 schedule、group、result 容器 op
> 文档类型：迁移架构设计

> 当前代码已由 `ScheduleToAfir`/`AfirBuilder` 将每个 legacy `ScheduledResult` 构造成独立 AFIR Module，并在
> 统一 dump 前缀（`ascir::utils::GetDumpFilePrefix()`） 设置时从 legacy Codegen 路径输出调试 IR；该入口不参与生产 lowering。Module entry
> 组合语义、MLIR Codegen、ATT Host Tiling Module、Transform Sequence 和 Schedule 迁移仍是本文规划内容。

### 设计文档检查结果

- [x] 跨特性交叉影响：已逐项检查 Autofuse optimize、ATT、Codegen、Python/C++ 接口、构建交付和测试路径
- [x] 编码红线：已检查 ABI/API、图改写确定性、Shape/tiling 数值安全和 runtime 生命周期
- [x] 测试设计：给出 bridge、Codegen、ATT、Schedule、静态/动态 Shape 的 UT/ST/E2E 覆盖
- [x] 性能评估：给出 IR 规模、编译耗时、runtime dispatch 和静态特化的性能边界

# 简介

## 目的

本文说明如何把 Autofuse 从 `ScheduledResult` 逐步迁移到 MLIR。重点不是把现有 C++ 容器逐层翻译成 Op，而是先
分清三个概念：完整执行方案、方案的生成过程，以及运行时的 tiling 选择。

全文回答五个问题：

1. `FusedScheduledResult` 的四层结构分别表示什么？
2. 为什么一个 `ScheduledResult` 对应一个 MLIR Module？
3. 如何在保持 legacy optimize 和 Codegen 不变的前提下先接入 MLIR Codegen？
4. ATT 迁移后，静态 Shape 和动态 Shape 分别在什么时候选择方案和 tiling？
5. Schedule 最终如何从 `OriginGraph` 生成 `Transform Sequence` 和候选 Module？

如果只需要了解结论，先读“项目介绍”中的术语表、功能需求 2，以及文末“最终结论”。其余章节给出迁移时序、
代码依据和验证方法。

## 范围

本设计覆盖：

- common/schedule_result.h 中的数据层级；
- mlir/bridge/ScheduleToAfir 当前桥接缺陷；
- codegen 的 ScheduledResult、ScheduleGroup 和 tiling-key 执行语义；
- optimize PartA/PartB 的实际职责；
- ATT 的 ModelInfo、solver、GetTiling 和可选 PGO；
- Codegen → ATT → Schedule 三阶段迁移；
- 动态 Shape 通用路径和静态 Shape 专用优化。

本次不做：

- 不修改 FusedScheduledResult、ScheduledResult、ScheduleGroup 的 legacy 定义；
- 不重构 legacy optimize/、att/、codegen/；
- 不在 AF dialect 增加 schedule/group/result 容器 op；
- 不增加 Autotune 数据库、JIT 调优服务或远程缓存；
- 不承诺第一阶段支持 node_idx_to_scheduled_results.size() > 1 的 FusedAscBackend 特殊路径。

# 总体概述

## 软件概述

### 项目介绍

当前主流程如下：

    OriginGraph
      → optimize::Optimizer
      → FusedScheduledResult
      → ATT 生成 host tiling solver
      → codegen 生成一个 AscendC kernel 入口及若干 device 函数
      → Ascend 编译器

`FusedScheduledResult` 的数据结构有四层：

    FusedScheduledResult
      node_idx_to_scheduled_results[node_index]
        ScheduledResult[result_index]
          ScheduleGroup[stage_index]
            impl_graphs[template_index]

这些层级是 legacy 数据组织方式，不需要在 MLIR 中逐层创建容器 Op。迁移时使用下面的对应关系：

| legacy 层级 | 真实语义 | MLIR 表达 |
|---|---|---|
| node_index | 一个融合输入中需要执行的逻辑图位置 | 第一阶段仅支持单 node；不建 op |
| ScheduledResult | 一个完整的执行方案，也是本设计的编译/下发单元 | 一个 builtin.module |
| ScheduleGroup | 方案中的一个执行 stage | private func 上的 stage_id 属性和 public entry 中的调用顺序 |
| impl_graph | 某个 stage 的一个实现 template | 一个 private func.func |

后文反复使用以下三个名称：

| 名称 | 含义 | 生命周期 |
|---|---|---|
| Transform Sequence | 从原始图生成某个方案的一组变换记录 | 只存在于 Schedule 和调优阶段 |
| Plan Module | 一个 `ScheduledResult` 对应的完整设备执行方案 | 供 ATT、Codegen 和设备编译使用 |
| Host Tiling Module | 根据具体 Shape 选择 Plan、stage template 和 tiling 参数的主机侧逻辑 | 静态特化或运行时执行 |

### 产品环境介绍

迁移后仍需保持以下外部契约：

- 上游 optimize 仍输出 FusedScheduledResult；
- Python Codegen 接口仍输出 CodegenResult；
- tiling data 的字段布局、block_dim、workspace 和 tiling-key 行为在 ATT 迁移完成前保持兼容；
- ENABLE_AUTOFUSE_MLIR=OFF 时不引入 MLIR/LLVM/PyAsc 依赖；
- MLIR 路径失败时可以切回 legacy 路径做行为和性能对比。

## 软件功能

迁移分为两个状态。迁移期间保留 legacy optimize，把它的结果桥接为 Plan Module；全部迁移完成后，直接从
`OriginGraph` 生成 Plan Module。

    迁移期
    legacy optimize
      → FusedScheduledResult
      → ScheduleToMlir
      → Plan Module[]
      → MLIR Codegen
      → legacy ATT 或 MLIR ATT

    终态
    OriginGraph
      → OriginGraphToMlir
      → DesignSpaceGenerator
      → Transform Sequence[]
      → Plan Module[]
      → MLIR ATT
      → MLIR Codegen

## 设计约束

1. 一个 ScheduledResult 只能生成一个 Module，不能把整个 FusedScheduledResult 压成一个 Module。
2. Module 内不能出现 af.schedule、af.group、af.schedule_result。
3. 每个 ScheduleGroup 选择一个 impl_graph；所有 ScheduleGroup 都执行。
4. public entry 必须表达 stage 顺序、workspace 传递、同步和 group parallel。
5. bridge 不做笛卡尔积展开；组合 key 在 entry 中解码，必要时由末端 specialization pass 展开。
6. 动态 Shape 的 plan 选择发生在运行时 GetTiling；静态 Shape 才能在编译/离线测量后裁剪为单一 winner。
7. 任何候选选择都以完整 ScheduledResult/Module 为单位，不能逐 Group 各选一个局部最优后假设全局最优。

## 假设和依赖关系

- `optimize::Optimizer::Optimize(AscGraph&, ...)` 当前把 `node_idx` 外层固定为 1，第一阶段只支持这种输入。
- `OptimizeFusedAscBackend` 可能产生多个 `node_index`。在明确多 Module 的 launch/link 语义前，这条路径继续使用
  legacy 实现。
- `score_func` 当前是 C++ 字符串。ATT 迁移期把它作为 opaque metadata，不假设它能稳定解析为 MLIR 表达式。
- `ScheduleToAfir` 当前只是调试 bridge，不是生产 Codegen 入口。

# 需求分析与设计

## 整体介绍

本设计把“如何生成方案”和“方案如何执行”分开：

    Transform Sequence
      记录如何从 OriginGraph 产生一个方案，只存在于 Schedule/调优阶段

    Plan Module
      表达这个方案在设备上如何完整执行，是 Codegen 和 ATT 的共同输入

    Host Tiling Module
      表达动态 Shape 下如何选择 plan、stage template 和数值 tiling 参数

Plan Module 不再保存 `ScheduleGroup` 容器。各 stage 的顺序、template 选择、workspace 传递和同步关系都由 public
entry 的控制流表达。

## 功能需求 1：确认当前代码的真实语义

### 1. 介绍

先根据当前代码确定四层结构的真实语义，避免把 C++ vector 层级直接复制成 MLIR 容器。

### 2. 输入

- common/schedule_result.h；
- optimize/；
- codegen/codegen_kernel.cpp；
- att/。

### 3. 处理

#### 3.1 Schedule 的真实职责

optimize/optimize.cpp:890 的 OptimizeForHintGraph 执行：

1. 图预处理、dtype 一致性和 GraphPass；
2. 合轴及图合法性处理；
3. ScheduleTaskGenerator::GenerateTasks；
4. 对每个 ScheduleTask 调用 AutoScheduler；
5. 最后统一做 Buf/Queue 分配、静态 UB 模板过滤、group parallel 和 load 顺序调整。

PartA 的 `GenerateTasks` 负责生成结构候选，不只是选择切分轴。`optimize/platform/v1/platformv1.cpp:74` 会依次
生成 Split、Concat、Transpose、Reduce 和 Recompute 任务；Reduce 还可能生成 RCore 两阶段 `grouped_graphs`。

PartB 对应 AutoSchedule::DoAutoSchedule：

- TilingGroup::GenTilingGroup 生成轴组；
- 枚举 TilingCase；
- 每个 case clone graph；
- Scheduler::DoScheduler 执行 split、merge、reorder、bind、cache 等变换；
- 产出多个 AutoScheduleOutput。

非 Reduce-first-stage 路径在 optimize/optimize.cpp:1020 把多个 AutoScheduleOutput 放入同一个 ScheduleGroup.impl_graphs。RCore 第一阶段在 :520 为每个第一阶段结果复制 ScheduledResult，再继续追加后续 stage。

因此，一个 `ScheduledResult` 是完整方案；一个 `ScheduleGroup` 是方案中的 stage；一个 `impl_graph` 是该 stage
的一种实现 template。

#### 3.2 Codegen 的真实执行语义

codegen/codegen_kernel.cpp:2616 的 GenMulGroupKernelWithRegTilingKey：

1. 遍历 ScheduledResult；
2. 对每个 ScheduleGroup 收集所有 impl_graph 的函数调用；
3. 调用 AppendFuncCall。

AppendFuncCall 位于 codegen/codegen_kernel.cpp:3438。它递归遍历 per_group_func_calls，枚举笛卡尔积。到叶子时，一个 tiling-key 分支会调用 current 中的全部函数。

假设两个 stage 分别有 2、3 个 template，则生成 6 个组合：

    key 0 → stage0.v0 + stage1.v0
    key 1 → stage0.v0 + stage1.v1
    key 2 → stage0.v0 + stage1.v2
    key 3 → stage0.v1 + stage1.v0
    key 4 → stage0.v1 + stage1.v1
    key 5 → stage0.v1 + stage1.v2

所以：

- 不是每个 Group 独立下发；
- 不是每个 AscGraph 独立下发；
- 一次下发执行一个 ScheduledResult 中每个 stage 选出的一个 template；
- stage 间存在 workspace 时，Codegen 会在中间插入 SyncAll/PipeBarrier。

#### 3.3 当前 MLIR bridge 的具体问题

mlir/bridge/ScheduleToAfir/ScheduleToAfir.cpp:216 的 ScheduleResultToAfir 实际接收整个 FusedScheduledResult。

其 :223-231 直接四层遍历，并把每个 impl_graph 追加到同一个 KernelInfo.graphs：

    for node results
      for scheduled result
        for group
          for impl graph
            kernel.graphs.push_back(...)

丢失的信息包括：

- impl_graph 属于哪个 ScheduledResult；
- 属于哪个 stage；
- 同 stage 中的 template 编号；
- ScheduledResult 的 score_func、var_relations、enable_group_parallel、cube_type；
- stage 的 double_buffer；
- public entry 中应如何选择并执行各 stage。

codegen/codegen.cpp:392 的 MaybeDumpAfirFromSchedule 位于 legacy kernel、tiling_data 和 tiling 生成之后，而且只受环境变量控制 dump。它不是 MLIR Codegen 路由。

### 4. 输出

第一项改动是让 bridge 保留方案边界，而不是新增容器 Op：

    FusedScheduledResult
      → [Module(plan0), Module(plan1), ...]

并为每个 Module 建立完整 public entry。

## 功能需求 2：ScheduledResult 对应 Module

### 1. 介绍

一个 Plan Module 至少包含：

- Module 属性：plan_id、legacy node/result 坐标、shape mode；
- private func：每个 impl_graph 一个；
- private func 属性：stage_id、template_id、double_buffer；
- 一个 public kernel entry；
- entry 中显式完成 template 选择、stage 调用、同步和 workspace 传递。

这些信息可以用 `builtin.module`、`func.func`、`arith`、`scf` 和后端计算 Op 表达，不需要额外定义 AF 容器 Op。

### 2. 输入

一个 ScheduledResult，加上 FusedScheduledResult 中共享的 inputs、outputs、workspace 和 origin_vars。

### 3. 处理

以下例子表示两阶段 Reduce，每个 stage 有两个 template。示例省略具体 AFIR 计算 op，只保留控制关系：

    module @reduce_plan_1 attributes {
      afir.plan_id = 1 : i64,
      autofuse.legacy_node_id = 0 : i64,
      autofuse.legacy_result_id = 1 : i64,
      afir.stage_count = 2 : i64,
      afir.template_counts_per_stage = array<i64: 2, 2>,
      autofuse.shape_mode = "dynamic"
    } {
      func.func private @stage0_v0(
          %input: memref<?xf16>, %workspace: memref<?xf32>,
          %tiling: memref<?xi8>)
          attributes {
            afir.stage_id = 0 : i64,
            afir.template_id = 0 : i64,
            afir.double_buffer = true
          } {
        // 切 AB 轴的第一次 Reduce
        return
      }

      func.func private @stage0_v1(...) attributes {
        afir.stage_id = 0 : i64,
        afir.template_id = 1 : i64
      } {
        // 切 AC 轴的第一次 Reduce
        return
      }

      func.func private @stage1_v0(...) attributes {
        afir.stage_id = 1 : i64,
        afir.template_id = 0 : i64
      } {
        // 对 workspace 做第二次 Reduce，template 0
        return
      }

      func.func private @stage1_v1(...) attributes {
        afir.stage_id = 1 : i64,
        afir.template_id = 1 : i64
      } {
        // 对 workspace 做第二次 Reduce，template 1
        return
      }

      func.func public @kernel_entry(
          %input: memref<?xf16>, %output: memref<?xf32>,
          %workspace: memref<?xf32>, %composite_key: i32,
          %tiling: memref<?xi8>) {
        %c0 = arith.constant 0 : i32
        %c2 = arith.constant 2 : i32
        %stage0_template = arith.divui %composite_key, %c2 : i32
        %stage1_template = arith.remui %composite_key, %c2 : i32

        %s0_is_v0 = arith.cmpi eq, %stage0_template, %c0 : i32
        scf.if %s0_is_v0 {
          func.call @stage0_v0(%input, %workspace, %tiling) : (...)
        } else {
          func.call @stage0_v1(%input, %workspace, %tiling) : (...)
        }

        ascendc.sync_all

        %s1_is_v0 = arith.cmpi eq, %stage1_template, %c0 : i32
        scf.if %s1_is_v0 {
          func.call @stage1_v0(%workspace, %output, %tiling) : (...)
        } else {
          func.call @stage1_v1(%workspace, %output, %tiling) : (...)
        }
        return
      }
    }

这个例子表达了六条规则：

1. Module 就是 ScheduledResult；
2. stage0 和 stage1 都执行；
3. 每个 stage 只选一个 template；
4. workspace 依赖通过函数参数和调用顺序表达；
5. SyncAll 是计算/同步 op，不是容器；
6. stage_id/template_id 只用于分析和校验，Codegen 不需要重建 legacy 四层 vector。

#### 3.1 tiling-key 不在 bridge 中做笛卡尔积展开

当前 AppendFuncCall 的 key 编码是 mixed-radix：

    key = v0 * (n1 * n2 * ... * nS-1)
        + v1 * (n2 * ... * nS-1)
        + ...
        + vS-1

bridge 只写 template_counts_per_stage，entry 用除法/取模解码。这样 IR 大小是：

    O(各 stage template 数量之和)

而不是：

    O(各 stage template 数量乘积)

如果 Ascend 编译器需要 TILING_KEY_IS 常量分支，在 AFIR lowering 末端增加 PlanDispatchSpecializePass：

1. 读取允许的 composite key；
2. clone kernel_entry；
3. 把 key 设为常量；
4. canonicalize 删除未选 template；
5. 发射当前 TILING_KEY_IS 形式。

这保证前端 IR 简单，后端仍能得到常量特化。

#### 3.2 与当前两种 tiling-key 路径的精确映射

当前代码存在两种不同的 key 语义，MLIR lowering 必须分别兼容。

动态/parse-tiling-data 路径：

    graph0_tiling_key
      = PlanId，即 ScheduledResult 下标

    graph0_result<PlanId>_g<StageId>_tiling_data.tiling_key
      = StageTemplateId[StageId]

    graph0_result<PlanId>_g<StageId>_tiling_data 的其他字段
      = 该 stage 的 TilingParams

代码依据：

- codegen/codegen_kernel.cpp:2917 用 graphN_tiling_key 与 ScheduledResult 下标 i 比较；
- att/generator/tiling_code_gen_impl.cpp:3883 也用 graphN_tiling_key 判断 result_id；
- att/generator/tiling_code_gen_impl.cpp:3237 调用所有 Group 的 GetTiling；
- :3253 从每个 group tiling data 读取 get_tiling_key()。

因此 MLIR ATT 的 PlanId + StageTemplateId[] 可以直接 pack 到现有字段，不需要修改 runtime ABI。

静态/注册 TILING_KEY_IS 路径把 plan 和 stage template tuple 压成一个全局 key。对 plan p：

    local_count[p] = product(template_counts_per_stage[p])

    key_base[p] = sum(local_count[q]), q < p

    global_key = key_base[p] + local_composite_key

例如 plan0 的 template_counts_per_stage=[2, 3]，plan1=[2]：

    plan0 使用 global key 0..5
    plan1 使用 global key 6..7

这与 GenMulGroupKernelWithRegTilingKey 中 tiling_key 在所有 ScheduledResult 外部初始化、由 AppendFuncCall 递增的行为一致。

#### 3.3 多 Module 如何保持当前单一 kernel 导出

ScheduledResult 对应 Module 是编译器内部边界，不要求迁移第一天就改变 CANN 的 kernel 打包 ABI。

M1 增加 PlanBundleLinker：

    Module(plan0).kernel_entry
    Module(plan1).kernel_entry
      → 一个兼容 wrapper
      → 当前 fused_graph_name 对应的单一 extern "C" __global__ __aicore__ 入口

动态路径 wrapper：

    switch tiling_data.graph0_tiling_key
      case 0: call plan0.kernel_entry(
                plan0 各 stage 子 tiling data)
      case 1: call plan1.kernel_entry(
                plan1 各 stage 子 tiling data)

静态注册 key 路径：

1. 根据 key_base 找到 plan；
2. 把 global key 转成 local_composite_key；
3. 对 plan entry 做常量 specialization；
4. 发射现有 TILING_KEY_IS(global_key) 分支。

所以对外仍是现有 kernel symbol、tiling data 和 tiling-key；Module 拆分只简化 MLIR 内部结构。

#### 3.4 group parallel

enable_group_parallel 不能只作为无效属性保存。entry 必须按当前 Codegen 语义生成：

- 无数据依赖：各 stage 根据 block range 条件执行；
- 有 var_relations/workspace 依赖：保持顺序并插入同步；
- block_dim 取各参与 stage 的最大值；
- 每个 stage 使用自己的 block offset/有效 core 数。

第一阶段不重新设计并行算法，只翻译 legacy enable_group_parallel 和 ATT 生成的 tiling 字段。

### 4. 输出

    FailureOr<SmallVector<OwningOpRef<ModuleOp>>>
    ScheduleToMlir(const ascir::FusedScheduledResult &result);

当 node_idx_to_scheduled_results.size() != 1 时返回明确的 Unsupported，由上层回退 legacy，不能继续扁平化生成错误 IR。

## 功能需求 3：与业界方案的关系

### 1. 介绍

比较的重点不是命名是否一样，而是编译单元、搜索历史和 runtime 选择是否分层。

### 2. 输入

TVM TensorIR/MetaSchedule、PyTorch Inductor、XLA 和 MLIR Transform dialect 的公开设计。

### 3. 处理

| 系统 | 实际做法 | 对 Autofuse 的结论 |
|---|---|---|
| TVM | IRModule/PrimFunc 是 payload；Schedule/Trace 记录 split、reorder、cache、bind；调优选择完整 workload 的 trace | 不把 schedule 层级做成 payload 容器；Module + Trace 合理 |
| Inductor | FX graph 经 scheduler/fusion 生成一个或多个 Triton/C++ kernel，由 wrapper 表达完整执行；静态 choice 可 benchmark，动态 Shape 用 guard/symbolic code | 完整候选可以含多个函数/stage，不能把单 Group 当完整候选 |
| XLA | HLO module/fusion 作为编译单元；对完整 fusion 或 library config 选优；静态 Shape 强特化 | StaticExact 应选择完整 Module winner |
| MLIR Transform | Transform IR 操作独立 payload IR，trace 可重放；搜索策略由上层 driver 控制 | Transform Sequence 不应进入 AF payload Module |

一个 ScheduledResult 对应 Module 与业界思路一致，但要注意：

- Module 是编译/候选单元，不要求内部只有一个函数；
- Autotune 选择的是完整 Module；
- Transform Sequence 是产生 Module 的过程，不是 Module 的运行时结构；
- 业界会用规则生成合法 trace，不会枚举 transform op 的任意排列。

### 4. 输出

采用 Module + Transform Sequence；不采用 AF 容器 op；不引入调优数据库作为迁移前提。

## 功能需求 4：Codegen 第一阶段迁移

### 1. 介绍

legacy optimize 和 legacy codegen 保持原样。新增一条并行的 MLIR Codegen 路径，在编排层选择 off/on/compare。

### 2. 输入

- FusedScheduledResult；
- CodegenOptions；
- ENABLE_AUTOFUSE_MLIR 和 AF_MLIR_CODEGEN 路由配置。

### 3. 处理

正确时序：

    legacy optimize
      → FusedScheduledResult
      → ScheduleToMlir
      → VerifyPlanModule
      → AFIR lowering
      → AscendC lowering/translation
      → CodegenResult

错误时序是当前 dump hook：

    legacy optimize
      → legacy kernel/tiling 已全部生成
      → codegen.cpp:392 再 dump MLIR

因此不继续扩展 MaybeDumpAfirFromSchedule。新路径放在 compiler 编排层，与 compiler/py_module/pyautofuse.cpp:244 当前调用 Codegen::GenerateForInductor 的位置并列：

    off     → legacy Codegen
    on      → MlirCodegenPipeline
    compare → 两边都生成，比较 schema/源码/数值，返回 legacy 结果

MlirCodegenPipeline 的步骤必须固定为：

1. ScheduleToMlir：一个 ScheduledResult 生成一个 Module；
2. VerifyPlanModule：校验一个 public entry、stage 连续、template 唯一、workspace 依赖完整；
3. LowerAscGraphOps：把每个 private func 内的 AscGraph 转成 AFIR；
4. LowerPlanDispatch：把 entry 的 key 解码、stage 调用和同步降到 AscendC；
5. PlanBundleLinker：把多个 Plan Module 组装为当前单一导出 kernel；
6. PackTilingData：保持现有两级动态字段和静态 global key ABI；
7. TranslateToAscendC：输出 CodegenResult.kernel；
8. ATT 未迁移前，tiling 和 tiling_data 仍调用 legacy ATT 生成结果。

第一阶段用例顺序：

1. 单 ScheduledResult、单 stage、单 template；
2. 单 ScheduledResult、单 stage、多 template；
3. 单 ScheduledResult、多 stage、每 stage 单 template；
4. 多 stage、多 template、workspace；
5. enable_group_parallel；
6. 多 ScheduledResult；
7. cube/CV 特殊路径；
8. multi-node FusedAscBackend，最后单独处理。

### 4. 输出

- 外部仍为 CodegenResult；
- legacy Codegen 无代码行为变化；
- compare 模式能逐 plan 比较函数数、stage/template 映射、tiling-key 映射和生成代码。

## 功能需求 5：ATT 第二阶段迁移

### 1. 介绍

ATT 接收一组 Plan Module，并生成 host tiling 代码。动态 Shape 在运行时选 Plan，静态 Shape 可以在编译和测量后
裁剪为单个 Plan，因此两条路径的时序必须分开。

### 2. 输入

- Plan Module[]；
- Shape、dtype、layout；
- UB、core num 等 target capability；
- 可选 profiling callback。

### 3. 处理

#### 3.1 当前 ATT 代码闭环

当前真实调用链为：

    att/gen_tiling_impl.cpp::GenTilingImplAutoFuseV3
      → GetModelInfoMap
      → TilingCodeGenerator::GenTilingCode
      → GenScheduleGroupTilingBodies
      → TilingCodeGenImpl::GenGetTilingKey
      → root GetTiling
      → 可选 GenPGOSearchTilingKey
      → codegen::TilingLib 生成 PGO wrapper
      → 可选输出 search/config 文件

对应代码事实：

- att/gen_model_info/gen_model_info.cpp:468 显式解析四层结构；
- :493 为每个 ScheduleGroup 生成 ModelInfo；
- att/generator/tiling_code_generator.cpp:265 为每个 Group 生成 solver body；
- att/generator/tiling_code_gen_impl.cpp:2573 生成 GetTilingKey；
- :2611 生成可选 PGOSearchTilingKey；
- :4491 组装 solver 和入口；
- codegen/codegen_tiling.cpp:2430 起连接 PGO callback，并由 SavePGOSearchTilingData/SavePGOConfigTilingData 写文件。

当前 ATT 不是 Autotune 数据库系统，本次迁移也不增加数据库。

#### 3.2 MLIR ATT 的具体产物

ATT 不修改 Device Plan Module，而是额外生成 Host Tiling Module：

    module @host_tiling {
      func.func private @solve_plan0_stage0(...)
      func.func private @solve_plan0_stage1(...)
      func.func private @score_plan0(...)

      func.func private @solve_plan1_stage0(...)
      func.func private @score_plan1(...)

      func.func public @autofuse_get_tiling(...) -> i1
      func.func public @autofuse_pgo_search(...) -> i1  // 仅开启 PGO 时
    }

函数映射：

| 当前 ATT | MLIR ATT |
|---|---|
| GetModelInfoMap/GenerateModelInfo | BuildTilingModelAnalysis(ModuleOp) |
| 每 Group 的 GenTiling | 每 stage 的 solve_planX_stageY |
| ScheduledResult score_func | score_planX；兼容期可调用 opaque legacy helper |
| root GetTiling | autofuse_get_tiling |
| PGOSearchTilingKey | autofuse_pgo_search |
| tiling data C++ setter | PackLegacyTilingData lowering |

逻辑层统一使用三个不同概念：

    PlanId
      选择哪个 ScheduledResult/Module

    StageTemplateId[]
      每个 stage 选择哪个 private func

    TilingParams[]
      每个 stage 的 block_dim、tile size、workspace 等数值

PackLegacyTilingData 在最后一步把它们编码回当前 tiling data，避免 MLIR 内部继续混用 tiling_key 一词。

具体编码沿用 M1：

- 动态：graphN_tiling_key 保存 PlanId，result/group 子结构的 tiling_key 保存 StageTemplateId；
- 静态注册 key：key_base[PlanId] + mixed-radix(StageTemplateId[])；
- TilingParams 写回对应 result/group 子结构；
- 不改变字段名、字段类型和对齐。

#### 3.3 动态 Shape

动态 Shape 不能在编译期选出唯一 Module。正确流程是：

    编译期
    Plan Module[]
      → BuildTilingModelAnalysis
      → 生成并编译 Host Tiling Module
      → 编译全部允许的 Device Plan

    运行期
    Concrete Shape + target runtime 参数
      → autofuse_get_tiling
      → 求解每个合法 Plan 的各 stage
      → 计算完整 Plan score
      → 选 PlanId
      → 写 StageTemplateId[]、TilingParams[]、block_dim、workspace
      → runtime 下发选中的 Device Plan

运行期比较的是完整 Plan：

- 顺序 stage：包含各 stage cost、同步和 workspace；
- group parallel：按并发模型计算整体耗时，不能简单求和；
- 任一 stage 无合法 tiling，则整个 Plan 淘汰；
- 所有 Plan 失败则返回当前 ATT 等价错误，不生成半合法 tiling。

#### 3.4 静态 Shape

静态 Shape 的测量必须在 Device Plan 完成 Codegen/Compile 之后：

    Plan Module[]
      → 绑定静态 Shape
      → 静态 specialization
      → ATT solver 常量求值
      → 编译 Top-N Device Plan
      → 可选 PGO callback 测量
      → 选择一个 Module winner
      → 删除其他 Module 和 winner 中未选 template
      → 打包

不开 PGO 时，用 ATT model score 选择 winner。开 PGO 时，沿用现有 callback 和 search/config 文件；不增加数据库。

#### 3.5 PGO 的位置

PGO 不能放在 Candidate Module 生成之后、Device 编译之前，因为 callback 需要可运行 kernel。

正确位置：

    Static Shape：
      Device compile → PGO measure → winner

    Dynamic Shape：
      对给定 representative shapes：
      Device compile → PGO measure → 生成有限的 shape/profile 配置
      未命中的 Shape 仍走 autofuse_get_tiling

### 4. 输出

- Host Tiling Module；
- 与当前 C ABI 兼容的 host tiling 函数；
- 动态 Shape：运行时 PlanId、StageTemplateId[] 和 TilingParams[]；
- 静态 Shape：一个裁剪后的 winner Module；
- 可选 PGO search/config 文件；
- 无 Autotune 数据库。

## 功能需求 6：静态 Shape 专用优化

### 1. 介绍

静态优化分为迁移期可做的后处理和 Schedule 迁移后才能做的完整搜索，二者不能混为一谈。

### 2. 输入

- 全部关键维度已知的 Plan Module；
- target capability；
- 编译和可选测量预算。

### 3. 处理

#### 3.1 Codegen/ATT 迁移期可实现

legacy Schedule 已经完成，能做的只有对现有候选特化：

1. BindStaticShape：把符号 dim 替换为常量；
2. canonicalize + CSE：折叠 Shape/stride/index 表达式；
3. PrunePlanTemplate：删除约束不满足的 template；
4. ResolvePlanDispatch：把 PlanId/StageTemplateId 常量化；
5. EliminateTail：仅在证明整除时删除 tail 分支；
6. LoopCanonicalize/Unroll：对小常量循环展开；
7. ExactMemoryCheck：重新校验 UB/workspace 峰值；
8. Codegen 后可选 PGO 选 winner。

这一阶段不能凭空产生 legacy Schedule 没有生成的 split/reorder/template，因此性能上限仍受 legacy design space 限制。

#### 3.2 Schedule 迁移后的完整 StaticExact

静态 Shape 必须在 DesignSpaceGenerator 之前可见：

    OriginGraph + exact Shape
      → 静态图简化
      → 结构方案枚举
      → 精确 tile 因子枚举
      → Transform Sequence
      → Plan Module
      → 精确资源过滤
      → cost Top-K
      → compile Top-N
      → 可选 PGO
      → winner

静态专用搜索维度：

| 层级 | 搜索内容 |
|---|---|
| 结构 | 单/多 stage Reduce、Common/AllLoad/RCore、fusion/partition/recompute |
| 轴 | merge、split factor、reorder、block bind、vector axis |
| 流水 | double buffer、queue depth、load/compute/store overlap |
| 代码 | tail-free vectorization、固定 mask/repeat/stride、小循环 unroll |
| 内存 | 精确 live range、buffer reuse、workspace 和 UB 峰值 |

约束：

- tile factor 优先枚举 Shape 的因子，另保留少量合法 tail 方案；
- 删除 tail 前必须证明整除；
- bufferization/allocation 是确定性 realization pass，不作为任意 transform 排列；
- 最终性能以完整 Plan 的设备耗时为准。

### 4. 输出

- 迁移期：现有 Plan 的静态特化版本；
- 终态：从静态专用 design space 选出的单一 winner。

## 功能需求 7：Schedule 第三阶段迁移

### 1. 介绍

Schedule 迁移不能从 FusedScheduledResult 反推 Transform Sequence，必须从未调度的 OriginGraph 开始。

### 2. 输入

- OriginGraph MLIR；
- Shape constraints；
- target capability；
- 搜索预算。

### 3. 处理

新增入口：

    OriginGraphToMlir(OriginGraph)
      → Payload Module

DesignSpaceGenerator 使用规则生成合法序列，不做 A/B/C/D/E 任意排列：

    normalize
      → choose algorithm/template
      → partition/decompose into stages
      → choose axis mapping
      → split/merge/reorder/bind
      → choose memory strategy
      → deterministic bufferization/memory planning
      → verify

建议的 Transform op 粒度：

| 当前 optimize 行为 | Transform/Pass 归属 |
|---|---|
| Split/Concat/Transpose/Reduce/Recompute task | rule + decompose/partition transform |
| RCore 两阶段 | decompose_reduce transform，产出两个 payload func |
| GenTilingGroup/GenTilingCase | DesignSpaceGenerator 的 decision |
| split/merge/reorder/bind | 原子 transform op |
| cache/double buffer 策略 | transform 参数 |
| Buf/Queue 分配 | 确定性 bufferization/memory planning pass |
| dtype consistency/toposort/合法化 | 普通 analysis/pass |
| ATT cost/测量 | ATT driver，不是 transform op |

每条 Transform Sequence 的结果是一个完整 Plan Module。两阶段 Reduce 仍是一个 Module，内部有两个 stage，不是两个候选。

Transform Sequence 最少记录：

- rule 名和版本；
- 每个 decision 的稳定参数；
- payload op 的稳定匹配标识；
- Shape/target 前置条件；
- 失败原因；
- canonical trace hash。

候选生成：

    rule legality
      → apply trace to cloned Origin Module
      → verifier
      → resource check
      → canonical hash 去重
      → Plan Module[]

### 4. 输出

- Transform Sequence[]；
- Plan Module[]；
- 不再生成 FusedScheduledResult 的 MLIR-native 路径。

# 非功能需求

## 可维护性

- payload 只包含计算、内存和控制流；
- stage/template 用函数属性，不增加容器层；
- ATT analysis 使用 typed C++ side table，不把大段 C++ 字符串塞进属性；
- Shape、target、tiling-key ABI 转换集中在 adapter；
- legacy 和 MLIR 路径通过同一输入/输出做 A/B。

## 可测试性

每层有独立验证入口：

- ScheduledResult → Module；
- VerifyPlanModule；
- mixed-radix key encode/decode；
- Module → AscendC；
- BuildTilingModelAnalysis；
- Host Tiling Module；
- Transform Sequence replay；
- StaticExact specialization。

## 可移植性

- 不硬编码芯片型号；
- UB、core、vector width、alignment 从 target capability 获取；
- ENABLE_AUTOFUSE_MLIR=OFF 不发现 MLIR 依赖；
- PyAsc/AscendC op 缺失时在 lowering 阶段返回明确 Unsupported。

## 可靠性

- bridge 校验空 ScheduledResult、空 stage、重复 template id；
- composite key 解码前校验 key 范围；
- Shape/tiling 的乘法、字节数、workspace offset 检查溢出；
- 静态优化删除 guard/tail 前必须有可验证证明；
- MLIR candidate 失败只淘汰当前 candidate，不污染 Origin Module；
- PGO 失败回退 model score；
- 动态 Shape 保留通用 solver fallback。

## 特性交叉影响

| 场景 | 适用性 | 结论 |
|---|---|---|
| SuperKernel Python 接口 | 不直接适用 | 本阶段不改 Python API；交付时跑既有 E2E |
| SuperKernel C++/AOT 接口 | 部分适用 | kernel symbol、tiling ABI 和打包布局必须保持 |
| Autofuse 图优化 | 适用 | M3 才替换；M1/M2 冻结 legacy optimize |
| Autofuse Codegen/Backend | 适用 | M1 核心范围 |
| AscendC API/Runtime | 适用 | 同步、workspace、block_dim 和 PGO callback 生命周期保持 |
| Python/C++ 混合绑定 | 适用 | 只增加 MLIR 路由，不改变 CodegenResult Python 形态 |
| 构建与打包 | 适用 | MLIR 依赖受开关隔离，run 包需验证新产物 |
| 测试与覆盖率 | 适用 | 新增 bridge/ATT UT 和 Codegen/Schedule E2E |
| 性能与日志 | 适用 | 禁止 bridge 笛卡尔积膨胀；runtime solver 需基准 |
| 兼容性 | 适用 | M1/M2 保持 FusedScheduledResult 和 tiling ABI |

# 性能

## 编译时长

- bridge IR 规模必须为各 stage template 数量之和；
- 仅末端对真正需要的 composite key 做 specialization；
- StaticExact 采用合法性过滤 → hash 去重 → cost Top-K → compile Top-N；
- compare 模式只用于开发/CI，默认关闭；
- 不在每个 op/template 上增加默认 INFO 日志。

## 执行性能

- 动态 entry 每个 stage 至多一次 template 选择；
- 常量 tiling-key 路径经过 canonicalize 后不得保留动态分支；
- group parallel 的 block range 和 block_dim 与 legacy 等价；
- StaticExact 最终不保留 PlanId/template dispatch；
- 多 stage 性能以整次下发耗时衡量。

## 内存和产物大小

- 不保存调优数据库；
- StaticExact 只打包 winner；
- Dynamic Shape 只打包合法 Plan 和必要 template；
- 临时 MLIR、AscendC、kernel_meta、PGO 文件放入已忽略目录；
- Module ownership 使用 OwningOpRef，失败路径自动释放。

# 接口设计

## 新增/修改接口描述

bridge：

    FailureOr<SmallVector<mlir::OwningOpRef<mlir::ModuleOp>>>
    ScheduleToMlir(mlir::MLIRContext &context,
                   const ascir::FusedScheduledResult &result);

Plan 坐标保存在 Module 标准属性中，不新增 AF 容器类。

MLIR Codegen：

    FailureOr<codegen::CodegenResult>
    RunMlirCodegen(ArrayRef<mlir::ModuleOp> plans,
                   const MlirCodegenOptions &options);

    FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
    LinkPlanModules(ArrayRef<mlir::ModuleOp> plans,
                    const LegacyKernelAbi &abi);

ATT：

    FailureOr<TilingModel>
    BuildTilingModelAnalysis(mlir::ModuleOp plan,
                             const TargetCapability &target);

    FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
    BuildHostTilingModule(ArrayRef<mlir::ModuleOp> plans,
                          ArrayRef<TilingModel> models);

Schedule 终态：

    FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
    OriginGraphToMlir(const ascir::HintGraph &origin);

    FailureOr<SmallVector<ScheduleCandidate>>
    GenerateCandidates(mlir::ModuleOp origin,
                       const ShapeConstraints &shape,
                       const TargetCapability &target);

ScheduleCandidate 是调度 driver 的临时对象，只拥有 Transform Sequence 和结果 Module，不是 dialect op。

## 接口检查项

| 检查项 | 是否涉及 | 说明 |
|---|---|---|
| 接口评审 | 是 | Module schema、tiling ABI adapter、错误语义 |
| 文档 | 是 | stage/template/key schema 和 pass pipeline |
| 行为兼容 | 是 | M1/M2 与 legacy A/B |
| ABI/API | 是 | 对外 FusedScheduledResult、CodegenResult 不变 |
| 非法输入 | 是 | 空层级、key 越界、Shape 不合法、workspace 溢出 |
| 独立接口用例 | 是 | bridge、verifier、ATT analysis、host tiling |

# 软件设计

## 关键数据结构

### Plan Module schema

Module 属性：

- afir.plan_id；
- autofuse.legacy_node_id；
- autofuse.legacy_result_id；
- afir.stage_count；
- afir.template_counts_per_stage；
- autofuse.shape_mode；
- 迁移期 opaque score metadata。

private func 属性：

- afir.stage_id；
- afir.template_id；
- afir.double_buffer；
- 必要的 target/codegen metadata。

不保存 ScheduleGroup 对象。

### TilingModel

编译期 C++ analysis side table：

- Plan/stage/template 坐标；
- Shape vars；
- eq/leq/divisibility constraints；
- UB/core/workspace constraints；
- cost expression；
- var relation；
- group parallel 信息。

生命周期到 Host Tiling Module 生成完成为止。

### Transform Sequence

只在 M3 Schedule driver 中存在。应用到 Origin Module clone 后可以释放；调试时可序列化。

## 关键技术/算法

### Plan 完整性校验

VerifyPlanModule 必须检查：

1. 只有一个 public kernel entry；
2. stage_id 从 0 连续；
3. 每个 stage 至少一个 template；
4. 同 stage 的 template_id 唯一且连续；
5. template_counts_per_stage 与函数集合一致；
6. entry 对每个合法 key 恰好调用每个 stage 一次；
7. workspace producer 在 consumer 前；
8. group parallel 与 var_relations 不冲突。

### 确定性

- plan_id 按 legacy result 顺序生成；
- stage_id 按 schedule_groups 顺序生成；
- template_id 按 impl_graphs 顺序生成；
- 不依赖 unordered_map 遍历或对象地址；
- Transform hash 使用稳定序列化。

## 流程设计

### M1：Codegen

    legacy Schedule
      → ScheduleToMlir
      → Plan Module[]
      → MLIR Codegen
      → legacy ATT tiling
      → CodegenResult

完成标志：多 stage、多 template 的 key 映射、调用顺序、同步、workspace 与 legacy 一致。

### M2：ATT

    Plan Module[]
      → BuildTilingModelAnalysis
      → Host Tiling Module
      → Static：compile → optional PGO → winner
      → Dynamic：runtime GetTiling → PlanId + templates + params

完成标志：不再从 FusedScheduledResult 生成 host tiling 源码，PGO 文件行为保持。

### M3：Schedule

    OriginGraph
      → OriginGraphToMlir
      → DesignSpaceGenerator
      → Transform Sequence[]
      → Plan Module[]
      → M2 ATT
      → M1 Codegen

完成标志：MLIR-native 路径不再构造 FusedScheduledResult。

## 对子模块的修改

| 目录 | 修改 |
|---|---|
| mlir/bridge/ScheduleToAfir | 改为一 ScheduledResult 一 Module，补 entry 和属性 |
| mlir/pipeline 或新增同级目录 | 新增 MlirCodegenPipeline、VerifyPlanModule、PlanBundleLinker、dispatch lowering |
| compiler/py_module | 增加 off/on/compare 路由；对外返回不变 |
| att | M2 新增 MLIR analysis/host tiling 路径；legacy 文件不重构 |
| optimize | M3 新增 MLIR-native Schedule；legacy 文件不重构 |
| codegen | legacy 实现保持；不再把 :392 dump hook 当迁移入口 |

## 错误处理

### 系统错误

- MLIR context/module 创建失败：返回 Failure；
- translation/backend compile 失败：报告 plan_id、stage_id、template_id；
- PGO callback/runtime 失败：释放资源并回退 model score；
- 文件写入失败：只影响显式 dump/PGO 文件，不吞掉主路径错误。

### 接口错误

- multi-node 输入在 M1 返回 Unsupported 并走 legacy；
- composite key 超范围返回 InvalidArgument；
- Shape 约束不可满足时淘汰 Plan；
- 所有 Plan 不合法时 GetTiling 返回失败。

# 安全检查

| 红线 | 设计措施 |
|---|---|
| 外部输入作索引 | node/result/stage/template/key 使用前检查范围 |
| 整数溢出/除 0 | mixed-radix 乘积、tensor bytes、workspace、tile factor 全部 checked arithmetic |
| 资源生命周期 | Module 用 OwningOpRef；callback/stream/临时文件走显式清理 |
| ABI/API | M1/M2 不改公开结构体布局和 Python 返回 |
| 图等价性 | entry 校验每个 stage 恰好执行一次并保持数据/控制依赖 |
| 图确定性 | 显式稳定编号，不使用 unordered_map 顺序 |
| runtime 生命周期 | 异步 PGO callback 的 host/device buffer 保持到 stream 完成 |
| 非开放接口 | 只使用公开 AscendC/runtime 接口 |
| 日志 | template 循环不新增默认高频日志 |

# 兼容性检查

- ENABLE_AUTOFUSE_MLIR=OFF：构建和运行完全走 legacy；
- AF_MLIR_CODEGEN=off：行为不变；
- compare：返回 legacy 结果，只记录差异；
- M1/M2：FusedScheduledResult、CodegenResult、tiling data ABI 不变；
- M3 完成前：legacy optimize 保留；
- multi-node、暂不支持 op、cube 特殊路径：显式回退，不生成错误 Module。

# 测试设计

## 测试边界

输入边界：

- 1 result/1 stage/1 template；
- 1 result/N stage/M template；
- 多 result；
- workspace/var_relations；
- enable_group_parallel；
- 静态 Shape、动态 Shape；
- 空层级、key 越界、资源超限；
- multi-node fallback。

输出验证：

- Module 数量等于 ScheduledResult 数量；
- private func 数量等于 impl_graph 总数；
- key 到 template tuple 的映射与 AppendFuncCall 一致；
- 每个 key 执行全部 stage；
- AscendC、tiling_data、数值和性能与 legacy 对比。

## 测试用例设计

| 类别 | 关键测试项 | 方法 | 类型 |
|---|---|---|---|
| Bridge | Module 拆分、属性、entry | 检查 MLIR 文本和 verifier | UT |
| Dispatch | 2x3 templates 的 6 个 key | 与 AppendFuncCall golden 对比 | UT |
| Codegen | 多 stage workspace/sync | legacy 与 MLIR AscendC 对比 | UT/ST |
| ATT 动态 | 多 Shape 选择 Plan/template | 比较 tiling fields 和数值 | UT/ST |
| ATT 静态 | model/PGO winner 和裁剪 | 检查单 Module/单 dispatch | ST |
| Schedule | Transform replay 确定性 | hash、IR、数值一致 | UT/ST |
| 异常 | 空 stage、key 越界、UB 超限 | 检查错误和 fallback | UT |
| 性能 | 编译时长、IR 大小、运行耗时 | benchmark/legacy A/B | RDV |
| 交叉 | Python Codegen、run 包 | 既有 E2E | ST/E2E |

## 测试命令

实施后按阶段执行：

    sh build.sh -u --module=autofuse_framework -j 8
    sh build.sh -s --module=autofuse_e2e -j 8
    sh build.sh --pkg -j 8

MLIR 专项 target 名称在实现计划中确定，但所有构建并行度限制为 8。

# 验收标准

- [x] 实现名称由规划中的 `ScheduleToMlir` 调整为 `ScheduleToAfir`；当前对每个 `ScheduledResult` 生成独立 AFIR `ModuleOp`，仅用于调试 dump；
- [x] AFIR dialect 没有 schedule/group/result 容器 op；schedule/group/result 仍是 bridge 输入侧的 C++ 结构；
- [ ] Module entry 对每个合法 key 选择每 stage 一个 template，并执行全部 stage；
- [ ] 2x3 template 用例与当前 AppendFuncCall 的 6 个组合完全一致；
- [ ] 生产 MLIR Codegen 路由从 legacy `codegen.cpp` 中独立出来；当前只有受 统一 dump 前缀（`ascir::utils::GetDumpFilePrefix()`） 控制的调试 bridge，不属于生产 lowering；
- [x] 当前调试 bridge 不修改 legacy optimize/codegen 结果，`CodegenResult` 外部接口保持不变；
- [ ] M2 动态 Shape 在运行时 GetTiling，静态 Shape 在 compile/可选 PGO 后选 winner；
- [ ] ATT 迁移不增加 Autotune 数据库；
- [ ] M3 从 OriginGraph 生成合法 Transform Sequence，不从 FusedScheduledResult 反推；
- [ ] multi-node 未支持前明确回退 legacy；
- [ ] UT/ST/E2E、性能和打包验证通过；
- [ ] mixed-radix、Shape、workspace 运算通过溢出和边界检查。

# 最终结论

迁移后的编译链只需要三个核心对象：

    Transform Sequence
      生成方案

    ScheduledResult Module
      表达并编译完整方案

    Host Tiling Module
      静态或运行时选择方案和 tiling

`ScheduleGroup` 不成为 MLIR 容器。bridge 把它转换为 `stage_id` 和 entry 控制流。`AscGraph` 也不是独立下发任务，
而是某个 stage 的一个 template。一次实际执行会为每个 stage 选择一个 template，然后执行完整的 `ScheduledResult`。

迁移顺序是 Codegen、ATT、Schedule。前两阶段保持 legacy optimize 的输出不变，通过并行 MLIR 路径验证新实现；
第三阶段再用 `OriginGraph + Transform Sequence` 替换 legacy Schedule。
