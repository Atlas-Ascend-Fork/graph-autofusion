# AFIR reduce AllLoad 模板设计

> **接口迁移说明（2026-10-22）**：正文提及的 `AF_MLIR_PINNED_*` / `AF_MLIR_ENABLE_PLAN_PINNING` 是历史接口，现已移除；当前使用 ATT `AUTOFUSE_DFX_FLAGS` 的 `--force_schedule_result` / `--force_tiling_case`。候选编号须核对当前 shape 的 schedule dump。

> 状态：已实施（待评审）
>
> 日期：2026-10-01
>
> 文档类型：调度模板接入与覆盖补齐设计

> **本文推翻了一条在册结论。** `2026-09-20-two-dim-transfer-addressing-design.md:1777` 与
> `2026-09-21-afir-reduce-sum-ra-lowering-design.md:1191` 都把「AllLoad 模板未编」列为
> 「RCommon / RCore / AllLoad × AR / RA 六组合 E2E 矩阵」的阻塞项。2026-10-01 的 S0 探针
> （见 2.3）证明这条结论**已经过期**：AllLoad 候选不但能编，而且在五种形状上数值正确。
>
> 因此本方案的内容不是「实现 AllLoad 模板」，而是：**把已经工作的东西钉住、把真正缺的
> 那两块补上、把过期结论改写掉**。如果照旧结论去「实现」，会重写一条本来就对的路径——
> 这正是本文必须先摆探针数据的原因。

# 简介

## 目的

让 reduce 的 **AllLoad（全载）调度模板**在 MLIR 后端成为一条有回归保护、有明确适用域、
有正确文档结论的路径，并据此关闭「六组合 E2E 矩阵」里与 AllLoad 相关的那一格。

目标读者是 AFIR Analysis / AFIRToAscendC / reduce 调度模板与 CPU E2E 语料的开发和评审
人员。期望输出是可以按 S1～S4 直接提交的文件级方案。

## 范围

### 包含

- AllLoad 在调度侧的生成条件、签名与适用域的成文记录（此前散在三个文件里，无一处汇总）
- CPU E2E 语料：把 AllLoad 候选**钉住**（ATT `--force_schedule_result`），五种形状
- CPU E2E 语料：新增三维 `[A0, R, A1]`（ARA）注册，连带第一次覆盖 RA 下 **A 组连乘**
- lit：钉住 AllLoad 的下降形态（无 workspace、无跨轮累加、单 scratch）
- `is_reduce_mem_reuse` 在 MLIR 侧的语义归属的成文结论
- 改写三处过期结论（两份设计文档 + `docs/README.md` 的 S5 条目）

### 不包含

- **不改调度器**。AllLoad 的生成条件、轴分组与 loop_axis 选择全部照现状接受（理由见 3.4）
- **不改 AllLoad 的下降代码**。探针证明现有 `ReducePlan` / `TransferPlan` / `Reduce.cpp`
  已经覆盖它；本方案一行产品代码都不碰下降链路
- 不动 ArgMax / Softmax 两族（前者未实现，后者有独立的 `SoftmaxPatternFusionPass` 路径）
- 不碰 ATT `CheckKeyValid` 缺少 tiling_key 唯一性校验这一点（红线 7：见 2.4，只记录不顺手修）
- 不碰 910B 实机验收（本环境无卡）

# 总体概述

## 软件概述

### 项目介绍

reduce 在 `autofuse/optimize` 里有三个调度模板（`optimize/optimize.h:33-37`）：

| 模板 | 枚举 | 含义 |
|---|---|---|
| RCommon | `kCommon` | R 轴切块，跨轮累加；必要时按 reduce 切图，中间结果过 GM workspace |
| RCore | `kRCore` | 在 RCommon 之上把 R 轴也拿去分核，拆成两相，相间过 workspace |
| **AllLoad** | `kAllLoad` | **R 轴整根一次进 UB**，单相、不跨轮、不切图、不要 workspace |

`ReducePartitionCaseGenerator::GeneratorTask`
（`optimize/task_generator/reduce_schedule_case_generator.cpp:339-352`）的顺序是
General → （非 force_all_load 时）RCore → **AllLoad 恒执行**。也就是说：只要图通过
`CanFullLoadReduceFuse`，候选列表末尾**一定**多出一个 AllLoad plan。

这一点决定了 AllLoad 不是「可选优化」而是**必须能编**：MLIR 侧任一候选下降失败即
`MlirCodegenStatus::kFailed`（`bridge/CodegenOrchestration/CodegenOrchestration.cpp:355`），
而 `kFailed` 在 `codegen/codegen.cpp:401` 直接变成 `af::FAILED`——整条编译失败，**不会**
退回 legacy（只有 `kNotSupported` 才退）。AllLoad 编不出来 = 这张图在 MLIR 后端编不出来。

### 产品环境介绍

- 上游：`autofuse/optimize` 的调度器产出 `ascir::FusedScheduledResult`，其中
  `ScheduledResult::is_reduce_mem_reuse` 是 AllLoad 的唯一标记（`optimize/optimize.cpp:1094`）
- 本层：`bridge/ScheduleToAfir`（:1077 透传）→ `bridge/AfirBuilder`（:1853 写成模块属性
  `afir.is_reduce_mem_reuse`）→ `dialect/AFIR/Analysis/ReducePlan` → `conversion/AFIRToAscendC`
- 下游：ATT 生成 host tiling；聚合入口按 `graph<N>_tiling_key` 分派到各 plan
- 外部接口：**无变化**。不新增 ABI、tiling schema 字段、环境变量或 runtime 调用

## 软件功能

1. AllLoad 候选在 MLIR 后端可编译、数值正确，且**有用例钉住**，不再依赖「运行期恰好没
   选到它」而侥幸通过。
2. AllLoad 的适用域（哪些 shape 会产生 AllLoad 候选、哪些不会）在文档里可查，且与代码
   判据逐条对应。
3. AllLoad 下的 RA 形态（只能由 ≥3 维 ARA 产生）进入语料，顺带覆盖 RA `srcShape` 公式里
   此前无语料的 **A 组连乘**项。
4. 三处过期结论改写为实测状态。

## 设计约束

- CANN Toolkit 9.2.0；本方案不引入版本相关行为
- **不改调度器**：AllLoad 的轴分组（`optimize/autoschedule/tiling_group.cpp:538`
  `GenReduceTilingGroupFullLoad`）与 `vectorized_axis` 组装（`schedule.cpp:714` 的
  `reduce_template_ != kAllLoad` 分支）同时服务 legacy codegen，改动风险远大于收益
- **不改下降代码**：产物必须逐字节不变（AR 既有 44 份产物的不变性是 2026-09-20 方案 S3
  以来的硬约束）
- 新增用例只许进 CPU E2E 与 lit；不新增日志，不新增默认开启的诊断
- 不新增硬编码芯片类型 / 框架类型判断

## 假设和依赖关系

- 依赖 CPU E2E harness（`test/e2e/cpu/AddCpuCase.cmake`）透传 ATT DFX 环境变量。
  使用 `AUTOFUSE_DFX_FLAGS="--force_schedule_result=N"`，不再注入 MLIR pinning 总闸。
- 尾维为 1 的 ARA 会退化为 AR：`amin(4,15,1), dim=1` 的 AllLoad 是 plan 2，固定回归为 `allload_ar_tail1`。固定 plan 3 的 ARA fuzz 排除两端 A 维为 1 的形状；不能只按输入 rank 选择编号。
- 假设 AllLoad 在候选列表中的下标稳定（当前恒为「General 的 plan 数 + RCore 的 plan 数」
  之后）。**这是一个脆弱假设**，缓解措施见 3.3 的「plan 下标漂移」一节
- 三维 ARA 的 AllLoad 走 RA 形态这一结论依赖调度器把 R 排在 `vectorized_axis` 首位；
  已由探针实测（2.3 第 4 条），但**不是**调度器的书面契约，故 lit 用例要把它钉住

# 需求分析与设计

## 整体介绍

AllLoad 在 AFIR 上的签名由三个互相独立的特征共同确定，缺一不可：

| 特征 | 取值 | 来源 |
|---|---|---|
| 调度图名 | `..._B0Y0_S<n>G0C0`，**没有 `R<k>` 段** | R 轴不占 UB 切分位 |
| 模块属性 | `afir.is_reduce_mem_reuse = true` | `optimize.cpp:1094` 只在 `kAllLoad` 置真 |
| `ReducePlan` | `accumulationAxes` 为空 → `isAccumulating()` 假 | R 轴整根落在 `loop_axis` 之内 |

第三条是下降侧真正依赖的那条：`loop_axis` 之内的归约维由一次 API 调用算完，不进累加表
（`dialect/AFIR/Analysis/ReducePlan.cpp` 的累加轴收集循环），于是不需要第二块 scratch、
不需要跨作用域逃逸、不需要 workspace。AllLoad 因此是三个模板里**对下降侧要求最低**的一个。

## 适用域：哪些图会产生 AllLoad 候选

`CanFullLoadReduceFuse`（`reduce_schedule_case_generator.cpp:863`）要求图里**每个** reduce
都满足两个谓词之一（`kMaxFullLoadAxisSize = 3`，同文件 :33）：

| 谓词 | 条件 | 直观含义 |
|---|---|---|
| `IsLegacyFullLoadReduce`（:40） | 输出轴数 ≤ 3 **且** `in.repeats[0] == out.repeats[0]` | 首轴不被归约的 ≤3 维图 |
| `IsTailReduceFullLoad`（:48） | 输入轴数 ≥ 3 **且** 除末轴外逐维 repeats 相等 **且** 归约落在末轴（Softmax 豁免） | ≥3 维的末轴归约 |

由此推出三条**适用域结论**（全部由 2.3 的探针逐条实测，不是推断）：

1. **二维 AR**（输出 `[M, 1]`）：`in[0]==out[0]` 成立 → 有 AllLoad。
2. **二维 RA**（输出 `[1, N]`）：`in[0] != out[0]`，轴数也不足 3 → **没有 AllLoad**。
   这一条直接否掉了「六组合矩阵」里「AllLoad × 二维 RA」那一格——它不是没测，是**不存在**。
3. **三维 ARA**（`[A0, R, A1]`，归约中间维）：输出轴数 3、`in[0]==out[0]` → 有 AllLoad；
   而它在 `vectorized_axis` 上是 `[R, A_inner, A1]` → **RA 形态**。所以矩阵里的
   「AllLoad × RA」只能、且恰好可以由 ARA 覆盖。
4. **四维 ARAR**：输出 4 维 > 3，且轴 1 是 R 使「除末轴外逐维相等」不成立 → **没有 AllLoad**。

## S0 探针结论（2026-10-01，CPU E2E + AFIR dump 实测）

探针做法：对 `test/e2e/cpu/cases/load_sum_store` 的生成器**不钉 plan**运行，
`AUTOFUSE_DFX_FLAGS="--codegen_compile_debug=true;--debug_dir=..."` 打开 dump，按
`is_reduce_mem_reuse = true` 认出 AllLoad plan；再把该 plan 钉住跑数值。

### 1. AllLoad 候选确实在，且已经能编

| 用例 | 任务数（General/RCore/AllLoad） | AllLoad plan | 生成器退出码 |
|---|---|---|---|
| `load_sum_store_16x32`（AR） | 1 / 2 / **3** | plan 2，`B0Y0_S2G0C0`，`mem_reuse = true` | 0 |
| `load_sum_mul_store_8x1024`（AR + 切图） | 1 / 2 / **3** | plan 2，同上 | 0 |
| `load_sum_store_ra_37x16`（二维 RA） | 1 / 2 / **2** | **无**（见适用域结论 2） | 0 |
| `load_sum_store_arar_case0/3`（四维） | 1 / 2 / **2** | **无**（结论 4） | 1（另一缺陷，见 2.4） |
| ARA 探针 `[4,32,8]` | 1 / 2 / **3**（共 4 个 plan） | plan 3，`B0Y0_S3G0C0`，`mem_reuse = true` | 0 |

### 2. 二维 AR AllLoad 的 AFIR 形态

`afir.sum` 的 descriptor：`axis = [4, 5, 3, 1]`、`vectorized_axis = [3, 1]`，
destination `vectorized_strides = [1, 0]`、operand `[32, 1]`；`sched = <axis = [4,5,3,1],
loop_axis = 5>`。轴 1（`z1`，size 32）**整根未切**，直接作为向量化内维。

`buildReducePlan` 的判定：`accumulationAxes` 为空（轴 4/5 都派生自 A 轴 0）→ 不累加；
`startsReduced = false` → **AR**；`first = [3]`、`last = [1]`、`tailLast = true`；
给了两块 scratch 而只需要一块。

### 3. 二维 AR AllLoad 的下降产物

```c++
DataCopyPadExtend<float>(v108, v107, 1, 32 * A_tile, 0, 0);          // 整块 A_tile × 32
ReduceInit<float, kReduceOpSum, true>(v113, A_actual, align8(32), align8(32), 32);
uint32_t shape[] = {A_actual, align8(32)};
AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(v110, v113, tmp, shape, true);
DataCopyPadExtend<float>(v129, v128, 1, A_actual, 0, 0);
```

一次搬入、一次归约、一次搬出，没有 workspace、没有跨轮累加、没有第二块 scratch 的使用。
二维 DMA 被正确折成一维（R 整根且连续时成立）。

### 4. 三维 ARA AllLoad 走 RA，并第一次用到 A 组连乘

`afir.sum` 的 `vectorized_axis = [1, 4, 2]`，destination `vectorized_strides = [0, 8, 1]`、
operand `[(8 * z0t_size), 8, 1]`：位置 0（R）被归约、位置 1/2（A）保留 →
`startsReduced` → **RA**，`first.axes` **两根**。

发出的 `srcShape` 是 `{32, align8(8) * z0t_size}`，即
`{Π R_actual, align(A_inner_nominal) × Π A_outer_nominal}`——正是
`2026-09-21-afir-reduce-sum-ra-lowering-design.md` 验收表里注明「连乘部分尚无用例覆盖」
的那一项。按 RA 契约没有 `ReduceInit`，发的是 `ReduceSum<..., RA, false>`。

### 5. 数值：五条钉住 AllLoad plan 的 CPU E2E 全部通过

| 形状 | 覆盖意图 | 结果 |
|---|---|---|
| `[16, 32] -> [16, 1]` | 基准 AR，A/R 都对齐 | 通过 |
| `[16, 37] -> [16, 1]` | **R 尾块**不对齐 | 通过 |
| `[17, 32] -> [17, 1]` | **A 尾块**不对齐 | 通过 |
| `[8, 1024] -> [8, 1]` | 大 R（整根 1024 入 UB） | 通过 |
| `[4, 32, 8] -> [4, 1, 8]` | ARA → RA + A 组连乘 | 通过 |

三条钉 plan 2 的用例均经 dump 复核 `is_reduce_mem_reuse = true`，确认钉到的就是 AllLoad。

### 6. `is_reduce_mem_reuse` 在 MLIR 侧没有读点，而这是对的

全仓检索：该字段的**唯一**实际作用点是
`optimize/buffer_allocate/buf_que_allocator.cpp:475 / :582`——它决定 reduce 之后的张量
`is_reusable`。也就是说「复用」的决策在调度器里**已经做完**，MLIR 读到的是它的结果
（`afir.buffer_desc` 的 `buffer_id` / `reuse_id`）。

结论：MLIR 侧不需要、也**不应该**再解释这个布尔值。它作为模块属性存在的意义只有两条——
进指纹（`common/fingerprint/schedule_fingerprint.cpp:109`）与人读 dump 时认模板。
本方案把这条结论写进注释，防止后来者把它当成一个「该实现但没实现的分支」。

## 顺带详查：`duplicate tiling_key in stage 0` 不是现网缺陷

初稿曾在此处记下「四维 ARAR 用例不钉 plan 时 plan 0 就挂在
`GenKernelEntry: duplicate tiling_key in stage 0`」。**这条记录是错的**，改正如下。

### 当时观到的现象是怎么来的

探针跑的是 build 树里的
`MlirE2eCpu_load_sum_store_arar_case0_generator` / `_case3_generator`。这四个
`arar_case0..3` 注册**已经被删除**（现在只剩一条钉 plan 0、不钉 template 的
`load_sum_store_arar`，见该目录 `CMakeLists.txt`），但 CMake 不会删掉已移除 target 的
可执行文件，那四个 2026-09-29 的二进制**作为孤儿留在 build 树里**。我直接按路径跳过
ctest 跑了它们，跑到的是一份早已不存在的注册。

用**当前**的 `load_sum_store_arar` 生成器复测，钉 plan 0 与不钉 plan 两种跑法都退出 0，
且 stage 0 的四个 template 拿到的是 `tiling_key = 0 / 1 / 2 / 3`：

| 跑法 | plan 0 stage 0 的 tiling_key | 退出码 |
|---|---|---|
| `AUTOFUSE_DFX_FLAGS="--force_schedule_result=0"` | 0, 1, 2, 3 | 0 |
| 不钉 plan（5 个候选全编） | 0, 1, 2, 3 | 0 |

所以 ARAR 在现网上没有这个问题，「钉 plan 掩住了缺陷」的判断也不成立。

### 但这个诊断本身是真的，且根因值得记下

`collectStageTilingKeys`（`conversion/AFIRToAscendC/GenKernelEntry.cpp:248`）在同一个 stage 内
看到重复 key 就报这条诊断，已由
`test/lit/conversion/afir-to-ascendc-gen-kernel-entry-invalid.mlir:87` 钉住。触发它的唯一
路径是「图自带 tiling_key」，而那条路径的根因是**粒度不匹配**：

1. `AscGraphAttr::tiling_key` 默认 `-1`
   （`inc/graph_metadef/graph/ascendc_ir/ascendc_ir_core/ascendc_ir_def.h:438`）。
2. ATT 与 MLIR 两处用**同一个三元式**让图自带的 key 覆盖自动编号：
   - ATT：`tiling_key = graph.GetTilingKey() >= 0 ? graph.GetTilingKey() : tiling_key`
     （`att/gen_model_info/gen_model_info.cpp:229` 与 `:441`，其中 `tiling_key` 是自增计数器）
   - MLIR：`info.tiling_key = graph_tiling_key >= 0 ? graph_tiling_key : templateId`
     （`bridge/ScheduleToAfir/ScheduleToAfir.cpp:751`）
3. 图级 key 的粒度是「一张 impl graph 一个」，而分派需要的粒度是「一个 stage 内每个
   template 一个」。一个 stage 的 N 个 tiling case 是从同一张图 `CopyFrom` 出来的，而
   `DoCopyAscGraphAttrImpl`（`graph_metadef/graph/ascendc_ir/core/ascendc_ir.cc:619`）是整个
   `AscGraphAttr` 的序列化/反序列化，`tiling_key` 就在这份 proto 里
   （`graph_metadef/proto/af_ir.proto:263`）——于是 N 张图拿到**同一个** key。
4. ATT 的 `CheckKeyValid`（同文件 `:214`）只校「全设或全不设」，**不校互不相同**，所以
   ATT 侧不报错，直接生成 N 个同名 `TilingCase<K>Impl` → C++ 重定义。

即：同一个三元式在两侧退化出两副完全不同的面孔——ATT 那边是生成代码重定义，MLIR 这边是
`duplicate tiling_key in stage 0`。MLIR 的检查是两侧唯一把它显式判出来的地方，这是对的；
缺的是 ATT `CheckKeyValid` 的唯一性校验。

现网不踩到是因为产品代码**没有任何地方调 `SetTilingKey`**（全仓检索只有测试与
`py_module` 的 UT），key 恒为 `-1`，两侧都走自动编号。`load_sum_store/Case.cpp:71-77`
的注释已经记下了这一条（「这里**不设** tiling key」），说明当时是在语料里手动设过 `0`
才踩到的，已通过「不设」规避。

**结论**：不开缺陷单。建议的改进是给 ATT `CheckKeyValid` 补一条唯一性校验，把
「图自带 key 且重复」从「生成重定义的 C++」提前成一条可读错误；该改动在 ATT 侧，
与本方案无关，按红线 7 只记录。

### 还有一条真的需要防

虽然 ARAR 没有这个问题，「钉 plan 既是工具也是一种遮蔽」这条顾虑仍然成立：钉了 plan
的注册永远编不到其余候选。本方案因此在需求 1 里同时保留「钉住 AllLoad」与
「至少一条不钉 plan 的 AllLoad 语料」两种注册。

### build 树里的孤儿二进制是一个真实的误导源

删掉一条 `add_mlir_e2e_cpu_case` 注册后，CMake 会从 `CTestTestfile.cmake` 里拿掉它，但
**不会删掉已经链好的 `MlirE2eCpu_*_generator`**。绕过 ctest 按路径直接跑生成器是诊断
reduce 候选的常用手法（本文 2.3 的探针就是这么做的），所以这个坑会反复踩。
**按路径跑生成器前先核对 `CTestTestfile.cmake` 里还有没有这条用例。**

## 功能需求

### 功能需求 1：把 AllLoad 候选钉进 CPU E2E

1. **介绍**：新增五条 CPU E2E 注册，各自钉住所属图的 AllLoad plan，覆盖 2.3 第 5 条的五种
   形状；同时保留至少一条**不钉 plan** 的 AllLoad 语料，使「全候选一起编」这条路也有人走。

2. **输入**：`test/e2e/cpu/cases/load_sum_store/CMakeLists.txt` 的注册参数
   （`ENVIRONMENT "AUTOFUSE_DFX_FLAGS=--force_schedule_result=<n>"`、`OUTPUT_SHAPE`、`INPUT_SHAPES`）；
   开闸由 harness 统一注入，不进逐条注册。
   输入形状限 f32、静态、≤3 维。

3. **处理**：
   - plan 下标由 2.3 的办法确定（不钉 plan 跑一遍，按 `is_reduce_mem_reuse = true` 认）。
   - **plan 下标漂移**是已知风险：调度器一旦多出 / 少掉一个候选，钉住的下标就指向别的
     plan，而用例仍会通过（它只验数值，不验模板）。缓解：每条 AllLoad 注册的 CMake 注释
     里写明「该下标指向 AllLoad」的判据与复核命令；并由需求 3 的 lit 用例独立钉住形态，
     使模板本身的回归不依赖 plan 下标。
   - 不钉 plan 的那条注册同时承担「候选全编」的保护。

4. **输出**：新增 5 条 `ctest` 用例（`mlir_e2e_cpu` 标签），`ctest --test-dir
   <build>/autofuse/mlir` 的用例数由 85 增至 90。无产物格式变化。

### 功能需求 2：三维 ARA 语料

1. **介绍**：`load_sum_store` 的建图与参考实现当前只认 2 维和 4 维
   （`Case.cpp` 的 `reduce_axes` 分派、`KernelTest.cpp:37` 的 `ASSERT_TRUE`）。补上 3 维
   分支：`[A0, R, A1]`，归约轴 `{1}`。

2. **输入**：`INPUT_SHAPES "[4, 32, 8]"`、`OUTPUT_SHAPE "[4, 1, 8]"`。

3. **处理**：
   - `Case.cpp`：`input_shape.size() == 3U` → `reduce_axes = {1U}`。
   - `Case.cpp` 的 `CheckGeneratedKernel`：3 维的 AR/RA 判据与 2 维不同（2 维按输出形状
     判，3 维的 ARA 恒为 RA），因此 3 维分支断言 `Pattern::Reduce::RA` **且**不含
     `ReduceInit<`——与 RA 契约一致，顺手把 2.3 第 4 条的实测形态钉死。
   - `KernelTest.cpp`：新增 3 维参考实现，三重循环沿中间维求和；输入仍用
     `index % 7 - 3`，保证每个部分和在 f32 下精确可表示，故用 `EXPECT_FLOAT_EQ`。
   - **不声明** `TILE_ACCESSOR` / `COVERS`：ARA 的 AllLoad 单 group，扁平
     `AutofuseTilingData` 可用，但 tile 字段名与 2 维不同，为免注册参数分叉，本期不声明。

4. **输出**：1 条新 `ctest` 用例（含在需求 1 的 5 条内）。

### 功能需求 3：lit 钉住 AllLoad 的下降形态

1. **介绍**：E2E 只验数值，验不出「它是不是仍然走 AllLoad 那条路」。用 lit 在 IR 层面钉
   三件事：不发 workspace 相关指令、不发跨轮累加（无第二块 scratch 的使用、无
   `EntireTailFold` 一类的跨 tile 合并）、AR 与 RA 两种 AllLoad 形态各发对应的 layout。

2. **输入**：两份手写 AFIR plan 模块，逐字取自 2.3 第 2 / 第 4 条的实测 descriptor
   （`afir.is_reduce_mem_reuse = true`、`vectorized_axis = [3, 1]` 与 `[1, 4, 2]`）。

3. **处理**：放在 `test/lit/conversion/`，与既有
   `afir-to-ascendc-reduce-sum.mlir` / `-ra.mlir` 并列，命名
   `afir-to-ascendc-reduce-allload-ar.mlir` / `-ra.mlir`。`CHECK` 行只钉形态判据，不钉
   SSA 编号。

4. **输出**：2 条新 lit 用例；`check-lit` 由 188 增至 190。

### 功能需求 4：改写过期结论

1. **介绍**：三处在册文字把 AllLoad 说成「未编」，会让接手人去重写一条本来就对的路径。

2. **输入 / 处理**：

| 位置 | 现状文字 | 改成 |
|---|---|---|
| `design/2026-09-20-...:60` | AllLoad「轴数更少，落进同一套逻辑，无额外要求」 | 保留，补一句「已由 2026-10-01 探针实测证实，并列出 AR / ARA 两种实测形态」 |
| `design/2026-09-20-...:1290, 1772, 1777` | 六组合矩阵阻塞于「AllLoad 模板未编」 | AllLoad 两格改为「已通过（CPU E2E 五形状）」；**「AllLoad × 二维 RA」一格注明不存在**，由 ≥3 维 ARA 替代 |
| `design/2026-09-21-...:1191` | 把 AllLoad 列为 S5 的阻塞 | 删去 AllLoad，只留 910B 与 RCore tiling_key |
| `docs/README.md:50` | 「现在卡在 910B 实机、RCore 的 tiling_key 命名分歧与 AllLoad 模板未编」 | 去掉 AllLoad，并指向本文档 |

3. **输出**：文档变更；`docs/README.md` 的「当前设计」表新增一行指向本文。

## 非功能需求

### 可维护性

- 不新增产品代码，故不引入新的维护面。唯一的新增注释落在 `AfirBuilder.cpp:1853`
  `afir.is_reduce_mem_reuse` 的写入处，写明「该标记在 MLIR 侧无读点，决策已在
  `buf_que_allocator` 做完」，防止被当成未实现分支。
- 五条 AllLoad 注册的 CMake 注释统一写清「plan 下标为何是这个」与复核命令，避免下标漂移
  后无人看得懂。

### 可测试性

- 测试入口照 `af-mlir-test-invocation`：`ctest --test-dir <build>/autofuse/mlir` 与
  `cmake --build <build> --target check-lit`。
- 桩：CPU E2E 用 `MlirE2eCpuTilingStub` + `tikicpulib`，与既有 reduce 用例一致，无新桩。
- 覆盖目标：AllLoad × {AR, RA} × {对齐, A 尾块, R 尾块, 大 R} 四维形状组合取五点；IR 层
  形态由 lit 独立钉住，不依赖 plan 下标。

### 可移植性

- 不依赖芯片型号：探针与用例的 soc 取 harness 默认值；AllLoad 的判据全在 repeats / stride
  上，无平台分支。
- 不新增 Python / CANN 版本依赖。

### 可靠性

- 不新增内存申请、不新增资源生命周期；AllLoad 路径比 RCommon **少**一块 scratch 的使用、
  少一段 workspace 交接，错误面是收缩的。
- 错误处理不变：下降失败仍走 `kFailed` → `af::FAILED`。本方案不新增失败分支。
- 已知脆弱点（plan 下标漂移）在需求 1 里给了缓解，且不会造成**静默算错**：漂移后用例仍
  验数值，错的只是「它测的是哪个模板」，由 lit 兜住。

### 特性交叉影响

按 `cross_feature_check.md` 逐项：

| 场景 | 适用性 | 分析说明 |
|------|--------|----------|
| SuperKernel Python 接口 | 不适用 | 不碰 `super_kernel/`、不改选项解析、不动 wheel 内容 |
| SuperKernel C++/AOT 接口 | 不适用 | 不碰 `libascendsk.so`、无对外符号与结构体变化 |
| Autofuse 图优化 | 不适用 | **不改调度器**；AllLoad 的生成条件与轴分组照现状接受（设计约束第 2 条） |
| Autofuse Codegen/Backend | 适用（只增测试） | AllLoad 是 codegen 的一条候选路径，本方案新增 E2E 与 lit 覆盖，**不改下降代码**，AR 既有产物逐字节不变 |
| AscendC API / Runtime 交互 | 不适用 | 不新增 `rt*` / `aclrt*` / AscendC 调用；AllLoad 发的 `ReduceSum` / `DataCopyPad` 是既有调用点 |
| Python/C++ 混合绑定 | 不适用 | 不碰 `autofuse/compiler/py_module/` |
| 构建与打包 | 适用（局部） | 新增 `add_mlir_e2e_cpu_case` 注册与 lit 文件，不改 `build.sh`、不改安装路径与 run 包内容 |
| 测试与覆盖率 | 适用 | 新增 5 条 CPU E2E、2 条 lit；见测试设计 |
| 性能与日志 | 适用（编译期） | 新增 5 条用例使 `ctest` 用例数 85→90，构建与测试时长见「性能」；**不新增任何日志** |
| 兼容性 | 不适用 | 无 API/ABI、配置、产物布局与脚本参数变化 |

## 性能

### 编译时长

每条 CPU E2E 用例在构建期要跑一次生成器（调度 + bridge + 下降 + ATT）并编译一个 kernel。
实测单条用例的生成器约 1.3～1.5 s，加上 kernel 编译，5 条新增用例对全量构建的增量在
分钟量级以内，与既有 reduce 用例同量级。不涉及图遍历或字符串处理的新增热点。

### 执行性能

不改产品代码，kernel 指令序列不变，故执行性能无影响。顺带记录 AllLoad 自身的性能性质
（来自 2.3 第 3 条，不是本方案的改动）：相对 RCommon 少一轮循环、少一次 UB 间累加、
在切图场景下少一段 GM workspace 往返——这正是调度器恒生成该候选的原因。

### 内存和产物大小

新增 5 个测试可执行文件与 2 个 lit 文件，只在 build 树；run 包与 wheel 内容不变。
常驻内存无变化。

## 接口设计

### 新增/修改接口描述

无新增或修改的对外接口。内部变更仅为测试语料与注释。

### 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|--------|----------|----------|------|
| 接口说明 | 是否需要接口评审 | 否 | 无对外接口变化 |
| 接口说明 | 是否需要补充文档 | 是 | 本文 + 三处过期结论改写 + README 索引 |
| 接口兼容 | 行为是否兼容 | 是（完全兼容） | 不改产品代码，产物逐字节不变 |
| 接口兼容 | ABI/API 是否兼容 | 是 | 无符号、结构体、枚举变化 |
| 接口约束 | 约束不满足时是否清晰报错 | 是 | AllLoad 不适用的图不产生该候选；下降失败仍报 `kFailed` 并附 `buildReducePlan` 的具体原因（e235db98 之后） |
| 接口测试 | 是否需要独立接口用例 | 否 | 覆盖落在 E2E 与 lit |

## 软件设计

### 关键数据结构

无新增数据结构。涉及的既有结构及其在 AllLoad 下的取值：

| 结构 | AllLoad 下的取值 | 说明 |
|---|---|---|
| `ScheduledResult::is_reduce_mem_reuse` | `true` | 唯一的模板标记；作用点在调度器，MLIR 不读 |
| `ReducePlan::accumulationAxes` | 空 | → `isAccumulating()` 假，这是下降侧真正依赖的判据 |
| `ReducePlan::accumulationScratchIndex` | `nullopt` | 单块 scratch 够用 |
| `ReducePlan::pattern` | 二维 AR → `AR`；三维 ARA → `RA` | 由 `vectorized_strides` 首项是否为 0 定 |
| `ReducePlan::first.axes` | AR 一根；ARA **两根** | 两根那条是 RA 连乘公式的首个语料 |

### 关键技术/算法

不新增算法。本方案的技术内容是**把既有算法在 AllLoad 下的取值固定住**，即 2.3 的五张表。
复杂度无变化。

### 流程设计

- 主流程：调度器产 AllLoad 候选 → bridge 建 AFIR → `buildReducePlan` 判出不累加 →
  `Reduce.cpp` 按 pattern 发 layout → 聚合入口按 tiling_key 分派。全部为既有流程。
- 异常流程：`CanFullLoadReduceFuse` 不成立 → 不产生 AllLoad 候选（不是错误）；
  `buildReducePlan` 失败 → 能力查询报 `UnsupportedShape`
  （`conversion/AFIRToAscendC/Implementations/Reduce.cpp:260`），LoopGenerate 报
  `InvalidSchedule` 并附具体原因。
- 回滚：本方案全部为测试与文档，回滚即删除新增用例与还原文档。

### 对子模块的修改

| 目录 / 文件 | 修改点 |
|---|---|
| `test/e2e/cpu/cases/load_sum_store/CMakeLists.txt` | 新增 5 条注册（4 条二维钉 plan + 1 条三维），每条带下标判据注释 |
| `test/e2e/cpu/cases/load_sum_store/Case.cpp` | `reduce_axes` 增 3 维分支；`CheckGeneratedKernel` 增 3 维（恒 RA、无 `ReduceInit`）分支 |
| `test/e2e/cpu/cases/load_sum_store/KernelTest.cpp` | 放开 `ASSERT_TRUE` 的轴数白名单；新增 3 维参考实现 |
| `test/lit/conversion/afir-to-ascendc-reduce-allload-{ar,ra}.mlir` | 新增 2 份 |
| `bridge/AfirBuilder/AfirBuilder.cpp:1853` | **仅注释**：写明该标记在 MLIR 侧无读点及原因 |
| `docs/design/2026-09-20-...`、`docs/design/2026-09-21-...`、`docs/README.md` | 改写过期结论 + 索引本文 |

### 错误处理

#### 系统错误

无新增内存申请、文件操作或 runtime 调用，故无新增系统错误分支。CPU E2E 既有的
`GmAlloc` 失败处理照用（`KernelTest.cpp` 已有 `FAIL()` 分支并释放已分配缓冲）。

#### 接口错误

对外错误码不变。AllLoad 相关失败的用户可见信息沿用既有两条：能力查询的
`UnsupportedShape` 与 LoopGenerate 的 `InvalidSchedule: <具体原因>`。

## 安全检查

按 `编码红线.md` 逐条：

| 红线 | 结论 |
|---|---|
| 敏感信息 / 公网地址 / 芯片类型 / 框架类型硬编码 | 不涉及。新增用例的 soc 取 harness 默认值，无芯片分支 |
| 外部输入作为索引 / 长度 / 偏移 / 申请大小 | 不涉及。新增代码只有测试参考实现，索引来自编译期常量形状 |
| 整数溢出 / 反转 / 除 0 | 不涉及新增计算。下降侧既有的 `multiplyChecked` / `alignUp` 溢出断言在 2.3 第 3/4 条的产物里可见，已生效 |
| 资源生命周期与异常分支释放 | 测试参考实现沿用既有 `GmAlloc` / `GmFree` 配对；AllLoad 路径比 RCommon 少一段 workspace 生命周期 |
| ABI/API 兼容 | 无变化 |
| 图改写等价性与确定性 | 不改图、不改 pass |
| runtime/aclrt/AscendC 生命周期约束 | 不新增调用 |
| **红线 7（禁止无理由扩大修改范围）** | 2.4 的 ATT `CheckKeyValid` 唯一性校验与两处既有未提交遗留（`DataMovement.cpp`、`TransferPlan.h` 的注释改动）**只记录不处理** |

## 兼容性检查

不影响已有脚本参数、配置文件、安装目录、动态库、Python 包、旧产物或用户代码。
AR 既有 44 份产物要求逐字节不变——本方案不改产品代码，该不变性自动成立，并在验收标准里
留一条实测。

## 测试设计

### 测试边界

- 入口：`ctest --test-dir <build>/autofuse/mlir`（CPU E2E + UT）与
  `cmake --build <build> --target check-lit`（lit）
- 出口：数值比对（`EXPECT_FLOAT_EQ`）与 IR / kernel 文本比对（`CHECK` / `EXPECT_NE`）
- 打桩：`MlirE2eCpuTilingStub`（host tiling）、`tikicpulib`（CPU 侧 AscendC 仿真）
- 验证数据：`input[i] = i % 7 - 3`，使每个部分和在 f32 下精确可表示

### 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|----------|------------|----------|----------|
| 功能 | AllLoad AR 基准 `[16,32]` 数值正确 | 钉 AllLoad plan 的 CPU E2E | E2E |
| 功能 | AllLoad ARA → RA `[4,32,8]` 数值正确，且覆盖 A 组连乘 | 钉 AllLoad plan 的 CPU E2E | E2E |
| 功能 | AllLoad 发 AR / RA 对应 layout，RA 不发 `ReduceInit` | lit `CHECK` | lit |
| 异常 | AllLoad 下不出现 workspace 交接与跨轮累加 | lit `CHECK-NOT` | lit |
| 异常 | 二维 RA 不产生 AllLoad 候选（适用域结论 2） | 不钉 plan 跑生成器，核 dump 无 `mem_reuse = true` 的 plan | 手工探针（结论入文档） |
| 边界 | R 尾块 `[16,37]`、A 尾块 `[17,32]`、大 R `[8,1024]` | 钉 AllLoad plan 的 CPU E2E | E2E |
| 性能 | AR 既有 44 份产物逐字节不变 | 生成前后 `diff` | 手工验证 |
| 兼容性 | 既有 85 条 ctest 与 188 条 lit 不回归 | 全量跑 | E2E/lit |
| 特性交叉 | 不钉 plan 时 AllLoad 与其余候选同时编进一个 kernel | 保留一条不钉 plan 的 AllLoad 语料 | E2E |

### 测试命令

```bash
# 容器入口（本机 macOS + colima）
AF_MLIR_DEV_IMAGE=autofuse-mlir-dev:arm64 AF_CANN_VOLUME=af-cann-910 \
AF_CANN_INSTALL_PREFIX=/usr/local/Ascend/cann-9.2.0 \
autofuse/mlir/scripts/enter_dev_container.sh -- bash -c '
  cd /workspace &&
  cmake -S . -B build &&
  cmake --build build -j 4 &&
  ctest --test-dir build/autofuse/mlir --output-on-failure -j 4 &&
  cmake --build build --target check-lit -j 4'
```

预期：`ctest` 90/90 通过（现状 85），`check-lit` 190/190 通过（现状 188）。

复核某条注册钉到的是不是 AllLoad（plan 下标漂移时用）：

```bash
AUTOFUSE_DFX_FLAGS="--codegen_compile_debug=true;--debug_dir=<dir>;--force_schedule_result=<n>" \
AF_ENABLE_MLIR_CODEGEN=1 \
LD_LIBRARY_PATH=<见 AddCpuCase.cmake 生成的 build.make> \
  <build>/.../MlirE2eCpu_<case>_generator
grep -o 'is_reduce_mem_reuse = \(true\|false\)' <dir>/.../afir_*_RunMlirCodegen_*.mlir
```

## 验收标准

- [x] 五条钉 AllLoad plan 的 CPU E2E 全部通过（`[16,32]`、`[16,37]`、`[17,32]`、`[8,1024]`、
      `[4,32,8]`），且每条经 dump 复核钉到的 plan 带 `is_reduce_mem_reuse = true`。
- [x] 至少保留一条**不钉 plan** 的 AllLoad 语料，使全候选一起编这条路有人走。
- [x] 两条 lit 分别钉住 AllLoad 的 AR 与 RA 形态：发对应 layout、RA 不发 `ReduceInit`、
      两者都不出现 workspace 交接与跨轮累加。
- [x] 三维 ARA 注册在 `Case.cpp` / `KernelTest.cpp` 有对应建图与参考实现，`CheckGeneratedKernel`
      的 3 维分支断言恒 RA。
- [x] 适用域四条结论（2.2）与探针六条结论（2.3）写入本文档，且与代码判据逐条对应。
- [x] `is_reduce_mem_reuse` 在 MLIR 侧无读点的结论写入 `AfirBuilder.cpp` 的注释。
- [x] 三处过期结论改写完成，`docs/README.md` 的「当前设计」表新增本文一行。
- [x] `ctest` 90/90、`check-lit` 190/190（2026-10-02 实测）。
- [x] AR 既有产物不变：本方案对产品代码的改动**只有 `AfirBuilder.cpp` 的 6 行注释**，
      其余全在 `test/` 与 `docs/` 下，指令序列无从改变。逐字节 diff 的旧基线随本次 build 树
      全量重建而不存在了，所以这一条按「改动面」立论而不是按 diff——差异写在这里，不含糊过去。
- [x] 不修改产品下降代码、不修改调度器、不新增 ABI / schema 字段 / 环境变量 / 日志。
- [x] 2.4 的详查结论（`duplicate tiling_key` 非现网缺陷、根因是图级 key 的粒度不匹配、
      ATT `CheckKeyValid` 缺唯一性校验、build 树孤儿二进制的误导）写入本文，**不开缺陷单**。

### 设计文档检查结果

- [x] 跨特性交叉影响：已按 `cross_feature_check.md` 逐项检查，结论见「特性交叉影响」表。
      影响面限于 Autofuse Codegen/Backend 的测试覆盖与构建注册，不涉及 SuperKernel、
      Python 绑定与打包内容。
- [x] 编码红线：已按 `编码红线.md` 逐条检查，见「安全检查」表。重点：红线 7 下两处既有
      遗留与 ATT `CheckKeyValid` 的唯一性校验只记录不处理。
- [x] 测试设计：含 S0 探针（已完成，结论入文）、5 条 CPU E2E、2 条 lit、产物逐字节 diff。
- [x] 性能评估：不改产品代码，执行性能无影响；编译期增量为 5 条用例，分钟量级以内。
