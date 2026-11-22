# 简介

## 目的

把 A5（reg-base）调度在 buffer 分配阶段做的二次融合——`VectorFunc`——在 MLIR 这条
codegen 路上接通，使带 `VectorFunc` 节点的图不再整图回落 legacy。

本文是 2026-10-16「elementwise / broadcast 收尾设计」**功能需求 4 的实现设计**。那一节只做
评估与度量并明确「实现另立文档」，度量已于 2026-10-17 闭环，结论是：

- reg 侧 200 格 legacy 用例里 **110 格（55%）的调度产物带 `VectorFunc`**
  （`docs/development/2026-10-17-vf-node-census-reg.txt`）；
- 其中 **17 格「只差 VF」**，另外 93 格含 VF 但更早被别的缺口挡住；
- 所以 VF 是那 55% 的**必要条件**：不接它，补 op、补 dtype 一格也换不回来。

*重量（2026-10-26，见 `docs/development/2026-10-26-vf-node-census-reg.txt`）*：**110 这个数
没变，但上面第二、第三条都要订正。**

- **110 / 90 逐个相等**，所以「110 不应因本设计而变」这条回归口径成立（本期改过 12 个 bf16
  用例的输入图，所以这一遍不是照抄，是真量过）。
- **「只差 VF」17 → 8 格。** M1 / M2 落地与本期补 op / 补 dtype 一起把其余 9 格推到了别处：
  4 格直接变 `takeover`（`add_abs` / `cast_abs` / `cast_abs_float16_float` / `div_abs`），
  6 格仍 `fallback` 但首因换成了 `afir.load`、int8 / bool dtype、`afir.mul` 无 reg 形态。
- **第三条「VF 是那 55% 的必要条件」被实测否证。** 本期补 op / 补 dtype 换回来的 48 格里
  **24 格是含 VF 的**。成因是口径：普查关掉 MLIR 量「legacy 调度融不融 VF」，而 MLIR 开着时
  plan selection 可能选到另一个不含 VF 的 plan。**「legacy 侧含 VF」≠「MLIR 这条路非接 VF
  不可」**，这句话是本设计排期的原始依据，它不成立。
- 剩下那 8 格的成因只有两类：**6 格卡 VF 的 dtype 白名单**（bf16 四格、uint8 两格）、
  **2 格卡 VF 子图里的 `Scalar` 节点没建模**。

*白名单那 6 格的归因（2026-10-26 放开重量）*：白名单的注释把它们整体归给「搬运能不能换
dtype」，而那个依据量在 `6a525700` 之前。放开 bf16 / uint8 重量之后，**部分成立**：

| 用例 | 放开后 | 真实首因 |
|---|---|---|
| `abs_bf16` / `add_exp_bf16` / `truediv_bf16` / `relu_uint8` | **硬失败** | `'afir.store' op transfer source and destination must have the same element type` |
| `abs_fma_bf16` | 回落 | `Fma` 节点类型 bridge 没接（功能需求 5） |
| `abs_uint8` | 回落 | VF 边界上还有 **int16** 这一位 |

所以 6 格里**只有 4 格真是「搬运换 dtype」**，白名单替它们挡着硬失败、今天仍然必要；另外
2 格的卡点不是 dtype，拿白名单当它们的原因会把活排错层。

**探针那一侧的剖面不同，别混着读**：同样放开之后探针 reg 侧 `ok` +17、硬失败 +5，那 5 格
的诊断是 `DescriptorCountMismatch: 'afir.destination_descs' expected 2 element but found 1`，
全是 T5c（菱形两消费者）融出来的**双输出 VF**，bridge 只挂了一个 destination desc。那是
VF 这条路自己的一个缺陷，不在原 M1–M4 的分期里。

### 那 8 格的落地（2026-10-26）

一次把这 8 格打到底，结果是 **4 格 takeover / 4 格干净回落 / 0 硬失败**：

| 用例 | 结果 | 落地的那一下 / 还差什么 |
|---|---|---|
| `abs_bf16` | **takeover** | 用例样板补 `Output` 的 dtype + 白名单放开 bf16 |
| `abs_uint8` | **takeover** | 同上（uint8 + int16 两位都要） |
| `relu_uint8` | **takeover** | 同上 |
| `truediv_bf16` | **takeover** | VF 算子表加 `afir.truediv` 一行（reg 侧与 `div` 同一个调用） |
| `add_exp_bf16` | 回落 | `afir.add` 在 bf16 上没有 implementation —— Any/Shared 家族取两列交集，要先把那份 implementation 拆成 mem / reg 两份 |
| `abs_fma_bf16` | 回落 | `Fma` 这个节点类型 bridge 没接（功能需求 5 的算子补齐配方；reg-only、3 入 1 出、不要 tmp buffer） |
| `abs_brc_add` | 回落 | VF 子图里的 `Scalar`（本设计 M4） |
| `load_compare_scalar_where_store` | 回落 | 同上 |

四处改动，按「谁是首因」分开记：

1. **用例样板补 5 行 `Output` 的 dtype**（`tests/framework/share_graph/src/share_graph.cc`）。
   `6a525700` 给另外 12 个补过同一行，当时就记了「`add_exp_bf16` 一类也缺，但它们先被
   VectorFunc 那道闸拦下」——这次放开闸门之后它们果然撞上同一条
   `'afir.store' op transfer source and destination must have the same element type`。
   **这不是放宽 verifier**：那几张图真的自相矛盾（`Output` 落在缺省 f32 而 Store 是 bf16），
   账收在用例样板一侧，与 `6a525700` 同一个办法。
2. **VF 边界 dtype 白名单放开 bf16 / uint8 / int16。** 原先只收 f32 / f16。
3. **多输出 VF 的 descriptor 逐输出各一份**（`AfirBuilder.cpp`）。原先只挂第一个，双输出
   VF 撞 `DescriptorCountMismatch`——那是 verifier、硬失败。修完它变成一条诚实的能力层拒收
   （`afir.vector_func with 2 destinations has no register-form lowering yet`），探针上 5 格
   从硬失败变干净回落。**多输出 VF 的下降本身仍未实现**，那是独立一项。
4. **算子表加 `afir.truediv` 一行**，带一格 lit（`afir-vector-func-reg-base-truediv.mlir`）。

量，两台量具都跑了：

- **探针**：reg 侧 `ok` 210 → **227（+17）**、硬失败 **75 → 75 不变**，逐格迁移只有
  `unsupported → ok` 一个方向、0 格变差；mem 侧三个数一个没动（V1 不产 VF）。
- **接管表**（reg 全量 200 格，见 `2026-10-26-legacy-takeover-table-reg-vf8.txt`）：
  `takeover` 63 → **67（+4）**、`fallback` 69 → 65、**`hard-failure` 66 → 66 不变**
  ——本设计功能需求 6 那条「一格都不行」的判据过了。逐行对比，只有上表那 4 格
  `fallback → takeover`，其余 **196 格状态 / 闸门 / 首因三列逐字未变**。

**一处过程教训**：中途只补了 4 行 `Output` dtype 而漏了 `abs_uint8`，于是放开 int16 之后
它从干净回落变成**硬失败**——「放开一道闸之前，先把这道闸后面那一层的账也补齐」，漏一个
就会在硬失败桶上看见。

目标读者：Autofuse MLIR 这条路上的实现者与检视者。期望输出：一条可分期落地的实现路径、
每期的验收手段。开工前的四条待确认已于 2026-10-18 全部跑完，结论见「假设和依赖关系」。

## 范围

**包含**：

- Bridge：把调度产物里的 `VectorFunc` 节点与它挂的嵌套子图翻译成 AFIR；
- AFIR：新增一个带 region 的算子 `afir.vector_func`（这是 AFIR 的第一个 region 算子）；
- 下降：region 体内从 tile 级 `ascendc.*` 算子降到寄存器级（`AscendC::Reg`，即
  `MicroAPI`）序列；
- 能力判定：reg 形态下降不了的 op 要报**能力不足**并干净回落，不得变成硬失败；
- 量具与用例：新增多节点 reg-base E2E、lit、UT，以及 VF 普查量具的回归口径。

**不包含**：

- 调度侧一处不动。分区器（`v35/optimize/partition/vector_func_partitioner.cpp`）、
  融合门槛 `kMinVfNodesNum`、VF-capable 判定（`AscIrCodegen::IsVectorFunctionSupported`）
  都不改——本设计是「接住它产出的东西」，不是「改它产出什么」。
- legacy 的 `v35/codegen/vec_func_call/` 一处不动，它继续服务回落路径。
- VF 的以下形态**明确不在本阶段**（分期见功能需求 5）：`Arange` 单点 VF、
  `IndirectLoad` 进 VF、CV/UBFuse（`cube_type == kUBFuse`）下的 VF、VF 内的 reduce、
  brc-inline 的轴重排与 `RemovePad`。
- SuperKernel、打包交付、Python 接口：不涉及（逐项结论见「特性交叉影响」）。

涉及组件：Autofuse（`autofuse/mlir/{bridge,dialect,conversion,artifact,test}`、
vendored `autofuse/mlir/api/ascendc`）、测试框架。

# 总体概述

## 软件概述

### 项目介绍

A5 调度在 `BufQueAllocator::ProcessSingleImplGraph`
（`autofuse/optimize/buffer_allocate/buf_que_allocator.cpp`）里对**每张 impl graph 无条件**
调用 `PlatformV2::PartitionSubFunctions`，把相邻且全部 VF-capable 的节点（≥
`kMinVfNodesNum = 2`）聚成一个小子图，在父图里替换成一个 `VectorFunc` 节点，子图作为
**兄弟 AscGraph** 挂在 impl graph 上（`impl_graph_.AddSubGraph`，节点属性
`sub_graph_name` 记名字，见 `vector_func_partitioner.cpp:1582` 一带）。

legacy codegen 为这个节点发一个独立函数：

```cpp
inline __simd_vf__ void VFCall<node>(__local_mem__ float *out, __local_mem__ float *in0, ...,
                                    uint32_t output_dims_0, uint32_t out_stride_0, ...)
```

函数内按 `AscendC::GetVecLen()` 做逐元素循环，中间结果**留在向量寄存器**里
（`RegTensor` / `MaskReg`、`LoadAlign` / `StoreAlign`、`UpdateMask`），kernel 体只发一次调用。

今天 MLIR 这一侧在 Bridge 的节点类型闸门上就拒了它（`ScheduleToAfir.cpp:1087` 的
`emitCapabilityShortfall`，诊断是 `does not support node '...' of type 'VectorFunc'`），
整图干净回落 legacy。

### 产品环境介绍

| 外部件 | 关系 | 本设计的依赖面 |
|---|---|---|
| CANN Toolkit 9.2.0 的 AscendC reg 接口 | `AscendC::Reg`（`namespace MicroAPI = Reg`，见 `x86_64-linux/asc/impl/basic_api/kernel_macros.h:131`）提供 `RegTensor` / `MaskReg` / `LoadAlign` / `StoreAlign` / `UpdateMask` / `CreateMask` 与各计算 API（`asc/include/basic_api/reg_compute/`） | 只用对外头文件里声明的接口；标记为 `__simd_callee__`，只能从 `__simd_vf__` 函数或 `__VEC_SCOPE__` 块里调 |
| tikicpulib 的 `Ascend950PR_9599` CPU 模拟器 | 既有 reg-base E2E 与 UT 的验证环境 | 本设计的全部功能验收都在它上面做，**不需要上板**（依据见下方「假设和依赖关系」第 1 条的实测） |
| PyAsc（vendored 在 `autofuse/mlir/api/ascendc`，上游原件在 `/home/zhe/code/pyasc`） | 已 vendored 的 `Asc` 方言带**完整的寄存器级算子与类型**（`IR/Basic/OpReg.td`、`IR/Core/Types.td`），`EmitAsc` 方言带 `vec_scope` / `vf_for`，发射器三者都支持 | 本设计复用它们；上游 `experimental/asctile/lib/Dialect/AscVF` 的 `vf_group` 与 `LowerToReg` 是移植来源 |
| 对外接口 | 无变化 | 不新增环境变量、不改 `Codegen::Generate*` 签名、不改 run 包布局 |

## 软件功能

1. Bridge 能把 `VectorFunc` 节点及其嵌套子图翻成 AFIR，不再在节点类型闸门上拒收。
2. AFIR 能表达「一段在寄存器里逐 lane 算完的子图」，且该表达对 AFIR 的既有四个物化 pass
   （循环生成、资源物化、流水同步、缓存守卫）有明确边界。
3. 子图体复用既有的 AFIR 算子、实现选择与 dtype 供给表——**不另立一套 op 清单**。
4. 下降能产出寄存器级 AscendC 代码，且在 CPU 模拟器上数值与 legacy 等价。
5. reg 形态下降不了的 op / dtype 组合判为能力不足、干净回落，并能被量具数出来。

## 设计约束

- **分层门禁**：`dialect/AFIR/`、`bridge/`、`artifact/` 三层**不得出现 SoC 名、NpuArch
  枚举或 NpuArch 数字**（`scripts/check_platform_agnostic_layers.sh`，禁 `2201/3510/5102/9202`
  字面量）。只有 `conversion/` 与 `target/` 可以点名平台。这条直接否掉了「照抄 legacy 的
  `#if defined(__DAV_C310__) || __NPU_ARCH__ == 3510` 架构门」这个做法——见功能需求 3。
- **Schedule 一处不动**（沿用收尾设计的约束）。UB 的队列与 tmp 容量由 legacy 的
  buffer 分配与 ATT 决定，MLIR 侧只消费 tiling 里的 `q*_size`（实证：MLIR 生成的 kernel
  里 `InitBuffer` 的尺寸来自 `tiling_data.q0_size`）。**MLIR 不得凭自己的需要增加 UB 占用。**
- **拒收只能拒在 capability 层**：verifier 失败是硬失败（见 `docs` 里既有记录与
  `af-capability-probe` 的 `DRIFT` 桶恒为 0 这条不变量）。本设计新增的任何「做不到」
  都必须走 `emitCapabilityShortfall` 或 plan selection 的模板过滤。
- **编译并行度**：本机 `-j 8` 会被 OOM killer 杀掉，一律 `-j 6`。
- **两档 build 互斥**：MLIR 测试档与 legacy ST 档共用一个 `build/`，换档要删
  `build/CMakeCache.txt`。度量用的 legacy 接管表 / VF 普查在 legacy ST 档，MLIR 的
  UT/ST/lit/E2E 在 MLIR 档。
- ABI/API：无对外接口改动，故无兼容负担（详见「兼容性检查」）。

## 假设和依赖关系

**四条待确认已于 2026-10-18 全部跑完，结论如下**（每条都给了量具或 file:line，不是读码觉得对）。

1. **`__VEC_SCOPE__` 就地块在 reg-base 目标上可用 —— 成立，而且比预期强。**

   量具：`test/probe/vf_emission_form/`（两档，`run.sh` 与 `run.sh --device`）。两种形态
   算同一件事 `y = |a + b|`，256 个 f32，输出缓冲预置哨兵 `-12345.0f`。

   | 档 | 问的是 | 结果 |
   |---|---|---|
   | CPU 模拟器（`g++` + tikicpulib `Ascend950PR_9599`） | 算得对不对 | 两种形态都 `mismatches=0` |
   | 真实 NPU 编译器（`ccec --cce-aicore-arch=dav-c310`） | 编得过吗、编出来是不是向量指令 | 两种形态都编过，**指令直方图逐项相同** |

   直方图（反汇编按符号分）：两种形态都被编译器外联成一个 `.vector.thread` 函数，核心是
   `VAG_V310` 1 / `VLOOPV2_V310` 1 / `VLD` 2 / `PLT.b32` 1 / `VADD.f32` 1 / `VABS.f32` 1 /
   `VST` 1 / `SEND` 1，标量侧各发一条 `VF` 指令去启动它。**也就是说两种形态的差别只在源码层，
   不在指令层**——`__simd_vf__` 那个外联函数是我们自己手写的，而 `__VEC_SCOPE__` 让编译器
   去做同一件外联。

   于是**选 B1 定稿**，B2 不再是退路而是「同一结果的另一种写法」，没有理由多付外联 pass、
   参数表和架构门那三份代价。

   **仍然没验的**：上板实测（真实时序与功耗），以及更复杂子图（多输出、跨 dtype、谓词进
   `Select`）下两种形态是否仍等价——那要靠 M1–M4 每期的 E2E 一格一格验，不能从这一格外推。

2. **vendored `api/ascendc` 与上游 `AscVF` 的接口匹配 —— 成立，无阻塞漂移。**

   逐文件比（`diff -w --strip-trailing-cr`，216 个 `.td` / `.h` / `.cpp`）：

   - **195 个逐字相同**，21 个有差异，37 个上游没有（我们自己加的 Autofuse 算子/属性）；
   - 21 个差异里，`AscVF` 的依赖面只碰到 4 个（`Asc.h`、`Base.td`、`Ops.td`、
     `Core/Attributes.td`）**且全部是我们单向加东西**（加 include、加 def）；
   - 唯一一处双向差异是 `ConstantOpBuilder.h` 的一行建 op 写法
     （上游 `builder.create<ConstantOp>(...)` vs 我们 `ConstantOp::create(builder, ...)`）。
     **这一行不构成迁移成本**：我们自己的 MLIR 代码里还有 181 处旧式 `X.create<...>` 调用，
     在 LLVM 21.1.8 上编得过，所以旧写法仍然可用。

   符号可用性：`AscVF` 的 IR 与 `LowerToReg` / `MaterializeLoadStore` 引用的
   **85 个 `ascendc::` / `emitasc::` 符号在我们树里全部存在**，且所在的 `.td` 都在
   `Asc/IR/Ops.td` 的 include 图里（`OpReg.td` 42 行、`OpSysVar.td` 46 行、
   `Adv/Reduction.td` 21 行、`Basic/OpVecReduce.td` 57 行）。

   建 op 写法的迁移面也量了：移植集里 88 处建 op，其中 74 处走的是 `LowerToReg.cpp:100`
   那个**本地一行包装** `adaptor.create`，所以真正要改的是 1 个包装 + 14 处直接调用。

   **仍然没验的**：真正编一遍。符号在、写法兼容只说明没有已知阻塞，不等于编得过——
   那是移植第一天的事，不是开工前的闸门。

3. **VF 子图能活着穿过候选收窄 —— 成立，实测。**

   先读码：子图挂在 impl graph 的**根 ComputeGraph** 上
   （`AscGraphImpl::AddSubGraph` → `FindRootGraph` → `root_graph->AddSubGraph`，
   `graph_metadef/graph/ascendc_ir/core/ascendc_ir.cc:882`），而 `af::AscGraph` 只持有
   `std::shared_ptr<AscGraphImpl>`，拷贝是句柄拷贝——所以收窄里的
   `graphs.push_back(stage.impl_graphs[template_id])`（`PlanSelection.cpp:426`）之后，
   子图仍然在同一张根图上。

   再实测（临时打点插在 `ToTemplateInfo` 里，即**收窄之后**，用完已撤）：

   | 用例 | 结果 |
   |---|---|
   | `add_abs_test` | `found=yes nodes=8`（Data×2 + Load×2 + Add + Abs + Store + Output） |
   | `abs_uint8_test` | `found=yes nodes=6` |
   | `load_brc_test` | `found=yes nodes=10` |
   | `select_bool_test` | `found=yes nodes=12` |

   **所以功能需求 1 不需要「在收窄处显式搬子图」这个前置步骤。** 注意每格只打出一行：
   收窄之后只剩选中的那一个模板，这正是 bridge 要处理的那一个。

4. **ATT 侧已经收得下 VF —— 成立；回填那条差异是真的，但对本设计射程内的格子是惰性的。**

   顺序差异确认了，而且方向与担心的一致：

   | 路径 | 顺序 |
   |---|---|
   | legacy `Codegen::Generate` | `GenerateKernel`（`codegen.cpp:453`，VF 发射顺带回填）→ `GenerateTilingData` → `GenerateTiling`（ATT，`:456`） |
   | MLIR | `PrepareMlirCandidates`（plan selection + ATT）→ 下降 → 发射 |

   也就是说**legacy 的 ATT 看得到回填值，MLIR 的 ATT 看不到**。

   但失败方式不是报错：`RegisterVectorFuncAscirNodeParams`
   （`common/ascir_node_param/ascir_param_builder.cpp:110`）一开始就把 VF 参数注册成
   `kBuilt` + 默认值（`is_double_loop=false`、空 strides/dims），ATT 照收，走
   `vf_perf_utils.cpp:205` 的 `!is_double_loop` 那一支算代价。这与「那 17 格都过了
   `candidate-preparation`」这个实测一致。

   **射程量了**：回填只发生在双层循环那一支（`vec_func_call.cpp:600`，条件是
   `stride_depth == MAX_VF_AXIS_MERGE_SIZE - 1 && only_loop_max_depth == MAX_VF_AXIS_MERGE_SIZE - 1`），
   而那一支会在产物里留下 `strides_equal` 判断。数 legacy 已生成的产物：
   **141 个含 VF 函数的 kernel 里只有 2 个带 `strides_equal`**，而且这 2 个都是
   `indirect_load_*`（`indirect_load_store_test` / `indirect_load_embedding_test`）。

   所以：本设计射程内（M1–M4 那 17 格）**legacy 自己也把 `is_double_loop` 留在 false**，
   两侧 ATT 拿到的是同一份输入，不存在差异。这条差异只在 `IndirectLoad` 进 VF 时才变成
   真问题，而那一形态**本设计明确不在范围**（而且它还要 F-4 与 IndirectLoad 建模）。
   **留一条验收项**：真把 VF 接通之后，拿 `indirect_load_store_test` 比一次 MLIR 与 legacy
   的 tiling，确认这条差异没有悄悄扩散。

# 需求分析与设计

## 整体介绍

今天的下降流水线分四相位（`conversion/AFIRToAscendC/AFIRToAscendCPipeline.cpp` 顶部那段
注释是真源）：

```
A 准备      kernel ABI 补参数表、尺寸符号 SSA 化
B AFIR 物化  轴 -> scf.for、缓冲 -> 资源、顺序 -> 同步边、缓存守卫
C 下降      选实现（capability）、AFIR -> AscendC
D host 产物  tiling schema、入口分派
```

本设计只在两处插东西，其余相位不动：

```
Bridge        VectorFunc 节点 + 嵌套子图  ->  afir.vector_func { 子图的 AFIR 算子 }
相位 B        四个 pass 对 region 的边界（不进 region 生成 tile 循环、不为 region 内张量分配 UB）
相位 C        region 体照常选实现、照常降成 ascendc.* 的 tile 算子
相位 C.5(新)  afir.vector_func -> ascvf.vf_group，再跑移植来的 LowerToReg
              产物：emitasc.vec_scope { emitasc.vf_for { ascendc.*_reg / ascvf.load / ascvf.store } }
相位 D        不变
发射          已支持（vec_scope 打 __VEC_SCOPE__、vf_for 打 for、reg 算子打 AscendC::Reg::*）
```

**为什么 region 算子必须建在 AFIR 层而不是直接产 `ascvf.vf_group`**：相位 B 的四个 pass
都要知道「这一段是一个整体」，而它们是 AFIR 层的 pass；让 AFIR 层认识一个 AscendC 侧方言的
算子会越过分层门禁。所以 AFIR 侧有自己的 `afir.vector_func`，到相位 C 末尾才换成
`ascvf.vf_group`——换的时候两边的边界列表语义相同，是一次机械替换。

### 路线选择

收尾设计里给的是两条路线（A 展平 / B 建模）并「初步倾向 B，但度量没出来前不定」。
度量出来了，连同本轮读码与实测，三条候选的结论如下。

| 路线 | 做法 | 结论 | 依据 |
|---|---|---|---|
| **A 展平** | Bridge 把 VF 子图展平成普通 tile 级 afir 节点，逐节点发 reg API | **否决，不是性能问题而是功能问题** | VF 的中间结果在寄存器里，**没有任何 UB 分配**（子图内部张量在 legacy 里是 `MicroApiTensor` 即 vreg，`vec_func_call.cpp:ParseSubGraph` 明说子图 `Data` 的输出不进 tensor manager）。展平后每个中间结果都要一块 UB，而 UB 的队列容量由 legacy 的 buffer 分配 + ATT 定、MLIR 只读 `tiling.q*_size`。于是展平要么溢出 UB，要么得重做分配与 tiling——后者越过「Schedule 一处不动」这条约束 |
| **C 文本复用** | MLIR 发调用，函数体文本交给 legacy 的 `VfCall` 生成后拼进产物 | 否决 | `VfCall` 的输入是 legacy 的 `TPipe`（`tpipe.tiler`、`GetTensor(tensor_id)`），要在 MLIR 侧重建一个 legacy TPipe 等于把 legacy codegen 整个拉回来；而且产物里会有一段不经 verifier、不经 lit 的文本，与 artifact 层「只装配更下游已经做完的决定」的定位冲突 |
| **B 建模** | 新建带 region 的算子，下降成寄存器级序列 | **采纳** | 代价比收尾设计估的低一个数量级，因为**寄存器级那一层已经存在**（下表） |

B 的代价之所以掉下来，是四件已经存在的东西：

| 已有 | 位置 | 省掉的工作 |
|---|---|---|
| 寄存器级算子与类型 | vendored `api/ascendc` 的 `IR/Basic/OpReg.td`（`add/sub/mul/div/max/min/and/or/xor/abs/exp/ln/log/neg/relu/sqrt/not`、`adds/muls/maxs/mins/leaky_relu/shift_*`、`select_reg`、`duplicate_reg`、`reduce_{sum,max,min}_reg`、`data_copy_vld_reg` / `data_copy_vst_reg`、`update_mask`、`create_mask`、`reg_tensor`、`mask_reg`、`local_mem_bar`），类型在 `IR/Core/Types.td`，且 `OpReg.td` 已在 `Ops.td` 的 include 图里 | 一整套 reg 级 ODS |
| 寄存器级发射 | `lib/Target/AscendC/Basic/Reg.cpp`，打的是 `AscendC::Reg::*`（与 legacy 的 `MicroAPI::*` 同一命名空间，`MicroAPI = Reg`） | 一个发射器 |
| VF 的循环与作用域 | `EmitAsc` 方言的 `vf_for` / `vec_scope`，发射见 `lib/Target/AscendC/EmitAsc.cpp:248`（`for (...)`）与 `:274`（`__VEC_SCOPE__ { ... }`） | lane 循环的建模与发射 |
| tile 级 -> reg 级的下降 | 上游 `experimental/asctile/lib/Dialect/AscVF/Transforms/LowerToReg.cpp`（659 行）与同目录的 `MaterializeLoadStore` / `FuseVFFor` / `ReorderOpsInVecScope` / `EliminateCommonMask` / `EraseBarrier`；**它吃的正是 tile 级 `ascendc.*` 算子**（`ConvertOp<ascendc::...>` 一族），也就是我们相位 C 已经在产的东西 | 下降的主体算法 |

于是 B 的新增只有：一个 AFIR region 算子、相位 B 四个 pass 的边界、一次机械替换、
一套移植 + 能力判定。**这是「接线」而不是「从零建一层」**，与收尾设计里「规模与向量化轴
分段统一同级或更大」的估计相比要小。

B 还有两种发射形态，差别只在最后一步：

- **B1 就地 `__VEC_SCOPE__`**（上游形态）：VF 块直接长在 kernel 体里，**不需要外联函数、
  不需要参数表**——尺寸与 stride 直接用 kernel 自己的 SSA 值，也**不需要架构 `#if`**
  （目标能力由 target profile 决定，而不是预处理器；这同时避开了分层门禁）。
- **B2 外联 `inline __simd_vf__`**（legacy 形态）：与 legacy 产物形状一致，但要多一个外联
  pass、一条参数表（dims/strides）、以及一个架构门的发射点（而架构门的字面量不能写在
  artifact 层，得由 conversion/target 层给）。

**本设计选 B1，而且 B2 不是退路而是同一结果的另一种写法**。依据是假设 1 的两档实测：
CPU 模拟器上两种形态数值都对，ccec（dav-c310）上两种形态都编得过**且指令直方图逐项相同**
——`__VEC_SCOPE__` 让编译器做了我们在 B2 里手写的那次外联。于是多付外联 pass、参数表与
架构门这三份代价换不回任何东西。
（CANN 自己的实现里两种形态都在用：`impl/basic_api/dav_m300/...` 是 `__VEC_SCOPE__`，
`impl/adv_api/detail/*/*_3510_impl.h` 是 `__simd_vf__`——这也解释了为什么两种都收。）

## 功能需求

### 功能需求 1：Bridge——`VectorFunc` 节点与嵌套子图翻成 AFIR

1. 介绍

   `ScheduleToAfir` 见到类型为 `VectorFunc` 的节点时，不再报能力不足，而是建一个
   `afir.vector_func`：操作数是它在父图上的输入/输出边界张量（tile 级
   `!ascendc.local_tensor`），region 里是子图各节点按拓扑序翻出来的 AFIR 算子。

2. 输入

   - 父图上的 `VectorFunc` 节点：`inputs`（动态输入 `x[]`）、`outputs`（`y[]`）、
     `attr.sched`（轴与 loop_axis，分区器从首个被融节点继承而来，
     `vector_func_partitioner.cpp:1395` 一带）、字符串属性 `sub_graph_name`；
   - 由 `sub_graph_name` 在 impl graph 上 `FindSubGraph` 找到的嵌套 `AscGraph`：内部含
     `Data` + `Load`（边界入）、计算节点、`Store` + `Output`（边界出），以及常量/运行期
     标量的 `Scalar` / `ScalarData` 副本；
   - 边界序号：子图 `Data` 节点的 ir 属性 `index`（子图命名空间）与 `vf_root_input_index`
     （父图命名空间）。后者由分区器建边后写入，**缺失是合法输入**（手工构造的 fixture），
     legacy 的做法是回退用 `index` 并打 warning。

   **非法输入与边界条件**（每一条都要有判据，不得当成「不会发生」）：

   | 输入 | 判据 | 处置 |
   |---|---|---|
   | `sub_graph_name` 缺失或找不到对应子图 | 空指针 / `FindSubGraph` 失败 | **硬失败**（图自称有 VF 却没有体，是产生方的 bug，不是能力不足） |
   | `vf_root_input_index` 越界（`< 0` 或 `>= 节点输入数`） | 与 legacy 同一条检查 | 硬失败（红线 2：外部数据作索引前必须校验） |
   | 边界序号重复 | 映射插入失败 | 硬失败 |
   | 子图内出现本阶段不支持的节点类型 / dtype | 既有 `isAFIRBufferFormOperationSupported` 与 dtype 供给表 | **能力不足**，整图回落（与今天同一条口径） |
   | 子图嵌套子图（VF 里还有 VF） | 子图内再出现 `VectorFunc` | 能力不足（分区器今天不产，但判据要在，不能靠「不会发生」） |

3. 处理

   - 读子图，按**拓扑序**遍历（`ModifySubgraphAttrs` 已经为子图做过
     `TopologicalSortingForVfGraph`，bridge 不得依赖容器顺序——红线 5：图改写结果必须确定）；
   - 边界入：子图的 `Data` + `Load` 对不翻成 AFIR 的 `load`（那是 GM→UB 的搬运），而是**映射
     成 region 的 block argument**，对应父图上的那条输入边。这是本需求最核心的一处语义
     转换：子图里的 `Load` 表达的是「从 UB 取到寄存器」，与父图的 `afir.load` 不是一件事；
   - 边界出：`Store` + `Output` 对映射成 region 的终结算子 `afir.vector_func_yield` 的操作数，
     对应父图上的输出边界张量；
   - 子图内的 `Scalar` / `ScalarData` 副本：本阶段只收常量 `Scalar`（物化成 AFIR 的标量
     常量），运行期 `ScalarData` 的值参数链归 M4（功能需求 5）；
   - 父图侧：`afir.vector_func` 占据原 `VectorFunc` 节点的位置，参与父图的 descriptor /
     队列资源推导，与一个普通多入多出算子**同形**——这是让相位 B 的资源与同步逻辑不用特判
     的前提。

4. 输出

   - AFIR：一个 `afir.vector_func`，region 内是 tile 级 AFIR 算子（**不是 lane 级**——
     lane 化发生在相位 C.5）；
   - 失败时：按上表分两种，能力不足走 `emitCapabilityShortfall`（整图干净回落），
     硬失败走 `emitError`；
   - 日志：不新增默认开启的逐节点日志（红线 6）。子图规模与边界数走既有的 `GELOGI` 等级。

### 功能需求 2：AFIR 相位 B 对 region 的边界

1. 介绍

   相位 B 的四个 pass 都是按「平坦的算子序列」写的。region 进来之后，每一个都要有明确
   的、写在代码注释里的边界，否则表现是**静默生成错的东西**而不是报错。

2. 输入

   带 `afir.vector_func` 的 AFIR 模块（相位 A 之后）。

3. 处理

   | pass | 边界 | 不处理会怎样 |
   |---|---|---|
   | `AscendCLoopGeneratePass`（轴 → `scf.for`） | **不进 region**：region 外照常生成 tile 循环；region 内的 lane 循环由相位 C.5 生成（`emitasc.vf_for`，上界是 `loop_times`） | 会在 region 里生成一层 tile 级 `scf.for`，与 C.5 的 lane 循环叠成两层，产物逐元素算两遍 |
   | `AFIRMemoryAllocatePass`（缓冲 → 资源） | **只物化边界张量**；region 内的中间结果不分配 UB（它们将成为 vreg） | 增加 UB 占用，而 UB 预算由 legacy 定（设计约束第 2 条），溢出且不报错 |
   | `AFIRPipelineSyncPass`（流水同步） | region 当作**一个** V→V 的整体插同步，不在 region 内部插 | region 内部插同步 = 在 `__VEC_SCOPE__` 里发 tile 级同步算子，发射不出来或语义错 |
   | `AFIRExecConditionPass`（缓存守卫） | 守卫包在 region 外 | 同上 |

   **M1 实测（2026-10-19）：要加边界的只有三处，而且不是上表猜的那三处。**
   `IsolatedFromAbove` 不挡遍历（MLIR 的 `walk` 照样下去），所以边界只能显式判，判据集中
   在 `afir::isInsideVectorFunc`（`dialect/AFIR/IR/AFIR.h`）一处，各 pass 调它：

   | 位置 | 要不要边界 | 实测原因 |
   |---|---|---|
   | `AscendCLoopGeneratePass` | **不要** | 它只遍历函数**顶层块**（`LoopGenerate.cpp` 的 `buildLoopTree(Block &)`），进不去 region。VF 算子自己按它继承来的 `afir.sched` 落进 tile 循环，与 legacy 的调用点同位置 |
   | `AFIRMemoryAllocate` 的资源绑定（`Analysis/CanonicalBindings.cpp`） | **要** | region 内的算子也挂 descriptor，不挡就会被要求「定义点要有 buffer_desc」——而那些值是寄存器，没有存储 |
   | `AFIRMemoryAllocate` 的资源改写（`MemoryAllocate/CanonicalRewrite.cpp`） | **要**，另加一条 | 同上；**另外**它按操作数/结果类型重建算子，而 `OperationState` 不带 region——不搬 region 的话 VF 被重建成空壳 |
   | `AFIRPipelineSyncPass` | **不要** | 它只收 destination-style 且带 `destination_descs` 的算子。region 内是 tensor 形态，天然不在其中；VF 算子自己**要**在其中，所以它必须声明 `AFIRDestinationStyleInterface` 并挂 descriptor（否则这一段不插 enque/deque，产物在 CPU 模拟器上照样算对，到真机才是偶发错数据） |
   | `AFIRExecConditionPass` | **要** | 它按 `isAfirOperation` 遍历，会给 region 内的算子插缓存守卫 |

   **还有一处不在相位 B**：相位 C 的实现选择（`SelectImplementation`）会走进 region，对
   tensor 形态的算子报「no supported Ascend implementation」。那条诊断看起来像「这个 op
   没接」，而事实是「它不该被问」。解法不是再加一道边界，而是**把相位 C.5 排在选型之前**
   （见功能需求 3）——region 本来就不走选型。

4. 输出

   AFIR：region 外资源与同步齐备，region 内仍是纯计算序列、无 tile 循环、无 UB 资源。

### 功能需求 3：相位 C.5——下降到寄存器级

1. 介绍

   **M1 实测后这一节改了做法（2026-10-19）。** 原计划是「换成 `ascvf.vf_group`，再跑移植来的
   `LowerToReg`」。读码核对发现对不上，两条都是硬的：

   - 上游 `LowerToReg` 的 pattern 匹配 `ascendc::*L2Op` 那一族 **stock** 算子，而我们这条路
     下降出来的是 Autofuse 的包装算子（`ascendc.sub_extend_reg_base` 之类）——**一条都匹配
     不到**；
   - 它发的 `Reg::DataCopy` 在这一档 CANN 的 reg 搬运头里**不存在**
     （`asc/include/basic_api/reg_compute/kernel_reg_compute_datacopy_intf.h` 给的是
     `Load` / `LoadAlign` / `Store` / `StoreAlign` 一族）。照它发出来的代码编不过，而且要到
     设备编译器上才报。

   所以 M1 改成**在我们自己的算子集上直接下降**：`afir.vector_func` 的 region（tensor 形态的
   AFIR 算子）一步换成 `emitasc.vec_scope { emitasc.vf_for { 寄存器算子 } }`。上游那份留作
   结构参照（lane 循环、mask 的用法、搬运的物化次序都抄它）。

   **排在实现选择之前**，不是之后：region 内的算子不走选型（见功能需求 2 末尾）。

2. 输入

   - `afir.vector_func`：边界 dst / src 列表 + region 体（tile 级 `ascendc.*`）；
   - 上游 `ascvf.vf_group` 的入参形状：`dst(...)` / `src(...)` 两个
     `Variadic<AscendC_LocalTensor>` 加一个 `groupType`（元素类型由它定，`LowerToReg` 用它
     算 `ELEMENT_PER_VECTOR_LENGTH`）。

3. 处理

   - **第一步（已完成，2026-10-18）**：假设 1、2 已验掉——形态定 B1（两档实测见假设 1），
     vendored 漂移清单已出（195/216 逐字相同，`AscVF` 依赖面全是单向增量，85 个引用符号
     全在，见假设 2）。移植可以直接开工；第一天的事是**真编一遍**，不是再查接口。
   - **M1 没有移植 `AscVF`**（理由见上）。实际用到的现成件是：`EmitAsc` 的 `vec_scope` /
     `vf_for`、`Asc` 的寄存器算子与类型（`reg_tensor` / `mask_reg` / `create_mask` /
     `update_mask` / `add_reg` / `abs_reg` / `div_reg` …）、`local_tensor.get_phy_addr_v2`
     与 `emitasc.ptr_offset`。**新增的只有两个算子**：`ascendc.load_align_reg` /
     `ascendc.store_align_reg`（`Autofuse/OpRegDataCopy.td` + 一个发射器），因为上游那两个
     `data_copy_*_reg` 打的函数这一档 CANN 没有。
   - **寄存器算子要声明副作用**。它们的目的地是操作数、没有结果，不声明就是死代码：
     整段 `__VEC_SCOPE__` 会被回收，而产物仍然编得过——两次 Load、一次 Store、中间什么都
     没有。同理 `afir.vector_func` **不能用 `RecursiveMemoryEffects`**（region 内是纯算子，
     递归推出来是「无副作用」），要显式声明 `srcs` 读、`dsts` 写。两条都是实测踩出来的。
   - 多出来的那一位：上游的 `LowerToReg` 按 `vf_group` 的 `groupType` 取单一元素类型，
     而我们的子图可以**中途变 dtype**（`Cast`，17 格里有 7 格是这种）。这一位归 M2，
     M1 的 capability 判据里显式排除它（见功能需求 4）。
   - 架构门：B1 下**不发** `#if`。这不是省事：artifact / bridge 层不许出现 NpuArch 字面量
     （设计约束第 1 条），而目标能力本来就由 target profile 在相位 C 决定——会不会走到这一步
     本身就是平台判定的结果。

4. 输出

   - IR：`emitasc.vec_scope { emitasc.vf_for { ascvf.load / ascendc.*_reg / ascvf.store } }`；
   - C++：`__VEC_SCOPE__ { for (uint16_t i = 0; i < n; i += 1) { AscendC::Reg::...(...); } }`；
   - 与 legacy 产物**不逐字一致**（legacy 是外联函数）。等价性由数值验收，不由文本 diff
     ——这一条要写进验收标准，避免下一轮有人拿 `diff` 当判据。

### 功能需求 4：能力不足的表达

1. 介绍

   B 路线引入了**第二张供给表**：一个 tile 级 `ascendc.*` 算子有没有 reg 形态的下降
   （`LowerToReg` 里有没有对应的 `ConvertOp`）。这张表与既有的「AFIR 能不能表达」、
   「平台供不供得起这个 API」是三件事，必须能分开回答。

2. 输入

   region 体内的 tile 级算子清单 + 目标 profile。

3. 处理

   - 判定点放在**能力层**，不放在 `LowerToReg` 里：`LowerToReg` 失败就是 pattern 失败，
     表现为硬失败（而且是报在 root op 上，查不出是哪一条），这正是既有不变量禁止的
     （`af-capability-probe` 的 `DRIFT: capability said yes, pattern said no` 桶恒为 0）。
   - 具体：在相位 C 的实现选择里给 region 体内的算子加一问——「这个 op / dtype 组合有
     reg 形态吗」。没有则 `emitCapabilityShortfall`，整图干净回落。
   - 这张表的真源只能有一份，且必须能机械核对（与功能需求 3「dtype 事实只写一份」同一条
     规矩）：`LowerToReg` 支持哪些 `ascendc` 算子是**代码事实**，所以表要么由它生成，要么
     有门禁比对。**写第三遍手写清单是本设计明令禁止的做法。**

4. 输出

   - 能力不足：`UnsupportedOperation: ... has no register-form lowering`（措辞按既有诊断
     风格定稿），`shortfall=yes`，接管表里落在 `fallback` 桶；
   - 探针可见：`af-capability-probe` 的桶表里新增一桶，且 `DRIFT` 桶仍恒为 0。

### 功能需求 5：分期范围

1. 介绍

   按那 17 格「只差 VF」的用例**实际用到的 micro API** 切期（清单是 2026-10-18 从
   legacy 生成产物里数出来的，不是按名字猜的）。

2. 输入

   17 格的 API 清单（`vf_node_census` 那一轮的副产物）。

3. 处理

   | 期 | 新增能力 | 覆盖的格 | 需要的 reg API |
   |---|---|---|---|
   | **M1** | 单输出、全程同 dtype、无标量、单一合并轴（`strides_equal` 那一支不涉及） | `add_abs_test`、`div_abs_test` | `Add` `Div` `Abs` + `RegTensor` `MaskReg` `CreateMask` `UpdateMask` `LoadAlign` `StoreAlign` |
   | **M2** | 图中途变 dtype（`Cast`，含 bf16 / uint8 两侧） | `abs_bf16` `abs_uint8` `abs_fma_bf16` `add_exp_bf16` `cast_abs` `cast_abs_float16_float` `relu_uint8` `truediv_bf16` | `Cast`（含 `CastTrait` / `SatMode` 模板位）`Exp` `Relu` |
   | **M3** | 谓词寄存器作为值（比较结果进 `Select`） | ~~`bool_mask_flow` `select_bool`~~ -> **`load_compare_where_store`**（2026-10-27 重筛，见下） | `Compare` `Select` |
   | **M4** | ~~运行期标量值参数链、brc-inline（`Duplicate`）、多模板 / 双层循环~~ -> **`afir.mul` 一行** | ~~`add_abs_{float,half,int8}_scalar` `abs_brc_add` `load_brc`~~ -> **`load_brc`** | `Mul` |
   | M5（新）| VF 子图内的**常量 `Scalar`**：AFIR tensor 形态的标量操作数 + 标量源广播 | `load_compare_scalar_where_store` `abs_brc_add` | `Compares` `Duplicate`（标量形态）|
   | M5（备选） | 上游的四个优化 pass（公共 mask 消除、搬运消除、重排、去 barrier） | —— | —— |

   **M1 已落地（2026-10-19）。** 实测补两条分期判据，都是写的时候才看见的：

   - **缓存守卫进不了 region**：子图内的算子可能带非 `kNoCache` 的 `exec_condition`，而它们
     在子图里的 `loop_axis` 是 -1；AFIR 的 `#afir.sched` 在属性构造时就查「有缓存周期必须有
     loop_axis」，那是**硬失败**。所以这一条在 bridge 的能力层拒掉（探针上是 T5 的 10 格）。
   - **M1 的算子表只有三行**（`add` / `div` / `abs`），按「加一行要带一格用例」定的。探针上
     `afir.neg` 那 10 格（T5c）因此报能力不足——那是如实的现状，不是漏。

   **M2 已落地（2026-10-19）。** 实测要改这一期的三处说法，都是写的时候才看见的：

   - **M2 不是「算子表里加一行 `Cast`」。** 跨 dtype 的 VF 还要两件事，两条都是硬的，因为
     **向量寄存器是定宽的**：`RegTensor<float>` 一段装 `VL/4` 个元素、`RegTensor<half>` 装
     `VL/2` 个，要把 half 转成 float，那些 half 必须先落在 **4 字节间隔的 lane 槽位**里。
     于是：
     1. **lane 的宽度由 VF 里最宽的元素类型定**（`UpdateMask<max>`、`VL/sizeof(max)`、
        循环次数与 UB 下标都跟着它），与 legacy 的 `max_dtype_size_` 同一个取法
        （`vec_func_call.cpp:342`、`vf_loop.cpp:397`）。
     2. **窄边界的搬运要带 dist**：取进来 `LoadDist::DIST_UNPACK_*`、存回去
        `StoreDist::DIST_PACK_*`（判据抄 legacy 的 `micro_load_api_call.cpp:29` /
        `micro_store_api_call.cpp:28`，**只收一步到位的那几种**；legacy 在一条 dist 不够时
        会再补一串 `Reg::UnPack` / `Reg::Pack` 加临时寄存器与 `MaskPack`，那归后续）。
        不带就是 `DIST_NORM`——**编得过、跑得完、静默算错**。
     这两条各自做过变异验证，而且**都是数值验收，不只是文本**：短路第 2 条（两条 dist 都
     不打）后 CPU 模拟器上 `element 1` 就错；短路第 1 条（lane 宽度按 destination 取）后
     **第一趟 lane 的 64 个元素仍然全对**，从 `element 64` 起才错——所以那一格的形状必须够
     跑多趟，只有一趟的形状会假绿。
     M2 的实际改动面因此是：算子表 3 行、一个新 op（`ascendc.cast_reg` + 发射器）、
     `load_align_reg` / `store_align_reg` 各加一个可选 `dist` 属性、能力层 3 条判据。

   - **bf16 那 4 格 M2 到不了，这一期的格子清单从 8 格修正为 4 格。**
     `abs_bf16` / `add_exp_bf16` / `truediv_bf16` / `abs_fma_bf16` 卡的**不是 VF**，是
     **AFIR 的 dtype 供给表**：`afir.vector_func` 的边界元素类型域是
     `[UI8, UI16, I16, I32, F16, F32]`（`dialect/AFIR/IR/Ops/VectorFunc.td:50`），
     `afir.abs` / `afir.add` / `afir.exp` 的域里也都没有 BF16
     （`Ops/Elementwise.td:321`、`:628`、`:343`）。补 BF16 是 2026-10-16 那份设计的
     **功能需求 3**，本设计「不包含」一节明言不在射程内。所以 M2 真正能拿到的是
     `cast_abs`、`cast_abs_float16_float`、`abs_uint8`、`relu_uint8` 这 4 格；
     bf16 那 4 格要等 dtype 供给表那一条，**不是 M2 漏了**。

   - **多一条分期判据：算的那一步只在最宽的 dtype 上做。** 窄 dtype 的算子不收——它的
     寄存器里元素是按宽槽位排的（窄边界被 `DIST_UNPACK_*` 撑开过），而
     `Reg::Abs(RegTensor<half>&, mask)` 读的是 `VL/2` 个 half，与 `UpdateMask<float>` 给的
     那张 mask 对不上几个 lane 活着。这件事只能在设备上量，量不出来就不该在能力层答
     「支持」。17 格里跨 dtype 的那几格都是「先 cast 到宽的、算完再 cast 回去」，所以这一条
     不挡它们。

   **`Reg::Cast` 的 `trait` 这张表**（(输入, 输出) -> `api_regbase/utils.h` 里那个具名
   `constexpr CastTrait`）逐行抄自 legacy 的 `micro_cast_api_call.cpp:18`，常量的**定义**
   仍然只有一份（`utils_reg_base.h` 被 `ascendc.cast_reg` 声明为扩展头内联进 kernel），
   复制的只是「挑哪一个」。**表外的 dtype 对报能力不足，不回落到 `cast_trait_none`**：
   legacy 查不到时用缺省 trait，而那条路上「没列过」与「列过且是 none」分不开。
   顺带一条读表要注意的：legacy 的 **VF-capable** 表
   （`CastAscIrCodegenImplV2::IsVectorFunctionSupported`，16 对）比 trait 表（18 对）**更窄**，
   所以「表外」那一条今天从调度侧到不了，只由手写 lit 钉住——它是判据，不是已命中的桶。

**`afir.neg` 那一行补上了（2026-10-19，M2 之后）**：带一格 E2E
`load_abs_neg_reg_base_store_256x128`（`load → abs → neg → store`，即 `-|x|`）。

图取 `abs` 在前、`neg` 在后不是随手排的：那样**两个算子与它们的顺序都可观测**——漏发 `neg`
得到 `|x|`、漏发 `abs` 得到 `-x`、顺序写反得到 `|x|`，三种都在 x≠0 上全错。反过来
（`neg → abs`）是 `|x|`，`neg` 被吃掉，漏发它一个元素都不会错，那一格就验不住这一行。
**这一行的判据只能是数值**：表里填错一个 API（比如填 `Reg::Not`）在 IR 与 C++ 文本上都看不
出不对。已有的 `load_neg_reg_base_store` 不能代替它——那一格只有一个计算节点，够不上
`kMinVfNodesNum = 2`，走的是 tile 级的 `NegExtend`。

实测：
- 探针 `--reg-base --topo`：**T5c 那 10 格从 `vector func: no register-form lowering` 转
  `ok`**（T5c 的图是 `load → abs(分叉) → {abs, neg} → add → store`），菱形那 40 格现在
  30 格 ok，只剩 T5 的 10 格卡在缓存 `exec_condition` 那一条。`DRIFT` 与 `FAILED` 仍为 0。
- 变异验证：删掉这一行后**只有这一格 E2E 变红**（诊断正是 `'afir.neg' has no register-form
  lowering`），lit 仍 210/210。lit 对这一行是盲区，而那是有意的：规矩要的是「一格用例」，
  而这一行的判据在数值上，不在文本上。
- reg 侧接管表**不变**（推导，未重测）：那 6 格 VF 层的首因分别是 cast ×2、`afir.mul`、
  `afir.eq`、白名单 ×2，没有一格报 neg；而这一行只影响已经进到这一层的图。

`afir-vector-func-unsupported.mlir` 第一格因此换成 `afir.mul`——它归 M4（brc-inline 那一支），
而且是**实测命中的**（接管表上 `load_brc_test` 报的就是它）。**M4 之后它又换了一次**，
见下面 M4 那一节。

### M3 / M4 开工前的重筛（2026-10-27）：旧格子清单与「17 格」都已失效

上面那张分期表的格子清单是 2026-10-18 按「只差 VF」那 17 格排的。2026-10-27 在 HEAD 上重量
reg 全量接管表（200 格，`docs/development/2026-10-27-c1-vf-hard-failure-rescreen.txt`）之后，
**那张清单一格都不能照用**。按干净表，今天首因真落在 VF 那一层的只有 7 格：

| 格 | 首因诊断 | 归属 |
|---|---|---|
| `load_compare_where_store_test` | `'afir.eq' has no register-form lowering` | **M3** |
| `load_brc_test` | `'afir.mul' has no register-form lowering` | **M4** |
| `abs_brc_add_test` | `VectorFunc subgraph node 'Scalar_scalar' of type 'Scalar' is not modeled yet` | M5（新） |
| `load_compare_scalar_where_store_test` | 同上（`'Scalar_scalar0'`） | M5（新） |
| `indirect_load_complex_input_broadcast_simd_test` | 同上（`'Scalar_input_scalar0'`） | 出范围 |
| `indirect_load_mixed_rank4_axis2_simd` | `VectorFunc node ... reads 'int32'` | 出范围 |
| `indirect_load_rank4_axis1_full_prefix_bessel_k0_sum_simd` | 同上 | 出范围 |

三格 `indirect_load_*` 出 elementwise / broadcast 范围（算子补齐的范围由 2026-10-16 那份设计
定），不做。

**旧清单那六格今天卡在更早的闸门上，M3 / M4 做完也不会动**，逐格复核过：

| 旧清单里的格 | 原定期 | 今天的首因 |
|---|---|---|
| `bool_mask_flow` / `select_bool` | M3 | scheduler dtype value 12（bool）未建模——到不了 VF 这一层 |
| `add_abs_float_scalar` / `add_abs_half_scalar` | M4 | `'afir.load' op has no supported Ascend implementation` |
| `add_abs_int8_scalar` | M4 | `element type 'int8' is not in the AFIR supply table` |
| `abs_fma_bf16` | （M2 清单）| `Fma` 节点类型 bridge 没接（功能需求 5 的算子补齐配方）|

**复核的口径**：那张表量在 `0413d574` 的内容上，而本期开工点 `f5ed44ca` 与它的差只有
`967a4027`（纯文档）与 `f5ed44ca`（143 个文件，全在 `autofuse/mlir/test/e2e/cpu/cases/` 与
一份文档里）。接管表量的是 `autofuse/tests/v35/st/backend_e2e_v2` 那 200 个生成器目标，
与这两笔一个文件都不重叠，所以那张表在 `f5ed44ca` 上按构造成立，不重量。

### M3 已落地（2026-10-27）

**这一期的那一件事：谓词寄存器作为值。** `afir.eq` 在寄存器形态下产出的不是一段 `ui8` 的
值，而是一条 `MaskReg`——一位一个 lane、没有元素类型；`afir.where` 直接拿它当
`Reg::Select` 的 mask。中间那条谓词一次都不落 UB，也没有被物化成 0/1 的向量。

实测要补三处说法，都是写的时候才看见的：

- **`ui8` 这个「元素类型」要整条排除在 dtype 判据之外。** AFIR 没有谓词类型，比较的结果写
  成 `tensor<?xui8>`。拿它去跟 lane 宽度比，M2 那条「算的那一步只在最宽的 dtype 上做」会把
  **每一格比较**都判成「算在比 lane 窄的 dtype 上」。所以多了一个
  `computeElementType()`：产值的看结果，产谓词的看它第一个值操作数。
  `widestElementType()` 一个字没动——legacy 的 `max_dtype_size` 也把比较输出一起数了
  （`vec_func_call.cpp:342`），而 1 字节永远不会是最宽的那一个，两边数的是同一个答案。

- **谓词的来路与去处各要一条能力判据**，判据抄 legacy 的 `ShouldInitAsMaskReg`
  （`vec_func_call.cpp:264`）：**所有**消费者都得是 Select 的 mask 那一位（有一个不是就不能
  当谓词用），而 Select 的条件必须来自 region 内产谓词的那一行。两条对应 legacy 的两条补丁
  指令——消费者不是 Select 时补 `Reg::Duplicate(dst, 1, mask)` 把谓词物化回值
  （`micro_compare_api_call.cpp:48`）、条件来自 UB 时补
  `Reg::CompareScalar<..., NE>(tmp, reg, 0, pReg)`（`micro_where_api_call.cpp:30`）——
  两条都不建模，所以都报能力不足、干净回落。

- **两行一起加，单独加任一行都没有用。** 比较产出的谓词今天只有 `where` 消费得了，而
  `where` 的条件今天只能由比较产出。所以「加一行要带一格用例」在这一期是「加两行带一格」。

改动面：一个新 op（`ascendc.compare_reg` + 发射器，`Reg::Compare<T, CMPMODE::EQ>`）、
`ascendc.mask_reg` 的发射器（上游 Basic 的 op，上游 printer 表里没有它；按既有的
「Autofuse 子树接缝」放在 `Target/Asc/Autofuse/VecCmpsel.h`，不动 vendored 的那两个文件）、
算子表 2 行、能力层 2 条判据、`select_reg` 直接用现成的。

### M4 已落地（2026-10-27）：这一期只剩 `afir.mul` 一行

重筛之后 M4 的射程只有 `load_brc_test` 一格，它报的就是
`'afir.mul' has no register-form lowering`。所以这一期是**算子表加一行**，带一格 E2E
（`load_mul_abs_reg_base_store`，`load → mul → abs → store` = `|x*y|`）。

图取 `mul` 在前、`abs` 在后与 `load_abs_neg_reg_base_store` 同一个考虑：漏发 `mul` 得
`|x|`、漏发 `abs` 得 `x*y`、顺序写反得 `|x|*y`，三种都在乘积为负的元素上错。**这一行的判据
只能是数值**——表里填错一个 API（比如填 `Reg::Add`）在 IR 与 C++ 文本上都看不出不对，所以
不给它配 lit（与 `afir.neg` 那一行同一条规矩，lit 对它是有意的盲区）。

`afir-vector-func-unsupported.mlir` 第一格因此**再换一次**，换成 `afir.sub`。与前两次不同：
`afir.sub` **不是实测命中的**。HEAD 的 200 格里首因落在这一层的只有 `afir.eq`（M3 收了）
与 `afir.mul`（M4 收了），没有第三个。所以那一格今天是**手写判据**，与第三格（cast trait
表外）同一性质。

**原定归 M4 的三件事（运行期标量值参数链、brc-inline 的 `Duplicate`、多模板 / 双层循环）
一件都不在这一期**，理由分三条：

1. **「运行期标量值参数链」这个前提本身不成立。** 两格真正需要的是**编译期常量** `Scalar`
   （`abs_brc_add` 的 `SetValue("1.0")`、`load_compare_scalar_where_store` 的
   `SetValue("0.5")`），不是 `ScalarData`。而 B1 发射形态是内联的 `__VEC_SCOPE__`，没有外联
   函数，所以也**不需要**legacy 那条值参数链——常量直接在作用域里物化就行。
2. **真正的卡点在 AFIR，不在 VF 的算子表。** AFIR 的 **tensor 形态今天完全不建模标量操作
   数**：`verifyLogicalForm` / `verifyElementwiseOp` 第一句就是
   「all operands are TensorType / ShapedType」，否则报 `expected tensor operands and result`
   （`Ops/Elementwise.cpp:109` / `:337`）；bridge 那一侧也因此在
   `handleMarkerNode` 里把 tensor 形态的 `Scalar` 直接拒掉
   （`AfirBuilder.cpp:1054`，注释写明了理由就是上面那条 verifier）。
   ODS 这一层其实已经放开了（`scalarOperand != "none"` 的 op 的操作数类型是
   `AnyTypeOf<[shaped] # elementTypes>`），所以 `afir.ge %tile, %f16` 建得出来——**只差那条
   共享 verifier**。
3. **`abs_brc_add` 还多一道**：它要的是**标量源的广播**，而 `afir.broadcast` 的 `$input`
   是 `AFIR_BufferizableShapedType`（`Ops/Broadcast.td:56`），标量源连表达都表达不出来。
   那是改一个 op 的操作数契约，不是加一行表。

两件都是 AFIR 方言层的改动（一条共享 verifier、一个 op 的操作数契约），各自带自己的门禁，
不是「一期一条判据」的算子表增量。把它们塞进 M4 会让这一期同时背三条互不相干的判据，而那
正是本设计自己禁止的（「M2–M4 各只加一件事，一期一条判据，便于变异验证时一格只违反一
条」）。所以**另立 M5**，格子清单见上面那张重筛表。

   M1 的两格不是「挑能过的那条」：它是**最小的、把全链路（bridge → region → C.5 →
   发射 → CPU 数值）走通一遍**的那一格。M2–M4 各只加一件事，一期一条判据，便于变异验证时
   一格只违反一条。

4. 输出

   每期一个可独立验收的闭环（验收手段见「测试设计」）。M1–M4 做完，那 17 格从
   `fallback` 变 `takeover`；另外 93 格要功能 3（dtype）、功能 5（算子）、F-4
   （plan selection）各自到位，**不在本设计射程内**——本设计只负责「VF 这一条不再是卡点」。

### 功能需求 6：量具与回归口径

1. 介绍

   VF 接通是个会把三张表一起改掉的改动，口径要先定好，否则下一轮读表的人会把「VF 接了」
   读成「能力涨了」。

2. 输入

   既有三台量具：`legacy_takeover_table.sh`、`vf_node_census.sh`、`af-capability-probe`。

3. 处理

   - 接管表：每期跑一遍 reg 侧 200 格，看 `takeover` 涨了几格、`fallback` 的首因分布
     怎么变。**`hard-failure` 不得增加**（一格都不行：那意味着能力不足漏到了 verifier）。
   - VF 普查：**口径不变**，它量的是 legacy 侧调度融不融 VF，与 MLIR 接不接无关——
     所以 110 这个数**不应该因为本设计而变**。变了说明量具或调度被动了。
   - 探针：`--reg-base --topo` 的那 40 格（四个菱形拓扑）今天全被 VF 拦下，M1 之后应当
     开始有格子变 `ok`；`vf=a/b` 那一列与结论的对应关系是验收依据之一。

4. 输出

   每期在本文件追加一行实测（表名 + 数 + 日期），不另开文档。

## 非功能需求

### 可维护性

- 新增算子、pass、移植文件各自单独成文件，命名跟随既有风格（`afir.vector_func`、
  `AFIRVectorFuncToAscVF.cpp`）。
- **移植来的代码要能看出是移植的**：文件头写明上游路径与对照的上游版本，便于后续跟进
  上游修改。不做「顺手重构」（红线 7）。
- 三张供给表（AFIR 能否表达 / 平台 API / reg 形态下降）各自一份真源，禁止手写第三遍。

### 可测试性

- 测试入口三层：lit（手写 AFIR → 看 IR 与 C++ 文本）、UT（bridge 的嵌套图翻译、能力判定）、
  E2E CPU（数值）。
- 桩与夹具：reg 侧 E2E 走既有 `tikicpulib_ascend950pr_9599`；SoC 名由
  `RuntimeStubV2::aclrtGetSocName` 给（上一轮已补成 `Ascend950PR_9599`，不要再动
  `rtGetSocVersion`）。
- **每期都要有一个会变红的东西**，不接受「读过代码觉得对」。lit 钉不住的（尺寸会被折叠、
  图里有队列才走得到的路径）放 UT。
- 变异验证：每期落地后对该期的判据做一次短路，确认**只有该期的用例变红**；若一格都不红，
  按「盲区／冗余／保守边界」三类分别处置。

### 可移植性

- 只用 CANN 对外头文件里的 `AscendC::Reg` 接口（红线 8）。
- **代码里不出现 SoC / NpuArch 字面量**；`__VEC_SCOPE__` 要不要发由 target profile 决定。
- 对 CANN 版本的依赖面：`Reg` 命名空间与 `MicroAPI` 别名、`GetVecLen()`、`LoadAlign` /
  `StoreAlign` / `UpdateMask` / `CreateMask` 的签名。版本变化的失败方式是编译错，不是静默
  算错——可接受。
- Python / wheel / run 包：不涉及。

### 可靠性

- 错误分两类且判据写死（功能需求 1 第 2 条的表）：产生方 bug → 硬失败；能力不足 → 干净
  回落。**不允许第三类**（静默跳过）。
- 资源：本设计不申请 runtime 资源、不开文件、不管生命周期跨 SO 的对象；MLIR 侧的
  `OwningOpRef` 沿用既有流水线的所有权约定（红线 4、12 不涉及新面）。
- 并发：codegen 单线程，不涉及。
- 内存安全：region 的 block argument 与边界张量一一对应，映射用显式序号校验（不靠下标
  默认对齐）。

### 特性交叉影响

逐项按 `docs/guidelines/cross_feature_check.md`：

| 场景 | 适用性 | 分析说明 |
|------|--------|----------|
| SuperKernel Python 接口 | 不适用 | 本设计只改 Autofuse 的 MLIR codegen 路径，不碰 `super_kernel/`，不改 Python 包与选项解析 |
| SuperKernel C++/AOT 接口 | 不适用 | 不导出符号、不改 `libascendsk.so`、无结构体/枚举布局变化 |
| Autofuse 图优化 | **适用（只读）** | 消费分区器产出的 VF 节点与嵌套子图，**不改写 ASCIR 图**、不新增/调整 pass、不改 pass 时序。子图遍历按拓扑序，保证结果确定（红线 5） |
| Autofuse Codegen/Backend | **适用** | 新增 AFIR region 算子、相位 C.5、reg 级下降；legacy 的 `vec_func_call` 一处不动，回落路径不受影响。新增 backend E2E 覆盖（多节点 reg-base） |
| AscendC API / Runtime 交互 | **适用** | 新增对 `AscendC::Reg` 一族的调用（生成代码里，不是 host 代码）。不新增 `rt*` / `aclrt*` 调用，无异步拷贝生命周期问题。`__simd_callee__` 的调用上下文约束由 `vec_scope` 保证 |
| Python/C++ 混合绑定 | 不适用 | 不碰 `compiler/py_module/` |
| 构建与打包 | **适用（小）** | 移植 `AscVF` 要在 `api/ascendc` 或新目录下加 CMake 目标并接进 MLIR 侧的库依赖；不改 `build.sh` 参数、不加第三方依赖、不改安装路径。要验 `--no-autofuse` 与增量构建仍可用（红线 10） |
| 测试与覆盖率 | **适用** | 每期 lit + UT + E2E CPU；先加会变红的用例再实现 |
| 性能与日志 | **适用** | 见「性能」。不新增默认开启的高频日志 |
| 兼容性 | 不适用 | 无对外 API/ABI、配置、产物布局变化；生成的 kernel 文本变化只影响 MLIR 这条路自己的产物 |

## 性能

### 编译时长

- Bridge 多遍历一遍子图：`O(子图节点数)`，与父图遍历同量级，可忽略。
- 相位 C.5 新增一趟 pass：只在有 `afir.vector_func` 的模块上做实事，空模块是一次遍历。
- 实现选择的表增大：功能需求 4 的 reg 形态表按 op × dtype 计，与既有
  `ImplementationRegistry` 同量级；收尾设计里对 83 个 SIMT op 全量接入估的是表变大约 70%，
  本设计不动那张表。
- 验收口径：M1 落地后比一次 `check-e2e-cpu` 全量时长，涨幅 > 5% 要给出解释。

### 执行性能

- **这是 B 路线的收益所在**：中间结果留在寄存器，不落 UB，搬运次数与 legacy 一致。
- 相对 legacy 的差别只在最后一步形态（B1 就地块 vs B2 外联函数），而**这一位已经量掉**：
  ccec 把两种形态编成同一段 `.vector.thread` 代码，指令直方图逐项相同（假设 1）。所以
  B1 不带来指令层面的性能差异。**没量的是上板时序**，以及更复杂子图上是否仍然等价。
- 相对今天（整图回落 legacy）的差别：接管后由 MLIR 产 kernel，性能差异来自整条产物而不是
  VF 这一段，属既有口径。
- 不新增 kernel launch、不新增拷贝、不新增默认日志。

### 内存和产物大小

- 常驻内存：新增一个方言与若干 pass，`af-opt` / `libaihac_codegen.so` 体积增加由移植的
  ~800 行（IR + `LowerToReg` + `MaterializeLoadStore`）决定，量级与一个既有 conversion
  目录相当。
- 生成 kernel 的大小：B1 比今天的回落产物**小**（不发外联函数、不发架构门）。
- UB 占用：不增加（这是否决路线 A 的那条理由的另一面）。

## 接口设计

### 新增/修改接口描述

全部是**内部接口**（MLIR 方言 / pass / 发射器），无对外 C/C++/Python 接口变化。

| 接口 | 形态 | 说明 |
|---|---|---|
| `afir.vector_func` | AFIR 算子，带一个 region、`Variadic<!ascendc.local_tensor>` 的 dst / src 列表、一个元素类型属性 | AFIR 的第一个 region 算子。verifier 只查结构性约束（边界数与 yield 操作数匹配、region 内无嵌套 VF），**不查能力**（能力在相位 C） |
| `afir.vector_func_yield` | region 终结算子 | 操作数是边界输出 |
| `createAFIRVectorFuncToAscVFPass()` | 相位 C.5 的 pass | 机械替换 + 调移植来的下降 |
| `ascvf` 方言（移植） | `vf_group` / `load` / `store` / `barrier` / `yield` | 上游接口，移植时不改签名；改了要在文件头记差异 |
| reg 形态供给查询 | 能力层新增一问 | 真源见功能需求 4，不手写清单 |

### 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|--------|----------|----------|------|
| 接口说明 | 是否需要接口评审 | 是 | `afir.vector_func` 是 AFIR 的第一个 region 算子，ODS 形态与 verifier 边界建议评审 |
| 接口说明 | 是否需要补充文档 | 是 | AFIR 稳定 ODS 文档门禁（`AFIR stable ODS doc gate`）会要求新算子带文档 |
| 接口兼容 | 行为是否兼容 | 是（向后兼容） | 今天回落的图将改为接管；回落路径本身不变 |
| 接口兼容 | ABI/API 是否兼容 | 不涉及 | 无对外导出变化 |
| 接口约束 | 约束不满足时是否清晰报错 | 是 | 功能需求 1 第 2 条的表逐条给判据与处置 |
| 接口测试 | 是否需要独立接口用例 | 是 | lit（算子语法与 verifier）+ UT（bridge 翻译） |

## 软件设计

### 关键数据结构

| 结构 | 生命周期 / 所有权 | 说明 |
|---|---|---|
| 嵌套 `af::AscGraph`（VF 子图） | 由 impl graph 持有（`AddSubGraph`），bridge 只读 | 经 `FindSubGraph(sub_graph_name)` 取；**已实测能穿过候选收窄**（假设 3，四格 `found=yes`） |
| 边界映射 `{子图 index → 父图 input index}` | bridge 局部 | 来源 `vf_root_input_index`，缺失回退 `index`；越界即硬失败 |
| `afir.vector_func` 的 region | MLIR 的 op 所有权 | region 内算子由 MLIR 管理，无手工释放 |
| reg 形态供给表 | 进程常驻（与 `ImplementationRegistry` 同寿） | 真源见功能需求 4 |

### 关键技术/算法

- **边界语义转换**（功能需求 1 第 3 条）：子图里的 `Load`/`Store` 是 UB↔寄存器，父图里的
  `afir.load`/`afir.store` 是 GM↔UB。这两个同名概念混了，产物会多一次搬运或少一次搬运，
  而且两种都能编过——所以这一处的判据要有 lit 钉住（CHECK 计数式，不是「存在一条」）。
- **lane 循环的上界**：`loop_times = ceil(element_count / (GetVecLen() / sizeof(T)))`。
  legacy 用 `uint16_t` 存 `loop_times`（见生成产物），**我们这边要么同样截断、要么校验**；
  `element_count` 很大时 `uint16_t` 是否够用由合并轴的上界决定。
  **开工前要核**：这是红线 3（整数溢出/反转）在本设计里的落点，不能靠「legacy 这么写的」
  当理由。
- 复杂度：bridge `O(子图节点数)`；下降 `O(region 内算子数)`。

### 流程设计

主流程：

```
调度产物（含 VectorFunc 节点 + 兄弟子图）
  → 候选准备（ATT tiling；今天已通过，回填差异对射程内格子惰性，见假设 4）
  → Bridge：afir.vector_func { tile 级 AFIR 算子 }        [功能需求 1]
  → 相位 A/B：region 外物化，region 内不动                  [功能需求 2]
  → 相位 C：region 体选实现 → tile 级 ascendc.*            [既有]
  → 相位 C.5：→ ascvf.vf_group → LowerToReg                [功能需求 3]
       → emitasc.vec_scope { vf_for { reg 级序列 } }
  → 相位 D + 发射                                          [既有]
```

异常流程：

- 能力不足（功能需求 4 / 功能需求 1 的能力类）：`emitCapabilityShortfall` → 整图回落
  legacy，缺省档静默、`strict` 档变红并打 `gate=lowering shortfall=yes`；
- 硬失败（产生方 bug 类）：`emitError` → 两档都红；
- 回滚/清理：MLIR 的 pattern 失败会整体回滚该 pass 的改写（既有机制）；本设计不持有需要
  手工清理的资源。

### 对子模块的修改

| 目录 | 修改点 |
|---|---|
| `autofuse/mlir/dialect/AFIR/IR/Ops/` | 新增 `VectorFunc.td` / `.cpp`：算子、yield、verifier、文档 |
| `autofuse/mlir/bridge/ScheduleToAfir/` | `VectorFunc` 分支：子图读取、边界映射、能力/硬失败分流（今天是 `:1087` 的一条 `emitCapabilityShortfall`） |
| `autofuse/mlir/bridge/PlanSelector/` | **不改**：假设 3 已实测子图能穿过收窄，不需要显式搬子图 |
| `autofuse/mlir/conversion/AFIRToAscendC/` | 相位 B 四个 pass 的 region 边界；新增相位 C.5 的 pass 并挂进 `AFIRToAscendCPipeline.cpp`（相位清单与依赖注释要一起改，那份注释是真源） |
| `autofuse/mlir/api/ascendc/`（或新目录） | 移植 `AscVF` 的 IR + `LowerToReg` + `MaterializeLoadStore`，加 CMake 目标 |
| 能力层（`Implementations/` 一带） | reg 形态供给的那一问 + 真源/门禁 |
| `autofuse/mlir/test/` | lit / UT / E2E CPU 新用例（见测试设计） |
| `autofuse/mlir/test/probe/` | 探针按需加一列；`vf_node_census.sh` 口径不变 |

### 错误处理

#### 系统错误

- 内存申请：不新增手工分配；MLIR 的分配失败按既有方式（异常/abort）处理。
- 文件与编译：不新增文件操作。生成代码编不过的失败方式是 E2E 第二阶段编译错，
  有用例名可定位。
- runtime 调用：不新增。

#### 接口错误

- 对外错误码不变：MLIR 这条路对调用方只有「接管 / 不支持 / 失败」三态
  （`MlirCodegenStatus`），本设计不新增第四态。
- 用户可见提示：诊断文本按既有风格（`UnsupportedOperation: ...`），一条诊断要能说出
  是哪个 op / 哪个 dtype / 哪一道闸门。

## 安全检查

逐条核 `docs/guidelines/编码红线.md`：

| 红线 | 结论 |
|---|---|
| 1 敏感信息硬编码 | 不涉及：不新增密钥、账号、地址、大数组 |
| 2 外部输入作索引/长度/偏移前校验 | **涉及**：`vf_root_input_index` / 子图 `index` / 边界数都来自图属性，全部先校验范围再用（功能需求 1 第 2 条的表），越界即硬失败 |
| 3 整数溢出/反转/除 0 | **涉及**：`loop_times` 的 `uint16_t`、`element_count` 的 `uint32_t`、lane 偏移 `i * ELEMENT_PER_VECTOR_LENGTH`。开工前要核上界（见「关键技术/算法」第 2 条），不以「legacy 这么写」为理由 |
| 4 资源释放覆盖异常分支 | 不涉及新面：无手工资源 |
| 5 内存申请前判大小 | 不涉及：不新增分配 |
| 6 未指定求值顺序/UB | 按常规检视 |
| 7 禁止无理由扩大范围 | **本设计显式约束**：legacy `vec_func_call` 一处不动、分区器一处不动、不顺手重构移植进来的代码 |
| G1 API/ABI 兼容 | 无对外接口改动 |
| G2 禁硬编码芯片/平台/框架类型 | **涉及且有门禁**：AFIR/bridge/artifact 三层不得出现 NpuArch 字面量；B1 不发架构 `#if` 正是为此（也更省） |
| G3 图改写保持数据边与控制边等价 | **涉及**：本设计不改 ASCIR 图，但 AFIR 侧把一段算子收进 region——边界映射要同时覆盖数据边与（子图内的）执行顺序约束，不能只验输出张量 |
| G4 有时序依赖的 pass 要说明依赖 | **涉及**：相位 C.5 必须在实现选择与 AFIR→AscendC 之后（它吃 tile 级 `ascendc.*`），必须在相位 D 之前。这条要写进 `AFIRToAscendCPipeline.cpp` 的相位注释 |
| G5 图改写结果必须确定 | **涉及**：子图按拓扑序遍历，不依赖 `unordered_map` 顺序或指针地址 |
| G6 高频路径禁默认海量日志 | 遵守：不新增逐节点默认日志 |
| G7 `rt*`/`aclrt*`/AscendC 生命周期 | 不新增 host 侧调用；生成代码里的 reg API 调用上下文由 `vec_scope` 保证 |
| G8 禁非开放 runtime 接口 | 遵守：只用 CANN 对外头文件里的 `AscendC::Reg` |
| G9 Python/C++ 绑定引用计数 | 不涉及 |
| G10 构建脚本不破坏增量/离线构建 | **涉及**：移植新目标后要验 `build.sh --pkg -j 8` 路径、`--no-autofuse`、增量构建 |
| G11 测试临时产物不入库 | 遵守：生成的 kernel 与 census 产物都在 `build/` 与文档目录（表是人工归档的量具产物，不是构建产物） |
| G12 全局/静态析构不跨 SO | 不涉及新面 |

## 兼容性检查

- **脚本参数、配置、安装目录、动态库、Python 包**：无变化。
- **已有产物**：MLIR 这条路生成的 kernel 文本会变（带 VF 的图从「不生成」变成「生成一段
  `__VEC_SCOPE__`」）；legacy 产物一个字节不变。
- **历史用例**：今天因 VF 回落而变绿的用例，接管后走 MLIR 产物——这正是目的。
  回归口径见功能需求 6：`hard-failure` 一格都不许增加。
- **`AF_ENABLE_MLIR_CODEGEN`**：语义不变（`0/1/strict` 三档）。

## 测试设计

### 测试边界

- **测试入口**：
  - lit：手写 AFIR（含 `afir.vector_func`）→ `af-opt --afir-to-ascendc-pipeline`
    → 看 IR；再 `ascir-translate -mlir-to-ascendc` → 看 C++ 文本；
  - UT：`test/ut/bridge/`（嵌套图翻译、边界映射、能力/硬失败分流）、
    `test/ut/conversion/`（相位 C.5 的替换与下降）；
  - E2E CPU：`test/e2e/cpu/cases/` 新增多节点 reg-base 用例，走
    `check-e2e-cpu-<name>` 目标，CPU 模拟器验数值。
- **测试出口**：IR 文本、生成的 C++ 文本、CPU 模拟器的输出缓冲。
- **打桩**：reg 侧 SoC 名由 `RuntimeStubV2::aclrtGetSocName` 给；tiling 走既有
  `TilingStub`。
- **验证数据**：输出缓冲预置哨兵（不是 0），确保「没算」与「算成 0」分得开。

**今天的 reg-base E2E 一张 VF 都不产**：10 个 `*_reg_base_*` 用例的计算节点数全是 0 或 1
（2026-10-18 数过），够不上 `kMinVfNodesNum = 2`。所以 M1 的第一件事是加一个两节点的
reg-base E2E 用例——加完**今天它就是红的**（strict 档下 VF 回落判失败），这就是 M1 的
那个「会变红的东西」。

### 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|----------|------------|----------|----------|
| 功能 | 两节点 reg-base 图（`load → add → abs → store`）数值正确 | E2E CPU，哨兵 + 逐元素比对 | ST/E2E |
| 功能 | `afir.vector_func` 的语法、verifier 边界（边界数与 yield 匹配） | lit + UT | UT |
| 功能 | 子图边界语义：region 内不得出现 GM↔UB 搬运；搬运次数计数式 CHECK | lit（计数式，不用「存在一条」） | UT |
| 功能 | 相位 C.5 产物形状：`vec_scope` 内恰好一层 `vf_for`，reg 算子序列顺序与子图拓扑序一致 | lit | UT |
| 功能 | M2/M3/M4 各自那一件事（Cast 跨 dtype、Compare→Select、标量链/Duplicate/双层循环） | 每期一格 lit + 一格 E2E | UT/ST |
| 异常 | `sub_graph_name` 缺失、子图找不到 → 硬失败且诊断点名 | UT | UT |
| 异常 | `vf_root_input_index` 越界 / 边界序号重复 → 硬失败 | UT | UT |
| 异常 | region 内 op 没有 reg 形态 → **能力不足**（不是硬失败），`shortfall=yes` | UT + 探针桶 | UT |
| 异常 | VF 里嵌 VF → 能力不足 | UT | UT |
| 性能 | `check-e2e-cpu` 全量时长涨幅、生成 kernel 行数 | 手工对比（M1 前后各一次） | 手工验证 |
| 兼容性 | 缺省档行为不变（回落仍然静默）、legacy 产物逐字不变 | legacy 接管表 + legacy ST 全量 | ST |
| 特性交叉 | reg 侧 200 格接管表：`takeover` 增加、`hard-failure` 不增加 | `legacy_takeover_table.sh` | ST/E2E |
| 特性交叉 | VF 普查 110 这个数不变（它量 legacy 侧，与本设计无关） | `vf_node_census.sh` | ST |
| 特性交叉 | 探针 `DRIFT` 桶仍恒为 0；菱形那 40 格开始出现 `ok` | `af-capability-probe --reg-base --topo` | 手工验证 |

变异验证（每期必做）：短路该期的判据重编，确认**只有该期用例变红**。一格不红时按
「盲区／冗余／保守边界」分类处置，不默认当成冗余删掉。

### 测试命令

```bash
bash ./mlir-test.sh
```

```bash
make -C build -j 6 check-e2e-cpu-load_add_abs_reg_base_store_256x128
```

```bash
bash autofuse/mlir/test/probe/legacy_takeover_table.sh -R '_v2$' -j 6 > /tmp/takeover-reg.txt
```

```bash
bash autofuse/mlir/test/probe/vf_node_census.sh -t autofuse/mlir/docs/development/2026-10-17-legacy-takeover-table-reg.txt
```

预期：`mlir-test.sh` 四个目标（`check-ut check-lit check-st check-e2e`）全绿；
新 E2E 用例数值全对；接管表 `takeover` 按期增加、`hard-failure` 不增加；
普查仍是 110 / 90。

**换档提醒**：接管表与普查要 legacy ST 档
（`bash scripts/test/run_autofuse_test.sh -s -m backend --enable-mlir -j 6`），
跑完回 MLIR 档要删 `build/CMakeCache.txt`。

## 验收标准

开工前（**四条已于 2026-10-18 全部完成**，结论与量具见「假设和依赖关系」）：

1. ~~发射形态定下来~~ → **B1**。ccec（dav-c310）两种形态都编得过且指令直方图逐项相同；
   复跑：`bash autofuse/mlir/test/probe/vf_emission_form/run.sh [--device]`。
2. ~~vendored 漂移清单~~ → 无阻塞。195/216 逐字相同，`AscVF` 依赖面全是单向增量，
   85 个引用符号全在，建 op 写法要改 1 个包装 + 14 处调用。
3. ~~VF 子图穿过候选收窄~~ → 能穿过，四格实测 `found=yes`，收窄处**不需要改**。
4. ~~ATT 回填差异~~ → 差异真实（legacy kernel 先 ATT 后，MLIR 反过来），但射程内惰性：
   141 个含 VF 的 kernel 里只有 2 个走双层循环那一支，且都是 `indirect_load_*`。

**M1 验收结果（2026-10-19，全部达标）**：

| 验收项 | 结果 |
|---|---|
| 新增两节点 reg-base E2E 用例数值全对，且实现前是红的 | `load_add_abs_reg_base_store_256x128`：实现前在 VF 那道闸门上报能力不足，现在 CPU 模拟器上 32768 个元素逐个比对全对（输出预置哨兵 `-12345.0f`） |
| lit 钉住边界语义与产物形状（计数式 CHECK） | `afir-vector-func-reg-base.mlir`（寄存器声明 4 条、lane 循环恰好 1 层、`LoadAlign` 2 条 / `StoreAlign` 1 条、`afir.load` / `afir.store` 一条不剩）+ `afir-vector-func-unsupported.mlir`（三格能力不足） |
| reg 侧接管表：两格变 `takeover`，`hard-failure` 不增加 | takeover 3 → 5（正是 `add_abs_test` / `div_abs_test`）；**hard-failure 79 → 66**，零格变差，见 `2026-10-19-legacy-takeover-table-reg-m1.txt` |
| 探针 `DRIFT` 桶为 0 | 为 0；`FAILED` 也为 0。菱形那四个拓扑 40 格里 20 格（T5d / T5e）转 `ok`，另 20 格（T5 / T5c）干净回落，新增一桶 `vector func: no register-form lowering` |
| 变异验证：短路 M1 判据后只有 M1 用例变红 | 删掉算子表里 `abs` 那一行：lit 209 格只有 `afir-vector-func-reg-base` 变红，M1 的 E2E 变红，两格对照（`load_neg_reg_base_store` / `load_add_store`）仍绿；撤回后恢复 |
| `check-ut` / `check-lit` / `check-st` / `check-e2e` | 全绿 |

**M1 顺带补的两条能力判据**（都不是 VF 的事，是 VF 这道闸门此前替它们挡着的漂移）：

1. **dtype 不在 AFIR 供给表里**就报能力不足（`InitializeTensorValue`）。此前 bridge 叫得出
   bf16 这类名字，而拦住它的是算子 ODS 的类型域——verifier 失败、也就是硬失败。
2. **调度搬运的运行期下标解不出来**时能力层也答「不支持」（`scheduledTileIndicesAvailable`，
   与 pattern 共用一条判据）。此前能力层答「支持」、pattern 报 `failed to legalize`，正是
   探针 DRIFT 那一桶盯的形态。

这两条合起来把 13 格原本硬失败的图变成干净回落。**它们是 M1 必须做的**：不做，接管表上
hard-failure 从 79 涨到 89（实测），而那是把本可以回落的图变红。

**M2 验收结果（2026-10-19，达标）**：

| 验收项 | 结果 |
|---|---|
| 新增跨 dtype reg-base E2E 用例数值全对，且实现前是红的 | `load_cast_abs_cast_reg_base_store_256x130`（half → cast f32 → abs → cast half → store）：实现前在「边界混了元素类型」那条判据上报能力不足，现在 CPU 模拟器上 33280 个元素逐个比**相等**（取值全在 half 里精确可表示，所以不用近似比），输出预置哨兵 `half(-12345.0f)` |
| lit 钉住跨 dtype 多出来的那三件事 | `afir-vector-func-reg-base-cast.mlir`（lane 宽度 `UpdateMask<float>`、两条 dist 的枚举值与发射出来的名字、一来一回两个不同的 trait、四条寄存器声明各自的 dtype）+ `afir-vector-func-unsupported.mlir` 扩到五格 |
| reg 侧接管表：`takeover` 增加、`hard-failure` **不增加** | `cast_abs` / `cast_abs_float16_float` 从 `fallback` 变 `takeover`；全表外推 takeover 5 → 7、hard-failure 66 → 66、fallback 127 → 125。见下「只量 6 格的理由」 |
| 探针 `DRIFT` 桶为 0 | `--reg-base --topo` 下 `DRIFT` 与 `FAILED` 都是 0；桶表与 M1 逐行相同（那 140 张图里没有 cast 进 VF，**不该变**） |
| 变异验证：短路 M2 判据后只有 M2 用例变红 | 三条判据各做一次，两条验到数值：① 判据回退成 M1 的「全程同一个元素类型」→ 只有 `afir-vector-func-reg-base-cast` 与 `afir-vector-func-unsupported` 两格 lit 变红（另 208 格绿）、M2 的 E2E 变红，M1 的 lit 与 E2E 仍绿；② 两条 dist 都不打 → CPU 模拟器 `element 1` 就错；③ lane 宽度按 destination 取 → **第一趟 lane 的 64 个元素仍然全对**，从 `element 64` 起才错 |
| `check-ut` / `check-lit` / `check-st` / `check-e2e` | 全绿（lit 210/210，全程 0 FAILED） |

**只量 6 格的理由**（口径，下一轮复测照这条）：M1 基线表里**首因落在 VF 下降那一层的只有 6 格**
（`cast_abs`、`cast_abs_float16_float`、`abs_uint8`、`relu_uint8`、`load_brc`、
`load_compare_where_store`），其余 194 格停在 M2 一处都没动过的更早闸门上
（candidate-preparation、dtype 供给表、bridge 的节点类型、cube-fusion）。而 M2 的能力判据是
M1 的**严格放宽**——M1 收的图在 M2 下逐条判据同样通过、产物逐字不变（M1 的 lit 原样绿就是这一条
的证据）。所以新增硬失败只可能出现在那 6 格里，194 格按构造与 M1 同值，不必重编重跑。
实测那 6 格 `hard-failure` 为 0。**要刷新 `fallback` 的首因分布才需要跑全量**，那不是验收项。

剩下两格的归属：`load_brc` 报 `afir.mul`（M4）、`load_compare_where_store` 报 `afir.eq`（M3），
都是如实的现状；`abs_uint8` / `relu_uint8` 卡在 **bridge 的 VF 边界 dtype 白名单**上
（`ScheduleToAfir.cpp` 的 `isModeledDtype`，M1 写死 `float32 || float16`），**到不了本期改的那一层**
——详见下面那一段。

**uint8 那 2 格不归 M2，而且原因不是 VF（2026-10-19 实测，实验已退掉）**：

`abs_uint8` / `relu_uint8` 的首因是 bridge 的 VF 边界 dtype 白名单
（`bridge/ScheduleToAfir/ScheduleToAfir.cpp` 的 `isModeledDtype`，M1 写死
`float32 || float16`），所以它们**到不了本期改的那一层**。把白名单放开成
`+ uint8 + int16` 量了一次，两格都变**硬失败**，而且是同一条诊断：

```
error: 'afir.store' op transfer source and destination must have the same element type
```

所以真正的卡点是 **`afir.store` 不建模换 dtype 的搬运**，与 VF 无关——同一条诊断在 M1 基线里
已经让另外两格（`sign_uint8`、`uint8_logical_not`）硬失败了。放开白名单只是把一个既有缺口从
「干净回落」翻成「红」：接管表上 hard-failure 会 66 → 68。这正是 M1 给那条白名单写的注释
警告的形态，所以**白名单原样留着**。

读表要注意：这一条让分期表 M2 那一行的「含 uint8 两侧」也站不住。M2 实际拿到的是
`cast_abs` 与 `cast_abs_float16_float` 两格；`abs_uint8` / `relu_uint8` 要等
「搬运可以换 dtype」那一条，它既不在 VF 的射程里，也不在 2026-10-16 那份设计的功能需求 3
（dtype 供给表）里——**是第三件事**，建议单独立项，判据就是上面那条 `afir.store` 的诊断。

**M3 验收结果（2026-10-27，全部达标）**：

| 判据 | 实测 |
|---|---|
| 新增 reg-base E2E 用例数值全对，且实现前是红的 | `load_eq_where_reg_base_store_256x128`（四路取数、`eq` 比前两路、`where` 在后两路间选）：CPU 模拟器上 32768 个元素逐个**相等**（取值都在 float 里精确可表示），输出预置哨兵 `-12345.0F`；实现前（短路本期判据）红在 `'afir.eq' has no register-form lowering`——与接管表上那一格的首条诊断逐字相同 |
| lit 钉住边界语义与产物形状，计数式 CHECK | `afir-vector-func-reg-base-predicate.mlir`：`= ascendc.reg_tensor` 5 条 + `= ascendc.mask_reg` 1 条（两条 `IR-NOT` 把声明段封住）、`load_align_reg` 4 条、`store_align_reg` 1 条；`compare_reg` 的目的地与 `select_reg` 的 mask **绑成同一个 SSA 值**（`%[[PRED]]`）；CPP 段钉 `Reg::Compare<float, AscendC::CMPMODE::EQ>` 与 `CPP-NOT: Reg::Duplicate`。拒收那一侧 `afir-vector-func-unsupported.mlir` 加两格（谓词被当值消费、条件不是谓词），各只违反一条 |
| reg 侧接管表：takeover 增加、hard-failure 不增加 | 见下 M4 那一行（M3 / M4 合并量了一遍全量 200 格） |
| 探针 `DRIFT` 与 `FAILED` 都为 0 | 两侧都没有这两桶（reg 1380 行、mem 1381 行）。**而且逐字未变**：与短路本期判据那一版逐行 diff 为 0 行——探针那 2761 行图里没有一张含比较 / 选择，所以本期**不该**动它，这是预期结果不是「没效果」 |
| 变异验证：短路本期判据后只有本期用例变红 | 三条各做一次，两条验到数值：① 删掉算子表那两行 → lit 215 格只有 `afir-vector-func-reg-base-predicate` 与 `afir-vector-func-unsupported` 两格变红、本期 E2E 变红，M1 / M2 的三格 E2E（`load_add_abs` / `load_cast_abs_cast` / `load_abs_neg`）仍绿；② `Reg::Select` 的两路候选接反 → **lit 全绿**（文本上看不出不对），CPU 模拟器 `element 0` 就错；③ 把 lane 活性掩码当条件传给 Select → lit 那一格红（`%[[PRED]]` 绑定照出来了），数值从 `element 1` 起错 |

**M4 验收结果（2026-10-27，全部达标）**：

| 判据 | 实测 |
|---|---|
| 新增 reg-base E2E 用例数值全对，且实现前是红的 | `load_mul_abs_reg_base_store_256x128`（`abs(x*y)`）：32768 个元素逐个相等，输出预置哨兵；实现前红在 `'afir.mul' has no register-form lowering`——与接管表上 `load_brc_test` 的首条诊断逐字相同 |
| lit 钉住 | **这一行没有 lit，是有意的盲区**（同 `afir.neg` 那一行的规矩）：填错一个 API 在 IR 与 C++ 文本上都看不出不对，判据只能是数值。lit 这一侧改的是 `afir-vector-func-unsupported.mlir` 第一格（`afir.mul` -> `afir.sub`） |
| reg 侧接管表：takeover 增加、hard-failure 不增加 | 全量 200 格重量一遍（`2026-10-27-legacy-takeover-table-reg-m3-m4.txt`）：takeover **67 -> 69**、fallback 120 -> 118、**hard-failure 11 -> 11 不变**、out-of-scope 2 -> 2；没打出标签的行 0。**逐格对比 200 行里只有 2 行变**（`load_compare_where_store_test` 归 M3、`load_brc_test` 归 M4），198 行四列逐字未变，**0 格变差** |
| 探针 `DRIFT` 与 `FAILED` 都为 0 | 同 M3 那一行（本期只加一行算子表，探针那批图里没有 `mul` 进 VF 的） |
| 变异验证：短路本期判据后只有本期用例变红 | 删掉 `afir.mul` 那一行 → **只有本期那一格 E2E 变红**（诊断正是 `'afir.mul' has no register-form lowering`），lit 215/215 全绿，M1 / M2 / M3 的四格 E2E 仍绿 |

**「只量 6 格」那条口径这一轮没有沿用，改成量全量 200 格**，理由两条：

- 本期判据**是**前期的严格放宽，这一点要重新论证而且论证得过来：M3 新增的两条拒收判据
  （谓词的来路 / 去处）只在**表里有那一行**的算子上触发，而 M3 之前 `eq` / `where` 根本不在
  表里、整个被拒；`computeElementType()` 对 `RegOpResult::kValue` 的行返回的就是结果元素
  类型，与改动前逐字同值。所以 M1 / M2 收得了的图在 M3 / M4 下逐条判据同样通过。
  **两条独立旁证**：M1 / M2 的 lit 原样绿；探针两侧 2761 行逐字未变。
- 但「只量 6 格」那条口径的前提是**上一轮有一份同口径的全量基线**，而 2026-10-27 那份
  C1 重筛刚好提供了（同一台量具、同一环境、HEAD 内容）。既然基线在手，全量一遍比外推便宜
  （远端 24 并发约 35 分钟），而且能一次答完「0 格变差」。**所以不是口径变严了，是基线恰好
  在手。**

M1 验收：

4. 新增的两节点 reg-base E2E 用例在 CPU 模拟器上数值全对，且该用例在实现前是红的；
5. lit 钉住 region 的边界语义与产物形状（计数式 CHECK）；
6. reg 侧接管表：`add_abs_test` / `div_abs_test` 从 `fallback` 变 `takeover`，
   `hard-failure` 不增加；
7. 探针 `DRIFT` 桶为 0；
8. 变异验证：短路 M1 判据后只有 M1 用例变红。

M2–M4 验收：各期按功能需求 5 的格子清单，同样四件（用例先红、lit 钉住、接管表变化、
变异验证）。**「做完 17 格全部变 takeover」这句已经不成立**：那 17 格是 2026-10-18 的数，
2026-10-26 重量成 8 格、2026-10-27 重量成 7 格（其中 3 格出范围），M1–M4 做完后首因还落在
VF 这一层的只剩 2 格，都归新立的 M5。

全程不变量：

9. legacy 产物逐字不变、缺省档回落行为不变；
10. VF 普查仍是 110 / 90（这个数变了说明动了不该动的东西）；
11. AFIR/bridge/artifact 三层不出现 SoC / NpuArch 字面量（门禁自动查）；
12. 本文件在每期落地后追加一行实测记录；
13. VF 接通后拿 `indirect_load_store_test` 比一次 MLIR 与 legacy 的 tiling，确认假设 4 那条
    回填差异没有扩散到射程内（今天它惰性，是因为那一支只有 2 格命中）。

---

### 设计文档检查结果

- [x] **跨特性交叉影响**：已按 `cross_feature_check.md` 十项逐项填表（见「特性交叉影响」）。
  适用四项：Autofuse 图优化（只读消费，不改写图）、Autofuse Codegen/Backend（主改动面）、
  AscendC API/Runtime 交互（生成代码新增 `AscendC::Reg` 调用，无 host 侧 runtime 调用）、
  构建与打包（移植目标的 CMake）。SuperKernel 两项、Python 绑定、兼容性四项不适用并给了理由。
- [x] **编码红线**：已按 `编码红线.md` 通用 7 条 + Graph-autofusion 12 条逐条核（见「安全检查」）。
  涉及四条并给了方案：红线 2（图属性作索引前校验）、红线 3（`loop_times` 的 `uint16_t`
  上界要核，不以 legacy 写法为理由）、G2（分层门禁，B1 不发架构 `#if`）、G3/G4/G5
  （region 收编要同时覆盖执行顺序、相位 C.5 的时序依赖写进相位注释、子图按拓扑序遍历）。
- [x] **测试设计**：UT（bridge 翻译 / 能力分流 / 相位 C.5）、lit（算子语法、边界语义、
  产物形状，计数式 CHECK）、E2E CPU（数值，哨兵预置）、量具回归（接管表 / 普查 / 探针），
  每期一个会变红的用例 + 一次变异验证。已记明今天 reg-base E2E 一张 VF 都不产，M1 必须
  先加多节点用例。
- [x] **性能评估**：编译期（bridge `O(子图节点数)`、新增一趟 pass、不动实现注册表）、
  执行期（中间结果留寄存器是本路线的收益；B1/B2 的寄存器分配差异本环境验不出来，列为待确认）、
  产物与内存（不增 UB 占用；生成 kernel 比回落产物更小）。
