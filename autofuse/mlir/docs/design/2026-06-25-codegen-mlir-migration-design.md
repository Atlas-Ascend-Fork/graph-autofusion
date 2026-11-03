# autofuse Codegen 模块 MLIR 化迁移设计

> **2026-10-22 状态更新**：生产 MLIR Codegen 路由、AFIR→AscendC、legacy/MLIR 对照入口已落地，
> 开关为 `AF_ENABLE_MLIR_CODEGEN`；ATT/Schedule 完整迁移和 legacy 退役仍是长期目标。
> 以下旧日期的状态及代码盘点是历史基线，不能据此判断生产链路尚未接入。当前流程见
> [Pass pipeline](2026-08-05-afir-to-ascendc-pass-pipeline-design.md) 和 [文档索引](../README.md)。

> 状态：草稿。Phase 1 基础设施和 AFIR bridge 已实现，Codegen 完整迁移尚未完成
> 日期：2026-06-25
> 重要更新：2026-06-27 按 mlir-af 代码重新梳理；2026-07-05 启动 Phase 2 并复核 `codegen/`
> 现状复核日期：2026-08-06
> 已实现：可选 MLIR 构建、官方 PyAsc 接入、`af-opt`/LIT、ScheduledResult → AFIR 调试桥、AFIR dialect、
> symbolic 基础和 AFIR → AscendC 的 ABI/signature/loop 基础。
> 未实现：生产 MLIR Codegen 路由、完整 AFIR → AscendC conversion、legacy/MLIR 双路 A/B、ATT/Schedule
> 迁移和 legacy Codegen 退役。文中的 `AF_MLIR_CODEGEN` 仍是规划接口，当前代码中不存在。
> 复核说明：早期文档中的部分行号和模块归属已经失效，本文已按 2026-07-05 的代码重新定位。需要特别注意：
> `CheckGraphValidity` 位于 `codegen_graph_check.cpp:235`；`MaterializeTilerAndTPipe` 是计划抽取的函数，当前逻辑
> 仍在 `ParseGraph:1926-2040` 内；`tiling_key` 编号来自 `AppendFuncCall:3182-3197`；tiling data 字段名定义在
> `codegen_tiling_data.cpp:108-141`；cube 走 `codegen_kernel.cpp` 中的 `GenCube*` 和
> `codegen_tiling_cube_wrapper.h`，不在 `api_call/` 目录。新增耦合点见 4.1 节子任务 2 和 3。
> 范围：autofuse `codegen/` 模块
> 参考：Ascend-MLIR-gsr（`gitcode.com:niuyuhu/Ascend-MLIR.git`，分支 `dev-gaoshuer`，本地 `/Users/neo/Code/Ascend-MLIR-gsr`）

---

## 0. 设计文档检查结果（AGENTS.md 要求）

- [x] **跨特性交叉影响**：已按 `docs/guidelines/cross_feature_check.md` 逐项检查（见第 9 节）。本设计仅改 `codegen/` 内部，SuperKernel、`optimize`/`ascir`/`att` 不动；对外契约（`FusedScheduledResult` 入 / `CodegenResult` 出）不变。
- [x] **编码红线**：已读 `docs/guidelines/编码红线.md` 并逐项核对（见第 9 节）。本设计为纯重构/迁移，不新增硬编码芯片分支、不改对外 ABI/API、不改图改写确定性（codegen 是纯发射器无 pass）、资源生命周期在退役 Stage 显式处理。
- [x] **测试设计**：Stage 0 每抽一子步骤跑全场景回归；Stage 2-5 双路 A/B 数值+性能比对；Stage 6 MLIR-only 端到端 + longrun（见第 8 节）。
- [x] **性能评估**：Stage 1 引入 LLVM/MLIR 预编译与 PyAsc 源码集成会增加编译时长（单次，非运行期）；运行期 codegen 仍为一次性发射，kernel 执行性能由行为等价门禁保障不劣化（见第 7 节）。

---

## 1. 背景与目标

### 阅读说明

这篇文档内容较多，可以按下面的顺序阅读：

1. 先看 1.3 和 1.5，确认本次迁移的目标与边界。
2. 再看第 3 节，了解七个实施阶段和每个阶段的验收门槛。
3. 准备实施某个阶段时，再读第 4 节的代码锚点和任务拆分。
4. `mlir-af` 只用于说明哪些 MLIR 形态已经验证过。它不是产品实现，也不是行为基准。

全文只有一条主线：先把现有 Codegen 拆成可验证的步骤，再逐步把内部表示切换为 MLIR，最后删除 legacy
Codegen。迁移前后的外部输入、输出和 kernel 行为保持兼容。

### 1.1 背景

Autofuse 当前仍用 C++17 自定义 IR（`af::AscGraph`/`AscNode`，通过 protobuf 序列化）完成张量编译和代码生成。
仓库已经加入默认关闭的 MLIR 子工程和 AFIR 调试桥，但生产 Codegen 仍走旧路径。当前主流程如下：

```
ascir(reg_func) ─┐
graph_metadef ───┼─> optimize ─> [FusedScheduledResult] ─> codegen ─> Ascend C 源码串
ascendc/api ─────┤        ^                                       │
                 │        └── att(tiling gen) <───────────────────┘
                 └──> compiler/py_module(orchestrate optimize+codegen)
```

`codegen` 接收 `ascir::FusedScheduledResult`，输出 `CodegenResult`。输入定义在 `common/schedule_result.h`；输出定义
在 `codegen/codegen.h`，包含 `kernel`、`tiling_data`、`tiling`、`infer_shape` 和 `proto`。其中的 Ascend C 源码
随后由 Ascend 编译器编译为 kernel 二进制。

现有 Codegen 以 C++ 对象保存状态，再通过字符串拼接生成 Ascend C：

- `ApiCall` 是虚基类，各算子族重写 `Generate(...)` 并向 `std::stringstream` 写入 API 调用。
- `Loop` 是循环树。树节点表示轴，叶子保存 `ApiCall*`；`ConstructFromNodes` 建树，`Generate` 输出嵌套 `for`。
- `Tiler`、`TPipe`、`Tensor`、`TQue`、`TBuf` 和 `MergeScope` 管理轴、队列、缓冲区及其复用关系。
- 芯片差异散落在 `npu_arch` 和 `soc_version` 字符串分支中。

这种实现可以工作，但中间状态不易序列化，调度决策和发射逻辑也交织在一起。直接替换成 MLIR Op 只会保留原有
耦合，因此本设计把“内部解耦”设为第一步。

### 1.2 参考基准：mlir-af 的定位

本文把 Ascend-MLIR-gsr 的 `dev-gaoshuer` 分支简称为 `mlir-af`。

`mlir-af` 是实验性实现。它用 MLIR pass 重新实现了部分 schedule、tiling 和 codegen 流程，证明这条技术路线可以
跑通，但没有覆盖 Autofuse 的全部能力，也没有完成产品验收。

本设计只借鉴其中已经验证的 MLIR 表达和 pass 边界，不直接复制代码，也不以它的功能或行为作为产品基线。任何
`mlir-af` 未覆盖或简化的能力，都以当前 legacy Codegen 的行为为准重新设计。

代码核实得到三项结论：

- 实际 pass 体系由 `auto-fuse-*`、`ascendc-*` 和 `linalg-to-ascendc` 组成。早期文档中的 dev-nyh 五段式命名
  在代码中不存在。
- `auto-fuse-codegen` pipeline 包含 29 个 pass，覆盖图规范化、符号 Shape、调度建环、bufferize、AscendC
  lowering、tiling data 打包和最终翻译。具体顺序见 `lib/Conversion/AutoFuse/Pipeline.cpp:59-152`。
- 主路径使用 `AFIR`、PyAsc 提供的 `ascendc`/`emitasc` 以及标准 MLIR 方言。`AFIR` 是 AscGraph 的抽象表示，
  不是 protobuf 的逐字段映射。`TmTensor` 只用于 attention stub。

Autofuse 正式接入 PyAsc 时使用官方 `https://gitcode.com/cann/pyasc.git` 子模块和仓内 patchset，不依赖个人 fork。

### 1.3 目标

第一阶段是本设计的范围：在 MLIR 基础设施上复刻现有 Codegen 行为，并最终替换 legacy Codegen。

- "原样复刻"：覆盖 autofuse 当前支持的全部场景，输出 kernel 的运行行为（数值、性能在可接受范围）与现状一致；Ascend C 源码**允许不同**。
- "行为等价"而非源码级一致：降低迁移成本，用端到端数值 + 性能回归保障。
- “完整替换”表示删除 legacy 实现，让 MLIR 成为唯一路径，并跑通 `optimize → codegen → Ascend C → 编译 → 运行`。
- 双路 A/B 只用于迁移和验证，不是最终架构。
- 设计参考 `mlir-af` 已验证的形态，但实现和验收仍以 Autofuse 当前产品行为为准。

第二阶段另行设计：把 `npu_arch`、`soc_version` 和 per-op 分支收敛到硬件描述与注册接口。

### 1.4 mlir-af 未实现或简化的能力

下表列出 `mlir-af` 没有实现或只做了简化实现的 Autofuse 能力。迁移这些能力时不能直接对照 `mlir-af`，需要按
legacy 行为重新设计并通过回归验证。

| autofuse 职责 | mlir-af 状态 | 证据 |
|---|---|---|
| 缓存决策（`enable_cache`/cache condition 复算） | **未实现**，简化为 `insert-tile-buffers` 固定 GM→VECIN/VECOUT 提升 + 共享 VECOUT 启发式 | `lib/Conversion/AutoFuse/TileFuse/InsertTileBuffers.cpp:202-262`；grep `enable_cache` 零命中 |
| reduce-mean 尾块 Muls 补全 | **未实现**，mlir-af 只做 `reduce_sum`，无 mean 尾块 ×recip | `lib/Conversion/LinalgToAscendC/ComputeReduce.cpp:605`；grep `Muls\|recip` 零命中 |
| reduce `kAllLoad`(FullLoad) codegen | 枚举但 ∞-scored 永不选中，codegen 未完成 | `include/Conversion/AutoFuse/TilePlan.h:55-59` |
| reduce `kRCore` 前端切轴（两阶段 partial→combine） | 仅末端 `ascendc-rcore-combine` pass，前端切轴未完整 | `Pipeline.cpp:151`；plan §3.3 标待做 |
| CV fusion Stage1/Stage2 划分 | **不做 stage 划分**，用 `CubeKind::MatmulVecFuse` 单 kernel（cube 输出留 UB 给 vector 消费）；CubeEmitter 仅 Phase 4a（level-1 tile） | `TilePlan.h:30-38,106`；`lib/Conversion/AutoFuse/TileFuse/CubeEmitter.cpp:123` |
| `tiling_key` | **完全移除**（v2.0 简化），用 `auto_fuse.tiling_infos` schema v2 替代 | `docs/AscGraph_Mapping.md:676`；grep `tiling_key` 零命中 |
| 多核分解（bind_block + 多轴） | 简化版，`ascendc-parallelize` 只替换最外层一个带 `ascendc.parallel` 的 scf.for | `lib/Conversion/AscendCParallelize/AscendCParallelizePass.cpp:57` |
| `score_func` 真正调参 | 占位 `costEstimate`，调参留后续 | plan §3.6/§8 |
| AFIR 与 AscGraph protobuf 1:1 | **不是**，AFIR 是简化抽象（移除 `SchedInfoDef`/`ApiInfoDef`/`tiling_key`，Mem 三合一，DataType 用 MLIR 类型） | `docs/AscGraph_Mapping.md:7,134-178,461-470,676` |

### 1.5 范围边界

| 维度 | 本阶段处理 | 本阶段不处理 |
|---|---|---|
| 模块 | `codegen/` 内部 | `optimize/`、`ascir/`、`att/`、`graph_metadef/` |
| 对外契约 | 保持不变 | 不改 `FusedScheduledResult` 入参、`CodegenResult` 出参结构 |
| MLIR 介入深度 | codegen 内部工作 IR 改为 MLIR | 上游不直接产出 MLIR |
| 场景覆盖 | 逐算子族横向扩展至全场景 | 无 |
| 分支收敛 | 无 | npu_arch/soc_version 字符串分支（留第二阶段） |

### 1.6 关键约束

1. 每个 Stage 都要通过数值和性能回归，失败时不进入下一阶段。每个 Stage 都应能独立交付和回滚。
2. Stage 0 先用纯 C++ 拆开现有流程并冻结中间产物。Codegen 自身主要负责发射，当前夹杂的调度职责要先显式
   分离，后续才能对应到合适的 MLIR pass 或 emitter。
3. PyAsc 通过 `add_subdirectory` 与 Autofuse 一起编译，提供 `ascendc` 和 `emitasc` 方言；LLVM/MLIR 使用匹配
   ABI 的预编译产物。
4. 第一条纵向链路覆盖 elementwise 和 broadcast，随后加入 elementwise 和 reduce，最后扩展到全部场景。
5. Stage 6 的完成条件是 legacy Codegen 已删除且 MLIR-only 全流程通过。仅完成双路覆盖不算迁移结束。

---

## 2. 整体架构

### 2.1 目标架构（MLIR 作为 codegen 内部工作 IR）

迁移后 codegen 内部数据流：

```
                ┌─────────────────────────────────────────────────────────────┐
  ascir::       │  1. 桥接器 (Builder)                                         │
  FusedScheduled│     FusedScheduledResult ──────────────>  MLIR region        │
  Result        │       (轴/管/复用/调度信息)            (scf.for 循环树 +      │
  (不变)        │                                          ascendc.* op +      │
 ───────────────▶                                          emitasc 类型 +      │
                │                                          codegen.* 属性)     │
                │                                                              │
                │  2. 统一 printer (Ascend C emitter)                          │
                │     注：codegen 本质是纯发射器，不做 IR 变换。                │
                │     actual-size/cache条件/CV阶段/管分配 均为                  │
                │     发射期局部计算或读上游决策后翻译，无 pass 层。             │
                │              ▼                                               │
                │     MLIR region ───────────────────>  Ascend C 源码串        │
                │              │                          │                    │
                │              │     + tiling_data / tiling / infer_shape     │
                │              │     （由 att + 结构化导出，行为等价）          │
                │              ▼                          ▼                    │
                └─────────────────────────────────────────────────────────────┘
                                          CodegenResult (不变)
```

迁移后的变化可以归纳为四点：
- `ApiCall` 子类的 `Generate()` 从"拼字符串"变为"构造 `ascendc` op"。算子族仍以类组织，但产出 MLIR op 而非 `std::stringstream`。
- `Loop` 树映射为 `scf.for` 嵌套。
- `Tiler`/`TPipe`/`Tensor` 的运行时状态（`axis_map`、队列/缓冲分配、复用链）编码为 **MLIR 属性 + `emitasc` 类型**，挂在 op 上；这些对象退化为"构造 MLIR 的 builder"而非一等运行时状态持有者。
- Codegen 入口接收 `const FusedScheduledResult&`，不改写上游图。现有 actual-size、cache 条件、CV 阶段和复用链
  要么是发射期局部计算，要么读取上游决策。因此这里没有一组现成的“Codegen pass”可以直接迁移。迁移后，跨 Op
  需要共享的信息进入属性，纯局部计算留在 MLIR 到 Ascend C 的 emitter 中。
- 末端 Ascend C 发射由**统一 printer/emitter** 完成（对照 mlir-af `CannTranslation.cpp` 的翻译职责），替代现状散落的 `FormatIndentation` 字符串拼接。

### 2.2 方言与 MLIR 设施选型

| 设施 | 来源 | 用途 | mlir-af 对照 |
|---|---|---|---|
| `ascendc` 方言 | PyAsc（源码集成，与 mlir-af 同 `externals/pyasc`） | 低层 Ascend C kernel op（`ascendc.data_copy`、`ascendc.add`、reduce、cube 等） | mlir-af `ascendc` = pyasc Asc 方言（`mlir::ascendc`） |
| `emitasc` 方言 | PyAsc（源码集成） | 发射级类型/属性（如 `!emitasc.py_struct<"TilingData">`）、queue/buf 描述、`ReinterpretCastOp` | mlir-af `emitasc` = pyasc EmitAsc 方言（`emitasc::PyStructType` 等） |
| `scf`/`arith`/`math`/`memref` | 上游 MLIR | 循环嵌套、索引算术、缓冲引用 | mlir-af 主路径用 scf.for + arith |
| `codegen.*` 自定义属性 | 本项目新增 | 轴标识（`logicalAxisId`）、轴大小表达式、复用链、CV 融合阶段等 codegen 内部语义。**只定义属性/类型，不另开 op 方言** | mlir-af 用 `auto_fuse.tiling_infos` module attribute 承载调度决策；本项目因不迁 schedule，仅需 codegen 内部语义属性 |
| LLVM/MLIR | 预编译（`find_package(MLIR)`） | tablegen、pass 基础设施、printer | mlir-af 用 `find_package(MLIR REQUIRED CONFIG)` + `externals/llvm-project/build` |

**为什么自定义属性而非完整 codegen 方言**：codegen 内部语义（轴/管/复用）主要是"挂在 ascendc op 上的元数据"，用属性 + `emitasc` 类型承载即可，不必为每个 codegen 概念定义 op。这保持与 `ascendc`/`emitasc` 的清晰边界，减少 tablegen 维护负担，也为未来与 mlir-af/Ascend-MLIR 统一留路（属性可平滑迁移）。

---

## 3. 分阶段总览

> 命名说明：本文的 Stage 0 到 Stage 6 是迁移 tracker 中 Phase 2（Codegen 模块交付）的内部实施阶段，
> 与 tracker 的 Phase 0 到 Phase 6（跨模块里程碑）不是同一组编号。下文所说的“阶段”均指 Stage。

迁移分为七个 Stage。每个 Stage 通过行为等价门禁后才能进入下一阶段，并且都应能独立交付和回滚。

| Stage | 名称 | 交付物 | 等价门禁 |
|---|---|---|---|
| 0 | 内部解耦重构（纯 C++，不碰 MLIR） | 拆分混合在一起的 Codegen 步骤，冻结中间产物，并明确调度职责的归属；不改外部契约和上游 | 每拆一个步骤都通过现有全场景回归 |
| 1 | 基础设施就绪 | PyAsc 源码集成 + LLVM/MLIR 构建 + `codegen/` CMake 接入 + 现状回归基线建立 | 现状全场景回归基线可复现 |
| 2 | 薄桥接纵向打通 | elewise+brc 端到端：`FusedScheduledResult→MLIR→AscendC`，行为等价 | elewise+brc 数值+性能回归过 |
| 3 | 发射下沉为 MLIR op | `ApiCall` 族逐个从字符串→`ascendc` op；`Loop→scf.for`；统一 printer | elewise+brc 仍等价（含逐算子回归） |
| 4 | 状态下沉为 MLIR 属性 | `Tiler`/`TPipe`/`Tensor` 运行时状态→`codegen.*` 属性 + `emitasc` 类型 | elewise+brc + elewise+reduce 等价 |
| 5 | 算子族横向扩展 | reduce/concat/transpose/gather/cube … 逐族迁移，覆盖现状全场景 | 全场景回归过（仍双路并存） |
| 6 | 替换与退役（第一阶段终点） | 删除 legacy Codegen，MLIR 成为唯一路径；跑通 `optimize→codegen→Ascend C→编译→运行` | 全场景 MLIR-only E2E 通过、无 legacy 残留、双路开关已删除 |
| 后续 | 第二阶段，另开设计 | 将 npu_arch/soc_version 分支收敛到 HW 描述符和注册表 | 全场景回归通过、无裸字符串分支 |

### 3.1 Stage 依赖与并行性

```
0 (内部解耦重构, 纯C++)  ← 关键前提：先解耦，再谈 MLIR
 └─ 1 (基础设施)
     └─ 2 (薄桥接纵向)  ← 验证端到端可行性与回归框架
         └─ 3 (发射下沉)
             └─ 4 (状态下沉)
                 └─ 5 (横向扩展)  ← 逐算子族，可多人并行（双路并存）
                     └─ 6 (替换与退役)  ← 第一阶段（本设计范围）终点：legacy 移除、MLIR-only 全流程跑通
                         └─ [第二阶段] npu_arch 收敛（另开设计）
```

Stage 0 必须先处理四个现有问题：`Kernel::ParseGraph` 同时做图解析、状态物化和 Loop 建树；
`Loop::ConstructFromNodes` 在建树时混入发射决策；`Tiler.actual_sizes` 在构建和发射阶段都可修改；kernel 与 tiling
之间依赖没有显式契约。此外，tiling 派生、循环结构化、缓存决策、多核分解和 CV fusion stage 划分本应属于调度，
却留在 Codegen 中。Stage 0 把这些职责拆成独立步骤并冻结产物，后续才能逐项映射为 MLIR pass、属性或 emitter
逻辑。

Stage 5 的算子族相互独立，可在 Stage 4 完成后并行推进。只有 Stage 5 的全场景双路回归全部通过，才能进入
Stage 6 并删除 legacy 实现。

### 3.2 codegen 职责的真实现状与 mlir-af 对照（解耦前提）

当前 Autofuse Codegen 不只是翻译层。除了生成代码，它还负责 tiling 派生、循环结构化、缓存决策复算、多核分解、
CV fusion stage 划分和 reduce 尾块补全。这些职责中有一部分本应属于上游调度。

下表列出每项职责的现有位置、在 `mlir-af` 中的对应层，以及 `mlir-af` 是否已经实现。实施时可以据此判断某项
能力能否参考现有 MLIR 形态。

| 职责类别 | codegen 现状位置 | 本属 mlir-af pass/translation | mlir-af 是否实现 | 本次处理 |
|---|---|---|---|---|
| 纯翻译：ApiCall 子类 Generate 拼 ascendc 调用、GlobalTensorInit、TPipe alloc、tiling_data struct、tiling_key 分支、顶层组装 | `api_call/*`、`codegen_kernel.cpp` GlobalTensorInit/TensorAlloc、`codegen_tiling_data.cpp`、`codegen_tiling.cpp`、`codegen.cpp` | **`linalg-to-ascendc`**（compute→ascendc op、data_move→ascendc op、建 Pipe/Queue/TBuf，`lib/Conversion/LinalgToAscendC/`）+ **`CannTranslation`**（末端翻译：GlobalTensorInit→`emitSupportedMixAicGlobalTensorSetup`@1697、TPipe alloc→`<<"TPipe pipe;"`@2006、顶层组装→`extern "C" __global__`@1999）+ **`ascendc-pack-tiling-data`**（tiling_data PyStruct 物化） | ✅ 已实现 | 保留为 codegen 本职 |
| 图规范化/校验 | `codegen.cpp:101` `EnrichScheduledResultAscirParams`（调用 `:465`）、`codegen_graph_check.cpp:235` `CheckGraphValidity`（`codegen_kernel.cpp:1844` 仅调用点） | **`auto-fuse-isolate-kernel-outputs`**（输出隔离规范化，`IsolateKernelOutputs.cpp:31-87`）+ **`auto-fuse-verify-tiling-info-schema`**（schema↔IR 校验，`VerifyTilingInfoSchemaPass.cpp:27-160`） | ✅ 已实现（无单一等价，分散承担） | Stage 0拎出为独立步骤 S0 |
| 调度上下文物化（IO/workspace + 轴/管静态状态） | 内联于 `ParseGraph:1926-2040`（**无 `MaterializeTilerAndTPipe` 函数，待抽取**；IO 解析 `:1845-1918`） | **`afir-symbolize-shapes`**（铸造轴符号 + extent 表达式 → `afir.dim_symbols`/`symbolic_shapes`/`iter_extents`，`AFIRSymbolizeShapes.cpp:66-417`）+ **`afir-symbolic-dim-cse`** | ✅ 已实现 | Stage 0显式化为独立步骤 S1 |
| tiling 派生（`is_split_b`/`actual_size`/`tail_size`/`loop_size`/`axis_size`） | `AddAxisSplitBAttr:476`(`is_split_b:479`)、`GenAxisSizeNew:692`、`GenInnerLoopSizeAndActualSize:726`(写 actual_sizes `:741`)、`codegen_kernel_loop.cpp` `GenerateLoop:602`(写 `:653`)/`ActualSizeDefine:804`(写 `:818`) | **`auto-fuse-tile-fuse`** 内 `TilePlanGen`(枚举+选优)+`TilePlanBuild`(物化)+`SliceComputer`(tail 切片)，决策结构 `TilePlan{TileLevel×AxisRole}`（注意：mlir-af 用 `XBLOCK`/`M_INNER`/`K_INNER` 系命名，非 dev-nyh 的 `is_split_b` 系） | ✅ 已实现 | Stage 0显式化为独立步骤 S3 |
| 循环结构化（`ConstructFromNodes`/`LoopAxisDistance`） | `codegen_kernel_loop.cpp:242-368,825-866` | **`auto-fuse-tile-fuse`** 内 `LoopNestBuilder`（按 Outer→BCast-Full→Inner 建 scf.for 嵌套 + tail peel + parallel 标注，不复用 `linalg::tileUsingForOp`，`LoopNestBuilder.cpp:13-131`） | ✅ 已实现 | Stage 0显式化为独立步骤 S2 |
| 缓存决策复算（`IsNodeSplitB`/`IsReduceDoubleTile`/`GenerateEnCacheCondition`） | `codegen_kernel_loop.cpp:179-210,423-439,736-800` | **无独立 pass**：简化为 `auto-fuse-insert-tile-buffers` 对 GM 输出 linalg.generic 固定提升 GM→VECIN/VECOUT + 共享 VECOUT 启发式（`InsertTileBuffers.cpp:134-308`） | ❌ 未实现条件决策 | Stage 0显式化为独立步骤 S4；**无法照 mlir-af 对照，自行设计** |
| 多核分解（`BlockOutterAxisDefine`/`CalcFromAxis`） | `CalcFromAxis:747`、`BlockOutterAxisDefine:775/795`（调用 `:2096`） | **`ascendc-parallelize`**（把 `ascendc.parallel` 标注的外层 scf.for 转 `get_block_idx`+`arith.muli`+越界 scf.if 守卫，`AscendCParallelizePass.cpp:148-170`；block_dim 推导在 `TilePlanGen` 的 `blockDimExpr`） | ⚠️ 简化版（每层只做一维 block_idx×step，AF 的 bind_block+多轴更复杂） | Stage 0显式化为独立步骤 S5 |
| CV fusion Stage 划分（`GetLifecycleEdge`/`InitApiCallContext`） | `codegen_kernel_loop.cpp:212-240` | **`auto-fuse-restore-matmul`**（还原 matmul 命名 op，`RestoreMatmul.cpp:27-62`）+ **`auto-fuse-tile-fuse`** 内 `CubeEmitter`（CubeKind::MatmulVecFuse 单 kernel 三级 tile+unit 标注，`CubeEmitter.cpp:104-312`）。mlir-af **不做 Stage1/Stage2 划分** | ❌ 未实现 stage 划分 | Stage 0显式化为独立步骤 S6；**无法照 mlir-af 对照，自行设计** |
| reduce Mean 尾块 Muls 补全 | `codegen_kernel_loop.cpp:666-705` | **无**（mlir-af 只 `reduce_sum`，`ComputeReduce.cpp:605`，无 mean 尾块 ×recip） | ❌ 未实现 | Stage 0归入 S7 lowering 语义补全；**无法照 mlir-af 对照，自行设计** |
| reduce 三模板（kCommon/kAllLoad/kRCore） | reduce 发射相关 | 枚举/选取/stamp 在 **`auto-fuse-tile-fuse`**(TilePlanGen，`afir.reduce_template` attr)；Common emit 在 **`linalg-to-ascendc`**(ComputeReduce)；RCore 二段式 combine 在 **`ascendc-rcore-combine`** | ⚠️ 部分（Common 完成；FullLoad ∞-scored 未编；RCore 仅末端 combine） | Stage 0标注本属 S3/S7，留意 FullLoad/RCore 未完成需自行补全 |
| `tiling_key` 编号 + tiling_data 字段名（隐式契约） | tiling_key 编号 `AppendFuncCall:3182-3197`(计数器初始化 `:2331/:2365`)、字段命名 `codegen_tiling_data.cpp:108-141`(`TILING_DATA_FIELD_DEF_T`/`common_tiling_fileds`) | **`auto-fuse-tile-fuse`**(TilePlanGen 生成 `auto_fuse.tiling_infos` schema v2) + **`auto-fuse-verify-tiling-info-schema`** + **`ascendc-pack-tiling-data`**。mlir-af **已移除 `tiling_key`** | ❌ mlir-af 已废弃该概念 | Stage 0 抽 `TilingKeyScheme`+`TilingDataFieldNamer` 共享层（本项目保留 `tiling_key`，行为等价优先） |
| 末端 finalize/flatten/signature | codegen 顶层组装内 | **`ascendc-finalize-kernel`**(func 定型为 aicore global void + 消除 affine)、**`ascendc-flatten-gm-ptr`**(GM 访问归一化为 flat 1D ptr+offset)、**`canonicalize-cann-signature`**(规整 AscendC kernel 签名 + 计算输入计数元数据)，均在 `lib/Conversion/AscendCPrepareForEmit/` | ✅ 已实现 | codegen 本职，迁移时对应到这些 pass |
| 上游已显式、codegen 只读（`exec_condition`/`reuse_id`/`alloc_type`/`cube_type`/`sched.axis`/`axis.type`） | codegen 各处只读 | 上游（mlir-af 侧对应 schedule 前 pipeline `auto-fuse`） | 不适用 | 本次不动，标注本属上游 |

结论分为三点：

1. Codegen 本身对应 `linalg-to-ascendc`、末端 AscendC pass 和 `CannTranslation`，职责是 lowering 与发射。
2. tiling 派生、循环结构化、CV/reduce 模板枚举、多核和缓存等职责属于调度或内存规划，在 `mlir-af` 中主要由
   `auto-fuse-tile-fuse`、`ascendc-parallelize` 和 `insert-tile-buffers` 处理。
3. 缓存条件、CV stage 划分和 reduce-mean 尾块补全在 `mlir-af` 中没有等价实现。这三项要根据 legacy 行为单独
   设计，不能照搬实验性实现。

Stage 0 的任务就是把上述职责变成边界明确的步骤和冻结产物，并记录每一步未来应进入 pass、属性还是 emitter。

---

## 4. 各 Stage 详细设计

### 4.1 Stage 0：内部解耦重构（纯 C++，不碰 MLIR）

目标：用纯 C++ 把现有流程拆成可独立测试的步骤，并冻结每一步的输出。拆分时明确哪些逻辑属于 Codegen，哪些
属于调度，以及 `mlir-af` 是否已有对应实现。这个 Stage 不引入 MLIR，不改外部契约和上游行为。

**目标子步骤序列（解耦后的 codegen 内部流程）**：每个步骤输入显式数据结构、输出冻结的中间产物，标注本属 mlir-af MLIR 层 + mlir-af 对照状态。

```
FusedScheduledResult (冻结, 不改上游)
 │
 ├─S0 图规范化           [mlir-af对照: auto-fuse-isolate-kernel-outputs + auto-fuse-verify-tiling-info-schema]  ← EnrichScheduledResultAscirParams + CheckGraphValidity 拎出
 │   产: 规范化 FusedScheduledResult (冻结)
 │
 ├─S1 调度上下文物化      [mlir-af对照: afir-symbolize-shapes + afir-symbolic-dim-cse]  ← 拆 ParseGraph 前半: IO/workspace 解析 + Tiler/TPipe 静态状态物化
 │   产: KernelState (冻结: inputs/outputs, AxisContext←axis_map, BufferContext←tensors/queues/bufs)
 │   注: axis_map/tensors 在此冻结为不可变，解除 TPipe 反向引用 Tiler
 │
 ├─S2 循环结构化          [mlir-af对照: auto-fuse-tile-fuse 内 LoopNestBuilder]  ← ConstructFromNodes 只建树, 剥离所有决策
 │   产: LoopTree (冻结: 纯结构, Loop 节点 + ApiCall 叶子, 无决策注解)
 │   注: LoopAxisDistance 折叠算法保留, 但 enable_cache/stage/reuse 链全部剥出
 │
 ├─S3 tiling 派生         [mlir-af对照: auto-fuse-tile-fuse 内 TilePlanGen+TilePlanBuild+SliceComputer]  ← is_split_b/actual_size/tail_size/loop_size 推导独立成纯计算
 │   产: TilingDerivation (冻结表: actual_sizes 不再是 Tiler mutable 成员, 而是显式产物)
 │   注: Tiler::GenInnerLoopSizeAndActualSize/GenAxisSizeNew/AddAxisSplitBAttr 收敛到此
 │
 ├─S4 缓存决策派生        [mlir-af对照: ❌ 无独立 pass，mlir-af 简化为 insert-tile-buffers 固定提升。自行设计]  ← IsNodeSplitB/IsReduceDoubleTile/enable_cache 复算独立
 │   产: CacheDecision (冻结: 每 ApiCall 的 enable_cache/enable_cache_with_condition)
 │   注: 读上游 exec_condition + S3 派生, 不再在 ConstructFromNodes 里写死
 │
 ├─S5 多核并行映射        [mlir-af对照: ascendc-parallelize（⚠️简化版，AF bind_block+多轴更复杂，自行补全）]  ← BlockOutterAxisDefine/CalcFromAxis 分解独立
 │   产: ParallelMap (冻结: block_dim 到 BlockOuter 轴的索引分解)
 │
 ├─S6 CV fusion 阶段划分  [mlir-af对照: ❌ mlir-af 不做 stage 划分（restore-matmul+CubeEmitter 用 CubeKind::MatmulVecFuse 单 kernel）。自行设计]  ← GetLifecycleEdge/InitApiCallContext 独立
 │   产: CVStageAssignment (冻结: 每 ApiCall 的 Stage1/Stage2 归属)
 │
 ├─S7 算子发射            [mlir-af对照: linalg-to-ascendc]  ← ApiCall 子类 Generate 拼 ascendc 调用 (纯翻译, 读 S1-S6 冻结产物)
 │   产: per-call Ascend C 片段
 │   注: reduce Mean Muls 补全归此 (mlir-af 未实现，自行保留)
 │
 ├─S8 函数体组装          [mlir-af对照: CannTranslation.cpp 末端翻译（GlobalTensorInit/TPipe alloc/顶层组装）]  ← GlobalTensorInit/TPipe alloc/Loop::Generate 读产物组装
 │   产: kernel 函数体字符串
 │
 ├─S9 tiling_data/tiling  [mlir-af对照: auto-fuse-tile-fuse(schema生成) + auto-fuse-verify-tiling-info-schema + ascendc-pack-tiling-data]  ← tiling_data struct + tiling func (att 回调)
 │   产: tiling_data/tiling/infer_shape 字符串
 │   注: 抽 TilingKeyScheme+TilingDataFieldNamer 共享层, 消除 kernel/tiling 隐式契约（mlir-af 已移除 tiling_key，本项目保留）
 │
 └─S10 源码组装           [mlir-af对照: CannTranslation.cpp 末端翻译]  ← IncludeAndDefines + CombineTilings 拼接
     产: CodegenResult (不变)
```

**子任务**（按收益/风险排序）：

1. **拎出图规范化 S0**：把 `EnrichScheduledResultAscirParams`（`codegen.cpp:94`）从 `GenerateKernel`（`codegen_kernel.cpp:380`）移出，作为 `Codegen::Generate`（`codegen.cpp:311`）第一步独立调用；`CheckGraphValidity`（`codegen_kernel.cpp:1793`）随之独立。代价极小。

2. **冻结 actual_sizes → S3**（收益最大、风险最小，先做）：把 `Tiler::actual_sizes`（`codegen_kernel.h:242` mutable）改为显式产物 `TilingDerivation` 表。**发射期共有 3 处写点，冻结必须全覆盖**：`Loop::GenerateLoop`（写 `codegen_kernel_loop.cpp:653`）、`Tiler::GenInnerLoopSizeAndActualSize`（写 `codegen_kernel.cpp:741`）、`Loop::ActualSizeDefine`（写 `codegen_kernel_loop.cpp:818`，经 `codegen_kernel.cpp:3860` `root_loop.ActualSizeDefine` 调用）。**注意写路径经 `tpipe.tiler` 别名**（TPipe 持 `const Tiler&` 反向引用，见子任务 3），非直接 `Tiler`。改为 S3 纯计算产出冻结表，S7/S8 发射期只读；`Tiler::ActualSize` 改为接收 `const TilingDerivation&`。这一步让 `Loop::Generate` 立刻变纯读，为后续抽步骤打基础。

3. **拆 ParseGraph 三段 → S1 + S2**（耦合最深，核心）：把 `Kernel::ParseGraph`（`codegen_kernel.cpp:1841-2043`，约 202 行三合一）拆为下列**待抽取的新函数**（现均内联，无既有函数名）：
   - S1a `ParseKernelIO`（现内联 `:1845-1918`）：只填 inputs/outputs/workspaces。
   - S1b `MaterializeTilerAndTPipe`（**当前无此函数，物化逻辑内联于 `:1926-2040`**，本子任务负责抽取）：填 axis_map/tensors/queues/bufs，**返回不可变 KernelState**，解除 `TPipe` 持有 `const Tiler&` 的反向引用（`codegen_kernel.h:290`；构造 `:305`）。**约束**：`Kernel::tiler`（`codegen_kernel.h:398` 值成员）须先于 `tpipe` 构造且地址稳定，抽取时保持该构造顺序依赖。
   - S2 `BuildLoopTree`（`ConstructFromNodes` 调用在 `:2042`）：只建纯结构树，不做任何决策。

4. **剥 ConstructFromNodes 的决策 → S4/S6**：把 `Loop::ConstructFromNodes`（`codegen_kernel_loop.cpp:242-368`，此函数行号未漂移）里的 `enable_cache`（`:283-285`）、`InitApiCallContext`（`:288` CV stage）、reuse/share 裸指针链（`:314-365`，`reuse_next/reuse_from/share_next/share_prev` 写于 `:333-363`；`:290-307` 为输入 reads 链）剥到独立步骤 S4（缓存决策）、S6（CV stage）、buffer 生命周期分析。reuse/share 链改为显式 `BufferReuseGraph`（边表）而非 `ApiTensor` 裸指针（`codegen_kernel_loop.h:52-55`）。**S4 注意**：`IsReduceDoubleTile`（`codegen_kernel_loop.cpp:423`）在发射期 4 处复算（`:479/497/666/751`），缓存决策独立化时这 4 个复算点须一并收敛到冻结产物。

5. **抽 S5 多核映射**：`CalcFromAxis`（`codegen_kernel.cpp:747`）/`BlockOutterAxisDefine`（`:775`、`:795`，调用 `:2096`）独立成 S5，产出 `ParallelMap` 冻结表。

6. **抽 S9 共享契约层**：把 tiling_key 编号规则（`AppendFuncCall:3182-3197`，计数器初始化 `:2331/:2365`）和 tiling_data 字段命名规则（`codegen_tiling_data.cpp:108-141`，`TILING_DATA_FIELD_DEF_T`/`common_tiling_fileds`）抽成 `TilingKeyScheme` + `TilingDataFieldNamer` 单一真相源，kernel 段（S8）与 tiling 段（S9 的 att 回调）共享。**涉及 att 外部库契约，风险较高，最后做。**（注意：mlir-af 已移除 `tiling_key` 概念，本项目在第一阶段保留 `tiling_key` 以行为等价优先，第二阶段再考虑对齐 mlir-af 的 schema 化。）

7. **职责标注 + mlir-af 对照标注**：每个新子步骤在代码/文档标注本属 mlir-af MLIR 哪层、mlir-af 是否已实现对应物（见上表"S0-S10 本属+mlir-af对照"列），为Stage 3-5 切 MLIR 时"子步骤→mlir-af pass/emitter"铺路。

**冻结的中间产物**（替代现状 mutable C++ 对象指针）：

| 现状可变状态 | 冻结为 | 产出步骤 |
|---|---|---|
| `Tiler::axis_map`（map 可变） | `AxisContext`（不可变） | S1 |
| `Tiler::actual_sizes`（mutable，发射期写） | `TilingDerivation`（不可变表） | S3 |
| `TPipe::tensors/queues/bufs`（map 可变） | `BufferContext`（不可变） | S1 |
| `TPipe` 持有 `const Tiler&` 反向引用 | 解除，各步骤显式传参 | S1 |
| `ApiCall.enable_cache/exec_condition/api_call_context`（构建期写死） | `CacheDecision`/`CVStageAssignment`（挂在 LoopTree 上） | S4/S6 |
| `ApiTensor.reuse_from/reuse_next/share_prev/share_next`（裸指针） | `BufferReuseGraph`（显式边表） | S4 |
| tiling_key 编号 + tiling_data 字段名（隐式契约） | `TilingKeyScheme`+`TilingDataFieldNamer`（单一真相源） | S9 |

**验收标准**：
- 现状全场景回归逐子步骤等价通过（每抽出一个步骤跑一次回归，绿了才抽下一个）。
- codegen 内部呈现 S0-S10 清晰子步骤序列，每步骤输入显式、输出冻结、可独立构造输入测试。
- 每个 schedule 越界职责都从"内联在 ParseGraph/ConstructFromNodes/GenerateLoop 里"变成"独立步骤 + 冻结产物"，并标注本属 mlir-af MLIR 层 + mlir-af 对照状态。
- `Kernel::ParseGraph` 三合一、`Loop::ConstructFromNodes` 决策掺杂、`Tiler.actual_sizes` mutable 三大病灶消除。
- 对外契约（FusedScheduledResult 入 / CodegenResult 出）不变，上游不动。

解耦后，Stage 3 到 Stage 5 才能逐项迁移。例如，S2 的循环结构化对应 `LoopNestBuilder`，S3 的 tiling 派生对应
`TilePlanGen`，S7 的算子发射对应 `linalg-to-ascendc`，S8 和 S10 对应 `CannTranslation`。如果跳过 Stage 0，
现有职责会原样进入 MLIR，pass 和 emitter 的边界仍然不清楚。

---

### 4.2 Stage 1：基础设施就绪

**目标**：让 MLIR/PyAsc 能在 autofuse 构建里编译通过，并把"现状输出"固化为可复现的回归基线。本 Stage **不改任何 codegen 行为**。

**子任务**：
1. **引入 MLIR 构建开关（默认 OFF）**：新增独立 CMake 开关 `ENABLE_AUTOFUSE_MLIR`，默认 `OFF`。`--no-autofuse`（`BUILD_AUTOFUSE=OFF`）含义不变，仍表示整个 Autofuse 后端不参与编译。`ENABLE_AUTOFUSE_MLIR` 与 `BUILD_AUTOFUSE` 解耦：
   ```text
   BUILD_AUTOFUSE=ON,  ENABLE_AUTOFUSE_MLIR=OFF  -> 默认 legacy Autofuse 构建，不发现/不依赖 LLVM/MLIR/PyAsc
   BUILD_AUTOFUSE=ON,  ENABLE_AUTOFUSE_MLIR=ON   -> MLIR 开发构建，发现并链接 LLVM/MLIR/PyAsc
   BUILD_AUTOFUSE=OFF                            -> 既有 --no-autofuse 行为，整个 Autofuse 不构建
   ```
   默认主线行为必须是 `BUILD_AUTOFUSE=ON` + `ENABLE_AUTOFUSE_MLIR=OFF` + `AF_MLIR_CODEGEN=off`，使其他开发者不受 LLVM 依赖和行为变更影响。
2. **LLVM/MLIR 预编译输入（仅 `ENABLE_AUTOFUSE_MLIR=ON` 时介入）**：通过 `autofuse/mlir/scripts/prepare_mlir_deps.sh` 解析 `LLVM_BUILD_DIR` 或团队预编译制品；Docker 编译环境通过 `autofuse/mlir/docker/build_llvm_image.sh` 准备。仅在 `ENABLE_AUTOFUSE_MLIR=ON` 分支内执行 `find_package(MLIR REQUIRED CONFIG)` + `include(AddMLIR/HandleLLVMOptions)`；`OFF` 时顶层 CMake 不得触发任何 MLIR/LLVM 发现。预编译产物须与 Autofuse 的 `_GLIBCXX_USE_CXX11_ABI=0`（`autofuse/CMakeLists.txt`）及编译器版本 ABI 匹配，否则 MLIR C++ API 链接会失败。
3. **PyAsc 源码集成**：将 PyAsc 作为官方 `https://gitcode.com/cann/pyasc.git` submodule 接入到 `autofuse/mlir/externals/pyasc`，并通过 Autofuse 仓内 `autofuse/mlir/patches/pyasc/` 维护 LLVM 21 适配和必要 AscendC op 补充。`autofuse/mlir/scripts/sync_pyasc_upstream.sh` 从官方固定 commit 应用 patchset；真正编译发生在 `ENABLE_AUTOFUSE_MLIR=ON` 的 Autofuse CMake 构建图中。按四步 `add_subdirectory(... EXCLUDE_FROM_ALL)` 接入，并 `include_directories` 其 include 与 bin include：
   ```text
   add_subdirectory(${PYASC_DIR}/lib/TableGen EXCLUDE_FROM_ALL)
   add_subdirectory(${PYASC_DIR}/include EXCLUDE_FROM_ALL)
   add_subdirectory(${PYASC_DIR}/lib     EXCLUDE_FROM_ALL)
   add_subdirectory(${PYASC_DIR}/bin     EXCLUDE_FROM_ALL)
   ```
   pyasc 复用主工程 MLIR_DIR，不重复编 LLVM。
4. **MLIR 工具入口接入**：Phase 1 不改 legacy codegen 行为，先新增 `af-opt` 和 `check-autofuse-mlir` 作为最小 MLIR 编译/parse/print 冒烟入口；后续真正接入 codegen 路径时，再在对应目标内链接 `MLIRAsc`、`MLIREmitAsc`、`MLIRTargetAsc`、`MLIRSCF`、`MLIRArith`、`MLIRMath`、`MLIRMemRef` 等目标。
5. **方言注册冒烟**：通过 `af-opt` / lit 构造最小 MLIR module，验证 MLIR parse/print 和工具链可用（仅 `ENABLE_AUTOFUSE_MLIR=ON` 下）。
6. **回归基线建立**：用现状 codegen 跑全场景测试用例（`tests/st/codegen`、`tests/v35`），**冻结**其 `CodegenResult` 输出为 golden。golden 入库边界见 6.1（结构化语义清单 + 小型代表性片段入库，大体积生成物放忽略目录）。基线脚本入库，CI 可复现。

**验收标准**：
- 默认主线 `BUILD_AUTOFUSE=ON, ENABLE_AUTOFUSE_MLIR=OFF` 构建通过，且**不依赖** LLVM/MLIR/PyAsc（即无 MLIR 环境也能编译，现有 codegen 既有代码不受影响）。
- `ENABLE_AUTOFUSE_MLIR=ON` 时 autofuse 带 MLIR/PyAsc 完整编译通过。
- `--no-autofuse`（`BUILD_AUTOFUSE=OFF`）行为不变。
- `af-opt` 和 `check-autofuse-mlir` 冒烟通过。
- 现状回归基线脚本可一键复现，golden 按入库边界处理。

### 4.3 Stage 2：薄桥接纵向打通（elewise + brc）

**目标**：最小端到端验证。用**最薄**的桥接把 `FusedScheduledResult` 翻成 MLIR，再用最薄的 printer 出 Ascend C，行为等价。**本 Stage 不求结构优美，只求管道通 + 回归框架可用。**

**子任务**：
1. **桥接器雏形** `FusedScheduledResult → MLIR`：
   - 一个 `ScheduleGraphBuilder`：遍历 `node_idx_to_scheduled_results`，为每个 schedule group 建 `func.func`。
   - `SchedInfo.axis` → `scf.for` 嵌套（轴顺序、上下界用 `Expression` 求 static/dynamic 值，dynamic 走 tiling param symbol）。
   - elewise/brc 节点 → 对应 `ascendc` op（先覆盖 Add/Sub/Mul/Div、Cast、Where、Load/Store、Broadcast）。
   - 轴/管/复用信息先**临时**挂在 side-table（C++ map，不落 MLIR 属性），只够 printer 还原现状输出。
2. **统一 printer 雏形**：MLIR region → Ascend C 源码串，与现状 `CodegenResult.kernel` 对齐（include/define/kernel 函数体、`FormatIndentation` 等价缩进）。对照 mlir-af `CannTranslation.cpp` 的翻译职责。
3. **tiling/tiling_data/infer_shape**：本 Stage 仍**直接复用**现状 `att` 调用与 infershape 生成（`codegen.cpp:322/346`），不迁；只把 kernel 段走 MLIR。这样收敛风险。
4. **双路开关**：codegen 加运行期路由开关 `AF_MLIR_CODEGEN=off|on|compare`（默认 `off`），MLIR 路径与 legacy 路径并存，便于 A/B 回归。构建能力与运行行为分离：构建能力开关 `ENABLE_AUTOFUSE_MLIR`（见 4.2）控制是否编译 MLIR 路径，运行路由开关 `AF_MLIR_CODEGEN` 控制实际走哪条路。`AF_MLIR_CODEGEN` 仅在 `ENABLE_AUTOFUSE_MLIR=ON` 时有意义；`ENABLE_AUTOFUSE_MLIR=OFF` 时其值非 `off` 应报错。
   ```text
   AF_MLIR_CODEGEN=off      # 仅 legacy codegen（默认）
   AF_MLIR_CODEGEN=on       # 仅 MLIR codegen
   AF_MLIR_CODEGEN=compare  # legacy + MLIR 双路 A/B 校验
   ```
5. **回归框架落地**：elewise+brc 用例同时跑 legacy 与 mlir 两路，比对 kernel 行为（数值+性能）。

**验收标准**：
- elewise+brc 场景，MLIR 路径输出 kernel 经 Ascend 编译+运行，数值与 legacy 一致、性能不劣化（在约定阈值内）。
- 回归框架能自动 A/B 比对并出报告。

**为什么先薄**：先证明"MLIR 端到端可行 + 回归框架能判等价"，再逐步把临时 side-table 下沉为正经 MLIR 属性。若一上来就追求Stage 4 的状态下沉，风险与返工都会大幅增加。

### 4.4 Stage 3：发射下沉为 MLIR op

**目标**：把Stage 2 的"薄桥接"里仍靠字符串拼接的部分，逐算子替换为真实 `ascendc` op 构造；`Loop` 树正式映射 `scf.for`；统一 printer 覆盖 elewise+brc 全部 op。

**子任务**：
1. **`ApiCall` 族重构**：elewise/brc 各 `ApiCall` 子类的 `Generate()` 改为构造 `ascendc` op（不再 `<<` 拼串）。算子族仍以类组织，但产物是 op。
2. **`Loop → scf.for`**：`Loop::Generate` 改为在 MLIR builder 里建 `scf.for` 嵌套；`LoopBody` 的 `ApiCall*` 叶子变为在循环体内构造 op。对照 mlir-af `LoopNestBuilder`。
3. **统一 printer 成型**：单一 printer 走 `scf`+`ascendc`+`emitasc` 出 Ascend C，替代现状 `FormatIndentation` 散落拼接。对照 mlir-af `CannTranslation.cpp`。
4. **逐算子回归**：每迁一个 `ApiCall` 子类，单独过 elewise+brc 回归（防止一次性返工）。
5. **side-table 保留**：轴/管/复用状态**仍**在 C++ side-table，Stage 4 才下沉。本 Stage 只动"发射形态"。

**验收标准**：
- elewise+brc 全部 op 走 MLIR op 构造 + 统一 printer，行为等价不变。
- 无遗留字符串拼接（除 printer 本身）。

### 4.5 Stage 4：状态下沉为 MLIR 属性

**目标**：把Stage 2/3 临时塞在 C++ side-table 的 `Tiler`/`TPipe`/`Tensor` 运行时状态，下沉为 `codegen.*` 属性 + `emitasc` 类型，使 MLIR region 自洽（可序列化、可重新加载还原）。**注意**：codegen 本质是纯发射器、无 IR 变换 pass（见 2.1），因此本 Stage 不是"把 codegen 变换搬成 MLIR pass"，而是"把发射期局部计算/读上游决策的依据显式化为 op 属性，让 emitter 逻辑有据可依"。

**子任务**：
1. **属性 schema 设计**：定义 `codegen.*` 属性/类型（tablegen）：
   - `codegen.axis`：`logicalAxisId`、`type`(Block/Tile Outer/Inner/Merged)、`size`(Expression 串)、`align`、`from`、`split_pair`。
   - `codegen.tiling_case`：x/y/r/n group 的轴 id。
   - `codegen.queue`/`codegen.buf`：复用 `emitasc` 的 TQue/TBuf 类型，附 reuse_chain 属性。
   - `codegen.cv_stage`：CV 融合阶段（Stage1/Stage2）。
2. **`Tiler`/`TPipe`/`Tensor` 退化为 builder**：这些类不再持有运行时状态，而是读/写 op 上的 `codegen.*` 属性。
3. **明确发射期逻辑的归属**：actual-size、cache 条件、CV 阶段和复用链要么是局部计算，要么读取上游决策。
   需要跨 Op 共享的信息写入属性，纯局部计算留在 MLIR 到 Ascend C 的 emitter 中。本阶段不为这些逻辑新增
   Codegen 专用 pass。
4. **扩 elewise+reduce**：在 MLIR 自洽基础上接入 reduce 算子族（reduce 的轴分组、reduce store）。**注意 mlir-af 缺口**：mlir-af 的 reduce `kAllLoad`(FullLoad) 未编、`kRCore` 仅末端 combine，本项目须自行补全这三模板的行为等价（对照现状 legacy）。

**验收标准**：
- elewise+brc + elewise+reduce 行为等价。
- MLIR region 自洽：无外部 side-table 也能还原全部 codegen 语义（可用"序列化 MLIR → 重新加载 → 重新出 Ascend C"验证）。

### 4.6 Stage 5：算子族横向扩展

**目标**：把剩余算子族逐个迁到 MLIR，覆盖现状全场景。各算子族相互独立，可并行。

**算子族清单**：
- **`codegen/api_call/` 目录内 ApiCall 族**：reduce、concat、gather、transpose、datacopy(Load/Store)、broadcast 其余形态、以及 `ascir/reg_func` 注册的 elewise 全部 op。
- **cube/CV（不在 api_call 目录）**：走 `codegen_kernel.cpp GenCube*`(`:2466` 起) + `codegen_tiling.cpp` + `codegen_tiling_cube_wrapper.h`(45KB) 的**独立 kernel 组装路径**，不属 ApiCall 族体系。迁移时需单独设计其 op 构造/组装，不能套用 ApiCall 子类模板。

**子任务（每族同构）**：
1. 该族 `ApiCall` 子类 → `ascendc` op 构造 + 必要 `codegen.*` 属性。
2. 该族经统一 printer 出 Ascend C。
3. 该族过现状回归（数值+性能）。

**cube/CV fusion 特别说明**：现状 codegen 的 CV fusion 有 Stage1/Stage2 划分，mlir-af 用 `CubeKind::MatmulVecFuse` 单 kernel 且仅 Phase 4a（level-1 tile）未完成 inner tile。本项目迁 cube 时**以现状 legacy 行为为等价基准**自行设计，不能照 mlir-af 的简化单 kernel（mlir-af 形态会改变执行行为）。这是本设计"对照 mlir-af 已验证形态、不直接复用"原则的关键体现。

**验收标准**：全场景回归过，覆盖 autofuse 当前支持的全部 op/场景。

### 4.7 Stage 6：替换与退役（第一阶段终点）

**目标**：codegen 完整复刻到 MLIR 后，MLIR 路径**替换掉原 legacy codegen**。legacy 代码移除，MLIR 成为唯一路径，全流程端到端跑通。这是第一阶段的终点门禁。

**前置条件**：Stage 5 全场景双路回归全绿（MLIR 路径与 legacy 行为完全等价），否则不具备退役安全前提。

**子任务**：
1. **全流程 MLIR-only 验证**：关闭双路开关，`compiler/py_module`（`pyautofuse.cpp:235` `Codegen::Generate*`）只走 MLIR 路径，跑 `optimize → codegen → Ascend C → Ascend 编译 → 运行` 全链路。
2. **legacy 代码移除**：删除 `codegen/` 下被 MLIR 完全取代的 legacy 发射路径（旧 `ApiCall` 字符串拼接 `Generate`、旧 `Loop::Generate` 拼串、`FormatIndentation` 等）；保留Stage 0 冻结的 golden 与回归脚本作为持续回归。
3. **双路开关移除**：删除 `AF_MLIR_CODEGEN` 路由开关及 `ENABLE_AUTOFUSE_MLIR` 构建开关的过渡语义（MLIR 路径成为唯一路径后不再需要开关），`Codegen::Generate` 入口直接调 MLIR 路径。
4. **遗留依赖清理**：确认 tiling/tiling_data/infer_shape 若本阶段仍复用 legacy 段，此刻一并迁移或确认其已由 MLIR 覆盖；清理 `npu_arch` 字符串分支（若Stage 6 内能顺带收敛则收敛，否则登记给第二阶段，**但不允许 legacy 发射逻辑残留**）。
5. **全量端到端回归**：全场景 MLIR-only 跑通，含 longrun 性能回归。

**验收标准**：
- `codegen/` 无 legacy 字符串拼接发射残留（grep `FormatIndentation`、旧 `Generate` 拼串模式为空）。
- 双路开关移除，MLIR 是唯一路径。
- 全场景 MLIR-only 端到端（数值 + 性能 + longrun）回归过。
- 第一阶段交付完成。

### 4.8 第二阶段（另开设计，不在本阶段交付）

npu_arch/soc_version 字符串分支 + per-op 分支收敛到 HW 描述符 + 注册表/接口。本设计仅标记为后续。

---

## 5. 关键数据结构映射表（现状 C++ ↔ MLIR ↔ mlir-af 对照）

### 5.1 AFIR 与 AscGraph 的关系（修正早期误判）

⚠️ **AFIR 与 autofuse AscGraph protobuf 不是 1:1 映射**（mlir-af `docs/AscGraph_Mapping.md:7` 明确）。AFIR 是简化抽象：DataType 用 MLIR 内置类型、MemAttr/MemQueue/MemBuf 三合一为 `AscTensorGroups`、移除 `SchedInfoDef`/`ApiInfoDef`/`tiling_key`（v2.0）。本项目因不迁 schedule、保留对外契约，**不直接采用 AFIR**，仅在 codegen 内部用 `codegen.*` 属性 + `emitasc` 类型表达 codegen 语义。AFIR 的简化思路可作为未来与 mlir-af/Ascend-MLIR 统一时的参考。

### 5.2 现状对象 → MLIR → mlir-af 对照

| 现状对象 (codegen/) | 含义 | MLIR 表达 | mlir-af 对照（已实现/未实现） |
|---|---|---|---|
| `Loop` (`codegen_kernel_loop.h:166`) | 循环嵌套树，节点=轴 | `scf.for` 嵌套；轴标识挂 `codegen.axis` | ✅ `LoopNestBuilder`（已实现） |
| `LoopBody{ApiCall*}` | 循环体叶子=算子调用 | `scf.for` body 内的 `ascendc.*` op | ✅ `GroupEmitter`（已实现） |
| `ApiCall` 子类 (`api_call/*`) | 单算子发射 | `ascendc.*` op | ✅ `linalg-to-ascendc`（已实现） |
| `Tiler::axis_map` (`codegen_kernel.h:240`) | 轴 id→轴对象 | `codegen.axis` 属性 | ✅ `AxisGrouping`/`TileParam`（已实现） |
| `Axis.Type` (Block/Tile Outer/Inner) | 轴类型 | `codegen.axis.type` enum 属性 | ✅ `TileLevel{Outer/Inner/Full}`（已实现） |
| `Tiler::BlockOutterAxisDefine` | 块轴定义(multicore) | `ascendc.get_block_idx` + `codegen.axis.type=BlockOuter` | ⚠️ `ascendc-parallelize`（简化版） |
| `TQue`/`TBuf` (`codegen_kernel.h`) | 队列/缓冲 | `emitasc` TQue/TBuf 类型 | ✅ `ascendc-buffer-placement`（已实现） |
| `TPipe` | 管道/管分配 | `emitasc` pipe 描述 + `codegen.queue` 属性 | ✅ `insert-tile-buffers`+`buffer-placement`（已实现） |
| `ApiTensor` reuse 链 (`codegen_kernel_loop.h:49`) | 缓冲复用 | `codegen.reuse_chain` 属性 | ⚠️ mlir-af 简化（共享 VECOUT 启发式） |
| `ApiCallContext.stage` (CV Stage1/2) | CV 融合阶段 | `codegen.cv_stage` 属性 | ❌ mlir-af 不做 stage（`CubeKind::MatmulVecFuse` 单 kernel） |
| reduce 三模板 kCommon/kAllLoad/kRCore | reduce 调度模板 | `codegen.reduce_template` 属性 | ⚠️ mlir-af `ReduceTemplate`（Common 完成，FullLoad 未编，RCore 仅末端） |
| reduce Mean 尾块 Muls | mean 尾块乘法补全 | emitter 逻辑 | ❌ mlir-af 未实现 |
| `exec_condition` | Call 执行条件（决策来自上游 `optimize/node_cache_marker.cpp:149`） | `scf.if` 或 `codegen.cond` 属性 | Codegen 只翻译决策，不做决策 |
| `Expression`/`SizeVar` (symbolic) | 符号形状算术 | 上游 `affine`/`arith` + symengine | ✅ mlir-af `afir-symbolize-shapes`/`symbolic-dim-cse`（已实现） |
| `SchedInfo.axis` | 节点所属循环轴(外→内) | 决定 `scf.for` 嵌套位置 | ✅ mlir-af collapsed linalg 轴（已实现） |
| tiling_key 编号 + tiling_data 字段名 | kernel/tiling 隐式契约 | `TilingKeyScheme`+`TilingDataFieldNamer` | ❌ mlir-af 已移除 tiling_key（用 `auto_fuse.tiling_infos` schema v2） |
| `npu_arch`/`soc_version` 分支 | 芯片差异 | **第二阶段**：`codegen.hw_profile` 属性 | mlir-af 使用 TargetProfile，供第二阶段参考 |

---

## 6. 迁移前准备清单

> "迁移之前该做好哪些准备" 分两块：Stage 0（内部解耦）前的准备，与Stage 1（MLIR 基础设施）前的准备。前者是纯 C++ 解耦的安全网，后者是 MLIR 引入的硬前提。

### 6.1 Stage 0（内部解耦）前的准备：安全网
- [ ] **回归基线**：盘点现状全场景用例（`tests/st/codegen`、`tests/v35/st`、`tests/ut/codegen`、e2e）；跑现状 codegen 冻结 `CodegenResult`(kernel/tiling/tiling_data/infer_shape) 为 golden；写一键复现脚本接入 CI；定义"行为等价"判据（数值容差、性能不劣化阈值、必跑用例集）。这是后续解耦检查和 MLIR codegen 可运行里程碑做整路径 A/B 的回归依据。
- [ ] **golden 入库边界**（编码红线 G11）：明确哪些内容进入 Git。优先提交结构化语义清单（tiling schema、IO/workspace/ABI 字段、关键语义段断言）和小型代表性 kernel 片段。完整生成的 kernel 源码、临时 `kernel_meta`、coverage 和大体积运行产物放入忽略目录或 CI artifacts。确需提交完整 golden 时，只保留小型且经过评审的代表用例。
- [ ] **契约冻结**：确认 `FusedScheduledResult`（`common/schedule_result.h`）与 `CodegenResult`（`codegen/codegen.h`）本 Stage **不改动**，作为边界契约写进文档。
- [ ] **盘点 codegen 内部耦合点**：按 3.2 表确认每处 schedule 越界职责的 file:line，作为解耦子任务清单的依据（已在Stage 0 子任务中列出）。
- [ ] **盘点 codegen 对上游/att 的调用点**：确认解耦不越界（如 `att::TilingLib` 调用 `codegen.cpp:331/351`、`EnrichScheduledResultAscirParams` `codegen.cpp:94`、att 回调读 FusedScheduledResult 的 tiling_key/字段名契约）。
- [ ] **解耦前置探针**：先确认 `actual_sizes` 冻结（Stage 0 子任务 2）能否在不改行为的前提下完成。这个步骤通过后，可以证明“冻结中间产物”的路线可行。
- [ ] **mlir-af 缺口盘点**：确认 3.2 表中 mlir-af 未实现项（缓存决策/CV stage/reduce Muls/FullLoad/RCore 前端）在迁移时需自行设计，不能照搬 mlir-af。

### 6.2 Stage 1（MLIR 基础设施）前的准备：硬前提
- [x] **构建开关与默认路径**：`ENABLE_AUTOFUSE_MLIR` 已实现且默认 `OFF`，与 `BUILD_AUTOFUSE` 解耦；默认构建和 `--no-autofuse` 不引入 LLVM/MLIR 依赖。
- [x] **构建基础设施**：显式 LLVM/MLIR 解析、`prepare_mlir_deps.sh`、官方 PyAsc submodule + patchset、`sync_pyasc_upstream.sh`、`af-opt` 和 `check-autofuse-mlir` 均已落地；生产 codegen 接入仍留在后续 Stage。
- [x] **预编译依赖 ABI 与来源约束**：`LLVM_BUILD_DIR` 显式解析、缺失依赖报错、`_GLIBCXX_USE_CXX11_ABI=0` 校验和环境/镜像 manifest 已实现；正常构建不会静默下载或编译 LLVM。
- [x] **PyAsc 源码集成可行性**：官方 PyAsc 以源码 submodule 方式由 Autofuse MLIR 子工程集成，并应用仓内 patchset；不使用独立 PyAsc 构建产物。
- [ ] **ascendc 方言 op 覆盖度**：核对 PyAsc `ascendc` 方言是否覆盖 Autofuse 现有全部算子族（elewise/reduce/brc/concat/gather/transpose/cube）。缺少的通用 AscendC API 映射通过 PyAsc patchset 补充 Op 定义、emitter 和 lit 测试。Autofuse 私有调度或融合语义不进入 PyAsc，使用自定义属性或已有 Op 组合表达。该结论决定 Stage 5 的范围。
- [ ] **emitasc 类型承载力**：核对 `emitasc` 的 TQue/TBuf/py_struct 是否够表达 autofuse 的管/缓冲/tiling_data。
- [ ] **tiling 段边界**：确认 tiling/tiling_data/infer_shape 复用现状不动是否真的可行（att 调用是否依赖 codegen 内部状态）。
- [ ] **退役可行性**：确认 legacy codegen 全部入口（`Codegen::Generate*`、`GenerateForInductor`、pybind `pyautofuse.cpp:235`）都能被 MLIR 路径替换；盘点 legacy 代码移除范围，确保Stage 6 退役无遗漏入口。

---

## 7. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| Stage 0 解耦破坏等价（拆 ParseGraph/剥决策/冻结 actual_sizes 引入行为漂移） | 解耦 Stage 回归红，卡在 Stage 0 | 每完成一个解耦子步骤立刻跑全场景回归；先做收益最大风险最小的 actual_sizes 冻结验证路线可行；保持对外契约与上游不变 |
| mlir-af 缺口项（缓存决策/CV stage/reduce Muls/FullLoad/RCore）误照搬 mlir-af 简化形态 | 行为漂移、cube/reduce 场景回归红 | 3.2 表明确标注 mlir-af 未实现项，迁移时以现状 legacy 行为为等价基准自行设计，不照搬 mlir-af |
| PyAsc 源码集成编译不过（依赖/版本/LLVM 不匹配） | Stage 1 卡死，全盘受阻 | 作为Stage 1 首个风险探针，通过官方 submodule + patchset 生成 patched source tree，并在 Autofuse 集成构建中验证；必要时锁定 official upstream commit |
| `ascendc` 方言 op 不覆盖某算子族 | Stage 5 该族无法直接映射 | Stage 1 探针盘点覆盖度；通用 AscendC API 缺口通过 PyAsc patchset 补，Autofuse 私有语义用自定义属性或已有 op 组合；首条纵向只选已覆盖的 elewise+brc |
| 行为等价难判定（性能波动、数值边界） | 门禁过不了，卡在 Stage 间 | Stage 0 前先定判据（容差/阈值/必跑集）；性能用多次取中位 |
| `Tiler/TPipe` 状态下沉破坏等价（Stage 4） | elewise+reduce 回归红 | Stage 4 前先有Stage 2/3 的 side-table 稳态作对照；逐属性下沉、逐次回归 |
| 双路并存导致 codegen 膨胀/维护负担 | 长期两套代码 | 双路开关是过渡手段，Stage 5 收尾时 legacy 逐步退场；每 Stage 结束清理已无用的 legacy 路径 |
| Stage 6 退役后某场景才发现 MLIR 路径有缺陷 | 退役后无法回退（legacy 已删） | 退役前 Stage 5 必须通过全场景双路回归和 longrun；保留 golden 与回归脚本；先将默认路径切到 MLIR 并观察，再删除 legacy |
| tiling/infershape 段未迁完就退役 legacy | 退役后 tiling 行为缺失 | Stage 6 验收明确核查 tiling/tiling_data/infer_shape 已由 MLIR 覆盖或确认复用路径不依赖被删 legacy |
| tiling 段复用现状但隐式依赖 codegen 内部状态 | Stage 2 后 tiling 行为漂移 | Stage 0 抽 TilingKeyScheme 共享层消除隐式契约；tiling_data/tiling 全程走现状路径直到能独立验证 |
| 上游 `FusedScheduledResult` 未来变动 | 迁移中契约漂移 | 契约冻结写入文档；与 optimize 团队约定本 Stage 不变更该结构 |

---

## 8. 测试与回归策略

### 8.1 三层回归
1. **单元**：每个 `ApiCall` 子类→`ascendc` op 的构造正确性（gtest，`tests/ut/codegen` 风格，断言 op 形态/属性）。
2. **源码级**：MLIR 路径 `CodegenResult` 与 golden 比对（允许结构差异，断言关键语义段存在）。golden 取自现状 codegen，在Stage 0 解耦前（见 6.1 回归基线）冻结入库。
3. **端到端行为**：MLIR 路径 kernel 经 Ascend 编译+运行，与 legacy 比**数值**（容差内）+**性能**（不劣化阈值内）。这是行为等价门禁的最终判据。

### 8.2 双路 A/B（过渡，Stage 6 终结）
- 运行路由开关 `AF_MLIR_CODEGEN=off|on|compare` 与构建能力开关 `ENABLE_AUTOFUSE_MLIR` 配合：`ENABLE_AUTOFUSE_MLIR=ON` 编译出 MLIR 路径，`AF_MLIR_CODEGEN=compare` 时每个用例两路都跑、自动比对。
- Stage 2-5 全程双路；**Stage 6 退役 legacy 并移除开关**，之后仅 MLIR 单路 + golden 持续回归。
- **未来 ATT/Schedule 迁移**：当前 codegen-only 阶段不引入 `AF_MLIR_COMPONENTS` 或 `AF_PIPELINE_BACKENDS` 全流水线矩阵（过早增加复杂度）。后续 ATT/Schedule 迁移时，可新增聚焦开关 `AF_MLIR_ATT=off|on|compare` / `AF_MLIR_SCHEDULE=off|on|compare`，或最终统一为流水线路由表如 `AF_PIPELINE_BACKENDS=codegen=mlir,att=mlir,schedule=legacy`。

### 8.3 Stage 门禁
- 每 Stage 结束必须：全量必跑用例集端到端行为等价 + 该 Stage 新增单元测试过 + 源码级 golden 比对过（允许登记的差异除外）。
- 门禁不过不进下一 Stage；门禁通过即该 Stage 可独立交付/回滚点。
- **Stage 0 门禁**（解耦）：每完成一个解耦子步骤跑全场景回归等价通过；解耦后 codegen 内部为 S0-S10 清晰子步骤序列，每步骤可独立测试。
- **Stage 6 终点门禁**（额外）：MLIR-only 全流程端到端跑通 + legacy 残留为空 + 双路开关移除。此门禁过 = 第一阶段交付完成。

### 8.4 用例集
- **首条纵向**：elewise+brc 最小集（Stage 2）。
- **扩 reduce**：elewise+reduce（Stage 4）。
- **全场景**：`tests/st/codegen` + `tests/v35/st` + e2e（Stage 5）。
- 长期跑（longrun）用例纳入回归以防性能回归。

---

## 9. 特性交叉影响与编码红线检查

> 按 AGENTS.md 要求，本节逐项核对 `docs/guidelines/cross_feature_check.md` 与 `docs/guidelines/编码红线.md`。

### 9.1 特性交叉影响（cross_feature_check.md）

| 场景 | 适用性 | 分析说明 |
|------|--------|----------|
| SuperKernel Python 接口 | 不适用 | 本设计仅改 autofuse `codegen/`，不触及 `super_kernel/*.py`、wheel 内容。 |
| SuperKernel C++/AOT 接口 | 不适用 | 不触及 `super_kernel/kernel/`、`libascendsk.so`、AOT、ABI/API。 |
| Autofuse 图优化 | 不适用 | `optimize/`/`ascir/`/`graph_metadef/` 本 Stage 不动；不新增图改写 pass，无 pass 时序变化。 |
| Autofuse Codegen/Backend | 适用 | 本设计核心范围。codegen 内部重构 + MLIR 化，新增生成路径，tiling/kernel 执行验证由双路 A/B + 端到端回归覆盖。 |
| AscendC API / Runtime 交互 | 部分适用 | codegen 本身是发射期产物、不新增运行时 `rt*`/`aclrt*` 调用。Phase 1 不引入 Runtime 预研、runtime 迁移或 artifact runner，只复用现有 `ascir_tool`/测试 stub 作为验证入口；后续 MLIR codegen 会生成 AscendC op / Ascend C 调用，须覆盖生成 AscendC API 的覆盖范围与限制、TPipe/TQue/TBuf 与 tiling data 生命周期、`ascir_tool` 适配器与真实 NPU launch 路径。 |
| Python/C++ 混合绑定 | 部分适用 | `compiler/py_module`（`pyautofuse.cpp:235`）`Codegen::Generate*` 入口在Stage 6 切 MLIR-only，须验证 C++ 绑定层与 Python 调用方式不变（行为兼容，非 ABI 改动）。 |
| 构建与打包 | 适用 | Stage 1 引入预编译 LLVM/MLIR + PyAsc 源码集成，影响 CMake/`build.sh`/第三方依赖/run 包内容。须确保 `build.sh --pkg -j 8`、`--no-autofuse`、增量/离线构建仍可用（编码红线第 10 条）。 |
| 测试与覆盖率 | 适用 | Stage 0 回归基线、Stage 2-5 双路 A/B、Stage 6 MLIR-only 端到端均需新增 UT/ST/E2E 覆盖。C++ 用 gtest/mockcpp。 |
| 性能与日志 | 部分适用 | 编译时长：Stage 1 引入 MLIR/PyAsc 增加首次编译时长（单次，非运行期）。执行性能：codegen 仍一次性发射，kernel 执行性能由行为等价门禁保障不劣化。不新增运行期海量日志。 |
| 兼容性 | 适用 | 对外契约（`FusedScheduledResult` 入 / `CodegenResult` 出）不变；run 包安装路径、脚本参数不变。legacy 退役在Stage 6 全绿后分两步进行。 |

### 9.2 编码红线核对（编码红线.md）

| 红线 | 涉及 | 说明 |
|---|---|---|
| 1 禁硬编码敏感信息 | 不涉及 | 迁移不新增敏感信息。 |
| 2 外部输入作索引/长度前校验 | 不涉及 | codegen 发射期不直接消费未校验外部输入为索引；轴/形状来自已校验的 FusedScheduledResult。 |
| 3 整数运算防溢出/除0 | 留意 | tiling 派生、actual_size/loop_size 计算须保留现状的边界检查（迁移时行为等价，不削弱既有校验）。 |
| 4 资源释放覆盖异常分支 | 适用 | Stage 6 legacy 退役、MLIRContext/Builder 生命周期须显式管理；无跨 SO 全局析构依赖（红线第 12 条）。 |
| 5 内存申请前判大小、申请后校验 | 不涉及 | 发射期以字符串/op 构造为主，无大块运行时内存申请。 |
| 6 禁依赖未指定求值顺序/UB | 不涉及 | 迁移保持现状确定性。 |
| 7 禁无理由扩大修改范围 | 适用 | 仅改 codegen 内部，不顺手重构 optimize/ascir/att；Stage 0 解耦严格按 S0-S10 子步骤。 |
| G1 对外接口 API/ABI 兼容 | 适用 | `FusedScheduledResult`/`CodegenResult`/pybind 入口本 Stage 不改 ABI/API；Stage 6 切 MLIR-only 保持行为兼容。 |
| G2 禁硬编码芯片/平台/框架类型 | 留意 | codegen 现有 `npu_arch`/`soc_version` 字符串分支本阶段**原样保留**（行为等价优先），收敛留第二阶段。迁移中不新增此类硬编码。 |
| G3 图改写保持数据/控制边等价 | 不适用 | codegen 无图改写 pass（纯发射器）。 |
| G4 有时序依赖的 pass 须说明 | 不适用 | 不新增图改写 pass；MLIR 内部 pass（若用 mlir-af pipeline 概念）时序在设计阶段说明。 |
| G5 图改写结果确定 | 不适用 | codegen 无图改写。 |
| G6 高频路径禁默认海量日志 | 不适用 | codegen 非运行期高频路径。 |
| G7 rt*/aclrt*/AscendC 生命周期 | 部分适用 | codegen 不直接调运行时接口；Phase 1 仅复用现有 `ascir_tool` launch 路径与测试 stub，不新增 runtime/artifact runner。真实 NPU 执行的资源生命周期由既有链路承担，MLIR codegen 接入后再按生成 AscendC 调用补充验证。 |
| G8 禁非开放 runtime 接口 | 不适用 | Phase 1 不新增 runtime 接口调用；只复用现有 `ascir_tool`/测试 stub 路径。 |
| G9 Python/C++ 绑定引用计数/异常 | 适用 | Stage 6 切 MLIR-only 时确认 pybind 层无错误吞掉、引用计数正确。 |
| G10 构建脚本不破坏增量/离线构建 | 适用 | Stage 1 MLIR/PyAsc 集成须保证首次/增量/离线/`--no-autofuse`/指定输出路径均可用。 |
| G11 测试产物不入版本库 | 适用 | golden/临时 kernel_meta/coverage 须在忽略目录。 |
| G12 全局/静态析构不跨 SO | 适用 | MLIR 相关全局对象须显式 Finalize/Cleanup，不在析构期跨 SO 释放。 |

---

## 附录 A：关键文件索引

### A.1 autofuse（迁移目标，本仓）
- 主文档：`autofuse/mlir/docs/design/2026-06-25-codegen-mlir-migration-design.md`
- 契约入：`common/schedule_result.h`；契约出：`codegen/codegen.h`
- 编排：`codegen/codegen.cpp`（`Generate`@377/@395、`GenerateTilingData`@406、`GenerateForInductor`@415、`GenerateTiling`@430/@442、`GenerateInferShape`@451、`GenerateForPgo`@459、`GenerateKernel`@462；att 经 `tiling_lib_.Generate`@434 封装，裸 `att::` 调用在 `codegen_tiling.cpp:209-243`）
- kernel/loop 发射：`codegen/codegen_kernel.h`/`.cpp`、`codegen/codegen_kernel_loop.h`/`.cpp`
- 算子族：`codegen/api_call/{elewise,reduce,broadcast,concat,gather,transpose,datacopy,utils}/`
- 测试：`tests/{ut,st}/codegen/`、`tests/v35/{ut,st}/`
- IR 核心：`inc/graph_metadef/graph/ascendc_ir/ascendc_ir_core/ascendc_ir_def.h`

### A.2 mlir-af（参考基准，只读）`/Users/neo/Code/Ascend-MLIR-gsr`
- pass 注册：`include/Conversion/Passes.td`；pipeline 编排：`lib/Conversion/AutoFuse/Pipeline.cpp`
- AutoFuse 核心：`lib/Conversion/AutoFuse/`(GroupAnalysis/GroupOutline/TileFuse 含 TilePlanGen/TilePlanBuild/SliceComputer/LoopNestBuilder/CubeEmitter/InsertTileBuffers)、`include/Conversion/AutoFuse/{GroupInfo,TileInfo,TilePlan}.h`
- lowering：`lib/Conversion/LinalgToAscendC/`、`lib/Conversion/AscendC*/`(BufferPlacement/Parallelize/PrepareForEmit/FlattenGMPtr/RCoreCombine/DecomposeMultiAxisBroadcast)
- 翻译：`lib/Target/CannKernel/CannTranslation.cpp`
- 运行时：`lib/Runtime/`
- AFIR：`include/Dialect/AFIR/`；TmTensor：`include/Dialect/TmTensor/`
- 概念映射参考：`docs/superpowers/plans/2026-05-11-port-af-scheduler-to-vector-plan.zh.md`、`docs/auto-fuse/00-architecture.md`、`docs/AscGraph_Mapping.md`
- ⚠️ 勿参考：`docs/Ascend-MLIR-Detailed-Implementation-V2-*.zh.md`（未实现的 dev-nyh 五段式蓝图，与 mlir-af 代码脱节）
- 子模块：`.gitmodules`（`externals/pyasc` → 官方 `https://gitcode.com/cann/pyasc.git`；Autofuse LLVM 21 适配和 op 补充通过仓内 patchset 维护；`externals/stablehlo`）

## 附录 B：autofuse 概念 → mlir-af MLIR 映射（吸收自 mlir-af 文档 + 代码核实）

| autofuse 概念 | mlir-af MLIR 对应 | 证据 |
|---|---|---|
| fused graph grouping | `auto-fuse-group-analysis`(group_id/topo_index) + `auto-fuse-group-outline` | `Pipeline.cpp:33,41`；`00-architecture.md:41-45` |
| `AxisGroup`(x/y/r/n_group) | `AxisGrouping`(`AxisClass[]` over collapsed 轴) | `include/Conversion/AutoFuse/GroupInfo.h` |
| `DoAutoSchedule` | `TilePlanGen`(决策) + `LoopNestBuilder`(物化) | `lib/Conversion/AutoFuse/TileFuse/TilePlanGen.cpp`、`LoopNestBuilder.cpp` |
| `TilingCase`(ub/block tiling id, reduce_is_block) | `TilePlan`(Outer/Inner/Full + blockFusedAxes + ubTilingAxis(x/y/r) + reduceTemplate) | `include/Conversion/AutoFuse/TilePlan.h:105-130` |
| reduce 三模板 kCommon/kAllLoad/kRCore | `TilePlan::ReduceTemplate{None,Common,FullLoad,RCore}` | `TilePlan.h:106`（FullLoad 未编、RCore 仅末端） |
| ATT model info | `TileParam` flag + `TileConstraint` + `auto_fuse.tiling_infos` module attribute | `TilePlan.h:81-103`；`TilePlanBuild.cpp:56` |
| ATT 符号化 tail_size/loop_num/block_dim | tunable func args + `TilePlan::blockDimExprs` | `TilePlan.h:111` |
| `score_func` | `costEstimate(TilePlan)`（占位规则） | plan §3.6/§8 |
| buffer allocate | `auto-fuse-insert-tile-buffers` + `ascendc-buffer-placement` | `Pipeline.cpp:123-124` |
| api_call emission | `linalg-to-ascendc` | `Pipeline.cpp:125`；`lib/Conversion/LinalgToAscendC/` |
| tiling data ABI | `ascendc-pack-tiling-data` + `auto_fuse.tiling_infos` schema v2 | `Pipeline.cpp:140`；`TilePlan.h:176 kSchemaVersion=2` |
| PGO/autotune | `tools/autotuner`（消费 `TunableTile` candidates） | `docs/auto-fuse/06-autotuner-design.md` |
| GlobalTensorInit / TPipe alloc / 顶层组装 | `CannTranslation.cpp`（`emitSupportedMixAicGlobalTensorSetup`/`<< "TPipe pipe;"`/`extern "C" __global__`） | `CannTranslation.cpp:1697,2006,1999` |
| multi-core dispatch | `ascendc-parallelize`（`get_block_idx` + `block_idx / grid_n` + `block_idx % grid_n`） | `AscendCParallelizePass.cpp`；`00-architecture.md:282-284` |
| CV fusion | `CubeKind::{None,MatmulOnly,MatmulVecFuse}` + `auto-fuse-restore-matmul` + `CubeEmitter`（⚠️ 仅 Phase 4a，无 stage 划分） | `TilePlan.h:30-38`；`Pipeline.cpp:89`；`CubeEmitter.cpp:123` |

## 附录 C：mlir-af 旧迁移文档吸收说明

mlir-af 仓 `docs/superpowers/specs/2026-06-12-autofuse-mlir-migration-overall-design.md`（status: design - pending review，与 mlir-af 实际独立重写脱节）描述"渐进迁移 autofuse"。其有价值内容已吸收进本主文档：
- **模块拆分表**（compiler/ascir/optimize/att/codegen/ascendc 顶层 + schedule/att/codegen 细分）→ 本主文档 1.1 编译链 + Stage 边界已体现 codegen 细分（S0-S10）；att/optimize 拆分属后续 Phase，本设计不迁故未展开，登记给后续。
- **运行承载厘清**（CANN runtime/ACL 为外部依赖、runtime test stub、单 kernel launch 工具、host/tiling/profile 生成物、外部框架加载路径）→ 本主文档 1.5 范围边界已明确 codegen 不含通用 runtime。
- **从后向前迁移顺序**（codegen → ATT → schedule → ASCIR/Python）→ 本主文档第一阶段聚焦 codegen，符合该顺序的"后向前"第一站。
- **四层测试门禁**（interface golden / artifact diff / E2E CPU 仿真 / real NPU）→ 本主文档第 8 节三层回归 + 端到端行为已覆盖其精神（CPU 仿真/real NPU 在本项目由 Ascend 编译+运行端到端承载）。
- **维测工具**（dump-autofuse-pipeline / dump-mlir-pipeline / diff-* / case-minimizer）→ 登记为Stage 2 回归框架的可选增强，本主文档 8.2 双路 A/B 已覆盖 diff 内核。

经用户确认，mlir-af 该份旧文档**暂留不动**（不删除、不归档），后续视情况再定。
