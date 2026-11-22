# Autofuse MLIR 架构与文档索引

本目录只保留三类文档：长期设计、阶段开发记录和当前使用指南。新开发者先阅读本索引，再按任务选择
一个分类，不需要遍历全部历史文档。

## 整体架构

```text
mlir/
├── api/ascendc/        AFIR 向 AscendC API lowering 依赖的 dialect/转换/序列化，fork 自 PyAsc 并独立演进
├── artifact/           交付物装配：多候选聚合与 device/host 两侧文本发射，mlir/ 里唯一不产出 IR 的一层
├── bridge/             legacy ScheduledResult/中立 DTO 到 AFIR 的桥接
├── cmake/              LLVM/MLIR 依赖解析
├── conversion/         AFIR 到目标 dialect 的正式跨方言 lowering
├── demo/               迁移演示和待正式 pipeline 替代的临时实现
├── dialect/AFIR/       AFIR Support、IR 和方言内 Transforms
├── docker/             dev+LLVM 镜像构建与校验脚本
├── docs/               design、development、guides 三类文档及统一索引（本文档）
├── scripts/            环境进入、依赖准备和轻量检查脚本
├── test/               Autofuse MLIR GTest 与 LIT
└── tools/              af-opt、AFIR 生成器和迁移辅助工具
```

数据流是单向分层：`bridge/` 把上游 ScheduledResult/DTO 构造成 `dialect/AFIR`（Support → IR →
Transforms 内部分层）；`conversion/AFIRToAscendC` 把 AFIR 收敛 lowering 到 `api/ascendc/` 提供的
AscendC/EmitAsc dialect；`artifact/` 在它下游，把已下降的多个候选模块聚合成一个 entry
module 并发射两侧的交付物（`KernelSource` 出 kernel 源码、TilingData header 与
host wrapper，`TilingSource` 出 Inductor host tiling ABI）；`demo/` 是待正式
pipeline 替代的临时对照路径，不在正式依赖方向上。完整分层
依据、模块边界和验证结果见 [Autofuse MLIR 架构重构设计](design/2026-08-06-autofuse-mlir-architecture-refactoring.md)。

## 目录

| 目录 | 内容 | 使用原则 |
|---|---|---|
| `design/` | 当前/目标架构、组件设计和长期迁移方案 | 回答“系统为什么这样设计”；方案变化时同步更新 |
| `development/` | 某波需求、Bugfix、重构和实施总结 | 回答“这一波具体改了什么”；统一使用日期前缀 |
| `guides/` | 开发、构建、协作手册及查询资料 | 回答“现在应该怎样操作或查询”；不记录阶段状态 |

## 推荐阅读顺序

1. [Autofuse MLIR 架构与构建重构设计](design/2026-08-06-autofuse-mlir-architecture-refactoring.md)：了解当前目录、target 和依赖边界。
2. [LLVM/MLIR 依赖策略](guides/llvm_dependency_policy.md)：准备开发环境。
3. 根据具体工作选择下面的专题设计或开发记录。

## 当前设计

| 文档 | 状态 | 当前说明 |
|---|---|---|
| [搬运的二维 tile 寻址设计](design/2026-09-20-two-dim-transfer-addressing-design.md) | 已完成（S1～S5，验收项全勾） | reduce 切 R 轴后 tile 是多维，而 `TransferPlan` 的四个标量轴字段一条都接不住：`.front()` 无 size 检查（**静默搬错**）、`splitPairOtherId` 反查遇合并块轴即断、偏移不乘 stride。改为无轴 id 的 `TransferDim` 数组 + **`axis \ vectorized_axis` 集合差**偏移式，有效视图四条归约规则照搬 legacy，> 2 根套 `scf.for`。2026-09-21 ARAR 探针实测四个 tiling case 为 4/3/3/2 根；S3 已由固定 plan 0 / template 3 的真实 ARAR CPU E2E 完成数值验收；S4 的外层循环与多轴 AR reduce 合并下降已落地并完成验收；S5 的债务清理已完成（轴绑定清单 + MemoryAllocate 结束校验、`proveExpressionsEqual` 改 `ParseRelaxed`、两项死代码清理、`TransferPlanSide` 补覆盖、map/descriptor 查证后归 L4），产物 44/44 逐字节不变；S5 两条验收项**已于 2026-10-02 全部完成**：**六组合 E2E 矩阵已于 2026-10-02 在 910B 实机全部通过**（RCommon/AllLoad 四格 10/10，RCore 两格各 3/3；注意「AllLoad × 二维 RA」这一格不存在，由 ≥3 维 ARA 顶上）。RCore 原先记的两条阻塞（`Workspace` 节点、tiling_key 命名分歧）都已被先前提交拆除。本组 RCore 探针采用小 A、大 R，跨核是否触发仍需检查实际 R 分片及 phase 2 合并份数；单核 CPU E2E 不能代替跨核验证；矩阵判据同时改成随归约长度走（`margin × sqrt(n) × eps`），写死 1e-5 会在长归约相消的格子上假报缺陷。lit 一侧也补齐：四条既有 reduce lit 各加一条 `Ascend950DT_950x` 的 RUN 行，RCore 两格新增 `conversion/afir-to-ascendc-reduce-rcore.mlir`（此前根本没有 lit，一份真实 dump 同时钉住两个 stage）；**五条在两个目标上的输出逐字节相同**，说明 reduce 的下降没有目标相关分支（AllLoad 已于 2026-10-01 实测可编且数值正确，见 [AllLoad 模板设计](design/2026-10-01-afir-reduce-allload-template-design.md)）。 |
| [AFIR reduce AllLoad 模板设计](design/2026-10-01-afir-reduce-allload-template-design.md) | 已实施（待评审） | **推翻了「AllLoad 模板未编」这条在册结论**：2026-10-01 探针实测 AllLoad 候选不但能编，钉住该 plan 的五种形状（`[16,32]`、`[16,37]` R 尾块、`[17,32]` A 尾块、`[8,1024]` 大 R、`[4,32,8]` ARA）CPU E2E 数值全对。三条适用域结论随之定下：二维 AR 有 AllLoad；**二维 RA 根本不产生** AllLoad 候选（`IsLegacyFullLoadReduce` 要 `in[0]==out[0]`、`IsTailReduceFullLoad` 要 ≥3 轴且归约末轴），所以六组合矩阵里「AllLoad × RA」只能由三维 ARA 覆盖——而 ARA 的 `vectorized_axis` 是 `[R, A, A]`，恰好第一次走到 RA `srcShape` 公式里此前无语料的 **A 组连乘**项。`is_reduce_mem_reuse` 的作用点在调度器的 `buf_que_allocator`，MLIR 侧无读点且不应有。本方案因此不改一行下降代码与调度器，内容是钉用例（5 条 E2E + 2 条 lit）、补三维语料、改写三处过期结论。 |
| [钉 plan / 钉 template 的总闸](design/2026-10-03-mlir-plan-pinning-switch-design.md) | 已移除 | 历史方案。MLIR 专用 pinning 和 `option_error` 已删除；Reduce 回归复用 ATT 的 `AUTOFUSE_DFX_FLAGS`，DFX 不做 MLIR 候选裁剪；正常编排仍尝试静态选优，无法定选时保留全部候选，不做失败候选剔除。ATT 使用原始 schedule/case 编号；用法见 [NPU 测试说明](../test/e2e/npu/README.md#reduce-模板选择)。 |
| [多维 tile 上 elementwise 的一维 calCount 设计](design/2026-09-22-elementwise-multidim-calcount-design.md) | 已完成 | S4 之后 `TransferPlan::dims` 可以是多项，但 elementwise 把计划长度直接当向量长度用，只盖得住最内一段，于是被一道 `extents.size() <= 1` 的临时门禁挡着——legacy 已支持的 elementwise-reduce 融合因此在 MLIR 后端被拒。本方案复现 legacy `Tiler::TensorActualSize`，把多维局部视图折算成一个一维 `calCount`（`1 + Σ (actual-1) * nominalStride`），继续调一次一维 AscendC API，不新增二维 API 也不加 compute 内层循环。附带把 reduce 的取用形态门禁由「必须是队列」放宽为「必须在 UB」，否则融合图的 `afir.sum` 下不来。 |
| [AFIR ReduceSum RA 下降设计](design/2026-09-21-afir-reduce-sum-ra-lowering-design.md) | 已实施并验证 | RA 物理外/内映射、resultSpan 跨步槽位合并已落地；AR/RA RCore 两阶段已完成 NPU 验证。Workspace 与 tiling_key 命名阻塞已解除。跨核覆盖以实际 R 分片及 phase 2 合并份数判断，不仅看输入 A 大小；旧 S4 未完成条目是历史记录。 |
| [AFIR 公开 DSL 架构重构设计](design/2026-09-09-afir-public-dsl-architecture-design.md) | 实施中 | 方案级重构：四层命名空间（afir / transform.afir / afplan / adapter）、值属性归位 Type、资源 SSA 化、Schedule 与 ATT 迁移定位。替代 2026-09-06 AFIR Kernel DSL 能力设计。本期范围内已全部落地；剩余项（退出 `TQue` / Op 补齐 / Schedule 迁移 / ATT 迁移 / 性能基线）见该文 5.7 |
| [AFIR Op 补齐：elementwise 切片开发方案](design/2026-09-17-afir-op-completion-elementwise-slice-design.md) | 待评审 | 5.7 剩余项 2「Op 补齐」的族切片与首切片方案。含已查证的链路事实（两侧覆盖面、平台差异 D0～D3 四档、tmp buffer 数据链路）、E0 五步地基与 E1～E8 波次。E0-1 已完成，结论见 [op lowering 对照表](guides/ascir_op_lowering_matrix.md)。按开发方案组织，未套设计文档模板 |
| [AFIR 分层重构设计](design/2026-09-14-afir-layering-refactor.md) | 六期已落地 | `conversion/AFIRToAscendC/` 的职责拆分与文件归属、`dialect/AFIR/` 不依赖后端方言的门禁、以及 `addPlanCandidateLoweringPipeline` 的相位重划。分六期，判据是「产物是不是 AFIR」 |
| [Autofuse MLIR 架构重构设计](design/2026-08-06-autofuse-mlir-architecture-refactoring.md) | 已完成 | AFIR 分层、conversion target 收敛、demo 隔离和文档归档均已落地 |
| [Kernel Entry 生成设计](design/2026-08-08-kernel-entry-generation-design.md) | 已实现 | 生成单 SR main，以及多 SR aggregate main 的 graph/plan 两级分派 |
| [TilingData Schema 与 Header 生成设计](design/2026-08-13-tiling-data-schema-and-header-generation-design.md) | 已实现 | 生成单 SR/aggregate schema，并递归输出 C++ TilingData header |
| [AFIR/AscendC Op Capability 架构设计](design/2026-08-05-afir-op-capability-architecture-design.md) | 架构基线 | Capability 基础设施 D0～D2 和 `afir.add` 纵向样例已实现；canonical AFIR 集成属于本次重构 |
| [AFIR → AscendC Pass Pipeline 设计](design/2026-08-05-afir-to-ascendc-pass-pipeline-design.md) | 已接入生产 | 当前四相位为 ABI/tiling 参数准备、循环与资源及同步物化、实现选择与下降、host 产物生成。含 Reduce scratch 逻辑 scope 与通用轴绑定存活校验；不改变 scratch 物理分配。 |
| [Autofuse Codegen MLIR 化迁移设计](design/2026-06-25-codegen-mlir-migration-design.md) | 长期方案，部分已落地 | 生产 MLIR Codegen 路由和 AFIR→AscendC 已接入；ATT/Schedule 完整迁移仍属长期目标。正文早期状态须结合后续专题阅读。 |
| [ScheduledResult → MLIR（Codegen/ATT/Schedule）迁移设计](design/2026-07-27-schedule-att-mlir-migration-design.md) | 长期方案，部分已落地 | ScheduledResult→AFIR 已用于生产编排；ATT/Schedule 仍由 legacy 提供，不能将这两项未迁移误读为 Codegen 未接入。 |
| [Inductor 生产入口接入 MLIR Codegen 与 bisheng 编译执行链路设计](design/2026-08-26-inductor-mlir-codegen-launch-pipeline-design.md) | 已实现 | MLIR 编排、AscendC 发射、bisheng 编译产物与 `AutofuseLaunch` 链路已落地；完整 NPU/E2E 验收见文档第 10 节 |
| [AFIR 复用 SymEngine 的符号表达式设计](design/2026-07-31-mlir-symengine-integration-design.md) | 草稿 | AFIR 符号基础能力已实现，完整 ShapeEnv/Host Tiling 尚未实现 |

## 开发记录

| 文档 | 状态 | 当前说明 |
|---|---|---|
| [Phase 1 MLIR 环境准备实施总结](development/2026-06-27-phase1-environment-preparation.md) | 已完成 | 当前保留能力和已退役的阶段性草案均已注明 |
| [AFIR 代码质量改进](development/2026-08-05-afir-code-quality-improvements.md) | 已完成 | enum、shape、function role、ABI、bridge registry 和 demo 归属改进已落地 |
| [AFIR 方言重构开发实施方案](development/2026-08-13-afir-dialect-refactoring-implementation-design.md) | 待实施 | 直接定义目标 IR、属性、Interface、bridge 映射、pass 和测试实现 |
| [AFIR Op 补齐（elementwise 切片）进度与剩余任务](development/2026-09-17-afir-op-completion-elementwise-progress.md) | 进行中 | E0～E6 各波次的实施现场：每个 op 为什么那么写、哪条卡点被推翻、改动面在哪几处。**待办清单已移出**，见下一行；这份只作详情来源 |
| [AFIR elementwise 切片：遗留问题清单](development/2026-09-22-afir-elementwise-remaining-issues.md) | 进行中 | 比较族接通后全量评审重排的 **唯一工作清单**（L-1～L-20，分六个阶段）。取代进度文档里的「遗留问题」表。含各条的证据、失败形态与完成判据；阶段 F 是可维护性（fork 隔离、dtype 校验分层、E2E 样板、注释约定）。2026-10-20 补充 Broadcast + Reduce 的 BR-1～BR-3 覆盖缺口及 NPU 实际路线证据 |
| [Inductor MLIR Codegen 全链路流程图与阶段说明](development/2026-08-31-inductor-mlir-codegen-pipeline-overview.md) | 现状说明 | 按 `427b8751` 落地后的实际代码画出 `RunMlirCodegen` 全链路 Mermaid 图，逐阶段列出输入/输出/职责，供快速导览 |
| [910B 被误判为不支持 DataCopyPadExtend](development/2026-09-20-datacopy-pad-extend-wrongly-disabled-on-910b.md) | 已关闭 | V2201/V3510 均启用 `DataCopyPadExtend`；910B API UT 与 `Ascend910_9362` 实机覆盖双向非零 stride、非对齐尾部有效数据。wrapper 保持上游原样；lowering matrix 保持为源码生成快照 |

## 开发指南

- [MLIR Agent 协作与 Git 提交指南](guides/agent_collaboration_guide.md)
- [LLVM/MLIR 依赖策略](guides/llvm_dependency_policy.md)
- [CLion Docker Toolchain 配置指南](guides/clion_docker_toolchain.md)
- [PyAsc AscendC 算子覆盖矩阵](guides/pyasc_ascendc_op_coverage.md)
- [PyAsc AscendC Op Inventory](guides/pyasc_ascendc_op_inventory.csv)
- [AscIR Op 两平台 lowering 对照表](guides/ascir_op_lowering_matrix.md)（与 [.csv](guides/ascir_op_lowering_matrix.csv)）：
  legacy v1(mem-based) 与 v2(reg-based) 两套 codegen impl 的 API 名 / 扩展头 / tmp / 标量约束逐 op 对照，
  由 `tools/af-mlir-tools/scan_ascir_op_lowering_matrix.py` 生成，改动 legacy impl 后重跑

## 归档规则

- `design/` 和 `development/` 文档使用 `YYYY-MM-DD-<topic>.md`；设计文档遵循仓库设计文档模板。
- 迁移文档按内容归类：长期目标和组件边界放 `design/`，具体波次计划与实施结果放 `development/`。
- 开发阶段的详细任务清单在阶段关闭后应收敛为实施总结，不长期保留复制的代码和命令。
- Op inventory 等查询资料与其生成说明相邻放在 `guides/`，数据与生成命令保持一致。
- 自动生成的 HTML、PNG、依赖目录和临时报告不进入版本库。
- 移动文档时必须同步更新仓库内引用；不要保留两份内容相同、路径不同的副本。
