# `afir.transpose` 端到端落地与泛化设计（transpose 类 op）

> **文档说明**：本文档描述 transpose 类 op（`afir.transpose` / `afir.pad`）在 MLIR codegen
> 路径上的端到端设计与后续泛化路线，覆盖：算子本体（方言定义、bridge 接入、发射器）、
> 跨步搬运的整图使能、调度/ATT 侧配套、测试与验收口径。旧基线曾通过回归；当前补丁的验证以 review 文档最新记录为准，
> 不能沿用旧数字作为本轮通过证据；泛化路线 P1-P4 已补充测试，P5 尚待当前版本真机验证。文中行号为写作时快照，以符号名
> 检索为准。
>
> **2026-11-02 验收范围更正**：本轮仅迁移 legacy Inductor，TF 不在交付范围。
> 已迁移的 7 个独立行为及 1 个 native 集成测试通过；完整 UT 14/14、lit 284/284，
> 910B3 上转置→Add 图的两形状、legacy/MLIR 两路线共 4/4 数值通过。
> 较早 P1–P5 泛化及 optimizer 借用场景不作为本次 Inductor 迁移欠项，详见
> [Inductor 迁移验收](2026-10-29-afir-transpose-generalization-test-design.md#inductor-用例迁移验收)。
>
> **2026-11-03 reg 仿真与迁移复核**：现有 CPU 7 组图已按同形状配齐 mem/reg，共 14 格逐元素数值通过。
> reg 紧凑搬运使用既有 `datacopy_reg_base.h` 的 Compact 包装，连续后缀的 stride/合轴随尾块实际尺寸变化；
> 不改 solver，不强制 tile 对齐。CANN 9.2 CPU 对 Compact 的容量检查修正在构建目录内进行，保留真实跨度检查，
> 不改 SDK 安装目录。NPU 新增两项 `dynamic=True` 清单，legacy/MLIR × 两形状，共 8 次数值验证通过。
> 独立 Python ST、native Inductor 的阶段契约/产物确定性/非法调度检查均已补充。报告中调用 `Generate` 的
> TF 固定源码、PGO、v35 IndirectLoad E2E 不计为 native Inductor 必交迁移；不宣称 MLIR 已有 PGO/IndirectLoad 能力。
>
> 上游设计：[AFIR Op 补齐 elementwise 切片设计](../design/2026-09-17-afir-op-completion-elementwise-slice-design.md)（族划分第 7 类）、
> [RemovePad 接通设计](2026-09-30-afir-remove-pad-design.md)（同族第二个落地者）、
> [向量化轴合轴设计](2026-09-28-afir-vectorized-axis-collapse-design.md)。
>
> 本文档按 `docs/guidelines/design_document_template.md` 的章节写。

# 简介

## 目的

让 transpose 类 op 在 MLIR codegen 路径上**端到端落地，并具备泛化到整个 transpose 融合族
的能力**。分三个阶段（均已交付，泛化路线除外）：

1. **转置算子本体**：`afir.transpose` 的方言定义、bridge 接入、两个下降实现
   （mem 档 `ConfusionTranspose` / reg 档 `TransposeExtend`）、配套 lit/UT。
2. **整图使能**：算子通了，但整图 E2E 跑不通——转置图把两类从未出现过的
   数据形态推到了 load/store/Pad 面前：跨步（多段）搬运、`Pad` 节点。本设计补齐这两块
   （照 legacy 已有实现搬，不发明新做法），外加一个调试中发现的崩溃 bug 修复。
3. **验证与泛化**：CPU E2E 双平台（a2/a3 对应的 910B mem 档 + a5 对应的 950
   reg 档仿真，`PLATFORM reg_base` 切换）→ a2/a3 上板实测（和 legacy 比结果、比速度；
   a5 无实机，止步 CPU 仿真，见「不做」①）→ 泛化到 transpose 融合族的其它图
   （elementwise 链融合、多 transpose、broadcast 交互）。

为什么第二阶段是难点（证据在「项目介绍」）：调度器把转置图切成两个 subgraph，静态形状下
算不出代价常数、全部候选保留——**任何一个降不下去，整图就失败**。缺口一卡在跨步搬运
（S0），缺口二卡在 Pad（S1），两个都得补。

交付物：

- 第二部分实现 + lit/UT 用例；
- 4 个 transpose CPU E2E 目标全绿；
- a2/a3（910B2/910B3 实机）：和 legacy 同图对比，结果一致（bit 级）、速度不慢
  （p50 不劣化）；
- a5（950）：没有实机，只做代码 + CPU 仿真验证。

## 术语速览

| 术语 | 白话解释 |
|------|----------|
| GM / UB | GM=显存（张量住的地方，GB 级）；UB=片上 buffer（约 240KB，一次只装得下一小块） |
| tile（块） | 一次循环迭代处理的那一小块数据。张量装不进 UB，所以切块；kernel 循环着"搬一块、算一块、写一块" |
| T 轴 / t 轴 | 调度器给每根轴起的成对名字：T=第几个块（外层循环变量）；t=块内沿这根轴有多少个（运行期 tiling 值，所以在 IR 里是符号） |
| 调度器 | 决定图怎么切、数据怎么搬的部件（`autofuse/optimize/`）。它的输出是本设计的输入，本设计不改它 |
| subgraph | 调度器切图得到的调度单元：各有名字和 tiling 方案，各自生成一份代码（plan），运行期靠 tiling_key 挑一份执行 |
| S0 / S1 | 本例两个 subgraph 的简称，见「项目介绍」的表 |
| 模板 / tiling case / plan | 一个 subgraph 的一种完整落地方案；case 是方案编号，plan 是该方案生成的代码 |
| desc | 搬运某一端的"布局说明书"，挂在 op 属性上。operand_descs=读端说明书；destination_descs=写端说明书；result_descs=结果值说明书 |
| 合轴 / 级（段） | 把逐轴的 tile 描述压缩成"段序列"。轴只是描述用的纸面分割线，内存里的数据不知道自己属于哪根轴——相邻两轴之间没洞（外轴步长恰好等于内侧总长）时，它们本来就是一段更长的连续，算一段。为什么必须合：硬件指令只认"段数×段长"两级，不合就表达不了；合了段长更长、指令更少（例子见功能需求 1） |
| block 形态 | 硬件块拷贝指令的形态：拷 blockCount 段、每段 blockLen 个、源侧段间隔 srcStride、目的侧段间隔 dstStride。硬件只认这种"两层"形态 |
| 包装调用 | autofuse 包在 CANN 官方 API 外面的辅助函数（如 `DataCopyPadExtend`，在 `autofuse/ascendc/api/`），单位换算在包装里做 |
| Size / ActualSize | tile 尺寸的两种口径：Size=补到 32B 对齐后的大小；ActualSize=真实大小（尾块修正后） |
| 物化 | 把符号表达式算成 IR 里的实际值/指令参数（materialize 的直译，代码里的正式叫法） |
| align_num | 一个 32B 块装几个元素：f16=16，f32=8 |
| 逐字等价 | 生成的代码和 legacy 生成的几乎一字不差：同函数、同参数算式、同分支条件 |
| perm | 转置的轴排列。转置算子**没有 perm 属性**——轴序信息完全由输入/输出两端 desc 的 axis 数组表达，perm 从两端轴 ID 对应关系推导 |
| mem 档 / reg 档 | 两套下降实现：mem=910B 系（供给表 `autofuse/ascendc/api/`）；reg=950 系（供给表 `v35/ascendc/api_regbase/`）。选哪档由 `afir.target` 驱动，代码里不出现芯片名 |
| ConfusionTranspose / TransposeExtend | 转置在两档各自的包装调用（mem / reg） |
| bridge | 把调度产物翻译成 AFIR IR 的层；靠一张注册表决定"调度节点 → IR 算子"的映射 |
| implementation | 把 IR 算子降成 AscendC 调用的实现；先由能力层（capability）判断支不支持 |
| ATT / TilingData / tiling_key | 运行期传参结构 + 模板选择键；kernel 入口按 `t.graph{i}_tiling_key == plan_id` 挑模板 |
| lit / UT / E2E | 三层测试：lit=对生成的中间代码做文本断言；UT=C++ 单元测试；E2E=生成器+数值比对的端到端 |

## 范围

**做**：

第一部分（转置算子本体，详见该章）：

- `afir.transpose` 的 ODS、bridge 注册表行、`indexing_maps`（perm 的显式表达）与
  `api_tiling_field` 属性。
- AscendC 方言两个包装 op 与手写发射器：`ascendc.confusion_transpose`（mem）、
  `ascendc.transpose_extend_reg_base`（reg）。
- 两个 implementation：`transpose.ub_ub_extend`（mem）、`transpose.ub_ub_reg_base`（reg）；
  >4 维外抛（reg 档复用 `VectorizedLoops`）。
- 官方头声明机制：`ApiExtensionHeader` 第三段列表（reg 依赖 CANN 官方头
  `kernel_reg_compute_datacopy_intf.h`）。
- 配套 lit 7 文件 + UT 计数 + e2e 用例注册。
- phase A 配套：`AFIRKernelBodySignature.cpp` 把 `afir.api_tiling_fields` 并进 canonical
  TilingData 实参。

第二部分（整图使能）：

- 跨步搬运的段几何物化与统一发射（`emitScheduledTransfer` + `DataCopyPadExtend`，
  参数算式与 legacy `SetDmaParams` 同式）。
- `afir.pad` 全链：方言定义、bridge 注册表行、下降实现（legacy `PadApiCall` 等价：
  运行期 `AscendC::Pad` / `AscendC::DataCopy` 二选一、PadTiling 字段、tmp_buf 操作数）、
  AscendC 方言包装算子和打印函数。
- RemovePad 整窗口 gather 形态、Nddma 判据放宽。
- allocator 失败回滚双重释放修复（让报错能看到）。
- a2/a3 上板实测（和 legacy 比结果、比速度）。

**不做**：

1. a5 上板实测——没有实机，上板推迟（验收标准第 9 条）。reg 侧必须执行 CPU 数值仿真；
   搬运包装按真实布局选择 Normal/Compact，不能用 lit 或 mem 真机结果代替 reg 数值。
2. 调度策略的任何改动——调度生成什么，我们降什么（含保留 vs 消除双模板与打分、
   `GetPaddingMode` 的 has_transpose 分支、API 缓存禁用，全部原样）。
3. `Ub2ub`：同族另一个（sched 阶段内部节点，不对 py 构图外显，`ascir_builtin_ops_v1.cpp:642`
   注释）。
4. nddma 路径的进一步改动：沿用新基线的 `afir.nddma` / `DataCopyNddma` 能力；
   上游已整体删除自洽判据，本分支不再修改 `Nddma.cpp`（见 A7 复核）。基线已有的
   `load_broadcast_nddma_store` 属 nddma 接通/broadcast 特性，非本特性范围。
5. 跨段搬运的等价改写或优化——legacy 没有的优化一律不加。
6. rollback 之外的 allocator 重构。
7. Pad 的 dtype 扩表——legacy 支持多少我们支持多少。
8. `DT_BOOL` 与 i64 暂不接入；mem 另有 i32/ui32 有意窄化（见 dtype 边界）。
9. mem 档枚举表外 perm 不做"拆两次调用"的等价改写（见「不做的事」⑨）。
10. Python 侧任何改动（`ascir.ops.Transpose` 与 `ascir_api.py:2266` 的 `Transpose()` 已存在）。

# 总体概述

## 软件概述

### 项目介绍

**转置算子的三个特殊性**（第一部分设计时查证的事实）：

1. 它是布局族里**唯一有独立调度路径**的算子：`ComputeType::kComputeTranspose`
   （`ascir_builtin_ops_v1.cpp:685`，v2 注册靠同名合并继承，`ascir_registry.cc:266`），
   tiling 走 `GenTransposeTilingGroup` 的 x/y 双切分（`tiling_group.cpp:739`），与
   elementwise 的单侧切分本质不同。
2. **它没有 perm 属性**：转置语义完全由输出 tensor view 的轴序表达
   （`docs/zh/autofuse/design/features/transpose.md:36`），codegen 从输入/输出
   `vectorized_axis` 的轴 ID 对应关系推导 perm。
3. legacy 两侧实现是覆盖矩阵里差异最大的 **D2 档**（API 名/扩展头/tmp 三项全差，
   `ascir_op_lowering_matrix.md:209`）：

| | mem（910B 系） | reg（950 系） |
|---|---|---|
| ApiCall | `TransposeApiCall` | `TransposeRegApiCall` |
| API | `codegen::ConfusionTranspose<T>(y[off], x[off], tmp_buf, transposeType, apiTilingData)` | `TransposeExtend<inner, total>(dst, src, tmp_buf, {dims}, {src_strides}, {dst_strides})` |
| perm 表达 | 21 种固定 perm 折成 7 个 `AutoFuseTransposeType` 枚举，表外 `TRANSPOSE_INVALID` 编译失败（`transpose_api_call.cpp:85-111`） | 任意 perm，dims/strides 直接传参（`reg_transpose_api_call.cpp:53-68`） |
| 轴数 | 表内隐式 ≤4D | inner ≤3、total ≤4；**>4 维外抛 `outer_for` 循环**（`reg_transpose_api_call.cpp:156-169`） |
| tmp | `CalcDefaultTmpSize` 固定值 | `CalcTransposeTmpSizeV2` 精确值（Gather 索引缓冲，`life_time_axis_id = -1` 全生命周期） |
| host tiling | `ConfusionTransposeTiling` 18 字段（ATT 生成） | 不需要 |
| 尾随同步 | `AscendC::PipeBarrier<PIPE_ALL>()`（`transpose_api_call.cpp:158`） | 无 |
| 扩展头 | `transpose_base_type.h` + `transpose.h` | `transpose_reg_base.h` |
| 官方头 | `basic_api/kernel_operator_vec_transpose_intf.h` | `basic_api/reg_compute/kernel_reg_compute_datacopy_intf.h` |
| dtype | I16/UI16/I32/UI32/F16/F32（`ascir_builtin_ops_v1.cpp:688`） | 上述 + **BF16/BOOL**（`ascir_builtin_ops_v2.cpp:1073`） |

**第一部分的 lit 覆盖**：`afir-transpose-{reg-base,reg-base-bf16,outer-loop,unsupported,
invalid,mem-extend}.mlir` + `target/ascendc-transpose-reg-base.mlir`；UT
`BridgeOperationRegistryTest` 名单 47→48 等全绿。

**但整图 E2E 跑不通**（`AF_ENABLE_MLIR_CODEGEN=1` +
`MlirE2eCpu_load_transpose_store_2x37x23_generator`）。一路排查到最后，发现卡在**整图**
层面：不是转置算子的问题，而是转置图把两类**之前的 E2E 图从来没出现过的数据形态**推到了
load/store/Pad 面前。

**调度器把图切成了两个 subgraph**（同一个转置的两套竞争方案，运行期打分选一个）：

| 简称 | 名字 | 路线 | 图内容 | tiling case |
|------|------|------|--------|-------------|
| S0（subgraph0） | `mlir_load_transpose_store_2x37x23_0` | 保留转置：在 UB 里做转置 | `Load → Transpose → Store` | 2 个 |
| S1（subgraph1） | `group_transpose` | 消除转置：转置折进 load/store 的取数图样里，图里没有 Transpose 节点 | C0/C1：`Load → RemovePad → Pad → Store`；C2：`Load → Store` | 3 个 |

编号规则：下文 `S0G0C0`、`S1C0` 之类 = 第几个 subgraph / group / case。

关键事实：这张图静态形状下算不出代价常数，调度器的 `SelectStaticWinner` 没法提前选出
赢家，于是**全部 5 个方案都保留**（`PlanSelector.cpp:82-101`）——每个都要过 MLIR 下降，
**任何一个降不下去，整图就失败**。这就是两个缺口都必须补的原因。

两个缺口分别是什么：

1. **S0 的 load/store 是跨步 tile**。以 load 为例，用例形状 `(2,37,23)` f16：tile 在
   GM 侧的取数脚印是 5 根轴、跨着步的（最内轴连续，行与行之间有洞，如行距
   `851=37×23`）；进 UB 后每行还要补到 32B 对齐（行距 `16*Ceil(z2t/16)`）。现有下降
   （`DataMovement.cpp:88-91`）只会发"一整段连续"的单段拷贝（`blockCount=1`），能力层
   一判"不是单段"就拒绝。代码注释原本的假设是"多段的走 ND-DMA 那条路"——但这张图的
   load 是普通 Load 节点，调度器没把它融成 Nddma，那条路在这张图上不存在。
2. **S1 的 C0/C1 带 Pad 节点**。Pad 在 MLIR 路径三层全缺：bridge 注册表没这行（节点
   翻译不进 IR）、方言没有 `afir.pad`、没有下降实现。对照 RemovePad 三层全有——所以
   Pad 就是把 RemovePad 的链路照着复制一份。

legacy 对这两个缺口都有现成答案：前者在 `load_api_call.cpp` / `store_api_call.cpp`
（+ `api_call_utils.cpp` 的合轴和拆循环），后者在 `pad_api_call.cpp`。本设计把它们
等价搬进 MLIR codegen。

### 产品环境介绍

- CANN Toolkit 9.2.0 的 AscendC 头（`dav_c220`）；CPU 仿真和 NPU 用同一份头文件
  （`AGENTS.md` 经验 3.3 核对过）。
- 生成器 / kernel 编译 / 数值执行都在 `autofuse/mlir/test/e2e/cpu` 既有框架里
  （`AddCpuCase.cmake` 的 `AF_ENABLE_MLIR_CODEGEN` 构建规则，双平台现成：默认跑
  Ascend910B1 mem 侧，`PLATFORM reg_base` 切 950），不新增环境变量或路径。
- 不碰 `autofuse/att/`（稳定性约束），不碰 Python 侧。
- 官方头机制已由第一部分建好（`ApiExtensionHeader` 第三段列表）：op 声明 CANN 官方头
  → kernel 顶部 `#include`，不内联，由 device 编译器 `-I` 解析。

**平台和验证环境**（双档定义见 `test/probe/CapabilityBoundaryProbe.cpp:62-76`）：

| 档位 | 芯片 | 有没有实机 | 本设计怎么验 |
|------|------|-----------|--------------|
| mem 档 | a2/a3 = Ascend910B2/910B3（NpuArch 2201 族） | **有** | 代码 + lit/UT + CPU E2E + **上板实测**（和 legacy 比结果与速度） |
| reg 档（v35 路径） | a5 = Ascend950PR_9599（NpuArch 3510） | **没有** | 代码 + lit/UT + CPU E2E 仿真（`*_reg_base` 目标）；**上板推迟** |

两部分的实现都是平台共享的（`afir.target` 驱动选档，代码里不出现芯片名）。mem 侧的包装
函数（`autofuse/ascendc/api/datacopy.h` 等）在 950 的 CPU 仿真上能编译能跑（既有
`*_reg_base` E2E 的先例），所以不引入 reg 档专用的下降形态。

### 软件功能

- F1：`afir.transpose` 从 bridge 到两档 AscendC 调用的全链下降，含 >4 维外抛。
- F2：跨步 load/store 的段几何物化与统一发射（`DataCopyPadExtend`），产物和 legacy 逐参数一致。
- F3：`afir.pad` 从 bridge 到 AscendC 调用的全链下降。
- F4：allocator 失败回滚不再崩溃，真正的错误信息能看到。
- F5（方法论）：a2/a3 上板与 legacy 对拍 + 泛化路线。

### 设计约束

1. **和 legacy 逐字等价**（两部分共同的验收根基）。生成的 kernel 代码要和 legacy 生成的
   几乎一字不差：
   - 调**同一个**包装函数（`ConfusionTranspose` / `TransposeExtend` / `DataCopyPadExtend`
     / `AscendC::Pad` / `AscendC::DataCopy`），不换函数、不绕过包装直接调官方 API；
   - 每个参数用**同样的算式**算出来（比如 `srcStride = 行距 − 块长`），不自己发明
     "数学上等价"的算法——等价的算式在整数溢出、尾块修正这些边界情况下可能算出不同
     的数；
   - if 的**分支条件原样照抄**（如 Pad 的条件是哪两个量不相等），不改比较方向、
    不换比较对象。

   为什么这么要求：a2/a3 上板验收是"和 legacy 结果一致 + 速度不慢"。代码一字不差，
   这两条自动成立，不用额外论证；任何一点自作主张的改动，都得单独上板举证。
2. **单位约定**。调用点的四个参数一律按**元素个数**传；字节/对齐换算全在包装函数里
   做（`datacopy.h:14-39`：GM 侧 ×sizeof(T)，UB 侧 ÷align_num），下降代码里不做任何
   换算（和 legacy 调用点一致）。UB 侧整数除法的坑见「关键技术」第 2 条。
3. **desc 是唯一事实源**。分析的输入只有 op 属性里的 desc（读端/写端说明书）+
   `indexing_maps`，不回头去查调度器的 C++ 图对象。好处：(a) IR 一旦生成，含义就定死
   （确定性）；(b) 能力层和下降层看同一份数据，不会出现"能力层说支持、下降做不出"的
   两头打架；(c) lit 测试可以手写一个带假 desc 的 op 就测，不用真跑调度器建整张图。
   一个细节：判 TileInner 要查的轴类型表在 `afir.asc_graph` 这个 op 里，它本身就在 IR
   模块中，读它仍然是"只读 IR"，不违反本条。
4. **既有路径零回归**。单段连续 load/store（全部既有绿 E2E）生成的代码必须一字
   不变——由"1 段的参数算式和现状完全相同"保证，回归靠全量 E2E 锁住。
5. **不得硬编码芯片**：双 implementation 按 `ApiFamily`（`MemBased` / `RegBased`）由
   TargetProfile 查询选择，stable id 不含平台名；轴数上限 4/3 来自 `TransposeExtend`
   的模板特化面（`api_regbase/transpose.h:309-335`），是 API 能力不是芯片判别。
6. **SuperKernel 解耦 / ATT 不迁移**：不碰 `super_kernel/`；`TransposePerf` 的指令级
   建模已存在且外抛已覆盖（`outer_count` 乘进总代价，`ascendc_regbase_perf.cpp:272-277,560`），
   本需求不新增性能建模。
7. **编码红线**。运行期参数（blockCount/blockLen/stride）的物化走既有的溢出检查机制
   （`materializeCheckedAdd/Mul` + `cf.assert`）。

### 假设和依赖关系

**第一部分设计时查证的事实表**（仍然有效，别再查第二遍）：

| 前提 | 状态 | 依据 |
|---|---|---|
| Transpose 无 perm 属性，轴序即 perm | **已验证** | v1/v2 注册均无 `.Attr`（`ascir_builtin_ops_v1.cpp:682-688`、`ascir_builtin_ops_v2.cpp:1070-1073`）；perm 由 `ReorderInputStrideByOutputAxisOrder` 从两端轴 ID 对应推导（`reg_transpose_api_call.cpp:53-68`）；测试构图 API `AscGraphBuilder::Transpose(name, input, axes)` 就是重排 `y.repeats`/`y.axis`（`easy_asc_graph/asc_graph_builder.cpp:290-310`） |
| mem 实现按**字节宽度**分派，整型是「声明即支持、从未验证」 | **已验证** | `transpose.h` 全部 20 处 `if constexpr (sizeof(T)==sizeof(half)/sizeof(float))`，分支内全是地址列表 + `TransDataTo5HD<T>` + `DataCopy`，零数值语义；但 legacy UT 只实例化 half/float、e2e 只跑 float。dtype 口径：MLIR 全收 6 种 + 补 int32 用例（**已修订**：布局/公式错位使 int32 数值必错且 legacy 同错，mem 档已摘除 i32/ui32，见验收标准 2） |
| reg dtype 边界 | **声明与数值验证分开** | MLIR ODS 明确拒收 i64/bool，拒收由 `afir-transpose-dtype-boundaries.mlir` 看护；legacy 声明 i64 不等于 MLIR 支持。bf16 已接入注册矩阵与发射 lit，其数值仍未覆盖；f32/f16 的 reg 全管线数值已通过 |
| tmp buffer 的 `life_time_axis_id = -1` 已被 bridge 接受 | **已验证** | `CollectTmpBufferLifetimes` 只拒 `< -1`（`ScheduleToAfir.cpp:832`）；Transpose 是 legacy 唯一用 -1 的 API（`transpose.md:61`） |
| apiTilingData 的 schema 侧已全线就绪 | **已验证** | bridge `GetApiTilingInfo` 已有 Transpose 分支（`ScheduleToAfir.cpp:745-751`，18 个 `paramN`，字段名 = `<nodeName>_tilingData_<templateId>`）→ func 级属性 → `GenTilingData.cpp:124` 建成嵌套 TilingSchema struct；phase A `AFIRKernelBodySignature.cpp` 把同一属性并进 canonical 实参，conversion 才能按字段名建 `emitasc.member` |
| 官方头机制（第一部分建） | **已验证** | `ApiExtensionHeader` 第三段列表 + `AddKernelIncludes.cpp` 的 `officialHeaders` 收集；inliner 取不到键原样保留 |
| >4 维外抛的 MLIR 对应物已存在 | **已验证** | `VectorizedLoops.h:11-24` 自述「legacy `CreateComputeNodeOuterFor` 的 MLIR 对应物，偏移算式与 `CalcInnerOffset` 同一个」；广播 segmented 档已在用 |
| CPU E2E 双平台都通 | **已验证** | `AddCpuCase.cmake:57-96`：默认 910B1（mem）；`PLATFORM reg_base` 切 950 |

**第二部分的假设**：

1. desc 的 `vectorized_strides` 单位是元素、且已经带上了对齐后的跨度（S0 load 的
   `16*Ceil(z2t/16)` 在 memory-allocate 里自洽验证过）。**待确认**：legacy
   `Tiler::Size()`（`codegen_kernel.cpp:843` 附近）和 desc 表达式的对齐语义在每种尾块
   场景下是否一一对应——实现期用 lit 对 S0/S1 全部 5 个模板和 legacy 逐字比对，有
   分歧就停。
2. S1 的候选确实含 Pad 模板（不是猜测，两个证据：`PlanSelector.cpp:100` 的全候选保留
   机制证实没被筛掉；调度 dump `ascgraph_00015/00016` 里亲眼看到 Pad 节点）。
3. RemovePad 的下降已存在（`remove_pad.tensor.extend`），S1 的 RemovePad 直接复用。
4. **硬件可用性**：a2/a3 实机可用（npussh 设备 + `AGENTS.md` 3.4 直调链路）；a5 没有
   实机——reg 档只做代码 + CPU 仿真，上板推迟（验收标准第 6 条）。
5. **reg 档的图里有没有 Pad**：`group_transpose` 由共享调度器生成、没有平台门控
   （`transpose_schedule_case_generator.cpp:138`），预期 950 侧也有 Pad；实现期跑
   `check-e2e-cpu-load_transpose_store_reg_base_2x37x23` 拿 dump 确认。legacy reg 侧
   没有 Pad 的 api call——如果 reg 图确实有 Pad，那 `pad.ub_ub_extend` 就是 reg 档第一
   个 Pad 下降，以 CPU 仿真数值为准验收。

# 需求分析与设计

## 整体介绍

数据流不变：能力层按 desc 判断支不支持 → 选档 → 下降 pattern 发 AscendC 调用。

```
afir.transpose ──────────┬─ mem 档：perm 折 21 种表 → 枚举 + apiTilingData + PipeBarrier
（第一部分）              └─ reg 档：perm 分析 → 三组数组 + <inner,total>；>4 维外抛 scf.for

afir.load/afir.store ────┬─ 能力层+pattern 共用：desc 的段几何能否物化（VectorSegmentation 分类）?
（第二部分）              │     ├ 1 段  → 单条 block(1, len, 0, 0)    【既有形态，一字不变】
                         │     ├ ≤2 段 → block(count, len, srcGap, dstGap)
                         │     └ >2 段 → 外层循环消段 + 内层段拷贝
                         │               （段几何由 materializeScheduledAxisExtents 符号物化）
afir.pad ────────────────┴─ 能力层：desc 满足 Pad 形态 → pattern 发运行期二选一
（第二部分）                   （AscendC::Pad / AscendC::DataCopy，scf.if 照搬）
```

第二部分的背景一句话：`DataCopyPadExtend` 本身就是硬件内置的两层 for 循环（拷
blockCount 段、每段 blockLen 个、段间隔固定）。合轴就是把 tile 的形状归约成这个形态；
归约不掉的段数用显式 for 循环消掉。

---

## 第一部分：转置算子本体

> 本章是设计决策与查证事实的存量记录，也是泛化路线（P1-P5）的实现依据。文件行号为
> 写作时快照，以符号名检索为准。

### A1. `afir.transpose` ODS 与 verifier

- `Ops/Layout.td` 新增 `AFIR_TransposeOp`，基类 `AFIR_BufferizableUnaryOpBase`，形态是
  **搬运类**：两端形状不同（轴序重排）、元素数相同。
- 元素类型集合 `[I16, UI16, I32, UI32, F16, F32, BF16]`（两侧注册表并集去掉 BOOL；ODS
  比 implementation 宽与 `RemovePad` 同一条判法）。
- 接口三件：`AFIRDataMoveInterface`（转置是"同一批元素换摆法"，语义就是搬）、
  `AFIRDestinationStyleInterface`、`AFIRIndexingMapInterface`。**不挂**
  `AFIR_TransferIn`/`AFIR_TransferOut`（两端都在片上，不跨 GM 边界；不挂即落 compute
  pipe，与 legacy 的 AIV_VEC 执行一致）。
- `verifyIndexingMaps` 走 `equalElementCount = true`（转置保元素数，与 `afir.load`
  同档）；verifier 追加：输入与 destination 的 rank 相同、无 scalar 输入（对应 legacy
  `IsNodeValid` 仅断言无 scalar，`v2_ascir_codegen_impl.h:3750-3754`）。

### A2. bridge 接入

- 注册表一行：`M("Transpose", afir::TransposeOp, BridgeOpForm::Both)`；`isTransferNodeType`
  收编 `"Transpose"`——`indexing_maps` 由 `makeTransferIndexingMapsAttr`（`:322-358`）
  从两端 `axis` 数组生成，**perm 不需要新属性**，两端轴序差就是 perm。
- 节点级属性 `api_tiling_field`：字段名按节点名生成（`<nodeName>_tilingData_<templateId>`），
  func 级属性区分不了同模板多个 Transpose 节点；挂 op 上是单一事实源，pattern 直接读。
  字段名的清洗必须**跟随 ATT host 侧同一函数**（`ascgen_utils::GenValidName`）——图节点
  名可含 `/`（torch_npu 的 `graph_hint/transpose`），两侧清洗函数不一致时 tiling 结构体
  字段名与 host `tiling_func.cpp` 引用对不上，host 编译报 `no member named ...`。CPU e2e
  探不到这条（图节点名无 `/`，模拟器路径不走 host tiling_func 编译链）。
- tmp buffer 走既有 `makeTmpBufferAllocation`，-1 生命周期由 `CanonicalGroupKey` 分组。
- `BridgeOperationRegistryTest` 名单 47 → 48。

### A3. 官方头声明机制

- `ApiExtensionHeader<headers, preludes>` 增加第三段列表 `officials`（`Autofuse/Base.td:48`），
  生成 `getExtensionOfficialIncludes()`；`AddKernelIncludes.cpp` 收集与发射（固定块之后、
  扩展头之前，与 legacy include 顺序一致）。
- **不进** `ApiExtensionInliner` 的替换预期：inliner 只替换注册表里有的键，取不到原样
  保留——官方头正好走这条保留路径。第一个用户：reg 档的
  `basic_api/reg_compute/kernel_reg_compute_datacopy_intf.h`（MicroAPI `Gather` 等声明）。
- **扩展头的发射次序是声明链拓扑序**（`AddKernelIncludes.cpp`）：include 的先后不能只按
  字典序——`ConfusionTranspose` 的 `["transpose_base_type.h", "transpose.h"]` 恰好反字典
  序（`'.'` ASCII 46 < `'_'` 95，但包装用的枚举必须先内联）。按声明链拓扑排 + 字典序
  平局裁决：存量（声明链恰为字典序）的 include 块逐字节不变，反字典序对由拓扑边保证
  次序，两个 op 声明链成环时报诊断。

### A4. 两个 ascendc 包装 op 与手写发射器

都不带 `AscFunc`（自动 emitter 发不出「栈数组+双模板实参+枚举+struct 传参」），手写
printer 落 `lib/Target/AscendC/Autofuse/Transpose.cpp`：

- reg：`AscendC_TransposeExtendRegBaseOp`。发射文本（对齐
  `reg_transpose_api_call.cpp` + `GenDimensionParam`）：

  ```cpp
  TransposeExtend<2, 4>(dst[off], src[off], tmp_buf, {static_cast<int16_t>(d0), ...},
                        {static_cast<int16_t>(s0), ...}, {static_cast<int16_t>(r0), ...});
  ```

  三组尺寸是**调用内联的花括号列表**（legacy 原形态）；每个元素 cast 到 `RangeType<T>`
  的索引宽度（`sizeof(T)≤2` 用 `int16_t`，否则 `int32_t`）。模板实参**只发
  `<inner, total>` 两个**：`T` 从 `LocalTensor` 形参推导，第三个实参写出来反而与 legacy
  文本不同。
- mem：`AscendC_ConfusionTransposeOp`。发射文本（对齐 `transpose_api_call.cpp:146-158`）：

  ```cpp
  codegen::ConfusionTranspose<T>(dst[off], src[off], tmp_buf, AutoFuseTransposeType::TRANSPOSE_ND2ND_XXX, tiling.<field>);
  AscendC::PipeBarrier<PIPE_ALL>();
  ```

  枚举名直接由属性打印（legacy 的局部变量中转省去，语义等价；若 review 要求逐字再加回）。
- 两者都进 `Autofuse/Ops.h` 的 `PrintableOpTypes`、`pyasc_ascendc_op_inventory.csv` 加行、
  `check_ascendc_td_reachable.py` 守 include 图（新 ODS 文件 `Autofuse/OpVecTranspose.td`）。

### A5. reg 档 implementation `transpose.ub_ub_reg_base`

- capability 顺序：`hasCanonicalDescriptors` + 两张 indexing maps → 执行单元 feature →
  `checkApiFamily(RegBased)`（undetermined 不倒向任何一边）→ `checkTmpBuffer(true)` →
  操作数结构（1 入/1 目的地/1 tmp）→ dtype 7 种（BOOL 除外）→ `buildTransferPlan`（必须
  `ScheduledContiguous`）→ perm 分析。inner 封顶 3、`total = min(向量化轴数, 4)`、
  >4 维走外抛——封顶与外抛都是可收形状，这一层没有"轴数不支持"的拒绝。
- **perm 分析**（本档核心，`ReorderInputStrideByOutputAxisOrder` +
  `GetContinuousInnerAxisNum` 的 MLIR 对应）：从 `indexing_maps` / 两端 desc 的 `axis`
  序推出"输出第 i 轴对应输入哪根轴"；按输出轴序取输入 `vectorized_strides` 得
  `src_strides[]`；输出 `repeats` 得 `dst_dims[]`、输出 `vectorized_strides` 得
  `dst_strides[]`。**已知偏差**：legacy 对末两轴 stride 发 `ActualSize`（Compact 模式），
  MLIR 发 desc 原文——stride 不含切分轴尺寸符号时两者逐字相同，含时是否分叉由 CPU E2E
  尾块用例与 NPU 对拍回答。
- **>4 维外抛**：`VectorizedLoops::materializeVectorizedSegments(op, rewriter, plan,
  transferPlan, tiles, /*apiWindowSize=*/4)` 为最内 4 段之外的每段生成一层 `scf.for`，
  tile 偏移 `Σ iv_j × stride_j`（与 legacy `CalcInnerOffset` 同式；落点不同：legacy 改
  inner offset，MLIR 改 tile 下标，生成的 C++ 语义等价）。
- 进 `registeredFamilies()` 侧表（reg 行，可建模子集），mem/reg 双侧覆盖断言由此管到它。
- lit：`afir-transpose-reg-base.mlir`（f32 3D+4D）、`afir-transpose-reg-base-bf16.mlir`
  （**reg 收 bf16**；GM 腿进不来 bf16——搬运族 ODS 元素类型表没有它，真实图靠 cast
  换入）、`afir-transpose-outer-loop.mlir`（5 维外抛，同一文件在 910B 上是表外 perm
  的反例）。**没有 2D 格**——reg 的 `ScheduledContiguous` 要求输出侧切分轴在最外，
  2D 转置必然把它换离，那种形态 reg 档在 plan 判断就拒，是 mem 专属。

### A6. mem 档 implementation `transpose.ub_ub_extend`

- capability：`checkApiFamily(MemBased)` → `checkTmpBuffer(true)` → dtype 4 种
  （f32/f16/i16/ui16；已摘除 i32/ui32，见验收标准 2 与
  `Implementations/Transpose.cpp` 的「口径 A 修订」注释）→ perm 折叠查表。
- **perm 折叠**：输出 `vectorized_axis` 在输入中的位置序列查 21 种表
  （`transpose_base_type.h:35-57` `kPermutationTable`：2D 1 种、3D 5 种、4D 15 种，
  归约到 7 个 `AutoFuseTransposeType`）。**表外 perm 一律 `unsupported` 回落 legacy**
  ——legacy 在这里是 `TRANSPOSE_INVALID` 编译失败，MLIR 报 unsupported 让整图回落，
  行为严格不劣于 legacy。
- **apiTilingData 访问链**：pattern 读 op 属性 `api_tiling_field` →
  `createTilingSchemaMember(rewriter, loc, tiling, "<field>")` 一次成员访问得 struct
  值 → 作为操作数传给包装 op。**不逐个取 paramN**：调用吃整个 struct，18 个成员由
  ATT 在 host 侧填好。
- 尾随 `AscendC::PipeBarrier<PIPE_ALL>()` 照发（emitter 自带）。按 `AGENTS.md` §3.5
  教训，该屏障**不可静默删**——CPU 仿真对同步语义全盲，NPU 上按产物级 A/B 验证。
- lit：`afir-transpose-mem-extend.mlir`（3 个代表枚举 ONLY/021/0321）。

### A7. Nddma 描述子口径（2026-10-31 新基线复核）

- 旧版将 `GenNddmaNode` 保留输入 `strides` 误判为漏抄；此结论撤销。搬运节点的
  `strides` 描述 GM 输入布局，`vectorized_strides` 描述片上布局，广播输入轴的零步长
  与片上非零步长可以合法并存，不应互相校验。
- 新基线 `4aa77dd6` 已包含上游 `a3a534ec`：在 `TransferPlan` 修复 Store 的 GM
  偏移来源，并整体删除 `resultDescriptorIsSelfConsistent`。本分支 rebase 时沿用
  基线实现，撤去原有零/非零翻转判据改动，不修改 `Nddma.cpp` 或调度侧。
- `afir-nddma-reg-base.mlir` 保留刻度差正向发射断言，删除依赖已移除判据的
  ZEROFLIP 负例。`load_broadcast_nddma_store` 已由新基线启用，属于基线 broadcast
  特性，不作为本转置特性的新增验收项。

---

## 第二部分：整图使能

### 功能需求 1：跨步 load/store 的段几何与统一发射

1. 介绍

   跨步 tile 的 desc 是**逐轴**描述的（每根轴几个、两侧步长多少），而硬件块拷贝
   `DataCopyPad` 只认"两层"形态：拷 blockCount 段、每段 blockLen 个、两侧各一档固定
   段间隔。把逐轴描述归约成这个形态，就是**合轴（段几何）**：

   - **为什么必须合**：硬件指令只有"段数 × 段长"两级，5 根轴的 tile 塞不进去。
   - **为什么能合**：轴只是纸面分割线——相邻两轴之间没洞（外轴步长恰好等于内侧总长）
     时，"走完内轴再走外轴"和"走一段更长的连续"是同一回事；**有洞绝对不能合**（合了
     会跳数据）。
   - **广播轴消隐**：两侧步长同时为 0 的轴不产生拷贝（`ShouldIgnoreDataCopyZeroAxis`）。
   - **段数超过两层**：最内段当段长、倒数第二段可当段数，其余的段用外层循环消掉，
     每次迭代换个起点再发一条（一条指令只有一档间隔，表达不了两种不同的间隔）。

   例子（S0 的 load，真实数字）：GM 侧行距 23、面距 `851=37×23`；UB 侧行距
   `16C`（C=Ceil(z2t/16)，每行补到 32B 对齐）。从内往外：z2t 自己成一段；z1 想并进
   去——GM 侧 `23≠z2t`（只有满尾块相等）、UB 侧 `16C≠z2t`（补位缝永远在）→ 并不了，
   新起一段；z0t 想并进 z1 段——GM 侧 `851==37×23` ✓、UB 侧 `592C==37×16C` ✓ →
   并进去。结果 2 段：`[(z0t×37, 23, 16C), (z2t, 1, 1)]`。

2. 输入

   op 的 desc（GM 侧：axis/repeats/strides；UB 侧：vectorized_axis /
   vectorized_strides）+ `indexing_maps` + 轴类型表（`afir.asc_graph` 的 axes，判
   TileInner 用，对应 legacy `tpipe.tiler.GetAxis(...).type`）。desc 是唯一事实源，
   不回查调度器的 C++ 图对象——能力层和下降层看同一份数据，不会两头打架。

3. 处理（现行发射链）

   - 段几何不预先压成一张表，而是在下降时由 `materializeScheduledAxisExtents`
     （符号步长物化）从 desc 逐轴物化，`emitScheduledTransfer`
     （`Implementations/DataMovement.cpp` / `TransferPlan.cpp`）据此统一发射
     `DataCopyPadExtend`：1 段是既有单块形态（block 四参数 `1, len, 0, 0`，一字不变）；
     2 段发一条 block 形态；>2 段套外层循环。
   - 能力层与 pattern 共用 `VectorSegmentation` 的分类——"能力层说支持、pattern 做不出"
     不可能出现。
   - 参数算式与 legacy `SetDmaParams`（`api_call_utils.cpp:175-208`）同式，调用点一律
     按元素个数传，字节/对齐换算在包装函数里做：

   | 参数 | 1 段 | 2 段（i=倒数第二轴） |
   |------|------|--------------------------|
   | blockCount | `1` | `ActualSize(repeat[i])`（尾块修正后） |
   | blockLen | `ActualSize(repeat[last])` | `ActualSize(repeat[last])` |
   | srcStride | `0` | load：`ActualSize(gmStride[i] − repeat[last])`；store 镜像（src 侧换成 UB） |
   | dstStride | `0` | `Size(ubStride[i]) − ActualSize(repeat[last])` |

   `Size(x)`=32B 对齐后的尺寸（desc 的 ubStride 表达式已带对齐值）；`ActualSize(x)`=
   尾块修正后的运行期值。接上面的例子：`(blockCount=z0t×37, blockLen=z2t,
   srcStride=23−z2t, dstStride=16C−z2t)`；代入满尾块 z2t=23（z0t=1）就是
   `(37, 23, 0, 9)`——GM 侧行挨着（间隔 0），UB 侧每行有 9 个补位。

     - **行距按实际 API 模式解释**：Normal 模式按 32B 补齐每行；reg 的紧凑行布局应沿用
       legacy 的 `PaddingMode::Compact`，不能套用 Normal 的步距。此前 `z2t=6` 的越界归因
       被误记为 solver 约束，本轮复现确认 MLIR 缺失了 Compact 包装选择。修复发射模式、
       连续后缀的实际 stride 与合轴后，原非对齐形状数值通过；不修改 ATT/solver。
       CANN 9.2 CPU 的 Compact 容量检查仍按 Normal 计算，测试使用私有头文件修正其
       边界公式为真实 `blockCount * blockLen`；Normal 检查与 DMA 执行不变，SDK 安装不变。
       `RegBaseSdkPreparation` UT 检查两方向的原有检查仍保留、SDK 刷新与重复配置稳定。

4. 输出

   一条或多条 `DataCopyPadExtendOp`（>2 段时外加 `scf.for`）；GM 侧带
   `GlobalTensorSubIndexOp(tileOffset)`。段几何判不了 → 能力层报不支持 → 整图回落
   legacy，不 emitError。

### 功能需求 2：`afir.pad` 的方言定义和 bridge 接入

1. 介绍

   让调度器吐出的 Pad 节点能进 IR。Pad 是 UB→UB 的布局补齐算子（把 32B 对齐布局整回
   尾轴真实长度，或反向），和 RemovePad 对偶。

2. 输入

   调度 `ScheduledResult` 里的 Pad 节点（dump 里见过的形态：`.y.axis = {z0TB, z0Tb,
   z0t, z2, z1}`，`.vectorized = {z0t:920, z2:40, z1:1}` 一类，`.tmp_buf = {{id=0,
   size=1024, life_cycle=-1}}`）。

3. 处理

   - 方言定义：`AFIR_PadOp : AFIR_BufferizableUnaryElementwiseOp<"pad",
     AFIR_RemovePadElementTypes>`（dtype 集与 RemovePad 同列 `[I16,UI16,I32,UI32,F16,F32]`；
     legacy `PadApiCall` 走 `BlkAlign<T>`，没有更宽的 dtype 通路）。校验规则与 RemovePad
     同构（两份 desc、元素数一致、tmp buffer 规则）。
   - bridge：注册表加一行 `M("Pad", afir::PadOp, BridgeOpForm::Both)`（`AfirBuilder.cpp`
     三处一致性由 `AfirBuilderTest` 锁住）；节点级 tiling 字段走既有
     `GetApiTilingInfo("Pad")`（`ScheduleToAfir.cpp:757-764` 已备好 PadTiling 25 字段
     清单，本需求不新增）。
   - `isTransferNodeType` 不收 Pad（UB→UB，同 RemovePad 的处理）。

4. 输出

   `afir.pad` 出现在 plan module 的 IR 里；`BridgeOperationRegistryTest` 名单 48→49，
   `AfirBuilderTest` 的三处一致性计数同步。

### 功能需求 3：`pad.ub_ub_extend` 下降实现

1. 介绍

   把 `afir.pad` 降成调用，legacy 对应 `PadApiCall::Generate`（`pad_api_call.cpp:26-88`）。

2. 输入

   op 的两份 desc + tmp buffer 操作数（`Transpose.cpp` 的 `tmp()` 先例）+ tiling 成员
   `t->{node}_tilingData_{id}`（PadTiling 结构，25 字段已由功能需求 2 的 func 级
   `api_tiling_fields` 声明）。

3. 处理

   legacy 是**运行期二选一**，移植为 `scf.if`：

   - 分支条件：`Size(vec_strides[n-2]) != ActualSize(last_axis_size)`——倒数第二根
     向量化轴的对齐跨度 ≠ 尾轴实际长度（`pad_api_call.cpp:64-69`），即"布局里真的有
     补位吗"；
   - 有补位：`AscendC::PadParams padParams = {0, (uint16_t)(Size(vec[n-2]) -
     ActualSize(last)), 0};` + `PadTiling apiTilingData = t->{field};` +
     `AscendC::Pad(dst[vecOffset], src[vecOffset], padParams, tmp_buf_{id},
     apiTilingData);`
   - 没补位：`AscendC::DataCopy(dst[vecOffset], src[vecOffset],
     BlkAlign<T>(axis_size_product));`（各向量化轴实际大小连乘，`pad_api_call.cpp:38-48`）；
   - 两侧向量化偏移的物化和功能需求 1 的偏移同一套设施，作用在向量化轴上；
   - 新增 AscendC 方言包装算子（`OpVecPad.td`，照 `OpVecRemovePad.td` 的结构）+ 打印
     函数；注册 `ApiExtensionHeader`（Pad 相关官方头按 legacy include 清单内联，
     `checkFamilyMatchesHeader` 键-文本一致校验锁住）。

4. 输出

   `select-implementation` 在 950/910B 上均选 `pad.ub_ub_extend`（或按能力层限定）；
   C++ 产物含 `scf.if` 二分支；950 侧只以 CPU E2E 仿真验证（无实机）。能力层判断标准：
   desc 形态可判定（两份向量化清单齐、tmp buffer 在）+ dtype 在集合内；判不了 →
   不支持（走既有 `no supported` 消息）。

### 功能需求 4：RemovePad 的整窗口 gather 形态

1. 介绍

   转置图（S1 的 C0/C1）把一种既有 RemovePad 下降没见过的形态推到了面前：TileInner
   轴把向量化区**切成 2 段**，且外段步长可证明等于内段步距 × 跨度。对这种形态，
   `Implementations/RemovePad.cpp` 增加 `gatherWholeWindow` 发射分支。

2. 判据与发射

   - 成立条件（可符号判定，判不了 → unsupported 回落）：外段步长 == 内段步距 ×
     外段跨度，即窗口在内存里是等距摆放的。
   - 发射：一条 gather 调用覆盖整个窗口，`repeatCount = 外段实际值 × 内段名义值`
     （对齐 legacy `is_last_axis_align` 口径）。
   - 与既有三形态（直拷 / 单段块拷 / 逐行块拷）并列，互斥判定。

3. 输出

   `afir.remove_pad` 在该形态下不再回落；`S1` 的 C0/C1 模板全绿。形态的合成 lit 是
   `afir-remove-pad-whole-window.mlir`（func 带 `afir.asc_graph` 触发合轴护栏才测得到）。

### 功能需求 5：allocator 回滚双重释放修复

1. 介绍

   调试时实际定位到的（gdb backtrace + 逐相断点）：`AFIRMemoryAllocate` 任何一相失败
   进 `rollback()`（`AFIRMemoryAllocate.cpp:98-109`）时，对**已经释放掉的 op** 再
   `erase()` 一次 → `double free or corruption` 进程崩溃，真正的错误信息被吞（表现为：
   失败时没有任何 `error:` 输出直接崩）。触发链：greedy 重写内置的 DCE 清掉了记录表里
   use 为空的 op，回滚不知道它已经没了。

2. 输入

   复现方法：任取一个多 subgraph 用例 loop-generate 之后的 IR +
   `af-opt --pass-pipeline="builtin.module(afir-memory-allocate)"`。

3. 处理

   定位被二次释放的具体来源（优先假设：DCE 与记录表的竞争），修复以**最小**为准：或让
   rollback 只 erase 还活着且在记录表里的 op，或让记录在 op 被删时同步移除。不重构
   allocator 分相结构。

4. 输出

   失败注入用例（lit：构造一相必失败的 IR）：进程不崩、`error:` 打印、退出码 1。

## 非功能需求

### 可维护性

- 新增一个 op 的三处改动（ODS、bridge 一行、implementation 注册）仍然成立；发扩展包装
  的 op 多出第四处（目标方言 def + 发射器），与 RemovePad/Nddma 同数。
- 跨段搬运的段几何物化、参数算式与发射集中在 `Implementations/DataMovement.cpp` /
  `TransferPlan.cpp`（分类在 `VectorSegmentation`），每处注释指向 legacy 的 file:line
  （先例：`RemovePad.cpp` 头注释）。
- `ApiExtensionHeader` 第三段列表给后续所有依赖官方头的 op 复用。
- 全部新代码不出现芯片名（`afir.target` 唯一事实源）。

### 可测试性

- lit 是判断主入口；`BridgeOperationRegistryTest` / `CodegenOrchestrationTest` /
  `ElementTypeMatrixTest` 表驱动自动收新行；CPU E2E 双平台现成。
- UT 覆盖合轴纯函数面（喂构造 desc）。
- CPU E2E 就是端到端验证入口，不需要新桩。

### 可移植性

- 无新平台分支。双档选择全部经 `ApiFamily` + TargetProfile；轴数上限与 `RangeType`
  宽度是 API 能力常量。

### 可靠性

- 所有"判不了"（表外 perm、dtype 不支持、tmp 缺失、索引映射缺失、段数物化失败、合轴
  判不了）一律 `unsupported` → 整图回落 legacy，不 emitError（在这里 emitError 是硬
  失败不是回落）。唯一例外：verifier 层的错误（scalar 输入、rank 不同）是构图期硬
  失败，不回落——它们在 legacy 也是构图错误。
- 运行期参数物化：搬运的 blockCount/blockLen 与两侧缝隙是运行期算术（缝隙为
  `SubI(行距, blockLen)`，GM 侧随实际尺寸、UB 侧随标称布局绑定），资源尺寸类走
  `materializeCheckedAdd` + 溢出断言。段几何的防御判据（UB 侧负缝隙、行距不整除
  ——load 严格、store 缝隙 0 豁免）在计划侧 `blockDmaSegmentGeometryIsSound` 拒，
  能力层与 pattern 共用同一份。
- rollback 修复后，任何一相失败都是干净报错 + 返回失败，不崩进程。

### 特性交叉影响

已按 [cross_feature_check.md](../../../../docs/guidelines/cross_feature_check.md)
逐项检查：

| 场景 | 适用性 | 分析说明 |
|------|--------|----------|
| SuperKernel Python 接口 | 不适用 | 不碰 `super_kernel/` 与 Python 包 |
| SuperKernel C++/AOT 接口 | 不适用 | 不碰 `libascendsk.so` / AOT / RDV |
| Autofuse 图优化 | 不触碰 | 调度侧零改动；Nddma 两侧布局口径沿用新基线，见 A7 复核 |
| Autofuse Codegen/Backend | 适用 | 本设计主体；legacy 的 `TransposeApiCall`/`TransposeRegApiCall`/`LoadApiCall`/`PadApiCall` 不动，两路并存 |
| AscendC API / Runtime 交互 | 适用 | 只改 kernel 内 AscendC 调用（ConfusionTranspose/TransposeExtend/DataCopyPadExtend/Pad/DataCopy），无 rt*/aclrt* 新调用；mem 档尾随 `PipeBarrier<PIPE_ALL>()` 与 legacy 逐字一致——按 `AGENTS.md` §3.5 教训**不可静默删**，NPU 上按产物级 A/B 验证 |
| Python/C++ 混合绑定 | 不适用 | 无接口外抛 |
| 构建与打包 | 不适用（轻） | 两个 CMake 源文件；不改安装路径与产物名 |
| 测试与覆盖率 | 适用 | lit 若干文件 + UT 三处计数 + CPU E2E + NPU E2E + probe；bug 修复带复现用例（功能需求 5）；a2/a3 上板实测，a5 仿真验收 |
| 性能与日志 | 适用 | 见「性能」；无新增默认日志 |
| 兼容性 | 适用 | 见「兼容性检查」 |

## 性能

### 编译时长

第一部分：每个 `afir.transpose` 上一次 perm 分析（O(轴数)）+ reg 档一次
`collapseVectorizedAxes` + O(段数) 数组物化，capability 与 pattern 各算一遍（×2），与
RemovePad/Nddma 同约定。第二部分：合轴对每个 load/store 一次 O(轴数≤5) 的符号比较。
对整图编译时长影响可忽略（<1%）。

### 执行性能

**a2/a3 上板 p50 不劣化是验收项**（方法见测试设计「NPU 用例」）。结构等价依据：同一条
包装调用、同参数算式、Pad 同二分支、外抛循环同偏移算式
（`Σ iv × stride`）——代码和 legacy 一致，结果与速度天然一致。风险点两个，都在 NPU
对拍里显式覆盖：① reg 档外抛的 `scf.for` 与 legacy `outer_for` 的循环边界/偏移是否
逐值等价；② mem 档 PipeBarrier 的取舍（本期**不省**，照发；如评估想省，必须按 §3.5
方法 A/B 实测）。跨步 block 形态相对"逐段单条拷贝"本就少指令（legacy 原选型），MLIR
不引入额外循环层级。reg 档（a5 无实机）不做性能对拍——CPU 仿真没有执行性能含义，
推迟到 a5 环境可用。

### 内存和产物大小

不新增常驻结构。tmp buffer 走既有 `b{id}_size` 聚合（reg `CalcTransposeTmpSizeV2` /
mem `CalcDefaultTmpSize` 都在 ATT 侧算，MLIR 只引用）。kernel 体积增量与 legacy 同图
产物相当。

## 接口设计

### 新增/修改接口描述

无对外接口（Python/C++/CLI 均不变）。内部新增：

- `afir.transpose`（含属性 `api_tiling_field?: StrAttr`——bridge 写、conversion 读）、
  `afir.pad`。
- `ascendc.transpose_extend_reg_base` / `ascendc.confusion_transpose`；
  `OpVecPad.td` 包装算子。
- `ApiExtensionHeader<headers, preludes, officials>` 第三段列表（TableGen 类扩展）。
- `registerTransposeImplementations(...)`；
  `Implementations/{Transpose,Pad,RemovePad,Nddma}.cpp`（Pad 为新文件，RemovePad/Nddma
  为形态扩展）、`Analysis/TransferPlan.*` / `VectorSegmentation.*`（段几何）。

### 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|--------|----------|----------|------|
| 接口说明 | 是否需要接口评审 | 否 | 无对外接口 |
| 接口说明 | 是否需要补充文档 | 是 | 本设计文档 + 覆盖矩阵 MLIR 状态列 + inventory csv + 矩阵 CSV Pad 行 |
| 接口兼容 | 行为是否兼容 | 是 | legacy 路径不变；既有单段 load/store 发射逐字不变（回归锁住）；MLIR 此前对含 Transpose 的图直接回落，现在多一条可走的路 |
| 接口兼容 | ABI/API 是否兼容 | 否 | 纯 codegen 内部，无导出符号变化 |
| 接口约束 | 约束不满足时是否清晰报错 | 是 | 全部走 `CapabilitySupport::unsupported`（理由码区分）；表外 perm 的诊断带上 perm 序列；合轴判不了 → 既有 `no supported` |
| 接口测试 | 是否需要独立接口用例 | 是 | lit 若干文件 + emitter lit |

## 软件设计

### 关键数据结构

```cpp
// 一个 afir.transpose 在 perm 分析之后的形态。
// 数组各记「哪根轴的哪一项」，capability 只判形态、pattern 才物化。
struct TransposeRegShape {
  unsigned source = 0;        // VectorPlan::tensors 下标
  unsigned destination = 0;
  SmallVector<unsigned> srcStrideAxes;  // 按输出轴序的输入轴下标（perm 本体）
  uint8_t innerDim = 0;       // 内层连续轴数，≤3
  uint8_t totalDim = 0;       // 参与调用的轴数，≤4（外层段走 VectorizedLoops）
  unsigned innerSegment = 0;  // 最内段下标（ActualSize 判断用）
};
struct TransposeMemShape {
  ascendc::TransposeType type;   // 7 个值之一
  StringRef tilingField;         // op 属性 api_tiling_field 原样
};
```

跨段搬运**没有**预先压缩的 plan 结构——段几何在下降时由
`materializeScheduledAxisExtents` 从 desc 逐轴符号物化，`emitScheduledTransfer` 直接消费
（见功能需求 1）。生命周期：上述形态结构全部函数局部，不进 IR、不跨 pass，能力层和
pattern 各判一次（幂等）。

### 关键技术/算法

第一部分：perm 分析（A5）、21 种枚举折叠（A6）、`VectorizedLoops` 外抛（A5）、官方头
第三段列表（A3）、apiTilingData 成员访问链（A6）——各自在对应小节展开。

第二部分：

1. **段几何物化**：见功能需求 1。和 legacy 的字段映射：

   | legacy | MLIR 侧对应物 |
   |--------|---------------|
   | `ub_tensor.vectorized_axis/strides` | UB 侧 desc 的 `vectorized_axis` / `vectorized_strides` |
   | `gm_tensor.axis_size/axis_strides`（按 axis id 对位） | GM 侧 desc 的 `repeats` / `strides` |
   | `tpipe.tiler.GetAxis(axis).type == TileInner` | `afir.asc_graph` axes 的 `axis_type` |
   | `StaticCheckEq`（af::Expression） | SymEngine 解析 + 结构相等（`SymbolicExpression`） |
   | `Tiler::Size / ActualSize` | desc 表达式已含对齐值 / `materializeTailAwareTileSize` 物化 |

2. **单位与 32B 对齐**：调用点单位一律元素，换算在包装里做（GM 侧 ×sizeof(T)，UB 侧
   ÷align_num，都是整数运算）。**UB 侧段起点间距**（desc 里的行距，如 S0 的
   `16*Ceil(z2t/16)`）必须是 align_num 整数倍——调度器的对齐布局已经保证；段几何对不
   满足的形态报不支持（防御，legacy 同样依赖调度保证）。注意"间距 − 段长"（实际传的
   dstStride，如 S0 满尾块 `32−23=9`）**可以**不是整数倍：`DataCopyPad` 会把每段自动
   补到 32B 边界，整数除法取整后恰好等于段间真实空隙。
3. **TilingData 头的外部类型表**（`TilingDataHeaderEmitter.cpp`）：api tiling 结构
   （`ConfusionTransposeTiling` / `PadTiling`）**只引用不定义**——成员声明 +
   `#include "kernel_tiling/kernel_tiling.h"` 打头，否则与官方头同 TU 重定义（legacy
   生成头 `codegen_tiling_data.cpp` 同款做法）。
4. **Pad 二分支**：条件与实参见功能需求 3；`uint16_t` 截断点（padParams 第二成员）物化为
   `arith.trunci` + `cf.assert` 值域检查。

### 流程设计

第一部分主流程（两档共用前半）：`queryLowering` → `hasCanonicalDescriptors` →
`checkApiFamily` → `checkTmpBuffer(true)` → 操作数划分 → 元素类型 → 档内判断 →
`supported()`。pattern：同一串判断重算 → `orderTilesForPlan` → 两 tile 各发偏移 0 的
`LocalTensorSubIndexOp` → tmp 物化 →（reg：外层段 `scf.for` + 三组数组 / mem：枚举 +
struct 值）→ 包装 op。

第二部分主流程：能力层查功能需求 1/3 的判断 → pattern 发 IR → 既有
`GenTilingData`/`GenKernelEntry` 不变。

异常流程（共用）：任一步失败 → `notifyMatchFailure`，dialect conversion 回滚本 pattern，
整图由 `AFIRSelectImplementation` 汇总后回落 legacy（本路径 E2E 内不回落）。功能需求 5
修复后 allocator 失败路径 = 干净报错 + 返回失败。

### 对子模块的修改

| 目录 | 修改点 |
|---|---|
| `dialect/AFIR/IR/Ops/Layout.td` | `AFIR_TransposeOp` / `AFIR_PadOp` |
| `bridge/AfirBuilder/AfirBuilder.cpp` | 注册表两行 + `isTransferNodeType` 一词 + `api_tiling_field` 挂载 |
| `bridge/ScheduleToAfir/ScheduleToAfir.cpp` | `GetApiTilingInfo` 字段名透传（清洗跟随 `GenValidName`） |
| `api/ascendc/.../Autofuse/OpVecTranspose.td` + `OpVecPad.td` + `Base.td` | 包装算子 def；`ApiExtensionHeader` 第三段列表 |
| `api/ascendc/lib/Target/AscendC/Autofuse/Transpose.cpp`（及 Pad 打印） | 手写 printer |
| `artifact/KernelSource/AddKernelIncludes.cpp` | `officialHeaders` 收集与发射；复用基线 `ApiExtensionOrder` 声明链拓扑排序 |
| `artifact/KernelSource/Emit/TilingDataHeaderEmitter.cpp` | 外部类型表（api tiling 结构只引用不定义） |
| `conversion/AFIRToAscendC/Implementations/Transpose.{h,cpp}` | 两档实现 + 注册 |
| `conversion/AFIRToAscendC/Implementations/Pad.{h,cpp}` | `pad.ub_ub_extend` |
| `conversion/AFIRToAscendC/Implementations/RemovePad.cpp` | 整窗口 gather 形态（功能需求 4） |
| `conversion/AFIRToAscendC/Implementations/Nddma.cpp` | 沿用新基线，无本特性改动；自洽判据已由上游删除 |
| `conversion/AFIRToAscendC/MemoryAllocate/AFIRMemoryAllocate.cpp` | rollback 双重释放修复（功能需求 5） |
| `dialect/AFIR/Analysis/{TransferPlan,VectorSegmentation}.*` | 段几何物化/分类（能力层与 pattern 共用） |
| `bridge/AttSchemaAdapter/` | `ExtractReferencedFieldNames` 只认 get_/set_ 后紧跟成员访问——裸 lambda（如 `set_semantic_fallback()`）不再误判为字段访问器 |
| `test/e2e/cpu/cases/load_transpose_store{,_2d}/` + `AddCpuCase.cmake`（HOST_TILING_API 关键字） | e2e 用例与框架接入 |
| `test/e2e/npu/load_transpose_add_store.py` | a2/a3 上板：legacy/MLIR 双路径 codegen + 直调 + 数值比对 + p50 计时 |
| `docs/guides/ascir_op_lowering_matrix.csv` | Pad 行 + load/store 跨步形态注记 |
| lit 7 文件 + UT 计数 + probe case + inventory csv | 配套测试 |

### 错误处理

系统错误：物化失败（表达式无法求值）→ `notifyMatchFailure` → `no supported` 分类，
不新增错误码。接口错误：能力不足统一走既有 `no supported Ascend implementation` 消息
与回落分类（`CapabilityShortfall` 标记机制）；表外 perm 的诊断带 perm 序列。verifier
层错误（scalar 输入、rank 不同）是构图期硬失败，不回落。功能需求 5 修复后 allocator 相失败打印
该相的 MLIR `error:`（`emitError` 已有，只是此前被崩溃吞掉）。

# 安全检查

已读取并检查 [编码红线.md](../../../../docs/guidelines/编码红线.md)：

- **硬编码敏感信息/公网地址**：无。新增字面量：轴数上限 3/4（API 模板特化面）、21 种
  perm 表（legacy `kPermutationTable` 逐字搬移，属算法数据非敏感值）、官方头名。
- **外部输入作索引/长度/偏移**：perm 分析的下标全部来自 desc `axis` 数组，取用前查
   长度与越界；blockCount/blockLen/strides 全部来自图 desc 的符号表达式，物化带
   `cf.assert` 溢出/值域检查（功能需求 1/3）；窗口段数与 Variadic 一一对应
   （AttrSizedOperandSegments 钉住段数一致）；无用户裸指针输入。
- **整数溢出/反转/除 0**：`materializeCheckedAdd/Mul` 先例沿用；`RangeType` 宽度判别只
   比较 `sizeof(T)` 不发除法；`padParams` 截断点显式 `trunci`+值域断言；符号层的
   有理系数除法**只允许 Ceiling/Floor 包住的 Mul**（精确下降为 `CeilDivU(p*x/q …)`，
   如 `Ceiling(Rational(1,16)*x)` → `ceildivu(x, 16)`），裸 Mul 里的 Rational 仍拒
   （index profile 精确性语义）；tmp 尺寸不进 AFIR（引用 `b{id}_size`）。
- **资源释放覆盖异常分支**：功能需求 5 即本项（rollback 双重释放）；tmp buffer 生命周期走
   TPipe 既有机制，pattern 不自管内存、不持资源（pattern 失败由 conversion 回滚）。
- **内存申请前判大小/申请后校验**：不新增裸申请。
- **未定义行为**：功能需求 5 修复消除了对已释放 `Operation*` 的解引用（UB）。
- **ABI/API 兼容**：无导出符号变化。
- **无理由扩大修改范围**：单段路径逐字不变；调度/ATT/Python 零触碰。
- **图改写确定性**：不改 AscGraph，调度与 ATT（`v35/`、`att/`）零触碰；段几何物化
   无迭代序依赖；`AddKernelIncludes` 的官方头集合确定序（声明链拓扑 + 字典序平局）。
- **高频路径日志**：不新增任何 GELOG。
- **runtime/aclrt/AscendC 生命周期**：只改 kernel 内 AscendC 调用，与 legacy 同形态；
  tmp buffer 是 `life_time_axis_id=-1` 的全生命周期 TBuf，`CanonicalGroupKey` 已按 -1
  分组，无跨 op 新依赖。
- **构建增量/离线**：仅新增源文件与 .td，走既有目录。

# 兼容性检查

- 不影响脚本参数、配置、安装目录、动态库名、Python 包与历史产物。legacy codegen 路径
  一个字不改（`TransposeApiCall` / `TransposeRegApiCall` / `LoadApiCall` / `PadApiCall`
  原样），两路并存。
- 调度与 ATT（`v35/`、`att/`）零触碰：legacy 调度/tiling 行为与合入前逐字一致，
   全量回归验证。
- 既有 E2E（nddma/broadcast/elementwise 全族）：单段 load/store 发射逐字不变，全量
  回归锁住。
- 既有 lit：`afir-load-*`/`afir-store-*` 系列的拒绝消息是通用 `no supported`，跨段形态
  放开后无断言漂移（全量回归绿）。
- `ascir_op_lowering_matrix.csv`：新增 Pad 行；load/store 行补跨步形态注记（不改既有格）。
- 官方头机制对无声明 op 零影响（`officialHeaders` 为空时 include 块一字节不变）。

# 测试设计

## 测试边界

**入口**：lit（`af-opt --target=... --afir-...pipeline`）、UT 二进制（gtest）、CPU E2E
generator（`Optimizer::Optimize → GenerateForInductor`，断言 kernel 名含
`<case>_subgraph0_plan0` 防 legacy 静默兜底）、a2/a3 实机（`AGENTS.md` 3.4 直调链路 /
`test/e2e/npu/` 脚本）。**出口**：IR/生成 C++ 文本断言、CPU 数值比对、NPU 数值比对与
p50 计时。**打桩**：CPU E2E 用 tikicpulib（按平台分库，`AddCpuCase.cmake` 已管）；真机
走真实 tiling 库，legacy 对照用裸 `AutofuserOptions()`（见 3.4 坑清单）。**回落语义
注意**：CPU E2E harness 的 tiling 库是恒 false 的 stub（`support/TilingStub.cpp`）——
该环境里没有可用的 legacy 回落，任何 MLIR 收不了的图在 harness 里就是红（真实产品路径
的回落由 NPU 对拍与能力 probe 验证）。**验证数据**：S0/S1 全部 5 个调度模板的 dump
形态（`AUTOFUSE_DFX_FLAGS='codegen_compile_debug=true;debug_dir=...'` dump）作为构造
用例的事实源。

**E2E 形状纪律**（`AGENTS.md` §3.1/§3.3 教训的强制项，两部分共同遵守）：

- **大 shape 必配**（元素数 ≳ 数十万，探 `a0Tb_size > 1` 的块内偏移与多趟 repeat 类
  bug——小 shape 用例对这类 bug 全盲）。
- 覆盖对齐/非对齐尾轴、2D/3D/4D/5D、多种 perm。
- **选形状要避开调度消除路线**：非尾轴转置且尾轴 >512B 时调度会把 Transpose 折进
  Load/NDDMA（`transpose_schedule_case_generator.cpp:26` 阈值），图上不再有 Transpose
  节点——主用例选尾轴转置（score=1 保留）；消除路线不依赖 e2e 数值覆盖：判据由
  `afir-nddma-reg-base.mlir`（随本设计）、发射由既有 `ascendc-data-copy-nddma-reg-base.mlir`
  锚定；reg 档现在另有独立数值 E2E（见验收第 9 条），文本中的候选枚举不等于每个候选均执行。

## 测试用例设计

第一部分用例（回归基线）：

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|----------|------------|----------|----------|
| 功能 | reg 档 2D/3D/4D 选中 `transpose.ub_ub_reg_base`，数组内容/模板实参正确 | `afir-transpose-reg-base.mlir`（IR+CPP+SELECT） | lit |
| 功能 | >4 维外抛：外层 `scf.for` + 最内 4 维调用 + `Σ iv×stride` 偏移 | `afir-transpose-outer-loop.mlir` | lit |
| 功能 | mem 档 3 个代表枚举选中 `transpose.ub_ub_extend`，`tiling.<field>` 传参 + PipeBarrier | `afir-transpose-mem-extend.mlir` | lit |
| 功能 | emitter 文本与 include 行（扩展头+官方头） | `test/lit/target/ascendc-transpose-*.mlir` | lit |
| 功能 | bf16（reg 收）与 5D 外抛 | `afir-transpose-reg-base-bf16.mlir` 等 | lit |
| 异常 | 表外 perm（5 维形态）/ dtype 不支持 / tmp 缺失 / 索引映射缺失 | `afir-transpose-outer-loop.mlir` 910B `not` run；`afir-transpose-unsupported.mlir`（`NOSUPPORT-COUNT-3`） | lit |
| 异常 | scalar 输入 / rank 不同（verifier 层） | `afir-transpose-invalid.mlir`（`INVALID-COUNT-2`） | lit |
| 兼容性 | bridge 白名单 ↔ conversion 覆盖集合相等；头名单；dtype 集合快照 | `BridgeOperationRegistryTest` / `CodegenOrchestrationTest` / `ElementTypeMatrixTest` | UT |

第二部分用例：

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|----------|------------|----------|----------|
| 功能 | 段几何端到端数值（1/2 段 load、2/3 段 store、整窗 gather 的 RemovePad 链） | CPU E2E：`load_transpose_store_2x37x23` / `4x16x32`（mem）、`load_transpose_store_2d_64x64`；以上三格均配有 `reg_base` 数值版本 | E2E |
| 功能 | >2 段外层循环 | 同上（2x37x23 的 store=3 段） | E2E |
| 功能 | Pad 二分支（有/无补位） | `afir-pad-ub-ub.mlir`（SELECT/IR/CPP/950 四面） | lit |
| 功能 | RemovePad 形态（直拷/单段块拷/逐行/整窗口 gather） | `afir-remove-pad{,-symbolic-pitch,-unsupported,-whole-window}.mlir`（整窗口档需 func 带 asc_graph 触发合轴护栏） | lit |
| 功能 | NPU 数值等价：MLIR vs legacy 同图逐元素比对（纯搬运类，预期 bit 级一致，有差异即算缺陷） | a2/a3 上板：`load_transpose_add_store.py` 双路径同输入，aclrt 搬回逐元素比对 | NPU |
| 性能 | NPU p50 不劣化：同图 legacy vs MLIR | 同上：launcher 逐次同步计时（≥500 iters/轮 × 多轮），同轮交替；p50(MLIR) ≤ p50(legacy)×1.05 | NPU |
| 异常 | 段几何判不了（UB 侧负缝隙/不整除）→ 干净回落 | `no supported` 消息 + 能力 probe 的 DRIFT 桶 = 0；`afir-transfer-segment-geometry-unsupported.mlir` 三分片（load 负跨度 / load 不整除 / store 不整除） | lit/probe |
| 异常 | allocator 失败注入 | lit：一相必失败 IR，进程不崩 + `error:` 打印 | lit |
| 兼容性 | 既有单段路径不变 | 全量 lit + 全量 e2e 回归 | lit/E2E |
| 特性交叉 | 注册表/头名单/dtype 集合快照 | `BridgeOperationRegistryTest` / `CodegenOrchestrationTest` / `ElementTypeMatrixTest` / `AfirBuilderTest` | UT |

**已补齐的覆盖欠账**：跨段 load/store 的合成参数断言 lit（`afir-load-store-strided-block.mlir`，
真实 dump 语料）、RemovePad 整窗口 gather 的合成 lit（`afir-remove-pad-whole-window.mlir`）、
Nddma 刻度差的合成 lit（`afir-nddma-reg-base.mlir` 的 `@scaled_stride_tile` 正例；
旧 ZEROFLIP 负例随上游自洽判据移除而删除）。段几何防御判据的负向 lit
（`afir-transfer-segment-geometry-unsupported.mlir`）与判据实现
（`blockDmaSegmentGeometryIsSound`）一并落地——登记时误以为判据已存在，实际
「负跨度在合轴层就拒」从未实现。判据以 UB 侧块推进模型
`AlignUp_g(行长) + g·trunc((行距−行长)/g)`（g = 32B/元素，探针实证）为准：推进恰等于
名义行距 ⟺ 行距 ≡ 0 (mod g)；只判 UB 侧（GM 侧负缝隙是转置读写的合法上线形态）、
方向分治——load 严判行距整除（布局由本 DMA 落出），store 对缝隙 0 形态放行（物理
行距由生产者决定，调度器会为未选中候选发射这种形态，须编得过的死代码）。

## 测试命令

```bash
# MLIR 全量 UT/ST/lit（容器或本地，见 autofuse/mlir/README.md）
bash scripts/test/run_autofuse_test.sh -u -s -m mlir -j 8

# lit（注意指向构建目录的 site 配置，不是源码目录）
llvm-lit -j 8 <build>/autofuse/mlir/test/lit

# e2e CPU 全量（含转置 4 用例）；单跑转置用例
ctest --test-dir autofuse/mlir/test/e2e -j 8
ctest --test-dir autofuse/mlir/test/e2e -R load_transpose_store -j 8

# UT / ST
ctest --test-dir autofuse/mlir/test/ut -j 8
ctest --test-dir autofuse/mlir/test/st -j 8

# legacy（v35 调度）回归——本设计零触碰调度/ATT，此条为兜底确认
sh build.sh -u --module=autofuse_framework -j 8

# a2/a3 上板（npussh 连接后；RUN_DIR 按环境准则 2.4.1，临时产物不放源码目录）
python3 autofuse/mlir/test/e2e/npu/load_transpose_add_store.py   # 数值等价 + p50
```

上板执行约定（`AGENTS.md` 3.4/3.5）：MLIR 路径显式传 `soc_version`（910B2/910B3）与
CPU e2e 的空 tiling stub；legacy 对照用裸 `AutofuserOptions()`；`ResLimit{aiv, ub_size}`
按板型取值；计时用 launcher 逐次同步、多轮交替，先数值后性能；真机按 eager → legacy
→ MLIR 顺序分阶段验证（2.4.8）。

# 验收标准

第一部分：

1. 双档 lit 覆盖第一部分用例表全部功能/异常项；UT 计数与名单同步（47→48）。
2. dtype 口径兑现：MLIR capability 与登记的可建模投影一致。mem 支持 f16/f32/i16/ui16；
   reg 再支持 i32/ui32/bf16。i64/bool 由 ODS 明确拒收。
   **mem 档有意摘除 i32/ui32**——根因是 legacy 布局（`SetAlignWidth` 只给 DT_FLOAT 的
   transpose 提 64B，4 字节整型折 8 元素）与 ConfusionTranspose 公式（写死尾轴 16
   元素）错位，legacy 生产同错（实测证实）。reg 档不受影响（regbase 显式传 strides）。
   落点：`Implementations/Transpose.cpp` 的「口径 A 修订」注释、`ElementTypeMatrixTest`
   的 `kKnownNarrower`（`transpose.ub_ub_extend`）、`afir-transpose-unsupported.mlir`
   的 `@i32_tile` 分片。恢复条件：legacy 布局/公式口径对齐后
   `kTransposeMemElementTypes` 加回 kI32、删 kKnownNarrower 行、恢复 e2e 用例。
3. nddma：沿用新基线实现，`Nddma.cpp` 无本特性改动；lit `afir-nddma-reg-base.mlir`
   保留刻度差正向覆盖。基线已启用的 `load_broadcast_nddma_store` 属 broadcast 特性
   用例，不属本特性新增验收。

第二部分：

4. transpose CPU E2E：3D 对齐/非对齐、2D f16、Sub→Transpose→Abs 两形状、
   Transpose→Add 两形状，共 7 个 mem 与 7 个 reg 数值格；全部必须实际注册并逐元素通过。
5. 全量 lit / UT / ST 回归通过（含新名单计数 48→49）。
6. 既有 E2E 全量回归绿（单段路径零变化）。
7. allocator 失败注入用例：不崩、报错打印、退出码 1。
8. **a2/a3 上板（mem 档；下列数据为历史基线，当前补丁仍待复验）**：同图 MLIR vs legacy 数值逐元素一致（bit 级，纯搬运类）；
   p50 不劣化（同轮交替测量，劣化超 5% 不通过）；PipeBarrier A/B 结论落档。
   - 范围：a2 豁免（与 a3 同代码，仅 910B3 实测）。
   - **数值**：legacy / MLIR 两路 × 2 形状（2x37x23、4x16x32）4/4 bit 级一致；接管
     证据独立核实（kernel 名 / dlsym `AutofuseLaunch` 执行链），非 fallback 假绿。
   - **p50**（launcher 逐次同步，3 轮 × 500 iters，best-of-3 口径）：
     (2,37,23) MLIR/legacy = **0.986**、(4,16,32) = **1.000**——均 ≤1.05 达标。
   - **PipeBarrier 结论**：照发（与 legacy 逐字一致）；实测带屏障不构成性能劣势，
     「省屏障」变体收益评估不在本设计（如要做须按 `AGENTS.md` §3.5 方法 A/B 实测）。
9. **a5/reg 验收分层**：CPU 7 格数值已通过；a5 实机验证仍待设备，不将两者混为一谈。
   原“legacy 没有同形状用例所以注释 reg”的理由撤销：合法输入图的泛化不能仅因缺少旧用例被排除。
   dtype 验收仍独立：本轮数值覆盖 f32/f16，不扩张为 bool/i64/bf16 数值均已通过。
10. 矩阵 CSV、`pyasc_ascendc_op_inventory.csv`、本文档同步。

**旧基线落地时全量回归快照**（不代表 rebase 到 `4aa77dd6` 后的验证结果；新基线重建及回归进行中）：e2e CPU 125/125（当时 reg 数值尚未注册，当前状态见验收第 9 条）、
lit 269/269、UT 13/13、ST 1/1、能力 probe
通过（DRIFT 桶 = 0）、NPU e2e 4/4 bit 级 + p50 0.986/1.000。

# 不做的事（以及为什么）

1. **跨段搬运不做结构改写**：不合并排布、不提循环不变量——legacy 没有的优化一律不加，
   对拍等价优先。
2. **`Ub2ub`**：sched 阶段内部节点，不对 py 构图外显（`ascir_builtin_ops_v1.cpp:642`
   注释），维持不接。
3. **调度侧"消除路线"（group_transpose）的取舍评判**：调度生成什么我们降什么，不在
   本设计内评判该路线的收益。放行条件：将来 ATT 迁移时，"保留 vs 消除"的打分函数
   （`v35` 侧 `GenerateScoreFuncForUbReorder` 已置空）随迁。
4. **rollback 之外的 allocator 重构**：只修双重释放这一处实际缺陷。
5. **Pad 的 dtype 扩表**：与 RemovePad 同列，legacy 没有的 dtype 不接。
6. **a5/950 上板对拍**：无实机，上板验证推迟（见验收第 9 条）；reg CPU 数值必须验证，
   包装选择已通过同图、同形状的 mem/reg 仿真。
7. **transpose 融合族的其它图**（elementwise 链融合、多 transpose、broadcast 交互、
   Reduce 冲突形态）：本设计只做 `load_transpose_store` 这块试金石；扩展步骤见
   「泛化路线」节。
8. **`DT_BOOL` 不接；i64 也属于未接入边界**：reg 侧注册表有 BOOL，但 legacy e2e
    `transpose_bool_test` 在 CPU 仿真 `GTEST_SKIP`（DT_BOOL 指令不支持 CPU 仿真）；
   MLIR 侧 **I1 元素类型通路不存在**（AFIR ODS 无 I1、ascendc `Types.td` 无 Bool、
   `memref<i1>` 在整条下降链上无先例）。补它 = 跨三层建 I1 通道，为一个 CPU 仿真都
   验不了的 dtype 不成比例。放行条件：布局族真的出现 bool tensor 端到端诉求时，作为
   独立小需求先建 I1 通道。
9. **mem 档枚举表外 perm 不做"拆两次调用"的等价改写**：技术上可把表外 perm 分解成
   两条表内转置级联，但 legacy 在这里直接失败，MLIR 报 unsupported 回落已严格不劣于
   legacy；级联改写改变 kernel 结构，性能不劣化验收反而说不清。放行条件：先有一条
   "级联比 legacy 整图回落更快"的用例。
10. **`GenNddmaNode` 属性完整性缺陷另立修复，修复也不改 `AddTransposeNodeAfter` 的
    插入行为**：插 Transpose 节点是调度的既定行为（NDDMA 广播布局的补偿），MLIR 侧
    接住它即可，不改插入条件与位置。

# 泛化路线（transpose 融合族扩展，后续工作）

`TransposeFusionCaseGenerator` 是**通用图模式**（`transpose_schedule_case_generator.cpp:104-114`
注释，即 P0 事实）：transpose 可与任意上游 elementwise 链同 kernel 融合（消除路线对
整条前驱路径 `ApplyTensorAxisReorder`，把转置折进 load 的取数图样）；单个 transpose
生成保留/消除双模板打分选取（场景 1/3 = 尾轴转置或尾轴 ≤512B → 保留 + UB 重排；
场景 2 = 非尾轴转置且尾轴 ≥512B → 消除、无 UB 重排）；多个 transpose 不生成保留模板、
全部推到 load；存在 Reduce 时在原图直接消除全部 transpose。本设计完成后**节点层能力
闭环**（转置本体 + 跨步 load/store + RemovePad/Pad + elementwise 链均有 MLIR 下降），
后续扩展预期主要是"能力已备、形态待验"，按下列步骤走。

## 扩展步骤（按风险递增）

1. **P0 前置**：本设计全部完成（含 a2/a3 上板）。**已达成**（e2e CPU 全量 + NPU 板
   bit 级 + p50 不劣于 legacy）。
2. **P1 消除路线·无补齐形态**（场景 2）：全对齐 3D 用例（如 f16 `(32,64,128)` 非尾轴
   转置）——比 group_transpose 更纯粹的跨步折叠（没有 Pad），回归验证段几何物化的
   一般性。
3. **P2 elementwise 链融合**：`load_cast_transpose_store`（单目先行）→
   `load_transpose_add_store`（双输入 tile 的 desc 形态）→ 确认消除路线中 elementwise
   **轴序重排后**的 desc 形态。
4. **P3 多 transpose**：级联两个 transpose，验证"全推到 load"的折叠形态。
5. **P4 交互形态**：broadcast + transpose（广播轴消隐判断
   `ShouldIgnoreDataCopyZeroAxis` 的实战形态）；Reduce 冲突（原图消除全部 transpose
   的整图形态）。
6. **P5 上板**：P1-P4 每类挑代表图，在 a2/a3 走上板同套标准（bit 级等价 + p50）。

## 每图固定循环（方法论）

新增 CPU E2E 用例 → 生成器 + `AUTOFUSE_DFX_FLAGS` dump 调度 → 与 legacy 同图产物对照
（参数逐字比对）→ 有缺口按本设计同流程补（dump 定位 → legacy 对齐 → 需求条目 +
lit/UT）→ 数值绿 → 新形态补进 lit 合成用例矩阵。**事实先行，不硬按预期实现**。

## 泛化性工程要求（"做好泛化"的验收口径）

- 跨段搬运能力必须对**任意** desc 形态工作，禁止只对已知 dump 形态写特判：lit 覆盖合成
  形态矩阵——1/2/3/4 段合轴、零跨度、广播轴消隐、尾轴虚拟段补齐、UB 侧负缝隙拒绝
  （行距 < 行长）、UB 侧行距不整除拒绝（load 严格、store 缝隙 0 豁免；GM 侧负缝隙
  是转置读写的合法上线形态，不判）。
- 能力层和 pattern 共用同一套判断函数（由 `VectorSegmentation` 分类承接，"能力层说
  支持、pattern 失败 = 整图直接失败"的教训，后续图沿用）。
- 每个新 E2E 图踩出的新形态必须进 lit/UT 合成矩阵，"这个 E2E 绿了"不等于"形态覆盖
  了"——用例要表达通用规则，不复刻单个 dump。

**覆盖欠账已补齐**：跨段 load/store 的合成参数断言 lit（`afir-load-store-strided-block.mlir`）、
RemovePad 整窗口 gather 形态的合成 lit（`afir-remove-pad-whole-window.mlir`，func 带
`afir.asc_graph` 才能触发合轴护栏）、Nddma 刻度差正向合成 lit
（`afir-nddma-reg-base.mlir`；旧零翻转负例随上游自洽判据移除而删除）。
段几何防御判据与其负向 lit（`afir-transfer-segment-geometry-unsupported.mlir`，三分片）
同批落地——判据此前从未实现，推进模型与方向分治的依据见测试节的欠账段。

后续泛化缺口的原因分析、修复方案与验证计划集中在[泛化缺口修复设计](2026-11-02-afir-transpose-generalization-fix-design.md)。该方案待实现，不改变本文已验收的 Inductor 迁移结论。
