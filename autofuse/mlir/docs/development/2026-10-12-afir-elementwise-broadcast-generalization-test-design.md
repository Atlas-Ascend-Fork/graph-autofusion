# elementwise / broadcast 泛化用例：测试设计

**日期**：2026-10-12
**前序**：`2026-09-28-afir-vectorized-axis-collapse-design.md`（分段统一、探针与「泛化重量」两节）、
`2026-09-23-afir-broadcast-design.md`、`2026-09-30-afir-remove-pad-design.md`
**本次基线实测**：2026-10-12，`af-capability-probe` 两侧各跑一遍，数据见「整体介绍」

---

# 简介

## 目的

给已接通的 elementwise / broadcast op 设计一份**按因子正交展开的用例集**，使「这类 op 具备泛化能力」
从一句定性判断变成一张可复算、可比对的清单：每一格要么有数值证据（E2E），要么有结构证据（lit / UT），
要么被记成**带解锁条件的拒收**（探针清单），不允许存在第四种状态「没人量过」。

因子按用户点名的五项展开：多轴、对齐 / 非对齐、间隔 / 连续广播、多节点混合级联、以及已知尚未实现的
`afir.nddma` 场景。

目标读者：Autofuse MLIR codegen 开发与检视者。

## 范围

**本次做**（全部在 `autofuse/mlir/test/` 内，不改产品代码）：

- `test/probe/CapabilityBoundaryProbe.cpp`：加**图拓扑**维（**已落地 2026-10-12**，结果见功能需求 6）、
  加**签到基线比对**（今天它只打表、不断言）、诊断行带上**命中的判据标签**；
- `test/e2e/cpu/`：`add_mlir_e2e_cpu_case` 加 `DTYPE` 参数并把 `KernelTest` 的元素类型参数化，
  新增一个按因子取代表格的级联用例目录；
- `test/lit/conversion/`：给「相邻广播段」与级联形态各补结构用例（相邻那一档 2026-10-11 已收，今天只有一份 lit）；
- 清单文件：把探针产物的期望状态签进版本库，作为泛化进度的棘轮。

**本次不做**，各有独立理由：

| 不做 | 理由 |
|---|---|
| 修 `GenNddmaNode` 漏抄 `strides` | 它是**调度侧**一行改动，且会同时改变 legacy 自己的生成结果，要按 v35 的 ST 套件单独验。见「功能需求 5」N1，用例已备好、挂起等它 |
| 换 mem 侧两段窗口的 API 形态（收尾轴不对齐） | 填充在 `BroadcastFirstDim` 的 `dst_inner_offset` 上没地方表达，要改 `autofuse/ascendc/api/broadcast.h`。独立 PR（与 RemovePad 的 P3 截断是同一处改动的两面） |
| 放开 `broadcast.h` 整块复制那一趟的 counter 上限 | 同上，包装侧改动。本设计把它当**已知硬上限**写进矩阵（F7），不当盲区 |
| 运行期（符号）尺寸维 | 是真因子（`provablyEqual` 证不了就封段，方向是多一层循环而不是算错），但它牵的是 `afir.symbolic_eval` / `tiling_param` 那条线，与本次五个因子正交。留作下一轮，构造法记在「测试边界」末尾 |
| camodel 性能验证 | 第二条并行的构建 + 运行路径，独立 PR（前序文档「性能」已声明） |
| reduce / transpose / gather / concat / split | 段语义各不相同，每个 op 单独分析。本设计的因子表对它们不成立 |

# 总体概述

## 软件概述

### 项目介绍

Autofuse MLIR 侧的 elementwise 族已全量接通两侧（bridge registry 46 行全 `Both`），broadcast 有三档
实现：常量源 `Duplicate`、mem 侧分段 `Segmented`、reg 侧 `Extend`。前序文档用一台量具
（`af-capability-probe`，380 张广播图 + 13 张 elementwise 对照）量出「哪一格支持」，结论是
**完全可用的只有一格**：尾轴 32B 对齐 且 最内轴不是广播轴。

本设计不改这些实现，只回答一个问题：**把用例集按因子铺开之后，泛化缺口分别落在哪里、各由什么挡着。**

### 产品环境介绍

- 下游：`libaihac_codegen.so` 的 `RunMlirCodegen`，由 `AF_ENABLE_MLIR_CODEGEN=1` 开关；
- CPU E2E 用 CANN 的 tikicpulib 模拟器按芯片分库链接，不需要 NPU；
- 两侧平台由 `soc_version` + **NpuArch** 两处同时决定（少一处则调度与 MLIR 两边对不上），
  数值取自 `target/Ascend/Profiles/platform_config/`；
- 不涉及 Python 包、run 包与对外接口。

## 软件功能

1. 一张**因子 → 判据**对照表：每个因子取值能跨过哪条具体判据（带 file:line），不含「看起来更丰富」这种理由；
2. 三层载体各答一个问题，不互相代劳：探针答「哪一格支持」、lit 答「发出来的是什么」、E2E 答「算得对不对」；
3. 探针从量具升级为**带签到基线的清单**：硬失败 / 漂移恒为 0，期望状态变化必须显式改基线；
4. `afir.nddma` 已知缺口逐条入册，每条带**解锁条件**和已备好的用例。

## 设计约束

- 探针不接任何 `check-*` 目标（它跑满两侧要 ~3 分钟），棘轮比对作为独立目标手动 / 定点跑；
- CPU E2E 每格是**一个生成器可执行文件 + 一次 kernel 编译**，格数直接换成构建时长，不能按因子全叉乘；
- 构建并行度 `-j 6`（本机 `-j 8` 会被 OOM killer 杀掉，表现为 exit 137 且无 error 行）；
- lit 的最小 func 过不了 kernel ABI 检查，conversion 里新建 afir op 会让整个 pattern 回滚
  （报在 root op 上），所以级联形态在 lit 里只能做到「固定 descriptor 的两个 op 相邻」，
  真正的级联证据只能来自探针与 E2E；
- 不新增硬编码芯片类型判断：平台维通过 `PLATFORM reg_base` / `--reg-base` 选参走既有分流。

## 假设和依赖关系

- 探针与 CPU E2E 生成器走**完全相同的两步**（`Optimizer::Optimize` → `RunMlirCodegen`），
  所以探针报 `ok` 的格在 E2E 里也走 MLIR 路径。这条是既有事实，不是本设计新引入的假设；
- `LD_LIBRARY_PATH` 必须与 CPU E2E 用的那一份一致，否则在 `ScheduleUtils::ClearAllSizeVar` 段错误
  （第一张图就挂，看起来像建图错）；
- 「报不支持就回落 legacy」在真实图上成立（2026-10-09 已修并实测：两侧 773 张图 `kFailed` 0 次）。
  本设计的棘轮把这条当不变量钉住。

# 需求分析与设计

## 整体介绍

**2026-10-12 实测基线**（`af-capability-probe`，380 张广播图 + 13 张 elementwise 对照，各侧一遍）：

| 侧 | ok | unsupported | kFailed | DRIFT 桶 | elementwise 对照 | 墙上时间 |
|---|---|---|---|---|---|---|
| mem（Ascend910B1） | **125** | 255 | 0 | 0 | 13 / 13 | 89.8 s |
| reg（Ascend950PR_9599） | **0** | 380 | 0 | 0 | 13 / 13 | 86.2 s |

mem 侧按 dtype：f32 41 / 95、i32 41 / 95、f16 32 / 95、**ui8 11 / 95**。

reg 侧那 380 条拒收**拆成两处，而且拆得很干净**：

- **326 条拒在 `afir.nddma`**；
- **54 条拒在 `afir.broadcast`，而且全部是 ui8**；非 ui8 的图**一张都到不了 broadcast**（0 条）。

这比前序文档「reg 侧 48 / 48 全拒且一张都走不到 broadcast」更精确，并给出一条本设计据以排序的结论：

> **N1（调度侧 `GenNddmaNode` 漏抄 `strides`）是 reg 侧一切的单点前置。** reg 侧那两档广播
> （`Extend` / `Duplicate`）至今零真实图证据，原因不是尾轴对不对齐——是非 ui8 的图全被
> `afir.nddma` 挡在上游，根本走不到广播那一层。

那 54 条 ui8 还有一层没解释：`kExtendElementTypes` 与 `kSegmentedElementTypes` **都含 ui8**
（`Broadcast.cpp:221` / `:528`），所以拦它们的是另一条判据，而**今天的诊断答不出是哪一条**
（`has no supported Ascend implementation` 只说"这个 op 没实现"）。这正是功能需求 2 要补的。

控制流：

```
因子表（F1..F9）
   │
   ├─> 探针（全维度，无断言 → 带基线断言）   产出：清单 + 每格命中的判据
   │        │
   │        └─ ok 的格 ──> 挑代表格 ──> CPU E2E（数值）
   │        └─ unsupported 的格 ──> 入册，带解锁条件
   │
   └─> lit / UT（固定 descriptor，答"发出来的是什么"）
```

**顺序是设计内容，不是流程图装饰**：E2E 代表格必须从探针报 `ok` 的集合里挑。先写 E2E 再看支持与否，
写出来的是**假绿用例**——`GenerateForInductor` 在 MLIR 报不支持时会接着跑 legacy 并返回成功，
数值照样全对。

## 功能需求

### 功能需求 1：因子 → 判据对照表

1. 介绍

   每个因子必须指出它能跨过哪条判据。跨不过任何判据的因子取值不进矩阵——那是在给用例集注水。

2. 输入

   | 因子 | 取值 | 跨的是哪条判据 |
   |---|---|---|
   | **F1 秩 / 段数** | 1, 2, 3, 4, 5 | `classifySegments` 的 `count < 2`（单段让给 `Duplicate` 档）；`windowStart = min(front, count-2)`；`kNddmaMaxDims = 5`（`Nddma.cpp:86`） |
   | **F2 尾轴字节 vs 32B** | 4 / 8 / 16 / 20 / 28 / 32 / 44 / 128 / 512 | mem 两段窗口的 `provablyEqual(outer.strides[dst], inner.extent)`：`kAligned` 下外段 stride 取 `Align(Z, 32/sizeof(T))`、内段 extent 是 `Z`，不整除就当场不等（`Broadcast.cpp:275` 起） |
   | **F3 dtype** | f32 / f16 / i32 / ui8 | 32B 一档装多少个元素由它定，同一形状换 dtype 就跨过 F2 的边界；`kDuplicateElementTypes` 缺 ui8（`Broadcast.cpp:109`，原因是 ui8 在 legacy 绑的是包装重载而不是 `AscendC::Duplicate`） |
   | **F4 最内轴是否广播轴** | brc / normal | 形态 A / B 的分野：`provablyLiteral(outer.strides[src], brcInner ? "1" : "0")`。**对齐形状上唯一的卡点**（实测 28 / 28 全是这一类）：最内那根广播轴在源 tile 上只有 1 个元素却占满一整 32B 档，于是源在次内轴上的 stride 是 `32/sizeof(T)` 而不是 1 |
   | **F5 广播段拓扑** | 单段 / **相邻** `[brc][brc][normal]` / **间隔** `[brc][normal][brc]` / 全广播 | `broadcastSegments` 列表与 `windowStart`。相邻那一档 2026-10-11 才收（此前拒），今天只有 `afir-broadcast-adjacent-brc.mlir` 一份 lit，**零数值证据**；全广播走 `Duplicate` 档 |
   | **F6 分块趟数** | 单趟 / 多趟 + 尾块 | 切分轴恒落第 0 段这条不变量（`getTileInnerAxis` 的 `front()`）；`materializeTailAwareTileSize`。两段窗口里 `dst_k` 既是行长又是行距，切分轴落到内侧段上就是静默算错 |
   | **F7 内侧整块元素数** | ≤ 64 个 f32 / > 64 | `broadcast.h` 的 `BroadcastFirstDim` else 支按 counter 模式设掩码、却按一行给 `dstRepeatStride`，超过一次 repeat 的量就散错地址。**已知硬上限**，> 64 的格是 expected-fail，不是盲区 |
   | **F8 图拓扑** | T1..T7，见下 | `classifySegments` 的「源与目的地各恰好一个」；`classifyNddmaDims` 的 `plan.tensors.size() != 1`；一个 SSA 上 descriptor 必须一致（`InvalidBinding: conflicting canonical descriptors`）；tmpBuffer 争用；流水线同步 |
   | **F9 平台侧** | mem / reg | `ApiFamily` 选哪一侧实现；reg 侧调度把 `Load → Broadcast` 融成 `Nddma`，于是 F1..F7 在 reg 侧先过 `afir.nddma` 那一关 |

   **F8 的七种拓扑**（今天探针只有 T1，E2E 只有 T1 与纯 elementwise 的 T0）：

   | 代号 | 图 | 它能跨什么 |
   |---|---|---|
   | T0 | `load → abs → store`（对照） | 区分「这条边界是广播独有的」与「它是搬运层的」——`afir.load` 多段搬运那条就属于后者 |
   | T1 | `load → brc → store` | 现状基准 |
   | T2 | `load → brc → abs → store` | 广播结果**进计算**：brc 的 tmpBuffer 与下游算子的 tmp 争用、`PIPE_V → PIPE_V` 同步 |
   | T3 | `add(load(y), brc(load(x))) → store` | **最真实的 inductor 形态**，今天零覆盖。两个 operand 的 descriptor 落在同一个 op 上，直接打到「源与目的地各恰好一个」与 nddma 的 `tensors.size() != 1` |
   | T4 | `load → brc → brc → store` | 第二个 brc 的源是**片上**张量而不是 Load 的结果，`globalAxisUnitStride` 在它身上答不出来 |
   | T5 | `brc` 一个结果两个消费者（diamond） | 同一个 SSA 上 descriptor 一致性；`Or` 不可达不变量在多消费者下仍成立 |
   | T6 | `load → brc → cast(f16) → store` | dtype 在图中途变 → 32B 档位在同一张图里变化，F2 / F3 的交叉 |
   | T7 | `brc` 之后两个 `store`（多输出） | 多输出的偏移逐张量各取各的 stride |

3. 处理

   矩阵**不全叉乘**。全叉乘是 9 个因子相乘上万格，而判据只有十几条——裁剪规则是
   **「每条判据至少被一格隔离」**：一格只违反一条判据（前序文档的变异验证已经栽过一次，
   「一格同时违反两条、互相遮蔽」会让判据看起来是冗余的）。

   具体切法：F1..F4、F6 由探针的形状 × dtype sweep 覆盖（今天已有，380 格）；F5 由广播轴子集覆盖
   （已有）；**F8 另开一条切片**：7 种拓扑 × 4 个代表形状格（按 F2 × F4 的四象限各取一个）× 2 个 dtype
   ≈ 56 格，不与 380 格叉乘。

4. 输出

   对照表进本文档；裁剪后的格数进探针代码的注释（每组形状 / 拓扑旁写它管哪一类，照既有写法）。

### 功能需求 2：探针加判据标签与图拓扑维

1. 介绍

   两件事，各自独立：诊断里要能读出**是哪条判据拦的**；sweep 里要有**图拓扑**这一维。

2. 输入

   - 判据标签：`capability: no implementation` 这一桶今天只说「这个 op 没实现」。
     把 `classifySegments` / `classifyNddmaDims` 的每条 `return failure()` 换成带标签的失败
     （标签是**判据名**，不是形状描述），由 provider 透到诊断上。
     **它不是只为探针服务**：reg 侧那 54 条 ui8 今天就是因为没有标签而归不了因。
   - 图拓扑维：`BuildBroadcastGraph` 旁边加 T2 / T3 / T4 / T5 / T6 / T7 六个构图函数，
     共用同一套「广播轴由 in 该维是 1、out 不是定出来」的写法。

3. 处理

   - 标签只在**失败路径**上多一次字符串构造，成功路径一行不动（成功路径在编译期热路径上）；
   - 拓扑切片与现有 sweep 分开打表，行尾沿用 `tail=<N>B` / `inner=brc|normal` 两列，再加 `topo=T<N>`；
   - 探针仍不接 `check-*`，仍是独立可执行文件。

4. 输出

   两张表（形状 sweep、拓扑切片），每行带状态 + 判据标签 + 签名列；桶计数里 `kFailed` 与 `DRIFT` 两格恒为 0。

### 功能需求 3：清单棘轮（基线签到 + 比对）

1. 介绍

   探针今天不断言任何事，所以「泛化推进了没有」只能靠人读表对比两次输出。本需求把**期望状态**
   签进版本库，让变化必须被显式承认。

2. 输入

   `test/probe/expected/mem.txt` 与 `reg.txt`：每行 `<签名> <ok|unsupported> [<判据标签>]`，
   签名是 `dtype shape_in->shape_out topo`，**不含** plan 名、不含 SSA 编号、不含调用形态。

3. 处理

   探针加 `--expect <file>`：逐行比对，差异分三类，处置不同——

   | 差异 | 含义 | 处置 |
   |---|---|---|
   | 出现 `kFailed` 或 `DRIFT` | 硬失败 / 能力层与 pattern 漂移 | **红**。这两条是不变量，任何情况下不许非 0（verifier 失败不是回落，拒收只能拒在 capability 层） |
   | 期望 `ok` 实测 `unsupported` | 泛化退化 | **红** |
   | 期望 `unsupported` 实测 `ok` | 泛化推进 | **红**，并提示「请更新基线」。推进也要显式签一次，否则下一次退化就没有锚点 |
   | 判据标签变了而状态没变 | 换了另一条判据拦 | **红**，提示同上。这一类最容易悄悄发生，而它意味着原来那条判据的隔离性没了 |

   **棘轮记的是整图的 ok / unsupported，不是具体 plan。** 不许把期望写成「选中了哪个模板」——
   plan 是代价模型的性能选择，钉住它等于拿测试去逼一个本该由代价模型决定的结果。

4. 输出

   `make -C build af-capability-probe -j 6` 之后一条命令出结论；退出码非 0 即红。基线文件的 diff
   就是这个 PR 的泛化账单。

### 功能需求 4：CPU E2E 代表格与 dtype 参数化

1. 介绍

   数值证据只能来自 E2E。格数是构建时长，所以**一格一个理由**，并且**只从探针报 `ok` 的集合里挑**。

2. 输入

   框架改动一处：`add_mlir_e2e_cpu_case` 加 `DTYPE` 参数（今天 dtype 烧在每个 `Case.cpp` 里，
   `load_axpy_half_store` 是靠复制一整个目录换的 f16）。它是**一致性改动**而不是顺手改动：
   F3 是本设计的因子之一，不参数化就只能按 dtype 复制目录，4 个 dtype × 代表格数直接翻倍。
   `CaseConfig.h` 已经是 `file(GENERATE)` 生成的，加一个 `kElementType` 常量与一个
   `using ElementType` 即可，`KernelTest` 的参考值计算本来就是按形状算的、与 dtype 无关。

   **已落地（2026-10-12）**：按拓扑切片报 `ok` 的格挑出 8 格，全部逐元素通过，既有 70 格零回归
   （`check-ut` / `check-lit` / `check-st` / `check-e2e` 全绿）。注册格数 70 → 78。

   | 新增格 | 拓扑 | 形状 / dtype | 一个理由 |
   |---|---|---|---|
   | `load_broadcast_abs_store_1x8` | T2 | `[1,8]→[128,8]` f32 | 广播结果进计算，形态 A |
   | `load_broadcast_abs_store_1x1x8` | T2 | `[1,1,8]→[4,4,8]` f32 | 下游算子在**多段广播分解之后**那块片上张量上算；相邻广播段此前只有 lit，这是它第一次有数值证据 |
   | `load_broadcast_add_store_8x1` | T3 | `[8,1]+[8,128]` f32 | 最真实的 inductor 形态；一个算子身上同时挂广播来的与普通搬入的两张片上张量 |
   | `load_broadcast_chain_store_1x1x8` | T4 | `[1,1,8]→[4,4,8]` f32 | 第二趟广播的源在片上；期望值按**一次**广播算，等价性本身就是被测的东西 |
   | `load_broadcast_cast_store_1x8` | T6 | `[1,8]→[128,8]` f16→f32 | 同一串轴跨过 32B 边界两侧（尾轴 16B → cast 后 32B） |
   | `load_broadcast_cast_store_1x1x8` | T6 | `[1,1,8]→[4,4,8]` f16→f32 | dtype 变化叠在多段分解上 |
   | `load_broadcast_two_output_store_1x8` | T7 | `[1,8]→[128,8]` f32 | 两条 Store 共用一个源，各算自己的 GM 偏移（形态 A） |
   | `load_broadcast_two_output_store_8x1` | T7 | `[8,1]→[8,128]` f32 | 同上，形态 B |
   | `load_broadcast_diamond_store_1x8` | T5 | `[1,8]→[128,8]` f32 | 广播作分叉点，形态 A；分叉点的结果与 `add` 的结果共用一个非队列 `b<id>` |
   | `load_broadcast_diamond_store_8x1` | T5 | `[8,1]→[8,128]` f32 | 同上，形态 B（最内轴就是广播轴，plan 把广播排在别处） |

   **末两行（T5）不属于上面那 8 格**，是 C-3 收完之后才解锁的（见功能需求 6）：那一条先卡
   在绑定层的 `alignment_bytes`、再卡在非队列资源的成员布局，两条都修完它才真的走得通。
   同样不在那 8 格里的还有 `load_broadcast_abs_store_8x1` 与 `load_broadcast_cast_store_8x1`
   ——它们由单元素窗口那条改动（commit `59a972e6`）解锁，属于 C-1。
   连同 elementwise 侧的 `load_abs_diamond_store_128x8`，注册格数现在是 **83**。

   落地时加了两个共用辅助——既有两个广播用例已各抄一份，再加 5 个目录就是抄 7 遍：
   `support/E2eCase.h` 的 `BindBroadcast`（建图侧，与探针同名函数逐字同一条约定）与
   `support/E2eCaseConfig.h` 的 `BroadcastSourceIndex`（数值侧）。后者放在 `E2eCaseConfig.h`
   是因为 `KernelTest` 跑在设备侧那个二进制里，包不了带 `AscGraph` 的头。**既有两个用例没有
   改成用它们**——那是与本需求无关的重构。

   `DTYPE` 参数**这一轮没做**：T6 的绿格只在 f16 源上，目录里写死 `DT_FLOAT16` 就够，没有触发
   「同一张图要扫四种 dtype」的需求。接口设计一节那一行留着，要它的时候再加。

   既有 6 格（`Duplicate` ×2、形态 A、形态 B、形态 B 多趟、多广播段四段）全部保留。

3. 处理

   - 级联形态放在**一个**新目录里按参数展开，不是一格一个目录——它们共用同一份参考值计算；
   - `CheckGeneratedKernel` 必须保留 `..._subgraph0_plan0` 那条断言：它是「没回落 legacy」的唯一证据，
     少了它整个用例在回落之后照样全绿；
   - **不许断言「没有某个 API」**：同一张图在 reg 侧有四个模板，`BroadcastExtend` 的出现不是回落的证据；
   - 期望值要在目标 dtype 里精确可表示（f16 那一格照 `load_axpy_half_store` 的选法）。

4. 输出

   `check-e2e-cpu-<name>` 每格一个目标；逐元素比对。新增格数按上表 ≤ 8，构建时长影响见「性能」。

### 功能需求 5：`afir.nddma` 已知缺口入册

1. 介绍

   用户点名的「已知尚未实现的 nddma 场景」。每条给**落点、现象、解锁条件、用例在哪**，
   不写成"待支持"了事。

2. 输入 / 处理 / 输出（逐条）

   **N1　调度侧 `GenNddmaNode` 漏抄 `strides`——reg 侧一切的单点前置**

   - 落点：`autofuse/v35/optimize/template/nddma_template.cpp:111-112` 只抄了 `repeats` 与
     `vectorized_strides`，`strides` 留着描述广播**之前**的输入；
   - 现象：`afir.nddma` 在 `resultDescriptorIsSelfConsistent`（`Nddma.cpp:148`）上报不支持、整图回落。
     **2026-10-12 实测：reg 侧 380 张图里 326 条拒在这里，非 ui8 的图一张都到不了 broadcast。**
     真正的受害者是下游 `afir.store`：`OffsetContext::strideOf` 回落到局部 `strides`，GM 偏移取到输入的行距，
     **编得过、跑得完、结果错**，而且只有多分块形状看得出来（`[8,128]` 一个分块全绿，
     `[1024,128]` 69 个分块、131042 / 131072 个元素错）；
   - 解锁条件：调度侧把 `strides` 一起抄过去（与另外两项同一个来源）。它会**同时改变 legacy 自己的
     生成结果**，所以要按 v35 的 ST 套件单独验，不能夹在测试 PR 里；
   - 用例已备：`test/e2e/cpu/cases/load_broadcast_nddma_store/` 四格（`[8,1]` / `[1,8]` / `[1024,1]` /
     `[1,2,1,8]`，各管一类，理由写在那个 CMakeLists 里），`test/e2e/cpu/CMakeLists.txt:73` 那行
     `add_subdirectory` 今天是注释掉的。N1 一解，**取消注释即可，不改用例文件**。

   **N2　维数超过 `kNddmaMaxDims = 5`**

   - 落点：`Nddma.cpp:86` / `:276`，超限报不支持，由 legacy 按它的 `CreateNddmaCall` 拆外层循环；
   - 现象：零用例——今天造不出命中它的图；
   - 解锁条件：先有一张**切完之后**维数 ≥ 6 的图。注意「分成哪几维」不等于合轴分出来的段数：
     `classifyNddmaDims` 在片上的段之上按 GM 连续性**再切一刀**，所以秩 6 不保证 6 维；
   - 本设计给的构造法：秩 6、广播轴间隔分布（保证每一段的 GM 侧都断开）。**先在探针里造**，
     确认它真的命中 `dims.size() > kNddmaMaxDims` 这条标签（功能需求 2 的标签正是为这种确认服务的），
     再决定要不要给它 E2E。

   **N3　第二条来路 `LoadToNddmaTemplate`（尾轴不连续的搬入，一个广播维都没有）**

   - 落点：`afir.nddma` 有两条来路，这条发同一条包装，所以 `classifyNddmaDims` **不要求有广播维**；
   - 现象：**零覆盖，而且今天的探针触发不到它**——reg 侧 elementwise 对照 13 / 13 全 ok
     （含尾轴不对齐的 `[33,5]` / `[2,3,5,7]`），说明这些形状上这条来路没被触发；
   - 解锁条件：先知道调度在什么条件下走 `LoadToNddmaTemplate`。**这是一条「先查清楚再开工」的项**，
     不是用例缺口——不知道触发条件就加形状，是在给一个可能不存在的形状铺砖（2026-09-23 那轮已经栽过一次）。

   **N4　reg 侧两档广播（`Extend` / `Duplicate`）零真实图证据**

   - 现象：只有 lit。原因**不是**尾轴对不对齐（reg 侧 2026-10-11 已收尾轴填充），是 N1 挡在上游；
   - 唯一能到 broadcast 的是 ui8（54 条），而那 54 条又被另一条判据拒，**今天的诊断答不出是哪一条**
     （两档的 dtype 集合都含 ui8）。所以 N4 的第一步是功能需求 2 的判据标签，不是加用例；
   - 解锁条件：N1 + 判据标签。两者齐备之后 reg 侧的 E2E 直接复用 `load_broadcast_nddma_store` 那四格。

   **排序**：N1 是单点前置，解它之前 N4 不可能有进展、N2 / N3 都只能停在"查清触发条件"。
   mem 侧的排序不变：最大单一卡点仍是尾轴不对齐（要换 API 形态），**对齐形状上唯一的卡点是 F4
   「最内轴是广播轴」**（28 / 28）。

### 功能需求 6：拓扑切片的实测结果与它暴露的三条缺口（2026-10-12）

1. 介绍

   拓扑维量完了。**它当场破了两条此前恒为 0 的不变量**——`kFailed`（硬失败，不是回落）与
   `DRIFT`（能力层答支持而 pattern 失败）。380 格的形状 sweep 两侧各 0 次，级联切片 112 格
   里就出现了 26 次硬失败和 4 次漂移。

2. 输入

   量具：`af-capability-probe --topo`，12 种拓扑 × 5 个形状格 × f32/f16（T4 只在两根广播轴
   那一格造得出来），共 112 张图，11 秒。mem 侧（Ascend910B1）：

   | 拓扑 | ok | 回落 | **硬失败** | 说明 |
   |---|---|---|---|---|
   | T0 对照（不含广播） | 10 | 0 | 0 | |
   | T1 基准 `brc → store` | 4 | 6 | 0 | 与 380 格 sweep 的结论一致 |
   | T2 `brc → abs` | 2 | 6 | **2** | `failed to legalize operation 'afir.abs'`（DRIFT） |
   | T3 `add(full, brc)` | 2 | 2 | **6** | `BindingConflict: node 'load0' ... incompatible tensor layout` |
   | **T3c 对照**（两输入都不含广播） | **10** | 0 | 0 | 二元算子本身没问题 |
   | T4 `brc → brc` | 1 | 1 | 0 | f32 过、f16 回落 |
   | T5 `brc` 作分叉点 | 0 | 0 | **10** | `InvalidBinding: conflicting descriptors for one canonical resource` |
   | **T5c 对照** unary 分叉 + unary 消费者 | 0 | 0 | **10** | **不含广播也挂** |
   | **T5d 对照** binary 分叉 + binary 消费者 | **10** | 0 | 0 | 与既有 E2E `load_add_diamond_store` 同形 |
   | **T5e 对照** unary 分叉 + binary 消费者 | 0 | 0 | **10** | |
   | T6 `brc → cast` | 2 | 6 | **2** | `failed to legalize operation 'afir.cast'`（DRIFT） |
   | T7 多输出 | 4 | 6 | 0 | 与 T1 同步——多输出本身不额外卡 |

3. 处理

   三条缺口，每条都由对照格隔离出来，不是按形状猜：

   **C-1　`brc → 计算` / `brc → cast` 在「最内轴是广播轴」那一格漂移。** 四次漂移全部命中
   `[8,1] → [8,128]`（形态 B，尾轴对齐），f32 / f16 各两次；其余形状上 T1 自己就回落了，
   下游算子轮不到报。**这是漂移，不是能力不足**：能力层答了支持，pattern 才失败，而仓库的
   约定是拒收只能拒在 capability 层。落点在 elementwise / Cast 那两处 pattern 怎么看待
   「源是两段窗口分解出来的片上张量」。

   **C-2　`add(full, brc)` 的队列冲突出在那条 stride 为 0 的 Load 上。**（已改成回落，见下）T3 六格硬失败，
   而把广播那一侧换成整形状 Load 的对照格（T3c）五个形状十格全过。所以不是「二元算子有两个
   输入」，是**广播轴 stride 为 0 的 Load 与整形状 Load 同处一个队列**（诊断点名的就是
   `load0`，`buffer_id=0 reuse_id=0`，说两张布局不同的张量被摆进了同一个资源）。
   **而它在 `[8,1] → [8,128]` 上是过的**，所以不是全盘不支持，是某个判据在另外几种布局上
   答错——这条有现成的最小复现（探针的 T3 与 T3c 一对）。

   **已改成干净回落（不是修成支持）。** 逐字段打点量出来，对不上的是**张量视图**：
   一侧 `repeats = [128, ...] strides = [8, ...]`（整形状 Load），另一侧
   `repeats = [1, ...] strides = [0, ...]`（广播轴 stride 为 0 的 Load）。
   `CheckResourceCompatibility` 那条等值断言的前提是「同一条队列上的张量今天必然是同一个
   tile」，广播让它不成立，于是合法调度被判成 `BindingConflict` 硬失败、不回落。
   改成：差别**能由 stride 0 解释**时报 `emitCapabilityShortfall`（整图回落 legacy），
   解释不了的仍旧是硬失败。判据写窄是为了保住「调度真的给了两张对不上的视图」那一半，
   UT `BroadcastTileSharingOneQueueIsCapabilityShortfallNotConflict` 两条断言成对钉这件事。
   **没有修成支持**：实测放开那条断言之后这六格并不会变 ok，而是接着报
   `afir.broadcast` 不支持——真正的卡点在广播自己的形状判据上，所以放开只是把硬失败换成
   另一条拒收，一张图都不会多编出来，也就没有数值可验。要支持得先收掉那一象限。

   **C-3　`brc` 作分叉点那十格硬失败与广播无关，判据是「分叉点是 unary」。** 三个对照格把它
   夹死了：T5d（分叉点 binary、消费者 binary，与既有绿用例 `load_add_diamond_store` 同形）
   十格全过；T5c（unary 分叉 + unary 消费者）与 T5e（unary 分叉 + binary 消费者）各十格全挂。
   **变的是分叉点，不是消费者，也不是广播**——`afir.broadcast` 跟着挂只是因为它也是单输入
   单输出那一档。这是一条先于本需求存在的缺口，既有用例集全盲（唯一的菱形用例分叉点是
   `Add`）。

   **已修（2026-10-12，紧随本轮）。** 判据落在 `registerResource`：unary 分叉点的结果被
   调度放进**非队列的计算缓冲**，而 unary 算子的 tmp buffer 与普通计算缓冲**同在一个 id
   空间**（见 `ScheduleToAfir` 里那段注释），于是两者可以落到同一个 `b<id>`。那里此前要求
   同一块存储上的 descriptor 逐字一致，而这两份只差 `alignment_bytes`（tmp buffer 由
   bridge 固定带 32、计算缓冲是 -1），于是整图报
   `InvalidBinding: conflicting descriptors for one canonical resource`。
   **原先猜的「unary 就地复用输入缓冲」不成立**——冲突的两份 descriptor 一份是 tmp buffer、
   一份是下一个算子的结果，与输入无关。
   线性链上看不见：那里 unary 的结果直接进队列，非队列那一侧只有 tmp buffer 一家，所以
   「分叉点是 unary」是那个把第二块张量挤进非队列空间的条件。
   修法是把 `alignment_bytes` 从「身份字段」改成「要求字段」：它说的是某个张量对这块存储
   字节数的要求（-1 = 没有要求），多个张量共用时取最小公倍数，与 `slot_count` 取 max 同理。
   legacy 那一侧每块 TBuf 的字节数本来就过 `BlkAlign<uint8_t>`，取 32 与它一致。
   实测 T5c / T5e 各十格全过，新增一格 E2E `load_abs_diamond_store`（与
   `load_add_diamond_store` 只差分叉点）数值正确。


   **T5（广播作分叉点）仍然收不了，但原因换了一条，而且是另一处缺口。** 放开绑定之后它
   走到了下一层并暴露出一个此前到不了的矛盾：片上资源的字节数由 ATT 填进 `b<id>_size`，
   那个数是 `max(同 id 的每个张量) ⊔ Σ(同 reuse 组)`，而 MLIR 的
   `materializeLocalMember` 把同一组的成员**顺序排**（下一个的偏移 = 上一个偏移 + 上一个
   字节数）。于是 `reuse_id = -1` 的两个成员同组时，ATT 给的是 max、MLIR 要的是和，
   第二个成员的偏移落在资源外面。**失败方式不是报错，是运行期越界 abort**——CPU 模拟器上
   的原话是 `offset is 32, which can not be larger than data len 32`，而且输出一个字节都
   没写（哨兵值原样留着）。T5 的 `[1,8] -> [128,8]` f32 实测就是这个形状：分叉点的结果与
   `add` 的结果都落到非队列 `b1` 且都不带 reuse_id。
   **也修了。** 收法不是拒收，是按成员自己的 `view.offset` 摆：`reuse_id = -1` 的两个成员
   偏移都是 0，意思是「在时间上复用这块存储」，legacy 对这种形状发的就是一次原地计算
   （`Add(dst, dst, other)`）。此前 `materializeLocalMember` 对队列与非队列一视同仁地累加
   偏移——队列上是对的（累加就是摆 slot），非队列上则要求 `b<id>_size` 是成员字节数之和，
   而那个数是 ATT 按 `max(同 id 的每个张量) ⊔ Σ(同 reuse 组)` 给的。
   「一个 buffer 一份存储、成员大小与资源大小是同一个数」本来就写在
   `materializeLocalMemberSize` 的非队列分支上，所以这是把两处对齐，不是新立规则。
   影响面只有非队列多成员分组，改之前绿用例里是 0 个。
   T5 由此变成「4 格 ok、6 格报 `afir.broadcast` 不支持」，与 T1 / T2 / T6 / T7 同一个
   剖面；新增 E2E `load_broadcast_diamond_store` 两格（形态 A `[1,8] -> [128,8]`、形态 B
   `[8,1] -> [8,128]`），数值正确。**它是这一条的唯一数值证据**——改之前这张图不报错，
   只是输出一个字节都不写。

4. 输出

   三条进「下一步」，各自有最小复现（探针里的一对格）。**C-3 不归本需求**：它不是广播的
   泛化缺口，归 elementwise 那条线，本设计只负责把它量出来并隔离——它已在紧随本轮的一条
   独立改动里修掉，见上。

   由此，`afir.broadcast` 今天的级联可用面是：**广播结果可以进计算、可以换 dtype、可以写到
   多个输出、可以串两趟、也可以当分叉点；但都只在「尾轴对齐」那一象限上成立。**
   这一句是 12 格广播级联 E2E 加 112 格清单共同支撑的，不是推的。

   「最内轴不是广播轴」这一条**不再是可用面的边界**：T2 / T6 的形态 B 由单元素窗口那条
   改动收掉（commit `59a972e6`），T5 的形态 B 有 E2E
   `load_broadcast_diamond_store_8x1`。剩下报不支持的那些格，判据是尾轴字节数——
   f16 的 `[1,8] -> [128,8]` 尾轴 16B、`[1,5] -> [33,5]` 尾轴 20B，都不足 32B。
   那一条是真实的硬件不变量（违反会静默算错），所以其中一部分**本来就该拒**，
   不是等着被收掉的缺口。

## 非功能需求

### 可维护性

- 探针的每组形状 / 每种拓扑旁必须写它管哪一类（既有写法已如此），"随手加一个形状"不收；
- 基线文件只记签名与状态，不记 plan、SSA 编号与调用形态——那些会随调度顺序漂移，钉它们等于每次改动都要改基线；
- 级联 E2E 共用一份参考值计算，不按形态复制。

### 可测试性

- 三层载体的出口各不相同：`VectorPlan` 结构（UT）、AscendC 文本（lit）、kernel 执行结果（E2E）、
  整图状态清单（探针）。一个问题只在一层回答；
- 棘轮本身要做变异验证：手改一行让某格从 `ok` 变 `unsupported`，比对必须红。

### 可移植性

不新增平台判断。平台维走既有两处分流（`soc_version` + NpuArch），数值取自 profile 文件。
CPU 模拟器与 NPU 编的是同一份 `dav_c220` 头，所以这批用例不需要 NPU。

### 可靠性

- 探针不改产品代码路径；判据标签只在失败路径上多一次字符串构造；
- 棘轮的两个不变量（`kFailed` 恒 0、`DRIFT` 恒 0）比对失败即红，不允许"已知失败"白名单。

### 特性交叉影响

按 `docs/guidelines/cross_feature_check.md` 逐项：

| 场景 | 适用性 | 分析说明 |
|------|--------|----------|
| SuperKernel Python 接口 | 不适用 | 改动全在 `autofuse/mlir/test/`，不碰 `super_kernel/` |
| SuperKernel C++/AOT 接口 | 不适用 | 不新增导出符号，不碰 `libascendsk.so` |
| Autofuse 图优化 | **适用（只读）** | 用例经 `Optimizer::Optimize` 产图，不改任何 pass。N1 的解锁**会**改调度（`GenNddmaNode`），已排除在本次范围外，并注明要按 v35 ST 单独验 |
| Autofuse Codegen/Backend | **适用（只读）** | 走 `RunMlirCodegen`，不改 codegen。判据标签是**诊断文本**的变化，不改判决结果；要确认既有 lit 里没有按诊断原文匹配的 CHECK |
| AscendC API / Runtime 交互 | 不适用 | 不新增 `rt*` / `aclrt*` 调用；E2E 跑 CPU 模拟器 |
| Python/C++ 混合绑定 | 不适用 | 不碰 `autofuse/compiler/py_module/` |
| 构建与打包 | **适用** | `AddCpuCase.cmake` 加 `DTYPE` 参数、新增 case 目录与 `add_subdirectory`。要确认 `--no-autofuse` 与增量构建不受影响；探针仍不进任何 `check-*` 目标，所以 CI 时长不变 |
| 测试与覆盖率 | **适用** | 本设计的全部内容 |
| 性能与日志 | **适用（弱）** | 新增 ≤ 8 格 E2E 的构建时长，见「性能」；不新增默认开启的日志 |
| 兼容性 | **适用（弱）** | `DTYPE` 是**可选**参数，默认 f32，既有 70 个注册格一行不改 |

## 性能

### 编译时长

- 探针：**实测 0.23 s / 图**（393 张图 89.8 s）。F8 切片按 ≈ 56 格算，增量 ≈ 13 s；两侧各跑一遍
  总计仍在 4 分钟内。它不接 `check-*`，不进 CI 时长；
- CPU E2E：每格一个生成器可执行文件 + 一次 kernel 编译。今天 70 个注册格（59 个目录挂进
  `check-e2e-cpu`），新增 ≤ 8 格 ≈ **+11%**。单格耗时**待实测**（本轮没量，不编）；
- `DTYPE` 参数化不增加格数，它只是把"复制目录换 dtype"换成一个参数。

### 执行性能

不涉及运行时路径。判据标签只在失败路径上，而失败路径以整图回落 legacy 收尾，一张图一次。

### 内存和产物大小

E2E 每格一个可执行文件（既有量级），新增 ≤ 8 格。基线文件两份纯文本，各 ≈ 400 行。
不进 run 包、不进 wheel。

## 接口设计

### 新增/修改接口描述

全部是测试侧接口，无对外 ABI/API：

| 接口 | 形态 | 兼容性 |
|---|---|---|
| `add_mlir_e2e_cpu_case(... DTYPE <name>)` | 新增**可选**单值参数，默认 `float`；生成的 `CaseConfig.h` 多一个 `kElementType` 与 `using ElementType` | 既有 70 个注册格不改一行 |
| `af-capability-probe --expect <file>` | 新增可选参数；不带它时行为与今天逐字相同（只打表、退出码 0） | 无调用方 |
| `af-capability-probe --topo` | 新增可选参数，只跑 F8 切片 | 同上 |
| 判据标签 | `classifySegments` / `classifyNddmaDims` 的失败返回带标签，由 provider 透到诊断 | **诊断文本变化**：要 grep 既有 lit 有没有按原文匹配的 CHECK |

### 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|--------|----------|----------|------|
| 接口说明 | 是否需要接口评审 | 否 | 测试侧接口 |
| 接口说明 | 是否需要补充文档 | 是 | 本文档 + 探针文件头的运行说明 |
| 接口兼容 | 行为是否兼容 | 是 | `DTYPE` / `--expect` 均可选，默认行为不变 |
| 接口兼容 | ABI/API 是否兼容 | 否 | 不导出符号 |
| 接口约束 | 约束不满足时是否清晰报错 | 是 | `DTYPE` 不认的值 `message(FATAL_ERROR)`（照既有 `PLATFORM` 的写法）；基线文件缺行 / 多行逐行报出 |
| 接口测试 | 是否需要独立接口用例 | 是 | 棘轮的变异验证（手改一格必须红） |

## 软件设计

### 关键数据结构

- 探针的一行：`{dtype, shape_in, shape_out, topo, status, criterion_label, tail_bytes, inner_is_brc}`。
  今天是 `snprintf` 拼出来的字符串，加两列之后改成结构体 + 一处格式化，便于与基线逐字段比对；
- 基线一行：`<签名> <状态> [<标签>]`，签名是前四个字段。**状态与标签之外不记任何东西。**

### 关键技术/算法

棘轮的比对是逐行集合比对，不是文本 diff：顺序由 sweep 的循环次序决定，而加一个 dtype 或形状会
整体移位。按签名建索引，缺行 / 多行 / 状态变 / 标签变四类分开报。

### 流程设计

主流程见「整体介绍」。异常流程两条：

- 探针在某张图上段错误 → 进度打在 stderr 且逐张 flush，所以看得出挂在哪一张（既有行为）。
  `LD_LIBRARY_PATH` 不对时就是这个现象，先对那一行；
- 基线文件不存在 → `--expect` 报错退出 2，不静默当通过。

### 对子模块的修改

| 目录 | 修改点 |
|---|---|
| `test/probe/` | 六个构图函数、判据标签列、`--expect` / `--topo`、`expected/{mem,reg}.txt` |
| `test/e2e/cpu/AddCpuCase.cmake` | `DTYPE` 可选参数 → `CaseConfig.h` |
| `test/e2e/cpu/cases/<新目录>/` | 级联代表格（T2..T7）一个目录、按参数展开 |
| `test/e2e/cpu/cases/load_broadcast_segmented_store/` | 补相邻广播段与秩 5 两格 |
| `test/e2e/cpu/CMakeLists.txt` | 新目录的 `add_subdirectory`；N1 解锁后取消 nddma 那一行的注释 |
| `test/lit/conversion/` | 级联形态的结构用例（受 lit 的最小 func 限制，见「设计约束」） |
| `conversion/AFIRToAscendC/Implementations/Broadcast.cpp` / `Nddma.cpp` | **仅**失败返回带标签，判决结果一行不改 |

### 错误处理

#### 系统错误

探针的 `GmAlloc` / 文件打开失败照既有写法处理（`FAIL()` 并释放已分配的）。基线文件读失败退出 2。

#### 接口错误

`DTYPE` 不认的值在 CMake 配置期 `FATAL_ERROR`；棘轮差异非 0 退出码 1，并把四类差异分开打印。

## 安全检查

按 `docs/guidelines/编码红线.md`：

- **硬编码敏感信息**：无。测试数据是 `index * 0.25 - 3.0` 这类明显占位值，不含密钥 / 地址 / 账号；
- **外部输入作索引**：E2E 的形状来自 CMake 生成的常量，`AddCpuCase.cmake` 已校验「正整数、
  元素数放得进 uint32_t」；探针的形状是代码里的字面量；
- **整数溢出**：新增格的元素数最大 `[2,3,5,7,32] = 6720`，远离边界；`DTYPE` 改变的是 `sizeof`，
  元素数校验在 CMake 侧按元素个数算，与 dtype 无关；
- **资源释放**：`KernelTest` 的分配失败路径照既有写法逐个释放；探针的 `StderrCapture` 已是 RAII
  且禁拷贝；
- **ABI/API 兼容**：不导出符号，`DTYPE` 可选；
- **图改写等价性 / 确定性**：本次不改任何 pass。棘轮的比对按签名索引而不依赖容器遍历顺序；
- **硬编码芯片类型**：平台维走既有两处分流，不新增芯片名判断。探针文件与 lit 的 stable id 里不出现芯片名；
- **runtime / AscendC 生命周期**：不新增此类调用。

## 兼容性检查

- `DTYPE` 可选、默认 f32 → 既有 70 个注册格不改；
- 探针不带新参数时行为逐字不变；
- 不改 `build.sh` 参数、安装路径、动态库名与 Python 包；
- **判据标签改了诊断文本**，这是唯一一处可能影响既有用例的改动：落地第一步先 grep
  `test/lit/` 有没有按 `has no supported Ascend implementation` 原文匹配的 CHECK。

## 测试设计

### 测试边界

- **入口**：`af-capability-probe`（整图清单）、`af-opt` 跑 conversion pipeline（lit）、
  `check-e2e-cpu-<name>`（数值）、`collapseVectorizedAxes`（UT 直接调）；
- **出口**：清单行（探针）、AscendC 文本（lit）、kernel 执行结果（E2E）、`VectorPlan`（UT）；
- **打桩**：无。探针与 E2E 走真实调度；lit 用手写 AFIR；
- **验证数据**：E2E 的参考值按形状在 host 侧重算（既有 `KernelTest` 的写法已是如此，与 dtype 无关）；
- **不在本次的构造法记在这里**（符号尺寸维）：`AscGraphBuilder::Sym` 登记运行期尺寸名、
  再把它当轴尺寸传给 `Loops`。预期方向是 `provablyEqual` 证不了 → 封段 → 多一层循环而不是算错；
  要先确认 conversion 里新建的 `tiling_param` 能 legalize。

### 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|----------|------------|----------|----------|
| 功能 | F1..F7 各格的支持与否可复算 | 探针 380 格 + `--expect` 比对 | 清单 |
| 功能 | F8 七种拓扑的支持与否 | 探针 `--topo` 切片 ≈ 56 格 | 清单 |
| 功能 | 相邻广播段 `[brc][brc][normal]` 算得对 | `[1,1,8] → [4,4,8]` 逐元素 | E2E |
| 功能 | 秩 5 多段分解算得对 | `[2,3,5,1,32] → [2,3,5,7,32]` | E2E |
| 功能 | 广播作为 binary 的一个 operand | T3 `add(y, brc(x))` | E2E |
| 功能 | 广播结果进计算（tmp 争用 + `PIPE_V` 同步） | T2 `brc → abs` | E2E |
| 功能 | 广播串联、源在片上 | T4 `brc → brc` | E2E |
| 功能 | 一个 brc 两个消费者的 descriptor 一致性 | T5 diamond | E2E |
| 功能 | dtype 中途变 | T6 `brc → cast(f16)` | E2E |
| 功能 | 分段档的 f16 数值 | `[8,1] → [8,128]` f16 | E2E |
| 异常 | 硬失败 / 漂移恒为 0 | 探针两个桶，比对即红 | 清单（不变量） |
| 异常 | 期望 ok 的格退化 | `--expect` 第二类差异 | 清单 |
| 异常 | 拦下的判据换了一条而状态没变 | `--expect` 第四类差异 | 清单 |
| 异常 | 内侧整块 > 64 个 f32（F7 硬上限） | 探针记成 expected-fail，**不写 E2E**（CPU 模拟器 `SIGABRT`） | 清单 |
| 异常 | N2 维数 ≥ 6 | 探针造秩 6 间隔广播，确认命中 `kNddmaMaxDims` 标签 | 清单 |
| 兼容性 | 既有 70 个注册格的生成 kernel 逐字不变 | 改 `AddCpuCase.cmake` 前后各跑一次、拷 `kernel.cpp` 逐字 diff | E2E |
| 兼容性 | 既有 lit 不按诊断原文匹配 | grep + 全量 `bash ./mlir-test.sh` | lit |
| 特性交叉 | N1 解锁后 reg 侧四格数值 | 取消 `CMakeLists.txt:73` 的注释，不改用例文件 | E2E（挂起） |
| 性能 | 新增格的构建时长 | 改前后 `check-e2e-cpu` 墙上时间对比 | 手工验证 |

**变异验证计划**（照前序文档的三类处置：盲区 / 冗余 / 保守边界）：

| 变异 | 预期 | 不红时的处置 |
|---|---|---|
| 手改一格的期望状态 | 棘轮红 | 红不了说明比对没接上，先修比对 |
| 删掉某条判据标签 | 棘轮第四类差异红 | —— |
| F4 那条源 stride 形态判反 | 探针里对齐格由 52 ok 变动 | —— |
| 级联 E2E 的参考值按错误的广播轴算 | 对应格红 | 不红说明那一格的形状分辨不出来（一维退化），换形状 |
| 相邻广播段那条判据改回"拒收" | `[1,1,8] → [4,4,8]` 红 | 不红说明这一格没走那条路，先确认它命中的标签 |

**一条规则写在这里，因为它最容易被违反**：**探针报 `unsupported` 的格不许写成 E2E。**
回落之后 `GenerateForInductor` 返回成功、数值照样全对，那是假绿；`..._subgraph0_plan0` 那条断言
只能证"没回落"，证不了"正确地拒收"。拒收的证据在清单里。

### 测试命令

```bash
# 1. 探针（不接 check-*，手动跑；一次只起一个）
make -C build af-capability-probe -j 6
LP=$(grep -ho 'LD_LIBRARY_PATH=[^";]*' \
     build/autofuse/mlir/test/e2e/cpu/cases/load_add_store/CTestTestfile.cmake | head -1 | sed 's/^LD_LIBRARY_PATH=//')
AF_ENABLE_MLIR_CODEGEN=1 LD_LIBRARY_PATH="$LP" \
  build/autofuse/mlir/test/probe/af-capability-probe \
  --expect autofuse/mlir/test/probe/expected/mem.txt
AF_ENABLE_MLIR_CODEGEN=1 LD_LIBRARY_PATH="$LP" \
  build/autofuse/mlir/test/probe/af-capability-probe --reg-base \
  --expect autofuse/mlir/test/probe/expected/reg.txt
```

```bash
# 2. lit + UT 全量（本机包装脚本，自带 LLVM_BUILD_DIR / CANN 环境）
bash ./mlir-test.sh
```

```bash
# 3. E2E 单格（不手工跑 generator：库路径不对会静默走 legacy）
make -C build check-e2e-cpu-load_broadcast_segmented_store_1x1x8 -j 6
```

预期结果：探针两侧 `kFailed` / `DRIFT` 均为 0 且与基线一致（退出码 0）；
`mlir-test.sh` 全绿；每个 `check-e2e-cpu-*` 逐元素相等。

## 验收标准

1. **因子表落地**：F1..F9 每个取值在探针代码或 E2E 的 CMakeLists 注释里写明它管哪一类判据，
   没有"看起来更丰富"这种理由；
2. **清单可复算**：`--expect` 两侧退出码 0；基线文件进版本库，PR 的 diff 就是泛化账单；
3. **两个不变量**：`kFailed` 与 `DRIFT` 两桶实测 0（本轮基线已满足：两侧 773 张图各 0）；
4. **F8 拓扑维有量（已达成 2026-10-12）**：12 种拓扑（含 4 个对照）各有清单行；报 `ok` 的每一类
   都有 E2E 数值（8 格，逐元素通过），报硬失败的三类各有对照格隔离；
5. **新增 E2E 格全部从 `ok` 集合里挑**，且每格在 CMakeLists 里有一句"它管哪一类"；
6. **既有用例零回归**：70 个注册格的生成 kernel 逐字不变（改 `AddCpuCase.cmake` 前后各生成一次对比）；
   `mlir-test.sh` 全绿；
7. **nddma 四条缺口入册**，每条带落点、现象、解锁条件与用例位置；N1 注明"调度侧一行 + v35 ST 单独验"，
   `load_broadcast_nddma_store` 四格保持备好状态，解锁时只改 `add_subdirectory` 那一行；
8. **变异验证逐条有处置**，不红的按盲区 / 冗余 / 保守边界三类之一归档，不留"不红但保留"而无理由的格；
9. 本文档与 `2026-09-28` 那份的「泛化重量」一节交叉引用；该节「reg 侧 48/48 全拒且一张都走不到
   broadcast」按本轮实测订正为 **326 拒在 `afir.nddma`、54 拒在 `afir.broadcast`（全为 ui8）**。

---

### 设计文档检查结果

- [x] **跨特性交叉影响**：已按 `cross_feature_check.md` 逐项填表（见「特性交叉影响」）。适用项三条：
      Autofuse 图优化 / Codegen 为**只读**（本次不改 pass 与 codegen 判决，仅诊断文本）、构建与打包
      （`AddCpuCase.cmake` 加可选参数 + 新增 case 目录）、测试与覆盖率（本设计全部内容）。
      N1 的解锁会改调度并影响 legacy 生成结果，已显式排除在本次范围之外。
- [x] **编码红线**：已按 `编码红线.md` 检查（见「安全检查」）。无硬编码敏感信息与芯片类型判断；
      外部输入作索引处沿用 `AddCpuCase.cmake` 既有校验；资源释放沿用既有 RAII 与失败路径写法；
      不导出符号故无 ABI/API 影响；不改图改写逻辑，棘轮比对不依赖容器遍历顺序。
- [x] **测试设计**：清单（探针 380 + 56 格，带基线棘轮）/ E2E（≤ 8 个新增数值格，全部从 `ok` 集挑）/
      lit（结构）/ UT（`VectorPlan`）四层分工，并给出五条变异验证与三类不红处置。
- [x] **性能评估**：探针 **实测 0.23 s / 图**（393 张 89.8 s），F8 切片增量 ≈ 13 s，且不进 CI 目标；
      E2E 新增 ≤ 8 格 ≈ +11% 构建量，单格耗时标注待实测，未编造数字；不涉及运行时路径与新增日志。
