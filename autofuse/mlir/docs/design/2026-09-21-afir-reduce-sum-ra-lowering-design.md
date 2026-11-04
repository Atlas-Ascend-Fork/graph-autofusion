# AFIR ReduceSum RA 下降设计

> 状态：已实施；RA、RCore 两阶段及设备数值链路已验证。下文 S4 未完成条目为 2026-09 历史记录。
> 2026-10-22 核对：Workspace 与 tiling_key 命名阻塞已解除；AR/RA RCore 已完成 NPU 回归。
> 历史探针命令即使改用 ATT DFX，也需重新确认当前 shape 的候选编号。
>
> 日期：2026-09-21，2026-09-23 按 S4 落地后的代码现状重写
>
> 文档类型：算子能力扩展与 RCore 前置能力设计

> **本次重写改了什么**：初稿写于 `2026-09-20-two-dim-transfer-addressing-design.md` 的 S4
> 之前，其中两条前提已经不成立，继续照初稿实施会做多余的事：
>
> | 初稿的说法 | 今天的事实 |
> |---|---|
> | 「跨 tile 累加把物理外维 `firstActual` 当结果元素数，RA 下该值是 R」 | S4 之后 `firstActual` 由 `plan->first.axes`（**语义 A 轴**）物化，累加 count 已经取 A 而非 R，初稿的功能需求 3 **无需实施**。但「取 A」不等于「count 正确」：A 组多轴时 destination 里的槽位还会跨步，那是另一条缺陷，见 3.4（2026-10-02 已修） |
> | 「当前开放边界是恰好两个有效 extent」 | S4 的多轴合并已落地，A / R 各自可以是多根轴，`materializeScheduledAxisExtents` 按轴列表物化 |
> | 「非队列 input/destination 继续拒绝」 | 2026-09-22 的 elementwise 方案已把该门禁放宽为「必须在 UB」，RA 不需要再碰 |
>
> 保留的结论：`ReduceInit` 只能在 AR 发、layout 要按 pattern 选、门禁两侧必须同源。
> 新增的结论：真正错位的是 **srcShape 的物理次序与 32B 对齐落在哪一组**，见 3.2。

# 简介

## 目的

让现有 `afir.sum` 的 AscendC lowering 在保持 AR 产物逐字节不变的前提下，支持物理布局为
`[R, A]` 的 RA 归约，并以 RCore phase 2 的 `[Rm, A] -> [1, A]` 形态证明该能力可以作为
后续开放 RCore 模板的前置条件。

目标读者是 AFIR Analysis、AFIRToAscendC、Reduce 调度与 E2E 测试的开发和评审人员。期望
输出是可以直接实施、验证、评审的文件级方案，而不是删除两行门禁。

## 范围

### 包含

- `f32`、单输入单输出 `afir.sum` 的 RA lowering。
- **R 组恰好一根轴**的 RA（理由见 3.3「为什么本期把 R 限制成一根轴」）。A 组轴数不限，
  与 AR 今天的约定相同。
- RA 的能力查询、AscendC op 发射、跨 tile 累加与测试闭环。
- AR 生成的 IR、kernel 文本和数值行为保持不变。
- ~~先用探针确认调度器是否会自然产出 RA~~ —— **已确认会**（2026-09-23，见 6.3），E2E 语料
  直接在 `load_sum_store` 上按 `reduce_axes = {0}` 注册。

### 不包含

| 项 | 理由 / 后续 |
|---|---|
| 完整开放 RCore 模板选择 | RA 只是 RCore phase 2 的硬前置。RCore 还有 phase 1、workspace、跨 stage 执行和多模板选择，另行收口 |
| R 组多根轴的 RA | 需要先把「组内非最外轴不带尾块」这条既有隐含前提变成可判定的门禁，见 3.3；本期返回 `UnsupportedShape` |
| A / R 多段交错布局 | `buildReducePlan` 已经拒绝，本方案不放宽 |
| Mean / Prod / Max / Min / ArgMax | 本期只扩展已注册的 `Sum` |
| `f16`、`int32` | 当前 mapping 只承诺 `f32`，类型扩展单独设计 |
| 修正「A 组内层轴带尾块时累加 count 数错槽位」 | 本方案内按红线 7 只记录不改，见 3.4。**2026-10-02 作为独立在册阻塞已修**（`resultSpan`），结论见 3.4 |
| 打通 workspace 偏移链路（`Workspace` 节点的 AFIR 表示 + kernel 侧按 `workspace<ID>` 取偏移） | 与 RA lowering 无关，属 RCore 开放前的必做项。按红线 7 只记录，见 3.8 与 [2026-09-23-mlir-workspace-offset-chain-design.md](2026-09-23-mlir-workspace-offset-chain-design.md) |
| 新增公开配置或脚本参数 | 无此需要 |

涉及组件：Autofuse MLIR 的 `conversion/AFIRToAscendC`、`dialect/AFIR/Analysis`（仅注释）
和测试。SuperKernel、Python/C++ 绑定、构建打包不受影响。

# 总体概述

## 软件概述

### 项目介绍

RA 的三层里，**两层已经完成**：

| 层 | 状态 | 证据 |
|---|---|---|
| `ReducePlan` 分析 | **已完成** | `ReducePlan.cpp:162` 按 destination 首个向量化 stride 是否为 0 判 AR/RA；`ReducePlanTest.ReducedLeadingAxisIsRa` 断言 RA 下 `first`=A 轴、`last`=R 轴 |
| AscendC 发射 | **已完成** | `ReduceSum2DL2Op` 带 `layout` 属性，printer 按 `isARLayout()` 输出 `Pattern::Reduce::{AR,RA}`（`VecReduceND.cpp:44`） |
| AFIR → AscendC 下降 | **缺口** | `Reduce.cpp:110` 能力查询拒绝，`Reduce.cpp:230` 改写再拒绝一次，`Reduce.cpp:321` layout 写死 `AR` |

所以本方案是一个**只改 `Reduce.cpp` 一个文件**的下降层变更，外加测试与文档。

### 产品环境介绍

- 入口：`bridge/CodegenOrchestration` 的 MLIR codegen 路径。
- 目标 API：`AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, false>`。
- 不新增 runtime / aclrt 调用，不改 kernel ABI、host ABI、TilingData schema。
- CPU E2E 用 910B 的 tikicpulib 验数值；真实 NPU 用例作为最终验收。

## 软件功能

1. 合法 RA `afir.sum` 的能力查询返回 supported，非法形态继续以原有 reason 拒绝。
2. RA 不发 `ReduceInit` 及其专属 `PIPE_V` barrier。
3. RA 发 `layout = RA`，`srcShape` 按**物理次序**给 `{R_actual, align 后的 A footprint}`。
4. 32B 对齐落在**物理最内**那根轴上——AR 是 R 的最内轴，RA 是 A 的最内轴。
5. 跨 tile 累加的 count 继续取 `align(A_actual)`；这一条 AR / RA 共用，S4 之后已经正确。
6. 能力查询与改写使用同一个谓词函数，不允许两边各写一份条件。

## 设计约束

- 不得通过删除门禁让未修正的 RA 流入 lowering；门禁只能换成更精确的条件。
- capability provider 与 conversion pattern 必须接受同一组形态（同一个函数）。
- `ReduceInit` 是否发射由 reduce pattern 决定，不通过芯片型号或框架名称判断。
- **物理次序**（外 / 内）与**语义 A/R** 必须在命名和结构上分开。今天 `Reduce.cpp` 的
  `firstActual` / `last` 是语义量，`ReducePlan::first/last` 也是语义量，而 legacy 的
  `first` / `last` 是**物理量**——三者同名不同义，是本方案最大的读代码陷阱。
- AR 现有生成 IR、kernel 文本和数值行为必须逐字节不变。
- 所有构建命令最多 `-j 8`。
- 失败方向必须是明确拒绝，不得回退到 AR 或交换维度猜测。

## 假设和依赖关系

| 假设 | 依据 | 风险 |
|---|---|---|
| 高阶 `ReduceSum` 的 RA 模板可用 | printer 已支持，legacy 同一 API 有 RA 分支 | 低 |
| RA 的 `srcShape` 与 AR 同一个表达式，因为 legacy 的 `first`/`last` 本来就是物理量 | `reduce_api_call.cpp:77` 对两种 pattern 都写 `{first_actual, last}` | 低 |
| RA 不需要 `ReduceInit` | `reduce_api_call_base.cpp:547` 的 `if (x.isAr)` | 低 |
| RA 的结果元素数是物理**内**维的 actual | `reduce_api_call_base.cpp:562`：Mean 的 `reduce_dim_a` 在 RA 下取 `last_actual` | 低 |
| A 维 padding 槽位产生的额外结果最终不会 Store 到 GM | store 的有效视图只覆盖 A_actual | 中，见 3.4 |
| 调度器会为「归约首维」产出 RA descriptor | **2026-09-23 S0 探针已证实**，见 6.3 | 已消除 |
| ATT 的 `CalcReduceTmpSize` 已按 AR/RA 分别算 scratch | legacy 同一份实现 | 中，由 E2E 验尺寸链 |

# 需求分析与设计

## 整体介绍

一句话：**把「语义 A/R」翻译成「物理 外/内」这一步，今天被写死成了恒等映射。**

```text
                 buildReducePlan
                        |
            first = A 轴组, last = R 轴组   （语义，已正确，RA 也对）
                        |
                        v
          ┌─────────  pattern  ─────────┐
          │                             │
       AR [A…,R…]                   RA [R…,A…]
  outer = first(A)                outer = last(R)
  inner = last(R)                 inner = first(A)
          │                             │
          └─────────────┬───────────────┘
                        v
        outerActual          = Π outer[i].actual
        innerFootprint       = align(inner.back().nominal) × Π inner[<back].nominal
        srcShape             = { outerActual, innerFootprint }
        resultActual         = Π first[i].actual          （恒取 A 组，与 pattern 无关）
                        |
          ┌─────────  pattern  ─────────┐
       AR: ReduceInit + PIPE_V       RA: 都不发
       layout = AR                   layout = RA
          └─────────────┬───────────────┘
                        v
            ReduceSum(srcShape, srcInnerPad=true)
                        |
        isAccumulating() ? Copy/Add(align(resultActual)) : —
```

今天的代码等价于把 `outer` 恒等成 `first`、`inner` 恒等成 `last`，在 AR 下二者重合，
所以看不出差别；RA 下就是三处一起错位（srcShape 两项对调、align 落在错的一组、
ReduceInit 写错地址）。

## 功能需求

### 功能需求 1：把语义 A/R 翻译成物理外/内

1. 介绍

   在 `Reduce.cpp` 建立一次性的物理形状视图，让后面所有喂给高阶 API 的量都从它取，
   消灭「在多个调用点各自判一次 pattern」的可能。

2. 输入

   - `ReducePlan`：`pattern`、`first.axes`（A 组，外→内）、`last.axes`（R 组，外→内）；
   - `materializeScheduledAxisExtents(op, builder, axes)` 给出的每轴 `{nominal, actual}`；
   - destination 元素类型（决定 `alignNum = 32 / sizeof(T)`）。

3. 处理

   `Reduce.cpp` 匿名 namespace 新增：

   ```cpp
   // 物理次序与语义 A/R 是两件事，只在 AR 下重合。
   //   AR 的布局是 [A…, R…]：物理外 = A 组，物理内 = R 组；
   //   RA 的布局是 [R…, A…]：物理外 = R 组，物理内 = A 组。
   // 高阶 API 的 srcShape、ReduceInit 的 dim_a / dim_r 全按**物理**次序取；
   // 跨 tile 累加的 count 按**语义** A 取。两套量必须分开命名，否则 AR 下换了看不出
   // 差别、RA 下就是静默错值。
   struct PhysicalReduceShape {
     Value outerActual;            // 物理外维的行数：恒取 actual
     Value innerFootprint;         // 物理内维的行距：align(最内 nominal) × 其余 nominal
     Value innerActualFootprint;   // 同上但取 actual，只有 AR 的 ReduceInit 用
     Value innermostInnerActual;   // 物理最内轴的 actual，只有 AR 的 ReduceInit 用
     Value resultActual;           // 语义 A 的元素数：累加 count 的来源
                                   // （2026-10-02 改名 `resultSpan`，语义改为槽位跨度，见 3.4）
   };

   FailureOr<PhysicalReduceShape> materializePhysicalReduceShape(
       Operation *op, OpBuilder &builder, const afir::ReducePlan &plan, int64_t alignNum);
   ```

   实现要点：

   - `outerAxes = isAR ? plan.first.axes : plan.last.axes`，`innerAxes` 取另一组；
   - `outerActual = multiplyAll(每根 outer 轴的 actual)`；
   - `innerFootprint = align(inner.back().nominal)`，再依次乘上其余 inner 轴的 `nominal`；
   - `innerActualFootprint`：inner 只有一根轴时是 `align(inner.back().actual)`，否则是
     `align(inner.back().nominal)`，再乘其余 inner 轴的 `actual`（**沿用今天 `lastActual`
     的写法，一行不改**，只是换了取哪一组轴）；
   - `resultActual = multiplyAll(每根 **first**（A）轴的 actual)`，与 pattern 无关。

   AR 下 `outerAxes == plan.first.axes`，于是 `outerActual` 与 `resultActual` 由同一组轴、
   同一段代码物化。**实施时必须让 AR 复用同一个 SSA Value 而不是算两遍**，否则 IR 会多出
   一串重复的 `arith.muli`，违反「AR 产物逐字节不变」。

4. 输出

   一次 lowering 内唯一可信的物理 shape 与结果长度。任一组轴物化失败即 `failure()`，
   由能力查询映射为 `UnsupportedShape`。

### 功能需求 2：按 pattern 决定 padding 与 layout

1. 介绍

   AR 的被归约维在物理内侧，对齐补出来的位置会被加进和里，必须先填单位元；RA 的被归约维
   在物理外侧，物理内维 padding 属于保留维 A，对应的额外结果槽位永远不 Store，不需要填。

2. 输入

   `ReducePlan::pattern`、`PhysicalReduceShape`、source、API scratch、destination。

3. 处理

   - AR：保持现有 `ReduceInitOp` 的五个参数与其后的 `PipeBarrierOp(PIPE_V)`，参数来源由
     `firstActual / last / lastActual / innerActual` 换成
     `outerActual / innerFootprint / innerActualFootprint / innermostInnerActual`——
     AR 下这四个值与今天逐一相等，产物不变；
   - RA：**不发** `ReduceInitOp`，也不发它后面那条 `PipeBarrierOp`；
   - 两者的 `srcShape` 都是 `{outerActual, innerFootprint}`，与 legacy
     `reduce_api_call.cpp:77` 的 `{first_actual, last}` 同形（legacy 的 first/last 是物理量）；
   - `layout` 由 `pattern == AR ? ReduceLayout::AR : ReduceLayout::RA` 决定，printer 不动；
   - 两者继续传 `srcInnerPad = true`、`isReuseSource = false`。RA 的物理内维 A 同样是
     「给了对齐后的长度」，这个参数说的是内维带 padding，与哪一维被归约无关；CANN 的参数
     检查对非 ArgMax 归约无条件断言它为 true（`reduce_check.h:93`，传 false 直接 abort）。

   **不采用**「给 RA 也改写一个 ReduceInit」：legacy 不调，且 RA 内维 padding 不参与任何
   有效 A 元素的归约，新增一套填充只增开销和风险。

4. 输出

   AR 生成结果逐字节不变；RA 生成
   `AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, false>`，其前没有 `ReduceInit`。

### 功能需求 3：跨 tile 累加保持不变

1. 介绍

   **本需求不需要改代码**，写在这里是为了销掉初稿的功能需求 3，并留下判据。

2. 依据

   S4 之前 `firstActual` 来自搬运计划的物理外维；S4 把它改成了
   `materializeScheduledAxisExtents(op, rewriter, plan->first.axes)` 的乘积，即**语义 A 轴组**
   的 actual 乘积。`emitAccumulation` 的 `count = align(firstActual)` 因此已经是 A，RA 下同样
   正确。legacy 的对照是 `reduce_api_call_base.cpp:562`：Mean 的 `reduce_dim_a` 在 AR 取
   `first_actual`、RA 取 `last_actual`，而 legacy 的 first/last 是**物理**量——两边换算之后
   说的是同一件事：**结果元素数恒为语义 A 的 actual**。

3. 处理

   把 `emitAccumulation` 的形参名 `firstActual` 改成 `resultActual`，传
   `PhysicalReduceShape::resultActual`。纯改名，AR 产物不变。

4. 输出

   RA 不会按 R 个元素 Copy/Add。lit 用一个 R ≠ A 的形状钉住 count 的 SSA 来源。

### 功能需求 4：门禁切换为同源谓词

1. 介绍

   RA 语义闭环后，两处 `pattern != AR` 的拒绝换成一个精确条件，且能力查询与改写调**同一个
   函数**——这是 2026-09-22 elementwise 方案上踩过的坑：两侧各写一份条件，会出现「能力说
   支持、改写因同一结构匹配失败」的分裂。

2. 输入

   `ReducePlan`。

3. 处理

   ```cpp
   // 能力查询与改写的唯一判据。RA 的 R 组限制成一根轴，理由见设计文档 3.3。
   bool isSupportedReduceLayout(const afir::ReducePlan &plan) {
     return plan.pattern == afir::ReducePattern::AR || plan.last.axes.size() == 1;
   }
   ```

   - `ReduceCapabilityProvider::queryLowering`：`pattern != AR` 的整段换成
     `!isSupportedReduceLayout(*plan)` → `UnsupportedShape`；
   - `ReduceSumConversionPattern::matchAndRewrite`：同样换掉，**保留这道重复检查**——
     本 pattern 可以不经 SelectImplementation 被单独驱动，漏判的后果是发出布局错误的
     kernel 而不是少一条诊断（两维寻址设计 5.3 对这种情形的规定）；
   - 其余检查（`f32`、UB 驻留、scratch、target feature）一律不动；
   - 实现 ID 继续是 `reduce_sum.tensor_ar_ra`，此时名称与能力终于一致。

4. 输出

   合法 RA 返回 supported；R 组多轴的 RA、非 f32、非 UB、scratch 不足继续得到原有诊断。

### 功能需求 5：形成 RCore phase 2 前置证明

1. 介绍

   RCore phase 2 固定构造 `[Rm, A]`，reduce 前 stride `[A, 1]`、reduce 后 `[0, 1]`，必然是
   RA 且 R 组恰好一根轴——正好落在本期开放的范围内。

2. 处理

   - 增加 bridge/调度集成测试，断言 phase 2 的 descriptor 被 `buildReducePlan` 判成 RA 且
     `last.axes.size() == 1`；
   - 断言其 lowering 发 `layout = RA` 且没有 `ReduceInit`；
   - **不**启用完整 RCore plan，也不宣称 phase 1 / workspace launch 已完成——workspace
     的寻址链路在 kernel 侧整段缺失，见 3.8。

3. 输出

   「RA 已满足 RCore phase 2 lowering」的可执行证据，为 RCore 专项移除一个硬阻塞。

## 非功能需求

### 可维护性

- 物理量用 `outer*` / `inner*`，语义量用 `result*`，禁止再出现一个同时表示两者的名字。
- AR/RA 只在三个集中点分支：`materializePhysicalReduceShape` 选轴组、ReduceInit 的发射
  条件、layout 映射。
- `ReducePlan.h` 中 `first` / `last` 的注释里「RA 目前在能力查询处被拒（见设计文档 6.1
  的 L3）」这段会变成错的，必须同步改写——它正是那天要用上的东西。
- `Reduce.cpp:100-109` 与 `:228-232` 两大段解释「为什么拒绝 RA」的注释要换成「为什么
  这样翻译物理次序」，不能留着误导下一个人。
- 不复制 legacy 的字符串生成逻辑，不往 `TransferPlan` 里加 Reduce 专用字段。

### 可测试性

- `ReducePlan` 已有 RA 的 gtest（`ReducedLeadingAxisIsRa`），本期补「RA + 多根 R 轴」的
  形状断言，供门禁用例引用。
- lowering 由 lit 检查：op 顺序、`layout`、`srcShape` 两项的 SSA 来源、`ReduceInit` 的有无、
  累加 count 的来源。
- 数值必须由 CPU E2E 和至少一次真实 NPU 验证，FileCheck 不能替代。
- AR 的「逐字节不变」用已有的 kernel diff 基线机制验证（S5 用过：22 个用例 ×
  `kernel.cpp` + `autofuse_tiling_data.h` = 44 份）。

### 可移植性

- 不硬编码 SoC；能力仍通过 `TargetProfile` / execution spec 查询。
- 本期边界是当前 memory-based `f32` 实现。其他 target 若 feature unavailable，保持原诊断。
- 不新增 CANN 私有 API。

### 可靠性

- 所有失败在改写前被 capability 拒绝，或在 conversion 中 `notifyMatchFailure` 终止，不生成
  半条 Reduce 序列。
- 不改变 LocalTensor / scratch / queue / workspace 的所有权和生命周期。
- RA 不发 `ReduceInit` 是防止越界写的必要修正（`reduce_init.h` 的填充按行主序写死，
  RA 下被归约维是首维，填充地址必然落错位置）。
- `alignUp` / `multiplyChecked` 的 `mulsi_extended` + `cf.assert` 溢出检查一条不减。

### 特性交叉影响

| 场景 | 适用性 | 分析说明 |
|---|---|---|
| SuperKernel Python 接口 | 不适用 | 不修改 SuperKernel、Python API 或 wheel |
| SuperKernel C++/AOT 接口 | 不适用 | 不修改 `libascendsk.so` 或 AOT 接口 |
| Autofuse 图优化 | 适用 | RCore phase 2 的 descriptor 是验收输入；本方案不改调度图结构 |
| Autofuse Codegen/Backend | 适用 | 核心修改位于 `Reduce.cpp` 的 capability 与 lowering |
| AscendC API / Runtime 交互 | 适用 | 使用既有 ReduceSum RA 模板；不新增 runtime 调用或生命周期 |
| Python/C++ 混合绑定 | 不适用 | 不修改 pyautofuse 接口和异常传播 |
| 构建与打包 | 不适用 | 不修改 CMake 安装、run 包、第三方依赖和产物路径 |
| 测试与覆盖率 | 适用 | 新增 gtest、lit 正反例、CPU E2E、RCore phase 2 集成与 NPU 验证 |
| 性能与日志 | 适用 | RA 少一次 `ReduceInit` 与一条 barrier；不新增日志 |
| 兼容性 | 适用 | AR 必须逐字节不变；RA 从明确拒绝变为受约束支持 |

## 性能

### 编译时长

新增一次 pattern 分支与少量 SSA value 选择，复杂度 O(1)。`buildReducePlan` 与
`materializeScheduledAxisExtents` 的调用次数不增加，不新增全图遍历，影响可忽略。

### 执行性能

- AR 指令序列逐字节不变。
- RA 相比「按 AR 发射」的错误路径少一次 `ReduceInit` 与一条 `PIPE_V` barrier，与 legacy 一致。
- ReduceSum 主体与跨 tile 累加次数不变。
- 不新增默认日志、GM 搬运或 kernel launch。

### 内存和产物大小

- `PhysicalReduceShape` 只在编译期栈上存放五个 `Value`，生命周期为单次 pattern rewrite。
- 不新增 device buffer、workspace 或 TilingData 字段。
- RA 路径的 kernel 源码比 AR 少一条 helper 调用和一条 barrier；AR 文本保持不变。

## 接口设计

### 新增/修改接口描述

| 接口 | 变更 | 说明 |
|---|---|---|
| `PhysicalReduceShape` | 新增，`Reduce.cpp` 匿名 namespace | 集中保存物理 shape 与语义结果长度，不形成对外 ABI |
| `materializePhysicalReduceShape` | 新增，同上 | 复用 `materializeScheduledAxisExtents` / `alignUp` / `multiplyChecked` |
| `isSupportedReduceLayout` | 新增，同上 | capability 与 conversion 的唯一共同判据 |
| `emitAccumulation` | 形参更名 | `firstActual` → `resultActual`，仍是私有成员函数（2026-10-02 随 3.4 的修复再改名 `resultSpan`，语义由元素数改为槽位跨度） |
| `ReduceCapabilityProvider::queryLowering` | 行为扩展 | 合法 RA 由 unsupported 变 supported |
| `ReduceSumConversionPattern` | 行为扩展 | 条件发 ReduceInit，按 pattern 选 layout 与物理 shape |

不修改 ODS、公开 AFIR op、kernel ABI、host ABI、TilingData schema 或安装接口。

### 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|---|---|---|---|
| 接口说明 | 是否需要接口评审 | 否 | 全部为内部 lowering 实现 |
| 接口说明 | 是否需要补充文档 | 是 | 本文档、`ReducePlan.h` 与 `Reduce.cpp` 的注释 |
| 接口兼容 | 行为是否兼容 | 是 | AR 逐字节不变；RA 从拒绝变为受约束支持 |
| 接口兼容 | ABI/API 是否兼容 | 否 | 无导出接口与结构体布局变化 |
| 接口约束 | 约束不满足时是否清晰报错 | 是 | 继续映射为 `UnsupportedShape` / `UnsupportedType` / `TargetFeature*` |
| 接口测试 | 是否需要独立接口用例 | 是 | capability / lit 正反例与 E2E |

## 软件设计

### 3.1 关键数据结构

`PhysicalReduceShape` 是 lowering 内部的非拥有视图：五个成员都是当前 function 中的 SSA
`Value`，不延长生命周期、不缓存到 pass 之外。它存在的唯一理由是把「pattern → 物理次序」
这次翻译收敛到一个地方。

### 3.2 关键技术/算法：物理次序与对齐落点

| Pattern | 物理布局 | 物理外 | 物理内 | `srcShape` | 累加 count | `ReduceInit` |
|---|---|---|---|---|---|---|
| AR | `[A…, R…]` | A 组 | R 组 | `{Π A_actual, align(R_inner_nom) × Π R_outer_nom}` | `align(Π A_actual)` | 需要 |
| RA | `[R…, A…]` | R 组 | A 组 | `{Π R_actual, align(A_inner_nom) × Π A_outer_nom}` | `align(Π A_actual)` | 不需要 |

三条判据的出处：

1. **srcShape 两项对调**。legacy 对 AR 与 RA 写的是同一行
   `uint32_t tmp_reduce_shape[] = {first_actual, last};`（`reduce_api_call.cpp:77`），而它的
   `first`/`last` 由 `ReduceMergedSizeCodeGen` 按**轴的物理顺序**分组产生——前导那一组进
   `first`，尾随那一组进 `last`。所以 legacy 从来不需要对调，是 MLIR 这边的 `first`/`last`
   取了语义含义，才必须显式翻译。
2. **对齐只落在物理最内轴**。`ReduceMergedSizeCodeGen` 只在 `i == num_axes - 1` 那一支写
   `SizeAlign(axis_size, 32/sizeof(T))`，其余轴直接相乘。这对应 UB 里「每一行按 32B 对齐、
   行与行之间可能有缝」的布局：行距就是最内轴对齐后的长度。AR 下最内是 R，RA 下最内是 A。
3. **外维恒取 actual，内维恒取 nominal**。内维给 nominal 是因为它是**行距**，行距不随尾块
   变化（缓冲按标称分配）；外维给 actual 是因为它是**行数**，尾块下必须少走几行。AR 之所以
   还需要 `ReduceInit`，正是因为它的被归约维是内维——内维给了 nominal，超出 actual 的那一段
   就必须先填成单位元；RA 的被归约维是外维，给的已经是 actual，天然没有这个问题。

### 3.3 为什么本期把 R 限制成一根轴

`srcShape` 的外维是一个**扁平行数**，API 按固定行距逐行走。多根外维轴时，只有当除最外一根
之外的每一根都「actual == nominal」时，这些行才真正连成一片；否则中间隔着尾块留下的空行，
扁平行数会把空行也算成数据行。

- **AR 下这条前提今天已经存在**：外维是 A 组，`outerActual = Π A_actual`。ARAR 的四个
  tiling case 之所以全绿，是因为尾块只落在**最内**的 tile 轴（R 组最内），而那一根由
  `ReduceInit` / `innerActualFootprint` 专门处理。
- **RA 会把这条前提扩展到 R 组**，而 R 组的错法更糟：AR 多算一行只是多算了几个 A 结果槽
  （最终不 Store），RA 多算一行是**把垃圾行加进了有效 A 结果的和里**。

要彻底解决，得先让「这根轴在这个 op 处会不会取尾块」变成编译期可判定的（今天
`materializeScheduledAxisExtents` 只给 SSA `Value`，比不了）。那是一件独立的事。

本期的处置：**RA 要求 `plan.last.axes.size() == 1`**。一根 R 轴时行距恒为内维 footprint，
行必然连续，`outerActual = R_actual` 无条件成立。RCore phase 2 的 `[Rm, A]` 正是一根 R 轴，
覆盖了驱动场景。R 组多轴的 RA 返回 `UnsupportedShape`，并由一条 lit 负例钉住。

### 3.4 A 组内层轴带尾块（本方案内只记录；2026-10-02 已单独修复）

`emitAccumulation` 的 `count = align(Π A_actual)` 假设 A 的结果槽位在 destination 里**连续**。
A 组多轴且内层轴带尾块时，有效槽位是跨步的，连续 count 会数错位置。

本方案按红线 7「禁止无理由扩大修改范围」**只记录不修改**，并在测试设计里留一条探测用例，
用来判断它是真缺陷还是被调度器的轴划分排除掉了。

**2026-10-02 探测结论：是真缺陷，且只在 RA 下成立；已作为独立在册阻塞修复。**

初稿把它记成「**AR** 今天已有的隐含前提」。探测证明这条记录是错的，AR 不受影响：

- **AR 的 destination 连续**。AR 布局是 `[A…, R…]`，对齐补白落在物理内维 R 上，与 A 无关；
  高阶 API 把 `Π A_actual` 个结果连续写在 destination 开头。所以 `align(Π A_actual)` 正是对的。
  既有的 `load_sum_store_arar` 注册就是 A 组两根轴、内层 A 为 4（`align(4) = 8`，不对齐）的
  语料，它修前修后都过——这就是 AR 连续的实证。
- **RA 的 destination 跨步**。RA 布局是 `[R…, A…]`，A 是物理内维，destination 镜像源的内维
  布局：槽位 `(a_outer, a_inner)` 落在 `a_outer × align(A_inner_nominal) + a_inner`。
  这里才会漏掉每个 tile 尾部的几行。

复现（`load_sum_store` 的三维 ARA 注册，钉 plan 0 = RCommon）：

| 形状 | 每 tile 的 A_outer | 旧 count | 真实跨度 | 修前错在 |
| --- | --- | --- | --- | --- |
| `[64, 32, 5]` | 3 | `align8(3×5) = 16` | `5 + 2×8 = 21`，对齐后 24 | A_outer = 2, 5, 8, … 整行（槽位 16..20 没被覆盖） |
| `[64, 8, 5]` | 5 | `align8(5×5) = 32` | `5 + 4×8 = 37`，对齐后 40 | A_outer = 4, 9, 14, … 整行（槽位 32..36 没被覆盖） |

小形状压不出来：`[4, 32, 5]` / `[8, 256, 3]` / `[16, 16, 12]` 修前就全过，因为 UB 预算把每
tile 的 A_outer 压成 1 行，此时跨步与连续恰好等价。这也是为什么这个缺陷此前从没被测到过。

修法与初稿预判的方向一致：`PhysicalReduceShape::resultActual`（唯一的两个消费点都是
count）改名 `resultSpan`，语义从「A 的元素数」改成「A 槽位在 destination 里的**跨度**」，
按 `materializeScheduledLocalSpan` 同一套公式物化：

    resultSpan = A_inner_actual + Σ_outer (actual - 1) × stride
    stride(第 i 根 A 轴) = align(A_inner_nominal) × Π_{j>i, j<内} A_j_nominal

跨度里夹着的补白槽位会跟着被累加 / 被 scale 成垃圾值，但它们最终不 Store 回 GM——这与
`load_sum_store_ra_32x17` 守的是同一条性质。跨度上界是缓冲 nominal 行距之积，对齐后不越界。

**两处退化保证既有产物逐字节不变**：AR 下 `resultSpan` 仍直接复用 `outerActual` 这个 SSA 值；
RA 下 A 组单轴时求和的循环一次都不进，`resultSpan` 就是 `inner->back().actual`，而旧代码的
`multiplyAll` 对单元素也是原值返回。实测 lit 190/190 条数与断言均未变，ctest 由 90 增到 92
（新增的就是上表两条回归用例）。

### 3.5 流程设计

正常流程：

1. capability 建 `ReducePlan`，`isSupportedReduceLayout` 过滤；
2. 校验 `f32`、UB 驻留、scratch、target feature；
3. conversion 调 `materializePhysicalReduceShape` 物化物理 shape 与 `resultActual`；
4. AR 发 `ReduceInit` + `PIPE_V`，RA 跳过；
5. 按 pattern 发 `ReduceSum2DL2Op`；
6. 有累加轴时按 `align(resultActual)` 首轮 Copy、其余轮 Add；
7. `replaceOp`。

异常流程：任一计划、轴值、descriptor、scratch 或 execution spec 缺失即 `failure()`；不生成
半条 Reduce 序列，不把 RA 改按 AR 发射，不交换维度猜测。

### 3.6 对子模块的修改

| 文件 | 修改点 |
|---|---|
| `conversion/AFIRToAscendC/Implementations/Reduce.cpp` | 新增 `PhysicalReduceShape` / `materializePhysicalReduceShape` / `isSupportedReduceLayout`；条件发 `ReduceInit`；layout 按 pattern；两处门禁换同源谓词；`emitAccumulation` 形参更名；改写两大段过期注释 |
| `dialect/AFIR/Analysis/ReducePlan.h` | **仅注释**：`first`/`last` 上「RA 目前在能力查询处被拒」那段改写为物理/语义翻译规则的指引 |
| `test/ut/dialect/ReducePlanTest.cpp` | 补「RA + 两根 R 轴」的计划形状断言，供门禁负例引用 |
| `test/lit/conversion/afir-to-ascendc-reduce-sum-invalid.mlir` | 改造成 RA **正例**并改名为 `afir-to-ascendc-reduce-sum-ra.mlir` |
| `test/lit/conversion/afir-to-ascendc-reduce-sum-ra-rejected.mlir` | 新增：R 组两根轴的 RA 仍被拒 |
| `test/e2e/cpu/cases/load_sum_store/` | 新增 RA 形状用例（**前提是 S0 探针确认调度器产得出 RA**） |
| `docs/design/2026-09-20-two-dim-transfer-addressing-design.md` | S5 条目 7/8 的 RA 阻塞说明随本方案落地后销号 |
| `docs/README.md` | 本文档状态由「待实施」改为「已完成」 |

### 3.7 错误处理

#### 系统错误

不新增内存申请、文件、runtime 资源和异步调用。MLIR builder 失败沿既有 `FailureOr` /
`LogicalResult` 返回，不引入回滚或额外清理路径。

#### 接口错误

| 条件 | 结果 |
|---|---|
| descriptor / plan 缺失 | `MissingInformation` |
| `buildReducePlan` 失败（轴形状不对、scratch 不够） | `UnsupportedShape` |
| RA 且 R 组多于一根轴 | `UnsupportedShape` |
| 非 `f32` | `UnsupportedType` |
| input / destination 不在 UB | `notifyMatchFailure`（现状不变） |
| target feature unavailable / 未建模 | `TargetFeatureUnavailable` / `TargetFeatureNotModeled` |
| conversion 被单独驱动且计划不合法 | `notifyMatchFailure`，不得发错误 kernel |

### 3.8 记录但不处理：workspace 链路的现状与 RCore 的真实缺口

2026-09-23 顺着「RA 是 RCore phase 2 的前置」查了一遍 workspace 的实现，当日晚些又对着
legacy 逐段核了一遍，**本节据此重写过**：初版记的四条里有两条是误判，下面标注了更正。
结论是 **MLIR 侧不阻塞 RCore 的 tiling 链路**，但 kernel 侧的读取整段缺失。按红线 7 只记录，
完整方案见 [2026-09-23-mlir-workspace-offset-chain-design.md](2026-09-23-mlir-workspace-offset-chain-design.md)。

#### 现状：三条口径不同的链路

| | 来源 | 挂在哪 | 谁消费 | 实际用途 |
|---|---|---|---|---|
| `afir.workspace_plan` | `PlanInfo::workspace_plan`，由 `AppendWorkspacePlan` 从 **fused 级**的 `fusedResult.workspace_nodes` 构造（`ScheduleToAfir.cpp:387`） | **每个** template func（`AfirBuilder.cpp:1535` 对每个 tmpl 传同一份） | `CanonicalBindings::validateWorkspacePlan` 只校验；`kWorkspaceTensorIdsAttrName` 取 `bufferId` | **只用了 `bufferId`**，在 TilingData 里生成 `workspace<ID>` 字段 |
| `afir.workspace_size` | `TemplateInfo::workspace_size`，由 `InitializeTemplateInfo` 从 **本 impl graph** 的 Workspace 节点算出（`ScheduleToAfir.cpp:712`），无节点则为 `"0"` | 每个 template func | `TilingProgramSpec` → `TilingFuncEmitter::AppendStageWorkspace` | 生成 `CalculateWorkspaceSize`，得到传给 host 的**总量** |
| `workspace<ID>` 字段值 | ATT 的 `workspace_size_map`，由 `AscendGraphParser::ParseWorkspaceNode`（`ascend_graph_parser.cpp:612`）按 `tensor_id` 取 `Max` | `AutofuseTilingData` 根层 | ATT 生成的 tiling func 自己 Reset / Update / Finalize | 存**偏移**（Finalize 之前存的是尺寸） |

#### 逐条结论

1. **`size_bytes_expr` 从来没人读**（初版判断成立）。`workspace_plan` 里的 size 表达式只被
   `validateWorkspacePlan` 检查「存在且非空」，没有任何地方拿它算过大小。但它不是「缺了一
   段」——ATT 自己从 ASC 图上用 `CalculateOneWorkspaceSize` 算尺寸，MLIR 侧用
   `CalculateWorkspaceSize` 算总量，两处都不需要这个表达式。它是**冗余**，不是缺口。

   顺带：`workspace_size` 的自由符号有校验（`TilingProgramSpec` 要求每个符号都在
   `stage->fields` 里），而 `workspace_plan` 里的 `A_org_size` / `Rm_org_size` 不经过这道检查。
   若将来改成消费 plan 的 size 表达式，这道校验要一并补上。

2. **更正（初版记的「两者已经不一致，而且发现不了」不成立）。** 初版观察到 reduce 语料里
   `workspace_plan` 声明 `buffer_id = 2`、`size_bytes_expr = "(4 * A_org_size * Rm_org_size)"`
   而 `workspace_size = 0`，把它记成了不一致。实际是**两者口径不同**：`workspace_plan` 取自
   fused 级的 `workspace_nodes`（整个融合结果里存在这块 workspace），`workspace_size` 取自
   本 template 自己的 impl graph（这个 template 不用它）。两个值各自都对。

   留下来的事实只有一条：**给用不到 workspace 的 plan 也生成了 `workspace<ID>` 死字段**。
   它不占执行开销，但会让任何「schema 声明了就应该有人写」式的校验产生误报——设计
   workspace 一致性门禁时必须按**消费端**（谁真的绑定了这块 GM）驱动，不能按 schema 驱动。

3. **更正（初版记的「跨 stage 是求和，不是取并集」不是 MLIR 引入的缺陷）。**
   `TilingFuncEmitter::EmitWorkspaceHelper`（`TilingFuncEmitter.cpp:278`）是 legacy
   `TilingLib::GenGetWorkspaceSizeFunc`（`codegen_tiling.cpp:1017`）的忠实移植：同样逐
   impl graph 累加、同样由该图的 `tiling_key` 守卫、同样 512 对齐，连尺寸表达式都来自同一个
   `ascgen_utils::CalculateWorkspaceSize`。

   两条路径在「多个 impl graph 共用同一块 workspace tensor」时都会把总量算重。总量偏大、
   偏移正确，是**共有的保守超配**，不是越界风险。本期钉住这个口径，不改。

4. **更正（初版记的「`workspace<ID>` 字段没人赋值」是语料结论，不是架构结论）。**
   赋值由 ATT 生成的 tiling func 完成，三段协议：`ResetWorkspaceSizes` 全置 0 →
   每个子图 `set_workspace<ID>(max(已有, workspace_map[ID]))` → `FinalizeWorkspaceOffsets`
   按固定 id 序做前缀和，把**尺寸改写成偏移**（`tiling_code_gen_impl.cpp:3524-3573`）。

   这段在 **MLIR 路径里已经在跑**：MLIR 把 ATT 产物原样拼接
   （`TilingFuncEmitter.cpp:378` `AssembleSources`），而 `AutofuseTiling` 调的正是
   `optiling::GetTiling`（`TilingFuncEmitter.cpp:368`）。字段名两侧一致（都是 `workspace<ID>`），
   作用域也一致（`AutofuseTilingData` 根层：MLIR 侧 `GenAggregateTilingData.cpp:116` /
   `GenTilingData.cpp:50`，schema 侧 `tiling_field_schema.cpp:211` 的 `"fusion"`）。

   当前语料里字段确实没被写过，原因是 RCommon 没有 `Workspace` 节点 → `workspace_size_map`
   为空 → ATT 生成的 `SetWorkspaceSize` 是 `(void)tiling_data; (void)workspace_map;`
   （`tiling_code_gen_impl.cpp:2264`），`GenWorkspaceOffsetHelpers` 也整段跳过。

#### 真正的缺口：kernel 侧的读取

legacy 在 `AppendWorkspaceTensorInit`（`codegen_kernel.cpp:2087`）把 workspace 张量绑到
`workspace + workspace<ID>`，落成指针算术（`codegen_kernel.cpp:521`）：

```cpp
name.SetGlobalBuffer((__gm__ T*)((__gm__ uint8_t*)(workspace) + (workspace2)));
```

MLIR 侧**没有任何对应物**，而且缺口比这更靠前：`Workspace` 节点在 `ScheduleToAfir` 里根本
没有映射成 AFIR 值（见 3.10）。所以 RCore 的 workspace 问题是**一件事**——表示 + 绑定 + 读取
的同一条链路，不是四件互不相干的风险。

#### 对 RA 验收的影响

「真实 RCore phase 2 `[Rm, A]` 产物可以完成 RA lowering」这条验收标准只覆盖 **lowering**，
不覆盖 workspace 的寻址。RA 用例全绿**不能**推出 RCore 可以整体开放：kernel 侧的偏移绑定
必须在 RCore 专项里单独实现并实测。

### 3.9 记录但不处理：RCore plan 卡在 tiling_key 的命名分歧上（已解决）

> **2026-10-02 复测：这道门已经没了。** `72eb7d23 fix(mlir): TilingData 的 schedule group
> 命名与布局对齐 ATT` 把命名约定统一掉了。现在钉 `AUTOFUSE_DFX_FLAGS="--force_schedule_result=1"` 的 RCore 用例
> 能一路编过 `ValidateGeneratedFieldReferences`，CPU E2E 产物与数值都出来了，见下面 3.10 的
> 复测补注。本节以下内容保留为当时的现场记录。


S4 实测（6.7）发现的：**RCore plan 今天根本走不到 lowering**，它在更早的 tiling schema
交叉校验处就失败了，和 RA 无关。

调度器确实产出了 RCore plan——phase 1 / phase 2 两张图都完成了 AutoSchedule 与 DoTiling，
最终结果里有 `..._r_multicore_phase_1_graph_B1Y1R0_S1G0C0` 和
`..._r_multicore_phase_2_graph_B0Y1R0_S1G1C0` 两张 `AutoFuseAfterOptimize` 产物。MLIR 侧
`BuildAllCandidateSelections` 也能把它列成 plan 1。挡住它的是这一句：

```
ATT generated tiling_func references field(s) not present in TilingFieldSchema:
graph0_tiling_key tiling_key
```

出处是 `bridge/AttSchemaAdapter/AttSchemaAdapter.cpp:130` 的 `ValidateGeneratedFieldReferences`。
根因在两侧对「多 stage plan 的 tiling_key 叫什么」各有一套命名：

- MLIR 侧 `common/tiling_schema/tiling_field_schema.cpp:178` 的 `is_flat` 要求「一个
  subgraph、一个 plan、一个 stage」三者同时成立。RCore 被 narrow 成唯一 plan 之后，
  `schedule_groups.size() == 2` 仍然让 `is_flat` 为假，于是字段名走
  `graph<sub>_result<plan>_tiling_key`（同文件 `:147`），各 stage 还多一层
  `graph0_result0_tiling_data.g<N>_tiling_data` 嵌套路径；
- ATT 生成的 tiling_func 引用的却是 `graph0_tiling_key` 和裸的 `tiling_key`。

两边都不认对方的拼法，所以 `PrepareMlirCandidates` 直接失败，AFIR module 压根没被构造出来。

这道校验本身是对的：它防的正是「ATT 与 MLIR 对同一个 TilingData 结构体各写各的字段名」
这种一旦放过就只会在运行期读错偏移的静默错位。要修的是命名约定，不是这道校验。

按红线 7 只记录。

> **2026-09-24 更正**：这条当时被写成「排在 RA 之后的下一个硬阻塞」，不准确。它只是**最先
> 撞到**的那一个，因为 ATT 那一步跑在 `PlanToAfir` 前面。真正排在最前面的是 3.10 的
> `Workspace` 节点。

### 3.10 RCore 前置探针（2026-09-24）：第一道门是 `Workspace` 节点，不是 tiling_key（已解决）

> **2026-10-02 复测：两道门都已拆除，RCore 现在能完整下降。** `Workspace` 的节点级处理由
> `a43b8511` / `cb5af127` / `c33ece68` / `46051008` 一串提交补齐：它已经进了
> `ScheduleToAfir.cpp` 的豁免名单（`:983`），并按本节末尾说的「同类、不同角色」解析到既有的
> `role = workspace` 参数上（`:877`）。
>
> 实测（容器 CPU E2E，钉 plan 1）：
>
> | 注册 | 形状 | 产物 | 数值 |
> | --- | --- | --- | --- |
> | `load_sum_store_rcore_ar_8x1024` | `[8, 1024] -> [8, 1]` | `stage0_template0` + `stage1_template0`，含 workspace，**同时**出 `Pattern::Reduce::AR` 与 `::RA` | PASS |
> | `load_sum_store_rcore_ra_1024x8` | `[1024, 8] -> [1, 8]` | 同上两个 stage，两相都是 `::RA` | PASS |
>
> AR 输入同时出两种 pattern，正是 3.3 说的「RA 是 RCore phase 2 的硬前置」：phase 1 在
> `[A, Rm_t]` 上是 AR，phase 2 在 workspace 的 `[Rm, A]` 上是 RA。
>
> **但这两条绿不是跨核归约的证据**：CPU E2E 这个规模下 block_dim = 1，`Ceiling(nR/a1Tt) = 1`，
> phase 2 只合并一份 partial。跨核合并要 nA < corenum 才触发，只能在 910B 上测——由
> `test/e2e/npu/reduce_template_matrix.py` 的 `rcore_ar` / `rcore_ra` 两个形态承担。
>
> 本节以下内容保留为当时的现场记录。


> 注意与 6.3 的「S0 探针」区分：那一条探的是 RA 本身，这一条探的是 RCore 的前置清单，
> 不属本方案的实施范围，按红线 7 只记录。

**方法**（两处临时改动，探完即删，均未提交）：

1. 临时注册一条 `[8192, 128] -> [1, 128]` 的 e2e 用例，确认 `AUTOFUSE_DFX_FLAGS="--force_schedule_result=1"`
   就是 RCore plan（plan 0 是 RCommon，plan 2 不存在）；
2. 临时用环境变量把 `ValidateGeneratedFieldReferences` 短路掉，看放过 3.9 之后下一步撞到
   什么。

**结果**：`PlanToAfir` 仍然失败，AFIR module 仍然产不出来，但原因换成了更靠前的一个：

```
error: UnsupportedOperation: Schedule buffer-form bridge does not support node
'..._r_multicore_phase_2_graph_workspace' of type 'Workspace'
```

**真实的 phase 2 图**（调度器产物，`AutoFuseAfterOptimize` dump，本期唯一能钉住的语料）：

```
func @..._phase_2_graph_B0Y1R0_S1G1C0(
    %workspace: f32[Rm_org_size, A_org_size],
    %output:    f32[1, A_org_size]) {
  for %ATB in ATB_size:
    for %ATb in ATb_size:
      for %RmT in RmT_size:
        %1 = Load(%workspace) -> vector<Rmt_size x At_size x f32>
        %2 = Sum(%1)          -> vector<1 x At_size x f32>
      %3 = Store(%2) -> %output
}
```

phase 2 的 `Workspace` 节点**扮演的是图输入**，与 `Data` 同类：它不产生 AFIR op，只产生一个
值。而 `ScheduleToAfir.cpp:873` 的豁免名单是 `{Data, Output, Scalar}`，漏了 `Workspace`；
`Workspace` 也不在 `isAFIRBufferFormOperationSupported` 那张表里，于是落进「不支持的节点」。

注意 `ScheduleToAfir` 并非完全不认识 `Workspace`：它在 **plan 级**认（`ToWorkspaceDescriptor`
把 `fusedResult.workspace_nodes` 变成 `afir.workspace_plan`，`InitializeTemplateInfo` 数
template 图里的 Workspace 节点算 `workspace_size`）。缺的只是**节点级**——一个出现在
template 图里、当输入用的 `Workspace` 节点。

**顺带确认的两件事**：

- phase 2 的 reduce 输入轴实测是 `{Rmt, At}`、stride `{8 * Ceiling(At_size / 8), 1}`，**恰好
  一根 R 轴**，与 6.7 那条照 `RMulticorePhase2Graph` 构造代码重建的 UT 形状一致——重建是对的；
- **ABI 的枚举与参数已就绪，偏移绑定没有**：`KernelArgRole::Workspace` 早就在枚举里
  （`AFIREnums.td:221`），`afir-kernel-body-signature` 也已经无条件给每个 kernel body 追加了
  workspace 参数（`AFIRKernelBodySignature.cpp:234-238`）。缺的是把一个带 `buffer_id` 的 GM
  张量**解析成这个参数上的偏移**——今天没有任何机制做这件事，见下面「为什么不能直接当
  `Data` 映射」。

#### 为什么不能直接当 `Data` 映射

`Workspace` 与 `Data` 在**节点分类**上确实同类：产生值、不产生 op，所以它该进
`AppendTemplateNode` 的豁免名单，而不该被 `isAFIRBufferFormOperationSupported` 那张
**op 表**拦住——那张表收的是 Load / Store / Add / Sum 这类会建出 AFIR op 的节点。这一半是
三行的事。

但 `Data` 这条路的终点是 `setKernelArgAttr(func, idx, KernelArgRole::Input, input.index)`
（`AfirBuilder.cpp:1242`），也就是变成**前端图的第 N 个输入**。落到 workspace 上有四个问题：

1. 前端图没有这个输入，host launch 按前端输入个数传参，参数个数对不上；
   `verifyCanonicalKernelArguments` 要求 `afir.arg_descs` 与外部数据参数一一对应，也会当场不过。
2. 同一块 GM 会出现两次：`afir-kernel-body-signature` 仍会在末尾追加真正的 workspace 参数，
   而 `getCanonicalTilingArgument` 还要校验「workspace 必须是倒数第二个」
   （`AscendCKernelABI.cpp:232-236`）。
3. `KernelArgRole::Input` 的 `index` 语义是「它在前端图输入里的序号」，workspace 没有这个序号。
4. **跨 stage 的身份会断**。实测 phase 1 的 Store 目标与 phase 2 的 Load 来源是**同一个**
   `Workspace` 节点（名字都是 `..._phase_2_graph_workspace`）：

   ```
   // phase 1
   func @..._phase_1_graph_B1Y1R0_S1G0C0(%input0: f32[8192,128], %workspace: f32[]) {
     ... %2 = Sum(%1); Store(%2) -> workspace: %workspace
   ```

   也就是说同一个节点在 phase 1 是 `Output`、在 phase 2 是 `Data`。按 Data/Output 映射，两侧
   会各自变成一个独立的 kernel 参数，维系「同一块 GM」的 `buffer_id` 就丢了。

所以正确的形状是**同类、不同角色**：节点分类按 `Data` / `Output` 走（按它在本 stage 里的方
向），但 kernel 参数不进 `tmpl.inputs` / `tmpl.outputs`，而是解析到那个已有的
`role = workspace` 参数上、按 `buffer_id` 取偏移。

**第二半今天完全没有机制**：`afir.workspace_tensor_ids` 的唯一消费者是 `GenTilingData` /
`GenAggregateTilingData`（生成 TilingData 里的 `workspace<ID>` 字段），它不是绑定机制。

按 3.8 更正后的结论，偏移这条链路**缺的是读者这一段**：字段有（MLIR 生成）、值也有（ATT
生成的 tiling func 自己 Reset/Update/Finalize，MLIR 原样拼接它的产物），唯独没有任何地方把
GM 张量绑到 `workspace + workspace<ID>`。当前语料里字段看起来没被写过，是因为 RCommon 没有
`Workspace` 节点使 `workspace_size_map` 为空，不是机制缺失。

#### RCore 前置清单（按实际撞到的顺序重排）

| # | 阻塞 | 位置 | 备注 |
|---|---|---|---|
| 1 | `Workspace` 节点没有 AFIR 映射 | 节点分类：`ScheduleToAfir.cpp` 的 `NodeOutputRole`(:792)、`AppendNodeIo`(:834)、`AppendTemplateNode`(:873)；参数绑定：无 | 分类是三行；**绑定到 workspace 参数的偏移链路三段全缺**，与第 4 条是同一条链 |
| 2 | 多 stage plan 的 tiling_key 命名有三套 | 3.9，外加 `GenKernelEntry.cpp:237` 自己拼的第三套 | |
| 3 | ~~`ascendc.need_sync_all` 只有消费者没有生产者~~ **已闭环（2026-09-24）** | 生产者补在 `AFIRKernelBodySignature.cpp` 的 `markNeedSyncAll`，判据与 legacy `codegen/codegen_kernel.cpp:2748` 同源（`has_workspace_node && 非末 stage`）；消费端 `GenKernelEntry.cpp` 不变 | phase 1 写 workspace、phase 2 读，缺这道全核同步就是数据竞争。真实 RCore plan 现在在 stage 0 的 `tiling_key` 分支里发 `AscendC::SyncAll();`，与 legacy 同位同形 |
| 4 | workspace 偏移的 kernel 侧读取缺失 | 3.8；方案见 [2026-09-23-mlir-workspace-offset-chain-design.md](2026-09-23-mlir-workspace-offset-chain-design.md) | 与第 1 条是同一条链：表示 + 绑定 + 读取 |

**S0 的结论是「AFIR 层的语料现在钉不成」**：RCore plan 产不出 AFIR module，lit 层没有输入可
钉。今天能钉的只有上面那份调度器侧的 phase 2 图。要拿到 AFIR 层语料，必须先做第 1 条。

## 安全检查

按 `编码红线.md` 逐条：

| 红线 | 结论 |
|---|---|
| 1 硬编码敏感信息 | 无。不新增密钥、账号、公网地址 |
| 2 外部输入作索引/长度 | extent 与 count 继续来自已校验的 descriptor / tiling SSA 值；`plan.last.axes.size()` 是本地容器长度 |
| 3 整数溢出 | `alignUp` / `multiplyChecked` 的 `mulsi_extended` + `cf.assert` 一条不减；新增的乘法全部走这两个 helper，不裸用 `muli` |
| 4 资源释放 | 不改 LocalTensor / scratch / queue / workspace 的所有权与生命周期 |
| 5 内存申请 | 不新增内存申请 |
| 6 未定义行为 | 无指针运算、无求值顺序依赖 |
| 7 扩大修改范围 | 3.4 的 A 组尾块问题**只记录不修改** |
| GA-1 API/ABI | 无导出接口、结构体布局、枚举值变化 |
| GA-2 硬编码芯片类型 | 无。能力仍走 `TargetProfile` / execution spec |
| GA-3 图改写等价性 | 不增删 AFIR 数据边/控制边，只改合法 RA 的 AscendC op 序列 |
| GA-4 pass 时序 | 不新增 pass，不调整顺序 |
| GA-5 改写确定性 | pattern 映射是确定的枚举分支；轴组来自 `ReducePlan` 的有序 `SmallVector`，不依赖无序容器或指针序 |
| GA-6 高频日志 | 不新增日志 |
| GA-7 runtime 生命周期 | 不新增 `rt*` / `aclrt*` 调用 |
| GA-8 非开放接口 | 无 |

两条**必要的安全修正**：RA 不发 `ReduceInit` 防的是越界写；RA 的 `srcShape` 外维取 R 的
actual 而非 nominal 防的是把垃圾行读进和里。

## 兼容性检查

- AR：lowering IR、生成 kernel 文本、E2E 数值**必须逐字节不变**，用 44 份产物 diff 验证。
- RA：从编译期明确拒绝变为受约束支持，是本需求的预期变化。
- R 组多轴 RA、非 f32、非 UB、target 不可用：行为不变，继续拒绝。
- 无脚本参数、配置、安装目录、动态库符号、Python 包或 run 包变化。
- `reduce_sum.tensor_ar_ra` 名称不变；开放后名实相符。
- RCore plan 选择保持现状，不因 RA 用例通过就宣称模板整体可用。

## 测试设计

### 测试边界

| 层 | 入口 | 验证目标 |
|---|---|---|
| ReducePlan UT | `AFIRDialectTest --gtest_filter='ReducePlanTest.*'` | RA 判定、R 组轴数、非法布局 |
| lowering lit | `af-opt --afir-kernel-body-signature --afir-to-ascendc-pipeline` | layout、srcShape 两项来源、ReduceInit 有无、累加 count 来源 |
| CPU E2E | `check-e2e-cpu` | RA 数值、R/A 尾块、跨 tile 累加、scratch 尺寸链 |
| 产物 diff | 生成 kernel 快照比对 | AR 44/44 逐字节不变 |
| bridge 集成 | RCore phase 2 真实产物 | `[Rm, A]` 可进入 RA lowering |
| NPU E2E | 910B 实机 | 真实设备数值与 API 行为 |

不 mock ReduceSum 数值；CPU E2E 用现有 tikicpulib。计划层 UT 继续用 `parseSourceString`。

### 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|---|---|---|---|
| 前置 | **调度器对「归约首维」是否产出 RA** | dump AFIR，看 `afir.sum` destination 的 `vectorized_strides` 首项是否为 `"0"` | 探针（S0，见 6.1） |
| 功能 | RA lowering 不发 `ReduceInit` 及其 barrier | FileCheck `CHECK-NOT` 与 op 顺序 | lit |
| 功能 | RA 发 `layout = RA` | 检查 `ReduceSum2DL2Op` 的 layout 属性 | lit |
| 功能 | RA 的 `srcShape` 外项来自 R 的 actual、内项来自 align 后的 A nominal | 取 R ≠ A 的形状，钉两项的 SSA 来源 | lit |
| 功能 | RA 累加 count 来自 `align(A_actual)` 而非 R | 同上形状，钉 `DataCopyL2` / `AddL2` 的 count 来源 | lit |
| 功能 | R 尾块 `[37, 16]` reduce axis 0 | 与 host reference 比较 | CPU E2E |
| 功能 | A 尾块 `[32, 17]` reduce axis 0 | 验证内维 padding 不污染有效输出 | CPU E2E |
| 功能 | R/A 双尾块 `[37, 17]` reduce axis 0 | 组合边界 | CPU E2E |
| 功能 | 多 R tile（R 大到需要跨轮累加） | 首轮 Copy、后续 Add 后数值正确 | CPU E2E |
| 功能 | RCore phase 2 `[Rm, A] -> [1, A]` | 调度器真实产物进入 lowering | 集成 + lit |
| 异常 | RA + R 组两根轴 | 仍报 `UnsupportedShape` | UT + lit |
| 异常 | 两维都归约 / 都保留 | `buildReducePlan` failure（已有用例） | UT |
| 异常 | 非 f32、非 UB、scratch 不足 | 保持既有 reason | UT/lit |
| 探测 | **A 组两根轴 + 内层轴带尾块** | 与 host reference 比较；失败即证实 3.4 是既有缺陷 | CPU E2E |
| 回归 | `load_sum_store_ra_agap_64x32x5` / `_64x8x5` | 上一行的探测**已于 2026-10-02 证实是缺陷并修复**（见 3.4）；原记作「AR」有误，缺陷只在 RA 下成立 | CPU E2E |
| 兼容性 | AR `16x32` / `16x37` / `17x32` / ARAR case0~3 | 现有 lit 与 E2E 全绿 | lit/E2E |
| 兼容性 | AR 产物逐字节不变 | 44 份 `kernel.cpp` + `autofuse_tiling_data.h` diff | 快照比对 |
| 性能 | RA kernel 无 `ReduceInit` | 比对生成 kernel 指令序列 | lit |

### 测试命令

```bash
# 进容器（本机开发环境）
AF_CANN_VOLUME=af-cann-910 ./autofuse/mlir/scripts/enter_dev_container.sh bash -lc '<cmd>'

# 0) 防呆：容器中途被杀会留下 0 字节 .o/.a，伪装成最新
find /workspace/build \( -name "*.o" -o -name "*.a" \) -size 0 -delete

# 1) ReducePlan 定向 UT
cmake --build /workspace/build --target AFIRDialectTest -j 8
/workspace/build/autofuse/mlir/test/ut/dialect/AFIRDialectTest --gtest_filter='ReducePlanTest.*'

# 2) 全量 UT
cmake --build /workspace/build --target check-ut -j 8

# 3) lit（必须指向 build 树；指源码树会 AttributeError）
cmake --build /workspace/build --target af-opt -j 8
llvm-lit -sv /workspace/build/autofuse/mlir/test/lit

# 4) CPU E2E（lit 不在 ctest 里，两者都要跑）
cmake --build /workspace/build --target check-e2e-cpu -j 8
ctest --test-dir /workspace/build/autofuse/mlir

# 5) S0 探针：dump 每个 pass 的 MLIR，确认 afir.sum 的 destination 首个
#    vectorized_stride 是否为 "0"
AUTOFUSE_DFX_FLAGS="--codegen_compile_debug=true;--debug_dir=/workspace/ra_probe" \
  <运行 load_sum_store 的生成阶段>
```

预期结果：UT / lit / ctest / check-e2e-cpu 全绿，无 warning，无超时；AR 的 44 份产物与
基线逐字节相同。实施阶段必须记录**实际执行的命令**与通过/失败数，不得只写"全绿"。

# 6 实施计划

## 6.1 分期

| 期 | 内容 | 是否改变 AR 产物 | 出口判据 |
|---|---|---|---|
| ~~**S0**~~ | ~~探针：确认调度器会不会为「归约首维」产出 RA descriptor~~ | 否（不改代码） | **已完成（2026-09-23），结论见 6.3** |
| ~~**S1**~~ | ~~`materializePhysicalReduceShape` + `isSupportedReduceLayout`，门禁仍拒 RA~~ | 否 | **已完成（2026-09-23），见 6.4** |
| ~~**S2**~~ | ~~条件发 `ReduceInit`、layout 按 pattern、门禁换同源谓词~~ | 否 | **已完成（2026-09-23），见 6.5** |
| ~~**S3**~~ | ~~RA 的 CPU E2E 数值用例~~ | 否 | **已完成（2026-09-23），见 6.6** |
| **S4** | RCore phase 2 集成验证 + NPU 实测 | 否 | phase 2 产物完成 RA lowering；910B 数值正确 —— **部分完成（2026-09-23），见 6.7**：descriptor 层已证；真实 phase 2 产物被 3.9 的 tiling_key 分歧挡住；NPU 本环境无设备 |

S1 单独成期的价值：它是纯重构，产物必须逐字节不变，这一条能把「翻译写错了」和
「RA 本身有问题」分开定位。

## 6.2 S0 已定的走向

探针证实调度器会自然产出 RA（6.3），因此走「成本低」的那一支：**S3 的 CPU E2E 直接在
`load_sum_store` 上按 `reduce_axes = {0}` 注册形状即可**，不需要为了造 RA 去搭 RCore
phase 2 的图。phase 2 的集成验证回到它原本的位置（S4），作为 RCore 前置证明而不是
E2E 语料的唯一来源。

## 6.3 S0 探针结论（2026-09-23）

**做法**：临时给 `load_sum_store` 注册一条 `[16, 32] -> [1, 32]`、`reduce_axes = {0}` 的
用例，只构建它的 `*_generator` 目标（不构建数值测试），用
`AUTOFUSE_DFX_FLAGS="--codegen_compile_debug=true;--debug_dir=…"` dump 每个 pass 的 MLIR。
探针的三个改动（`Case.cpp` 选轴、`CMakeLists.txt` 注册、`Reduce.cpp` 短路门禁）在取得
结论后全部回滚，并已核对与 HEAD 逐字节一致。

### 结论 1：调度器自然产出 RA

`00_afir-kernel-body-signature.mlir` 里 `afir.sum` 的 descriptor：

| 侧 | `vectorized_axis` | `vectorized_strides` |
|---|---|---|
| `destination_descs` | `[5, 3]` | `[<"0">, <"1">]` |
| `operand_descs` | `[5, 3]` | `[<"(8 * Ceiling((Rational(1 , 8) * z1t_size)))">, <"1">]` |

destination 首项为 `"0"` → `startsReduced` → `ReducePattern::RA`。且 **A 组与 R 组各恰好
一根轴**（`first.axes = [3]`、`last.axes = [5]`），正落在 3.3 限定的范围内——本期的 R 组
单轴限制不会把真实语料挡在外面。

### 结论 2：RA 的全部阻塞就是那两道门禁

- 不改任何代码：`error: 'afir.sum' op has no supported Ascend implementation`，
  `06_ascend-select-implementation.mlir` 里 `afir.sum` 没有被标上实现。
- **只**把 `Reduce.cpp` 的两处 `pattern != afir::ReducePattern::AR` 短路掉、其余一律不动：
  代码生成**通过**（generator 用例 `[ OK ]`）。

也就是说 RA 不缺任何基础设施：`buildReducePlan`、scratch 分配（dump 里 `afir.sum` 带
`scratch(%57, %59)` 两块）、轴值物化、printer 全都是现成的。

### 结论 3：门禁放开后的（错误）kernel 逐条印证了 3.2 的三项预测

```cpp
ReduceInit<float, kReduceOpSum, true>(v130, v108, v132, v138, v114);   // ① 不该发
AscendC::PipeBarrier<PIPE_V>();
uint32_t v143_reduce_shape[] = {v144, v145};                            // ③ 两项对调
AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(         // ② layout 错
    v143, v130, v81, v143_reduce_shape, true);
```

`srcShape` 两项的来源，顺着 SSA 追下去：

| 值 | 定义 | 是什么 |
|---|---|---|
| `v144 = (int32)v108` | `v108 = v107 ? v14 : v85`（尾块选择） | **A_actual** |
| `v145 = (int32)v132` | `v131 = (v18 + 7) / 8`、`v132 = v131 * 8` | **align(R_nominal)** |

正是「AR 的公式套在 RA 的布局上」：给出 `{A_actual, align(R_nominal)}`，而 RA 要的是
`{R_actual, align(A_nominal)}`。32B 对齐也落在了 R 上，而 RA 的物理最内轴是 A。

### 结论 4：累加 count 已经取语义 A，功能需求 3 确认无需改代码

> **2026-10-02 补注**：本结论的标题初稿写的是「累加 count 已经正确」，**说大了**。S0 探针只
> 证明了 count 取的是语义 A 而不是物理外维 R，这一点仍然成立；它没有覆盖 A 组多于一根轴的
> 情形，而那时 destination 里的 A 槽位是跨步的，`align(Π A_actual)` 会漏掉每个 tile 尾部几行。
> 探针用的 `[37, 16]` 是 A 组单轴，恰好测不到。该缺陷见 3.4，2026-10-02 已修。

```cpp
uint32_t v147 = (v108 + 8 - 1) / 8;   // v108 = A_actual
uint32_t v148 = v147 * 8;             // align(A_actual)
int32_t  v153 = (int32_t)v148;
if (v146) { AscendC::DataCopy(v110, v143, v153); }
else      { AscendC::Add(v110, v110, v143, v153); }
```

count 取的是 `align(A_actual)`，与 R 无关——S4 把 `firstActual` 改由语义 A 轴组物化之后，
这一条对 RA 天然成立。初稿的功能需求 3（「按 A 维执行跨 tile 累加」）**确认作废**，只剩
一个形参改名。

## 6.4 S1 实施记录（2026-09-23）

**改了什么**（只 `Reduce.cpp` 一个文件，+106 / -47）：

- 新增 `isSupportedReduceLayout(const ReducePlan &)`，**本期语义仍是 `pattern == AR`**，
  能力查询与改写两处都改调它——先把「同源」这件结构性的事做掉，S2 只需要改这个函数的函数体。
- 新增 `PhysicalReduceShape` 与 `materializePhysicalReduceShape`，把原先散在改写体里的
  `firstActual` / `last` / `lastActual` / `innerActual` 收敛成
  `outerActual` / `innerFootprint` / `innerActualFootprint` / `innermostInnerActual`
  四个**物理**量，外加一个**语义**量 `resultActual`。
- `emitAccumulation` 的形参由 `firstActual` 改名 `resultActual`（2026-10-02 随 3.4 的修复再改名 `resultSpan`）。
- 改写了两段已经过期的注释：原注释说「累加 count 取 `first_actual`，它来自 `blockCount`，
  是 tile 的物理外维」——S4 之后不成立，count 已经是语义 A。

**保证 AR 逐字节不变的两个着力点**：

1. 物化顺序按**物理**外、内而不是语义 A、R。AR 下 outer 组就是 first 组，`materializeScheduledAxisExtents`
   的两次调用顺序与改动前完全一致，发出的表达式 IR 因此一一对应。
2. `resultActual` 在 AR 下**复用** `outerActual` 这个 SSA 值，不重新 `multiplyAll` 一遍。
   重算会多出一串等价的 `arith.muli`，产物就不再逐字节相同。

**验证**：

| 项 | 结果 |
|---|---|
| `AFIRToAscendC` 编译 | 通过，无 warning |
| **AR 产物 diff** | **44 / 44 逐字节不变**（22 个用例 × `kernel.cpp` + `autofuse_tiling_data.h`） |
| `make check-e2e-cpu` | rc=0，无 FAILED |
| `make check-ut` | rc=0，无 FAILED |
| lit | 125 / 125 |
| `ctest --test-dir build/autofuse/mlir` | 33 / 33 |

**已知的、S2 才会消除的一点**：`materializePhysicalReduceShape` 里 `if (!isAR)` 那一支
（RA 的 `resultActual` 单独 `multiplyAll`）当前打不到——门禁把 RA 拦在外面。这是分期的
代价，也是分期的价值：S1 是纯结构、产物可证不变，S2 只剩三处行为改动（门禁条件、
`ReduceInit` 的发射条件、layout 映射），两类风险不会混在一次 diff 里。

## 6.5 S2 实施记录（2026-09-23）

**三处行为改动**（`Reduce.cpp`）：

1. `isSupportedReduceLayout` 的函数体由 `pattern == AR` 改为
   `pattern == AR || last.axes.size() == 1`。S1 已经把两处门禁并到这一个谓词上，所以放开
   RA 只动了这一行。
2. `ReduceInit` 与其后的 `PIPE_V` barrier 只在 AR 下发。
3. `layout` 由 `pattern` 选。

**一处 S1 没预见到的修正**：`innerActualFootprint` / `innermostInnerActual` 只有 AR 的
`ReduceInit` 用得到，但它们由 `alignUp` / `multiplyChecked` 物化，而这两个 helper 会带出
`cf.assert`——**有副作用，DCE 掉不了**。RA 下不发 `ReduceInit`，这一串就成了 kernel 里每一
轮白跑的 ceildiv + mul + 两次 cmp + and + assert。改成只在 AR 下物化。第一版跑出来的 IR
里能看到这段死算，是读 dump 时发现的，不是推出来的。

**测试**：

| 文件 | 变化 |
|---|---|
| `test/lit/conversion/afir-to-ascendc-reduce-sum-ra.mlir` | **新增正例** |
| `test/lit/conversion/afir-to-ascendc-reduce-sum-ra-rejected.mlir` | **新增负例**：R 组两根轴仍被拒 |
| `test/lit/conversion/afir-to-ascendc-reduce-sum-invalid.mlir` | **删除**：它钉的是「RA 一律拒绝」，已被上面两条取代 |
| `test/ut/dialect/ReducePlanTest.cpp` | 新增 `RaWithMultipleReducedAxesStillBuildsAPlan` |

两条 lit 的输入都是**真实调度产物**，不是手改 stride。旧的 `-invalid.mlir` 只翻转了
`vectorized_strides` 没翻 `strides`，同一根轴在两处一个说被归约、一个说保留——做拒绝用例
无所谓（走不到下降），做正例就是钉一个永远不会出现的形状，所以它被删掉而不是改造。

新增的 UT 与那条负例 lit 是配对的：能力查询把「计划建不出来」和「轴数不支持」都映射成
`UnsupportedShape`，光看诊断分不出来；UT 证明这个形状的计划**建得出来**（pattern 是 RA、
`last.axes` 有两根），于是 lit 里那次拒绝只可能来自轴数。

正例 lit 钉住的四件事，都按 `#afir.axis_binding` 的轴 id 锚定，不依赖 SSA 编号：

| 断言 | 值 |
|---|---|
| 没有 `ascendc.reduce_init` | 用 `--implicit-check-not` 覆盖全文件 |
| `layout` | `1`（RA） |
| `srcShape` 外项 | 轴 5（R）的 **actual**，未对齐 |
| `srcShape` 内项 | 轴 3（A）**nominal** 的 32B 对齐 |
| 累加 count | 轴 3（A）**actual** 的 32B 对齐，**不是** R（A 组单轴下这就是对的；多轴见 3.4） |

**验证**：

| 项 | 结果 |
|---|---|
| **AR 产物 diff** | **44 / 44 逐字节不变** |
| `make check-e2e-cpu` | rc=0，无 FAILED |
| `make check-ut` | rc=0，无 FAILED；`ReducePlanTest` 13 / 13 |
| lit | **126 / 126**（较 S1 的 125 净 +1：新增两条、删除一条） |
| `ctest --test-dir build/autofuse/mlir` | 33 / 33 |

**尚未验证的**：RA 的**数值**。lit 只证明发出的 op 与参数来源对，不证明算得对。那是 S3
（CPU E2E）和 S4（910B 实机）的事，验收标准里对应的几条仍未勾。

## 6.6 S3 实施记录（2026-09-23）

RA 的**数值**验收。S2 之前只证明了发出的 op 和参数来源对，这一期证明它算得对。

**三条用例**（都在 `test/e2e/cpu/cases/load_sum_store/`，走 CPU tikicpulib）：

| 用例 | 形状 | 压的边界 |
|---|---|---|
| `load_sum_store_ra_37x16` | `[37, 16] -> [1, 16]` | R 尾块 |
| `load_sum_store_ra_32x17` | `[32, 17] -> [1, 17]` | A 尾块（A=17 不按 32B 对齐） |
| `load_sum_store_ra_37x17` | `[37, 17] -> [1, 17]` | R / A 双尾块 |

三条都实测走到跨轮累加（生成的 kernel 里有 `DataCopy` / `Add` 各一处），所以设计里单列的
「多 R tile」不需要第四条用例——R=37 本来就被切成多个 tile。

A 尾块那条压的是 S2 最不直观的一条论断：RA 不发 `ReduceInit`，A 方向对齐补出来的槽位里
是**未初始化的垃圾**，靠的是「它们只产生额外结果槽、最终不 Store 回 GM」。这条用例如果
数值错了，那条论断就是错的。

**归约轴的选法没有新增注册参数**，而是由已有的 `OUTPUT_SHAPE` 决定：`[M, 1]` 归约末维
（`[A, R]`，AR），`[1, N]` 归约首维（`[R, A]`，RA）。建图（`Case.cpp`）与参考实现
（`KernelTest.cpp`）读同一个来源，不会各说各话。

**顺带补强的一处断言**：`CheckGeneratedKernel` 原先无条件要求生成的 kernel 内联了
`reduce_init.h`——那是 AR 专用的。现在按同一条判据分开：AR 路径保持原样，RA 路径反过来
断言 kernel 里**没有** `ReduceInit<`，两条路径都额外检查 `Pattern::Reduce::{AR,RA}` 模板
参数与注册的输出形状一致。这条断言第一次跑就把三条 RA 用例挡下来了，说明它是有效的。

**验证**：

| 项 | 结果 |
|---|---|
| 三条 RA 用例数值 | 全部通过 |
| RA kernel 形态实测 | `Pattern::Reduce::RA`、`ReduceInit` 计数 0、带 Copy/Add 累加 |
| **AR 基线 diff** | **44 / 44 逐字节不变** |
| `make check-e2e-cpu` | rc=0，无 FAILED，32 个用例通过 |
| `make check-ut` | rc=0，无 FAILED |
| lit | 126 / 126 |
| `ctest --test-dir build/autofuse/mlir` | **36 / 36**（较 S2 的 33 增 3） |

**仍未验证的**：910B 实机数值与设备编译（S4），以及 RCore phase 2 真实产物进入 RA lowering
的集成验证。CPU E2E 用的是 tikicpulib 的软件模拟，不能替代实机。

## 6.7 S4 实施记录（2026-09-23）

S4 的两半都**没有**按原计划完成，原因各不相同，都不是 RA 本身的问题。下面逐条说清楚做到了
哪一步、被什么挡住、以及退而求其次给出的是什么证据。

### 做成的：descriptor 层的 phase 2 前置证明

新增 UT `ReducePlanTest.PhaseTwoDescriptorIsRaWithASingleReducedAxis`。它的输入不是跑出来
抓的，是**照着构造 phase 2 的那段代码抄的**——
`optimize/task_generator/reduce_schedule_case_generator.cpp` 的 `RMulticorePhase2Graph`：

| 来源 | 内容 |
|---|---|
| `CreateVarAxis()`（`:924`） | 建两根轴：`Rm`(id 0, size `Rm_org_size`)、`A`(id 1, size `A_org_size`) |
| `CompleteNodeAttr(..., before_reduce=true)`（`:940`） | `strides = {A_org_size, 1}`、`repeats = {Rm_org_size, A_org_size}` |
| 同上 `before_reduce=false` | `strides = {0, 1}`、`repeats = {1, A_org_size}` |

被归约的恒是**首维** `Rm`，而且恒**只有一根**。所以 phase 2 不是「碰巧可能是 RA」，是构造上
只能是 RA，且必然落在本期开放的「R 组单轴」范围内（3.3）。UT 断言 `pattern == RA`、
`last.axes == {0}`、`first.axes == {1}`——其中 `last.axes.size() == 1` 正是 `Reduce.cpp`
的 `isSupportedReduceLayout` 放行 RA 的那条判据。

`vectorized_*` 在 phase 2 构造完成时还不存在（由后续 tiling 产生），UT 里按未切分的二维 UB
张量补成两根轴全向量化。这一段是**重建**，不是抓取，UT 的注释里写明了。

下降侧（`layout = RA`、不发 `ReduceInit`）由 `test/lit/conversion/afir-to-ascendc-reduce-sum-ra.mlir`
钉住，那条用例的输入是真实调度产物的 `[R, A]`，与 phase 2 同形。没有再手搓一份「phase 2 形状」
的 lit——手搓一个没跑出来过的调度产物，正是本方案开头删掉那条无效负例的原因。

### 没做成的之一：真实 phase 2 产物进不了 lowering

不是「调度器不产 RCore」。用一条临时注册的大 R 用例（`[8192, 128] -> [1, 128]`，探完即删）
打开 `AUTOFUSE_DFX_FLAGS` dump，能看到完整的 RCore 链路：

```
ascgraph_00008_..._r_multicore_phase_2_graph_phase2graph_construct_0.txt
ascgraph_00012_..._r_multicore_phase_1_graph_B1Y1R0_AfterDoTiling_0.txt
ascgraph_00014_..._r_multicore_phase_2_graph_B0Y1R0_AfterDoTiling_0.txt
ascgraph_00016_..._r_multicore_phase_1_graph_B1Y1R0_S1G0C0_AutoFuseAfterOptimize_0.txt
ascgraph_00017_..._r_multicore_phase_2_graph_B0Y1R0_S1G1C0_AutoFuseAfterOptimize_0.txt
```

phase 1 / phase 2 都完成了调度与 tiling 并进了最终结果，MLIR 侧也能把它列成 plan 1。
`AUTOFUSE_DFX_FLAGS="--force_schedule_result=1"` 失败在更早的地方，与 reduce 下降无关，详见 **3.9**：ATT 与 MLIR
对「多 stage plan 的 tiling_key 叫什么」各有一套命名，交叉校验因此报错，AFIR module 根本没被
构造。修它要动 tiling schema 的命名约定，按红线 7 只记录。

历史结论（已失效）：当时「真实 RCore phase 2 产物完成 RA lowering」尚不可达，
它的前置不是 RA，而是 3.9。验收框保持未勾选。

### 历史未完成项之二：910B 实机（后续已完成）

本环境无 NPU：`/dev/davinci*` 不存在，`npu-smi` 不在 PATH，`/usr/local/Ascend/driver` 不存在。
NPU 数值与设备编译两条验收框保持未勾选，需要在有卡的环境补。

### 顺带修正一条探针方法学（影响本期此前的判读）

直接裸跑 `MlirE2eCpu_*_generator` 是**错的**，会得到假结论。它必须带上 ctest 给的两样东西
（`test/e2e/cpu/AddCpuCase.cmake:173-181` 的 custom command 里就是这么调的）：

- `AF_ENABLE_MLIR_CODEGEN=1`——不带它 codegen 直接返回失败，看起来像「这个 plan 不存在」；
- ctest 的 `LD_LIBRARY_PATH`（`build/.../CTestTestfile.cmake` 里那一长串，**仓库自建的
  `libaihac_codegen.so` 等排在最前**）。容器默认的 `LD_LIBRARY_PATH` 让 CANN 9.2.0 预装的
  `/usr/local/Ascend/cann-9.2.0/lib64/libaihac_codegen.so`（2026-02 的发布构建）胜出，
  与仓库头文件混用会在 `Optimizer::Optimize` 里段错误——表现为「某些形状一跑就崩」，
  很容易误读成调度器缺陷。

正确的裸跑姿势：

```bash
D=/workspace/build/autofuse/mlir/test/e2e/cpu/cases/<套件>
LP=$(grep -oE 'LD_LIBRARY_PATH=[^"]*' $D/CTestTestfile.cmake | head -1 | cut -d= -f2-)
cd $D/generated/<用例>
AF_ENABLE_MLIR_CODEGEN=1 LD_LIBRARY_PATH=$LP \
  AUTOFUSE_DFX_FLAGS="--force_schedule_result=1" \
  $D/MlirE2eCpu_<用例>_generator
```

要看诊断再加 `ASCEND_SLOG_PRINT_TO_STDOUT=1 ASCEND_GLOBAL_LOG_LEVEL=2`——`PrepareMlirCandidates`
的失败原因只走 `GELOGW`，不开日志时整条信息是看不见的。

### 验证

| 项 | 结果 |
|---|---|
| `AFIRDialectTest`（含新增 phase 2 UT） | 通过 |
| 新增用例定向执行 | `PhaseTwoDescriptorIsRaWithASingleReducedAxis` OK |

本期只新增一条 UT，不动产品代码，AR 产物不受影响。

# 验收标准

- [x] S0 探针有明确结论并记录在本文档（见 6.3）。
- [x] 合法二维 f32 RA（R 组一根轴）在 capability 阶段返回 supported。
- [x] RA lowering 发 `layout = RA`，且 `ReduceSum` 前没有 `ReduceInit` 及其 `PIPE_V` barrier。
- [x] RA 的 `srcShape` 为 `{Π R_actual, align(A_inner_nominal) × Π A_outer_nominal}`，
      两项的 SSA 来源由 lit 钉住（本期 R 组限一根轴、语料 A 组也只有一根，所以公式里的
      连乘部分尚无用例覆盖）。
- [x] RA 跨 tile Copy/Add 的 count 来自 `align(Π A_actual)`，不来自 R。
- [x] R 组多于一根轴的 RA 仍返回 `UnsupportedShape`，由 UT 与 lit 各钉一条。
- [x] `[37,16]`、`[32,17]`、`[37,17]` 与多 R tile 的 CPU E2E 数值正确（三条用例都走到跨轮累加）。
- [x] phase 2 的 descriptor 被判成 RA 且 R 组恰好一根轴，由 UT
      `PhaseTwoDescriptorIsRaWithASingleReducedAxis` 钉住（输入照 `RMulticorePhase2Graph`
      的构造代码重建，见 6.7）。
- [x] 真实 RCore phase 2 `[Rm, A]` 产物可以完成 RA lowering（只覆盖 lowering；workspace
      的寻址见 3.8，不由本条担保）。
      —— 初稿记作「本期做不到」，前置是 3.9 的 tiling_key 命名分歧；**该前置与 3.10 的
      `Workspace` 节点都已被先前提交拆除，本条于 2026-10-02 补验通过**：
      `conversion/afir-to-ascendc-reduce-rcore.mlir` 用 `load_sum_store_rcore_ar_8x1024` 钉
      plan 1 的真实 dump 钉住两个 stage，其中 stage 1（phase 2）就是 `[Rm, A]` 上的 RA
      （`layout = 1`，且**不发** `ReduceInit`）；数值由 `load_sum_store_rcore_{ar,ra}` 两条
      CPU E2E 与 910B 实机 `rcore_ar` / `rcore_ra` 各 3/3 覆盖。见 3.10 的复测补注。
- [x] 至少一个 910B NPU RA 用例数值正确，设备编译通过。
      —— **2026-10-02 在 910B 实机完成**：`test/e2e/npu/reduce_template_matrix.py` 的
      `--form ra`（钉 plan 0 = RCommon）三条全过：`(37,16)` R 尾块、`(32,17)` A 尾块、
      `(37,17)` 两边都带尾块。`--form ara`（钉 plan 3 = AllLoad）的 `(4,32,8)` 也过，
      该条是 RA 下 **A 组连乘**的首个实机语料。
- [x] AR 既有 lit / UT / CPU E2E 全绿，**44 份产物逐字节不变**（S3 实测，见 6.6）。
- [x] 3.4 的「A 组内层轴带尾块」探测用例有明确结论：**失败，是真缺陷**，且只在 RA 下成立
      （初稿记作 AR 有误）。未开单——按「一个在册阻塞一次提交」当场修复，见 3.4；回归用例
      `load_sum_store_ra_agap_64x32x5` / `_64x8x5`。
- [x] 不修改 ABI、TilingData schema、打包路径，不新增日志和 runtime 调用。
- [x] `ReducePlan.h` 与 `Reduce.cpp` 中「RA 被拒」的注释已改写；
      `2026-09-20-two-dim-transfer-addressing-design.md` 的 S5 条目 7/8 已改写为新的阻塞项
      （910B / RCore tiling_key / AllLoad），RA 不再列为其阻塞。**2026-10-01 追记**：AllLoad 实测本来就能编，
      已从阻塞里销掉，并顺带给本文 3.3 的 `srcShape` 公式补上了 **A 组连乘**的首个语料
      （三维 ARA，见 [AllLoad 模板设计](2026-10-01-afir-reduce-allload-template-design.md)）。

### 设计文档检查结果

- [x] 跨特性交叉影响：已按 `cross_feature_check.md` 逐项检查；影响 Autofuse Codegen/Backend、
      AscendC API 发射、图优化输入与测试，不影响 SuperKernel、Python/C++ 绑定与打包。
- [x] 编码红线：已按 `编码红线.md` 逐条检查（见「安全检查」表）。重点：新增乘法全部走
      `multiplyChecked` / `alignUp` 的溢出检查；3.4 的既有问题按红线 7 只记录不顺手修。
- [x] 测试设计：包含 S0 探针、ReducePlan UT、正反 lit、CPU E2E 数值、AR 产物逐字节 diff、
      RCore phase 2 集成与 NPU 验证。
- [x] 性能评估：AR 指令序列逐字节不变；RA 少一次 `ReduceInit` 与一条 barrier；无新增搬运、
      launch 或日志。
