# 广播缓存（brc_cache）与内联广播（brc_inline）问题清单与修复方案

> 2026-11-09 · 代码基线 `d28990bd6`（develop，工作区干净）
> 上游材料：[elementwise 遗留问题清单](2026-09-22-afir-elementwise-remaining-issues.md) 的 BR-1、
> [测试临时调整与恢复条件](2026-11-08-test-adjustments-and-reenable.md) 中 `LegacyGraphGuardRegSt.ContinuesBrc`
> 的「T4 缺少 cache guard」一条。

# 简介

## 目的

本文记录对 MLIR 侧广播缓存（`afir.sched` 的 `exec_condition`）与内联广播（legacy `brc_inline` 的对等能力）
的一次完整代码评审结果：五个已复现的问题、每个问题的修复方案，以及分批修复计划。目标读者是后续承接这
五项修改的开发者与检视者。

每个问题都配有一个可独立运行的复现用例，位置与命令见「证据附录」。本文不记录未经复现的推测；评审中被
推翻的两条假设单列一节，避免后续重复排查。

## 范围

- 包含：`autofuse/mlir` 内 `exec_condition` 的 lowering（`ExecConditionPass`）、谓词构造
  （`materializeExecConditionGuard`）、调度轴解析（`buildScheduledPlan`），以及 binary elementwise
  的广播内联能力判定与 lowering。
- 不包含：`afir.broadcast` 自身的三档 lowering（Duplicate / Segmented / Extend）——本次评审未发现其
  判据缺陷；不包含 legacy 侧 `brc_inline_template` 与 `node_cache_marker` 的任何修改。
- 涉及组件：Autofuse（MLIR codegen 链路）。不涉及 SuperKernel。

# 总体概述

## 软件概述

### 项目介绍

`exec_condition` 是上游 autoschedule 对「可缓存算子」的标注：被标注算子的输出值在一个广播复用周期内不
变，周期起点计算一次，其余迭代直接读上一次留在 UB 中的结果。MLIR 侧由 `ExecConditionPass` 把这句话翻
译成循环体内的一个 `scf.if`。

`brc_inline` 是 legacy optimize 阶段的一个模板：它删除 Broadcast 节点、给消费者打上 `kInline` 语义角色，
由 codegen 发射 `src_repeat_stride = 0` 的 elementwise 调用，从而省掉一次广播物化与一块 UB 缓冲。MLIR
侧目前没有对等能力。

### 产品环境介绍

两项能力都只作用于编译期：`ExecConditionPass` 产出 `scf.if`，最终落到 AscendC kernel 源码里的 `if`
语句；内联广播若实现，发射的是 `autofuse/ascendc/api/brc_inline_api.h` 中已有的
`BinaryBrcInlineApiWithTwoVectorizedAxis`。不新增对外 C/C++/Python 接口，不改动 run 包布局。

## 软件功能

1. 广播缓存谓词的构造必须与算子自身的调度轴一致，不得沿用同循环内其他算子的周期。
2. 只有单 slot 缓冲上的算子才能被缓存守卫包裹，多 slot 队列必须拒收。
3. 写 GM 的算子不得被缓存守卫包裹。
4. 多维图上的 fused 缓存条件要能构造出谓词，而不是一律拒收。
5. 具备与 legacy `brc_inline` 对等的内联广播能力。

其中 1～3 是收紧（把原先会静默产生错误结果或行为不确定的路径改为拒收或改为正确），4 是放宽，5 是新增功能。

## 设计约束

- 拒收只能发生在能力判定层（`emitCapabilityShortfall`），由调用方整图回落 legacy；不得静默放过。
  这是既有纪律：verifier 失败属于硬失败，不能当作能力边界。
- 放宽任何闸门都必须有数值证明，不能只核对生成文本。
- 不得为 MLIR 的验收去修改 legacy。legacy 不支持的场景答「不支持」并回落。
- 不新增硬编码芯片型号或平台判断。

## 假设和依赖关系

- 上游对可缓存张量给出 `buf_num = 1`。该假设**不成立**：`buf_que_allocator.cpp:700` 对 VecOut 位置是
  `is_node_cached && !is_reusable` 才给 1，因此存在被缓存且多 slot 的张量。问题二正是针对这一条。
- 切分轴恒落在合轴后的第 0 段（`getTileInnerAxis` 取 `front()`）。本文方案不放宽该假设。
- 问题四涉及一处与 legacy 行为不同的取舍，已由用户确认采取拒收策略。

# 需求分析与设计

## 整体介绍

前三项修改集中在 `dialect/AFIR/Transforms/ExecConditionPass.cpp`，都是在插入守卫前增加一条判据或修正
谓词复用条件，数据流与控制流不变。第四项修改 `dialect/AFIR/Analysis/TransferPlan.cpp` 的轴解析与谓词
构造，是行为放宽。第五项是跨 `target/Ascend/Capability` 与 `conversion/AFIRToAscendC` 的新增能力，需要
单独的设计文档。

## 功能需求

### 功能需求 1：守卫谓词不得跨算子调度轴复用（P3，第一批）

1. **介绍**

   `GuardCache` 的缓存键是 `(loop, condition)`，谓词值却取自该循环中第一个被处理算子的 `TransferPlan`
   （`ExecConditionPass.cpp:59`、`:61`）。fused 谓词依赖 `blockOuterAxis`、`loopAxis`、`tileOuterAxis`
   三根轴，其中 `tileOuterAxis` 来自算子自身 tile 的最外向量化轴，同一循环内不同算子可以不同。后果是
   同一张图、同一个算子，MLIR 给出的答案取决于遍历顺序：

   - 用例 `brc_cache_3a`：`load` 先带条件、`broadcast` 的轴解析不出 → 生成两个 `scf.if`，无诊断，
     `broadcast` 用的是 `load` 的周期；
   - 用例 `brc_cache_3b`：同一张图只有 `broadcast` 带条件 → 编译期拒收。

2. **输入**

   一个函数内多个带非 `NoCache` 条件的 AFIR 算子，落在同一根 `scf.for` 上。

3. **处理**

   `GuardCache::get` 在命中缓存前，先用当前算子自己的 plan 解析三根轴：解析失败则返回 failure（由
   `guardOperation` 翻成能力不足）；解析成功但与缓存项记录的轴不同，则另建一份谓词。等价实现是把缓存
   键扩展为 `(loop, condition, blockOuterAxis, loopAxis, tileOuterAxis)`，并在插入缓存项时记录这三根
   轴。`origin` 条件不依赖这三根轴，键中它们恒为 -1，复用行为与今天一致。

4. **输出**

   同周期算子仍共用一份谓词（既有用例的 `CNT-COUNT-2` 计数断言不变）；不同周期各自一份；轴解析不出的
   算子一律拒收，与它在循环中的次序无关。

### 功能需求 2：多 slot 缓冲上的算子拒收缓存守卫（P2，第一批）

1. **介绍**

   守卫跳过的只是算子本身，`AllocTensor` / `EnQue` / `DeQue` / `FreeTensor` 每趟照发。只有单 slot 时
   地址每趟相同，跳过一趟写入之后读到的才是上一趟的结果；多 slot 队列每趟轮转，跳过后读到的是两趟之
   前的数据。`ExecConditionPass` 全程不检查 slot 数（用例 `brc_cache_2`：仅把 `slot_count` 由 1 改为 2，
   守卫照常生成，`init_queue` 按 2 个 slot 初始化，无任何诊断）。

2. **输入**

   被守卫算子的每个目的地操作数所对应的 `afir.buffer_desc`。该属性在资源物化之后仍挂在
   `reinterpret_cast` 上，用 `getBufferDescAttr(Value)`（`dialect/AFIR/IR/AFIR.h:250`）取得。

3. **处理**

   在 `guardOperation` 插入 `scf.if` 之前遍历目的地：取不到 buffer_desc 时报缺信息；`slot_count > 1`
   时 `emitCapabilityShortfall`，诊断说明「缓存守卫要求单 slot 缓冲」。`slot_count == -1`（TBuf 等未设
   该字段的缓冲）按 1 处理——它不轮转。

4. **输出**

   单 slot 图行为不变；多 slot 图整份 plan 被拒、回落 legacy，并给出可定位的诊断。

### 功能需求 3：写 GM 的算子拒收缓存守卫（P4，第一批）

1. **介绍**

   `runOnFunction` 对所有 AFIR 算子一视同仁地收集带条件的算子（`ExecConditionPass.cpp:140`），因此
   `afir.store` 带缓存条件时会被包进守卫，输出只在周期起点写出（用例 `brc_cache_4`：三个 `scf.if`，
   store 在其中）。legacy 行为相同（`enable_cache = IsValidCacheCondition(exec_condition)`，不区分算子
   类别），上游也不会这样标注，所以这不是对等缺口，而是一条未校验的输入约束。本次采取拒收策略。

2. **输入**

   带非 `NoCache` 条件、且目的地位于 GM 空间的算子。

3. **处理**

   收集阶段判断算子目的地的 `buffer_desc` 空间：GM 则 `emitCapabilityShortfall`，诊断说明缓存条件只适
   用于片上结果。判据写在空间上而不是算子类型上，避免逐个列举 `afir.store` 之外的写 GM 算子。

4. **输出**

   带缓存条件的 store 整份 plan 被拒并给出诊断，不再静默只写首趟。

### 功能需求 4：多维图上构造 fused 缓存谓词（P1，第二批）

1. **介绍**

   `collectTransferOuterAxes` 只在 `BlockOuter.split_pair_other_id == plan.loopAxis` 时解析
   `blockOuterAxis`（`TransferPlan.cpp:464`），即要求 `loop_axis` 必须是配对的 BlockInner。一维图恰好
   成立；多维图里 `loop_axis` 通常是最内维的 TileOuter，于是 `blockIndex` 与 `tileLoopSize` 都取不到，
   fused 条件一律拒收（用例 `brc_cache_1`：诊断 `cannot build the cache guard`）。同一张图改成 `origin`
   条件可正常生成，说明缺口精确落在 fused 这一支。这正是遗留清单 BR-1 的诊断来源，也是
   `LegacyGraphGuardRegSt.ContinuesBrc` 中 T4 的阻塞点。

2. **输入**

   多维调度：分块切分在一根原始轴上，被缓存算子的 tile 最外向量化轴可能与 `loop_axis` 不同源。

3. **处理**

   把谓词的轴解析与 `loop_axis` 解耦：

   - `tileOuterAxis` 仍取被缓存算子 tile 最外向量化轴的配对 TileOuter，它决定复用周期长度；
   - 块轴改取与 `tileOuterAxis` **同源**（`from` 链追溯到同一根 Original 轴）的 BlockOuter / BlockInner，
     不再要求与 `loop_axis` 配对；
   - `loop_axis` 与 `tileOuterAxis` 同源时，谓词仍为
     `iv == 0 || (blockIndex * blockInnerSize + iv) % tileOuterLoopSize == 0`；
   - 不同源时，该维的 tile 序号在本循环内不变，缓存值整个循环内不变，谓词收敛为 `iv == 0`。这比
     legacy 的跨维取模更紧：legacy 的公式在这种形态下只会多算、不会算错，所以收紧方向安全。
   - 仍然解析不出同源块轴、或轴值未物化时，保持拒收。

4. **输出**

   多维图上的广播缓存不再一律回落；谓词形态由轴的同源关系决定，不由芯片决定。

### 功能需求 5：内联广播（P5，第三批，需独立设计文档）

1. **介绍**

   MLIR 侧没有内联广播：全仓库没有任何生产代码发起 `CapabilityKind::BroadcastInline` 查询，九处
   `queryBroadcastInline` 一律返回 unsupported，`BroadcastInlineQuery` 的 `axis` / `sourceStrides` /
   `destinationStrides` / `hasDynamicShape` 从未被填充；lowering 侧只有
   `isSingletonTensorInput`（`ElementwiseTransferPlan.cpp:25`）的「repeats 全为 1 → 当标量读」一档。
   MLIR 也不读上游的模板语义角色（`template_semantic`、`kInline` 在 `autofuse/mlir` 内零命中），所以
   legacy 选中 `brc_inline` 模板的 plan 在 MLIR 看来就是一张形状不兼容的 elementwise 图
   （用例 `brc_inline_1`：`'afir.add' op has no supported Ascend implementation`）。

2. **输入**

   收的形态以 legacy `IsGeneralizeBrcInlineScene`（`common/common_utils.cpp:856`）为准：两个张量输入、
   仅一侧有广播轴、合并连续轴后两侧各为 2 组、广播侧外层组 repeat 为 1，即 `(1, A) op (B, A)`。

3. **处理**

   分两步，均在独立设计文档中细化：

   - 能力判定：把「一侧为行广播」做成 binary elementwise 的一档形态（与现有 `ScalarOperand` 同构），
     由它发起 `BroadcastInline` 查询并填充 query 字段，使该接口有真实调用方；
   - lowering：发射 `src_repeat_stride = 0` 的形态，复用 `brc_inline_api.h` 的
     `BinaryBrcInlineApiWithTwoVectorizedAxis`，需要在 Asc 方言与扩展头注册表中补对应 op。

4. **输出**

   `(1, A) op (B, A)` 形态不再回落；legacy ST 用例 `brc_inline_test` 可进入接管表。

## 已被推翻的两条假设

列在此处避免重复排查：

- **`vector_func` region 内算子的条件被跳过**不是缺陷。legacy 的 `VectorFuncPartitioner` 让 vf 节点继承
  首个输出生产者的 `sched`，各输出条件不一致时整体降为 `kNoCache`
  （`vector_func_partitioner.cpp:1404`、`:1419`），与 MLIR「region 内不插守卫、由 vf 自身条件决定」等价。
- **块轴序号取 `AxisIndex` 而非 `GetBlockIdx()`** 不是缺陷。单块轴时两者相等；多块轴时按轴取坐标比
  legacy 的裸 `GetBlockIdx()` 更准确。

## 非功能需求

### 可维护性

前三项修改都落在 `ExecConditionPass.cpp` 既有的两个函数里（`GuardCache::get`、`guardOperation` /
`runOnFunction`），不新增类型，不改变 pass 时序。判据一律通过 `emitCapabilityShortfall` 表达，与该文件
已有的三条诊断同构。

### 可测试性

三项修改各有一个已复现的 lit 用例，修复后由拒收诊断或谓词计数转为通过断言。问题四另需两例（同源 /
不同源各一）与数值验证。

### 可移植性

不涉及芯片能力查询、不新增平台分支。lit 用例沿用既有的 `--target=Ascend910B1`。

### 可靠性

三项修改都只增加拒收路径，不新增资源申请、不改变资源生命周期；谓词复用修正只影响新建 `arith` 常量与
比较的条数。

### 特性交叉影响

按 [cross_feature_check.md](../../../../docs/guidelines/cross_feature_check.md) 逐项：

| 场景 | 适用性 | 分析说明 |
|------|--------|----------|
| SuperKernel Python 接口 | 不适用 | 修改范围在 `autofuse/mlir`，不触及 `super_kernel/` |
| SuperKernel C++/AOT 接口 | 不适用 | 不导出符号，不改 `libascendsk.so` |
| Autofuse 图优化 | 不适用 | 不修改 ASCIR 图结构与 legacy pass；`exec_condition` 只被读取 |
| Autofuse Codegen/Backend | 适用 | `ExecConditionPass` 与 `TransferPlan` 属于 MLIR codegen；前三项只增加拒收，受影响的图整图回落 legacy，生成文本对单 slot 同周期图逐字不变 |
| AscendC API / Runtime 交互 | 不适用 | 不新增 `rt*` / `aclrt*` / AscendC 调用；守卫内外的同步与生命周期次序不变 |
| Python/C++ 混合绑定 | 不适用 | 不涉及 `py_module` |
| 构建与打包 | 不适用 | 不改 CMake 与安装路径；新增 lit 用例随既有目录自动纳入 |
| 测试与覆盖率 | 适用 | 三项修改各配一个 lit 用例；问题四另加两例与数值验证 |
| 性能与日志 | 适用 | 见「性能」一节；不新增默认开启日志 |
| 兼容性 | 适用 | 前三项会把今天能生成 kernel 的两类图（多 slot 缓存、store 带缓存条件）改为回落 legacy，行为变化见「兼容性检查」 |

## 性能

### 编译时长

第一批新增的判据是常数次属性读取（每个被守卫算子一次目的地遍历），相对该 pass 已有的 plan 构造可忽略。
谓词复用修正会在同一循环内不同周期的算子上多建一份谓词（三条 `arith` 运算加一次比较），只在今天本应
拒收的图上出现。

### 执行性能

单 slot 同周期图的生成文本逐字不变，kernel 执行性能不变。问题四放宽后，多维图上的广播由每趟重算变为
周期起点重算，是收益方向；其中「不同源」形态的谓词比 legacy 更紧，重算次数更少。

### 内存和产物大小

不新增数据结构与常驻内存，不影响 run 包与动态库大小。

## 接口设计

### 新增/修改接口描述

无对外接口变化。内部函数签名变化仅一处（第二批）：`materializeExecConditionGuard` 的轴解析改由
`TransferPlan` 新增字段承载（候选方案是新增 `blockInnerAxis` 字段，或在该函数内按 `from` 链现场解析）；
具体取舍在第二批开工时定，并同步修改 `TransferPlan.h` 中的字段注释。

### 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|--------|----------|----------|------|
| 接口说明 | 是否需要接口评审 | 否 | 无对外接口 |
| 接口说明 | 是否需要补充文档 | 是 | 本文；第二批另需更新 `TransferPlan.h` 字段注释 |
| 接口兼容 | 行为是否兼容 | 是 | 前三项把两类图由生成改为回落，见「兼容性检查」 |
| 接口兼容 | ABI/API 是否兼容 | 否 | 不导出符号 |
| 接口约束 | 约束不满足时是否清晰报错 | 是 | 一律走 `emitCapabilityShortfall`，诊断指向具体算子 |
| 接口测试 | 是否需要独立接口用例 | 是 | 每项一个 lit 用例 |

## 软件设计

### 关键数据结构

- `GuardCache`（`ExecConditionPass.cpp:59`）：缓存项值从 `Value` 扩展为「`Value` + 三根轴」，或把三根轴
  并入键。两种写法等价，取决于是否需要在不一致时给出诊断——给诊断则值里要留轴。
- `TransferPlan`（`TransferPlan.h`）：第二批可能新增 `blockInnerAxis`。字段注释已声明「无法解析时保留
  -1，消费者须按无对应轴处理」，新增字段沿用同一约定。

### 关键技术/算法

问题四的同源判断：沿 `AxisAttr::from` 链上溯到 Original 轴，比较两根轴的 Original 根是否相同。轴表来自
`collectGraphAxes`，`from` 链有限且无环（图属性由 bridge 构造时保证），上溯深度以轴表大小为界，避免
构造出的非法轴表导致死循环。

### 流程设计

主流程不变：`runOnFunction` 收集 → 逐个 `guardOperation` → 构造谓词 → `detachResults` → 建 `scf.if`
→ 移入。新增判据全部插在「构造谓词」之前，任一条不满足即整函数返回 failure，由 pass 的
`signalPassFailure` 终止，不留半成品 IR。

### 对子模块的修改

| 目录/文件 | 修改点 | 批次 |
|---|---|---|
| `dialect/AFIR/Transforms/ExecConditionPass.cpp` | 谓词复用判据、多 slot 拒收、GM 目的地拒收 | 第一批 |
| `dialect/AFIR/Analysis/TransferPlan.cpp` | 块轴解析改按同源、fused 谓词分两形态 | 第二批 |
| `dialect/AFIR/Analysis/TransferPlan.h` | 字段与注释 | 第二批 |
| `test/lit/conversion/` | 新增三例（第一批）、两例（第二批）、一例（第三批） | 各批 |
| `target/Ascend/Capability/`、`conversion/AFIRToAscendC/Implementations/BinaryElementwise.cpp` | 内联广播能力与 lowering | 第三批 |

### 错误处理

#### 系统错误

本次修改不申请内存、不访问文件、不调用 runtime。属性读取失败（取不到 buffer_desc、取不到轴表）按
缺信息拒收。

#### 接口错误

全部错误通过 `emitCapabilityShortfall` 报出，调用方据此整图回落 legacy。不新增错误码，不改变对外提示。

## 安全检查

按 [编码红线.md](../../../../docs/guidelines/编码红线.md) 逐项：

- 敏感信息与公网地址：不涉及。
- 外部输入作索引/长度/偏移：`slot_count`、轴 id 均来自图属性，使用前判断 `-1` 与范围；`from` 链上溯以
  轴表大小为界。
- 整数溢出与除 0：谓词中的 `%` 除数是 `tileOuterLoopSize`，取自轴的 `loop_size`，与今天同源，本次不改变
  其取法；不新增乘法。
- 资源释放与异常分支：不新增资源。
- ABI/API 兼容：不导出符号。
- 图改写等价性与确定性：新增判据不改写图；谓词条数只依赖轴 id 的比较，不依赖哈希容器遍历顺序或指针
  地址（缓存键若扩展为含轴的元组，仍按轴 id 比较）。
- runtime/AscendC 生命周期：守卫内外的 `AllocTensor` / `EnQue` / `DeQue` / 屏障次序不变。

## 兼容性检查

前三项修改把两类今天能生成 kernel 的图改为编译期拒收、整图回落 legacy：

1. 被缓存张量位于多 slot 队列的图——今天生成的 kernel 会读到两趟之前的数据，会静默产生错误结果，拒收是修正；
2. `afir.store` 带缓存条件的图——今天生成的 kernel 只在周期起点写出，同样会静默产生错误结果。

两类都不改变 legacy 路线的行为，也不改变单 slot 同周期图的生成文本。需要确认接管表中是否有条目因此由
「MLIR 通过」变为「回落」。若有，这些条目原本就是错误的结果，应在接管表记录中说明原因，不得为了维持
通过数量而放宽判据。

## 测试设计

### 测试边界

- 测试入口：`af-opt` 运行 pass 组合（lit），以及 `--afir-plan-candidate-lowering-pipeline` 全链路。
- 测试出口：对拒收项检查诊断文本；对谓词项检查 `scf.if` 与 `arith.cmpi` 的计数与结构。
- 不打桩：全部用现成 pass 与手写 AFIR 输入。

### 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|----------|------------|----------|------|
| 功能 | 同周期算子共用一份谓词（既有计数断言不回退） | `afir-exec-condition-cache-guard.mlir` 保持通过 | lit |
| 异常 | 不同周期/轴解析不出的算子一律拒收，与遍历次序无关 | 新增一对用例（对应 `brc_cache_3a/3b`） | lit |
| 异常 | 多 slot 队列上的缓存守卫被拒收 | 新增用例（对应 `brc_cache_2`） | lit |
| 异常 | GM 目的地带缓存条件被拒收 | 新增用例（对应 `brc_cache_4`） | lit |
| 功能 | 多维图 fused 谓词两种形态 | 第二批新增两例（对应 `brc_cache_1` 及其同源变体） | lit |
| 特性交叉 | 多维图广播缓存数值正确 | 第二批：CPU e2e 与 NPU 对数 | E2E |
| 功能 | `(1,A) op (B,A)` 内联广播 | 第三批 | lit + E2E |
| 兼容性 | 接管表条目变化 | 第一批合入后按既有测量方法重新测量受影响的行 | 探针 |

### 测试命令

```bash
# 快速回归（本地，已编译的 af-opt）
/opt/llvm/bin/llvm-lit -q build/autofuse/mlir/test/lit

# 全量档
bash ./mlir-test.sh
```

预期结果：lit 全部通过；第一批合入后既有 `afir-exec-condition-*.mlir` 三例不变，新增三例通过。

## 验收标准

- 第一批：三个新增 lit 用例通过；`afir-exec-condition-cache-guard.mlir` 的谓词计数断言不回退；lit 全量
  通过；接管表中受影响条目的变化有记录。
- 第二批：多维图 fused 条件能生成 kernel，两种谓词形态各有用例；数值在 CPU e2e 与 NPU 上与 legacy 对齐；
  `LegacyGraphGuardRegSt.ContinuesBrc` 的 T4 阻塞点消除。
- 第三批：按独立设计文档验收。

# 修复计划

| 批次 | 内容 | 方向 | 验收依赖 |
|---|---|---|---|
| 第一批 | 功能需求 1、2、3（P3 → P2 → P4） | 收紧 | 本地 lit |
| 第二批 | 功能需求 4（P1） | 放宽 | 远端 e2e / NPU 数值 |
| 第三批 | 功能需求 5（P5） | 新增功能 | 独立设计文档 |

三者都落在 `ExecConditionPass.cpp`，改动互不冲突，第一批一次提交。第二批依赖第一批先把「建不出就拒收」
的路径收干净，否则放宽后无法区分是周期算错还是谓词被复用。第三批与前两批不争同一批文件，可并行推进，
但需先出设计文档。

# 证据附录

复现用例位于本次评审的工作目录（未纳入版本库）：
`/tmp/claude-1000/-home-zhe-code-autofuse/c0aed98b-c37a-484a-b928-c7bff9d59139/scratchpad/brc/cases/`，
其中 `README.md` 给出每例的当前行为与判据位置，各 `.mlir` 文件头部记录复现命令与期望行为。

| 用例 | 对应功能需求 | 当前行为 |
|---|---|---|
| `brc_cache_1_two_dim_loop_axis.mlir` | 4 | 拒收：`cannot build the cache guard` |
| `brc_cache_2_multi_slot_queue.mlir` | 2 | 两个 `scf.if`，队列 2 个 slot，无诊断 |
| `brc_cache_3a_guard_reused_across_plans.mlir` | 1 | 两个 `scf.if`，无诊断 |
| `brc_cache_3b_guard_rejected_alone.mlir` | 1 | 拒收（与 3a 同图同算子） |
| `brc_cache_4_store_guarded.mlir` | 3 | 三个 `scf.if`，store 被包进条件分支 |
| `brc_inline_1_row_broadcast.mlir` | 5 | 拒收：`'afir.add' op has no supported Ascend implementation` |

验证工具与基线：`build/autofuse/mlir/tools/af-opt/af-opt`（基线 `d28990bd6`）；
`/opt/llvm/bin/llvm-lit -q build/autofuse/mlir/test/lit` 在评审开始时 290 项全部通过。

# 第一批验证记录（2026-11-09）

改动：`dialect/AFIR/Transforms/ExecConditionPass.cpp`（+70 −3），新增三个 lit 用例
`afir-exec-condition-guard-reuse-unsupported.mlir`、`afir-exec-condition-multi-slot-unsupported.mlir`、
`afir-exec-condition-store-unsupported.mlir`。

- lit：293 项全部通过（原 290 项 + 新增 3 项），既有三个 `afir-exec-condition-*.mlir` 的断言未改。
- 变异验证：将生产代码的改动退回并重新编译后，恰好这三个新增用例失败，既有三个用例仍然通过。这
  说明新增用例锁定的正是本次改动，而非其他因素。
- 全量档：`bash ./mlir-test.sh` 结论行
  `Autofuse MLIR targets succeeded: check-ut check-lit check-st check-e2e`，耗时 3m31s。
  即现有 UT/ST/E2E CPU 语料里没有「被缓存张量落在多 slot 队列」或「GM 目的地带缓存条件」的图，
  本次收紧没有让任何既有用例由生成转为回落。
- **未覆盖**：接管表条目的变化需要远端探针重新测量，本轮未执行。若后续测量发现有条目由「MLIR 通
  过」变为「回落」，按本文「兼容性检查」一节处理，不得为了维持通过数量而放宽判据。

# 第二批验证记录（2026-11-10）

改动：`dialect/AFIR/Analysis/TransferPlan.{h,cpp}`，新增 lit 用例
`afir-exec-condition-two-dim-cross-origin.mlir`、`afir-exec-condition-two-dim-same-origin.mlir`。

方案与原设计有一处订正：原方案写的是「块轴取与切分轴**同源**（`from` 链追到同一根 Original
轴）的那一对」，按单根原始轴实现后，`load_broadcast_segmented_store_8x1` 由可生成变成拒收。
那张图先合轴再分块（`z0z1T` 同时源自 z0 与 z1），单根原始轴的判断对合轴给不出答案。实现改为
比较**原始轴集合**：循环轴的集合覆盖切分轴的集合即算同维（合轴形态落在这一支），两集合不相交
算另一维，部分重叠按解析不出处理。

- lit：295 项全部通过（第一批之后的 293 项 + 新增 2 项）。
- 变异验证：退回本批改动后只有 `afir-exec-condition-two-dim-cross-origin.mlir` 变红。
  同源那一例在改动前后都通过，它是回归守卫而不是变异见证，用例头部已写明这一点。
- 本地全量档：`Autofuse MLIR targets succeeded: check-ut check-lit check-st check-e2e`。
- 远端真机（`npussh mlir`，`~/niu/brc-verify`，从 `cann/graph-autofusion` 的 `develop` 克隆，
  HEAD 即基线 `d28990bd6`，`-j 40`）：`check-ut`、`check-lit`（295/295）、`check-st` 通过；
  `check-e2e` 在三个 transpose 用例的链接阶段失败，原因是 CANN 的 `libtiling_api.a` 需要
  `dlopen` 而链接命令缺 `-ldl`。把本次改动 stash 掉后在基线上重编同一目标，报同样的错，
  因此是该机器的环境问题，与本次改动无关。
- 远端 NPU 实测（`run_npu_e2e.sh --mode mlir`，7 个广播用例）：`All NPU E2E cases passed.`，
  数值全部正确，`backend_route=mlir`。`load_broadcast_to_abs` 与
  `load_broadcast_inner_brc_cascade` 为 `PASS`；原先记为 `PASS_FALLBACK` 的五个仍是数值通过
  带已知拒收诊断，而它们剩下的诊断全部是
  `'afir.add' / 'afir.mul' op has no supported Ascend implementation`，属于功能需求 5
  （brc_inline）的缺口，不是缓存守卫。

## 本批次尚未取得的证据

**「另一维」那一支没有语料见证。** `cache guard` 这条诊断在改动前后的本地全量档日志、远端
全量档日志以及 NPU 日志里都是 0 次；单独跑 `load_broadcast_to_cascade` 的改动前基线也是 0 次
（它的阻塞点一直是 elementwise 那一条）。也就是说现有语料里没有任何图被这道闸门拦住，本批
改动在语料上**不多编出任何一张图**，新分支目前只由手写 lit 用例钉住。

因此本批次的结论只能写成两条：

1. 不回退——两档全量档与 NPU 数值均通过；重写后的轴解析确实被真实图走到（合轴形态的
   `load_broadcast_segmented_store_8x1`、`load_broadcast_duplicate_store` 等），第一版实现的
   错误正是被它们挡下来的；
2. 「另一维」分支的数值验证仍然欠缺。按 2026-11-08 测试调整记录，被移除的 ST
   `LegacyGraphGuardRegSt.ContinuesBrc` 的 T4 阻塞点正是「缺少 cache guard」，恢复该 ST（或
   按它的形状补一个 e2e 用例）是取得这个见证的下一步，不能用本批次的全绿代替。

# 「另一维」见证的补充验证（2026-11-10）

目标是给功能需求 4 的「另一维」分支找一张真实 optimizer 产出的图。三条路都没有拿到见证，
过程中发现一个新的静默数值错误，记在下一节。

**一、恢复 `LegacyGraphGuardRegSt.ContinuesBrc` 当探针。** 按 2026-11-08 记录里的源码原样恢复
并单独运行，reg 侧的首个阻塞点不是缓存谓词，而是
`UnsupportedOperation: VectorFunc subgraph node 'brc2' carries a cache exec_condition, which the
register-form lowering does not model yet`（`bridge/ScheduleToAfir/ScheduleToAfir.cpp:1454`）——
VectorFunc 子图内带缓存条件的节点在 bridge 这一层就被按「尚未建模」拒收，根本到不了谓词构造。
把同一张图挂到 mem 侧跑，阻塞点是 `'afir.broadcast' op has no supported Ascend implementation`
（五维链式广播、多消费者）。两侧都不是见证。探针已从源码中移除。

**二、扫既有 CPU E2E 的生成 kernel。** 按谓词形态分类 70 份带 `if` 的生成 kernel：30 份广播
相关用例**全部是相位形态**（`bool g = iv == 0 | phase == 0`，即同维那一支），没有一份是「只算
首趟」的形态。也就是说现有语料确实不覆盖「另一维」。

**三、构造一个必须切分内轴的形状。** 用 `load_broadcast_abs_store` 注册
`[1, 65536] → [33, 65536]`：内轴 65536 个 f32 超过 UB，调度必须切分内轴并在它上面发循环，而
广播轴在外、落在 TILE_IN 上——这正是「另一维」的几何。实测该形状下调度根本没有给出缓存条件，
生成的 kernel 里一个 `Broadcast` / `Duplicate` 都没有，只有一次 `DataCopy`，且数值错（见下节）。
该注册与为取得 kernel 文本临时放开的形态断言均已还原。

**结论**：「另一维」分支目前仍只由 `afir-exec-condition-two-dim-cross-origin.mlir` 钉住。要拿到
语料见证，需要一张同时满足两条的图：TILE_IN 是广播轴（否则 `NodeCacheMarker` 不给 fused 条件），
且另一维也被切分并在它上面发循环。第三条路的形状满足几何条件，但撞上了下一节那个缺陷。

# 新增问题 6：大内轴二维广播生成不含广播调用的 kernel，输出大面积为 0

**现象**：`load_broadcast_abs_store` 注册 `OUTPUT_SHAPE "[33, 65536]"`、`INPUT_SHAPES
"[1, 65536]"`（f32）后，`AF_MLIR_STRICT gate=none shortfall=no`，即 MLIR 声称接管这张图；但生成
的 kernel 里没有任何 `Broadcast` / `Duplicate` 调用，只有一次 `AscendC::DataCopy`，CPU 仿真数值
对照大面积失败（期望 3 实得 0、期望 2.75 实得 0）。

**性质**：声称接管却静默产生错误结果，比回落严重。**不是本轮两批改动引入的**：把
`TransferPlan.{h,cpp}` 与 `ExecConditionPass.cpp` 三个文件回退到基线 `d28990bd6` 后重编，同一
形状复现同样的错误输出。

**复现**：在 `test/e2e/cpu/cases/load_broadcast_abs_store/CMakeLists.txt` 里加上该形状注册，
重配 cmake 后 `cmake --build build --target check-e2e-cpu-load_broadcast_abs_store_1x65536`。
该用例自身的 `CheckGeneratedKernel` 会先因为「找不到 `Broadcast(`」失败而不写出 kernel 文本，
定位时需要临时放开那一条断言。

**下一步**：先确认广播是在哪一层消失的（调度是否把 Broadcast 节点删了，还是 lowering 把它折成了
搬运），再决定是补能力判据拒收，还是修正那一层。这一条独立于本文的五项，优先级应高于「另一维」
见证——它会静默产生错误结果，而缺见证只是覆盖不足。

**已修复（2026-11-10，`01c798c`）**：根因不是广播调用消失，是 `TransferPlan.cpp` 的
`getGlobalDescriptor` 把 Store 的 GM 偏移错取成生产者 UB 侧的步长，广播轴那一项被「步长为 0
就整项跳过」这条优化当成 0 剔掉（与 `AGENTS.md` 3.4 条「跨核归约全核写同一个地址」同一族）。
修正为 Store 的 GM 起点取目的调度视图。复核：`check-e2e-cpu-load_broadcast_abs_store_1x65536`
现在 `PASSED`，数值对照正确；生成的 kernel 里仍然没有 `Broadcast`/`Duplicate` 调用——这不是
缺陷，这一档选中的调度计划本来就按行发 `DataCopy` 复制，不经过专门的广播指令。另外，这个形状
修复后也没有带出「另一维」缓存条件的见证——重新生成的 kernel 里没有任何 `if`，说明这档选中的
调度计划原本就不需要缓存条件，从一开始就不是「另一维」的候选，H.3 仍然缺语料。
