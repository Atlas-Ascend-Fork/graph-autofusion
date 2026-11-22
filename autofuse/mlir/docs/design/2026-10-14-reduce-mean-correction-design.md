# 简介

> 2026-10-23 测试范围调整：本文涉及的 `test/e2e/npu/direct_ascir/` 设备专项及其 host 自测已删除。相关命令、路径和验收记录仅作历史参考，可从提交 `0706a660dee787142366e65c43c7abe409f1aa89` 恢复；当前 NPU E2E 仅保留 Torch / Inductor 路线，不代表仍覆盖直接 Mean 节点的设备验收。

## 目的

修正 MLIR Mean 的全局分母语义，支持核内跨轮累加，并与现有 RCore 两阶段调度衔接。面向 AFIR、调度桥接和下降实现维护者。**已实施；带保留轴的 f32 Mean 已完成单阶段及 RCore 两阶段真机验收，见文末实施和验收记录。f32 中间和溢出为已知限制，全轴 Mean 不在本次范围。**

**本次验收范围限定为「带保留轴的 Mean」。** 全轴归约（输出标量，A 组为空）既无真实调度语料、
也无数值证据，不在本次验收内，见文末「后续项：全轴 Mean 的最小闭环」。因此不得据本次结果
宣称「Mean 场景已完整支持」。

## 范围

仅涉及 mem-based、f32 Mean，以及必要的 AFIR 语义分析、下降和测试。保留既有 AR/RA 布局能力、scratch 约定和 Mean 第二阶段 Sum 的调度规则。不扩 dtype、布局、ArgMax、v35 regbase，不修改 ATT 选优算法或 legacy 算法。已有 dtype 未提交改动作为基线保留。

# 总体概述

## 软件概述

### 项目介绍

Mean 数学语义为 S/N，S 是完整归约域的和，N 是逻辑元素数。当前 MLIR 非累加 Mean 在 ReduceSum 后按当前 R actual extent 归一化，累加 Mean 被拒绝。当前 tile 的 R actual 在 RCore 第一阶段不一定等于全局 N。

需要纠正两条旧结论：legacy 的一部分多轮路径在 `codegen_kernel_loop.cpp:891` 归一化，门禁是 `IsReduceDoubleTile`：任一张量的 vectorized_axis 含至少两根 TileInner，并且有 reduce。它不等价于多轮，必须检查真实产物，不能只检查 ReduceApiCall 或据此宣称 legacy 多轮全部正确。本文初稿据分支条件推断「仅 R 切分而 A 未切分时漏发」，**该推断已取证撤回**（在真实调度产物里不可达，见文末「legacy 的「仅 R 切分」漏档：取证结论」）；判据不等价留下的实际口子只剩全轴归约。此外，`reduce_schedule_case_generator.cpp:165` 已将 Mean 第二阶段设置为 Sum，这仅说明 RCore 的第二阶段规则，不能推广到其他任意分期路径。相关 Reduce.cpp 注释必须同步更正。

### 产品环境介绍

沿用现有 AFIR→AscendC→设备编译流程、CANN 9.2.0 和本地 Docker 验证环境。Python 前端、运行时接口、安装包格式不变。真机验证使用匹配本次源码的构建及 Python overlay。

## 软件功能

1. 获取调度前完整逻辑归约域的元素数。
2. 每个输出 tile 的核内归约完成后恰好归一化一次。
3. RCore 第一阶段输出核内部分和除以全局 N，第二阶段 Sum 合并贡献。
4. 对无法恢复分母或无法证明完成位置的候选给出明确拒绝原因。

## 设计约束

不让 LoopGenerate 依赖完整 ReducePlan 或 AscendC 能力；不引入通用归约框架。归一化不得使用 padding 尺寸、最后一个 tile 的元素数、block 数或各核均值的无权平均。继续使用现有缓冲与执行同步机制。

## 假设和依赖关系

- 现有轴 `from` 链和 Original size 能否完整保留 Mean 归约域，尤其 RCore 第一阶段：实施第一步用真实调度产物确认，不能由手写夹具推定。
- 当前 RCore 图改写保持第一阶段 Mean、第二阶段 Sum：源码已确认，仍补实际桥接产物回归。
- 成功支持范围为现有后端可承载的非空归约。空归约不在本次扩展范围，必须沿用或补齐明确能力拒绝，不能生成除零代码。
- 核内累加轴必须映射到实际祖先 `scf.for`。BlockOuter 在 LoopGenerate 中绑定 block index、不创建 for；不能将它当作核内累计轮数。
- 完成归一化前，外部消费者不能读取最终 Mean；已有生命周期/作用域规则需要真实用例确认。

# 需求分析与设计

## 整体介绍

推荐沿用 legacy 的数值组织方式，而不复制其循环发射代码：

```text
全局归约域 → N（保留 Original 逻辑尺寸）
每个核、每个输出 tile：
  各轮 ReduceSum → 原始部分和累加
  核内最后一轮：结果 *= 1/N
RCore 第二阶段：Sum(各核贡献)
```

首选在 Reduce 下降中用“所有核内累加循环均到最后一轮”的条件包住 Muls。它在数学位置上等价于核内循环结束后处理，避免跨 region 移动 destination 或引入新的 finalize op。非累加路径直接缩放。归一化后不允许继续向同一累加器加入原始部分和。

不采用“每轮缩放部分和再 Add”为默认方案：虽然实数代数等价，但增加 Muls 次数、改变舍入及下溢行为。也不提前把图中 Mean 普遍拆成 Sum+Div：这会影响调度分组、融合及生命周期，超出当前缺口。

## 功能需求

### 功能需求 1：全局分母

1. 介绍：恢复完整逻辑归约元素数 N，所有轮次和核使用相同 N。
2. 输入：Mean 完整输入/输出描述、函数 asc_graph 的轴来源和尺寸、原始 shape 参数绑定。
3. 处理：从逻辑归约轴追溯 Original 轴，稳定去重后计算尺寸乘积。拆分 inner/outer 的共同来源只计一次。不能从 vectorized R 或物理 stride 单独推导逻辑计数；广播维可能逻辑上有多个元素但物理 stride 为零，需要依据真实 bridge 表达确认，不能直接复用“输入非零 stride”的物理归约判据。使用原始参数绑定，不做 tile actual 替换。计数保留足够整数宽度，乘法检查溢出，转浮点前不截断到 i32。
4. 输出：可分析的全局归约域及 N；来源丢失、混合来源不能区分等情况明确拒绝。不得退回 tile actual。必须区分原始逻辑输入与已有归约的 partial 输入；仅追溯到名为 Original 的轴不能证明后者是完整语义域。沿生产者及已知分期改写确认含义，合法的普通 Mean(workspace tensor) 也不能仅因 workspace 存储位置被误拒。无法证明完整域覆盖时拒绝。若真实路径在桥接前已丢失来源，启用备选：在语义仍完整的边界保存 Mean 专用归约域属性（全局 N 的符号表达及其归属），bridge 透传并验证绑定；若只在 bridge 丢失，则只修改 bridge。此备选已纳入范围，仅在来源检查失败时启用，不从 partial 的 shape 反猜 N。

### 功能需求 2：跨轮及两阶段归一化

1. 介绍：先完成核内求和，再执行一次全局分母缩放。
2. 输入：全局 N、既有 ReducePlan、核内累加循环及 destination。
3. 处理：保持首轮复制和后续 Add；所有核内累加轴均完成时 Muls。当前 LoopGenerate.cpp 的 emitLoop 创建 lower=0、step=1 的 scf.for，upperBound 在建循环前形成，可能依赖外层循环，但对本层迭代不变。复用对应 for 的 SSA upperBound/step，不重新读取或计算 loop_size；不能证明正步长及本层不变性时拒绝。以剩余距离 `upperBound - iv <= step` 判断最后迭代，避免直接计算 `iv + step` 溢出。条件只涉及实际核内循环，不能检查 block index 是否为最后一个核。单轮循环也执行一次。使用现有 resultSpan 和 dtype 对齐规则覆盖 destination 有效槽位跨度。RCore 第一阶段每核均缩放，第二阶段仅求和。
4. 输出：单核为最终 Mean；RCore 第一阶段为 `local_sum/N`，第二阶段为最终 Mean。不新增 workspace 或 count tensor。

## 非功能需求

### 可维护性

归约域分析保持纯 AFIR，不依赖 scratch、布局或后端。SSA 物化和完成条件发射属于下降。只复用含义一致的 helper，不把逻辑元素计数塞入物理形状。更正旧注释、边界说明与拒绝测试。

### 可测试性

以数值结果和实际选中调度为主要依据。源码出现某个调用、没有某个调用，均不是数值验收标准。能力分析与下降对同一语义使用同一来源，防止候选接受后才发现分母缺失。

### 可移植性

使用现有 arith/scf/AscendC 接口，不增加 SoC 硬编码。维持 f32 支持范围和既有环境依赖。

### 可靠性

全局尺寸不可恢复时拒绝候选，不猜分母。静态零/非法尺寸拒绝；动态输入遵循已有 shape 合法性约束，无约束时必须补可下降的检查。空工作核不得读取未初始化累加器。零轮数的调度处理列为 RCore 真实产物检查项。

### 特性交叉影响

| 场景 | 适用性 | 结论 |
|---|---|---|
| SuperKernel Python 接口 | 不适用 | 不改接口、wheel 或选项 |
| SuperKernel C++/AOT 接口 | 不适用 | 不改 ABI 或 AOT |
| Autofuse 图优化 | 适用 | 验证既有 RCore 分期和来源保留；默认不改调度器；来源丢失时仅在必要边界保留 Mean 语义 |
| Autofuse Codegen/Backend | 适用 | 修改 Mean 分母和累加收尾，覆盖候选及生成代码 |
| AscendC API / Runtime 交互 | 适用 | 不额外新增同步机制；保留现有 Mean Muls 前的 PIPE_V，核对 Add/首轮 DataCopy→Muls 两条依赖，不增 ACL 调用 |
| Python/C++ 混合绑定 | 不适用 | 不改绑定，E2E 使用匹配源码的现有接口 |
| 构建与打包 | 不适用 | 不新增依赖、安装路径或包内容；测试按既有机制接入 |
| 测试与覆盖率 | 适用 | 新增 UT/lit/CPU/NPU Mean 覆盖 |
| 性能与日志 | 适用 | 每轮少量结束判断，每个核内输出 tile 一次 Muls，无新增默认日志 |
| 兼容性 | 适用 | 原来合法的单轮 Mean 保持结果；此前拒绝的合法候选开放，错误分母修正 |

## 性能

### 编译时长

每个 Mean 做一次来源遍历和去重，复杂度与相关轴/来源边数线性相关。不新增整图 pass。记录代表性图的编译时间，不能把未测量的开销写成零。

### 执行性能

每轮增加与累加轴数成正比的标量完成判断，Muls 仅在最后一轮执行。全局 N/倒数是循环不变量，复用现有可用的绑定及优化，检查产物是否重复计算；不为提升位置新增通用优化 pass。与单轮基线比较；legacy 仅在确认生成归一化且数值正确的同调度路径作性能对照，分离新增候选带来的变化。

### 内存和产物大小

沿用两块 scratch 和 destination，不增加跨核 count/workspace。增加少量标量表达式和条件代码。

## 接口设计

### 新增/修改接口描述

内部新增概念接口 `analyzeMeanReductionDomain(Operation*, diagnostic)`，返回 Original 归约轴 ID 及其逻辑尺寸；纯分析、不发诊断、不创建 SSA。名称及承载类型实现时匹配既有风格。

下降侧新增局部全局计数物化 helper，复用 `materializeSymbolicExpr` 与现有参数绑定入口。若必须共享内部绑定查询，仅开放最小已有能力，不复制一套符号解析。

能力查询检查归约域可恢复性，下降检查实际 SSA 支配关系及循环条件。失败原因分别说明缺少来源、尺寸不可物化、核内完成位置不可表达或已有布局/资源不足。无外部 ABI 变化。

### 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|---|---|---|---|
| 接口说明 | 是否需要接口评审 | 是 | AFIR 内部语义接口 |
| 接口说明 | 是否需要补充文档 | 是 | 分母及两阶段输出含义 |
| 接口兼容 | 行为是否兼容 | 是 | 保持合法单轮行为，修正错误分母 |
| 接口兼容 | ABI/API 是否兼容 | 否 | 无外部变更 |
| 接口约束 | 约束不满足时是否清晰报错 | 是 | 不回退到局部计数 |
| 接口测试 | 是否需要独立接口用例 | 是 | 来源去重、计数和失败原因 |

## 软件设计

### 关键数据结构

轻量 MeanReductionDomain 保存稳定顺序的 Original 轴及 ExprAttr 尺寸，仅在编译期存活。复用 ReducePlan 的布局/scratch 和 ReduceAccumulationInfo 的语义事实，不新增运行时状态。核内循环与核间分片必须明确区分；不得简单删除查不到 for 的轴来掩盖错误。

### 关键技术/算法

跨轮判据使用 accumulationAxes 表达的真实归约调度语义，并区分核内循环与核间分片；不迁移 legacy 的双 TileInner 门禁，必须覆盖仅 R 切分的漏档。

N 为逻辑归约尺寸乘积；每核 j 输出 `S_j/N`，第二阶段 `Σ_j S_j/N`。各核/各轮长度不等不影响该公式。尾块补零属于 Sum 的物理填充，不计入 N。

最终完成条件为所有实际核内累计循环的末轮条件之合取。在核内最后一轮的 Add/Copy 后缩放，与 legacy 循环外缩放保持相同的部分和组织。scf.for 的上界是循环外 SSA 操作数；允许内层上界依赖外层 iv，不要求整个循环嵌套全局不变。必须确认保留轴循环下每个 destination 的初始化/完成配对，防止上一输出的归一化结果被后续输出复用。

### 流程设计

1. 保存真实单轮、跨轮、RCore 第一/第二阶段 ASCIR/AFIR 产物，确认完整逻辑域/partial 来源、block 轴分类和消费者位置。先验 legacy 的仅 R 切分形态：以 A=8、R=131072 等形状生成候选，记录实际 tile/计划和 kernel，检查 dimr_recip 或等价缩放是否存在以及执行位置；形状不能证明门禁已触发，须确认只有一根 TileInner 且 IsNeedMultiReduce 成立。用独立参考记录数值结果。缺缩放的 legacy 计划只能作缺陷对照，不能作正确性或性能基准。
2. 补全局域分析及 UT，更新非累加 Mean 分母。首先解决第一阶段可能误用 local actual 的问题。
3. 增加末轮归一化，复用累加路径；语义和资源检查完整后开放 Mean 累加能力。
4. 跑固定调度数值用例，再跑真实前端候选选择和 RCore 真机验证，最后更新能力说明。

若步骤 1 证实来源丢失，启用上述专用语义属性备选并补往返测试；消费者在完成前读取则保持拒绝，明确需要调整的具体生命周期边界，不叠加局部猜测。默认不新增 pass、不修改 LoopGenerate 的后端依赖。

### 对子模块的修改

- `dialect/AFIR/Analysis/ReducePlan.{h,cpp}`：复用轴图事实，增加独立 Mean 逻辑域分析；不可改变其他 op 的既有能力范围。
- `conversion/AFIRToAscendC/Implementations/Reduce.cpp`：全局分母、末轮缩放、能力检查及错误注释修正。
- AFIR 符号绑定工具：仅在现有公开入口不能承载原始尺寸物化时做必要的最小复用调整。
- `test/ut/dialect/ReducePlanTest.cpp`、reduce lit、CPU/NPU E2E：增加逻辑域及数值覆盖。
- 原 Mean accumulation rejected 测试：合法场景改为成功验证；非法来源/资源条件仍分别测试明确拒绝。

### 错误处理

#### 系统错误

沿用现有编译、运行时和缓冲释放路径。不新增资源。测试失败保留日志到忽略目录，不清理既有远端验证目录。

#### 接口错误

非法描述/缺失 provenance 返回分析失败；缺资源保持能力诊断；不将能力不足归因为循环调度损坏。动态溢出/零分母检查必须验证实际 emit/设备编译支持，不能只留下不可执行 assert。

## 安全检查

已按编码红线核对：无密钥、账号、平台硬编码；计数乘法防溢出且不截断 i32；分母非零；不增加分配/释放；归一化写入限于现有 resultSpan；保留同步与消费者执行顺序；来源遍历和去重结果确定；不改外部 ABI。

## 兼容性检查

Python/C++ 接口、脚本参数、安装目录、运行包不变。不修改 legacy 算法。空归约不声称新增支持。int32 Mean 是既有能力差：MLIR 能力侧和下降侧均只支持 f32 Mean，legacy 的 Int32 Mean SUCCESS UT 不构成数值支持证明，不把这一差异记为本次退化。Sum i32、Max/Min f16 等当前 dtype 改动需回归，不能与本次 Mean 功能混淆。

## 测试设计

### 测试边界

UT 验证纯分析；lit 验证实际下降及拒绝原因；CPU 验证本次生成代码的核内计算；NPU 验证同版本完整前端/调度/两阶段/设备结果。CPU 的 blockDim=1 不能替代多核验收。legacy 是对照组，不是唯一 oracle。

### 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|---|---|---|---|
| 功能 | Original、split、merged、多根 R 的计数去重 | 纯分析验证期望域和 N | gtest UT |
| 功能 | 广播逻辑维和 extent=1；实际支持形态 | 真实 descriptor 夹具，不能以非零 stride 代替逻辑计数 | UT/ST |
| 功能 | AR/RA，单轮/多轮，R 尾块不等长，A 尾块及嵌套循环 | 固定调度生成代码，独立高精度 sum/N 参考 | lit/CPU ST |
| 功能 | N=10、tile=8 等不等长分块，非零均值、分块不同均值、全 1 | 区分漏除、重复除、局部均值相加和错误分母 | CPU/NPU ST |
| 功能 | RCore 各核不等长，每核一轮/多轮 | 保存两阶段 IR、执行所选计划，核对全局 N | NPU E2E/RDV |
| 功能 | 分母超过 i32 的表达式 | 标量表达式测试，无需分配超大 tensor | UT/lit |
| 功能 | 接近 f32 上限、大 R（如 262144）、有限期望均值但部分和溢出 | 高精度参考；分别检查 tile 内求和与跨轮累加溢出 | CPU/NPU ST |
| 异常 | 来源缺失、计数溢出/零、缺 scratch、不能识别完成位置 | 验证正确拒绝层和原因 | UT/lit |
| 性能 | 小 R/大 R、不同轮数的耗时 | 同输入同调度重复测量；记录判断/Muls 次数 | Benchmark |
| 兼容性 | 七类 reduce、现有 dtype 扩展、后接 Store/Add/Mul | 现有回归及融合 E2E | UT/ST |

数值参考使用 FP64 求和再除完整 N，并按 f32 输出评估误差。普通有限值初始采用 rtol=1e-4、atol=1e-5；强抵消输入另外报告绝对误差，不通过扩大统一容差掩盖错分母。全 1 等诊断输入独立检查明显倍数错误。若有限输入、有限参考均值出现中间和溢出，判定为本算法数值缺陷，不放宽容差。此时必须重新选择数值策略：每轮部分和先除全局 N 可缓解跨轮累加溢出，却不能修复该轮 ReduceSum 已溢出；后者需在求和前缩放输入或使用可支持的更宽累加，并评估 scratch、舍入及性能。该结果是实现开放能力前的决策门禁，不把每轮缩放写成保证有效的通用退路。特殊值按已有支持契约单独验证，不用 CPU 特殊值模拟独自裁决设备结果。

### 测试命令

当前已有构建树的本地增量入口：

```bash
AF_MLIR_DEV_IMAGE=autofuse-mlir-dev:arm64 \
AF_CANN_VOLUME=af-cann-910 \
AF_CANN_INSTALL_PREFIX=/usr/local/Ascend/cann-9.2.0 \
bash autofuse/mlir/scripts/enter_dev_container.sh -- bash -lc '
  cmake --build build --target af-opt ascir-translate AFIRDialectTest -j 8 &&
  build/autofuse/mlir/test/ut/dialect/AFIRDialectTest &&
  /opt/llvm/bin/llvm-lit -j 8 -v --filter=reduce build/autofuse/mlir/test/lit &&
  cmake --build build --target check-e2e-cpu -j 8
'
```

不得在该 build/ 中以不完整参数重跑 cmake 配置；历史上会移除测试构建图而保留误导性的 help target。新增 CPU 测试如需更新配置，先核对原缓存和完整配置入口，使用保全 MLIR 测试选项的配置，必要时另建测试树。验收记录实际发现/执行数量以及新增用例名称，空执行不算成功。

NPU 在本次匹配源码构建及 overlay 已就绪的隔离目录中，使用其一致 Python 运行：

```bash
python3 autofuse/mlir/test/e2e/npu/reduce_op_fuzz.py --fuzz-ops mean --fuzz-per-op 12 --fuzz-seed 20261014
```

该命令仅为补充随机覆盖，不能替代固定 RCore/多轮数值用例；而且它与整条 torch 路径一样**到不了
`afir.mean`**（见「真机验收：前端不产出 Mean 节点」）。

直达 `afir.mean` 的真机入口（在同一隔离目录、`devenv.sh` 已就绪的前提下，逐 shape 跑）：

```bash
AF_MEAN_INPUT_PATTERN=shard \
python3 autofuse/mlir/test/e2e/npu/direct_ascir/tools/run_reduce_validation.py \
  --case autofuse/mlir/test/e2e/npu/direct_ascir/cases/reduce_mean_retained_axis \
  --soc-profile ascend910_9362 \
  --profile autofuse/mlir/test/e2e/npu/direct_ascir/profiles/ascend910_9362.json \
  --shape 8 65537 --warmup 0 --repeat 3 --output-dir <out>
# 实际执行的是哪套 plan，从产物读，不按 shape 推：
python3 autofuse/mlir/test/e2e/npu/direct_ascir/tools/read_executed_tiling.py <out>/<case>-*/
```

`--warmup 0 --repeat 3` 不是口味问题：CLI 默认的 `--warmup 10 --repeat 1` 恒定触发
`report.cpp` 的 `samples.size() > warmup_count` 断言，报 `timing_summary`。
实施时将新增固定用例的确切入口和真实调度证据补入验证记录。使用 npussh 连接，不复用旧扩展模块声称验证本次代码。

## 验收标准

以下各条的适用对象是**带保留轴的 Mean**；全轴归约不在本次验收内，其闭环条件单列在文末后续项。

1. 单轮、核内多轮和实际多核两阶段 Mean 均与独立参考一致，覆盖不等长尾块。
2. 同一完整归约域的所有轮次/核使用同一 N；对齐 padding 不进入分母，split 来源不重复计数。
3. 每核每个输出 tile 在部分和完成后恰好归一化一次；第二阶段只合并贡献。
4. 缺失来源/完成位置无法证明时明确拒绝，无 tile 分母兜底；其他 op 和 dtype 回归通过。
5. NPU 证据包含当前源码版本、实际两阶段产物及所选计划，不能仅凭随机形状或 fallback 成功判通过。
6. 大幅值用例不能以放宽容差通过；若中间和溢出，重新确定数值策略后再验收。
7. 修正旧注释与能力文档，报告实测误差和性能；未执行的项目明确标记。

## 实施记录（2026-10-14）

### 落地的改动

| 位置 | 内容 |
|---|---|
| `dialect/AFIR/Analysis/ReducePlan.{h,cpp}` | 新增 `MeanReductionDomain` 与 `analyzeMeanReductionDomain`：顺 `from` 链追到 Original 轴、按 destination 轴序去重、拒绝来源丢失/轴未声明/无 size/字面量非正 |
| `conversion/AFIRToAscendC/Implementations/Reduce.cpp` | 分母改成全局 N（`index -> i64 -> f32`，不经 i32）；新增 `buildAtLastRound`；`Muls` 与屏障一起进末轮 `scf.if`；`ReduceTraits<MeanOp>` 开放累加；来源判据读 `afir::getProvenReductionDomain` |
| `dialect/AFIR/IR/AFIRAttrs.td`、`AFIRDialect.cpp`、`AFIRAttributes.h`、`AFIR.h` | 新增 `#afir.reduction_domain<axes, sizes>`（verifier：非空、一一对应、不重复），并进 `AFIRRejectUnknownAttributes` 的已知名单 |
| `dialect/AFIR/Analysis/ReducePlan.{h,cpp}` | 新增 `annotateReductionDomains`（写）与 `getProvenReductionDomain`（读 + 校验绑定） |
| `bridge/AfirBuilder/AfirBuilder.cpp` | `BuildAfirModule` 在全部 template 建完之后调一次 `annotateReductionDomains` |
| 同上 | `kSupportsAccumulation` 七个 op 全为 true 后不再区分任何东西，整个字段删除 |
| `tests/framework/easy_asc_graph/asc_graph_builder.cpp` | 补 `Reduce<ascir_op::Mean>` 的显式实例化 |

### 与本方案的偏离

**1. 「输入是 partial」的判据：先用阶段号顶过，现已按方案的备选实现。**

方案需求 1 要求沿生产者链确认含义，并要求「合法的普通 Mean(workspace tensor) 不能仅因
workspace 存储位置被误拒」。**在下降侧做不到**：能力查询发生在 `AFIRMemoryAllocatePass`
之后，那时缓冲由 `memref.alloc` 产出，reduce 的输入与 `afir.load` 之间**没有 SSA 边**——按
生产者锥走必然到不了 kernel 输入参数。第一版就是这么写的，诊断实测为
`the input provenance never reaches a kernel input`，对**所有** Mean 都失败。

第一轮实施改用 `checkMeanDomainIsFullyVisible`（`afplan.variant` 的 `stage > 0` 一律拒）。
它不丢现有能力（RCore 第二阶段是 `Sum`），但它是**阶段限制**而不是来源证明，正是方案点名
不要的那种保守，已**移除**。

现在按方案 3.4 的备选实现（«在语义仍完整的边界保存 Mean 专用归约域属性，bridge 透传并验证
绑定»）：

| 环节 | 落点 |
|---|---|
| 证明 | `afir::annotateReductionDomains`，由 `BuildAfirModule` 在**全部 template 建完之后**调一次。判据是顺 SSA 生产者锥走有没有走到一块**装着归约产物的跨阶段 GM workspace**（跨 func 的那一半按 buffer id 传递，算到不动点；不收敛则一个属性都不签发） |
| 载体 | `#afir.reduction_domain<axes = [...], sizes = [...]>`，即全局 `N` 的符号表达（`sizes` 之积）及其归属（`axes`）。**缺席即未证明** |
| 读 + 校验绑定 | `afir::getProvenReductionDomain`。属性缺席即失败；属性与轴图**都**答得出来而不一致时失败（一方被改写而另一方没跟上）；轴图答不出来**不**算错误——属性正是为那种图存在的 |

**判据不是「祖先里有没有归约算子」**（评审 P2 第 1 条，已改）。那两件事不等价：

```
x[A, B, C] -> Sum(dim = C) -> y[A, B] -> Mean(dim = B)
```

这里 `Sum` 是用户图里**已经算完**的另一次归约，`Mean` 的域就是 B，分母取 B 完全正确；而本次
改动之前这种图是**支持**的（旧判据只看阶段号）。按算子类别判会把它判成未证明，那是一次能力
回退，不是更安全。改后：

- **同一个 func 内不设防**。那张轴图的 `from` 链原封不动，`Mean` 的归约轴顺着它走到的就是前
  端图里真正的 Original 轴；核内跨轮累加发生在**单个 reduce 算子内部**（destination 当累加
  器）而不是两个算子之间，所以一个 reduce 的 SSA 结果就是它自己那几根轴上算完的结果。轴图不
  说谎，分析的结论就是对的。
- **污染只从跨阶段那一个边界进来**。那一侧的图是分期改写新合成的，`Rm` 那种轴 `from` 为空、
  数的是 partial 份数——轴图唯一说谎的地方。
- 跨阶段这一侧**仍然保守**：写进 workspace 的只要经过归约就拒，不区分「算完的结果」与「等待
  合并的 partial」。要分开这两者，判据得来自分期改写自己的语义身份（它知道哪个输出是
  partial），不能由分析从算子类别反猜。今天唯一的分期形态是 RCore、其第二阶段是 `Sum`，所以
  这一档不丢现有能力；真要支持「上一阶段交完整结果给下一阶段求均值」时再按语义身份放开。

**不动点不收敛时不签发**（评审 P2 第 2 条，已改）。标记只增不减、键是有限的 workspace id 集
合，所以轮数必然以「id 个数 + 1」为界；真超了说明前提被破坏，此时返回失败——全部 Mean 变成
「未证明」，而不是拿一张算到一半的表去签发属性。**这条路径没有测试**：按上述论证它不可达，
构造它需要在 walk 过程中改 IR。

为什么证明必须落在 bridge 建完模块的那一刻：那是 SSA 来源**最后**看得见的地方。时机早一点
（建每个算子时）看不到后面的阶段，晚一点（任何一个 pass 之后）看不到 `afir.load`。

这一改放宽了能力：`stage > 0` 的 Mean 只要来源证明得出，照样下降
（`afir-to-ascendc-reduce-mean-later-stage-accepted.mlir`），而缺证明的在 stage 0 也会被拒
（`...-unproven-domain-rejected.mlir`）。两条合起来钉的是「决定成败的是来源，不是阶段号」。

**没有照做的一处**：属性只在 bridge 写。手写 AFIR（lit、`af-opt`）必须自己带上它——这是刻意
的，证明不能由消费侧凭空补。

**2.（仍然如此）末轮条件的循环不变性没有额外发检查。** 方案要求「不能证明正步长及本层不变性时拒绝」。
步长那一半照做了（取 `scf.for` 自己的 `getStep()`，要求正字面量）。上界那一半**没有**单独
检查：直接复用 `loop.getUpperBound()` 这个 SSA 值，而它是循环的操作数、按支配关系必然在循环
外，本层不变性由此成立，无需额外判据。

### 验收状态

| 项 | 结果 |
|---|---|
| UT（`AFIRDialectTest`） | 155 全过，含 8 条 `MeanDomain*`（纯轴图分析）与 9 条 `ReductionDomain*` / `ProvenDomain*`（来源分类、跨阶段 workspace 的两侧对照、绑定校验） |
| lit（`build/autofuse/mlir/test/lit`） | 231 全过；新增 `afir_reduction_domain_roundtrip` / `_invalid`（属性文本形态与四条 verifier 判据）；Mean 四条：`reduce-mean-accumulation`（原 `-rejected` 改名）、`reduce-mean-allload-ar`、`reduce-mean-later-stage-accepted`、`reduce-mean-unproven-domain-rejected` |
| CPU E2E `load_mean_store` | 六档全过，见下「覆盖矩阵」 |
| CPU E2E 全量目标 | **未通过，且与本次改动无关**：`MlirE2eCpu_load_sum_store_arar` 的生成器挂在 `'afir.sum' op has no supported Ascend implementation`，`git stash` 掉本地改动后逐字复现。它挡在 75% 处，后面的用例不执行——不能拿 `check-e2e-cpu` 当本次的门禁 |
| 真机 / RCore 两阶段 | **已满足（手搭 ASCIR Mean，直达验证）**：18 格（6 shape × 3 输入形态）全 PASS，单阶段与 RCore 两阶段都由产物确认实际执行，见下「真机验收：手搭 ASCIR Mean 的直达验证（2026-10-15）」。验收标准 1、5 由此闭合 |
| 真机 / torch 路径 | **到不了 `afir.mean`**：前端把 `mean` 拆成 Sum + TrueDiv，见下「真机验收：前端不产出 Mean 节点」。`load_mean_store_rcore.py` 作为该行为的回归证据与跳线保留 |
| 设备验证框架扩展（`shape_rule`） | 归约类用例此前无法表达输出形状；两侧规则 + 6 条测试 + 一次反向对照，见下节「框架为此做的最小扩展」 |
| 全量构建（设备，`build.sh --pkg --enable-mlir`） | **修了一处本次引入的链接缺陷**：`AfirBuilder` 没链 `AFIRAnalysis`，`af-afir-gen` 链接期 undefined reference。容器里编 af-opt / 各 e2e 生成器都看不见它，只有全量构建会编那个工具 |
| 大幅值 / 中间和溢出（验收标准 6） | **已处理**：会溢出成 inf，但**不是本次改动引入的**；已决定记录为已知限制，措辞见下节 |

### 大幅值溢出：实测结论与归因（2026-10-14 复跑）

验收标准 6 那条门禁**触发了**，但它指向的不是本次改动。

**现象**（CPU 模拟，探针与日志在 `build/mean-review/`、`build/mean-review2/`）：

| 路径 | 输入 | 结果 | FP64 参考 |
|---|---|---|---|
| 跨轮（`8 x 65536`） | 全 `1e35` | `inf` | `1.0e35`（有限） |
| **单轮**（`8 x 1024`） | 全 `1e36` | `inf` | `1.0e36`（有限） |
| numpy `float32` `mean` | 同上两组 | `inf` | 同上 |

**归因**：溢出发生在 `ReduceSum` **内部**，与「分几轮算」「末轮才缩放」都无关。
决定性证据是第二行——单轮路径同样 `inf`，而单轮是本次改动**之前**唯一被支持的 Mean 形态，
它的求和那一半与改动前逐字相同（改动只换了分母的来源，`8 x 1024` 单轮下新旧分母同为 1024）。
legacy 的两个发射点也都是 `ReduceSum` 之后 `Muls(1/N)`（`reduce_api_call_base.cpp:490`、
`codegen_kernel_loop.cpp:928`），同一个公式、同一个结果。numpy 的 f32 `mean` 亦然。

所以这是 **f32「先求和再除」公式的动态范围**，不是错分母、也不是本次新增的缺陷。
门禁原文那句「不放宽容差」针对的是**错分母**被容差掩盖；这里分母是对的。

**决定（2026-10-15）：取选项 1，记录为已知限制。** 可选项与真实代价如下，选 1 的理由是它
与 legacy、与 numpy f32 行为一致，而另两条都要付真实代价去换一个公式性质的改变：

1. **记录为已知限制**：Mean 继承 `ReduceSum` 的动态范围，与 legacy、与 numpy f32 一致。
   代价是 `|x| x N` 越过 f32 上限时静默给 inf。
2. **求和前先缩放输入**（`Muls(src, src, 1/N)` 再 `ReduceSum`）：这是唯一能修**轮内**溢出的
   办法。代价有三条，都不轻：多一条**满尺寸**向量操作（R 个元素，而现在那条只有 A 个
   元素，量级差 R/A 倍）；它会**原地改写输入 tile**，而融合图里同一块 UB 可能还要喂给别的
   消费者，`isReuseSource = false` 的前提会被破坏；每个元素各舍入一次，小幅值输入的精度
   反而变差。
3. 更宽的累加器：当前 f32 归约的高阶 API 没有这个入口。

**只把缩放移到每轮 `Add` 之前修不了这个问题**——那只影响跨轮累加的量级，而上表第二行证明
单轮、轮内就已经溢出了。

#### 这条限制的措辞（随实现一起维护）

> MLIR 的 f32 `Mean` 按「先 `ReduceSum` 再乘 `1/N`」计算。当 `|x| x N` 超出 f32 上限
> （约 3.4e38）时中间和溢出为 `inf`，即使真实均值在 f32 范围内。此行为与 legacy codegen
> 及 numpy 的 f32 `mean` 一致，不随切分方式（单轮/跨轮/跨核）变化。

验收标准 6 据此**判为已处理**：门禁要求的是「重新确定数值策略后再验收」，策略已确定为维持现
公式并记录限制，不是放宽容差掩盖错分母（分母是对的，见上）。大幅值用例因此**不**进回归门禁
——它测的是 f32 的动态范围，不是本实现的正确性。

### 真机验收：前端不产出 Mean 节点（2026-10-15）

**结论：`afir.mean` 的下降在今天的 torch 路径上不可达，所以验收标准 1/5 不能靠 NPU E2E 闭合。**

做法与证据：在 `DevEnvC_1AtLk` 的**独立目录** `/mnt/workspace/af-mean-accept`（不碰设备上另一
条分支的 76 个脏文件与 5 个 stash）全量编一遍本次源码，搭独立 overlay
`overlay-mean-accept`，跑新增的 `test/e2e/npu/load_mean_store_rcore.py`——它与
`load_sum_store_rcore.py` 同一组「小 A 大 R」shape（`8x131072`、`16x65536`、`20x131072`、
`40x32768`），正是 RCore 真正启用跨核归约的区间。

四格全 PASS（`backend_route=mlir`），但**四格的 `asc_graph.py` 里都没有 Mean 节点**：

```
ascir.ops.{AscBackend, Broadcast, Data, Load, Output, Scalar, Store, Sum, TrueDiv}
```

`aten.mean.dim` 在后端看到图之前就被 torch Inductor 拆成「求和 + 除常数」，`TrueDiv` 还落在
**另一个 op**（`op1_loop_body` 里是 `ops.truediv(load, 32768.0)`，一个 40 元素的
elementwise）。所以这一跑验的是 Sum 的 RCore 两阶段加一次 elementwise 除法，**与本次改动
无关**——拿它的绿当 Mean 的验收就是把「测到了」与「跑通了」混为一谈。

因此：

- `reduce_op_fuzz.py --fuzz-ops mean`、`reduce_fuzz_matrix.py` 的 mean 格子**同样**不覆盖
  `afir.mean`（同一条前端路径）。两处注释已更正。
- 今天仓内唯一产出 ASCIR Mean 节点的是**手搭图**：CPU E2E 的
  `AscGraphBuilder.Reduce<ascir_op::Mean>`，以及 Python 的 `ascir.ops.Mean`。CPU E2E 的六档
  因此仍是 Mean 下降**唯一**的数值覆盖。
- 要在设备上验 `afir.mean` 本身，必须绕开 torch：用手搭的 ASCIR Mean 图走
  `device_validation_runner`（配方见记忆「用 device_validation_runner 跑 legacy/mlir 对照」，
  四条对齐约束：soc_version、显式 ResLimit、4 参 `AutofuseTiling` 转发、flat request 的
  逐元素一致性）。**这条路已经做了**，见下节「真机验收：手搭 ASCIR Mean 的直达验证
  （2026-10-15）」；它确实是一次独立的搭建，不是「再跑一个脚本」。

`load_mean_store_rcore.py` 保留：它是上述结论的可复现证据，也是**跳线**——前端哪天不再拆，
它会自动开始走 `afir.mean`。

### 真机验收：手搭 ASCIR Mean 的直达验证（2026-10-15）

**结论：带保留轴的 Mean 在 910B（`Ascend910_9362`）上单阶段与 RCore 两阶段都实际执行并数值
通过；每核单轮、每核多轮、跨核不等分与轮内不等长尾块都由产物确认走到了。**

上一节的结论是「torch 路径到不了 `afir.mean`」，所以这一跑不走 torch：新增用例
`autofuse/mlir/test/e2e/npu/direct_ascir/cases/reduce_mean_retained_axis/` 手搭
`Data -> Load -> Mean(末轴) -> Store -> Output`，经 `device_validation_runner` 上板。沿用上一节
的隔离目录 `/mnt/workspace/af-mean-accept` 与同一次构建，不碰设备上另一条分支的在制仓库。

#### 硬检查：没测到就不许报 PASS

`input_ascir.py` 在 codegen 之后要求 **AFIR dump 里确实出现 `afir.mean`**，否则抛
`MeanCoverageError`、退出码 3；`run_device_validation.py:458` 对 codegen 非零退出直接判
RuntimeError，于是「未覆盖」不可能冒充「通过」。判据取自 pass dump 而不是我们自己建的那张
图——中间还隔着调度器，它若把 Mean 改写掉（正如前端所为），只看输入图是看不出来的。

反向对照（必须有，否则这条硬检查无法与「恒为真」区分）：把图里的 Mean 换成 Sum，其余不动，
检测器报 `uncovered: no 'afir.mean' in the AFIR dump`。

每一跑都落一份 `mean_coverage.json`，按 `(plan, stage)` 分别记录算子角色（逐 `func.func` 切分，
不整文件判——整文件判会让第二阶段也显示 `mean: true`，把最该验的那条结论糊掉）：

```json
"plan0/stage0": {"mean": true,  "sum": false}
"plan1/stage0": {"mean": true,  "sum": false}
"plan1/stage1": {"mean": false, "sum": true}
```

即 RCore 第一阶段仍是 Mean（带缩放），第二阶段只求和、没有第二次缩放。

#### 实际执行的调度：由产物确认，不由 shape 推断

一份 `device_impl.cpp` 里同时含 plan0（单阶段）与 plan1（RCore 两阶段）两套 kernel，走哪一套
由 host tiling 在运行期填的 `graph0_tiling_key` 决定（顶层 `switch`）。所以执行计划用
`tools/read_executed_tiling.py` 读出来：dlopen 本次跑的 `kernel_module.so`，按后端一字不差的方式
（`ResLimit* = nullptr`，同 `backend/ascendc_backend.cpp:295`）调一次 `AutofuseTiling`，再用
同一产物的 `tiling.h` 解码字段。

| shape | `graph0_tiling_key` | 实际形态 | block_dim | 跨核 R 片数 | R tile | 每核轮数 | 尾块 |
|---|---|---|---|---|---|---|---|
| 40×1024 | 0 | 单阶段 | 5 | — | 1024（= R，不切） | **1（每核单轮）** | A 轴 40 = 5×8 等分 |
| 40×131072 | 0 | 单阶段 | 40 | — | 15672 | **9（每核多轮）** | 末轮 5696，**不等长** |
| 40×131073 | 0 | 单阶段 | 40 | — | 15672 | 9 | 末轮 5697，**不等长** |
| 8×131072 | 1 | **RCore 两阶段** | 40 | 5（8 行×5 片） | 13112 | 2 | 片长 131072/5 **不等分** |
| 8×65537 | 1 | **RCore 两阶段** | 24 | 3（8 行×3 片） | 7288 | 3 | 片长 65537/3 **不等分** + 轮内尾块 |
| 20×131073 | 1 | **RCore 两阶段** | 40 | 2（20 行×2 片） | 13112 | 5 | 片长 65536.5 **不等分** |

两件事值得单独记下来：

- **key 不是按 shape 猜得到的**。`nA = 40`（= corenum）时代价模型选单阶段；`nA < corenum`
  （8、20）时才切到 RCore 两阶段，并真的开了 512B workspace 放 partial。若只按「大 R 必走
  RCore」写验收，三格会测错东西。
- 记忆里「`nA < corenum` 时跨核归约算错」那条，在这三格（8×131072、8×65537、20×131073）上
  **没有复现**：最大绝对误差 ≤ 9.6e-7。

#### 全局分母

缩放因子在两条路上都取**完整逻辑归约域**，每核片长与每轮 tile 长都没有出现在分母里。
`8×65537` 的产物（plan0 stage0 与 plan1 stage0 各一处，整个文件只有这两处 `Muls`）：

```cpp
int64_t v160 = static_cast<int64_t>(c65537_idx);   // 不是 21846（片长），不是 7288（tile）
float v161 = static_cast<float>(v160);
float v162 = c1_f32 / v161;
if (v159) { PipeBarrier<PIPE_V>(); Muls<float, 1>(v113, v113, v162, v157); }   // 仅末轮
```

RCore 第一阶段除的也是 65537：各核只求自己那片的和，但按线性性先除全局 N 再相加，第二阶段
因此只需求和——这正是上面 `mean_coverage.json` 第二阶段 `sum: true, mean: false` 的由来。

#### 输入与参考

`gen_input.py` 三种形态（`AF_MEAN_INPUT_PATTERN`，默认 `shard`）：`ones`（全 1，均值恒为 1，
漏缩放会立刻变成 N）、`ramp`（非零均值、逐元素不同）、`shard`（8 段交替 10.0 / 1e-2 再加逐行
偏移，**各分片均值不同**——跨核归约若把片搞混，结果会偏）。幅值限制在 `[1e-3, 1e1]`：更大的
幅值只会再报一次已记录的 f32 中间和溢出限制，不构成新信息。

参考值是独立的 FP64 `sum / N`（`reference.py`），刻意不用 `np.mean`：`np.mean` 自己也是「先求和
再除」，f32 输入下累加顺序与设备不同，对的和错的会落在同一侧。`N` 直接取 `x.shape[-1]`，不从
输出形状或 tile 尺寸反推——那正是被测实现可能取错的那个量。

18 格（6 shape × 3 输入形态）全 PASS，`atol = rtol = 1e-4`：

| 输入 | 最大绝对误差 |
|---|---|
| ones | 0.0（6 格全为 0） |
| ramp | ≤ 3.8e-6 |
| shard | ≤ 9.6e-7 |

#### 性能：只记录，不作结论

`--warmup 0 --repeat 3` 下的 `runner_wall_clock`（`ones`，3 次均值）：40×1024 32µs、
40×131072 33µs、40×131073 34µs、8×65537 44µs、8×131072 56µs、20×131073 65µs；host tiling
50–74µs。这组数**不能当性能结论**：这些 kernel 的量级与 wall clock 的开销同阶（记忆里「wall
clock 在小 kernel 上会把结论测反」），而且本次只跑 MLIR 一侧、没有 legacy 对照。要比性能得另起
一次对照验证（profiler 或 legacy/mlir 同条件对比），不在本次验收范围。

#### 四条对齐约束，逐项

| 约束 | 结论 | 依据 |
|---|---|---|
| soc_version | 一致 | profile `ascend910_9362` 的 `soc_version = Ascend910_9362`，与设备一致 |
| 显式 ResLimit | **不需要显式传，且默认值就是设备实际能力** | 后端按 `RunTiling(..., nullptr)` 调用（`ascendc_backend.cpp:295`）；生成的 host tiling 里 `g_no_limit_res = {1, 40, 0, 196608}`，来自 codegen 期的平台设置，`AutofuseTiling` 在 `res_limit == nullptr` 时用它，并把 UB 记为 `196608 - 256 = 196352`。探针读回的 `ub_size = 196352`、`block_dim ≤ 40` 与之吻合 |
| 4 参 `AutofuseTiling` 转发 | 已满足，无需改动 | 生成签名 `AutofuseTiling(AutofuseTilingData*, uint32_t*, uint32_t*, ResLimit* = nullptr)`，后端函数指针 `AutofuseTilingFunc` 正是 4 参 |
| flat request 逐元素一致性 | 满足 | `tensor_files == inputs`、`input_count == |inputs| == |input_specs|`、`launch_abi == abi`，18 格都过了 `ValidateFlatRequestForBackend` |

`profiles/ascend910_9362.json` 这次只加了 `float32` 到 `dtypes`。`core_type: 40` /
`max_block_dimension: 40` / `ub_size: 196608` **原样不动**：平台 ini 里 `Ascend910_9362` 的
`vector_core_cnt = 40`、`ub_size = 196608`，profile 写的就是设备实际能力。（此前我在记忆里把
「40 核挡掉实际 48 核」记在了这颗芯上，是错的：48 核是 Ascend910B1。）

#### 框架为此做的最小扩展

归约类用例此前**根本表达不出来**：`dynamic: true` 只有「等于本次 shape」一种含义，于是 Mean
的输出被当成 `[A, R]`，设备侧按 `A*R` 收结果，报出来是一句笼统的
`precision_failed / output 0 metadata mismatch`。写死 `shape` 也不行——一个用例要跑 6 个 shape。

因此加了 `shape_rule`（`same` 缺省 / `reduce_last`），**两侧同一套规则**：
`python/shape.py::resolve_tensor_shape` 构造请求，`device_validation_runner.cpp::`
`ResolveDeclaredShape` 校验契约。两侧必须同时认同一组拼写——少认一条，表现是一句不指向真因的
`flat variant or step is not declared by case`（这次就先踩了一遍）。不认识的拼写直接报错，不静默
退回 `same`。

覆盖：`test_device_validation.py` 3 条（规则本身、未知拼写、输出 spec 真的带上了算出来的形状），
`flat_step_contract_test.cpp` 的 `RunShapeRuleCases` 3 条（接受 `[A,1]`、拒绝写成本次 shape 的
输出、拒绝未知拼写）。反向对照做过：把 `ResolveDeclaredShape` 里 `reduce_last` 那支改成
`if (false)`，重编 runner 后契约测试立刻报
`reduce_last output shape must satisfy the case contract`；还原后再过。

#### 证据位置

设备侧 `/mnt/workspace/af-mean-accept/`：`dv-s<rows>x<cols>-<pattern>/<case>-*/` 每格一份，含
`tiling.h` / `host_impl.cpp` / `device_impl.cpp` / `mean_coverage.json` / `afir_dump/`（逐 pass
AFIR）/ `input_0.bin` / `output_0.bin` / `report.json`；`neg-control/` 是硬检查的反向对照；
上面那张执行计划表的取数脚本已入仓：`tools/read_executed_tiling.py`。

#### 这一跑**不**涵盖什么

- 全轴 Mean（无保留轴）仍是后续项，见「后续项：全轴 Mean 的最小闭环」。
- 大幅值下 f32 中间和溢出仍是已知限制，`shard` 的幅值刻意避开它。
- 来源分析的两个问题——「完整上游归约结果被当成 partial」与不动点截断——由 UT 对照
  （`ReductionDomainIsProvenAfterACompletedUpstreamReduction` 对
  `ReductionDomainIsNotProvenAcrossAWorkspaceHoldingAPartial`）与 fail-closed 的
  `computeWorkspaceCarriesReduction` 各自闭合，**真机数值通过不替代它们**：这一跑的两个 plan
  都只有一层归约，根本到不了那两条判据的分界处。

### `load_mean_store` 的覆盖矩阵

「有效」一列不是形状看起来覆盖了什么，而是**反向对照实测**的结果：把末轮条件短路成
`if (false && *atLastRound)` 让 `Muls` 每轮都发，看该档会不会挂。挂了才说明它对末轮逻辑
有区分度。

| 形状 | 轮数 | 末轮 | 反向对照下 | 它**独有**地覆盖什么 |
|---|---|---|---|---|
| `16x32` | 1 | — | PASSED（无区分度） | 基准 AR，非累加形态不发 `scf.if` |
| `16x37` | 1 | — | PASSED（无区分度） | 分母必须是 37 而不是 `align(37) = 40` |
| `8x1024` | 1 | — | PASSED（无区分度） | 一轮算完的大 R，与下面两档对照说明分母同源 |
| `8x65536` | 多 | 等长 | **FAILED** ✓ | 「只在末轮除一次」 |
| `8x65537` | 多 | **1 个元素** | **FAILED** ✓ | 末轮是尾块时，末轮条件仍只触发一次、分母仍是 65537 |
| `ra_37x16` | 1 | — | 未测 | RA 形态，分母取 R(37) 而不是 A(16) |

来源**分类**的反向对照：把消费侧判据改回「祖先里有没有归约算子」，
`ReductionDomainIsProvenAfterACompletedUpstreamReduction` 立刻 FAILED，而跨 workspace 那条
仍然 PASSED——两条合起来说明放宽的只有「同一个 func 内的完整上游归约」这一档，跨阶段的保护
没有一起松掉。

另一个反向对照针对**来源证明**那一步：把 `BuildAfirModule` 里的 `annotateReductionDomains`
注掉重编 `16x32`，生成器挂在 `error: 'afir.mean' op has no supported Ascend implementation`。
所以这六档的绿**可归因**到 bridge 真的写了属性、且属性真的活过了整条流水线——这是 lit 替代
不了的一环：lit 的输入是手写 IR，属性是我自己填进去的。

**CPU 覆盖边界与补充证据**：

- RCore 各核分片不等长、每核多轮不由上述 CPU 用例证明；已由本次直达 ASCIR Mean 的真机验收补齐。
- legacy「仅 R 切分而 A 未切分」那一档是否漏缩放 —— **已取证，结论见下节**。
- Mean 的分母与归一化落点由 Mean 自身的数值用例验证，不能由 Sum 用例替代。

### legacy 的「仅 R 切分」漏档：取证结论（2026-10-15）

流程设计第 1 步要求先验这一档。结论是：**它存在于判据里，但在真实调度产物里不可达；真正
未验的是一个更窄的形态——全轴归约。**

两条 legacy 判据不等价，这一点没变：`IsReduceDoubleTile` 数的是某张量 `vectorized_axis` 里
`TileInner` 的根数（`codegen_kernel_loop.cpp:557`），多轮与否看的是循环轴的来源。但扫过仓内
**全部** 20 份真实调度语料（RCommon / AllLoad / RCore / ARAR / ARA / RA）之后：

| vectorized_axis 形态 | 样本 | TileInner 根数 | 两条判据 |
|---|---|---|---|
| A、R 都切（`z0t`,`z1t`） | 14 份 | 2 | 多轮为真、double tile 为真 → 循环外发 `Muls` ✓ |
| A 切、R 整根（`z0t`,`z1`） | AllLoad 2 份 | 1 | 多轮为假 → api call 内发 `Muls` ✓ |
| 手写夹具（`a`,`r` 全 Original） | 4 份 | 0 | 非调度产物 |

**「R 切而 A 不切」一份都没有**，而且有机制解释：`TilingGroup::NormGroup`
（`optimize/autoschedule/tiling_group.cpp:720`）只在组**为空**时才塞默认值，所以只要 A 轴
存在，`AutoSchedule` 的三重枚举（`autoschedule.cpp:281`）就一定给它配一个切分 id；A 与 R
要切一起切。AllLoad 是唯一让某根轴保持 Original 的模板，而它按定义保持的是 **R**。

所以此前那句「legacy 在仅 R 切分时两处都不发缩放」**没有依据，应当撤回**。

**真正剩下的口子**：A 组为空只有一种情形——**全轴归约**。那时 `vectorized_axis` 里只有 R 的
`TileInner`（1 根）→ `IsReduceDoubleTile` 为假，而 R 被切 → 多轮为真，两处确实都不发。
但这一档**无法从现有语料确认**：CPU E2E 的 `OUTPUT_SHAPE` 里没有任何 `[1, 1]`，lit 语料里
每一份都留着保留轴，全仓**没有一个全轴归约的调度样本**。

这个口子对 MLIR 侧同样未验：本次新增的六档 `load_mean_store` 也都带保留轴。

### 后续项：全轴 Mean 的最小闭环

**已决定（2026-10-15）：不在本次范围，留作后续项。** 验收口径因此收窄到「带保留轴的 Mean」
（见「目的」与「验收标准」），不得据本次结果宣称 Mean 场景已完整支持。
闭环的最小动作，顺序不可省：

1. **先造出一个确实跨轮的全轴 Mean**：输出 `[1, 1]`，R 的元素数超过一个 UB tile（CPU 侧
   49152 个 f32 是门槛，参照 `8x65536` 那一档的理由）。`Case.cpp` 的归约轴选法要改成全轴，
   `KernelTest.cpp` 的参考值要从按行改成单标量。
2. **留下实际调度描述**：dump 选中候选的 AFIR（`vectorized_axis`、各轴 `axis_type`/`from`），
   证明它真的是「A 组为空、R 被切」而不是被调度器改成了别的形态——否则测的不是这一档。
3. **两套生成代码都存档**：MLIR 侧与 legacy 侧各一份 kernel 文本，用于区分下面两种结果。
4. **用非零均值输入对照独立参考**：全零或对称输入会让漏缩放与正确结果都趋近同一个数。

两种可能结果的归因**完全不同**，必须分别判定，不能合并成一句「全轴 Mean 不行」：

| 观察 | 归因 | 处置 |
|---|---|---|
| 成功生成，但结果缺一次除法（= 和，或 `S/N` 之外的值） | 漏缩放：判据覆盖不到这个形态 | 补发射点/补判据，legacy 与 MLIR 分别评估 |
| 在能力或布局阶段就被拒 | 该形态压根进不到归一化代码 | 先判断拒绝是否恰当；若恰当则只需把拒绝原因写准，无数值缺陷 |

注意 legacy 侧走不通「直接跑生成器看文本」这条路：e2e 的 tiling stub 是给 MLIR 流程用的，
不带 `AF_ENABLE_MLIR_CODEGEN` 时挂在 `GetTilingHeaders`（`tilings(...) is invalid`），kernel
根本不产出。要么另找 legacy 的 tiling 入口，要么退回「只读两条判据各自消费的调度产物」。

### 一条必须随用例一起维护的判据

CPU E2E 的 UB 是 196608 字节 = 49152 个 f32，所以「看起来很大的 R」可能一轮就算完、
`accumulationAxes` 为空，末轮那段代码压根不发。**实测**：把末轮条件短路成
`if (false && *atLastRound)` 让 `Muls` 每轮都发之后——

- `8x1024`（8192 元素）与 `16x37` 仍然 **PASSED**，对跨轮逻辑零区分度；
- `8x65536`（512K 元素）**FAILED**，这一条才测得到。

所以 `load_mean_store` 里只有 `8x65536` 覆盖「只在末轮除一次」。**改那条形状之前必须重做这个
反向对照**，判据写在该用例的 CMakeLists 注释里。推论一般化：用 CPU E2E 验切分/多轮/尾块相关
行为时，先确认形状真的触发了切分，别从数值大小推断。

### 设计文档检查结果

- [x] 已读取设计模板并覆盖全部章节。
- [x] 已按 cross_feature_check.md 逐项分析。
- [x] 已按编码红线检查语义、计数、同步、生命周期和兼容边界。
- [x] 已设计 gtest/lit/CPU/NPU 与性能验证；本文不宣称这些测试已完成。

### 评审修订说明

上述 legacy 单 TileInner 漏档在初稿里由分支条件推得，已按实施首步取证：**带保留轴的那一档不可达，推断撤回**；剩下的全轴归约一档仍缺真实调度与数值证据，按上节的最小闭环单列为后续项。同步方面，legacy 未显式插入屏障不能证明 MLIR 不需要屏障；Add 前的 PIPE_V 不能单独证明 Add 后到 Muls 的依赖已满足，且首轮为 DataCopy。保留当前 emitMeanScale 已有的同步，不在此次需求中凭先例删除；若优化，须另以对应 API/同步契约和设备产物验证。

### 覆盖检查的运行隔离（2026-10-15）

每次 codegen 创建独立的 `mean-run-*` dump 子目录，仅扫描本次目录。调用方指定的
`AUTOFUSE_DFX_FLAGS.debug_dir` 作为父目录保留，其余 DFX 选项保持；生成结束或异常时恢复原环境。
历史 dump 不删除。`mean_coverage.json` 记录本次绝对 `dump_dir`，`mean_files` 相对该目录解析。
回归覆盖同一 output-dir 先 Mean 后 Sum、调用方自定义 dump 目录和生成异常时的环境恢复。

本次在开发容器复跑设备验证框架 Python 测试：**174 passed / 9 skipped**，含新增 3 条隔离回归。
测试使用替代 codegen 写出本轮 dump，实际调用生产入口和覆盖检测；未重新执行 NPU。
本次修复仅涉及验收工具和文档，不改变已验收的 Mean kernel 或已接受的能力边界。


### 2026-10-17：真机测试契约隔离

归约 `shape_rule` 仅由 `mlir/test/e2e/npu/direct_ascir/tools/run_reduce_validation.py`
消费。入口要求显式 `--shape A R`，按本次形状生成静态输入/输出描述后调用上游公共 runner；
产物目录保留静态 manifest，Python 建图和参考模块链接到原用例源码。
`autofuse/tests/st/device_validation` 的八处剩余修改全部恢复上游。
直接 ASCIR Mean、当次 AFIR 覆盖门禁、FP64 参考值及 1e-4 容差均保持。
本次验证：Python 182 passed / 9 skipped / 3 deselected；上游 C++ flat contract
和 MLIR 静态 Mean contract 均通过（含错误形状/dtype 拒绝）。未重新上板，不把本次
host 回归作为新的 NPU 数值验收证据。详见 direct_ascir/README.md。


### 2026-10-17：隔离后 910B 真机回归（9/9）

使用当前工作区源码白名单快照（基于 `62d8bfca`，5177 个文件逐项 SHA256 核验），
在 `DevEnvC_1AtLk` 的独立目录
`/mnt/workspace/agent/mean-isolation-62d8bfca-20261017` 从零构建 `pyautofuse`、
`af-afir-gen`，并从恢复后的上游源码重编公共 `device_validation_runner`。
匹配版本的 Python overlay 与 codegen 动态库均来自该目录，未使用旧验收二进制。
LLVM/CANN 和第三方构建依赖复用设备既有安装；未改动设备上的其他源码目录。

通过 MLIR 的 `run_reduce_validation.py` 入口运行以下 3 shape × 3 输入形态
（ones / ramp / shard），9 格均实际执行并通过，`atol=rtol=1e-4` 未变。

| shape | 实际 plan / block_dim | ones 最大绝对误差 | ramp 最大绝对误差 | shard 最大绝对误差 |
|---|---|---|---|---|
| 40×1024 | plan0 单阶段 / 5 | 0 | 3.8146973e-6 | 4.7683716e-7 |
| 40×131073 | plan0 单阶段 / 40 | 0 | 3.8146973e-6 | 9.5367432e-7 |
| 8×65537 | plan1 RCore 两阶段 / 24 | 0 | 4.7683716e-7 | 4.7683716e-7 |

实际 plan 由每格本次 `kernel_module.so` 的 `AutofuseTiling` 读回，不按 shape 猜测。
汇总脚本同时断言：报告为 execution/passed、零 mismatch、静态输入为 `[A,R]`、
输出为 `[A,1]`、被执行 plan 的 stage0 确有 Mean；RCore stage1 为 Sum 且不含 Mean。
因此本轮覆盖了静态契约适配、新入口、上游公共 runner，以及当前源码的直接 Mean 执行链路。
全轴 Mean、ArgMax 和极大幅值导致 f32 中间和溢出仍不在本轮验收范围。

远端 `acceptance-summary.json`、`acceptance-evidence.tar.gz` 保存报告、实际 tiling、
Mean 覆盖、源码清单及二进制哈希；本地证据回收至
`build/mean-isolation-npu-evidence/`（不入仓）。用户已明确保留远端目录。

提交前附加检查：平台无关分层、AFIR 依赖方向、AscendC td 可达性通过。
`ascir_op_lowering_matrix.csv` 快照检查发现 rebase 后已有的注册表漂移；扫描器、
注册输入与快照在本次改动前后均未变化，未将该既有问题混入此次隔离修改。
