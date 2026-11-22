# AFIR Transpose 泛化组合资格验证与候选修复设计

> 状态：五个探索组合尚未通过 legacy CPU 数值门槛；后文修复方案为候选，尚不具备实施前提。2026-11-02。
> 前置：[需求设计](2026-10-19-afir-transpose-e2e-design.md)、[测试设计与迁移验收](2026-10-29-afir-transpose-generalization-test-design.md)。
> 验证顺序：构图与 oracle 审计 → legacy Inductor 生成/编译/数值通过 → 同图同平台 MLIR 对照 → 确认差异后修复。
> 不能把“legacy optimizer 成功”或“legacy 代码生成成功”替代端到端通过；本轮没有实施 MLIR 修复。

# 简介

## 目的

集中记录五类泛化问题的已知证据、待验证假设、修复边界和验收方法，供实现与检视使用。
本文仅提出有前置条件的候选方案，不将探索失败直接认定为 MLIR 缺陷，不宣称泛化问题已经修复。

## 范围

包含 Cast/chain/broadcast/Sum/GEWhere 与 Transpose 组合的 MLIR 布局问题调查及修复设计。
legacy 迁移只关注 Inductor；TF 不在交付范围。借用的框架无关 optimizer 图仅为补充泛化探索，
不改变原有迁移验收。排除 legacy ATT/v35 公共实现、SuperKernel 和新调度策略修改。

# 总体概述

## 软件概述

### 项目介绍

Graph-autofusion 的 AFIR/AscendC 下降复用既有调度候选及资源信息。
单图生成或基本转置数值通过，不能证明转置与搬运、广播、谓词及归约组合时的布局一致。
原有需求设计与测试验收继续分别维护，后续缺口修复统一在本文维护。

### 产品环境介绍

CANN 9.2.0、同源码生产 codegen/pyautofuse overlay、CPU E2E 与严格 MLIR 入口。
mem/reg 使用各自既有目标能力；910B3 已有数值证据不覆盖本文失败组合，未验证 a3 或性能。

## 软件功能

证明搬运访问不超过资源容量；保持广播、置换及归约逻辑轴身份；支持同逻辑域下不同 dtype
的独立物理步距；无法合法实现的候选应准确拒收，不静默回落。

## 设计约束

遵循既有 TensorDesc/TransferPlan/canonical resource 的职责，能力检查和发射共用判据。
不硬编码图名、芯片或框架，不修改公开 tiling ABI，不修改安装环境，不引入独立 allocator。
新增 API 或布局转换须同步语义命名；构建统一限制为 `-j 8`。

## 假设和依赖关系

原始构图有效、优化后的轴 provenance 完整、SDK DMA 对齐语义与测试运行目标匹配均须逐项验证。
当前保留的是历史失败与源码证据；最新上游可能改变调度候选，须重新复现后确认根因。

# 需求分析与设计

## 整体介绍

以同一候选为单位贯通逻辑域、物理布局、DMA 参数及资源容量，再在对应信息所有者修复。
不把五类失败统一归为 Transpose API 缺陷，不同时叠加多层特判。

### 原因分析与证据强度

| 场景 | 已确认事实及源码依据 | 尚待确认的根因 |
|---|---|---|
| Cast→Transpose `[64,64]` | 最新基线 CPU 仍报告 GM→UB 需要 256B、分配 128B。对实际 kernel 入口采样得到 `blockCount=8, blockLen=8, srcStride=56, dstStride=0`，元素为 half：每行有效 16B，但 DMA 每行按 32B 推进，8 行实际 256B；紧密 tile 的 `8*8*2` 仅 128B | 已确认 DMA 对齐行布局与当前紧密分配/view 契约不一致。实施前还需核对同调度 legacy 的行布局、queue member 和后续 Transpose/Cast 读法，决定规范化物理行距还是改变合法搬运分块，不能仅扩大 buffer |
| Transpose chain `[2,3,4,5]` | 同调度旧产物中 legacy 对跨步内维发 `blockLen=1` 与外循环。当前 `TransferPlan.cpp::emitScheduledTransfer` 直接取最后一维 extent 为连续 block 长度，缺少两侧单位步长的完整前置证明。旧实验修复该点后仍有 index 56 错值，已撤回 | 单元素分块不能独立解决紧密 UB Store 的硬件对齐推进；最终错误还需逐 tile 验证 load、transpose 和 store 的地址映射 |
| Broadcast→Transpose `[1,37,23]→[8,23,37]` | 最新失败 Transpose operand 的轴 0 为 repeat=1/stride=0，destination 为 repeat=8/stride=851；各自向量化轴为 `[6,4]` 和 `[4,6]`，均不含轴 0。当前 verifier 对完整 repeats 求积，源为 851、目的为 6808，因而拒收 | 已确认直接原因是完整 descriptor 的广播外域被纳入纯置换元素数比较；还需证明轴 0 的外层循环 provenance/覆盖及每次 tile 的双射，才能区分合法外层复制与真正的非法 Transpose 扩域。不能仅凭 vectorized_axis 不含轴 0 就跳过校验 |
| Transpose→Sum `[2,37,23]→[2,23,1]` | 对最新原图调用既有 `buildReducePlan(op, &diagnostic)`，明确返回 `kept and reduced vectorized axes alternate more than once, which no two-dimensional AR/RA layout can express`。vectorized_axis 为 `[4,1,6]`，destination vectorized_strides 为 `[pitch,0,1]`，对应 A-R-A | 当前 ReducePlan 只表达二维 AR/RA，不能表达该候选的 A-R-A 物理布局；不是缺少 Sum 注册或 scratch 的泛化猜测。设计优先研究剥离外 A 循环后复用现有 RA 实现，而不是合并不相邻 A 轴 |
| GE→Where→Transpose | 完整图含独立 tiled GE，f32 输入 stride 8、ui8 输出 stride 32。`ElementwiseTransferPlan.cpp::checkLocalViewCompatibility` 要求各方 extent/localStride 相同，不能表示这种不同 dtype 带来的物理行距差异；直接谓词用例通过不代表完整图通过 | 需确认不同元素步距对应同一逻辑迭代域后，才能选择逐行实现或布局规范化；Where 的 condition/value/output 必须独立核对 |

本节已用最新 `aaab745c` 上游和本分支实现重新构建并复现；Cast、Sum、broadcast 的补充采样来自同一版本。chain 的旧实验仅解释调查方向，已撤回，不能视作当前修复证据。后续实施仍须记录具体候选与 tiling，避免把不同候选的布局混用。

## 功能需求

### 功能需求 1：搬运发射与资源容量采用一致的物理布局契约

1. 输入：既有 `TensorDescAttr`、`TransferPlan`、canonical resource/member/reuse 信息及目标 API 能力。
2. 处理：先证明块内 GM 与 UB 均连续。最内非单位步长时，不将 extent 直接当连续 block；
   通过既有计划表达外循环和合法连续块。能力查询与发射消费同一规范化结果。
3. 容量：分别计算逻辑有效元素跨度、DMA 实际访问跨度和 API 对齐 footprint；不得混用。
   对二维 DMA，以 SDK 定义的实际 UB 行推进 `P` 与行访问字节数 `L` 校验
   `offsetBytes + (blockCount - 1) * P + L <= memberCapacityBytes`，多层循环再加各层最大偏移。
   `P/L` 须由所选 API 和方向确定，不能一律以逻辑 stride 或名义 count 代替。
4. 输出：可证明安全的 DMA 参数及容量约束；未满足时由候选准备/能力查询拒收。

资源侧继续由现有 canonical resource 机制拥有分配事实，搬运侧声明真实访问需求。
只在证据证明原资源表达式漏算时修改对应 MLIR 适配点；不扩大 legacy ATT 公共分配规则，
不按图名翻倍，不增加第二套字段表或独立 allocator。queue member offset 与 reuse 边界同时校验。

紧密 Store 若不能由目标 block DMA 表达，优先使用已有合法布局转换并纳入显式 scratch/resource
生命周期；只有证明无额外转换也安全时才走直接路径。不能通过删除几何校验、单元素循环
或把负/非整除 stride 转成无符号数绕过硬件限制。若现有转换不能表达，先保持准确拒收，
再单独设计该边界，不能继续堆局部特判。

### 功能需求 2：广播、转置与归约保持逻辑轴身份

1. 输入：同一候选的原始图、优化图、axis provenance 和 operand/result/destination desc。
2. 处理：广播扩域只由 Broadcast/Nddma 或已成功的调度变换解释；保留 Transpose 的纯置换
   元素数契约。若 bridge 将广播物理存储量误当逻辑域，修正 descriptor 来源；若测试构图有误，
   修正图并证明 NumPy/Torch 语义一致。当前原图的 Broadcast 确实将 `[1,37,23]` 扩至
   `[8,37,23]`；需追踪该外域如何被优化/bridge 表达。候选正向条件至少包括：外域扩张可由
   调度循环 provenance 覆盖、源为 repeat=1/stride=0、该域不参与本次 API 向量化、两侧每次
   tile 在轴重排后元素数相同且输出覆盖无遗漏/重叠。条件未证全时继续拒收。
   不得把所有 Transpose 的 equalElementCount 改为 false。
3. Sum 的 A-R-A 已确认无法进入现有二维计划。优先在现有归约/向量循环规划边界显式剥离外 A，
   对剩余 R-A 复用既有 RA 归约；分别计算 input/output/scratch 的外 A 偏移，并证明初始化、
   round_result 和跨 tile 累加仍属于各自逻辑输出。不能直接把两根不相邻 A 相乘。
   若已有结构无法表达该循环与资源生命周期，再评估显式布局转换到 AR/RA，记录额外 scratch
   和代价；不直接新增独立 Sum 算法或修改调度器。
4. 输出：语义一致的 desc 或可定位的能力拒收原因，不靠回落或替换输入图获得假通过。

### 功能需求 3：不同 dtype 的逐点运算支持独立物理步距

1. 输入：Compare/Where 各操作数的逻辑轴、有效 extent、dtype 及物理 stride。
2. 处理：复用现有 elementwise 计划边界，先证明共同逻辑迭代域，再分别计算各张量的行起点。
   只有各行内部满足所选 API 连续要求时，才复用现有逐行发射；否则走已建模的布局转换或拒收。
   不把所有操作数改成相同步距，不仅删除 `checkLocalViewCompatibility` 的比较。
3. 输出：正确的谓词值和按逻辑元素对齐的 Where 结果；不改变 compare 真值编码或 dtype。

## 非功能需求

### 可维护性

复用既有分析与计划边界，不为每张失败图单独建分支。

### 可测试性

先用 UT/lit 表达通用规则，再以原始整图的独立数值 oracle 验证。

### 可移植性

以目标能力和公开 SDK API 确定对齐与指令行为，mem/reg 分别验收。

### 可靠性

检查容量、偏移及对齐算术，复用既有错误传播；不能将未知符号默认为安全。

### 特性交叉影响

| 场景 | 适用性与结论 |
|---|---|
| SuperKernel Python、C++/AOT | 不适用，不修改 SuperKernel 或其交付 |
| Autofuse 图优化 | 适用，只核对已有候选/轴来源；不擅自调整调度选择或改写顺序 |
| Codegen/Backend | 适用，覆盖资源、搬运、逐点及归约边界 |
| AscendC/Runtime | 适用，核对 API 的对齐、参数单位、scratch 与生命周期，无新增非开放 runtime 接口 |
| Python/C++ 绑定 | 适用，测试整图错误传播与严格接管；不新增公开绑定 |
| 构建/打包 | 设计不新增依赖、安装文件或选项；若实现新增源文件，沿用现有 CMake 目标 |
| 测试/覆盖率 | 适用，见测试文档补充矩阵 |
| 性能/日志 | 适用，见下节；不增加默认诊断开销 |
| 兼容性 | 适用，保留已有合法布局、候选与 tiling ABI |

已按编码红线检查：不引入敏感信息、芯片/框架特判、未检查的索引/容量算术；图改写须保持
数据和控制依赖；资源归属显式；不依赖跨 SO 静态析构；不修改安装环境。既有能力拒收在
完整方案验证前保留，不以减弱断言、扩大 oracle 容差或静默回落替代修复。

## 性能

### 编译时长

布局分析复用现有 descriptor，目标复杂度随轴数线性增长；不新增全图扫描或求解。

### 执行性能

新增转换可能增加搬运与循环，须在 INFO 关闭时测量；当前未测，不承诺性能通过。

### 内存和产物大小

如需 scratch，纳入 canonical resource 和候选代价；报告峰值 UB 与产物变化，不预设零开销。

## 接口设计

### 新增/修改接口描述

预期仅扩展 MLIR 内部计划信息；公开 Python/C ABI、tiling 布局和安装路径保持既有契约。
具体接口随证据确定；若必须改变公开布局，先修订设计和兼容测试。

### 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|---|---|---|---|
| 接口说明 | 内部接口评审 | 可能 | 确认真实信息缺口后评审 |
| 接口说明 | 文档 | 是 | 本文随实现更新 |
| 接口兼容 | 行为 | 是 | 既有合法候选与尾块回归 |
| 接口兼容 | ABI/API | 设计不涉及 | 不修改公开 tiling 契约 |
| 接口约束 | 报错 | 是 | 区分缺失、不可证明与能力不足 |
| 接口测试 | 独立覆盖 | 是 | 复用 analysis/resource/bridge 测试 |

## 软件设计

### 关键数据结构

descriptor 为 canonical 属性，计划为候选局部数据，scratch 由现有 resource schema 持有。

### 关键技术/算法

以 SDK 定义的实际行推进计算 footprint，与 member capacity 对账；逻辑域证明与物理寻址分离。

### 流程设计

原图有效性 → 最新基线重现 → 定位首个契约失配 → 所有者修复 → 候选选择 → 发射 → 数值回归。

### 对子模块的修改

| 模块 | 职责与拟修改边界 |
|---|---|
| `Analysis/TransferPlan` | 推导可表达的 DMA 几何，能力检查与发射共用 |
| `Implementations/DataMovement` | 按选择的 API 发射参数，核对方向与单位 |
| `MemoryAllocate/CanonicalResources` | 校验实际 footprint 与容量，不另建 allocator |
| bridge | 核对 axis provenance 和 descriptor 来源，修复仅限已证明的失配 |
| elementwise | 同逻辑迭代域下独立计算各张量物理行起点 |
| ReducePlan/ReduceGeometry | 先传递已有失败原因，按实际缺口修改 |


### 错误处理

#### 系统错误

复用既有分配、编译、runtime 错误传播和临时目录清理，不吞异常或隐藏 scratch。

#### 接口错误

复用能力原因与 ReducePlan 等已有诊断；调试导出只在显式开关开启时执行。

## 安全检查

已读取编码红线：不引入敏感数据、未检查的索引/容量算术、非开放 runtime 接口或跨 SO 静态析构。
图改写保持数据与控制依赖，资源生命周期显式；不扩大修改到无关模块。

## 兼容性检查

保留既有合法布局、候选选择、Python 导入与异常语义、公开 ABI 和安装目录。
不通过放宽 verifier、容差或静默回落达到假通过；未选中候选拒收不能阻断其它合法候选。

## 测试设计

### 测试边界

分析 UT、生成 lit、严格整图 ST、CPU E2E 和匹配 overlay 的 NPU 数值分层记录。
不新建 RDV 框架；reg 真机不可用时明确未执行，不以 mem 真机替代。

### 测试用例设计

本节定义未来修复的成功标准，不代表以下失败已关闭。
重新生成所有产物并记录同一候选的优化图、canonical descriptor、tiling、资源大小和源码版本；
旧版 index 56、stride 8/32 等只用于提出假设，不能替代最新上游的复现。

| 验证项 | 输入与变体 | 通用断言及证据 |
|---|---|---|
| DMA 容量/布局 | 原 Cast `[64,64]` f16→f32；去 Cast、去 Transpose 的对照；不同 dtype、尾块及多 tile | 对每个 DMA 校验实际行推进、物理 footprint、member offset/capacity；捕捉第一处失配；不能只把 buffer 翻倍后看绿 |
| 级联置换 | `[2,3,4,5]` 两次 perm 组合为 `[0,3,2,1]`；内维 GM/UB 单位与非单位 stride 的组合 | UT/lit 检查连续块前置条件和参数；CPU 独立坐标 oracle 全元素精确比较，load/store 两方向与紧密/对齐 UB 分别覆盖 |
| 广播 | `[1,37,23]→[8,37,23]→[8,23,37]`；广播轴在外/内、单元素维 | 原图到优化图逻辑域一致；Broadcast 允许扩域、Transpose 仍为纯置换；相邻非法 Transpose 元素数负例仍拒收 |
| GE/Where | 原 `[41,54,38,55]` 图；f32/ui8 不同行距及尾块 | 单独确认各方逻辑坐标相同；分别计算输入、谓词、输出行起点；Compare/Where 直接用例与带 Transpose 的整图均通过 |
| Sum | `[2,37,23]`、perm021、reduce 最后维，输出 `[2,23,1]` | 记录 ReducePlan/checkReduceLayout/驻留/目标能力的首个实际失败原因；校验逻辑归约轴及初始化/累加次数；原有 AR/RA、尾块和多核场景无回归 |
| 候选与资源 | 多候选，至少一个布局不可表达但另一个合法；queue member/reuse，多 group | 未选中失败候选不能阻断合法候选；member 不跨界、无溢出；不能通过 fallback 冒充 MLIR 成功 |
| 语义名称 | 仅在实现引入新 API/转换分支时适用 | 对应实际分支及完整逻辑身份；失败/复制/reuse/回退后名称与选中实现一致，沿用公共词表 |

复用 `TransferPlanTest`、既有 canonical resource、elementwise、ReducePlan 及 bridge 测试，
不为每张失败图建立一套布局特判。修复前先确认原图有效性；独立 oracle 不从生成 tiling/kernel
反推期望，保留原有容差，NaN 未写位置不得视作通过。


### 测试命令

环境配置沿用仓库构建指引，同一 build 的构建依次执行且统一 `-j 8`。可执行检查入口：

```bash
# 通用回归；具体构建目录按既有配置。
cmake --build build/autofuse/mlir/test/ut -j 8
ctest --test-dir build/autofuse/mlir/test/ut --output-on-failure -j 8
cmake --build build --target af-opt ascir-translate af-afir-gen -j 8
llvm-lit -j 8 -v build/autofuse/mlir/test/lit

# 单项生成/数值验证：先构建对应 target，不能执行陈旧产物。
cmake --build build --target MlirE2eCpu_cast_transpose_64x64 -j 8
ctest --test-dir build/autofuse/mlir/test/e2e/cpu --output-on-failure -R '^MlirE2eCpu_cast_transpose_64x64$'
cmake --build build --target MlirE2eCpu_transpose_chain_2x3x4x5 -j 8
ctest --test-dir build/autofuse/mlir/test/e2e/cpu --output-on-failure -R '^MlirE2eCpu_transpose_chain_2x3x4x5$'
cmake --build build --target MlirE2eCpu_broadcast_transpose_8x37x23 -j 8
cmake --build build --target MlirE2eCpu_transpose_sum_2x37x23 -j 8
cmake --build build/autofuse/mlir/test/st -j 8
ctest --test-dir build/autofuse/mlir/test/st --output-on-failure -j 8
```

Broadcast/Sum 当前可能在生成阶段失败，应保留该退出码，不能跳过生成后运行旧二进制。
最终仍需完整 CPU E2E、全部 ST、UT/lit 回归；只报专项通过不能宣称全量通过。
910B3 用例须在匹配的 Python/动态库 overlay 下复验严格 MLIR 数值；a3、reg 真机及性能
未执行时单独列出。各问题必须在“复现 → 原因 → 修改 → 原图通过 → 通用/邻接回归”后关闭。

## 验收标准

先重现最新基线五个失败，再按“搬运/容量 → chain → 独立步距 → 广播 → Reduce 实际失败分支”推进。
这是调查优先级，不是未经证据就同时改五层。每项先保留失败用例，再修复其所有者并回归邻接能力。
具体输入矩阵、命令与数值要求见测试文档的“泛化缺口修复验证计划”。
不额外新增设备 RDV 框架；UT/gtest、lit、CPU E2E 覆盖通用规则，Python/910B3 验证可执行 mem
代表图；reg 真实设备无证据时明确记录未执行。验收必须同时满足容量无越界、精确数值、严格接管、
既有 UT/lit/ST 无新增回归和各失败原图通过；若证明原图无效，须附构图语义证据，不改图冒充同例通过。

## 语义命名与补充性能约束

编译期布局分析复用已有 descriptor，目标复杂度与轴数线性相关，不新增全图扫描/求解。
显式布局转换可能增加 UB、搬运和循环，须报告转换次数、峰值资源与生成代码大小，不能预设零开销。
候选可接管性和代价必须在选择前确定，不能胜出后才发现额外 scratch 不可分配。
性能使用 INFO 关闭的固定环境对比编译/tiling/kernel；目前未测性能，不作性能通过承诺。

仅参数/容量修正且最终实现分类未变时沿用原语义名，并给出依据。若新增 API 分支或成功布局转换，
遵循 Codegen/Schedule 语义命名规范，使用公共词表与 helper，在真实成功选择点采集；
保留 graph/result/group/key/subcase/template 完整身份，失败、复制、reuse 和回退时同步信息。
不得借父名或 `Adaptive` 掩盖缺失；名称行为测试随对应实现交付，不另建分类算法。

## 最新上游同步与复验证据（2026-11-02）

基线为 `origin/af-mlir` 的 `aaab745c`，两份功能提交重放后已推送，GitCode PR #95
曾返回 `mergeable=true`。以下为最新产物的独立验证结果，不将已知失败从分母中删掉：

| 检查 | 最新结果 |
|---|---|
| LLVM 包路径检查 | 通过 |
| CMake build gate | 开发容器中设置 `LLVM_BUILD_DIR=/opt/llvm`，8/8 通过 |
| 完整 UT | 14/14 个 CTest 项通过 |
| 完整 lit | 284/284 通过 |
| 完整 ST | 82 个 gtest 中 81 通过；GE→Where→Transpose 失败 |
| Cast→Transpose | 重建成功，CPU 数值失败；256B 对 128B，另已采样实际 DMA 参数 |
| Transpose chain | 重建成功，CPU 数值失败；首个错误 index=1，期望 -10.75 |
| Broadcast→Transpose | 生成失败，源/目的完整 descriptor 元素数无法证明相等 |
| Transpose→Sum | 生成失败；已确认 A-R-A 无法进入二维 ReducePlan |

日志在构建容器 `/tmp/transpose-rebase-*.log`；诊断采样为 `sum-diagnostic3.log`、
`broadcast-diagnostic3.log` 和 `cast-diagnostic2.log`（均带上述统一前缀）。采样只用于定位，
不替代未加调试器的实际构建/数值退出码。未重跑全部 CPU E2E、NPU 或性能，不能宣称全量通过。

## legacy Inductor 前置资格验证（优先于前文候选修复方案）

### 用例来源

| 组合 | 来源与性质 | 是否为既有 legacy Inductor 用例直接迁移 |
|---|---|---|
| Cast→Transpose | `tests/v35/st/optimize/optimize_st_v2.cpp::LoadCastTransposeCase`，UT 有同名场景；新增端到端 oracle | 否，原用例仅验证优化/调度 |
| GE→Where→Transpose | 上述文件 `LoadGEWhereTransposeCase`，UT 有同名场景；新增完整数值验证 | 否，原用例仅验证优化/调度 |
| Transpose chain | 人为构造的两次 permute 组合 `[2,3,4,5]→[2,5,4,3]` | 否，未找到同图现成用例 |
| Broadcast→Transpose | 人为构造 `[1,37,23]→[8,37,23]→[8,23,37]` | 否，未找到同图现成用例 |
| Transpose→Sum | 人为构造 perm021 后归约最后一维，输出 `[2,23,1]` | 否，未找到同图现成用例 |

它们不属于已验收的 7 项 legacy Inductor 迁移行为。组合在数学上可定义，不等于 legacy
backend 已支持，也不等于任意手工 AscGraph 都有效。

### 构图和 oracle 审计

逐节点核对原始 Data/Load/计算/Store/Output 的 axis、repeats 与 dense stride，
包括长度为 1 的轴的零 stride；五个组合全部符合预定视图。
chain 第二次 permute 的轴身份是第一次结果上的组合置换 `[0,3,2,1]`，不能误用原始 loop 位置。
GE 的临时验证图显式将 Output 的视图同步为 Store 的 `[41,55,54,38]`；这是合法视图的
校准版本，不把原 optimizer fixture 自动补出的 Output 元数据当作端到端数值契约。

五个独立 C++ 坐标 oracle 均与 NumPy 的 astype/transpose/broadcast_to/sum/where 直接运算
逐元素一致；GE 校验 4,627,260 个输出。四分之一整数输入可被当前 float/half 精确表示，
不通过放宽容差掩盖问题。此审计证明数学语义及本次检查的静态 metadata 一致，不覆盖所有后端约束。

### legacy 路线与资源配置

验证调用真实 `Codegen::GenerateForInductor`，设置 `--mlir_codegen_config=none`，检查产物
没有 `_subgraph0_plan`。使用真实 `TilingLib` 默认 ATT 入口，不使用 MLIR CPU 生成器的
返回 false 的 `CodegenTiling` 测试 stub。原有生成器中的 stub 只能用于原 MLIR 测试环境，
不能拿来证明 legacy 能运行。

原 CPU fixture 还显式设置 `using_att_calc_qbt_size=false`；本轮先保留该对照，再以生产默认
`true` 补跑全部五图，检查 kernel 确实使用 `t->q*_size` 等 ATT 资源字段。
各图平台、原始 shape、dtype、独立 oracle 和 kernel ABI 保持一致；产物、工作目录、coverage
全部隔离在 `/tmp`，未覆盖原 MLIR 生成文件、安装目录或 legacy 源码。

### 生产默认配置下的实际结果

| 组合 | 构图审计 | legacy Inductor 生成及编译 | legacy CPU 数值结果 | 是否进入 MLIR 修复 |
|---|---|---|---|---|
| Cast→Transpose | 通过 | 通过 | 失败：GM→UB 要求 256B，当前 128B | 否 |
| Transpose chain | 通过 | 通过 | 失败：index 56，实际 -8.75，期望 -2.25 | 否 |
| Broadcast→Transpose | 通过 | 通过 | 失败：Copy 要求 21376B，当前 18944B | 否 |
| Transpose→Sum | 通过 | 通过 | 失败：SDK ReduceSum 输出容量要求至少 888，当前 48，单位沿用 SDK 原始诊断 | 否 |
| GE→Where→Transpose | 通过（校准 Output 视图） | 通过 | 失败：GM→UB 要求 30688B，当前 30144B | 否 |

正向控制使用既有 Transpose→Add `[4,16,32]` / `[4,32,16]`，同一验证工具、同一 legacy
Inductor 入口、生产默认 ATT 资源配置下，生成、编译和 CPU 数值全部通过。关闭 ATT 容量计算
的对照也通过该形状；工具能够区分成功与失败，不把未执行项当作通过。

结论：本轮五个探索组合均未达到 legacy CPU 端到端门槛，没有已确认的“legacy 数值通过而
MLIR 数值失败”的新增修复项。前文描述的 MLIR 拒收点和生成差异仅保留为调查证据，不能据此
扩大交付范围或实施额外布局转换。CPU 模拟器的失败也不能直接推导为所有真实硬件均不支持；
如需推进硬件能力结论，必须在匹配平台上补同图 legacy 数值证据，再进入 MLIR 对照。

### 原始证据与复现边界

构建容器中的 `/tmp/transpose-legacy-inductor-default/` 保存四个基础组合及 GE 的
`generate.log`、编译日志、`numeric.log`、原始图 metadata、实际 kernel/tiling 和结果 JSON。
`graph-audit.json` 保存逐节点静态审计；宿主机 `/tmp/transpose-legacy-oracle-audit.json`
保存五项 NumPy oracle 校验。正向控制在容器
`/tmp/transpose-legacy-inductor-default-control/transpose_add_4x16x32/`。

本次临时验证脚本分别为 `/tmp/transpose-legacy-default.py`、
`/tmp/transpose-legacy-ge-default.py`、`/tmp/transpose-legacy-default-control.py`，
复用 CMake 已生成的编译/链接参数并逐文件串行编译；没有启动并行的同 build 构建。
脚本为工作区调查产物，不属于正式测试接口。结果只覆盖 CANN 9.2.0 当前 CPU 模拟环境，
未宣称这些精确手工图的 NPU、性能或其它平台已经验证通过。

## 目录归位与最终泛化验收口径（2026-11-02）

CPU 仿真目录 `test/e2e/cpu/cases/transpose_generalization/` 仅保留 Transpose→Add 两个稳定对照实例；Cast、级联、Broadcast 和 Sum 不再注册为 CPU E2E，因为它们的 CPU 模拟资源检查与真实 910B3 行为不一致，继续放在 CPU 通过矩阵会产生误导。

已在本机 910B3 通过真实 TorchInductor legacy/MLIR 双路线的四个组合统一放入 `test/e2e/npu/`：`cast_transpose.py`、`transpose_chain.py`、`broadcast_transpose.py`、`transpose_sum.py`，并在 `cases.py` 标记 PASS。四项均使用独立 cache、匹配 Python/codegen overlay、legacy 与严格 MLIR 两路线，数值通过。

GE→Where→Transpose 保留为 NPU 待验证补充场景，不列入 Inductor 迁移验收或 PASS 清单。其数学图合法，真实 TorchInductor lowering 已成功，但当前完整源码 overlay 仍存在 graph/AscIR/protobuf 动态库闭包 ABI 不一致；CANN 原生 pyautofuse 又与 Torch 2.10 API 不兼容。不得用这两类环境错误宣称 legacy 或 MLIR 结果。

## 新增代码检视问题：Pad 完整窗口 stride（2026-11-03）

复核 `Pad.cpp::analyzePad` 后确认 P1：原判据只比较向量化窗口的倒数第二维与尾轴，`emitPadBranches` 随后从零偏移发出单次 Pad/DataCopy，三维外层间隙会静默访问错误平面。已补充逐轴 dense stride 校验：源 descriptor 必须证明完整窗口连续，目的 descriptor 必须证明外层按统一 padded 行距推进；否则能力层和 pattern 均拒收。原 `afir-pad-ub-ub.mlir` 生成断言通过，增量 `AFIRToAscendC` 编译通过。后续仍应补充三维 gap 的明确负向 lit 或数值用例，防止该判据回退。
