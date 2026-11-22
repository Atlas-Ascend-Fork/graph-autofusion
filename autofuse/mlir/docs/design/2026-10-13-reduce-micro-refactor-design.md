# 简介

> **当前状态（2026-10-22）**：累加语义分析与完整 ReducePlan 已解耦；本文的原实施计划保留用于溯源。
> 后续已支持 Sum i32、Max/Min f16 和带保留轴的 f32 Mean 跨轮累加，见同目录 dtype 与 mean-correction 设计。
> scratch 已由 `afir.scratch_roles` 显式区分 `api_scratch` / `round_result`，不依赖操作数固定顺序推断角色。
> 循环生成后的 `afir-scope-reduce-scratch` 分析共享使用作用域及支配关系；物理 buffer ID、大小及生命周期不因此收窄。
> 轴绑定存活检查已于 2026-10-23 收进 MemoryAllocate 的结束校验，不再提供独立 pass；它不是 Reduce 专用检查。具体顺序见 [pipeline 设计](2026-08-05-afir-to-ascendc-pass-pipeline-design.md)。

## 目的

将 Reduce 的循环累加语义与二维 API/资源要求解耦，并为 AR 外维合并建立可验证的局部布局条件。面向实现者、代码检视者和测试开发者；本文提供可分提交执行的实现方案，不包含代码实现。

基线为 `codex/mlir-rebase-upstream` 的 `088ad987` **加当前工作区修改**：AR 单 A 门禁、共享 ReduceInit repeat/stride 修复和对应测试。实施前记录实际 HEAD 和 diff；不得覆盖这些改动。这是原方案提出时的基线；原计划不等于验收记录，当前状态见文首。

## 范围

包含两项独立工作：

- A：抽取归约累加轴分析，让 LoopGenerate 不依赖二维布局或 scratch 数量。
- B：复用 `provablyContiguous`，在 Reduce 能力查询与改写共享的入口建立 AR 外维布局证明；检查当前单 A 路径依赖的布局不变量，不放宽多 A。

不涉及七类 ReduceTraits、候选选择、ATT、调度器、TransferDim 表达、Mean 跨轮算法、dtype 扩展、RA 多 R、AR 三根及以上 R、emitter 命名治理、运行期分段/重排。不按文件长度拆 Reduce.cpp，不重写整个 ReducePlan。现有 helper 修复作为基线独立保留，不混入微重构提交。

# 总体概述

## 软件概述

### 项目介绍

Autofuse MLIR 将调度图转为 buffer-form AFIR，再生成循环、资源、同步和 AscendC 调用。当前 `buildReducePlan` 同时推导 accumulationAxes、二维 AR/RA 轴组及 scratch 角色。LoopGenerate 为重定向跨循环结果调用完整计划，资源缺失因而可能被错误归因为 InvalidSchedule。

当前 Reduce lowering 以物理外维 actual 的乘积作为 API 连续行数，但名义 stride 连续不足以证明内部尾块下有效行连续。现有 AR 单 A 门禁对此采取保守拒绝。

### 产品环境介绍

沿用现有 LLVM/MLIR、CANN Toolkit、AscendC 和 CPU/NPU 测试环境。不新增依赖，不新增 runtime 调用，不修改 Python 包或 run 包接口。验证环境必须使用与实施源码对应的工具，不能混用其他工作区已重编的 af-opt。

## 软件功能

1. 独立查询归约的循环累加轴及失败原因。
2. 循环生成只依据语义和逃逸路径决定 destination 重定向。
3. 能力查询和改写使用同一套无副作用布局判据。
4. 当前 AR 单 A 的源行距和目标紧凑布局得到显式检查；多 A 继续拒绝。
5. 已支持的单 A、RA 多 A、跨轮累加及候选失败处理保持原有行为。

## 设计约束

- 不使用仓内 skill；按仓库设计模板和红线执行。
- 构建并行度最多使用 `-j 8`。
- 不将 Reduce 并入 TransferDim：搬运的 GM/UB 双侧合并与归约的 UB 布局条件不同。
- 不新增 IR 属性、隐式缓存、全局状态或 pass；不改 pass 顺序。
- 能力查询只读，不允许为证明创建 SSA、改属性或发运行期 guard。
- 本次微重构不扩 dtype 或 Mean 算法；原基线的 f32/Mean 不跨轮限制已由后续专项解除，布局能力仍以 ReducePlan 的证明为准。
- 无法证明不能视为支持；不以符号字符串相同代替物理布局证明。
- 依赖方向：循环生成只依赖归约语义，不依赖后端资源与二维 API 可表达性。这是架构约束而非验收判据——不得通过缓存完整计划、或把资源检查挪进语义查询来「满足」它。

## 假设和依赖关系

| 前提 | 处理 |
|---|---|
| 当前调度器自然产物的 AR A 组只有一轴 | 作为待用调度矩阵回归核实的兼容性前提，不作为正确性证明 |
| descriptor 的 vectorized_strides 描述 UB 布局 | 沿用现有契约；B 必须读取实际 descriptor，不能从轴尺寸猜 stride |
| `provablyContiguous` 证明 outerStride=innerStride×innerExtent | 已核对源码；该函数本身不证明 actual=nominal |
| actual extent 由 TransferPlan 的符号替换物化 | 复用既有过程；本期不引入内部多 A 尾块证明，后续放宽时需核对符号替换 |
| legacy scratch 由 CalcReduceTmpSize 产生 | bridge 归一化为 scratch_roles；完整 ReducePlan 按角色解析，物理分配策略保持 |
| 当前未提交修复会保留 | 实施前确认基线，独立落提交或完整保留 diff |

# 需求分析与设计

## 整体介绍

A 保持现有文件布局，在 ReducePlan.h/.cpp 中增加最小语义查询。完整计划和 LoopGenerate 共用它，避免复制 provenance 推导。

B-proof 留在 conversion 的 Reduce.cpp 中，补充布局谓词的输入和逻辑，不新增布局框架。B-relax（多 A 放宽及相应数值夹具）推后，待真实工作负载或前端规则变化提出需求后另行设计。谓词从 op、plan 和 descriptor 得出静态可行性；capability 和 rewrite 都先调用它，materializer 只消费已验证的布局。

A、B 互不依赖，可独立提交、独立回滚。建议执行顺序为基线固定 → A → B；每一步单独验证。

## 功能需求

### 功能需求 1：独立归约累加语义查询

1. 介绍：LoopGenerate 能取得累加轴集合，不受二维高阶 API 可表达性或 scratch 个数影响。
2. 输入：buffer-form Reduce op、输入/目标 descriptor 的完整 axis/stride、schedule、用于 split/merge provenance 的 asc_graph。
3. 处理：复用 collectReducedAxes 和 isDerivedOnlyFromReducedAxes；按 schedule 顺序收集到 loop_axis 为止。保留混合 A/R merge 不当作纯归约轴的规则。检查 op 属于七类归约、destination 数量、轴表一致性以及 loop_axis 存在；拒绝必要语义信息缺失。不得读取 scratch 个数、执行 AR/RA 二维化检查，或要求 vectorized_axis 至少两轴。
4. 输出：有序 accumulationAxes；合法非累加返回成功且为空；错误返回 failure，并提供具体原因，不自行 emitError。

### 功能需求 2：循环生成与完整计划分工

1. 介绍：循环生成只验证跨循环存活语义；完整计划继续负责实现所需结构。
2. 输入：语义查询结果、生产者/消费者循环节点、destination operand partition。
3. 处理：LoopGenerate 保留祖先关系、非空累加轴、每条逃逸轴都被归约、唯一 destination、operand 重定向等检查。不能简化为只问 isAccumulating 的 bool。buildReducePlan 先调用语义查询，再执行现有 vectorized 布局和 scratch 检查；采用 move 写入计划，不长期保存两份轴列表。
4. 输出：原有重定向和副作用关系不变。scratch 不足不再导致 LoopGenerate 报 InvalidSchedule，后续完整计划/实现选择仍拒绝该 op；不静默放过资源错误。

### 功能需求 3：当前 AR 布局不变量检查与共用可行性入口

1. 介绍：把当前 AR 路径依赖的源行距与目标连续写入假设变成显式检查。本期保留单 A 支持边界。
2. 输入：op 的 operand/destination/result descriptors、plan.first/last、graph axis 信息；仅使用编译期事实。
3. 处理：单 A 也执行源行距、R 组物理布局与目标 stride 检查；退化轴按访问范围证明。多 A 继续明确拒绝。RA 路径保留现有边界与 resultSpan 行为。本期不顺带放开 RA 多 R 或 AR 多 R 上限。查询和改写调用同一个谓词，不在 materializer 内另写第二份条件。
4. 输出：可证明安全时支持；结构缺失或无法证明时拒绝。失败原因描述输入跨度、内部尾块风险或目标 stride 不匹配，不改变候选淘汰机制。

## 非功能需求

### 可维护性

只抽一次语义分析；七类算子继续使用原 traits。布局检查在 Reduce.cpp 局部实现，复用 SymbolicBuilder 的表达式乘法/判等/连续性谓词。保持 API/资源检查与语义查询的依赖方向，不因“纯 AFIR 类型”就把后端条件再次塞回语义查询。

### 可测试性

语义查询可直接 UT；布局行为由正反向 conversion lit 和数值测试验证。不以“源文件出现某函数调用”作为验收。

### 可移植性

不新增芯片或框架名称分支。沿用 target 能力接口。固定形状反例与平台无关；设备执行沿用项目支持的平台。

### 可靠性

未知证明结果保守拒绝。查询无 IR 写入，失败不产生部分资源或 SSA；通过完整计划但修改前状态变化时，rewrite 的同源检查仍有效。保持累加器 memory effects 和循环支配关系。

### 特性交叉影响

| 场景 | 适用性 | 结论 |
|---|---|---|
| SuperKernel Python 接口 | 不适用 | 不改包、Python 选项或 wheel |
| SuperKernel C++/AOT 接口 | 不适用 | 不改符号、结构体和 AOT |
| Autofuse 图优化 | 适用 | LoopGenerate 改查询依赖；保持数据/控制边和 destination 重定向 |
| Autofuse Codegen/Backend | 适用 | Reduce 合法性判据变化，需核对生成代码及候选兼容性 |
| AscendC API / Runtime 交互 | 适用 | 既有调用形态不变；当前单 A 路径需验证正常输入不被误拒绝 |
| Python/C++ 混合绑定 | 不适用 | 无对外绑定改动；E2E 验证仍经原入口 |
| 构建与打包 | 不适用 | 不新增库/依赖/安装项，复用既有测试目标 |
| 测试与覆盖率 | 适用 | UT、lit；既有 NPU reduce 矩阵用于数值集成确认 |
| 性能与日志 | 适用 | 新增编译期符号证明，不新增设备算术、日志或缓存 |
| 兼容性 | 适用 | 内部 C++ 查询新增；既有候选不得因重构被误拒绝 |

## 性能

### 编译时长

A 复用原有归约轴分析，不增加全图遍历；LoopGenerate 不再执行布局/scratch 检查。B 对当前 AR 轴组遍历并做符号证明，复杂度为 O(k) 次证明，表达式求解成本单独计入。能力查询和 rewrite 各执行一次，不引入缓存。通过现有代表性图多次 codegen 对比记录耗时；发现明显回退时检查是否重复建图或重复证明。

### 执行性能

不新增 runtime guard、转置、拷贝或分段循环。既有支持路径的 srcShape、padding、累加和屏障保持不变，预期运行期零额外开销；以生成代码对比确认，不能只凭预期验收。

### 内存和产物大小

新增短生命周期轴列表；不持有 Operation 所有权、不保存跨 pass 状态。无常驻内存、包布局或 TilingData 变化。本期不放宽支持面，不额外申请缓冲。

## 接口设计

### 新增/修改接口描述

在既有 ReducePlan.h 中增加内部接口，命名在实施时遵循所在文件习惯：

```cpp
struct ReduceAccumulationInfo {
  llvm::SmallVector<int64_t, 2> accumulationAxes;
  bool isAccumulating() const { return !accumulationAxes.empty(); }
};
FailureOr<ReduceAccumulationInfo>
analyzeReduceAccumulation(Operation *op, std::string *diagnostic = nullptr);
```

不新增外部 C/Python API。`buildReducePlan` 签名和 ReducePlan 现有字段保留，内部调用新接口，避免一次性迁移所有测试与 lowering 访问方式。

Reduce.cpp 的局部布局入口调整为接收 `Operation *op` 与 `const ReducePlan &plan`，可选诊断文本；以同一个只读实现供 queryLowering 和 matchAndRewrite 调用。沿用现有 CapabilityReason 映射，不为本次引入公共错误码框架。

### 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|---|---|---|---|
| 接口说明 | 是否需要接口评审 | 是 | 内部语义查询的输入、合法空集与 failure 含义 |
| 接口说明 | 是否需要补充文档 | 是 | 更新 ReducePlan 与 LoopGenerate 注释 |
| 接口兼容 | 行为是否兼容 | 是 | 合法既有图不变；错误归因改善，不符合物理布局的输入被显式拒绝 |
| 接口兼容 | ABI/API 是否兼容 | 否（公开接口） | 不更改已交付 API、IR schema 或 Python 绑定 |
| 接口约束 | 约束不满足时是否清晰报错 | 是 | 区分语义缺失与实现不可表达 |
| 接口测试 | 是否需要独立接口用例 | 是 | 同语义、不同 scratch/布局应得到同一累加分析 |

## 软件设计

### 关键数据结构

仅新增 ReduceAccumulationInfo，存有序轴 ID，无资源所有权。现有 ReducePlan 继续作为完整计划；PhysicalReduceShape 继续保存物化后的 SSA。B 不新增公共布局类型，不将 SSA 存入分析对象。

### 关键技术/算法

**A：语义抽取。** 从 buildReducePlan 提取完整 axis/stride 上的 reduced-axis 判断及 schedule/provenance 遍历。新接口验证是归约 op；合法空 accumulationAxes 表示不跨轮。vectorized A/R、pattern、tailLast、scratchCount 及下标仍由完整计划计算。LoopGenerate 的逃逸检查逐轴使用新结果。

**B-proof：检查实际使用的单 A 路径。** 能力查询与改写共用只读谓词，不改变物化公式。

1. 读取 input/destination 的 vectorized strides、轴 extent 与 graph 引用；结构缺失时给出明确诊断。保留 AR R 组最多两轴、RA R 组单轴等既有边界。
2. 对当前 AR 单 A 也检查：源 A stride 必须与物化算法的 R footprint 一致；目标 A stride 必须匹配连续结果写入。不得以单 A 为由绕过检查。
3. R footprint 使用 `alignUp(R_last_nominal, alignNum) × Π R_other_nominal`，其中 `alignNum = kBlockBytes / elementBytes`，与当前 lowering 一致，不写死 8。还需检查 R 最内轴单位 stride，以及两根 R 时外 R stride 与对齐后的最内 extent 一致；仅检查 A stride 等于总 footprint 不能排除 R 组内部空洞。复用现有表达式构造/判等；在相邻 stride 乘法关系适用处使用 `provablyContiguous`，对齐表达式由既有 helper 表达。
4. destination 的 R stride 为 0，非退化 A stride 为 1；沿用既有 result/destination 一致性校验。源/目标条件分别给出诊断。
5. 对静态证明 extent 为 1 的轴，stride 不影响地址序列，允许按此消去无效 stride 约束，覆盖全归约退化 A 的零 stride；该规则依赖访问范围证明，不按算子名称开特例。动态 extent 不能凭某次运行值为 1 放行。此处仍须验证其余实际访问维度。
6. 多 A 本期继续拒绝，诊断明确为“当前 AR 多 A 合并未受支持/未建立完整证明”。不能为了把拒绝理由改成布局失败而假称某条可成立的等式不成立。轴数限制是当前实现边界，不是非连续性的证明。
7. 正常真实调度产物应保持通过。若新增检查拒绝真实输入，追查 descriptor、布局生成或当前物化公式的实际矛盾，不直接删除检查，也不预先宣称捕获的必然是后端错值。

**B-relax：推后。** 触发条件是真实工作负载产生 AR 多 A，或前端/调度规则改变。届时再设计内部尾块与多 A 连续性证明，并估算从零建立 CPU reduce E2E 的成本。`axisCanHaveTail` 的 AxisType 判据有既有语义依据：Original/Merged/切分 outer 的 tailSize 为 0；额外应核对 extent 中符号是否被 actual-size 绑定替换。此前只允许 Original 常量、拒绝 Merged 的条件是保守限制，不是无尾块的必要条件，本期删除该放宽算法。

保留内部尾块反例 `[2,3,8] → [2,1,8]`：nominal stride 等式成立仍不能证明有效行连续。本期该输入继续被多 A 支持边界拒绝，不声称已实现其精细证明。

### 流程设计

| 阶段 | 工作 | 完成判据 |
|---|---|---|
| 0 | 固定含当前修复的基线，记录 HEAD/diff/工具来源 | 不覆盖他人修改；基线 UT/lit 结果可追溯 |
| A1 | 先添加语义独立性和诊断归因测试 | 原实现可复现资源错误污染循环分析 |
| A2 | 抽取查询、切换 LoopGenerate、完整计划复用 | 语义与资源/二维布局解耦；原生成路径不变 |
| B1 | 添加单 A 正向、源行距/R 内部空洞/目标 stride 反例及退化轴用例 | 原实现缺少的检查可复现 |
| B2 | 实现共享证明、接入查询/改写 | 两端结论一致，合法退化轴通过 |
| B3 | 真实调度矩阵回归及生成代码对比 | 支持面不放宽、单 A 不误拒绝、RA 不退化 |

不变更 pass 顺序。A 的语义查询在 LoopGenerate 中不需要运行期 axis_binding；B 在既有实现选择阶段只读图/descriptor，不要求重新物化循环。失败沿用候选淘汰机制。A、B 各自形成独立提交；任一回滚不要求回滚基线 helper 修复。

### 对子模块的修改

| 文件/模块 | 必需修改 |
|---|---|
| dialect/AFIR/Analysis/ReducePlan.h/.cpp | 增加语义返回面，完整计划复用；更新误导性的“完全后端无关”注释 |
| dialect/AFIR/Transforms/LoopGenerate.cpp | 改用语义查询，保留全部逃逸轴检查及 destination 重定向 |
| conversion/AFIRToAscendC/Implementations/Reduce.cpp | 共用只读布局证明；复用 SymbolicBuilder，不拆 traits/物化/发射编排 |
| test/ut/dialect/ReducePlanTest.cpp | 加语义独立性、非法语义与完整计划拒绝对照 |
| test/lit/demo、test/lit/conversion | 加错误阶段、连续/不连续/内部尾块/目标布局案例；保留已有多 A 拒绝测试 |
| 既有 NPU reduce 矩阵 | 验证当前支持路径的数值和候选兼容性；本期不建立新的 CPU reduce E2E 家族 |
| docs/design 本文及相关源码注释 | 记录真实支持子集与测试结果，不复制历史调测叙事 |

无需修改 TransferPlan、SymbolicBuilder、调度器和 ATT 的接口；若实施证明必须扩大这些范围，应先单独说明具体必要性，不能边实施边叠加通用框架。

### 错误处理

#### 系统错误

沿用 MLIR 分析失败与 pass failure 机制。不新增文件/设备资源，不新增异常分支释放责任；测试产物置于独立 build 或临时目录。

#### 接口错误

语义错误由 LoopGenerate 报具体 InvalidSchedule 原因；资源与二维实现限制在完整计划/能力阶段拒绝。A 不保证所有资源错误都拥有新增专用枚举，只保证不再被循环语义查询误报。B 不能证明与输入标注缺失分别给出诊断文本，复用现有能力分类及候选回退。

## 安全检查

已逐项核对编码红线：不新增敏感配置、芯片/框架分支、runtime 私有接口、静态跨 SO 析构或 Python 引用处理；不改公开 ABI。索引前检查轴/stride 数组和 ID 映射。常量 extent 要验证正值与可表示范围，符号计算使用既有表达式工具，保留运行期乘法溢出检查。按 schedule/descriptor 顺序遍历，不能依赖哈希遍历顺序生成 IR。保留 memory effects、同步和支配关系，不新增生命周期假设。共享 helper 的编码范围修复独立保留。

## 兼容性检查

IR 文本格式、脚本选项、环境变量、安装目录、TilingData、kernel ABI 和 Python 包不变。A 改善失败阶段和诊断文本；诊断测试需同步更新。B 不放宽多 A，只检查现有单 A 路径的隐含契约。既有真实调度矩阵的候选集合和生成调用应保持，若出现删减必须定位，不可用“最后还有一个候选成功”掩盖回退。

## 测试设计

### 测试边界

UT 验证纯分析；lit 从合法 buffer/已物化 IR 边界验证 pass 和能力结论；数值验证使用既有 NPU reduce 矩阵。当前 CPU cases 下未发现 reduce 用例，建立 CPU reduce E2E 属于新测试基础设施，不能按追加一格估算；随 B-relax 另行评估。人工 AFIR 必须自洽并通过 verifier，明确 nominal 与 actual 来源。不能把主机 Duplicate 地址模型等同于设备执行。NPU/RDV 用于真实调度集成确认，无法运行时明确记录缺口。

### 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|---|---|---|---|
| 功能 A | 直接 R、纯 R merge、混合 A/R merge、非累加 | 新旧语义结果对照，轴顺序稳定 | gtest UT |
| 功能 A | 同 descriptor/schedule，scratch 为 0/1/2 | 新查询结果相同；完整计划按原规则拒绝不足资源 | gtest UT |
| 功能 A | 语义合法但 A/R 多次交替 | 语义查询成功、完整二维计划拒绝 | gtest UT |
| 异常 A | 非 reduce、缺 descriptor、未知 loop_axis、跨非 R 轴逃逸 | 正确错误来源；不扩大可逃逸范围 | UT/lit ST |
| 异常 A | 跨循环 reduce scratch 缺失 | loop-generate 不因 scratch 失败，完整流水线仍拒绝 | lit ST |
| 功能 B | 单 A 的正常布局、动态 extent/对齐表达式 | 能力与下降一致通过；生成参数不变 | lit ST |
| 异常 B | 单 A source 行距错误、R 内部空洞、destination 跨步 | 分别拒绝，诊断指出实际不一致条件 | lit ST |
| 边界 B | 全归约 extent=1、零 stride；动态 extent 不可证明为 1 | 合法退化通过，未知不冒充退化 | UT/lit ST |
| 兼容性 B | 多 A 内部尾块反例 | 继续按当前支持边界拒绝，不放宽 | lit ST |
| 兼容性 | 单 A、全归约退化轴、RA 多 A resultSpan | 原结果及调用参数保持 | UT/lit/NPU |
| 特性交叉 | RCommon/RCore/AllLoad；reduce 前后 elementwise | 候选/tiling/同步和数值不变 | NPU ST或RDV |
| 性能 | 代表图 codegen 与生成内核 | 无新增运行期分支/拷贝，记录编译时长 | 手工验证 |

测试不要求新增 pytest：本次无 Python 行为改动，复用现有 Python NPU runner 做集成即可。不新增逐条镜像实现的测试，以输入语义不变性、布局反例和输出数值为判据。

### 测试命令

标准环境全链 UT/ST 入口（构建限制并行度）：

```bash
AF_MLIR_DEV_IMAGE=autofuse-mlir-dev:arm64 \
AF_CANN_VOLUME=af-cann-910 \
AF_CANN_INSTALL_PREFIX=/usr/local/Ascend/cann-9.2.0 \
bash autofuse/mlir/scripts/enter_dev_container.sh -- \
  bash scripts/test/run_autofuse_test.sh -u -s -m mlir -j 8
```

在已配置且源码/构建一致的开发容器内，定向迭代：

```bash
cmake --build build --target AFIRDialectTest af-opt ascir-translate -j 8
build/autofuse/mlir/test/ut/dialect/AFIRDialectTest --gtest_filter='ReducePlanTest.*'
/opt/llvm/bin/llvm-lit -j 8 -v --filter='reduce' build/autofuse/mlir/test/lit
```

以上容器参数采用本机配置，不是通用默认值。不要在既有 build/ 中直接重跑 cmake；缺失测试配置时会撤掉 MLIR 测试树，旧子目录和 make help 仍可能残留目标。执行前检查 build/Makefile、CMakeFiles/Makefile2 中的实际依赖及测试发现结果，执行后核对用例数量与日志，零用例不算通过。需要重新配置时使用独立构建目录并显式启用测试。check-e2e-cpu 即使存在，也不能证明 reduce 被测到；本期不把该目标作为 reduce 数值覆盖。NPU 沿用现有 reduce_template_matrix.py/reduce_fuzz_matrix.py 及 runner，在匹配版本环境执行，记录选中模板与形状。

## 验收标准

1. 相同归约语义在 scratch 为 0/1/2 或二维布局不可表达时，累加轴分析结果保持一致；非法语义仍失败。
2. 每条逃逸轴检查、唯一 destination 和副作用关系仍成立。
3. 资源不足在后续阶段拒绝，不再错误归因成循环调度语义；完整流水线不会因此接受缺 scratch 的 op。
4. 对相同输入，能力查询与改写给出一致可行性结论；查询无 IR 修改，无新增运行期 guard。
5. 当前单 A 的合法源行距及目标布局通过；源行距、R 内部 stride 或目标 stride 不匹配分别拒绝；合法退化轴通过。
6. 多 A 支持面不放宽，内部尾块反例继续拒绝。不以出现某函数调用或删除门禁作为正确性验收。
7. 单 A、RA 多 A、全归约及真实模板矩阵不退化；既有生成路径不新增设备算术/拷贝/屏障。
8. A、B-proof 独立提交；helper 修复与无关内容不混入。记录工具版本、实际执行的非零用例数量及测试缺口。NPU 无法执行时明确未完成数值验证。

### 设计文档检查结果

- [x] 跨特性交叉影响：已逐项覆盖十类场景，明确 LoopGenerate、codegen 与 AscendC 验证范围。
- [x] 编码红线：已检查资源、ABI/API、图改写确定性、索引/溢出与运行时生命周期，不新增违反项。
- [x] 测试设计：已定义 gtest UT、lit ST 与 NPU/RDV 集成策略，并明确 CPU reduce E2E 基础设施缺口；不把源码调用检查当验收。
- [x] 性能评估：只增加编译期静态证明，要求验证既有生成路径无新增运行期开销。
