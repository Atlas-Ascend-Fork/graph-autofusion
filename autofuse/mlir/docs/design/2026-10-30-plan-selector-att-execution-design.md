# 简介

## 目的

整理 MLIR bridge 的 PlanSelector，使模板选择、ATT 求解、资源预算与代码生成之间具有明确契约。本文供 bridge、ATT、AFIR 和测试维护者评审与实施。

状态：**历史设计，静态 winner 冻结方案未接入当前生产流程，相关实现已删除**。本文下方的静态执行、冻结结果、工具链及验收描述保留为历史方案，不代表当前功能或测试覆盖。

当前实现（2026-11-09）：

- `PreparePlanGeneration` 枚举全部候选并返回 `TilingCodeBundle`；静态和动态 shape 均由生成的 host tiling 程序选择 plan/template 和求解 tile size。
- 已删除独立静态 solver 的结果解析器、冻结快照及 emitter、`afplan.static_selection` 和专属测试；请求中不再保留无行为语义的 `mode` / `shape` 参数。
- 当前 MLIR emitter 的 `AutofuseIsStaticShape()` 返回 `false`，不启用 Python const tiling bake。本次清理不改变该行为，也不删除 Python 侧的通用 bake 能力。
- 保留候选枚举、资源预算、字段契约、运行时 tiling 源码生成和原始模板身份测试；这些测试不证明提前选出静态 winner。
- 本次仅调整 MLIR 内部 C++ 接口及 IR 属性；不改变 SuperKernel、公开 Python 接口、安装打包、runtime 调用或生产 tiling ABI。不新增运行时求解、日志或数据拷贝。历史手工带 `afplan.static_selection` 的 IR 不再受支持。


## 范围

包含候选枚举、静态/动态编排、ATT host 执行边界、统一目标资源、模板身份与 tiling 交付、错误分类及回归。

不修改 ATT 评分或求解算法，不恢复 capability 前置过滤，不新增模板家族，不调整算子 lowering、数值容差、SuperKernel 接口或 NPU 运行工具。LLD 慢链接单独排查，不在此设计内解决。

# 总体概述

## 软件概述

### 项目介绍

Schedule 生成候选，ATT 决定执行方案，MLIR 实现该方案。PlanSelector 的职责是决定进入代码生成的范围，并保证它与 ATT 的模板身份和 tiling 参数一致。

实施前（`196052cf`）的混杂点：

| 位置 | 现状 | 整理目标 |
|---|---|---|
| `PlanSelector.cpp` | 静态入口枚举后，solver 再枚举；动态只是枚举 | 统一枚举一次，显式区分编译模式 |
| `AttStaticTiling.cpp` | 查询平台、生成 ATT、编译执行、解码混在一起 | ATT 准备与执行分离，执行器不制定选择策略 |
| `CodegenOrchestration.cpp` | 静态内部准备 ATT，动态外部准备；完整 schedule 仍叫 `narrowed_result` | 统一准备流程，保留原始 schedule 身份 |
| `TilingCodeBundle` | 源码和可选静态结果并存 | 区分 ATT 源码产物与最终静态/动态交付 |
| 资源来源 | ATT 读取 PlatformContext，AFIR 读取 TargetProfile | 入口确定一次，后续显式传递 |

历史简化 `AttCostEvaluator` 已从当前分支删除，不作为可选回退路径重新引入。

### 产品环境介绍

静态求解依赖 CANN host tiling 代码、平台信息及可用的 host C++ 编译器。当前通过生成源码、编译临时可执行文件、执行并读取结果实现。动态产物携带运行时 ATT tiling 入口。

内部 C++ 编排接口可以调整；Python 调用接口、公开 kernel/tiling ABI、安装目录保持既有契约。跨机器构建必须区分 host 工具链与 target 资源，不能从 host CPU 架构推断 NPU 预算。

## 软件功能

- 枚举完整候选并验证结构。
- 静态场景执行真实 ATT，只生成胜出方案的全部阶段。
- 动态场景生成全部候选，运行时由 ATT 选择。
- 原子交付选择身份、参数和资源上下文。
- 明确报告结构错误、ATT 无解、工具链失败及 lowering 能力不足。

## 设计约束

1. 不重新评分、不自行选择次优模板、不在失败时改为全部候选。
2. 不裁剪后重新编号原始 schedule；schema、字段和 dispatch 使用可核对的身份映射。
3. 不通过前置算子/广播白名单代替 lowering capability。
4. 沿用既有模板语义词表、DFX 选择机制和错误处理策略，不建立第二套分类或钉 plan 开关。
5. 生成及执行 ATT 不依赖网络；临时资源异常退出可清理；构建并行度使用 `-j 8`。
6. 不借本次整理新增缓存、JIT 框架或性能评分体系。

## 假设和依赖关系

- 当前静态入口指 schedule 中 shape 已具体化；`ConcreteShape` 为空并不表示缺少输入。带符号却被标记为静态必须失败，不能默认填零。
- ATT 返回的 plan key、模板 key 与原始身份可唯一对应；模板显式 key 不一定等于数组下标。
- ATT 的真实调用必须保持现有 DFX/场景语义。PGO、TopN 或其他可能改变胜者的配置若不能与静态冻结兼容，应明确拒绝静态冻结，不能静默忽略。
- 已核实当前 MLIR host tiling ABI 与 `afir.target` verifier 使用 256 字节预留；本次将该既有约定收拢到 `resource_budget.h`，不将其推广到其他后端。ATT 当前接入使用源码生成接口，本次保留执行临时 host 程序的方式。
- 当前 WIP 曾遇到 `245760` 与 `196608` 的 UB 口径差异。该记录是统一预算的验证输入，不是选用任一值的依据。

# 需求分析与设计

## 整体介绍

```text
完整 schedule + 模式 + 场景/ATT 配置 + 统一目标资源
                         |
                 结构校验、枚举一次
                         |
           构建完整 schema、生成一次 ATT 程序
                         |
            +------------+------------+
            |                         |
           静态                      动态
            |                         |
     执行 ATT、验证结果         保留全部候选及 ATT 程序
            |                         |
     胜出身份 + 具体参数           运行时 ATT 求解
            +------------+------------+
                         |
            按原始身份构建 AFIR / 生成产物
```

“收窄”仅指代码生成范围，不修改 schedule 容器索引。动态运行时求解与静态编译期求解共用 ATT 契约，但调用时机不同。

## 功能需求

### 功能需求 1：候选枚举与身份

1. 介绍：只检查结构与坐标，不判定资源可解性或实现能力。
2. 输入：完整 `FusedScheduledResult`。
3. 处理：按输入顺序枚举 subgraph、plan、stage、template；拒绝空结构、非法索引、重复/歧义 key。只枚举一次，后续复用。删除枚举接口永远为 false 的 `unsupported` 输出。
4. 输出：沿用 `PlanSelection`；保留原始坐标及现有 key 映射。一个 plan 是完整阶段组合，RCore phase 1/2 不独立挑选。非连续显式 key 用映射解析，不按数组下标猜测。

#### ATT 与 codegen 的编号契约

编号必须区分坐标与分派 key，统一的是双方对同一执行方案的解释，不要求不同层次使用同一个整数。

| 标识 | 作用域与含义 | 契约 |
|---|---|---|
| subgraph_id | 完整 schedule 中的子图坐标 | 贯穿 ATT、AFIR 和产物，不因静态筛选改号 |
| plan_id | 子图内原始 ScheduledResult 坐标 | graph 级 ATT selector 必须唯一对应该 plan；禁止用筛选后位置替换 |
| stage_id | plan 内原始 schedule group 坐标 | 保持阶段顺序和依赖，RCore 两阶段分别携带身份 |
| template_id | stage 内原始 impl_graph 坐标 | 用于定位待 lowering 图，不等同于 tiling_key |
| tiling_key | ATT 对该 stage 返回的模板分派值 | 在完整 (subgraph, plan, stage) 作用域内唯一映射到 template_id |

已核实 ATT `CheckKeyValid` 要求同一 stage 全部显式 key 或全部默认 key；因此混合设置必须拒绝，不能自行补号。默认 key 在该前提下等于原始 template 下标；显式 key 必须唯一。bridge 枚举、结果解码及 AFIR 通过公共 `template_keys.h` 校验和解析，保持 ATT 现有编码不变。映射不存在或有歧义时报告契约错误，不能取第一个匹配项。

例如原始 subgraph 0 / plan 2：stage 0 的 template 3 对应 key 17，stage 1 的 template 1 对应 key 9。ATT 返回该组合后，静态仅生成这两个阶段模板；它们仍保留 plan 2 和原始 template 坐标，不能因为只剩一个 plan 就改成 plan 0。动态则保留全部组合的既有分派实现，运行时的 key 17/9 仍映射到相同模板。单层 flat schedule 即使省略 graph selector，也不能改变模板 key 的含义。

schema 字段路径、workspace 归属、AFIR 属性、dispatch 与诊断都使用该身份；物理 reuse 不替代逻辑身份。最终胜者、各 stage key 与 tiling 参数作为同一次求解的结果整体交付。此契约不修改 ATT 对外 key 编码，只消除 bridge/codegen 的独立推断。

### 功能需求 2：静态选择与冻结

1. 介绍：使用 ATT 的真实求解结果，冻结本次具体调用的执行方案。
2. 输入：完整候选、schema、ATT 程序、具体 shape、资源预算与场景配置。
3. 处理：调用执行器；解析胜出身份与全部必需字段；校验每个 subgraph 恰有一个胜出 plan，每个 stage 的模板可唯一定位。多阶段及 workspace 信息必须来自同一次求解。只 lowering ATT 选中的完整方案：每个 subgraph 一个胜出 plan，每个 stage 一个选中模板；该 plan 的全部必要阶段必须生成。单 subgraph、单 stage 时才是只生成一个模板；RCore 必须同时生成胜出 plan 的 phase 1 和 phase 2。未选候选不进入 AFIR lowering，但求解前 ATT 仍接收完整候选集。不能再次求解来补参数。
4. 输出：不可分离的静态结果，包含 selections、具体 tiling 与求解资源上下文。失败不返回部分可用结果。

**静态 ATT 未选出完整方案时，PlanSelector 返回失败并停止当前 MLIR 路径的后续 lowering。** 不生成全部候选、不改选次优、不使用近似评分。是否向用户报错或整图回退 legacy，由外层 MLIR 环境变量对应的模式策略统一控制，PlanSelector 不将此类失败自行定义为不可回退的硬错误。ATT 已选出方案、随后 lowering 能力不足也交由外层处理，不擅自改选。

失败来源与传播职责：

- ATT 判断当前输入、资源与配置约束下能否求得方案，返回失败并提供已有诊断。DFX 钉选下无可行结果只代表本次约束下失败，不能据此声称全部候选均无解。
- 执行器保留 ATT 的失败结果与诊断；bridge 保留失败类别并向外层传播，由 MLIR 模式策略决定报错或 legacy fallback；不能仅记录日志后继续，也不能把求解失败误标为 lowering 能力不足。
- 无有效返回、工具链失败、非法 key 或不完整阶段组合不一定由 ATT 的求解算法主动报错。执行器和 bridge 分别检测这些错误，同样终止当前 MLIR 编译路径并向外层报告，但必须保留实际原因，不能统一报成“ATT 无解”。

当前 WIP 的生成调用为 `if (!optiling::GetTiling(tiling, -1, nullptr)) return 3;`：失败判定来自 ATT 返回值，进程退出码由包装程序转换。这个布尔结果本身不足以区分全部内部原因。实施时需保留 ATT 诊断，并让执行器区分该求解失败与编译失败、异常退出；本文要求的错误传播不能视为当前实现已经完成。


完整 schema 的求解结果映射到最终静态 ABI 时，按既有字段身份和类型处理，不能复制完整结构体内存到另一种布局。校验字段宽度、类型、作用域、缺失/重复及越界；仅能剔除明确属于未生成候选的字段。保留 block_dim、workspace 大小/偏移及所有运行必需字段。

静态固定参数只对相同输入与资源契约有效。产物复用必须包含 shape、目标资源及相关配置身份；资源不兼容时明确拒绝复用，不能调用运行时 solver 选出一个未生成的模板。

### 功能需求 3：动态保留全部模板

1. 介绍：编译期不求胜者，运行时由 ATT 按实际输入选择。
2. 输入：完整候选、符号化 schema、同一目标资源契约及 ATT 程序。
3. 处理：生成全部候选的 AFIR 和 dispatch；验证运行时每个可返回 key 都有对应代码。任一候选 lowering 失败时，按现有整体严格模式/fallback 策略处理，不静默删除该候选。
4. 输出：全部 selections 与运行时 tiling 程序；不存在伪静态参数包。

### 功能需求 4：ATT 执行边界

1. 介绍：执行器执行准备好的程序，不自行枚举候选、生成 schema 或查询平台。
2. 输入：ATT 程序、调用参数、资源预算、明确的 host 工具链配置。
3. 处理：优先复用已有执行能力；若无适用接口，保留当前临时源码编译执行机制，将文件、子进程、诊断、清理封装在执行层。直接传 argv，不拼 shell 命令；保留失败阶段和退出码。结果编解码在 ATT adapter 边界验证，选择策略不处理文件协议。
4. 输出：已验证的求解数据，或明确的生成/编译/执行/结果错误。超时和取消必须结束本次子进程并回收资源，不终止其他任务。

### 功能需求 5：统一资源预算

1. 介绍：求解和 lowering 使用一致的目标资源事实，不允许两处独立查询后各自解释。
2. 输入：入口通过现有平台能力接口取得的物理资源、执行场景预留规则及资源来源。
3. 处理：形成单一预算，并传给 ATT、AFIR target 和内存校验。已有 TargetProfile/平台对象优先复用；需要补充字段时仅补足缺失语义。
4. 输出：物理 UB 容量、预留 UB、可用 UB、可用向量核数及兼容性身份。

预算契约：`available_ub = physical_ub - reserved_ub`，减法前校验值域。预留项只扣一次，tiling 自身、运行时保留等项必须写明归属。ATT 接口若要求可用容量则传 available；AFIR 若存物理容量则同时由统一规则导出相同可用预算，不能混用同名数值。

核数区分“可用核数”和“ATT 实际选择的 block_dim”，后者由求解结果确定且不能超过可用范围。GM workspace 不从 UB 扣减；workspace 配置来自胜出方案，与启动及分配接口交叉验证。

## 非功能需求

### 可维护性

PlanSelection 保留坐标与结构校验；PlanSelector 表达静态/动态策略；AttSchemaAdapter 生成程序、映射字段和 key；ATT 执行器处理工具链；CodegenOrchestration 负责串联和对外失败策略。不建立第二份轴、模板或资源分类表。

### 可测试性

策略 UT 注入 ATT 结果或执行桩，执行器测试覆盖真实进程成功/失败；真实 ATT 集成测试验证两者连接，不能只用伪结果自证。复用现有 PlanSelectorPolicyTest、AttSchemaAdapterTest、bridge UT 和 lit。

### 可移植性

支持项目既有 host 工具链；编译器、CANN 路径从配置边界传入。保留 CPU 仿真和 NPU 的各自能力来源，不硬编码芯片或框架分支。未配置工具链的静态执行应明确失败，不能启用近似评分。

### 可靠性

结果校验成功后整体提交；失败时不遗留旧 selections 或 tiling。输入 schedule 只读，临时目录按调用隔离。执行器保留现有 120 秒进程超时上限，分别标识编译与执行失败。工具链错误不得伪装成能力不足；禁止全局可变“最后一次求解结果”。

### 特性交叉影响

| 场景 | 适用性 | 结论 |
|---|---|---|
| SuperKernel Python | 不适用 | 不改变其接口与 wheel |
| SuperKernel C++/AOT | 不适用 | 不修改 libascendsk 或 AOT 路径 |
| Autofuse 图优化 | 适用 | 不改图；完整保留 Schedule 身份、组级角色与复制语义 |
| Autofuse Codegen/Backend | 适用 | 静态生成范围、动态 dispatch、tiling 与 workspace 为核心验证面 |
| AscendC/Runtime | 适用 | 不新增设备 API；验证启动核数、参数及 workspace 生命周期 |
| Python/C++ 绑定 | 适用 | 不改公开签名；验证编译失败传播和真实模块导入 |
| 构建与打包 | 适用 | 静态执行依赖 host 编译器与 CANN；验证安装态头文件/库可用，不新增联网依赖 |
| 测试与覆盖率 | 适用 | UT、lit、CPU 与 NPU 分层验证 |
| 性能与日志 | 适用 | 单独测量 ATT 生成、编译、执行和总编译耗时 |
| 兼容性 | 适用 | 保留 DFX、公开 ABI、字段及模板身份契约；旧缓存不可跨不兼容预算复用 |

## 性能

### 编译时长

一次编译请求最多枚举一次、生成一次 ATT 程序；静态执行一次求解，动态编译期不执行求解。复用现有产物，不为日志再次求解。分别记录 ATT 生成、host 编译、执行、lowering 和总耗时。原 WIP 的编译耗时作为基线，沿用已有性能 gate；没有既有阈值的项先建立基线，不宣称无回退。

### 执行性能

静态使用冻结参数，不重新执行 solver；动态保留既有求解路径。相同资源/配置下，应与真实 ATT 基线的模板及执行参数一致，不以放宽数值容差或性能阈值验收。观测关闭与开启分别测量。

### 内存和产物大小

避免为“收窄”复制完整 schedule；按现有生命周期持有只读引用。静态只输出胜出代码和必需参数，动态保留全部代码。记录 host 临时产物、最终代码大小与峰值内存；不引入常驻 solver 缓存。

## 接口设计

### 新增/修改接口描述

内部接口的逻辑契约如下，名称为实施建议，不要求建立通用插件框架：

```text
PreparePlanGeneration(schedule, mode, scene, targetBudget, attOptions)
  -> StaticPlanResult | DynamicPlanResult | existing diagnostic

StaticPlanResult = selections + solvedTiling + targetBudget
DynamicPlanResult = allSelections + attProgram/schema + targetBudget

ExecuteAtt(attProgram, invocation, hostToolchain)
  -> raw solver result | execution diagnostic
```

编排入口拥有最终结果；ATT 程序及 schema 在准备/执行期间有效；借用 schedule 的接口不得让引用逃逸调用生命周期。静态结果中 solvedTiling 必填，移除当前可选输出指针。采用现有错误体系承载阶段与原因，不另建一套对外状态码。

### 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|---|---|---|---|
| 接口说明 | 接口评审 | 是 | 内部策略/执行边界和预算契约 |
| 接口说明 | 文档 | 是 | 本设计及实施后的调用链说明 |
| 接口兼容 | 行为兼容 | 是 | 保持已确定的静态胜出、动态全保留规则 |
| 接口兼容 | ABI/API | 是 | 内部签名调整；公开 ABI 不变，检查实际导出边界 |
| 接口约束 | 清晰报错 | 是 | 无解、工具链失败、坏 key、预算不符分别定位 |
| 接口测试 | 独立用例 | 是 | 策略、执行、解码及集成分层 |

## 软件设计

### 关键数据结构

复用 `PlanSelection`、`TilingFieldSchema` 与字段身份。`TilingCodeBundle` 回归 ATT 源码产物职责；静态具体参数从源码 bundle 中分离，由互斥的最终结果类型表达。预算不另建平台数据库，只承载一次查询的明确结果。

### 关键技术/算法

候选枚举按 schedule 顺序线性遍历；胜出 key 通过既有身份 helper 映射。若需索引，仅在当前请求内构造，不维护持久模板清单。PlanSelector 不计算 cost 表达式、不读局部 group_id 猜全局身份。

沿用 [Codegen 语义命名规范](../../../../docs/zh/autofuse/design/features/codegen.md#template-semantic-naming) 和 [Schedule 语义传递](../../../../docs/zh/autofuse/design/features/schedule.md#template-semantic-propagation)。不新增实现分类；策略/角色从 Schedule 传递，API feature 仍在最终 API 选择点采集。完整 graph/result/group/key/subcase/template 与 reuse 逻辑身份随胜出快照传递；不补做求解获取名称，不用父名掩盖丢失信息。

### 流程设计

先解析目标与模式，再统一准备候选/schema/ATT；静态执行、验证并冻结，动态直接交付程序；随后按 selections lowering，最后组装产物。静态 ATT 未选出完整方案及工具链/结果契约错误终止当前 MLIR 路径，保留错误类别并交由外层模式策略处理；禁止提交半成品。动态运行时返回无效 key 时必须报错，不能默认运行 plan 0。

### 对子模块的修改

| 模块 | 必需修改 |
|---|---|
| PlanSelector/PlanSelection | 删除重复枚举与无效 unsupported 输出；策略结果成对交付 |
| AttSchemaAdapter | 分离 ATT 准备与执行；场景与资源显式输入 |
| CodegenOrchestration | 统一准备入口；移除 narrowed_result 误名及不必要复制；统一目标解析 |
| target / 内存预算消费点 | 对齐物理容量、预留与可用容量，禁止重复扣减 |
| 静态参数/产物组装消费点 | 接收必填静态结果，验证字段映射、workspace 和 dispatch 身份 |
| UT / lit / E2E | 增补契约边界用例，复用现有 fixture |

实施顺序：先职责整理与现有行为对照，再补齐预算和原子结果契约；最终验收前两部分必须闭环。不得以第一步完成为由宣称已修复预算不一致。

### 错误处理

#### 系统错误

文件、编译器缺失、链接、进程退出、超时、读取失败报告实际阶段，保留受控诊断；无 solver 输出不等于 ATT 无解。清理路径覆盖失败、取消和成功。

#### 接口错误

结构损坏、缺字段、重复/未知 key、非法字节长度、目标预算不一致属于契约错误。ATT 明确拒收、求解无可行解、lowering 能力不足分别记录。PlanSelector 不决定是否允许 legacy fallback；各类别由外层 MLIR 环境变量对应策略处理。基础设施或数据损坏必须保留真实错误类别，不能伪装成“不支持”以影响策略。

## 安全检查

已按编码红线检查设计：索引/key/字段长度先验证，预算减法和 workspace 大小防溢出；临时目录及进程明确所有权；不拼 shell、不硬编码敏感信息/芯片；不改图依赖或公开 ABI；无跨 SO 静态析构清理。实施时需对真实代码再次检查，本文不替代实现审查。

## 兼容性检查

保留现有脚本、安装布局、Python API、ATT DFX 与 kernel 入口。内部结果类型变化不外泄。静态 ABI 若剔除未选字段，必须按现有 schema 映射组装并核对产物缓存版本；不兼容旧产物不能命中。CPU 和设备各自使用经确认的资源来源，禁止为通过测试改成同一个虚构容量。

## 测试设计

### 测试边界

策略 gtest 可打桩执行器；解码测试使用真实 schema/手工异常数据；执行器覆盖临时编译程序；集成使用真实 ATT，不打桩胜者。lit 验证 AFIR 模板身份与静态参数传递，CPU/NPU 验证编译到计算结果。所有数值检查保持现有容差。

### 测试用例设计

| 类别 | 关键测试项 | 方法 | 类型 |
|---|---|---|---|
| 功能 | 静态胜出 plan/template 不为 0，显式非连续 key | 与同输入真实 ATT 的胜出身份、参数逐项比较 | gtest/ST |
| 功能 | 多 subgraph、多 stage、RCore AR/RA | 完整保留阶段组合，检查 phase 1 workspace 与 phase 2 消费 | ST/CPU/NPU E2E |
| 功能 | 动态多 shape 选择不同模板 | 验证全候选代码存在、实际 dispatch key 和数值 | ST/NPU E2E |
| 功能 | 单候选仍需要求解 | 断言获得实际参数，不能仅凭数量跳过 ATT | UT/ST |
| 功能 | Sum + Max 共享 scratch | 同预算对照 TBuf/queue、tiling、workspace 和结果 | CPU/NPU E2E |
| 身份 | 显式/隐式 key 混合拒绝、非连续 key、跨 stage 相同 key、同 stage 重复 key、flat schedule | 对照 ATT 实际编号，验证静态原始身份和动态 dispatch；同 stage 歧义必须失败 | UT/ST |
| 身份 | reuse、多 group、语义名称快照 | 使用公共 helper，验证逻辑身份不被物理 primary 覆盖 | UT/ST |
| 预算 | 物理/预留/可用容量、核数上下界 | 零值、预留超容量、恰好边界、无符号下溢；ATT 与 AFIR 一致 | gtest |
| 异常 | 静态 ATT 返回失败，在严格和允许 fallback 模式下分别执行 | 均停止当前 MLIR lowering；严格模式报错，允许回退模式按外层策略进入 legacy；保留 ATT 诊断 | bridge UT/ST |
| 异常 | 无解、坏 key、缺/重复字段、字段宽度错误、半截输出 | 不返回部分结果、不选 plan 0、不回退近似评分 | gtest |
| 异常 | host compiler 缺失、编译失败、执行失败、超时 | 分类、诊断、进程与目录清理 | UT/ST |
| 异常 | 静态胜者不支持、动态某候选不支持 | 验证整体策略，不换次优、不静默删候选 | bridge UT/ST |
| 兼容 | DFX 固定 plan/case、场景选项、资源变化后复用 | 固定身份一致；不兼容调用明确拒绝旧产物 | UT/E2E |
| 交付 | 安装态 CANN 头文件/库、Python 扩展 | 真实环境编译 solver 与导入，不依赖源码树偶然路径 | ST/NPU |
| 性能 | 枚举/生成/执行次数及分阶段耗时 | 静态 1 次求解，动态编译期 0 次；记录总耗时、内存和产物大小 | UT/Benchmark |

“数值通过”不能单独证明选择正确：每个选择测试必须检查实际执行身份；动态至少覆盖两种实际胜者，RCore 至少一个形状满足 `nA < corenum`，避免单核或未启用 R 方向分核造成假覆盖。

### 测试命令

已配置 MLIR 测试的构建树：

```bash
cmake --build build --target check-ut check-lit check-e2e-cpu -j 8
```

Mac 使用仓内开发容器：

```bash
AF_CANN_VOLUME=cann-volume bash autofuse/mlir/scripts/enter_dev_container.sh -- \
  bash scripts/test/run_autofuse_test.sh -u -s -m mlir -j 8
```

设备上按现有统一入口装配环境后运行：

```bash
python3 autofuse/mlir/test/e2e/npu/run.py --list
python3 autofuse/mlir/test/e2e/npu/run.py
```

真实 ATT 静态/动态专项 fixture 接入现有测试目标，不新建平行 runner。设备验证必须记录 commit、工具链、资源预算、模块来源及实际 key；fallback 单独列出，不能算 MLIR 路径通过。本次实施结果按下方记录更新；未完成项仍保留为待验收。

### 实施与验证记录（2026-10-31）

已落地统一 `PreparePlanGeneration` 入口，静态结果与动态 ATT 源码使用不同的 variant 分支；资源从同一份 TargetProfile 传给 ATT、AFIR 和 host ABI，预留 UB 统一扣除一次。字段快照包含每个 stage 自己的资源字段，不能用 fusion 作用域的同名值替代。静态聚合保留胜出的原始 plan ID，动态仍校验完整候选编号。

独立 ATT 程序复用 `TilingLib::GenGetWorkspaceSizeFunc` 生成真实 workspace 查询，满足 ATT 的 host 依赖；同一次调用保存 workspace 大小和偏移，静态 ABI 返回冻结的大小，不再重算符号公式。不提供固定返回零的占位实现。字段映射失败会指出具体 scope/field。外层 fallback 策略不变，失败不交付部分代码产物。

实机验证发现冻结后的单文件 tiling 曾误带 ATT 拆分标记，导致 Inductor 报 `split host source has no TilingHead`。静态产物已改用既有单文件编译协议，并在 bridge UT 中断言不产生拆分标记；动态 ATT 多文件协议不变，Python 侧校验保持不变。

- 选择器、协议和 bridge UT：201/201 通过，包含静态非零 plan 聚合、动态编号缺口拒绝及资源/字段协议的负向回归。
- 完整 lit：271/271 通过；Inductor Python adapter 现有单文件/拆分协议测试：40/40 通过。
- CPU E2E：RCore AR/RA 各 3 个数值测试通过；Sum + Max、静态广播及动态广播各 1 个数值测试通过。RCore 日志检查了实际 `ATT_SELECTED plan=1`。
- NPU：DevEnvC_1AtLk / CANN 9.2.0，工作区代码快照 `311622910d3cdca05ddb03055bd51d1cea363217`。运行目录 `/mnt/workspace/af-npu/runs/311622910d3c-45daa5776fe0`，Reduce 清单 14 项、62 格全部 PASS，fallback、缺陷和预期偏差均为 0。未将这轮称为整个 NPU 清单的全量回归。
- 动态补充：复用现有 `load_sum_store.py --dynamic`，默认选择与 DFX 固定 plan 1 各运行三个 shape，两个入口均返回 0；容差和 PASS 预期不变。结果在 `dynamic-v2-cases`、`dynamic-plan1-v2-cases`。首轮补充驱动的 case ID 与预期名称未同步，产生对账误报；修正驱动后使用新目录重跑。
- 产物身份审计：76 个静态 host 单元各只含一个原始 plan，覆盖 plan 0/1/2；两个动态场景各只有一个编译单元，均保留 plan 0/1/2。审计记录为运行目录中的 `artifact-identity-audit.json`。
- 动态原生 ABI：直接调用上述动态产物的 `kernel.so`，对 `ks0/ks1=(16,32)、(16,37)、(17,32)` 读取真实结果。默认选择均为 plan 0、block_dim=2、workspace=0；DFX 固定选择均为 plan 1、block_dim=2、workspace=512。记录在 `abi-probe/*/result.json`，不只依赖环境变量推断身份。
- 实机 RCore 冻结产物保留原始 plan 1；输入预算为 40 核、物理 UB 196608 字节，可用 UB 为 196352 字节，预留仅扣除一次。
- 本地 CANN LLD 长时间停留在链接，验证使用系统 LLD 14；设备默认 CANN LLD 同样耗时异常，最终设备 host 构建临时使用 GNU ld.bfd 2.42，覆盖记录保存在运行目录的 `linker-override.json`。项目 linker 配置未改，不能据此宣称默认 CANN LLD 构建已通过。

## 验收标准

- [x] 两种模式共享完整候选/schema/ATT 准备，静态只执行一次真实求解。
- [x] 静态只 lowering ATT 胜出的完整方案，包含每个 subgraph 的胜出 plan 及全部必要 stage；未选模板不进入 lowering。
- [x] 静态胜出身份与参数不可分离；动态候选全集与 runtime key 覆盖一致。
- [x] 静态 ATT 未选出完整方案时，不进入当前 MLIR lowering；报错或 legacy fallback 由外层环境变量策略决定，ATT 原因与执行器/bridge 错误可区分。
- [ ] 多 subgraph/RCore 完整阶段、显式 key、reuse 与语义名称身份均有行为验证。
- [x] ATT 与 AFIR 资源预算同源，预留只扣一次，workspace 与 UB 分开核算。
- [x] 对照真实 ATT：模板、tile、block_dim、workspace 一致，数值与性能 gate 不放宽。
- [x] 无解、工具链、协议错误及能力不足清晰区分；失败无部分产物、无隐式替代选择。
- [x] UT、lit、CPU E2E 与相关 NPU 验证完成，未执行或 fallback 项明确报告。
- [ ] 记录编译耗时、峰值内存和产物大小，更新调用链文档并检查公开兼容边界。

设计文档检查结果：已阅读并逐项覆盖跨特性交叉检查、编码红线及模板语义规范；明确 UT/ST/E2E 策略和性能测量。ATT 执行仍沿用既有临时程序方式；显式启用运行时 PGO 时拒绝静态冻结，由外层配置决定报错或 legacy fallback。其他运行时调优组合仍需专项验证，不能从默认配置通过推断已覆盖。


### 代码检视检查结果

- [x] 编码红线：已核对 `docs/guidelines/编码红线.md`，检查了真实 ATT 选择、失败无部分产物、预算扣减和临时目录生命周期。
- [x] 跨特性交叉影响：已按 `cross_feature_check.md` 检查 Autofuse bridge/codegen、host ABI、Python 编译入口和设备执行；不修改 SuperKernel。静态单文件产物与 Inductor 的协议问题已由实机复现并修复。
- [x] 贡献规范：已检查 `CONTRIBUTING.md`；本次提交仅包含 PlanSelector 相关代码、测试和文档，排除其他任务的修改；独立 Git 快照用于可追溯的设备验证，本轮未创建 PR。
- [x] 代码格式：相关 C++ 文件已按仓库 `.clang-format` 处理，`git diff --check` 通过。
- [x] Pre-commit：已确认 `.pre-commit-config.yaml` 的 clang-format/OAT 要求；本轮未执行全仓 OAT 或打包回归，不将其计为已通过。
