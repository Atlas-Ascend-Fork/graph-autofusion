# MLIR 钉 plan / 钉 template 的总闸

> **历史方案，已移除（2026-10-22 核对）**：下文的 `AF_MLIR_PINNED_PLAN_ID`、
> `AF_MLIR_PINNED_TEMPLATE_ID` 和 `AF_MLIR_ENABLE_PLAN_PINNING` 已无生产代码读点，不能再用于复现。
> 当前使用 `AUTOFUSE_DFX_FLAGS="--force_schedule_result=N;--force_tiling_case=M"`，不需要 MLIR 总闸。
> N/M 必须按当前 shape 的实际候选确认；例如 `(4,15,1)` 消除尾维后 AllLoad 为 plan 2，非退化 ARA 为 plan 3。
> 候选失败不会被逐个剔除；详见[当前选择行为](2026-09-28-mlir-plan-selection-feasibility-fallback-design.md)。
> 下文的需求、命令和验收均为历史记录，不是现行操作指南。

# 简介

## 目的

给 `AF_MLIR_PINNED_PLAN_ID` / `AF_MLIR_PINNED_TEMPLATE_ID` 这两个「强制指定调度候选」的能力加一道
**默认关闭**的总闸 `AF_MLIR_ENABLE_PLAN_PINNING`，把它们从「谁都能顺手设的环境变量」收成「必须显式
声明自己在调试」的能力。

目标读者：MLIR 后端开发者、写 reduce/elementwise E2E 语料的人、排查「为什么这个形状在设备上 init
failed」的人。

## 范围

**在范围内**：`bridge/CodegenOrchestration` 的 `ReadMlirCodegenOptions` 选项解析；CPU E2E 注册脚手架
`test/e2e/cpu/AddCpuCase.cmake`；NPU E2E 与各设计文档里可直接执行的复现命令行。

**不在范围内**：钉中之后的候选裁剪逻辑本身（`PrepareMlirCandidates` 一行不改）、ATT 求解器、
`2026-09-28-mlir-plan-selection-feasibility-fallback-design.md` 的功能需求 2（逐候选下降容错，至今
未实现）、把钉的键从位置下标改成模板类型（见下「备选方案」C）。

# 总体概述

## 软件概述

### 项目介绍

`AF_MLIR_PINNED_PLAN_ID` 是 `43f7582e` 为调试引入的：当时 reduce 图不钉 plan 直接失败
（见 `2026-09-19-ub-tile-addressing-model-design.md` 的记录），钉选是唯一能把某个调度候选单独拎出来
下降的手段。此后它被大量用于钉住 RCommon / RCore / AllLoad 三个模板写 E2E 语料
（`2026-09-27`、`2026-10-01` 两份设计的验收都建立在它之上）。

`f2ff1c66` 落了 `2026-09-28` 设计的功能需求 1：静态形状不再由 `SelectStaticWinner` 收窄到单个赢家，
而是保留全部候选，交给运行期 ATT 按 perf 选。**钉选从此与现网路径的设计意图相反**——它做的正是
被那次变更废除的收窄。本方案给它加闸，把「调试能力」这个定位写进代码而不是只写在注释里。

### 产品环境介绍

不涉及 CANN Toolkit、AscendC、runtime、Python 包或 run 包的任何接口。唯一的外部面是**进程环境变量**，
读取点与既有的 `AF_ENABLE_MLIR_CODEGEN` 完全同构（`std::getenv` + 字面量 `"1"` 比较），不新增解析框架。

## 软件功能

1. 新增环境变量 `AF_MLIR_ENABLE_PLAN_PINNING`，默认关闭，取值 `"1"` 才算开。
2. 闸关闭时，`AF_MLIR_PINNED_PLAN_ID` / `AF_MLIR_PINNED_TEMPLATE_ID` **不被采信**。
3. 闸关闭却设了钉：报配置错误（`option_error`），**不静默忽略**。
4. 闸开启时两个钉的行为与变更前逐字一致。
5. 全部测试路径（CPU E2E 脚手架、NPU E2E 脚本、文档里的复现命令）同步带上开闸。

## 设计约束

1. **不改钉中之后的任何逻辑。** 本方案只决定「这两个变量算不算数」，候选裁剪、模板选择、产物装配
   一行不动，以便「行为变化」这个结论可以由产物逐字对照验证。
2. **判据只认 `"1"`。** 与 `kMlirCodegenEnableEnvName` 保持同一套写法。两道闸出现两套真值判定
   （一道认 `"true"` 另一道不认）是比默认值错更难查的坑。
3. **不新增配置文件、不新增 CMake 选项、不新增 Python kwarg。** 钉选是调试能力，承载它的机制应当
   和它的生命周期一样轻。
4. 现网路径（不设任何钉）必须零行为变化、零额外开销。

## 假设和依赖关系

- 依赖 `2026-09-28` 设计功能需求 1 已合入（`f2ff1c66`）：静态路径保留全部候选。若该变更被回退，
  钉选重新变成 reduce 图能跑的前提，本方案的默认值就得重新论证。
- 假设所有正当的钉选调用方都在仓内（CPU E2E 注册、NPU E2E 脚本、设计文档的复现命令）。已用
  `grep -rn "AF_MLIR_PINNED"` 全仓核对，无遗漏；仓外若有个人脚本，会以明确报错的形式暴露，而不是
  静默改变行为。

# 需求分析与设计

## 整体介绍

钉选的危害不在「钉错了」，而在**钉对了也会少一条退路**。

ATT 生成的 host tiling 对交给它的 selections 逐个试：
`GenGetAllSchedulesResults` 把每个候选展成 `kScheduleResultFunctions` 的一项，循环调用、丢弃返回值、
用 `best_perf` 留最优，只有**全部**候选都解不出时 `GetResultSummary` 才失败
（`att/generator/tiling_code_gen_impl.cpp`）。这条退路的**长度等于交给它的候选数**。钉到一个，退路
长度就是 1。

实测证据：`(41, 7, 2)` 归约 dim=1、`amax`，钉 plan 2（一个 RCore 两阶段候选）必挂。设备侧日志：

```
MulticoreTiling: Calculation failed in the first check, core_num=40, ub_size=196352,
  tiling_vars=[1,7,2], pure_mc_vars=[1], ub_threshold=0.1, corenum_threshold=0.4
Failed to get tiling of AscGraph0
```

而用户看到的只有 `RuntimeError: NPU kernel <path> init failed`——既不提 tiling，也不提是哪个候选。
同一形状**不钉**时 ATT 换一个 `tiling_key` 直接通过；p0 / p1 / p3 也都通过。即：这是钉选制造的失败，
不是产品缺陷。

所以本方案的全部内容是：**让这种失败不可能从现网路径到达**，并让误用在编译期就暴露。

## 功能需求

### 功能需求 1：总闸默认关闭

1. 介绍

   新增 `AF_MLIR_ENABLE_PLAN_PINNING`。未设置或取值非 `"1"` 时，`ReadMlirCodegenOptions` 返回的
   `pinned_plan_id` / `pinned_template_id` 均为 `std::nullopt`，与「没设过钉」不可区分。

2. 输入

   进程环境变量，任意字符串或不存在。只有逐字节等于 `"1"` 算开；`"true"` / `"TRUE"` / `"on"` /
   `"0"` / 空串一律算关。非法输入不存在——任何取值都有确定语义。

3. 处理

   在 `ReadMlirCodegenOptions` 里前置一次 `std::getenv` + 字符串比较，闸关则在解析两个钉之前
   `return options`。

4. 输出

   `MlirCodegenOptions` 的两个 `optional` 字段保持空；下游 `PrepareMlirCandidates` 走未钉分支，
   产物与「环境里从没有这两个变量」逐字一致。

### 功能需求 2：闸关着却设了钉 ⇒ 报错，不静默忽略

1. 介绍

   检测到 `AF_MLIR_PINNED_PLAN_ID` 或 `AF_MLIR_PINNED_TEMPLATE_ID` 存在而总闸未开，写
   `options.option_error`，由既有的 `RunMlirCodegen` 前置检查转成 `MlirCodegenStatus::kFailed`。

   **为什么不是静默忽略**：静默忽略的后果不是「少钉一次」。一条本以为钉住了 AllLoad 的 E2E 用例会
   变成「测 ATT 挑中的那个候选」——用例照样过，但测的东西变了，而且**没有任何迹象**。
   `2026-10-01` 设计的五条 AllLoad 验收、`2026-09-27` 的 RCore 验收全部依赖「钉真的生效了」这个前提；
   把这个前提的失效做成静默，等于把三份设计的验收一起变成假绿。

   这与既有的「设了 `AF_MLIR_PINNED_TEMPLATE_ID` 却没设 `AF_MLIR_PINNED_PLAN_ID`」是同一类
   **配置自相矛盾**，处置方式因此保持一致（同样写 `option_error`）。

2. 输入

   两个钉变量的存在性（不解析取值——取值合法与否在闸开之后才判，否则同一个误用会按环境里哪个变量
   先被读到而报出两种不同的错）。

3. 处理

   二者任一存在即构造错误串；消息里给出**是哪个变量**、**要设什么**、以及**为什么默认关**
   （裁掉候选集就裁掉了 ATT 的运行期退路）。只报第一个命中的变量名，不拼两条。

4. 输出

   `option_error` 非空 ⇒ `GELOGW("MLIR codegen: invalid option: %s", ...)` ⇒ `kFailed` ⇒
   `GenerateForInductor` 返回 `af::FAILED`。

   **这是硬失败，不回落 legacy**（`codegen/codegen.cpp:401`：`kNotSupported` 之外的状态都直接返回，
   只有 `kSucceeded` 映射成 `SUCCESS`）。这正是想要的：配置自相矛盾时应当停下来，而不是悄悄换一条
   路把人测的东西换掉。

   边界：`option_error` 的检查在 `options.enabled` **之后**。所以 `AF_ENABLE_MLIR_CODEGEN` 没开时，
   残留的钉变量不会报错（整条 MLIR 通路都没进），返回 `kNotSupported`、照常走 legacy。这与变更前
   一致，不是新增的沉默。

### 功能需求 3：测试路径全量开闸

1. 介绍

   钉选的正当用户全部是测试。逐条给它们补上开闸，且**补在脚手架层而不是逐条注册里**。

2. 输入

   `test/e2e/cpu/AddCpuCase.cmake` 的 `add_mlir_e2e_cpu_case`；`test/e2e/npu/reduce_template_matrix.py`
   与 `reduce_fuzz_matrix.py`；各设计文档的复现命令行。

3. 处理

   - CPU E2E：在 generator 的 `add_custom_command` 里紧挨 `AF_ENABLE_MLIR_CODEGEN=1` 加一条
     `AF_MLIR_ENABLE_PLAN_PINNING=1`。**一处覆盖全部 21 条钉选注册**。选脚手架层的理由：「这是测试」
     对本文件生成的每一条都成立；抄 21 份就一定会有漏一份而静默不钉的那天（功能需求 2 会把那天变成
     编译失败而不是假绿，但不如不给机会）。不设钉的用例不受影响——开闸本身什么都不做。
   - NPU E2E 的 Python harness：由脚本自己设置，与 `AF_MLIR_PINNED_PLAN_ID` 成对出现、成对清除。
   - 文档：凡是标成「可直接执行」的命令行一律补上。

4. 输出

   CPU E2E 的实际命令行变为
   `... -E env AF_ENABLE_MLIR_CODEGEN=1 AF_MLIR_ENABLE_PLAN_PINNING=1 AF_MLIR_PINNED_PLAN_ID=2 ...`，
   已在强制重生成（`rm -rf cases/*/generated`）后由 25 条用例的构建日志确认。

## 非功能需求

### 可维护性

新增一个常量和一段前置判断，`ReadMlirCodegenOptions` 的两个钉解析分支把 `std::getenv` 的结果从
`if` 初始化语句里提到函数作用域（三处读点合为两处），净复杂度基本持平。

闸的**理由**写在 `CodegenOrchestration.h` 常量定义处，含实测形状与失败现象。理由不写在这里只写在设计
文档里，是因为下一个想删掉这道闸的人读的是头文件。

### 可测试性

入口就是 `ReadMlirCodegenOptions`，纯函数（只读环境变量），已有 `mlir::test::ScopedEnvVar` 夹具，
无需打桩。覆盖目标：开/关 × 设钉/不设钉 × 取值 `"1"`/`"true"` 的组合。

**改造既有用例是本方案的必要组成部分，不是附带**：`PinnedMissingPlanOrInvalidValueFails` 在不补开闸时
**照样全绿**，但绿的理由变成「闸关着拒绝了钉」，与用例名声称的「钉了不存在的 plan / 非法值」毫无关系。
三条既有用例因此都显式开闸。

### 可移植性

不涉及芯片型号、SoC 字符串、框架类型或平台判断（编码红线 Graph-autofusion 规则 2）。不新增编译器、
Python 版本或 CANN Toolkit 依赖。`std::getenv` 是 C++ 标准设施。

### 可靠性

- 无内存分配（除错误串，仅在误用路径构造一次）、无资源获取、无异常分支需要清理。
- 读的是 `const char *`，只做 `nullptr` 判断与 `std::string` 比较，不做算术、不作索引。
- 单线程：`ReadMlirCodegenOptions` 在 `RunMlirCodegen` 开头调用一次。`getenv` 的线程安全性问题与变更
  前同构（既有代码已有三处 `getenv`），本方案不加剧也不改善。

### 特性交叉影响

已读 `docs/guidelines/cross_feature_check.md`。逐项：

| 场景 | 适用性 | 分析说明 |
|------|--------|----------|
| SuperKernel Python 接口 | 不适用 | 不涉及 `super_kernel/` 的任何文件、选项解析与 wheel 内容 |
| SuperKernel C++/AOT 接口 | 不适用 | 不涉及 `libascendsk.so` 导出面与 ABI |
| Autofuse 图优化 | 不适用 | 不改 ASCIR 图结构、融合 pass 与 pass 时序 |
| Autofuse Codegen/Backend | **适用** | 改 `bridge/CodegenOrchestration` 的选项解析。候选裁剪逻辑本身不改；**现网路径（不设钉）产物逐字不变** |
| AscendC API / Runtime 交互 | 不适用 | 不新增 `rt*` / `aclrt*` / AscendC 调用 |
| Python/C++ 混合绑定 | 不适用 | 不改 `autofuse/compiler/py_module/` 的导出面。NPU E2E 的 Python 改动在测试脚本内，不是绑定层 |
| 构建与打包 | **适用** | 改 `test/e2e/cpu/AddCpuCase.cmake`。只在测试目录内，不涉及 install/package/run 包/第三方依赖；增量与离线构建不受影响（只是 `-E env` 多一个赋值） |
| 测试与覆盖率 | **适用** | 新增 1 条 UT，改造 3 条既有 UT（其中 1 条此前是假绿），见测试设计 |
| 性能与日志 | **适用（可忽略）** | 每次 codegen 多一次 `getenv`；新增日志只在误用路径且沿用既有 `GELOGW` 点位，不新增日志点 |
| 兼容性 | **适用** | 已有钉选调用方的行为改变（不开闸即失败）。仓内调用方已全量同步；见兼容性检查 |

## 性能

### 编译时长

每次 `RunMlirCodegen` 多一次 `std::getenv` + 一次短字符串比较，量级为纳秒，相对整条 codegen 流水线
（MLIR 下降 + ATT tiling 生成 + bisheng 编译，秒级）不可测。

不涉及 `autofuse/optimize`、`autofuse/codegen` 的循环或图遍历，不新增全量扫描。

CPU E2E 的 generator 命令行多一个环境变量赋值，`cmake -E env` 的开销同样不可测。

### 执行性能

零影响。改动全在编译期的 host 侧选项解析，设备侧生成代码一字不变（现网路径产物逐字不变，已由 44 条
lit 与 25 条 CPU E2E 的重生成确认）。

### 内存和产物大小

新增一个 `llvm::StringLiteral` 常量（字面量，进 rodata，约 30 字节）。不新增结构体字段、缓存或常驻
内存。wheel、run 包、动态库大小无变化。

## 接口设计

### 新增/修改接口描述

```cpp
// bridge/CodegenOrchestration/CodegenOrchestration.h
inline constexpr llvm::StringLiteral kMlirPlanPinningEnableEnvName = "AF_MLIR_ENABLE_PLAN_PINNING";
```

对外面只有环境变量：

| 变量 | 取值 | 默认 | 语义 |
|---|---|---|---|
| `AF_MLIR_ENABLE_PLAN_PINNING` | `"1"` 为开，其余为关 | 关 | 是否采信下面两个钉 |
| `AF_MLIR_PINNED_PLAN_ID` | 非负十进制整数 | 无 | 钉住的 plan 下标；闸关时设置 ⇒ `kFailed` |
| `AF_MLIR_PINNED_TEMPLATE_ID` | 非负十进制整数 | 无 | 钉住的 template 下标；需同时设 plan；闸关时设置 ⇒ `kFailed` |

`MlirCodegenOptions` 的字段、`ReadMlirCodegenOptions` 的原型、`RunMlirCodegen` 的原型均不变。

### 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|--------|----------|----------|------|
| 接口说明 | 是否需要接口评审 | 否 | `mlir::bridge` 内部常量 + 调试用环境变量，无对外导出符号 |
| 接口说明 | 是否需要补充文档 | 是 | 本文档；`reduce_template_matrix.py` docstring；各设计文档的复现命令行 |
| 接口兼容 | 行为是否兼容 | **是** | 不设钉的路径完全兼容；设了钉的调用方必须加开闸，否则硬失败。见兼容性检查 |
| 接口兼容 | ABI/API 是否兼容 | 否 | 无对外符号、结构体布局、枚举值变更 |
| 接口约束 | 约束不满足时是否清晰报错 | 是 | 功能需求 2 即为此项；错误串给出变量名、要设什么、为什么默认关 |
| 接口测试 | 是否需要独立接口用例 | 是 | `PlanPinningIsOffByDefaultAndRejectsPinsWhenOff` |

## 软件设计

### 关键数据结构

无新增数据结构。`MlirCodegenOptions`（`CodegenOrchestration.h`）的生命周期不变：`RunMlirCodegen`
栈上 `const` 局部量，不持有任何资源。

有意**不**在其中加 `plan_pinning_enabled` 字段：闸的状态在解析完之后就没有读点了，留一个字段只会制造
第二个判据。

### 关键技术/算法

不涉及算法。唯一的设计决定是**在哪一层拦**：

- 在 `ReadMlirCodegenOptions`（选中）：钉选能力的唯一入口，拦在这里下游一行不改，「闸关 ⇔ 没设过钉」
  可以由产物逐字对照证明。
- 在 `PrepareMlirCandidates`：钉的消费点。拦在这里需要把闸状态传下去，多一个字段、多一个判据。
- 在 CMake / Python 调用方：拦不住仓外脚本，默认值不成立。

### 流程设计

```
ReadMlirCodegenOptions
  ├─ 读 AF_ENABLE_MLIR_CODEGEN → options.enabled
  ├─ 读 AF_MLIR_ENABLE_PLAN_PINNING → pinningEnabled（只认 "1"）
  ├─ 读两个钉变量的指针（此处不解析取值）
  ├─ pinningEnabled == false
  │    ├─ 两个钉都没设 ⇒ 原样返回（现网路径）
  │    └─ 任一已设     ⇒ 写 option_error，返回
  └─ pinningEnabled == true ⇒ 既有解析（逐字不变）
       ├─ plan 取值非法     ⇒ option_error
       ├─ template 取值非法 ⇒ option_error
       └─ 设了 template 没设 plan ⇒ option_error

RunMlirCodegen
  ├─ !enabled || Cube 融合 ⇒ kNotSupported（残留的钉不报错，照常走 legacy）
  └─ option_error 非空 ⇒ GELOGW + kFailed ⇒ af::FAILED（不回落 legacy）
```

无回滚/清理流程——不获取任何资源。

### 对子模块的修改

| 模块 | 修改 |
|------|------|
| `bridge/CodegenOrchestration/CodegenOrchestration.h` | 新增 `kMlirPlanPinningEnableEnvName` 常量 + 理由注释（实测形状与失败现象） |
| `bridge/CodegenOrchestration/CodegenOrchestration.cpp` | `ReadMlirCodegenOptions` 前置总闸判断；两个钉的 `getenv` 结果提到函数作用域 |
| `test/ut/bridge/CodegenOrchestrationTest.cpp` | 新增 1 条用例；3 条既有用例补开闸（其中 `PinnedMissingPlanOrInvalidValueFails` 此前会变成假绿） |
| `test/e2e/cpu/AddCpuCase.cmake` | generator 的 `add_custom_command` 加 `AF_MLIR_ENABLE_PLAN_PINNING=1`，覆盖全部钉选注册 |
| `test/e2e/npu/reduce_template_matrix.py` | docstring 说明开闸要求 |
| `test/e2e/npu/reduce_fuzz_matrix.py` | 成对设置/清除开闸与钉 |
| `docs/design/2026-09-21 / 09-27 / 09-28 / 10-01` | 复现命令行补开闸；`09-28` 的功能需求 1、兼容性、T10 三处更新 |
| `test/lit/{conversion,demo}/*reduce*.mlir` | 产物来源注释里的命令行补开闸（RUN 行跑的是 `af-opt` 读预制 `.mlir`，与环境变量无关，无功能变化） |

### 错误处理

#### 系统错误

无内存申请、无文件操作、无动态库加载、无 runtime 调用。错误串的 `std::string` 拼接失败会抛
`std::bad_alloc`，由 `RunMlirCodegen` 既有的 `try` 块兜住——但该拼接发生在 `try` 之前（在
`ReadMlirCodegenOptions` 内）。这与既有的三处 `option_error` 拼接同构，不是本方案新引入的暴露面。

#### 接口错误

误用（闸关着设钉）⇒ `option_error` ⇒ `GELOGW` 一条，等级 W，非高频路径（每次 codegen 至多一条）
⇒ `kFailed` ⇒ `af::FAILED`。用户可见的是编译失败和那条 `GELOGW`，消息自带修复方法
（「要设 `AF_MLIR_ENABLE_PLAN_PINNING=1`」）。

## 安全检查

已读 `docs/guidelines/编码红线.md`。逐条：

- **1 硬编码敏感信息**：不涉及。新增的唯一字面量是环境变量名。
- **2 外部输入作索引/长度/偏移**：**本方案正是对这一条的收紧**。`AF_MLIR_PINNED_PLAN_ID` 是外部输入
  且被用作候选集下标；既有代码已有范围校验（`PrepareMlirCandidates` 的「pinned plan N does not exist
  in every subgraph」），本方案在其之前再加一道「你有没有资格设它」。注意取值解析被移到闸之后，所以
  闸关时根本不解析外部输入。
- **3 整数溢出/反转/除 0**：不新增算术。既有的 `getAsInteger(10, uint64_t)` + `static_cast<size_t>`
  路径未改动。
- **4 资源释放覆盖异常分支**：不获取资源。`getenv` 返回的指针不拥有、不释放、不保存（只在函数内使用，
  不跨 `setenv` 调用）。
- **5 内存申请前判大小**：不申请内存（除错误串）。
- **6 未指定求值顺序/未定义行为**：三次 `getenv` 是三条独立语句，无参数求值顺序依赖。
  `getenv` 返回 `nullptr` 的情形每处都先判。
- **7 禁止无理由扩大修改范围**：本次只动钉选相关的读点与其测试/文档调用方。工作区里既有的
  `DataMovement.cpp` / `TransferPlan.h` 注释改动**不属于本方案**，不随本方案提交。
- **G1 API/ABI 兼容**：无导出符号变更。环境变量的兼容影响见「兼容性检查」。
- **G2 禁止硬编码芯片/平台/框架类型**：不涉及判型。
- **G3 图改写等价性** / **G4 pass 时序** / **G5 图改写确定性**：不改图、不改 pass、不新增遍历。
  现网路径的候选集与编号与变更前逐字一致。
- **G6 高频路径禁止默认开启海量日志**：新增日志只在误用路径，且复用既有 `GELOGW` 点位。
- **G7/G8 runtime 接口生命周期与开放性**：不涉及。
- **G9 Python/C++ 绑定引用计数**：不涉及绑定层。
- **G10 构建脚本不破坏增量/离线构建**：`AddCpuCase.cmake` 只在 `-E env` 的赋值列表里加一项，不改
  依赖图、不下载依赖。**但见下「已知缺陷」**。
- **G11 测试临时产物不入库**：不新增产物。
- **G12 全局/静态对象析构跨 SO**：新增的是 `inline constexpr llvm::StringLiteral`，无动态初始化、
  无析构。

### 已知缺陷（本方案未修，单独记录）

CPU E2E 的 generated 源码**不依赖 `AddCpuCase.cmake`**：`build.make` 里该 `add_custom_command` 的
`DEPENDS` 只有 generator 二进制与 tiling stub。所以改了 generator 的环境变量而不碰 generator 源码时，
已有的 generated 产物不会重生成，用例照样绿——绿的是旧产物。

本方案的验证因此采用**强制重生成**（`rm -rf cases/*/generated`）而非依赖增量构建。
根治办法是把 `AddCpuCase.cmake` 加进该命令的 `DEPENDS`；这是既有问题（不是本方案引入），按编码红线
第 7 条记录而不顺手改。

## 兼容性检查

- **不设钉的调用方**（现网、以及绝大多数 lit/UT）：完全不受影响，产物逐字不变。
- **设了钉的调用方**：必须同时设 `AF_MLIR_ENABLE_PLAN_PINNING=1`，否则编译硬失败。**这是刻意的
  不兼容**——备选方案 A（静默忽略）兼容但会制造假绿，见功能需求 2 的论证。
- **仓内调用方已全量同步**：`grep -rn "AF_MLIR_PINNED"` 核对，CPU E2E 的 21 条注册由脚手架一处覆盖，
  NPU E2E 两个 harness 自带，文档命令行逐条补。
- **仓外个人脚本**：会以带修复提示的编译失败形式暴露，不会静默改变被测对象。
- 不影响安装目录、动态库、Python 包、脚本参数、配置文件与产物命名。
- 不影响历史产物：钉选只作用于编译期候选集，没有持久化格式。

## 测试设计

### 测试边界

**入口**：`ReadMlirCodegenOptions`（UT，经 `mlir::test::ScopedEnvVar` 控制环境）；`RunMlirCodegen`
（UT，经 `BuildValidFusedScheduleResult` 夹具）；CPU E2E 的 generator 命令行（构建日志）。

**出口**：`MlirCodegenOptions::{pinned_plan_id, pinned_template_id, option_error}`；
`MlirCodegenStatus`；generated 源码。

**打桩**：无。被测函数只读环境变量，`ScopedEnvVar` 这个既有夹具就是全部所需。

### 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|----------|------------|----------|----------|
| 功能 | T1：闸关 + 不设钉 ⇒ 两个 optional 空、`option_error` 空（现网路径） | `PlanPinningIsOffByDefaultAndRejectsPinsWhenOff` | UT |
| 功能 | T2：闸开 + 设 plan ⇒ 解析生效、无错误 | 同上用例末段；`ReadsPinnedPlanId`、`PinnedExistingPlanSucceeds` | UT |
| 异常 | T3：闸关 + 设 plan ⇒ `option_error` 非空、optional 空 | 同 T1 用例 | UT |
| 异常 | T4：闸关 + 设 template ⇒ 同样报错 | 同 T1 用例 | UT |
| 异常 | T5：闸取值 `"true"` ⇒ 不算开，设了钉仍报错（判据只认 `"1"`） | 同 T1 用例 | UT |
| 异常 | T6：闸开 + 钉不存在的 plan / 非法取值 ⇒ 仍按既有路径报错 | `PinnedMissingPlanOrInvalidValueFails`（补开闸，否则假绿） | UT |
| 性能 | T7：编译时长 | 不单独测：改动是一次 `getenv`，无可测增量 | 手工判断 |
| 兼容性 | T8：不设钉的现网路径产物逐字不变 | lit 全量（44 条）+ `check-ut` | ST |
| 特性交叉 | T9：三个模板的钉选 E2E 在开闸后仍通过 | 强制重生成后逐个构建并运行 9 条钉选 reduce E2E，覆盖 RCommon/RCore/AllLoad × AR/RA/ARA/ARAR + 钉 template | E2E |
| 特性交叉 | T10：CPU E2E 的实际命令行确实带上了开闸 | `rm -rf cases/*/generated` 后看构建日志 | ST |

### 测试命令

```bash
# 容器内（autofuse/mlir/scripts/enter_dev_container.sh）
clang-format -i autofuse/mlir/bridge/CodegenOrchestration/CodegenOrchestration.{h,cpp} \
                autofuse/mlir/test/ut/bridge/CodegenOrchestrationTest.cpp
cmake --build /workspace/build --target check-ut -j 8
cmake --build /workspace/build --target check-lit -j 8

# T9 / T10：强制重生成（CPU E2E 的 generated 不依赖 AddCpuCase.cmake，见「已知缺陷」）
rm -rf /workspace/build/autofuse/mlir/test/e2e/cpu/cases/*/generated
cmake --build /workspace/build --target check-e2e-cpu -j 8
```

预期：`check-ut` 全绿（含新增用例）；`check-lit` 全绿且输出逐字不变；重生成后的 generator 命令行含
`AF_ENABLE_MLIR_CODEGEN=1 AF_MLIR_ENABLE_PLAN_PINNING=1 AF_MLIR_PINNED_PLAN_ID=<n>`。

## 验收标准

1. 新增 UT `PlanPinningIsOffByDefaultAndRejectsPinsWhenOff` 覆盖 T1、T3、T4、T5、T2(末段)。
   — **已完成**（2026-10-02）。
2. 三条既有钉选 UT 补上显式开闸，其中 `PinnedMissingPlanOrInvalidValueFails` 的假绿被消除。
   — **已完成**。
3. `check-ut` 全量通过。 — **已完成**：84/84。
4. CPU E2E 强制重生成后，generator 实际命令行含 `AF_MLIR_ENABLE_PLAN_PINNING=1`。
   — **已完成**：25 条用例重生成并通过各自的 generator 测试，命令行已在构建日志中确认。
5. 9 条钉选 reduce E2E（RCommon/RCore/AllLoad × AR/RA/ARA/ARAR，含一条钉 template）逐个通过。
   — **已完成**：`load_sum_store_16x32`、`..._ra_37x17`、`..._ra_agap_64x8x5`、`..._arar`、
   `..._allload_17x32`、`..._allload_ara_4x32x8`、`..._rcore_ar_8x1024`、`..._rcore_ra_1024x8`、
   `load_neg_sum_store_arar_case3` 全部 PASS。
6. 全仓 `AF_MLIR_PINNED` 调用点逐个核对并补齐开闸（含设计文档的可执行命令行）。 — **已完成**。
7. 改动的 C++ 文件通过 `clang-format -i`。 — **已完成**：clang-format 18.1.8 无改动。
8. 本文档随代码合入；`2026-09-28` 设计中「`AF_MLIR_PINNED_PLAN_ID` 行为不变」的三处过期表述更新。
   — 随本次提交。

### 设计文档检查结果

- [x] 跨特性交叉影响：已按 `cross_feature_check.md` 逐项检查，结论见「特性交叉影响」一节。适用的四项是
  Autofuse Codegen/Backend（选项解析）、构建与打包（测试目录内的 CMake）、测试与覆盖率（1 新 + 3 改 UT）、
  兼容性（设了钉的调用方需加开闸）；其余六项不适用并给出理由。
- [x] 编码红线：已按 `编码红线.md` 通用 7 条 + Graph-autofusion 12 条逐条检查，结论见「安全检查」一节。
  无违反；顺带按第 7 条记录了一条**不随本方案修**的既有缺陷（CPU E2E generated 不依赖
  `AddCpuCase.cmake`，改 generator 环境变量不触发重生成）。
- [x] 测试设计：新增 UT `PlanPinningIsOffByDefaultAndRejectsPinsWhenOff`（覆盖开/关 × 设钉/不设钉 ×
  取值 `"1"`/`"true"`）；3 条既有钉选 UT 补显式开闸，其中 `PinnedMissingPlanOrInvalidValueFails` 此前
  会退化成假绿；ST 由 lit 全量 + CPU E2E 强制重生成覆盖；E2E 由 9 条钉选 reduce 用例覆盖三个模板。
- [x] 性能评估：每次 codegen 多一次 `getenv` + 一次短字符串比较，无可测增量；设备侧生成代码一字不变；
  新增日志只在误用路径且复用既有 `GELOGW` 点位。详见「性能」一节。

# 备选方案（未采纳）

**A：闸关着设了钉 ⇒ 静默忽略。** 实现更简单，且对仓外脚本「兼容」。不采纳的理由见功能需求 2：被忽略的
不是一次设置，而是三份设计文档的验收前提——用例照样绿，测的对象悄悄换成了 ATT 挑中的候选。宁可让误用
在编译期硬失败。

**B：把钉选能力整体删掉。** 它今天仍是把单个模板拎出来做数值验收的唯一手段（`2026-09-27` 的 RCore、
`2026-10-01` 的 AllLoad 都靠它），删了就没有替代品。本方案保留能力、收紧入口。

**C：把钉的键从位置下标改成模板类型。** 更值得做——plan 下标是**形状相关**的（同一个
`AF_MLIR_PINNED_PLAN_ID=3` 在不同形状上指向不同模板，`reduce_template_matrix.py` 曾把它当不变量而
记错），而 `ReduceTemplateType`（kCommon/kRCore/kAllLoad）在 MLIR 侧是可识别的
（AllLoad 由 `is_reduce_mem_reuse` 透出，RCore 由两 stage 结构）。但那是**换一种键**，与「这个能力
默认该不该开」是两件事，范围也大得多（要改全部钉选注册与 harness）。本次不混进来。

**D：做成 CMake 选项或 Python kwarg。** 用户已明确选择环境变量形式。环境变量与被它门控的两个变量同构，
也与 `AF_ENABLE_MLIR_CODEGEN` 同构；CMake 选项无法在运行期区分，Python kwarg 要动绑定层导出面。
