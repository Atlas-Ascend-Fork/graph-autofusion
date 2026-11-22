# AFIR 代码质量改进开发文档

> 状态：已完成
> 日期：2026-08-05
> 现状复核日期：2026-08-06
> 文档类型：开发与重构记录

> 代码一致性：未知 scheduler enum 显式失败、shape verifier/inference 复用、强类型 function role、集中
> AscendC kernel ABI 和 bridge operation registry 均已落地；旧最小 `AFIRToAscendC` pass 及其专属
> support matrix test 已删除，Queue demo 已迁到 `mlir/demo`。

# 简介

## 目的

本文记录 AFIR bridge、dialect verifier 和 AscendC lowering 中四类代码质量问题的背景、修改理由、
实现方案、影响范围与验证方式，供后续维护者评审、回归和继续演进使用。

目标读者包括 Autofuse MLIR bridge、AFIR dialect、schedule、codegen 和测试维护者。本文也明确本次
bridge 算子注册表与未来完整 Capacity/Capability 架构的边界，避免将粗粒度的“能否构造”误认为
schedule、ATT、canfuse 和 codegen 所需的细粒度算子能力模型。

## 范围

本次包含：

- ScheduleToAfir 对未知 dtype 和 axis type 的错误处理；
- AFIR binary elementwise 广播、reduce 的 verifier 与 shape inference 规则复用；
- AFIR 函数角色与 AscendC kernel ABI 的分层定义；
- bridgeable operation 的唯一注册点；旧的最小 AscendC lowering 及其支持矩阵测试随目录重构删除；
- 与上述修改直接相关的 CMake、lit、gtest 和说明文档。

本次不包含：

- legacy `AFIRToAscendCQueue` demo 的 shape、buffer、tiling 算术安全重构；该 demo 将由新的完整 pass 取代；
- demo 的镜像、CANN 路径和平台参数通用化；后续在 `test/` 独立子目录建设通用 E2E 看护；
- shell 脚本公共库抽取；在没有统一用户入口前不增加额外脚本层级；
- 完整 Capacity/Capability 服务，包括 scalar input、inline、inplace、canfuse、schedule、ATT 和 codegen
  的细粒度查询接口；
- SuperKernel、Python/C++ binding、runtime 或打包交付接口修改。

# 总体概述

## 软件概述

### 项目介绍

AFIR 是 ASCIR schedule 结果迁移到 MLIR 后的中间表示。当前数据流为：schedule 结果由 bridge 转为 AFIR，
AFIR verifier 检查 IR 语义，若干 pass 生成循环和 AscendC kernel 边界，正式 lowering 再把已支持的 AFIR op
转换为 AscendC dialect op。

原实现存在四类问题：

1. bridge 对未知枚举静默映射为合法值，非法输入可能改变数据类型或调度语义；
2. verifier 和 inference 分别维护 shape 规则，已经出现显式错误 result shape 能通过 verifier 的语义漂移；
3. 函数角色、GM memory space、TilingData 和 kernel 属性散落在多个 pass，template 与 entry 的行为不清晰；
4. bridge 使用手写 if 链，无法从实现注册点稳定获得可构造算子集合；旧的最小 lowering 没有产品使用者，
   不再作为正式 backend 能力来源。

### 产品环境介绍

修改位于 `autofuse/mlir/`，依赖项目既有 LLVM/MLIR、PyAsc/EmitAsc 和 Autofuse ASCIR 组件。
`AscendCKernelABI` 描述的是本仓内 AscendC kernel 的 MLIR 边界约定，不引入所谓“CANN kernel”组件，
也不假设仓库中存在额外的 CANN translation/runtime 消费者。

本次不新增 runtime/aclrt/AscendC API 调用，不修改 CANN Toolkit 安装路径、run 包、wheel 或动态库对外接口。

## 软件功能

- bridge 仅接受有明确定义的 scheduler dtype 和 axis type，未知值产生诊断并终止构造；
- binary elementwise op 使用 NumPy 风格右对齐广播规则，verifier 与 inference 得到一致 shape；
- reduce op 的 axis 范围和降维 shape 由同一实现计算；
- AFIR 函数以强类型 `#afir.function_role<kernel_body|kernel_entry>` 标记角色；
- signature pass 按角色生成并校验 AscendC kernel ABI，重复执行保持幂等；
- bridge 从注册点导出可构造算子集合；正式 lowering 支持集合由后续完整 pipeline 重新建立。

## 设计约束

- 保持现有 LLVM/MLIR 和 PyAsc 依赖版本，不引入新的第三方库；
- 不改变 SuperKernel 与 Autofuse 的组件边界；
- `kernel_body` 是私有计算函数，`kernel_entry` 是可作为 kernel 入口的非 private 函数；
- 正式 ABI 支持层只由 signature materialization 使用，lowering pass 不自行推断函数角色；
- legacy queue demo 保持自包含实现，只同步 `ascendc.num_inputs` 命名，不扩展其生命周期；
- `.td` 中的 AFIR op 是 dialect 可表达全集，不等于 bridge/backend/E2E 支持集合；
- 编译命令限制为 `-j 8`，避免 Autofuse 构建并行度过高导致 OOM。

## 假设和依赖关系

- scheduler 的已知 dtype/axis 枚举值与当前 ASCIR 定义保持一致；新增枚举必须显式补充映射与测试；
- AFIR binary op 保持两个 shaped operand 和一个 shaped result 的现有定义；
- AscendC kernel 边界保持“输入 GM memref、输出 GM memref 参数、workspace、TilingData”的参数顺序；
- `ascendc.num_inputs` 当前用于标识输入参数与后续输出参数的边界，并用于 ABI 幂等校验；输出个数可由
  `总参数数 - num_inputs - workspace - TilingData` 推导，因此本次不重复增加 `num_outputs`；
- 完整 Capacity/Capability 架构的字段、查询协议和生成方式尚未定稿，本次注册 API 不作为其稳定公共接口。

# 需求分析与设计

## 整体介绍

修改后的职责边界如下：

```text
ASCIR schedule result
        |
        v
ScheduleToAfir --校验枚举--> AfirBuilder --bridge registry--> AFIR module
                                                               |
                                      verifier/inference 共用规则
                                                               |
                         function_role --> signature materialization
                                                               |
                                  AscendCKernelABI support
                                                               |
                       formal conversion pipeline（后续建设） --> AscendC dialect
```

四项改进彼此正交：枚举校验保护 bridge 输入，shape 规则保证 dialect 语义，函数角色/ABI 明确函数边界，
bridge 注册表解决构造分支与可构造集合的维护漂移。完整 Capacity/Capability 架构未来可以读取或替换该
粗粒度集合，但不需要回退前三项改进。

## 功能需求

### 功能需求 1：拒绝未知 scheduler 枚举

1. 介绍

   旧实现将未知 dtype 默认为 `float16`，将未知 axis type 默认为 `original`。这种容错会把非法输入伪装成
   合法 AFIR，可能造成数据类型或调度语义改变。新实现要求所有枚举转换显式成功。

2. 输入

   输入为 `ge::DataType` 和 `af::Axis::Type`。已知值按既有字符串映射处理；未定义值、invalid sentinel 或
   后续新增但尚未适配的值均视为非法输入。

3. 处理

   `DTypeStr` 和 `AxisTypeStr` 返回 `std::optional<std::string>`。bridge DTO 构造链改为
   `mlir::FailureOr<T>`，任一转换失败即在当前 MLIR location 发出诊断并向上传播 failure。

4. 输出

   合法输入生成与原实现等价的 AFIR；非法输入返回空 module，不再生成带错误语义的 IR。

### 功能需求 2：统一 shape verifier 与 inference

1. 介绍

   binary verifier 原来只在 operand 互相冲突时检查部分 result 维度；reduce verifier 只检查 rank 上界。
   inference 却维护了更完整的广播和 axis 规则，导致两条语义路径漂移。

2. 输入

   binary 输入为两个 shaped operand 和显式/推导 result type；reduce 输入为 ranked tensor、`axis` 和
   result type。动态维按 MLIR `ShapedType::kDynamic` 处理。

3. 处理

   binary verifier 和 inference 共用 `inferBroadcastShape`：右对齐比较维度，支持不同 rank、维度 `1`
   和动态维，不兼容维度返回 failure。reduce verifier 和 inference 共用 `inferReduceShape`：先校验
   `0 <= axis < rank`，再删除指定维度。verifier 使用 `verifyCompatibleShape` 比较推导结果和显式 result。

4. 输出

   正确广播/降维 IR 保持可用；错误 result shape、非法 reduce axis 和不兼容广播在 verifier 阶段被拒绝。

### 功能需求 3：分层定义函数角色和 AscendC kernel ABI

1. 介绍

   旧实现通过散落的 UnitAttr 和字符串判断 template/global 角色，并在多个 pass 重复定义 GM memory space、
   TilingData 和输入个数属性。新设计把 AFIR 语义角色与 AscendC kernel ABI 分为两层。

2. 输入

   输入为携带 `#afir.function_role<kernel_body|kernel_entry>` 的 tensor-based `func.func`。kernel body 必须是
   private；kernel entry 不得是 private。未携带角色的函数不由 signature pass 处理。

3. 处理

   AFIR dialect verifier 校验角色 attribute 的类型、挂载对象和 visibility。signature pass 收集输入、输出
   tensor 边界，转换为 GM memory space 22 的 memref 参数，追加 workspace 和 TilingData，并调用
   `AscendCKernelABI` 设置 `ascendc.aicore`、`ascendc.num_inputs`；只有 kernel entry 添加
   `ascendc.global`。再次运行时先校验已有 ABI，完整且一致则跳过，部分构造或角色冲突则失败。

4. 输出

   template 保持 private 且不带 `ascendc.global`；kernel entry 带 `ascendc.global`。两者均获得一致的
   AscendC kernel 参数布局与输入计数元数据。

### 功能需求 4：从 bridge 注册实现导出 op 支持集合

1. 介绍

   `.td` 只声明 AFIR 可表达的 op 全集，不能代表 scheduler bridge、正式 backend 或 E2E 的真实支持度。
   本次消除 bridge 构造 if 链的重复维护入口；正式 backend 能力留给后续完整 lowering pipeline。

2. 输入

   bridge 注册项包含 scheduler op name 与 AFIR operation name。

3. 处理

   `AFIR_BRIDGE_OPERATION_REGISTRY` 同时驱动 bridge op 构造和 bridgeable mapping 导出。bridge UT 检查
   注册项无重复。旧最小 lowering 的静态支持集合不迁移，避免把 demo 能力误报为正式 backend 能力。

4. 输出

   当前可查询 bridgeable 状态，不再新增另一份手写“全支持”列表。lowerable 和 E2E verified 状态由后续
  正式 pipeline/capability 设计统一提供。

## 非功能需求

### 可维护性

- 错误通过 `FailureOr` 向上传播，不依赖默认值；
- shape 规则在 verifier/inference 之间单源实现；
- AFIR 角色和 AscendC ABI 分层，避免 backend 细节扩散到 dialect op 定义；
- bridge 只有一个支持注册点；
- 不保留旧最小 lowering 的粗粒度 support matrix。

### 可测试性

- scheduler enum 转换提供独立 gtest；
- shape 正向、动态维、不同 rank、错误 result、非法 axis 和不兼容广播使用 lit 覆盖；
- 函数角色、visibility、ABI golden 和 pass 幂等使用 lit 覆盖；
- bridge op 注册无重复使用 gtest 覆盖。

### 可移植性

修改不新增芯片型号、架构、镜像或框架硬编码。GM memory space 22 属于当前 AscendC kernel ABI 支持层的
既有约定，已集中定义，未来适配变化只需修改该层。修改不引入宿主 x86/aarch64 分支。

### 可靠性

修改不申请资源、不增加并发或 runtime 生命周期。非法 scheduler 枚举、部分 ABI 状态和不一致 shape 均
显式失败。registry 使用静态、确定顺序的数组，不依赖无序容器遍历生成 IR。

### 特性交叉影响

| 场景 | 适用性 | 分析说明 |
|------|--------|----------|
| SuperKernel Python 接口 | 不适用 | 未修改 Python 包、选项、pytest 或 wheel |
| SuperKernel C++/AOT 接口 | 不适用 | 未修改 `libascendsk.so`、AOT、RDV 或其 ABI/API |
| Autofuse 图优化 | 适用 | bridge 输入校验会阻止非法 schedule 结果进入 AFIR；不改 ASCIR 图结构、融合 pass、数据边或控制边 |
| Autofuse Codegen/Backend | 适用 | 修改 AFIR signature materialization、AscendC kernel 属性和正式 conversion pattern 注册方式；lowering 语义不变 |
| AscendC API / Runtime 交互 | 不适用 | 未新增或修改 runtime/aclrt/设备端 AscendC API 调用，仅描述 MLIR ABI 元数据 |
| Python/C++ 混合绑定 | 不适用 | 未修改 pyautofuse、CPython 引用计数、导入路径或异常传播 |
| 构建与打包 | 适用 | ABI、signature 和 loop pass 合并到单一 `AFIRToAscendC` library；未修改 install/package、run 包或第三方依赖 |
| 测试与覆盖率 | 适用 | 保留 bridge gtest 和 shape/function role lit；删除只服务旧最小 lowering 的 support matrix gtest |
| 性能与日志 | 适用 | 增加线性 rank 校验和静态注册表查询；无运行时路径和新增日志，影响可忽略 |
| 兼容性 | 适用 | 内部 textual IR 从 UnitAttr/`cann.num_inputs` 迁移到强类型角色/`ascendc.num_inputs`；对外 API/ABI 不变 |

## 性能

### 编译时长

shape verifier/inference 的复杂度为 `O(rank)`，与原逻辑同阶。bridge 注册表在编译期展开为静态数组。
ABI、signature 和 loop pass 合并到单一 conversion target，减少独立 target 数；`af-opt` 显式链接实际
依赖，避免由 `MLIRMlirOptMain` 传递全量 MLIR 静态库。

### 执行性能

本次不修改生成 kernel 的计算、搬运、tiling 或 launch 路径，不新增 runtime 分支和日志，因此无设备执行
性能影响。ABI materialization 只在编译期 pass 中执行。

### 内存和产物大小

新增静态映射、attribute 定义和测试代码规模较小；不新增常驻缓存、运行时内存或交付数据文件。
未修改安装规则，run 包和 wheel 布局不变。

## 接口设计

### 新增/修改接口描述

| 接口 | 参数/返回 | 用途与失败行为 |
|------|-----------|--------------|
| `#afir.function_role<kernel_body|kernel_entry>` | AFIR enum attr | 标识 AFIR 函数角色；visibility 不匹配时 verifier 失败 |
| `getFunctionRoleAttr` / `hasFunctionRole` / `setFunctionRole` | `Operation*`、角色 | 统一读写强类型角色，供 builder 和 pass 使用 |
| `getAscendCGlobalMemRefType` 等 ABI helper | tensor/function/context | 集中生成 GM memref、workspace、TilingData |
| `checkAscendCKernelABI` | `func::FuncOp`、角色，返回 `FailureOr<bool>` | 区分未生成、已完整生成和非法部分状态 |
| `setAscendCKernelAttributes` | 函数、角色、输入个数 | 设置 `ascendc.aicore/num_inputs/global` |
| `getAFIRBridgeableOperations` | 返回只读 mapping array | 查询当前 scheduler bridge 可构造的 op |

schedule→AFIR bridge 的 DTO 与构造接口统一位于 `mlir::afir::bridge`，不再使用带临时性质的
`af_demo` 命名空间；AFIR dialect 类型和 pass 接口仍位于 `mlir::afir`。

上述接口均位于 Autofuse MLIR 内部 C++ 层，不作为 Graph-autofusion 对外稳定 ABI。未来完整 CapacityService
可以组合 declared/bridgeable/lowerable/E2E 状态，也可以替换查询外观；不应把 `GetApiCallName`、
`IsNodeValid` 等旧 ASCIR codegen 接口机械复制到 MLIR op 定义中。

### 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|--------|----------|----------|------|
| 接口说明 | 是否需要接口评审 | 是 | AFIR role 与内部 ABI helper 需由 MLIR/codegen 维护者评审 |
| 接口说明 | 是否需要补充文档 | 是 | 本文记录语义、边界和使用方 |
| 接口兼容 | 行为是否兼容 | 是 | 合法输入/lowering 行为保持；非法输入由静默降级改为失败 |
| 接口兼容 | ABI/API 是否兼容 | 否（对外） | 未修改导出动态库、Python、脚本或配置接口；内部 textual IR 有迁移 |
| 接口约束 | 约束不满足时是否清晰报错 | 是 | enum、shape、role visibility 和部分 ABI 状态均有诊断 |
| 接口测试 | 是否需要独立接口用例 | 是 | gtest 与 lit 已覆盖内部接口和 attribute 语义 |

## 软件设计

### 关键数据结构

- `FunctionRole`/`FunctionRoleAttr`：由 TableGen 生成，生命周期归 MLIRContext 管理；
- `AFIRBridgeOpMapping`：两个 `StringRef` 的静态只读映射，底层字符串具有静态生命周期；
- `TilingData`：继续使用 `emitasc::PyStructType`，字段来自 `afir.asc_graph` size vars；无 graph 时保留 demo fallback；
- `ascendc.num_inputs`：i32 attribute，表示 AscendC kernel 参数列表中输入 tensor 参数个数。

### 关键技术/算法

广播 shape 采用右对齐逐维合并：`1` 可扩展到另一维，两个已知且不等、均非 `1` 的维度不兼容；动态维
与已知维兼容并保留可验证结果。reduce 只接受非负且小于 rank 的 axis。本次不引入负 axis 规范化。

ABI 幂等判断先检查 `ascendc.aicore` 与 `ascendc.num_inputs` 是否成对存在，再检查参数数量、无返回值、
workspace/TilingData 类型和 `ascendc.global` 是否与角色一致。

### 流程设计

正常流程：bridge 校验输入并构造带 kernel body role 的 AFIR；dialect verifier 检查 shape 和角色；signature
pass 生成 AscendC kernel ABI；后续正式 lowering pipeline 完成转换。

异常流程：未知枚举、shape 不兼容、role/visibility 冲突、部分 ABI 或未注册 op 均发出 MLIR diagnostic
并使当前构造/pass 失败，不生成看似合法的降级结果。

回滚流程：四个代码主题分别提交，可按枚举校验、shape、role/ABI、op support registry 独立回滚。

### 对子模块的修改

| 子模块 | 修改点 |
|--------|--------|
| `mlir/bridge/ScheduleToAfir` | enum 转换返回 optional，DTO 构造返回 FailureOr |
| `mlir/bridge/AfirBuilder` | 使用强类型 kernel body role；以单一宏表构造并导出 bridgeable op |
| `mlir/dialect/AFIR` | 新增 FunctionRole attr/verifier；统一 math/reduce shape 规则 |
| `mlir/conversion/AFIRToAscendC` | 统一容纳 AscendC kernel ABI、template function signature 和 loop generation |
| `mlir/conversion/AFIRToAscendC` | 删除无产品使用者的旧最小 lowering，保留正式 pipeline 的 ABI/loop 基础 |
| `mlir/demo/AFIRToAscendCQueue` | legacy demo 仅同步 `ascendc.num_inputs` 名称和说明 |
| `mlir/test` | 新增 gtest/lit 和 CMake 测试入口 |

### 错误处理

#### 系统错误

本次不新增内存、文件、动态库或 runtime 资源操作。MLIR/CMake 编译失败由现有构建系统报告；不吞掉错误。

#### 接口错误

内部 C++ 转换使用 `std::optional`/`mlir::FailureOr`/`LogicalResult` 返回失败，并通过 MLIR diagnostic
给出 dtype 值、axis type/axis id、shape 或 ABI 状态上下文。没有新增用户可见错误码。

## 安全检查

- 未新增密码、Token、账号、公网地址、邮箱或真实凭据；
- 未新增芯片、平台、框架或网络名称硬编码；
- 未新增数组索引、内存长度、buffer size、tiling 除法或内存申请；reduce axis 在删除维度前已校验；
- 未新增资源生命周期、异步 runtime、aclrt 或 AscendC API 调用；
- 未修改图节点、数据边或控制边，不涉及图改写等价性；
- registry 使用显式静态顺序，不依赖 `unordered_map` 或指针地址，结果确定；
- 未新增高频日志；
- legacy queue demo 已知的 shape/buffer/tiling 算术检查缺口不在本次修改范围，未来完整 pass 必须在将
  外部值用于乘法、除法、队列长度和 i32 转换前完成 checked arithmetic 与 runtime guard；
- 本次新增静态对象只保存常量 `StringRef`/数组，不在析构阶段调用跨 SO 接口。

结论：本次实现未触碰编码红线；明确保留的 legacy demo 风险没有因本次修改扩大。

## 兼容性检查

- 合法 schedule 输入的 AFIR 结果保持一致；非法/未知枚举从静默降级改为显式失败，属于预期纠错；
- binary/reduce 的正确 IR 保持兼容，过去错误但能通过 verifier 的 IR 现在被拒绝；
- 内部 textual IR 从 `afir.template_function` 迁移到 `afir.function_role`，仓内用例和 builder 已同步；
- 内部 C++ bridge namespace 从 `af_demo` 迁移到 `mlir::afir::bridge`，仓内 codegen、工具和测试调用方已同步；
- 内部属性从 `cann.num_inputs` 统一为 `ascendc.num_inputs`。当前仓库没有独立 CANN translation/runtime
  组件消费旧名称；formal pass 与 legacy demo 用例已同步；
- 未增加 `ascendc.num_outputs`，因为输出参数个数可由函数参数布局稳定推导，避免重复元数据漂移；
- 不修改 Python/C++ 对外接口、动态库符号、脚本参数、配置文件、安装路径、run 包或 wheel；
- conversion target 合并与 `af-opt` 显式链接不影响 `--no-autofuse` 的 SuperKernel 路径。

## 测试设计

### 测试边界

输入边界包括已知/未知 scheduler enum、ranked/dynamic/unranked shape、合法/非法 reduce axis、正确/错误
function role visibility、未生成/完整/部分 ABI，以及 bridge 注册集合。输出以 module 构造结果、
MLIR diagnostic、FileCheck golden 和 gtest assertion 为准。

测试不调用真实设备、runtime 或网络，不需要 mock runtime；AscendC dialect lowering 仅检查生成 IR。

### 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|----------|------------|----------|----------|
| 功能 | 已知 dtype/axis 正常映射 | `ScheduleToAfirTest` | UT/gtest |
| 异常 | unknown dtype/invalid axis 被拒绝 | `ScheduleToAfirTest` | UT/gtest |
| 功能 | 不同 rank、维度 1、动态维广播和合法 reduce | `afir_shape_verifier.mlir` | UT/lit |
| 异常 | 错误 result、不兼容广播、非法 reduce axis | `afir_shape_verifier_invalid.mlir` | UT/lit |
| 功能 | template/entry 角色、ABI golden、重复 pass | `afir_function_role.mlir` | UT/lit |
| 异常 | role visibility 和挂载对象非法 | `afir_function_role_invalid.mlir` | UT/lit |
| 特性交叉 | bridge 注册无重复 | `ScheduleToAfirTest` | UT/gtest |
| 性能 | 无新增运行时路径；检查测试/增量编译无明显异常 | 手工观察 | 手工验证 |

### 测试命令

在当前 MLIR 开发容器中：

```bash
bash mlir/scripts/enter_dev_container.sh -- \
  gmake -C build ScheduleToAfirTest/fast af-opt/fast -j8

bash mlir/scripts/enter_dev_container.sh -- \
  /opt/llvm/bin/llvm-lit -v build/autofuse/mlir/test/lit

bash mlir/scripts/enter_dev_container.sh -- \
  build/autofuse/mlir/test/ut/bridge/ScheduleToAfirTest

```

若宿主 `build/CMakeCache.txt` 记录容器路径，不直接在宿主运行 `cmake --build`。编译必须使用 `-j8`。

## 验收标准

- 未知 scheduler dtype/axis 不产生 AFIR module，并有明确诊断；
- binary/reduce verifier 与 inference 使用同一 shape 规则，所有正负向 lit 通过；
- template/entry visibility、global 属性和 AscendC kernel ABI 参数布局符合设计，signature pass 可幂等执行；
- 源码中无 `cann.num_inputs` 和旧 `afir.template_function` 残留；
- bridge 支持集合由实现注册点导出且无重复；
- 受影响 C++ 目标、全部 MLIR lit 和 bridge gtest 通过；
- `git diff --check` 通过，工作区无 build、日志、凭据或其他临时产物；
- 本文覆盖模板所有章节、编码红线和跨特性交叉影响；
- 完整 Capacity/Capability 架构明确留作独立设计，不把本次 coarse support registry 当作最终能力接口。

### 设计文档检查结果

- [x] 跨特性交叉影响：已按 `cross_feature_check.md` 检查 SuperKernel、Autofuse、AscendC/runtime、
  Python/C++、构建交付、测试、性能和兼容性场景；影响集中在 Autofuse MLIR bridge/codegen 与测试。
- [x] 编码红线：已检查敏感信息、输入校验、整数运算、资源生命周期、ABI/API、图改写确定性和 runtime
  约束；未发现本次实现违反项。
- [x] 测试设计：覆盖 bridge gtest、shape/role lit 正负向用例；旧最小 lowering 的专属 golden/support matrix 已删除。
- [x] 性能评估：新增逻辑均为编译期线性校验或静态注册，无 kernel/runtime 执行性能影响。
