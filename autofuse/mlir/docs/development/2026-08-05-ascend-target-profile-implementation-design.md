# Ascend Target Profile 实现设计

> 状态：Target Profile、typed `NpuArch`、执行规格能力查询和首个正式 AFIRToAscendC lowering 已完成
> 首次设计：2026-08-05
> 最近更新：2026-08-12
> 所属架构：[AFIR Op Capability 架构设计](../design/2026-08-05-afir-op-capability-architecture-design.md)

# 1 简介

## 1.1 当前问题

Autofuse MLIR 已具备统一硬件目标模型、独立 Target 选择入口和正式 AFIRToAscendC Target-dependent pipeline，但
schedule、ATT、memory planner、Kernel 编译入口和其他 consumer 尚未全部接入。未迁移代码仍可能接收零散参数；
后续接线必须统一消费
同一份只读 `TargetProfile`，不能重新引入平台名称分支或第二套硬件事实。

本设计解决三个具体问题：

1. 如何把 CANN `platform_config/*.ini` 中的设备事实转换成强类型、只读的硬件模型；
2. 硬件模型包含哪些字段，字段的来源、单位、缺失行为和使用者分别是什么；
3. 编译 pipeline、能力提供者（capability provider，即判断某项后端实现是否适用的查询实现）以及未来的 schedule 和
   ATT 如何使用同一份模型。

## 1.2 设计结论

每次编译只构造一份 `TargetProfile`，构造完成后只读。Profile 包含设备身份、`NpuArch`、计算单元、内存层级、寄存器、
硬件 feature、执行模式能力、直接搬运边和边上的可选速率。它不包含 AFIR Op 名称、AscendC API 名称、conversion pattern、
调度选择、成本估算结果或 CCE intrinsic 清单。

产品模式只接收平台名，例如 `--target=Ascend910B1`，然后读取构建时嵌入工具的原始 CANN INI 字节。开发测试模式可以使用
`--target-profile=<file>` 读取显式 INI 文件。两种模式互斥，不搜索用户机器上的 CANN 安装目录。

配置文件中的原始字段只由 Target loader 解释一次。后续代码只能查询强类型字段，不读取 INI，也不能编写
`if (socVersion == "...")` 形式的平台分支。

## 1.3 范围

本文包含：

- `TargetProfile` 的硬件建模；
- CANN INI 字段到硬件模型的映射；
- 内置 Profile 和外部测试 Profile 的加载；
- 生命周期、诊断、构建时嵌入和使用方法；
- loader、模型和内置资源的测试设计。

本文不迁移通用 scalar、broadcast inline、inplace、resource consumer、ATT 成本算法、CCE Intrinsic lowering 或
PTO-ISA lowering。本期只实现 Target 执行能力查询，并用 `afir.add -> ascendc.add_l2` 验证 provider 与 conversion 的接线。

## 1.4 文档结构

- 第 2 章说明当前代码事实和整体数据流；
- 第 3 章定义硬件模型、字段来源、加载和校验；
- 第 4～5 章说明非功能约束和性能；
- 第 6 章给出 C++ 接口和使用示例；
- 第 7～10 章说明目录修改、错误处理、兼容性、测试和验收。

# 2 总体概述

## 2.1 当前代码事实

Autofuse 的 `mlir/target/Ascend` 已包含 Target Profile 基础数据模型、合成 INI loader、910/950 内置资源、
`TargetSelection` 和统一选择加载入口。`af-opt` 已接入互斥的 `--target=<platform>` 与
`--target-profile=<file>`：指定任一参数时在 optimizer 启动阶段构造一份只读 Profile，并保持到本次
`MlirOptMain` 调用结束；两者都不指定时保持 legacy 行为。正式 AFIRToAscendC pipeline 已使用
`TargetSelectionRequirement::Required`，并使 selection 和 conversion 共用同一份 Profile 和冻结 registry。schedule、ATT、
memory planner、Kernel 编译入口和其他 consumer 仍待迁移。

对 CANN `platform_config` 的前期调查确认：

- A2/A3 和 A5 配置都包含 core 数、GM/L2/L1/L0/UB 容量、UB block/bank 和 memory rate；
- 已检查的 A5 配置额外包含 `vector_reg_width`、`predicate_reg_width`、`wide_reg_width` 和
  `Intrinsic_vectorized_*`；
- 当前内置的全部受支持 Profile 都包含 `version.NpuArch`：910 系列为 `2201`，950 系列为 `3510`；外部测试
  Profile 仍允许缺失该字段，缺失时执行能力返回 `Unknown`；
- CANN 9.1.0 中 `Ascend910A/B/PremiumA/ProA/ProB` 和 `Ascend910B2C` 的 `SoCInfo.memory_size` 为空，不能猜测 GM 容量；
- 执行模式采用受控的架构能力规则：所有已支持架构支持 SIMD + MemoryBased，A5（950 系列）额外支持
  SIMD + RegBased 和 SIMT；SIMT 不区分 MemoryBased/RegBased；
- NDDMA 仍没有已确认的开放事实来源，保持 `Unknown`。

## 2.2 软件功能

Target Profile 模块提供以下功能：

- 根据平台名查询编译进工具的原始 INI；
- 根据显式路径加载开发测试 INI；
- 解析、规范化并校验硬件事实；
- 为每个字段保留明确的强类型、单位和缺失状态；
- 构造稳定的硬件事实摘要（fingerprint），用于诊断、缓存键和测试比较；
- 为 capability、schedule、ATT、memory planner 和未来 CCE provider 提供只读查询。

## 2.3 数据流

```text
平台名 -> 内置 INI 字节查询 ─┐
                             ├-> CANN INI parser -> 字段映射与单位规范化
外部测试文件 -> 文件读取 ───┘
  -> 基础字段校验
  -> TargetProfile（每次编译一份，之后只读）
       ├── capability provider 查询强类型硬件特性和完整执行规格
       ├── schedule 查询 core 和片上容量（迁移时接入）
       ├── memory planner 查询容量、bank 和对齐（迁移时接入）
       └── ATT 读取直连边的可选 rate 并建立自己的成本模型（迁移时接入）
```

Target loader 是原始配置字段的唯一解释者。`TargetProfile` 是编译链内硬件事实的唯一查询入口。ATT 的启动开销、
重叠关系、周期估算和收益不写回 Profile。

## 2.4 产品环境

生成 AFIR、AscendC IR 和 AscendC C++ 只需要 Autofuse 自带的 Profile 资源，不要求安装 CANN。现有
`compiler/python/ascendc_compile.py` 使用 CANN Toolkit 中的 `bisheng` 编译 AscendC C++，因此生成 Kernel 二进制仍需要
CANN Toolkit；真机执行还需要 runtime、驱动和 NPU。

本期只复制明确支持平台所需的原始 CANN INI，不复制完整 `platform_config` 目录。资源来自 `cann-volume` volume 中
CANN 9.1.0 的 `/opt/cann/aarch64-linux/data/platform_config`，适用仓库根目录的 CANN Open Software License Agreement
Version 2.0。来源版本用于审计和追溯，不表示硬件事实随 CANN 版本变化。

# 3 需求分析与设计

## 3.1 输入与内置资源

### 3.1.1 Target 选择

Target-dependent pipeline 的输入只有两种，不设计第三层优先级：

| 模式 | 命令行 | 用途 | CANN 依赖 |
|---|---|---|---|
| 内置平台 | `--target=<platform>` | 产品使用和普通 CPU 验证 | 无 |
| 外部文件 | `--target-profile=<file>` | loader UT、新平台 bring-up | 无 |

启用 Capability、正式 AFIRToAscendC 或其他 Target-dependent pipeline 时，两者同时指定或都未指定都会初始化失败。
不使用这些 pipeline 的既有 `af-opt` 用法可以不指定 Target。外部文件模式不会覆盖内置平台，也不会退化为默认平台。

平台名只允许字母、数字、下划线和连字符，禁止路径分隔符和 `..`。内置模式在生成的只读索引中查找合法平台名；参数
manifest 中声明为 alias 的平台共享一份 canonical INI，但构造 Profile 时保留各自的 `identity.socVersion`。

### 3.1.2 源码资源和构建产物

基础功能源码目录如下：

```text
autofuse/mlir/target/Ascend/
├── AscendTargetProfile.h
├── AscendTargetProfile.cpp
├── TargetSelection.h
├── TargetSelection.cpp
├── TargetProfileLoader.h
├── TargetProfileLoader.cpp
└── Profiles/
    ├── manifest.txt
    ├── generate_builtin_profiles.py
    └── platform_config/
        └── <canonical-platform>.ini
```

`manifest.txt` 使用 `A:A|B|C|D` 格式：左侧 `A` 表示 canonical `A.ini`，右侧表示请求 A/B/C/D 时均加载该文件。
别名平台使用 canonical INI 中的硬件事实，同时保留请求时的 `identity.socVersion`。构建时由
`generate_builtin_profiles.py` 机械生成私有 `BuiltinTargetProfiles.inc`，只嵌入 manifest 引用的原始字节，不解释字段；
生成文件位于 build 目录，不提交源码仓。

当前支持 46 个平台名：910B1～B4、B4-1、910_9362/9372/9381/9382/9391/9392，以及全部 34 个 950 平台，共使用 38 份
canonical INI。其余 CANN 910 配置只用于 volume 对照测试，不进入产品支持索引。

### 3.1.3 交付和运行时生命周期

生成的只读 INI 字节编译进 Target Profile 所属库，并随 `af-opt` 或使用该库的产品编译器一起交付。`.run` 包不安装独立
`<platform>.ini` 或资源目录，也不通过安装路径、当前工作目录或环境变量查找内置 Profile。

内置模式按平台名命中资源后取得 `MemoryBufferRef`；外部文件模式读取 `--target-profile=<file>` 后也得到
`MemoryBufferRef`。两种来源进入同一 parser、字段单位、缺失语义、基础校验和 fingerprint 格式。

Profile 只在 Autofuse 编译任务初始化时构造一次。生成的源码、目标文件、Kernel 二进制和真机运行过程都不读取 Profile。
新增或正式修改内置平台需要重新构建并交付工具；临时新增平台和测试仍使用外部 INI buffer，不要求修改内置表。

## 3.2 硬件模型

### 3.2.1 建模原则

Profile 按硬件事实分成七组：

1. 设备身份：用于选择、诊断和 fingerprint；
2. core 拓扑：描述可用执行单元数量和组合方式；
3. 内存资源：每个 `MemorySpace` 唯一表示一种共享或 core-local 资源；
4. 计算访问权限：描述 Cube、Vector 能直接读写哪些内存资源；
5. 直接搬运边：描述两个内存资源之间是否存在一条硬件直连及其搬运引擎、可选速率；
6. 计算与寄存器规格：描述 Cube、Vector 和 RegBased 的静态资源；
7. 强类型硬件特性（typed feature）：给调用方提供不依赖平台字符串的能力判断。

每个 `MemorySpace` 的归属是固定类型语义，不作为 Profile 实例数据保存：GM、L2 不属于任何计算单元；L1、L0A、L0B、
L0C 属于 Cube；UB 属于 Vector。`getMemoryOwner()` 集中提供这一映射，其他模块不能自行维护第二份归属表。模型不允许
同一种 `MemorySpace` 在多个计算单元上出现独立实例。

Target Profile 只保存直接边，不搜索多跳路径。memory planner 需要把 Cube 输出放入 Vector UB 时，读取同一张图：A2/A3
选择 `Cube.L0C -> GM -> Vector.UB`，A5 可选择 `Cube.L0C -> Vector.UB`。路径合法性由边决定，成本比较由 planner/ATT
完成。

所有字符串和容器由 Profile 自己拥有。Profile 成员不能保存指向 parser 缓冲区的 `StringRef` 或 `ArrayRef`。

### 3.2.2 数据结构

旧代码中的 `ExecutionUnit` 同时包含处理器、AICore 执行单元和 SIMT 执行模式。当前实现已经将处理器、AICore
单元、执行模式和指令形态分开建模：

```cpp
enum class NpuArch {
  V2201,
  V3510,
  Unknown,
};

enum class ProcessorType {
  AICore,
  AICPU,
};

enum class AICoreUnit {
  Cube,
  Vector,
};

enum class ExecutionModel {
  SIMD,
  SIMT,
};

enum class InstructionForm {
  MemoryBased,
  RegBased,
};

struct ExecutionSpec {
  ProcessorType processor;
  std::optional<AICoreUnit> unit;
  std::optional<ExecutionModel> executionModel;
  std::optional<InstructionForm> instructionForm;
};

enum class MemorySpace {
  GM,
  L2,
  L1,
  L0A,
  L0B,
  L0C,
  UB,
};

enum class TargetFeature {
  BF16,
  InfNan,
  NDDMA,
};

enum class FeatureState {
  Available,
  Unavailable,
  Unknown,
};

enum class CubeVectorTopology {
  Combined,
  Split,
  Unknown,
};

struct TargetIdentity {
  std::string socVersion;
  std::string shortSocVersion;
  NpuArch npuArch;
};

struct CoreTopology {
  int64_t aiCoreCount;
  int64_t cubeCoreCount;
  int64_t vectorCoreCount;
  CubeVectorTopology cubeVectorTopology;
};

struct MemorySpec {
  std::optional<int64_t> capacityBytes;
  std::optional<int64_t> blockBytes;
  std::optional<int64_t> bankBytes;
  std::optional<int64_t> bankCount;
  std::optional<int64_t> subBankCount;
  std::optional<int64_t> bankGroupCount;
};

struct MemoryResource {
  MemorySpace space;
  MemorySpec spec;
};

enum class AccessMode {
  Read,
  Write,
};

struct ComputeAccess {
  AICoreUnit unit;
  MemorySpace space;
  AccessMode mode;
};

enum class TransferEngine {
  MTE1,
  MTE2,
  MTE3,
  Fixpipe,
};

struct TransferEdge {
  MemorySpace source;
  MemorySpace destination;
  TransferEngine engine;
  std::optional<int64_t> bytesPerCycle;
};

struct ComputeUnitSpec {
  std::optional<int64_t> frequencyMHz;
  std::optional<int64_t> vectorCalculationBytes;
  std::optional<int64_t> vectorRegisterBytes;
  std::optional<int64_t> predicateRegisterBytes;
  std::optional<int64_t> wideRegisterBytes;
};

```

`TransferEngine` 使用硬件流水线名称，不使用泛化的 `DataMove`：

| 流水线 | 当前建模的主要方向 |
|---|---|
| `MTE1` | `L1 -> L0A/L0B` |
| `MTE2` | `GM -> L1/L0A/L0B/UB`、`L1 -> UB` |
| `MTE3` | `L1/UB -> GM`、`UB -> L1` |
| `Fixpipe` | `L0C -> L1/GM/UB` |

同一对 `MemorySpace` 可以由不同流水线形成多条 `TransferEdge`。`hasDirectTransfer(source, destination)` 只回答
是否至少存在一条直连边；流水线竞争和性能计算必须读取具体边的 `engine` 与 `bytesPerCycle`。

`TargetProfile` 拥有上述结构，并提供只读 accessor。不存在 set 方法，也不暴露内部可变容器。

资源存在性和资源规格是两个独立事实：`MemoryResource` 条目存在表示该 `MemorySpace` 存在；`MemorySpec` 中某个字段为
`std::nullopt` 表示资源存在但该项规格未知。GM、L2 始终存在；Cube local memory 仅在 Cube Core 数大于 0 时存在；UB
仅在 Vector Core 数大于 0 时存在。调用方不能再用容量字段是否为空推断资源是否存在。

`CoreTopology` 是计算单元存在性的唯一事实来源。`cubeCoreCount` 或 `vectorCoreCount` 大于 0 表示对应单元存在，等于 0
表示不存在；两个字段在 INI 中都必须出现。计算单元存在但 `ComputeUnitSpec` 字段缺失时，规格查询成功并返回相应
`std::nullopt`，不能把“规格未知”解释为“单元不存在”。

`NpuArch` 是 `version.NpuArch` 的强类型表示。loader 集中完成 `2201 -> V2201`、`3510 -> V3510`的字符串转换，
缺失或未识别值使用 `Unknown`。不再增加 `AICoreGeneration` 或逐平台代际表。manifest 只维护 canonical platform、
alias 和 INI 资源关系。

`ExecutionSpec` 描述昇腾处理器的执行方式，不属于 AscendC 语言模型。它的组合约束固定如下：

- `AICore + Vector + SIMD` 必须指定 `MemoryBased` 或 `RegBased`；
- `AICore + Vector + SIMT` 的 `instructionForm` 必须为空，SIMT 不区分 MemoryBased/RegBased；
- 对于不适用的维度使用 `std::nullopt`，不填写虚假默认值；
- 属性读写和 Target 能力查询都必须先校验这些组合约束。

### 3.2.3 首期字段映射

下表是首期应实现的字段。容量统一规范化为 bytes，频率统一为 MHz，速率统一为 bytes/cycle。CANN schema 没有明确
单位的字段必须先确认单位，确认前不能进入公共 accessor。

| 类别 | CANN section/key | 规范化字段 | 是否必需 | 缺失行为 | 主要使用者 |
|---|---|---|---|---|---|
| 身份 | `version.SoC_version` | `identity.socVersion` | 是 | loader failure | 诊断、fingerprint |
| 身份 | `version.Short_SoC_version` | `identity.shortSocVersion` | 否 | 空值 | 诊断 |
| 身份 | `version.NpuArch` | `identity.npuArch` | 否 | `Unknown` | 执行能力查询、诊断、fingerprint |
| core | `SoCInfo.ai_core_cnt` | `topology.aiCoreCount` | 是 | loader failure | schedule |
| core | `SoCInfo.cube_core_cnt` | `topology.cubeCoreCount` | 是 | loader failure | schedule、Cube provider |
| core | `SoCInfo.vector_core_cnt` | `topology.vectorCoreCount` | 是 | loader failure | schedule、Vector provider |
| core | `SoCInfo.cube_vector_combine` | `topology.cubeVectorTopology` | 否 | `Unknown` | schedule |
| 内存 | `SoCInfo.memory_size` | `GM.capacityBytes` | 否 | GM 存在，容量为 unknown | memory planner |
| 内存 | `SoCInfo.l2_size` | `L2.capacityBytes` | 否 | L2 存在，容量为 unknown | schedule、ATT |
| 内存 | `AICoreSpec.l1_size` | `Cube.L1.capacityBytes` | 否 | Cube 存在时 L1 存在，容量为 unknown | schedule、memory planner |
| 内存 | `AICoreSpec.l0_a_size` | `Cube.L0A.capacityBytes` | 否 | Cube 存在时 L0A 存在，容量为 unknown | Cube schedule |
| 内存 | `AICoreSpec.l0_b_size` | `Cube.L0B.capacityBytes` | 否 | Cube 存在时 L0B 存在，容量为 unknown | Cube schedule |
| 内存 | `AICoreSpec.l0_c_size` | `Cube.L0C.capacityBytes` | 否 | Cube 存在时 L0C 存在，容量为 unknown | Cube schedule |
| 内存 | `AICoreSpec.ub_size` | `Vector.UB.capacityBytes` | 否 | Vector 存在时 UB 存在，容量为 unknown | Vector schedule、memory planner |
| UB 组织 | `AICoreSpec.ubblock_size` | `Vector.UB.blockBytes` | 否 | UB 存在，block 规格为 unknown | 对齐、memory planner |
| UB 组织 | `AICoreSpec.ubbank_size` | `Vector.UB.bankBytes` | 否 | unknown | bank conflict 分析 |
| UB 组织 | `AICoreSpec.ubbank_num` | `Vector.UB.bankCount` | 否 | unknown | bank conflict 分析 |
| UB 组织 | `AICoreSpec.ubsubbank_num` | `Vector.UB.subBankCount` | 否 | unknown | A5 bank 分析 |
| UB 组织 | `AICoreSpec.ubbank_group_num` | `Vector.UB.bankGroupCount` | 否 | unknown | bank conflict 分析 |
| 计算 | `AICoreSpec.cube_freq` | Cube `frequencyMHz` | 否 | unknown | ATT |
| 计算 | `VectorCoreSpec.vec_freq` | Vector `frequencyMHz` | 否 | unknown | ATT |
| 计算 | `AICoreSpec.vec_calc_size` 或 `VectorCoreSpec.vec_calc_size` | `vectorCalculationBytes` | 否 | unknown | Vector schedule |
| 寄存器 | `AICoreSpec.vector_reg_width` 或 `VectorCoreSpec.vector_reg_width` | `vectorRegisterBytes` | 否 | unknown | RegBased 资源计算 |
| 寄存器 | `AICoreSpec.predicate_reg_width` 或 `VectorCoreSpec.predicate_reg_width` | `predicateRegisterBytes` | 否 | unknown | RegBased 资源计算 |
| 寄存器 | `AICoreSpec.wide_reg_width` 或 `VectorCoreSpec.wide_reg_width` | `wideRegisterBytes` | 否 | unknown | RegBased 资源计算 |
| feature | `SoCInfo.support_bf16` | `BF16` | 否 | `Unknown` | dtype capability |
| feature | `SoCInfo.support_inf_nan` | `InfNan` | 否 | `Unknown` | 数值语义选择 |
| 计算访问 | core-local buffer 语义 | `ComputeAccess` | 是 | 不创建对应访问项 | schedule、memory planner |
| 搬运声明 | `AICoreintrinsicDtypeMap` section | 闭世界搬运图输入 | 是 | loader failure | loader |
| 直接通路 | `AICoreintrinsicDtypeMap.*` 中已映射的搬运 intrinsic | `TransferEdge` | 否 | 不创建对应边 | memory planner、provider |
| 基础通路 | INI schema 未一致枚举的 GM→UB MTE2、UB→GM MTE3 通路 | `TransferEdge` | 是 | GM、UB、Vector 都存在时由 schema adapter 补充 | memory planner |
| 速率 | `AICoreMemoryRates.*`、`VectorCoreMemoryRates.*` | 既有 `TransferEdge.bytesPerCycle` | 否 | 边保留、rate 为空 | ATT Cost Model |

当前内置的 A2/A3 和 A5 Profile 都声明 `cube_vector_combine=split`，Cube Core 与 Vector Core 在硬件拓扑中均独立存在。
`VectorCoreSpec` 是否单独成节只是 CANN 配置 schema 的差异，不能用于判断硬件拓扑。A2/A3 没有独立
`VectorCoreSpec`，Vector 相关字段保存在 `AICoreSpec`；A5 同时提供 `AICoreSpec` 和 `VectorCoreSpec`，Vector 规格优先读取
后者。loader 只能按同一字段的明确来源选择 section，不能用 `cube_freq` 等不同语义的字段补齐缺失的 Vector 事实。

拓扑独立和跨单元内存访问是两个维度。`Vector.UB` 表示 Vector 侧的 UB 资源；`cube_vector_combine=split` 只说明
Cube Core 与 Vector Core 独立，不能据此判断 `Cube.L0C` 是否能直接搬到 `Vector.UB`。该事实由 `TransferEdge` 表达，
不通过 UB 容量、INI section 布局或 topology 推断。

Loader 完成规范化后，`TransferEdge` 集合采用闭世界语义：集合中没有的直接边就是不可用。Loader 只允许根据明确的 INI
intrinsic 映射和上述基线 schema 规则创建边，并且创建前必须确认 source、destination 和相关计算单元存在。速率字段只能
标注已创建的边，不能创建新边。`AICoreSpec.support_fixpipe` 不再规范化为独立 `TargetFeature`；Fixpipe 能力只由具体
`TransferEdge.engine == Fixpipe` 表达，避免 Boolean 与搬运图形成两套事实。

### 3.2.4 架构能力规则

执行模式能力不以独立的 `SIMT` 和 `RegBased` Boolean 对外提供。`TargetProfile` 查询完整的
`ExecutionSpec`，避免调用方将多个独立 feature 自由组合成硬件不支持的形态。

受控的架构能力规则如下：

| 规范化架构 | `Vector + SIMD + MemoryBased` | `Vector + SIMD + RegBased` | `Vector + SIMT` |
|---|---:|---:|---:|
| `V2201` | `Available` | `Unavailable` | `Unavailable` |
| `V3510` | `Available` | `Available` | `Available` |
| `Unknown` | `Available` | `Unknown` | `Unknown` |

A5 对应 `NpuArch::V3510`，A2/A3 对应 `NpuArch::V2201`。canonical platform 和 alias 读取同一份 INI，因此自然获得
相同的 `NpuArch` 和执行能力。provider、conversion 和 emitter 不得按
`SoC_version` 前缀或具体型号分支。

`Unknown` 仍支持 `Vector + SIMD + MemoryBased`，因为这是当前所有硬件的基线能力。这样缺少 `NpuArch` 的外部或合成
Profile 不会破坏现有 Add L2 lowering。只有 A5 新增形态在架构未知时返回 `Unknown`。

Cube 到 Vector UB 的直接通路从 INI 中的通路声明解析，不通过 `NpuArch` 推导。当前配置以
`AICoreintrinsicDtypeMap` 中的 `Intrinsic_fix_pipe_l0c2ub` 建立 `Cube.L0C -> Vector.UB` 的 `Fixpipe` 直连边。

当前内置 A2/A3 Profile 不包含该边，A5 Profile 包含该边。`AICoreMemoryRates.l0_c_to_ub_rate` 在两代配置中都存在，
只描述已存在边的速率，不能创建边。若没有直连，memory planner 仍可通过 `Cube.L0C -> GM -> Vector.UB` 搜索多跳路径；
Target Profile 本身不实现路径搜索或最短路策略。

寄存器宽度继续作为 RegBased 资源计算的输入，但不再用于决定硬件是否具有 RegBased 执行形态。是否能使用
某个具体 AscendC API，仍由该 implementation 的 provider 结合 execution spec、dtype、shape 和 operand 约束判断。

NDDMA 仍作为独立 `TargetFeature` 保留。在事实来源确认前返回 `Unknown`。`Unknown` 表示模型缺少事实，
`Unavailable` 表示硬件明确不支持。Capability provider 不得将 `Unknown` 当作支持。

### 3.2.5 CCE Intrinsic 预留边界

公共实现类型为未来 `CCE Intrinsic Implementation` 预留枚举值。本期不把 INI 中的完整 intrinsic dtype map 暴露为
指令清单，也不注册、不查询、不 lowering CCE Intrinsic。loader 只把已确认的搬运 intrinsic 映射为 `TransferEdge`，
consumer 不接触 intrinsic 字符串。

只有未来确定执行 AFIR 直接 lowering 到 CCE Intrinsic 时，才根据真实 provider 查询需求设计 CCE 指令模型，并决定
intrinsic dtype map 应进入 Target Profile 还是独立的 CCE 指令 Profile。当前 AscendC API 路径只查询 AscendC API provider
和必要的 typed target feature，不逐项检查底层 CCE 指令。

### 3.2.6 暂不建模的配置字段

首期不因为 INI 中存在字段就全部暴露。下列字段只有出现真实 consumer 后才进入公共模型：

- stream 数量和深度；
- AICPU 数量及 CPU cache 配置；
- `DtypeMKN`；
- unzip engine；
- runtime、virtualization 和 simulator 专用字段。

loader 可以忽略这些未消费字段，但不能把未知字段自动解释成 feature。新增字段时必须补充字段来源、单位、缺失行为、
consumer 和测试。

## 3.3 加载、规范化和校验

### 3.3.1 加载步骤

```text
校验 TargetSelection
  -> 内置平台：查询嵌入的 canonical INI 字节
     外部文件：读取显式 INI 文件
  -> 解析 section/key/value 并映射已建模字段
  -> 校验字段范围和字段间关系
  -> 按稳定顺序计算 fingerprint
  -> 返回只读 TargetProfile
```

loader 读取完成后关闭文件。Profile 自己拥有所有字符串和容器，不保留文件句柄、parser buffer 或外部路径的非 owning
引用。

### 3.3.2 校验规则

- 平台名必须通过字符集和长度校验；外部路径必须是可读的普通文件；
- 单个 Profile 最大 256 KiB、单行最大 4096 bytes、最多 4096 个字段；外部文件在读取前检查大小，buffer 入口再次校验；
- 诊断中的外部路径和 buffer identifier 必须转义控制字符，防止伪造多行诊断；
- 必需整数必须能够完整解析为非负 `int64_t`，不能把解析失败静默转换为 `0`；
- `ai_core_cnt` 必须大于零；`cube_core_cnt` 和 `vector_core_cnt` 必须存在且不能为负；
- 容量、UB block 和计算规格字段允许缺失，缺失时保持 `nullopt`；显式提供时必须大于零；
- `AICoreintrinsicDtypeMap` 必须存在；规范化边只能连接已经存在的资源，最终边集合采用闭世界语义；
- bank 数、sub-bank 数和 group 数必须大于零，并通过乘法溢出检查；
- canonical INI 的 `SoC_version` 必须与 manifest 的 canonical 平台一致；构造后 identity 使用请求的平台别名；
- register width 存在时必须为正数，并满足 CANN schema 规定的粒度；
- feature 的显式值与相关资源矛盾时初始化失败；execution spec 的字段组合非法时查询或属性写入失败；
- 未识别字段不导致失败，但不进入公共模型和 fingerprint。

### 3.3.3 Fingerprint

fingerprint 对规范化后的全部公开硬件事实计算 SHA-256。输入按固定字段名和稳定顺序序列化，包括 schema 版本、identity
（含 `npuArch`）、core、memory、register、feature 和 rate。执行能力由 `npuArch` 和版本化规则派生，不在 fingerprint 中
再复制一份能力矩阵。内存字段只编码稳定的 memory space 名称，不编码 execution unit。fingerprint
不包含绝对文件路径、加载时间或容器迭代顺序。

规范化 fingerprint 用于判断两个编译任务看到的硬件事实是否相同；CANN 来源目录和版本另行记录用于审计。共享同组参数的
平台仍因 `identity.socVersion` 不同而具有不同 fingerprint。

## 3.4 派生模型和所有权

`TargetProfile` 保存规范化事实。需要图结构或策略计算的消费者在此基础上构造自己的只读模型：

| 派生模型 | 输入 | 输出 | 所有者 |
|---|---|---|---|
| `MemoryModel` | memory、UB 组织、访问权限、直接边 | memory place、容量、对齐、合法路径 | 编译任务上下文 |
| ATT Cost Model | direct-edge rate、频率、profiling | startup、overlap、cycle、收益 | ATT |

一次编译任务持有 Profile 和需要的派生模型。pass 与 provider 只持有查询期间有效的 `const` 引用。禁止可变
全局单例、进程级隐式当前平台和依赖静态初始化顺序的注册。

## 3.5 与 PyAsc `ascendc` Op 的一致性

### 3.5.1 硬件能力、实现选择和目标 IR 属性

PyAsc 的 `ascendc` dialect 描述“要调用哪个 AscendC API 以及参数是什么”。当前代码中的
`ascendc::Hardware`（定义在 `externals/pyasc/include/ascir/Dialect/Asc/IR/Core/Attributes.td`）只是
`LocalMemAllocator` 和 emitter 使用的内存位置枚举。它没有描述 core 数、L2 容量、RegBased、SIMT 或 NDDMA。

编译链必须区分三类协议：

| 信息 | 唯一事实来源 | 下游处理 |
|---|---|---|
| 硬件支持什么 | `TargetProfile` | provider 使用完整 `ExecutionSpec` 查询 |
| AFIR Op 选择哪个后端实现 | `ascend.implementation` 中的 kind 和稳定 ID | conversion 定位已选中的 registry entry |
| 该实现具体生成什么代码 | AscendC Op 上的 `unit`、`execution_model` 和可选 `instruction_form` | emitter 只读取 IR，不重新猜测实现类型 |
| AFIR 到 AscendC API 的参数排列和执行规格 | AscendC 实现注册项和 typed mapping | provider 和 conversion 共享同一份 mapping |
| AscendC API 的 IR 形状和参数类型 | PyAsc ODS/API Op 定义 | 作为目标 IR 的类型检查和 emitter 输入 |
| `GM/UB/L1/L0A/L0B/L0C` 内存位置 | Profile 的 `MemorySpace` | 通过显式 adapter 映射到 `ascendc::Hardware` |

`ascend.implementation` 的 `kind` 选择后端实现路径，当前协议支持 `ascendc_api`、`cce_intrinsic` 和 `pto_isa`；`id`
选择该路径下的具体实现。对于 AscendC 路径，`id` 对应具体 AscendC API 实现；对于 CCE Intrinsic 或 PTO-ISA 路径，
`id` 对应各自的 intrinsic 或指令实现。当前 registry 只注册 AscendC 实现，但选择协议不限定为 AscendC。

`ascend.implementation` 不增加执行模式或指令形态字段。它是稳定的后端实现选择协议；lowering 后目标 Op 的执行属性是
代码生成协议。两者生命周期和消费方不同，不能合并。

### 3.5.2 内存枚举必须显式映射

`MemorySpace` 与 PyAsc `ascendc::Hardware` 不能依赖数值相等。两者的取值集合也不完全相同：Profile 有 L2，
PyAsc `Hardware` 目前有 BIAS 和 FIXBUF，但没有 L2。

规划的 lowering adapter 提供显式转换：

```cpp
FailureOr<ascendc::Hardware>
mapMemorySpaceToAscendC(MemorySpace space);
```

映射规则为：

- `GM`、`UB`、`L1`、`L0A`、`L0B`、`L0C` 映射到同名 PyAsc `Hardware`；
- `L2` 没有直接的 PyAsc local-memory 表示，不能强行映射；需要 L2 的计划必须在 lowering 前完成 L2 搬运或选择支持
  L2 的专用 API；
- `BIAS`、`FIXBUF` 只有在 Profile 增加对应物理资源并且 lowering 真实需要时，才扩展 `MemorySpace`；不能因为
  PyAsc 有枚举值就把它们加入 Target Profile。

adapter 使用穷举 `switch`，不识别的值返回 failure。UT 必须覆盖所有合法映射和 L2/未知值的负向路径。

### 3.5.3 从 AFIR 到 PyAsc Op 的调用顺序

lowering 不能先创建 PyAsc Op，再把合法性检查推迟给 PyAsc verifier 或后续工具链。正确顺序是：

```text
同一个 Target Profile + registry
  -> query AFIR Op 的 AscendC 候选
  -> provider 用 mapping.executionSpec 查询 Target Profile，再检查 dtype、operand 和 shape
  -> 得到 selected ImplementationKey
  -> conversion 使用同一份 typed mapping 创建 PyAsc ascendc Op
  -> 将 mapping.executionSpec 显式写入目标 IR
  -> emitter 只读取目标 IR 属性并生成代码
```

选择属性只保存 kind 和稳定 ID，不保存 Target fingerprint。Pipeline 为 selection 和 conversion 传入同一份只读 Target
Profile 和 registry；pattern 根据 `ImplementationKey` 创建对应 PyAsc Op，不能重新从平台名查询能力。若 Profile、IR 或
registry 在选择后发生变化，pipeline 必须清除旧选择并重新运行 selection；conversion 不重新调用 provider 判断属性是否
失效。

当前 Add L2 和后续实现使用的 AscendC 执行属性例如：

```mlir
// 当前 Add L2：SIMD + MemoryBased
ascendc.add_l2 ... {
  ascendc.unit = "AiCore.Vector",
  ascendc.execution_model = "SIMD",
  ascendc.instruction_form = "MemoryBased"
}

// A5 SIMD + RegBased
ascendc.add ... {
  ascendc.unit = "AiCore.Vector",
  ascendc.execution_model = "SIMD",
  ascendc.instruction_form = "RegBased"
}

// A5 SIMT：不写 instruction_form
ascendc.add ... {
  ascendc.unit = "AiCore.Vector",
  ascendc.execution_model = "SIMT"
}
```

IR 字符串沿用已有外部协议的拼写，例如 `"AiCore.Vector"`。C++ 类型使用统一的 `AICore`；枚举与字符串的
转换只存在于 AscendC 属性模块。

### 3.5.4 与现有 `bisheng` Kernel 编译链的目标一致性

Autofuse 当前已经使用 `compiler/python/ascendc_compile.py` 调用 CANN Toolkit 中的 `bisheng` 编译 Kernel。PyAsc 只提供
Ascend/EmitAsc dialect 和 emitter；本工程不使用 PyAsc 的 `asc.runtime.compiler` Python 编译封装。

完整链路为：

```text
AFIR
  -> AFIRToAscendC
  -> PyAsc Ascend/EmitAsc IR
  -> AscendC C++ 源码
  -> compiler/python/ascendc_compile.py
  -> bisheng --npu-arch=<arch>
  -> Kernel 二进制
```

当前 `get_soc_type(args)` 根据 `args.soc_version` 生成 `--npu-arch`。Target Profile 接入后，顶层编译任务必须只选择一次
目标平台：同一个规范化 `socVersion` 既用于加载 Profile，也传给现有 Kernel 编译入口。外部 Profile 模式使用文件内经过
校验的 `identity.socVersion`。源码生成阶段可以在没有 CANN 的普通 CPU 环境结束；请求 Kernel 二进制时才检查 CANN 和
`bisheng` 是否可用。

本期不新增一套 PyAsc 编译目标类型，也不改变 `ascendc_compile.py` 的公共参数。如何把 MLIR 编译任务中的规范化 target
接到现有 `soc_version` 参数，是产品 pipeline 接线工作；接线时必须增加 target 一致性检查，不能让 Capability 按一个
平台选择实现、`bisheng` 再按另一个平台编译。

### 3.5.5 一致性失败行为

当前链路遇到以下情况必须失败，而不是默默使用另一套信息：

- selection 与 conversion 没有使用 pipeline 持有的同一份 Target Profile 和 registry；
- Profile 内存位置无法映射到 PyAsc `Hardware`；
- PyAsc ODS 中没有注册 selected implementation 所需的目标 Op；
- Profile 声明 feature 为 `Unknown`，但 provider 将其当作 `Available` 使用。

请求 Kernel 二进制时，编译任务中的 `soc_version` 必须与 Profile 的 `identity.socVersion` 一致；不一致时在调用
`bisheng` 前失败。

# 4 非功能需求

## 4.1 可维护性

- CANN 9.1.0 INI 是平台事实来源；源码保留受支持平台的 canonical INI，字段解释集中在 loader；
- 新增平台只增加 INI、manifest 映射和测试，不修改 pass；alias 平台共享 manifest 指定的 canonical INI；
- 新增公共字段必须同时增加 mapping、单位、校验、accessor 和测试；
- 不提供任意字符串键值查询，防止 consumer 重新依赖 CANN 字段名。

## 4.2 可测试性

parser、字段映射、feature 构造、fingerprint 和内置资源查询均能使用合成 INI 测试，不依赖 CANN 或 NPU。真实配置只用于
授权后的兼容性和嵌入结果验证。

## 4.3 可移植性

loader 只依赖 LLVM/MLIR 的文件和字符串基础设施，不调用 CANN runtime。普通 x86_64、AArch64 和 macOS CPU 环境都能
加载 Profile 并生成源码。

## 4.4 可靠性、线程安全和确定性

Profile 构造完成后不可变。不同编译任务不共享可变状态，因此可以并行编译不同平台。所有输出顺序和
fingerprint 都使用显式排序，不能依赖 `DenseMap`、`StringMap` 或文件目录遍历顺序。

## 4.5 特性交叉影响

| 场景 | 适用性 | 结论 |
|---|---|---|
| SuperKernel Python 接口 | 不适用 | 不修改 Python API 或 wheel |
| SuperKernel C++/AOT 接口 | 不适用 | 不修改 `libascendsk.so` 或 AOT ABI |
| Autofuse 图优化 | 适用 | 后续 pass 只读取 Profile；本期不迁移图改写 |
| Autofuse Codegen/Backend | 适用 | 编译任务上下文和 capability provider 使用 Profile |
| AscendC API / Runtime | 部分适用 | 源码生成不调用 runtime；Kernel 继续由现有 `ascendc_compile.py` 和 `bisheng` 编译 |
| Python/C++ 混合绑定 | 不适用 | 本期不增加对外绑定 |
| 构建与打包 | 适用 | 受支持平台的 canonical INI 生成只读字节并链接进工具，不安装独立 Profile 文件 |
| 测试与覆盖率 | 适用 | 新增 loader UT、生成器测试、LIT 和无外部资源的交付 ST |
| 性能与日志 | 适用 | 每次编译解析一次；不在每 Op 查询时打印日志 |
| 兼容性 | 适用 | 新命令行输入尚未对外发布；不改变现有接口默认行为 |

# 5 性能

## 5.1 编译时长

内置平台和外部测试输入在每个编译任务初始化时只解析一个 INI。后续 capability 查询使用枚举和只读数据，不重新解析
文件。

## 5.2 执行性能

Target Profile 只影响编译期，不增加 Kernel 运行时调用、内存拷贝或日志。

## 5.3 内存和产物大小

每个编译任务保存一份 owning Profile。工具包含 46 个支持平台别名和 38 份 canonical INI，不嵌入完整 CANN
`platform_config` 目录。平台数量继续增长时优先复用 manifest 中的 canonical/alias 映射，不引入差分继承或运行时合并逻辑。

# 6 接口设计和使用方法

## 6.1 基础加载接口

基础功能提供 Target 选择、内置平台和外部 buffer 三个明确入口，三者构造相同的 owning `TargetProfile`：

```cpp
class TargetProfileLoader {
public:
  static FailureOr<TargetProfile>
  load(const TargetSelection &selection, raw_ostream &diagnostic);
  static FailureOr<TargetProfile>
  loadBuiltin(StringRef platform, raw_ostream &diagnostic);
  static FailureOr<TargetProfile>
  loadFromBuffer(MemoryBufferRef buffer, raw_ostream &diagnostic);
};
```

`TargetSelection` 已封装 `--target` 与 `--target-profile` 的二选一规则；`load(selection)` 只对显式外部路径执行普通文件
检查和读取，不搜索 CANN 安装目录。调用方必须保证 `MemoryBufferRef` 在 `loadFromBuffer` 返回前有效；返回的
`TargetProfile` 已复制字符串和容器，不保存 buffer 或文件句柄。`loadBuiltin` 不接受资源根目录，也不搜索环境变量。

## 6.2 Profile 查询接口

```cpp
class TargetProfile {
public:
  const TargetIdentity &getIdentity() const;
  const CoreTopology &getCoreTopology() const;

  bool hasMemory(MemorySpace space) const;
  FailureOr<MemorySpec>
  getMemorySpec(MemorySpace space) const;
  bool hasComputeUnit(AICoreUnit unit) const;
  FailureOr<ComputeUnitSpec>
  getComputeUnitSpec(AICoreUnit unit) const;

  FeatureState getFeatureState(TargetFeature feature) const;
  bool hasFeature(TargetFeature feature) const;

  FeatureState
  getExecutionSupport(const ExecutionSpec &spec) const;

  ArrayRef<MemoryResource> getMemoryResources() const;
  ArrayRef<ComputeAccess> getComputeAccesses() const;
  bool canAccess(AICoreUnit unit,
                 MemorySpace space,
                 AccessMode mode) const;

  ArrayRef<TransferEdge> getTransferEdges() const;
  bool hasDirectTransfer(
      MemorySpace source,
      MemorySpace destination) const;

  StringRef getFingerprint() const;
};
```

`hasFeature()` 只在状态为 `Available` 时返回 `true`。需要区分明确不支持和信息缺失的调用方必须使用
`getFeatureState()`。

资源直接按 `MemorySpace` 查询。归属由 `getMemoryOwner()` 的固定映射给出，不从 Profile 数据解析，也不由 consumer 猜测。

调用方不需要先调用 `hasMemory()` 再调用 `getMemorySpec()`，也不需要先调用 `hasComputeUnit()` 再调用
`getComputeUnitSpec()`。规格查询已经统一包含存在性语义：failure 表示资源或单元不存在；success 且字段为
`std::nullopt` 表示对象存在但该项规格未知。`hasMemory()` 和 `hasComputeUnit()` 只服务纯布尔存在性判断，避免调用方为
“存在且可查询”重复拼接条件。

`canAccess()` 只查询计算单元对资源的直接读写权限；`hasDirectTransfer()` 只查询两个资源之间是否存在一条直连边。
两个接口内部统一检查相关计算单元和资源存在性。Loader 又保证规范化图中不产生指向不存在对象的访问项或搬运边。
调用方不读取 `NpuArch`、intrinsic 名称或具体平台自行分支。

`getExecutionSupport()` 是 SIMD、SIMT、MemoryBased 和 RegBased 的唯一合法性查询入口，并在内部检查请求的 AICore
单元是否存在。调用方不得先检查 core 数再查询执行能力，也不得分别查询多个 feature 后自行组合。

## 6.3 产品模式使用示例

以下为当前 `af-opt` 已采用的初始化调用链。调用方先解析 Target 选择，再构造一份在 optimizer 调用期间保持有效的
只读 Profile。普通 pipeline 使用 `Optional`；启用 AFIRToAscendC 产品 pipeline 时使用 `Required`。

```cpp
TargetSelectionRequirement requirement =
    afirToAscendCPipelineEnabled
        ? TargetSelectionRequirement::Required
        : TargetSelectionRequirement::Optional;
auto selection = resolveTargetSelection(
    target, targetProfile,
    requirement, llvm::errs());
if (failed(selection)) {
  return failure();
}
if (selection->has_value()) {
  auto profile = TargetProfileLoader::load(selection->value(), llvm::errs());
  if (failed(profile)) {
    return failure();
  }
  // 持有 profile，直到本次 optimizer/pipeline 调用结束。
}
```

加载失败时，编译任务停止创建 pipeline。不能切换到默认平台，也不能继续使用部分 Profile。

## 6.4 Capability provider 使用示例

以下示例说明 provider 如何使用 implementation mapping 中的完整执行规格。Target Profile 只回答硬件是否支持
该组合；具体 AscendC API 的 dtype、operand 和 shape 条件仍由 provider 判断。

```cpp
FeatureState state =
    context.target.getExecutionSupport(mapping.executionSpec);
if (state == FeatureState::Unknown) {
  return CapabilitySupport::undetermined(
      CapabilityReason::TargetFeatureNotModeled);
}
if (state == FeatureState::Unavailable) {
  return CapabilitySupport::unsupported(
      CapabilityReason::TargetFeatureUnavailable);
}

return queryAscendCApiConstraints(op, request, context);
```

conversion pattern 使用同一份 `mapping.executionSpec` 写入 AscendC IR：

```cpp
if (failed(setExecutionSpec(targetOp, mapping.executionSpec))) {
  return rewriter.notifyMatchFailure(
      op, "invalid AscendC execution spec");
}
```

属性 adapter 提供成对接口：

```cpp
FailureOr<ExecutionSpec>
getExecutionSpec(Operation *op);

LogicalResult setExecutionSpec(
    Operation *op, const ExecutionSpec &spec);
```

setter 在写入前校验完整组合，禁止 SIMD 缺少 `instruction_form`，也禁止 SIMT 携带
`instruction_form`。emitter 可以校验属性，但不得根据平台或 implementation ID 补默认值。

AscendC API 路径不会查询 CCE intrinsic 清单。未来开始直接 lowering 到 CCE Intrinsic 时，再按真实查询需求设计并接入
相应指令模型。

## 6.5 schedule、memory planner 和 ATT 使用示例

这些 consumer 当前尚未迁移。下面只规定未来的使用方式，不新增 pass 接口。

```cpp
const CoreTopology &topology = target.getCoreTopology();
auto ub = target.getMemorySpec(MemorySpace::UB);
if (failed(ub)) {
  return emitScheduleFailure("Vector UB does not exist");
}
if (!ub->capacityBytes) {
  return emitScheduleFailure("Vector UB capacity is not modeled");
}

int64_t vectorCoreCount = topology.vectorCoreCount;
int64_t ubBudgetBytes = *ub->capacityBytes;

if (target.hasDirectTransfer(MemorySpace::L0C,
                             MemorySpace::UB)) {
  // A5 可直接搬运；planner 记录这一条候选边。
} else {
  // A2/A3 由 planner 在 TransferEdge 图中搜索 L0C -> GM -> UB。
}
```

schedule 使用 core 数、容量和 `ComputeAccess` 枚举合法方案；memory planner 在 `TransferEdge` 图中搜索路径；ATT 使用边上的
可选 `bytesPerCycle` 比较成本。三者都不能修改 Profile，路径搜索和成本选择也不写回 Profile。

## 6.6 外部测试文件和负向示例

```cpp
auto buffer = llvm::MemoryBuffer::getFile(testProfilePath);
if (!buffer) {
  return failure();
}
auto profile = TargetProfileLoader::loadFromBuffer(
    (*buffer)->getMemBufferRef(), llvm::errs());
```

文件不可读、core 数或搬运声明 section 等必需字段缺失、整数非法或字段矛盾时返回 failure。可选规格缺失时保留
`std::nullopt`，可选 feature 缺失时保留 `Unknown`；只有实际依赖该规格或 feature 的 consumer 才拒绝或返回
`undetermined`。

## 6.7 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|---|---|---|---|
| 接口说明 | 是否需要接口评审 | 是 | Target 字段和生命周期供多个 MLIR 模块使用 |
| 接口说明 | 是否需要补充文档 | 是 | 新增字段时同步更新 mapping 表和使用示例 |
| 接口兼容 | 行为是否兼容 | 是 | 新 CLI 不能改变未启用 Target pipeline 的现有行为 |
| 接口兼容 | ABI/API 是否兼容 | 否 | 本期只新增 MLIR 内部 C++ 接口 |
| 接口约束 | 约束不满足时是否清晰报错 | 是 | 使用稳定错误码和 section/key 定位 |
| 接口测试 | 是否需要独立接口用例 | 是 | loader、内置索引、accessor 和并行 session UT |

# 7 软件设计

## 7.1 关键技术

### 7.1.1 Target Profile 模块归属

Target Profile、内置 INI 资源和 `TargetProfileLoader` 由 Autofuse MLIR 的 `target/Ascend` 模块实现并维护。其他 pass 只通过
强类型 accessor 查询硬件事实，不能各自解析 INI 或维护字段映射。

Autofuse 内部只能维护一份规范化字段语义。内置资源与外部 Profile fixture 必须经过同一 loader，避免不同输入路径产生
不同单位、缺失行为或 fingerprint 顺序。

### 7.1.2 确定性

parser 使用 section/key 查找字段，但生成 fingerprint 和诊断列表前必须排序。平台目录不能通过无序遍历自动形成支持
列表；支持平台以 manifest 中的稳定顺序为准。

## 7.2 对子模块的修改

当前实现与后续接线点如下：

| 目录 | 修改内容 |
|---|---|
| `mlir/target/Ascend/` | 已新增 Profile、loader、TargetSelection、原始 INI、manifest 和构建期资源生成器 |
| `mlir/target/Ascend/` | 已新增 typed `NpuArch`、`ExecutionSpec`、固定 `MemorySpace` 归属、计算访问权限和直接搬运边 |
| `mlir/target/Ascend/` | 后续新增 Profile 到 PyAsc 内存枚举的 adapter |
| `mlir/tools/af-opt/` | 已接收 Target 选择并在 optimizer 生命周期内持有只读 Profile |
| `mlir/conversion/AFIRToAscendC/` | mapping 保存 execution spec；provider 查询 Profile；conversion 写入 AscendC 执行属性 |
| `mlir/externals/pyasc/` | 当前只作为 Ascend/EmitAsc dialect 和 emitter 依赖；本期不接入 PyAsc Python 编译工程 |
| `compiler/python/ascendc_compile.py` | 继续使用现有 `bisheng` 编译路径；产品 pipeline 接线时校验 `soc_version` 与 Profile 设备身份一致 |
| `mlir/test/ut/target/` | loader、模型、fingerprint 和线程安全 UT |
| `mlir/test/lit/target/` | CLI 互斥、诊断和 Profile 打印测试 |
| CMake | 将 loader 和 Capability 子模块接入 Target library |

本期不修改 `AFIRToAscendCQueue` demo，不迁移 schedule、ATT 或 memory planner。

## 7.3 流程与回滚边界

Target 模块使用合成 fixture 验证 parser，并使用 CANN 9.1.0 的受支持平台原始 INI 验证内置加载；不会复制完整
`platform_config` 目录。产品 `--target=<platform>` 的 CLI 加载已完成；AFIRToAscendC 产品 pipeline 已强制要求
Target，其他 pipeline 仍按自身是否依赖 Target 选择 `Optional` 或 `Required`。

回滚时可以删除 Target library、生成步骤和 CLI 接线，不影响现有 AFIR dialect、bridge 和 queue demo。不得让旧 pass 在
回滚后残留对 Profile 的强制依赖。

## 7.4 错误处理

### 7.4.1 系统错误

外部 Profile 文件不存在、不是普通文件或读取失败时，loader 返回 failure，并给出来源模式和安全处理后的路径。内置平台
未登记或生成资源不完整属于构建/初始化错误。诊断不得输出配置全文。

### 7.4.2 接口错误

诊断至少区分：

- `TargetSelectionConflict`：两种输入同时出现或都缺失；
- `TargetNameInvalid`：平台名包含非法字符；
- `TargetProfileNotFound`：内置平台未登记或外部文件不存在；
- `TargetProfileParseError`：INI 语法或整数解析失败；
- `TargetProfileMissingField`：必需字段缺失；
- `TargetProfileInvalidValue`：范围、单位或字段关系非法；
- `TargetProfileIdentityMismatch`：请求平台与 `SoC_version` 不一致；
- `TargetFeatureNotModeled`：consumer 查询的 feature 为 `Unknown`。
- `TargetExecutionSpecInvalid`：execution spec 字段组合非法；
- `TargetExecutionSpecUnavailable`：当前架构明确不支持该 execution spec。

# 8 安全与兼容性检查

## 8.1 安全检查

- 平台名在内置资源查询前校验字符集和长度；外部 Profile 只接受显式普通文件，并限制总大小、单行长度和字段数；
- 外部路径和 buffer identifier 进入诊断前转义控制字符，防止伪造多行日志；
- 所有整数在作为容量、数量和乘法输入前校验范围；
- fingerprint 和 bank 相关乘法使用 checked arithmetic；
- loader 不保存文件句柄或 parser buffer 引用；
- 不引入敏感信息、公网地址、runtime 句柄或可变全局对象；
- pass 不按芯片型号、平台名或框架名分支。

## 8.2 兼容性检查

本期不修改对外 Python/C++ API 或动态库 ABI。Target 模块内部 C++ 类型和查询接口发生变化，所有仓内调用方同步迁移；
这些类型当前未通过 Python binding 或安装头文件对外导出。已审计的内置 Profile 随 Target library 进入当前构建产物，
不增加安装目录。现有未加载 Target Profile 的 demo pipeline 保持原有行为；完整 Capability pipeline 启用后必须显式
选择 Target。

当前内置 INI 已完成人工许可证和来源审计。38 份 canonical INI 的 SHA-256 记录在 `Profiles/SHA256SUMS`，生成器 UT
检查清单与 Profile 目录完全一致并逐文件校验内容。构建需要覆盖首次构建、INI 变化后的增量重建、离线构建和
`--no-autofuse`；交付验证需要确认工具脱离源码 Profile 目录后仍能加载内置平台。

# 9 测试设计

## 9.1 测试边界

测试输入是合成 INI、已审计的内置 Profile 资源和 Target 选择参数。测试输出是生成的内置索引、规范化字段、feature
状态、fingerprint、稳定诊断和逐文件 SHA-256 完整性结果。测试不需要 CANN runtime 或 NPU。

## 9.2 测试用例

| 类别 | 关键测试项 | 测试方法 | 类型 |
|---|---|---|---|
| 字段映射 | identity、core、GM/L2/L1/L0A/B/C/UB、固定 owner、bank、register、edge rate | 合成 A2/A3 与 A5 fixture | gtest |
| 资源存在性 | GM/L2 始终存在；Cube/Vector local memory 随对应 core 存在 | 合成 Profile fixture | gtest |
| 规格未知 | 资源或单元存在但容量、block、频率等字段缺失时查询成功且字段为 `nullopt` | 合成 Profile fixture | gtest |
| core 必需字段 | `cube_core_cnt`、`vector_core_cnt` 缺失时 loader failure；值为 0 时对应资源、访问和边不存在 | 合成 Profile fixture | gtest |
| feature | BF16、InfNan、NDDMA 的 available/unavailable/unknown | 三组最小 fixture | gtest |
| 执行能力 | `NpuArch=2201` 支持 SIMD + MemoryBased，拒绝 SIMD + RegBased 和 SIMT | 组合查询 | gtest |
| 执行能力 | A5 支持 SIMD + MemoryBased、SIMD + RegBased 和 SIMT | 组合查询 | gtest |
| 架构解析 | `2201 -> V2201`、`3510 -> V3510`、缺失/未知值 -> `Unknown` | loader/profile fixture | gtest |
| 计算访问 | Cube 读 L0A/L0B、写 L0C；Vector 读写 UB；未声明访问返回 false | Profile/loader fixture | gtest |
| 统一查询 | `getExecutionSupport()` 在 Vector Core 不存在时直接返回 unavailable，provider 不重复判断 core 数 | Profile/provider fixture | gtest |
| 搬运闭世界 | intrinsic map section 缺失失败；不存在的资源不产生边；GM/UB/Vector 都存在时才补基线边 | 合成 Profile fixture | gtest |
| 跨单元访存 | A2/A3 只有 L0C→GM→UB，多跳可达但无直连；A5 存在 L0C→UB 直连 | 内置 Profile fixture | gtest |
| 速率负向 | `l0_c_to_ub_rate` 不能创建缺失直连，只能标注已声明边 | 合成 Profile fixture | gtest |
| 执行属性 | SIMD 两种 form 与 SIMT 往返；SIMD 缺 form、SIMT 带 form 失败 | attribute adapter UT | gtest |
| PyAsc 映射 | 六个可映射内存位置、L2/未知值失败 | adapter UT | gtest |
| feature 缺口 | NDDMA 无来源时保持 unknown | 负向断言 | gtest |
| 输入选择 | 两种输入互斥、都缺失、非法平台名 | CLI/LIT | LIT |
| 内置资源生成 | manifest 重复 alias、文件缺失、非法平台名、canonical/alias 顺序错误；生成字节与原始 INI 一致 | unittest/CMake |
| 支持范围 | 12 个 910 和全部 34 个 950 平台可加载；其他 910 平台返回 unknown | manifest + gtest | gtest |
| 两种来源一致性 | 同一份 INI 分别通过内置字节和外部 buffer 加载 | 规范化字段和 fingerprint 相同 | gtest |
| 解析异常 | 文件缺失、重复字段、非法整数、溢出、容量为零 | failure 和错误码断言 | gtest |
| 一致性 | 请求平台与 `SoC_version` 不匹配 | failure 断言 | gtest |
| 一致性 | selection/conversion 共用 Profile；IR 或 Profile 变化后重跑 selection | pipeline fixture | gtest/LIT |
| Add lowering | `ascendc.add_l2` 显式写入 Vector + SIMD + MemoryBased | conversion check | LIT |
| 编译目标一致性 | Profile 设备身份与现有编译入口的 `soc_version` 不一致 | 调用 `bisheng` 前失败 | pytest |
| 确定性 | section/key 顺序不同但事实相同 | fingerprint 相同 | gtest |
| 并发 | 两个 session 加载不同平台 | 无共享状态和串扰 | gtest |
| 内置资源交付 | 工具脱离源码 INI 和当前工作目录后仍可按平台名加载；INI 修改触发增量重建 | build dependency + package fixture | ST |
| 离线源码生成 | 无 CANN 环境加载内置 Profile 并生成 AscendC C++ | 隔离环境 | ST/E2E |

## 9.3 测试位置和命令

当前测试文件：

```text
mlir/test/ut/target/TargetProfileLoaderTest.cpp
mlir/test/ut/target/AscendTargetProfileTest.cpp
mlir/test/ut/target/TargetSelectionTest.cpp
mlir/test/ut/target/BuiltinTargetProfilesGeneratorTest.py
mlir/test/lit/target/target-profile-options.mlir
```

当前已提供可单独执行的 `check-ascend-target-profile` 和 Target CLI LIT。CLI LIT 覆盖 legacy、内置 Profile、外部
Profile、参数冲突、未知平台、非法平台名和文件缺失；loader UT 覆盖文件/buffer 总大小、单行长度、字段数和诊断转义
边界。2026-08-12 本轮验证结果为 Target Profile 43 个 C++ 测试和 7 个 Python 测试全部通过，Capability 49 个 C++
测试全部通过。构建命令限制为 `-j 8`。

# 10 验收标准

- 文档第 3.2.3 节中的首期字段均有唯一 mapping、单位、校验和 accessor；
- `--target` 与 `--target-profile` 互斥，错误可诊断且不回退默认平台；
- Profile 每次编译构造一次，之后只读，无可变全局当前平台；
- `hasMemory()`/`hasComputeUnit()` 只回答存在性；规格查询用 failure 表示对象不存在、用 `nullopt` 表示规格未知，调用方不重复组合判断；
- `cube_core_cnt` 和 `vector_core_cnt` 是必需字段且是计算单元存在性的唯一事实来源；
- Loader 输出的 `TransferEdge` 是闭世界结果，不产生引用不存在资源的边；`TargetFeature` 不包含独立 Fixpipe Boolean；
- `NpuArch=2201` 和 `NpuArch=3510` fixture 能正确区分 SIMD + MemoryBased、SIMD + RegBased 和 SIMT 组合；
- A2/A3 Profile 存在 `Cube.L0C -> GM` 和 `GM -> Vector.UB`，但不存在 `Cube.L0C -> Vector.UB`；A5 Profile 存在直连边；
- rate 字段只能附加到已声明的边，不能创造通路；
- AFIR lowering 创建 PyAsc Op 前完成 Capability 查询，conversion 把同一份 execution spec 写入目标 IR；
- `ascend.implementation` 以 kind 选择后端实现路径、以稳定 ID 选择路径内的具体实现，不增加 execution model 或
  instruction form；
- emitter 只读取 AscendC IR 中的执行属性，不按平台或 implementation ID 重新猜测；
- `MemorySpace` 到 PyAsc `Hardware` 通过显式 adapter 映射，L2 和未知值有负向诊断；
- NDDMA 在事实来源未确定时保持 unknown；
- AscendC API provider 不重复检查 CCE intrinsic；
- 同一规范化 Profile 在字段顺序变化后得到相同 fingerprint；
- 无 CANN 的普通 CPU 环境可以使用内置平台生成 AscendC C++；
- 请求 Kernel 二进制时继续调用现有 `bisheng` 编译路径，并在调用前校验 `soc_version` 与 Profile 设备身份一致；
- manifest 与 INI 生成只读内置索引并链接进交付工具；工具脱离源码 Profile 目录后仍能按平台名加载，生成的 Kernel
  产物不携带也不读取 Profile；
- 修改内置 Profile 需要重新构建工具，临时平台和临时测试可通过 `--target-profile=<file>` 使用外部文件。

## 10.1 分阶段开发门禁

Target 选择值对象、统一 parser/schema、owning Profile、合成 INI fixture、基础诊断、fingerprint、内置资源和
`af-opt` CLI 加载已经完成。当前只接入单位已经确认的 identity、core 和 memory 字段；未知可选字段继续保持未建模，
不猜测数值。

以下事项只阻塞对应的产品能力，不阻塞上述基础设施：

1. 真实内置平台：已完成首批 910/950 平台筛选、人工来源和许可证审计、逐文件 SHA-256 记录以及内置资源接入；
2. 执行模式：typed `NpuArch -> execution capability`、AscendC 执行属性和 Add L2 接线已完成；
3. RegBased 资源：确认 `vector_reg_width`、`predicate_reg_width`、`wide_reg_width` 的 CANN schema 单位后，增加对应
   bytes accessor；它们用于资源计算，不用于重新判断 RegBased 执行形态；
4. 产品 pipeline：`af-opt` Target 选择、Profile 生命周期和正式 AFIRToAscendC Required Target 接线已完成；
5. Kernel 二进制：将同一规范化 Target 传给现有 `soc_version` 入口，并在调用 `bisheng` 前完成一致性检查。

## 10.2 设计文档检查结果

- [x] 跨特性交叉影响：已逐项检查 Autofuse backend、构建交付、接口、测试和性能；
- [x] 编码红线：已检查路径输入、整数溢出、资源生命周期、平台分支、确定性和全局对象；
- [x] 测试设计：已覆盖资源存在性与规格未知、core 必需字段、执行能力统一判断、搬运闭世界、字段映射、异常和构建嵌入；
- [x] 性能评估：每次编译只加载一次 Profile，consumer 只进行只读查询；
- [x] 可读性：已区分当前事实、未来预留和实施前待确认项。
