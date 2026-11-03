# Ascend 指令 Capability 与 AFIR Lowering 开发方案

> 状态：D0～D2 公共基础设施、Target CLI 接线和首个正式 AscendC compute lowering 已完成；canonical AFIR Attr/Bridge/MemoryAllocate
> 接入属于 2026-08-13 AFIR 方言重构，CANN 源码生成、Kernel 编译和设备 E2E 仍单独验收
> 日期：2026-08-09
> 最终评审：2026-08-14
> 所属架构：[AFIR Op Capability 架构设计](../design/2026-08-05-afir-op-capability-architecture-design.md)

# 1 简介

## 1.1 目的

本文定义 AFIR Op Capability、Ascend 实现选择和标准 MLIR conversion 的开发方案。重点解决两个问题：

- AscendC API 等实现如何注册并提供能力查询；
- 如何在 buffer-form AFIR 上选择实现，并保证 conversion 只执行已选 pattern。

## 1.2 范围

本期设计包含：

- Ascend 指令实现注册、能力提供者（provider）和统一查询；
- buffer-form AFIR 上的实现选择属性；
- AFIR 到 AscendC 的标准 conversion pattern 接线；
- scalar、broadcast inline、inplace 和 resource 查询接口；
- UT、LIT 和后续 E2E 验证方案。

本文假设上游已经产生可供后端 lowering 的 buffer-form AFIR。AFIR tensor/memref 统一、ODS 定义、
`BufferizableOpInterface`、memory planning 和现有 `afir.bufferized.*` 的迁移由独立设计负责，不属于本需求。

本期不迁移 can-fuse、schedule、ATT 和 memory planner，只说明它们以后如何查询 Capability；不实现 CCE Intrinsic、
PTO-ISA 和计划废弃的 `AFIRToAscendCQueue` demo。

## 1.3 文档结构

- 第 2 章给出结论和约束；
- 第 3 章说明数据流及四项功能；
- 第 5、6 章给出接口、目录、生命周期和迁移步骤；
- 第 7～10 章说明安全、兼容、测试和验收。

# 2 总体概述

## 2.1 软件概述

### 2.1.1 项目介绍

AFIR 是 Autofuse 迁移到 MLIR 后的统一图 IR。本文接收上游 bufferization 产出的 buffer-form AFIR，在此基础上选择
Ascend 指令实现，并转换为具体后端 IR。上游如何产生 buffer-form AFIR，不在本文中设计。

### 2.1.2 产品环境介绍

生成 AFIR、执行 Capability 查询和生成 AscendC C++ 源码只依赖内置 `TargetProfile`，不要求本机安装 CANN。
将 AscendC C++ 编译为 Kernel 二进制仍需要 CANN Toolkit。

本文只新增 Autofuse MLIR 内部接口，不修改公共 Python/C++ API 或 ABI。

## 2.2 软件功能

目标编译链为：

```text
buffer-form AFIR
  -> Ascend implementation selection
       查询最终 buffer type、layout、alias 和 target
       写入 ascend.implementation
  -> AFIRToAscendC conversion
       一次性加载当前 pipeline 的全部已注册 patterns
       只有属性指定的 pattern 匹配成功
  -> AscendC dialect
  -> translation / emission
```

## 2.3 设计约束

- AFIR dialect 不依赖 AscendC、CCE Intrinsic 或 PTO-ISA 实现；
- Capability 只回答“能不能”，选择策略回答“选哪个”；
- 实现选择属性是可丢弃的 lowering plan，不是 AFIR 固有语义；
- conversion 使用标准 pass 级 `RewritePatternSet`，不为每个 Op 动态创建 pattern 集合；
- registry、provider、Target Profile 和 query 在一次编译中只读；
- 不使用 raw entry pointer、可变全局 registry 或静态初始化顺序；
- 不按 SoC 名称、Op 名称字符串白名单或 pattern 注册顺序决定结果；
- buffer size、alignment、shape 和索引运算必须校验范围和溢出。

## 2.4 假设和依赖关系

- `TargetProfile` 由独立 loader 构造并在 pipeline 生命周期内保持只读；
- 上游已经产生合法的 buffer-form AFIR；其具体 ODS、bufferization 和 memory planning 方案由独立设计定义；
- AFIR Op 属性能够提供 resource query 所需的 repeats、strides 和 vectorization 信息；
- `mlir::afir::Expression` 继续作为自有数据的符号资源大小值对象；
- 第一阶段只要求一个正式 compute lowering 纵向跑通，不把具体 Op 写成架构前提；
- 本文实现开始前，上游必须明确 buffer-form AFIR 的稳定输入约定。

## 2.5 当前代码事实

当前已实现并验证以下基础设施：

- owning `CapabilitySupport`/candidate result、冻结后只读的 `ImplementationRegistry` 和 `CapabilityQuery`；
- selection attribute/helper、`SelectImplementationPass`、多候选 policy 校验和稳定诊断；
- conversion preflight、`populateEnabledPatterns` 和 mock standard conversion；
- `check-afir-capability` 独立测试目标；
- `af-opt --target=<platform>` / `--target-profile=<file>` Target 选择和只读 Profile 生命周期接线。

当前 D0～D2 已接入正式的 `afir.add -> ascendc.add_l2` compute pattern、selection/conversion 纵向样例；
scalar、broadcast inline、inplace、resource query 和 `soc_version`/Kernel 编译一致性仍是后续门禁。

以上是当前代码事实。`ImplementationRegistry::registerImplementation()` 对 D2 lowering entry 仍要求 pattern registration
非空；未来允许“无 pattern entry、仅参与通用 Capability 查询”的 D3 entry 时，必须同步调整注册完整性校验、pattern population
和相关 UT，不能把该规划接口误读为当前行为。

# 3 需求分析与设计

## 3.1 整体介绍

本期在 buffer-form AFIR 上完成实现查询、选择和 conversion：

```text
buffer-form AFIR
  -> queryCapability（通用能力 consumer 按需调用）
  -> queryLowering（lowering selection 专用）
  -> 单候选直接选择；多候选交给显式 selection policy
  -> 写入 selected implementation 属性
  -> conversion pattern 按已选 implementation 改写
```

## 3.2 功能需求

### 3.2.1 功能需求 1：Ascend 指令实现注册与查询

1. 介绍

   一个实现候选表示某个 AFIR Op 到一种 Ascend 指令实现的完整路径。本期只注册 AscendC API；CCE Intrinsic 和
   PTO-ISA 只保留实现类型。

2. 输入

   每个注册项包含：

   - 源 AFIR Op 类型；
   - implementation kind 和稳定 ID；
   - 只读 Capability provider；
   - backend 持有的强类型 operand/result 映射（typed mapping）；
   - conversion pattern 注册器。当前 D0～D2 实现要求该注册器非空；D3 规划允许只参与通用 Capability 查询的 entry 不提供
     pattern，但不得进入 lowering candidates。

3. 处理

   typed builder 用同一个 mapping 构造 provider 和 pattern registration，避免两处分别维护参数位置。Registry 与一个
   `MLIRContext` 绑定，并从 `SourceOp::getOperationName()` 取得 Op 名。注册完成后执行 `freeze()`，随后只读。

   `CapabilityQuery::queryCapability()` 按当前 Op 查找注册项并调用 provider 的能力方法；
   `CapabilityQuery::queryLowering()` 只查找当前 pipeline 可转换的 entry，并调用 provider 的 lowering 方法。两个
   查询入口共享同一 registry、provider、typed mapping 和 `QueryContext`，但 lowering pattern 是否加载只影响
   `queryLowering()`，不影响通用能力查询。

   调用参数非法、指定 implementation 不属于当前 Op、registry 未冻结或 registry 损坏返回 failure；合法查询没有候选
   时返回空集合；unsupported 和 undetermined 都是正常候选状态，并携带稳定原因。

4. 输出

   Query 返回自有数据的 candidate 值，不包含 registry entry 指针。候选按 implementation kind 和稳定 ID 排序，不维护
   `stableOrder`。未指定 implementation 的通用能力查询返回每个实现的结果，不隐式折叠为单个 Boolean；指定 implementation
   时只返回该实现的结果。

### 3.2.2 功能需求 2：最终实现选择

1. 介绍

   最终选择接收上游产出的 buffer-form AFIR。此时 buffer type、layout、memory space、alignment 和 alias 已经确定。

2. 输入

   输入为 buffer-form AFIR Op、只读 Target Profile、冻结 registry 和可选 selection policy。

3. 处理

   `SelectImplementationPass` 调用 `queryLowering`，只在 supported 候选中选择：

   - 恰好一个 supported 候选时直接选择；
   - 多个候选时调用显式 policy；
   - 没有 policy 时不得按注册顺序隐式选择；
   - IR 已有选择属性时，先校验该实现仍在 registry 中且当前 Op 仍受支持，再按当前 0/1/多候选规则重新选择并覆盖属性；
     已有属性不能绕过多候选必须提供 policy 的约束。

   若 IR、Target Profile 或 registry 已经变化，pipeline 必须先清除旧属性再运行 selection。外部加载的 IR 若保留选择属性，
   selection 会先校验该属性，避免静默接受未知或失效的实现标识。

   选择结果写入可丢弃（discardable）属性 `ascend.implementation`。属性只保存 kind 和稳定 ID，不保存 entry、
   provider、pattern、成本或 API 拼写。Kind 和稳定 ID 必须使用可读字符串，禁止使用整数枚举值、哈希值或注册序号。
   稳定 ID 描述实现形式，例如 `add.tensor_scalar`，不能依赖当前 AscendC API 的具体拼写。

4. 输出

   输出仍为 buffer-form AFIR，仅增加可序列化的选择属性。未知 ID、属性格式错误或没有可用实现时 pass 失败并定位到
   当前 Op。

### 3.2.3 功能需求 3：标准 MLIR conversion

1. 介绍

   Conversion pass 在开始时一次性加载当前 registry 中的所有 enabled patterns。每个 pattern 固定绑定一个
   `ImplementationKey`。

2. 输入

   输入为带选择属性的 buffer-form AFIR、`TypeConverter`、Target Profile 和冻结 registry。

3. 处理

   Conversion pass 在调用 MLIR conversion driver 前，检查每个待 lowering Op 都有格式正确的选择属性，并确认 kind 和
   稳定 ID 存在于当前 registry。该检查只验证选择结果能否定位实现，不重新判断实现是否支持当前 Op。

   Pattern 按以下顺序执行：

   1. 校验 Op 满足约定的 buffer form；
   2. 读取已经过 pass 检查的 `ascend.implementation`；
   3. 属性与自身 implementation key 不一致时返回 match failure；
   4. 属性一致时按 typed mapping 创建 AscendC Op。

   多个 pattern 可以同时存在，但只有被选择的实现能够匹配，结果不依赖 pattern benefit 或注册顺序。
   Conversion 不再调用 provider。其前置条件是 selection pass 已在相同 Target Profile 和 registry 下成功执行，并且两个
   pass 之间没有修改 Capability 查询输入。加载带选择属性的外部 IR，或者中间 pass 修改了 dtype、shape、layout、alias
   等输入时，必须先重新运行 selection pass。

4. 输出

   成功时 AFIR Op 和选择属性一起被替换为 AscendC Op；conversion 结束后不得残留必须 lowering 的 buffer-form AFIR
   compute Op。

### 3.2.4 功能需求 4：Resource Capability

1. 介绍

   Provider 返回实现所需的临时 buffer、workspace、alignment、memory space、lifetime 和 execution resource。
   Capability 不负责决定是否分配或复用资源。

2. 输入

   Provider 从当前 Op 的 schedule/layout 属性、Target Profile 和 query context 读取数据。`ResourceQuery` 只说明请求的
   资源种类，不复制 repeats、strides、vectorized axis 或 vectorized strides。

3. 处理

   大小使用 `mlir::afir::Expression` 表达。静态数值和单位在构造时校验；信息不足返回 undetermined；明确不支持返回
   unsupported；非法属性、越界、溢出或除零返回 failure。

   AFIR Core 不再定义 `tmp_buffers`。Provider 是资源需求的事实来源，返回 owning 的 `ResourceRequirements`；在当前
   Schedule 尚未迁移到 MLIR 的阶段没有调用方写入 canonical AFIR。未来由 selection 在 `ResourceOwnerContext` 已提供后，将有
   SSA owner 的 temporary buffer 写入对应 `BufferDescAttr`，将无 SSA owner 的 temporary/workspace 按稳定 owner ordinal 写入
   `func.func` 的 `afir.workspace_plan`。`life_time_axis_id` 属于 owner/Schedule 事实，不由 provider 推断。

4. 输出

   Resource result 自有全部 constraints 和表达式，不返回临时 `ArrayRef`，不保存 Operation 或 Target Profile 引用。

## 3.3 非功能需求

### 3.3.1 可维护性

新增实现只修改对应 backend 的 mapping、provider、pattern 和单个注册入口。AFIR ODS、bridge registry 和 Target Profile
不维护 backend 支持列表。

### 3.3.2 可测试性

Registry/query、selection pass 和 conversion 分层测试。测试通过 IR fixture 构造输入，不复制 provider
支持矩阵。

### 3.3.3 可移植性

普通 CPU 环境可以加载内置 Target Profile、运行 pass 并生成 AscendC C++。Kernel 二进制编译和设备 E2E 才依赖 CANN
与 Ascend 设备。

### 3.3.4 可靠性

Registry 和 provider 在 pass 运行期间只读；不同 `MLIRContext` 使用不同 registry；候选排序稳定；查询结果自有数据；
不存在跨 registry raw pointer。

### 3.3.5 特性交叉影响

| 场景 | 适用性 | 结论 |
|---|---|---|
| SuperKernel Python 接口 | 不适用 | 不修改 Python 包和 wheel |
| SuperKernel C++/AOT 接口 | 不适用 | 不修改 `libascendsk.so` 和 AOT ABI |
| Autofuse 图优化 | 适用 | 后续 consumer 可查询 Capability，本期不迁移 pass |
| Autofuse Codegen/Backend | 适用 | 新增 selection 和 conversion 接线；buffer-form AFIR 由上游提供 |
| AscendC API / Runtime | 适用 | 生成 AscendC IR；本期不新增 runtime 调用 |
| Python/C++ 混合绑定 | 不适用 | 不修改 pyautofuse |
| 构建与打包 | 适用 | 新增 MLIR 内部库和测试目标，不改变公共包接口 |
| 测试与覆盖率 | 适用 | 增加 gtest、LIT 和后续 AscendC E2E |
| 性能与日志 | 适用 | Selection 对每个 Op 查询一次候选，不增加默认逐 Op 日志 |
| 兼容性 | 适用 | 新路径不改变 legacy 对外行为；bufferization 迁移不属于本需求 |

# 4 性能

## 4.1 编译时长

Registry 按 `OperationName` 建索引，单次查询复杂度为查找成本加当前 Op 候选数。Selection 和 conversion 各遍历一次待
lowering Op；第一阶段不增加跨 pass cache。

## 4.2 执行性能

本方案只改变编译期 IR 和代码生成，不增加 Kernel runtime 调用。最终 Kernel 性能由后续实现选择 policy 和 E2E 验证。

## 4.3 内存和产物大小

每次编译只持有一份 Target Profile、registry 和少量 pattern 对象。Candidate 和选择属性只保存小型值；不保存 API
源码、全 backend Op 字符串表或跨 pass `Operation *` cache。

# 5 接口设计

## 5.1 新增/修改接口描述

以下均为规划中的 MLIR 内部接口。

以下接口均位于 `mlir::ascend::instruction` 命名空间。命名空间表达 Ascend NPU 指令这一层归属，类型名不再重复
`Ascend`、`Instruction` 或 `AFIR` 前缀。公共接口保留完整单词，不使用 `instr`、`Impl` 等缩写。为突出接口主体，以下
声明省略命名空间外壳。

```cpp
enum class ImplementationKind {
  AscendC,
  CCEIntrinsic,
  PTOISA,
};

struct ImplementationId {
  std::string value;
};

struct ImplementationKey {
  OperationName sourceOp;
  ImplementationKind kind;
  ImplementationId id;
};

enum class SupportState {
  Supported,
  Unsupported,
  Undetermined,
};

enum class CapabilityReason {
  None,
  QueryNotImplemented,
  UnsupportedType,
  UnsupportedShape,
  UnsupportedTarget,
  TargetFeatureNotModeled,
  TargetFeatureUnavailable,
  MissingInformation,
  ResourceExpressionError,
};

enum class ValuePositionKind {
  Operand,
  Result,
};

struct ValuePosition {
  ValuePositionKind kind;
  uint32_t index;
};

struct ScalarOperandConstraint {
  SmallVector<uint32_t, 2> allowedOperandIndices;
  bool allowsAllScalar;
};

struct TypeConstraint {
  ValuePosition value;
  SmallVector<Type, 4> allowedTypes;
};

struct RankConstraint {
  ValuePosition value;
  int64_t minimumRank;
  int64_t maximumRank;
};

enum class ShapeRelation {
  Equal,
  BroadcastCompatible,
};

struct ShapeConstraint {
  ValuePosition lhs;
  ValuePosition rhs;
  ShapeRelation relation;
};

struct AlignmentConstraint {
  ValuePosition value;
  uint64_t alignmentBytes;
};

struct TargetFeatureConstraint {
  TargetFeature feature;
};

enum class AliasAccessMode {
  ReadOnlyInput,
  ReadWriteInput,
};

struct AliasConstraint {
  uint32_t inputOperandIndex;
  uint32_t outputResultIndex;
  AliasAccessMode accessMode;
};

using CapabilityConstraint = std::variant<
    ScalarOperandConstraint, TypeConstraint, RankConstraint,
    ShapeConstraint, AlignmentConstraint, TargetFeatureConstraint,
    AliasConstraint>;

struct CapabilitySupport {
  SupportState state;
  CapabilityReason reason;
  SmallVector<CapabilityConstraint, 4> constraints;
};

struct QueryContext {
  const TargetProfile &target;
};

struct Candidate {
  ImplementationKey key;
  CapabilitySupport support;
};

class Candidates {
 public:
  Candidates() = default;
  explicit Candidates(SmallVector<Candidate, 2> candidates);

  ArrayRef<Candidate> getAll() const;
  SmallVector<Candidate, 2> getSupported() const;
  bool hasUndetermined() const;

 private:
  SmallVector<Candidate, 2> candidates_;
};

enum class ResourceLifetime {
  Operation,
  Region,
  Function,
};

struct ExecutionResourceRequirement {
  ExecutionUnit unit;
  uint32_t count;
};

struct TemporaryBufferRequirement {
  mlir::afir::Expression sizeInBytes;
  uint64_t alignmentBytes;
  MemorySpace memorySpace;
  ResourceLifetime lifetime;
};

struct WorkspaceRequirement {
  mlir::afir::Expression sizeInBytes;
  uint64_t alignmentBytes;
  MemorySpace memorySpace;
};

struct ResourceRequirements {
  SmallVector<TemporaryBufferRequirement, 2> temporaryBuffers;
  std::optional<WorkspaceRequirement> workspace;
  SmallVector<ExecutionResourceRequirement, 2> executionResources;
};

struct ResourceSupport {
  CapabilitySupport support;
  std::optional<ResourceRequirements> requirements;
};

struct ResourceCandidate {
  ImplementationKey key;
  ResourceSupport resource;
};

using ResourceCandidates = SmallVector<ResourceCandidate, 2>;
```

`CapabilityReason` 使用稳定 enum；`CapabilityConstraint` 是封闭的 typed variant，不使用任意字符串作为逻辑输入。
`ValuePosition` 的下标在查询入口校验；rank 范围两端均包含；alignment 和所有资源大小的单位都是 bytes；执行资源数量的
单位是对应 execution unit 的个数。`ShapeRelation` 只表达无额外参数的二元关系：相等和 broadcast-compatible；可分解的
多输入关系使用多条 `ShapeConstraint`。Reduce、Concat、Split、Gather、Transpose 和 Matmul 等参数化或 N 元关系不扩展为
无参数枚举值，也不增加任意表达式字符串。相关 Op 固有语义由 dialect verifier 或 shape inference 保证，provider 负责检查
具体 implementation 额外施加的限制；只有真实 provider 和 consumer 都需要机器可读约束时，才增加对应的 owning typed
constraint，并同时规定静态参数规范化、动态参数错误语义和 ValuePosition 范围校验。

`CapabilitySupport` 为 `Supported` 时 reason 必须是 `None`；`Unsupported` 和 `Undetermined` 必须携带非空 reason。
Target feature 为 `Unknown` 时返回 `Undetermined + TargetFeatureNotModeled`，明确为 `Unavailable` 时返回
`Unsupported + TargetFeatureUnavailable`。
`ResourceSupport` 为 supported 时必须携带 requirements，其他状态不得携带。Workspace 的 size、alignment 和 memory
space 作为一个整体出现，不能产生“没有 workspace 但存在 workspace alignment”的状态。

`CapabilitySupport` 的 `SmallVector`、约束中的索引集合和 resource 容器都由结果对象自己拥有。`mlir::Type` 和
`OperationName` 是 `MLIRContext` 管理的不可变句柄，因此结果可以 copy/move，但不能在所属 `MLIRContext` 销毁后继续使用。
`Expression` 自己拥有表达式节点。结果不保存 provider 的临时 `ArrayRef`、`Operation *` 或 Target Profile 引用。
Capability constraints 不写入 IR；只有最终选择的 implementation kind/ID 序列化到选择属性。统一诊断函数负责把 typed
constraint 转成文本。

`Candidates` 自己拥有全部 candidate。`getAll()` 返回的只读 view 只在 `Candidates` 存活期间有效；`getSupported()` 返回
owning 值，避免过滤结果引用内部临时存储。合法查询没有注册候选时返回成功的空 `Candidates`。只有调用参数非法、registry
未冻结或内部状态损坏才返回 `FailureOr` failure；“无候选”不对应 `CapabilityReason`。

通用 Capability 查询和 lowering 查询使用不同的结果语义。通用查询在未指定 implementation 时返回所有已注册实现的
逐实现结果；指定 implementation 时只返回该实现的结果。它不把多个实现隐式折叠成一个 Boolean 或三态。调用方如果
需要“任意实现支持”或“所有实现支持”，必须在自己的策略层显式聚合。Lowering 查询仍然返回当前 pipeline 可转换的
`Candidates`，并且只包含带 conversion pattern 的 entry。

查询请求携带与能力类型匹配的结构化 payload，不能把 axis、stride、alias pair 或 scalar operand mask 编码为字符串：

```cpp
enum class CapabilityKind {
  ScalarInput,
  BroadcastInline,
  Inplace,
};

struct ScalarInputQuery {
  llvm::SmallVector<uint32_t, 2> scalarOperandIndices;
  bool allScalar = false;
};

struct BroadcastInlineQuery {
  // 由 BroadcastPatternAnalysis 生成的结构化 pattern。
  BroadcastPattern pattern;
};

struct InplaceQuery {
  uint32_t inputOperandIndex;
  uint32_t outputResultIndex;
  AliasAccessMode accessMode;
};

using CapabilityQueryDetail =
    std::variant<ScalarInputQuery, BroadcastInlineQuery, InplaceQuery>;

struct CapabilityQueryRequest {
  CapabilityKind kind;
  Operation *op;
  std::optional<ImplementationKey> implementation;
  CapabilityQueryDetail detail;
};
```

实现时必须校验 `kind` 与 `detail` 的 variant 类型一致，并校验所有 operand/result 下标。若请求还缺少动态 shape、layout
或 alias 分析结果，provider 返回 `Undetermined`；请求结构本身非法则由 query 入口返回 `FailureOr` failure。

Provider 接口：

```cpp
class CapabilityProvider {
 public:
  virtual ~CapabilityProvider() = default;

  virtual CapabilitySupport queryLowering(
      Operation *op, const QueryContext &context) const = 0;

  virtual CapabilitySupport queryScalarInput(
      Operation *op, const ScalarInputQuery &query,
      const QueryContext &context) const;

  virtual CapabilitySupport queryBroadcastInline(
      Operation *op, const BroadcastInlineQuery &query,
      const QueryContext &context) const;

  virtual CapabilitySupport queryInplace(
      Operation *op, const InplaceQuery &query,
      const QueryContext &context) const;

  virtual FailureOr<ResourceSupport> queryResource(
      Operation *op, const ResourceQuery &query,
      const QueryContext &context) const;
};
```

未覆盖的可选查询默认返回 `Unsupported` 和 `QueryNotImplemented`，不能默认支持。外部调用方不能直接持有或调用具体
provider；所有能力查询必须通过 `CapabilityQuery` facade 完成。

Registry 和 query：

```cpp
class ImplementationRegistry {
 public:
  explicit ImplementationRegistry(MLIRContext &context);

  template <typename SourceOp, typename Mapping,
            typename Provider, typename Pattern>
  LogicalResult registerImplementation(
      ImplementationKind kind,
      ImplementationId id, Mapping mapping);

  LogicalResult freeze();
  bool isFrozen() const;
  bool contains(const ImplementationKey &key) const;
};

class CapabilityQuery {
 public:
  explicit CapabilityQuery(const ImplementationRegistry &registry);

  FailureOr<Candidates> queryCapability(
      const CapabilityQueryRequest &request,
      const QueryContext &context) const;

  FailureOr<Candidates> queryLowering(
      Operation *op, const QueryContext &context) const;
};
```

`queryCapability()` 的 `implementation` 为空表示查询当前 registry 中该 Op 的所有实现；有值时必须验证 key 的
`sourceOp` 与请求 Op 一致，并只查询该 entry。`queryLowering()` 只面向 selection，不承担通用 scalar、broadcast inline
或 inplace 判断。Resource 查询后续沿用同一 facade，但其返回类型仍为 `ResourceCandidates`，不与普通 Capability 结果
混用。

Typed registration builder 创建一份只读 `Mapping`，并用它构造 `Provider` 和 `Pattern` registration；调用方不能再
单独传入一个可能使用其他 mapping 的 provider。`freeze()` 校验 source Op、kind、ID、provider 和 pattern registration
完整且 key 唯一。`contains()` 只在冻结后用于校验选择结果能否定位 enabled entry，不暴露 entry、provider 或 pattern
地址。

Registry 内部 entry 的等价结构如下。它不是公共查询接口，但实现时字段不能少于这些内容：

```cpp
struct PatternBuildContext;

struct RegistrationEntry {
  ImplementationKey key;
  std::shared_ptr<const void> mappingOwner;
  std::unique_ptr<const CapabilityProvider> provider;
  std::function<void(RewritePatternSet &,
                     const PatternBuildContext &)> addPattern;
};
```

`registerImplementation<SourceOp, Mapping, Provider, Pattern>()` 从 `SourceOp::getOperationName()` 构造 `key`，把传入的
`Mapping` 放入一个 `shared_ptr<const Mapping>`。Provider 持有这份 typed mapping；`addPattern` 闭包也捕获同一个 typed
mapping，并在创建 Pattern 时显式传入 `key`。`mappingOwner` 只负责类型擦除后的生命周期，不向调用方暴露或参与查询。
因此 provider 与 pattern 不可能由注册调用方分别传入两份 mapping。

`addPattern` 的参数就是 `RewritePatternSet` 和后文定义的 `PatternBuildContext`。`addPattern` 可以为空，仅表示该 entry
只参与通用 Capability 查询而不参与 lowering；如果 entry 被声明为 lowering implementation，则 `freeze()` 必须拒绝空的
pattern registration。Pattern 构造后固定保存自己的 `ImplementationKey`；匹配时只比较选择属性，不根据 entry 地址、
pattern benefit 或注册顺序选择实现。Registry 按 key 排序并在 `freeze()` 时拒绝重复 key 或空 provider。Candidate 只复制
key 和 support result，不保存 entry 地址。实现可以用等价的私有 type-erasure 类替代 `shared_ptr<const void>`，但所有权和
绑定关系不得改变。

Selection policy 的最小接口只处理多个 supported 候选：

```cpp
class SelectionPolicy {
 public:
  virtual ~SelectionPolicy() = default;

  virtual FailureOr<ImplementationKey> choose(
      Operation *op, ArrayRef<Candidate> supported,
      const QueryContext &context) const = 0;
};

FailureOr<ImplementationKey> selectImplementation(
    Operation *op, const Candidates &candidates,
    const QueryContext &context, const SelectionPolicy *policy);
```

`selectImplementation()` 对 0 个 supported 候选返回 failure，对 1 个候选直接返回其 key；只有候选数大于 1 时才调用
`policy->choose()`。多候选但 `policy == nullptr` 时返回 failure。Helper 必须校验 policy 返回的 key 属于传入的 supported
集合，不能接受 registry 中的其他实现，也不能使用注册顺序兜底。

选择属性只通过 helper 读写：

```cpp
FailureOr<ImplementationKey> getSelectedImplementation(Operation *op);

LogicalResult setSelectedImplementation(
    Operation *op, const ImplementationKey &selected);

void clearSelectedImplementation(Operation *op);

LogicalResult validateSelectedImplementations(
    Operation *root, const ImplementationRegistry &registry,
    function_ref<bool(Operation *)> requiresSelection);
```

属性名固定为 `ascend.implementation`，值使用 `DictionaryAttr` 保存经过校验的 kind 和 ID。Helper 通过
discardable attribute API 读写该属性；Source Op 从属性所在 Operation 取得。IR dump 应能直接看出实现类型和实现形式。
`validateSelectedImplementations()` 在 conversion driver 启动前遍历本次 conversion target 要求 lowering 的
buffer-form AFIR compute Op，检查属性存在、格式正确，并通过 `registry.contains()` 确认 kind 和 ID 定位当前 Op 的 enabled
entry。缺少属性、未知 kind/ID 或 source Op 不匹配时，helper 直接在当前 Op 产生诊断并返回 failure。
属性片段例如：

```mlir
"ascend.implementation" = {
  kind = "ascendc_api",
  id = "add.tensor_scalar"
}
```

`kind` 使用 `ascendc_api`、`cce_intrinsic`、`pto_isa` 等稳定名称。稳定 ID 在同一个 source Op 和 kind 内唯一；重命名
稳定 ID 会使已保存的中间 IR 失效，因此不能因类名、文件名或 API 拼写调整而变化。诊断需要更多信息时，根据 kind 和
稳定 ID 查询 registry，不向属性复制 provider、pattern 或 API 信息。未来如果 Ascend target dialect 定义专用 typed
attribute，可以提供等价的自定义打印形式；本期仍使用 `DictionaryAttr`。

Pattern 构造所需依赖显式传入：

```cpp
struct PatternBuildContext {
  TypeConverter &typeConverter;
  const TargetProfile &target;
};

void populateEnabledPatterns(
    RewritePatternSet &patterns,
    const ImplementationRegistry &registry,
    const PatternBuildContext &context);
```

`PatternBuildContext::target` 只用于构造后端 IR 所需的目标参数，不能在 conversion pattern 中重新判断 implementation
是否合法。实现合法性只由 selection 阶段的 provider 查询决定。

## 5.2 使用方法

产品 pipeline 构造一份 registry，并把同一个只读实例传给 selection 和 conversion pass：

```cpp
auto registry = std::make_shared<
    mlir::ascend::instruction::ImplementationRegistry>(mlirContext);
if (failed(registerAscendCImplementations(*registry)) ||
    failed(registry->freeze())) {
  return failure();
}

// 前置条件：上游 pipeline 已经产生合法的 buffer-form AFIR。
passManager.addPass(
    mlir::ascend::instruction::createSelectImplementationPass(
        registry, targetProfile, selectionPolicy));
passManager.addPass(createConvertAFIRToAscendCPass(
    registry, targetProfile));
```

Selection pass 的正常路径。`selectionPolicy` 是可空指针；单候选路径不依赖 policy：

```cpp
auto candidates = capabilities.queryLowering(op, queryContext);
if (failed(candidates)) {
  return signalPassFailure();
}

auto selected = selectImplementation(
    op, *candidates, queryContext, selectionPolicy);
if (failed(selected) ||
    failed(setSelectedImplementation(op, *selected))) {
  return signalPassFailure();
}
```

Conversion pass 先执行属性预检，再接入标准 MLIR dialect conversion：

```cpp
ConversionTarget target(context);
configureAFIRToAscendCConversionTarget(target);

RewritePatternSet patterns(context);
populateEnabledPatterns(patterns, registry, patternBuildContext);

if (failed(validateSelectedImplementations(
        root, registry, requiresAscendLowering)) ||
    failed(applyFullConversion(root, target, std::move(patterns)))) {
  return signalPassFailure();
}
```

`configureAFIRToAscendCConversionTarget()` 将 AscendC dialect 和保留的非 compute Op 标为 legal，将本次必须 lowering 的
buffer-form AFIR compute Op 标为 illegal。属性预检与 `ConversionTarget` 使用同一组待 lowering Op 定义，不能各自维护
字符串白名单。两者调用 conversion 模块内同一个 typed Op 分类 helper；该 helper 不是 Capability 公共接口。这样未知 ID
在进入 driver 前产生定点诊断；driver 只负责验证 illegal Op 是否都被选中 pattern 改写。

Conversion pattern 只处理属性指定的实现：

```cpp
auto selected = getSelectedImplementation(op);
if (failed(selected) || *selected != implementationKey) {
  return rewriter.notifyMatchFailure(
      op, "another Ascend implementation was selected");
}

return lowerToAscendC(op, adaptor, rewriter);
```

该 pattern 不重新查询 Capability。Pipeline 必须保证 selection 成功后直接进入 conversion；如果输入 IR 中已有选择属性，
仍需先运行 selection 对属性进行校验或刷新。

负向行为：

- buffer-form Op 缺少选择属性：conversion 预检在当前 Op diagnostic failure；
- 属性 kind/ID 未注册或 source Op 不匹配：conversion 预检在当前 Op diagnostic failure；
- 没有 supported 候选：selection failure；
- 多个候选但没有 selection policy：selection failure；
- provider 返回 unsupported/undetermined：selection 不得选择该实现；
- pattern 与属性 ID 不一致：普通 match failure，不改写 IR。

## 5.3 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|---|---|---|---|
| 接口说明 | 是否需要接口评审 | 是 | registry、provider、选择属性和 pass |
| 接口说明 | 是否需要补充文档 | 是 | 架构文档、开发方案和使用示例 |
| 接口兼容 | 行为是否兼容 | 是 | 新增内部 lowering 路径，legacy 路径保持不变 |
| 接口兼容 | ABI/API 是否兼容 | 否（对外） | 只新增 MLIR 内部接口 |
| 接口约束 | 是否清晰报错 | 是 | 非法 buffer form、未知 ID、无候选和失效选择属性 |
| 接口测试 | 是否需要独立用例 | 是 | gtest、LIT 和后续 E2E |

# 6 软件设计

## 6.1 关键数据结构

- Registry 由 compilation pipeline 持有，与一个 `MLIRContext` 绑定；`freeze()` 后只读；
- Provider、mapping 和 pattern registration 由 registry entry 拥有，生命周期覆盖两个 pass；
- Target Profile、registry 和 selection policy 由 pipeline 以只读 session 对象持有，pass 不保存临时对象引用；
- Candidate、Capability result 和 resource result 都自有数据；
- 选择属性由 `MLIRContext` 管理，不保存 C++ 地址；
- `QueryContext` 只在同步查询期间借用 Target Profile，不存入结果。

## 6.2 关键技术和算法

### 6.2.1 候选排序

Registry 按 source Op 建索引。查询结果按 `kind + stable ID` 排序，保证多次运行一致。Selection policy 不使用容器遍历
顺序。

### 6.2.2 Capability 与 lowering 查询

通用 consumer 通过 `CapabilityQuery::queryCapability()` 查询 scalar、broadcast inline 或 inplace。请求不指定
implementation 时，query 对当前 Op 的全部 registered provider 分别求值；指定 implementation 时只调用对应 provider。
结果保留 implementation key、状态、reason 和 typed constraints，调用方负责显式聚合多个实现的结果。

`LoweringQuery` 仍通过 `CapabilityQuery::queryLowering()` 提供 lowering 专用候选。它只遍历当前 pipeline 带 conversion
pattern 的 entry，并调用 provider 的 `queryLowering()`。Selection 只消费 lowering 结果并写入选择属性；Conversion 根据选择属性
找到绑定同一 registry entry 的 pattern，不再查询 provider。这样既避免重复计算，也避免 pattern 再实现一份 dtype、shape 或
target 判断。

### 6.2.3 属性失效

最终选择发生在上游产生 buffer-form AFIR 后。Selection 与 conversion 之间不安排会改变 dtype、shape、layout、alias
或相关属性的 pass。若以后插入此类 pass，必须清除选择属性并重新运行 selection。外部加载的 IR 即使带有选择属性，
也必须先运行 selection；conversion 不负责判断属性是否因 IR 或 Target Profile 变化而失效。

## 6.3 流程设计

```text
构造 Target Profile 和 registry
  -> 注册 AscendC implementations
  -> freeze registry
  -> consumer 按需调用 queryCapability
  -> 接收上游 buffer-form AFIR
  -> 校验约定的 buffer form
  -> queryLowering
  -> implementation selection
  -> 写入 discardable attribute
  -> populate all enabled patterns
  -> 预检每个待 lowering Op 的选择属性和 registry key
  -> configure ConversionTarget
  -> applyFullConversion
```

异常路径不修改 Capability 事实。Selection pass 应先完成当前 Op 校验再写属性；缺失或未知选择属性由 conversion
预检直接定位，pattern 改写失败和残留 illegal Op 由 MLIR conversion driver 诊断。

## 6.4 对子模块的修改

```text
mlir/target/Ascend/
  Capability/                   # registry、query、result、resource types
  Transforms/                   # implementation selection pass 和属性 helper

mlir/conversion/AFIRToAscendC/
  Capability/                   # AscendC providers 和 typed mappings
  Patterns/                     # 标准 AFIR -> AscendC patterns
  RegisterAscendCImplementations.*
```

`AFIRToCCEIntrinsic` 和 `AFIRToPTOISA` 本期不创建。未来直接 lowering 到对应 backend IR 时再增加并列目录。

## 6.5 分阶段实施与回滚边界

为避免与总体架构第 6.6 节的 P0～P4 阶段混淆，本文使用 D0～D3 表示本开发方案内部里程碑：

1. D0（已完成）：实现 owning result、registry、`CapabilityProvider::queryLowering` 和
   `CapabilityQuery::queryLowering`，使用 mock implementation 做 UT；
2. D1（已完成）：实现 selection 属性、selection pass、conversion preflight 和 mock pattern 的标准 conversion，覆盖
   单候选、多候选和未知 ID；
3. D2（已完成）：接入首个正式 `afir.add -> ascendc.add_l2` AscendC compute pattern，完成 selection/conversion
   纵向样例和 AFIR/AscendC IR LIT；源码生成、CANN 编译和设备 E2E 仍是独立验收门禁；
4. D3（待开发）：先实现 `CapabilityKind`、类型化 `CapabilityQueryRequest`、逐 implementation 的
   `CapabilityQuery::queryCapability()` 和 provider 默认能力接口，再以 `afir.add` 迁移 scalar、broadcast inline、inplace
   和 resource 查询；能力 registry 不依赖 conversion pattern，`queryLowering()` 只筛选可转换 entry。

D0～D1 属于总体架构 P1 的公共基础设施，D2～D3 按实际能力场景进入总体架构 P2。总体架构 P0 的事实和门禁仍使用原编号，
不由本文重新定义。

每阶段独立提交。任一阶段失败时只回滚该阶段，不修改 legacy optimize/codegen 行为，也不使用 queue demo 作为 fallback。

## 6.6 错误处理

### 6.6.1 系统错误

内存分配失败由标准容器/MLIR 错误路径处理；Profile、registry 或 pattern 初始化失败时不启动相关 pass。不得依赖全局对象
析构释放跨动态库资源。

### 6.6.2 接口错误

非法 operand/result index、错误 buffer form、未知 implementation、重复注册、未冻结 registry 和表达式错误返回
failure 并产生 MLIR diagnostic。合法但不支持返回 structured support result，不打印默认逐 Op 日志。

# 7 安全检查

- 校验所有 operand/result index、mask、shape、stride、alignment 和 memory space；
- buffer 和 workspace 大小使用 checked arithmetic，拒绝负值、溢出和除零；
- 不保存 entry、Operation 或 Target Profile 的悬空指针；
- 图改写同时保持数据边和控制边；
- 候选和 pattern 选择不依赖无序容器或指针地址；
- 不硬编码 SoC 型号、账号、路径或非开放 runtime 接口；
- 本期不新增 runtime 资源，不涉及异步内存生命周期。

# 8 兼容性检查

- 不修改公共 Python/C++ API、ABI、脚本和已有交付包接口；
- 上游 bufferization 及 `afir.bufferized.*` 迁移由独立设计处理，不在本需求中修改；
- bridge registry 继续只负责过渡期 AFIR 构造，不保存 Capability；
- 未进入本 pipeline 的 legacy optimize/codegen 行为不变；
- Target Profile 和 selection 属性使用版本稳定的 kind/ID，未知值明确失败。

# 9 测试设计

## 9.1 测试边界

- gtest：registry、provider、query、自有数据的 result、selection helper 和多 context；
- LIT：buffer-form 输入校验、selection diagnostic 和 conversion；
- E2E：正式 pattern 接入后验证 AscendC C++；Kernel 二进制和设备执行在 CANN 环境验证。

## 9.2 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 类型 |
|---|---|---|---|
| 功能 | 合法查询无注册候选时返回成功的空 `Candidates` | empty registry fixture | gtest |
| 通用 Capability | 未指定 implementation 时返回逐实现结果，不隐式折叠三态 | multi-implementation provider fixture | gtest |
| 通用 Capability | 指定 implementation 时校验 source Op、key 和当前 context | keyed query fixture | gtest |
| 通用 Capability | `kind` 与类型化 request payload 不匹配时返回 failure | malformed request fixture | gtest |
| 能力分层 | 不带 conversion pattern 的 entry 可参与通用 Capability，但不进入 lowering candidates | registry fixture | gtest |
| 能力状态 | Supported/Unsupported/Undetermined 分别保留 reason 和 typed constraints | provider fixture | gtest |
| 功能 | `getAll()` view 生命周期和 `getSupported()` owning copy | result copy/move fixture | gtest |
| 功能 | 单候选选择并写入属性 | mock provider | gtest/LIT |
| 功能 | 多 pattern 只有已选实现匹配 | conversion fixture | LIT |
| 异常 | 输入不满足约定的 buffer form | verifier/conversion diagnostic | LIT |
| 异常 | 重复注册、未冻结、未知 ID | registry/selection/conversion preflight fixture | gtest/LIT |
| 异常 | 多候选无 policy、policy 返回非候选 key | selection fixture | gtest/LIT |
| 异常 | unsupported、undetermined、失效或未知选择属性 | provider/selection fixture | gtest/LIT |
| Resource | 动静态 Expression、溢出、缺失 layout | AFIR fixture | gtest/LIT |
| 确定性 | 不同注册顺序产生相同候选顺序 | repeated fixture | gtest |
| 并发 | 两个 MLIRContext 使用不同 target/registry | parallel fixture | gtest |
| E2E | AFIR 到 AscendC C++ | CPU emission/device fixture | ST/E2E |

## 9.3 测试命令

当前已新增 `check-afir-capability`，并复用 `check-autofuse-mlir` LIT 目标。所有构建命令使用 `-j 8`。至少执行：

```bash
cmake --build build --target check-afir-capability -j 8
cmake --build build --target check-autofuse-mlir -j 8
git diff --check
```

最近验证中 `check-afir-capability` 29/29、`check-autofuse-mlir` 27/27 通过；正式 pattern 合入后仍需增加对应
LIT/E2E。设备 E2E 仅在具备 CANN 和目标设备的环境执行。

# 10 验收标准

- Registry 冻结后只读，候选结果自有数据、顺序稳定且不含 entry pointer；
- selection 只接收符合上游约定的 buffer-form AFIR；
- selection 属性只保存 kind/ID，未知或失效选择能够定位到具体 Op；
- 单候选不依赖 policy，多候选无 policy 或 policy 返回非候选 key 时确定性失败；
- conversion driver 前完成选择属性格式和 registry lookup 预检；
- conversion 一次性加载 enabled patterns，只有属性指定的 pattern 能匹配；
- Capability 只在 selection 阶段查询一次，conversion 不重复查询或复制 dtype/shape 条件；
- Resource requirement 由 provider 定义，canonical AFIR 只使用 `BufferDescAttr` 和 `afir.workspace_plan` 保存物化事实；
- 不修改公共 API/ABI、legacy optimize/codegen 和 queue demo；
- gtest、LIT、负向用例和首个正式 AscendC lowering 验证通过。

## 10.1 D2 开发准入判断

D0～D2 公共基础设施和首个正式 `afir.add -> ascendc.add_l2` compute lowering 已完成，不能再把本文件标记为“等待 D2”。
当前剩余门禁是：

- canonical `TensorDescAttr`/`BufferDescAttr`、ScheduleToAfir Bridge 和两阶段 `AFIRMemoryAllocatePass` 接入，转由
  2026-08-13 AFIR 方言重构实施方案负责；
- scalar、broadcast inline、inplace、resource 的 legacy consumer 迁移，另立 consumer 设计，不在本文件扩展；
- AscendC 源码生成、CANN Kernel 编译、Profile identity 与 `soc_version` 一致性、设备 E2E 单独验证。

## 10.2 设计文档检查结果

- [x] 跨特性交叉影响：已逐项检查 SuperKernel、Autofuse、AscendC、绑定、构建、测试、性能和兼容性；
- [x] 编码红线：已检查索引、整数、生命周期、图改写确定性、全局状态和 runtime 边界；
- [x] 测试设计：已覆盖接口、pass、负向、并发、确定性和 E2E；
- [x] 性能评估：只增加线性遍历和候选查询，不增加 runtime 开销或跨 pass cache；
- [x] 接口示例：已给出 pipeline、selection 和 conversion 的正常/异常调用方式；
- [x] 最终评审：接口、生命周期、错误语义、阶段边界和测试门禁已闭环；D0～D2 已完成，批准进入 D3/后续 consumer 与
  canonical AFIR 接入开发。
