# 简介

## 目的

补齐 `afir.broadcast` 对**标量源**（ASCIR `Scalar` / `IndexExpr` / `ScalarData` 节点直连 `Broadcast`）的建模，使这一形态与 legacy 的 `BroadcastScalar` 第一分支对齐；同时消除当前该形态触发的 MLIR verifier 硬失败，使任何仍不支持的组合在能力层被拒收并干净回落 legacy。

目标读者：Autofuse MLIR 接管链路（bridge / AFIR dialect / AFIRToAscendC）开发与检视人员。

期望输出：ODS 类型约束、形状校验、能力判据、下降 pattern 四处的协同修改，以及对应的 lit / UT / E2E 覆盖。

## 范围

**包含**：

- `afir.broadcast` operand #0 的 ODS 类型约束与 `verifyBroadcastShapeImpl` 的标量源分支；
- `Duplicate` 一档的能力判据与下降生成（含 `queryScalarInput` 的回答）；
- bridge 侧该形态的失败归类（能力不足标记而非编译错误）；
- 上述修改的 lit 正负例、capability UT 与 CPU E2E 语料。

**不包含**：

- 修改 legacy 的 `ScalarBroadcastOptimizationPass` 使其多消除一类 scalar broadcast（不为 MLIR 的验收去改 legacy；legacy 当前行为正确）；
- 在 bridge 做图改写，把标量直接喂给下游 `afir.truediv`（偏离"接管 legacy 调度产物"的定位，且该 broadcast 可能有多个消费者）；
- 在 bridge 把标量物化成单元素 tile（比 legacy 产物多一块 UB、一次写回与一次同步事件）；
- `BroadcastScalar` 第二分支（`is_ub_scalar`）的建模；
- reduce / concat / transpose / matmul 等其它算子的能力补齐。

涉及组件：Autofuse（bridge、AFIR dialect、AFIRToAscendC、codegen 编排）。不涉及 SuperKernel。

# 总体概述

## 软件概述

### 项目介绍

线上 fuzz 记录（seed `EDL4VTLLGBGY5RIZOUBMFG1GMO`，attempt `53076cc0-201a-4807-b413-bcf1fffe878b`，backend `af-mlir`，CANN 9.3.0）以硬失败收场，诊断为：

```
error: 'afir.broadcast' op operand #0 must be Tensor or MemRef of ... 32-bit float ... or target dialect shaped type, but got 'f32'
AF_MLIR_STRICT gate=lowering shortfall=no graph=autofused_mean_...
```

该诊断**不是 dtype 不受支持**：F32 在 ODS 元素类型表（`dialect/AFIR/IR/Ops/Broadcast.td:29`）、AFIR dtype 位表（`support/AfirElementTypes.h:61-68`）和四个 implementation 的 dtype 集合里都在。约束要求的是"元素类型属于该表的 **shaped type**"，而实际 operand #0 是**裸标量** `f32`。

根因链路（逐段已核）：

1. ASCIR 图把 `mean` 分解为 `sum` + `truediv`，除数（归约元素个数）是一个 `Scalar` 或 `IndexExpr` 节点，经 `Broadcast` 展开后喂给 `truediv`（线上 dump：`truediv1.x2 = broadcast.y`）。
2. bridge 对这两类节点**不建 op、只建值**：`bridge/AfirBuilder/AfirBuilder.cpp:1183-1203`（`handleMarkerNode`，注释即"`Scalar` 不产生 op，只产生一个值"）。
3. 值的类型取自 `parseDtypeImpl` 的**元素类型本身**，不经 `makeTensorType`：`AfirBuilder.cpp:1056`，最终 `arith::ConstantOp` 建出裸标量（`:1100` 浮点 / `:1129` 整数）。`IndexExpr` 的符号路径同样以裸标量收尾（`:1169` `index_cast`、`:1179-1180` `sitofp`）。
4. 下游 `Broadcast` 走通用 `buildComputeNode`，operand 直接取自 `valueMap`、不重建类型（`AfirBuilder.cpp:1428-1436`）。
5. `afir.broadcast` 的 operand #0 约束是 `AFIR_BufferizableShapedType`（`Ops/Broadcast.td:34-41` → `IR/AFIROpBase.td:36-38`，即 `AnyTypeOf<[TensorOrMemRef<elementTypes>, AFIR_TargetShapedType], "", "::mlir::ShapedType">`），没有 `scalarOperand` 口子，于是 `AfirBuilder.cpp:2110` 的 module 校验失败。
6. 该失败未携带能力不足标记，`CodegenOrchestration.cpp:446-451` 的 `classify` 因 `sawShortfall() == false` 返回 `kFailed`——硬失败、不回落，与日志 `gate=lowering shortfall=no` 逐项吻合。

与 legacy 的差距是确定的：legacy 的 `BroadcastScalar` 分三支（`codegen/api_call/broadcast/broadcast_api_call.cpp:242-273`），第一支 `in.is_constant` 直接把常量名拼进 `Duplicate(out[off], in.name, out.actual_size)`（`:249-251`），**既不需要 tile 也不读 `GetValue(0)`**；入口判据 `IsBroadcastConstantTensor`（`:227-240`）对 `Constant`/`IndexExpr`/`Scalar`/`ScalarData` 恒真（`graph_metadef/graph/ascendc_ir/utils/asc_tensor_utils.cc:15-20`）。MLIR 的 `Duplicate` 一档只实现了第三支（真 tile + 事件同步 + `GetValue(0)`，见 `conversion/AFIRToAscendC/Implementations/Broadcast.cpp:892-905`）。

本缺口与当前分支的内联广播改动（`d1b8cd0e4`、`77c390732`、`aca71764f`、`2e0249dbf`）无关：那四个提交只落在 `AFIRToAscendC/Implementations/*`、`Analysis/TransferPlan.cpp`、`Transforms/ExecConditionPass.cpp`，其中 `Implementations/Broadcast.cpp` 仅新增两行注释；而本失败发生在 module 校验阶段，早于这些下降代码介入。

### 产品环境介绍

- 上游：legacy 调度器产出的 ASCIR 图（含 `Scalar` / `IndexExpr` 节点与 `Broadcast` 节点），经 `bridge/ScheduleToAfir` + `bridge/AfirBuilder` 转为 AFIR。
- 下游：`AFIRToAscendC` 生成 AscendC 调用，`Duplicate` 对应 `AscendC::Duplicate` / mem 侧 ui8 包装 `DuplicateU8`。
- 外部接口边界：不新增、不修改任何对外 C/C++ 符号、Python 接口、脚本参数或产物布局。implementation 的稳定 id 属内部注册表身份。

## 软件功能

1. AFIR 能表达"源是标量的广播"，并在 buffer 形态下通过校验；logical（tensor）形态保持不收标量。
2. `Duplicate` 一档在标量源下生成 `Duplicate(dst, <标量>, count)`，不读片上张量、不发同步事件，与 legacy 第一支逐条对应。
3. 标量源下仍不支持的组合（分段两档、不在 `Duplicate` dtype 集合内、目的地多段等）在能力层答 unsupported，整图干净回落 legacy。
4. 任何该形态引发的失败不得再以未标记的编译错误形式出现。

## 设计约束

- CANN Toolkit：按现网 9.3.0 验证；`AscendC::Duplicate` 的签名与两侧头文件现状不变。
- ABI/API：不触碰对外接口，无 ABI 影响。
- 元素类型域：标量源的 dtype 集合**不得宽于** `Duplicate` 一档既有集合——`kDuplicateElementTypes = kF16|kF32|kI16|kI32|kUI16`，mem 侧额外 `|kUI8`（`Implementations/Broadcast.cpp:101, 1105-1111`）。dtype 真源仍是 `support/AfirElementTypes.h`，不另建词表。
- tmp buffer 契约按既有注册值保持：mem 侧 `broadcast.tensor.duplicate` 为 `needsTmpBuffer=true`，reg 侧 `broadcast.tensor.duplicate_reg_base` 为 `false`（`Implementations/Broadcast.cpp:1129-1136`），`checkTmpBuffer` 严格对等不放松。
- 拒收只能发生在能力层：verifier 失败是硬失败，不可作为拒收手段。
- 不得硬编码芯片型号、SoC 字符串或框架名；`Implementations/Broadcast.cpp` 的既有约定是"stable id 与注释都不出现芯片名"。

## 假设和依赖关系

- **待确认 1**：线上该图的除数节点是 `Scalar` 还是 `IndexExpr`。平台 dump 被截断，`ScalarData` 可排除（nnsmith 图只有一个张量输入、无按值标量形参）。两者在 MLIR 侧结局相同（同一个 `buildTypedConstant` / 同样的裸标量），故方案不依赖此区分；判别方法是从完整 dump 读 `broadcast.x` 的生产者节点类型。
- **待确认 2**：该 `Broadcast` 的消费者个数。legacy 的 `ScalarBroadcastOptimizationPass` 要求出入度均为 1（`optimize/schedule_utils.cpp:1100-1111`，`:1107` 处判定）才会消除 scalar broadcast；本图里它未命中，最可能的原因是该除数被两处除法共用。若实测只有一个消费者，则需另查 `IsNodeSupportsScalarInput` 对 TrueDiv x2 位置的返回值——但无论哪种，MLIR 侧都应能独立承接这一形态，不以 legacy 消除为前提。
- **~~待确认 3~~（2026-11-13 已结论）**：裸标量操作数在描述子与 TransferPlan 上本来就有既定处理，不需要新机制：
  - `verifyCanonicalBufferDescriptors` 显式把标量操作数排除在「半个 buffer 形态」判据之外（`IR/AFIRVerifierUtils.cpp:208-218`，注释里举的就是 `afir.add %tile, %scalar`）；
  - `verifyDescriptorList` 以 `allowsScalars=true` 收 `operand_descs`（`AFIRVerifierUtils.cpp:237`）：标量**仍要占一格**描述子、但内容整格跳过（`:105-110`），占位是为了描述子下标不与操作数下标错开；
  - `buildTransferPlan(op)` 默认取 `result_descs[0]`（`Analysis/TransferPlan.cpp:87-100`），与源是 tile 还是标量无关，所以 `Duplicate` 一档的 count 计算一行都不用改。

# 需求分析与设计

## 整体介绍

数据流不变，只在两处放宽并新增一条生成支：

```
ASCIR  Scalar/IndexExpr ──┐
                          ├─► Broadcast ─► TrueDiv ─► Store
       (归约元素个数)      ┘

bridge  裸标量 arith.constant ─► afir.broadcast(operand#0 = 裸标量, destination = tile)
                                     │
ODS/verifier   ① operand#0 放开为「shaped 或裸标量」；形状校验走标量分支
                                     │
能力层         ② Duplicate 一档接受标量源；分段两档仍拒 ─► 不支持则整图回落 legacy
                                     │
下降           ③ 直发 Duplicate(dst, <标量>, count)，不读 tile、不发事件
```

控制流关键点：拒收权只在能力层。ODS 放开后，表达得出不等于生成得出；任何能力层答 unsupported 的组合经 `CodegenOrchestration.cpp:469-477` 的 `unsupported` 分支标记为能力不足，`classify` 返回 `kNotSupported` 并回落 legacy。

## 功能需求

### 功能需求 1：`afir.broadcast` 接受标量源并正确校验

1. **介绍**

   buffer 形态的 `afir.broadcast`，其 operand #0 允许是元素类型属于 `AFIR_BroadcastElementTypes` 的裸标量；destination 与 result 仍必须是 shaped type。logical（tensor）形态行为不变，仍拒绝非 tensor 操作数。

2. **输入**

   - 合法：operand #0 为 `f16/f32/i16/ui16/i32/ui32/ui64/i8/ui8/bf16` 中任一裸标量，destination 为 `LocalTensorType`/memref，二者元素类型相同。
   - 非法并需明确报错：operand #0 为裸标量而 result/destination 元素类型与之不同；logical 形态出现裸标量 operand；destination 缺失。
   - 边界：ODS 元素类型表宽于 AFIR dtype 位表（`I8`/`UI32`/`UI64` 无位），这些类型可建图但选不到 implementation，按既有行为干净回落。

3. **处理**

   - ODS：operand #0 的类型约束改为 `AnyTypeOf<[AFIR_BufferizableShapedType<AFIR_BroadcastElementTypes>] # AFIR_BroadcastElementTypes>`，**复用** `Elementwise.td:216-244` 已有的 `scalarOperand` 写法，不新造机制。destination/result 两位不动。
   - `verifyBroadcastShapeImpl`（`IR/Ops/Broadcast.cpp:26-61`）增标量源分支：operand #0 非 `ShapedType` 时，用 `getElementTypeOrSelf` 与 result 元素类型比对后返回成功，跳过 rank 与 broadcastable 循环（标量无形状可比）。result 侧仍要求 `ShapedType`。
   - `BroadcastOp::verify()`（`:91-106`）的 logical 分支（`:96-98`）天然把裸标量挡在 tensor 形态外，无需改动；bridge 侧 `AfirBuilder.cpp:1190-1193` 也已对 tensor 形态的 `Scalar` 节点给出专门诊断。

4. **输出**

   IR 可构建并通过校验；非法组合给出指向 broadcast op 的明确 verifier 诊断。模块校验（`AfirBuilder.cpp:2110`）不再因该形态失败。

### 功能需求 2：`Duplicate` 一档生成标量源广播

1. **介绍**

   标量源**另注册两个 implementation**（mem / reg 各一个），与 tile 源那两个判据互斥。

   > 2026-11-13 修订：初稿写的是"在既有 `Duplicate` 一档内部分支、不新增 id"，理由是少动注册表的手工清单。实现时按既有先例改掉了——`afir.add` 的 tensor 形态与标量形态就是**同一个 op 的两个 implementation**，判据按操作数形态互斥、不靠优先级（`test/lit/conversion/afir-scalar-binary-api.mlir` 文件头，注册写法见 `Implementations/BinaryElementwise.cpp:980-1008` 的 `registerScalarBinary` 系列）。身份层面「标量从哪来」决定了生成什么（读不读 tile、发不发事件），正是 implementation 该区分的事。代价是要在 `ElementTypeMatrixTest` 的 `kKnownNarrower` 里为 reg 侧标量档补一行快照（与 tile 侧同一个 ui8 理由）。

2. **输入**

   `afir.broadcast`，operand #0 为裸标量，destination 为单段 tile，元素类型属于该 implementation 的 dtype 集合。

3. **处理**

   - 能力层 `BroadcastCapabilityProvider::queryTensorContract`（`Implementations/Broadcast.cpp:780-831`）：
     - 当前 `:786-790` 对非 `LocalTensorType` 的源直接答 `UnsupportedType`；改为先分流——源是裸标量且 `tier == Duplicate` 时走标量支，其余保持原判（分段两档必须继续拒标量源，它们依赖源的几何）。
     - 标量支的判据：元素类型在 `mapping_->elementTypes` 内且与 destination 元素类型相同；"源在每根向量化轴上都是 1"（`:799` 的 `isConstantOnVectorizedAxes`）对标量天然成立，不再查 operand descriptor；`scalarReadEvent`（`:802`）这道判据只对 tile 源有意义，标量支不查。
     - **必须保留**`:813` 那道闸：`Scheduled` 计划下要求 `localTileIsSingleSegment`。原因与源无关——它约束的是"按跨度填会写进行间填充区"，依据在目的地侧。
   - `queryScalarInput`（`:838-841`）：当前一律 `unsupported`。改为 `tier == Duplicate` 且查询为 `scalarOperandIndices == {0}`、`!allScalar` 时答 supported，其余保持 unsupported。该查询今天在生产代码中无调用方（仅 `test/ut/conversion/BinaryElementwiseL2CapabilityTest.cpp` 在用），此处补齐是为了让 op 的能力自描述与 ODS 放开保持一致，不新增调用方。
   - 下降 `BroadcastDuplicateConversionPattern::matchAndRewrite`（`:860-909`）：源是裸标量时，跳过 `:885` 的源侧 `LocalTensorSubIndexOp`、`:892-895` 的 tile 类型检查与 `:896` 的 `materializeScalarRead`，直接以该 SSA 值作为 `DuplicateL2Op` / `DuplicateU8Op` 的标量实参；destination 侧的零偏移与 `count` 计算（`:877-890`）保持原样。
   - dtype 分支按既有写法：`ui8` 走 `DuplicateU8Op`（`:901-902`），其余 `DuplicateL2Op`（`:904`）。

4. **输出**

   生成的 kernel 中该节点为一条 `Duplicate(dst[off], <标量>, count)`，与 legacy 第一支（`broadcast_api_call.cpp:249-251`）形态一致，且比 tile 源少一组同步事件。

### 功能需求 3（条件性，**未实现**）

2026-11-13：功能需求 1/2 已同批交付，因此这一条按设计里写的"两者同批则不引入"**不做**，不留临时补丁。以下保留原文作为依据记录。

仅在需要先解除 fuzz 门禁阻塞、而功能需求 1/2 不能同批交付时引入：在 bridge 建 `Broadcast` op 之前检出"输入是裸标量"，用 `emitCapabilityShortfall`（`support/CapabilityShortfall.h`）报出并返回 failure，使 `classify` 走 `kNotSupported` 回落 legacy。写法照 `AfirBuilder.cpp:794-796`、`:1058-1060`、`:1151-1153` 三处既有先例。

该分支是临时补丁，功能需求 1/2 落地时**必须删除**；若两者同批交付则不引入，避免留下结构性补丁。

## 非功能需求

### 可维护性

- 复用 `Elementwise.td` 的 `scalarOperand` 写法与 `Duplicate` 一档的既有结构，不新增 implementation id、不新增 tier、不另建 dtype 词表或选择算法。
- 标量支与 tile 支在能力层和 pattern 内各一处分流点，分流依据单一（operand #0 是否 `ShapedType`），不散落条件。
- 注释只说明"为何标量支不查 descriptor 与事件"这一非显然点，并指向 legacy 对应分支行号。

### 可测试性

- lit：`test/lit/dialect/afir/` 补正例（标量源 broadcast 可解析可校验）与负例（元素类型不匹配、logical 形态裸标量）；负例一格只违反一条判据并用 `--split-input-file`，CHECK 用计数式。
- 字面量尺寸会被折叠，对齐类判据改用符号尺寸表达，钉不住的部分放 UT。
- capability UT：照 `test/ut/conversion/` 既有 broadcast capability 测试补标量源的 supported / unsupported 两类，覆盖 dtype 不在集合内、目的地多段、分段两档拒标量源。
- E2E：新增一条 `Scalar → Broadcast(多消费者) → TrueDiv` 的 CPU 语料，形状需使 `a0Tb_size > 1`，避免落进"每 tile 只一行"的盲区。
- 量具：`af-capability-probe` 的 FAILED / DRIFT 必须为 0；lit 全绿不足以作为结论。

### 可移植性

mem / reg 两侧分别覆盖，dtype 集合各依其头文件现状（reg 侧 `Duplicate` 档无 `UI8`，是已记录的收窄，不在本次放宽）。不依赖芯片型号判断。

### 可靠性

- 标量源不涉及 UB 分配、不新增资源持有，无释放路径变化。
- 不新增 `rt*` / `aclrt*` 调用。
- 风险点：ODS 放开后若能力层漏判，失败形态会从"硬失败"变成"静默生成错代码"。因此能力层判据与 pattern 分流必须同批落地，且 E2E 必须验数值（输出填哨兵而非 0），不得以"编过了"作为通过依据。

### 特性交叉影响

按 `docs/guidelines/cross_feature_check.md` 逐项：

| 场景 | 适用性 | 分析说明 |
|------|--------|----------|
| SuperKernel Python 接口 | 不适用 | 不涉及 `super_kernel/`、选项解析或 wheel 内容 |
| SuperKernel C++/AOT 接口 | 不适用 | 不涉及 `libascendsk.so`、AOT 与设备端 RDV |
| Autofuse 图优化 | 不适用 | 不新增/不调整 ASCIR pass，不改写图；legacy `ScalarBroadcastOptimizationPass` 明确不动 |
| Autofuse Codegen/Backend | 适用 | 改 AFIR ODS/verifier、`Duplicate` 一档能力判据与下降生成；需覆盖生成代码与 kernel 执行验证 |
| AscendC API / Runtime 交互 | 适用 | 复用既有 `AscendC::Duplicate` / `DuplicateU8`，签名与 tmp buffer 契约不变；标量支比 tile 支少一组 `SetFlag`/`WaitFlag`，无生命周期新增约束 |
| Python/C++ 混合绑定 | 不适用 | 不涉及 `autofuse/compiler/py_module/` 接口 |
| 构建与打包 | 不适用 | 不改 CMake、`build.sh`、依赖与安装路径 |
| 测试与覆盖率 | 适用 | 新增 lit 正负例、capability UT 与 CPU E2E 语料，见测试设计 |
| 性能与日志 | 适用 | 编译期新增一次类型判定，量级可忽略；执行期减少一组同步事件；不新增默认开启日志 |
| 兼容性 | 适用（无破坏） | implementation id 与 tmp buffer 契约保持；既有 tile 源产物逐字不变，需以基线对照证明 |

## 性能

### 编译时长

新增的是每个 broadcast 节点一次 `isa<ShapedType>` 判定与能力层一处分流，常数级，不引入新的图遍历或字符串处理，对 `autofuse/codegen` 编译时长无可测影响。

### 执行性能

标量源比"物化成单元素 tile 再读"的替代方案少一块 UB、一次写回与一组 `SetFlag`/`WaitFlag`；与 legacy 第一支生成的指令序列一致，故相对 legacy 无退化。本形态每节点仅一条 `Duplicate`，不在高频路径新增开销。

### 内存和产物大小

不新增常驻结构或缓存；不新增 implementation 注册项，注册表体积不变；动态库与 run 包大小无可测变化。

## 接口设计

### 新增/修改接口描述

均为内部接口，无对外签名变化：

1. `AFIR_BroadcastOp` operand #0 类型约束：`AFIR_BufferizableShapedType<T>` → `AnyTypeOf<[AFIR_BufferizableShapedType<T>] # T>`（`Ops/Broadcast.td:34-41`）。
2. `verifyBroadcastShapeImpl(Operation *)`：增标量源分支（`IR/Ops/Broadcast.cpp:26-61`）。
3. `BroadcastCapabilityProvider::queryTensorContract` / `queryScalarInput`：增标量源判据（`Implementations/Broadcast.cpp:780-841`）。
4. `BroadcastDuplicateConversionPattern::matchAndRewrite`：增标量源生成支（`Implementations/Broadcast.cpp:860-909`）。

### 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|--------|----------|----------|------|
| 接口说明 | 是否需要接口评审 | 否 | 均为内部 ODS / conversion 实现 |
| 接口说明 | 是否需要补充文档 | 是 | 更新 `docs/development/2026-09-23-afir-broadcast-design.md` 中"源为 tile"的表述，并在 `Implementations/Broadcast.cpp` 文件头的四条 legacy 分支对照里标明第一支已接入 |
| 接口兼容 | 行为是否兼容 | 是 | 放宽方向；既有 tile 源路径行为与产物须逐字不变 |
| 接口兼容 | ABI/API 是否兼容 | 否 | 无对外符号、Python 接口、脚本参数或产物布局变化 |
| 接口约束 | 约束不满足时是否清晰报错 | 是 | 元素类型不匹配给 verifier 诊断；能力不足走 `emitCapabilityShortfall` 并回落，不再以未标记编译错误出现 |
| 接口测试 | 是否需要独立接口用例 | 是 | lit 正负例 + capability UT |

## 软件设计

### 关键数据结构

- `BroadcastMapping`（`Implementations/Broadcast.cpp`）：新增一位 `BroadcastSource source`（`Tile` / `Scalar`），默认 `Tile`，加在 `needsTmpBuffer` 之后以免打断既有三处聚合初始化的位置参数。能力层与 pattern 都读这一位：能力层拿它与 IR 上 operand #0 的实际形态比对，不符即答不支持（互斥由此成立）；pattern 拿它决定走不走标量读。
- `ScalarInputQuery`（`target/Ascend/Capability/Capability.h:139-142`，`{scalarOperandIndices, allScalar}`）：沿用，不改结构。

### 关键技术/算法

标量支无需几何分析：`Duplicate` 的语义是"用一个值填满目的地的一段连续跨度"，长度仍由 `materializeScheduledLocalSpan` 给出（首末元素间的物理跨度，不是元素个数），因此 `:813` 的单段闸门必须保留；源侧原有的"每根向量化轴 repeat 为 1"判据在标量源下退化为恒真，不是被放弃的检查。复杂度 O(1)。

### 流程设计

主流程：bridge 建裸标量 → 建 broadcast op → module 校验（标量分支）→ 能力查询（Duplicate 标量支）→ plan 选择 → 下降发 `Duplicate` → 产物组装。

异常流程：能力层答 unsupported → `unsupported` 标记 → `emitCapabilityShortfall` → `classify` 返回 `kNotSupported` → 整图回落 legacy（`CodegenOrchestration.cpp:446-451, 469-477`）。

回滚/清理：无持久化资源，失败即回落，无需清理。

### 对子模块的修改

| 目录/文件 | 修改点 |
|-----------|--------|
| `autofuse/mlir/dialect/AFIR/IR/Ops/Broadcast.td` | operand #0 类型约束放开为"shaped 或裸标量"；补注释说明对应 legacy 第一支 |
| `autofuse/mlir/dialect/AFIR/IR/Ops/Broadcast.cpp` | `verifyBroadcastShapeImpl` 增标量源分支 |
| `autofuse/mlir/conversion/AFIRToAscendC/Implementations/Broadcast.cpp` | 能力层源形态分流、`queryScalarInput` 回答、下降标量支；文件头 legacy 分支对照表更新 |
| `autofuse/mlir/test/lit/conversion/afir-broadcast-duplicate-scalar.mlir` | 新增：标量源正例（含目的地跨步的拒收档） |
| `autofuse/mlir/test/lit/conversion/afir-broadcast-scalar-invalid.mlir` | 新增：三条负例（元素类型不符 / dtype 不在 ODS 表 / logical 形态） |
| `autofuse/mlir/test/lit/conversion/afir-broadcast-duplicate.mlir` | 补 `--implicit-check-not=broadcast.scalar.duplicate`，钉互斥 |
| `autofuse/mlir/test/ut/conversion/ElementTypeMatrixTest.cpp` | `kKnownNarrower` 补 reg 侧标量档一行 |
| `autofuse/tests/framework/easy_asc_graph/asc_graph_builder.{h,cpp}` | 新增 `BroadcastFromScalar`：标量源的输出轴取图的循环轴（源的输出视图是空的，没有别的输入可跟）。纯新增，既有三个 `Broadcast` 重载行为不变 |
| `autofuse/mlir/test/e2e/cpu/cases/load_scalar_broadcast_store/` | 新增：`Scalar → Broadcast(两个消费者) → Add → Mul → Store`，两个形状档 |
| `autofuse/mlir/test/e2e/cpu/CMakeLists.txt` | 注册上面那个用例目录，并把 `_131` 档挂进 `check-e2e-cpu-nonreduce-addressing` |
| `autofuse/mlir/docs/development/2026-09-23-afir-broadcast-design.md` | 待办：修订"源为 tile"的表述 |

bridge（`AfirBuilder.cpp`）在功能需求 1/2 同批交付时**不改**；仅在采用条件性功能需求 3 时增临时拒收并随后删除。

### 错误处理

#### 系统错误

无内存申请、文件或 runtime 调用新增路径。pattern 内任何失败均走 `notifyMatchFailure`，由 conversion 框架回滚，既有行为不变。

#### 接口错误

- 元素类型不匹配 / logical 形态裸标量：verifier 诊断，指向 broadcast op。
- 能力不足：`emitCapabilityShortfall` 标记 + 既有诊断文本，普通模式回落、严格模式 `kFailed` 并打出 `gate=` 标签，两模式诊断文本相同。

## 安全检查

按 `docs/guidelines/编码红线.md` 逐条确认：

- 无硬编码敏感信息、公网地址、账号或芯片/框架类型判断（第 1、Graph-autofusion 第 2 条）。
- 外部输入作索引/长度：`count` 仍由既有 `materializeScheduledLocalSpan` / `LocalTensorGetSizeOp` 给出，未新增由图属性直接充当长度或偏移的路径；单段闸门保留，防止按跨度写进填充区（第 2、3 条）。
- 整数运算：标量常量解析沿用 `buildTypedConstant` 既有的溢出/非法文本拒绝逻辑（`AfirBuilder.cpp:1095-1128`），不放宽（第 3 条）。
- 资源释放与内存申请：不新增资源持有或分配（第 4、5 条）。
- 不依赖未指定求值顺序；不新增 `unordered_map` 遍历顺序或指针地址顺序决定产物的逻辑，图改写确定性不受影响（第 6、Graph-autofusion 第 5 条）。
- 不扩大修改范围：不顺手改 legacy、不重构相邻代码、不删除既有死代码（第 7、Graph-autofusion 第 3 条）。
- ABI/API：无对外接口变化（Graph-autofusion 第 1 条）。
- pass 时序：不新增、不调整 pass 顺序（Graph-autofusion 第 4 条）。
- 日志：不新增默认开启日志（Graph-autofusion 第 6 条）。
- AscendC 生命周期：标量支不涉及异步拷贝，`Duplicate` 的目的地在当前 tile 内有效；少发的同步事件原本只服务于"从 tile 读标量"，标量支无该读取（Graph-autofusion 第 7 条）。
- 不使用非开放 runtime 接口；不涉及 Python 绑定引用计数；不改构建脚本；测试产物落已忽略目录（Graph-autofusion 第 8—12 条）。

## 兼容性检查

- 既有 tile 源的 `Duplicate` 产物必须逐字不变：需先用改动前代码跑 `check-e2e-cpu` 留一份 kernel 基线，再以改动后产物对照（build 目录平时不留 kernel.cpp）。
- implementation 稳定 id、dtype 集合与 tmp buffer 契约不变，`ElementTypeMatrixTest` 的既有快照不应出现新增或变更行；若出现，说明改动越界。
- 不影响脚本参数、配置文件、安装目录、动态库、Python 包与历史产物。

## 测试设计

### 测试边界

- 入口：`af-opt` 跑 AFIR 校验与 AFIRToAscendC（已编译，可做秒级回路）；capability UT 直接构造 op 查询；E2E 走 `check-e2e-cpu-<case>` 目标。
- 出口：verifier 诊断、能力查询结果、生成的 AscendC 调用形态、CPU 数值结果。
- 打桩：沿用既有 capability UT 的构造方式，不新增桩。
- 验证数据：E2E 输出缓冲预填哨兵值，确保"没算"与"算对"可区分。

### 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|----------|------------|----------|----------|
| 功能 | 标量源 broadcast 可校验、下降为 `Duplicate(dst, 标量, count)` | lit（符号尺寸，CHECK 计数式） | UT |
| 功能 | `Scalar → Broadcast(多消费者) → TrueDiv` 端到端数值正确 | `check-e2e-cpu-<case>`，形状满足 `a0Tb_size > 1` | ST |
| 异常 | 标量源与 result 元素类型不同、logical 形态裸标量、缺 destination | lit 负例，`--split-input-file`，一格一条判据 | UT |
| 异常 | dtype 不在 `Duplicate` 集合内、目的地多段、分段两档遇标量源 | capability UT 断言 unsupported 且原因码正确 | UT |
| 性能 | 标量支相对 tile 支少一组同步事件；相对 legacy 无额外指令 | 生成代码人工对照 + 基线 diff | 手工验证 |
| 兼容性 | 既有 tile 源 kernel 产物逐字不变 | 改动前后 kernel 基线 diff | ST |
| 特性交叉 | 接管表中 broadcast 相关格子不出现新增硬失败 | `af-capability-probe`，FAILED/DRIFT 为 0 | E2E |

### 测试命令

```bash
bash ./mlir-test.sh
```

```bash
make -C build check-e2e-cpu-<case>x128 -j 6
```

```bash
cmake --build build --target af-opt -j 6
```

预期结果：`mlir-test.sh` 结论行为全绿（只认最后那行，不数 `[ FAILED ]`）；新增 E2E 用例数值通过；`af-capability-probe` 的 FAILED / DRIFT 为 0；线上该 seed 复测由硬失败转为 MLIR 正常接管。

## 验收标准

1. 线上 seed `EDL4VTLLGBGY5RIZOUBMFG1GMO` 复测不再出现 `operand #0 must be ... but got 'f32'`，且该图由 MLIR 接管（若因其它判据回落，诊断中必须带能力不足标记，不得是未标记的 `kFailed`）。
2. 标量源 broadcast 生成的调用与 legacy `broadcast_api_call.cpp:249-251` 的形态一致。
3. 既有 tile 源产物经基线对照逐字不变；`ElementTypeMatrixTest` 快照无变更行。
4. 新增 lit 正负例、capability UT 与 CPU E2E 全部通过；`mlir-test.sh` 结论行全绿。
5. `af-capability-probe` 的 FAILED / DRIFT 为 0。
6. 变异验证：将标量支的元素类型判定或单段闸门任一处反向改动，对应负例必须变红（区分盲区／冗余／保守边界三种情形）。
7. 文档已更新：`2026-09-23-afir-broadcast-design.md` 的"源为 tile"表述与 `Implementations/Broadcast.cpp` 文件头的 legacy 分支对照表。

## 落地验证记录（2026-11-13）

实现已按功能需求 1/2 完成，功能需求 3 未引入。本机（x86_64 + CPU 模拟器）实测：

| 项 | 结果 |
|----|------|
| 全量 MLIR 套件 `bash ./mlir-test.sh` | 结论行 `Autofuse MLIR targets succeeded: check-ut check-lit check-st check-e2e`，14m42s |
| lit | 299 / 299 通过（含新增两份） |
| `AFIRToAscendCTest` | 88 / 88 通过（含 `ElementTypeMatrixTest` 的新快照行） |
| 新增 E2E `load_scalar_broadcast_store` | `_131` 与 `_40x128` 两档均通过，逐元素断言 `(input + 2.5) * 2.5` |
| 生成形态 | `AscendC::Duplicate(v80, c2_5_f32, v79);`——常量直接进调用，无 `.GetValue(`、无 `MTE2_S` 事件对 |
| 两个消费者共用同一块广播 tile | `AscendC::Add(v94, v92, v93, v76);` 与 `Mul(v97, v94, v93, v76);` 的 `v93` 同源 |
| clang-format | 涉及的 7 个 C++ 文件 `--dry-run -Werror` 无输出 |

变异验证（验收标准 6）：把能力层那条源形态互斥判据改成恒不拒收（`if (false && …)`）后重编 `af-opt`，
`afir-broadcast-duplicate.mlir` 与 `afir-broadcast-duplicate-scalar.mlir` **两份同时变红**，确认这条判据
有真实覆盖而不是写了没人查；随后已还原并重新跑绿。

尚未完成 / 未覆盖：

- **线上那个 seed 未复测**：`EDL4VTLLGBGY5RIZOUBMFG1GMO` 要在 fuzz 平台那套环境（aarch64 + 真 NPU + torch）上跑，本机没有。
- **接管表探针未重量**：它在远端 NPU 机器那套环境里，本机不存在，所以"硬失败格数的变化"这项没有数据。
- **`IndexExpr` 形态没单独加用例**：新增用例用的是 `Scalar` 节点。两者在 bridge 里走同一个 `buildTypedConstant` / 同样以裸标量收尾，下降侧把操作数当不透明 SSA 值用，所以机制同路；但线上那张图的除数究竟是 `Scalar` 还是 `IndexExpr` 仍未判定（见"假设和依赖关系"待确认 1）。
- 英文版 `_en.md` 未写。

### 设计文档检查结果

- [x] 跨特性交叉影响：已按 `cross_feature_check.md` 逐项填写，适用项为 Autofuse Codegen/Backend、AscendC API 交互、测试与覆盖率、性能与日志、兼容性，其余说明不适用理由
- [x] 编码红线：已按 `编码红线.md` 通用 7 条与 Graph-autofusion 12 条逐条确认，无违反；重点确认不改 legacy、不扩大范围、图改写确定性与 AscendC 生命周期
- [x] 测试设计：lit 正负例 + capability UT + CPU E2E + `af-capability-probe`，并给出变异验证与基线对照要求
- [x] 性能评估：编译期常数级新增判定；执行期较"物化成 tile"方案少一块 UB 与一组同步事件，相对 legacy 无退化
