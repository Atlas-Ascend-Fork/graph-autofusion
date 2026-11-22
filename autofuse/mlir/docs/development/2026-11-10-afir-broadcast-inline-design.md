# 内联广播（brc_inline）：MLIR 侧接通设计

> 2026-11-10 · 上游：[广播缓存与内联广播问题清单与修复方案](2026-11-09-brc-cache-and-brc-inline-issues-design.md)
> 的功能需求 5（第三批）。本文是那一条要求的独立设计文档。

# 简介

## 目的

让 MLIR 具备与 legacy `brc_inline` 模板对等的能力：二元 elementwise 的一侧是行广播时，不再要求
上游先物化一块完整的广播 tile，而是把广播折进 elementwise 调用本身（源侧 repeat stride 取 0）。

目标读者是实现与检视这一批改动的开发者。验收以「legacy 选中 brc_inline 模板的 plan 能被 MLIR
接管，且数值与 legacy 一致」为准。

## 范围

- 包含：`target/Ascend/Capability` 的 `BroadcastInline` 能力查询接通；binary elementwise 的
  能力判定与 lowering 新增一档形态；Asc 方言新增一个扩展 API op 及其 emitter；对应的 lit 与
  数值用例。
- 不包含：`afir.broadcast` 自身三档 lowering 的任何改动；一元与三元 elementwise；reduce；
  legacy 侧任何修改。
- 涉及组件：Autofuse（MLIR codegen 链路）。不涉及 SuperKernel。

# 总体概述

## 软件概述

### 项目介绍

legacy 的 `BrcInlineTemplate`（`optimize/platform/v1/template/brc_inline_template.cpp`）在 optimize
阶段**删除 Broadcast 节点**、给消费者打上 `kInline` 语义角色；codegen 的 `BinaryApiCall` 随后发
`BinaryBrcInlineApiWithTwoVectorizedAxisV2`（`codegen/api_call/elewise/binary_api_call.cpp:196`），
由这个包装按两根向量化轴分块，并把广播那一侧的 repeat stride 乘以 0。

MLIR 侧今天没有这条路：
`CapabilityKind::BroadcastInline` 在全仓库没有生产调用方，九处 `queryBroadcastInline` 一律答
unsupported，`BroadcastInlineQuery` 的字段从未被填充；lowering 侧只有
`isSingletonTensorInput`（`ElementwiseTransferPlan.cpp:25`）的「repeats 全为 1 → 当标量读」一档。
MLIR 也不读上游的模板语义角色，所以一张被 legacy 判成 brc_inline 的图在 MLIR 看来就是「两个
输入局部视图不兼容」的 elementwise，被 `checkLocalViewCompatibility` 拒收。

2026-11-10 的 NPU 实测给出了这条缺口的当前代价：七个广播 e2e 用例里有五个仍带已知拒收诊断，
全部是 `'afir.add' / 'afir.mul' op has no supported Ascend implementation`（4 + 1 条）。

### 产品环境介绍

编译期能力。发射的函数来自 `autofuse/ascendc/api/brc_inline_api.h`，该头文件已随 run 包交付
（`autofuse/ascendc/api/CMakeLists.txt:39`），legacy 以 `brc_inline_api_str.h` 的形式内联进生成
kernel。MLIR 侧走既有的扩展头机制（`ApiExtensionHeader`），不新增交付物。

## 软件功能

1. 识别「二元 elementwise 的一侧是行广播」这一形态，判据与 legacy `IsGeneralizeBrcInlineScene`
   等价。
2. 该形态下由 `BroadcastInline` 能力查询给出结论，使这套接口有真实调用方。
3. lowering 发 `BinaryBrcInlineApiWithTwoVectorizedAxisV2`，参数与 legacy 逐项对应。
4. 不支持的形态照旧报告能力不足、整图回落，不得静默产生错误结果。

## 设计约束

- 拒收只能发生在能力判定层；不得静默放过。
- 新增 implementation 的 dtype 集合**不得窄于 legacy 在同一形态下的集合**（见
  [[elementwise-registration-hand-maintained-gates]] 所述的两道手工门禁）。
- 发射的最后一个实参是函数指针 `&AscendC::<Api><T, false>`。这里的 `false` 是
  `isSetMask`，**与 AGENTS 经验 3.3 的结论不冲突**：包装自己在 counter 模式下设 mask
  （`brc_inline_api.h` 的 `SetMaskCount` / `SetVectorMask` / `SetMaskNorm`），调用方必须交出
  mask 控制权，这正是 `false` 的正当用途。设计里要把这条写进注释，避免后人按 3.3 改回 `true`。
- 不新增硬编码芯片型号或平台判断。

## 假设和依赖关系

三条原本标「待确认」的前提已在实施前实测，结论见文末「实施前的三项实测结论」，这里只留结果：

- 上游确实会把 brc_inline 的 plan 交给 MLIR，且 bridge 不会更早拒收：inline 模板在 AFIR 里完整
  到达，唯一失败点是 `afir.add` 没有可用 implementation。
- `brc_inline_api.h` 的 V2 版本是 legacy 今天实际发的那一个（V1 仍在头文件里，codegen 发 V2）。
- 包装对「两根向量化轴」的假设在这个形态上成立；三根以上向量化轴时与 MLIR 合轴口径是否一致
  没有验证，因此实现按 legacy 的分组写。
- reg 侧没有对等包装，本批次只接 mem 侧。

# 需求分析与设计

## 整体介绍

分三步，每步都能单独验收：

1. **形态判据**：在 `conversion/AFIRToAscendC/Implementations/` 新增一条「行广播输入」判据，
   与 legacy `IsGeneralizeBrcInlineScene` 等价，并由 binary elementwise 的 capability provider
   在 `queryBroadcastInline` 里给出结论。这一步只改判据与查询，不发任何新调用，用 UT 钉住。
2. **API op**：Asc 方言新增 `AscendC_BinaryBrcInlineOp` 及其 emitter，发
   `BinaryBrcInlineApiWithTwoVectorizedAxisV2<T>(dst, src0, src1, shape0, shape1,
   is_input0_block_brc, is_input1_block_brc, first_axis_v_stride, &AscendC::<Api><T, false>)`。
   这一步用 lit 钉发射文本。
3. **接通**：binary elementwise 新增一档 implementation，capability 用第 1 步的判据、pattern 发
   第 2 步的 op；补 lit 与 CPU E2E 数值用例，并在 NPU 上对五个今天带拒收诊断的用例重新测一轮。

## 功能需求

### 功能需求 1：行广播形态判据

1. **介绍**

   判断一个二元 elementwise op 是否满足 legacy 的内联广播形态。

2. **输入**

   op 的两个输入 descriptor 与目的地 descriptor（`afir.operand_descs` / `afir.destination_descs`）。

3. **处理**

   与 `IsGeneralizeBrcInlineScene`（`common/common_utils.cpp:856`）逐条对齐：

   - 两个输入都是张量，向量化轴根数相同；
   - 按向量化轴逐位比较 repeats，**恰好一侧**存在「本侧为 1、另一侧不为 1」的位（两侧都有或
     都没有时均不支持）；
   - 合并连续的广播位与连续的非广播位之后，两侧各为 2 组；
   - 广播侧的**外层**组 repeat 为 1，即形态是 `(1, A) op (B, A)`。

   判据只读 descriptor，不读 `vectorized_strides`：legacy 在判场景时也只看 repeats，stride 只在
   算 `first_axis_v_stride` 时用到。

4. **输出**

   给出「哪一侧是广播侧」以及合并后的两组 repeat；不满足时返回不支持。

### 功能需求 2：`BroadcastInline` 能力查询接通

1. **介绍**

   让这套接口有真实调用方，并由 binary elementwise 的 provider 给出有意义的结论。

2. **输入**

   `BroadcastInlineQuery`：`sourceOperandIndex`（广播侧）、`destinationOperandIndex`、`axis`、
   两侧 `sourceStrides` / `destinationStrides`、`hasDynamicShape`。字段按功能需求 1 的判据结果
   填充——这些字段今天从未被填过，含义要在本次一并钉住，不能各处各填。

3. **处理**

   `queryBroadcastInline` 在形态满足且 dtype 在该 implementation 的集合内时答支持并附约束，
   否则返回不支持。其余八处实现保持 unsupported（它们确实不支持），但要把「为什么不支持」从
   占位注释改成一句真实理由。

4. **输出**

   `CapabilitySupport`；调用方据此选择 implementation。

### 功能需求 3：`BinaryBrcInlineOp` 与 emitter

1. **介绍**

   Asc 方言新增一个扩展 API op，发 `brc_inline_api.h` 的 V2 包装。

2. **输入**

   `dst`、`src0`、`src1` 三块 tile；`shape0`、`shape1`、`isInput0BlockBrc`、`isInput1BlockBrc`、
   `firstAxisVStride` 五个标量；被包装的 API 名（属性）。

3. **处理**

   ODS 照 `AscendC_BroadcastDim3Op` 的写法：`VectorOp<"binary_brc_inline", ...>` +
   `ApiExtensionHeader<["brc_inline_api.h"]>`。emitter 写在
   `api/ascendc/lib/Target/AscendC/Autofuse/` 并在 `Translation.cpp` 注册，负责把最后一个实参
   渲染成 `&AscendC::<Api><T, false>`。

   **形参表逐字照抄 C++**，不省略恒定项——与 `BroadcastDim3Op` 保留恒为 0 的 `src_z` 同一条
   理由：省一个就得在 emitter 里补字面量，把「发的是什么」拆到两个文件去读。

4. **输出**

   一行调用文本；扩展头与前导头按既有机制加入生成 kernel。

### 功能需求 4：接通 binary elementwise

1. **介绍**

   新增一档 implementation，使满足形态的图走内联广播而不是回落。

2. **输入**

   `BinaryElementwiseMapping` 增加一位（或把 `ScalarOperand` 扩成一个更一般的操作数形态枚举）；
   注册点给出该形态的 dtype 集合与 `ExecutionSpec`。

3. **处理**

   - capability：先过既有的 feature / dtype / tmp buffer 检查，再用功能需求 1 的判据；
   - pattern：按判据结果决定两个 `is_inputN_block_brc`，按 legacy 的取法算
     `first_axis_v_stride`：逐根向量化轴看广播侧的 repeat，只要它是 1 就把**目的地**在该轴上的
     `vectorized_strides` 记下来（非 0 时才覆盖），遇到第一个 repeat 不为 1 的轴停止——取的是
     目的地的行距，不是源的（`binary_api_call.cpp:177`），物化两组 repeat，建功能需求 3 的 op。
   - 与缓存守卫的关系：legacy 允许内联广播的消费者同时带 `exec_condition`
     （`DoesInlineNodeNeedCache`、UT `CodegenKernel.BroadcastInlineWithExecCondition`）。MLIR 这边
     `ExecConditionPass` 对任意 AFIR op 一视同仁，所以这个组合自动成立，不需要额外代码；但要
     有一个用例覆盖它，否则「自动成立」只是推断。

4. **输出**

   满足形态的图生成内联调用；不满足的照旧回落。

## 非功能需求

### 可维护性

判据只写一份，capability 与 pattern 共用（与 `analyzeSegmentedBroadcast` 同一条组织方式）。
dtype 集合写在注册点，不散进判据。

### 可测试性

每一步都有会变红的东西：功能需求 1/2 由 UT 钉（判据表 + 查询结论），功能需求 3 由 lit 钉发射
文本，功能需求 4 由 lit + CPU E2E 数值用例钉，最后在 NPU 上对五个用例重测。

### 可移植性

mem 侧与 reg 侧要分别回答：`brc_inline_api.h` 是 mem 侧（Level-0 `BinaryRepeatParams` 形态）的
包装，reg 侧**已实测没有对等物**（见文末第三条）；本设计先只接 mem 侧，reg 侧在确认发射形态之前不注册，避免出现
「注册了但发不出正确调用」。

### 可靠性

新增路径不申请资源。包装内部有 `SetMaskCount` / `SetMaskNorm` 配对，emitter 不得在调用前后插入
其他 mask 操作。

### 特性交叉影响

按 [cross_feature_check.md](../../../../docs/guidelines/cross_feature_check.md) 逐项：

| 场景 | 适用性 | 分析说明 |
|------|--------|----------|
| SuperKernel Python 接口 | 不适用 | 不触及 `super_kernel/` |
| SuperKernel C++/AOT 接口 | 不适用 | 不导出符号 |
| Autofuse 图优化 | 不适用 | 不修改 legacy 的 optimize 与模板；只消费它产出的 plan |
| Autofuse Codegen/Backend | 适用 | 新增 implementation 与 API op，直接影响生成 kernel；需要 lit + E2E 双重验证 |
| AscendC API / Runtime 交互 | 适用 | 新发一个 AscendC 包装调用，内部有 mask 状态与 repeat 参数；要核对 mask 配对与 repeat stride 上限（包装自己处理 `> 255` 的情形） |
| Python/C++ 混合绑定 | 不适用 | 不涉及 |
| 构建与打包 | 不适用 | 头文件已在交付清单内；新增 op 随既有 tablegen 流程 |
| 测试与覆盖率 | 适用 | 新增 UT、lit 与 E2E；NPU 侧重测五个用例 |
| 性能与日志 | 适用 | 见「性能」。不新增默认开启日志 |
| 兼容性 | 适用 | 只把今天回落的形态变成接管，不改变已接管图的生成文本——这一条要用「既有 E2E 生成文本逐字不变」验证 |

## 性能

### 编译时长

新增一条判据（按向量化轴走一遍 descriptor）与一档 implementation 的查询，相对既有候选遍历可
忽略。

### 执行性能

这是收益方向：省掉一次广播物化（一块完整 UB tile 的写入）与一块缓冲。**但不能先声明收益**：
包装在 `cut_quotient >= block || repeat_stride_in_blocks > 255` 时退化成逐 block 的 counter 模式
循环，与「先广播再逐元素」谁快取决于形状。本批次不做性能结论，只要求不比回落慢；真要给数就
在 NPU 上按形状分档测。

### 内存和产物大小

省一块 UB 缓冲；生成 kernel 多一个内联头。

## 接口设计

### 新增/修改接口描述

无对外接口。内部新增：`AscendC_BinaryBrcInlineOp`（Asc 方言）、一条行广播判据的内部函数、
`BinaryElementwiseMapping` 的一位。`BroadcastInlineQuery` 的字段语义首次被真正定义，要同步补
`Capability.h` 的字段注释。

### 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|--------|----------|----------|------|
| 接口说明 | 是否需要接口评审 | 否 | 无对外接口 |
| 接口说明 | 是否需要补充文档 | 是 | 本文 + `Capability.h` 字段注释 |
| 接口兼容 | 行为是否兼容 | 是 | 今天回落的形态变为接管；已接管图的生成文本必须逐字不变 |
| 接口兼容 | ABI/API 是否兼容 | 否 | 不导出符号 |
| 接口约束 | 约束不满足时是否清晰报错 | 是 | 一律 `emitCapabilityShortfall` |
| 接口测试 | 是否需要独立接口用例 | 是 | 每一步各有用例 |

## 软件设计

### 关键数据结构

- 行广播形态的结果结构（广播侧下标、合并后的两组 repeat、`first_axis_v_stride` 的表达式），
  capability 与 pattern 共用一份。
- `BinaryElementwiseMapping` 的新增位；注册点同时决定它与 pattern 的模板实参，保持
  「唯一来源」那条既有约定。

### 关键技术/算法

合并连续广播位 / 非广播位这一步要与 legacy 的 `MergeBrcAxisRepeats` 对齐。**这是本设计里最可能
出偏差的一处**：MLIR 已有 `collapseVectorizedAxes`（按两侧 stride 连续性合轴），与 legacy 按
「是否广播位」分组不是同一条口径。开工第一步要在同一张图上把两者的结果打出来对照，一致才能复用
既有合轴，不一致就得单独写一条分组，并在注释里写明为什么不能复用。

### 流程设计

`queryLowering` → 形态判据 → 选中 implementation → pattern 物化两组 repeat 与 stride → 建 op →
emitter 渲染调用。任一步失败都走能力不足。

### 对子模块的修改

| 目录/文件 | 修改点 |
|---|---|
| `conversion/AFIRToAscendC/Implementations/BinaryElementwise.cpp` | 新增形态、capability 判据、pattern |
| `conversion/AFIRToAscendC/Implementations/ElementwiseTransferPlan.{h,cpp}` | 行广播判据（与 `isSingletonTensorInput` 并列） |
| `target/Ascend/Capability/Capability.h` | `BroadcastInlineQuery` 字段注释 |
| `api/ascendc/include/ascir/Dialect/Asc/IR/Autofuse/OpVecBinary*.td` | 新增 op |
| `api/ascendc/lib/Target/AscendC/Autofuse/` + `Translation.cpp` | emitter 与注册 |
| `test/ut/`、`test/lit/conversion/`、`test/e2e/cpu/cases/` | 用例 |

### 错误处理

#### 系统错误

不申请内存、不访问文件、不调用 runtime。属性缺失按缺信息拒收。

#### 接口错误

全部通过 `emitCapabilityShortfall` 报出，调用方整图回落 legacy。

## 安全检查

按 [编码红线.md](../../../../docs/guidelines/编码红线.md)：

- 敏感信息：不涉及。
- 外部输入作索引/长度：向量化轴下标来自 descriptor，使用前校验长度一致；两个
  `is_inputN_block_brc` 只取 0/1。
- 整数溢出与除 0：`first_axis_v_stride` 与两组 repeat 都是符号表达式，物化走既有
  `materializeStrideValue` / `materializeSegmentExtent`；包装内部的 `/` 除数是 `sizeof(T)` 推出的
  常量。
- 资源释放：不新增资源。
- ABI/API：不导出符号。
- 图改写等价性：本设计不改写图。
- AscendC 生命周期与 mask：包装内部 `SetMaskCount` / `SetMaskNorm` 必须配对，emitter 不在调用
  前后插 mask 操作；`isSetMask=false` 的理由写进注释（见「设计约束」）。

## 兼容性检查

只把今天回落的形态变成接管。已接管图的生成文本必须逐字不变——用既有 CPU E2E 的生成文本作对照。
接管表上会有格子由 fallback 变 takeover，按既有测量方法重新测量并记录。

## 测试设计

### 测试边界

- 入口：UT（判据与能力查询）、`af-opt`（lit）、CPU E2E（数值）、NPU e2e（真机数值与路由）。
- 出口：判据的真假表；发射文本；数值对照；`AF_MLIR_STRICT` 的 `gate=` / `shortfall=`。
- 不打桩。

### 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|----------|------------|----------|------|
| 功能 | 行广播判据的真假表（含两侧都有广播轴、合并后不是 2 组、外层组 repeat 不为 1 三种反例） | 新增 UT | UT |
| 功能 | `BroadcastInline` 查询在该形态下答支持，其余 op 仍返回不支持 | 扩 `BinaryElementwiseL2CapabilityTest` | UT |
| 功能 | 发射文本与 legacy 逐项对应（含 `&AscendC::Add<float, false>`） | 新增 lit | lit |
| 功能 | `(1,A) + (B,A)` 端到端数值 | 新增 CPU E2E 用例 | E2E |
| 特性交叉 | 内联广播 + 缓存条件同时生效 | 新增 lit | lit |
| 异常 | 不满足形态时报告能力不足、整图回落 | 新增 lit（`not af-opt`） | lit |
| 兼容性 | 既有 E2E 生成文本逐字不变 | 全量档 + 文本对照 | E2E |
| 兼容性 | 五个 NPU 用例的诊断是否消失 | NPU 实测 | E2E |

### 测试命令

```bash
# 本地快速回归
/opt/llvm/bin/llvm-lit -q build/autofuse/mlir/test/lit

# 本地全量
bash ./mlir-test.sh

# 远端真机（在 ~/niu 下的独立仓库里，-j 40）
bash autofuse/mlir/test/run_npu_e2e.sh --mode mlir -j 40 --case load_broadcast_add_direct
```

## 验收标准

1. 行广播判据的 UT 真假表通过，且每条反例都能单独变红；
2. 发射文本的 lit 与 legacy 逐项对应；
3. 新增 CPU E2E 数值通过，既有 E2E 生成文本逐字不变；
4. 五个今天带 `'afir.add' / 'afir.mul' op has no supported Ascend implementation` 的 NPU 用例，
   该诊断消失且数值仍然正确；
5. 接管表上受影响的格子重新测量并记录；
6. 文末三条实测结论在实现过程中如被推翻，要就地订正而不是另起一份。

# 实施前的三项实测结论（2026-11-10）

三条都已实测，用的探针是一条临时 ST（`AscGraphBuilder` 建 `(1, N) + (M, N)`，走真实
optimizer 与 codegen，并把 AFIR dump 到工作目录），探针已从源码移除。

**一、bridge 不会更早拒收，形态完整到达。** 这张图的 plan 有四个模板，`afir.original_graph_name`
分别是 `B0Y0`、`B0Y1`、`B0Y0_unaligned`、`B0Y0_inline`。把 dump 交给
`af-opt --afir-plan-candidate-lowering-pipeline` 跑，**只有 `_inline` 那一个失败**，诊断是
`'afir.add' op has no supported Ascend implementation`；其余三个今天都能下降。所以缺口精确落在
lowering/capability，bridge 侧无需改动。

这同时把要收的 descriptor 形态钉死了（`_inline` 模板里 `afir.add` 的三张 descriptor）：

| 侧 | repeats | vectorized_axis | vectorized_strides |
|---|---|---|---|
| 广播侧输入 | `(1, 1, 1, s1)` | `[3, 1]` | `(0, 1)` |
| 另一侧输入 | `(s0/(z0Tb*z0t), z0Tb, z0t, s1)` | `[3, 1]` | `(8*Ceiling(s1/8), 1)` |
| 目的地 | 同另一侧 | `[3, 1]` | 同另一侧 |

判据因此可以写得很紧：广播侧在某根向量化轴上 repeat 为 1 **且** `vectorized_strides` 为 0，另一侧
与目的地在同一根轴上是真实行距。注意行距是补齐到 32B 的 `8*Ceiling(s1/8)`，不是 `s1`——
`first_axis_v_stride` 要取它。

**二、合轴口径在这张图上重合，轴数更多时未验证。** 该形态只有两根向量化轴
（`[3, 1]`），legacy 按「广播位 / 非广播位」分组得到的两组与 MLIR 合轴后的两段是同一个结果。
三根以上向量化轴时两条口径是否仍一致**没有验证**，所以实现仍按 legacy 的分组写，不复用
`collapseVectorizedAxes`；等有三轴以上的语料再决定能不能合并。

**三、reg 侧没有对等包装，本批次只接 mem 侧。** `autofuse/v35/ascendc/api_regbase/` 下没有
brc inline 包装，`autofuse/v35/codegen/reg_api_call/` 也没有对应的特殊发射（`BinaryApiCallV2`
只按通用参数构造走）。但 `autofuse/v35/optimize/template/brc_inline_template_v2.cpp` 存在，说明
reg 侧的 optimizer 照样产出 inline plan——reg 侧那条路**怎么发**要单独核，本批次不注册 reg 侧，
避免出现「注册了但发不出正确调用」。

**复现**：把探针 ST 加回 `test/st/bridge/LegacyGraphGuardSt.cpp`（内容见本节第一段的描述），
`GTEST_FILTER=... ctest --test-dir build/autofuse/mlir/test/st -R '^MlirCodegenSt$' -V`，再把
dump 出来的 `afir_0_RunMlirCodegen_*_plan0.mlir` 交给 `af-opt` 的候选下降管线。

# 第三批验证记录（2026-11-10）

改动：三个 commit——`01c798c`（Store 落点共享的非向量化轴错取生产者广播步长，H.1，独立缺陷）、
`7bd7c89`（订正 ContinuesBrc T4 的错误归因，纯文档）、`da63a31`（本文档的功能需求，接通
`afir.add`/`afir.mul` 的行广播内联形态）。NPU 实测过程中又发现并修了一个遗漏，见下。

- 本地全量档、远端全量档（`npussh mlir`，独立 clone，`-j 32`）：`check-ut`、`check-lit`、
  `check-st`、`check-e2e` 均通过；唯一失败是三个 transpose 用例链接阶段缺 `-ldl`，与第二批验证
  记录里登记的环境问题同源，与本次改动无关。
- 远端 NPU 实测（`run_npu_e2e.sh --mode mlir -j 32`，七个广播用例，`AF_MLIR_STRICT`）：
  - 第一轮（仅 `da63a31`）：原先带诊断的四个用例（`load_broadcast_add_direct`、
    `load_broadcast_diamond_cascade`、`load_broadcast_interleaved_axes_cascade`、
    `load_broadcast_tail_pitch_cascade`）诊断确实消失（`gate=none shortfall=no`），但全部在
    device 编译阶段报 `fatal error: 'brc_inline_api.h' file not found`，`AF_NPU_CASE` 记成
    `ERROR(InductorError)`。
  - 根因：`da63a31` 给新 op 声明了 `ApiExtensionHeader<["brc_inline_api.h"]>`，但
    `autofuse/codegen/ascendc_api_registry.cpp` 那张手工维护的 `api_to_file` 表里没有登记这一
    项——`inlineApiExtensionHeaders` 查不到内容，把 `#include "brc_inline_api.h"` 原样留在生成
    kernel 里，而 device 编译器没有这个文件的 `-I` 路径，于是在编译阶段失败。`brc_inline_api.h`
    本身一直在 `autofuse/ascendc/api/CMakeLists.txt` 的 `ascendc_api_extend_src` 列表里（能生成
    `brc_inline_api_str.h`），只是没人把它接进这张注册表，这一步在 `da63a31` 里被漏掉了。
  - 修复：`39fb89868`（`fix(mlir): 注册 brc_inline_api.h 到 AscendCApiRegistry`），补上这一项，
    已推到 `brc-inline-h1-h2-batch3`。
  - 第二轮（`da63a31` + `39fb89868`）：七个用例里 **六个 PASS**——新增的四个（见上）全部转正，
    原有两个（`load_broadcast_inner_brc_cascade`、`load_broadcast_to_abs`）保持 PASS。
    `load_broadcast_to_cascade` 仍是 `REFUSED(no-impl)`，诊断不变（`'afir.mul' op has no
    supported Ascend implementation`），下沉到它的第三个子形状
    `(1, 8, 1, 8) -> (4, 8, 4, 8)`——另两个子形状（`(1,5)→(33,5)`、`(8,1)→(8,128)`）已经 PASS。
  - `load_broadcast_to_cascade` 残留诊断的归因（选不中，非选中后发错）：该子形状在一个操作数
    上有两根不相邻的广播轴（`a0`、`a2`，中间隔着真实轴 `a1`、`a3`），上游要经两次链式
    `Broadcast` 才能物化，到达 `Mul` 时两个操作数已经是全稠密张量——根本不落在本批次实现的
    「单轴行广播」形态里。代码层确认：`matchBroadcastInlineOperands`
    （`ElementwiseTransferPlan.cpp:153`）要求 `mergeBrcAxisRepeats` 把全部向量化轴合并成恰好
    2 组（一组广播、一组真实），这个形状的广播轴不连续，合并失败、函数返回
    `std::nullopt`——判据按设计拒绝，不是选中后发错。这与本文档「产品环境介绍」一节原始统计的
    「`4 + 1` 条诊断」吻合：4 条 `afir.add` 全部是单轴行广播（本批次已接通），唯一那条
    `afir.mul` 正是这个双广播轴形状，从一开始就不是本批次判据要收的场景，也呼应「开工前三条
    实测结论」第二条「三根以上向量化轴时未验证」的范围边界。不在本批次范围内，不认为是缺陷，
    也未修改判据。
  - 调用路径核实：在生成的 kernel 源码里 grep `BinaryBrcInlineApiWithTwoVectorizedAxisV2`——
    四个新转正的用例全部命中（9～12 处不等，含多个子形状/多份 debug 目录），确认真的发了新调用；
    两个原有 PASS 的用例命中数为 0，确认它们走的是各自既有机制，不是误判为本批次接通的效果。
- **本批次未覆盖**：接管表受影响格子的重新测量，沿用第一批、第二批的做法，本轮未执行，留给
  后续；`load_broadcast_to_cascade` 的双广播轴形状仍待单独立项（不在本设计文档范围内）。
