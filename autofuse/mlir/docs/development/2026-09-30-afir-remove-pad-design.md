# `afir.remove_pad` 接通设计（T5）

> 2026-09-30 · 上游：[向量化轴合轴设计](2026-09-28-afir-vectorized-axis-collapse-design.md) 探针结论 P5、
> [elementwise 遗留问题清单](2026-09-22-afir-elementwise-remaining-issues.md)
>
> 本文档按 `docs/guidelines/design_document_template.md` 的章节写。

# 简介

## 目的

把 `RemovePad` 接进 MLIR 下降路径：新增 `afir.remove_pad`、bridge 注册、AscendC 侧的
`RemovePadExtend` 扩展 op 与发射器、一个 `ApiFamily::Shared` 的 implementation，以及钉住判据
边界的 lit 用例。读者是继续做布局族（`Pad` / `Transpose` / `Ub2ub`）的人。

它解决的是**非对齐小尾轴下 MTE3 带宽利用率低**的问题：尾轴按 32B 对齐算完之后，搬出去的是
带填充的行，MTE3 要按行发，行短则每行一次，而 `RemovePad` 先在 UB 内把填充挤掉，搬出就变成
一次连续搬运。它与合轴正交。

## 范围

**包含**：

- `afir.remove_pad` 的 ODS（buffer 与 tensor 两形态，与其余 bufferizable 一元 op 同形）。
- bridge 算子注册表一行。
- AscendC 方言的 `ascendc.remove_pad_extend` 与它的手写发射器。
- 一个 implementation（`remove_pad.tensor.extend`，`ApiFamily::Shared`）。
- 分段判据、32B 整除判据，以及它们的正/反例 lit。

**不包含**：

- `Pad` / `Transpose` / `Ub2ub`：同族的其余三个，各自独立。
- 广播那一档「收带填充的 descriptor」：见下文「不做的事 ①」。它是本文档**明确不做**的一项，
  而不是漏项——今天广播的对齐不变量正是靠拒收它守住的。
- 调度侧任何改动。`optimize/` 一个字不改，两个插入点（见「假设和依赖关系」）原样保留。

# 总体概述

## 软件概述

### 项目介绍

`RemovePad` 在 legacy 里有两个插入点，**两侧不同**，这一点决定了本需求能被哪些图触发：

| 插入点 | 条件 | 哪些平台 | 位置 |
|---|---|---|---|
| 主对齐策略 | Load 的尾向量化轴 stride ≠ 1（尾轴非连续读） | **只有 mem 侧**（`is_support_compat_mode == false`） | `optimize/platform/common/base_alignment_strategy.cpp:302` `AddRemovePadForOneNode` |
| 非对齐模板 | 非 scalar 的 Broadcast 之后，或非连续的 Load 之后 | 两侧（v35 另有一份同形实现） | `optimize/platform/v1/template/unaligned_template.cpp:69`、`v35/optimize/partition/vector_func_partitioner.cpp:438` |

两处都走 `ScheduleUtils::AddRemovePadAfter`（`optimize/schedule_utils.cpp:936`）：新节点的
`outputs[0].attr` 整份拷贝生产者的输出属性，`compute_type` 置 `kComputeElewise`、
`unit` 置 `kUnitVector`。**所以 `RemovePad` 的输入与输出逻辑形状完全相同，差别只在
`vectorized_strides`**——输入是带填充的布局，输出是紧凑布局。

### 产品环境介绍

无新增 CANN 版本要求。发的是 Autofuse 自己的扩展包装
`RemovePadExtend`（`autofuse/ascendc/api/removepad.h:14`），它内部一条
`AscendC::GatherMask`。前置头 `basic_api/kernel_operator_vec_gather_mask_intf.h` 由 legacy 两侧的
`IncludeApiHeaderFiles` 声明，MLIR 侧的 include 归 `AddKernelIncludes` 那一层按扩展头名解析。

## 软件功能

1. `afir.remove_pad` 的 ODS 与 verifier（形状、dtype、descriptor 规范性），复用一元 elementwise 那一套。
2. bridge 能把 `RemovePad` 节点建成这个 op。
3. 一份分段判据：把 `collapseVectorizedAxes` 的段列表翻成 `RemovePadExtend` 的三个尺寸。
4. 一条 32B 整除判据：证不了就报 unsupported。
5. `ascendc.remove_pad_extend` 与发射器，扩展头 `removepad.h`。

## 设计约束

- **CANN Toolkit**：无新增版本要求。
- **ABI/API**：无对外接口变化，全部是 MLIR 内部 C++ 与 TableGen。
- **构建打包**：`conversion/AFIRToAscendC/CMakeLists.txt` 与
  `api/ascendc/lib/Target/AscendC/CMakeLists.txt` 各加一个源文件；不改安装路径与产物名。
- **不得硬编码芯片**：这个 op 两侧共用同一个包装（矩阵 tier D1，
  `docs/guidelines/ascir_op_lowering_matrix.md:181`），所以只有一个 implementation、
  `ApiFamily::Shared`，选不选中由供给表查询决定，不由 SoC 字符串决定。
  32B 这个数来自 UB 的访问粒度，与 bridge 里 `kTmpBufferAlignmentBytes` 同一个常量口径，
  不是芯片判别。
- **日志**：不新增默认日志。判据在每个 `afir.remove_pad` 上跑一次（O(段数 × 2)）。
- **SuperKernel 解耦**：不碰 `super_kernel/`。

## 假设和依赖关系

| 前提 | 状态 | 依据 |
|---|---|---|
| `RemovePad` 不要 tmp buffer | **已验证** | `v1_ascir_codegen_impl.h:246` / `v2_ascir_codegen_impl.h:1701` 都没有覆写 `CalcTmpBufSize`，基类默认返回空（`ascir_registry.h:118`）。legacy ST `e2e_load_removepad_store.cpp:52` 手写了一块 8192 的 tmp，那是手搭图，不是调度产物——**别照着它定 `needsTmpBuffer`** |
| 输入与输出的逻辑形状、dtype 相同 | **已验证** | `AddRemovePadAfter` 整份拷贝生产者的 `outputs[0].attr`（`schedule_utils.cpp:947`）；`REG_ASC_IR(RemovePad)` 是 `Input("x","T").Output("y","T")`（`ascir_builtin_ops_v1.cpp:156`） |
| 输出侧恒为紧凑布局 | **已验证** | 两个插入点之后都跟一次 `UnAlignVectorizedStrides`（`unaligned_template.cpp:72`）或输出被标成 `kFixedNotAligned`（`base_alignment_strategy.cpp:189`），两条都把 `vectorized_strides` 置成 repeat 的累积乘积 |
| 输入侧的行距是 32B 的整数倍 | **已验证（调度不变量），但本设计不依赖它** | `SetVectorizedStridesForTensor`（`base_alignment_strategy.cpp:663`）：`kAligned` 时次内层 stride 取 `Align(repeat_inner, 32/sizeof(T))`，`kDiscontinuous` 时尾轴 stride 取 `Align(1, 32/sizeof(T))`（`:695`）。**不依赖**是有意的，见「功能需求 2」 |
| 段分析对两侧 stride 不同的 op 有意义 | **已验证** | `collapseVectorizedAxes` 的判据是「对**所有**参与张量都连续才合并」（`VectorSegmentation.h` 头注释），两侧 stride 不同只会让它少合一段，不会误合 |
| `GatherMask` 的 `repeatTimes` 是 `uint16_t` | **已验证** | `$CANN/x86_64-linux/asc/include/basic_api/kernel_struct_gather.h:69`。包装的形参是 `uint64_t`，赋进去会静默截断——见「不做的事 ③」 |

---

# 需求分析与设计

## 整体介绍

`RemovePadExtend(dst, src, R, P, N)` 的语义，按它内部那条 `GatherMask` 反推：

```
从 src 取 N 行，行与行相距 P 个元素；每行取前 R 个元素；写到 dst，连续、无间隙。
                                   ^^^^^^^^^^^^^^^^^^^^
                                   P 只经 `P * sizeof(T) / 32` 进 src0RepeatStride
```

`R` 进 counter 模式的 `mask`，`N` 进 `repeatTimes`，`P` 只以 `P × sizeof(T) / 32`（单位：32B 块）
进 `src0RepeatStride`。**dst 侧没有任何跨度参数**——`GatherMask` 的写是连续的。

数据流（一个 op 的下降）：

```
TensorDescAttr（src / dst 各自的逐轴 axis/repeats/strides/vectorized_*）
      │
      ├─► collapseVectorizedAxes()        ← 与 elementwise / 广播同一份合轴
      │        输出 VectorPlan{ segments[] }，每段 { extent, strides[src], strides[dst] }
      │
      └─► classifyRemovePadSegments()     ← 本设计新增，纯读 plan
               把段列表翻成 (R, P, N)，三种形态：
                 1 段 + src stride 1     → R = extent, N = 1,      P 不参与     【纯搬运】
                 1 段 + src stride P     → R = 1,      N = extent,  P = stride   【逐元素收集】
                 2 段                    → R = 内段 extent, N = 外段 extent, P = 外段 src stride
               其余段数一律 failure() → unsupported → 整图不走 MLIR
```

控制流边界：判据在 conversion 层（`Implementations/RemovePad.cpp`），因为它读的是段结果；
段分析本身在 `dialect/AFIR/Analysis/VectorSegmentation.cpp`，一个字不改。

**本设计不建 `scf.for`。** 段数上限 2 是硬边界，所以没有「外层段」这个概念——理由见
「功能需求 1」的第三点。

## 功能需求

### 功能需求 1：分段形态判据

1. **介绍**

   给定 `collapseVectorizedAxes` 的结果，判断这个 `afir.remove_pad` 是不是本档收的形状；
   收的话算出 `(R, P, N)` 三个尺寸表达式。不收一律 `failure()`，由 capability 翻成
   `CapabilitySupport::unsupported(UnsupportedShape)`。

2. **输入**

   `VectorPlan`：`tensors` 恰好两项（源一个、目的地一个，各自带 `VectorTensor::side`），
   `segments` 外→内。非法输入：张量数不是「源 1 + 目的地 1」、某段的 `strides` 长度与
   `tensors` 不等长、段数为 0。

3. **处理**

   三条判据，按顺序：

   1. **目的地在整串段上紧凑。** 最内段 `strides[dst]` 可证为 1；两段时外段
      `strides[dst]` 可证等于内段 `extent`。
      理由不是「顺手查一下」：`GatherMask` 的写是连续的，目的地有任何跨度或填充，
      写出来的位置就是错的，而且**不报错**。
   2. **段数 ≤ 2。**
      legacy 把**所有**向量化轴塞进一次调用（`GetNonLastAxisMergeSize` 把非尾轴的尺寸
      连乘起来，`remove_pad_api_call.cpp:52`），所以它隐含假设「行与行的间距处处相同」。
      合轴之后段数 ≥ 3 恰好说明那个假设不成立（某两轴之间还有第二种间距），legacy 在那里
      是**静默算错**，本档报不支持。
      **段数 > 2 时不生成外层 `scf.for`**，虽然技术上做得到：外层循环会把 dst 的起始地址
      推到 `k × 内侧元素数` 处，而那个地址不是 32B 的整数倍时 `GatherMask` 的目的地
      起始地址检查会失败（形态与 P3 里「A > 1 且不对齐时 SIGABRT」同一条）。
      要收它得先有一条那个形状的用例证明地址落得对，那是一次独立的扩展。
   3. **`P × sizeof(T) % 32 == 0` 可证。** 见功能需求 2。

   `(R, P, N)` 的取法（`P` 指源侧行距）：

   | 段数 | 最内段 `strides[src]` | R | N | P |
   |---|---|---|---|---|
   | 1 | 可证为 1 | 内段 extent | 1 | 不参与（`repeatTimes == 1` 时 `src0RepeatStride` 读不到） |
   | 1 | 其余 | 1 | 内段 extent | 最内段 `strides[src]` |
   | 2 | 可证为 1 | 内段 extent | 外段 extent | 外段 `strides[src]` |
   | 2 | 其余 | — | — | — （`failure()`） |

   前两行正是 legacy 那两条分支（`remove_pad_api_call.cpp:78` 的 `is_last_axis_align`）在段
   语言里的样子，第三行是两者的共同一般形：
   **legacy 的 `is_last_axis_align` 判的是「x 的尾向量化 stride 是不是既非 1 也非 0」**，
   与这里「最内段的源 stride 是不是 1」是同一件事，只是这边问的是合轴之后的段。
   第四行拒掉：两段而最内段源 stride 不是 1，意味着一行内部也有跨度，`GatherMask` 的
   `src0BlockStride` 表达不了（包装写死 1）。

   `N == 1` 那一行的 `P` 取 `Align(R)` 还是别的都不影响结果，但**仍然要发一个值**（形参在）。
   取最内段 `strides[dst]`……不行，那是 1。**取 `R` 本身**：它一定过得了整除判据吗？不一定。
   所以这一行**单独放行整除判据**，理由写在代码里：`repeatTimes == 1` 时
   `src0RepeatStride` 在硬件上不被读（只有第 2..N 次 repeat 用它）。这是一条**有前置条件的
   放行**，前置条件是 `N` 可证为 1，而 `N` 在这一行是字面量 1，不是符号。

4. **输出**

   `FailureOr<RemovePadShape>`，字段是三个 `ExprAttr`（或它们所在的段下标）加两个张量下标。
   capability 与 pattern 各算一遍、不缓存——与 `analyzeSegmentedBroadcast` 同一个理由
   （两处问同一个值上的同一件事，缓存反而多一条要对齐的路）。

### 功能需求 2：32B 整除判据

1. **介绍**

   `removepad.h:22` 写的是

   ```cpp
   params.src0RepeatStride = last_axis_size * sizeof(T) / 32;
   ```

   整除截断。`P × sizeof(T)` 不是 32 的整数倍时，repeat 前进的距离小于一行，**零报错、
   结果错**——与 P3 那条 `dst_rpt_stride` 的机理逐字相同。legacy 结构性地躲开了它：它不读
   descriptor 的 stride，而是自己重算 `SizeAlign(x_last_axis_size, 32 / sizeof(T))`
   （`remove_pad_api_call.cpp:88`），那个值必然是整数倍。

   本设计**读 descriptor 的 stride**（那是真实布局，重算出来的只是「应该是」），
   代价是要自己挡住截断。

2. **输入**

   `P` 的 `ExprAttr` 与元素类型的字节数。

3. **处理**

   判据：`P` 可证是 `32 / sizeof(T)` 的整数倍。走 `detail::ExpressionAccess::GetTree`
   的**结构判据**，不走算术证明：

   - 树是 `Integer`：值 % 因子 == 0 即可证。
   - 树是 `Mul`：其中**有一个** `Integer` 因子 % 因子 == 0，**且其余每个因子都是整数值
     节点**（`Integer` / `Symbol` / `Add` / `Mul` / `Max` / `Min` / `Abs` / `Ceiling` / `Floor`）。
     `Pow` / `Rational` / `Real` / `Log` 与各谓词节点一律判不可证——`Pow` 在这套表达式里
     同时承载除法（`x^-1`），放过去就等于放过一个可能不是整数的因子。
   - 其余树形：判不了。

   **为什么不走 `ProveExpressionsEqual`。** 想证的是整除，而这套接口只答「两个表达式是否
   可证相等」。把整除写成相等要用 `P == f × Ceiling(P / f)`，代进 `P = f × Ceiling(Z / f)`
   之后左右两边差一层 `Ceiling(Ceiling(Z / f))`——SymEngine 不把 `ceiling` 套 `ceiling`
   折掉（它的规范化只在实参是 `Number` 时求值），于是**符号 pitch 一律证不出来**，
   而符号 pitch 恰好是动态 shape 下的常态。结构判据能过：`Align(Z, f)` 序列化出来就是
   `(f * Ceiling((Z / f)))`，`Mul` 里那个 `f` 是字面量。
   > **这一条是实测过的，不是读代码推的**：`af-opt` 报出的规范形式就是
   > `(8 * Ceiling((Rational(1 , 8) * z_size)))`（注意内层是 `Rational`，不是 `/`），
   > 而 `afir-remove-pad-symbolic-pitch.mlir` 在这个拼写上选中了 implementation。
   > 那条用例红了说明结构判据没命中，那时要回来改的是这一节而不是用例。

4. **输出**

   `true` / `false`。`false` 一律 unsupported，不发诊断（与合轴、`buildTransferPlan` 同一约定：
   在这里 `emitError` 会让 pass 直接失败，而要的是这张图不走 MLIR）。

### 功能需求 3：`ascendc.remove_pad_extend` 与发射器

1. **介绍**

   目标方言里 `RemovePadExtend` 的 op 与它的手写发射器。

2. **输入**

   `dst` / `src` 两个 `AscendC_LocalTensor`，三个 `AnyType` 尺寸（`index` 或 `i32` 的 SSA 值）。

3. **处理**

   照抄 C++ 形参表，一个不省：`RemovePadExtend(dst, src, R, P, N)`。三个尺寸都
   `static_cast<uint64_t>`（形参是 `const uint64_t`，值来自 `index`）。模板实参取源的元素类型，
   与 `printApiExtensionUnary` 同一条（两端同类型，取源是为了与那一族的写法一致）。

   扩展头 `["removepad.h"]`，**无前置头**：包装里的名字是 `AscendC::GatherMask` /
   `AscendC::GatherMaskParams`，都是限定名，不靠 `utils.h` 的 `KernelUtils` /
   `MAX_REPEAT_TIME`。与 legacy 两侧的 `LoadApiHeaderFiles` 逐字一致（各返回
   `{"removepad.h"}`）。

4. **输出**

   kernel 文本里一行 `RemovePadExtend<float>(v3, v2, static_cast<uint64_t>(c8_idx), ...)`。

## 非功能需求

### 可维护性

新增一个 op 的三处改动（ODS 一行、一个 implementation、一行 bridge registry）在这里仍然成立，
本 op 多出第四处：目标方言的 def + 发射器，因为它发的是扩展包装而不是标准调用。
`Ops/Layout.td` 是新文件，承载设计文档里划的第 7 族「布局与搬运变换」，`Pad` / `Transpose` /
`Ub2ub` 后续落在同一个文件。

`afir.remove_pad` 复用 `AFIR_BufferizableUnaryElementwiseOp`。**这不是说它是 elementwise**：
复用的是「一元、两端同 dtype、两端同逻辑形状」这组形状契约与它的 verifier，而那组契约与
elementwise 逐字相同。它与 elementwise 的差别全部落在 lowering（发的不是逐元素 API、
读的是两侧不同的 stride），而那一层本来就是 per-implementation 的。

### 可测试性

lit 是主入口（详见「测试用例设计」）。`ElementTypeMatrixTest` 与
`BridgeOperationRegistryTest` 的断言都由注册表驱动，新增一行自动进表。

### 可移植性

不新增平台判别。`ApiFamily::Shared` 那一档的 `checkApiFamily` 不问 feature
（`ElementwiseApiFamily.cpp:82`），所以两侧同一个 implementation。

### 可靠性

所有「判不了」都倒向 unsupported。唯一一处**有前置条件的放行**是 `N == 1` 时跳过整除判据，
前置条件在同一个判据里查（`N` 的字面量为 1），不是靠注释约定。

### 特性交叉影响

按 `docs/guidelines/cross_feature_check.md` 逐项：

| 场景 | 适用性 | 分析说明 |
|---|---|---|
| SuperKernel Python 接口 | 不适用 | 不碰 `super_kernel/` |
| SuperKernel C++/AOT 接口 | 不适用 | 同上，无导出符号变化 |
| Autofuse 图优化 | **不适用** | `optimize/` 一个字不改。两个插入点、对齐推断、非对齐模板全部原样；本需求只是让已经产出的 `RemovePad` 节点在 MLIR 侧有落点 |
| Autofuse Codegen/Backend | **适用** | 新增 bridge 注册行、AFIR op、一个 implementation、一个目标方言 op 与发射器。legacy codegen 的 `RemovePadApiCall` 不动，两条路并存 |
| AscendC API / Runtime 交互 | **适用** | 新发一条 `RemovePadExtend`（内部 `GatherMask`），在 V 管线上。它是 op 内的一条调用，没有 op 之间的新依赖，`AFIRPipelineSync` 不需要改；不新增 `rt*` / `aclrt*` 调用 |
| Python/C++ 混合绑定 | 不适用 | 不碰 `compiler/py_module/` |
| 构建与打包 | **适用（轻）** | 两个 CMakeLists 各加一个源文件，不改安装路径、产物名与第三方依赖 |
| 测试与覆盖率 | **适用** | 四条 lit（两正两反）；E2E 见「测试边界」 |
| 性能与日志 | **适用（轻）** | 编译期每个 `afir.remove_pad` 上一次 O(段数) 判据；执行期是净收益（这个节点存在的理由就是省 MTE3 带宽）；不新增日志 |
| 兼容性 | 不适用 | 无对外接口、配置、产物布局变化；MLIR 路径仍由 `AF_ENABLE_MLIR_CODEGEN` 控制 |

## 性能

### 编译时长

每个 `afir.remove_pad` 上：一次 `collapseVectorizedAxes`（与 elementwise 同量级）+ O(1) 次
`ProveExpressionsEqual` + 一次表达式树遍历。capability 与 pattern 各算一遍，即 ×2。
量级与广播那两档相同，不引入新的全图遍历。

### 执行性能

接通它就是为了执行性能：尾轴不对齐时，搬出从「每行一次 MTE3」变成「一次连续搬运」。
本设计不改变这个收益的来源，也不引入额外调用——一个节点一条 `RemovePadExtend`，与 legacy 同。

### 内存和产物大小

不新增结构体、缓存或常驻内存。不要 tmp buffer。产物多一个 op 的 TableGen 生成代码与一个
发射器函数，量级与既有任一扩展 op 相同。

## 接口设计

### 新增/修改接口描述

全部是内部接口：

- `afir.remove_pad`（AFIR 方言 op，稳定性档 `internal`）。
- `ascendc.remove_pad_extend`（目标方言 op）。
- `mlir::ascend::kernel::registerRemovePadImplementations(ImplementationRegistry &, SelectionPolicyRegistry *)`。
- `mlir::ascendc::printOperation(CodeEmitter &, ascendc::RemovePadExtendOp)`。

### 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|---|---|---|---|
| 接口说明 | 是否需要接口评审 | 否 | 全部内部 C++/TableGen |
| 接口说明 | 是否需要补充文档 | 是 | 本文档 + 算子矩阵的 MLIR 状态 |
| 接口兼容 | 行为是否兼容 | 是 | legacy 路径不变；MLIR 路径此前对含它的图直接失败，现在多一条可走的路 |
| 接口兼容 | ABI/API 是否兼容 | 否 | 无导出符号变化 |
| 接口约束 | 约束不满足时是否清晰报错 | 是 | 全部走 `CapabilitySupport::unsupported`，理由码区分 `UnsupportedShape` / `UnsupportedType` / `MissingInformation` |
| 接口测试 | 是否需要独立接口用例 | 是 | 四条 lit |

## 软件设计

### 关键数据结构

```cpp
// 一个 afir.remove_pad 在分段之后的形态。三个尺寸各记「哪一段的哪一项」，不记物化出来的
// 值——capability 只判形态、不建 IR，pattern 才物化。
struct RemovePadShape {
  unsigned source = 0;       // VectorPlan::tensors 下标
  unsigned destination = 0;
  size_t rowSegment = 0;     // R 的来源段（最内段）
  size_t repeatSegment = 0;  // N 的来源段；等于 rowSegment 表示「逐元素收集」形态
  bool rowIsSegmentExtent = true;  // R 取段 extent（true）还是字面量 1（false）
  afir::ExprAttr pitch;      // P；`repeatCountIsOne` 时不参与
  bool repeatCountIsOne = false;
};
```

生命周期：函数局部，不进 IR，不跨 pass。

### 关键技术/算法

两处已在「功能需求」里展开：分段形态分类（O(段数)）与 32B 整除的结构判据（O(树大小)）。

### 流程设计

主流程：`queryLowering` → `hasCanonicalDescriptors` → `buildTransferPlan` →
`checkApiFamily` → `checkTmpBuffer(false)` → 操作数划分（1 输入 1 目的地）→ 元素类型 →
`analyzeRemovePad`（合轴 + 形态 + 整除）→ `supported()`。

pattern：同一串判据重算一遍 → `orderTilesForPlan` → 物化三个尺寸 → 两个 tile 各发一条
偏移 0 的 `LocalTensorSubIndexOp`（本地侧偏移恒为 0，与 `UnaryElementwise.cpp` /
广播两档同一条判据）→ 一条 `ascendc.remove_pad_extend` → `replaceOp(整块目的地)`。

异常流程：任一步失败 → `notifyMatchFailure`，dialect conversion 回滚本 pattern。
**没有回滚清理**：本 pattern 只建 op，不持有资源。

### 对子模块的修改

| 目录 | 修改点 |
|---|---|
| `dialect/AFIR/IR/Ops/Layout.td` | 新文件，`AFIR_RemovePadOp` |
| `dialect/AFIR/IR/AFIROps.td` | 一行 include |
| `bridge/AfirBuilder/AfirBuilder.cpp` | 注册表一行 |
| `api/ascendc/include/.../Autofuse/OpVecRemovePad.td` | 新文件，`AscendC_RemovePadExtendOp` |
| `api/ascendc/include/.../Asc/IR/Ops.td` | 一行 include |
| `api/ascendc/include/.../Target/Asc/Autofuse/VecRemovePad.h` + `Ops.h` | 发射器声明 + 加入 op 列表 |
| `api/ascendc/lib/Target/AscendC/Autofuse/VecRemovePad.cpp` + `CMakeLists.txt` | 发射器实现 |
| `api/ascendc/lib/Target/AscendC/Translation.cpp` | 一行 op 名 |
| `conversion/AFIRToAscendC/Implementations/RemovePad.{h,cpp}` + `CMakeLists.txt` | 新文件 |
| `conversion/AFIRToAscendC/RegisterAscendCImplementations.cpp` | 一行注册 |
| `test/lit/conversion/afir-remove-pad*.mlir` | 三个文件（正例三种形态、符号行距、反例四片） |
| `test/ut/bridge/BridgeOperationRegistryTest.cpp` | production coverage 的计数快照 45 → 46，名单加一行 |

### 错误处理

#### 系统错误

无内存申请、文件或 runtime 调用。表达式解析/证明失败一律当「判不了」处理，不抛不吞：
`Result<T>` 的错误分支直接翻成 `false` / `failure()`。

#### 接口错误

对外只有 `CapabilitySupport`，由 `AFIRSelectImplementation` 汇总；没有任何
implementation 支持时那个 pass 报错，整张图不走 MLIR。

## 安全检查

按 `docs/guidelines/编码红线.md` 逐条：

- **硬编码敏感信息**：无。新增的字面量只有 32（UB 访问粒度）与扩展头名。
- **外部输入作索引/长度/偏移**：段下标全部来自 `collapseVectorizedAxes` 的输出，取用前
  查 `segments.size()` 与 `strides.size() == tensors.size()`；张量下标取用前由
  `orderTilesForPlan` 查越界（它对越界返回 `failure()` 而不是跳过——跳过会让 tile 与
  tensor 错位）。
- **整数溢出/反转/除 0**：整除判据里的 `sizeof(T)` 来自元素类型的位宽，取值 1/2/4；
  因子 `32 / sizeof(T)` 恒为正。字面量解析走 `getAsInteger` 并检查返回值。
  `P × sizeof(T)` 不在编译期算——判的是 `P % (32 / sizeof(T)) == 0`，等价且不乘。
- **资源释放/异常分支**：无资源。
- **ABI/API 兼容**：无导出符号变化。
- **图改写等价性与确定性**：本设计不改 AscGraph。MLIR 侧建的 IR 只依赖段列表的顺序
  （`VectorPlan::segments` 外→内，确定）与张量下标，不依赖任何哈希容器遍历顺序。
- **runtime/AscendC 生命周期**：新发的 `RemovePadExtend` 读 src、写 dst，两块都是同一个
  kernel 内的 UB 张量，生命周期由既有的队列/TBuf 规划管；不新增跨 op 依赖，
  所以 `AFIRPipelineSync` 不需要改。

## 兼容性检查

不影响脚本参数、配置文件、安装目录、动态库名、Python 包与历史产物。legacy codegen 路径
一个字不改。

既有 CPU E2E 的生成 kernel **结构上不可能变**：本需求没有改任何共享代码路径，只加了一个 op、
一个 implementation、一个目标方言 def 和几张表里的一行，没有 `RemovePad` 的图一条都走不到它们。
**这句话的兑现方式是「全套跑一遍全绿」，不是逐字 diff**——后者要先用改动前的代码生成一份基线
kernel（build 目录平时不留 `kernel.cpp`），而本轮没有任何共享路径被改，那道功夫不成比例。
真要 diff 的那一天（改了 `VectorizedLoops.cpp` / `TransferPlan.cpp` 这类共享件时）必须先出基线。

## 测试设计

### 测试边界

**入口**：`af-opt --afir-plan-candidate-lowering-pipeline`（lit）。
**出口**：下降后的 IR 与 `--afir-aggregate-launch` 产出的 kernel 文本。
**打桩**：无。lit 的输入是手写的 AFIR 模块，descriptor 照 `afir-broadcast-two-segment.mlir`
的形状写。

**E2E**：本轮**不加**，而且连「一张端到端的 lit」都做不到。理由必须说清楚（清单纪律 1），
而这一条是核过的，不是推的：

`materializeScheduledTileIndices`（`dialect/AFIR/Analysis/TransferPlan.cpp:699` 与 `:720`）对
Load / Store 的局部侧要求**合轴后只有一段，且那一段的 stride 字面量为 1**。`RemovePad` 的两个
真实上游各违反其中一条：

- **带填充的 tile**（非对齐模板那条路，`Load → Broadcast → RemovePad → Store`）：那块 tile 是
  **两段**（行距与行长不连续），Load 当场报失败。
- **尾轴非连续的 Load**（主对齐策略那个插入点）：它只有一段，但那一段的 stride 是
  `Align(1, 32/sizeof(T))`（f32 是 8），**不是 1**，同样报失败。

两条都不是本需求能解的，前置是**逐轴的 `TransferPlan`**（L-7 里「说对一半」那一条）。
所以本轮的 lit 是四个独立的 `func`，不是一张图——这一点写在
`test/lit/conversion/afir-remove-pad.mlir` 的头注释里，与既有的
`afir-broadcast-segmented.mlir` / `afir-broadcast-extend-reg-base-unsupported.mlir` 同一种写法。
逐轴 `TransferPlan` 落地之后补 E2E，形状用「尾轴非连续的 Load → RemovePad → Store」
（`Data([A,B]) → Load(strides = {1, A}) → RemovePad → Store`，即一次转置读），不需要重新设计。

**另一条同样要说明白**：接通 `RemovePad` 之后，非对齐模板那种图**仍然**进不了 MLIR，
因为它里面的 `broadcast` 报 unsupported（「不做的事 ①」）。`RemovePad` 是那条路的必要条件，
不是充分条件。

### 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|---|---|---|---|
| 功能 | 两段形态 `[N=4][R=8]`、行距 16（64 B）、目的地紧凑 → `RemovePadExtend(8, 16, 4)` | `afir-remove-pad.mlir` 的 `remove_pad_two_segment`，IR + CPP + SELECT | lit |
| 功能 | 逐元素收集：单段、源 stride 8（32 B）→ `R == 1`、`N == 4`、`P == 8` | 同文件 `remove_pad_gather` | lit |
| 功能 | 纯搬运：单段、源已连续 → `N == 1`，行距那一位发行长 | 同文件 `remove_pad_copy` | lit |
| 功能 | 两侧共用同一个 implementation（tier D1）：mem 与 reg 两个 target 上选中的都是它 | 同文件两条 SELECT RUN（`Ascend910B1` / `Ascend950DT_950x`） | lit |
| 功能 | 符号行距 `Align(z_size, 8)` 过整除判据 | `afir-remove-pad-symbolic-pitch.mlir`（只跑到 select：符号行距物化出 `afir.tiling_param`，而带它的函数要有 workspace / TilingData 形参） | lit |
| 异常 | 目的地不紧凑（外段 dst stride 16 ≠ 内段 extent 8）→ unsupported | `afir-remove-pad-unsupported.mlir` 第一片 | lit |
| 异常 | 三段 → unsupported | 同上第二片 | lit |
| 异常 | 行距不是 32 B 整数倍（字面量 9）→ unsupported | 同上第三片 | lit |
| 异常 | 行内也有跨度（内段源 stride 8 ≠ 1）→ unsupported | 同上第四片 | lit |
| 兼容性 | 既有 CPU E2E 全绿（没有 `RemovePad` 的图走不到新代码） | `check-e2e` | E2E |
| 特性交叉 | bridge 白名单里的 buffer 形态 op 都有 implementation；元素类型集合不比 legacy 窄；production coverage 计数 | `BridgeOperationRegistryTest` / `ElementTypeMatrixTest`，表驱动 | UT |

反例四片用的是**计数式** CHECK 而不是单条：`--split-input-file` 下某一片被误放行时总退出码
仍然非零、单条 CHECK 也仍然匹配得到，四片就只剩三片在测。

反例一律 `--split-input-file`，一格只违反一条判据——否则一格能同时被两条判据拒掉，
删掉其中一条不会红（变异不变红的「冗余」那一种）。

### 测试命令

```bash
bash ./mlir-test.sh
```

四个目标 `check-ut` / `check-lit` / `check-st` / `check-e2e` 全绿、退出码 0。

## 验收标准

1. `bash ./mlir-test.sh` 全绿。
2. 新增三个 lit 文件覆盖上表的九项，四条反例各自单独一格、且用计数式 CHECK。
3. 既有 E2E 全绿；没有 `RemovePad` 的图走不到新代码，所以生成的 kernel 不会变（见「兼容性检查」里为什么不做逐字 diff）。
4. 本文档的每条结论在正文中可追溯到 `文件:行`。
5. 「不做的事」三条在文档里写明**放行的代价**与**收它要先加什么用例**。

---

# 不做的事（以及为什么）

### ① 广播不收带填充的 descriptor —— 这是 P3 的挡法，不是漏项

任务描述里的推断是：「今天广播的尾轴 32B 对齐不变量之所以没被违反，正是因为含 RemovePad 的图
进不来」，于是接通之后 P3 会暴露。**核过之后这一条不成立，挡住它的是另一件东西。**

- `unaligned_template.cpp:86` 的注释与实现说的是「broadcast **以及 broadcast 之前**的节点
  尾轴正常对齐」，`RemovePad` 插在 broadcast **之后**。所以非对齐模板里
  **broadcast 自己仍然是 `kAligned`**。
- `kAligned` 的含义在 `SetVectorizedStridesForTensor`（`base_alignment_strategy.cpp:663`）：
  最内向量化轴的 stride 仍是 1，但**次内层的 stride 取 `Align(repeat_inner, 32/sizeof(T))`**。
- 于是 `Z × sizeof(T) % 32 != 0` 时，广播目的地的外段 stride 是 `Align(Z, f)` 而内段 extent
  是 `Z`，`classifySegments` 的 `provablyEqual(outer.strides[dst], inner.extent)`
  （`Broadcast.cpp:334`）**当场不成立**，这一档报 unsupported。reg 侧的
  `classifyExtendSegments` 那条「两个 tile 都稠密」（`Broadcast.cpp:612`）同样不成立。
- `Z × sizeof(T) % 32 == 0` 时 `Align(Z, f) == Z`，判据成立，而那正是 P3 的安全格。

**结论：P3 的挡法是「目的地必须稠密」这条判据，它与 `RemovePad` 接不接通无关。**
接通 `RemovePad` 之后，尾轴不对齐的广播图**仍然**进不了 MLIR——`RemovePad` 有了落点，
它上游的 `broadcast` 还没有。这不是本需求没做完：收那种 descriptor 的正确做法是
`classifyExtendSegments` 注释里写的那一条——**`dst_shape` 的最尾一位改按 stride 取
（即发行距而不是行长）**，而那么发的话 `dst_rpt_stride = 行距 × sizeof(T) / 32` 恒整除，
P3 那条截断根本不出现。也就是说放宽判据与挡住 P3 是**同一处改动的两面**，谁做那一处，
就要同时加那条带填充的正例 lit。本文档把这件事记在这里，作为那次改动的前置。

### ② 段数 > 2 不建外层循环

技术上做得到（广播那一档就建了），代价是 dst 起始地址不再保证 32B 对齐，而
`GatherMask` 的目的地地址检查会因此失败。要收它得先有一条那个形状的用例。

### ③ 三条继承的、本设计不挡的暴露

| 暴露 | 机理 | 为什么不挡 |
|---|---|---|
| `repeatTimes` 截断 | 包装形参是 `uint64_t`，`GatherMaskParams::repeatTimes` 是 `uint16_t`（`kernel_struct_gather.h:69`），`N > 65535` 静默回绕 | `N` 在真实图上是符号（tile 尺寸），编译期判不了；「判不了就 unsupported」会把整档关掉。与 P6 同一处置 |
| counter 模式 `mask` 超过一个 repeat | `R` 大于一个 repeat 覆盖的元素数时，`repeatTimes` 与 `src0RepeatStride` 的配合未核 | 同上，`R` 也是符号。**legacy 承担同一条**，而且它的 CPU E2E 全部取 `s2 = 16`（f32，64 B）——`load_removepad_store` 那五个形状一个都没超过一个 repeat，是 `[128]` 那类盲区的同族 |
| `CheckFuncVecGatherMask` 的目的地余量模型 | CPU 模拟器侧的检查在 `libcpudebug` 里，读的是 `dstSize` / `src0Size`，判据读不到源码 | 与 P6 同形：真撞上是一条 CPU 侧硬失败（SIGABRT），不是 NPU 上的正确性问题。补 E2E 时如果撞上，按 P6 的办法把公式测出来记在这里 |
