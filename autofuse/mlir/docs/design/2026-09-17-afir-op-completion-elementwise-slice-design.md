# AFIR Op 补齐：elementwise 切片开发方案

> 2026-09-17 · 待评审 · 上游：[AFIR DSL 架构重构设计](2026-09-09-afir-public-dsl-architecture-design.md) 5.7 剩余项 2
>
> 本文按开发方案组织，未套用仓库设计文档模板（经确认）。需要走评审模板时另出一份。

## 目标

打通 elementwise 族从 AscGraph 到 AscendC kernel 的完整链路，并沉淀三项地基
（tmp buffer、扩展 API 供给、平台差异收敛），供 reduce / concat / split / gather /
cube / 布局变换五个后续族复用。

今天 MLIR 路径端到端能跑的只有 **Add / Load / Store** 三个计算搬运 op，加
BlockId / TilingParam / 三个同步原语。

---

## 已定的决策

1. **按族切片**，先 elementwise。族划分补第 7 类「布局与搬运变换」（Transpose /
   Pad / RemovePad / Ub2ub）；`MaskedFill` 归 elementwise；`Scalar` / `ScalarData` /
   `IndexExpr` 是 elementwise 的标量源子族，不做就只能跑 tensor⊕tensor。
2. **平台差异下沉到 lowering，上层不感知**。AFIR op 集取 v1 ∪ v2 并集（176 个），
   不选代际基线。
3. **ATT 不迁移**。PlanSelector 与 host tiling 继续走 legacy。
4. **继续用 TQue**，短期不改队列模型。
5. **VectorFunc / Micro / SIMT 标量路径搁置**（仅 reg-based）。
6. **tmp buffer 的尺寸不进 AFIR**，只引用 ATT 产出的 `b{id}_size` 字段。

---

## 查证过的事实

这些是花了功夫查的，后面别再查第二遍。

### 两侧覆盖面

| 环节 | 全集 | 今天能过 | 判据位置 |
|---|---|---|---|
| AscGraph → AFIR（生产 buffer 形态） | v1 86 / v2 172 个有 codegen impl 的 `REG_ASC_IR` | **4**：Load / Store / Broadcast / Add | `AfirBuilder.cpp:67` `isBufferFormOperationSupportedImpl` |
| AFIR 方言 | — | 57 个 ODS Op，buffer 形态仅 4 个 | `dialect/AFIR/IR/Ops/*.td` |
| AFIR → AscendC | AFIR 57 | **3** 个计算搬运 + 5 个辅助 | `RegisterAscendCImplementations.cpp:31` |
| AscendC → C++ | `ascendc` 方言约 415 op | 约 252 有 emitter | `api/ascendc/lib/Target/AscendC/` |

- `AFIR_BRIDGE_OPERATION_REGISTRY` 的 13 条是 **tensor 形态遗留路径**；生产路径
  `ScheduleToAfir.cpp:721` 无条件 `buffer_form = true`，真实白名单是 4 个。
- **`Broadcast` 是断头路**：在 bridge 白名单里，`conversion/` 没有任何 pattern。
- `Elementwise.td` 35 个 op 里**只有 `afir.add`** 是 buffer 形态；连同 `Cast` /
  `Select` / `Where` 共 **37 个**要迁形态。

### elementwise 族规模

| | 个数 | 说明 |
|---|---|---|
| 两平台共有 | **47** | 一元 18 + 二元 21 + 三元 4 + Broadcast 1 + 标量源 3 |
| 仅 reg-based | **81** | Sin/Cos/双曲、Bessel/Chebyshev/Hermite/Legendre 多项式、取整族、Log2/Exp2 等 |
| AFIR ODS 缺 | 5 | Axpy / ClipByValue / IsInf / MaskedFill / Remainder（共有部分） |

### 平台差异只有四档（E0-1 实测）

判据是「API 名 / 扩展头 key / 是否需 tmp」三项在两套实现上是否一致。
逐 op 明细见 [ascir_op_lowering_matrix.md](../guides/ascir_op_lowering_matrix.md)。

| 档 | 含义 | elementwise 计数 | 落点 |
|---|---|---|---|
| **D0** | 三项一致且不依赖扩展头 | **9** | 一个 implementation，零平台代码 |
| **D1** | 三项一致，两侧引用**同一个**扩展头 key | **5**（Add / Axpy / Mul / Maximum / Minimum，均为 `scalar_*.h`） | 一个 implementation + 一条 include |
| **D2** | 三项至少一项不同 | **33** | 两个 implementation + 两个 stable id |
| **D3** | 只在一侧注册 | 0（共有部分）；reg 独有 81 个全在此 | ODS 取并集，实现按供给注册 |

**D2 占 47 个共有 op 的 70%，且每一个 D2 的差异都包含扩展头**——这是 E0-2 必须排在
所有 op 波次之前的实测依据，不是估计。差异来源分布：API 名+扩展头+tmp 三项全差 17 个，
仅扩展头差 10 个，API 名+扩展头 4 个，扩展头+tmp 2 个。

**扩展头的 key 空间是全局且不相交的**：mem 侧 `xxx.h`、reg-base 侧 `xxx_reg_base.h`、
cube 另一批，三者注册进同一个 `AscendCApiRegistry` 单例
（`ascendc_api_registry.cpp:218` / `ascendc_reg_base_api_register.cpp:402` /
`ascendc_cube_api_register.cpp:105`）。**所以「头名 → 头文本」的查表不需要知道平台**，
头名本身就是判别式——这一条简化了 E0-2 的层 3。
少数 reg 侧的 op 有意复用 mem 的头（`Add` → `scalar_add.h`，`Axpy` → `axpy.h`），
这正是 D1 的来源。

纪律三条：stable id 里不出现平台名；`ImplementationKind` 不加 `RegBase`
（它是**目标语言**轴，mem/reg 都是 AscendC）；`TargetFeature` 按 API 供给族命名。

**一处 emitter 陷阱**：`GetApiName()` 的约定是裸名，但 `Maximum` / `Minimum`
返回的是 `"AscendC::Max"` / `"AscendC::Min"`——带命名空间前缀。emitter 不能无条件拼
`AscendC::`。这是 E0-5 定命名空间写法时的直接输入。

`dialect/AFIR/`、`bridge/`、`artifact/` 三层今天对 soc 字面量与 `NpuArch::`
**已是零命中**——门禁是防回退，不是先清理。

### 扩展 API 的现状

- 两套头：mem-based `autofuse/ascendc/api/` **46 个**；reg-based
  `autofuse/v35/ascendc/api_regbase/` **95 个**。函数是**全局命名空间**的
  `XxxExtend`（`AbsExtend` / `BitwiseAndExtend` / `IsnanExtend`）。
- legacy 把整段头文本**内联**进 kernel 源码（`codegen_kernel.cpp:3729` +
  `AscendCApiRegistry`，构建期由 `sed` 把头转成 raw string）。
- 本段记录的是设计时状态。当前 `DataCopyPadExtendOp` 已通过
  `ApiExtensionOpInterface` 声明 `datacopy.h`，由 `AddKernelIncludes` 收集；
  `TargetFeature::DataCopyPadExtend` 在 V2201/V3510 上均可用。
- `ascendc` 方言里的 `*Ext` op（`is_nan` / `bitwise_and` / `where_extend` /
  `cast_extend` / `axpy_extend` …）**是只有声明没有 emitter 的空壳**：都不在
  `Translation.cpp` 的分发表里；apiName 部分对不上（`is_nan`→"IsNan" vs 实际
  `IsnanExtend`）；`AscFunc` trait 无任何 emitter 消费，全局 / `AscendC::` 命名空间
  的问题没人解决过。**没有 `abs_extend` 这个 op。**

### tmp buffer 的数据链路

**规模**：47 个共有 elementwise op 里，mem 侧 **34 个**需要 tmp，reg 侧只有 15 个
（E0-1 实测）。也就是说 tmp buffer 不是少数 op 的特例，是 mem-based 路径的常态，
E0-5 定下的形式会被 70% 的 op 复用。

| 阶段 | 位置 | 产出 |
|---|---|---|
| 计算 | `v1_ascir_codegen_impl.h::CalcTmpBufSize` → `ascir/reg_func/*.cpp` | `TmpBufDesc{Expression size; int64_t life_time_axis_id;}` |
| 落盘 | `buf_que_allocator.cpp:402` `GetAndSetNodeTempBuffer` | `node->attr.tmp_buffers[i].buf_desc` |
| 分配 id | `mem_reuse_manager.cpp:342` `AllocTmpBuffGroup` | `tmp_buffers[i].id`，**跨节点可复用**，与普通 buffer 共用 id 空间 |
| 聚合一遍 | `buf_occupy_expr.cpp:124` | `container_exprs["b{id}_size"] = max(各节点内 Σ)`，无下限 |
| 聚合二遍（覆盖） | `asc_graph_ub_expr_builder.cpp:277` ← `gen_model_info.cpp:204` | 同名 key 重写为 `max(Σ, 8192)`，**8 KB 下限在这一遍进入最终值** |
| host 落地 | `tiling_data_generator.cpp:535` | 逐项生成 `SetB{id}Size(tiling_data)` |

- 命名两侧都是 `"b" + id + "_size"`，与 MLIR 的 `MakeBufferSizeName` 逐字相同；
  可覆盖的 `attr.buf.name` **全仓零写入点**，不存在命名分叉。
- **bridge 能拿到**：`CollectTmpBufferLifetimes`（`ScheduleToAfir.cpp:755`）已经在读
  `id` 与 `life_time_axis_id`；`size` 是 `af::Expression`，现成的 `ExprStr()` 能转。
- **但不搬 size**：单节点表达式既不是实际分配值（差一个 max 和 8KB 下限），
  聚合口径又在不迁移的 ATT 里。用 `#afir.buffer_desc<size_from_resource = true>`
  引用 `b{id}_size` —— `CanonicalResources.cpp:288` 的注释已经把这个理由写死了。
- 边角：`AddTmpBuffer` 对 `!HasPrintableExpr(size)` 的 tmp 会跳过
  （`asc_graph_ub_expr_builder.cpp:270`），这种 id 拿不到下限。legacy 既有行为，
  不在本切片修。

### tmp buffer 在 AFIR 里的形态

```mlir
%s = afir.tiling_param "b7_size"
%tmp = memref.alloc(%s) {
  afir.buffer_desc = #afir.buffer_desc<space = <ub>, buffer_id = 7,
                                       size_from_resource = true,
                                       alignment_bytes = 32,
                                       life_time_axis_id = 1>
} : memref<?xui8, #ub>

%y = afir.isnan %x outs(%dst) tmp(%tmp) { ... }
```

- **tmp 是可选操作数不是属性**：谁用它、活多久是 use-def 事实。
- **元素类型固定 `ui8`**：AscendC 侧就是 `LocalTensor<uint8_t>`，由被调 API 自己
  `ReinterpretCast`（`ascendc/api/isnan.h:52`）。
- **动态维就是 `b{id}_size` 本身**——诚实，且与 `InitBuffer` 共用同一个
  `afir.tiling_param`。顺带解决 `AfirBuilder.cpp:392` 注释吐槽过的
  「写 `arith.constant 1` 这句假话」。
- **`memref.alloc` 在 bridge 建**，与普通 buffer 中间量同一个位置
  （`AfirBuilder.cpp:425` `makeBufferDestination`）。理由：id 与生命周期只有
  `FusedScheduledResult` 有；放到 lowering 里建等于退回 `afir.tmp_buffer_ids`
  那种扁平属性；且 AFIR 形态的 kernel 必须能看出自己占多少片上内存（C1/C3）。
- **MemoryAllocate 一行不用改**：没有 `slot_count` / `queue_depth` →
  `isCanonicalQueueResource` 判成 buffer → 收进 `bufferIds` → 追加 `b{id}_size` →
  `materializeResourceSize` 绑回 → `TBuf` + `InitBuffer`。`life_time_axis_id` 也
  已经在 `CanonicalGroupKey` 里。
- **副产物**：`afir.tmp_buffer_ids` 这个 func 级扁平属性可整条删掉，
  `GenTilingData.cpp:86` 那条特判一起删。**已做**（合入前整改），但**不是**按这里设想的
  「让 GenTilingData 顺着 use-def 去读 alloc」——`AFIRMemoryAllocate` 在它之前就把那些
  alloc 删了。改成 bridge 收集时把 tmp buffer id 并进 `afir.buffer_ids`：同一个 id 空间、
  同一套 `b{id}_size`，合表之后那条特判自然消失。

---

## E0：五步

E0-1 阻塞其余四步；E0-2 / 3 / 4 相互独立，按下面顺序做最省事；E0-5 是集成点。

### E0-1　逐 op 对照表　**已完成**

`tools/af-mlir-tools/scan_ascir_op_lowering_matrix.py` 从
`v1_ascir_codegen_impl.h` / `v2_ascir_codegen_impl.h` 抽每个 op 的
**API 名 / 扩展头 / tmp 计算函数 / 标量输入约束**（解析 `REG_ASC_IR` → impl 类，
含单继承回溯与多分支 return 的并集），输出
[ascir_op_lowering_matrix.md](../guides/ascir_op_lowering_matrix.md) 与同名 `.csv`。
共 176 op，其中 elementwise 47。

改变了后续计划的三条结论：

1. **扩展头的 key 全局唯一，provider 不需要平台绑定**（见上文），E0-2 层 3 简化。
2. **D2 占 70% 且全部涉及扩展头** → E0-2 在 33 个 op 的关键路径上，必须先做。
3. **E1 的原定组成是错的**：Sub 是 D2 且 mem 侧要 tmp，Mul / Maximum / Minimum 是
   D1。最便宜的第一波实际是 D0 且两侧都不要 tmp 的 8 个（见 E1）。

脚本还做两项 QA：引用了未注册头名的 op（结果：无，所有头引用都能在三个 register
里解析到）、API 名书写异常（结果：`Maximum` / `Minimum` 带 `AscendC::` 前缀，
`VectorFunc` 无 API 名只有 `VfCall`——后者是已搁置的 VF 融合，符合预期）。

出口（已达成）：表进库；脚本可重跑。与后续实现注册表的一致性由 UT 交叉校验。

### E0-2　扩展 API 供给通道　**已完成**

三层分离：

- **层 1（IR 只表达依赖）**：`ascendc` 方言加
  `AscendCApiExtensionInterface::getExtensionHeader()`，需要扩展头的 op 声明它。
  删掉 `Translation.cpp:76` 的硬编码。
- **层 2（收集去重）**：`AddKernelIncludes.cpp` 在现有 5 条固定 include 之上，
  walk module 按接口收集，**字典序**追加 `emitc.include`。此层之后 LIT 看到的是
  干净的 `#include "isnan.h"`。
- **层 3（内容装配）**：`AscendCApiExtensionProvider`
  （`function_ref<StringRef(StringRef)>`），由 `CodegenOrchestration` 注入。
  **是一张平的「头名 → 头文本」表，不按平台分支**——E0-1 查明三个 register
  （mem / reg-base / cube）注册进同一个 `AscendCApiRegistry` 单例且 key 空间不相交，
  头名自己就是判别式；平台选择已经在层 1（选哪个 implementation → 声明哪个头名）
  完成了，层 3 再分一次就是重复。两套表都已编进 legacy 二进制，
  **不新增构建产物、不改打包**。standalone / LIT 不注入，include 原样保留。

**为什么排在 op 之前**：`datacopy.h` / `DataCopyPadExtendOp` 已经是现成的验证用例，
不需要新 op 就能把通道跑通；反过来，任何 D2 的 op 都要等这条通道。

**ADR-E1**：选「内联文本」而非「随包交付头 + `-I`」，因为后者要让 `CodegenResult`
携带多文件（今天 `kernel` 是单字符串）并改 `ascendc_compile.py`。**推翻条件**：
`CodegenResult` 一旦因其他需求扩成多文件，切到把头写进 `{temp_dir}/device/`
（device 编译已带 `-I {temp_dir}/device`），此时只换层 3。

顺带把三层零命中门禁（soc 字面量 + `NpuArch::`）加进
`scripts/check_platform_agnostic_layers.sh`（当前零命中，已用埋点验证会红）。

落地：接口是 `ApiExtensionOpInterface`，头名由 ODS 的 `ApiExtensionHeader<"x.h">`
生成成静态方法；层 3 是 `artifact/KernelSource/Emit/ApiExtensionInliner`，把
`#include "x.h"` 整行（含换行）替换成头文本，取不到就原样保留。

出口（已达成）：`ApiExtensionInlinerTest` UT；`afir-api-extension-include.mlir`
两个 `--target` 交叉验证 `datacopy.h` 只在 reg-base 侧出现。

### E0-3　ODS 地基　**已完成**

- 三个 `AFIR_Bufferizable*OpBase` 加 `Optional<MemRefOf<[UI8]>>:$tmpBuffer`；
- 新增 `AFIR_BufferizableTernaryOpBase`；
- 三个 elementwise 包装类（一元 / 二元已有 / 三元），集中承载「广播兼容结果形状」
  与「输入结果 dtype 可不同」；
- `inferElementwiseShape` 单一推导函数，verifier 与 `inferReturnType*` 共用
  （范本是 `Reduce.cpp` 的 `inferReduceShape`）。

tmp buffer 的形式：可选的**尾部**操作数，类型 `memref<?xui8, #afir.space<ub>>`，
extent 取 canonical 资源名 `b{id}_size`。AFIR 自己不算尺寸，MemoryAllocate 不必改。
操作数因此分三段 `inputs | destinations | tmpBuffers`，段长由
`afir::getOperandPartition` 单点推导——verifier、descriptor 计数、canonical
binding、tie 查询、PipelineSync 五处都切这三段，各推一次就是五个 off-by-one。

**偏差 1**：`AFIR_BufferizableUnaryOptionalResultOpBase`（Load / Store）不加
`tmpBuffer`。两套 API 家族的搬运都不要 tmp buffer，加了只是给一个不可能的形态留门；
代码里有注释说明。

**偏差 2**：`AFIRDestinationStyleInterface::getNumTmpBuffers` 是**必需**方法，没有
默认实现。给了默认实现时 `DeclareOpInterfaceMethods` 不会路由到 op 自己的定义，
返回值静默变 0；改成必需之后，漏定义是链接错误。

出口（已达成）：`afir_tmp_buffer_roundtrip.mlir`；`afir.add` / `load` / `store` /
`broadcast` 行为零变化。

### E0-4　bridge registry 加形态列 + 标量源　**已完成**

- `AFIR_BRIDGE_OPERATION_REGISTRY` 加「形态」列，一张表同时驱动 tensor 构造、
  buffer 构造、bridgeable mapping 导出；删掉 `isBufferFormOperationSupportedImpl`
  里的 `StringSwitch` 硬编码。
- `UnsupportedOperation:` 的文案不再枚举四个 op 名（会过期）。
- 顺带修掉 `buildBufferNodeOp` 里写死的四路 op 名链（默认回落到 `afir.add`，
  于是 `Abs` 被建成了 `afir.add`）；改成由 registry 驱动的 `afirOperationNameFor`，
  未知节点类型直接报错。

**偏差 3**：`Scalar` / `ScalarData` / `IndexExpr` 未进白名单——它们还没有 buffer
形态，这一项归到 E1。

出口（已达成）：`BridgeOperationRegistryTest` 把「新增一个 op 只改三处」与
buffer 形态的实际覆盖集钉住。

### E0-5　纵向探针 `afir.abs`　**已完成**

**一个 op 同时命中三项地基**，这是选它的全部理由：

- 需要 tmp buffer（`CalcAbsTmpSize` 对所有 dtype 都返回 tmp）；
- 需要扩展 API（mem-based 走全局 `AbsExtend` + `abs.h`）；
- 是 D2 的标准案例（reg-based 走标准 `AscendC::Abs`，无头无 tmp）。

要做的：新增 `ascendc.abs_extend` op + emitter（**这是 `*Ext` 空壳族第一个被真正
接通的**，要在这里定下全局命名空间 vs `AscendC::` 的写法，作为后续所有扩展 op 的
模板）；两个 implementation 与 stable id（`abs.tensor` / `abs.tensor.extend`）；
新 `TargetFeature::StockVectorMathApi`（profile 未声明按 `NpuArch` 回退，`Unknown`
落 `Unknown` 并诊断，不得默认视为支持）；bridge 侧 tmp buffer 的 `memref.alloc`。
（这个 feature 在 E2 改名为 `RegBasedApiFamily`，理由见 E2。）

**偏差 4**：原定出口「同一份 AFIR IR 在两套 profile 下分别 lowering」不成立。
tmp buffer 不是可选装饰：调度分配了 tmp buffer 的 IR 不能落到 stock 调用上（那块 UB
白占），反之扩展调用也拿不到 tmp buffer。所以 capability provider 要求**tmp buffer 的
有无与选中的 API 匹配**，同一份 IR 只有一个合法目标。改成两条 LIT，一条一个家族，
各自带一条对另一个 `--target` 的反向检查——配对才是「差异」，单条只是「一种 lowering」。

`AbsExtend` 是全局函数不是 `AscendC::` 成员，这就是 `*Ext` 族的命名空间写法：
emitter 手写在 `Basic/VecUnary.cpp`，ODS 不给 `AscFunc`（给了会自动生成带
`AscendC::` 前缀的 emitter）。

出口（已达成）：`afir-abs-extension-api.mlir` / `afir-abs-stock-api.mlir`；
CPU E2E `load_abs_store` 数值正确，生成的 kernel 里是 `AbsExtend<float>(...)` 与
一个 `TBuf<VECCALC>`。

### E0 检视后的遗留

检视（辅助 session）翻出四处没改到的三段划分，全部已修并各自补了判据：
`RaceCheck.cpp`（tmp buffer 当成写、destination 当成读，方向是**漏报**）、
`AFIRToAscendC.cpp` 的下降前置门禁（顺带把 Add/Load/Store 三路硬编码换成问接口）、
`AFIRPipelineSync.cpp::isBufferizedOperation`、`LoweringStage.cpp`。两处顺手的收敛：
`getNumDestinations` / `getNumTmpBuffers` 的方法体下沉到 ODS base（原本 5 个 op 各手抄
两份，是隐性的第四处改动点）；`isBufferizedForm` 只留接口默认实现。

两条带出来的风险，本轮不解决，记在这里：

- **同一个 tmp id 跨节点复用**。legacy 的 `mem_reuse_manager` 允许两个节点共享
  `tmp_buffers[i].id`，而 bridge 是每个节点建一个独立 `memref.alloc`，
  `life_time_axis_id` 取自各自的 `buf_desc`。两者 lifetime 不同时
  `CanonicalResources` 会报 descriptor 冲突、整个 kernel 生成失败。当前唯一的 E2E 只有
  一个 abs 节点，探不到。E2 补多节点用例时一并验。
- **`Broadcast` 是 buffer 形态白名单里唯一没有 conversion 实现的**（改造前就如此）。
  `BufferFormSetIsCoveredByAscendCImplementations` 把这条交叉断言建起来了，Broadcast
  作为**显式**豁免列在用例里，且豁免本身也被反向断言钉住——补上实现时必须删掉那一行，
  否则测试红。

`load_abs_store` 的 E2E 用 int32 而不是 float：stock `AscendC::Abs` 根本没有整型重载，
int32 才是这个 wrapper 存在的理由；float 上 wrapper 只是转发给 stock 调用。

---

## E1～E8

每波先补 LIT 正负例再写实现，一个波次一组提交。

| 波次 | 内容 |
|---|---|
| E1 | D0 且两侧都不要 tmp 的：Exp / Ln / Relu / Sqrt。零平台分支、零扩展头、零 tmp，**只验 ODS 与 bridge 两条地基**。**已完成**，见下 |
| E1.5 | D1 的 tensor-tensor 形态：Mul / Maximum / Minimum（Add 已在 E0-3）。**已完成**，见下 |
| E2 | 一元族其余的 D2：Abs 之外的 tmp + 扩展头组合（Neg / Reciprocal / Sigmoid）。**已完成**，见下；Rsqrt / Tanh / Erf / LogicalNot 另有去处，理由见下 |
| E3 | 标量变体：`tensor_scalar` / `scalar_tensor`，含扩展 `ScalarSub` / `ScalarDiv`。第一档（`Scalar` 源 + 四个交换律 op 的 tensor-scalar）**已完成**，其余分档见下 |
| E4 | 比较与逻辑（输入结果 dtype 不同 + tmp）。谓词族 Isnan / IsFinite / IsInf **已完成**，见下；比较 6 个与逻辑 3 个另有去处，理由见下 |
| E5 | 三元与选择：Select / Where / ClipByValue / MaskedFill |
| E6 | **Broadcast 接通断头路** + Cast + 余下（Sub / Div / TrueDiv / Pow / FloorDiv / Remainder / BitwiseAnd / Sign / Axpy / Gelu / LeakyRelu）。**原先漏列 Sub / BitwiseAnd / Sign**，见进度文档「剩余任务」 |
| ~~E8~~ | **取消**：上游 PyAsc 已补齐 RegBased micro api，本切片 PR 合入后先同步 pyasc 代码，再按同步后的实际供给重新评估。与 E1～E7 的验收本就脱钩（R-1） |
| E7 | 收尾：~~删 `afir.tmp_buffer_ids` 与 `GenTilingData` 特判~~（已做，合入前整改）；~~C4 负例换 `afir.sum`~~（已做）；~~更新覆盖矩阵文档~~（重跑无漂移）；~~编译耗时复测~~（已测，见进度文档）。**这一波实际已经空了** |

**Broadcast 放 E6 不放最前**：它看着最便宜（唯一「能进 AFIR 下不去」的 op），实际是
四项全差的 D2——mem 侧 `Broadcast` + `broadcast.h;duplicate.h` + `CalcBroadCastTmpSize`，
reg 侧 `BroadcastExtend` + `broadcast_reg_base.h;duplicate.h` + 无 tmp，
还要引用两个扩展头。它同时依赖 E0-2 和 E0-5 的全部产出，
不值得为它单搭临时通道。

### E1　D0 四个一元 op　**已完成**

Exp / Ln / Relu / Sqrt 接通 buffer 形态。一个 implementation、一个 stable id
（`<op>.tensor`）、**不查任何 feature**——`UnaryElementwiseMapping` 的家族字段留空
（E2 之后是 `ApiFamily::Any`）。D2 的 op 才需要问 feature；D0 问了反而会把两侧都有的
API 在其中一侧判成不可用。
「tmp buffer 的有无必须与选中的 API 匹配」这条对 D0 仍然成立且必须保留：调度给了 tmp buffer
就报 unsupported，不会悄悄把那块 UB 丢掉。

**先做的一件事：把六个方法体下沉到 ODS。** 一个 bufferizable elementwise op 需要
verify / getRole / getTiedResultIndex / getEffects / getSpeculatability /
inferReturnTypes 六个方法，同一 arity 下逐字相同。现在由 `AFIR_BufferizableElementwiseMethods`
按 arity 生成（实现在 `AFIRElementwiseSupport.h` + `Ops/Elementwise.cpp`），
「新增一个 op 只改三处」这条验收才真的成立——否则每个 op 还要手抄六个函数，而抄错的
表现是静默放行。

**两个被两节点用例逼出来的真 bug**（都不是本波次引入的，是此前没有链式图走到）：

1. `ScheduleToAfir` 按 `position` 反推 queue/buffer——「VecIn / VecOut 是队列，VecCalc
   是 buffer」。两个计算节点串联时中间张量的 position 是 VecCalc 而调度把它分进了队列，
   于是报 `buffer domain cannot use AllocType::Queue`。legacy 只看 `alloc_type`
   （`codegen_kernel.cpp` 的 `TPipe::AddTensor` / `TensorAlloc`），改成同一判据。
   顺带删掉 `ScheduleBindingDTO` 里 `position` / `positionId` / `alloc_type` 三个只写
   字段——信息已经转成 `space` 与 `domain`，再留一份就是第二份事实源。
2. 同一条队列上不同张量的 `slot_count` 被要求相等。legacy 取的是最大值
   （`codegen_kernel.cpp` 对 `que_buf_num_value` 取 max，遍历整条队列而非某个 reuse 组）。
   bridge 的 `CheckResourceCompatibility` 与 conversion 的 `registerResource` 两处都改。
   聚合后的 descriptor 上 `reuse_id` 已无意义，清成 -1。

**偏差**：LeakyRelu 与 Scalar / ScalarData / IndexExpr 推迟到 E3。LeakyRelu 是
`VecScalarL012Op`，要标量操作数通路；后三个没有 conversion 实现，现在进 bridge 白名单
会立刻违反 E0 建立的 bridge↔conversion 覆盖面交叉断言——那正是 Broadcast 断头路的老问题。

**顺带记一条既有偏差**（不在本波次范围）：legacy 的 `TQue` 深度是硬编码的 1
（`codegen_kernel.cpp` 的 `TQue` 构造），`que_depth_value` 只被定义成常量、从不进模板
参数；而 MLIR 用 descriptor 的 `queue_depth` 做 `QueueType` 的深度。调度给的 depth 不是
1 时两边会不一致。

### E1.5　D1 的 tensor-tensor 形态　**已完成**

Mul / Maximum / Minimum 接通 buffer 形态，各一个 implementation
（`<op>.tensor_tensor`），不查 feature。LIT 同样是「同一份 IR、两个 `--target`、
同一组 CHECK」。

**原定的 E1.5 前提要修正。** 计划里写的是「两侧同实现，只多一条 include，是 E0-2 通道
的第二个用例」。查 legacy 之后不成立：这几个 op 的扩展头是 `scalar_mul.h` /
`scalar_maximum.h` / `scalar_minimum.h`，里面只有**全标量**那一路的实现（例如
`scalar_add.h` 只提供 `Adds(dst, x, y)`）。tensor-tensor 形态根本不碰它们——legacy 是
按节点类型无条件内联，那是 legacy 的一处浪费，不是判据。所以这三个 op 在这一波里和 D0
没有区别，扩展头那条路真正落在 E3 的标量变体上。

`Maximum` / `Minimum` 确实在这里第一次撞上 `AscendC::` 前缀：对照表里它们的 API 名连
前缀一起写着（E0-1 的 QA 发现）。MLIR 这边 op 的 APIName 是裸的 `Max` / `Min`，前缀由
emitter 加，LIT 用 `{{^ *}}AscendC::Max(` 钉住「恰好一个前缀」。

**偏差**：`Axpy` 推迟到 E6。它是这一档里唯一真需要扩展通道的（两侧都是
`AxpyExtend` + `axpy.h` + tmp），但它是**二元 + 额外标量属性 + tmp buffer**，要的是二元
扩展 API 那套机制——而 E6 的 Sub / Div 也要同一套。为一个 op 先造一遍不划算。

### E2　一元族其余的 D2　**已完成**

Neg / Reciprocal / Sigmoid 接通 buffer 形态。这一波真正做的事不是「再复制三遍
E0-5」，而是**把 E0-5 里被绑在一起的三件事拆开**。E0-5 只有 `afir.abs` 一个样本，
「标准 / 扩展」「哪张供给表」「要不要 tmp buffer」三项恰好同向，于是被压成了一个 bool。
这三个 op 各打破其中一项：

| op | mem-based（`xxx.h`） | reg-base（`xxx_reg_base.h`） | 打破的是 |
|---|---|---|---|
| Neg | `NegExtend`，无 tmp | `NegExtend`，无 tmp | 两侧都只有包装，差异只在头文件；且**走包装却不要 tmp buffer** |
| Reciprocal | `ReciprocalExtend` + tmp | `AscendC::Reciprocal`，无 tmp | 无（与 Abs 同形，第二个样本） |
| Sigmoid | `SigmoidExtend` + tmp | `AscendC::Sigmoid` + tmp | **两侧都要 tmp buffer**，tmp 与「用不用包装」无关 |

**`TargetFeature::StockVectorMathApi` 改名为 `RegBasedApiFamily`**，语义从「标准向量
数学 API 可不可用」改成「这个平台的扩展 API 由哪张供给表提供」。原来的问法在 Neg 上
答不了（两侧都没有标准调用），在 Tanh / Erf 上会答反（它们的标准调用在 mem-based
一侧，和 Abs 相反）。改名后每个 implementation 自己声明属于哪一家
（`ApiFamily::MemBased` / `RegBased` / `Any`），feature 只回答平台是哪一家。

`UnaryElementwiseMapping` 相应地把 `needsTmpBuffer` 变成独立字段，并且它和「建 op 时
传几个实参」由**同一个模板实参** `TakesTmpBuffer` 决定——两者对不上的后果是少传一块
UB 且没人报错，所以不能是两个可以各写各的字段。

**偏差 5：Rsqrt 不在本波次。** `RsqrtExtend(dst, src, blk_tensor_with_value_1, tmp_buf,
size)` 比其余包装多一个操作数——一块预先填好 1 的 block 张量。当前「一个 op 至多一块
tmp buffer」的模型表达不了它，硬塞会把这个多出来的张量伪装成普通 tmp buffer。放到 E6 和
同样要额外操作数的那批一起做。

**偏差 6：Tanh / Erf 不在本波次。** 它们的扩展包装在 reg-base 一侧（`tanh_reg_base.h` /
`erf_reg_base.h`），**标准调用在 mem-based 一侧**——这个方向本身已经被 `RegBasedApiFamily`
覆盖了。卡住的是标准那一侧：ascendc dialect 里只有 `UnaryMathOp` 形态的 `ascendc.tanh` /
`ascendc.erf`，它们带一个 `isReuseSource` 操作数，emitter 把它当 **C++ 模板实参**打印
（`AscendC::Tanh<T, isReuseSource>(...)`）。dialect 里没有任何东西能产出一个会被 emit
成编译期常量的值，AFIR 侧也没有这个信息。要么给这两个 API 加一组不带该模板实参的 op，
要么让 CodeEmitter 认常量——都是 ascendc dialect 自己的活，不该夹在本波次里。

**偏差 7：LogicalNot 归 E4。** 它的结果 dtype 是 uint8 而输入不是，属于「输入结果
dtype 不同」那一族，和比较算子共用同一套 ODS 与 capability 判据。

**顺带补上 E0-2 扩展通道的一个洞：kernel 前导。** 扩展头不是自足的翻译单元——它们照着
legacy 的前导写，名字可以不带 `AscendC::` 限定，还可以直接用只在 `utils.h` 里的
`KernelUtils` / `MAX_REPEAT_TIME`。legacy 无条件把 using-directive 和 `utils.h` 写进每个
kernel（`codegen_kernel.cpp` 的 `IncludeAndDefines`），MLIR 这一侧一条都没有。先接通的
`abs.h` / `neg.h` 恰好把名字都写全了，所以没人发现；`reciprocal.h` / `sigmoid.h` 一接上
就露出来了，而且**失败不在 MLIR 这一层**：include 照发、文本照内联，最后炸在 device
编译器上，报的是内联进来的那段头文本的行号。

补法是：`AddKernelIncludes` 在有扩展头时发一条 `using namespace AscendC;`；前导头由 op
自己声明（`ApiExtensionHeader<header, prelude>` → `getExtensionPreludeHeader()`），这一层
照旧只收名字、不认家族——reg-base 的 op 将来声明 `utils_reg_base.h` 就是了。前导头单独
一张表而不是混进扩展头那张，是因为那张按字典序发，`utils.h` 排在 `reciprocal.h` 后面。
`utils.h` 还要补进 `AscendCApiRegistry`：legacy 走的是另一条无条件内联的路，从不查这张表。

**一个 implementation 的家族与它建出来的 op 上写死的扩展头名，在注册时对上。** 两者是
同一件事的两份编码，写反了没有任何一层会报错：capability 照答 Supported，include 照发，
那个 key 也解析得到，于是一份 reg-base 的头文本被内联进 mem-based 的 kernel。今天两份
`neg.h` 除 include guard 外逐字相同所以看不出来，E6 的 `broadcast.h` 那一对不是。对不上
就让整张注册表建不起来。

`afir.neg` 的 dtype 表去掉了 I16：`NegExtend` 的 `@tparam` 只声称 int32_t / float / half。
ODS 收下而 lowering 接不住，表现是走到 SelectImplementation 才报「没有可用实现」。

顺带删掉一个死文件：`Basic/OpVecActivationExt.td` 从没被 `Ops.td` include 过，而且与
`Adv/Activation.td` 里真正生效的 `SwiGLUOp` 重名——试着 include 它会直接 tablegen 报
`def already exists`，也就是说它永远进不来。它还在 op inventory 里留下两条指向错误
操作数顺序的记录。本波次的两个 Sigmoid op 定义在 `Adv/Activation.td`，inventory 用
`scan_pyasc_coverage.py` 重新生成。

出口（已达成）：`afir-neg-api-family.mlir`（同一份 IR、两个 `--target`、生成的调用
文本完全相同、只有 ascendc op 与 stable id 不同）、`afir-neg-api-extension-include.mlir`
（这一档唯一真正的差异——内联的是 `neg.h` 还是 `neg_reg_base.h`）、
`afir-sigmoid-api-family.mlir`、`afir-reciprocal-{extension,stock}-api.mlir`、
`afir-unary-tmp-buffer-mismatch.mlir`（把 tmp buffer 判据从家族判据里单独拎出来的负例）；
CPU E2E `load_neg_store`（走包装、不要 tmp buffer、一个 `TBuf` 都没有）与
`load_reciprocal_store`（**内联进来的头文本编译得过**——LIT 和 UT 都验不到这一层）。

### E2.5　二元路径的 E2 化　**已完成**

E3 与 E6 共同的前置，本身不含任何标量内容。`BinaryElementwise.cpp` 一直停在 E0 的形态：
位置下标切操作数、没有 tmp buffer 通路、没有 `ApiFamily`。

四件东西搬进 `ElementwiseApiFamily.h`，一元二元共用一份：家族、tmp buffer、规范描述子
三条判据，外加注册期那条「家族 ↔ 扩展头名」的对照（`checkFamilyMatchesHeader`）。
合并的理由不是少写几行，是**分开写就会分开错**：家族判据答错，capability 照答
Supported、include 照样解析得到，只是内联进 kernel 的是另一张表的头文本；tmp buffer 判据
两条路松紧不一，同一份调度在一元上被拒、在二元上把那块 UB 悄悄丢掉。

**元素类型的取值范围没有并进去**，两族收的集合本来就不同（`afir.neg` 不收 i16、
`afir.mul` 收），合成一个的代价是把某一侧悄悄放宽。各自留在各自的文件里。

操作数改按 `afir::getOperandPartition()` 切。以前那句 `getNumOperands() != 3` 同时兼着
「个数对」和「没有 tmp buffer」两件事，换成三段划分时很容易只保留前一件——
`afir.add %a, %b outs(%d) tmp(%t)` 的划分是 `2 | 1 | 1`，输入数目的地数全对。
`afir-binary-operand-partition.mlir` 钉的就是这一条：把 `checkTmpBuffer` 关掉做反向对照，
`add.tensor_tensor` 当场被选中，那块 tmp buffer 无声消失。

出口（已达成）：`afir-binary-operand-partition.mlir`（多出来的 tmp buffer；两个输入形状
不一致——这条以前由「整个类型逐字相等」顺带管着，拆成元素类型与形状两条之后要单独钉）。

### E3　标量变体　**第一档已完成**

E4 先做了，因为 E3 不是「再接几个 op」。查证到的：

**标量在 AscGraph 里是三种 producer 节点，不是操作数。** `Scalar`（编译期常量，值在
`ir_attr` 里是个字符串）、`ScalarData`（运行期，来自 tiling）、`IndexExpr`（轴与 tiling
参数的表达式）。它们各自产出一个**没有 sched.axis、没有 repeats、没有 strides** 的输出，
被普通二元 op 当成 `x2` 吃掉——legacy 的 `e2e_load_scalar_div_store` 就是这个形状：
`Scalar constant; constant.ir_attr.SetValue("2.0"); div.x2 = constant.y`。
E0-4 的偏差 3 说这三个「还没有 buffer 形态、归 E1」，E1 做的是四个一元 op，没碰它们。

**legacy 靠一遍全图分析把它们变回标量。** `is_ub_scalar` 由「所有 repeats 都是 1」推出
（`codegen_kernel.cpp:462`），再由下游决定 `need_gen_get_value_of_ub_scalar` /
`need_duplicate_value_of_ub_scalar`（`:1291-1316`），另有一张 `force_non_ub_scalar`
黑名单（`:1348`）。**这套不该照搬。** MLIR 侧这个信息是结构性的、不需要分析：
`Scalar` 是 `arith.constant`，`ScalarData` 是 `afir.tiling_param "<字段名>"`（这个 op
E0 已经有了，正是为「运行期尺寸」造的），`IndexExpr` 是循环归纳变量上的算术——
`LoopGenerate` 已经把轴变成真的 `scf.for`，IV 是现成 SSA 值。三条通道全都已经存在。

**该建的是 AFIR 那一层的操作数形态，这正是要先定的一件事：**「同一个 `afir.div`、
第二个操作数换成标量 SSA 值」，还是「另开一族 op（`tensor_scalar` / `scalar_tensor`
两个方向，非交换的 Sub / Div 两个都要）」。两种走法的 ODS、`getOperandPartition`
的三段划分、bridge 的折叠、conversion 的注册、测试全是两套写法，选错要整片重写。

**下游的量**：AscendC 侧 `ascendc.adds_l2` / `muls_l2` / `maxs_l2` / `mins_l2` 与
emitter 都已经有了；mem 一侧另有六个包装头（`scalar_{add,sub,mul,div,maximum,minimum}.h`），
reg-base 一侧一个都没有——于是这一族的家族划分与一元族不同，`ApiFamily` 那套要再走一遍。
而 `BinaryElementwise.cpp` 还停在 E0 的形态（位置下标切操作数、没有 tmp buffer 通路、
没有 `ApiFamily`），偏差 9 已经记过：给它做一遍 E2 给一元路径做过的改造是独立的一件事，
E3 与 E6 都压在它上面。

所以 E3 = 「二元路径的 E2 化」+「标量操作数形态定型」+「三条 producer 的折叠」，
是一个与 E0 同量级的波次，不是 E1/E2/E4 那种一晚上的接线。第一件已单独做掉（见 E2.5），
第二、三件的第一档在下面。

**形态定死为「同一个 op、第二个操作数换成标量 SSA 值」**，不另开一族 op。
`afir.add %tile, %tile` 与 `afir.add %tile, %scalar` 是同一个 op 的两种操作数形态，
由两个 implementation 接住（`add.tensor_tensor` / `add.tensor_scalar`）；选哪个由操作数
形态定死，两者的判据互斥（`getTensorTypes` 两个方向都挡），不靠优先级。
ODS 那一位是 `allowScalarRhs`，只放宽右操作数这一个位置；`AllElementTypesMatch` 对标量
仍然成立（它比的是 `getElementTypeOrSelf`），所以放宽这一位不会顺手放掉 `f16 + f32`。

**「标量是输入但不是 tile」这句话要在六处分别说出来**，少一处就错一处，而且六处的
表现各不相同。前四处写代码时就想到了，后两处是补调度形态的用例时被报错逼出来的——
说明这类「每个操作数都是一块缓冲」的隐含假设不容易靠通读找全：

- `inferElementwiseShape` 里**跳过**标量，而不是退回无秩。一个无秩的 shaped 操作数说
  「结果形状不知道」，一个标量什么都没说；混为一谈的表现是合法 IR 被 verifier 判死。
- `getBufferizedElementwiseEffects` 不给标量挂 `Read`——那会让 `arith.constant` 参与
  访存排序，说的是一件不存在的事。
- `verifyCanonicalBufferDescriptors` 的「不许混 memref 与遗留 shaped 类型」只对 shaped
  类型成立。标量两种形态下都是同一个值，算进去的话 buffer 形态整个建不起来。
- `verifyDescriptorList` 里标量**占一格但不检查**。占格是因为不占就跟操作数下标错开
  了——错开之后每条检查都还在跑，只是查的是别的操作数。
- `CanonicalBindingCollector::recordArray` 不给标量要 `afir.buffer_desc`：它不是一块
  存储，没有可绑定的资源。
- `CanonicalFunctionMemoryAllocator::rewriteCanonicalOperation` 把标量原样带过去，不去
  查它有没有物化出来的 `!ascendc.local_tensor`。

后两处的豁免都只对 `afir.operand_descs` 开——目的地和结果不可能是标量，那条兜底不该
一并撤掉。

标量操作数**只有 buffer 形态有**：tensor 形态的 `verifyLogicalForm` 要求所有操作数都是
`TensorType`，放过去报出来的是一句跟标量无关的话，所以 bridge 在那一侧直接拒。

bridge 那一半：`Scalar` 与 `Data` / `Output` 同类，不产生 AFIR op，只产生一个值。字面量
在 AscGraph 上是一串文本（legacy 直接把它拼进生成的 C++），这里按 dtype 解析成一个有
类型的常量——解析不了的写法在 bridge 就报出来，不留到设备编译器。

**没有 CPU E2E，原因不在 MLIR 这一半**：legacy 的 ATT tiling 对含 `Scalar` 节点的图
直接返回失败（`codegen_tiling.cpp:872` 的 `codegen_func_`，ATT 侧不留日志），发生在
`GenerateForInductor` 的 tiling 阶段，比 MLIR bridge 早——同一张图 legacy 自己的 kernel
codegen 是好的（生成出 `const float scalar_2 = 2.5;`）。换 1D / 3D、给 `Scalar` 节点补
`sched.axis` 与 `y.axis` 都不改变结果。所以 bridge 那条折叠改由 UT 直接钉，不绕 E2E：
`FoldsScalarNodeIntoATypedConstant` 钉 `AfirBuilder` 那一半（手填 `NodeInfo`），
`BridgesAGraphContainingAScalarNode` 从真实 `AscGraph` 出发走完 ScheduleToAfir——
后者独有的失败方式是 `Scalar` 的输出仍要过 `InitializeTensorValue` 的 position /
alloc_type 检查，而 `Scalar` 节点这两个字段是默认值。

出口（已达成）：`afir-scalar-binary-api.mlir`（两个 `--target` 同一份 IR，四个 op 各钉
一遍——「哪个 AFIR op 配哪个 AscendC op」是一张手写对照表，只验 add 的话把 Maxs 与 Mins
互换不会有任何用例变红）、`afir-scalar-binary-invalid.mlir`（三条，按谁拦住的分：ODS
一条、capability 两条）、`afir-scalar-binary-scheduled.mlir`（过循环生成 / 内存分配 /
队列绑定整条流水）、以及上面两条 bridge UT 加 `ParsesScalarLiteralsAtTheirDeclaredType`
/ `RejectsScalarLiteralsItCannotRepresent`（字面量解析，这一族的失败方式全是静默给出
另一个值）。

**已知没覆盖到的**：`materializeOperandTiles` 里 `ScheduledContiguous` 那一支对标量的
处理（两块 tile 加块内偏移、标量不加）。整个 lit 里没有任何用例到得了那一支——
tensor-tensor 形态靠 CPU E2E 覆盖它，而标量形态按上面的原因没有 E2E。

**下一档，按依赖顺序：**

1. `scalar_tensor` 方向（`afir.add %scalar, %tile`）。对这四个交换律 op 它只是操作数
   反过来，但 AFIR 现在**没有任何地方声明过交换律**，而在 bridge 里静默对调操作数要
   连 `afir.operand_descs` 一起对调——这是个该单独定的事，不是顺手加一行。
2. `ScalarData`（运行期标量）。通道是现成的 `afir.tiling_param "<字段名>"`，缺的是
   AscGraph 的 index 到 tiling 字段名的对应关系。
3. `IndexExpr`。要在 `LoopGenerate` 之后才有归纳变量，折叠不能发生在 bridge，得换个
   接入点。
4. Sub / Div 的标量形态。它们是这一族里唯一走包装的（`scalar_sub.h` / `scalar_div.h`，
   且只有 mem-based 一侧有），压在 Sub / Div 的 tensor-tensor 形态接通之后（E6）。

### E4　浮点谓词族　**已完成**

Isnan / IsFinite / IsInf 接通 buffer 形态。形态与 `afir.abs` 同档（标准调用在 reg-base
一侧、不要 tmp buffer；mem-based 一侧走包装、要一块），新的那一件事是**结果 dtype 与输入
不同**：输入 f16/f32，结果 ui8。这一条要在三处分别说出来，少一处就错一处：

- ODS 的 `sameElementType` 清零（`AllElementTypesMatch` 会把 `f32 -> ui8` 判死），
  同时 tensor 形态的结果元素类型要显式传给 `inferLogicalReturnType`——默认「取第一个
  输入的」对它们是错的，而错的表现是静默造出一个 `f32` 结果。
- capability 里「输入与目的地同元素类型」换成「输入浮点、结果一个字节宽」。两件事写在
  一起是因为它们互相约束：只查结果是 ui8 会把 `isnan(int32)` 放进来。
- emitter 的显式模板实参要取**源**的元素类型。`IsnanExtend<T>(LocalTensor<uint8_t> dst,
  LocalTensor<T> src, ...)` 的目的地恒为 uint8，按目的地取会实例化出参数对不上的重载。
  这一条只有真把 kernel 编出来才挡得住——所以 `load_isnan_store` 这个 E2E 不是锦上添花。

还有一处两侧不对称：标准 `AscendC::IsNan` 写的是 `LocalTensor<bool>`，包装收的是
`LocalTensor<uint8_t>`。lowering 在标准那一侧就地 `ReinterpretCast<bool>()`（legacy 是
先声明一个 `y_cast` 变量再下标，这里是先下标再 cast——两者都是一个字节，等价）。

**偏差 8：比较 6 个（Eq / Ne / Lt / Le / Gt / Ge）不在本波次。** 它们不是「多一个 op」
而是多一套调用形态：调用名两侧都是 `CompareExtend`，但模板实参不同（mem 是
`<T, CMPMODE>`，reg 是 `<T, dim, CMPMODE>`，`dim` 由 tile 的轴结构推出来），运行时参数
里还带 second-to-last stride——当前所有 AFIR conversion 用的都是「连续 tile + 元素个数」
那一种形态，表达不了。而且 mem 一侧要引用**两个**扩展头
（`compare.h` + `compare_v2.h`），`ApiExtensionHeader` 现在一个 op 只说得出一个。
（订正：这不是「第一个」，未接通的 31 个 op 里有 12 个引用两个头，最先撞上的是 E5 的
Select / Where / MaskedFill——清单见进度文档「前置 A」。）

**偏差 9：LogicalAnd / LogicalOr 归到二元路径的改造一起做。** 它们本身是干净的
（两侧都是包装，名字与 tmp 需求不同），但 `BinaryElementwise.cpp` 还停在 E0 的形态：
用位置下标切操作数、没有 tmp buffer 通路、没有 `ApiFamily`。给它做一遍 E2 给一元路径做过
的改造是一件独立的事，而 E6 的 Sub / Div / Axpy 要的是同一套——不值得为这两个 op 先做
一半。

**偏差 10：LogicalNot 与 Rsqrt 同因推迟。** `LogicalNot(dst, src, blk_tensor_with_value_1,
tmp_buf, size)` 比其余包装多一个预先填好 1 的 block 张量，当前「一个 op 至多一块
tmp buffer」的模型表达不了。它的 reg 一侧是干净的，但只注册半边比整个推迟更容易骗人。

顺手删掉了 `Basic/OpVecUnaryExt.td`：里头给这三个 op 的另一份建模从来没被 `Ops.td`
include 过，留着只会让 `scan_pyasc_coverage.py` 的清单出现重影。清单已重新生成。

出口（已达成）：`afir-predicate-{extension,stock}-api.mlir` 两份正例，负例
`afir-predicate-type-invalid.mlir` 三条——目的地 f32、输入 i32、源与目的地形状不同，
分别钉住上面那三处判据里的一处；`AllElementTypesMatch` 一关，parser 和 verifier 对这
几份 IR 都不再说话，只剩 capability 拦得住。CPU E2E 三份 `load_is{nan,finite,inf}_store`
各按 `std::is*` 逐元素比对（输入按四个一组喂有限值 / ±inf / NaN），并钉住模板实参取的
是源的 `float`——三份扩展头是三份互不相同的文本，只编一份挡不住另外两份。

---

## 验收

1. 共有的 47 个 elementwise 节点类型端到端生成 kernel，CPU E2E 数值正确，
   NPU E2E 与 legacy 一致。
2. bridge 白名单与 conversion 实现覆盖面在族内**相等**，由 gtest 断言两个集合相等，
   不靠人工核对。
3. `afir-verify-kernel-complete --criteria=c4 --target=<soc>` 在纯 elementwise
   kernel 上通过；C4 负例换成 `afir.sum`。
4. 新增一个 elementwise op 的改动点 ≤ 3 处，UT 钉住。
5. 三层对 soc 字面量与 `NpuArch::` 保持零命中。
6. `AF_ENABLE_MLIR_CODEGEN` 关闭时 legacy 输出逐字节不变；
   `build.sh --pkg -j 8` 与 `--no-autofuse` 成功。
7. `load_add_store` 端到端编译耗时回退 ≤ 5%。

---

## 风险

| # | 风险 | 处理 |
|---|---|---|
| R-1 | 并集口径下 elementwise 是 128 个而非 47 个 | 81 个 reg 独有项全落 D3 最简单一档（标准 API、无 tmp），单列 E8，可后移，不影响 E1～E7 验收 |
| R-2 | 扩展 API 通道是新机制，`Translation.cpp:76` 说明此前没人正面解决过 | E0-2 先于所有 op 波次；用现成的 `datacopy.h` 验证 |
| R-3 | `*Ext` op 全是空壳（无 emitter、apiName 有错、命名空间未定） | E0-5 在 `abs_extend` 上定模板；已有空壳 op 按对照表逐个核对 apiName，不盲信 |
| R-4 | D2 判定要逐 op 比对四项 | 已由 E0-1 脚本生成对照表；与实现注册表的一致性再由 UT 交叉校验 |
| R-5 | `Cast` 的 dtype 组合多，且 `cast.h` 是 D1 同名头 | 超期就拆 E6-b |
| R-6 | 8 KB 下限在 `!HasPrintableExpr` 分支拿不到 | legacy 既有行为，MLIR 只引用字段不重复施加。E0-5 补 ST 观察是否在真实图出现，出现则单独提 issue |

---

## 编码红线要点

- **通用-2/3**：tmp buffer 的 `life_time_axis_id` / `buffer_id` / `alignment_bytes`
  来自 scheduler，用前校验（复用 `CollectTmpBufferLifetimes` 口径）。尺寸不进 AFIR，
  对齐与溢出走 `alignResourceSize` 已有的 `ceildivsi` + `mulsi_extended` + 两条
  `cf.assert`。
- **Graph-2**：变体选择一律经 `TargetFeature`；stable id 禁含平台名；
  `ImplementationKind` 不加 `RegBase`。本次顺带消除 `Translation.cpp:76` 一处已有违反。
- **Graph-4**：`SelectImplementation` 与 `MemoryAllocate` 的顺序依赖必须写进
  `AFIRToAscendCPipeline.cpp` 注释——「要不要 tmp、要多大」是**所选实现**的属性
  （`CalcAbsTmpSize` 按 dtype 分支；`Abs` 在 mem 要 tmp、reg 不要）。本期两者都由
  上游按 target 给定所以不显形；Schedule 迁移后 Select 必须前移到 MemoryAllocate 之前。
- **Graph-5**：扩展头按字典序，候选按注册序，不依赖哈希序。
- **Graph-6**：逐 op 的选择与 pattern 匹配日志走 `LLVM_DEBUG`，默认关闭。
