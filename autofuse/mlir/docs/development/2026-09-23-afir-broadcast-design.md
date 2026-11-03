# L-7　`afir.broadcast` 接通断头路——设计

**日期**：2026-09-23　**对应清单项**：L-7（`2026-09-22-afir-elementwise-remaining-issues.md`）

本文只写判断和证据。**清单里 L-7 那一节记的三条前置有两条不成立，记的「真卡点」也只说对了
一半**，改写见「二、探针结论」；模板中范围 / 非功能需求 / 安全检查等章节从简。

---

## 一、目的与范围

`Broadcast` 是 bridge buffer 形态白名单里唯一没有 conversion 实现的 op，含它的图整张回落
legacy，并由 `BufferFormSetIsCoveredByAscendCImplementations` 的豁免名单显式钉住。

**本次做**：`afir.broadcast` 的**常量源**一档（legacy 的 `Duplicate` 分支），两侧各一个
implementation，删掉豁免名单那一行，补 bridge UT 与 lit。

**本次不做**（各自被独立证据挡住，见 2.4）：legacy 的三轴 / 四轴 `Broadcast(...)` 调用，
以及 **CPU E2E**——今天一条广播图都到不了 CPU E2E 的执行那一步，三处卡点见 2.4
（**2026-09-29 订正**：这一句只对分段那一档成立，`Duplicate` 那一档已补 E2E，见 2.4 的两处订正）。

涉及组件：Autofuse MLIR（dialect ODS、conversion、测试）。不动 `autofuse/` 下 MLIR 以外的文件。

---

## 二、探针结论（动产品代码之前跑的）

探针做法：`test/e2e/cpu/cases/load_broadcast_store` 建图 → `optimize::Optimizer` +
`codegen::Codegen` → 看调度后的 AscGraph 与 AFIR。共跑 8 种形状。

### 2.1 供给侧全都在，「两侧各新建两个 td def」不成立

进度文档「前置 C」把 `Broadcast` 列为「两侧都缺 td def」的最后一行。实际这一档一个 def 都不用建：

| 要什么 | 现成的 | 位置 |
|---|---|---|
| `AscendC::Duplicate(dst, scalar, calCount)` | `AscendC_DuplicateL2Op` | `Basic/OpVecDuplicate.td:37` |
| `src.GetValue(0)` | `AscendC_LocalTensorGetValueOp` | `Core/Tensor.td:252` |
| `SetFlag` / `WaitFlag` + `MTE2_S` / `V_S` | `SetFlagOp` / `WaitFlagOp`，枚举已含两档 | `Basic/OpBlockSync.td:47`、`Core/Attributes.td:202,204` |

顺带：三轴 / 四轴那一档要的 `AscendC::Broadcast<T, dim, axis>` 也已经有
（`AscendC_BroadcastL2Op`，`Autofuse/OpVecBroadcast.td`），**而且它的手写 emitter
`Autofuse/VecBroadcast.cpp` 已经会在调用前打两条局部 `uint32_t[]` 声明**——L-7 记的「reg 侧
emitter 必须先打两条局部数组声明」这条前置同样不成立。

`asc_graph_builder.h` 的 `Broadcast` 方法本来就有（`:105` 起三个重载），任务书已指出。

### 2.2 ODS 的类型约束挡在最前面（真卡点，一行）

`AFIR_BroadcastOp` 用的是 `TensorOrMemRef<...>`，而**其余每一个可 bufferize 的 op 用的都是
`AFIR_BufferizableShapedType<...>`**（`Ops/Elementwise.td:167`、`Ops/Cast.td:42`、`Ops/Data.td:38` …）。
两者差的正是 `AFIR_TargetShapedType`——下降途中 memref 会先变成 `!ascendc.local_tensor<...>`，
而那时 op 本身还没被重写。

探针实测，没这一行连 verifier 都过不去：

```
error: 'afir.broadcast' op operand #0 must be Tensor or MemRef of ... but got '!ascendc.local_tensor<1x1x1xf32>'
error: canonical AFIR operation failed verification after AscendC materialization
```

改成 `AFIR_BufferizableShapedType` 之后，整条 pipeline 一路走到 conversion 才停：

```
error: failed to legalize operation 'afir.broadcast'
```

`05_ascend-select-implementation.mlir` 里 `afir.broadcast` 是唯一没有 `ascend.implementation`
的 op；descriptor、tmp buffer、循环生成、内存分配全部就绪。**这就是「只差一个 implementation」
的现场。**

### 2.3 `TransferPlan` 只建模一根向量化轴——成立，但它只挡住三轴 / 四轴那一档

调度后的 `TensorDescAttr` 逐轴信息是全的（`axis` / `repeats` / `strides` /
`vectorized_axis` / `vectorized_strides`），legacy 的 `DimensionCollapse` 读的就是这几项。
问题只在 `getVectorizedAxis()` 取 `.front()`、`ScheduledTileIndices` 只给一个 `count`。

**两轴广播的 tile 确实是二维的**（`x8x128` 的 `AutoFuseAfterOptimize`）：

```
[1] load0 .y .vectorized = {z0t:0, z1:1}        # 广播轴 z0t 的 stride 是 0
[2] brc   .y .vectorized = {z0t:128, z1:1}
```

但**一根向量化轴时，legacy 能走到的分支只有 `Duplicate` 一条**，这是可以证明的：
`BroadcastApiCall::Generate` 先查 `IsBroadcastConstantTensor(x)`——它要求 x 的**每一根**
向量化轴 repeat 都是 1。只有一根轴时，「这根轴被广播」等价于「所有向量化轴 repeat 都是 1」，
于是必然先命中 `Duplicate`；`broadcast_num == 1` 的三轴调用永远够不着。
`broadcast_num == 0` 的 `DataCopy` 分支 legacy 自己注释说调度会把这种节点删掉
（探针里 `RemoveRedundantBroadcastNode` 确实删了，见 2.4）。

**所以「一根向量化轴」与「`Duplicate` 一档」是同一条边界**，不是我挑出来的切口。

### 2.4 真正挡住 CPU E2E 的是另外四件事，都不在 L-7 里

**(a) 计划选择：只要有 2 个以上 plan 就选不出来。** 探针八种形状的 plan 数：

| 图 | plan 数 |
|---|---|
| `[1] → [128]`（一维） | **1** |
| `[1,32]→[4,32]`、`[4,1]→[4,32]`、`[1,128]→[8,128]`、`[8,1]→[8,128]`、`[1,128]→[1024,128]`、`[1,2048]→[2,2048]`、`[1,64]→[64,64]` | 2 |
| `[4,1,32]→[4,8,32]` | 3 |
| `load_add_store_128x23x5` / `_128x128x128`（无广播，对照组） | 1 |

plan 唯一时 `HasUniqueCandidatePerSubgraph` 直接放行；不唯一就落到
`SelectWinnersFromModels` 的 ATT 代价模型，而**真实模型里有 ATT 求解器自己的未知量**，
代不进去（`AttCostEvaluator.cpp` 临时埋点打出来的原文）：

```
[PROBE] non-const cost: ((15.89 * block_dim) + 882.09 + load0_Load_AIV_MTE2_contrib)
```

于是一律报 `ATT did not provide a complete, constant-cost plan for subgraph 0`。
`AttCostEvaluatorTest` 喂的全是合成的常量 `ModelInfo`，所以这条路**在真实图上一次都没跑通过**；
对照组说明这不是广播引入的缺陷，而是广播是第一个把 tiling 变成真选择题的 op。

> **订正（2026-09-29 重测）：卡的是「模板不唯一」，不是「plan 不唯一」。**
>
> 2026-09-29 用 CPU E2E 用例重测 `[1,8] → [128,8]`（形态 A）与 `[8,1] → [8,128]`（形态 B），
> 两个形状的 **plan 都是唯一的**，不唯一的是一个 stage 里的模板：`B0Y0`（广播轴在 tile 内、
> 带 brc cache）与 `B0Y1`（广播轴排到 tile 外、`RemoveRedundantBroadcastNode` 把 `brc` 删掉）
> 同时在。`HasUniqueCandidatePerSubgraph`（`PlanSelector.cpp:37`）要求每个 stage 恰好一个模板，
> 于是照样落到 ATT 代价模型、照样报同一句（`AttCostEvaluator.cpp:162`）。
>
> 也就是说 (a) 与 (d) 是同一件事——**模板不唯一**——而与 plan 数无关。上面那张表量的是
> plan 数，对本条卡点不是判据。重测详情见
> `2026-09-28-afir-vectorized-axis-collapse-design.md` 的「广播 CPU E2E 重测（2026-09-29）」。

**(b) `exec_condition` 整个没建模。** 留下来的那个 plan（Y0）的 Load 与 Broadcast 都带
`exec_condition = cache_block_split_fused_brc_axis`（brc cache：整块 UB 只算一次，
Store 每次迭代写一片）。`autofuse/mlir` 全树 grep `exec_condition` / `ExecuteCondition`
**零命中**，bridge 既不读也不拒。照搬会静默丢掉这个条件。

> **订正（2026-09-29 重测）：丢掉它不会算错，只是漏掉一次优化。** 强制选中 `Y0` 之后，
> 形态 A / 形态 B 各两个形状（含必须切多趟的 `[1,128] → [1024,128]` 与 `[1024,1] → [1024,128]`）
> 在 CPU 模拟器上逐元素全对，而那几张图的 `load0` / `brc` 上 `exec_condition` 确实是
> `kCacheBlockSplitFusedBroadcastAxis`。这个条件说的是「整块 UB 只算一次、可以复用」，
> 不读它等于每趟重算——慢，不错。所以 (b) 不该与 (a) 并列成「挡住 CPU E2E」的一件事。

> **已落地（2026-09-30）：`exec_condition` 现在进 `afir.sched`，守卫由 `afir-exec-condition`
> pass 包成 `scf.if`。** 落点与验收见
> `2026-09-28-afir-vectorized-axis-collapse-design.md` 的「下一步」第 1 条。

另一半 plan（Y1）把广播轴排到 tile 外，`RemoveRedundantBroadcastNode` 把 `brc` 整个删掉，
图退化成 `Load → Store`——**那条路今天就能下降，一行 Broadcast 代码都不用**。

**(c) 唯一能到 MLIR codegen 的那个形状，Load 发不出合法的搬运。** 一维 `[1] → [128]` 是
探针里唯一 plan 唯一、模板也唯一的形状，它一路走到了生成 kernel。但常量源广播的源 tile
**恒为 1 个元素**（判据就是「每根向量化轴上 repeat 都是 1」），于是喂它的 Load 要从 GM 搬
4 个字节，而 `AscendC::DataCopy` 要求元素数 × sizeof(T) 32B 对齐。CPU 模拟器当场报出来：

```
[ERROR][DataCopyCheck] count * sizeof(T) must be 32B align, current count value is 1.
In NPU mode, no error is reported. The value is rounded down by 32B.
```

「向下取整到 0」——也就是一个元素都不搬，Duplicate 铺的是没初始化的内容。

legacy 不撞这个：它的 Load 一律发 `DataCopyPadExtend`（`LoadApiCall::Generate`，v1 / v2 共用）。
MLIR 这边也有这条路（`DataMovementMapping::Api::DataCopyPadExtend`），但由
`TargetFeature::DataCopyPadExtend` 把着，而那个门写的是「reg-base 家族才有这个 wrapper」
（`AscendTargetProfile.cpp:114`）。**这句话与事实不符**：`DataCopyPadExtend` 定义在
`autofuse/ascendc/api/datacopy.h`——mem 侧的供给目录（`api/`，不是 `api_regbase/`），
而 v1 的 `LoadApiCall` 正在用它。

> **订正（2026-09-23 当天）：这一条已修，而且它不是广播独有的。**
>
> 写「不在本次范围里改」时，把这个门当成了只挡住广播那一个 1 元素 tile。其实**任何**
> 总量切不齐的图都撞它：tile 的元素个数是运行期 TilingData 字段，尾块与整块走同一条
> 语句，尾块就是不对齐的那一趟。此前探不到只因为 CPU E2E 的形状全是 `[128]`（f32 512B，
> 恒对齐）。把 44 条 `[128]` 换成 2~4 轴当场红 26 条，全是这一个根因。
>
> 修法：`DataMovement.cpp` 一律发 `DataCopyPadExtend`，`TargetFeature::DataCopyPadExtend`
> 整条删掉（它只有这一个消费者，而它的判断本身是错的）。
>
> **原先估的代价也是错的**：不是「40 多个 mem 侧 E2E 与一批 lit 的文本全部改变」——实际是
> 2 个 lit 文件，E2E 的 `CheckGeneratedKernel` 一个都没查。**「会牵连很多」这种估计，grep
> 一遍再写**；而且 `DataCopy`（C++ 文本）与 `data_copy_l2`（方言 op 名）两种拼法都要搜，
> 第一次数成 1 个就是漏了后者。
>
> 对广播这一档的影响：(c) 不再是卡点，但 (a) / (b) / (d) 仍在，所以这一档照旧没有 CPU E2E。

**(d) reg 侧同样到不了**：同一张图在 `Ascend950PR_9599` 上 plan 唯一，但模板有两个
（多出一个 `nddma`），`HasUniqueCandidatePerSubgraph` 要求每个 stage 恰好一个模板，于是
一样落到 (a) 那条死路。对照组 `load_neg_reg_base_store` 只有一个模板。

**结论**：本次不新增 CPU E2E，不是省略，是**今天没有一条广播图跑得到那一步**。
把 `TransferPlan` 补成逐轴之后仍然如此。(a) (b) 两件各自比 L-7 大，应各自立项（(c) 已修，见上）；
在它们之前做逐轴 `TransferPlan`，是给一条走不通的路铺砖。

> **订正（2026-09-29 重测）**：「没有一条广播图跑得到那一步」**只对分段那一档成立**。
> `Duplicate` 那一档（源只有一个元素）在 (c) 修掉之后就跑得通，且算得对——写上面那段时
> (c) 已经修了，但没有重测。2026-09-29 已补两条 CPU E2E
> （`load_broadcast_duplicate_store_128` / `_8x128`）。分段那一档仍卡在 (a)；
> (b) 不是卡点：强制选中 `B0Y0` 之后两种形态都跑通且算得对，`exec_condition` 被忽略只影响
> 性能。`B0Y0` 这一侧的唯一卡点是 (a)；`B0Y1` 那一侧另有两个独立缺口（`MTE2 → MTE3`
> 同步未建模、向量化轴被剔空时 GM 偏移无条件取 0 导致静默算错）。已立成清单，见
> `2026-09-28-afir-vectorized-axis-collapse-design.md` 的「遗留问题清单（2026-09-29 立）」：
> **(a)、静默算错那条、「把 GM 偏移真算出来」与它的强度削减（B-1 / B-3 / B-5 / B-7）当天已修**，
> 连同 `MTE2 → MTE3`（B-2）一起，**形态 A 与形态 B 都有了 CPU E2E**（共三条），
> 而既有 62 条用例的生成 kernel 逐字未变。清单里 B-1 ~ B-7 全部已修。

---

## 三、功能需求

### 需求 1：`afir.broadcast` 常量源一档的 conversion 实现

1. **行为**：源张量在所有向量化轴上 repeat 均为 1 时，发
   `AscendC::Duplicate(dst[0], src.GetValue(0), count)`，前置一对标量读事件。
2. **输入**：`afir.broadcast`，一个输入、一个 destination、mem 侧另有一块 tmp buffer；
   `TransferPlan` 为 `ScheduledContiguous` 或 `Identity`。
3. **处理**：
   - 判据：`operand_descs[0]` 的 `vectorized_axis` 非空，且每一根轴对应的 `repeats` 静态等于 1；
     `result_descs[0]` 与之同 `vectorized_axis`。不满足 → `unsupported(UnsupportedShape)`，整张图回落 legacy（今天的行为）。
   - `count`：`ScheduledContiguous` 走 `materializeScheduledTileIndices` 的 `count`；
     `Identity` 走 `LocalTensorGetSizeOp`——与一元 / 二元 elementwise 同一条路。
   - 标量读同步：照 legacy `BroadcastScalar`，源在 `VECIN` 队列上发 `MTE2_S`、在 `VECCALC` 上发 `V_S`
     的 `FetchEventID` + `SetFlag` + `WaitFlag` 三件套。**这一对必须由本 pattern 自己发**：
     `AFIRPipelineSync` 按 V 管线配对，给不出 S 管线的依赖。
4. **输出**：kernel 文本里一条 `AscendC::Duplicate(dst, v, n)` 加前置事件对；无模板实参、无扩展头。

### 需求 2：两侧各注册一个 implementation

两侧发的调用逐字相同——legacy 的 `BroadcastRegApiCall::Generate` 在 `IsBroadcastConstantTensor`
为真时同样调 `BroadcastScalar`。差别只有一处：**mem 侧调度给 tmp buffer，reg 侧不给**
（`BroadcastAscIrCodegenImpl::CalcTmpBufSize → CalcBroadCastTmpSize` vs
`BroadcastAscIrCodegenImplV2` 没有这个覆写）。

`checkTmpBuffer` 拿 `needsTmpBuffer` 与调度给的块数严格对等，所以**不能用一个 `ApiFamily::Any`
盖住两侧**，要 `MemBased`（`needsTmpBuffer = true`）+ `RegBased`（`false`）两个注册。
这样 `EveryElementwiseOpCoversBothApiFamilies` 按 `mem && reg` 满足。

mem 侧那块 tmp buffer **申请了但不发给调用**——与 legacy 一致：`BroadcastScalar` 只在
int64 / uint64 时才把它接到参数尾巴上（`int64_tmp_buf`），本档收的 dtype 不含 64 位整数。

### 需求 3：删掉豁免名单那一行

`BufferFormSetIsCoveredByAscendCImplementations` 里 `knownWithoutImplementation = {"Broadcast"}`
那一行删掉。**不删它自带的反向断言会红**（「名字还在表里却已经有实现了」），这是设计好的。

### 需求 4：元素类型集合

`AFIR_BroadcastElementTypes` 已有 `[I8, UI8, I16, UI16, I32, UI32, UI64, F16, F32]`。
本档注册处收 `{f16, f32, i16, i32, ui16}`。**判据是「legacy 那句不带限定的 `Duplicate`
到底绑到谁」，不是供给表的 dtype 列**：`BroadcastScalar` 发的是 `Duplicate(dst, v, n)`，
而 `duplicate.h` 在全局作用域给 int8 / uint8 / int64 / uint64 各摆了一条非模板重载，这四种
绑的是那条包装（8 位先 `ReinterpretCast<uint16_t>`，64 位还要 tmp buffer），其余的靠 ADL
绑到 `AscendC::Duplicate`——正是本档发的 `DuplicateL2Op`。

AFIR 能建模的六位里落在那四条包装上的只有 `ui8`，于是本档比 legacy 窄这一位。
`ElementTypeMatrixTest` 的判据是「不得比 legacy 窄」，按它自己写的办法处理：加一张
`{id, 注册的, 按表算出来的}` 的快照表 + 「表里的 id 必须真的还窄着」的反向断言，**循环本身
一点没放宽**。要收 `ui8` 得另起一个带 `duplicate.h` 扩展头、发包装调用的 implementation，
不在本次范围。按 `UnaryElementwise.cpp` 注释处的规矩，集合是每个 op 自己的事实，不设缺省。

### 需求 5：`materializeScheduledTileIndices` 认得「向量化轴上 stride 为 0」

**这一条是实现过程中被探针逼出来的，不在原计划里，但没有它这个 op 接通即错。**

> **实现位置已迁移（2026-09-28）。** 本节描述的守卫写在 `materializeScheduledTileIndices`
> 里，而二维寻址落地之后这个函数已不存在（拆成了 `emitScheduledTransfer`，单根
> `plan.vectorizedAxis` 换成多项 `plan.extents`）。按本节改是改不动的。
>
> 需求本身仍然成立，但**根因消失了**：这条守卫挡的是「按轴长算长度」，而新路径的长度
> 本来就取自张量自己的 `repeats`——也就是本节说 legacy 不会错的那个做法。于是它退化成
> 一个更小的问题：广播源的 `repeats` 全是 1、两侧 stride 全是 0，会被有效视图归约的
> 规则 1（范围为 1 的轴不占一维）整个剔光，`extents` 变空导致计划失败，op 从「能跑」
> 退回「不支持」。
>
> 今天的落点是 `reduceEffectiveView`（`dialect/AFIR/Analysis/TransferPlan.cpp`）剔光
> 之后的那一档：取最内那根向量化轴的 repeat 与它自己的两侧 stride 造一项，于是
> `blockLen` 取那个 repeat、外层不发循环、GM 偏移的每一项都被剔零跳过——与本节要求的
> 「偏移恒为 0、长度取 tile 自己的 repeat」逐字同解。

`materializeScheduledTileIndices` 按 `tileIndex × tileSize` 算 GM 偏移、按 `tileSize` 算元素数。
这条式子的前提是「这块 tile 每个迭代前进一整段、长度就是轴长」。**广播的源不满足**：它在
向量化轴上 stride 是 0（每个迭代读同一份）、repeat 是 1。照原式子算，喂它的 Load 会从一个
只有 1 个元素的 GM 缓冲里搬 `tileSize` 个、还带上迭代偏移——**静默越界**，而且越界的是 Load，
不是 Broadcast。

接通 Broadcast 之前这条路走不到（含 Broadcast 的图整张回落 legacy），所以今天不是活的缺陷；
接通之后它是第一个撞上的。

改法（`TransferPlan.cpp`）：读局部侧 descriptor 的 `vectorized_strides[0]`，

- 字面量 `1`：原样，今天所有用例走这条；
- 字面量 `0`：偏移取 0（并要求 GM 侧 descriptor 的 strides 全是字面量 0，否则「偏移恒为 0」
  这句话没有依据），元素数取这块 tile 自己在该轴上的 repeat；
- 其余（含符号表达式）：**报失败**。原先这一档也是按 stride 1 算的，照算的后果是静默搬错
  地址；今天没有用例走到，报失败比猜对得多。

legacy 对应的事实是：它的偏移与长度都取自**张量自己的** descriptor
（`LoadApiCall::Generate` 的 `tiler.Offset(current_axis, ub.axis, ub.axis_strides)` 与
`CalculateDmaParams(tpipe, ub, ub, ...)`），不是取自轴长。这里补的是同一条事实里今天唯一
用得到的那一档，**不是**把 `TransferPlan` 改成逐轴。

---

## 四、非功能需求（从简）

- **可维护性**：落点是 `conversion/AFIRToAscendC/Implementations/` 下新增
  `BroadcastElementwise.{h,cpp}`，复用 `ElementwiseApiFamily.h` 的 `checkApiFamily` /
  `checkTmpBuffer` / `hasCanonicalDescriptors`，不新增 pass、不改 AFIR 操作数三段划分、不动 bridge。
- **可测试性**：三层分工见第六节。
- **可移植性**：不新增芯片判断；两侧由 `ApiFamily` 查 TargetProfile 决定，与既有 op 同路。
- **可靠性**：判据不满足时返回 `CapabilitySupport::unsupported`，图整张回落 legacy——与今天等价，
  不会静默出错值。
- **特性交叉影响**：见第七节。

## 五、性能

编译期：新增一个 implementation 的注册与一次 descriptor 扫描，O(向量化轴数)，可忽略。
执行期：发的调用与 legacy 逐字相同，无差异。内存：mem 侧那块 tmp buffer 本来就由 ATT 算进预算，
不新增占用。无新增默认日志。

## 六、测试设计

每条判据都要有一个会变红的东西：

| 层 | 钉什么 | 用例 | 拿掉实现会怎样 |
|---|---|---|---|
| bridge UT | `Broadcast` 不再在豁免名单里 | `BufferFormSetIsCoveredByAscendCImplementations`（豁免表清空）与 `EveryElementwiseOpCoversBothApiFamilies` | 清空豁免表后没有实现 → 第一条循环红；只注册一侧 → 第二条红 |
| lit（mem 正例） | AFIR → AscendC 文本：`MTE2_S` 事件对 + `GetValue(0)` + `AscendC::Duplicate(`，且**不出现** `Broadcast` | `afir-broadcast-duplicate.mlir` | 少发事件对、发成 `Broadcast`、或 legalize 不掉都红 |
| lit（reg 正例） | 同一个调用在 reg 侧也发得出，且选中的是 `..._reg_base` 那个 id | `afir-broadcast-duplicate-reg-base.mlir` | reg 侧没注册 → 选不出实现，红 |
| lit（两侧反向） | 带 tmp 的那份在 reg 侧选不出、不带 tmp 的那份在 mem 侧选不出 | 上面两份各一条 `not af-opt ... | FileCheck --check-prefix=WRONGSIDE` | 把两个 implementation 合成一个 `Any` → 两条都红 |
| UT（元素类型） | 注册的集合与 `ascir_op_lowering_matrix.csv` 对账，`ui8` 那一位的窄差是钉死的快照而不是豁免 | `ElementTypeMatrixTest`（`kKnownNarrower` 两行 + 反向断言） | 注册处集合改了 / 对照表改了 / 窄差补齐了却忘了删行——三样都红 |
| lit（反例） | 两根向量化轴时报 unsupported，不被 legalize | `afir-broadcast-multi-axis-unsupported.mlir` | 判据放宽 → 发出一条会静默算错的 `Duplicate`，红 |

**本次没有 CPU E2E**，理由见 2.4——不是没写，是今天没有一条广播图跑得到那一步。三层分工里
「算得对」这一层因此缺位；补上它的前置是 2.4 的 (a) 或 (c)，两条都在本次范围之外。
（**2026-09-29 已补**：(c) 修掉之后 `Duplicate` 那一档跑得通，见 2.4 结论处的订正。）
需求 5（stride 0 那一档）由 mem 正例那份 lit 覆盖：没有它，那份用例里 Load 发出来的
`DataCopy` 会带上迭代偏移、元素数取 `zt_size`。

测试命令：

```bash
bash ./mlir-test.sh
```

## 七、特性交叉影响（`cross_feature_check.md` 逐项）

| 场景 | 适用性 | 说明 |
|---|---|---|
| SuperKernel Python 接口 | 不适用 | 不碰 `super_kernel/` |
| SuperKernel C++/AOT 接口 | 不适用 | 同上 |
| Autofuse 图优化 | 不适用 | 不新增 / 不调整 pass，不改图改写；调度侧一字未动 |
| Autofuse Codegen/Backend | **适用** | 新增一个 AFIR→AscendC implementation，并收紧 `materializeScheduledTileIndices`（需求 5）。**收紧影响所有 Load / Store**：今天全部走 stride 1 那一档，输出逐字不变；stride 既非 0 也非 1 的从「按 1 算」变成报失败，回落 legacy |
| AscendC API / Runtime 交互 | **适用** | 新发 `Duplicate` / `GetValue` / `SetFlag` / `WaitFlag`。事件号走 `TPipe::FetchEventID`，与 `LowerEventRecordOp` 同一条既有路径；`SetFlag` / `WaitFlag` 成对、同一基本块内发出 |
| Python/C++ 混合绑定 | 不适用 | 不碰 `py_module` |
| 构建与打包 | **适用（仅源文件表）** | `conversion/AFIRToAscendC/CMakeLists.txt` 加一个 `.cpp`；不改安装路径与产物 |
| 测试与覆盖率 | **适用** | 见第六节 |
| 性能与日志 | 适用但无影响 | 见第五节 |
| 兼容性 | 不适用 | 无对外接口、脚本参数、产物布局变化。ODS 约束是**放宽**，原先合法的 IR 仍合法 |

## 八、安全检查（`编码红线.md`）

- 敏感信息 / 芯片类型硬编码：无。两侧靠 `ApiFamily` 查能力，不出现 SoC 字符串。
- 外部输入作索引 / 长度：`count` 沿用 `materializeScheduledTileIndices` 的产出，
  `GetValue` 的下标是字面量 0；判据先确认 `vectorized_axis` 非空、`axis` 与 `repeats` 等长、
  轴 id 在 `axis` 里找得到，再按下标取用。需求 5 新增的那一档只接受字面量，且要求 `> 0`。
- 整数溢出 / 除 0：本档不做新的尺寸算术；需求 5 的元素数是从 descriptor 读出的字面量。
- 资源生命周期：不申请资源；tmp buffer 由调度分配、`MemoryAllocate` 绑定。
- ABI/API：无对外接口变化。
- 图改写等价性与确定性：不改图；判据遍历 `vectorized_axis` 数组，顺序由 IR 决定，无哈希序依赖。
- 修改范围：不含与本需求无关的重构与格式化。

## 九、验收标准

1. `afir.broadcast` 常量源一档在两侧各有一个 implementation，发的调用与 legacy `BroadcastScalar` 逐字相同。
2. `knownWithoutImplementation` 空掉，`BufferFormSetIsCoveredByAscendCImplementations` 与
   `EveryElementwiseOpCoversBothApiFamilies` 全绿。
3. 新增 lit 三份（mem 正例 / reg 正例 / 多轴反例）全绿。
4. `bash ./mlir-test.sh` 全量绿；**CPU E2E 计数仍是 47**，理由见 2.4。
5. 清单 L-7 与进度文档按第二节改写：原先记的三条前置里两条划掉，换成 2.4 的四条真卡点，
   每条各自留下落点。
