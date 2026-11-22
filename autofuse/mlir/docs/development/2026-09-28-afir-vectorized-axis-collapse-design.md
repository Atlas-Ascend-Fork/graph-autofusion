# 向量化轴分段统一：取消 Broadcast 的多轴打包——设计

**日期**：2026-09-28
**前序**：`2026-09-23-afir-broadcast-design.md`（L-7，常量源 `Duplicate` 一档已接通）
**探针**：`scratchpad/brcprobe`（单文件 tikicpulib，插桩副本追踪原语）与 `scratchpad/copyslack`（P6），结论见「探针结论」一节

---

# 简介

## 目的

把 legacy 阶段 2 合轴的**两份实现**（`GenerateVectorizedAxisMergeStatus` 与 Broadcast 专用的
`DimensionCollapse`）在 MLIR 侧统一成一份，并按统一后的模型下降 `afir.broadcast`：
**外层普通段生成 `scf.for`，最内两段交给固定形态的两轴 AscendC API**。

由此达到三个可验证的结果：

1. 后端 API 的参数个数不再随张量秩增长——legacy 的三轴 / 四轴 `Broadcast` 形态在 MLIR 路线上不再需要；
2. **`Or` 模拟广播从 MLIR 发出的调用里结构性消失**（判据与证据见「探针结论」P2 与「功能需求 7」）；
3. elementwise 与 broadcast 共用同一份分段分析，后续 op 复用同一扩展点。

目标读者：Autofuse MLIR codegen 开发与检视者。

## 范围

**本次做**：

- `dialect/AFIR/Analysis/` 新增向量化轴分段分析，取代 `TransferPlan` 的单轴假设；
- `conversion/AFIRToAscendC/` 新增一份共享的外层循环生成 helper；
- `afir.broadcast` 的**单广播段**一档（两种段方向）在两侧各注册 implementation；
- `afir.broadcast` 的**多广播段**分解（N 个广播段 → N 次两段调用），覆盖面严格超过 legacy 四轴；
- 一元 / 二元 / 三元 elementwise 改用新分析，**单段时生成文本逐字不变**。

**本次不做**，各有独立理由：

| 不做 | 理由 |
|---|---|
| `RemovePad` 建模 | 它解决的是非对齐小尾轴下 MTE3 带宽利用率低的问题，与合轴正交，独立 PR |
| camodel 性能验证 | 部件齐备（`RUN_MODE=sim` + `ccec` + `libruntime_camodel.so`），但它是第二条并行的构建 + 运行路径，独立 PR。见「性能」的执行性能 |
| 计划选择 (a) 与 `exec_condition` (b) | 前序文档 2.4，各自比本需求大，且都不是合轴能解的 |
| reduce / transpose / gather / concat / split 的分段 | **每个 op 单独分析、单独支持**。分段分析可复用，但段语义各不相同（reduce 是「累积」而非「重复发同一条 API」，循环生成 helper 需新增一档），一起做等于把几件事的判据混在一轮里 |
| ~~Load / Store 的分段~~ | **本次已做**，见下：不做它的话广播图连 conversion 都跑不到 |

**关于 Load / Store（范围内的一次扩张，理由如下）**：接口设计一节原本只打算保留
`materializeScheduledTileIndices` 的 0 / 1 分流给 GM 偏移用。实现中发现这条保留有一个没预料到
的后果——**任何多段图今天都跑不到 conversion**：

- Load 落在 `vectorized_strides.front() == 0` 那一档（广播源的最外段就是广播段），那一档要求
  GM 侧 stride 全为 0，而广播源的 GM 输入是一段连续数据；元素数又取 `plan.vectorizedAxis`
  上的 repeat，在广播轴上是 1 而不是实际长度；
- Store 走同一个函数，它的 `front()` 是内段元素数（例如 8）——既非 0 也非 1，直接报失败；
- 而**一个 SSA 值上的 descriptor 必须一致**（`InvalidBinding: conflicting canonical
  descriptors`），所以也不能给 Load / Store 单独换一份单轴 descriptor 绕过去。

病根不是「广播轴 stride 为 0」这个约定——**那个约定是对的，而且正是让两边都不需要特判的
那一个**。legacy 里它在两处各自自然消解：`Tiler::Offset`（`codegen/codegen_kernel.cpp:639`）
就是一个 `Σ 轴下标 × stride` 的点积，`stride == 0` 那一支只是少打印一项；
`CalculateDmaParams`（`codegen/api_call/utils/api_call_utils.cpp:116`）把 gm 与 ub 两侧 stride
都为 0 的轴整根丢出搬运形状。**病根是 MLIR 这一侧把 stride 当分类标签用，再从轴长去取尺寸。**

所以这一次一起改了，判据与 legacy 对齐：

| | 改前 | 改后 |
|---|---|---|
| GM 偏移 | `tile 序号 × findAxisValue(切分轴, AxisSize)` | `tile 序号 × 这一段按整块算的元素数`；切分轴不在这一段里时恒 0 |
| 元素数 | 按 `front()` 分流取轴长或广播轴上的 repeat | 丢掉 stride 为 0 的段之后那一段的 extent 之积（含尾块修正） |
| `front()` 的 0 / 1 分流 | 有 | **整段删除** |

**「向调度问轴长」这条留着，而且应该留**：尾块长度（`materializeTailAwareTileSize`）问的是
「这根轴这趟有多长」，descriptor 的 `repeats` 上写的是整块 tile 的长度，答不了这个问题。
被消灭的是「拿轴长冒充张量尺寸」，不是「向调度查询」。

**两条实测出来的边界（都已在后续提交里补上）**：`afir::materializeSizeAt` 会就地建两种 op，
而 conversion 那一层原先对两种都没准备。

其一是 `afir.symbolic_eval`。进 conversion 之前这个 op 已经被 `AFIRMemoryAllocatePass` 清空
（它在末尾跑 `populateSymbolicLoweringPatterns`），所以输入里没有；但 conversion 的 pattern
**自己会建**——非字面量的 stride / extent 物化出来就是它。新建的 op 没人 legalize，dialect
conversion 把**整个** pattern 回滚，报的是 `failed to legalize operation 'afir.load'`，看起来像
判据挂了。本需求实现期间是靠「切分轴那一项被 `tileSize` / `tailAwareSize` 替换掉、其余项都是
字面量」绕过去的；随后在 `ConvertAFIRToAscendCPass` 里补了同一组 pattern，由
`afir-symbolic-segment-extent.mlir` 钉住（把那一行注释掉就只有它变红）。

其二是 `afir.tiling_param`（descriptor 里的自由符号绑在它上面），故障形态比上一条更糟：
它此前是用一个**自己新开的 `OpBuilder`** 建的，不进 dialect conversion 的建档，
`applyFullConversion` 连它非法都不知道——那一条被原样留在下降完的函数里，**一条诊断都没有**。
两处都要改：`materializeSizeAt` 绑符号时用调用方的 `builder`（`getOrCreateTilingParam` 的
`OpBuilder &` 重载把插入点临时挪到入口块顶部、返回前还原），`LowerTilingParamOp` 收没有
`ascend.implementation` 属性的那一条（实现选择跑在 conversion 之前，新建的拿不到那个属性；
放行不会选错，`afir.tiling_param` 全仓只有一个 implementation）。相位 A
（`AFIRMaterializeTilingParamsPass`）的 `collectRequiredSymbols` 同时补上 tensor descriptor 的
`repeats` / `strides` / `vectorized_strides` 那一遍扫描——它自述收的是「kernel body 的每一个
运行期尺寸符号」，而 descriptor 此前不在其中，所以「conversion 里才新建」是真会发生的形态。
由 `afir-tiling-param-created-in-conversion.mlir`（相位 A 不跑那条路）与
`afir-tiling-param-descriptor-symbol.mlir`（相位 A 自己那一遍）钉住。

涉及组件：Autofuse MLIR。**不改** `autofuse/` 下 MLIR 以外的任何文件，特别是不改
`autofuse/ascendc/api/broadcast.h` 与 `autofuse/optimize/`。

---

# 总体概述

## 软件概述

### 项目介绍

legacy 的合轴是**两个阶段**，划分依据是作用对象：

| | 阶段 1（图级） | 阶段 2（指令映射级） |
|---|---|---|
| 落点 | `optimize/optimize.cpp:845` `MergeContinuousAxis` | `ApiCall::Generate` 内 |
| 作用对象 | 全图共享的迭代空间（`ApplyMerge` 改写轴集合） | 单条指令的参数，不改图 |
| 约束 | 必须全图一致（tiling / block 切分 / 内存规划 / 代价模型都读它） | 只需本算子的 operand 一致，可以更激进 |

两阶段是必要结构，不是历史包袱：阶段 1 的产物必须是一套全图同意的轴，阶段 2 的产物每个算子各不相同。
业界同构——TorchInductor 的 scheduler 侧 `simplify_and_reorder` 对应阶段 1、Triton codegen 侧 tiling
对应阶段 2；MLIR 上游是 `linalg` collapse + `vector` flatten 两层；PyTorch TensorIterator 的
`coalesce_dimensions` 与 `CheckAxisContinuous` 的判据逐字同构。

**缺陷不在分两阶段，在阶段 2 有两份判据不同的实现**：

- `codegen/api_call/utils/api_call_utils.cpp:343` `CheckAxisContinuous`——对所有 operand 检查
  `stride[i] == stride[i+1] × repeat[i+1]`，业界标准判据；
- `codegen/api_call/broadcast/broadcast_api_call.cpp:40` `DimensionCollapse`——按相邻轴
  `is_brc`（`repeat_in != repeat_out`）标志相同即合，**不检查 stride 连续性**。

后者不是"在另一个维度上更激进"，而是**判据更弱**：UB tile 上出现对齐填充时（`AxisAttr` 有 `align` 字段），
两根相邻普通轴会被合并、尺寸按 `GetFormerMergedSize` 直接相乘，而那个乘积不等于实际内存跨度。
未找到活的触发用例（跑过的 tile 轴 `align` 均为 1），故不宣称为现存缺陷，但它不应被保留进统一实现。

**MLIR 侧今天既没有阶段 2 也没有第三份**——它白拿了阶段 1 的结果：4 维 elementwise
`77x5x92x7` 的 `afir.asc_graph` 里 z0/z1/z2/z3 已被合成一根 `z0z1z2z3`（size 247940），
到 MLIR 手上 `vectorized_axis = [6]` 只有一根。elementwise 能通、broadcast 不能通，
就是因为阶段 1 对纯 elementwise 图能合到底，对广播图合不到底（广播段与普通段交替）。

### 产品环境介绍

- **AscendC**：本设计只**改变发什么形态的调用**，不新增也不修改 `autofuse/ascendc/api/broadcast.h`
  的任何函数签名。用到的两个形态都是现成的（见「功能需求 3」的形态表）。
- **CANN Toolkit**：依赖原生 `AscendC::Broadcast<T, 2, 1>`（`compiler/ascendc/include/highlevel_api/`），
  v220 实现在 `impl/pad/broadcast/broadcast_v220_impl.h`。无版本下限变化——legacy 已在用同一个 API。
- **外部接口边界**：无。本设计全部在 MLIR 内部 C++，不导出符号、不动 Python 绑定、不动 run 包布局。

## 探针结论（动产品代码之前跑的）

探针做法：把 `autofuse/ascendc/api/broadcast.h` **复制**一份，在每个叶子原语处插 printf，
单文件 g++ 直链 tikicpulib（Ascend910B1）跑 f32。产品代码一字未动。
`ICPU_RUN_KF` 是 fork 出子进程跑的，host 侧全局计数器收不到，标记必须在 kernel 里直接打印。

### P1 核心等价性成立

三段形态 `[A normal][B brc][C normal]`（A=3, B=4），legacy 三轴与「外层 for + 两段调用」
在 C ∈ {8, 32, 128} 上结果一致，全 PASS。

### P2 `Or` 只出现在「外层普通段留在 API 内」这条路上

插桩实测的原语（不是读代码推的）：

| 发的参数形态 | 落到 | 原语 |
|---|---|---|
| `src_m=1, src_k=Z, src_z=0` | autofuse `BroadcastFirstDim` | `Copy`（`srcRepStride=0`）；Z ≤ 一个 block 时单条 counter-mask `Copy` |
| `src_m=M, src_k=1, src_z=0` | 原生 `Broadcast<T,2,1>` → v220 `TwoDimBroadCastLastDim` | `Brcb` + `Copy` + `GatherMask`，B ∈ {8,9,32,33,209} 与 A=64/B=7 全 PASS |
| `src_m=M, src_k=1, src_z=Z`（legacy 三轴） | autofuse `BroadcastMiddleDim` | **`Or`**（Z > 8）/ `MiddleDimWithCopy:Copy`（Z ≤ 8） |

即 **legacy 在 Z > 8 的所有测到形状上都落 `Or`，统一方案全程只有 `Copy`**。
legacy 四轴的 `(A,1,A,1)` pattern 也踩 `Or`（P4 的 C2 实测 `MiddleDim:Or(tail)`），分解后变 `FirstDim:Copy`。
`Or` 只在 `broadcast.h:79` / `:88` 出现，`Or` 版本在 CANN 原生实现里也只在
`broadcast_common_utils.h` 的 `TwoDimBroadCastDimAlign` / `LoopBroadCast`，
而 v220 的 `TwoDimBroadCastLastDim` 不走它们。

### P3 内层段 32B 对齐是既有前置条件，不是本方案新增的

`Z × sizeof(T) % 32 != 0` 时（Z ∈ {9, 12, 33, 209}）：

- **零报错、零 SIGABRT，结果错**（A=1）。机理：`dst_rpt_stride = Z × sizeof(T) / 32` 整除截断，
  repeat 前进 `floor(Z×4/32) × 8` 个元素而不是 Z 个——Z=9 首错 idx=8、Z=33 首错 idx=32、
  Z=209 首错 idx=208，与算式吻合。
- **legacy 用三轴参数发同一形状错得逐字相同**（Z=9: 28/36 首错 idx=8；Z=33: 100/132 首错 idx=32；
  Z=209: 628/836 首错 idx=208）。`MiddleDim` 的 `Or` 路径有同一个 32B 假设，
  其 `dstBlkStride` 同样是整除截断。
- A > 1 且不对齐时两条路径都 SIGABRT（子张量起始地址 32B 检查）。
- CANN 原生 `TwoDimBroadCastFirstDim` 对同一件事有显式 `ASCENDC_ASSERT(numBlocks * sizeof(T) % ONE_BLK_SIZE == 0)`；
  autofuse 的包装没有。

**结论**：这是 `broadcast.h` 整体的一条未文档化前置条件，两条路径以相同方式违反它，
**本方案不需要新增对齐兜底**，继承同一条调度不变量即可（假设表第 4 行）。
段方向 B（广播段在最内）不受此限——P2 表第二行的 B=9/33/209 全 PASS，
因为 `TwoDimBroadCastLastDim` 有 Align220 / NotAlign220 两个变体。

### P4 多广播段可以在编译器侧分解（重叠探针）

分解规则：**由内而外，每个广播段一次两段调用**，外层广播段用「偏移一整块 + repeat 减一」
复制已算好的块——`Broadcast(dst[blk], dst, 1, blk, 0, K-1, blk, 0, tmp)`。
源区 `[0, blk)` 与写区 `[blk, K×blk)` 正好错开，所以 `src == dst` **其实不重叠**。
legacy 四轴内部就是这么写的（`broadcast.h:528` 起）。

| 形状 | legacy | 统一分解 | 实测原语 |
|---|---|---|---|
| C1 `(1,B,1,B)→(A,B,A,B)`，A=3 B=8 | PASS | PASS | legacy: `MiddleDimWithCopy:Copy` + `FirstDim:Copy`；统一: `FirstDim:Copy` ×2 |
| C2 `(A,1,A,1)→(A,B,A,B)`，A=8 B=3 | PASS | PASS | legacy: 原生 `<T,2,1>` + **`MiddleDim:Or`**；统一: 原生 `<T,2,1>` + `FirstDim:Copy` |
| C3 五段 `(M,1,N,1,K)→(M,B1,N,B2,K)`，2×3×4×2×8 | **ASSERT（无五轴重载）** | PASS | `FirstDim:Copy` ×2 |

三条结论：

1. **重叠写法正确**，且统一分解在 legacy 能做的两个 pattern 上与 legacy 结果一致。
2. **C2 这一格 legacy 要 `inter_buf` 中转，分解后不需要**——因为外层普通段提成循环后，
   两趟的顺序可以自由安排，内层块直接写到 dst 的最终位置。调度给的那块更大的 tmp 因此可能用不上，
   它已在 ATT 预算内，不额外占用。
3. **覆盖面严格超过 legacy**：C3 legacy 直接 `ASSERT(false)`。

**两趟之间要一条 `PipeBarrier<PIPE_V>`**（legacy 同样发）。**这一条探针验不了**——
CPU 模拟器顺序执行，barrier 缺失测不出来。所以它的依据是 legacy 先例，不是实测。

**顺带探出一条前置条件**：counter 模式 `Copy` 的尺寸检查要求目的张量从其起点起有额外余量。
末段子张量刚好写满时报「needs at least 1792 bytes, while current tensor size is only 1536 bytes」
（`kernel_operator_data_copy_intf_impl.h:594`，CPU-debug 断言）。补 256 B（一个 repeat 粒度）
后 C1 两边全绿，说明实际写入在范围内、那 256 B 是检查的保守余量。**具体公式我没推出来，
不在文档里编**。legacy 四轴撞同一条（紧缓冲时两边同样失败），所以 legacy 在生产里能跑
即说明调度侧的 tile 尺寸满足它——但这条要在实现前确认，列入假设表。


### P6 counter 模式 `Copy` 的目的张量尺寸检查（假设表第 7 行，实现前核的那一次）

探针：`scratchpad/copyslack`，单文件 g++ 直链 tikicpulib（Ascend910B1，f32），直接调产品的
`BroadcastFirstDim`，只改目的 `LocalTensor` 的 `SetSize`。

**检查的公式测出来了**，它读的是目的张量的 `GetSize()`，与传进去的 `repeatTime` 无关：

```
elemsPerRepeat = 256 / sizeof(T)                       // f32 下 64
n              = ceil(Z / elemsPerRepeat)              // Z 是 counter 模式的元素数，即 dst_k
tail           = Z - (n - 1) * elemsPerRepeat
required_bytes = (n - 1) * floor(Z * sizeof(T) / 32) * 32 + tail * sizeof(T)
```

实测对齐（`(dst_m, Z, dst_size)` → 报的 needs-at-least）：`(2,192,384)`→1792、`(1,128,128)`→768、
`(5,384,1920)`→7936、`(3,192,576)` 与 `(1,65,65)` 通过。**`Z ≤ 256/sizeof(T)` 时 `n = 1`、
`required = Z × sizeof(T)`，恒过**——今天所有单段用例都在这一格。

**三条与文档原先的推断不同**：

1. **不是恒定 256 B 的余量**。缺口随 `Z` 增长：`Z=384` 时是 7936 vs 7168。P4 末段看到的
   「补 256 B 就绿」只是 `Z=192` 那一格的数。
2. **它同样卡功能需求 3 的形态 A**，不只是功能需求 4 的第 i 趟：`(dst_m=2, Z=192)` 直接报。
   判据是 `dst.GetSize() × sizeof(T) ≥ required_bytes(Z)`，而形态 A 在最后一趟外层循环上
   可用的正好是 `K × Z × sizeof(T)`。
3. **它不是正确性前提**。把目的张量放大到过检查的尺寸后，用哨兵逐元素查：写区之外一个字节
   没动、写区之内逐元素正确，`(1,192)`、`(2,192)`、`(1,128)`、`(5,384)`、`(1,384)` 与第 i 趟形态的
   `(3,192)`、`(3,384)`、`(4,64)`、`(8,8)` 全部 `wrong=0 touched_beyond=0`。即 API 只写
   `dst_m × dst_k` 个元素，检查的 `n = ceil(count/elemsPerRepeat)` 模型比实际写入保守。

**结论与落点**：不新增编译期兜底，也不因此把任何一档退回 unsupported——判据里的 `Z` 与 `K`
在真实图上都是符号（tile 尺寸），编译期判不了，而「判不了就 unsupported」会把整档一起关掉。
legacy 的 `BroadcastOneAxis` 发的是逐字相同的调用，今天承担的是同一条暴露，本设计不扩大它。
这条只在 CPU 模拟器（`ASCENDC_CPU_DEBUG`）下会 SIGABRT，NPU 上没有这个检查。
真要兜，正确的落点是**调度侧给广播目的 tile 留够 `required_bytes(Z)`**，那是 `optimize/` 的事，
不在本设计范围。


### P5 探出一个与合轴正交的缺口

`RemovePad` 不在 `bridge/AfirBuilder/AfirBuilder.cpp:59` 的算子注册表里，
`autofuse/mlir/` 全树 grep 零命中 → 含它的图整图回落 legacy。
这既是当前的安全兜底（尾轴不对齐的广播图进不来，所以撞不到 P3），
也是一个独立缺口：它解决的是非对齐小尾轴下 MTE3 带宽利用率低的问题，与合轴正交，独立 PR。

> **订正（2026-09-30，`RemovePad` 接通时核实）**：括号里那句「所以撞不到 P3」**因果说反了**。
> 挡住 P3 的不是「含 `RemovePad` 的图进不来」，而是 `classifySegments` 的
> `provablyEqual(outer.strides[dst], inner.extent)`（`Broadcast.cpp:334`）与 reg 侧那条稠密判据：
> 非对齐模板里 `RemovePad` 插在 broadcast **之后**，broadcast 自己仍是 `kAligned`，它目的地的
> 外段 stride 是 `Align(Z, 32/sizeof(T))`，`Z` 不对齐时那条判据当场不成立。
> 所以接通 `RemovePad` **不会**让 P3 暴露：那种图里的 `broadcast` 照旧报 unsupported。
> 放宽那条稠密判据（`classifyExtendSegments` 注释里写的「尾位改按 stride 取」）与挡住 P3 是
> **同一处改动的两面**——按行距发时 `dst_rpt_stride` 恒整除。做那一处的人要同时加带填充的正例。
> 详见 `2026-09-30-afir-remove-pad-design.md` 的「不做的事 ①」。


## 软件功能

1. 一份向量化轴分段分析：读 `TensorDescAttr` 的逐轴信息，按 stride 连续性对所有 operand 同时合并，输出分段列表。
2. 一份外层循环生成：分段数 > 1 时生成 `scf.for` 嵌套并把各 operand 偏移到最内窗口。
3. `afir.broadcast` 单广播段一档的下降，覆盖两种段方向。
4. `afir.broadcast` 多广播段的分解，由内而外每段一次两段调用。
5. elementwise 改用同一分析，单段行为不变。
6. 一条可测的不变量：MLIR 发出的广播调用不会落到 `Or`。

## 设计约束

- **CANN Toolkit**：无新增版本要求。
- **ABI/API**：无对外接口变化，全部是 MLIR 内部 C++。
- **构建打包**：只在 `conversion/AFIRToAscendC/CMakeLists.txt` 与 `dialect/AFIR/Analysis/CMakeLists.txt`
  各加源文件，不改安装路径与产物。
- **不得硬编码芯片**：两种段方向的选择由**段的 stride 形态**决定，不由 SoC 字符串决定；
  两侧（mem / reg）由 `ApiFamily` 查 TargetProfile，与既有 op 同路。
- **日志**：编译期新增代码在每个 op 上跑一次分段（O(段数 × operand 数)），不新增默认日志。
- **SuperKernel 解耦**：不碰 `super_kernel/`。

## 假设和依赖关系

| 前提 | 状态 | 依据 |
|---|---|---|
| bridge 已把逐轴信息完整传到 `TensorDescAttr` | **已验证** | `AFIRAttrs.td:119` 的 `axis`/`repeats`/`strides`/`vectorized_axis`/`vectorized_strides`；`afir-broadcast-multi-axis-unsupported.mlir:31` 是两轴信息完整到达的实证 |
| 多轴时 `sched.axis` 把向量化轴排在 `loop_axis` 之后 | **已验证** | 同一文件 `afir.sched = <axis = [3, 4, 2, 5], loop_axis = 4>`，`vectorized_axis = [2, 5]` |
| `AFIRPipelineSync` 在 conversion 之前跑完 | **已验证** | pass dump 顺序 `04_afir-pipeline-sync` → `06_convert-afir-to-ascendc`。这条决定了新循环必须在 conversion 层生成（见「功能需求 2」处理第三点与「安全检查」G3） |
| broadcast 及其上游的 tile 尾轴是 32B 对齐的 | **已验证（调度不变量）** | `optimize/platform/v1/template/unaligned_template.cpp:86` 注释：「broadcast 以及 broadcast 之前的节点尾轴正常对齐，broadcast 之后的节点尾轴不做 32B 对齐」，并在 broadcast 之后插 `RemovePad` |
| 含 `RemovePad` 的图今天不会进 MLIR | ~~已验证~~ **已过期（2026-09-30）**，但**本设计不依赖它** | 当时：`AFIR_BRIDGE_OPERATION_REGISTRY` 无 `RemovePad`。现在它在表里了（`afir.remove_pad`），而本设计的对齐前提并不靠这一条——靠的是「目的地必须稠密」那条判据，见 P5 的订正 |
| 调度已为两广播段的情形备好 tmp buffer | **已验证** | `ascir/reg_func/broadcast.cpp:60` `NeedExtraTmpBuf` 的判据正是 `brc_num == 2`，命中走 `GetExtraTmpBuf`，返回**一块**（更大，`broadcast.cpp:54`）。MLIR 侧 `checkTmpBuffer`（`ElementwiseApiFamily.cpp:105`）查块数不查大小，所以 bool `needsTmpBuffer` 原样够用 |
| counter 模式 `Copy` 的目的张量余量要求被满足 | **已确认：它不是正确性前提，是 CPU-debug 检查的保守模型** | 探针结论 P6。检查读的是目的张量的 `GetSize()`，公式已测出来（见 P6）；实际写入严格落在 `dst_m × dst_k` 之内且结果正确，8 组形状逐一用哨兵验过。所以**不因这一条把任何一档退回 unsupported**，但它是 CPU 侧（tikicpulib / E2E）的一条硬失败，落点与判据记在 P6 |

---

# 需求分析与设计

## 整体介绍

数据流（单个 op 的下降）：

```
TensorDescAttr（逐轴 axis/repeats/strides/vectorized_*）
      │
      ├─► collapseVectorizedAxes()        ← 阶段 2 合轴，一份实现
      │        输出 VectorPlan{ segments[] }，每段 { extent, strides[operand] }
      │
      ├─► segments.size() == 1 ──► 直接发 API，count = segments[0].extent   【今天所有 elementwise 走这条，文本不变】
      │
      └─► segments.size()  > 1 ──► materializeVectorizedSegments()
                                       生成普通段的 scf.for（elementwise 留最内 1 段，brc 留 2 段）
                                       各 operand LocalTensorSubIndexOp 到最内窗口
                                       ├─► |Bset| ≤ 1：最内窗口发一条 API
                                       └─► |Bset| ≥ 2：由内而外每个广播段一趟，
                                             趟间发 PipeBarrier<PIPE_V>（功能需求 4）
```

控制流边界：分段分析在 `dialect/AFIR/Analysis/`（纯读属性，不改 IR）；循环生成在
`conversion/AFIRToAscendC/`（建 IR）。两者的分界就是「算」与「发」。

## 功能需求

### 功能需求 1：向量化轴分段分析

1. **介绍**

   给定一个带 canonical descriptor 的 AFIR op，按 stride 连续性把它的向量化轴合并成若干段，
   每段记录 extent 与**每个 operand 各自**的 stride。这是 legacy 阶段 2 合轴在 MLIR 侧的唯一实现。

2. **输入**

   op 上的 `afir.operand_descs` / `afir.destination_descs` / `afir.result_descs`（`TensorDescAttr`）与
   `afir.sched`。要求：`vectorized_axis` 非空；`axis` 与 `repeats` 等长；每根 `vectorized_axis`
   都能在 `axis` 里找到。任一条不满足 → 返回 `failure()`，由调用方报 `MissingInformation`。

3. **处理**

   从最内向外，对候选轴 `i`（当前段最内轴为 `j`）：若对**所有**参与的 tensor 都有
   `stride[i] == stride[j] × repeat[j]`，则把 `i` 并入当前段（extent 相乘）；否则封段、开新段。
   判据与 `CheckAxisContinuous` 逐字相同，也与 PyTorch `can_coalesce` 相同。

   - 广播轴（stride 为 0）只与 stride 为 0 的轴合并：`0 == 0 × r` 恒真，`s == 0 × r` 当 `s ≠ 0` 恒假。
     这不是特例，是同一条判据的自然结果——**前序文档"需求 5"里那个 `vectorized_strides.front()`
     按字面量 0 / 1 分流的特判因此被删除，而不是被保留**。
   - 符号表达式：能静态判等就判，判不了一律**封段**（保守方向：多分一段只是多一层循环，
     少分一段是算错地址）。
   - `extent` 为字面量 1 的段直接跳过，不入列表。

4. **输出**

   `VectorPlan{ SmallVector<VectorSegment> segments; }`，段序外→内。
   `VectorSegment{ Value extent; SmallVector<Value> operandStrides; SmallVector<Value> destinationStrides; }`。
   `segments` 为空表示这个 op 在向量化轴上没有工作量，调用方报 unsupported。

### 功能需求 2：外层循环生成 helper

1. **介绍**

   `materializeVectorizedSegments(op, builder, plan, transferPlan, tiles, apiWindowSize)`：
   为 `segments` 里除最内 `apiWindowSize` 段之外的每一段生成一层 `scf.for`，把插入点移到
   最内层循环体内，返回各张量已偏移到当前迭代起点的值与窗口各段的元素数。

2. **输入**

   `VectorPlan`（功能需求 1 的产物）、`TransferPlan`（取尾块长度用）、按 `plan.tensors` 顺序
   给出的目标方言 tile 值、`apiWindowSize`（elementwise 传 1，broadcast 传 2）、一个 `OpBuilder`。

   取 `OpBuilder &` 而不是 `ConversionPatternRewriter &`：本函数只用 `create` 与
   `setInsertionPoint`，而 `OpBuilder::create` 走 listener，传一个 rewriter 进来照样进它的
   建档。换成基类只为 UT 里造得出调用方；`materializeScheduledTileIndices` 也是这个签名。

   `plan.segments.size() < apiWindowSize`、`tiles` 与 `plan.tensors` 不等长、尺寸表达式物化
   不了，都返回 `failure()`。

3. **处理**

   - 由外而内为每一段建 `scf.for %iv = 0 to extent step 1`（无 `iter_args`——buffer 形态下没有
     SSA 值流出循环，与 `LoopGenerate` 生成的 tile 循环同形）。上界在建循环**之前**算，
     否则它会落进循环体里，既支配不了循环头也白算一遍。
   - **段 extent 逐轴相乘，切分轴那一项换成带尾块修正的长度**。这一条是实现过程中被逼出来的：
     `TileInner` 在最后一个 tile 上的实际长度是尾块长度而不是轴长，而 `VectorSegment::extent`
     已经乘成一个数、换不回去。所以功能需求 1 的段里要同时存逐轴 extent（`axisExtents`）。
     **少了这一步，含切分轴的段在尾块那一趟会算多。**
   - 尾块长度取自新增的 `materializeTailAwareTileSize`，不是 `materializeScheduledTileIndices`。
     后者除了算尾块还要按局部 stride 是 0 还是 1 分流去算 GM 偏移，**stride 既非 0 也非 1 时
     直接报失败**——多段 tile（广播的目的地 stride 是 8）正好撞它，可那与「这根轴这趟有多长」
     无关。所以把「问轴」从「问某块 tile」里拆出来，原函数原样留给 Load / Store 的 GM 偏移。
   - 每个张量的偏移 `Σ_j (iv_j × stride_j)`，用 `arith.muli` / `arith.addi` 累加，再发一条
     `ascendc.local_tensor.subindex`。这与 legacy `CalcInnerOffset` 算的是同一个式子。
     **stride 与 extent 的物化要分成两个函数**：stride 可以是 0（广播轴每趟读同一份），
     extent 不行；合用一个就得把「必须为正」那条判据放宽，那样 extent 为 0 的错误 descriptor
     也会被放过去。
   - **不发同步原语**：`AFIRPipelineSync` 已在 conversion 之前跑完，Enque / DeQue 对是本 op 在
     既有 tile 循环里的兄弟节点，新循环只包住计算调用本身。这与 legacy
     `CreateComputeNodeOuterFor` 只包 API 调用一致。

4. **输出**

   `VectorizedWindow{ SmallVector<Value> tiles; SmallVector<Value> extents; }`——`tiles` 与
   `plan.tensors` 一一对应，`extents` 是最内 `apiWindowSize` 段的元素数（外→内）。

   **副作用**：返回时插入点在最内层**新建**循环体内（终结符之前）；没有新建循环时插入点不变。
   调用方随后就在那里建 API 调用。

   `segments.size() == apiWindowSize` 时不建循环、偏移落成常量 0、插入点不变——**这一条保证
   今天所有单段用例的生成文本逐字不变**（今天单段发的就是「常量 0 + subindex」）。

### 功能需求 3：`afir.broadcast` 单广播段一档的下降

1. **介绍**

   `|Bset| == 1`（恰好一段的 src stride 为 0）时，把最内两段交给两轴 API、其余段交给循环。

2. **输入**

   `afir.broadcast`，一输入一 destination，mem 侧另有一块 tmp buffer。`VectorPlan` 满足
   `|Bset| == 1` 且广播段落在最内两段之内。

3. **处理**

   按最内两段的方向二选一。两种形态都是 `broadcast.h` 现成的 C++ 签名（`broadcast.h:512`
   那条 `Broadcast(dst, src, src_m, src_k, src_z, dst_m, dst_k, dst_z, tmp_buf)`），
   **但 MLIR 侧要新建一个 td def**：全仓只有 `AscendC_BroadcastL2Op`（发原生
   `AscendC::Broadcast<T, dim, axis>(dst, src, dstShape[], srcShape[])`，无 tmp，且它的手写
   emitter 从**静态**张量 shape 反推 `axis`，而这条路上的 tile 一律是
   `!ascendc.local_tensor<?xf32>`）与 `Adv/Broadcast.td` 的 `AscendC_BroadcastOp`（带
   `BroadcastTiling` 的另一套），两个都不是这条包装。前序文档说「`BroadcastL2Op` 已经有」
   是把原生 API 当成了包装 API。新建的是一个 API extension op（`ApiExtensionOpInterface`，
   扩展头 `broadcast.h`），两种段方向共用它，只差实参：

   | 最内两段 | 发的调用 | 落到 | 原语 |
   |---|---|---|---|
   | `[brc K][normal Z]` | `Broadcast(dst, src, 1, Z, 0, K, Z, 0, tmp)` | `BroadcastFirstDim` | `Copy`（`srcRepStride=0`）；`Z ≤ 一个 block` 时单条 counter-mask `Copy` |
   | `[normal M][brc K]` | `Broadcast(dst, src, M, 1, 0, M, K, 0, tmp)` | 原生 `Broadcast<T,2,1>` → v220 `TwoDimBroadCastLastDim` | `Brcb` + `Copy` + `GatherMask` |

   只剩一段且它是广播段（src 恒 1 元素）→ 走**已有的** `broadcast.tensor.duplicate` implementation，
   本需求不改它。

   `|Bset| ≥ 2` 交给功能需求 4。其余判据不满足（广播段不在最内两段内、分段拿不到、
   广播段与内侧段构不成合法两段窗口）→ `CapabilitySupport::unsupported`，整图回落 legacy——
   与今天等价，不会静默出错值。

4. **输出**

   kernel 文本里 `b` 层 `for` 包一条 `Broadcast(...)`；`b == 0` 时就是一条裸调用。无 `Or`。

### 功能需求 4：`afir.broadcast` 多广播段的分解

1. **介绍**

   `|Bset| ≥ 2` 时，由内而外每个广播段发一次两段调用，段间发一条 `PipeBarrier<PIPE_V>`。
   覆盖面严格超过 legacy 四轴——legacy 只支持 `(1,B,1,B)` 与 `(A,1,A,1)` 两个硬编码 pattern，
   其余 `ASSERT(false)`（探针结论 P4 的 C3）。

2. **输入**

   `VectorPlan` 且 `|Bset| ≥ 2`。每个广播段必须与它内侧相邻的那一段构成合法的两段窗口
   （即广播段之间不相邻——相邻的广播段在功能需求 1 里已被合并成一段）。

3. **处理**

   设广播段自内向外为 `Sb0, Sb1, ..., Sbk`（`Sb0` 最内）。

   - **第 0 趟**：`Sb0` 与它的两段窗口，按功能需求 3 的形态表发调用。
   - **第 i 趟**（i ≥ 1）：`Sbi` 的工作是把内侧已算好的整块复制 `extent(Sbi)` 份。发
     `Broadcast(dst[blk], dst, 1, blk, 0, extent(Sbi) - 1, blk, 0, tmp)`，其中 `blk` 是
     `Sbi` 内侧所有段的元素数之积。**偏移一整块 + repeat 减一**使源区 `[0, blk)` 与写区
     `[blk, extent × blk)` 错开，所以 `src == dst` 并不重叠；legacy 四轴内部就是这么写的。
   - **每两趟之间发一条 `PipeBarrier<PIPE_V>`**。必须由本 pattern 自己发——`AFIRPipelineSync`
     已在 conversion 之前跑完，它按 op 间的边配对，给不出 op **内部**两趟之间的依赖。
     先例是现有 `Duplicate` 一档自己发 `MTE2_S` / `V_S` 事件三件套，同一个理由。
     **这一条测不出来**：CPU 模拟器顺序执行，barrier 缺失不会红。依据是 legacy 先例，
     不是实测，测试设计里也不假装能覆盖它。
   - 广播段之间的普通段照功能需求 2 变成循环，层次嵌在对应的两趟之间。

   不需要中转缓冲：legacy 的 `(A,1,A,1)` pattern 用 `inter_buf` 是因为它把外层普通段留在
   API 内、两趟顺序不能自由安排；提成循环后内层块直接写到 dst 的最终位置（P4 的 C2 实测）。
   调度按 `NeedExtraTmpBuf`（判据正是 `brc_num == 2`）给的那块更大的 tmp 因此可能用不上——
   它已在 ATT 预算内，不额外占用，`checkTmpBuffer` 查的是**块数**（`GetExtraTmpBuf` 返回一块），
   所以两侧注册处的 `needsTmpBuffer` 原样够用。

4. **输出**

   kernel 文本里 `k+1` 条 `Broadcast(...)`、`k` 条 `PipeBarrier<PIPE_V>`，外加普通段的循环。
   每条调用都满足功能需求 7 的 `Or`-free 判据。

### 功能需求 5：elementwise 改用同一分析

1. **介绍**

   一元 / 二元 / 三元 elementwise 的 `materializeScheduledTileIndices` 调用点换成功能需求 1 + 2，
   `apiWindowSize = 1`。

2. **输入**

   不变。

3. **处理**

   - `collapseVectorizedAxes` + `orderTilesForPlan` + `materializeVectorizedSegments`。
     `orderTilesForPlan` 按 `VectorTensor::side` / `::index` 把 adaptor 的值排成 `plan.tensors`
     的顺序；下标越界报失败而不是跳过——跳过会让 `tiles` 与 `tensors` 错位，之后按下标取
     stride 就取到别的张量头上。
   - **标量操作数没有向量化轴**。调度给它的 descriptor 是
     `axis = [0, 1], repeats = [1, 1], strides = [0, 0], role = compute`，**一个
     `vectorized_axis` 都没有**（`load_scalar_add_store_201x39` 的真实 dump）。所以功能需求 1
     要跳过这种张量，否则「各张量同一串向量化轴」立刻不成立，所有标量二元图整体回落 legacy。
     跳过之后，下降侧原本按 `isa<ascendc::LocalTensorType>` 写的「标量不是 tile、不发
     subindex」那几条判断不必再写一遍——它从分析里自然落出来。
   - **二元的 `BinaryOperandOrder` 重排的是发射次序，不是 descriptor 次序**。
     `VectorTensor::index` 说的是 `afir.operand_descs` 里的位置，两者不是一回事：按重排后的
     次序传值会静默错位。所以传 adaptor 的原始次序，再用 `tensor.index == order.tile /
     order.other` 把结果认回重排后的位置。

   今天所有 elementwise 图在这里都只有一段（图级合轴已经合到底），所以走的是「不建循环、
   偏移恒为常量 0」那条分支。

4. **输出**

   **60 个 CPU E2E 的生成 kernel 逐字不变**（2026-09-28 实测：快照 60 份，替换一元 / 二元 /
   三元三处之后逐份 `diff`，相同 60、不同 0）。预期中的两处 IR 差异——新路径少发一条
   `arith.muli`（旧的 `materializeScheduledTileIndices` 顺带算 GM 偏移，elementwise 不用）、
   多发一条 `arith.constant 0`（偏移按张量各建一个）——都在 CSE 之后被并掉了。

### 功能需求 6：**本轮只注册 mem 侧**

原计划是「两侧各注册一个 implementation」，沿用前序文档需求 2 的结论（mem 侧调度给 tmp
buffer、reg 侧不给，`checkTmpBuffer` 与调度给的块数严格对等，所以不能用一个
`ApiFamily::Any` 盖住两侧）。**tmp buffer 那条结论仍然成立，但「同一条调用、只差 tmp 块数」
不成立**：reg 侧的广播根本是另一个 API——

- 发的是 `BroadcastExtend<T, rank>(dst, src, dst_shape, src_shape)`
  （`v35/ascendc/api_regbase/broadcast.h:17`，`v35/codegen/reg_api_call/reg_broadcast_api_call.cpp:120`），
  吃的是**整串向量化轴的 shape 数组**，没有 tmp buffer，rank 上限 9，合轴与分段由
  `BroadcastExtendImpl` 自己在 tiling 里做（`v35/ascendc/api_regbase/broadcast_extend_impl.h:277`）；
- 也就是说「最内两段窗口 + 外层 `scf.for`」这套模型在 reg 侧既不需要、也对不上。

所以本轮功能需求 3 / 4 **只注册 mem 侧**；reg 侧保留现有的
`broadcast.tensor.duplicate_reg_base` 一档，非常量源在 reg 侧照旧 unsupported、整图回落
legacy——与接通之前逐字相同。reg 侧的 `BroadcastExtend` 是一条独立的 lowering 路径，
归独立 PR。

> **2026-09-29 补**：那条独立路径已接通，id `broadcast.tensor.extend_reg_base`
> （`Implementations/Broadcast.cpp` 的 `BroadcastTier::Extend`）。它不建循环，整串段直接落成
> 两个 shape 数组；收的边界是「两个 tile 在这串段上都稠密」，带填充的 descriptor 仍报
> unsupported、整图回落 legacy。正例 / 反例见
> `test/lit/conversion/afir-broadcast-extend-reg-base{,-unsupported}.mlir`。

### 功能需求 7：`Or`-free 不变量及其测试

1. **介绍**

   MLIR 发出的广播调用不得落到 `BroadcastMiddleDim` 的 `Or` 路径。

2. **输入**

   功能需求 3 发出的调用参数。

3. **处理**

   判据是一行：**`src_m == 1` 或 `src_z == 0`**。查三个 dtype 家族的分派条件：

   - `BroadcastCommon`（i16/ui16/f16/f32/i32/ui32）：`src_m == 1 && src_k == dst_k && src_z == dst_z`
     → `BroadcastFirstDim`（只有 `Copy`）；`src_k == 1 && src_z == 0` → 原生 `Broadcast<T,2,1>`。
   - `BroadcastWithCast`（i8/ui8）：`src_m == 1 && src_k == dst_k` → `BroadcastFirstDim`。
   - `BroadcastInt64`（i64/ui64）：`src_m == 1 && src_k == dst_k` → `BroadcastFirstDim`。

   `Or` 只在 `broadcast.h:79` / `:88`（`BroadcastMiddleDim`）出现，而 `MiddleDim` 只由
   `src_k == 1 && src_z != 0`（即"外层普通段留在 API 内"）那一支调用。功能需求 3 的两种形态
   一个是 `src_m = 1`、一个是 `src_z = 0`，**都够不着它**。

4. **输出**

   lit 用例断言生成文本里 `Broadcast(` 的第三个实参是 `1`（形态 A）或第五个实参是 `0`（形态 B）。
   这是"`Or` 不可达"在编译期唯一可断言的形式——`Or` 的选择发生在设备端头文件里，lit 看不到。

## 非功能需求

### 可维护性

- 分段分析落在 `dialect/AFIR/Analysis/VectorSegmentation.{h,cpp}`，循环生成落在
  `conversion/AFIRToAscendC/VectorizedLoops.{h,cpp}`。两个新文件，不新增 pass。
- `TransferPlan` 的 `Identity` / `ScheduledContiguous` 两档与 `isQueueBuffer` 保留原样——
  它们回答的是"这个 op 的搬运计划是什么"，与分段是两件事。删掉的是
  `ScheduledTileIndices::count`（被功能需求 1 吸收）；`TransferPlan` 的单轴字段**改名**为
  `tileInnerAxis` 而不是删掉，理由见验收标准 1。
- 复用 `ElementwiseApiFamily.h` 的 `checkApiFamily` / `checkTmpBuffer` / `hasCanonicalDescriptors`。
- 不改 AFIR 操作数三段划分、不改 bridge、不改 ODS。

### 可测试性

三层分工见「测试设计」。分段分析是纯函数（属性进、结构体出），可以直接 UT，
不需要跑整条 pipeline——这是把它放在 `Analysis/` 而不是塞进 pattern 的主要收益。

### 可移植性

- 不新增芯片判断。段方向的选择由 stride 形态决定。
- 两侧由 `ApiFamily` 查 TargetProfile，与既有 op 同路。
- 依赖的原生 `Broadcast<T,2,1>` 在 v200 / v220 / 3510 各有实现，CANN 自己分派；
  本设计不感知也不硬编码其中任何一个。

### 可靠性

- 所有"判不了"一律返回 `failure()` / `unsupported`，回落 legacy。方向是"宁可不支持，不可算错"。
- 不申请资源；tmp buffer 由调度分配、`MemoryAllocate` 绑定。
- 新建的 `scf.for` 无 `iter_args`、无 region 参数逃逸，`rewriter` 失败时由 dialect conversion 回滚。

### 特性交叉影响

逐项按 `cross_feature_check.md`：

| 场景 | 适用性 | 分析说明 |
|------|--------|----------|
| SuperKernel Python 接口 | 不适用 | 不碰 `super_kernel/`，不影响 wheel、选项解析、pytest |
| SuperKernel C++/AOT 接口 | 不适用 | 不碰 `super_kernel/kernel/`，无导出符号变化，ABI/API 不变 |
| Autofuse 图优化 | 不适用 | 不新增 / 不调整 pass，不改图改写；`optimize/` 与调度侧一字未动。阶段 1 合轴照旧 |
| Autofuse Codegen/Backend | **适用** | 新增分段分析与循环生成，新增一个 broadcast implementation；改一元 / 二元 / 三元 elementwise 的取值点。单段输出逐字不变，多段从"拒绝"变"支持"。tiling 不变 |
| AscendC API / Runtime 交互 | **适用** | 新发 `Broadcast(...)` 两种形态（现成签名），新发 `local_tensor.subindex`。不新增 `rt*` / `aclrt*` 调用。无异步拷贝、无 size 为 0 的调用（extent 为 1 的段已被剔除，`Duplicate` 一档不走本路径） |
| Python/C++ 混合绑定 | 不适用 | 不碰 `autofuse/compiler/py_module/` |
| 构建与打包 | **适用（仅源文件表）** | 两个 `CMakeLists.txt` 各加源文件。不改安装路径、产物命名、第三方依赖；`--pkg` / `--no-autofuse` 路径不受影响 |
| 测试与覆盖率 | **适用** | 见「测试设计」 |
| 性能与日志 | **适用但影响可忽略** | 见「性能」。不新增默认日志 |
| 兼容性 | 不适用 | 无对外接口、脚本参数、配置项、产物布局变化 |

---

## 性能

### 编译时长

每个 op 上一次分段，复杂度 O(向量化轴数 × operand 数)，轴数 ≤ 个位数、operand ≤ 3。
多段时额外建 `scf.for` 与几条 `arith` op。相对既有 pass 遍历可忽略。

### 执行性能

**测到的**：

- 对齐形状上统一方案与 legacy 结果一致（CPU 模拟器，f32，`[A][B_brc][C]`，C ∈ {8, 32, 128}）。
- 原语路径（插桩副本实测）：legacy 在 `C > 8` 时全部落 `MiddleDim:Or`，`C = 8` 落
  `MiddleDimWithCopy:Copy`；统一方案全程 `FirstDim:Copy`。另一方向 `[A normal][B brc]`
  落原生 `Broadcast<T,2,1>`，B ∈ {8, 9, 32, 33, 209} 与 A=64/B=7 全部正确。
- 多广播段：legacy 四轴的 `(A,1,A,1)` pattern 落 `MiddleDim:Or`，分解后落 `FirstDim:Copy`
  （探针结论 P4 的 C2）。所以 `Or` 在两条路上各出现一次，分解把两处都绕开了。

**没测到的**：CPU 模拟器（tikicpulib）不产出时序，本轮无性能数字。以下是读代码的推断，待实测确认：

- **预期更快**：`Or` 换成 `Copy`。`Or` 要多读一个全零操作数、走 ALU 而非 MOV、
  先 `Duplicate` 一个零块，且 `BinaryRepeatParams` 的 `dstBlkStride` 是 `uint8`，
  内层段超过 `MAX_REPEAT_TIME × blk`（f32 下 2040）时还要退化成标量 `DataCopy` 循环
  （`broadcast.h:61`）。`Copy` 的 `dstRepStride` 是 `uint16`，没这个限制。
- **不变**：外层段从"API 内部 for"变成"编译器生成 for"，向量单元指令流相同。
  legacy 的三轴形态本来就是 `broadcast.h:321` 的 `for (i < src_m)`，不是硬件多维寻址。
- **可能变差**：编译器循环每迭代多发一次 `SetMaskCount` / `SetVectorMask` / `SetMaskNorm` /
  `ResetMask`（在 `BroadcastFirstDim` 内）。对照的 legacy `MiddleDim` 路径每迭代发一条
  `PipeBarrier<PIPE_V>`。哪个更贵要量。内层段极小、外层 extent 极大时两者都退化。
- **新增的编译器可见性**：地址运算成为 `arith` IR，LICM 与强度削减可作用其上；在 C++ 头文件里拿不到。

**怎么补上这些数字（独立 PR）**：CANN 里 camodel 的部件是齐的——

| 部件 | 位置 |
|---|---|
| camodel runtime，本 arch 的 | `simulator/dav_2201/lib/libruntime_camodel.so`（`dav_2201` 即 `__NPU_ARCH__=2201` / `__DAV_C220__`）；reg 侧 `Ascend950PR_9599` 另有完整 `camodel/` 目录 |
| 官方 sim 构建模式 | `tikcpp/ascendc_kernel_cmake/legacy_modules/function.cmake:413`：`$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>` |
| `ccec` 编译器 | `$CANN/x86_64-linux/bin/ccec`。**必需**——camodel 跑的是真 CCE 机器码，不是 g++ 产物 |
| profiling 打点 | `libruntime_camodel.so` 里有 `AdcProfiler`、`AiCpuProfilerStart/Stop`、按 stream/task 的 task profiling |

但**它不是给现有 E2E 加个开关**，而是第二条并行路径：编译器 g++ → `ccec`（CMake 里是一门新语言，
`asc-config.cmake` 提供）、启动 `ICPU_RUN_KF` → acl 风格 launch、`GmAlloc` → `aclrtMalloc`，
且 cycle-accurate 比 CPU 模拟慢得多，**不能进 `mlir-test.sh` 的默认回路**。
建议做成按需触发的 `check-perf-sim-<case>`，覆盖少数形状，性能类验收改成"这几个 case 的
指令 / 周期数不退化"。另外「能拿到指令级周期还是只有 kernel 总耗时」我没验过——
裸跑 `msprof` 挂在 HAL 符号（`undefined symbol: halProfDataFlush`）上，说明要 sim 模式那套完整环境。
**这些都归独立 PR，不在本设计范围。**

### 内存和产物大小

不新增结构体缓存、不新增常驻内存。mem 侧 tmp buffer 本就由 ATT 算进预算。
两个新 `.cpp` 对 `.so` 的增量在 KB 量级。

---

## 接口设计

### 新增/修改接口描述

全部是 MLIR 内部 C++，无导出符号。

```cpp
namespace mlir::afir {
// 参与判据的一个张量，以及它在 IR 里的出处。出处要带上，因为 `afir.operand_descs` 里
// 并非每一项都参与（Load 的 GM 侧角色是 `input`），按数组下标硬对是错的。
struct VectorTensor {
  enum class Side { Operand, Destination };
  Side side;
  unsigned index;                      // 在该侧 descriptor 数组里的下标
  TensorDescAttr descriptor;
};
struct VectorSegment {
  SmallVector<int64_t> axes;           // 段内轴，外→内
  ExprAttr extent;                     // 段内各轴 extent 之积，已 Simplify
  SmallVector<ExprAttr> axisExtents;   // 逐轴 extent，与 axes 对应；切分轴那一项要被换掉
  SmallVector<ExprAttr> strides;       // 与 tensors 一一对应：本段**最内**轴上的 stride
};
struct VectorPlan {
  SmallVector<VectorTensor> tensors;   // operand 侧在前，destination 侧在后
  SmallVector<VectorSegment> segments; // 外→内
};

FailureOr<VectorPlan> collapseVectorizedAxes(Operation *op);
}  // namespace mlir::afir

namespace mlir::ascend::kernel {
struct MaterializedWindow {
  SmallVector<Value> operands;
  Value destination;
  SmallVector<Value> windowExtents;       // 长度 == apiWindowSize
};
FailureOr<MaterializedWindow> materializeVectorizedSegments(
    Operation *op, ConversionPatternRewriter &rewriter,
    const afir::VectorPlan &plan, unsigned apiWindowSize);
}  // namespace mlir::ascend::kernel
```

**为什么产物是 `ExprAttr` 而不是 `Value`**：extent 与 stride 都是符号表达式，合轴的判据
（`stride[外] == stride[内] × extent[内]`）是表达式层面的比较，不需要 IR。用 `Value` 就得在
分析里物化 `afir.tiling_param` / `afir.symbolic_eval`，分析不再是纯函数，「可以直接 UT、
不用跑整条 pipeline」这句话也就兑现不了。物化推迟到「功能需求 2」的循环生成 helper。

**新增的两个支撑函数**：

```cpp
// dialect/AFIR/Analysis/TransferPlan.h
// 切分轴在当前 tile 上的实际长度（尾块那一趟取尾长）。只问轴，不问某块 tile 的 stride。
FailureOr<Value> materializeTailAwareTileSize(Operation *op, OpBuilder &builder, const TransferPlan &plan);

// dialect/AFIR/IR/SymbolicBuilder.h
// 自由符号绑到入口块的 `afir.tiling_param`，表达式在 builder 的当前插入点物化（不进尺寸序言）。
FailureOr<Value> materializeSizeAt(OpBuilder &builder, Block &entry, Location location, ExprAttr expression);
```

**修改**：`ScheduledTileIndices` 删 `count` 字段；`TransferPlan` 的单轴字段改名为
`tileInnerAxis`——尾块修正必须知道哪一根轴在最后一趟取尾长，那是一根轴的属性、分段答不了。
**`materializeScheduledTileIndices` 的 `vectorized_strides.front()` 0 / 1 分流保留**：它是
Load / Store 算 GM 偏移的判据，与段无关；与段有关的那一半（尾块长度）已经拆进
`materializeTailAwareTileSize`。

### 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|--------|----------|----------|------|
| 接口说明 | 是否需要接口评审 | 否 | 内部 C++，无跨组件边界 |
| 接口说明 | 是否需要补充文档 | 是 | 本设计文档；两个新头文件的 doc 注释 |
| 接口兼容 | 行为是否兼容 | 是 | 单段逐字不变；多段从 unsupported 变 supported（能力扩展） |
| 接口兼容 | ABI/API 是否兼容 | 否 | 无导出符号变化 |
| 接口约束 | 约束不满足时是否清晰报错 | 是 | 一律 `failure()` / `unsupported(<具体 Reason>)`，不猜不兜 |
| 接口测试 | 是否需要独立接口用例 | 是 | `collapseVectorizedAxes` 是纯函数，独立 UT（8.2 第 1 行） |

---

## 软件设计

### 关键数据结构

见「新增/修改接口描述」。`VectorSegment` 持有的是 `Value`（已物化的 extent / stride），不持有属性引用——
生命周期跟随所在 `func::FuncOp`，由 `builder` 的插入点决定。`VectorPlan` 按值返回，无所有权问题。

### 关键技术/算法

合轴（功能需求 1 处理段）。复杂度 O(n × m)，n = 向量化轴数，m = operand + destination 数。
边界条件：

- n == 0 → `failure()`；
- 所有轴 extent 均为字面量 1 → `segments` 为空 → 调用方报 unsupported；
- 符号 stride 判不了 → 封段（保守）；
- 段 extent 相乘用 `arith.muli` 在 IR 里算，不在编译期算——extent 可能是 TilingData 字段。

**符号表达式只能用规范形式构造**：`Expression::Parse` 拒绝非规范输入
（`"((1) * (32))"` 会报「expression is not canonical; canonical form is '32'」），所以段 extent
的乘积**不能拼字符串再 Parse**，要走 `detail::ExpressionAccess::Substitute`——模板
`"(__arg0 * __arg1)"` 本身规范，替换值来自现成的 `ExprAttr`。同理，判等不能只靠
`Parse("ExpectEq(a, b)")`：两侧恰好相等时该谓词的规范形式是 `True`，拼出来的串 Parse 不过，
所以 `ProveExpressionsEqual` 先比化简后的规范串、比不出来再交给 `Prove`。
**这两处如果写错，表现都是「合不了轴」而不是报错**——静默多出几层循环，不会有任何用例变红，
所以 UT 里「符号可证相等要能合」那一格是必需的。

**为什么不重排轴**：PyTorch TensorIterator 在合并前还会 `reorder_dimensions` 按 stride 排序，
让更多轴可合。本设计**不做**，因为重排会改变哪一段是"API 内段"，而 UB 上的物理布局由上游
schedule 与 DMA 决定，MLIR 这一层改不动它。这是有意识的取舍，不是遗漏。

#### 流程设计

主流程见「整体介绍」。异常流程：任一步 `failure()` → `notifyMatchFailure` / `unsupported` →
该 op 不被 legalize → `AFIRVerifyKernelCompletePass` 报错 → 整图回落 legacy。
无回滚需求：`ConversionPatternRewriter` 在 pattern 失败时自动撤销已建的 op，包括新建的 `scf.for`。

### 对子模块的修改

| 文件 | 修改点 |
|---|---|
| `dialect/AFIR/Analysis/VectorSegmentation.{h,cpp}` | **新增**。功能需求 1 |
| `dialect/AFIR/Analysis/CMakeLists.txt` | 加一个源文件 |
| `dialect/AFIR/Analysis/TransferPlan.{h,cpp}` | 新增 `materializeTailAwareTileSize`；单轴字段改名 `tileInnerAxis`、删 `count` 字段；**保留** stride 0/1 分流（它是 GM 偏移的判据） |
| `conversion/AFIRToAscendC/VectorizedLoops.{h,cpp}` | **新增**。功能需求 2 |
| `dialect/AFIR/IR/SymbolicBuilder.{h,cpp}` | 新增 `materializeSizeAt`；后续补 `getOrCreateTilingParam` 的 `OpBuilder &` 重载（见「两条实测出来的边界」） |
| `dialect/AFIR/Support/SymbolicExpression.{h,cpp}` | 新增 `ProveExpressionsEqual` |
| `api/ascendc/include/ascir/Dialect/Asc/IR/Autofuse/OpVecBroadcast.td` | **新增** `AscendC_BroadcastDim3Op`：`broadcast.h:512` 那条包装的 td def（功能需求 3） |
| `api/ascendc/{include/ascir/Target/Asc/Autofuse/VecBroadcast.h,lib/Target/AscendC/Autofuse/VecBroadcast.cpp}` | 新增它的手写 emitter；`Target/Asc/Autofuse/Ops.h` 的 op 表加一项 |
| `conversion/AFIRToAscendC/Implementations/Broadcast.cpp` | 新增单广播段 implementation（两种段方向，功能需求 3）与多广播段分解（功能需求 4）；现有 `Duplicate` 一档的判据改写成"所有段 extent 为 1"，行为不变 |
| `conversion/AFIRToAscendC/Implementations/{Unary,Binary,Ternary}Elementwise.cpp` | 取值点换成新 helper，`apiWindowSize = 1` |
| `conversion/AFIRToAscendC/CMakeLists.txt` | 加一个源文件 |
| `test/ut/` `test/lit/` | 见「测试设计」 |

### 错误处理

#### 系统错误

不申请内存、不开文件、不调 runtime。`rewriter.create` 失败由 MLIR 自身处理。

#### 接口错误

无对外错误码。内部一律 `FailureOr` / `CapabilitySupport::unsupported(Reason)`，
Reason 按语义选 `MissingInformation`（拿不到 descriptor）/ `UnsupportedShape`（段形态不收）。
不新增用户可见提示——回落 legacy 对用户是透明的。

---

## 安全检查

逐条按 `编码红线.md`：

| 红线 | 结论 |
|---|---|
| 1 硬编码敏感信息 | 无。不出现密钥、地址、邮箱；测试数据是 `index * 0.25 - 3.0` 这类明显占位值 |
| 2 外部输入作索引/长度前校验 | **适用**。分段前先确认 `vectorized_axis` 非空、`axis` 与 `repeats` 等长、每根轴 id 在 `axis` 里找得到，再按下标取用。`apiWindowSize` 是编译期常量（1 或 2），进 helper 时断言 `segments.size() >= apiWindowSize` |
| 3 整数溢出/反转/除 0 | extent 相乘在 IR 里用 `arith.muli`，由运行期 TilingData 决定，与既有 `symbolic_eval` 同路；编译期不做尺寸算术。无除法 |
| 4 资源释放覆盖异常分支 | 不持有资源。`rewriter` 失败自动回滚 |
| 5 内存申请前判大小 | 不申请内存。extent 为字面量 1 的段已剔除，不会发 size 为 0 的调用 |
| 6 未指定求值顺序/UB | 无多副作用实参；`SmallVector` 下标访问前均已查长度 |
| 7 禁止扩大修改范围 | 不动 `optimize/`、`codegen/`、`ascendc/api/`、bridge、ODS。`TransferPlan` 的删改只涉及被新分析吸收的字段 |
| G1 API/ABI 兼容 | 无导出符号、无 Python 绑定、无脚本参数、无产物布局变化 |
| G2 禁止硬编码芯片/平台/框架 | **重点**。段方向由 stride 形态选，两侧由 `ApiFamily` 查 TargetProfile。代码与注释不出现 SoC 字符串 |
| G3 图改写保持数据边与控制边等价 | **适用**。新建 `scf.for` 只包住计算调用本身；Enque/Deque 与 `SetFlag`/`WaitFlag` 由 `AFIRPipelineSync` 在 conversion 之前放好，是本 op 在 tile 循环里的兄弟节点，不被卷入新循环（假设表第 3 行） |
| G4 有时序依赖的 pass 说明依赖 | 不新增 pass。对 pass 顺序的依赖是"`AFIRPipelineSync` 必须在 conversion 之前"，这是既有顺序，本设计在 8.3 说明为何它决定了落点 |
| G5 图改写结果确定 | 分段遍历 `vectorized_axis` 数组，顺序由 IR 决定；不用 `unordered_map`、不依赖指针地址 |
| G6 高频路径禁止默认海量日志 | 不新增日志 |
| G7/G8 runtime 接口生命周期与开放性 | 不调 `rt*` / `aclrt*`。新发的 AscendC 调用均为同步向量指令，源与目的都是本迭代内有效的 UB tile |
| G9 Python/C++ 绑定引用计数 | 不涉及 |
| G10 构建脚本不破坏增量/离线构建 | 只加源文件，不动依赖下载与打包逻辑 |
| G11 测试临时产物不进版本库 | 探针在 scratchpad，不入库；E2E 产物在 `build/` |
| G12 全局/静态对象析构 | 不新增全局对象 |

---

## 兼容性检查

不影响脚本参数、配置文件、安装目录、动态库、Python 包、旧产物或用户代码。
对已有 IR：原先合法的 AFIR 仍然合法（不改 ODS 约束）；原先被拒绝的多段形态变为被接受。

---

## 测试设计

### 测试边界

- **入口**：`collapseVectorizedAxes`（UT 直接调）、`af-opt` 跑 conversion pipeline（lit）、
  `check-e2e-cpu-<name>`（E2E）。
- **出口**：`VectorPlan` 结构（UT）、生成的 AscendC 文本（lit）、kernel 执行结果（E2E）。
- **打桩**：无。分段分析是纯函数，lit 用手写 AFIR 作输入。
- **验证数据**：lit 直接复用 `afir-broadcast-multi-axis-unsupported.mlir` 的 descriptor
  （`vectorized_strides` src `[0, 1]` / dst `[8, 1]`）——按本设计它是两段规范形态、
  **零层循环、一条调用**，所以这个文件从"反例"改成"正例"，这本身就是最好的回归锚点。

### 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|----------|------------|----------|----------|
| 功能 | 合轴判据：相邻轴对所有 operand 连续才合；某一 operand 否决即封段 | `VectorSegmentationTest`：手写 descriptor，断言 `segments` 的段数与每段 extent/stride | UT |
| 功能 | 广播段 stride 0 不与非 0 合并，且**不作为特例**存在 | 同上，一格 `src [0,1] / dst [8,1]` 断言 2 段 | UT |
| 功能 | 单段时不建循环、偏移恒为 0 | `VectorizedLoopsTest.SingleSegmentBuildsNoLoop`（结构）+ 一元 elementwise lit（文本与改动前逐字一致） | UT + lit |
| 功能 | 外层段变循环、上界是段 extent、窗口 extent 取内段 | `VectorizedLoopsTest.OuterSegmentBecomesALoop` / `TwoSegmentWindowLeavesNoLoop` | UT |
| 功能 | 偏移**逐张量各取各的 stride**（广播段上 src 为 0、dst 为 8） | `VectorizedLoopsTest.OffsetsUsePerTensorStrides`；改成共用第一个张量的 stride 即红 | UT |
| 异常 | 段数少于窗口 / `tiles` 与 `tensors` 不等长 → 失败 | `VectorizedLoopsTest` 两格 | UT |
| 功能 | **含切分轴的段在尾块那一趟不多算** | `afir-elementwise-scheduled-tail.mlir`：IR 侧钉 `cmpi eq` → `cmpi ne` → `andi` → `select` 这条链喂给算子的最后一个实参；CPP 侧钉 `AbsExtend(..., static_cast<uint32_t>([[COUNT]]))` 就是那个三元式的结果。把 count 改成直接取轴长即红 | lit |
| 功能 | 新路径确实被走到（不是静默回落） | 让 `collapseVectorizedAxes` 恒失败：上面那条 lit 红，真实标量调度图 `failed to legalize` | 变异验证 |
| 兼容性 | **既有 CPU E2E 生成 kernel 逐字不变** | 替换前后逐份 `diff`；65 份里 62 份相同，不同的三份正是分段广播（改前那三份或算错或生成不出来） | E2E |
| 功能 | 段方向 A `[brc][normal]`：零循环，一条 `Broadcast`，第三实参为 `1` | `afir-broadcast-segmented.mlir` 的 `broadcast_outer_dim` | lit |
| 功能 | 段方向 B `[normal][brc]`：一条 `Broadcast`，`src_k` 为 `1`、第五实参为 `0` | 同一文件的 `broadcast_inner_dim` | lit |
| 功能 | 三段 `[normal][brc][normal]`：一层 `scf.for` + 一条调用，偏移逐张量各取各的 stride | 同一文件的 `broadcast_outer_loop` | lit |
| 功能 | reg 侧那一档（`BroadcastExtend`）选得出、shape 数组对 | `afir-broadcast-extend-reg-base.mlir` / `-unsupported.mlir`。**只有 lit**：E2E 卡在 bridge 不收 `Nddma`，见「reg 侧那一档还没有 E2E」 | lit |
| 功能 | 两广播段 `[brc][normal][brc][normal]`：两条 `Broadcast` + 一条 `PipeBarrier`，第二条的源是第一条的目的、目的偏移一整块、repeat 为 `extent-1` | `afir-broadcast-multi-brc.mlir` 的 `broadcast_two_brc` | lit |
| 功能 | 五段两广播段：两层 `scf.for` + 两条 `Broadcast` + 一条 `PipeBarrier`，barrier 与整块复制在**内层循环之后、外层循环之内** | 同一文件的 `broadcast_five_segment`（legacy 此形状 `ASSERT`） | lit |
| 功能 | 真实 dump 的两段形态**端到端**：Load 偏移恒 0 / 长度 8、Broadcast 第三实参 1、Store 前进 `zt×8` / 搬 `尾块×8` | `afir-broadcast-two-segment.mlir`（IR + kernel 文本 + selection 三道） | lit |
| 异常 | 广播段够不着最内两段窗口 → unsupported，不被 legalize | `afir-broadcast-unsupported.mlir` | lit |
| 异常 | 符号 stride 判不了 → 封段而非合并（错误方向是多一层循环，不是错地址） | `VectorSegmentationTest` 一格符号 stride | UT |
| 异常 | mem 侧那一档在 reg 侧选不出（`checkApiFamily` / `checkTmpBuffer` 对等） | `not af-opt ... \| FileCheck --check-prefix=WRONGSIDE` | lit |
| 性能 | `Or` 不可达：发出的调用第三实参为 `1` 或第五实参为 `0` | 上述两份正例各加一条 CHECK；理由见功能需求 7 输出 | lit |
| 性能 | `Or` → `Copy` 的实际收益 | **camodel 验证，独立 PR**，理由与部件清单见「性能」的执行性能 | Benchmark（挂起） |
| 兼容性 | bridge 白名单与两侧覆盖断言仍绿 | `BufferFormSetIsCoveredByAscendCImplementations`、`EveryElementwiseOpCoversBothApiFamilies` | UT |
| 特性交叉 | 多段 broadcast 图端到端算得对 | `load_broadcast_segmented_store_1x2x1x8`（四段分解，128 个元素全对）。内侧一整块的大小有上限，见「整块复制那一趟的上限」 | E2E |
| 异常 | 段间 `PipeBarrier<PIPE_V>` 缺失 | **测不到**。CPU 模拟器顺序执行，缺 barrier 不会红；lit 只能断言它被发出来，不能断言它是必需的。依据是 legacy 先例（`broadcast.h:528` 起两趟之间就有一条） | lit（只钉「发出来了」） |

**变异验证（三轮，逐条推理由）**：

| 变异 | 结果 | 处置 |
|---|---|---|
| 形态 A / B 选反 | 变红 | —— |
| 整块复制 repeat 不减一 | 变红 | —— |
| 整块复制的目的不偏移一整块 | 变红 | —— |
| 趟间不发 `PipeBarrier` | 变红（只钉「发出来了」） | —— |
| `blk` 取源侧 stride 而不是目的侧 | 变红 | —— |
| 「广播段必须落在最内两段之内」 | **没红** | **判据冗余，删掉**：广播段更靠外时最内两段都是普通段，源 stride 形态那一条两个方向都过不去 |
| 两段窗口的源 stride 形态 | 起初没红 → 拆片后变红 | 用例盲区：原反例一格同时违反两条，互相遮蔽。改成三片、一片只违反一条 |
| 目的地外段 stride == 内段 extent | 起初没红 → 拆片后变红 | 同上 |
| 「广播段之间不相邻」 | 没红 | **有意的保守边界**：目的地有填充时这个形状造得出来，手推分解结果是对的，但没有用例证明。保留并在代码里写明要收它得先加哪条正例 |
| 「让位给 `Duplicate` 一档」 | 没红 | **独立边界，保留**：那一档查的是 repeat、分段查的是 stride，两者在调度写出来的 descriptor 上总是一起出现，但不是同一件事 |

**Load / Store 那一轮的变异验证**：

| 变异 | 结果 | 处置 |
|---|---|---|
| GM 前进量退回「轴长」 | 变红 | —— |
| 不判「切分轴是否在这一段里」，一律算偏移 | 变红 | —— |
| 元素数退回「尾块长度」本身、不乘段内其余轴 | 变红 | —— |
| 局部张量按下标 0 硬对、不按 descriptor 认 | 起初没红 → 补断言后变红 | 用例盲区：`afir-broadcast-duplicate.mlir` 当时没钉 `Duplicate` 的元素数，而那个 op 有两个张量、认错列只会少写不会报错。补上 `duplicate_l2` 第三实参的断言 |
| 「丢掉 stride 为 0 的段」 | 没红 | **删掉**：今天走不到（广播轴是被 `extent == 1` 剔掉的，不是被 stride 为 0 剔掉的），而它是**放宽**不是冗余——放行一个没验过的形状的代价是静默搬错。legacy 会丢，我们报失败回落给它 |
| 段数大于一时不报失败 | 没红 | **保守边界**：多段要发 ND-DMA，今天表达不了。没有能隔离它的用例——要一张局部 tile 不连续的完整调度图 |
| 不要求这一段的局部 stride 为 1 | 没红 | 同上，一并留着 |

**`windowIsTailFree` 已删（2026-09-29）**。它原先是本档的一条判据：切分轴（`TileInner`）
落到「尺寸兼作跨度」的段上就不收——`materializeVectorizedSegments` 给的段 extent 带尾块
修正，而包装的 `dst_k` 既是一行的长度也是行与行的间距，尾块上行长变短、行距不变。

删它的理由不是那个危险不存在，是**这条判据的输入集是空的**：`getTileInnerAxis` 取局部
descriptor `vectorized_axis` 的 `front()`，合轴保持这个次序，所以切分轴要么落在**第 0 段**、
要么（整根 extent 为 1 被剔掉时）不在任何段里；而 `classifySegments` 保证段数 ≥ 2，
`windowIsTailFree` 的扫描起点 `min(front + 1, count - 1)` 恒 ≥ 1，第 0 段永远不在范围里。
两头一夹，它只可能答「是」，删掉它编译结果逐字节不变。处置与本轮另外两条冗余判据一致
（「广播段必须落在最内两段之内」「丢掉 stride 为 0 的段」）。

「切分轴在第 0 段」这半边是**量出来**的，不是推的：把扫描起点改成
`broadcastSegments.front()`（把最外那个广播段本身也算进去），用到 mem 侧分段那一档的三张图
全部变红（`afir-broadcast-two-segment` / `afir-symbolic-segment-extent` /
`afir-tiling-param-created-in-conversion`）。

**不变量本身不许丢**，所以同时改了三处注释，三处都指向「放宽 `getTileInnerAxis` 时要回来
补什么」：`getTileInnerAxis` 自己（两档分段广播都靠它才对、都没再查一遍）、
`materializeExtendShapeEntry`（reg 侧无条件只给第 0 段上尾块修正）、`classifySegments`
的边界清单（mem 侧两段窗口的 `dst_k` 兼作行距）。要恢复这条判据，应与放宽
`getTileInnerAxis`（真去轴表里找 `TileInner`）一起做——那时它变成可达的判据，也就隔离得出来。

### 测试覆盖现状

**lit**：功能需求 2 的两条（「段 extent 逐轴相乘、切分轴换成尾块长度」与「尾块长度从
`materializeScheduledTileIndices` 拆出来」）由 `afir-elementwise-scheduled-tail.mlir` 钉住
——走 `ScheduledContiguous`、断言元素数取的是 `arith.select` 出来的尾长而不是轴长。
符号尺寸下整除关系编译期看不出来，所以这一条不挑形状。

**CPU E2E**：广播共 6 条，都跑 `check-e2e-cpu-<name>` 目标（不手工跑 generator）。

| 用例 | 形状 | 钉住的是 |
|---|---|---|
| `load_broadcast_duplicate_store_128` | `[1] → [128]` | `Duplicate` 一档：`GetValue(0)` + `AscendC::Duplicate` |
| `load_broadcast_duplicate_store_8x128` | `[1,1] → [8,128]` | 同上，二维 |
| `load_broadcast_segmented_store_1x8` | `[1,8] → [128,8]` | 形态 A `[brc][normal]` |
| `load_broadcast_segmented_store_8x1` | `[8,1] → [8,128]` | 形态 B `[normal][brc]`，且 `size(z1T) > 1` |
| `load_broadcast_segmented_store_1024x1` | `[1024,1] → [1024,128]` | 形态 B 切多趟 |
| `load_broadcast_segmented_store_1x2x1x8` | `[1,2,1,8] → [4,2,4,8]` | **多广播段**四段分解：每段一趟、由内而外，趟间 `PipeBarrier<PIPE_V>` |

**仍然测不到的两件**：

- **段间 `PipeBarrier<PIPE_V>` 的必要性**。CPU 模拟器顺序执行，缺一条 barrier 不会红；
  lit 只能断言它被发出来。依据是 legacy 先例（`broadcast.h:528` 起两趟之间就有一条）。
- **性能**。camodel 是第二条并行的构建 + 运行路径，独立 PR（见「性能」的执行性能），
  所以 `Or` → `Copy` 的收益至今没有数字。

### 整块复制那一趟的上限

多广播段分解的外层那一趟是「把内侧已经算好的一整块复制 `extent - 1` 份」，发的是
`Broadcast(dst[offset], dst, 1, offset, 0, extent - 1, offset, 0, tmp)`——与
`autofuse/ascendc/api/broadcast.h:537` 处理 `(1,B,1,B) → (A,B,A,B)` 时**自己发的那一条逐字同形**。
这一趟落到 `BroadcastFirstDim` 的 else 分支，而那条路**对内侧块的大小有上限**：

| 内侧一整块（f32 个数） | 结果 |
|---|---|
| 64（`[1,2,1,8] → [4,2,4,8]`） | 跑通，128 个元素全对 |
| 128（`[1,4,1,8] → [4,4,4,8]`） | CPU 模拟器 `SIGABRT`：`tensor size needs to be at least 768 bytes, while current tensor size is only 512 bytes` |
| 256（`[1,8,1,8] → [4,8,4,8]`） | 同上，`at least 3328 bytes / only 1024 bytes` |

根因在 else 分支把两种记数方式混在一起：它按 counter 模式设掩码
（`SetVectorMask<T, COUNTER>(dst_inner_offset)`），但 counter 模式下掩码是**总元素数**、
repeat 次数由包装自己推（f32 一次 repeat 覆盖 8 block × 8 = 64 个），而它给的
`dstRepeatStride` 是**按一行**算的。内侧块超过一次 repeat 的量，包装就按 2 次以上 repeat
走地址，把本该填满一行的那些元素散到相隔一行的几个块上——CPU 侧被 `Copy` 的尺寸检查当场抓住，
NPU 侧没有这个检查。上表两行的字节数与「repeat 数 = ⌈总数 / 64⌉、跨度 = 一行」推出来的完全对上
（128 个：`(2-1) × 512 + 256 = 768`）。

**本次不改 `broadcast.h`**（范围一节已声明），所以 E2E 用例的形状取在上限上：`[1,2,1,8]`
内侧正好 64 个。要放开它，改的是包装那一支（按行发就该用 normal 模式的掩码 + repeat 次数，
或者干脆逐行 `DataCopy`），与 MLIR 侧无关。

### reg 侧那一档还没有 E2E：卡在 bridge 不收 `Nddma`

试过把同一张广播图挂到 `PLATFORM reg_base`（`Ascend950PR_9599`），generator 就地报

```
error: UnsupportedOperation: Schedule buffer-form bridge does not support node 'load0' of type 'Nddma'
```

`[1,8] → [128,8]`（分段）与 `[1,1] → [8,128]`（`Duplicate`）都一样，所以**与广播是哪一档无关**：
广播的 load 在 950 上被调度成 `Nddma` 节点，而 schedule → AFIR 的 bridge 不认这个类型。
reg 侧那一档（`BroadcastExtend`）因此仍然只有 lit 覆盖，没有被编译器与执行器看过。
这条与前面说的 reg 侧 (d) 是同一个方向上的下一步，不在本需求范围内。

### 「报不支持就回落 legacy」此前不成立（2026-10-09 订正 + 已修）

这份文档、`2026-09-30-afir-remove-pad-design.md`、`Broadcast.cpp` 的几处收边界注释、以及
`PlanSelector.cpp` 里那段回落说明，都写着「报 unsupported → 整图回落 legacy」。**核过代码：
那时不成立。** `RunMlirCodegen`（`bridge/CodegenOrchestration/CodegenOrchestration.cpp`）只在
两种情况返回 `kNotSupported`——MLIR codegen 没开、或者是 cube 融合图；capability 报不支持、
bridge 拒收、pass 失败一律 `kFailed`，而 `codegen/codegen.cpp` 把 `kFailed` 直接翻成
`af::FAILED`，**不往下走 legacy 那几行**。也就是说每一条保守边界的代价不是「慢一点」，
而是「这张图编译失败」。

实践上没咬人，只因为 MLIR codegen 是 `AF_ENABLE_MLIR_CODEGEN` 显式打开的，生产跑的是 legacy。
但整个「宁可拒收」的策略以**拒收很便宜**为前提，所以这条必须先修。

**已修（2026-10-09）：把状态拆开。** 判据是诊断上的一个标记，不是诊断文本的前缀：

- `support/CapabilityShortfall.h`：`emitCapabilityShortfall(...)` 发的错误在
  `Diagnostic::getMetadata()` 里带一个常量；`CapabilityShortfallObserver` 在
  `RunMlirCodegen` 的作用域内旁听（不吞诊断）。**缺省是 `kFailed`**——没标记的失败路径只会
  多报一次失败（吵但看得见），反过来漏判是把真 bug 静默回落。
- **为什么不按 `UnsupportedOperation` 前缀分类**：那个前缀两种都在用。「节点类型没注册」是
  能力不足，「`Scalar` 节点没有 value」是输入坏了，前者该回落、后者该大声失败。按它分类
  就是拿一个不可靠的标签当判据，而且等于把分类信息编码进要给人读的那句话里再解析回来。
- 标记点是有意选的，不是把 36 处 `UnsupportedOperation` 全换掉：capability 层的全部拒收
  收束在 `ImplementationSelection.cpp` 的 `has no supported Ascend implementation` 一处；
  bridge 侧标了「节点类型没建模」「没有注册 AFIR op」「dtype 不认识」「`exec_condition`
  取值不认识」四类；`exec_condition` 那个 pass 标了三条能力判据。其余保持 `kFailed`。
- **顺带堵上一处 capability 与 pattern 的语义漂移**：「搬运的 tile 只收一段」这条判据原先
  只长在 `materializeScheduledTileIndices` 里，能力层不问，于是多段的 tile 在能力层答
  「支持」、到 conversion 才报 `failed to legalize`——而那一句在新分类下算坏了。现在两侧
  共用 `localTileIsSingleSegment`（`Analysis/TransferPlan.h`），Load / Store 与广播的
  `Duplicate` 一档都问它。这同时把「Load / Store 合轴后段数 > 1」从硬失败变成回落。
- **不标记的一处判断**：SoC profile 查不到。那里分不开「一颗 MLIR 还没建模的真芯片」与
  「一个解析不出来的 `soc_version`」，而开关是人显式打开的，开的人需要知道这条路不覆盖
  他的芯片。保持 `kFailed`，既有用例 `InvalidSocVersionFails` 也钉着这一条。

验收：`CodegenOrchestrationTest` 两条（标记机制本身；能力不足走到
`RunMlirCodegen` 返回 `kNotSupported` 且 `result` 没被动过），
`test/lit/conversion/afir-transfer-multi-segment-unsupported.mlir` 一条（多段搬运**在能力层**
就被拒，不是 `failed to legalize`；去掉那条能力判据即变红，验过）。

### 那几条「报不支持」各自命中什么形状（2026-10-09 量过）

「下一步」第 3 条要的那张表。量具是 `test/probe/CapabilityBoundaryProbe.cpp`
（独立可执行文件 `af-capability-probe`，**不接任何 check-\* 目标**，产物是打到 stdout 的表）：
每个输出形状 × 每个非空的广播轴子集，走和 CPU E2E 生成器完全相同的两步
（`Optimizer::Optimize` → `RunMlirCodegen`），记状态与第一条诊断。48 张广播图 + 8 张同形状的
elementwise 对照。

**结果：广播 21 拒 / 27 过；elementwise 对照 8 全过。**

| 输出形状 | 尾轴字节 | 广播图 | 被拒 |
|---|---|---|---|
| `[128]` / `[1024]` | 512 / 4096（对齐） | 2 | 0 |
| `[8,128]` / `[128,8]` | 512 / 32（对齐） | 6 | 0 |
| `[33,5]` | **20（不对齐）** | 3 | 2 |
| `[4,8,4,8]` | 32（对齐） | 15 | 4 |
| `[2,3,5,7]` | **28（不对齐）** | 15 | 14 |
| `[128,128,128]` | 512（对齐） | 7 | 1 |

四条结论，按对排序的影响排列：

1. **尾轴对不对齐是主因。** 不对齐的两组（`[33,5]` / `[2,3,5,7]`）18 张里拒 16，只有「整张都是
   广播轴」那一格（走 `Duplicate`）过；对齐的四组 30 张里只拒 5。拒掉它们的是**「目的地在
   窗口里稠密」**那条判据——`kAligned` 下次内层 stride 取 `Align(Z, 32/sizeof(T))`，而内段
   extent 是 `Z`，`provablyEqual(outer.strides[dst], inner.extent)` 当场不成立。
   **这正是 A2**，也正是 `2026-09-30-afir-remove-pad-design.md`「不做的事 ①」里写好了改法、
   并且与 RemovePad 的 P3 截断是同一处改动两面的那一条。

2. **A1（广播段之间不相邻）是真的，但小。** 对齐组里那 5 条是
   `[1,8,4,1]` / `[4,1,4,1]` / `[1,1,4,1]` / `[1,8,1,1]` → `[4,8,4,8]` 与
   `[1,128,1]` → `[128,128,128]`，共同点是**最内那根轴是广播轴，且与它之间隔着非广播轴**
   （广播轴子集含最内轴、又不与它相邻）。相邻的那些（`{2,3}` / `{1,2,3}`）都过。

3. **A3（源在内段 stride 既非 0 也非 1）48 张图一次都没命中。** 原先的怀疑成立：
   **不要为它开工**，先等一个真产得出它的图。

4. **A4（搬运的 tile 多于一段）不独立可达。** 它在 5 条图上是第一个报出来的（诊断里的 op 是
   `afir.load`），但把那条能力判据去掉重跑，**21 / 27 一个数都没变**，只是改由
   `afir.broadcast` 报——也就是说这个 sweep 里没有任何一张图的唯一问题是多段搬运。
   elementwise 对照 8 全过，进一步说明它今天不是搬运层的独立缺口。那条判据本身仍然要留
   （它把一次 `failed to legalize` 换成一次回落，机制由
   `afir-transfer-multi-segment-unsupported.mlir` 钉住），但**它从「3~5 天的大件」降级**。

**另外两条值得一并记下：**

- **56 张图里 `kFailed` 0 次、`failed to legalize` 0 次。** 也就是上一节那条拆开失败分类的
  改动在真实图上成立：每一次拒收都是回落，没有一张图被硬失败掉。探针的桶里专门留了
  `DRIFT: capability said yes, pattern said no` 一格，它应当恒为 0，非 0 就说明又漂了一处。
- **这是合成形状 sweep，不是真实模型。** 它答得了「哪个形状族命中哪条判据」，答不了
  「真实模型里出现多频繁」——那要接 inductor 跑真模型。另外上表的两簇是**按形状读出来的**，
  不是判据自己报的；要把「这一张到底违反了 A1 还是 A2」钉死，得让 `classifySegments`
  在失败时带上是哪一条判据。两簇的分界已经够排序用，所以那一步没做。

### 泛化重量：上一张表的归属是错的（2026-10-11）

**上一节那张表按形状把 21 条拒收分给 A2（16，尾轴不对齐）+ A1（5，广播段不相邻），这个分法
不成立。** 它当时就标注过「两簇是按形状读出来的，不是判据自己报的」——现在实测证实了那层保留：
**把 A1 那条相邻判据删掉之后，它点名的 5 条依旧全被拒**，mem 侧 21 / 27 一个数不变。手推也对不上，
`[1,8,4,1] → [4,8,4,8]` 合轴后是 `[brc][normal][brc]`，两个广播段本来就不相邻，那条判据碰不到它们。

探针因此扩了两个维度，并在每行打出**可据以分簇的签名**（`test/probe/CapabilityBoundaryProbe.cpp`）：

- **dtype**（f32 / f16 / i32 / ui8）。不是凑数：32B 一档装多少个元素由它定，而「目的地在窗口里
  稠密」查的正是 `Align(Z, 32/sizeof(T))` 与 `Z` 相不相等，同一个形状换 dtype 就跨过那条边界。
- **五个形状**补在 32B 边界两侧，外加一个秩 5：`[7,8]`（尾 32B）、`[64,4]`（16B）、`[8,32]`（128B）、
  `[3,5,32]`（中间维不对齐、尾维对齐）、`[2,3,5,7,11]`。
- 行尾两列：`tail=<N>B` 与 `inner=brc` / `inner=normal`。

48 张图扩到 **380 张**（4 dtype × 13 形状 × 非空广播轴子集）。**结果 125 过 / 255 拒，`kFailed` 0 次、
`DRIFT` 桶 0 次**；同形状 elementwise 对照 13 / 13 全过（含秩 5），所以这些拒收都是广播族自己的。

| 尾轴 32B | 最内轴 | 过 | 拒 | 通过率 | 诊断里的 op |
|---|---|---|---|---|---|
| 对齐 | 非广播 | **50** | **0** | **100%** | — |
| 对齐 | 广播轴 | 52 | 28 | 65% | 全是 `afir.broadcast` |
| 不对齐 | 非广播 | 6 | 108 | 5% | 66 `afir.load` + 42 `afir.broadcast` |
| 不对齐 | 广播轴 | 17 | 119 | 12% | 全是 `afir.broadcast` |

**完全可用的只有一格：尾轴 32B 对齐、且最内轴不是广播轴——50 / 50。** 其余三格都是部分支持。

拒收拆成**四条互相独立**的卡点，这次每一条都有判据或实测隔离，不是按形状猜：

1. **尾轴不对齐**（227 / 255 的拒收落在那两格）。mem 侧那条窗口稠密判据。**reg 侧 2026-10-11 已收**
   （最内那一位发行距而不是行长，`classifyDenseSide` / `sideIsDense`）；**mem 侧收不了**：两段窗口里
   `dst_k` 既是行长又是行距（`BroadcastFirstDim` 的 `dst_inner_offset`），填充没地方表达，要换 API 形态。

2. **最内轴是广播轴。** 对齐那一格里被拒的 **28 条全是这一类，没有例外**。根因挖到了，三张图
   （`probe_brc_19` / `20` / `45`，即 `[1,8,4,1]` / `[4,1,4,1]` → `[4,8,4,8]` 与
   `[1,128,1]` → `[128,128,128]`）签名一致：

   ```
   vectorized_axis        = [..., 次内, 最内]
   src vectorized_strides = [...,   8,   0]     ← 次内轴上源的 stride 是 8，不是 1
   dst vectorized_strides = [...,   8,   1]
   ```

   `8 = 32 / sizeof(f32)`。**最内那根广播轴在源 tile 上虽然只有 1 个元素，却按 32B 占满一整档**，
   于是源在次内轴上不连续；而两段窗口的形态 B 要求源是 `(1, 0)`
   （`classifySegments` 的 `provablyLiteral(outer.strides[src], "1")`）。这与第 1 条是**两件独立的事**,
   也不是原先 A1 / A2 里的任何一条。它预测 f16 上那个数是 16、ui8 上是 32——还没验。

3. **`ui8` 在常量源那一档没注册**（`kDuplicateElementTypes` 缺它，`Broadcast.cpp` 文件头记着原因：
   它要的是包装调用而不是 `AscendC::Duplicate`）。实测隔离干净：`[1] → [128]` 在 f32 / f16 / i32 上
   过、**ui8 上拒**，而它 `tail=128` 是对齐的。ui8 整体 11 过 / 84 拒，四种 dtype 里最差。
   **这是四条里最便宜的一条**，而且现在有现成的实测用例。

4. **`afir.load` 的多段搬运**（A4）占 66 / 255，而且**只出现在「不对齐 + 最内轴非广播」那一格**：
   尾轴不对齐让 load 的 tile 带上填充、分成多段，同时最内轴真的在动。泛化之后它比原先的 4 / 21
   显著得多，原先「不独立可达」那条结论要按这个数重读。

**reg 侧 48 / 48 全拒，而且一张都走不到 broadcast**：那一侧调度把 `Load → Broadcast` 融成 `Nddma`，
而 `afir.nddma` 今天被挡在 `resultDescriptorIsSelfConsistent` 上（调度的 `GenNddmaNode` 漏抄
`strides`，见 `conversion/AFIRToAscendC/Implementations/Nddma.cpp`）。**所以 reg 侧两档广播至今没有
任何真实图的数值证据，只有 lit。**

排序上的含义：最大的单一卡点是第 1 条，第二是第 2 条（且它是对齐形状上的唯一卡点），第 3 条最便宜。
原先那份「A2 + A1 占 21 / 21」的排序作废。

### 下一步（大颗粒）

1. ~~**broadcast 读 `exec_condition`，生成 cache 条件与 `if` 分支。**~~ `B0Y0` 上 `load0` 与
   `brc` 的 `exec_condition` 是 `1`（`kCacheBlockSplitFusedBroadcastAxis`，
   `ascendc_ir_def.h:223`），bridge 一直没读它。它说的是「整块 UB 只算一次、可以复用」，
   不读等于每趟重算——慢，但不错（所以它是性能项，不是正确性项）。要做的是把这个条件接成
   生成侧的一个 `if` 分支。

   > **已落地（2026-09-30）。** 三段：`exec_condition` 进 `afir.sched`（`AFIRAttrs.td` 的
   > `ScheduleAttr` 多一个枚举参数，bridge 在 `ScheduleToAfir` 的 `ExecConditionStr` 一处
   > 翻译整个上游枚举）；谓词由 `TransferPlan` 的 `materializeExecConditionGuard` 按轴关系建
   > （与 legacy `GenerateEnCacheCondition` 同一个式子，ingredient 与
   > `materializeTailAwareTileSize` 是同一组）；`scf.if` 由新 pass `afir-exec-condition`
   > 包出来，排在 `afir-pipeline-sync` **之后**——被跳过的只有算子本身，
   > `AllocTensor` / `EnQue` / `DeQue` / `FreeTensor` 与流水屏障每趟照发，与 legacy 的
   > `Loop::GenerateBody` 逐字一致。
   >
   > 六条广播 CPU E2E 逐元素全对，生成 kernel 与改动前的差异只有守卫本身加 SSA 重编号。
   > `cache_block_split_origin_brc_axis` 那一格今天没有生产者（`MarkNodeCacheable` 无条件
   > 写 fused），只有 lit 覆盖。
2. **camodel 仿真。** 部件齐备（`RUN_MODE=sim` + `ccec` + `libruntime_camodel.so`），
   它是第二条并行的构建 + 运行路径，独立 PR。没有它，`Or` → `Copy` 这类改动只有推理没有数字。
3. **分析那几条「报不支持」各自对应什么真实场景。** 今天它们都是保守边界：判据推得出来、
   但没有用例，于是宁可回落 legacy。要收哪一条，先得知道它对应的图在真实模型里长什么样：
   - 广播段之间相邻（目的地在两段之间有填充，`[brc][brc][normal]`）；
   - Load / Store 合轴后段数 > 1（要发 ND-DMA），以及那一段的局部 stride 不为 1。

### 遗留问题清单（2026-09-29 立）

按「解一个、放一条用例」的顺序记。三条彼此独立，B-2 / B-3 今天被 B-1 挡在后面，但 B-1 一解就都活。

**状态：B-1 ~ B-7 全部已修（2026-09-29）。** 顺序上 B-1 一解就把 B-2 / B-3 顶到台面上；
B-3 先按「宁可不支持」拒掉，B-5 把偏移真算了出来，B-7 把它折回原来那条乘法，
B-2 把 `MTE2 → MTE3` 接上。形态 A 与形态 B 因此都有了 CPU E2E，而既有用例的生成 kernel
与改动前逐字相同（65 份里 62 份相同，不同的三份正是分段广播——改动前那三份或算错或生成不出来）。
下面每条按「落点 / 现象 / 验收 / 已修」记，留着是因为它们各自的根因与判据还在代码里。

**B-1　一个 stage 里模板不唯一时选不出来（阻塞项）——已修**

- 落点：`bridge/PlanSelector/PlanSelector.cpp:37`（`HasUniqueCandidatePerSubgraph` 要求每个
  stage 恰好一个模板）→ `bridge/PlanSelector/AttCostEvaluator.cpp:162` 报
  `ATT did not provide a complete, constant-cost plan for subgraph 0`，根因在
  `AttCostEvaluator.cpp:31` 的 `IsConstExpr()` 判否：代价式里带 ATT 求解器自己的未知量
  （`load0_Load_AIV_MTE2_contrib`）。
- 现象：分段广播两种形态、四个形状全部卡在这里，一条 CPU E2E 也加不了。
- 影响面不止广播：reg 侧 (d)（`load_neg_reg_base_store` 之外多一个 `nddma` 模板）是同一件事。
- 验收：形态 A / 形态 B 各一条 CPU E2E 能由代价模型自己选出模板并跑通。

> **已修（2026-09-29）：代不出常数时保留全部候选，把选择交给生成的 tiling 在运行期做。**
>
> 落点 `bridge/PlanSelector/PlanSelector.cpp` 的 `SelectStaticWinner`：
> `SelectWinnersFromModels` 返回 false 时不再报失败，而是 `selections = candidates`。
>
> **这不是盲挑一个**。`PlanToAfir` 本来就按 `template_counts_per_stage` 建多模板的 AFIR
> （`ScheduleToAfir.cpp:1054`），ATT 随后给每个模板各生成一份 tiling 与 score func，由生成的
> host tiling 按实际 shape 比分——**legacy 压根不在编译期选模板**，它在 tiling 函数里选
> （生成的 `tiling.cpp` 里的 `autofuse_tiling_data_best` / `PgoSolverGenerateAllTilingDataInner`）。
> 挑第一个才是盲挑：那等于拿输入顺序冒充性能结论。**「能代出常数就还在编译期定」这条路
> 没被吃掉**，由 `MultipleTemplatesRequireModelsEvenWithOnePlan` 反向钉住。
>
> 为什么代价式代不出常数不是异常：里面除了 shape 符号，还有 ATT 自己的求解量——分核数
> `block_dim` 与逐轴切分尺寸，都要等 tiling 解完才有值，而这一步跑在 tiling 之前。
>
> 覆盖：`PlanSelectorPolicyTest` 新增 `SolverBoundCostsDeferSelectionToRuntime`（一个 plan
> 两个模板，两个都留）与 `SolverBoundCostsRetainEveryPlan`（两个 plan 都留）。变异验证：
> 把回落改回 `return false`，这两条各自变红，其余 6 条不动。
>
> **实测效果**：形态 A / 形态 B 四个形状的计划选择全部通过，不再报
> `ATT did not provide a complete, constant-cost plan`。保留全部候选意味着 `Y1` 那份模板也要
> 下降，当时因此顶出了 B-2（形态 A）与 B-3（形态 B），两条随后都修掉。
> kernel 里多一份模板的代码与一次运行期比分是这条路的固有代价，**与 legacy 同一设计**
> ——legacy 也在生成的 tiling 函数里按实际 shape 比分，不在编译期定。

**B-2　`MTE2 → MTE3` 的流水线同步没建模——已修**

- 落点：`conversion/AFIRToAscendC/AFIRPipelineSync.cpp:425`。
  `synchronizeTransition` 只认 `MTE2→Vector`、`Vector→MTE3`、`Vector→Vector` 三种交接，
  其余一律 `emitError`。
- 触发：形态 A 的 `Y1` 把广播轴排到 tile 外，`RemoveRedundantBroadcastNode` 把 `brc` 删掉，
  图退化成 `Load → Store`——中间没有 V 计算，于是是一条 `MTE2 → MTE3` 的直接交接。
  两个形态 A 形状都必撞，报错报在 `load0` 上。
- **不是广播独有的**：撞的时候图里已经没有广播 op 了，任何「搬进来直接搬出去」的图都是这条路。
- 验收：一条 `Load → Store` 的 lit 正例（`MTE2 → MTE3`），加上形态 A 的 `Y1` 能生成。

> **已修（2026-09-29）。**
>
> 落点 `AFIRPipelineSync.cpp` 的 `synchronizeTransition`：`MTE2 → MTE3` 走
> `insertQueueSync`——那个函数本来就够用（producer 是 MTE2 就发 `PipeBarrier<PIPE_MTE2>`，
> 释放屏障按 consumer 选 `PIPE_MTE3`），缺的只是分派。
>
> **这一档不做 barrier 回落**：`PipeBarrier` 只在同一条流水内排序，MTE2 与 MTE3 之间没有
> barrier 可用，顺序只能由队列的 enque / deque 给。拿不到队列就报错——放过去就是 Store
> 可能读到还没搬完的 UB。
>
> 修它的过程中发现**偏移的权威是 GM 侧 descriptor，不是局部的**（B-5 原先的 premise 不全对）：
> 形态 A `Y1` 的 Store 局部 stride 在 `z0` 上是 0（那是被删掉的广播语义，每趟重读同一块 UB），
> 而 GM 目的地每趟前进 8，只看局部会漏掉 `z0` 那一项、输出只有第一行对。但**也不能只看 GM**：
> 图级合轴之后 GM 那几根原始轴根本不在调度链上（`z0z1z2` 是 `Original` 且不记 `from`），
> 这时只有局部 descriptor 答得出来。所以规则定成「GM 知道这根轴就用 GM 的，否则用局部的」
> （`OffsetContext::strideOf`）。
>
> 另外修掉尾块判定的同一类缺陷：`materializeTailAwareTileSize` 原先拿**线性序号**比
> 「是不是切分轴上最后一块」，`Y1` 里线性序号是 `z0` 与 `z1T` 合起来的，于是只有整张图的第一行
> 命中尾块，其余每行的尾块趟都按整块长度搬——越界写进下一行（被下一趟覆盖，中间几行看不出来），
> 最后一行直接写出缓冲区。改成拿**切分轴自己的坐标**比。
>
> 覆盖：
> - CPU E2E 新增 `load_broadcast_segmented_store_1x8`（形态 A）。三个形状 × 两个模板
>   （强制选择）共 6 种组合逐一验过数值。
> - lit `afir-transfer-load-store-mte2-mte3.mlir`（真实 dump 的形态 A `B0Y1` 模板）钉队列那一对；
> - `demo/afir_pipeline_sync_unsupported.mlir` 从「整档不支持」改成「缓冲是 TBuf 时报错」，
>   继续钉住不做 barrier 回落这条。
>
> **E2E 仍然只跑到运行期 tiling 选中的那个模板**（都是 `Y0`），`Y1` 只有上面两份 lit 钉得住。

**B-5　向量化轴被剔空时，GM 偏移要按非向量化的循环轴算出来（B-3 的正面解法）——已修**

- 落点：同 B-3（`TransferPlan.cpp` 的 `segments.empty()` 一档）。B-3 只是把这一档拒掉，
  没有把偏移算出来。
- 要什么：GM 偏移 = 循环归纳变量按合轴关系拆回各原始轴之后，逐轴坐标 × GM 侧 stride 之和。
  真实形状 `[1024,1] → [1024,128]` 的 `B0Y1` 里，循环走的是 `z0z1Tb`（`z0` 与 `z1T` 合成再分核），
  要用 `Ceiling(128 / z1t_size)` 做除 / 模把 `z0` 拆出来。
- 为什么单独立项：`materializeScheduledTileIndices` 今天只认向量化轴的分段，没有「循环轴 →
  GM 偏移」这条路；而拆合轴的除 / 模算错是静默的。
- 验收：形态 B 的 CPU E2E（`[8,1] → [8,128]` 与 `[1024,1] → [1024,128]`）算得对，
  且 B-3 那份反例 lit 从「拒掉」改成正例、断言偏移里有那条除 / 模。

> **已修（2026-09-29）**，但**范围比这一条原先写的大**：同一个缺陷在**两个分支里都有**，
> 分开修就是两份会漂移的同一公式，所以把搬运偏移统一成了一条式子。
>
> **统一后的式子**（`TransferPlan.cpp` 的 `materializeTensorOffset`）：
>
> ```
> GM 偏移 = Σ 坐标(a) × strides[a]      // a 取 descriptor 里非向量化的轴
> ```
>
> 判据来源与 legacy 相同（`LoadApiCall::Generate` 的
> `tiler.Offset(current_axis, ub.axis, ub.axis_strides)`）：偏移取自**张量自己的 descriptor**，
> 逐轴累加。`strides[a]` 记的就是被搬那个张量的逐轴元素步长（Load 的 `result_descs`、
> Store 的 `operand_descs`，与 GM 侧 descriptor 在 `Y0`/`Y1` 的 Load/Store 四处都核对一致）。
> stride 是字面量 0 的轴整项跳过。
>
> **坐标怎么来的**：`materializeAxisIndexFromLinear` 从线性序号
> （`分核序号 × 块内长度 + 归纳变量`）按图上轴树逐层拆（行主序：某一支的跨度是它内侧各支
> 循环次数之积）。**合轴与切分对是同一件事**——`Merged` 轴的各支取 `from`，被切分的轴取它的
> `(outer, inner)` 对，两者都是一层行主序拆解，所以 `axisChildren` 一个函数覆盖两种。
> 循环次数取图上那根轴的 `size`（切分轴写成 `Ceiling(...)`，与 `AxisValues::loopSize` 同一个量）。
>
> **不从 IR 里找回坐标**：`LoopGenerate` 确实物化过每根源轴的坐标并打了 `#afir.axis_binding`
> （`generateMergedAxisSourceIndices`），但那些值今天没有消费者，`afir-memory-allocate`
> 之后作为死代码消失（实测：`axis = 0, role = axis_index` 在 `02_afir-loop-generate` 的 dump 里有、
> `03_afir-memory-allocate` 里没有）。所以按图重算，不依赖某个值活到 conversion。
> **公式与 `LoopGenerate` 重复**是已知的，收敛它见 B-7。
>
> **为什么两个分支都要改**：原先「一段都不剩」那一档返回 0（B-3），另一档返回
> `线性序号 × 段长`。后者只在「段里只有切分轴一根、且循环轴恰好是它的分块」时成立。
> `[1024,1] → [1024,128]` 的 `size(z1T)` 正好是 1，那条乘法碰巧也对——所以**只测这个形状
> 测不出问题**；`[8,1] → [8,128]` 的 `size(z1T) > 1`，Store 每趟该走
> `z0 × 128 + z1T × z1t_size`，按段长算成 `L × z1t_size`，**静默算错**（先只改了 Load 的那一版
> 实测如此：Load 对了、Store 还错，CPU 模拟器从第 128 个元素起不符）。两个形状都进了用例。
>
> 覆盖：
> - CPU E2E `load_broadcast_segmented_store_8x1` / `_1024x1`（形态 B 两个形状）。
> - lit `afir-transfer-collapsed-tile-gm-offset.mlir`（真实 dump 的 `B0Y1` 模板，
>   钉 Load 偏移是 `(线性序号 / size(z1T)) % 1024 × 1` 而不是常量 0）。
> - lit `afir-broadcast-two-segment.mlir` 的 Store 段 CHECK 改成逐轴两项相加。
> - 变异验证：把逐轴坐标退回「线性序号」本身，上面那条 lit 与 `_8x1` 的 E2E 各自变红。

**B-7　逐轴偏移没做强度削减：既有 kernel 每趟多几条整数运算（性能项）——已修**

- 现象：B-5 把偏移统一成逐轴式子之后，**62 份既有 CPU E2E 的 kernel 文本全部改变**
  （此前那条「逐字不变」的性质被这次改动打破，实测 62 份全不同、但 62 条 E2E 数值全过）。
  以 `load_add_store_128x23x5` 为例，每个搬运每趟从 1 条 `muli` 变成
  `divsi` + 2 × `remsi` + 2 × `muli` + `addi` 共 6 条（另加 3 条循环不变量，可被下游提出循环）。
- 为什么会这样：切分对的两项 `coord(outer) × (b × s) + coord(inner) × s` 在
  `stride(outer) == stride(inner) × 循环次数(inner)` 时可以约回 `线性序号 × s`——也就是旧式子。
  逐轴式子没做这一步约化。
- 修法：在 `materializeTensorOffset` 里加一条**步长连续性**的折叠（与
  `collapseVectorizedAxes` 合并向量化轴用的是同一条判据）：相邻两支满足
  `s_i == s_{i+1} × 循环次数(a_{i+1})` 时折成一项、用父层的线性序号。需要 ExprAttr 的符号判等
  （`Broadcast.cpp` 里有 `provablyEqual` 可参照）。
- 收益：既有 62 份 kernel 恢复逐字不变——那同时是一条比「E2E 数值全过」更强的等价性验证。
- 顺带：收敛 B-5 与 `LoopGenerate::generateMergedAxisSourceIndices` 的同一公式，
  抽成两边共用的 helper。

> **已修（2026-09-29）。**
>
> 落点 `TransferPlan.cpp` 的 `axisUnitStride`：求一根轴的**单位步长**——整棵子树的偏移能写成
> `index(R) × unit` 时返回那个 `unit`，写不成就报失败。递归两条：
>
>   - `R` 自己在 descriptor 里 → `unit = strides[R]`（到底了）；
>   - 否则各支各自要能折成 `unit_i`，且**相邻两支步长连续**
>     `unit_i == unit_{i+1} × 循环次数(a_{i+1})`；成立时整段折成最内那支的 `unit`。
>
> 这条连续性判据与合轴合并向量化轴用的是同一条，所以把 `VectorSegmentation.cpp` 里那个
> 文件内的 `multiply` 提成了 `SymbolicBuilder.h` 的 `multiplyExprs`（第二个消费者出现），
> 判等直接用 `ProveExpressionsEqual`，不再加第三个 `provablyEqual` 包装。
> **不能拼字符串造乘积**——`Expression::Parse` 只收规范形式，那样会静默判不成立。
>
> 顺带修掉一处「指令数没变但文本变了」：线性序号改成**按需建**。偏移恒为 0 时（广播源那一档）
> 一条 Op 都不发，否则那条 `分核序号 × 块内长度` 会从 Store 处提前到 Load 处。
>
> **验证**：
> - 与 **HEAD**（本轮改动之前）逐份对比生成 kernel：**62 / 64 逐字相同**，不同的两份是
>   分段广播——HEAD 那两份正是算错的那个版本。这比「E2E 数值全过」强：它证明既有
>   60 条 elementwise 用例的输出一个字节都没动。
> - 变异验证两个方向都有用例挡着：
>   - 折叠整个关掉（退回纯逐轴式子）→ `afir-broadcast-two-segment.mlir` 红；
>   - 去掉连续性判据（无条件折成最内那支）→ `afir-transfer-collapsed-tile-gm-offset.mlir` 红。
> - **E2E 在第二个方向上是盲的**：`Y1` 那份模板由运行期 tiling 选，E2E 跑到的是 `Y0`。
>   所以那条判据只有 lit 钉得住——加分段档的用例时别只看 E2E 绿。
>
> **重复的处置（2026-09-29）**：
>
> - **文件内那一份已合掉**。`materializeAxisOffset`（逐支求和）与
>   `materializeAxisIndexFromLinear`（尾块判定要切分轴自己的坐标）原先各写了一遍行主序展开，
>   现在共用 `materializeChildCoordinate`。
> - **与 `LoopGenerate::generateMergedAxisSourceIndices` 那一份不合**，判据写在
>   `materializeChildCoordinate` 的注释里：那边的循环次数是它自己早先物化好的
>   `AxisValues::loopSize` 值、跨度在进循环之前就算好复用；这边只能从图上的 `size` 表达式重新
>   物化（那些值没有消费者，`afir-memory-allocate` 之后已被 DCE），且必须发在
>   `ConversionPatternRewriter` 的当前插入点。强行共用要么把那边的乘法挪进循环（62 份既有
>   kernel 的文本全变），要么把跨度当参数传进来（共用的就只剩 `divsi` + `remsi` 两条）。
>   两条都不划算。

**B-6　多模板时 `tiling_key` 取图级，会撞重复（已被诊断挡住，不是静默）——已修**

- 落点：`bridge/ScheduleToAfir/ScheduleToAfir.cpp:711`：
  `info.tiling_key = graph.GetTilingKey() >= 0 ? graph.GetTilingKey() : templateId`。
  图级 key 对一个 stage 里的**每个**模板都返回同一个值。
- 现象：`GenKernelEntry: duplicate tiling_key in stage 0`
  （`conversion/AFIRToAscendC/GenKernelEntry.cpp:227`）。**报得出来，不会静默**。
- 触发条件：调用方在输入图上调过 `AscGraph::SetTilingKey`，且某个 stage 有多个模板。
  E2E 用例都调了 `SetTilingKey(0)`（此前每个 stage 只有一个模板，撞不上）；B-1 修完之后
  形态 B 立刻撞上，去掉那一行就过了。
- 验收：报错时说得出成因；`tiling_key` 的规则与 ATT 一致这件事写进代码。

> **已修（2026-09-29）。查清之后结论是：AFIR 那条规则本来就是对的，缺的是诊断。**
>
> ATT 那边的规则（`att/gen_model_info/gen_model_info.cpp` 的 `GenerateModelInfoWithContext`）是：
>
> ```
> uint32_t tiling_key = 0;
> for (auto &graph : graph_list) {        // graph_list 就是一个 stage 的 impl_graphs
>   tiling_key = graph.GetTilingKey() >= 0 ? graph.GetTilingKey() : tiling_key;
>   model_info.tiling_case_id = tiling_key;
>   ...
>   tiling_key++;                         // 循环末尾递增，一开始没看到这一行
> }
> ```
>
> 图上没设 key 时那个计数器就是 stage 内的模板序号，所以 `ScheduleToAfir` 取 `templateId`
> **与 ATT 等价**；设了 key 就两边都用它。`CheckKeyValid` 要求一个 stage 内全设或全不设，
> 所以「前一个设了、后一个没设」那种会让计数器与 `templateId` 分叉的组合进不来。
>
> **所以撞重复不是 AFIR 的缺陷，是输入本身不成立**：图级 key 被每个 impl_graph 继承，一个 stage
> 出多个模板时所有模板拿到同一个值，**host 侧同样分不开**（ATT 只查「全设或全不设」，不查重）。
> 按 `templateId` 忽略图级 key 会更糟——host 写图级 key、device 比模板序号，变成静默不匹配，
> 一个模板都选不上。
>
> 修法因此是诊断：`ScheduleToAfir` 在建 AFIR 时就查一个 stage 内 key 是否互不相同，撞了就报错
> 并把成因和改法说出来（「去掉图级 `SetTilingKey`，让 per-template key 落回模板序号」），
> 不再等到 `GenKernelEntry` 以「key 重复」的形式冒出来——那条诊断看不出该改哪儿。
> `GenKernelEntry` 的检查留着兜底，消息里也补上成因。
>
> 覆盖：`ScheduleToAfirTest.RejectsTemplatesSharingAGraphLevelTilingKey`（两个 impl_graph 共用
> 图级 key 0 → `PlanToAfir` 失败且诊断里有 `templates sharing tiling_key 0`）。
> 变异验证：去掉那条查重即变红。已有的 `DefaultsUnsetTilingKeysToTemplateIds` 从反方向钉住
> 「没设 key 时落回模板序号」。
>
> 顺带：本轮新增的 E2E 用例（`load_broadcast_segmented_store`）不调 `graph.SetTilingKey`，
> 理由就是这条——它的 stage 有两个模板。

**B-3　`segments.empty()` 那一档漏了 GM 侧 stride 判据——静默算错（已修，后由 B-5 从"拒掉"改成"算对"）**

- 落点：`dialect/AFIR/Analysis/TransferPlan.cpp:317`：

  ```
  if (segments->segments.empty()) {
    // 一段都不剩：…… 这块 tile 就是一个元素，地址也不随 tile 前进。
    return ScheduledTileIndices{zero, ... 1};
  }
  ```

  「局部 tile 只有一个元素」推不出「GM 地址不前进」，而这一档**直接返回偏移 0**。
  2026-09-23 文档需求 5 原本写了这条判据——「偏移取 0（**并要求 GM 侧 descriptor 的 strides
  全是字面量 0**，否则「偏移恒为 0」这句话没有依据）」——实现里只留下了取 0，判据没落地。
- 现象（形态 B 的 `Y1`，两个形状都是）：生成得出来、编得过、跑得完，**结果错**。
  `[1024,1] → [1024,128]` 的第 128 个元素起全部等于 `input0[0]`，应当是 `input0[1]`。
  生成的 kernel 里 Load 在循环内恒读 `v28[c0_idx]`（偏移常量 0），而 Store 的偏移在走
  （`v29[v84]`）——每趟把 `input0[0]` 铺满一行。
- **今天已加的两条 `Duplicate` E2E 抓不到它**：它们的 GM 源本身就只有一个元素，
  偏移 0 是对的。
- 验收：补上判据（不满足时报失败、回落 legacy），反例 lit 钉住判据不满足时不被 legalize。

> **已修（2026-09-29）**，但**原先记的判据本身不准，两句话都要改**。
>
> 一是「GM 侧 strides 全是字面量 0」太严也太松：`[1,1] → [8,128]` 那条 Duplicate 正例的
> GM 源是 `repeats = [1,1], strides = [0, ...]`，按「strides 全为 0」会把它一起拒掉。
> 真正的判据是**地址走不动**：GM 侧每一根轴要么 `repeat` 是字面量 1（只有一个元素），
> 要么 `stride` 是字面量 0（走了也是同一个地址）。符号 `repeat` 一律算「可能大于 1」。
>
> 二是原注释里「这块 tile 就是一个元素」也不对。真实 dump（`B0Y1`）的局部 tile 是
> `1024x1x1`——**1024 个元素**，只是只有一根轴（`z1t`）参与向量化、而它 extent 是 1。
> 被剔空说的是向量化轴，不是整块 tile。地址走的是 `z0` 与 `z1T` 合成那根**非向量化**的循环轴。
>
> 落点：`dialect/AFIR/Analysis/TransferPlan.cpp` 的 `segments.empty()` 那一档，新增
> `getGlobalDescriptor`（与 `getLocalDescriptor` 严格互补：Load 取 `operand_descs`，
> Store 取 `destination_descs`）。
>
> 覆盖：`test/lit/conversion/afir-transfer-collapsed-tile-gm-advances-unsupported.mlir`
> ——**直接用真实 dump**（那个模板原样，只把 template 序号与 tiling_key 归 0），钉
> `failed to legalize operation 'afir.load'`。变异验证：删掉判据即变红。
> 两条 Duplicate E2E 照旧绿。
>
> **后果：形态 B 的 MLIR 路径因此整图回落 legacy**（`Y1` 那份模板拒了，整个 plan 模块下降失败）。
> 也就是说 B-3 修掉的是「静默算错」，换来的是「明确不支持」——正确，但形态 B 的 E2E 仍然
> 加不上，要等 B-5。

### 测试命令

```bash
bash ./mlir-test.sh
```

预期：全量绿；CPU E2E 计数 **66**（本需求开始时 60，广播用例从 0 补到 6 条，见「测试覆盖现状」）。

---

## 验收标准

1. `collapseVectorizedAxes` 是阶段 2 合轴在 MLIR 侧的唯一实现；`vectorized_strides.front()`
   的 0 / 1 分流被删除而不是保留；`TransferPlan` 的单轴字段**改名**为 `tileInnerAxis`
   （`getTileInnerAxis` 一并改名）。字段留着不是残留的单轴假设：搬运的长度与地址都改成从分段
   推了，但**尾块修正必须知道哪一根轴在最后一趟取尾长**，那是一根轴的属性、分段答不了。
   「取最外那根向量化轴」这条判据的射程写在 `getTileInnerAxis` 的注释里。
2. `afir.broadcast` 单广播段一档在 **mem 侧**有一个 implementation（reg 侧本轮不做，功能需求 6），两种段方向各有 lit 正例。
3. `afir.broadcast` 多广播段分解有 lit 正例两份（四段 `(1,B,1,B)` + 五段），断言
   「N+1 条 `Broadcast` + N 条 `PipeBarrier`，外层那条的目的偏移一整块、repeat 为 `extent-1`」；
   **覆盖面严格超过 legacy 四轴**（五段那份 legacy 会 `ASSERT`）。
4. 发出的每一条 `Broadcast` 调用第三实参恒为 `1` 或第五实参恒为 `0`（功能需求 7 的不变量），由 lit 钉住。
5. **现有 CPU E2E 的生成 kernel 文本逐字不变**（Load / Store 改动之后实测：60 份逐份 `diff`，相同 60、不同 0），`bash ./mlir-test.sh` 全量绿，E2E 计数不变。基线是 **60** 个 `check-e2e-cpu-*`（2026-09-28 实测；前序文档写的 47 是当时的数，已过期）。
6. `afir-broadcast-multi-axis-unsupported.mlir` 从反例改为正例（更名
   `afir-broadcast-two-segment.mlir`），**descriptor 一字未改**，且跑完整条下降 pipeline：
   零层额外循环、一条 `Broadcast`，外加 Load / Store 各自读对了同一批 stride。
7. 假设表里「counter 模式 `Copy` 的目的张量余量」这条已核（探针结论 P6）：它是 CPU-debug
   检查的保守模型，不是正确性前提，公式已测出并记入文档。因此**不**据此退回 unsupported，
   也不新增编译期兜底；判据里的 `Z` / `K` 在真实图上是符号，编译期判不了。
8. 本文档「探针结论」一节的每条结论在正文中可追溯到 file:line；`RemovePad`、camodel 性能验证、
   计划选择、`exec_condition` 四项在"本次不做"表里各有理由与落点。
