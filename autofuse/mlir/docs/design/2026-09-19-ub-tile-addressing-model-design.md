# UB tile 寻址模型：现状缺陷与通用化方向

> **接口迁移说明（2026-10-22）**：正文提及的 `AF_MLIR_PINNED_*` / `AF_MLIR_ENABLE_PLAN_PINNING` 是历史接口，现已移除；当前使用 ATT `AUTOFUSE_DFX_FLAGS` 的 `--force_schedule_result` / `--force_tiling_case`。候选编号须核对当前 shape 的 schedule dump。

**分支**：`feat/mlir-reduce-common-fusion`（基线 `upstream/af-mlir` = `0af07117`）

**状态**：**已实施**（2026-10-03 复核）。§1–§8 是调查结论，§9 是落到文件与函数的实施方案。
§9 的四个动作都已落地：步骤一的统一偏移原语与步骤二的 elementwise 二维跨度由
[2026-09-22-elementwise-multidim-calcount-design.md](2026-09-22-elementwise-multidim-calcount-design.md)
接完；步骤三的 reduce 非队列前置条件已去掉（`Reduce.cpp` 今天的门禁是「两端都在 UB」，
不再问容器），§4 的 **Reduce 后融合**随之解开——语料是 `load_sum_relu_store`（一元、不切图）、
`load_sum_mul_store`（带 `Scalar`、切图）、`load_sum0_mul_store`（首轴 + 切图）与
`load_sum_add_store`（二元、切图）。§7.2 探针表里 `load_add_sum_store` /
`load_sum_add_store` 两行也已于 2026-10-03 注册并转绿。
两部分都还没有代码落地。

**读者**：接手 reduce common 模板正向疏通、或后续要接入 transpose / concat / gather 等搬运类算子的人。

---

# 1 结论先行

三件事，根因是同一个：

1. **主线上有一个正确性缺陷**，与 reduce 无关，已有复现用例：VecCalc 中间值落进 TBuf、tile 是一维、
   且 tile 循环跑多于一轮时，生成的 kernel 会越界读写，输出错值。见 §3。
2. **两个融合场景被挡住**：Reduce 前融合（elementwise 拿到二维 tile）与 Reduce 后融合（reduce 的
   destination 不再是队列）。两者卡的都不是"算不出地址"，而是寻址被按容器和算子分叉实现，分叉出来的
   格子没人填。见 §4。
3. **接口形状不可扩展**：`ScheduledTileIndices` 是照 `DataCopyPadExtend` 的参数表设计的六元组。
   elementwise 硬套它，reduce 绕过它，而 transpose / concat / gather 三类算子**都套不进去**。见 §5。

建议不要继续往六元组里加字段，而是拆成三个正交原语。方向见 §6，实施方案见 §9。

补充结论（§3.5 / §8.1，已查证）：`iv × tileSize` 错的不是系数而是**自变量**。TileOuter 轴被三道
相互独立的结构性屏障挡在 legacy 的 UB 偏移之外，所以那不是正确公式的退化版，是另一个量。同时，
统一公式对**调度器生成的循环**恒为 0 是可证的（屏障三），真正非 0 的实例来自**算子自己发的循环**
（gather），这决定了偏移必须是**函数**而不是预计算好的字段。

---

# 2 现状：寻址的三层

一个张量在 UB 上定位"本轮要处理的那块数据"，实际由三层决定：

| 层 | 是什么 | 谁决定 | 今天在哪实现 |
|---|---|---|---|
| ① 容器基址 | 这一次取到的是哪一块内存 | 容器（`TQue` / `TBuf`） | `que_bind.alloc_tensor` / `deque_tensor` / `tbuf.get_tensor` |
| ② 资源内成员偏移 | 同一资源里多个张量并排摆时各自的起点 | 资源布局 | `CanonicalMaterialization.cpp` 的 `slab.offset` 游标 |
| ③ 本轮 tile 偏移 | 本轮循环要处理的是哪一块 | **今天按容器分叉** | `TransferPlan.cpp` 的 `localOffset` |

①②对两种容器一视同仁。**按容器分叉的只有第③层，也正是出问题的那层。**

## 2.1 第③层今天有三套规则

`materializeScheduledTileIndices`（`dialect/AFIR/Analysis/TransferPlan.cpp`）：

```cpp
Value localOffset;
if (plan.tiles.size() == 1) {
  localOffset = iv * runtimes.front().size;   // 一维
} else {
  localOffset = 0;                            // 二维，注释写着「没有已验证的落点规则，不猜」
}
```

两个使用方（`Implementations/DataMovement.cpp`、`Implementations/BinaryElementwise.cpp`）再各自覆盖一次：

```cpp
Value localOffset = *localIsQueue ? zero : indices->localOffset;
```

合起来：

| 容器 | tile 维度 | localOffset | 是否被执行过 |
|---|---|---|---|
| 队列 | 任意 | `0` | ✅ 所有已注册 E2E 走的都是这条 |
| 非队列 | 一维 | `iv × tileSize` | ❌ 直到本次构造探针前，从未执行 |
| 非队列 | 二维 | `0` | ❌ 从未执行 |

## 2.2 两种容器为什么都不需要第③层偏移

理由**不同**，但结论相同：

- **TQue**：`AllocTensor` / `DeQue` 返回的就是本次该用的槽，轮转由队列内部按 `buf_num` 管理。
  注意轮转的粒度是"每次取张量"，**不是"每轮循环"**——`load_sum_store` 里 VecOut 的 `alloc_tensor`
  写在外层循环，整个内层循环共用同一个槽，因为那个槽是 reduce 的累加器，地址必须恒定。
- **TBuf**：`get_tensor` 返回整块起点，永不变；而 `b<id>_size` **只有一个 tile 那么大**——ATT 侧
  `asc_graph_ub_expr_builder.cpp` 里 buffer 走 `AddContainer(name, ContainerSlotBytes(...))`，
  **不乘 `buf_num`**（队列才乘）。整块就是本轮那一个 tile，每轮原地覆盖重用。

所以 `iv × tileSize` 假设的是第三种布局——"UB 上按轮次并排摆着所有 tile"——**分配器从来没有这样分过
内存**，队列不是，buffer 也不是。

## 2.3 证据：dump 里它是一个死值

`build/sum_probe/.../06_convert-afir-to-ascendc.mlir` 的 Store 现场：

```mlir
%95 = arith.muli %89, %11 : index          // GM 偏移，被 %98 用了
%96 = arith.muli %arg4, %11 : index        // ← iv × tileSize，全文件只此一处，无使用者
%c0_17 = arith.constant 0 : index
%97 = ascendc.local_tensor.subindex %87[%c0_17]    // 用的是 0，不是 %96
```

这份 dump 里 UB 侧 8 个取址点偏移**全是字面量 0**，唯一算出来的偏移在 GM 侧。

---

# 3 缺陷：`iv × tileSize` 会越界（已复现）

## 3.1 触发条件

三条同时满足：

1. 图里有 VecCalc 中间值落进 **TBuf**——需要 ≥2 个连续向量算子，且该中间值拿不到队列槽
   （`MemReuseManager::FindCandidateQueBlockWithSizeCheck` 对 VecIn/VecOut 成员要求
   `info->merged_life_end < tensor_group.merged_life_start`，**严格小于**；紧接 Load 的第一个中间值
   生命起点等于 Load 的终点，拿不到，只能落 `kCalc`）；
2. tile 是**一维**（纯 elementwise 调度）；
3. 数据量大到 **tile 循环跑多于一轮**。

## 3.2 复现

用例已在分支上：`test/e2e/cpu/cases/load_add_add_store/`（`Load → Add → Add → Store`，`[8192, 128]`），
已注册进 `test/e2e/cpu/CMakeLists.txt`。构建是绿的（生成器成功），运行时数值失败：

```
第一个错误下标：index=9713      （0 ~ 9712 全对）
错误总数：      599094 / 1048576
错误的值：      output = 0，期望 -4
```

生成的 kernel：

```cpp
v15.InitBuffer(v16, v8);                                   // TBuf 只有 v8 = 一个 tile
AscendC::LocalTensor<uint8_t> v39 = v16.Get<uint8_t>();    // 循环【外】取一次
AscendC::LocalTensor<float>   v41 = v40.ReinterpretCast<float>();

for (uint32_t v57 = 0; v57 < v56; v57++) {
  uint32_t v74 = v57 * v12;                    // iv × tileSize
  AscendC::LocalTensor<float> v76 = v41[v74];
  AscendC::Add(v76, v75, v75, v67);            // add0 写 v41[iv×tile]，iv≥1 越界
  AscendC::Add(v80, v76, v76, v67);            // add1 从同一个越界地址读回
}
```

第 0 轮 `v74 = 0`，恰好正确；第 1 轮起写到 TBuf 外面，再读回来是未写过的内存。

## 3.3 为什么一直没暴露

- 现有用例里**没有任何 VecCalc 张量**：`load_add_store` 的 TBuf 数为 0，`load_sum_store` 的 2 个 TBuf 是
  reduce scratch，走 `Implementations/Reduce.cpp` 自己的路径。
- 同一个图的小形状（`[1024, 128]`）会通过——每核只有一个 tile，循环只跑一轮，`iv` 恒为 0，
  **恰好踩在唯一正确的那个点上**。
- 三个向量算子以上会先撞 bridge 的 `InvalidBinding: buffer domain cannot use AllocType::Queue`
  （调度器把 VecCalc 放进了队列槽，另一个问题，已交由他人处理），根本走不到这里。

## 3.4 与 legacy 的关系

legacy 没有这个量。它的 `Tiler::TensorVectorizedOffset` 算的是 **tile 内部**的位置
（`Σ 轴变量 × 向量化 stride`），永远落在已分配的那块里；MLIR 这个算的是**整 tile 的整数倍**，
`iv ≥ 1` 时必然出界。**不是"实现不同"，是这个量本身建模就错了。**

更准确地说：错的不是系数，是**自变量**。`iv` 是 TileOuter 轴的归纳变量，而 legacy 的寻址模型
用三道相互独立的结构性屏障把 TileOuter 挡在 UB 偏移之外（见 §3.5）。MLIR 这一版等于绕过整个
模型另起了一个量。

## 3.5 三道屏障：TileOuter 结构上不可能进入 UB 偏移

`Tiler::Offset`（`codegen_kernel.cpp:639-677`）的命中判据是 `IsFrom(ca, iter)`——问的是"当前循环轴
`ca` 是否**派生自**向量化轴 `iter`"。所以要证明 TileOuter 贡献恒为 0，光说"它不在 `vectorized_axis`
里"不够，还要堵住祖先链和循环轴两条路。三道屏障各堵一条：

**屏障一：TileOuter 不进集合。** `FindInnerAxes`（`optimize/autoschedule/schedule.cpp:202-216`）是
`vectorized_axis` 的唯一构造入口：

```cpp
for (const auto &axis : axis_group) {
  if (axis == ub_tiling_id) {
    find_tile_in = true;
    vectorize_axis.push_back(ub_tiling.second->id);   // 只推 .second
    continue;                                        // 原轴本身被跳过
  }
  if (find_tile_in) { vectorize_axis.push_back(axis); }
}
```

`.first`/`.second` 的含义在 `graph_metadef/.../ascendc_ir.cc:419-426` 固定：
`inner_type = kAxisTypeTileInner`、`outer_type = kAxisTypeTileOuter`，`return {outer, inner}`。
推进集合的只有 inner。

**屏障二：被切的原轴也不进集合。** 上面那句 `continue` 把 `ub_tiling_id` 本身跳过了。TileOuter 的
祖先链是 `{自己, ub_tiling_id}`（`CreateAxis(name, outer_type, outer_size, {axis_id}, inner_id)`），
两个祖先都不在集合里 ⇒ `IsFrom` 恒假。这条是闭合的，不是抽样观察。

**屏障三：循环轴的选取主动排除所有向量化轴。** `SelectLoopAxis`
（`optimize/autoschedule/autoschedule.cpp:35-41, 217-230`）：

```cpp
void FindNotLoopAxis(...) {
  for (auto output : node->outputs()) {
    not_loop_axis_set.insert(output->attr.vectorized_axis.begin(), output->attr.vectorized_axis.end());
  }
  ...
}
// ... 随后把 not_loop_axis_set 命中的轴置为 kIdNone，loop_axis 取剩下最内的那根
```

即 **`sched.loop_axis ∉ vectorized_axis` 是按节点构造出来的性质**。映射到 MLIR 就是
`plan.loopAxis ∉ {plan.tiles[i].axis}`——`plan.tiles` 正是从 `getVectorizedAxis()` 建的
（`TransferPlan.cpp:149-183`）。所以调度器发出的循环轴与向量化轴交集为空，统一公式对它们求值为 0
**不是碰巧，是第三道屏障的直接推论**。

其余 `vectorized_axis` 赋值点逐一核对过，都不构成反例：

| 位置 | 性质 |
|---|---|
| `schedule.cpp:714-720` 追加 `n_group` | `n_group` 定义即"不可切分轴，或者只能作向量化轴"（`inc/autoschedule/axis_group.h:23`），从不被 tile 切 |
| `indirect_load_schedule_case_generator.cpp:875-891` | `CreateFixedTileSplit` 同时产出 outer/inner，只有 inner 进集合，outer 另存 `axes.tile_outer_axis` |
| `schedule.cpp:566-587` `SetInputInnerVectorizedView` | 只收 `input_inner_axis` 或其 `from` 成员 |
| `v35/.../vector_func_partitioner.cpp:276` | 删 zero-stride 项，取子集 |
| `v35/.../vector_func_partitioner.cpp:353` | 把若干向量化轴合并成 `kAxisTypeMerged`，源全是向量化轴 |
| `nddma_template.cpp` 数处 | 从别的张量整体拷贝，继承 |

一个边界：`GetDefaultVectorizedAxis(axis, -1)` 返回整个 `tensor_axis`
（`graph_metadef/.../axis_utils.cc:229-236`），而 `TileSplit()` 主循环对 buffer 节点、
`skip_main_tiling` 节点、`preserves_vectorized_axis` 的 indirect-load 节点会 `continue`，这些节点保留
默认值。但默认值写进去时图里还不存在 tile 切分，所以不是反例。

---

# 4 两个被挡住的融合场景

| 场景 | 探针用例 | 今天的表现 | 卡在哪 |
|---|---|---|---|
| Reduce 前融合 | `load_add_sum_store` | `failed to legalize operation 'afir.add'` | reduce 调度下每个张量都有两根向量化轴，`BinaryElementwise.cpp` 要求 `tiles.size() == 1` |
| Reduce 后融合 | `load_sum_add_store` | `failed to legalize operation 'afir.sum'` | Sum 的 destination 变成普通 buffer，`Reduce.cpp` 要求 queue-backed destination |

两个探针都在分支上，**未注册**（注册后生成器失败会让整个构建变红，因为生成器是构建期 custom command）。

两点值得注意：

- 后融合里 **Add 那一侧其实是通的**：归约维在 destination 两侧 stride 都是 0，
  `buildScheduledPlan` 会跳过它，只剩一根 tile 轴。卡住的是 Sum。
- **legacy 支持后融合**：`ReducePartitionPostFusion` 只在 reduce 之后节点数超过
  `NODE_COUNT_AFTER_REDUCE = 4`、或后继链有外部输入时才切图；`Load→Sum→Add→Store` 只有 2 个节点，
  留在同一张图里。而 legacy 的 reduce 目的地寻址（`reduce_api_call.cpp`）**不看容器**，没有任何
  "必须是队列"的前置判断。所以这是 MLIR 相对 legacy 的**能力退化**，不是双方都没有的新场景。

---

# 5 接口形状：六元组套不进未来的算子

`ScheduledTileIndices`（`TransferPlan.h`）固定返回六个量：

```cpp
struct ScheduledTileIndices {
  Value globalOffset, localOffset;
  Value blockCount, blockLen;
  Value globalStride, localStride;
};
```

头文件注释自己写明了来历：「`DataCopyPadExtend` 的四个参数加两端的偏移」。**它是照一个具体 API 的参数表
设计的。** 于是：

- **elementwise 硬套**：只用得上 `blockLen`（最内一根轴的长度），二维时表达不了，所以直接拒绝。
- **reduce 绕过**：自己算 `srcShape = {first_actual, last}`，并额外要求 destination 是队列。

查了 legacy 里三类搬运算子实际需要什么，**六元组一个都装不下**：

| 算子 | 偏移用什么轴表 | 还需要什么 | 六元组能表达吗 |
|---|---|---|---|
| `transpose_api_call.cpp` | `TensorVectorizedOffset`（与 elementwise 同） | tmp_buf + 从 TilingData 取的 `apiTilingData` + `AutoFuseTransposeType` | ❌ 缺 API 级 tiling 参数 |
| `concat_api_call.cpp` | 不用单一偏移 | 每个输入和输出各一组 `ConcatParams{ .shape[N], .stride[N], .tensor }`，N = 向量化轴数 | ❌ 缺 N 维 shape/stride，且是多输入 |
| `gather_api_call.cpp` | `Offset(current_axis, x1.axis, x1.axis_strides)`——**张量级**轴表，不是向量化的；外加 `CalGatherOuterAxisOffset` 自己合成的一组轴与 stride | 自己发 `for` 循环；`param_offset` 由 indices 的**运行期值**算出 | ❌ 缺张量级偏移、自定义轴表、运行期偏移 |

## 5.1 legacy 的做法：一个原语，各算子自选轴表

legacy 只有一个寻址原语：

```cpp
Tiler::Offset(current_axis, axis, strides)   // = Σ 命中的轴: 轴变量 × stride
```

算子决定传哪张轴表和哪张 stride 表：

- elementwise / transpose → `(vectorized_axis, vectorized_strides)`，即 `TensorVectorizedOffset`
- gather → `(axis, axis_strides)` 张量级，以及自己合成的 `(gather_outer_axis, 推导出的 strides)`
- concat → 不调它，直接把 `(shape[], stride[])` 整体交给 API

长度同理，也是算子自己的事：elementwise 取线性跨度
（`TensorActualSize` = `Σ(size−1)×stride + 1`）、搬运取 DMA 四参、reduce 取 `srcShape`、
concat 取每输入一组 shape/stride、gather 自己发循环。

**容器（TQue/TBuf）在这套模型里只影响基址，不影响偏移，也不影响长度。**

---

# 6 建议方向：三个正交原语

把今天混在 `materializeScheduledTileIndices` 一个函数里的三件事拆开：

| 原语 | 回答什么 | 由谁决定 | 与什么无关 |
|---|---|---|---|
| **基址** | 这一次取到哪一块内存 | 容器 | 算子、tile 维度、循环轮次 |
| **偏移** | 在这块内存里从哪开始 | 算子选轴表 + stride 表，走统一的 `Offset` 原语 | **容器** |
| **形状 / 长度** | 本轮处理多大、什么排布 | 算子自己的 plan（已有先例：`ReducePlan`） | 容器 |

落到具体动作：

1. **去掉第③层的容器分叉**（两个使用方的 `? :` 整个删掉），换成 legacy 那个统一公式。
   在当前所有调度形态下求值为 0，所以**今天生成的代码一个字节都不会变**；变的是它从
   "碰巧没走到错误分支"变成"不存在错误分支"。§3 的缺陷随之消失。
   注意这里说的不是"保留 `localOffset` 字段、换个算法"——`ScheduledTileIndices` 里那个**字段本身**
   要删掉，换成一个可被算子调用的构造器。理由见 §8.1 推论 1，具体形状见 §9.1。
2. **elementwise 的长度改用线性跨度**，对齐 legacy。二维 tile 随之可用，§4 的前融合解开。
   注意 `BinaryElementwise.cpp` 那条注释（「一次 count 覆盖不到整块」）说反了：用跨度公式恰好
   **覆盖得到**，代价是顺带算了行尾 padding，而 padding 多算对 elementwise 无害——这正是 legacy
   敢用一条 API 平掉二维的原因。
3. **去掉 `Reduce.cpp` 的 queue-backed destination 前置条件**，§4 的后融合解开。
4. **能力查询与改写对齐**：`BinaryElementwise` 的 `queryLowering` 今天不检查 `tiles.size()`，
   于是 SelectImplementation 判"支持"、改写时才失败，报成 `failed to legalize operation` 而不是
   `UnsupportedShape`。Reduce 那边检查了自己的 plan，这里没对齐。
5. **新算子接入时不要扩六元组**，而是新增自己的 plan（`TransposePlan` / `ConcatPlan` / `GatherPlan`），
   复用基址与偏移两个原语。`ReducePlan` 已经是这个形状，可作模板。

---

# 7 分支现状与建议顺序

## 7.1 分支上已有的提交

```
feat/mlir-reduce-common-fusion   基线 upstream/af-mlir (0af07117)
  43f7582e  feat: AF_MLIR_PINNED_PLAN_ID 钉选 plan 候选
  2d295899  feat: 搬运支持二维 tile
  4cf42fdb  feat: afir.sum 的 buffer 形态与跨作用域累加
  47c11116  feat: afir.sum 下降为 ReduceInit + 高阶 ReduceSum + 累加
  af0d0ee5  docs: reduce 跨作用域累加设计按实施结果回写（已在工作区删除，见 7.3）
  42ce7963  fix: 按评审收紧 reduce 的契约与适用范围
  716720a2  test: 补四个融合缺口的复现用例，暂不注册
```

未带过来的四个（`origin/af-mlir` 上）：`73260d52`（符号下降阻塞）、`496a2a14`（删无用 bridge API）、
`07828528`（docs）、`c2bed123`（二维 Store stride 方向 + 评审债务）。缺东西时最可能缺前者与后者。

## 7.2 探针用例

| 用例 | 图 | 注册 | 今天的表现 |
|---|---|---|---|
| `load_add_add_store` | `Load→Add→Add→Store`，`[8192, 128]` | ✅ 已注册 | **数值错**，§3 的缺陷 |
| `load_add_sum_store` | `Load→Add→Sum→Store` | ❌ | `failed to legalize 'afir.add'` |
| `load_sum_add_store` | `Load→Sum→Add→Store` | ❌ | `failed to legalize 'afir.sum'` |
| `load_add_add_sum_store` | `Load→Add→Add→Sum→Store` | ❌ | bridge `InvalidBinding`（他人处理中） |
| `load_add_chain_store` | 四个 Add | ❌ | 同上，最小形态 |

## 7.3 建议顺序

1. 先修 §3 的缺陷（动作 1）。它独立于 reduce，`load_add_add_store` 现成就是验收用例，
   且修完之后现有用例的生成结果应当**逐字节不变**——这是一个很强的回归判据。
2. 再做动作 2 + 3，把两个探针注册上转绿。验收标准建议直接对齐 legacy 行为，而不是自定边界。
3. 动作 4 顺手做掉。
4. 动作 5 是接新算子时的约束，本期不需要实现。

落到文件与函数级别的实施方案见 §9。

---

# 8 未验证 / 风险

明确列出本次**没有**查证的部分，接手时不要当成结论：

- **~~统一公式在什么形态下非 0，没有找到实例。~~** 已查清，结论移入 §8.1，不再是未验证项。
- **二维非队列形态的落点**只做了推导，没有跑通数值。前融合修完后需要实测。
- **RA 形态与多归约轴**不在本文范围。RA 被双侧显式拒绝（改动点：`ReduceInit` 的发射条件、
  累加 count 的取法）；多轴在 `buildReducePlan` 就 `failure()`（要求恰好两根向量化轴、恰好一根被归约）。
  ARAR 还需要 UB 内轴调序，需要单独方案。
- **plan 选择**：不设 `AF_MLIR_PINNED_PLAN_ID` 时 reduce 图直接失败
  （`ATT did not provide a complete, constant-cost plan for subgraph 0`），RCommon 目前只能靠开发开关进入。
  与 RCore / AllLoad 模板一并单独讨论。
- **`ctest --test-dir <build> -N` 列不出这些 e2e 用例**，本次是直接跑测试二进制验证的。
  原因未查，可能影响"注册用例看红"的工作流。
- transpose / concat / gather 三节的结论来自**读 legacy 代码**，没有实际接入验证。

## 8.1 已结清：统一公式何时非 0

分两问，答案不同，**这个区分决定了接口怎么设计**。

**问一：`Tiler::Offset` 这个函数能不能返回非 0？能。** UT 直接给出了形态
（`tests/ut/codegen/test_codegen_kernel.cpp:311-350`，测试名 `WhenLocalTensor_VectorizedOnCurrentAxis`）：

```cpp
vectorized_axis = {z1, z2};          // tile 跨 z1、z2；z0 是 kAxisTypeTileOuter
TensorVectorizedOffset({z0})          == "0"
TensorVectorizedOffset({z0, z1})      == "z1 * s2"              // ← 非 0
TensorVectorizedOffset({z0, z1, z2})  == "z1 * s2 + z2"
```

注意 `z0` 在三行里贡献恒为 0——这正是 §3.5 三道屏障的直接体现。非 0 来自 `z1`，一根**向量化轴同时
成为当前循环轴**的情形。

**问二：调度器会不会造出这种形态？目前没找到，而且屏障三说明它造不出来。**
`SelectLoopAxis` 把节点自己的向量化轴全部排除出 loop_axis 候选，所以"调度器发出的循环轴恰好是向量化轴"
这件事按构造不会发生。**结论是：对调度器生成的循环，统一公式恒为 0，且这是可证的，不是观察到的。**

**但真正的非 0 实例确实存在，它走的是另一条路：算子自己发的循环。**
`gather_api_call.cpp:48-56`：

```cpp
ss << "for (" << axis0.AsArg() << " = 0; " << axis0 << " < " << axis0.actual_size << "; " << axis0 << "++) {";
...
ss << "DataCopyPadExtend(" << y << "[" << axis0 << " * " << tpipe.tiler.Size(y.vectorized_strides[0]) << "], "
   << x1 << "[param_offset], 1, " << axis1.actual_size << ", 0, 0);";
```

`y` 是 UB 目的地，偏移 `axis0 × vectorized_strides[0]` 非 0；而 `axis0` 这个循环变量是 **gather 自己造的**，
plan 里根本没有它。注意它**没有调 `Offset`**，是手写的同形式子。

**对设计的两个推论：**

1. `localOffset` 作为**预计算好的 `Value` 字段**是错的形状。预计算的前提是"偏移只依赖 plan 里已有的
   循环"，gather 证伪了这个前提——它的自变量在 plan 建好之后才诞生。legacy 的
   `Offset(current_axis, axis, strides)` 是个**函数**不是字段，这不是风格差异。
2. 原 §8 那条"按公式实现而不是硬编码 0"的建议**仍然成立，但理由要换**。不是"将来可能冒出非 0 形态"
   （屏障三挡着），而是：写成公式后这条不变量变得**可检查**——交集为空所以为 0，而不是靠一句注释断言；
   且二维 tile 前融合若选择"外层向量化轴发一层循环"的实现路线，需要的正是同一个原语（见 §9.2）。

---

# 9 实施方案

本节把 §6 的五个动作落到文件与函数。四步互相独立可分别提交，每步都带自己的验收判据。

## 9.1 步骤一：UB 偏移换成原语（对应动作 1，修 §3 缺陷）

### 9.1.1 接口改动

`dialect/AFIR/Analysis/TransferPlan.h`：

```diff
 // `DataCopyPadExtend` 的四个参数加两端的偏移。一维搬运时 `blockCount` 为 1、两个
 // stride 为 0，与二维走同一条发射路径。
 struct ScheduledTileIndices {
   Value globalOffset;
-  Value localOffset;
   Value blockCount;
   Value blockLen;
   Value srcStride;
   Value dstStride;
 };
+
+// UB 块内定位本轮要处理的那一小段。
+//
+// = Σ（轴变量 × 该轴的 UB stride），只累加**同时是当前循环轴**的向量化轴。与 legacy
+// `Tiler::Offset(current_axis, vectorized_axis, vectorized_strides)` 是同一个式子。
+//
+// 三点与旧 `localOffset` 字段不同，每一点都有出处：
+//  - **与容器无关**。TQue 的轮转由 AllocTensor/DeQue 吃掉，TBuf 只有一个 tile 大，
+//    两者基址都不含轮次信息，所以这一层不该按容器分叉（§2.2）。调用方不要再写
+//    `isQueue ? zero : offset`。
+//  - **自变量是向量化轴，不是 TileOuter 轴**。TileOuter 被三道结构性屏障挡在 UB 偏移
+//    之外（§3.5），`iv × tileSize` 引用的正是它，所以那是另一个量而不是本式的特例。
+//  - **是函数不是字段**。gather 一类算子自己发循环，自变量在 plan 建好之后才诞生
+//    （§8.1），预计算成字段装不下，所以留 `extraLoopAxes` 给算子补自己的循环变量。
+//
+// 当前所有调度形态下返回常量 0——因为 `plan.loopAxis ∉ {plan.tiles[i].axis}`（屏障三）。
+// 这不是硬编码，是交集为空的结果，形态一变式子自己会跟上。
+Value materializeLocalTileOffset(Operation *op, OpBuilder &builder, const TransferPlan &plan,
+                                 ArrayRef<std::pair<int64_t, Value>> extraLoopAxes = {});
```

`isQueueBuffer` 的声明连同那段注释一并删掉——它今天**唯一**的用途就是驱动第③层的容器分叉
（三个使用方全在 `Implementations/` 下，见 9.1.3）。留着会让"容器影响偏移"这个已被否定的模型
继续有承载物。

### 9.1.2 实现

`dialect/AFIR/Analysis/TransferPlan.cpp`，替换 `materializeScheduledTileIndices` 尾部那段
`if (plan.tiles.size() == 1)`：

```cpp
Value materializeLocalTileOffset(Operation *op, OpBuilder &builder, const TransferPlan &plan,
                                 ArrayRef<std::pair<int64_t, Value>> extraLoopAxes) {
  const Location location = op->getLoc();
  auto func = op->getParentOfType<func::FuncOp>();
  if (!func) {
    return builder.create<arith::ConstantIndexOp>(location, 0);
  }
  // 当前循环轴 = 所有**外层** scf.for 的轴，不只是 plan.loopAxis 那一层。legacy 的
  // `current_axis` 同样是整个栈（`Loop::GenerateLoop` 逐层 push），少收集就会在循环
  // 下探进 tile 时漏项。
  llvm::SmallDenseMap<int64_t, Value, 4> axisToInductionVar;
  for (Operation *parent = op->getParentOp(); parent; parent = parent->getParentOp()) {
    auto loop = dyn_cast<scf::ForOp>(parent);
    if (!loop) {
      continue;
    }
    auto binding = loop->getAttrOfType<afir::AxisBindingAttr>(afir::kAxisBindingAttrName);
    if (binding && binding.getRole() == afir::AxisValueRole::Loop) {
      axisToInductionVar.try_emplace(binding.getAxis(), loop.getInductionVar());
    }
  }
  // 算子自己发的循环后插入，允许覆盖同名轴。
  for (auto [axisId, inductionVar] : extraLoopAxes) {
    axisToInductionVar[axisId] = inductionVar;
  }

  const SymbolBindingMap symbols = collectTilingParamBindings(func);
  Value offset;
  for (const TransferTileAxis &tile : plan.tiles) {
    auto iter = axisToInductionVar.find(tile.axis);
    if (iter == axisToInductionVar.end() || isZeroStride(tile.ubStride)) {
      continue;
    }
    FailureOr<Value> stride = materializeStride(builder, location, tile.ubStride, symbols);
    if (failed(stride)) {
      continue;
    }
    Value term = isOneStride(tile.ubStride)
                     ? iter->second
                     : builder.create<arith::MulIOp>(location, iter->second, *stride).getResult();
    offset = offset ? builder.create<arith::AddIOp>(location, offset, term).getResult() : term;
  }
  return offset ? offset : builder.create<arith::ConstantIndexOp>(location, 0);
}
```

**与 legacy 的一处已知差异，故意不对齐。** legacy 的命中判据是 `IsFrom(ca, iter)`，即"循环轴
**派生自**向量化轴"也算命中；这里只做精确相等。差异只在"某根向量化轴又被继续切分、循环跑在它
外半段"时才显现，而屏障二（§3.5）说明当前调度产生不出这种轴。接新算子时若真出现，症状是偏移
少算而不是越界，且能被 §9.1.4 的断言抓到——所以不提前实现，但要把这条写进注释。

### 9.1.3 三个使用方

| 文件 | 今天 | 改成 |
|---|---|---|
| `Implementations/DataMovement.cpp:193-211` | `isQueueBuffer` + 两处 `*localIsQueue ? zero : indices->localOffset` | 一次 `materializeLocalTileOffset(op, rewriter, plan)`，两个分支共用 |
| `Implementations/BinaryElementwise.cpp:204-218` | 三次 `isQueueBuffer` + 三个 `? :` | 同上，三个操作数共用同一个偏移值 |
| `Implementations/Reduce.cpp:252-259` | `isQueueBuffer` 两次 + queue-backed 前置条件 | 见步骤三 |

改动面就这么大：`isQueueBuffer` 全仓共 6 个调用点，全在上表三个文件里；`ScheduledTileIndices`
与 `localOffset` 在 `test/` 下**零引用**。所以这一步不牵动任何现有 UT，验收完全落在 §9.1.4 的
生成物对比上。

三处的 `isQueueBuffer` 调用与随之而来的 `notifyMatchFailure`（"has no 'afir.buffer_desc' at its
definition"）一并删除。注意这会**减少**一条诊断：原先缺 `buffer_desc` 会在这里报错，改后不报。
这是对的——那条检查存在的理由是"漏判会静默走错分支"，分支没了理由也就没了；descriptor 缺失
仍然会在 `buildTransferPlan` 的 `hasCanonicalDescriptors` 处被挡住。

### 9.1.4 验收

1. **逐字节不变**：`load_add_store`、`load_sum_store` 两个已注册用例的
   `06_convert-afir-to-ascendc.mlir` 与最终 kernel 与改动前 `diff` 为空。这是最强的判据——
   偏移在这些形态下本来就求值为 0，只是来路从 `? :` 换成了空交集。
2. **`load_add_add_store` 转绿**：`[8192, 128]` 数值全对（改前 599094/1048576 错）。
3. 生成的 kernel 里不再出现无使用者的 `iv × tileSize`（§2.3 里那个 `%96`）。
4. 建议顺带加一条 lit 断言：`[1024, 128]` 的小形状（只跑一轮）与 `[8192, 128]`（跑多轮）生成的
   UB 取址表达式**相同**。这条能防止"偏移又被按轮次引回来"的回归，比数值判据更早失败。

## 9.2 步骤二：elementwise 的长度换成线性跨度（对应动作 2，解前融合）

### 9.2.1 公式

legacy `Tiler::TensorActualSize`（`codegen_kernel.cpp:869-904`）：

```
span = Σ_{ubStride_i ≠ 0} (actualSize_i − 1) × ubStride_i + 1
```

`actualSize` 用的是本轮实际长度（TileInner 轴取 `actual_size`），所以尾 tile 自动收窄。

### 9.2.2 改动

`Implementations/BinaryElementwise.cpp`：

1. 删掉 `if (plan->tiles.size() != 1) return notifyMatchFailure(...)`（第 197-199 行）。
2. `count` 从 `indices->blockLen` 换成线性跨度。跨度需要每根轴的 `actualSize` 与 `ubStride`，
   `ScheduledTileIndices` 今天只外抛最内轴的 `blockLen`，所以要么给 `ScheduledTileIndices`
   加一个 `Value linearSpan`，要么外抛 `SmallVector<TileRuntime>`。**建议前者**——后者会把
   `TileRuntime` 从实现细节升格成接口，而 §5 的教训正是不要把内部结构外抛成公共形状。

第 195-196 行那条注释要改掉，它说反了：

> ~~二维 tile 的计算算子要按行展开（外层 stride 不等于内层长度时，一次 `count` 覆盖不到整块）~~

用跨度公式恰好**覆盖得到**整块，代价是顺带算了行尾 padding。padding 多算对 elementwise 无害
（算出来的值不会被读），这正是 legacy 敢用一条 API 平掉二维的原因。

### 9.2.3 验收

`test/e2e/cpu/cases/load_add_sum_store/` 注册进 `test/e2e/cpu/CMakeLists.txt` 并转绿。
这一步同时解掉 §8 那条"二维非队列形态的落点只做了推导，没有跑通数值"。

**风险点**：padding 位置参与计算时若产生 `inf`/`nan`，某些 dtype 下会污染后续。本期 elementwise
只有 Add，无害；接 Div/Log 一类前需要复核。这条建议写进代码注释而不是只留在文档里。

## 9.3 步骤三：去掉 reduce 的 queue-backed 前置条件（对应动作 3，解后融合）

`Implementations/Reduce.cpp:249-259` 整段删除。那段注释的理由是——

> 非队列的输入要按轮次加偏移，默认成 0 会读错 tile 而不是报错

——在步骤一之后不再成立：**不存在"按轮次加偏移"这件事**。`source` 与 `destination` 改为统一经
`materializeLocalTileOffset`（今天求值为 0），`apiScratch` 保持不加偏移（它是整块工作区，
不按轮次切分，这条理由与容器无关，仍然成立）。

### 9.3.1 验收

`test/e2e/cpu/cases/load_sum_add_store/` 注册并转绿。

注意 §4 已确认这一侧的 Add 本来就是通的（归约维在 destination 两侧 stride 都是 0，
`buildScheduledPlan` 会跳过它，只剩一根 tile 轴），所以这步只需要动 Sum 一侧。

## 9.4 步骤四：能力查询与改写对齐（对应动作 4）

`BinaryElementwiseL2CapabilityProvider::queryLowering`（`BinaryElementwise.cpp:85-125`）已经调了
`buildTransferPlan`，但从不检查 `plan->tiles.size()`，而改写侧要求 `== 1`。于是
SelectImplementation 判"支持"、改写时才失败，报成 `failed to legalize operation 'afir.add'`
而不是 `UnsupportedShape`——§4 表格里前融合那一行的现象就是这么来的。

步骤二做完后二维可用，`buildTransferPlan` 自己会拒掉三维及以上（`tiles.size() > 2` 时
`failure()`），所以这步**不需要**新增 size 检查，只需确认 `queryLowering` 把
`buildTransferPlan` 的失败映射成 `UnsupportedShape` 而不是当前的 `MissingInformation`
（第 94-96 行）。两者的区别：前者是"这个实现做不到"，后者是"信息不全、换个实现可能行"，
对 SelectImplementation 的诊断质量影响不同。

Reduce 那边已经检查了自己的 plan（`buildReducePlan` 失败 → 不支持），可作对照。

## 9.5 提交与顺序

```
1) fix(mlir): UB tile 偏移改为与容器无关的统一原语        ← 9.1，独立，逐字节不变
2) feat(mlir): elementwise 用线性跨度支持二维 tile        ← 9.2，依赖 1
3) feat(mlir): reduce 允许非队列的输入与 destination      ← 9.3，依赖 1，与 2 无依赖
4) fix(mlir): elementwise 能力查询与改写对齐诊断          ← 9.4，依赖 2
```

2 与 3 可并行。四步都做完后，§7.2 表格里前三行应全部转绿；后两行（`load_add_add_sum_store`、
`load_add_chain_store`）卡的是 bridge 的 `InvalidBinding`，与本方案无关，仍由他人处理。

## 9.6 本方案不做的事

- **不新增 `TransposePlan` / `ConcatPlan` / `GatherPlan`**（动作 5）。那是接新算子时的约束，
  本期没有接入需求，提前实现会在没有使用者的情况下固化一次猜测——这正是 `ScheduledTileIndices`
  今天的处境。
- **不实现 legacy `IsFrom` 的派生轴命中**，理由见 9.1.2。
- **不碰 RA 形态、多归约轴、plan 选择**，见 §8 其余各条。
