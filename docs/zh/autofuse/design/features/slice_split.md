# Slice/Split/StridedSlice/SplitV(D) 自动融合算子 — 综合技术文档

简体中文 | [English](../../../../en/autofuse/design/features/slice_split.md)


---

## 1. 特性背景

### 1.1 项目定位

在自动融合框架中支持 **8 种搬运类算子**的自动融合：

| 算子 | 输入 | 输出 | 核心语义 |
|------|------|------|---------|
| **Slice** | x, offsets, size | y | `output[i] = input[begin+i]`，按 begin/size 取连续子张量 |
| **SliceD** | x | y | 同 Slice，但 offsets/size 作为属性（非输入） |
| **SliceDV2** | x, offsets | y | 同 Slice，但 size 作为属性 |
| **StridedSlice** | x, begin, end, strides | y | Slice 超集，支持步进 + mask（begin_mask/end_mask/ellipsis_mask/new_axis_mask/shrink_axis_mask） |
| **StridedSliceV2/V3** | x, begin, end, [axes], [strides] | y | StridedSlice 变体 |
| **Split** | split_dim, x | y（动态） | 在指定轴**等分**为 N 份，`output_k[i,j] = input[k*S+i, j]` |
| **SplitV** | x, size_splits, split_dim | y（动态） | 在指定轴**非等分**切分（每份 size 由 size_splits 指定） |
| **SplitD** | x | y（动态） | 同 Split，但 split_dim/num_split 作为属性 |

> GE 中的算子注册定义见 `ge-master/tests/framework/ge_running_env/include/ge_running_env/op_reg.h:304-452`。

### 1.2 核心思想

Slice/Split/StridedSlice/SplitV(D) 均属 **View 类算子**——lowering 阶段不涉及计算，只做 shape/数量变化。

**关键洞察**：通过带 offset 和 stride 的 load/store 实现这些算子，将"物理切分"变为"逻辑映射"。后续算子（Add/Concat 等）直接从原大张量指定位置（offset+stride）读数，消除无意义内存搬运，使 Slice 变成"读取方式"而非"计算任务"，从而与后续算子合并为单一任务。

> 类比：给书加一个书签（Offset），而不是把那一页撕下来复印一份。

### 1.3 功能目标

主要实现自动融合框架的三大核心能力：
1. **融合策略求解**（FusionStrategySolver）——基于内存节省和邻近性打分
2. **融合决策判断**（FusionDecider）——垂直/水平融合规则
3. **节点融合处理**（NpuFusionDecider::Fuse）——子图合并与循环合并

---

## 2. 特殊背景及限制

### 2.1 Split/SplitV 场景

SplitV 在 GE 图上可能呈多级结构，需要先将多级合并成一级，再进行 lowering。

---

### 2.2 StridedSlice 约束

| 约束 | 说明 |
|------|------|
| **stride 为负值不 lowering** | 负 stride 涉及数据倒排，暂不支持 |
| **升维/降维不 lowering** | StridedSlice 的 new_axis_mask/shrink_axis_mask 改变维度的场景不 lowering |
| **输出尾轴为 1 不 lowering** | 输出数据尾轴为 1 的场景不 lowering |

### 2.3 通用融合限制

- Slice/Split/StridedSlice/SplitV(D) 这 8 种算子**限制向前融合**，但前节点也是 Slice/Split/StridedSlice/SplitV(D) 时支持向前融合。
- Split 类在 lowering 的时候**不进行向后融合**，直接 realize。
- Split 不与 reduce 融合。
- Split 不做水平融合。
- Split/SplitV 后接多 Concat 场景下，前端限制输出个数超过 48 不进行 lowering。

### 2.4 平台约束

- 多版本（Tuscany/Florence/Milan/Helper）并行，需保证 3~8 包在多平台完整交付。
- GE 是 onetrack 部件，不允许区分芯片。
- 加载占用内存需考虑小海思和 MDC 等内存敏感形态，新增特性原则上不应导致内存上涨。

### 2.5 A3 与 A5 行为差异（BackendSpec 控制）

代码库中通过 `BackendSpec::SliceSplitSpec`（`autofuse/inc/backend/backend_spec.h:30-35`）区分 A2/A3 与 A5 的行为：

```cpp
struct SliceSplitSpec {
  bool split_lowered_to_split;    // false: A2/A3 将 Split Lowering 成多个 StridedSlice
                                  // true:  A5  将 Split Lowering 成多个 Split
  bool slice_fuse_with_end_dim_1; // false: A2/A3 Slice 尾轴为1不融合
                                  // true:  A5  Slice 尾轴为1可以融合
  bool enable_split_flatten;      // false: A2/A3 暂不使能 flatten
                                  // true:  A5  使能 flatten
};
```

此外，`BackendSpec::max_input_nums_after_fuse`（`backend_spec.h:49`）限制融合后单节点最大输入数：**A2A3=8，A5=14**。

---

## 3. 对外接口

### 3.1 融合决策接口（FusionDecider）

定义于 `autofuse/inc/fusion/fusion_decider.h`：

```cpp
class FusionDecider {
  // 检查两个节点是否可以垂直融合
  virtual bool CanFuseVertical(const NodePtr &node1, const NodePtr &node2) = 0;

  // 检查两个节点是否可以水平融合
  virtual bool CanFuseHorizontal(const NodePtr &node1, const NodePtr &node2) = 0;

  // 获取融合对的优先级
  virtual uint32_t GetFusionPairPriority(const NodePtr &node1, const NodePtr &node2) = 0;

  // 融合两个节点
  virtual NodePtr Fuse(const NodePtr &node1, const NodePtr &node2);
};
```

**融合优先级枚举**（`FusionPriority`）：

| 优先级 | 含义 | 适用场景 |
|--------|------|---------|
| HIGHEST | 最高 | Split 的 AscBackend 融合（保证 N 个 Split backend 融合在一起） |
| HIGHER | 较高 | — |
| HIGH | 高 | — |
| DEFAULT | 默认 | 一般算子 |
| LOW | 低 | — |

> Priority 值越小优先级越高。`GetPossibleFusionsWithHighestPriority` 只返回最高优先级对应的节点对。

### 3.2 子图合并接口

```cpp
// 检查两个子图是否满足合并条件（输入输出数量匹配 + 调度轴相同）
Status CanMergeAscGraph(const ComputeGraphPtr &subgraph1, const ComputeGraphPtr &subgraph2,
                        const NodePtr &node1, const NodePtr &node2);

// 获取两个节点的输入输出信息，记录相同输入和节点间链接关系
Status GetFuseNodeInfo(const NodePtr &node1, const NodePtr &node2);

// 将两个节点融合为一个新节点，更新数据边和控制边
NodePtr FuseNode(NodePtr node1, NodePtr node2, const ComputeGraphPtr merged_graph);

// 通过循环分析合并两个子图
virtual ComputeGraphPtr MergeAscGraphByLoop(const ComputeGraphPtr &subgraph1,
                                             const ComputeGraphPtr &subgraph2,
                                             const NodePtr &node1, const NodePtr &node2);

// 将两个 AscBcNode 子图合并为融合子图
ComputeGraphPtr MergeGraphToFusedAscBcNode(const ComputeGraphPtr &subgraph1,
                                            const ComputeGraphPtr &subgraph2,
                                            const NodePtr &node1, const NodePtr &node2,
                                            const NodePtr &fused_node);

// 创建新的 AscBcNode 子图
ComputeGraphPtr CreateAscBcNodeSubGraph(const NodePtr &node1, uint32_t in_nums,
                                         uint32_t out_nums,
                                         const std::vector<uint32_t> &node_output_index);

// 合并两个子图（subgraph2 merge 到 subgraph1）
Status MergeSubGraph(const ComputeGraphPtr &subgraph1, const ComputeGraphPtr &subgraph2);

// 子图 netoutput 与输入节点连接
Status LinkSubGraphNode(const NodePtr &subgraph_netoutput,
                        const ComputeGraph::Vistor<NodePtr> &inputs,
                        const std::vector<std::pair<int32_t, int32_t>> &subgraph_link_map,
                        std::vector<NodePtr> &del_data_nodes);
```

---

## 4. 整体架构

### 4.1 编译流水线

```
GE 图 (Slice/Split/StridedSlice/SplitV)
  │
  ▼
┌──────────────────┐
│  算子符号化分析    │  输出 shape 符号化，已实现
└──────┬───────────┘
       ▼
┌──────────────────┐
│  算子 Lowering     │  View 类算子 → load/store(带 offset+stride)
│                   │  Slice/StridedSlice: store → StoreStridedSliceOp → load
│                   │  Split: store → StoreSplitOp → load (N个输出→N个AscBackend)
└──────┬───────────┘
       ▼
┌──────────────────┐
│  CanFuse 融合决策  │  FusionStrategySolver + FusionDecider
│                   │  打分排序 → 成环检测 → 节点融合
└──────┬───────────┘
       ▼
┌──────────────────┐
│  Scheduler 调度    │  SplitFusionCaseGenerator: 三场景模板选择
│                   │  场景1: 首轴拆分 → 转多 load
│                   │  场景2: 非首轴+全小包 → UB内切分模板
│                   │  场景3: 非首轴+非全小包 → 分组转load
└──────┬───────────┘
       ▼
┌──────────────────┐
│  Codegen 代码生成  │  SplitRegApiCall: UB切分SIMD代码生成
│                   │  AllAligned → SplitAllAligned (DataCopy)
│                   │  未对齐 → GenerateDefault (临时buffer+scatter)
└──────────────────┘
```

### 4.2 关键数据结构

#### 4.2.1 节点对（NodePair）

保存可融合的节点对，缓存融合后节省的内存大小和邻近性信息：

```cpp
class NodePair {
 public:
  FusingNodePtr first;
  FusingNodePtr second;
  Expression memory_score;    // 融合后节省的内存
  int64_t proximity_score;    // 邻近性评分
};
```

#### 4.2.2 读写内存信息（MemoryBuffer）

存储 Node 上输出 Anchor 和符号化的内存大小，使用指针减少临时对象：

```cpp
class MemoryBuffer {
 public:
  const Anchor *buffer;
  const Expression *size;
};
```

#### 4.2.3 融合过程临时节点（FusingNode）

融合处理中间过程的临时 node，存储读写内存大小、原始节点数、最大最小排序等信息：

```cpp
class FusingNode {
 private:
  std::vector<FusingNodePtr> nodes_;
  std::set<MemoryBuffer> read_writes_;
  std::set<NodePtr> ancestors_;
  int64_t min_order_;
  int64_t max_order_;
  int64_t id_;
  const NodePtr node_;
  size_t fusion_nodes_size_;
};
```

#### 4.2.4 Split IR 定义

```cpp
REG_ASC_IR(Split)
    .Input("x", "T")
    .DynamicOutput("y", "T")
    .Attr<int64_t>("index")
    .Attr<int64_t>("gid")   // global_id, SplitOp 的全局编号
    .ComputeType(ge::ComputeType::kComputeSplit);
```

- `index`：输出索引
- `gid`：Split Op 的全局编号，用于 post_process 中合并同一 gid 的 Split
- `ComputeType::kComputeSplit`：标识 Split 计算类型

---

## 5. 核心实现

### 5.1 Lowering 阶段

#### 5.1.1 设计原理

Slice/Split/StridedSlice 属 View 类算子，lowering 不涉及计算，只做 shape/数量变化。需根据符号化的输出 shape 反向推导每个输出元素在原输入的坐标，然后 Store。

**关键结论**：StridedSlice 是功能全集，Slice 是其子集（stride=1）。

#### 5.1.2 A3 实现

| 步骤 | 实现 |
|------|------|
| Lowering 入口 | `lowering_impl.cpp` 新增 `LowerSlice`/`LowerSplit`/`LowerStridedSlice`，在 `LoweringManager` 注册 |
| Loop API | `loop_api.cpp` 新增 `loop::StoreStridedSlice`/`loop::StoreSplit` |
| Loop Ops | `loop_ops.h` 定义 `StoreStridedSliceOp`/`StoreSplitOp` |
| 坐标推导 | `StoreStridedSliceOp` 重写 `reindex`：`i_k = start[k] + o_k × stride[k]` |

**reindex 公式**：对于输出张量中元素索引 `(o_0, o_1, ..., o_n)`，对应输入张量坐标 `(i_0, i_1, ..., i_n)`：

```
i_k = start[k] + o_k × stride[k]
```

这使得后续算子直接从原张量读，避免内存搬运。

**E2E 测试验证**（`autofuse/tests/st/codegen/e2e/e2e_load_strided_slice_store.cpp`）：

该测试验证了 StridedSlice 的完整 lowering 流程，确认了 `StridedSlice → Load(offset, stride) → Store` 的实现方式：

| 阶段 | 节点状态 |
|------|---------|
| **BeforeAutofuse** | `x0`（Data，axis `[z0, z1=s1+s2]`）→ `load0`（Load，`SetOffset(s1)` 跳过 s1 元素，strides `{s1+s2, One}` 实现步进）→ `store`(Store) → `y`(Output) |
| **AfterInferOutput** | compute_type 赋值：Load→`kComputeLoad`，Store→`kComputeStore` |
| **AfterGetApiInfo** | API 类型：Load/Store→`kAPITypeCompute`，计算单元→`kUnitMTE2`（内存搬运引擎） |
| **AfterScheduler** | 轴切分（`TileSplit`/`BlockSplit`/`ApplySplit`），向量化轴/步长设置，对齐到 8 元素（32B/sizeof(float)） |
| **AfterQueBufAlloc** | 内存分配：x0→GM，load0 输出→UB（Queue, depth=2, buf_num=2），store 输出→GM |

> `load0.ir_attr.SetOffset(s1)` 设置偏移（StridedSlice 的 begin），strides `{s1+s2, One}` 实现步进模式——每行跳过 `s1` 个元素，精确对应 reindex 公式。

#### 5.1.3 A5 实现

**Slice/StridedSlice**：同 A3，未修改。

**Split/SplitV**：A3 中 lowering 成多个 load/store，但未融合的 load/store codegen 性能劣化严重。A5 需支持 **UB 内切分模板**。

| 变更 | 说明 |
|------|------|
| `LowerSplit` → `StoreSplit` | lowering 成 `store → StoreSplitOp → load` |
| `StoreSplitOp` | 实现 `compute`/`realizeImpl`/`InferType` |
| `lower_split_helper.cpp` | 处理 Split 对齐场景，决定是否需要 lifting |
| `FuseType = KsplitType` | 新增融合类型 |
| N 个输出 → N 个 AscBackend | lowering 后每个输出一个 AscBackend |

**Split lowering 不向前/向后融合，直接 realize。**

### 5.2 CanFuse 融合决策

#### 5.2.1 融合主流程

```cpp
Status FusionStrategySolver::Fuse(const ComputeGraphPtr &graph) const {
  std::vector<FusingNodePtr> nodes;
  GE_ASSERT_SUCCESS(GetNodes(graph, nodes));  // topo 排序，创建 FusingNode
  const auto cycle_detector = GraphUtils::CreateSharedCycleDetector(graph);
  GE_ASSERT_SUCCESS(ComputeAncestors(nodes));
  for (uint32_t i = 0U; i < config.max_fuse_rounds; ++i) {
    const size_t old_nodes_size = nodes.size();
    GE_ASSERT_SUCCESS(FuseNodesOnce(graph, cycle_detector, nodes));
    const size_t new_nodes_size = nodes.size();
    if ((new_nodes_size == old_nodes_size) || (new_nodes_size == 1U)) break;
  }
  return SUCCESS;
}
```

#### 5.2.2 融合条件判断

`FusionStrategySolver::CanFuse` 检查三个条件：

```cpp
bool CanFuse(const ComputeGraphPtr &graph, const FusingNodePtr &node1,
             const FusingNodePtr &node2) const {
  // 1. 融合后节点数不超过 max_fusion_size
  if ((node1->GetFusionNodesSize() + node2->GetFusionNodesSize()) > config.max_fusion_size)
    return false;

  // 2. 融合后节省的内存读写大小不为 0
  if (ScoreFusion::ScoreFusionMemory(*node1, *node2) == kSymbolZero)
    return false;

  // 3. 纵向/横向融合规则
  if (node2->IsAncestor(node1)) {
    if (!GetBackEnd(graph)->CanFuseVertical(node1->GetOrgNode(), node2->GetOrgNode()))
      return false;
  } else {
    // 横向融合导致内存峰值增加的不融合
    if (CanFusionIncreasePeakMemory(node1, node2)) return false;
    if (!GetBackEnd(graph)->CanFuseHorizontal(node1->GetOrgNode(), node2->GetOrgNode()))
      return false;
  }
  return true;
}
```

#### 5.2.3 融合打分算法

计算两个节点融合后可节省的内存（读写相同内存即为可减少的搬运内存）：

```cpp
Expression ScoreFusion::ScoreFusionMemory(const FusingNode &node1, const FusingNode &node2) {
  Expression reduced_memory = kSymbolZero;
  for (const auto &memory_buffer : node2.GetReadWrites()) {
    auto it = node1_memory_buffers.find(memory_buffer);
    if (it != node1_memory_buffers.end()) {
      reduced_memory = reduced_memory + *memory_buffer.size;
    }
  }
  return reduced_memory;
}
```

**排序比较**：内存大小优先，相同则看邻近性，再相同看 min_order。

#### 5.2.4 融合节点对获取

1. 按公共读写内存分组（`buffer_grouping`）
2. 每组内两两组对，去重（`repeat_check`）
3. `CanFuse` 判断
4. 按优先级分组，只取最高优先级
5. 按融合打分排序

#### 5.2.5 Slice 类融合规则

定义在 `slice_split_fusion_strategy.cpp`（A3）/对应实现：

| 规则 | 说明 |
|------|------|
| (a) 向前融合 | 前节点必须是 Slice 类，否则不融合 |
| (b) 水平融合 | Slice 不做水平融合；但两 Slice 融合节点有垂直关系（slice 后接 concat）需允许融合 |
| (c) broadcast | Slice 融合节点含 broadcast 不与其它节点融合 |
| (d) reduce | 向后融合时后续节点是 reduce 则不融合 |

#### 5.2.6 Split 类融合规则（A5 新增）

定义在 `split_fusion_strategy.cpp`：

| 规则 | 说明 |
|------|------|
| (1) 优先级最高 | 保证 N 个 Split AscBackend 在 canfuse 阶段融合在一起 |
| (2) 不与 reduce 融合 | — |
| (3) 不向前融合 | — |
| (4) 不做水平融合 | — |

canfuse 后 N 个 AscBackend 融合成一个 Fuse 节点。

#### 5.2.7 Post-process 合并

`adaption_combine_split.h`：将同一 `gid` 的 Split 合并成一个 Split Op。

#### 5.2.8 节点融合实现

`NpuFusionDecider::Fuse` 执行流程：

1. 获取两个节点的子图属性
2. `GetFuseNodeInfo` 获取输入输出链接关系
3. `CreateOrUpdateSubgraphOutputAttr` 更新子图输出属性
4. 白名单检查 → `UnifySubgraphAxis` 统一调度轴
5. 若两个都是 AscBc 类型 → 尝试循环合并（`CanMergeAscGraph` + `MergeAscGraphByLoop`）
6. `FuseNode` 创建新节点
7. 若未循环合并 → `MergeGraphToFusedAscBcNode` 创建 AscBc 子图
8. 更新子图属性和轴属性

### 5.3 Scheduler 调度

核心代码：`autofuse/optimize/task_generator/split_schedule_case_generator.{h,cpp}` — `SplitFusionCaseGenerator` 类。

#### 5.3.1 三种调度场景

| 场景 | 条件 | 策略 | 优势 | 劣势 |
|------|------|------|------|------|
| **场景1：首轴拆分** | `split_dim == 0` 或 `split_dim > 0` 且之前各轴取值均为 1（X Group 为空或内积为 1） | 转为多个 load（连续大块内存搬运） | 实现简单；有调度收益；有 store-load 对消收益 | UB→GM 搬运基本没减少 |
| **场景2：非首轴+全小包** | 非首轴切分，各分块都是小包，UB 可全载 | UB 内切分模板：先 load 完整数据，再 UB 内 split | 极致优化，跨 UB 搬运最少，性能最优 | 要求全载 + 512B CacheLine 对齐；UB 排布复杂 |
| **场景3：非首轴+非全小包** | 非首轴切分，分块非全小包或 UB 不能全载 | 对各输出分组，像场景1那样转 load，各分支独立处理 | 实现简单；有 store/load 对消收益 | 性能大概率不如单算子，仅补齐架构约束 |

> 例：`[1,1,3,4]` split_dim=2 等价于 `[3,4]` split_dim=0，属于首轴拆分场景。

#### 5.3.2 SplitFusionCaseGenerator 关键方法

| 方法 | 功能 |
|------|------|
| `FindSplitNodes` | 遍历图找到所有 Split 节点 |
| `ResolveSplitDim` | 解析 Split 的切分维度 |
| `ConvertSplitToLoads` | 将 Split 转换为多个 load（场景1/3） |
| `SplitSplits` | UB 内切分（场景2） |
| `Prepare` | 预处理 |


#### 5.3.3 Split 分组

| 组件 | 功能 |
|------|------|
| `SplitGroupPartitioner`（`split_group_partitioner.h`） | Split 输出分组，含 `SplitGroup` 结构体 |
| `SplitScoreFunctionGenerator`（`split_score_function_generator.h`） | Split 场景的打分函数生成 |

> Split 后接多 Concat 时，Concat 分组从输入考虑，Split 分组从输出考虑。目前前端限制输出个数超过 48 不进行 lowering。

#### 5.3.4 A5 Slice 调度新增

| 能力 | 状态 |
|------|------|
| Slice 转 NDDMA | 当前已支持（之前有功能问题已回退） |
| Slice UB 内切分 | 当时因 tiling case 太多未开发，当前大部分性能问题可通过 NDDMA 解决 |

### 5.4 Codegen 代码生成

#### 5.4.1 A3 Codegen

Slice/split 转 load，slice 需把 offset 加入 load 代码，其余复用 load。

#### 5.4.2 A5 Split UB 切分 Codegen

核心代码：`autofuse/v35/codegen/reg_api_call/split_reg_api_call.{h,cpp}` — `SplitRegApiCall` 类。

**生成流程**：

```cpp
Status SplitRegApiCall::Generate(...) {
  // 1. 解析 split_dim
  size_t split_dim;
  ParseSplitDim(x, y0, split_dim);

  // 2. 初始化 tiling 信息
  SplitTiling split_tiling;
  InitializeTiling(split_dim, outputs, x, split_tiling);

  // 3. 确保无 padding
  GE_ASSERT_TRUE(split_tiling.src_col_actual_size_expr.Simplify() ==
                 split_tiling.src_col_size_expr.Simplify(),
                 "Padding is not supported by split yet");

  // 4. 分支：对齐 vs 未对齐
  if (IsAllAligned(split_tiling)) {
    GenerateForAllAligned(outputs, x, split_tiling, tpipe.tiler, ss);
  } else {
    GenerateDefault(outputs, x, split_tiling, tpipe, ss, id);  // 临时 buffer
  }
}
```

**SplitTiling 结构体**（`split_reg_api_call.h`）：

| 字段 | 含义 |
|------|------|
| `src_col_size_expr` | 源列大小表达式 |
| `src_col_actual_size_expr` | 源列实际大小表达式 |
| `dst_col_sizes` | 各输出列大小数组 |
| `src_offsets` | 各输出在源中的偏移数组 |

**对齐检查**（`IsAllAligned`）：

```cpp
constexpr uint32_t kDataBlockSize = 32U;  // SIMD 数据块宽度 32B
// align_size = 32 / sizeof(T)
//   fp16  → 16 元素
//   uint8 → 32 元素
```

检查所有输出的列大小和间隔是否对齐到 `kDataBlockSize`（32 字节）。

**AllAligned SIMD 模板**（`SplitTilingAllAligned<N>`）：

```cpp
template <typename T, uint32_t OUTPUT_NUM>
inline __aicore__ void SplitAllAligned(uint32_t num_rows,
                                        const SplitTilingAllAligned<OUTPUT_NUM> &tiling,
                                        LocalTensor<T> &src_tensor,
                                        LocalTensor<T> (&dst_tensors)[OUTPUT_NUM]) {
  constexpr uint32_t kDataBlockSize = 32U;
  constexpr auto align_size = static_cast<uint16_t>(kDataBlockSize / sizeof(T));
#pragma unroll
  for (uint32_t i = 0U; i < OUTPUT_NUM; ++i) {
    const auto size = tiling.dst_col_sizes[i];
    DataCopyParams copy_params{
        static_cast<uint16_t>(num_rows),           // 行数
        static_cast<uint16_t>(size / align_size),  // 每行拷贝块数
        static_cast<uint16_t>((tiling.src_col_size - size) / align_size),  // 行间跳过
        0};
    DataCopy(dst_tensors[i], src_tensor[tiling.src_offsets[i]], copy_params);
  }
}
```

- 每个输出用 `DataCopy` 按 `copy_params` 搬运
- `src_offsets[i]` 定位各输出在源张量中的起始位置
- 行间通过 `src_col_size - size` 跳过非目标数据

**未对齐场景 — `SplitExtend` 路径**：

`GenerateDefault`（`split_reg_api_call.cpp:204`）在 `IsAllAligned` 返回 false 时调用。核心流程：

1. **数据类型提升优化**：对非对齐场景做位宽转换以减少搬运开销
   - `uint64` → `uint32_t`：`kB64ToB32 = 2`，列大小 ×2，用 `uint32_t` 视图搬运
   - `uint8` → `uint16_t`：`NeedB8ToB16()` 检查所有输出列大小是否偶数字节对齐，若满足则列大小 ÷2，用 `uint16_t` 视图搬运
   - 其他类型：直接使用原始 dtype
2. **生成 `SplitTiling` 结构体**：`DefineSplitTiling` 输出 `.num_rows`、`.num_src_cols`、`.num_dsts_cols`、`.src_offsets` 等字段
3. **生成 `SplitExtend` 调用代码**：

```cpp
ss << "split::SplitExtend<" << dtype_name << ", " << outputs.size() << ">("
   << "(" << dtype_name << " *)" << x.GetPhyAddr()
   << ", split_dst_addrs, " << tmp_buf << "_" << tmp_buf_id << ", split_tiling);";
```

4. **`SplitExtend` 内核实现**（`autofuse/v35/ascendc/api_regbase/split.h:148`）：

```cpp
template <typename T, size_t OUTPUT_NUM>
__aicore__ inline void SplitExtend(T *src_addr, T *out_addrs[OUTPUT_NUM],
                                    LocalTensor<uint8_t> &tmp_buf,
                                    const SplitTiling<OUTPUT_NUM> &tiling) {
  if constexpr (sizeof(T) == sizeof(uint32_t)) {
    SplitExtendInner<uint32_t, uint32_t, int32_t, OUTPUT_NUM>(...);
  } else {
    SplitExtendInner<uint16_t, uint16_t, int16_t, OUTPUT_NUM>(...);
  }
}
```

5. **`SplitExtendInner` 策略选择**（`split.h:130`）：对每个输出按列大小选择搬运方式
   - `num_dsts_cols[i] > kGatherMaxLen`（`VECTOR_REG_WIDTH / sizeof(T)`）→ 使用 `SplitCopy`（逐行 DataCopy）
   - 否则 → 使用 `DataCopyGatherVf`（gather 向量化搬运，利用 tmp_buf 作为中间寄存器）
   - `src_col_offset` 逐输出累加，定位下一个输出在源中的起始偏移

#### 5.4.3 Split 打分函数

`autofuse/optimize/task_generator/split_score_function_generator.{h,cpp}` — `SplitScoreFunctionGenerator` 类。

为 split 调度生成运行时打分函数 `CalcScore(tiling_data)`，返回值：
- `1` = 对齐，选择 UB split 模板
- `-1` = 不对齐，选择 split-to-load 模板
- `0` = 编译期无法确定

**生成流程**（`Generate`）：

```
1. ParseStride()  — 计算split_dim之后的stride（dtype_size × 后续维度大小的乘积）
2. if (const_part_stride_ % 32 == 0) → return 1   // 常量部分已对齐
3. TryGetScoreByConstExpr(score)  — 编译期尝试求分
4. if (score != 0) → return score
5. GenerateForUnaligned()  — 生成运行时判断代码
```

**`TryGetScoreByConstExpr`**（编译期打分）：
- 遍历所有输出，检查 `dim × const_part_stride` 是否满足 32B 对齐
- 对齐率阈值：`kMaxUnalignedRate = 0.1`（允许最多 10% 不对齐）
  - `align_threshold = ceil(split_dim_size × (1 - 0.1))`
- 若对齐输出数 ≥ `align_threshold` → score = 1
- 若 stride 全为常量但不满足对齐率 → score = -1
- 若 stride 含非常量 → score = 0（需运行时判断）

**`GenerateForUnaligned`**（运行时打分代码生成）：
- 将图中的 size_var 替换为 `tiling_data` 字段引用
- 生成运行时 `CalcScore` 函数：
  ```cpp
  int32_t CalcScore(const AutofuseTilingData &tiling_data) {
    const auto &t = tiling_data.graph0_result0_g0_tiling_data;
    auto stride = static_cast<int64_t>(<stride_expr>);
    if (stride % 32 == 0) { return 1; }
    std::vector<int64_t> split_dims;
    split_dims.reserve(N);
    // ... 填充各输出的 split_dim 大小
    size_t num_unaligned = 0U;
    size_t max_unaligned = ceil(split_dim_size × 0.1);
    bool already_unaligned = false;
    for (int32_t i = 0; i < N; ++i) {
      if (already_unaligned || (stride * split_dims[i] % 32) != 0) {
        num_unaligned += split_dims[i];
        already_unaligned = true;
        if (num_unaligned >= max_unaligned) { return -1; }
      }
    }
    return 1;
  }
  ```
- 逻辑：逐输出检查对齐性，一旦某个输出不对齐则后续全部视为不对齐；累计不对齐元素数超过 `max_unaligned` 则返回 -1

**与调度的关系**（`split_schedule_case_generator.cpp:73`）：
- `score_functions.resize(2U)`：对应两个模板（`ub_split` + `split_2_load`）
- 默认不生成打分函数（空字符串），即编译期直接决定模板选择

#### 5.4.4 Split 注册函数

`autofuse/ascir/reg_func/split.cpp`：定义 split 的 tiling 常量和对齐辅助函数。
`autofuse/v35/ascendc/api_regbase/split.h`：split API 注册。

---

## 6. 算子注册定义（GE）

GE 中相关算子定义（`ge-master/tests/framework/ge_running_env/include/ge_running_env/op_reg.h`）：

```cpp
REG_OP(Slice)
    .INPUT(x, TensorType::BasicType())
    .INPUT(offsets, TensorType::IndexNumberType())
    .INPUT(size, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(Slice)

REG_OP(SliceD)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(offsets, ListInt)
    .REQUIRED_ATTR(size, ListInt)
    .OP_END_FACTORY_REG(SliceD)

REG_OP(StridedSlice)
    .INPUT(x, TensorType::BasicType())
    .INPUT(begin, TensorType::IndexNumberType())
    .INPUT(end, TensorType::IndexNumberType())
    .INPUT(strides, TensorType::IndexNumberType())
    .ATTR(begin_mask, Int, 0)
    .ATTR(end_mask, Int, 0)
    .ATTR(ellipsis_mask, Int, 0)
    .ATTR(new_axis_mask, Int, 0)
    .ATTR(shrink_axis_mask, Int, 0)
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(StridedSlice)

REG_OP(Split)
    .INPUT(split_dim, TensorType({DT_INT32}))
    .INPUT(x, TensorType::BasicType())
    .DYNAMIC_OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(Split)

REG_OP(SplitV)
    .INPUT(x, TensorType::BasicType())
    .INPUT(size_splits, TensorType::IndexNumberType())
    .INPUT(split_dim, TensorType({DT_INT32, DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(SplitV)
```

---

## 7. 关键源码文件汇总

### 7.1 历史设计名称与当前实现对应关系

早期 A3 设计资料中的文件名、类名和当前源码存在差异。阅读旧设计文档或历史提交时，可按下表定位当前实现。表中的“职责变化”表示当前版本的实际职责，不代表旧接口仍然存在。

| 历史设计主题 | 当前实现位置 | 当前关键符号 | 职责变化 |
|------|-----------|-------------|---------|
| Slice/Split lowering 入口 | `ge/compiler/graph/optimize/autofuse/autofuse/lowering/op_lowering_impl/lowering_impl.cpp` | `LowerSlice()`、`LowerStridedSlice()`、`LowerStridedSliceV3()`；Split 相关 `LowerSplit()` | Slice 仍通过 `StoreStridedSlice()` 表达；Split 是否形成 ASCIR Split 由平台能力和 lowering 配置决定 |
| Loop API | `ge/compiler/graph/optimize/autofuse/autofuse/lowering/asc_lowerer/loop_api.cpp` | `StoreStridedSlice()`、`StoreSplit()` | 负责创建 Loop IR 操作，不直接生成最终 AscendC kernel |
| Loop 操作定义 | `ge/compiler/graph/optimize/autofuse/autofuse/lowering/asc_lowerer/loop_ops.h` | `StoreStridedSliceOp`、`StoreSplitOp` | Slice 主要通过索引重写进入 Load view；Split 通过 Split 相关 Loop Op 进入后续 realize |
| Loop IR 到 AscBackend | `ge/compiler/graph/optimize/autofuse/autofuse/lowering/lowerings.cpp`、`asc_ir_lowerer.cpp` | `LoweringManager::LoweringGraph()`、`FusedLoopToAscBackendOp()`、`AscIrLowerer::Lowering()` | 负责 lowering 遍历、fallback，以及将 Loop IR realize 为 `AscBackend` |
| Slice CanFuse 规则 | `ge/compiler/graph/optimize/autofuse/autofuse/can_fuse/strategy/slice_split_fusion_strategy.cpp` | `SliceSplitFusionStrategy` | 基于 Load view 的 repeats、strides、offset 及 Broadcast/Transpose 组合判断，而不是查找 Slice 节点名 |
| Split CanFuse 规则 | `ge/compiler/graph/optimize/autofuse/autofuse/can_fuse/strategy/split_fusion_strategy.cpp` | `SplitFusionStrategy` | 处理 Split 类型 AscBackend 的融合边界；不等同于 graph-autofusion 的 Split 候选生成 |
| Slice lifting | `ge/compiler/graph/optimize/autofuse/autofuse/lowering/liftings.cpp` | `IsSkipLifting()`、`LiftingManager::LiftingGraph()` | 决定单独或低收益的 Slice lowering 结果是否恢复为原 GE 节点 |
| Split lifting 辅助逻辑 | `ge/compiler/graph/optimize/autofuse/autofuse/lowering/op_helper/lower_split_helper.cpp` | `LowerSplitHelper::NeedLifting()` | 仅用于 GE Split lowering 的 lifting 判断，不负责 graph-autofusion 的 schedule 候选选择 |
| Split 后处理 | `ge/compiler/graph/optimize/autofuse/autofuse/lowering/asc_lowerer/` 及 AscBackend post-process 相关实现 | 以当前版本的 Split/AscBackend post-process 符号为准 | 旧资料中的 `adaption_combine_split.h` 不应作为当前源码入口；应从 Split 的 `gid`、AscBackend 属性和 post-process 调用链反查 |
| Slice stride/offset 反推 | `ge/compiler/graph/optimize/autofuse/autofuse/lowering/asc_lowerer/loop_common.cpp`、`loop_ops.cpp` | `LoadOp::Compute()`、`StoreStridedSliceOp::ReIndex()`、`GetStrideAndOffset()` | 将索引映射拆分为 Data/Load stride 和 Load IR offset；不再以独立 Slice ASCIR 节点承载语义 |
| Split 调度候选 | `graph-autofusion/autofuse/optimize/task_generator/split_schedule_case_generator.cpp` | `SplitFusionCaseGenerator::Generate()`、`ResolveSplitDim()`、`ConvertSplitToLoads()` | 当前 graph-autofusion 的 Split 专用入口；负责保留 UB Split 或改写为多路 Load |
| Split codegen | `graph-autofusion/autofuse/v35/codegen/reg_api_call/split_reg_api_call.cpp` | `SplitRegApiCall::IsAllAligned()`、`GenerateForAllAligned()`、`GenerateDefault()` | 根据最终 ImplGraph 生成 `SplitAllAligned` 或 `split::SplitExtend`，不负责候选选择 |

### 7.2 代码库中实际存在的关键文件

| 模块 | 文件路径 | 核心类/函数 |
|------|---------|------------|
| 融合决策基类 | `autofuse/inc/fusion/fusion_decider.h` | `FusionDecider`, `FusionPriority` |
| Scheduler | `autofuse/optimize/task_generator/split_schedule_case_generator.{h,cpp}` | `SplitFusionCaseGenerator` — `FindSplitNodes`, `ResolveSplitDim`, `ConvertSplitToLoads`, `SplitSplits`, `Prepare` |
| Codegen | `autofuse/v35/codegen/reg_api_call/split_reg_api_call.{h,cpp}` | `SplitRegApiCall`， `SplitTiling`， `SplitTilingAllAligned<N>`， `SplitAllAligned`（对齐），`GenerateDefault` → `SplitExtend`（未对齐）， `IsAllAligned`, `NeedB8ToB16` |
| Split+Concat 优化 | `autofuse/v35/optimize/graph_pass/split_concat_optimization_pass.{h,cpp}` | `SplitConcatOptimizationPass` — `RunPass`, `OptimizeOutSplit`, `OptimizeOutConcat` |
| Split 分组 | `autofuse/optimize/task_generator/split_group_partitioner.h` | `SplitGroupPartitioner`, `SplitGroup` |
| Split 打分 | `autofuse/optimize/task_generator/split_score_function_generator.{h,cpp}` | `SplitScoreFunctionGenerator` — `Generate`，`ParseStride`，`TryGetScoreByConstExpr`，`GenerateForUnaligned`（`kMaxUnalignedRate=0.1`, `kAlignment_=32`） |
| Split 注册 | `autofuse/ascir/reg_func/split.cpp` | Tiling 常量、对齐辅助 |
| Split API | `autofuse/v35/ascendc/api_regbase/split.h` | Split API 注册，`SplitExtend`/`SplitExtendInner`（未对齐）， `SplitAllAligned`（对齐）， `SplitCopy`， `DataCopyGatherVf` |
| GE 算子注册 | `ge-master/tests/framework/ge_running_env/include/ge_running_env/op_reg.h:304-452` | `REG_OP(Slice/SliceD/StridedSlice/Split/SplitV)` |

### 7.3 已确认非 Slice/Split 专用的通用基础设施文件

| 文件 | 实际用途 |
|------|---------|
| `autofuse/inc/graph_metadef/exe_graph/lowering/lowering_opt.h` | gert `LoweringOption` 通用配置 |
| `autofuse/inc/graph_metadef/exe_graph/lowering/lowering_definitions.h` | lowering 结果通用定义 |
| `autofuse/inc/graph_metadef/register/graph_optimizer/fusion_common/op_slice_info.h` | FE buffer-fusion 轴切分信息（非 autofusion） |
| `autofuse/inc/graph_metadef/common/sgt_slice_type.h` | FFTS 线程切分类型（非 autofusion） |

---

## 8. 需要支持融合的场景汇总

| 场景 | 说明 |
|------|------|
| 多个 StridedSlice 融合 | 连续切片合并为一次带偏移读取 |
| StridedSlice 和 Slice 融合 | StridedSlice 是全集，Slice 是子集（stride=1） |
| 多个 Split 融合 | N 个 AscBackend 融合成一个 Fuse 节点 |
| Slice 多输出 → 多级 Concat | 训练反向，数据在 HBM 反复倒换，需自动融合减少 HBM 读写 |
| Split/SplitV 多级合并 | GE 图上多级 SplitV 合并成一级再 lowering |
| Split + Concat | 首轴 Split + Concat 可优化掉 Concat |

**核心思想**：通过带 offset 和 stride 的 load/store 实现 Slice/Split/StridedSlice（含 D 变体）的自动融合，将"物理切分"变为"逻辑映射"——后续算子直接从原大张量指定位置（offset+stride）读数，消除无意义内存搬运，使 Slice 变成"读取方式"而非"计算任务"，从而与后续算子（Add/Concat 等）合并为单一任务。

## 相关链接

- 返回 [AutoFuse 架构介绍](../../introduction/architecture.md)
