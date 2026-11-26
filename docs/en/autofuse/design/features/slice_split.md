# Slice/Split/StridedSlice/SplitV(D) Automatic Fusion Operators — Comprehensive Technical Document

English | [简体中文](../../../../zh/autofuse/design/features/slice_split.md)


---

## 1. Feature Background

### 1.1 Project Positioning

The automatic fusion framework supports automatic fusion of **8 data-movement operators**:

| Operator | Input | Output | Core Semantics |
|------|------|------|---------|
| **Slice** | x, offsets, size | y | `output[i] = input[begin+i]`; extracts a contiguous sub-tensor according to begin/size |
| **SliceD** | x | y | Same as Slice, but offsets/size are attributes (not inputs) |
| **SliceDV2** | x, offsets | y | Same as Slice, but size is an attribute |
| **StridedSlice** | x, begin, end, strides | y | A superset of Slice, supporting strides + masks (begin_mask/end_mask/ellipsis_mask/new_axis_mask/shrink_axis_mask) |
| **StridedSliceV2/V3** | x, begin, end, [axes], [strides] | y | Variants of StridedSlice |
| **Split** | split_dim, x | y (dynamic) | Splits the specified axis into N **equal parts**, `output_k[i,j] = input[k*S+i, j]` |
| **SplitV** | x, size_splits, split_dim | y (dynamic) | Splits the specified axis into **unequal parts** (the size of each part is specified by size_splits) |
| **SplitD** | x | y (dynamic) | Same as Split, but split_dim/num_split are attributes |

> The operator registration definitions in GE are available at `ge-master/tests/framework/ge_running_env/include/ge_running_env/op_reg.h:304-452`.

### 1.2 Core Idea

Slice/Split/StridedSlice/SplitV(D) are all **view operators**: the lowering stage does not involve computation and only changes the shape/number of outputs.

**Key insight**: Implementing these operators with loads/stores carrying offsets and strides turns "physical splitting" into a "logical mapping". Subsequent operators (Add/Concat, etc.) read directly from specified positions in the original large tensor (offset+stride), eliminating unnecessary memory movement. Slice thus becomes a "read method" rather than a "computation task", and can be fused with subsequent operators into a single task.

> Analogy: add a bookmark (Offset) to a book instead of tearing out the page and making a copy of it.

### 1.3 Functional Goals

The implementation primarily provides the three core capabilities of the automatic fusion framework:
1. **Fusion strategy solving** (`FusionStrategySolver`) — scoring based on memory savings and proximity
2. **Fusion decision making** (`FusionDecider`) — vertical/horizontal fusion rules
3. **Node fusion processing** (`NpuFusionDecider::Fuse`) — subgraph merging and loop merging

---

## 2. Special Background and Constraints

### 2.1 Split/SplitV Scenarios

SplitV may have a multi-level structure on the GE graph. The multiple levels must be merged into one level before lowering.

---

### 2.2 StridedSlice Constraints

| Constraint | Description |
|------|------|
| **Do not lower a negative stride** | A negative stride involves reversing the data order and is not currently supported |
| **Do not lower dimension increasing/decreasing** | Do not lower StridedSlice scenarios in which new_axis_mask/shrink_axis_mask changes the number of dimensions |
| **Do not lower when the output last axis is 1** | Do not lower scenarios where the last axis of the output data is 1 |

### 2.3 General Fusion Constraints

- These 8 operators, Slice/Split/StridedSlice/SplitV(D), are **restricted from forward fusion**, except that forward fusion is supported when the preceding node is also Slice/Split/StridedSlice/SplitV(D).
- Split operators **do not perform backward fusion** during lowering and are directly realized.
- Split is not fused with reduce.
- Split does not perform horizontal fusion.
- For the scenario where Split/SplitV is followed by multiple Concat operators, the frontend prevents lowering when the number of outputs exceeds 48.

### 2.4 Platform Constraints

- Multiple versions (Tuscany/Florence/Milan/Helper) run in parallel. Complete delivery of packages 3–8 on multiple platforms must be ensured.
- GE is an onetrack component and must not distinguish between chips.
- Loading memory consumption must account for memory-sensitive configurations such as Small HiSilicon and MDC. In principle, a new feature should not increase memory usage.

### 2.5 A3 and A5 Behavioral Differences (Controlled by BackendSpec)

The codebase distinguishes the behavior of A2/A3 and A5 through `BackendSpec::SliceSplitSpec` (`autofuse/inc/backend/backend_spec.h:30-35`):

```cpp
struct SliceSplitSpec {
  bool split_lowered_to_split;    // false: A2/A3 lower Split into multiple StridedSlice operators
                                  // true:  A5  lower Split into multiple Split operators
  bool slice_fuse_with_end_dim_1; // false: A2/A3 do not fuse Slice when the last axis is 1
                                  // true:  A5  can fuse Slice when the last axis is 1
  bool enable_split_flatten;      // false: A2/A3 do not enable flatten for now
                                  // true:  A5  enables flatten
};
```

In addition, `BackendSpec::max_input_nums_after_fuse` (`backend_spec.h:49`) limits the maximum number of inputs of a single node after fusion: **A2A3=8, A5=14**.

---

## 3. External Interfaces

### 3.1 Fusion Decision Interface (FusionDecider)

Defined in `autofuse/inc/fusion/fusion_decider.h`:

```cpp
class FusionDecider {
  // Check whether two nodes can be fused vertically
  virtual bool CanFuseVertical(const NodePtr &node1, const NodePtr &node2) = 0;

  // Check whether two nodes can be fused horizontally
  virtual bool CanFuseHorizontal(const NodePtr &node1, const NodePtr &node2) = 0;

  // Get the priority of a fusion pair
  virtual uint32_t GetFusionPairPriority(const NodePtr &node1, const NodePtr &node2) = 0;

  // Fuse two nodes
  virtual NodePtr Fuse(const NodePtr &node1, const NodePtr &node2);
};
```

**Fusion priority enumeration** (`FusionPriority`):

| Priority | Meaning | Applicable Scenario |
|--------|------|---------|
| HIGHEST | Highest | AscBackend fusion for Split (ensures that N Split backends are fused together) |
| HIGHER | Relatively high | — |
| HIGH | High | — |
| DEFAULT | Default | General operators |
| LOW | Low | — |

> A smaller priority value indicates a higher priority. `GetPossibleFusionsWithHighestPriority` returns only node pairs with the highest priority.

### 3.2 Subgraph Merging Interface

```cpp
// Check whether two subgraphs meet the merging conditions (matching input/output counts + same scheduling axis)
Status CanMergeAscGraph(const ComputeGraphPtr &subgraph1, const ComputeGraphPtr &subgraph2,
                        const NodePtr &node1, const NodePtr &node2);

// Obtain the input/output information of two nodes and record identical inputs and links between nodes
Status GetFuseNodeInfo(const NodePtr &node1, const NodePtr &node2);

// Fuse two nodes into a new node and update data edges and control edges
NodePtr FuseNode(NodePtr node1, NodePtr node2, const ComputeGraphPtr merged_graph);

// Merge two subgraphs through loop analysis
virtual ComputeGraphPtr MergeAscGraphByLoop(const ComputeGraphPtr &subgraph1,
                                             const ComputeGraphPtr &subgraph2,
                                             const NodePtr &node1, const NodePtr &node2);

// Merge two AscBcNode subgraphs into a fused subgraph
ComputeGraphPtr MergeGraphToFusedAscBcNode(const ComputeGraphPtr &subgraph1,
                                            const ComputeGraphPtr &subgraph2,
                                            const NodePtr &node1, const NodePtr &node2,
                                            const NodePtr &fused_node);

// Create a new AscBcNode subgraph
ComputeGraphPtr CreateAscBcNodeSubGraph(const NodePtr &node1, uint32_t in_nums,
                                         uint32_t out_nums,
                                         const std::vector<uint32_t> &node_output_index);

// Merge two subgraphs (merge subgraph2 into subgraph1)
Status MergeSubGraph(const ComputeGraphPtr &subgraph1, const ComputeGraphPtr &subgraph2);

// Connect the netoutput of a subgraph to its input nodes
Status LinkSubGraphNode(const NodePtr &subgraph_netoutput,
                        const ComputeGraph::Vistor<NodePtr> &inputs,
                        const std::vector<std::pair<int32_t, int32_t>> &subgraph_link_map,
                        std::vector<NodePtr> &del_data_nodes);
```

---

## 4. Overall Architecture

### 4.1 Compilation Pipeline

```
GE graph (Slice/Split/StridedSlice/SplitV)
  │
  ▼
┌──────────────────┐
│  Operator symbolic analysis  │  Output shape symbolization, implemented
└──────┬───────────┘
       ▼
┌──────────────────┐
│  Operator Lowering  │  View operators → load/store (with offset+stride)
│                   │  Slice/StridedSlice: store → StoreStridedSliceOp → load
│                   │  Split: store → StoreSplitOp → load (N outputs → N AscBackends)
└──────┬───────────┘
       ▼
┌──────────────────┐
│  CanFuse fusion decision  │  FusionStrategySolver + FusionDecider
│                   │  Score and sort → cycle detection → node fusion
└──────┬───────────┘
       ▼
┌──────────────────┐
│  Scheduler scheduling  │  SplitFusionCaseGenerator: three-scenario template selection
│                   │  Scenario 1: split along the first axis → convert to multiple loads
│                   │  Scenario 2: non-first axis + all small blocks → split within UB template
│                   │  Scenario 3: non-first axis + not all small blocks → group and convert to loads
└──────┬───────────┘
       ▼
┌──────────────────┐
│  Codegen code generation  │  SplitRegApiCall: UB split SIMD code generation
│                   │  AllAligned → SplitAllAligned (DataCopy)
│                   │  Unaligned → GenerateDefault (temporary buffer+scatter)
└──────────────────┘
```

### 4.2 Key Data Structures

#### 4.2.1 Node Pair (NodePair)

Stores fusable node pairs and caches the memory savings and proximity information after fusion:

```cpp
class NodePair {
 public:
  FusingNodePtr first;
  FusingNodePtr second;
  Expression memory_score;    // Memory saved after fusion
  int64_t proximity_score;    // Proximity score
};
```

#### 4.2.2 Read/Write Memory Information (MemoryBuffer)

Stores the output Anchor and symbolic memory size of a Node, using pointers to reduce temporary objects:

```cpp
class MemoryBuffer {
 public:
  const Anchor *buffer;
  const Expression *size;
};
```

#### 4.2.3 Temporary Nodes During Fusion (FusingNode)

A temporary node used during fusion. It stores read/write memory size, the number of original nodes, minimum/maximum ordering, and other information:

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

#### 4.2.4 Split IR Definition

```cpp
REG_ASC_IR(Split)
    .Input("x", "T")
    .DynamicOutput("y", "T")
    .Attr<int64_t>("index")
    .Attr<int64_t>("gid")   // global_id, global ID of the SplitOp
    .ComputeType(ge::ComputeType::kComputeSplit);
```

- `index`: output index
- `gid`: global ID of the Split Op, used to merge Split operators with the same gid in post_process
- `ComputeType::kComputeSplit`: identifies the Split computation type

---

## 5. Core Implementation

### 5.1 Lowering Stage

#### 5.1.1 Design Principle

Slice/Split/StridedSlice are view operators. Lowering does not involve computation and only changes the shape/number of outputs. Based on the symbolic output shape, the coordinates of each output element in the original input must be derived in reverse, followed by Store.

**Key conclusion**: StridedSlice is the complete feature set, and Slice is its subset (`stride=1`).

#### 5.1.2 A3 Implementation

| Step | Implementation |
|------|------|
| Lowering entry | Add `LowerSlice`/`LowerSplit`/`LowerStridedSlice` in `lowering_impl.cpp` and register them in `LoweringManager` |
| Loop API | Add `loop::StoreStridedSlice`/`loop::StoreSplit` in `loop_api.cpp` |
| Loop Ops | Define `StoreStridedSliceOp`/`StoreSplitOp` in `loop_ops.h` |
| Coordinate derivation | `StoreStridedSliceOp` overrides `reindex`: `i_k = start[k] + o_k × stride[k]` |

**reindex formula**: For an element index `(o_0, o_1, ..., o_n)` in the output tensor, the corresponding coordinates `(i_0, i_1, ..., i_n)` in the input tensor are:

```
i_k = start[k] + o_k × stride[k]
```

This allows subsequent operators to read directly from the original tensor and avoids memory movement.

**E2E test verification** (`autofuse/tests/st/codegen/e2e/e2e_load_strided_slice_store.cpp`):

This test verifies the complete lowering flow of StridedSlice and confirms the implementation approach of `StridedSlice → Load(offset, stride) → Store`:

| Stage | Node State |
|------|---------|
| **BeforeAutofuse** | `x0` (Data, axis `[z0, z1=s1+s2]`) → `load0` (Load, `SetOffset(s1)` skips the first s1 elements, strides `{s1+s2, One}` implement the stride pattern) → `store` (Store) → `y` (Output) |
| **AfterInferOutput** | compute_type assignment: Load→`kComputeLoad`, Store→`kComputeStore` |
| **AfterGetApiInfo** | API type: Load/Store→`kAPITypeCompute`; compute unit→`kUnitMTE2` (data movement engine) |
| **AfterScheduler** | Axis splitting (`TileSplit`/`BlockSplit`/`ApplySplit`), vectorization axis/stride settings, aligned to 8 elements (32B/sizeof(float)) |
| **AfterQueBufAlloc** | Memory allocation: x0→GM, load0 output→UB (Queue, depth=2, buf_num=2), store output→GM |

> `load0.ir_attr.SetOffset(s1)` sets the offset (the begin of StridedSlice), and strides `{s1+s2, One}` implement the stride pattern: skip `s1` elements in each row, exactly corresponding to the reindex formula.

#### 5.1.3 A5 Implementation

**Slice/StridedSlice**: same as A3, unchanged.

**Split/SplitV**: In A3, these are lowered into multiple load/store operations, but the codegen performance of unfused load/store operations degrades severely. A5 must support the **split-within-UB template**.

| Change | Description |
|------|------|
| `LowerSplit` → `StoreSplit` | Lower into `store → StoreSplitOp → load` |
| `StoreSplitOp` | Implement `compute`/`realizeImpl`/`InferType` |
| `lower_split_helper.cpp` | Handle Split alignment scenarios and determine whether lifting is required |
| `FuseType = KsplitType` | Add a new fusion type |
| N outputs → N AscBackends | Each output gets one AscBackend after lowering |

**Split lowering does not perform forward/backward fusion and is directly realized.**

### 5.2 CanFuse Fusion Decision

#### 5.2.1 Main Fusion Flow

```cpp
Status FusionStrategySolver::Fuse(const ComputeGraphPtr &graph) const {
  std::vector<FusingNodePtr> nodes;
  GE_ASSERT_SUCCESS(GetNodes(graph, nodes));  // Topologically sort and create FusingNodes
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

#### 5.2.2 Fusion Condition Checks

`FusionStrategySolver::CanFuse` checks three conditions:

```cpp
bool CanFuse(const ComputeGraphPtr &graph, const FusingNodePtr &node1,
             const FusingNodePtr &node2) const {
  // 1. The number of nodes after fusion does not exceed max_fusion_size
  if ((node1->GetFusionNodesSize() + node2->GetFusionNodesSize()) > config.max_fusion_size)
    return false;

  // 2. The read/write memory size saved after fusion is not 0
  if (ScoreFusion::ScoreFusionMemory(*node1, *node2) == kSymbolZero)
    return false;

  // 3. Vertical/horizontal fusion rules
  if (node2->IsAncestor(node1)) {
    if (!GetBackEnd(graph)->CanFuseVertical(node1->GetOrgNode(), node2->GetOrgNode()))
      return false;
  } else {
    // Do not fuse when horizontal fusion increases peak memory
    if (CanFusionIncreasePeakMemory(node1, node2)) return false;
    if (!GetBackEnd(graph)->CanFuseHorizontal(node1->GetOrgNode(), node2->GetOrgNode()))
      return false;
  }
  return true;
}
```

#### 5.2.3 Fusion Scoring Algorithm

Calculates the memory that can be saved by fusing two nodes (reading and writing the same memory means that the corresponding data movement can be eliminated):

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

**Sort comparison**: memory size takes precedence; proximity is considered when the memory sizes are equal; `min_order` is considered when both are equal.

#### 5.2.4 Obtaining Fusion Node Pairs

1. Group by common read/write memory (`buffer_grouping`)
2. Form all pairs within each group and remove duplicates (`repeat_check`)
3. Check with `CanFuse`
4. Group by priority and retain only the highest priority
5. Sort by fusion score

#### 5.2.5 Slice Fusion Rules

Defined in `slice_split_fusion_strategy.cpp` (A3) / the corresponding implementation:

| Rule | Description |
|------|------|
| (a) Forward fusion | The preceding node must be a Slice-type node; otherwise, fusion is not performed |
| (b) Horizontal fusion | Slice does not perform horizontal fusion; however, when two Slice fusion nodes have a vertical relationship (Concat after Slice), fusion must be allowed |
| (c) broadcast | A Slice fusion node containing broadcast is not fused with other nodes |
| (d) reduce | During backward fusion, fusion is not performed when the subsequent node is reduce |

#### 5.2.6 Split Fusion Rules (Added in A5)

Defined in `split_fusion_strategy.cpp`:

| Rule | Description |
|------|------|
| (1) Highest priority | Ensures that N Split AscBackends are fused together during the canfuse stage |
| (2) Do not fuse with reduce | — |
| (3) Do not perform forward fusion | — |
| (4) Do not perform horizontal fusion | — |

After canfuse, the N AscBackends are fused into one Fuse node.

#### 5.2.7 Post-process Merging

`adaption_combine_split.h`: merges Split operators with the same `gid` into one Split Op.

#### 5.2.8 Node Fusion Implementation

`NpuFusionDecider::Fuse` executes the following flow:

1. Obtain the subgraph attributes of the two nodes
2. Use `GetFuseNodeInfo` to obtain input/output links
3. Use `CreateOrUpdateSubgraphOutputAttr` to update subgraph output attributes
4. Whitelist check → use `UnifySubgraphAxis` to unify the scheduling axis
5. If both are AscBc types → attempt loop merging (`CanMergeAscGraph` + `MergeAscGraphByLoop`)
6. Use `FuseNode` to create a new node
7. If loop merging was not performed → use `MergeGraphToFusedAscBcNode` to create an AscBc subgraph
8. Update subgraph attributes and axis attributes

### 5.3 Scheduler Scheduling

Core code: `autofuse/optimize/task_generator/split_schedule_case_generator.{h,cpp}` — class `SplitFusionCaseGenerator`.

#### 5.3.1 Three Scheduling Scenarios

| Scenario | Condition | Strategy | Advantage | Disadvantage |
|------|------|------|------|------|
| **Scenario 1: split along the first axis** | `split_dim == 0` or `split_dim > 0` and the values of all preceding axes are 1 (X Group is empty or its product is 1) | Convert to multiple loads (contiguous large-block memory movement) | Simple to implement; scheduling benefits; store-load cancellation benefits | UB→GM movement is largely not reduced |
| **Scenario 2: non-first axis + all small blocks** | Non-first-axis split, all blocks are small, and UB can hold the entire input | Split-within-UB template: first load the complete data, then split it within UB | Maximum optimization, minimal movement across UB, best performance | Requires full loading + 512B CacheLine alignment; complex UB layout |
| **Scenario 3: non-first axis + not all small blocks** | Non-first-axis split, blocks are not all small or UB cannot hold the entire input | Group each output and convert it to loads as in Scenario 1; process each branch independently | Simple to implement; store/load cancellation benefits | Performance is likely worse than that of a single operator; mainly satisfies architectural constraints |

> Example: `[1,1,3,4]` with split_dim=2 is equivalent to `[3,4]` with split_dim=0 and belongs to the first-axis split scenario.

#### 5.3.2 Key SplitFusionCaseGenerator Methods

| Method | Function |
|------|------|
| `FindSplitNodes` | Traverse the graph to find all Split nodes |
| `ResolveSplitDim` | Resolve the Split dimension |
| `ConvertSplitToLoads` | Convert Split to multiple loads (Scenario 1/3) |
| `SplitSplits` | Split within UB (Scenario 2) |
| `Prepare` | Preprocessing |


#### 5.3.3 Split Grouping

| Component | Function |
|------|------|
| `SplitGroupPartitioner` (`split_group_partitioner.h`) | Groups Split outputs and contains the `SplitGroup` structure |
| `SplitScoreFunctionGenerator` (`split_score_function_generator.h`) | Generates the scoring function for Split scenarios |

> When Split is followed by multiple Concat operators, Concat grouping considers inputs, while Split grouping considers outputs. The frontend currently prevents lowering when the number of outputs exceeds 48.

#### 5.3.4 New A5 Slice Scheduling

| Capability | Status |
|------|------|
| Slice to NDDMA | Currently supported (previously rolled back due to functional issues) |
| Slice split within UB | Not developed at the time because there were too many tiling cases; most current performance issues can be addressed through NDDMA |

### 5.4 Codegen

#### 5.4.1 A3 Codegen

Slice/split is converted to load. For Slice, the offset must be added to the load code; the rest reuses load.

#### 5.4.2 A5 Split-within-UB Codegen

Core code: `autofuse/v35/codegen/reg_api_call/split_reg_api_call.{h,cpp}` — class `SplitRegApiCall`.

**Generation flow**:

```cpp
Status SplitRegApiCall::Generate(...) {
  // 1. Parse split_dim
  size_t split_dim;
  ParseSplitDim(x, y0, split_dim);

  // 2. Initialize tiling information
  SplitTiling split_tiling;
  InitializeTiling(split_dim, outputs, x, split_tiling);

  // 3. Ensure that there is no padding
  GE_ASSERT_TRUE(split_tiling.src_col_actual_size_expr.Simplify() ==
                 split_tiling.src_col_size_expr.Simplify(),
                 "Padding is not supported by split yet");

  // 4. Branch: aligned vs unaligned
  if (IsAllAligned(split_tiling)) {
    GenerateForAllAligned(outputs, x, split_tiling, tpipe.tiler, ss);
  } else {
    GenerateDefault(outputs, x, split_tiling, tpipe, ss, id);  // temporary buffer
  }
}
```

**SplitTiling structure** (`split_reg_api_call.h`):

| Field | Meaning |
|------|------|
| `src_col_size_expr` | Source column size expression |
| `src_col_actual_size_expr` | Actual source column size expression |
| `dst_col_sizes` | Array of output column sizes |
| `src_offsets` | Array of offsets of each output in the source |

**Alignment check** (`IsAllAligned`):

```cpp
constexpr uint32_t kDataBlockSize = 32U;  // SIMD data block width: 32B
// align_size = 32 / sizeof(T)
//   fp16  → 16 elements
//   uint8 → 32 elements
```

Checks whether the column sizes and intervals of all outputs are aligned to `kDataBlockSize` (32 bytes).

**AllAligned SIMD template** (`SplitTilingAllAligned<N>`):

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
        static_cast<uint16_t>(num_rows),           // number of rows
        static_cast<uint16_t>(size / align_size),  // number of blocks copied per row
        static_cast<uint16_t>((tiling.src_col_size - size) / align_size),  // rows to skip between rows
        0};
    DataCopy(dst_tensors[i], src_tensor[tiling.src_offsets[i]], copy_params);
  }
}
```

- Each output uses `DataCopy` and `copy_params` for movement
- `src_offsets[i]` locates the starting position of each output in the source tensor
- Between rows, `src_col_size - size` skips non-target data

**Unaligned scenario — `SplitExtend` path**:

`GenerateDefault` (`split_reg_api_call.cpp:204`) is called when `IsAllAligned` returns false. The core flow is:

1. **Data type promotion optimization**: perform width conversion for unaligned scenarios to reduce movement overhead
   - `uint64` → `uint32_t`: `kB64ToB32 = 2`; column sizes ×2, and move data using a `uint32_t` view
   - `uint8` → `uint16_t`: `NeedB8ToB16()` checks whether all output column sizes are aligned to an even number of bytes; if so, column sizes ÷2, and move data using a `uint16_t` view
   - Other types: use the original dtype directly
2. **Generate the `SplitTiling` structure**: `DefineSplitTiling` outputs fields such as `.num_rows`, `.num_src_cols`, `.num_dsts_cols`, and `.src_offsets`
3. **Generate `SplitExtend` call code**:

```cpp
ss << "split::SplitExtend<" << dtype_name << ", " << outputs.size() << ">("
   << "(" << dtype_name << " *)" << x.GetPhyAddr()
   << ", split_dst_addrs, " << tmp_buf << "_" << tmp_buf_id << ", split_tiling);";
```

4. **`SplitExtend` kernel implementation** (`autofuse/v35/ascendc/api_regbase/split.h:148`):

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

5. **`SplitExtendInner` strategy selection** (`split.h:130`): select a movement method for each output according to its column size
   - `num_dsts_cols[i] > kGatherMaxLen` (`VECTOR_REG_WIDTH / sizeof(T)`) → use `SplitCopy` (row-by-row DataCopy)
   - Otherwise → use `DataCopyGatherVf` (gather vectorized movement, using tmp_buf as an intermediate register)
   - Accumulate `src_col_offset` for each output to locate the starting offset of the next output in the source

#### 5.4.3 Split Scoring Function

`autofuse/optimize/task_generator/split_score_function_generator.{h,cpp}` — class `SplitScoreFunctionGenerator`.

Generates the runtime scoring function `CalcScore(tiling_data)` for Split scheduling. Its return values are:
- `1` = aligned; select the UB split template
- `-1` = unaligned; select the split-to-load template
- `0` = cannot be determined at compile time

**Generation flow** (`Generate`):

```
1. ParseStride()  — calculate the stride after split_dim (dtype_size × product of the sizes of subsequent dimensions)
2. if (const_part_stride_ % 32 == 0) → return 1   // the constant part is already aligned
3. TryGetScoreByConstExpr(score)  — try to calculate the score at compile time
4. if (score != 0) → return score
5. GenerateForUnaligned()  — generate runtime condition-checking code
```

**`TryGetScoreByConstExpr`** (compile-time scoring):
- Iterate through all outputs and check whether `dim × const_part_stride` satisfies 32B alignment
- Alignment-rate threshold: `kMaxUnalignedRate = 0.1` (at most 10% unaligned allowed)
  - `align_threshold = ceil(split_dim_size × (1 - 0.1))`
- If the number of aligned outputs ≥ `align_threshold` → score = 1
- If all strides are constant but the alignment rate is not satisfied → score = -1
- If a stride contains a non-constant → score = 0 (runtime checking required)

**`GenerateForUnaligned`** (runtime scoring code generation):
- Replace size_var in the graph with references to `tiling_data` fields
- Generate the runtime `CalcScore` function:
  ```cpp
  int32_t CalcScore(const AutofuseTilingData &tiling_data) {
    const auto &t = tiling_data.graph0_result0_g0_tiling_data;
    auto stride = static_cast<int64_t>(<stride_expr>);
    if (stride % 32 == 0) { return 1; }
    std::vector<int64_t> split_dims;
    split_dims.reserve(N);
    // ... fill in the split_dim size of each output
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
- Logic: check alignment output by output. Once an output is unaligned, treat all subsequent outputs as unaligned; return -1 when the accumulated number of unaligned elements exceeds `max_unaligned`

**Relationship with scheduling** (`split_schedule_case_generator.cpp:73`):
- `score_functions.resize(2U)`: corresponds to two templates (`ub_split` + `split_2_load`)
- By default, no scoring function is generated (empty string), meaning that template selection is decided directly at compile time

#### 5.4.4 Split Registration Functions

`autofuse/ascir/reg_func/split.cpp`: defines Split tiling constants and alignment helper functions.
`autofuse/v35/ascendc/api_regbase/split.h`: Split API registration.

---

## 6. Operator Registration Definitions (GE)

Related operator definitions in GE (`ge-master/tests/framework/ge_running_env/include/ge_running_env/op_reg.h`):

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

## 7. Summary of Key Source Files

### 7.1 Mapping Between Historical Design Names and the Current Implementation

The file names and class names in early A3 design materials differ from those in the current source code. When reading old design documents or historical commits, use the following table to locate the current implementation. The "Responsibility Change" column describes the actual responsibility in the current version and does not mean that the old interface still exists.

| Historical Design Topic | Current Implementation Location | Current Key Symbols | Responsibility Change |
|------|-----------|-------------|---------|
| Slice/Split lowering entry | `ge/compiler/graph/optimize/autofuse/autofuse/lowering/op_lowering_impl/lowering_impl.cpp` | `LowerSlice()`, `LowerStridedSlice()`, `LowerStridedSliceV3()`; Split-related `LowerSplit()` | Slice is still expressed through `StoreStridedSlice()`; whether Split forms ASCIR Split is determined by platform capabilities and lowering configuration |
| Loop API | `ge/compiler/graph/optimize/autofuse/autofuse/lowering/asc_lowerer/loop_api.cpp` | `StoreStridedSlice()`, `StoreSplit()` | Creates Loop IR operations and does not directly generate the final AscendC kernel |
| Loop operation definitions | `ge/compiler/graph/optimize/autofuse/autofuse/lowering/asc_lowerer/loop_ops.h` | `StoreStridedSliceOp`, `StoreSplitOp` | Slice primarily enters Load view through index rewriting; Split enters subsequent realization through Split-related Loop Ops |
| Loop IR to AscBackend | `ge/compiler/graph/optimize/autofuse/autofuse/lowering/lowerings.cpp`, `asc_ir_lowerer.cpp` | `LoweringManager::LoweringGraph()`, `FusedLoopToAscBackendOp()`, `AscIrLowerer::Lowering()` | Handles lowering traversal and fallback, and realizes Loop IR as `AscBackend` |
| Slice CanFuse rules | `ge/compiler/graph/optimize/autofuse/autofuse/can_fuse/strategy/slice_split_fusion_strategy.cpp` | `SliceSplitFusionStrategy` | Makes decisions based on repeats, strides, offsets, and Broadcast/Transpose combinations of Load views, rather than looking up Slice node names |
| Split CanFuse rules | `ge/compiler/graph/optimize/autofuse/autofuse/can_fuse/strategy/split_fusion_strategy.cpp` | `SplitFusionStrategy` | Handles fusion boundaries for Split-type AscBackends; it is not equivalent to Split candidate generation in graph-autofusion |
| Slice lifting | `ge/compiler/graph/optimize/autofuse/autofuse/lowering/liftings.cpp` | `IsSkipLifting()`, `LiftingManager::LiftingGraph()` | Determines whether the result of standalone or low-benefit Slice lowering should be restored to the original GE node |
| Split lifting helper logic | `ge/compiler/graph/optimize/autofuse/autofuse/lowering/op_helper/lower_split_helper.cpp` | `LowerSplitHelper::NeedLifting()` | Used only for the lifting decision in GE Split lowering; does not select graph-autofusion schedule candidates |
| Split post-processing | `ge/compiler/graph/optimize/autofuse/autofuse/lowering/asc_lowerer/` and AscBackend post-process implementations | Use the Split/AscBackend post-process symbols in the current version | `adaption_combine_split.h` in old materials should not be used as the current source entry; trace back from Split `gid`, AscBackend attributes, and the post-process call chain |
| Slice stride/offset reverse derivation | `ge/compiler/graph/optimize/autofuse/autofuse/lowering/asc_lowerer/loop_common.cpp`, `loop_ops.cpp` | `LoadOp::Compute()`, `StoreStridedSliceOp::ReIndex()`, `GetStrideAndOffset()` | Splits index mapping into Data/Load strides and Load IR offsets; the semantics are no longer carried by an independent Slice ASCIR node |
| Split scheduling candidates | `graph-autofusion/autofuse/optimize/task_generator/split_schedule_case_generator.cpp` | `SplitFusionCaseGenerator::Generate()`, `ResolveSplitDim()`, `ConvertSplitToLoads()` | Dedicated Split entry point in the current graph-autofusion; retains UB Split or rewrites it into multiple Load paths |
| Split codegen | `graph-autofusion/autofuse/v35/codegen/reg_api_call/split_reg_api_call.cpp` | `SplitRegApiCall::IsAllAligned()`, `GenerateForAllAligned()`, `GenerateDefault()` | Generates `SplitAllAligned` or `split::SplitExtend` according to the final ImplGraph; does not select candidates |

### 7.2 Key Files Actually Present in the Codebase

| Module | File Path | Core Classes/Functions |
|------|---------|------------|
| Fusion decision base class | `autofuse/inc/fusion/fusion_decider.h` | `FusionDecider`, `FusionPriority` |
| Scheduler | `autofuse/optimize/task_generator/split_schedule_case_generator.{h,cpp}` | `SplitFusionCaseGenerator` — `FindSplitNodes`, `ResolveSplitDim`, `ConvertSplitToLoads`, `SplitSplits`, `Prepare` |
| Codegen | `autofuse/v35/codegen/reg_api_call/split_reg_api_call.{h,cpp}` | `SplitRegApiCall`, `SplitTiling`, `SplitTilingAllAligned<N>`, `SplitAllAligned` (aligned), `GenerateDefault` → `SplitExtend` (unaligned), `IsAllAligned`, `NeedB8ToB16` |
| Split+Concat optimization | `autofuse/v35/optimize/graph_pass/split_concat_optimization_pass.{h,cpp}` | `SplitConcatOptimizationPass` — `RunPass`, `OptimizeOutSplit`, `OptimizeOutConcat` |
| Split grouping | `autofuse/optimize/task_generator/split_group_partitioner.h` | `SplitGroupPartitioner`, `SplitGroup` |
| Split scoring | `autofuse/optimize/task_generator/split_score_function_generator.{h,cpp}` | `SplitScoreFunctionGenerator` — `Generate`, `ParseStride`, `TryGetScoreByConstExpr`, `GenerateForUnaligned` (`kMaxUnalignedRate=0.1`, `kAlignment_=32`) |
| Split registration | `autofuse/ascir/reg_func/split.cpp` | Tiling constants, alignment helpers |
| Split API | `autofuse/v35/ascendc/api_regbase/split.h` | Split API registration, `SplitExtend`/`SplitExtendInner` (unaligned), `SplitAllAligned` (aligned), `SplitCopy`, `DataCopyGatherVf` |
| GE operator registration | `ge-master/tests/framework/ge_running_env/include/ge_running_env/op_reg.h:304-452` | `REG_OP(Slice/SliceD/StridedSlice/Split/SplitV)` |

### 7.3 General Infrastructure Files Confirmed Not to Be Slice/Split-Specific

| File | Actual Use |
|------|---------|
| `autofuse/inc/graph_metadef/exe_graph/lowering/lowering_opt.h` | General gert `LoweringOption` configuration |
| `autofuse/inc/graph_metadef/exe_graph/lowering/lowering_definitions.h` | General lowering result definitions |
| `autofuse/inc/graph_metadef/register/graph_optimizer/fusion_common/op_slice_info.h` | FE buffer-fusion axis splitting information (not autofusion) |
| `autofuse/inc/graph_metadef/common/sgt_slice_type.h` | FFTS thread splitting type (not autofusion) |

---

## 8. Summary of Scenarios Requiring Fusion Support

| Scenario | Description |
|------|------|
| Fusion of multiple StridedSlice operators | Merge contiguous slices into one offset-based read |
| StridedSlice and Slice fusion | StridedSlice is the complete feature set, and Slice is its subset (`stride=1`) |
| Fusion of multiple Split operators | Fuse N AscBackends into one Fuse node |
| Slice with multiple outputs → multi-level Concat | During training backpropagation, data is repeatedly moved back and forth in HBM; automatic fusion is required to reduce HBM reads and writes |
| Multi-level Split/SplitV merging | Merge multi-level SplitV on the GE graph into one level before lowering |
| Split + Concat | Concat can be optimized away for first-axis Split + Concat |

**Core idea**: Implementing automatic fusion of Slice/Split/StridedSlice (including D variants) with loads/stores carrying offsets and strides turns "physical splitting" into a "logical mapping": subsequent operators read directly from specified positions in the original large tensor (offset+stride), eliminating unnecessary memory movement. Slice thus becomes a "read method" rather than a "computation task", and can be fused with subsequent operators (Add/Concat, etc.) into a single task.

## Related Links

- Back to [AutoFuse Architecture Introduction](../../introduction/architecture.md)
