# Reduce Operator

## Reduce Operator Overview

### 1. What is a Reduce Operator

A Reduce (Reduction) operator is a class of operators that perform aggregation along a specified axis of an input tensor. It iterates over all elements along the specified axis, using a binary operation (such as addition or max) to progressively "reduce" multiple values into a single value, causing the dimension along that axis to disappear (or become 1) in the output.

**Typical examples**:

```
Input x: shape [2, 3] = [[1, 2, 3],
                          [4, 5, 6]]

ReduceSum(x, axis=1, keepdims=False) -> shape [2] = [6, 15]
ReduceMax(x, axis=0, keepdims=False) -> shape [3] = [4, 5, 6]
ReduceMean(x, axis=None, keepdims=False) -> shape [] = 3.5   # all-axis reduction
```

### 2. Reduce Operator Interfaces in Frameworks

#### TensorFlow Interface

TensorFlow unifies all Reduce operators into the `tf.reduce_*` family, with highly consistent interface signatures:

| Interface | Computation Semantics |
|-----------|----------------------|
| `tf.reduce_sum(input_tensor, axis, keepdims)` | Sum |
| `tf.reduce_mean(input_tensor, axis, keepdims)` | Mean |
| `tf.reduce_max(input_tensor, axis, keepdims)` | Maximum |
| `tf.reduce_min(input_tensor, axis, keepdims)` | Minimum |
| `tf.reduce_prod(input_tensor, axis, keepdims)` | Product |
| `tf.reduce_all(input_tensor, axis, keepdims)` | Logical AND (bool) |
| `tf.reduce_any(input_tensor, axis, keepdims)` | Logical OR (bool) |

**Common parameters**:

- `input_tensor`: Input tensor
- `axis`: Reduction axis, can be `None` (all-axis reduction), an integer, or a list/tuple of integers
- `keepdims`: Whether to retain the reduction axis (default `False`, axis disappears; `True` makes the axis become 1)

#### PyTorch Interface

PyTorch also provides `torch.*` Reduce functions, with additional `dim` and `out` parameters:

| Interface | Computation Semantics |
|-----------|----------------------|
| `torch.sum(input, dim, keepdim)` | Sum |
| `torch.mean(input, dim, keepdim)` | Mean (float only) |
| `torch.max(input, dim, keepdim)` | Maximum (returns values + indices when dim is specified) |
| `torch.min(input, dim, keepdim)` | Minimum (returns values + indices when dim is specified) |
| `torch.prod(input, dim, keepdim)` | Product |
| `torch.all(input, dim, keepdim)` | Logical AND (bool) |
| `torch.any(input, dim, keepdim)` | Logical OR (bool) |
| `torch.argmax(input, dim, keepdim)` | Index of maximum |
| `torch.argmin(input, dim, keepdim)` | Index of minimum |

**Key differences from TF**:

- Different parameter name: `axis` -> `dim`
- `torch.max/min` returns `(values, indices)` when dim is specified; TF requires a separate `tf.argmax/argmin` call
- PyTorch's `mean` only supports floating-point types; integers must be cast first

#### Reduce Operator Types in GE

GE defines the following Reduce operator types at the Ascend IR level, mapped to framework operators:

| GE Operator Type | AscIR Fusion Layer Type | Corresponding Framework Operator |
|-----------------|------------------------|--------------------------------|
| ReduceSum | Sum | tf.reduce_sum / torch.sum |
| ReduceMean | Mean | tf.reduce_mean / torch.mean |
| ReduceMax | Max | tf.reduce_max / torch.max |
| ReduceMin | Min | tf.reduce_min / torch.min |
| ReduceProd | Prod | tf.reduce_prod / torch.prod |
| ReduceAll | All | tf.reduce_all / torch.all |
| ReduceAny | Any | tf.reduce_any / torch.any |
| ArgMax | ArgMax | tf.argmax / torch.argmax |

Additionally, GE has variants with a "D" suffix (ReduceSumD, ReduceMaxD, etc.), where axes information is passed via attributes rather than a second input tensor.

#### Prod Data Type Support

In the AscIR V2 path, Prod supports `int8`, `int16`, `int32`, `int64`, and `float32`:

| Data Type | Execution Type | Description |
|-----------|----------------|-------------|
| `int8` | `int16` | Widens input to `int16` for execution and truncates output back to `int8` |
| `int16` | `int16` | Direct execution |
| `int32` | `int32` | Direct execution |
| `int64` | `int64` | Uses the b64 dual-register branch |
| `float32` | `float32` | Direct execution |

The `int8` execution result is truncated back to `int8`. For example, `100 * 100 = 10000` is output as `16` rather than the saturated value `127`.

`bool` and `uint8` are not supported in this release. `Reg::Mul` on 3510-class architectures does not support 8-bit types, so these inputs retain the existing non-fused behavior.

#### Any/All Data Type Support

In the AscIR V2 path, Any/All supports `bool`, `uint8`, `int8`, `int16`, `int32`, `int64`, and `float32`:

| Data Type | Execution Type | Description |
|-----------|----------------|-------------|
| `bool` | `uint8` | Executes through zero-copy reinterpretation of the 0/1 bit pattern |
| `uint8` | `uint8` | Direct execution |
| `int8` | `int8` | Direct execution |
| `int16` | `int16` | Direct execution |
| `int32` | `int32` | Direct execution |
| `int64` | `int64` | Uses the b64 dual-register branch |
| `float32` | `float32` | Direct execution |

Logical inputs must be in the 0/1 value range; non-bool inputs should be normalized to nonzero checks before entering Any/All. `uint16`, `uint64`, `fp16`, `bf16`, and `fp64` are not registered, so these inputs retain the existing non-fused behavior.

### 3. Computation Logic

The core computation logic of Reduce operators can be uniformly described as:

```
output[i] = init_value (⊙) x[i, j1, j2, ..., jk]  (iterate over all elements along reduction axis j)
```

Where:

- `(⊙)` is the binary aggregation operation (`+` for Sum/Mean, `max` for Max, `min` for Min, `x` for Prod, `&&` for All, `||` for Any)
- `init_value` is the initial value for aggregation (Sum->0, Max->-inf, Min->+inf, Prod->1, All->True, Any->False)
- `i` is the index along the non-reduction axis (A-axis), `j` is the index along the reduction axis (R-axis)
- Mean is computed as `Sum / total R-axis element count`, i.e., sum first then divide by the reduction axis length

**Special logic for ArgMax**:

```
output[i] = argmax_j(x[i, j])  # returns the index j that maximizes x[i,j], not the value itself
```

### 4. Essential Differences from Element-wise Operators

| Dimension | Element-wise Operators | Reduce Operators |
|-----------|----------------------|-----------------|
| **Axis transformation** | Axes unchanged, output shape = input shape | Axes reduced, reduction axis disappears or becomes 1 |
| **Data volume** | Output data volume = input data volume | Output data volume < input data volume (compressed along R-axis) |
| **Computation dependency** | Each output element depends only on the input element at the same position | Each output element depends on **all** input elements along the R-axis |
| **Parallelism** | Fully independent, each element can be computed separately | A-axis can be parallelized, R-axis must be serially aggregated |
| **Memory access** | 1 read 1 write, no intermediate data | Multiple reads 1 write, requires intermediate accumulation buffer |
| **Fusion role** | Can fuse forward/backward, can fuse horizontally | Can only backward fuse elementwise (max 3), no horizontal fusion |
| **Hardware alignment** | No special alignment requirements | Last axis must be 32B aligned; only AR or RA mode supported |
| **Initial value** | Not required | Must have an initial value (0, -inf, +inf, 1, etc.) |

**Core difference summary**: Element-wise operators are "per-element transformations" with one-to-one input-output correspondence and natural parallelism. Reduce operators are "cross-element aggregations" where one output depends on multiple inputs, introducing serial dependency and accumulation along the R-axis. This necessitates special handling in parallel scheduling, memory layout, and fusion strategies.

---

## Core Concepts in Auto-Fusion

### 1. A-axis and R-axis

Reduce operators classify the input tensor's axes into two types:

- **A-axis (Axis/Active axis)**: Axes that are not reduced and are preserved in the output. Elements along A-axes can be computed independently in parallel — different A-axis indices correspond to different output elements with no dependencies between them.
- **R-axis (Reduce axis)**: Axes that are reduced, disappearing in the output (`keepdims=False`) or becoming 1 (`keepdims=True`). All elements along R-axes must be aggregated into a single value, introducing serial dependency.

**Example**:
```
Input: shape [B, M, K], ReduceSum along axis=2

A-axis = {B, M}   -> preserved in output, B and M dimensions unchanged
R-axis = {K}       -> eliminated in output

Output: shape [B, M], each output[b, m] = sum(x[b, m, :])  // sum of K elements
```

**Multi-axis reduction**: When `axis` specifies multiple axes, all specified axes are R-axes and the rest are A-axes. For example, `ReduceSum(x, axis=[0,2])` on shape `[B, M, K]` gives R-axis={B, K}, A-axis={M}, output shape `[M]`.

**Norm-like determination**: In auto-fusion, a Reduce where the total R-axis size <= 32 and total A-axis size >= 128 is called a **Norm-like Reduce**. This scenario has a dedicated optimization path (AllLoad template, R-axis loaded in one pass).

### 2. Vectorized Axis and the "Global Single Axis Set" Principle

#### Vectorized Axis

In auto-fusion, each Tensor has two sets of axis descriptions:

- **Full axes (axis)**: All axes that the Tensor logically contains, along with corresponding `repeats` (axis sizes) and `strides` (axis strides)
- **Vectorized axis (vectorized_axis)**: **The axes corresponding to the data actually stored in the UB (Unified Buffer)** — a subset of the full axes

After multi-core partitioning, a Tensor in a single core's UB may only retain data for some axes (intra-block axes). Extra-block axes are traversed through loop unrolling. `vectorized_axis` describes the axes that actually reside in the UB.

**Vectorized strides (vectorized_strides)**: Corresponds one-to-one with `vectorized_axis`, representing the stride when indexing by vectorized axes within the UB:
- `stride = 0`: The axis is a broadcast axis, occupying no storage in UB (data is replicated along this axis, no independent storage needed)
- `stride != 0`: The axis has actual data in the UB; indexing must skip stride elements

Strides are accumulated from the last axis backward: the last axis has stride=1 (or 1 after alignment); intermediate axes have stride=0 if they are broadcast axes, otherwise stride equals the cumulative size of all preceding axes.

#### Global Single Axis Set Principle

A fundamental principle of auto-fusion: **all operators after fusion must share the same axis ID space**.

Before fusion, each operator has independent axis IDs. During fusion, all operators are unified into a single axis ID set through axis mapping:
- **Vertical fusion**: Predecessor node output axes -> successor node input axes, matched by equal axis size values
- **Horizontal fusion**: Between two nodes sharing the same input, data axes are matched by axis size
- After matching is complete, all nodes' axis properties are refreshed to unified IDs

#### Special Role of Reduce in the Global Axis Set

Element-wise operators satisfy "axes unchanged" and naturally share the same axis set with predecessor and successor operators. Reduce operators are different:
- The input R-axes disappear (or become 1) in the output, making the output axis set a subset of the input axis set
- During fusion, the Reduce's input axes must be mapped to the successor elementwise operator's axes. R-axes must be restored to a consistent axis system with the successor via broadcasting (stride=0)
- This is also why Reduce only supports backward fusion with elementwise: the successor operator's axis system is a subset or equal set of the Reduce output, making mapping feasible; forward fusion may fail due to axis count mismatch

### 3. AscendC Interface

AscendC is the low-level programming interface for Ascend AI processors, providing C++ APIs that directly operate on hardware compute units. The kernel code ultimately generated by the auto-fusion CodeGen module is AscendC code.

#### Key Characteristics

- **LocalTensor**: All AscendC API data operations are on `LocalTensor<T>`, which is local memory on the UB. It has limited size; data must be moved from GM to UB before computation
- **Templated**: Most APIs are provided as templates, with parameters including data type `T`, mode `pattern`, whether to reuse source `isReuseSource`, etc.
- **Explicit synchronization**: Data movement and computation require explicit synchronization through barrier instructions such as PipeBarrier

#### Reduce-Related AscendC Interfaces

Auto-fusion wraps the following AscendC interfaces for Reduce operators:

| Interface | Function |
|-----------|----------|
| `WholeReduceSumAdapt` | Sum adaptation layer along the last axis |
| `ReduceLast` | General reduce along the last axis (custom aggregation function) |
| `ReduceSumInt32` | int32-specific reduce sum |
| `ReduceInit` | Fill padding initial values (initialization before Reduce) |
| `ReduceProdExtend` | Product reduce (supports AR/RA dual mode) |
| `ReduceMaxExtend` | int32-specific reduce max |
| `ArgMaxWithValueExtend` | ArgMax (outputs both value and index) |

**ReduceInit padding initial values** (each Reduce type has a corresponding initial value):

| Reduce Type | Padding Value |
|-------------|---------------|
| Sum / Mean | 0 |
| Max | -INF |
| Min | +INF |
| Prod | 1 |
| All | True |
| Any | False |

#### AR/RA Mode

The AscendC Reduce API supports only two data layout modes:

- **AR mode**: Data layout is `[A-axis..., R-axis]`, reducing along the innermost (last) axis. For example, `[M, K]` reduced along the K-axis: the K elements of each row are contiguous, summing row by row
- **RA mode**: Data layout is `[R-axis..., A-axis]`, reducing along the outermost (first) axis. For example, `[K, N]` reduced along the K-axis: N elements are distributed across K rows, summing column by column

**Determination method**: Determined by the output Tensor's vectorized strides — last axis stride=0 means the last axis is an R-axis (eliminated), i.e., AR mode; last axis stride!=0 means the last axis is an A-axis (preserved), i.e., RA mode.

**AR mode requires ReduceInit**: When reducing along the last axis, the last axis size may not be 32B-aligned. Padding initial values must be filled into the unaligned region before executing reduce. RA mode reduces along the first axis and does not require padding initialization.

### 4. Alignment

#### What is Alignment

The vector compute unit of Ascend AI processors requires data access addresses to be **32-byte (32B)** aligned. This is a fundamental hardware constraint for DMA transfers and vector instructions. 32B is the size of 1 Block and is the atomic access granularity for all vector operations.

Number of aligned elements for different data types:

| Data Type | Element Size | 32B-Aligned Element Count |
|-----------|-------------|--------------------------|
| float32 | 4B | 8 |
| float16 | 2B | 16 |
| int32 | 4B | 8 |
| int8 | 1B | 32 |

**Meaning**: A float32 tensor's last axis element count must be a multiple of 8 for efficient access by the vector unit.

#### Alignment Handling in Reduce Scenarios

Reduce is one of the most complex scenarios for alignment handling, involving the following strategies:

**1. Last-axis merge**: When the last axis element count is insufficient for 32B alignment, merge the second-to-last axis with the last axis so the new last axis satisfies the alignment requirement.

**2. ReduceInit padding**: In AR mode, when the R-axis size does not satisfy 32B alignment, fill the corresponding type's padding initial value in the UB (e.g., 0 for Sum, -INF for Max), so the data can be correctly reduced within the aligned range.

**3. Alignment propagation**: The A-axis alignment requirement of the Reduce output must be propagated backward to the input, ensuring the input's last axis satisfies 32B alignment.

**4. RemovePad**: For non-contiguously arranged data, auto-fusion inserts a RemovePad node to compactly rearrange the data before reduce, eliminating padding gaps.

## Root Causes of Special Reduce Handling

All special handling for Reduce in auto-fusion stems from two categories of constraints: **the computational characteristics of Reduce operators themselves** and **the constraints of the underlying AscendC interface**. Understanding these two root causes connects the Reduce handling logic across all subsequent modules.

### Constraint 1: Computational Characteristics of Reduce Operators

#### 1. Fewer Axes After Reduce

After Reduce aggregates along the specified axis, the R-axis disappears (or becomes 1) in the output, making the output shape a subset of the input shape. This causes the following cascading effects:

- **Axis mapping difficulty**: Element-wise operators have unchanged input/output axes, naturally fitting the "global single axis set." Reduce has fewer output axes; when a successor operator wants to fuse with Reduce, the R-axis must reappear in the unified axis system via broadcasting (stride=0), increasing mapping complexity
- **Fusion direction limitation**: Reduce can only backward fuse with elementwise, not forward fuse — because Reduce's output axis set is a subset of the input. In backward fusion, the successor's axes can be filled via broadcasting; in forward fusion, the predecessor's axis set is larger, and the extra axes cannot be mapped to Reduce's output
- **No horizontal fusion support**: Horizontal fusion requires two operators sharing the same input with identical axis sets, but Reduce has a different number of output axes compared to other operators, making unification impossible

#### 2. R-axis Must Be Fully Computed Before Producing Results

Each output element of Reduce depends on the aggregation result of all input elements along the R-axis; partial computation cannot produce output. This leads to:

- **Two-phase execution**: When the R-axis is large, it cannot be reduced in one pass within the UB. Multiple loads of R-axis data are needed with incremental accumulation — Phase 1 performs chunked accumulation along the R-axis, Phase 2 merges intermediate results to produce the final value
- **Serial dependency**: Computation along the R-axis must be serial, unlike elementwise which can be fully parallel. A-axes can be parallelized, but R-axis traversal at the same A-axis position must be completed sequentially
- **Intermediate buffer requirement**: Two-phase execution requires allocating temporary buffers in the UB to store intermediate accumulation results, increasing UB space planning complexity
- **ArgMax is more complex**: ArgMax must track not only the maximum value but also its index. During two-phase merging, the index and value must be passed together, doubling the intermediate data compared to pure value reduction

### Constraint 2: Underlying AscendC Interface Constraints

#### 1. Only AR or RA Input Layouts Supported

The AscendC Reduce API requires input data layout to satisfy one of two modes: either A-axis before R-axis (AR) or R-axis before A-axis (RA). Other axis orders (such as RAA, ARA, or other mixed layouts) are not supported. This leads to:

- **Possible Transpose insertion**: If the original data is neither AR nor RA arranged, auto-fusion must insert a Transpose before Reduce to rearrange axes into AR or RA
- **AR/RA choice affects execution strategy**: AR mode reduces along the last axis, which is more efficient when data is contiguous in the UB; RA mode reduces along the first axis, suitable for Norm-like scenarios where the R-axis is loaded in one pass
- **Axis merge must preserve AR/RA order**: When merging the last axis (merging a small last axis with an adjacent axis to satisfy 32B alignment), the new axis order must still satisfy AR or RA, without violating the mode constraint

#### 2. Last Axis Must Be 32B Aligned

AscendC vector instructions use 32B as the atomic access granularity, requiring the last axis size of operated data to be a multiple of 32B. This leads to:

- **Last-axis merge**: When the last axis element count is insufficient for 32B alignment, the last axis must be merged with the second-to-last axis so the new last axis satisfies alignment
- **ReduceInit padding**: In AR mode, when reducing along the last axis, if the R-axis (last axis) size does not satisfy 32B alignment, padding initial values must be filled to the alignment boundary in the UB. Different Reduce types have different padding initial values (Sum->0, Max->-INF, Prod->1, etc.) to ensure padding does not affect reduction results
- **Alignment propagation**: The A-axis alignment requirement of the Reduce output must be propagated backward to the input, ensuring the entire fused subgraph's last axis satisfies 32B alignment
- **RemovePad**: Non-contiguously arranged data must be compacted before reduction, eliminating gaps introduced by alignment

### Root Cause to Handling Correspondence

Mapping the two root causes to specific handling measures:

| Root Constraint | Specific Manifestation | Derived Handling Measure |
|-----------------|----------------------|------------------------|
| Fewer axes | Output axes are a subset of input axes | Fusion direction limited (backward only), no horizontal fusion, R-axis broadcast restoration |
| Fewer axes | Axis mapping difficulty | Special mapping strategy in global single axis set |
| R-axis serial | All R-axis computation required before output | Two-phase execution (Common/RCore templates), intermediate buffer allocation |
| R-axis serial | ArgMax needs index tracking | ArgMax two-phase (Phase1 local max + index, Phase2 merge) |
| AR/RA only | Other axis orders not supported | Insert Transpose, axis merge preserves AR/RA constraint |
| AR/RA only | Mode choice affects efficiency | Norm-like determination -> AllLoad template (R-axis loaded in one pass) |
| 32B alignment | Last axis below alignment factor | Last-axis merge |
| 32B alignment | AR mode R-axis not aligned | ReduceInit padding |
| 32B alignment | Alignment requirement must be globally satisfied | Alignment backward propagation, RemovePad |

The specific implementations of subsequent modules (Lowering, CanFuse, Schedule, CodeGen) essentially address the above constraints at different stages: Lowering identifies Reduce and marks its type, CanFuse determines whether fusion is possible under the axis-reduction constraint, Schedule plans the two-phase execution strategy and R-axis partitioning, and CodeGen generates AscendC code satisfying AR/RA and 32B alignment.

## Module Processing Details

### Lowering Module

#### 1. Overview

The Lowering module converts Reduce operators in the GE computation graph into AscIR intermediate representation, generating compute units for reduction computation.

#### 2. Key Interface

**`StoreReduction` API**:
```cpp
loop::StoreReduction(loop::ReduceType::SUM, node->GetOutDataAnchor(0), z,
                     {batch_size, m_size, k_size}, {2});  // reduction axis index
```

**ReduceType types**:
- `SUM` -> Sum/Mean operators
- `MAX` -> Max operator
- `MIN` -> Min operator
- `PROD` -> Prod operator

#### 3. Experimental Feature Switch

```cpp
if (meta->type == FuseType::kReduction &&
    !ge::AutoFuseConfig::LoweringConfig().experimental_lowering_reduce) {
    GELOGI("Drop lower result... you can enable it by setting "
           "AUTOFUSE_FLAGS=\"--autofuse_enable_pass=reduce\"");
    meta->type = FuseType::kExtern;  // convert to external call
}
```

**How to enable**:
```bash
export AUTOFUSE_FLAGS="--autofuse_enable_pass=reduce"
```

#### 4. MatMul to Reduce Lowering

When MatMul satisfies specific conditions, it can be lowered to Reduce:

```cpp
// [BS, M, K] * [BS, K, 1] -> [BS, M, 1]
auto x = loop::Load(x_anchor);
auto y = loop::Load(y_anchor);
y = transpose_b ? y : loop::Permute(y, {0, 2, 1});  // [BS, K, 1] -> [BS, 1, K]
y = loop::Broadcast(y, {batch_size, Symbol(1), k_size}, {batch_size, m_size, k_size});
const auto z = loop::Mul(x, y);  // [BS, M, K] * [BS, M, K]
loop::StoreReduction(loop::ReduceType::SUM, node->GetOutDataAnchor(0), z,
                     {batch_size, m_size, k_size}, {2});  // K-axis reduction
```

**Condition checks**:
- `n_size == 1` (MatMul's N dimension is 1)
- `k_size <= max_k_for_vectorize_mm` (K dimension does not exceed threshold)
- `transpose_a == false` (A matrix is not transposed)

---

### CanFuse Module

#### 1. Overview

The CanFuse module determines whether a Reduce operator can fuse with other operators, defining the fusion scope.

#### 2. ReduceFusionStrategy Core Logic

```cpp
bool ReduceFusionStrategy::CanFuse(const NodePtr &node1, const NodePtr &node2) {
  // 1. Check and initialize norm-like reduce state
  CheckAndInitReduceAllLoadState(node1, attr1, node1_desc);
  CheckAndInitReduceAllLoadState(node2, attr2, node2_desc);

  // 2. Check if norm-like state allows fusion
  if (attr1->GetReduceAllLoadState() != REDUCE_ALL_LOAD_NOT_ALL &&
      attr2->GetReduceAllLoadState() != REDUCE_ALL_LOAD_NOT_ALL) {
    return true;
  }

  // 3. Reduce does not support horizontal fusion
  if (BackendUtils::IsHorizontal(node1, node2)) {
    GELOGI("Reduce cannot fuse horizontally");
    return false;
  }

  // 4. Reduce can only backward fuse with elementwise nodes
  if (attr1->HasFuseType(loop::FuseType::kReduction)) {
    if (!BackendUtils::IsOnlyPointwise(node2)) {
      GELOGI("Reduce can only backward fuse with elementwise");
      return false;
    }
    // 5. At most 3 elementwise nodes can be backward fused
    if (attr1->GetReduceFusedElementwiseNodeNum() +
        BackendUtils::GetComputeNodeNumInAscgraph(node2) >
        config.max_reduce_can_fuse_elementwise_nums) {
      GELOGI("Reduce can only backward fuse with at most 3 elementwise");
      return false;
    }
  }
  return true;
}
```

#### 3. Fusion Rule Summary

| Rule | Description |
|------|-------------|
| **No horizontal fusion** | Reduce operators cannot horizontally fuse with other Reduce or non-Reduce operators |
| **Backward fusion only** | At most 3 elementwise nodes can be fused after Reduce |
| **Norm-like priority** | Reduce satisfying norm-like conditions has a special fusion path |

#### 4. Norm-like Reduce Check

```cpp
bool CheckReduceNodeNormLike(const ge::AscNodePtr &asc_node) {
  constexpr int64_t kThresholdTR = 32;    // R-axis threshold (actually computed)
  constexpr int64_t kThresholdTA = 128;   // A-axis threshold (actually >= 16)

  // Compute R-axis and A-axis total sizes
  int64_t r_axis_total_size = 1;
  int64_t a_axis_total_size = 1;
  CalculateRAxisTotalSize(*input_attr_ptr, output.attr,
                          r_axis_total_size, a_axis_total_size);

  // Check conditions
  // R-axis <= 65536 (actual threshold adjusted based on computation)
  // A-axis >= 16 (actual threshold adjusted based on computation)
  if (r_axis_total_size > kThresholdTR || a_axis_total_size < kThresholdTA) {
    return false;
  }
  return true;
}
```

**Norm-like Reduce definition**:
- R-axis (reduction axis) total size is small (<=65536)
- A-axis (non-reduction axis) total size is large (>=16)
- Typical scenarios: LayerNorm, Softmax, etc.

---

### Schedule Module

#### 1. Overview

The Schedule module generates scheduling plans for Reduce operators, determining how the R-axis is partitioned, multi-core strategies, etc.

#### 2. Reduce Template Types

```cpp
// reduce_schedule_case_generator.h
enum class ReduceTemplateType {
  kCommon,    // Common template: R-axis partitioning, multi-phase execution
  kAllLoad,   // All-load template: load all R-axis data at once
  kRCore      // R-core split template: R-axis distributed across multiple cores
};
```

#### 3. Template Selection Logic

```cpp
Status ReducePartitionCaseGenerator::GeneratorTask(...) {
  bool is_norm_like_reduce = optimize::NormLikeReduceChecker::IsNormLikeReduceGraph(optimize_graph);

  if (is_norm_like_reduce) {
    // Norm-like scenario: generate only AllLoad template
    GELOGI("Graph satisfies norm-like reduce, only generate AllLoad tasks");
    GE_CHK_STATUS_RET(GeneratorAllLoadTask(optimize_graph, tasks));
  } else {
    // Non-Norm-like scenario: generate all template types
    GE_CHK_STATUS_RET(GeneratorGeneralTask(optimize_graph, tasks));    // Common
    GE_CHK_STATUS_RET(GeneratorRCoreTask(optimize_graph, tasks));      // RCore
    GE_CHK_STATUS_RET(GeneratorAllLoadTask(optimize_graph, tasks));    // AllLoad
  }
}
```

#### 4. Three Templates in Detail

| Template Type | Applicable Scenario | Execution Method | Pros & Cons |
|---------------|-------------------|-----------------|-------------|
| **Common** | Large R-axis, general scenarios | R-axis partitioning, two-phase execution | Many templates, complex ATT selection |
| **AllLoad** | Norm-like (R-axis<=65536, A-axis>=16) | R-axis loaded at once, single-phase | Good performance, few templates |
| **RCore** | R-axis can be split across cores | R-axis distributed to multiple cores, two-phase | Multi-core parallelism, requires workspace |

#### 5. R-axis Partitioning Logic

```cpp
// 1. Post-fusion R-axis partitioning
GE_CHK_STATUS_RET(ReducePartitionPostFusion(optimize_graph));

// 2. Norm partitioning by loop start/end points
GE_CHK_STATUS_RET(PartitionNorm(optimize_graph, loop_start_end));

// 3. Reduce multi-reference structure partitioning
GE_CHK_STATUS_RET(ReducePartitionMultipleCitations(optimize_graph));
```

**R-axis partitioning impact**:
- Fine axis partitioning -> many templates -> difficult ATT selection
- Cannot merge axes for computation like elementwise
- Requires special handling for last-axis alignment

---

### CodeGen Module

#### 1. Overview

The CodeGen module generates AscendC code for Reduce operators, handling AR/RA mode, last-axis alignment, etc.

#### 2. ReduceApiCall Core Logic

```cpp
Status ReduceApiCall::Generate(...) {
  // 1. Get reduce type and instruction type
  auto &[type_value, instr_type] = reduce_type_map.find(this->api_name_);

  // 2. Determine AR/RA mode
  std::string reduce_pattern;
  GetIsArAndPattern(x, y, x.isAr, reduce_pattern);

  // 3. Generate dtype name (ArgMax special handling)
  GE_CHK_STATUS_RET(GetDtypeNameForReduce(this->api_name_, x, y, dtype_name));

  // 4. Last-axis merge (satisfy 32B alignment)
  ReduceMergedSizeCodeGen(tpipe, ss, x, y);
  ReduceDimACodeGen(x, this->api_name_, ss);

  // 5. Generate ReduceInit code
  ReduceInitCodeGen(x, y, type_value, ss, tpipe, dtype_name);

  // 6. Generate Reduce computation code
  if (!IsNeedMultiReduce(tpipe.tiler, x, y, current_axis.back())) {
    // Single reduce
    ss << "Reduce" << new_api_name << "<" << dtype_name << ", " << reduce_pattern << ", false>"
       << "(y[offset], x[offset], tmp_buf, tmp_reduce_shape, true);";
  } else {
    // Multiple reduces (R-axis chunking)
    // Requires intermediate result accumulation
  }
}
```

#### 3. AR/RA Mode Code Generation

```cpp
void GetIsArAndPattern(const Tensor &x, const Tensor &y, bool &isAr, std::string &reduce_pattern) {
  isAr = (y.vectorized_strides.back() == 0);  // vectorized stride of 0 means AR mode

  const std::map<bool, std::string> reduce_pattern_map = {
    {true, "AR"},   // A-axis first, R-axis last
    {false, "RA"}   // R-axis first, A-axis last
  };
  reduce_pattern = reduce_pattern_map[isAr];
}
```

**AR mode characteristics**:
- `vectorized_strides.back() == 0`: last axis (R-axis) stride is 0
- A-axis is vectorized, R-axis is in the inner loop

**RA mode characteristics**:
- `vectorized_strides.back() != 0`: R-axis is vectorized
- R-axis is in the outer loop, A-axis is inner

#### 4. Last-Axis Alignment Handling

```cpp
void ReduceDimACodeGen(const Tensor &x, const std::string &api_name, std::stringstream &ss) {
  // Check if last axis satisfies 32B alignment
  size_t last_dim_size = x.last_axis_size;
  size_t aligned_elements = 32 / sizeof(dtype);  // aligned element count

  if (last_dim_size < aligned_elements) {
    // Last axis insufficient, need axis merge
    ss << "first_actual = " << merged_axis_size << ";";
    ss << "last = " << last_dim_aligned_size << ";";
  } else {
    // Last axis already satisfies
    ss << "first_actual = " << first_axis_size << ";";
    ss << "last = " << last_dim_size << ";";
  }
}
```

#### 5. ArgMax Multi-Phase Processing

ArgMax requires two-phase processing when the R-axis is large:

```cpp
// Phase1: chunked computation of local maximum and index
ss << "ArgMaxWithValueExtend<int64_t, dtype, pattern>"
   << "(tmp_index, tmp_value, x[offset], tmp_buf, shape);";

// Accumulate offset (for index computation)
if (x.isAr) {
  ss << "accumulated_offset += vectorized_axis.actual_size;";
} else {
  ss << "accumulated_offset += first_actual;";
}

// Phase2: merge local results from all chunks
ss << "UpdateMaxIndexAndValue<dtype>(tmp_index, tmp_value, y[0], saved_value, offset, tmp_buf);";
```

---

### ATT Module

#### 1. Overview

The ATT module performs template search and selection, generating optimal tiling parameters for Reduce.

#### 2. R-axis Identification

```cpp
void AttUtils::CollectReduceAxisNames(const ge::AscNodePtr &node,
                                      std::set<std::string> &reduce_axis_orig_names) {
  // Iterate over node, collect R-axis names
  for (auto &dim : node_info.dims) {
    if (dim->is_reduce_axis) {
      reduce_axis_orig_names.insert(dim->name);
    }
  }
}
```

#### 3. Reduce Axis Marking

```cpp
bool CheckAndMarkReduceSplitAxis(SubAxis *axis,
                                 const std::set<std::string> &reduce_axis_orig_names) {
  if (reduce_axis_orig_names.find(orig_name) != reduce_axis_orig_names.end()) {
    axis->is_reduce_split_axis = true;  // Mark as R-axis split axis
    return true;
  }
  return false;
}
```

**R-axis split characteristics**:
- `is_reduce_split_axis = true`: This axis is an R-axis split axis
- Affects tiling strategy and multi-core plan

#### 4. A-axis Lookup

```cpp
SubAxis *FindAAxis(NodeInfo &node_info) {
  std::set<std::string> reduce_split_axis_names = CollectReduceSplitAxisNames();

  // Find non-R-axis (A-axis) from right to left
  for (auto it = node_info.vectorized_axes.rbegin(); it != node_info.vectorized_axes.rend(); ++it) {
    SubAxis *sub_axis = *it;

    // Skip R-axes and B-axes
    if (ShouldSkipAxis(sub_axis, reduce_split_axis_names)) {
      continue;
    }

    // First non-R-axis found is the A-axis
    return sub_axis;
  }
  return nullptr;
}
```

#### 5. R-axis Partitioning and Performance

**Special handling in ATT**:
```cpp
// 1. If the split axis is simultaneously an R-axis or non-A-axis, divide the performance formula by the split axis loop count
// 2. If the split axis is simultaneously an A-axis and R-axis related loop count is 1, divide by the B split axis loop count
```

---

## Typical Scenario Analysis

### Scenario 1: LayerNorm (Norm-like Reduce)

**Operator sequence**: `Load -> Mean -> Sub -> Div -> Store`

**Characteristics**:
- Small R-axis (e.g., 1024)
- Large A-axis (e.g., 4096)
- Satisfies norm-like condition

**Processing flow**:
1. **Lowering**: Mean -> StoreReduction(SUM)
2. **CanFuse**: All fused into a single subgraph
3. **Schedule**: Select AllLoad template
4. **CodeGen**: AR mode, last axis satisfies 32B alignment
5. **ATT**: R-axis not split across cores, computed in one pass

### Scenario 2: ArgMax with Large R-axis

**Operator sequence**: `Load -> ArgMax -> Store`

**Characteristics**:
- Very large R-axis (e.g., 100000)
- Requires multi-phase processing

**Processing flow**:
1. **Lowering**: ArgMax -> StoreReduction(MAX)
2. **CanFuse**: Single subgraph
3. **Schedule**: Select RCore template (R-axis split across cores)
4. **CodeGen**: Two-phase processing
   - Phase1: Chunked computation of local maximum
   - Phase2: Merge local results from all chunks
5. **ATT**: R-axis multi-core tiling

### Scenario 3: Reduce + Elementwise Fusion

**Operator sequence**: `Load -> Sum -> Add -> Mul -> Store`

**Characteristics**:
- Reduce followed by 2 elementwise operators
- Satisfies fusion condition (<=3)

**Processing flow**:
1. **Lowering**: Sum -> StoreReduction(SUM)
2. **CanFuse**: Allow backward fusion of Add and Mul
3. **Schedule**: Common template, two-phase
   - Stage1: Sum reduction -> workspace
   - Stage2: Add + Mul computation
4. **CodeGen**: RA mode may require Transpose
5. **ATT**: Need to select optimal tiling

---

## Common Issue Troubleshooting

### Issue 1: R-axis Partitioning Causes Too Many Templates

**Symptom**: Template count after fusion far exceeds elementwise fusion

**Causes**:
- Reduce-specific R-axis constraints
- Last-axis alignment requirements prevent axis merging
- AR/RA mode requires Transpose

**Troubleshooting**:
1. Check the template count generated by the Schedule module
2. Analyze R-axis partitioning strategy
3. Check whether norm-like conditions are satisfied

### Issue 2: Last-Axis 32B Alignment Failure

**Symptom**: Reduce API call failure or performance degradation

**Causes**:
- Last-axis element count below alignment requirement
- dtype calculation error

**Troubleshooting**:
1. Check `ReduceDimACodeGen` output
2. Verify that the `last` variable satisfies alignment
3. Check whether axis merge logic is triggered

### Issue 3: AR/RA Mode Selection Error

**Symptom**: Extra or missing Transpose

**Causes**:
- `isAr` determination logic error
- Vectorized axis stride calculation error

**Troubleshooting**:
1. Check `GetIsArAndPattern` output
2. Verify `vectorized_strides.back()` value
3. Analyze Tensor's vectorized axis configuration

### Issue 4: Fusion Scope Not as Expected

**Symptom**: Reduce not fused or too many elementwise fused

**Causes**:
- `max_reduce_can_fuse_elementwise_nums` configuration
- Norm-like check failure

**Troubleshooting**:
1. Check CanFuse logs
2. Check `GetReduceAllLoadState` status
3. Verify Norm-like conditions (R-axis, A-axis sizes)

---

## Appendix: Key Code Locations

| Module | File | Key Function |
|--------|------|-------------|
| Lowering | `autofuse/lowering/asc_lowerer/loop_api.cpp` | `StoreReduction` |
| Lowering | `autofuse/lowering/op_lowering_impl/lowering_impl.cpp` | `TryLowerMatMulToReduce` |
| CanFuse | `autofuse/can_fuse/strategy/reduce_fusion_strategy.cpp` | `ReduceFusionStrategy::CanFuse` |
| Schedule | `optimize/task_generator/reduce_schedule_case_generator.cpp` | `GeneratorTask` |
| Schedule | `optimize/norm_like_reduce_checker.cpp` | `IsNormLikeReduceGraph` |
| CodeGen | `codegen/api_call/reduce/reduce_api_call.cpp` | `ReduceApiCall::Generate` |
| CodeGen | `codegen/api_call/reduce/reduce_api_call_base.cpp` | `GetIsArAndPattern`, `ReduceDimACodeGen` |
| ATT | `att/util/att_utils.cpp` | `CollectReduceAxisNames` |
| ATT | `att/gen_model_info/expr_gen/generate_tiling_expr.cpp` | `FindAAxis` |

## Related Links

- Back to [AutoFuse Architecture Introduction](../../introduction/architecture.md)
