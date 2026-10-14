# Concat Automatic Fusion Operator — Comprehensive Technical Document

## 1. Feature Background

### 1.1 Concat Operator Overview

The Concat operator is one of the most fundamental and widely used tensor operations in deep learning and scientific computing. Its core function is to concatenate multiple tensors along a specified dimension.

Example (taking PyTorch as an example):

```python
import torch

a = torch.tensor([[1, 2], [3, 4]])   # shape: (2, 2)
b = torch.tensor([[5, 6], [7, 8]])   # shape: (2, 2)

# Concatenate along dim=0 (stack by rows)
c = torch.cat([a, b], dim=0)
# shape: (4, 2)
# [[1, 2],
#  [3, 4],
#  [5, 6],
#  [7, 8]]

# Concatenate along dim=1 (stack by columns)
d = torch.cat([a, b], dim=1)
# shape: (2, 4)
# [[1, 2, 5, 6],
#  [3, 4, 7, 8]]
```

### 1.2 Fusion Scenarios and Design Goals

Concat fusion is one of the key features of the automatic fusion subsystem. By fusing the Concat operator with its preceding and following operators, it reduces the transfer overhead of intermediate data and improves overall execution performance.

Concat supports the following forward fusion scenarios:

- ElementWise + Concat
- Gather + Concat
- Slice + Concat
- Split + Concat

Among them, Split + Concat fusion has a restriction: Split operates on the first axis and/or Concat operates on the first axis (axes before the split/concat axis whose sizes are all 1 are also treated as the first axis). See 2.2 for details.

Concat supports the following backward fusion scenario:

- Concat + ElementWise

## 2. Special Background and Limitations

### 2.1 Platform and dtype Support Differences

- **A2/A3** (v1 registration): Concat does not support `DT_BOOL` or `DT_INT64` (source code comment: DT_INT64 to be enabled later as needed);
- **A5** (v2 registration): supports `DT_BOOL` and `DT_INT64`.

### 2.2 Split-Concat First-Axis Mutual Exclusion Constraint

At least one of Split and Concat must operate on the first axis for fusion to be allowed; this constraint is guaranteed by the frontend. Graphs that violate the constraint do not enter this feature's processing flow.

### 2.3 Concat Axis Is Not Splittable

In the current implementation, the concat axis and the axes after it do not support splitting. These axes are grouped into the N-group during TilingGroup grouping (see 4.1), which is also the root cause of the "UB not fully loaded" problem that fusion template generation needs to handle.

### 2.4 Input Count Limit in the Inductor Scenario

This scenario invokes the kernel via `<<<>>>` and does not support the ListTensorDesc feature; too many inputs introduce high Scalar overhead. In addition, this scenario does not support concurrent compilation, and too many inputs also lead to longer compilation time. It is recommended not to lower concats whose input count exceeds the threshold.

### 2.5 Backward Fusion Restriction

Operators fused after Concat must not Broadcast along the concat axis again; otherwise the UB buffer size available for Concat becomes too small, causing performance degradation.

### 2.6 Dynamic Shape Scenario Restrictions

1. If dynamic shapes exist on the concat axis or the axes after it, their sizes cannot be computed at compile time and concat grouping is impossible; in this case the Concat can only be placed in a separate group. Too many groups cause long compilation time, oversized function stacks, and other problems, and performance is likely poor. Therefore the frontend needs to limit the number of Concat inputs more strictly, e.g., to 32;
2. If the subgraph involves too many dynamic shape symbols, problems such as oversized tiling data occur. The frontend needs to limit the maximum number of symbols; above that, Concat is not lowered. The current TF frontend limit is 16.

### 2.7 Fusion Threshold Restriction

Due to some AF codegen limitations, the performance of codegen-generated Concat is worse than that of the standalone operator in some scenarios. If the fusion ratio of cat in the generated fused operator is low, degradation occurs. To mitigate this, the frontend computes the fusion ratio, and scenarios below the threshold should not undergo Concat fusion.

- If backward fusion exists, the ratio is considered sufficient and fusion is allowed;
- Otherwise, compute the input ratio: `sum(concat_dim of fusible inputs) / output concat_dim`.

Different cat cases can have different thresholds. For example, TF distinguishes first-axis concat, first-axis concat with no task, all-aligned concat, general Concat, and so on:

```c++
std::map<ConcatCase, ge::float64_t> threshold = {
    {ConcatCase::kFirstDim, 0.0},
    {ConcatCase::kFirstDimNoTaskLarge, 1.0},
    {ConcatCase::kFirstDimNoTaskMid, 0.8},
    {ConcatCase::kFirstDimNoTaskSmall, 0.5},
    {ConcatCase::kFirstDimNoTaskTiny, 0.3333},
    {ConcatCase::kAllAligned, 0.1666},
    {ConcatCase::kOther, 0.1666},
};
```

## 3. External Interfaces

### 3.1 IR Registration

AF only defines the AscIR for Concat. Operators similar to Concat (e.g., tf.stack) have no separate IR definitions and need to be converted into Concat form by the frontend.

The v1 registration is located in `autofuse/ascir/generator/ascir_builtin_ops_v1.cpp`, and the v2 registration is located in `autofuse/v35/ascir/generator/ascir_builtin_ops_v2.cpp`:

```c++
// A2/A3 (v1, DT_BOOL/DT_INT64 not supported)
REG_ASC_IR(Concat)
    .DynamicInput("x", "T")
    .Output("y", "T")
    .ComputeType(ComputeType::kComputeConcat)
    .Impl(v1_soc_versions, {af::ascir::AscIrImplCreator<af::ascir::ConcatAscIrAttImpl>(),
                            af::ascir::AscIrImplCreator<af::ascir::ConcatAscIrCodegenImpl>(),
                            {{"T", TensorType{DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_UINT64,
                                              DT_BF16, DT_FLOAT16, DT_FLOAT}}}});

// A5 (v2, supports DT_BOOL/DT_INT64)
REG_ASC_IR(Concat).Impl(v2_soc_versions,
                        {af::ascir::AscIrImplCreator<af::ascir::ConcatAscIrAttImplV2>(),
                         af::ascir::AscIrImplCreator<af::ascir::ConcatAscIrCodegenImplV2>(),
                         {{"T", TensorType{DT_BOOL, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32,
                                           DT_INT64, DT_UINT64, DT_BF16, DT_FLOAT16, DT_FLOAT}}}});
```

### 3.2 Device-Side AscendC APIs

**A2/A3 (`autofuse/ascendc/api/concat.h`):**

| API | Implementation | Trigger Condition |
| --- | --- | --- |
| `ConcatAllAligned` | DataCopy | All inputs aligned to 32B |
| `ConcatExtendV2` / `ConcatExtendV2Dyn` | TransDataTo5HD | Specialization for small-tail-axis scenarios, outperforms ConcatExtend; `V2Dyn` for dynamic shapes |
| `ConcatExtend` | TransDataTo5HD | Default implementation, used when none of the above specializations apply |

**A5 (`autofuse/v35/ascendc/api_regbase/concat.h`):**

| API | Implementation | Trigger Condition |
| --- | --- | --- |
| `ConcatOneAxis` | Single-axis first-dim concatenation expansion | Single-dim first-axis Concat, tail axis within one vector length |
| `ConcatAllAligned` | DataCopy | All inputs aligned to 32B |
| `ConcatExtendByGather` | DataCopyGather | Input shapes identical |
| `ConcatExtendDyn` | Runtime dispatch | Dynamic shape scenarios; selects between Gather/Extend at runtime |
| `ConcatExtend` | DataCopyScatter | Default implementation, used when none of the above specializations apply |

The API is selected by Codegen based on node information; see 4.4 for the specific trigger conditions.

## 4. Core Implementation

### 4.1 TilingGroup Axis Grouping (tiling_group.cpp)

`autofuse/optimize/autoschedule/tiling_group.cpp` groups loop axes into four groups based on the node's `compute_type`:

| Group | Meaning | Included Axes |
| --- | --- | --- |
| X-group | Transpose input-exclusive axes | Axes present in the Transpose input but absent from its output |
| Y-group | Main computation axes | All loop axes of ElementWise/Broadcast; non-Reduce axes of Reduce |
| R-group | Reduce axes | Axes reduced by Reduce |
| N-group | Non-splittable axes | Axes of size 1; operation axes of Concat/Split |

As described in 2.3, the concat axis and the axes after it do not support splitting and are grouped into the N-group.

### 4.2 Handling of Split-Concat Fusion (split_concat_optimization_pass.cpp)

`autofuse/v35/optimize/graph_pass/split_concat_optimization_pass.cpp`.

At least one of Split and Concat must operate on the first axis for fusion to be allowed (this constraint is guaranteed by the frontend, see 2.2). This pass performs the following graph rewrites to ensure the graph contains only Split or only Concat:

1. If Split operates on the first axis, convert Split into multiple non-contiguous Loads;
2. Otherwise, in this case Concat operates on the first axis, convert Concat into multiple non-contiguous Stores.

### 4.3 Fusion Template Generation (concat_schedule_case_generator.cpp)

`autofuse/optimize/task_generator/concat_schedule_case_generator.cpp`.

Because of the N-group, a complete Concat may not be fully loaded into UB, so templates that split the Concat need to be generated to handle this scenario.

For ease of description, the original axes can be merged into 2 axes with the concat axis as the boundary: the axes before the concat axis are merged into the m axis, and the concat axis and the axes after it are merged into the n axis.

#### 4.3.1 Non-First-Axis Concat: Input Grouping and Splitting

For a **non-first-axis** Concat, the Concat inputs are grouped (`autofuse/optimize/task_generator/concat_group_partitioner.cpp`) to ensure that the fused subgraph of each group can be fully loaded into UB.

The grouping logic is as follows:

1. The sum of the n-axis sizes within a group exceeds the upper limit (if the n axis contains dynamic shapes, it must be placed in a separate group);
2. The number of inputs in a group exceeds the upper limit;
3. Group types do not match.

Different groups may share common ancestor nodes; if these cannot be split apart later, the subgraph still cannot be fully loaded into UB. Therefore nodes shared across groups need to be recomputed, ensuring that each group can be completely split apart later.

After grouping, the original Concat is split according to the grouping result:

- If a group has only one input, connect that input to the output of the backed-up Concat;
- If a group has multiple inputs, construct a new smaller Concat, reconnect the inputs to the new Concat, and connect the new Concat's output to the output of the backed-up Concat;
- After reconnecting the nodes, update the associated axis information and refresh the offset attributes of the associated Load/Store nodes.

#### 4.3.2 First-Axis Concat

For a **first-axis** Concat, if all axes are in the N-group, subsequent processing is impossible, so it is split with each input forming its own group.

However, if the n axis is too small, transfer efficiency drops and performance may degrade. In this case, templates that split into sub-Concats are also generated:

- If there is no axis of size 1 before the concat axis, add an axis of size 1 to ensure that not all axes are located in the N-group;
- Mark that axis as not to be merged.

#### 4.3.3 Template List

In summary:

A first-axis Concat generates 2 templates:

1. Split into sub-Concats;
2. Completely eliminate the Concat.

A non-first-axis Concat generates 2 templates:

1. Keep the Concat (not generated when the input count exceeds the upper limit);
2. Split the Concat, handling the scenario where UB is not fully loaded.

#### 4.3.4 Platform-Specific Scenarios

1. [A5] If there is only one axis and its size fits in UB, add an axis of size 1 in front and keep the original Concat;
2. [A3] If the small-tail-axis scenario is matched, only the keep-Concat template is generated;
3. [A3] If the shape is dynamic, an additional keep-Concat template is generated, marked to choose at runtime whether to use the small-tail-axis version.

#### 4.3.5 Scoring Function

Scoring functions are provided to mitigate inaccurate template selection by ATT:

- **A3**: prefer the small-tail-axis template; when it is not applicable, if the proportion of aligned inputs is high, choose the keep-Concat template; otherwise choose the split-Concat template;
- **A5**: if there is no recomputation, every group has exactly one element after grouping, and the output size is small, prefer the keep-Concat template.

### 4.4 Codegen

Generates the code required to execute Concat based on Concat node information. Computations that can be simplified at compile time should be done here and not left to runtime.

#### 4.4.1 A2/A3: ConcatApiCall

`autofuse/codegen/api_call/concat/concat_api_call.cpp` generates three call forms in the following priority order:

1. **Small-tail-axis path**: taken when the node carries the `_concat_small_tail` attribute (corresponding to the small-tail-axis specialization in 4.3.4):
   - Static concat axis size: `CalcTiling` pre-computes all tiling parameters at compile time and generates `ConcatExtendV2(concat_context, tiling, y, tmp_buf)`;
   - Dynamic concat axis size: generates a runtime `ConcatShape` and generates `ConcatExtendV2Dyn(concat_context, concat_shape, y, tmp_buf)`;
   - `concat_context` is typed by `SameDim/DiffDim` (whether all input concat-segment sizes are 1) and `Padded` (whether padding exists).
2. **All-aligned path**: when `IsAllAligned` holds (each input's concat-segment actual size × dtype size is a multiple of 32B), generates `ConcatAllAligned<dtype, N>(rows, concat_tiling, y, concat_src_tensors)`;
3. **Default path**: generates `ConcatExtend<dtype, vectorized axis count, input count>(dst_params, srcs_params, concat_dim, tmp_buf)`, where the output and each input are described by `ConcatParams` (shape/stride/tensor) structures.

Additional notes:

- **Concat axis identification**: compare the axis sizes of the first input and the output along the vectorized axes; the first position where they differ is the concat axis;
- **tmp buffer**: except for `ConcatAllAligned`, all other paths require a full-lifetime TBuf (`life_time_axis_id = -1`) reuse id;
- **padding**: currently padding is allowed only on the last dimension and is unrelated to the concat axis; when padding exists, extra row stride information is generated in the tiling/shape.

Generated call form example:

```cpp
ConcatAllAligned<float, 3>(concat_rows, concat_tiling, y, concat_src_tensors);
ConcatExtend<float, 2, 3>(dst_params, srcs_params, concat_dim, tmp_buf_0);
ConcatExtendV2(concat_context, tiling, y, tmp_buf_0);
```

#### 4.4.2 A5: ConcatRegApiCall

`autofuse/v35/codegen/reg_api_call/concat_reg_api_call.cpp` selects a path in the following priority order:

1. **`ConcatOneAxis`**: single-dim first-axis Concat (the output has only two vectorized axes with the first stride being 0, and each input's tail axis is static with an element count not exceeding `256 / dtype_size`), generating a constexpr tiling (each input's tail-axis size and output offsets) and calling `concat::ConcatOneAxis`;
2. **`ConcatAllAligned`**: the all-aligned scenario (same determination as A2/A3);
3. **`ConcatExtendByGather`**: generated when the `CanUseGather` conditions are met — no padding, identical input shapes (or the tile case where all inputs come from the same source node), and a single input's concat-segment byte size not exceeding 128B. The gather path prefers contiguous input buffers;
4. **`ConcatExtend` / `ConcatExtendDyn`**: the default scatter path; if the gather conditions are met but the shape is dynamic, `ConcatExtendDyn` is generated, dispatching between Gather/Extend at runtime.

dtype reduction (completed before tiling computation):

- 8-byte dtype (`DT_INT64/DT_UINT64`) → split into `uint32_t` computation (B64ToB32);
- 1-byte dtype with all segment sizes divisible by 2 → converted to `uint16_t` computation (B8ToB16);
- `int8_t`/`bool` → `uint8_t` (pack does not support int8_t).

Generated call form example:

```cpp
concat::ConcatOneAxis<float, 3>((float *)y.GetPhyAddr(), concat_src_addrs, concat_tiling);
concat::ConcatExtendByGather<float, 3>((float *)y.GetPhyAddr(), concat_src_addrs, tmp_buf_0, concat_tiling);
concat::ConcatExtend<float, 3>((float *)y.GetPhyAddr(), concat_src_addrs, tmp_buf_0, concat_tiling);
```

## 5. Common Issue Troubleshooting

### 5.1 Issue 1: Compilation Failure — No Available Tiling

Check the AfterSplitConcat dump graph. After the Concat is split, check whether a repeat update error causes the concat axis to be misidentified.

### 5.2 Issue 2: Compilation Failure — Redefinition Exists

Check the AfterSplitConcat dump graph for the problem of different groups sharing a common ancestor. If so, the recomputation logic has a bug, causing the groups to not be completely split apart.

### 5.3 Issue 3: Accuracy Issues

1. Confirm the actually selected template;
2. If it is a split-Concat template, infer the corresponding group from the position of the data with accuracy issues;
3. Check the corresponding dump graph for problems such as incorrect axis sizes;
4. If it still cannot be located, use the replace_kernel approach to add **printf** statements for debugging.

### 5.4 Issue 4: Performance Issues

1. Rule out performance issues of other operators;
2. Check the template selected by ATT; you can force a specific template to test whether a better choice exists.

## 6. Key File Index

| File Path | Responsibility |
| --- | --- |
| `autofuse/ascir/generator/ascir_builtin_ops_v1.cpp` | Concat IR v1 registration (A2/A3) |
| `autofuse/v35/ascir/generator/ascir_builtin_ops_v2.cpp` | Concat IR v2 registration (A5, supports BOOL/INT64) |
| `autofuse/optimize/autoschedule/tiling_group.cpp` | Loop axis grouping (N-group) |
| `autofuse/v35/optimize/graph_pass/split_concat_optimization_pass.cpp` | Split-Concat graph rewrite pass |
| `autofuse/optimize/task_generator/concat_schedule_case_generator.cpp` | Fusion template generation and scoring functions |
| `autofuse/optimize/task_generator/concat_group_partitioner.cpp` | Concat input grouping |
| `autofuse/codegen/api_call/concat/concat_api_call.cpp` | A2/A3 Codegen |
| `autofuse/v35/codegen/reg_api_call/concat_reg_api_call.cpp` | A5 Codegen |
| `autofuse/ascendc/api/concat.h` | A2/A3 device-side API implementation |
| `autofuse/v35/ascendc/api_regbase/concat.h` | A5 device-side API implementation |

## Related Links

- Back to [AutoFuse Architecture Introduction](../../introduction/architecture.md)
