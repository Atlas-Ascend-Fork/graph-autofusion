# super_kernel Use Case Demonstration

## Use Case Function

Model contains 6 sk fragments. Some sk are identical and reuse cache, some add optional input bias or different option configurations requiring online compilation.

Compile models using super_kernel and not using super_kernel, output performance data.

## Operator Fusion

Use the following with statement block (super_kernel). Operators within the statement block are all fused into one super kernel for computation:
```python
with torchair.scope.super_kernel("sk1"):
```
For detailed function introduction, see [Mark SuperKernel Scope in Graph](https://www.hiascend.com/document/redirect/PytorchTorchairSuperKernel).

## Environment Requirements

This sample supports the following product models:

- Atlas A3 training series products/Atlas A3 inference series products
- Atlas A2 training series products/Atlas A2 inference series products

Follow the [source build guide](../../../../docs/en/build.md) and install the officially released matching
PyTorch and TorchNPU versions. See the
[Ascend Extension for PyTorch user guide](https://www.hiascend.com/document/redirect/pytorchuserguide).
Finally, install the sample's Python dependencies:

```bash
pip install -r super_kernel/examples/requirements.txt
```

## Execution Command

In the sample directory, run:

```bash
python3 superkernel_compare.py
```

## Expected Result

On success, the sample prints the following key log:
```text
execute sample success
```

A prof_result folder is generated in execution directory with the following structure. After obtaining data, compare time consumption:
```text
prof_result
├── sk_model                             # with superkernel result
│  ├── localhost.localdomain_ascend_pt
│     ├── PROF_*
│        ├── mindstudio_profiler_output
│           ├── op_statistic.csv         # profiling data
├── no_sk_model                          # without superkernel result
│  ├── localhost.localdomain_ascend_pt
│     ├── PROF_*
│        ├── mindstudio_profiler_output
│           ├── op_statistic.csv         # profiling data
```

Extract the following data from both op_statistic.csv files respectively:

| OP_Type | Core Type | Total Time(us) |
|--|--|--|
| GroupedMatmul | MIX_AIC | 126.26 |
| Transpose | AI_VECTOR_CORE | 90.02 |
| MoeGatingTopK | AI_VECTOR_CORE |  68.32|
|Tile  |  AI_VECTOR_CORE| 24.96 |
| DequantSwigluQuant |AI_VECTOR_CORE  | 24.18 |
| ReduceMeanD |  MIX_AIV| 16.36 |
| ConcatV2D|  AI_VECTOR_CORE| 14.9 |
| ReduceMeanD | AI_VECTOR_CORE | 14.14 |
| SplitVD |AI_VECTOR_CORE  | 10.04 |
| MatMul |AI_CORE  | 6.26 |
| Cast |  AI_VECTOR_CORE|3.96 |
| Data |  AI_VECTOR_CORE| 3.3 |
| StridedSliceD | AI_VECTOR_CORE |3.18  |
| AutomaticBufferFusionOp |AI_VECTOR_CORE  |1.66  |
|no_sk_model|Total Time|407.54|

| OP_Type | Core Type | Total Time(us) |
|--|--|--|
|SuperKernel | MIX_AIC| 172.4|
|Transpose |AI_VECTOR_CORE | 92.42|
|Tile |AI_VECTOR_CORE |24.66 |
|SuperKernel |MIX_AIV | 18.48|
|ReduceMeanD |MIX_AIV | 16.34|
|ConcatV2D |AI_VECTOR_CORE |14.74 |
|ReduceMeanD | AI_VECTOR_CORE| 14.24|
|SplitVD |AI_VECTOR_CORE |10.12 |
|MatMul |AI_CORE | 8.6|
|Cast |AI_VECTOR_CORE |4.08 |
|Data |AI_VECTOR_CORE | 3.72|
|StridedSliceD |AI_VECTOR_CORE |3.1 |
|AutomaticBufferFusionOp |AI_VECTOR_CORE | 1.76|
|sk_model|Total Time|384.66|

Compare to obtain that using super_kernel operator fusion benefit is 5.61%
