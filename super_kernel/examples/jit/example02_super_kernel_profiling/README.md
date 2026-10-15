# super_kernel 用例演示

## 用例功能

模型包含6个sk片段，部分sk相同复用缓存，部分多了可输入bias或option配置不同走在线编译

编译使用super_kernel和不使用super_kernel的模型，将性能数据输出

## 融合算子

使用如下with语句块（super_kernel），语句块内算子均被融合为一个超级Kernel进行计算
```python
with torchair.scope.super_kernel("sk1"):
```
详细功能介绍见[图内标定SuperKernel范围](https://www.hiascend.com/document/redirect/PytorchTorchairSuperKernel)。

## 环境依赖

支持如下产品型号：

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

请先参考[源码构建指南](../../../../docs/zh/build.md)完成环境准备，并按照官方发布的Pytorch与TorchNPU版本进行配套安装，
参见《[Ascend Extension for PyTorch 用户指南](https://www.hiascend.com/document/redirect/pytorchuserguide)》，
再安装对应的 Python 依赖：

```bash
pip install -r super_kernel/examples/requirements.txt
```

## 执行命令

在样例目录下执行：

```bash
python3 superkernel_compare.py
```

## 预期执行结果

样例执行成功时会输出如下关键日志：

```text
execute sample success
```
在执行目录生成prof_result文件夹，目录如下，获取数据后对比耗时
```text
prof_result
├── sk_model                             # 带superkernel结果
│  ├── localhost.localdomain_ascend_pt
│     ├── PROF_*
│        ├── mindstudio_profiler_output
│           ├── op_statistic.csv         # profiling数据
├── no_sk_model                          # 不带superkernel结果
│  ├── localhost.localdomain_ascend_pt
│     ├── PROF_*
│        ├── mindstudio_profiler_output
│           ├── op_statistic.csv         # profiling数据
```
分别从两份op_statistic.csv表中得到如下数据
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
|no_sk_model|总耗时|407.54|

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
|sk_model|总耗时|384.66|

对比获得使用super_kernel融合算子的收益是5.61%
