# super_kernel 用例演示

## 用例功能

sk1 融合 GroupedMatmul+GroupedMatmul+MoeGatingTopK 三个算子

## 使用super_kernel融合算子

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
python3 superkernel_scope.py
```

## 预期执行结果

样例执行成功时会输出如下关键日志：

```text
execute sample success
```
