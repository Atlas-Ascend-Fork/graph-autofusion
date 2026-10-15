# SuperKernel 样例使用指导

## 功能描述

本目录提供两类 SuperKernel Python 样例。各样例的支持范围和运行方法详见样例目录下的 `README.md`。

- `jit/`：通过 SuperKernel JIT 接口完成融合、编译和运行。
- `aot/`：通过 TorchAir `npugraph_ex` 静态编译并启用 SuperKernel 优化。

## 目录结构

```text
examples/
├── run_example.sh                                   # 工程测试批量执行脚本
├── jit/
│   ├── example01_super_kernel_base/                 # SuperKernel 基础功能
│   ├── example02_super_kernel_profiling/            # SuperKernel profiling 对比
│   └── example03_super_kernel_runtime_ascendc_only/ # AscendC + Runtime 极简样例
└── aot/
    ├── scripts/                                     # AOT 样例公共 Bash 函数
    ├── example01_dual_stream/                       # 双流与 NPU Event 控制边
    ├── example02_sk_options/                        # SuperKernel options
    └── example03_kernel_pybind/                     # Pybind 自定义算子融合
```

## 环境依赖

支持如下产品型号：

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

请先参考[源码构建指南](../../docs/zh/build.md)完成环境准备，并按照官方发布的Pytorch与TorchNPU版本进行配套安装，
参见《[Ascend Extension for PyTorch 用户指南](https://www.hiascend.com/document/redirect/pytorchuserguide)》，
再安装对应的 Python 依赖：

```bash
pip install -r super_kernel/examples/requirements.txt
```

## 样例汇总

下表用于快速定位样例。支持范围和运行方法请以对应样例目录下的 README 为准。

| 类型 | 样例 | 文档与运行方法 |
| --- | --- | --- |
| JIT | [example01_super_kernel_base](./jit/example01_super_kernel_base/README.md) | [中文 README](./jit/example01_super_kernel_base/README.md) / [English README](./jit/example01_super_kernel_base/README_en.md) |
| JIT | [example02_super_kernel_profiling](./jit/example02_super_kernel_profiling/README.md) | [中文 README](./jit/example02_super_kernel_profiling/README.md) / [English README](./jit/example02_super_kernel_profiling/README_en.md) |
| JIT | [example03_super_kernel_runtime_ascendc_only](./jit/example03_super_kernel_runtime_ascendc_only/README.md) | [中文 README](./jit/example03_super_kernel_runtime_ascendc_only/README.md) / [English README](./jit/example03_super_kernel_runtime_ascendc_only/README_en.md) |
| AOT | [example01_dual_stream](./aot/example01_dual_stream/README.md) | [中文 README](./aot/example01_dual_stream/README.md) / [English README](./aot/example01_dual_stream/README_en.md) |
| AOT | [example02_sk_options](./aot/example02_sk_options/README.md) | [中文 README](./aot/example02_sk_options/README.md) / [English README](./aot/example02_sk_options/README_en.md) |
| AOT | [example03_kernel_pybind](./aot/example03_kernel_pybind/README.md) | [中文 README](./aot/example03_kernel_pybind/README.md) / [English README](./aot/example03_kernel_pybind/README_en.md) |
