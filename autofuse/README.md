# Autofuse

简体中文 | [English](./README_en.md)

## 简介

AutoFuse是基于Ascend C的自动融合框架，支持自动融合范围识别、自动算子代码生成、Auto Tiling优化、动态shape及混合精度等特性；在算法网络中，由于存在大量的Vector计算，各个Vector计算之间会产生大量的内存搬运，导致Memory Bound问题。而AutoFuse通过自动将多个算子融合为一个算子，减少网络中的算子数量和内存搬运，从而缓解了Memory Bound问题，释放昇腾算力，提升模型的执行性能。

详细介绍，请参考《[Autofuse自动融合](https://www.hiascend.com/document/detail/zh/canncommercial/850/graph/autofuse)》

## Autofuse 目录结构

```text
autofuse/
├── ascendc                # ascendc api 定义
├── ascir                  # 算子注册 ascir
├── att                    # 自动 tiling 生成 模块
├── cmake                  # cmake 脚本文件
├── codegen                # kernel 代码生成 模块
├── common                 # 通用工具方法
├── compiler               # 对外API 接口
├── examples               # 示例脚本，演示典型用法
├── graph_metadef          # 基本图接口
├── inc                    # 供 GE 调用接口
├── optimize               # 调度切分 模块
├── scripts                # 脚本路径
├── tests                  # 测试用例与测试框架
├── tools                  # 调试与分析工具
├── v35                    # 昇腾950 芯片相关优化
├── CMakeLists.txt         # CMake 配置文件
├── blacklist.txt          # 工程配置文件
├── README.md              # 中文说明文档
└── README_en.md           # 英文说明文档
```

## 构建与安装

参考[执行构建](../docs/zh/build.md)。

## 上板验证指导

用户如果想在昇腾设备上体验 Autofuse 的功能与性能，可以先参考[快速安装](../docs/zh/quick_install.md)准备环境。无论是没有昇腾设备的开发者，还是已有昇腾设备的开发者，都可以快速搭建好环境。在此基础上，按照上一步[构建与安装](../docs/zh/build.md)，增量安装了graph-autofusion仓编译生成的cann包。

AutoFuse 当前提供 PyTorch 和 TensorFlow 两种框架下的 Sample 用例，未来我们可能会支持更多框架。可根据实际使用场景参考对应文档完成环境安装和用例执行：

- [PyTorch 场景用例](./examples/pytorch/README.md)
- [TensorFlow 场景用例](./examples/tensorflow/README.md)
