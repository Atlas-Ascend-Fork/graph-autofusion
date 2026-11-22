# 模型常量化编译样例

## 功能描述

本目录提供模型常量化编译样例，分别演示对模型进行优化和仅对指定 Scope 内的算子进行优化。同一模型仅使用一种优化接口。

## 目录结构

```text
examples/
├── common/                     # 共用资源管理
├── custom_op/                  # 共用 AddCustom 算子工程
├── example01_optimize/          # 模型常量化编译
└── example02_optimize_by_flag/  # 指定 Scope 常量化编译
```

## 样例汇总

| 样例 | 场景 | 覆盖接口 | 文档 |
| --- | --- | --- | --- |
| [example01_optimize](example01_optimize/README.md) | 两个 AddCustom 的整个模型常量化编译 | aclmdlRISpecOptimize | [中文 README](./example01_optimize/README.md) / [English README](./example01_optimize/README_en.md) |
| [example02_optimize_by_flag](example02_optimize_by_flag/README.md) | 两个 AddCustom，仅优化 Scope 内的算子 | aclmdlRISpecScopeBegin、aclmdlRISpecScopeEnd、aclmdlRISpecOptimizeByFlag | [中文 README](./example02_optimize_by_flag/README.md) / [English README](./example02_optimize_by_flag/README_en.md) |

## 共用算子

`custom_op` 提供两个形状为 `[1024]` 的 float32 Tensor 逐元素相加的 Add 算子。标准算子包工程已配置 `KERNEL_SPEC_MODE Normal`，构建时一并生成二进制、编译资源和源码包，无需额外的资源准备脚本。

请按各样例 README 中的完整命令依次构建算子包、安装、编译 Host 和运行。两个样例共用 `build/operator` 和 `build/opp`，算子包只需构建和安装一次。
