# TensorFlow 场景用例演示

## 功能描述

使用 AutoFuse 完成 TensorFlow 网络下的算子融合。通过 GE（Graph Engine）中的 AutoFuse fusion pass 自动识别可融合算子并完成融合。

## 目录结构

```text
├── README.md
├── README_en.md
├── common/
│   ├── __init__.py
│   ├── config.py                           # 公共配置
│   ├── profiling_utils.py                  # Profiling 公共工具
│   └── tf_runner.py                        # TF1/TF2 公共运行框架
├── af_tf_eleandele/
│   ├── README.md
│   ├── README_en.md
│   └── test_abs_relu_exp.py                # abs + relu + exp 融合样例
├── af_tf_eleandbroadcast/
│   ├── README.md
│   ├── README_en.md
│   └── test_abs_add_relu.py                # abs + add + relu 融合样例
└── af_tf_eleandreduce/
    ├── README.md
    ├── README_en.md
    └── test_abs_reducesum.py               # abs + reduce_sum 融合样例
```

## 前置说明

运行本用例前，需依次完成以下步骤：

> 以下命令均在 graph-autofusion 仓库根目录执行。

1. 通过 [安装指导](../../../docs/zh/quick_install.md) 正确安装 toolkit 和 ops 包，并配置环境变量
2. 通过 [环境编译部署](../../../docs/env_install/tensorflow/env_tf.md) 搭建 TensorFlow 环境（x86_64 可直接 pip 安装，aarch64 需源码编译）
3. 也可使用一键配置脚本自动搭建环境（**仅 x86_64 架构可用**）：

   ```bash
   bash scripts/env_install/tensorflow/setup_tf_env.sh
   ```

   脚本完成后激活环境：

   ```bash
   source scripts/env_install/tensorflow/env/activate_tf1.sh    # TF 1.15
   # 或
   source scripts/env_install/tensorflow/env/activate_tf2.sh    # TF 2.6.5
   ```

   > **aarch64 架构不支持此脚本**：aarch64 请按 [aarch64 架构 TF 源码编译](../../../docs/env_install/tensorflow/build_tf_aarch64.md) 手动编译。

## 设置环境变量

```bash
# cann包安装路径变量定义，值根据实际安装位置设置
export CANN_INSTALL_PATH=/usr/local/Ascend
# 激活cann包中驱动相关环境变量
source $CANN_INSTALL_PATH/driver/bin/setenv.bash
# 激活cann包中toolkit相关环境变量
source $CANN_INSTALL_PATH/ascend-toolkit/set_env.sh

# 假设跑在 device0
export ASCEND_DEVICE_ID=0

# 开启自动融合
export AUTOFUSE_FLAGS="--enable_autofuse=true"
```

## 执行用例

当前包含以下三个用例，可根据实际场景参考对应文档执行：

- [Elementwise + Elementwise](./af_tf_eleandele/README.md)：`abs + relu + exp`
- [Elementwise + Broadcast](./af_tf_eleandbroadcast/README.md)：`abs + add + relu`
- [Elementwise + Reduce](./af_tf_eleandreduce/README.md)：`abs + reduce_sum`

## 预期执行结果

各用例脚本均执行 100 步推理，无报错表示用例执行成功；是否发生融合，需要通过 Dump 图或 Profiling 进一步确认。

## 参考

- [AutoFuse 简介](../../README.md)
- [环境编译部署](../../../docs/env_install/tensorflow/env_tf.md)
- [aarch64 架构 TF 源码编译](../../../docs/env_install/tensorflow/build_tf_aarch64.md)
- [精度调试工具指南](https://hiascend.com/document/redirect/CannCommunityToolAccucacy)
- [Profiling 性能分析工具指南](https://hiascend.com/document/redirect/CannCommunityToolProfiling)
