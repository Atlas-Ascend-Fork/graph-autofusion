<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# standalone_nddma

[English](README_en.md)

这是 NDDMA 灵敏度扫描采集程序，只测量 v35 NDDMA 的 MTE2 搬入，不将 UB 数据搬回 GM。
构建前应按实际设备设置 `NPU_ARCH`，或传入 `-DNPU_ARCH=...`。
CPU 调试还需指定已安装的 `-DNDDMA_CPU_TARGET=tikicpulib::<target>`。

在本目录为灵敏度组构建可执行文件：

```bash
cmake -S . -B build/bytes_shape -DNDDMA_STAGE=2
cmake --build build/bytes_shape --target demo_nddma -j 8
```

使用生成的配置 CSV 运行：

```bash
./build/bytes_shape/demo_nddma --factor-csv=/path/to/v1_round1_bytes_shape.csv
```

`NDDMA_STAGE` 仅为灵敏度组的编译期标签：

- `1`：dtype/维数。
- `2`：字节数/形状。
- `3`：输入 stride。
- `4`：输出 stride。
- `5`：对齐。
- `6`：输入 stride 为零/广播。

程序从配置 CSV 读取 dtype、dim、output_dims、output_stride、input_stride、偏移和重复次数，
通过 dim=1..5 的模板实例调用 `DataCopyNddma<T, dim>`。
没有空阶段或外层循环建模路径；每个测量用例都重复执行 NDDMA API。

默认 API 头文件为仓内 `autofuse/v35/ascendc/api_regbase/datacopy_nddma.h`。
可通过 `-DREGBASE_INCLUDE_DIR=/path/to/api_regbase` 使用外部检出目录。
CANN 环境要求及历史 `NdDmaDci` 测量差异见
[模型说明](../../../../../README_en.md#dependencies-and-reproducibility)。
