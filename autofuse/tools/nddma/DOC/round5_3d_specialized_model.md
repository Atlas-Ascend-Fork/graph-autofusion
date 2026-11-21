<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Round5 三维特化模型

[English](round5_3d_specialized_model_en.md)

Round5 使用 Round3 的相同多维公式，但只用三维测量数据重新拟合修正参数。
Round2 中的 `base` 参数完整复制且不参与优化；不增加或拟合 `d_0`，不使用二维 `N_G2`。
Round3、Round4 的模型及既有预测工具继续使用各自原有参数。

## 公式和参数

API 中 `output_dims`、`input_stride`、`output_stride` 均按外层到内层排列；
计算各轴项时反转为内层到外层，记为 `ls_j`、`is_j`、`os_j`。

```text
B = product(output_dims) * dtype_size                 # 每核字节数，不除以核数
N_base = B/T_1 + H_1, k<=2; B/T_2 + H_2, k>2         # 参数固定继承 Round2
B_j = B / product(ls_t for t<j)
is_hat_0 = is_0; os_hat_0 = os_0
is_hat_j = abs(is_j - sum(ls_t*is_t for t<j)) + 1     # j>0
os_hat_j = abs(os_j - sum(ls_t*os_t for t<j)) + 1
s_j = min(is_hat_j * dtype_size, 128)
g_j = min(1, max(0, os_hat_j - 1))

N_G(j) = (a1 + a2*B_j)*s_j
N_GU(j) = (b1 + b2*s_j + (b3 + b4*s_j)*B_j)*g_j
rho(j) = c1 + c2*s_j + g_j*(c3 + c4*s_j)

N_3 = N_base + sum(N_G(j) + N_GU(j)), k<=2
N_3 = N_base + sum((N_G(j) + N_GU(j))*rho(j)), k>2
```

每个 dtype 独立拟合 10 个参数：`a1,a2,b1,b2,b3,b4,c1,c2,c3,c4`。

1. `k<=2` 的三维数据：从实测 cycles 减去固定的 `N_base`，对各轴求和后的
   `[s_j, s_j*B_j, g_j, g_j*s_j, g_j*B_j, g_j*s_j*B_j]` 六列联合做 OLS，求出 `a/b`。
2. `k>2` 的三维数据：固定刚拟合的 `a/b`，令 `q_j=N_G(j)+N_GU(j)`，
   对各轴求和后的 `[q_j, q_j*s_j, q_j*g_j, q_j*g_j*s_j]` 做 OLS，求出 `c`。

两步都对列做数值缩放后使用 `numpy.linalg.lstsq(rcond=1e-12)`，再恢复系数；
没有岭正则、系数裁剪或额外截距。此流程是分阶段 OLS，不是十参数联合非线性优化。
四个 dtype 都必须具有两个核数分支的数据，且设计矩阵分别满秩 6 和 4；否则报错，
不会静默沿用旧修正参数。仅有 HW Round5 J / Round6 单核数据不足以拟合 `rho`。

## 数据集

采集复用 V2 Round3 的三维配置：HW Round5 J 的 18743 个配置，以及
HW Round6 的 6780 个配置；保留原有布局、stride 和容量过滤。
包含三维转置、原有 `continuous` 布局及 GM 非连续访问。
`continuous` 的 stride 也照搬既有 Round3 实现。

每个配置扩展到 `block_dim=(1,2,4,8,16,32,56)`，共 **178661** 个采集配置，
dtype 为 `int8_t/int16_t/int32_t/int64_t`。`repeat=200`，默认执行重复次数为 3。
`bytes_per_core` 与 `total_bytes` 为每核数据量，`logical_total_bytes=bytes_per_core*block_dim`。
GM 上限 4 MiB、UB 上限 256 KiB，沿用每核 64 元素 guard。

拟合只读取 `dim=3` 的行，以 dtype、核数、形状及两侧 stride 为键合并，
先对各次执行的正数 cycles 归一化，再取中位数。同一配置出现在多个源网格时也会合并。
`actual_y` 已按单次内核归一化；其他支持的 cycles 列按 `repeat` 归一化。
输入 CSV 的三维行都作为训练数据，包括从原 Round3 验证集中复用的三维行。
输出误差为训练集误差，不代表独立验证集表现。

## 运行

以下命令在 `autofuse/tools/nddma/` 下执行；采集前设置匹配实际设备的 `NPU_ARCH`，或传入 `--npu-arch`。

在 `autofuse/tools/nddma/` 目录运行，依赖 NumPy；采集需要 Ascend/CANN 和 msprof。

```bash
python3 Modeling/Data_collection/round5/scripts/e2e.py all \
  --msprof-bin "$(command -v msprof)"
```

也可以分开执行：

```bash
python3 Modeling/Data_collection/round5/scripts/e2e.py collect --msprof-bin "$(command -v msprof)"
python3 Modeling/Data_collection/round5/scripts/e2e.py fit
python3 Modeling/Data_collection/round5/scripts/e2e.py draw
```

直接使用已有测量数据，`--measurement-csv` 可重复指定：

```bash
python3 Modeling/Data_collection/round5/scripts/e2e.py fit \
  --round2-model Modeling/Ana/round2/round2_1d_noncontiguous_model.json \
  --measurement-csv /path/to/3d_single_dual_core.csv \
  --measurement-csv /path/to/3d_multicore.csv
```

`e2e.py` 支持 `--run-output-dir`、`--data-dir`、`--model-output-dir`、`--factor-csv`，
以及只打印命令、不执行子程序的 `--dry-run`。更多采集选项可直接传给 `collect.py`。

默认输出在 `Modeling/Ana/round5/`：

- `round5_3d_factor.csv`：采集配置。
- `collection/`：编译、执行、profiling 和测量结果。
- `round5_3d_specialized_model.json`：固定 Base 与新拟合的修正参数、拟合样本计数和误差。
- `round5_3d_specialized_predictions.csv`：逐配置预测及原 Round2 参数预测对照，包含核数。
- `figures/`：按 dtype 和核数分别绘制新参数及继承参数的相对误差图。

使用新模型预测：

```bash
python3 Modeling/Data_collection/round5/scripts/predict.py \
  4x8x16 1x4x64 128x16x1 8 int32_t \
  --model Modeling/Ana/round5/round5_3d_specialized_model.json
```

模型 JSON 必须由实测数据拟合生成；仓库不预置虚构的三维拟合参数。
