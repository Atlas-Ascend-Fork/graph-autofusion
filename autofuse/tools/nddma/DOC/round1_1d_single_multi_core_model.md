<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# NDDMA2 Round1 一维多核连续建模

[English](round1_1d_single_multi_core_model_en.md)

## 1. 目标

Round1 复用 `HW_GE_ATT` round2 的 A+C 数据集，建立一维连续 NDDMA 的
分段多核基础模型。

以下路径相对于 `autofuse/tools/nddma/`；采集前设置匹配设备的 `NPU_ARCH` 或传入 `--npu-arch`。

## 2. 运行入口

```text
Modeling/Data_collection/round1/scripts/
├── e2e.py
├── collect.py
├── fit.py
└── draw.py
```

默认产物位于：

```text
Modeling/Ana/round1/
```

## 3. 建模范围

每条配置固定为：

```text
dim = 1
input_stride = 1
output_stride = 1
GM = contiguous
UB = contiguous
MTE2 only; no GM store
src_offset_elem = 0
dst_offset_elem = 0
```

覆盖 `int8_t`、`int16_t`、`int32_t`、`int64_t`，其中
`block_dim` 表示核数。

## 4. 数据集

Round1 使用与 `HW_GE_ATT/NDDMA/Modeling/Data_collection/round2/scripts/plan_common.py`
一致的 A+C fit 数据：

| 组别 | `bytes_per_core` 来源 | `block_dim` | 用途 |
| --- | --- | --- | --- |
| A | `STAGE1_X_BY_DTYPE` | `1,2,4,8,16,32,56` | 代表性核数拟合 |
| C | `C_X_BY_DTYPE` | `1..56` | 全核数拟合 |

当前配置数：

```text
A = 483
C = 1904
A + C = 2387
```

模型中的字节量为：

```text
bytes_per_core = 单个核负责的搬运量
logical_total_bytes = bytes_per_core * block_dim
```

Factor 文件名保留 `round1_1d_single_core_factor.csv` 以兼容现有工作流，
但内容已经是 round2 A+C 数据。

## 5. 建模公式

核数在 `block_dim=3` 处分段：

```text
block_dim <= 2:
    cycles = bytes_per_core / T_1(dtype) + H_1(dtype)

block_dim > 2:
    cycles = bytes_per_core / T_2(dtype) + H_2(dtype)
```

这里的 `cycles` 是单次 NDDMA 的 `mte2` 周期，`T` 单位为
`bytes/cycle`，`H` 单位为 `cycles`。

对每个 dtype，使用与 HW_GE_ATT Round2 相同的分阶段拟合：

1. 仅选择 A/C 的 `fit_role=fit` 一维连续样本。
2. 对相同 dtype、block_dim、bytes_per_core 的 cycles 取算术平均；合并 A/C 重叠点及重复测量。
3. 每个 block_dim 独立普通最小二乘拟合 `cycles = beta*B + H`，得到 `T=1/beta`。
4. 对 `block_dim<=2`、`block_dim>2` 分别将各核数的 T 和 H 等权算术平均。

模型 JSON 的 `per_core_fits` 保存各核数的局部拟合，便于检查。
A+C 原始配置共 2387 条；合并重复 B 后的拟合点数会减少。
数据网格已与 HW Round2 对齐；重新采集的测量值不保证与历史运行相同。

仓库 `Modeling/Ana/round*/*.json` 是此前实测拟合的快照，尚未按此流程重拟合。
本地未包含原始测量 CSV；不能仅靠旧参数恢复新的拟合结果。
获得测量 CSV 后执行 Round1 fit，并依次重拟合 Round2、Round3、Round4，
拟合输出路径与 ad-hoc 预测工具读取的 Ana 路径一致，无需复制到 DOC。
