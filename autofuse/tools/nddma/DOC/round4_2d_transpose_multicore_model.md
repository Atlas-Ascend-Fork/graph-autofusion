<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Round4 二维 UB 连续 N_G2 特化：对齐 HW Round5

[English](round4_2d_transpose_multicore_model_en.md)

## 训练集

不再仅使用 G 组；对齐 HW Round5 端到端拟合实际使用的 D/F/G 单核二维数据：

| 组 | 内容 | 配置数 |
| --- | --- | ---: |
| D | is2=1，普通 input stride 的二维 transpose | 2586 |
| F | is2=1，高 stride 增量配置 | 1930 |
| G | is2=2/4/6/8/16 的外层 stride 配置 | 8858 |
| 合计 | 单核训练配置 | 13374 |

D/F 约束 is1>=M；G 约束 is2*M<=is1；统一 GM/UB 安全边界与 HW 相同。
重复测量按 config_id/token 取中位数。其他组及多核样本不会进入参数估计；
缺少任一 dtype 的 D/F/G 组时报告错误，避免静默沿用旧 G-only 训练集。

HW 的 A/E 一维实测锚点用于其分析入口的配对诊断；最终 N_G2 回归分母使用的是
冻结的一维模型预测。本版本直接继承 Round2，不重复采集这些不参与 N_G2 参数估计的锚点。
HW 的 I/K/L/M 外推验证不纳入此训练集，也不参与拟合。

## 公式和拟合

```text
output_dims=[M,N]; input_stride=[is2,is1]; output_stride=[N,1]
B=M*N*dtype_size         # 每核字节数
B1=N*dtype_size
N_base = B/T_1+H_1 (k<=2), B/T_2+H_2 (k>2)
N_G1 = N_1_prime(B1,is1,1,k)  # k>2 时使用一维 rho
observed_N_G2 = (actual_cycles-N_base)/N_G1
N_G2 = g10*is2 + g11_M*M*is2 + g00 + g01_M*M
N_2 = N_base + N_G1*N_G2
```

冻结 Round2 的一维参数；按 dtype 以 `[is2,M*is2,1,M]` 为设计矩阵，
对 observed_N_G2 执行 `numpy.linalg.lstsq(..., rcond=None)`。
与 HW 当前代码一样，为一次联合普通最小二乘，不是两级拟合；不添加正则项、
不做列缩放、不钳制预测，不使用 d_0。
仅单核数据拟合 N_G2；多核预测沿用一维 rho，不额外估计二维多核系数。

## 运行

以下命令在 `autofuse/tools/nddma/` 下执行；采集前设置匹配实际设备的 `NPU_ARCH`，或传入 `--npu-arch`。

```bash
python3 Modeling/Data_collection/round4/scripts/e2e.py all --msprof-bin "$(which msprof)"
python3 Modeling/Data_collection/round4/scripts/e2e.py fit --measurement-csv /path/to/measurements.csv
```

默认继承 `Modeling/Ana/round2/round2_1d_noncontiguous_model.json`，输出到 `Modeling/Ana/round4`。
使用 NumPy。模型 JSON 保存 N_G2 参数、冻结的一维参数及 fit_scope。
Modeling/Ana 中的 JSON 仍是旧实测快照，尚未重拟合；请在更新 Round1/2 后，用新的 D/F/G 实测数据
重新拟合并验证；预测工具直接读取 Ana 模型。此修改不会伪造或覆盖历史实测参数。
