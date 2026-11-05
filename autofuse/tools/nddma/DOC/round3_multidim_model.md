<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Round3 多维统一模型：对齐 HW_GE_ATT，不使用 d_0

[English](round3_multidim_model_en.md)

继承 Round2 的 base、N_G、N_GU、rho，不拟合任何新的多维参数。
与 HW Round7 的未缩放预测（以及统一预测工具）对齐，**不拟合或应用 d_0**。
HW Round7 拟合 d_0 后的结果不作为本版本对齐目标。

## 数据集

| 来源 | 内容 | 安全筛选后配置数 |
| --- | --- | ---: |
| HW Round4 二维 D/E | 完整 GM padding 与 UB gap 网格 | 8532 |
| HW Round5 J | 三维参数继承配置，也是 HW Round7 联合评估的三维来源 | 18743 |
| HW Round6 | 四种基础布局及 11 个 GM stride 节点 | 6780 |
| HW Round7 | 四维、五维三种布局 | 14313 |
| 合计 | 全部用于评估，不估计新参数 | 48368 |

安全筛选为 GM span（含 64 元素 guard）不超过 4 MiB，UB span（含 guard）不超过 256 KiB。
默认 kernel repeat=200、execution repeat=3。不同 stride/来源保留独立配置 ID，避免误合并。
Round6 的 `continuous` 布局严格保留 HW 代码的 `(n*k,n,1)` 输入和输出；
它不总是通常意义的连续布局，但这里不擅自更改参考实验描述符。

## 计算

将 API 的 output_dims/input_stride/output_stride 反转为内层优先的逻辑轴顺序。

```text
B = product(output_dims)*dtype_size  # 每核描述符字节数，不再除以核数
N_base = B/T_1+H_1 (k<=2), B/T_2+H_2 (k>2)
B_j = B / product(ls_t for t<j)
is_hat_0=is_0; os_hat_0=os_0
is_hat_j=abs(is_j-sum(ls_t*is_t for t<j))+1
os_hat_j=abs(os_j-sum(ls_t*os_t for t<j))+1
N_D = N_base + sum_j N_1_prime(B_j,is_hat_j,os_hat_j,k)
```

N_1_prime 为 Round2 的 N_G+N_GU，k>2 时乘一维 rho。
N_base 只计算一次；不使用旧 h 参数，也不增加 d_0 缩放。
测量值只用于计算评估误差，不影响继承参数或预测。

## 运行

以下命令在 `autofuse/tools/nddma/` 下执行；采集前设置匹配实际设备的 `NPU_ARCH`，或传入 `--npu-arch`。

```bash
python3 Modeling/Data_collection/round3/scripts/e2e.py all --msprof-bin "$(which msprof)"
python3 Modeling/Data_collection/round3/scripts/e2e.py fit --measurement-csv /path/to/measurements.csv
```

默认读取 `Modeling/Ana/round2/round2_1d_noncontiguous_model.json`；输出在 `Modeling/Ana/round3`。
Modeling/Ana 内 JSON 是历史实测快照，尚未按对齐后的数据更新；旧数据不代表完整的新评估集。
