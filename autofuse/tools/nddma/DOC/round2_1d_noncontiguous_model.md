<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Round2 一维非连续建模（对齐 HW_GE_ATT Round4）

[English](round2_1d_noncontiguous_model_en.md)

本轮继承 Round1 连续 Base，仅拟合一维非连续 N_G、N_GU 和多核 rho。
实现对齐 HW_GE_ATT Round4 的最终一维模型，不包含其历史候选模型/特征搜索诊断。
拟合需要 Python 与 NumPy，采用与参考实现相同的最小二乘求解方式。

## 数据集

默认 factor 为 `Modeling/Ana/round2/round2_1d_noncontiguous_factor.csv`。
安全筛选为 GM span（含 64 元素 guard）不超过 4 MiB，UB span（含 guard）不超过 256 KiB。
四种 dtype 为 int8_t/int16_t/int32_t/int64_t；kernel repeat 默认 200，外部重复默认 3。

| 组 | 配置数 | 用途 |
| --- | ---: | --- |
| A | 480 | 单核 GM 非连续，低字节 stride 拟合 N_G |
| B | 350 | 单核 UB 非连续验证，不参与最终 N_GU 参数估计 |
| C | 4075 | 单核联合非连续，os=2 子集拟合 N_GU |
| F | 440 | 多核连续/输出非连续参考验证 |
| G | 840 | 多核 GM 非连续校准 |
| H | 840 | 多核联合非连续校准 |
| I | 120 | 独立核数上的输出非连续验证 |
| J | 1 | int32_t、55 核、每核 9656 元素、is=3/os=1 验证 |
| 合计 | 7146 | 配置数量不包含重复执行次数 |

单核输出维度：`8,16,32,64,128,512,1024,2048,4196,10001,20001,30001,42302`。
A 输入 stride：`2,4,8,16,32,64,128,256,512,1024,2048,4096`。
C 输入 stride 额外包含 `3`；B/C 输出 stride 为 `2,4,6,8,10,12,14,16,32`。

G/H 核数为 `1,2,4,8,15,32,56`，每核字节数为 `64,128,512,1024,4096,16384`。
两组输入 stride 均为 `2,8,32,128,256`；G 的 os=1，H 的 os=2。
F 核数为 `1,2,4,8,16,32,56,6,24,40`，stride 对为 `(1,1),(1,2),(1,16)`。
I 核数为 `6,24,40`，stride 对为 `(2,4),(32,16),(128,32)`。
F/I 每核字节数均为 `64,512,4096,16384`。

## 公式

```text
B = bytes_per_core = logical_total_bytes / block_dim
s = min(input_stride*dtype_size, 128)
g = min(1, output_stride-1)
N_base = B/T_1+H_1 (k<=2), B/T_2+H_2 (k>2)
N_G = (a1+a2*B)*s
N_GU = ((b1+b2*s)+(b3+b4*s)*B)*g
rho = (c1+c2*s)+g*(c3+c4*s)
N_1 = N_base+N_G+N_GU (k<=2)
N_1 = N_base+rho*(N_G+N_GU) (k>2)
```

## 拟合流程

重复测量先按 config_id/token 取中位数；原始 cycles 除以 kernel repeat，actual_y 视为已归一化。

1. 冻结 Round1 Base，不使用 F 重拟合。
2. N_G：仅用单核 A、B<50000、input_stride*dtype_size<128。
   每个固定 B 对 `actual-base` 与字节 stride 做过原点回归，得到 g_prime；
   再对 `g_prime=a1+a2*B` 做回归。
3. N_GU：仅用单核 C、os=2、B<50000。
   扣除 Base 和 N_G；对每个固定 input_stride 将残差按 B 取中位数，拟合 `q0+q1*B`。
   再仅用字节 stride<128 的 q0/q1 节点，分别拟合其随字节 stride 的直线，得到 b1..b4。
   高 stride 节点仍计算一级回归以检查数据完整性，但不进入二级参数估计。
4. rho：冻结单核参数，仅用 G/H、k>2、input_stride*dtype_size<=128。
   以 `[delta,s*delta,g*delta,g*s*delta]` 回归 `actual-base`，delta=N_G+N_GU。
   此步骤不施加单核的 B<50000 筛选。

单核回归使用列缩放及 `numpy.linalg.lstsq(rcond=1e-12)`；rho 使用未缩放矩阵及 `rcond=None`。
所有安全配置均用于预测评估；超出拟合范围的点保留为验证点。

## 运行

以下命令在 `autofuse/tools/nddma/` 下执行；采集前设置匹配实际设备的 `NPU_ARCH`，或传入 `--npu-arch`。

```bash
python3 Modeling/Data_collection/round2/scripts/e2e.py all --msprof-bin "$(which msprof)"
python3 Modeling/Data_collection/round2/scripts/e2e.py fit --measurement-csv /path/to/measurements.csv
```

拟合器默认读取 `Modeling/Ana/round1/round1_1d_single_multi_core_model.json`。
输出模型、逐点预测和图位于 `Modeling/Ana/round2`；JSON 的 fit_scope 记录拟合条件。

仓库 Modeling/Ana 下已保存的模型 JSON 是旧实测快照，尚未按新流程重拟合。
旧版 7221 配置数据缺少新网格中的点，不应直接当作完整新训练集。
应先获取对齐网格的测量数据，重拟合 Round1、Round2，然后更新依赖它们的 Round3/4，
验证后单点预测工具直接读取 Ana 下模型。

## 对照验证

```bash
HW_NDDMA_MODELING=/path/to/HW_GE_ATT/NDDMA/Modeling \
  python3 -m unittest discover -s Modeling/tests -v
```

测试逐项比较完整采集配置（包含重复计数），并使用同一批含扰动数据比较
N_G/N_GU/rho 系数；额外扰动验证点以确认其不影响参数。
