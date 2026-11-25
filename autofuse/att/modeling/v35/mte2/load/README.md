# v35 MTE2 Load（GM2UB）性能模型归档

[English](README_en.md)

`load` 对应 ATT 的 `LoadApi`：GM → UB 单条 `DataCopy`，使用 MTE2 管线。本目录归档 A5 采集证据和 v35 模型；A3 采集证据见 [V1 Load 归档](../../../v1/mte2/load/README.md)。

## 目录

```text
modeling/v35/mte2/load/
├── README.md
├── README_en.md
├── design-baseline.md / design-baseline_en.md  # A5 基础模型采集设计
├── cases-baseline.csv                         # A5 采集阶段与证据索引
├── parameters.json                            # 基础实现参数
├── design.md / design_en.md                   # 后续标定分支与适用域
├── cases.csv                                 # 标定分支可执行描述符
└── results/                                  # 标定与验证证据
```

## 统一口径

| 符号 | 含义 | 单位 |
| --- | --- | --- |
| S | 每核每次搬入字节数；连续维度合并后为 dims 乘积乘 dtype 字节数 | Byte |
| T | 实现中的吞吐表达式 `a + b / block_dim` | Byte/cycle |
| h | `LoadStoreFunc` 的指令头参数 | cycle |
| H | `GetMTE2PipeHead` 返回的 pipe 头参数 | cycle |
| CoreNum | `block_dim`，参与搬运的核数 | 个 |
| BlockLen | DataCopy 尾轴的块长 | Byte |
| stride | GM 侧 `gm_stride`，按 dtype 字节数换算并受上限限制 | 计数单位 |

GM = src = 源侧，UB = dst = 目的侧。实现的计算结果为单次 DMA 表达式与 stride 表达式之和；外层调用次数由 DMA 调度路径处理。

## 基础实现模型

本节记录基础参数。域内估值还包含 [增补设计](design.md) 中的标定分支，实际分支顺序以当前实现为准。A5 BlockCount、ApiCount、UnAlign、GmStride 和 UbStride 的采集覆盖见 [基础模型设计](design-baseline.md) 及 [采集清单](cases-baseline.csv)，参数见 [parameters.json](parameters.json)。

```text
t = a + b / block_dim
base = (product(dims) * data_type_size) / t + h
penalty = hl * (256 - (h + 512 / t))
stride_cost = k * block_count * min(gm_stride * data_type_size, u)
load_cost = base + penalty + stride_cost
```

当前参数为：大块 `a=11.8292,b=6.6155,h=160,hl=0`；小块 `a=13.1355,b=6.4088,h=160,hl=0`；单核使用 `a=23.0520,b=0,h=160,hl=0`；stride 使用 `k=0.005,u=4096.0`。pipe 头按最大 blocklen 分支为 `1174.3`（`>=256B`）或 `775.0`（`<256B`）。LoadPerf 按 blocklen `>256B` 选择大块表达式，否则选择小块表达式，并在 `block_dim==1` 时选择单核参数表。

## 实现位置

| 内容 | 实现位置 |
| --- | --- |
| LoadApi 注册和分支 | [ascir_api_perf_v2.cpp](../../../../../v35/att/api_perf_register/ascir_api_perf_v2.cpp) |
| pipe 头和参数表 | [perf_param_v2.cpp](../../../../../v35/att/api_perf_register/perf_param_v2.cpp) |
| LoadStoreFunc / LoadStoreStrideV2Func | [api_perf_utils.cpp](../../../../../att/gen_model_info/api_perf_register/utils/api_perf_utils.cpp) |
| LoadPerf 组装 | [data_move_api_perf_v2.cpp](../../../../../v35/att/api_perf_register/ascendc_api_perf/data_move_api_perf_v2.cpp) |

LoadApi 的路径为 `MergeTensorContinuousDims → GetDmaPerf → SetStride/GetDMAActualPerf`。V1 和 v35 分别使用各自实现参数表，目录按模型家族组织。

## 证据索引

原始 profiling、`PROF_*`、`op_summary_*.csv`、`*.db` 和二进制不入库，统一使用 `mte2_mte3_perf_analysis:` 前缀索引采集仓路径。

| 平台 | 采集与分析索引 |
| --- | --- |
| A5 | `mte2_mte3_perf_analysis:A5/MTE2/1_Script/`、`mte2_mte3_perf_analysis:A5/MTE2/2_Data/`、`mte2_mte3_perf_analysis:A5/MTE2/4_Analyse/`、`mte2_mte3_perf_analysis:A5/MTE2/5_Summary/` |
| A3 | `mte2_mte3_perf_analysis:A3/mte2_gm2ub_continuous/`、`mte2_mte3_perf_analysis:A3/mte2_gm2ub_uncontinuous/`、`mte2_mte3_perf_analysis:A3/scripts/` |

公共归档约定见 [模型归档说明](../../README.md)。
