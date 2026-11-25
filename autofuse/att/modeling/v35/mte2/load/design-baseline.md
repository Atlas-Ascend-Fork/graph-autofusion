# A5 MTE2 Load（GM2UB）模型设计归档

[English](design-baseline_en.md)

本文件保留 A5 基础模型的采集设计，采集清单见 [cases-baseline.csv](cases-baseline.csv)。后续 v35 标定分支及其适用域见 [增补设计](design.md)。

## 适用范围

面向 ATT `LoadApi` 的 GM → UB 单条 `DataCopy`，覆盖 CoreNum 1–64、每核每次 S 8B–256KB、连续、非对齐和 GM 侧非连续场景。原始 profiling 不入库，采集仓路径只作为证据索引。

## 数据来源与实验设计

采集根为 `mte2_mte3_perf_analysis:A5/MTE2/`。`run_*.sh` 用 `sed` 注入 corenum、blockcount、stride 等参数后重建运行，`msprof` 采集，`Ana_gen_info.py` 从 op_summary 抽取 `aiv_mte2_time/aiv_total_cycles/aiv_mte2_ratio/aiv_mte2_cycles`。A5 的 `aiv_mte2_cycles` 已按 ratio 折算到 MTE2 时间，数据时间跨度为 2025-08-12 ~ 2025-09-10。

| 阶段 | 变量 / 固定量 | 采集入口 | 数据 / 拟合 | 用途 |
| --- | --- | --- | --- | --- |
| 1_BlockCount | 核数 1/8/16/32/48/64；blockcount 1/20/40/60/80；shape 0–256KB，32B 对齐 | `mte2_mte3_perf_analysis:A5/MTE2/1_Script/1_BlockCount/run_count.sh` | `mte2_mte3_perf_analysis:A5/MTE2/2_Data/1_BlockCount/`、`mte2_mte3_perf_analysis:A5/MTE2/4_Analyse/1_BlockDim/` | 大尺度曲线 |
| 2_ApiCount | apicount 1–100（10KB）/1–1000（256KB）；核数 1–64/16/32/64 | `mte2_mte3_perf_analysis:A5/MTE2/1_Script/2_ApiCount/run_count.sh` | `mte2_mte3_perf_analysis:A5/MTE2/2_Data/2_ApiCount/`、`mte2_mte3_perf_analysis:A5/MTE2/4_Analyse/2_ApiCount/` | 多调用曲线 |
| 3_UnAlign | 核数 1/8/16/32/48/64；blockcount 20/40/60/80；shape 8B–256KB | `mte2_mte3_perf_analysis:A5/MTE2/1_Script/3_UnAlign/run_count.sh` | `mte2_mte3_perf_analysis:A5/MTE2/2_Data/3_UnAlign/`、`mte2_mte3_perf_analysis:A5/MTE2/4_Analyse/4_UnAlign/` | 非对齐覆盖 |
| 4_GmStride | shape、blockcount、srcStride 扫描 | `mte2_mte3_perf_analysis:A5/MTE2/1_Script/4_GmStride/run_stride.sh` | `mte2_mte3_perf_analysis:A5/MTE2/2_Data/4_UnStride/0.GM_Stride_Test/`、`1.GM_Stride/`、`mte2_mte3_perf_analysis:A5/MTE2/4_Analyse/5_Stride/` | GM 源侧非连续 |
| 5_UbStride | 核数 1/8/16/64；blockcount 2/4/16/32/64/128/256/512；dstStride 0–8 | `mte2_mte3_perf_analysis:A5/MTE2/1_Script/5_UbStride/run_stride.sh` | `mte2_mte3_perf_analysis:A5/MTE2/2_Data/4_UnStride/2.UB_Stride/`、`mte2_mte3_perf_analysis:A5/MTE2/4_Analyse/5_Stride/Results_UB_Stride/` | UB 目的侧非连续 |

`4_Analyse/3_BlockCount/Ana.py` 复用 1_BlockCount 数据，验证 blockcount 合并到 S 的处理方式。

## 基础模型与实验观察

统一骨架为：

```text
mte2_once  = S / T(CoreNum) + h
mte2_total = mte2_once * ApiCount + H
```

实验数据按大尺度和小尺度分组，连续维度合并为 S=blockcount*blocklen。大尺度曲线用于观察吞吐随 CoreNum 的变化，小尺度数据用于覆盖实现的 small block 分支；非对齐与 stride 数据用于覆盖 LoadPerf 的额外路径。

当前归档模型参数直接采用实现参数表：

| 条目 | 参数 |
| --- | --- |
| Load | `a=11.8292`、`b=6.6155`、`h=160`、`hl=0` |
| LoadSmallBlk | `a=13.1355`、`b=6.4088`、`h=160`、`hl=0` |
| LoadSingleCore | `a=23.0520`、`b=0`、`h=160`、`hl=0` |
| LoadStride | `k=0.005`、`u=4096.0`、`penalty_coeff=0.0` |
| pipe head | `1174.3`（blocklen ≥256B）或 `775.0`（blocklen <256B） |

`LoadStoreFunc` 的计算为 `D/(a+b/block_dim)+h+hl*(256-(h+512/t))`，其中 D 为 dims 乘积与 dtype 字节数的乘积。
`LoadStoreStrideV2Func` 的计算为 `0.005*block_count*min(gm_stride*data_type_size,4096.0)`。

## 分支、非对齐与 stride

`GetLoadCase` 按 codegen 尾轴 blocklen 选择 Load 或 LoadSmallBlk；`block_dim==1` 时选择对应的 SingleCore 条目。动态 blocklen 使用三元表达式。
GM 侧为 src 源侧，UB 侧为 dst 目的侧。GM stride 经过 `SetStride` 传入，stride 项按 block_count_idx 计算 block_count；非连续且 blocklen 小于 cache line 时，LoadPerf 会先扩展 blocklen。
当前 `hl=0`，非对齐惩罚项不产生额外周期；stride 上限由参数 `u=4096.0` 控制。

## 实现路径

- [ascir_api_perf_v2.cpp](../../../../../v35/att/api_perf_register/ascir_api_perf_v2.cpp)：LoadApi、MergeTensorContinuousDims、GetDmaPerf。
- [perf_param_v2.cpp](../../../../../v35/att/api_perf_register/perf_param_v2.cpp)：pipe head 和 Load 参数表。
- [api_perf_utils.cpp](../../../../../att/gen_model_info/api_perf_register/utils/api_perf_utils.cpp)：LoadStoreFunc、LoadStoreStrideV2Func、SetStride、GetDMAActualPerf。
- [data_move_api_perf_v2.cpp](../../../../../v35/att/api_perf_register/ascendc_api_perf/data_move_api_perf_v2.cpp)：LoadPerf 分支组装。

## 验证设计

| 测试类别 | 覆盖内容 | 方法 |
| --- | --- | --- |
| 结构 | JSON、CSV、V1/v35 目录导航和证据索引 | 解析器与脚本检查 |
| 参数 | Load、SmallBlk、SingleCore、Stride 参数及 pipe head | 与实现源码现场核对 |
| 分支 | blocklen、block_dim、GM stride、UB stride | 依据 cases-baseline.csv 覆盖入口检查 |
| 性能 | 实测数据与实现公式 | 使用采集仓原始证据复核 |

本文件记录模型设计、实验覆盖和实现参数，不修改实现源码。
