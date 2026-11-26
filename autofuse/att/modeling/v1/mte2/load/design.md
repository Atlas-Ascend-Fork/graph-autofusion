# A3 MTE2 Load（GM2UB）模型设计归档

[English](design_en.md)

## 适用范围

面向 ATT `LoadApi` 的 GM → UB 单条 `DataCopy`。本目录按 V1 家族归档 A3 采集证据；CoreNum、shape、连续和非连续覆盖以 cases.csv 为准。下文保留原归档引用的 v35 参数快照，不代表当前 V1 参数；V1 实现见 [perf_param_v1.cpp](../../../../gen_model_info/api_perf_register/v1/perf_param_v1.cpp)。

## 数据来源与实验设计

A3 证据根为 `mte2_mte3_perf_analysis:A3/`，结论和数据索引来自两个实验目录及两个 notebook。A3 CSV 提供 `shape`、`idx`、`aiv_mte2_time(us)`、`aiv_mte2_cycles` 和全核合计带宽列；观测列为 `aiv_mte2_cycles`，CSV 未提供 `aiv_mte2_ratio`。

| 实验 | 变量 / 固定量 | 入口 | 数据 | 分析 |
| --- | --- | --- | --- | --- |
| 连续搬运 | 核数 1–48；shape 32B–512B、512B–100KB、100KB–192KB | `mte2_mte3_perf_analysis:A3/mte2_gm2ub_continuous/run.sh` | `mte2_mte3_perf_analysis:A3/mte2_gm2ub_continuous/profiling_data/` | `mte2_mte3_perf_analysis:A3/scripts/mte2_analysis.ipynb` |
| 多次调用 | shape 196608B；tilenum 8–1000；单核 | 同上 | `mte2_mte3_perf_analysis:A3/mte2_gm2ub_continuous/profiling_data/mte2_headcost_test.csv` | `mte2_mte3_perf_analysis:A3/scripts/mte2_analysis.ipynb` |
| 非连续 src | 单核；shape 196608B；blocklen 8–768B；srcstride 1–40 | `mte2_mte3_perf_analysis:A3/mte2_gm2ub_uncontinuous/run.sh`、`run_prof.sh` | `mte2_mte3_perf_analysis:A3/mte2_gm2ub_uncontinuous/profiling_data/` | `mte2_mte3_perf_analysis:A3/scripts/mte2_uncontinuous_analysis.ipynb` |

## 模型公式

原归档引用的 v35 实现参数如下：

| 条目 | 参数 |
| --- | --- |
| Load | `a=11.8292`、`b=6.6155`、`h=160`、`hl=0` |
| LoadSmallBlk | `a=13.1355`、`b=6.4088`、`h=160`、`hl=0` |
| LoadSingleCore | `a=23.0520`、`b=0`、`h=160`、`hl=0` |
| LoadStride | `k=0.005`、`u=4096.0`、`penalty_coeff=0.0` |
| pipe head | `1174.3`（blocklen ≥256B）或 `775.0`（blocklen <256B） |

设 `D=product(dims)*data_type_size`、`N=block_dim`、`t=a+b/N`：

```text
base = D / t + h
unalign_penalty = hl * (256 - (h + 512 / t))
stride_cost = k * block_count * min(gm_stride * data_type_size, u)
load_cost = base + unalign_penalty + stride_cost
```

`block_dim==1` 使用 SingleCore 条目；其他情况使用 Load 或 LoadSmallBlk。`GetLoadCase` 按 codegen 尾轴 blocklen 选择大块和小块表达式，动态值使用三元表达式。

## 实现路径

- [ascir_api_perf_v2.cpp](../../../../../v35/att/api_perf_register/ascir_api_perf_v2.cpp)：LoadApi 入口。
- [perf_param_v2.cpp](../../../../../v35/att/api_perf_register/perf_param_v2.cpp)：pipe head 和参数表。
- [api_perf_utils.cpp](../../../../../att/gen_model_info/api_perf_register/utils/api_perf_utils.cpp)：LoadStoreFunc、LoadStoreStrideV2Func 和 DMA 参数计算。
- [data_move_api_perf_v2.cpp](../../../../../v35/att/api_perf_register/ascendc_api_perf/data_move_api_perf_v2.cpp)：normal、small block、stride、single core 分支组装。

## 验证设计

连续数据覆盖大、小 shape 与 CoreNum 1–48；headcost 覆盖多次调用；srcstride 覆盖 GM 源侧非连续路径。结构验证检查 parameters.json 的实现参数、cases.csv 的 10 列格式、采集证据索引和 Markdown 链接。

A3 的 `mte2_pad` 实验不属于本次 load 归档范围。本文只描述归档模型和证据入口，不修改实现源码。
