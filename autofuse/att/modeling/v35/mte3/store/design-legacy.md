# A5 MTE3 Store 建模设计

归档说明：本文件保留 `cdc88a04` 的基础模型记录及原始实现行号。当前 `StoreSmallPayload` 增补见 [design.md](design.md)；原始实验用例见 [cases-legacy.csv](cases-legacy.csv)。

## 适用范围

本模型用于 A5 `StoreApi` 的 UB→GM 单次和多次搬运，活动公式覆盖归档用例中的 `S`、`CoreNum`、`ApiCount`、`BlockCount` 和 GM stride。UB 侧 stride 不增加 Store 项，物理 AIV 数以上的核数不外推。

## 数据来源与实验设计

主标定证据来自 `A5/MTE3/5_Summary/初版建模分析_20261021.md`，主机为 144、`Ascend950PR_9599`、CANN 9.2.0、dtype `uint8`。归档用例按 `aiv_total_cycles × aiv_mte3_ratio / BlockNum` 记录每核每次调用 cycle，`S=blocklen×blockcount`。

| 阶段 | 固定量 | 扫描量 | 用途 |
| --- | --- | --- | --- |
| `1_BlockDim` | 连续搬运 | `S`、`CoreNum`、BlockCount | 连续模型覆盖 |
| `2_ApiCount` | 短载荷和大载荷 | `ApiCount`、CoreNum | 多次调用覆盖 |
| `3_UnAlign` | 非 32B 对齐 shape | shape、CoreNum | 非对齐覆盖 |
| `4_GmStride` | `S=102400B`、uint8 | GM stride `0–4096` | GM 目的侧 stride 覆盖 |
| `5_UbStride` | `S=2048B` | UB stride `0–8` | 源侧覆盖，不生成 Store 参数 |

## 基础公式与活动参数

```text
single_mte3 = S / T + h
T = 11.774 + 10.265 / CoreNum
h = 160
overall_mte3 = single_mte3 * ApiCount + H
H = 571
```

活动模型按核数得到以下 `T`：

| CoreNum | T（B/cycle） |
| --- | --- |
| 1 | 22.039 |
| 16 | 12.4155625 |
| 32 | 12.09478125 |
| 48 | 11.987854166666665 |
| 64 | 11.934390625 |

## GM 侧非连续（dst）

归档用例 dtype 为 `uint8`，活动项为：

```text
stride_cost = 0.0049922306718019 * BlockCount * min(Stride, 4096)
```

实现的 dtype 参数表在 `../../../../../v35/att/api_perf_register/perf_param_v2.cpp@cdc88a04:1024`，计算逻辑在 `../../../../../att/gen_model_info/api_perf_register/utils/api_perf_utils.cpp@cdc88a04:291`。其它 dtype 先乘 `dtype_size`，再应用 `u=4096`。

## 小尺度、非对齐和 UB 侧

原归档版本的小尺度沿用 Store 连续公式，尚无独立 `StoreSmallPayload`；当前版本的增补适用域见 [design.md](design.md)。非对齐项由 `hl=0` 关闭。UB 侧 stride 不进入 Store 公式，`NddmaStride` 属于另一条路径，不改写 Store 参数。`ExpandMTE3BlockLen` 的 GM 非连续小 block 处理见 `../../../../../v35/att/api_perf_register/ascendc_api_perf/data_move_api_perf_v2.cpp@cdc88a04:404`。

## 跨芯片复核

156 和 230 的原始结果路径在 `cases-legacy.csv` 中保留为覆盖证据；原归档参数取本设计的 Store 表。230 仅将锁频 dev2、`CoreNum<=32` 作为规范基准，dev0 只作对照。

## 误差与适用范围

活动公式只适用于归档覆盖的尺寸、核数和 stride 上限，不外推到物理 AIV 数以上的超订阅场景、未覆盖 dtype 或 `u>4096` 的 stride。新增覆盖必须同步更新中文/英文设计文档和 `parameters.json`。

## 实现位置

- Pipe head：`../../../../../v35/att/api_perf_register/perf_param_v2.cpp@cdc88a04:20`
- Store 表：`../../../../../v35/att/api_perf_register/perf_param_v2.cpp@cdc88a04:1014`
- Store stride 表：`../../../../../v35/att/api_perf_register/perf_param_v2.cpp@cdc88a04:1024`
- Store 计算：`../../../../../v35/att/api_perf_register/ascendc_api_perf/data_move_api_perf_v2.cpp@cdc88a04:334`
