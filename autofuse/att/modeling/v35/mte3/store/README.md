# A5 MTE3 Store 归档

本目录按 A5 → v35 归档。当前小块连续 Store 增补见 [design.md](design.md)、[cases.csv](cases.csv) 和[验证结果](results/pr2921-d68/report.md)；下文保留 `cdc88a04` 中的原始标定口径，作为未命中增补适用域时的基础模型记录。

A5 MTE3 是 UB→GM 路径：UB 为源侧 `src`，GM 为目的侧 `dst`。本归档的活动参数与仓内 `Store`、`StoreStride` 实现表一致。

## 活动模型

- 单次调用：`S / (11.774 + 10.265 / CoreNum) + 160`。
- 多次调用：`单次调用 × ApiCount + 571`。
- 本归档用例为 `uint8`：`0.0049922306718019 × BlockCount × min(Stride,4096)`。
- 非对齐惩罚关闭：`hl=0`。
- 原归档的小尺度用例沿用 Store 公式；当前实现另有 `StoreSmallPayload` 分支，其适用域和参数见 [增补设计](design.md)。

## 五阶段覆盖

`1_BlockDim`、`2_ApiCount`、`3_UnAlign`、`4_GmStride`、`5_UbStride` 的入口和证据见 [cases-legacy.csv](cases-legacy.csv)。156 与 230 复核批次也保留在用例索引中；230 规范基准使用锁频 1650 MHz 的 dev2，物理核数上限为 32。

## 与实现对照

| 项目 | 归档活动值 | 实现 | 判定 |
| --- | --- | --- | --- |
| Pipe head | `H=571` | `kMte3PipeHeadCost=571.0f` | 一致 |
| 指令头 | `h=160` | `h=160` | 一致 |
| 带宽 | `T=10.265/N+11.774` | `a=11.774,b=10.265` | 一致 |
| GM stride | `uint8 k=0.0049922306718019,u=4096` | `StoreStride` | 一致 |
| 非对齐 | `hl=0` | `hl=0` | 一致 |
| BlockCount | 进入 `S` 和 stride 项 | `StorePerf` 与 stride helper | 一致 |

## 待确认项

1. 补充小尺度和非对齐覆盖用例。
2. 保持 `H=731-160` 的残差定义，直到有独立测量。
3. 重复 UB stride 覆盖测试，不新增 Store 参数。
4. 继续以锁频 dev2、`CoreNum<=32` 作为 230 基准。

## 原始证据索引

| 内容 | 外部证据 |
| --- | --- |
| A5 主总结 | `A5/MTE3/5_Summary/初版建模分析_20261021.md` |
| 连续拟合 | `A5/MTE3/4_Analyse/1_BlockDim/combine/256KB/fit_parameters.csv` |
| GM stride 拟合 | `A5/MTE3/4_Analyse/5_Stride/GM_stride/GMToUB_fit_results.csv` |
| 156 复核 | `results/156_ascend950/20261021_min/report.md` |
| 230 复核 | `results/230_ascend350_dev2/20261105_max32/report.md` |

原始设计、用例和结构化参数见 [design-legacy.md](design-legacy.md)、[cases-legacy.csv](cases-legacy.csv)、[parameters.json](parameters.json)。A3 归档见 [V1 Store](../../../v1/mte3/store/README.md)。
