# A3 MTE3 Store 归档

本归档按 A3 → V1 迁入当前目录。下文公式和参数是提交 `cdc88a04` 的 v35 Store 历史快照，不代表 V1 参数。现有 A3 数据索引未记录采集时的实现版本，不能据此认定使用了 v35 或完成了 V1 实测验证。

原始连续、headcost 和非连续数据集仅用于覆盖范围与证据索引。V1 的 `Store` 表为 `h=12.09,a=9.96,b=3.79`，`StoreStride k=0.02`，MTE3 head 为 `497.36`，与下列 v35 快照不同；本归档不将快照应用于 A3/V1。

## v35 历史快照口径

| 项目 | v35 快照模型 |
| --- | --- |
| 单次连续 | `S / (11.774 + 10.265 / CoreNum) + 160` |
| 多次调用 | `单次 × ApiCount + 571` |
| GM 侧 stride | `0.0049922306718019 × BlockCount × min(Stride,4096)`（uint8） |
| 非对齐 | `hl=0`，不增加惩罚 |

## 数据覆盖

连续数据覆盖 46 个核（缺 9、43）×384 个 shape，`512B–192KB`；headcost 数据覆盖 `tilenum=2–642`；非连续数据覆盖单核 stride `1–30`。这些路径用于复核用例覆盖，不证明 v35 快照适用于 A3。

## 结论

- 快照来自 `perf_param_v2.cpp@cdc88a04` 的 `Store` 和 `StoreStride` 表，不是 V1 参数表。
- v35 快照为 `h=160`、`H=571` 和 `T=11.774+10.265/CoreNum`。
- v35 快照中的 `StoreStride` 使用 dtype 独立系数；`uint8` 系数为 `0.0049922306718019`。
- 小尺度、非对齐和原始非连续实验保留为覆盖证据；采集实现版本和 V1 参数验证仍待补充。

## 证据

- notebook：`A3/scripts/mte3_analysis.ipynb`
- 连续数据：`A3/mte3_ub2gm_continuous/`
- headcost：`A3/mte3_ub2gm_continuous/mte3_headcost_test.csv`
- 非连续：`A3/mte3_ub2gm_uncontinuous/`

详细字段见 [design.md](design.md)、[cases.csv](cases.csv)、[parameters.json](parameters.json)。
