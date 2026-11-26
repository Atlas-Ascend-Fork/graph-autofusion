# A3 MTE3 Store 建模设计

归档说明：按 A3 → V1 迁移目录；下文仅保留 `cdc88a04` 的 v35 历史参数快照，不代表 V1 参数。采集时使用的实现版本未记录，不能认定 A3 数据使用 v35 采集。V1 参数差异见 [README.md](README.md)，V1 标定和实测验证待补充。

## 适用范围

本归档索引 A3 的 UB→GM Store 用例，并保留原记录所引用的 v35 参数表。快照描述 `S=512B–192KB` 的连续搬运、单次或多次调用和 GM 侧非连续搬运公式，不表示这些公式适用于 A3/V1。连续数据覆盖单核到 48 核；缺失核和未覆盖尺寸不外推。最小已归档尺寸为 512B；32B–512B 小尺度数据未归档。

## 数据来源

连续数据在 `A3/mte3_ub2gm_continuous/`，文件 `data_GMToUB_check_{1..48}.csv` 共 46 个核（缺 9、43），shape 网格为 `512B–192KB`。headcost 文件为 `A3/mte3_ub2gm_continuous/mte3_headcost_test.csv`，非连续数据在 `A3/mte3_ub2gm_uncontinuous/`。这些路径记录实验覆盖，不作为当前实现的独立参数来源；A3 CSV 没有 `aiv_mte3_ratio` 列。

## 连续搬运

v35 快照公式为：

```text
single_mte3 = S / (11.774 + 10.265 / CoreNum) + 160
overall_mte3 = single_mte3 * ApiCount + 571
```

`h=160`、`H=571`、`T=11.774+10.265/CoreNum` 来自 `../../../../../v35/att/api_perf_register/perf_param_v2.cpp@cdc88a04:1014` 和 `../../../../../v35/att/api_perf_register/perf_param_v2.cpp@cdc88a04:20`。这些值仅与归档版本的 v35 实现对照，不是 A3 数据拟合所得的 V1 参数。

## GM 侧非连续（dst）

本目录用例 dtype 为 `uint8`，v35 快照公式为：

```text
stride_cost = 0.0049922306718019 * BlockCount * min(Stride, 4096)
```

归档版本的 stride 表在 `../../../../../v35/att/api_perf_register/perf_param_v2.cpp@cdc88a04:1024`，计算逻辑在 `../../../../../att/gen_model_info/api_perf_register/utils/api_perf_utils.cpp@cdc88a04:291`。其它 dtype 必须先按该版本实现的 `data_type_size` 缩放 stride。

## 小尺度、headcost 与 src 侧 stride

A3 原始 headcost 和 src 侧非连续数据只用于覆盖记录。现存小尺度记录仅包含 512B 点，未归档 32B–512B 小尺度数据。`cdc88a04` v35 历史快照没有独立 `StoreSmallPayload` 表项：其中小尺度沿用 Store 连续公式，pipe 固定项为 `571`，UB stride 不增加 Store 项，`hl=0`。当前 v35 实现后续增加的 `StoreSmallPayload` 分支不属于该历史快照；以上均不构成 V1 的分支或参数说明。

## 与 cdc88a04 的 v35 实现对照

| 项目 | v35 快照值 | v35 归档版本实现 | 判定 |
| --- | --- | --- | --- |
| `T` | `11.774+10.265/CoreNum` | `Store` 的 `a=11.774,b=10.265` | 一致 |
| `h` | `160` | `h=160` | 一致 |
| `H` | `571` | `kMte3PipeHeadCost=571.0f` | 一致 |
| GM stride | uint8 `0.0049922306718019`、上限 `4096` | `StoreStride` | 一致 |
| UB stride | 无 Store 项 | Store 公式不读取 UB stride | 一致 |
| 非对齐 | `hl=0` | `hl=0` | 一致 |

## 适用范围与待确认项

- 覆盖索引仅包含已列出的 shape 和核数；缺失的 9、43 核不补写参数。
- 非连续、headcost 和已归档的 512B 点可继续用于覆盖索引；32B–512B 数据未归档。确认采集实现并完成 V1 标定后，才能声明这些 A3 数据对应的模型参数。
- A3 pad 路径不在本归档快照中，若需要支持必须单独设计用例。
- 归档版本的动态 GM stride block length padding 见 `ExpandMTE3BlockLen`，位置为 `../../../../../v35/att/api_perf_register/ascendc_api_perf/data_move_api_perf_v2.cpp@cdc88a04:404`。
