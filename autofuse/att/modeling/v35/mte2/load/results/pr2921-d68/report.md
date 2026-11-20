# PR 2921 建模实测索引

[用例清单](../../cases.csv) · [逐点数据](samples.csv) · [拟合复验工具](../../../../../../../tools/modeling/v35/mte2/README.md)

行级角色沿用原采集。无预测值表示该点未按此拟合式比较。以下误差为对应采集方法下的绝对相对误差，产品默认和全图收益需另行验收。

| 分支 | 来源 | 角色 | 点数 | 有预测的点数 | 最大误差 |
| --- | --- | --- | ---: | ---: | ---: |
| LoadSmallPayload | S01 | train | 54 | 54 | 25.63% |
| LoadSmallPayloadCompact | S01 | train | 126 | 126 | 26.90% |
| LoadSmallPayload | S01 | holdout | 24 | 24 | 24.22% |
| LoadSmallPayloadCompact | S01 | holdout | 56 | 56 | 26.39% |
| LoadSmallPayload | S01 | known | 10 | 10 | 15.74% |
| LoadSmallPayloadCompact | S01 | known | 10 | 10 | 17.82% |
| LoadAlignedLoopPayload | S02 | holdout | 96 | 96 | 31.87% |
| LoadAlignedDensePayload | S03 | frozen_fit_validation | 192 | 192 | 46.44% |
| LoadAlignedDensePayload | S03 | known_candidate_full_block_only | 24 | 24 | 38.81% |
| LoadAlignedSingleRow | S04 | train | 8 | 8 | 20.24% |
| LoadAlignedSingleRow | S04 | holdout | 4 | 4 | 19.72% |
| LoadAlignedWideGap | S05 | control | 7 | 1 | 0.11% |
| LoadAlignedWideGap | S05 | train | 5 | 5 | 0.96% |
| LoadAlignedWideGap | S05 | holdout | 7 | 2 | 0.84% |
| LoadAlignedWideGap | S05 | boundary_control | 2 | 0 | — |
| LoadAlignedWideGap | S05 | known_candidate_not_for_fit | 5 | 3 | 2.18% |
| LoadAlignedWideGap | S05 | independent_holdout | 6 | 6 | 0.96% |
| LoadWidePitch128B | S06 | train | 7 | 4 | 15.16% |
| LoadWidePitch128B | S06 | holdout | 3 | 2 | 7.22% |
| LoadWidePitch128B | S06 | known | 2 | 1 | 9.45% |

## S07：64 B 行宽 LoopMode Load 校准（4aea 后续增补）

本节是后续独立校准，采集协议与参数单独冻结；不将目录历史标签 d68 当作 S07 的源码版本。父提交为 `4aea615961c6fdda6dfab7096b37fd1dc2112591`。

来源 S07：独立的 FP32 物理稠密外轴排列 Load 微基准。公式保持 `h + B/(a+b/C)`，参数在留出采集前冻结；不从 kernel 总耗时拟合，不扣 ATT head，也不调整其他流水线估值。

| 角色 | 独立描述符数 | 策略/顺序观测数 | 最大相对误差 |
| --- | ---: | ---: | ---: |
| train | 20 | 60 | 23.07% |
| holdout | 8 | 24 | 27.84% |
| known | 3 | 9 | 19.96% |
| known_tail_control（域外） | 2 | 6 | 15.55%（仅公式诊断，产品回退） |

三轮为 N-forward、C-forward、N-reverse。N 的每次 API、核及 launch 均使用不同 GM 地址；C 另外设置禁用 L2 hint。每组 16 warmup、16 active。表中使用 K=1 的每核 MTE2 busy cycle 中位数；K=2/4/10 单独保存，不与 K=1 混合。精度逐字节检查；K1 的 IQR≤20%、组内漂移≤10%、正逆序漂移≤15% 全部通过。训练 M 不含已知生产描述符的 38/50。

测量为 `aiv_total_cycles * aiv_mte2_ratio / BlockNum`，不等于整 kernel 调度延迟。已知点不是训练点，统一尾块控制也不能替代实际混合尾核。排序、PGO-off 默认及工程/全量回归必须另验，不能由本报告的误差达标推出已全部通过。

复采入口见 [DMA 采集工程](../../../../../../../tools/modeling/v35/dma/README.md)。
历史脚本 ID 为 `matrix17-fp32-narrow-loop-training-v1/run.sh` 和
`matrix17-fp32-narrow-loop-holdout-v1/run.sh`。外部证据根目录由使用者提供，
证据相对路径为 `20261113-pr2921-active-penalty-ratio/narrow-loop-load`。
仓内用例与数据足以离线核验本表误差，外部原始 profile 用于复核采集协议。

采集器 SHA256：`0babc2bb3a24ae027a3d3bac0b8e105aa32f1493b0250c12fe9d5a5e172d4bf1`；生产与采集器的 `datacopy.h` SHA256：`490fb50993bd2dbdb9f40767211922f5af8157f6082ebcd6de798c7453d11a04`。源码 SHA256：`89312e1f5f649c4f606e4ffa3621c379b302014025cb6cd17c25452bce0567da`。
