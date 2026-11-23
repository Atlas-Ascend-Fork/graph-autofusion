# v35 小块连续 Store 模型增补

`StoreSmallPayload` 适用于 FP32、1–2 维、无 LoopMode、GM 与 UB 无行间隙、内行宽按 32 B 对齐、每核 1024–8192 B、8–56 活跃核的描述符。命中时复用 `LoadStoreFunc`：

`cycle = 221.962839 + bytes / (23.856552 + 373.752654 / block_dim)`

参数来源为 `small-tile-contiguous-store-v1` 的 NORMAL L2、轮换地址的独立 Store 采集。用 4/16 次重复间的 MTE3 cycle 斜率估计单次成本。训练、留出和已知候选分别保存。禁用 L2 的样本用于对照，归档按当前产品系数计算预测。非连续 Store、其他类型、LoopMode 与适用域之外的描述符沿用原模型。

[用例清单](cases.csv) · [实测与复现索引](results/pr2921-d68/report.md)

UT 覆盖见 `UTestAscirPerfV2.TestSmallDmaCalibratedDomains`、`TestSmallDmaUnsupportedDescriptorsKeepLegacy` 和 `TestStoreStrideLoopModeUsesDmaBlockCount`。

上板采集与复现见 [DMA 建模采集工程](../../../../../tools/modeling/v35/dma/README.md)。
