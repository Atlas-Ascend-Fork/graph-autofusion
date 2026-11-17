# v35 Small Contiguous Store Model Addition

`StoreSmallPayload` applies to FP32, 1-2-dimensional descriptors with no LoopMode, no GM/UB row gap, an inner row
width aligned to 32 B, 1024-8192 B per core, and 8-56 active cores. When matched, it reuses `LoadStoreFunc`:

`cycle = 221.962839 + bytes / (23.856552 + 373.752654 / block_dim)`

The parameters come from the independent `small-tile-contiguous-store-v1` Store collection with NORMAL L2 and
rotating addresses. The per-call cost is estimated from the MTE3 cycle slope between 4 and 16 repetitions.
Training, holdout, and known candidates are stored separately. Disabled-L2 samples are controls, and the archive
computes predictions with the current product coefficients. Non-contiguous Store, other dtypes, LoopMode, and
out-of-domain descriptors keep the original model.

[Case list](cases.csv) | [Measurement and reproduction index](results/v35/pr2921-d68/report_en.md)

UT coverage is provided by `UTestAscirPerfV2.TestSmallDmaCalibratedDomains`,
`TestSmallDmaUnsupportedDescriptorsKeepLegacy`, and `TestStoreStrideLoopModeUsesDmaBlockCount`.

Device collection and reproduction are described in the
[DMA modeling collection project](../../../../tools/modeling/dma/README_en.md).
