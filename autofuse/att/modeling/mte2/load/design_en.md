# v35 Load Model Additions

## Scope and Formulas

v35 FP32 Load selects an estimate from the actual codegen transfer descriptor. The branches below reuse
`LoadStoreFunc` and are expressed in cycles. Let B be the effective bytes of one per-core transfer and C the
active core count; the estimate is `h + B / (a + b/C)`. Widths, gaps, and strides in the table are in FP32
elements. Ordinary Load outside these branches keeps the original payload, stride, and cacheline path.

| Branch | Descriptor condition | h | a | b |
| --- | --- | ---: | ---: | ---: |
| SmallPayload | 1-2 dimensions, no GM/UB row gap, row width aligned to 32 B, 512-8192 B, 8-48 cores | 292.936412 | 20.724828 | 873.958753 |
| SmallPayloadCompact | 2-18 rows, row width 32-256 and 32 B aligned, GM row gap 64-4096, no UB row gap, 512-8192 B, 8-48 cores | 282.480993 | 20.736478 | 779.354414 |
| AlignedSingleRow | Single row of a 2-D descriptor, row width 512-30680 and 32 B aligned, 24-56 cores, the single row does not advance the stride | 253.290566 | 0 | 1310.182034 |
| AlignedDensePayload | 1-2 dimensions after removing static unit axes, no GM/UB row gap, row width aligned to 32 B, 1024-4096 B, 16-56 cores | 394.947854 | 41.831205 | 1134.576300 |
| AlignedWideGap | 16 or 32 rows, row width 384-768 and 32 B aligned, GM pitch 131072, no UB row gap, 40-56 cores | 0 | 0 | 987.983765 |
| WidePitch128B | 256-512 rows, row width 32, GM pitch a multiple of 4096 in 4096-16384, no UB row gap | 160 | 21.813945 | 6.4088 |
| AlignedLoopPayload | One hardware LoopMode layer, shape `[M,N,W]`, GM stride `[W,M*W,1]`, UB stride `[(N+p)*W,W,1]`, M=24-56, N=2-6, W=32-64 and aligned to the platform cache line, p=0 or 1, 16-56 cores | 294.009388 | 32.296047 | 252.871196 |
| NarrowLoopPayload | Same LoopMode physical descriptor, M=24-56, N=2-8, W=16 (64 B), p=0 or 1, 16-56 active cores; FP32 only | 531.6421388441569 | 18.86785800658518 | 0.7583841815917162 |

The same descriptor may match multiple conditions. The actual priority follows the expression override order in
`LoadPerf` and `LoadApi`. The LoopMode branches read the raw three-axis descriptor and preserve outer-axis address
progression. The estimate formulas reuse the existing tiling-expression interface; the codegen API, template
semantics, and device synchronization keep their existing implementations.

A 64 B row width does not match the original cacheline-aligned branch and falls back to the stride estimate. The
new branches use independent training and holdout coefficients and reuse the same formula and descriptor checks;
wide-row parameters are not extrapolated to narrow rows, and the NDDMA compensation for Load is not modified.
Other dtypes and out-of-domain descriptors keep the original path. The core-count range is the measured applicable
domain; the model is not selected by device model or total device core count.

## Cases and Validation

The [case list](cases.csv) keeps the original point IDs, data roles, device core counts, shapes, and strides. The
[measurement archive](results/v35/pr2921-d68/report_en.md) records the fitting source, collection semantics, and
reproduction entries. Disabled-L2 results for wide pitch are cold-read calibration; other modes are recorded
separately. LoopMode training and holdout are independent, and dense Pad Load validates both flattened and
multi-row equivalence. The model unit is per-core cycles of one API call; end-to-end graph and kernel gains are
accepted separately.

UT entries are `UTestAscirPerfV2.TestSmallDmaCalibratedDomains`, `TestAlignedSingleRowLoadEstimate`,
`TestAlignedLoopLoadDescriptorDomain`, `TestAlignedDenseLoadDomainAndDescriptorEquivalence`,
`TestAlignedWideGapLoadEstimate`, and `TestWidePitch128BLoadEstimate`.

The narrow-row UT is `UTestAscirPerfV2.TestNarrowLoopLoadDescriptorDomain`. Independent sampling, frozen parameters,
out-of-domain tail control, and reproduction indexes are in the
[narrow-row report](results/v35/pr2921-d68/report_en.md).

Device collection and reproduction are described in the
[DMA modeling collection project](../../../../tools/modeling/dma/README_en.md).
