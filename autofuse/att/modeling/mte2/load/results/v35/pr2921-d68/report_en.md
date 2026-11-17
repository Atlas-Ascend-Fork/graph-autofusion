# PR 2921 Modeling Measurement Index

[Case list](../../../cases.csv) | [Point-level data](samples.csv) | [Fitting revalidation tool](../../../../../../../tools/modeling/mte2/README_en.md)

Row-level roles follow the original collection. A dash means the point was not compared with this fitted formula.
The errors below are absolute relative errors under the corresponding collection method. Product defaults and
end-to-end gains must be accepted separately.

| Branch | Source | Role | Points | Points with prediction | Max error |
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
| LoadAlignedWideGap | S05 | boundary_control | 2 | 0 | - |
| LoadAlignedWideGap | S05 | known_candidate_not_for_fit | 5 | 3 | 2.18% |
| LoadAlignedWideGap | S05 | independent_holdout | 6 | 6 | 0.96% |
| LoadWidePitch128B | S06 | train | 7 | 4 | 15.16% |
| LoadWidePitch128B | S06 | holdout | 3 | 2 | 7.22% |
| LoadWidePitch128B | S06 | known | 2 | 1 | 9.45% |

## S07: 64 B Row-Width LoopMode Load Calibration (follow-up to 4aea)

This section is a later independent calibration with a separately frozen collection protocol and parameters; the
historical directory label d68 is not the source version of S07. The parent commit is
`4aea615961c6fdda6dfab7096b37fd1dc2112591`.

Source S07 is an independent FP32 physically dense outer-permutation Load microbenchmark. The formula remains
`h + B/(a+b/C)`, and the parameters were frozen before the holdout collection. It is not fitted from total kernel
time, does not subtract the ATT head, and does not adjust other pipeline estimates.

| Role | Independent descriptors | Strategy/order observations | Max relative error |
| --- | ---: | ---: | ---: |
| train | 20 | 60 | 23.07% |
| holdout | 8 | 24 | 27.84% |
| known | 3 | 9 | 19.96% |
| known_tail_control (out of domain) | 2 | 6 | 15.55% (formula diagnosis only; product falls back) |

The three rounds are N-forward, C-forward, and N-reverse. Every API, core, and launch in N uses a different GM
address; C additionally disables the L2 hint. Each group has 16 warmup and 16 active calls. The table uses the
median per-core MTE2 busy cycles at K=1; K=2/4/10 are stored separately and not mixed with K=1. Accuracy is checked
byte by byte; K1 passes IQR <= 20%, intra-group drift <= 10%, and forward/reverse drift <= 15%. Training M excludes
the known production descriptors 38/50.

The measurement is `aiv_total_cycles * aiv_mte2_ratio / BlockNum`, not the whole-kernel scheduling latency. Known
points are not training points, and uniform tail-block control cannot replace a real mixed tail. Ordering, PGO-off
defaults, engineering, and full regression must be validated separately; meeting the errors in this report does not
imply that all of them pass.

Re-collection entries are in the [DMA collection project](../../../../../../../tools/modeling/dma/README_en.md).
Historical script IDs are `matrix17-fp32-narrow-loop-training-v1/run.sh` and
`matrix17-fp32-narrow-loop-holdout-v1/run.sh`. The external evidence root is provided by the user, and the relative
evidence path is `20261113-pr2921-active-penalty-ratio/narrow-loop-load`. The in-repo cases and data are sufficient
to verify the errors in this table offline; the external raw profiles are used to audit the collection protocol.

Collector SHA256: `0babc2bb3a24ae027a3d3bac0b8e105aa32f1493b0250c12fe9d5a5e172d4bf1`; `datacopy.h` SHA256 for both
production and collector: `490fb50993bd2dbdb9f40767211922f5af8157f6082ebcd6de798c7453d11a04`. Source SHA256:
`89312e1f5f649c4f606e4ffa3621c379b302014025cb6cd17c25452bce0567da`.
