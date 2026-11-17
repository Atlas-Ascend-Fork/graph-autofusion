# v35 NDDMA Model Additions

ATT first checks the codegen descriptor and dtype size, then normalizes the physical axes. Continuous GM/UB axes
can be folded into a physical 1-D form, and the existing 1-D calibrated parameters are selected for 1/2/4/8-byte
types. Non-continuous effective axes are kept and estimated by the formula for the corresponding layout. Dynamic
conditions select between the original and fitted expressions; low-core descriptors and descriptors that cannot be
proven applicable keep their original estimate paths.

| Branch | Estimate and applicable domain |
| --- | --- |
| Packed NG2 | 1/2/4-byte types, high core count, inner GM unit stride, and UB pitch equal to inner. Small and large inner are fitted separately; the split uses the inner-byte threshold in code. A symbolic pitch selects it through an equality condition. |
| Padded FP32 NG2 | M=12-32, N=193-519, GM stride `[1,M]`, UB pitch padded from N to 8 elements, 16-56 cores. Estimate: `162.529402 + B/65536 * (6594.471272 + 565.933501 * max(C-32,0)/24)`. |
| Small INT64 NG2 | M=24-64, N=2-8, GM stride `[1,M]`, UB pitch N or N+1, 16-56 cores; startup-dominated at about 506 cycles. |
| Packed rank 3 | 4-byte types, contiguous batch-to-batch pitch, each batch swaps the two in-plane axes between GM and UB, actual outer extent > 1, 3-56 cores. Estimate: `177.398972 + 76.277101 * B/1024`. |
| Sparse rank 3 | 4-byte types, `[W,32,2]`, W=192-256, GM pitch 65536 or 131072 elements, UB packed, 40-56 cores. Keeps the bandwidth and startup terms and scales the old stride residual by 0.017212331646550515. |
| Sparse rank 4 | 4-byte types, W=512/1024, GM pitch 65536 elements, UB pitch aligned to the production descriptor. The two tile axes are bounded by `tile1<=3`, `tile2<=4 or tile2=8`, `tile1*tile2<=12`, and `W*tile1<=2048`. The packed/padded old residuals are scaled by 0.016809401000806315 and 0.030215324807753154, respectively. |

B is the payload bytes per core per call and C is the actual active core count. The core bounds here come from each
fitted domain. The generic NDDMA cacheline-padding parameter is computed separately from the MTE2 shared head.
`NDDMA_RUNTIME_DISPATCH[match|else]` describes the model branches retained at generation time. Both sides may carry
guards, and the solver evaluates one estimate expression.

The [case list](cases.csv) and [measurement archive](results/v35/pr2921-d68/report_en.md) keep the archived training,
holdout, known-candidate, and out-of-domain controls. Normalization, physical 1-D, dynamic packed/padded guards,
rank 3/4, and INT64 equivalence and fallback cases are covered by `NddmaModelV2.*`. The same core count may have
different descriptors; residual ratios must not be extrapolated from the layouts in the table to other transposes.

## Narrow Contiguous Source and Long Transpose Axis

When the contiguous GM width is only 1-8 FP32 elements and the transpose axis is 512-16384 elements, the packed
NG2 formula based on UB payload may underestimate the actual read requests. The new branch keeps the original
startup, linear-transfer, and inter-core-contention structure and charges by the number of strided positions. See the
[independent report](results/v35/narrow-source-v1/report_en.md) for the exact domain and frozen coefficients.
It applies only to valid physical descriptors with FP32, a source contiguous-axis step of 1, GM pitch 30-512
elements, and packed UB. A static physical 1-D unit axis and a symbolic tile=1 use the same fitted cost.
Out-of-domain descriptors keep the original formula, and neither model name nor total device core count participates
in dispatch. The active core count C is still provided by the current candidate.

The correction targets the underestimated default narrow-read candidates 0120, 0134, and 0166. The sampled core
counts are 24/32/40/48/56; real-device acceptance at higher core counts still requires the corresponding environment.
UT covers the physical unit axis, other dtypes, out-of-domain descriptors, and symbolic boundaries.
