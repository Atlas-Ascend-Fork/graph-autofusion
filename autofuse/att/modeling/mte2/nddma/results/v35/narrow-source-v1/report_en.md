# FP32 Narrow Contiguous Source Long-Axis NDDMA Calibration

[Cases](../../../cases-narrow-source.csv) | [Point-level data](samples.csv) | [Collection protocol](manifest.json)

The product applies this formula to FP32 descriptors with physical GM contiguous width M<=8 elements, transpose
axis N=512-16384, GM pitch 30-512 elements, and packed UB. When a static unit axis is folded into a physical 1-D
form, the same measurement is used; the low-core branch still follows the original selection.
The expression is startup + N x (position_cost + contention_cost x max(C-32,0)/24).
This is the original startup, transfer, and inter-core-contention structure, with the transfer amount expressed as
the number of strided positions so that the UB payload does not hide the GM narrow read.

Only five independent training points participate in the minimax relative-error fit, and training excludes the
actual production M=1/3. Known candidates and the later long-axis validation are compared after the coefficients are
frozen; the original S03 script roles do not participate in fitting. Predictions for out-of-domain controls are left
empty so that extrapolation of the fitted formula is not treated as product behavior.

| Data role | Points | Max absolute relative error |
| --- | ---: | ---: |
| domain_control | 16 | - |
| train | 5 | 14.95% |
| holdout | 3 | 21.28% |
| known | 3 | 15.59% |
| frozen_fit_validation | 5 | 21.43% |

Collection first verifies all outputs and the UB protection region; profiling uses only the slope of the MTE2 API
call count. Raw medians, linear errors, and startup terms for each call count are stored in the external source
index. Single-API fitting and final template ordering are accepted separately. Default results with PGO disabled are
shown below. The 49 GiB input pool covers this experiment's long spans and different addresses per call; the device
must be idle and the remaining HBM checked. The tool runs only when the fixed input pool, output, and check region
together fit the HBM budget.

## Default Selection Validation

The original default 0120 descriptor is `[3,4096]`, GM stride `[1,512]`, UB stride `[4096,1]`;
0134 is `[1,12032]`, GM stride `[1,512]`, UB stride `[12032,1]`; 0166 is `[1,2048]`, GM stride `[1,30]`.
Reconstruct the descriptor from the final kernel dispatch and API parameters; other tile fields not referenced by
the final call cannot be used to infer the long axis.

| Case | Default before fix (us) | Default after fix (us) | no-AF in the same round (us) | Known PGO replay (us) |
| --- | ---: | ---: | ---: | ---: |
| 0120 | 2332.6261 | 553.9494 | 1615.7963 | 499.7241 |
| 0134 | 6754.6522 | 549.8557 | 1616.6457 | 497.2605 |
| 0166 | 51.8580 | 15.0718 | 19.7234 | 7.7748 |
| 0177 | 8.1054 | 8.1048 | 23.5434 | 7.4293 |
| 0180 | 15.1197 | 15.1591 | 18.8403 | 7.8130 |

All five cases meet `no-AF/AF >= 1 or AF - no-AF < 2 us`, and the CPU, single_op, and AF outputs have identical
SHA-256. The PGO column comes from an independent candidate replay on the same SDK and was not rerun on the new
library. The new defaults for 0120 and 0134 select UB Transpose. This only proves that the default business gate is
met; it does not mean that the gap to the fastest PGO time is fully removed.

The same final library completes 48 cases: 48/48 numerical output summaries match and 45/48 meet the performance
gate. The known PGO replay for 0173 and 0183 also misses the gate in this round; 0561 failed before the fix and still
fails in cross-validation, and this is not hidden. 0561 has identical full templates, tiling, and pipeline estimates,
and differs only in the generated operator index. The original framework's graph-node-count checkError and the
performance gate are kept separately; an original Failed must not be turned into Pass automatically.

62 targeted UT and 454 related model/Host/solver UT pass, with one pre-existing disabled test retained.
Both call windows of the independent collection project build and pass the verify/profile output checks.
The 17/453 full suite and the Host-time gates were not rerun this round; active core counts outside 24-56 were not
validated on a real device. Per-case data, raw profiles, fitting plots, and full selections are archived in the
CANN-DevDoc target17 design directory, under the relative evidence directory
`evidence/20261114-narrow-source-default-fix`.
