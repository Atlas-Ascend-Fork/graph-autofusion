# Direct-store NDDMA: frozen results

Business-candidate records, full tiling, Host timings and 17/48-case regression evidence are tracked at a pinned revision in [CANN-DevTools](https://gitcode.com/zhang_shengjie/CANN-DevTools/blob/6733c737b04f2e6289ebeb5784adaccb7c1e6d30/att_generalization/report/20261119/pr3065-model-regression/README.md). This directory retains API modeling rationale, independent training/holdout data and validation summaries.

Product baseline: PR #3065 caeee3d1e64cfd57a5765291a5ec9c046ef09d3a. Device: Ascend950PR_9579 with 56 AIVs. These measurements do not establish the user's 64-core whole-network performance.

## Independent collection and fitting

Initially 19 descriptors were collected. The final domain is FP32 M=2–8 (GM contiguous source≤32B): seven training, five holdout and two known points; five others are domain controls. Five additional unseen holdouts were collected after freezing the narrowed domain and coefficients, giving 19 final in-domain points. All outputs and guards passed.

Use repository steady_load.asc padded-5D NDDMA, double-buffered TQueBind, NORMAL hint and a 4 GiB GM pool with fresh offsets across calls, cores and launches. Three warmups and ten measured samples for each K=1/2/4/10. Per-core MTE2 cycles are aiv_total_cycles*aiv_mte2_ratio/Block Num; the slope of K medians estimates cycles/API.

isolated-samples.json uses tail_store; samples.json uses full span_store. The collector includes guard regions, whereas production Store writes actual data with row pitch; this is a controlled intervention, not an instruction-identical production kernel. Known `[8,3952]` increases from 9637.1 to 16107.2 cycles. Production uses `[8,3960]`; the nearby same-alignment 3952 descriptor accommodates guard-region budget and must not be presented as identical. Controls rule out padded5D vs native2D and cross-core address order as primary explanations.

Keep the packed NG2 API model; add `k*max(B*C-U,0)` only for structurally proven full-writeback paths. Wider-than-32B sources could lose performance by choosing too few cores. The `[1024,1536,12]` control therefore keeps its original model and 55-core tiling, rather than the initial 38-core result. The boundary follows physical DMA block width, not case names or device total core count.

Only seven training points fit the narrowed model. Five fresh holdouts reach at most 26.06% error and never participate in fitting. Refit from samples.json with the existing tool.

| Split | Points | Mean absolute relative error | Maximum |
| --- | --- | --- | --- |
| Training | 7 | 10.27% | 23.73% |
| Holdout, original five plus fresh five | 10 | 13.04% | 26.06% |
| Known | 2 | 15.88% | 22.54% |

k=0.0038940645164791633, U=4456448 bytes. See design_en.md for domains and fallbacks. Measured active cores stop at 56; 64/72 formula tests are not device validation.

## Default selection

Pure PyTorch permute is a view. Materialize permute(2,0,1).contiguous(), FP32 `[2048,2048,8]`→`[8,2048,2048]`. This is not the whole-network BMM→standalone transpose→Softmax boundary. Separate caches/processes; 20 warmups, five Event groups of 40 calls, profiler warmup two and active five.

| Version | PGO | Kernel median μs | Event median μs | Dominant pipe | Precision |
| --- | --- | --- | --- | --- | --- |
| Original #3065 | Off | 217.780 | 219.722 | MTE2 99.2% | Pass |
| Final direct-store fix | Off | 200.597 | 202.868 | VEC 98.3% | Pass |
| Same-package fast solution | Requested Top10 | 200.580 | 202.304 | VEC 98.3% | Pass |

Original codegen SHA256: 9046392aebea224846602973bfbb20565252f9eaf6495d488ad9688dc5cf2d1e. Final: 0aa255b9ebd8db824168b4339a3c070e2a182220bcb64643c39a164ba345adae.

Default and PGO both use result0/group0/case0, tile=[8,1585], 48 loops/core, block_dim=56 and q0/q1/index buffers of 50720 bytes. Original case1 uses tile=[8,3960], 19 loops/core, q0=126720 bytes. This entry dispatches on result/local key in tiling data; there is no separate TILING_KEY macro number. Complete tiling and device source SHA match; actual function is graph_hint_0_b0_x0_y1_s0_g0_c0. Index Store dtype correction alone does not change the default; MTE3 and search rules are not manipulated.

[Compact evidence](https://gitcode.com/zhang_shengjie/CANN-DevTools/blob/6733c737b04f2e6289ebeb5784adaccb7c1e6d30/att_generalization/report/20261119/pr3065-model-regression/direct-store-v1/verification.json) includes actual libraries, source hashes, complete tiling, dispatch and profiler summaries. [Fit plot](fit-comparison.svg) separates training, holdout and known points.

## Host and controls

Thirty reverse-order paired rounds run the native static Inductor AutofuseTiling entry in fresh processes; first call once, then 1000 repeat calls. First-call median 84.0865→85.2830μs (+1.42%); median repeat-call mean 0.029524→0.029539μs. Static-cache hits do not solve again, which does not mean first tiling is free. See [raw scalars](https://gitcode.com/zhang_shengjie/CANN-DevTools/blob/6733c737b04f2e6289ebeb5784adaccb7c1e6d30/att_generalization/report/20261119/pr3065-model-regression/direct-store-v1/host-timing.json). Device runtime logic is unchanged and final device source matches PGO.

| Control | Base kernel μs | Fixed-control measurement μs | Complete execution identity |
| --- | --- | --- | --- |
| FP32 transpose+Softmax | 1135.920 | 1139.869 | Same |
| FP16 materialized transpose | 115.880 | 109.538 | Same |
| FP32 48B contiguous source | 107.623 | 94.985 | Same, restored 55 cores |
| FP32 128B contiguous source | 354.561 | 352.631 | Same |

Controls were measured using the narrowed package, then regenerated using the final package. Generated Host/device sources are byte-identical, allowing evidence reuse. Timing fluctuations are not credited as gains. Accuracy_level=0 only isolated AutoTuning during diagnosis; final acceptance uses level=1.

## Validation state

297 native NddmaModelV2 / UTestAscirPerfV2 tests pass, covering static/symbolic inputs, dtype, padding, source-width boundaries and real-graph consumer gates. Complete tiling, result/group/case dispatch and source identity are verified. The control set covers vector consumers, FP16, 48B and 128B sources. The 17/48-case Host impact check shows no changed selections; device evidence is reused as authorized: [report](https://gitcode.com/zhang_shengjie/CANN-DevTools/blob/6733c737b04f2e6289ebeb5784adaccb7c1e6d30/att_generalization/report/20261119/pr3065-model-regression/direct-store-v1/regression-17-48_en.md). Host 64/72-core replay selects case0. The 453-case check, actual 64/72-core timing and dynamic-shape execution remain unverified in this evidence round. Later RA/Load calibration results are archived separately.
