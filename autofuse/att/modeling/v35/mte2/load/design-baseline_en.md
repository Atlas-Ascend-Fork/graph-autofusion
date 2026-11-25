# A5 MTE2 Load (GM2UB) Model Design Archive

[中文](design-baseline.md)

This document preserves the A5 base-model collection design; see [cases-baseline.csv](cases-baseline.csv) for the collection inventory. Later v35 calibrated branches and their domains are documented in the [calibration design](design_en.md).

## Scope

This archive covers a single GM-to-UB `DataCopy` through ATT `LoadApi`: CoreNum 1–64, S of 8B–256KB per core per call, continuous transfers, unaligned transfers, and non-continuous access on the GM side. Raw profiling data remains outside the repository; collection repository paths are evidence references only.

## Data Sources and Experimental Design

The collection root is `mte2_mte3_perf_analysis:A5/MTE2/`. The `run_*.sh` scripts inject corenum, blockcount, stride, and other parameters with `sed`, then rebuild and run the operator. `msprof` collects profiling data. `Ana_gen_info.py` extracts `aiv_mte2_time/aiv_total_cycles/aiv_mte2_ratio/aiv_mte2_cycles` from op_summary. A5 `aiv_mte2_cycles` is adjusted by the ratio to represent MTE2 time. Data timestamps span 2025-08-12 ~ 2025-09-10.

| Stage | Variables / fixed values | Collection entry point | Data / fitting | Purpose |
| --- | --- | --- | --- | --- |
| 1_BlockCount | CoreNum 1/8/16/32/48/64; blockcount 1/20/40/60/80; shape 0–256KB, 32B aligned | `mte2_mte3_perf_analysis:A5/MTE2/1_Script/1_BlockCount/run_count.sh` | `mte2_mte3_perf_analysis:A5/MTE2/2_Data/1_BlockCount/`, `mte2_mte3_perf_analysis:A5/MTE2/4_Analyse/1_BlockDim/` | Large-scale curves |
| 2_ApiCount | apicount 1–100 (10KB) / 1–1000 (256KB); CoreNum 1–64 / 16/32/64 | `mte2_mte3_perf_analysis:A5/MTE2/1_Script/2_ApiCount/run_count.sh` | `mte2_mte3_perf_analysis:A5/MTE2/2_Data/2_ApiCount/`, `mte2_mte3_perf_analysis:A5/MTE2/4_Analyse/2_ApiCount/` | Multiple-call curves |
| 3_UnAlign | CoreNum 1/8/16/32/48/64; blockcount 20/40/60/80; shape 8B–256KB | `mte2_mte3_perf_analysis:A5/MTE2/1_Script/3_UnAlign/run_count.sh` | `mte2_mte3_perf_analysis:A5/MTE2/2_Data/3_UnAlign/`, `mte2_mte3_perf_analysis:A5/MTE2/4_Analyse/4_UnAlign/` | Unaligned coverage |
| 4_GmStride | Shape, blockcount, and srcStride sweeps | `mte2_mte3_perf_analysis:A5/MTE2/1_Script/4_GmStride/run_stride.sh` | `mte2_mte3_perf_analysis:A5/MTE2/2_Data/4_UnStride/0.GM_Stride_Test/`, `1.GM_Stride/`, `mte2_mte3_perf_analysis:A5/MTE2/4_Analyse/5_Stride/` | Non-continuous GM source access |
| 5_UbStride | CoreNum 1/8/16/64; blockcount 2/4/16/32/64/128/256/512; dstStride 0–8 | `mte2_mte3_perf_analysis:A5/MTE2/1_Script/5_UbStride/run_stride.sh` | `mte2_mte3_perf_analysis:A5/MTE2/2_Data/4_UnStride/2.UB_Stride/`, `mte2_mte3_perf_analysis:A5/MTE2/4_Analyse/5_Stride/Results_UB_Stride/` | Non-continuous UB destination access |

`4_Analyse/3_BlockCount/Ana.py` reuses 1_BlockCount data to examine merging blockcount into S.

## Base Model and Experimental Observations

The shared formula structure is:

```text
mte2_once  = S / T(CoreNum) + h
mte2_total = mte2_once * ApiCount + H
```

Experimental data is grouped into large and small scales. Continuous dimensions are merged into S=blockcount*blocklen. Large-scale curves examine throughput changes with CoreNum. Small-scale data covers the implementation's small-block branch. Unaligned and stride data covers additional LoadPerf paths.

Archived model parameters come directly from the implementation table:

| Entry | Parameters |
| --- | --- |
| Load | `a=11.8292`, `b=6.6155`, `h=160`, `hl=0` |
| LoadSmallBlk | `a=13.1355`, `b=6.4088`, `h=160`, `hl=0` |
| LoadSingleCore | `a=23.0520`, `b=0`, `h=160`, `hl=0` |
| LoadStride | `k=0.005`, `u=4096.0`, `penalty_coeff=0.0` |
| Pipeline overhead | `1174.3` (blocklen ≥256B) or `775.0` (blocklen <256B) |

`LoadStoreFunc` computes `D/(a+b/block_dim)+h+hl*(256-(h+512/t))`, where D is the product of dims multiplied by the dtype size.
`LoadStoreStrideV2Func` computes `0.005*block_count*min(gm_stride*data_type_size,4096.0)`.

## Branches, Alignment, and Stride

`GetLoadCase` selects Load or LoadSmallBlk using the codegen tail-axis blocklen. `block_dim==1` selects the corresponding SingleCore entry. Dynamic blocklen uses a ternary expression.
The GM side is src (source), and the UB side is dst (destination). `SetStride` supplies the GM stride. The stride expression calculates block_count using block_count_idx. For non-continuous access with blocklen smaller than a cache line, LoadPerf expands blocklen first.
With `hl=0`, the alignment penalty contributes no additional cycles. Parameter `u=4096.0` controls the stride upper bound.

## Implementation Path

- [ascir_api_perf_v2.cpp](../../../../../v35/att/api_perf_register/ascir_api_perf_v2.cpp): LoadApi, MergeTensorContinuousDims, and GetDmaPerf.
- [perf_param_v2.cpp](../../../../../v35/att/api_perf_register/perf_param_v2.cpp): pipeline overhead and Load parameter tables.
- [api_perf_utils.cpp](../../../../../att/gen_model_info/api_perf_register/utils/api_perf_utils.cpp): LoadStoreFunc, LoadStoreStrideV2Func, SetStride, and GetDMAActualPerf.
- [data_move_api_perf_v2.cpp](../../../../../v35/att/api_perf_register/ascendc_api_perf/data_move_api_perf_v2.cpp): LoadPerf branch assembly.

## Validation Design

| Category | Coverage | Method |
| --- | --- | --- |
| Structure | JSON, CSV, V1/v35 navigation, and evidence references | Parsers and script checks |
| Parameters | Load, SmallBlk, SingleCore, Stride, and pipeline overhead | Check against implementation source code |
| Branches | blocklen, block_dim, GM stride, and UB stride | Check entry points covered by cases-baseline.csv |
| Performance | Measurements and implementation formulas | Review original collection evidence |

This document records model design, experimental coverage, and implementation parameters without modifying implementation source code.
