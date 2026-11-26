# v35 MTE2 Load (GM2UB) Performance Model Archive

[中文](README.md)

`load` corresponds to ATT `LoadApi`: a single `DataCopy` from GM to UB using the MTE2 pipeline. This directory archives A5 collection evidence and v35 models. A3 collection evidence is in the [V1 Load archive](../../../v1/mte2/load/README_en.md).

## Directories

```text
modeling/v35/mte2/load/
├── README.md
├── README_en.md
├── design-baseline.md / design-baseline_en.md  # A5 base-model collection design
├── cases-baseline.csv                         # A5 collection stages and evidence references
├── parameters.json                            # Base implementation parameters
├── design.md / design_en.md                   # Later calibrated branches and domains
├── cases.csv                                 # Executable calibration descriptors
└── results/                                  # Calibration and validation evidence
```

## Shared Definitions

| Symbol | Meaning | Unit |
| --- | --- | --- |
| S | Bytes transferred per core per call; the product of merged dims multiplied by the dtype size | Byte |
| T | Implementation throughput expression `a + b / block_dim` | Byte/cycle |
| h | Instruction overhead parameter of `LoadStoreFunc` | cycle |
| H | Pipeline overhead returned by `GetMTE2PipeHead` | cycle |
| CoreNum | `block_dim`, the number of participating cores | count |
| BlockLen | Block length of the DataCopy tail axis | Byte |
| stride | GM-side `gm_stride`, converted using the dtype size and limited by an upper bound | stride count |

GM is src (source); UB is dst (destination). The implementation adds the single-DMA expression and the stride expression. The DMA scheduling path handles outer call counts.

## Base Implementation Model

This section records base parameters. In-domain estimates also use calibrated branches described in the [calibration design](design_en.md); the current implementation defines branch precedence. A5 BlockCount, ApiCount, UnAlign, GmStride, and UbStride coverage is recorded in the [base-model design](design-baseline_en.md) and [collection inventory](cases-baseline.csv), with parameters in [parameters.json](parameters.json).

```text
t = a + b / block_dim
base = (product(dims) * data_type_size) / t + h
penalty = hl * (256 - (h + 512 / t))
stride_cost = k * block_count * min(gm_stride * data_type_size, u)
load_cost = base + penalty + stride_cost
```

Current parameters are `a=11.8292,b=6.6155,h=160,hl=0` for normal blocks; `a=13.1355,b=6.4088,h=160,hl=0` for small blocks; `a=23.0520,b=0,h=160,hl=0` for the single-core entry; and `k=0.005,u=4096.0` for stride. Pipeline overhead is `1174.3` when the maximum blocklen is `>=256B`, and `775.0` when it is `<256B`. LoadPerf selects the normal expression for blocklen `>256B` and the small-block expression otherwise, using single-core entries when `block_dim==1`.

## Implementation Locations

| Content | Location |
| --- | --- |
| LoadApi registration and branches | [ascir_api_perf_v2.cpp](../../../../../v35/att/api_perf_register/ascir_api_perf_v2.cpp) |
| Pipeline overhead and parameter tables | [perf_param_v2.cpp](../../../../../v35/att/api_perf_register/perf_param_v2.cpp) |
| LoadStoreFunc / LoadStoreStrideV2Func | [api_perf_utils.cpp](../../../../../att/gen_model_info/api_perf_register/utils/api_perf_utils.cpp) |
| LoadPerf assembly | [data_move_api_perf_v2.cpp](../../../../../v35/att/api_perf_register/ascendc_api_perf/data_move_api_perf_v2.cpp) |

The LoadApi path is `MergeTensorContinuousDims → GetDmaPerf → SetStride/GetDMAActualPerf`. V1 and v35 use their respective implementation parameter tables, and directories follow model families.

## Evidence Index

Raw profiling data, `PROF_*`, `op_summary_*.csv`, `*.db`, and binaries remain outside this repository. Collection repository references use the `mte2_mte3_perf_analysis:` prefix.

| Platform | Collection and analysis references |
| --- | --- |
| A5 | `mte2_mte3_perf_analysis:A5/MTE2/1_Script/`, `mte2_mte3_perf_analysis:A5/MTE2/2_Data/`, `mte2_mte3_perf_analysis:A5/MTE2/4_Analyse/`, `mte2_mte3_perf_analysis:A5/MTE2/5_Summary/` |
| A3 | `mte2_mte3_perf_analysis:A3/mte2_gm2ub_continuous/`, `mte2_mte3_perf_analysis:A3/mte2_gm2ub_uncontinuous/`, `mte2_mte3_perf_analysis:A3/scripts/` |

See the [archive conventions](../../README_en.md) for shared requirements.
