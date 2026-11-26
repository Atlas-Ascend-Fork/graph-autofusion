# A3 MTE2 Load (GM2UB) Model Design Archive

[中文](design.md)

## Scope

This archive covers a single GM-to-UB `DataCopy` through ATT `LoadApi`. This directory records A3 collection evidence under the V1 family. See cases.csv for CoreNum, shape, continuous, and non-continuous coverage. The v35 parameter snapshot cited by the original archive is preserved below and does not represent current V1 parameters; see [perf_param_v1.cpp](../../../../gen_model_info/api_perf_register/v1/perf_param_v1.cpp) for the V1 implementation.

## Data Sources and Experimental Design

Evidence references use the `mte2_mte3_perf_analysis:A3/` root. Conclusions and data references come from two experiment directories and two notebooks. A3 CSV files provide `shape`, `idx`, `aiv_mte2_time(us)`, `aiv_mte2_cycles`, and aggregate bandwidth across all cores. The observation column is `aiv_mte2_cycles`; the CSV files do not provide `aiv_mte2_ratio`.

| Experiment | Variables / fixed values | Entry point | Data | Analysis |
| --- | --- | --- | --- | --- |
| Continuous transfer | CoreNum 1–48; shape 32B–512B, 512B–100KB, 100KB–192KB | `mte2_mte3_perf_analysis:A3/mte2_gm2ub_continuous/run.sh` | `mte2_mte3_perf_analysis:A3/mte2_gm2ub_continuous/profiling_data/` | `mte2_mte3_perf_analysis:A3/scripts/mte2_analysis.ipynb` |
| Multiple calls | shape 196608B; tilenum 8–1000; single core | Same as above | `mte2_mte3_perf_analysis:A3/mte2_gm2ub_continuous/profiling_data/mte2_headcost_test.csv` | `mte2_mte3_perf_analysis:A3/scripts/mte2_analysis.ipynb` |
| Non-continuous src | Single core; shape 196608B; blocklen 8–768B; srcstride 1–40 | `mte2_mte3_perf_analysis:A3/mte2_gm2ub_uncontinuous/run.sh`, `run_prof.sh` | `mte2_mte3_perf_analysis:A3/mte2_gm2ub_uncontinuous/profiling_data/` | `mte2_mte3_perf_analysis:A3/scripts/mte2_uncontinuous_analysis.ipynb` |

## Model Formulas

The original archive cites the following v35 implementation parameters:

| Entry | Parameters |
| --- | --- |
| Load | `a=11.8292`, `b=6.6155`, `h=160`, `hl=0` |
| LoadSmallBlk | `a=13.1355`, `b=6.4088`, `h=160`, `hl=0` |
| LoadSingleCore | `a=23.0520`, `b=0`, `h=160`, `hl=0` |
| LoadStride | `k=0.005`, `u=4096.0`, `penalty_coeff=0.0` |
| Pipeline overhead | `1174.3` (blocklen ≥256B) or `775.0` (blocklen <256B) |

Let `D=product(dims)*data_type_size`, `N=block_dim`, and `t=a+b/N`:

```text
base = D / t + h
unalign_penalty = hl * (256 - (h + 512 / t))
stride_cost = k * block_count * min(gm_stride * data_type_size, u)
load_cost = base + unalign_penalty + stride_cost
```

`block_dim==1` selects SingleCore entries. Other values select Load or LoadSmallBlk. `GetLoadCase` selects the normal or small-block expression using the codegen tail-axis blocklen and uses a ternary expression for dynamic values.

## Implementation Path

- [ascir_api_perf_v2.cpp](../../../../../v35/att/api_perf_register/ascir_api_perf_v2.cpp): LoadApi entry point.
- [perf_param_v2.cpp](../../../../../v35/att/api_perf_register/perf_param_v2.cpp): pipeline overhead and parameter tables.
- [api_perf_utils.cpp](../../../../../att/gen_model_info/api_perf_register/utils/api_perf_utils.cpp): LoadStoreFunc, LoadStoreStrideV2Func, and DMA parameter calculation.
- [data_move_api_perf_v2.cpp](../../../../../v35/att/api_perf_register/ascendc_api_perf/data_move_api_perf_v2.cpp): normal, small-block, stride, and single-core branch assembly.

## Validation Design

Continuous data covers small and large shapes and CoreNum 1–48. The headcost experiment covers multiple calls. The srcstride experiment covers non-continuous access on the GM source side. Structural validation checks implementation parameters in parameters.json, the 10-column cases.csv format, evidence references, and Markdown links.

The A3 `mte2_pad` experiment is outside this load archive's scope. This document describes the archived model and evidence entry points without modifying implementation source code.
