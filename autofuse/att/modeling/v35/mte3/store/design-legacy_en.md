# A5 MTE3 Store Design

Archive note: this file preserves the base model record and implementation line numbers from `cdc88a04`. See [design.md](design_en.md) for the current `StoreSmallPayload` extension and [cases-legacy.csv](cases-legacy.csv) for the original experiment index.

## Scope

This archive covers A5 UB-to-GM Store transfers. The active model applies to the listed single and multiple calls, core counts, sizes, and GM stride cases. UB-side stride does not add a Store term.

## Active formulas and parameters

```text
single_mte3 = S / T + h
T = 11.774 + 10.265 / CoreNum
h = 160
overall_mte3 = single_mte3 * ApiCount + H
H = 571
stride_cost = k * BlockCount * min(Stride * dtype_size, u)
```

For the `uint8` cases in this archive, `k=0.0049922306718019`, `u=4096`, and `penalty_coeff=0`. The implementation parameter table is at `../../../../../v35/att/api_perf_register/perf_param_v2.cpp@cdc88a04:1014` and `perf_param_v2.cpp@cdc88a04:1024`; the pipe head is at `../../../../../v35/att/api_perf_register/perf_param_v2.cpp@cdc88a04:20`.

## Data sources and coverage

The A5 evidence index covers host 144 (`Ascend950PR_9599`, CANN 9.2.0, `uint8`), with the five stages `1_BlockDim`, `2_ApiCount`, `3_UnAlign`, `4_GmStride`, and `5_UbStride`. Measurements use the per-core cycle convention recorded in the source material. The 156 and 230 validation paths are indexed separately.

| Stage | Fixed dimensions | Sweep | Purpose |
| --- | --- | --- | --- |
| `1_BlockDim` | Continuous transfer | `S`, `CoreNum`, BlockCount | Continuous coverage |
| `2_ApiCount` | Short and large payloads | `ApiCount`, `CoreNum` | Multiple-call coverage |
| `3_UnAlign` | Non-32B-aligned shapes | Shape and core count | Unaligned coverage |
| `4_GmStride` | `S=102400B`, `uint8` | GM stride 0–4096 | GM stride coverage |
| `5_UbStride` | `S=2048B` | UB stride 0–8 | Source-side coverage |

## Core and payload handling

`BlockCount` enters through the dimensions and the stride term. It is not an additional bandwidth parameter. The active bandwidth values by core count are derived directly from the implementation formula:

| CoreNum | Active T (B/cycle) |
| --- | --- |
| 1 | 22.039 |
| 16 | 12.4155625 |
| 32 | 12.09478125 |
| 48 | 11.987854166666665 |
| 64 | 11.934390625 |

## Unaligned and small payload handling

`hl=0`, so the unaligned penalty term is disabled. The archived revision had no `StoreSmallPayload` branch; small payloads used the same Store formula and `h/H` values. See [design.md](design_en.md) for the extension's domain in the current implementation.

## Cross-platform indexes

The 156 and 230 paths are retained for reproducibility and coverage. The 230 baseline is the locked 1650 MHz dev2 data, with `CoreNum` capped at 32 because the device has 32 AIVs. The 100 MHz dev0 path is a comparison-only dataset.

## Implementation mapping

| Component | Implementation |
| --- | --- |
| Store table | `../../../../../v35/att/api_perf_register/perf_param_v2.cpp@cdc88a04:1014` |
| Store stride table | `../../../../../v35/att/api_perf_register/perf_param_v2.cpp@cdc88a04:1024` |
| Stride calculation | `../../../../../att/gen_model_info/api_perf_register/utils/api_perf_utils.cpp@cdc88a04:291` |
| Store evaluation | `../../../../../v35/att/api_perf_register/ascendc_api_perf/data_move_api_perf_v2.cpp@cdc88a04:334` |
| Block-length expansion | `../../../../../v35/att/api_perf_register/ascendc_api_perf/data_move_api_perf_v2.cpp@cdc88a04:404` |

## Limits

Do not extrapolate beyond the indexed sizes, core counts, physical AIV count, or the `u=4096` stride cap. New coverage should update both language versions of this design and the structured parameter file.
