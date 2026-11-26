# A3 MTE3 Store Design

Archive note: the directory follows A3-to-V1. The parameters below are a historical v35 snapshot from `cdc88a04`, not V1 parameters. The collection implementation revision is unrecorded, so A3 collection using v35 cannot be established. See [README.md](README_en.md) for the V1 differences; V1 calibration and measured validation remain open.

## Scope

This archive indexes A3 UB-to-GM Store cases and preserves the v35 table cited in the original record. Its continuous, multiple-call, and GM-stride formulas do not establish applicability to A3/V1. Continuous data spans single-core through 48-core cases; missing cores and unlisted sizes are not extrapolated. The minimum archived size is 512B; 32B–512B small-payload data is not archived.

## Data sources

Continuous data is indexed under `A3/mte3_ub2gm_continuous/`, with 46 core CSV sets (cores 9 and 43 are missing) and shapes from `512B` to `192KB`. The headcost file is `A3/mte3_ub2gm_continuous/mte3_headcost_test.csv`; non-contiguous data is under `A3/mte3_ub2gm_uncontinuous/`. These paths describe coverage only. A3 CSV files have no `aiv_mte3_ratio` column.

## Continuous model

```text
single_mte3 = S / (11.774 + 10.265 / CoreNum) + 160
overall_mte3 = single_mte3 * ApiCount + 571
```

The v35 snapshot values are `h=160`, `H=571`, and `T=11.774+10.265/CoreNum`, from `../../../../../v35/att/api_perf_register/perf_param_v2.cpp@cdc88a04:20` and `perf_param_v2.cpp@cdc88a04:1014`. These compare with the archived v35 revision; they are not V1 parameters fitted from A3 data.

## GM-side non-contiguous transfer

For the `uint8` cases, the v35 snapshot formula is:

```text
stride_cost = 0.0049922306718019 * BlockCount * min(Stride, 4096)
```

The archived parameter is defined at `../../../../../v35/att/api_perf_register/perf_param_v2.cpp@cdc88a04:1024`; the dtype-aware calculation is at `../../../../../att/gen_model_info/api_perf_register/utils/api_perf_utils.cpp@cdc88a04:291`.

## Small payloads, headcost, and source-side stride

The raw A3 headcost and source-side stride files remain coverage indexes. The available small-scale record contains only the 512B point; 32B–512B small-payload data is not archived. The `cdc88a04` v35 historical snapshot has no separate `StoreSmallPayload` table entry: its small payloads use the Store continuous formula, the pipe term is `571`, source-side stride adds no Store term, and `hl=0`. The current v35 implementation added a `StoreSmallPayload` branch later, which is outside this historical snapshot. None of this describes V1 branch selection or parameters.

## Comparison with the cdc88a04 v35 implementation

| Item | v35 snapshot | Archived v35 implementation | Result |
| --- | --- | --- | --- |
| `T` | `11.774+10.265/CoreNum` | Store `a=11.774,b=10.265` | Match |
| `h/H` | `160/571` | `h=160`, pipe head `571.0f` | Match |
| GM stride | `uint8 k=0.0049922306718019`, `u=4096` | StoreStride | Match |
| Source stride | No Store term | Store formula does not read it | Match |
| Unaligned | `hl=0` | `hl=0` | Match |

## Limits and follow-up

Keep the indexed shape and core limits. The 32B–512B small-payload range is not archived. Confirm the collection implementation and complete V1 calibration before declaring parameters fitted to the A3 data. The A3 pad path requires a separate model if it becomes a supported workload. The archived dynamic GM block-length handling is at `../../../../../v35/att/api_perf_register/ascendc_api_perf/data_move_api_perf_v2.cpp@cdc88a04:404`.
