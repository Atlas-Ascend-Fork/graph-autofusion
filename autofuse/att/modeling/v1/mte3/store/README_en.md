# A3 MTE3 Store Archive

This archive follows the A3-to-V1 directory mapping. The formulas and parameters below are a historical v35 Store snapshot from commit `cdc88a04`, not V1 parameters. The A3 data index does not record the implementation revision used for collection, so it establishes neither v35 usage nor measured validation against V1.

Continuous, headcost, and non-contiguous datasets are retained as coverage and evidence indexes. V1 uses Store `h=12.09,a=9.96,b=3.79`, StoreStride `k=0.02`, and MTE3 head `497.36`. These differ from the v35 snapshot below, which is not applied to A3/V1 here.

## Historical v35 snapshot conventions

| Item | v35 snapshot model |
| --- | --- |
| Single transfer | `S / (11.774 + 10.265 / CoreNum) + 160` |
| Multiple calls | `single_call * ApiCount + 571` |
| GM stride | `0.0049922306718019 * BlockCount * min(Stride,4096)` for `uint8` |
| Unaligned | `hl=0` |

Coverage includes 46 core CSV sets (missing 9 and 43), 384 shapes from `512B` to `192KB`, headcost `tilenum=2–642`, and single-core stride `1–30`.

The snapshot cites the Store and StoreStride tables in `perf_param_v2.cpp@cdc88a04`. Raw A3 coverage does not validate these values for V1. The collection implementation revision and V1 parameter validation remain open.

Evidence: `A3/scripts/mte3_analysis.ipynb`, `A3/mte3_ub2gm_continuous/`, `A3/mte3_ub2gm_uncontinuous/`.

See [design.md](design_en.md), [cases.csv](cases.csv), and [parameters.json](parameters.json).
