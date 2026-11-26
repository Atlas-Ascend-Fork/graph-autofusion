# A5 MTE3 Store Archive

This directory follows A5-to-v35. The current small contiguous Store extension is documented in [design.md](design_en.md), [cases.csv](cases.csv), and the [validation report](results/pr2921-d68/report_en.md). The original calibration conventions below are preserved from `cdc88a04` as the base model record outside the extension's domain.

MTE3 is a UB-to-GM path: UB is the source (`src`) and GM is the destination (`dst`). The active model is the implementation Store table.

## Active model

- Single call: `S / (11.774 + 10.265 / CoreNum) + 160`.
- Multiple calls: `single_call * ApiCount + 571`.
- GM stride for the `uint8` cases in this archive: `0.0049922306718019 * BlockCount * min(Stride,4096)`.
- Unaligned penalty is disabled with `hl=0`.
- The original small-payload cases use the Store formula. The current implementation also has a `StoreSmallPayload` branch; see the [extension design](design_en.md) for its domain and coefficients.

## Experiment coverage

The [original case index](cases-legacy.csv) covers the five A5 stages: `1_BlockDim`, `2_ApiCount`, `3_UnAlign`, `4_GmStride`, and `5_UbStride`. It also indexes the 156 and 230 validation batches. The 230 reference is limited to locked-frequency dev2 data with the physical 32 AIV limit; dev0 is a non-baseline comparison.

## Implementation alignment

| Item | Archive value | Implementation value | Result |
| --- | --- | --- | --- |
| Pipe head | `H=571` | `kMte3PipeHeadCost=571.0f` | Match |
| Instruction head | `h=160` | `h=160` | Match |
| Bandwidth | `T=10.265/N+11.774` | `a=11.774,b=10.265` | Match |
| GM stride | `uint8 k=0.0049922306718019`, `u=4096` | `StoreStride` | Match |
| Unaligned penalty | `hl=0` | `hl=0` | Match |
| Block count | Included through `S` and stride block count | `StorePerf` and stride helper | Match |

## Open work

1. Add small-size and unaligned coverage where required.
2. Keep the pipe residual definition `H=731-160` until an independent measurement exists.
3. Repeat UB-stride coverage without adding a Store parameter.
4. Keep the 230 baseline at locked dev2 and `CoreNum<=32`.

## Original evidence index

| Content | External evidence |
| --- | --- |
| A5 summary | `A5/MTE3/5_Summary/初版建模分析_20261021.md` |
| Continuous fit | `A5/MTE3/4_Analyse/1_BlockDim/combine/256KB/fit_parameters.csv` |
| GM stride fit | `A5/MTE3/4_Analyse/5_Stride/GM_stride/GMToUB_fit_results.csv` |
| 156 validation | `results/156_ascend950/20261021_min/report.md` |
| 230 validation | `results/230_ascend350_dev2/20261105_max32/report.md` |

See the original [design](design-legacy_en.md), [case index](cases-legacy.csv), and [parameter snapshot](parameters.json). The A3 archive is under [V1 Store](../../../v1/mte3/store/README_en.md).
