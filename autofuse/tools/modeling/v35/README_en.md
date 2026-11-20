# v35 ATT Modeling Tools

V35 tools only; configurable architecture options do not establish V1 support.

Collection and fitting code is centralized here. Designs, cases, and concise measured results are archived in the
[ATT model directory](../../../att/modeling/README_en.md).

| Tool | Purpose |
| --- | --- |
| [NDDMA](nddma/README_en.md) | Round1-5 collection, fitting, prediction, and single-configuration measurement |
| [Ordinary Load / Store](dma/README_en.md) | v35 FP32 DMA device collection, accuracy validation, and reproduction |
| [MTE2](mte2/README_en.md) | Error checking and refitting for archived Load / NDDMA data |
| [Gather](vector/gather/README_en.md) | Device collection of bank and mask components |
| [IsInf](vector/isinf/README_en.md) | Device collection and fitting for a single API |
| [GetOpHeadCost](common/op_head_cost/README_en.md) | Device collection, evidence checking, and offline recomputation of the shared per-core cost |

Run these tools manually as needed. Template, tiling, and log-location tools remain in
[ATT Analyze](../../att_analyze/README_en.md).

## Archive Coverage and Paths

| Modeling item | Collection, fitting, or validation entry | Coverage boundary |
| --- | --- | --- |
| Nine FP32 Load/Store branches | [DMA](dma/README_en.md), [CSV revalidation](mte2/README_en.md) | For some early collectors only same-series source is available; see the DMA README for exact replay limits |
| NDDMA physical 1-D, packed NG2, multi-axis models | [Round1-5](nddma/README_en.md) | Generic descriptor collection and fitting; PR calibration domains are described in the model designs |
| Padded NG2, packed rank3, sparse rank3/4, small INT64 | [CSV revalidation](mte2/README_en.md) | Padded NG2 supports refitting; other branches retain frozen predictions and raw evidence indexes, and not all dedicated historical collectors are archived |
| Gather bank/mask | [Gather](vector/gather/README_en.md) | Component collection source is archived; the full product formula is validated by targeted UT |
| TrueDiv/Rsqrt VF | [VF design](../../../att/modeling/v35/vector/vf/design_en.md) | Reuses existing instruction parameters; no new measured coefficient; validation entry is the UT listed in the design page |
| Shared core-count cost | [GetOpHeadCost](common/op_head_cost/README_en.md) | Two collectors plus raw-profiling conversion and offline fitting |
| IsInf | [IsInf](vector/isinf/README_en.md) | Reuses the existing independent project |

In-repo paths in the commands are run from the repository root, and internal Python/CMake defaults are resolved from
the script directory. CANN, device architecture, input data, and experiment output are provided by arguments or
environment variables.

The manifest `case_inventory` and `reproduction` entries are resolved relative to its own directory.
`source_root=external_evidence` means the source is relative to the external evidence root provided by the user,
and `source_root=external_runs` means it is relative to the historical experiment runs root.
Source-file checksums are used to verify historical evidence and remain in use after path migration.
External evidence locations are for provenance only; offline error revalidation reads the in-repo cases and samples directly.
