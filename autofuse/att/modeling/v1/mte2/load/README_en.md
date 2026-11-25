# A3 MTE2 Load (GM2UB) Model

[中文](README.md)

A3 collection evidence covers GM-to-UB `LoadApi` and is archived under the `v1/mte2/load/` model family. The top-level `parameters`, `formula`, and `source.implementation` in [parameters.json](parameters.json) describe the current V1 implementation. The original v35 parameter snapshot, formulas, and implementation sources are preserved separately under `v35_snapshot`. See [perf_param_v1.cpp](../../../../gen_model_info/api_perf_register/v1/perf_param_v1.cpp) for the current V1 implementation, the [model design](design_en.md#model-formulas) for snapshot formulas, and [shared definitions](../../../v35/mte2/load/README_en.md#shared-definitions) for measurement conventions.

## Documents

| File | Content |
| --- | --- |
| [design_en.md](design_en.md) | Implementation formulas, branches, and data coverage |
| [cases.csv](cases.csv) | A3 collection stages and evidence references |
| [parameters.json](parameters.json) | Implementation parameters and scope |

## Original Parameter Snapshot (v35)

```text
Load:                 t = 11.8292 + 6.6155 / block_dim; h = 160; hl = 0
LoadSmallBlk:         t = 13.1355 + 6.4088 / block_dim; h = 160; hl = 0
LoadSingleCore:       t = 23.0520; h = 160; hl = 0
LoadStride:           0.005 * block_count * min(gm_stride * data_type_size, 4096.0)
pipe head:            1174.3 for blocklen >= 256B, otherwise 775.0
```

`block_dim==1` selects single-core parameter entries; other values select multi-core entries. LoadPerf selects the normal or small-block expression by blocklen and adds the stride cost to the selected branch.

## Data Coverage

See [cases.csv](cases.csv) for A3 continuous, small-scale, headcost, and srcStride evidence references. These files record collection coverage and reproduction entry points. Current model calculations use the V1 implementation parameter table.
