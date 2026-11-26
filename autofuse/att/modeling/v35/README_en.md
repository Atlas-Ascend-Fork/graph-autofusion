# ATT Performance Model Archive

Archive model designs, test cases, and validation results. Model owners will fill in the content later.

## Directories

```text
modeling/v35/
├── README.md
├── common/                 # shared overhead
│   └── op_head_cost/
├── mte2/                   # Load models
│   ├── README.md
│   ├── load/
│   └── nddma/
├── mte3/                   # Store models
│   ├── README.md
│   └── store/
├── vector/                 # Vector API models
│   ├── README.md
│   ├── broadcast/
│   ├── concat/
│   ├── elewise/
│   │   └── cast/
│   ├── gather/
│   ├── reduce/
│   └── transpose/
└── simt/                   # Add models for actual execution paths
    └── README.md
```

Each category README describes its modeling methods. Shared archive conventions belong in this file.

Collection and fitting code lives in [Autofuse modeling tools](../../../tools/modeling/README.md). See [GetOpHeadCost](common/op_head_cost/design.md) for the shared overhead model.

See the [MTE3 Store archive](mte3/store/README_en.md) for the original A5 calibration and the small contiguous Store extension. The A3 archive is under [V1 Store](../v1/mte3/store/README_en.md).

## Model Archives

Add the following files to each model directory as needed:

```text
<model>/
├── design.md               # Modeling rationale, formulas, parameter sources, and scope
├── cases.csv               # Cases, coverage conditions, and execution entry points
└── results/<run_id>/
    ├── manifest.json       # Environment, versions, commands, and raw evidence references
    ├── samples.csv         # Model predictions and actual measurements
    └── report.md           # Error analysis, conclusions, and limitations
```

Focus designs on modeling evidence, provide executable cases, and use actual validation results.
Record model and environment versions, align prediction and measurement scopes, and separate fitting from independent validation data.
Keep raw logs, profiling artifacts, and binaries outside the repository and reference them through the evidence index.
Use `.gitkeep` to preserve empty directories and remove it when archive content is added.
