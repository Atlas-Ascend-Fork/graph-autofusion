# ATT performance model archive

Archive model designs, descriptors and compact validation evidence by model family: [V1](v1/README_en.md) contains the A3 Load and Store archives; [v35](v35/README_en.md) contains the A5 Store archive, existing calibrated models and [the calibration design](v35/design_en.md). `V2` is the v35 registration key, not a third platform family.

```text
modeling/
├── README.md / README_en.md
├── v1/
│   ├── README.md / README_en.md
│   ├── mte2/load/           # A3 evidence and original parameter snapshot
│   └── mte3/store/          # A3 evidence and historical v35 parameter snapshot
└── v35/
    ├── design.md / design_en.md
    ├── common/op_head_cost/
    ├── mte2/{load,nddma}/
    ├── mte3/store/          # A5
    └── vector/
```

Load collection archives map A3 to [v1/mte2/load](v1/mte2/load/README_en.md) and A5 to [v35/mte2/load](v35/mte2/load/README_en.md). A3/A5 paths in the original collection repository remain unchanged as evidence references.

Each model contains `design.md`, executable descriptors in `cases.csv`, and `results/<run_id>/` with version/command indexes, samples and reports. Existing summaries without a run ID remain directly under `results/`. The family is no longer repeated as `results/v35`. Run IDs such as `direct-store-v1` are experiment revisions, not the V1 platform. Placeholder categories do not prove model support.

Preserve units, sample IDs, train/holdout roles, coefficients and historical validity flags. Keep model and API versions explicit. Raw profiling, binaries and business tiling regression data are linked through versioned evidence indexes instead of copied here. Relative `case_inventory` and `reproduction` paths resolve from the manifest directory; external source roots retain their declared meaning. Collection and fitting code is in [modeling tools](../../tools/modeling/README_en.md).
