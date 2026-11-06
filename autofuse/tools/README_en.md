<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Autofuse Tools

These tools support Autofuse development, debugging, and issue analysis. They are not runtime APIs. Use each command's `--help` output as the authoritative reference for options.

## NDDMA: modeling and ad-hoc prediction/collection

Tool directory: [`nddma/`](nddma/README_en.md).

- Purpose: Round1–5 collection, fitting and analysis, plus single-configuration prediction and measurement.
- Inputs: shape, GM/UB strides, core count and dtype, or measurement CSVs and model JSONs.
- Commands (from the repository root):

  ```bash
  python3 autofuse/tools/nddma/Modeling/Data_collection/common/tools/predict_nddma_ad_hoc.py --help
  python3 autofuse/tools/nddma/Modeling/Data_collection/common/tools/run_nddma_ad_hoc.py --help
  ```

- Results: cycle predictions, formula terms, fitted parameters and measurement analysis. Historical snapshot metrics are marked invalid; collection requires CANN and an Ascend device.


## ATT-ANALYZE: template, tiling, and profiling analysis

Tool directory: [`att_analyze/`](att_analyze/). The common entry point is:

```bash
python3 autofuse/tools/att_analyze/src/att.py --help
```

- `summary`: parses a log file or directory and exports operator, case, tiling, objective, and measured performance fields to CSV or text. Missing fields remain empty and are marked by `parse_status`.
- `compare`: compares two `summary` CSV files, such as default Autofuse versus PGO or a forced template. It reports matched operators, case/tiling differences, and performance changes.
- `evidence`: converts logs to JSONL evidence with source paths, line numbers, and parse status for subsequent automated analysis.
- `split-slog`: separates compiler DFX and runtime PROF fragments by operator and graph/result/group/case. It accepts slog, stdout, or a log directory without requiring fixed filenames.
- `perf-formula`: parses `[PERF]` pipe formulas from tiling output, identifies the bottleneck, and writes a `perf_formula.svg` comparison. It returns a non-zero status when the required evidence is absent.
- `verify-tiling`: compiles user-provided TensorFlow or Inductor tiling code and checks the `AutofuseTiling` ABI, including block dimensions and workspace results. It prints the selected `aiv_num` and saves `result.json`; `preset_B` defaults to 56 but must be checked against the target hardware.

Examples:

```bash
python3 autofuse/tools/att_analyze/src/att.py summary run.log -f csv -o summary.csv
python3 autofuse/tools/att_analyze/src/att.py compare default.csv candidate.csv -f text -o compare.txt
python3 autofuse/tools/att_analyze/src/att.py evidence run.log -o evidence/
python3 autofuse/tools/att_analyze/src/att.py split-slog slog/ --op FlashAttentionScore -o split/
python3 autofuse/tools/att_analyze/src/att.py perf-formula generated/ run.log -o perf/
python3 autofuse/tools/att_analyze/src/att.py verify-tiling generated/ --scene tf --preset B --aiv-num 56 -o verify/
```

`FINAL_TILING` records include graph/result/group/case/key, template name, `score`, pipe estimates, and tiling representation. Long records use `FINAL_TILING_BEGIN/CHUNK/END`; the producer uses `hash_alg=att_mix64_v1` for integrity checks, while the parser remains compatible with historical SHA-256 framed logs.

## NWA `fusion_precision_analyzer`: fusion precision diagnosis

Tool directory: [`nwa_tool/`](nwa_tool/). Compare graphs and NPY data from Autofuse enabled and disabled runs to locate the fusion operator responsible for a precision regression.

```bash
python3 autofuse/tools/nwa_tool/fusion_precision_analyzer.py \
  --af-open-graph open/Build.json --af-close-graph close/Build.json \
  --af-open-data open/npy --af-close-data close/npy --compare-input

python3 autofuse/tools/nwa_tool/fusion_precision_analyzer.py --mode 2 \
  --npy-a open.npy --npy-b close.npy
```

The tool reports cosine similarity, maximum absolute and relative errors, and statuses such as `OK`, `FILE_NOT_FOUND`, and `SHAPE_MISMATCH`. See [`nwa_tool/README.md`](nwa_tool/README.md) for the complete option and format reference.
