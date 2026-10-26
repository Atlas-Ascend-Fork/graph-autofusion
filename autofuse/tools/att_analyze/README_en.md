# ATT Analyze

In-repository ATT log analysis utilities. Run:

```bash
python3 autofuse/tools/att_analyze/src/att.py --help
```

The CLI provides `summary`, `compare`, `split-slog`, `perf-formula`, `verify-tiling`, `evidence`, and `topn-templates`. `summary` is read-only. `verify-tiling` compiles and executes code, so review its inputs and authorization first.

`LogParser` exposes `OperatorSummary.parse_status` to make incomplete evidence explicit. `ok` means complete selection records; `inferred_graph_result` means graph/result came from template lines; `missing_group_case`, `missing_result_performance`, and `missing_graph_result` identify missing records. Existing CSV column meanings are unchanged.

## FINAL_TILING log contract

`FINAL_TILING` means that the runtime entry has written the final tiling data. The runtime final
selection uses `source="runtime"` (`selection_mode` is `default` or `explicit`), and a PGO-loaded
final tiling uses `source="pgo"` (`selection_mode="pgo"`); candidate searches emit no
`FINAL_TILING` records, and candidate `GetTilingDataRepr` calls are never treated as runtime
final records. The parser only accepts the full field names; the historical compact aliases
(`s=`, `src=`, `k=`, and so on) are no longer parsed. A single-line record is:

```text
[ATT][FINAL_TILING] schema=1 source="runtime" selection_mode="default" operator="Fusion_0" graph=0 result=0 group=1 case_id=2 tiling_key=5 score=1 sub_case_tag="" template="ConcatCase2" repr_kind="full_json" pipe_estimates="{\"AIV_MTE2\":120.000000,\"V\":null}" tiling_repr="{\"tile_m\":64,\"tile_n\":128}"
```

`case_id` (case) and `tiling_key` are independent fields. Multi-group and multi-result records
must retain `graph`, `result`, and `group`. `score` is the final template `CalcScore` value used
by template selection; it is distinct from pipe cycles, objective values, and measured profiling
data. `sub_case_tag` is the sub-case label and is an empty string when no sub-case applies;
`template` is the template name; `repr_kind` is either `full_json` (`tiling_repr` holds the
complete JSON) or `unavailable` (no repr could be produced, `tiling_repr` is an empty string).
`pipe_estimates` contains ATT model pipe estimates. Missing estimates are
JSON `null`; profiling cycles and numeric zero are not substitutes.

If the single-line record exceeds the 700-character line budget, emit one
`FINAL_TILING_BEGIN`, contiguous `FINAL_TILING_CHUNK` lines, and one `FINAL_TILING_END` with
the same `id`:

```text
[ATT][FINAL_TILING_BEGIN] schema=1 source="runtime" selection_mode="default" operator="Fusion_0" graph=0 result=0 group=1 case_id=2 tiling_key=5 score=1 sub_case_tag="" template="ConcatCase2" repr_kind="full_json" pipe_estimates="{...}" id="8:Fusion_0|0|0|1|2|5|0:" chunks=3 len=1842 hash_alg=att_mix64_v1 hash=0e2418542347c1a0
[ATT][FINAL_TILING_CHUNK] id="8:Fusion_0|0|0|1|2|5|0:" seq=0 data="{\"tiling_key\":5,"
[ATT][FINAL_TILING_END] id="8:Fusion_0|0|0|1|2|5|0:" chunks=3 len=1842 hash_alg=att_mix64_v1 hash=0e2418542347c1a0
```

After a multi-group runtime selection completes, one result-level `FINAL_TILING_SUMMARY` line is
also emitted; its `groups` field is the JSON of every group's selection. Oversized summaries use
the `FINAL_TILING_SUMMARY_BEGIN/CHUNK/END` framing in the same way.

`hash_alg=att_mix64_v1` identifies the algorithm used for the following `hash`, which covers the reconstructed `tiling_repr` (or summary groups JSON). The parser uses it to detect missing, reordered, truncated, or modified chunks; it is not used for template selection, score calculation, performance estimation, or encryption. The current producer uses `att_mix64_v1`; the parser remains compatible with historical `sha256` framed logs. The parser validates chunk order/count, UTF-8 byte length, and the hash. Missing
END or failed validation yields `incomplete_final_tiling` without a partial representation.
Repeated identities (`source`, `operator`, `graph`, `result`, `group`, `case_id`, `tiling_key`)
keep the first valid record and mark later records
`duplicate_final_tiling`.

| Input evidence | What it establishes | What it cannot establish |
| --- | --- | --- |
| plog/compiler logs only | Candidate template, case/key, and modeled result | The runtime final choice; do not invent `source="runtime"` |
| plog plus profiling | Modeled candidates compared with measured pipe cycles | Which tiling was written, unless `FINAL_TILING` is present |
| A runtime log containing `FINAL_TILING` | Final group/result/case/key, template, repr, and available estimates | Profiling cycles, which remain separate evidence |

`summary` appends `Final Source`, `Tiling Key`, `Score`, `Pipe Estimate`, `Tiling Repr`, and
`Final Parse Status`. `evidence` writes one JSONL object per final record and preserves
`source_path/source_line`. ATT codegen now emits `FINAL_TILING` after the final selection;
candidate `GetTilingDataRepr` calls are not treated as runtime final records. A cache hit whose
sub-case identity cannot be recovered is intentionally omitted rather than reported with the
wrong template identity.

`topn-templates` reads only an Autofuse worker log and writes static Top1/TopN template CSV and
Markdown tables. Top1 uses every group under `topn_output output_index=0`. If the log contains
`output[i]: perf=`, the command records TopN modeled performance; otherwise that field is left
empty. The command does not infer measured Task Duration from the log and does not read analysis
CSV files produced by an external test framework.

## TEMPLATE log contract

Vocabulary and normalization are defined in `autofuse/common/common_utils.*`. The producer stores the
canonical `semantic_name` in ModelInfo; logging and offline tools consume it without separate role dictionaries
or reclassification.

The core `[ATT][TEMPLATE]` line uses the field `semantic_name` in the format
`<family>/<strategy>[@<role>][#<api-feature>|...][+<transform>...]`. Fields and identity are emitted separately:

```text
[ATT][TEMPLATE] stage=topn_output output[0] group=0 semantic_index=0 semantic_name_count=1 semantic_name="Pointwise/Common"
[ATT][TEMPLATE] stage=topn_output output[0] group=0 semantic_index=0 semantic_name_count=1 fields semantic_schema=1 family="Pointwise" strategy="Common" role="" api_features="" transforms=""
[ATT][TEMPLATE] stage=topn_output output[0] group=0 semantic_index=0 semantic_name_count=1 detail graph=0 result=0 tiling_key=0 template="graph0_result0_g0_0"
```

`name_status` is emitted only as `unavailable` or `ambiguous`; a default unique match omits it.
When a field or an entire line exceeds 700 bytes, the formatter emits `<field>_hash`, `<field>_len`,
`line_hash`, or `line_len` summaries instead of truncating the string. TopN analysis separates rows by
case/visit/shape, the operator in `GenerateTopnSolutions enter`, and the call ordinal of that operator;
multiple kernels or repeated calls in one visit are never merged. If the log has no operator association, the CSV
`association_status` is `unconfirmed` instead of treating the call boundary as confirmed.

## verify-tiling ABI input

Custom input JSON must include an explicit ABI contract. `tf_static` uses no shape dimensions; `tf_dynamic` and `inductor` require one or more shape dimensions. `block_dim_width` is either 32 or 64.

```json
{"dynamic_dims": [], "aiv_num": 48, "ub_size": 196608,
 "abi": {"kind": "tf_static", "shape_dims": 0, "block_dim_width": 32}}
```

```json
{"dynamic_dims": [1024, 512], "aiv_num": 56, "ub_size": 262144,
 "abi": {"kind": "tf_dynamic", "shape_dims": 2, "block_dim_width": 32}}
```

Unknown or missing ABI contracts are rejected before native code is called.

## Using the ATT template/tiling analysis skill

`att_analyze` is maintained in this repository. The skill always invokes
`autofuse/tools/att_analyze/src/att.py`; no checkout of another repository is
required. To analyze an existing run offline from the repository root:

```bash
python3 .claude/skills/att-template-tiling-analysis/scripts/att_analysis.py \
  analyze --run-root <run-root> --output <report-dir>
```

Place user-collected logs in `run-root/default` and `run-root/pgo` (or `base`),
or pass arbitrary variant directories with `--default-root` and `--candidate-root`.
The command recursively discovers logs, profiling, `kernel_meta/`, and `dump/`;
it does not require a producer-specific directory layout. The command
is read-only: it does not invent case scope or commands and does not rerun a
workload. Python 3.9+ is sufficient for the base analysis; installing
`openpyxl` additionally enables `summary.xlsx`.

For live execution, provide the exact cases and command first. Local execution
requires `python3`; remote execution uses standard `ssh` and a checkout on the
remote host. A site-specific `devssh` wrapper is accepted only when explicitly
provided by the user. Build, profiling, PGO, and `verify-tiling` actions require
separate confirmation.

Keep raw evidence and conclusions in separate archives:

```text
run-root/                         # raw run data
  default/  pgo/
    att.log  profile/  kernel_meta/  dump/
evidence-archive/<run-name>/      # raw files
report-archive/<run-name>/        # report.md, summary.csv, root-cause.jsonl, ...
```

The archive helper creates an incremented directory for duplicate run names and
writes `archive-manifest.json`; existing archives are never overwritten. See
the skill references for the execution contract and archive rules.

## Presets and real-log maintenance

`preset_B.json` is an example TensorFlow dynamic-ABI input. Its default
`aiv_num=56` and `ub_size=262144` are not guaranteed hardware specifications
for every chip. `verify-tiling` prints the effective `aiv_num`, its source, and
dynamic dimensions before compilation. Check the value against the target
device and override it with `--aiv-num` or `--input-json` when needed.
`aiv_num` is passed to the TensorFlow tiling ABI; the Inductor ABI does not use
this field.

Logs under `tests/data/` are fixed regression fixtures and are not synchronized
with live runs. When a CANN, TensorFlow, or Inductor log format changes, add a
sanitized fixture from a real run and update the expected `summary`/`evidence`
results while keeping older fixtures for compatibility coverage.
