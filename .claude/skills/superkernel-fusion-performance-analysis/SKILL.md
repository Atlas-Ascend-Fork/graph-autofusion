---
name: superkernel-fusion-performance-analysis
description: >-
  Use whenever Ascend SuperKernel SK-off and SK-on profiling must be compared per
  fused SK, including single-child or shallow SKs, interval-versus-duration-sum
  analysis, performance regression attribution, no-option versus DCCI-disable
  per-SK and child attribution, Cube/Vector scheduling hypotheses, scope keep/prune
  decisions, cross-layer fusion-family consistency analysis, and source-only P range
  recommendations with frozen post-Stage-O options. This read-only skill analyzes
  existing artifacts and returns Chinese evidence; it does not run inference or edit
  scopes.
---

# SuperKernel Fusion Performance Analysis

## Entry Gate

Require two fresh, immutable, distinct collection roots and their manifests:

- `baseline_profile` archives `kernel_details`, `task_time`, and `trace_view`;
- `candidate_profile` archives `kernel_details`, `task_time`, `trace_view`,
  `sk_graph_origin`, `sk_graph_updated`, `sk_fused_nodes`, `sk_scope_split`, and
  `super_kernel` from the profiling process itself. A `sk_prof` trace is optional and
  used only for post-mapping scheduling diagnosis.

`artifact_contract.py validate-set` accepts this exact two-role profile comparison
set. It also accepts the legacy complete four-role audit set when compat/verify were
collected, but never requires those optional audit roles for profile-vs-baseline
analysis.

There is one separate Stage O DCCI regression mode. It compares two fresh SK-on
collections: no DCCI option versus exact
`dcci_disable_on_kernel=[".*"]`. Each side must own
`kernel_details.csv`, complete SK metadata, and a complete
`sk_prof_<device>.json` (or the runtime's equivalent
`sk_prof_x.json` name). Read the parent
[DCCI Stage O protocol](../superkernel-auto-tune/references/dcci-option-tuning.md)
before this mode. Do not feed its candidate-vs-candidate result into the ordinary
SK-off/SK-on schema 1.2 source-action lifecycle.

For this DCCI mode, the two collection sessions and manifests must have been created
before either profiling command and finalized by the collection producer after it
archives its own artifacts. The manifests must bind the exact execution config,
workload, source revision, role/round, launcher identity and timing,
`ASCEND_PROF_SK_ON`, expected profiler outputs, and profile-owned metadata location.
Raw `kernel_details`, `sk_meta`, or `sk_prof` copied into a role root after the process,
including from a shared metadata directory, cannot establish profile ownership and must
not be used to reconstruct a manifest or support DCCI/child attribution. Run
`artifact_contract.py validate-set` before opening the analysis inputs. A missing or
invalid manifest is a collection blocker: report the missing path and validator output,
then require fresh collection from new empty roots; do not emit a DCCI regression table,
child regex, or repair recommendation.

The `candidate_profile` metadata must come from the process that produced its profile;
metadata from a compat or verify run cannot replace it. Require
baseline/candidate config manifests, workload manifests, source revision, round
identity, and analysis-agent ID.
Validate both collection manifests before analysis and revalidate them after
reading the artifacts so the analyzed bytes remain immutable.

The `config` and `control` embedded in both collection manifests are shared
structural-association inputs: they contain only graph targets and collection/runtime
invariants identical across SK-off and SK-on captures. They are not either run's
execution config or the config-derived control object.
Validate the actual SK-off/SK-on configs separately through `--baseline-config`,
`--candidate-config`, and `--declared-change-set`. The validated manifest-set config
and control fingerprints are reported as
`association_protocol.association_config_fingerprint` and
`association_protocol.association_control_fingerprint`; never compare them with
`candidate_config_fingerprint` or the config-derived `control_fingerprint`.
Read [performance-classification.md](references/performance-classification.md) before
classifying any SK. Read [structural-association.md](references/structural-association.md)
before accepting a structural mapping. Read
[regression-diagnosis.md](references/regression-diagnosis.md) before proposing an
option experiment. Read
[source-calibration-mapping.md](references/source-calibration-mapping.md) before
accepting or producing any SK-to-source mapping.
After per-SK classification, read
[cross-layer-family-analysis.md](references/cross-layer-family-analysis.md) whenever
the model has repeated block occurrences. This companion analysis is mandatory for
winner-only profiling with at least three comparable instances of a fusion topology.

## Read-Only Boundary

Do not launch inference, edit source, change a scope, or claim final promotion. Return
`insufficient_evidence` when mapping, occurrences, fingerprints, or artifacts are not
reliable enough for an automatic decision.

Missing profile-process metadata, a missing profile role manifest, or a mixed/stale
profile root is a collection blocker; never fill the gap from another root. A
post-collection manifest assembled from existing raw artifacts is equally invalid and
requires fresh collection.

## Structural Association Boundary

Apply `kernel_projection_trace_v2` to every fused SK before performance analysis.
Project the profile-process `sk_graph_origin` to observable kernel nodes, align each
complete per-stream kernel sequence to every baseline `Step Id`, and materialize fused
children by `(stream role, kernel ordinal)`. `child_count` is descriptive and there is
no minimum child-count filter: counts 1 through 4 use the same projection, uniqueness,
and performance-classification gates as larger fusions.

Do not require an absolute clock offset between `kernel_details` and `sk_prof`.
`kernel_details` uses host-domain timestamps while `sk_prof` may use a device counter.
Bind repeated parent occurrences by the explicit `(device_id, model_id, sk_id,
step_id)` domain. Use `sk_prof` only after the baseline fragment has been mapped, for
optional child scheduling and Cube/Vector analysis.

Raw Task/model/stream/node IDs, generated names, and local op-name windows are
diagnostic evidence only across processes. Never use a raw-ID exact fallback or a
local-name exact fallback. Exact mapping requires a unique injective stream-role
assignment, complete sequence equality for every measured step, and a unique child
ordinal in the baseline trace. Emit
`kernel_projection_structural + exact_projected_trace` only when all gates pass;
ambiguity remains `insufficient_evidence` with an explicit blocker. There is no
whole-graph or absolute-clock fallback after a v2 projection failure.

`exact_projected_trace` identifies a measured graph occurrence, not a source interval.
Never convert `graph_occurrence_fingerprint` or graph position automatically into a
source file, scope, line, or offset. Automatic-AOT evidence remains `proposed`; a
regression may request winner-only source-map completion keyed by the graph occurrence
fingerprint set, but it must not invent source locations or require replacement with a
manual named-scope candidate. Any source edit,
applied/verified lifecycle transition, or final source decision requires an
independent `source_scope_map + exact` mapping with proven source offsets.

The only source-exact protocol is `source_scope_map_v2`. It binds the current
`graph_occurrence_fingerprint` and baseline projection fingerprint to an
`exact_cover` semantic unit reconstructed through a unique original-to-calibration
labeled business-graph isomorphism. The loader must revalidate the archived source
snapshot and actionable revision, at least three runtime
steps with exact assignment for every child of that SK, fused child set, complete
target-unit assignment, collection manifests, and provenance DAG. Full-graph
isomorphism remains mandatory, but unrelated unassigned nodes may be recorded as
skipped; a group whose own child is unassigned remains diagnostic-only while other
groups continue. A group whose children cross block instances is skipped the same way;
it must not abort unrelated groups. It never imports a model adapter. A legacy list,
`task_ranges`, layer
map, repeated signature, or local op window is `diagnostic_only` and can never emit
`source_scope_map + exact`.

The named-scope path uses a `stable_marker` manifest. The automatic-AOT path uses a
marker-free `stable_source` manifest plus a sealed calibration manifest/snapshot and
`marker_only_calibration_v1`; replay must prove that deleting only the recorded marker
insertions recreates the stable source byte-for-byte. Exact ranges then address the
unmarked automatic source. Read
[source-calibration-mapping.md](references/source-calibration-mapping.md) for both modes.

Keep the two scope namespaces distinct throughout this protocol. `source_scope` is
the calibration marker/block binding; `candidate_source_scope` is parsed from the
original fused SK and is the only scope compared with the current performance row.
Both are sealed in normalized evidence. Never compare a marker scope directly with
an original SK scope, and never create a second graph-occurrence hash formula outside
the shared projected-trace helper.

When source mapping is requested but no semantic-unit manifest exists, create a
separate calibration worktree through the registered model/language adapter. The
adapter may only add balanced leaf markers and auditable instance/unit binding; it
must not alter business operators, dependencies, communication, cache behavior, or
control flow. Calibration timing is diagnostic only. Delete temporary markers after
calibration. Named-scope paths retain selected markers only through a committed
stable-marker revision and fresh correctness/profile. Automatic-AOT paths retain no
calibration marker in production and instead require the marker-only bridge to the
frozen stable source before a source action.

## Evidence-Gated Answers

When a prompt supplies summary numbers but no input files, separate the answer into a
conditional calculation and the actual artifact-backed decision. The conditional
calculation may say what the classification and scope action would be if every gate
passed. The actual classification remains `insufficient_evidence`, with `reprofile`

Before any automatic `keep`, `prune`, or source-action recommendation, explicitly account for all
of these gates in the answer, even when the prompt asserts only some of them:

- exact, unambiguous SK-to-baseline mapping;
- baseline and candidate each having at least three aligned occurrences; alignment
  must use a reliable key containing `model_id`, `device_id`, and `step_id` (or
  another explicit occurrence discriminator when `step_id` is absent);
- P50/P90/MAD and the resulting dynamic threshold;
- matching profile, config, and workload fingerprints; and
- a verified declared-change set containing every intended control difference.

Prompt wording, SK-name similarity, a combined occurrence count, or a stated config
value is not proof that the corresponding artifact passed. List every missing gate and
request the original relative artifacts. Keep `child_count` descriptive only.

Compat and verify runs are optional candidate-side reproducibility audits. They may
compare canonical projected fusion inventories, but they are not SK-off mapping inputs
and never gate the measured classification. Do not require signature replay, do not
emit `fusion_replay_identity_unmatched`, and do not downgrade an exact profile-to-
baseline mapping because repeated op signatures cannot be paired across processes.
If independent reproducibility is required for promotion, record it separately or
repeat the instrumented candidate profile; never merge that audit into mapping errors.

When optional child-scheduling diagnosis is requested, reject that diagnosis if
`super_kernel.log` reports `buffer is full, stop dump the time of nodes`; the incomplete
lane set cannot prove Cube/Vector or DCCI behavior. This does not invalidate an exact
projected baseline mapping or its interval classification.

After option tuning has frozen `Sbest-BASE`, screen every performance-exact fused SK for
multi-stream source-reorder opportunities. Preserve the canonical ordered child set,
stream roles, SK-off child intervals, and bound `sk_prof` AIC/AIV lane intervals. Report
whether complementary work was serialized by the SK dispatch order and estimate the
maximum recoverable overlap. A MIX child may be decomposed only when every AIC/AIV lane
is complete and bound to the same parent/device/model/step/range occurrence; otherwise
mark component scheduling `insufficient_evidence`. This screen proposes candidates for
the multistream skill only. It never authorizes a source edit by itself.

For repeated block models, perform the cross-layer fusion-family companion analysis
after every exact per-SK result is available and before summarizing scope actions. A
family uses child topology, not raw SK IDs or layer numbers. Report all family-layer
instances, layer-template coverage, P50/MAD distributions, classification splits,
actual `regressed` instances, and weak-relative-benefit review candidates. A large
child count or a small positive gain is a diagnostic priority only: it never changes
the per-SK classification/action or authorizes a family-wide scope/source change. See
[cross-layer-family-analysis.md](references/cross-layer-family-analysis.md).

`--min-occurrences=1` or `2` is diagnostic input only. The emitted threshold and every
automatic classification/action still enforce baseline and candidate counts of at
least three; lower counts are `insufficient_evidence` with `reprofile`.

After Stage O freezes `Sbest-BASE`, never recommend an Option experiment for P/FINAL.
Option-related signals may remain diagnostic hypotheses, but post-BASE
`recommended_experiments` must be empty. Record source-range evidence and any unresolved
diagnostic hypothesis in `diagnostic_hypotheses` and `next_agent_guidance_zh`; the parent
may create a P plan only for exact `neutral` or `regressed` ranges, with
`declared_option_changes=[]` and fresh validation. The Stage O DCCI combined-repair protocol remains the sole post-screening
Option exception and completes before `Sbest-BASE` is frozen.

## Chinese Human-Facing Report

Read and follow
[chinese-performance-analysis-report-template.md](references/chinese-performance-analysis-report-template.md)
only when producing a detailed Chinese report for an ordinary `S0 SK-off` versus `Sx SK-on`
profile comparison. The template is not a general SuperKernel report format. Do not use it
for SK-on versus SK-on source-reorder or multi-stream experiments, Stage O O0/O1 DCCI
comparisons, correctness, clean performance, compilation, or any other non-profile-vs-baseline
scenario; use that scenario's dedicated analysis/report instead.

For eligible S0/Sx reports, the JSON/CSV outputs remain authoritative for complete per-SK data;
the Markdown report is a compact, reader-facing presentation and must not expand every
occurrence into a large table.

The report must keep these presentation boundaries:

- explain P50, P90, and MAD, and classify by S0 interval P50 versus Sx parent P50;
  `duration_sum` and `union` are diagnostic only;
- do not display aligned step IDs, per-step child durations, or `S0 duration_sum P50`
  in reader-facing tables; retain raw samples in machine-readable artifacts;
- for repeated-block models, use identical family labels in the layer-coverage view and
  the Fusion Family performance table;
- list each model layer's fused family set, and summarize each family's covered layer
  ranges and all-layer coverage, only when `source_scope_map_v2 + exact` proves the
  source-layer mapping;
- without that mapping, explicitly mark layer coverage as `unproven`; do not infer layer
  membership, coverage percentage, or all-layer presence from occurrence counts, raw IDs,
  or graph ordinals;
- include fusion-internal child/lane scheduling diagnosis only with complete, bound
  `sk_prof`; a buffer-full or incomplete trace is a diagnostic blocker, not a root cause;
- do not add an end-to-end clean-performance summary or a fixed Top-N
  regressed/weak-benefit list to this profiling report.

## Required Outputs

Write `profiling-analysis-result.json` and Chinese `PROFILING_ANALYSIS.md`. Use only
relative artifact paths. Include `analysis_agent_id`, all fingerprints, thresholds,
per-SK decisions, scope actions, hypotheses, recommended experiments, blockers, and
`next_agent_guidance_zh`. Every scope action preserves `source_scope`, `boundary`, and
`ordered_child_op_sequence`. A source interval is proven only by a non-empty relative
source file without `..` segments and valid increasing start/end offsets; otherwise
emit `interval_unproven=true` so strategy handles one range per P round instead of
inferring order from op names.
When repeated block occurrences exist, also write the diagnostic companion
`FUSION_FAMILY_LAYER_ANALYSIS.md`, `fusion-family-summary.csv`, and
`fusion-family-layer-comparison.csv`. They must carry input fingerprints and remain
separate from the schema 1.2 authority: neither a family split nor a weak-benefit
warning mutates `per_sk_decisions` or `scope_actions`.
For every screened multi-stream SK, also preserve the source-mapped statement order,
child-to-statement binding, stream roles, resource/lane identity, observed overlap, and
the proposed overlap direction. Do not translate a sub-microsecond timestamp tie into
a source-order claim; use normalized dispatch identity and dependency evidence.
The renderer requires at least one per-SK decision and an exactly matching scope
action for every unique `(sk_id, range_id)`. It rejects duplicate identities,
classification/action pairs outside the schema enum, and any mismatch in source
scope, boundary, ordered child sequence, or interval proof. It also recomputes
  `analysis_id` from the analyzer identity fields and requires `round_id` to belong to
  the declared candidate's `BASE/Pn/FINAL` family.

When distinct native SK inventory entries share the same derived fusion identity,
disambiguate their report IDs with a unique `graph_occurrence_fingerprint`. If no
unique structural occurrence exists, only `reprofile` or `block` entries may use an
explicit `native_inventory_diagnostic` discriminator; it is report-local evidence and
must not be treated as cross-process identity. Never use a raw runtime ID to diagnostic a
`keep` or `prune` decision.

The analyzer exposes only the complete profile-vs-baseline schema 1.2 CLI. Do not add
inventory-only, candidate/replay, or partial-profile compatibility modes.

For the separate Stage O DCCI regression mode, write
`dcci-regression-analysis.json` and Chinese
`DCCI_REGRESSION_ANALYSIS.md` instead of fabricating a schema 1.2
profile-vs-baseline result. Include immutable input fingerprints, cross-condition
structural matching statistics, every SK's P50/P90/MAD classification, every
significantly slower SK's ordered child table, the stable union of significantly
slower canonical child ops, complete symbol evidence, and the exact identical
before/after regex list proposed to the execution owner. This output is diagnostic:
it has no scope action, source action, promotion, or ledger authority.

If the analysis request has started and a manifest check, analyzer, renderer, schema
validation, or other tool step fails, do not fabricate these complete outputs. Preserve
the immutable request, exact argv, stdout/stderr, exit status, partial outputs, and last
passed/failed gate, then return them to the owning profiling round under the parent
[failure-scene-reporting contract](../superkernel-auto-tune/references/failure-scene-reporting.md).
The owner must append that scene to its experiment-local report before retrying. A
parent summary is not a substitute, and a later successful analysis must not overwrite
the failed attempt.

Run the complete analyzer from this skill directory. First produce the authoritative
fused-child mapping:

```bash
python3 scripts/projected_trace_mapping.py \
  --baseline-kernel-details artifacts/S0/profile/profiler/kernel_details.csv \
  --profile-collection-manifest artifacts/S1/profile/association-artifact-manifest.json \
  --device-id 0 --model-id 48 \
  --json-out output/projected-trace-mapping.json \
  --markdown-out output/PROJECTED_TRACE_MAPPING.md
```

Require `source_kernel_nodes == baseline_rows_per_step[step]`, zero alternative
stream-role solutions, at least three steps, and `exact_projected_trace` for an SK
before using its baseline interval. Then run the performance classifier:

```bash
python3 scripts/analyze_fusion_performance.py \
  --baseline-profile artifacts/S0/profile/profiler/kernel_details.csv \
  --candidate-profile artifacts/S1/profile/profiler/kernel_details.csv \
  --sk-meta artifacts/S1/sk_meta \
  --baseline-config artifacts/S0/config.json \
  --candidate-config artifacts/S1/config.json \
  --baseline-workload artifacts/S0/workload.json \
  --candidate-workload artifacts/S1/workload.json \
  --declared-change-set artifacts/S1/declared-change.json \
  --candidate-name S1 --experiment-id exp-1 --round-id S1-BASE \
  --analysis-agent-id analysis-agent-1 --source-revision REVISION \
  --sk-prof artifacts/S1/profile/profiler/sk_prof_device_0.json \
  --environment-evidence artifacts/environment.json \
  --source-scope-map artifacts/source-mapping/source-scope-map-v2.json \
  --source-root worktrees/stable-marker \
  --baseline-collection-manifest artifacts/S0/profile/association-artifact-manifest.json \
  --profile-collection-manifest artifacts/S1/profile/association-artifact-manifest.json \
  --json-out output/profiling-analysis-result.json \
  --markdown-out output/PROFILING_ANALYSIS.md
```

All analyzer inputs and outputs recorded in the result must be relative artifact
paths. The two manifests do not relax any existing config, workload, declared-change,
profile, environment, or source-map gate.

The optional `--sk-prof` value above is a diagnostic child-scheduling trace. It is not
an input to baseline child mapping and does not need an absolute clock offset to
`kernel_details`.

Regenerate the deterministic Markdown from an existing result with:

```bash
python3 scripts/render_fusion_performance_report.py \
  --json-in output/profiling-analysis-result.json \
  --markdown-out output/PROFILING_ANALYSIS.md
```

## Diagnostic Contract

`diagnostic_hypotheses` entries contain `kind`, `confidence`, one `range_id`,
`evidence`, `explanation_zh`, and `requires_ab_test`. They never contain
`root_cause`. For post-BASE P/FINAL analysis, emit
`recommended_experiments=[]`; never encode an Option trial there. A source-only source-action idea
stays diagnostic until the parent freezes and validates a separate source-action plan.

DCCI state used for mechanism diagnosis must come from explicit config/environment
`runtime_evidence.dcci_state`; scalar/cache counters cannot supply it. Merge all
explicit sources: unknown does not override a known state. Only after a pruned range
enters the DCCI scalar/cache counter gate do conflicting known states block DCCI
hypotheses and all-unknown evidence add a Chinese blocker. Without that
counter evidence, no unconditional DCCI-state blocker is required. These states do
not change interval classification or an otherwise exact scope action.

Cube/Vector serialization is high confidence only when the baseline overlaps and an
associated child trace contains complete CUBE and VECTOR events with an explicitly
nonparallel overlap result and reliable stream identity for every CUBE/VECTOR event.
Missing, empty, incomplete, unassociated, or stream-ambiguous child traces produce a
per-range low-confidence hypothesis and Chinese blocker, with no
`auto_op_parallel` experiment. In the separate Stage O protocol, never recommend standalone DCCI
before-only or after-only trials. Analyze exact global disable `[".*"]` first.
Only after measured disable-all regression, unique all-SK matching, and complete child
traces may the analysis return one stable deduplicated union of significantly slower
child op regexes. Every regex must compile, match a full metadata child symbol, contain
a literal full symbol or canonical op token, and avoid empty/unrelated negative-control
matches. The exact same accepted list is used for both
`dcci_before_kernel_start` and `dcci_after_kernel_end` while
disable-all remains set. This combined repair is evaluated directly against no-option
O0 by the execution owner. Experiment IDs for ordinary P recommendations still bind
the range, option, and canonical value fingerprint. The renderer accepts only complete
schema 1.2 reports with identity, fingerprints, inputs, hard thresholds, all four named
output lists, blockers, next guidance, and a matching recomputed
`analysis_content_fingerprint`. It escapes HTML/Markdown payloads and rejects
conflicting or unresolvable paths.

The schema 1.2 analyzer and renderer only read input artifacts and write the requested result
files. They do not run inference, edit a scope, apply an option, or claim promotion.
