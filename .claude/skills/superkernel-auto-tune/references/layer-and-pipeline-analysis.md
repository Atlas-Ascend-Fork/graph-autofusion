# Layer, Fusion, And Pipeline Analysis

Use this reference to reconstruct the network and explain performance. Structural
metadata narrows hypotheses; measured profiling decides keep/prune.

## Required Questions

1. Which operators and streams execute in every repeated layer?
2. Which child operators, ranges, and streams belong to every generated SK?
3. Where do scope, resource, dependency, runtime-task, and unsupported-op breaks occur?
4. Which fused source intervals improve, remain neutral, or regress?
5. Did baseline Cube/Vector parallelism survive inside SK?

Do not answer these from aggregate latency or child depth alone.

## Required Evidence

Under identical workload controls collect:

- SK-off `kernel_details.csv` for model inventory and source intervals;
- SK-on `kernel_details.csv` only for the selected winner and its BASE plus explicitly requested P/FINAL rounds;
- candidate profile process 自有 `sk_meta`；
- `baseline_profile` 与 `candidate_profile` 两份 collection manifest；
- optional `sk_prof_device_<deviceId>.json` for child scheduling;
- source-confirmed task/range map;
- active wrapper option probe.

Profiler, metadata, child trace, and debug artifacts are diagnostic. Disable them for
clean timing.

## Candidate Lifecycle

Stage A reconstructs the model primarily from source plus the single SK-off diagnostic
profile, then screens S candidates with correctness and clean timing. It does not map
individual fused SKs. After one winner is selected, for every winner profiling round:

1. execution and correctness pass;
2. candidate diagnostic profile and its own complete metadata are collected and fingerprinted;
3. both profile collection manifests validate;
4. a fresh read-only analysis Agent classifies every reliably mapped SK against SK-off;
5. only the analysis result drives BASE actions and any explicitly requested P/FINAL actions.

Non-winning S candidates stop before this lifecycle. `deep_fusion_reproducible`, child count, depth histogram, and fragmentation are
descriptive topology fields. They never decide whether profiling runs or whether a
range is beneficial.

## 1. Reconstruct Every Layer

Use the sibling performance analyzer's schema 1.2 profile-vs-baseline workflow. Do not
use partial-profile or inventory-only compatibility CLI modes.

A complete inventory includes:

- expected repeated-layer count confirmed from source/config;
- every layer's ordered `operator_sequence`, task range, op state, core family,
  duration distribution, and stream IDs;
- per-layer Cube/Vector/MIX counts and overlap;
- embedding, final norm, LM head, sampling, communication, and helper work outside
  repeated layers;
- static-kernel ratio and explicit unknown/non-static rows.

If `layer_analysis.status` is partial/unavailable, provide a source-confirmed map:

```json
{
  "task_ranges": [
    {"layer": 0, "model_id": "48", "start_task_id": 3, "end_task_id": 39},
    {"layer": 1, "model_id": "48", "start_task_id": 40, "end_task_id": 76}
  ]
}
```

Missing layer names are evidence gaps, not zero-work layers.

## 2. Inventory Every SK

```bash
python3 <runtime-skill-dir>/scripts/analyze_sk_meta.py \
  experiments/S3/S3-BASE/compat/sk_meta \
  --json-out experiments/S3/S3-BASE/sk-meta-summary.json \
  --round-name S3-BASE \
  --effective-min-child-nodes 1 \
  --round-report-out experiments/S3/S3-BASE/round-report.json
```

For every SK retain:

- model ID, generated function, source scope, layer/segment, exact boundary;
- declared/parsed child count and `count_reliable`;
- full ordered child operator list and kernel/core types;
- task/block counts, stream IDs, MIX split, and `isScheModeOn`;
- source fragments, break reasons, and compat/verify occurrence counts.

Per layer retain SK count, child-count min/P50/P90/max/histogram, single-child count,
scope fragments, streams, breaks, disconnect metrics, and control-core state. This
preserves metadata observation and helps explain launch/fragmentation behavior.

Do not translate these structural fields into automatic keep/prune decisions.

## 3. Interpret Fragmentation

Deduplicate repeated `sk_scope_split.log` snapshots by source, scope, trigger, and
reason. Keep raw, duplicate, and unique counts. `fused_groups_per_100_child_nodes`
measures fragmentation, not fusion coverage.

Compare every layer; aggregate means can hide first/last-layer exceptions. Use source
boundaries and profiler occurrence mapping before acting on a fragment.

Common break phases:

1. scope placement, unintended outside regions, excess names, resource boundaries;
2. an option with a matching direct reason and accepted exact value;
3. unsupported/custom/non-static/runtime-task work for operator adaptation.

Aggressive options do not replace dependency proof.

## 4. Classify Per-SK Performance

Dispatch the fresh analysis Agent to
[superkernel-fusion-performance-analysis](../../superkernel-fusion-performance-analysis/SKILL.md).
The primary comparison is:

```text
baseline interval P50 = P50(max(child end) - min(child start))
candidate SK P50 = P50(SK duration)
improvement = baseline interval P50 - candidate SK P50
```

Duration sum P50 is secondary because multi-stream child durations overlap. Use P90,
occurrence count, MAD threshold, mapping confidence, and fingerprints.

Actions:

- `beneficial/keep`;
- exact reliable `neutral/regressed -> prune`;
- `insufficient_evidence -> reprofile|block` without scope change.

Every human performance table includes interval P50, duration sum P50, SK P50, MAD
threshold, classification/action, mapping confidence, analysis Agent, and conditional
evidence.

## 5. Analyze Cube/Vector Scheduling

Baseline profiling identifies cross-stream C/V overlap. Candidate SK child trace must
have complete CUBE and VECTOR events with reliable stream IDs before claiming
serialization or preserved overlap.

| Evidence | Interpretation | Next experiment |
|---|---|---|
| No baseline C/V overlap | No parallelism to preserve | Keep performance-driven scope search |
| Baseline overlap and SK remains parallel | Scheduling preserved | Keep only if interval is beneficial |
| Baseline overlap but SK serializes | Direct scheduling hypothesis | One-range `auto_op_parallel=1` diagnostic if accepted |
| Child trace incomplete/ambiguous | Evidence insufficient | Reprofile; no scope change |
| MIX/Cube or same-resource competition | Resource hypothesis | One exact boundary experiment |

Manual reordering is a fallback:

1. prove producer/consumer, event, communication, cache mutation, and barrier edges;
2. reorder only independent work;
3. preserve deterministic scopes and stream semantics;
4. archive source diff and generated task order;
5. repeat correctness, fresh profile-vs-baseline analysis, and FINAL interaction check;
   run any candidate reproducibility audit separately.

## 6. Diagnose DCCI

DCCI is a hypothesis only when:

- the range is neutral/regressed;
- direct scalar/cache profiler evidence exists;
- explicit config/environment provides DCCI state;
- a narrow exact accepted value matches the target child symbol.

After Stage O, keep DCCI and all other Option values frozen. P/FINAL may retain DCCI
as a diagnostic hypothesis but must not schedule a DCCI or other Option experiment.
Conflicting or unknown DCCI state does not change interval classification.

## 7. Feed The Winner Lifecycle

Only the clean-timing winner enters this default lifecycle: complete BASE and required
SMAP settlement, then `whole_scope_clean_validation`. `P/FINAL` is an optional
source-range branch that starts only on an explicit user or frozen experiment-plan
request; BASE or SMAP completion alone never starts it. An absent, invalid, `no_gain`,
`blocked`, or `failed` optional branch preserves the incumbent and does not block
whole-scope validation. When requested:

- BASE retains beneficial and proposes exact neutral/regressed ranges for P.
- P removes only non-overlapping exact source ranges while preserving execution.
  it never changes an Option.
- FINAL combines retained/retained ranges and freshly checks interactions.
- S1 automatic AOT keeps all results proposed until manual re-test under matching
  conditional fingerprints.

Store all paths, content fingerprints, analysis Agent IDs, declared changes, and
decisions in schema 2. Write the layer/pipeline interpretation and blockers in Chinese.
