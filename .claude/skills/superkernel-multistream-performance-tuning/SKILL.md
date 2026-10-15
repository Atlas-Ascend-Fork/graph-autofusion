---
name: superkernel-multistream-performance-tuning
description: Use after an Ascend SuperKernel winner has fresh per-SK profiling when fused multi-stream work lost overlap or has scheduling headroom. Runs an optional isolated tuning branch and returns either a validated incremental clean-performance winner or the unchanged incumbent; it does not perform initial S-candidate screening or generic per-SK classification.
---

# SuperKernel Multistream Performance Tuning

Treat multi-stream tuning as an optional winner-only branch. Preserve the validated
incumbent unless a trial proves a repeatable incremental end-to-end gain. Restoring
Cube/Vector overlap or improving one SK is mechanism evidence, not an acceptance gate.

This branch may be entered during the full controller flow or from a new derived
session created with `superkernel-auto-tune derive-session --optional-mode multistream`
or `both`. The derived entry must bind a completed verified parent, its sealed BASE
handoff, schema 1.2 analysis, and matching schema 2 ledger. User approval does not
bypass those gates, and the parent session remains immutable.

## Entry Gate

Accept work only after the parent `superkernel-auto-tune` flow has frozen
`Sbest-BASE`, passed correctness, collected fresh profile-owned metadata, and received
a valid schema 1.2 result from `superkernel-fusion-performance-analysis`.
Source-reorder work also requires exact child-to-statement source mapping. A MIX
component exception that spans multiple functions requires a sealed multi-span exact
map and reviewed component materializer; one `source_scope_map_v2` boundary is not a
substitute. Read
[component-overlap-reorder.md](references/component-overlap-reorder.md) when the target
is an operator-order or component-overlap trial.

Require either the legacy `superkernel-multistream-request-v1` for option/scope/reorder,
or `superkernel-multistream-request-v2` validated by
`scripts/multistream_critical_path_contract.py validate-request` for event/stage tuning. Read
[evidence-contract.md](references/evidence-contract.md) before collecting or
interpreting multi-stream evidence.

Do not run for Stage A S candidates, the pre-profile Stage O sweep, non-winners, or a
candidate without a stable five-process incumbent. Do not repeat an identical global
option/value already settled by Stage O.

Before the first NPU command, profile recollection, or trace-budget attempt in this
branch, run the mandatory adapter capability preflight in
[adapter-capability-preflight.md](references/adapter-capability-preflight.md). Freeze the
complete requested action set, validate the network capability matrix, freeze a
run-specific collection plan, and require a replayable preflight receipt with
`status=ready`. The plan must declare `round_kind=multistream`, bind semantically
validated source/config/control/workload identity JSON files, and bind a valid model run
spec. For reorder work this plan must bind the raw capture plugin, identity,
source-mapping and dependency providers, reorder materializer, four-profile launcher,
correctness/clean validators, and every required output. A missing or blocked capability
is a pre-experiment blocker: do not launch inference, do not collect a speculative trace,
and do not consume the three-attempt recollection budget. Never use a generic conformance
or resource-screening adapter as proof that the current network supports source reorder.

## Isolation And Fallback

The request must name a dedicated source worktree, experiment root, config root, and
cache namespace. Never edit the incumbent source/config, overwrite its artifacts, or
merge directly into the parent's schema 2 ledger. Serialize device use with the parent
and preserve the frozen cache policy; never clear a shared cache as cleanup.
Use the parent skill's shared NPU lease protocol and inventory. A matching parent lease
must be inherited rather than reacquired; a direct device phase acquires the same
per-device locks, and four-profile orchestration requires the parent marker.

Every branch starts from the immutable incumbent. Return one of:

- `accepted`: a candidate passed the complete incremental gate;
- `no_gain`: trials were valid but none beat the incumbent;
- `blocked`: required multi-stream evidence or a legal action was unavailable;
- `failed`: execution, correctness, collection, validation, or tool execution failed.

For every status except `accepted`, return `incumbent_unchanged=true`, no selected
candidate, and the exact incumbent as fallback. A missing, malformed, stale, or
unvalidated result has the same fallback semantics. The optional branch never blocks
the parent's unchanged `Sbest-BASE` promotion or ordinary P path. If the branch
discovers that an incumbent artifact or fingerprint was already invalid, report
`incumbent_evidence_invalid`; the parent must revalidate that pre-existing evidence.

Use `scripts/multistream_contract.py fallback` to produce a deterministic non-accepted
result when the branch cannot continue.

## Evidence And Classification

Reuse the sibling performance skill rather than implementing a second generic SK
mapper. Require exact projected occurrence mapping and at least three aligned
occurrences. A short-window `sk_prof` trace is required for a direct internal scheduling
claim and must bind events to the target parent SK/range with complete lane identity.
An overflowed or unbound trace yields `blocked`, not a guessed serialization result.

Before selecting targets, run `scripts/multistream_opportunity_discovery.py discover`
against the validated schema 1.2 profiling result. Screen every performance-exact
multi-stream SK with at least three occurrences on both sides, including `beneficial`
SKs; never limit multi-stream screening to `neutral` or `regressed`. After the bound
SK-off/SK-on short-trace analysis, run `audit-coverage`. A missing beneficial target,
extra unbound target, trace blocker, or incomplete target set means the latent-
opportunity screen is incomplete and must not be reported as "no opportunity".

Treat `beneficial + degraded` as a first-class latent opportunity: fusion may reduce
launch or child work enough to hide lost overlap. Preserve the beneficial incumbent
while investigating the lost parallelism. Separate observed parallelism loss from
actionability: loss of any stable cross-stream overlap may produce
`parallelism_effect=degraded`, but only a stable lost pure `Cube <-> Vector` overlap can
produce `optimization_status=opportunity` through the ordinary route. Whole-MIX,
same-engine, wait, or communication overlap remains `degraded + blocked`; only a MIX
target satisfying every component-aware gate below may use the exception route.

Business-operator reorder is a multistream-only action. Authorize the ordinary pure
resource route only when at least
three aligned SK-off occurrences prove a stable overlapping `Cube <-> Vector` pair on
distinct reliable streams, and matching SK-on `sk_meta` entries prove a stable dispatch
inversion for the same operator identity. Outside the component-aware exception, `Mix`
with any family, `Cube <-> Cube`,
`Vector <-> Vector`, communication/wait, single-stream targets, incomplete stream
identity, generic performance regression, and operator-name similarity must never
authorize a reorder: no data dependency does not imply executable parallelism.
The one exception is the component-aware route in
[component-overlap-reorder.md](references/component-overlap-reorder.md): a MIX business
operator may participate only when complete bound short traces expose all of its AIC/AIV
components, one component forms the intended complementary overlap, the remaining
component's same-engine contention is explicitly measured, and the full hard dependency
DAG permits the source permutation. Do not relabel the whole MIX operator as Cube or
Vector.
Normalize SK-off execution rows and SK-on dispatch entries through
`scripts/multistream_operator_order.py`; do not compare raw CSV/log ordinals directly.
Preserve the input row order from `kernel_details.csv` as the SK-off execution sequence.
Use `Start Time(us)` and `Duration(us)` only to prove overlap; sub-microsecond start
ties or reversals must not replace the profiler's row-order sequence.
Before producing that capture, build and require a complete identity registry with
`scripts/multistream_identity_binding.py`; read
[identity-binding.md](references/identity-binding.md). A registry ambiguity or missing
domain/occurrence is a blocker, never an invitation to fall back to names or ordinals.
Build hard dependencies through `scripts/multistream_dependency_evidence.py`; read
[dependency-provider.md](references/dependency-provider.md). Operator-order capture v3
must bind complete provider coverage and an acyclic audited hard-edge union. Missing
coverage or an unbound hand-written edge blocks reorder authorization.
Derive every SK-off resource family from the sealed `kernel_details.csv` triple
`Accelerator Core + Block Num + Mix Block Num`; never infer it from operator names or
only from AIC/AIV trace lanes. `AI_VECTOR_CORE` is Vector and `AI_CORE` is Cube.
`MIX_AIV` is Vector only when `Mix Block Num=0`, otherwise Mix; `MIX_AIC` is Cube only
when `Mix Block Num=0`, otherwise Mix. The capture must preserve all three raw fields,
and the analyzer recomputes `core_family` rather than trusting a producer label.

For model-level event/stage optimization, read
[critical-path-optimization.md](references/critical-path-optimization.md). This route
uses a sealed logical graph and critical-path capture to identify a stable critical
join, then tries event-edge refinement before stage splitting. Cross-parent work is
permitted only for this graph-level route; it never relaxes the same-parent requirement
for business-operator reorder. A branch with positive slack is already hidden and is
not a candidate. `Mix`, same-engine work, unbound event reuse, fewer than three complete
occurrences, or a predicted P50 E2E upper bound below 3% must return
`no_event_or_stage_candidate`, not a speculative source edit.

Keep these axes independent:

```text
net_effect: beneficial | neutral | regressed | insufficient_evidence
parallelism_effect: improved | preserved | degraded | unknown
optimization_status: no_action | opportunity | validated | blocked
```

`beneficial + degraded` means keep the incumbent and optionally tune it. It must never
be converted to prune merely because overlap was lost. Read
[classification.md](references/classification.md) for metrics and evidence levels.
Use `scripts/multistream_trace_analysis.py` for the bound short-window comparison; the
sibling analyzer's diagnostic-only child summary is not direct occurrence evidence.
Freeze the bounded trial matrix with `scripts/multistream_candidate_planner.py`. Read
[automation-pipeline.md](references/automation-pipeline.md) for schemas and commands.
Model-specific raw profiler or metadata parsing must run behind
`scripts/multistream_capture_producer.py`. Bind the plugin and every raw input by
SHA256; the host seals the capture and replays the standard analyzer before returning a
production receipt. Never add a model name, fixed model path, or model operator rule to
the generic analyzers.

Before declaring a new network class supported, run
`scripts/multistream_network_conformance.py` with at least two sealed adapters. One case
must bind a real-NPU closure; a second network may legitimately return
`no_reorder_candidate` when normalized screening proves zero stable same-parent matches.
The report must bind one unchanged generic-core fingerprint and contain no adapter ID in
those core files. Read [network-conformance.md](references/network-conformance.md).

After conformance, build the adapter capability declaration and action matrix with
`scripts/multistream_adapter_support.py`; read
[adapter-support-matrix.md](references/adapter-support-matrix.md). Never infer network
support from its name or from an informal experiment report. Every supported capability
must cite sealed evidence; every unsupported capability and unavailable action must
return a structured blocker. Matrix availability permits planning only and does not
replace candidate-specific gates.

## Isolated Trial Loop

Read [experiment-lifecycle.md](references/experiment-lifecycle.md) and
[action-execution-and-handoff.md](references/action-execution-and-handoff.md). Read
[cleanup-and-rollback.md](references/cleanup-and-rollback.md) before disposing of any
terminal trial. Every terminal state must run the frozen cleanup plan. Quarantine only
explicit mutable targets inside the isolated branch, preserve audit artifacts, and
verify the immutable incumbent before and after cleanup. Never recursively delete a
trial root or automatically delete quarantine content.

Read
[model-adapter-schema.md](references/model-adapter-schema.md) before freezing a model
adapter. Freeze a bounded trial matrix before execution. A trial changes exactly one factor:

- one exact wrapper-accepted option at one RFC6901 pointer; or
- one analyzer-validated `source_scope_map_v2 + exact` source boundary; or
- one exact source range's fusion participation; or
- one dependency-safe permutation of complete statements inside one exact range, only
  when direct multistream evidence authorizes `dependency_safe_operator_reorder`.
- one reviewed cross-function, multi-hunk `component_overlap_reorder`, only when the MIX
  component exception has a complete bound AIC/AIV capture and exact AST source spans.
- one analyzer-authorized `event_edge_refinement` or `stage_split` inside the model's
  multi-stream-only source path.

Option-only trials do not require a source map. Scope split, range exclusion, and
operator reorder always require a source map with proven file offsets. Never combine an
option and source edit in one trial. Reorder only complete, contiguous, explicitly
movable and side-effect-free statements while preserving every DATA, STREAM_ORDER,
event, wait, barrier, communication, cache-mutation, side-effect, and control-flow edge.
Outside a directly proven multistream target, never reorder business operators.

Use the structured source-transform materializer for event/stage actions. It validates
every original byte range, declared multi-stream source ranges, the analyzer action
identity, action-specific safety proofs, post-transform dependency evidence, explicit
stage state, and the unchanged single-stream projection fingerprint. Event refinement
is immediate only when delayed EventNotify evidence exists. A stage split is immediate
when independent stage-delay evidence exists; when the same join also has an event
candidate it becomes a derivative. Scope derivatives require the bound parent trial to
be correct, mechanism-validated, clean3 non-regressing, and below the incremental-gain
threshold, and must report clean comparisons against both the parent and incumbent.
Never combine a parent and derivative factor in one trial.

For a MIX component exception, use
`scripts/multistream_component_reorder.py`; never route it through ordinary
`materialize-reorder` or event/stage `stage_split`. The component map must bind every
affected target to exact AST spans, including operator calls, synchronization/join
statements, and at least two cross-function transform regions. Validate that map against
`artifacts.component_immutable_source_root`, a read-only tree distinct from the mutable
candidate `isolation.source_worktree`, so result replay keeps the original exact mapping.
The component capture
must contain all AIC/AIV intervals for at least three occurrences per target, preserve
the whole-MIX identity, measure remaining same-engine contention, and prove the current
dispatch inversion. The reviewed transform must bind every hunk's original bytes,
preserve the hard-dependency and single-stream projection fingerprints, and emit an
action-manifest-v2. Any missing span, lane, target, proof, or post-transform audit is
`blocked`.

Use route 2 first: generate one deterministic minimal permutation that repairs stable
dispatch inversions under the hard dependency DAG. Freeze route 3 alternatives with a
bounded topology search, but activate them only after route 2 has changed dispatch order
and produced stable, non-regressing clean3 evidence below the incremental-gain threshold.
Route 3 materialization must bind that route 2 dispatch evidence and rejected clean3
semantic evidence.

For each trial:

1. Materialize the only change with `scripts/multistream_execution.py` in the dedicated
   config/worktree and archive its action manifest. Do not write
   `single_change_verified` by hand.
2. Generate the first reviewed adapter from a model run spec with
   `scripts/multistream_adapter_generator.py`, or freeze an existing reviewed draft with
   `scripts/multistream_plan_compiler.py freeze-adapter`. Then compile the trial-bound
   five-phase plan with `compile`. Archive
   both the adapter and compilation manifest. Direct `freeze-plan` is diagnostic-only;
   accepted trials must replay to the adapter compilation. Do not pass a shell command
   string or inherit an ambient environment implicitly.
3. Run the frozen plan with `scripts/multistream_runner.py run`, using the same explicit
   device lease root as every cooperating parent process. The runner executes full
   correctness before profiling, kills a timed-out process group, seals command and
   validator logs, hashes required artifacts, and advances the trial state only after
   the phase manifest is durable.
   Every adapter phase validator must invoke `scripts/multistream_evidence.py` with the
   phase-matching subcommand. The compiler rejects arbitrary validators and binds the
   standard validator, its interpreter, output path, trial, request, and input paths
   into the frozen plan.
   A reorder plan must freeze `dispatch_order_evidence` as pre-profile evidence. After
   correctness, generate it with `scripts/multistream_operator_order.py verify-dispatch`.
   The runner blocks the profile command unless deterministic replay proves the intended
   statement dispatch order in at least three occurrences, preserves the child set, and
   retains at least two reliable streams in every occurrence.
   An event/stage plan must instead freeze `event_stage_dispatch_evidence`. The runner
   blocks profile unless at least three aligned occurrences prove the analyzer-authorized
   logical EventNotify or stage dispatch moved earlier on a preserved multi-stream window.
4. Collect a fresh candidate profile and complete profile-owned metadata. Include the
   bounded short scheduling trace in the profiling phase when testing internal
   parallelism. For reorder trials, retain the incumbent SK-off/SK-on pair and collect
   the reordered SK-off/SK-on pair; all four profiles are mechanism evidence.
   Freeze and run this matrix with `scripts/multistream_four_profile.py`. Execute it
   only inside the runner's device-held profile phase; require four passed role
   manifests and a replayable four-profile summary before mechanism analysis.
5. Dispatch a fresh read-only sibling performance-analysis Agent and inspect every
   reliable SK for spillover, not only the target range.
6. Run at least three clean processes against the stable incumbent. A potentially
   accepted trial must be extended to exactly five and pass stability, strictly positive
   mean and median-run direction, P90, and standard-deviation gates. Multi-stream source
   trials have no fixed percentage-improvement floor: configure
   `min_improvement_pct=0.0`; zero is not a gain and is rejected.
7. Roll back the branch state after every rejected, blocked, failed, or incorrect trial.
   Before rollback, retry, or cleanup of a non-environment failure, read and enforce the
   parent skill's
   [failure-scene-reporting.md](../superkernel-auto-tune/references/failure-scene-reporting.md).
   Seal the attempt artifacts and append the complete failure scene to this branch's
   `MULTISTREAM_TUNING.md`; the parent report is not a substitute. Pre-experiment
   environment failures remain environment evidence and do not create a failed trial.

Create one `superkernel-multistream-trial-state-v1` per trial and advance it one gate at
a time. Every accepted result must bind the replayable accepted state and action
manifest. For source boundary actions, use only the bounded insertion executor described
in the reference. For ordinary reorder actions, use only `materialize-reorder` with an
analyzer-generated route 2/3 order. For the MIX exception, use only the reviewed
`component_overlap_reorder` transform named by the frozen adapter. An unsupported edit
is blocked rather than performed manually.

The clean validators use an explicit exit-action contract. A valid three-run no-gain
result maps to `rejected` and stops before clean5; it is not an execution failure. The
runner stops at `clean5_passed`; it never promotes a trial by itself. Advance to
`accepted` only after the result contract's incremental gates are evaluated. A durable
passed phase manifest with an unadvanced state is recoverable without rerunning. Logs
or outputs without a complete manifest are an ambiguous attempt and must fail closed.
The standard validator emits compact semantic evidence without raw timing sample arrays.
Result-v2 validation rechecks its identity, decision, content fingerprint, and every
declared source-file SHA256 in addition to the enclosing phase manifest.

Limit short-trace recollection to three attempts per logical trial and preserve every
failed attempt. Stop when the request budget is exhausted.
The attempt budget starts only after the adapter capability preflight is `ready`; a
missing producer, provider, materializer, validator, or declared output is not a trace
attempt.

## Acceptance And Parent Handoff

Only clean incremental end-to-end gain can select a candidate. Local overlap recovery,
target-SK interval improvement, launch reduction, or a profiler counter cannot replace
that gate. Report mechanism validation separately when local and end-to-end signals
disagree.

Validate the final `superkernel-multistream-result-v2` with:

```bash
python3 <skill-dir>/scripts/multistream_contract.py validate-result \
  experiments/multistream-request.json \
  experiments/multistream-result.json
```

For the first real-NPU qualification of an adapter or a corrected generic mechanism,
seal the entire executed trial with `scripts/multistream_npu_closure.py`. The closure
requires request, authorized operator-order capture, materialized action, passed
post-reorder dispatch evidence, all four profile roles, clean3 evidence, and a validated
result. It accepts only `accepted` or a `no_gain` result containing the matching rejected
trial; `no_reorder_candidate` cannot qualify a corrected tuning loop. See
[real-npu-closure.md](references/real-npu-closure.md).

Event/stage capability uses the separate
`scripts/multistream_critical_path_npu_closure.py` validator. It requires request-v2,
action-manifest-v2, effective event/stage dispatch evidence, all four profile roles,
effective join validation, clean evidence, result-v3, and cleanup replay. A legacy
reorder closure must never be cited as event/stage capability evidence.

An accepted result remains outside the old optimizer family. The parent creates a new
derived experiment family from the returned source/config and starts that family with
a fresh ordinary `*-BASE` round. It must not import the optional branch's trials as
P/FINAL evidence. Option changes invalidate the candidate config/profile binding;
source changes also invalidate the old source revision and source map. Both require
fresh standard profiling before later SMAP/P/FINAL or whole-scope promotion.

The parent must run `bootstrap_derived_family.py` with the current incumbent identity.
This registers an idempotent `seed_registered` family under a file lock and deliberately
sets `ledger_merge_allowed=false`. A stale result creates no family. Do not merge the
schema 2 performance ledger until the derived family has run its fresh BASE and later
produced a standard validated experiment result.
Use `derived_family_lifecycle.py` to freeze/run fresh BASE, validate its receipt, record
ordinary lifecycle completion, and merge the ledger. Parent ordinary commands must use
`device_lease_runner.py` with the same absolute lease root as the child runner.

Write a Chinese `MULTISTREAM_TUNING.md` covering every attempted, blocked, failed,
rejected, and accepted trial, the unchanged fallback identity, local mechanism signals,
all fingerprints, relative artifact paths, and exact commands.
For every non-environment `failed` phase or trial, include one immutable per-attempt
failure-scene section defined by the parent contract. Preserve earlier failed sections
after retry or later success and link the failure-isolation handoff when applicable.
