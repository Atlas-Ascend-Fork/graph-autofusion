---
name: superkernel-auto-tune
description: Orchestrate the complete evidence-gated Ascend SuperKernel auto-tuning lifecycle through named stage agents, from intake through final Chinese E2E reporting.
---

# SuperKernel Auto Tune

`superkernel-auto-tune` is the only top-level adaptation entry point. It owns the
frozen experiment session, schema 2 ledger, candidate ranking, stage transitions, and
the final Chinese report. It never performs a stage directly: dispatch exactly one
named stage agent at a time, wait for its sealed handoff, validate the handoff, then
decide whether to continue, retry within the existing budget, select the next ranked
S candidate, or finish the report.

## Named Stage Delegation

At intake, ask the user to select and freeze one optional mode: `none`,
`multistream`, `source-range`, or `both`. Never infer a request for a multistream or
P/FINAL experiment from later profiling evidence.

Dispatch the following agents and Skills serially. Include the experiment/candidate ID
in the task title, but do not change the fixed agent name.

| Agent | Skill | Stage |
|---|---|---|
| `sk-intake-preparation` | `superkernel-intake-preparation` | 环境确认与实验准备 |
| `sk-s0-baseline` | `superkernel-s0-baseline` | S0 基线性能测量 |
| `sk-stage-a-scope-selection` | `superkernel-stage-a-scope-selection` | SK 框定方式选择 |
| `sk-stage-o-option-tuning` | `superkernel-stage-o-option-tuning` | Option 调优 |
| `sk-base-profile-source-mapping` | `superkernel-base-profile-source-mapping` | BASE profile、独立分析与 SMAP |
| `sk-optional-experiments` | `superkernel-optional-experiments` | 双流与 P/FINAL 实验 |
| `sk-final-e2e-report` | `superkernel-final-e2e-report` | 最终 E2E 确认与报告 |

Every stage must produce `PHASE_REPORT.md` and `phase-result.json` below the frozen
artifact root. A handoff includes the session ID, phase ID, agent ID, terminal status,
input fingerprints, consumed and produced artifact paths, decisions, blockers, failure
scene path when applicable, and Chinese next-step guidance. These records reference and
never replace collection manifests, analysis results, cleanup receipts, or the schema 2
experiment ledger.

Stages share model source, cache and NPU resources, so they never run in parallel. A
`no_gain`, `blocked`, or `failed` optional branch restores the exact incumbent. A
multistream `accepted` result starts a derived family; run a fresh ordinary BASE through
the profile/source-mapping stage before a requested source-range branch or final E2E.
The final report stage always runs. When no candidate can legally reach final clean E2E,
it records `not_run` and the authoritative blocker rather than inventing a comparison.

The remaining sections are the preserved hard adaptation contract. Stage Skills route
to the relevant sections and references; no stage may weaken, duplicate with a different
meaning, or bypass these gates.

Read [auto-tune-session.md](auto-tune-session.md) before dispatching the
first stage or accepting a stage handoff.

## Executable Control Plane

Create one session before the Intake delegation. The `--artifact-root` is the frozen
run root and the user must choose `--optional-mode` before it is created:

```bash
python3 scripts/auto_tune_session.py init \
  --session <artifact-root>/auto-tune-session.json \
  --artifact-root <artifact-root> --session-id <run-id> \
  --optional-mode <none|multistream|source-range|both>
```

Before every dispatch, run `next --session ...`. It returns the sole legal
`step_id`, `phase`, `agent_id`, and `skill`; dispatch exactly that agent and include
the returned object, frozen artifact root, and session path in its task. A stage must
return a candidate `superkernel-auto-tune-phase-result-v1` JSON document. Validate and
seal it before discussing the next stage:

```bash
python3 scripts/auto_tune_session.py seal \
  --session <artifact-root>/auto-tune-session.json --result <handoff.json>
python3 scripts/auto_tune_session.py verify \
  --session <artifact-root>/auto-tune-session.json
```

Never hand-edit a session, dispatch a later agent on a verbal result, or reuse a
sealed handoff. The controller owns `not_run` and failure explanations when a legal
comparison cannot continue. In `both` mode an `accepted` multistream handoff causes
the control plane to insert `base-profile-derived`; dispatch that fresh BASE/SMAP step
before the source-range step. The final agent must seal a final result in every
terminal path; sealing it produces `FINAL_E2E_REPORT.md`.

Treat adaptation as an evidence-gated search for repeatable end-to-end latency gain.
Fusion existence, fusion depth, and one successful process are not performance proof.
Any performance collection, mapping, or fingerprint mismatch is untrusted performance
evidence. Candidate reproducibility audit results remain a separate evidence domain.

## Required Intake

Before changing model code or launching inference, obtain or detect:

- target repository, revision, working directory, model/config paths, exact command,
  environment setup, and allowed changes;
- Python, CANN, PyTorch, torch_npu, NPU model/count, visible devices, backend,
  static-kernel mode, precision, TP topology, and resource budget;
- input shape, prompt/dataset, warmup, correctness method, baseline artifacts,
  artifact root, and success threshold.

Read [intake-checklist.md](intake-checklist.md). Stop on missing critical
data, failed setup, unsupported runtime, or unexplained detected-versus-declared
mismatch.

## Non-Negotiable Gates

1. Use a supported A2/A3 device, `npugraph_ex`, static kernel compilation, and active
   scope APIs. Before diagnosing or cleaning up static compilation, read and enforce
   [static-compile-process-safety.md](static-compile-process-safety.md).
   A live `*_compile_error.log`, elapsed process age, or the number of such files is
   never sufficient failure or hang evidence. The parent Agent and diagnostic helpers
   must not kill `opc`, `op_compiler`, an inference parent, a rank, or a process group
   from those signals. Only the frozen experiment runner's top-level timeout may
   terminate the exact experiment-owned process group; externally terminated compiler
   output is invalid failure evidence and requires a clean retry. Profiler schedule
   changes are also compile-startup controls: never implement `active`, `skip_first`,
   `warmup`, or `repeat` changes through `sitecustomize.py`, `.pth`, `PYTHONSTARTUP`, a
   startup `PYTHONPATH` shim, or another interpreter-startup monkey patch. Such hooks
   can import `torch_npu` before the launcher's normal multiprocessing/static-compiler
   initialization. Use the framework's ordinary profiler construction path after the
   approved CANN setup; if it cannot express the requested schedule, retain the default
   and report the limitation. Any profiler-construction or import-order change requires
   the isolated compile-health regression defined by the same safety reference before
   its profile is accepted.
2. Probe every option against the installed wrapper from the same runtime environment
   as inference. Before importing `torch_npu`, load the exact approved CANN setup used
   by the inference launcher and require CANN/HCCL paths in `LD_LIBRARY_PATH`.
   `cann_environment_not_loaded`, `torch_npu_runtime_load_failed`, or
   `option_probe_status=not_run` means the probe did not execute; it is never evidence
   that an option or value was rejected. Use only exact `accepted_values` from a
   successful `ready=true` report and archive that report as environment evidence.
3. Freeze source, command, config, workload, precision, TP, cache, and runtime controls.
4. Run exactly five clean SK-off S0 processes. Require post-warmup TP worst-rank mean
   spread `(max-min)/mean <= 5%`.
5. Split adaptation into two stages. S1/S2/S3/S4/... screening runs execution,
   correctness, and clean end-to-end timing only. It must not create P rounds,
   candidate diagnostic profiles, per-SK mappings, or source calibration.
   Every Stage-A S config must explicitly contain
   `super_kernel_optimize_options: {}` and `super_kernel_debug_options: {}` under
   `model_config.custom_params`. Missing fields are not empty: wrapper defaults may
   inject an option. Any non-empty, missing, or non-mapping value blocks the candidate
   before execution. DCCI, `auto_op_parallel`, aggressive, and other options start only
   after the winner is frozen and Stage O begins.
   Inspect the active model glue that constructs `npugraph_ex` compile options: it must
   forward these explicit empty maps without a hard-coded option or a non-empty fallback.
   If effective empty options cannot be proven from the frozen config and glue revision,
   Stage A is blocked.
6. Before the first S candidate executes, freeze a
   `superkernel-screening-matrix-v2` candidate set covering at least
   `automatic_aot`, `broad_decode`, `per_block`, and `semantic_segment`. Every entry
   must be settled as `executed`, `blocked`, or `skipped` before selection. A blocked
   or skipped entry requires a named blocker, Chinese reason, and evidence path. An
   existing eligible candidate or provisional winner is never a valid skip reason.
   `analyze_performance.py --selection-only` must receive this matrix and reject
   missing strategy kinds, pending/deferred entries, or any mismatch between matrix
   `executed` IDs and `--candidate` inputs.
7. Select one best eligible S candidate deterministically from frozen-S0 clean timing.
   Freeze it as `Sbest-SEED`. Before deep profiling, settle a winner-only option matrix
   covering the DCCI family, `auto_op_parallel`, aggressive, and other applicable
   experimental values. Ordinary O trials test one exact accepted option value against
   the current stable incumbent. DCCI is the sole conditional exception: test only
   `dcci_disable_on_kernel=[".*"]` first; never schedule standalone before/after trials.
   If disable-all regresses, follow
   [dcci-option-tuning.md](dcci-option-tuning.md), collect paired diagnostic
   profiles, locate every regressed SK and child, and test one combined repair whose
   identical child regex list is set on both before and after while disable-all remains.
   Retain only measured incremental gains and freeze the resulting config as
   `Sbest-BASE`. Lack of a profiling hypothesis is never a reason to skip an ordinary
   non-DCCI O trial. Try the next pre-ranked eligible candidate one at a time only
   when a default winner gate fails at BASE profiling/analysis, required SMAP
   settlement, or whole_scope correctness/promotion. An
   `absent/no_gain/blocked/failed/invalid` optional P branch ends only that branch,
   restores the preserved BASE incumbent, and continues whole_scope validation; it
   never triggers next-ranked fallback. Never profile every S candidate.
8. Read [winner-option-sweep.md](winner-option-sweep.md) and finish the
   `Sbest-SEED -> O0/O* -> Sbest-BASE` stage before collecting the winner profile.
   Ordinary O rounds run execution, correctness, and clean timing only. The only
   diagnostic-profile exception is a measured disable-all regression inside the DCCI
   family; those paired profiles and child traces diagnose and construct the combined
   repair value, do not count as clean timing, do not perform source mapping, and do
   not create P actions.
9. Before any optional multi-stream branch or explicitly requested P/FINAL round starts
   its first NPU command, run the adapter capability and run-specific collection preflight
   defined in
   [adapter-capability-preflight.md](../../superkernel-multistream-performance-tuning/references/adapter-capability-preflight.md).
   Freeze the actual action set rather than the stage label: P normally requires
   `range_exclusion` or `scope_split`; FINAL requires the union of all materialized source
   actions. P/FINAL must never request the `option` action. They inherit the exact
   `Sbest-BASE` optimize/debug option maps, require `declared_option_changes=[]`, and
   reject any option JSON Pointer or config drift. Multi-stream uses its requested
   reorder/event/stage actions. Require the archived
   collection plan to declare `round_kind=multistream/P/FINAL`, bind the actual
   source/config/control/workload identity JSON files, and pass their semantic identity
   checks. Every P/FINAL plan additionally binds its profile launcher and round action
   manifest and requires fresh baseline/candidate profile manifests, profile-owned
   `sk_meta`, and profiling analysis output. Require the archived
   preflight receipt to have `status=ready` before acquiring devices. The run-specific
   collection plan must bind every producer/validator/materializer and declare every
   required output for those actions. A missing capability or collection binding is a
   pre-experiment blocker, not a failed experiment or a profiling/trace attempt. Do not
   spend NPU time hoping a later artifact will supply information that was absent from
   preflight. This applies to every P round, not only P1. Re-run preflight when the source revision, adapter, action set, capture
   plugin, provider, launcher, validator, or expected-output contract changes.
10. For the selected winner and each later explicitly requested P/FINAL optimization
   round, require full
   execution, correctness, fresh profile-owned metadata, and two validated profile
   collection manifests before performance analysis. Collect distinct immutable
   `baseline_profile` and `candidate_profile` roots. The
   candidate profile root must archive
   that profiling process's profiler data and complete SK metadata. `sk_prof` is an
   optional post-mapping scheduling trace;
   metadata from compat or verify runs cannot substitute for profile metadata.
   When optional `ASCEND_PROF_SK_ON` scheduling evidence reports `buffer is full`,
   discard only that scheduling attribution and collect a shorter diagnostic trace if
   needed; do not invalidate the independent kernel projection mapping.
   Both manifests must embed the same structural `association-config.json`, which
   contains only graph targets and collection invariants shared by SK-off and SK-on.
   Their `control` input likewise contains shared collection/runtime invariants and is
   distinct from the control object derived from the actual execution configs.
   Never substitute the actual baseline or candidate execution config for this field;
   those configs are validated separately with the declared-change set.
11. Dispatch a fresh read-only profiling analysis child agent for every profiled winner
   optimization round. The experiment child must not classify its own profiling data.
12. child_count is descriptive only and never a profiling gate. Record depth and
   fragmentation, but classify every reliably mapped SK from measured profiling.
13. Use profiler interval as the primary per-SK criterion; duration sum is secondary
   diagnosis. Diagnostic profiler runs never count as clean timing.
14. Every selected winner must settle SK-to-editable-source mapping after its first
    winner-only per-SK profile. This applies to automatic AOT and every named/manual
    winner. If any performance-exact SK lacks an exact source half-open interval, run one
    batched `Sbest-SMAP` before an optional P branch or whole-scope clean promotion. After SMAP, exact
    ranges remain actionable and partial/skipped ranges are recorded and omitted from
    source edits. Whole-scope enablement still depends on complete performance mapping
    and clean end-to-end gain, not on every local range being source-actionable.
15. Multi-stream performance tuning is an optional winner-only branch after fresh
    `Sbest-BASE` profiling analysis. Complete exact SK-to-editable-source mapping for
    every source-reordered target before dispatching the branch; option-only trials do
    not require that map. Dispatch it only through
    [superkernel-multistream-performance-tuning](../../superkernel-multistream-performance-tuning/SKILL.md)
    with a validated request and dedicated worktree/config/artifact/cache roots. A
    missing, invalid, `no_gain`, `blocked`, or `failed` result preserves the exact
    incumbent and never blocks the default whole-scope or separately requested
    optional P path. An `accepted` result starts a new derived experiment family at
    a fresh ordinary `*-BASE`; never add an M round to the existing schema 2 family.
16. Produce a Chinese report for every attempted, failed, blocked, or skipped round.
    For every non-environment failure, read and enforce
    [failure-scene-reporting.md](failure-scene-reporting.md): preserve the
    immutable attempt artifacts first and record the failure scene in that experiment's
    own report before retry, fallback, cleanup, or next-candidate work. A parent summary
    never substitutes for the experiment-local record. Pre-experiment environment
    failures stay in environment evidence and must not be mislabeled as experiment
    failures.

    `Sbest-P* -> Sbest-FINAL` is an optional source-range branch. Start it
    only when the user or frozen experiment plan explicitly requests it; never create
    P merely because BASE or SMAP finished. A requested branch that is absent,
    invalid, `no_gain`, `blocked`, or `failed` preserves the incumbent and never blocks
    unchanged `whole_scope_clean_validation`.

Never manufacture a gain by changing a frozen performance-sensitive control.

## Agent Boundaries

After S0 stability passes, read
[experiment-agent-orchestration.md](experiment-agent-orchestration.md).
The parent owns the frozen baseline, schema 2 ledger, cross-experiment conditional
evidence, and final Chinese synthesis. Dispatch one fresh child agent for each
parallel/main experiment in serial order because experiments share source, devices,
and caches.

Each S-screening child owns only execution, correctness, and clean timing for its one
strategy. It must not collect candidate profiling or start P. After the parent selects
one winner, that winner experiment child first owns the clean O option sweep and the
conditional paired DCCI regression collections, then owns ordinary winner profiling,
metadata, optional candidate-side reproducibility audit, and later source-range changes. Before either
DCCI diagnostic profiling command, it must create the two independent collection sessions/manifests and
make the profiling producer archive its own artifacts into those roots; a later `raw-run` copy or manual
`sk_meta` copy cannot repair this contract. It does not
interpret per-SK performance. For every winner profiling round, nested dispatch is preferred:
the experiment child dispatches a different fresh read-only analysis agent. If nested
dispatch is unavailable, the experiment child writes a read-only analysis request;
the parent dispatches the fresh analysis agent and returns the response.

The analysis agent must read and follow
[superkernel-fusion-performance-analysis](../../superkernel-fusion-performance-analysis/SKILL.md).
It may read immutable artifacts and write only its dedicated analysis result and
Chinese report. It must not run inference, edit source, change scope, or alter options.

The optional multi-stream child must read and follow
[superkernel-multistream-performance-tuning](../../superkernel-multistream-performance-tuning/SKILL.md).
It owns only its isolated branch and may run inference solely under the parent's frozen
execution and device-serialization contract. It must not edit the incumbent, mutate the
parent ledger, or promote local Cube/Vector overlap as a substitute for clean
incremental end-to-end gain.

If an SK-enabled candidate crashes, hangs, times out, or fails correctness, read fresh
plog and metadata through
[superkernel-sk-failure-isolation](../../superkernel-sk-failure-isolation/SKILL.md).
The experiment child must read and follow that skill before changing the scope. Never
exclude an arbitrary child merely because it appears near the error.

## Controlled Baseline

Create S0 with SuperKernel disabled and all intended production controls frozen. Keep
profiler, metadata dump, event tracing, debug sync, and `ASCEND_PROF_SK_ON` disabled for
the five clean processes.

```bash
python3 <runtime-skill-dir>/scripts/analyze_performance.py \
  --baseline experiments/S0 \
  --warmup 8 \
  --json-out experiments/S0/baseline-stability.json
```

For clean timing of an unchanged candidate that already passed the complete
whole-scope profiling gate, declare the lifecycle explicitly and do not provide replay
evidence:

```bash
python3 <runtime-skill-dir>/scripts/analyze_performance.py \
  --baseline experiments/S0 \
  --candidate S2=experiments/S2/S2-CLEAN \
  --promotion-mode S2=whole_scope \
  --warmup 8 \
  --json-out experiments/S2/S2-CLEAN/performance-summary.json
```

The analyzer CLI keeps its legacy `range_optimized` fallback for compatibility, but
the default workflow explicitly passes `--promotion-mode NAME=whole_scope`.
An explicitly requested range-optimized branch still requires a matching
`--replay-evidence NAME=PATH`. `whole_scope` only removes that source-edit replay gate;
it does not relax baseline stability, candidate repeat count, mean, P90, or standard
deviation checks. The profiling inventory and correctness gates remain ledger-owned.

If S0 is unstable, stop before scope, profiler, DCCI, or option experiments. A later
change to source, baseline config, control, or workload fingerprint invalidates S0.

Collect one separate SK-off diagnostic profile for layer inventory and per-range
comparison. This profile is read-only evidence, not a clean latency sample.

## Stage A: Screen S Candidates

After S0 is stable, execute S1/S2/S3/S4/... as independent strategy candidates. Each
candidate changes one declared major factor and must pass full execution and
correctness, then produce at least three independent clean processes with profiler,
metadata dump, `sk_prof`, event tracing, debug sync, and calibration disabled. During
this stage:

- freeze both option maps explicitly in every S config:
  `super_kernel_optimize_options: {}` and `super_kernel_debug_options: {}`;
- run `validate_stage_a_options.py` on all frozen S configs before S1 starts, archive
  its JSON result, and reject a missing field instead of relying on backend defaults;
- do not collect candidate diagnostic profiling or invoke the sibling performance
  analyzer;
- do not map fused SKs to SK-off operators or source intervals;
- do not insert calibration markers or generate source actions;
- optional compile/fusion-existence logs are sanity evidence only and never classify SKs.

Freeze the complete candidate set before the first candidate process. The matrix must
retain all four required strategy kinds even when one cannot run. Use `blocked` or
`skipped` only with direct evidence such as an unsupported source boundary, a denied
source-change permission, an environment incompatibility, or an approved budget limit.
Do not mark S4 or another planned strategy deferred merely because an earlier candidate
is eligible. Selection is valid only after every matrix entry is settled.

Before execution, validate the frozen S1/S2/S3/S4 configs:

```bash
python3 <skill-dir>/scripts/validate_stage_a_options.py \
  path/to/S1.yaml path/to/S2.yaml path/to/S3.yaml path/to/S4.yaml \
  --json-out experiments/stage-a-option-validation.json
```

Every clean run must archive the exact input as `run-*/config.yaml`.
`analyze_performance.py --selection-only` revalidates all archived configs and fails
closed before reading timing when either option map is absent, non-empty, or not a map.

Rank all correct candidates in one decision using frozen S0 and the same fingerprints:

```bash
python3 <runtime-skill-dir>/scripts/analyze_performance.py \
  --baseline experiments/S0 \
  --candidate S1=experiments/S1/CLEAN \
  --candidate S2=experiments/S2/CLEAN \
  --candidate S3=experiments/S3/CLEAN \
  --candidate S4=experiments/S4/CLEAN \
  --screening-matrix experiments/screening-candidate-matrix.json \
  --selection-only --warmup 8 \
  --json-out experiments/screening-performance-summary.json
```

Only candidates meeting the fixed mean threshold, positive median-run direction,
P90/stddev gates, minimum three-run count, and stable five-run S0 are eligible. Freeze
`selection.selected_for_deep_analysis` as `Sbest-SEED`; the screening result is not a
promotion decision. If no candidate is eligible, retain S0 and stop without option,
profiling, or P work.

## Reconstruct Layers And Draft Scopes

Read [layer-and-pipeline-analysis.md](layer-and-pipeline-analysis.md), then
inventory every repeated layer, ordered operator, task range, static state, Cube/
Vector/MIX family, stream, overlap, and unassigned prelude/postlude work. Missing
layers are evidence gaps, not empty layers.

### Direct Scope-Marker Requirement

Every model-source scope marker for `npugraph_ex` **must** be emitted by the direct
active-runtime APIs `torch.npu.super_kernel_scope_begin(name)` and
`torch.npu.super_kernel_scope_end(name)`. Do not substitute a repository helper,
`torchair` scope context manager, generic Python context manager, decorator, or
boolean-only wrapper for those marker calls. Such abstractions can change what the
captured graph observes and are not proof of an SK scope boundary.

Use one deterministic name, then balance the same name on every executed source path:

```python
if enable_target_scope:
    torch.npu.super_kernel_scope_begin("model.block.12.moe_expert_gmm")
try:
    output = target_unit(...)
finally:
    if enable_target_scope:
        torch.npu.super_kernel_scope_end("model.block.12.moe_expert_gmm")
```

Keep the `try` body limited to the intended semantic unit. Do not place either marker
inside a different conditional, return path, exception path, or asynchronous callback
from its mate. The begin/end name must match exactly, names must be deterministic and
within the runtime limits, and fresh `sk_meta` must prove the resulting child set.
Only an explicit source-level no-op branch may be used when SuperKernel is disabled;
it must not replace the direct calls when enabled.

To remove a source unit from a surrounding named SK scope, use the direct balanced
`None` exclusion marker, after the active-environment capability probe accepts it:

```python
torch.npu.super_kernel_scope_begin("model.block.12")
before_target(...)
torch.npu.super_kernel_scope_begin(None)
excluded_operator(...)
torch.npu.super_kernel_scope_end(None)
after_target(...)
torch.npu.super_kernel_scope_end("model.block.12")
```

`None` is the Python value, never the string `"None"`; it has unfusible priority only
over the enclosed source unit. If the direct `None` pair is unsupported, close the
outer named scope before that unit and reopen it after the unit, preserving all
operator execution and dependencies. A successful API call is insufficient: fresh
`sk_scope_split.log` and `sk_fused_nodes.log` must show the excluded child absent while
the intended surrounding scope remains intact.

Before editing markers, prepare a source-confirmed strategy matrix. Vary one high-level
scope/script factor per S experiment: automatic AOT, broad decode, per-block,
dependency-aligned segment, or adjacent composite. Do not schedule DCCI or other
source-range restoration experiments during S screening; those belong only to the
direct `torch.npu.super_kernel_scope_begin(name)` / `scope_end(name)` calls and
deterministic scope names. Preserve operator execution, data dependencies, events, barriers,
communication, cache mutation, and Cube/Vector stream semantics.

Task-range/layer reconstruction is inventory and display evidence only. It must never
be promoted to an SK-to-source exact mapping, even when every repeated block appears
structurally identical. For a source-actionable range, read the sibling skill's
`../superkernel-fusion-performance-analysis/references/source-calibration-mapping.md`
and use its model-independent calibration protocol. The model/language adapter may
discover semantic units and add temporary balanced leaf markers, but the core identity
must remain opaque `block_instance_id` plus unique labeled business-graph occurrence;
never encode model names, fixed layer counts, odd/even rules, or global ordinals.

Calibration runs establish identity only and do not provide clean timing. After
calibration, delete unused temporary markers. A named-scope selected unit becomes
actionable only after its marker is fixed in a new stable-marker revision. An automatic
AOT unit instead uses the `stable_source + marker_only_calibration` bridge described in
Stage B; its temporary markers must not remain in the production source. In both modes,
regenerate the source manifest/snapshot and satisfy the mode-specific byte-equality,
correctness, collection, and analyzer replay gates. The analyzer must then validate
`source_scope_map_v2`; legacy
`task_ranges`, repeated signatures, local op windows, or raw IDs remain
`diagnostic_only`.
The analyzer consumes only the standard artifacts and must never import or execute the
model adapter. It must compare every current `--source-root` file byte-for-byte with
the archived actionable source snapshot; a revision label without byte equality is not
source-actionable. State both constraints explicitly in any calibration plan or
handoff.

Read [adaptation-playbook.md](adaptation-playbook.md) before changing model
code and [torchair-superkernel-reference.md](torchair-superkernel-reference.md)
before selecting version-sensitive options.

Run the option probe in an isolated subprocess after the inference environment setup,
for example:

```bash
source /path/to/active/cann/bin/setenv.bash
python3 <skill-dir>/scripts/check_environment.py --json
```

Use the target launcher's approved setup path rather than assuming the example path.
Inspect repository setup scripts before sourcing them because they may contain
unrelated side effects; do not source a broad launcher merely to run the probe. Confirm
that `runtime_environment.loaded`, `ready`, API checks, device topology, and the exact
option's `accepted_values` all pass. A direct `python check_environment.py` from an
unprepared login shell is invalid evidence even when it happens to find the Python
package.

## Stage O: Sweep Winner Options Before Profiling

Read [winner-option-sweep.md](winner-option-sweep.md) and the DCCI-specific
[dcci-option-tuning.md](dcci-option-tuning.md). Freeze a
`superkernel-winner-option-matrix-v1` before O1. Cover accepted non-baseline values for
DCCI, `auto_op_parallel`, `aggressive_opt_strategies`, and other applicable experimental
controls. Represent DCCI as one conditional family, not independent before/after items.
Every item must end as accepted, rejected, failed, blocked, or skipped with evidence.
Missing per-SK diagnostic evidence is not a blocker for the initial disable-all trial;
it becomes required only after disable-all has a measured regression.

Create a fresh five-process stable `O0-INCUMBENT` from `Sbest-SEED`. Ordinary O trials
change one option/value at one RFC6901 Pointer, pass full correctness, and produce at
least three independent clean processes with all profiling and diagnostics disabled.
Compare them incrementally against the current five-process incumbent:

```bash
python3 <runtime-skill-dir>/scripts/analyze_performance.py \
  --baseline experiments/Sbest/O0-INCUMBENT/CLEAN \
  --candidate O1=experiments/Sbest/O1/CLEAN \
  --option-trial --warmup 8 \
  --json-out experiments/Sbest/O1/option-trial-summary.json
```

Retain a value only when its mean improvement is strictly positive (no fixed percentage
threshold), the median-run direction is positive, and the P90 and standard-deviation
gates pass. Extend an accepted trial to exactly five clean
processes and revalidate stability before using it as the next incumbent. Roll back a
rejected, failed, unstable, or incorrect trial.

For DCCI, first compare only exact `dcci_disable_on_kernel=[".*"]` with O0. Accept it
directly if it passes. Reject a neutral/no-gain result without boundary trials. On a
measured regression, collect fresh no-option and disable-all `kernel_details.csv`,
profile-owned SK metadata, and complete `sk_prof_<device>.json`; compare every SK,
attribute every regressed SK to its significantly slower children, then test one
combined repair with the stable union of those child regexes set identically on
`dcci_before_kernel_start` and `dcci_after_kernel_end`, while retaining disable-all.
This combined two-pointer change is the only Stage O exception to the ordinary
one-pointer rule. It must beat O0 directly; recovery relative only to disable-all is
not acceptance evidence. After every matrix item is settled, freeze the final incumbent
as `Sbest-BASE`; only then start ordinary Stage B profiling.

### DCCI Diagnostic Collection Contract

Before the first DCCI diagnostic NPU command, create separate empty O0 and disable-all
profile roots with the collection-side `artifact_contract.py begin` protocol in
[experiment-agent-orchestration.md](experiment-agent-orchestration.md). The
corresponding session and finalized manifest are part of collection, not post-processing:
each must bind the exact execution config, workload, source revision, round/role identity,
launcher command and timing, `ASCEND_PROF_SK_ON` setting, expected profiler outputs, and
the destination for profile-process-owned SK metadata.

The collection producer must archive `kernel_details`, SK metadata, and optional
`sk_prof` from that same profiling process into its role root before finalizing its
manifest. A runner that only copies a result directory into `raw-run`, or a later manual
copy from a shared `sk_meta/` directory, is not an attributable DCCI collection. After
both runs, invoke `artifact_contract.py validate-set` on the two manifests before
dispatching read-only DCCI analysis. Raw `kernel_details`, `sk_meta`, and `sk_prof` files
without valid capture-time manifests must be reported as a collection blocker and require
a fresh reprofile from new empty roots. Never reconstruct, backfill, or "repair" a
manifest after the profiling process has finished.

## Stage B: Profile And Optimize The Winner

Only `Sbest-BASE` and its later explicitly requested P/FINAL rounds compare the immutable SK-off
`baseline_profile` with one fresh
`candidate_profile` that owns both timing data and complete SK metadata. Create and
finalize those two roots with the collection-side `artifact_contract.py` protocol in
[experiment-agent-orchestration.md](experiment-agent-orchestration.md).
Execution and correctness must pass for the profiled candidate process.

Compat/verify runs may be collected as an independent candidate-side reproducibility
audit. Their signature replay result must not enter SK-to-baseline mapping errors, must
not block measured classification, and must not be passed to the performance analyzer.
When promotion needs independent repeatability evidence, repeat the instrumented
candidate profile or compare a separate canonical fusion inventory instead of requiring
unique op signatures.

The optional metadata reproducibility audit remains diagnostic only:

```bash
python3 <runtime-skill-dir>/scripts/analyze_sk_meta.py experiments/S3/S3-BASE/compat/sk_meta \
  --json-out experiments/S3/S3-BASE/sk-meta-summary.json \
  --round-name S3-BASE \
  --effective-min-child-nodes 1 \
  --round-report-out experiments/S3/S3-BASE/round-report.json

python3 <runtime-skill-dir>/scripts/analyze_sk_meta.py \
  experiments/S3/S3-BASE/compat/sk_meta \
  --verify-root experiments/S3/S3-BASE/verify/sk_meta \
  --candidate-name S3 \
  --round-id S3-BASE \
  --source-revision REVISION \
  --compat-config-manifest experiments/S3/S3-BASE/compat/config.json \
  --verify-config-manifest experiments/S3/S3-BASE/verify/config.json \
  --replay-min-child-nodes 1 \
  --replay-report-out experiments/S3/S3-BASE/fusion-replay.json
```

`fusion_reproducible`, `deep_fusion_reproducible`, child depth, and unmatched signature
counts are audit descriptions only. They do not gate profile-vs-baseline mapping or
beneficial/neutral/regressed classification.

Collection session fingerprints, native PIDs and times, source revision, and command
fingerprints are protocol evidence only. Never pass them to the model as parameters,
environment graph attributes, compiler options, graph nodes, or runtime markers.

## Benefit-Driven Round Lifecycle

After winner selection and completion of the O matrix, the default winner lifecycle is:

`Sbest-SEED -> O0/O* -> Sbest-BASE -> [Sbest-SMAP for source-reordered targets] -> [optional isolated multistream branch -> derived-family BASE] -> whole_scope_clean_validation`

`Sbest-P* -> Sbest-FINAL` is a separate optional source-range branch. It
requires an explicit user or frozen experiment-plan request and must not be created
merely because BASE or SMAP completed. Preserve every source-action gate below when it
is requested. An absent, invalid, `no_gain`, `blocked`, or `failed` branch retains the
incumbent and does not block unchanged `whole_scope_clean_validation`.
The option maps frozen into `Sbest-BASE` are immutable throughout P/FINAL. These rounds
must not probe, add, remove, replace, narrow, broaden, or combine option values. Historical

Every listed winner round repeats execution, correctness,
diagnostic profile collection, and fresh independent profile-vs-baseline analysis.
No non-winning S candidate may enter this state machine. After the winner's unchanged
BASE profile, inspect every performance-exact SK, including beneficial SKs. If any lacks
an analyzer-validated editable source half-open interval, generate one batched
winner-only `Sbest-SMAP` and do not generate P or `whole_scope_clean_validation` until
that mapping attempt is settled. A settled bundle may contain exact and skipped ranges:
only exact ranges enter source actions, while skipped ranges remain unchanged. Do not
convert automatic fusion into a named manual-scope family merely to obtain source
identity.

### Sbest-BASE

Profile every mapped SK regardless of child count. The analysis agent maps
SK-on occurrences to the corresponding SK-off graph-occurrence timing interval and
returns one of:

- `beneficial` -> `keep`;
- `source_scope_map + exact + proven source offsets` with `neutral` or `regressed`
  -> `prune` candidate;
- `insufficient_evidence` -> `reprofile` or `block`, with no scope change.

The main experiment child consumes the validated artifact result; it must not recalculate
classification from CSV rows or summaries.

Here `source_scope_map + exact` means only analyzer-validated
`source_scope_map_v2`: `C_sk == C_unit`, unique original-to-calibration graph
correspondence, every child of that SK exact-assigned in at least three validated
steps, an actionable source revision,
archived byte-span evidence, provenance DAG, current candidate collection binding, and
the same baseline projection fingerprint. A producer-written boolean or old layer map
cannot satisfy this gate.

### Optional Multistream Tuning Delegation

After the unchanged winner's fresh `Sbest-BASE` profile has a valid schema 1.2 analysis,
the parent may dispatch the isolated multi-stream skill when evidence shows degraded
parallelism or actionable scheduling headroom. Before a source reorder, complete
`source_scope_map_v2 + exact` for every target SK and bind each fused child to the
corresponding complete model-source statement. This branch is optional: for example,
`beneficial + degraded` remains a keep decision and may become a tuning opportunity,
but it is not a prune decision and does not force a trial.

Build and validate `superkernel-multistream-request-v1`. Option-only trials do not
require source mapping. Scope split or range exclusion requires analyzer-validated
`source_scope_map_v2 + exact` with actionable byte offsets. Do not repeat an identical
global option/value already settled in Stage O.

Business-operator reorder is permitted only inside the optional multistream branch and
only for a directly proven multistream target. Require at least three aligned SK-off
occurrences with stable cross-stream overlap, matching SK-on `sk_meta` dispatch identity
and a stable inversion, exact statement spans, and a complete hard dependency DAG. A
single-stream target or generic regression must never authorize source reordering.
For a component-aware MIX target, additionally require complete bound AIC/AIV child
lanes and prove that one pure lane can overlap the other operator without hiding
same-engine contention from the MIX operator's remaining lane. Read the multistream
skill's `references/component-overlap-reorder.md`. Run the minimal dependency-safe
route first; bounded alternative topology orders may run only after that route changes
dispatch order and settles without incremental gain.

Validate `superkernel-multistream-result-v2` before consuming it. For absent, malformed,
stale, `no_gain`, `blocked`, or `failed` output, retain the exact `Sbest-BASE` incumbent
and continue the normal lifecycle. For `accepted`, use the returned source/config only
to create a separately named derived family, then run fresh correctness, profile-owned
metadata, profiling analysis, and an ordinary `*-BASE`. The isolated trial artifacts
are not imported as BASE/P/FINAL evidence, and local overlap improvement never
replaces clean five-run incremental acceptance.

For an accepted result, read the multi-stream skill's
`references/action-execution-and-handoff.md`, then run
`scripts/bootstrap_derived_family.py` with the current incumbent, request, result,
dedicated derived-family registry, and family root. The tool must return
`seed_registered`, `fresh_base_required=true`, and `ledger_merge_allowed=false`.
This operational registry is not the schema 2 performance ledger. A stale result or
lineage collision creates no family. Run parent NPU commands through
`scripts/device_lease_runner.py` with the same absolute lease root used by the child.
Read [shared-npu-lease.md](shared-npu-lease.md) and validate
`scripts/shared_npu_lease_inventory.py` after changing any production process-launch
path. A partial or conflicting parent marker and an unregistered launcher are blockers.
Use `scripts/derived_family_lifecycle.py` to run the frozen fresh-BASE plan, validate
the five-run receipt, record the ordinary validated schema 2 experiment result, and
merge the standard ledger. The tool enforces `seed_registered -> fresh_base_completed
-> ordinary_lifecycle_completed -> ledger_merged`; no earlier status may merge.

### Sbest-SMAP

`Sbest-SMAP` is a mandatory identity-completion gate for any selected winner whose
measured SK occurrences are performance-exact but not source-actionable. It is not an S
candidate, timing round, P round, or alternative scope strategy. Run it once for all
such winner SKs, including beneficial, neutral, and regressed classifications; batch by
the target range/SK identity and available `graph_occurrence_fingerprint` set instead of
creating one calibration process per SK.

For a named-scope winner, use the existing `stable_marker` source-map path. For an
automatic AOT winner, freeze the unmarked production source as `stable_source`, create
an independent marker-only calibration worktree, and read the sibling source mapping
protocol. The adapter may insert temporary balanced semantic markers only. The bridge
must prove that removing the sealed insertion byte spans recreates every frozen AUTO
source file byte-for-byte, and that stable/calibration units have identical opaque
unit IDs, syntax hashes, adapter identity, and block templates. Then require the unique
original-to-calibration business-graph projection, at least three exact assignment
steps, and `C_sk == C_unit`.

Build and validate `source_scope_map_v2`, then re-run the read-only analyzer against the
immutable winner profile. Named/manual winners use `source_revision_role=stable_marker`;
automatic winners use schema 2.1 `source_revision_role=stable_source`, the calibration
manifest/snapshot, and `marker_only_calibration_v1`. `exact_cover` neutral/regressed
ranges may enter `Sbest-P1` only when the optional source-range branch was explicitly
requested; `partial_intersection`, skipped, conflicting, or
insufficient groups remain unchanged while independently exact groups continue. Seal
the result counts and skipped reasons so the strategy can distinguish a completed
partial attempt from an unstarted mapping. Delete temporary calibration markers after
the bundle is sealed. AUTO offsets always address the frozen unmarked source snapshot.

### Sbest-P* (Explicitly Requested Source-Range Branch)

Create a performance-prune round only for exact, reliable source ranges classified
`neutral` or `regressed`. Change fusion participation for only those source ranges
without deleting their operator execution or dependencies. Source mapping decides
whether P is permitted; it does not prescribe the exclusion mechanism. For a named
scope, or for an automatic candidate when the intended automatic-fusion delta is
independently verifiable, one available mechanism is a balanced explicit exclusion
accepted by the active `npugraph_ex` wrapper:

```python
torch.npu.super_kernel_scope_begin(None)
target = source_unit(...)
torch.npu.super_kernel_scope_end(None)
```

Use the Python value `None`, never the string `"None"`. The pair may be nested inside
the retained named scope; `None` has unfusible priority for only that exact source
unit. Archive a successful `scope_capabilities.explicit_none_exclusion.accepted=true`
environment probe, then prove from fresh fusion metadata that the target unit is no
longer a fused child and that the surrounding named scope remains intact. API
acceptance alone is not semantic proof. If the target wrapper does not support the
explicit `None` pair, close the named outer scope before the unit and reopen the same
scope after it. Use exactly one exclusion method in a P round and record it in the
declared change; method A/B comparisons are separate rounds. A batch may contain
multiple ranges only when
every pair has machine-proven non-overlapping `source_file/start_offset/end_offset`
intervals in the same `source_file`. Adjacent half-open intervals are non-overlapping;
cross-file, overlap, containment, or equal intervals cannot be batched. Op-name-only
boundaries prove no source interval, so emit one range per P round. Every P requires
fresh profiling and a fresh analysis Agent. A prune becomes `verified` only when a
strictly later P round explicitly identifies that removed range and validates the
remaining fusion state.

Before applying any source-exact `prune`, compare the same stable range identity with
all earlier BASE/P reports in the current winner family. If the history contains both
`keep` and `prune`, or the current minimum-three-occurrence sample reports an
unrepresented tail, treat the range as `insufficient_evidence/reprofile`: do not edit
source. Recollection stays inside the same logical round, expands the effective
profiling occurrence window when the runner supports it, and is limited to three
fresh attempts. Repeating the same three-step window is not evidence amplification.
Only a fresh report with the required occurrence count and no dispersion blocker may
restore source actionability; unresolved cross-round action flips remain blocked and
are preserved in the Chinese report.

 ### Sbest-FINAL (Explicitly Requested Source-Range Branch)

This is the range-optimized path. Combine only retained beneficial ranges, then repeat
the fresh lifecycle because interactions can
invalidate isolated wins. Every edited final range must have a current source-actionable
`beneficial/keep` decision with no conflict. Do not apply these per-range
requirements to an unchanged `whole_scope` candidate. FINAL materializes source actions
only and must preserve the exact `Sbest-BASE` option maps with
`declared_option_changes=[]`.

### Whole-Scope Promotion

Generate the default `whole_scope_clean_validation` after the winner source-mapping stage is
already satisfied by exact source intervals or has completed once with explicit skipped
ranges, and every profiled SK has a performance-exact mapping, profiling analysis has no
blocker, correctness passed, and there is no `insufficient_evidence`. Preserve the exact
candidate source, scope markers, config, control, workload, and fingerprints; source
edits are forbidden. Local classifications and source-map skips explain the profile but
do not decide feature enablement. Promotion is decided by at least three independent
clean candidate processes against frozen five-run S0: mean gain must meet the agreed
threshold and P90 and standard deviation must not regress.

## Automatic AOT Candidate

If automatic AOT is one of the S candidates, its screening round is clean timing only.
It receives no per-SK classification and creates no P round. If it wins, freeze that
exact script/config as `Sbest-SEED`, finish the ordinary winner option sweep, freeze the
retained option combination as `Sbest-BASE`, then profile and classify every mapped
SK, including single-child and shallow groups.

- Carry classified graph occurrences as proposed performance evidence; do not call
  them source ranges before source-map completion.
- For every performance-exact SK without source exact, generate one batched
  `Sbest-SMAP` request keyed by range/SK identity and the available
  `graph_occurrence_fingerprint` set. Do not generate a manual named-scope replacement.
- After `stable_source + marker_only_calibration` produces analyzer-validated
  `source_scope_map + exact` offsets, mark those ranges source-actionable and eligible
  for the optional source-range branch. Only when the user or frozen experiment plan
  explicitly requests that branch may the automatic winner enter `Sbest-P*`,
  otherwise preserve the unchanged incumbent and continue to
  `whole_scope_clean_validation`.
- Preserve `insufficient_evidence` without changing scope.

`kernel_projection_structural + exact_projected_trace` alone identifies only a measured
graph occurrence and cannot create source offsets or a source edit. The independent
`Sbest-SMAP` evidence chain, not the existence of a manual scope, is what upgrades an
automatic occurrence to source-actionable. The initial AUTO-BASE analysis remains
proposed. When the optional branch was explicitly requested, after SMAP a source-exact
AUTO decision may become `applied/verified` only by naming a strictly later fresh P
round as its verification round; it may then enter conditional evidence under the
ordinary artifact and fingerprint gates. Source-action evidence still arises from fresh
rounds.

These restrictions apply to source actions, not unchanged feature promotion. However,
source mapping must be attempted and settled before unchanged feature promotion is
recommended. A full winner profile with exact performance mappings, no analyzer
blocker, no `insufficient_evidence`, a settled SMAP outcome, and passing correctness may
generate `whole_scope_clean_validation`; its clean end-to-end result decides whether
automatic AOT is enabled.

## Profiling Analysis Delegation

Read [profiling-fusion-analysis.md](profiling-fusion-analysis.md). The
execution child records immutable baseline/candidate profiles, their content
fingerprints, profile-owned metadata, two manifests, source map, and `declared_change_set` in a
relative analysis request. The fresh analysis child runs the sibling analyzer:

Here `<skill-dir>` is the resolved root of the current `superkernel-auto-tune` skill
(the directory containing this `SKILL.md`), independent of the shell working directory.

```bash
python3 <skill-dir>/../superkernel-fusion-performance-analysis/scripts/analyze_fusion_performance.py \
  --baseline-profile experiments/S0/profile/profiler/kernel_details.csv \
  --candidate-profile experiments/S3/S3-BASE/profile/profiler/kernel_details.csv \
  --sk-meta experiments/S3/S3-BASE/profile/sk_meta \
  --baseline-config experiments/S0/config.json \
  --candidate-config experiments/S3/S3-BASE/config.json \
  --baseline-workload experiments/S0/workload.json \
  --candidate-workload experiments/S3/S3-BASE/workload.json \
  --declared-change-set experiments/S3/S3-BASE/declared-change.json \
  --baseline-collection-manifest experiments/S0/profile/association-artifact-manifest.json \
  --profile-collection-manifest experiments/S3/S3-BASE/profile/association-artifact-manifest.json \
  --candidate-name S3 --experiment-id S3 --round-id S3-BASE \
  --analysis-agent-id analysis-S3-BASE --source-revision REVISION \
  --json-out experiments/S3/S3-BASE/profiling-analysis/profiling-analysis-result.json \
  --markdown-out experiments/S3/S3-BASE/profiling-analysis/PROFILING_ANALYSIS.md
```

The shared runtime `scripts/analyze_profiler.py` is the executable launcher. New
analysis requests call the sibling analysis Skill through that runtime boundary.

## Ledger Contract

Validate every experiment result before merge:

```bash
python3 <runtime-skill-dir>/scripts/experiment_ledger.py validate \
  experiments/S3/experiment-result.json
python3 <runtime-skill-dir>/scripts/experiment_ledger.py merge \
  --ledger experiments/experiment-ledger.json \
  --result experiments/S3/experiment-result.json \
  --output experiments/experiment-ledger.json
```

Use schema 2. Include `child_agent_id`, per-round `profiling_analysis_agent_ids`,
`profiling_analysis_result`, baseline/candidate profile paths and fingerprints,
`declared_change_set`, all five experiment fingerprints, lifecycle state, decisions,
blockers, and Chinese `next_agent_guidance_zh`. All artifact paths are relative and
each path has one semantic role. Agent IDs, candidate profiles, and analysis outputs
must be fresh. Legacy schema 1 records are read-only history and never promote into
active conditional performance evidence.

Within the lifecycle, execution correctness and the fresh candidate profile gate
profiling evidence. Compat/verify/replay fields, when present, describe an independent
audit and never gate profiling, round continuation, or profile-vs-baseline decisions.

The CLI treats the directory containing `experiment-result.json` as the artifact root.
It resolves and reads every `profiling_analysis_result`, rejects physical path escape
or missing inputs, replays the sibling analyzer read-only, and checks schema, content
hash, identity, Agent, round, all fingerprints, and each `(round_id, range_id)` decision
against analyzer `per_sk_decisions` and `scope_actions`. For every analysis round, the
union of `performance_scope_decisions`, `verified_ranges`, and
`unresolved_performance_ranges` must equal the producer range set, with each range
recorded exactly once. Preserve producer mapping fields; never submit
`mapping_reliable`. `kernel_projection_structural + exact_projected_trace` is performance-exact
but not source-actionable. Only `source_scope_map + exact` with proven source offsets
can advance an applied/verified range-optimized FINAL source lifecycle. A
separate `promotion_mode=whole_scope` record may retain exact projected local
neutral/regressed diagnostics because it performs no source action; it requires full
inventory coverage and rejects any `insufficient_evidence`. Historical
`sk_meta_node_ids + exact`, raw IDs, names, and local windows are diagnostic only.
Object-only validation or merge without an artifact root is structural only and cannot
create conditional evidence.

Conditional evidence is valid only for the exact composite of source revision,
baseline config, candidate config, control, workload, and `range_id`. Do not apply it
to a different fingerprint set.

## Options And Diagnosis

The pre-profile O stage does not use `recommend_sk_strategy.py`; it uses the frozen
option matrix, accepted-value evidence, correctness, and `analyze_performance.py
--option-trial`. Use `recommend_sk_strategy.py` only after reliable metadata and
profiling analysis.
Pass one `--profiling-analysis NAME=PATH` per round in increasing order, repeating the
same family `NAME` for its BASE/P history. The strategy validates and replays every
round, records the source round/path for each P action, and verifies that its target
range was actually pruned.
By default the strategy records
`scope_strategy.source_range_optimization_status=not_requested`, emits no P plan or
P-specific blocker, and leaves eligible `whole_scope_clean_validation` unchanged.
To authorize the optional branch, pass repeatable
`--source-range-optimization NAME` once for each exact candidate family named by the
user or frozen experiment plan. Include the flag on every invocation that supplies
that family's P/FINAL history; the flag itself is the explicit authorization and
unknown, empty, or duplicate names fail closed. For example:

```bash
python3 scripts/recommend_sk_strategy.py \
  --profiling-analysis S3=S3-BASE/profiling-analysis/profiling-analysis-result.json \
  --profiling-analysis S3=S3-P1/profiling-analysis/profiling-analysis-result.json \
  --source-range-optimization S3
```

Treat every recommendation as a hypothesis until the active environment probe accepts
the exact value. The probe must be a saved `ready=true` artifact collected after the
same approved CANN setup as inference; an import/load failure leaves option acceptance
unknown rather than rejected. DCCI, `auto_op_parallel`, ValueWait, event, task-breaker,
and aggressive controls never replace dependency or correctness proof.

Use Cube/Vector overlap and SK child trace to distinguish preserved parallelism from
serialization. In the pre-profile O stage, wrapper acceptance plus runtime prerequisites
is enough to schedule an `auto_op_parallel`, aggressive, or initial DCCI disable-all
clean trial; no per-SK hypothesis is required. DCCI before/after are never independently
scheduled. They appear only together after a measured disable-all regression and
`auto_op_parallel` still requires direct
scheduling evidence, and DCCI still requires direct scalar/cache evidence plus explicit
runtime state. Never infer a local cause from option-level end-to-end gain.

## Clean Performance And Reporting

Clean timing disables profiler, metadata, traces, debug sync, and event diagnostics.
Use clean timing first to select one S candidate, then for the winner-only O stage, and
again after the winner's profiling/optimization lifecycle. Screening and O samples do
not prove final promotion. Compare at least three unchanged whole-scope or
range-optimized FINAL candidate
processes against the frozen five-run S0 with the same warmup and TP worst-rank
aggregation. Require the agreed mean threshold plus P90 and standard-deviation
non-regression; per-SK interval gain is diagnostic, not a substitute.

Read [final-report-template.md](final-report-template.md) and
[report-summary.md](report-summary.md). Every terminal outcome must reach final reporting,
even if no legal final measurement exists. Begin the report with the overall outcome,
key S0/experiment/final comparison, and exact winner or verified fallback. Screening,
option and diagnostic timings are not final clean measurements. New final handoffs
include structured `report_summary`; legacy ledgers remain readable without it.
Every human per-SK diagnostic performance table must include interval P50, duration sum P50, SK P50, MAD threshold,
classification/action, graph occurrence, mapping method/confidence, source-actionable
status, analysis Agent, and conditional evidence.
These per-SK columns do not apply to the compact E2E summary table.
Write per-round analysis and the parent summary in Chinese. Preserve negative evidence,
unresolved mappings, failures, artifact paths, fingerprints, and exact commands.
Before reporting completion, verify that every non-environment terminal failure links
to a complete experiment-local failure scene under
[failure-scene-reporting.md](failure-scene-reporting.md).

## Completion Checklist

- S0 five-run stability passed under frozen controls.
- Layer/operator/stream inventory is complete or explicitly blocked.
- Every S-screening candidate passed execution/correctness and clean timing without
  candidate profiling, SK mapping, calibration, P, or source-range work.
- Every executed S-screening config and archived `run-*/config.yaml` explicitly set
  `super_kernel_optimize_options: {}` and `super_kernel_debug_options: {}`; the
  preflight and selection-time option-control validations both passed.
- The frozen screening matrix covers automatic AOT, broad decode, per-block, and
  semantic-segment strategies; every entry is executed/blocked/skipped with valid
  evidence, and its executed IDs exactly match the selection inputs. No candidate was
  skipped because another candidate was already eligible.
- The deterministic screening artifact selected at most one winner; only that winner
  and its P/FINAL descendants received fresh read-only profiling analysis.
- The selected winner was frozen as `Sbest-SEED`; its DCCI family,
  `auto_op_parallel`, aggressive, and applicable experimental option matrix was fully
  settled before profiling. Ordinary O trials changed one accepted option value. DCCI
  ran disable-all first; standalone before/after trials were absent, and any combined
  repair was backed by paired per-SK plus child evidence and compared directly with O0.
  Every retained trial passed correctness and clean timing, and only measured
  incremental gains entered `Sbest-BASE`.
- Every delta-limit recollection stayed within one logical round, used no more than
  three attempts, selected only a successful fresh attempt, and preserved all failed
  attempt evidence; a minimum-three-occurrence overflow blocked the round.
- If optional multi-stream tuning was invoked, its request/result contracts validated,
  all non-accepted outcomes retained the exact incumbent, and any accepted output
  entered a new derived family through a fresh ordinary BASE rather than an M round.
- Every winner completed one batched SMAP when any performance-exact SK lacked a source
  interval; exact/partial/skipped counts and per-range reasons are recorded. Automatic
  winners used `stable_source`; named/manual winners used `stable_marker`.
- P decisions prune only source-map exact neutral/regressed ranges with proven source
  offsets and preserve execution.
- P/FINAL inherit the exact `Sbest-BASE` option maps and contain no option action,
  option JSON Pointer, or declared option change.
- Promotion uses either unchanged whole-scope full performance-exact mapping with no
  insufficient evidence after source mapping is settled, or a range-optimized FINAL
  with source-actionable beneficial ranges; the selected candidate then passes clean
  timing.
- The schema 2 ledger validates and all artifact roles/fingerprints are auditable.
- The Chinese report covers every attempted, failed, blocked, and skipped round.
