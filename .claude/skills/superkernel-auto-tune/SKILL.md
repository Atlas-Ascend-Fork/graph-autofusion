---
name: superkernel-auto-tune
description: Use when orchestrating an evidence-gated Ascend SuperKernel tuning session across intake, baseline, scope selection, option tuning, profiling, optional experiments, and final E2E reporting.
---

# SuperKernel Auto Tune

`superkernel-auto-tune` is the sole controller. It freezes one session, dispatches a
fresh host-native generic subagent for each pending phase, validates its structured
handoff, and always reaches the Final report step. It never performs a tuning phase
itself and requires no pre-registered custom Agent profiles.

It supports both a full Intake entry and evidence-gated derived entries. Never reopen a
completed session or mark omitted phases as artificially sealed.

## Start And Resume

At Intake, ask once for the optional mode and freeze `none`, `multistream`,
`source-range`, or `both`.

```bash
python3 scripts/auto_tune_session.py init \
  --session <artifact-root>/auto-tune-session.json \
  --artifact-root <artifact-root> --session-id <run-id> \
  --optional-mode <none|multistream|source-range|both>
python3 scripts/auto_tune_session.py next \
  --session <artifact-root>/auto-tune-session.json
python3 scripts/auto_tune_session.py dispatch \
  --session <artifact-root>/auto-tune-session.json \
  --output <artifact-root>/dispatch-task.json
```

`run-next` remains available when the host provides a command adapter. It appends
`--task <path> --result <path>`, captures logs, validates the result, seals the handoff,
and records an immutable dispatch receipt:

```bash
python3 scripts/auto_tune_session.py run-next \
  --session <artifact-root>/auto-tune-session.json \
  --runner-command-json <agent-host-command.json> \
  --timeout-seconds 3600
```

The host adapter launches a generic subagent and explicitly passes the task's Skill;
it does not resolve a custom Agent registration. A failed, timed-out, or invalid attempt
remains under `dispatches/` and leaves the step pending for a recoverable retry.

After an interruption, do not infer progress from chat history. Resume only from the
persisted session:

```bash
python3 scripts/auto_tune_session.py resume \
  --session <artifact-root>/auto-tune-session.json
```

## Import Accepted Evidence

When the user later requests multistream or P/FINAL, derive a new session from the
completed parent. Require explicit approval plus the exact BASE/SMAP-bound schema 1.2
analysis and schema 2 ledger. Source-range and `both` also require the matching
digest-valid `source_scope_map_v2` with an `exact_cover` range.

```bash
python3 scripts/auto_tune_session.py derive-session \
  --parent-session <parent-root>/auto-tune-session.json \
  --session <new-root>/auto-tune-session.json --artifact-root <new-root> \
  --session-id <new-run-id> --optional-mode <multistream|source-range|both> \
  --profiling-analysis <parent-root>/<analysis.json> \
  --ledger <parent-root>/<ledger.json> \
  --source-scope-map <parent-root>/<source-scope-map.json> \
  --approve-imported-evidence
```

For P/FINAL only, use the equivalent `source-range-from-smap` command or the standalone
`superkernel-source-range-from-smap` Skill. The derived task carries the immutable
`evidence_import` reference. Revalidate it before every dispatch. `both` begins with
multistream; acceptance still inserts `base-profile-derived` before P/FINAL.

## Dynamic Native Delegation

Run `dispatch`, read the generated task, then create one fresh generic subagent using
the current host's built-in delegation mechanism. `agent_id` is the stable logical
phase/task/audit identifier and the user-visible task name; it is not the name of a
registered Agent profile.

| Host | Built-in generic subagent |
|---|---|
| Codex | `worker`; use `default` when only the generic default role is available |
| Claude Code | `general-purpose` |
| OpenCode | `general` |

The controller must include all of the following in the child prompt:

- the exact `phase-task.json` path and its `step.agent_id`, `step.phase`, and `step.skill`;
- an explicit instruction to load and follow `step.skill`, because a generic child does
  not inherit this controller Skill's instructions;
- the matching phase `scripts/execute_phase.py --task <phase-task.json>` validation
  command before any experiment command;
- the exact writable result path and required
  `superkernel-auto-tune-phase-result-v1` schema;
- the boundary that the child owns only this phase and must not dispatch or execute a
  later phase.

Do not reuse a child between phases. Wait for it to finish, validate and seal its result,
then query `next` again. If the host cannot create a native generic subagent, keep the
step pending, preserve the dispatch failure scene, and report the blocker; the
controller must not impersonate the phase worker.

The generated `superkernel-auto-tune-phase-task-v1` binds the current session
fingerprint, input handoffs, one legal step, and the required result schema. A child
must first run its own `scripts/execute_phase.py --task <dispatch-task.json>`; stale or
wrong-logical-worker tasks are rejected before it can claim a handoff.

Each stage entrypoint prints its owned tool plan by default. To run one listed tool, use
`scripts/execute_phase.py --task <dispatch-task.json> --tool <listed-tool> --lease-root
<lease-root> --device-id <id> -- <tool args>`; the wrapper refuses tools outside the
phase allowlist and executes every child command through the shared NPU lease runner.

| Logical worker ID | Skill | User-visible responsibility |
|---|---|---|
| `sk-intake-preparation` | `superkernel-intake-preparation` | 环境确认与实验准备 |
| `sk-s0-baseline` | `superkernel-s0-baseline` | S0 基线性能测量 |
| `sk-stage-a-scope-selection` | `superkernel-stage-a-scope-selection` | Stage A 框定方式选择 |
| `sk-stage-o-option-tuning` | `superkernel-stage-o-option-tuning` | Stage O Option 调优 |
| `sk-base-profile-source-mapping` | `superkernel-base-profile-source-mapping` | BASE profile、独立分析与 SMAP |
| `sk-optional-experiments` | `superkernel-optional-experiments` | 双流与 P/FINAL 实验 |
| `sk-final-e2e-report` | `superkernel-final-e2e-report` | 最终 E2E 确认与报告 |

Do not delegate a later phase on a verbal claim. The child returns one
`superkernel-auto-tune-phase-result-v1` document; the controller seals and verifies it
before asking `next` again:

```bash
python3 scripts/auto_tune_session.py seal \
  --session <artifact-root>/auto-tune-session.json --result <handoff.json>
python3 scripts/auto_tune_session.py verify \
  --session <artifact-root>/auto-tune-session.json
```

Sealing writes immutable `phase-result.json` and `PHASE_REPORT.md`. The session
validator rejects any non-canonical phase order or phase-invalid status. A blocked
Intake, S0, Stage A, BASE, or blocked/failed Stage O automatically creates `not_run`
scenes for inapplicable later phases and leaves only Final legal. In `both`, an accepted
multistream result inserts a mandatory fresh `base-profile-derived` BASE/SMAP step
before source-range work.

## Support Delegation

The support IDs below are also logical task/audit IDs. Whenever a phase needs one,
spawn another fresh host-native generic subagent, use the logical ID as its visible
task name, and explicitly load the referenced tool Skill. They are not custom profiles.

- `sk-fusion-performance-analyst` reads
  [superkernel-fusion-performance-analysis](../superkernel-fusion-performance-analysis/SKILL.md)
  for immutable per-SK analysis.
- `sk-multistream-tuning-executor` reads
  [superkernel-multistream-performance-tuning](../superkernel-multistream-performance-tuning/SKILL.md).
- `sk-source-range-experiment-executor` reads
  [superkernel-optional-experiments](../superkernel-optional-experiments/SKILL.md) and
  executes only the explicitly authorized P/FINAL source-range branch.
- `sk-failure-isolation-analyst` must read and follow
  [superkernel-sk-failure-isolation](../superkernel-sk-failure-isolation/SKILL.md)
  before changing a failed scope.
- `sk-prof-timeline-analyst` reads
  [superkernel-sk-prof-timeline](../superkernel-sk-prof-timeline/SKILL.md) only for
  focused Cube/Vector scheduling inspection.

## Domain Contracts

The preserved hard evidence gates, safety constraints, option rules, profiling rules,
and historical failure limitations are in
[controller-legacy-contract.md](references/controller-legacy-contract.md). Each stage
skill identifies the exact sections it owns. The schema 2 experiment ledger remains
the authority for experiment evidence. Its structured `final_e2e` object is the sole
terminal performance verdict and records classification, exact candidate, SK scope
strategy, Option map, evidence, reason, and clean baseline/candidate metrics when
executed. Final binds the relative `ledger_path`; the controller validates status
consistency and renders `FINAL_E2E_REPORT.md` for every terminal outcome, with result-first
`report_summary` from new workers per [report-summary.md](references/report-summary.md).
