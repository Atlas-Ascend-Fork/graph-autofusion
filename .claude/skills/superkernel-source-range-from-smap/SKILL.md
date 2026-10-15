---
name: superkernel-source-range-from-smap
description: Use when starting an independent Ascend SuperKernel P/FINAL source-range experiment from a completed, verified parent session with accepted BASE profiling, schema 2 ledger, and exact SMAP evidence.
---

# SuperKernel Source Range From SMAP

Create a new immutable child session for P/FINAL instead of reopening or editing the
completed parent session. The user's approval authorizes evidence import; it never
bypasses evidence validation.

## Entry Gate

Require all of the following:

- one completed parent `superkernel-auto-tune-session-v1` that passes `verify`;
- the sealed successful BASE/SMAP handoff from that parent;
- a digest-valid schema 1.2 independent profiling analysis;
- the matching schema 2 experiment ledger and BASE round;
- a digest-valid `source_scope_map_v2` with the same source revision and at least one
  `exact_cover` range;
- explicit user approval to import those exact artifacts.

Every imported file must live under the parent artifact root and be named by the
selected BASE/SMAP handoff. The controller writes `imports/evidence-import.json`, binds
all file digests, and revalidates them on every resume and dispatch. A stale, moved,
modified, unbound, or merely verbal result is a blocker.

## Create The Derived Session

Run the standalone wrapper from this Skill directory:

```bash
python3 scripts/source_range_from_smap.py \
  --parent-session <parent-artifact-root>/auto-tune-session.json \
  --session <new-artifact-root>/auto-tune-session.json \
  --artifact-root <new-artifact-root> \
  --session-id <new-run-id> \
  --profiling-analysis <parent-artifact-root>/<base-analysis.json> \
  --ledger <parent-artifact-root>/<experiment-ledger.json> \
  --source-scope-map <parent-artifact-root>/<source-scope-map.json> \
  --approve-imported-evidence
```

The only legal first step is `optional-source-range`; Final follows it. Intake, S0,
Stage A, Stage O, and BASE are not recreated as fake sealed child steps. Their accepted
evidence is referenced through the import manifest and must be described as imported in
the final report.

## Dispatch And Execute

Use the sibling `superkernel-auto-tune` controller for `next`, `dispatch`, `run-next`,
`seal`, `resume`, and `verify`. For each pending step, create a fresh host-native generic
subagent, use `step.agent_id` as the visible task name, and explicitly instruct it to
load and follow `step.skill`.

The P/FINAL worker must follow
[superkernel-optional-experiments](../superkernel-optional-experiments/SKILL.md) and the
preserved source-range contract in
[controller-legacy-contract.md](../superkernel-auto-tune/references/controller-legacy-contract.md).
Keep the imported BASE Option map frozen. Run isolated P trials only for exact mapped
ranges authorized by the analysis, recollect correctness, clean timing, profile, and
independent analysis for every trial, then build FINAL only from retained beneficial
ranges. Local SK improvement never substitutes for the final clean E2E gate.

Always run `sk-final-e2e-report` after the source-range branch and report beneficial,
no-gain, failed, blocked, invalid, or not-run outcomes with the imported configuration,
scope strategy, Option map, evidence paths, and exact fallback incumbent.
