---
name: superkernel-optional-experiments
description: Run only user-authorized isolated SuperKernel multistream and P/FINAL source-range experiments with exact incumbent fallback.
---

# SuperKernel Optional Experiments

## Control-Plane Handoff

## Inputs

Accept only `optional_experiments` for `sk-optional-experiments`; the returned
`optional_branch` is authoritative: `none`, `multistream`, or `source-range`.

## Execution

Run `scripts/execute_phase.py --task <dispatch-task.json>` first. It validates the
current optional-branch task and emits derived-family and ledger tools.

Read optional branch rules in
[controller-legacy-contract.md](../superkernel-auto-tune/references/controller-legacy-contract.md).
For `none`, return `not_requested`. For multistream, delegate only to the preserved
multistream skill under an isolated worktree/config/cache/artifact branch and restore
the BASE incumbent on every non-accepted result. Source-range requires explicit user
authorization, exact source mapping, and frozen BASE options. In `both`, wait for the
controller's inserted derived BASE/SMAP step before source-range work.

The task may come from a full session or a derived session created through the
controller's approved evidence-import protocol. For a derived task, load the bound
`evidence_import` manifest; do not demand fake child Intake/S0/A/O/BASE handoffs and do
not accept unbound paths supplied only in the prompt.

## Required Handoff

Return branch request evidence, isolated outcome, incumbent/fallback identity,
ledger/receipt paths, blockers, and Chinese guidance. Local overlap is not promotion.

Run only when delegated by `superkernel-auto-tune` as `sk-optional-experiments`. Read
the controller's optional multistream and P/FINAL contracts. The session's frozen mode
is authoritative: `none` returns `not_requested`; `multistream` runs only the isolated
multistream path; `source-range` runs only P/FINAL; `both` runs multistream first.

Use `sk-multistream-tuning-executor` with the existing multistream Skill after its
capability and collection preflight. An accepted result creates a derived family and
must return to BASE profile/source mapping before P/FINAL. A no-gain, blocked, failed,
invalid, or absent result restores the exact incumbent. Use
`sk-source-range-experiment-executor` only for explicitly requested P/FINAL and preserve
the frozen BASE option maps with no option action. Never combine multiple factors or
promote local scheduling evidence as clean E2E gain.
