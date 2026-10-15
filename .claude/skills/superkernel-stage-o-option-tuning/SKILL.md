---
name: superkernel-stage-o-option-tuning
description: Tune wrapper options for one frozen SuperKernel Stage A winner through clean incremental trials before profiling.
---

# SuperKernel Stage O Option Tuning

## Control-Plane Handoff

## Inputs

Accept only `stage_o_option_tuning` for `sk-stage-o-option-tuning`, with one frozen
`Sbest-SEED`, accepted option values, and S0-bound clean artifacts.

## Execution

Run `scripts/execute_phase.py --task <dispatch-task.json>` first. It validates the
current Stage O task and emits option, ledger, and per-round report tools.

Read Stage O and its DCCI collection contract in
[controller-legacy-contract.md](../superkernel-auto-tune/references/controller-legacy-contract.md),
`winner-option-sweep.md`, and `dcci-option-tuning.md`. Freeze the option matrix,
establish five-run `O0-INCUMBENT`, run ordinary one-RFC6901-pointer trials with
`analyze_performance.py --option-trial`, and restore the incumbent after each
non-accepted result. DCCI begins only with disable-all; before/after child trials are
illegal. A disable-all regression requires paired manifests and read-only analysis.

## Required Handoff

Return the settled option matrix, all evidence/fallback scenes, retained exact option
maps, `Sbest-BASE` identity, ledger paths, and Chinese next-step guidance. Do not
profile BASE or start optional work before the controller seals this result.

Run only when delegated by `superkernel-auto-tune` as `sk-stage-o-option-tuning`.
Read the controller's Stage O and `winner-option-sweep.md` contracts. Start with a new
five-run O0 incumbent, freeze the complete winner option matrix, and test one accepted
option value per clean trial against the current stable incumbent. Preserve correctness,
sample, median, P90, standard-deviation, timeout and rollback gates.

Use the DCCI disable-all first trial and its sole paired-diagnostic/combined-repair
exception exactly as specified. Do not profile ordinary options or modify source/scope.
Settle all matrix entries and return a frozen `Sbest-BASE`, including failure scenes and
the exact retained option maps even when no option gained performance.
