---
name: superkernel-stage-a-scope-selection
description: Screen the complete SuperKernel scope-strategy matrix with clean timing and select one eligible winner for deep tuning.
---

# SuperKernel Stage A Scope Selection

## Control-Plane Handoff

## Inputs

Accept only `stage_a_scope_selection` for `sk-stage-a-scope-selection`, with stable
S0, source-change permission, and frozen screening budget.

## Execution

Run `scripts/execute_phase.py --task <dispatch-task.json>` first. It validates the
current Stage A task and emits matrix-validation, recommendation, and analysis tools.

Read Stage A and Reconstruct Layers And Draft Scopes in
[controller-legacy-contract.md](../superkernel-auto-tune/references/controller-legacy-contract.md),
plus `layer-and-pipeline-analysis.md` and `adaptation-playbook.md`. Freeze all four
required strategies in `superkernel-screening-matrix-v2`; validate explicit empty
option maps with the owned `validate_stage_a_options.py` tool; then collect correctness
and clean timing only. Rank only after all entries settle using the owned
`analyze_performance.py --selection-only` tool. Invoke both through
`scripts/execute_phase.py --tool ... -- <tool args>`. Do not create P, profiling, SMAP,
or O work.

## Required Handoff

Return matrix, config validation, S0-bound selection summary, all scope decisions,
and `Sbest-SEED` or a no-winner blocker. No winner is never permission to start O.

Run only when delegated by `superkernel-auto-tune` as `sk-stage-a-scope-selection`.
Read the controller's Stage A contract and `validate_stage_a_options.py` protocol.
Freeze a complete screening matrix containing `automatic_aot`, `broad_decode`,
`per_block`, and `semantic_segment`. Every executed candidate uses explicit empty
SuperKernel optimize/debug maps, complete correctness, and clean timing only.

Do not profile a candidate, create a source map, calibrate, run P/FINAL, or test an
Option. Settle every matrix item as executed, blocked, or skipped with Chinese evidence,
then call the preserved deterministic selection analysis. Return either one frozen
`Sbest-SEED` with ranked eligible candidates, or a no-winner result for final reporting.
