---
name: superkernel-final-e2e-report
description: Perform the final legal SuperKernel clean E2E validation and produce the complete Chinese tuning report for every terminal outcome.
---

# SuperKernel Final E2E And Report

## Control-Plane Handoff

## Inputs

Accept only `final_e2e_report` for `sk-final-e2e-report`, with sealed session, frozen
S0 identity, exact incumbent, and schema 2 ledger relative path.

## Execution

Run `scripts/execute_phase.py --task <dispatch-task.json>` first. It validates the
current Final task and emits final timing and ledger tools. The controller owns report
rendering after it seals this phase.

Read Clean Performance And Reporting and Completion Checklist in
[controller-legacy-contract.md](../superkernel-auto-tune/references/controller-legacy-contract.md),
then `final-report-template.md`. Run final clean E2E only for a legal whole-scope or
explicit FINAL path. Screening, option, diagnostic, and profile timing are never final
timing. If no comparison is legal, report `not_run` and the evidence.

## Required Handoff

Return the seven Chinese `details_zh` sections: `environment`, `s0`, `stage_a`,
`stage_o`, `base_profile_source_mapping`, `optional_experiments`, `final_e2e`; include
the relative `ledger_path`, config/fingerprint, outcome, and blockers. The ledger must
contain `final_e2e.classification`, `candidate_id`, `scope_strategy`, `option_config`,
`evidence_artifacts`, and `reason_zh`. `beneficial` and `no_gain` additionally require
clean baseline/candidate metrics and a mathematically consistent `improvement_pct`.
The result status must match that classification. The controller seals it and renders
`FINAL_E2E_REPORT.md` automatically.

For every new final handoff, including failed/blocked/not_run outcomes, populate the
ledger's `report_summary` using [report-summary.md](../superkernel-auto-tune/references/report-summary.md).
Collect frozen S0 and key measured attempts even when `experiments` is empty. Read the
screening/option/optional artifacts; do not replace missing measurements with zero.
The title must be followed immediately by the outcome, key comparison table, and
winner configuration or verified fallback, before environment/session details.
Show the best measured ineligible attempt when no candidate passes, explicitly as
non-winner. Only final clean validation can establish the terminal winner. Include
both effective optimize/debug option maps, config path/fingerprint and promotion path
for a winner; unknown options are N/A, not assumed defaults. Preserve all failure
scenes in the body. Producing a report never authorizes an otherwise illegal E2E run.

Run only when delegated by `superkernel-auto-tune` as `sk-final-e2e-report`. Read the
controller's whole-scope promotion, ledger, and final-report contracts. When a legal
candidate exists, run the unchanged whole-scope or requested FINAL clean validation
against frozen S0; do not reuse screening, option, diagnostic, or profiling timing.

Always produce the final Chinese report and session handoff. It must explain environment
status, S0, Stage A scope selection, Stage O option matrix, BASE classifications/source
mapping, requested optional branches, exact configuration/fingerprints, clean E2E
result, incumbent fallback, and every blocker/failure scene. If no legal E2E comparison
exists, mark it `not_run` with evidence rather than claiming a gain or loss.
