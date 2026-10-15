# SuperKernel Auto Tune Session Contract

The executable authority is `scripts/auto_tune_session.py`; schemas live in
`schemas/auto-tune-session-v1.schema.json`, `schemas/phase-task-v1.schema.json`,
`schemas/phase-result-v1.schema.json`, `schemas/dispatch-receipt-v1.schema.json`, and
`schemas/final-e2e-v1.schema.json`. Derived sessions additionally bind
`schemas/evidence-import-v1.schema.json`.
The controller creates a session before Intake, obtains the only legal next step with
`next`, and seals each stage result with `seal`. Do not hand-edit a session or a sealed
`phase-result.json`.

```bash
python3 scripts/auto_tune_session.py init \
  --session artifacts/auto-tune-session.json \
  --artifact-root artifacts \
  --session-id run-20260905-01 \
  --optional-mode both
python3 scripts/auto_tune_session.py next --session artifacts/auto-tune-session.json
python3 scripts/auto_tune_session.py dispatch \
  --session artifacts/auto-tune-session.json --output artifacts/dispatch-task.json
python3 scripts/auto_tune_session.py run-next \
  --session artifacts/auto-tune-session.json \
  --runner-command-json agent-host-command.json
python3 scripts/auto_tune_session.py seal \
  --session artifacts/auto-tune-session.json --result handoff.json
python3 scripts/auto_tune_session.py verify --session artifacts/auto-tune-session.json
```

`seal` writes immutable `phases/<step-id>/phase-result.json` and
`PHASE_REPORT.md`. `run-next` additionally records one immutable attempt directory
under `dispatches/` and binds a successful receipt digest into the sealed session step.
Sealing a final handoff writes `FINAL_E2E_REPORT.md`; its
`details_zh` must cover environment, S0, Stage A, Stage O, BASE/SMAP, optional
experiments, and final clean E2E, including `not_run` outcomes.

New final workers also populate the schema 2 ledger's optional `report_summary` using
[report-summary.md](report-summary.md) and `schemas/report-summary-v1.schema.json`.
The shared renderer validates it before sealing and puts the outcome, key comparison
table, and winner/fallback ahead of metadata. Legacy ledgers remain readable with
explicit N/A fields. `render-final-report --summary <json>` supplies a read-only
display override for historical reports; it never changes the terminal classification.

The controller persists one `superkernel-auto-tune-session-v1` document in the frozen
artifact root. It is an orchestration index, not a replacement for the schema 2 ledger,
profile manifests, multistream result, or phase-local reports.

`dispatch` writes one `superkernel-auto-tune-phase-task-v1`. It binds the exact pending
step, logical worker ID and Skill, sealed predecessor digests, and current session
fingerprint. The controller creates a fresh host-native generic subagent, gives it the
logical ID as its visible task name, and explicitly tells it to load the bound Skill.
The local phase `scripts/execute_phase.py` rejects stale or misrouted tasks before the
subagent invokes its owned tool plan. No root-level custom Agent registration is part of
this contract. The validator accepts one of these canonical schedules:

- `full`: Intake -> S0 -> Stage A -> Stage O -> BASE -> optional -> Final;
- `optional-from-base`: multistream -> optional source-range -> Final;
- `source-range-from-smap`: optional source-range -> Final.

An accepted multistream always inserts a fresh derived BASE before source-range work in
`both`, including a derived session.

## Required Session Fields

```json
{
  "schema_version": "superkernel-auto-tune-session-v1",
  "session_id": "SK-20260905-001",
  "optional_mode": "none|multistream|source-range|both",
  "entrypoint": "full|optional-from-base|source-range-from-smap",
  "status": "active|completed",
  "artifact_root": "/absolute/frozen/artifact/root",
  "steps": [
    {
      "step_id": "intake-preparation",
      "phase": "intake_preparation",
      "agent_id": "sk-intake-preparation",
      "skill": "superkernel-intake-preparation",
      "state": "pending|sealed"
    }
  ]
}
```

A non-full session also contains `evidence_import.path` and `evidence_import.sha256`.
The referenced manifest records explicit approval, the completed parent session and
BASE handoff digests, and the exact imported analysis, ledger, and optional SMAP file
digests. Every load, resume, verify, and dispatch rechecks those external files. The
parent and child artifact roots must be disjoint.

`agent_id` is a stable logical responsibility and audit key. It is validated across the
phase manifest, task, result, and session, but it does not select a host-specific custom
Agent profile.

Each sealed step binds its immutable `result_path`, `result_sha256`, terminal status,
and optional successful `dispatch_receipt_path`/digest. Detailed inputs, decisions,
blockers, artifacts, failure scene, and Chinese guidance live in its phase result.

## Transition Rules

1. Intake freezes the optional mode. A critical environment blocker goes directly to
   final reporting with E2E `not_run`.
2. S0 may advance only when the preserved five-run stability gate passes.
3. Stage A returns one `Sbest-SEED` or a terminal no-winner result. A no-winner result
   goes to final reporting without Stage O.
4. Stage O settles its whole matrix before returning `Sbest-BASE`. Individual failed or
   rejected trials remain phase evidence and do not by themselves terminate the matrix.
5. BASE profile/source mapping returns a legal current incumbent or a default-winner
   failure. Only the existing ranked-candidate policy may re-enter Stage O for another
   Stage A candidate.
6. Optional mode `none` records `not_requested`. With `both`, execute multistream first.
   A valid multistream acceptance creates a derived family, marks the old BASE
   `superseded`, and returns to BASE profile/source mapping before P/FINAL. Any other
   multistream terminal state preserves its precise incumbent.
7. P/FINAL receives only the current mapped BASE incumbent and never changes frozen
   Option maps. Its non-accepted terminal states preserve that incumbent.
8. Final reporting always runs. It runs a clean E2E comparison only when the preserved
   promotion gate admits one; otherwise it records the decisive evidence as `not_run`.

Every terminal phase report must describe its result in Chinese and include the source
path of its failure scene when the status is `failed` or a post-start `blocked` result.
