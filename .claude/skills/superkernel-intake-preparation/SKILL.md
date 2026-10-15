---
name: superkernel-intake-preparation
description: Prepare and validate an Ascend SuperKernel tuning session before S0 by freezing runtime, workload, controls, and optional experiment intent.
---

# SuperKernel Intake Preparation

## Control-Plane Handoff

## Inputs

Accept only `intake_preparation` for `sk-intake-preparation`, with the artifact root,
target revision, command, model/config paths, workload/correctness definition,
source-change permission, budget, and user-selected optional mode.

## Execution

Run `scripts/execute_phase.py --task <dispatch-task.json>` first. It validates the
current Intake task and emits this phase's environment and lease tool entrypoints.

Read Required Intake and Non-Negotiable Gates in
[controller-legacy-contract.md](../superkernel-auto-tune/references/controller-legacy-contract.md),
then `intake-checklist.md`. Confirm topology, approved CANN setup, static compile
safety, scope APIs, frozen controls, and artifact root. Run option probing only in the
target launcher's approved environment. Do not launch S0 or edit source.

## Required Handoff

Return one `superkernel-auto-tune-phase-result-v1` with environment evidence,
frozen fingerprints, optional mode, blockers, and Chinese guidance. Missing or
unsupported prerequisites are `blocked` with a failure scene; do not guess values.

Run only when delegated by `superkernel-auto-tune` as `sk-intake-preparation`.
Read the controller's preserved intake, static-compile safety, shared-NPU lease, and
failure-scene contracts. Confirm the target repository, revision, command, model
configuration, workload, correctness method, artifact root, budget and success
threshold. Probe the active wrapper in the exact CANN runtime used by inference.

Ask and freeze exactly one optional mode: `none`, `multistream`, `source-range`, or
`both`. Record it in the session handoff; later stages must not expand it. Stop before
NPU experiments on a critical mismatch, unavailable A2/A3 static runtime, inactive scope
API, or a probe that did not execute. Do not describe an unrun option probe as rejection.

Write the standard phase report and handoff. On success, the handoff supplies frozen
five identity inputs, accepted option values, lease root, and the optional mode for S0.
