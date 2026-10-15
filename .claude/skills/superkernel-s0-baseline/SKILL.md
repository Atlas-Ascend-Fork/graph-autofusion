---
name: superkernel-s0-baseline
description: Establish the frozen five-process SK-off clean performance baseline for a validated SuperKernel tuning session.
---

# SuperKernel S0 Baseline

## Control-Plane Handoff

## Inputs

Accept only `s0_baseline` for `sk-s0-baseline`, with Intake's frozen command,
workload, source/config/control fingerprints, and clean artifact root.

## Execution

Run `scripts/execute_phase.py --task <dispatch-task.json>` first. It validates the
current S0 task and emits this phase's baseline-analysis entrypoint.

Read Controlled Baseline in
[controller-legacy-contract.md](../superkernel-auto-tune/references/controller-legacy-contract.md).
Collect exactly five independent SK-off clean processes with profiler, metadata,
event trace, debug sync, and SK profiling disabled. Archive each config and invoke
`scripts/execute_phase.py --tool analyze_performance.py -- --baseline <S0-root>
--warmup 8` with the current task and lease arguments.

## Required Handoff

Return S0 stability, five-run artifacts, fingerprint binding, optional diagnostic
profile paths, and `succeeded` only after the spread gate passes. Otherwise return
`blocked` with instability evidence; never start Stage A.

Run only when delegated by `superkernel-auto-tune` as `sk-s0-baseline`. Read the
controller's Controlled Baseline contract. Use the frozen intake identity without any
SK, profiler, metadata, trace, debug-sync or calibration controls. Run exactly five
independent clean processes, omit the required warmup samples, aggregate TP worst-rank
decode means, and require the preserved spread gate.

If the baseline is unstable, emit a blocked phase handoff and do not start Stage A. On
success, seal S0 clean evidence, config/control/workload/source fingerprints and the
baseline summary for all later comparisons.
