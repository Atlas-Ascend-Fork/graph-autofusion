---
name: superkernel-runtime-common
description: Internal shared runtime used by the SuperKernel Auto Tune phase skills. It owns reusable analysis, lease, lifecycle, ledger, and reporting executables; users should invoke the phase skills rather than this internal skill directly.
---

# SuperKernel Runtime Common

This is an internal implementation skill for `superkernel-auto-tune` and its
phase skills. It provides the reusable deterministic command-line tools
listed in each phase's `phase.json` manifest.

Do not use it as a workflow entry point. Start with `superkernel-auto-tune`,
which dispatches a fresh host-native generic subagent for the appropriate phase and
records every handoff.
