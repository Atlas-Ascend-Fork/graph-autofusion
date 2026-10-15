---
name: superkernel-base-profile-source-mapping
description: Collect winner profiles, delegate independent per-SK analysis, and settle exact SuperKernel source mapping before later experiments.
---

# SuperKernel BASE Profile And Source Mapping

## Control-Plane Handoff

## Inputs

Accept only `base_profile_source_mapping`, including `base-profile-derived` with
`derived_family_rebase=true`, with `Sbest-BASE`, immutable S0 profile identity, frozen
options, and fresh empty profiling roots.

## Execution

Run `scripts/execute_phase.py --task <dispatch-task.json>` first. It validates the
current BASE task and emits profile, metadata, derived-family, and lifecycle tools.

Read Stage B, Benefit-Driven Round Lifecycle, and Profiling Analysis Delegation in
[controller-legacy-contract.md](../superkernel-auto-tune/references/controller-legacy-contract.md).
Create capture-time manifests and fresh profiles, execute correctness, then dispatch
`sk-fusion-performance-analyst` with `superkernel-fusion-performance-analysis`. The
executor never classifies its own profile. Settle per-SK evidence, source identity,
calibration bridge, and SMAP. Derived BASE is fresh evidence, never prior reuse.

## Required Handoff

Return profile manifest paths/fingerprints, independent analysis result/Agent ID, SMAP
settlement, schema 2 ledger path, unresolved ranges, blockers, and Chinese guidance.
Do not edit source ranges or run optional experiments.

Run only when delegated by `superkernel-auto-tune` as `sk-base-profile-source-mapping`.
Read the controller's Stage B, SMAP, automatic-AOT, and profiling-analysis delegation
contracts. Create fresh immutable baseline and candidate profile roots, validate their
collection manifests, and dispatch `sk-fusion-performance-analyst` to use
`superkernel-fusion-performance-analysis`. The execution agent must never classify its
own profile.

Analyze every reliably mapped SK regardless of child count. When any performance-exact
SK lacks editable source identity, run one batched SMAP with the preserved named-scope
or marker-only AUTO protocol. Keep partial/skipped mappings as sealed non-actionable
facts. Return only when profile evidence and source identity settlement are complete, or
return the authoritative reprofile/blocker result to the controller.
