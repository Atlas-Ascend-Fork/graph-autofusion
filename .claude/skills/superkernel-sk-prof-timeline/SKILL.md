---
name: superkernel-sk-prof-timeline
description: >-
  Compact an Ascend SuperKernel sk_prof_device_*.json Chrome Trace by collapsing
  per-core AIC and AIV events into one Cube lane and one Vector lane per operator
  occurrence. Use for focused Cube/Vector scheduling inspection, not for SK-off/SK-on
  performance comparison or source-order decisions.
---

# SuperKernel SK Prof Timeline

## Purpose

Use this skill when a raw `sk_prof_device_*.json` Chrome Trace is too dense to inspect
because every AIC/AIV core is rendered separately. The transformer preserves each
operator occurrence's envelope:

- `Vector`: the earliest AIV start through the latest AIV end across its vector cores.
- `Cube`: the earliest AIC start through the latest AIC end across its cube cores.

The output is a top-level Chrome Trace event array, matching raw `sk_prof`'s wire
format, with three horizontal lanes explicitly sorted `SK E2E`, `Cube`, then `Vector`.
`SK E2E` is the
original parent SK span merged across its AIC/AIV cores, so the child component events
contained by that bar identify the fused work. This is a scheduling visualization, not
a sum-of-core-time or a critical-path calculation. An envelope can contain idle gaps
between cores.

## Run

Use the streaming transformer; it does not load the complete input JSON into memory.
It writes a Chrome Trace plus an adjacent summary JSON.

```bash
python3 scripts/compact_sk_prof_trace.py \
  --input /path/to/sk_prof_device_0.json \
  --output /path/to/sk_prof_device_0.compact.json
```

Open the generated `*.compact.json` with `chrome://tracing` or Perfetto's trace viewer.
Each event name contains the readable operator name and `sk`, `node`, and occurrence
indices. Its `args` retain the original normalized kernel name and the number of raw
core events merged into the envelope.

## Association Rules

- The default output preserves original whole-SK events without `args.nodeId` on the
  `SK E2E` lane. A parent span is merged across both component types and all cores;
  it is not duplicated on the Cube or Vector lanes. Use
  `--exclude-superkernel-spans` when only child components are wanted.
- A logical occurrence is identified by component type, `modelId`, `skId`, `nodeId`,
  and normalized raw kernel name. Core events with the same identity are split when
  adjacent starts are more than `--occurrence-gap-us` apart (default: `100`). Tune
  that threshold only after inspecting the launch cadence in the source trace.
- The transformer supports runtime labels `AIC` and `AIV` by default. Override them
  with `--cube-pid` and `--vector-pid` for a trace using different labels.
- Compare timestamps only within this one trace. Do not align its device clock with
  `kernel_details.csv` host timestamps by subtraction.

## Boundaries

Use `superkernel-fusion-performance-analysis` for SK-off/SK-on mapping, performance
classification, and evidence-gated conclusions. Use
`superkernel-multistream-performance-tuning` only after its required capability and
source-dependency checks. This skill only creates the compact visualization and its
aggregation receipt; it does not run inference, change scopes, or justify a source
reorder on its own.
