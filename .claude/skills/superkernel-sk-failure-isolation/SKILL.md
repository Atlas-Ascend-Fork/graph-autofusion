---
name: superkernel-sk-failure-isolation
description: Use when a scoped Ascend SuperKernel candidate fails during execution, correctness checking, device execution, timeout, or hang diagnosis and fresh plog/sk_meta evidence is available
---

# SuperKernel SK Failure Isolation

Use this skill as a focused follow-up after a named SuperKernel scope has been
selected and the candidate fails at runtime or correctness. The goal is to
identify the failing child operator from runtime evidence and produce a
dependency-safe next scope that excludes it. Keep this workflow separate from
ordinary scope design, fusion-depth analysis, and performance tuning.

## Entry Gate

Before changing markers, confirm all of the following:

- The failed candidate, exact command, rank/device, process ID, and timestamp
  are known.
- The `plog` and fresh `sk_meta/<pid>` artifacts belong to that attempt.
- The failure is execution, correctness, timeout, hang, or device fault. A
  compile-only failure or a shallow/no-fusion result belongs to the parent
  adaptation/debugging flow.
- The active wrapper and source revision are known well enough to interpret the
  diagnostic fields and scope API.

If a fresh log or artifact is missing, stop and report an evidence gap. Never
infer a child operator from an old run or from a generated SK hash alone.

## Isolation Workflow

### 1. Freeze And Inventory Evidence

Copy or record the immutable paths before rerunning anything. Build an evidence
index containing:

| Item | Required value |
|---|---|
| Run identity | command, rank, device, PID, timestamp |
| Candidate | scope name, source marker location, options, model/config revision |
| Runtime log | `plog` path and fault line offsets |
| Metadata | `sk_meta/<pid>` path, `sk_fused_nodes.log`, `sk_node_detail.log`, `sk_scope_split.log` |
| Scheduling map | `sk_task_queue.json` when `opId` or `scopeId` is needed |
| Diagnostic mode | whether execution trace or cross-core checks changed the run |

Keep diagnostic runs separate from clean performance evidence. Debug options can
change fusion and timing.

### 2. Locate The SK Fault In plog

Search the fresh `plog` for the active release's fault markers. Start with these
known markers and add the exact strings printed by the local source:

```text
fault kernel_name
Exception is from superkernel function
PrintDfxInfo
sk_entry_aic
sk_entry_aiv
sk_entry_mix
opId
scopeId
COND
PC
```

Confirm that the reported entry is an SK entry. A generic device fault without
an SK entry is not enough to attribute the failure to a child operator.

### 3. Map The Fault To A Child Operator

Use evidence in this order:

1. An explicit child start/finish record from `debug_op_exec_trace` or the
   equivalent active diagnostic option.
2. A program-counter-to-symbol match from `PrintDfxInfo`, including the child
   function symbol, kernel type, and stream.
3. `opId` and `scopeId` correlation with `sk_task_queue.json` when symbols are
   repeated or the PC is not unique.
4. Cross-check of node IDs and ordered child symbols in `sk_fused_nodes.log`,
   node details, and the first actionable trigger in `sk_scope_split.log`.

Read the active SK source or [the parent reference](../superkernel-auto-tune/references/torchair-superkernel-reference.md)
to verify what each printed field means. Source-level error prints are the
authority for the release being debugged. Do not select the last logged child,
the nearest textual name, or a scope boundary just because it appears near the
fault.

### 4. Assign Confidence

Record `high`, `medium`, or `low` confidence:

- `high`: a child execution record or unique PC/symbol match agrees with the
  task queue or fused-node metadata;
- `medium`: `opId`/`scopeId` and metadata agree, but the child symbol or PC is
  repeated;
- `low`: only the SK entry, a broad error message, or incomplete logs identify
  the candidate.

For `low` confidence, do not exclude an arbitrary child. Reproduce using only
diagnostic options accepted by the active wrapper, preferably
`debug_op_exec_trace`; escalate to cross-core or per-op core checks only when
the failure suggests synchronization or resource behavior. If ambiguity
remains, shrink the scope until the failing child is isolated.

### 5. Generate The Next Scope

Return an explicit proposal containing:

- the exact child operator, symbol, node/op ID, stream, and scope ID to exclude;
- the original scope range and the new before/after ranges;
- the scope API or marker form supported by the active wrapper/source revision;
- dependency, event, barrier, cache, and stream-ordering constraints;
- the reason this child is excluded and the evidence confidence.

Exclude only the implicated child and any directly unsafe dependent fragment.
Preserve unrelated safe children and the original graph dependency order. Use the
runtime-supported unfusible/`None` marker mechanism when it is explicitly
available; otherwise split the named scope around the child. Do not invent an
API signature from a different CANN release.

If the child participates in a required event, wait/notify, full-core barrier,
collective, cache mutation, or cross-stream dependency, retain the edge. When a
single-child exclusion cannot represent that safety boundary, split before and
after the dependency boundary or abandon the candidate. Never bypass ordering
merely to recover fusion depth.

### 6. Revalidate

Treat the exclusion as a new candidate. Re-run the applicable gates in this
order:

1. preflight and option compatibility;
2. execution and correctness, including all ranks and the previously failing
   path;
3. fresh SK metadata proving the intended children fused and the excluded child
   did not re-enter the SK;
4. diagnostic profiler only when needed to confirm child ordering;
5. clean repeated performance runs only after the first four pass.

A compile success, disappearance of one log line, or a larger fused child count
does not prove that the exclusion is correct.

## Output Contract

Return this record in the final diagnosis and hand it back to the parent skill:
The record must satisfy the parent
[failure-scene-reporting contract](../superkernel-auto-tune/references/failure-scene-reporting.md)
and be appended to the failed experiment's own report. This skill's response or the
parent final summary is not a substitute for that experiment-local record.

```text
Failed attempt: rank/device/PID, command, timestamp, artifact paths
Failure phase: execution | correctness | timeout | hang | device fault
Fault evidence: file:line or field, with a short exact excerpt
Suspect child: operator/symbol, node or op ID, stream, scope ID
Confidence: high | medium | low; why
Exclusion proposal: exact child/range and marker or scope change
Safety notes: dependencies, events, barriers, streams, cache state
Verification: each gate result, artifact path, or not_run reason
Residual risk: unresolved ambiguity or follow-up needed
```

The parent skill must retain the failed candidate as negative evidence and use
the exclusion proposal as a new round, not overwrite the failed result. A successful
retry appends a new attempt; it must not remove the prior failure scene.

## Failure Routing

| Symptom | Action |
|---|---|
| No fresh `plog` or `sk_meta` | Stop; report missing evidence |
| Only `sk_entry_*` is known | Continue child correlation; do not claim attribution |
| Repeated symbols or ambiguous PC | Correlate `opId`/`scopeId` with task queue or narrow the scope |
| Fault disappears only under debug mode | Treat debug run as localization evidence, then retest clean |
| Exclusion violates an event/barrier/dependency edge | Split at the edge or abandon candidate |
| Child is identified but not safely separable | Preserve correctness; report candidate blocked |
| Candidate passes after exclusion | Start fresh metadata and clean gates; do not reuse stale evidence |

## Parent Skill Handoff

When invoked by `superkernel-auto-tune`, return control after the output record
is complete. The parent owns scope source edits, experiment numbering, baseline
comparison, and the Chinese final report. Do not duplicate those responsibilities
inside this skill.
