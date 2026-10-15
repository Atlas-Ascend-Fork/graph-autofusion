# SuperKernel Debugging

Diagnose in order: binding/configuration, fusion identity, execution/correctness, then
measured performance. Keep runtime failure isolation separate from performance pruning.

## Binding And Configuration

Run the environment probe before model loading. A key present in AOT source may still
be rejected by the installed wrapper. Use only exact values reported accepted; do not
patch validators or silently omit rejected options.

Verify:

- A2/A3 device, visible devices, `npugraph_ex`, and static kernel compilation;
- balanced direct scope markers;
- frozen source/config/control/workload fingerprints;
- optional controls absent unless requested and accepted;
- profiler, dump, child trace, and debug sync disabled for clean timing.

Before treating static compilation as failed or cleaning up compiler processes, apply
[static-compile-process-safety.md](static-compile-process-safety.md). In particular,
`*_compile_error.log` may be a provisional file created before `opc` starts. Its
existence, file count, or process age must never trigger a parent-side kill command.
Wait for natural terminal status and the final aggregate summary, unless the frozen
experiment runner's declared top-level timeout terminates its exact owned process group.

## No Fusion Or Untrusted Fusion

Run `scripts/analyze_sk_meta.py`. Missing `sk_fused_nodes.log` means fusion is not
proven. For present SKs, inspect model ID, source scope, boundary, ordered child
sequence, parsed/declared count, streams, `isScheModeOn`, and break reasons.

Child count, depth and fragmentation remain useful structural diagnosis:

- a single-child SK may expose launch overhead or an unexpected scope split;
- many small SKs may indicate resource/dependency boundaries;
- count histograms and per-layer outliers locate irregular regions.

These observations do not decide whether profiling runs and do not imply keep/prune.
Every complete reliable profile mapping proceeds to measured analysis.

compat/verify and generic replay are optional candidate-side reproducibility audits.
Their incomplete, duplicate, count-unreliable, or unmatched identities do not enter
profile-vs-baseline mapping errors. Only incomplete or ambiguous mapping in the two
profile roots yields `insufficient_evidence`; do not repair it from profiler name alone.

Interpret common reasons:

| Reason | Meaning / next step |
|---|---|
| `NOT_IN_SCOPE` | Verify intentional marker boundary |
| `IN_UNFUSIBLE_SCOPE` | Verify explicit exclusion and priority |
| `OP_UNSUPPORT` | Inspect node detail/custom kernel |
| `DYNAMIC_TASK_UNSUPPORT` | Exclude or adapt runtime-mutating operation |
| `SIMT_OP_UNSUPPORT` | Record unsupported SIMT path |
| `EXCEED_CORE_MAX` | Inspect core request |
| `EXCEED_SCOPE_MAX` | Reduce scope-name pressure |
| `RESOURCE_INSUFFICIENT` | Test one exact resource boundary |
| `DEADLOCK_DETECTED` | Preserve wait/notify order and isolate failure |

Deduplicate repeated scope-split snapshots before drawing structural conclusions.

## Execution Or Correctness Failure

For error, timeout, hang, device fault, or correctness failure, use
[superkernel-sk-failure-isolation](../../superkernel-sk-failure-isolation/SKILL.md).
Provide fresh plog, metadata, model/source logs, source revision, and exact candidate
config. Do not use stale logs.

For hangs:

1. confirm all ranks and phases;
2. locate the exact failed SK and child from direct log evidence;
3. inspect wait/notify, event, barrier, communication, cache and resource ordering;
4. map the child to an exact source range;
5. change scope only when the isolation result reaches required confidence.

Never delete operator execution or choose an arbitrary adjacent child. The repaired
round repeats process, correctness, profile-owned metadata, and profiling. Any replay
audit is repeated and reported separately.

For output differences, first measure S0 self-determinism. Use exact token parity only
when S0 repeats; otherwise prefer deterministic mode or teacher-forced logits/tokens
plus task-level checks. Record first divergence and baseline variation.

## Profiling Evidence Failure

The fresh read-only analysis Agent returns `insufficient_evidence` when:

- baseline/candidate profile is missing, stale, or fingerprint-mismatched;
- replay identity cannot map exactly and uniquely;
- occurrence count is too small;
- start/duration fields are invalid;
- source/config/control/workload/change manifests disagree;
- analysis Agent or result path is reused.

The only actions are `reprofile` or `block`; scope remains unchanged.

## Performance Regression

Use sibling
[superkernel-fusion-performance-analysis](../../superkernel-fusion-performance-analysis/SKILL.md)
through a fresh read-only Agent. The experiment child does not classify CSV rows.

Check:

- baseline interval P50/P90 versus SK P50/P90;
- duration sum P50 as secondary context;
- MAD dynamic threshold and occurrence count;
- mapping confidence and exact source range;
- all-rank TP behavior;
- C/V overlap and complete SK child trace;
- direct DCCI scalar/cache evidence and explicit runtime state;
- wait/event/control-core/resource evidence.

Classification controls scope:

- `beneficial/keep`;
- exact reliable `neutral/regressed -> prune`;
- `insufficient_evidence -> reprofile|block`, no scope change.

Prune only removes the range from SK markers. It preserves the original operator,
dependencies, ordering, events, communication, cache updates and stream semantics.

## P 裁剪诊断边界

只有用户或冻结实验计划明确请求可选 P/FINAL 源码范围分支时，才允许执行 P。BASE 或
SMAP 完成不自动创建该分支。`absent`、`invalid`、`no_gain`、`blocked` 或 `failed` 保留
incumbent，且不阻塞 unchanged `whole_scope_clean_validation`。P 必须使用 fresh candidate
profile、analysis Agent 与结果，并且 `declared_option_changes=[]`、冻结的 `Sbest-BASE`
option map 完全一致。P 只裁剪已有 direct source evidence 的 exact range；不会以 option、
恢复或边界细化替代该动作。

## Optional FINAL And Clean Timing

FINAL freshly profiles the combination of retained beneficial ranges. All final
ranges must remain exact reliable `beneficial/keep`; any regression or blocker
prevents clean timing.

Clean timing uses at least three independent FINAL candidate processes against the
frozen five-run S0. Disable every diagnostic control. Report interval P50, duration sum
P50, SK P50, MAD threshold, classification/action, mapping confidence, analysis Agent,
conditional evidence, and final end-to-end TP worst-rank metrics in Chinese.
