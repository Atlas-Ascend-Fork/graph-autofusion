# AOT Internals Reference

This file separates source facts from experiment-derived guidance. Source snapshot: local `graph-autofusion` revision `8d0b4fbc707624fbddf6e5f283e4447ba1f422d2` under `super_kernel/src/aot`.

Official references:

- TorchAir SuperKernel: <https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/advanced/superkernel.md>
- TorchAir cases: <https://gitcode.com/Ascend/torchair/blob/master/docs/zh/appendix/cases/superkernel_cases.md>
- Asc-devkit operator adaptation: <https://gitcode.com/cann/asc-devkit/tree/master/docs/zh/guide/编程指南/高级编程/SuperKernel>

Bundled offline summaries:

- [TorchAir behavior and diagnostics](torchair-superkernel-reference.md)
- [Ascend C operator adaptation](ascend-c-superkernel-adaptation.md)

## Source Facts

### Scope grouping

`sk_graph.cpp` assigns scope bit flags using stack/LIFO semantics. Begin nodes belong to their parent scopes, end nodes close the matching name, and regular nodes inherit active scopes. Operations outside user-marked ranges become `NOT_IN_SCOPE`. Reusing names is legal under stack semantics, but deterministic unique names make evidence easier to audit.

### Fusion and split evidence

`sk_optimizer.cpp` emits fused-node records to `sk_fused_nodes.log`. `sk_graph.cpp` emits unfusible records to `sk_fusion_fail_reasons.log`. `sk_scope_split.cpp` emits split decisions to `sk_scope_split.log`.

Primary fusion failure reasons in `sk_node.h` include:

- `OP_UNSUPPORT`, `DYNAMIC_TASK_UNSUPPORT`, `NOT_IN_SCOPE`, `IN_UNFUSIBLE_SCOPE`;
- `EXCEED_CORE_MAX`, `ISOLATED_EVENT`, `EXIST_DEADLOCK`, `EXTERNAL_DEPEND`;
- unsupported event/memory-only/default-node cases;
- `SIMT_OP_UNSUPPORT`, `KERNEL_ATTR_GET_FAILED`, and `EXCEED_SCOPE_MAX`.

Scope processing statuses in `sk_scope_info.h` are `SUCCESS`, `RESOURCE_INSUFFICIENT`, `NO_TARGET_NODE`, and `UNRECOVERABLE_FAIL`. Break reasons include `UNFUSIBLE_NODE`, `DEADLOCK_DETECTED`, `SYNCALL_OP_DROP`, and `DEBUG_PER_OP_MAX_CORE`.

`sk_fused_nodes.log` prints `SK Function`, `scope id`, `Node Count`, and one child
record per fused node. The generated function name usually contains the user scope and
`start_<first_child>_end_<last_child>`. Use `Node Count` and child records to judge
fusion depth; one-child groups prove fusion but rarely explain a real latency win.

Child `KernelInfos` records also expose revision-dependent scheduling and resource
evidence such as `numBlocks`, `cubeNum`, `vecNum`, `taskRatio`,
`needMixKernelSplit`, and `isScheModeOn`. Use `isScheModeOn` as the metadata indicator
for scheduling/control-core mode, but confirm its semantics against the active source
revision before changing runtime policy.

`sk_fusion_fail_reasons.log` prints the node id, stream id, function name, kernel type,
and human-readable reason. `sk_scope_split.log` prints `BreakInfo` with trigger node,
trigger stream, split reason, and fusion failure detail. These files should drive the
next candidate rather than blindly changing scope width.

### Option defaults in this source snapshot

`sk_options_manager.cpp` defines:

| Option | Default | Range |
|---|---:|---:|
| `auto_op_parallel` | 0 | 0-1 |
| `early_start` | disabled | disabled/enabled |
| `aggressive_opt_strategies.taskBreakerBypass` | 0 | 0-1 |
| `aggressive_opt_strategies.valueBreakerBypass` | none | bitmask |
| `aggressive_opt_strategies.eventBreakerBypass` | 0 | unsigned |

The source also contains DCCI, execution trace, cross-core check, aggressive strategy, and other controls. This does **not** prove the installed Python wrapper accepts those names.

### Profiling

`ASCEND_PROF_SK_ON` enables child-kernel event recording and creates `sk_meta/<pid>` output. The value is interpreted as per-core buffer size in KB in this source. The buffer is finite and non-wrapping: once `super_kernel.log` reports `buffer is full, stop dump the time of nodes`, later events on that core are absent and the capture must be reprofiled with a shorter diagnostic window or a sufficiently larger value. This instrumentation changes the run and must not be included in performance samples.

## Experiment-Derived Guidance

- Probe the active wrapper because local source, installed runtime, and online docs can be from different revisions.
- Use narrower scopes to turn resource, unsupported-op, and deadlock evidence into a localized boundary.
- Preserve symbolic decode inputs when their values change per token; static kernel compilation and Dynamo dynamic shape capture are separate controls.
- Judge TP performance by the slowest rank and repeated runs, not by one rank or one process.
- Establish baseline output determinism before using free-running text as a strict correctness oracle.
