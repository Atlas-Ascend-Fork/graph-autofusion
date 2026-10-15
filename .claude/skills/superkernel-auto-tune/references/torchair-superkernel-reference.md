# TorchAir SuperKernel Reference

This is a distilled offline reference for `npugraph_ex` adaptation. It preserves the
constraints and diagnostic workflow needed by this skill; it is not a copy of the
upstream manuals.

## Provenance And Authority

- Upstream repository: <https://gitcode.com/Ascend/torchair.git>
- Source revision: `ca572a3bfcf6e0f9e92f522e4dcc748732b9d1e6`
- Snapshot checked: 2026-08-01
- Primary source: `docs/zh/npugraph_ex/advanced/superkernel.md`
- Diagnostic source: `docs/zh/appendix/cases/superkernel_cases.md`

This snapshot explains behavior and gives candidate option names. The installed
`torch_npu` active wrapper is the authority for whether an option is accepted in the
user's environment. Run `scripts/check_environment.py --json` before model loading.
Do not patch validators, silently drop rejected options, or assume a newer online
manual matches the installed CANN/torch_npu release.

Run that command only after loading the same approved CANN environment as inference,
normally the active installation's `bin/setenv.bash` in an isolated subprocess. The
probe must see CANN/HCCL library paths before importing `torch_npu`. A missing
`libhccl.so`, backend-extension load failure, `cann_environment_not_loaded`, or
`option_probe_status=not_run` means no option validation occurred; acceptance remains
unknown rather than rejected. Archive only a `ready=true` report as option-acceptance
evidence. Inspect repository setup scripts before sourcing them because they may also
delete caches or perform unrelated mutations.

For evidence-derived DCCI regex or revision-specific values, pass a JSON value matrix
to `check_environment.py --option-probes <file> --json`. A key accepting one sample
does not validate a different regex; only exact returned `accepted_values` are usable.

## Contents

1. Runtime constraints
2. Scope semantics
3. Compile options
4. Fusion and failure evidence
5. Execution diagnosis
6. Performance diagnosis

## Runtime Constraints

- The documented products are Atlas A2 and Atlas A3 training/inference products.
- SuperKernel fuses consecutive compiled Tasks into one binary-scheduled Task. An
  unfusible operation ends the current fused segment; fusion may resume after it.
- Documented communication operations include `AllReduce`, `ReduceScatter`,
  `AllGather`, and `AlltoAll`.
- `static_kernel_compile` must be enabled with `super_kernel_optimize`.
- Data Dump is unavailable while SuperKernel optimization is enabled.
- In one process, same-shape operations from different models may reuse static
  artifacts. Keep SuperKernel options aligned across those models, or enable static
  compilation only where needed.

The official example uses `dynamic=False`, but Dynamo dynamic-shape capture and
static kernel compilation are separate controls. Preserve the target model's required
symbolic inputs; do not force `dynamic=False` merely to enable static kernels.

## Scope Semantics

Mark a graph-construction region with balanced direct calls:

```python
torch.npu.super_kernel_scope_begin("decoder.layer.0")
y = layer(x)
torch.npu.super_kernel_scope_end("decoder.layer.0")
```

This direct pair is mandatory in model source. Do not replace either call with
`torchair` scope helpers, repository wrappers, Python context managers, decorators, or
another abstraction. A wrapper may make source code look balanced while failing to
emit the marker pair observed by the captured `npugraph_ex` graph.

- A scope name is at most 255 bytes.
- At most 1024 distinct scope names are supported globally.
- Begin/end names must match. Marker presence does not prove that fusion occurred.
- Overlapping or nested scopes group operations by their complete scope-marker set;
  an inner region can therefore become a separate group.
- `None` marks an unfusible region and has highest priority.
- For direct `npugraph_ex` marker APIs, express an explicit exclusion as a balanced
  `torch.npu.super_kernel_scope_begin(None)` / `scope_end(None)` pair. This is the
  Python null value, not the string `"None"`; the string creates an ordinary named
  scope and must never be used as an exclusion sentinel.
- If named fusion scopes exist, unmarked operations are outside those scopes. If the
  graph contains only `None` scopes, unmarked operations still enter automatic fusion.

Use deterministic names and begin with decode-only scopes. Before selecting explicit
exclusion, require the environment report to contain
`scope_capabilities.explicit_none_exclusion.accepted=true`. That eager pair probe only
proves the active wrapper accepts the optional argument; confirm actual exclusion and
surrounding boundaries from fresh compilation metadata rather than inferring them from
the API signature or source indentation.

## Compile Options

Minimum candidate configuration:

```python
options = {
    "static_kernel_compile": True,
    "super_kernel_optimize": True,
}
compiled = torch.compile(model, backend="npugraph_ex", options=options)
```

`super_kernel_optimize` defaults to false. Add
`super_kernel_optimize_options` or `super_kernel_debug_options` only when non-empty
and accepted by the active wrapper.

Documented optimization controls in this snapshot:

| Option | Intended use | Guardrail |
|---|---|---|
| `dcci_before_kernel_start` | Entire DCache refresh before matched child | In Stage O, do not test independently; use the evidence-derived child list together with `dcci_after_kernel_end` |
| `dcci_after_kernel_end` | Entire DCache refresh after matched child | In Stage O, use the same exact regex list as before and validate correctness |
| `dcci_disable_on_kernel` | Keep per-access refresh but omit child-boundary refresh | Stage O DCCI entry point is exactly `[".*"]`; accept directly on gain or diagnose a regression before combined repair |
| `auto_op_parallel` | Rearrange multi-stream Cube/Vector work without changing dependencies | Release-dependent; prove the resulting schedule |
| `early_start` | Honor adapted `SetNextTaskStart`/`WaitPreTaskEnd` calls | Requires operator-side adaptation |
| `aggressive_opt_strategies` | Bypass selected task/value fusion breakers | Treat as high risk; use only with targeted evidence |

`aggressive_opt_strategies` is a dictionary in the documented TorchAir snapshot. Use
`task_breaker_bypass=1` only for matching non-AICore task breaker evidence. Use
`value_breaker_bypass` as a bitmask: bit0 allows paired ValueWait fusion after deadlock
checks, bit1 allows unpaired ValueWait fusion. Values `1`, `2`, and `3` should be
separate experiments. The local AOT source also contains `event_breaker_bypass`; treat
it as source-revision-specific and require active-wrapper acceptance plus owner-approved
event semantics.

Documented debug controls:

| Option | Diagnostic effect |
|---|---|
| `debug_sync_all` | Inserts `SyncAll` between children to isolate synchronization faults |
| `debug_op_exec_trace` | Records per-child start/finish state and prints it on faults |
| `debug_cross_core_sync_check` | Checks Cube/Vector synchronization counts and implies execution trace |
| `debug_per_op_max_core_num` | Runs each child as a maximum-core SuperKernel and implies cross-core checking |

Every debug option can alter fusion or runtime performance. Never include a debug run
in benchmark samples.

## Fusion And Failure Evidence

Diagnose in this order: fusion existence, execution/correctness, then performance.

1. Collect SK-off and SK-on framework profiling. Check candidate `kernel_details.csv`
   for `Type=SuperKernel`. The `Name` commonly contains `sk_` and start/end boundary
   information, but generated hashes make exact full-name matching brittle.
2. For compile metadata, set `ASCEND_OP_COMPILE_SAVE_KERNEL_META=1` before the
   compatibility run. Analyze the fresh `sk_meta/<pid>` tree.
3. Inspect `sk_fusion_fail_reasons.log`, `sk_scope_split.log`, and
   `sk_fused_nodes.log`. A switch or generated binary without fused-node records is
   not fusion proof.

Useful interpretations:

| Evidence | Next action |
|---|---|
| `OP_UNSUPPORT` | Identify the child symbol; replace it, move it outside the scope, or adapt the custom operator |
| `DYNAMIC_TASK_UNSUPPORT` | Exclude or adapt the task-info refresh path |
| `UNFUSIBLE_NODE` | Inspect the first boundary and scope placement |
| `DEADLOCK_DETECTED` | Restore required stream ordering or split at the dependency boundary |
| `RESOURCE_INSUFFICIENT` | Reduce multi-stream/scope resource pressure |

For required multi-stream order, preserve or insert explicit event record/wait edges,
including through an FX pass when that is how the target graph expresses ordering.

## Execution Diagnosis

For device errors or timeouts:

1. Search plog for `fault kernel_name` and
   `Exception is from superkernel function`.
2. Confirm the entry resembles `sk_entry_aic`, `sk_entry_aiv`, or `sk_entry_mix*`.
3. Use `PrintDfxInfo` child nodes and original function symbols to locate the failing
   operation; match the reported PC to a child symbol when available.
4. If repeated child symbols make PC matching ambiguous, correlate the COND register
   state/opId/scopeId with `sk_task_queue.json`.
5. Reproduce with `debug_op_exec_trace`, then escalate to cross-core checks or
   `debug_per_op_max_core_num`. These are diagnosis-only runs.

When a custom Ascend C child is implicated, read
[ascend-c-superkernel-adaptation.md](ascend-c-superkernel-adaptation.md).

## Performance Diagnosis

- Compare the fused Device interval with the original interval: latest end minus
  earliest start. Do not sum child durations in a multi-stream region.
- Set `ASCEND_PROF_SK_ON` only for diagnosis and treat its positive value as per-core
  buffer KB, not a boolean. Choose the smallest capacity that covers the bounded
  diagnostic window. It produces
  `sk_prof_device_<deviceId>.json`, which can be viewed in Chrome tracing to inspect
  child ordering and duration. Reject and reprofile when `super_kernel.log` contains
  `buffer is full`; do not analyze the resulting incomplete lane set.
- In single-stream regressions, inspect scalar GM access and DCCI overhead before
  changing DCCI policy. Keep only settings that pass correctness regression.
- In multi-stream regressions, compare the original overlap with the fused schedule.
  Cube/Vector resource complement is favorable; MIX/Cube or same-resource overlap can
  serialize. Try an accepted `auto_op_parallel` setting, reorder only when dependency
  semantics remain unchanged, or narrow the scope.
- Final acceptance still requires clean repeated end-to-end runs on every TP rank.
