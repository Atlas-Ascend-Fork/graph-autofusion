# Adapter Capability 与采集就绪 Preflight

本门禁用于所有可选多流动作，以及明确请求的 P/FINAL 源码范围分支。它必须发生在对应
分支的首个 NPU 命令、profiling recollection 或 trace attempt 之前。

## 两层检查

第一层是网络 capability matrix，回答该 adapter 是否已有密封证据支持某种动作。第二层是
run-specific collection plan，回答本轮是否已经绑定具体 producer、validator、materializer
以及必须生成的输出。两层都通过才允许启动设备实验。

Preflight 只证明“本轮具备采集足够证据的可重放配置”。它不证明运行一定成功，也不替代
运行后的 correctness、manifest、trace、analysis 或 clean validator。

## 阶段到动作的映射

阶段名不是 capability：

- 多流普通算子调序：`dependency_safe_operator_reorder`；
- 多流 event/stage：`event_edge_refinement`、`stage_split` 或实际 derivative action；
- MIX component 例外：`component_overlap_reorder`，并显式绑定
  `component_capture_provider`、`component_source_mapping_provider`、
  `component_reorder_materializer`、dependency provider、四 profile launcher 与
  post-component dispatch 输出；
- P：按实际修改选择 `range_exclusion` 或 `scope_split`；
- range-optimized FINAL：使用最终组合实际包含的全部 source action 并集；
- P/FINAL：禁止 `option` action，Option map 必须与 `Sbest-BASE` 完全一致，且
  `declared_option_changes=[]`。

如果动作尚未确定，只能运行不启动 NPU 的 discovery。不得用一个较弱 action 的 ready receipt
为较强动作授权；例如 `option=ready` 不能授权 reorder，也不能授权 P/FINAL。

## Collection Plan

先准备 `superkernel-adapter-collection-plan-v1` draft。`bindings` 必须列出本轮使用的实际文件，
`expected_outputs` 必须列出本轮 validator 将检查的全部产物。对于普通算子调序，至少包括：

```json
{
  "schema_version": "superkernel-adapter-collection-plan-v1",
  "plan_id": "S3-MS-R1-COLLECTION",
  "adapter_id": "deepseek-v4-a3-v1",
  "round_kind": "multistream",
  "source_revision": "REVISION",
  "source_fingerprint": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
  "config_fingerprint": "sha256:2222222222222222222222222222222222222222222222222222222222222222",
  "control_fingerprint": "sha256:3333333333333333333333333333333333333333333333333333333333333333",
  "workload_fingerprint": "sha256:4444444444444444444444444444444444444444444444444444444444444444",
  "requested_actions": ["dependency_safe_operator_reorder"],
  "experiment_artifacts": [
    {"role": "config_identity", "path": "intake/config-identity.json"},
    {"role": "control_identity", "path": "intake/control-identity.json"},
    {"role": "source_identity", "path": "intake/source-identity.json"},
    {"role": "workload_identity", "path": "intake/workload-identity.json"}
  ],
  "bindings": [
    {"role": "clean_validator", "path": "adapters/deepseek/clean_validator.py"},
    {"role": "correctness_validator", "path": "adapters/deepseek/correctness.py"},
    {"role": "dependency_provider", "path": "adapters/deepseek/dependencies.py"},
    {"role": "device_lease_runner", "path": "tools/device_lease_runner.py"},
    {"role": "four_profile_launcher", "path": "adapters/deepseek/four_profile.py"},
    {"role": "identity_provider", "path": "adapters/deepseek/identity.py"},
    {"role": "model_run_spec", "path": "adapters/deepseek/model-run-spec.json"},
    {"role": "raw_capture_plugin", "path": "adapters/deepseek/capture_plugin.py"},
    {"role": "reorder_materializer", "path": "adapters/deepseek/reorder.py"},
    {"role": "source_mapping_provider", "path": "adapters/deepseek/source_map.py"},
    {"role": "source_materializer", "path": "adapters/deepseek/source_edit.py"}
  ],
  "expected_outputs": [
    "baseline_profile_manifest",
    "bound_short_trace_analysis",
    "candidate_profile_manifest",
    "clean_performance_summary",
    "correctness_summary",
    "dependency_evidence",
    "four_profile_summary",
    "identity_registry",
    "operator_order_capture",
    "post_reorder_dispatch_evidence",
    "profile_owned_sk_meta",
    "profiling_analysis_result",
    "source_scope_map"
  ]
}
```

`freeze-collection-plan` 不接受孤立的自报 fingerprint。它会读取四个 experiment identity JSON，
验证其语义字段与 plan 一致，再密封 identity 和每个工具文件的 SHA256。`model_run_spec` 必须使用
`superkernel-multistream-model-run-spec-v1`。工具按 action 确定性检查必需 role 和 expected
output；调用方不得自行删减列表。任一实验身份、provider 或 launcher 变更后旧 plan 失效。

`round_kind` 只能是 `multistream/P/FINAL`。所有 P/FINAL 额外强制绑定
`profile_launcher` 和 `round_action_manifest`；manifest 必须绑定当前 round kind、source revision
和 action 集合。它们还必须声明 fresh `baseline_profile_manifest`、
`candidate_profile_manifest`、`profile_owned_sk_meta` 与 `profiling_analysis_result`，以保证后续
per-SK、child trace、源码映射和交互分析有足够输入。

## 执行门禁

依次执行：

1. `freeze-adapter`；
2. `build` 与 `validate` capability matrix；
3. `freeze-collection-plan`；
4. `preflight`；
5. 仅当命令返回 0 且 receipt `status=ready` 时获取设备租约。

使用 `validate-preflight` 可确定性重放已归档 receipt。任何可解析但不满足条件的 matrix、plan、
adapter、action、collection binding 或 expected output 都必须先写出 `blocked` receipt，CLI 返回
2。只有输入 JSON 自身无法解析时才作为 CLI 输入错误退出而不能生成语义回执。blocked 属于
pre-experiment evidence：不创建设备 trial，不计实验失败，不消耗 profiling/trace recollection
次数。

## 运行后闭环

Preflight 中声明的 expected output 必须由正式 phase validator 验证实际存在、身份一致、内容
完整且属于当前 process。仅创建同名文件不算完成。若运行后缺失，记录该 attempt 的执行失败
现场；不得回头把 preflight ready 改写成 blocked，也不得用其他 process 的 artifact 补位。
