# 模型 Adapter Schema

`superkernel-multistream-model-adapter-v1` 是模型运行入口与通用 runner 之间的受审边界。
它不是单个 trial 的计划；同一模型、同一运行工具和同一环境可复用一个 frozen adapter。

## 顶层字段

```json
{
  "schema_version": "superkernel-multistream-model-adapter-v1",
  "adapter_id": "model-a3-v1",
  "workspace_root": "/abs/isolated/worktree",
  "artifact_root": "/abs/experiment",
  "lease_root": "/abs/shared/device-leases",
  "environment": {"PATH": "/fixed/path", "PYTHONPATH": "/fixed/python/path"},
  "device_ids": [0, 1],
  "lease_timeout_seconds": 300,
  "phases": [],
  "adapter_fingerprint": "sha256:..."
}
```

`adapter_fingerprint` 只能由 `freeze-adapter` 生成。三个 root 都是绝对路径；环境完全显式，
不会继承 compiler 或 runner 的 ambient environment。设备 ID 必须升序且不重复。

首个 adapter 不需要手工展开 validator。准备
`superkernel-multistream-model-run-spec-v1` 后运行
`scripts/multistream_adapter_generator.py --spec SPEC --out ADAPTER`；generator 自动生成本页
五阶段结构，再调用同一个 `freeze_adapter` 门禁。模型运行命令、输出路径和 validator input
仍必须由模型维护者审查。

## Phase 模板

必须按以下顺序恰好声明五个 phase：

```text
correctness_passed, profile_collected, analysis_validated,
clean3_passed, clean5_passed
```

单个 phase 形状：

```json
{
  "state_after": "clean3_passed",
  "phase_id_template": "{trial_id}-clean3",
  "argv_template": [
    "/abs/tools/run_clean_series.py",
    "--config", "{candidate_config}",
    "--output", "{artifact_root}/trials/{trial_id}/clean",
    "--runs", "3"
  ],
  "validator_argv_template": [
    "{python_executable}",
    "{skill_root}/scripts/multistream_evidence.py",
    "clean",
    "--artifact-root", "{artifact_root}",
    "--state-after", "clean3_passed",
    "--trial-id", "{trial_id}",
    "--request-fingerprint", "{request_fingerprint}",
    "--out", "{artifact_root}/trials/{trial_id}/clean3-summary.json",
    "--baseline-root", "{baseline_clean}",
    "--candidate-root", "{artifact_root}/trials/{trial_id}/clean",
    "--candidate-name", "{trial_id}",
    "--expected-ranks", "8",
    "--warmup", "8",
    "--expected-runs", "3"
  ],
  "cwd_template": "model",
  "timeout_seconds": 1800,
  "validator_timeout_seconds": 120,
  "environment_overrides": {},
  "validator_exit_actions": {"0": "pass", "10": "reject"},
  "program_file_templates": [
    "/abs/tools/run_clean_series.py"
  ],
  "required_artifact_templates": [
    "trials/{trial_id}/clean3-summary.json"
  ],
  "manifest_template": "trials/{trial_id}/phases/clean3.json"
}
```

内置 placeholder：

```text
trial_id, request_id, parent_experiment_id, request_fingerprint,
candidate_config, workspace_root, artifact_root, lease_root,
python_executable, skill_root
```

模型额外值通过 compile variables JSON 提供。例如 `baseline_clean`、
`baseline_profile_manifest`、`analysis_agent_id`。额外变量不能覆盖内置值。

`dependency_safe_operator_reorder` 计划还必须提供
`dispatch_order_evidence` compile variable，值是 artifact root 下的安全相对路径。该文件在
compile 时可以尚不存在，但 correctness 完成后必须由
`multistream_operator_order.py verify-dispatch` 生成。compiler 把它冻结为计划顶层
`pre_profile_evidence`；runner 验证通过前不会启动 profile phase。非 reorder action 禁止携带
该变量。

## Validator 规则

- exit code `0` 必须映射为 `pass`，其他 code 不能映射为 pass；
- correctness/profile 的未知非零 code 为 `failed`；
- analysis 可以声明一个明确 code 为 `block`；
- clean3/clean5 必须至少声明一个 `reject` code；
- clean3 返回 reject 后不得执行 clean5；
- validator 必须调用 `{skill_root}/scripts/multistream_evidence.py`，subcommand 与 phase 固定对应；
- correctness/profile/analysis/clean 的必需参数、artifact root、state、trial/request identity、
  `--out` 及 clean3/clean5 run count 均由 compiler 校验；
- `--out` 必须位于 artifact root，并同时列入 `required_artifact_templates`；
- command executable 必须使用绝对路径并列入 `program_file_templates`；compiler 自动加入并
  绑定 Python executable 和标准 validator 文件 SHA256；
- clean validator 调用正式 `analyze_performance.py` option-trial API，只输出紧凑摘要，
  不复制包含全部 timing samples 的多 MB JSON；
- result-v2 会重放 semantic evidence 的身份、决策、内容指纹和所有 source file SHA256。
- profile validator 可声明 `--trace-analysis`；此时 semantic evidence 会绑定已校验的
  short-trace analysis、capture 和 source file identity。

adapter 可复用模型仓已有的 clean/profile launcher 和 artifact validator，但路径、参数、
环境和输出目录都必须显式模板化。模型专用解析逻辑放入 capture producer 插件，不得加入
通用 analyzer、planner 或 runner。

## 编译门禁

compiler 会验证 request、action manifest、candidate config 和 adapter 指纹，渲染模板后再
调用 runner 的严格 plan validator。`plan-compilation.json` 绑定：

```text
request fingerprint + action fingerprint + adapter fingerprint
+ candidate config + compile variables + plan fingerprint
```

result 校验时会重新执行同一编译过程。只同步修改 execution plan 和 plan fingerprint，或连同
compilation fingerprint 一起伪造，都无法通过 deterministic replay。
