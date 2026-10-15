# 动作执行与主流程回接

## 已实现的执行门禁

使用 `scripts/multistream_execution.py` 物化动作和维护 trial 状态。不要手工生成
`single_change_verified=true`。

### Option 动作

```bash
python3 <skill-dir>/scripts/multistream_execution.py materialize-option \
  --input isolated/config/incumbent.yaml \
  --output isolated/config/MS-O1.yaml \
  --json-pointer /model_config/custom_params/super_kernel_optimize_options/auto_op_parallel \
  --before-json 0 --after-json 1 \
  --trial-id MS-O1 \
  --artifact-root . \
  --manifest-out isolated/experiments/MS-O1/action-manifest.json
```

工具要求输入不可变、输出不存在、声明的 before 与输入一致，并重新加载输出证明只有
一个结构化 JSON Pointer 发生变化。manifest 保存输入/输出文件指纹、结构化内容指纹和
唯一 changed pointer。

### Source 动作

`materialize-source` 只允许在 analyzer-validated 半开区间的 start/end byte offset 插入
`torch.npu.super_kernel_scope_begin/end`。它禁止删除、替换、业务语句和区间外插入，
并重新解析修改后的 Python 文件。

插入计划是 JSON 数组：

```json
[
  {"offset": 100, "text": "    torch.npu.super_kernel_scope_begin(None)\n"},
  {"offset": 180, "text": "    torch.npu.super_kernel_scope_end(None)\n"}
]
```

运行示例：

```bash
python3 <skill-dir>/scripts/multistream_execution.py materialize-source \
  --input isolated/source/model.py \
  --output isolated/source-trials/MS-X1/model.py \
  --source-file model.py --start-offset 100 --end-offset 180 \
  --boundary-change 'exclude one exact range from fusion' \
  --change-kind range_exclusion \
  --insertions isolated/experiments/MS-X1/insertions.json \
  --trial-id MS-X1 \
  --artifact-root . \
  --manifest-out isolated/experiments/MS-X1/action-manifest.json
```

支持的安全形态只有：named scope 的 `end -> begin` split、平衡的 `None` exclusion、
同名 outer scope 的 close/reopen。其他源码编辑必须 blocked，不能自行扩大 allowlist。

### 多流业务算子重排

`materialize-reorder` 只接受 `multistream_operator_order.py` 确定性生成的 route 2 或 route 3
顺序。它要求一个 source-exact range 内的 statement span 连续、互不重叠、显式 movable 且
side-effect-free；输出必须是这些完整 byte block 的纯 permutation。分析必须已经绑定稳定的
跨 stream `Cube <-> Vector` pair；`Mix`、同类计算引擎、通信/wait pair 即使无数据依赖也不得
物化：

route 3 还必须提供同一 range/analysis 的 route 2 passed `dispatch_order_evidence` 和
decision=reject 的 clean3 semantic evidence。执行器复核 trial/request/order/analysis 指纹；
缺任一证据均拒绝物化 route 3。

```bash
python3 <skill-dir>/scripts/multistream_execution.py materialize-reorder \
  --input isolated/source/model.py \
  --output isolated/source-trials/MS-R1/model.py \
  --operator-order-analysis isolated/experiments/operator-order-analysis.json \
  --range-id RANGE --after-order isolated/experiments/MS-R1/after-order.json \
  --trial-id MS-R1 --artifact-root . \
  --manifest-out isolated/experiments/MS-R1/action-manifest.json
```

route 3 额外传入 `--route2-dispatch-evidence` 与 `--route2-clean-evidence`；route 2 不传。

执行器逐块绑定 SHA256，验证输入源码指纹、hard dependency 拓扑、完整 Python AST 可解析和
源区间长度不变，并写入 `multistream_only_verified=true`。它不会自行推断依赖或接受手工
构造的顺序。只有至少三个 SK-off occurrence 的跨 stream overlap 与相同 SK-on operator 的
稳定 dispatch inversion 才能授权该动作；单流时必须 blocked。

### MIX component 跨函数重排

不要调用普通 `materialize-reorder`。使用：

```bash
python3 <skill-dir>/scripts/multistream_component_reorder.py materialize \
  --transform experiments/MS-C1/component-transform.json \
  --map experiments/component-source-map.json \
  --capture experiments/component-order-capture.json \
  --source-root isolated/source \
  --artifact-root . \
  --output isolated/source-trials/MS-C1/model.py
```

该执行器要求至少两个跨函数 hunk，每个 hunk 精确匹配受审 map 中的 AST
`transform_region`，并重放完整 MIX component capture、依赖覆盖、单流投影和安全证明。
输出使用 action-manifest-v2。profile 前使用同一脚本的 `verify-dispatch` 生成
`component_dispatch_evidence`；所有目标至少三个 occurrence 都达到预期顺序后 runner 才会继续。

component trial 的 `profile_collected` 语义校验可绑定完整 four-profile plan/summary，替代
传统两份 association manifest 的唯一来源。validator 必须确定性重放四个 role manifest，且
四份角色都属于同一个 trial/request；不得把单一 SK-on profile 冒充四象限闭环。其后的
`analysis_validated` 可绑定
`superkernel-multistream-component-performance-analysis-v1`，但必须同时证明至少三个
occurrence、所有 target 的逐层记录、完整/改善/劣化层数守恒，以及 parent、overlap、同引擎
contention 数值齐全。普通 schema 1.2 analysis 路径不受影响。

## Trial 状态

创建状态文件后逐门禁推进：

```text
planned -> materialized -> diff_verified -> correctness_passed
        -> profile_collected -> analysis_validated
        -> clean3_passed -> clean5_passed -> accepted
```

`rejected`、`blocked`、`failed` 是 terminal。除 `materialized` 外，每个推进事件都必须
引用相对 evidence path。`accepted` result 的 `execution_state` 必须能够从完整事件历史
重放到 accepted；不能跳过阶段或在 terminal 后追加事件。

每个 trial 使用从 immutable incumbent 新建的独立工作区。任意终态都必须按
[cleanup-and-rollback.md](cleanup-and-rollback.md) 冻结并执行 cleanup plan，不对共享源码执行
reverse patch。执行器把显式可变目标原子迁入可恢复 quarantine，并复核 incumbent 前后指纹；
artifact 和状态文件必须列入 `preserved_paths`。失败注入后的重跑必须从 partial move 幂等续行。

## 冻结计划与真实进程执行

动作通过 `diff_verified` 后，使用模型 adapter 编译五阶段 plan。adapter 只需为一个模型和
运行入口冻结一次，trial 不再手工拼接命令：

```bash
python3 <skill-dir>/scripts/multistream_plan_compiler.py freeze-adapter \
  --draft adapters/MODEL-draft.json \
  --output adapters/MODEL.json

python3 <skill-dir>/scripts/multistream_plan_compiler.py compile \
  --request experiments/multistream-request.json \
  --adapter adapters/MODEL.json \
  --action-manifest isolated/experiments/MS-O1/action-manifest.json \
  --plan-out isolated/experiments/MS-O1/execution-plan.json \
  --compilation-out isolated/experiments/MS-O1/plan-compilation.json
```

adapter 模板允许使用 `trial_id`、`request_id`、`parent_experiment_id`、
`request_fingerprint`、`candidate_config`、三个冻结 root 和显式 compile variables。
未知 placeholder、adapter 指纹变化、candidate config 越界或不存在均 fail closed。
result validator 会按原 request/action/variables 重新编译，协调修改 plan 和 compilation
fingerprint 也不能绕过 adapter。adapter 列出的 command/validator 程序文件也会逐文件绑定
SHA256；冻结后替换脚本会在执行前失败。

编译结果为
`superkernel-multistream-execution-plan-v1`：

```text
correctness_passed -> profile_collected -> analysis_validated
                   -> clean3_passed -> clean5_passed
```

reorder plan 仍复用五阶段 adapter，但计划顶层额外冻结 `pre_profile_evidence`。compile variable
`dispatch_order_evidence` 指定其相对路径。runner 在 correctness 后、profile 前调用
`validate_dispatch_evidence` 确定性重放；证据必须绑定 action/request/trial，证明目标下发序、
child set 不变、至少三个 occurrence 且每个 occurrence 至少两个 stream。缺失或不匹配时
profile 不会启动。

每个阶段必须声明 `argv`、`validator_argv`、相对 `cwd`、命令与 validator 超时、是否需要
设备、环境覆盖、required artifact 和 phase manifest 路径。计划顶层声明完整基础
`environment`、有序 `device_ids` 和 lease timeout。命令必须是参数数组，不能是 shell
字符串；环境不会从 runner 进程隐式继承。顶层还必须冻结绝对 `workspace_root`、
`artifact_root` 和 `lease_root`，CLI 运行值必须完全一致，防止同一计划换锁目录绕过互斥。
secret-like 环境变量禁止写入计划；凭据也不能出现在 argv。需要凭据的工作负载必须由模型
运行环境的安全注入机制处理，且不能进入归档。

```bash
python3 <skill-dir>/scripts/multistream_runner.py run \
  --plan isolated/experiments/MS-O1/execution-plan.json \
  --state isolated/experiments/MS-O1/execution-state.json \
  --workspace-root isolated/worktree \
  --artifact-root . \
  --lease-root /shared/superkernel-device-leases
```

runner 使用 `subprocess` 参数数组和独立进程组，超时时先终止、再强制结束整个进程组。
需要设备的 command 与 validator 在同一租约内完成；多个设备按升序逐卡获取
`npu-device-X.lock`，避免 `[0]` 与 `[0,1]` 的交叉占用。correctness、profile、clean3、
clean5 固定要求设备租约；只有离线 analysis 阶段不占设备，draft 不能改写该规则。

每个 passed phase manifest 固定记录计划指纹、实际 argv、cwd、完整显式环境、PID、起止
时间、返回码、timeout 状态、设备 ID、四份 stdout/stderr 以及 required artifact 的大小和
SHA256。compiler 强制 validator 使用 `scripts/multistream_evidence.py`，并校验阶段对应的
subcommand、request/trial identity、artifact root 和 `--out`。不能使用 `true`、任意脚本或
只检查文件存在来代替 correctness/profile/analysis/clean 语义校验。

标准 validator 的信息来源固定为：correctness 读取 launcher exit、correctness status 和完整
rank log；profile 调用 `artifact_contract.validate_manifest_set`；analysis 调用 schema 1.2
analysis validator；clean 调用正式 `analyze_performance.compare_candidates(...,
option_trial=True)`。输出只保留门禁指标、决策及源文件 SHA256，不携带 `samples_ms` 和原始
profiling 大文件。result-v2 会再次校验 evidence 指纹、身份、pass 决策和每个源文件 SHA256。

phase manifest 先原子落盘，trial state 后推进。因此崩溃发生在两者之间时，重跑会验证文件
指纹并仅补状态事件，不重复实验。若只有日志或输出而没有完整 manifest，fresh 性无法证明，
runner fail closed；必须保留该 attempt，使用新的 trial/attempt 路径重新执行。

runner 只运行到 `clean5_passed`。是否 `accepted` 仍由端到端增量收益、稳定性和 result
contract 决定，不能因局部 C/V overlap 改善自动晋级。

clean3/clean5 validator 必须把有效但无收益的比较用 adapter 声明的 reject exit code 返回。
runner 将其记录为 `rejected` 并立即停止；未知非零返回码仍是 `failed`，analysis 可使用显式
block exit code。这样三次无收益不会被误算为故障，也不会违规补跑到五次。

## Result 强证据

每个 trial 必须增加：

```text
action_manifest: relative/path/action-manifest.json
model_adapter: relative/path/model-adapter.json
plan_compilation: relative/path/plan-compilation.json
execution_plan: relative/path/execution-plan.json
execution_state: relative/path/execution-state.json
```

accepted trial 的五个 gate event 必须逐一引用 execution plan 声明的 phase manifest。result
validator 会重算计划指纹、命令/环境绑定，以及每个 sealed file 的 SHA256；仅创建同名 evidence
文件不能通过门禁。

accepted selected candidate 还必须增加：

```text
config_snapshot
source_manifest
```

option accepted 时，`config_manifest` 必须是该 trial 的 `action_manifest`，validator 会从
`config_snapshot` 重算结构化 fingerprint 并与 selected candidate 比较。Source trial
还需要完整 `superkernel-source-snapshot-manifest-v1`：列出所有冻结源码文件的相对路径和
文件 SHA256。validator 会逐文件重算组合 source fingerprint；单文件 action manifest
不能代替整个 source identity。

source snapshot 的 artifact root 是结果目录的边界，不是候选 worktree 的边界。若候选
worktree 位于 artifact root 外，adapter 必须在 action 后、执行前把审计列出的冻结源码快照
复制或链接到 artifact root 内的显式 snapshot root，并由 snapshot manifest 重新绑定。不得
放宽 `candidate_config` 的 root 检查，也不得让 result 直接引用可变的外部 worktree。

## 父流程回接

父流程使用：

```bash
python3 <adaptation-skill-dir>/scripts/bootstrap_derived_family.py \
  --request experiments/multistream-request.json \
  --result experiments/multistream-result.json \
  --current-incumbent experiments/current-incumbent.json \
  --registry experiments/derived-family-registry.json \
  --family-root experiments/derived
```

工具重新验证 request/result，比较当前 incumbent 与 request incumbent，在文件锁内幂等
登记 derived family，并原子创建 `SEED`。SEED 同时归档 adapter、compilation、plan、state
和五个 phase manifest；大体积 profiler/原始日志不复制，由 lineage 保留原 artifact root 与
文件指纹。输出状态固定为 `seed_registered`、
`fresh_base_required=true`、`ledger_merge_allowed=false`。

这一步不修改 schema 2 performance ledger。父流程必须从 SEED 启动 fresh ordinary BASE，
重新运行 correctness、profile-owned metadata 和独立 profiling analysis。新 family 完成普通
生命周期并生成标准 experiment result 后，才允许调用 `experiment_ledger.py merge`。

## 模型接入边界

上述通用入口已经实现：`multistream_adapter_generator.py`、父流程
`device_lease_runner.py`、`multistream_execution.py snapshot-source` 和
`derived_family_lifecycle.py`。具体模型仍需提供受审 model-run spec、short-trace capture
producer 和 fresh-BASE plan；通用 skill 不得猜测模型命令。

真实 NPU 作业通过 `multistream_npu_closure.py` 验收；失败 workspace 的自动回收策略仍是待
实现事项。所有并发父/子流程必须使用同一个绝对 `lease-root`。完整命令见
`automation-pipeline.md`。
