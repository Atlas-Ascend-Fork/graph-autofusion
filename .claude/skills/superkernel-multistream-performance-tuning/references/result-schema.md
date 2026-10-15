# Request/Result Schema 摘要

确定性字段和交叉校验由 `scripts/multistream_contract.py` 实现。这里说明消费语义。

## Request

`superkernel-multistream-request-v1` 包含：

- request/parent identity；
- immutable incumbent identity 和五类 fingerprint；
- clean/profile/manifest/projection/environment/option-history artifact；
- 一个或多个 target range 与 graph occurrence；
- dedicated worktree/config/experiment/cache roots；
- exact execution/correctness contract；
- trial 和短 trace 重采预算。

request 声明任何 `scope_split` 或 `range_exclusion` 时必须同时携带 source map；纯
option request 可以缺省。validator 会把 target 绑定到 sibling schema 1.2 analysis 的
内容 fingerprint、唯一 exact decision 和双方至少三个 occurrence。

所有 artifact path 相对 request 文件目录。隔离输出路径可以尚不存在，但必须是无 `..` 的
相对路径。

当 `requested_change_kinds` 包含 `dependency_safe_operator_reorder` 时，request 必须同时提供
`operator_order_capture` 和 `source_scope_map_v2 + exact`。所有 target 必须为直接证据支持的
`parallelism_effect=degraded + optimization_status=opportunity`；合同会重算 capture，并拒绝
单流、stream identity 不完整、缺 stable dispatch inversion 或缺 statement/dependency 证明的
请求。

## Result

`superkernel-multistream-result-v2` 绑定 request content fingerprint，并始终包含精确
fallback identity。

v2 新增 action manifest、frozen model adapter、plan compilation、execution plan、
execution state、phase manifest、config snapshot 和 source manifest 强门禁。
旧 v1 result 不能进入自动回接；父流程按 malformed result 保留 incumbent。

`accepted` 要求：

- exactly one accepted trial；
- `incumbent_unchanged=false`；
- selected candidate 绑定该 trial；
- correctness/profile/clean 全部 passed；
- exactly five clean runs；
- stable、incremental mean gain、positive median direction、P90/stddev non-regression；
- selected source 或 config fingerprint 至少一个不同于 incumbent。
- 每个 trial 的 `action_manifest` 能证明唯一真实修改；
- 每个 trial 的 `model_adapter` 与 `plan_compilation` 能按原 request/action/variables
  确定性重编译出同一 execution plan；
- 每个 trial 的 `execution_plan` 内容指纹、绝对运行根目录和 request/trial identity 可重算；
- 每个 trial 的 `execution_state` 能按顺序重放到其 terminal decision；
- accepted trial 的五个执行状态逐一引用计划声明的 passed phase manifest，且所有 command、
  validator、环境和 sealed artifact SHA256 可重放；
- 五个标准 semantic evidence 的 schema、阶段、trial/request identity、pass 决策、内容指纹和
  全部 source file SHA256 可重放；clean evidence 不包含原始 timing sample 数组；
- selected candidate 提供 `config_snapshot` 和 `source_manifest`；
- option trial 的 config fingerprint 从 snapshot 重新计算，并与 action manifest 一致。
- source manifest 的每个冻结源码文件 SHA256 和组合 source fingerprint 均可重算。
- reorder trial 的 action 必须绑定 analyzer 生成的 route 2/3 permutation、statement block
  SHA256、hard dependency 和 `multistream_only_verified=true`；不得接受手工顺序。
- reorder trial 进入 profiling 前必须绑定 passed `dispatch_order_evidence`，证明至少三个
  post-reorder occurrence 的目标下发序已生效、child set 不变且每次仍为至少双 stream。
- route 3 action 还必须绑定同 range/analysis 的 route 2 passed dispatch evidence 和 rejected
  clean3 evidence；clean3 必须证明基线、P90 和方差门禁通过，均值/中位不回归，只是增益未达阈值。

若声称 `mechanism_status=validated`，trial 还必须携带无 overflow、至少三个 aligned
occurrence 的短 trace 与 parent-SK binding artifact；没有这组证据只能写 `unproven`、
`not_observed` 或 `blocked`，不能声称内部并行机制已验证。
validator 会加载 `superkernel-multistream-trace-analysis-v2`，复核 request/trial/range、
capture/source SHA256、无 blocker、occurrence count 和 `parallelism_effect=improved`；对应
finding 必须写 `optimization_status=validated`。

“已完成多流机会筛查”还必须引用 `superkernel-multistream-opportunity-coverage-v1` 且
`complete=true`。若 coverage 缺少任一 beneficial target，只能报告 incomplete/blocked，不能
把候选数 0 解释为没有隐性机会。`beneficial + degraded` 始终保留 incumbent；只有后续单变量
trial 通过 clean E2E 增量门禁才替换它。

`no_gain/blocked/failed` 要求：

- zero accepted trial；
- `incumbent_unchanged=true`；
- `selected_candidate=null`；
- fallback candidate/source/config/control/workload 与 request incumbent 完全一致；
- 中文 fallback reason；
- blocked/failed 至少一个 blocker。

父流程必须把 result 与当前 incumbent 再次比较。即使 result 自身合法，只要上层 incumbent
已经变化，也必须按 stale result 拒绝并保持当前 incumbent。

父流程先把 accepted result 注册为 `seed_registered` derived family，不能直接写 schema 2
performance ledger。fresh ordinary BASE 和普通生命周期完成后，才合并标准 experiment result。
`derived_family_lifecycle.py` 对应状态为 `fresh_base_completed`、
`ordinary_lifecycle_completed`、`ledger_merged`；只有倒数第二个状态临时允许 merge。
