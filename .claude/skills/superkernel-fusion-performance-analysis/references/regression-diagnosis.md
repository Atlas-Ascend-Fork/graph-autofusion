# 性能劣化诊断

## 证据等级

先区分可观测事实、受证据支持的假设和已由单变量 A/B 复现的结论。profiling interval、
stream overlap、wait/scalar 指标和 fresh metadata 是可观测事实；配置 manifest、环境探测
或 active wrapper/source 才能证明选项状态；只有 Stage O 的单选项实验或 post-BASE 的
单范围 source-only 实验稳定改变目标信号，才能形成对应的有效修复或根因证据。

- `high`：仅用于直接 config 或可可靠归属目标 range 的 trace 证据，例如融合前
  Cube/Vector overlap，且融合后同一 range 的 child trace 同时含 CUBE、VECTOR 完整事件
  并明确显示两者不并行。
- `medium`：仅用于 profiler counter/统计相关证据，例如 scalar ratio，或 SK interval
  劣化但 SK duration 仍优于 baseline duration sum 所提示的并行重叠丢失。
- `low`：仅表示缺少 artifact 的待验证假设，必须同时列出 blocker；不能据此生成自动
  scope action 或 option experiment。

每条 `diagnostic_hypotheses` 必须写 `kind`、`confidence`、单个 `range_id`、`evidence`、
`explanation_zh` 和 `requires_ab_test=true`。不得写 `root_cause`；只有后续单变量 A/B
稳定复现后，主实验流程才能另行形成根因证据。

证据缺失或互相冲突时，保留假设并列出 blocker。不得为了生成推荐而把未知状态写成
已开启、已关闭或已证明。

## Cube/Vector 串行化

比较 SK-off child 的 stream、时间区间、`baseline_duration_sum`、
`baseline_union` 与 `baseline_interval`。若基线存在 Cube/Vector overlap，而 SK-on child
trace 显示同一范围在 SK 内串行，才形成 Cube/Vector 串行化假设。

child trace 只有在所有有效事件明示同一个目标 `range_id`，或本轮只有一个已裁剪且可执行
的 range 时才能归属。未提供 trace、空 trace、缺少任一 CUBE/VECTOR 完整事件，或多 range
下 trace 不可归属时，为每个受影响 range 输出 `cube_vector_child_trace_missing` low 假设和
中文 blocker；不得生成 `auto_op_parallel` 实验，也不得把同一 trace 复用到所有 range。
摘要中的 `stream_count` 不计未知 ID，并输出 `known_stream_event_count` 与
`stream_identity_complete`。用于 direct evidence 的每个 CUBE/VECTOR 完整事件都必须有
可靠 `stream_id` 或等价字段；缺任一 identity 时同样降为 low 并阻止自动实验。

仅当 environment evidence 精确接受整数 `auto_op_parallel=1` 时，建议对目标
`range_id` 单独测试。未接受、只接受 `0` 或接受布尔值 `true` 均不得生成实验。
duration sum 大于 interval 只能诊断重叠；即使 SK duration 优于 duration sum，性能
classification 和 action 仍只由 baseline interval 决定。

## DCCI 状态与 Scalar/Cache 证据

DCCI 是否开启必须来自 candidate config manifest、环境选项探测或 active
wrapper/source 证据。scalar/cache 指标与劣化同时出现时只能支持 DCCI 假设，不能反向
推断配置状态。

实现收集 environment 与 candidate config 中所有显式的 `runtime_evidence.dcci_state`，并在
输出 evidence 中记录全部来源。`unknown` 或缺失不覆盖已知状态；所有已知来源一致时采用该
状态。仅当已裁剪 range 的 scalar/cache counter 已进入 DCCI 假设/实验门槛时，`enabled`
与 `disabled` 冲突才输出 `dcci_state_conflict` blocker，全为 unknown/缺失才输出
`dcci_state_unknown` blocker；未进入该 counter gate 时无需无条件生成 DCCI blocker。
这些状态不阻止 interval classification 或 exact scope pruning。

Stage O DCCI 不从 scalar/cache 假设开始，也不独立测试 before/after。唯一首轮是
`dcci_disable_on_kernel=[".*"]`，并且 accepted values 中必须精确存在该值；
窄 regex、带空格的近似值或其他全局表达式都不能替代。若 clean 有收益，执行 owner
直接采纳 disable-all；若 neutral/no-gain，直接拒绝 DCCI family。

只有 disable-all clean 出现可重复劣化时，才进入 paired no-option versus disable-all
诊断。两侧必须有 fresh `kernel_details.csv`、同进程完整 SK metadata 和完整
`sk_prof_<device>.json`。先结构唯一匹配全部 SK，以 interval/wall P50、P90、
MAD 分类全部 SK；再对每个 significantly slower SK 按 ordered child position 比较
wall span、max-lane、P90、MAD，duration sum 只作辅助。对全部显著劣化 SK 的显著劣化
child canonical op 取稳定去重并集，不能只分析或修复最慢的一个 SK。

每个联合修复 regex 必须可编译、非全局、实际匹配 metadata child 完整 symbol，并包含
完整 child function 或 canonical op token 的可验证字面片段。`.*`、`^.*$`、
`.+`、`^.+$` 等无 symbol/op literal 的全局或近全局表达式不可用；候选不得
匹配空串、`CompletelyUnrelated` 或 `static_kernel_Unrelated_hash`
负控制。最终 exact regex list 必须由 active environment probe 同时接受为
`dcci_before_kernel_start` 和 `dcci_after_kernel_end` 的值。

Stage O 只生成一个联合修复建议：保留 disable-all，并把完全相同的 child regex list
同时设置到 before 与 after。不要生成 before-only、after-only、逐 child 或排列组合
实验。该组合由执行 owner 直接与五轮无 DCCI O0 做 clean A/B；只有优于 O0 并通过完整
门禁才采纳。只比 disable-all 快不能采纳。无收益时报告逐 SK/child 分析、exact 设置并
拒绝整个 DCCI family。

## Wait、同步与控制核

结合 wait/sync 指标、event、barrier、stream wait、metadata 和控制核证据检查调度开销。
只有配置、source 或运行证据能确认 `isScheModeOn` 状态；不得从 SK 名称、单次 duration
或某个 wait 指标猜测控制核模式。

诊断必须保留算子顺序、依赖、event、barrier、cache 更新、通信和 Cube/Vector 多流关系。
若异常与资源分配、共享 stream/cache 或控制核交互相关，记录可验证信号和 blocker，
不要建议绕过同步来换取表面收益。

## P 单范围源码 A/B 实验

Stage O 已经结算并冻结全部 Option。P/FINAL 不得再生成 Option experiment。
post-BASE 分析固定输出 `recommended_experiments=[]`。诊断假设只能记录在
`diagnostic_hypotheses` 和 `next_agent_guidance_zh`；父流程仅可据此冻结 P 的范围裁剪
计划，且目标必须是已有 exact `neutral` 或 `regressed` range。

分析 skill 不启动实验、不编辑 scope，也不声明最终 promotion。父流程创建的 P 计划必须
只含一个或多个已证明 range 的 source action，显式 `declared_option_changes=[]`，且不得包含 `option`、
`value` 或 Option JSON Pointer。本节不适用于 Stage O DCCI 的唯一 before+after 联合修复。

scope action 固定映射为 `beneficial -> keep`、`regressed -> prune`、
`neutral -> prune`、`insufficient_evidence -> reprofile`。若原动作是 `prune`，但不是
analyzer 重放通过的 `source_scope_map_v2 + exact`，或 identity 缺少
`source_scope`、boundary start/end，动作
降级为 `block`，classification 不变。

`scope_actions` 必须保留 decision 的 `source_scope`、完整 boundary 和
`ordered_child_op_sequence`。boundary 的 start/end op 只能证明融合边界，不能证明源码
区间互不重叠；只有 source file 与递增 start/end offset 同时存在时才写
`interval_unproven=false`。缺少该证明不改变已有 classification，但 strategy 必须逐
range 生成 P，不得批量裁剪或从 op 名称顺序推断 source interval。

## 禁止的因果推断

- 不得把 interval 相关性、duration sum 变化或单次 profiling 结果写成已证明根因。
- 不得在配置证据缺失时声明 DCCI、`isScheModeOn`、aggressive 或并行选项状态。
- 不得仅凭 scalar、cache、wait 或控制核指标推荐多个同时变化的选项。
- 不得仅凭 child count、函数名、Task ID 或 occurrence index 推断性能原因。
- 不得把诊断模式下的结果当作 clean performance 或最终 promotion 证据。
- 只有 AOT 日志明确支持时才可提出 aggressive 类选项假设，仍须单变量验证。
