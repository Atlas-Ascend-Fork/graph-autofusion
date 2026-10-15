# 性能分类规则

## 输入与映射门禁

分析前必须具备同一 workload 的 SK-off baseline profiling、当前轮 SK-on
candidate profiling、当前轮新鲜 `sk_meta`、baseline/candidate config manifest、
workload manifest、source revision、round identity 和 `analysis_agent_id`。
所有输入 artifact 必须使用相对于结果文件目录的路径。

逐 SK 性能分类只比较当前 candidate profile 与 SK-off baseline profile。compat/verify
的候选侧可复现性审计是独立证据，不参与 baseline 映射或性能分类。禁止按重复 op
signature 建立 replay identity，禁止因为 compat/verify 无法区分重复 signature 而写入
`fusion_replay_identity_unmatched` 或把 exact projected mapping 降级。

配置和 workload 使用 canonical JSON 的 SHA-256 fingerprint。分别保留
`baseline_config_fingerprint`、`candidate_config_fingerprint` 和
`workload_fingerprint`；另以移除本轮 `declared_change_set` 后的冻结控制生成
`control_fingerprint`。若 workload 或冻结控制 fingerprint 不一致，或实际配置变化
超出 `declared_change_set`，不得形成自动 scope 决定。

SK 使用以下稳定 identity 聚合：

```text
model_id + source_scope + ordered child op sequence + boundary
```

schema 1.2 只承认两种可用于 interval 性能分类的精确映射：

```text
source_scope_map + exact
kernel_projection_structural + exact_projected_trace
```

`source_scope_map + exact` 必须来自 source-confirmed layer/scope map；
`kernel_projection_structural + exact_projected_trace` 必须通过完整 step kernel 投影、
唯一 injective stream-role assignment、显式第二解搜索、至少三个 step 和 fingerprint
门禁。原始 Task ID、node ID、SK 名称及局部名称窗口只能写入 `mapping_hints`，不能提升为精确映射。

性能精确不等于源码可动作。只有 `source_scope_map + exact` 且 boundary 同时证明
`source_file/start_offset/end_offset` 时，`neutral` 或 `regressed` 才能驱动 prune。
`kernel_projection_structural + exact_projected_trace` 可以保留 measured classification；但 automatic
AOT 的图 occurrence 不能证明 Python 源码 offset，因此其源码动作必须保持 blocked 或
proposed source-mapping completion，不能直接变成 applied、verified、verified 或 FINAL
prune。winner-only SMAP 通过 `stable_source + marker_only_calibration` 独立取得
`source_scope_map + exact` 后，后续 AUTO-P/FINAL 按普通源码动作门禁执行。

## End Time、Interval、Duration Sum 与 Union

对每个 child 先计算：

```text
end_time = start_time + duration
baseline_interval = max(child.end_time) - min(child.start_time)
```

`baseline_interval` 保留不同 stream 的并行关系，是性能分类的主判据。另计算：

```text
baseline_duration_sum = sum(child.duration)
baseline_union = union(child.start_time, child.end_time)
```

`baseline_duration_sum` 表示累计设备工作时间；`baseline_union` 去除重叠后表示设备
忙碌区间。二者仅用于诊断并行串行化、额外同步、资源开销和调度行为，不得直接决定
keep/prune。

## Occurrence 聚合

融合前每组 child occurrence 与融合后每个相同 SK identity 必须分别聚合，baseline
和 candidate 都至少需要 3 个可用 occurrence。每个 occurrence 独立计算 interval、
duration sum 和 union；不得把整个 profiling 文件中多次执行的同一组子算子合并成一个
超长 interval。

baseline child 必须按可靠 occurrence key 对齐。key 至少绑定 `model_id`、`device_id`
和 `step_id`；只有 `step_id` 缺失且存在显式 `occurrence_id`、`iteration_id`、
`request_id` 或 `batch_id` 时，才允许用该 discriminator。不同 step 集合、漏步、同 key
重复、跨 model/device 混用都返回 `insufficient_evidence` blocker。输入行可以乱序，但
绝不得按 start time 排序后的序号硬配。

对每组指标计算：

```text
P50
P90
MAD = median(abs(value - median(value)))
```

baseline 或 candidate 低于 3 次、低于更高的 `min_occurrences`、identity 不稳定、时间
单位不一致或缺少 start/duration 时，返回 `insufficient_evidence`。

## P50、P90、MAD 与动态阈值

默认最低阈值必须写入输出：

```text
min_relative_change_pct = 3.0
min_absolute_change_us = 1.0
min_occurrences = 3
```

`min_occurrences=3` 是自动 `keep/prune` 的硬下限，不能被 CLI 降低。CLI 可以接收 1
或 2 用于诊断兼容，但输出 `thresholds.min_occurrences` 仍为 3，低样本 classification /
action 必须是 `insufficient_evidence/reprofile`。调用方只能把该门禁调高。

每个 SK 的动态相对阈值为：

```text
noise_pct = max(
    min_relative_change_pct,
    2 * baseline_interval_MAD / baseline_interval_P50 * 100,
    2 * sk_duration_MAD / sk_duration_P50 * 100
)
```

定义：

```text
improvement_us = baseline_interval_P50 - sk_duration_P50
improvement_pct = improvement_us / baseline_interval_P50 * 100
```

P50 用于主分类，P90 和 MAD 用于报告尾部行为与噪声。无法可靠计算上述统计值时，
不得用单次 duration、均值或猜测值替代。

当 baseline 或 candidate 恰好只有最低 3 个 occurrence 时，还必须检查 P50 两侧的
最大距离。若该距离大于 `max(3 * MAD, P50 * min_relative_change_pct,
min_absolute_change_us)`，说明一个单侧长尾没有被 MAD 噪声带表示；此时不得让 P50
所在的两点多数直接决定源码动作，必须输出
`insufficient_evidence/reprofile`，并记录
`minimum_sample_unrepresented_tail`。调用方应在同一逻辑轮次内扩大有效 profiling
窗口或重新采集，不得用另一次三点采样碰运气。

## 四种分类与 Scope Action

| 枚举 | 中文名称 | 含义 |
|---|---|---|
| `beneficial` | 明确有性能收益 | SK interval 显著优于融合前 interval |
| `regressed` | 明确性能劣化 | SK interval 显著差于融合前 interval |
| `neutral` | 无明确性能收益 | 数据充分，但变化落在动态噪声带内 |
| `insufficient_evidence` | 证据不足 | 映射、样本、fingerprint 或 artifact 不足以形成决定 |

分类规则：

- `beneficial`：`improvement_us >= min_absolute_change_us` 且
  `improvement_pct >= noise_pct`；
- `regressed`：`improvement_us <= -min_absolute_change_us` 且
  `improvement_pct <= -noise_pct`；
- `neutral`：证据充分，但不满足上面两类；
- `insufficient_evidence`：映射不可靠、occurrence 不足、fingerprint 不一致、metadata
  count mismatch 或输入缺失。

对应 scope action 为：`beneficial` 建议 `keep`；`neutral` 和 `regressed` 只有在
source-map exact 且源码区间已证明时才建议 `prune` 并进入待挽救队列。结构 exact 的
`neutral`/`regressed` 保留性能分类，但以
`prune_requires_exact_source_scope_boundary_mapping` 阻塞源码动作。可补采的
`insufficient_evidence` 建议 `reprofile`，否则建议 `block`。这些 action 是只读分析
结果，不得由本 skill 直接修改 scope。

每个 `scope_actions` entry 必须从对应 decision 原样保留 `source_scope`、`boundary`、
`ordered_child_op_sequence`。只有 boundary 同时包含无 `..` 段的非空相对
`source_file`、非负 `start_offset` 和严格更大的 `end_offset` 时，source interval 才可
证明；否则必须写 `interval_unproven=true`。下游 strategy 对 unproven interval 必须逐
range 生成 P，不能根据 op 名称顺序猜测 source 区间不重叠。

schema 1.2 的 Markdown renderer 只接受完整 producer report：identity、所有两侧
fingerprint、inputs、thresholds、四个列表、blockers、`next_agent_guidance_zh` 和
`analysis_content_fingerprint` 均必需。renderer 必须从 identity 字段重算
`analysis_id`，核验 `round_id` 属于 candidate 的 `BASE/Pn/Rn/FINAL` 家族，并在删除
content fingerprint 后按 canonical JSON 重算 SHA-256。`per_sk_decisions` 必须非空，classification/action 只能是
`beneficial/keep`、`neutral/prune`、`regressed/prune`、
`insufficient_evidence/reprofile|block`。`scope_actions` 必须按唯一的
`(sk_id, range_id)` 与 decisions 一一对应，且 classification、action、source scope、
boundary、ordered child sequence 和 interval proof 完全一致；缺失、多余、重复、签名后
置空、重算 hash 的语义空报告或截断 action 均拒绝，同时保持所有 HTML/Markdown 输入转义。

## Child Count 仅作描述

`child_count` 必须写入逐 SK 结果，但不参与 profiling 门禁、动态阈值或分类。单子算子
和浅层 SK 只要证据充分就必须分类；单子算子可以是 `beneficial`，多子算子也可以是
`neutral` 或 `regressed`。不得以 `child_count >= 5` 作为 profiling 条件，也不得以
`child_count == 1` 自动生成排除决定。
