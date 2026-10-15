# SuperKernel Profiling 与融合收益分析

> 映射协议更新：融合前子算子片段必须优先使用
> `superkernel-fusion-performance-analysis/scripts/projected_trace_mapping.py`
> 的 `kernel_projection_trace_v2`。不要再用静态 control 全图同构或
> `kernel_details`/`sk_prof` 绝对时钟偏移作为主映射门禁。`sk_prof` 只在 baseline
> fragment 已由完整 step kernel projection 唯一证明后用于融合内调度归因。

本文件定义执行 Agent 如何采集数据、只读分析 Agent 如何分类，以及父 Agent 如何
消费结果。所有人类可读分析结果必须是中文。

## 1. Profiling 入口门禁

本文只适用于阶段 A clean timing 选出的 winner、其 BASE，以及已明确请求的 P/FINAL
深度优化轮。
S1/S2/S3/S4/... 筛选轮不得采集 candidate diagnostic profiling、执行 SK 映射或调用
本 analyzer。winner 的每个 profiling round 均不受 `child_count`、single-child、
shallow 或 deep 标签影响。`child_count=1,2,3,4` 与更深融合走完全
相同的 inventory、关联、采样和分类路径，不允许设置 minimum child-count filter。
开始 profiling 分析前要求：

1. 被 profiling 的 candidate process 完整执行并通过正确性；
2. `baseline_profile` 与 `candidate_profile` 两个 fresh immutable root 的 collection
   manifest 全部通过，并使用同一份只含共同结构不变量的 association config；
3. candidate profile 根同时包含该 profiling 进程自己的 profiler 三件套、
   origin/updated graph、fused/scope/super-kernel metadata；compat metadata 不得替代
   profile metadata；`sk_prof` 只作为可选的融合内调度诊断输入；
4. baseline/candidate config 与 workload manifest 可生成稳定 fingerprint。

compat/verify 与 generic replay 可以作为候选侧独立复现审计，但不是性能分析入口，
不得传给 analyzer，也不得因重复 signature、unmatched replay identity 或审计缺失把
已由 profile 与 baseline 唯一证明的映射降级。只有两份 profiling 根自身的映射、
样本、fingerprint 或 artifact 不完整才返回 `insufficient_evidence`。

## 2. 执行与分析隔离

主实验子 Agent：

- 运行 SK-off/SK-on diagnostic profiler；
- 冻结 profile 内容并生成 fingerprint；
- 写 relative-path analysis request；
- 不自行分类、不手工计算阈值、不修改 analyzer 输出。

每轮由不同的 fresh read-only profiling analysis child agent 使用
[superkernel-fusion-performance-analysis](../../superkernel-fusion-performance-analysis/SKILL.md)。
分析 Agent 只读取 request artifacts，只写本轮专属 JSON 与中文 Markdown。

nested dispatch 不可用时，主实验子 Agent 把 request 交给父 Agent，父 Agent 派
fresh analysis Agent 后回填 response。主实验子 Agent不得代算。

## 3. 两角色性能 Artifact 布局

```text
experiments/S0/profile/association-artifact-manifest.json
experiments/S0/profile/profiler/kernel_details.csv
experiments/S3/S3-BASE/profile/association-artifact-manifest.json
experiments/S3/S3-BASE/profile/profiler/kernel_details.csv
experiments/S3/S3-BASE/profile/profiler/task_time.csv
experiments/S3/S3-BASE/profile/profiler/trace_view.json
experiments/S3/S3-BASE/profile/profiler/sk_prof_device_0.json  # 可选诊断 trace
experiments/S3/S3-BASE/profile/sk_meta/{sk_graph_origin,sk_graph_updated,...}
experiments/S3/S3-BASE/profiling-analysis/request.json
experiments/S3/S3-BASE/profiling-analysis/profiling-analysis-result.json
experiments/S3/S3-BASE/profiling-analysis/PROFILING_ANALYSIS.md
```

Request 至少绑定：

- experiment ID、round ID、analysis Agent ID、source revision；
- baseline/candidate config 和 workload manifest；
- baseline/candidate profile 相对路径及内容 fingerprint；
- control、workload、baseline config、candidate config fingerprints；
- association config 及其独立 fingerprint；
- candidate profile 自有 sk_meta 与 source scope map；
- `baseline_profile`、`candidate_profile` 两个 collection manifest；
- `declared_change_set`；
- 可选 SK child trace 与 environment evidence。

任何 mismatch 都是 untrusted，返回 `insufficient_evidence`，不得尝试“合理补齐”。

## 4. 直接调用 Sibling Analyzer

新流程直接调用 sibling skill。`<skill-dir>` 必须先解析为当前
`superkernel-auto-tune` skill 根目录（即包含 `SKILL.md` 的目录），不得依赖 shell
当前工作目录：

```bash
python3 <skill-dir>/../superkernel-fusion-performance-analysis/scripts/analyze_fusion_performance.py \
  --baseline-profile experiments/S0/profile/profiler/kernel_details.csv \
  --candidate-profile experiments/S3/S3-BASE/profile/profiler/kernel_details.csv \
  --sk-meta experiments/S3/S3-BASE/profile/sk_meta \
  --baseline-config experiments/S0/config.json \
  --candidate-config experiments/S3/S3-BASE/config.json \
  --baseline-workload experiments/S0/workload.json \
  --candidate-workload experiments/S3/S3-BASE/workload.json \
  --declared-change-set experiments/S3/S3-BASE/declared-change.json \
  --source-scope-map experiments/S3/S3-BASE/source-scope-map.json \
  --baseline-collection-manifest experiments/S0/profile/association-artifact-manifest.json \
  --profile-collection-manifest experiments/S3/S3-BASE/profile/association-artifact-manifest.json \
  --candidate-name S3 --experiment-id S3 --round-id S3-BASE \
  --analysis-agent-id analysis-S3-BASE --source-revision REVISION \
  --json-out experiments/S3/S3-BASE/profiling-analysis/profiling-analysis-result.json \
  --markdown-out experiments/S3/S3-BASE/profiling-analysis/PROFILING_ANALYSIS.md
```

需要分析 SK 内部调度或 option 支持时追加：

```bash
  --sk-prof experiments/S3/S3-BASE/profile/profiler/sk_prof_device_0.json \
  --environment-evidence experiments/environment.json
```

这里可选的 `--sk-prof` 只用于 child-scheduling 诊断。映射不得要求它与
`kernel_details` 建立绝对时钟 offset，也不得因它缺失而否决已经由完整 step kernel
projection 证明的 baseline fragment。

两个 manifest 不能替换原有 profile、config、workload、change-set 和可选 source-map
输入。跨进程关联不得以 raw
Task/model/stream/node ID、生成名称或局部同名窗口作为 exact fallback；这些字段只可
保留为 diagnostic provenance。kernel projection 产生的
`graph_occurrence_fingerprint` 也不得自动转换成源码文件、行号或 byte offset。

## 5. 时间语义

Profiler 只有开始时间和执行时长时，analyzer 自动计算：

```text
end_time = start_time + duration
baseline_interval = max(child.end_time) - min(child.start_time)
baseline_duration_sum = sum(child.duration)
baseline_union = union(child.start_time, child.end_time)
```

主判据是端到端 baseline graph-occurrence timing interval；这是 profiler 时间区间，
不是源码 byte interval：

- `baseline_interval_P50`：融合前该 graph occurrence 从最早 child start 到最晚
  child end 的 P50；
- `sk_duration_P50`：融合后 SK occurrence duration 的 P50；
- `improvement_us = baseline_interval_P50 - sk_duration_P50`。

`baseline_duration_sum_P50` 是辅助诊断。多流 child 可以重叠，直接求和会重复计算
并夸大融合前耗时，因此不得用 duration sum 替代 interval 作分类。

## 6. MAD 动态阈值与分类

每个 range 至少需要三个可靠 occurrence。使用 P50、P90 和 MAD；动态相对噪声带
至少为 3%，绝对变化至少为 1 us。以 sibling skill 的实际阈值字段为准。

| classification | action | 语义 |
|---|---|---|
| `beneficial` | `keep` | interval 改善越过动态 MAD threshold |
| `neutral` | `prune` | 数据充分但没有明确收益 |
| `regressed` | `prune` | SK interval 明确劣化 |
| `insufficient_evidence` | `reprofile` 或 `block` | 映射、样本、fingerprint 或 artifact 不足，不改 scope |

`child_count`、launch saved、fusion depth 和 fragmentation 可以解释结果，不能
覆盖上述 classification。
`child_count` 不得作为 SK inventory、mapping 或 profiling 的过滤条件。

## 7. Winner-Only Whole-Scope、BASE 与可选 P/FINAL 消费规则

阶段 A 的 `selected_for_deep_analysis` 先冻结为 `Sbest-SEED`，完成不采 profile 的
winner option sweep 后再把保留配置冻结为 `Sbest-BASE`。非 winner 候选不得进入本节，
也不得为比较完整性补 profile。整 scope 特性晋级与源码细粒度优化是两条
独立路径。若当前 winner 的所有 SK 都是
performance-exact，analysis 无 blocker 且没有 `insufficient_evidence`，strategy 必须
生成 `whole_scope_clean_validation`。该路径禁止源码/scope/config/workload 变更，局部
neutral/regressed 只作诊断，最终由至少三个 clean 进程相对冻结 S0 的端到端均值、P90
和标准差决定。`P/FINAL` 是可选源码范围分支，仅在用户或冻结实验计划明确请求时
启动；BASE 或 SMAP 完成不得自动创建它。源码 offsets 仅是该分支范围优化 FINAL 的门禁。
分支 absent、invalid、`no_gain`、`blocked` 或 `failed` 时保留 incumbent，且不得阻塞
unchanged `whole_scope_clean_validation`。

### BASE

为每个可靠 mapped SK 形成决策。`kernel_projection_structural + exact_projected_trace` 可支撑
measured interval 的 beneficial/neutral/regressed 分类，但只能保留 proposed 性能
证据。只有 producer `mapping_method=source_scope_map`、
`mapping_confidence=exact` 且有可证明源码区间的 neutral/regressed range 才可交给
P。这里的 exact 只接受 analyzer 重放通过的 `source_scope_map_v2`：它必须绑定当前
graph occurrence、baseline projection、可动作 source revision、三步 calibration
assignment、archived source bytes 和 provenance DAG。automatic 路径还必须包含
stable_source + marker-only bridge。历史 task range/layer map、
`sk_meta_node_ids + exact`、raw ID/name/local-window 关联和任何 ambiguous、
unmapped、diagnostic-only 结果都必须降级为 `insufficient_evidence`。不得提交或信任
consumer 自报的 `mapping_reliable`。

### P

实验 Agent 只移除 response 指定的 exact source range。批量 P 要求所有范围位于同一
`source_file`，且每对 `start_offset/end_offset` half-open 区间可机器证明互不重叠。
相邻区间允许；跨文件、交叠、包含和同边界拒绝。只有 op 名的 boundary 不能证明
源码区间，因此一 range 一 P。当前 fresh analysis 仍有 prune action 时只生成下一 P；
所有保留范围均为 beneficial 时才生成 FINAL。

 ### FINAL

把 retained beneficial ranges 组合后重新 fresh profiling。每个最终 range 必须
在 FINAL 当前数据中得到 `beneficial/keep`。这一步检查范围间交互；
通过后才允许 clean timing。

## 8. Winner Source Mapping And Automatic AOT

所有 winner 在 Sbest-BASE 的 per-SK profiling 后都检查 source exact。任一
performance-exact SK（包括 beneficial、neutral、regressed）尚无可编辑源码半开区间
时，先进入一次批量 winner-only `Sbest-SMAP`；在映射结算前不生成可选 P 或
whole-scope clean 晋级。命名/manual winner 使用 `stable_marker` 路径，automatic winner
使用 `stable_source` marker-only bridge。`insufficient_evidence` 不猜测源码，只保持
unresolved。映射完成后，partial/skipped range 跳过源码动作，exact range 独立继续。

automatic AOT 在 S 筛选阶段不做 profiling。只有它被选为 winner 后，才执行上述流程，
并且不得把 automatic candidate 重建为 manual named-scope family。

SMAP 在独立 worktree 增加临时语义 marker，以 `stable_source +
marker_only_calibration_v1` 证明删除 marker 后逐字节还原冻结 AUTO 源码，再通过唯一
original/calibration 图投影、三步 assignment 和 `C_sk == C_unit` 取得指向原始 AUTO
源码的 offsets。analyzer 复验 `source_scope_map + exact` 后，automatic winner 可直接
在 P/FINAL 已明确请求时进入该分支；partial/skipped/insufficient 的 SK 继续跳过，不阻塞 independently exact
范围。标定 marker 不保留到生产源码。
具体标定、临时 marker 生命周期和模型无关 adapter 约束见 sibling skill 的
`references/source-calibration-mapping.md`。

## 9. Regression Diagnosis

对 neutral/regressed SK，按直接证据检查：

1. baseline Cube/Vector 是否跨流重叠，SK child trace 是否串行化；
2. DCCI state 是否来自明确 config/environment，是否有 scalar/cache counter；
3. wait/notify、event、barrier、通信和 cache mutation 是否增加；
4. control-core `isScheModeOn` 是否有可靠 metadata；
5. resource、launch、task 或 scope boundary 是否改变。

DCCI 与 `auto_op_parallel` 仍可作为机制诊断线索，但它们已在 Stage O 结算，不能在
P/FINAL 中形成新的 option 或源码实验。O 的端到端收益不得反向写成 `root_cause`。

## 10. 中文分析结果

`profiling-analysis-result.json` 保存机器可校验字段；
`PROFILING_ANALYSIS.md` 保存中文分析结果。每个人类可读 SK 性能表至少包含：

| Range/SK | child count | interval P50 | duration sum P50 | SK P50 | MAD/dynamic threshold (%) | classification（英文 + 中文）/action | mapping confidence | analysis Agent | 条件证据绑定 |
|---|---:|---:|---:|---:|---|---|---|---|---|

其中“条件证据绑定”仅汇总 analyzer 当前报告中的 `source_revision`、baseline/
candidate config fingerprint、control/workload fingerprint 与本行 `range_id`，不推断
父账本中的晋级或应用状态。相对 MAD 动态阈值取本行 `noise_threshold_pct`；绝对
阈值继续在报告的“判定阈值”表中展示。

同时报告：

- interval P90、SK P90、occurrence count 和 mapping method；
- profile 路径与 fingerprints、source revision、declared change；
- scope/range boundary、ordered child sequence、stream/overlap；
- blocker、diagnostic hypothesis、recommended diagnostic experiment；
- 本轮对下一轮的中文指导。

若任一必填指标不可得，表格写 `N/A` 并说明 blocker，不能省略列。

## 11. Clean Timing 边界

Diagnostic profile、SK child trace、metadata dump、debug sync 和 event trace 都不
进入 clean 性能样本。range-optimized 路径要求 FINAL fresh profiling 证明组合范围
仍全部 beneficial；unchanged whole-scope 路径要求完整精确映射、无
`insufficient_evidence` 且 candidate identity 不变。通过对应门禁后，实验 Agent 才
关闭诊断开关并运行至少三个 clean process。最终端到端结果使用原始冻结 S0；per-SK
interval 只解释原因，不替代端到端结论。
