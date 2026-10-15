# 跨层 Fusion Family 性能分析

## 触发与输入

在完成同一 workload 的 SK-off/SK-on `kernel_projection_trace_v2` 与逐 SK
classification 后，只要同一 model/device 存在至少三个可比较的重复 block occurrence，必须
增加这份只读诊断。它用于回答“同类融合在不同层是否一致”，不能替代逐 SK 的
`beneficial`/`neutral`/`regressed`/`insufficient_evidence` 判定。

只允许使用下列输入：

- 当前轮 `profiling-analysis-result.json` 中 `mapping_confidence=exact_projected_trace`
  的逐 SK 结果；
- 同一轮 `projected-trace-mapping.json` 的 ordered child、source stream role、core family
  与 step-level S0 child interval/duration-sum；
- 同一 candidate profile 的 S3 parent SK P50/P90/MAD。

禁止混入其他 model、device、workload、candidate、compat/verify 或旧 round。没有 exact
projection 的实例单列为未比较项；不得用 raw SK ID、Task ID、graph ordinal、层号、生成名称
或局部算子窗口补配。

## Family Identity

跨层 family 不是原逐 SK identity 的替代。它必须同时保留：

```text
model_id
+ ordered canonical child op sequence
+ ordered child source-stream-role pattern
+ ordered child core-family pattern
+ fusion boundary type
+ any stable dtype/shape/port signature available in projection evidence
```

`candidate_source_scope`、raw runtime ID、baseline task/stream ID、绝对 kernel ordinal 和
layer index 不进入 family identity；它们分别作为实例位置或诊断字段保留。若 dtype、shape、
port、core family 或流拓扑不同，必须拆为不同 family，即使 op 名称序列相同。不得假设“同名
算子序列”必然是同一执行模板。

## 必需产物

在常规 `PROFILING_ANALYSIS.md` 之外，输出下列 companion artifacts：

- `FUSION_FAMILY_LAYER_ANALYSIS.md`：中文结论、family 覆盖层、层模板布局、离群实例及逐层表；
- `fusion-family-summary.csv`：每个 family 的子算子/流数、覆盖层、S0/S3 P50 min/median/max/MAD、
  P90、收益分布、分类分布、奇偶层中位差与异常实例；
- `fusion-family-layer-comparison.csv`：每个 family-layer instance 的 raw SK ID、scope、完整 child
  序列/stream topology、S0 interval、S0 duration sum、S3 duration、P90/MAD、收益和 classification。

逐层表必须保留 child 数和完整拓扑，不能把不同结构的实例合并成“每层平均值”。`duration_sum`
仍只用于解释并行工作量；S0 `interval` 对 S3 parent duration 仍是收益主口径。

若网络存在交替模板，显式列出其 layer sets，例如“所有 decode 层共享”、“偶数层模板”、
“奇数层模板”或“前缀特殊模板”。不要将偶数层与奇数层的不同 family 误写为同一 SK 的性能波动。

## Family 诊断

先报告可观测事实，不做根因宣告：

1. 对每个 family 统计 S0 interval P50、S3 duration P50、收益率的 min/median/max/MAD/P90，
   并列出每层原始值；奇偶对比只对同一 family 同时覆盖奇偶层时计算。
2. 以 `max(3 * family_MAD, 10% * family_median, 1 us)` 标记 S3 duration 的阅读性离群实例。该
   标记不能覆盖逐 SK 的动态 classification 门禁。
3. family 中同时出现至少一个 `beneficial` 和至少一个 `regressed` 时，写
   `family_classification_split`。所有 `regressed` 实例是优先审阅对象；不要因为同 family 的其他层
   beneficial 而稀释、平均掉或把它们自动归为噪声。
4. `insufficient_evidence` 只表示该实例不能作分类。若是 `minimum_sample_unrepresented_tail`，
   先扩大同一逻辑 profiling window；不得借用同 family 其他层的样本来替代它。

### 多 child 的弱收益复核

child 数量是复杂度和审阅优先级信号，永远不是 keep/prune 的门槛。对 child 较多而正收益很小的
融合，必须增加 `weak_relative_benefit` 诊断，而不是自动拒绝它：

- 只在该 family 至少有 5 个已分类（非 `insufficient_evidence`）实例时计算 family 收益中位数和
  MAD；
- 若一个实例 `improvement_pct > 0`，但低于
  `family_gain_median - max(3 * family_gain_MAD, 3 percentage points)`，标为
  `weak_relative_benefit`；
- 若 owner 已经在实验控制中声明 `expected_improvement_pct`，还要列出低于该期望的实例；没有
  明确期望时，不得自行发明 2%、5% 等绝对收益门槛；
- 报告按 child_count、S0 interval 和与 family 中位差排序，保留每个 child 的时间与流布局，优先
  供人工检查较大融合范围。

`weak_relative_benefit` 是复核队列，不是 `regressed`，不会改变既有 classification/action，
也不能直接触发 scope prune、option trial 或源码重排。先确认它是否来自 S0 baseline 的 layer
差异、S3 尾部、失去并行，还是已知的最小样本长尾；只有后续单变量实验才能形成原因结论。

## 后续动作边界

- `family_classification_split`：在报告中逐层展开 S0/S3 指标和 child 拓扑；对每个实际
  `regressed` instance 沿既有 source-map gate 处理。禁止 family 级批量 prune 或 batch source edit。
- `weak_relative_benefit`：列为人工 review priority；可在有完整 child trace 时形成调度/覆盖损失
  假设，但不得宣称根因或直接执行多流重排。
- 多流重排仍须遵从独立的 multistream capability、依赖审计与单变量 A/B 流程。跨层相关性不能证明
  一个 reorder 对整个 family 或所有 layer 有效。
- 报告中的“同一 family”只证明融合拓扑相同，不证明底层形状、资源状态、调度和端到端收益相同。
