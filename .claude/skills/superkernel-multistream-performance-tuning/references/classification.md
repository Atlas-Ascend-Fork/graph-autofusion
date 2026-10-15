# 多流分类与度量

## 三个独立维度

主性能分类继续使用 sibling analyzer 的 SK-off interval P50 对 fused SK P50：

```text
beneficial | neutral | regressed | insufficient_evidence
```

多流维度单独记录：

```text
improved | preserved | degraded | unknown
```

调优状态记录：

```text
no_action | opportunity | validated | blocked
```

因此 `beneficial + degraded + opportunity` 是合法且重要的状态：当前 SK 保留，调优失败
不影响晋级。

`beneficial` 不是多流筛选终点。融合可能靠 launch 减少、child work 缩短或其他局部收益
覆盖原有并行损失，形成“已有净收益但仍有额外空间”的隐性机会。所有 performance-exact、
双方至少三个 occurrence、`multi_stream_detected=true` 的 SK 都必须进入双侧 trace 筛选，
不得只筛 `neutral/regressed`。使用 `multistream_opportunity_discovery.py discover` 生成完整
目标清单，并在分析后用 `audit-coverage` 证明 beneficial target 没有被遗漏。

## 每个 occurrence 的指标

融合前和融合后分别计算，不跨时钟域比较 timestamp：

```text
interval = max(end) - min(start)
duration_sum = sum(duration)
union = union([start, end))
max_active_streams
stream_pair_overlap
core_family_pair_overlap
wait/sync duration
```

`core_family` 必须由 SK-off `kernel_details.csv` 的三字段派生，而不是按名称或 lane 猜测：

| Accelerator Core | Mix Block Num | family |
| --- | ---: | --- |
| `AI_VECTOR_CORE` | 任意合法值 | `VECTOR` |
| `AI_CORE` | 任意合法值 | `CUBE` |
| `MIX_AIV` | `0` | `VECTOR` |
| `MIX_AIV` | `>0` | `MIX` |
| `MIX_AIC` | `0` | `CUBE` |
| `MIX_AIC` | `>0` | `MIX` |

`Block Num` 也必须随 capture 保存并校验为非负整数，三字段共同构成资源身份。

输出至少包含 baseline/candidate 的 P50、P90、MAD，以及：

```text
cube_vector_overlap_us
cube_vector_overlap_ratio
same_resource_overlap_us
mix_competition_overlap_us
overlap_work_us
```

多 stream pair 的 overlap sum 可能重复计算同一 wall-clock 区间，只能作资源并发诊断；
不得把它当成可直接回收的理论收益。

## 诊断分解

允许报告以下观测量：

```text
lost_overlap_us
child_work_inflation_us
wait_sync_delta_us
unexplained_residual_us
```

trace analyzer 还必须显式报告：

```text
lost_stream_pair_overlap_us
lost_cube_vector_overlap_us
cube_vector_overlap_retention_ratio
max_active_streams_delta
latent_opportunity
actionable_opportunity
```

`parallelism_effect` 先按双方稳定 cross-stream overlap 的保持程度判断，不以资源可动作性
替代事实分类。之后再判断动作资格：只有纯 Cube/Vector overlap 的稳定损失可得到
`optimization_status=opportunity`；Mix、同资源、wait/communication 即使确实从并行变串行，
也只能是 `degraded + blocked`。baseline/candidate child origin identity 集合必须完全一致且
每侧 occurrence 内稳定，否则不能比较 overlap。

默认至少要求 P50 overlap 绝对变化 `1 us`，并同时使用 retention ratio：降到 baseline 的
`50%` 或以下才判 degraded，增至 `120%` 或以上才判 improved，中间为 preserved。绝对阈值
避免极小 overlap 的比例放大。若 SK-off 至少双 stream 但稳定 overlap 小于绝对阈值，则记录
`parallelism_basis=baseline_cross_stream_overlap_absent` 并 `no_action`，不能把它当成证据缺失。
纯 C/V overlap 的显著损失优先判 degraded，不能被新增 Mix/同资源 overlap 的总量掩盖。
阈值必须写入输出并参与 deterministic replay。

它们不是严格可加的根因公式。只有单变量 trial 稳定改变目标信号并改善端到端结果，才能
把该动作记录为 validated mechanism。局部 overlap 恢复但 clean E2E 不改善时，trial 必须
回退，机制证据可保留。

## 优先级

优先分析高频且对 decode critical interval 贡献较大的 target。不得把 occurrence 频次乘
单次 overlap loss 直接声明成端到端收益。排序只用于预算分配，不改变 acceptance gate。
