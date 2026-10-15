# SK 融合算子性能分析报告模板

## 0. 文档定位

本模板用于同一 workload 下的 `SK-off`（S0）与 `SK-on`（Sx）profile 对比，回答两个不同的问题：

1. 一个融合 SK 相比其在 S0 中对应的原始子算子集合，端到端设备区间是否变快；
2. 若融合 SK 的表现异常，哪个 S0 child 或融合内 child 是主要审阅对象。

主比较口径始终是 **S0 child interval P50 对 Sx parent SK duration P50**。S0
`duration_sum` 与 `union_duration` 只解释串并行和累计工作量，不能替代 interval 作
keep/prune 或收益结论。Sx 融合内 child 时间仅在完整、可归属的 `sk_prof` 下作为补充诊断；
不能把它与 S0 的 host 时钟做绝对时间偏移比较。

本模板不包含任何特定模型、算子序列、raw SK ID、step ID 或性能数值。真实采集数据只能出现在
独立的性能分析报告中；模板只定义报告字段、统计口径和结论边界。

### 适用范围（硬边界）

本模板**仅**用于普通 `S0 SK-off` 对 `Sx SK-on` 的融合性能对比，且两侧必须满足本 skill 的
profile、结构映射和统计门禁。不得用于以下场景：

- 两个 SK-on 候选之间的源码调序、多流或 source-reorder 对比；
- Stage O 的 O0/O1 DCCI option 对比；
- correctness、clean 性能、编译性能或其他非 profile-vs-baseline 对比。

这些场景必须使用各自的实验/诊断报告格式和原生统计口径；不得复用本模板的 S0 interval、
Sx parent、逐 SK classification、层级 family 或 scope-action 展示结构。

---

## 1. 一页结论（必填）

| 项目 | 固定输出 |
| --- | --- |
| 对比对象 | S0 SK-off vs Sx SK-on；模型、device、workload、轮次 |
| 采集有效性 | manifest-set、profile/process ownership、配置和 workload 一致性、声明变更 |
| 映射覆盖 | SK 总数、精确映射数、阻塞数、每项配对样本数 |
| 分类汇总 | beneficial / neutral / regressed / insufficient_evidence 的实例数 |
| 主要收益 family | 以主口径列出 family、实例数、P50 收益和离散度 |
| 主要劣化 family | 列出所有实际 regressed 实例，不以 family 平均值掩盖 |
| 行动边界 | 可测量结论、待 reprofile 项、是否具备源码动作证据 |

## 2. 指标定义与符号（必填）

对 S0 中同一 SK 的一个 child occurrence 集合：

```text
child_end = child_start + child_duration
S0 interval = max(child_end) - min(child_start)
S0 duration_sum = sum(child_duration)
S0 union = child 区间并集长度
Sx parent duration = Sx kernel_details 中该 SuperKernel 的 Duration(us)

improvement_us  = P50(S0 interval) - P50(Sx parent duration)
improvement_pct = improvement_us / P50(S0 interval) * 100
noise_pct = max(3.0, 2*S0_interval_MAD/S0_interval_P50*100,
                2*Sx_parent_MAD/Sx_parent_P50*100)
```

| 统计量 | 含义 | 在报告中的用途 |
| --- | --- | --- |
| P50 | 第 50 百分位数，即中位数；50% 的样本不大于该值 | 作为 S0 interval 与 Sx parent 的主比较值，计算收益或劣化 |
| P90 | 第 90 百分位数；90% 的样本不大于该值 | 观察慢尾与波动上界，不单独决定 classification |
| MAD | `median(|x_i - P50|)`，样本相对中位数的中位绝对偏差 | 稳健衡量离散度，参与 `noise_pct` 计算；不易被单个异常值主导 |

所有时间统计量以 `us` 记录。P50 表示典型耗时，P90 用于发现尾部变慢，MAD 用于判断观测到的
差异是否大于重复采样波动；三者必须同时随逐 SK 比较输出。

### 2.1 网络层数与层级融合覆盖（条件必填）

从当前模型配置读取 `num_hidden_layers`，并以配置中的一基 layer index 报告网络层数与层类型。
逐层融合表必须区分“配置层信息”和“已证明的 SK-to-source-layer 映射”：只有
`source_scope_map_v2 + exact` 可以把一个 profile SK 写入具体源码层；graph occurrence ordinal、
raw SK ID 或 family occurrence 数都不能替代层映射。下表中的 `<family_label>` 必须与第 5 节的
family 标签完全一致，以便从某层的融合清单直接跳转到该 family 的性能结论。

| 字段 | 固定输出 |
| --- | --- |
| 总层数 | `<num_hidden_layers>` |
| 层类型分布 | `<layer_type>`、一基 layer index 列表及计数 |
| 每层融合清单 | 每一个一基 `<layer_index>` 的 `<family_label>` 列表；没有融合时显式写“无” |
| family 层覆盖汇总 | `<family_label> / <covered_layer_count> / <covered_layer_ranges> / <appears_in_all_layers>` |
| 映射边界 | `<exact source layer mapping | unproven>` |

有 exact source-layer mapping 时，正文按下列两个表展示。逐层表可以按连续、且融合 family 列表
完全相同的层范围折叠，但必须保留所有层号；family 汇总中的 `covered_layer_ranges` 必须由逐层表
可逆展开。`appears_in_all_layers=是` 仅当该 family 覆盖 `1..num_hidden_layers` 的每一层时成立。

| 层号或连续层范围 | 配置层类型 | 融合 family（与第 5 节同标签） | 映射状态 |
| --- | --- | --- | --- |
| `<layer_index_or_range>` | `<configured_layer_type>` | `<family_label_0>`、`<family_label_1>`、... / 无 | `source_scope_map_v2 + exact` |

| family（与第 5 节同标签） | 覆盖层数 / 总层数 | 覆盖层号或范围 | 是否覆盖全部层 | 映射状态 |
| --- | ---: | --- | --- | --- |
| `<family_label>` | `<covered_layer_count> / <num_hidden_layers>` | `<covered_layer_ranges>` | `<是 / 否>` | `source_scope_map_v2 + exact` |

没有 exact source-layer mapping 时，报告必须写明“当前 profile 不能证明每一个 SK 属于哪一层”，
不得输出或推测逐层融合清单、覆盖层号、覆盖比例或“全部层出现”。此时仅可列出 profile family
及其 occurrence 数，并将层覆盖标记为 `unproven`。

### 2.2 分类规则

| 分类 | 条件 | 固定行动语义 |
| --- | --- | --- |
| beneficial | `improvement_us >= 1us` 且 `improvement_pct >= noise_pct` | 性能上建议 keep；非源码 action |
| regressed | `improvement_us <= -1us` 且 `improvement_pct <= -noise_pct` | 优先审阅；无 source map 不做源码 prune |
| neutral | 数据充分但未达到上两类 | 无明确收益；不代表错误 |
| insufficient_evidence | 映射、样本、尾部、fingerprint 或 metadata 门禁失败 | 记录 blocker 并 reprofile/block |

---

## 3. 逐 SK 性能结论（摘要 + 全量明细）

本章面向阅读：先展示四类结论的数量，再定位需要审阅的实例。每一个精确 graph occurrence 的
完整行记录保存在机器可读 JSON/CSV 中；Markdown 不直接铺开全量 SK 表，避免 100+ 行表格掩盖
实际 regression 和证据不足项。

### 3.1 分类概览（必填）

| classification | 实例数 | 占比 | 阅读动作 |
| --- | ---: | ---: | --- |
| beneficial | `<count>` | `<pct>` | 在 family 表中确认是否一致；不等同于端到端晋级 |
| neutral | `<count>` | `<pct>` | 只在有明确优化假设时抽样审阅 |
| regressed | `<count>` | `<pct>` | 全部进入“明确劣化实例”表 |
| insufficient_evidence | `<count>` | `<pct>` | 按 blocker 汇总，列出重采条件 |

### 3.2 阅读路径与全量明细（必填）

| 读者问题 | Markdown 展示 | 机器可读全量数据 |
| --- | --- | --- |
| 是否存在明确劣化 | 列出全部 `regressed` 实例及其主指标 | `profiling-analysis-result.json` |
| 哪些融合 family 值得关注 | family 汇总及 split / weak-benefit | `fusion-family-summary.csv` |
| 某一个 SK 的全部统计和 identity | 正文只展示代表性或异常实例 | `profiling-analysis-result.json`、`fusion-family-layer-comparison.csv` |

每个全量逐 SK 记录必须包含：稳定报告身份、ordered child sequence、stream/core topology、对齐
occurrence、S0 interval/duration_sum/union、Sx parent P50/P90/MAD、收益、noise threshold、
classification、blocker 和 scope action。它是自动判定的权威数据，不以 op 名称或 raw SK ID 聚合。

### 3.3 代表性多 child SK（条件展示）

正文针对多 child、明确劣化或需解释的 SK，使用下面的紧凑卡片；不把该实例外推为同 family 的
所有实例。

| 项目 | 数值 |
| --- | ---: |
| child topology | `<child_0>(<core_0>) -> <child_1>(<core_1>) -> ... -> <child_n>(<core_n>)` |
| S0 interval P50 | `<s0_interval_p50_us> us` |
| S0 union P50 | `<s0_union_p50_us> us` |
| Sx parent P50 / P90 / MAD | `<sx_parent_p50> / <sx_parent_p90> / <sx_parent_mad> us` |
| improvement / noise | `<improvement_us> us / <improvement_pct>% / <noise_pct>%` |
| classification | `<beneficial | neutral | regressed | insufficient_evidence>` |

不能用 child duration 的加和替代 S0 interval，也不能用单个实例替代同 family 的其他实例。

---

## 4. 融合内 child 与调度诊断（条件必填）

仅当 Sx `sk_prof` 完整、无 buffer-full、且 parent occurrence 与 child/lane 可按
`device_id + model_id + sk_id + step_id` 绑定时，追加下表。否则固定输出一句：
“`sk_prof` 不完整或不可归属，融合内 child/lane 诊断未执行；不得推断 Cube/Vector 串行、
DCCI 状态或根因。”

| parent identity | child position/symbol | lane 类型 | Sx child wall/max-lane P50/P90/MAD | 对应 S0 child P50 | 观察 | 置信度 |
| --- | --- | --- | --- | --- | --- | --- |
| `<填充>` | `<填充>` | CUBE / VECTOR / MIX | `<填充>` | `<填充>` | 仅事实与可检验假设 | high / low / insufficient |

允许写“调度假设”，禁止把没有 A/B 验证的结论写成根因。若有 DCCI disable-all 数据，另用
Stage O 专用表比较 O0/O1 和逐 child，不与普通 S0/Sx 表混用。

---

## 5. Fusion Family 与跨层/重复实例比较（必填，出现至少三个重复实例时）

family identity 固定为：

```text
ordered canonical child op sequence
+ ordered child source-stream-role pattern
+ ordered child core-family pattern
+ fusion boundary
+ stable dtype/shape/port signature（若可得）
```

不能用 layer index、raw SK ID、task ID 或生成名称定义 family。若没有 exact source mapping，
仅称“graph occurrence”，不可称“源码层”。

| family | topology | instances | B/N/R/I | S0 interval P50 min/median/max/MAD | Sx parent P50 min/median/max/MAD | 收益 P50 | split | weak benefit / outlier |
| --- | --- | ---: | --- | --- | --- | ---: | --- | --- |
| `<family_id>` | `<child_0> -> <child_1> -> ... -> <child_n>` | `<count>` | `<B/N/R/I>` | `<min / median / max / MAD>` | `<min / median / max / MAD>` | `<pct>` | `<是 / 否>` | `<count / count>` |

family 表是阅读性诊断，不能改变逐 SK classification 或用 beneficial 实例覆盖同 family 的
regressed/insufficient 实例。family 同时出现 beneficial 与 regressed 时标记
`family_classification_split`，所有实际 regressed 实例另列优先审阅表。

---

## 6. 结论、优先级与后续动作（必填）

结论必须分为三层，避免越权：

| 层级 | 允许结论 | 禁止结论 |
| --- | --- | --- |
| 逐 SK 性能 | beneficial / neutral / regressed / insufficient_evidence | 用家族平均值取代逐实例结论 |
| 调度诊断 | 有证据时提出可检验假设 | 从不完整 `sk_prof` 宣称 root cause |
| 源码行动 | 仅 source scope map exact 时提出范围动作 | 从 raw ID、graph ordinal 或 op 名称反推源码行号 |

固定排序：先列所有 `regressed`，再列 `insufficient_evidence`，再列多 child 的
`weak_relative_benefit`，最后列 `neutral`。每个条目必须有：报告 identity、数据事实、
门禁状态、可执行的下一步及其前置采集条件。

报告结论必须同时写明：逐 SK 分类是否可用、收益或劣化的 family 分布、不能由 family
覆盖的异常实例、是否具备源码 action 证据，以及未执行的融合内诊断及其原因。

---

## 7. 固定交付物与机器可读约束

每次普通 S0/Sx 性能对比拟固定生成：

1. `PROFILING_ANALYSIS.md`：一页结论、全量逐 SK 主比较和结论边界；
2. `profiling-analysis-result.json`：逐 SK 机器结果、fingerprint、统计、分类和 blocker；
3. `PROJECTED_TRACE_MAPPING.md` / `projected-trace-mapping.json`：S0 child 与 Sx SK 的精确映射；
4. `FUSION_FAMILY_LAYER_ANALYSIS.md`、`fusion-family-summary.csv`、
   `fusion-family-layer-comparison.csv`：重复实例/跨层诊断；
5. 仅在 lane 采集完整时增加 `SK_CHILD_SCHEDULING_ANALYSIS.md` 与对应 JSON；
6. 仅在 Stage O DCCI 模式时增加 `DCCI_REGRESSION_ANALYSIS.md` 与
   `dcci-regression-analysis.json`，不混入 schema 1.2 结果。
