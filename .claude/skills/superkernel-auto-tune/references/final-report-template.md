# SuperKernel 中文最终报告模板

每次适配都输出中文 `REPORT.md`，即使没有融合、全部候选失败、性能无收益、正确性
失败或预算耗尽。结论必须能从相对 artifact 和 schema 2 账本重放。

自动生成的 `FINAL_E2E_REPORT.md` 与人工完整 `REPORT.md` 遵循相同的首页规则。
标题后立即依次放置“结果速览 → 关键数据对比 → 配置与建议”；会话、环境和逐阶段细节后移。
结构化数据遵循 [report-summary.md](report-summary.md)，新 final worker 必须提供。

## 首页：结果速览、关键数据对比与配置

先用一句话分开说明整体调优结果和最终 clean E2E 状态，例如：
“Stage A 未获得合格候选，保留已验证 SK-off；最终 E2E 未执行（not_run）”。
成功、无收益、失败、阻塞和未执行都必须有首页，不能仅成功时输出数据。

| 阶段/候选 | 测量口径 | 基线耗时 ms | 实验耗时 ms | 耗时差 ms | 改善率 | 运行数 | 门禁/状态 | 证据 |
|---|---|---:|---:|---:|---:|---|---|---|
| S0（冻结基线） | 明确聚合口径 | 实测值或 N/A | N/A | N/A | N/A | 独立进程数 | 稳定性或 blocker | 相对 artifact |
| 各阶段关键候选 | screening / option / optional_clean / profiling + 聚合口径 | 同口径基线 | 实测值或 N/A | 实验−基线 | (基线−实验)/基线×100% | 独立进程数 | 门禁及状态 | 相对 artifact |
| 最终 E2E | final clean + 聚合口径 | 合法最终比较基线或 N/A | 最终实测或 N/A | 同口径差值 | 同口径收益 | 独立进程数 | beneficial/no_gain/failed/blocked/not_run | 相对 artifact |

仅列各阶段已接受最佳候选；全部不合格时列最佳已测尝试并标注“未通过门禁，非优胜者”，
优先沿用 selector 排名。无测量的阶段保留状态行，N/A 必须有原因。正改善表示更快；
mean、median、P50、不同 rank 聚合及 profiling 不互算。筛选退化不是最终 E2E 实测失败。
紧随表格给出阈值、P90/stddev 门禁与统计说明；完整候选和逐 run 数据放正文。

成功时给出最终 candidate、scope strategy、whole-scope/FINAL 路径、实际生效的优化与
debug option maps、配置路径、fingerprint 和证据。失败/无收益时写“优胜者：无”，展示
最后经过验证的回退配置。缺失选项不等于空 map，最佳尝试不等于推荐配置。

## 逐 SK 诊断表格契约（不适用于首页 E2E 表）

每个人类可读 SK 性能表至少包含以下列；不可得时写 `N/A` 和 blocker，不得删列：

| Range/SK | child count | graph occurrence | interval P50 | duration sum P50 | SK P50 | MAD threshold | classification/action | mapping method/confidence | source-actionable | analysis Agent | conditional evidence |
|---|---:|---|---:|---:|---:|---|---|---|---|---|---|

其中：

- interval P50 是融合前 source range 端到端区间主判据；
- duration sum P50 是融合前 child duration 总和，只作辅助诊断；
- SK P50 是融合后 SK duration；
- MAD threshold 必须给出绝对与相对动态噪声带；
- classification/action 使用
  `beneficial/keep`、`neutral/prune`、`regressed/prune` 或
  `insufficient_evidence/reprofile|block`；
- graph occurrence 给出 `graph_occurrence_fingerprint` 或 `N/A + blocker`；
- mapping confidence 同时给出 method 与 confidence；
- source-actionable 只有 `source_scope_map + exact + proven offsets` 才能写 `yes`；
- analysis Agent 必须是本轮 fresh read-only Agent；
- conditional evidence 给出六字段 composite key 或 `not_promoted`。

## 1. 执行背景与详细结论（首页之后）

用短段落回答：

- 目标模型、环境、workload、预算和用户成功阈值；
- S0 是否稳定；
- 最终保留、裁剪、挽救和未决范围；
- 采用 whole-scope 还是 range-optimized FINAL 晋级路径及其门禁；
- clean end-to-end 是否达到阈值；
- 推荐配置或明确 blocker。

不要把 fusion proven、child depth 或 profiler 局部改善写成端到端成功。

## 2. 环境与冻结控制

报告：

- device、CANN、Python、PyTorch、torch_npu、backend、static kernel、dtype、TP；
- source revision、命令、配置、prompt/input、batch/length、warmup、cache；
- baseline config、control、workload fingerprints；
- exact accepted option values 和 probe artifact；
- profiler/debug/SK child trace 在 clean timing 中均关闭。
- Stage-A S1/S2/S3/S4 的两个 option map 均显式为空，并链接 preflight
  `stage-a-option-validation.json`；缺省字段不得写成“等价为空”。

任何 detected-versus-declared mismatch 单独列出。

### 每个实验的失败现场

读取并执行 [failure-scene-reporting.md](failure-scene-reporting.md)。任何实验命令启动后的
非环境类失败，都必须先写入该实验自己的 `EXPERIMENT_REPORT.md` 或专用实验 report；
本最终 `REPORT.md` 只建立失败索引，不得用摘要替代实验本地现场。索引至少包含：

| Experiment/Round/Attempt | Failure phase | Local report | Scene artifacts | Fallback | Next diagnosis |
|---|---|---|---|---|---|

环境未加载、runtime import 失败、设备不支持或 option probe 未执行等实验启动前失败，写入
环境 artifact 和本节 blocker，不伪造实验 attempt。重试成功后仍保留先前失败行。

## 3. S0 稳定性

列出五个 clean process 的 TP worst-rank post-warmup mean、P50、P90、标准差、
绝对 spread、relative spread 和 `stable`。要求 `(max-min)/mean <= 5%`。

若失败，后续所有候选写 `not_run: unstable S0`，不得呈现伪比较。

## 4. Agent 与 Artifact Ownership

先报告 S 筛选 Agent 及其 clean timing artifact；这些行没有 analysis Agent、candidate
profile 或 per-SK response，必须写 `not_applicable: screening stage`。再报告父 Agent、
winner `child_agent_id`、每轮
`profiling_analysis_agent_ids` 和 `profiling_analysis_result` map。

| Experiment/Round | Experiment Agent | Analysis Agent | Request | Response JSON | 中文分析 | Candidate profile fingerprint |
|---|---|---|---|---|---|---|

所有路径相对于实验根。说明 nested dispatch 或 parent fallback。实验 Agent 不得与
analysis Agent 相同，candidate profile 和 response 不得跨轮复用。

逐轮列出 `baseline_profile`、`candidate_profile` 两个性能 manifest、session
fingerprint、producer PID/time/command
fingerprint 与 `validate-set` 结果。明确这些字段只存在于采集协议，未注入模型参数、
环境 graph attr、compiler option、graph node 或 runtime marker。candidate profile
必须列出该 profiling 进程自己的 profiler 三件套和完整 SK metadata；`sk_prof` 仅为
映射后调度诊断可选输入；compat metadata 替代 profile metadata 时本轮必须
blocked。

## 5. S 候选筛选与 Winner 选择

只报告 execution/correctness、至少三次 clean timing、mean/P50/P90/stddev、相对 S0
收益、fingerprints、eligibility 和排序：

| S candidate | Only change | Correctness | Clean runs | Mean gain | P90/stddev gate | Eligible | Rank | Result |
|---|---|---|---:|---:|---|---|---:|---|
| S1 | automatic AOT | ... | ... | ... | ... | ... | ... | ... |
| S2 | broad decode | ... | ... | ... | ... | ... | ... | ... |

表格必须覆盖冻结矩阵的全部 candidate，包括 `blocked/skipped`；未执行项填写 named
blocker、中文原因和 evidence 路径，不得从表中删除。单列
`screening-candidate-matrix.json` 的 schema、内容 SHA256、四类必需策略覆盖、
`settlement_complete`、planned IDs 与 executed IDs，并说明 executed IDs 与实际 clean
输入完全一致。另列 `stage_a_option_controls.all_explicitly_empty`、每个 archived
`run-*/config.yaml` 的哈希，以及两个 required empty fields。已有 eligible/winner 不能
作为任何未执行项的原因。

链接 `screening-performance-summary.json`，明确
`selection.selected_for_deep_analysis`。本节不得出现 P、candidate profiling、SK
mapping、source calibration 或 per-SK classification。筛选结果不是最终 promotion。

## 6. Winner 选项探索与优化矩阵

先报告 `Sbest-SEED` 和冻结的 `superkernel-winner-option-matrix-v1`。O 轮表格不得包含
per-SK profiling 结论：

| O trial | Category | Option/value | Compare to | Probe accepted | Correctness | Clean runs | Incremental gain | P90/stddev | Result | Artifact |
|---|---|---|---|---|---|---:|---:|---|---|---|

矩阵至少结算 DCCI family、`auto_op_parallel`、aggressive 和其他适用实验选项。普通项
逐项说明一个 RFC6901 Pointer 的 before/after、独立进程/timeout、结果、回退和当前
incumbent。DCCI 单独报告 disable-all 首轮；不得列出独立 before-only/after-only trial。
若 disable-all 劣化，必须增加 paired profile/manifest、全量逐 SK 对比、每个劣化 SK 的
child timing、最终 child regex 并集、两个相同 before/after exact list、联合修复相对 O0
的 clean 门禁和采纳/回退结论。不得以“profiling 没有明显证据”解释普通项 skipped。
accepted trial 必须补足五次并通过稳定性，全部结算后的 incumbent 才能写为
`Sbest-BASE`。

明确 O 阶段只证明选项组合的端到端增量收益。DCCI diagnostic profiling 可定位 SK/child
相关性并构造联合修复 list，但不能证明 cache 机制根因；筛选/O timing 均不得代替最终
promotion timing。

### 可选多流调优旁路

仅在 `Sbest-BASE` fresh profiling analysis 后报告本节；未触发时写 `not_invoked` 及原因。
触发时列出 request/result contract 校验状态、独立 worktree/config/artifact/cache、目标 SK
与三轴分类、每个 option 或 source-exact 单变量 trial、短窗口调度证据、局部 C/V overlap
变化和 clean 端到端变化。局部改善不能替代端到端门禁。

结果为 `no_gain/blocked/failed` 或无效时，明确写出精确 fallback incumbent 且正常主流程
未被阻塞。结果为 `accepted` 时，报告新 derived family 名称和 fresh `*-BASE` 状态；不得
把旁路 trial 写成旧 family 的 P/FINAL 或新增 M 轮。

### 可选 Source-Range 分支状态

`Sbest-P* -> Sbest-FINAL` 不是默认生命周期。报告必须将该分支标为
`not_requested`、`attempted` 或已请求后的 terminal outcome（`accepted`、`no_gain`、
`blocked`、`failed`、`invalid`）。`not_requested` 不是未完成项，也不阻塞
`whole_scope_clean_validation`。只有用户或冻结实验计划明确请求时，才填写 P/FINAL
round；被请求时仍逐项报告既有 `source_scope_map + exact + proven offsets` 门禁。

| Source-range branch | Request evidence | Current/terminal outcome | Incumbent | Whole-scope status |
|---|---|---|---|---|
| `not_requested` / `attempted` / terminal outcome | user request or frozen plan reference | ... | ... | ... |

| Experiment | Round chain | Only change | Compare to | Entry gate | Result | Blocker | Artifacts |
|---|---|---|---|---|---|---|---|
| Sbest | SEED -> O* -> BASE -> [optional isolated multistream -> derived-family BASE] -> [SMAP] -> whole_scope_clean_validation | 阶段 A 获胜脚本及已保留选项 | S0 | selected + option matrix + schema 2 | ... | ... | ... |

对 winner 说明 source diff、`declared_change_set`、生命周期和停止原因。

非 winner 候选不得出现在此矩阵。仅当默认 winner 门禁在 BASE profiling/analysis、
required SMAP settlement 或 whole_scope correctness/promotion 失败时，才说明回退到
阶段 A 固化排名中的哪个 next eligible candidate。
`absent/no_gain/blocked/failed/invalid` 可选 P 只终止该分支，恢复已保留的
BASE incumbent 并继续 whole_scope；不得触发 next eligible 候选回退。

## 7. Winner 每轮 Lifecycle

只为 winner 的 BASE 及明确请求的 P/FINAL profiling round 报告：

- candidate execution、correctness 与 profile-owned fresh metadata；
- 两个 profile manifest 的完整性、freshness 和 immutable validation；
- kernel projection proof、完整 step 状态、alternative solution count、
  mapping coverage 与 blocker counts；
- `deep_fusion_reproducible`、child count/depth histogram 作为结构描述；
- diagnostic profile path/fingerprint；
- analysis request/response 和 Agent ID；
- clean status，只有 range-optimized FINAL 或 unchanged whole-scope profiling 门禁通过后
  才可 `passed`。

`count_reliable=false` 或 profile 映射存在多个候选解的 SK 进入
`insufficient_evidence`，不能支撑动作。compat/verify/replay 审计结果如有，应单列，
不得改变性能映射或分类。

## 8. Per-SK Profiling 分析

先给统一表格，再补充：

- interval P90、duration sum P90、SK P90；
- baseline/SK occurrence count；
- graph occurrence fingerprint、source scope（如有）、boundary、ordered child sequence；
- stream IDs、baseline union、Cube/Vector overlap 和 SK child schedule；
- launch before/after/saved，仅作解释；
- diagnostic hypotheses、直接证据与 blockers。

人类表的每一行都要能定位到
`profiling-analysis-result.json` 中的同一 `range_id`。

## 9. BASE 决策

按 experiment 展示：

- beneficial -> keep；
- `source_scope_map + exact + proven offsets` 的 neutral/regressed -> P candidate；
- insufficient_evidence -> scope 不变，reprofile 或 block。

说明 child count 仅作描述，所有可靠 SK 都已 profiling。不得以 single-child 或
shallow 标签替代性能分类。

同时给出 child-count distribution（至少单列 1、2、3、4 和 `>=5`）、
`filtered_by_child_count=0`、mapping coverage、ambiguous/unmapped/diagnostic-only
blocker counts。raw Task/model/stream/node ID、名称或局部窗口不得报告为 exact fallback。

## 10. P 裁剪验证

每个 P 轮报告：

- 被移出 marker 的 exact `range_id` 和
  `source_scope_map + exact + source_file/start_offset/end_offset`；
- exclusion method：`explicit_none_scope` 或 `outer_scope_split`；前者必须列出
  `scope_capabilities.explicit_none_exclusion.accepted=true` 的环境证据，并明确使用的是
  Python `None` 而非字符串 `"None"`；
- 来源 classification、analysis Agent、source analysis result；
- batch ranges 是否互不重叠；
- 算子执行、依赖、event、barrier、通信、cache 和多流保持证明；
- fresh metadata 对“目标 unit 未融合、周围命名 scope 保持”的验证；
- P 轮 fresh profiling 结果；
- `verified_in_round_id`，必须是严格后续且明确引用该范围的 P 轮。

不能可靠映射或 insufficient 的范围不得出现在 prune 列表。

## 11. FINAL 交互检查

列出 `retained_range_ids`，并为每个最终 range 给出
FINAL 当前数据的统一性能表行。要求所有范围都有
`source_scope_map + exact + proven offsets`，且为 `beneficial/keep`，没有
conflicting regressed/blocked decision。

说明 FINAL candidate profile、analysis Agent、analysis result 都是 fresh。若组合
后任一范围失去收益，FINAL 失败并结束该分支，不进入 clean timing。

所有 winner 在本节先报告 BASE profiling 后的 source-action mapping：是否生成
`Sbest-SMAP`、stable_source/stable_marker 路径、目标 SK/range 数、exact/partial/skipped
计数和每个 skip 原因。本节逐 range 可动作门禁仅适用于执行过源码裁剪的
range-optimized FINAL。若采用 whole-scope 晋级，仍须证明 source mapping 已结算，再
列出完整 inventory 覆盖、performance-exact 映射数、`insufficient_evidence=0`、candidate
identity 未变化证明和局部分类计数；不得因已结算但缺少源码 offsets 或存在局部
neutral/regressed 就写成特性失败。

## 13. Automatic AOT Winner

automatic AOT 若不是 winner，只在第 5 节保留 clean timing，不生成本节。若它成为
winner，报告每个 reliable SK 的 profiling classification，并独立报告是否满足 unchanged
whole-scope clean 晋级门禁：

- beneficial、neutral、regressed 在 AUTO-BASE 均为 proposed 性能证据；
- insufficient 保留 unresolved；
- 任一 performance-exact SK 缺 source exact 时，报告一次批量 winner-only
  `Sbest-SMAP` 的状态、目标 range/SK identity、可用 fingerprint 集合、
  stable_source/marker-only bridge 和 exact/partial/skipped 计数；
- analyzer 得到 source exact 后，若 P/FINAL 已明确请求则报告 AUTO-P/FINAL；未 exact 的 SK 跳过。

明确写出 automatic AOT 的 `kernel_projection_structural + exact_projected_trace` 只证明性能
occurrence，未直接晋级 verified prune/diagnostic。只有独立 SMAP 的
`stable_source + marker_only_calibration + source_scope_map exact` 链路才能补出原始 AUTO
源码 offsets；不得从 graph occurrence、层号或 signature 猜测。

## 14. Final Clean End-To-End

只列 winner 深度分析/优化完成后新采集的 clean process，不复用筛选样本作为最终证据。
至少三个 whole-scope 或 FINAL candidate 对比冻结五次 S0，统一 warmup 和 TP
worst-rank 聚合。报告：

- mean、P50、P90、standard deviation；
- candidate 每次相对 S0 的方向；
- absolute/relative improvement；
- 用户阈值是否达到；
- profiler、metadata、trace、debug 已关闭的证据。

端到端是最终主判据；per-SK interval 是范围取舍和原因诊断证据。

## 15. Layer 与 Pipeline

报告每层 ordered operators、SK inventory、child count histogram、fragmentation、
break reasons、streams、Cube/Vector/MIX、overlap 和 control-core。Depth 保留为结构
观察，不能自动决定 keep/prune/profiling。

若 baseline 多流重叠而 SK child trace 串行，说明是否执行过一个
`auto_op_parallel` diagnostic；若没有，写 `unproven` 及缺失证据。

## 16. Failure Scene 与 Failure Isolation

对所有已启动实验的非环境类失败，先链接该实验 report 中按
`failure-scene-reporting.md` 写入的完整现场。这里汇总 terminal status、fallback 和后续
入口。对 crash/hang/timeout/correctness failure，另外报告：

- fresh plog、metadata、source log；
- 失败 SK 与具体 child identity；
- confidence 和备选解释；
- 精确排除范围及依赖保持方式，或 blocked；
- failure-isolation sibling 的结果路径。

不得把运行失败混入性能 prune。

## 17. Schema 2 与条件证据

报告 ledger validation/merge 结果。每条活动条件证据列出：

```text
source_revision
baseline_config_fingerprint
candidate_config_fingerprint
control_fingerprint
workload_fingerprint
range_id
```

同时列出 baseline/candidate profile path+fingerprint、source/verification analysis
result、declared change 和状态。只接受 profiling analysis schema 1.2。

## 18. 最终结论

用中文给出：

1. 推荐配置、scope 和 source revision，或不推荐的明确结论；
2. 端到端收益及置信度；
3. 保留/裁剪/挽救/未决范围；
4. 仍需的单范围源码恢复/细化实验；
5. 预算停止点和不可外推条件；
6. 下一 Agent 可使用的 exact conditional fingerprints。

## 19. Artifact Index

至少链接：

- environment probe、S0 stability、S0 diagnostic profile；
- 每轮 baseline/candidate config、共同 association config、workload/change manifest
  和两个 profile collection manifests；
- 可选的 compat/verify metadata 与 generic replay 独立审计；
- candidate profile 和 fingerprint；
- analysis request、JSON response、中文 Markdown；
- round evidence/report、experiment result、schema 2 ledger；
- FINAL clean logs 与 performance result；
- source diff、failure evidence 和最终 REPORT。
