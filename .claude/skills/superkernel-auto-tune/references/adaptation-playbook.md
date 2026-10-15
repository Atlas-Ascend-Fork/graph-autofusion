# SuperKernel Adaptation Playbook

本 playbook 把每个候选组织成可审计实验，而不是把“融合更深”当作目标。最终目标是
冻结控制下可重复的端到端收益。

## 1. Freeze S0

记录 source revision、环境、命令、YAML、输入、warmup、precision、TP、device、
cache 和 debug/profiler 状态。运行五个 clean SK-off process：

```bash
python3 <runtime-skill-dir>/scripts/analyze_performance.py \
  --baseline experiments/S0 --warmup 8 \
  --json-out experiments/S0/baseline-stability.json
```

要求 TP worst-rank post-warmup mean 的 `(max-min)/mean <= 5%`。任一性能敏感
fingerprint 变化都要求重新建立 S0。S0 不稳定时停止后续工作。

未修改 scope 的 whole-scope 候选通过完整 profiling 门禁后，clean timing 使用
`--promotion-mode NAME=whole_scope`，不传 `--replay-evidence`。细粒度
`range_optimized` 候选保持默认模式，仍必须提供同候选 replay evidence。该模式只区分
源码动作门禁，不放宽 S0 稳定性、三次候选运行、均值、P90 或方差门禁。

另采一份 SK-off diagnostic profile，用于层级重建和 source interval 比较，不计入
clean baseline。

## 2. Build The Main Strategy Matrix

从源码和 S0 profile 建立策略矩阵，每个主实验只改变一个因素：

| ID | 策略 | 唯一变化 | 主要问题 |
|---|---|---|---|
| S1 | automatic AOT | SK on，无 marker | 自动范围产生了哪些实测收益 |
| S2 | broad decode | 一个宽 decode scope | 宽范围是否破坏依赖或并行 |
| S3 | per-block | 每层/块 deterministic scope | 层间一致性与分裂 |
| S4 | segment | attention/MLP/MoE 等依赖段 | 哪些局部范围有收益 |
| S4+ | composite | 合并相邻安全段 | 交互后是否仍有收益 |
| O only | DCCI family | disable-all 首轮；劣化后仅一个 before+after 联合修复 | 相对无 DCCI O0 有收益才保留 |
| O only | scheduling/risky | winner 的一个 accepted option | profile 前端到端探索，收益才保留 |
| P only | source source-range adjustment | 一个已裁剪 range 的源码恢复或边界细化 | P 后 fresh analysis 的源码假设；Option 保持冻结 |

表中记录 readable name、test point、only change、compare-to、expected evidence、
entry/next gate、artifact 和 blocker。不要按 scope 宽度盲搜。`O only` 行不参加 S
筛选，在 winner 选出后、profiling 前按
[winner-option-sweep.md](winner-option-sweep.md) 实测；不要求 profiling 先提供直接证据。
`P only` 仍只在 P 后由局部 profiling 直接假设触发。

在第一个 S candidate process 启动前，冻结
`superkernel-screening-matrix-v2`。至少包含以下 `strategy_kind`：

```text
automatic_aot
broad_decode
per_block
semantic_segment
```

候选项必须包含 `id`、`strategy_kind`、`only_change` 和最终 `status`。selection 前
`status` 只能是 `executed`、`blocked` 或 `skipped`；不得保留 `planned`、`pending`、
`deferred`。`blocked/skipped` 必须包含 `blocker.code`、中文 `reason_zh` 和相对
`evidence` 路径。缺少合法语义边界、用户不允许源码变化、环境不支持或经批准的预算限制
可以形成 blocker；“已有 eligible candidate/winner”不能形成 blocker。

## 3. Stage A: Screen Main Strategies

父 Agent 读取
[experiment-agent-orchestration.md](experiment-agent-orchestration.md)，为每个主
实验派 fresh child_agent_id。主实验默认串行。S1/S2/S3/S4/... 在本阶段只执行完整
推理、正确性和至少三个独立 clean timing process。关闭 profiler、metadata dump、
`sk_prof`、trace、debug sync 和 calibration。

每个 S config 必须在 `model_config.custom_params` 中显式写出：

```yaml
super_kernel_optimize_options: {}
super_kernel_debug_options: {}
```

缺省字段不能解释为空，因为 active wrapper 或模型 glue 可能注入默认 option。第一个 S
进程启动前，对冻结的 S1/S2/S3/S4 配置统一执行：

```bash
python3 <skill-dir>/scripts/validate_stage_a_options.py \
  path/to/S1.yaml path/to/S2.yaml path/to/S3.yaml path/to/S4.yaml \
  --json-out experiments/stage-a-option-validation.json
```

任一字段缺失、非 mapping 或非空时停止 Stage A，不允许通过代码默认值补齐。每次 clean
run 把实际输入归档为 `run-*/config.yaml`，selection CLI 会再次结构化校验。DCCI、
`auto_op_parallel`、aggressive 或其他 option 只能从 winner 的 Stage O 开始。
同时审查冻结的 `npugraph_ex` compile glue：两个显式空 map 必须原样控制有效选项，禁止
硬编码 option 或在字段缺省时注入非空 fallback。无法用 config 与 glue revision/hash
共同证明有效 option 为空时，不得启动 S1。

禁止在本阶段为任何 S 候选执行 per-SK profiling/mapping，禁止生成 P/FINAL，禁止
插入标定 marker。所有正确候选完成后统一运行：

```bash
python3 <runtime-skill-dir>/scripts/analyze_performance.py \
  --baseline experiments/S0 \
  --candidate S1=experiments/S1/CLEAN \
  --candidate S2=experiments/S2/CLEAN \
  --candidate S3=experiments/S3/CLEAN \
  --candidate S4=experiments/S4/CLEAN \
  --screening-matrix experiments/screening-candidate-matrix.json \
  --selection-only --warmup 8 \
  --json-out experiments/screening-performance-summary.json
```

`selected_for_deep_analysis` 是唯一 winner；冻结为 `Sbest-SEED`，它不是最终
promotion。无 eligible candidate
时保留 S0。仅当默认 winner 门禁在 BASE profiling/analysis、required SMAP
settlement 或 whole_scope correctness/promotion 失败时，才按
`ranked_eligible_candidates` 顺序逐个回退，不为所有候选批量补 profiling。
`absent/no_gain/blocked/failed/invalid` 可选 P 只终止该分支，恢复已保留的
BASE incumbent 并继续 whole_scope；不得触发 next eligible 候选回退。

`--selection-only` 会校验四类必需策略均出现、所有候选已经结算、矩阵中
`status=executed` 的 ID 与命令行 `--candidate` 集合完全一致，并把矩阵内容 SHA256 写入
筛选结果；还会校验每个 clean run 的两个 option map 显式为空，并在结果写入
`stage_a_option_controls`。任一漏项、额外项、非法 early-stop blocker、缺省 option map
或非空 option 都必须在读取性能日志前失败。

## 4. Stage O: Sweep Winner Options

读取 [winner-option-sweep.md](winner-option-sweep.md)。在 O1 前冻结覆盖 DCCI、
`auto_op_parallel`、`aggressive_opt_strategies` 及其他适用实验选项的矩阵。缺少
profiling 直接证据不能跳过 O 项；exact value 必须由同一推理环境的 ready probe 接受。

先从 `Sbest-SEED` 建立五次稳定的 `O0-INCUMBENT`。普通 O 轮只改一个 option/value，
执行 correctness 和至少三次 clean timing，调用 `analyze_performance.py --option-trial`
与当前 incumbent 增量比较。DCCI 按 [dcci-option-tuning.md](dcci-option-tuning.md)：只先试
`dcci_disable_on_kernel=[".*"]`；有收益直接采纳，neutral/no-gain 直接拒绝。只有实测
劣化时才采集 O0/disable-all paired profiler、SK meta 与完整 `sk_prof`，全量定位劣化 SK
及其显著劣化 child，再把 child regex 并集同时设置到 before 和 after，保留 disable-all，
运行唯一一次联合修复。该组合必须直接优于 O0 才能采纳。普通 trial 和 DCCI 联合修复均
需达到 mean、median direction、P90、stddev 门禁；accepted trial 补足五次并重验稳定性
后成为下一 incumbent。未收益、失败或不正确立即回退。所有矩阵项结算后，最终 incumbent
冻结为 `Sbest-BASE`。

普通 O clean 轮关闭 profiler、metadata、`sk_prof`、trace、debug sync 和 calibration。
DCCI disable-all 劣化后的 paired diagnostic collection 是唯一例外；它不计 clean timing，
不执行 source mapping，也不直接决定采纳。冒险选项使用独立进程和 timeout；失败按
failure-isolation 留证，但不得污染 incumbent。

## 5. Stage B: Dispatch The Winner Family

完成 O 矩阵后的 `Sbest-BASE` 默认进入 whole-scope promotion 路径：

`Sbest-SEED -> O0/O* -> Sbest-BASE -> [optional isolated multistream branch -> derived-family BASE] -> [Sbest-SMAP] -> whole_scope_clean_validation`

`Sbest-P* -> Sbest-FINAL` 是独立可选源码范围分支，仅在用户或冻结实验计划
明确请求时启动；不得因 BASE 或 SMAP 结束而自动创建。请求的分支为 absent、invalid、
`no_gain`、`blocked` 或 `failed` 时精确保留 incumbent，且不阻塞 unchanged
`whole_scope_clean_validation`。请求后仍适用全部 source-action gate。

BASE profiling 后，所有 winner（AUTO 与命名/manual）都必须结算一次 SK 到可编辑源码
半开区间的映射。任一 performance-exact SK 尚无 source exact 时，先生成一个批量
`Sbest-SMAP`，暂不生成可选 P 或 `whole_scope_clean_validation`。SMAP 结算后，exact 的
range 仅在 P 已明确请求时可进入 P；partial/skipped range 只记录并跳过源码动作。此后 BASE 完整性能
映射、无 analyzer blocker、无 `insufficient_evidence` 时，才生成不改源码的
`whole_scope_clean_validation`。局部 neutral/regressed 或已结算的源码映射跳过项不否决
整 scope 端到端晋级。

同一个 winner child 管理这个实验族的执行状态；每个 profiling round 则由不同 fresh
read-only analysis Agent 分类。nested dispatch 优先；无法 nested spawn 时生成
analysis request 交父 Agent。实验 child 不得代替 sibling skill 做性能分类。

`Sbest-BASE` 的 fresh schema 1.2 分析若显示多流并行退化或可动作的调度优化机会，父
Agent 可选派发 sibling `superkernel-multistream-performance-tuning`。该 child 只能在专用
worktree/config/artifact/cache 中尝试，并用独立 request/result contract 交付。option-only
不要求 source map；scope split/range exclusion 必须有 `source_scope_map_v2 + exact`。
`no_gain/blocked/failed` 或无效结果精确保留 incumbent，不阻塞 whole-scope/P。仅
`accepted` 结果可建立新派生 family，并从 fresh 普通 `*-BASE` 重跑；不得向旧 schema 2
family 插入 M 轮，也不得用局部 C/V overlap 改善替代 clean 端到端增量门禁。

## 6. Execute Every Winner Profiling Round

每轮执行：

1. 在每个 profiling run 前调用 collection `begin`，分别建立 fresh immutable
   `baseline_profile` 与 `candidate_profile` role root。
2. 运行完整 warmup、prefill、decode 和 correctness。
3. candidate profile 进程把 profiler 的 `kernel_details/task_time/trace_view` 和该进程
   自己的 origin/updated graph、fused/scope/super-kernel metadata 全部归档到 profile
   root；compat metadata 不能代替它。`sk_prof` 仅按需作为映射后的调度诊断输入。
4. run 完成并归档后调用 collection `finalize`，再对两个 profile manifest 执行
   `validate-set`。session fingerprint、PID/times/revision/command fingerprint 只写
   采集协议，绝不注入模型参数、环境 graph attr、编译选项、图节点或 runtime marker。
5. 派 fresh read-only analysis Agent 调用 sibling
   `superkernel-fusion-performance-analysis`。
6. 将 response 与 request、Agent ID、profile fingerprint、declared change 交叉
   校验后，才决定下一轮。

`child_count`、single-child ratio、depth histogram 和 launch reduction 都是结构
描述。无论 child count 为多少，SK 都进入 inventory 和 profiling 分析；尤其不得
过滤 child count 1 至 4。

compat/verify 与 generic replay 可作为独立候选复现审计：

```bash
python3 <runtime-skill-dir>/scripts/analyze_sk_meta.py \
  experiments/S3/S3-BASE/compat/sk_meta \
  --verify-root experiments/S3/S3-BASE/verify/sk_meta \
  --candidate-name S3 \
  --round-id S3-BASE \
  --source-revision REVISION \
  --compat-config-manifest experiments/S3/S3-BASE/compat/config.json \
  --verify-config-manifest experiments/S3/S3-BASE/verify/config.json \
  --replay-min-child-nodes 1 \
  --replay-report-out experiments/S3/S3-BASE/fusion-replay.json
```

该审计不传给性能 analyzer。`fusion_reproducible`、重复 signature、matched/unmatched
identity 和 Deep 指标都不能决定 profile-vs-baseline 映射、classification 或晋级。

## 7. Handle Runtime Or Correctness Failure

筛选候选发生 error、timeout、hang 或 correctness failure 时，记录失败并继续其他 S，
不进入 profiling。winner 发生同类失败时读取
[superkernel-sk-failure-isolation](../../superkernel-sk-failure-isolation/SKILL.md)，
用 fresh plog、sk_meta 和源码日志定位具体失败子算子。只有高/中可信直接证据可形成
精确排除范围；不得删除算子执行，也不得凭邻近位置排除。

修复后的轮次使用新的 process、profile-owned metadata 和 artifacts。

## 8. BASE Classification

Fresh analysis response 对每个范围给出：

- `beneficial/keep`：保留；
- `source_scope_map + exact + proven source offsets` 的 `neutral/prune`：加入
  performance prune；
- 同等 source-map 门禁下的 `regressed/prune`：加入 performance prune 并保留诊断；
- `insufficient_evidence/reprofile|block`：scope 不变。

以 interval P50 相对 SK P50 为主，duration sum P50 为辅助；MAD threshold 控制
噪声。实验 child 不得用 child count 或 launch saved 覆盖 classification。

源码标定按 SK group 处理。完整业务图仍必须唯一投影；每个 SK 的全部 child 必须在至少
三个 step 中 exact-assigned。无关的未归属图节点记录为 skipped，不阻塞其他 SK；SK 自身
child 未归属或跨 block instance 时只跳过该 SK，保持 `diagnostic_only`，不得产生 prune。

## 9. Performance Prune

P 轮只改变 SK marker 边界，移出 exact neutral/regressed ranges。原算子继续按原
顺序执行，依赖、event、barrier、通信、cache mutation 和多流语义不变。

优先使用显式非融合子作用域，且 begin/end 必须成对：

```python
torch.npu.super_kernel_scope_begin("decode")
# retained SK work
torch.npu.super_kernel_scope_begin(None)
target = exact_source_unit(...)
torch.npu.super_kernel_scope_end(None)
# retained SK work
torch.npu.super_kernel_scope_end("decode")
```

这里必须传 Python `None`，不能传字符串 `"None"`。使用前要求同推理环境的
`check_environment.py --json` 报告
`scope_capabilities.explicit_none_exclusion.accepted=true`。该字段只证明当前 wrapper 接受
成对调用；P 轮仍需用 fresh `sk_scope_split/sk_fused_nodes` 证明目标 unit 没有进入融合，
外层命名 scope 和其他 child 未被意外改变。

如果 active wrapper 不接受 `None` pair，回退到在目标 unit 前
`super_kernel_scope_end(outer_name)`、unit 后
`super_kernel_scope_begin(outer_name)`。同一 P 轮只能采用一种排除方法，并在
declared change 中记录 `explicit_none_scope` 或 `outer_scope_split`；比较两种方法时必须
拆成独立轮次。接口探测失败、pair 不平衡或 fresh metadata 无法证明排除效果时，本轮
fail-closed，不得把 API 可调用或源码缩进当成移出成功。

可在一个 P 轮批量移出多个范围，但必须：

- 每个范围必须由 producer 给出 `mapping_method=source_scope_map`、
  `mapping_confidence=exact` 以及 proven source offsets；
- 每对范围都有可机器比较的 `source_file/start_offset/end_offset`，且 half-open
  source interval 互不重叠；所有范围必须位于同一 `source_file`，相邻区间可批量，
  跨文件、交叠、包含、同边界不可批量；
- 只有字符串 op 名的 boundary 不能证明源码区间，默认一 range 一 P；
- 每个 `range_id` 可独立审计；
- `declared_change_set` 精确列出允许变更的 marker Pointer 和唯一 exclusion method；
- 本轮 fresh profiling 分析其余所有可靠 SK。

裁剪结论只有在严格后续 P 轮明确引用被移出范围并通过全生命周期后，才能标记
`verified`。失败 P 轮保留 blocker，并终止该手工链，不能跳到下一轮。
只要当前 fresh analysis 仍有 prune action，就只生成下一 P，不得同时生成 FINAL。

 ## 11. FINAL Interaction Check And Whole-Scope Promotion

FINAL 组合：

- BASE/P 后仍保持 beneficial 的 retained ranges；

FINAL 只物化上述 source actions；Option map 必须与 `Sbest-BASE` 完全一致。

组合后重新执行 correctness、fresh profile 和 fresh analysis。每个最终 range 必须有
`source_scope_map + exact + proven source offsets`，
并在 FINAL 当前数据中得到 `beneficial/keep`，且没有 conflicting
blocked/regressed decision。

上述逐 range 可动作门禁只适用于发生源码裁剪的 range-optimized FINAL。未修改的
整 scope 候选无需让每个局部 range 都取得源码 offsets 或 beneficial；但必须已经执行并
结算 winner source mapping，逐项报告 exact/partial/skipped，并证明完整 SK inventory
均为 performance-exact、无 `insufficient_evidence`，保持 candidate identity 不变。

FINAL 或 `whole_scope_clean_validation` 通过各自 profiling 门禁后，关闭所有诊断开关
运行至少三个 clean candidate processes。最终性能用 frozen S0 五次结果比较，要求
均值达到阈值且 P90/标准差不劣化；禁止把独立 range 的改善简单求和。

## 12. Automatic AOT Candidate

automatic AOT 在阶段 A 只是一个 clean S 候选，不在原地创建 P，也不做 per-SK
profiling。只有它成为 winner 后，才冻结为 `Sbest-SEED`，完成普通 O 选项矩阵并生成
`Sbest-BASE`，再对每个 reliable SK profiling：

- 已分类 graph occurrence -> proposed 性能证据；
- 任一 performance-exact SK 无 source exact -> 按 range/SK identity 与可用 fingerprint
  集合创建一次批量 `Sbest-SMAP`；
- insufficient -> unresolved，不改 scope。

SMAP 冻结原始无 marker AUTO 源码，独立标定 worktree 仅插入临时语义 marker，并以
`marker_only_calibration_v1` 证明删除插入 byte span 后逐字节还原生产源码。再完成唯一
original/calibration 图投影、三步 assignment 和 `C_sk == C_unit`。analyzer 验证
`source_scope_map + exact + proven source offsets` 后，AUTO winner 在 P/FINAL 已明确请求时可进入该分支；
否则只跳过未 exact 的 SK。初始 S1-AUTO 分析不直接晋级；SMAP source exact 后，只有
严格更晚 P 的 fresh verification 才能把 AUTO 源 decision 标为 verified。

## 13. Schema 2 Result

阶段 A 只输出独立的 `screening-performance-summary.json`，不得把筛选 timing 伪装为
schema 2 conditional evidence。阶段 B 的 winner optimizer 结果包含：

- `child_agent_id`；
- `profiling_analysis_agent_ids` 和 `profiling_analysis_result` round maps；
- 每轮 baseline/candidate profile path 与 content fingerprint；
- source、baseline config、candidate config、control、workload fingerprints；
- `declared_change_set` 和 artifact role；
- `performance_scope_decisions`、`verified_ranges`、
  `unresolved_performance_ranges`；
- lifecycle、blockers、relative artifacts、中文 `next_agent_guidance_zh`。

运行：

```bash
python3 <runtime-skill-dir>/scripts/experiment_ledger.py validate \
  experiments/S3/experiment-result.json
python3 <runtime-skill-dir>/scripts/experiment_ledger.py merge \
  --ledger experiments/experiment-ledger.json \
  --result experiments/S3/experiment-result.json \
  --output experiments/experiment-ledger.json
```

Legacy schema 1 记录只读迁移，不晋级到 conditional performance evidence。活动
证据只在六字段 composite fingerprint 完全匹配时复用。
CLI 以 `experiment-result.json` 所在目录为 artifact root，逐轮读取并只读重放
`profiling_analysis_result`，交叉校验 analyzer 的 per-SK/scope decision。纯对象 API
没有 artifact root 时只做结构校验，不产生 conditional evidence。

## 14. Stop Conditions

已明确请求的 P/FINAL 轮执行、正确性、profiling、证据或交互门禁失败时，
只终止该可选源码范围分支：保存完整失败证据，恢复分支前冻结的 BASE
incumbent，并在原有门禁通过时继续不改变的 `whole_scope_clean_validation`。
不得仅因可选 P/FINAL 分支失败而转到下一个 eligible 候选。

对默认 winner 路径或当前 incumbent（不含上述可选 P/FINAL 分支），
停止某实验族并保留完整负面结果，当：

- execution/correctness/generic replay 失败；
- 可靠 source identity 无法建立；
- profiling artifacts 或 fingerprints 不可信；
- fresh analysis 返回 blocker 且预算内无法补证；
- whole-scope profiling 存在非精确映射或 `insufficient_evidence`；
- whole-scope clean end-to-end 不达用户阈值；
- 实验预算耗尽。

winner 的默认深度门禁失败不代表适配完成。只有当 BASE/SMAP/
whole-scope 等默认路径无法继续时，父 Agent 才按阶段 A 固化的 clean 排名
逐个尝试下一 eligible 候选；不得回到“每个 S 都 profiling”的旧流程。

## 15. Reporting

阶段 A 中文报告只列 correctness、clean timing、fingerprints、eligibility、排序和 winner。
Stage O 列出冻结选项矩阵、accepted-value、每轮唯一配置 diff、correctness、增量 clean
结果、保留/回退和最终 `Sbest-BASE`；不得虚构 per-SK 因果。阶段 B 每个 winner
profiling round 才列 interval P50、duration sum
P50、SK P50、MAD threshold、classification/action、mapping confidence、analysis Agent、
conditional evidence、fingerprints、artifact 和 blocker。最终报告区分筛选 timing、
diagnostic profiling 与 fresh final clean timing，并覆盖所有成功、失败、裁剪、挽救、
未决和跳过的范围。
