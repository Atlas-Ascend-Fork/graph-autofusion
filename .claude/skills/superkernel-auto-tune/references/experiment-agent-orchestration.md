# 实验子 Agent 编排与交接协议

本协议用于长周期 SuperKernel 适配，目的是把执行上下文、只读性能分析和父
Agent 的跨实验决策分开。所有人类可读交接、分析和总结均使用中文。

## 1. 三类 Agent 的职责

### 父 Agent

S0 稳定性通过后，父 Agent：

1. 冻结 source、baseline config、control、workload 和 baseline profile
   fingerprint。
2. 为每个 S 候选派一个 fresh 筛选子 Agent，默认串行，避免共享源码、设备和
   编译缓存相互污染。
3. 汇总所有正确候选的 clean timing，以 `--selection-only` 固化排序和唯一 winner。
4. 仅为 winner 建立 `schema_version: 2` 的 optimizer ledger 并进入 profiling。
5. 合并前校验 winner artifact 角色、Agent ID、profile fingerprint 和条件证据。
6. 仅当默认 winner 门禁在 BASE profiling/analysis、required SMAP settlement
   或 whole_scope correctness/promotion 失败时，才按固化的
   `ranked_eligible_candidates` 逐个尝试，不并行 profiling。
   `absent/no_gain/blocked/failed/invalid` 可选 P 只终止该分支，恢复已保留的
   BASE incumbent 并继续 whole_scope；不得触发 next eligible 候选回退。
7. 所有阶段完成后，用中文输出跨实验结论。

父 Agent 维护：

```text
experiments/PARENT_CONTEXT.md
experiments/experiment-ledger.json
experiments/S3/experiment-result.json
experiments/S3/EXPERIMENT_REPORT.md
```

不要把安装目录、当前工作目录或任何绝对路径写入协议。所有路径均相对于实验
artifact 根目录。

### 主实验子 Agent

筛选阶段的主实验子 Agent 只负责一个 S 候选。它只可执行 candidate、correctness
和至少三个 clean timing process；不得采集 candidate profiler、执行 SK 映射/源码
标定，或创建 P/FINAL。

winner 选定后，该实验子 Agent 才可以：

- 修改该实验声明允许的 scope marker 或 option；普通 O/ trial 仍遵守其单变量规则，
  Stage O DCCI 联合修复仅按专用协议同时设置 before/after；
- 执行 candidate、correctness 和 diagnostic profiling；
- 生成不可变 profile、profile-owned metadata、request 和 round evidence；
- 按需生成独立的 compat/verify/replay 可复现性审计；
- 仅在用户或冻结实验计划明确请求可选源码范围分支时，根据独立分析 response 执行
  P/FINAL 下一轮。

它不得自行读取 profiler 行后给出 `beneficial`、`neutral`、`regressed` 或
`insufficient_evidence` 分类，也不得改写分析结果。

### Profiling 分析子 Agent

只有 winner 的 BASE 及明确请求的 P/FINAL profiling round 使用不同的 fresh read-only profiling analysis child agent。
非 winner S 候选不得派该 Agent。该 Agent：

- 读取 sibling
  [superkernel-fusion-performance-analysis](../../superkernel-fusion-performance-analysis/SKILL.md)；
- 只读取 request 指定的不可变 artifacts；
- 只写专属 `profiling-analysis-result.json` 与中文
  `PROFILING_ANALYSIS.md`；
- 不运行推理、不编辑源码、不调整 scope、不修改 option、不执行 clean timing。

## 2. Nested Dispatch 与回退

nested dispatch 优先。主实验子 Agent 在每轮 profiling 完成后直接派 fresh
analysis Agent，并等待 response 后再决定下一轮。

若主实验子 Agent 不能继续 spawn：

1. 它生成 analysis request，写入 `Sx/<round>/profiling-analysis/request.json` 和中文
   `ANALYSIS_REQUEST.md`。
2. 它向父 Agent 返回 `analysis_pending`，不得代算分类。
3. 父 Agent 派 fresh read-only analysis Agent。
4. 分析 Agent 写入 `Sx/<round>/profiling-analysis/profiling-analysis-result.json` 和
   `Sx/<round>/profiling-analysis/PROFILING_ANALYSIS.md`。
5. 父 Agent 将 response 路径和 analysis Agent ID 回填给原实验子 Agent；原
   Agent 仅在该可选源码范围分支已被明确请求时再继续 P/FINAL。

Request 至少包含：

```json
{
  "experiment_id": "S3",
  "round_id": "S3-BASE",
  "child_agent_id": "experiment-agent-S3",
  "requested_analysis_agent_id": "analysis-agent-S3-BASE",
  "source_revision": "REVISION",
  "baseline_config_fingerprint": "sha256:...",
  "candidate_config_fingerprint": "sha256:...",
  "control_fingerprint": "sha256:...",
  "workload_fingerprint": "sha256:...",
  "baseline_profile": "S0/profile/profiler/kernel_details.csv",
  "baseline_profile_fingerprint": "sha256:...",
  "candidate_profile": "S3/S3-BASE/profile/profiler/kernel_details.csv",
  "candidate_profile_fingerprint": "sha256:...",
  "metadata": "S3/S3-BASE/profile/sk_meta",
  "collection_manifests": {
    "baseline_profile": "S0/profile/association-artifact-manifest.json",
    "candidate_profile": "S3/S3-BASE/profile/association-artifact-manifest.json"
  },
  "declared_change_set": {
    "allowed_json_pointers": ["/superkernel/scope"],
    "only_change_zh": "只修改本轮声明的 SuperKernel 范围。"
  },
  "json_response": "S3/S3-BASE/profiling-analysis/profiling-analysis-result.json",
  "markdown_response": "S3/S3-BASE/profiling-analysis/PROFILING_ANALYSIS.md"
}
```

Response 的 `analysis_agent_id`、round、五个实验 fingerprint、两个 profile
路径及内容 fingerprint、`declared_change_set` 必须与 request 完全一致。

## 3. 两角色性能 Artifact Contract 采集协议

每轮性能结构关联只交付两个不可变 role root：

| role | unchanged run 必需归档 |
|---|---|
| `baseline_profile` | SK-off `kernel_details`、`task_time`、`trace_view` |
| `candidate_profile` | profiler 三件套，以及该 profiling 进程自己的全部 SK metadata；`sk_prof` 仅为可选的映射后调度诊断输入 |

每次 run 之前先对空目录执行 `begin`。两个 session/root/producer identity 必须彼此
独立；candidate profile 必须归档其自身 metadata，不得从 compat/verify 复制。

在采集前冻结一份 `/abs/experiment/association-config.json`，只写两侧共同的结构关联
不变量，例如 graph target、model role 和 collection scope strategy。不要写 SK enabled、
具体 scope、profiling 开关或只属于 S0/S1 某一侧的执行选项。两次 `finalize --config`
必须使用这同一份文件；实际 S0/S1 执行 config 仍分别保留，并由
baseline/candidate config fingerprint 与 declared-change 门禁校验。
同理，两份 manifest 的 `--control` 指向同一份 collection/runtime control；analyzer
从 baseline/candidate config 删除 declared pointers 后计算的 control fingerprint 属于
另一验证域，两者分别报告，禁止直接比较。

```bash
python3 superkernel-fusion-performance-analysis/scripts/artifact_contract.py begin \
  --role baseline_profile \
  --root /abs/experiment/S0/profile \
  --session-out /abs/experiment/S0/profile.capture-session.json

python3 superkernel-fusion-performance-analysis/scripts/artifact_contract.py begin \
  --role candidate_profile \
  --root /abs/experiment/S1-AUTO/profile \
  --session-out /abs/experiment/S1-AUTO/profile.capture-session.json
```

随后由既有 orchestrator 原样启动对应模型命令，记录 native PID、开始/结束时间，
并在 finalize 前把该进程 artifacts 归档到刚创建的 role root。不要为关联功能修改模型
命令。各角色归档完成后分别执行：

```bash
python3 superkernel-fusion-performance-analysis/scripts/artifact_contract.py finalize \
  --session /abs/experiment/S0/profile.capture-session.json \
  --source-revision REVISION \
  --native-pid BASE_PID --started-ns BASE_START_NS --ended-ns BASE_END_NS \
  --command-fingerprint BASE_COMMAND_SHA256 \
  --workload /abs/experiment/workload.json \
  --config /abs/experiment/association-config.json \
  --control /abs/experiment/S0/control.json \
  --file kernel_details=profiler/kernel_details.csv \
  --file task_time=profiler/task_time.csv \
  --file trace_view=profiler/trace_view.json \
  --out /abs/experiment/S0/profile/association-artifact-manifest.json

python3 superkernel-fusion-performance-analysis/scripts/artifact_contract.py finalize \
  --session /abs/experiment/S1-AUTO/profile.capture-session.json \
  --source-revision REVISION \
  --native-pid PROFILE_PID --started-ns PROFILE_START_NS --ended-ns PROFILE_END_NS \
  --command-fingerprint PROFILE_COMMAND_SHA256 \
  --workload /abs/experiment/workload.json \
  --config /abs/experiment/association-config.json \
  --control /abs/experiment/S1-AUTO/control.json \
  --file kernel_details=profiler/kernel_details.csv \
  --file task_time=profiler/task_time.csv \
  --file trace_view=profiler/trace_view.json \
  --file sk_graph_origin=sk_meta/sk_graph_origin.json \
  --file sk_graph_updated=sk_meta/sk_graph_updated.json \
  --file sk_fused_nodes=sk_meta/sk_fused_nodes.log \
  --file sk_scope_split=sk_meta/sk_scope_split.log \
  --file super_kernel=sk_meta/super_kernel.log \
  --out /abs/experiment/S1-AUTO/profile/association-artifact-manifest.json
```

compat/verify 进程及 generic replay 若需要，可另行采集为候选侧复现审计。其 manifest、
重复 signature 和 replay identity 不写入 profiling analysis request，不参与 SK 到
SK-off fragment 的映射，也不改变性能 classification。

`sk_prof` 不参与 baseline fragment 映射。需要分析融合内调度时，可把诊断 trace 作为
独立可选输入传给 analyzer；不得要求它与 `kernel_details` 建立绝对时钟 offset，也不得
用它替代 profile-process 自有 origin graph、fused metadata 或完整 step kernel projection。
`ASCEND_PROF_SK_ON` 的正整数值是每核 KB 容量，不是布尔开关。若可选调度 trace 的
`super_kernel.log` 出现 `buffer is full, stop dump the time of nodes`，只否决该 trace 的
Cube/Vector/DCCI 调度归因；不得据此否决独立的 kernel projection 映射和 interval 分类。

## 4. 两阶段状态机

### 阶段 A：S 候选筛选

S1/S2/S3/S4/... 各自只运行 `execute -> correctness -> clean timing`。统一汇总：

父 Agent 在派发第一个候选前必须结构化校验所有 S config；每个 config 的
`model_config.custom_params` 必须显式包含空的 `super_kernel_optimize_options: {}` 和
`super_kernel_debug_options: {}`。字段缺失不等于空，禁止依赖 wrapper/backend 默认值。
子 Agent 启动前再次确认本候选两字段为空，并把实际 config 归档为每个
`CLEAN/run-*/config.yaml`。任一非空、缺失或类型错误均 fail-closed；DCCI 和其他 option
不得混入 S1/S2/S3/S4，只能在 winner 的 O 阶段实验。

父 Agent 必须在第一个候选运行前冻结 `superkernel-screening-matrix-v2`。矩阵至少覆盖
`automatic_aot`、`broad_decode`、`per_block`、`semantic_segment`。selection 前每个候选
必须结算为 `executed`、`blocked` 或 `skipped`；后两者需要 named blocker、中文原因和
相对 evidence。不得因已有候选达到收益阈值、已有临时 winner 或为了提前 winner-only
profiling 而跳过余下候选。

最小结构示例：

```json
{
  "schema_version": "superkernel-screening-matrix-v2",
  "candidate_set_frozen_before_execution": true,
  "candidates": [
    {
      "id": "S1-AUTO",
      "strategy_kind": "automatic_aot",
      "only_change": "启用无 marker automatic AOT",
      "status": "executed"
    },
    {
      "id": "S4-SEGMENT",
      "strategy_kind": "semantic_segment",
      "only_change": "使用依赖对齐的语义分段 scope",
      "status": "blocked",
      "blocker": {
        "code": "semantic_boundary_unavailable",
        "reason_zh": "源码证据无法建立保持依赖的语义边界。",
        "evidence": "intake/S4-BLOCKER.md"
      }
    }
  ]
}
```

示例省略的 `broad_decode`、`per_block` 在实际矩阵中仍为必需项。

```bash
python3 scripts/analyze_performance.py \
  --baseline S0 \
  --candidate S1=S1/CLEAN --candidate S2=S2/CLEAN \
  --candidate S3=S3/CLEAN --candidate S4=S4/CLEAN \
  --screening-matrix screening-candidate-matrix.json \
  --selection-only --json-out screening-performance-summary.json
```

此阶段没有 candidate profile、profiling analysis Agent、SK-to-baseline 映射、
promotion；先冻结为 `Sbest-SEED`。若没有 eligible candidate，保留 S0 并停止。

CLI 必须先验证矩阵和 archived config option controls，再读取 clean timing：四类策略覆盖完整，所有状态已结算，矩阵
`executed` ID 与全部 `--candidate` ID 完全相等，且不存在“已有 eligible/winner”型
early-stop blocker。筛选结果记录矩阵 schema、内容 fingerprint、planned/executed IDs、
非执行候选及 blocker，以及 `stage_a_option_controls`。矩阵或空 option 门禁不通过时
不得产生 winner。

### 阶段 O：winner 选项探索

读取 [winner-option-sweep.md](winner-option-sweep.md)。winner child 在任何 candidate
profiling 前冻结 `superkernel-winner-option-matrix-v1`，覆盖 active wrapper 接受的
DCCI、`auto_op_parallel`、aggressive 和其他适用实验值。每项必须结算；缺少 per-SK
profiling 证据不是合法跳过理由。

`Sbest-SEED` 先产生五个稳定 clean process 作为 `O0-INCUMBENT`。普通 O1/O2/... 每轮只在
当前 incumbent 上增加一个 option/value，运行 correctness 和至少三次 clean process，
用 `analyze_performance.py --option-trial` 判断增量收益。DCCI family 只先试
disable-all；有收益直接采纳，neutral/no-gain 直接拒绝。只有 disable-all 劣化时，winner
child 才采集 O0/disable-all paired profiler、SK meta 和完整 `sk_prof`。在任一 profiling
命令前，child 必须按第 3 节为 O0/disable-all 两个新空 role root 分别 `begin`，由采集
producer 在进程结束前归档 profile-owned metadata 并 `finalize`；`raw-run` 事后复制或共享
`sk_meta` 手工复制均不可替代 manifest。先以 `artifact_contract.py validate-set` 验证两份
manifest，验证失败即记录 collection blocker 并从新 root 重采，不能进入逐 SK/child 归因。
验证通过后才交给 fresh read-only analysis child 做全量 SK/child 归因，再按证据运行一次 before+after 联合修复。联合修复与
五轮 O0 比较，不运行独立 before-only/after-only trial。accepted trial 补足五次并通过
稳定性后成为下一 incumbent；rejected/failed trial 回退。除该 DCCI 条件诊断外，O 轮
没有 profile、analysis Agent、SK mapping、source calibration 或 P。全部结算后的配置
冻结为 `Sbest-BASE`。

### 阶段 B：winner 深度优化

只对选中的 winner 建立人工实验族：

默认：`Sbest-SEED -> O0/O* -> Sbest-BASE -> [optional isolated multistream branch -> derived-family BASE] -> [Sbest-SMAP] -> whole_scope_clean_validation`

`Sbest-P* -> Sbest-FINAL` 是可选源码范围分支，仅当用户或冻结实验计划明确请求时
才启动；不得因为 BASE 或 SMAP 已完成而创建 P。该分支 absent、invalid、`no_gain`、
`blocked` 或 `failed` 时精确保留 incumbent，且不得阻塞 unchanged
`whole_scope_clean_validation`。被请求时仍保留所有既有 source-action gate。

[adapter-capability-preflight.md](../../superkernel-multistream-performance-tuning/references/adapter-capability-preflight.md)
把本轮 stage 映射为实际 action 集合，冻结 capability matrix 与 run-specific collection plan，
归档 `status=ready` receipt。P/FINAL 还必须绑定本轮 profile launcher、round action manifest，
并要求 fresh baseline/candidate profile manifest、profile-owned `sk_meta` 与 analysis result。
缺少 adapter、capture/identity/source/dependency provider、执行器、validator 或 expected output
时直接结算为 pre-experiment blocker；不得先跑 profile 再期待后续
日志补齐，也不得消耗该轮 profiling/trace retry 预算。每一轮 action、源码 revision 或 provider
发生变化后必须生成新的 receipt。

示例：

```text
S3-BASE   初始 per-block scope
S3-P1     移除 BASE 中精确 neutral/regressed 范围
S3-P2     如有必要，继续验证非重叠裁剪批次
S3-FINAL  合并 retained beneficial 范围后重新完整分析
```

示例中的 S3 必须是阶段 A 选定的 winner；非 winner 的 S3 不得进入该链。这是源码优化
分支的严格单向状态机。当前 fresh analysis 仍有任一 prune action 时，只生成下一
`P`，不得从同一 BASE/P analysis 同时生成 `FINAL`。只有完成 P 且取得基于该 P 的 fresh
analysis 后才可生成 `FINAL`，且 `comparison_to` 必须是最近完成的 P。

BASE fresh analysis 后允许一次可选的多流调优派发。父 Agent 冻结 incumbent、设备串行
约束、命令和预算，验证 `superkernel-multistream-request-v1`；独立 child 在专用 source
worktree、config root、artifact root 和 cache namespace 中执行。child 不得修改 incumbent
或父 ledger。父 Agent 对返回的 `superkernel-multistream-result-v2` fail closed：无效或
`no_gain/blocked/failed` 时精确回退 incumbent；`accepted` 时建立新的 derived family，
通过共享 device lease runner 做 fresh ordinary BASE，再允许进入该新 family 的
SMAP、默认 whole-scope 或已明确请求的 P/FINAL。只有 `derived_family_lifecycle.py` 验证 fresh-BASE receipt 和普通 schema 2
result 后，才允许合并 ledger。

BASE profile 完成后，所有 winner 先统一执行 source-action mapping。若任一
performance-exact SK 尚无源码 exact 半开区间，strategy 生成一个批量 `Sbest-SMAP`，并
暂缓可选 P 与 `whole_scope_clean_validation`。SMAP 结算后，exact range 仅在明确请求 P 时继续，partial/skipped
range 仅记录并跳过源码动作。随后当前完整候选若全 inventory performance-exact、
analysis 无 blocker 且无 `insufficient_evidence`，strategy 才生成
`whole_scope_clean_validation`。它不修改源码、scope、option 或 workload，局部
neutral/regressed 与已结算的 source-map skip 不阻塞端到端 clean 晋级。

strategy CLI 按轮次递增重复传入同一 family ID，例如：

```bash
python3 scripts/recommend_sk_strategy.py \
  --profiling-analysis S3=S3-BASE/profiling-analysis/profiling-analysis-result.json \
  --profiling-analysis S3=S3-P1/profiling-analysis/profiling-analysis-result.json \
  --source-range-optimization S3
```

每份报告都必须通过 schema、content hash、artifact 和只读重分析校验，且同一 family
目标 range 在历史 prune 报告中的原始 recommendation，并记录源 analysis round/path。
`--source-range-optimization NAME` 可重复，但每个 NAME 在单次调用中只能出现一次；它是
该精确候选由用户或冻结实验计划授权可选源码范围分支的机器可读凭据。任何包含该候选
P/FINAL 历史的调用也必须继续传入此 flag。省略时 strategy 记录
`source_range_optimization_status=not_requested`，不生成 P 或其专属 blocker，并保持
满足门禁的 `whole_scope_clean_validation`；空、未知或重复 NAME 均 fail closed。

winner 的每轮使用独立进程、candidate profile、profile fingerprint、profile-owned
metadata、round evidence、analysis Agent 和 analysis result。winner 的 BASE 与已明确请求的 P/FINAL
round 在 execution、correctness 后进入 diagnostic profiling；`child_count` 只作结构描述，
1、2、3、4 均不得过滤。

Schema 2 lifecycle 中只有 candidate 执行正确性与 fresh profiling 是性能证据门禁。
`compat`、`verify`、`replay` 字段如存在，只记录独立复现审计；其缺失或失败不得阻塞
profiling、手工轮继续或 profile-vs-baseline classification。

### BASE

- `beneficial/keep`：保留。
- `source_scope_map + exact + proven source offsets` 的
  `neutral/prune` 或 `regressed/prune`：仅在可选分支已明确请求时进入 P 候选。
- `insufficient_evidence/reprofile|block`：不改 scope，补证据或终止。

### P

P 轮只移出分析结果精确绑定的 neutral/regressed source range。移出 SK marker
不等于删除算子；算子执行、顺序、依赖、事件、barrier、通信、cache 更新和多流
关系必须保留。

默认实现为在外层命名 scope 内，用平衡的
`super_kernel_scope_begin(None)` / `super_kernel_scope_end(None)` 包围 exact source
range。必须使用 Python `None`，不得使用字符串 `"None"`。只有同推理环境探测报告包含
`scope_capabilities.explicit_none_exclusion.accepted=true` 时才可选择该实现；探测只证明
wrapper 接受调用，fresh metadata 还必须证明目标节点未融合且周围命名 scope 未被破坏。
不支持时可回退到 target 前关闭 outer scope、target 后重新打开同名 outer scope。
一个 P 轮只能有一种 exclusion method；`explicit_none_scope` 与
`outer_scope_split` 的 A/B 必须拆轮，declared change 必须记录所选方法和对应 marker
变化。

一次 batch prune 仅允许 producer `mapping_method=source_scope_map`、
`mapping_confidence=exact`，并且每对范围都有
`source_file/start_offset/end_offset` 可机器证明不重叠的 source ranges。跨文件和
缺失源码区间均不可批量；所有范围还必须位于同一 `source_file`。相邻 half-open
区间可批量；交叠、包含、同边界不可批量。只有 op 名字符串的
boundary 不构成源码区间证明，默认一 range 一 P。不得使用 consumer 自报
`mapping_reliable`。历史 `sk_meta_node_ids + exact`、raw Task/model/stream/node ID、
名称和局部同名窗口都只是 diagnostic provenance，不能作为 exact fallback。profile
映射存在多个候选解、`count_reliable=false` 或低 mapping confidence 时拆轮或
reprofile，禁止猜测。P 轮必须通过 fresh analysis；被裁剪
范围只能由严格后续且明确引用该范围的 P 轮标记为 `verified`。

 ### FINAL

FINAL 合并 retained beneficial ranges，必须重新执行完整 fresh
profiling。交互检查要求：

- `retained_range_ids` 无冲突；
- 每个最终范围都有 `source_scope_map + exact + proven source offsets`，并在 FINAL
  本轮得到 `beneficial/keep`；
- FINAL 不含 `regressed/block/blocked` 或相互冲突的同范围 decision；
- FINAL 的 candidate profile、analysis Agent 和 result 路径均未复用。

上述交互检查仅约束 range-optimized FINAL。未做源码动作的 whole-scope 候选也必须先
结算 source mapping，但不要求每个局部范围最终都 beneficial/source-actionable；它以
完整 performance-exact 映射、无 `insufficient_evidence` 和明确的 source-map
exact/partial/skipped 结果为门禁。两条路径通过各自门禁后都允许 clean timing；FINAL
decision 不重复晋级为历史条件证据。

## 5. Automatic AOT Winner

S1 在筛选阶段只有 clean timing。只有 S1 成为 winner 后，才冻结为 `Sbest-SEED`，完成
相同的 O 矩阵并生成 `Sbest-BASE`，然后使用 `round_kind=automatic_aot` 进入 profiling。
每个可靠 mapped SK 仍要 profiling 和 fresh 分析；任一 performance-exact SK 没有
source exact 时先建立一个 winner-only `Sbest-SMAP`，而不是把 AUTO 改造成新的 manual
scope candidate。这与所有 manual winner 使用的必经映射门禁相同，仅 source revision
角色和标定 bridge 不同。

- 已分类 graph occurrence 仅作为 proposed 性能证据，不称为源码范围证据；
- 任一 performance-exact SK 无 source exact 时，按 range/SK identity 与可用 fingerprint
  集合提出一个批量 SMAP；
- `insufficient_evidence` 保留在 unresolved 集合，不裁剪；
- SMAP 完成前不产生 prune 证据或源码动作。

SMAP 在独立标定 worktree 使用临时 marker，通过 `stable_source +
marker_only_calibration_v1`、唯一图投影、三步 assignment 与 `C_sk == C_unit` 把范围回投到
原始 AUTO 源码。analyzer 取得 `source_scope_map + exact + proven offsets` 后，后续
已明确请求的 P/FINAL 可执行源码动作；初始 S1-AUTO 分析仍为 proposed，只有严格更晚 P 的 fresh
verification 才能更新 AUTO 源 decision 的 applied/verified 状态；
SMAP 完成并记录 exact/partial/skipped 后，未 exact range 只跳过源码动作，不禁止原样
候选进入 `whole_scope_clean_validation`。全量 performance-exact 映射、无
`insufficient_evidence` 且 correctness 通过时，S1 的端到端 clean 结果可决定 automatic
AOT 是否使能。

## 6. Schema 2 交接

`experiment-result.json` 的核心字段：

```text
experiment_id
parent_experiment_id
child_agent_id
baseline_revision
source_revision
baseline_config_fingerprint
control_fingerprint
workload_fingerprint
rounds
profiling_analysis_agent_ids
profiling_analysis_result
performance_scope_decisions
verified_ranges
unresolved_performance_ranges
inherited_exclusions_for_next_agents
lifecycle
blockers
next_agent_guidance_zh
```

`profiling_analysis_agent_ids` 和 `profiling_analysis_result` 都是
`round_id -> value` map。analysis Agent ID 必须不同于 `child_agent_id`，且
所有 profiling 轮之间唯一。result 路径也必须唯一。

每个 profiling round 以及其 decision 都绑定：

```text
baseline_profile
baseline_profile_fingerprint
candidate_profile
candidate_profile_fingerprint
declared_change_set
```

Candidate profile 每轮必须新鲜。冻结 baseline 只可在
`baseline_revision + baseline_config_fingerprint + control_fingerprint +
workload_fingerprint` 完全相同时复用。路径和内容 fingerprint 双向绑定；
baseline、candidate、analysis result、round report、metadata、replay、config、
source evidence 与 clean artifact 不得跨语义角色复用。

活动 conditional evidence 的 key 是：

```text
source_revision
baseline_config_fingerprint
candidate_config_fingerprint
control_fingerprint
workload_fingerprint
range_id
```

父 Agent 只在六字段完全一致时向后续 Agent 提供条件结论。账本每次 merge 都从
不可变实验记录重建该索引。schema 1 legacy 实验和旧单子算子排除只读保留，不晋级。

## 7. 校验与合并

```bash
python3 <runtime-skill-dir>/scripts/experiment_ledger.py validate \
  experiments/S3/experiment-result.json

python3 <runtime-skill-dir>/scripts/experiment_ledger.py merge \
  --ledger experiments/experiment-ledger.json \
  --result experiments/S3/experiment-result.json \
  --output experiments/experiment-ledger.json
```

只有 `valid=true` 才可合并。相同 `experiment_id` 是不可变记录；内容变化必须
使用新的实验 ID 和 fresh artifacts。失败或末轮 blocked 也要保留生命周期、
blocker、中文结论和相对 artifact 路径。任何非环境类失败还必须先按
[failure-scene-reporting.md](failure-scene-reporting.md) 封存现场，并在该实验自己的
report 中逐 attempt 记录；父 Agent 的最终摘要不能代替这份本地记录。

validate/merge CLI 固定以 `experiment-result.json` 所在目录为 artifact root，读取并
只读重放每个 `profiling_analysis_result`。路径缺失/逃逸、schema/content hash、
identity/Agent/round/fingerprint 或 `(round_id, range_id)` analyzer decision 不一致均
拒绝。无 artifact root 的纯对象调用不可向 `conditional_performance_evidence` 晋级。

## 8. 中文交接内容

阶段 A 的候选报告只说明目标、唯一变量、execution/correctness、clean timing、
fingerprints 和 eligibility，不含以下 per-SK 字段。winner 的 `EXPERIMENT_REPORT.md`
至少说明：

- 主实验目标、唯一变量、source diff 和 `declared_change_set`；
- 每轮 candidate/correctness/profiling 状态，以及独立列出的可选复现审计状态；
- interval P50、duration sum P50、SK P50、MAD threshold；
- classification/action、mapping confidence、analysis Agent、conditional evidence；
- beneficial 保留、neutral/regressed 裁剪、insufficient evidence blocker；
- P/FINAL 的调整、交互结果和 clean timing 资格；
- 相对 artifact、Agent ID map、profile fingerprints、失败证据；
- 给父 Agent 和后续 Agent 的中文指导。

如果任一 candidate/round/trial 已启动后失败，`EXPERIMENT_REPORT.md` 还必须包含
`failure-scene-reporting.md` 定义的身份与阶段、冻结调用、可观察失败、现场 artifact、
门禁进度、初步判断、控制与回退、后续定位入口。重试按 attempt 追加，禁止覆盖失败小节。
只有实验启动前的环境探测失败可只写 environment artifact 和父级 blocker。

父 Agent 不用自然语言覆盖 validator 的拒绝，也不从摘要重新推导 profiler 分类。
