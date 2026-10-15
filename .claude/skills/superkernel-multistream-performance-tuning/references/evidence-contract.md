# 多流证据契约

## 输入层次

新 skill 不重做通用 SK 映射。父流程必须提供 sibling analyzer 已验证的
`profiling-analysis-result.json`、两份 collection manifest、
`projected-trace-mapping.json`、五次稳定 incumbent clean 汇总和 Stage O option 历史。

融合前调度来自 SK-off `kernel_details.csv` 映射后的 child rows。每个 occurrence 按
`device_id + model_id + step_id` 独立计算 interval、duration sum、union、stream-role
overlap 和 core-family overlap。raw stream/task/node ID 只用于同 artifact 解析。

融合后的父 SK interval 来自 candidate `kernel_details.csv`。SK 内 child lane 调度来自
短窗口 `sk_prof_device_X.json`。直接调度证据要求每个事件可绑定：

```text
device_id + model_id + parent_sk_id + step_id + range_id
child origin identity + lane/stream identity + core family + start/duration
```

若 active trace 不输出所有字段，可使用同一 profile process 的密封 metadata 建立关联，
但不得按名称、绝对时间 offset 或 occurrence ordinal 猜测。`kernel_details` 与 `sk_prof`
可属于不同时钟域；两侧分别计算相对 interval/overlap，不求跨域绝对 offset。

## 必需 artifact

请求中的以下相对路径必须存在且不能逃逸 request 根目录：

- incumbent clean performance summary；
- sibling profiling analysis result；
- baseline/candidate collection manifest；
- projected trace mapping；
- Stage O option history；
- environment evidence。

`sk_prof` 可在请求时缺失，因为独立分支可以短窗口补采；但没有成功、完整且可归属的
trace 时，不得形成内部串行化直接结论或执行基于该结论的 trial。

短窗口统一表示为 `superkernel-multistream-short-trace-capture-v2`。它不是原始 trace 的副本，
而是对原始文件 SHA256 的规范化绑定：每个 target 包含 parent device/model/SK identity，
baseline/candidate 使用相同的至少三个 `alignment_id`，每个 child event 提供 origin、lane、
stream、`Accelerator Core/Block Num/Mix Block Num`、start/duration。
`multistream_trace_analysis.py` 会先派生 core family，再确定性重算全部指标；缺字段或
capture/source 被修改时 fail closed。

在生成 short-trace request 前，必须以 sibling schema 1.2 result 运行
`multistream_opportunity_discovery.py discover`。eligible 集合包含所有 performance-exact、
SK-off/SK-on 各至少三个 occurrence、已证明至少双 stream 的 `beneficial/neutral/regressed`
SK。`beneficial` 不得因已有净收益被排除。短 trace 完成后运行 `audit-coverage`；只有 target
identity 集合完全相等、trace 无 blocker 时，才可以声称完成了隐性并行损失筛查。

baseline/candidate 每个 target 的 `child_origin_identity` 集合必须完全相等，并在各自所有
occurrence 中稳定。并行退化先由任意 cross-stream overlap 保持程度判定；可动作性再由三字段
派生的纯 Cube/Vector overlap 独立判定。因此 Mix 或同资源 overlap 消失可以证明
`parallelism_effect=degraded`，但必须产生结构化 action blocker，不能授权 reorder/event/stage。

`source_scope_map` 也是条件输入：option-only trial 不要求它；scope split 和 range
exclusion 必须要求 `source_scope_map_v2 + exact`、当前 candidate binding 以及有效
`source_file/start_offset/end_offset`。

`dependency_safe_operator_reorder` 还要求 request 提供
`superkernel-multistream-operator-order-capture-v3`。模型 adapter 必须先把 SK-off
`kernel_details.csv` 中的 kernel 聚合到业务 operator/statement identity，再把 SK-on
`sk_meta` 下发序绑定到同一 identity。不得直接比较 raw CSV 行号、日志行号、算子名称或
Task ID。每个 target 还必须提供：

- 同一 exact range 内连续、互不重叠的完整 statement byte span；
- 显式 `movable=true`、`side_effect_free=true`；
- DATA、STREAM_ORDER、event、wait、barrier、通信、cache mutation、副作用和控制流 hard edge；
- 至少三个对齐 occurrence 的 SK-off start/duration/stream，以及每个算子的原始
  `Accelerator Core/Block Num/Mix Block Num`；
- 相同 occurrence 的 SK-on dispatch order。

每个 target 还必须绑定通过 `multistream_dependency_evidence.py --require-complete` 的
dependency evidence；capture 内 hard edges 必须与该 evidence 完全一致。历史 v2 capture
缺少这一绑定，只能保留为诊断 artifact，不能授权新的业务算子重排。

普通重排授权是硬多流门禁：每个 occurrence 必须存在同一稳定 `Cube <-> Vector` pair 的跨
stream overlap，且 SK-on 下发序必须对该 execution preference 形成一致 inversion。
`Cube <-> Cube`、`Vector <-> Vector`、通信或等待 pair 都不是可授权 pair；无数据依赖不等于
可并行。单流、stream identity 不完整或只观察到性能劣化时不得授权。

MIX 业务算子默认仍不可按整体资源类型授权。只有
`component-overlap-reorder.md` 定义的 component-aware route 可以例外：同一父 SK/设备/模型/
step/range occurrence 中的全部 AIC/AIV lane 必须完整绑定，目标 lane 必须与另一算子形成稳定
互补 overlap，MIX 的剩余 lane 必须纳入同资源竞争分析，并且完整依赖证据允许该源码调序。
任何缺 lane、按名称猜测 lane、把整个 MIX 强行标成 Cube/Vector 的 capture 都必须阻塞。

`core_family` 不得由 capture producer 直接填写。合同按三字段确定性派生：

- `AI_VECTOR_CORE -> VECTOR`；
- `AI_CORE -> CUBE`；
- `MIX_AIV + Mix Block Num=0 -> VECTOR`，非零为 `MIX`；
- `MIX_AIC + Mix Block Num=0 -> CUBE`，非零为 `MIX`。

`Block Num` 即使不改变上述分支，也必须存在且为非负整数，用于绑定完整 profiler 资源身份。
只看 `Accelerator Core`、operator name 或 `sk_prof` 的 AIC/AIV lane 都不能授权重排。

合同工具会重新校验 incumbent schema 1.2 analysis 的内容 fingerprint、candidate/source/
config/control/workload binding、每个 target 的唯一 decision、performance-exact mapping 和
双方至少三个 occurrence。源码 trial 还必须与该 analysis 中
`source_scope_mapping.status=exact` 及同一 decision 的精确 boundary 完全一致；仅放置一个
同名或空 JSON 文件不能授权源码动作。

重排 correctness 通过后还要生成
`superkernel-multistream-post-reorder-dispatch-capture-v1`，并由
`multistream_operator_order.py verify-dispatch` 生成可重放 evidence。runner 在 profile 前
要求证据证明计划顺序已生效、child set 未改变、至少三个 occurrence 且每次仍有两个以上
可靠 stream。该 evidence 缺失或失败时不能进入 profiling。

## Trace 溢出

`super_kernel.log` 出现 `buffer is full, stop dump the time of nodes` 时，该 attempt 不能
证明内部 lane、Cube/Vector 或 DCCI 行为。最多用更短窗口重采三次；每次使用新的输出根，
保存失败日志和 fingerprint。溢出只阻塞多流分支，不否决 incumbent 已有的 interval
classification。

## Incumbent 异常

普通多流证据缺失返回 `blocked` 并回退 incumbent。若验证发现 request 引用的 incumbent
profile/config/workload/clean artifact 已被修改、缺失或互相不匹配，输出专门 blocker
`incumbent_evidence_invalid`。这是原证据失效，不得伪装成普通可忽略的多流失败。
