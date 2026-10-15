# SuperKernel Kernel Projection 映射协议

## 唯一协议：kernel_projection_trace_v2

融合前子算子映射只使用完整 step 的可观测 kernel 投影。静态图包含 profiler
不稳定观测的 control/event 节点，`kernel_details` 与 `sk_prof` 也可能属于不同
时钟域，因此禁止静态控制全图同构、跨时钟绝对 offset、raw ID 或局部名称窗口
作为映射方案或 fallback。

映射流程：

1. 从 profile 进程自己的 `sk_graph_origin.json` 删除 topology-only control/event
   节点。不得按算子名全局删除 kernel。若 origin 独有 scope sentinel，只有
   `scope_sentinel_exclusion_v2` 的完整 occurrence、同会话 origin/updated/fused/profile、全 step
   计数与唯一完整序列、所有 fused child 落点门禁同时通过，才可排除已记录的具体
   node key；任一门禁失败保持原始 projection；
2. 对每个 graph stream 保留完整 kernel 顺序，节点 identity 为稳定 stream role 与
   kernel ordinal；
3. 从 baseline `kernel_details.csv` 按 `device_id + model_id + Step Id` 构造每个
   step 的完整流内 kernel 序列；
4. 规范化 profiler 的 `aclnn..._<Op>` wrapper 和通信 kernel 名称，搜索唯一的
   injective stream-role assignment；
5. 要求每个 measured step 的节点总数、stream 数、完整序列和 core family 均一致，
   并显式搜索第二解；
6. 从同进程 `sk_fused_nodes.log` 读取 fused child 的 origin nodeId，将它转换为
   `(source stream role, kernel ordinal)`，再落到每个 baseline step 的唯一 kernel row；
7. 对映射行计算 `interval=max(end)-min(start)`、`duration_sum` 与区间 union。

只有 stream-role 注入唯一、所有 step 全量一致、每个 child 唯一落点、候选父 SK
在至少三个 step 中均存在时，才输出
`kernel_projection_structural + exact_projected_trace`。任一门禁失败都返回
`diagnostic_only + insufficient_evidence`；不得执行其他映射算法补洞。

`scope_sentinel_exclusion_v2` 不是 op-name ignore list。它只允许 anchored
`sk_scope_kernel_begin_dav_<tag>` / `sk_placeholder_kernel_dav_<tag>` /
`sk_scope_kernel_end_dav_<tag>` 在同一 stream role 中按 marker occurrence ordinal 切分
`B,P,P / P,P,E` 双链候选；tag 只校验单个 occurrence 内部一致性，不作为 occurrence
identity。每个 occurrence 的两段必须分别连续且中间存在业务节点，所有 marker 必须恰好
被分区，并只排除最终证据中列出的 node key。
候选链不完整、不连续、跨 stream、被 fused group 引用、出现在任一 profiler 或 updated
graph、过滤后计数不等、流映射不唯一/跨 step 不一致、任一 fused child 无法精确落点时，
整个候选集合拒绝生效，原始 projection 保持不变。候选集合、逐门禁结果与 blockers 必须
写入 `projection_exclusions` 并参与 mapping fingerprint。

## 时钟与 occurrence 边界

`sk_prof` 不参与 baseline 子片段映射。按显式
`device_id + model_id + sk_id + Step Id` 绑定父 occurrence。`sk_prof` 仅在映射完成后
用于融合内 child 调度、Cube/Vector 或 DCCI 归因；其缺失、lane 不完整或无法跨时钟域
求绝对 offset，只限制内部调度归因，不影响已经证明的 projected mapping。

raw Task ID、raw stream ID 和 raw node ID 只允许在同一 artifact 内解析或诊断展示，
不能跨进程产生 exact confidence。

## 性能映射与可复现性审计分层

融合前后性能比较只建立 `candidate_profile -> baseline_profile(SK off)` 映射。
profile 进程自己的 origin graph、fused metadata 和 candidate kernel rows 共同证明
当前 SK occurrence；baseline kernel rows 提供融合前 interval。compat/verify 不参与该
映射，也不允许把跨进程 signature 匹配失败写入 `mapping_errors`。

compat/verify 可作为独立的候选侧可复现性审计：比较配置、workload 和规范化 fusion
inventory，或在需要晋级复验时重新采集 instrumented candidate profile。审计结果可以
单独标记 `reproducibility_unproven`，但不得把已经 exact 的 profile-to-baseline 映射
降级为 `insufficient_evidence`。性能映射不得出现
`fusion_replay_identity_unmatched` 门禁。

## Blockers

```text
kernel_projection_step_missing
kernel_projection_stream_count_mismatch
kernel_projection_stream_unmapped
kernel_projection_injective_assignment_missing
kernel_projection_stream_assignment_ambiguous
kernel_projection_step_inconsistent
candidate_projection_occurrence_domain_mismatch
metadata_origin_node_not_in_kernel_projection
baseline_projection_ordinal_missing
baseline_projection_child_mismatch
kernel_projection_artifact_association_failed
```

## 输出与源码边界

运行 `scripts/projected_trace_mapping.py` 生成独立 JSON 与中文报告。JSON 必须保存
完整 step 集、每步 stream-role assignment、替代解数、每个 child 的 baseline source
row、task/stream 诊断字段、三类区间统计和全部输入指纹。

`exact_projected_trace` 只证明 measured graph occurrence，不等价于任何源码语言的区间。
源码动作仍要求独立、由 analyzer 重放通过的 `source_scope_map_v2 + exact`，以及有效递增的
`source_file/start_offset/end_offset`。缺少源码映射时只能提出由
`graph_occurrence_fingerprint` 标识的人工单范围复测，不能生成或修改 scope。
完整协议见 [source-calibration-mapping.md](source-calibration-mapping.md)。旧 task range、
layer map、重复 signature 或局部 op window 不能产生源码 exact。
