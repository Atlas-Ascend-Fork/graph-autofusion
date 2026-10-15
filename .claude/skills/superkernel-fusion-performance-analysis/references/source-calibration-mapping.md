# SK 到可编辑源码的通用标定映射

## 适用边界

本协议只解决 `SK graph occurrence -> semantic source unit`。性能映射仍由
`kernel_projection_trace_v2` 独立完成：

```text
SK-on candidate occurrence
  -> origin fused child occurrence
  -> SK-off baseline child rows
```

源码映射使用另一条证据链：

```text
original business graph
  -> unique labeled multigraph isomorphism
  -> calibration business occurrence
  -> calibration unit assignment
  -> archived source bytes + [start_offset, end_offset)
```

不要把两条链合并成 ordinal fallback。raw task/node/stream/SK ID、生成名称、层号、
重复 op signature、局部 op window 和跨进程时间戳都不能产生源码 exact。

## Adapter 边界

核心实现不包含模型名、固定层数、奇偶层、Attention/MoE 或 Python 假设。注册的 model/
language adapter 只负责产出标准 JSON：

- block template、opaque `block_instance_id` 和 runtime scope binding；
- 可被平衡、互斥 marker 包围的 semantic unit；
- parser-node byte span、normalized syntax 和 marker operation span；
- 临时 calibration patch 与 declared change；
- calibration occurrence 到 `(block_instance_id, unit_id)` 的可审计归属，以及每个
  measured step 中 exact-assigned、skipped、conflicting 三个互斥节点集合。

分析器不 import 或执行 adapter。DeepSeek 等模型只能作为 adapter 示例，不能进入通用
identity 或通过条件。

## Artifact 流程

1. adapter 从 calibration 或 stable-marker 源码输出 adapter JSON。
2. `generate_source_unit_manifest.py` 生成 manifest 和不可变 source snapshot。
3. original 与 calibration origin graph 分别规范化为
   `normalized_business_graph_v2`。
4. `project_calibration_graph.py` 搜索唯一 stream-role 注入和 node 双射。
5. `partition_calibration_scope_log.py` 从标定日志生成
   `calibration_unit_assignments_v1`；`generate_original_fused_inventory.py` 从 original
   exact projection、图双射和已证明 block assignment 生成
   `original_fused_inventory_v1`。inventory 必须复用 structural analyzer 的 canonical
   graph-occurrence fingerprint，并分别封存原始 SK 的 `candidate_source_scope` 与 marker
   block 的 `source_scope`；两种 namespace 不得互相比较。无 block 归属或跨 block 的 SK
   必须跳过。
6. `build_sk_source_map.py` 重算 `C_sk` 与 `C_unit` 的集合关系。
7. 命名 scope 路径固化选定 marker 后，fresh correctness/profile；automatic AOT 路径
   冻结无 marker 的 `stable_source`，并生成可证明纯插入的
   `marker_only_calibration_v1`；不得复用 calibration timing。
8. `source_scope_map_v2.py build` 封装可动作 source revision、源快照、投影、assignment、
   fused inventory、三份 collection manifest 和 provenance DAG。automatic 路径还必须
   封装 calibration manifest/snapshot 与 marker-only bridge。
9. 性能分析器加载 v2 时再次重放全部证据，只让 `exact_cover` 进入源码动作。

标定 marker 是临时诊断设施。命名 scope 路径中，进入 range-optimized FINAL 的 marker
必须固化到新的 `stable_marker` revision。automatic AOT 路径中，标定 marker 不得进入
生产源码；bundle 封存后删除标定 worktree，最终区间必须指向冻结的无 marker
`stable_source`。P/FINAL 的实际源码变化仍需 fresh correctness/profile。

## 图投影门禁

节点 label 包含版本化 `canonical_op`、`core_family` 和可稳定观测的 dtype/shape/port/
业务属性。边保留 `DATA`、`BUSINESS_CONTROL` 和 `STREAM_ORDER` 类型及端口。投影必须满足：

- device/model identity 相同；
- business node 和 typed edge multiset 完全相等；
- stream-role 单射和 node 双射唯一；
- 显式搜索第二个 correspondence，替代解为零；
- 非平凡自同构为零；
- 图同构始终覆盖完整业务图。源码归属按 SK group 独立验证：该 SK 的全部 child
  occurrence 必须在至少三个 measured step 中位于 exact-assigned 集合，且不能出现在
  skipped/conflicting 集合；与该 SK 无关的未归属节点可以跳过。

重复 label 或对称图不能靠 ordinal 人工选解。图信息无法消歧时保持 `diagnostic_only`。

CANN origin graph 可把模型实例写成 `48_1`，而 profiler/投影 identity 使用基础模型号
`48`。规范化器保存完整 `origin_model_id` 作为证据，只用第一个下划线前的整数部分参与
跨 artifact model identity；非整数前缀必须 fail-closed。

当每条 stream 的 `STREAM_ORDER` 边完整且恰好构成连续全链时，同构器先搜索唯一
stream-role 双射，再由链拓扑导出 node correspondence，并对全部 typed edge 做全量重放；
仍需显式搜索第二个 stream assignment。这里的流内位置来自已密封的完整链拓扑，不是
缺边图上的 ordinal fallback。无法证明完整链时回退到保守的通用 node 同构搜索。

## 源码 manifest 门禁

- offset 是原始文件 UTF-8 byte offset，不做换行或 Unicode 归一化；
- source path 必须是 snapshot root 内无 `..` 的相对路径；
- `0 <= start < end <= file_size`，相邻半开区间允许，重叠 unit 禁止；
- unit envelope、begin marker operation、end marker operation 分别保存 exact byte SHA；
- calibration/stable_marker 的 begin/end operation 必须成对、唯一、位于 unit envelope 内；
- `stable_source` unit 不得包含 marker operation，其 byte span 直接指向生产源码；
- normalized syntax hash、adapter fingerprint、source revision 和每个文件 SHA 均参与 identity；
- 可动作 round 接受两种互斥角色：`stable_marker` 要求
  `source_revision == stable_marker_revision == --source-revision`；`stable_source`
  要求 `source_revision == --source-revision`、无 `stable_marker_revision`，并通过下述
  marker-only bridge。

## Automatic AOT 的 Marker-Only Bridge

automatic AOT winner 缺少现成命名 scope 时，不得把它重建成新的 manual winner family。
adapter 应分别生成：

- 冻结生产源码的 `stable_source` manifest/snapshot，unit span 不含 marker；
- 独立 calibration worktree 的 `calibration` manifest/snapshot，unit identity 与 syntax
  hash 相同，额外记录 begin/end marker operation 和完整 insertion span；
- `marker_only_calibration_v1`，逐文件列出所有互不重叠的 marker insertion byte span。

运行 `marker_only_calibration.py` 删除 calibration snapshot 中封存的 insertion spans；
结果必须与 stable snapshot 每个文件逐字节一致。bridge 同时绑定两个 manifest
fingerprint、base/calibration revision、adapter、block template、unit ID 和 syntax hash。
任何非 marker 业务源码差异、插入区间重叠、缺失 marker、文件集合变化或 SHA 不符都
fail-closed。bridge 只证明源码变换可逆；业务节点身份仍必须通过完整 original/calibration
图唯一投影、至少三步 assignment 和 `C_sk == C_unit`。

## 集合关系与重复结构

`C_sk` 是原始 SK 全部 child occurrence 经唯一投影后的集合；`C_unit` 是一个 calibration
unit 内全部已证明属于该 unit 的业务 occurrence，包括未融合节点。只有 `C_sk == C_unit`
是 `exact_cover`。与当前 SK 无关、且没有 unit assignment 的图节点记录为 skipped，不参与
该 SK 的集合比较；如果 SK 自身任一 child 缺失 assignment 或三步验证，该 SK 输出
`processing_status=skipped + diagnostic_only`，其他 SK 继续处理。不得把已属于目标 unit
但未融合的节点标成 skipped 来伪造 `C_sk == C_unit`。部分交叉、多解 composite 和
shared-unit 仍不可动作。若一个 SK 的 child 跨越多个 block instance，同样只跳过该 SK，
记录 `target_child_block_mismatch`，不得中止其他独立 SK group。

structural family 仅由 block-local stream role、block-local ordinal、完整业务序列与 local SK
child set 构成，不含全图 ordinal、层号、block ID 或模型名。三种置信度：

- `source_unit_exact`：该实例直接标定且至少三步一致；
- `source_unit_template_exact`：同 family 至少三个不同实例、无反例且 unit syntax hash 相同；
- `diagnostic_only`：任何歧义、部分交叉、hash/revision 失效或 assignment 不完整。

不存在重复 block 的网络仍可取得 instance exact。block 内相同 stream signature 无法唯一
编号时，不允许升级模板共识。

每条 unit assignment 和 fused SK group 必须引用至少一条
`artifact_id + record_id + record_fingerprint` 标准日志 record。对应 catalog 保存相对
artifact path 和 SHA；validator 重新核对 record 中的 node/block/unit 或 SK/child
identity。仅有 producer 汇总 JSON、无法解引用到 sealed normalized log record 时，不得
输出 exact。

解析 `sk_scope_split.log` 时，源码单元成员默认只来自对应 scope 的
`PrintScopeNodes` 节点列表。CANN 因不可融合节点把一个 marker 拆为连续多个 scope 时，
只有下一条 scope 具有完全相同的 marker identity，当前 `BreakInfo.triggerNode` 才作为内部
断点归属该 unit。最后一个同名片段的 trigger 位于 marker 结束之后，只作诊断、不得归属
当前 unit；否则会把相邻的 Cast/MatMul 等节点错误吸入 `C_unit`，把真实 `exact_cover`
降级为 `partial_intersection`。

## `source_scope_map_v2` Fail-Closed 规则

loader 必须从 map 所在 bundle root 解析相对路径，拒绝绝对路径、`..`、symlink escape、
缺失文件和 SHA 不符。它重算：

- source manifest 和 source snapshot 文件/span SHA；`stable_marker` 额外重放 marker SHA，
  `stable_source` 额外重放 calibration snapshot 与 marker-only bridge；
- calibration projection fingerprint 和全图唯一性，以及每个 SK target child 的三步
  exact-assignment 门禁；
- original fused children 经 correspondence 得到的 `C_sk`；
- unit assignments 得到的目标 unit 完整 `C_unit`，以及未处理节点清单；
- `sk-source-map.json` 和 map content fingerprint；
- provenance DAG 的 artifact hash、依赖闭包与无环性；
- 当前 candidate collection manifest SHA；
- 当前 structural association 的 graph occurrence 与 baseline projection fingerprint。
- source range 的 `candidate_source_scope` 与当前原始 SK scope；marker `source_scope` 只与
  block-instance runtime binding 比较。

旧 list、`task_ranges` 或 layer map 只可作为展示标签，loader 返回
`diagnostic_only`，永远不能产生 `source_scope_map + exact`。

## CLI

```bash
python3 scripts/normalize_business_graph.py \
  --origin-graph original/sk_meta/sk_graph_origin.json \
  --collection-fingerprint ORIGINAL_COLLECTION_SHA256 \
  --projection-trace original/kernel-projection-trace.json \
  --output original-business-graph.json

python3 scripts/generate_source_unit_manifest.py \
  --adapter-output adapter-output.json \
  --source-root stable-marker-worktree \
  --snapshot-root bundle/source/snapshot \
  --output bundle/source/source-unit-manifest.json

# automatic AOT only: prove calibration is marker-only relative to stable AUTO source
python3 scripts/marker_only_calibration.py \
  --stable-source-manifest bundle/source/source-unit-manifest.json \
  --calibration-source-manifest bundle/source/calibration-source-unit-manifest.json \
  --stable-source-snapshot-root bundle/source/snapshot \
  --calibration-source-snapshot-root bundle/source/calibration-snapshot \
  --output bundle/mapping/marker-only-calibration.json

python3 scripts/project_calibration_graph.py \
  --original-graph original-business-graph.json \
  --calibration-graph calibration-business-graph.json \
  --runtime-validation runtime-validation.json \
  --evidence-catalog evidence-catalog.json \
  --output bundle/mapping/calibration-projection.json

python3 scripts/partition_calibration_scope_log.py \
  --scope-log calibration/sk_scope_split.log \
  --business-graph calibration-business-graph.json \
  --source-manifest bundle/source/source-unit-manifest.json \
  --steps 3 4 5 \
  --artifact-root bundle \
  --output-dir bundle/mapping/partition

python3 scripts/generate_original_fused_inventory.py \
  --projection-trace original/kernel-projection-trace.json \
  --calibration-projection bundle/mapping/calibration-projection.json \
  --block-instance-inventory bundle/mapping/partition/block-instance-inventory.json \
  --unit-assignments bundle/mapping/partition/unit-assignments.json \
  --artifact-root bundle \
  --evidence-output bundle/mapping/original-fused-records.json \
  --output bundle/mapping/original-fused-inventory.json

python3 scripts/build_sk_source_map.py \
  --calibration-projection bundle/mapping/calibration-projection.json \
  --source-manifest bundle/source/source-unit-manifest.json \
  --block-instance-inventory bundle/mapping/block-instance-inventory.json \
  --original-fused-inventory bundle/mapping/original-fused-inventory.json \
  --unit-assignments bundle/mapping/unit-assignments.json \
  --output bundle/mapping/sk-source-map.json
```

automatic AOT 路径还要给 `build_sk_source_map.py` 传入
`--calibration-source-manifest`。block inventory 绑定 calibration manifest，因为 scope-log
成员来自标定图；最终 source interval 则只从 stable-source manifest 读取。两个 manifest
的 unit ID、template、source symbol/file 和 syntax hash 必须一致。

`normalize_business_graph.py` 复用现有 origin graph parser。若图中存在 scope sentinel，
必须提供同一 origin graph fingerprint 绑定且自身 fingerprint 可复算的
`kernel_projection_trace_v2`；只有其中 `scope_sentinel_exclusion_v2` 已 accepted 且恰好
覆盖全部 marker node key 时才执行排除。origin graph 不提供 DATA/shape 时工具不会补造，
由此造成的同构多解会在下一步 fail closed。

sentinel 提案兼容两种可证明布局：单流且非嵌套时继续验证固定的
`begin + 2 placeholder / business / 2 placeholder + end` 链；显式标定 manifest
声明非空 `config.marker_namespace` 时，允许嵌套作用域和 marker 跨多个 stream role，
但必须逐 tag 满足 begin/end 一一配额及每次 occurrence 四个 placeholder，并要求所有
origin graph 的逐流 marker 签名完全一致。普通候选仍要求 profiler 中不可见 marker；显式
标定运行允许 profiler 可见，但来源一致、融合子节点未引用、基线不可见、updated graph
不可见、排除后全序列唯一投影和跨 step 一致等门禁保持不变。

随后构造包含上述文件、三份 collection manifest 和所有 projection evidence 的无环
`source_mapping_provenance_dag_v1`，再运行 `source_scope_map_v2.py build`。最后可独立执行：

```bash
python3 scripts/source_scope_map_v2.py validate \
  --map bundle/source-scope-map-v2.json \
  --source-revision STABLE_REVISION \
  --source-root actionable-source-worktree \
  --candidate-manifest-sha256 SHA256
```

构建 automatic bundle 时使用 `--stable-source-revision`，并额外传入
`--calibration-source-manifest`、`--calibration-source-snapshot-root` 和
`--marker-only-calibration`；命名 scope bundle 继续使用 `--stable-marker-revision`。

外部 validate 不是 analyzer 门禁的替代品；`analyze_fusion_performance.py` 会再次验证。
analyzer 在 v2 exact 时强制要求 `--source-root`，并逐文件比较当前待编辑 worktree 与
archived actionable source snapshot；任何字节差异都会使源码映射失效。
