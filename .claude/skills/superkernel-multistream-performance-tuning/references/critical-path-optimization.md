# 关键路径多流优化协议

本协议用于模型多流分支的 event 编排和业务阶段拆分。它不替代既有的同父 SK 业务算子重排协议：reorder 仍必须满足同一父 SK、稳定 Cube/Vector 下发逆序和完整 statement DAG。

## 入口

adapter 先提交 `superkernel-multistream-logical-graph-v1`，再提交至少三个对齐 occurrence 的 `superkernel-multistream-critical-path-capture-v1`。逻辑图必须绑定完整 dependency evidence；stage 必须声明可靠 stream 和副作用 blocker；event 必须证明迭代/ring 复用安全。capture 必须声明时钟域、时间分辨率、无 overflow 和每个 occurrence 时间线完整。每个 join 的所有时间端点必须来自同一个绑定的短窗口时钟域；`kernel_details.csv` 和 `sk_prof` 不能通过绝对时间直接拼接。

逻辑图的稳定主键是 `stage_id`、`event_edge_id`、`fork_id`、`join_id` 和 `stream_role`。原始 event 地址、Task ID、Stream ID、算子名称及 CSV 行号不得作为跨 profile 身份。

## 授权

`multistream_critical_path.py` 分开计算 event 延迟与 stage 延迟。只有稳定的 `EventNotify - producer end > 0` 才授权 event 前移；只有关键阶段 ready 后晚启动、阶段可移动且另一可靠分支存在 Cube/Vector 互补区间时才授权 stage split。两类证据不能互相替代。至少三个 occurrence、稳定关键分支和 P50 预测 E2E 可恢复上限不低于 3% 是共同门禁。

`Mix`、同资源、单流、绑定缺失、tie critical path、已隐藏的非关键分支均不得授权动作。`no_event_or_stage_candidate` 是合法 no-gain screening 结果。

## 路线和血缘

1. `event_edge_refinement`：仅有 event 延迟证据时为立即候选。
2. `stage_split`：仅有 stage 延迟证据时可直接执行；同一 join 同时存在 event 与 stage 证据时，先执行 event，stage 只在父 event 正确、机制有效、clean3 不回退但无增益后释放。
3. `scope_event_derivative`：同样只能从上述父候选派生。
4. `dependency_safe_operator_reorder`：维持既有严格独立流程。

source transform 必须使用 `superkernel-multistream-source-transform-v1`：每个 hunk 有精确半开字节区间、原始字节 SHA256、UTF-8 和 Python AST 校验，并完全位于声明的多流源码区间。一个 transform 可修改多处源码，但只能绑定一个 analyzer action ID，必须声明相同的修改前后单流投影 fingerprint、独立的修改后 dependency evidence，以及动作特定的安全证明。stage split 还必须声明显式输入、输出和跨阶段存活值。

## 机制与接受

correctness 后、profile 前，runner 使用 `multistream_event_stage_dispatch.py` 在至少三个对齐 occurrence 上验证 EventNotify 或 stage 的逻辑下发位置按 action 前移。修改后使用 `multistream_join_validation.py` 比较 baseline/candidate：至少一个 join 必须在 P50 和 P90 上提前且 join stall P90 不恶化，才是 `effective` mechanism。仅增加 C/V overlap 或只改变下发顺序不是机制成功。最终接受仍只取决于 clean E2E 增量门禁。

## 命令

```bash
python3 scripts/multistream_logical_graph.py validate --graph logical-graph.json
python3 scripts/multistream_critical_path.py analyze --capture critical-path.json --out critical-analysis.json
python3 scripts/multistream_event_stage_action.py build --analysis critical-analysis.json --graph logical-graph.json --out action-catalog.json
python3 scripts/multistream_event_stage_planner.py plan --analysis critical-analysis.json --catalog action-catalog.json --history action-history.json --max-trials 3 --out candidate-matrix.json
python3 scripts/multistream_event_stage_dispatch.py --evidence event-stage-dispatch.json --action-manifest action-manifest.json
python3 scripts/multistream_critical_path_npu_closure.py --manifest critical-path-npu-closure.json --artifact-root EXPERIMENT_ROOT
```

所有输出都必须绑定输入 fingerprint。adapter v2 只有引用真实 `superkernel-multistream-critical-path-npu-closure-v1` 才能把 event/stage 动作标为 available。当前通用代码与自动测试已完成，但现有真实 NPU closure 只覆盖 reorder，event/stage 硬件验收仍未完成。
