# 隔离多流实验生命周期

## 冻结 incumbent

分支入口冻结 `Sbest-BASE` 的 source/config/control/workload、五次 clean 汇总、profile 和
analysis fingerprint。所有 trial 从这个 immutable identity 派生；不得把被拒绝 trial
残留到下一轮。

## Trial 类型

### Option-only

只修改一个 RFC6901 pointer 的一个 exact accepted value。读取 Stage O 历史，禁止重复
完全相同的全局 option/value。全局 option 即使由一个 target 触发，也必须重新分析全部
可靠 SK，不能称为 range-local change。

### Dependency-aligned split

只修改一个 `source_scope_map_v2 + exact` target 的一个 scope 边界。保持业务算子顺序、
stream、DATA、STREAM_ORDER、event、wait、barrier、通信和 cache mutation。目标是让
原有独立 stream 保持调度自由，不是人工交换算子。

### Exact range exclusion

只改变一个 source-exact range 的 fusion participation。可以使用 active wrapper 已证明
接受的平衡 `None` exclusion，或在目标前后关闭/恢复同名 outer scope；一个 trial 只能
使用一种机制。

### Dependency-safe operator reorder（仅多流）

只在一个 `source_scope_map_v2 + exact` range 内重排完整、连续、显式无副作用的 Python
statement block。授权证据必须同时证明：SK-off 至少三个 occurrence 存在稳定的跨 stream
`Cube <-> Vector` overlap，SK-on `sk_meta` 对相同 operator identity 存在稳定 dispatch
inversion，所有 stream identity 完整。`Mix` pair、同类计算引擎 pair、通信/wait pair 直接
拒绝；单流 target 不得进入该动作。

路线 2 先生成满足 hard dependency DAG 的最小 permutation。路线 3 只在路线 2 已实际改变
下发序但 clean3 无端到端增量收益时，启用预先冻结的 bounded topology 候选；物化路线 3
必须同时绑定路线 2 的 passed dispatch evidence 和 rejected clean3 semantic evidence。任何路线
都不得跨越已观测稳定执行偏序、DATA、STREAM_ORDER、event、wait、barrier、通信、cache
mutation、副作用或控制流依赖。

### MIX component overlap reorder（仅受审例外）

普通 reorder 仍拒绝 whole-MIX。只有完整绑定的短 trace 暴露 MIX 的全部 AIC/AIV component、
目标互补 component 存在稳定可恢复重叠、其余同引擎竞争已量化、完整 hard dependency DAG
允许变换时，才可请求 `component_overlap_reorder`。跨函数修改必须绑定多个 exact AST span，
只能由冻结 adapter 指定的受审 multi-hunk materializer 执行，并且保持单流投影、事件、wait、
通信和 child set。映射必须在只读 `component_immutable_source_root` 上复验，该目录必须与可写
候选 `source_worktree` 分离。correctness 后必须先验证所有受影响 target 的 component dispatch evidence，
再启动四 profile 与 clean3/clean5。

对该例外，profile gate 复验完整 four-profile summary，analysis gate 复验 component
performance analysis 的 target/occurrence 覆盖和数值完整性。accepted result 仍必须使用
artifact-root 内的 source snapshot，不得以隔离 worktree 的绝对路径替代 source identity。

## Trial 状态

```text
accepted | rejected | blocked | failed
```

trial `accepted` 只表示该 trial 完成所有门禁；只有 result 顶层也为 `accepted` 且合同
校验通过，父流程才可把它作为新派生 family 的输入。
correctness、profile、clean 任一失败均不得继续补齐其他门禁来挽救本 trial。
对 reorder trial，correctness 后还必须先通过 `dispatch_order_evidence`；runner 在该证据缺失、
下发序未改变、child set 改变或不再是多流时，不得启动 profile。

## 接受矩阵

| 局部调度 | 目标 SK | clean E2E | 结果 |
|---|---|---|---|
| 改善 | 改善 | 增量改善 | 可 accepted |
| 改善 | 改善 | 不变或回归 | rejected，保留机制证据 |
| 未改善 | 改善 | 增量改善 | 可 accepted，但 mechanism=unproven |
| 改善 | 未改善 | 增量改善 | 可 accepted，但不得宣称目标 SK 修复 |
| 任意 | 任意 | correctness/P90/stddev 失败 | failed/rejected |

潜在 accepted trial 先有至少三次 clean，再补到 exactly five 并验证稳定性。接受后不要把
本分支 trial 直接写进旧 optimizer ledger。父流程从返回 source/config 创建新的实验 family，
用标准 fresh `*-BASE` 重新建立可进入 SMAP/P/FINAL 的证据。

## 终止条件

- 请求预算耗尽；
- 三次短窗口重采仍 overflow 或无法归属；
- 所有合法 trial 已结算；
- correctness/runtime failure；
- source action 缺 exact map；
- reorder target 缺直接多流证据、稳定 inversion、statement exact map 或依赖证书；
- route 2 未改变编译后的下发序；
- option 已在 Stage O 以相同 payload 结算；
- 没有 trial 达到 incumbent 增量 clean 门禁。

终止时生成合法 fallback result，不删除失败 artifact，不修改 incumbent。非环境类失败必须
先按 [失败现场报告契约](../../superkernel-auto-tune/references/failure-scene-reporting.md)
封存 attempt 现场，并写入本分支 `MULTISTREAM_TUNING.md` 的对应 trial/phase 小节；重试或
fallback 不得覆盖该小节。实验启动前的环境失败只进入 environment evidence。

执行前使用 `multistream_candidate_planner.py` 冻结
`superkernel-multistream-candidate-matrix-v1`。矩阵只能读取受审 action catalog；不得从诊断
文字自动发明 option value 或 source insertion。Stage O 完全相同的 global pointer/value
进入 deduplicated list，不占 trial budget。
