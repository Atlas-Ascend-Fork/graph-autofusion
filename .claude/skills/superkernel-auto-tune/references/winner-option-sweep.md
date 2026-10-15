# Winner 选项探索协议

本协议用于 S 候选粗选完成后、普通 winner profiling、源码映射或 P 深度调优
之前。目标是用端到端 clean 性能实测筛出值得保留的全局编译选项。普通 option 不等待
profiling 假设；DCCI 仅在 disable-all 已实测劣化时进入一次条件诊断 profiling。

## 生命周期位置

```text
S0 -> S1/S2/S3/S4 clean screening -> Sbest-SEED
   -> O0-INCUMBENT -> O1/O2/... -> Sbest-BASE
   -> [optional isolated multistream branch -> derived-family BASE]
   -> [Sbest-SMAP] -> whole_scope_clean_validation

optional only when explicitly requested by the user or frozen experiment plan:
Sbest-P* -> Sbest-FINAL
```

`Sbest-SEED` 是粗选胜出脚本。`Sbest-BASE` 是全部 O 轮结算后保留下来的脚本和选项

## 冻结选项矩阵

在 O1 启动前冻结 `superkernel-winner-option-matrix-v1`。矩阵至少覆盖以下类别：

- `dcci`：一个条件 family。唯一首轮值是 active wrapper 接受的
  `dcci_disable_on_kernel=[".*"]`；`dcci_before_kernel_start` 和
  `dcci_after_kernel_end` 不各自形成显式矩阵 trial。只有 disable-all 劣化时，才由 paired
  SK/child profiling 生成一个 before+after 联合修复值；
- `auto_op_parallel`：active wrapper 接受的非基线值，通常为 `1`；
- `aggressive`：active wrapper 接受的 `aggressive_opt_strategies` 非基线值，
  每个 task/value/event breaker 组合必须拆成独立 trial；
- `other_experimental`：例如 active wrapper 接受且已满足运行前提的 `early_start`。

每项必须包含 `id`、`category`、`option`、`value`、配置中的 RFC6901 JSON Pointer、
原值、probe evidence、风险等级和最终状态。最终状态只能是 `accepted`、`rejected`、
`failed`、`blocked` 或 `skipped`。`blocked/skipped` 必须给出中文原因和 evidence；
“profiling 没有直接证据”不是合法原因。wrapper 未接受 exact value、缺少运行前提、
用户未授权对应风险或预算已明确耗尽可以形成 blocker。

不得把 `check_environment.py` 内置样例当成网络专用候选全集。DCCI disable-all 固定只
探测 exact `[".*"]`。条件联合修复的窄正则必须来自 fresh regressed-child evidence 和
当前编译产物的真实 symbol inventory，并通过 `--option-values-json` 交给同一推理环境中
的 probe；不得硬编码模型名、固定层数或预设某个网络独有算子。只消费成功
`ready=true` 报告中的 exact `accepted_values`。

## O0 与逐项试验

1. 从 `Sbest-SEED` 复制不带新增选项的 `O0-INCUMBENT`，运行完整 correctness 和五个
   独立 clean process。要求 worst-rank mean spread 不超过 5%。筛选阶段的三次样本
   不得代替 O0。
2. 按冻结矩阵顺序处理每个 option/value。普通 O 轮只相对当前 incumbent 修改一个
   RFC6901 Pointer；不得改变 scope、源码、workload、precision、TP、cache、warmup、
   设备或其他运行控制。DCCI 联合修复是唯一例外，允许同时设置 before/after 两个
   pointer，但必须保留 disable-all 且两者使用相同的 evidence-derived regex list。
3. 每个 clean trial 先执行完整推理和 correctness，再运行至少三个独立 clean process。
   profiler、SK metadata、`sk_prof`、trace、debug sync 和 calibration 必须关闭。
4. 将 trial 与当前五次稳定 incumbent 比较：均值增量必须严格为正（不设固定百分比
   收益门槛）、median-run 方向为正、P90 和标准差不劣化时才 `accepted`。调用：

```bash
python3 <runtime-skill-dir>/scripts/analyze_performance.py \
  --baseline experiments/Sbest/O0-INCUMBENT/CLEAN \
  --candidate O1=experiments/Sbest/O1/CLEAN \
  --option-trial --warmup 8 \
  --json-out experiments/Sbest/O1/option-trial-summary.json
```

5. accepted trial 把该 exact value 累积到 incumbent；补足到恰好五个 clean process，
   再验证 spread 不超过 5%，之后才能作为下一 O 轮 baseline。未通过稳定性复核则撤销
   接受并标记 `failed`。
6. rejected/failed trial 立即恢复上一个 incumbent。保留日志、配置 diff、正确性、
   超时/崩溃和性能证据，不得继续携带该值。`failed` 若不是实验启动前的环境原因，必须
   先按 [failure-scene-reporting.md](failure-scene-reporting.md) 在 winner
   `EXPERIMENT_REPORT.md` 的该 O trial/attempt 小节记录现场；后续 O 轮不能覆盖它。
7. 全部矩阵项结算后，把最终 incumbent 的源码、scope、配置、选项、command 和五类
   fingerprint 冻结为 `Sbest-BASE`。若没有选项获益，`Sbest-BASE` 与
   `Sbest-SEED` 等价，但仍需记录完整 O 矩阵。

## DCCI 条件分支

完整执行 [dcci-option-tuning.md](dcci-option-tuning.md)：

1. 只运行 `dcci_disable_on_kernel=[".*"]`，直接与无 DCCI 的五轮 O0 对比。
2. 有收益则补足五轮并采纳；neutral/no-gain 则拒绝，不运行 before/after。
3. 劣化时，分别为 O0 和 disable-all 打开 profiler、SK metadata 和完整
   `sk_prof_<device>.json`。这些是 diagnostic process，不计 clean 样本。
4. 全量对比每一个 SK，再分析每一个显著劣化 SK 内的 child wall/execution。取全部显著
   劣化 child canonical op 的稳定去重并集，生成同一组 exact 窄 regex。
5. 保留 disable-all，并把该 regex list 同时设置到 `dcci_before_kernel_start` 与
   `dcci_after_kernel_end`，只运行一次联合修复 clean trial。不要运行 before-only、
   after-only、逐 child 或排列组合实验。
6. 联合修复必须直接优于 O0 并通过完整门禁才采纳三项组合；仅优于 disable-all 不够。
   无收益时拒绝整个 DCCI family，并在报告中保留逐 SK、逐 child 与 exact option 证据。

## 冒险选项安全边界

`aggressive_opt_strategies` 的每个 accepted value 必须在独立进程中运行，设置明确
timeout，并在 hang、device error、crash 或 correctness failure 后停止该 trial、回退
incumbent，再按 failure-isolation skill 读取 fresh plog。失败不能据此跳过矩阵中的
其他独立 option/value。

`event_breaker_bypass` 只有在 event 语义已经审查且用户允许该风险时运行；否则以
风险授权 blocker 结算。`early_start=1` 缺少配套 operator-side adaptation 时结算为
blocked，而不是把“wrapper 接受”误写成“运行语义已满足”。Debug options 永不进入 O
矩阵或 clean 样本。

## 与 P/FINAL 的边界

O 轮是 winner 全局选项探索，因此不需要某个 fused SK 的直接 scalar/cache、
Cube/Vector 或 breaker 诊断证据。它仍需要 accepted-value、correctness 和 clean 增量
收益。

当用户或冻结实验计划已明确请求可选 P 分支时，P 仅裁剪 BASE 分析已证明的
neutral/regressed exact range。`Sbest-BASE` Option map 在 P/FINAL 中保持冻结，不得再次
调优。P 必须使用 fresh profile 与独立分析 Agent；O 轮收益不能替代某个局部 range 的
直接性能与源码映射证据，也不能自动证明其因果归属。
