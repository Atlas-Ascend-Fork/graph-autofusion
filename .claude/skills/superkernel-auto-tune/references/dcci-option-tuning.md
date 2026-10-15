# DCCI Stage O 调优协议

本协议只约束 winner 的 Stage O DCCI option 调优。Stage O 冻结 `Sbest-BASE` 后，
Diagnostic profiling 不能作为 clean 性能证据。

## 固定顺序

1. 从不带 DCCI option 的五轮稳定 `O0-INCUMBENT` 派生唯一首轮 DCCI trial，仅设置：

   ```yaml
   dcci_disable_on_kernel:
   - .*
   ```

   `dcci_before_kernel_start` 和 `dcci_after_kernel_end` 不作为独立的显式 O trial。
   exact `dcci_disable_on_kernel=[".*"]` 必须由同一推理环境的 `ready=true` probe 接受。
2. 完成 correctness 和至少三个独立 clean process，与不带任何 DCCI option 的 O0
   直接比较。若 mean improvement 严格为正（不设固定百分比收益门槛），且通过
   median-run direction、P90 和 stddev 门禁，则补足五轮并复核稳定性，采纳
   `dcci_disable_on_kernel=[".*"]`；不再运行 before/after trial。
3. 若 disable-all 没有收益且落在噪声带内，拒绝整个 DCCI family，保留 O0，不为制造
   候选而运行 before/after。若发生可重复的性能劣化，则进入下述诊断分支。
4. correctness failure、crash、hang 或 timeout 不是性能劣化诊断入口。按失败现场协议
   保存证据并拒绝 DCCI family；不得用 profiling 掩盖功能失败。

## 劣化诊断采集

对同一个冻结 source、scope、workload、precision、TP、cache、warmup 和设备，分别采集
不带 DCCI option 与 `dcci_disable_on_kernel=[".*"]` 的 fresh diagnostic profile。
**在任一 profiling NPU 命令启动前**，必须为两侧各自的新空 profile root 执行
`artifact_contract.py begin`，并在采集结束后由同一 collection producer 执行
`finalize`。manifest/session 是采集时生成的不可变身份链，至少绑定 exact execution
config、workload、source revision、round/role、launcher command/PID/起止时间、
`ASCEND_PROF_SK_ON`、预期 artifact 和 profile-owned `sk_meta` 归档位置；不能在运行后
根据已有文件补造。

两侧都必须包含：

- profiler 的 `kernel_details.csv` 以及 collection manifest；
- profiling 进程自己产生的完整 SK metadata，包括 origin/updated graph、
  `sk_fused_nodes.log`、scope/split 和可用的 fusion failure evidence；
- 完整的 `sk_prof_<device>.json`（也可为 active runtime 实际产生的等价
  `sk_prof_x.json` 文件名），并记录 `ASCEND_PROF_SK_ON` 设置；
- exact config、workload、source revision、命令、health、fingerprint 和 declared change。

profiling producer 必须在 `finalize` 前将自己产生的 profiler 文件、SK metadata 和可选
`sk_prof` 归档至该侧 role root。只把结果目录复制到 `raw-run` 的 runner，或在结束后从
共享 `sk_meta/` 目录手工复制 metadata，均不能证明 artifact 属于该 profile process，
不得作为 DCCI 归因输入。两侧完成后，先对两份 manifest 执行
`artifact_contract.py validate-set`，再交给只读分析 Agent。

若任一 session/manifest 缺失、无效，或 artifact 不是 profile-process-owned，则这是
**采集阻断**：保留原始日志和缺失路径，停止逐 SK/child DCCI 归因，并从新的空 root 重新
采集。不得从 `raw-run`、共享 metadata 或 `sk_prof` 事后重建/修补 manifest，也不得在该
情况下输出 child regex 或执行联合修复轮。

若 `super_kernel.log` 或等价证据出现 `buffer is full, stop dump the time of nodes`，
该 child trace 不完整。保留 per-SK profiler 比较，但缩短采集窗口重新取得完整
`sk_prof` 后才能做 child 归因。

## 逐 SK 与逐 Child 归因

1. 对比两侧每一个 fused SK。跨进程使用结构 identity 和 graph occurrence，禁止用 raw
   SK/task/stream/node ID、生成 hash 或绝对时间戳直接连接。要求映射唯一、双方至少三个
   对齐的 post-warmup occurrence，并使用 P50/P90/MAD 动态阈值。
2. 输出所有 significantly slower、significantly faster 和 neutral SK；不能只报告最慢
   一个。SK interval/wall 是主判据，duration sum 只作调度或 lane 工作量诊断。
3. 对每一个显著劣化 SK，使用完整且可归属的 `sk_prof` 与同进程 metadata 按 ordered
   child position 分解。逐 child 比较 wall span、max-lane、P90 和 MAD；duration sum
   只作次要证据。child wall 或 execution 通过显著性门禁才能进入修复集合。
4. 对全部显著劣化 SK 的显著劣化 child 取 canonical op token 的稳定去重并集。报告每个
   token 来自哪些 SK/position、完整 metadata symbol、样本数、P50/P90/MAD、delta 和分类。
   不得从算子名称、邻接关系或仅 duration-sum 增长猜测修复目标。

## 联合修复轮

对 child 并集生成能匹配完整 metadata symbol、包含 canonical op token 字面量的窄 regex。
先用同一推理环境 probe 验证两个 option 的最终 exact list。修复配置必须：

```yaml
dcci_disable_on_kernel:
- .*
dcci_before_kernel_start: <全部显著劣化 child 的稳定去重 regex list>
dcci_after_kernel_end: <与 before 完全相同的 regex list>
```

这是 DCCI family 的一个联合修复 trial，允许同时改变 before/after 两个 RFC6901 Pointer；
它是普通 Stage O one-pointer 规则的唯一 DCCI 例外。不要执行 before-only、after-only、
逐 child 或排列组合 trial。配置 diff 必须证明除这两个 pointer 外均与 disable-all
诊断配置一致，并证明相对 O0 只包含完整 DCCI family 设置。

联合修复轮完成 correctness 和至少三个 clean process，并直接与五轮稳定无 DCCI O0
比较。只有达到完整 option-trial 门禁，才补足五轮并把 disable、before、after 三项作为
不可拆分的组合采纳。仅相对 disable-all 恢复、但仍不优于 O0，不能采纳。

若联合修复仍无收益、劣化、失败或不正确，拒绝整个 DCCI family 并恢复 O0。停止继续
扩展 child list；后续只有新的用户要求或新的独立证据才能创建另一轮 DCCI 调优。

## 必须报告

DCCI family 的中文报告和机器结果至少包含：

- O0、disable-all、联合修复的 clean run 数、worst-rank mean、median-run、P50、P90、
  stddev、spread、门禁与 incumbent 变化；
- 两份 diagnostic collection 的路径和 fingerprint，以及 `kernel_details.csv`、SK meta、
  `sk_prof` 的完整性状态；
- 全量逐 SK 对比统计、显著劣化 SK 表和阈值；
- 每个劣化 SK 的 child timing 表、最终 child 并集及 symbol-to-regex 映射；
- exact disable/before/after option 设置、probe evidence 和 config diff；
- 联合修复是否采纳；未采纳时说明 SK/child 分析结论和未能转化为端到端收益的事实。

不得把相关性写成 DCCI/cache 根因。只有完整联合修复相对 O0 的 clean A/B 能证明该
option 组合是否值得采纳；它仍不能单独证明某个 child 或 before/after 位置的机制因果。
