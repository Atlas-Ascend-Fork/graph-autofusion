# 实验失败现场报告契约

本契约适用于已经创建实验、round、trial 或 attempt identity 后发生的非环境类失败。
失败现场必须写入该实验自己的中文 report；父级 `REPORT.md` 可以汇总和链接，但不能
替代实验本地记录。后续重试、fallback 或新候选不得覆盖、删除或改写先前失败现场。

## 环境类例外

以下情况发生在实验启动前时，只写入 environment probe、intake blocker 和父级报告，
不要求创建实验失败现场：CANN 环境未加载、`torch_npu` runtime 无法加载、设备型号或
拓扑不受支持、运行前 option probe 未执行，以及 active wrapper 拒绝尚未启动的 exact
option value。必须保留原始 environment artifact 和错误码，不能把 `not_run` 写成实验
`failed`。

一旦实验命令已经启动，后续出现的编译、执行、正确性、超时、hang、device fault、
采集、profiling、metadata、clean timing、稳定性、映射、校准、source action、validator、
schema、ledger 或工具执行失败都不是本例外。环境变化如果只是一种待验证推测，也不能
省略实验现场。

静态编译还必须遵守
[static-compile-process-safety.md](static-compile-process-safety.md)：编译进程仍存活时，
`*_compile_error.log` 是 provisional artifact，文件存在、数量和进程时长都不能触发父
Agent、诊断脚本或清理脚本发送 kill。只有编译器自然返回后的非零状态或最终汇总才能证明
编译失败。顶层 runner 超时或外部信号导致的终止必须记录为 `timed_out/interrupted`，其
残留 error log 不得作为算子编译失败证据，必须 clean retry 后再判断。

## 先封存现场，再重试或清理

非环境类失败发生后，先停止该 attempt 的后续门禁并封存可恢复现场，再进行回退、重试、
failure isolation 或清理。至少保留：

- 原始 stdout、stderr、plog、runner/validator log，以及已产生的 profiler、`sk_meta`、
  trace、core dump 或 phase manifest；不存在的 artifact 写明 `missing` 和原因；
- 失败 attempt 使用的 source/config/control/workload、declared change 和 command
  fingerprint；报告只引用相对路径和 SHA256，不复制大段日志；
- 失败前已经持久化的中间产物和最后一个通过的 gate。禁止用重试产物补写旧 attempt。

报告中的短错误摘录只保留最早可操作错误和必要上下文，并移除 token、密码、私钥或其他
凭据。完整原始日志保持不可变并由相对 artifact 路径引用。

## 每个失败 attempt 的必填内容

实验本地 report 必须为每个失败 attempt 建立独立小节，并包含：

1. **身份与阶段**：experiment/round/trial/attempt ID，失败 phase/gate，时间，rank、device、
   PID/进程组（若已创建）。
2. **冻结调用**：exact argv、工作目录、环境设置 artifact、source revision、配置路径，
   source/config/control/workload/command fingerprints 和本轮唯一变化。
3. **可观察失败**：exit code、signal、timeout、异常类型或 correctness mismatch；给出最早
   可操作错误的短摘录和原始日志 `path:line` 或字段定位。
4. **现场 artifact**：stdout/stderr/plog、metadata、profile/trace、manifest、validator
   输出等相对路径、SHA256 与完整性状态；缺失项及缺失原因也必须列出。
5. **门禁进度**：最后通过的 gate、失败 gate、之后所有 `not_run` gate 及停止原因。
6. **初步判断**：已证明事实、待验证假设、归因 confidence 和至少一个仍可能的备选解释；
   不得把邻近日志、局部性能或环境猜测写成根因。
7. **控制与回退**：是否终止进程组、释放设备租约、恢复 immutable incumbent、隔离可变
   worktree/cache，以及失败是否允许继续同一矩阵中的其他独立 trial。
8. **后续定位入口**：failure-isolation 结果或请求路径、建议复现命令、仍缺证据和下一步。

对于多次重试，按 attempt 顺序逐项记录，指出下一 attempt 相对上一 attempt 的唯一变化。
最终成功只能追加新结果，不能删除历史失败小节。

## 各类报告的落点

- S 候选、O trial、BASE/SMAP/P/FINAL 和 whole-scope：写入其候选或 winner
  `EXPERIMENT_REPORT.md` 的对应 round/attempt 小节。
- 多流调优：写入 `MULTISTREAM_TUNING.md` 的对应 trial/phase 小节。
- profiling 分析或校验失败：写入拥有该 profiling round 的 `EXPERIMENT_REPORT.md`，并
  链接 analysis Agent 的 stdout/stderr、请求和已有输出；不得伪造完整 analysis result。
- SK runtime/correctness/timeout/hang/device failure：把 failure-isolation 输出原样绑定回
  失败实验小节。

最终父级 `REPORT.md` 必须在失败索引中列出每个非环境失败的实验 ID、本地 report 路径、
terminal status、fallback identity 和下一定位入口。
