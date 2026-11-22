# 公式分解与证据边界

根据估值偏差所在的 pipe，追溯模型实现和实际执行路径。

| 检查项 | 核对内容 |
| --- | --- |
| 数据搬运 | 模型中的数据量、维度和 stride 与生成 kernel 的合轴、循环及 API 参数是否一致 |
| 基础成本和延迟 | 性能表的平台与拟合范围是否适用，核数、频率和 profiling 聚合口径是否一致 |
| 执行次数 | 模型乘数与 codegen 实际循环次数是否一致，是否覆盖边界 tile 和尾块 |
| result 级代价 | 各 group 的 objective、串行边集合及罚项是否符合实际串行或并行调度分支 |

建议先用 `rg -n 'LoadStoreStrideV2|StorePerf|LoadPerf|pipe_perf_expr' autofuse/att autofuse/v35/att`
定位模型实现，再从当前 codegen 和生成 kernel 追溯实际调用。检索入口随代码变化时，
沿源码调用链继续查找。

对每个待核对项记录符号公式、变量来源、代入结果及其占总估值的比例。
确认统计口径一致后，再判断偏差来自哪一项成本。

对搬运模型，沿 codegen 到实际 API 调用记录搬运字节数、维度合并、源/目的 stride 及其单位、
循环与 repeat 次数、有效核数、核间分工和尾块。将单次 API 成本与外层执行次数分开验算，
同时检查尾块是否走了不同 API 或参数分支。公式代入结果应能还原日志中的 pipe 估值；
无法还原时先定位变量、路径或版本差异，再做拟合。

## 排序归因

估值倍差用于定位可疑项，是否导致选错需要通过候选排序验证。对已有实测的快慢候选：

1. 固定两者完整身份和 tiling，保留原始各 pipe、各 group 及 result objective。
2. 分别只替换一个 pipe 的实测成本，再按需要联合替换。每种情况都使用实际调度分支、
   group 聚合方式和边罚项重算，记录胜出身份、objective 及两候选差值。
3. 如果某 pipe 虽倍差大，单独修正后仍选慢候选，将其记录为偏差项；
   能改变排序的项才有对应的排序归因证据。修改引发换模板不等于该 pipe 是唯一根因。
4. 修改模型后重新求解，对照新的 tiling、循环和 API 路径，再实测默认选择。
   固定 tiling 下的替换分析不能替代重新求解后的验证。

仅用可比较的 pipe 实测值进行替换；多 group 无法拆分、采样口径不同或调用归属不明时，
相应实验标为不可执行，不将整核实测成本填入单个 group。测量分布重叠时说明排序的不确定性。

## Pipe 名称兼容

生产端使用 `AIV_MTE2`、`AIV_MTE3`、`AIV_VEC`。离线脚本同时接受 `M2`、`M3`、`V`，
分别映射到上述规范名称；同一字段同时出现两种名称时，以规范名称的值为准。
该规则沿用仓内 [evidence.py](../../../../autofuse/tools/att_analyze/src/commands/evidence.py) 和
[summary_templates.py](../../../../autofuse/tools/att_analyze/src/summary_templates.py) 的 `_PIPE_NAMES` 兼容映射。

## 对照表状态

- `candidate_only`：候选估值记录，不关联实测；缺少 FINAL 时仍可生成候选表。
- `invalid_candidate_estimate`：候选数值转换后为非有限值，例如科学计数值溢出为无穷大。
  核对原始日志和模型计算，排除该候选值，不用于排序或倍差分析。
- `unsupported_final_source`：FINAL 的 source/selection_mode 组合不在当前支持的
  `runtime/default`、`runtime/explicit`、`pgo/pgo` 中。保留原始身份，实测和倍差留空；
  核对日志来源及生产端版本，确认契约后再决定是否扩展支持，不将未知来源改写成默认选择。
- `ambiguous_final`：同一 run/operator 有多个最终记录，可能是多个 group、shape、调用或 subcase。
  按调用和完整身份隔离后再关联实测。
- `missing_profile` / `ambiguous_profile`：没有唯一的 profiling 行。核对 Op Name 和调用范围，
  名称映射用于处理命名差异，重复调用还需单独确认归属。
- `invalid_or_missing_metric`：AIV cycle、block 或该 pipe ratio 缺失或无效，实测留空。
- `missing_estimate`：最终记录未提供该 pipe 估值，不借用非最终候选估值。
- `conflicting_final_summary`：result summary 与最终 group 的身份不一致，先补齐或隔离日志。
- `invalid_estimate`：pipe 估值为负数或非法类型，不计算倍差。
- `zero_measured`：有效实测为 0，保留数值，倍差留空。
- 其他 FINAL 解析状态沿用仓内解析器；损坏或不完整的分片需补齐后再确认最终选择。

脚本按 run 保留每条候选记录及其来源，支持科学计数法及 `R0` 标签。
候选 CaseLabel 和最终 SubCaseTag 分别记录，关联时核对精确映射。
重复测量先按设备、输入、调用和最终身份分组，再选择统计方法。
同一条对照记录中的各 pipe 实测值应来自同一次采样。
保留每个样本的 cycle、block、ratio 和原始 CSV 行号，先逐行计算 pipe cycle。
可以分别汇总各 pipe 的统计量，但用于联合替换的整组 pipe 值应取自同一样本，
或逐样本复算排序后汇总结果，避免把不同采样的中位数组合成一次虚构测量。
