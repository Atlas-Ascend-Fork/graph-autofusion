# 中文结果速览数据契约

最终 worker 在 schema 2 ledger 中写入 `report_summary`，格式见
`../schemas/report-summary-v1.schema.json`。旧账本不强制迁移；缺失摘要时生成器仍出报告，
显示可用的最终指标，其余写 `N/A` 与原因。不要从中文段落猜测数字。

## 收集与选择

- `baseline` 保存冻结 S0 的 `metric`（聚合口径）、`value_ms`、独立进程 `run_count`、
  稳定性及缺失原因 `reason_zh`、相对 `evidence_artifacts`。数值缺失用 null 或省略，不能填 0。
- `stages` 保存关键候选；也可提供同阶段多项记录，由渲染器选取。阶段与 kind 必须对应：
  `stage_a/screening`、`stage_o/option`、`base_profile_source_mapping/profiling`，
  `optional_multistream`、`optional_source_range`、`optional_experiments` 均为 `optional_clean`。
- 每条记录含 `candidate_id`、`metric`、`baseline_ms`、`candidate_ms`、`run_count`、
  `status`、`eligible`、`reason_zh`、`gates_zh`、`evidence_artifacts`；可附已有 selector 的
  `selector_rank`。`eligible=true` 仅用于已测且 `status=accepted` 的阶段优胜者。
- 同阶段、同口径优先已接受候选，再选有测量的候选、selector 排名、较低耗时及 candidate ID。
  没有通过门禁的候选必须标注“最佳已测尝试，未通过门禁，非优胜者”。全部失败则给出状态行。
- clean 阶段表中的比较基线必须与摘要 S0 的数值和 metric 一致；Stage O 的增量比较留正文，
  首页使用其对冻结 S0 的同口径结果。profiling 独立列示，不能冒充 clean。
- 从实际 S0、筛选矩阵、选项试验和可选分支 artifacts 收集数据；不能只读 `experiments`，
  因为全量筛选失败的账本可能 `experiments={}`，但仍有 `screening_candidates` 和筛选 artifacts。
- 所有测量要求正的有限 ms、独立进程数和存在且不越出 artifact root 的证据。worker 必须核对
  数值与原始 artifact 内容；渲染器核对形状、路径、S0 口径及收益公式，不重新做性能判定。

## 最终测量与配置

最终行只取 `final_e2e`；摘要不能传入 `final_e2e` 阶段或 `final_clean` kind 来覆盖终局。
`beneficial/no_gain` 的 baseline/candidate median 和 improvement 校验保持原有契约。
最终 metrics 可添加 `run_count` 表示独立进程数；`sample_count` 仍表示采样数量，不自动换算。
`final_gates_zh` 写明成功阈值、P90/stddev 门禁及统计口径；没有测量则解释为何不适用。

`winner` 只允许出现在 `beneficial` 终态；其 `candidate_id`、`scope_strategy`、`option_config`
必须精确匹配 `final_e2e`。补充 `promotion_path`（`whole-scope` 或 `FINAL`）、`config_path`、
`config_fingerprint`、`reason_zh` 和 `evidence_artifacts`。

新配置的 `option_config` 应包含实际生效的 `super_kernel_optimize_options` 和
`super_kernel_debug_options` 两个 map。旧的平铺 option_config 仍可读，表示优化选项；
可用有证据的 `debug_option_config` 补充 debug map。缺失 map 显示 N/A，不能视作空 map。
失败或无收益时用相同配置字段填写最后经过验证的 `fallback`，无证据时省略并显示 N/A，
不要默认断言已恢复 SK-off。正文保留所有候选、逐 run 数据及失败现场。

## 历史报告重渲染

```bash
python3 scripts/auto_tune_session.py render-final-report \
  --session <artifact-root>/auto-tune-session.json \
  --summary <artifact-root>/report-summary.json \
  --output <artifact-root>/FINAL_E2E_REPORT.summary.md
```

`--summary` 替代本次展示使用的摘要，不写回账本、session 或封存 handoff；证据路径仍相对
session artifact root。补充必须与既有终局一致。先检查报告是否被哈希清单绑定：被绑定时
保留原件并另写 `REPORT.summary.md` / `FINAL_E2E_REPORT.summary.md`，不更新原清单来掩盖变更。
报告必须注明只更新展示，未重跑实验。诊断报告保持诊断性质，不能伪装成最终 E2E 完成。
可用 `display_note_zh` 添加展示更新说明；它只显示在配置之后，不覆盖终局判定。
