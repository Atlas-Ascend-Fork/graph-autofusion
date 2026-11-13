# 工作流

先按输入分流：已有 Graph/DFX HTML 且请求 AI 汇报时，使用 [ai-report.md](ai-report.md) 读取全部 Model 并按模板总结，不把 HTML 传给目录解析 CLI。遇到旧格式或解析／展示异常时，按 [compatibility-repair.md](compatibility-repair.md) 做最小修复和验证。以下流程用于原始结果目录。

1. 从用户结果目录发现 JSON/Log 资产和模型边界；触发分析时选择 `--parser both|json|log`，默认 Both。
2. 两侧独立提取，不使用另一侧事实修补自身后再比较。不同模型实例分开组织。
3. 标准化对象、来源与覆盖范围；同阶段共同字段双向校验。有效 JSON 优先，不可用时按能力回退，局部有效时保留且标记不完整。
4. 默认只生成用户版 Graph / DFX 单 HTML，超阈值附加压缩包。需要机器结果时用 `--export-summary` 导出 `data/run-summary.json`；需要两侧解析快照时用 `--export-parsers`，缓存读写用 `--cache` 开启。这些选项彼此独立。Graph 不提供解析器切换，只提供 Update 状态切换；需要原始开发排查信息时，使用 `--dfx-debug` 生成 `run-portal-debug.html`，无需同时导出 JSON。
5. 从整图统计进入 scope 原始排布，再进入 SK 队列。DFX 是 Graph 拓扑看板：先看融合是否正常和 ERROR / Warning 数量，再按告警卡进入 Graph 定位；关系补充和解析器内部状态不在用户看板占用空间。
6. 对外回答结论、证据、受影响对象、完整度和下一步。真实耗时当前不分析。
7. 按配置压缩 HTML。报告离线可用；缓存和原始资产不自动进入压缩包。

Scope/Task 为 Graph 内部下钻能力。`diagnose-hang-crash` 生成相同统一报告。
