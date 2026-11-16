# 依赖边界

默认报告依赖 Python 和用户提供的 SK 资产，不依赖网络服务或 AI 后端。HTML 内嵌脚本、样式和展示数据，接收者只需浏览器；维护环境 Playwright/Chromium 用于交互验收，不是用户运行依赖。

JSON 路径只读取 sk_graph_origin.json、sk_graph_updated.json、sk_task_queue.json。Log 路径复用旧日志解析；scope 与融合等日志特有事实在 Both 中按明确身份补充。大 event/prof JSON 暂不进入新 Graph/DFX 主链。

目标仓无需携带设计文档、维护测试或固定真实样例。参考事实优先来自 skill references 和结构化运行产物。发布必须包含 scripts/graph_report/assets/，不能只拷贝 Python 文件。

当前局限：Log 未提供完整最终图时为部分还原；同步代际不明确时不生成确定连边；多实例 JSON 无身份时单独展示；真实耗时和自动性能归因暂不提供。
