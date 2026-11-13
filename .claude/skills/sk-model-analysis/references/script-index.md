# 脚本索引

- `scripts/model_analysis.py`：公共 CLI，analyze/hang 接入统一 Graph/DFX。
- `scripts/graph_report/json_parser.py`：独立 JSON 资产解析和校验。
- `scripts/graph_report/log_parser.py`：独立日志适配、实例隔离与运行诊断证据。
- `scripts/fusion_analysis.py`：可选融合断图证据解析、算子影响统计与日志覆盖状态；不参与图布局。
- `scripts/graph_report/compare.py`：共同事实比较、主来源选择、统计及变换映射。
- `scripts/graph_report/pipeline.py`：资产发现、模型调度、内容缓存和标准报告。
- `scripts/graph_report/cli.py`：输出目录、阶段进度、机器 JSON 与 HTML 生成。
- `scripts/graph_report/dfx_projection.py`：按用户版／调试版投影 DFX 结论、可信范围和原始证据。
- `scripts/graph_report/render.py`、`assets/`：单 HTML 组装、Canvas 图和精简 DFX，资源发布时必须一起携带。
- `scripts/graph_report/archive.py`：原子生成 ZIP/tar.gz，只打包离线报告。
- `scripts/sk_library_extractor.py`：既有日志事实提取和信息库工具；Log adapter 关闭 JSON shadow 防止交叉污染。
- `scripts/diagnose_run.py`：保留历史诊断辅助函数供适配器复用；CLI 转发新统一入口。
- `scripts/sk_scope_visualizer.py` / `sk_task_queue_visualizer.py` / `sk_visualizer_shared.py`：历史工具及内部依赖，新主流程不生成这些独立页面。
- `scripts/regression_runner.py`：只读样例、在临时输出目录执行新报告回归。
