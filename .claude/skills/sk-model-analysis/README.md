# SK 模型分析

对 SK 运行结果进行完整执行图与 DFX 分析，不依赖 AI 或网络服务。

## 快速使用

```bash
python scripts/model_analysis.py analyze <result_root_or_model_dir> --output <report_dir>
```

默认使用 `--parser both`：JSON 和 Log 独立解析后比较，有效 JSON 为主。也可显式选择 `--parser json` 或 `--parser log`；单模式不读取另一种来源补齐。模式在执行时选择。

打开 `<report_dir>/run-portal.html` 即可离线查看：

- **Graph**：固定水平 Stream 泳道，连续 RESET 默认合并，支持类型与 Stream 过滤。单击高亮关联，双击查看详情，连续 RESET 可以按钮展开；单个 `Update` 切换更新前后。跨流 Scope 成员对齐后使用完整矩形边界，标明 SK 关联状态，SK 可查看内部 AIC/AIV 队列。整图 Event / value 默认以高亮呈现，SK 内部 sync 保留连线。只由 Scope begin/end/placeholder 边界节点组成的内部范围从用户图中过滤；实际刷新任务的单 Kernel SK 仍按普通绿色 SK 保留、可搜索和下钻。
- **DFX 看板**：先给出当前 Model 的解析情况，同时显示解析日志中的 ERROR / Warning 数量。没有日志问题时，“融合概览”直接展示实际融合算子 / 期望融合算子、形成的 SK 数量和真正未融合的算子；算子导致融合断开、同步信号导致融合断开作为融合概览证据，不重复生成告警卡。期望融合集合来自更新前的真实 KERNEL，只有日志明确标记 `NOT_IN_SCOPE` 的记录才可排除；`OP_UNSUPPORT`、资源不足等融合失败仍计入期望集合和未融合列表。`FORMAT_ERROR` 等解析器内部状态、关系补充信息和 INVALID 保留信息不进入用户看板。默认页面不展示开发排查信息。

需要查看原始解析证据时显式生成调试版：

```bash
python scripts/model_analysis.py analyze <input> --dfx-debug --output <debug_report_dir>
```

调试版输出 `run-portal-debug.html`，包含原始诊断、来源路径和解析器交叉校验；这些内容不会出现在默认 `run-portal.html` 中。

当前暂不展示重复结构识别与折叠。Scope 默认折叠，框头提供折叠/展开和独立 Update，工具栏支持全部折叠。局部 Update 只改变当前 Scope 内部排布；全局 Update 统一切换并清除局部覆盖。没有明确更新后数据时局部 Update 不可用。

原 Analysis View 已下线；Scope、Task 能力合并至 Graph。当前不展示真实执行时长，不提供性能热点或耗时归因。

## 通过 skill 生成 AI 汇报

把已有 HTML 交给加载了本 skill 的 agent，例如：

> 用 sk-model-analysis 分析 `build/my-report/run-portal.html`，汇报全部进程／Model 的融合情况，找出主要影响算子、未融合原因和实际拆分位置，给出处理建议。

agent 会提取 HTML 内嵌的完整 Model 数据，按统一口径汇总，并结合具体节点与更新记录解释原因。无需 API Key、模型后端或新的 Python AI 命令；可以只提供 HTML，不要求重新提供原始 sk_meta。临时解压与计数不承担 AI 判断。

默认在 `build/model-analysis-ai-report/<本轮任务名>/ai-summary.md` 保存完整汇报，对话中给出核心结论和路径。原 Graph/DFX HTML 不变；需要独立汇报页面时可另外指定。用户版没有携带的原始诊断不会被当成零或由 AI 补造。

旧版数据遇到明确的解析或展示兼容问题时，agent 可根据实际样本局部修复、重生成并验证；只有 HTML 时不会假装恢复丢失的原始日志。分析方法和模板见 [AI 汇报](references/ai-report.md)，修复策略见 [兼容诊断](references/compatibility-repair.md)。

## 输入与证据

支持结果根目录、`sk_meta`、单 model 目录及多进程/模型/实例。JSON 路径读取 `sk_graph_origin.json`、`sk_graph_updated.json`、`sk_task_queue.json`；Log 路径复用 SK 日志提取，保持独立。

JSON 原图和更新图是两个真实快照。日志 update 只能证明局部修改时，更新图标为部分还原。相同 event/address 不能自动代表确定依赖。报告能生成不代表诊断完整，未知也不表示零节点或模型正常。

图语义会把“结构性纯 Scope”与“单 Kernel 更新”分开：只有全部成员都能确认是 Scope begin/end/placeholder 时才过滤其边界；成员不完整或名称无法确认时保留。`data/run-summary.json` 的 `graph_semantics` 和统计字段保留两类数量，便于机器侧解释用户图与原始快照的差异。

scope 成员、融合成员等当前主要来自日志，因此 JSON 单模式可能无法展示完整 scope 下钻。Both 可以补充单侧信息并保留来源。多次 optimize 的 JSON 无明确实例身份时独立展示，避免误拼。

## 参数和产物

```bash
python scripts/model_analysis.py analyze <input> --parser both --jobs 4 --profile
python scripts/model_analysis.py analyze <input> --export-summary --export-parsers --cache --output <report_dir>
python scripts/model_analysis.py analyze <input> --no-cache --no-parallel
python scripts/model_analysis.py analyze <input> --archive zip --output <report_dir>
python scripts/model_analysis.py analyze <input> --archive tar.gz --output <report_dir>
python scripts/model_analysis.py analyze <input> --dfx-debug --output <debug_report_dir>
```

- `--output`：输出目录，不修改输入资产。省略时沿用推断结果根目录下的 `reports/`。
- `--jobs`：模型并行数，自动默认上限 4；Both 不强制两解析器同时占用进程。
- `--export-summary`：导出 `data/run-summary.json`，用于脚本分析或重新渲染；现有消费该文件的调用需补上此参数。
- `--export-parsers`：导出 `data/parser-results.json`，用于两侧解析证据排查，与摘要导出独立。
- `--cache`：开启缓存读写，缓存基于来源内容、解析模式及代码指纹；默认关闭。
- `--no-cache`：明确禁用缓存读写，与默认行为一致。
- `--profile`：单独导出工具执行耗时，不自动导出摘要、解析快照或缓存。
- `--archive auto|none|zip|tar.gz`：默认 auto；HTML 超过 `--archive-threshold-mb`（默认 50 MiB）时额外生成 ZIP。
- `--dfx-debug`：生成独立的 `run-portal-debug.html`，只用于开发排查；默认用户版不包含调试数据和调试入口。
- `--with-ai`：仅保留兼容提示，未接入 AI 后端；agent 的 AI 汇报由 skill 工作流完成，不使用此参数。

```text
report_dir/
  run-portal.html            # 默认用户版，内嵌 Graph 与用户 DFX
  run-portal-debug.html      # --dfx-debug 时生成的开发排查版
  run-portal.zip             # 按配置生成，也可为 .tar.gz
  data/run-summary.json       # --export-summary：图、结论、完整度、校验和来源
  data/parser-results.json    # --export-parsers：两侧独立解析结果
  data/diagnose-profile.json  # --profile 时生成
  .cache/                     # --cache：可重建解析缓存
```

默认只生成 `run-portal.html`；超出阈值时额外生成压缩包。HTML 包含完整浏览数据，可以单独交付并用于 AI 汇报，Graph、DFX、多 Model、内部任务和关联功能均不依赖外部 JSON 或缓存。复用旧输出目录不会自动清理或重写未选择导出的历史文件；干净交付请指定新目录。

压缩包默认仅包含单 HTML，不包含原始日志、二进制、缓存或样例。压缩解决传输体积，解压后打开 HTML。使用第三方图像或网络字体不是报告运行前提。

退出码：0 为已产出可用结果（可以包含运行异常或解析降级）；1 为无可用分析结果或生成/交付失败；2 为参数错误或已下线命令。详细状态见 HTML 的 DFX 和生成状态，显式导出的 run-summary 保留完整机器结果。

## 其他入口

```bash
python scripts/model_analysis.py diagnose-hang-crash <input> --output <report_dir>
python scripts/model_analysis.py trace-nodes <model_dir> -o <node-trace.json>
python scripts/model_analysis.py trace-nodes <model_dir> --model-instance-id mi01 -o <node-trace.json>
```

`diagnose-hang-crash` 同样输出统一报告。`trace-nodes` 是独立导出工具，不自动生成旧页面。`diagnose-performance` 明确返回下线说明。

## 验证与维护

```bash
python scripts/model_analysis.py --help
python scripts/regression_runner.py <sample_root>
```

维护环境应运行对应单元、集成和离线浏览器回归；Playwright/Chromium 仅用于维护侧验收，不是报告接收者依赖。

源码职责见 [references/script-index.md](references/script-index.md)，来源和降级见 [references/artifact-contract.md](references/artifact-contract.md)。维护侧真实样例不随 skill 发布。

Update 保留明确标为 INVALID 的节点及原始来源，但用户 Graph 的更新后拓扑和统计默认排除它们；原始节点总数仍在结构化结果中保留，`visible_after_nodes` 表示用户可见拓扑数量。折叠后的 SK 区域按融合成员数显示浅至深的绿色色阶，同 Model 使用统一尺度，数字标签保留。

节点显示过滤与 Update 阶段切换独立：INVALID 和 RESET 默认过滤，Scope 默认不过滤；图上方提供明确的“过滤 INVALID”和“过滤 RESET”复选框。过滤应用于整图及局部 Scope 内容，只改变显示，不删除源数据，不改接依赖；切换 Update 保留过滤状态。RESET 的显示过滤与连续 RESET 聚合分别控制。

“过滤 Scope”默认关闭，隐藏 Scope 容器、折叠卡片及明确命名的 Scope begin/end/placeholder Kernel，保留其他内部 Task/SK；不是删除整个 Scope 的任务。关闭后恢复各 Scope 保存的折叠状态。全图与局部 Update 的节点阶段不因此改变。


缩放始终使用同一执行图坐标与绘制路径，不切换卡片索引或按 Stream 拆分 Scope。缩小后完整 Scope 区域及融合色阶保持一致，只按比例隐藏文字和控件，避免固定宽边框覆盖融合颜色。

节点语义颜色：明确融合成员与 SK 为绿色；Scope begin/end/placeholder 为灰色；未融合 Kernel 使用普通 Kernel 样式，融合失败和断开原因继续通过 DFX 与选中联动呈现。用户手动 Scope 使用独立轻量虚线边框，默认显示，由“整图操作”的“显示用户 Scope 边框”开关控制；不改变布局或节点语义色。同 Stream 的顺序边始终水平直连，不因 Scope 对齐产生的间距绕折线；跨流 Event/value 仍按真实依赖路由。

Scope 的结构浏览统一在整图：框头及选中操作栏提供折叠/展开和局部 Update；选中操作栏及 SK 节点按钮支持向下展开内部队列（多个 SK 时显式选择）。Scope 双击详情只保留状态、成员/类型统计、融合记录和来源证据，不再复制 Stream 排布或更新后跳转列表。查看 SK 内部不改变全局/局部 Update 或折叠状态，返回恢复原详情及证据展开状态。SK 内部以纯图结构下挂展示，只保留归属标题、收起入口、队列节点与依赖边，不嵌入搜索、分页、统计或证据面板；完整队列共享整图拖动与缩放，允许多个同时展开、独立收起；展开只扩展下方空间，不移动主图节点。虚线挂接表示归属，内部 sync 连线表示依赖。内部单击高亮关联，双击打开 Task 参数、来源、上下游依赖及明确映射外部关联的悬浮详情；详情关联项可定位对应任务；主图双击同样只看节点详情。下挂子图明确标注更新后 AIC/AIV 队列，Update 保留展开状态，切换 Model 清空展开。

跨层关联独立于执行依赖：FUNC 来源通过融合成员索引与函数地址交叉核对；PRELOAD/SYNC 使用同队列来源标记表达配套关系，不增加依赖箭头。内部 Event 的 args 与同一 Model 实例更新后 value 事件 devAddr 建立地址关联，区分等待、写入和值，轮次未知保留说明并汇总 DFX。内部默认关联对齐，支持切回原始排布；对齐保留各队列顺序。主图 Kernel、外部 value 节点均可展开对应内部任务，双向高亮；Scope/SK 入口继续保留。

内部图保留类型颜色与轮廓、水平队列基线及独立过滤。SET/WAIT 使用动态虚线，ALL 使用实线；不显示问号或一对多候选连线。SYNC 按固定方向 flag 每次 WAIT 消耗一次 SET 的规则，恢复重复信号的一对一配对；不使用 nodeIndex。配对元数据包含 flag_counter_fifo 依据、ALL 区间、序号和对端 Task。缺端点、数量不符或队列错误用红色标记，解析不足不标成运行异常。每个 SK 的内部过滤入口位于随子图缩放的工具条，独立保存显示类型、仅看关联和对齐；收起重开恢复。批量应用是菜单中的显式操作。

Scope/SK 单击可汇总内部任务的外部关联，穿透跨 SK 信号、融合来源与更新前后映射。关联查询与显示层级分离：目标折叠时高亮 Scope/SK，内部任务已展示时高亮具体 Task，外框不重复高亮；展开/收起保留原追踪来源。图中仅蓝色激活、紫色关联两类交互边框，保留节点业务填充色；关联依据放进详情，不在图中细分直接/间接类别。默认只扩展一跳执行图关系，不做无限依赖传播。

图配置入口展示当前实例的只读 graph_options，保留 JSON 前后阶段和 Log 行来源；Both 共同字段冲突进入 DFX，JSON 缺失时按 Log 显式记录回退，不填默认值。controls.js 管理图配置及子图工具条呈现，配置提取及校验由 options.py 管理。工具条不随图缩放，跟随子图边界并避让其他工具条；关联对象收进菜单，导航位于画布底部，快捷键和批量折叠位于视图设置。

Both 额外校验 Event／Value 的关键参数；跨流 Reset 校验利用已建立的依赖路径。Log 模式可通过当前实例日志中的明确节点、线程、函数名与函数地址恢复 FUNC 来源。关联方向、跳数与仅看关联统一作用于关联查询和显示，内部 Task 选择同样可过滤整图。

浏览层级统一为“进程 → Model”，Model 是最小选择单元；同名 Model 在同级列表加序号区分，不增加实例选择器。Model 总览按进程分组并支持搜索。Options 位于 Model 上下文区，展开为非悬浮配置表，原始字段名及更新前后值并列保留。图内区域使用 SK 文案，实际 Scope begin/end/placeholder 节点及原始证据名称不改；内部展开入口明确为“展开 SK 内部任务流”。右下角缩略图支持点击、拖动及方向键移动视野，独立收起；导航不改变选择或 Update。
