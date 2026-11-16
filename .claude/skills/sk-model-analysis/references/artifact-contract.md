# 产物契约

## 输入与能力

| 事实 | JSON | Log |
| --- | --- | --- |
| 更新前执行图 | sk_graph_origin.json | super_kernel.log / sk_node_detail.log 等原节点证据 |
| 更新后执行图 | sk_graph_updated.json | super_kernel.log 的已观察修改，不能假设完整 |
| scope / 融合成员 | skinfos 提供 SK 到 scope 关联，不提供完整成员表 | sk_scope_split.log / sk_fused_nodes.log |
| SK 内部队列 | sk_task_queue.json | sk_device_args.log |
| 运行 DFX | 按现有来源覆盖 | SK runtime dump、counter、PC、异常日志 |

JSON 模式只读 JSON，Log 模式只读日志。Both 独立提取后比较；日志独有信息可以按可信身份关联。不同实例不得只按 nodeId 合并。

## 覆盖与来源选择

快照状态为 complete / partial / unavailable。JSON 局部有效仍优先并显示 partial；不可用时才回退 Log。双方未提供某字段不代表校验通过。比较区分字段差异、单侧信息和不可比较。

Log 的原始 after 仍只记录明确修改；当主结果采用 Log 时，Graph 以更新前节点为基线叠加修改。没有修改记录的节点保留原位置和来源，并标记 `update_status=not_recorded`；不据此宣称最终状态已确认、不生成 retained 变换、不复制更新前依赖边，快照仍为 partial。该规则不补回 JSON 快照明确省略的节点，不覆盖明确 INVALID 或实际更新记录。

节点、scope、融合、队列为独立对象；graph 节点身份、融合成员映射、task 队列序号不能互换。同步边需要配对证据，不按相同 event/address 建全连接。原始 nodeId 在更新后可能变成其他类型。

## 输出

`run-portal.html` 内嵌 Graph / DFX 及全部查看所需资源；`file://` 离线可用，无 fetch 或 CDN 依赖。可配置额外 ZIP / tar.gz，默认 auto 以 50 MiB 为阈值产生 ZIP；压缩包只包含交付 HTML。

默认只写独立 HTML，超过配置阈值时附加压缩包。`--export-summary` 才导出 `data/run-summary.json`，schema_version=1.0，包括 models、diagnostics、parser_mode、generation 和 profile。为兼容结构化机器消费者，用户版摘要保留原始 `runtime`、`diagnostics`、`comparison`、`fusion_analysis` 等字段，并在每个 model 附带 `dfx.profile=user` 用户投影；这些原始字段不会嵌入默认用户 HTML。显式使用 `--dfx-debug` 时，`run-portal-debug.html` 和按需导出的 summary 还会附带 `dfx.debug`，页面展示完整开发排查字段。融合证据仍保留最终 Scope 边界、融合失败记录、规范化算子影响统计及前后相邻 Scope 上下文，不改变 Graph 拓扑。每个 model 的 `graph_semantics` 区分结构性纯 Scope、单 Kernel 更新和多 Kernel 融合；`statistics` 同时保留原始 `scope_count` 与用户图可见的 `visible_scope_count`。

`--export-parsers` 独立导出 `data/parser-results.json`，保留独立解析器快照。`data/diagnose-profile.json` 按 --profile 输出工具执行耗时，不是模型设备耗时。`--dfx-debug`、`--profile` 均不隐式导出其余 JSON。

缓存默认关闭，`--cache` 显式开启 `.cache/` 读写；`--no-cache` 禁止读取、写入及指纹计算。既有调用若需要原来的完整产物，应指定 `--export-summary --export-parsers --cache`。复用输出目录不自动删除未选择的旧导出、缓存或压缩包；仅更新本次指定产物。

AI 汇报可以直接消费 HTML：`report-data` 的 `payload_encoding` 未设置时读取完整 `models`；值为 `gzip-base64-models-v1` 时，`models` 只含目录摘要，完整数据位于 `report-model-{index}` 的 Base64/gzip JSON 块，须逐 Model 解码。普通用户 HTML 优先读取 `model.dfx.fusion`，不要求存在原始 `fusion_analysis` 或 `diagnostics`。汇报由 agent 按 [ai-report.md](ai-report.md) 完成，默认另存 Markdown，不改变 HTML 载荷或图数据，也不增加 AI 后端或 CLI。

图页面仅提供单 Update，不切换解析器。Scope / Task 在 Graph 中查看。开发排查信息不提供页面内开关，只能通过 `--dfx-debug` 重新生成独立 HTML。

## 状态原则

用户版 DFX 是 Graph 拓扑看板：没有解析日志 ERROR / Warning 时显示“解析情况”，并在“融合概览”中展示实际融合算子 / 期望融合算子、形成的 SK 数量和真正未融合的算子；算子导致融合断开、同步信号导致融合断开作为概览证据，不重复生成告警卡。期望融合集合来自更新前的真实 KERNEL，只有日志明确标记 `NOT_IN_SCOPE` 的记录才可排除；`OP_UNSUPPORT`、资源不足等融合失败仍计入期望集合和未融合列表。出现日志问题时结论标题为“日志诊断”，并在该区域显示对应级别、影响、Graph 定位入口和处理动作。关系补充、INVALID 保留、`FORMAT_ERROR` 和跨层确认不在用户看板展示；运行覆盖不足、解析器内部状态和原始日志只保留在 debug profile 与结构化机器结果中。JSON/Log 差异、提取失败、缺失和降级只有在影响当前结果时才用用户可读方式说明。

## 重复结构与大报告载荷

`model.exploration.groups` 为暂不展示的兼容元数据，不参与当前 Graph 投影或 DFX 展示。它保留原图重复次数、逐次 `iterations`、完整块内拓扑的校验状态和 `differences`。`×N` 不代表运行时循环；原图不完整或拓扑不同的组标为候选，诊断可定位到 Scope。更新后逐次节点数量和类型差异单独列出。

超过 8 MiB 的展示 JSON 按 Model 分块 gzip/base64 内嵌，`report-data` 只含摘要，浏览器通过 `ReportPayload.loadModel(index)` 读取当前完整模型。结构化消费者可通过 `--export-summary` 获得 `data/run-summary.json`，或按 AI 汇报规范解码 HTML，不要假定 `report-data.models` 总是完整图。离线载荷无需外部 JSON 或网络，保留原始证据，旧浏览器缺少解压支持时显示中文错误。

Graph 交互约定：固定水平 Stream 泳道；连续 RESET 默认合并显示。单击高亮、双击详情、按钮展开，搜索仅定位。Event / value 跨流依赖默认通过高亮呈现，Stream 顺序线默认显示；SK 内部 sync 连线仍显示。过滤只影响显示，不删除原始证据或生成跨隐藏节点的虚假依赖。Scope 按各 Stream 顺序进行块对齐，`boxes` 正常情况下仅包含一个完整外框，`rows` 保留各泳道成员范围。矩形内不得混入其他节点；成员交错导致约束冲突时保留原顺序和分段边界，并在 DFX 标明降级。

Scope 折叠仅改变展示，保留原始成员身份与边界端口。局部 Update 内容来自 Scope 对应真实快照，内部依赖仅采用该快照证据；跨阶段外部关联仍以整图阶段为依据，不生成连接新旧节点的推测边。局部 Task 双击可查看对应阶段详情与 SK 队列；全局阶段、局部阶段、折叠状态分开保存并支持返回恢复。

明确 INVALID 更新仍保存在 after.nodes，保留原始位置与更新来源；用户 Graph 的更新后拓扑默认排除 INVALID，用户统计同步排除，详情和调试数据仍可定位。`statistics.after_nodes` 保留原始快照数量，`statistics.visible_after_nodes` 表示用户拓扑数量。Both 的 JSON 未列出该节点而 Log 明确 INVALID 时可补充此失效记录，JSON 同 ID 节点始终优先，来源补充进入结构化结果。缺失本身不能作为 INVALID 证据。

节点显示过滤与 Update 阶段切换独立：INVALID 和 RESET 默认过滤，Scope 默认不过滤；图上方提供明确的“过滤 INVALID”和“过滤 RESET”复选框。过滤应用于整图及局部 Scope 内容，只改变显示，不删除源数据，不改接依赖；切换 Update 保留过滤状态。RESET 的显示过滤与连续 RESET 聚合分别控制。

“过滤 Scope”默认关闭，隐藏 Scope 容器、折叠卡片及明确命名的 Scope begin/end/placeholder Kernel，保留其他内部 Task/SK；不是删除整个 Scope 的任务。关闭后恢复各 Scope 保存的折叠状态。全图与局部 Update 的节点阶段不因此改变。


缩放始终使用同一执行图坐标与绘制路径，不切换卡片索引或按 Stream 拆分 Scope。缩小后完整 Scope 区域及融合色阶保持一致，只按比例隐藏文字和控件，避免固定宽边框覆盖融合颜色。

节点语义颜色：明确融合成员与 SK 为绿色；Scope begin/end/placeholder 为灰色；未融合 Kernel 使用普通 Kernel 样式，融合失败和断开原因继续通过 DFX 与选中联动呈现。用户手动 Scope 使用独立轻量虚线边框，默认显示，由“整图操作”的“显示用户 Scope 边框”开关控制；不改变布局或节点语义色。同 Stream 的顺序边始终水平直连，不因 Scope 对齐产生的间距绕折线；跨流 Event/value 仍按真实依赖路由。

Scope 的结构浏览统一在整图：框头及选中操作栏提供折叠/展开和局部 Update；选中操作栏及 SK 节点按钮支持向下展开内部队列（多个 SK 时显式选择）。Scope 双击详情只保留状态、成员/类型统计、融合记录和来源证据，不再复制 Stream 排布或更新后跳转列表。查看 SK 内部不改变全局/局部 Update 或折叠状态，返回恢复原详情及证据展开状态。SK 内部以纯图结构下挂展示，只保留归属标题、收起入口、队列节点与依赖边，不嵌入搜索、分页、统计或证据面板；完整队列共享整图拖动与缩放，允许多个同时展开、独立收起；展开只扩展下方空间，不移动主图节点。虚线挂接表示归属，内部 sync 连线表示依赖。内部单击高亮关联，双击打开 Task 参数、来源、上下游依赖及明确映射外部关联的悬浮详情；详情关联项可定位对应任务；主图双击同样只看节点详情。下挂子图明确标注更新后 AIC/AIV 队列，Update 保留展开状态，切换 Model 清空展开。

跨层关联独立于执行依赖：FUNC 来源通过融合成员索引与函数地址交叉核对；PRELOAD/SYNC 使用同队列来源标记表达配套关系，不增加依赖箭头。内部 Event 的 args 与同一 Model 实例更新后 value 事件 devAddr 建立地址关联，区分等待、写入和值，轮次未知保留说明并汇总 DFX。内部默认关联对齐，支持切回原始排布；对齐保留各队列顺序。主图 Kernel、外部 value 节点均可展开对应内部任务，双向高亮；Scope/SK 入口继续保留。

`exploration.sync_by_scope` 中的 `relations` 保存 origin / companion / address 关联，与 `links` 执行依赖分离；`layout.raw`、`layout.aligned` 保存两种任务列坐标，`relation_issues` 保存来源、轮次及对齐未确认项。后端 relations.py 负责证据与布局，前端 relations.js 提供实例内索引和关联查询，dependencies.js 只负责子图及详情呈现。

内部图保留类型颜色与轮廓、水平队列基线及独立过滤。SET/WAIT 使用动态虚线，ALL 使用实线；不显示问号或一对多候选连线。SYNC 按固定方向 flag 每次 WAIT 消耗一次 SET 的规则，恢复重复信号的一对一配对；不使用 nodeIndex。配对元数据包含 flag_counter_fifo 依据、ALL 区间、序号和对端 Task。缺端点、数量不符或队列错误用红色标记，解析不足不标成运行异常。每个 SK 的内部过滤入口位于子图旁的固定字号工具条，独立保存显示类型、仅看关联和对齐；收起重开恢复。批量应用是菜单中的显式操作。

`signal_groups` 在单个 Model 实例内按归一化地址聚合跨 Scope 内部 Event 和外部 value 端点，通过保留 nodeId 桥接原始 Event ID 的 NOTIFY/WAIT/RESET。`signal_bridge` 关系不依赖 RESET 是否可见。`sync_candidates` 保留为空以兼容旧数据契约；不生成笛卡尔积候选。`barrier_pairs` 保存按队列执行顺序恢复的共同 ALL 屏障关联，不属于有向执行依赖。signals.py 负责信号组与屏障关联，taskgraph.js 负责纯子图样式、过滤和同步绘制；详情仍由 dependencies.js 提供。

同步配对源码依据：graph-autofusion 的 `super_kernel/kernel/sk_entry.asc::AutoCoreSyncImpl` 将 C→V 映射到 `SYNC_AIC_AIV_FLAG`、V→C 映射到 `SYNC_AIV_FLAG`；`FullCoreSyncImpl` 在 ALL 内发送并等待两个方向。asc-devkit 的 `CrossCoreWaitFlag_ISASI.md` 明确 WAIT 每次消费一个 flag 计数。因此固定通道内的顺序消费可以重建配对，不能以“日志没有轮次”为由跳过重复 SYNC。先校验 ALL 两侧数量和各区间端点平衡，再验证配对与队列顺序不构成环；异常保留在 DFX，不生成矛盾连线。

Scope/SK 单击可汇总内部任务的外部关联，穿透跨 SK 信号、融合来源与更新前后映射。关联查询与显示层级分离：目标折叠时高亮 Scope/SK，内部任务已展示时高亮具体 Task，外框不重复高亮；展开/收起保留原追踪来源。图中仅蓝色激活、紫色关联两类交互边框，保留节点业务填充色；关联依据放进详情，不在图中细分直接/间接类别。默认只扩展一跳执行图关系，不做无限依赖传播。

默认关联高亮排除 Stream 顺序，仅追踪 Event/内存等关联；流顺序可手动选择。更新后补回 INVALID 时，保留 nodeId 且 Stream 未变的节点使用同一更新前位置基准显示，避免混排 JSON 压缩序号与 Log 原图序号；原始 order 不修改。局部 Scope Update 与整图使用相同显示排序规则。

局部 Update 的节点选择只能从该节点当前显示阶段追踪，不得以所属 Scope 代理节点代替起点。只有明确选中 Scope/SK 才汇总其成员关联。SK 下挂子图只画真实内部 Task，不重复绘制外部 VALUE_WAIT/WRITE；外部地址、值和定位入口保留在详情，主图继续关联高亮。取消节点常驻更新类型角标，前后类型保留在详情中。

图配置入口展示当前实例的只读 graph_options，保留 JSON 前后阶段和 Log 行来源；Both 共同字段冲突进入 DFX，JSON 缺失时按 Log 显式记录回退，不填默认值。controls.js 管理图配置及子图工具条呈现，配置提取及校验由 options.py 管理。工具条不随图缩放，跟随子图边界并避让其他工具条；关联对象收进菜单，导航位于画布底部，快捷键和批量折叠位于视图设置。

关联校验补充：JSON 与 Log 独立提取后，共用已建立依赖路径检查 Reset／竞争写入是否位于生产消费区间之外。待恢复边不得证明自身，也不得造成循环；未匹配等待项的诊断不能因同组其他项恢复而消失。Both 对共同 Event ID、Event 标志、Value 地址、值和等待标志按数值规范化比较，保留双方原值和来源。

Log 的 FUNC 来源补充从当前实例 super_kernel.log 提取，要求线程、明确节点编号和函数名称一致，并通过融合索引及实际函数地址交叉确认。跨层信号诊断在 SK 关联建立后细化，保留原始诊断及内部来源；地址关联不升级为确定通知值或轮次依赖。

整图、局部 Scope、内部子图与关联列表使用同一关联查询。方向和跳数约束依赖遍历；来源、配套和地址桥接独立保留关联依据。直接选择 SYNC 时按同步端点查询，不把配套 FUNC 的整组任务作为额外选择起点。仅看关联依据逻辑选择，支持内部 Task 和局部节点，不依赖主图是否存在激活节点。

当前界面层级为进程 → Model，不提供 Model 下的实例选择器；解析结果的隔离键和原始实例证据继续保留，同名 Model 在同级列表明确区分。graph-options 在 model-context 中以行内配置区展开，保留原始 Option 名称、前后值与来源。用户文案将区域称为 SK，保留真实 Scope 节点名；SK 内部任务流入口与区域成员展开入口分别表意。minimap.js 负责整图导航缩略图，以世界边界映射定位，点击与拖动只更新相机，并记录可返回的视野。
