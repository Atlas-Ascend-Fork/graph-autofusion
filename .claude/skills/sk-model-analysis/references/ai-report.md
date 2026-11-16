# AI 融合汇报

用于用户要求“分析这个 HTML”“汇报整体融合情况”“找出主要断图影响项”等任务。由当前 agent 分析和表达，不接模型后端，不新增 AI CLI，不修改 Graph/DFX 页面。

## 先得到可分析的数据

1. 记录用户提供的 HTML 路径，读取 `report-data` JSON，识别 schema、载荷编码、进程／Model 清单及生成状态。路径优先使用用户提供的文件；HTML 中记录的原始服务器路径只是来源，不能假定当前机器存在。
2. 未设置 `payload_encoding` 时，完整 Model 在 `report-data.models`。若为 `gzip-base64-models-v1`，其中的 Model 只是目录摘要；按索引读取 `report-model-{index}` 的 Base64 → gzip → JSON，才能取得完整 Model。**只读目录摘要或浏览器当前选中的 Model 不算完成全量分析。**
3. 逐 Model 解压、提取紧凑指标和问题索引，保留可按需回查的 ID；不要把几十 MB 的 HTML、压缩字符串或所有原始节点直接塞进上下文。若一个 Model 块损坏，记录其覆盖缺口并继续其他 Model；汇总中说明实际成功读取的范围，不把缺失项填零。
4. 默认直接提取 HTML 数据，无需运行页面 JavaScript 或联网。可以使用一次性本地 Python/Node 片段、临时文件或现有浏览器工具；这些只是读取与计数工具，不承担 AI 判断，也不作为新的发布脚本。安全读取 JSON、Base64 和 gzip，不能用 `eval` 执行页面文本。
5. 用户版保留 `dfx`、图、转换与关联数据，但会删除原始 `fusion_analysis`、`diagnostics` 和 `comparison`。优先读 `model.dfx.fusion`；缺少原始字段通常是用户投影，不是解析故障。调试版和同源 `data/run-summary.json` 可提供更多证据，但不是开始 HTML 汇报的前置条件。

用户投影中的 `break_details`、定位 targets 和部分 ID 列表可能只保留前若干项。先核对 `break_detail_count`、总计数字与列表长度；列表长度不能替代完整数量，展示样例不能声称覆盖全部位置。需要更多对象时用完整 boundaries、节点和显式 transformations/fusions 回查；HTML 未保留的具体原因不补造。

可用下面的标准库片段理解当前载荷读取方式。实际运行时传入用户路径，先输出紧凑摘要，再按需检查单个 Model。遇到未知编码或不同旧格式，转入兼容诊断，不自行执行另一套页面脚本。

```python
import base64
import gzip
import json
import sys
import zlib
from html.parser import HTMLParser
from pathlib import Path

class Payload(HTMLParser):
    def __init__(self):
        super().__init__()
        self.blocks, self.current = {}, None
    def handle_starttag(self, tag, attrs):
        ident = dict(attrs).get("id", "")
        if tag == "script" and (ident == "report-data" or ident.startswith("report-model-")):
            self.current = ident
            self.blocks[ident] = []
    def handle_data(self, data):
        if self.current is not None:
            self.blocks[self.current].append(data)
    def handle_endtag(self, tag):
        if tag == "script":
            self.current = None

payload = Payload()
payload.feed(Path(sys.argv[1]).read_text(encoding="utf-8"))
header = json.loads("".join(payload.blocks["report-data"]))
encoding = header.get("payload_encoding")
if encoding not in (None, "gzip-base64-models-v1"):
    raise ValueError(f"需要识别报告载荷格式：{encoding}")
for index, summary in enumerate(header.get("models", [])):
    try:
        if encoding:
            packed = "".join(payload.blocks[f"report-model-{index}"]).strip()
            model = json.loads(gzip.decompress(base64.b64decode(packed, validate=True)))
            if summary.get("key") is not None and model.get("key") != summary["key"]:
                raise ValueError("Model 索引与数据块身份不一致")
        else:
            model = summary
        print(model.get("key"), model.get("dfx", {}).get("metrics", model.get("statistics", {})))
    except (KeyError, ValueError, OSError, EOFError, zlib.error) as error:
        print("未读取的 Model", index, summary.get("key"), str(error), file=sys.stderr)
```

片段只演示读取指标，不等于已经完成汇报。还需逐 Model 收集下面的融合证据和异常对象。

## 分析顺序与数据入口

| 分析内容 | 优先读取 | 必须回答 |
| --- | --- | --- |
| 分析覆盖 | header.models、各 Model 的 key、coverage、dfx.status | 一共多少进程／Model，实际读取多少；哪个缺口影响了哪项结论 |
| 融合规模 | dfx.fusion.operator_outcomes、dfx.metrics、statistics | 期望／实际／未融合算子、形成的 SK 数；单 Kernel 与多 Kernel 分别多少 |
| 主要影响算子 | dfx.fusion.operator_impacts、node_failures | 哪些算子没有融合；哪些算子导致拆分；各自影响次数和范围 |
| 拆分与整体失败 | dfx.fusion.boundaries、operator_impacts[].break_details、graph_fusion | 受影响 SK、触发节点、实际拆分点；是否整体未融合，触发节点后来是否仍融合 |
| 更新结果 | before、after、transformations、fusions、graph_semantics | 节点具体变成什么、进入哪个 SK；是否有可验证的异常变化 |
| 关联关系 | before/after.edges、exploration.sync_by_scope、signal_groups | 主要问题涉及哪些内外部任务，哪些是确认依赖，哪些只是关联 |
| 配置与日志 | graph_options、dfx.findings、dfx.coverage | 有哪些实际 ERROR/Warning、相关配置值和可定位的问题 |

同一份报告的优先统计顺序为 `dfx.fusion.operator_outcomes` → `dfx.metrics` → `statistics`。三者字段名不完全相同，例如 outcomes 的 `expected_node_count/actual_node_count/unfused_node_count` 对应 metrics 的 `expected_operator_count/fused_operator_count/unfused_operator_count`。记录采用的字段和口径；若同口径数值矛盾，核对节点与转换证据，不默默选有利的数字。

## 统计和因果规则

- **算子与节点分开。** `fused_nodes` 包含非 Kernel 成员，不等于实际融合算子；SK 数、Scope 数、队列 Task 数也不能相互替代。只有净减少节点数不能推出融合算子数。
- **期望融合不排除失败。** 真实业务 Kernel 只有明确 `NOT_IN_SCOPE` 才可排除；`OP_UNSUPPORT`、资源不足和静态编译问题仍计入期望及未融合。Scope begin/end/placeholder 等结构标记不当作业务算子；`scopeFlags=0` 不能证明算子不应融合。单 Kernel SK 仍是真实更新结果，纯结构性 Scope 不作为业务融合成果。
- **聚合先计数再求比例。** 对齐口径且覆盖完整时验证“期望 = 实际 + 未融合”，总体比例为 `sum(实际)/sum(期望)`，不能平均各 Model 的百分比；分母为零时不计算比例。部分覆盖的范围不混入“全部完成”的结论。
- **保持身份。** 进程／Model 是展示分组，完整 `model.key` 是定位命名空间。不同进程、重复 Model 捕获中的同名算子和同 ID 节点不能合并去重。总数注明按采集记录累计；多个输入报告分别列规模，不因结构相似就当成同一次执行。
- **不增加浏览层级。** 面向用户的汇报沿用“进程 → Model”；instance_id 只用于完整 key 或同名捕获消歧，不把每行相同的 `mi01` 再列成一个层级。
- **记录条数与对象数分开。** 同一节点可有多条原因记录，历史 pass 也可能重复记录同一次拆分。未融合算子按唯一原始节点计；断开按同 Model 的最终边界身份计，不把 previous/following SK 各算一次。缺少边界 ID 时用可验证的 scope／触发节点／实际拆分点组合核对；不能直接把各算子影响范围相加当全图唯一受影响 SK 数。
- **覆盖提示不重复累加。** 同一来源回退提示可能在多个 Model 中携带相同的总体 `count`；按实际受影响的 Model key 统计覆盖范围，不把每份重复提示的 count 再求和。日志问题记录按各自对象与来源定位，不和覆盖提示混算。
- **算子归一化保留原名。** 优先采用 `operator.normalized_name`，去除有明确格式的 hash 与编译后缀后按 OpType 汇总；无法识别的名字保留，不截掉可能是算子语义的数字。保留原始名称或节点 ID 以便定位。不凭前缀把未知实现强行归类为动态 AscendC、静态 AscendC、Triton 或自定义算子。
- **拆分不等于未融合。** 分开说明受影响的 SK、导致拆分的节点、实际拆分位置，以及触发节点后来是否进入其他 SK。`break_details` 的明确映射优先；没有该字段时按原始边界和显式 transformations 核对，不能套用别的报告结论。
- **中间观察不当作最终结果。** `operator_impacts[].observed_effect.observed_unfused_kernel_count` 等阶段观察字段不能直接当作最终未融合算子数；触发节点随后可能已融合。最终状态使用 `operator_outcomes` 和显式更新／融合记录核对，不因字段名含 `unfused` 就省略阶段判断。
- **同号节点不是完整转换结论。** 原 Kernel 的同号 after 节点可能成为 MEMORY_WAIT，而原算子仍作为成员进入另一个 nodeId 承载的 SK。分别核对 modified 与 fusion 记录；SK 入口必须对应实际 after 的 `is_sk` 节点，不能按同号、最小成员 ID 或区域首节点猜测。
- **解释有条件的原因。** `DEADLOCK_DETECTED` 是融合时为避免死锁而拆分，不直接当成运行死锁。只有明确“Deadlock at node A, split at Wait node B”的记录才可给出 A、B。`BINDMAP_INIT_EMPTY` 且原始 Kernel 名没有 `static_kernel_` 前缀时，可表述为“不支持静态编译，或静态编译失败”；不能推广到所有同枚举记录。Batch mode 等原因也必须有明确原因记录，不能仅凭参数值归因。
- **关联不升级成依赖。** 更新前 EventId 与更新后 Value 地址分别解释，通过显式 nodeId 转换、融合成员和内部来源记录桥接。相同地址、同一 Stream 邻接、相同 nodeIndex 都不足以单独证明依赖；同 Event 的一对多关联本身也不算异常。内部 SET/WAIT 使用已重建的配对事实；不因缺少消费轮次额外制造告警。
- **日志与工具问题分开。** 日志明确 ERROR/Warning 才作为日志问题；用户版不含原始诊断不等于零条错误。解析器内部 `FORMAT_ERROR`、无关的关联补充和 INVALID 保留信息不复制成用户告警。缺口确实影响判断时，只说明具体影响范围和可执行处理方式。没有问题时省略异常小节，不用“运行证据不足”占位。
- **不推断性能收益。** 节点减少、融合比例、绿色深浅和工具解析耗时都不能证明设备加速或实际运行耗时。

## 汇报范式

默认在对话中先给出 3～5 条核心结论，完整结果写入 `build/model-analysis-ai-report/<本轮任务名>/ai-summary.md`。用户指定路径时遵从；说明原 HTML 路径，便于重新打开。不要要求运行一个不存在的 AI 命令。

按有意义的信息选用下面结构，不为凑齐模板输出空表、重复解释或未分析的结论。先列全量 Model 的核心指标，再对主要问题下钻；结论必须与实际读取范围一致。

```markdown
# SK 融合分析

输入：原 HTML 路径；覆盖：已读取进程数／Model 数与未覆盖范围。

## 整体结论
用一段话说明融合规模、主要问题和优先处理项。

## 融合概览
| 进程 / Model | 期望算子 | 实际融合 | 未融合 | SK 数 | 融合比例 |
| --- | ---: | ---: | ---: | ---: | ---: |

说明单 Kernel、多 Kernel、结构性范围的口径；总比例使用加权计数。

## 主要影响项
| 算子 | 未融合算子数 | 导致断开次数 | 具体原因 | 定位 |
| --- | ---: | ---: | --- | --- |

按可确认的影响大小排序。断开与未融合分列，不编造可融合上限或速度收益。

## 关键问题与定位
每项给出“结论 → 证据 → 影响对象 → 建议”。
定位写清进程、Model、更新前／后、SK ID、nodeId，内部任务附队列与 Task ID。
没有稳定锚点的报告不伪造 #node-xxx 链接，说明打开原 HTML 后搜索的标识。

## 建议处理顺序
只列与已确认问题对应的建议；区分算子支持／资源／同步拆分／工具兼容。
必要时单列影响结论的数据缺口和最小补充信息。
```

交付前核对：全 Model 覆盖、统计口径、跨进程隔离、原因条数去重、触发点与拆分点区别、更新前后身份、是否将关联误称依赖、是否臆测耗时，以及报告路径是否可打开。需要兼容修复时继续读 [兼容诊断与自适应修复](compatibility-repair.md)。
