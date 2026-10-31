# ATT Analyze

仓内 ATT 日志分析工具，入口为：

```bash
python3 autofuse/tools/att_analyze/src/att.py --help
```

支持 `summary`、`compare`、`split-slog`、`perf-formula`、`verify-tiling`、`evidence` 和 `topn-templates` 七个命令。`summary` 默认只读解析日志；`verify-tiling` 会编译并执行代码，使用前请确认输入目录和授权。

`LogParser` 的 `OperatorSummary.parse_status` 用于标明证据是否完整：`ok` 表示日志包含完整选择信息，`inferred_graph_result` 表示仅从模板行推断 graph/result，`missing_group_case`、`missing_result_performance` 和 `missing_graph_result` 表示相应日志缺失。CSV 列保持历史含义不变。

```bash
python3 autofuse/tools/att_analyze/src/att.py summary path/to/att.log -f csv -o /tmp/summary.csv
python3 autofuse/tools/att_analyze/src/att.py compare baseline.csv candidate.csv
python3 autofuse/tools/att_analyze/src/att.py evidence path/to/att.log -o /tmp/evidence
python3 autofuse/tools/att_analyze/src/att.py verify-tiling generated/ --scene tf --preset B --aiv-num 56
python3 autofuse/tools/att_analyze/src/att.py topn-templates path/to/worker_round_1_af.log -o /tmp/topn-templates
```

工具不会把缺失值当作有效的 0；请在后续分析中根据 `parse_status` 决定是否需要补充日志。

`topn-templates` 只读取 Autofuse worker 日志，输出静态 Top1/TopN 模板 CSV 与 Markdown。
Top1 取 `topn_output output_index=0` 下所有 group。若日志包含 `output[i]: perf=`，脚本记录
TopN 的 modeled perf；日志中没有该行时留空。脚本不从日志推断实测 Task Duration，也不读取
外部测试框架生成的 analysis CSV。

## TEMPLATE 日志契约

词表与规范化规则集中在 `autofuse/common/common_utils.*`。生产端在生成 ModelInfo 时保存规范化的
`semantic_name`；日志和离线工具直接消费名称，不各自维护 role 词表或重新分类。

`[ATT][TEMPLATE]` 的核心行使用字段 `semantic_name`，格式为
`<family>/<strategy>[@<role>][#<api-feature>|...][+<transform>...]`。字段和定位信息分成独立行输出：

```text
[ATT][TEMPLATE] stage=topn_output output[0] group=0 semantic_index=0 semantic_name_count=1 semantic_name="Pointwise/Common"
[ATT][TEMPLATE] stage=topn_output output[0] group=0 semantic_index=0 semantic_name_count=1 fields semantic_schema=1 family="Pointwise" strategy="Common" role="" api_features="" transforms=""
[ATT][TEMPLATE] stage=topn_output output[0] group=0 semantic_index=0 semantic_name_count=1 detail graph=0 result=0 tiling_key=0 template="graph0_result0_g0_0"
```

`name_status` 仅在无法确认或匹配不唯一时输出 `unavailable` 或 `ambiguous`；默认唯一匹配不输出该字段。
单字段或整行超过 700 字节时，输出 `<field>_hash`、`<field>_len` 或 `line_hash`、`line_len` 摘要，不截断字符串。
TopN 分析按 case/visit/shape、`GenerateTopnSolutions enter` 中的 operator 以及同名 operator 的调用序号隔离；
同一 visit 内多个 kernel 或多次调用不会合并。日志缺少 operator 关联时，CSV 的 `association_status` 输出
`unconfirmed`，不当作已确认调用边界。

## FINAL_TILING 日志契约

`FINAL_TILING` 只表示运行期已经写入最终 tiling data 的记录。运行期最终选择使用
`source="runtime"`（`selection_mode` 为 `default` 或 `explicit`），PGO 加载的最终 tiling 使用
`source="pgo"`（`selection_mode="pgo"`）；候选搜索阶段不产生 `FINAL_TILING` 记录，仅有候选
`GetTilingDataRepr` 调用不会被误认为运行期最终记录。解析器只识别完整字段名，历史简略别名
（`s=`、`src=`、`k=` 等）不再被解析。单行记录的字段如下：

```text
[ATT][FINAL_TILING] schema=1 source="runtime" selection_mode="default" operator="Fusion_0" graph=0 result=0 group=1 case_id=2 tiling_key=5 score=1 sub_case_tag="" template="ConcatCase2" repr_kind="full_json" pipe_estimates="{\"AIV_MTE2\":120.000000,\"V\":null}" tiling_repr="{\"tile_m\":64,\"tile_n\":128}"
```

其中 `case_id`（case）和 `tiling_key` 是两个独立字段；多 group 或多 result 必须保留完整的
`graph`、`result`、`group` 身份。`score` 是最终模板的 `CalcScore` 打分，用于模板筛选；它不是
pipe cycle、objective 或 profiling 实测值。`sub_case_tag` 是子场景标签，无子场景时为空串；
`template` 是模板名；`repr_kind` 取 `full_json`（`tiling_repr` 为完整 JSON）或 `unavailable`
（无法获取 repr，`tiling_repr` 为空串）。`pipe_estimates` 是 ATT 模型的 pipe 估值 JSON，
无法估计的 pipe 使用 JSON `null`，不能用 profiling 的实测 cycle 或数字 0 代替。

当单行记录超过 700 字符预算时，使用同一个 `id` 的 `FINAL_TILING_BEGIN`、连续
`FINAL_TILING_CHUNK` 和 `FINAL_TILING_END`：

```text
[ATT][FINAL_TILING_BEGIN] schema=1 source="runtime" selection_mode="default" operator="Fusion_0" graph=0 result=0 group=1 case_id=2 tiling_key=5 score=1 sub_case_tag="" template="ConcatCase2" repr_kind="full_json" pipe_estimates="{...}" id="8:Fusion_0|0|0|1|2|5|0:" chunks=3 len=1842 hash_alg=att_mix64_v1 hash=0e2418542347c1a0
[ATT][FINAL_TILING_CHUNK] id="8:Fusion_0|0|0|1|2|5|0:" seq=0 data="{\"tiling_key\":5,"
[ATT][FINAL_TILING_END] id="8:Fusion_0|0|0|1|2|5|0:" chunks=3 len=1842 hash_alg=att_mix64_v1 hash=0e2418542347c1a0
```

多 group 的运行期选择完成后还会输出一条 result 级 `FINAL_TILING_SUMMARY`，`groups` 字段是
各 group 选择信息的 JSON；超长时同样使用 `FINAL_TILING_SUMMARY_BEGIN/CHUNK/END` 分片。

分片记录中的 `hash_alg=att_mix64_v1` 表示后面的 `hash` 使用 ATT-Mix64-v1 对完整 `tiling_repr`（或 summary 的 groups JSON）计算。解析器用它在跨行重组后检测丢块、乱序、截断和内容修改；它不参与模板选择、score 计算、性能估值，也不提供加密。当前 producer 使用 `att_mix64_v1`；解析器仍兼容历史 `sha256` 分片日志。解析器会校验 chunk 顺序、数量、UTF-8 字节长度和哈希；缺少 END 或校验失败时输出 `incomplete_final_tiling`，不会返回部分 `tiling_repr`。同一身份（`source`、`operator`、`graph`、`result`、`group`、`case_id`、`tiling_key`）重复输出时保留第一条有效记录，并将后续记录标为 `duplicate_final_tiling`。

输入证据和结果的关系如下：

| 输入 | 能确认的内容 | 不能推断的内容 |
| --- | --- | --- |
| 只有 plog/编译日志 | 候选模板、case/key、模型 result | 运行期最终选择；不得伪造 `source="runtime"` |
| plog + profiling | 候选模型与实测 pipe cycle 的对照 | 仍不能证明最终写入了哪个 tiling，除非存在 `FINAL_TILING` |
| 含 `FINAL_TILING` 的运行日志 | 最终 group/result/case/key、模板名、`tiling_repr` 和可用的 pipe 估值 | profiling cycle 仍需从独立 profiling 证据读取 |

`summary` 会追加 `Final Source`、`Tiling Key`、`Score`、`Pipe Estimate`、`Tiling Repr` 和
`Final Parse Status`；`evidence` 每条最终记录输出一条 JSONL，并保留
`source_path/source_line`。ATT 代码生成路径会在最终选择完成后生成 `FINAL_TILING`；仅有候选
`GetTilingDataRepr` 调用不会被工具误认为运行期最终记录。无法恢复 sub-case 的缓存命中会跳过
最终记录，避免输出错误模板身份。

## 与 ATT 模板/tiling 分析 Skill 配合

ATT 成功求解但默认模板慢于 PGO/实测候选时，使用仓内
[att-perf-selection-analysis](../../../.claude/skills/att-perf-selection-analysis/SKILL.md)。
该 Skill 提供按 run 隔离的 pipe 估值/实测对照脚本，并结合最终选择、公式和 result 级
代价定位排序原因；没有唯一的最终身份或 profiling 归属时会标明证据不足。
通过 PGO 查找快模板时，先按 Skill 的
[PGO 对照流程](../../../.claude/skills/att-perf-selection-analysis/references/pgo-comparison.md)
查使能入口、完成搜索并确认实际选中解，再对默认与 PGO 结果独立 profiling 和比较估值。

`att_analyze` 由本仓维护，Skill 调用的固定入口是
`autofuse/tools/att_analyze/src/att.py`，不需要安装或访问其他仓库。对已经采集的
数据做离线分析时，在仓库根目录执行：

```bash
python3 autofuse/tools/att_analyze/src/att.py evidence <default-log> -o <out>/default-evidence
python3 autofuse/tools/att_analyze/src/att.py evidence <candidate-log> -o <out>/candidate-evidence
python3 .claude/skills/att-perf-selection-analysis/scripts/est_vs_measured.py \
  --plog <default-log> --op-summary <default-op-summary.csv> \
  --plog <candidate-log> --op-summary <candidate-op-summary.csv> \
  -o <out>/est_vs_measured.csv
```

第 N 个 `--plog` 与第 N 个 `--op-summary` 配对；脚本只读取数据，输出文件必须不存在。
可增加 `--op-filter <substring>`，按日志原始算子名区分大小写筛选；重复参数按“或”匹配。
全部无命中时生成仅有表头的 CSV，退出码为 0 并提示无可比较项。
重复 FINAL、多个 group 或重复 profiling 行会标记歧义，不自动聚合采样。
先按已确认的调用边界隔离数据，再按 Skill 的采样规则比较。

现场执行按用户已授权的 case 范围和命令开展，并核对远端 checkout 能访问所用脚本。
构建及测试分别遵循仓内 `af-build-runner` 和 `af-test-developer`。

原始证据与分析结论分开保存，建议布局如下：

```text
run-root/                         # 原始运行目录
  default/  pgo/
    att.log  profile/  kernel_meta/  dump/
evidence-archive/<run-name>/      # 原始文件归档
report-archive/<run-name>/        # report.md、summary.csv、root-cause.jsonl 等
```

使用新建目录保存输出，并在报告或现有 manifest 中记录原始路径、源码及工具版本、实际命令和库内容标识。
采样、缓存条件、模型拟合与成对回归的要求见 Skill 的
[validation-protocol.md](../../../.claude/skills/att-perf-selection-analysis/references/validation-protocol.md)。

## preset 和真实日志维护

`preset_B.json` 是 TensorFlow 动态 ABI 的示例输入，默认 `aiv_num=56`、
`ub_size=262144`，不代表所有芯片的硬件规格。执行 `verify-tiling` 时会打印
实际传入的 `aiv_num`、参数来源和动态维度；请根据目标设备核对，必要时使用
`--aiv-num` 或 `--input-json` 修改。`aiv_num` 是传给 TensorFlow tiling 的配置值，
Inductor ABI 不使用该字段。

`tests/data/` 中的日志是固定回归样例，不会自动同步现场日志。遇到新的 CANN、
TensorFlow 或 Inductor 日志格式时，请对真实日志脱敏后新增样例，并同步增加
`summary`/`evidence` 的期望结果；保留旧样例以防止已有格式回归。
