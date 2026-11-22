---
name: att-perf-selection-analysis
description: 当 ATT 成功求解，但默认模板或 tiling 性能不符合预期时，通过 PGO 寻找实测更快的候选，对照最终选择、profiling 和模型估值，定位 pipe 成本及 result 排序问题。
---

# ATT 模板选择性能定位

通过 PGO 查找待测快模板，根据 ATT 日志和 profiling 数据比较默认选择与候选的性能，核对各 pipe 的成本估值，
还原 result 级排序。结论附原始文件绝对路径和行号，并说明哪些已得到验证、哪些仍是假设。

## 范围与输入

- 读取已采集的日志和 profiling，将结果写入新建报告目录。若只有 `GetTiling Failed` /
  `SatisfyCons` 失败且没有成功候选，先排查求解约束。求解失败影响 result 排序时，记录其贡献。
- 每个 run 使用各自的 plog 和 `op_summary*.csv`，分别处理默认选择、PGO 和显式强制模板。
  核对输入 shape/dtype、设备、源码提交、运行命令、缓存条件和实测单位，在报告中列出缺失信息。
  采集前确认实际加载的动态库路径及 SHA-256，并与构建提交、安装包和产物记录对应。
- 在报告或现有运行 manifest 中记录 skill、解析器及分析脚本所在提交、实际执行命令、输入文件和输出路径。
  区分执行 skill 命令、直接调用解析器和人工补充分析；补充脚本复用仓内解析能力，记录其路径和内容标识。
- 现场采集和修复验证按用户已授权的范围与命令执行，缺少运行信息时先补齐。
  构建及测试分别遵循仓内 `af-build-runner`、
  `af-test-developer`，构建并行度限制为 `-j 8`。
- `<repo>` 为包含本 skill 和 `autofuse/` 的仓库根目录；以下命令在该根目录执行。
  `<out>` 是新建目录，使用日期/运行标识区分，禁止覆盖原始证据。

现场采集前读取 [references/validation-protocol.md](references/validation-protocol.md) 的采样与缓存口径；
任务涉及模型重拟合或修复回归时，再按其中对应章节执行。

## 1. 确认默认选择

先读 `autofuse/tools/att_analyze/README.md` 的 FINAL_TILING 日志契约，再执行：

```bash
python3 autofuse/tools/att_analyze/src/att.py evidence <log> -o <out>/evidence
```

用解析有效的 FINAL_TILING 确认实际下发的选择：`runtime/default` 是默认选择，
`runtime/explicit` 是显式选择，`pgo/pgo` 是 PGO 加载结果。候选日志和组级最优行只用于分析
搜索过程。缺少 FINAL、分片不完整或记录重复、冲突时，说明最终选择无法确认的原因。

保留 run/source/selection_mode/operator/graph/result/group/case_id/tiling_key/sub_case_tag/template 完整身份；
`case_id` 与 `tiling_key` 分别记录，按完整身份关联数据。语义名使用已有
`semantic_name`，不从算子名或生成代码猜测分类。多 group 的完整选择需结合 FINAL_TILING_SUMMARY。

在相同输入、设备和计时口径下比较默认选择与实测候选，列出耗时、单位、
`默认耗时 / 候选耗时` 和退化百分比。PGO 选择本身不证明更快或全局最优。
缺少对应实测时，报告模型排序，并将估值偏差和性能差异列为待验证项。

## 2. 通过 PGO 查找并确认快模板

读取 [references/pgo-comparison.md](references/pgo-comparison.md)，先从仓内环境变量文档查使能方式，
再沿实际前端的配置、编译器和生成代码确认当前版本支持的流程。
保留关闭 PGO 的默认基线，在独立目录和进程中运行 PGO 搜索；已有同环境的 PGO 证据可直接核验，
不重复调优。不支持的场景、调优失败或普通 ATT 回退需记录原因，再按任务范围显式验证其他候选。

搜索阶段记录参数、候选范围、实测记录和胜出产物；profiling 阶段复用已选中的 PGO 结果，
避免首次 PGO 调优与外部 profiler 同时采集。确认实际执行的完整身份、tiling、精度和独立实测时延后，
才将其作为快模板对照。GE 的 PGO-loaded FINAL、Inductor 的 Top-N 输出和实际执行证据按各自契约核对，
不能仅凭使能开关、配置文件存在或 Top-N 第一名认定 PGO 生效或更快。

记录默认与 PGO 的模板、tiling、block_dim、时延和各 pipe 估值/实测，并说明搜索覆盖及缺失证据。
PGO 未找到更快候选时如实报告；按已授权预算扩大搜索，结论限定为已搜索并实测的候选范围。

## 3. 整理候选数据

```bash
python3 autofuse/tools/att_analyze/src/att.py summary <log> -a -f console > <out>/summary_all.txt
rg -n 'Among the templates|Among all schedule results|The value of graph[0-9]+_result' <log>
```

为每个 graph/result/group 记录成功候选的 objective、MTE2/MTE3/VEC、block_dim、tiling 变量
和组级胜出 case。`summary -a` 汇总每组最优 case；检查全部候选时，读取日志原文或本 skill
脚本的 candidate 行。有 FINAL 时，CSV 输出以 final 记录为主。
工具可能遗漏 `R0` 标签、VEC 动态列或尚未支持的格式，summary 缺行时用原文核对。
保留原始 R 标签，分别记录它与普通 case 的身份。

## 4. 估值与实测逐 pipe 对照

```bash
python3 .claude/skills/att-perf-selection-analysis/scripts/est_vs_measured.py \
  --plog <default-log> --op-summary <default-op-summary.csv> \
  --plog <pgo-log> --op-summary <pgo-op-summary.csv> \
  -o <out>/est_vs_measured.csv
```

第 N 个 `--plog` 与第 N 个 `--op-summary` 配对。每个 run 对应明确的输入和调用范围，
不同 shape、设备或库版本的数据分别输入。多算子日志可增加 `--op-filter <substring>`：
按日志原始算子名做区分大小写的子串筛选，可重复传入，任一条件命中即保留候选和 FINAL 记录。
该筛选先于 profiling 名称匹配；全部无命中时生成仅有表头的 CSV，退出码为 0 并提示无可比较项。
脚本输出按 Run 隔离的长表：

- `candidate` 行保留每条 PROF pipe/objective 记录与原始 R 标签，用于查看候选估值。
- `final` 行使用仓内 `LogParser` 解析最终 pipe 估值，支持单行和校验完整的分片；
  保留完整身份及 `ParseStatus`，不借用其他 run 或非最终 tiling 的估值。
- profiling 使用 `aiv_total_cycles × aiv_<pipe>_ratio / Block Num`，单位为每 block cycle。
  只适用于已确认上述统计口径的 AIV 数据；不能将 AIC、总耗时、score、objective 当作同一量。
  ratio 支持小数或显式 `%`，缺失、非有限值、非法 block/ratio 留空并标状态，真实 0 保留。
- 默认在完整名称去除分隔符并忽略大小写后做唯一匹配。两侧存在命名差异时，
  根据原始证据创建 JSON 对象，将日志算子名映射到 profiling 完整 Op Name，传入
  `--op-map <mapping.json>`。每个 profiling 行只能对应一个已确认的算子。
- 重复 FINAL、多 group 或重复 profiling 行标记为 `ambiguous_final` 或 `ambiguous_profile`。
  取得调用、组级映射或独立采集数据后再比较；未下发候选的 measured 留空。
  当前脚本不自动聚合重复采样；先按已确认的调用边界拆分输入，或保留逐样本证据另行统计。
  不通过取首条 FINAL、按模板名去重或吞掉解析异常来绕过歧义状态。
- 只有 `Status=ok` 且存在 Ratio 时可读为 `Estimate / Measured`。输出文件必须不存在。
  正常退出表示表格已生成，定位结论还需检查可比较项和证据是否完整。

逐项检查所有可用 pipe 的估值和实测，并核对测量抖动、核数、聚合方式、频率及输入一致性。
所有 pipe 同比例偏差时，先检查统计口径和数据关联。

## 5. 还原误差公式

```bash
rg -n 'GetNodePerf.*perf res|StorePerf:Dma with|LoadPerf|LoadStoreStrideV2' <log>
```

以当前 tiling 变量代入对应节点公式，计算各项贡献和主导项占比。`perf-formula` 需要 tiling 源目录
和 `[PERF]` 行；只有 GELOGD 的 GetNodePerf 行时，从日志提取公式并代入计算。
需要深入建模时按仓内 `af-perf-modeler` 追溯真实 codegen 传参与 API 源码。

根据偏差所在的 pipe，核对模型公式与生成 kernel 的循环、搬运参数和 API 调用是否一致。
详见 [references/diagnosis-notes.md](references/diagnosis-notes.md)。

## 6. 还原 result 级选择

```bash
rg -n 'The value of graph[0-9]+_result|objective=|serialized_edge|axes reorder solver' <log>
```

先确认串行/并行调度分支。对串行结果摘录并验算 group_sum、edge base、penalty、最终 objective；
边集合和常量以日志及当前 `autofuse/att/generator/tiling_code_gen_impl.cpp`、
`autofuse/att/base/att_group_perf_config.h` 为准。无串行边时罚项可为 0；并行组按实际调度公式复算。

分析 pipe 估值偏差、同 result 其他 group 的成本和 edge penalty 分别对排序产生多大影响。
组级 solver 失败回退时，记录其成本贡献，再单独排查求解原因。候选实测接近时说明不确定性。

对能取得成对实测的快慢候选，在固定候选和 tiling 下逐 pipe 替换估值，按实际 result 公式复算排序。
记录单项替换及联合替换后的胜出身份、objective 和差值，区分触发模板切换的修改与决定错误排序的估值项。
替换得到的排序仍是诊断结果；随后重新求解并实测，确认模型修改是否改变了 tiling 或执行路径。
具体步骤见 [references/diagnosis-notes.md](references/diagnosis-notes.md) 的排序归因部分。

## 7. 修复验证

按已授权的范围成对运行基线与候选，保留相同输入、计时口径和缓存条件。
模型拟合需说明覆盖范围并使用独立数据验证；扩大用例回归时，检查显著变化项的最终模板和退化原因。
采样、拟合及批测结果的记录要求见 [references/validation-protocol.md](references/validation-protocol.md)。

分别报告命令退出状态、profile 是否完整、精度判定和性能判定。
有 profile 或命令正常退出不能替代精度通过，性能诊断失败也不能直接归为进程崩溃。
尚有超出已约定退化容限的用例时，列明风险和未闭环原因，不宣称修复无退化。

## 8. 报告与修复建议

中文报告包含：现象与环境/输入清单、完整选择身份及耗时对照、逐 pipe 估值/实测/倍差表、
公式分解与源码/kernel 证据、逐 result 排序及替换估值复算、根因与备选解释、修复建议及未覆盖证据。
说明修复涉及模型、结果级选择机制还是求解器。

报告注明 PGO 使能依据、搜索范围、搜索与独立复测的区别、缓存与采样口径、工具使用记录，
以及最终身份可比较的用例数。
`strings` 特征串、时间戳或源码祖先关系作为加载库核验的辅助信息。
缺少实际加载路径或产物记录时，将修复是否生效列为待验证项。

结论引用 `<原始文件绝对路径>:<行号>`；CSV 引用物理行号，跨文件不复用 offset。
诊断报告按上述内容组织；若任务进一步要求设计文档/spec，必须另按仓库
`docs/guidelines/design_document_template.md` 完整输出，并执行交叉影响和编码红线检查。
