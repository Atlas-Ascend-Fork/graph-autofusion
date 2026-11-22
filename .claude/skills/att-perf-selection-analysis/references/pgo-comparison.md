# 通过 PGO 查找快模板并对照默认选择

PGO 用于在当前生成的候选范围内发现实测更快的解。完整过程包括默认基线、PGO 搜索、
选中结果的独立复测和模型对照。默认与 PGO 使用相同输入、设备和代码/库版本；
缓存和采样协议按 [validation-protocol.md](validation-protocol.md) 记录。

## 查找使能入口

先识别 GE/TBE 或 TorchAir/Inductor 路径，并在实际运行的 checkout 中检查以下入口：

| 目的 | 仓内入口 |
| --- | --- |
| 开关、取值及适用路径 | [环境变量说明](../../../../docs/zh/autofuse/introduction/environment_variables.md) 中的 `AUTOFUSE_FLAGS`、`AUTOFUSE_DFX_FLAGS` |
| 搜索流程、限制和缓存身份 | [Auto Tiling 的 PGO 机制](../../../../docs/zh/autofuse/design/features/auto_tiling.md#55-pgo-机制) 及其当前能力状态表 |
| 环境变量实际解析 | `autofuse/common/autofuse_config/`、`autofuse/compiler/python/compile_adapter.py` 的 `get_pgo_env_flag`、`get_pgo_topn` |
| GE 搜索及配置加载 | `autofuse/compiler/python/asc_codegen_compile.py` 的 `asc_pgo_exec`、`pgo_second_optimization`；`autofuse/codegen/codegen_tiling_pgo_search.cpp` |
| Inductor 实测与产物 | `autofuse/codegen/codegen_tiling_inductor_pgo_runner.cpp`、`codegen_tiling_inductor_topn.cpp`；`autofuse/compiler/python/ascendc_compile.py` 的 PGO bundle/manifest |
| 最终身份解析契约 | [ATT Analyze README](../../../../autofuse/tools/att_analyze/README.md#final_tiling-日志契约) |

可用检索命令：

```bash
rg -n 'autofuse_enable_pgo|autofuse_pgo_topn|autofuse_pgo_algo' \
  docs/zh/autofuse/introduction/environment_variables.md autofuse/common/autofuse_config \
  autofuse/compiler/python/compile_adapter.py
rg -n 'asc_pgo_exec|pgo_second_optimization|config_path|search_path' \
  autofuse/compiler/python/asc_codegen_compile.py
```

核对静态 shape 和前端支持范围；当前能力状态表中的动态 shape/CV 不支持项不套用此搜索流程。
依赖按当前采样后端确认：文档中的 mspti 要求需要结合安装版本、
生成产物及实现核验；Inductor bundle 还需检查 `profiling_backend`、`requires_mspti`、协议版本和产物 hash。
不能将某一路径的依赖、配置格式或缓存复用规则套到另一条路径。

## 默认基线与 PGO 搜索

先保存原有环境和框架配置，确认测试框架是否会重写环境变量或启用编译缓存。
关闭已有的强制模板/result/case 配置，避免将强制选择误作默认基线。
下面是 GE 的最小配置示例；实际执行时合并所需的原有功能项，同一控制项只保留一个值，
分别在独立进程中运行已确认的同一 case 命令。

```bash
# 默认基线进程
export AUTOFUSE_FLAGS="--enable_autofuse=true;--autofuse_enable_pgo=false"
export AUTOFUSE_DFX_FLAGS="--codegen_compile_debug=true;--debug_dir=<out>/default"

# PGO 搜索进程
export AUTOFUSE_FLAGS="--enable_autofuse=true;--autofuse_enable_pgo=true"
export AUTOFUSE_DFX_FLAGS="--codegen_compile_debug=true;--debug_dir=<out>/pgo-search"
```

Inductor 通过实际框架入口开启 AscendC 后端，并核对 `--autofuse_enable_pgo=true` 是否到达该编译链；
不能仅靠 GE 的 `--enable_autofuse` 开关认定 Inductor 后端已启用。
诊断采集可在编译和运行时设置 `ASCEND_GLOBAL_LOG_LEVEL=0`，保留编译 worker 日志、运行 plog、
生成源码和产物。按当前 `GenLogDefine`、`ShouldEmitFinalTiling` 核对生成期和运行期日志条件；
生成时关闭了日志的产物，不能仅靠重放时修改日志等级补出证据。

GE 的搜索参数位于 `AUTOFUSE_DFX_FLAGS`：`--autofuse_pgo_algo` 默认 `core_select`，
可选 `pruning`；`--autofuse_pgo_step_max` 仅用于 `pruning`；`--autofuse_pgo_topn` 默认 `5`，
`0` 表示取当前生成的全部候选进入后续处理。取值范围先核对当前文档和解析代码。
记录实际参数、成功/失败采样数、候选覆盖和是否提前终止；需要扩大搜索时按已有预算调整。
`topn=0` 仍受候选生成、算法和前端限制，不能据此声称穷举了全部 tiling 空间。

首次 PGO 搜索使用自身采样，不同时嵌套 msprof 或框架的其他 profiling 功能。
使用新建的编译输出与缓存目录，保留已有证据，避免误用历史 PGO 结果。
GE 当前以 `<pgo_dir>/<graph_name>_config.txt` 存在判断跳过调优；`pgo_dir` 由实际编译目录推导，
`debug_dir` 本身不能证明缓存已隔离。归档真实路径和内容标识，核对日志是新搜索还是缓存复用。

GE 保留 `<graph_name>_search.txt`、最终 `<graph_name>_config.txt`、搜索及二次调优日志、
相关 Host/Device 产物。Inductor 保留 Top-N 协议、候选 repr/hash、runner/DSO/bundle manifest、
实测回调输出和最终选中记录。检查 warning、失败哨兵值、采样错误和普通 ATT 回退；
配置开关为 true 或进程退出成功不能替代搜索成功证据。
仓内带 stub 或 fake callback 的 E2E 用例只能验证协议，不能作为快模板实测证据。

## 确认 PGO 选中的模板和 tiling

为默认和选中 PGO 结果分别建立运行目录及最终证据：

```bash
python3 autofuse/tools/att_analyze/src/att.py evidence <default-log> -o <out>/default-evidence
python3 autofuse/tools/att_analyze/src/att.py evidence <pgo-replay-log> -o <out>/pgo-evidence
```

GE 默认运行应有有效的 `source="runtime" selection_mode="default"` FINAL；
PGO 配置加载后应核对有效的 `source="pgo" selection_mode="pgo"` FINAL。
将最终配置中的选择与 FINAL 的 graph/result/group/case_id/tiling_key/sub_case_tag、
template、完整 `tiling_repr`、block_dim 对应。多 group 结合可用的 FINAL_TILING_SUMMARY 和
已确认的 group 清单，缺少完整组合证据时标明缺口。
`search.txt` 的某条最快记录、生成阶段的候选打印或 `config.txt` 存在均不能单独证明实际执行身份。

Inductor 的 `topn_output output_index=0` 先作为导出候选证据。核对是否走了真实实测回调、
最终选中的 key/repr/hash，以及它与已编译二进制和实际 launch 的对应关系。
可用 `att.py topn-templates <worker-log> -o <out>/topn-templates` 整理导出候选，
但该命令不提供设备实测或运行期最终选择证明。缺少运行期 FINAL 时，保留前端选中、编译与 launch
的原始映射证据及其边界，不能伪造 PGO FINAL。

同一模板可能仅改变 tile 或 block_dim，也应保留为独立解。跨重新生成产物时 case/key 可能重编号，
应核对当前产物的完整身份及 tiling/API 映射，不能只比较编号或模板名。
如需显式重放 PGO 解，先查当前版本的强制选择接口并核验完整 tiling；
该重放记录保留其 `runtime/explicit` 身份，并另外保存它与原 PGO 解的对应证据。
只固定 case 而未固定实际 tiling，不能认定复现了 PGO 的解。

## 对选中结果做独立 profiling

PGO 搜索完成后，在独立进程中重放默认与 PGO 选中结果。
PGO 重放保持加载已选配置所需的开关和产物，检查没有重新进入搜索或回退普通 ATT，
然后再启动外部 profiler。若无法将重放与调优分开，先补齐结果固化/重放方法，
不把包含候选搜索的整段时间当作选中模板的耗时。

优先使用任务中已有的框架 profiling 入口，避免重复嵌套采集。
需要直接使用 msprof 时先查目标环境的 `msprof --help`，确认版本支持的 application、
task time、AI Core 指标和导出选项。下例仅表示已完成调优后的重放采集模板：

```bash
msprof --application="<default-replay-command>" --output=<out>/default-prof \
  --task-time=on --ai-core=on --aic-metrics=PipeUtilization
msprof --application="<pgo-replay-command>" --output=<out>/pgo-prof \
  --task-time=on --ai-core=on --aic-metrics=PipeUtilization
```

按实际工具版本导出并保留原始 PROF、采集命令和 `op_summary*.csv`。
两侧使用相同的 warmup、采样和缓存条件，分别记录用于性能判定的时延与深度指标采集的用途。
检查目标调用的 `Op Name`、`Task Duration(us)`、`Block Num`、`aiv_total_cycles` 和
`aiv_mte2_ratio`/`aiv_mte3_ratio`/`aiv_vec_ratio`，按调用与最终身份映射，排除编译、
PGO 搜索、warmup 及无关 kernel。重复采样的处理沿用主 skill 的歧义和统计规则。
缺少 pipe 指标时仍可报告有效的时延对比，但相关估值倍差留空。
`--att_profiling` 用于 ATT 模块自身执行时间定位，不能替代上述设备 kernel 与 pipe 指标采集。

搜索的内部时延、独立 kernel 时延与端到端时延分开记录，并查采样实现确认单位。
当前内部 collector 的 `AutofusePgoCollectorGetCandidateDurationNs` 返回 ns，不能直接与
`Task Duration(us)` 或 ATT cycle 比较。PGO 内部测量不能替代同输入的正常执行精度验证。

## 提取估值并与默认选择比较

从各自 run 的 FINAL `pipe_estimates` 获取对应最终 tiling 的模型成本，再执行：

```bash
python3 .claude/skills/att-perf-selection-analysis/scripts/est_vs_measured.py \
  --plog <default-log> --op-summary <default-op-summary.csv> \
  --plog <pgo-replay-log> --op-summary <pgo-op-summary.csv> \
  -o <out>/default_vs_pgo_pipes.csv
```

名称不一致时传入基于原始证据的 `--op-map`。只解释状态有效的记录；PGO 只有最终配置而无
FINAL 估值时，不能借用默认运行中同名 case 的最优估值，因为 PGO 可能改变了 tile 或 block_dim。
需要补估值时，用已确认的完整 PGO tiling 代入同版本生成公式，记录源码行、参数及计算过程，
与脚本提取值分开标注；无法确定的项留空。score、搜索耗时和 modeled_perf 的含义按生产接口核对，
不能只凭字段名当作 ATT pipe cycle。

对比报告至少包含：

| 项目 | 默认运行 | PGO 选中结果 |
| --- | --- | --- |
| 输入、版本、加载库与缓存条件 | 原始证据 | 原始证据 |
| 最终选择与语义名 | 完整身份、FINAL 来源 | 完整身份、FINAL 或前端执行映射 |
| 实际 tiling 与资源 | repr、block_dim、workspace | repr、block_dim、workspace |
| 精度及独立实测 | 判定、单位、样本分布 | 判定、单位、样本分布 |
| 各 pipe 模型/实测/倍差 | 逐项及来源 | 逐项及来源 |
| result 排序 | 各 group、边罚项、objective | 相同模型下该解的代价及缺失项 |

给出 `默认时延 / PGO时延` 和 `(PGO时延 / 默认时延 - 1) × 100%`。
只有 PGO 解在独立同口径复测中更快且精度验证通过，才确认其为当前对照的快模板。
同身份同 tiling 却出现大幅时延差时，先排查产物、缓存和采集关联。
PGO 未更快或差异落在抖动范围时，记录没有确认到更快解，不能据此判定模型排序错误。
确认快解后进入主 skill 的公式分解与逐 pipe 替换排序分析，结论限定为本次成功搜索和实测的范围。
