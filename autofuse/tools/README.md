<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Autofuse 工具说明

这里的工具用于开发调试和问题定位，不属于 Autofuse 运行时接口。下面按“职责—输入—命令—效果”说明当前工具；参数变化以命令的 `--help` 为准。

## NDDMA：建模与单配置预测、采集

工具目录：[`nddma/`](nddma/README.md)。

- 职责：Round1–5 数据采集、拟合与分析，以及 ad-hoc 单配置预测和实测。
- 输入：shape、GM/UB stride、核数、dtype，或已有测量 CSV 和模型 JSON。
- 使用（从仓库根目录）：

  ```bash
  python3 autofuse/tools/nddma/Modeling/Data_collection/common/tools/predict_nddma_ad_hoc.py --help
  python3 autofuse/tools/nddma/Modeling/Data_collection/common/tools/run_nddma_ad_hoc.py --help
  ```

- 效果：输出 cycles 预测、公式分项、拟合参数及测量分析结果。历史快照的 metrics 已标记无效；实际采集需要 CANN/Ascend 环境。


## ATT-ANALYZE：模板、tiling 和 profiling 分析

工具目录：[`att_analyze/`](att_analyze/)，统一入口：

```bash
python3 autofuse/tools/att_analyze/src/att.py --help
```

### `summary`：汇总 ATT 日志

- 职责：从 `[PROF]` 日志提取算子、graph/result/group/case、tiling 参数、objective 和 result performance。
- 输入：一个日志文件或日志目录。
- 使用：

  ```bash
  python3 autofuse/tools/att_analyze/src/att.py summary run.log -f csv -o summary.csv
  ```

- 效果：生成包含 `Operator`、`Case`、`Objective Value` 等列的 CSV；日志字段缺失时保留空值并标记 `parse_status`，不会用 0 或 objective 冒充实测值。

`FINAL_TILING` 运行期记录使用可读值 `src=runtime`（历史紧凑别名 `src=r` 也兼容），候选记录使用 `src=compile` 或
`src=candidate`。记录至少包含 `graph/result/group/case/key`、模板名、`score`、`pipe_est` 和
`tiling_repr`；`case` 与 `key` 不应混用。单行记录超过约 800 字符时使用
`FINAL_TILING_BEGIN/CHUNK/END`，解析器校验 chunk 顺序、长度和 ATT-Mix64-v1，失败标记
`incomplete_final_tiling` 且不保留部分 repr。只有 plog 时只能分析候选；加入 profiling
后可以比较模型估值与实测 cycle，但仍需 `FINAL_TILING` 才能确认运行期最终选择。

### `compare`：比较两次选择结果

- 职责：比较默认配置与 PGO、强制模板或其他候选配置的 CSV，识别字段和性能差异。
- 输入：两个 `summary` 生成的 CSV；同一 graph/result/group 即使 case 不同也会匹配。
- 使用：

  ```bash
  python3 autofuse/tools/att_analyze/src/att.py compare default.csv candidate.csv -f text -o compare.txt
  ```

- 效果：输出共同算子、只存在于一侧的算子、case/tiling 差异和性能变化；负的周期差通常表示候选更快。

### `evidence`：导出机器可读证据

- 职责：把原始 ATT 日志转换为带来源行号和解析状态的 JSONL，供 skill 或其他程序继续分析。
- 输入：日志文件或目录。
- 使用：

  ```bash
  python3 autofuse/tools/att_analyze/src/att.py evidence run.log -o evidence/
  ```

- 效果：生成 `att-evidence.jsonl` 和工具清单；每条记录包含算子、case、tiling、objective、`source_path`、`source_line` 和 `parse_status`。

### `split-slog`：拆分 DFX/PROF 日志

- 职责：按算子和 graph/result/group/case 拆分编译期 `[DFX]` model-info 片段及运行期 `[PROF]` 片段。
- 输入：slog、stdout 或日志目录；不要求固定文件名。
- 使用：

  ```bash
  python3 autofuse/tools/att_analyze/src/att.py split-slog slog/ --op FlashAttentionScore --case r=1,g=0,c=2 -o split/
  ```

- 效果：在 `split/<operator>/compiler/...` 和 `split/<operator>/runtime/...` 下生成真实编号的 `case*.log`，便于逐 case 查看 DFX 边界。

### `perf-formula`：分析 `[PERF]` pipe 公式

- 职责：解析 tiling 函数输出的 Load/Store/Vector pipe 公式，标出瓶颈并比较多个 case 的敏感参数。
- 输入：包含 tiling 源文件的目录和 `[PERF]` 日志。
- 使用：

  ```bash
  python3 autofuse/tools/att_analyze/src/att.py perf-formula generated/ run.log --case r=0,g=0,c=1 -o perf/
  ```

- 效果：生成 `perf_formula.svg`；图中红色节点是当前最大 pipe，跨 case 区域显示方差最大的参数。没有 `[PERF]` 时命令返回非零并提示证据不足。

### `verify-tiling`：验证 TensorFlow/Inductor tiling ABI

- 职责：编译用户提供的 tiling 函数并按明确 ABI 调用 `AutofuseTiling`，检查 block_dim 和 workspace 返回值。
- 输入：TensorFlow tiling C++ 文件目录，或包含 `output_code.py` 的 Inductor 目录；需要 Ascend 编译环境。
- 使用：

  ```bash
  python3 autofuse/tools/att_analyze/src/att.py verify-tiling generated/ \
    --scene inductor --input-json input.json --keep-build -o verify/

  # TensorFlow 动态 ABI；执行前会打印实际使用的 aiv_num，--aiv-num 可覆盖 preset
  python3 autofuse/tools/att_analyze/src/att.py verify-tiling generated/ \
    --scene tf --preset B --aiv-num 56 -o verify/
  ```

- 效果：控制台显示编译和运行结果，`verify/result.json` 保存状态、返回值和失败原因；编译或运行失败会返回非零退出码。`output_code.py` 只读取字面量 artifact，不执行其中代码。
- 执行前会打印 `aiv_num`、`ub_size`、动态维度和配置来源，方便用户核对硬件；`preset_B` 默认 `aiv_num=56`，不应视为所有芯片的固定规格。`aiv_num` 仅用于 TensorFlow tiling ABI，Inductor ABI 不使用它。

## NWA `fusion_precision_analyzer`：融合精度定位

工具目录：[`nwa_tool/`](nwa_tool/)。当关闭自动融合精度正常、开启自动融合精度下降时，比较两侧 dump 图和 NPY 数据，定位造成误差的融合算子。

- 模式 1（默认）：按 dump 图映射批量比较融合算子输入/输出。
- 模式 2：直接比较两个 NPY 文件。

示例：

```bash
python3 autofuse/tools/nwa_tool/fusion_precision_analyzer.py \
  --af-open-graph open/Build.json --af-close-graph close/Build.json \
  --af-open-data open/npy --af-close-data close/npy --compare-input

python3 autofuse/tools/nwa_tool/fusion_precision_analyzer.py --mode 2 \
  --npy-a open.npy --npy-b close.npy
```

效果：输出余弦相似度、最大绝对误差、最大相对误差及状态（如 `OK`、`FILE_NOT_FOUND`、`SHAPE_MISMATCH`）；完整参数和格式转换规则见 [`nwa_tool/README.md`](nwa_tool/README.md)。

## 共同注意事项

- ATT `[PROF]`/`[DFX]` 采集通常需要在用户执行命令中设置 `ASCEND_SLOG_PRINT_TO_STDOUT=1`、`ASCEND_GLOBAL_LOG_LEVEL=1`；`att_profiling` 只表示 tiling 函数耗时统计。
- skill 默认只读。重新执行、编译、profiling 或 PGO 必须由用户提供命令并明确授权。
- 原始证据和分析结论应分开归档，已有归档不会覆盖。
- 安装依赖：ATT Excel 输出需要 `openpyxl`，NWA 工具需要 `numpy`。
- `autofuse/tools/att_analyze/tests/data/` 是固定回归样例。真实日志格式变化时，应脱敏新增 fixture 并同步 `summary`/`evidence` 期望结果，不要覆盖旧样例。
