<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# NDDMA 建模与 ad-hoc 工具

[English](README_en.md)

用于 NDDMA 数据采集、性能公式拟合、预测和误差分析，属于开发工具。
Round1–5 建模流程和 ad-hoc 单配置工具统一放在本目录。

## 目录

```text
nddma/
├── DOC/                             # 建模说明
├── Modeling/Ana/round1/ ... round4/  # 参数 JSON，与拟合输出路径一致
├── Modeling/Data_collection/
│   ├── common/tools/                # ad-hoc 预测与采集入口
│   ├── common/executables/          # 采集 kernel 和 profiling 分析
│   └── round1/ ... round5/          # 各轮 collect/fit/draw/e2e
└── Modeling/tests/                  # 公式及输入校验回归测试
```

## 依赖

- Python 3.9+，拟合和测试需要 NumPy：`python3 -m pip install numpy`。
- 预测、拟合和绘图不依赖 GE 源码。
- 设备采集需要兼容的 CANN Toolkit、Ascend 设备和 `msprof`。
- 默认引用 `autofuse/v35/ascendc/api_regbase/datacopy_nddma.h`。
  采集脚本支持 `--cmake-extra-arg=-DREGBASE_INCLUDE_DIR=/path/to/api_regbase`
  指定仓库外头文件；无需复制 ge-develop。

## ad-hoc：单配置预测与采集

以下命令从 graph-autofusion 仓库根目录运行，参数以 `--help` 为准。
shape 和 stride 按外轴到内轴排列，stride 单位为元素。

```bash
# 通用 1D–5D 预测：shape、GM stride、UB stride、核数、dtype
python3 autofuse/tools/nddma/Modeling/Data_collection/common/tools/predict_nddma_ad_hoc.py \
  12x32x32 512x65536x2097152 1024x32x1 72 int32_t --kernel-repeat 72

# 二维转置预测，要求 UB stride=[N,1] 且 is2<is1
python3 autofuse/tools/nddma/Modeling/Data_collection/common/tools/predict_nddma_2d_transpose_ad_hoc.py \
  63x2 1x2 2x1 8 int16_t

# 实际编译、运行并采集一个配置，需要设备和 profiling 环境
python3 autofuse/tools/nddma/Modeling/Data_collection/common/tools/run_nddma_ad_hoc.py \
  63x2 1x2 2x1 8 int16_t --msprof-bin "$(command -v msprof)"
```

预测输出公式分项和 cycles，并保存 JSON 等分析结果。采集工具输出测量与
profiling 分析结果。可用 `--output-dir` 指定输出目录；默认生成目录已被 Git 忽略。

## Round1–5：采集、拟合与分析

| 轮次 | 职责 | 输入与输出 |
|---|---|---|
| Round1 | 连续 1D 基线 | 测量 CSV → T/H 参数 |
| Round2 | 非连续 1D 与多核修正 | Round1 参数及测量 → NG/NGU/rho |
| Round3 | 2D–5D 统一展开 | 继承 Round2 参数，逐维评估 |
| Round4 | 二维转置特化 | Round2 参数及 D/F/G 测量 → NG2 |
| Round5 | 3D 特化 | 固定 Round2 基线，重新拟合修正项 |

### 运行准备与顺序

本节所有命令都在工具目录执行。先从仓库根目录进入：

```bash
cd autofuse/tools/nddma
python3 -m pip install numpy
# 在设备侧加载已安装的 CANN 环境；替换为实际安装位置
source /path/to/cann/set_env.sh
command -v cmake
command -v msprof
```

只做已有 CSV 的拟合、绘图时不需要设备和 profiling 环境。
完整采集需要 `cmake`、ASC 编译器、可用设备和 `msprof`，默认编译并行度为 8。
本采集工具仅面向大于等于 v35 NDDMA API。设备采集前必须设置 `NPU_ARCH` 环境变量
（例如 `export NPU_ARCH=dav-3510`，取值需匹配实际设备和 CANN），或直接调用
`e2e.py --npu-arch ...`、`collect.py --npu-arch ...` 或 ad-hoc 采集入口。
显式参数优先于环境变量；e2e 仅在 `all` / `collect` 阶段转发该参数。
未配置时，实际采集会在写文件及编译前退出并显示配置方法；collect/ad-hoc 的
`--dry-run` 会显示警告并继续预览。拟合和绘图不要求配置架构。

```bash
python3 Modeling/Data_collection/round1/scripts/e2e.py all \
  --npu-arch dav-3510 --msprof-bin "$(command -v msprof)"
```

CPU 调试还需通过 `--cmake-extra-arg=-DNDDMA_CPU_TARGET=<target>` 指定
所安装 tikicpulib 提供的 v35 target；工具不再硬编码芯片默认值。
采集 kernel 仅执行 MTE2 NDDMA，不支持 `--enable-store`，新 CSV 不包含 `enable_store`。
旧 CSV 的额外列可以读取，但不再触发 MTE3。
所有 collect/ad-hoc 的 `--dry-run` 仅预览命令，不创建、改写或删除文件；
即使同时指定 `--generate-only`，dry-run 仍优先。

从零开始时按以下依赖顺序执行，并检查每一步退出码：

```text
Round1 → Round2 → Round3（通用多维评估）
                ├→ Round4（二维转置 NG2 拟合）
                └→ Round5（3D 修正项拟合）
```

Round3、Round4、Round5 相互不依赖，可以在 Round2 完成后分别运行。
每轮的 `all` 仅执行本轮 `collect → fit → draw`，不会自动执行上游轮次。
任一步返回非零退出码后，本轮流程停止。

| 操作 | 行为 | 是否需要设备 |
|---|---|---|
| `e2e.py all` | 生成用例、编译采集、解析 profiling、拟合、绘图 | 是 |
| `e2e.py collect` | 仅采集并解析测量结果 | 是 |
| `e2e.py fit` | 使用已有测量及上游模型生成模型 JSON 和预测 CSV | 否 |
| `e2e.py draw` | 使用拟合生成的预测 CSV 绘图 | 否 |
| `e2e.py all --dry-run` | 仅打印将调用的命令，不执行子脚本 | 否 |

默认目录是 `Modeling/Ana/roundX/`：`collection/` 保存编译、运行、profiling 和分析结果，
模型 JSON 与预测 CSV 位于该轮目录顶层，图表在 `figures/`。
`fit` 默认递归查找本轮 `collection/` 下的 `profiling_with_params_mean.csv`，
若没有则查找 `profiling_with_params.csv`。`--measurement-csv` 可直接指定分析后的测量 CSV；
它不是原始 `op_summary_*.csv`，必须同时包含配置参数和测量字段。
不要把不同硬件、API 实现或重复采集的结果混在同一个拟合输入目录里。

仓库已提供 Round1–4 的历史参数 JSON，但没有提供原始测量和预测 CSV。
因此干净检出可以使用历史参数预测，却不能直接执行 `draw`：需要先采集或提供 CSV 完成 `fit`。
默认拟合会覆盖同名模型 JSON；ad-hoc 工具下次运行就会读取更新后的参数。



```bash
python3 Modeling/Data_collection/round1/scripts/e2e.py all \
  --msprof-bin "$(command -v msprof)"
```

或者分步执行，便于检查测量后再拟合；与上面的一键运行二选一即可：

```bash
python3 Modeling/Data_collection/round1/scripts/e2e.py collect --msprof-bin "$(command -v msprof)"
python3 Modeling/Data_collection/round1/scripts/e2e.py fit
python3 Modeling/Data_collection/round1/scripts/e2e.py draw
```

如果已有测量 CSV，跳过采集，用下面的命令重拟合后再执行 `draw`：

```bash
python3 Modeling/Data_collection/round1/scripts/e2e.py fit \
  --measurement-csv /path/to/round1_measurements.csv
python3 Modeling/Data_collection/round1/scripts/e2e.py draw
```

输出在 `Modeling/Ana/round1/`：

- `round1_1d_single_multi_core_model.json`：模型参数、公式和统计信息。
- `round1_1d_single_core_predictions.csv`：逐配置预测与实测，用于绘图和检查误差。
- `figures/`：SVG 图表。

## 参数与结果口径

仓库预置的 Round1–4 JSON 为历史参数快照，均标记 `snapshot.metrics_valid=false`。
预测工具会提示，旧 metrics 不代表当前公式、训练网格或对齐参数的精度。
Round3 JSON 的基线说明与当前实现一致，使用 `B/T+H`。
仓库不预置 Round5 拟合结果，需要运行 Round5 生成。测量数据、日志、构建产物不入库。

## 验证

在工具目录执行（与上面各轮命令使用相同工作目录）：

```bash
python3 -m unittest discover -s Modeling/tests -v
```

外部参考模型对比需设置 `HW_NDDMA_MODELING`；未设置时相关测试跳过。
输入校验的主机端测试需要 C++ 编译器。

重新拟合会覆盖同路径 JSON，ad-hoc 工具下次运行即读取新参数。Git 仅放行四份模型 JSON，其他生成结果仍被忽略。
