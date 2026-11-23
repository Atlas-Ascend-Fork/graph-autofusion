# v35 FP32 Load / Store 上板采集

本工程使用现有 ATT 模型归档的输入描述符，支持普通 Load、单层硬件 LoopMode Load 和小块连续 Store。
原模型公式、采样角色与实测数据见 [Load 归档](../../../../att/modeling/v35/mte2/load/results/pr2921-d68/report.md)
和 [Store 归档](../../../../att/modeling/v35/mte3/store/results/pr2921-d68/report.md)。
其中训练、留出、已知候选和域外对照沿用归档标记，复采结果请写入独立目录。

## 构建

在支持 v35、设备 0 至少有 56 个可用 AIV 的 Linux 环境中使用已安装的 CANN 9.3.0、bisheng、msprof、
CMake 3.20+、Python 3.9+ 和 flock。设备 0 应空闲，所有采集任务使用同一设备锁。下列路径示例从仓库根执行：

```bash
source /path/to/cann/set_env.sh
tool=autofuse/tools/modeling/v35/dma
archive=autofuse/att/modeling/v35
work=/path/to/dma-calibration
lock=/path/to/device0.lock
mkdir -p "$work"
cmake -S "$tool" -B "$work/build" -DASC_DIR="$ASCEND_HOME_PATH/lib64/cmake" -DNPU_ARCH=dav-3510
cmake --build "$work/build" -j 4
```

默认使用本仓的 RegBase 头文件。复现时可用 `-DAUTOFUSE_ROOT=/path/to/frozen-source` 指向实测时冻结的源码，
并记录 `datacopy.h`、二进制和输入清单的 SHA-256。一个二进制可执行所选批次的全部描述符。

## 用例与采集协议

| 模型 | CSV policy | 目标程序 | 证据及采集版本 |
| --- | --- | --- | --- |
| LoadSmallPayload | `N`、`C` | `steady` | S01 原短序列结果继续使用现有归档。工程采用同系列的长调用校验版本 |
| LoadSmallPayloadCompact | `N`、`C` | `steady` | 同 S01。输入走 Compact Load，通过原描述符保留 GM 行间隙 |
| LoadAlignedSingleRow | `N` | `steady` | S04 OneRec 单行。使用同系列长调用版本，原训练/留出误差引用归档 |
| LoadAlignedWideGap | `fresh` | `steady` | S05 0238 独立留出。用 `C` 复采大 pitch 形态，采样序列与历史冷读批次不同 |
| LoadWidePitch128B | `C` | `steady` | S06 0254 窄行大 pitch。保留现有冷读报告，工程可复采描述符 |
| LoadAlignedDensePayload | `N-forward`、`C-forward`、`N-reverse` | `stability` | S03 原采集版本，32 次预热 + 64 次有效采样。展平输入转成等价的一行 DMA |
| LoadAlignedLoopPayload | 同上 | `stability` | S02 原留出采集版本，32 次预热 + 64 次有效采样 |
| LoadNarrowLoopPayload | 同上 | `loop` | S07 原采集版本，16 次预热 + 16 次有效采样 |
| StoreSmallPayload | `normal`、`bypass` | `store` | S01，使用同系列 Store 源码复采，4／16 次调用各 3 次预热 + 10 次有效采样 |

`N` 在核、调用和 launch 间轮换 GM 地址并使用 NORMAL L2 hint，`C` 另外设禁用 L2 hint。LoopMode 使用
原始三轴描述符。Store 使用 `DataCopyPadExtend<Normal>` 单行零间隙，按核、重复和样本轮换 GM 地址，
验证写回数据及保护区。Store 清单包含 7 核与超过 8 KiB 的归档对照点，产品适用域见 Store 设计文档。

LoopMode 短窗口和长窗口分别来自历史 `matrix17-int64-contiguous-slope-v1` 与 `matrix17-fp32-stability-v3`，
在本工程共用同一份源码，通过编译定义选择窗口。普通 Load 来自 `small-fp32-load-steady-domain-v1`，
每组 16／32／64／128 次调用。Store 来自 `transpose0139-wide-store-mte3-v1` 的同系列采集器，
复采限定小块连续布局，并为每个写回样本分配独立的对齐区域。原 S01／S04／S05／S06 普通 Load
及 S01 Store 的精确源码快照未在现有证据目录找到。以上替代采集仅用于复采相同描述符，
历史拟合数值继续引用已归档的原始结果。

## 示例

从现有清单生成采集输入。支持 `--role train`、`--role holdout` 等筛选，可重复指定角色。
只检查入口时可加 `--limit 1`。完整采集时不设上限。Load 的 profile 分别保留不同调用次数。
下面以窄行 LoopMode 和连续 Store 为例：

```bash
python3 "$tool/prepare_cases.py" "$archive/mte2/load/cases.csv" \
  LoadNarrowLoopPayload N-forward "$work/narrow-N.csv"
ATT_HOST_THREADS=4 ATT_FIXED_POOL_GIB=4 bash "$tool/collect.sh" \
  loop "$work/build" "$work/narrow-N.csv" N "$lock" "$work/narrow-N"

python3 "$tool/prepare_cases.py" "$archive/mte3/store/cases.csv" \
  StoreSmallPayload normal "$work/store-normal.csv"
bash "$tool/collect.sh" store "$work/build" "$work/store-normal.csv" normal "$lock" "$work/store-normal"
```

按同一命令切换表中的模型、policy、程序和缓存模式。复现 `N-reverse` 时，保持文件中该 policy 的归档顺序，
将采集模式设为 `N`。大 pitch 更耗 HBM，`steady` 可按实测协议提高 `ATT_FIXED_POOL_GIB`，例如 S06 使用
24 GiB 的输入池。采集器会检查设备可用内存和描述符，待采一组有足够空间再使用相应预算。

每次采集使用新目录，保留精度回执、launch JSONL 和 msprof 的 op_summary。对照 launch 的 ID、核数、样本号、预热、
缓存提示和实际 kernel 后，计算每核流水周期 `aiv_total_cycles × aiv_mte2_ratio / Block Num`
或 `aiv_total_cycles × aiv_mte3_ratio / Block Num`。Load 的 S07 主结果取 K=1 中位数，旧短序列及
长调用校验按各自报告的调用窗口统计。Store 用 16 次与 4 次的周期中位数差除以 12。
[现有拟合复验工具](../mte2/README.md)可直接复核归档的 Load 数据。

## FP32 窄连续源 NDDMA

原始描述符、训练与独立验证见
[用例](../../../../att/modeling/v35/mte2/nddma/cases-narrow-source.csv)及
[结果](../../../../att/modeling/v35/mte2/nddma/results/narrow-source-v1/report.md)。
使用相同的队列、地址隔离、L2 hint、同步和保护区检查，按调用次数拟合 MTE2 斜率。
同一程序批量读取清单，按需选择角色。

从仓库根执行以下命令，延续前文设置的 tool、work、lock。

~~~bash
archive=autofuse/att/modeling/v35/mte2/nddma
python3 "$tool/prepare_cases.py" "$archive/cases-narrow-source.csv" \
  'NG2 narrow source FP32' N_fresh_normal "$work/nddma.csv" --role train --role holdout
ATT_HOST_THREADS=4 ATT_FIXED_POOL_GIB=49 bash "$tool/collect.sh" \
  nddma "$work/build" "$work/nddma.csv" N "$lock" "$work/nddma-run"
~~~

nddma 为 1／2／4／10 次调用，每组 3 次预热和 10 次有效采样。
长轴冻结验证使用 --role frozen_fit_validation 生成另一份清单，
选择 nddma_long 程序，1／2／3／4 次调用，每组 1 次预热和 4 次有效采样。
两种计量窗口各自统计，不混合拟合。
49 GiB 输入池用于大跨度 fresh GM 复采，工具同时检查实际空闲 HBM 和校验输出预算。
较小描述符可以降低该输入池参数。

## NDDMA 完整写回对照

[直通流水归档](../../../../att/modeling/v35/mte2/nddma/results/direct-store-v1/design.md)区分孤立 NDDMA 与完整写回的有效 MTE2 成本。
延续上文 tool、work、lock，使用同一个二进制、清单和 NORMAL hint：

```bash
archive=autofuse/att/modeling/v35/mte2/nddma/results/direct-store-v1
ATT_HOST_THREADS=2 ATT_FIXED_POOL_GIB=4 bash "$tool/collect.sh" \
  nddma_full_store "$work/build" "$archive/cases.csv" N "$lock" "$work/full-store"
ATT_HOST_THREADS=2 ATT_FIXED_POOL_GIB=4 bash "$tool/collect.sh" \
  nddma "$work/build" "$archive/cases.csv" N "$lock" "$work/tail-store"
python3 "$tool/summarize_nddma.py" "$archive/cases.csv" "$work/full-store" "$work/samples.json"
python3 "$tool/fit_direct_store.py" "$work/samples.json" "$work/refit.json" --plot "$work/fit.svg"
```

先按前文计量方式计算 K 斜率；归档 samples.json 保留各 K 中位值及斜率，拟合仅使用 role=train 且 GM 连续源不超过一个 32B DMA block 的点。
其余采样留作控制，不用于这条分支。独立 holdout 清单保存在归档的 independent-holdout-cases.csv。
`span_store` 复用采集器已有分支，每次写回完整 UB span；`tail_store` 只写回尾部校验数据。
新输出文件必须尚不存在，避免混入历史拟合。

`--plot` 可选，需要 matplotlib；不指定时拟合只依赖 Python 标准库。

## 稠密窄行普通 Load 对照

`LoadAlignedDenseBulkPayload` 对应 `LOAD_NORMAL`。复用 `loop` 程序，原始三 API 对照、补充训练和冻结后留出分别用归档的 `api-controls.csv`、`extra-train.csv`、`frozen-holdout.csv`；不要用二维描述符代替硬件 LoopMode 的三轴输入。`summarize_nddma.py --samples-per-k 16` 可校验混合 API 的 launch/profile 配对并计算 K=1/2/4/10 斜率；默认仍为原 NDDMA 短窗口的 10 个有效样本。完整命令和结果见[报告](../../../../att/modeling/v35/mte2/load/results/dense-bulk-v1/report.md)。
