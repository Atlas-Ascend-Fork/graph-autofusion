# IsInf 上板性能验证工具

单 API 上板验证代码，用于 IsInf 性能建模的第一步：测量各 dtype 分支在不同 `count` 下的执行周期，为拟合 `cycles = a * repeat_time + b` 提供实测数据。

## 源码与分支依据

- API 声明：`asc-devkit/include/adv_api/math/is_inf.h`（官方 API，v35 目标 arch 为 `__NPU_ARCH__ == 3510`）。
- API 实现：`asc-devkit/impl/adv_api/detail/math/isinf/is_inf_common_impl.h` `IsInfImplVF`。
- codegen 调用：`autofuse/v35/codegen/reg_api_call/unary_bitwidth_change_api_call_v2.cpp`，输出恒经 `ReinterpretCast<bool>`。
- 设备：dav-3510（Ascend 950PR/950DT 系列）；910B 等其他架构不在 `is_inf.h` 支持列表内，无法运行。

覆盖的 4 个源码 dtype 分支：

| combo | DstT | SrcT | repeat_elm | 源码差异 |
| --- | --- | --- | --- | --- |
| `bool_float` | bool | float | 64 | MaskPack×2 + 尾部 MaskPack×2，CompareScalar(uint32)×2 |
| `bool_half` | bool | half | 128 | MaskPack×1 + 尾部 MaskPack×1，CompareScalar(uint16)×2 |
| `float_float` | float | float | 64 | 无 MaskPack，Select 直接输出 |
| `half_half` | half | half | 128 | 无 MaskPack，Select 直接输出 |

其中 `bool_float` / `bool_half` 为 v35 codegen 实际路径。

## 测量口径

- 单 block 执行，kernel 内使用 `TPipe` + `TQue`（VECIN/VECOUT）分配 UB，与 codegen 生成的 kernel 结构一致；不带 tmpBuffer 的 `IsInf` 形式内部 `PopStackBuffer` 依赖 TPipe 上下文，不可用 `LocalMemAllocator` 替代。
- **bench 不做内部计时**：性能数据完全由 npu 模式 msprof op_summary 采集（见批量采集说明）。
- **无 warmup**：msprof op_summary 采集的是整个 kernel 的 AIV 周期，kernel 内 API 调用总次数必须恒等于 `loop_count`（含额外调用会使解析值按倍数偏大）；冷启动开销（icache miss 等）计入拟合截距 b。
- API 内部 `repeat_time = ceil(count / repeat_elm)`，`repeat_elm = 256 / sizeof(SrcT)`。
- 输入含 +Inf/-Inf/普通值，每次运行同时校验前 `count` 个输出的正确性。

## 编译

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
mkdir -p build && cd build
cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..
make -j8
```

CPU 仿真验证正确性（不输出周期）：

```bash
cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..
make -j8
```

## 运行

### 单点手测

```bash
./isinf_perf_bench <combo> <count> [loop_count]
# 例：
./isinf_perf_bench bool_float 1024 1000
```

输出（CSV）：

```text
VERIFY,PASS,bool_float,1024
RESULT,bool_float,1024,64,16,1000
```

`RESULT` 列含义：`combo, count, repeat_elm, repeat_time, loop_count`（参数回显，无性能字段；
性能数据由 msprof op_summary 采集）。

### 批量采集（run_bench.py，用例读自 cases.csv）

用例定义在本目录 `cases.csv`，列 `combo,count,repeat,loop_count`（`repeat` 即 repeat_time，
`ceil(count/repeat_elm)`，工具会校验一致性）。当前用例矩阵：每个分支 8 个对齐点
（`repeat_time ∈ {1,2,4,8,16,32,64,128}`，`count = rt * repeat_elm`），共 32 例；
`loop_count` 固定为 1（计时循环内单次 API 调用，直接测单次完整周期）。
每点固定重复执行 5 次取中位数。

```bash
# CPU 仿真跑通全链路（本机可测，仅验证正确性，loop_count 自动收敛）
python3 run_bench.py --run-mode cpu --cases mini_cases.csv

# 目标设备（dav-3510）实测全量用例；npu 模式自动开启 profiling：
# 每个用例经 msprof --parse=on --application="{exe} {combo} {count} {loop_count}" 调起，
# profiling 数据按批次隔离落盘：
#   {profile-dir}/{run-id}/{combo}_{count}/PROF_{timestamp}_*/
# run-id 标识一次执行批次（--run-id 指定，默认 run_YYYYmmdd_HHMMSS），多次运行产生的
# 多个 PROF 文件均在当前批次目录下，解析只取本次 run-id 的数据
source ${ASCEND_HOME_PATH}/set_env.sh
python3 run_bench.py --output samples.csv

# 尾块点等外推验证补测（另建 csv，不改正式用例集）
python3 run_bench.py --skip-build --cases extra_cases.csv
```

npu 模式下每个用例测试完后在其 profiling 子目录递归查找 `op_summary_*.csv`，
取 `OP Type` 值包含被测 API 名称（本工具 `API_NAME = "isinf"`，忽略大小写）的行，
读 `Block Num`、`aiv_total_cycles`、`aiv_vec_ratio` 三列，计算
`aiv_total_cycles / Block Num * aiv_vec_ratio` 作为本用例实测值
（多次运行取中位数），刷新 samples.csv 的 `measured_cycles`（npu 模式实测值以 msprof 解析为准，
bench 内部 clock 不再作为实测来源）。

npu 模式启动时经 `npu-smi` 自检设备环境：命令可用则打印设备信息（确认目标形态、归档环境证据）；
命令不存在或执行失败时不阻断，默认按 `--arch`（dav-3510）继续。

`profiling/` 不入 git，但**测试后必须保留**（实测原始证据，归档时由
`results/<platform>/<run_id>/manifest.json` 索引关联），任何验证、清理操作不得删除。

输出 `samples.csv` 包含 cases.csv 中所有用例（逐行一致，失败点也占行），追加两列：

| 列 | 填写时机 | 说明 |
| --- | --- | --- |
| `measured_cycles` | 本工具实测时填写 | 计时循环**总周期**的中位数；拟合时除以 loop_count 得单次周期。失败/无效点留空 |
| `model_predict_cycles` | 拟合出 a、b 并完成 perf 代码后回填 | 按 `a * repeat + b` 计算，用于模型与实测对齐归档 |

npu 模式下实测 cycles 为 0 判 INVALID，存在 FAILED/INVALID 时退出码非 0。

### 拟合（fit_curve.py）

读取 samples.csv 的 `loop_count,repeat,measured_cycles` 列按 combo（源码分支）分组拟合：
先归一化 `y = measured_cycles / loop_count`（单次调用周期），再最小二乘拟合
`y = a * repeat + b`，每种分支独立一组 a、b 与 R2、逐点残差，输出到本目录 `report.md`。

```bash
python3 fit_curve.py                        # 读 tools/samples.csv，写 tools/report.md
python3 fit_curve.py --result r.csv --report report.md
```

`report.md` 总是生成并保留：数据不足的分支在报告中标注原因（不中断其他分支），
存在未拟合分支时退出码非 0。`measured_cycles` 为空或非正的行跳过（0 值占位数据不参与拟合）。
当前 report.md 基于 CPU 仿真占位数据（全 0），各分支均为"未拟合"状态，
待目标设备实测后重新执行生成正式拟合结果。

## 拟合用法

对每个 combo 扫描多组 `count`（建议覆盖 repeat_time 1~256 及尾块点），收集 `RESULT` 行后按
`cycles_per_call = a * repeat_time + b` 做最小二乘拟合，得到各分支的 `a` 与 `b`。

## 已验证情况

- dav-3510 NPU 模式编译通过；`run_bench.py` npu/cpu 两种模式全链路（编译、矩阵、运行、采集、CSV）验证通过。
- CPU 仿真（`CMAKE_ASC_RUN_MODE=cpu`）下 4 个 combo 及尾块场景（count=1/63/65/129 等）全部 `VERIFY,PASS`。
