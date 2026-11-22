---
name: af-api-modeling
description: Use when graph-autofusion 用户需要对单个 AscendC API 做上板性能建模：定位分析 API 源码的输入与逻辑分支、编写单 API 上板验证代码（bench）、编写编译/传参/运行采集工具、设计测试用例实测、拟合各分支公式（a*repeat+b）并编写 perf 计算代码，归档到 autofuse/att/modeling。涉及 isinf/isnan/isfinite/cast/compare 等 API 的 perf bench、上板验证、repeat 拟合时必须使用。
---

# 单 API 上板性能建模流程

对单个 AscendC API 建立基于实测的性能公式。最终每个源码分支得到简化公式 `cycles = a * repeat_time + b`（拟合出 a、b），并落成 API 的 perf 计算代码。

核心原则：**分支划分必须与 API 源码实现一一对应；公式参数 a、b 必须来自上板实测拟合，不允许只凭指令表推算。**

## 总流程

| 步骤 | 内容 | 状态 |
| --- | --- | --- |
| 1 | API 源码定位，输入与逻辑分支分析，编写上板验证代码（bench） | 已完善，见下文 |
| 2 | 编写运行工具（编译、读用例、批量运行、数据采集） | 已完善，见下文 |
| 3 | 设计测试用例（`tools/cases.csv`）并调用工具实测 | 已完善（用例文件就绪，实测需目标设备） |
| 4 | 根据实测数据拟合各分支公式 a*repeat+b | 已完善，见下文 |
| 5 | 编写 API 的 perf 计算代码并归档 modeling | 已完善，见下文 |

每完成一步，逐步补充本文档对应章节与工具，本文档随流程演进。

## 通用约定

- 归档位置：`autofuse/att/modeling/`，按其 README.md 分类定位 API 目录（如 vector/elewise/isinf/）；
  已有目录复用，无则按归档约定新建（design.md / cases.csv / results/）。
- 上板验证与工具代码放 `<api 目录>/tools/`：
  `<api>_perf_bench.asc`（bench 主体）、`CMakeLists.txt`、`README.md`、`run_bench.py`（运行工具）。
- 目标形态：dav-3510（`__NPU_ARCH__ == 3510`，Ascend 950PR/950DT 系列）。
- 公式形态：`cycles_per_call = a * repeat_time + b`；`repeat_time = ceil(count / repeat_elm)`，
  `repeat_elm = GetVecLen() / sizeof(SrcT)`（v35 GetVecLen()=256B，fp32=64、fp16=128）。
- 参考实例：`autofuse/att/modeling/vector/elewise/isinf/tools/`（IsInf，首个完整样例）。

## 第一步：API 源码定位与上板验证代码

### 1.1 定位 API 源码

沿用 af-perf-modeler 的定位流程，必须依次完成：

1. 在 `autofuse/v35/ascir/generator/v2_ascir_codegen_impl.h` 找 `{Api}AscIrCodegenImplV2`，
   读取 `GetApiName()`、`GetApiCallName()`、`LoadApiHeaderFiles()`、`IncludeApiHeaderFiles()`。
2. 判定源码归属：
   - `LoadApiHeaderFiles()` 返回 `*_reg_base.h` → 项目内 regbase，V2 在 `autofuse/v35/ascendc/api_regbase/`。
   - `IncludeApiHeaderFiles()` 返回 `adv_api/...`、`basic_api/...` → 官方 API，在 `../asc-devkit` 定位
     （不存在先 `git clone https://gitcode.com/cann/asc-devkit ../asc-devkit`）：
     声明头 `include/<...>/<api>.h`，实现头 `impl/adv_api/detail/.../<api>_common_impl.h`，
     确认 `__NPU_ARCH__ == 3510` 分支是实现目标。
3. 读 `{ApiCallName}::Generate`（`autofuse/v35/codegen/reg_api_call/` 下），确认生成代码的实际传参
   与调用形式（如是否 ReinterpretCast 输出、是否带 tmpBuffer、count 用什么对齐）。
   bench 的调用形式必须与 codegen 生成的形式一致。

### 1.2 分析输入与逻辑分支

产出一张分支表，每个分支必须能回溯到源码行：

- API 签名：dst/src/.../count 及模板参数；dtype 支持组合（看实现内 static_assert）。
- 循环结构：`oneRepElm`、`repeat_time` 计算方式；循环外指令（Duplicate 等常量初始化）与
  循环内指令（每 repeat 执行的 MicroAPI 序列）分开列出。
- 分支维度：dtype 组合、`if constexpr`（bool/非 bool 输出、fp16/fp32）、特殊路径。
- 标注哪些分支是 codegen 实际路径（如 IsInf 输出恒 `ReinterpretCast<bool>`），
  非实际路径的分支也测（成本一致），但公式应用范围要写清。

### 1.3 编写上板验证代码

在 `<api 目录>/tools/` 下新建三个文件。以 IsInf 为参考实例。

`<api>_perf_bench.asc` 结构（kernel + host 同文件）：

```text
kernel（__global__ __vector__，模板 <DstT, SrcT> 覆盖 codegen 实际路径分支）：
  AscendC::InitSocState()
  GlobalTensor x/y SetGlobalBuffer
  TPipe pipe + TQue<VECIN,1> inQueue + TQue<VECOUT,1> outQueue   <- 必须用 TPipe 结构
  pipe.InitBuffer(inQueue, 1, copyCount*sizeof(SrcT))
  pipe.InitBuffer(outQueue, 1, copyCount*sizeof(DstT))
  CopyIn: AllocTensor -> DataCopy(srcLocal, xGm, copyCount) -> EnQue -> DeQue
  dstLocal = outQueue.AllocTensor<DstT>()
  for loopCount 次: API(dstLocal, srcIn, count)   <- 与 codegen 相同的调用形式
  PipeBarrier<PIPE_V>                              <- 只等 Vector 流水，保证 CopyOut 前计算完成
  CopyOut: EnQue -> DeQue -> DataCopy(yGm, dstOut, copyCount) -> FreeTensor
```

**bench 不做内部计时**：性能数据完全依赖 npu 模式 msprof op_summary 采集
（见第二步）；kernel 内 API 调用总数必须恒等于 `loop_count`——任何额外调用
（warmup、预执行）都会使 op_summary 解析值按倍数偏大；冷启动开销
（icache miss、流水线排空）计入拟合截距 b，属测量口径。

host main（同文件下半部分）：

- combo 命名 `<dst>_<src>`（如 bool_float），switch 到模板实例 `RunBench<DstT, SrcT>`。
- `copyCount` 按 src、dst 分别向上对齐到 32B 元素数（DataCopy 硬约束）。
- 输入按位模式构造（uint16/uint32 bits），必须包含特殊值（±Inf 等 API 语义相关值）+ 普通值。
- golden 按位模式构造，输出 memcpy 级 memcmp 校验前 count 个元素。
- UB 预算校验（src+dst 合计 <= 200KB）。
- 启动 `<<<1, 0, stream>>>` 单 block，排除多 block 干扰。
- 输出两行 CSV：`VERIFY,{PASS|FAILED},combo,count` 和
  `RESULT,combo,count,repeat_elm,repeat_time,loop_count`（参数回显，无性能字段）。

`CMakeLists.txt`（照抄 asc-devkit examples 模板）：

```cmake
cmake_minimum_required(VERSION 3.16)
set(CMAKE_ASC_RUN_MODE "npu" CACHE STRING "Run mode: npu, cpu, sim")
set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")
find_package(ASC REQUIRED)
project(<api>_perf_bench LANGUAGES ASC CXX)
add_executable(<api>_perf_bench <api>_perf_bench.asc)
target_link_libraries(<api>_perf_bench PRIVATE tiling_api register platform m dl)
target_compile_options(<api>_perf_bench PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)
```

`README.md`：源码与分支依据表、测量口径、编译/运行方法、输出格式、拟合用法。

### 1.4 编译与验证（完成标准）

1. NPU 模式编译必须通过（本机交叉编译即可，无需设备）：

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
mkdir -p build && cd build && cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 .. && make -j8
```

2. CPU 仿真验证正确性（用另一个构建目录，避免产物混入 npu 版）：

```bash
mkdir -p build_cpu && cd build_cpu
cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 .. && make -j8
# 逐 combo 跑对齐点与尾块点（count=1、repeat_elm*k±1 等），必须全部 VERIFY,PASS
```

3. 清理构建目录或确认被 .gitignore 覆盖（全局已忽略 build/，其余目录名需处理）。

### 1.5 实测环境约束

- dav-3510 bench 二进制只能在 Ascend 950PR/950DT 系列上出有效周期数据。
- 设备不在 API 头文件 arch 支持列表时（如 910B3 对 is_inf.h），API 为空实现：
  运行必现 `VERIFY,FAILED` 且 `cycles=0`。此时只做编译 + CPU 仿真验证，
  周期实测标注"待目标设备"，不得用空实现的 cycles 数据拟合。
- 正确判定口径：`VERIFY,PASS`（npu 模式另需 msprof 解析值 > 0）才是有效测量点。

## 第二步：运行工具（编译、传参、运行、采集）

`<api 目录>/tools/run_bench.py`，标准库实现，职责：

1. **编译**：调用 cmake/make，`--run-mode {npu,cpu}` 决定 `CMAKE_ASC_RUN_MODE`，
   构建目录按模式区分（`build/`、`build_cpu/`），`--skip-build` 复用已有产物。
   要求环境已 `source ${ASCEND_HOME_PATH}/set_env.sh`（脚本检测 `ASCEND_HOME_PATH`）。
   npu 模式启动时经 npu-smi 自检设备环境：`npu-smi` 可用时打印设备信息
   （供确认是否为目标形态并作为归档环境证据）；命令不存在或执行失败时
   不阻断，默认按 `--arch`（dav-3510）继续。
2. **用例来源（cases.csv，第三步测试用例设计的落点）**：
   - 用例定义在 `<api 目录>/tools/cases.csv`，列：`combo,count,repeat,loop_count`，入参全部显式写出。
     `repeat` 为该用例的 repeat_time（`ceil(count/repeat_elm)`），写入是为了一眼看懂覆盖点。
   - 用例设计规则：每个分支（combo）约 8 个用例，取对齐点 `count = rt * repeat_elm`，
     rt 覆盖 `1,2,4,8,16,32,64,128`；`loop_count` 固定为 1（计时循环内单次 API 调用，
     直接测单次调用的完整周期）。尾块点等外推验证按需另建 csv 用 `--cases` 补测，不入正式用例集。
   - 每点固定重复执行 5 次取 `measured_cycles` 中位数抑制抖动（工具内常量，不进用例文件）。
   - 工具对 cases.csv 做校验：未知 combo、非正 count/repeat/loop_count、`repeat` 与
     `ceil(count/repeat_elm)` 不一致、空文件、缺列均带行号报错退出。
3. **运行采集**：逐用例执行 bench，解析 stdout 的 `VERIFY` / `RESULT` 行；
   VERIFY,FAILED 或进程失败记录为失败点并继续（不中断整批）。
   - **npu 模式必须开启 profiling**：每个用例经
     `msprof --parse=on --application="{exe} {combo} {count} {loop_count}"` 包裹执行。
     目录结构：`{profile-dir}/{run_id}/{combo}_{count}/PROF_{timestamp}_*`，
     其中 `run_id` 标识一次 run_bench.py 执行批次（`--run-id` 指定，默认
     `run_YYYYmmdd_HHMMSS`），多次运行（每 run 一个 msprof 进程）产生多个 PROF 文件，
     均落在当前批次目录下；`--output` 目录必须预创建。
   - **实测值来自 op_summary**：每用例测试完后（全部重复运行结束），在**本次 run_id** 的
     profiling 子目录递归查找 `op_summary_*.csv`（不解析历史批次，避免跨批次数据污染），
     取 `OP Type` 值包含**被测 API 名称**（工具内 `API_NAME` 常量，如 isinf，
     忽略大小写）的行，读 `Block Num`、`aiv_total_cycles`、`aiv_vec_ratio` 三列，计算
     `aiv_total_cycles / Block Num * aiv_vec_ratio` 作为本用例实测值，
     多次运行（每 run 一个 op_summary）取中位数后**刷新 samples.csv 的 `measured_cycles`**。
     npu 模式下 `measured_cycles` 以 msprof 解析值为准（INVALID 判定亦基于该值）。
   - 解析容错：`aiv_vec_ratio` 兼容 `0.85` 与 `85%` 两种格式；`OP Type` 按被测 API 名称
     子串匹配且忽略大小写（注意避免与其他 API 名互为子串，如 isnan/isfinite 需确认）；
     op_summary 缺 `OP Type` 及三列之一时告警跳过该文件；
     无任何 op_summary 产出（如非目标设备、kernel 未执行）时该用例实测值留空。
   - cpu 模式不经 msprof，直接执行 bench，**仅做正确性校验**（VERIFY），
     不采集性能数据，`measured_cycles` 留空。
   - **profiling 数据测试后必须保留**：任何验证、清理操作不得删除 `profiling/` 目录；
     `.gitignore` 中的 `profiling/` 仅表示不进 git，工作区数据是实测原始证据，
     归档时由 `results/<platform>/<run_id>/manifest.json` 索引关联；
     首次在目标设备实测时确认 op_summary 实际列名与 ratio 格式，必要时调整解析。
4. **输出（samples.csv，两阶段回填流程）**：`--output` CSV（默认 `samples.csv`），
   包含 cases.csv 中**所有**用例（逐行一致，失败点也占行），列：
   `combo,count,repeat,loop_count,model_predict_cycles,measured_cycles`。
   - 阶段一（本工具，实测）：npu 模式填写 `measured_cycles`（op_summary 解析值
     `aiv_total_cycles / Block Num * aiv_vec_ratio` 的中位数，拟合时由 fit_curve.py
     除以 loop_count 归一化为单次调用周期）；cpu 模式与失败/无效点留空
     （禁止写入 0 值脏数据）；`model_predict_cycles` 留空。
   - 阶段二（拟合 + perf 代码完成后，第四/五步）：由回填工具按最终公式
     `a * repeat + b` 计算 `model_predict_cycles` 并更新，用于模型与实测对齐归档。
   - **samples.csv 测试后必须保留**：它是第四步拟合的输入和阶段二回填的载体，
     任何验证、清理操作都不得删除；仅在重新实测时被工具整体覆盖。
     samples.csv 纳入 git 跟踪（用例实测结果随用例归档），`build*/` 构建产物不入库。
   - 结束打印汇总（总点数/实测点数/失败点数），存在失败/无效点时退出码非 0。
5. **实测判定**：npu 模式下 msprof 解析值缺失或 <= 0 视为无效点（非目标设备上
   kernel 未执行时 op_summary 无产出），`measured_cycles` 留空并计入失败；
   cpu 模式仅验证正确性（loop_count 自动收敛到不超过 2，
   `VERIFY,PASS` 即 PASS，不采集性能数据）。

用法示例：

```bash
# cpu 仿真跑通全链路（本机可测，只看 VERIFY；mini 用例另建 csv）
python3 run_bench.py --run-mode cpu --cases mini_cases.csv

# 目标设备实测全量用例（读 tools/cases.csv，npu 模式自动经 msprof 开启 profiling）
python3 run_bench.py --output samples.csv
```

完成标准（以 isinf 为参考，均已在无目标设备环境下验证）：
- cpu 模式：编译通过，全矩阵 `VERIFY,PASS`，samples.csv 正确产出
  （`measured_cycles` 留空、`model_predict_cycles` 留空），退出码 0。
- npu 模式：交叉编译通过；每个用例经 `msprof --application` 调起且 profiling 数据在
  `--profile-dir` 按用例落盘；无目标设备时（如 910B3）运行如实上报 FAILED/INVALID 且退出码非 0，
  失败点 `measured_cycles` 留空。
- 工具目录加 `.gitignore`（仅 `build*/`、`profiling/`、`__pycache__/`），构建产物不入库；
  samples.csv 纳入 git 跟踪；profiling 原始数据不入库但测试后必须保留在工作区。

## 第三步：测试用例设计与实测

用例设计落点为 `tools/cases.csv`（每分支约 8 例、loop_count=1），由 run_bench.py 读取执行；
设计规则、校验与实测判定见「第二步」中「用例来源（cases.csv，第三步测试用例设计的落点）」、
「运行采集」与「实测判定」小节。周期实测需 dav-3510 目标设备，非目标设备只做编译与
CPU 仿真正确性验证。

## 第四步：拟合各分支公式（fit_curve.py）

`<api 目录>/tools/fit_curve.py`，标准库实现，职责：

1. **读取**：`--result`（默认 `tools/samples.csv`），取 `loop_count, repeat, measured_cycles`
   三列按 `combo` 分组；`measured_cycles` 为空或非正的行跳过（0 值占位数据禁止参与拟合），
   校验缺列与非法 `loop_count`。
2. **拟合**（`fit_linear(samples)`，样本为多组 `(loop_count, repeat, measured_cycles)`）：
   - 先归一化 `y = measured_cycles / loop_count`（单次 API 调用周期）；
   - 再对 `y = a * repeat + b` 做最小二乘，闭式解求 a、b 最优值；
   - 输出 R2、逐点预测值与残差；repeat 全相同（分母为 0）、有效样本不足 2 个时拒绝拟合。
3. **输出**：`--report`（默认 `tools/report.md`），**总是生成并保留**（数据无效也落盘说明状态，
   不得删除）：
   - 按 combo（源码分支）分组，**每种分支独立一节、各有一组拟合结果**；
   - 汇总表列出各 combo 的有效/无效样本数、a、b、公式、R2、状态；
   - 有效样本不足或 repeat 全相同的分支在报告中标注原因（如 CPU 仿真占位数据需目标设备实测），
     不中断其他分支拟合；
   - 全部分支未拟合或存在未拟合分支时退出码非 0（提示流程未完成）。

用法示例：

```bash
python3 fit_curve.py                          # 读 tools/samples.csv，写 tools/report.md
python3 fit_curve.py --result r.csv --report report.md
```

完成标准：
- 拟合正确性自测：用 `y = a0 * repeat + b0` 加噪声的模拟数据与 `loop_count > 1` 的整倍数据
  验证 a、b 恢复（归一化必须精确还原 a0、b0）。
- 正式 samples.csv 为 0 值占位数据（cpu 仿真）时，report.md 仍生成并保留，每种分支一节
  标注"未拟合"及原因，退出码非 0；拟合成功的 report 只能由目标设备实测数据重新生成。
- report.md 中的公式即第五步 perf 代码 `a * repeat + b` 的参数来源。

## 第五步：perf 计算代码编写与归档

输入：`tools/report.md` 各分支拟合出的 a、b + 第一步的源码分支梳理。
产出：API 的 perf 计算代码（实测简化公式直落）、samples.csv 回填、design.md 与 results 归档。

### 5.1 定位 perf 代码链路

按 af-perf-modeler 的链路定位（以 IsInf 为例，其他 API 类推）：

- 注册入口：`autofuse/v35/att/api_perf_register/ascir_api_perf_v2.cpp` 的 `{Api}Api`
  （如 `IsInfApi`）：`SetNodeDetail(input_shapes, output_shapes, node_info)` 组装
  `NodeDetail`，透传节点参数（如 `node_info.unary_bitwidth_change_node_params =
  node.unary_bitwidth_change_node_params`），调用 `ascendcperf_v2::{Api}Perf`。
- perf 实现：`autofuse/v35/att/api_perf_register/ascendc_regbase_perf.cpp` 的 `{Api}Perf`
  （如 `IsInfPerf`）。
- 参数来源（perf 函数内可得的量）：
  - `GetUnaryBitWidthChangeCallCount(node_info, cal_count)` 等 helper 取 codegen 透传的
    `cal_count` 与 `call_count = Π outer_repeats`（参数无效时按输入总元素数、call_count=1）；
  - `CalculateRepeatParams(input_dtype, cal_count)` 算 `repeat_time =
    Ceil(cal_count / repeat_elm)`（`kRptEleMap`：fp32=64、fp16=128，与 bench 口径一致）。
- 输出落点：`perf.pipe_res[PipeType::AIV_VEC] = res;` 且 `res.Simplify();`。

### 5.2 实测公式落地（替换理论推算）

实测简化公式**必须写在 perf 实现**：`autofuse/v35/att/api_perf_register/ascendc_regbase_perf.cpp`
的 `{Api}Perf`（如 `IsInfPerf`）中，改为直接实现实测公式
`res = (a * repeat_time + b) * call_count`（`res.Simplify()` 后写
`perf.pipe_res[PipeType::AIV_VEC]`），**不保留** `AddVfInstructPerf` 理论序列
（理论依据转移到 design.md 对齐分析，见 5.4）。

硬性口径约束：

- **b 已含全部固定开销**（head cost、max_latency、单次 for 循环与收尾 barrier），
  禁止再叠加 `VfPerfUtils::GetVFHeadCost()` 或任何 `AddVfInstructPerf` 固定项——
  重复计费是本步骤最常见错误。
- `repeat_time`、`call_count` 的计算必须与 bench/用例口径一致（同一 `repeat_elm`、
  同一 cal_count 来源），否则公式对不上实测。
- 常数命名 `k{Api}{Branch}A/B`，注释必须指向 `tools/report.md` 并注明数据环境
  （设备形态、CANN 版本、拟合日期），保证可追溯。
- 分支维度与第一步梳理一致：IsInf 的 codegen 实际路径输出恒 bool，按 input_dtype 分支；
  若 API 存在多分支组合，每个源码分支一组常数，未实测分支不得使用其他分支常数冒充
  （宁可用理论值并在注释/design.md 标注，等补测后替换）。
- 修改前先读 `autofuse/v35/att/api_perf_register/` 下现有实现与头文件声明，同步更新
  `.h` 中函数签名/常量声明位置，遵循文件内既有风格；改动后对相关文件执行
  `clang-format -i`。

### 5.3 回填 model_predict_cycles

perf 代码定稿后，将 `tools/samples.csv` 的 `model_predict_cycles` 列按**同一组 a、b**
（与 perf 代码常数完全一致）计算 `a * repeat + b` 回填，完成两阶段流程闭环：
- 回填值与 perf 代码必须同源（同一 report.md 拟合值），禁止手工微调；
- 用一次性 python 脚本（内联或临时文件）读写 csv 完成，不引入常驻工具；
- 回填后与 `measured_cycles` 的相对误差即模型精度证据，供 design.md 引用。

### 5.4 验证与归档

1. **UT**：perf 计算变更属 ATT 行为变更，运行/补充 perf 相关 UT
   （`autofuse/tests/ut/att/testcase/gen_model_info/api_perf_register/` 与 v35 对应目录，
   用 af-test-developer 技能执行 `sh build.sh -u --module=autofuse_framework -j 8`）；
   新增用例覆盖：各分支 dtype 的 repeat_time 展开、call_count>1、参数无效回退路径。
2. **design.md 更新**：公式章节改为实测拟合公式（引用 `tools/report.md`），
   保留理论推算（MicroAPI 计数）作为对照，说明 a、b 与理论值的差异及来源
   （如 b 含测量口径开销、表项 latency 为标称值等）。
 3. **归档**：将 `tools` 中的输出结果汇总，按 `autofuse/att/modeling/README.md`
    的模型归档约定生成对应目录和文件（`<model>/design.md`、`<model>/cases.csv`、
    `<model>/results/<platform>/<run_id>/{manifest.json,samples.csv,report.md}`）：
    - `cases.csv`、`samples.csv` **直接复制** `tools/` 下对应文件
      （`tools/cases.csv -> <model>/cases.csv`，
      `tools/samples.csv -> <model>/results/<platform>/<run_id>/samples.csv`），
      不手工重写内容；
    - `report.md` 直接读取 `tools/report.md` 中的拟合结果**总结输出**：
      提取各分支的 a、b、R2、残差等拟合结论，整理为归档格式
      （误差分析、结论及限制）；不重新拟合，不得引入 `tools/report.md`
      之外的数据（mock/占位数据的拟合结果禁止归档）；
    - `manifest.json` 记录环境（npu-smi 设备信息）、CANN 版本、执行命令及
      原始证据索引（关联 `tools/profiling/{run_id}/`，原始数据不入库）；
      `<run_id>` 与 profiling 批次一致，`<platform>` 为设备形态（如 dav-3510）。
 4. **跨特性检查**：perf 公式变更影响 ATT 整体估算，读
    `docs/guidelines/cross_feature_check.md` 确认对 SuperKernel、E2E 测试无连带影响。

### 完成标准

- [ ] perf 代码使用 report.md 的 a、b，无重复叠加 head cost/固定项。
- [ ] 分支维度、repeat_time、call_count 口径与 bench/cases 一致。
- [ ] samples.csv 的 `model_predict_cycles` 已按同源 a、b 回填。
- [ ] perf 相关 UT 通过，新增分支/call_count/无效参数用例。
- [ ] design.md、results/ 归档更新，公式来源可追溯到 report.md 与实测环境。
- [ ] 相关 C++ 文件已 `clang-format -i`。

## 已知约束与坑（实测沉淀）

| 现象 | 原因与规避 |
| --- | --- |
| `AscendC::half` 编译报 no member | `half` 是全局类型，直接写 `half` |
| LocalMemAllocator 分配 UB 后调不带 tmpBuffer 的 adv_api，CPU/NPU 均崩 | 该形式内部 `PopStackBuffer` 依赖 TPipe 上下文（`GetTPipePtr()` 为 null 直接断言），kernel 必须用 `TPipe + TQue` 结构，与 codegen 生成的 kernel 一致 |
| DataCopy 崩溃或校验错位 | DataCopy 要求 32B 对齐，copyCount 需按 src、dst 元素宽度分别向上对齐；padding 区域不校验 |
| 910B 等设备上 VERIFY,FAILED 且 cycles=0 | 设备不在 API arch 支持列表，API 为空实现；换 dav-3510 设备，空实现数据禁止用于拟合 |
| 拟合截距 b 偏大 | 无 warmup 口径下 b 含冷启动（icache miss、流水线排空）及 for 循环/barrier 开销，属测量口径的一部分，README 中注明 |
| op_summary 解析值按倍数偏大（如 1 次调用测出 ~9 倍） | kernel 内存在计时循环之外的额外 API 调用（warmup/预执行），op_summary 统计整个 kernel 周期；禁止 warmup，调用总数必须恒等于 loop_count |
| perf 代码叠加 GetVFHeadCost 后结果偏大 | 实测拟合的 b 已含 head/固定 latency/循环开销，公式 `(a*repeat+b)*call_count` 直落，禁止再叠加固定项 |
| perf 结果与 samples.csv 对不上 | 检查三处口径是否一致：repeat_elm 映射（kRptEleMap vs bench COMBOS）、cal_count 来源（节点参数 vs 用例 count）、call_count 乘法位置 |
