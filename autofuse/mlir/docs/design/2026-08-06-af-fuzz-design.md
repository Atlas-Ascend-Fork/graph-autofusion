# af-fuzz：Torch 自动用例生成与差分验证工具设计

> 状态：草稿
> 日期：2026-08-06
> 适用目录：`autofuse/mlir/test/fuzz/`

# 简介

## 目的

本文设计 `af-fuzz`，一个面向 Torch 前端的自动用例生成和差分验证工具。它通过 `torch.compile` 和
TorchInductor 进入 Autofuse，用同一用例比较 CPU 参考、Legacy Autofuse 和 MLIR Autofuse 的行为。

工具需要发现并区分编译崩溃、超时、结果不一致、错误 fallback、没有实际进入 Autofuse、覆盖缺口和性能退化。
每个失败都保存 seed、环境信息和重放命令，便于稳定复现。

目标读者是 Autofuse optimize/codegen、MLIR 迁移、TorchInductor NPU 接入和测试基础设施的维护者。本文先定义
可以落地的 MVP，再说明覆盖引导、reducer 和性能 corpus 等后续能力。

## 范围

本设计包含：

- 在 `mlir/test/fuzz/` 新增独立命令行工具，不挂入默认 `check-autofuse-mlir`；
- 约束驱动、类型合法、seed 可重放的 Torch/FX 图和输入生成；
- PyTorch eager CPU、TorchInductor CPU、Legacy Autofuse NPU 和未来 MLIR Autofuse NPU 的统一 backend 契约；
- dtype/op 相关的差分精度规则、异常分类、timeout 和 fallback/融合命中检查；
- 失败证据、覆盖数据、最小复现和 regression corpus 的保存规则；
- NNSmith 可选接入方式及其与本地 MVP generator 的边界；
- nightly fuzz 与独立性能 corpus 的后续执行方式。

本设计不包含：

- 把 AFIR 或 `ScheduledResultInfo` 作为 fuzz 共同输入；
- 在本阶段实现完整约束求解器、coverage-guided 搜索或通用图 reducer；
- 在 MLIR 正式 codegen 尚未接入时伪造 `mlir-npu` 成功结果；
- 修改 TorchInductor、torch_npu 或 inductor-npu-ext 的核心融合逻辑；
- 将随机 corpus、编译 kernel、profiling、日志或输入 tensor 数据提交 Git；
- 将 fuzz/nightly/performance 加入日常增量测试目标；
- CPU 与 NPU 的横向性能比较。

# 总体概述

## 软件概述

### 项目介绍

当前 Torch 路径由外部 `inductor_npu_ext` 注册 Inductor NPU scheduling/codegen。它生成 AscIR 后调用本仓库的
`autofuse.pyautofuse.Autofuser`：`Autofuser.schedule()` 进入 legacy `optimize::Optimizer`，
`Autofuser.codegen()` 最终调用 `Codegen::GenerateForInductor()`。

`ENABLE_AUTOFUSE_MLIR` 当前只控制构建。即使打开，实际执行仍走 legacy Codegen；设置
统一 dump 前缀（`ascir::utils::GetDumpFilePrefix()`） 只会额外生成 AFIR 调试文件。`AF_MLIR_CODEGEN=off|on|compare` 和正式 MLIR lowering
尚未接入产品路径。

因此，MVP 可以执行 Legacy NPU，但不能执行 MLIR NPU。用户请求 `mlir-npu` 时，工具必须返回 `unavailable`，
不能静默改走 Legacy。正式路由接入后再启用该 backend。

### 产品环境介绍

- eager CPU 和 Inductor CPU 依赖与目标 NPU 环境匹配的 PyTorch 版本；
- Legacy NPU 依赖 torch_npu、inductor-npu-ext、本仓构建安装的 `autofuse.pyautofuse`、CANN Toolkit 和 NPU；
- MLIR NPU 还依赖 `ENABLE_AUTOFUSE_MLIR=ON` 构建产物以及尚待实现的运行时路由；
- NNSmith 仅作为固定版本的可选测试依赖，由测试环境预装，不复制源码、不在构建时联网下载；
- `af-fuzz` 不进入 run 包、wheel 和产品动态库，默认不影响交付内容。

## 软件功能

工具的主流程是“生成一次，隔离执行，统一比较”：

1. 根据 seed、op 集、dtype、rank、shape 上限和拓扑约束生成合法 Torch/FX 用例。
2. 从同一份 CPU 输入派生各 backend 输入，分别执行 eager CPU、Inductor CPU、Legacy NPU 和可用时的 MLIR NPU。
3. 以 eager CPU 为语义参考，按 dtype 和 Op 规则比较输出；Inductor CPU 作为独立编译路径单独报告。
4. 对生成、编译、执行、超时、差分、Autofuse 命中和 fallback 分别分类。
5. 保存失败证据和重放信息。后续再增加图简化、覆盖引导和稳定 regression corpus。
6. 性能只比较同一 NPU 环境中的 Legacy 和 MLIR，并把编译时间与 kernel 执行时间分开。

## 设计约束

- 共同输入固定为生成的 Torch/FX 用例和同一组逻辑输入；
- 随机源必须由显式 seed 控制，遍历和输出顺序必须稳定；
- MVP 默认使用少量规则的本地 generator，NNSmith 通过 adapter 可选接入；
- 每个 backend 在独立进程执行，timeout 后终止该 worker，避免污染后续用例；
- NPU 输入由同一份 CPU tensor 拷贝产生，不分别随机生成；
- eager CPU 是主 Oracle；Inductor CPU 失败单独报告，不能自动改用它覆盖 eager 结果；
- 浮点比较必须定义 dtype/op 容差、NaN/Inf 和 signed-zero 规则；整数和 bool 默认精确比较；
- `legacy-npu` 或 `mlir-npu` 执行成功但未生成融合 kernel 时，不得记为 Autofuse 验证通过；
- 性能只比较同设备、同版本、同输入、同配置下的 Legacy NPU 与 MLIR NPU；
- 默认运行不得产生高频 INFO 日志；详细图、kernel 和 profiling 产物仅在失败或显式 debug 下保留；
- 所有构建命令限制为 `-j 8`；fuzz/nightly/performance 不加入 `check-autofuse-mlir`；
- 临时产物写入用户指定目录或系统临时目录，不写入源码树；
- 不预创建无实现的目录和空抽象。

## 假设和依赖关系

- 假设目标 NPU 环境提供兼容版本的 PyTorch、torch_npu 和 inductor-npu-ext；具体版本矩阵待 CI/NPU 环境确认；
- 假设 MVP 先覆盖静态 shape、前向推理和无随机语义算子；训练、反向和动态 shape 在后续阶段加入；
- 假设 inductor-npu-ext 的 debug 目录中 `autofused_*` 与 `fuse_summary.csv` 可作为当前融合命中证据；
- `inductor_npu_ext.lowering.common._summary` 可用于开发期补充观测，但它是私有接口，不作为长期兼容契约；
- 待确认 nightly 的 NPU 队列、设备隔离、单用例/单批次 timeout 和性能噪声阈值；
- 待确认正式 MLIR codegen 路由的环境变量/API、失败返回和 compare 模式输出格式；
- 待确认 inductor-npu-ext 是否接受新增公开、结构化、可清零的编译统计接口。

# 需求分析与设计

## 整体介绍

主数据流如下：

```text
seed + generation config
    -> CaseSpec（稳定、可序列化）
    -> Torch FX GraphModule + CPU inputs
        -> eager-cpu       -> reference outputs
        -> inductor-cpu    -> independent compiled outputs
        -> legacy-npu      -> NPU outputs + fusion/fallback evidence
        -> mlir-npu        -> NPU outputs + fusion/fallback evidence（路由可用后）
    -> correctness oracle + outcome classifier
    -> result.json / failure reproducer / coverage summary
```

`CaseSpec` 与执行层分离，各模块只承担一种职责：

- generator 生成合法图、输入描述和 feature 标签；
- backend 在隔离进程中执行用例，返回统一的 `BackendResult`；
- observer 把外部 backend 的调试产物转换为 `FusionEvidence`；
- oracle 只比较结构化结果，不直接解析 backend 日志。

MVP 只实现有实际使用者的文件：CLI、规则生成器、backend runner、correctness oracle 和对应 pytest。覆盖反馈、
reducer、performance runner 与 corpus 管理在进入下一阶段时再新增。

## 功能需求

### 功能需求 1：确定性合法用例生成

1. 介绍

   用显式 seed 和白名单规则生成静态 Torch FX 图。生成阶段就保证 dtype、rank、broadcast 和各 Op 参数合法，
   不把明显非法用例交给 backend。

2. 输入

   `seed`、节点数范围、允许的 op/dtype/rank、每个 tensor 最大元素数和 shape 边界。非法范围、空 op 集、
   非正节点数和元素数溢出必须在 materialize 前拒绝。

3. 处理

   MVP 使用本地规则生成 chain，至少包含两个可融合计算节点；shape 先生成再按 broadcast 规则派生输入 shape；
   使用局部 `random.Random(seed)`，Torch 输入使用独立 `torch.Generator().manual_seed(seed)`。后续 NNSmith adapter
   调用固定版本的 `model_gen(..., seed=..., rank_choices=..., dtype_choices=...)`，并把结果转换为相同用例契约。

4. 输出

   稳定可序列化的 `CaseSpec`、FX `GraphModule`、CPU inputs、图特征和一条 `--seed` 重放命令。

### 功能需求 2：多 backend 隔离执行

1. 介绍

   对同一用例执行 CPU 参考和 Autofuse NPU 路径。每个 backend 使用独立进程，避免崩溃、编译缓存和环境变量
   污染其他结果。

2. 输入

   `CaseSpec`、逻辑 inputs、backend 名称、timeout、设备号和 backend 配置。

3. 处理

   worker 内重新 materialize 用例；在编译前设置 debug/cache 环境；记录 compile 和 execution 阶段；NPU 路径
   将 CPU 输入拷贝到同一目标 device 并同步后再取回输出。父进程按 timeout 终止 worker，并把 signal/exit code
   归类为 crash 或 timeout。不存在依赖、设备或 MLIR 路由时返回 `unavailable`，不伪装为 skip/pass。

4. 输出

   `BackendResult`，包含状态、失败阶段、输出、耗时、异常摘要、版本、device 和 `FusionEvidence`。

### 功能需求 3：正确性与异常差分

1. 介绍

   eager CPU 是语义参考。其他 backend 分别与它比较；Inductor CPU 的编译稳定性仍作为独立结果保留。

2. 输入

   backend 输出树、输出 dtype/op 标签和 tolerance profile。

3. 处理

   先校验输出树、shape 和 dtype；整数/bool 精确比较；浮点默认按下表处理。NaN 必须出现在相同位置，Inf 必须
   同位置且同符号。MVP 忽略 `+0/-0` 差异并单独计数；对明确要求符号零的 op profile 才升级为失败。

   | dtype | MVP `rtol` | MVP `atol` | 说明 |
   |---|---:|---:|---|
   | `float32` | `1e-4` | `1e-5` | elementwise/broadcast 基线 |
   | `float16` | `1e-3` | `1e-3` | elementwise/broadcast 基线 |
   | `bfloat16` | `1.6e-2` | `1.6e-2` | 后续 NPU 环境确认后启用 |
   | integer/bool | `0` | `0` | 精确比较 |

   reduce、transcendental 和混合精度算子必须使用独立 profile，未定义 profile 的组合不得进入 fuzz 白名单。

4. 输出

   每个 backend 的 pass/fail、最大绝对/相对误差、首个差异位置及 `crash/timeout/compile-error/runtime-error/
   mismatch/unavailable` 分类。

### 功能需求 4：Autofuse 命中与 fallback 观测

1. 介绍

   模型执行成功不等于 Autofuse 已被验证。工具还要确认本次执行确实生成了 Autofuse 融合 kernel。

2. 输入

   backend debug 目录、融合 summary、lower/fallback 统计和生成 kernel 元数据。

3. 处理

   MVP 从隔离工作目录统计 `autofused_*` kernel 目录并读取存在的 `fuse_summary.csv`；同时可在兼容版本上读取
   inductor-npu-ext 的 LowerSummary 快照。结果必须记录证据来源。没有融合 kernel 时状态为 `not-exercised`；
   有 fallback 时记录 op、原因和数量。后续应在 inductor-npu-ext 增加公开结构化接口，替换私有对象/目录约定。

4. 输出

   `FusionEvidence`：`autofuse_entered`、`fused_kernel_count`、`fused_subgraph_sizes`、`lowered_ops`、
   `fallback_ops`、`fallback_reasons` 和 `evidence_source`。

### 功能需求 5：失败保存与重放

1. 介绍

   每个失败都要能用 seed 和环境信息重放，但随机输入、kernel 和大体积日志不能进入 Git。

2. 输入

   `CaseSpec`、backend 结果、版本、CLI 参数、FX 图文本和输入元数据。

3. 处理

   原始失败写到显式 `--artifact-dir` 或系统临时目录；使用原子目录命名避免并发覆盖；仅保存 shape/dtype/
   stride/value-domain 元数据和必要的小型复现数据。保存前校验路径、单文件和总大小上限。清理仅作用于本次创建
   的明确目录。reducer 后续按删 op、简化 shape/dtype 的顺序工作，每一步都重新验证相同 failure signature。

4. 输出

   `result.json`、`case.json`、`fx_graph.py`/`fx_graph.txt`、版本清单、最小重放命令和可选的小型 input 文件。
   只有人工确认稳定且最小化的复现可以进入 `corpus/correctness/` 或 `corpus/performance/`。

### 功能需求 6：覆盖与性能扩展

1. 介绍

   后续按覆盖价值保留用例，并在固定 NPU corpus 上比较 Legacy/MLIR 性能。

2. 输入

   图结构、FusionEvidence、Autofuse/MLIR pass 命中、kernel profile 和稳定 corpus。

3. 处理

   覆盖维度包括 op、pair/triple、chain/diamond/fan-in/fan-out、多输出、dtype/rank/shape/layout、tile/UB/尾块/
   多核边界、Schedule Case、AFIR pass/pattern 和 placement/queue/buffer 策略。性能分别预热并重复测量，compile
   time 与 kernel time 分开；只在同设备/版本/输入/配置下比较 Legacy/MLIR，报告 median、P90 和几何平均。

4. 输出

   coverage summary、有价值 seed 列表、稳定性能报告和阈值退化列表。

## 非功能需求

### 可维护性

- backend、generator 和 observer 使用小型显式协议，避免继承复杂框架；
- op 合法性和 tolerance 是数据表，不散落在 runner 分支中；
- 外部私有 API 只在单一 adapter 内使用，并带版本检查；
- MVP 不创建 reducer/coverage/performance 空目录；
- NNSmith adapter 直接组合公开 Python API，不复制或 patch 上游源码。

### 可测试性

- 不需要 Torch/NPU 的 UT 覆盖 seed 稳定性、规则合法性、配置校验、结果分类和 artifact 限额；
- 有 PyTorch 的 CPU pytest 覆盖 FX materialize、eager/Inductor 差分和失败重放；
- NPU ST 在独立环境覆盖真实融合命中、fallback 和超时隔离；
- MLIR NPU E2E 在生产路由接入后再启用，不用 AFIR dump 代替执行验证。

### 可移植性

- CLI 和纯逻辑层支持项目 Python 基线；Torch/torch_npu 版本由环境矩阵限定；
- CPU 路径不得 import torch_npu；NPU 依赖使用惰性 import；
- 不硬编码芯片型号，设备能力由 torch_npu/CANN 环境查询；
- 进程隔离优先使用 Python `spawn` 或显式子进程，避免依赖 Linux-only `fork` 状态。

### 可靠性

- 每个 backend 单独进程，父进程负责 timeout、退出码和产物收集；
- 子进程不跨进程传递 NPU tensor，只传可序列化 CaseSpec/结果摘要；
- 文件写入使用本次工作目录，异常后关闭句柄并保留失败证据；
- 输入元素数、rank、维度和 artifact 大小均设置上限并在使用前校验；
- 重放时检查 schema/version，不接受未知字段静默改变语义。

### 特性交叉影响

| 场景 | 适用性 | 分析说明 |
|---|---|---|
| SuperKernel Python 接口 | 不适用 | 工具位于 Autofuse MLIR 测试目录，不修改 SuperKernel Python 包、选项、wheel 或 pytest |
| SuperKernel C++/AOT 接口 | 不适用 | 不修改 `libascendsk.so`、AOT、RDV 或其 ABI/API |
| Autofuse 图优化 | 适用 | Legacy NPU 会覆盖真实 ASCIR/optimize；工具本身不改图优化逻辑，失败按 backend 分类 |
| Autofuse Codegen/Backend | 适用 | 通过 `GenerateForInductor` 验证 codegen、tiling、kernel 编译和执行；MLIR 路由完成后增加双路验证 |
| AscendC API / Runtime 交互 | 适用 | 测试会间接执行既有 NPU backend；不新增直接 `rt*`/`aclrt*`/AscendC 调用，worker 退出前同步 |
| Python/C++ 混合绑定 | 适用 | Legacy 路径经 `autofuse.pyautofuse`；MVP 不改绑定，后续结构化统计接口若落在绑定层需独立评审和测试 |
| 构建与打包 | 适用 | 新增独立测试脚本/可选依赖，不进默认 CMake、run 包和 wheel；禁止构建时联网下载 NNSmith |
| 测试与覆盖率 | 适用 | 新增纯 Python UT、CPU pytest、NPU ST/nightly 和未来 MLIR NPU E2E |
| 性能与日志 | 适用 | 默认短测不 profiling；debug/性能按需启用并限制产物，性能只做 NPU 同类比较 |
| 兼容性 | 适用 | 不改既有 API/ABI/默认目标；外部私有观测仅封装为临时 adapter，不承诺兼容 |

## 性能

### 编译时长

`af-fuzz` 不加入默认构建和 `check-autofuse-mlir`，因此日常增量构建无新增时长。CPU/NPU 每 seed 触发编译，
nightly 可按 worker 数受控并行；NPU 设备默认单 worker，避免编译内存和设备竞争。缓存命中会掩盖编译问题，
稳定性 fuzz 默认每 worker 使用独立 cache；性能 corpus 明确区分冷编译和热执行。

### 执行性能

工具不修改产品运行路径。debug 统计和产物落盘只在 fuzz worker 中开启。性能模式先预热，再同步并重复采样；
compile time 不混入 kernel time。不得用 CPU 时间作为 NPU 性能结论。

### 内存和产物大小

每个 tensor 元素数、单 case 输入总字节数、单 artifact 和总 artifact 均设上限。随机 corpus 默认只保存 seed 和
元数据；kernel/debug/profiling 保存到忽略目录并按保留策略清理。工具不进入产品包，不增加动态库常驻内存。

## 接口设计

### 新增/修改接口描述

MVP 对外仅新增测试 CLI：

```text
python3 mlir/test/fuzz/af_fuzz.py \
  --seed <uint64> \
  --backends eager-cpu,inductor-cpu[,legacy-npu,mlir-npu] \
  --timeout-seconds <positive-int> \
  --artifact-dir <path>
```

退出码：`0` 表示所有请求且可用的 backend 达到验收条件；`1` 表示发现失败或 NPU 未实际进入 Autofuse；`2`
表示参数/环境错误。请求 `mlir-npu` 但路由不存在时返回 `unavailable` 并以环境错误退出，禁止回退到 Legacy。

内部协议：

```text
Generator.generate(config, seed) -> CaseSpec
Backend.run(case_spec, inputs, run_config) -> BackendResult
Observer.collect(work_dir, backend) -> FusionEvidence
Oracle.compare(reference, candidate, tolerance_profile) -> ComparisonResult
```

MVP 不修改既有 C/C++/Python 产品接口。未来若给 inductor-npu-ext 或 pyautofuse 增加结构化统计接口，单独设计、
评审和测试，不在本设计中默认授权。

### 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|---|---|---|---|
| 接口说明 | 是否需要接口评审 | 否（MVP） | 仅测试 CLI；未来产品/外部统计 API 需单独评审 |
| 接口说明 | 是否需要补充文档 | 是 | `mlir/test/fuzz/README.md` 说明环境、命令、结果和重放 |
| 接口兼容 | 行为是否兼容 | 是 | 不改变既有路径，新增 CLI 参数从首版固定语义 |
| 接口兼容 | ABI/API 是否兼容 | 是 | MVP 不修改产品 ABI/API |
| 接口约束 | 约束不满足时是否清晰报错 | 是 | 参数、依赖、设备和 MLIR 路由分别报告 unavailable/error |
| 接口测试 | 是否需要独立接口用例 | 是 | pytest 覆盖参数校验、退出码和 result schema |

## 软件设计

### 关键数据结构

- `CaseSpec`：schema version、seed、输入规格、有序 op 列表、输出引用、feature 标签；不可包含 live tensor；
- `TensorSpec`：dtype、shape、stride/layout、value domain；所有维度为受限正整数；
- `BackendResult`：backend、status、stage、outputs/metadata、compile/run time、error、version、device、evidence；
- `FusionEvidence`：Autofuse 进入、融合 kernel/子图数量、lower/fallback 统计及证据来源；
- `ComparisonResult`：结构/dtype/shape 检查、容差、误差、NaN/Inf/signed-zero 统计和 failure signature。

对象均由单次 case/worker 持有；文件产物由父进程的 case work directory 持有和清理。NPU tensor 只存在 worker。

### 关键技术/算法

- 本地 generator：按拓扑顺序选择规则，用已知 tensor 类型推导候选 op，只有满足约束才插入；复杂度为
  `O(node_count * candidate_count)`，MVP 节点数很小；
- NNSmith adapter：复用 Z3 约束生成并限制 opset/dtype/rank/元素数；若固定版本接口不匹配立即报错；
- 差分：递归展平相同输出树，逐 tensor 比较，复杂度为输出元素数线性；
- failure signature：backend + stage + exception type/首行或 mismatch 类别，用于 reducer 保持失败语义；
- 覆盖 key：使用稳定的 op 名、拓扑序、dtype/rank/边界 bucket，不使用对象地址或无序容器顺序。

### 流程设计

正常流程：解析配置，生成并校验 `CaseSpec`，运行 eager CPU 建立参考结果，再执行其他 backend，收集融合证据，
完成差分并写入摘要。成功用例按保留策略清理产物，失败用例保留重放所需的最小证据。

异常流程：非法生成配置在执行前拒绝；worker 超时或崩溃由父进程分类；NPU 执行成功但没有融合证据时记为
`not-exercised`；差分失败时保留 case 目录；MLIR 路由未实现时返回 `unavailable`。

回滚流程：MVP 仅新增独立目录和文档，不改默认入口；删除独立目录即可回滚，不影响产品构建和执行。

### 对子模块的修改

| 目录/模块 | 修改 |
|---|---|
| `mlir/test/fuzz/` | 新增 CLI、生成/执行/比较最小实现、README 和 pytest |
| `mlir/docs/design/` | 新增本设计文档 |
| `mlir/docs/README.md` | 增加设计索引 |
| `mlir/test/CMakeLists.txt` | MVP 不修改；后续仅新增独立显式 target，禁止挂默认 check |
| `compiler/py_module/` | MVP 不修改；未来公开观测接口另案处理 |
| `inductor-npu-ext` | 当前不修改；建议后续提供结构化统计 API |

### 错误处理

#### 系统错误

依赖缺失、NPU 不可见、编译失败、kernel load/launch 失败、文件失败和 worker timeout 均转换为带 stage 的结果；
不得吞掉原始异常类型和首段 traceback。临时目录创建失败立即停止，清理只处理本次明确拥有的目录。

#### 接口错误

CLI 对未知 backend、非法 seed/timeout/shape 上限返回退出码 `2`。backend unavailable 与 case failure 分开显示。
JSON schema 带版本；未知版本拒绝读取。不得因 MLIR 不可用而静默运行 Legacy。

## 安全检查

- 不硬编码密钥、账号、公网 IP、域名、芯片型号或网络名称；依赖由环境预装和版本清单声明；
- seed、节点数、rank、shape、元素数、输入总字节数和 artifact 大小使用前做范围及乘法溢出检查；
- 不直接调用 `rt*`/`aclrt*`/AscendC API；NPU 生命周期由 PyTorch 管理，计时/取回输出前显式同步；
- worker、文件和临时目录所有权明确，异常分支关闭/终止并保留必要证据；
- 不修改产品图；生成图按稳定规则构造，序列化和覆盖 key 不依赖无序遍历或对象地址；
- debug 高频日志默认关闭，仅 fuzz worker 显式开启并受 artifact 限额约束；
- 不修改产品 ABI/API；未来 Python/C++ 统计绑定必须单独检查引用计数、GIL 和异常转换；
- 随机产物、kernel、日志、coverage 和 profiling 不进入版本库；
- 不新增退出阶段跨 SO 清理逻辑，外部私有 `_summary` 只读取快照，不注册新的全局析构。

## 兼容性检查

MVP 不改变已有构建选项、脚本参数、安装目录、动态库、Python 包和用户模型。`ENABLE_AUTOFUSE_MLIR=OFF`
保持完全不依赖 MLIR；`--no-autofuse` 和默认测试目标不受影响。新增 CLI 首版 schema/参数会在 README 固定；后续
不兼容变更需升级 schema version。NNSmith、Torch 和 inductor-npu-ext 均记录实际版本，不能跨版本假定私有接口。

## 测试设计

### 测试边界

纯逻辑 UT 以 `CaseSpec`/伪 backend 结果为输入，验证生成、校验、分类和序列化。CPU 集成测试以真实 FX
`GraphModule` 为边界。NPU ST 以 `torch.compile(..., options={"npu_backend": "ascendc"})` 输出和融合证据为
出口，不 mock pyautofuse。MLIR NPU 必须走正式路由并执行 kernel，AFIR dump 只能作为补充证据。

### 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|---|---|---|---|
| 功能 | 同 seed 同 CaseSpec、不同 seed 有变化、合法 chain/broadcast | 固定 seed 比较序列化结果和 shape 推导 | pytest UT |
| 功能 | eager CPU 与 Inductor CPU 一致 | float32/float16 小图真实编译执行 | pytest ST |
| 功能 | Legacy NPU 实际融合 | 检查输出与 `fused_kernel_count > 0` | NPU ST |
| 异常 | 非法配置、依赖缺失、MLIR 路由缺失 | 检查状态、退出码和错误文本 | pytest UT/ST |
| 异常 | compile/runtime crash 与 timeout | 伪 worker/受控失败用例 | pytest UT/ST |
| 精度 | NaN/Inf/signed-zero、整数/bool、容差边界 | 构造 tensor 输出逐项检查 | pytest UT |
| 重放 | case/result schema 和 seed 重放 | 保存后重新生成并比较 graph/input metadata | pytest UT |
| 性能 | Legacy/MLIR 冷编译与热 kernel | 固定 NPU corpus，预热和重复采样 | Benchmark/nightly |
| 兼容性 | 默认 MLIR check 和产品构建不变 | 确认未注册默认 target；必要时全量回归 | UT/build |
| 特性交叉 | optimize/codegen/pyautofuse/Inductor 全链路 | 真实 NPU E2E 与融合证据 | NPU E2E |

### 测试命令

MVP 纯逻辑与 CPU 环境：

```bash
python3 -m pytest mlir/test/fuzz -v
python3 mlir/test/fuzz/af_fuzz.py --seed 1 --backends eager-cpu,inductor-cpu
```

Legacy NPU 环境：

```bash
python3 mlir/test/fuzz/af_fuzz.py --seed 1 --backends eager-cpu,legacy-npu --timeout-seconds 300
```

MLIR 路由实现后：

```bash
python3 mlir/test/fuzz/af_fuzz.py --seed 1 --backends eager-cpu,legacy-npu,mlir-npu --timeout-seconds 300
```

如修改 MLIR/C++ 集成，再执行：

```bash
bash scripts/test/run_autofuse_test.sh -u -m mlir -j 8
```

## 验收标准

MVP：

- 同 seed 在相同版本/config 下生成相同 CaseSpec、FX 结构和输入元数据；
- 首批静态 chain/broadcast 的 float32/float16 用例可在 eager CPU 和 Inductor CPU 执行并正确差分；
- Legacy NPU 环境可执行同一输入，输出按 profile 通过，且 `fused_kernel_count > 0` 才记 Autofuse 验证通过；
- crash、timeout、compile/runtime error、mismatch、fallback、not-exercised 和 unavailable 可区分；
- 失败可用单条 seed 命令重放，成功默认不保留大体积产物；
- `mlir-npu` 在正式路由不存在时清晰失败，不走 Legacy 冒充；
- pytest 全部通过，无新增默认构建/测试时长，无随机产物进入 Git。

后续完整验收：

- 覆盖指标包含 op、pair/triple、拓扑、dtype/rank/shape/layout、Autofuse 进入率/fallback 和 backend feature；
- 固定 correctness corpus 可同时跑 Legacy/MLIR NPU 并分类 Legacy-only、MLIR-only 和共同 failure；
- 固定 performance corpus 报告 compile time、kernel median/P90/几何平均和阈值退化；
- reducer 输出的稳定最小复现经人工确认后才进入 regression corpus。

### 设计文档检查结果

- [x] 跨特性交叉影响：已逐项检查 SuperKernel、Autofuse optimize/codegen、AscendC/runtime、Python/C++ 绑定、构建交付、测试、性能日志和兼容性。
- [x] 编码红线：已检查敏感信息、输入范围/溢出、资源生命周期、ABI/API、图确定性、日志、runtime 接口和临时产物，无设计性违反。
- [x] 测试设计：已覆盖 pytest UT、CPU ST、Legacy/MLIR NPU E2E、异常/timeout、精度、重放和性能 corpus。
- [x] 性能评估：已分开评估编译时长、执行性能和内存/产物，且禁止 CPU/NPU 横向性能结论。
