# Phase 1 MLIR 环境准备实施总结

> 状态：已完成（2026-07-04）
> 初始设计日期：2026-06-27
> 实施完成日期：2026-07-04
> 文档类型：阶段开发与实施总结
> 现状复核日期：2026-08-06

> 代码一致性：当前仓库保留 `ENABLE_AUTOFUSE_MLIR`、显式 LLVM/MLIR 依赖、官方 PyAsc + patchset、
> `af-opt`、`check-autofuse-mlir`、环境/同步脚本和 op inventory；原详细计划中的 `AF_MLIR_CODEGEN`、
> `af_mlir_diff.py`、`freeze_codegen_baseline.py` 和独立验证矩阵未作为当前能力保留，已在本文明确排除。

# 简介

## 目的

本文合并并取代原《Phase 1 环境准备工作方案》和《Phase 1 环境准备详细开发方案》，记录 Autofuse
接入 LLVM/MLIR 与 PyAsc 基础设施时最终保留的范围、关键决策、落地结果、验证结论和遗留问题。

本文只描述 Phase 1 已完成的环境与工程底座，不再保留逐任务执行脚本、已独立维护的指南全文或未进入
当前仓库的工具草案。后续 Codegen、ATT 和 Schedule 迁移方案见 `docs/design` 下的专题设计。

## 范围

本阶段包含：

- 默认关闭的 Autofuse MLIR 构建能力；
- LLVM/MLIR 显式依赖解析和 Docker 编译验证环境；
- 官方 PyAsc submodule 与仓内 patchset 管理；
- 最小 `af-opt`、MLIR LIT 和依赖静态检查；
- PyAsc AscendC op inventory；
- 对既有 `ascir_tool` 的兼容包装和环境信息采集。

本阶段不包含：

- 用 MLIR 替换默认 Codegen 路径；
- 实现 `AF_MLIR_CODEGEN=off|on|compare` 运行路由；
- 实现完整 legacy/MLIR manifest diff 或 baseline 冻结系统；
- 修改 SuperKernel、runtime 或产品对外 API/ABI；
- 把 `af-opt`、LLVM、MLIR 或 PyAsc runtime 安装进默认 run 包。

# 总体概述

## 软件概述

### 项目介绍

Phase 1 的目标不是交付新的 MLIR Codegen，而是在不影响默认 Autofuse 构建和运行路径的前提下，建立后续
AFIR、Codegen、ATT 和 Schedule 迁移所需的可选工程底座。核心原则是“构建能力与业务切换分离”：
`ENABLE_AUTOFUSE_MLIR` 只控制是否编译 MLIR 相关目标，不改变默认业务路径。

### 产品环境介绍

- 默认产品构建不要求 LLVM/MLIR/PyAsc；
- MLIR 开发构建依赖项目固定的 LLVM/MLIR 和 PyAsc 版本；
- CANN Toolkit 不打入 LLVM 开发镜像，通过宿主机挂载或环境注入提供；
- Docker/Linux 用于编译和测试，真实 NPU 环境用于设备验证；
- `af-opt`、环境 manifest 和 coverage inventory 都是开发资产，不是产品运行时接口。

## 软件功能

Phase 1 最终形成以下能力：

1. `ENABLE_AUTOFUSE_MLIR=OFF` 时不解析 LLVM/MLIR/PyAsc；
2. `ENABLE_AUTOFUSE_MLIR=ON` 时通过显式路径发现 LLVM/MLIR，并接入官方 PyAsc；
3. 提供开发容器、依赖准备、环境采集和 PyAsc 同步脚本；
4. 提供 `af-opt` 和 `check-autofuse-mlir` 开发验证入口；
5. 提供 PyAsc AscendC op coverage/inventory 查询；
6. 保持 `--no-autofuse`、默认 package 和既有 `ascir_tool` 语义不变。

## 设计约束

- `ENABLE_AUTOFUSE_MLIR` 默认必须为 `OFF`；
- `--no-autofuse` 继续表示跳过整个 Autofuse，不复用为 MLIR 开关；
- 依赖准备必须显式，不允许默认联网下载或静默编译 LLVM；
- PyAsc 必须来自官方仓固定 commit，Autofuse 修改通过仓内 patchset 维护；
- LLVM/PyAsc/CANN 版本和 ABI 必须可追溯；
- 默认 run 包不得安装 MLIR 开发工具和 LLVM/MLIR/PyAsc runtime；
- 所有构建命令限制并行度，优先使用 `-j 8`；
- generated code、coverage、profiling、`kernel_meta` 和环境产物不得入库。

## 假设和依赖关系

- Autofuse 默认构建仍以 legacy 路径为兼容基线；
- LLVM/MLIR 预编译产物需与项目编译器和 `_GLIBCXX_USE_CXX11_ABI` 配置匹配；
- CANN 环境通过 `ASCEND_HOME_PATH` 或开发容器挂载提供；
- 后续业务迁移通过独立设计推进，不由 Phase 1 环境开关自动启用；
- 本文的“Closed”表示环境底座范围关闭，不表示所有 legacy NPU/CANN 兼容问题均已解决。

# 需求分析与设计

## 整体介绍

Phase 1 将 MLIR 能力集中在 `autofuse/mlir`，仅在顶层构建入口做必要接线：

```text
build.sh / autofuse/CMakeLists.txt
  └─ ENABLE_AUTOFUSE_MLIR=ON
      └─ mlir/cmake/AutofuseMlirDeps.cmake
          ├─ LLVM/MLIR config
          ├─ externals/pyasc + patches/pyasc
          ├─ tools/af-opt
          ├─ tools/af-mlir-tools
          └─ test/check-autofuse-mlir
```

默认关闭时，MLIR 子工程在入口处返回，不向 legacy 构建图传播 LLVM/MLIR target。

## 功能需求

### 功能需求 1：保持可选构建边界

1. 介绍

   通过独立开关控制 Autofuse MLIR 能力是否进入构建图。

2. 输入

   `BUILD_AUTOFUSE`、`ENABLE_AUTOFUSE_MLIR`、`--no-autofuse` 和 `--enable-mlir` 构建选项。

3. 处理

   默认关闭时跳过 MLIR 依赖解析；显式打开后加载 LLVM/MLIR/PyAsc，并构建项目 MLIR targets。

4. 输出

   默认构建无新增 LLVM 依赖；开发构建获得 `af-opt`、AFIR libraries 和 MLIR 测试入口。

### 功能需求 2：提供显式依赖与环境输入

1. 介绍

   统一解析开发者提供的 LLVM/MLIR 目录，并记录环境信息。

2. 输入

   `LLVM_BUILD_DIR`、CMake package 路径、CANN 环境和可选开发镜像。

3. 处理

   通过依赖解析脚本和 CMake probe 检查 `MLIRConfig.cmake`；通过开发容器挂载 CANN；通过环境脚本记录
   架构、工具链和关键依赖版本。缺失依赖时直接报错，不隐式下载。

4. 输出

   可复现的 MLIR 开发环境和清晰的依赖错误信息。

### 功能需求 3：接入官方 PyAsc

1. 介绍

   PyAsc 作为固定版本 submodule 接入，并与主工程复用同一 LLVM/MLIR 配置。

2. 输入

   官方 PyAsc gitlink 和 `mlir/patches/pyasc` patchset。

3. 处理

   `sync_pyasc_upstream.sh` 恢复固定 upstream、应用仓内 patch，并保护本地 gitlink；
   `export_pyasc_patch.sh` 从独立 checkout 导出评审过的 patch。

4. 输出

   可追溯、可复现且不依赖个人 fork 的 PyAsc 源码输入。

### 功能需求 4：建立最小验证入口

1. 介绍

   提供能验证依赖、方言注册、IR parse/print 和 PyAsc coverage 的最小工具集。

2. 输入

   Autofuse MLIR 源码、PyAsc TableGen 文件和代表性 MLIR 测试文件。

3. 处理

   构建 `af-opt`，运行 `check-autofuse-mlir`，执行构建开关/依赖/PyAsc 静态检查，并扫描 PyAsc op。

4. 输出

   开发工具、LIT 结果、静态检查结果和 op inventory CSV。

## 非功能需求

### 可维护性

- MLIR 依赖逻辑集中在 `mlir/cmake` 和 `mlir/scripts`；
- 环境、PyAsc 和验证工具各有单一入口；
- 当前使用方法放在 `docs/guides`，本文只记录阶段实施结论；
- 未落地的历史工具草案不再作为当前能力展示。

### 可测试性

构建开关、LLVM 依赖解析和 PyAsc 接入分别有静态检查脚本；MLIR dialect/pass 通过 `af-opt` 和 LIT 验证。

### 可移植性

依赖输入支持开发容器和显式本地路径。预编译 LLVM/MLIR 需要分别提供兼容架构的制品，不允许假设
x86_64 产物可在 aarch64 复用。

### 可靠性

缺失 LLVM/MLIR/PyAsc 时显式失败；默认关闭路径不触发依赖发现。脚本生成物进入被忽略目录，不修改产品
运行状态。

### 特性交叉影响

| 场景 | 适用性 | 结论 |
|---|---|---|
| SuperKernel Python 接口 | 不适用 | 不修改 Python 包、选项和 wheel |
| SuperKernel C++/AOT 接口 | 不适用 | 不修改 `libascendsk.so` 或 AOT ABI/API |
| Autofuse 图优化 | 不适用 | Phase 1 不修改图优化和 pass 语义 |
| Autofuse Codegen/Backend | 适用 | 提供后续迁移需要的构建与测试底座，不切换默认 Codegen |
| AscendC API / Runtime 交互 | 部分适用 | 依赖 CANN/PyAsc 编译环境，不新增 runtime/aclrt 调用 |
| Python/C++ 混合绑定 | 部分适用 | 不改变对外绑定；开发工具错误需清晰返回 |
| 构建与打包 | 适用 | 新增可选 MLIR 构建图，默认包边界保持不变 |
| 测试与覆盖率 | 适用 | 新增静态检查、LIT 和 op inventory |
| 性能与日志 | 适用 | MLIR 开启后增加构建成本，默认路径无运行时开销 |
| 兼容性 | 适用 | `--no-autofuse`、默认构建和既有 `ascir_tool` 行为保持 |

## 性能

### 编译时长

`ENABLE_AUTOFUSE_MLIR=OFF` 时不构建 LLVM/MLIR/PyAsc targets，不增加默认编译时长。显式打开后会增加
TableGen、静态库编译和链接成本；统一使用预编译 LLVM/MLIR，并限制 `-j 8` 避免 OOM。

### 执行性能

Phase 1 不切换业务 Codegen 或 runtime 路径，无设备执行性能影响。

### 内存和产物大小

LLVM/MLIR 开发依赖和构建目录体积较大，但不进入默认 run 包。环境 manifest、coverage 和测试产物进入
被忽略目录或 CI artifact。

## 接口设计

### 新增/修改接口描述

| 接口 | 类型 | 当前用途 |
|---|---|---|
| `ENABLE_AUTOFUSE_MLIR` | CMake/build option | 控制是否编译 Autofuse MLIR 能力，默认 `OFF` |
| `LLVM_BUILD_DIR` | 环境/CMake 路径 | 指定已有 LLVM/MLIR 配置 |
| `af-opt` | 开发工具 | 注册项目 dialect/pass，执行 parse/print 和 LIT |
| `check-autofuse-mlir` | CMake target | 运行 Autofuse MLIR LIT |
| `scan_pyasc_coverage.py` | 开发脚本 | 生成 PyAsc op coverage/inventory |
| `af_mlir_case.py` | 开发脚本 | 包装既有 `ascir_tool` 并限制内部并行度 |

`AF_MLIR_CODEGEN=off|on|compare` 只存在于后续迁移设计，本阶段和当前代码未实现，不应作为现有接口使用。

### 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|---|---|---|---|
| 接口说明 | 是否需要接口评审 | 是 | 构建开关和依赖输入需要维护者评审 |
| 接口说明 | 是否需要补充文档 | 是 | 当前用法位于 `mlir/README.md` 和 `docs/guides` |
| 接口兼容 | 行为是否兼容 | 是 | 默认关闭，不改变 legacy 行为 |
| 接口兼容 | ABI/API 是否兼容 | 是 | 不修改产品导出 ABI/API |
| 接口约束 | 失败是否清晰 | 是 | 缺失 MLIR/PyAsc 时 CMake 显式失败 |
| 接口测试 | 是否需要独立用例 | 是 | 静态脚本、CMake probe 和 LIT 覆盖 |

## 软件设计

### 关键数据结构

Phase 1 不新增产品运行时数据结构。环境 manifest 和 op inventory 是开发期结构化数据，生命周期限于开发或
CI 过程。

### 关键技术/算法

- CMake 入口早返回隔离默认构建与 MLIR 构建；
- `find_package(MLIR CONFIG)` 使用显式路径，不进行隐式下载；
- PyAsc patchset 与 upstream gitlink 分离，避免 submodule 工作区成为修改来源；
- coverage 工具从 TableGen 源提取 op 信息，不手工维护重复清单。

### 流程设计

```text
准备 LLVM/MLIR 与 CANN 环境
  → 同步固定 PyAsc + 应用 patchset
  → ENABLE_AUTOFUSE_MLIR=ON 配置
  → 构建 af-opt / AFIR targets
  → 运行静态检查与 check-autofuse-mlir
  → 按需生成 PyAsc op inventory
```

### 对子模块的修改

| 子模块 | Phase 1 落地结果 |
|---|---|
| `mlir/cmake` | MLIR 依赖解析与构建隔离 |
| `mlir/docker` | 开发镜像和 CANN volume 使用说明 |
| `mlir/scripts` | 依赖准备、环境进入/采集、PyAsc 同步和静态检查 |
| `mlir/externals/pyasc` | 官方 PyAsc 固定版本输入 |
| `mlir/patches/pyasc` | Autofuse 维护的最小 patchset |
| `mlir/tools/af-opt` | MLIR 开发优化器入口 |
| `mlir/tools/af-mlir-tools` | ascir wrapper 和 op coverage 工具 |
| `mlir/test` | MLIR LIT 与 UT 接线 |

历史详细计划曾描述 `af_mlir_diff.py`、`freeze_codegen_baseline.py` 和独立 Phase 1 验证矩阵；这些内容未保留为
当前仓库能力，已从实施总结中移除。需要重新引入时必须按新需求设计和验证，不能引用旧草案直接实现。

### 错误处理

#### 系统错误

缺失 `MLIRConfig.cmake`、PyAsc submodule 或不兼容依赖时，由 CMake/脚本直接失败并给出配置指引。构建失败
不回退到隐式下载或自动修改系统环境。

#### 接口错误

开发脚本使用非零退出码报告非法参数、缺失文件和下游工具失败；不吞掉 `ascir_tool` 返回值。

## 安全检查

- 不新增密码、Token、账号或个人 fork 依赖；
- 不以外部输入直接进行未校验的数组索引、内存长度或 runtime 调用；
- 不新增 runtime/aclrt 资源生命周期；
- 环境和测试产物不进入版本库；
- 默认构建不联网、不静默下载或编译重型依赖；
- 不修改图结构和 pass 语义，不涉及图改写确定性；
- manifest/inventory 输出应保持稳定顺序；
- 所有构建限制 `-j 8`，避免无限并行导致 OOM。

## 兼容性检查

- `ENABLE_AUTOFUSE_MLIR` 默认关闭；
- `--no-autofuse` 语义不变；
- 默认产品包不包含 `af-opt`、LLVM、MLIR 或 PyAsc runtime；
- 不修改 SuperKernel、Python/C++ 绑定或产品 ABI/API；
- 既有 `ascir_tool` mode 0/1/2 接口不因 wrapper 改变；
- PyAsc 来源固定为官方仓，不依赖个人 fork。

## 测试设计

### 测试边界

- 静态检查：构建开关、LLVM 依赖解析、PyAsc submodule/patch 管理契约；
- 构建：默认路径、`--no-autofuse` 和 MLIR 开启路径；
- MLIR：真实 LLVM/PyAsc/AFIR 配置、`af-opt` 构建和完整 `check-autofuse-mlir`；
- 工具：环境采集、PyAsc 同步、coverage 扫描和 `af_mlir_case.py --help`；
- NPU：既有 `ascir_tool` 代表 case，失败按环境/编译/launch 分层记录。

### 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|---|---|---|---|
| 功能 | MLIR OFF/ON、PyAsc 固定输入、`af-opt` | 脚本 + 定向构建 | UT/LIT |
| 异常 | 缺失 MLIRConfig、缺失 PyAsc、非法工具参数 | 依赖脚本 + 真实 CMake 失败路径 | UT |
| 构建 | 默认 package、`--no-autofuse`、MLIR package | 构建矩阵 | ST |
| 兼容性 | 默认包边界、既有 ascir 工具入口 | 文件检查 + E2E | ST |
| 性能 | MLIR 开启后的构建成本和内存 | 构建记录 | 手工验证 |

### 测试命令

```bash
bash autofuse/mlir/scripts/check_mlir_build_switch.sh
bash autofuse/mlir/scripts/check_llvm_dependency_resolution.sh
bash autofuse/mlir/scripts/check_pyasc_integration.sh

AF_CANN_VOLUME=<cann-volume> bash autofuse/mlir/scripts/enter_dev_container.sh -- \
  bash build.sh --pkg --no-autofuse -j 8

AF_CANN_VOLUME=<cann-volume> bash autofuse/mlir/scripts/enter_dev_container.sh -- \
  bash build.sh --pkg --enable-mlir -j 8

AF_CANN_VOLUME=<cann-volume> bash autofuse/mlir/scripts/enter_dev_container.sh -- \
  bash scripts/test/run_autofuse_test.sh -u -m mlir -j 8
```

Phase 1 关闭时记录的已知限制：

- 默认 package 路径修复 CANN/tikicpulib 挂载后，继续暴露了既有 `isinf_maskedfill_test_e2e` 与当时
  CANN cpudebug `vabs(int*)` 的兼容问题；
- 真实 NPU 链路未完整打通；
- 上述问题归入后续真实链路验证，不作为环境底座关闭阻塞项。

## 验收标准

- 默认构建不依赖 LLVM/MLIR/PyAsc；
- MLIR 开启后能够解析固定依赖并构建项目 MLIR targets；
- `--no-autofuse` 和默认 package 边界保持兼容；
- PyAsc 使用官方固定版本和仓内 patchset；
- `af-opt`、LIT、环境脚本和 op inventory 入口可用；
- 环境、coverage、generated code 和测试临时产物不入库；
- 已知 CANN/NPU 限制被明确记录，不被误写成已解决；
- 后续 Codegen/ATT/Schedule 迁移由 `docs/design` 专题文档继续管理。

### 设计文档检查结果

- [x] 跨特性交叉影响：已检查 SuperKernel、Autofuse Codegen/Backend、AscendC、Python/C++、构建打包、
  测试和兼容性场景。
- [x] 编码红线：已检查离线/增量构建、临时产物、依赖来源、ABI/API、资源生命周期和并行构建约束。
- [x] 测试设计：已覆盖静态检查、默认/MLIR 构建路径、LIT、工具和后续 NPU 验证。
- [x] 性能评估：已说明默认路径无新增成本、MLIR 开启后的构建开销及产物边界。
