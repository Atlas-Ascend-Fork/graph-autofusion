# NPU 用例

本目录通过 Torch / Inductor 验证编译、代码生成和 NPU 数值结果，包含基础逐元素、广播及
Reduce 用例。设备测试不接入宿主机 `check-e2e`；运行工具的离线自测接入 `check-ut`。

| 入口 | 用途 |
|---|---|
| `autofuse/mlir/scripts/npu.py` | 从本地按 commit 同步代码、构建并启动设备回归 |
| `run.py` | 在已配置的 NPU 环境中按清单运行、汇总测试 |
| 各用例脚本 | 单独运行或调试一个用例 |

## 从本地发起设备回归

设备需要已安装 CANN、LLVM、torch_npu 及构建依赖。首次初始化专用工作区，再将第三方
依赖放入 `deps/`；不要使用已有未提交修改的开发目录。

```bash
python3 autofuse/mlir/scripts/npu.py --host <SSH别名> --root /mnt/workspace/af-npu init

python3 autofuse/mlir/scripts/npu.py --host <SSH别名> --root /mnt/workspace/af-npu run \
  --commit HEAD \
  --cann <设备CANN目录> --llvm <设备LLVM目录> \
  --deps /mnt/workspace/af-npu/deps/third_party \
  --cann-cmake /mnt/workspace/af-npu/deps/cann-cmake

python3 autofuse/mlir/scripts/npu.py --host <SSH别名> --root /mnt/workspace/af-npu status <run_id>
```

- 通过 Git bundle 同步指定 commit，无需 push；不包含未提交修改。LFS 文件保留指针。
- 设备固定使用 `source/`、`build/`、`overlay/`，日志存入 `runs/<commit>-<run-id>/`。
- 任务在后台执行，SSH 断开不影响运行；同一工作区一次只允许一个任务。
- 默认增量构建 `pyautofuse -j 8`；构建配置变化时自动清理，也可指定 `--clean-build`。
- 每轮重建 overlay 并校验模块来源、动态库和设备可用性，失败则不运行用例。
- Python 默认 `/usr/bin/python3.12`，标准库前缀默认 `/usr`，分别用 `--python`、
  `--python-home` 修改。

默认串行运行全量 MLIR 清单，可用 `--group reduce` 或 `--id load_store_direct` 缩小范围。
提交成功仅表示任务已启动；**`status.json` 中 `phase=complete` 且 `returncode=0` 才表示通过**。
详细日志见运行目录中的 `build.log`、`test.log` 和 `cases/`。

## 在设备上运行清单

在已加载 CANN、具备匹配 build 的设备环境中执行：

```bash
python3 autofuse/mlir/test/e2e/npu/run.py --list
python3 autofuse/mlir/test/e2e/npu/run.py --group smoke
python3 autofuse/mlir/test/e2e/npu/run.py --id load_sum_store
python3 autofuse/mlir/test/e2e/npu/run.py                   # 全量
python3 autofuse/mlir/test/e2e/npu/run.py --group reduce --calibrate
```

`cases.py` 定义脚本、参数、环境变量和预期结果。同一脚本可按不同 plan 注册多项；每项使用
独立子进程，避免 ATT DFX 配置相互影响。单脚本与统一入口共用判级逻辑。

### 结果判定

- 数值不匹配、执行错误、子进程非零退出或缺失预期结果，均判失败。
- 配置 `expect` 时，实测比预期差判失败，比预期好提示更新基线；未配置时，支持边界状态
  不单独判失败。
- `PASS_FALLBACK` 表示数值通过但出现已知拒收诊断，不代表 MLIR 全链路通过。
- `--calibrate` 仅输出建议的预期配置，不修改文件；异常退出的用例不生成基线。

### 环境与并发

`support/device_env.py` 配置 overlay、动态库路径并核对 Python 模块来源，确保使用匹配的
源码与构建产物。可用 `--build-dir`、`--overlay` 指定路径；单独使用 `run.py` 且源码删除过
Python 文件时，使用 `--rebuild-overlay` 清除残留。正常回归不要跳过 preflight。

默认 `--jobs 1`。`--jobs 2` 为两个 worker 分配独立设备，但设备绑定尚未完成实机验证，
暂建议使用串行。

## 离线自测

无需 NPU，覆盖清单一致性、判级、异常退出、环境配置和 commit 同步流程：

```bash
python3 autofuse/mlir/test/e2e/npu/support/test_run_entry.py
python3 autofuse/mlir/test/e2e/npu/support/test_allload_cases.py
python3 autofuse/mlir/test/e2e/npu/support/test_npu_session.py
# 已配置 MLIR 测试的构建中，也可统一运行：
cmake --build build --target check-npu-runner-offline -j 8
```

## 测试框架介绍

简易的测试框架： `support/npu_runner.py` 。

### 功能介绍
后续用例只需定义 `torch.nn.Module` 的计算图和输入 shape，
并在入口把用例名、构图函数和输入 shape 传给 `NpuRunner.main()`。工具类会自动生成固定随机的 CPU 输入，
执行每组 shape 的 CPU eager 生成期望输出，再执行 NPU eager、Inductor 和 profiling。

### 参数介绍
`--mode mlir|legacy|both` 三种模式：
* `mlir`: 默认模式，走 MLIR-Codegen 后端（`--mlir_codegen_config=mlir`）；
* `legacy`: 走旧路径（`--mlir_codegen_config=none`）；
* `both`：分别运行 `legacy` 和 `mlir`，并把两条路线的中间结果输出到 `--work-dir` 目录下。

`--dynamic` 两种场景：
* `dynamic=False`: 默认场景，静态Shape输入；
* `dynamic=True`: 动态Shape输入后；

`--steps`: 执行次数，默认为 `100`，则表示执行100次且开启`profiling`，可设置为 `0` 关闭；
`--device`: 指定硬件资源ID，默认 `npu:0`；
`--work-dir`: 中间结果落盘路径，默认为当前目录，与测试用例同名。

## 用例执行

### 代码示例

```python
import sys

from support.npu_runner import NpuRunner

INPUT_SHAPES = ((128, 50), (128, 50))


def build_module(torch):
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return torch.add(x, y)

    return Model()


if __name__ == "__main__":
    sys.exit(NpuRunner.main("MY_CASE", build_module, INPUT_SHAPES))
```

入口用 `NpuRunner.main` 而不是 `NpuRunner.run`：前者在 fd 层面捕获 stderr、按统一词汇判级、
打一行机读结果给 `run.py` 对账，退出码只在真缺陷上非零。`run` 保留给要拿返回张量的调用方，
它把异常原样抛出去，于是所有失败长得一样——已记录的支持边界和真缺陷分不开。

### 环境设置

脚本默认设置 `AUTOFUSE_DFX_FLAGS` 中的 `--mlir_codegen_config=mlir` 和 `TORCH_COMPILE_DEBUG=1`。

走 `run.py` 时下面这些**不用自己设**，它按 §2.4 装好并在跑之前验过（见「环境与并发」）。手工单跑某一个脚本时才需要照下面设：

```bash
# 当前仓库的 build/autofuse 目录，里面应包含 libaihac_codegen.so
export AUTOFUSE_BUILD_DIR=/path/to/autofuse/build/autofuse

# 远端示例：/home/developer/autofuse/build/autofuse
export LD_LIBRARY_PATH="${AUTOFUSE_BUILD_DIR}:${AUTOFUSE_BUILD_DIR}/graph_metadef/graph/ascendc_ir:${AUTOFUSE_BUILD_DIR}/graph_metadef/graph/expression:${LD_LIBRARY_PATH:-}"

# 如果使用当前仓库的 Python overlay，将其放在 CANN site-packages 之前
export PYTHONPATH=/path/to/autofuse/python/site-packages:${PYTHONPATH:-}

# 避免 NPU kernel.so 被 umask 生成为组或其他用户可写
umask 077
```

切换构建版本后如仍命中旧的 NPU kernel 缓存，可以先清理当前用户的融合 kernel 缓存：

```bash
rm -rf "/tmp/.npu_kernels_${USER}"/autofused_*
```

### 执行用例
```bash
python3 autofuse/mlir/test/e2e/npu/load_add_store.py --work-dir /tmp/mlir-load-add-store
python3 autofuse/mlir/test/e2e/npu/load_add_store.py --dynamic --steps 0  --work-dir /tmp/mlir-load-add-store-dynamic
python3 autofuse/mlir/test/e2e/npu/load_add_store.py --mode both --work-dir /tmp/mlir-load-add-store
python3 autofuse/mlir/test/e2e/npu/load_store_direct.py --work-dir /tmp/mlir-load-store-direct
python3 autofuse/mlir/test/e2e/npu/mixed_consumption.py --work-dir /tmp/mlir-mixed-consumption
```

### Broadcast / 多节点级联用例

下列用例均使用 `--mode mlir --steps 0` 验证 MLIR 路线；输出中的
`new_autofused_artifact_count` 用于确认当前 shape 生成了新的 `autofused_*` artifact。

| 用例 | 覆盖点 |
|---|---|
| `load_broadcast_add_direct.py` | 直接 `add(small, full)`，覆盖外轴、内轴与交错轴广播 |
| `load_broadcast_to_abs.py` | `broadcast -> abs` 最小链路 |
| `load_broadcast_to_cascade.py` | `broadcast -> abs -> mul -> relu` 多节点级联，覆盖 unary inline broadcast |
| `load_broadcast_inner_brc_cascade.py` | 内轴广播，inner row 大于一个向量化窗口 |
| `load_broadcast_tail_pitch_cascade.py` | 20B / 12B / 4B 尾部 pitch 与 32B 对齐 |
| `load_broadcast_interleaved_axes_cascade.py` | `[brc][normal][brc][normal]` 交错轴与非 32B 对齐行 |
| `load_broadcast_diamond_cascade.py` | diamond 拓扑：`add(broadcast) -> {abs, neg} -> add`，覆盖时间复用与多轴混合 |

## 仓内用例一览

| 脚本 | 测什么 |
|---|---|
| `load_add_store.py` | 逐元素基线 |
| `load_sum_store.py` | reduce 基线 |
| `load_sum_store_rcore.py` | RCore 专用形状；是否实际跨核需核对 tiling 的 R 分片数 |
| `reduce_template_matrix.py` | 调度模板 × 归约形态的固定矩阵 |
| `reduce_all.py` | 全轴归约（all-reduce），秩 × keepdim 铺开 |
| `reduce_rarr.py` | 保留轴**不在末位**的多轴归约（`(8,8,128,128)` 归约 `(0,2,3)`） |
| `reduce_fusion_matrix.py` | 归约**带邻居**的图（喂入 / 消费 / fork / 一图两归约），测切图与 workspace |

## 成败词汇（全部用例共用）

这套词汇 2026-10-16 随三张矩阵表引入，当时只有那三张表在用；现在**所有脚本共用**同一份，
定义在 `support/case_status.py`，等级与严重度次序是 `run.py` 能与清单预期对账的前提。

六个有序等级：`PASS` → `PASS_FALLBACK` → `REFUSED` → `NOKERNEL` → `MISMATCH` →
`ERROR`，以及不在严重度阶梯上的 `SKIP`。`PASS_FALLBACK` 表示数值通过但出现已知拒收诊断，
不能作为 MLIR 全链路通过的证据。当前能力不足路径会转 legacy codegen，并非仅淘汰部分
MLIR 候选；实际路径需结合 codegen 与产物核对。

`reduce_template_matrix`、`reduce_all` 和 `reduce_rarr` 使用
`support/matrix_report.run_matrix`，统一捕获 fd 2 的下降诊断并输出机读结果。
单脚本退出码与清单预期共同决定整轮结果，数值失败和异常退出不能被已知支持边界掩盖。

## Reduce 模板选择

Reduce 矩阵使用 legacy/MLIR 共用的 ATT DFX 选择机制：

```bash
AUTOFUSE_DFX_FLAGS="--force_schedule_result=0" python3 reduce_template_matrix.py
AUTOFUSE_DFX_FLAGS="--force_schedule_result=0;--force_tiling_case=0" python3 reduce_template_matrix.py
```

编号对应 schedule 生成的 plan/case。MLIR 不在编排阶段过滤模板或重编号：
所有模板都必须成功 lowering，随后才生成 ATT tiling。任何候选的能力不足触发整图回落，
其他编译错误仍报失败。CPU 的 `load_neg_sum_store_arar_case3` 直接指定 case 3；
CPU 测试会断言实际执行的 plan/case，再比较数值。

ATT 在进程内缓存 DFX 配置。
`run.py` 对整张清单贯彻同一条约束：每条用例一个子进程，否则第二条钉的还是第一条那个 plan。
指定无解或不存在的候选会失败；检查生成的 tiling 和实际结果，不应只看环境变量。
