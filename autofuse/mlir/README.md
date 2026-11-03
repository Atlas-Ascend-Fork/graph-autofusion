# Autofuse MLIR 开发指导

| 文档说明    | 文档路径  |
|-------------|-----------|
| 整体架构    | [docs/README.md](docs/README.md) |
| AscendC API | [api/ascendc/README.md](api/ascendc/README.md) |

## 说明

先跑轻量检查，确认 LLVM 依赖路径解析逻辑本身没问题:

```bash
bash autofuse/mlir/scripts/check_llvm_dependency_resolution.sh
```

真实编译执行分两种方案，任选其一。

## 方案一: Docker（推荐 macOS/Windows）

### Step1: 拉取预编译好的 LLVM 镜像

```bash
bash autofuse/mlir/docker/pull_dev_image.sh
bash autofuse/mlir/docker/validate_llvm_image.sh autofuse-mlir-dev:arm64   # AArch64，x86-64 用 :amd64
```

### Step2: 安装 CANN Toolkit包（制作为 Docker Volume）

| CANN包版本    | 下载路径  |
|-------------|-----------|
| 开发版本    | [每日最新开发版本](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master)|
| 商用版本 | [CANN 官网下载中心](https://www.hiascend.com/cann/download) |

```bash
bash autofuse/mlir/docker/prepare_cann_volume.sh --toolkit-url <cann-toolkit-run-url-或本地文件路径>
# 默认创建的 Docker Volume 名称为 cann-volume
docker volume inspect cann-volume
```
`enter_dev_container.sh` 会把 volume 挂进容器、自动 source `set_env.sh`，并设好
`LLVM_BUILD_DIR`/`LLVM_DIR`/`MLIR_DIR`/`ASCEND_HOME_PATH` 等变量，不需要手动导出。

### Step3: 进入容器，编译与执行用例

```bash
# 构建 run 包（关闭 MLIR 接入）
bash autofuse/mlir/scripts/enter_dev_container.sh -- bash build.sh --pkg -j 8

# 构建 run 包（启用 MLIR 接入）
bash autofuse/mlir/scripts/enter_dev_container.sh -- bash build.sh --pkg --enable-mlir -j 8

# 跑全量 Autofuse UT
bash autofuse/mlir/scripts/enter_dev_container.sh -- bash build.sh -u --module=autofuse_framework -c -j 8

# 跑全部 MLIR UT、ST、lit 和 CPU E2E
bash autofuse/mlir/scripts/enter_dev_container.sh -- bash scripts/test/run_autofuse_test.sh -m mlir -j 8

# 只跑 MLIR UT 和 lit
bash autofuse/mlir/scripts/enter_dev_container.sh -- bash scripts/test/run_autofuse_test.sh -u -m mlir -j 8

# 只跑 MLIR ST 和 CPU E2E
bash autofuse/mlir/scripts/enter_dev_container.sh -- bash scripts/test/run_autofuse_test.sh -s -m mlir -j 8
```

### Step4: 在 CLion 中配置 Docker
[CLion Docker Toolchain 配置指南](docs/guides/clion_docker_toolchain.md)。

## 方案二: 本地编译（Linux 本机/远端）

### Step1: 编译 LLVM/MLIR
> 建议使用一下脚本克隆并编译 LLVM/MLIR（可自行修改脚本中的具体编译选项、参数等），否则会有接口形式不兼容问题

默认下载编译 `LLVM/MLIR 21.1.8 版本` 并安装到 `/opt/llvm`。

```bash
# 先指定 LLVM 源码存放路径，如果是CANN WebIDE开发环境，则建议：
LLVM_SOURCE_DIR=/mnt/workspace/gitCode/llvm-project
# 否则个人环境自行定义：
LLVM_SOURCE_DIR="${HOME}/llvm-project"
# 检查安装必要的工具，克隆、编译代码，安装 LLVM 编译结果
bash autofuse/mlir/scripts/build_llvm.sh "${LLVM_SOURCE_DIR}"
```

### Step2: 安装 CANN Toolkit包（下载路径见方案一的Step2）:

```bash
chmod +x Ascend-cann-toolkit_<version>_linux-<arch>.run
./Ascend-cann-toolkit_<version>_linux-<arch>.run --install --install-path=<install_path>
```

### Step3：编译和执行 Autofuse 测试用例（UT、ST、E2E CPU）
> 若在 CANNLab 环境中进行编译，CANN 包通常会安装在 `${HOME}/Ascend` 路径下，
> 即 `source ${HOME}/Ascend/ascend-toolkit/set_env.sh`，其他环境自行设置。

```bash
export LLVM_BUILD_DIR=/opt/llvm
export LLVM_DIR=/opt/llvm/lib/cmake/llvm
export MLIR_DIR=/opt/llvm/lib/cmake/mlir
source <install_path>/ascend-toolkit/set_env.sh

# 构建 run 包（启用 MLIR 接入）
bash build.sh --pkg --enable-mlir -j 8

# 跑全部 MLIR UT、ST、lit 和 CPU E2E
bash scripts/test/run_autofuse_test.sh -m mlir -j 8

# 只跑 MLIR UT 和 lit
bash scripts/test/run_autofuse_test.sh -u -m mlir -j 8

# 只跑 MLIR ST 和 CPU E2E
bash scripts/test/run_autofuse_test.sh -s -m mlir -j 8
```

### Step4：执行 Autofuse E2E NPU 用例
> 通常使用 CANNLab 提供的真实 NPU 环境

1. 先初始化 TensorFlow 或 PyTorch/Inductor 环境
```shell
# 当前 E2E NPU 用例仅支持 Inductor，初始化命令如下：
bash scripts/env_install/pytorch/setup_torch_npu_daily.sh
```

2. 执行用例
```shell
python3 autofuse/mlir/test/e2e/npu/load_add_store.py
```

## MLIR codegen 运行时配置

运行时通过既有 `AUTOFUSE_DFX_FLAGS` 中的 `--mlir_codegen_config` 控制 MLIR Codegen：

- `none`：默认值，不启用 MLIR Codegen，直接走 legacy。
- `mlir`：只走 MLIR，失败或暂不支持都返回失败，不回退；测试和验证使用该配置。
- `fallback`：优先走 MLIR，MLIR 失败或暂不支持时回退 legacy。
- 非法值直接返回失败，不静默走 legacy。

```bash
export AUTOFUSE_DFX_FLAGS="--mlir_codegen_config=mlir;--codegen_compile_debug=true;--debug_dir=/tmp/af_debug;"
```

该配置独立于 `--codegen_compile_debug`，无需开启 Dump 即可生效。
若构建时 `ENABLE_AUTOFUSE_MLIR=OFF`，即使运行时配置为 `mlir` 或 `fallback`，也会走 legacy。
