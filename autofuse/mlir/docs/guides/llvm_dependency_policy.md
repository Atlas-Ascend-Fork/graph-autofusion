# LLVM/MLIR 依赖策略

> 文档类型：开发与构建指南

## 目标场景

1. 本地开发：开发者使用团队统一 dev+LLVM Docker 镜像，或通过 `LLVM_BUILD_DIR`
   指向本地已编译的 LLVM/MLIR 目录。
2. Docker 编译验证：容器内默认从 `/opt/llvm` 提供 LLVM/MLIR，环境变量由镜像或
   `enter_dev_container.sh` 设置。
3. Linux/NPU 实测：使用已确认的 LLVM/MLIR 镜像或预编译目录，不要求 NPU 环境临时全量编译
   LLVM。
4. CI 流水线：拉取团队镜像或预编译制品；普通验证任务不得隐式下载或编译 LLVM。


## 镜像约定

默认团队镜像为统一的 LLVM 镜像：

```text
AF_MLIR_DEV_IMAGE=autofuse-mlir-dev:<arch>
LLVM_BUILD_DIR=/opt/llvm
LLVM_DIR=/opt/llvm/lib/cmake/llvm
MLIR_DIR=/opt/llvm/lib/cmake/mlir
```

`AF_MLIR_LLVM_IMAGE` 和 `AF_MLIR_LLVM_BASE_IMAGE` 只作为兼容别名保留。新脚本、README 和 CI
配置应优先使用 `AF_MLIR_DEV_IMAGE` 与 `AF_MLIR_DEV_BASE_IMAGE`。

## 本地目录输入

本地已编译目录必须满足：

```text
<llvm-root>/bin/llvm-config
<llvm-root>/bin/mlir-opt
<llvm-root>/bin/mlir-tblgen
<llvm-root>/bin/llvm-lit
<llvm-root>/lib/cmake/llvm/LLVMConfig.cmake
<llvm-root>/lib/cmake/mlir/MLIRConfig.cmake
```

可通过以下方式显式构建 LLVM 镜像：

```bash
LLVM_BUILD_DIR=/path/to/prebuilt/llvm bash autofuse/mlir/docker/build_llvm_image.sh
```

如果从 `llvm-project` 源码显式构建，保留构建 manifest：

```bash
AF_LLVM_SOURCE_DIR=/path/to/llvm-project bash autofuse/mlir/docker/build_llvm_image.sh
```

## 必须校验

`validate_llvm_image.sh` 是团队镜像发布前的最低校验入口：

```bash
bash autofuse/mlir/docker/validate_llvm_image.sh "${AF_MLIR_DEV_IMAGE}"
```

校验项包括：

- `llvm-config`、`mlir-opt`、`mlir-tblgen`、`llvm-lit` 可执行。
- `LLVMConfig.cmake` 和 `MLIRConfig.cmake` 存在。
- `llvm-config --cxxflags` 包含 `_GLIBCXX_USE_CXX11_ABI=0`。
- LLVM 开启 RTTI 和 EH，不包含 `-fno-exceptions`。
- 镜像不包含 `/opt/cann` 或 `/usr/local/Ascend/ascend-toolkit`。
- 开发常用工具、Python 包和 `pybind11==2.13.1` 可用。
- `/opt/llvm/share/autofuse-mlir/llvm-image-manifest.json` 存在。

## 推荐 LLVM 构建参数

`build_llvm_image.sh` 从源码构建时使用以下基线：

```text
CMAKE_BUILD_TYPE=Release
LLVM_ENABLE_PROJECTS=mlir
LLVM_TARGETS_TO_BUILD=host
LLVM_ENABLE_ASSERTIONS=ON
LLVM_ENABLE_RTTI=ON
LLVM_ENABLE_EH=ON
LLVM_INSTALL_UTILS=ON
LLVM_BUILD_TOOLS=ON
LLVM_BUILD_UTILS=ON
LLVM_BUILD_EXAMPLES=OFF
LLVM_ENABLE_LIBEDIT=OFF
MLIR_ENABLE_BINDINGS_PYTHON=ON
_GLIBCXX_USE_CXX11_ABI=0
```

该参数集用于保证后续 Autofuse MLIR C++ 目标能链接 MLIR 库，同时保持与当前 Autofuse 构建
ABI 约束一致。

## Manifest

LLVM 镜像必须内置：

```text
/opt/llvm/share/autofuse-mlir/llvm-image-manifest.json
```

manifest 至少记录：

```json
{
  "image": "autofuse-mlir-dev:<arch>",
  "base_image": "autofuse-mlir-dev-base:<arch>",
  "llvm_root": "/opt/llvm",
  "llvm_version": "<llvm-config --version>",
  "source_mode": "source or prebuilt",
  "source_ref": "<llvm-project path or prebuilt path>",
  "source_commit": "<llvm-project commit when available>",
  "required_tools": ["llvm-config", "mlir-opt", "mlir-tblgen", "llvm-lit"],
  "required_cmake_configs": ["LLVMConfig.cmake", "MLIRConfig.cmake"],
  "cxx_abi": "_GLIBCXX_USE_CXX11_ABI=0",
  "llvm_enable_rtti": "YES",
  "llvm_enable_eh": "ON",
  "llvm_enable_assertions": "ON",
  "mlir_enable_bindings_python": "ON",
  "llvm_config_cxxflags": "<llvm-config --cxxflags>",
  "build_jobs": 8
}
```

远端镜像仓提供 digest 时，应通过 `AF_MLIR_DEV_IMAGE_DIGEST` 写入环境 manifest。纯本地临时镜像
没有 RepoDigest 时，记录 image id 即可。

## 后续接入原则

- `ENABLE_AUTOFUSE_MLIR=OFF` 时不得发现或依赖 LLVM/MLIR。
- `ENABLE_AUTOFUSE_MLIR=ON` 后，CMake 只读取显式传入的 `LLVM_BUILD_DIR`、
  `LLVM_DIR` 或 `MLIR_DIR`。
- 找不到 LLVM/MLIR 时必须给出明确错误，不做自动下载或源码编译。
