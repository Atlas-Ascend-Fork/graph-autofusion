# CLion Docker Toolchain 配置指南

本文档面向用 CLion + Docker Toolchain 做 MLIR 开发的开发者，典型场景是 macOS/Windows 等非 Linux
开发机。前置条件是已按 [../../README.md](../../README.md) 的 Docker 方案拉好 LLVM 镜像、准备好
CANN volume。

> 注意：Linux 本机/远端开发直接用 CLion 的 Local Toolchain 或 Remote Toolchain 配置本机路径即可，不需要看本文档。

以下用 `<SOURCE_CODE_ROOT>` 表示本机 `graph-autofusion` 仓库根目录。

## 确认 Docker 服务可用

`Settings > Build, Execution, Deployment > Docker` 里确认 Docker server 显示 `Connection successful`。

> 注意：源码在本机的 `/Volumes` 目录下的话，Docker server 的 path mappings 要加上：`/Volumes -> /Volumes`
>
> 说明：Docker 中的映射路径通常如下（无需设置）：
>
> ```text
> CLion project:   <SOURCE_CODE_ROOT>
> Repo mount:      <SOURCE_CODE_ROOT> -> /tmp/graph-autofusion
> Component path:  /tmp/graph-autofusion
> ```

## Docker Toolchain 配置

```text
Settings:  Build, Execution, Deployment > Toolchains
Name:      Docker
Server:    <Docker: Auto detected server>
Image:     autofuse-mlir-dev:arm64（AArch64）或 autofuse-mlir-dev:amd64（x86-64）
```

Docker Toolchain 的 `Container settings`：

```text
--entrypoint=
--rm
-v cann-volume:/opt/cann:ro
-e ASCEND_HOME_PATH=/opt/cann
-e LLVM_BUILD_DIR=/opt/llvm
```

## CMake Profile

```text
Settings:         Build, Execution, Deployment > CMake
Name:             Debug-Docker
Toolchain:        Docker
Build type:       Debug
Generator:        Unix Makefiles
Build directory:  build-clion
Build options:    -j 8
CMake options:    -DENABLE_AUTOFUSE_MLIR=ON
                  -DENABLE_AUTOFUSE_MLIR_TESTS=ON
                  -DASCEND_INSTALL_PATH=/opt/cann
                  -DASCEND_3RD_LIB_PATH=/tmp/graph-autofusion/output/third_party
                  -DCANN_3RD_LIB_PATH=/tmp/graph-autofusion/output/third_party
                  -DENABLE_OPEN_SRC=True
                  -DENABLE_SYMENGINE=True
                  -DCMAKE_C_COMPILER=gcc
                  -DCMAKE_CXX_COMPILER=g++
                  -DLLVM_BUILD_DIR=/opt/llvm
                  -DLLVM_DIR=/opt/llvm/lib/cmake/llvm
                  -DMLIR_DIR=/opt/llvm/lib/cmake/mlir
```
> 注意：主工程目前用 `Unix Makefiles`，**不能用 Ninja**，LLVM/MLIR 镜像内部预编译还是用 Ninja 的。
