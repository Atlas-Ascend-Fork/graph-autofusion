#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash mlir/docker/validate_llvm_image.sh <dev-image>

Environment:
  AF_LLVM_INSTALL_PREFIX  LLVM path inside image, default /opt/llvm
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

host_arch="$(uname -m)"
case "${host_arch}" in
    arm64 | aarch64)
        image_arch="arm64"
        ;;
    amd64 | x86_64)
        image_arch="amd64"
        ;;
    *)
        echo "ERROR: unsupported CPU architecture: ${host_arch}" >&2
        exit 2
        ;;
esac
image="${1:-${AF_MLIR_LLVM_IMAGE:-${AF_MLIR_DEV_IMAGE:-autofuse-mlir-dev:${image_arch}}}}"
llvm_root="${AF_LLVM_INSTALL_PREFIX:-/opt/llvm}"
llvm_enable_assertions="${AF_LLVM_ENABLE_ASSERTIONS:-ON}"

docker image inspect "${image}" >/dev/null

docker run --rm \
    -e AF_LLVM_ENABLE_ASSERTIONS="${llvm_enable_assertions}" \
    "${image}" \
    bash -lc '
set -euo pipefail
root="$1"
for tool in llvm-config mlir-opt mlir-tblgen llvm-lit; do
    if [ ! -x "${root}/bin/${tool}" ]; then
        echo "ERROR: missing executable ${root}/bin/${tool}" >&2
        exit 1
    fi
done
if [ ! -f "${root}/lib/cmake/llvm/LLVMConfig.cmake" ]; then
    echo "ERROR: missing ${root}/lib/cmake/llvm/LLVMConfig.cmake" >&2
    exit 1
fi
if [ ! -f "${root}/lib/cmake/mlir/MLIRConfig.cmake" ]; then
    echo "ERROR: missing ${root}/lib/cmake/mlir/MLIRConfig.cmake" >&2
    exit 1
fi
cxxflags="$("${root}/bin/llvm-config" --cxxflags)"
case "${cxxflags}" in
    *-D_GLIBCXX_USE_CXX11_ABI=0*)
        ;;
    *)
        echo "ERROR: llvm-config --cxxflags does not contain -D_GLIBCXX_USE_CXX11_ABI=0" >&2
        echo "${cxxflags}" >&2
        exit 1
        ;;
esac
case "${cxxflags}" in
    *-fno-exceptions*)
        echo "ERROR: llvm-config --cxxflags contains -fno-exceptions; expected LLVM_ENABLE_EH=ON" >&2
        echo "${cxxflags}" >&2
        exit 1
        ;;
esac
has_rtti="$("${root}/bin/llvm-config" --has-rtti 2>/dev/null || true)"
case "${has_rtti}" in
    YES | yes | ON | on | TRUE | true | 1)
        ;;
    *)
        echo "ERROR: llvm-config --has-rtti is not enabled: ${has_rtti}" >&2
        exit 1
        ;;
esac
cache="${root}/share/autofuse-mlir/llvm-cmake-cache.txt"
if [ -f "${cache}" ]; then
    grep -q "^LLVM_ENABLE_RTTI:BOOL=ON$" "${cache}"
    grep -q "^LLVM_ENABLE_EH:BOOL=ON$" "${cache}"
    grep -q "^LLVM_ENABLE_ASSERTIONS:BOOL=${AF_LLVM_ENABLE_ASSERTIONS}$" "${cache}"
fi
if [ -e /opt/cann ] || [ -e /usr/local/Ascend/ascend-toolkit ]; then
    echo "ERROR: LLVM image must not contain CANN Toolkit paths" >&2
    exit 1
fi
for tool in \
    gcc g++ clang ld.lld make cmake ninja ccache pkg-config \
    clang-format pre-commit \
    python3 git git-lfs ssh curl wget rsync \
    gdb lldb strace lsof ps killall file patchelf time \
    vim less tree jq rg tar zip unzip xz zstd \
    patch autoconf automake libtoolize gperf \
    lcov genhtml gcov; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "ERROR: missing development tool: ${tool}" >&2
        exit 1
    fi
done
read -r _ _ clang_format_version _ <<< "$(clang-format --version)"
if [ "${clang_format_version}" != "18.1.8" ]; then
    echo "ERROR: clang-format version ${clang_format_version} != 18.1.8" >&2
    exit 1
fi
read -r _ pre_commit_version <<< "$(pre-commit --version)"
PRE_COMMIT_VERSION="${pre_commit_version}" python3 - <<'"'"'PY'"'"'
import os

raw_version = os.environ["PRE_COMMIT_VERSION"]
version = tuple(int(part) for part in raw_version.split("."))
if not (version >= (4, 0, 0) and version < (5, 0, 0)):
    raise SystemExit(f"pre-commit version {raw_version} is not in [4.0.0, 5.0.0)")
PY
python3 - <<'"'"'PY'"'"'
import importlib
import pybind11

expected_pybind11 = "2.13.1"
if pybind11.__version__ != expected_pybind11:
    raise SystemExit(f"pybind11 version {pybind11.__version__} != {expected_pybind11}")

for module in ["setuptools", "wheel", "build", "numpy", "nanobind", "yaml", "pygments", "pytest", "coverage"]:
    importlib.import_module(module)
PY
echo "llvm_version=$("${root}/bin/llvm-config" --version)"
echo "clang_format_version=${clang_format_version}"
echo "pre_commit_version=${pre_commit_version}"
echo "llvm_has_rtti=${has_rtti}"
echo "llvm_cxxflags=${cxxflags}"
"${root}/bin/mlir-opt" --version | head -n 1
"${root}/bin/mlir-tblgen" --version | head -n 1
"${root}/bin/llvm-lit" --version | head -n 1
test -f "${root}/share/autofuse-mlir/llvm-image-manifest.json"
' _ "${llvm_root}"

docker image inspect "${image}" --format 'image_id={{.Id}} size={{.Size}}'
