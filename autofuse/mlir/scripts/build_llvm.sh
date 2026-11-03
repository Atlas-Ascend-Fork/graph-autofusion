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
  bash mlir/scripts/build_llvm.sh /path/to/llvm-project

Clones llvm-project at AF_LLVM_TAG and builds LLVM/MLIR with the flags required by
Autofuse (ABI-compatible with the CANN Toolkit, RTTI/EH enabled). Intended for
local/remote Linux compilation (see README.md 方案二: 本地编译). Use
mlir/docker/build_llvm_image.sh instead to produce the team's dev+LLVM image.

Environment:
  AF_LLVM_TAG            llvm-project tag to clone, default llvmorg-21.1.8
  AF_LLVM_SOURCE_DIR     Existing llvm-project checkout to build instead of cloning
  AF_LLVM_INSTALL_PREFIX Install prefix, default /opt/llvm
  AF_LLVM_BUILD_DIR      CMake build directory, default <llvm-project>/../build-llvm
  AF_LLVM_BUILD_JOBS     Parallel build jobs, default logical CPU count
  AF_LLVM_ENABLE_ASSERTIONS
                          LLVM_ENABLE_ASSERTIONS value, default ON
  AF_LLVM_ENABLE_PYTHON_BINDINGS
                          MLIR_ENABLE_BINDINGS_PYTHON value, default ON
  AF_LLVM_USE_CLANG      Use clang/clang++ to build LLVM when available, default 1
  AF_LLVM_USE_CCACHE     Use ccache as the LLVM C/C++ compiler launcher, default 1
  AF_LLVM_CCACHE_DIR     ccache directory, default autofuse/mlir/.artifacts/ccache

On Debian/Ubuntu, missing build tools are installed automatically with apt-get.

When AF_LLVM_SOURCE_DIR is unset, the positional path is used as the clone destination;
an existing llvm-project checkout at that path is reused automatically.
EOF
}

run_as_root() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        echo "ERROR: missing build tools and neither root nor sudo is available for installation" >&2
        return 1
    fi
}

ensure_build_dependencies() {
    local required_commands=(cmake ninja gcc g++ python3 python3-config git)
    local missing_commands=()
    local command_name
    for command_name in "${required_commands[@]}"; do
        if ! command -v "${command_name}" >/dev/null 2>&1; then
            missing_commands+=("${command_name}")
        fi
    done

    if [ "${#missing_commands[@]}" -eq 0 ]; then
        return
    fi

    if ! command -v apt-get >/dev/null 2>&1; then
        echo "ERROR: missing required commands: ${missing_commands[*]}" >&2
        echo "Install cmake, ninja, gcc/g++, python3, python3-dev, and git, then retry." >&2
        return 1
    fi

    echo "Missing build commands: ${missing_commands[*]}"
    echo "Installing Debian/Ubuntu dependencies..."
    run_as_root apt-get update
    run_as_root apt-get install -y --no-install-recommends \
        build-essential \
        ninja-build \
        cmake \
        python3 \
        python3-dev \
        python3-pip \
        git

    for command_name in "${required_commands[@]}"; do
        if ! command -v "${command_name}" >/dev/null 2>&1; then
            echo "ERROR: required command is still missing after installation: ${command_name}" >&2
            return 1
        fi
    done
}

detect_cpu_count() {
    local cpu_count=""
    if command -v nproc >/dev/null 2>&1; then
        cpu_count="$(nproc)"
    elif command -v getconf >/dev/null 2>&1; then
        cpu_count="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
    elif command -v sysctl >/dev/null 2>&1; then
        cpu_count="$(sysctl -n hw.logicalcpu 2>/dev/null || true)"
    fi
    if [[ "${cpu_count}" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s\n' "${cpu_count}"
    else
        printf '%s\n' 1
    fi
}

ensure_nanobind() {
    if ! python3 -m pip --version >/dev/null 2>&1; then
        if ! command -v apt-get >/dev/null 2>&1; then
            echo "ERROR: nanobind >= 2.4 is missing and python3-pip/apt-get is unavailable" >&2
            return 1
        fi
        echo "python3-pip is missing; installing it before nanobind..."
        run_as_root apt-get update
        run_as_root apt-get install -y --no-install-recommends python3-pip
    fi

    echo "Installing Python MLIR binding dependencies..."
    local pip_err
    pip_err="$(mktemp -t autofuse-nanobind-pip.XXXXXX.err)"
    if ! python3 -m pip install --upgrade "pybind11==2.13.1" "nanobind>=2.4,<3" 2>"${pip_err}"; then
        if grep -q "externally-managed-environment" "${pip_err}"; then
            python3 -m pip install --upgrade --break-system-packages "pybind11==2.13.1" "nanobind>=2.4,<3"
        else
            cat "${pip_err}" >&2
            rm -f "${pip_err}"
            return 1
        fi
    fi
    rm -f "${pip_err}"
}

install_prefix_needs_root() {
    local prefix="$1"
    if [ -e "${prefix}" ]; then
        find "${prefix}" ! -writable -print -quit 2>/dev/null | grep -q .
        return
    fi
    local parent
    parent="$(dirname "${prefix}")"
    [ ! -d "${parent}" ] || [ ! -w "${parent}" ]
}

llvm_source_dir_arg=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        -h | --help)
            usage
            exit 0
            ;;
        -*)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if [ -n "${llvm_source_dir_arg}" ]; then
                echo "ERROR: only one LLVM source path may be provided" >&2
                usage >&2
                exit 2
            fi
            llvm_source_dir_arg="$1"
            shift
            ;;
    esac
done

af_llvm_tag="${AF_LLVM_TAG:-llvmorg-21.1.8}"
install_prefix="${AF_LLVM_INSTALL_PREFIX:-/opt/llvm}"
build_jobs="${AF_LLVM_BUILD_JOBS:-$(detect_cpu_count)}"
llvm_enable_assertions="${AF_LLVM_ENABLE_ASSERTIONS:-ON}"
llvm_enable_python_bindings="${AF_LLVM_ENABLE_PYTHON_BINDINGS:-ON}"
use_clang="${AF_LLVM_USE_CLANG:-1}"
use_ccache="${AF_LLVM_USE_CCACHE:-1}"
repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
ccache_dir="${AF_LLVM_CCACHE_DIR:-${repo_root}/autofuse/mlir/.artifacts/ccache}"

ensure_build_dependencies
ensure_nanobind

install_needs_root=0
if install_prefix_needs_root "${install_prefix}"; then
    install_needs_root=1
    if [ "$(id -u)" -ne 0 ] && ! command -v sudo >/dev/null 2>&1; then
        echo "ERROR: install prefix requires elevated privileges and neither root nor sudo is available: ${install_prefix}" >&2
        exit 1
    fi
    echo "Install prefix requires elevated privileges; will use sudo for the install step: ${install_prefix}"
fi

if [ -n "${AF_LLVM_SOURCE_DIR:-}" ]; then
    if [ -n "${llvm_source_dir_arg}" ]; then
        if [ "${llvm_source_dir_arg}" != "${AF_LLVM_SOURCE_DIR}" ]; then
            echo "ERROR: do not provide different positional and AF_LLVM_SOURCE_DIR paths" >&2
            exit 2
        fi
    fi
    llvm_source_dir="${AF_LLVM_SOURCE_DIR}"
    if [ ! -d "${llvm_source_dir}/llvm" ]; then
        echo "ERROR: AF_LLVM_SOURCE_DIR does not look like an llvm-project checkout: ${llvm_source_dir}" >&2
        exit 2
    fi
else
    if [ -z "${llvm_source_dir_arg}" ]; then
        echo "ERROR: provide the LLVM source path, or set AF_LLVM_SOURCE_DIR" >&2
        usage >&2
        exit 2
    fi
    llvm_source_dir="${llvm_source_dir_arg}"
    if [ -d "${llvm_source_dir}" ]; then
        if [ ! -d "${llvm_source_dir}/llvm" ]; then
            echo "ERROR: ${llvm_source_dir} exists but does not look like an llvm-project checkout" >&2
            exit 2
        fi
        echo "Using existing LLVM checkout: ${llvm_source_dir}"
    elif [ -e "${llvm_source_dir}" ]; then
        echo "ERROR: ${llvm_source_dir} exists but is not a directory" >&2
        exit 2
    else
        git clone --branch "${af_llvm_tag}" --depth 1 https://github.com/llvm/llvm-project.git "${llvm_source_dir}"
    fi
fi

build_dir="${AF_LLVM_BUILD_DIR:-${llvm_source_dir}/../build-llvm}"
abi_hook="$(mktemp -t autofuse-llvm-abi.XXXXXX.cmake)"
lit_wrapper="$(mktemp -t autofuse-llvm-lit.XXXXXX)"
cleanup() {
    rm -f "${abi_hook}" "${lit_wrapper}"
}
trap cleanup EXIT
cat > "${abi_hook}" <<'EOF'
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
EOF

if [ "${use_clang}" = "1" ] && command -v clang >/dev/null 2>&1 && command -v clang++ >/dev/null 2>&1; then
    export CC=clang
    export CXX=clang++
fi

cmake_launcher_args=()
if [ "${use_ccache}" = "1" ] && command -v ccache >/dev/null 2>&1; then
    export CCACHE_DIR="${ccache_dir}"
    ccache --zero-stats >/dev/null 2>&1 || true
    cmake_launcher_args=(-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache)
fi

cmake -S "${llvm_source_dir}/llvm" -B "${build_dir}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${install_prefix}" \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_ENABLE_ASSERTIONS="${llvm_enable_assertions}" \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_ENABLE_EH=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_BUILD_TOOLS=ON \
    -DLLVM_BUILD_UTILS=ON \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_ENABLE_LIBEDIT=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DMLIR_INCLUDE_TESTS=OFF \
    -DMLIR_INCLUDE_INTEGRATION_TESTS=OFF \
    -DMLIR_ENABLE_BINDINGS_PYTHON="${llvm_enable_python_bindings}" \
    -DPython3_EXECUTABLE="$(command -v python3)" \
    -DCMAKE_PROJECT_INCLUDE="${abi_hook}" \
    "${cmake_launcher_args[@]}"
cmake --build "${build_dir}" --target llvm-config mlir-opt mlir-tblgen -j "${build_jobs}"
if [ "${install_needs_root}" = "1" ]; then
    run_as_root cmake --build "${build_dir}" --target install
else
    cmake --build "${build_dir}" --target install
fi

lit_source_dir="${llvm_source_dir}/llvm/utils/lit/lit"
if [ ! -d "${lit_source_dir}" ]; then
    echo "ERROR: LLVM lit sources are missing: ${lit_source_dir}" >&2
    exit 1
fi

lit_install_dir="${install_prefix}/share/autofuse-mlir/lit"
cat > "${lit_wrapper}" <<'EOF'
#!/usr/bin/env python3
import os
import sys

llvm_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(llvm_root, "share", "autofuse-mlir", "lit"))
from lit.main import main

main()
EOF
if [ "${install_needs_root}" = "1" ]; then
    run_as_root install -d "${lit_install_dir}"
    run_as_root cp -a "${lit_source_dir}" "${lit_install_dir}/"
    run_as_root install -m 0755 "${lit_wrapper}" "${install_prefix}/bin/llvm-lit"
else
    mkdir -p "${lit_install_dir}"
    cp -a "${lit_source_dir}" "${lit_install_dir}/"
    install -m 0755 "${lit_wrapper}" "${install_prefix}/bin/llvm-lit"
fi

if [ "${use_ccache}" = "1" ] && command -v ccache >/dev/null 2>&1; then
    ccache --show-stats || true
fi

echo "Installed LLVM/MLIR ${af_llvm_tag} to ${install_prefix}"
"${install_prefix}/bin/llvm-config" --version
"${install_prefix}/bin/llvm-lit" --version | head -n 1
