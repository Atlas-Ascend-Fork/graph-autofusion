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
  bash mlir/docker/build_llvm_image.sh --base-only
  AF_LLVM_SOURCE_DIR=/path/to/llvm-project bash mlir/docker/build_llvm_image.sh
  LLVM_BUILD_DIR=/path/to/prebuilt/llvm bash mlir/docker/build_llvm_image.sh

Environment:
  AF_MLIR_DEV_IMAGE        Output combined dev+LLVM image, default autofuse-mlir-dev:<arch>
  AF_MLIR_LLVM_IMAGE       Backward-compatible output image alias
  AF_MLIR_DEV_BASE_IMAGE   Internal tool-only base image, default autofuse-mlir-dev-base:<arch>
  AF_MLIR_LLVM_BASE_IMAGE  Backward-compatible base image alias
  AF_MLIR_DEV_BASE_OS_IMAGE
                           OS base image used when building the internal base, default ubuntu:24.04
  AF_PYTHON_BUILD_REQUIREMENTS
                           Python packages installed in the internal base
  AF_PIP_INDEX_URL         Optional Python package index for the internal base
  AF_PIP_TRUSTED_HOST      Optional pip trusted host for the internal base
  AF_MLIR_BUILD_BASE_IF_MISSING
                           Build internal base image when missing, default 1 for the default base
  AF_LLVM_SOURCE_DIR       llvm-project source tree. Builds LLVM/MLIR in the base image.
  LLVM_BUILD_DIR           Existing prebuilt LLVM/MLIR directory to package.
  AF_LLVM_INSTALL_PREFIX   Path inside output image, default /opt/llvm
  AF_LLVM_ENABLE_ASSERTIONS
                           LLVM_ENABLE_ASSERTIONS value, default ON
  AF_LLVM_ENABLE_PYTHON_BINDINGS
                           MLIR_ENABLE_BINDINGS_PYTHON value, default ON
  AF_LLVM_USE_CLANG        Use clang/clang++ to build LLVM when available, default 1
  AF_LLVM_USE_CCACHE       Use ccache as the LLVM C/C++ compiler launcher, default 1
  AF_LLVM_CCACHE_DIR       Host ccache directory, default autofuse/mlir/.artifacts/ccache
  AF_LLVM_COPY_SOURCE      Copy llvm-project into a container-local path before building, default 1
  AF_LLVM_BUILD_JOBS       Parallel build jobs, default logical CPU count

Use --base-only to build only the internal tool-only base image.

The source build uses the logical CPU count by default. It follows the gsr LLVM baseline and adds Autofuse C++ compatibility:
  LLVM_ENABLE_PROJECTS=mlir
  LLVM_TARGETS_TO_BUILD=host
  LLVM_ENABLE_ASSERTIONS=ON
  LLVM_ENABLE_RTTI=ON
  LLVM_INSTALL_UTILS=ON
  LLVM_BUILD_EXAMPLES=OFF
  LLVM_ENABLE_LIBEDIT=OFF
  MLIR_ENABLE_BINDINGS_PYTHON=ON
  LLVM_ENABLE_EH=ON
  CMAKE_PROJECT_INCLUDE=<generated hook adding _GLIBCXX_USE_CXX11_ABI=0>
EOF
}

repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
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
image="${AF_MLIR_LLVM_IMAGE:-${AF_MLIR_DEV_IMAGE:-autofuse-mlir-dev:${image_arch}}}"
default_base="autofuse-mlir-dev-base:${image_arch}"
base="${AF_MLIR_LLVM_BASE_IMAGE:-${AF_MLIR_DEV_BASE_IMAGE:-${default_base}}}"
base_os="${AF_MLIR_DEV_BASE_OS_IMAGE:-${AF_MLIR_DEV_BASE_IMAGE_FROM:-ubuntu:24.04}}"
base_overridden=0
if [ -n "${AF_MLIR_LLVM_BASE_IMAGE:-}" ] || [ -n "${AF_MLIR_DEV_BASE_IMAGE:-}" ]; then
    base_overridden=1
fi
build_base_if_missing="${AF_MLIR_BUILD_BASE_IF_MISSING:-}"
if [ -z "${build_base_if_missing}" ]; then
    if [ "${base_overridden}" = "0" ]; then
        build_base_if_missing=1
    else
        build_base_if_missing=0
    fi
fi
llvm_source_dir="${AF_LLVM_SOURCE_DIR:-}"
llvm_root="${LLVM_BUILD_DIR:-}"
llvm_install_prefix="${AF_LLVM_INSTALL_PREFIX:-/opt/llvm}"
llvm_enable_assertions="${AF_LLVM_ENABLE_ASSERTIONS:-ON}"
llvm_enable_python_bindings="${AF_LLVM_ENABLE_PYTHON_BINDINGS:-ON}"
llvm_use_clang="${AF_LLVM_USE_CLANG:-1}"
llvm_use_ccache="${AF_LLVM_USE_CCACHE:-1}"
llvm_copy_source="${AF_LLVM_COPY_SOURCE:-1}"
llvm_ccache_dir="${AF_LLVM_CCACHE_DIR:-${repo_root}/autofuse/mlir/.artifacts/ccache}"
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

build_jobs="${AF_LLVM_BUILD_JOBS:-$(detect_cpu_count)}"
default_python_build_requirements="setuptools>=68,<80 wheel>=0.42 build>=1.1.1 numpy pybind11==2.13.1 nanobind PyYAML pygments pytest coverage pytest-cov pytest-timeout pytest-xdist"
build_base_only=0

case "${1:-}" in
    -h | --help)
        usage
        exit 0
        ;;
    --base-only)
        build_base_only=1
        shift
        ;;
    "")
        ;;
    *)
        echo "ERROR: unknown argument: ${1}" >&2
        usage >&2
        exit 2
        ;;
esac
if [ "$#" -ne 0 ]; then
    echo "ERROR: unexpected argument: ${1}" >&2
    usage >&2
    exit 2
fi

if [ "${build_base_only}" = "0" ]; then
    if [ -n "${llvm_source_dir}" ] && [ -n "${llvm_root}" ]; then
        echo "ERROR: set only one of AF_LLVM_SOURCE_DIR or LLVM_BUILD_DIR" >&2
        exit 2
    fi
    if [ -z "${llvm_source_dir}" ] && [ -z "${llvm_root}" ]; then
        usage >&2
        exit 2
    fi
    if [[ "${llvm_install_prefix}" != /* ]]; then
        echo "ERROR: AF_LLVM_INSTALL_PREFIX must be an absolute path: ${llvm_install_prefix}" >&2
        exit 2
    fi
fi

json_string() {
    python3 -c 'import json, sys; print(json.dumps(sys.stdin.read().rstrip("\n")))' <<< "${1:-}"
}

build_base_image() {
    local build_args=(
        --build-arg BASE_IMAGE="${base_os}"
        --build-arg PYTHON_BUILD_REQUIREMENTS="${AF_PYTHON_BUILD_REQUIREMENTS:-${default_python_build_requirements}}"
    )
    if [ -n "${AF_PIP_INDEX_URL:-}" ]; then
        build_args+=(--build-arg PIP_INDEX_URL="${AF_PIP_INDEX_URL}")
    fi
    if [ -n "${AF_PIP_TRUSTED_HOST:-}" ]; then
        build_args+=(--build-arg PIP_TRUSTED_HOST="${AF_PIP_TRUSTED_HOST}")
    fi

    docker build \
        --target dev-base \
        "${build_args[@]}" \
        -t "${base}" \
        -f "${repo_root}/autofuse/mlir/docker/Dockerfile.llvm" \
        "${repo_root}"
}

ensure_base_image() {
    if docker image inspect "${base}" >/dev/null 2>&1; then
        return
    fi
    if [ "${build_base_if_missing}" != "1" ]; then
        echo "ERROR: base image does not exist: ${base}" >&2
        echo "Set AF_MLIR_BUILD_BASE_IF_MISSING=1 to build it from ${base_os}, or run build_llvm_image.sh --base-only first." >&2
        exit 1
    fi
    build_base_image
}

require_base_without_cann() {
    docker run --rm "${base}" bash -lc '
set -euo pipefail
for path in /opt/cann /usr/local/Ascend/ascend-toolkit; do
    if [ -e "${path}" ]; then
        echo "ERROR: base image already contains CANN path: ${path}" >&2
        exit 1
    fi
done
'
}

validate_llvm_root_in_container() {
    local root="$1"
    docker run --rm \
        -e AF_LLVM_ENABLE_ASSERTIONS="${llvm_enable_assertions}" \
        -v "${root}:/opt/llvm-candidate:ro" \
        "${base}" \
        bash -lc '
set -euo pipefail
root=/opt/llvm-candidate
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
"${root}/bin/llvm-config" --version
"${root}/bin/mlir-opt" --version | head -n 1
"${root}/bin/mlir-tblgen" --version | head -n 1
"${root}/bin/llvm-lit" --version | head -n 1
'
}

build_llvm_from_source() {
    local source_dir="$1"
    local output_dir="$2"
    local docker_run_args=(
        --rm
        --ulimit nofile=65536:65536
        -v "${source_dir}:/src/llvm-project:ro"
        -v "${output_dir}:/out/llvm"
    )

    if [ ! -d "${source_dir}/llvm" ] || [ ! -d "${source_dir}/mlir" ]; then
        echo "ERROR: AF_LLVM_SOURCE_DIR must point to an llvm-project checkout containing llvm/ and mlir/: ${source_dir}" >&2
        exit 2
    fi

    rm -rf "${output_dir}"
    mkdir -p "${output_dir}"
    chmod 0777 "${output_dir}"

    if [ "${llvm_use_ccache}" = "1" ]; then
        mkdir -p "${llvm_ccache_dir}"
        chmod 0777 "${llvm_ccache_dir}"
        docker_run_args+=(-v "${llvm_ccache_dir}:/ccache")
    fi

    docker run "${docker_run_args[@]}" \
        "${base}" \
        bash -lc "
set -euo pipefail
touch /out/llvm/.write-probe
rm -f /out/llvm/.write-probe
if [ \"${llvm_use_clang}\" = \"1\" ] && command -v clang >/dev/null 2>&1 && command -v clang++ >/dev/null 2>&1; then
    export CC=clang
    export CXX=clang++
fi
cmake_launcher_args=()
if [ \"${llvm_use_ccache}\" = \"1\" ] && command -v ccache >/dev/null 2>&1; then
    export CCACHE_DIR=/ccache
    ccache --zero-stats >/dev/null 2>&1 || true
    cmake_launcher_args=(-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache)
fi
cat > /tmp/autofuse-llvm-abi.cmake <<'CMAKE'
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
CMAKE
llvm_project_dir=/src/llvm-project
if [ \"${llvm_copy_source}\" = \"1\" ]; then
    llvm_project_dir=/tmp/llvm-project-local
    rm -rf \"\${llvm_project_dir}\"
    mkdir -p \"\${llvm_project_dir}\"
    rsync -a --delete --exclude .git /src/llvm-project/ \"\${llvm_project_dir}/\"
fi
rm -rf /tmp/llvm-build
cmake -S \"\${llvm_project_dir}/llvm\" -B /tmp/llvm-build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/out/llvm \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_ENABLE_ASSERTIONS=${llvm_enable_assertions} \
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
    -DMLIR_ENABLE_BINDINGS_PYTHON=${llvm_enable_python_bindings} \
    -DPython3_EXECUTABLE=\"\$(command -v python3)\" \
    -DCMAKE_PROJECT_INCLUDE=/tmp/autofuse-llvm-abi.cmake \
    \"\${cmake_launcher_args[@]}\"
cmake --build /tmp/llvm-build --target llvm-config mlir-opt mlir-tblgen install -j ${build_jobs}
if [ \"${llvm_use_ccache}\" = \"1\" ] && command -v ccache >/dev/null 2>&1; then
    ccache --show-stats || true
fi
mkdir -p /out/llvm/share/autofuse-mlir
cp /tmp/llvm-build/CMakeCache.txt /out/llvm/share/autofuse-mlir/llvm-cmake-cache.txt
mkdir -p /out/llvm/share/autofuse-mlir/lit
cp -a \"\${llvm_project_dir}/llvm/utils/lit/lit\" /out/llvm/share/autofuse-mlir/lit/
cat > /out/llvm/bin/llvm-lit <<'PY'
#!/usr/bin/env python3
import os
import sys

llvm_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(llvm_root, 'share', 'autofuse-mlir', 'lit'))
from lit.main import main

main()
PY
chmod +x /out/llvm/bin/llvm-lit
"
}

create_manifest() {
    local root="$1"
    local manifest="$2"
    local mode="$3"
    local source_ref="$4"
    local probe_file="${tmpdir}/llvm-manifest-probe.txt"
    local source_commit=""
    local llvm_version=""
    local cxxflags=""
    local has_rtti=""

    if [ "${mode}" = "source" ] && git -C "${source_ref}" rev-parse --short HEAD >/dev/null 2>&1; then
        source_commit="$(git -C "${source_ref}" rev-parse HEAD)"
    fi

    docker run --rm \
        -v "${root}:/opt/llvm-candidate:ro" \
        "${base}" \
        bash -lc '
set -euo pipefail
root=/opt/llvm-candidate
printf "%s\n" "$("${root}/bin/llvm-config" --version)"
printf "%s\n" "$("${root}/bin/llvm-config" --cxxflags)"
printf "%s\n" "$("${root}/bin/llvm-config" --has-rtti 2>/dev/null || true)"
' > "${probe_file}"
    llvm_version="$(sed -n '1p' "${probe_file}")"
    cxxflags="$(sed -n '2p' "${probe_file}")"
    has_rtti="$(sed -n '3p' "${probe_file}")"

    cat > "${manifest}" <<EOF
{
  "image": $(json_string "${image}"),
  "base_image": $(json_string "${base}"),
  "llvm_root": $(json_string "${llvm_install_prefix}"),
  "llvm_version": $(json_string "${llvm_version}"),
  "source_mode": $(json_string "${mode}"),
  "source_ref": $(json_string "${source_ref}"),
  "source_commit": $(json_string "${source_commit}"),
  "required_tools": ["llvm-config", "mlir-opt", "mlir-tblgen", "llvm-lit"],
  "required_cmake_configs": ["LLVMConfig.cmake", "MLIRConfig.cmake"],
  "cxx_abi": $(json_string "_GLIBCXX_USE_CXX11_ABI=0"),
  "llvm_enable_rtti": $(json_string "${has_rtti}"),
  "llvm_enable_eh": $(json_string "ON"),
  "llvm_enable_assertions": $(json_string "${llvm_enable_assertions}"),
  "mlir_enable_bindings_python": $(json_string "${llvm_enable_python_bindings}"),
  "llvm_enable_libedit": $(json_string "OFF"),
  "llvm_build_examples": $(json_string "OFF"),
  "llvm_config_cxxflags": $(json_string "${cxxflags}"),
  "build_jobs": ${build_jobs}
}
EOF
}

if [ "${build_base_only}" = "1" ]; then
    build_base_image
    require_base_without_cann
    echo "${base}"
    exit 0
fi
ensure_base_image
require_base_without_cann

mkdir -p "${repo_root}/autofuse/mlir/.artifacts"
tmpdir="$(mktemp -d "${repo_root}/autofuse/mlir/.artifacts/llvm-image.XXXXXX")"
cleanup() {
    rm -rf "${tmpdir}"
}
trap cleanup EXIT

if [ -n "${llvm_source_dir}" ]; then
    llvm_root="${tmpdir}/llvm"
    build_llvm_from_source "${llvm_source_dir}" "${llvm_root}"
    manifest_mode=source
    manifest_source="${llvm_source_dir}"
else
    if [ ! -d "${llvm_root}" ]; then
        echo "ERROR: LLVM_BUILD_DIR does not exist or is not a directory: ${llvm_root}" >&2
        exit 2
    fi
    manifest_mode=prebuilt
    manifest_source="${llvm_root}"
fi

validate_llvm_root_in_container "${llvm_root}"

mkdir -p "${tmpdir}/context/llvm/share/autofuse-mlir"
rsync -a --delete "${llvm_root}/" "${tmpdir}/context/llvm/"
create_manifest "${llvm_root}" "${tmpdir}/context/llvm/share/autofuse-mlir/llvm-image-manifest.json" \
    "${manifest_mode}" "${manifest_source}"

DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}" docker build \
    --target dev-llvm \
    --build-arg BASE_IMAGE="${base}" \
    --build-arg LLVM_INSTALL_PREFIX="${llvm_install_prefix}" \
    -t "${image}" \
    -f "${repo_root}/autofuse/mlir/docker/Dockerfile.llvm" \
    "${tmpdir}/context"

bash "${repo_root}/autofuse/mlir/docker/validate_llvm_image.sh" "${image}"
echo "${image}"
