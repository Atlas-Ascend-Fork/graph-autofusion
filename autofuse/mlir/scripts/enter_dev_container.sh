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

repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"
host_arch="$(uname -m)"
case "${host_arch}" in
    aarch64 | arm64)
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
image="${AF_MLIR_DEV_IMAGE:-autofuse-mlir-dev:${image_arch}}"
cann_root="${AF_CANN_ROOT:-}"
cann_volume="${AF_CANN_VOLUME:-}"
llvm_root="${LLVM_BUILD_DIR:-}"

mount_args=(-v "${repo_root}:/workspace" -w /workspace)
env_args=(-e AF_MLIR_DEV_IMAGE="${image}")

if [ -n "${AUTOFUSE_DFX_FLAGS:-}" ]; then
    env_args+=(-e AUTOFUSE_DFX_FLAGS="${AUTOFUSE_DFX_FLAGS}")
fi

if [ -n "${AF_MLIR_DEV_IMAGE_DIGEST:-}" ]; then
    env_args+=(-e AF_MLIR_DEV_IMAGE_DIGEST="${AF_MLIR_DEV_IMAGE_DIGEST}")
fi

if [ -n "${llvm_root}" ]; then
    if [ ! -d "${llvm_root}" ]; then
        echo "ERROR: LLVM_BUILD_DIR does not exist or is not a directory: ${llvm_root}" >&2
        exit 1
    fi
    mount_args+=(-v "${llvm_root}:/opt/llvm:ro")
fi
env_args+=(
    -e LLVM_BUILD_DIR=/opt/llvm
    -e LLVM_DIR=/opt/llvm/lib/cmake/llvm
    -e MLIR_DIR=/opt/llvm/lib/cmake/mlir
)

cann_mount=/opt/cann
cann_install_prefix=""

detect_cann_install_prefix_from_root() {
    local root="$1"
    local prefix="${AF_CANN_INSTALL_PREFIX:-}"
    if [ -n "${prefix}" ]; then
        echo "${prefix}"
        return
    fi
    if [ -f "${root}/set_env.sh" ]; then
        prefix="$(sed -n 's/^[[:space:]]*version_dirpath="\([^"]*\)".*/\1/p' "${root}/set_env.sh" | head -n 1)"
        if [ -n "${prefix}" ]; then
            echo "${prefix}"
            return
        fi
    fi
    echo "ERROR: cannot detect CANN install prefix from ${root}/set_env.sh; set AF_CANN_INSTALL_PREFIX explicitly" >&2
    return 1
}

detect_cann_install_prefix_from_volume() {
    local volume="$1"
    local prefix=""
    if prefix="$(docker run --rm -v "${volume}:${cann_mount}:ro" "${image}" \
        bash -lc 'sed -n '\''s/^[[:space:]]*version_dirpath="\([^"]*\)".*/\1/p'\'' /opt/cann/set_env.sh | head -n 1' 2>/dev/null)"; then
        if [[ "${prefix}" == /* ]]; then
            echo "${prefix}"
            return
        fi
    fi
    echo "ERROR: cannot detect CANN install prefix from volume ${volume}; set AF_CANN_INSTALL_PREFIX explicitly" >&2
    return 1
}

mount_cann() {
    local source="$1"
    local install_prefix="$2"
    mount_args+=(-v "${source}:${cann_mount}:ro")
    if [ "${install_prefix}" != "${cann_mount}" ]; then
        mount_args+=(-v "${source}:${install_prefix}:ro")
    fi
}

if [ -n "${cann_root}" ] && [ -n "${cann_volume}" ]; then
    echo "ERROR: set only one of AF_CANN_ROOT or AF_CANN_VOLUME" >&2
    exit 2
fi

cann_source=""
if [ -n "${cann_root}" ]; then
    if [ ! -d "${cann_root}" ]; then
        echo "ERROR: AF_CANN_ROOT does not exist or is not a directory: ${cann_root}" >&2
        exit 1
    fi
    cann_source="${cann_root}"
    cann_install_prefix="$(detect_cann_install_prefix_from_root "${cann_root}")"
elif [ -n "${cann_volume}" ]; then
    if ! docker volume inspect "${cann_volume}" >/dev/null 2>&1; then
        echo "ERROR: AF_CANN_VOLUME does not exist: ${cann_volume}" >&2
        exit 1
    fi
    cann_source="${cann_volume}"
    if [ -n "${AF_CANN_INSTALL_PREFIX:-}" ]; then
        cann_install_prefix="${AF_CANN_INSTALL_PREFIX}"
    else
        cann_install_prefix="$(detect_cann_install_prefix_from_volume "${cann_volume}")"
    fi
fi

if [ -n "${cann_install_prefix}" ]; then
    if [[ "${cann_install_prefix}" != /* ]]; then
        echo "ERROR: AF_CANN_INSTALL_PREFIX must be an absolute container path: ${cann_install_prefix}" >&2
        exit 1
    fi
    mount_cann "${cann_source}" "${cann_install_prefix}"
    env_args+=(-e CANN_ROOT="${cann_mount}" -e ASCEND_HOME_PATH="${cann_install_prefix}" -e ASCEND_CUSTOM_PATH="${cann_install_prefix}")
fi

if [ "$#" -gt 0 ] && [ "$1" = "--" ]; then
    shift
fi

docker_args=(--rm "${mount_args[@]}" "${env_args[@]}")
if [ -t 0 ] && [ -t 1 ]; then
    docker_args=(-it "${docker_args[@]}")
fi

if [ -n "${cann_install_prefix}" ]; then
    if [ "$#" -eq 0 ]; then
        set -- bash
    fi
    docker run "${docker_args[@]}" "${image}" bash -lc 'set +u; if [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then source "${ASCEND_HOME_PATH}/set_env.sh"; fi; exec "$@"' bash "$@"
else
    docker run "${docker_args[@]}" "${image}" "$@"
fi
