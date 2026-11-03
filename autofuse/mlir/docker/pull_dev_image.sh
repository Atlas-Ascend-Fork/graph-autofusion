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

readonly IMAGE_REPOSITORY="swr.cn-east-2.myhuaweicloud.com/ascendmlir/autofuse-mlir-dev"
readonly REGISTRY="swr.cn-east-2.myhuaweicloud.com"

if [[ "$#" -ne 0 ]]; then
    echo "ERROR: this script does not accept parameters; run it without arguments" >&2
    exit 2
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker is required to pull the image" >&2
    exit 1
fi

case "$(uname -m)" in
    aarch64 | arm64)
        image_arch="arm64"
        platform="linux/arm64"
        ;;
    amd64 | x86_64)
        image_arch="amd64"
        platform="linux/amd64"
        ;;
    *)
        echo "ERROR: unsupported CPU architecture: $(uname -m)" >&2
        exit 2
        ;;
esac

remote_image="${IMAGE_REPOSITORY}:${image_arch}"
local_image="autofuse-mlir-dev:${image_arch}"
registry_user="${AF_MLIR_REGISTRY_USER:-${SWR_USER:-}}"
registry_token="${AF_MLIR_REGISTRY_TOKEN:-${SWR_TOKEN:-}}"

if [[ -n "${registry_user}" || -n "${registry_token}" ]]; then
    if [[ -z "${registry_user}" || -z "${registry_token}" ]]; then
        echo "ERROR: AF_MLIR_REGISTRY_USER and AF_MLIR_REGISTRY_TOKEN must be set together" >&2
        exit 2
    fi
    printf '%s' "${registry_token}" | docker login -u "${registry_user}" --password-stdin "${REGISTRY}"
fi

docker pull --platform "${platform}" "${remote_image}"
docker tag "${remote_image}" "${local_image}"

docker image inspect "${local_image}" \
    --format 'arch={{.Architecture}} os={{.Os}} id={{.Id}} size={{.Size}}'

if [[ "${AF_MLIR_PULL_LOGOUT:-0}" == "1" && -n "${registry_user}" ]]; then
    docker logout "${REGISTRY}"
fi

echo "Pulled latest ${platform} image: ${remote_image}"
echo "Use image: ${local_image}"
