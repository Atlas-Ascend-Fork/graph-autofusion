#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../../../scripts/common.sh"
sk_parse_npu_arch "$@"

SAMPLE_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
PACKAGE_DIR="${SAMPLE_DIR}/tmp/python_packages"
if [ -L "${SAMPLE_DIR}/tmp" ] || [ -e "${PACKAGE_DIR}" ] || [ -L "${PACKAGE_DIR}" ]; then
    echo "ERROR: temporary package directory must be unused and not redirected" >&2
    exit 1
fi

SK_NPU_ARCH="${NPU_ARCH}" "${PYTHON_CMD:-python3}" -m pip install --target "${PACKAGE_DIR}" --no-deps --no-build-isolation "${SCRIPT_DIR}"

echo "Installed ACLGraph add op_extension package from ${SCRIPT_DIR}."
