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

CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${CURRENT_DIR}" || exit 1
source ../scripts/common.sh

sk_parse_npu_arch "$@" || exit $?
case "${NPU_ARCH}" in
    dav-2201) PY_SCRIPT=main-dav-2201.py ;;
    dav-3510) PY_SCRIPT=main-dav-3510.py ;;
esac
sk_cleanup_local || exit 1

RUN_LOG="${CURRENT_DIR}/tmp/run.log"
trap 'sk_uninstall_static_kernel_from_log "${RUN_LOG}"' EXIT
sk_run_python_with_log "${PY_SCRIPT}" "${RUN_LOG}" && \
    sk_check_static_kernel_outputs "${PWD}/static_kernel_compile_outputs" required
