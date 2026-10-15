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

cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
source ../scripts/common.sh

sk_parse_npu_arch "$@" || exit $?
sk_cleanup_local || exit 1

RUN_LOG="${PWD}/tmp/run.log"
trap 'sk_uninstall_static_kernel_from_log "${RUN_LOG}"' EXIT
sk_run_python_with_log main.py "${RUN_LOG}" && \
    sk_check_static_kernel_outputs "${PWD}/static_kernel_compile_outputs" required
