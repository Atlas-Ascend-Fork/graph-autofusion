#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

set -e

run_sample() {
    local sample_name="$1"
    shift

    echo "[INFO][SAMPLE] running ${sample_name}"
    if "$@"; then
        echo "[INFO][SAMPLE] passed ${sample_name}"
        return 0
    fi

    local rc=$?
    echo "[ERROR][SAMPLE] failed ${sample_name}, exit code: ${rc}" >&2
    exit "${rc}"
}

# Engineering test batch script; individual sample execution is documented in each sample directory.
BASEPATH=$(cd "$(dirname "$0")"; pwd)
REPO_ROOT=$(cd "${BASEPATH}/../.."; pwd)

# 选择运行样例使用的 Python 解释器。
if [ -n "${VIRTUAL_ENV:-}" ]; then
    PYTHON_CMD="python3"
elif [ -f "${REPO_ROOT}/venv/bin/python" ]; then
    PYTHON_CMD="${REPO_ROOT}/venv/bin/python"
else
    PYTHON_CMD="python3"
fi

pip3 list
pip3 show torch_npu
pip install torch-npu==2.7.1.post10 --extra-index-url https://ascend.devcloud.huaweicloud.com/pypi/simple/
pip install tqdm
echo "pip3 inistall torch_npu for run example"

source "${BASEPATH}/aot/scripts/common.sh"

sk_parse_npu_arch "$@" || exit $?

echo "---------------- Start engineering example test batch ----------------"
if [[ "${NPU_ARCH}" == "dav-2201" ]]; then
    run_sample "jit/example01_super_kernel_base" \
        "${PYTHON_CMD}" "${BASEPATH}/jit/example01_super_kernel_base/superkernel_scope.py"
    run_sample "jit/example02_super_kernel_profiling" \
        "${PYTHON_CMD}" "${BASEPATH}/jit/example02_super_kernel_profiling/superkernel_compare.py"
    run_sample "jit/example03_super_kernel_runtime_ascendc_only" \
        "${PYTHON_CMD}" \
        "${BASEPATH}/jit/example03_super_kernel_runtime_ascendc_only/superkernel_runtime_ascendc_basic.py"
else
    echo "[INFO] Skipping SuperKernel JIT examples on ${NPU_ARCH}."
fi

export PYTHON_CMD

if sk_check_driver_version_for_aot; then
    run_sample "aot/example01_dual_stream" \
        bash "${BASEPATH}/aot/example01_dual_stream/run.sh" --npu-arch="${NPU_ARCH}"
    run_sample "aot/example02_sk_options" \
        bash "${BASEPATH}/aot/example02_sk_options/run.sh" --npu-arch="${NPU_ARCH}"
    run_sample "aot/example03_kernel_pybind" \
        bash "${BASEPATH}/aot/example03_kernel_pybind/run.sh" --npu-arch="${NPU_ARCH}"
else
    echo "[WARNING][AOT-SKIP] AOT examples skipped; treating the environment skip as success."
fi

echo "Run all examples success"
