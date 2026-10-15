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

echo "start run test case, please wait ..."
cd ${WORKSPACE}
: "${NPU_ARCH:?NPU_ARCH is required}"

export ASCEND_GLOBAL_LOG_LEVEL=2
export ASCEND_SLOG_PRINT_TO_STDOUT=0
source /usr/local/Ascend/cann/set_env.sh

log() {
  local dt
  dt=$(date '+%Y%m%d.%H%M%S')
  echo "===================================================================="
  echo "$dt : $*"
  echo "===================================================================="
}

log "init test case, please wait ..."


# ==============================
# 运行测试主循环
# ==============================

source /usr/local/Ascend/cann/set_env.sh
arm_package=$(basename "${arm_run_url}")
wget -nv ${arm_run_url} && chmod 777 ${arm_package}
/opt/conda/envs/python39/bin/python3 -m pip install packaging
/opt/conda/envs/python39/bin/python3 -m pip install build
/opt/conda/envs/python39/bin/python3 -m pip install 'setuptools>=68,<80'
chmod +x ${arm_package}
echo 'y' | bash ${arm_package} --full --quiet --pylocal
echo "bash super_kernel/examples/run_example.sh --npu-arch=${NPU_ARCH}"
source /opt/conda/bin/activate python39
pip3 install build
source /usr/local/Ascend/cann/set_env.sh && \
  bash super_kernel/examples/run_example.sh --npu-arch="${NPU_ARCH}" 2>&1 | tee -a ./run_test.log
source /opt/conda/bin/deactivate

# ==============================
# 打包log
# ==============================
mkdir -p /root/ascend
slog_name="slog.tar.gz"
tar -zcf "${slog_name}" -C /root/ascend log

# ==============================
# 检查 NPU 状态
# ==============================
log "checking NPU status ..."
mkdir -p ./npu_log
npu-smi info  2>&1 | tee ./npu_log/npu_info.log

# ==============================
# 检查测试结果
# ==============================
log "checking test results ..."

date_time=`date +%Y%m%d`"."`date +%H%M%S`
if grep -w -e "Run all examples success" "./run_test.log"; then
  echo "$date_time : run test case success"
else
  echo "$date_time : run test case failed"
  exit 1
fi
