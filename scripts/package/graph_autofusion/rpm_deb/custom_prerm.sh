#!/bin/bash
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------------------

if [ -z "${INSTALL_PATH:-}" ]; then
    echo "================ graph_autofusion rpm/deb prerm skipped: INSTALL_PATH is empty ================"
    exit 0
fi

sourcedir="${INSTALL_PATH}"
WHL_INSTALL_DIR_PATH="${sourcedir}/python/site-packages"
export PYTHONPATH="${WHL_INSTALL_DIR_PATH}"
export PIP_BREAK_SYSTEM_PACKAGES=1

python3 -m pip uninstall -y superkernel >/dev/null 2>&1 || pip3 uninstall -y superkernel >/dev/null 2>&1 || true
if [ -d "${WHL_INSTALL_DIR_PATH}/multistream" ] ||
    compgen -G "${WHL_INSTALL_DIR_PATH}/multistream-*.dist-info" >/dev/null 2>&1; then
    python3 -m pip uninstall -y multistream >/dev/null 2>&1 || pip3 uninstall -y multistream >/dev/null 2>&1 || true
fi

rm -fr "${WHL_INSTALL_DIR_PATH}/superkernel" 2>/dev/null
rm -fr "${WHL_INSTALL_DIR_PATH}"/superkernel-*.dist-info 2>/dev/null
rm -fr "${WHL_INSTALL_DIR_PATH}/multistream" 2>/dev/null
rm -fr "${WHL_INSTALL_DIR_PATH}"/multistream-*.dist-info 2>/dev/null
chmod +w -R "${WHL_INSTALL_DIR_PATH}/autofuse/__pycache__" 2>/dev/null || true
rm -fr "${WHL_INSTALL_DIR_PATH}/autofuse/__pycache__" 2>/dev/null

echo "================ graph_autofusion rpm/deb prerm done: packages=superkernel,multistream, site_packages=${WHL_INSTALL_DIR_PATH:-unknown} ================"
