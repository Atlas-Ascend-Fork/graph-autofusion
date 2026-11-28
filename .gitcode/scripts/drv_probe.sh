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
# 临时诊断: 记录 /usr/local/Ascend/driver 在冒烟作业各阶段的状态, 定位 version.info 何时不可读。
# boot_id 用于区分节点, 同一节点多次作业的 boot_id 相同。

label="${1:-unknown}"
driver_root=/usr/local/Ascend/driver
addons_root=/usr/local/Ascend/add-ons

host=$(cat /proc/sys/kernel/hostname 2>/dev/null || echo unknown)
boot_id=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo unknown)
count=$(ls -A "${driver_root}" 2>/dev/null | wc -l | tr -d ' ')
entries=$(ls -A "${driver_root}" 2>/dev/null | tr '\n' ' ')
dir_mtime=$(stat -c '%y' "${driver_root}" 2>/dev/null || echo unknown)
drv_ver=$(sed -n 's/^Version=//p' "${driver_root}/version.info" 2>/dev/null | head -1)
addons=$(ls -A "${addons_root}" 2>/dev/null | wc -l | tr -d ' ')
if [ -r "${driver_root}/version.info" ]; then version_info=yes; else version_info=no; fi

echo "[DRV-PROBE] ${label} ts=$(date '+%Y-%m-%d %H:%M:%S') host=${host} boot_id=${boot_id}"
echo "[DRV-PROBE] ${label} version_info=${version_info} drv_ver=${drv_ver:-none} count=${count} addons=${addons} dir_mtime=${dir_mtime}"
echo "[DRV-PROBE] ${label} entries: ${entries}"
grep -i ascend /proc/self/mountinfo 2>/dev/null | sed "s|^|[DRV-PROBE] ${label} mount: |"

exit 0
