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
#
# Keep platform-specific decisions in conversion/ and target/.
# AFIR describes the union of supported semantics; bridge constructs it and artifact
# assembles already-selected output without choosing a platform.
# Reject SoC names case-insensitively, NpuArch enumerators and architecture numbers
# in dialect/AFIR/, bridge/ and artifact/. soc_version values may pass through as data.
# Extend pattern for new literals; do not add per-file exemptions.

set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
mlir_root="$(cd "${script_dir}/.." && pwd)"

layers=(dialect/AFIR bridge artifact)
pattern='[Aa]scend[0-9][0-9A-Za-z_]*|NpuArch::|\bV(2201|3510|5102|9202)\b|\b(2201|3510|5102|9202)\b'

status=0
for layer in "${layers[@]}"; do
  layer_path="${mlir_root}/${layer}"
  if [[ ! -d "${layer_path}" ]]; then
    echo "ERROR: ${layer} does not exist; update this check's layer list" >&2
    exit 1
  fi
  if hits="$(grep -rniE -- "${pattern}" "${layer_path}")"; then
    echo "ERROR: ${layer}/ must not name a platform:" >&2
    echo "${hits}" | sed 's|^|  |' >&2
    status=1
  fi
done

if [[ ${status} -ne 0 ]]; then
  echo "" >&2
  echo "Move the decision into conversion/ (an implementation + stable id selected" >&2
  echo "through Capability) or target/ (a TargetProfile field or TargetFeature)." >&2
  exit 1
fi

echo "Platform-agnostic layer check passed (${layers[*]})."
