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
# Platform differences belong in lowering, not above it: AFIR is the union of
# what the platforms can express, the bridge builds it without knowing the
# target, and the artifact layer assembles text a decision further down already
# made. Only conversion/ and target/ may name a chip.
#
# Concretely this rejects, inside dialect/AFIR/, bridge/ and artifact/:
#   - SoC names, in any case  (Ascend910B1, ascend910b1, ...) -- legacy spells
#     soc_version lowercase, so a case-sensitive pattern would miss the most
#     likely way this gets reintroduced
#   - NpuArch enumerators, qualified or bare (NpuArch::V3510, V3510)
#   - the NpuArch numbers, quoted or bare (2201, 3510, 5102, 9202)
#
# Passing a soc_version through as data -- reading it from the caller, handing
# it to TargetSelection -- stays allowed; it is the literal that is banned.
# Extend `pattern` below rather than adding a per-file exemption: an exemption
# is how this check stops meaning anything.

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
