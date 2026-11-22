/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_COMMON_TEMPLATE_KEYS_H
#define AUTOFUSE_MLIR_COMMON_TEMPLATE_KEYS_H
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <vector>
#include "common/schedule_result.h"
namespace mlir::common {
// ATT CheckKeyValid requires all explicit keys or all default keys in a stage.
// Under that invariant GenerateModelInfo's counter equals the original index.
inline int64_t TemplateKey(const af::AscGraph &graph, size_t index) {
  return graph.GetTilingKey() >= 0 ? graph.GetTilingKey() : static_cast<int64_t>(index);
}
inline bool ValidateTemplateKeys(const ascir::ScheduleGroup &group, std::string *diagnostic) {
  if (group.impl_graphs.empty()) {
    return false;
  }
  const bool explicitKeys = group.impl_graphs.front().GetTilingKey() >= 0;
  std::set<int64_t> keys;
  for (size_t i = 0; i < group.impl_graphs.size(); ++i) {
    const auto &graph = group.impl_graphs[i];
    const auto key = TemplateKey(graph, i);
    std::string error;
    if ((graph.GetTilingKey() >= 0) != explicitKeys) {
      error = "stage mixes explicit and implicit tiling keys (ATT requires all or none)";
    } else if (key < 0 || static_cast<uint64_t>(key) > std::numeric_limits<uint32_t>::max()) {
      error = "stage tiling_key is outside uint32 range";
    } else if (!keys.insert(key).second) {
      error = "templates sharing tiling_key " + std::to_string(key) +
              "; AscGraph::SetTilingKey must identify a unique template";
    }
    if (!error.empty()) {
      if (diagnostic) {
        *diagnostic = error;
      }
      return false;
    }
  }
  return true;
}
}  // namespace mlir::common
#endif
