/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AUTOFUSE_MLIR_BRIDGE_ATTSCHEMAADAPTER_TILINGCODEBUNDLE_H
#define AUTOFUSE_MLIR_BRIDGE_ATTSCHEMAADAPTER_TILINGCODEBUNDLE_H

#include <cstdint>
#include <variant>
#include "common/tiling_schema/resource_budget.h"
#include <vector>
#include <map>
#include <string>

namespace mlir::bridge {

using AttOptions = std::map<std::string, std::string>;

// Concrete ATT values keyed by original group and field names. Mapping by name
// preserves the solved parameters while the emitted ABI drops unused groups.
struct StaticTilingData {
  common::ResourceBudget resources;
  bool flat_schedule{false};
  uint64_t workspace_size{0};
  std::map<std::string, std::map<std::string, std::vector<uint8_t>>> fields;
};

// Schedule-neutral ATT output consumed by host tiling source assembly.
struct TilingCodeBundle {
  std::map<std::string, std::string> tiling_functions;
  std::string schema_version;
  bool schema_validated{false};
};

// Exactly one delivery mode: runtime source or frozen ATT values.
using TilingArtifact = std::variant<TilingCodeBundle, StaticTilingData>;

}  // namespace mlir::bridge

#endif  // AUTOFUSE_MLIR_BRIDGE_ATTSCHEMAADAPTER_TILINGCODEBUNDLE_H
