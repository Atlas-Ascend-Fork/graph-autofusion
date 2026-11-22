/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "axes_callback_bindings.h"
#include <algorithm>

namespace att {
namespace {
bool ValidIndices(const std::vector<uint32_t> &indices, size_t expected, uint32_t limit) {
  return indices.size() == expected &&
         std::all_of(indices.begin(), indices.end(), [limit](uint32_t index) { return index < limit; });
}
bool ValidMixedIndices(const std::vector<uint32_t> &indices, size_t expected, uint32_t flag, uint32_t plain_limit,
                       uint32_t tagged_limit) {
  return indices.size() == expected && std::all_of(indices.begin(), indices.end(), [=](uint32_t encoded) {
           const auto limit = (encoded & flag) != 0 ? tagged_limit : plain_limit;
           return (encoded & ~flag) < limit;
         });
}
}  // namespace

TilingExpressionStatus VerifyAxesCallbackBindings(const AxesCallbackBindings &bindings, uint32_t tiling_count,
                                                  uint32_t shape_count) {
  using Status = TilingExpressionStatus;
  using Role = AxesProgramEvaluationRole;
  if (tiling_count > kTilingExpressionMaxItems || shape_count > kTilingExpressionMaxItems ||
      bindings.constraints.size() > kTilingExpressionMaxItems || bindings.upper_bounds.size() > tiling_count) {
    return Status::kInvalidProgram;
  }
  for (const auto &binding : bindings.constraints) {
    const auto &callback = binding.callback;
    if (!callback || callback->Role() != Role::kConstraint ||
        !ValidIndices(binding.tiling_indices, callback->RelatedTilingCount(), tiling_count) ||
        !ValidMixedIndices(binding.shape_indices, callback->RelatedInputCount(), kAxesTilingRelatedBit, shape_count,
                           tiling_count)) {
      return Status::kInvalidProgram;
    }
  }
  std::vector<bool> bound_targets(tiling_count);
  for (const auto &binding : bindings.upper_bounds) {
    const auto &callback = binding.callback;
    if (!callback || callback->Role() != Role::kUpperBound || binding.target_index >= tiling_count ||
        bound_targets[binding.target_index] ||
        !ValidMixedIndices(binding.parent_indices, callback->ParentCount(), kAxesShapeParentBit, tiling_count,
                           shape_count)) {
      return Status::kInvalidProgram;
    }
    bound_targets[binding.target_index] = true;
  }
  return Status::kOk;
}
}  // namespace att
