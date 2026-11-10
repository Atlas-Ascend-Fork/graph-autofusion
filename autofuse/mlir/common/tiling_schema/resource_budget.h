/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_COMMON_RESOURCE_BUDGET_H
#define AUTOFUSE_MLIR_COMMON_RESOURCE_BUDGET_H
#include <cstdint>
#include <limits>
namespace mlir::common {
// Existing host tiling ABI reservation; physical capacity includes these bytes.
inline constexpr uint32_t kTilingReservedUbBytes = 256;
struct ResourceBudget {
  uint64_t aiv_num{0};
  uint64_t ub_size{0};
  bool valid() const {
    return aiv_num > 0 && aiv_num <= std::numeric_limits<uint32_t>::max() && ub_size > kTilingReservedUbBytes &&
           ub_size <= std::numeric_limits<uint32_t>::max();
  }
  uint64_t availableUb() const {
    return ub_size - kTilingReservedUbBytes;
  }
};
}  // namespace mlir::common
#endif
