/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NDDMA_VALIDATION_H_
#define NDDMA_VALIDATION_H_

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nddma {
constexpr uint32_t kMaxDim = 5;

inline uint32_t CheckedUint32(uint64_t value) {
  if (value > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("value exceeds uint32 tiling field");
  }
  return static_cast<uint32_t>(value);
}

inline uint32_t CheckedDimension(uint64_t dim) {
  if (dim < 1 || dim > kMaxDim) {
    throw std::invalid_argument("dim must be in [1, 5]");
  }
  return static_cast<uint32_t>(dim);
}

inline void ValidateShape(uint32_t dim, const std::vector<int64_t> &dims, const std::vector<int64_t> &input_stride,
                          const std::vector<int64_t> &output_stride) {
  CheckedDimension(dim);
  if (dims.size() != dim || input_stride.size() != dim || output_stride.size() != dim) {
    throw std::invalid_argument("dim/stride length mismatch");
  }
  for (uint32_t i = 0; i < dim; ++i) {
    // Zero GM stride is valid for broadcast.
    if (dims[i] <= 0 || input_stride[i] < 0 || output_stride[i] <= 0) {
      throw std::invalid_argument("dims and UB strides must be positive; GM strides must be nonnegative");
    }
  }
}

inline uint32_t CheckedUbBufferBytes(uint32_t offset, uint32_t span, uint32_t dtype_size) {
  const uint64_t elems = static_cast<uint64_t>(offset) + (span == 0 ? 1 : span) + 64;
  if (dtype_size == 0 || elems > std::numeric_limits<uint32_t>::max() / dtype_size) {
    throw std::invalid_argument("UB buffer bytes exceed uint32 tiling field");
  }
  return static_cast<uint32_t>(elems * dtype_size);
}
}  // namespace nddma

#endif  // NDDMA_VALIDATION_H_
