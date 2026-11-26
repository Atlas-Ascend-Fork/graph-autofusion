/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IMPL_PAD_BROADCAST_BROADCAST_EXTEND_UTILS_H
#define IMPL_PAD_BROADCAST_BROADCAST_EXTEND_UTILS_H

#include "kernel_basic_intf.h"
#include "kernel_tensor.h"

namespace AscendC {

namespace BroadcastInternal {
constexpr uint32_t U16_MAX = 65536;

template <uint32_t size = sizeof(int8_t)>
struct ExtractSignedTypeBySize {
  using T = int16_t;
};

template <>
struct ExtractSignedTypeBySize<sizeof(int16_t)> {
  using T = int16_t;
};

template <>
struct ExtractSignedTypeBySize<sizeof(int32_t)> {
  using T = int32_t;
};

template <>
struct ExtractSignedTypeBySize<sizeof(int64_t)> {
  using T = int32_t;
};

template <uint32_t size = sizeof(uint8_t)>
struct ExtractUnsignedTypeBySize {
  using T = uint8_t;
};

template <>
struct ExtractUnsignedTypeBySize<sizeof(uint16_t)> {
  using T = uint16_t;
};

template <>
struct ExtractUnsignedTypeBySize<sizeof(uint32_t)> {
  using T = uint32_t;
};

template <>
struct ExtractUnsignedTypeBySize<sizeof(uint64_t)> {
  using T = uint32_t;
};

template <uint32_t size = sizeof(uint8_t)>
struct ExtractIndexTypeBySize {
  using T = uint16_t;
};

template <>
struct ExtractIndexTypeBySize<sizeof(uint16_t)> {
  using T = uint16_t;
};

template <>
struct ExtractIndexTypeBySize<sizeof(uint32_t)> {
  using T = uint32_t;
};

template <>
struct ExtractIndexTypeBySize<sizeof(uint64_t)> {
  using T = uint32_t;
};

__aicore__ inline void DstShapeCheck(const uint32_t *dstShape, uint32_t dim) {
  for (uint16_t i = 0; i < dim; ++i) {
    ASCENDC_ASSERT((dstShape[i] <= U16_MAX), {
      KERNEL_LOG(KERNEL_ERROR, "shape should be less than uint16 max, got shape[%u]=%u", i, dstShape[i]);
    });
  }
}

__aicore__ inline void ShapeCheck(uint32_t *tillingShape, const uint32_t *shape, uint32_t rank) {
  for (uint16_t i = 0; i < rank; ++i) {
    ASCENDC_ASSERT((shape[i] == tillingShape[i]), {
      KERNEL_LOG(KERNEL_ERROR, "Tiling shape should be equal to shape, tillingShape[%u]=%u, shape[%u]=%u", i,
                 tillingShape[i], i, shape[i]);
    });
  }
}
}  // namespace BroadcastInternal

}  // namespace AscendC

#endif  // IMPL_PAD_BROADCAST_BROADCAST_EXTEND_UTILS_H
