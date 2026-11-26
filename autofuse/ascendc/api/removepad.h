/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REMOVEPAD_H__
#define __ASCENDC_API_REMOVEPAD_H__

template <typename T>
inline __aicore__ void RemovePadExtend(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                       const uint64_t last_axis_actual_size, const uint64_t last_axis_size,
                                       const uint64_t non_last_axis_merge) {
  uint64_t rsvd_cnt = 0;
  uint8_t mask = 7;
  AscendC::GatherMaskParams params;
  params.src0BlockStride = 1;
  params.repeatTimes = non_last_axis_merge;
  params.src0RepeatStride = last_axis_size * sizeof(T) / 32;
  params.src1RepeatStride = 0;
  if (last_axis_size == last_axis_actual_size) {
    // Flatten compact rows into one count-mode operation. A repeat spans 256 bytes,
    // so the source repeat stride is 8 blocks, independent of dtype and row size.
    constexpr uint16_t kVectorRepeatBytes = 256;
    constexpr uint16_t kBlockBytes = 32;
    params.repeatTimes = 1;
    params.src0RepeatStride = kVectorRepeatBytes / kBlockBytes;
    AscendC::GatherMask(dst, src, mask, true, non_last_axis_merge * last_axis_actual_size, params, rsvd_cnt);
    return;
  }
  AscendC::GatherMask(dst, src, mask, true, last_axis_actual_size, params, rsvd_cnt);
}
#endif  // __ASCENDC_API_REMOVEPAD_H__
