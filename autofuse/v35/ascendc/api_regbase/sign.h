/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_SIGN_H__
#define __ASCENDC_API_SIGN_H__

// 目前只支持half/float int32/int64
template <typename T>
inline __aicore__ void SignExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                  AscendC::LocalTensor<uint8_t> &tmp_buf, const uint32_t size) {
  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, float> || std::is_same_v<T, int64_t>) {
    // AscendApi 支持 half/float/int64类型直接调用
    AscendC::Sign(dst, src, tmp_buf, size);
    return;
  }

  // int32 转换成float类型 再调用AscendC::Sign
  uint32_t offset = tmp_buf.GetSize() / 3U / 32U * 32U;

  ASCENDC_ASSERT(offset > AscendC::ONE_REPEAT_BYTE_SIZE, {
    KERNEL_LOG(KERNEL_ERROR, "sign api tmpbuf is too small, tmpbuf is %u bytes, required more than %u bytes!",
               tmp_buf.GetSize(), AscendC::ONE_REPEAT_BYTE_SIZE);
  });

  // Prepare float type tensor for src
  AscendC::LocalTensor<float> float_src = tmp_buf.template ReinterpretCast<float>();
  float_src.SetSize(offset / sizeof(float));

  // Prepare float type tensor for dst
  AscendC::LocalTensor<float> float_dst = tmp_buf[offset].template ReinterpretCast<float>();
  float_dst.SetSize((offset) / sizeof(float));

  // Prepare sharedTmpBuffer for sign
  AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmp_buf[offset + offset].template ReinterpretCast<uint8_t>();
  sharedTmpBuffer.SetSize(tmp_buf.GetSize() - offset - offset);

  int calc_size = 0;

  // Calc when size > max_repeat
  int max_repeat = float_src.GetSize() / (AscendC::ONE_REPEAT_BYTE_SIZE / sizeof(float));
  max_repeat = max_repeat > MAX_REPEAT_TIME ? MAX_REPEAT_TIME : max_repeat;
  const int max_repeat_calc_size = max_repeat * AscendC::ONE_REPEAT_BYTE_SIZE / sizeof(float);
  for (; calc_size + max_repeat_calc_size < size; calc_size += max_repeat_calc_size) {
    CastExtend(float_src, src[calc_size], sharedTmpBuffer, max_repeat_calc_size);
    AscendC::Sign(float_dst, float_src, sharedTmpBuffer, max_repeat_calc_size);
    CastExtend(dst[calc_size], float_dst, sharedTmpBuffer, max_repeat_calc_size);
  }

  // Calc max_repeat > size > one_repeat
  constexpr int one_repeat_calc_size = AscendC::ONE_REPEAT_BYTE_SIZE / sizeof(float);
  if (calc_size + one_repeat_calc_size <= size) {
    const int repeat = (size - calc_size) / one_repeat_calc_size;
    CastExtend(float_src, src[calc_size], sharedTmpBuffer, repeat * one_repeat_calc_size);
    AscendC::Sign(float_dst, float_src, sharedTmpBuffer, repeat * one_repeat_calc_size);
    CastExtend(dst[calc_size], float_dst, sharedTmpBuffer, repeat * one_repeat_calc_size);
    calc_size += repeat * one_repeat_calc_size;
  }

  // Calc when one_repeat > size
  if (calc_size < size) {
    CastExtend(float_src, src[calc_size], sharedTmpBuffer, size - calc_size);
    AscendC::Sign(float_dst, float_src, sharedTmpBuffer, size - calc_size);
    CastExtend(dst[calc_size], float_dst, sharedTmpBuffer, size - calc_size);
  }
}

#endif  // __ASCENDC_API_SIGN_H__
