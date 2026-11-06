/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_BRC_INLINE_API_H__
#define __ASCENDC_API_BRC_INLINE_API_H__

template <typename T>
inline __aicore__ void BinaryBrcInlineApiWithTwoVectorizedAxis(
    const LocalTensor<T> &dstLocal, const LocalTensor<T> &src0Local, const LocalTensor<T> &src1Local,
    const int64_t shape0,  // 输出的两个向量化轴的repeate.由于都是对齐的，直接取repeate
    const int64_t shape1,
    const uint8_t is_input0_block_brc,  // index0是否支持广播
    const uint8_t is_input1_block_brc,  // index1是否支持广播
    const int64_t first_axis_v_stride,  // 首轴v_stride
    const int64_t dtype_size,
    void (*FUNC1)(const LocalTensor<T> &, const LocalTensor<T> &, const LocalTensor<T> &, const int32_t &),
    void (*FUNC2)(const LocalTensor<T> &, const LocalTensor<T> &, const LocalTensor<T> &, uint64_t, const uint8_t,
                  const BinaryRepeatParams &)) {
  int64_t element = shape1;
  int64_t block = shape0;
  int64_t elem_in_one_repeat = 256 / dtype_size;
  int64_t elem_in_one_block = 32 / dtype_size;
  int64_t cut_quotient = element / elem_in_one_repeat;
  int64_t cut_reminder = element - cut_quotient * elem_in_one_repeat;
  int64_t repeat_stride_in_blocks = first_axis_v_stride / elem_in_one_block;
  if (cut_quotient >= block || repeat_stride_in_blocks > 255) {
    // 将block层外抛作为for循环，原有的element整体使用counter模式
    for (int64_t outer_for = 0; outer_for < block; outer_for++) {
      FUNC1(dstLocal[outer_for * first_axis_v_stride], src0Local[outer_for * first_axis_v_stride * is_input0_block_brc],
            src1Local[outer_for * first_axis_v_stride * is_input1_block_brc], element);
    }
    return;
  }
  // 切分系数小于for循环长度， 则将for循环作为repeat层，将切分系数外抛作为for循环
  constexpr uint8_t dst_block_stride = 1;
  constexpr uint8_t src0_block_stride = 1;
  constexpr uint8_t src1_block_stride = 1;
  uint8_t dst_repeat_stride = static_cast<uint8_t>(repeat_stride_in_blocks);
  uint8_t src0_repeat_stride = dst_repeat_stride * is_input0_block_brc;
  uint8_t src1_repeat_stride = dst_repeat_stride * is_input1_block_brc;
  uint32_t calcSize = 0;
  uint32_t offset = 0;
  for (int64_t outer_for = 0; outer_for < cut_quotient; outer_for++) {
    calcSize = outer_for * elem_in_one_repeat;
    while (block > 255) {
      FUNC2(dstLocal[calcSize + offset], src0Local[calcSize + is_input0_block_brc * offset],
            src1Local[calcSize + is_input1_block_brc * offset], elem_in_one_repeat, 255,
            {dst_block_stride, src0_block_stride, src1_block_stride, dst_repeat_stride, src0_repeat_stride,
             src1_repeat_stride});
      block -= 255;
      offset += first_axis_v_stride * 255;
    }
    FUNC2(dstLocal[calcSize + offset], src0Local[calcSize + is_input0_block_brc * offset],
          src1Local[calcSize + is_input1_block_brc * offset], elem_in_one_repeat, block,
          {dst_block_stride, src0_block_stride, src1_block_stride, dst_repeat_stride, src0_repeat_stride,
           src1_repeat_stride});
  }
  //  处理尾块
  if (cut_reminder > 0) {
    calcSize = cut_quotient * elem_in_one_repeat;
    while (block > 255) {
      FUNC2(dstLocal[calcSize + offset], src0Local[calcSize + is_input0_block_brc * offset],
            src1Local[calcSize + is_input1_block_brc * offset], cut_reminder, 255,
            {dst_block_stride, src0_block_stride, src1_block_stride, dst_repeat_stride, src0_repeat_stride,
             src1_repeat_stride});
      block -= 255;
      offset += first_axis_v_stride * 255;
    }
    FUNC2(dstLocal[calcSize + offset], src0Local[calcSize + is_input0_block_brc * offset],
          src1Local[calcSize + is_input1_block_brc * offset], cut_reminder, block,
          {dst_block_stride, src0_block_stride, src1_block_stride, dst_repeat_stride, src0_repeat_stride,
           src1_repeat_stride});
  }
}

#endif  // __ASCENDC_API_BRC_INLINE_API_H__
