/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_COMPARE_V2_H__
#define __ASCENDC_API_COMPARE_V2_H__

constexpr uint32_t kShortBitWidth = 16;

template <typename T, CMPMODE mode>
inline __aicore__ void CompareNormalNoLoop(const AscendC::LocalTensor<uint8_t> &dst,  // output
                                           const AscendC::LocalTensor<T> &src0,       // input 0 is a tensor
                                           const AscendC::LocalTensor<T> &src1,       // input 1 is a tensor
                                           const uint8_t repeat_times, const uint32_t last_axis,
                                           const uint8_t dst_repeat_stride, const uint8_t src_repeat_stride,
                                           AscendC::LocalTensor<uint8_t> &tmp_buf, const bool src1_is_one_block) {
  uint32_t tmp_offset = 0;
  // compare_out用1bit表示1个元素
  LocalTensor<uint8_t> compare_out = tmp_buf[tmp_offset].template ReinterpretCast<uint8_t>();
  uint8_t compare_element_num = 256 / sizeof(T);
  compare_out.SetSize((repeat_times * compare_element_num) /
                      8);  // Compare指令一次处理256B的数据，根据数据类型计算需要的空间
  // compare每次根据数据类型读取固定数量的数据，并进行固定数量的stride
  tmp_offset += (repeat_times * 256 / sizeof(T)) * sizeof(uint8_t) / 8;
  LocalTensor<half> select_out = tmp_buf[KernelUtils::BlkAlign<uint8_t>(tmp_offset)].template ReinterpretCast<half>();
  uint32_t select_repeat_times = DivCeil(repeat_times * 2, sizeof(T));
  select_out.SetSize(
      128 *
      select_repeat_times);  // == ONE_REPEAT_BYTE_SIZE / sizeof(half) * DivCeil(repeat_times * sizeof(half), sizeof(T))

  // 由于轴内是连续的， mask直接使用last_axis
  const uint64_t mask = last_axis;
  // compare mask必须是连续的，所以dst_repeat_stride为8
  BinaryRepeatParams repeat_params = {1, 1, 1, 8, src_repeat_stride, src_repeat_stride};
  if (src1_is_one_block) {
    repeat_params = {1, 1, 0, 8, src_repeat_stride, 0};
  }
  // 处理 float/half 数据类型输入tensor存在NaN场景
  // NE模式下，Compare对NaN总是返回0，但期望NaN与任何值做NE比较返回1
  // 使用位掩码修正法：result = NE_mask | is_nan0 | is_nan1
  // 其中 is_nan = ~Compare(x, x, EQ)，通过Not翻转EQ结果得到
  if constexpr ((AscendC::IsSameType<T, float>::value || AscendC::IsSameType<T, half>::value) && mode == CMPMODE::NE) {
    uint32_t select_bits_per_repeat = ONE_REPEAT_BYTE_SIZE / sizeof(half);
    uint8_t cast_dst_blk_stride = dst_repeat_stride;
    const uint32_t src_offset_per_repeat = static_cast<uint32_t>(src_repeat_stride) * ONE_BLK_SIZE / sizeof(T);
    // compare_out的bit数按int16对齐（Not/Or只支持int16），每个int16包含16个比较bit
    uint32_t mask_int16_count = (compare_element_num + kShortBitWidth - 1) / kShortBitWidth;
    // nan_buf复用select_out空间（步骤1-7使用，步骤8的Duplicate会覆盖）
    LocalTensor<uint8_t> nan_buf = select_out.ReinterpretCast<uint8_t>();
    LocalTensor<int16_t> nan_buf_int16 = select_out.ReinterpretCast<int16_t>();
    LocalTensor<int16_t> compare_out_int16 = compare_out.ReinterpretCast<int16_t>();
    BinaryRepeatParams src1_repeat_params = {1, 1, 1, 8, src_repeat_stride, src_repeat_stride};
    if (src1_is_one_block) {
      src1_repeat_params = {1, 1, 0, 8, src_repeat_stride, 0};
    }

    for (uint8_t r = 0; r < repeat_times; r++) {
      const uint32_t src0_offset = r * src_offset_per_repeat;
      const uint32_t src1_offset = src1_is_one_block ? 0 : src0_offset;
      // Step 1: Compare(src0, src0, EQ) → nan_buf (NaN=0, 非NaN=1)
      AscendC::PipeBarrier<PIPE_V>();
      Compare(nan_buf, src0[src0_offset], src0[src0_offset], CMPMODE::EQ, mask, 1,
              {1, 1, 1, 8, src_repeat_stride, src_repeat_stride});
      // Step 2: Not → is_nan0 (NaN=1, 非NaN=0)
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Not(nan_buf_int16, nan_buf_int16, mask_int16_count);

      // Step 3: Compare(src1, src1, EQ) → compare_out (NaN=0, 非NaN=1)
      AscendC::PipeBarrier<PIPE_V>();
      Compare(compare_out, src1[src1_offset], src1[src1_offset], CMPMODE::EQ, mask, 1, src1_repeat_params);
      // Step 4: Not → is_nan1 (NaN=1, 非NaN=0)
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Not(compare_out_int16, compare_out_int16, mask_int16_count);

      // Step 5: Or → is_nan = is_nan0 | is_nan1
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Or(compare_out_int16, compare_out_int16, nan_buf_int16, mask_int16_count);

      // Step 6: Compare(src0, src1, NE) → nan_buf (原始NE结果)
      AscendC::PipeBarrier<PIPE_V>();
      Compare(nan_buf, src0[src0_offset], src1[src1_offset], mode, mask, 1, repeat_params);

      // Step 7: Or → 修正后的NE结果 (NE | is_nan)，写入compare_out
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Or(compare_out_int16, nan_buf_int16, compare_out_int16, mask_int16_count);

      // Step 8: Select+Cast → dst（与非NE路径相同）
      AscendC::PipeBarrier<PIPE_V>();
      Duplicate<half, true>(select_out, (half)1, select_bits_per_repeat, 1, 1, 8);
      AscendC::PipeBarrier<PIPE_V>();
      Select<half, uint8_t, true>(select_out, compare_out, select_out, (half)0, SELMODE::VSEL_TENSOR_SCALAR_MODE,
                                  select_bits_per_repeat, 1, {1, 1, 1, 8, 8, 8});
      AscendC::PipeBarrier<PIPE_V>();
      Cast<uint8_t, half, true>(dst[r * dst_repeat_stride * ONE_BLK_SIZE], select_out, RoundMode::CAST_NONE, mask, 1,
                                {1, 1, cast_dst_blk_stride, 8});
    }
  } else {
    AscendC::PipeBarrier<PIPE_V>();
    Compare(compare_out, src0, src1, mode, mask, repeat_times, repeat_params);
    Duplicate<half, true>(select_out, 1, ONE_REPEAT_BYTE_SIZE / sizeof(half), select_repeat_times, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    repeat_params = {1, 1, 1, 8, 8, 8};
    Select<half, uint8_t, true>(select_out, compare_out, select_out, (half)0, SELMODE::VSEL_TENSOR_SCALAR_MODE,
                                ONE_REPEAT_BYTE_SIZE / sizeof(half), select_repeat_times, repeat_params);
    // 为了更高效，select的repeat进行完整的8个block，而与T无关，因为compare out上是连续的
    // 因为可能一次select跨了多个src对应的repeat stride，所以repeat times可以更少
    uint8_t select_out_stride =
        compare_element_num * sizeof(half) / ONE_BLK_SIZE;  // select_out中有效数据的stride由compare的输出决定
    UnaryRepeatParams unary_repeat_params = {1, 1, dst_repeat_stride, select_out_stride};
    AscendC::PipeBarrier<PIPE_V>();
    Cast<uint8_t, half, true>(dst, select_out, RoundMode::CAST_NONE, mask, repeat_times, unary_repeat_params);
  }
}

template <typename T, CMPMODE mode>
inline __aicore__ void CompareNormal(const AscendC::LocalTensor<uint8_t> &dst,  // output
                                     const AscendC::LocalTensor<T> &src0,       // input 0 is a tensor
                                     const AscendC::LocalTensor<T> &src1,       // input 1 is a tensor
                                     const uint8_t repeat_times, const uint32_t last_axis,
                                     const uint32_t input_last_dim_stride, const uint32_t output_last_dim_stride,
                                     AscendC::LocalTensor<uint8_t> &tmp_buf, const bool src1_is_one_block) {
  const uint8_t dst_repeat_stride = output_last_dim_stride / ONE_BLK_SIZE;  // 由于输出均为uint8_t，所以不用乘类型大小
  const uint8_t src_repeat_stride =
      input_last_dim_stride * sizeof(T) / ONE_BLK_SIZE;  // 这里的stride，应该由用户保证是按block对齐的
  const uint32_t elem_in_one_repeat = ONE_REPEAT_BYTE_SIZE / sizeof(T);

  // last_axis < 256B, 直接使用first_dim作为repeat_times执行
  // 临时空间不满足一条指令做完
  if (last_axis < elem_in_one_repeat) {
    CompareNormalNoLoop<T, mode>(dst, src0, src1, repeat_times, last_axis, dst_repeat_stride, src_repeat_stride,
                                 tmp_buf, src1_is_one_block);
  } else {  // 如果last_dim轴大于256B，一次repeat做不完，
    uint32_t element_extent = last_axis / elem_in_one_repeat;
    uint32_t element_reminder = last_axis - element_extent * elem_in_one_repeat;
    if (element_extent <= repeat_times) {
      uint32_t tmp_offset = 0;
      LocalTensor<uint8_t> compare_out = tmp_buf[tmp_offset].template ReinterpretCast<uint8_t>();
      compare_out.SetSize((repeat_times * elem_in_one_repeat) / 8);
      tmp_offset += (repeat_times * elem_in_one_repeat) * sizeof(uint8_t) / 8;
      LocalTensor<half> select_out =
          tmp_buf[KernelUtils::BlkAlign<uint8_t>(tmp_offset)].template ReinterpretCast<half>();
      select_out.SetSize(ONE_REPEAT_BYTE_SIZE / sizeof(half) * repeat_times);

      for (uint32_t outer_for = 0; outer_for < element_extent; outer_for++) {
        uint32_t mask = elem_in_one_repeat;
        // 处理 T = float/half, mode = NE 存在NaN场景
        // NE模式下，Compare对NaN总是返回0，需要先将NaN替换后再做Compare
        if constexpr ((AscendC::IsSameType<T, float>::value || AscendC::IsSameType<T, half>::value) &&
                      mode == CMPMODE::NE) {
          // 逐repeat处理，复用select_out空间作为nan_buf
          uint32_t select_bits_per_repeat = ONE_REPEAT_BYTE_SIZE / sizeof(half);
          // compare_out的bit数按int16对齐（Not/Or只支持int16），每个int16包含SHORT_BIT_WIDTH个比较bit
          uint32_t mask_int16_count = (elem_in_one_repeat + kShortBitWidth - 1) / kShortBitWidth;
          LocalTensor<uint8_t> nan_buf = select_out.ReinterpretCast<uint8_t>();
          LocalTensor<int16_t> nan_buf_int16 = select_out.ReinterpretCast<int16_t>();
          LocalTensor<int16_t> compare_out_int16 = compare_out.ReinterpretCast<int16_t>();
          BinaryRepeatParams src1_repeat_params = {1, 1, 1, 8, src_repeat_stride, src_repeat_stride};
          if (src1_is_one_block) {
            src1_repeat_params = {1, 1, 0, 8, src_repeat_stride, 0};
          }

          for (uint8_t r = 0; r < repeat_times; r++) {
            // Step 1: Compare(src0, src0, EQ) → nan_buf (NaN=0, 非NaN=1)
            AscendC::PipeBarrier<PIPE_V>();
            Compare(nan_buf, src0[outer_for * elem_in_one_repeat + r * elem_in_one_repeat],
                    src0[outer_for * elem_in_one_repeat + r * elem_in_one_repeat], CMPMODE::EQ, mask, 1,
                    {1, 1, 1, 8, src_repeat_stride, src_repeat_stride});
            // Step 2: Not → is_nan0
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Not(nan_buf_int16, nan_buf_int16, mask_int16_count);

            // Step 3: Compare(src1, src1, EQ) → compare_out
            AscendC::PipeBarrier<PIPE_V>();
            if (src1_is_one_block) {
              Compare(compare_out, src1[0], src1[0], CMPMODE::EQ, mask, 1, src1_repeat_params);
            } else {
              Compare(compare_out, src1[outer_for * elem_in_one_repeat + r * elem_in_one_repeat],
                      src1[outer_for * elem_in_one_repeat + r * elem_in_one_repeat], CMPMODE::EQ, mask, 1,
                      src1_repeat_params);
            }
            // Step 4: Not → is_nan1
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Not(compare_out_int16, compare_out_int16, mask_int16_count);

            // Step 5: Or → is_nan = is_nan0 | is_nan1
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Or(compare_out_int16, compare_out_int16, nan_buf_int16, mask_int16_count);

            // Step 6: Compare(src0, src1, NE) → nan_buf
            AscendC::PipeBarrier<PIPE_V>();
            if (src1_is_one_block) {
              Compare(nan_buf, src0[outer_for * elem_in_one_repeat + r * elem_in_one_repeat], src1[0], mode, mask, 1,
                      src1_repeat_params);
            } else {
              Compare(nan_buf, src0[outer_for * elem_in_one_repeat + r * elem_in_one_repeat],
                      src1[outer_for * elem_in_one_repeat + r * elem_in_one_repeat], mode, mask, 1,
                      {1, 1, 1, 8, src_repeat_stride, src_repeat_stride});
            }

            // Step 7: Or → 修正后的NE结果
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Or(compare_out_int16, nan_buf_int16, compare_out_int16, mask_int16_count);

            // Step 8: Select+Cast → dst
            AscendC::PipeBarrier<PIPE_V>();
            Duplicate<half, true>(select_out, (half)1, select_bits_per_repeat, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            Select<half, uint8_t, true>(select_out, compare_out, select_out, (half)0, SELMODE::VSEL_TENSOR_SCALAR_MODE,
                                        select_bits_per_repeat, 1, {1, 1, 1, 8, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();
            Cast<uint8_t, half, true>(dst[outer_for * elem_in_one_repeat + r * dst_repeat_stride * ONE_BLK_SIZE],
                                      select_out, RoundMode::CAST_NONE, mask, 1, {1, 1, dst_repeat_stride, 8});
          }
        } else {
          PipeBarrier<PIPE_V>();
          BinaryRepeatParams repeat_params = {1, 1, 1, 8, src_repeat_stride, src_repeat_stride};
          if (src1_is_one_block) {
            repeat_params = {1, 1, 0, 8, src_repeat_stride, 0};
            Compare(compare_out, src0[outer_for * elem_in_one_repeat], src1[0], mode, mask, repeat_times,
                    repeat_params);
          } else {
            Compare(compare_out, src0[outer_for * elem_in_one_repeat], src1[outer_for * elem_in_one_repeat], mode, mask,
                    repeat_times, repeat_params);
          }
          PipeBarrier<PIPE_V>();
          Duplicate<half, true>(select_out, 1, ONE_REPEAT_BYTE_SIZE / sizeof(half), repeat_times, 1, 8);
          AscendC::PipeBarrier<PIPE_V>();
          repeat_params = {1, 1, 1, 8, 8, 8};
          Select<half, uint8_t, true>(select_out, compare_out, select_out, (half)0, SELMODE::VSEL_TENSOR_SCALAR_MODE,
                                      ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t), repeat_times, repeat_params);
          UnaryRepeatParams unary_repeat_params = {1, 1, dst_repeat_stride, 4};
          AscendC::PipeBarrier<PIPE_V>();
          Cast<uint8_t, half, true>(dst[outer_for * elem_in_one_repeat], select_out, RoundMode::CAST_NONE, mask,
                                    repeat_times, unary_repeat_params);
        }
      }
      if (element_reminder != 0) {
        uint32_t mask = element_reminder;
        // 处理 T = float/half, mode = NE 存在NaN场景
        if constexpr ((AscendC::IsSameType<T, float>::value || AscendC::IsSameType<T, half>::value) &&
                      mode == CMPMODE::NE) {
          uint32_t select_bits_per_repeat = ONE_REPEAT_BYTE_SIZE / sizeof(half);
          uint32_t mask_int16_count = (elem_in_one_repeat + kShortBitWidth - 1) / kShortBitWidth;
          LocalTensor<uint8_t> nan_buf = select_out.ReinterpretCast<uint8_t>();
          LocalTensor<int16_t> nan_buf_int16 = select_out.ReinterpretCast<int16_t>();
          LocalTensor<int16_t> compare_out_int16 = compare_out.ReinterpretCast<int16_t>();
          BinaryRepeatParams src1_repeat_params = {1, 1, 1, 8, src_repeat_stride, src_repeat_stride};
          if (src1_is_one_block) {
            src1_repeat_params = {1, 1, 0, 8, src_repeat_stride, 0};
          }

          for (uint8_t r = 0; r < repeat_times; r++) {
            // Step 1: Compare(src0, src0, EQ) → nan_buf
            AscendC::PipeBarrier<PIPE_V>();
            Compare(nan_buf, src0[element_extent * elem_in_one_repeat + r * elem_in_one_repeat],
                    src0[element_extent * elem_in_one_repeat + r * elem_in_one_repeat], CMPMODE::EQ, mask, 1,
                    {1, 1, 1, 8, src_repeat_stride, src_repeat_stride});
            // Step 2: Not → is_nan0
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Not(nan_buf_int16, nan_buf_int16, mask_int16_count);

            // Step 3: Compare(src1, src1, EQ) → compare_out
            AscendC::PipeBarrier<PIPE_V>();
            if (src1_is_one_block) {
              Compare(compare_out, src1[0], src1[0], CMPMODE::EQ, mask, 1, src1_repeat_params);
            } else {
              Compare(compare_out, src1[element_extent * elem_in_one_repeat + r * elem_in_one_repeat],
                      src1[element_extent * elem_in_one_repeat + r * elem_in_one_repeat], CMPMODE::EQ, mask, 1,
                      src1_repeat_params);
            }
            // Step 4: Not → is_nan1
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Not(compare_out_int16, compare_out_int16, mask_int16_count);

            // Step 5: Or → is_nan
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Or(compare_out_int16, compare_out_int16, nan_buf_int16, mask_int16_count);

            // Step 6: Compare(src0, src1, NE) → nan_buf
            AscendC::PipeBarrier<PIPE_V>();
            if (src1_is_one_block) {
              Compare(nan_buf, src0[element_extent * elem_in_one_repeat + r * elem_in_one_repeat], src1[0], mode, mask,
                      1, src1_repeat_params);
            } else {
              Compare(nan_buf, src0[element_extent * elem_in_one_repeat + r * elem_in_one_repeat],
                      src1[element_extent * elem_in_one_repeat + r * elem_in_one_repeat], mode, mask, 1,
                      {1, 1, 1, 8, src_repeat_stride, src_repeat_stride});
            }

            // Step 7: Or → 修正后的NE结果
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Or(compare_out_int16, nan_buf_int16, compare_out_int16, mask_int16_count);

            // Step 8: Select+Cast → dst
            AscendC::PipeBarrier<PIPE_V>();
            Duplicate<half, true>(select_out, (half)1, select_bits_per_repeat, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            Select<half, uint8_t, true>(select_out, compare_out, select_out, (half)0, SELMODE::VSEL_TENSOR_SCALAR_MODE,
                                        select_bits_per_repeat, 1, {1, 1, 1, 8, 8, 8});
            AscendC::PipeBarrier<PIPE_V>();
            Cast<uint8_t, half, true>(dst[element_extent * elem_in_one_repeat + r * dst_repeat_stride * ONE_BLK_SIZE],
                                      select_out, RoundMode::CAST_NONE, mask, 1, {1, 1, dst_repeat_stride, 8});
          }
        } else {
          AscendC::PipeBarrier<PIPE_V>();
          BinaryRepeatParams repeat_params = {1, 1, 1, 8, src_repeat_stride, src_repeat_stride};
          if (src1_is_one_block) {
            repeat_params = {1, 1, 0, 8, src_repeat_stride, 0};
            Compare(compare_out, src0[element_extent * elem_in_one_repeat], src1[0], mode, mask, repeat_times,
                    repeat_params);
          } else {
            Compare(compare_out, src0[element_extent * elem_in_one_repeat], src1[element_extent * elem_in_one_repeat],
                    mode, mask, repeat_times, repeat_params);
          }
          PipeBarrier<PIPE_V>();
          Duplicate<half, true>(select_out, 1, ONE_REPEAT_BYTE_SIZE / sizeof(half), repeat_times, 1, 8);
          repeat_params = {1, 1, 1, 8, 8, 8};
          AscendC::PipeBarrier<PIPE_V>();
          Select<half, uint8_t, true>(select_out, compare_out, select_out, (half)0, SELMODE::VSEL_TENSOR_SCALAR_MODE,
                                      ONE_REPEAT_BYTE_SIZE / sizeof(uint8_t), repeat_times, repeat_params);
          UnaryRepeatParams unary_repeat_params = {1, 1, dst_repeat_stride, 4};
          AscendC::PipeBarrier<PIPE_V>();
          Cast<uint8_t, half, true>(dst[element_extent * elem_in_one_repeat], select_out, RoundMode::CAST_NONE, mask,
                                    repeat_times, unary_repeat_params);
        }
      }
    } else {
      uint32_t tmp_offset = 0;
      LocalTensor<uint8_t> compare_out = tmp_buf[tmp_offset].template ReinterpretCast<uint8_t>();
      compare_out.SetSize(KernelUtils::BlkAlign<uint8_t>(last_axis) / 8);
      tmp_offset += (KernelUtils::BlkAlign<uint8_t>(last_axis)) * sizeof(uint8_t) / 8;
      LocalTensor<half> select_out =
          tmp_buf[KernelUtils::BlkAlign<uint8_t>(tmp_offset)].template ReinterpretCast<half>();
      select_out.SetSize(last_axis);
      for (auto outer_for = 0; outer_for < repeat_times; outer_for++) {
        // 处理 T = float/half, mode = NE 存在NaN场景
        if constexpr ((AscendC::IsSameType<T, float>::value || AscendC::IsSameType<T, half>::value) &&
                      mode == CMPMODE::NE) {
          LocalTensor<uint8_t> nan_buf = select_out.ReinterpretCast<uint8_t>();
          LocalTensor<int16_t> nan_buf_int16 = select_out.ReinterpretCast<int16_t>();
          LocalTensor<int16_t> compare_out_int16 = compare_out.ReinterpretCast<int16_t>();
          // last_axis个元素 → ceil(last_axis/kShortBitWidth)个int16元素
          uint32_t mask_int16_count = (last_axis + kShortBitWidth - 1) / kShortBitWidth;

          // Step 1: Compare(src0, src0, EQ) → nan_buf
          AscendC::PipeBarrier<PIPE_V>();
          Compare(nan_buf, src0[outer_for * input_last_dim_stride], src0[outer_for * input_last_dim_stride],
                  CMPMODE::EQ, last_axis);
          // Step 2: Not → is_nan0
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Not(nan_buf_int16, nan_buf_int16, mask_int16_count);

          // Step 3: Compare(src1, src1, EQ) → compare_out
          AscendC::PipeBarrier<PIPE_V>();
          if (src1_is_one_block) {
            Compare(compare_out, src1, src1, CMPMODE::EQ, ONE_BLK_SIZE / sizeof(T));
          } else {
            Compare(compare_out, src1[outer_for * input_last_dim_stride], src1[outer_for * input_last_dim_stride],
                    CMPMODE::EQ, last_axis);
          }
          // Step 4: Not → is_nan1
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Not(compare_out_int16, compare_out_int16, mask_int16_count);

          // Step 5: Or → is_nan
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Or(compare_out_int16, compare_out_int16, nan_buf_int16, mask_int16_count);

          // Step 6: Compare(src0, src1, NE) → nan_buf
          AscendC::PipeBarrier<PIPE_V>();
          if (src1_is_one_block) {
            uint32_t calcount_aligned = (last_axis + elem_in_one_repeat - 1) / elem_in_one_repeat * elem_in_one_repeat;
            Compare(nan_buf, src0[outer_for * input_last_dim_stride], src1, mode, calcount_aligned);
          } else {
            uint32_t calcount_aligned = (last_axis + elem_in_one_repeat - 1) / elem_in_one_repeat * elem_in_one_repeat;
            Compare(nan_buf, src0[outer_for * input_last_dim_stride], src1[outer_for * input_last_dim_stride], mode,
                    calcount_aligned);
          }

          // Step 7: Or → 修正后的NE结果
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Or(compare_out_int16, nan_buf_int16, compare_out_int16, mask_int16_count);
        } else {
          PipeBarrier<PIPE_V>();
          if (src1_is_one_block) {
            event_t eventid_v_to_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventid_v_to_s);
            WaitFlag<HardEvent::V_S>(eventid_v_to_s);
            auto scalar_value = src1.GetValue(0);
            event_t eventid_s_to_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventid_s_to_v);
            WaitFlag<HardEvent::S_V>(eventid_s_to_v);
            CompareScalar(compare_out, src0[outer_for * input_last_dim_stride], scalar_value, mode, last_axis);
          } else {
            uint32_t calcount_aligned = (last_axis + elem_in_one_repeat - 1) / elem_in_one_repeat * elem_in_one_repeat;
            Compare(compare_out, src0[outer_for * input_last_dim_stride], src1[outer_for * input_last_dim_stride], mode,
                    calcount_aligned);
          }
        }
        PipeBarrier<PIPE_V>();
        Duplicate<half>(select_out, static_cast<half>(1), last_axis);
        AscendC::PipeBarrier<PIPE_V>();
        Select<half, uint8_t>(select_out, compare_out, select_out, (half)0, SELMODE::VSEL_TENSOR_SCALAR_MODE,
                              last_axis);
        AscendC::PipeBarrier<PIPE_V>();
        Cast<uint8_t, half>(dst[outer_for * output_last_dim_stride], select_out, RoundMode::CAST_NONE, last_axis);
      }
    }
  }
}

template <CMPMODE mode>
inline __aicore__ void ApplyCompareModeNormal(LocalTensor<int32_t> &inter_buf, uint32_t mask, uint8_t repeat_times,
                                              const UnaryRepeatParams &unary_repeat_params) {
  // 根据比较模式进行不同的后处理
  switch (mode) {
    case CMPMODE::GE:
      AscendC::Adds(inter_buf, inter_buf, (int32_t)1, mask, repeat_times, unary_repeat_params);
      AscendC::PipeBarrier<PIPE_V>();
    case CMPMODE::GT:
      AscendC::Maxs(inter_buf, inter_buf, (int32_t)0, mask, repeat_times, unary_repeat_params);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mins(inter_buf, inter_buf, (int32_t)1, mask, repeat_times, unary_repeat_params);
      break;

    case CMPMODE::LE:
      AscendC::Adds(inter_buf, inter_buf, (int32_t)(-1), mask, repeat_times, unary_repeat_params);
      AscendC::PipeBarrier<PIPE_V>();
    case CMPMODE::LT:
    case CMPMODE::NE:
      AscendC::Maxs(inter_buf, inter_buf, (int32_t)(-1), mask, repeat_times, unary_repeat_params);
      AscendC::PipeBarrier<PIPE_V>();
      if (mode == CMPMODE::NE) {
        AscendC::Mins(inter_buf, inter_buf, (int32_t)1, mask, repeat_times, unary_repeat_params);
      } else {
        AscendC::Mins(inter_buf, inter_buf, (int32_t)0, mask, repeat_times, unary_repeat_params);
      }
      AscendC::PipeBarrier<PIPE_V>();
      BinaryRepeatParams binary_repeat_params = {1, 1, 1, 8, 8, 8};
      AscendC::Mul(inter_buf, inter_buf, inter_buf, mask, repeat_times, binary_repeat_params);
      break;
  }
  AscendC::PipeBarrier<PIPE_V>();
}

template <CMPMODE mode>
inline __aicore__ void ApplyCompareModeCount(LocalTensor<int32_t> &inter_buf, uint32_t count) {
  // 根据比较模式进行不同的后处理
  switch (mode) {
    case CMPMODE::GE:
      AscendC::Adds(inter_buf, inter_buf, (int32_t)1, count);
      AscendC::PipeBarrier<PIPE_V>();
    case CMPMODE::GT:
      AscendC::Maxs(inter_buf, inter_buf, (int32_t)0, count);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mins(inter_buf, inter_buf, (int32_t)1, count);
      break;

    case CMPMODE::LE:
      AscendC::Adds(inter_buf, inter_buf, (int32_t)(-1), count);
      AscendC::PipeBarrier<PIPE_V>();
    case CMPMODE::LT:
    case CMPMODE::NE:
      AscendC::Maxs(inter_buf, inter_buf, (int32_t)(-1), count);
      AscendC::PipeBarrier<PIPE_V>();
      if (mode == CMPMODE::NE) {
        AscendC::Mins(inter_buf, inter_buf, (int32_t)1, count);
      } else {
        AscendC::Mins(inter_buf, inter_buf, (int32_t)0, count);
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul(inter_buf, inter_buf, inter_buf, count);
      break;
  }
  AscendC::PipeBarrier<PIPE_V>();
}

inline __aicore__ void PerformTypeConversionNormal(const AscendC::LocalTensor<uint8_t> &dst,
                                                   AscendC::LocalTensor<int16_t> &int16_buf,
                                                   AscendC::LocalTensor<half> &half_buf,
                                                   AscendC::LocalTensor<int32_t> &inter_buf, uint32_t mask,
                                                   uint8_t repeat_times, uint8_t dst_repeat_stride,
                                                   uint32_t dst_offset = 0) {
  UnaryRepeatParams unary_repeat_params = {1, 1, 4, 8};
  AscendC::Cast(int16_buf, inter_buf, RoundMode::CAST_NONE, mask, repeat_times, unary_repeat_params);

  unary_repeat_params = {1, 1, 4, 4};
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::Cast(half_buf, int16_buf, RoundMode::CAST_NONE, mask, repeat_times, unary_repeat_params);

  unary_repeat_params = {1, 1, dst_repeat_stride, 4};
  AscendC::PipeBarrier<PIPE_V>();
  if (dst_offset == 0) {
    AscendC::Cast(dst, half_buf, RoundMode::CAST_NONE, mask, repeat_times, unary_repeat_params);
  } else {
    AscendC::Cast(dst[dst_offset], half_buf, RoundMode::CAST_NONE, mask, repeat_times, unary_repeat_params);
  }
}

inline __aicore__ void PerformTypeConversionCount(const AscendC::LocalTensor<uint8_t> &dst,
                                                  AscendC::LocalTensor<int16_t> &int16_buf,
                                                  AscendC::LocalTensor<half> &half_buf,
                                                  AscendC::LocalTensor<int32_t> &inter_buf, uint32_t num_elements,
                                                  uint32_t dst_offset = 0) {
  AscendC::Cast(int16_buf, inter_buf, RoundMode::CAST_NONE, num_elements);
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::Cast(half_buf, int16_buf, RoundMode::CAST_NONE, num_elements);
  AscendC::PipeBarrier<PIPE_V>();
  if (dst_offset == 0) {
    AscendC::Cast(dst, half_buf, RoundMode::CAST_NONE, num_elements);
  } else {
    AscendC::Cast(dst[dst_offset], half_buf, RoundMode::CAST_NONE, num_elements);
  }
}

template <CMPMODE mode>
inline __aicore__ void ProcessSmallAxisCase(const AscendC::LocalTensor<uint8_t> &dst,
                                            const AscendC::LocalTensor<int32_t> &src0,
                                            const AscendC::LocalTensor<int32_t> &src1, const uint8_t repeat_times,
                                            const uint64_t mask, const uint8_t src_repeat_stride,
                                            const uint8_t dst_repeat_stride, AscendC::LocalTensor<int32_t> &inter_buf,
                                            AscendC::LocalTensor<int16_t> &int16_buf,
                                            AscendC::LocalTensor<half> &half_buf, const bool src1_is_one_block) {
  AscendC::PipeBarrier<PIPE_V>();
  BinaryRepeatParams repeat_params = {1, 1, 1, 8, src_repeat_stride, src_repeat_stride};
  if (src1_is_one_block) {
    repeat_params = {1, 1, 0, 8, src_repeat_stride, 0};
  }

  AscendC::Sub(inter_buf, src0, src1, mask, repeat_times, repeat_params);
  AscendC::PipeBarrier<PIPE_V>();

  UnaryRepeatParams unary_repeat_params = {1, 1, 8, 8};
  ApplyCompareModeNormal<mode>(inter_buf, mask, repeat_times, unary_repeat_params);

  PerformTypeConversionNormal(dst, int16_buf, half_buf, inter_buf, mask, repeat_times, dst_repeat_stride);
}

template <CMPMODE mode>
inline __aicore__ void ProcessBlock(const AscendC::LocalTensor<uint8_t> &dst, const AscendC::LocalTensor<int32_t> &src0,
                                    const AscendC::LocalTensor<int32_t> &src1, const uint8_t repeat_times,
                                    const uint32_t mask, const uint8_t src_repeat_stride,
                                    const uint8_t dst_repeat_stride, AscendC::LocalTensor<int32_t> &inter_buf,
                                    AscendC::LocalTensor<int16_t> &int16_buf, AscendC::LocalTensor<half> &half_buf,
                                    uint32_t offset_src0, uint32_t offset_src1, uint32_t offset_dst,
                                    const bool src1_is_one_block) {
  BinaryRepeatParams repeat_params = {1, 1, 1, 8, src_repeat_stride, src_repeat_stride};
  AscendC::PipeBarrier<PIPE_V>();

  if (src1_is_one_block) {
    repeat_params = {1, 1, 0, 8, src_repeat_stride, 0};
    AscendC::Sub(inter_buf, src0[offset_src0], src1[0], mask, repeat_times, repeat_params);
  } else {
    AscendC::Sub(inter_buf, src0[offset_src0], src1[offset_src1], mask, repeat_times, repeat_params);
  }

  UnaryRepeatParams unary_repeat_params = {1, 1, 8, 8};
  AscendC::PipeBarrier<PIPE_V>();
  ApplyCompareModeNormal<mode>(inter_buf, mask, repeat_times, unary_repeat_params);

  PerformTypeConversionNormal(dst, int16_buf, half_buf, inter_buf, mask, repeat_times, dst_repeat_stride, offset_dst);
}

template <CMPMODE mode>
inline __aicore__ void ProcessMediumAxisCase(const AscendC::LocalTensor<uint8_t> &dst,
                                             const AscendC::LocalTensor<int32_t> &src0,
                                             const AscendC::LocalTensor<int32_t> &src1, const uint8_t repeat_times,
                                             const uint32_t last_axis, const uint32_t input_last_dim_stride,
                                             const uint32_t output_last_dim_stride,
                                             AscendC::LocalTensor<int32_t> &inter_buf,
                                             AscendC::LocalTensor<int16_t> &int16_buf,
                                             AscendC::LocalTensor<half> &half_buf, const bool src1_is_one_block) {
  constexpr uint32_t elem_in_one_repeat = ONE_REPEAT_BYTE_SIZE / sizeof(int32_t);
  const uint8_t src_repeat_stride = input_last_dim_stride * sizeof(int32_t) / ONE_BLK_SIZE;
  const uint8_t dst_repeat_stride = output_last_dim_stride / ONE_BLK_SIZE;

  uint32_t element_extent = last_axis / elem_in_one_repeat;
  uint32_t element_reminder = last_axis - element_extent * elem_in_one_repeat;

  for (uint32_t outer_for = 0; outer_for < element_extent; outer_for++) {
    constexpr uint32_t mask = elem_in_one_repeat;
    ProcessBlock<mode>(dst, src0, src1, repeat_times, mask, src_repeat_stride, dst_repeat_stride, inter_buf, int16_buf,
                       half_buf, outer_for * elem_in_one_repeat, outer_for * elem_in_one_repeat,
                       outer_for * elem_in_one_repeat, src1_is_one_block);
  }

  if (element_reminder != 0) {
    uint32_t mask = element_reminder;
    ProcessBlock<mode>(dst, src0, src1, repeat_times, mask, src_repeat_stride, dst_repeat_stride, inter_buf, int16_buf,
                       half_buf, element_extent * elem_in_one_repeat, element_extent * elem_in_one_repeat,
                       element_extent * elem_in_one_repeat, src1_is_one_block);
  }
}

template <CMPMODE mode>
inline __aicore__ void ProcessLargeAxisCase(const AscendC::LocalTensor<uint8_t> &dst,
                                            const AscendC::LocalTensor<int32_t> &src0,
                                            const AscendC::LocalTensor<int32_t> &src1, const uint8_t repeat_times,
                                            const uint32_t last_axis, const uint32_t input_last_dim_stride,
                                            const uint32_t output_last_dim_stride,
                                            AscendC::LocalTensor<int32_t> &inter_buf,
                                            AscendC::LocalTensor<int16_t> &int16_buf,
                                            AscendC::LocalTensor<half> &half_buf, const bool src1_is_one_block) {
  constexpr uint32_t elem_in_one_repeat = ONE_REPEAT_BYTE_SIZE / sizeof(int32_t);
  uint32_t element_extent = last_axis / elem_in_one_repeat;
  uint32_t element_reminder = last_axis - element_extent * elem_in_one_repeat;
  const uint8_t src_repeat_stride = input_last_dim_stride * sizeof(int32_t) / ONE_BLK_SIZE;

  for (auto outer_for = 0; outer_for < repeat_times; outer_for++) {
    AscendC::PipeBarrier<PIPE_V>();

    if (src1_is_one_block) {
      BinaryRepeatParams repeat_params = {1, 1, 0, 8, src_repeat_stride, 0};
      AscendC::Sub(inter_buf, src0[outer_for * input_last_dim_stride], src1[0], elem_in_one_repeat, element_extent,
                   repeat_params);

      if (element_reminder != 0) {
        AscendC::Sub(inter_buf[element_extent * elem_in_one_repeat],
                     src0[outer_for * input_last_dim_stride + element_extent * elem_in_one_repeat], src1[0],
                     element_reminder, 1, repeat_params);
      }
    } else {
      AscendC::Sub(inter_buf, src0[outer_for * input_last_dim_stride], src1[outer_for * input_last_dim_stride],
                   last_axis);
    }

    AscendC::PipeBarrier<PIPE_V>();
    ApplyCompareModeCount<mode>(inter_buf, last_axis);

    // 额外操作：Maxs和Mins
    AscendC::Maxs(inter_buf, inter_buf, (int32_t)0, last_axis);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mins(inter_buf, inter_buf, (int32_t)1, last_axis);
    AscendC::PipeBarrier<PIPE_V>();

    PerformTypeConversionCount(dst, int16_buf, half_buf, inter_buf, last_axis, outer_for * output_last_dim_stride);
  }
}

template <CMPMODE mode>
inline __aicore__ void CompareExtendInt32(const AscendC::LocalTensor<uint8_t> &dst,
                                          const AscendC::LocalTensor<int32_t> &src0,
                                          const AscendC::LocalTensor<int32_t> &src1, const uint8_t repeat_times,
                                          const uint32_t last_axis, const uint32_t input_last_dim_stride,
                                          const uint32_t output_last_dim_stride, AscendC::LocalTensor<uint8_t> &tmp_buf,
                                          const bool src1_is_one_block) {
  const uint8_t dst_repeat_stride = output_last_dim_stride / ONE_BLK_SIZE;
  const uint8_t src_repeat_stride = input_last_dim_stride * sizeof(int32_t) / ONE_BLK_SIZE;
  constexpr uint32_t elem_in_one_repeat = ONE_REPEAT_BYTE_SIZE / sizeof(int32_t);

  LocalTensor<int32_t> inter_buf = tmp_buf.ReinterpretCast<int32_t>();
  LocalTensor<int16_t> int16_buf = inter_buf.ReinterpretCast<int16_t>();
  LocalTensor<half> half_buf = inter_buf.ReinterpretCast<half>();

  if (last_axis < elem_in_one_repeat) {
    const uint64_t mask = last_axis;
    ProcessSmallAxisCase<mode>(dst, src0, src1, repeat_times, mask, src_repeat_stride, dst_repeat_stride, inter_buf,
                               int16_buf, half_buf, src1_is_one_block);
  } else {
    uint32_t element_extent = last_axis / elem_in_one_repeat;
    if (element_extent <= repeat_times) {
      ProcessMediumAxisCase<mode>(dst, src0, src1, repeat_times, last_axis, input_last_dim_stride,
                                  output_last_dim_stride, inter_buf, int16_buf, half_buf, src1_is_one_block);
    } else {
      ProcessLargeAxisCase<mode>(dst, src0, src1, repeat_times, last_axis, input_last_dim_stride,
                                 output_last_dim_stride, inter_buf, int16_buf, half_buf, src1_is_one_block);
    }
  }
}

template <CMPMODE mode>
inline __aicore__ void CompareExtendInt64EqNe(const AscendC::LocalTensor<uint8_t> &dst,   // output
                                              const AscendC::LocalTensor<int64_t> &src0,  // input 0 is a tensor
                                              const AscendC::LocalTensor<int64_t> &src1,  // input 1 is a tensor
                                              const uint8_t repeat_times, const uint32_t last_axis,
                                              const uint32_t input_last_dim_stride,
                                              const uint32_t output_last_dim_stride,
                                              AscendC::LocalTensor<uint8_t> &tmp_buf, const bool src1_is_one_block) {
  const uint8_t dst_repeat_stride = output_last_dim_stride / ONE_BLK_SIZE;  // 由于输出均为uint8_t，所以不用乘类型大小
  const uint8_t src_repeat_stride = input_last_dim_stride * sizeof(int64_t) / ONE_BLK_SIZE;
  LocalTensor<half> all_one_buf = KernelUtils::NewTensor<half>(tmp_buf, 0, KernelUtils::RptSize<half>());
  Duplicate(all_one_buf, (half)1.0f, KernelUtils::RptSize<half>());

  LocalTensor<int32_t> sub_res_buf = KernelUtils::NewTensor<int32_t>(
      tmp_buf, ONE_REPEAT_BYTE_SIZE, ONE_REPEAT_BYTE_SIZE * repeat_times / sizeof(int32_t));
  LocalTensor<int64_t> sub_res_64_buf = sub_res_buf.ReinterpretCast<int64_t>();
  LocalTensor<uint8_t> sub_res_reuse_buf = sub_res_buf.template ReinterpretCast<uint8_t>();
  LocalTensor<float> cast_res_buf = KernelUtils::NewTensor<float>(tmp_buf, ONE_REPEAT_BYTE_SIZE * (1 + repeat_times),
                                                                  ONE_REPEAT_BYTE_SIZE * repeat_times / sizeof(float));
  LocalTensor<uint8_t> cmp_res_buf =
      KernelUtils::NewTensor<uint8_t>(sub_res_reuse_buf, 0, (ONE_REPEAT_BYTE_SIZE * repeat_times) / 8);
  LocalTensor<half> sel_res_buf = KernelUtils::NewTensor<half>(
      sub_res_reuse_buf, (ONE_REPEAT_BYTE_SIZE * repeat_times) / 8, ONE_REPEAT_BYTE_SIZE * repeat_times / sizeof(half));

  constexpr uint32_t elem_in_one_repeat = ONE_REPEAT_BYTE_SIZE / sizeof(int64_t);

  LocalTensor<int32_t> src0_new = src0.ReinterpretCast<int32_t>();
  LocalTensor<int32_t> src1_new = src1.ReinterpretCast<int32_t>();

  // last_axis < 256B, 直接使用first_dim作为repeat_times执行
  // 临时空间不满足一条指令做完
  if (last_axis <= elem_in_one_repeat) {
    const uint64_t mask = last_axis;
    BinaryRepeatParams repeat_params = {1, 1, 1, 8, src_repeat_stride, src_repeat_stride};
    AscendC::PipeBarrier<PIPE_V>();
    if (src1_is_one_block) {
      repeat_params = {1, 1, 0, 8, src_repeat_stride, 0};
    }
    AscendC::Sub(sub_res_buf, src0_new, src1_new, mask * 2, repeat_times, repeat_params);
    UnaryRepeatParams unary_repeat_params = {1, 1, 8, 8};
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(cast_res_buf, sub_res_64_buf, AscendC::RoundMode::CAST_RINT, mask, repeat_times, unary_repeat_params);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::CompareScalar(cmp_res_buf, cast_res_buf, 0.0f, mode, mask, repeat_times, unary_repeat_params);
    repeat_params = {1, 1, 1, 8, 0, 8};
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Select<half, uint8_t, true>(sel_res_buf, cmp_res_buf, all_one_buf, (half)0,
                                         AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, KernelUtils::RptSize<half>(),
                                         repeat_times, repeat_params);
    AscendC::PipeBarrier<PIPE_V>();
    unary_repeat_params = {1, 1, dst_repeat_stride, 4};
    AscendC::Cast(dst, sel_res_buf, AscendC::RoundMode::CAST_NONE, mask, repeat_times, unary_repeat_params);
    AscendC::PipeBarrier<PIPE_V>();
  } else {  // 如果last_dim轴大于256B，一次repeat做不完，
    uint32_t element_extent = last_axis / elem_in_one_repeat;
    uint32_t element_reminder = last_axis - element_extent * elem_in_one_repeat;
    if (element_extent <= repeat_times) {
      for (uint32_t outer_for = 0; outer_for < element_extent; outer_for++) {
        constexpr uint64_t mask = elem_in_one_repeat;
        BinaryRepeatParams repeat_params = {1, 1, 1, 8, src_repeat_stride, src_repeat_stride};
        AscendC::PipeBarrier<PIPE_V>();
        if (src1_is_one_block) {
          repeat_params = {1, 1, 0, 8, src_repeat_stride, 0};
          AscendC::Sub(sub_res_buf, src0_new[outer_for * elem_in_one_repeat * 2], src1_new[0], mask * 2, repeat_times,
                       repeat_params);
        } else {
          AscendC::Sub(sub_res_buf, src0_new[outer_for * elem_in_one_repeat * 2],
                       src1_new[outer_for * elem_in_one_repeat * 2], mask * 2, repeat_times, repeat_params);
        }
        UnaryRepeatParams unary_repeat_params = {1, 1, 4, 8};
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(cast_res_buf, sub_res_64_buf, AscendC::RoundMode::CAST_RINT, mask, repeat_times,
                      unary_repeat_params);
        unary_repeat_params = {1, 1, 8, 8};
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::CompareScalar(cmp_res_buf, cast_res_buf, 0.0f, mode, mask, DivCeil(repeat_times, 2),
                               unary_repeat_params);
        repeat_params = {1, 0, 0, 8, 0, 0};
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select<half, uint8_t, true>(sel_res_buf, cmp_res_buf, all_one_buf, (half)0,
                                             AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, 128, DivCeil(repeat_times, 4),
                                             repeat_params);
        AscendC::PipeBarrier<PIPE_V>();
        unary_repeat_params = {1, 1, dst_repeat_stride, 2};
        AscendC::Cast(dst[outer_for * elem_in_one_repeat], sel_res_buf, AscendC::RoundMode::CAST_NONE, mask,
                      repeat_times, unary_repeat_params);
        AscendC::PipeBarrier<PIPE_V>();
      }
      if (element_reminder != 0) {
        uint32_t mask = element_reminder;
        BinaryRepeatParams repeat_params = {1, 1, 1, 8, src_repeat_stride, src_repeat_stride};
        AscendC::PipeBarrier<PIPE_V>();
        if (src1_is_one_block) {
          repeat_params = {1, 1, 0, 8, src_repeat_stride, 0};
          AscendC::Sub(sub_res_buf, src0_new[element_extent * elem_in_one_repeat * 2], src1_new[0], mask * 2,
                       repeat_times, repeat_params);
        } else {
          AscendC::Sub(sub_res_buf, src0_new[element_extent * elem_in_one_repeat * 2],
                       src1_new[element_extent * elem_in_one_repeat * 2], mask * 2, repeat_times, repeat_params);
        }
        UnaryRepeatParams unary_repeat_params = {1, 1, 4, 8};
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(cast_res_buf, sub_res_64_buf, AscendC::RoundMode::CAST_RINT, mask, repeat_times,
                      unary_repeat_params);
        unary_repeat_params = {1, 1, 8, 8};
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::CompareScalar(cmp_res_buf, cast_res_buf, 0.0f, mode, mask, DivCeil(repeat_times, 2),
                               unary_repeat_params);
        repeat_params = {1, 0, 0, 8, 0, 0};
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select<half, uint8_t, true>(sel_res_buf, cmp_res_buf, all_one_buf, (half)0,
                                             AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, 128, DivCeil(repeat_times, 4),
                                             repeat_params);
        AscendC::PipeBarrier<PIPE_V>();
        unary_repeat_params = {1, 1, dst_repeat_stride, 2};
        AscendC::Cast(dst[element_extent * elem_in_one_repeat], sel_res_buf, AscendC::RoundMode::CAST_NONE, mask,
                      repeat_times, unary_repeat_params);
        AscendC::PipeBarrier<PIPE_V>();
      }
    } else {
      element_reminder = last_axis - element_extent * elem_in_one_repeat;
      for (uint32_t outer_for = 0; outer_for < repeat_times; outer_for++) {
        uint64_t mask = elem_in_one_repeat;
        BinaryRepeatParams repeat_params = {1, 1, 1, 8, 8, 8};
        AscendC::PipeBarrier<PIPE_V>();
        if (src1_is_one_block) {
          repeat_params = {1, 1, 0, 8, 8, 0};
          AscendC::Sub(sub_res_buf, src0_new[outer_for * input_last_dim_stride * 2], src1_new[0], mask * 2,
                       element_extent, repeat_params);
        } else {
          AscendC::Sub(sub_res_buf, src0_new[outer_for * input_last_dim_stride * 2],
                       src1_new[outer_for * input_last_dim_stride * 2], mask * 2, element_extent, repeat_params);
        }
        UnaryRepeatParams unary_repeat_params = {1, 1, 4, 8};
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(cast_res_buf, sub_res_64_buf, AscendC::RoundMode::CAST_RINT, mask, element_extent,
                      unary_repeat_params);
        unary_repeat_params = {1, 1, 8, 8};
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::CompareScalar(cmp_res_buf, cast_res_buf, 0.0f, mode, mask, DivCeil(element_extent, 2),
                               unary_repeat_params);
        repeat_params = {1, 0, 0, 8, 0, 0};
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select<half, uint8_t, true>(sel_res_buf, cmp_res_buf, all_one_buf, (half)0,
                                             AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, 128, DivCeil(element_extent, 4),
                                             repeat_params);
        AscendC::PipeBarrier<PIPE_V>();
        unary_repeat_params = {1, 1, 1, 2};
        AscendC::Cast(dst[outer_for * output_last_dim_stride], sel_res_buf, AscendC::RoundMode::CAST_NONE, mask,
                      element_extent, unary_repeat_params);
        AscendC::PipeBarrier<PIPE_V>();
        if (element_reminder != 0) {
          mask = element_reminder;
          BinaryRepeatParams repeat_params = {1, 1, 1, 8, 8, 8};
          AscendC::PipeBarrier<PIPE_V>();
          if (src1_is_one_block) {
            repeat_params = {1, 1, 0, 8, 8, 0};
            AscendC::Sub(sub_res_buf,
                         src0_new[outer_for * input_last_dim_stride * 2 + element_extent * elem_in_one_repeat * 2],
                         src1_new[0], mask * 2, 1, repeat_params);
          } else {
            AscendC::Sub(sub_res_buf,
                         src0_new[outer_for * input_last_dim_stride * 2 + element_extent * elem_in_one_repeat * 2],
                         src1_new[outer_for * input_last_dim_stride * 2 + element_extent * elem_in_one_repeat * 2],
                         mask * 2, 1, repeat_params);
          }
          UnaryRepeatParams unary_repeat_params = {1, 1, 4, 8};
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Cast(cast_res_buf, sub_res_64_buf, AscendC::RoundMode::CAST_RINT, mask, 1, unary_repeat_params);
          unary_repeat_params = {1, 1, 8, 8};
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::CompareScalar(cmp_res_buf, cast_res_buf, 0.0f, mode, mask, 1, unary_repeat_params);
          repeat_params = {1, 0, 0, 8, 0, 0};
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Select<half, uint8_t, true>(sel_res_buf, cmp_res_buf, all_one_buf, (half)0,
                                               AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, 128, 1, repeat_params);
          AscendC::PipeBarrier<PIPE_V>();
          unary_repeat_params = {1, 1, 1, 2};
          AscendC::Cast(dst[outer_for * output_last_dim_stride + element_extent * elem_in_one_repeat], sel_res_buf,
                        AscendC::RoundMode::CAST_NONE, mask, 1, unary_repeat_params);
          AscendC::PipeBarrier<PIPE_V>();
        }
      }
    }
  }
}

inline __aicore__ void GetSignBitTensorNormal(const AscendC::LocalTensor<uint16_t> &dst,
                                              const AscendC::LocalTensor<int64_t> &src,
                                              const AscendC::LocalTensor<uint32_t> &inner_dup, const uint32_t mask,
                                              const uint32_t repeat_times, const uint8_t src_repeat_stride,
                                              const uint8_t dst_stride, const bool src_is_one_block) {
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::Duplicate(inner_dup, uint32_t(0x80000000), inner_dup.GetSize());
  AscendC::LocalTensor<uint16_t> inner_dup_tmp = inner_dup.ReinterpretCast<uint16_t>();
  AscendC::LocalTensor<uint16_t> src_tmp = src.ReinterpretCast<uint16_t>();
  AscendC::PipeBarrier<PIPE_V>();
  BinaryRepeatParams repeat_params = {1, 1, 1, dst_stride, src_repeat_stride, 0};
  if (src_is_one_block) {
    repeat_params = {1, 0, 0, dst_stride, 0, 0};
  }
  AscendC::And(dst, src_tmp, inner_dup_tmp, mask * 4, repeat_times, repeat_params);
  AscendC::LocalTensor<int32_t> dst_tmp = dst.ReinterpretCast<int32_t>();
  AscendC::PipeBarrier<PIPE_V>();
  UnaryRepeatParams unary_repeat_params = {1, 1, dst_stride, dst_stride};
  AscendC::Maxs(dst_tmp, dst_tmp, -1, mask * 2, repeat_times, unary_repeat_params);
  AscendC::Duplicate(inner_dup, uint32_t(0xFFFFFFFF), inner_dup.GetSize());
  uint64_t mask_half[2] = {uint64_t(0x5555555555555555), 0};
  AscendC::Duplicate(inner_dup, 1U, mask_half, repeat_times, 1, 8);
  repeat_params = {1, 1, 1, dst_stride, dst_stride, dst_stride};
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::And(dst, dst, inner_dup_tmp, mask * 4, repeat_times, repeat_params);
}

inline __aicore__ void CastTensorToHalfNormal(const AscendC::LocalTensor<int32_t> &dst,
                                              const AscendC::LocalTensor<uint16_t> &src0_bits,
                                              const AscendC::LocalTensor<uint16_t> &src1_bits,
                                              const AscendC::LocalTensor<uint32_t> &inner_dup, const uint32_t mask,
                                              const uint32_t repeat_times, const uint8_t dst_stride) {
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::Duplicate(inner_dup, uint32_t(0x0000BC00), inner_dup.GetSize());
  AscendC::LocalTensor<uint16_t> inner_dup_tmp = inner_dup.ReinterpretCast<uint16_t>();
  AscendC::PipeBarrier<PIPE_V>();
  BinaryRepeatParams repeat_params = {1, 1, 1, dst_stride, dst_stride, dst_stride};
  AscendC::And(src1_bits, src0_bits, inner_dup_tmp, mask * 4, repeat_times, repeat_params);
  AscendC::LocalTensor<int32_t> sub_tmp = inner_dup.ReinterpretCast<int32_t>();
  AscendC::Duplicate(sub_tmp, 0, sub_tmp.GetSize());
  AscendC::Sub(dst, sub_tmp, src0_bits.ReinterpretCast<int32_t>(), mask * 2, repeat_times, repeat_params);
  AscendC::Duplicate(inner_dup, uint32_t(0x00003C00), inner_dup.GetSize());
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::And(src0_bits, dst.ReinterpretCast<uint16_t>(), inner_dup_tmp, mask * 4, repeat_times, repeat_params);
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::Or(src0_bits, src1_bits, src0_bits, mask * 4, repeat_times, repeat_params);
  AscendC::LocalTensor<half> dst_tmp = dst.ReinterpretCast<half>();
  AscendC::LocalTensor<half> sign_tmp = src0_bits.ReinterpretCast<half>();
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::PairReduceSum(dst_tmp, sign_tmp, repeat_times, mask * 4, 1, 1, dst_stride);
}

inline __aicore__ void CalcWeightedTensorNormal(const AscendC::LocalTensor<half> &dst,
                                                const AscendC::LocalTensor<half> &src,
                                                const AscendC::LocalTensor<uint32_t> &inner_dup, const uint32_t mask,
                                                const uint32_t repeat_times, const half weight0, const half weight1,
                                                const uint8_t dst_stride) {
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::LocalTensor<half> inner_dup_tmp = inner_dup.ReinterpretCast<half>();
  AscendC::Duplicate(inner_dup_tmp, weight0, inner_dup_tmp.GetSize());
  uint64_t mask_half[2] = {uint64_t(0x5555555555555555), uint64_t(0x5555555555555555)};
  AscendC::Duplicate(inner_dup_tmp, weight1, mask_half, repeat_times, 1, 8);
  AscendC::PipeBarrier<PIPE_V>();
  BinaryRepeatParams repeat_params = {1, 1, 1, dst_stride, dst_stride, dst_stride};
  AscendC::Mul(src, src, inner_dup_tmp, mask * 2, repeat_times, repeat_params);
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::PairReduceSum(dst, src, repeat_times, mask * 2, 1, 1, 4);
}

template <CMPMODE mode>
inline __aicore__ void CompareExtendInt64GtGeLe(const AscendC::LocalTensor<uint8_t> &dst,   // output
                                                const AscendC::LocalTensor<int64_t> &src0,  // input 0 is a tensor
                                                const AscendC::LocalTensor<int64_t> &src1,  // input 1 is a tensor
                                                const uint8_t repeat_times, const uint32_t last_axis,
                                                const uint32_t input_last_dim_stride,
                                                const uint32_t output_last_dim_stride,
                                                AscendC::LocalTensor<uint8_t> &tmp_buf, const bool src1_is_one_block) {
  const uint8_t dst_repeat_stride = output_last_dim_stride / ONE_BLK_SIZE;  // 由于输出均为uint8_t，所以不用乘类型大小
  const uint8_t src_repeat_stride = input_last_dim_stride * sizeof(int64_t) / ONE_BLK_SIZE;
  constexpr uint32_t elem_in_one_repeat = ONE_REPEAT_BYTE_SIZE / sizeof(int64_t);

  // Divide int64_t into 2 int32_t digits,
  // calculate the sign bits and value bits separately, and calculate as a whole in the end
  uint32_t quadruple_cal_cnt = 4 * repeat_times * (last_axis < elem_in_one_repeat ? elem_in_one_repeat : last_axis);
  uint32_t double_cal_cnt = 2 * repeat_times * (last_axis < elem_in_one_repeat ? elem_in_one_repeat : last_axis);
  // make sure every tensor address is aligned to 32B
  // uint32_t offset_aligned = DivCeil(repeat_times * elem_in_one_repeat * sizeof(int64_t), ONE_BLK_SIZE) *
  // ONE_BLK_SIZE;
  uint32_t offset = 0;
  // src0_bits is used to store the sign bits of src0, while the value bits are all zeros
  AscendC::LocalTensor<uint16_t> src0_bits = KernelUtils::NewTensor<uint16_t>(tmp_buf, 0, quadruple_cal_cnt);
  offset += DivCeil(quadruple_cal_cnt * sizeof(uint16_t), ONE_BLK_SIZE) * ONE_BLK_SIZE;
  // src1_bits is used to store the sign bits of src0, while the value bits are all zeros
  AscendC::LocalTensor<uint16_t> src1_bits = KernelUtils::NewTensor<uint16_t>(tmp_buf, offset, quadruple_cal_cnt);
  offset += DivCeil(quadruple_cal_cnt * sizeof(uint16_t), ONE_BLK_SIZE) * ONE_BLK_SIZE;
  // sub_res is used to store the compute result of src0_bits and src1_bits
  AscendC::LocalTensor<int32_t> sub_res = KernelUtils::NewTensor<int32_t>(tmp_buf, offset, double_cal_cnt);
  offset += DivCeil(double_cal_cnt * sizeof(int32_t), ONE_BLK_SIZE) * ONE_BLK_SIZE;
  // sign_res is used to store the weighted compute result of sign bits
  AscendC::LocalTensor<half> sign_res = KernelUtils::NewTensor<half>(tmp_buf, offset, double_cal_cnt);
  offset += DivCeil(double_cal_cnt * sizeof(half), ONE_BLK_SIZE) * ONE_BLK_SIZE;
  // value_res is used to store the weighted compute result of value bits
  AscendC::LocalTensor<half> value_res = KernelUtils::NewTensor<half>(tmp_buf, offset, double_cal_cnt);
  offset += DivCeil(double_cal_cnt * sizeof(half), ONE_BLK_SIZE) * ONE_BLK_SIZE;
  // inner_dup is used for internal-compute
  AscendC::LocalTensor<uint32_t> inner_dup = KernelUtils::NewTensor<uint32_t>(tmp_buf, offset, double_cal_cnt);
  AscendC::LocalTensor<uint16_t> inner_dup_tmp = inner_dup.ReinterpretCast<uint16_t>();

  if (last_axis < elem_in_one_repeat) {
    uint32_t mask = last_axis;
    uint8_t dst_stride = DivCeil(mask, (ONE_BLK_SIZE / sizeof(uint64_t)));
    // step 1. Get src0 and src1 sign bit
    GetSignBitTensorNormal(src0_bits, src0, inner_dup, mask, repeat_times, src_repeat_stride, dst_stride,
                           src0.GetSize() * sizeof(int64_t) == ONE_BLK_SIZE);
    GetSignBitTensorNormal(src1_bits, src1, inner_dup, mask, repeat_times, src_repeat_stride, dst_stride,
                           src1_is_one_block);
    // step 2. use src0_bits to store sign bit sub results
    AscendC::LocalTensor<int32_t> src0_bits_tmp = src0_bits.ReinterpretCast<int32_t>();
    AscendC::LocalTensor<int32_t> src1_bits_tmp = src1_bits.ReinterpretCast<int32_t>();
    AscendC::PipeBarrier<PIPE_V>();
    BinaryRepeatParams repeat_params = {1, 1, 1, dst_stride, dst_stride, dst_stride};
    AscendC::Sub(src0_bits_tmp, src0_bits_tmp, src1_bits_tmp, mask * 2, repeat_times, repeat_params);
    // step 3. change sign bit sub results which represented as int32_t into half which only keep the results of sign
    // bit
    CastTensorToHalfNormal(sub_res, src0_bits, src1_bits, inner_dup, mask, repeat_times, dst_stride);
    // step 4. calculate the weighted values of sign bit and sum them up
    CalcWeightedTensorNormal(sign_res, sub_res.ReinterpretCast<half>(), inner_dup, mask, repeat_times, half(8), half(2),
                             dst_stride);
    // step 5. get the value bits of src0 and src1
    AscendC::Duplicate(inner_dup, uint32_t(0x7FFFFFFF), inner_dup.GetSize());
    AscendC::LocalTensor<uint16_t> src0_tmp = src0.ReinterpretCast<uint16_t>();
    AscendC::LocalTensor<uint16_t> src1_tmp = src1.ReinterpretCast<uint16_t>();
    AscendC::PipeBarrier<PIPE_V>();
    repeat_params = {1, 1, 1, dst_stride, src_repeat_stride, src_repeat_stride};
    AscendC::And(src0_bits, src0_tmp, inner_dup_tmp, mask * 4, repeat_times, repeat_params);
    if (src1_is_one_block) {
      repeat_params = {1, 0, 1, dst_stride, 0, src_repeat_stride};
    }
    AscendC::And(src1_bits, src1_tmp, inner_dup_tmp, mask * 4, repeat_times, repeat_params);
    // step 6. sub the two value bits interval, and change the result into 0/1/-1
    repeat_params = {1, 1, 1, dst_stride, src_repeat_stride, src_repeat_stride};
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(src0_bits_tmp, src0_bits_tmp, src1_bits_tmp, mask * 2, repeat_times, repeat_params);
    UnaryRepeatParams unary_repeat_params = {1, 1, src_repeat_stride, src_repeat_stride};
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Maxs(src0_bits_tmp, src0_bits_tmp, -1, mask * 2, repeat_times, unary_repeat_params);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mins(src0_bits_tmp, src0_bits_tmp, 1, mask * 2, repeat_times, unary_repeat_params);
    // step 7. change value bits sub results which represented as int32_t into half which only keep the results of value
    // bits
    CastTensorToHalfNormal(sub_res, src0_bits, src1_bits, inner_dup, mask, repeat_times, dst_stride);

    // step 8. calculate the weighted values of value bits and sum them up.
    CalcWeightedTensorNormal(value_res, sub_res.ReinterpretCast<half>(), inner_dup, mask, repeat_times, half(4),
                             half(1), dst_stride);
    // step 9. sum up the results of sign bit and value bits, change the result into 0/1
    repeat_params = {1, 1, 1, 1, 4, 4};
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(sign_res, sign_res, value_res, mask, repeat_times, repeat_params);
    AscendC::PipeBarrier<PIPE_V>();
    unary_repeat_params = {1, 1, 1, 1};
    if (mode == CMPMODE::GT) {
      AscendC::Maxs(sign_res, sign_res, half(0), mask, repeat_times, unary_repeat_params);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mins(sign_res, sign_res, half(1), mask, repeat_times, unary_repeat_params);
      // step 10. cast the final result into int8
      unary_repeat_params = {1, 1, dst_repeat_stride, 1};
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(dst, sign_res, AscendC::RoundMode::CAST_NONE, mask, repeat_times, unary_repeat_params);
    } else {
      AscendC::Maxs(sign_res, sign_res, half(-1), mask, repeat_times, unary_repeat_params);
      AscendC::PipeBarrier<PIPE_V>();

      AscendC::Mins(sign_res, sign_res, half(0), mask, repeat_times, unary_repeat_params);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Adds(sign_res, sign_res, half(1), mask, repeat_times, unary_repeat_params);
      // step 10. cast the final result into int8
      unary_repeat_params = {1, 1, dst_repeat_stride, 1};
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(dst, sign_res, AscendC::RoundMode::CAST_NONE, mask, repeat_times, unary_repeat_params);
    }
  } else {
    uint32_t element_extent = last_axis / elem_in_one_repeat;
    uint32_t element_reminder = last_axis - element_extent * elem_in_one_repeat;
    for (uint32_t outer_for = 0; outer_for < element_extent; outer_for++) {
      uint32_t mask = elem_in_one_repeat;
      uint8_t dst_stride = DivCeil(mask, (ONE_BLK_SIZE / sizeof(uint64_t)));
      // step 1. Get src0 and src1 sign bit
      GetSignBitTensorNormal(src0_bits, src0[outer_for * elem_in_one_repeat], inner_dup, mask, repeat_times,
                             src_repeat_stride, dst_stride,
                             src0[outer_for * elem_in_one_repeat].GetSize() * sizeof(int64_t) == ONE_BLK_SIZE);
      if (src1_is_one_block) {
        GetSignBitTensorNormal(src1_bits, src1, inner_dup, mask, repeat_times, src_repeat_stride, dst_stride,
                               src1_is_one_block);
      } else {
        GetSignBitTensorNormal(src1_bits, src1[outer_for * elem_in_one_repeat], inner_dup, mask, repeat_times,
                               src_repeat_stride, dst_stride, src1_is_one_block);
      }
      // step 2. use src0_bits to store sign bit sub results
      AscendC::LocalTensor<int32_t> src0_bits_tmp = src0_bits.ReinterpretCast<int32_t>();
      AscendC::LocalTensor<int32_t> src1_bits_tmp = src1_bits.ReinterpretCast<int32_t>();
      AscendC::PipeBarrier<PIPE_V>();
      BinaryRepeatParams repeat_params = {1, 1, 1, dst_stride, dst_stride, dst_stride};
      AscendC::Sub(src0_bits_tmp, src0_bits_tmp, src1_bits_tmp, mask * 2, repeat_times, repeat_params);
      // step 3. change sign bit sub results which represented as int32_t into half which only keep the results of sign
      // bit
      CastTensorToHalfNormal(sub_res, src0_bits, src1_bits, inner_dup, mask, repeat_times, dst_stride);
      // step 4. calculate the weighted values of sign bit and sum them up
      CalcWeightedTensorNormal(sign_res, sub_res.ReinterpretCast<half>(), inner_dup, mask, repeat_times, half(8),
                               half(2), dst_stride);
      // step 5. get the value bits of src0 and src1
      AscendC::Duplicate(inner_dup, uint32_t(0x7FFFFFFF), inner_dup.GetSize());
      AscendC::PipeBarrier<PIPE_V>();
      repeat_params = {1, 1, 1, dst_stride, src_repeat_stride, src_repeat_stride};
      AscendC::LocalTensor<uint16_t> src0_tmp = src0[outer_for * elem_in_one_repeat].ReinterpretCast<uint16_t>();
      AscendC::And(src0_bits, src0_tmp, inner_dup_tmp, mask * 4, repeat_times, repeat_params);
      if (src1_is_one_block) {
        repeat_params = {1, 0, 1, dst_stride, 0, src_repeat_stride};
        AscendC::LocalTensor<uint16_t> src1_tmp = src1.ReinterpretCast<uint16_t>();
        AscendC::And(src1_bits, src1_tmp, inner_dup_tmp, mask * 4, repeat_times, repeat_params);
      } else {
        AscendC::LocalTensor<uint16_t> src1_tmp = src1[outer_for * elem_in_one_repeat].ReinterpretCast<uint16_t>();
        AscendC::And(src1_bits, src1_tmp, inner_dup_tmp, mask * 4, repeat_times, repeat_params);
      }
      // step 6. sub the two value bits interval, and change the result into 0/1/-1
      repeat_params = {1, 1, 1, dst_stride, dst_stride, dst_stride};
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sub(src0_bits_tmp, src0_bits_tmp, src1_bits_tmp, mask * 2, repeat_times, repeat_params);
      UnaryRepeatParams unary_repeat_params = {1, 1, dst_stride, dst_stride};
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Maxs(src0_bits_tmp, src0_bits_tmp, -1, mask * 2, repeat_times, unary_repeat_params);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mins(src0_bits_tmp, src0_bits_tmp, 1, mask * 2, repeat_times, unary_repeat_params);
      // step 7. change value bits sub results which represented as int32_t into half which only keep the results of
      // value bits
      CastTensorToHalfNormal(sub_res, src0_bits, src1_bits, inner_dup, mask, repeat_times, dst_stride);

      // step 8. calculate the weighted values of value bits and sum them up.
      CalcWeightedTensorNormal(value_res, sub_res.ReinterpretCast<half>(), inner_dup, mask, repeat_times, half(4),
                               half(1), dst_stride);
      // step 9. sum up the results of sign bit and value bits, change the result into 0/1
      repeat_params = {1, 1, 1, 2, 4, 4};
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Add(sign_res, sign_res, value_res, mask, repeat_times, repeat_params);
      AscendC::PipeBarrier<PIPE_V>();
      unary_repeat_params = {1, 1, 2, 2};
      if (mode == CMPMODE::GT) {
        AscendC::Maxs(sign_res, sign_res, half(0), mask, repeat_times, unary_repeat_params);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mins(sign_res, sign_res, half(1), mask, repeat_times, unary_repeat_params);
        // step 10. cast the final result into int8
        unary_repeat_params = {1, 1, dst_repeat_stride, 2};
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(dst[outer_for * elem_in_one_repeat], sign_res, AscendC::RoundMode::CAST_NONE, mask, repeat_times,
                      unary_repeat_params);
      } else {
        AscendC::Maxs(sign_res, sign_res, half(-1), mask, repeat_times, unary_repeat_params);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Mins(sign_res, sign_res, half(0), mask, repeat_times, unary_repeat_params);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(sign_res, sign_res, half(1), mask, repeat_times, unary_repeat_params);
        // step 10. cast the final result into int8
        unary_repeat_params = {1, 1, dst_repeat_stride, 2};
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(dst[outer_for * elem_in_one_repeat], sign_res, AscendC::RoundMode::CAST_NONE, mask, repeat_times,
                      unary_repeat_params);
      }
    }
    if (element_reminder != 0) {
      uint32_t mask = element_reminder;
      uint8_t dst_stride = DivCeil(mask, (ONE_BLK_SIZE / sizeof(uint64_t)));
      // step 1. Get src0 and src1 sign bit
      GetSignBitTensorNormal(src0_bits, src0[element_extent * elem_in_one_repeat], inner_dup, mask, repeat_times,
                             src_repeat_stride, dst_stride,
                             src0[element_extent * elem_in_one_repeat].GetSize() * sizeof(int64_t) == ONE_BLK_SIZE);
      if (src1_is_one_block) {
        GetSignBitTensorNormal(src1_bits, src1, inner_dup, mask, repeat_times, src_repeat_stride, dst_stride,
                               src1_is_one_block);
      } else {
        GetSignBitTensorNormal(src1_bits, src1[element_extent * elem_in_one_repeat], inner_dup, mask, repeat_times,
                               src_repeat_stride, dst_stride, src1_is_one_block);
      }
      // step 2. use src0_bits to store sign bit sub results
      AscendC::LocalTensor<int32_t> src0_bits_tmp = src0_bits.ReinterpretCast<int32_t>();
      AscendC::LocalTensor<int32_t> src1_bits_tmp = src1_bits.ReinterpretCast<int32_t>();
      AscendC::PipeBarrier<PIPE_V>();
      BinaryRepeatParams repeat_params = {1, 1, 1, dst_stride, dst_stride, dst_stride};
      AscendC::Sub(src0_bits_tmp, src0_bits_tmp, src1_bits_tmp, mask * 2, repeat_times, repeat_params);
      // step 3. change sign bit sub results which represented as int32_t into half which only keep the results of sign
      // bit
      CastTensorToHalfNormal(sub_res, src0_bits, src1_bits, inner_dup, mask, repeat_times, dst_stride);
      // step 4. calculate the weighted values of sign bit and sum them up
      CalcWeightedTensorNormal(sign_res, sub_res.ReinterpretCast<half>(), inner_dup, mask, repeat_times, half(8),
                               half(2), dst_stride);
      // step 5. get the value bits of src0 and src1
      AscendC::Duplicate(inner_dup, uint32_t(0x7FFFFFFF), inner_dup.GetSize());
      AscendC::PipeBarrier<PIPE_V>();
      repeat_params = {1, 1, 1, dst_stride, src_repeat_stride, src_repeat_stride};
      AscendC::LocalTensor<uint16_t> src0_tmp = src0[element_extent * elem_in_one_repeat].ReinterpretCast<uint16_t>();
      AscendC::And(src0_bits, src0_tmp, inner_dup_tmp, mask * 4, repeat_times, repeat_params);
      if (src1_is_one_block) {
        repeat_params = {1, 0, 1, dst_stride, 0, src_repeat_stride};
        AscendC::LocalTensor<uint16_t> src1_tmp = src1.ReinterpretCast<uint16_t>();
        AscendC::And(src1_bits, src1_tmp, inner_dup_tmp, mask * 4, repeat_times, repeat_params);
      } else {
        AscendC::LocalTensor<uint16_t> src1_tmp = src1[element_extent * elem_in_one_repeat].ReinterpretCast<uint16_t>();
        AscendC::And(src1_bits, src1_tmp, inner_dup_tmp, mask * 4, repeat_times, repeat_params);
      }
      // step 6. sub the two value bits interval, and change the result into 0/1/-1
      repeat_params = {1, 1, 1, dst_stride, dst_stride, dst_stride};
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sub(src0_bits_tmp, src0_bits_tmp, src1_bits_tmp, mask * 2, repeat_times, repeat_params);
      UnaryRepeatParams unary_repeat_params = {1, 1, dst_stride, dst_stride};
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Maxs(src0_bits_tmp, src0_bits_tmp, -1, mask * 2, repeat_times, unary_repeat_params);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mins(src0_bits_tmp, src0_bits_tmp, 1, mask * 2, repeat_times, unary_repeat_params);
      // step 7. change value bits sub results which represented as int32_t into half which only keep the results of
      // value bits
      CastTensorToHalfNormal(sub_res, src0_bits, src1_bits, inner_dup, mask, repeat_times, dst_stride);

      // step 8. calculate the weighted values of value bits and sum them up.
      CalcWeightedTensorNormal(value_res, sub_res.ReinterpretCast<half>(), inner_dup, mask, repeat_times, half(4),
                               half(1), dst_stride);
      // step 9. sum up the results of sign bit and value bits, change the result into 0/1
      repeat_params = {1, 1, 1, 2, 4, 4};
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Add(sign_res, sign_res, value_res, mask, repeat_times, repeat_params);
      AscendC::PipeBarrier<PIPE_V>();
      unary_repeat_params = {1, 1, 2, 2};
      if (mode == CMPMODE::GT) {
        AscendC::Maxs(sign_res, sign_res, half(0), mask, repeat_times, unary_repeat_params);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mins(sign_res, sign_res, half(1), mask, repeat_times, unary_repeat_params);
        // step 10. cast the final result into int8
        unary_repeat_params = {1, 1, dst_repeat_stride, 2};
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(dst[element_extent * elem_in_one_repeat], sign_res, AscendC::RoundMode::CAST_NONE, mask,
                      repeat_times, unary_repeat_params);
      } else {
        AscendC::Maxs(sign_res, sign_res, half(-1), mask, repeat_times, unary_repeat_params);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Mins(sign_res, sign_res, half(0), mask, repeat_times, unary_repeat_params);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(sign_res, sign_res, half(1), mask, repeat_times, unary_repeat_params);
        // step 10. cast the final result into int8
        unary_repeat_params = {1, 1, dst_repeat_stride, 2};
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(dst[element_extent * elem_in_one_repeat], sign_res, AscendC::RoundMode::CAST_NONE, mask,
                      repeat_times, unary_repeat_params);
      }
    }
  }
}
template <typename T, CMPMODE mode>
inline __aicore__ void CompareExtendRepeat(const AscendC::LocalTensor<uint8_t> &dst,  // output
                                           const AscendC::LocalTensor<T> &src0,       // input 0 is a tensor
                                           const AscendC::LocalTensor<T> &src1,       // input 1 is a tensor
                                           const uint8_t repeat_times, const uint32_t last_axis,
                                           const uint32_t input_last_dim_stride, const uint32_t output_last_dim_stride,
                                           AscendC::LocalTensor<uint8_t> &tmp_buf, const bool src1_is_one_block) {
  const uint32_t max_stride = 255;
  if constexpr (AscendC::IsSameType<T, int32_t>::value &&
                (mode == CMPMODE::GT || mode == CMPMODE::NE || mode == CMPMODE::GE || mode == CMPMODE::LE ||
                 mode == CMPMODE::LT)) {
    if (input_last_dim_stride * sizeof(int32_t) / ONE_BLK_SIZE <= max_stride &&
        output_last_dim_stride / ONE_BLK_SIZE <= max_stride) {
      CompareExtendInt32<mode>(dst, src0, src1, repeat_times, last_axis, input_last_dim_stride, output_last_dim_stride,
                               tmp_buf, src1_is_one_block);
    } else {
      for (uint32_t i = 0; i < repeat_times; ++i) {
        CompareExtendInt32<mode>(dst[i * output_last_dim_stride], src0[i * input_last_dim_stride],
                                 src1[src1_is_one_block ? 0 : i * input_last_dim_stride], 1, last_axis,
                                 input_last_dim_stride, output_last_dim_stride, tmp_buf, src1_is_one_block);
      }
    }
  } else if constexpr (AscendC::IsSameType<T, int64_t>::value) {
    if constexpr (mode != CMPMODE::EQ && mode != CMPMODE::NE && mode != CMPMODE::GT && mode != CMPMODE::GE &&
                  mode != CMPMODE::LE) {
      ASSERT(false && "CompareExtend mode only support EQ/NE/GT/GE/LE when DataType is int64.");
    }
    if constexpr (mode == CMPMODE::EQ || mode == CMPMODE::NE) {
      if (input_last_dim_stride * sizeof(int64_t) / ONE_BLK_SIZE <= max_stride &&
          output_last_dim_stride / ONE_BLK_SIZE <= max_stride) {
        CompareExtendInt64EqNe<mode>(dst, src0, src1, repeat_times, last_axis, input_last_dim_stride,
                                     output_last_dim_stride, tmp_buf, src1_is_one_block);
      } else {
        for (uint8_t i = 0; i < repeat_times; ++i) {
          CompareExtendInt64EqNe<mode>(dst[i * output_last_dim_stride], src0[i * input_last_dim_stride],
                                       src1[src1_is_one_block ? 0 : i * input_last_dim_stride], 1, 0,
                                       input_last_dim_stride, output_last_dim_stride, tmp_buf, src1_is_one_block);
        }
      }
    } else if constexpr (mode == CMPMODE::LE || mode == CMPMODE::GT || mode == CMPMODE::GE) {
      if (input_last_dim_stride * sizeof(int64_t) / ONE_BLK_SIZE <= max_stride &&
          output_last_dim_stride / ONE_BLK_SIZE <= max_stride) {
        CompareExtendInt64GtGeLe<mode>(dst, src0, src1, repeat_times, last_axis, input_last_dim_stride,
                                       output_last_dim_stride, tmp_buf, src1_is_one_block);
      } else {
        for (uint8_t i = 0; i < repeat_times; ++i) {
          CompareExtendInt64GtGeLe<mode>(dst[i * output_last_dim_stride], src0[i * input_last_dim_stride],
                                         src1[src1_is_one_block ? 0 : i * input_last_dim_stride], 1, 0,
                                         input_last_dim_stride, output_last_dim_stride, tmp_buf, src1_is_one_block);
        }
      }
    }
  } else {
    if (input_last_dim_stride * sizeof(T) / ONE_BLK_SIZE <= max_stride &&
        output_last_dim_stride / ONE_BLK_SIZE <= max_stride) {
      CompareNormal<T, mode>(dst, src0, src1, repeat_times, last_axis, input_last_dim_stride, output_last_dim_stride,
                             tmp_buf, src1_is_one_block);
    } else {
      for (uint8_t i = 0; i < repeat_times; ++i) {
        CompareNormal<T, mode>(dst[i * output_last_dim_stride], src0[i * input_last_dim_stride],
                               src1[src1_is_one_block ? 0 : i * input_last_dim_stride], 1, last_axis,
                               input_last_dim_stride, output_last_dim_stride, tmp_buf, src1_is_one_block);
      }
    }
  }
}

// src1_valid_bytes is src1's block-aligned storage requirement before buffer reuse,
// not its allocation capacity or the number of compared elements. A one-block
// source is reused without source offsets; scalar sources must fill that block.
template <typename T, CMPMODE mode>
inline __aicore__ void CompareExtend(const AscendC::LocalTensor<uint8_t> &dst,  // output
                                     const AscendC::LocalTensor<T> &src0,       // input 0 is a tensor
                                     const AscendC::LocalTensor<T> &src1,       // input 1 is a tensor
                                     const uint32_t first_axis, const uint32_t last_axis,
                                     const uint32_t input_last_dim_stride, const uint32_t output_last_dim_stride,
                                     AscendC::LocalTensor<uint8_t> &tmp_buf, const uint32_t src1_valid_bytes) {
  const bool src1_is_one_block = src1_valid_bytes == ONE_BLK_SIZE;
  uint32_t calc_repeat_time = 0;
  uint64_t inputOffset = calc_repeat_time * input_last_dim_stride;
  uint64_t outputOffset = calc_repeat_time * output_last_dim_stride;
  if (first_axis > MAX_REPEAT_TIME) {
    for (; (calc_repeat_time + MAX_REPEAT_TIME) < first_axis; calc_repeat_time += MAX_REPEAT_TIME) {
      inputOffset = calc_repeat_time * input_last_dim_stride;
      outputOffset = calc_repeat_time * output_last_dim_stride;
      if (src1_is_one_block) {
        CompareExtendRepeat<T, mode>(dst[outputOffset], src0[inputOffset], src1, MAX_REPEAT_TIME, last_axis,
                                     input_last_dim_stride, output_last_dim_stride, tmp_buf, src1_is_one_block);
      } else {
        CompareExtendRepeat<T, mode>(dst[outputOffset], src0[inputOffset], src1[inputOffset], MAX_REPEAT_TIME,
                                     last_axis, input_last_dim_stride, output_last_dim_stride, tmp_buf,
                                     src1_is_one_block);
      }
    }
  }
  uint8_t remain_repeat_time = (uint8_t)(first_axis - calc_repeat_time);
  inputOffset = calc_repeat_time * input_last_dim_stride;
  outputOffset = calc_repeat_time * output_last_dim_stride;
  if (src1_is_one_block) {
    CompareExtendRepeat<T, mode>(dst[outputOffset], src0[inputOffset], src1, remain_repeat_time, last_axis,
                                 input_last_dim_stride, output_last_dim_stride, tmp_buf, src1_is_one_block);
  } else {
    CompareExtendRepeat<T, mode>(dst[outputOffset], src0[inputOffset], src1[inputOffset], remain_repeat_time, last_axis,
                                 input_last_dim_stride, output_last_dim_stride, tmp_buf, src1_is_one_block);
  }
}

// Backward-compatible entry point. The legacy API used the LocalTensor capacity as
// the only available indication of a one-block source. New callers should pass
// the source tensor's valid storage size explicitly through the overload above.
template <typename T, CMPMODE mode>
inline __aicore__ void CompareExtend(const AscendC::LocalTensor<uint8_t> &dst, const AscendC::LocalTensor<T> &src0,
                                     const AscendC::LocalTensor<T> &src1, const uint32_t first_axis,
                                     const uint32_t last_axis, const uint32_t input_last_dim_stride,
                                     const uint32_t output_last_dim_stride, AscendC::LocalTensor<uint8_t> &tmp_buf) {
  CompareExtend<T, mode>(dst, src0, src1, first_axis, last_axis, input_last_dim_stride, output_last_dim_stride, tmp_buf,
                         src1.GetSize() * sizeof(T));
}
#endif  // __ASCENDC_API_COMPARE_V2_H__
