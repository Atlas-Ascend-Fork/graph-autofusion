/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_BROADCAST_H__
#define __ASCENDC_API_BROADCAST_H__

constexpr uint32_t BRCB_ONE_SIZE_A = 8;
constexpr uint32_t BRCB_HALF_MAX_REPEATE_TIMES_A = 254;
constexpr uint32_t BRCB_FLOAT_MAX_REPEATE_TIMES_A = 255;
constexpr uint32_t GATHER_MASK_PATTERN_A = 7;

template <typename T>
inline __aicore__ void BroadcastFirstDim(const LocalTensor<T> &dst, const LocalTensor<T> &src, const uint32_t src_m,
                                         const uint32_t src_k, const uint32_t src_z, const uint32_t dst_m,
                                         const uint32_t dst_k, const uint32_t dst_z, LocalTensor<uint8_t> &tmp_buf) {
  uint32_t dst_inner_offset = dst_k * (src_z == 0 ? 1 : src_z);
  uint32_t one_blk_num = KernelUtils::BlkSize<T>();
  AscendC::SetMaskCount();
  if (dst_inner_offset <= one_blk_num) {
    uint32_t total_count = dst_m * dst_inner_offset;
    AscendC::SetVectorMask<T, MaskMode::COUNTER>(total_count);
    AscendC::CopyRepeatParams repeat_params{1, 0, 1, 0};
    AscendC::Copy<T, false>(dst, src, MASK_PLACEHOLDER, 1, repeat_params);
  } else {
    const uint32_t max_rpt_cnt = dst_m / MAX_REPEAT_TIME;
    uint32_t calc_size = 0;
    AscendC::SetVectorMask<T, MaskMode::COUNTER>(dst_inner_offset);
    const uint16_t dst_rpt_stride = dst_inner_offset * sizeof(T) / ONE_BLK_SIZE;
    AscendC::CopyRepeatParams repeat_params{1, 1, dst_rpt_stride, 0};
    const uint32_t max_rpt_calc_num = dst_inner_offset * MAX_REPEAT_TIME;
    for (uint32_t idx = 0; idx < max_rpt_cnt; idx++) {
      AscendC::Copy<T, false>(dst[calc_size], src, MASK_PLACEHOLDER, MAX_REPEAT_TIME, repeat_params);
      calc_size += max_rpt_calc_num;
    }
    uint32_t tail_rpt_times = dst_m - max_rpt_cnt * MAX_REPEAT_TIME;
    if (tail_rpt_times != 0) {
      AscendC::Copy<T, false>(dst[calc_size], src, MASK_PLACEHOLDER, tail_rpt_times, repeat_params);
    }
  }
  AscendC::SetMaskNorm();
  AscendC::ResetMask();
}

template <typename T>
inline __aicore__ void BroadcastMiddleDim(const LocalTensor<T> &dst, const LocalTensor<T> &src, const uint32_t src_m,
                                          const uint32_t src_k, const uint32_t src_z, const uint32_t dst_m,
                                          const uint32_t dst_k, const uint32_t dst_z, LocalTensor<uint8_t> &tmp_buf) {
  uint32_t inner_offset = src_z == 0 ? 1 : src_z;
  uint32_t dst_inner_offset = dst_k * inner_offset;
  constexpr uint32_t one_blk_num = ONE_BLK_SIZE / sizeof(T);
  constexpr uint32_t outer_dim_max_loop = MAX_REPEAT_TIME * one_blk_num;
  const uint32_t src_inner_offset = src_k * inner_offset;
  if (src_inner_offset >= outer_dim_max_loop) {
    for (uint32_t i = 0; i < dst_m; i++) {
      AscendC::DataCopy(dst[i * dst_inner_offset], src, dst_inner_offset);
    }
    AscendC::PipeBarrier<PIPE_V>();
  } else {
    AscendC::Duplicate(tmp_buf.ReinterpretCast<uint16_t>(), (uint16_t)0, ONE_BLK_SIZE / sizeof(uint16_t));
    AscendC::PipeBarrier<PIPE_V>();
    int32_t dtype_count = 1;
    if constexpr (sizeof(T) == sizeof(float)) {
      dtype_count = 2;
    }
    uint32_t loop_cnt = dst_m / 8U;
    uint32_t rpt = dst_inner_offset / one_blk_num;
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<uint16_t, MaskMode::NORMAL>(128);
    uint32_t dst_blk_stride = dst_inner_offset * dtype_count / 16U;
    BinaryRepeatParams binary_params((uint8_t)dst_blk_stride, 0, 0, 1, 1, 0);
    uint32_t dst_offset = 0;
    for (uint32_t i = 0; i < loop_cnt; i++) {
      AscendC::Or<uint16_t, false>(dst[dst_offset].template ReinterpretCast<uint16_t>(),
                                   src.template ReinterpretCast<uint16_t>(),
                                   tmp_buf.template ReinterpretCast<uint16_t>(), MASK_PLACEHOLDER, rpt, binary_params);
      dst_offset += dst_inner_offset * 8;
    }
    uint32_t loop_tail = dst_m - loop_cnt * 8;
    if (loop_tail > 0) {
      AscendC::SetMaskNorm();
      AscendC::SetVectorMask<uint16_t, MaskMode::NORMAL>(loop_tail * 16);
      AscendC::Or<uint16_t, false>(dst[dst_offset].template ReinterpretCast<uint16_t>(),
                                   src.template ReinterpretCast<uint16_t>(),
                                   tmp_buf.template ReinterpretCast<uint16_t>(), MASK_PLACEHOLDER, rpt, binary_params);
    }
    AscendC::SetMaskNorm();
    AscendC::ResetMask();
  }
}

template <typename T>
__aicore__ inline void BrcbSrcToOneBlock(const LocalTensor<T> &src_local, const uint32_t first_dim,
                                         uint32_t one_blk_num, LocalTensor<T> &brcb_buf) {
  const uint32_t brcb_rpt_times = (first_dim + BRCB_ONE_SIZE_A - 1) / BRCB_ONE_SIZE_A;
  uint32_t brcb_max_rpt_times = BRCB_HALF_MAX_REPEATE_TIMES_A;
  if constexpr (sizeof(T) == sizeof(float)) {
    brcb_max_rpt_times = BRCB_FLOAT_MAX_REPEATE_TIMES_A;
  }
  const uint32_t loop_cnt = brcb_rpt_times / brcb_max_rpt_times;
  const uint32_t tail_rpt = brcb_rpt_times % brcb_max_rpt_times;
  uint32_t brcb_src_offset = 0;
  uint32_t brcb_dst_offset = 0;
  for (uint32_t i = 0; i < loop_cnt; i++) {
    Brcb(brcb_buf[brcb_dst_offset], src_local[brcb_src_offset], brcb_max_rpt_times, {1, DEFAULT_REPEAT_STRIDE});
    brcb_dst_offset += brcb_max_rpt_times * one_blk_num * BRCB_ONE_SIZE_A;
    brcb_src_offset += brcb_max_rpt_times * BRCB_ONE_SIZE_A;
  }
  if (tail_rpt != 0) {
    Brcb(brcb_buf[brcb_dst_offset], src_local[brcb_src_offset], tail_rpt, {1, DEFAULT_REPEAT_STRIDE});
  }
  AscendC::PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void TwoDimBroadCastLastDimAlign(const LocalTensor<T> &dst_local, const LocalTensor<T> &src_local,
                                                   LocalTensor<T> &tmp_buf, const uint32_t first_dim,
                                                   const uint32_t block_count) {
  constexpr uint32_t one_blk_num = ONE_BLK_SIZE / sizeof(T);
  BrcbSrcToOneBlock(src_local, first_dim, one_blk_num, tmp_buf);
  SetVectorMask<T, MaskMode::COUNTER>(block_count);
  const CopyRepeatParams copy_params = {1, 0, (uint16_t)(block_count / one_blk_num), 1};  // overflow check
  uint32_t loop_cnt = first_dim / MAX_REPEAT_TIMES;
  uint32_t dst_offset = 0;
  uint32_t src_offset = 0;
  for (uint32_t i = 0; i < loop_cnt; i++) {
    Copy<T, false>(dst_local[dst_offset], tmp_buf[src_offset], MASK_PLACEHOLDER, MAX_REPEAT_TIMES, copy_params);
    dst_offset += MAX_REPEAT_TIMES * block_count;
    src_offset += MAX_REPEAT_TIMES * one_blk_num;
  }
  uint32_t tail_rpt = first_dim % MAX_REPEAT_TIMES;
  if (tail_rpt != 0) {
    Copy<T, false>(dst_local[dst_offset], tmp_buf[src_offset], MASK_PLACEHOLDER, tail_rpt, copy_params);
  }
  AscendC::PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void TwoDimBroadCastLastDimNotAlign(const LocalTensor<T> &dst_local, const LocalTensor<T> &src_local,
                                                      LocalTensor<T> &tmp_buf, const uint32_t first_dim,
                                                      const uint32_t block_count) {
  constexpr uint32_t one_blk_num = ONE_BLK_SIZE / sizeof(T);
  BrcbSrcToOneBlock(src_local, first_dim, one_blk_num, tmp_buf);
  const uint32_t align_blk_num = (block_count + one_blk_num - 1) / one_blk_num;
  const uint32_t block_count_align = align_blk_num * one_blk_num;
  SetVectorMask<T, MaskMode::COUNTER>(block_count_align);
  const CopyRepeatParams copy_params = {1, 0, (uint16_t)align_blk_num, 1};
  uint32_t copy_counts = first_dim / MAX_REPEAT_TIMES;
  uint32_t dst_offset = 0;
  uint32_t src_offset = 0;
  auto copy_tmp_buffer = tmp_buf[first_dim * one_blk_num];
  for (uint32_t i = 0; i < copy_counts; i++) {
    Copy<T, false>(copy_tmp_buffer[dst_offset], tmp_buf[src_offset], MASK_PLACEHOLDER, MAX_REPEAT_TIMES, copy_params);
    dst_offset += MAX_REPEAT_TIMES * block_count_align;
    src_offset += MAX_REPEAT_TIMES * one_blk_num;
  }
  uint32_t tail_rpt = first_dim % MAX_REPEAT_TIMES;
  if (tail_rpt != 0) {
    Copy<T, false>(copy_tmp_buffer[dst_offset], tmp_buf[src_offset], MASK_PLACEHOLDER, tail_rpt, copy_params);
  }
  AscendC::PipeBarrier<PIPE_V>();
  const GatherMaskParams gather_params = {1, (uint16_t)first_dim, (uint16_t)align_blk_num, 0};  // uint32 cast to uint16
  uint64_t rsvd_cnt = 0;
  GatherMask(dst_local, copy_tmp_buffer, GATHER_MASK_PATTERN_A, true, block_count, gather_params, rsvd_cnt);
  SetMaskCount();
  AscendC::PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GetBrcAlignLoopNumbers(const uint32_t first_dim, const uint32_t block_count,
                                              const uint32_t tmp_buf_size, uint32_t &one_repeat_size, uint32_t &range_m,
                                              uint32_t &tail_m) {
  constexpr uint32_t one_blk_num = ONE_BLK_SIZE / sizeof(T);
  constexpr uint32_t min_brcb_temp_buffer_size = one_blk_num * one_blk_num + one_blk_num;
  constexpr uint32_t min_tmp_buf_size = min_brcb_temp_buffer_size;
  ASCENDC_ASSERT((tmp_buf_size >= min_tmp_buf_size), {
    KERNEL_LOG(KERNEL_ERROR,
               "tmp_buf_size can't be smaller than min_tmp_buf_size, tmp_buf_size is %u, min_tmp_buf_size is %u!",
               tmp_buf_size, min_tmp_buf_size);
  });
  one_repeat_size = tmp_buf_size / min_tmp_buf_size * one_blk_num;
  range_m = first_dim / one_repeat_size;
  tail_m = first_dim - one_repeat_size * range_m;
}

template <typename T>
__aicore__ inline void GetBrcNotAlignLoopNumbers(const uint32_t first_dim, const uint32_t block_count,
                                                 const uint32_t tmp_buf_size, uint32_t &one_repeat_size,
                                                 uint32_t &range_m, uint32_t &tail_m) {
  constexpr uint32_t one_blk_num = ONE_BLK_SIZE / sizeof(T);
  constexpr uint32_t min_brcb_temp_buffer_size = one_blk_num * one_blk_num + one_blk_num;
  const uint32_t align_blk_num = (block_count + one_blk_num - 1) / one_blk_num;
  const uint32_t block_count_align = align_blk_num * one_blk_num;
  const uint32_t min_copy_temp_buffer_size = one_blk_num * block_count_align;
  const uint32_t min_tmp_buf_size = min_brcb_temp_buffer_size + min_copy_temp_buffer_size;
  ASCENDC_ASSERT((tmp_buf_size >= min_tmp_buf_size), {
    KERNEL_LOG(KERNEL_ERROR,
               "tmp_buf_size can't be smaller than min_tmp_buf_size, tmp_buf_size is %u, min_tmp_buf_size is %u!",
               tmp_buf_size, min_tmp_buf_size);
  });
  one_repeat_size = tmp_buf_size / min_tmp_buf_size * one_blk_num;
  range_m = first_dim / one_repeat_size;
  tail_m = first_dim - one_repeat_size * range_m;
}

template <typename T>
inline __aicore__ void GetSrcTensorWithoutStride(const LocalTensor<T> &tmp_src, const LocalTensor<T> &src,
                                                 const uint32_t cal_cnt, const uint32_t offset) {
  uint64_t src_offset = 0;
  event_t event_id_v_to_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
  event_t event_id_s_to_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
  AscendC::SetFlag<HardEvent::V_S>(event_id_v_to_s);
  AscendC::WaitFlag<HardEvent::V_S>(event_id_v_to_s);
  for (uint32_t i = 0; i < cal_cnt; i++) {
    auto tmp = src.GetValue(src_offset);
    tmp_src.SetValue(i, tmp);
    src_offset += offset;
  }
  AscendC::SetFlag<HardEvent::S_V>(event_id_s_to_v);
  AscendC::WaitFlag<HardEvent::S_V>(event_id_s_to_v);
}

template <typename T>
inline __aicore__ void BroadcastWithStride(const LocalTensor<T> &dst, const LocalTensor<T> &src, const uint32_t src_m,
                                           const uint32_t src_k, const uint32_t src_z, const uint32_t dst_m,
                                           const uint32_t dst_k, const uint32_t dst_z, LocalTensor<T> &tmp_buf) {
  uint32_t one_repeat_size = 0;
  uint32_t range_m = 0;
  uint32_t tail_m = 0;
  uint64_t dst_offset = 0;
  uint64_t src_offset = 0;
  if (dst_k * sizeof(T) % ONE_BLK_SIZE == 0) {
    GetBrcAlignLoopNumbers<T>(dst_m, dst_k, tmp_buf.GetSize(), one_repeat_size, range_m, tail_m);
    LocalTensor<T> tmp_src = tmp_buf;
    tmp_src.SetSize(one_repeat_size);
    LocalTensor<T> brcb_buf = tmp_buf[one_repeat_size];
    for (uint32_t i = 0; i < range_m; i++) {
      GetSrcTensorWithoutStride(tmp_src, src[src_offset], one_repeat_size, src_k);
      TwoDimBroadCastLastDimAlign(dst[dst_offset], tmp_src, brcb_buf, one_repeat_size, dst_k);
      dst_offset += one_repeat_size * dst_k;
      src_offset += one_repeat_size * src_k;
    }
    if (tail_m != 0) {
      GetSrcTensorWithoutStride(tmp_src, src[src_offset], tail_m, src_k);
      TwoDimBroadCastLastDimAlign(dst[dst_offset], tmp_src, brcb_buf, tail_m, dst_k);
    }
  } else {
    GetBrcNotAlignLoopNumbers<T>(dst_m, dst_k, tmp_buf.GetSize(), one_repeat_size, range_m, tail_m);
    LocalTensor<T> tmp_src = tmp_buf;
    tmp_src.SetSize(one_repeat_size);
    LocalTensor<T> brcb_buf = tmp_buf[one_repeat_size];
    for (uint32_t i = 0; i < range_m; i++) {
      GetSrcTensorWithoutStride(tmp_src, src[src_offset], one_repeat_size, src_k);
      TwoDimBroadCastLastDimNotAlign(dst[dst_offset], tmp_src, brcb_buf, one_repeat_size, dst_k);
      dst_offset += one_repeat_size * dst_k;
      src_offset += one_repeat_size * src_k;
    }
    if (tail_m != 0) {
      GetSrcTensorWithoutStride(tmp_src, src[src_offset], tail_m, src_k);
      TwoDimBroadCastLastDimNotAlign(dst[dst_offset], tmp_src, brcb_buf, tail_m, dst_k);
    }
  }
}

template <typename T>
inline __aicore__ void BroadcastMiddleDimWithCopy(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                                  const uint32_t src_m, const uint32_t src_k, const uint32_t src_z,
                                                  const uint32_t dst_m, const uint32_t dst_k, const uint32_t dst_z,
                                                  LocalTensor<uint8_t> &tmp_buf, const uint32_t last_dim_stride = 1) {
  const uint32_t max_rpt_cnt = dst_m / MAX_REPEAT_TIME;
  uint32_t calc_size = 0;
  const uint32_t dst_inner_offset = dst_k * dst_z;
  AscendC::SetMaskCount();
  AscendC::SetVectorMask<T, MaskMode::COUNTER>(dst_inner_offset);
  const uint16_t dst_rpt_stride = dst_inner_offset * sizeof(T) / ONE_BLK_SIZE;
  const uint32_t max_rpt_calc_num = dst_inner_offset * MAX_REPEAT_TIME;
  AscendC::CopyRepeatParams repeat_params{1, 0, dst_rpt_stride, 1};
  for (uint32_t idx = 0; idx < max_rpt_cnt; idx++) {
    AscendC::Copy<T, false>(dst[calc_size], src, MASK_PLACEHOLDER, MAX_REPEAT_TIME, repeat_params);
    calc_size += max_rpt_calc_num;
  }
  uint32_t tail_rpt_times = dst_m - max_rpt_cnt * MAX_REPEAT_TIME;
  if (tail_rpt_times != 0) {
    AscendC::Copy<T, false>(dst[calc_size], src, MASK_PLACEHOLDER, tail_rpt_times, repeat_params);
  }
  AscendC::SetMaskNorm();
  AscendC::ResetMask();
}

template <typename T>
inline __aicore__ void BroadcastCommon(const LocalTensor<T> &dst, const LocalTensor<T> &src, const uint32_t src_m,
                                       const uint32_t src_k, const uint32_t src_z, const uint32_t dst_m,
                                       const uint32_t dst_k, const uint32_t dst_z, LocalTensor<uint8_t> &tmp_buf,
                                       const uint32_t last_dim_stride = 1) {
  uint32_t inner_offset = src_z == 0 ? 1 : src_z;
  uint32_t dst_inner_offset = dst_k * inner_offset;
  if (src_m == 1 && src_k == dst_k && src_z == dst_z) {  //(1,B) ->(A,B) or (1,B,C) ->(A,B,C)
    BroadcastFirstDim(dst, src, src_m, src_k, src_z, dst_m, dst_k, dst_z, tmp_buf);
  } else if ((src_k == 1) && (src_m == dst_m) && (src_z == dst_z)) {
    if (src_z == 0) {  // (A,1) -> (A,B)
      if (last_dim_stride > 1) {
        LocalTensor<T> tmp_t_buf = tmp_buf.template ReinterpretCast<T>();
        AscendC::SetMaskCount();
        BroadcastWithStride(dst, src, src_m, last_dim_stride, src_z, dst_m, dst_k, dst_z, tmp_t_buf);
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
      } else {
        const uint32_t dst_shape[2]{dst_m, dst_k};
        const uint32_t src_shape[2]{src_m, src_k};
        AscendC::Broadcast<T, 2, 1>(dst, src, dst_shape, src_shape, tmp_buf);
      }
    } else {  // (A,1,C) -> (A,B,C)
      if (dst_z <= KernelUtils::BlkSize<T>()) {
        BroadcastMiddleDimWithCopy(dst, src, src_m, src_k, src_z, dst_m, dst_k, dst_z, tmp_buf);
      } else {
        for (int i = 0; i < src_m; i++) {
          BroadcastMiddleDim(dst[i * dst_inner_offset], src[i * src_z], src_k, src_z, 0, dst_k, dst_z, 0, tmp_buf);
        }
      }
    }
  } else if (src_k == 1 && src_m == 1 && src_z == 0 && dst_z == 0) {  // (1,1) -> (A,B)
    const uint32_t dst_shape[2]{dst_m, dst_k};
    const uint32_t src_shape[2]{src_m, src_k};
    AscendC::Broadcast<T, 2, 1>(dst, src, dst_shape, src_shape, tmp_buf);
  } else {
    ASSERT(false && "Broadcast size is not supported.");
  }
}

template <typename T>
inline __aicore__ void BroadcastWithCast(const LocalTensor<T> &dst, const LocalTensor<T> &src, const uint32_t src_m,
                                         const uint32_t src_k, const uint32_t src_z, const uint32_t dst_m,
                                         const uint32_t dst_k, const uint32_t dst_z, LocalTensor<uint8_t> &tmp_buf,
                                         const uint32_t last_dim_stride = 1) {
  if (src_z != dst_z) {
    ASSERT(false && "Broadcast inner_axis mismatch is not supported.");
  }
  LocalTensor<half> dup_tmp = tmp_buf.template ReinterpretCast<half>();
  if (src_m == 1 && src_k == dst_k) {  //(1,B) ->(A,B) or (1,B,C) ->(A,B,C)
    uint32_t inner_offset = src_z == 0 ? 1 : src_z;
    constexpr uint32_t kRatio = sizeof(uint16_t) / sizeof(T);
    uint32_t dst_inner_offset = dst_k * inner_offset / kRatio;
    LocalTensor<uint16_t> src_tmp = src.template ReinterpretCast<uint16_t>();
    LocalTensor<uint16_t> dst_tmp = dst.template ReinterpretCast<uint16_t>();
    BroadcastFirstDim(dst_tmp, src_tmp, src_m, dst_inner_offset, 0, dst_m, dst_inner_offset, 0, tmp_buf);
  } else if (src_k == 1 && src_m == dst_m) {
    if (src_z == 0) {
      const uint32_t dst_shape[2]{dst_m, dst_k};
      const uint32_t src_shape[2]{src_m, src_k};
      AscendC::Broadcast<T, 2, 1>(dst, src, dst_shape, src_shape, tmp_buf);
    } else {
      const uint8_t scalar = 0.0;
      const uint32_t dst_shape[2]{dst_k, dst_z};
      const uint32_t src_shape[2]{src_k, src_z};
      uint32_t inner_offset = src_z == 0 ? 1 : src_z;
      uint32_t dst_inner_offset = dst_k * inner_offset;
      for (int i = 0; i < src_m; i++) {
        AscendC::Broadcast<T, 2, 0>(dst[i * dst_inner_offset], src[i * inner_offset], dst_shape, src_shape, tmp_buf);
      }
    }
  } else if (src_k == 1 && src_m == 1 && src_z == 0 && dst_z == 0) {  // (1,1) -> (A,B)
    const uint32_t src_shape[2]{src_m, src_k};
    const uint32_t dst_shape[2]{dst_m, dst_k};
    AscendC::Broadcast<T, 2, 1>(dst, src, dst_shape, src_shape, tmp_buf);
  } else {
    ASSERT(false && "Broadcast size is not supported.");
  }
}

inline __aicore__ void BroadcastInt64LastDim(const LocalTensor<int32_t> &dst_int32,
                                             const LocalTensor<int32_t> &src_int32, const uint32_t src_m,
                                             const uint32_t src_k, const uint32_t src_z, const uint32_t dst_m,
                                             const uint32_t dst_k, const uint32_t dst_z, LocalTensor<int32_t> &calc_buf,
                                             const uint32_t last_dim_stride = 1) {
  constexpr uint32_t ONE_LOOP_CALC_NUM = 2;
  auto event_id_vs = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
  auto event_id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
  AscendC::SetMaskNorm();
  const uint32_t loop_cnt = src_m / ONE_LOOP_CALC_NUM;
  uint32_t calc_buf_offset = 0;
  constexpr uint32_t one_loop_offset = ONE_BLK_SIZE / sizeof(int32_t);
  uint32_t src_offset = 0;
  AscendC::SetFlag<HardEvent::V_S>(event_id_vs);
  AscendC::WaitFlag<HardEvent::V_S>(event_id_vs);
  for (uint32_t loop = 0; loop < loop_cnt; loop++) {
    int32_t calc_element_1 = src_int32.GetValue(src_offset++);
    AscendC::SetFlag<HardEvent::S_V>(event_id);
    AscendC::WaitFlag<HardEvent::S_V>(event_id);
    AscendC::SetVectorMask<int32_t, MaskMode::NORMAL>(0, 0x55);
    AscendC::Duplicate<int32_t, false>(calc_buf[calc_buf_offset], calc_element_1, MASK_PLACEHOLDER, 1, 1, 0);
    int32_t calc_element_2 = src_int32.GetValue(src_offset++);
    AscendC::SetFlag<HardEvent::S_V>(event_id);
    AscendC::WaitFlag<HardEvent::S_V>(event_id);
    AscendC::SetVectorMask<int32_t, MaskMode::NORMAL>(0, 0xaa);
    AscendC::Duplicate<int32_t, false>(calc_buf[calc_buf_offset], calc_element_2, MASK_PLACEHOLDER, 1, 1, 0);
    calc_buf_offset += one_loop_offset;
    int32_t calc_element_3 = src_int32.GetValue(src_offset++);
    AscendC::SetFlag<HardEvent::S_V>(event_id);
    AscendC::WaitFlag<HardEvent::S_V>(event_id);
    AscendC::SetVectorMask<int32_t, MaskMode::NORMAL>(0, 0x55);
    AscendC::Duplicate<int32_t, false>(calc_buf[calc_buf_offset], calc_element_3, MASK_PLACEHOLDER, 1, 1, 0);
    int32_t calc_element_4 = src_int32.GetValue(src_offset++);
    AscendC::SetFlag<HardEvent::S_V>(event_id);
    AscendC::WaitFlag<HardEvent::S_V>(event_id);
    AscendC::SetVectorMask<int32_t, MaskMode::NORMAL>(0, 0xaa);
    AscendC::Duplicate<int32_t, false>(calc_buf[calc_buf_offset], calc_element_4, MASK_PLACEHOLDER, 1, 1, 0);
    calc_buf_offset += one_loop_offset;
  }
  const uint32_t remain_cnt = src_m - loop_cnt * ONE_LOOP_CALC_NUM;
  if (remain_cnt != 0) {
    int32_t calc_element = src_int32.GetValue(src_offset++);
    AscendC::SetFlag<HardEvent::S_V>(event_id);
    AscendC::WaitFlag<HardEvent::S_V>(event_id);
    AscendC::SetVectorMask<int32_t, MaskMode::NORMAL>(0, 0x55);
    AscendC::Duplicate<int32_t, false>(calc_buf[calc_buf_offset], calc_element, MASK_PLACEHOLDER, 1, 1, 0);
    int32_t calc_element_2 = src_int32.GetValue(src_offset++);
    AscendC::SetFlag<HardEvent::S_V>(event_id);
    AscendC::WaitFlag<HardEvent::S_V>(event_id);
    AscendC::SetVectorMask<int32_t, MaskMode::NORMAL>(0, 0xaa);
    AscendC::Duplicate<int32_t, false>(calc_buf[calc_buf_offset], calc_element_2, MASK_PLACEHOLDER, 1, 1, 0);
  }
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::SetMaskCount();
  constexpr uint32_t RATIAO = sizeof(int64_t) / sizeof(int32_t);
  const uint32_t calc_cnt = dst_k * RATIAO;
  AscendC::SetVectorMask<int32_t, MaskMode::COUNTER>(calc_cnt);
  AscendC::CopyRepeatParams repeat_params{1, 0, uint16_t(dst_k * sizeof(int64_t) / ONE_BLK_SIZE), 1};
  const uint32_t max_repeat_count = src_m / MAX_REPEAT_TIMES;
  const uint32_t dst_stride = MAX_REPEAT_TIMES * calc_cnt;
  constexpr uint32_t src_stride = MAX_REPEAT_TIMES * ONE_BLK_SIZE / sizeof(int32_t);
  uint32_t buf_offset = 0;
  uint32_t dst_offset = 0;
  for (uint32_t rpt_time = 0; rpt_time < max_repeat_count; rpt_time++) {
    AscendC::Copy<int32_t, false>(dst_int32[dst_offset], calc_buf[buf_offset], MASK_PLACEHOLDER, MAX_REPEAT_TIMES,
                                  repeat_params);
    dst_offset += dst_stride;
    buf_offset += src_stride;
  }
  uint32_t tail_rpt_times = src_m - max_repeat_count * MAX_REPEAT_TIMES;
  if (tail_rpt_times > 0) {
    AscendC::Copy<int32_t, false>(dst_int32[dst_offset], calc_buf[buf_offset], MASK_PLACEHOLDER, tail_rpt_times,
                                  repeat_params);
  }
}

// 当内轴长度不是一个完整数据块时，按标量逐个复制 int64 行。
// CopyRepeatParams 只能表达完整数据块的步长，直接使用会漏掉尾部元素，
// 或者把地址推进到错误的行。
template <typename T>
inline __aicore__ void BroadcastInt64MiddleDimUnaligned(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                                        const uint32_t src_m, const uint32_t src_z,
                                                        const uint32_t dst_k, const uint32_t dst_z) {
  // 标量读写与向量流水线之间需要同步，保证前序向量操作已经完成。
  auto event_id_v_to_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
  auto event_id_s_to_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
  AscendC::SetFlag<HardEvent::V_S>(event_id_v_to_s);
  AscendC::WaitFlag<HardEvent::V_S>(event_id_v_to_s);
  for (uint32_t i = 0; i < src_m; ++i) {
    const uint32_t src_row_offset = i * src_z;
    const uint32_t dst_base_offset = i * dst_k * dst_z;
    for (uint32_t j = 0; j < dst_k; ++j) {
      for (uint32_t k = 0; k < src_z; ++k) {
        dst.SetValue(dst_base_offset + j * dst_z + k, src.GetValue(src_row_offset + k));
      }
    }
  }
  AscendC::SetFlag<HardEvent::S_V>(event_id_s_to_v);
  AscendC::WaitFlag<HardEvent::S_V>(event_id_s_to_v);
}

template <typename T>
inline __aicore__ void BroadcastInt64(const LocalTensor<T> &dst, const LocalTensor<T> &src, const uint32_t src_m,
                                      const uint32_t src_k, const uint32_t src_z, const uint32_t dst_m,
                                      const uint32_t dst_k, const uint32_t dst_z, LocalTensor<uint8_t> &tmp_buf,
                                      const uint32_t last_dim_stride = 1) {
  if (src_z != dst_z) {
    // ASSERT 在 release 下为空宏，这里补一条错误日志避免静默不写数据
    KERNEL_LOG(KERNEL_ERROR, "BroadcastInt64 inner_axis mismatch is not supported, src_z %u, dst_z %u.", src_z, dst_z);
    ASSERT(false);
    return;
  }
  LocalTensor<int32_t> dst_int32 = dst.template ReinterpretCast<int32_t>();
  LocalTensor<int32_t> src_int32 = src.template ReinterpretCast<int32_t>();
  constexpr uint32_t kRatio = sizeof(T) / sizeof(int32_t);
  uint32_t inner_offset = src_z == 0 ? 1 : src_z;
  uint32_t dst_inner_offset = dst_k * inner_offset * kRatio;
  if (src_m == 1 && src_k == dst_k) {  //(1,B) ->(A,B) or (1,B,C) ->(A,B,C)
    BroadcastFirstDim(dst_int32, src_int32, src_m, dst_inner_offset, 0, dst_m, dst_inner_offset, 0, tmp_buf);
  } else if (src_k == 1 && src_m == dst_m) {
    if (src_z == 0) {  // (A,1) -> (A,B)
      const uint32_t buf_max_calc_num = tmp_buf.GetSize() * sizeof(uint8_t) / ONE_BLK_SIZE;
      LocalTensor<int32_t> calc_buf = tmp_buf.template ReinterpretCast<int32_t>();
      uint32_t buf_loop_cnt = src_m * src_k / buf_max_calc_num;
      uint32_t src_offset = 0;
      uint32_t dst_offset = 0;
      const uint32_t dst_stride = buf_max_calc_num * dst_inner_offset;
      const uint32_t src_stride = buf_max_calc_num * kRatio;
      for (uint32_t loop_idx = 0; loop_idx < buf_loop_cnt; loop_idx++) {
        BroadcastInt64LastDim(dst_int32[dst_offset], src_int32[src_offset], buf_max_calc_num, src_k, src_z,
                              buf_max_calc_num, dst_k, dst_z, calc_buf, last_dim_stride);
        dst_offset += dst_stride;
        src_offset += src_stride;
      }
      const uint32_t tail_calc_num = src_m - buf_loop_cnt * buf_max_calc_num;
      if (tail_calc_num != 0) {
        BroadcastInt64LastDim(dst_int32[dst_offset], src_int32[src_offset], tail_calc_num, src_k, src_z, tail_calc_num,
                              dst_k, dst_z, calc_buf, last_dim_stride);
      }
      AscendC::SetMaskNorm();
      AscendC::ResetMask();
    } else {  // (A,1,C) -> (A,B,C)：按 int32 视图逐行复制，与 BroadcastCommon 保持相同布局
      const uint32_t src_inner_offset = src_z * kRatio;
      constexpr uint32_t one_blk_num = ONE_BLK_SIZE / sizeof(int32_t);
      if (src_inner_offset % one_blk_num == 0) {
        // 内轴已按块对齐，可以使用高效的向量广播实现。
        for (uint32_t i = 0; i < src_m; i++) {
          BroadcastMiddleDim(dst_int32[i * dst_inner_offset], src_int32[i * src_inner_offset], src_k, src_inner_offset,
                             0, dst_k, src_inner_offset, 0, tmp_buf);
        }
      } else {
        // 内轴未对齐时改用标量路径，确保尾部元素和行步长都正确。
        BroadcastInt64MiddleDimUnaligned(dst, src, src_m, src_z, dst_k, dst_z);
      }
    }
  } else if (src_k == 1 && src_m == 1 && src_z == 0 && dst_z == 0) {  // (1,1) -> (A,B)
    const T scalarVlue = src.GetValue(0);
    auto event_id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    AscendC::SetFlag<HardEvent::S_V>(event_id);
    AscendC::WaitFlag<HardEvent::S_V>(event_id);
    Duplicate(dst, scalarVlue, dst_m * dst_k, tmp_buf);
  } else {
    KERNEL_LOG(KERNEL_ERROR, "BroadcastInt64 size is not supported, src %ux%ux%u -> dst %ux%ux%u.", src_m, src_k, src_z,
               dst_m, dst_k, dst_z);
    ASSERT(false);
  }
}

template <typename T>
inline __aicore__ void Broadcast(const LocalTensor<T> &dst, const LocalTensor<T> &src, const uint32_t src_m,
                                 const uint32_t src_k, const uint32_t src_z, const uint32_t dst_m, const uint32_t dst_k,
                                 const uint32_t dst_z, LocalTensor<uint8_t> &tmp_buf,
                                 const uint32_t last_dim_stride = 1) {
  if constexpr (AscendC::SupportType<T, int64_t, uint64_t>()) {
    BroadcastInt64(dst, src, src_m, src_k, src_z, dst_m, dst_k, dst_z, tmp_buf, last_dim_stride);
  } else if constexpr (AscendC::SupportType<T, uint8_t, int8_t>()) {
    BroadcastWithCast(dst, src, src_m, src_k, src_z, dst_m, dst_k, dst_z, tmp_buf, last_dim_stride);
  } else if constexpr (AscendC::SupportType<T, int16_t, uint16_t, half, float, int32_t, uint32_t>()) {
    BroadcastCommon(dst, src, src_m, src_k, src_z, dst_m, dst_k, dst_z, tmp_buf, last_dim_stride);
  } else {
    KERNEL_LOG(KERNEL_ERROR, "Broadcast type is not supported.");
    ASSERT(false);
  }
}

template <typename T>
inline __aicore__ void Broadcast(const LocalTensor<T> &dst, const LocalTensor<T> &src, const uint32_t src_m,
                                 const uint32_t src_n, const uint32_t src_k, const uint32_t src_z, const uint32_t dst_m,
                                 const uint32_t dst_n, const uint32_t dst_k, const uint32_t dst_z,
                                 LocalTensor<uint8_t> &tmp_buf, const uint32_t last_dim_stride = 1) {
  if (src_m == 1 && src_k == 1 && src_n == dst_n &&
      src_z == dst_z) {  // (1,B,1,B) -> (A, B, A, B), broadcast first and third dim
    Broadcast(dst, src, src_n, src_k, src_z, dst_n, dst_k, dst_z, tmp_buf);  // (B, 1, B) -> (B, A, B)
    AscendC::PipeBarrier<PIPE_V>();
    const uint32_t offset = dst_n * dst_k * dst_z;
    Broadcast(dst[offset], dst, src_m, offset, 0, dst_m - 1, offset, 0, tmp_buf);  // (1, BAB) -> (A, BAB)
  } else if (src_n == 1 && src_z == 1) {  // (A, 1, A, 1) -> (A, B, A, B), broadcast second and fourth dim
    const uint32_t offset = dst_m * src_n * dst_k;
    ASSERT((tmp_buf.GetSize() > (offset * dst_z + dst_m * dst_n) * sizeof(T)) && "tmp_buf size is not enough.");
    LocalTensor<T> inter_buf = tmp_buf.template ReinterpretCast<T>();
    inter_buf.SetSize(offset * dst_z);
    LocalTensor<uint8_t> left_buf = tmp_buf[offset * dst_z * sizeof(T)];
    Broadcast(inter_buf, src, offset, 1, 0, offset, dst_z, 0, left_buf, last_dim_stride);  // (A1A, 1) -> (A1A, B)
    AscendC::PipeBarrier<PIPE_V>();
    // (A, 1, AB) -> (A, B, AB)
    Broadcast(dst, inter_buf, src_m, 1, dst_k * dst_z, dst_m, dst_n, dst_k * dst_z, left_buf);
  } else {
    KERNEL_LOG(KERNEL_ERROR, "Broadcast type is not supported.");
    ASSERT(false);
  }
}

#endif  // __ASCENDC_API_BROADCAST_H__
