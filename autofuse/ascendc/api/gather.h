/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_GATHER_H__
#define __ASCENDC_API_GATHER_H__

template <typename T1>
inline __aicore__ void ZeroGatherPadding(const LocalTensor<T1> &dst, uint32_t row, uint32_t inner_size,
                                         uint32_t row_stride, uint32_t span) {
  // Padding slots are never stored out, but downstream vector ops may read full 256-bit blocks covering
  // them; residual garbage (esp. NaN/Inf) can pollute logical outputs via reductions, so clear defensively.
  // Less than ONE_BLK_SIZE/sizeof(T1) elements per row: scalar cost is negligible vs vector Duplicate.
  const uint32_t begin = row * row_stride + inner_size;
  uint32_t end = row * row_stride + row_stride;
  if (end > span) {
    end = span;
  }
  for (uint32_t i = begin; i < end; ++i) {
    dst.SetValue(i, static_cast<T1>(0));
  }
}

// 向量 Gather 指令要求个数为 8 的倍数：非 8 倍数时按 8 倍数前缀向量 Gather + 尾部标量补齐，
// 避免大 tile 全量标量回退的性能悬崖（尾部最多 7 个元素标量）。
template <typename T1>
inline __aicore__ void GatherWithTail(const LocalTensor<T1> &dst, const LocalTensor<T1> &src0_ub,
                                      const LocalTensor<uint32_t> &src1_offsets, uint32_t dst_pos, uint32_t idx_pos,
                                      uint32_t count) {
  const uint32_t vec_count = count / 8U * 8U;
  if (vec_count != 0U) {
    AscendC::Gather(dst[dst_pos], src0_ub, src1_offsets[idx_pos], 0, vec_count);
  }
  for (uint32_t i = vec_count; i < count; ++i) {
    const uint32_t offset = src1_offsets.GetValue(idx_pos + i) / sizeof(T1);
    dst.SetValue(dst_pos + i, src0_ub.GetValue(offset));
  }
}

template <typename T1>
inline __aicore__ void GatherRows(const LocalTensor<T1> &dst, const LocalTensor<T1> &src0_ub,
                                  const LocalTensor<uint32_t> &src1_offsets, uint32_t vectorized_axis_size,
                                  uint32_t inner_size, uint32_t row_stride, uint32_t rows, uint32_t span) {
  const bool single_shot = (inner_size == 0 || row_stride == 0 || inner_size >= vectorized_axis_size);
  const bool has_scalar_tail = single_shot ? ((vectorized_axis_size % 8U) != 0U) : ((inner_size % 8U) != 0U);
  if (has_scalar_tail) {
    // src1_offsets is produced by the vector Muls in GatherUbCompute; the scalar tail in GatherWithTail reads it,
    // so a vector-to-scalar sync is required (same pattern as broadcast.h / argmax.h).
    int32_t event_id_v_to_s = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(event_id_v_to_s);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(event_id_v_to_s);
  }
  if (single_shot) {
    GatherWithTail(dst, src0_ub, src1_offsets, 0U, 0U, vectorized_axis_size);
    return;
  }
  for (uint32_t row = 0; row < rows; ++row) {
    GatherWithTail(dst, src0_ub, src1_offsets, row * row_stride, row * inner_size, inner_size);
    ZeroGatherPadding(dst, row, inner_size, row_stride, span);
  }
}

template <typename T1, typename T2>
inline __aicore__ void GatherUbCompute(const LocalTensor<T1> &dst, const LocalTensor<T1> &src0_ub,
                                       const LocalTensor<T2> &src1_ub, LocalTensor<uint8_t> &tmp_buf, uint32_t offset,
                                       uint32_t vectorized_axis_size, uint32_t inner_size, uint32_t row_stride,
                                       uint32_t rows, uint32_t span, int32_t data_bytes_t1) {
  int32_t event_id_mte2_to_v = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
  int32_t event_id_mte3_to_v = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
  AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id_mte2_to_v);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id_mte2_to_v);
  AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event_id_mte3_to_v);
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event_id_mte3_to_v);
  if constexpr (AscendC::IsSameType<T2, int64_t>::value) {
    auto src1_tmp_ub = tmp_buf[offset].template ReinterpretCast<int32_t>();
    src1_tmp_ub.SetSize(vectorized_axis_size);
    AscendC::Cast(src1_tmp_ub, src1_ub, AscendC::RoundMode::CAST_NONE, vectorized_axis_size);
    AscendC::Muls(src1_tmp_ub, src1_tmp_ub, data_bytes_t1, vectorized_axis_size);
    auto src1_offsets = src1_tmp_ub.template ReinterpretCast<uint32_t>();
    GatherRows(dst, src0_ub, src1_offsets, vectorized_axis_size, inner_size, row_stride, rows, span);
  } else {
    auto src1_mul_ub = src1_ub;
    AscendC::Muls(src1_mul_ub, src1_mul_ub, data_bytes_t1, vectorized_axis_size);
    auto src1_offsets = src1_mul_ub.template ReinterpretCast<uint32_t>();
    GatherRows(dst, src0_ub, src1_offsets, vectorized_axis_size, inner_size, row_stride, rows, span);
  }
}

template <typename T1, typename T2>
inline __aicore__ void GatherUbNotFullLoad(const LocalTensor<T1> &dst, const GlobalTensor<T1> &src0,
                                           const GlobalTensor<T2> &src1, uint32_t vectorized_axis_size,
                                           LocalTensor<uint8_t> &tmp_buf, uint32_t inner_size = 0,
                                           uint32_t row_stride = 0) {
  AscendC::DataCopyExtParams param_src1;
  param_src1.blockCount = 1;
  param_src1.blockLen = vectorized_axis_size * sizeof(T2);
  param_src1.srcStride = 0;
  param_src1.dstStride = 0;

  auto src1_ub = tmp_buf[0].template ReinterpretCast<T2>();
  src1_ub.SetSize(vectorized_axis_size);
  AscendC::DataCopyPad(src1_ub, src1, param_src1, AscendC::DataCopyPadExtParams<T2>());
  int32_t event_id_mte2_to_s = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S));
  AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(event_id_mte2_to_s);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(event_id_mte2_to_s);
  const bool padded = inner_size > 0 && row_stride > 0 && inner_size < vectorized_axis_size;
  if (!padded) {
    for (uint32_t i = 0; i < vectorized_axis_size; i++) {
      T2 index = src1_ub.GetValue(i);
      T1 value = src0.GetValue(static_cast<uint64_t>(index));
      dst.SetValue(i, value);
    }
    return;
  }
  ASCENDC_ASSERT((vectorized_axis_size % inner_size == 0), {
    KERNEL_LOG(KERNEL_ERROR, "vectorized_axis_size(%u) must be divisible by inner_size(%u)!", vectorized_axis_size,
               inner_size);
  });
  const uint32_t rows = vectorized_axis_size / inner_size;
  const uint32_t span = (rows - 1U) * row_stride + inner_size;
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t i = 0; i < inner_size; ++i) {
      T2 index = src1_ub.GetValue(row * inner_size + i);
      T1 value = src0.GetValue(static_cast<uint64_t>(index));
      dst.SetValue(row * row_stride + i, value);
    }
    ZeroGatherPadding(dst, row, inner_size, row_stride, span);
  }
}

template <typename T1, typename T2>
inline __aicore__ void GatherUbFullLoad(const LocalTensor<T1> &dst, const GlobalTensor<T1> &src0,
                                        const GlobalTensor<T2> &src1, uint32_t param_last_axis_size,
                                        uint32_t vectorized_axis_size, uint32_t param_last_axis_bytes,
                                        uint32_t vectorized_axis_bytes, LocalTensor<uint8_t> &tmp_buf,
                                        uint32_t inner_size = 0, uint32_t row_stride = 0) {
  AscendC::DataCopyExtParams param_src0;
  param_src0.blockCount = 1;
  param_src0.blockLen = param_last_axis_size * sizeof(T1);
  param_src0.srcStride = 0;
  param_src0.dstStride = 0;

  AscendC::DataCopyExtParams param_src1;
  param_src1.blockCount = 1;
  param_src1.blockLen = vectorized_axis_size * sizeof(T2);
  param_src1.srcStride = 0;
  param_src1.dstStride = 0;

  auto src0_ub = tmp_buf[0].template ReinterpretCast<T1>();
  src0_ub.SetSize(param_last_axis_size);
  uint32_t offset = param_last_axis_bytes;
  auto src1_ub = tmp_buf[offset].template ReinterpretCast<T2>();
  src1_ub.SetSize(vectorized_axis_size);
  offset += vectorized_axis_bytes;

  int32_t event_id_v_to_mte2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
  int32_t event_id_v_to_mte3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
  int32_t event_id_mte3_to_mte2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
  int32_t event_id_mte2_to_mte3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));

  AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id_v_to_mte2);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event_id_v_to_mte2);
  AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id_mte3_to_mte2);
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id_mte3_to_mte2);

  AscendC::DataCopyPad(src0_ub, src0, param_src0, AscendC::DataCopyPadExtParams<T1>());
  AscendC::DataCopyPad(src1_ub, src1, param_src1, AscendC::DataCopyPadExtParams<T2>());

  const bool padded = inner_size > 0 && row_stride > 0 && inner_size < vectorized_axis_size;
  ASCENDC_ASSERT((inner_size == 0U || vectorized_axis_size % inner_size == 0U), {
    KERNEL_LOG(KERNEL_ERROR, "vectorized_axis_size(%u) must be divisible by inner_size(%u)!", vectorized_axis_size,
               inner_size);
  });
  const uint32_t rows = padded ? vectorized_axis_size / inner_size : 0U;
  const uint32_t span = padded ? (rows - 1U) * row_stride + inner_size : 0U;
  GatherUbCompute(dst, src0_ub, src1_ub, tmp_buf, offset, vectorized_axis_size, inner_size, row_stride, rows, span,
                  sizeof(T1));

  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event_id_v_to_mte3);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event_id_v_to_mte3);
  AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id_mte2_to_mte3);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id_mte2_to_mte3);
}

template <typename T1, typename T2>
inline __aicore__ void GatherExtend(const LocalTensor<T1> &dst, const GlobalTensor<T1> &src0,
                                    const GlobalTensor<T2> &src1, uint32_t param_last_axis_size,
                                    uint32_t vectorized_axis_size, LocalTensor<uint8_t> &tmp_buf,
                                    uint32_t inner_size = 0, uint32_t row_stride = 0) {
  const bool padded = inner_size > 0 && row_stride > 0 && inner_size < vectorized_axis_size;
  if (tmp_buf.GetSize() < vectorized_axis_size * sizeof(T2)) {  // tmpbuffer smaller than index space: read both from GM
    if (!padded) {
      for (uint32_t i = 0; i < vectorized_axis_size; i++) {
        T2 index = src1.GetValue(i);
        T1 value = src0.GetValue(static_cast<uint64_t>(index));
        dst.SetValue(i, value);
      }
      return;
    }
    ASCENDC_ASSERT((vectorized_axis_size % inner_size == 0), {
      KERNEL_LOG(KERNEL_ERROR, "vectorized_axis_size(%u) must be divisible by inner_size(%u)!", vectorized_axis_size,
                 inner_size);
    });
    const uint32_t rows = vectorized_axis_size / inner_size;
    const uint32_t span = (rows - 1U) * row_stride + inner_size;
    for (uint32_t row = 0; row < rows; ++row) {
      for (uint32_t i = 0; i < inner_size; ++i) {
        T2 index = src1.GetValue(row * inner_size + i);
        T1 value = src0.GetValue(static_cast<uint64_t>(index));
        dst.SetValue(row * row_stride + i, value);
      }
      ZeroGatherPadding(dst, row, inner_size, row_stride, span);
    }
    return;
  }
  uint32_t param_last_axis_bytes = AscendC::AlignUp(param_last_axis_size * sizeof(T1), ONE_BLK_SIZE);
  uint32_t vectorized_axis_bytes = AscendC::AlignUp(vectorized_axis_size * sizeof(T2), ONE_BLK_SIZE);
  uint32_t ub_full_load_bytes = 0;
  if constexpr (AscendC::IsSameType<T2, int64_t>::value) {
    ub_full_load_bytes = param_last_axis_bytes + vectorized_axis_bytes + vectorized_axis_size * sizeof(uint32_t);
  } else {
    ub_full_load_bytes = param_last_axis_bytes + vectorized_axis_bytes;
  }
  if (tmp_buf.GetSize() < ub_full_load_bytes) {  // index from UB, param from GM
    GatherUbNotFullLoad(dst, src0, src1, vectorized_axis_size, tmp_buf, inner_size, row_stride);
  } else {  // both from UB
    GatherUbFullLoad(dst, src0, src1, param_last_axis_size, vectorized_axis_size, param_last_axis_bytes,
                     vectorized_axis_bytes, tmp_buf, inner_size, row_stride);
  }
}
#endif  // __ASCENDC_API_GATHER_H__
