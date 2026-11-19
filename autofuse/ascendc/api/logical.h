/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_LOGICAL_H__
#define __ASCENDC_API_LOGICAL_H__

template <typename T>
inline __aicore__ void CastInt64LogicalInput(const LocalTensor<half> &dst, const LocalTensor<T> &src,
                                             const LocalTensor<float> &tmp, const uint32_t size) {
  Cast(tmp, src, RoundMode::CAST_RINT, size);
  Abs(tmp, tmp, size);
  Mins(tmp, tmp, 1.0f, size);
  AscendC::PipeBarrier<PIPE_V>();
  Cast(dst, tmp, RoundMode::CAST_NONE, size);
  AscendC::PipeBarrier<PIPE_V>();
}

template <typename T, void (*ApiCall)(const LocalTensor<half> &dstLocal, const LocalTensor<half> &src0Local,
                                      const LocalTensor<half> &src1Local, uint64_t mask, const uint8_t repeatTimes,
                                      const BinaryRepeatParams &repeatParams)>
inline __aicore__ void Int64LogicalCommon(const LocalTensor<uint8_t> &dst, const LocalTensor<T> &src1,
                                          const LocalTensor<T> &src2, const uint32_t size,
                                          LocalTensor<uint8_t> &tmp_buf) {
  /*
   * 原数据类型是int64或者int32，需要将数据先转成float，然后再转成half计算，因此tmp_buf要分成4份：
   * 1）存储src1转成float后的结果
   * 2）存储src2转成float后的结果
   * 3）存储1转成half后的结果
   * 4）存储2转成half后的结果
   * 由于float的大小是half的两倍，相同数量下，1和2分配的空间应该是3、4的两倍，因此1、2各占tmp_buf的1/3，3、4各占tmp_buf的1/6
   */
  constexpr uint32_t tmp_buf_splits = 6U;
  constexpr uint32_t cast_dst_align = 32U;  // cast算子要求目标地址32字节对齐
  const uint32_t cast_buf_size = tmp_buf.GetSize() / tmp_buf_splits / cast_dst_align * cast_dst_align / sizeof(half);

  LocalTensor<float> cast_float_src1 = tmp_buf.template ReinterpretCast<float>();
  cast_float_src1.SetSize(cast_buf_size);
  int32_t offset = cast_buf_size * sizeof(float);

  LocalTensor<float> cast_float_src2 = tmp_buf[offset].template ReinterpretCast<float>();
  cast_float_src2.SetSize(cast_buf_size);
  offset += cast_buf_size * sizeof(float);

  LocalTensor<half> cast_half_src1 = tmp_buf[offset].template ReinterpretCast<half>();
  cast_half_src1.SetSize(cast_buf_size);
  offset += cast_buf_size * sizeof(half);

  LocalTensor<half> cast_half_src2 = tmp_buf[offset].template ReinterpretCast<half>();
  cast_half_src2.SetSize(cast_buf_size);

  BinaryRepeatParams repeat_params;
  repeat_params.blockNumber = size / ONE_BLK_HALF_NUM;

  uint64_t max_repeat = cast_buf_size / ONE_REPEAT_HALF_SIZE;
  max_repeat = max_repeat > MAX_REPEAT_TIME ? MAX_REPEAT_TIME : max_repeat;
  const uint32_t max_repeat_size = max_repeat * ONE_REPEAT_HALF_SIZE;

  uint32_t cal_size = 0;
  for (; cal_size + max_repeat_size < size; cal_size += max_repeat_size) {
    CastInt64LogicalInput(cast_half_src1, src1[cal_size], cast_float_src1, max_repeat_size);
    CastInt64LogicalInput(cast_half_src2, src2[cal_size], cast_float_src2, max_repeat_size);
    ApiCall(cast_half_src1, cast_half_src1, cast_half_src2, ONE_REPEAT_HALF_SIZE, max_repeat, repeat_params);
    Cast(dst[cal_size], cast_half_src1, RoundMode::CAST_NONE, max_repeat_size);
  }

  if (cal_size + ONE_REPEAT_HALF_SIZE <= size) {
    uint32_t repeat = (size - cal_size) / ONE_REPEAT_HALF_SIZE;
    uint32_t repeat_size = repeat * ONE_REPEAT_HALF_SIZE;

    CastInt64LogicalInput(cast_half_src1, src1[cal_size], cast_float_src1, repeat_size);
    CastInt64LogicalInput(cast_half_src2, src2[cal_size], cast_float_src2, repeat_size);
    ApiCall(cast_half_src1, cast_half_src1, cast_half_src2, ONE_REPEAT_HALF_SIZE, repeat, repeat_params);
    Cast(dst[cal_size], cast_half_src1, RoundMode::CAST_NONE, repeat_size);
    cal_size += repeat_size;
  }

  if (cal_size < size) {
    repeat_params.blockNumber = (size - cal_size + ONE_BLK_HALF_NUM - 1) / ONE_BLK_HALF_NUM;
    uint32_t left_size = size - cal_size;

    CastInt64LogicalInput(cast_half_src1, src1[cal_size], cast_float_src1, left_size);
    CastInt64LogicalInput(cast_half_src2, src2[cal_size], cast_float_src2, left_size);
    ApiCall(cast_half_src1, cast_half_src1, cast_half_src2, repeat_params.blockNumber * ONE_BLK_HALF_NUM, 1,
            repeat_params);
    Cast(dst[cal_size], cast_half_src1, RoundMode::CAST_NONE, left_size);
  }
}

template <void (*ApiCall)(const LocalTensor<half> &dstLocal, const LocalTensor<half> &src0Local,
                          const LocalTensor<half> &src1Local, uint64_t mask, const uint8_t repeatTimes,
                          const BinaryRepeatParams &repeatParams)>
inline __aicore__ void HalfLogicalCommon(const LocalTensor<uint8_t> &dst, const LocalTensor<half> &src1,
                                         const LocalTensor<half> &src2, const uint32_t size,
                                         LocalTensor<uint8_t> &tmp_buf) {
  /*
   * 输入是half类型，计算前不用做类型转换，tmp_buf所有空间可以全部存储ApiCall的结果
   */
  LocalTensor<half> api_buf = tmp_buf.template ReinterpretCast<half>();
  api_buf.SetSize(tmp_buf.GetSize() / sizeof(half));

  BinaryRepeatParams repeat_params;
  repeat_params.blockNumber = size / ONE_BLK_HALF_NUM;

  uint64_t max_repeat = api_buf.GetSize() / ONE_REPEAT_HALF_SIZE;
  max_repeat = max_repeat > MAX_REPEAT_TIME ? MAX_REPEAT_TIME : max_repeat;
  const uint32_t max_repeat_size = max_repeat * ONE_REPEAT_HALF_SIZE;

  uint32_t cal_size = 0;
  for (; cal_size + max_repeat_size < size; cal_size += max_repeat_size) {
    ApiCall(api_buf, src1[cal_size], src2[cal_size], ONE_REPEAT_HALF_SIZE, max_repeat, repeat_params);
    Cast(dst[cal_size], api_buf, RoundMode::CAST_NONE, max_repeat_size);
  }

  if (cal_size + ONE_REPEAT_HALF_SIZE <= size) {
    uint32_t repeat = (size - cal_size) / ONE_REPEAT_HALF_SIZE;
    uint32_t repeat_size = repeat * ONE_REPEAT_HALF_SIZE;

    ApiCall(api_buf, src1[cal_size], src2[cal_size], ONE_REPEAT_HALF_SIZE, repeat, repeat_params);
    Cast(dst[cal_size], api_buf, RoundMode::CAST_NONE, repeat_size);
    cal_size += repeat_size;
  }

  if (cal_size < size) {
    repeat_params.blockNumber = (size - cal_size + ONE_BLK_HALF_NUM - 1) / ONE_BLK_HALF_NUM;
    uint32_t left_size = size - cal_size;

    ApiCall(api_buf, src1[cal_size], src2[cal_size], repeat_params.blockNumber * ONE_BLK_HALF_NUM, 1, repeat_params);
    Cast(dst[cal_size], api_buf, RoundMode::CAST_NONE, left_size);
  }
}

template <typename T, void (*ApiCall)(const LocalTensor<half> &dstLocal, const LocalTensor<half> &src0Local,
                                      const LocalTensor<half> &src1Local, uint64_t mask, const uint8_t repeatTimes,
                                      const BinaryRepeatParams &repeatParams)>
inline __aicore__ void OtherTypeLogicalCommon(const LocalTensor<uint8_t> &dst, const LocalTensor<T> &src1,
                                              const LocalTensor<T> &src2, const uint32_t size,
                                              LocalTensor<uint8_t> &tmp_buf) {
  /*
   * 其它类型只需要经过一次转换，因此tmp_buf分成两份：
   * 1）存储src1转成half后的结果
   * 2）存储src2转成half后的结果
   */
  constexpr uint32_t tmp_buf_splits = 2U;
  constexpr uint32_t cast_dst_align = 32U;  // 地址32字节对齐
  uint32_t cast_buf_size = tmp_buf.GetSize() / tmp_buf_splits / cast_dst_align * cast_dst_align / sizeof(half);

  LocalTensor<half> cast_src1 = tmp_buf.template ReinterpretCast<half>();
  cast_src1.SetSize(cast_buf_size);
  uint32_t offset = cast_buf_size * sizeof(half);

  LocalTensor<half> cast_src2 = tmp_buf[offset].template ReinterpretCast<half>();
  cast_src2.SetSize(cast_buf_size);

  BinaryRepeatParams repeat_params;
  repeat_params.blockNumber = size / ONE_BLK_HALF_NUM;

  uint32_t cal_size = 0;
  uint64_t max_repeat = cast_buf_size / ONE_REPEAT_HALF_SIZE;
  max_repeat = max_repeat > MAX_REPEAT_TIME ? MAX_REPEAT_TIME : max_repeat;
  const uint32_t max_repeat_size = max_repeat * ONE_REPEAT_HALF_SIZE;
  for (; cal_size + max_repeat_size < size; cal_size += max_repeat_size) {
    Cast(cast_src1, src1[cal_size], RoundMode::CAST_NONE, max_repeat_size);
    Cast(cast_src2, src2[cal_size], RoundMode::CAST_NONE, max_repeat_size);
    AscendC::PipeBarrier<PIPE_V>();
    ApiCall(cast_src1, cast_src1, cast_src2, ONE_REPEAT_HALF_SIZE, max_repeat, repeat_params);
    Cast(dst[cal_size], cast_src1, RoundMode::CAST_NONE, max_repeat_size);
  }

  if (cal_size + ONE_REPEAT_HALF_SIZE <= size) {
    uint32_t repeat = (size - cal_size) / ONE_REPEAT_HALF_SIZE;
    uint32_t repeat_size = repeat * ONE_REPEAT_HALF_SIZE;

    Cast(cast_src1, src1[cal_size], RoundMode::CAST_NONE, repeat_size);
    Cast(cast_src2, src2[cal_size], RoundMode::CAST_NONE, repeat_size);
    AscendC::PipeBarrier<PIPE_V>();
    ApiCall(cast_src1, cast_src1, cast_src2, ONE_REPEAT_HALF_SIZE, repeat, repeat_params);
    Cast(dst[cal_size], cast_src1, RoundMode::CAST_NONE, repeat_size);
    cal_size += repeat_size;
  }

  if (cal_size < size) {
    repeat_params.blockNumber = (size - cal_size + ONE_BLK_HALF_NUM - 1) / ONE_BLK_HALF_NUM;
    uint32_t left_size = size - cal_size;

    Cast(cast_src1, src1[cal_size], RoundMode::CAST_NONE, left_size);
    Cast(cast_src2, src2[cal_size], RoundMode::CAST_NONE, left_size);
    AscendC::PipeBarrier<PIPE_V>();
    ApiCall(cast_src1, cast_src1, cast_src2, repeat_params.blockNumber * ONE_BLK_HALF_NUM, 1, repeat_params);
    Cast(dst[cal_size], cast_src1, RoundMode::CAST_NONE, left_size);
  }
}

template <typename T,
          void (*ApiCall)(const LocalTensor<half> &dstLocal, const LocalTensor<half> &srcLocal, const half &scalarValue,
                          uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams &repeatParams)>
inline __aicore__ void Int64LogicalCommonScalarExtend(const LocalTensor<uint8_t> &dst, const LocalTensor<T> &src1,
                                                      const T src2, const uint32_t size,
                                                      LocalTensor<uint8_t> &tmp_buf) {
  /*
   * 原数据类型是int64或者int32，需要将数据先转成float，然后再转成half计算，因此tmp_buf要分成2份：
   * 1）存储src1转成float后的结果
   * 2）存储1转成half后的结果
   * 由于float的大小是half的两倍，相同数量下，1分配的空间应该是2的两倍，因此占tmp_buf的2/3，2占tmp_buf的1/3
   */
  constexpr uint32_t tmp_buf_splits = 3U;
  constexpr uint32_t cast_dst_align = 32U;  // cast算子要求目标地址32字节对齐
  const uint32_t cast_buf_size = tmp_buf.GetSize() / tmp_buf_splits / cast_dst_align * cast_dst_align / sizeof(half);

  LocalTensor<float> cast_float_src1 = tmp_buf.template ReinterpretCast<float>();
  cast_float_src1.SetSize(cast_buf_size);
  int32_t offset = cast_buf_size * sizeof(float);

  LocalTensor<half> cast_half_src1 = tmp_buf[offset].template ReinterpretCast<half>();
  cast_half_src1.SetSize(cast_buf_size);

  UnaryRepeatParams repeat_params;
  repeat_params.blockNumber = size / ONE_BLK_HALF_NUM;

  uint64_t max_repeat = cast_buf_size / ONE_REPEAT_HALF_SIZE;
  max_repeat = max_repeat > MAX_REPEAT_TIME ? MAX_REPEAT_TIME : max_repeat;
  const uint32_t max_repeat_size = max_repeat * ONE_REPEAT_HALF_SIZE;

  float normalized_src2 = src2 == static_cast<T>(0) ? 0.0f : 1.0f;
  half cast_half_src2 = ScalarCast<float, half, RoundMode::CAST_ODD>(normalized_src2);

  uint32_t cal_size = 0;
  for (; cal_size + max_repeat_size < size; cal_size += max_repeat_size) {
    CastInt64LogicalInput(cast_half_src1, src1[cal_size], cast_float_src1, max_repeat_size);
    ApiCall(cast_half_src1, cast_half_src1, cast_half_src2, ONE_REPEAT_HALF_SIZE, max_repeat, repeat_params);
    Cast(dst[cal_size], cast_half_src1, RoundMode::CAST_NONE, max_repeat_size);
  }

  if (cal_size + ONE_REPEAT_HALF_SIZE <= size) {
    uint32_t repeat = (size - cal_size) / ONE_REPEAT_HALF_SIZE;
    uint32_t repeat_size = repeat * ONE_REPEAT_HALF_SIZE;

    CastInt64LogicalInput(cast_half_src1, src1[cal_size], cast_float_src1, repeat_size);
    ApiCall(cast_half_src1, cast_half_src1, cast_half_src2, ONE_REPEAT_HALF_SIZE, repeat, repeat_params);
    Cast(dst[cal_size], cast_half_src1, RoundMode::CAST_NONE, repeat_size);
    cal_size += repeat_size;
  }

  if (cal_size < size) {
    repeat_params.blockNumber = (size - cal_size + ONE_BLK_HALF_NUM - 1) / ONE_BLK_HALF_NUM;
    uint32_t left_size = size - cal_size;

    CastInt64LogicalInput(cast_half_src1, src1[cal_size], cast_float_src1, left_size);
    ApiCall(cast_half_src1, cast_half_src1, cast_half_src2, repeat_params.blockNumber * ONE_BLK_HALF_NUM, 1,
            repeat_params);
    Cast(dst[cal_size], cast_half_src1, RoundMode::CAST_NONE, left_size);
  }
}

template <void (*ApiCall)(const LocalTensor<half> &dstLocal, const LocalTensor<half> &srcLocal, const half &scalarValue,
                          uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams &repeatParams)>
inline __aicore__ void HalfLogicalCommonScalarExtend(const LocalTensor<uint8_t> &dst, const LocalTensor<half> &src1,
                                                     const half src2, const uint32_t size,
                                                     LocalTensor<uint8_t> &tmp_buf) {
  /*
   * 输入是half类型，计算前不用做类型转换，tmp_buf所有空间可以全部存储ApiCall的结果
   */
  LocalTensor<half> api_buf = tmp_buf.template ReinterpretCast<half>();
  api_buf.SetSize(tmp_buf.GetSize() / sizeof(half));

  UnaryRepeatParams repeat_params;
  repeat_params.blockNumber = size / ONE_BLK_HALF_NUM;

  uint64_t max_repeat = api_buf.GetSize() / ONE_REPEAT_HALF_SIZE;
  max_repeat = max_repeat > MAX_REPEAT_TIME ? MAX_REPEAT_TIME : max_repeat;
  const uint32_t max_repeat_size = max_repeat * ONE_REPEAT_HALF_SIZE;

  uint32_t cal_size = 0;
  for (; cal_size + max_repeat_size < size; cal_size += max_repeat_size) {
    ApiCall(api_buf, src1[cal_size], src2, ONE_REPEAT_HALF_SIZE, max_repeat, repeat_params);
    Cast(dst[cal_size], api_buf, RoundMode::CAST_NONE, max_repeat_size);
  }

  if (cal_size + ONE_REPEAT_HALF_SIZE <= size) {
    uint32_t repeat = (size - cal_size) / ONE_REPEAT_HALF_SIZE;
    uint32_t repeat_size = repeat * ONE_REPEAT_HALF_SIZE;

    ApiCall(api_buf, src1[cal_size], src2, ONE_REPEAT_HALF_SIZE, repeat, repeat_params);
    Cast(dst[cal_size], api_buf, RoundMode::CAST_NONE, repeat_size);
    cal_size += repeat_size;
  }

  if (cal_size < size) {
    repeat_params.blockNumber = (size - cal_size + ONE_BLK_HALF_NUM - 1) / ONE_BLK_HALF_NUM;
    uint32_t left_size = size - cal_size;

    ApiCall(api_buf, src1[cal_size], src2, repeat_params.blockNumber * ONE_BLK_HALF_NUM, 1, repeat_params);
    Cast(dst[cal_size], api_buf, RoundMode::CAST_NONE, left_size);
  }
}

template <typename T,
          void (*ApiCall)(const LocalTensor<half> &dstLocal, const LocalTensor<half> &srcLocal, const half &scalarValue,
                          uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams &repeatParams)>
inline __aicore__ void OtherTypeLogicalCommonScalarExtend(const LocalTensor<uint8_t> &dst, const LocalTensor<T> &src1,
                                                          const T src2, const uint32_t size,
                                                          LocalTensor<uint8_t> &tmp_buf) {
  /*
   * 其它类型只需要经过一次转换，因此tmp_buf所有空间可以全部存储src1转成half后的结果
   */
  uint32_t cast_buf_size = tmp_buf.GetSize() / sizeof(half);

  LocalTensor<half> cast_src1 = tmp_buf.template ReinterpretCast<half>();
  cast_src1.SetSize(cast_buf_size);

  UnaryRepeatParams repeat_params;
  repeat_params.blockNumber = size / ONE_BLK_HALF_NUM;

  float cast_float_src2 = 0;
  if constexpr (AscendC::IsSameType<T, uint8_t>::value) {
    cast_float_src2 = (src2 & 0xFF) * 1.0f;
  } else {
    cast_float_src2 = static_cast<float>(src2);
  }
  half cast_src2 = ScalarCast<float, half, RoundMode::CAST_ODD>(cast_float_src2);

  uint32_t cal_size = 0;
  uint64_t max_repeat = cast_buf_size / ONE_REPEAT_HALF_SIZE;
  max_repeat = max_repeat > MAX_REPEAT_TIME ? MAX_REPEAT_TIME : max_repeat;
  const uint32_t max_repeat_size = max_repeat * ONE_REPEAT_HALF_SIZE;
  for (; cal_size + max_repeat_size < size; cal_size += max_repeat_size) {
    Cast(cast_src1, src1[cal_size], RoundMode::CAST_NONE, max_repeat_size);
    AscendC::PipeBarrier<PIPE_V>();
    ApiCall(cast_src1, cast_src1, cast_src2, ONE_REPEAT_HALF_SIZE, max_repeat, repeat_params);
    Cast(dst[cal_size], cast_src1, RoundMode::CAST_NONE, max_repeat_size);
  }

  if (cal_size + ONE_REPEAT_HALF_SIZE <= size) {
    uint32_t repeat = (size - cal_size) / ONE_REPEAT_HALF_SIZE;
    uint32_t repeat_size = repeat * ONE_REPEAT_HALF_SIZE;

    Cast(cast_src1, src1[cal_size], RoundMode::CAST_NONE, repeat_size);
    AscendC::PipeBarrier<PIPE_V>();
    ApiCall(cast_src1, cast_src1, cast_src2, ONE_REPEAT_HALF_SIZE, repeat, repeat_params);
    Cast(dst[cal_size], cast_src1, RoundMode::CAST_NONE, repeat_size);
    cal_size += repeat_size;
  }

  if (cal_size < size) {
    repeat_params.blockNumber = (size - cal_size + ONE_BLK_HALF_NUM - 1) / ONE_BLK_HALF_NUM;
    uint32_t left_size = size - cal_size;

    Cast(cast_src1, src1[cal_size], RoundMode::CAST_NONE, left_size);
    AscendC::PipeBarrier<PIPE_V>();
    ApiCall(cast_src1, cast_src1, cast_src2, repeat_params.blockNumber * ONE_BLK_HALF_NUM, 1, repeat_params);
    Cast(dst[cal_size], cast_src1, RoundMode::CAST_NONE, left_size);
  }
}

template <typename T>
inline __aicore__ void LogicalOr(const LocalTensor<uint8_t> &dst, const LocalTensor<T> &src1,
                                 const LocalTensor<T> &src2, LocalTensor<uint8_t> &tmp_buf, const uint32_t size) {
  if constexpr (std::is_same<T, int64_t>::value || std::is_same<T, int32_t>::value) {
    Int64LogicalCommon<T, Max>(dst, src1, src2, size, tmp_buf);
  } else if constexpr (std::is_same<T, half>::value) {
    HalfLogicalCommon<Max>(dst, src1, src2, size, tmp_buf);
  } else {
    OtherTypeLogicalCommon<T, Max>(dst, src1, src2, size, tmp_buf);
  }
}

template <typename T>
inline __aicore__ void LogicalAnd(const LocalTensor<uint8_t> &dst, const LocalTensor<T> &src1,
                                  const LocalTensor<T> &src2, LocalTensor<uint8_t> &tmp_buf, const uint32_t size) {
  if constexpr (std::is_same<T, int64_t>::value || std::is_same<T, int32_t>::value) {
    Int64LogicalCommon<T, Mul>(dst, src1, src2, size, tmp_buf);
  } else if constexpr (std::is_same<T, half>::value) {
    HalfLogicalCommon<Mul>(dst, src1, src2, size, tmp_buf);
  } else {
    OtherTypeLogicalCommon<T, Mul>(dst, src1, src2, size, tmp_buf);
  }
}

template <typename T>
inline __aicore__ void LogicalOrScalarExtend(const LocalTensor<uint8_t> &dst, const LocalTensor<T> &src1, const T src2,
                                             LocalTensor<uint8_t> &tmp_buf, const uint32_t size) {
  if constexpr (std::is_same<T, int64_t>::value || std::is_same<T, int32_t>::value) {
    Int64LogicalCommonScalarExtend<T, Maxs<half, true>>(dst, src1, src2, size, tmp_buf);
  } else if constexpr (std::is_same<T, half>::value) {
    HalfLogicalCommonScalarExtend<Maxs<half, true>>(dst, src1, src2, size, tmp_buf);
  } else {
    OtherTypeLogicalCommonScalarExtend<T, Maxs<half, true>>(dst, src1, src2, size, tmp_buf);
  }
}

template <typename T>
inline __aicore__ void LogicalAndScalarExtend(const LocalTensor<uint8_t> &dst, const LocalTensor<T> &src1, const T src2,
                                              LocalTensor<uint8_t> &tmp_buf, const uint32_t size) {
  if constexpr (std::is_same<T, int64_t>::value || std::is_same<T, int32_t>::value) {
    Int64LogicalCommonScalarExtend<T, Muls<half, true>>(dst, src1, src2, size, tmp_buf);
  } else if constexpr (std::is_same<T, half>::value) {
    HalfLogicalCommonScalarExtend<Muls<half, true>>(dst, src1, src2, size, tmp_buf);
  } else {
    OtherTypeLogicalCommonScalarExtend<T, Muls<half, true>>(dst, src1, src2, size, tmp_buf);
  }
}

#endif  // __ASCENDC_API_LOGICAL_H__
