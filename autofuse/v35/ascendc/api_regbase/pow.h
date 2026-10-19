/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_POW_H__
#define __ASCENDC_API_POW_H__

static constexpr AscendC::PowerConfig pow_config = {PowerAlgo::DOUBLE_FLOAT_TECH};

template <typename T>
inline __aicore__ void Pow(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src1,
                           const AscendC::LocalTensor<T> &src2, AscendC::LocalTensor<uint8_t> &tmp_buf,
                           const uint32_t calCount) {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, bfloat16_t>) {
    Power<T, false, pow_config>(dst, src1, src2, tmp_buf, calCount);
  } else {
    Power(dst, src1, src2, tmp_buf, calCount);
  }
}

template <typename T>
inline __aicore__ void Pow(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src1, const T &src2,
                           AscendC::LocalTensor<uint8_t> &tmp_buf, const uint32_t calCount) {
  if (static_cast<float>(src2) == 0.0f) {
    Duplicate(dst, static_cast<T>(1.0), calCount);
    return;
  } else if (static_cast<float>(src2) == 1.0f) {
    DataCopy(dst, src1, KernelUtils::BlkAlign<T>(calCount));
    return;
  } else if (static_cast<float>(src2) == -1.0f) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
      Reciprocal(dst, src1, calCount);
      return;
    }
  } else if (static_cast<float>(src2) == 2.0f) {
    if constexpr (!std::is_same_v<T, int8_t> && !std::is_same_v<T, uint8_t>) {
      Mul(dst, src1, src1, calCount);
      return;
    }
  } else if (static_cast<float>(src2) == 0.5f) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
      Sqrt(dst, src1, calCount);
      return;
    }
  } else if (static_cast<float>(src2) == -0.5f) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
      Rsqrt(dst, src1, calCount);
      return;
    }
  } else if (static_cast<float>(src2) == 3.0f) {
    if constexpr (!std::is_same_v<T, int8_t> && !std::is_same_v<T, uint8_t>) {
      Mul(dst, src1, src1, calCount);
      Mul(dst, dst, src1, calCount);
      return;
    }
  } else if constexpr (std::is_same_v<T, float>) {
    if (IsIntegerWithinUint32Magnitude(src2)) {
      const bool is_negative = src2 < 0.0f;
      const float abs_src2 = is_negative ? -src2 : src2;
      uint32_t src_u32 = static_cast<uint32_t>(abs_src2);
      LocalTensor<T> base_buf = tmp_buf.template ReinterpretCast<T>();
      DataCopy(base_buf, src1, KernelUtils::BlkAlign<T>(calCount));
      Duplicate(dst, static_cast<float>(1.0), calCount);
      while (src_u32 > 0) {
        if (src_u32 & 0x01) {
          Mul(dst, dst, base_buf, calCount);
        }
        src_u32 >>= 1;
        Mul(base_buf, base_buf, base_buf, calCount);
      }
      if (is_negative) {
        Reciprocal(dst, dst, calCount);
      }
    } else {
      Power<T, false, pow_config>(dst, src1, src2, tmp_buf, calCount);
    }
    return;
  } else if constexpr (std::is_same_v<T, bfloat16_t>) {
    Power<T, false, pow_config>(dst, src1, src2, tmp_buf, calCount);
    return;
  }
  Power(dst, src1, src2, tmp_buf, calCount);
}

template <typename T>
inline __aicore__ void Pow(const AscendC::LocalTensor<T> &dst, const T &src1, const AscendC::LocalTensor<T> &src2,
                           AscendC::LocalTensor<uint8_t> &tmp_buf, const uint32_t calCount) {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, bfloat16_t>) {
    Power<T, false, pow_config>(dst, src1, src2, tmp_buf, calCount);
  } else {
    Power(dst, src1, src2, tmp_buf, calCount);
  }
}

template <typename T>
inline __aicore__ void Pow(const AscendC::LocalTensor<T> &dst, const T &src1, const T &src2,
                           AscendC::LocalTensor<uint8_t> &tmp_buf, const uint32_t calCount) {
  auto block_cnt = KernelUtils::BlkSize<T>();
  // 将src1扩充为一个blockSize的tensor
  LocalTensor<T> src1_buf = tmp_buf.template ReinterpretCast<T>();
  src1_buf.SetSize(block_cnt);
  LocalTensor<T> dst_buf = tmp_buf[ONE_BLK_SIZE].template ReinterpretCast<T>();
  dst_buf.SetSize(block_cnt);
  bool need_power = true;

  LocalTensor<uint8_t> left_tmp_buf = tmp_buf[2 * ONE_BLK_SIZE].template ReinterpretCast<uint8_t>();
  Duplicate(src1_buf, src1, block_cnt);
  // 调用Power基础API：tensor(block size) + scalar
  if (static_cast<float>(src2) == 0.0f) {
    Duplicate(dst, static_cast<T>(1.0), calCount);
    return;
  } else if (static_cast<float>(src2) == 1.0f) {
    Duplicate(dst, src1, calCount);
    return;
  } else if (static_cast<float>(src2) == -1.0f) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
      Reciprocal(dst_buf, src1_buf, block_cnt);
      need_power = false;
    }
  } else if (static_cast<float>(src2) == 2.0f) {
    if constexpr (!std::is_same_v<T, int8_t> && !std::is_same_v<T, uint8_t>) {
      Mul(dst_buf, src1_buf, src1_buf, block_cnt);
      need_power = false;
    }
  } else if (static_cast<float>(src2) == 0.5f) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
      Sqrt(dst_buf, src1_buf, block_cnt);
      need_power = false;
    }
  } else if (static_cast<float>(src2) == -0.5f) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
      Rsqrt(dst_buf, src1_buf, block_cnt);
      need_power = false;
    }
  } else if (static_cast<float>(src2) == 3.0f) {
    if constexpr (!std::is_same_v<T, int8_t> && !std::is_same_v<T, uint8_t>) {
      Mul(dst_buf, src1_buf, src1_buf, block_cnt);
      Mul(dst_buf, dst_buf, src1_buf, block_cnt);
      need_power = false;
    }
  } else if constexpr (std::is_same_v<T, float>) {
    if (IsIntegerWithinUint32Magnitude(src2)) {
      const bool is_negative = src2 < 0.0f;
      const float abs_src2 = is_negative ? -src2 : src2;
      uint32_t src_u32 = static_cast<uint32_t>(abs_src2);
      LocalTensor<T> base_buf = tmp_buf.template ReinterpretCast<T>();
      DataCopy(base_buf, src1_buf, KernelUtils::BlkAlign<T>(block_cnt));
      Duplicate(dst_buf, static_cast<float>(1.0), block_cnt);
      while (src_u32 > 0) {
        if (src_u32 & 0x01) {
          Mul(dst_buf, dst_buf, base_buf, block_cnt);
        }
        src_u32 >>= 1;
        Mul(base_buf, base_buf, base_buf, block_cnt);
      }
      if (is_negative) {
        Reciprocal(dst_buf, dst_buf, block_cnt);
      }
    } else {
      Power<T, false, pow_config>(dst_buf, src1_buf, src2, tmp_buf, block_cnt);
    }
    need_power = false;
  } else if constexpr (std::is_same_v<T, bfloat16_t>) {
    Power<T, false, pow_config>(dst_buf, src1_buf, src2, tmp_buf, block_cnt);
    need_power = false;
  }
  if (need_power) {
    Power(dst_buf, src1_buf, src2, tmp_buf, block_cnt);
  }
  // 取block tensor中的一个scalar元素，扩充为dst size长度的tensor
  event_t event_id_v_to_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
  AscendC::SetFlag<HardEvent::V_S>(event_id_v_to_s);
  AscendC::WaitFlag<HardEvent::V_S>(event_id_v_to_s);
  Duplicate(dst, dst_buf.GetValue(0), calCount);
}
#endif  // __ASCENDC_API_POW_H__
