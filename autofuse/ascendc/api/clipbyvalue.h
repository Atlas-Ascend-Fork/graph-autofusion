/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_CLIPBYVALUE_H__
#define __ASCENDC_API_CLIPBYVALUE_H__

// 场景1，x, min, max, 是tensor
template <typename T>
inline __aicore__ void ClipByValue(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src_x,
                                   const AscendC::LocalTensor<T> &src_min, const AscendC::LocalTensor<T> &src_max,
                                   AscendC::LocalTensor<uint8_t> &tmp_buf, const uint32_t calCount) {
  const uint32_t max_rpt_cnt = KernelUtils::MaxRptSize<T>();
  const uint32_t loop_cnt = calCount / max_rpt_cnt;
  uint32_t offset = 0;
  for (uint32_t loop = 0; loop < loop_cnt; loop++) {
    AscendC::Min(dst[offset], src_x[offset], src_max[offset], max_rpt_cnt);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Max(dst[offset], dst[offset], src_min[offset], max_rpt_cnt);
    offset += max_rpt_cnt;
  }
  const uint32_t tail_cnt = calCount - offset;
  if (tail_cnt != 0) {
    AscendC::Min(dst[offset], src_x[offset], src_max[offset], tail_cnt);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Max(dst[offset], dst[offset], src_min[offset], tail_cnt);
  }
}

// 场景1的另一种实参顺序：`calCount` 在前、`tmp_buf` 在后。
//
// **这个重载只有 MLIR 路径在用**（`ascendc.clip_by_value_contiguous` 的 emitter）。
// 这一族的其余扩展包装（`where.h` 的 `Where`、`true_div.h` 的 `TrueDivExtend` 等）都是
// 「先 calCount 后 tmp_buf」，只有上面那个反过来。AFIR 侧一条路上两种顺序会让 emitter
// 多一个只为这一个 op 存在的参数，所以在这里补一个转发重载把顺序对齐。
//
// 上面那个原样保留，legacy 的 `ClipByValueApiCall` 仍按原顺序发调用，行为一个字不变；
// 但**这个头是整段内联进 kernel 的**（`ascendc_api_registry` 的 `clipbyvalue.h`），所以
// legacy 生成的 kernel 源码里会多出下面这十行——文本不同，语义相同。
//
// 两者的第 5 / 6 个形参类型不同（`LocalTensor<uint8_t>&` 与 `uint32_t`），不会有重载歧义。
template <typename T>
inline __aicore__ void ClipByValue(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src_x,
                                   const AscendC::LocalTensor<T> &src_min, const AscendC::LocalTensor<T> &src_max,
                                   const uint32_t calCount, AscendC::LocalTensor<uint8_t> &tmp_buf) {
  ClipByValue(dst, src_x, src_min, src_max, tmp_buf, calCount);
}

// 场景2，x是tensor，min, max是scalar
template <typename T>
inline __aicore__ void ClipByValue(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src_x,
                                   const T &src_min, const T &src_max, AscendC::LocalTensor<uint8_t> &tmp_buf,
                                   const uint32_t calCount) {
  constexpr auto minSize = sizeof(T) * 32;
  const auto maxTmpSize = (tmp_buf.GetSize() - minSize) / sizeof(T);
  constexpr auto repeatSize = 256 / sizeof(T);
  const auto calsize = (calCount * sizeof(T) + 31) / 32 * 32;
  if (maxTmpSize * sizeof(T) > calsize) {
    auto tmp0 = tmp_buf.ReinterpretCast<T>();
    auto sharedTmpBuffer = tmp_buf[calsize];
    AscendC::ClampMax(tmp0, src_x, sharedTmpBuffer, src_max, calCount);
    AscendC::ClampMin(dst, tmp0, sharedTmpBuffer, src_min, calCount);
  } else {
    const int32_t repeatTime = DivCeil(calCount, repeatSize);
    const int32_t maxRepeatTime = maxTmpSize / repeatSize;
    const int32_t loopTime = repeatTime / maxRepeatTime;
    const int32_t loopSize = maxRepeatTime * repeatSize;
    const int32_t tailSize = calCount - (loopSize * loopTime);
    auto tmp0 = tmp_buf.ReinterpretCast<T>();
    auto sharedTmpBuffer = tmp_buf[loopSize * sizeof(T)];
    for (int32_t i = 0; i < loopTime; i++) {
      AscendC::ClampMax(tmp0, src_x[loopSize * i], sharedTmpBuffer, src_max, loopSize);
      AscendC::ClampMin(dst[loopSize * i], tmp0, sharedTmpBuffer, src_min, loopSize);
    }
    if (tailSize > 0) {
      AscendC::ClampMax(tmp0, src_x[loopSize * loopTime], sharedTmpBuffer, src_max, tailSize);
      AscendC::ClampMin(dst[loopSize * loopTime], tmp0, sharedTmpBuffer, src_min, tailSize);
    }
  }
}

// 场景3，x、min是tensor，max是scalar
template <typename T>
inline __aicore__ void ClipByValue(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src_x,
                                   const AscendC::LocalTensor<T> &src_min, const T &src_max,
                                   AscendC::LocalTensor<uint8_t> &tmp_buf, const uint32_t calCount) {
  const uint32_t max_rpt_cnt = KernelUtils::MaxRptSize<T>();
  const uint32_t loop_cnt = calCount / max_rpt_cnt;
  uint32_t offset = 0;
  for (uint32_t loop = 0; loop < loop_cnt; loop++) {
    AscendC::Mins(dst[offset], src_x[offset], src_max, max_rpt_cnt);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Max(dst[offset], dst[offset], src_min[offset], max_rpt_cnt);
    offset += max_rpt_cnt;
  }
  const uint32_t tail_cnt = calCount - offset;
  if (tail_cnt != 0) {
    AscendC::Mins(dst[offset], src_x[offset], src_max, tail_cnt);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Max(dst[offset], dst[offset], src_min[offset], tail_cnt);
  }
}

// 场景4，x、max是tensor，min是scalar
template <typename T>
inline __aicore__ void ClipByValue(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src_x,
                                   const T &src_min, const AscendC::LocalTensor<T> &src_max,
                                   AscendC::LocalTensor<uint8_t> &tmp_buf, const uint32_t calCount) {
  const uint32_t max_rpt_cnt = KernelUtils::MaxRptSize<T>();
  const uint32_t loop_cnt = calCount / max_rpt_cnt;
  uint32_t offset = 0;
  for (uint32_t loop = 0; loop < loop_cnt; loop++) {
    AscendC::Min(dst[offset], src_x[offset], src_max[offset], max_rpt_cnt);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Maxs(dst[offset], dst[offset], src_min, max_rpt_cnt);
    offset += max_rpt_cnt;
  }
  const uint32_t tail_cnt = calCount - offset;
  if (tail_cnt != 0) {
    AscendC::Min(dst[offset], src_x[offset], src_max[offset], tail_cnt);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Maxs(dst[offset], dst[offset], src_min, tail_cnt);
  }
}

// 场景 2 / 3 / 4 的另一种实参顺序：`calCount` 在前、`tmp_buf` 在后。
//
// 与场景 1 那个转发重载（上面第 35 行那段）**同一个理由、同一个做法**：MLIR 这条路上这一族
// 的实参序只有一种（count 在 tmp 之前，与 `where.h` 一致），emitter 才能与选择族逐字共用。
// legacy 仍按各场景原来的顺序发，行为一个字不变；代价是这个头整段内联进 kernel 时多出下面
// 这些行。
//
// 三个转发都不会有重载歧义：第 5 / 6 个形参类型不同（`LocalTensor<uint8_t>&` 与 `uint32_t`）。
template <typename T>
inline __aicore__ void ClipByValue(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src_x,
                                   const T &src_min, const T &src_max, const uint32_t calCount,
                                   AscendC::LocalTensor<uint8_t> &tmp_buf) {
  ClipByValue(dst, src_x, src_min, src_max, tmp_buf, calCount);
}

template <typename T>
inline __aicore__ void ClipByValue(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src_x,
                                   const AscendC::LocalTensor<T> &src_min, const T &src_max, const uint32_t calCount,
                                   AscendC::LocalTensor<uint8_t> &tmp_buf) {
  ClipByValue(dst, src_x, src_min, src_max, tmp_buf, calCount);
}

template <typename T>
inline __aicore__ void ClipByValue(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src_x,
                                   const T &src_min, const AscendC::LocalTensor<T> &src_max, const uint32_t calCount,
                                   AscendC::LocalTensor<uint8_t> &tmp_buf) {
  ClipByValue(dst, src_x, src_min, src_max, tmp_buf, calCount);
}

// 场景4， x，min，max都是scalar
template <typename T>
inline __aicore__ void ClipByValue(const AscendC::LocalTensor<T> &dst, const T &src_x, const T &src_min,
                                   const T &src_max) {
  T val;
  uint32_t count = dst.GetSize();
  if (static_cast<float>(src_x) < static_cast<float>(src_min)) {
    val = src_min;
  } else if (static_cast<float>(src_x) > static_cast<float>(src_max)) {
    val = src_max;
  } else {
    val = src_x;
  }
  AscendC::Duplicate(dst, val, count);
}
#endif  // __ASCENDC_API_CLIPBYVALUE_H__
