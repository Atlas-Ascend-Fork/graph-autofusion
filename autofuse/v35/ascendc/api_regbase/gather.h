/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_GATHER_H__
#define __ASCENDC_API_REGBASE_GATHER_H__
constexpr int32_t CASE1 = 0;  // 只有一根轴
constexpr int32_t CASE2 = 1;  // 对首轴gather
constexpr int32_t CASE3 = 2;  // 对尾轴gather
constexpr int32_t CASE4 = 3;  // 对中间轴gather
constexpr int32_t INT64_OFFSET = 64;
constexpr int32_t INT32_OFFSET = 32;
#define THREAD_NUMBER 2048
#ifndef AUTOFUSE_SIMT_RESERVED_UB_SIZE
#define AUTOFUSE_SIMT_RESERVED_UB_SIZE 40960  // 32 * 1024 + 8 * 1024
#endif
// 生成单个向量轴的参数声明
// 首先定义每个变量的宏
#define VECTORIZED_AXIS_SIZE_M(n) vectorized_axis_##n##_size_m
#define VECTORIZED_AXIS_SIZE_SHIFT(n) vectorized_axis_##n##_size_shift
#define VECTORIZED_AXIS_SIZE(n) vectorized_axis_##n##_size
#define VECTORIZED_AXIS_STRIDE(n) vectorized_axis_##n##_stride
#define Y_VECTORIZED_AXIS_SIZE_STRIDE(n) y_vectorized_axis_##n##_size_stride

#define DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(n)                                  \
  uint32_t VECTORIZED_AXIS_SIZE(n) = 0, uint32_t VECTORIZED_AXIS_STRIDE(n) = 0, \
           uint32_t Y_VECTORIZED_AXIS_SIZE_STRIDE(n) = 0

// 宏定义：用于生成单个向量轴的参数声明
#define DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(n)                                                                  \
  uint32_t VECTORIZED_AXIS_SIZE_M(n), uint32_t VECTORIZED_AXIS_SIZE_SHIFT(n), uint32_t VECTORIZED_AXIS_SIZE(n), \
      uint32_t VECTORIZED_AXIS_STRIDE(n), uint32_t Y_VECTORIZED_AXIS_SIZE_STRIDE(n)

// 宏定义：用于生成单个向量轴的参数声明
#define DECLARE_VECTORIZED_AXIS_PARAMS_SIMT_CONST_REF(n)                                    \
  const uint32_t &VECTORIZED_AXIS_SIZE_M(n), const uint32_t &VECTORIZED_AXIS_SIZE_SHIFT(n), \
      const uint32_t &VECTORIZED_AXIS_SIZE(n), const uint32_t &VECTORIZED_AXIS_STRIDE(n),   \
      const uint32_t &Y_VECTORIZED_AXIS_SIZE_STRIDE(n)

// 宏定义：用于传递单个向量轴的参数
#define PASS_VECTORIZED_AXIS_PARAMS_SIMT(n)                                                                     \
  VECTORIZED_AXIS_SIZE_M(n), VECTORIZED_AXIS_SIZE_SHIFT(n), VECTORIZED_AXIS_SIZE(n), VECTORIZED_AXIS_STRIDE(n), \
      Y_VECTORIZED_AXIS_SIZE_STRIDE(n)

#define PASS_VECTORIZED_AXIS_PARAMS_SIMD(n) \
  VECTORIZED_AXIS_SIZE(n), VECTORIZED_AXIS_STRIDE(n), Y_VECTORIZED_AXIS_SIZE_STRIDE(n)

// 宏定义：声明单个向量轴的 magic 和 shift 变量
#define DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_VARIABLES(n) \
  uint32_t VECTORIZED_AXIS_SIZE_M(n){0};                 \
  uint32_t VECTORIZED_AXIS_SIZE_SHIFT(n) {               \
    0                                                    \
  }

#define DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(n) \
  uint32_t &VECTORIZED_AXIS_SIZE_M(n), uint32_t &VECTORIZED_AXIS_SIZE_SHIFT(n), const uint32_t &VECTORIZED_AXIS_SIZE(n)

#define PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(n) \
  VECTORIZED_AXIS_SIZE_M(n), VECTORIZED_AXIS_SIZE_SHIFT(n), VECTORIZED_AXIS_SIZE(n)

template <typename T, AscendC::PaddingMode mode = AscendC::PaddingMode::Normal>
inline __aicore__ void GatherDataCopyPadExtend(const AscendC::LocalTensor<T> &dst, const AscendC::GlobalTensor<T> &src,
                                               uint32_t block_count, uint32_t block_len, uint32_t src_stride,
                                               uint32_t dst_stride) {
  uint32_t align_num = AscendC::ONE_BLK_SIZE / sizeof(T);
  AscendC::DataCopyExtParams param;
  param.blockCount = block_count;
  param.blockLen = block_len * sizeof(T);
  param.srcStride = src_stride * sizeof(T);
  param.dstStride = dst_stride / align_num;
  AscendC::DataCopyPadExtParams<T> pad_params = {true, 0, 0, 0};
  AscendC::DataCopyPad<T, mode>(dst, src, param, pad_params);
}

template <uint32_t VECTORIZED_AXIS_SIZE>
inline __aicore__ void GetMagicShiftForVectiorizedSize(
    DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(1), DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(2),
    DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(3), DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(4),
    DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(5), DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(6),
    DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(7), DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(8)) {
  if constexpr (VECTORIZED_AXIS_SIZE >= 1) {
    AscendC::GetUintDivMagicAndShift(VECTORIZED_AXIS_SIZE_M(1), VECTORIZED_AXIS_SIZE_SHIFT(1), VECTORIZED_AXIS_SIZE(1));
  }
  if constexpr (VECTORIZED_AXIS_SIZE >= 2) {
    AscendC::GetUintDivMagicAndShift(VECTORIZED_AXIS_SIZE_M(2), VECTORIZED_AXIS_SIZE_SHIFT(2), VECTORIZED_AXIS_SIZE(2));
  }
  if constexpr (VECTORIZED_AXIS_SIZE >= 3) {
    AscendC::GetUintDivMagicAndShift(VECTORIZED_AXIS_SIZE_M(3), VECTORIZED_AXIS_SIZE_SHIFT(3), VECTORIZED_AXIS_SIZE(3));
  }
  if constexpr (VECTORIZED_AXIS_SIZE >= 4) {
    AscendC::GetUintDivMagicAndShift(VECTORIZED_AXIS_SIZE_M(4), VECTORIZED_AXIS_SIZE_SHIFT(4), VECTORIZED_AXIS_SIZE(4));
  }
  if constexpr (VECTORIZED_AXIS_SIZE >= 5) {
    AscendC::GetUintDivMagicAndShift(VECTORIZED_AXIS_SIZE_M(5), VECTORIZED_AXIS_SIZE_SHIFT(5), VECTORIZED_AXIS_SIZE(5));
  }
  if constexpr (VECTORIZED_AXIS_SIZE >= 6) {
    AscendC::GetUintDivMagicAndShift(VECTORIZED_AXIS_SIZE_M(6), VECTORIZED_AXIS_SIZE_SHIFT(6), VECTORIZED_AXIS_SIZE(6));
  }
  if constexpr (VECTORIZED_AXIS_SIZE >= 7) {
    AscendC::GetUintDivMagicAndShift(VECTORIZED_AXIS_SIZE_M(7), VECTORIZED_AXIS_SIZE_SHIFT(7), VECTORIZED_AXIS_SIZE(7));
  }
  if constexpr (VECTORIZED_AXIS_SIZE >= 8) {
    AscendC::GetUintDivMagicAndShift(VECTORIZED_AXIS_SIZE_M(8), VECTORIZED_AXIS_SIZE_SHIFT(8), VECTORIZED_AXIS_SIZE(8));
  }
}

template <typename INDEX_SIZE_T, uint32_t VECTORIZED_AXIS_SIZE>
inline __simt_callee__ __aicore__ void ComputeDestPositionAndSrcPosition(
    INDEX_SIZE_T &v_offset, INDEX_SIZE_T &src_positon, INDEX_SIZE_T &dest_position,
    const uint32_t &vectorized_axis_size_m, const uint32_t &vectorized_axis_size_shift,
    const uint32_t &vectorized_axis_size, const uint32_t &vectorized_axis_stride,
    const uint32_t &y_vectorized_axis_size) {
  INDEX_SIZE_T tmp1 = Simt::UintDiv(v_offset, static_cast<INDEX_SIZE_T>(vectorized_axis_size_m),
                                    static_cast<INDEX_SIZE_T>(vectorized_axis_size_shift));
  INDEX_SIZE_T tmp11 = (v_offset - tmp1 * vectorized_axis_size);
  dest_position += tmp11 * vectorized_axis_stride;
  src_positon += tmp11 * y_vectorized_axis_size;
  v_offset = tmp1;
}

template <typename INDEX_SIZE_T, uint32_t VECTORIZED_AXIS_SIZE>
inline __simt_callee__ __aicore__ void ComputeDestPositionAndSrcPosition(
    INDEX_SIZE_T v_offset, INDEX_SIZE_T &src_position, INDEX_SIZE_T &dest_position,
    DECLARE_VECTORIZED_AXIS_PARAMS_SIMT_CONST_REF(1), DECLARE_VECTORIZED_AXIS_PARAMS_SIMT_CONST_REF(2),
    DECLARE_VECTORIZED_AXIS_PARAMS_SIMT_CONST_REF(3), DECLARE_VECTORIZED_AXIS_PARAMS_SIMT_CONST_REF(4),
    DECLARE_VECTORIZED_AXIS_PARAMS_SIMT_CONST_REF(5), DECLARE_VECTORIZED_AXIS_PARAMS_SIMT_CONST_REF(6),
    DECLARE_VECTORIZED_AXIS_PARAMS_SIMT_CONST_REF(7), DECLARE_VECTORIZED_AXIS_PARAMS_SIMT_CONST_REF(8)) {
  if constexpr (VECTORIZED_AXIS_SIZE >= 1) {
    ComputeDestPositionAndSrcPosition<INDEX_SIZE_T, VECTORIZED_AXIS_SIZE>(v_offset, src_position, dest_position,
                                                                          PASS_VECTORIZED_AXIS_PARAMS_SIMT(1));
  }
  if constexpr (VECTORIZED_AXIS_SIZE >= 2) {
    ComputeDestPositionAndSrcPosition<INDEX_SIZE_T, VECTORIZED_AXIS_SIZE>(v_offset, src_position, dest_position,
                                                                          PASS_VECTORIZED_AXIS_PARAMS_SIMT(2));
  }
  if constexpr (VECTORIZED_AXIS_SIZE >= 3) {
    ComputeDestPositionAndSrcPosition<INDEX_SIZE_T, VECTORIZED_AXIS_SIZE>(v_offset, src_position, dest_position,
                                                                          PASS_VECTORIZED_AXIS_PARAMS_SIMT(3));
  }
  if constexpr (VECTORIZED_AXIS_SIZE >= 4) {
    ComputeDestPositionAndSrcPosition<INDEX_SIZE_T, VECTORIZED_AXIS_SIZE>(v_offset, src_position, dest_position,
                                                                          PASS_VECTORIZED_AXIS_PARAMS_SIMT(4));
  }
  if constexpr (VECTORIZED_AXIS_SIZE >= 5) {
    ComputeDestPositionAndSrcPosition<INDEX_SIZE_T, VECTORIZED_AXIS_SIZE>(v_offset, src_position, dest_position,
                                                                          PASS_VECTORIZED_AXIS_PARAMS_SIMT(5));
  }
  if constexpr (VECTORIZED_AXIS_SIZE >= 6) {
    ComputeDestPositionAndSrcPosition<INDEX_SIZE_T, VECTORIZED_AXIS_SIZE>(v_offset, src_position, dest_position,
                                                                          PASS_VECTORIZED_AXIS_PARAMS_SIMT(6));
  }
  if constexpr (VECTORIZED_AXIS_SIZE >= 7) {
    ComputeDestPositionAndSrcPosition<INDEX_SIZE_T, VECTORIZED_AXIS_SIZE>(v_offset, src_position, dest_position,
                                                                          PASS_VECTORIZED_AXIS_PARAMS_SIMT(7));
  }
  if constexpr (VECTORIZED_AXIS_SIZE >= 8) {
    ComputeDestPositionAndSrcPosition<INDEX_SIZE_T, VECTORIZED_AXIS_SIZE>(v_offset, src_position, dest_position,
                                                                          PASS_VECTORIZED_AXIS_PARAMS_SIMT(8));
  }
}

template <typename T1, typename T2, typename INDEX_SIZE_T, int32_t SRC_NUMBER>
inline __simt_callee__ __aicore__ void CommonCalOffset(uint32_t &dst_p, INDEX_SIZE_T &param_offset) {
  if constexpr (SRC_NUMBER == 2) {
    param_offset = param_offset >> 1;
    dst_p = dst_p >> 1;
  } else if constexpr (SRC_NUMBER == 4) {
    param_offset = param_offset >> 2;
    dst_p = dst_p >> 2;
  } else if constexpr (SRC_NUMBER == 8) {
    param_offset = param_offset >> 3;
    dst_p = dst_p >> 3;
  }
}

template <typename T1, typename T2, typename INDEX_SIZE_T, int32_t SRC_NUMBER, bool negative_index_support>
inline __simt_callee__ __aicore__ void CopyInCase1(__ubuf__ T1 *dst, __gm__ T1 *x1_gm, __gm__ T2 *x2_gm, uint32_t dst_p,
                                                   const INDEX_SIZE_T &x1_gather_dim_size,
                                                   const INDEX_SIZE_T &y_offset) {
  T2 param_offset = x2_gm[y_offset];
  if constexpr (negative_index_support) {
    if (unlikely(param_offset < 0)) {
      param_offset += x1_gather_dim_size;
    }
  }
  dst[dst_p] = param_offset < 0 || param_offset >= x1_gather_dim_size ? 0 : x1_gm[param_offset];
}

template <typename T1, typename T2, typename INDEX_SIZE_T, int32_t SRC_NUMBER, bool negative_index_support>
inline __simt_callee__ __aicore__ void CopyInCase2(__ubuf__ T1 *dst, __gm__ T1 *x1_gm, __gm__ T2 *x2_gm, uint32_t dst_p,
                                                   const INDEX_SIZE_T &x1_gather_dim_size, const INDEX_SIZE_T &y_offset,
                                                   const INDEX_SIZE_T &x1_gather_dim_stride_m,
                                                   const INDEX_SIZE_T &x1_gather_dim_stride_shift,
                                                   const INDEX_SIZE_T &x1_gather_dim_stride) {
  INDEX_SIZE_T index_idx = Simt::UintDiv(y_offset, x1_gather_dim_stride_m, x1_gather_dim_stride_shift);
  T2 index_value = x2_gm[index_idx];
  if constexpr (negative_index_support) {
    if (unlikely(index_value < 0)) {
      index_value += x1_gather_dim_size;
    }
  }
  INDEX_SIZE_T param_offset = index_value * x1_gather_dim_stride + (y_offset - index_idx * x1_gather_dim_stride);
  CommonCalOffset<T1, T2, INDEX_SIZE_T, SRC_NUMBER>(dst_p, param_offset);
  dst[dst_p] = index_value < 0 || index_value >= x1_gather_dim_size ? 0 : x1_gm[param_offset];
}

template <typename T1, typename T2, typename INDEX_SIZE_T, int32_t SRC_NUMBER, bool negative_index_support>
inline __simt_callee__ __aicore__ void CopyInCase3(__ubuf__ T1 *dst, __gm__ T1 *x1_gm, __gm__ T2 *x2_gm, uint32_t dst_p,
                                                   const INDEX_SIZE_T &x1_gather_dim_size, const INDEX_SIZE_T &y_offset,
                                                   const INDEX_SIZE_T &x2_tensor_size_m,
                                                   const INDEX_SIZE_T &x2_tensor_size_shift,
                                                   const INDEX_SIZE_T &x2_tensor_size) {
  INDEX_SIZE_T tmp = Simt::UintDiv(y_offset, x2_tensor_size_m, x2_tensor_size_shift);
  INDEX_SIZE_T index_idx = y_offset - tmp * x2_tensor_size;
  T2 index_value = x2_gm[index_idx];
  if constexpr (negative_index_support) {
    if (unlikely(index_value < 0)) {
      index_value += x1_gather_dim_size;
    }
  }
  INDEX_SIZE_T param_offset = tmp * x1_gather_dim_size + index_value;
  dst[dst_p] = index_value < 0 || index_value >= x1_gather_dim_size ? 0 : x1_gm[param_offset];
}

template <typename T1, typename T2, typename INDEX_SIZE_T, int32_t SRC_NUMBER, bool negative_index_support>
inline __simt_callee__ __aicore__ void CopyInCase4(
    __ubuf__ T1 *dst, __gm__ T1 *x1_gm, __gm__ T2 *x2_gm, uint32_t dst_p, const INDEX_SIZE_T &x1_gather_dim_size,
    const INDEX_SIZE_T &y_offset, const INDEX_SIZE_T &x2_tensor_size_m, const INDEX_SIZE_T &x2_tensor_size_shift,
    const INDEX_SIZE_T &x2_tensor_size, const INDEX_SIZE_T &x1_gather_dim_stride_m,
    const INDEX_SIZE_T &x1_gather_dim_stride_shift, const INDEX_SIZE_T &x1_gather_dim_stride) {
  INDEX_SIZE_T tmp = Simt::UintDiv(y_offset, x1_gather_dim_stride_m, x1_gather_dim_stride_shift);
  INDEX_SIZE_T tmp1 = Simt::UintDiv(tmp, x2_tensor_size_m, x2_tensor_size_shift);
  INDEX_SIZE_T index_idx = tmp - tmp1 * x2_tensor_size;
  T2 index_value = x2_gm[index_idx];
  if constexpr (negative_index_support) {
    if (unlikely(index_value < 0)) {
      index_value += x1_gather_dim_size;
    }
  }
  INDEX_SIZE_T param_offset = tmp1 * x1_gather_dim_size * x1_gather_dim_stride + index_value * x1_gather_dim_stride +
                              (y_offset - tmp * x1_gather_dim_stride);
  CommonCalOffset<T1, T2, INDEX_SIZE_T, SRC_NUMBER>(dst_p, param_offset);
  dst[dst_p] = index_value < 0 || index_value >= x1_gather_dim_size ? 0 : x1_gm[param_offset];
}

template <typename T1, typename T2, typename INDEX_SIZE_T, int32_t CASE, int32_t VECTORIZED_AXIS_SIZE,
          uint32_t SRC_NUMBER, bool negative_index_support>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUMBER) inline void GatherSimt(
    __ubuf__ T1 *dst, __gm__ T1 *x1_gm, __gm__ T2 *x2_gm, uint32_t ub_actual_size, INDEX_SIZE_T offset,
    INDEX_SIZE_T x1_gather_dim_size, INDEX_SIZE_T x2_tensor_size_m, INDEX_SIZE_T x2_tensor_size_shift,
    INDEX_SIZE_T x2_tensor_size, INDEX_SIZE_T x1_gather_dim_stride_m, INDEX_SIZE_T x1_gather_dim_stride_shift,
    INDEX_SIZE_T x1_gather_dim_stride, DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(1), DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(2),
    DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(3), DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(4),
    DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(5), DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(6),
    DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(7), DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(8)) {
  for (INDEX_SIZE_T i = static_cast<INDEX_SIZE_T>(Simt::GetThreadIdx()) * SRC_NUMBER; i < ub_actual_size;
       i += static_cast<INDEX_SIZE_T>(Simt::GetThreadNum<0>()) * SRC_NUMBER) {
    auto y_offset = offset;
    uint32_t dst_p = 0;
    ComputeDestPositionAndSrcPosition<INDEX_SIZE_T, VECTORIZED_AXIS_SIZE>(
        i, y_offset, dst_p, PASS_VECTORIZED_AXIS_PARAMS_SIMT(1), PASS_VECTORIZED_AXIS_PARAMS_SIMT(2),
        PASS_VECTORIZED_AXIS_PARAMS_SIMT(3), PASS_VECTORIZED_AXIS_PARAMS_SIMT(4), PASS_VECTORIZED_AXIS_PARAMS_SIMT(5),
        PASS_VECTORIZED_AXIS_PARAMS_SIMT(6), PASS_VECTORIZED_AXIS_PARAMS_SIMT(7), PASS_VECTORIZED_AXIS_PARAMS_SIMT(8));
    if constexpr (CASE == CASE1) {
      CopyInCase1<T1, T2, INDEX_SIZE_T, SRC_NUMBER, negative_index_support>(dst, x1_gm, x2_gm, dst_p,
                                                                            x1_gather_dim_size, y_offset);
    }
    if constexpr (CASE == CASE2) {
      CopyInCase2<T1, T2, INDEX_SIZE_T, SRC_NUMBER, negative_index_support>(
          dst, x1_gm, x2_gm, dst_p, x1_gather_dim_size, y_offset, x1_gather_dim_stride_m, x1_gather_dim_stride_shift,
          x1_gather_dim_stride);
    }
    if constexpr (CASE == CASE3) {
      CopyInCase3<T1, T2, INDEX_SIZE_T, SRC_NUMBER, negative_index_support>(
          dst, x1_gm, x2_gm, dst_p, x1_gather_dim_size, y_offset, x2_tensor_size_m, x2_tensor_size_shift,
          x2_tensor_size);
    }
    if constexpr (CASE == CASE4) {
      CopyInCase4<T1, T2, INDEX_SIZE_T, SRC_NUMBER, negative_index_support>(
          dst, x1_gm, x2_gm, dst_p, x1_gather_dim_size, y_offset, x2_tensor_size_m, x2_tensor_size_shift,
          x2_tensor_size, x1_gather_dim_stride_m, x1_gather_dim_stride_shift, x1_gather_dim_stride);
    }
  }
}

template <typename T1, typename T2, typename INDEX_SIZE_T, int32_t SRC_NUMBER, bool negative_index_support>
inline __simt_callee__ __aicore__ void CopyInCase1ParamFullLoad(__ubuf__ T1 *dst, __ubuf__ T1 *x1_gm, __gm__ T2 *x2_gm,
                                                                uint32_t dst_p, const INDEX_SIZE_T &x1_gather_dim_size,
                                                                const INDEX_SIZE_T &y_offset) {
  T2 param_offset = x2_gm[y_offset];
  if constexpr (negative_index_support) {
    if (unlikely(param_offset < 0)) {
      param_offset += x1_gather_dim_size;
    }
  }
  dst[dst_p] = param_offset < 0 || param_offset >= x1_gather_dim_size ? 0 : x1_gm[param_offset];
}

template <typename T1, typename T2, typename INDEX_SIZE_T, int32_t SRC_NUMBER, bool negative_index_support>
inline __simt_callee__ __aicore__ void CopyInCase2ParamFullLoad(__ubuf__ T1 *dst, __ubuf__ T1 *x1_gm, __gm__ T2 *x2_gm,
                                                                uint32_t dst_p, const INDEX_SIZE_T &x1_gather_dim_size,
                                                                const INDEX_SIZE_T &y_offset,
                                                                const INDEX_SIZE_T &x1_gather_dim_stride_m,
                                                                const INDEX_SIZE_T &x1_gather_dim_stride_shift,
                                                                const INDEX_SIZE_T &x1_gather_dim_stride) {
  INDEX_SIZE_T index_idx = Simt::UintDiv(y_offset, x1_gather_dim_stride_m, x1_gather_dim_stride_shift);
  T2 index_value = x2_gm[index_idx];
  if constexpr (negative_index_support) {
    if (unlikely(index_value < 0)) {
      index_value += x1_gather_dim_size;
    }
  }
  INDEX_SIZE_T param_offset = index_value * x1_gather_dim_stride + (y_offset - index_idx * x1_gather_dim_stride);
  CommonCalOffset<T1, T2, INDEX_SIZE_T, SRC_NUMBER>(dst_p, param_offset);
  dst[dst_p] = index_value < 0 || index_value >= x1_gather_dim_size ? 0 : x1_gm[param_offset];
}

template <typename T1, typename T2, typename INDEX_SIZE_T, int32_t SRC_NUMBER, bool negative_index_support>
inline __simt_callee__ __aicore__ void CopyInCase3ParamFullLoad(__ubuf__ T1 *dst, __ubuf__ T1 *x1_gm, __gm__ T2 *x2_gm,
                                                                uint32_t dst_p, const INDEX_SIZE_T &x1_gather_dim_size,
                                                                const INDEX_SIZE_T &y_offset,
                                                                const INDEX_SIZE_T &x2_tensor_size_m,
                                                                const INDEX_SIZE_T &x2_tensor_size_shift,
                                                                const INDEX_SIZE_T &x2_tensor_size) {
  INDEX_SIZE_T tmp = Simt::UintDiv(y_offset, x2_tensor_size_m, x2_tensor_size_shift);
  INDEX_SIZE_T index_idx = y_offset - tmp * x2_tensor_size;
  T2 index_value = x2_gm[index_idx];
  if constexpr (negative_index_support) {
    if (unlikely(index_value < 0)) {
      index_value += x1_gather_dim_size;
    }
  }
  INDEX_SIZE_T param_offset = tmp * x1_gather_dim_size + index_value;
  dst[dst_p] = index_value < 0 || index_value >= x1_gather_dim_size ? 0 : x1_gm[param_offset];
}

template <typename T1, typename T2, typename INDEX_SIZE_T, int32_t SRC_NUMBER, bool negative_index_support>
inline __simt_callee__ __aicore__ void CopyInCase4ParamFullLoad(
    __ubuf__ T1 *dst, __ubuf__ T1 *x1_gm, __gm__ T2 *x2_gm, uint32_t dst_p, const INDEX_SIZE_T &x1_gather_dim_size,
    const INDEX_SIZE_T &y_offset, const INDEX_SIZE_T &x2_tensor_size_m, const INDEX_SIZE_T &x2_tensor_size_shift,
    const INDEX_SIZE_T &x2_tensor_size, const INDEX_SIZE_T &x1_gather_dim_stride_m,
    const INDEX_SIZE_T &x1_gather_dim_stride_shift, const INDEX_SIZE_T &x1_gather_dim_stride) {
  INDEX_SIZE_T tmp = Simt::UintDiv(y_offset, x1_gather_dim_stride_m, x1_gather_dim_stride_shift);
  INDEX_SIZE_T tmp1 = Simt::UintDiv(tmp, x2_tensor_size_m, x2_tensor_size_shift);
  INDEX_SIZE_T index_idx = tmp - tmp1 * x2_tensor_size;
  T2 index_value = x2_gm[index_idx];
  if constexpr (negative_index_support) {
    if (unlikely(index_value < 0)) {
      index_value += x1_gather_dim_size;
    }
  }
  INDEX_SIZE_T param_offset = tmp1 * x1_gather_dim_size * x1_gather_dim_stride + index_value * x1_gather_dim_stride +
                              (y_offset - tmp * x1_gather_dim_stride);
  CommonCalOffset<T1, T2, INDEX_SIZE_T, SRC_NUMBER>(dst_p, param_offset);
  dst[dst_p] = index_value < 0 || index_value >= x1_gather_dim_size ? 0 : x1_gm[param_offset];
}

template <typename T1, typename T2, typename INDEX_SIZE_T, int32_t CASE, int32_t VECTORIZED_AXIS_SIZE,
          uint32_t SRC_NUMBER, bool negative_index_support>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUMBER) inline void GatherSimtParamFullLoad(
    __ubuf__ T1 *dst, __ubuf__ T1 *x1_gm, __gm__ T2 *x2_gm, uint32_t ub_actual_size, INDEX_SIZE_T offset,
    INDEX_SIZE_T x1_gather_dim_size, INDEX_SIZE_T x2_tensor_size_m, INDEX_SIZE_T x2_tensor_size_shift,
    INDEX_SIZE_T x2_tensor_size, INDEX_SIZE_T x1_gather_dim_stride_m, INDEX_SIZE_T x1_gather_dim_stride_shift,
    INDEX_SIZE_T x1_gather_dim_stride, DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(1), DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(2),
    DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(3), DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(4),
    DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(5), DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(6),
    DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(7), DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(8)) {
  for (INDEX_SIZE_T i = static_cast<INDEX_SIZE_T>(Simt::GetThreadIdx()) * SRC_NUMBER; i < ub_actual_size;
       i += static_cast<INDEX_SIZE_T>(Simt::GetThreadNum<0>()) * SRC_NUMBER) {
    auto y_offset = offset;
    uint32_t dst_p = 0;
    ComputeDestPositionAndSrcPosition<INDEX_SIZE_T, VECTORIZED_AXIS_SIZE>(
        i, y_offset, dst_p, PASS_VECTORIZED_AXIS_PARAMS_SIMT(1), PASS_VECTORIZED_AXIS_PARAMS_SIMT(2),
        PASS_VECTORIZED_AXIS_PARAMS_SIMT(3), PASS_VECTORIZED_AXIS_PARAMS_SIMT(4), PASS_VECTORIZED_AXIS_PARAMS_SIMT(5),
        PASS_VECTORIZED_AXIS_PARAMS_SIMT(6), PASS_VECTORIZED_AXIS_PARAMS_SIMT(7), PASS_VECTORIZED_AXIS_PARAMS_SIMT(8));
    if constexpr (CASE == CASE1) {
      CopyInCase1ParamFullLoad<T1, T2, INDEX_SIZE_T, SRC_NUMBER, negative_index_support>(dst, x1_gm, x2_gm, dst_p,
                                                                                         x1_gather_dim_size, y_offset);
    }
    if constexpr (CASE == CASE2) {
      CopyInCase2ParamFullLoad<T1, T2, INDEX_SIZE_T, SRC_NUMBER, negative_index_support>(
          dst, x1_gm, x2_gm, dst_p, x1_gather_dim_size, y_offset, x1_gather_dim_stride_m, x1_gather_dim_stride_shift,
          x1_gather_dim_stride);
    }
    if constexpr (CASE == CASE3) {
      CopyInCase3ParamFullLoad<T1, T2, INDEX_SIZE_T, SRC_NUMBER, negative_index_support>(
          dst, x1_gm, x2_gm, dst_p, x1_gather_dim_size, y_offset, x2_tensor_size_m, x2_tensor_size_shift,
          x2_tensor_size);
    }
    if constexpr (CASE == CASE4) {
      CopyInCase4ParamFullLoad<T1, T2, INDEX_SIZE_T, SRC_NUMBER, negative_index_support>(
          dst, x1_gm, x2_gm, dst_p, x1_gather_dim_size, y_offset, x2_tensor_size_m, x2_tensor_size_shift,
          x2_tensor_size, x1_gather_dim_stride_m, x1_gather_dim_stride_shift, x1_gather_dim_stride);
    }
  }
}

/**************************************************************************************** 模板1 通用SIMT模板
 * *********************************************************************/
template <typename T1, typename T2, int32_t CASE, int32_t VECTORIZED_AXIS_SIZE, bool negative_index_support>
inline __aicore__ void GatherExtendDefault(
    AscendC::LocalTensor<T1> &dst, const AscendC::GlobalTensor<T1> &src1, const AscendC::GlobalTensor<T2> &src2,
    uint32_t ub_actual_size, uint64_t offset, uint32_t x2_tensor_size, uint32_t x1_gather_dim_size,
    uint32_t x1_gather_dim_stride, DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(1), DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(2),
    DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(3), DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(4),
    DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(5), DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(6),
    DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(7), DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(8)) {
  __gm__ T1 *x1_gm = (__gm__ T1 *)(src1.GetPhyAddr());
  __gm__ T2 *x2_gm = (__gm__ T2 *)(src2.GetPhyAddr());
  __ubuf__ T1 *dst_p = (__ubuf__ T1 *)(dst.GetPhyAddr());
  uint32_t x1_gather_dim_stride_m{0}, x1_gather_dim_stride_shift{0}, x2_tensor_size_m{0}, x2_tensor_size_shift{0};
  if constexpr (CASE == CASE2) {
    AscendC::GetUintDivMagicAndShift(x1_gather_dim_stride_m, x1_gather_dim_stride_shift, x1_gather_dim_stride);
  }
  if constexpr (CASE == CASE3) {
    AscendC::GetUintDivMagicAndShift(x2_tensor_size_m, x2_tensor_size_shift, x2_tensor_size);
  }
  if constexpr (CASE == CASE4) {
    AscendC::GetUintDivMagicAndShift(x1_gather_dim_stride_m, x1_gather_dim_stride_shift, x1_gather_dim_stride);
    AscendC::GetUintDivMagicAndShift(x2_tensor_size_m, x2_tensor_size_shift, x2_tensor_size);
  }
  DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_VARIABLES(1);
  DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_VARIABLES(2);
  DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_VARIABLES(3);
  DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_VARIABLES(4);
  DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_VARIABLES(5);
  DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_VARIABLES(6);
  DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_VARIABLES(7);
  DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_VARIABLES(8);
  GetMagicShiftForVectiorizedSize<VECTORIZED_AXIS_SIZE>(
      PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(1), PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(2),
      PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(3), PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(4),
      PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(5), PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(6),
      PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(7), PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(8));
  int32_t event_id_v_to_mte3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
  AscendC::Simt::VF_CALL<GatherSimt<T1, T2, uint32_t, CASE, VECTORIZED_AXIS_SIZE, 1, negative_index_support>>(
      AscendC::Simt::Dim3(THREAD_NUMBER), dst_p, x1_gm, x2_gm, ub_actual_size, static_cast<uint32_t>(offset),
      static_cast<uint32_t>(x1_gather_dim_size), static_cast<uint32_t>(x2_tensor_size_m),
      static_cast<uint32_t>(x2_tensor_size_shift), static_cast<uint32_t>(x2_tensor_size),
      static_cast<uint32_t>(x1_gather_dim_stride_m), static_cast<uint32_t>(x1_gather_dim_stride_shift),
      static_cast<uint32_t>(x1_gather_dim_stride), PASS_VECTORIZED_AXIS_PARAMS_SIMT(1),
      PASS_VECTORIZED_AXIS_PARAMS_SIMT(2), PASS_VECTORIZED_AXIS_PARAMS_SIMT(3), PASS_VECTORIZED_AXIS_PARAMS_SIMT(4),
      PASS_VECTORIZED_AXIS_PARAMS_SIMT(5), PASS_VECTORIZED_AXIS_PARAMS_SIMT(6), PASS_VECTORIZED_AXIS_PARAMS_SIMT(7),
      PASS_VECTORIZED_AXIS_PARAMS_SIMT(8));
  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event_id_v_to_mte3);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event_id_v_to_mte3);
}
/***********************************************************************************************************************************************************************************/

/**************************************************************************************** 模板2 长连续数据搬运模板
 * *********************************************************************/
template <typename T1>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUMBER) inline void GatherSimtContinuous(
    __ubuf__ T1 *dst, __gm__ T1 *x1_gm, uint32_t ub_actual_size, uint32_t dst_p, uint64_t param_offset, bool is_out) {
  for (uint32_t i = Simt::GetThreadIdx(); i < ub_actual_size; i += Simt::GetThreadNum<0>()) {
    dst[i + dst_p] = is_out ? 0 : x1_gm[i + param_offset];
  }
}

template <typename T1>
inline __aicore__ void DataCopySimdSimt(AscendC::LocalTensor<T1> &dst, const AscendC::GlobalTensor<T1> &src1,
                                        uint32_t dst_p, uint64_t src_p, uint64_t length, bool is_out) {
  if (length <= 0) {
    return;
  }
  __gm__ T1 *x1_gm = (__gm__ T1 *)(src1.GetPhyAddr());
  __ubuf__ T1 *dst_p1 = (__ubuf__ T1 *)(dst.GetPhyAddr());
  uint32_t padding = 0, current_addr = 0, aligned_addr = 0;
  if ((dst_p * sizeof(T1)) % 32 != 0) {
    current_addr = dst_p * sizeof(T1);
    padding = (32 - (current_addr % 32)) % 32;
    uint32_t padding_elements = padding / sizeof(T1);
    uint32_t padding_copy_length = length <= padding_elements ? static_cast<uint32_t>(length) : padding_elements;
    int32_t event_id_mte2_to_v = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id_mte2_to_v);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id_mte2_to_v);
    AscendC::Simt::VF_CALL<GatherSimtContinuous<T1>>(AscendC::Simt::Dim3(128), dst_p1, x1_gm, padding_copy_length,
                                                     dst_p, src_p, is_out);
    if (length <= padding_elements) {
      return;
    }
    dst_p += padding_elements;
    src_p += padding_elements;
    length -= padding_elements;
    if (unlikely(is_out)) {
      T1 value{};
      AscendC::Duplicate(dst[dst_p], value, length);
      return;
    }
    GatherDataCopyPadExtend(dst[dst_p], src1[src_p], 1, length, 0, 0);
  } else {
    if (unlikely(is_out)) {
      T1 value{};
      AscendC::Duplicate(dst[dst_p], value, length);
    } else {
      GatherDataCopyPadExtend(dst[dst_p], src1[src_p], 1, length, 0, 0);
    }
  }
}

template <typename T1, typename T2, int32_t CASE, int32_t VECTORIZED_AXIS_SIZE, bool negative_index_support>
inline __aicore__ void GatherExtendDataCopy(
    AscendC::LocalTensor<T1> &dst, const AscendC::GlobalTensor<T1> &src1, const AscendC::GlobalTensor<T2> &src2,
    uint32_t ub_actual_size, uint64_t offset, uint64_t x2_tensor_size, uint64_t x1_gather_dim_size,
    uint64_t x1_gather_dim_stride, DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(1), DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(2),
    DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(3), DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(4),
    DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(5), DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(6),
    DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(7), DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(8)) {
  uint64_t index_idx_from = (offset / x1_gather_dim_stride) % x2_tensor_size;
  uint64_t index_idx_to = ((offset + ub_actual_size - 1) / x1_gather_dim_stride) % x2_tensor_size;
  uint64_t pre_gather_idx_from = (offset / x1_gather_dim_stride) / x2_tensor_size;
  uint64_t pre_gather_idx_to = ((offset + ub_actual_size - 1) / x1_gather_dim_stride) / x2_tensor_size;
  if (pre_gather_idx_from == pre_gather_idx_to && index_idx_from < index_idx_to) {
    uint64_t back_gather_idx_first = offset % x1_gather_dim_stride;
    uint64_t back_gather_idx_last = (offset + ub_actual_size - 1) % x1_gather_dim_stride;
    uint64_t dst_p = 0;
    T2 index_value = src2.GetValue(index_idx_from);
    if constexpr (negative_index_support) {
      index_value = index_value < 0 ? index_value + x1_gather_dim_size : index_value;
    }
    bool is_out = index_value < 0 || index_value >= x1_gather_dim_size;
    uint64_t param_offset = pre_gather_idx_from * x1_gather_dim_size * x1_gather_dim_stride +
                            index_value * x1_gather_dim_stride + back_gather_idx_first;
    DataCopySimdSimt<T1>(dst, src1, 0, param_offset, x1_gather_dim_stride - back_gather_idx_first, is_out);
    dst_p += (x1_gather_dim_stride - back_gather_idx_first);
    for (int i = index_idx_from + 1; i < index_idx_to; i++) {
      index_value = src2.GetValue(i);
      if constexpr (negative_index_support) {
        index_value = index_value < 0 ? index_value + x1_gather_dim_size : index_value;
      }
      is_out = index_value < 0 || index_value >= x1_gather_dim_size;
      param_offset =
          pre_gather_idx_from * x1_gather_dim_size * x1_gather_dim_stride + index_value * x1_gather_dim_stride;
      DataCopySimdSimt<T1>(dst, src1, dst_p, param_offset, x1_gather_dim_stride, is_out);
      dst_p += x1_gather_dim_stride;
    }
    index_value = src2.GetValue(index_idx_to);
    if constexpr (negative_index_support) {
      index_value = index_value < 0 ? index_value + x1_gather_dim_size : index_value;
    }
    is_out = index_value < 0 || index_value >= x1_gather_dim_size;
    param_offset = pre_gather_idx_from * x1_gather_dim_size * x1_gather_dim_stride + index_value * x1_gather_dim_stride;
    DataCopySimdSimt<T1>(dst, src1, dst_p, param_offset, back_gather_idx_last + 1, is_out);
  } else if (pre_gather_idx_from == pre_gather_idx_to && index_idx_from == index_idx_to) {
    T2 index_value = src2.GetValue(index_idx_from);
    if constexpr (negative_index_support) {
      index_value = index_value < 0 ? index_value + x1_gather_dim_size : index_value;
    }
    bool is_out = index_value < 0 || index_value >= x1_gather_dim_size;
    uint64_t back_gather_idx_first = offset % x1_gather_dim_stride;
    uint64_t param_offset = pre_gather_idx_from * x1_gather_dim_size * x1_gather_dim_stride +
                            index_value * x1_gather_dim_stride + back_gather_idx_first;
    DataCopySimdSimt<T1>(dst, src1, 0, param_offset, ub_actual_size, is_out);
  } else {
    GatherExtendDefault<T1, T2, CASE, VECTORIZED_AXIS_SIZE, negative_index_support>(
        dst, src1, src2, ub_actual_size, offset, x2_tensor_size, x1_gather_dim_size, x1_gather_dim_stride,
        PASS_VECTORIZED_AXIS_PARAMS_SIMD(1), PASS_VECTORIZED_AXIS_PARAMS_SIMD(2), PASS_VECTORIZED_AXIS_PARAMS_SIMD(3),
        PASS_VECTORIZED_AXIS_PARAMS_SIMD(4), PASS_VECTORIZED_AXIS_PARAMS_SIMD(5), PASS_VECTORIZED_AXIS_PARAMS_SIMD(6),
        PASS_VECTORIZED_AXIS_PARAMS_SIMD(7), PASS_VECTORIZED_AXIS_PARAMS_SIMD(8));
  }
}
/***********************************************************************************************************************************************************************************/

/**************************************************************************************** 模板3 微指令处理单轴场景
 * *********************************************************************/
template <typename T1, typename T2>
__aicore__ inline void ConvertNegIndices(const LocalTensor<int32_t> &indicesLocal, int32_t num, int32_t dim_size) {
  __local_mem__ int32_t *indice_addr = (__local_mem__ int32_t *)indicesLocal.GetPhyAddr();
  constexpr int16_t vf_len = AscendC::VECTOR_REG_WIDTH / sizeof(int32_t);
  uint16_t vf_loop_num = (num + vf_len - 1) / vf_len;
  __VEC_SCOPE__ {
    MicroAPI::RegTensor<int32_t> indice;
    MicroAPI::RegTensor<int32_t> dst;
    MicroAPI::MaskReg lt_preg;
    uint32_t size = num;
    __local_mem__ int32_t *cur_indice_addr = indice_addr;
    for (uint16_t i = 0; i < vf_loop_num; i++) {
      MicroAPI::MaskReg preg = AscendC::MicroAPI::UpdateMask<int32_t>(size);
      MicroAPI::DataCopy(indice, cur_indice_addr);
      MicroAPI::CompareScalar<int32_t, CMPMODE::LT>(lt_preg, indice, 0, preg);
      MicroAPI::Adds(dst, indice, dim_size, lt_preg);
      MicroAPI::Copy<int32_t, MicroAPI::MaskMergeMode::MERGING>(indice, dst, lt_preg);
      MicroAPI::DataCopy(cur_indice_addr, indice, preg);
      cur_indice_addr += vf_len;
    }
  }
}

template <typename TARGET_T, typename ORG_T>
inline __aicore__ void LoadIndicesORG64(MicroAPI::RegTensor<TARGET_T> &vreg_indice, MicroAPI::MaskReg &gather_mask,
                                        __local_mem__ ORG_T *indice_addr, MicroAPI::MaskReg preg, ORG_T dim_size) {
  if constexpr (sizeof(TARGET_T) == sizeof(int32_t)) {
    MicroAPI::RegTensor<int64_t, MicroAPI::RegTraitNumTwo> tmp_reg;
    MicroAPI::DataCopy(tmp_reg, indice_addr);
    MicroAPI::MaskReg gt_preg;
    MicroAPI::MaskReg lt_preg;
    MicroAPI::CompareScalar<int64_t, CMPMODE::GE>(gt_preg, tmp_reg, 0, preg);
    MicroAPI::CompareScalar<int64_t, CMPMODE::LT>(lt_preg, tmp_reg, dim_size, preg);
    MicroAPI::MaskAnd(gather_mask, gt_preg, lt_preg, preg);
    MicroAPI::Pack((MicroAPI::RegTensor<uint32_t> &)vreg_indice, tmp_reg);
  } else {
    constexpr int16_t vf_len = AscendC::VECTOR_REG_WIDTH / sizeof(int32_t);
    MicroAPI::RegTensor<int64_t, MicroAPI::RegTraitNumTwo> tmp_reg0, tmp_reg1;
    MicroAPI::RegTensor<int32_t> tmp_B32_reg0, tmp_B32_reg1;
    MicroAPI::MaskReg low_preg, high_preg;
    MicroAPI::MaskInterleave<int16_t>(low_preg, high_preg, preg, preg);
    MicroAPI::DataCopy(tmp_reg0, indice_addr);
    MicroAPI::DataCopy(tmp_reg1, indice_addr + vf_len);
    MicroAPI::Pack((MicroAPI::RegTensor<uint32_t> &)tmp_B32_reg0, tmp_reg0);
    MicroAPI::Pack((MicroAPI::RegTensor<uint32_t> &)tmp_B32_reg1, tmp_reg1);
    MicroAPI::DeInterleave<int16_t>(
        (MicroAPI::RegTensor<int16_t> &)vreg_indice, (MicroAPI::RegTensor<int16_t> &)tmp_B32_reg0,
        (MicroAPI::RegTensor<int16_t> &)tmp_B32_reg0, (MicroAPI::RegTensor<int16_t> &)tmp_B32_reg1);
    MicroAPI::MaskReg gt_preg, lt_preg;
    MicroAPI::CompareScalar<int64_t, CMPMODE::GE>(gt_preg, tmp_reg0, 0, low_preg);
    MicroAPI::CompareScalar<int64_t, CMPMODE::LT>(lt_preg, tmp_reg0, dim_size, low_preg);
    MicroAPI::MaskAnd(low_preg, gt_preg, lt_preg, low_preg);
    MicroAPI::CompareScalar<int64_t, CMPMODE::GE>(gt_preg, tmp_reg1, 0, high_preg);
    MicroAPI::CompareScalar<int64_t, CMPMODE::LT>(lt_preg, tmp_reg1, dim_size, high_preg);
    MicroAPI::MaskAnd(high_preg, gt_preg, lt_preg, high_preg);
    MicroAPI::MaskDeInterleave<int16_t>(gather_mask, lt_preg, low_preg, high_preg);
  }
}
template <typename TARGET_T, typename ORG_T>
inline __aicore__ void LoadIndicesORG32(MicroAPI::RegTensor<TARGET_T> &vreg_indice, MicroAPI::MaskReg &gather_mask,
                                        __local_mem__ ORG_T *indice_addr, MicroAPI::MaskReg preg, ORG_T dim_size) {
  if constexpr (sizeof(TARGET_T) == sizeof(int32_t)) {
    MicroAPI::DataCopy((MicroAPI::RegTensor<int32_t> &)vreg_indice, indice_addr);
    MicroAPI::MaskReg gt_preg;
    MicroAPI::MaskReg lt_preg;
    MicroAPI::CompareScalar<int32_t, CMPMODE::GE>(gt_preg, (MicroAPI::RegTensor<int32_t> &)vreg_indice, 0, preg);
    MicroAPI::CompareScalar<int32_t, CMPMODE::LT>(lt_preg, (MicroAPI::RegTensor<int32_t> &)vreg_indice, dim_size, preg);
    MicroAPI::MaskAnd(gather_mask, gt_preg, lt_preg, preg);
  } else {
    constexpr int16_t vf_len = AscendC::VECTOR_REG_WIDTH / sizeof(int32_t);
    MicroAPI::RegTensor<int32_t> tmp_reg0, tmp_reg1;
    MicroAPI::RegTensor<int32_t> tmp_B32;
    MicroAPI::MaskReg low_preg, high_preg;
    MicroAPI::MaskInterleave<int16_t>(low_preg, high_preg, preg, preg);
    MicroAPI::DataCopy(tmp_reg0, indice_addr);
    MicroAPI::DataCopy(tmp_reg1, indice_addr + vf_len);

    MicroAPI::DeInterleave<int16_t>((MicroAPI::RegTensor<int16_t> &)vreg_indice,
                                    (MicroAPI::RegTensor<int16_t> &)tmp_B32, (MicroAPI::RegTensor<int16_t> &)tmp_reg0,
                                    (MicroAPI::RegTensor<int16_t> &)tmp_reg1);
    MicroAPI::MaskReg gt_preg, lt_preg;
    MicroAPI::CompareScalar<int32_t, CMPMODE::GE>(gt_preg, tmp_reg0, 0, low_preg);
    MicroAPI::CompareScalar<int32_t, CMPMODE::LT>(lt_preg, tmp_reg0, dim_size, low_preg);
    MicroAPI::MaskAnd(low_preg, gt_preg, lt_preg, low_preg);
    MicroAPI::CompareScalar<int32_t, CMPMODE::GE>(gt_preg, tmp_reg1, 0, high_preg);
    MicroAPI::CompareScalar<int32_t, CMPMODE::LT>(lt_preg, tmp_reg1, dim_size, high_preg);
    MicroAPI::MaskAnd(high_preg, gt_preg, lt_preg, high_preg);
    MicroAPI::MaskDeInterleave<int16_t>(gather_mask, lt_preg, low_preg, high_preg);
  }
}

template <typename TARGET_T, typename ORG_T>
inline __aicore__ void LoadIndices(MicroAPI::RegTensor<TARGET_T> &vreg_indice, MicroAPI::MaskReg &gather_mask,
                                   __local_mem__ ORG_T *indice_addr, MicroAPI::MaskReg preg, ORG_T dim_size) {
  if constexpr (sizeof(ORG_T) == sizeof(int64_t)) {
    LoadIndicesORG64(vreg_indice, gather_mask, indice_addr, preg, dim_size);
  } else {
    LoadIndicesORG32(vreg_indice, gather_mask, indice_addr, preg, dim_size);
  }
}

template <typename T>
struct IndexTypeGet {
  using type =
      typename std::conditional<sizeof(T) == sizeof(int8_t) || sizeof(T) == sizeof(int16_t), uint16_t, uint32_t>::type;
};

template <typename T1, typename T2>
inline __aicore__ void VRegGather(__local_mem__ T1 *y_addr, __local_mem__ T1 *x_addr,
                                  __local_mem__ int32_t *indice_addr, uint32_t num_per_loop, int32_t gather_dim_size,
                                  int16_t vf_len, uint16_t vf_loop_num) {
  using indiceType = typename IndexTypeGet<T1>::type;
  __VEC_SCOPE__ {
    using RegDstT = typename std::conditional<sizeof(T1) == sizeof(int64_t),
                                              AscendC::MicroAPI::RegTensor<T1, AscendC::MicroAPI::RegTraitNumTwo>,
                                              AscendC::MicroAPI::RegTensor<T1>>::type;
    RegDstT vd0;
    AscendC::MicroAPI::RegTensor<indiceType> vreg_indice;
    AscendC::MicroAPI::MaskReg gather_mask;
    uint32_t size = num_per_loop;
    __local_mem__ int32_t *cur_indice_addr = indice_addr;
    for (uint16_t i = 0; i < vf_loop_num; i++) {
      MicroAPI::MaskReg preg0 = AscendC::MicroAPI::UpdateMask<indiceType>(size);
      LoadIndices<indiceType, int32_t>(vreg_indice, gather_mask, cur_indice_addr, preg0, gather_dim_size);
      cur_indice_addr += vf_len;
      __local_mem__ T1 *cur_y_addr = y_addr + i * vf_len;
      __local_mem__ T1 *cur_x_addr = x_addr;
      if constexpr (sizeof(T1) == 1) {
        AscendC::MicroAPI::DataCopyGather((AscendC::MicroAPI::RegTensor<int16_t> &)vd0, cur_x_addr, vreg_indice,
                                          gather_mask);
        AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(cur_y_addr, vd0, preg0);
      } else {
        AscendC::MicroAPI::DataCopyGather(vd0, cur_x_addr, vreg_indice, gather_mask);
        AscendC::MicroAPI::DataCopy(cur_y_addr, vd0, preg0);
      }
    }
  }
}
template <typename T1, typename T2, bool negative_index_support>
inline __aicore__ bool GatherExtendVReg(int32_t offset, AscendC::LocalTensor<uint8_t> &tmp_buf,
                                        uint32_t gather_dim_size, const AscendC::LocalTensor<T1> &yLocal,
                                        const AscendC::GlobalTensor<T1> &src1, const AscendC::GlobalTensor<T2> &src2,
                                        int32_t cols, uint32_t tmp_buf_size) {
  using indiceType = typename IndexTypeGet<T1>::type;
  constexpr uint32_t vf_len = AscendC::GetVecLen() / sizeof(indiceType);
  const uint64_t param_bytes = static_cast<uint64_t>(gather_dim_size) * sizeof(T1);
  const uint64_t indice_offset =
      (param_bytes + AscendC::ONE_BLK_SIZE - 1) / AscendC::ONE_BLK_SIZE * AscendC::ONE_BLK_SIZE;
  if (indice_offset >= tmp_buf_size) {
    return false;
  }
  uint32_t remaining_bytes = tmp_buf_size - indice_offset;
  uint32_t bytes_per_index = sizeof(T2);
  if constexpr (sizeof(T2) == sizeof(int64_t)) {
    // Reserve the int32 conversion buffer and its worst-case 64-byte alignment gap.
    if (remaining_bytes < INT64_OFFSET) {
      return false;
    }
    remaining_bytes -= INT64_OFFSET - 1;
    bytes_per_index += sizeof(int32_t);
  }
  // LoadIndices reads a full register, including on the final masked iteration.
  // Full-register chunks also keep every output chunk aligned for vector stores.
  const uint32_t max_out_cols = remaining_bytes / bytes_per_index / vf_len * vf_len;
  if (max_out_cols == 0) {
    return false;
  }

  LocalTensor<T1> xLocal;
  xLocal = tmp_buf[0].template ReinterpretCast<T1>();

  LocalTensor<T2> indicesLocal;
  indicesLocal = tmp_buf[indice_offset].template ReinterpretCast<T2>();

  LocalTensor<int32_t> tmpLocal;

  DataCopyPadExtParams<T1> param_src_pad;
  param_src_pad.isPad = false;
  param_src_pad.leftPadding = 0;
  param_src_pad.rightPadding = 0;
  param_src_pad.paddingValue = 0;
  AscendC::DataCopyExtParams param_src;
  param_src.blockCount = 1;
  param_src.blockLen = gather_dim_size * sizeof(T1);
  param_src.srcStride = 0;
  param_src.dstStride = 0;
  int32_t event_id_v_to_mte2_0 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
  AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id_v_to_mte2_0);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event_id_v_to_mte2_0);
  AscendC::DataCopyPad(xLocal, src1, param_src, param_src_pad);
  int64_t out_loop_num = (cols + max_out_cols - 1) / max_out_cols;
  int32_t tail_out_cols = cols - (out_loop_num - 1) * max_out_cols;
  __local_mem__ T1 *y_addr = (__local_mem__ T1 *)yLocal.GetPhyAddr();
  __local_mem__ T1 *x_addr = (__local_mem__ T1 *)xLocal.GetPhyAddr();
  int32_t event_id_mte2_to_v = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
  int32_t event_id_v_to_mte2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));

  for (int64_t j = 0; j < out_loop_num; j++) {
    int32_t cur_out_cols = (j == out_loop_num - 1) ? tail_out_cols : max_out_cols;
    AscendC::DataCopyExtParams indice_src;
    indice_src.blockCount = 1;
    indice_src.blockLen = cur_out_cols * sizeof(T2);
    indice_src.srcStride = 0;
    indice_src.dstStride = 0;
    AscendC::DataCopyPad(indicesLocal[0], src2[j * max_out_cols], indice_src, AscendC::DataCopyPadExtParams<T2>());

    __local_mem__ int32_t *indice_addr;

    uint32_t num_per_loop = cur_out_cols;
    uint16_t vf_loop_num = (num_per_loop + vf_len - 1) / vf_len;
    int32_t dim_size = gather_dim_size;
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id_mte2_to_v);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id_mte2_to_v);
    if constexpr (sizeof(T2) == sizeof(int64_t)) {
      int32_t base_offset = indice_offset + num_per_loop * sizeof(T2);
      int32_t tmp_offset = (base_offset + INT64_OFFSET - 1) / INT64_OFFSET * INT64_OFFSET;
      tmpLocal = tmp_buf[tmp_offset].template ReinterpretCast<int32_t>();
      Cast(tmpLocal, indicesLocal, AscendC::RoundMode::CAST_NONE, cur_out_cols);
      if constexpr (negative_index_support) {
        ConvertNegIndices<T1, T2>(tmpLocal, cur_out_cols, gather_dim_size);
      }
      indice_addr = (__local_mem__ int32_t *)tmpLocal[0].GetPhyAddr();
    } else {
      if constexpr (negative_index_support) {
        ConvertNegIndices<T1, T2>(indicesLocal, cur_out_cols, gather_dim_size);
      }
      indice_addr = (__local_mem__ int32_t *)indicesLocal[0].GetPhyAddr();
    }
    VRegGather<T1, T2>(y_addr, x_addr, indice_addr, num_per_loop, dim_size, vf_len, vf_loop_num);
    y_addr += num_per_loop;
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id_v_to_mte2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event_id_v_to_mte2);
  }
  return true;
}
/***********************************************************************************************************************************************************************************/

/**************************************************************************************** 模板4
 * 小尾轴场景，使用短向量搬运 *********************************************************************/
template <typename T1, typename T2, int32_t CASE, int32_t VECTORIZED_AXIS_SIZE, bool negative_index_support>
inline __aicore__ void GatherExtendShortVector(
    AscendC::LocalTensor<T1> &dst, const AscendC::GlobalTensor<T1> &src1, const AscendC::GlobalTensor<T2> &src2,
    uint32_t ub_actual_size, uint64_t offset, uint32_t x2_tensor_size, uint32_t x1_gather_dim_size,
    uint32_t x1_gather_dim_stride, AscendC::LocalTensor<uint8_t> &tmp_buf, uint32_t tmp_buf_size, uint32_t param_size,
    uint32_t param_axis_size, DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(1), DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(2),
    DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(3), DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(4),
    DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(5), DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(6),
    DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(7), DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(8)) {
  __gm__ int64_t *x1_gm = (__gm__ int64_t *)(src1.GetPhyAddr());
  __gm__ T2 *x2_gm = (__gm__ T2 *)(src2.GetPhyAddr());
  __ubuf__ int64_t *dst_p = (__ubuf__ int64_t *)(dst.GetPhyAddr());
  uint32_t x1_gather_dim_stride_m{0}, x1_gather_dim_stride_shift{0}, x2_tensor_size_m{0}, x2_tensor_size_shift{0};
  LocalTensor<T1> xLocal;
  xLocal = tmp_buf[0].template ReinterpretCast<T1>();
  if constexpr (CASE == CASE2) {
    AscendC::GetUintDivMagicAndShift(x1_gather_dim_stride_m, x1_gather_dim_stride_shift, x1_gather_dim_stride);
  }
  if constexpr (CASE == CASE3) {
    AscendC::GetUintDivMagicAndShift(x2_tensor_size_m, x2_tensor_size_shift, x2_tensor_size);
  }
  if constexpr (CASE == CASE4) {
    AscendC::GetUintDivMagicAndShift(x1_gather_dim_stride_m, x1_gather_dim_stride_shift, x1_gather_dim_stride);
    AscendC::GetUintDivMagicAndShift(x2_tensor_size_m, x2_tensor_size_shift, x2_tensor_size);
  }
  DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_VARIABLES(1);
  DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_VARIABLES(2);
  DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_VARIABLES(3);
  DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_VARIABLES(4);
  DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_VARIABLES(5);
  DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_VARIABLES(6);
  DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_VARIABLES(7);
  DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_VARIABLES(8);
  GetMagicShiftForVectiorizedSize<VECTORIZED_AXIS_SIZE>(
      PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(1), PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(2),
      PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(3), PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(4),
      PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(5), PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(6),
      PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(7), PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(8));
  int32_t event_id_v_to_mte3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
  if (param_size <= 20000 && tmp_buf_size > 8192 && x2_tensor_size >= 750) {
    DataCopyPadExtParams<T1> param_src_pad;
    param_src_pad.isPad = false;
    param_src_pad.leftPadding = 0;
    param_src_pad.rightPadding = 0;
    param_src_pad.paddingValue = 0;
    AscendC::DataCopyExtParams param_src;
    param_src.blockCount = 1;
    param_src.blockLen = param_size * sizeof(T1);
    param_src.srcStride = 0;
    param_src.dstStride = 0;
    int32_t event_id_v_to_mte2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id_v_to_mte2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event_id_v_to_mte2);
    AscendC::DataCopyPad(xLocal, src1, param_src, param_src_pad);
    int32_t event_id_mte2_to_v = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id_mte2_to_v);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id_mte2_to_v);
    __ubuf__ int64_t *xAddr = (__ubuf__ int64_t *)xLocal.GetPhyAddr();
    AscendC::Simt::VF_CALL<GatherSimtParamFullLoad<int64_t, T2, uint32_t, CASE, VECTORIZED_AXIS_SIZE,
                                                   sizeof(int64_t) / sizeof(T1), negative_index_support>>(
        AscendC::Simt::Dim3(THREAD_NUMBER), dst_p, xAddr, x2_gm, ub_actual_size, static_cast<uint32_t>(offset),
        static_cast<uint32_t>(x1_gather_dim_size), static_cast<uint32_t>(x2_tensor_size_m),
        static_cast<uint32_t>(x2_tensor_size_shift), static_cast<uint32_t>(x2_tensor_size),
        static_cast<uint32_t>(x1_gather_dim_stride_m), static_cast<uint32_t>(x1_gather_dim_stride_shift),
        static_cast<uint32_t>(x1_gather_dim_stride), PASS_VECTORIZED_AXIS_PARAMS_SIMT(1),
        PASS_VECTORIZED_AXIS_PARAMS_SIMT(2), PASS_VECTORIZED_AXIS_PARAMS_SIMT(3), PASS_VECTORIZED_AXIS_PARAMS_SIMT(4),
        PASS_VECTORIZED_AXIS_PARAMS_SIMT(5), PASS_VECTORIZED_AXIS_PARAMS_SIMT(6), PASS_VECTORIZED_AXIS_PARAMS_SIMT(7),
        PASS_VECTORIZED_AXIS_PARAMS_SIMT(8));
  } else {
    AscendC::Simt::VF_CALL<GatherSimt<int64_t, T2, uint32_t, CASE, VECTORIZED_AXIS_SIZE, sizeof(int64_t) / sizeof(T1),
                                      negative_index_support>>(
        AscendC::Simt::Dim3(THREAD_NUMBER), dst_p, x1_gm, x2_gm, ub_actual_size, static_cast<uint32_t>(offset),
        static_cast<uint32_t>(x1_gather_dim_size), static_cast<uint32_t>(x2_tensor_size_m),
        static_cast<uint32_t>(x2_tensor_size_shift), static_cast<uint32_t>(x2_tensor_size),
        static_cast<uint32_t>(x1_gather_dim_stride_m), static_cast<uint32_t>(x1_gather_dim_stride_shift),
        static_cast<uint32_t>(x1_gather_dim_stride), PASS_VECTORIZED_AXIS_PARAMS_SIMT(1),
        PASS_VECTORIZED_AXIS_PARAMS_SIMT(2), PASS_VECTORIZED_AXIS_PARAMS_SIMT(3), PASS_VECTORIZED_AXIS_PARAMS_SIMT(4),
        PASS_VECTORIZED_AXIS_PARAMS_SIMT(5), PASS_VECTORIZED_AXIS_PARAMS_SIMT(6), PASS_VECTORIZED_AXIS_PARAMS_SIMT(7),
        PASS_VECTORIZED_AXIS_PARAMS_SIMT(8));
  }
  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event_id_v_to_mte3);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event_id_v_to_mte3);
}
/***********************************************************************************************************************************************************************************/

template <typename T1, typename T2, int32_t CASE, int32_t VECTORIZED_AXIS_SIZE, bool negative_index_support>
inline __aicore__ void GatherExtend(AscendC::LocalTensor<T1> &dst, const AscendC::GlobalTensor<T1> &src1,
                                    const AscendC::GlobalTensor<T2> &src2, uint32_t ub_actual_size, uint64_t offset,
                                    uint64_t x2_tensor_size, uint64_t x1_gather_dim_size, uint64_t x1_gather_dim_stride,
                                    AscendC::LocalTensor<uint8_t> &tmp_buf, uint32_t tmp_buf_size, uint32_t param_size,
                                    uint32_t param_axis_size, DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(1),
                                    DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(2), DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(3),
                                    DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(4), DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(5),
                                    DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(6), DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(7),
                                    DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(8)) {
  {
    bool run_later = (VECTORIZED_AXIS_SIZE == 1 && param_size <= 30000 && param_axis_size == 1 && tmp_buf_size > 8192);
    if (run_later && GatherExtendVReg<T1, T2, negative_index_support>(offset, tmp_buf, param_size, dst, src1,
                                                                      src2[offset], ub_actual_size, tmp_buf_size)) {
      return;
    }
  }
  {
    bool run_later = ub_actual_size > THREAD_NUMBER && VECTORIZED_AXIS_SIZE == 1 &&
                     Y_VECTORIZED_AXIS_SIZE_STRIDE(1) == 1 && x1_gather_dim_stride >= THREAD_NUMBER;
    if (run_later) {
      GatherExtendDataCopy<T1, T2, CASE, VECTORIZED_AXIS_SIZE, negative_index_support>(
          dst, src1, src2, ub_actual_size, offset, x2_tensor_size, x1_gather_dim_size, x1_gather_dim_stride,
          PASS_VECTORIZED_AXIS_PARAMS_SIMD(1), PASS_VECTORIZED_AXIS_PARAMS_SIMD(2), PASS_VECTORIZED_AXIS_PARAMS_SIMD(3),
          PASS_VECTORIZED_AXIS_PARAMS_SIMD(4), PASS_VECTORIZED_AXIS_PARAMS_SIMD(5), PASS_VECTORIZED_AXIS_PARAMS_SIMD(6),
          PASS_VECTORIZED_AXIS_PARAMS_SIMD(7), PASS_VECTORIZED_AXIS_PARAMS_SIMD(8));
      return;
    }
  }
  {
    bool run_later = sizeof(int64_t) / sizeof(T1) > 1 && Y_VECTORIZED_AXIS_SIZE_STRIDE(1) == 1 &&
                     x1_gather_dim_stride < THREAD_NUMBER && x1_gather_dim_size > 1 &&
                     x1_gather_dim_stride % (sizeof(int64_t) / sizeof(T1)) == 0 &&
                     ub_actual_size % (sizeof(int64_t) / sizeof(T1)) == 0 &&
                     VECTORIZED_AXIS_SIZE(1) % (sizeof(int64_t) / sizeof(T1)) == 0;
    if (run_later) {
      GatherExtendShortVector<T1, T2, CASE, VECTORIZED_AXIS_SIZE, negative_index_support>(
          dst, src1, src2, ub_actual_size, offset, x2_tensor_size, x1_gather_dim_size, x1_gather_dim_stride, tmp_buf,
          tmp_buf_size, param_size, param_axis_size, PASS_VECTORIZED_AXIS_PARAMS_SIMD(1),
          PASS_VECTORIZED_AXIS_PARAMS_SIMD(2), PASS_VECTORIZED_AXIS_PARAMS_SIMD(3), PASS_VECTORIZED_AXIS_PARAMS_SIMD(4),
          PASS_VECTORIZED_AXIS_PARAMS_SIMD(5), PASS_VECTORIZED_AXIS_PARAMS_SIMD(6), PASS_VECTORIZED_AXIS_PARAMS_SIMD(7),
          PASS_VECTORIZED_AXIS_PARAMS_SIMD(8));
      return;
    }
  }
  GatherExtendDefault<T1, T2, CASE, VECTORIZED_AXIS_SIZE, negative_index_support>(
      dst, src1, src2, ub_actual_size, offset, x2_tensor_size, x1_gather_dim_size, x1_gather_dim_stride,
      PASS_VECTORIZED_AXIS_PARAMS_SIMD(1), PASS_VECTORIZED_AXIS_PARAMS_SIMD(2), PASS_VECTORIZED_AXIS_PARAMS_SIMD(3),
      PASS_VECTORIZED_AXIS_PARAMS_SIMD(4), PASS_VECTORIZED_AXIS_PARAMS_SIMD(5), PASS_VECTORIZED_AXIS_PARAMS_SIMD(6),
      PASS_VECTORIZED_AXIS_PARAMS_SIMD(7), PASS_VECTORIZED_AXIS_PARAMS_SIMD(8));
  return;
}

#endif  // __ASCENDC_API_GATHER_REGBASE_H__
