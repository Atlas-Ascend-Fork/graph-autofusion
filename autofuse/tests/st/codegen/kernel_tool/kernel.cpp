#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling) AutofuseTilingData t = *(AutofuseTilingData *)tiling;
#include "kernel_operator.h"
#include "autofuse_tiling_data.h"

using namespace AscendC;

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_UTILS_H__
#define __ASCENDC_API_UTILS_H__

#include "kernel_operator.h"

namespace {

constexpr inline __aicore__ unsigned int Ceiling(float num) {
  int ceil_num = static_cast<int>(num);
  float epsilon = 1e-6;
  if (num >= epsilon) {
    return (num - static_cast<float>(ceil_num) < epsilon) ? ceil_num : (ceil_num + 1);
  } else {
    return (static_cast<float>(ceil_num) - num < epsilon) ? ceil_num : (ceil_num + 1);
  }
}

constexpr inline __aicore__ unsigned int Ceiling(unsigned int num) {
  return num;
}

constexpr inline __aicore__ int64_t Ceiling(int64_t num) {
  return num;
}

constexpr inline __aicore__ unsigned int Ceiling(int num) {
  return num;
}

constexpr inline __aicore__ int64_t Floor(float num) {
  float epsilon = 1e-6;
  if (num >= epsilon) {
    return static_cast<int64_t>(num);
  } else {
    int64_t int_num = static_cast<int64_t>(num);
    return (static_cast<float>(int_num) - num < epsilon) ? int_num : int_num - 1;
  }
}

constexpr inline __aicore__ int64_t Floor(int64_t num) {
  return num;
}

constexpr inline __aicore__ int Floor(int num) {
  return num;
}

constexpr inline __aicore__ unsigned int Floor(unsigned int num) {
  return num;
}

constexpr inline __aicore__ float Rational(float a, float b) {
  return (a / b);
}

template <typename T1, typename T2>
constexpr inline __aicore__ T2 Mod(T1 a, T2 b) {
  if constexpr (std::is_same<T1, float>::value || std::is_same<T2, float>::value) {
    ASCENDC_ASSERT(((b > 1e-6) || (b < -1e-6)), { KERNEL_LOG(KERNEL_ERROR, "b can't be equal to 0, b is %f!", b); });
    int64_t trunc_num = static_cast<int>(a / b);
    return static_cast<T1>(a - trunc_num * b);
  } else if constexpr (std::is_same<T1, uint64_t>::value && std::is_same<T2, uint64_t>::value) {
    ASCENDC_ASSERT(b != 0, { KERNEL_LOG(KERNEL_ERROR, "b can't be equal to 0, b is %d!", b); });
    uint64_t a_tmp = static_cast<uint64_t>(a);
    uint64_t b_tmp = static_cast<uint64_t>(b);
    uint64_t mod_num = a_tmp % b_tmp;
    return static_cast<uint64_t>(mod_num);
  } else if constexpr (std::is_same<T1, uint64_t>::value || std::is_same<T2, uint64_t>::value) {
    ASCENDC_ASSERT(true, {
      KERNEL_LOG(KERNEL_ERROR,
                 "does not support mixed type of uint64 because of possible "
                 "overflow!");
    });
    return 0;
  } else {
    ASCENDC_ASSERT(b != 0, { KERNEL_LOG(KERNEL_ERROR, "b can't be equal to 0, b is %d!", b); });
    int64_t a_tmp = static_cast<int64_t>(a);
    int64_t b_tmp = static_cast<int64_t>(b);
    int64_t mod_num = a_tmp % b_tmp;
    return static_cast<T1>(mod_num);
  }
}

template <typename TilingData>
inline __aicore__ bool MatchTilingKeyAndBlockDim(TilingData &t, uint32_t tiling_key) {
  auto block_dim = GetBlockIdx();
  // reuse ub_size as block_offset
  return (t.tiling_key == tiling_key) && (block_dim >= t.ub_size) && ((block_dim - t.ub_size) < t.block_dim);
}
}  // namespace
namespace KernelUtils {
static constexpr uint64_t CONST1 = 1;
static constexpr uint64_t CONST2 = 2;
static constexpr uint64_t CONST4 = 4;
static constexpr uint64_t CONST63 = 63;

constexpr inline __aicore__ int Ceil(float num) {
  int ceil_num = (int)num;
  return num == (float)ceil_num ? ceil_num : (ceil_num + 1);
}

template <typename T>
constexpr inline __aicore__ T Min(const T a) {
  return a;
}

template <typename T, typename... Ts>
constexpr inline __aicore__ T Min(const T a, const Ts... ts) {
  return a > Min(ts...) ? Min(ts...) : a;
}

template <typename T>
constexpr inline __aicore__ T Max(const T a) {
  return a;
}

template <typename T, typename... Ts>
constexpr inline __aicore__ T Max(const T a, const Ts... ts) {
  return a > Max(ts...) ? a : Max(ts...);
}

template <typename T, typename... Ts>
constexpr inline __aicore__ T Sum(const T a, const Ts... ts) {
  return (a + ... + ts);
}

template <typename DataType>
constexpr inline __aicore__ uint16_t BlkSize() {
  return ONE_BLK_SIZE / sizeof(DataType);
}

template <typename DataType>
constexpr inline __aicore__ uint32_t BlkNum(uint32_t size) {
  return size / BlkSize<DataType>();
}

template <typename DataType>
constexpr inline __aicore__ uint16_t RptSize() {
  return ONE_REPEAT_BYTE_SIZE / sizeof(DataType);
}

template <typename DataType>
constexpr inline __aicore__ uint32_t RptNum(uint32_t size) {
  return size / RptSize<DataType>();
}

template <typename DataType>
constexpr inline __aicore__ uint32_t MaxRptSize() {
  return MAX_REPEAT_NUM * RptSize<DataType>();
}

template <typename DataType>
constexpr inline __aicore__ uint32_t BlkAlign(uint32_t size) {
  return (size * sizeof(DataType) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE / sizeof(DataType);
}

template <typename DataType>
constexpr inline __aicore__ uint32_t BlkFloorAlign(uint32_t size) {
  if ((size * sizeof(DataType) + 1) > ONE_BLK_SIZE) {
    return ((size * sizeof(DataType) + 1) - ONE_BLK_SIZE) / ONE_BLK_SIZE * ONE_BLK_SIZE / sizeof(DataType);
  }
  return 0U;
}

constexpr inline __aicore__ uint32_t SizeAlign(uint32_t size, uint32_t align_size) {
  return (size + align_size - 1) / align_size * align_size;
}

template <typename T, typename T1>
inline __aicore__ LocalTensor<T> NewTensor(LocalTensor<T1> &tmp_buf, uint32_t offset, uint32_t cnt) {
  AscendC::LocalTensor<T> alloc_buf = tmp_buf[offset].template ReinterpretCast<T>();
  alloc_buf.SetSize(cnt);
  return alloc_buf;
}

__aicore__ inline uint64_t FindNearestPower2(const uint64_t value) {
  if (value == 0) {
    return 0;
  } else if (value <= CONST2) {
    return CONST1;
  } else if (value <= CONST4) {
    return CONST2;
  } else {
    const uint64_t num = value - CONST1;
    const uint64_t pow = CONST63 - clz(num);
    return (CONST1 << pow);
  }
}

__aicore__ inline uint64_t CalLog2(uint64_t value) {
  uint64_t res = 0;
  while (value > CONST1) {
    value = value >> CONST1;
    res++;
  }
  return res;
}

}  // namespace KernelUtils

template <typename T>
constexpr inline __aicore__ T Min(const T a) {
  return a;
}

template <typename T, typename... Ts>
constexpr inline __aicore__ T Min(const T a, const Ts... ts) {
  return KernelUtils::Min(a, ts...);
}

template <typename T>
constexpr inline __aicore__ T Max(const T a) {
  return a;
}

template <typename T, typename... Ts>
constexpr inline __aicore__ T Max(const T a, const Ts... ts) {
  return KernelUtils::Max(a, ts...);
}

inline __aicore__ uint64_t Log(uint64_t value) {
  return KernelUtils::CalLog2(value);
}

inline __aicore__ float Pow(const int base, int exponent) {
  float result = 1.0;
  if (exponent >= 0) {
    for (int i = 0; i < exponent; ++i) {
      result *= base;
    }
  } else {
    exponent = -exponent;
    for (int i = 0; i < exponent; ++i) {
      result *= base;
    }
    result = 1 / result;
  }

  return result;
}

// 定义inf
template <typename T, typename U>
__aicore__ inline U GetScalarValueByBitCode(T bit_code) {
  union ScalarBitcode {
    __aicore__ ScalarBitcode() {}
    T input;
    U output;
  } data;

  data.input = bit_code;
  return static_cast<U>(data.output);
}

template <typename T>
constexpr __aicore__ static inline T AfInfinity() {
  static_assert(SupportType<T, half, float>(), "current data type does not support inf");
  if constexpr (std::is_same_v<T, half>) {
    return GetScalarValueByBitCode<uint16_t, T>(0x7C00U);
  } else if constexpr (std::is_same_v<T, float>) {
    return GetScalarValueByBitCode<uint32_t, T>(0x7F800000U);
  }

  return T();
}

static constexpr float ROUND_TO_NEAREST_INT_BIAS = 0.5f;
template <typename T>
inline __aicore__ uint32_t ConvertToUint32(T value) {
  if constexpr (std::is_floating_point<T>::value) {
    // 默认tiling_data是uint32_t的，而ascendc不支持将uint32_t转为float
    // 因此参与float计算时，aicore函数中不能直接将float转换为unsigned int，需要转换为int，再转换为unsigned int
    return static_cast<uint32_t>(static_cast<int32_t>(value + ROUND_TO_NEAREST_INT_BIAS));
  } else {
    return static_cast<uint32_t>(value);
  }
}

#endif  // __ASCENDC_API_UTILS_H__

#ifndef __ASCENDC_API_BRC_INLINE_API_H__
#define __ASCENDC_API_BRC_INLINE_API_H__

#include "kernel_operator.h"

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
  if (cut_quotient >= block) {
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
  uint8_t dst_repeat_stride = first_axis_v_stride / elem_in_one_block;
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

#ifndef __ASCENDC_API_DATACOPY_H__
#define __ASCENDC_API_DATACOPY_H__

#include "kernel_operator.h"

template <typename T>
inline __aicore__ void DataCopyPadExtend(const AscendC::LocalTensor<T> &dst, const AscendC::GlobalTensor<T> &src,
                                         uint32_t block_count, uint32_t block_len, uint32_t src_stride,
                                         uint32_t dst_stride) {
  uint32_t align_num = ONE_BLK_SIZE / sizeof(T);
  AscendC::DataCopyExtParams param;
  param.blockCount = block_count;
  param.blockLen = block_len * sizeof(T);
  param.srcStride = src_stride * sizeof(T);
  param.dstStride = dst_stride / align_num;

  AscendC::DataCopyPadExtParams<T> pad_params = {true, 0, 0, 0};
  AscendC::DataCopyPad(dst, src, param, pad_params);
}

template <typename T>
inline __aicore__ void DataCopyPadExtend(const AscendC::GlobalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                         uint32_t block_count, uint32_t block_len, uint32_t src_stride,
                                         uint32_t dst_stride) {
  uint32_t align_num = ONE_BLK_SIZE / sizeof(T);
  AscendC::DataCopyExtParams param;
  param.blockCount = block_count;
  param.blockLen = block_len * sizeof(T);
  param.srcStride = src_stride / align_num;
  param.dstStride = dst_stride * sizeof(T);

  AscendC::DataCopyPad(dst, src, param);
}

template <typename T>
inline __aicore__ void DataCopyExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                      const uint32_t size) {
  AscendC::DataCopy(dst, src, size);
}

#endif  // __ASCENDC_API_DATACOPY_H__
inline __aicore__ void autofuse_pointwise_0__abs__add_0_general_0_nil_2_nil(GM_ADDR Add_out0_graph_Data_0,
                                                                            GM_ADDR Add_out0_graph_Data_1,
                                                                            GM_ADDR Add_out0_graph_Output_0,
                                                                            GM_ADDR workspace,
                                                                            const AutofuseTilingData *t) {
  const int z0_axis_size = t->s2;
  const int z0_loop_size = z0_axis_size;
  const int z0_actual_size = z0_axis_size;
  const int z1_axis_size = t->s3;
  const int z1_loop_size = z1_axis_size;
  const int z1_actual_size = z1_axis_size;
  const int z0z1_axis_size = (t->s2 * t->s3);
  const int z0z1_loop_size = z0z1_axis_size;
  const int z0z1_actual_size = z0z1_axis_size;
  const int z0z1t_axis_size = t->z0z1t_size;
  const int z0z1t_tail_size = z0z1_loop_size % z0z1t_axis_size;
  const int z0z1T_axis_size = z0z1_loop_size / z0z1t_axis_size;
  const int z0z1T_loop_size = z0z1T_axis_size + (z0z1t_tail_size > 0);
  const int z0z1Tb_axis_size = t->z0z1Tb_size;
  const int z0z1Tb_tail_size = z0z1T_loop_size % z0z1Tb_axis_size;
  const int z0z1TB_axis_size = z0z1T_loop_size / z0z1Tb_axis_size;
  const int z0z1TB_loop_size = z0z1TB_axis_size + (z0z1Tb_tail_size > 0);
  int block_dim = GetBlockIdx();
  if (block_dim >= t->block_dim) {
    return;
  }
  const int z0z1TB = block_dim % z0z1TB_loop_size;

  GlobalTensor<float> global_0;
  global_0.SetGlobalBuffer((__gm__ float *)Add_out0_graph_Data_0);
  GlobalTensor<float> global_1;
  global_1.SetGlobalBuffer((__gm__ float *)Add_out0_graph_Data_1);
  GlobalTensor<float> global_2;
  global_2.SetGlobalBuffer((__gm__ float *)Add_out0_graph_Output_0);

  TPipe tpipe;

  const uint32_t local_3_size = KernelUtils::BlkAlign<float>((t->z0z1t_size - 1) + 1);
  const uint32_t local_3_que_buf_num = 2;
  const uint32_t local_4_size = KernelUtils::BlkAlign<float>((t->z0z1t_size - 1) + 1);
  const uint32_t local_5_size = KernelUtils::BlkAlign<float>((t->z0z1t_size - 1) + 1);
  const uint32_t local_5_que_buf_num = 2;
  const uint32_t local_6_size = KernelUtils::BlkAlign<float>((t->z0z1t_size - 1) + 1);
  const uint32_t local_6_que_buf_num = 2;

  // const uint32_t b0_size = KernelUtils::Max(local_4_size * sizeof(float));
  TBuf<TPosition::VECCALC> b0;
  // tpipe.InitBuffer(b0, KernelUtils::BlkAlign<uint8_t>(b0_size));
  tpipe.InitBuffer(b0, t->b0_size);
  LocalTensor<float> local_4 = b0.Get<float>();

  // const uint32_t q0_size = KernelUtils::Max(local_3_size * sizeof(float));
  const uint32_t q0_buf_num = KernelUtils::Max(2);
  TQue<TPosition::VECIN, 1> q0;
  // tpipe.InitBuffer(q0, q0_buf_num, KernelUtils::BlkAlign<uint8_t>(q0_size));
  tpipe.InitBuffer(q0, q0_buf_num, t->q0_size);

  // const uint32_t q1_size = KernelUtils::Max(local_5_size * sizeof(float));
  const uint32_t q1_buf_num = KernelUtils::Max(2);
  TQue<TPosition::VECIN, 1> q1;
  // tpipe.InitBuffer(q1, q1_buf_num, KernelUtils::BlkAlign<uint8_t>(q1_size));
  tpipe.InitBuffer(q1, q1_buf_num, t->q1_size);

  // const uint32_t q2_size = KernelUtils::Max(local_6_size * sizeof(float));
  const uint32_t q2_buf_num = KernelUtils::Max(2);
  TQue<TPosition::VECOUT, 1> q2;
  // tpipe.InitBuffer(q2, q2_buf_num, KernelUtils::BlkAlign<uint8_t>(q2_size));
  tpipe.InitBuffer(q2, q2_buf_num, t->q2_size);

  int z0z1Tb_actual_size = z0z1TB < z0z1TB_axis_size ? z0z1Tb_axis_size : z0z1Tb_tail_size;
  int z0z1Tb_loop_size = z0z1Tb_actual_size;
  int32_t block_dim_offset = z0z1TB * t->z0z1Tb_size;
  for (int z0z1Tb = 0; z0z1Tb < z0z1Tb_loop_size; z0z1Tb++) {
    int z0z1T = block_dim_offset + z0z1Tb;
    int z0z1t_actual_size = z0z1T < z0z1T_axis_size ? z0z1t_axis_size : z0z1t_tail_size;
    int z0z1t_loop_size = z0z1t_actual_size;
    uint32_t q0_reuse0_offset = 0;
    LocalTensor<uint8_t> q0_buf = q0.AllocTensor<uint8_t>();
    const uint32_t local_3_actual_size = (z0z1t_actual_size - 1) + 1;
    LocalTensor<float> local_3;
    local_3 = q0_buf[q0_reuse0_offset].template ReinterpretCast<float>();
    DataCopyPadExtend(local_3,
                      global_0[(int64_t)z0z1TB * (int64_t)(t->z0z1Tb_size * t->z0z1t_size) +
                               (int64_t)z0z1Tb * (int64_t)t->z0z1t_size + 0],
                      1, z0z1t_actual_size, 0, 0);
    q0.EnQue(q0_buf);

    q0_buf = q0.DeQue<uint8_t>();
    const uint32_t local_4_actual_size = (z0z1t_actual_size - 1) + 1;
    Abs(local_4[0], local_3[0], KernelUtils::BlkAlign<float>(local_3_actual_size));
    q0.FreeTensor(q0_buf);

    uint32_t q1_reuse1_offset = 0;
    LocalTensor<uint8_t> q1_buf = q1.AllocTensor<uint8_t>();
    const uint32_t local_5_actual_size = (z0z1t_actual_size - 1) + 1;
    LocalTensor<float> local_5;
    local_5 = q1_buf[q1_reuse1_offset].template ReinterpretCast<float>();
    DataCopyPadExtend(local_5,
                      global_1[(int64_t)z0z1TB * (int64_t)(t->z0z1Tb_size * t->z0z1t_size) +
                               (int64_t)z0z1Tb * (int64_t)t->z0z1t_size + 0],
                      1, z0z1t_actual_size, 0, 0);
    q1.EnQue(q1_buf);

    AscendC::PipeBarrier<PIPE_V>();
    q1_buf = q1.DeQue<uint8_t>();
    uint32_t q2_reuse2_offset = 0;
    LocalTensor<uint8_t> q2_buf = q2.AllocTensor<uint8_t>();
    const uint32_t local_6_actual_size = (z0z1t_actual_size - 1) + 1;
    LocalTensor<float> local_6;
    local_6 = q2_buf[q2_reuse2_offset].template ReinterpretCast<float>();
    Add(local_6[0], local_4[0], local_5[0], local_4_actual_size);
    q2.EnQue(q2_buf);
    q1.FreeTensor(q1_buf);

    q2_buf = q2.DeQue<uint8_t>();
    DataCopyPadExtend(global_2[(int64_t)z0z1TB * (int64_t)(t->z0z1Tb_size * t->z0z1t_size) +
                               (int64_t)z0z1Tb * (int64_t)t->z0z1t_size + 0],
                      local_6, 1, z0z1t_actual_size, 0, 0);
    q2.FreeTensor(q2_buf);
  }
}
extern "C" __global__ __aicore__ void autofuse_pointwise_0__abs__add(GM_ADDR Add_out0_graph_Data_0,
                                                                     GM_ADDR Add_out0_graph_Data_1,
                                                                     GM_ADDR Add_out0_graph_Output_0, GM_ADDR workspace,
                                                                     GM_ADDR gm_tiling_data) {
  REGISTER_TILING_DEFAULT(AutofuseTilingData);
  GET_TILING_DATA(t, gm_tiling_data);
  if (TILING_KEY_IS(0)) {
    autofuse_pointwise_0__abs__add_0_general_0_nil_2_nil(Add_out0_graph_Data_0, Add_out0_graph_Data_1,
                                                         Add_out0_graph_Output_0, workspace, &t);
  }
}
