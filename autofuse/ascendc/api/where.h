/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_WHERE_H__
#define __ASCENDC_API_WHERE_H__

struct DoSelectParams {
  __aicore__ DoSelectParams() {
    do_size = 0;
    mask = 0;
    repeat_times = 0;
    calc_size = 0;
    src0_select_offset = 1;
    src1_select_offset = 1;
    // normal
    src0_slice_offset = 0;
    src1_slice_offset = 0;
    output_stride = 0;
    input0_stride = 0;
    input1_stride = 0;
    mask_stride = 0;
    mask_offset = 0;
    src0_offset = 0;
    src1_offset = 0;
    dst_offset = 0;
  }

  uint32_t do_size;                              // will calculate number size
  uint64_t mask;                                 // Select api mask
  uint8_t repeat_times;                          // Select api repeatTimes
  BinaryRepeatParams rpt_params;                 // Select api repeatParams
  uint32_t calc_size;                            // has calculated number size
  AscendC::LocalTensor<half> mask_cast_buf;      // cast condition to half
  AscendC::LocalTensor<int16_t> mask_shift_buf;  // mask left shift when input is int64
  AscendC::LocalTensor<float> src0_cast_buf;     // cast src1 to float
  AscendC::LocalTensor<float> sel_res_buf;       // temp buffer for select result
  uint32_t src0_select_offset;
  uint32_t src1_select_offset;
  // normal
  uint32_t src0_slice_offset;
  uint32_t src1_slice_offset;
  uint32_t output_stride;
  uint32_t input0_stride;
  uint32_t input1_stride;
  uint32_t mask_stride;
  uint32_t dst_offset;                        // will calculate number size
  uint32_t src0_offset;                       // will calculate number size
  uint32_t src1_offset;                       // will calculate number size
  uint32_t mask_offset;                       // will calculate number size
  AscendC::LocalTensor<uint8_t> select_pool;  // int64 select 的谓词缓冲池（tmp_buf 复用）
};

template <typename O, typename I>
inline __aicore__ void SafeCast(const AscendC::LocalTensor<O> &dst,
                                const AscendC::LocalTensor<I> &src,  // condition
                                const uint32_t size) {
  // float -> int64/int32/int16 :: CAST_RINT
  if constexpr (std::is_same<I, float>::value and (std::is_same<O, int64_t>::value || std::is_same<O, int32_t>::value ||
                                                   std::is_same<O, int16_t>::value)) {
    Cast(dst, src, RoundMode::CAST_RINT, size);
    // int64 -> float :: CAST_RINT
  } else if constexpr (std::is_same<I, int64_t>::value and std::is_same<O, float>::value) {
    Cast(dst, src, RoundMode::CAST_RINT, size);
  } else if constexpr (std::is_same<I, half>::value and std::is_same<O, int16_t>::value) {
    Cast(dst, src, RoundMode::CAST_RINT, size);
  } else {
    Cast(dst, src, RoundMode::CAST_NONE, size);
  }
}

template <typename T>
inline __aicore__ void DoSelect(const AscendC::LocalTensor<T> &dst,
                                const AscendC::LocalTensor<uint8_t> &sel_mask,  // condition
                                const AscendC::LocalTensor<float> &src0,        // then
                                const AscendC::LocalTensor<float> &src1,        // else
                                const DoSelectParams &params) {
  uint32_t offset = params.calc_size;
  // Cast sel_mask.u8 -> sel_mask.half
  SafeCast(params.mask_cast_buf[0], sel_mask[offset], params.do_size);
  // Compare sel_mask.half -> mask.u8 by bit
  uint32_t cmp_size = (params.do_size + ONE_REPEAT_HALF_SIZE - 1) / ONE_REPEAT_HALF_SIZE * ONE_REPEAT_HALF_SIZE;
  LocalTensor<uint8_t> mask_tmp_bit_buf = params.mask_cast_buf.ReinterpretCast<uint8_t>();  // reuse this buffer
  CompareScalar(mask_tmp_bit_buf, params.mask_cast_buf[0], (half)1.0, CMPMODE::EQ, cmp_size);

  // Do Select
  if constexpr (std::is_same<T, float>::value) {
    Select(dst[offset],                               // output
           mask_tmp_bit_buf,                          // condition
           src0[params.src0_select_offset * offset],  // then
           src1[params.src1_select_offset * offset],  // else
           SELMODE::VSEL_TENSOR_TENSOR_MODE,          // condition is tensor
           params.mask,                               // once repeat number count
           params.repeat_times,                       // repeat times
           params.rpt_params);
  } else {
    Select(params.sel_res_buf[0],                     // output
           mask_tmp_bit_buf,                          // condition
           src0[params.src0_select_offset * offset],  // then
           src1[params.src1_select_offset * offset],  // else
           SELMODE::VSEL_TENSOR_TENSOR_MODE,          // condition is tensor
           params.mask,                               // once repeat number count
           params.repeat_times,                       // repeat times
           params.rpt_params);

    SafeCast(dst[offset], params.sel_res_buf[0], params.do_size);
  }
}

template <typename O, typename I0, typename I1>
inline __aicore__ void CastBeforeSelect(const AscendC::LocalTensor<O> &dst,
                                        const AscendC::LocalTensor<uint8_t> &sel_mask,  // condition
                                        const AscendC::LocalTensor<I0> &src0,           // then
                                        const AscendC::LocalTensor<I1> &src1,           // else
                                        DoSelectParams &params) {
  constexpr bool src0_is_float = std::is_same<I0, float>::value;
  constexpr bool src1_is_float = std::is_same<I1, float>::value;
  uint32_t offset = params.calc_size;
  if constexpr (src0_is_float && src1_is_float) {
    DoSelect(dst, sel_mask, src0, src1, params);
  } else if constexpr (!src0_is_float && src1_is_float) {
    // src0 is not float, and src1 is scalar
    SafeCast(params.sel_res_buf[0], src0[offset], params.do_size);
    params.src0_select_offset = 0;
    DoSelect(dst, sel_mask, params.sel_res_buf, src1, params);
  } else if constexpr (src0_is_float && !src1_is_float) {
    // src1 is not float, and scr0 is scalar
    SafeCast(params.sel_res_buf[0], src1[offset], params.do_size);
    params.src1_select_offset = 0;
    DoSelect(dst, sel_mask, src0, params.sel_res_buf, params);
  } else {
    SafeCast(params.src0_cast_buf[0], src0[offset], params.do_size);
    SafeCast(params.sel_res_buf[0], src1[offset], params.do_size);
    params.src0_select_offset = 0;
    params.src1_select_offset = 0;
    DoSelect(dst, sel_mask, params.src0_cast_buf, params.sel_res_buf, params);
  }
}

inline __aicore__ void CastAndSelect(const AscendC::LocalTensor<int64_t> &dst,
                                     const AscendC::LocalTensor<uint8_t> &sel_mask,  // condition
                                     const AscendC::LocalTensor<int64_t> &src0,      // then
                                     const AscendC::LocalTensor<int64_t> &src1,      // else
                                     DoSelectParams &params) {
  uint32_t offset = params.calc_size;
  uint32_t mask_offset = offset >> 1;
  uint32_t do_size = params.do_size;
  uint32_t mask_do_size = do_size >> 1;

  const AscendC::LocalTensor<float> dst_buf = dst.ReinterpretCast<float>();
  const AscendC::LocalTensor<float> src0_buf = src0.ReinterpretCast<float>();
  const AscendC::LocalTensor<float> src1_buf = src1.ReinterpretCast<float>();

  // Step1: interleave mask
  // 1.1 Cast sel_mask.u8 -> sel_mask.half
  SafeCast(params.mask_cast_buf[0], sel_mask[mask_offset], mask_do_size);
  // 1.2 Cast sel_mask.half -> sel_mask.16
  SafeCast(params.mask_shift_buf[0], params.mask_cast_buf[0], mask_do_size);
  // 1.3 Shift sel_mask.16 LEFT by 8 bits
  const AscendC::LocalTensor<int16_t> mask_cast_buf_reuse_buf = params.mask_cast_buf.ReinterpretCast<int16_t>();
  constexpr int16_t shift_left_size = 8;
  ShiftLeft(mask_cast_buf_reuse_buf, params.mask_shift_buf[0], shift_left_size, static_cast<int32_t>(mask_do_size));
  // 1.4 Or low 8bits and high 8bits
  Or(params.mask_shift_buf[0], mask_cast_buf_reuse_buf, params.mask_shift_buf[0], static_cast<int32_t>(mask_do_size));

  // Step2: Convert mask.u8 to mask.bit
  const AscendC::LocalTensor<uint8_t> mask_shift_cast_buf = params.mask_shift_buf.ReinterpretCast<uint8_t>();
  AscendC::LocalTensor<half> mask_cast_buf_resize_buf = params.mask_cast_buf.ReinterpretCast<half>();
  mask_cast_buf_resize_buf.SetSize(params.mask_cast_buf.GetSize() + params.mask_shift_buf.GetSize());
  SafeCast(mask_cast_buf_resize_buf, mask_shift_cast_buf, do_size);
  uint32_t cmp_size = (do_size + ONE_REPEAT_HALF_SIZE - 1) / ONE_REPEAT_HALF_SIZE * ONE_REPEAT_HALF_SIZE;
  LocalTensor<uint8_t> mask_tmp_bit_buf = mask_cast_buf_resize_buf.ReinterpretCast<uint8_t>();  // reuse this buffer
  CompareScalar(mask_tmp_bit_buf, mask_cast_buf_resize_buf, (half)1.0, CMPMODE::EQ, cmp_size);

  // Step3: DO Select
  Select(dst_buf[offset],                               // output
         mask_tmp_bit_buf,                              // condition
         src0_buf[params.src0_select_offset * offset],  // then
         src1_buf[params.src1_select_offset * offset],  // else
         SELMODE::VSEL_TENSOR_TENSOR_MODE,              // condition is tensor
         params.mask,                                   // once repeat number count
         params.repeat_times,                           // repeat times
         params.rpt_params);
}

template <typename T, typename T1, typename T2>
inline __aicore__ void WhereBase(const AscendC::LocalTensor<T> &dst,
                                 const AscendC::LocalTensor<uint8_t> &sel_mask,  // condition
                                 const AscendC::LocalTensor<T1> &src0,           // then
                                 const AscendC::LocalTensor<T2> &src1,           // else
                                 const uint32_t size,                            // elements num
                                 DoSelectParams &params) {
  constexpr uint32_t ONE_RPT_SIZE = KernelUtils::RptSize<float>();

  params.rpt_params.blockNumber = KernelUtils::BlkNum<float>(size);  // real block number
  params.mask = ONE_RPT_SIZE;
  uint32_t max_buf_size = params.sel_res_buf.GetSize() / sizeof(float);
  uint32_t max_buf_rpt_num = KernelUtils::RptNum<float>(max_buf_size);
  uint32_t max_do_rpt_num = KernelUtils::Min(MAX_REPEAT_TIME, max_buf_rpt_num);
  uint32_t max_do_size = max_do_rpt_num * ONE_RPT_SIZE;
  // do max repeat once time
  if (max_do_size <= size) {
    params.repeat_times = max_do_rpt_num;
    params.do_size = max_do_size;
    for (; params.calc_size + params.do_size < size; params.calc_size += params.do_size) {
      CastBeforeSelect(dst, sel_mask, src0, src1, params);
    }
  }
  // do left repeats
  if (params.calc_size + ONE_RPT_SIZE <= size) {
    uint32_t left_rpt_num = KernelUtils::RptNum<float>(size - params.calc_size);
    params.repeat_times = left_rpt_num;
    params.do_size = left_rpt_num * KernelUtils::RptSize<float>();
    CastBeforeSelect(dst, sel_mask, src0, src1, params);
    params.calc_size += params.do_size;
  }
  // do left blocks
  if (params.calc_size < size) {
    constexpr uint32_t redundant_size = KernelUtils::BlkSize<float>() - 1;
    uint32_t left_size = size - params.calc_size;
    params.rpt_params.blockNumber = KernelUtils::BlkNum<float>(left_size + redundant_size);
    params.do_size = left_size;
    params.mask = params.rpt_params.blockNumber * KernelUtils::BlkSize<float>();
    params.repeat_times = 1;
    CastBeforeSelect(dst, sel_mask, src0, src1, params);
  }
}

inline __aicore__ void WhereBase(const AscendC::LocalTensor<int64_t> &dst,
                                 const AscendC::LocalTensor<uint8_t> &sel_mask,  // condition
                                 const AscendC::LocalTensor<int64_t> &src0,      // then
                                 const AscendC::LocalTensor<int64_t> &src1,      // else
                                 const uint32_t size_,                           // elements num
                                 DoSelectParams &params) {
  const uint32_t mask_size = size_;
  const uint32_t data_size = size_ << 1;

  constexpr uint32_t ONE_RPT_SIZE = KernelUtils::RptSize<float>();

  params.rpt_params.blockNumber = KernelUtils::BlkNum<float>(data_size);
  params.mask = ONE_RPT_SIZE;
  // 临时空间一次能计算多少个repeat
  uint32_t max_buf_rpt_num = KernelUtils::RptNum<float>(params.mask_shift_buf.GetSize());
  // 实际一次能计算的repeat数(因为指令限制255)
  uint32_t max_do_rpt_num = KernelUtils::Min(MAX_REPEAT_TIME, max_buf_rpt_num);
  // 实际一次计算个数据个数
  uint32_t max_do_size = max_do_rpt_num * ONE_RPT_SIZE;

  // 按照一次指令最大repeat=255计算
  if (max_do_size <= data_size) {
    params.repeat_times = max_do_rpt_num;
    params.do_size = max_do_size;
    for (; params.calc_size + params.do_size < data_size; params.calc_size += params.do_size) {
      CastAndSelect(dst, sel_mask, src0, src1, params);
    }
  }
  // 剩余数据不足255个repeat后，使用一条指令计算完整的repeat数
  if (params.calc_size + ONE_RPT_SIZE <= data_size) {
    uint32_t left_rpt_num = KernelUtils::RptNum<float>(data_size - params.calc_size);
    params.repeat_times = left_rpt_num;
    params.do_size = left_rpt_num * KernelUtils::RptSize<float>();
    CastAndSelect(dst, sel_mask, src0, src1, params);
    params.calc_size += params.do_size;
  }
  // 剩余数据不足1个repeat后，使用一条指令计算完剩余的block数
  if (params.calc_size < data_size) {
    constexpr uint32_t redundant_size = KernelUtils::BlkSize<float>() - 1;
    uint32_t left_size = data_size - params.calc_size;
    params.rpt_params.blockNumber = KernelUtils::BlkNum<float>(left_size + redundant_size);
    params.do_size = left_size;
    params.mask = params.rpt_params.blockNumber * KernelUtils::BlkSize<float>();
    params.repeat_times = 1;
    CastAndSelect(dst, sel_mask, src0, src1, params);
  }
}

inline __aicore__ void WherePartition2Buffer(AscendC::LocalTensor<uint8_t> &tmp_buf, DoSelectParams &params,
                                             uint32_t used_size) {
  // split the rest tmp_buf to 2 parts:
  //   1. mask_cast_buf: for cast mask.u8 -> mask.half
  //   2. sel_res_buf: for select sel_mask.u8 -> dst.float
  constexpr uint32_t partition_number = sizeof(half) + sizeof(float);

  uint32_t rest_buf_byte_size = tmp_buf.GetSize() - used_size;
  uint32_t each_part_buf_byte_size =
      rest_buf_byte_size / partition_number / ONE_REPEAT_BYTE_SIZE * ONE_REPEAT_BYTE_SIZE;

  params.mask_cast_buf = tmp_buf[used_size].template ReinterpretCast<half>();
  params.mask_cast_buf.SetSize(each_part_buf_byte_size);

  params.sel_res_buf = tmp_buf[used_size + each_part_buf_byte_size].template ReinterpretCast<float>();
  params.sel_res_buf.SetSize(each_part_buf_byte_size);
}

inline __aicore__ void WherePartition3Buffer(AscendC::LocalTensor<uint8_t> &tmp_buf, DoSelectParams &params,
                                             uint32_t used_size) {
  // split the rest tmp_buf to 3 parts:
  //   1. mask_cast_buf: for cast mask.u8 -> mask.half
  //   2. sel_res_buf: for select sel_mask.u8 -> dst.float
  //   3. src0_cast_buf: for cast src0.other -> src0.float
  constexpr uint32_t partition_number = sizeof(half) + sizeof(float) + sizeof(float);

  uint32_t rest_buf_byte_size = tmp_buf.GetSize() - used_size;
  uint32_t each_part_buf_byte_size =
      rest_buf_byte_size / partition_number / ONE_REPEAT_BYTE_SIZE * ONE_REPEAT_BYTE_SIZE;

  params.mask_cast_buf = tmp_buf[used_size].template ReinterpretCast<half>();
  params.mask_cast_buf.SetSize(each_part_buf_byte_size);

  uint32_t offset = used_size + each_part_buf_byte_size;

  params.sel_res_buf = tmp_buf[offset].template ReinterpretCast<float>();
  params.sel_res_buf.SetSize(each_part_buf_byte_size);

  offset += each_part_buf_byte_size;
  params.src0_cast_buf = tmp_buf[offset].template ReinterpretCast<float>();
  params.src0_cast_buf.SetSize(each_part_buf_byte_size);
}

inline __aicore__ void WherePartitionBufferInt64(AscendC::LocalTensor<uint8_t> &tmp_buf, DoSelectParams &params,
                                                 uint32_t used_size) {
  // int64 select 用 select_pool 自己做切分：mask_cast_buf / mask_shift_buf 保持为两段互不重叠的临时空间，
  // 供不走 stride（WhereBase）的 int64 分支继续使用。select_pool 复用同一区间供乘法选择链使用。
  // 注意 SetSize 的入参单位是该视图类型的元素数，不是字节数：mask 两段按 2 字节宽度换算，
  // 否则容量被按 2 倍需求校验，used_size > 0 的场景（如 MLIR e2e select 前序算子已占用 tmp）
  // 会触发 ICPU SetSize 越界（load_mask_select_i64_store 1x171x306 崩溃）。
  const uint32_t rest_buf_byte_size = (tmp_buf.GetSize() - used_size) / ONE_REPEAT_BYTE_SIZE * ONE_REPEAT_BYTE_SIZE;
  const uint32_t each_part_buf_byte_size = rest_buf_byte_size / 2;

  params.mask_cast_buf = tmp_buf[used_size].template ReinterpretCast<half>();
  params.mask_cast_buf.SetSize(each_part_buf_byte_size / sizeof(half));

  params.mask_shift_buf = tmp_buf[used_size + each_part_buf_byte_size].template ReinterpretCast<int16_t>();
  params.mask_shift_buf.SetSize(each_part_buf_byte_size / sizeof(int16_t));

  params.select_pool = tmp_buf[used_size];
  params.select_pool.SetSize(each_part_buf_byte_size * 2);
}

/**
 * 场景1： src0和src1都是标量，输出Shape与mask相同
 */
template <typename T>
inline __aicore__ void Where(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<uint8_t> &mask, T src0,
                             T src1, const uint32_t size, AscendC::LocalTensor<uint8_t> &tmp_buf) {
  LocalTensor<float> src0_buf = tmp_buf[0].template ReinterpretCast<float>();
  Duplicate(src0_buf[0], (float)src0, KernelUtils::BlkSize<float>());

  LocalTensor<float> src1_buf = tmp_buf[ONE_BLK_SIZE].template ReinterpretCast<float>();
  Duplicate(src1_buf[0], (float)src1, KernelUtils::BlkSize<float>());

  DoSelectParams params;
  WherePartition2Buffer(tmp_buf, params, ONE_BLK_SIZE * 2);

  // Prepare binary repeat params
  params.rpt_params.src0BlkStride = 0;
  params.rpt_params.src0RepStride = 0;
  params.rpt_params.src1BlkStride = 0;
  params.rpt_params.src1RepStride = 0;
  params.src0_select_offset = 0;
  params.src1_select_offset = 0;

  WhereBase(dst, mask, src0_buf, src1_buf, size, params);
}

/**
 * 场景2： src0是标量，src1及输出的Shape与mask相同
 */
template <typename T>
inline __aicore__ void Where(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<uint8_t> &mask, T src0,
                             const AscendC::LocalTensor<T> &src1, const uint32_t size,
                             AscendC::LocalTensor<uint8_t> &tmp_buf) {
  LocalTensor<float> src0_buf = tmp_buf[0].template ReinterpretCast<float>();
  Duplicate(src0_buf[0], (float)src0, KernelUtils::BlkSize<float>());

  DoSelectParams params;
  WherePartition2Buffer(tmp_buf, params, ONE_BLK_SIZE);

  params.rpt_params.src0BlkStride = 0;
  params.rpt_params.src0RepStride = 0;
  params.src0_select_offset = 0;
  WhereBase(dst, mask, src0_buf, src1, size, params);
}

inline __aicore__ void Where(const AscendC::LocalTensor<int64_t> &dst, const AscendC::LocalTensor<uint8_t> &mask,
                             float src0, const AscendC::LocalTensor<int64_t> &src1, const uint32_t size,
                             AscendC::LocalTensor<uint8_t> &tmp_buf) {
  Where(dst, mask, static_cast<int64_t>(src0), src1, size, tmp_buf);
}

/**
 * 场景3： src1是标量，src0及输出的Shape与mask相同
 */
template <typename T>
inline __aicore__ void Where(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<uint8_t> &mask,
                             const AscendC::LocalTensor<T> &src0, T src1, const uint32_t size,
                             AscendC::LocalTensor<uint8_t> &tmp_buf) {
  LocalTensor<float> src1_buf = tmp_buf[0].template ReinterpretCast<float>();
  Duplicate(src1_buf[0], (float)src1, KernelUtils::BlkSize<float>());

  DoSelectParams params;
  WherePartition2Buffer(tmp_buf, params, ONE_BLK_SIZE);

  params.rpt_params.src1BlkStride = 0;
  params.rpt_params.src1RepStride = 0;
  params.src1_select_offset = 0;
  WhereBase(dst, mask, src0, src1_buf, size, params);
}

/**
 * 场景4: src0和src1都不是标量，且Shape均与mask相同，且不需要广播。
 */
template <typename T>
inline __aicore__ void Where(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<uint8_t> &mask,
                             const AscendC::LocalTensor<T> &src0, const AscendC::LocalTensor<T> &src1,
                             const uint32_t size, AscendC::LocalTensor<uint8_t> &tmp_buf) {
  DoSelectParams params;
  WherePartition3Buffer(tmp_buf, params, 0);

  WhereBase(dst, mask, src0, src1, size, params);
}

/**
 * 场景5： int64+场景1：src0和src1都是标量，输出Shape与mask相同
 */
inline __aicore__ void Where(const AscendC::LocalTensor<int64_t> &dst, const AscendC::LocalTensor<uint8_t> &mask,
                             int64_t src0, int64_t src1, const uint32_t size, AscendC::LocalTensor<uint8_t> &tmp_buf) {
  LocalTensor<int64_t> src0_buf = tmp_buf[0].template ReinterpretCast<int64_t>();
  LocalTensor<uint8_t> left_tmp_buf = tmp_buf[ONE_BLK_SIZE].template ReinterpretCast<uint8_t>();
  Duplicate(src0_buf[0], src0, KernelUtils::BlkSize<int64_t>(), left_tmp_buf);

  LocalTensor<int64_t> src1_buf = left_tmp_buf[0].template ReinterpretCast<int64_t>();
  left_tmp_buf = left_tmp_buf[ONE_BLK_SIZE].template ReinterpretCast<uint8_t>();
  Duplicate(src1_buf[0], src1, KernelUtils::BlkSize<int64_t>(), left_tmp_buf);

  DoSelectParams params;
  WherePartitionBufferInt64(tmp_buf, params, ONE_BLK_SIZE * 2);

  // Prepare binary repeat params
  params.rpt_params.src0BlkStride = 0;
  params.rpt_params.src0RepStride = 0;
  params.rpt_params.src1BlkStride = 0;
  params.rpt_params.src1RepStride = 0;
  params.src0_select_offset = 0;
  params.src1_select_offset = 0;

  WhereBase(dst, mask, src0_buf, src1_buf, size, params);
}

/**
 * 场景6： int64+场景2： src0是标量，src1及输出的Shape与mask相同
 */
inline __aicore__ void Where(const AscendC::LocalTensor<int64_t> &dst, const AscendC::LocalTensor<uint8_t> &mask,
                             int64_t src0, const AscendC::LocalTensor<int64_t> &src1, const uint32_t size,
                             AscendC::LocalTensor<uint8_t> &tmp_buf) {
  LocalTensor<int64_t> src0_buf = tmp_buf[0].template ReinterpretCast<int64_t>();

  LocalTensor<uint8_t> left_tmp_buf = tmp_buf[ONE_BLK_SIZE].template ReinterpretCast<uint8_t>();
  Duplicate(src0_buf[0], src0, KernelUtils::BlkSize<int64_t>(), left_tmp_buf);

  DoSelectParams params;
  WherePartitionBufferInt64(tmp_buf, params, ONE_BLK_SIZE);

  params.rpt_params.src0BlkStride = 0;
  params.rpt_params.src0RepStride = 0;
  params.src0_select_offset = 0;
  WhereBase(dst, mask, src0_buf, src1, size, params);
}

/**
 * 场景7： int64+场景3： src1是标量，src0及输出的Shape与mask相同
 */
inline __aicore__ void Where(const AscendC::LocalTensor<int64_t> &dst, const AscendC::LocalTensor<uint8_t> &mask,
                             const AscendC::LocalTensor<int64_t> &src0, int64_t src1, const uint32_t size,
                             AscendC::LocalTensor<uint8_t> &tmp_buf) {
  LocalTensor<int64_t> src1_buf = tmp_buf[0].template ReinterpretCast<int64_t>();

  LocalTensor<uint8_t> left_tmp_buf = tmp_buf[ONE_BLK_SIZE].template ReinterpretCast<uint8_t>();
  Duplicate(src1_buf[0], src1, KernelUtils::BlkSize<int64_t>(), left_tmp_buf);

  DoSelectParams params;
  WherePartitionBufferInt64(tmp_buf, params, ONE_BLK_SIZE);

  params.rpt_params.src1BlkStride = 0;
  params.rpt_params.src1RepStride = 0;
  params.src1_select_offset = 0;
  WhereBase(dst, mask, src0, src1_buf, size, params);
}

/**
 * 场景8: int64+场景4：src0和src1都不是标量，且Shape均与mask相同，且不需要广播。
 */
inline __aicore__ void Where(const AscendC::LocalTensor<int64_t> &dst, const AscendC::LocalTensor<uint8_t> &mask,
                             const AscendC::LocalTensor<int64_t> &src0, const AscendC::LocalTensor<int64_t> &src1,
                             const uint32_t size, AscendC::LocalTensor<uint8_t> &tmp_buf) {
  DoSelectParams params;
  WherePartitionBufferInt64(tmp_buf, params, 0);
  WhereBase(dst, mask, src0, src1, size, params);
}
/**
 * Normal 模式：两根轴
 */
template <typename O, typename I, bool isSlice2Buf>
inline __aicore__ void SafeCastNormal(const AscendC::LocalTensor<O> &dst, const AscendC::LocalTensor<I> &src,
                                      const uint64_t mask, const uint8_t repeat_times, const uint32_t stride) {
  uint8_t dstRepStride = 8;
  uint8_t srcRepStride = 8;
  if constexpr (isSlice2Buf) {
    dstRepStride = static_cast<uint8_t>(mask * sizeof(O) / ONE_BLK_SIZE);
    srcRepStride = static_cast<uint8_t>(stride * sizeof(I) / ONE_BLK_SIZE);
  } else {
    dstRepStride = static_cast<uint8_t>(stride * sizeof(O) / ONE_BLK_SIZE);
    srcRepStride = static_cast<uint8_t>(mask * sizeof(I) / ONE_BLK_SIZE);
  }
  // float -> int64/int32/int16 :: CAST_RINT
  if constexpr (std::is_same<I, float>::value and (std::is_same<O, int64_t>::value || std::is_same<O, int32_t>::value ||
                                                   std::is_same<O, int16_t>::value)) {
    Cast(dst, src, RoundMode::CAST_RINT, mask, repeat_times, {1, 1, dstRepStride, srcRepStride});
    // int64 -> float :: CAST_RINT
  } else if constexpr (std::is_same<I, int64_t>::value and std::is_same<O, float>::value) {
    Cast(dst, src, RoundMode::CAST_RINT, mask, repeat_times, {1, 1, dstRepStride, srcRepStride});
  } else if constexpr (std::is_same<I, half>::value and std::is_same<O, int16_t>::value) {
    Cast(dst, src, RoundMode::CAST_RINT, mask, repeat_times, {1, 1, dstRepStride, srcRepStride});
  } else {
    Cast(dst, src, RoundMode::CAST_NONE, mask, repeat_times, {1, 1, dstRepStride, srcRepStride});
  }
}

inline __aicore__ void MaskSafeCastNormal(const AscendC::LocalTensor<half> &dst,
                                          const AscendC::LocalTensor<uint8_t> &src, const uint64_t mask,
                                          const uint8_t repeat_times, const uint32_t stride) {
  uint8_t dstRepStride = 8;
  uint8_t srcRepStride = 8;
  // uint8 -> half :: CAST_NONE
  // AscendC::Select : input is float(32bit), sel mask rptstride = 64(bit)
  dstRepStride = static_cast<uint8_t>(KernelUtils::RptSize<float>() * sizeof(half) / ONE_BLK_SIZE);
  srcRepStride = static_cast<uint8_t>(stride * sizeof(uint8_t) / ONE_BLK_SIZE);
  Cast(dst, src, RoundMode::CAST_NONE, mask, repeat_times, {1, 1, dstRepStride, srcRepStride});
}

inline __aicore__ void MaskSafeCastInt64Normal(const AscendC::LocalTensor<half> &dst,
                                               const AscendC::LocalTensor<uint8_t> &src, const uint64_t mask,
                                               const uint8_t repeat_times, const uint32_t stride) {
  uint8_t dstRepStride = 8;
  uint8_t srcRepStride = 8;
  // uint8 -> half :: CAST_NONE
  dstRepStride = static_cast<uint8_t>(KernelUtils::RptSize<float>() / 2 * sizeof(half) / ONE_BLK_SIZE);
  srcRepStride = static_cast<uint8_t>(stride * sizeof(uint8_t) / ONE_BLK_SIZE);
  Cast(dst, src, RoundMode::CAST_NONE, mask, repeat_times, {1, 1, dstRepStride, srcRepStride});
}

template <typename T>
inline __aicore__ void DoSelectNormal(const AscendC::LocalTensor<T> &dst,
                                      const AscendC::LocalTensor<uint8_t> &sel_mask,  // condition
                                      const AscendC::LocalTensor<float> &src0,        // then
                                      const AscendC::LocalTensor<float> &src1,        // else
                                      DoSelectParams &params) {
  // Cast sel_mask.u8 -> sel_mask.half
  MaskSafeCastNormal(params.mask_cast_buf[0], sel_mask[params.mask_offset], params.mask, params.repeat_times,
                     params.mask_stride);
  // Compare sel_mask.half -> mask.u8 by bit
  uint32_t cmp_size = (KernelUtils::RptSize<float>() * params.repeat_times + ONE_REPEAT_HALF_SIZE - 1) /
                      ONE_REPEAT_HALF_SIZE * ONE_REPEAT_HALF_SIZE;
  LocalTensor<uint8_t> mask_tmp_bit_buf = params.mask_cast_buf.ReinterpretCast<uint8_t>();  // reuse this buffer
  CompareScalar(mask_tmp_bit_buf, params.mask_cast_buf[0], (half)1.0, CMPMODE::EQ, cmp_size);
  // Do Select
  if constexpr (std::is_same<T, float>::value) {
    Select(dst[params.dst_offset],                                // output
           mask_tmp_bit_buf,                                      // condition
           src0[params.src0_select_offset * params.src0_offset],  // then
           src1[params.src1_select_offset * params.src1_offset],  // else
           SELMODE::VSEL_TENSOR_TENSOR_MODE,                      // condition is tensor
           params.mask,                                           // once repeat number count
           params.repeat_times,                                   // repeat times
           params.rpt_params);
  } else {
    params.rpt_params.dstRepStride = params.mask * sizeof(float) / ONE_BLK_SIZE;
    Select(params.sel_res_buf[0],                                 // output
           mask_tmp_bit_buf,                                      // condition
           src0[params.src0_select_offset * params.src0_offset],  // then
           src1[params.src1_select_offset * params.src1_offset],  // else
           SELMODE::VSEL_TENSOR_TENSOR_MODE,                      // condition is tensor
           params.mask,                                           // once repeat number count
           params.repeat_times,                                   // repeat times
           params.rpt_params);
    SafeCastNormal<T, float, false>(dst[params.dst_offset], params.sel_res_buf[0], params.mask, params.repeat_times,
                                    params.output_stride);
  }
}

template <typename O, typename I0, typename I1>
inline __aicore__ void CastSelectNormal(const AscendC::LocalTensor<O> &dst,
                                        const AscendC::LocalTensor<uint8_t> &sel_mask,  // condition
                                        const AscendC::LocalTensor<I0> &src0,           // then
                                        const AscendC::LocalTensor<I1> &src1,           // else
                                        DoSelectParams &params) {
  constexpr bool src0_is_float = std::is_same<I0, float>::value;
  constexpr bool src1_is_float = std::is_same<I1, float>::value;
  if constexpr (src0_is_float && src1_is_float) {
    DoSelectNormal(dst, sel_mask, src0, src1, params);
  } else if constexpr (!src0_is_float && src1_is_float) {
    // src0 is not float, and src1 is float
    SafeCastNormal<float, I0, true>(params.sel_res_buf[0], src0[params.src0_offset], params.mask, params.repeat_times,
                                    params.input0_stride);
    params.src0_select_offset = 0;
    params.rpt_params.src0RepStride = params.mask * sizeof(float) / ONE_BLK_SIZE;
    DoSelectNormal(dst, sel_mask, params.sel_res_buf, src1, params);
  } else if constexpr (src0_is_float && !src1_is_float) {
    // src1 is not float, and scr0 is float
    SafeCastNormal<float, I1, true>(params.sel_res_buf[0], src1[params.src1_offset], params.mask, params.repeat_times,
                                    params.input1_stride);
    params.src1_select_offset = 0;
    params.rpt_params.src1RepStride = params.mask * sizeof(float) / ONE_BLK_SIZE;
    DoSelectNormal(dst, sel_mask, src0, params.sel_res_buf, params);
  } else {
    SafeCastNormal<float, I0, true>(params.src0_cast_buf[0], src0[params.src0_offset], params.mask, params.repeat_times,
                                    params.input0_stride);
    SafeCastNormal<float, I1, true>(params.sel_res_buf[0], src1[params.src1_offset], params.mask, params.repeat_times,
                                    params.input1_stride);
    params.src0_select_offset = 0;
    params.src1_select_offset = 0;
    params.rpt_params.src0RepStride = params.mask * sizeof(float) / ONE_BLK_SIZE;
    params.rpt_params.src1RepStride = params.mask * sizeof(float) / ONE_BLK_SIZE;
    DoSelectNormal(dst, sel_mask, params.src0_cast_buf, params.sel_res_buf, params);
  }
}

// int64 select 用到的临时空间按 32 B 对齐
inline __aicore__ uint32_t Int64SelectAlignBytes(uint32_t bytes) {
  return (bytes + ONE_BLK_SIZE - 1U) / ONE_BLK_SIZE * ONE_BLK_SIZE;
}

// int64 select 每个 repeat（一行）需要的临时空间：掩码交织用的 half/int16、按 lane 展开的 half、
// 0/1 的 float/int32 掩码，以及选择用的两份 int32 暂存。
inline __aicore__ uint32_t Int64SelectRowBytes(uint32_t lane_cnt) {
  const uint32_t mask_bytes = lane_cnt / 2;
  const uint32_t lane_bytes = Int64SelectAlignBytes(lane_cnt * sizeof(int32_t));
  return Int64SelectAlignBytes(mask_bytes * sizeof(half)) + Int64SelectAlignBytes(mask_bytes * sizeof(int16_t)) +
         Int64SelectAlignBytes(lane_cnt * sizeof(half)) + 4U * lane_bytes;
}

inline __aicore__ void CastSelectInt64Normal(const AscendC::LocalTensor<int64_t> &dst,
                                             const AscendC::LocalTensor<uint8_t> &sel_mask,  // condition
                                             const AscendC::LocalTensor<int64_t> &src0,      // then
                                             const AscendC::LocalTensor<int64_t> &src1,      // else
                                             DoSelectParams &params) {
  // int64 一个元素占两个 32bit lane。Select 的 bit 掩码在“每 repeat 的 lane 数不是 64 的整数倍”时，
  // 与 Cast/CompareScalar 生成的掩码排布无法对齐（0002 精度问题），因此这里改用 0/1 向量掩码做选择：
  //   dst = src0 * mask + src1 * (1 - mask)
  // 每个 32bit lane 上 mask 为 1 或 0，乘以 1/0 只是保留或清零位模式，不会改变 int64 的数值语义。
  const AscendC::LocalTensor<int32_t> dst_buf = dst.ReinterpretCast<int32_t>();
  const AscendC::LocalTensor<int32_t> src0_buf = src0.ReinterpretCast<int32_t>();
  const AscendC::LocalTensor<int32_t> src1_buf = src1.ReinterpretCast<int32_t>();

  const uint32_t lane_cnt = static_cast<uint32_t>(params.mask);  // 一个 repeat 处理的 32bit lane 数
  const uint32_t mask_bytes = lane_cnt / 2;                      // 每个 int64 元素对应一个掩码字节
  constexpr int16_t shift_left_size = 8;

  const uint32_t mask_half_bytes = Int64SelectAlignBytes(mask_bytes * sizeof(half));
  const uint32_t mask_int16_bytes = Int64SelectAlignBytes(mask_bytes * sizeof(int16_t));
  const uint32_t lane_half_bytes = Int64SelectAlignBytes(lane_cnt * sizeof(half));
  const uint32_t lane_bytes = Int64SelectAlignBytes(lane_cnt * sizeof(int32_t));
  const uint8_t mask_half_stride = static_cast<uint8_t>(mask_half_bytes / ONE_BLK_SIZE);
  const uint8_t mask_int16_stride = static_cast<uint8_t>(mask_int16_bytes / ONE_BLK_SIZE);
  const uint8_t lane_half_stride = static_cast<uint8_t>(lane_half_bytes / ONE_BLK_SIZE);
  const uint8_t lane_stride = static_cast<uint8_t>(lane_bytes / ONE_BLK_SIZE);
  const uint8_t mask_stride = static_cast<uint8_t>(params.mask_stride / ONE_BLK_SIZE);
  const uint8_t dst_stride = params.rpt_params.dstRepStride;
  const uint8_t src0_blk_stride = params.rpt_params.src0BlkStride;
  const uint8_t src0_rpt_stride = params.rpt_params.src0RepStride;
  const uint8_t src1_blk_stride = params.rpt_params.src1BlkStride;
  const uint8_t src1_rpt_stride = params.rpt_params.src1RepStride;

  // 临时空间按 repeat 分段：相邻段之间相差 repeat_times 个单位，避免不同段互相踩踏
  AscendC::LocalTensor<uint8_t> pool = params.select_pool;
  const uint32_t mask_area_bytes = mask_half_bytes + mask_int16_bytes;
  const uint32_t lane_half_offset = mask_area_bytes;
  const uint32_t lane_float_offset = lane_half_offset + lane_half_bytes;
  const uint32_t lane_mask_offset = lane_float_offset + lane_bytes;
  const uint32_t not_mask_offset = lane_mask_offset + lane_bytes;
  const uint32_t select_tmp_offset = not_mask_offset + lane_bytes;
  AscendC::LocalTensor<half> mask_half_buf = pool.template ReinterpretCast<half>();
  AscendC::LocalTensor<int16_t> mask_int16_buf =
      pool[mask_half_bytes * params.repeat_times].template ReinterpretCast<int16_t>();
  AscendC::LocalTensor<half> lane_half_buf =
      pool[lane_half_offset * params.repeat_times].template ReinterpretCast<half>();
  AscendC::LocalTensor<float> lane_float_buf =
      pool[lane_float_offset * params.repeat_times].template ReinterpretCast<float>();
  AscendC::LocalTensor<int32_t> lane_mask_buf =
      pool[lane_mask_offset * params.repeat_times].template ReinterpretCast<int32_t>();
  AscendC::LocalTensor<int32_t> not_mask_buf =
      pool[not_mask_offset * params.repeat_times].template ReinterpretCast<int32_t>();
  AscendC::LocalTensor<int32_t> select_tmp_buf =
      pool[select_tmp_offset * params.repeat_times].template ReinterpretCast<int32_t>();

  // Step1: 掩码字节(0/1) -> half(0.0/1.0)，每个 int64 元素一个值
  Cast(mask_half_buf[0], sel_mask[params.mask_offset], RoundMode::CAST_NONE, mask_bytes, params.repeat_times,
       {1, 1, mask_half_stride, mask_stride});
  // Step2: half -> int16，再把每个 int16 左移 8bit 后与自身按位或，得到 0x0101（一个元素两个掩码字节）
  Cast(mask_int16_buf[0], mask_half_buf[0], RoundMode::CAST_RINT, mask_bytes, params.repeat_times,
       {1, 1, mask_int16_stride, mask_half_stride});
  const AscendC::LocalTensor<int16_t> mask_half_i16_buf = pool.template ReinterpretCast<int16_t>();
  ShiftLeft(mask_half_i16_buf[0], mask_int16_buf[0], shift_left_size, mask_bytes, params.repeat_times,
            {1, 1, mask_int16_stride, mask_int16_stride});
  Or(mask_int16_buf[0], mask_half_i16_buf[0], mask_int16_buf[0], mask_bytes, params.repeat_times,
     {1, 1, 1, mask_int16_stride, mask_int16_stride, mask_int16_stride});
  // Step3: 按字节展开，使 int64 的两个 32bit lane 都拿到自己那一位掩码
  Cast(lane_half_buf[0], mask_int16_buf.template ReinterpretCast<uint8_t>()[0], RoundMode::CAST_NONE, lane_cnt,
       params.repeat_times, {1, 1, lane_half_stride, mask_int16_stride});
  // Step4: half(0.0/1.0) -> float -> int32，得到每个 lane 的 0/1 掩码
  Cast(lane_float_buf[0], lane_half_buf[0], RoundMode::CAST_NONE, lane_cnt, params.repeat_times,
       {1, 1, lane_stride, lane_half_stride});
  Cast(lane_mask_buf[0], lane_float_buf[0], RoundMode::CAST_RINT, lane_cnt, params.repeat_times,
       {1, 1, lane_stride, lane_stride});
  // Step5: 1 - mask
  Adds(not_mask_buf[0], lane_mask_buf[0], static_cast<int32_t>(-1), lane_cnt, params.repeat_times,
       {1, 1, lane_stride, lane_stride});
  Muls(not_mask_buf[0], not_mask_buf[0], static_cast<int32_t>(-1), lane_cnt, params.repeat_times,
       {1, 1, lane_stride, lane_stride});
  // Step6: dst = src0 * mask + src1 * (1 - mask)
  Mul(select_tmp_buf[0], src0_buf[params.src0_select_offset * params.src0_offset], lane_mask_buf[0], lane_cnt,
      params.repeat_times, {1, src0_blk_stride, 1, lane_stride, src0_rpt_stride, lane_stride});
  Mul(not_mask_buf[0], not_mask_buf[0], src1_buf[params.src1_select_offset * params.src1_offset], lane_cnt,
      params.repeat_times, {1, 1, src1_blk_stride, lane_stride, lane_stride, src1_rpt_stride});
  Add(dst_buf[params.dst_offset], select_tmp_buf[0], not_mask_buf[0], lane_cnt, params.repeat_times,
      {1, 1, 1, dst_stride, lane_stride, lane_stride});
}

template <typename T, typename T1, typename T2>
inline __aicore__ void WhereExtend(const AscendC::LocalTensor<T> &dst,
                                   const AscendC::LocalTensor<uint8_t> &sel_mask,  // condition
                                   const AscendC::LocalTensor<T1> &src0,           // then
                                   const AscendC::LocalTensor<T2> &src1,           // else
                                   const uint32_t first_axis, const uint32_t last_axis, DoSelectParams &params) {
  constexpr uint32_t ONE_RPT_SIZE = KernelUtils::RptSize<float>();
  // 1. 以256 B为单位长度，根据临时空间计算切分尾轴
  uint32_t element_extent = last_axis / ONE_RPT_SIZE;
  uint32_t element_reminder = last_axis - element_extent * ONE_RPT_SIZE;
  // 2. 根据临时空间，计算外抛实际一次能计算的repeat times
  uint32_t max_do_rpt_num =
      KernelUtils::Min(MAX_REPEAT_TIME, params.sel_res_buf.GetSize() / sizeof(float) / ONE_RPT_SIZE);
  // 确保根据params.src0_offset、params.src1_offset和params.dst_offset计算的地址，可以32字节对齐
  max_do_rpt_num = max_do_rpt_num / (sizeof(float) / sizeof(T)) * (sizeof(float) / sizeof(T));

  uint32_t repeat_throw_for_extent = first_axis / max_do_rpt_num;
  uint32_t repeat_reminder = first_axis - repeat_throw_for_extent * max_do_rpt_num;
  params.mask = ONE_RPT_SIZE;
  for (uint32_t outer_for = 0; outer_for < element_extent; outer_for++) {
    // 3. 按照一次指令最大repeat times计算
    params.mask_offset = outer_for * ONE_RPT_SIZE;
    params.dst_offset = outer_for * ONE_RPT_SIZE;
    params.src0_offset = outer_for * ONE_RPT_SIZE;
    params.src1_offset = outer_for * ONE_RPT_SIZE;
    params.repeat_times = max_do_rpt_num;
    for (uint32_t inner_for = 0; inner_for < repeat_throw_for_extent; inner_for++) {
      CastSelectNormal(dst, sel_mask, src0, src1, params);
      params.mask_offset += max_do_rpt_num * params.mask_stride;
      params.dst_offset += max_do_rpt_num * params.output_stride;
      params.src0_offset += max_do_rpt_num * params.input0_stride;
      params.src1_offset += max_do_rpt_num * params.input1_stride;
    }
    // 4. 首轴剩余数据不足最大repeat后，使用一条指令计算完整的repeat数
    if (repeat_reminder != 0) {
      params.repeat_times = repeat_reminder;
      CastSelectNormal(dst, sel_mask, src0, src1, params);
    }
  }
  if (element_reminder != 0) {
    // 5. 尾轴剩余数据不足1个repeat后，调整mask
    params.mask = KernelUtils::BlkNum<float>(element_reminder + KernelUtils::BlkSize<float>() - 1) *
                  KernelUtils::BlkSize<float>();
    // 按照一次指令最大repeat times计算
    params.mask_offset = element_extent * ONE_RPT_SIZE;
    params.dst_offset = element_extent * ONE_RPT_SIZE;
    params.src0_offset = element_extent * ONE_RPT_SIZE;
    params.src1_offset = element_extent * ONE_RPT_SIZE;
    params.repeat_times = max_do_rpt_num;
    for (uint32_t inner_for = 0; inner_for < repeat_throw_for_extent; inner_for++) {
      CastSelectNormal(dst, sel_mask, src0, src1, params);
      params.mask_offset += max_do_rpt_num * params.mask_stride;
      params.dst_offset += max_do_rpt_num * params.output_stride;
      params.src0_offset += max_do_rpt_num * params.input0_stride;
      params.src1_offset += max_do_rpt_num * params.input1_stride;
    }
    // 首轴剩余数据不足最大repeat后，使用一条指令计算完整的repeat数
    if (repeat_reminder != 0) {
      params.repeat_times = repeat_reminder;
      CastSelectNormal(dst, sel_mask, src0, src1, params);
    }
  }
}

inline __aicore__ void WhereExtend(const AscendC::LocalTensor<int64_t> &dst,
                                   const AscendC::LocalTensor<uint8_t> &sel_mask,  // condition
                                   const AscendC::LocalTensor<int64_t> &src0,      // then
                                   const AscendC::LocalTensor<int64_t> &src1,      // else
                                   const uint32_t first_axis, const uint32_t last_axis, DoSelectParams &params) {
  // 1. int64 一个元素占两个 32bit lane，按 64 lane（一个 repeat）为单位切分尾轴
  constexpr uint32_t ONE_RPT_SIZE = KernelUtils::RptSize<float>();
  const uint32_t size = last_axis << 1;
  uint32_t element_extent = size / ONE_RPT_SIZE;
  uint32_t element_reminder = size - element_extent * ONE_RPT_SIZE;
  // 2. 根据临时空间计算一次能处理的 repeat（行）数
  const uint32_t max_lane_cnt = KernelUtils::Min(static_cast<uint32_t>(ONE_RPT_SIZE), size);
  const uint32_t row_bytes = Int64SelectRowBytes(max_lane_cnt);
  uint32_t max_do_rpt_num = KernelUtils::Min(MAX_REPEAT_TIME, params.select_pool.GetSize() / row_bytes);
  if (max_do_rpt_num == 0U) {
    KERNEL_LOG(KERNEL_ERROR, "select tmp buf is too small for int64 where: buf=%u, one row needs %u.",
               params.select_pool.GetSize(), row_bytes);
    ASSERT(false);
    return;
  }

  uint32_t repeat_throw_for_extent = first_axis / max_do_rpt_num;
  uint32_t repeat_reminder = first_axis - repeat_throw_for_extent * max_do_rpt_num;
  params.mask = ONE_RPT_SIZE;
  for (uint32_t outer_for = 0; outer_for < element_extent; outer_for++) {
    // 3. 按照一次指令最大repeat times计算
    params.mask_offset = outer_for * ONE_RPT_SIZE / 2;
    params.dst_offset = outer_for * ONE_RPT_SIZE;
    params.src0_offset = outer_for * ONE_RPT_SIZE;
    params.src1_offset = outer_for * ONE_RPT_SIZE;
    params.repeat_times = max_do_rpt_num;
    for (uint32_t inner_for = 0; inner_for < repeat_throw_for_extent; inner_for++) {
      CastSelectInt64Normal(dst, sel_mask, src0, src1, params);
      params.mask_offset += max_do_rpt_num * params.mask_stride;
      params.dst_offset += max_do_rpt_num * params.output_stride * 2;
      params.src0_offset += max_do_rpt_num * params.input0_stride * 2;
      params.src1_offset += max_do_rpt_num * params.input1_stride * 2;
    }
    // 4. 首轴剩余数据不足最大repeat后，使用一条指令计算完整的repeat数
    if (repeat_reminder != 0) {
      params.repeat_times = repeat_reminder;
      CastSelectInt64Normal(dst, sel_mask, src0, src1, params);
    }
  }
  if (element_reminder != 0) {
    // 5. 尾轴剩余数据不足1个repeat：按剩余 lane 数计算，多算的 lane 会落在行尾 32 B 对齐的填充上
    params.mask = element_reminder;
    params.mask_offset = element_extent * ONE_RPT_SIZE / 2;
    params.dst_offset = element_extent * ONE_RPT_SIZE;
    params.src0_offset = element_extent * ONE_RPT_SIZE;
    params.src1_offset = element_extent * ONE_RPT_SIZE;
    params.repeat_times = max_do_rpt_num;
    for (uint32_t inner_for = 0; inner_for < repeat_throw_for_extent; inner_for++) {
      CastSelectInt64Normal(dst, sel_mask, src0, src1, params);
      params.mask_offset += max_do_rpt_num * params.mask_stride;
      params.dst_offset += max_do_rpt_num * params.output_stride * 2;
      params.src0_offset += max_do_rpt_num * params.input0_stride * 2;
      params.src1_offset += max_do_rpt_num * params.input1_stride * 2;
    }
    // 首轴剩余数据不足最大repeat后，使用一条指令计算完整的repeat数
    if (repeat_reminder != 0) {
      params.repeat_times = repeat_reminder;
      CastSelectInt64Normal(dst, sel_mask, src0, src1, params);
    }
  }
}

template <bool isBcastSrc0 = false, bool isBcastSrc1 = false, typename T, typename T1, typename T2>
inline __aicore__ void Where(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<uint8_t> &mask,
                             const AscendC::LocalTensor<T1> &src0, const AscendC::LocalTensor<T2> &src1,
                             const uint32_t first_axis, const uint32_t last_axis,
                             const uint32_t output_last_axis_stride, const uint32_t mask_last_axis_stride,
                             const uint32_t input0_last_axis_stride, const uint32_t input1_last_axis_stride,
                             AscendC::LocalTensor<uint8_t> &tmp_buf, const uint32_t used_size) {
  constexpr uint32_t MAX_VALID_STRIDE_BYTES = ONE_BLK_SIZE * 256 /*range of uint8*/;
  bool useWhereExtend = input0_last_axis_stride * sizeof(T1) < MAX_VALID_STRIDE_BYTES &&
                        input0_last_axis_stride * sizeof(float) < MAX_VALID_STRIDE_BYTES &&
                        input1_last_axis_stride * sizeof(T2) < MAX_VALID_STRIDE_BYTES &&
                        input1_last_axis_stride * sizeof(float) < MAX_VALID_STRIDE_BYTES &&
                        output_last_axis_stride * sizeof(T) < MAX_VALID_STRIDE_BYTES &&
                        output_last_axis_stride * sizeof(float) < MAX_VALID_STRIDE_BYTES &&
                        mask_last_axis_stride * sizeof(uint8_t) < MAX_VALID_STRIDE_BYTES;
  DoSelectParams params;
  constexpr bool is_int64_scene =
      std::is_same<T, int64_t>::value && std::is_same<T1, int64_t>::value && std::is_same<T2, int64_t>::value;
  if constexpr (isBcastSrc0 && isBcastSrc1) {
    // DoSelectParams params;
    if constexpr (is_int64_scene) { /* int64 */
      WherePartitionBufferInt64(tmp_buf, params, used_size);
    } else {
      WherePartition2Buffer(tmp_buf, params, used_size);
    }
    // Prepare binary repeat params
    params.rpt_params.src0BlkStride = 0;
    params.rpt_params.src1BlkStride = 0;
    params.rpt_params.src0RepStride = 0;
    params.rpt_params.src1RepStride = 0;
    params.src0_select_offset = 0;
    params.src1_select_offset = 0;
  } else if constexpr (isBcastSrc0) {
    // DoSelectParams params;
    if constexpr (is_int64_scene) { /* int64 */
      WherePartitionBufferInt64(tmp_buf, params, used_size);
    } else {
      WherePartition2Buffer(tmp_buf, params, used_size);
    }
    params.rpt_params.src0BlkStride = 0;
    params.rpt_params.src0RepStride = 0;
    params.src0_select_offset = 0;
    if (useWhereExtend) {
      params.rpt_params.src1BlkStride = 1;
      params.rpt_params.src1RepStride = input1_last_axis_stride * sizeof(T2) / ONE_BLK_SIZE;
    }
  } else if constexpr (isBcastSrc1) {
    // DoSelectParams params;
    if constexpr (is_int64_scene) { /* int64 */
      WherePartitionBufferInt64(tmp_buf, params, used_size);
    } else {
      WherePartition2Buffer(tmp_buf, params, used_size);
    }
    params.rpt_params.src1BlkStride = 0;
    params.rpt_params.src1RepStride = 0;
    params.src1_select_offset = 0;
    if (useWhereExtend) {
      params.rpt_params.src0BlkStride = 1;
      params.rpt_params.src0RepStride = input0_last_axis_stride * sizeof(T1) / ONE_BLK_SIZE;
    }
  } else {
    // DoSelectParams params;
    if constexpr (is_int64_scene) { /* int64 */
      WherePartitionBufferInt64(tmp_buf, params, used_size);
    } else {
      WherePartition3Buffer(tmp_buf, params, used_size);
    }
    if (useWhereExtend) {
      params.rpt_params.src0BlkStride = 1;
      params.rpt_params.src1BlkStride = 1;
      params.rpt_params.src0RepStride = input0_last_axis_stride * sizeof(T1) / ONE_BLK_SIZE;
      params.rpt_params.src1RepStride = input1_last_axis_stride * sizeof(T2) / ONE_BLK_SIZE;
    }
  }
  if (useWhereExtend) {
    params.rpt_params.dstRepStride = output_last_axis_stride * sizeof(T) / ONE_BLK_SIZE;
    params.output_stride = output_last_axis_stride;
    params.mask_stride = mask_last_axis_stride;
    params.input0_stride = input0_last_axis_stride;
    params.input1_stride = input1_last_axis_stride;
    WhereExtend(dst, mask, src0, src1, first_axis, last_axis, params);
  } else {
    for (uint32_t i = 0; i < first_axis; i++) {
      DoSelectParams tempParams = params;
      WhereBase(dst[i * output_last_axis_stride], mask[i * mask_last_axis_stride],
                src0[i * params.input0_stride * params.src0_select_offset],
                src1[i * params.input1_stride * params.src1_select_offset], last_axis, tempParams);
    }
  }
}
#endif  // __ASCENDC_API_WHERE_H__
