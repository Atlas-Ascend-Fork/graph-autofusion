/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_COMPARE_H__
#define __ASCENDC_API_COMPARE_H__

inline __aicore__ void TmpDupInt64(int64_t src, AscendC::LocalTensor<uint8_t> &tmp_buf) {
  uint16_t bit_00 = src & 0xFFFF;        // 最低16位
  uint16_t bit_16 = src >> 16 & 0xFFFF;  // 次低16位
  uint16_t bit_32 = src >> 32 & 0xFFFF;  // 次高16位
  uint16_t bit_48 = src >> 48 & 0xFFFF;  // 最高16位

  constexpr uint32_t TRANSPOSE_MIN_NUM = 16;
  constexpr uint32_t TRANSPOSE_MIN_CUBE_BYTE_SIZE = TRANSPOSE_MIN_NUM * TRANSPOSE_MIN_NUM * sizeof(uint16_t);

  AscendC::LocalTensor<uint16_t> init_buf = KernelUtils::NewTensor<uint16_t>(tmp_buf, 0, TRANSPOSE_MIN_CUBE_BYTE_SIZE);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 0], bit_00, TRANSPOSE_MIN_NUM);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 1], bit_16, TRANSPOSE_MIN_NUM);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 2], bit_32, TRANSPOSE_MIN_NUM);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 3], bit_48, TRANSPOSE_MIN_NUM);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 4], bit_00, TRANSPOSE_MIN_NUM);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 5], bit_16, TRANSPOSE_MIN_NUM);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 6], bit_32, TRANSPOSE_MIN_NUM);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 7], bit_48, TRANSPOSE_MIN_NUM);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 8], bit_00, TRANSPOSE_MIN_NUM);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 9], bit_16, TRANSPOSE_MIN_NUM);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 10], bit_32, TRANSPOSE_MIN_NUM);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 11], bit_48, TRANSPOSE_MIN_NUM);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 12], bit_00, TRANSPOSE_MIN_NUM);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 13], bit_16, TRANSPOSE_MIN_NUM);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 14], bit_32, TRANSPOSE_MIN_NUM);
  Duplicate(init_buf[TRANSPOSE_MIN_NUM * 15], bit_48, TRANSPOSE_MIN_NUM);
  AscendC::PipeBarrier<PIPE_V>();
  Transpose(init_buf, init_buf);
}

template <typename T>
inline __aicore__ void CompareScalarExtend(const AscendC::LocalTensor<T> &dst,         // output
                                           const AscendC::LocalTensor<int64_t> &src0,  // input 0 is a tensor
                                           const int64_t src1,                         // input 1 is a constant
                                           CMPMODE mode,                               // compare mode
                                           const uint32_t cal_cnt,                     // total numbers
                                           AscendC::LocalTensor<uint8_t> &tmp_buf) {
  if (mode != CMPMODE::EQ && mode != CMPMODE::NE) {
    ASSERT(false && "CompareScalarExtend mode only supports EQ/NE when DataType is int64.");
  }

  // ------ Step1: duplicate tensors -----
  TmpDupInt64(src1, tmp_buf);
  LocalTensor<int64_t> src1_buf = KernelUtils::NewTensor<int64_t>(tmp_buf, 0, KernelUtils::BlkSize<int64_t>());
  LocalTensor<uint8_t> rest_buf = tmp_buf[ONE_BLK_SIZE].template ReinterpretCast<uint8_t>();

  LocalTensor<half> all_one_buf = KernelUtils::NewTensor<half>(tmp_buf, ONE_BLK_SIZE, KernelUtils::RptSize<half>());
  Duplicate(all_one_buf, (half)1.0f, KernelUtils::RptSize<half>());

  // ----- Step2: allocate buffer, split the rest tmp_buf to 2 parts -----
  //   1. sub_res_buf: for Sub result and reuse for CompareScalar & SelectScalar result.
  //   2. cast_res_buf: for Cast sub result from int64 -> float
  constexpr uint32_t RATIO = sizeof(int64_t) / sizeof(int32_t);
  constexpr uint32_t ONE_RPT_SIZE = KernelUtils::RptSize<int32_t>();
  constexpr uint32_t PARTITION_NUMBER = sizeof(int32_t) + sizeof(float);
  const uint32_t rest_buf_byte_size = tmp_buf.GetSize() - ONE_BLK_SIZE - ONE_REPEAT_BYTE_SIZE;
  const uint32_t each_part_buf_byte_size =
      rest_buf_byte_size / PARTITION_NUMBER / ONE_REPEAT_BYTE_SIZE * ONE_REPEAT_BYTE_SIZE;
  const uint32_t size = cal_cnt * RATIO;

  LocalTensor<int32_t> sub_res_buf =
      KernelUtils::NewTensor<int32_t>(tmp_buf, ONE_BLK_SIZE + ONE_REPEAT_BYTE_SIZE, each_part_buf_byte_size);
  LocalTensor<int64_t> sub_res_64_buf = sub_res_buf.template ReinterpretCast<int64_t>();

  uint32_t offset = ONE_BLK_SIZE + ONE_REPEAT_BYTE_SIZE + each_part_buf_byte_size * sizeof(int32_t);
  LocalTensor<float> cast_res_buf = KernelUtils::NewTensor<float>(tmp_buf, offset, each_part_buf_byte_size);

  // reuse buf from sub_res_buf for CompareScalar & SelectScalar
  LocalTensor<uint8_t> sub_res_reuse_buf = sub_res_buf.template ReinterpretCast<uint8_t>();
  LocalTensor<uint8_t> cmp_res_buf = KernelUtils::NewTensor<uint8_t>(sub_res_reuse_buf, 0, each_part_buf_byte_size);
  LocalTensor<half> sel_res_buf =
      KernelUtils::NewTensor<half>(sub_res_reuse_buf, each_part_buf_byte_size, each_part_buf_byte_size);

  LocalTensor<int32_t> src0_new = src0.ReinterpretCast<int32_t>();
  LocalTensor<int32_t> src1_new = src1_buf.ReinterpretCast<int32_t>();

  // ----- Step3: do compare -----
  BinaryRepeatParams sub_params;
  sub_params.src1BlkStride = 0;
  sub_params.src1RepStride = 0;

  BinaryRepeatParams sel_params;
  sel_params.src0BlkStride = 0;
  sel_params.src0RepStride = 0;
  sel_params.src1RepStride = 0;
  sel_params.src1RepStride = 0;

  uint32_t max_buf_size = sub_res_buf.GetSize();
  uint32_t max_buf_rpt_num = KernelUtils::RptNum<int32_t>(max_buf_size);
  uint32_t max_do_rpt_num = KernelUtils::Min(MAX_REPEAT_TIME, max_buf_rpt_num);
  uint32_t max_do_size = max_do_rpt_num * ONE_RPT_SIZE;
  sub_params.blockNumber = KernelUtils::BlkNum<int32_t>(max_do_size);  // real block number
  uint32_t max_dst_rpt_num = max_do_rpt_num / RATIO;
  uint32_t max_dst_size = max_do_size / RATIO;
  sel_params.blockNumber = KernelUtils::BlkNum<half>(max_dst_size);  // real block number
  uint32_t calc_size = 0;

  // ----- Step3.1: do compare: do max repeat once time -----
  for (; calc_size + max_do_size <= size; calc_size += max_do_size) {
    AscendC::Sub(sub_res_buf[0], src0_new[calc_size], src1_new[0], ONE_RPT_SIZE, max_do_rpt_num, sub_params);
    AscendC::Cast(cast_res_buf[0], sub_res_64_buf[0], AscendC::RoundMode::CAST_RINT, max_dst_size);
    AscendC::CompareScalar(cmp_res_buf[0], cast_res_buf[0], 0.0f, mode, max_dst_size);
    AscendC::Select(sel_res_buf[0],                             // output
                    cmp_res_buf[0],                             // condition
                    all_one_buf[0],                             // then
                    (half)0,                                    // else
                    AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,  // scalar mode
                    KernelUtils::RptSize<half>(),               // mask
                    max_dst_rpt_num,                            // repeat times
                    sel_params);
    AscendC::Cast(dst[calc_size / RATIO], sel_res_buf[0], AscendC::RoundMode::CAST_NONE, max_dst_size);
  }

  // ----- Step3.2: do compare: do rest repeats -----
  if (calc_size + ONE_RPT_SIZE <= size) {
    uint32_t left_rpt_num = KernelUtils::RptNum<int32_t>(size - calc_size);
    uint32_t do_size = left_rpt_num * KernelUtils::RptSize<int32_t>();
    uint32_t dst_size = do_size / RATIO;

    AscendC::Sub(sub_res_buf[0], src0_new[calc_size], src1_new[0], ONE_RPT_SIZE, left_rpt_num, sub_params);
    AscendC::Cast(cast_res_buf[0], sub_res_64_buf[0], AscendC::RoundMode::CAST_RINT, dst_size);
    AscendC::CompareScalar(cmp_res_buf[0], cast_res_buf[0], 0.0f, mode, do_size);
    AscendC::Select(sel_res_buf[0],                             // output
                    cmp_res_buf[0],                             // condition
                    all_one_buf[0],                             // then
                    (half)0,                                    // else
                    AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,  // scalar mode
                    KernelUtils::RptSize<half>(),               // mask
                    left_rpt_num,                               // repeat times
                    sel_params);
    AscendC::Cast(dst[calc_size / RATIO], sel_res_buf[0], AscendC::RoundMode::CAST_NONE, dst_size);

    calc_size += do_size;
  }

  // ----- Step3.3: do compare: do rest blocks -----
  if (calc_size < size) {
    uint32_t left_size = size - calc_size;
    uint32_t dst_size = left_size / RATIO;
    uint32_t mask = KernelUtils::SizeAlign(left_size, KernelUtils::BlkSize<int32_t>());
    sub_params.blockNumber = KernelUtils::BlkNum<int32_t>(mask);
    sel_params.blockNumber = KernelUtils::SizeAlign(dst_size, KernelUtils::BlkSize<half>());

    AscendC::Sub(sub_res_buf[0], src0_new[calc_size], src1_new[0], mask, 1, sub_params);
    AscendC::Cast(cast_res_buf[0], sub_res_64_buf[0], AscendC::RoundMode::CAST_RINT, dst_size);
    AscendC::CompareScalar(cmp_res_buf[0], cast_res_buf[0], 0.0f, mode, ONE_RPT_SIZE);
    AscendC::Select(sel_res_buf[0],                             // output
                    cmp_res_buf[0],                             // condition
                    all_one_buf[0],                             // then
                    (half)0,                                    // else
                    AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,  // scalar mode
                    KernelUtils::RptSize<half>(),               // mask
                    1,                                          // repeat times
                    sel_params);
    AscendC::Cast(dst[calc_size / RATIO], sel_res_buf[0], AscendC::RoundMode::CAST_NONE, dst_size);
  }
}

inline __aicore__ void ApplyCompareMode(LocalTensor<int32_t> &inter_buf, CMPMODE mode, uint32_t num_elements) {
  // 根据比较模式进行不同的后处理
  switch (mode) {
    case CMPMODE::GE:
      AscendC::Adds(inter_buf, inter_buf, (int32_t)1, num_elements);
      AscendC::PipeBarrier<PIPE_V>();
    case CMPMODE::GT:
      AscendC::Maxs(inter_buf, inter_buf, (int32_t)0, num_elements);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mins(inter_buf, inter_buf, (int32_t)1, num_elements);
      break;

    case CMPMODE::LE:
      AscendC::Adds(inter_buf, inter_buf, (int32_t)(-1), num_elements);
      AscendC::PipeBarrier<PIPE_V>();
    case CMPMODE::LT:
    case CMPMODE::NE:
      AscendC::Maxs(inter_buf, inter_buf, (int32_t)(-1), num_elements);
      AscendC::PipeBarrier<PIPE_V>();
      if (mode == CMPMODE::NE) {
        AscendC::Mins(inter_buf, inter_buf, (int32_t)1, num_elements);
      } else {
        AscendC::Mins(inter_buf, inter_buf, (int32_t)0, num_elements);
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul(inter_buf, inter_buf, inter_buf, num_elements);
      break;
    default:
      break;
  }
  AscendC::PipeBarrier<PIPE_V>();
}

template <typename OutT>
inline __aicore__ void PerformTypeConversion(const LocalTensor<OutT> &dst, LocalTensor<int32_t> &inter_buf,
                                             uint32_t offset, uint32_t num_elements) {
  // int32 -> int16
  LocalTensor<int16_t> int16_buf = inter_buf.template ReinterpretCast<int16_t>();
  AscendC::Cast(int16_buf, inter_buf, RoundMode::CAST_NONE, num_elements);

  // int16 -> half
  LocalTensor<half> half_buf = inter_buf.template ReinterpretCast<half>();
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::Cast(half_buf, int16_buf, RoundMode::CAST_NONE, num_elements);

  // half -> OutT
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::Cast(dst[offset], half_buf, RoundMode::CAST_NONE, num_elements);
}

// 处理标量比较的一个计算块
template <typename OutT>
inline __aicore__ void ProcessScalarCompareBlock(const LocalTensor<OutT> &dst, const LocalTensor<int32_t> &src0,
                                                 const LocalTensor<int32_t> &tensor_src1,
                                                 LocalTensor<int32_t> &inter_buf, uint32_t offset, uint32_t mask,
                                                 uint32_t repeat_times, uint32_t num_elements, CMPMODE mode,
                                                 const BinaryRepeatParams &binary_param) {
  AscendC::Sub(inter_buf, src0[offset], tensor_src1[0], mask, repeat_times, binary_param);
  AscendC::PipeBarrier<PIPE_V>();

  ApplyCompareMode(inter_buf, mode, num_elements);
  PerformTypeConversion(dst, inter_buf, offset, num_elements);
}

template <typename OutT>
inline __aicore__ void CompareScalarExtendInt32(const LocalTensor<OutT> &dst, const LocalTensor<int32_t> &src0,
                                                const int32_t scalar_src1, CMPMODE mode, const uint32_t cal_cnt,
                                                LocalTensor<uint8_t> &tmp_buf) {
  uint32_t tmp_buf_size = tmp_buf.GetSize() * sizeof(uint8_t) - ONE_BLK_SIZE;
  uint32_t loop_cnt;
  uint32_t repeat_times = MAX_REPEAT_TIME;
  uint32_t one_blk_num = KernelUtils::BlkSize<int32_t>();
  uint32_t one_repeat_num = KernelUtils::RptSize<int32_t>();
  // 计算循环参数
  if (tmp_buf_size >= (ONE_REPEAT_BYTE_SIZE * MAX_REPEAT_TIME)) {
    loop_cnt = cal_cnt / (one_repeat_num * MAX_REPEAT_TIME);
  } else {
    repeat_times = tmp_buf_size / ONE_REPEAT_BYTE_SIZE;
    loop_cnt = cal_cnt / (repeat_times * one_repeat_num);
  }

  uint32_t one_step_num = repeat_times * one_repeat_num;
  LocalTensor<int32_t> tensor_src1 = tmp_buf[0].template ReinterpretCast<int32_t>();
  tensor_src1.SetSize(one_blk_num);
  Duplicate(tensor_src1, scalar_src1, one_blk_num);

  uint32_t inter_buf_offset = KernelUtils::BlkAlign<uint8_t>(one_blk_num * sizeof(int32_t));
  LocalTensor<int32_t> inter_buf = tmp_buf[inter_buf_offset].template ReinterpretCast<int32_t>();
  inter_buf.SetSize(one_step_num);

  BinaryRepeatParams binary_param(1, 1, 0, 8, 8, 0);
  uint64_t mask = one_repeat_num;
  uint32_t offset = 0;
  // 主循环处理
  for (uint32_t i = 0; i < loop_cnt; i++) {
    ProcessScalarCompareBlock(dst, src0, tensor_src1, inter_buf, offset, mask, repeat_times, one_step_num, mode,
                              binary_param);
    offset += one_step_num;
  }
  // 剩余完整repeat块处理
  uint32_t remain_rpt_times = (cal_cnt - offset) / one_repeat_num;
  uint32_t remain_nums = remain_rpt_times * one_repeat_num;

  if (remain_rpt_times != 0) {
    ProcessScalarCompareBlock(dst, src0, tensor_src1, inter_buf, offset, mask, remain_rpt_times, remain_nums, mode,
                              binary_param);
    offset += remain_nums;
  }
  // 尾部处理
  uint32_t calc_tail = cal_cnt - offset;
  if (calc_tail != 0) {
    uint32_t aligned_tail = KernelUtils::BlkAlign<int32_t>(calc_tail);
    BinaryRepeatParams tail_param(1, 1, 0, 8, 8, 0);
    tail_param.src0RepStride = aligned_tail;
    tail_param.dstRepStride = aligned_tail;
    ProcessScalarCompareBlock(dst, src0, tensor_src1, inter_buf, offset, calc_tail, 1, calc_tail, mode, tail_param);
  }
}

template <typename InT, typename OutT>
inline __aicore__ void CompareScalarExtend(const LocalTensor<OutT> &dst, const LocalTensor<InT> &src,
                                           const InT constant_y, CMPMODE mode, const uint32_t cal_cnt,
                                           LocalTensor<uint8_t> &tmp_buf) {
  // 如果输入是int32, 且model是GE/GT/LE/LT/NE, 则走特化处理方法
  if constexpr (AscendC::IsSameType<InT, int32_t>::value) {
    if (mode == CMPMODE::GT || mode == CMPMODE::GE || mode == CMPMODE::LE || mode == CMPMODE::LT ||
        mode == CMPMODE::NE) {
      return CompareScalarExtendInt32(dst, src, constant_y, mode, cal_cnt, tmp_buf);
    }
  }

  constexpr int32_t one_block_cnt = ONE_BLK_SIZE / sizeof(InT);
  constexpr uint16_t compare_size = KernelUtils::BlkAlign<uint8_t>((64 * MAX_REPEAT_TIMES) / 8);
  constexpr uint16_t max_compare_size = (compare_size > 3072) ? 3072 : compare_size;
  constexpr uint16_t max_block_cnt = max_compare_size / sizeof(InT);
  const int32_t loop_num = cal_cnt / max_block_cnt;
  uint32_t tmp_offset = 0;
  LocalTensor<half> tensor_src1 = tmp_buf[0].template ReinterpretCast<half>();
  tensor_src1.SetSize(KernelUtils::BlkSize<half>());
  Duplicate(tensor_src1, (half)1.0, KernelUtils::BlkSize<half>());
  tmp_offset += ONE_BLK_SIZE;
  LocalTensor<uint8_t> compare_out = tmp_buf[tmp_offset].template ReinterpretCast<uint8_t>();
  compare_out.SetSize(max_block_cnt / 8);
  tmp_offset += max_block_cnt * sizeof(uint8_t) / 8;
  LocalTensor<half> select_out = tmp_buf[tmp_offset].template ReinterpretCast<half>();
  select_out.SetSize(max_block_cnt);
  tmp_offset += max_block_cnt * sizeof(half);
  BinaryRepeatParams binary_param(1, 0, 0, 8, 0, 0);
  constexpr uint16_t one_repeat_cnt_half = KernelUtils::RptSize<half>();
  constexpr int32_t one_repeat_cnt_in = KernelUtils::RptSize<InT>();
  constexpr int32_t one_repeat_cnt_out = KernelUtils::RptSize<OutT>();
  constexpr uint64_t cast_mask = sizeof(OutT) > sizeof(half) ? one_repeat_cnt_out : one_repeat_cnt_half;
  UnaryRepeatParams cast_param{1, 1, 8 * cast_mask / one_repeat_cnt_out, 8 * cast_mask / one_repeat_cnt_half};
  int32_t cnt = 0;
  int loop = 0;
  for (loop = 0; loop < loop_num; loop++) {
    AscendC::PipeBarrier<PIPE_V>();
    CompareScalar(compare_out[0], src[cnt], constant_y, mode, one_repeat_cnt_in, max_block_cnt / one_repeat_cnt_in,
                  {1, 1, 8, 8});
    AscendC::PipeBarrier<PIPE_V>();
    Select(select_out[0], compare_out[0], tensor_src1[0], (half)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE,
           one_repeat_cnt_half, max_block_cnt / one_repeat_cnt_half, {1, 0, 0, 8, 0, 0});
    AscendC::PipeBarrier<PIPE_V>();
    Cast(dst[cnt], select_out[0], RoundMode::CAST_NONE, cast_mask, max_block_cnt / cast_mask, cast_param);
    cnt += max_block_cnt;
  }
  // loop for repeat
  int32_t left_cnt_num = cal_cnt % max_block_cnt;
  if (left_cnt_num >= one_repeat_cnt_in) {
    int32_t do_block_cnt = left_cnt_num / one_repeat_cnt_in * one_repeat_cnt_in;
    AscendC::PipeBarrier<PIPE_V>();
    CompareScalar(compare_out[0], src[cnt], constant_y, mode, one_repeat_cnt_in, do_block_cnt / one_repeat_cnt_in,
                  {1, 1, 8, 8});
    AscendC::PipeBarrier<PIPE_V>();
    Select(select_out[0], compare_out[0], tensor_src1[0], (half)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE,
           one_repeat_cnt_half, (do_block_cnt + one_repeat_cnt_half - 1) / one_repeat_cnt_half, binary_param);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(dst[cnt], select_out[0], RoundMode::CAST_NONE, do_block_cnt);
    cnt += do_block_cnt;
  }
  int32_t block_cnt = DivCeil((int32_t)cal_cnt - cnt, one_block_cnt);
  if (block_cnt > 0) {
    const int32_t left_cnt = block_cnt * one_block_cnt;
    AscendC::PipeBarrier<PIPE_V>();
    CompareScalar(compare_out[0], src[cnt], constant_y, mode, left_cnt, 1, {1, 1, 8, 8});
    AscendC::PipeBarrier<PIPE_V>();
    Select(select_out[0], compare_out[0], tensor_src1[0], (half)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, left_cnt, 1,
           {1, 0, 0, 8, 0, 0});
    AscendC::PipeBarrier<PIPE_V>();
    Cast(dst[cnt], select_out[0], RoundMode::CAST_NONE, left_cnt, 1, cast_param);
  }
}

template <typename T>
inline __aicore__ void CompareScalarExtend(const LocalTensor<T> &dst, const LocalTensor<T> &src, const T constant_y,
                                           CMPMODE mode, const uint32_t cal_cnt, LocalTensor<uint8_t> &tmp_buf) {
  const int32_t one_block_cnt = ONE_BLK_SIZE / sizeof(T);
  const uint16_t compare_size = KernelUtils::BlkAlign<uint8_t>((64 * MAX_REPEAT_TIMES) / 8);
  const uint16_t max_compare_size = (compare_size > 49152) ? 49152 : compare_size;
  const uint16_t max_block_cnt = max_compare_size / sizeof(T);
  const int32_t loop_num = cal_cnt / max_block_cnt;
  uint32_t tmp_offset = 0;
  LocalTensor<uint8_t> compare_out = tmp_buf[tmp_offset].template ReinterpretCast<uint8_t>();
  compare_out.SetSize(max_block_cnt / 8);
  Duplicate(dst[0], (T)1.0, max_block_cnt);
  tmp_offset += max_block_cnt * sizeof(T) / 8;
  LocalTensor<T> src_tmp = tmp_buf[tmp_offset].template ReinterpretCast<T>();
  int32_t cnt = 0;
  for (int loop = 0; loop < loop_num; loop++) {
    AscendC::PipeBarrier<PIPE_V>();
    CompareScalar(compare_out[0], src[cnt], constant_y, mode, max_block_cnt);
    AscendC::PipeBarrier<PIPE_V>();
    Select(dst[cnt], compare_out[0], dst[cnt], (T)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, max_block_cnt);
    cnt += max_block_cnt;
  }
  const int32_t block_cnt = ((int32_t)cal_cnt - cnt) / one_block_cnt;
  if (block_cnt > 0) {
    int32_t left_cnt = block_cnt * one_block_cnt;
    if (left_cnt * sizeof(T) < 256) {
      AscendC::PipeBarrier<PIPE_V>();
      DataCopy(src_tmp[0], src[cnt], left_cnt);
      AscendC::PipeBarrier<PIPE_V>();
      // 尾块数据拷贝到临时缓冲起始位置，因此下标改为 0。
      CompareScalar(compare_out[0], src_tmp[0], constant_y, mode, 256 / sizeof(T));
    } else {
      AscendC::PipeBarrier<PIPE_V>();
      CompareScalar(compare_out[0], src[cnt], constant_y, mode, left_cnt);
    }
    AscendC::PipeBarrier<PIPE_V>();
    Select(dst[cnt], compare_out[0], dst[cnt], (T)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, left_cnt);
    cnt += left_cnt;
  }
  const int32_t tail_size = cal_cnt - cnt;
  if (tail_size > 0) {
    ASSERT(false && "CompareScalarExtend size is not supported.");
  }
}

// ================================ 以下是两个tensor compare的api实现 =================================================
// int64_t类型的两个tensor比较，输出dtype是uint8_t类型的, 支持的mode：NE，EQ
template <typename T>
inline __aicore__ void CompareExtendEQ(const AscendC::LocalTensor<T> &dst,         // output
                                       const AscendC::LocalTensor<int64_t> &src0,  // input 0 is a tensor
                                       const AscendC::LocalTensor<int64_t> &src1,  // input 1 is a tensor
                                       CMPMODE mode,                               // compare mode
                                       const uint32_t cal_cnt,                     // total numbers
                                       AscendC::LocalTensor<uint8_t> &tmp_buf) {
  // ------ Step1: duplicate tensors -----
  LocalTensor<half> all_one_buf = KernelUtils::NewTensor<half>(tmp_buf, ONE_BLK_SIZE, KernelUtils::RptSize<half>());
  Duplicate(all_one_buf, (half)1.0f, KernelUtils::RptSize<half>());

  // ----- Step2: allocate buffer, split the rest tmp_buf to 2 parts -----
  //   1. sub_res_buf: for Sub result and reuse for CompareScalar & SelectScalar result.
  //   2. cast_res_buf: for Cast sub result from int64 -> float
  constexpr uint32_t RATIO = sizeof(int64_t) / sizeof(int32_t);
  constexpr uint32_t ONE_RPT_SIZE = KernelUtils::RptSize<int32_t>();
  constexpr uint32_t PARTITION_NUMBER = sizeof(int32_t) + sizeof(float);
  const uint32_t rest_buf_byte_size = tmp_buf.GetSize() - ONE_BLK_SIZE - ONE_REPEAT_BYTE_SIZE;
  const uint32_t each_part_buf_byte_size =
      rest_buf_byte_size / PARTITION_NUMBER / ONE_REPEAT_BYTE_SIZE * ONE_REPEAT_BYTE_SIZE;
  const uint32_t size = cal_cnt * RATIO;

  LocalTensor<int32_t> sub_res_buf =
      KernelUtils::NewTensor<int32_t>(tmp_buf, ONE_BLK_SIZE + ONE_REPEAT_BYTE_SIZE, each_part_buf_byte_size);
  LocalTensor<int64_t> sub_res_64_buf = sub_res_buf.ReinterpretCast<int64_t>();

  uint32_t offset = ONE_BLK_SIZE + ONE_REPEAT_BYTE_SIZE + each_part_buf_byte_size * sizeof(int32_t);
  LocalTensor<float> cast_res_buf = KernelUtils::NewTensor<float>(tmp_buf, offset, each_part_buf_byte_size);

  // reuse buf from sub_res_buf for CompareScalar & SelectScalar
  LocalTensor<uint8_t> sub_res_reuse_buf = sub_res_buf.template ReinterpretCast<uint8_t>();
  LocalTensor<uint8_t> cmp_res_buf = KernelUtils::NewTensor<uint8_t>(sub_res_reuse_buf, 0, each_part_buf_byte_size);
  LocalTensor<half> sel_res_buf =
      KernelUtils::NewTensor<half>(sub_res_reuse_buf, each_part_buf_byte_size, each_part_buf_byte_size);

  LocalTensor<int32_t> src0_new = src0.ReinterpretCast<int32_t>();
  LocalTensor<int32_t> src1_new = src1.ReinterpretCast<int32_t>();

  // ----- Step3: do compare -----
  BinaryRepeatParams sel_params;
  sel_params.src0BlkStride = 0;
  sel_params.src0RepStride = 0;
  sel_params.src1RepStride = 0;
  sel_params.src1RepStride = 0;

  uint32_t max_buf_size = sub_res_buf.GetSize();
  uint32_t max_buf_rpt_num = KernelUtils::RptNum<int32_t>(max_buf_size);
  uint32_t max_do_rpt_num = KernelUtils::Min(MAX_REPEAT_TIME, max_buf_rpt_num);
  uint32_t max_do_size = max_do_rpt_num * ONE_RPT_SIZE;
  uint32_t max_dst_rpt_num = max_do_rpt_num / RATIO;
  uint32_t max_dst_size = max_do_size / RATIO;
  sel_params.blockNumber = KernelUtils::BlkNum<half>(max_dst_size);  // real block number
  uint32_t calc_size = 0;

  // ----- Step3.1: do compare: do max repeat once time -----
  for (; calc_size + max_do_size <= size; calc_size += max_do_size) {
    AscendC::Sub(sub_res_buf[0], src0_new[calc_size], src1_new[calc_size], max_do_size);
    AscendC::Cast(cast_res_buf[0], sub_res_64_buf[0], AscendC::RoundMode::CAST_RINT, max_dst_size);
    AscendC::CompareScalar(cmp_res_buf[0], cast_res_buf[0], 0.0f, mode, max_dst_size);
    AscendC::Select(sel_res_buf[0],                             // output
                    cmp_res_buf[0],                             // condition
                    all_one_buf[0],                             // then
                    (half)0,                                    // else
                    AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,  // scalar mode
                    KernelUtils::RptSize<half>(),               // mask
                    max_dst_rpt_num,                            // repeat times
                    sel_params);
    AscendC::Cast(dst[calc_size / RATIO], sel_res_buf[0], AscendC::RoundMode::CAST_NONE, max_dst_size);
  }

  // ----- Step3.2: do compare: do rest repeats -----
  if (calc_size + ONE_RPT_SIZE <= size) {
    uint32_t left_rpt_num = KernelUtils::RptNum<int32_t>(size - calc_size);
    uint32_t do_size = left_rpt_num * KernelUtils::RptSize<int32_t>();
    uint32_t dst_size = do_size / RATIO;

    AscendC::Sub(sub_res_buf[0], src0_new[calc_size], src1_new[calc_size], (ONE_RPT_SIZE * left_rpt_num));
    AscendC::Cast(cast_res_buf[0], sub_res_64_buf[0], AscendC::RoundMode::CAST_RINT, dst_size);
    AscendC::CompareScalar(cmp_res_buf[0], cast_res_buf[0], 0.0f, mode, do_size);
    AscendC::Select(sel_res_buf[0],                             // output
                    cmp_res_buf[0],                             // condition
                    all_one_buf[0],                             // then
                    (half)0,                                    // else
                    AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,  // scalar mode
                    KernelUtils::RptSize<half>(),               // mask
                    left_rpt_num,                               // repeat times
                    sel_params);
    AscendC::Cast(dst[calc_size / RATIO], sel_res_buf[0], AscendC::RoundMode::CAST_NONE, dst_size);

    calc_size += do_size;
  }

  // ----- Step3.3: do compare: do rest blocks -----
  if (calc_size < size) {
    uint32_t left_size = size - calc_size;
    uint32_t dst_size = left_size / RATIO;
    uint32_t mask = KernelUtils::SizeAlign(left_size, KernelUtils::BlkSize<int32_t>());
    sel_params.blockNumber = KernelUtils::SizeAlign(dst_size, KernelUtils::BlkSize<half>());

    AscendC::Sub(sub_res_buf[0], src0_new[calc_size], src1_new[calc_size], mask);
    AscendC::Cast(cast_res_buf[0], sub_res_64_buf[0], AscendC::RoundMode::CAST_RINT, dst_size);
    AscendC::CompareScalar(cmp_res_buf[0], cast_res_buf[0], 0.0f, mode, ONE_RPT_SIZE);
    AscendC::Select(sel_res_buf[0],                             // output
                    cmp_res_buf[0],                             // condition
                    all_one_buf[0],                             // then
                    (half)0,                                    // else
                    AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,  // scalar mode
                    KernelUtils::RptSize<half>(),               // mask
                    1,                                          // repeat times
                    sel_params);
    AscendC::Cast(dst[calc_size / RATIO], sel_res_buf[0], AscendC::RoundMode::CAST_NONE, dst_size);
  }
}

template <typename OutT>
inline __aicore__ void ProcessTensorCompareBlock(const LocalTensor<OutT> &dst, const LocalTensor<int32_t> &src0,
                                                 const LocalTensor<int32_t> &src1, LocalTensor<int32_t> &inter_buf,
                                                 uint32_t offset, uint32_t num_elements, CMPMODE mode) {
  AscendC::Sub(inter_buf, src0[offset], src1[offset], num_elements);
  AscendC::PipeBarrier<PIPE_V>();

  ApplyCompareMode(inter_buf, mode, num_elements);
  PerformTypeConversion(dst, inter_buf, offset, num_elements);
}

template <typename OutT>
inline __aicore__ void CompareExtendInt32(const LocalTensor<OutT> &dst, const LocalTensor<int32_t> &src0,
                                          const LocalTensor<int32_t> &src1, CMPMODE mode, const uint32_t cal_cnt,
                                          LocalTensor<uint8_t> &tmp_buf) {
  uint32_t buf_max_cnt = tmp_buf.GetSize() * sizeof(uint8_t) / sizeof(int32_t);
  uint32_t loop_cnt = cal_cnt / buf_max_cnt;
  uint32_t max_rpt_cnts = buf_max_cnt * sizeof(int32_t) / ONE_REPEAT_BYTE_SIZE;

  LocalTensor<int32_t> inter_buf = tmp_buf.ReinterpretCast<int32_t>();
  uint32_t offset = 0;

  // 主循环处理完整块
  for (uint32_t i = 0; i < loop_cnt; i++) {
    ProcessTensorCompareBlock(dst, src0, src1, inter_buf, offset, buf_max_cnt, mode);
    offset += buf_max_cnt;
  }

  // 处理尾部剩余元素
  uint32_t tail_cnt = cal_cnt - offset;
  if (tail_cnt > 0) {
    ProcessTensorCompareBlock(dst, src0, src1, inter_buf, offset, tail_cnt, mode);
  }
}

template <typename InT, typename OutT>
inline __aicore__ void CompareExtend(const LocalTensor<OutT> &dst, const LocalTensor<InT> &src0,
                                     const LocalTensor<InT> &src1, CMPMODE mode, const uint32_t cal_cnt,
                                     LocalTensor<uint8_t> &tmp_buf) {
  if constexpr (AscendC::IsSameType<InT, int32_t>::value && AscendC::IsSameType<OutT, uint8_t>::value) {
    if (mode != CMPMODE::GT && mode != CMPMODE::GE && mode != CMPMODE::LE && mode != CMPMODE::LT &&
        mode != CMPMODE::NE && mode != CMPMODE::EQ) {
      ASSERT(false && "CompareExtend mode only supports EQ/GT/GE/LE/LT/NE when DataType is int32.");
    }
    if (mode == CMPMODE::GT || mode == CMPMODE::GE || mode == CMPMODE::LE || mode == CMPMODE::LT ||
        mode == CMPMODE::NE) {
      return CompareExtendInt32(dst, src0, src1, mode, cal_cnt, tmp_buf);
    }
  }
  constexpr int32_t one_block_cnt = ONE_BLK_SIZE / sizeof(half);
  constexpr int32_t one_repeat_cnt_in = KernelUtils::RptSize<InT>();
  uint32_t max_block_cnt = (tmp_buf.GetSize() - ONE_BLK_SIZE) / 3;
  max_block_cnt = max_block_cnt * sizeof(half) / ONE_REPEAT_BYTE_SIZE * ONE_REPEAT_BYTE_SIZE / sizeof(half);
  const int32_t loop_num = max_block_cnt == 0 ? 0 : cal_cnt / max_block_cnt;
  uint32_t tmp_offset = 0;
  LocalTensor<half> ones = tmp_buf.template ReinterpretCast<half>();
  Duplicate(ones, (half)1.0, one_block_cnt);
  tmp_offset += ONE_BLK_SIZE;
  LocalTensor<half> select_out = tmp_buf[tmp_offset].template ReinterpretCast<half>();
  tmp_offset += max_block_cnt == 0 ? ONE_REPEAT_BYTE_SIZE : max_block_cnt * sizeof(half);
  LocalTensor<uint8_t> compare_out = tmp_buf[tmp_offset].template ReinterpretCast<uint8_t>();
  int32_t cnt = 0;
  int loop = 0;
  BinaryRepeatParams select_param{1, 0, 0, 8, 0, 0};
  BinaryRepeatParams compare_param{1, 1, 1, 8, 8, 8};
  constexpr uint16_t one_repeat_cnt_half = KernelUtils::RptSize<half>();
  constexpr int32_t one_repeat_cnt_out = KernelUtils::RptSize<OutT>();
  const uint32_t compare_max_rpt_times = max_block_cnt / one_repeat_cnt_in;
  const uint32_t select_max_rpt_times = max_block_cnt / one_repeat_cnt_half;
  for (loop = 0; loop < loop_num; loop++) {
    Compare(compare_out, src0[cnt], src1[cnt], mode, one_repeat_cnt_in, compare_max_rpt_times, compare_param);
    AscendC::PipeBarrier<PIPE_V>();
    Select(select_out, compare_out, ones, (half)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, one_repeat_cnt_half,
           select_max_rpt_times, select_param);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(dst[cnt], select_out, RoundMode::CAST_NONE, max_block_cnt);
    cnt += max_block_cnt;
  }
  // loop for repeat
  int32_t left_cnt_num = cal_cnt - cnt;
  if (left_cnt_num >= one_repeat_cnt_in) {
    const int32_t repeat_times = left_cnt_num / one_repeat_cnt_in;
    const int32_t do_block_cnt = repeat_times * one_repeat_cnt_in;
    Compare(compare_out, src0[cnt], src1[cnt], mode, one_repeat_cnt_in, repeat_times, compare_param);
    AscendC::PipeBarrier<PIPE_V>();
    Select(select_out, compare_out, ones, (half)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, one_repeat_cnt_half,
           (do_block_cnt + one_repeat_cnt_half - 1) / one_repeat_cnt_half, select_param);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(dst[cnt], select_out, RoundMode::CAST_NONE, do_block_cnt);
    cnt += do_block_cnt;
  }
  const int32_t tail_cnt = cal_cnt - cnt;
  if (tail_cnt > 0) {
    Compare(compare_out, src0[cnt], src1[cnt], mode, tail_cnt, 1, compare_param);
    AscendC::PipeBarrier<PIPE_V>();
    Select(select_out, compare_out, ones, (half)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, tail_cnt, 1, select_param);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(dst[cnt], select_out, RoundMode::CAST_NONE, tail_cnt);
  }
}

template <typename T>
inline __aicore__ void CompareExtend(const LocalTensor<T> &dst, const LocalTensor<T> &src0, const LocalTensor<T> &src1,
                                     CMPMODE mode, const uint32_t cal_cnt, LocalTensor<uint8_t> &tmp_buf) {
  const int32_t one_block_cnt = ONE_BLK_SIZE / sizeof(T);
  const uint16_t compare_size = KernelUtils::BlkAlign<uint8_t>((64 * MAX_REPEAT_TIMES) / 8);
  const uint16_t max_compare_size = (compare_size > 49152) ? 49152 : compare_size;
  const uint16_t max_block_cnt = max_compare_size / sizeof(T);
  const int32_t loop_num = cal_cnt / max_block_cnt;
  uint32_t tmp_offset = 0;
  LocalTensor<uint8_t> compare_out = tmp_buf[tmp_offset].template ReinterpretCast<uint8_t>();
  compare_out.SetSize(max_block_cnt / 8);
  Duplicate(dst[0], (T)1.0, max_block_cnt);
  tmp_offset += max_block_cnt * sizeof(T) / 8;
  LocalTensor<T> src_tmp = tmp_buf[tmp_offset].template ReinterpretCast<T>();
  int32_t cnt = 0;
  int loop = 0;
  for (loop = 0; loop < loop_num; loop++) {
    AscendC::PipeBarrier<PIPE_V>();
    Compare(compare_out[0], src0[cnt], src1[cnt], mode, max_block_cnt);
    AscendC::PipeBarrier<PIPE_V>();
    Duplicate(dst[cnt], (T)1.0, max_block_cnt);
    AscendC::PipeBarrier<PIPE_V>();
    Select(dst[cnt], compare_out[0], dst[cnt], (T)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, max_block_cnt);
    cnt += max_block_cnt;
  }
  const int32_t block_cnt = ((int32_t)cal_cnt - cnt) / one_block_cnt;
  if (block_cnt > 0) {
    int32_t left_cnt = block_cnt * one_block_cnt;
    if (left_cnt * sizeof(T) < 256) {
      AscendC::PipeBarrier<PIPE_V>();
      DataCopy(src_tmp[0], src0[cnt], left_cnt);
      AscendC::PipeBarrier<PIPE_V>();
      DataCopy(src_tmp[256 / sizeof(T)], src1[cnt], left_cnt);
      AscendC::PipeBarrier<PIPE_V>();
      // 尾块数据拷贝到临时缓冲起始位置，因此下标改为 0。
      Compare(compare_out[0], src_tmp[0], src_tmp[256 / sizeof(T)], mode, 256 / sizeof(T));
    } else {
      AscendC::PipeBarrier<PIPE_V>();
      Compare(compare_out[0], src0[cnt], src1[cnt], mode, left_cnt);
    }
    AscendC::PipeBarrier<PIPE_V>();
    Duplicate(dst[cnt], (T)1.0, left_cnt);
    AscendC::PipeBarrier<PIPE_V>();
    Select(dst[cnt], compare_out[0], dst[cnt], (T)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, left_cnt);
    cnt += left_cnt;
  }
  const int32_t tail_size = cal_cnt - cnt;
  if (tail_size > 0) {
    ASSERT(false && "CompareExtend size is not supported.");
  }
}

inline __aicore__ void GetSignBitTensor(const AscendC::LocalTensor<uint16_t> &dst0,
                                        const AscendC::LocalTensor<uint16_t> &dst1,
                                        const AscendC::LocalTensor<int64_t> &src0,
                                        const AscendC::LocalTensor<int64_t> &src1,
                                        const AscendC::LocalTensor<uint32_t> &inner_dup, const uint32_t cal_cnt) {
  AscendC::PipeBarrier<PIPE_V>();
  uint32_t quadruple_cal_cnt = 4 * cal_cnt;
  AscendC::Duplicate(inner_dup, uint32_t(0x80000000), 2 * cal_cnt);
  AscendC::LocalTensor<uint16_t> inner_dup_tmp = inner_dup.ReinterpretCast<uint16_t>();
  AscendC::LocalTensor<uint16_t> src0_tmp = src0.ReinterpretCast<uint16_t>();
  AscendC::LocalTensor<uint16_t> src1_tmp = src1.ReinterpretCast<uint16_t>();
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::And(dst0, src0_tmp, inner_dup_tmp, quadruple_cal_cnt);
  AscendC::And(dst1, src1_tmp, inner_dup_tmp, quadruple_cal_cnt);
  AscendC::LocalTensor<int32_t> dst0_tmp = dst0.ReinterpretCast<int32_t>();
  AscendC::LocalTensor<int32_t> dst1_tmp = dst1.ReinterpretCast<int32_t>();
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::Maxs(dst0_tmp, dst0_tmp, -1, 2 * cal_cnt);
  AscendC::Maxs(dst1_tmp, dst1_tmp, -1, 2 * cal_cnt);
  uint32_t one_rpt_cnt = KernelUtils::RptSize<int64_t>();
  uint32_t rpt_times = cal_cnt / one_rpt_cnt;
  uint32_t tail_rpt_cnt = cal_cnt - rpt_times * one_rpt_cnt;
  uint64_t mask[2] = {uint64_t(0x5555555555555555), 0};
  AscendC::Duplicate(inner_dup, uint32_t(0xFFFFFFFF), 2 * cal_cnt);
  AscendC::Duplicate(inner_dup, 1U, mask, rpt_times, 1, 8);
  if (tail_rpt_cnt != 0) {
    uint64_t mask_tail = 0b01;
    for (uint32_t i = 1; i < tail_rpt_cnt; i++) {
      mask_tail += (uint64_t(1) << (2 * i));
    }
    mask[0] = mask_tail;
    AscendC::Duplicate(inner_dup[2 * (cal_cnt - tail_rpt_cnt)], 1U, mask, 1, 1, 8);
  }
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::And(dst0, dst0, inner_dup_tmp, quadruple_cal_cnt);
  AscendC::And(dst1, dst1, inner_dup_tmp, quadruple_cal_cnt);
}

inline __aicore__ void CastTensorToHalf(const AscendC::LocalTensor<int32_t> &dst,
                                        const AscendC::LocalTensor<uint16_t> &src0_bits,
                                        const AscendC::LocalTensor<uint16_t> &src1_bits,
                                        const AscendC::LocalTensor<uint32_t> &inner_dup, uint32_t cal_cnt) {
  AscendC::PipeBarrier<PIPE_V>();
  uint32_t quadruple_cal_cnt = 4 * cal_cnt;
  uint32_t double_cal_cnt = 2 * cal_cnt;
  int32_t repeat_times = AscendC::DivCeil(cal_cnt * sizeof(int64_t), ONE_REPEAT_BYTE_SIZE);
  AscendC::Duplicate(inner_dup, uint32_t(0x0000BC00), double_cal_cnt);
  AscendC::LocalTensor<uint16_t> inner_dup_tmp = inner_dup.ReinterpretCast<uint16_t>();
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::And(src1_bits, src0_bits, inner_dup_tmp, quadruple_cal_cnt);
  AscendC::LocalTensor<int32_t> sub_tmp = inner_dup.ReinterpretCast<int32_t>();
  AscendC::Duplicate(sub_tmp, 0, double_cal_cnt);
  AscendC::Sub(dst, sub_tmp, src0_bits.ReinterpretCast<int32_t>(), double_cal_cnt);
  AscendC::Duplicate(inner_dup, uint32_t(0x00003C00), double_cal_cnt);
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::And(src0_bits, dst.ReinterpretCast<uint16_t>(), inner_dup_tmp, quadruple_cal_cnt);
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::Or(src0_bits, src1_bits, src0_bits, quadruple_cal_cnt);
  uint32_t one_rpt_cnt = KernelUtils::RptSize<half>();
  uint32_t rpt_cnt = quadruple_cal_cnt / one_rpt_cnt;
  uint32_t tail_cnt = quadruple_cal_cnt - rpt_cnt * one_rpt_cnt;
  AscendC::LocalTensor<half> dst_tmp = dst.ReinterpretCast<half>();
  AscendC::LocalTensor<half> sign_tmp = src0_bits.ReinterpretCast<half>();
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::PairReduceSum(dst_tmp, sign_tmp, rpt_cnt, 128, 1, 1, 8);
  if (tail_cnt != 0) {
    AscendC::PairReduceSum(dst_tmp[rpt_cnt * one_rpt_cnt / 2], sign_tmp[rpt_cnt * one_rpt_cnt], 1, tail_cnt, 1, 1, 8);
  }
}

inline __aicore__ void CalcWeightedTensor(const AscendC::LocalTensor<half> &dst, const AscendC::LocalTensor<half> &src,
                                          const AscendC::LocalTensor<uint32_t> &inner_dup, const uint32_t cal_cnt,
                                          const half weight0, const half weight1) {
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::LocalTensor<half> inner_dup_tmp = inner_dup.ReinterpretCast<half>();
  AscendC::Duplicate(inner_dup_tmp, weight0, 2 * cal_cnt);
  uint32_t one_rpt_cnt = KernelUtils::RptSize<half>();
  uint32_t rpt_times = 2 * cal_cnt / one_rpt_cnt;
  uint32_t tail_rpt_cnt = 2 * cal_cnt - rpt_times * one_rpt_cnt;
  uint64_t mask[2] = {uint64_t(0x5555555555555555), uint64_t(0x5555555555555555)};
  AscendC::Duplicate(inner_dup_tmp, weight1, mask, rpt_times, 1, 8);
  if (tail_rpt_cnt != 0) {
    if (tail_rpt_cnt * sizeof(half) < ONE_REPEAT_BYTE_SIZE / 2) {
      uint64_t mask_tail = 0b01;
      for (uint32_t i = 1; i < tail_rpt_cnt / 2; i++) {
        mask_tail += (uint64_t(1) << (2 * i));
      }
      mask[0] = mask_tail;
      mask[1] = 0;
    } else {
      mask[0] = uint64_t(0x5555555555555555);
      uint32_t tail_cnt = tail_rpt_cnt - 128 / sizeof(half);
      uint64_t mask_tail = 0b01;
      for (uint32_t i = 1; i < tail_cnt / 2; i++) {
        mask_tail += (uint64_t(1) << (2 * i));
      }
      mask[1] = mask_tail;
    }
    AscendC::Duplicate(inner_dup_tmp[rpt_times * one_rpt_cnt], weight1, mask, 1, 1, 8);
  }
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::Mul(src, src, inner_dup_tmp, 2 * cal_cnt);
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::PairReduceSum(dst, src, rpt_times, 128, 1, 1, 8);
  if (tail_rpt_cnt != 0) {
    AscendC::PairReduceSum(dst[rpt_times * one_rpt_cnt / 2], src[rpt_times * one_rpt_cnt], 1, tail_rpt_cnt, 1, 1, 8);
  }
}

template <typename T>
inline __aicore__ void CompareSingleLoopGT(const AscendC::LocalTensor<T> &dst,         // output
                                           const AscendC::LocalTensor<int64_t> &src0,  // input 0 is a tensor
                                           const AscendC::LocalTensor<int64_t> &src1,  // input 1 is a tensor
                                           CMPMODE mode,
                                           const uint32_t cal_cnt,  // total numbers
                                           AscendC::LocalTensor<uint8_t> &tmp_buf) {
  // Divide int64_t into 2 int32_t digits,
  // calculate the sign bits and value bits separately, and calculate as a whole in the end
  uint32_t quadruple_cal_cnt = 4 * cal_cnt;
  uint32_t double_cal_cnt = 2 * cal_cnt;
  // make sure every tensor address is aligned to 32B
  uint32_t offset_aligned = DivCeil(cal_cnt * sizeof(int64_t), ONE_BLK_SIZE) * ONE_BLK_SIZE;
  uint32_t offset = 0;
  // src0_bits is used to store the sign bits of src0, while the value bits are all zeros
  AscendC::LocalTensor<uint16_t> src0_bits = KernelUtils::NewTensor<uint16_t>(tmp_buf, 0, quadruple_cal_cnt);
  offset += offset_aligned;
  // src1_bits is used to store the sign bits of src0, while the value bits are all zeros
  AscendC::LocalTensor<uint16_t> src1_bits = KernelUtils::NewTensor<uint16_t>(tmp_buf, offset, quadruple_cal_cnt);
  offset += offset_aligned;
  // sub_res is used to store the compute result of src0_bits and src1_bits
  AscendC::LocalTensor<int32_t> sub_res = KernelUtils::NewTensor<int32_t>(tmp_buf, offset, double_cal_cnt);
  offset += offset_aligned;
  // sign_res is used to store the weighted compute result of sign bits
  AscendC::LocalTensor<half> sign_res = KernelUtils::NewTensor<half>(tmp_buf, offset, cal_cnt);
  offset += DivCeil(offset_aligned / 4, ONE_BLK_SIZE) * ONE_BLK_SIZE;
  // value_res is used to store the weighted compute result of value bits
  AscendC::LocalTensor<half> value_res = KernelUtils::NewTensor<half>(tmp_buf, offset, cal_cnt);
  offset += DivCeil(offset_aligned / 4, ONE_BLK_SIZE) * ONE_BLK_SIZE;
  // inner_dup is used for internal-compute
  AscendC::LocalTensor<uint32_t> inner_dup = KernelUtils::NewTensor<uint32_t>(tmp_buf, offset, double_cal_cnt);
  AscendC::LocalTensor<uint16_t> inner_dup_tmp = inner_dup.ReinterpretCast<uint16_t>();
  // step 1. Get src0 and src1 sign bit
  GetSignBitTensor(src0_bits, src1_bits, src0, src1, inner_dup, cal_cnt);
  // step 2. use src0_bits to store sign bit sub results
  AscendC::LocalTensor<int32_t> src0_bits_tmp = src0_bits.ReinterpretCast<int32_t>();
  AscendC::LocalTensor<int32_t> src1_bits_tmp = src1_bits.ReinterpretCast<int32_t>();
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::Sub(src0_bits_tmp, src0_bits_tmp, src1_bits_tmp, double_cal_cnt);
  // step 3. change sign bit sub results which represented as int32_t into half which only keep the results of sign bit
  CastTensorToHalf(sub_res, src0_bits, src1_bits, inner_dup, cal_cnt);

  // step 4. calculate the weighted values of sign bit and sum them up
  CalcWeightedTensor(sign_res, sub_res.ReinterpretCast<half>(), inner_dup, cal_cnt, half(8), half(2));
  // step 5. get the value bits of src0 and src1
  AscendC::Duplicate(inner_dup, uint32_t(0x7FFFFFFF), 2 * cal_cnt);
  AscendC::LocalTensor<uint16_t> src0_tmp = src0.ReinterpretCast<uint16_t>();
  AscendC::LocalTensor<uint16_t> src1_tmp = src1.ReinterpretCast<uint16_t>();
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::And(src0_bits, src0_tmp, inner_dup_tmp, quadruple_cal_cnt);
  AscendC::And(src1_bits, src1_tmp, inner_dup_tmp, quadruple_cal_cnt);
  // step 6. sub the two value bits interval, and change the result into 0/1/-1
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::Sub(src0_bits_tmp, src0_bits_tmp, src1_bits_tmp, double_cal_cnt);
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::Maxs(src0_bits_tmp, src0_bits_tmp, -1, double_cal_cnt);
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::Mins(src0_bits_tmp, src0_bits_tmp, 1, double_cal_cnt);
  // step 7. change value bits sub results which represented as int32_t into half which only keep the results of value
  // bits
  CastTensorToHalf(sub_res, src0_bits, src1_bits, inner_dup, cal_cnt);

  // step 8. calculate the weighted values of value bits and sum them up.
  CalcWeightedTensor(value_res, sub_res.ReinterpretCast<half>(), inner_dup, cal_cnt, half(4), half(1));

  // step 9. sum up the results of sign bit and value bits, change the result into 0/1
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::Add(sign_res, sign_res, value_res, cal_cnt);
  AscendC::PipeBarrier<PIPE_V>();
  if (mode == CMPMODE::GT) {
    AscendC::Maxs(sign_res, sign_res, half(0), cal_cnt);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mins(sign_res, sign_res, half(1), cal_cnt);
    // step 10. cast the final result into int8
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(dst, sign_res, AscendC::RoundMode::CAST_NONE, cal_cnt);
  } else {
    AscendC::Maxs(sign_res, sign_res, half(-1), cal_cnt);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mins(sign_res, sign_res, half(0), cal_cnt);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(sign_res, sign_res, half(1), cal_cnt);
    // step 10. cast the final result into int8
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(dst, sign_res, AscendC::RoundMode::CAST_NONE, cal_cnt);
  }
}

template <typename T>
inline __aicore__ void CompareExtendGT(const AscendC::LocalTensor<T> &dst,         // output
                                       const AscendC::LocalTensor<int64_t> &src0,  // input 0 is a tensor
                                       const AscendC::LocalTensor<int64_t> &src1,  // input 1 is a tensor
                                       CMPMODE mode,
                                       const uint32_t cal_cnt,  // total numbers
                                       AscendC::LocalTensor<uint8_t> &tmp_buf) {
  const uint32_t max_buf_cnt = tmp_buf.GetSize() / (5 * sizeof(uint64_t));
  const uint32_t one_max_rpt_cal_cnt = MAX_REPEAT_TIME * ONE_REPEAT_BYTE_SIZE / sizeof(int64_t);
  uint32_t max_cnt = KernelUtils::Min(max_buf_cnt, one_max_rpt_cal_cnt);
  max_cnt = max_cnt / ONE_BLK_SIZE * ONE_BLK_SIZE;  // make sure the dst tensor address each loop is aligned to 32B
  const uint32_t loop_num = cal_cnt / max_cnt;
  uint32_t offset = 0;
  for (uint32_t loop = 0; loop < loop_num; loop++) {
    CompareSingleLoopGT(dst[offset], src0[offset], src1[offset], mode, max_cnt, tmp_buf);
    offset += max_cnt;
  }
  uint32_t tail_cnt = cal_cnt - offset;
  if (tail_cnt > 0) {
    CompareSingleLoopGT(dst[offset], src0[offset], src1[offset], mode, tail_cnt, tmp_buf);
  }
}

template <typename T>
inline __aicore__ void CompareExtend(const AscendC::LocalTensor<T> &dst,         // output
                                     const AscendC::LocalTensor<int64_t> &src0,  // input 0 is a tensor
                                     const AscendC::LocalTensor<int64_t> &src1,  // input 1 is a tensor
                                     CMPMODE mode,                               // compare mode
                                     const uint32_t cal_cnt,                     // total numbers
                                     AscendC::LocalTensor<uint8_t> &tmp_buf) {
  if (mode != CMPMODE::EQ && mode != CMPMODE::NE && mode != CMPMODE::GT && mode != CMPMODE::GE && mode != CMPMODE::LE) {
    ASSERT(false && "CompareExtend mode only supports EQ/NE/GT/GE/LE when DataType is int64.");
  }

  if (mode == CMPMODE::EQ || mode == CMPMODE::NE) {
    CompareExtendEQ(dst, src0, src1, mode, cal_cnt, tmp_buf);
  } else if (mode == CMPMODE::LE) {
    CompareExtendGT(dst, src1, src0, CMPMODE::GE, cal_cnt, tmp_buf);
  } else {
    CompareExtendGT(dst, src0, src1, mode, cal_cnt, tmp_buf);
  }
}

#endif  // __ASCENDC_API_COMPARE_H__
