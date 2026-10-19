/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_SPLIT_H__
#define __ASCENDC_API_REGBASE_SPLIT_H__

namespace split {
template <size_t OUTPUT_NUM>
struct SplitTiling {
  uint32_t num_rows;
  uint32_t num_src_cols;
  uint32_t num_dsts_cols[OUTPUT_NUM];
};

template <typename T>
__aicore__ inline void SplitCopy(__ubuf__ T *src_addr_base, __ubuf__ T *dst_addr_base, uint32_t rows, uint32_t cols,
                                 uint32_t row_stride) {
  constexpr uint32_t vfLen1 = AscendC::VECTOR_REG_WIDTH / sizeof(T);
  auto num_rows = static_cast<uint16_t>(rows);
  uint16_t repeat_times = cols / vfLen1;
  uint32_t tail_cols = cols - repeat_times * vfLen1;

  auto dst_addr = dst_addr_base;
  __VEC_SCOPE__ {
    AscendC::MicroAPI::UnalignReg u0;
    AscendC::MicroAPI::UnalignReg u1;
    AscendC::MicroAPI::RegTensor<T> vd0;
    for (uint16_t i = 0; i < num_rows; i++) {
      __ubuf__ T *src_ptr1 = src_addr_base + i * row_stride;
      AscendC::MicroAPI::DataCopyUnAlignPre(u0, src_ptr1);
      for (uint16_t j = 0; j < repeat_times; j++) {
        AscendC::MicroAPI::DataCopyUnAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vd0, u0, src_ptr1,
                                                                                                vfLen1);
        AscendC::MicroAPI::DataCopyUnAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst_addr, vd0, u1,
                                                                                                vfLen1);
      }
      AscendC::MicroAPI::DataCopyUnAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(vd0, u0, src_ptr1,
                                                                                              vfLen1);
      AscendC::MicroAPI::DataCopyUnAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst_addr, vd0, u1,
                                                                                              tail_cols);
      AscendC::MicroAPI::DataCopyUnAlignPost<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst_addr, u1, 0);
    }
  }
}

template <typename T, typename U, typename Y>
__aicore__ inline void DataCopyGatherVf(__ubuf__ T *dst_addr, __ubuf__ T *src_addr, uint32_t rows, uint32_t src_cols,
                                        uint32_t dst_cols, uint32_t src_col_offset,
                                        AscendC::LocalTensor<uint8_t> &tmp_buf) {
  {
    // 索引/gather/mask 均以工作类型 U 计算 lane 数，b8 场景下 U 为 b16。
    constexpr uint32_t vflen = AscendC::VECTOR_REG_WIDTH / sizeof(U);
    uint32_t size0 = vflen / dst_cols;
    if (size0 > rows) {
      size0 = rows;
    }
    uint32_t offset = size0 * src_cols;
    uint32_t num = size0 * dst_cols;
    uint16_t times = (rows - size0) / size0;
    uint32_t tail = (rows - size0 - times * size0) * dst_cols;
    uint32_t mask1 = num;
    uint32_t mask2 = tail;

    __ubuf__ T *curDstPtr = dst_addr + num;
    __ubuf__ U *dstPtr = (__ubuf__ U *)dst_addr;

    __VEC_SCOPE__ {
      AscendC::MicroAPI::MaskReg p0;
      AscendC::MicroAPI::MaskReg p1;
      AscendC::MicroAPI::RegTensor<Y> indexReg;
      AscendC::MicroAPI::RegTensor<U> tmp;
      AscendC::MicroAPI::RegTensor<U> addReg;
      AscendC::MicroAPI::RegTensor<U> addReg1;
      AscendC::MicroAPI::RegTensor<U> dstReg;
      AscendC::MicroAPI::RegTensor<U> tmp1;
      AscendC::MicroAPI::RegTensor<U> tmp2;
      AscendC::MicroAPI::RegTensor<U> subReg;
      AscendC::MicroAPI::RegTensor<U> niReg;
      AscendC::MicroAPI::RegTensor<T> dstRegO;
      AscendC::MicroAPI::UnalignReg u0;
      p0 = AscendC::MicroAPI::UpdateMask<U>(mask1);
      AscendC::MicroAPI::Duplicate(niReg, (U)dst_cols, p0);
      AscendC::MicroAPI::Arange(indexReg, 0);
      AscendC::MicroAPI::Div(tmp, (AscendC::MicroAPI::RegTensor<U> &)indexReg, niReg, p0);
      AscendC::MicroAPI::Muls(tmp1, tmp, (U)src_cols, p0);
      AscendC::MicroAPI::Mul(subReg, tmp, niReg, p0);
      AscendC::MicroAPI::Sub(tmp2, (AscendC::MicroAPI::RegTensor<U> &)indexReg, subReg, p0);
      AscendC::MicroAPI::Add(addReg, tmp1, tmp2, p0);
      AscendC::MicroAPI::Adds(addReg, addReg, (U)src_col_offset, p0);
      AscendC::MicroAPI::DataCopyGather(dstReg, src_addr, addReg, p0);
      if constexpr (sizeof(T) == sizeof(int8_t)) {
        AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(dstPtr, dstReg, p0);
      } else {
        AscendC::MicroAPI::DataCopy(dstPtr, dstReg, p0);
      }
      for (uint16_t ii = 0; ii < times; ii++) {
        AscendC::MicroAPI::Adds(addReg1, addReg, (U)(offset * (ii + 1)), p0);
        AscendC::MicroAPI::DataCopyGather(dstReg, src_addr, addReg1, p0);
        if constexpr (sizeof(T) == sizeof(int8_t)) {
          AscendC::MicroAPI::Pack(dstRegO, dstReg);
          AscendC::MicroAPI::DataCopyUnAlign(curDstPtr, dstRegO, u0, num);
          AscendC::MicroAPI::DataCopyUnAlignPost(curDstPtr, u0, 0);
        } else {
          AscendC::MicroAPI::DataCopyUnAlign(curDstPtr, dstReg, u0, num);
          AscendC::MicroAPI::DataCopyUnAlignPost(curDstPtr, u0, 0);
        }
      }
      p1 = AscendC::MicroAPI::UpdateMask<U>(mask2);
      AscendC::MicroAPI::Adds(addReg1, addReg, (U)(offset * (times + 1)), p1);
      AscendC::MicroAPI::DataCopyGather(dstReg, src_addr, addReg1, p1);
      if constexpr (sizeof(T) == sizeof(int8_t)) {
        AscendC::MicroAPI::Pack(dstRegO, dstReg);
        AscendC::MicroAPI::DataCopyUnAlign(curDstPtr, dstRegO, u0, tail);
        AscendC::MicroAPI::DataCopyUnAlignPost(curDstPtr, u0, 0);
      } else {
        AscendC::MicroAPI::DataCopyUnAlign(curDstPtr, dstReg, u0, tail);
        AscendC::MicroAPI::DataCopyUnAlignPost(curDstPtr, u0, 0);
      }
    }
  }
}

template <typename T, typename U, typename Y, size_t OUTPUT_NUM>
__aicore__ inline void SplitExtendInner(__ubuf__ T *src_addr, __ubuf__ T *dst_addrs[OUTPUT_NUM],
                                        AscendC::LocalTensor<uint8_t> &tmp_buf, const SplitTiling<OUTPUT_NUM> &tiling) {
  const uint32_t kGatherMaxLen = AscendC::VECTOR_REG_WIDTH / sizeof(U);

  uint32_t src_col_offset = 0;
  for (size_t i = 0UL; i < OUTPUT_NUM; ++i) {
    if (tiling.num_dsts_cols[i] > kGatherMaxLen) {
      // by copy
      SplitCopy(src_addr + src_col_offset, dst_addrs[i], tiling.num_rows, tiling.num_dsts_cols[i], tiling.num_src_cols);
    } else {
      DataCopyGatherVf<T, U, Y>(dst_addrs[i], src_addr, tiling.num_rows, tiling.num_src_cols, tiling.num_dsts_cols[i],
                                src_col_offset, tmp_buf);
    }
    src_col_offset += tiling.num_dsts_cols[i];
  }
}

template <typename T, size_t OUTPUT_NUM>
__aicore__ inline void SplitExtend(__ubuf__ T *src_addr, __ubuf__ T *out_addrs[OUTPUT_NUM],
                                   AscendC::LocalTensor<uint8_t> &tmp_buf, const SplitTiling<OUTPUT_NUM> &tiling) {
  if constexpr (sizeof(T) == sizeof(uint32_t)) {
    SplitExtendInner<uint32_t, uint32_t, int32_t, OUTPUT_NUM>(reinterpret_cast<__ubuf__ uint32_t *>(src_addr),
                                                              reinterpret_cast<__ubuf__ uint32_t **>(out_addrs),
                                                              tmp_buf, tiling);
  } else if constexpr (sizeof(T) == sizeof(uint8_t)) {
    // b8 无法整体按 b16 处理时（存在奇数输出列），统一按无符号字节(uint8_t)处理，
    // 索引/gather 的工作类型提升为 b16(U)。SDK 的 Pack(vpack) 目标寄存器要求 b8 为
    // unsigned char，不能用 int8_t，故此处显式以 uint8_t 实例化（纯字节搬移，符号无关）。
    SplitExtendInner<uint8_t, uint16_t, int16_t, OUTPUT_NUM>(reinterpret_cast<__ubuf__ uint8_t *>(src_addr),
                                                             reinterpret_cast<__ubuf__ uint8_t **>(out_addrs), tmp_buf,
                                                             tiling);
  } else {
    SplitExtendInner<uint16_t, uint16_t, int16_t, OUTPUT_NUM>(reinterpret_cast<__ubuf__ uint16_t *>(src_addr),
                                                              reinterpret_cast<__ubuf__ uint16_t **>(out_addrs),
                                                              tmp_buf, tiling);
  }
}
}  // namespace split

template <size_t OUTPUT_NUM>
struct SplitTilingAllAligned {
  uint32_t src_col_size;
  uint32_t dst_col_sizes[OUTPUT_NUM];
  uint32_t src_offsets[OUTPUT_NUM];
};

template <typename T, uint32_t OUTPUT_NUM>
inline __aicore__ void SplitAllAligned(uint32_t num_rows, const SplitTilingAllAligned<OUTPUT_NUM> &tiling,
                                       LocalTensor<T> &src_tensor, LocalTensor<T> (&dst_tensors)[OUTPUT_NUM]) {
  constexpr uint32_t kDataBlockSize = 32U;
  constexpr auto align_size = static_cast<uint16_t>(kDataBlockSize / sizeof(T));
#pragma unroll
  for (uint32_t i = 0U; i < OUTPUT_NUM; ++i) {
    const auto size = tiling.dst_col_sizes[i];
    DataCopyParams copy_params{static_cast<uint16_t>(num_rows), static_cast<uint16_t>(size / align_size),
                               static_cast<uint16_t>((tiling.src_col_size - size) / align_size), 0};
    DataCopy(dst_tensors[i], src_tensor[tiling.src_offsets[i]], copy_params);
  }
}

#endif  // __ASCENDC_API_REGBASE_SPLIT_H__
