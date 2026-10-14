/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_CONCAT_H__
#define __ASCENDC_API_REGBASE_CONCAT_H__

namespace concat {
template <size_t INPUT_NUM>
struct ConcatTiling {
  uint32_t num_rows;
  uint32_t num_dst_cols;
  uint32_t num_srcs_cols[INPUT_NUM];
};

struct ConcatByGatherTiling {
  uint32_t num_rows;
  uint32_t num_dst_cols;
  uint32_t num_src_cols;
};

template <size_t INPUT_NUM>
struct ConcatTilingPadded {
  uint32_t num_rows;
  uint32_t num_dst_cols;
  uint32_t num_srcs_cols[INPUT_NUM];
  uint32_t src_row_strides[INPUT_NUM];
  uint32_t src_second_last_dim_strides[INPUT_NUM];
  uint32_t gather_mask_dim_sizes[INPUT_NUM];
};

template <size_t INPUT_NUM>
struct ConcatTilingOneAxis {
  uint32_t src_col_sizes[INPUT_NUM];
  uint32_t dst_col_offsets[INPUT_NUM];
};

__aicore__ inline uint32_t NumSrcCols(const ConcatByGatherTiling &tiling) {
  return tiling.num_src_cols;
}

template <size_t INPUT_NUM>
__aicore__ inline uint32_t NumSrcCols(const ConcatTiling<INPUT_NUM> &tiling) {
  return tiling.num_srcs_cols[0];
}

template <typename T>
struct ArangeTypeGet {};

template <>
struct ArangeTypeGet<uint32_t> {
  using T = int32_t;
};

template <>
struct ArangeTypeGet<uint16_t> {
  using T = int16_t;
};

template <typename U>
__aicore__ inline void GenSequence(__ubuf__ U *seq_addr) {
  uint32_t kVfLen = AscendC::VECTOR_REG_WIDTH / sizeof(U);
  using SeqType = typename ArangeTypeGet<U>::T;
  auto seq_buf_addr = (__ubuf__ SeqType *)seq_addr;
  __VEC_SCOPE__ {
    AscendC::MicroAPI::RegTensor<SeqType> reg_seq;
    AscendC::MicroAPI::Arange(reg_seq, 0);
    uint32_t num = kVfLen;
    AscendC::MicroAPI::MaskReg p0 = AscendC::MicroAPI::UpdateMask<SeqType>(num);
    AscendC::MicroAPI::DataCopy(seq_buf_addr, reg_seq, p0);
  }
}

template <typename T>
__aicore__ inline void Copy(__ubuf__ T *dst_addr_base, __ubuf__ T *src_addr_base, uint32_t rows, uint32_t cols,
                            uint32_t row_stride) {
  constexpr uint32_t kVfLen = AscendC::VECTOR_REG_WIDTH / sizeof(T);
  auto num_rows = static_cast<uint16_t>(rows);
  uint16_t repeat_times = cols / kVfLen;
  uint32_t tail_cols = cols - repeat_times * kVfLen;

  auto src_addr = src_addr_base;
  __VEC_SCOPE__ {
    AscendC::MicroAPI::UnalignReg u0;
    AscendC::MicroAPI::UnalignReg u_reg;
    AscendC::MicroAPI::RegTensor<T> vd0;
    for (uint16_t i = 0; i < num_rows; ++i) {
      auto dst_addr = dst_addr_base + i * row_stride;
      for (uint16_t j = 0; j < repeat_times; ++j) {
        AscendC::MicroAPI::DataCopyUnAlignPre(u0, src_addr);
        AscendC::MicroAPI::DataCopyUnAlign(vd0, u0, src_addr, kVfLen);
        AscendC::MicroAPI::DataCopyUnAlign(dst_addr, vd0, u_reg, kVfLen);
      }
      AscendC::MicroAPI::DataCopyUnAlignPre(u0, src_addr);
      AscendC::MicroAPI::DataCopyUnAlign(vd0, u0, src_addr, tail_cols);
      AscendC::MicroAPI::DataCopyUnAlign(dst_addr, vd0, u_reg, tail_cols);
      AscendC::MicroAPI::DataCopyUnAlignPost(dst_addr, u_reg, 0);
    }
  }
}

template <typename T>
__aicore__ inline void CopyPadded(__ubuf__ T *dst_addr_base, __ubuf__ T *src_addr_base, uint32_t rows,
                                  uint32_t src_row_stride, uint32_t row_stride, uint32_t gather_mask_repeat_stride,
                                  uint32_t gather_mask_dim_size) {
  constexpr uint32_t kVfLen = AscendC::VECTOR_REG_WIDTH / sizeof(T);
  auto num_rows = static_cast<uint16_t>(rows);
  uint16_t repeat_times = src_row_stride / gather_mask_repeat_stride;
  uint16_t sub_repeat_times = gather_mask_dim_size / kVfLen;
  uint16_t sub_tail_cols = gather_mask_dim_size - sub_repeat_times * kVfLen;
  __VEC_SCOPE__ {
    AscendC::MicroAPI::UnalignReg u0;
    AscendC::MicroAPI::UnalignReg u_reg;
    AscendC::MicroAPI::RegTensor<T> vd0;
    AscendC::MicroAPI::RegTensor<T> vd1;
    for (uint16_t i = 0; i < num_rows; ++i) {
      auto dst_addr = dst_addr_base + i * row_stride;
      for (uint16_t j = 0; j < repeat_times; ++j) {
        auto src_addr = src_addr_base + i * src_row_stride + j * gather_mask_repeat_stride;
        for (uint16_t k = 0; k < sub_repeat_times; ++k) {
          AscendC::MicroAPI::DataCopyUnAlignPre(u0, src_addr);
          AscendC::MicroAPI::DataCopyUnAlign(vd0, u0, src_addr, kVfLen);
          AscendC::MicroAPI::DataCopyUnAlign(dst_addr, vd0, u_reg, kVfLen);
        }
        AscendC::MicroAPI::DataCopyUnAlignPre(u0, src_addr);
        AscendC::MicroAPI::DataCopyUnAlign(vd0, u0, src_addr, sub_tail_cols);
        AscendC::MicroAPI::DataCopyUnAlign(dst_addr, vd0, u_reg, sub_tail_cols);
        AscendC::MicroAPI::DataCopyUnAlignPost(dst_addr, u_reg, 0);
      }
    }
  }
}

template <typename U>
__aicore__ inline void GenScatterIndex(uint32_t src_cols, uint32_t dst_cols, uint32_t dst_col_offset,
                                       __ubuf__ U *seq_addr, __ubuf__ U *index_addr) {
  constexpr uint32_t kVfLen = AscendC::VECTOR_REG_WIDTH / sizeof(U);
  __VEC_SCOPE__ {
    AscendC::MicroAPI::RegTensor<U> v0, v1, v2, v3;
    AscendC::MicroAPI::RegTensor<U> vd0, vd2, vd3, vd6, vd7, vd10;

    auto num = kVfLen;
    AscendC::MicroAPI::MaskReg p0 = AscendC::MicroAPI::UpdateMask<U>(num);
    AscendC::MicroAPI::DataCopy(v0, seq_addr);
    AscendC::MicroAPI::Duplicate(v1, (U)src_cols, p0);
    AscendC::MicroAPI::Duplicate(v2, (U)dst_col_offset, p0);
    AscendC::MicroAPI::Duplicate(v3, (U)dst_cols, p0);
    AscendC::MicroAPI::Div(vd2, v0, v1, p0);
    AscendC::MicroAPI::Mul(vd6, vd2, v1, p0);
    AscendC::MicroAPI::Sub(vd7, v0, vd6, p0);
    AscendC::MicroAPI::Mul(vd0, vd2, v3, p0);
    AscendC::MicroAPI::Add(vd3, vd0, vd7, p0);
    AscendC::MicroAPI::Add(vd10, vd3, v2, p0);
    AscendC::MicroAPI::DataCopy(index_addr, vd10, p0);
  }
}

template <typename U>
__aicore__ inline void GenGatherIndex(uint32_t src_cols, uint32_t dst_cols, uint32_t stride, __ubuf__ U *index_addr) {
  constexpr uint32_t kVfLen = AscendC::VECTOR_REG_WIDTH / sizeof(U);
  __VEC_SCOPE__ {
    using SeqType = typename ArangeTypeGet<U>::T;
    AscendC::MicroAPI::RegTensor<SeqType> reg_seq;
    AscendC::MicroAPI::RegTensor<U> v0, v1, v2, v3;
    AscendC::MicroAPI::RegTensor<U> vd0, vd1, vd2, vd3, vd4, vd5, vd6, vd7, vd8, vd9, vd10;

    auto num = kVfLen;
    AscendC::MicroAPI::MaskReg p0 = AscendC::MicroAPI::UpdateMask<U>(num);
    AscendC::MicroAPI::Arange(reg_seq, 0);
    v0 = (AscendC::MicroAPI::RegTensor<U> &)(reg_seq);
    AscendC::MicroAPI::Duplicate(v1, (U)src_cols, p0);
    AscendC::MicroAPI::Duplicate(v2, (U)stride, p0);
    AscendC::MicroAPI::Duplicate(v3, (U)dst_cols, p0);

    AscendC::MicroAPI::Div(vd2, v0, v3, p0);  // dst_row_index
    AscendC::MicroAPI::Muls(vd6, vd2, (U)dst_cols, p0);
    AscendC::MicroAPI::Sub(vd7, v0, vd6, p0);  // dst_col_index

    AscendC::MicroAPI::Div(vd0, vd7, v1, p0);          // src index
    AscendC::MicroAPI::Muls(vd1, vd0, (U)stride, p0);  // src offset

    AscendC::MicroAPI::Div(vd3, vd7, v1, p0);  // src index
    AscendC::MicroAPI::Muls(vd4, vd3, (U)src_cols, p0);
    AscendC::MicroAPI::Sub(vd5, vd7, vd4, p0);  // col_index

    AscendC::MicroAPI::Muls(vd8, vd2, (U)src_cols, p0);
    AscendC::MicroAPI::Add(vd9, vd1, vd5, p0);
    AscendC::MicroAPI::Add(vd10, vd8, vd9, p0);
    AscendC::MicroAPI::DataCopy(index_addr, vd10, p0);
  }
}

template <typename T, typename U>
__aicore__ inline void ScatterInput(__ubuf__ T *dst_addr, __ubuf__ T *src_addr, __ubuf__ U *index_addr, uint32_t rows,
                                    uint32_t dst_cols, uint32_t src_cols) {
  constexpr uint32_t kVfLen = AscendC::VECTOR_REG_WIDTH / sizeof(U);
  auto rows_per_loop = static_cast<uint16_t>(kVfLen / src_cols);
  auto loop_times = static_cast<uint16_t>(rows / rows_per_loop);
  uint16_t tail_rows = rows - loop_times * rows_per_loop;

  __VEC_SCOPE__ {
    AscendC::MicroAPI::RegTensor<T> vd2;
    AscendC::MicroAPI::RegTensor<T> src;
    AscendC::MicroAPI::RegTensor<T> tmp;
    AscendC::MicroAPI::RegTensor<T> dst0;
    AscendC::MicroAPI::RegTensor<T> dst1;
    AscendC::MicroAPI::RegTensor<U> vd0;
    AscendC::MicroAPI::RegTensor<U> vd1;
    AscendC::MicroAPI::UnalignReg u0;

    uint32_t num = rows_per_loop * src_cols;
    uint32_t tail_num = tail_rows * src_cols;
    uint32_t pnum = num;
    uint32_t tail_pnum = tail_num;
    AscendC::MicroAPI::MaskReg p0 = AscendC::MicroAPI::UpdateMask<U>(num);
    AscendC::MicroAPI::MaskReg p1 = AscendC::MicroAPI::UpdateMask<U>(tail_num);

    AscendC::MicroAPI::DataCopy(vd0, index_addr);
    AscendC::MicroAPI::DataCopyUnAlignPre(u0, src_addr);
    for (uint16_t i = 0; i < loop_times; ++i) {
      AscendC::MicroAPI::DataCopyUnAlign(vd2, u0, src_addr, pnum);
      AscendC::MicroAPI::Adds(vd1, vd0, static_cast<U>(i * rows_per_loop * dst_cols), p0);
      if constexpr (sizeof(T) == 1) {
        AscendC::MicroAPI::Interleave(dst0, dst1, vd2, tmp);
        AscendC::MicroAPI::DataCopyScatter(dst_addr, dst0, vd1, p0);
      } else {
        AscendC::MicroAPI::DataCopyScatter(dst_addr, vd2, vd1, p0);
      }
    }
    AscendC::MicroAPI::DataCopyUnAlign(vd2, u0, src_addr, tail_pnum);
    AscendC::MicroAPI::Adds(vd1, vd0, static_cast<U>(loop_times * rows_per_loop * dst_cols), p1);
    if constexpr (sizeof(T) == 1) {
      AscendC::MicroAPI::Interleave(dst0, dst1, vd2, tmp);
      AscendC::MicroAPI::DataCopyScatter(dst_addr, dst0, vd1, p1);
    } else {
      AscendC::MicroAPI::DataCopyScatter(dst_addr, vd2, vd1, p1);
    }
  }
}

template <typename U>
__aicore__ inline AscendC::MicroAPI::MaskReg GenMaskReg(uint32_t block_size, uint32_t gather_mask_repeat_stride,
                                                        uint32_t gather_mask_dim_size, __ubuf__ uint32_t *index_addr) {
  constexpr uint32_t kVfLen = VECTOR_REG_WIDTH / sizeof(U);
  constexpr uint16_t kDataBlockSize = 32 / sizeof(U);
  const uint16_t loop_times = static_cast<uint16_t>(kVfLen / block_size);
  // 不对齐, 必然有tail
  const uint16_t all_num = (gather_mask_repeat_stride / kDataBlockSize) - 1;
  const uint16_t tail_num = gather_mask_dim_size % (32 / sizeof(U));
  const uint32_t tail_mask = (1U << (tail_num * sizeof(U))) - 1U;
  if (all_num == 0) {
    for (uint16_t i = 0; i < loop_times; ++i) {
      index_addr[i] = tail_mask;
    }
    for (uint16_t i = loop_times; i < 8; ++i) {
      index_addr[i] = 0;
    }
  } else {
    uint16_t index = 0;
    for (uint16_t i = 0; i < loop_times; ++i) {
      for (uint16_t j = 0; j < all_num; ++j) {
        index_addr[index++] = 0xffffffffU;
      }
      index_addr[index++] = tail_mask;
    }
    for (; index < 8; ++index) {
      index_addr[index] = 0;
    }
  }

  MicroAPI::MaskReg p_copy_mask = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALLF>();
  MicroAPI::DataCopy(p_copy_mask, index_addr);
  return p_copy_mask;
}

template <typename T, typename U>
__aicore__ inline void ScatterInputPadded(__ubuf__ T *dst_addr, __ubuf__ T *src_addr, __ubuf__ U *index_addr,
                                          uint32_t rows, uint32_t dst_cols, uint32_t src_cols, uint32_t src_row_stride,
                                          uint32_t gather_mask_repeat_stride, uint32_t gather_mask_dim_size) {
  constexpr uint32_t kVfLen = VECTOR_REG_WIDTH / sizeof(U);
  auto rows_per_loop = static_cast<uint16_t>(kVfLen / src_row_stride);
  auto loop_times = static_cast<uint16_t>(rows / rows_per_loop);
  uint16_t tail_rows = rows - loop_times * rows_per_loop;
  __VEC_SCOPE__ {
    MicroAPI::RegTensor<T> vd_src;
    MicroAPI::RegTensor<T> vd_src_unpadded;
    MicroAPI::RegTensor<T> tmp;
    MicroAPI::RegTensor<T> dst0;
    MicroAPI::RegTensor<T> dst1;
    MicroAPI::RegTensor<U> vd_dst_index_base;
    MicroAPI::RegTensor<U> vd_dst_index;

    uint32_t num = rows_per_loop * src_cols;
    uint32_t tail_num = tail_rows * src_cols;
    MicroAPI::MaskReg p0 = MicroAPI::UpdateMask<U>(num);
    MicroAPI::MaskReg p1 = MicroAPI::UpdateMask<U>(tail_num);
    uint32_t main_num = rows_per_loop * src_row_stride;

    auto tmp_buf_addr = (__ubuf__ uint32_t *)(index_addr + kVfLen);
    MicroAPI::MaskReg copy_in_mask =
        GenMaskReg<U>(src_row_stride, gather_mask_repeat_stride, gather_mask_dim_size, tmp_buf_addr);
    MicroAPI::DataCopy(vd_dst_index_base, index_addr);

    for (uint16_t i = 0; i < loop_times; ++i) {
      MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vd_src, src_addr, main_num);
      MicroAPI::GatherMask(vd_src_unpadded, vd_src, copy_in_mask);
      MicroAPI::Adds(vd_dst_index, vd_dst_index_base, static_cast<U>(i * rows_per_loop * dst_cols), p0);
      if constexpr (sizeof(T) == 1) {
        MicroAPI::Interleave(dst0, dst1, vd_src_unpadded, tmp);
        MicroAPI::DataCopyScatter(dst_addr, dst0, vd_dst_index, p0);
      } else {
        MicroAPI::DataCopyScatter(dst_addr, vd_src_unpadded, vd_dst_index, p0);
      }
    }
    if (tail_rows > 0) {
      MicroAPI::DataCopy(vd_src, src_addr);
      MicroAPI::GatherMask(vd_src_unpadded, vd_src, copy_in_mask);
      MicroAPI::Adds(vd_dst_index, vd_dst_index_base, static_cast<U>(loop_times * rows_per_loop * dst_cols), p1);
      if constexpr (sizeof(T) == 1) {
        MicroAPI::Interleave(dst0, dst1, vd_src_unpadded, tmp);
        MicroAPI::DataCopyScatter(dst_addr, dst0, vd_dst_index, p1);
      } else {
        MicroAPI::DataCopyScatter(dst_addr, vd_src_unpadded, vd_dst_index, p1);
      }
    }
  }
}

template <typename T, size_t INPUT_NUM>
__aicore__ inline void ConcatExtend(__ubuf__ T *dst_addr, __ubuf__ T *src_addrs[INPUT_NUM],
                                    LocalTensor<uint8_t> &tmp_buf, const ConcatTiling<INPUT_NUM> &tiling) {
  using U = std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t, uint16_t>;
  const uint32_t kScatterMaxLen = VECTOR_REG_WIDTH / sizeof(T) / 2;
  auto seq_tensor = tmp_buf[512].ReinterpretCast<U>();
  auto index_tensor = tmp_buf.ReinterpretCast<U>();
  auto seq_addr = (__ubuf__ U *)seq_tensor.GetPhyAddr();
  auto index_addr = (__ubuf__ U *)index_tensor.GetPhyAddr();

  GenSequence(seq_addr);
  uint32_t dst_col_offset = 0;
  for (size_t i = 0UL; i < INPUT_NUM; ++i) {
    if (tiling.num_srcs_cols[i] > kScatterMaxLen) {
      Copy(dst_addr + dst_col_offset, src_addrs[i], tiling.num_rows, tiling.num_srcs_cols[i], tiling.num_dst_cols);
    } else {
      GenScatterIndex(tiling.num_srcs_cols[i], tiling.num_dst_cols, dst_col_offset, seq_addr, index_addr);
      ScatterInput(dst_addr, src_addrs[i], index_addr, tiling.num_rows, tiling.num_dst_cols, tiling.num_srcs_cols[i]);
    }
    dst_col_offset += tiling.num_srcs_cols[i];
  }
}

template <typename T, size_t INPUT_NUM>
__aicore__ inline void ConcatExtend(__ubuf__ T *dst_addr, __ubuf__ T *src_addrs[INPUT_NUM],
                                    LocalTensor<uint8_t> &tmp_buf, const ConcatTilingPadded<INPUT_NUM> &tiling) {
  using U = std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t, uint16_t>;
  const uint32_t kScatterMaxLen = VECTOR_REG_WIDTH / sizeof(T) / 2;
  auto seq_tensor = tmp_buf[512].ReinterpretCast<U>();
  auto index_tensor = tmp_buf.ReinterpretCast<U>();
  auto seq_addr = (__ubuf__ U *)seq_tensor.GetPhyAddr();
  auto index_addr = (__ubuf__ U *)index_tensor.GetPhyAddr();

  GenSequence(seq_addr);
  uint32_t dst_col_offset = 0;
  for (size_t i = 0UL; i < INPUT_NUM; ++i) {
    if (tiling.num_srcs_cols[i] == tiling.src_row_strides[i]) {
      if (tiling.num_srcs_cols[i] > kScatterMaxLen) {
        Copy(dst_addr + dst_col_offset, src_addrs[i], tiling.num_rows, tiling.num_srcs_cols[i], tiling.num_dst_cols);
      } else {
        GenScatterIndex(tiling.num_srcs_cols[i], tiling.num_dst_cols, dst_col_offset, seq_addr, index_addr);
        ScatterInput(dst_addr, src_addrs[i], index_addr, tiling.num_rows, tiling.num_dst_cols, tiling.num_srcs_cols[i]);
      }
    } else {
      if (tiling.src_row_strides[i] > kScatterMaxLen) {
        CopyPadded(dst_addr + dst_col_offset, src_addrs[i], tiling.num_rows, tiling.src_row_strides[i],
                   tiling.num_dst_cols, tiling.src_second_last_dim_strides[i], tiling.gather_mask_dim_sizes[i]);
      } else {
        GenScatterIndex(tiling.num_srcs_cols[i], tiling.num_dst_cols, dst_col_offset, seq_addr, index_addr);
        ScatterInputPadded(dst_addr, src_addrs[i], index_addr, tiling.num_rows, tiling.num_dst_cols,
                           tiling.num_srcs_cols[i], tiling.src_row_strides[i], tiling.src_second_last_dim_strides[i],
                           tiling.gather_mask_dim_sizes[i]);
      }
    }
    dst_col_offset += tiling.num_srcs_cols[i];
  }
}

template <typename T, typename U>
__aicore__ inline void GatherInputs(__ubuf__ T *dst_addr, __ubuf__ T *src_addr, __ubuf__ U *index_addr, uint32_t rows,
                                    uint32_t dst_cols, uint32_t src_cols) {
  constexpr uint32_t kVfLen = AscendC::VECTOR_REG_WIDTH / sizeof(U);
  auto rows_per_loop = static_cast<uint16_t>(kVfLen / dst_cols);
  auto loop_times = static_cast<uint16_t>(rows / rows_per_loop);
  auto stride = rows_per_loop * src_cols;
  uint16_t tail_rows = rows - loop_times * rows_per_loop;

  __VEC_SCOPE__ {
    AscendC::MicroAPI::RegTensor<U> vd0;
    AscendC::MicroAPI::RegTensor<U> vd1;
    AscendC::MicroAPI::RegTensor<T> vd2;
    AscendC::MicroAPI::RegTensor<uint16_t> vd3;  // for b8
    AscendC::MicroAPI::RegTensor<U> vd4;
    AscendC::MicroAPI::RegTensor<U> vd5;
    AscendC::MicroAPI::UnalignReg u0;

    uint32_t num = rows_per_loop * dst_cols;
    uint32_t tail_num = tail_rows * dst_cols;
    uint32_t pnum = num;
    uint32_t tail_pnum = tail_num;
    AscendC::MicroAPI::MaskReg p0 = AscendC::MicroAPI::UpdateMask<U>(num);
    AscendC::MicroAPI::MaskReg p1 = AscendC::MicroAPI::UpdateMask<U>(tail_num);

    AscendC::MicroAPI::DataCopy(vd0, index_addr);
    for (uint16_t i = 0; i < loop_times; ++i) {
      AscendC::MicroAPI::Adds(vd1, vd0, static_cast<U>(i * stride), p0);
      if constexpr (sizeof(T) == 1) {
        AscendC::MicroAPI::DataCopyGather(vd3, src_addr, vd1, p0);
        AscendC::MicroAPI::Pack(vd2, vd3);
      } else {
        AscendC::MicroAPI::DataCopyGather(vd2, src_addr, vd1, p0);
      }
      AscendC::MicroAPI::DataCopyUnAlign(dst_addr, vd2, u0, pnum);
      AscendC::MicroAPI::DataCopyUnAlignPost(dst_addr, u0, 0);
    }
    AscendC::MicroAPI::Adds(vd1, vd0, static_cast<U>(loop_times * stride), p1);
    if constexpr (sizeof(T) == 1) {
      AscendC::MicroAPI::DataCopyGather(vd3, src_addr, vd1, p1);
      AscendC::MicroAPI::Pack(vd2, vd3);
    } else {
      AscendC::MicroAPI::DataCopyGather(vd2, src_addr, vd1, p1);
    }
    AscendC::MicroAPI::DataCopyUnAlign(dst_addr, vd2, u0, tail_pnum);
    AscendC::MicroAPI::DataCopyUnAlignPost(dst_addr, u0, 0);
  }
}

template <typename T, typename U>
__aicore__ inline void GatherInputsSplitCols(__ubuf__ T *dst_addr, __ubuf__ T *src_addr, __ubuf__ U *index_addr,
                                             uint32_t rows, uint32_t dst_cols, uint32_t src_cols,
                                             uint32_t tensor_stride) {
  constexpr uint32_t kVfLen = AscendC::VECTOR_REG_WIDTH / sizeof(U);
  const auto loop_times = static_cast<uint16_t>(rows);
  const uint16_t loop_1_input_num = kVfLen / src_cols;
  const uint16_t loop_1_cols = loop_1_input_num * src_cols;
  const uint16_t loop_1_times = dst_cols / loop_1_cols;
  const uint16_t loop_1_tail_cols = dst_cols - loop_1_times * loop_1_cols;

  __VEC_SCOPE__ {
    AscendC::MicroAPI::RegTensor<U> vd0;
    AscendC::MicroAPI::RegTensor<U> vd1;
    AscendC::MicroAPI::RegTensor<T> vd2;
    AscendC::MicroAPI::RegTensor<uint16_t> vd3;  // for b8
    AscendC::MicroAPI::RegTensor<U> vd4;
    AscendC::MicroAPI::RegTensor<U> vd5;
    AscendC::MicroAPI::UnalignReg u0;
    AscendC::MicroAPI::UnalignReg u1;

    AscendC::MicroAPI::DataCopy(vd0, index_addr);
    auto num = static_cast<uint32_t>(loop_1_cols);
    auto tail_num = static_cast<uint32_t>(loop_1_tail_cols);
    uint32_t pnum = num;
    uint32_t tail_pnum = tail_num;
    AscendC::MicroAPI::MaskReg p0 = AscendC::MicroAPI::UpdateMask<U>(num);
    AscendC::MicroAPI::MaskReg p1 = AscendC::MicroAPI::UpdateMask<U>(tail_num);
    const uint32_t stride = loop_1_input_num * tensor_stride;
    for (uint16_t i = 0; i < loop_times; ++i) {
      auto cur_dst_addr = dst_addr + i * dst_cols;
      const auto row_offset = i * src_cols;
      for (uint16_t j = 0; j < loop_1_times; ++j) {
        AscendC::MicroAPI::Adds(vd1, vd0, static_cast<U>(row_offset + j * stride), p0);
        if constexpr (sizeof(T) == 1) {
          AscendC::MicroAPI::DataCopyGather(vd3, src_addr, vd1, p0);
          AscendC::MicroAPI::Pack(vd2, vd3);
        } else {
          AscendC::MicroAPI::DataCopyGather(vd2, src_addr, vd1, p0);
        }
        AscendC::MicroAPI::DataCopyUnAlign(cur_dst_addr, vd2, u0, pnum);
        AscendC::MicroAPI::DataCopyUnAlignPost(cur_dst_addr, u0, 0);
      }
      AscendC::MicroAPI::Adds(vd1, vd0, static_cast<U>(row_offset + loop_1_times * stride), p1);
      if constexpr (sizeof(T) == 1) {
        AscendC::MicroAPI::DataCopyGather(vd3, src_addr, vd1, p1);
        AscendC::MicroAPI::Pack(vd2, vd3);
      } else {
        AscendC::MicroAPI::DataCopyGather(vd2, src_addr, vd1, p1);
      }
      AscendC::MicroAPI::DataCopyUnAlign(cur_dst_addr, vd2, u1, tail_pnum);
      AscendC::MicroAPI::DataCopyUnAlignPost(cur_dst_addr, u1, 0);
    }
  }
}

template <typename T, size_t INPUT_NUM, typename TilingType>
__aicore__ inline void ConcatExtendByGather(__ubuf__ T *dst_addr, __ubuf__ T *src_addrs[INPUT_NUM],
                                            LocalTensor<uint8_t> &tmp_buf, const TilingType &tiling) {
  using U = std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t, uint16_t>;
  auto seq_tensor = tmp_buf[512].ReinterpretCast<U>();
  auto index_tensor = tmp_buf.ReinterpretCast<U>();
  auto seq_addr = (__ubuf__ U *)seq_tensor.GetPhyAddr();
  auto index_addr = (__ubuf__ U *)index_tensor.GetPhyAddr();

  GenSequence(seq_addr);
  const auto stride = static_cast<uint32_t>(src_addrs[1] - src_addrs[0]);
  const auto num_src_cols = NumSrcCols(tiling);
  GenGatherIndex(num_src_cols, tiling.num_dst_cols, stride, index_addr);
  if (tiling.num_dst_cols * sizeof(U) <= AscendC::VECTOR_REG_WIDTH) {
    GatherInputs(dst_addr, src_addrs[0], index_addr, tiling.num_rows, tiling.num_dst_cols, num_src_cols);
  } else {
    GatherInputsSplitCols(dst_addr, src_addrs[0], index_addr, tiling.num_rows, tiling.num_dst_cols, num_src_cols,
                          stride);
  }
}

template <typename T, size_t INPUT_NUM, bool CHECK_SHAPE = false>
__aicore__ inline bool CanUseGather(const ConcatTiling<INPUT_NUM> &tiling) {
  using U = std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t, uint16_t>;
  if (tiling.num_srcs_cols[0] * sizeof(U) > (AscendC::VECTOR_REG_WIDTH / 2)) {
    return false;
  }
  if constexpr (CHECK_SHAPE) {
    const auto first_col_size = tiling.num_srcs_cols[0];
    for (size_t i = 1; i < INPUT_NUM; i++) {
      if (tiling.num_srcs_cols[i] != first_col_size) {
        return false;
      }
    }
  }
  return true;
}

template <typename T, size_t INPUT_NUM, bool CHECK_SHAPE = false>
__aicore__ inline void ConcatExtendDyn(__ubuf__ T *dst_addr, __ubuf__ T *src_addrs[INPUT_NUM],
                                       AscendC::LocalTensor<uint8_t> &tmp_buf, const ConcatTiling<INPUT_NUM> &tiling) {
  if (CanUseGather<T, INPUT_NUM, CHECK_SHAPE>(tiling)) {
    ConcatExtendByGather<T, INPUT_NUM>(dst_addr, src_addrs, tmp_buf, tiling);
  } else {
    ConcatExtend<T, INPUT_NUM>(dst_addr, src_addrs, tmp_buf, tiling);
  }
}

template <typename T>
__aicore__ inline void CopyOneAxis(__ubuf__ T *dst_addr, __ubuf__ T *src_addr, uint32_t size) {
  __VEC_SCOPE__ {
    AscendC::MicroAPI::UnalignReg u0;
    AscendC::MicroAPI::RegTensor<T> vd0;
    AscendC::MicroAPI::DataCopy(vd0, src_addr);
    AscendC::MicroAPI::DataCopyUnAlign(dst_addr, vd0, u0, size);
    AscendC::MicroAPI::DataCopyUnAlignPost(dst_addr, u0, 0);
  }
}

template <typename T, size_t INPUT_NUM, size_t... Is>
inline __aicore__ void ConcatOneAxisUnroll(__ubuf__ T *dst_addr, __ubuf__ T *src_addrs[INPUT_NUM],
                                           const ConcatTilingOneAxis<INPUT_NUM> &tiling, std::index_sequence<Is...>) {
  ((CopyOneAxis(dst_addr + tiling.dst_col_offsets[Is], src_addrs[Is], tiling.src_col_sizes[Is])), ...);
}

template <typename T, size_t INPUT_NUM>
inline __aicore__ void ConcatOneAxis(__ubuf__ T *dst_addr, __ubuf__ T *src_addrs[INPUT_NUM],
                                     const ConcatTilingOneAxis<INPUT_NUM> &tiling) {
  ConcatOneAxisUnroll(dst_addr, src_addrs, tiling, std::make_index_sequence<INPUT_NUM>{});
}
}  // namespace concat

template <size_t INPUT_NUM>
struct ConcatTilingAllAligned {
  uint32_t dst_col_size;
  uint32_t src_col_sizes[INPUT_NUM];
  uint32_t dst_offsets[INPUT_NUM];
};

template <typename T, uint32_t INPUT_NUM>
inline __aicore__ void ConcatAllAligned(uint32_t num_rows, const ConcatTilingAllAligned<INPUT_NUM> &tiling,
                                        LocalTensor<T> &dst_tensor, LocalTensor<T> (&src_tensors)[INPUT_NUM]) {
  constexpr uint32_t kDataBlockSize = 32U;
  constexpr auto align_size = static_cast<uint16_t>(kDataBlockSize / sizeof(T));
#pragma unroll
  for (uint32_t i = 0U; i < INPUT_NUM; ++i) {
    const auto size = tiling.src_col_sizes[i];
    DataCopyParams copy_params{static_cast<uint16_t>(num_rows), static_cast<uint16_t>(size / align_size), 0,
                               static_cast<uint16_t>((tiling.dst_col_size - size) / align_size)};
    DataCopy(dst_tensor[tiling.dst_offsets[i]], src_tensors[i], copy_params);
  }
}

#endif  // __ASCENDC_API_REGBASE_CONCAT_H__
