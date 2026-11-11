/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_SHIFT_H__
#define __ASCENDC_API_REGBASE_SHIFT_H__

#include <type_traits>

using namespace AscendC;

// 对齐 pytorch eager 实测语义:移位量出界(负数/越界)时,left 的 lane 输出 0;
// right 统一按移 bits-1 位处理(符号提取:正数->0,负数->-1)。
// 筛法(左右共用):移位量按无符号位模式比较 [0, bits-1],负数按无符号读即大数,与越界一并筛出。
// 真机硬件不保证越界移位量语义(实测按位宽取模),必须显式修正:left 谓词归零,right 改写移位量。
// T = x/输出类型(signed/unsigned 均可);ST = 移位量类型(注册表恒为 signed,与 T 同位宽)。
// Reg::ShiftLeft/ShiftRight 移位量仅支持 signed,统一转同宽 signed 视图(位模式不变,满足指令类型契约)。
template <bool IsLeft, typename T, typename ST>
__aicore__ inline void ShiftImplVF(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ ST *src1, uint32_t calc_cnt) {
  uint32_t vlSize = static_cast<uint32_t>(GetVecLen() / sizeof(T));
  uint16_t repeatTime = static_cast<uint16_t>(AscendC::CeilDivision(calc_cnt, vlSize));

  AscendC::Reg::RegTensor<T> xReg, zReg;
  AscendC::Reg::RegTensor<ST> yReg, maxShiftReg;
  AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
  // bits-1 常量广播:移位量有效性上界
  AscendC::Reg::Duplicate(maxShiftReg, static_cast<ST>(sizeof(T) * 8 - 1), fullMask);

  using UnsignedST = typename std::make_unsigned<ST>::type;
  AscendC::Reg::MaskReg mask, validMask;
  for (uint16_t i = 0U; i < repeatTime; ++i) {
    mask = AscendC::Reg::UpdateMask<T>(calc_cnt);
    AscendC::Reg::DataCopy(xReg, src0 + i * vlSize);
    AscendC::Reg::DataCopy(yReg, src1 + i * vlSize);
    // 位移量按无符号位模式比较: [0, bits-1] 有效, 负数/越界均为无效
    AscendC::Reg::Compare<UnsignedST, AscendC::CMPMODE::LE>(validMask, (AscendC::Reg::RegTensor<UnsignedST> &)yReg,
                                                            (AscendC::Reg::RegTensor<UnsignedST> &)maxShiftReg, mask);
    if constexpr (IsLeft) {
      // 无效 lane 结果归零(谓词 ZEROING),移位量无需改写
      AscendC::Reg::ShiftLeft(zReg, xReg, (AscendC::Reg::RegTensor<typename std::make_signed<ST>::type> &)yReg,
                              validMask);
    } else {
      // 无效 lane 的移位量统一改写为 bits-1(符号提取),此后恒在 [0, bits-1]
      AscendC::Reg::Select(yReg, yReg, maxShiftReg, validMask);
      AscendC::Reg::ShiftRight(zReg, xReg, (AscendC::Reg::RegTensor<typename std::make_signed<ST>::type> &)yReg, mask);
    }
    AscendC::Reg::DataCopy(dst + i * vlSize, zReg, mask);
  }
}

template <typename T, typename ST>
__aicore__ inline void ShiftLeftExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src0,
                                       const AscendC::LocalTensor<ST> &src1, const uint32_t calc_cnt) {
  VF_CALL<ShiftImplVF<true, T, ST>>((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src0.GetPhyAddr(),
                                    (__ubuf__ ST *)src1.GetPhyAddr(), calc_cnt);
}

template <typename T, typename ST>
__aicore__ inline void ShiftRightExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src0,
                                        const AscendC::LocalTensor<ST> &src1, const uint32_t calc_cnt) {
  VF_CALL<ShiftImplVF<false, T, ST>>((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src0.GetPhyAddr(),
                                     (__ubuf__ ST *)src1.GetPhyAddr(), calc_cnt);
}

#endif  // __ASCENDC_API_REGBASE_SHIFT_H__
