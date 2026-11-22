/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_FLOOR_DIV_H
#define __ASCENDC_API_REGBASE_FLOOR_DIV_H

template <typename T>
__aicore__ inline void FloorDivImplVF(__ubuf__ T *dst, __ubuf__ T *src1, __ubuf__ T *src2, uint32_t count,
                                      uint16_t repeat_time) {
  constexpr uint32_t oneRepElm = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(T));
  AscendC::MicroAPI::RegTensor<T> srcVreg1;
  AscendC::MicroAPI::RegTensor<T> srcVreg2;
  AscendC::MicroAPI::RegTensor<T> dstVreg;
  AscendC::MicroAPI::MaskReg mask;
  AscendC::MicroAPI::MaskReg needAdjust;
  AscendC::MicroAPI::MaskReg needAdjustZero;
  AscendC::MicroAPI::RegTensor<T> r;
  AscendC::MicroAPI::RegTensor<T> qFloor;
  AscendC::MicroAPI::RegTensor<T> tmpReg0;
  AscendC::MicroAPI::RegTensor<T> tmpReg1;
  AscendC::MicroAPI::RegTensor<T> tmpRegN;
  AscendC::NotNumUnion notNum;
  notNum.i = AscendC::F32_NAN;
  AscendC::MicroAPI::Duplicate(tmpReg0, 0.0f);
  AscendC::MicroAPI::Duplicate(tmpReg1, -1.0f);
  AscendC::MicroAPI::Duplicate(tmpRegN, notNum.f);
  for (uint16_t i = 0; i < repeat_time; i++) {
    mask = AscendC::MicroAPI::UpdateMask<T>(count);
    AscendC::MicroAPI::DataCopy(srcVreg1, src1 + i * oneRepElm);
    AscendC::MicroAPI::DataCopy(srcVreg2, src2 + i * oneRepElm);
    AscendC::MicroAPI::Div(dstVreg, srcVreg1, srcVreg2, mask);
    AscendC::MicroAPI::Truncate<T, AscendC::RoundMode::CAST_FLOOR>(dstVreg, dstVreg, mask);
    AscendC::MicroAPI::Compare<T, AscendC::CMPMODE::EQ>(needAdjustZero, srcVreg1, tmpReg0, mask);
    AscendC::MicroAPI::Mul(srcVreg1, srcVreg1, tmpReg1, mask);
    AscendC::MicroAPI::MulAddDst(srcVreg1, dstVreg, srcVreg2, mask);
    AscendC::MicroAPI::Mul(srcVreg1, srcVreg1, srcVreg2, mask);
    AscendC::MicroAPI::Compare<T, AscendC::CMPMODE::GT>(needAdjust, srcVreg1, tmpReg0, mask);
    AscendC::MicroAPI::Add(qFloor, dstVreg, tmpReg1, mask);
    AscendC::MicroAPI::Select(dstVreg, qFloor, dstVreg, needAdjust);
    AscendC::MicroAPI::Compare<T, AscendC::CMPMODE::EQ>(needAdjust, srcVreg2, tmpReg0, mask);
    AscendC::MicroAPI::MaskAnd(needAdjust, needAdjust, needAdjustZero, mask);
    AscendC::MicroAPI::Select(dstVreg, tmpRegN, dstVreg, needAdjust);
    AscendC::MicroAPI::DataCopy(dst + i * oneRepElm, dstVreg, mask);
  }
}

// int64 走 SIMT 逐元素实现（参考 ops-math 单算子 FloorDivInt_1）：向量化 64 位整除开销大，
// SIMT 按线程直接做标量除法并修正向下取整语义
constexpr uint32_t FLOOR_DIV_THREAD_NUM = 1024;

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(FLOOR_DIV_THREAD_NUM) inline void FloorDivInt64SimtCompute(
    __ubuf__ T *dst, __ubuf__ T *src1, __ubuf__ T *src2, const int64_t total_num) {
  for (int64_t i = threadIdx.x; i < total_num; i += blockDim.x) {
    if (src2[i] == 0) {
      // 与向量实现 FloorDivIntImplVF 保持一致：除 0 输出 0
      dst[i] = 0;
      continue;
    }
    const bool signsDiffer = (src1[i] < 0) != (src2[i] < 0);
    if (signsDiffer) {
      const auto quot = src1[i] / src2[i];
      const auto rem = src1[i] - src2[i] * quot;
      dst[i] = rem != 0 ? quot - 1 : quot;
    } else {
      dst[i] = src1[i] / src2[i];
    }
  }
}

template <typename T>
__aicore__ inline void FloorDivIntImplVF(__ubuf__ T *dst, __ubuf__ T *src1, __ubuf__ T *src2, uint32_t count,
                                         uint16_t repeat_time) {
  constexpr uint32_t oneRepElm = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(T));
  AscendC::MicroAPI::RegTensor<T> srcVreg1;
  AscendC::MicroAPI::RegTensor<T> srcVreg2;
  AscendC::MicroAPI::RegTensor<T> quotientVreg;
  AscendC::MicroAPI::RegTensor<T> modVreg;
  AscendC::MicroAPI::RegTensor<T> tmpVreg;
  AscendC::MicroAPI::RegTensor<T> dstVreg;
  AscendC::MicroAPI::RegTensor<T> subVreg;
  AscendC::MicroAPI::MaskReg mask, zeroMask, mask1, mask2, negMask;
  AscendC::MicroAPI::MaskReg fullMask = AscendC::MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
  AscendC::MicroAPI::Duplicate(subVreg, -1);

  for (uint16_t i = 0; i < repeat_time; i++) {
    mask = AscendC::MicroAPI::UpdateMask<T>(count);
    AscendC::MicroAPI::DataCopy(srcVreg1, src1 + i * oneRepElm);
    AscendC::MicroAPI::DataCopy(srcVreg2, src2 + i * oneRepElm);
    AscendC::MicroAPI::Div(dstVreg, srcVreg1, srcVreg2, mask);
    AscendC::MicroAPI::Mul(tmpVreg, dstVreg, srcVreg2, mask);
    AscendC::MicroAPI::Sub(modVreg, srcVreg1, tmpVreg, mask);
    AscendC::MicroAPI::Compares<T, AscendC::CMPMODE::NE>(zeroMask, modVreg, 0u, mask);
    AscendC::MicroAPI::Xor(tmpVreg, srcVreg1, srcVreg2, mask);
    AscendC::MicroAPI::Compares<T, AscendC::CMPMODE::LT>(mask1, tmpVreg, 0u, mask);
    AscendC::MicroAPI::And(mask2, mask1, zeroMask, fullMask);
    AscendC::MicroAPI::Add<T, AscendC::MicroAPI::MaskMergeMode::MERGING>(dstVreg, dstVreg, subVreg, mask2);
    AscendC::MicroAPI::Compares<T, AscendC::CMPMODE::EQ>(zeroMask, srcVreg2, 0u, mask);
    AscendC::MicroAPI::Duplicate<T, AscendC::MicroAPI::MaskMergeMode::MERGING>(dstVreg, 0u, zeroMask);
    AscendC::MicroAPI::DataCopy(dst + i * oneRepElm, dstVreg, mask);
  }
}

template <typename T>
__aicore__ inline void FloorDivUintImplVF(__ubuf__ T *dst, __ubuf__ T *src1, __ubuf__ T *src2, uint32_t count,
                                          uint16_t repeat_time) {
  constexpr uint32_t oneRepElm = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(T));
  AscendC::MicroAPI::RegTensor<T> srcVreg1;
  AscendC::MicroAPI::RegTensor<T> srcVreg2;
  AscendC::MicroAPI::RegTensor<T> dstVreg;
  AscendC::MicroAPI::RegTensor<T> subVreg;
  AscendC::MicroAPI::MaskReg mask;
  AscendC::MicroAPI::MaskReg fullMask = AscendC::MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
  AscendC::MicroAPI::Duplicate(subVreg, -1);

  for (uint16_t i = 0; i < repeat_time; i++) {
    mask = AscendC::MicroAPI::UpdateMask<T>(count);
    AscendC::MicroAPI::DataCopy(srcVreg1, src1 + i * oneRepElm);
    AscendC::MicroAPI::DataCopy(srcVreg2, src2 + i * oneRepElm);
    AscendC::MicroAPI::Div(dstVreg, srcVreg1, srcVreg2, mask);
    AscendC::MicroAPI::DataCopy(dst + i * oneRepElm, dstVreg, mask);
  }
}

template <typename T>
__aicore__ inline void FloorDivExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src1,
                                      const AscendC::LocalTensor<T> &src2, const uint32_t size) {
  constexpr uint32_t oneRepElm = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(T));
  uint16_t repeat_time = static_cast<uint16_t>(AscendC::CeilDivision(size, oneRepElm));
  if constexpr (AscendC::IsSameType<T, int64_t>::value) {
    AscendC::Simt::VF_CALL<FloorDivInt64SimtCompute<T>>(AscendC::Simt::Dim3(FLOOR_DIV_THREAD_NUM),
                                                        (__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src1.GetPhyAddr(),
                                                        (__ubuf__ T *)src2.GetPhyAddr(), size);
  } else if constexpr (AscendC::SupportType<T, int16_t, int32_t>()) {
    VF_CALL<FloorDivIntImplVF<T>>((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src1.GetPhyAddr(),
                                  (__ubuf__ T *)src2.GetPhyAddr(), size, repeat_time);
  } else if constexpr (AscendC::SupportType<T, uint16_t, uint32_t, uint64_t>()) {
    VF_CALL<FloorDivUintImplVF<T>>((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src1.GetPhyAddr(),
                                   (__ubuf__ T *)src2.GetPhyAddr(), size, repeat_time);
  } else if constexpr (AscendC::SupportType<T, half, float>()) {
    VF_CALL<FloorDivImplVF<T>>((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src1.GetPhyAddr(),
                               (__ubuf__ T *)src2.GetPhyAddr(), size, repeat_time);
  }
}

#endif  // __ASCENDC_API_REGBASE_FLOOR_DIV_H
