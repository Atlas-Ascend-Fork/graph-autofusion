/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_SCALAR_MAXIMUM_H
#define __ASCENDC_API_REGBASE_SCALAR_MAXIMUM_H

// ui16 / ui32 的 tensor-scalar：连续 LocalTensor 接口 `AscendC::Maxs` 不收这两种类型，改走
// 寄存器标量接口 `MicroAPI::Maxs`。循环写法与 `floor_div.h` 的 `FloorDivUintImplVF` 相同：
// 寄存器声明在循环外、归纳变量是 uint16_t、掩码计数器由 `UpdateMask` 逐轮自减。
template <typename T>
__aicore__ inline void MaxsImplVF(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, uint32_t count,
                                  uint16_t repeat_time) {
  constexpr uint32_t oneRepElm = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(T));
  AscendC::MicroAPI::RegTensor<T> srcVreg;
  AscendC::MicroAPI::RegTensor<T> dstVreg;
  AscendC::MicroAPI::MaskReg mask;

  for (uint16_t i = 0; i < repeat_time; i++) {
    mask = AscendC::MicroAPI::UpdateMask<T>(count);
    AscendC::MicroAPI::DataCopy(srcVreg, src + i * oneRepElm);
    AscendC::MicroAPI::Maxs(dstVreg, srcVreg, scalarValue, mask);
    AscendC::MicroAPI::DataCopy(dst + i * oneRepElm, dstVreg, mask);
  }
}

// 标量的 C++ 类型单独做一个模板形参：`T` 只从两个 LocalTensor 推，标量按 `MicroAPI::Maxs` 的形参
// （按值传 `T`）在这里转一次，与直接调标量接口时的隐式转换相同。
template <typename T, typename ScalarT>
__aicore__ inline void MaxsExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                  const ScalarT scalarValue, const uint32_t size) {
  constexpr uint32_t oneRepElm = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(T));
  uint16_t repeat_time = static_cast<uint16_t>(AscendC::CeilDivision(size, oneRepElm));
  VF_CALL<MaxsImplVF<T>>((__ubuf__ T *)dst.GetPhyAddr(), (__ubuf__ T *)src.GetPhyAddr(), static_cast<T>(scalarValue),
                         size, repeat_time);
}

#endif  // __ASCENDC_API_REGBASE_SCALAR_MAXIMUM_H
