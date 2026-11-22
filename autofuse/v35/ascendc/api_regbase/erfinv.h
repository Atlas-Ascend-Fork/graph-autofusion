/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_ERFINV_H__
#define __ASCENDC_API_REGBASE_ERFINV_H__

constexpr uint32_t ERFINV_THREAD_NUM = 1024;

// 对齐 ops-nn 单算子 activation/erfinv arch35 实现：SIMT 逐元素调用 erfinvf。
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(ERFINV_THREAD_NUM) inline void ErfinvSimtCompute(__ubuf__ T *x, __ubuf__ T *y,
                                                                                     const int64_t totalNum) {
  for (int64_t i = threadIdx.x; i < totalNum; i += blockDim.x) {
    y[i] = erfinvf(x[i]);
  }
}

template <typename T>
__aicore__ inline void ErfinvExtend(const LocalTensor<T> &dst, const LocalTensor<T> &src, const uint32_t calCount) {
  static_assert(AscendC::SupportType<T, float>(), "ErfinvExtend only supports float on current device!");
  AscendC::Simt::VF_CALL<ErfinvSimtCompute<T>>(AscendC::Simt::Dim3(ERFINV_THREAD_NUM), (__ubuf__ T *)src.GetPhyAddr(),
                                               (__ubuf__ T *)dst.GetPhyAddr(), calCount);
}

// MLIR bridge 现有映射固定 apiName 为 `Erfinv`，保留同名转发以兼容 MLIR 路径；legacy codegen 使用 ErfinvExtend。
template <typename T>
__aicore__ inline void Erfinv(const LocalTensor<T> &dst, const LocalTensor<T> &src, const uint32_t calCount) {
  ErfinvExtend<T>(dst, src, calCount);
}

#endif  // __ASCENDC_API_REGBASE_ERFINV_H__
