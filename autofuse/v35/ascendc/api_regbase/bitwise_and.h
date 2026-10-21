/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_BITWISE_AND_H__
#define __ASCENDC_API_REGBASE_BITWISE_AND_H__

// AscendC BitwiseAnd 底层不支持 bool（bitwise_template 仅支持 8/16/32/64 位整型）。
// bool 与 uint8 同为 1 字节且取值域 0/1，按位与结果仍落在 0/1 域，语义与 bool 逐元素与等价，
// 因此 bool 输入输出统一 ReinterpretCast 为 uint8 视图后调用官方 BitwiseAnd。
template <typename T>
inline __aicore__ void BitwiseAndExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src0,
                                        const AscendC::LocalTensor<T> &src1, const uint32_t count) {
  AscendC::BitwiseAnd(dst, src0, src1, count);
}

inline __aicore__ void BitwiseAndExtend(const AscendC::LocalTensor<bool> &dst, const AscendC::LocalTensor<bool> &src0,
                                        const AscendC::LocalTensor<bool> &src1, const uint32_t count) {
  AscendC::BitwiseAnd(dst.template ReinterpretCast<uint8_t>(), src0.template ReinterpretCast<uint8_t>(),
                      src1.template ReinterpretCast<uint8_t>(), count);
}

#endif  // __ASCENDC_API_REGBASE_BITWISE_AND_H__
