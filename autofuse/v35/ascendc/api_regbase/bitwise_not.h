/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_BITWISE_NOT_H__
#define __ASCENDC_API_REGBASE_BITWISE_NOT_H__

// AscendC BitwiseNot 底层不支持 bool（bitwise_template 仅支持 8/16/32/64 位整型）。
// bool 的按位非语义为逻辑非（0<->1）：uint8 域按位非（~1 = 0xFE）会产出非法 bool 值，
// 无法通过 uint8 视图复用 BitwiseNot，bool 分支调用官方 LogicalNot（其 dst 支持 bool）。
template <typename T>
inline __aicore__ void BitwiseNotExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                        const uint32_t count) {
  AscendC::BitwiseNot(dst, src, count);
}

inline __aicore__ void BitwiseNotExtend(const AscendC::LocalTensor<bool> &dst, const AscendC::LocalTensor<bool> &src,
                                        const uint32_t count) {
  AscendC::LogicalNot(dst, src, count);
}

#endif  // __ASCENDC_API_REGBASE_BITWISE_NOT_H__
