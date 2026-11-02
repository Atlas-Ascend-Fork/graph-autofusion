/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_BROADCAST_H
#define __ASCENDC_API_REGBASE_BROADCAST_H

#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102) || (__NPU_ARCH__ == 9201) || \
    (__NPU_ARCH__ == 9202)
#include "broadcast_extend_impl.h"

template <typename T, int constRank = -1>
inline __aicore__ void BroadcastExtend(const LocalTensor<T> &dst, const LocalTensor<T> &src, const uint32_t *dst_shape,
                                       const uint32_t *src_shape) {
  AscendC::BroadcastTilingExtend runningTiling;
  AscendC::GetBroadcastTilingInfoExtendImpl<T, constRank>(constRank, dst_shape, src_shape, false, runningTiling);
  AscendC::BroadcastExtendImpl<T, constRank>(dst, src, dst_shape, src_shape, &runningTiling);
}
#endif

#endif  // __ASCENDC_API_REGBASE_BROADCAST_H
