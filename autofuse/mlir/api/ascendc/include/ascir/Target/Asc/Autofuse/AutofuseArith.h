/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_AUTOFUSE_ARITH_H
#define ASCIR_TARGET_ASC_AUTOFUSE_ARITH_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {

// The width the `*ExtendedOp` emitters compute the high half at. `index` has no
// bit width of its own and `getIntOrFloatBitWidth` asserts on it, so it is read
// as the 32 bits AscendC lowers it to. Upstream's `MulUIExtendedOp` emitter reads
// it through here too -- it hits `index` for the same reason ours do.
uint32_t getResultBitWidth(Type type);

LogicalResult printOperation(CodeEmitter& emitter, arith::AddUIExtendedOp op);

LogicalResult printOperation(CodeEmitter& emitter, arith::MulSIExtendedOp op);

} // namespace mlir

#endif // ASCIR_TARGET_ASC_AUTOFUSE_ARITH_H
