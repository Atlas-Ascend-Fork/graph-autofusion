/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_AUTOFUSE_VEC_REMOVE_PAD_H
#define ASCIR_TARGET_ASC_AUTOFUSE_VEC_REMOVE_PAD_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {
namespace ascendc {

//===----------------------------------------------------------------------===//
// Vector layout operations
//===----------------------------------------------------------------------===//

// Emits the Autofuse wrapper
// `RemovePadExtend<T>(dst, src, last_axis_actual_size, last_axis_size, non_last_axis_merge)`.
//
// Unqualified, like every other Autofuse API extension wrapper: they are global
// functions, not members of `AscendC::`. The element type is spelled out rather
// than deduced, matching `printApiExtensionUnary` next door -- both tensors carry
// the same `T`, so it only has to agree with itself.
//
// The three sizes are `const uint64_t` parameters while the values reaching here
// are `index`, so each is wrapped in a `static_cast<uint64_t>` -- the same reason
// `BroadcastDim3Op`'s printer casts its six.
LogicalResult printOperation(CodeEmitter& emitter, ascendc::RemovePadExtendOp op);

} // namespace ascendc
} // namespace mlir

#endif // ASCIR_TARGET_ASC_AUTOFUSE_VEC_REMOVE_PAD_H
