/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_AUTOFUSE_VEC_BROADCAST_H
#define ASCIR_TARGET_ASC_AUTOFUSE_VEC_BROADCAST_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {
namespace ascendc {

//===----------------------------------------------------------------------===//
// Vector broadcast operations
//===----------------------------------------------------------------------===//

LogicalResult printOperation(CodeEmitter& emitter, ascendc::BroadcastL2Op op);
LogicalResult printOperation(CodeEmitter& emitter, ascendc::DuplicateU8Op op);

// Emits the Autofuse wrapper `Broadcast(dst, src, src_m, src_k, src_z, dst_m, dst_k, dst_z, tmp)`.
//
// Unqualified, like every other Autofuse API extension wrapper: they are global
// functions, not members of `AscendC::`. The element type is left to deduction
// rather than spelled out, matching what legacy codegen writes for this call
// (`codegen/api_call/broadcast/broadcast_api_call.cpp:189`) -- both tensors carry
// the same `T`, so there is nothing to disambiguate.
LogicalResult printOperation(CodeEmitter& emitter, ascendc::BroadcastDim3Op op);

// Emits the reg-base wrapper `BroadcastExtend<T, rank>(dst, src, dstShape[], srcShape[])`.
//
// Unqualified, like every other Autofuse API extension wrapper. Unlike the
// `Broadcast` wrapper next to it, `T` and `rank` are spelled out: `rank` is a
// non-type template parameter the call cannot deduce from a decayed pointer, and
// legacy writes both (`v35/codegen/reg_api_call/reg_broadcast_api_call.cpp:120`).
// The two shapes become local `const uint32_t[]` declarations, the same way
// `BroadcastL2Op` above materializes its shape arrays.
LogicalResult printOperation(CodeEmitter& emitter, ascendc::BroadcastExtendRegBaseOp op);

} // namespace ascendc
} // namespace mlir

#endif // ASCIR_TARGET_ASC_AUTOFUSE_VEC_BROADCAST_H
