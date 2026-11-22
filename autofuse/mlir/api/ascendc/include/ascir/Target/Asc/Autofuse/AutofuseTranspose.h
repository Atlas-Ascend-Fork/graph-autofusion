/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You should refer to the License for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_AUTOFUSE_TRANSPOSE_H
#define ASCIR_TARGET_ASC_AUTOFUSE_TRANSPOSE_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {
namespace ascendc {

//===----------------------------------------------------------------------===//
// Vector transpose operations
//===----------------------------------------------------------------------===//

// Emits the reg-based wrapper
// The three dimension groups go out as braced initializer lists inline in the
// call -- the exact shape legacy `GenTransposeDimParam` writes -- with every
// element cast to `int16_t`/`int32_t` per `RangeType<T>`'s width rule. `T` is
// not spelled: legacy emits only the two non-type template arguments and lets
// the compiler deduce `T` from the tensor parameters.
LogicalResult printOperation(CodeEmitter& emitter, ascendc::TransposeExtendRegBaseOp op);

// Emits the mem-based wrapper
// `codegen::ConfusionTranspose<T>(dst, src, tmp, AutoFuseTransposeType::X, tiling);`
// followed by `AscendC::PipeBarrier<PIPE_ALL>();`.
// The `codegen::` qualifier is spelled here: the wrapper lives in `namespace
// codegen`, unlike the other Autofuse extension wrappers. The trailing pipe
// barrier is part of the call, matching legacy `transpose_api_call.cpp:158`.
LogicalResult printOperation(CodeEmitter& emitter, ascendc::ConfusionTransposeOp op);

} // namespace ascendc
} // namespace mlir

#endif // ASCIR_TARGET_ASC_AUTOFUSE_TRANSPOSE_H
