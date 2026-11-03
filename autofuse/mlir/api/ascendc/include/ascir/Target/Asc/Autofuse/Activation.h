/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_AUTOFUSE_ACTIVATION_H
#define ASCIR_TARGET_ASC_AUTOFUSE_ACTIVATION_H

// For printApiExtensionUnary: SigmoidExtend prints like every other Autofuse
// wrapper, and that rule is written down once, next to the unary emitters.
#include "ascir/Target/Asc/Autofuse/VecUnary.h"

namespace mlir {
namespace ascendc {

// `AscendC::Sigmoid(dst, src, tmp, count)` -- the stock call, on the platforms
// that ship it. It still wants a tmp buffer, which is why the AFIR side does not tie
// "needs a tmp buffer" to "uses the Autofuse wrapper".
// `AscendC::Gelu(dst, src, tmp, count)` -- same stock shape as Sigmoid below:
// a caller-provided tmp buffer, no Autofuse wrapper on either API family.
LogicalResult printOperation(CodeEmitter& emitter, ascendc::GeluOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::SigmoidOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::SigmoidExtendOp op);

} // namespace ascendc
} // namespace mlir

#endif // ASCIR_TARGET_ASC_AUTOFUSE_ACTIVATION_H
