/*
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Basic/Aipp.h"

using namespace mlir;
using namespace mlir::ascendc;

//===----------------------------------------------------------------------===//
// AIPP Emitters
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, SetAippFunctionsOp op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName();

    os << "(";
    os << emitter.getOrCreateName(op.getSrc0());

    if (auto src1 = op.getSrc1()) {
        os << ", " << emitter.getOrCreateName(src1);
    }

    os << ", " << ascNamespace << "::AippInputFormat::" << ascendc::stringifyEnum(op.getFormat());
    os << ", " << emitter.getOrCreateName(op.getConfig());
    os << ")";

    return success();
}