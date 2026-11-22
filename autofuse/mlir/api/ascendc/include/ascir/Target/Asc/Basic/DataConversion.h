/*
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_BASIC_DATA_CONVERSION_H
#define ASCIR_TARGET_ASC_BASIC_DATA_CONVERSION_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {
namespace ascendc {

//===----------------------------------------------------------------------===//
// Data Conversion operations
//===----------------------------------------------------------------------===//

LogicalResult printOperation(CodeEmitter& emitter, ascendc::TransDataTo5HDTensorListOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::TransDataTo5HDUintListOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::TransDataTo5HDOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::TransDataTo5HDTensorOp op);

} // namespace ascendc
} // namespace mlir

#endif // ASCIR_TARGET_ASC_BASIC_DATA_CONVERSION_H