/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_ADV_KFC_H
#define ASCIR_TARGET_ASC_ADV_KFC_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {
namespace ascendc {

// Resource Management

LogicalResult printOperation(CodeEmitter& emitter, ascendc::KfcInitOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::KfcInitObjOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::KfcIsRunOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::KfcRunOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::KfcQuitOp op);

} // namespace ascendc
} // namespace mlir

#endif // ASCIR_TARGET_ASC_ADV_KFC_H
