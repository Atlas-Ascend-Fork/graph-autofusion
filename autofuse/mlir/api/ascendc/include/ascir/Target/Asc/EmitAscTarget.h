/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_EMITASC_H
#define ASCIR_TARGET_ASC_EMITASC_H

#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Target/Asc/Common.h"

namespace mlir {
namespace emitasc {

LogicalResult printOperation(CodeEmitter& emitter, emitasc::CallOpaqueOp op);

LogicalResult printOperation(CodeEmitter& emitter, emitasc::CopyStructOp op);

LogicalResult printOperation(CodeEmitter& emitter, emitasc::DeclarePyStructOp op);

LogicalResult printOperation(CodeEmitter& emitter, emitasc::DereferenceOp op);

LogicalResult printOperation(CodeEmitter& emitter, emitasc::InitStructOp op);

LogicalResult printOperation(CodeEmitter& emitter, emitasc::MaskOp op);

LogicalResult printOperation(CodeEmitter& emitter, emitasc::MemberOp op);

LogicalResult printOperation(CodeEmitter& emitter, emitasc::MemberPtrOp op);

LogicalResult printOperation(CodeEmitter& emitter, emitasc::MemberRefOp op);

LogicalResult printOperation(CodeEmitter& emitter, emitasc::PtrOffsetOp op);

LogicalResult printOperation(CodeEmitter& emitter, emitasc::ReinterpretCastOp op);

LogicalResult printOperation(CodeEmitter& emitter, emitasc::SetMemberOp op);

LogicalResult printOperation(CodeEmitter& emitter, emitasc::VariableOp op);

LogicalResult printOperation(CodeEmitter& emitter, emitasc::VerbatimOp op);

LogicalResult printOperation(CodeEmitter& emitter, emitasc::VFForOp op);

LogicalResult printOperation(CodeEmitter& emitter, emitasc::VecScopeOp op);

LogicalResult printOperation(CodeEmitter&, emitasc::YieldOp);

} // namespace emitasc
} // namespace mlir

#endif // ASCIR_TARGET_ASC_EMITASC_H
