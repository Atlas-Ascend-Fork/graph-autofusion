/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_BASIC_BLOCK_SYNC_H
#define ASCIR_TARGET_ASC_BASIC_BLOCK_SYNC_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {
namespace ascendc {

//===----------------------------------------------------------------------===//
// Synchronization operations
//===----------------------------------------------------------------------===//

LogicalResult printOperation(CodeEmitter& emitter, ascendc::PipeBarrierOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::WaitFlagOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::CrossCoreSetFlagOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::CrossCoreWaitFlagOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::GetBufOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::RlsBufOp op);

} // namespace ascendc
} // namespace mlir

#endif // ASCIR_TARGET_ASC_BASIC_BLOCK_SYNC_H
