/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_AUTOFUSE_VEC_VCONV_H
#define ASCIR_TARGET_ASC_AUTOFUSE_VEC_VCONV_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {
namespace ascendc {

// Autofuse 的 `CastExtend` 包装。mem 侧与其余 `XxxExtend` 一元包装逐字同形，走共用的
// `printApiExtensionUnary`（在 VecUnary.h）；reg 侧的三个数组实参是这条路上第一个
// brace-init 形态，自己一个函数。
LogicalResult printOperation(CodeEmitter& emitter, ascendc::CastExtendContiguousOp op);

LogicalResult printOperation(CodeEmitter& emitter, ascendc::CastExtendContiguousRegBaseOp op);
LogicalResult printOperation(CodeEmitter& emitter, ascendc::Ceil2IntExtendRegBaseOp op);

// 寄存器形态的一条 `Reg::Cast`（VF 里用的那个，不是上面两个 tile 级包装）。
LogicalResult printOperation(CodeEmitter& emitter, ascendc::CastRegOp op);

} // namespace ascendc
} // namespace mlir

#endif // ASCIR_TARGET_ASC_AUTOFUSE_VEC_VCONV_H
