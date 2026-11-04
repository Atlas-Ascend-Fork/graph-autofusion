/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Autofuse/AutofuseVecCmpsel.h"

using namespace mlir;
using namespace mlir::ascendc;

// `AscendC::Reg::Compare<T, CMPMODE::EQ>(dstMask, src0, src1, mask)`。
//
// 两个模板实参都要写出来：`mode` 排在 `T` 之后（签名见
// `kernel_reg_compute_vec_cmpsel_intf.h:33`），要写第二个就得把第一个也写出来。`T` 取
// **源**的元素类型——目的地是谓词，没有元素类型可取。与 legacy 的
// `micro_compare_api_call.cpp:38` 逐字同形（它那边写 `MicroAPI::`，而
// `namespace MicroAPI = Reg`，同一个函数）。
LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::CompareRegOp op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::Reg::" << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), op.getSrc0Reg().getType().getElementType()));
    os << ", " << ascNamespace << "::CMPMODE::" << ascendc::stringifyEnum(op.getCmpMode()) << ">("
       << emitter.getOrCreateName(op.getDstMask()) << ", " << emitter.getOrCreateName(op.getSrc0Reg()) << ", "
       << emitter.getOrCreateName(op.getSrc1Reg()) << ", " << emitter.getOrCreateName(op.getMaskReg()) << ")";
    return success();
}

// 一行变量声明，与 `Basic/Reg.cpp` 的 `RegTensorOp` 逐字同形。legacy 那边是
// `AscendC::MicroAPI::MaskReg <name>;`（`micro_api_call.cpp:52` 的 `MaskRegTypes`）。
LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::MaskRegOp op)
{
    FAIL_OR(emitter.emitVariableDeclaration(op->getResult(0), false));
    return success();
}
