/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Autofuse/RegDataCopy.h"

using namespace mlir;
using namespace mlir::ascendc;

// 两条都打成 `AscendC::Reg::` 限定。legacy 发的是 `AscendC::MicroAPI::`，而
// `namespace MicroAPI = Reg`（`asc/impl/basic_api/kernel_macros.h`）——同一个函数，这里取
// 真名。元素类型是显式模板实参，与 `LoadAlign` 的签名
// `LoadAlign<T, LoadDist>(U& dstReg, __ubuf__ T* srcAddr)` 对齐：缺省模板实参下 `T` 由
// `srcAddr` 推不出来（它是 `__ubuf__ T*`，推得出来），但显式写出来与 legacy 的产物一致，
// 也让读产物的人不必去推。
LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::LoadAlignRegOp op)
{
    auto& os = emitter.ostream();
    auto srcType = dyn_cast<MemRefType>(op.getSrc().getType());
    if (!srcType) {
        return op.emitOpError("LoadAlign requires a memref source");
    }
    os << ascNamespace << "::Reg::" << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), srcType.getElementType()));
    // `dist` 缺席就不打：那是 `DIST_NORM`，与 M1 的产物逐字相同。跨 dtype 的 VF 要靠它把
    // 窄元素取进宽的 lane 槽位，详见这个 op 的 ODS 注释。
    if (auto dist = op.getDist()) {
        os << ", " << ascNamespace << "::Reg::LoadDist::" << ascendc::stringifyEnum(*dist);
    }
    os << ">(" << emitter.getOrCreateName(op.getDstReg()) << ", " << emitter.getOrCreateName(op.getSrc()) << ")";
    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::StoreAlignRegOp op)
{
    auto& os = emitter.ostream();
    auto dstType = dyn_cast<MemRefType>(op.getDst().getType());
    if (!dstType) {
        return op.emitOpError("StoreAlign requires a memref destination");
    }
    os << ascNamespace << "::Reg::" << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), dstType.getElementType()));
    // 同 `LoadAlign`：缺席即 `DIST_NORM`。
    if (auto dist = op.getDist()) {
        os << ", " << ascNamespace << "::Reg::StoreDist::" << ascendc::stringifyEnum(*dist);
    }
    os << ">(" << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getSrcReg()) << ", "
       << emitter.getOrCreateName(op.getMaskReg()) << ")";
    return success();
}
