/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Autofuse/AutofuseDataCopy.h"

using namespace mlir;
using namespace mlir::ascendc;

namespace {

template <typename CopyOpTy>
LogicalResult emitDataCopyPadExtend(CodeEmitter& emitter, CopyOpTy op)
{
    auto& os = emitter.ostream();
    auto tensorType = dyn_cast<ascendc::LocalTensorType>(op.getDst().getType());
    if (!tensorType) {
        tensorType = dyn_cast<ascendc::LocalTensorType>(op.getSrc().getType());
    }
    if (!tensorType) {
        return op.emitOpError("DataCopyPadExtend requires a local tensor operand");
    }

    os << "DataCopyPadExtend<";
    FAIL_OR(emitter.emitType(op.getLoc(), tensorType.getElementType()));
    os << ">(" << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getSrc()) << ", "
       << "static_cast<uint32_t>(" << emitter.getOrCreateName(op.getBlockCount()) << "), "
       << "static_cast<uint32_t>(" << emitter.getOrCreateName(op.getBlockLen()) << "), "
       << "static_cast<uint32_t>(" << emitter.getOrCreateName(op.getSrcStride()) << "), "
       << "static_cast<uint32_t>(" << emitter.getOrCreateName(op.getDstStride()) << "))";
    return success();
}

} // namespace

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::DataCopyPadExtendOp op)
{
    return emitDataCopyPadExtend(emitter, op);
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::DataCopyNddmaRegBaseOp op)
{
    auto& os = emitter.ostream();

    auto dstTy = dyn_cast<ascendc::LocalTensorType>(op.getDst().getType());
    if (!dstTy) {
        return op.emitOpError("DataCopyNddma requires a local tensor destination");
    }
    const size_t rank = op.getOutputDims().size();
    if (rank == 0 || op.getOutputStride().size() != rank || op.getInputStride().size() != rank) {
        return op.emitOpError("DataCopyNddma needs one output dim, output stride and input stride per segment");
    }

    auto dstName = emitter.getOrCreateName(op.getDst());
    // 一份数组一个名字，后缀取 `(dst, src)` 对：同一个作用域里两条 ND-DMA 搬到不同 dst 时，
    // 只按 dst 命名会撞名。与 `BroadcastExtendRegBaseOp` 的 `dst_shape_<src>_brc_to_<dst>`
    // 同一条规则。
    auto suffix = (emitter.getOrCreateName(op.getSrc()) + "_nddma_to_" + dstName).str();

    // 形参是 `const int64_t (&)[dim]`，而这里的值来自 `index` 的 SSA 值，所以逐个
    // `static_cast<int64_t>`。
    auto emitSizes = [&](llvm::StringRef name, mlir::OperandRange sizes) {
        os << "const int64_t " << name << "[" << sizes.size() << "] = {";
        llvm::interleaveComma(sizes, os,
                              [&](Value v) { os << "static_cast<int64_t>(" << emitter.getOrCreateName(v) << ")"; });
        os << "};\n";
    };
    const std::string outputDimsName = ("output_dims_" + suffix);
    const std::string outputStrideName = ("output_stride_" + suffix);
    const std::string inputStrideName = ("input_stride_" + suffix);
    emitSizes(outputDimsName, op.getOutputDims());
    emitSizes(outputStrideName, op.getOutputStride());
    emitSizes(inputStrideName, op.getInputStride());

    // `dim` 是第二个模板实参。包装里 `for (i < dim)` 按它反序填 `MultiCopyLoopInfo`，
    // 所以三个数组的次序必须一致，且与段的次序一致（外 → 内）。
    os << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), dstTy.getElementType()));
    os << ", " << rank << ">(" << dstName << ", " << emitter.getOrCreateName(op.getSrc()) << ", " << outputDimsName
       << ", " << outputStrideName << ", " << inputStrideName << ")";
    return success();
}
