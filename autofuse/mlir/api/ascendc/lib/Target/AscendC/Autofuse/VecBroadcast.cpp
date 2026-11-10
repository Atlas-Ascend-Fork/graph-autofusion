/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Autofuse/VecBroadcast.h"

using namespace mlir;
using namespace mlir::ascendc;

//===----------------------------------------------------------------------===//
// Vector broadcast operations
//===----------------------------------------------------------------------===//

// Emits `AscendC::Broadcast<T, dim, axis>(dst, src, dstShape[], srcShape[])`.
//
// The C++ API takes the two shapes as `const uint32_t[dim]` arrays and needs two
// non-type template params: `dim` (rank) and `axis` (the dimension being
// stretched). `dim` comes from `constRank`; `axis` is recovered from the tensor
// types as the single dimension where the source extent is 1 but the destination
// extent is not. The shape arrays are materialized as local `uint32_t[]` decls
// from the op's shape operands, mirroring how DataCopySlice emits SliceInfo[].
LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::BroadcastL2Op op)
{
    auto& os = emitter.ostream();

    auto dstTy = cast<LocalTensorType>(op.getDst().getType());
    auto srcTy = cast<LocalTensorType>(op.getSrc().getType());
    auto dstShape = dstTy.getShape();
    auto srcShape = srcTy.getShape();

    // axis = the broadcast dimension (src extent 1, dst extent != 1).
    int64_t axis = 0;
    for (auto [i, dims] : llvm::enumerate(llvm::zip(srcShape, dstShape))) {
        auto [s, d] = dims;
        if (s == 1 && d != 1) {
            axis = static_cast<int64_t>(i);
            break;
        }
    }

    auto dstName = emitter.getOrCreateName(op.getDst());
    auto srcName = emitter.getOrCreateName(op.getSrc());
    auto dstShapeName = (dstName + "_bc_dst_shape").str();
    auto srcShapeName = (dstName + "_bc_src_shape").str();

    os << "uint32_t " << dstShapeName << "[] = {";
    llvm::interleaveComma(op.getDstShape(), os, [&](Value v) { os << emitter.getOrCreateName(v); });
    os << "};\n";
    os << "uint32_t " << srcShapeName << "[] = {";
    llvm::interleaveComma(op.getSrcShape(), os, [&](Value v) { os << emitter.getOrCreateName(v); });
    os << "};\n";

    os << ascNamespace << "::" << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), dstTy.getElementType()));
    os << ", " << op.getConstRank() << ", " << axis << ">(" << dstName << ", " << srcName << ", " << dstShapeName
       << ", " << srcShapeName << ")";

    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::BroadcastDim3Op op)
{
    auto& os = emitter.ostream();

    // 六个尺寸按 C++ 形参表的次序发，每个都 `static_cast<uint32_t>`：包装的形参是
    // `const uint32_t`，而这里的值来自 `index` / `i32` 的 SSA 值。
    const Value sizes[] = {op.getSrcM(), op.getSrcK(), op.getSrcZ(),
                           op.getDstM(), op.getDstK(), op.getDstZ()};

    os << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc());
    for (Value size : sizes) {
        os << ", static_cast<uint32_t>(" << emitter.getOrCreateName(size) << ")";
    }
    os << ", " << emitter.getOrCreateName(op.getSharedTmpBuffer()) << ")";

    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::BroadcastExtendRegBaseOp op)
{
    auto& os = emitter.ostream();

    auto dstTy = cast<LocalTensorType>(op.getDst().getType());
    auto dstName = emitter.getOrCreateName(op.getDst());
    auto srcName = emitter.getOrCreateName(op.getSrc());
    // 一个 `(src, dst)` 对一份数组，与 legacy 的 `dst_shape_<src>_brc_to_<dst>` 同一条命名
    // 规则：同一个 dst 上挂两条广播时，按 dst 单独命名会在同一个作用域里撞名。
    auto suffix = (srcName + "_brc_to_" + dstName).str();
    auto dstShapeName = ("dst_shape_" + suffix);
    auto srcShapeName = ("src_shape_" + suffix);

    // 数组元素都 `static_cast<uint32_t>`：形参是 `const uint32_t *`，而这里的值来自
    // `index` 的 SSA 值。
    auto emitShape = [&](llvm::StringRef name, mlir::OperandRange shape) {
        os << "const uint32_t " << name << "[" << shape.size() << "] = {";
        llvm::interleaveComma(shape, os,
                              [&](Value v) { os << "static_cast<uint32_t>(" << emitter.getOrCreateName(v) << ")"; });
        os << "};\n";
    };
    emitShape(dstShapeName, op.getDstShape());
    emitShape(srcShapeName, op.getSrcShape());

    os << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), dstTy.getElementType()));
    os << ", " << op.getDstShape().size() << ">(" << dstName << ", " << srcName << ", " << dstShapeName << ", "
       << srcShapeName << ")";

    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::DuplicateU8Op op)
{
    emitter.ostream() << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
                      << emitter.getOrCreateName(op.getScalar()) << ", static_cast<uint32_t>("
                      << emitter.getOrCreateName(op.getCount()) << "))";
    return success();
}
