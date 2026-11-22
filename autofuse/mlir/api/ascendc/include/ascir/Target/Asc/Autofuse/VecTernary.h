/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_AUTOFUSE_VEC_TERNARY_H
#define ASCIR_TARGET_ASC_AUTOFUSE_VEC_TERNARY_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {
namespace ascendc {

//===----------------------------------------------------------------------===//
// Autofuse API extension wrappers -- ternary selection / clip family
//===----------------------------------------------------------------------===//

// Shared body for the contiguous-tile `Where` / `WhereExtend` / `ClipByValue`
// wrappers -- one shape, four ops. Like the unary wrappers, the call is
// unqualified because these live in the global namespace rather than in
// `AscendC::`. Unlike them, there is **no explicit template argument**: legacy
// emits `Where(dst, cond, src0, src1, count, tmp)` with the element type deduced
// (`where_api_call.cpp`), and spelling one out would pick a different overload.
//
// The tmp buffer, when there is one, comes **after** the count -- the opposite of
// the unary wrappers, where it comes before. Pass a null `sharedTmpBuffer` for
// the reg-based wrappers, which allocate their own registers instead.
//
// `clipbyvalue.h`'s original overload spells the last two the other way round
// (`tmp_buf, calCount`); rather than give this function a per-op argument-order
// switch, that header carries a forwarding overload in this order, so all four
// ops share this body verbatim. Legacy still calls the original one.
LogicalResult printApiExtensionTernary(CodeEmitter& emitter, Operation* op, StringRef apiName, Value dst,
                                       Value condition, Value src0, Value src1, Value calCount,
                                       Value sharedTmpBuffer);

template <typename ApiExtensionOp>
auto printOperation(CodeEmitter& emitter, ApiExtensionOp op)
    -> LogicalResultForT<ApiExtensionOp, ascendc::WhereContiguousOp>
{
    return printApiExtensionTernary(emitter, op, op.getAPIName(), op.getDst(), op.getCondition(), op.getSrc0(),
                                    op.getSrc1(), op.getCalCount(), op.getSharedTmpBuffer());
}

template <typename ApiExtensionOp>
auto printOperation(CodeEmitter& emitter, ApiExtensionOp op)
    -> LogicalResultForT<ApiExtensionOp, ascendc::WhereContiguousRegBaseOp>
{
    return printApiExtensionTernary(emitter, op, op.getAPIName(), op.getDst(), op.getCondition(), op.getSrc0(),
                                    op.getSrc1(), op.getCalCount(), /*sharedTmpBuffer=*/Value());
}

template <typename ApiExtensionOp>
auto printOperation(CodeEmitter& emitter, ApiExtensionOp op)
    -> LogicalResultForT<ApiExtensionOp, ascendc::ClipByValueContiguousOp>
{
    return printApiExtensionTernary(emitter, op, op.getAPIName(), op.getDst(), op.getSrcX(), op.getSrcMin(),
                                    op.getSrcMax(), op.getCalCount(), op.getSharedTmpBuffer());
}

template <typename ApiExtensionOp>
auto printOperation(CodeEmitter& emitter, ApiExtensionOp op)
    -> LogicalResultForT<ApiExtensionOp, ascendc::ClipByValueContiguousRegBaseOp>
{
    return printApiExtensionTernary(emitter, op, op.getAPIName(), op.getDst(), op.getSrcX(), op.getSrcMin(),
                                    op.getSrcMax(), op.getCalCount(), /*sharedTmpBuffer=*/Value());
}

// `FmaL2Op` is the one ternary stock call: `AscendC::Fma`, qualified, no wrapper
// header. Two things differ from the four wrappers above, and both live here
// rather than in the operation's `arguments`:
//
//   - the call is qualified with `AscendC::`, because `Fma` is declared in that
//     namespace rather than in the global one the Autofuse wrappers live in;
//   - the tmp buffer comes **before** the count, the reverse of the wrappers.
//
// The operation spells its operands in the order the shared ternary pattern
// builds them (dst, src0, src1, src2, calCount, sharedTmpBuffer); reordering is
// this printer's job.
template <typename TernaryTmpL2Op>
auto printOperation(CodeEmitter& emitter, TernaryTmpL2Op op) -> LogicalResultForT<TernaryTmpL2Op, ascendc::FmaL2Op>
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc0()) << ", " << emitter.getOrCreateName(op.getSrc1()) << ", "
       << emitter.getOrCreateName(op.getSrc2()) << ", " << emitter.getOrCreateName(op.getSharedTmpBuffer()) << ", "
       << emitter.getOrCreateName(op.getCalCount()) << ")";
    return success();
}

} // namespace ascendc
} // namespace mlir

#endif // ASCIR_TARGET_ASC_AUTOFUSE_VEC_TERNARY_H
