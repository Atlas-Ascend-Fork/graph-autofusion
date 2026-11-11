/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Autofuse/VecReduceND.h"
#include <type_traits>

using namespace mlir;
using namespace mlir::ascendc;

//===----------------------------------------------------------------------===//
// N-dimensional vector reduce operations
//===----------------------------------------------------------------------===//

// Emits:
//
//   uint32_t <dst>_reduce_shape[] = {a, r};
//   AscendC::ReduceSum<T, AscendC::Pattern::Reduce::AR, false>(
//       dst, src, sharedTmpBuffer, <dst>_reduce_shape, false);
//
// The shape array has to be a real `uint32_t[]` lvalue because the API takes
// `const uint32_t srcShape[]` (reduce.h:233) — same situation as Broadcast's two
// shape arrays, and emitted the same way. `pattern` is a type, not a value:
// `AscendC::Pattern::Reduce::{AR,RA}` are empty structs (reduce_common.h:34-35).
//
// `BaseReduce2DL2Op` 那一类（Sum / Max / Min / Prod / Any / All）的 C++ 形态逐字相同，只差
// 函数名，而函数名从 `getAPIName()` 取，所以这份实现按 op 类型模板化，六个 printOperation
// 重载各自转发进来。
namespace {
template <typename OpT> LogicalResult printReduce2DL2(CodeEmitter& emitter, OpT op)
{
    auto& os = emitter.ostream();

    auto dstTy = cast<LocalTensorType>(op.getDst().getType());
    auto dstName = emitter.getOrCreateName(op.getDst());
    auto shapeName = (dstName + "_reduce_shape").str();

    // 每一项都要 `static_cast<uint32_t>`：`srcShape` 在 ODS 上是 `Variadic<I32>`，而 MLIR 的
    // `i32` 是 signless，`CodeEmitter::shouldMapToUnsigned` 把 signless 与 signed 一并映射成
    // **有符号**，所以这些值在 C++ 里是 `int32_t`。往 `uint32_t[]` 的花括号初始化列表里塞
    // `int32_t` 是 C++11 起禁止的窄化转换（[dcl.init.list]/7：源类型的负值目标类型表示不了，
    // 且这些是运行期值、不是值域内的常量表达式），clang 系的 bisheng 默认把
    // `-Wc++11-narrowing` 报成 error。数组类型改不了——API 签名是 `const uint32_t srcShape[]`
    // （`reduce.h:233`）。
    //
    // 转换本身是安全的：这两项是 tile 的 extent，由 `materializePhysicalReduceShape` 从轴尺寸
    // 与对齐算出，恒为正。类型系统表达不了这一点（`arith.index_cast` 产出的就是 signless
    // `i32`），所以把理由写在这里而不是只加一个 cast。
    os << "uint32_t " << shapeName << "[] = {";
    llvm::interleaveComma(op.getSrcShape(), os,
                          [&](Value v) { os << "static_cast<uint32_t>(" << emitter.getOrCreateName(v) << ")"; });
    os << "};\n";

    if constexpr (!std::is_same_v<OpT, ascendc::ReduceSumInt32Op>) {
        os << ascNamespace << "::";
    }
    os << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), dstTy.getElementType()));
    os << ", " << ascNamespace << "::Pattern::Reduce::" << (op.isARLayout() ? "AR" : "RA") << ", "
       << (op.getIsReuseSource() ? "true" : "false") << ">(" << dstName << ", "
       << emitter.getOrCreateName(op.getSrc()) << ", " << emitter.getOrCreateName(op.getSharedTmpBuffer()) << ", "
       << shapeName << ", " << (op.getSrcInnerPad() ? "true" : "false") << ")";

    return success();
}
} // namespace

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::ReduceSum2DL2Op op)
{
    return printReduce2DL2(emitter, op);
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::ReduceSumInt32Op op)
{
    return printReduce2DL2(emitter, op);
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::ReduceMax2DL2Op op)
{
    return printReduce2DL2(emitter, op);
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::ReduceMin2DL2Op op)
{
    return printReduce2DL2(emitter, op);
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::ReduceProd2DL2Op op)
{
    return printReduce2DL2(emitter, op);
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::ReduceAny2DL2Op op)
{
    return printReduce2DL2(emitter, op);
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::ReduceAll2DL2Op op)
{
    return printReduce2DL2(emitter, op);
}

// ReduceInit is a lowering-only padding description, never a device helper.
LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::ReduceInitOp op)
{
    return op.emitOpError("must be expanded by convert-afir-to-ascendc before emission");
}
