/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_TARGET_ASC_AUTOFUSE_VEC_BINARY_H
#define ASCIR_TARGET_ASC_AUTOFUSE_VEC_BINARY_H

#include "ascir/Target/Asc/Basic/BasicVecBinary.h"

namespace mlir {
namespace ascendc {

//===----------------------------------------------------------------------===//
// Autofuse API extension wrappers -- binary family
//===----------------------------------------------------------------------===//

// Shared body for the binary `XxxExtend` wrappers. Like the unary ones, the call
// is unqualified because these live in the global namespace rather than in
// `AscendC::`, and the tmp buffer comes **before** the count. Unlike them, there
// is no explicit template argument: legacy emits `BitwiseAndExtend(dst, s0, s1,
// tmp, size)` with the element type deduced (`binary_api_tmp_call.cpp`), and
// spelling one out would pick a different overload for the reinterpreting path.
//
// Pass a null `sharedTmpBuffer` for a wrapper that does not take one, and a
// null `extraScalar` for the wrappers that take only tiles -- which is all of
// them but `AxpyExtend`, whose `alpha` sits between src1 and the tmp buffer.
LogicalResult printApiExtensionBinary(CodeEmitter& emitter, Operation* op, StringRef apiName, Value dst, Value src0,
                                      Value src1, Value extraScalar, Value sharedTmpBuffer, Value calCount);

template <typename ApiExtensionOp>
auto printOperation(CodeEmitter& emitter, ApiExtensionOp op)
    -> LogicalResultForT<ApiExtensionOp, ascendc::BitwiseAndExtendOp, ascendc::FloorDivExtendOp,
                         ascendc::RemainderExtendOp, ascendc::RemainderExtendRegBaseOp,
                         ascendc::LogicalAndExtendOp, ascendc::LogicalOrExtendOp,
                         ascendc::TrueDivExtendOp, ascendc::PowOp, ascendc::PowRegBaseOp,
                         ascendc::FmodExtendRegBaseOp, ascendc::NextAfterExtendRegBaseOp,
                         ascendc::Atan2ExtendRegBaseOp, ascendc::ZetaExtendRegBaseOp,
                         ascendc::IgammaExtendRegBaseOp, ascendc::IgammacExtendRegBaseOp>
{
    return printApiExtensionBinary(emitter, op, op.getAPIName(), op.getDst(), op.getSrc0(), op.getSrc1(),
                                   /*extraScalar=*/Value(), op.getSharedTmpBuffer(), op.getCalCount());
}

template <typename ApiExtensionOp>
auto printOperation(CodeEmitter& emitter, ApiExtensionOp op)
    -> LogicalResultForT<ApiExtensionOp, ascendc::FloorDivExtendRegBaseOp, ascendc::SubExtendRegBaseOp,
                         ascendc::DivExtendRegBaseOp, ascendc::LogicalAndExtendRegBaseOp,
                         ascendc::LogicalOrExtendRegBaseOp, ascendc::LogicalXorExtendRegBaseOp,
                         ascendc::CopySignExtendRegBaseOp, ascendc::TruncDivExtendRegBaseOp,
                         ascendc::BitwiseOrExtendRegBaseOp, ascendc::BitwiseXorExtendRegBaseOp>
{
    return printApiExtensionBinary(emitter, op, op.getAPIName(), op.getDst(), op.getSrc0(), op.getSrc1(),
                                   /*extraScalar=*/Value(), /*sharedTmpBuffer=*/Value(), op.getCalCount());
}

// 标量形态的二元包装。body 在 `VecBinary.cpp`，那里写了为什么模板实参必须写出来。
// `templateResultTensor` 非空时多打一个 dtype（只有 `TrueDivExtends` 要）。
LogicalResult printApiExtensionScalarBinary(CodeEmitter& emitter, Operation* op, StringRef apiName, Value dst,
                                           Value src, Value scalar, Value templateResultTensor, bool isScalarLatter,
                                           Value sharedTmpBuffer, Value calCount);

template <typename ApiExtensionOp>
auto printOperation(CodeEmitter& emitter, ApiExtensionOp op) -> LogicalResultForT<ApiExtensionOp, ascendc::SubsOp>
{
    return printApiExtensionScalarBinary(emitter, op, op.getAPIName(), op.getDst(), op.getSrc(), op.getScalar(),
                                         /*templateResultTensor=*/Value(), op.getIsScalarLatter(),
                                         op.getSharedTmpBuffer(), op.getCalCount());
}

template <typename ApiExtensionOp>
auto printOperation(CodeEmitter& emitter, ApiExtensionOp op) -> LogicalResultForT<ApiExtensionOp, ascendc::DivsOp>
{
    return printApiExtensionScalarBinary(emitter, op, op.getAPIName(), op.getDst(), op.getSrc(), op.getScalar(),
                                         /*templateResultTensor=*/Value(), op.getIsScalarLatter(),
                                         op.getSharedTmpBuffer(), op.getCalCount());
}

// `TrueDivExtends<T, U, LATTER>`：`U` 是结果 dtype，**与 `T` 相同时也要写**。
template <typename ApiExtensionOp>
auto printOperation(CodeEmitter& emitter, ApiExtensionOp op)
    -> LogicalResultForT<ApiExtensionOp, ascendc::TrueDivExtendsOp>
{
    return printApiExtensionScalarBinary(emitter, op, op.getAPIName(), op.getDst(), op.getSrc(), op.getScalar(),
                                         /*templateResultTensor=*/op.getDst(), op.getIsScalarLatter(),
                                         op.getSharedTmpBuffer(), op.getCalCount());
}

// reg 侧不收 tmp buffer——`div_reg_base.h` 那份自己在寄存器里转存。
template <typename ApiExtensionOp>
auto printOperation(CodeEmitter& emitter, ApiExtensionOp op)
    -> LogicalResultForT<ApiExtensionOp, ascendc::DivExtendsRegBaseOp>
{
    return printApiExtensionScalarBinary(emitter, op, op.getAPIName(), op.getDst(), op.getSrc(), op.getScalar(),
                                         /*templateResultTensor=*/Value(), op.getIsScalarLatter(),
                                         /*sharedTmpBuffer=*/Value(), op.getCalCount());
}

// 比较族。两侧的形状差得比这个文件里任何一对都远，所以是两个 body 而不是一个带开关的：
//
//   mem:  CompareExtend(dst, src0, src1, CMPMODE::EQ, count, tmp)
//   reg:  CompareExtend<InT, 1, CMPMODE::EQ>(dst, src0, src1, {count}, {1}, {1})
//
// 同一个 `CMPMODE` 属性，一边落在实参位、一边落在模板实参位。两个 body 各自也服务
// tensor-tensor 与 tensor-scalar 两种形态：标量就是 `src1` 位置上的另一个 SSA 值，
// 打印上没有区别（与 `LogicalAndScalarExtend` 那一对同一个理由）。
//
// **mem 侧的 tmp buffer 在 count 之后**，与 `printApiExtensionBinary` 写死的次序相反，
// 所以不能复用那一个。
LogicalResult printApiExtensionCompare(CodeEmitter& emitter, Operation* op, StringRef apiName, Value dst, Value src0,
                                       Value src1, ascendc::CMPMODE cmpMode, Value calCount, Value sharedTmpBuffer);

// reg 侧。`InT` 取 `src0` 的元素类型——`dim` 是第二个模板实参，写出它就得把前一个也
// 写出来，而 `OutT` 排在最后、照旧由 `dst` 推。三个 brace-init 数组收的是
// `const uint16_t (&)[dim]`（**不是 `cast_reg_base.h` 的 `uint32_t`**），所以字面量要
// 显式转成 `uint16_t`：`{1U}` 在窄化初始化里编不过。
LogicalResult printApiExtensionCompareRegBase(CodeEmitter& emitter, Operation* op, StringRef apiName, Value dst,
                                              Value src0, Value src1, ascendc::CMPMODE cmpMode, Value calCount);

template <typename ApiExtensionOp>
auto printOperation(CodeEmitter& emitter, ApiExtensionOp op)
    -> LogicalResultForT<ApiExtensionOp, ascendc::CompareExtendContiguousOp>
{
    return printApiExtensionCompare(emitter, op, op.getAPIName(), op.getDst(), op.getSrc0(), op.getSrc1(),
                                    op.getCmpMode(), op.getCalCount(), op.getSharedTmpBuffer());
}

template <typename ApiExtensionOp>
auto printOperation(CodeEmitter& emitter, ApiExtensionOp op)
    -> LogicalResultForT<ApiExtensionOp, ascendc::CompareScalarExtendContiguousOp>
{
    return printApiExtensionCompare(emitter, op, op.getAPIName(), op.getDst(), op.getSrc(), op.getScalar(),
                                    op.getCmpMode(), op.getCalCount(), op.getSharedTmpBuffer());
}

template <typename ApiExtensionOp>
auto printOperation(CodeEmitter& emitter, ApiExtensionOp op)
    -> LogicalResultForT<ApiExtensionOp, ascendc::CompareExtendContiguousRegBaseOp>
{
    return printApiExtensionCompareRegBase(emitter, op, op.getAPIName(), op.getDst(), op.getSrc0(), op.getSrc1(),
                                           op.getCmpMode(), op.getCalCount());
}

template <typename ApiExtensionOp>
auto printOperation(CodeEmitter& emitter, ApiExtensionOp op)
    -> LogicalResultForT<ApiExtensionOp, ascendc::CompareScalarExtendContiguousRegBaseOp>
{
    return printApiExtensionCompareRegBase(emitter, op, op.getAPIName(), op.getDst(), op.getSrc(), op.getScalar(),
                                           op.getCmpMode(), op.getCalCount());
}

// `AxpyExtend` is the first binary wrapper carrying a non-tile argument: a
// `const float alpha` that comes from an AFIR attribute rather than an operand.
template <typename ApiExtensionOp>
auto printOperation(CodeEmitter& emitter, ApiExtensionOp op)
    -> LogicalResultForT<ApiExtensionOp, ascendc::AxpyExtendContiguousOp>
{
    return printApiExtensionBinary(emitter, op, op.getAPIName(), op.getDst(), op.getSrc0(), op.getSrc1(),
                                   op.getAlpha(), op.getSharedTmpBuffer(), op.getCalCount());
}

// The scalar-right-operand wrappers reuse the same body: a scalar is just another
// SSA value to name, and it sits in the same argument position the tensor does.
// They need their own overloads only because the accessors are `getSrc`/`getScalar`
// rather than `getSrc0`/`getSrc1` -- these are `VecScalarOp`s, not `BinaryOp`s.
template <typename ApiExtensionOp>
auto printOperation(CodeEmitter& emitter, ApiExtensionOp op)
    -> LogicalResultForT<ApiExtensionOp, ascendc::LogicalAndScalarExtendOp, ascendc::LogicalOrScalarExtendOp>
{
    return printApiExtensionBinary(emitter, op, op.getAPIName(), op.getDst(), op.getSrc(), op.getScalar(),
                                   /*extraScalar=*/Value(), op.getSharedTmpBuffer(), op.getCalCount());
}

template <typename ApiExtensionOp>
auto printOperation(CodeEmitter& emitter, ApiExtensionOp op)
    -> LogicalResultForT<ApiExtensionOp, ascendc::LogicalAndExtendsRegBaseOp>
{
    return printApiExtensionBinary(emitter, op, op.getAPIName(), op.getDst(), op.getSrc(), op.getScalar(),
                                   /*extraScalar=*/Value(), /*sharedTmpBuffer=*/Value(), op.getCalCount());
}

// `BitwiseAndL2Op` is an Autofuse operation on a stock AscendC API, so its emitter
// is the stock L2 binary one -- whose type list lives in an upstream header this
// subtree does not edit. A second overload carrying a one-element list does the
// job: upstream's template SFINAEs out for this operation, so only this one is
// viable and the call is not ambiguous.
template <typename BinaryL2Op>
auto printOperation(CodeEmitter& emitter, BinaryL2Op op) -> LogicalResultForT<BinaryL2Op, ascendc::BitwiseAndL2Op>
{
    emitter.ostream() << ascNamespace << "::" << op.getAPIName();
    printBinaryL2Params(emitter, op);
    return success();
}

// `HypotL2Op` is a stock call too, but with a tmp buffer between the sources and
// the count -- `printBinaryL2Params` prints the four-argument form, so the
// parameter list is spelled out here rather than reused.
template <typename BinaryTmpL2Op>
auto printOperation(CodeEmitter& emitter, BinaryTmpL2Op op) -> LogicalResultForT<BinaryTmpL2Op, ascendc::HypotL2Op>
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc0()) << ", " << emitter.getOrCreateName(op.getSrc1()) << ", "
       << emitter.getOrCreateName(op.getSharedTmpBuffer()) << ", " << emitter.getOrCreateName(op.getCalCount())
       << ")";
    return success();
}

} // namespace ascendc
} // namespace mlir

#endif // ASCIR_TARGET_ASC_AUTOFUSE_VEC_BINARY_H
