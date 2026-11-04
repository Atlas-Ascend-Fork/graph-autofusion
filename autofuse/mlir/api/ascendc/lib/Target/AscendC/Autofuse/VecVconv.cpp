/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Autofuse/AutofuseVecVconv.h"
#include "ascir/Target/Asc/Autofuse/AutofuseVecUnary.h"

using namespace mlir;
using namespace mlir::ascendc;

// mem 侧：`CastExtend<InT>(dst, src, tmp, static_cast<uint32_t>(count))`。显式模板实参
// 是 `printApiExtensionUnary` 一贯的做法，取的是**源**的元素类型；`OutT` 由 dst 推出来。
// legacy 一个模板实参都不打（两个都推得出来），这里多打一个 `InT` 不改变选中的重载
// ——4 参与 8 参那两个 `CastExtend` 只靠实参个数就分得开。
LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::CastExtendContiguousOp op)
{
    return printApiExtensionUnary(emitter, op, op.getAPIName(), op.getDst(), op.getSrc(), op.getSharedTmpBuffer(),
                                  op.getCalCount());
}

// reg 侧：`CastExtend(dst, src, {count}, {1U}, {1U})`。
//
// **这条路上第一个 brace-init 数组实参的调用。** 包装收的是三个
// `const uint32_t (&)[dim]`（输出各维长度、输出各维 stride、输入各维 stride），`dim`
// 由 brace-init 的长度推出来——所以这里打一个元素就是在说 `dim == 1`，即连续形态。
// 两个 stride 恒为 1，不是从 IR 里读的，而是这个 op 的助记符（`..._contiguous_...`）
// 本身的含义。
//
// 不打显式模板实参：`InT` / `OutT` 从两个 LocalTensor 推得出来，而 `dim` **只能**从
// brace-init 推——写出 `CastExtend<InT, OutT>` 就得把 `dim` 也写出来，否则模板实参
// 的位置对不上（第三个形参是 `uint8_t dim`）。
LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::CastExtendContiguousRegBaseOp op)
{
    auto& os = emitter.ostream();
    os << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc()) << ", {static_cast<uint32_t>("
       << emitter.getOrCreateName(op.getCalCount()) << ")}, {1U}, {1U})";
    return success();
}

// VF 里的那一条：`AscendC::Reg::Cast<OutT, InT, trait>(dstReg, srcReg, mask)`。
//
// **三个模板实参一个都不能省。** `OutT` / `InT` 推不出来：两个形参的类型是 `S&` / `V&`，
// 推出来的是 `RegTensor<T>` 整体而不是元素类型，而 `T` / `U` 正是要转的那两个元素类型
// （签名见 `kernel_reg_compute_vec_vconv_intf.h:34`，两者的缺省都是 `DefaultType`）。
// `trait` 排在它们之后，所以要写第三个就得把前两个都写出来。
//
// `trait` 是 `api_regbase/utils.h` 里那一组具名 `constexpr CastTrait` 之一的名字（引用类型
// 的非类型模板实参绑不了临时量，详见 `Autofuse/OpVecVconv.td` 上的那段）。挑哪一个由
// `AFIRVectorFuncToRegBase` 定，这里原样打出去。
LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::CastRegOp op)
{
    auto& os = emitter.ostream();
    os << ascNamespace << "::Reg::" << op.getAPIName() << "<";
    FAIL_OR(emitter.emitType(op.getLoc(), op.getDstReg().getType().getElementType()));
    os << ", ";
    FAIL_OR(emitter.emitType(op.getLoc(), op.getSrcReg().getType().getElementType()));
    os << ", " << op.getTrait() << ">(" << emitter.getOrCreateName(op.getDstReg()) << ", "
       << emitter.getOrCreateName(op.getSrcReg()) << ", " << emitter.getOrCreateName(op.getMaskReg()) << ")";
    return success();
}

// `Ceil2IntExtend(dst, src, {count}, {1U}, {1U})`。与上面那条
// `CastExtendContiguousRegBaseOp` 的 body **逐字同形**，只差函数名——两处共用一个辅助
// 函数反而要多传一个 apiName 再在两处各写一行调用，省不下什么，而且会让「这两个是
// 不同的 op」这件事在 emitter 这一层看不出来。三个 brace-init 数组与两个恒 1 的 stride
// 的含义见上面那段。
LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::Ceil2IntExtendRegBaseOp op)
{
    auto& os = emitter.ostream();
    os << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc()) << ", {static_cast<uint32_t>("
       << emitter.getOrCreateName(op.getCalCount()) << ")}, {1U}, {1U})";
    return success();
}
