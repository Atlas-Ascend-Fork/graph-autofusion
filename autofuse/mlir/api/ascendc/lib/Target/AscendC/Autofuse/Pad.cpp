/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details.
 * You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Autofuse/Pad.h"

using namespace mlir;
using namespace mlir::ascendc;

//===----------------------------------------------------------------------===//
// Vector layout operations (Pad)
//===----------------------------------------------------------------------===//

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::PadOp op)
{
    auto& os = emitter.ostream();

    // 局部变量名带 `(src, dst)` 对：同一作用域里两条 Pad 落到不同 dst 上时不能重定义
    // 同一个 `padParams`——与 `DataCopyNddmaRegBaseOp` 的 `_nddma_to_<dst>` 同一条规则。
    auto suffix = (emitter.getOrCreateName(op.getSrc()) + "_pad_to_" + emitter.getOrCreateName(op.getDst())).str();

    // `AscendC::Pad`（`lib/pad/pad.h:53`）的 `PadParams&` 与 `PadTiling&` 都是非常量引用，
    // 所以逐字照 legacy 的两语句形态（`pad_api_call.cpp:66-73`）：先落局部 `PadParams`
    // （左补 0、右补 padNum、补值 0），再发调用；`PadTiling` 那份局部拷贝由下降侧的
    // `emitasc.member` 备好，从操作数直接取名字。
    os << "AscendC::PadParams padParams_" << suffix << " = {0, static_cast<uint16_t>("
       << emitter.getOrCreateName(op.getPadNum()) << "), 0};\n"
       << "AscendC::Pad(" << emitter.getOrCreateName(op.getDst()) << ", " << emitter.getOrCreateName(op.getSrc())
       << ", padParams_" << suffix << ", " << emitter.getOrCreateName(op.getTmpBuffer()) << ", "
       << emitter.getOrCreateName(op.getApiTilingData()) << ")";

    return success();
}

LogicalResult mlir::ascendc::printOperation(CodeEmitter& emitter, ascendc::DataCopyBlkAlignedOp op)
{
    auto& os = emitter.ostream();

    // 模板实参取**源**的元素类型，与 `RemovePadExtendOp` 的 printer 同一条口径。
    auto tensorType = dyn_cast<LocalTensorType>(op.getSrc().getType());
    if (!tensorType) {
        return op->emitOpError(op.getAPIName()) << " requires a local tensor source";
    }

    // `KernelUtils::BlkAlign<T>(n)`（`utils.h:172`）把元素数抬到 32B 对齐；legacy 逐字发
    // 这个模板形态，不在这里重算算术。形参 `size` 是 `uint32_t`，值来自 `index` 的 SSA 值。
    // 三个闭括号：cast、`BlkAlign`、外层 `DataCopy`（尾随分号由翻译层补）。
    os << "AscendC::" << op.getAPIName() << "(" << emitter.getOrCreateName(op.getDst()) << ", "
       << emitter.getOrCreateName(op.getSrc()) << ", KernelUtils::BlkAlign<";
    FAIL_OR(emitter.emitType(op.getLoc(), tensorType.getElementType()));
    os << ">(static_cast<uint32_t>(" << emitter.getOrCreateName(op.getCount()) << ")))";

    return success();
}
