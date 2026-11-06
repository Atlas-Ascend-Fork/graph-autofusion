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

#ifndef ASCIR_TARGET_ASC_AUTOFUSE_PAD_H
#define ASCIR_TARGET_ASC_AUTOFUSE_PAD_H

#include "ascir/Target/Asc/Common.h"

namespace mlir {
namespace ascendc {

//===----------------------------------------------------------------------===//
// Vector layout operations (Pad)
//===----------------------------------------------------------------------===//

// Emits the official `AscendC::Pad` call as two statements:
// The generated statements declare PadParams and call AscendC::Pad.
// The local `PadParams` variable is not a style choice: the official signature
// takes `PadParams&` -- a non-const reference -- so a braced temporary cannot
// bind to it, and legacy lowers the very same local first
// (`pad_api_call.cpp:66`). The middle initializer is `rightPad`; `leftPad` and
// `padValue` stay 0, exactly as legacy writes them.
// The local name carries the `(src, dst)` pair -- the same uniqueness rule as
// `DataCopyNddmaRegBaseOp`: two Pad calls in one scope with distinct
// destinations must not redefine the same local.
LogicalResult printOperation(CodeEmitter& emitter, ascendc::PadOp op);

// Emits `AscendC::DataCopy(<dst>, <src>, KernelUtils::BlkAlign<T>(<count>))`.
// The template is spelled out rather than recomputed arithmetically: legacy
// emits this exact form (`KernelUtils::BlkAlign<dtype>(product)`,
// `codegen_kernel.cpp:1109`), `utils.h` declares it `constexpr`, and keeping
// the text identical keeps the two paths diffable. `<T>` comes from the source
// tensor -- the same "element type from the source" rule as
// `RemovePadExtendOp`'s printer.
LogicalResult printOperation(CodeEmitter& emitter, ascendc::DataCopyBlkAlignedOp op);

} // namespace ascendc
} // namespace mlir

#endif // ASCIR_TARGET_ASC_AUTOFUSE_PAD_H
