/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Autofuse/VecTernary.h"

using namespace mlir;

LogicalResult mlir::ascendc::printApiExtensionTernary(CodeEmitter& emitter, Operation* op, StringRef apiName, Value dst,
                                                      Value condition, Value src0, Value src1, Value calCount,
                                                      Value sharedTmpBuffer)
{
    auto& os = emitter.ostream();
    // No `AscendC::` qualifier: the extension wrappers are global functions.
    os << apiName << "(" << emitter.getOrCreateName(dst) << ", " << emitter.getOrCreateName(condition) << ", "
       << emitter.getOrCreateName(src0) << ", " << emitter.getOrCreateName(src1) << ", "
       << "static_cast<uint32_t>(" << emitter.getOrCreateName(calCount) << ")";
    if (sharedTmpBuffer) {
        os << ", " << emitter.getOrCreateName(sharedTmpBuffer);
    }
    os << ")";
    return success();
}
