/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Autofuse/Index.h"

using namespace mlir;

LogicalResult mlir::printOperation(CodeEmitter& emitter, index::CmpOp op)
{
    if (failed(emitter.emitAssignPrefix(*op.getOperation()))) {
        return failure();
    }
    auto& os = emitter.ostream();
    os << emitter.getOrCreateName(op.getLhs()) << " ";
    switch (op.getPred()) {
        case index::IndexCmpPredicate::EQ:
            os << "==";
            break;
        case index::IndexCmpPredicate::NE:
            os << "!=";
            break;
        case index::IndexCmpPredicate::SLE:
        case index::IndexCmpPredicate::ULE:
            os << "<=";
            break;
        case index::IndexCmpPredicate::SLT:
        case index::IndexCmpPredicate::ULT:
            os << "<";
            break;
        case index::IndexCmpPredicate::SGE:
        case index::IndexCmpPredicate::UGE:
            os << ">=";
            break;
        case index::IndexCmpPredicate::SGT:
        case index::IndexCmpPredicate::UGT:
            os << ">";
            break;
    }
    os << " " << emitter.getOrCreateName(op.getRhs());
    return success();
}
