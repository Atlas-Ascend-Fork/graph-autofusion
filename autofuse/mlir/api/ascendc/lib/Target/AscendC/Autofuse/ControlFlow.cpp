/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir/Target/Asc/Autofuse/ControlFlow.h"

using namespace mlir;

namespace {
// Escapes `text` into the body of a double-quoted C++ string literal.
// assert(cond && "message") only ever needs to survive `"`, `\`, and
// embedded newlines -- op.getMsg() is diagnostic text, not attacker input,
// but an unescaped `"` or newline would otherwise corrupt the emitted source.
std::string escapeAsStringLiteral(llvm::StringRef text)
{
    std::string escaped;
    escaped.reserve(text.size());
    for (char c : text) {
        switch (c) {
            case '\\': escaped += "\\\\"; break;
            case '"': escaped += "\\\""; break;
            case '\n': escaped += "\\n"; break;
            default: escaped += c; break;
        }
    }
    return escaped;
}
} // namespace

LogicalResult mlir::printOperation(CodeEmitter& emitter, cf::AssertOp op)
{
    auto& os = emitter.ostream();
    os << "assert(" << emitter.getOrCreateName(op.getArg());
    if (!op.getMsg().empty()) {
        os << " && \"" << escapeAsStringLiteral(op.getMsg()) << "\"";
    }
    os << ")";
    return success();
}
