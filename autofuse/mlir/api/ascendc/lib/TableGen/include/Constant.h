/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOOLS_ASCIR_TBLGEN_CONSTANT_H
#define TOOLS_ASCIR_TBLGEN_CONSTANT_H

#include "llvm/ADT/StringRef.h"
namespace mlir {
namespace asc {
// emit function constant define
static constexpr llvm::StringRef kRetType = "LogicalResult";
static constexpr llvm::StringRef kPrintFuncName = "printOperation";
static constexpr llvm::StringRef kAscDialectNameSpace = "ascendc::";
static constexpr llvm::StringRef kAscDialectName = "AscendC";
static constexpr llvm::StringRef kFuncCallStart = "(";
static constexpr llvm::StringRef kFuncCallEnd = ")";
static constexpr llvm::StringRef kTemplateCallStart = "<";
static constexpr llvm::StringRef kTemplateCallEnd = ">";
static constexpr llvm::StringRef kOutTemplate = R"(os << "template ";)";
static constexpr llvm::StringRef kOutFuncCallStart = R"(os << "(";)";
static constexpr llvm::StringRef kOutTemplateCallStart = R"(os << "<";)";
static constexpr llvm::StringRef kOutTemplateCallEnd = R"(os << ">";)";
static constexpr llvm::StringRef kOutFuncName = R"(os << op.getAPIName();)";
static constexpr llvm::StringRef kOutAscNamespace = R"(os << ascNamespace << "::";)";
static constexpr llvm::StringRef kOutSeparate = R"(os << ", ";)";

// Function format constant define
static constexpr llvm::StringRef kLineBreak = "\n";
static constexpr llvm::StringRef kIndentationSpace = "  ";
static constexpr llvm::StringRef kSpaceSeparated = " ";

// Tablegen constant define
static constexpr llvm::StringRef kGetFunPrefix = "get";
static constexpr llvm::StringRef kTraitName = "traits";
static constexpr llvm::StringRef kAutoEmitAttr = "genEmitter";

static constexpr int64_t kNormalType = 0;
static constexpr int64_t kInferType = 1;
static constexpr int64_t kInferElementType = 2;
static constexpr int64_t kInferEnumType = 3;
static constexpr int64_t kInferValue = 4;
static constexpr int64_t kTemplateType = 5;
static constexpr int64_t kInferTypeAttr = 6;
static constexpr int64_t kPointerValue = -3L;
static constexpr int64_t kPointerToIntValue = -2L;

} // namespace asc
} // namespace mlir

#endif