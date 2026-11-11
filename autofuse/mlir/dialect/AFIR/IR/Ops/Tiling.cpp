/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dialect/AFIR/IR/AFIR.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::afir {

// 参数名同时用于 expr 符号引用和目标 ABI 字段访问，因此必须是合法标识符。
// 在此拒绝空格、数字开头等非法形式，避免到 emitasc.member 生成 C++ 时才报告错误。
LogicalResult TilingParamOp::verify() {
  llvm::StringRef name = getSymbol();
  if (name.empty()) {
    return emitOpError() << kInvalidDescriptorDiagnostic << ": tiling parameter name must not be empty";
  }
  auto isIdentifierBody = [](char c) { return llvm::isAlnum(c) || c == '_'; };
  if (!(llvm::isAlpha(name.front()) || name.front() == '_') || !llvm::all_of(name, isIdentifierBody)) {
    return emitOpError() << kInvalidDescriptorDiagnostic << ": tiling parameter name must be an identifier, got '"
                         << name << "'";
  }
  return success();
}

}  // namespace mlir::afir
