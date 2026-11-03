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

// 名字要同时活在两个地方：`#afir.expr` 的符号表（尺寸表达式按名字引用它）与目标 ABI 的
// 字段名（AscendC 侧是 C++ 结构体成员）。两边的交集就是标识符，因此在这里判掉其余形态。
//
// 放宽到任意字符串的代价是静默的：一个带空格或以数字开头的名字在 IR 里合法，到
// `emitasc.member` 才变成生成不出来的 C++，而那时诊断已经离写错的地方很远了。
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
