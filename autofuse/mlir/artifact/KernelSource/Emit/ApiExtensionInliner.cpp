/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "artifact/KernelSource/Emit/ApiExtensionInliner.h"

#include "llvm/ADT/StringExtras.h"

namespace mlir::ascend::kernel {
namespace {

// Returns the name from an exact #include "name" line, or an empty StringRef.
// System includes, indented conditional includes, and ordinary code are not rewritten.
llvm::StringRef getQuotedIncludeName(llvm::StringRef line) {
  constexpr llvm::StringLiteral kPrefix = "#include \"";
  if (!line.consume_front(kPrefix) || !line.consume_back("\"")) {
    return {};
  }
  return line;
}

}  // namespace

std::string inlineApiExtensionHeaders(llvm::StringRef kernelSource, ApiExtensionProvider provider) {
  std::string result;
  result.reserve(kernelSource.size());
  while (!kernelSource.empty()) {
    const size_t newline = kernelSource.find('\n');
    const llvm::StringRef line =
        kernelSource.take_front(newline == llvm::StringRef::npos ? kernelSource.size() : newline);

    const llvm::StringRef name = getQuotedIncludeName(line);
    const llvm::StringRef text = name.empty() ? llvm::StringRef() : provider(name);
    if (text.empty()) {
      result.append(line.begin(), line.end());
      if (newline != llvm::StringRef::npos) {
        result.push_back('\n');
      }
    } else {
      // Replace the entire include line, including its terminator, with the header text.
      // Normalize the trailing newline so the next line is neither joined nor separated by an extra blank line.
      result.append(text.begin(), text.end());
      if (!text.ends_with("\n")) {
        result.push_back('\n');
      }
    }

    if (newline == llvm::StringRef::npos) {
      break;
    }
    kernelSource = kernelSource.drop_front(newline + 1);
  }
  return result;
}

}  // namespace mlir::ascend::kernel
