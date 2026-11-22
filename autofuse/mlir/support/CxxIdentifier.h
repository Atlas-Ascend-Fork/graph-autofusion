/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AUTOFUSE_MLIR_SUPPORT_CXXIDENTIFIER_H
#define AUTOFUSE_MLIR_SUPPORT_CXXIDENTIFIER_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <cctype>
#include <string>

namespace mlir::afir {

inline bool isCxxKeyword(llvm::StringRef name) {
  static constexpr llvm::StringLiteral keywords[] = {
      "alignas",     "alignof",   "and",        "and_eq",    "asm",      "auto",         "bitand",
      "bitor",       "bool",      "break",      "case",      "catch",    "char",         "char8_t",
      "char16_t",    "char32_t",  "class",      "compl",     "concept",  "const",        "consteval",
      "constexpr",   "constinit", "const_cast", "continue",  "co_await", "co_return",    "co_yield",
      "decltype",    "default",   "delete",     "do",        "double",   "dynamic_cast", "else",
      "enum",        "explicit",  "export",     "extern",    "false",    "float",        "for",
      "friend",      "goto",      "if",         "inline",    "int",      "long",         "mutable",
      "namespace",   "new",       "noexcept",   "not",       "not_eq",   "nullptr",      "operator",
      "or",          "or_eq",     "private",    "protected", "public",   "register",     "reinterpret_cast",
      "requires",    "return",    "short",      "signed",    "sizeof",   "static",       "static_assert",
      "static_cast", "struct",    "switch",     "template",  "this",     "thread_local", "throw",
      "true",        "try",       "typedef",    "typeid",    "typename", "union",        "unsigned",
      "using",       "virtual",   "void",       "volatile",  "wchar_t",  "while",        "xor",
      "xor_eq"};
  return llvm::is_contained(keywords, name);
}

inline bool isValidCxxIdentifier(llvm::StringRef name) {
  if (name.empty() || (!std::isalpha(static_cast<unsigned char>(name.front())) && name.front() != '_')) {
    return false;
  }
  if (!llvm::all_of(name, [](char ch) { return std::isalnum(static_cast<unsigned char>(ch)) || ch == '_'; })) {
    return false;
  }
  return !isCxxKeyword(name);
}

inline std::string makeCxxIdentifier(llvm::StringRef name) {
  std::string result;
  constexpr char hex[] = "0123456789abcdef";
  auto isHexDigit = [](char ch) {
    return std::isdigit(static_cast<unsigned char>(ch)) || (ch >= 'a' && ch <= 'f') || (ch >= 'A' && ch <= 'F');
  };
  for (size_t index = 0; index < name.size(); ++index) {
    unsigned char ch = name[index];
    bool startsEncodedCharacter = ch == '_' && index + 3 < name.size() && name[index + 1] == 'x' &&
                                  isHexDigit(name[index + 2]) && isHexDigit(name[index + 3]);
    if (std::isalnum(ch) || (ch == '_' && !startsEncodedCharacter)) {
      result.push_back(static_cast<char>(ch));
    } else {
      result += "_x";
      result.push_back(hex[ch >> 4]);
      result.push_back(hex[ch & 0xf]);
    }
  }
  if (result.empty()) {
    result = "_x00";
  }
  if (std::isdigit(static_cast<unsigned char>(result.front()))) {
    result.insert(result.begin(), '_');
  }
  if (isCxxKeyword(result)) {
    result.push_back('_');
  }
  return result;
}

}  // namespace mlir::afir

#endif
