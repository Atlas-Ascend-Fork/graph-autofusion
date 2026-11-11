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

#include <string>

#include "llvm/ADT/StringMap.h"
#include "gtest/gtest.h"

namespace mlir::ascend::kernel {
namespace {

// Mock AscendCApiRegistry with one header-name-to-text table. Both mem-based
// and reg-based names share the table without collisions or platform selection.
class FakeRegistry {
 public:
  FakeRegistry() {
    headers_["abs.h"] = "// mem-based\nvoid AbsExtend();\n";
    headers_["abs_reg_base.h"] = "// reg-based\nvoid AbsExtend();\n";
    headers_["no_trailing_newline.h"] = "void F();";
  }

  llvm::StringRef lookup(llvm::StringRef name) const {
    auto it = headers_.find(name);
    return it == headers_.end() ? llvm::StringRef() : llvm::StringRef(it->second);
  }

 private:
  llvm::StringMap<std::string> headers_;
};

std::string inlineWith(const FakeRegistry &registry, llvm::StringRef source) {
  return inlineApiExtensionHeaders(source, [&](llvm::StringRef name) { return registry.lookup(name); });
}

TEST(ApiExtensionInlinerTest, ReplacesKnownHeaderWithItsText) {
  FakeRegistry registry;
  EXPECT_EQ(inlineWith(registry, "#include \"kernel_operator.h\"\n#include \"abs.h\"\nvoid kernel();\n"),
            "#include \"kernel_operator.h\"\n// mem-based\nvoid AbsExtend();\nvoid kernel();\n");
}

// The two families are told apart by the header name alone, with one table.
TEST(ApiExtensionInlinerTest, PicksTheFamilyTheNameNames) {
  FakeRegistry registry;
  EXPECT_EQ(inlineWith(registry, "#include \"abs_reg_base.h\"\n"), "// reg-based\nvoid AbsExtend();\n");
}

TEST(ApiExtensionInlinerTest, LeavesUnknownAndSystemIncludesAlone) {
  FakeRegistry registry;
  constexpr llvm::StringLiteral kSource = "#include <cstdint>\n#include \"autofuse_tiling_data.h\"\n";
  EXPECT_EQ(inlineWith(registry, kSource), kSource);
}

// Only the exact shape emitc emits is a candidate, so neither a mention inside
// code nor an indented include in a conditional block is rewritten.
TEST(ApiExtensionInlinerTest, OnlyRewritesWholeIncludeLines) {
  FakeRegistry registry;
  constexpr llvm::StringLiteral kSource = "  #include \"abs.h\"\n// #include \"abs.h\" is inlined\n";
  EXPECT_EQ(inlineWith(registry, kSource), kSource);
}

// A header that does not end in a newline must not splice onto what follows.
TEST(ApiExtensionInlinerTest, SeparatesHeaderTextFromTheNextLine) {
  FakeRegistry registry;
  EXPECT_EQ(inlineWith(registry, "#include \"no_trailing_newline.h\"\nvoid kernel();\n"),
            "void F();\nvoid kernel();\n");
}

TEST(ApiExtensionInlinerTest, PreservesAMissingFinalNewline) {
  FakeRegistry registry;
  EXPECT_EQ(inlineWith(registry, "void kernel();"), "void kernel();");
  EXPECT_EQ(inlineWith(registry, ""), "");
}

}  // namespace
}  // namespace mlir::ascend::kernel
