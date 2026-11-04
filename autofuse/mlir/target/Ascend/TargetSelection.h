/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_TARGET_ASCEND_TARGETSELECTION_H
#define AUTOFUSE_MLIR_TARGET_ASCEND_TARGETSELECTION_H

#include <optional>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Support/LogicalResult.h"

namespace llvm {
class raw_ostream;
}

namespace mlir::ascend {

inline constexpr llvm::StringLiteral kTargetOptionName = "target";
inline constexpr llvm::StringLiteral kTargetProfileOptionName = "target-profile";

// af-opt's own --target/--target-profile flags. Defined non-anonymously here
// (rather than in tools/af-opt/main.cpp) so that passes living outside af-opt,
// such as the session-scoped AFIR-to-AscendC pipeline passes, can resolve the
// same command-line-selected target from within runOnOperation()/initialize()
// without af-opt needing to inject it explicitly: mlir::Pass::Option<> fields
// cannot backstop these as independent top-level flags in bare (non
// -pass-pipeline) invocation, so a shared extern cl::opt, the same idiom
// LLVM itself uses for cross-pass flags (see llvm/include/llvm/IR/PrintPasses.h's
// PrintChanged), is the only way to keep the existing bare --target=... syntax.
//
// Consumers outside this directory must not read these directly: go through
// TargetProfileRegistry.h's getSelectedTargetProfile, which keeps the
// command-line dependency inside this layer and parses the profile once.
extern llvm::cl::opt<std::string> targetOption;
extern llvm::cl::opt<std::string> targetProfileOption;

enum class TargetSelectionMode {
  BuiltinPlatform,
  ExternalProfile,
};

enum class TargetSelectionRequirement {
  Optional,
  Required,
};

class TargetSelection {
 public:
  static TargetSelection builtinPlatform(std::string platform);
  static TargetSelection externalProfile(std::string path);

  TargetSelectionMode getMode() const;
  llvm::StringRef getValue() const;

 private:
  TargetSelection(TargetSelectionMode mode, std::string value);

  TargetSelectionMode mode_;
  std::string value_;
};

FailureOr<std::optional<TargetSelection>> resolveTargetSelection(llvm::StringRef builtinPlatform,
                                                                 llvm::StringRef externalProfile,
                                                                 TargetSelectionRequirement requirement,
                                                                 llvm::raw_ostream &diagnostic);

}  // namespace mlir::ascend

#endif  // AUTOFUSE_MLIR_TARGET_ASCEND_TARGETSELECTION_H
