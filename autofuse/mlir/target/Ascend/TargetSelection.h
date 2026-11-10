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

// Shared af-opt --target/--target-profile flags. Session-scoped passes outside
// af-opt resolve them during runOnOperation()/initialize(). mlir::Pass::Option<>
// does not expose independent top-level flags for bare (non -pass-pipeline)
// invocation, so shared extern cl::opt values preserve the --target=... syntax.
// LLVM uses the same pattern for cross-pass flags, such as PrintChanged in
// llvm/include/llvm/IR/PrintPasses.h.
//
// Consumers outside this directory use getSelectedTargetProfile in
// TargetProfileRegistry.h rather than reading these options directly. This keeps
// the command-line dependency in this layer and parses each profile once.
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
