/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "target/Ascend/TargetSelection.h"

#include <optional>
#include <string>
#include <utility>

#include "llvm/Support/raw_ostream.h"

namespace mlir::ascend {

llvm::cl::opt<std::string> targetOption(llvm::StringRef(kTargetOptionName),
                                        llvm::cl::desc("Load the builtin Ascend target profile"),
                                        llvm::cl::value_desc("platform"));
llvm::cl::opt<std::string> targetProfileOption(llvm::StringRef(kTargetProfileOptionName),
                                               llvm::cl::desc("Load an Ascend target profile from an INI file"),
                                               llvm::cl::value_desc("file"));

TargetSelection::TargetSelection(TargetSelectionMode mode, std::string value) : mode_(mode), value_(std::move(value)) {}

TargetSelection TargetSelection::builtinPlatform(std::string platform) {
  return TargetSelection(TargetSelectionMode::BuiltinPlatform, std::move(platform));
}

TargetSelection TargetSelection::externalProfile(std::string path) {
  return TargetSelection(TargetSelectionMode::ExternalProfile, std::move(path));
}

TargetSelectionMode TargetSelection::getMode() const {
  return mode_;
}

llvm::StringRef TargetSelection::getValue() const {
  return value_;
}

FailureOr<std::optional<TargetSelection>> resolveTargetSelection(llvm::StringRef builtinPlatform,
                                                                 llvm::StringRef externalProfile,
                                                                 TargetSelectionRequirement requirement,
                                                                 llvm::raw_ostream &diagnostic) {
  if (!builtinPlatform.empty() && !externalProfile.empty()) {
    diagnostic << "TargetSelectionConflict: --" << kTargetOptionName << " and --" << kTargetProfileOptionName
               << " cannot be specified together\n";
    return failure();
  }
  if (!builtinPlatform.empty()) {
    return std::optional<TargetSelection>(TargetSelection::builtinPlatform(builtinPlatform.str()));
  }
  if (!externalProfile.empty()) {
    return std::optional<TargetSelection>(TargetSelection::externalProfile(externalProfile.str()));
  }
  if (requirement == TargetSelectionRequirement::Required) {
    diagnostic << "TargetSelectionConflict: one of --" << kTargetOptionName << " or --" << kTargetProfileOptionName
               << " is required\n";
    return failure();
  }
  return std::optional<TargetSelection>();
}

}  // namespace mlir::ascend
