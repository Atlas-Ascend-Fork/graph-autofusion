/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "target/Ascend/TargetProfileRegistry.h"

#include <map>
#include <mutex>
#include <string>
#include <utility>

#include "target/Ascend/TargetProfileLoader.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::ascend {
namespace {

std::string makeCacheKey(const TargetSelection &selection) {
  // The mode prefix keeps a builtin platform named like a path from aliasing an
  // external profile file that happens to have the same string.
  return (selection.getMode() == TargetSelectionMode::BuiltinPlatform ? "builtin:" : "profile:") +
         selection.getValue().str();
}

std::mutex &cacheMutex() {
  static std::mutex mutex;
  return mutex;
}

std::map<std::string, std::shared_ptr<const TargetProfile>> &cache() {
  static std::map<std::string, std::shared_ptr<const TargetProfile>> profiles;
  return profiles;
}

}  // namespace

FailureOr<std::shared_ptr<const TargetProfile>> getTargetProfile(const TargetSelection &selection,
                                                                 llvm::raw_ostream &diagnostic) {
  const std::string key = makeCacheKey(selection);
  {
    std::lock_guard<std::mutex> guard(cacheMutex());
    auto entry = cache().find(key);
    if (entry != cache().end()) {
      return entry->second;
    }
  }

  // Loaded outside the lock: parsing touches the filesystem, and a failure must
  // not be cached so a corrected profile file is picked up on the next attempt.
  auto loaded = TargetProfileLoader::load(selection, diagnostic);
  if (failed(loaded)) {
    return failure();
  }
  auto profile = std::make_shared<const TargetProfile>(std::move(*loaded));

  std::lock_guard<std::mutex> guard(cacheMutex());
  // Another thread may have inserted the same key meanwhile; keep whichever
  // instance is already published so callers always observe one object.
  return cache().emplace(key, std::move(profile)).first->second;
}

FailureOr<std::shared_ptr<const TargetProfile>> getSelectedTargetProfile(TargetSelectionRequirement requirement,
                                                                         llvm::raw_ostream &diagnostic) {
  auto selection = resolveTargetSelection(targetOption, targetProfileOption, requirement, diagnostic);
  if (failed(selection)) {
    return failure();
  }
  if (!selection->has_value()) {
    // Only reachable for Optional: Required already failed above.
    return std::shared_ptr<const TargetProfile>();
  }
  return getTargetProfile(selection->value(), diagnostic);
}

}  // namespace mlir::ascend
