/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "target/Ascend/Capability/ImplementationRegistry.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir::ascend::instruction {

ImplementationRegistry::ImplementationRegistry(MLIRContext &context) : context_(context) {}

LogicalResult ImplementationRegistry::freeze() {
  if (frozen_) {
    return success();
  }

  llvm::sort(entries_, [](const RegistrationEntry &lhs, const RegistrationEntry &rhs) {
    if (lhs.sourceOpName != rhs.sourceOpName) {
      return lhs.sourceOpName < rhs.sourceOpName;
    }
    if (lhs.key.kind != rhs.key.kind) {
      return lhs.key.kind < rhs.key.kind;
    }
    return lhs.key.id.value < rhs.key.id.value;
  });
  for (auto &entry : entries_) {
    entry.key.sourceOp = OperationName(entry.sourceOpName, &context_);
  }
  for (size_t index = 1; index < entries_.size(); ++index) {
    if (entries_[index - 1].key == entries_[index].key) {
      return failure();
    }
  }

  size_t begin = 0;
  while (begin < entries_.size()) {
    size_t end = begin + 1;
    while (end < entries_.size() && entries_[begin].sourceOpName == entries_[end].sourceOpName) {
      ++end;
    }
    entryRanges_.emplace(entries_[begin].sourceOpName, std::make_pair(begin, end));
    begin = end;
  }
  frozen_ = true;
  return success();
}

bool ImplementationRegistry::isFrozen() const {
  return frozen_;
}

bool ImplementationRegistry::contains(const ImplementationKey &key) const {
  if (!frozen_) {
    return false;
  }
  auto range = entryRanges_.find(key.sourceOp.getStringRef().str());
  if (range == entryRanges_.end()) {
    return false;
  }
  for (size_t index = range->second.first; index < range->second.second; ++index) {
    const auto &entry = entries_[index];
    if (entry.key.kind == key.kind && entry.key.id == key.id) {
      return true;
    }
  }
  return false;
}

bool ImplementationRegistry::containsSourceOp(OperationName sourceOp) const {
  return frozen_ && entryRanges_.find(sourceOp.getStringRef().str()) != entryRanges_.end();
}

}  // namespace mlir::ascend::instruction
