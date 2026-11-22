/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "target/Ascend/Capability/Capability.h"

#include <cassert>
#include <utility>

#include "llvm/ADT/STLExtras.h"

namespace mlir::ascend::instruction {

bool operator==(const ImplementationId &lhs, const ImplementationId &rhs) {
  return lhs.value == rhs.value;
}

bool operator!=(const ImplementationId &lhs, const ImplementationId &rhs) {
  return !(lhs == rhs);
}

bool operator==(const ImplementationKey &lhs, const ImplementationKey &rhs) {
  return lhs.sourceOp.getStringRef() == rhs.sourceOp.getStringRef() && lhs.kind == rhs.kind && lhs.id == rhs.id;
}

bool operator!=(const ImplementationKey &lhs, const ImplementationKey &rhs) {
  return !(lhs == rhs);
}

CapabilitySupport::CapabilitySupport(SupportState state, CapabilityReason reason,
                                     llvm::SmallVector<CapabilityConstraint, 4> constraints)
    : state_(state), reason_(reason), constraints_(std::move(constraints)) {}

CapabilitySupport CapabilitySupport::supported(llvm::SmallVector<CapabilityConstraint, 4> constraints) {
  return CapabilitySupport(SupportState::Supported, CapabilityReason::None, std::move(constraints));
}

CapabilitySupport CapabilitySupport::unsupported(CapabilityReason reason,
                                                 llvm::SmallVector<CapabilityConstraint, 4> constraints) {
  assert(reason != CapabilityReason::None && "unsupported capability requires a reason");
  return CapabilitySupport(SupportState::Unsupported, reason, std::move(constraints));
}

CapabilitySupport CapabilitySupport::undetermined(CapabilityReason reason,
                                                  llvm::SmallVector<CapabilityConstraint, 4> constraints) {
  assert(reason != CapabilityReason::None && "undetermined capability requires a reason");
  return CapabilitySupport(SupportState::Undetermined, reason, std::move(constraints));
}

SupportState CapabilitySupport::getState() const {
  return state_;
}

CapabilityReason CapabilitySupport::getReason() const {
  return reason_;
}

llvm::ArrayRef<CapabilityConstraint> CapabilitySupport::getConstraints() const {
  return constraints_;
}

Candidates::Candidates(llvm::SmallVector<Candidate, 2> candidates) : candidates_(std::move(candidates)) {
  llvm::sort(candidates_, [](const Candidate &lhs, const Candidate &rhs) {
    if (lhs.key.kind != rhs.key.kind) {
      return lhs.key.kind < rhs.key.kind;
    }
    if (lhs.key.id.value != rhs.key.id.value) {
      return lhs.key.id.value < rhs.key.id.value;
    }
    return lhs.key.sourceOp.getStringRef() < rhs.key.sourceOp.getStringRef();
  });
}

llvm::ArrayRef<Candidate> Candidates::getAll() const {
  return candidates_;
}

llvm::SmallVector<Candidate, 2> Candidates::getSupported() const {
  llvm::SmallVector<Candidate, 2> supported;
  for (const Candidate &candidate : candidates_) {
    if (candidate.support.getState() == SupportState::Supported) {
      supported.push_back(candidate);
    }
  }
  return supported;
}

bool Candidates::hasUndetermined() const {
  return llvm::any_of(candidates_, [](const Candidate &candidate) {
    return candidate.support.getState() == SupportState::Undetermined;
  });
}

}  // namespace mlir::ascend::instruction
