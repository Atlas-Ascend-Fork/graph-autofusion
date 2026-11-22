/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "target/Ascend/Capability/CapabilityQuery.h"

#include <cstddef>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Operation.h"

namespace mlir::ascend::instruction {

CapabilityQuery::CapabilityQuery(const ImplementationRegistry &registry) : registry_(registry) {}

bool CapabilityQuery::isValidOperation(Operation *op) const {
  return op != nullptr && registry_.frozen_ && op->getContext() == &registry_.context_;
}

namespace {
bool hasMatchingCapabilityDetail(const CapabilityQueryRequest &request) {
  if ((request.kind == CapabilityKind::ScalarInput && !std::holds_alternative<ScalarInputQuery>(request.detail)) ||
      (request.kind == CapabilityKind::BroadcastInline &&
       !std::holds_alternative<BroadcastInlineQuery>(request.detail)) ||
      (request.kind == CapabilityKind::Inplace && !std::holds_alternative<InplaceQuery>(request.detail))) {
    return false;
  }
  return true;
}
}  // namespace

bool CapabilityQuery::isValidCapabilityRequest(const CapabilityQueryRequest &request) const {
  Operation *op = request.op;
  if (!isValidOperation(op) || (request.implementation && request.implementation->sourceOp != op->getName())) {
    return false;
  }
  if (!hasMatchingCapabilityDetail(request)) {
    return false;
  }

  if (const auto *query = std::get_if<ScalarInputQuery>(&request.detail)) {
    if (request.kind != CapabilityKind::ScalarInput ||
        !llvm::all_of(query->scalarOperandIndices, [&](uint32_t index) { return index < op->getNumOperands(); })) {
      return false;
    }
  } else if (const auto *query = std::get_if<BroadcastInlineQuery>(&request.detail)) {
    if (request.kind != CapabilityKind::BroadcastInline || query->sourceOperandIndex >= op->getNumOperands() ||
        query->destinationOperandIndex >= op->getNumOperands()) {
      return false;
    }
  } else if (const auto *query = std::get_if<InplaceQuery>(&request.detail)) {
    if (request.kind != CapabilityKind::Inplace || query->inputOperandIndex >= op->getNumOperands() ||
        query->outputResultIndex >= op->getNumResults()) {
      return false;
    }
  }
  return true;
}

FailureOr<Candidates> CapabilityQuery::collectCapabilityCandidates(const CapabilityQueryRequest &request,
                                                                   const QueryContext &context) const {
  Operation *op = request.op;
  llvm::SmallVector<Candidate, 2> candidates;
  auto range = registry_.entryRanges_.find(op->getName().getStringRef().str());
  if (range == registry_.entryRanges_.end()) {
    if (request.implementation) {
      return failure();
    }
    return Candidates(std::move(candidates));
  }
  for (size_t index = range->second.first; index < range->second.second; ++index) {
    const ImplementationRegistry::RegistrationEntry &entry = registry_.entries_[index];
    if (request.implementation && entry.key != *request.implementation) {
      continue;
    }
    ImplementationKey key{op->getName(), entry.key.kind, entry.key.id};
    candidates.push_back({std::move(key), entry.provider->queryCapability(request.kind, op, request.detail, context)});
  }
  if (request.implementation && candidates.empty()) {
    return failure();
  }
  return Candidates(std::move(candidates));
}

FailureOr<Candidates> CapabilityQuery::collectLoweringCandidates(Operation *op, const QueryContext &context) const {
  llvm::SmallVector<Candidate, 2> candidates;
  auto range = registry_.entryRanges_.find(op->getName().getStringRef().str());
  if (range == registry_.entryRanges_.end()) {
    return Candidates(std::move(candidates));
  }
  for (size_t index = range->second.first; index < range->second.second; ++index) {
    const ImplementationRegistry::RegistrationEntry &entry = registry_.entries_[index];
    ImplementationKey key{op->getName(), entry.key.kind, entry.key.id};
    candidates.push_back({std::move(key), entry.provider->queryLowering(op, context)});
  }
  return Candidates(std::move(candidates));
}

FailureOr<Candidates> CapabilityQuery::queryCapability(const CapabilityQueryRequest &request,
                                                       const QueryContext &context) const {
  if (!isValidCapabilityRequest(request)) {
    return failure();
  }
  return collectCapabilityCandidates(request, context);
}

FailureOr<Candidates> CapabilityQuery::queryLowering(Operation *op, const QueryContext &context) const {
  if (!isValidOperation(op)) {
    return failure();
  }
  return collectLoweringCandidates(op, context);
}

}  // namespace mlir::ascend::instruction
