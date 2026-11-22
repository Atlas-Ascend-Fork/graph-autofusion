/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_TARGET_ASCEND_CAPABILITY_IMPLEMENTATIONSELECTION_H
#define AUTOFUSE_MLIR_TARGET_ASCEND_CAPABILITY_IMPLEMENTATIONSELECTION_H

#include "target/Ascend/Capability/Capability.h"

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/STLFunctionalExtras.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class MLIRContext;
class Pass;
class Operation;
class RewritePatternSet;
}  // namespace mlir

namespace mlir::ascend::instruction {

inline constexpr llvm::StringLiteral kImplementationAttrName = "ascend.implementation";
inline constexpr llvm::StringLiteral kImplementationKindKey = "kind";
inline constexpr llvm::StringLiteral kImplementationIdKey = "id";
inline constexpr llvm::StringLiteral kAscendCImplementationKind = "ascendc_api";
inline constexpr llvm::StringLiteral kCCEIntrinsicImplementationKind = "cce_intrinsic";
inline constexpr llvm::StringLiteral kPTOISAImplementationKind = "pto_isa";

class ImplementationRegistry;
struct PatternBuildContext;

class SelectionPolicy {
 public:
  virtual ~SelectionPolicy() = default;

  virtual FailureOr<ImplementationKey> choose(Operation *op, llvm::ArrayRef<Candidate> supported,
                                              const QueryContext &context) const = 0;
};

class SelectionPolicyRegistry {
 public:
  explicit SelectionPolicyRegistry(MLIRContext &context);

  template <typename SourceOp>
  LogicalResult registerPolicy(std::shared_ptr<const SelectionPolicy> policy) {
    llvm::StringRef sourceOpName = SourceOp::getOperationName();
    if (sourceOpName.empty()) {
      return failure();
    }
    return registerPolicy(OperationName(sourceOpName, &context_), std::move(policy));
  }

  LogicalResult registerPolicy(OperationName sourceOp, std::shared_ptr<const SelectionPolicy> policy);
  LogicalResult freeze();
  bool isFrozen() const;
  const SelectionPolicy *lookup(OperationName sourceOp) const;

 private:
  MLIRContext &context_;
  std::map<std::string, std::shared_ptr<const SelectionPolicy>> policies_;
  bool frozen_ = false;
};

FailureOr<ImplementationKey> selectImplementation(Operation *op, const Candidates &candidates,
                                                  const QueryContext &context, const SelectionPolicy *policy);

FailureOr<ImplementationKey> getSelectedImplementation(Operation *op);
LogicalResult setSelectedImplementation(Operation *op, const ImplementationKey &selected);
void clearSelectedImplementation(Operation *op);
LogicalResult validateSelectedImplementations(Operation *root, const ImplementationRegistry &registry,
                                              llvm::function_ref<bool(Operation *)> requiresSelection);

void populateEnabledPatterns(RewritePatternSet &patterns, const ImplementationRegistry &registry,
                             const PatternBuildContext &context);

std::unique_ptr<Pass> createSelectImplementationPass(std::shared_ptr<const ImplementationRegistry> registry,
                                                     std::shared_ptr<const TargetProfile> target,
                                                     std::shared_ptr<const SelectionPolicy> policy = nullptr);

std::unique_ptr<Pass> createSelectImplementationPass(std::shared_ptr<const ImplementationRegistry> registry,
                                                     std::shared_ptr<const TargetProfile> target,
                                                     std::shared_ptr<const SelectionPolicyRegistry> policies,
                                                     std::shared_ptr<const SelectionPolicy> fallbackPolicy);

}  // namespace mlir::ascend::instruction

#endif  // AUTOFUSE_MLIR_TARGET_ASCEND_CAPABILITY_IMPLEMENTATIONSELECTION_H
