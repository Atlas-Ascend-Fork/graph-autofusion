/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "target/Ascend/Capability/ImplementationSelection.h"

#include "target/Ascend/Capability/CapabilityQuery.h"
#include "target/Ascend/Capability/ImplementationRegistry.h"
#include "support/CapabilityShortfall.h"

#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"

namespace mlir::ascend::instruction {
namespace {

llvm::StringRef stringifyImplementationKind(ImplementationKind kind) {
  switch (kind) {
    case ImplementationKind::AscendC:
      return kAscendCImplementationKind;
    case ImplementationKind::CCEIntrinsic:
      return kCCEIntrinsicImplementationKind;
    case ImplementationKind::PTOISA:
      return kPTOISAImplementationKind;
  }
  return "";
}

std::optional<ImplementationKind> parseImplementationKind(llvm::StringRef value) {
  return llvm::StringSwitch<std::optional<ImplementationKind>>(value)
      .Case(kAscendCImplementationKind, ImplementationKind::AscendC)
      .Case(kCCEIntrinsicImplementationKind, ImplementationKind::CCEIntrinsic)
      .Case(kPTOISAImplementationKind, ImplementationKind::PTOISA)
      .Default(std::nullopt);
}

bool containsKey(llvm::ArrayRef<Candidate> candidates, const ImplementationKey &key) {
  return llvm::any_of(candidates, [&](const Candidate &candidate) { return candidate.key == key; });
}

FailureOr<ImplementationKey> validateSelectedImplementation(Operation *op, const ImplementationRegistry &registry) {
  auto selected = getSelectedImplementation(op);
  if (failed(selected)) {
    return failure();
  }
  if (!registry.contains(*selected)) {
    op->emitOpError("selected Ascend implementation is not registered for this operation");
    return failure();
  }
  return *selected;
}

struct SelectImplementationPass : public PassWrapper<SelectImplementationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SelectImplementationPass)

  SelectImplementationPass(std::shared_ptr<const ImplementationRegistry> registry,
                           std::shared_ptr<const TargetProfile> target,
                           std::shared_ptr<const SelectionPolicyRegistry> policies,
                           std::shared_ptr<const SelectionPolicy> fallbackPolicy)
      : registry_(std::move(registry)),
        target_(std::move(target)),
        policies_(std::move(policies)),
        fallbackPolicy_(std::move(fallbackPolicy)) {}

  SelectImplementationPass(const SelectImplementationPass &other)
      : PassWrapper(other),
        registry_(other.registry_),
        target_(other.target_),
        policies_(other.policies_),
        fallbackPolicy_(other.fallbackPolicy_) {}

  llvm::StringRef getArgument() const final {
    return "ascend-select-implementation";
  }

  llvm::StringRef getDescription() const final {
    return "Select Ascend instruction implementations for registered operations";
  }

  void runOnOperation() override {
    if (!registry_ || !target_ || !registry_->isFrozen() || (policies_ && !policies_->isFrozen())) {
      getOperation().emitError("requires a frozen Ascend implementation registry and target profile");
      signalPassFailure();
      return;
    }

    CapabilityQuery query(*registry_);
    QueryContext queryContext{*target_};
    WalkResult result = getOperation().walk([&](Operation *op) -> WalkResult {
      std::optional<ImplementationKey> previousSelection;
      if (op->getDiscardableAttr(kImplementationAttrName)) {
        FailureOr<ImplementationKey> selected = validateSelectedImplementation(op, *registry_);
        if (failed(selected)) {
          return WalkResult::interrupt();
        }
        previousSelection = *selected;
      }

      if (!registry_->containsSourceOp(op->getName())) {
        return WalkResult::advance();
      }

      FailureOr<Candidates> candidates = query.queryLowering(op, queryContext);
      if (failed(candidates)) {
        return WalkResult::interrupt();
      }
      if (previousSelection && !containsKey(candidates->getSupported(), *previousSelection)) {
        op->emitOpError("selected Ascend implementation is no longer supported for the current operation and target");
        return WalkResult::interrupt();
      }
      const SelectionPolicy *selectionPolicy = policies_ ? policies_->lookup(op->getName()) : nullptr;
      if (selectionPolicy == nullptr) {
        selectionPolicy = fallbackPolicy_.get();
      }
      FailureOr<ImplementationKey> selected = selectImplementation(op, *candidates, queryContext, selectionPolicy);
      if (failed(selected)) {
        return WalkResult::interrupt();
      }
      if (failed(setSelectedImplementation(op, *selected))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }

 private:
  std::shared_ptr<const ImplementationRegistry> registry_;
  std::shared_ptr<const TargetProfile> target_;
  std::shared_ptr<const SelectionPolicyRegistry> policies_;
  std::shared_ptr<const SelectionPolicy> fallbackPolicy_;
};

}  // namespace

SelectionPolicyRegistry::SelectionPolicyRegistry(MLIRContext &context) : context_(context) {}

LogicalResult SelectionPolicyRegistry::registerPolicy(OperationName sourceOp,
                                                      std::shared_ptr<const SelectionPolicy> policy) {
  if (frozen_ || sourceOp.getStringRef().empty() || !policy || policies_.count(sourceOp.getStringRef().str()) != 0) {
    return failure();
  }
  policies_.emplace(sourceOp.getStringRef().str(), std::move(policy));
  return success();
}

LogicalResult SelectionPolicyRegistry::freeze() {
  if (frozen_) {
    return success();
  }
  frozen_ = true;
  return success();
}

bool SelectionPolicyRegistry::isFrozen() const {
  return frozen_;
}

const SelectionPolicy *SelectionPolicyRegistry::lookup(OperationName sourceOp) const {
  if (!frozen_) {
    return nullptr;
  }
  auto it = policies_.find(sourceOp.getStringRef().str());
  return it == policies_.end() ? nullptr : it->second.get();
}

FailureOr<ImplementationKey> selectImplementation(Operation *op, const Candidates &candidates,
                                                  const QueryContext &context, const SelectionPolicy *policy) {
  if (op == nullptr) {
    return failure();
  }

  llvm::SmallVector<Candidate, 2> supported = candidates.getSupported();
  if (supported.empty()) {
    // 能力层说不支持 —— 整个 capability 层的拒收都收束在这一处，所以标记也只打在这里。
    // 它与下面两条的性质不同：下面两条（多个实现没有策略、策略返回了候选集外的实现）是
    // 注册表与策略的自相矛盾，回落会把它们藏起来。
    mlir::afir::markCapabilityShortfall(op->emitOpError("has no supported Ascend implementation"));
    return failure();
  }
  if (supported.size() == 1) {
    return supported.front().key;
  }
  if (policy == nullptr) {
    op->emitOpError("has multiple supported Ascend implementations but no selection policy");
    return failure();
  }

  FailureOr<ImplementationKey> selected = policy->choose(op, supported, context);
  if (failed(selected)) {
    return failure();
  }
  if (!containsKey(supported, *selected)) {
    op->emitOpError("selection policy returned an implementation outside the supported candidate set");
    return failure();
  }
  return *selected;
}

FailureOr<ImplementationKey> getSelectedImplementation(Operation *op) {
  if (op == nullptr) {
    return failure();
  }

  auto selected = dyn_cast_or_null<DictionaryAttr>(op->getDiscardableAttr(kImplementationAttrName));
  if (!selected) {
    op->emitOpError() << "requires " << kImplementationAttrName << " dictionary attribute";
    return failure();
  }

  auto kindAttr = dyn_cast_or_null<StringAttr>(selected.get(kImplementationKindKey));
  auto idAttr = dyn_cast_or_null<StringAttr>(selected.get(kImplementationIdKey));
  if (!kindAttr || !idAttr || idAttr.getValue().empty()) {
    op->emitOpError() << "has malformed " << kImplementationAttrName << " attribute";
    return failure();
  }

  std::optional<ImplementationKind> kind = parseImplementationKind(kindAttr.getValue());
  if (!kind) {
    op->emitOpError("has unknown Ascend implementation kind");
    return failure();
  }
  return ImplementationKey{op->getName(), *kind, {idAttr.getValue().str()}};
}

LogicalResult setSelectedImplementation(Operation *op, const ImplementationKey &selected) {
  if (op == nullptr || selected.id.value.empty() || selected.sourceOp != op->getName()) {
    return failure();
  }

  MLIRContext *context = op->getContext();
  Builder builder(context);
  llvm::StringRef kind = stringifyImplementationKind(selected.kind);
  if (kind.empty()) {
    return failure();
  }

  NamedAttrList attributes;
  attributes.append(kImplementationKindKey, builder.getStringAttr(kind));
  attributes.append(kImplementationIdKey, builder.getStringAttr(selected.id.value));
  op->setDiscardableAttr(kImplementationAttrName, builder.getDictionaryAttr(attributes));
  return success();
}

void clearSelectedImplementation(Operation *op) {
  if (op == nullptr) {
    return;
  }
  op->removeDiscardableAttr(kImplementationAttrName);
}

LogicalResult validateSelectedImplementations(Operation *root, const ImplementationRegistry &registry,
                                              llvm::function_ref<bool(Operation *)> requiresSelection) {
  if (root == nullptr || !registry.isFrozen()) {
    return failure();
  }

  WalkResult result = root->walk([&](Operation *op) -> WalkResult {
    if (!requiresSelection(op)) {
      return WalkResult::advance();
    }
    if (failed(validateSelectedImplementation(op, registry))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

void populateEnabledPatterns(RewritePatternSet &patterns, const ImplementationRegistry &registry,
                             const PatternBuildContext &context) {
  if (!registry.isFrozen()) {
    return;
  }
  for (const ImplementationRegistry::RegistrationEntry &entry : registry.entries_) {
    entry.addPattern(patterns, context);
  }
}

std::unique_ptr<Pass> createSelectImplementationPass(std::shared_ptr<const ImplementationRegistry> registry,
                                                     std::shared_ptr<const TargetProfile> target,
                                                     std::shared_ptr<const SelectionPolicy> policy) {
  return std::make_unique<SelectImplementationPass>(std::move(registry), std::move(target), nullptr, std::move(policy));
}

std::unique_ptr<Pass> createSelectImplementationPass(std::shared_ptr<const ImplementationRegistry> registry,
                                                     std::shared_ptr<const TargetProfile> target,
                                                     std::shared_ptr<const SelectionPolicyRegistry> policies,
                                                     std::shared_ptr<const SelectionPolicy> fallbackPolicy) {
  return std::make_unique<SelectImplementationPass>(std::move(registry), std::move(target), std::move(policies),
                                                    std::move(fallbackPolicy));
}

}  // namespace mlir::ascend::instruction
