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

#include "target/Ascend/Capability/CapabilityProvider.h"
#include "target/Ascend/Capability/ImplementationRegistry.h"

#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "gtest/gtest.h"

namespace mlir::ascend::instruction {
namespace {

struct MockAddOp {
  static llvm::StringRef getOperationName() {
    return "afir.add";
  }
};

struct MockMapping {
  bool supported = true;
  std::string replacementName = "mock.lowered";
};

struct OperationDeleter {
  void operator()(Operation *op) const {
    op->destroy();
  }
};

using OwningOperation = std::unique_ptr<Operation, OperationDeleter>;

class MockProvider final : public CapabilityProvider {
 public:
  explicit MockProvider(std::shared_ptr<const MockMapping> mapping) : mapping_(std::move(mapping)) {}

  CapabilitySupport queryLowering(Operation *, const QueryContext &) const override {
    if (!mapping_->supported) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedTarget);
    }
    return CapabilitySupport::supported();
  }

 private:
  std::shared_ptr<const MockMapping> mapping_;
};

class SelectedImplementationRewritePattern final : public RewritePattern {
 public:
  SelectedImplementationRewritePattern(MLIRContext *context, ImplementationKey key, llvm::StringRef replacementName)
      : RewritePattern("afir.add", PatternBenefit(1), context),
        key_(std::move(key)),
        replacementName_(replacementName.str()) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    FailureOr<ImplementationKey> selected = getSelectedImplementation(op);
    if (failed(selected) || *selected != key_) {
      return rewriter.notifyMatchFailure(op, "another Ascend implementation was selected");
    }

    OperationState state(op->getLoc(), replacementName_);
    Operation *replacement = rewriter.create(state);
    rewriter.replaceOp(op, replacement->getResults());
    return success();
  }

 private:
  ImplementationKey key_;
  std::string replacementName_;
};

class FirstCandidatePolicy final : public SelectionPolicy {
 public:
  FailureOr<ImplementationKey> choose(Operation *, llvm::ArrayRef<Candidate> supported,
                                      const QueryContext &) const override {
    return supported.front().key;
  }
};

class InvalidCandidatePolicy final : public SelectionPolicy {
 public:
  explicit InvalidCandidatePolicy(ImplementationKey key) : key_(std::move(key)) {}

  FailureOr<ImplementationKey> choose(Operation *, llvm::ArrayRef<Candidate>, const QueryContext &) const override {
    return key_;
  }

 private:
  ImplementationKey key_;
};

TargetProfile CreateTargetProfile() {
  return TargetProfile(
      {{"AscendSynthetic", "AscendTest", NpuArch::Unknown}, {8, 4, 4, CubeVectorTopology::Split}, {}, {}, {}, {}, {}},
      "fingerprint");
}

ImplementationKey CreateKey(MLIRContext &context, llvm::StringRef id,
                            ImplementationKind kind = ImplementationKind::AscendC,
                            llvm::StringRef operationName = "afir.add") {
  return {OperationName(operationName, &context), kind, {id.str()}};
}

OwningOperation CreateOperation(MLIRContext &context, llvm::StringRef name = "afir.add") {
  context.allowUnregisteredDialects();
  OperationState state(UnknownLoc::get(&context), name);
  return OwningOperation(Operation::create(state));
}

ImplementationRegistry::PatternRegistration<MockMapping> CreatePatternRegistration() {
  return [](RewritePatternSet &patterns, const PatternBuildContext &, const ImplementationKey &key,
            std::shared_ptr<const MockMapping> mapping) {
    patterns.add<SelectedImplementationRewritePattern>(patterns.getContext(), key, mapping->replacementName);
  };
}

LogicalResult Register(ImplementationRegistry &registry, llvm::StringRef id, MockMapping mapping = {}) {
  return registry.registerImplementation<MockAddOp, MockMapping, MockProvider>(
      ImplementationKind::AscendC, {id.str()}, std::move(mapping), CreatePatternRegistration());
}

Candidates CreateCandidates(MLIRContext &context, llvm::ArrayRef<llvm::StringRef> supportedIds) {
  llvm::SmallVector<Candidate, 2> values;
  for (llvm::StringRef id : supportedIds) {
    values.push_back({CreateKey(context, id), CapabilitySupport::supported()});
  }
  values.push_back(
      {CreateKey(context, "unsupported"), CapabilitySupport::unsupported(CapabilityReason::UnsupportedType)});
  return Candidates(std::move(values));
}

OwningOpRef<ModuleOp> CreateModuleWithOps(MLIRContext &context, llvm::ArrayRef<llvm::StringRef> operationNames) {
  context.allowUnregisteredDialects();
  OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
  OpBuilder builder(&context);
  builder.setInsertionPointToStart((*module).getBody());
  for (llvm::StringRef name : operationNames) {
    OperationState state(UnknownLoc::get(&context), name);
    builder.create(state);
  }
  return module;
}

Operation *FindNestedOp(Operation *root, llvm::StringRef operationName) {
  Operation *found = nullptr;
  root->walk([&](Operation *op) {
    if (op->getName().getStringRef() == operationName) {
      found = op;
    }
  });
  return found;
}

bool IsMockSourceOp(Operation *op) {
  return op->getName().getStringRef() == "afir.add";
}

TEST(ImplementationSelectionTest, SelectsSingleSupportedCandidateWithoutPolicy) {
  MLIRContext context;
  TargetProfile target = CreateTargetProfile();
  OwningOperation op = CreateOperation(context);

  auto selected = selectImplementation(op.get(), CreateCandidates(context, {"add.tensor"}), QueryContext{target},
                                       /*policy=*/nullptr);

  ASSERT_TRUE(succeeded(selected));
  EXPECT_EQ(selected->id.value, "add.tensor");
}

TEST(ImplementationSelectionTest, RejectsNoSupportedCandidateAndMultipleWithoutPolicy) {
  MLIRContext context;
  ScopedDiagnosticHandler diagnosticHandler(&context, [](Diagnostic &) { return success(); });
  TargetProfile target = CreateTargetProfile();
  OwningOperation op = CreateOperation(context);
  llvm::SmallVector<Candidate, 2> unsupportedValues;
  unsupportedValues.push_back(
      {CreateKey(context, "unsupported"), CapabilitySupport::unsupported(CapabilityReason::UnsupportedTarget)});

  EXPECT_TRUE(failed(selectImplementation(op.get(), Candidates(std::move(unsupportedValues)), QueryContext{target},
                                          /*policy=*/nullptr)));
  EXPECT_TRUE(failed(selectImplementation(op.get(), CreateCandidates(context, {"add.tensor", "add.scalar"}),
                                          QueryContext{target}, /*policy=*/nullptr)));
}

TEST(ImplementationSelectionTest, UsesPolicyOnlyForMultipleSupportedCandidates) {
  MLIRContext context;
  TargetProfile target = CreateTargetProfile();
  OwningOperation op = CreateOperation(context);
  FirstCandidatePolicy policy;

  auto selected = selectImplementation(op.get(), CreateCandidates(context, {"z.tensor", "a.scalar"}),
                                       QueryContext{target}, &policy);

  ASSERT_TRUE(succeeded(selected));
  EXPECT_EQ(selected->id.value, "a.scalar");
}

TEST(ImplementationSelectionTest, RejectsPolicyChoiceOutsideSupportedCandidates) {
  MLIRContext context;
  ScopedDiagnosticHandler diagnosticHandler(&context, [](Diagnostic &) { return success(); });
  TargetProfile target = CreateTargetProfile();
  OwningOperation op = CreateOperation(context);
  InvalidCandidatePolicy policy(CreateKey(context, "not.supported"));

  EXPECT_TRUE(failed(selectImplementation(op.get(), CreateCandidates(context, {"add.tensor", "add.scalar"}),
                                          QueryContext{target}, &policy)));
}

TEST(ImplementationSelectionTest, ReadsWritesAndClearsSelectedImplementationAttribute) {
  MLIRContext context;
  OwningOperation op = CreateOperation(context);
  ImplementationKey key = CreateKey(context, "add.tensor_scalar");

  ASSERT_TRUE(succeeded(setSelectedImplementation(op.get(), key)));
  auto selected = getSelectedImplementation(op.get());

  ASSERT_TRUE(succeeded(selected));
  EXPECT_EQ(*selected, key);
  auto attribute = dyn_cast<DictionaryAttr>(op->getDiscardableAttr("ascend.implementation"));
  ASSERT_TRUE(attribute);
  EXPECT_EQ(cast<StringAttr>(attribute.get("kind")).getValue(), "ascendc_api");
  EXPECT_EQ(cast<StringAttr>(attribute.get("id")).getValue(), "add.tensor_scalar");

  clearSelectedImplementation(op.get());
  EXPECT_FALSE(op->getDiscardableAttr("ascend.implementation"));
}

TEST(ImplementationSelectionTest, RejectsMalformedSelectedImplementationAttribute) {
  MLIRContext context;
  ScopedDiagnosticHandler diagnosticHandler(&context, [](Diagnostic &) { return success(); });
  Builder builder(&context);
  OwningOperation op = CreateOperation(context);

  EXPECT_TRUE(failed(getSelectedImplementation(op.get())));

  op->setDiscardableAttr("ascend.implementation", builder.getDictionaryAttr({}));
  EXPECT_TRUE(failed(getSelectedImplementation(op.get())));

  NamedAttrList attributes;
  attributes.append("kind", builder.getStringAttr("unknown_kind"));
  attributes.append("id", builder.getStringAttr("add.tensor"));
  op->setDiscardableAttr("ascend.implementation", builder.getDictionaryAttr(attributes));
  EXPECT_TRUE(failed(getSelectedImplementation(op.get())));
}

TEST(ImplementationSelectionTest, ValidatesSelectedImplementationAgainstFrozenRegistry) {
  MLIRContext context;
  ScopedDiagnosticHandler diagnosticHandler(&context, [](Diagnostic &) { return success(); });
  ImplementationRegistry registry(context);
  ASSERT_TRUE(succeeded(Register(registry, "add.tensor")));
  ASSERT_TRUE(succeeded(registry.freeze()));
  OwningOperation op = CreateOperation(context);

  ASSERT_TRUE(succeeded(setSelectedImplementation(op.get(), CreateKey(context, "add.tensor"))));
  EXPECT_TRUE(succeeded(validateSelectedImplementations(op.get(), registry, IsMockSourceOp)));

  ASSERT_TRUE(succeeded(setSelectedImplementation(op.get(), CreateKey(context, "missing"))));
  EXPECT_TRUE(failed(validateSelectedImplementations(op.get(), registry, IsMockSourceOp)));
}

TEST(ImplementationSelectionTest, RejectsSourceMismatchWhenSettingSelectedImplementation) {
  MLIRContext context;
  OwningOperation op = CreateOperation(context, "afir.add");

  EXPECT_TRUE(failed(
      setSelectedImplementation(op.get(), CreateKey(context, "add.tensor", ImplementationKind::AscendC, "afir.mul"))));
}

TEST(ImplementationSelectionTest, SelectImplementationPassWritesSelectionForRegisteredOperationsOnly) {
  MLIRContext context;
  auto registry = std::make_shared<ImplementationRegistry>(context);
  ASSERT_TRUE(succeeded(Register(*registry, "add.tensor")));
  ASSERT_TRUE(succeeded(registry->freeze()));
  auto target = std::make_shared<TargetProfile>(CreateTargetProfile());
  OwningOpRef<ModuleOp> module = CreateModuleWithOps(context, {"afir.add", "mock.ignored"});

  PassManager passManager(&context);
  passManager.addPass(createSelectImplementationPass(registry, target));

  ASSERT_TRUE(succeeded(passManager.run(*module)));
  Operation *source = FindNestedOp(module.get(), "afir.add");
  ASSERT_NE(source, nullptr);
  auto selected = getSelectedImplementation(source);
  ASSERT_TRUE(succeeded(selected));
  EXPECT_EQ(selected->id.value, "add.tensor");
  EXPECT_EQ(FindNestedOp(module.get(), "mock.ignored")->getDiscardableAttr("ascend.implementation"), nullptr);
}

TEST(ImplementationSelectionTest, SelectImplementationPassFailsWhenMultipleCandidatesNeedPolicy) {
  MLIRContext context;
  ScopedDiagnosticHandler diagnosticHandler(&context, [](Diagnostic &) { return success(); });
  auto registry = std::make_shared<ImplementationRegistry>(context);
  ASSERT_TRUE(succeeded(Register(*registry, "add.tensor")));
  ASSERT_TRUE(succeeded(Register(*registry, "add.scalar")));
  ASSERT_TRUE(succeeded(registry->freeze()));
  auto target = std::make_shared<TargetProfile>(CreateTargetProfile());
  OwningOpRef<ModuleOp> module = CreateModuleWithOps(context, {"afir.add"});

  PassManager passManager(&context);
  passManager.addPass(createSelectImplementationPass(registry, target));

  EXPECT_TRUE(failed(passManager.run(*module)));
}

TEST(ImplementationSelectionTest, SelectImplementationPassUsesPolicyRegisteredForSourceOperation) {
  MLIRContext context;
  auto registry = std::make_shared<ImplementationRegistry>(context);
  ASSERT_TRUE(succeeded(Register(*registry, "add.tensor")));
  ASSERT_TRUE(succeeded(Register(*registry, "add.scalar")));
  auto policies = std::make_shared<SelectionPolicyRegistry>(context);
  ASSERT_TRUE(succeeded(policies->registerPolicy<MockAddOp>(std::make_shared<FirstCandidatePolicy>())));
  ASSERT_TRUE(succeeded(policies->freeze()));
  ASSERT_TRUE(succeeded(registry->freeze()));
  auto target = std::make_shared<TargetProfile>(CreateTargetProfile());
  OwningOpRef<ModuleOp> module = CreateModuleWithOps(context, {"afir.add"});

  PassManager passManager(&context);
  passManager.addPass(createSelectImplementationPass(registry, target, policies, nullptr));

  ASSERT_TRUE(succeeded(passManager.run(*module)));
  Operation *source = FindNestedOp(module.get(), "afir.add");
  ASSERT_NE(source, nullptr);
  auto selected = getSelectedImplementation(source);
  ASSERT_TRUE(succeeded(selected));
  EXPECT_EQ(selected->id.value, "add.scalar");
}

TEST(ImplementationSelectionTest, SelectImplementationPassDoesNotReusePreviousSelectionWithoutPolicy) {
  MLIRContext context;
  ScopedDiagnosticHandler diagnosticHandler(&context, [](Diagnostic &) { return success(); });
  auto registry = std::make_shared<ImplementationRegistry>(context);
  ASSERT_TRUE(succeeded(Register(*registry, "add.tensor")));
  ASSERT_TRUE(succeeded(Register(*registry, "add.scalar")));
  ASSERT_TRUE(succeeded(registry->freeze()));
  auto target = std::make_shared<TargetProfile>(CreateTargetProfile());
  OwningOpRef<ModuleOp> module = CreateModuleWithOps(context, {"afir.add"});
  Operation *source = FindNestedOp(module.get(), "afir.add");
  ASSERT_NE(source, nullptr);
  ASSERT_TRUE(succeeded(setSelectedImplementation(source, CreateKey(context, "add.tensor"))));

  PassManager passManager(&context);
  passManager.addPass(createSelectImplementationPass(registry, target));

  EXPECT_TRUE(failed(passManager.run(*module)));
}

TEST(ImplementationSelectionTest, SelectImplementationPassRejectsMalformedPreviousSelection) {
  MLIRContext context;
  ScopedDiagnosticHandler diagnosticHandler(&context, [](Diagnostic &) { return success(); });
  auto registry = std::make_shared<ImplementationRegistry>(context);
  ASSERT_TRUE(succeeded(Register(*registry, "add.tensor")));
  ASSERT_TRUE(succeeded(registry->freeze()));
  auto target = std::make_shared<TargetProfile>(CreateTargetProfile());
  OwningOpRef<ModuleOp> module = CreateModuleWithOps(context, {"afir.add"});
  Operation *source = FindNestedOp(module.get(), "afir.add");
  ASSERT_NE(source, nullptr);
  source->setDiscardableAttr("ascend.implementation", Builder(&context).getDictionaryAttr({}));

  PassManager passManager(&context);
  passManager.addPass(createSelectImplementationPass(registry, target));

  EXPECT_TRUE(failed(passManager.run(*module)));
}

TEST(ImplementationSelectionTest, SelectImplementationPassRejectsUnknownPreviousSelection) {
  MLIRContext context;
  ScopedDiagnosticHandler diagnosticHandler(&context, [](Diagnostic &) { return success(); });
  auto registry = std::make_shared<ImplementationRegistry>(context);
  ASSERT_TRUE(succeeded(Register(*registry, "add.tensor")));
  ASSERT_TRUE(succeeded(registry->freeze()));
  auto target = std::make_shared<TargetProfile>(CreateTargetProfile());
  OwningOpRef<ModuleOp> module = CreateModuleWithOps(context, {"afir.add"});
  Operation *source = FindNestedOp(module.get(), "afir.add");
  ASSERT_NE(source, nullptr);
  ASSERT_TRUE(succeeded(setSelectedImplementation(source, CreateKey(context, "removed.implementation"))));

  PassManager passManager(&context);
  passManager.addPass(createSelectImplementationPass(registry, target));

  EXPECT_TRUE(failed(passManager.run(*module)));
}

TEST(ImplementationSelectionTest, SelectImplementationPassRejectsUnsupportedPreviousSelection) {
  MLIRContext context;
  ScopedDiagnosticHandler diagnosticHandler(&context, [](Diagnostic &) { return success(); });
  auto registry = std::make_shared<ImplementationRegistry>(context);
  ASSERT_TRUE(succeeded(Register(*registry, "add.tensor")));
  ASSERT_TRUE(succeeded(Register(*registry, "add.scalar", MockMapping{false, "mock.lowered.scalar"})));
  ASSERT_TRUE(succeeded(registry->freeze()));
  auto target = std::make_shared<TargetProfile>(CreateTargetProfile());
  OwningOpRef<ModuleOp> module = CreateModuleWithOps(context, {"afir.add"});
  Operation *source = FindNestedOp(module.get(), "afir.add");
  ASSERT_NE(source, nullptr);
  ASSERT_TRUE(succeeded(setSelectedImplementation(source, CreateKey(context, "add.scalar"))));

  PassManager passManager(&context);
  passManager.addPass(createSelectImplementationPass(registry, target));

  EXPECT_TRUE(failed(passManager.run(*module)));
}

TEST(ImplementationSelectionTest, SelectImplementationPassRejectsSelectionOnUnregisteredOperation) {
  MLIRContext context;
  ScopedDiagnosticHandler diagnosticHandler(&context, [](Diagnostic &) { return success(); });
  auto registry = std::make_shared<ImplementationRegistry>(context);
  ASSERT_TRUE(succeeded(Register(*registry, "add.tensor")));
  ASSERT_TRUE(succeeded(registry->freeze()));
  auto target = std::make_shared<TargetProfile>(CreateTargetProfile());
  OwningOpRef<ModuleOp> module = CreateModuleWithOps(context, {"mock.ignored"});
  Operation *source = FindNestedOp(module.get(), "mock.ignored");
  ASSERT_NE(source, nullptr);
  ASSERT_TRUE(succeeded(setSelectedImplementation(
      source, CreateKey(context, "removed.implementation", ImplementationKind::AscendC, "mock.ignored"))));

  PassManager passManager(&context);
  passManager.addPass(createSelectImplementationPass(registry, target));

  EXPECT_TRUE(failed(passManager.run(*module)));
}

TEST(ImplementationSelectionTest, PreflightRequiresSelectionForLoweringTargets) {
  MLIRContext context;
  ScopedDiagnosticHandler diagnosticHandler(&context, [](Diagnostic &) { return success(); });
  ImplementationRegistry registry(context);
  ASSERT_TRUE(succeeded(Register(registry, "add.tensor")));
  ASSERT_TRUE(succeeded(registry.freeze()));
  OwningOpRef<ModuleOp> module = CreateModuleWithOps(context, {"afir.add", "mock.ignored"});
  Operation *source = FindNestedOp(module.get(), "afir.add");
  ASSERT_NE(source, nullptr);

  EXPECT_TRUE(failed(validateSelectedImplementations(module.get(), registry, IsMockSourceOp)));
  ASSERT_TRUE(succeeded(setSelectedImplementation(source, CreateKey(context, "add.tensor"))));
  EXPECT_TRUE(succeeded(validateSelectedImplementations(module.get(), registry, IsMockSourceOp)));
}

TEST(ImplementationSelectionTest, PopulateEnabledPatternsLowersOnlySelectedImplementation) {
  MLIRContext context;
  ImplementationRegistry registry(context);
  ASSERT_TRUE(succeeded(Register(registry, "z.tensor", MockMapping{true, "mock.lowered.tensor"})));
  ASSERT_TRUE(succeeded(Register(registry, "a.scalar", MockMapping{true, "mock.lowered.scalar"})));
  ASSERT_TRUE(succeeded(registry.freeze()));
  TargetProfile target = CreateTargetProfile();
  TypeConverter typeConverter;
  PatternBuildContext patternContext{typeConverter, target};
  RewritePatternSet patterns(&context);
  populateEnabledPatterns(patterns, registry, patternContext);
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  OwningOpRef<ModuleOp> module = CreateModuleWithOps(context, {"afir.add"});
  Operation *source = FindNestedOp(module.get(), "afir.add");
  ASSERT_NE(source, nullptr);
  ASSERT_TRUE(succeeded(setSelectedImplementation(source, CreateKey(context, "z.tensor"))));
  ASSERT_TRUE(succeeded(validateSelectedImplementations(module.get(), registry, IsMockSourceOp)));

  ConversionTarget conversionTarget(context);
  conversionTarget.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  conversionTarget.addIllegalOp(OperationName("afir.add", &context));
  conversionTarget.addLegalOp(OperationName("mock.lowered.tensor", &context));
  conversionTarget.addLegalOp(OperationName("mock.lowered.scalar", &context));

  ASSERT_TRUE(succeeded(applyFullConversion(module.get(), conversionTarget, frozenPatterns)));
  EXPECT_EQ(FindNestedOp(module.get(), "afir.add"), nullptr);
  EXPECT_NE(FindNestedOp(module.get(), "mock.lowered.tensor"), nullptr);
  EXPECT_EQ(FindNestedOp(module.get(), "mock.lowered.scalar"), nullptr);
}

}  // namespace
}  // namespace mlir::ascend::instruction
