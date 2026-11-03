/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dialect/AFIR/IR/AFIR.h"
#include "target/Ascend/Capability/CapabilityQuery.h"
#include "target/Ascend/Capability/ImplementationRegistry.h"

#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "gtest/gtest.h"

namespace mlir::ascend::instruction {
namespace {

struct MockAddOp {
  static llvm::StringRef getOperationName() {
    return "afir.add";
  }
};

struct MockMapping {
  bool supported;
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

  CapabilitySupport queryLowering(Operation *, const QueryContext &context) const override {
    if (context.target.getFeatureState(TargetFeature::NDDMA) == FeatureState::Unknown) {
      return CapabilitySupport::undetermined(CapabilityReason::MissingInformation);
    }
    if (!mapping_->supported) {
      return CapabilitySupport::unsupported(CapabilityReason::UnsupportedTarget);
    }
    return CapabilitySupport::supported();
  }

 private:
  std::shared_ptr<const MockMapping> mapping_;
};

TargetProfile CreateTargetProfile(FeatureState featureState = FeatureState::Available) {
  return TargetProfile({{"AscendSynthetic", "AscendTest", NpuArch::Unknown},
                        {8, 4, 4, CubeVectorTopology::Split},
                        {},
                        {},
                        {{TargetFeature::NDDMA, featureState}},
                        {},
                        {}},
                       "fingerprint");
}

OwningOperation CreateOperation(MLIRContext &context, llvm::StringRef name = "afir.add") {
  context.allowUnregisteredDialects();
  OperationState state(UnknownLoc::get(&context), name);
  return OwningOperation(Operation::create(state));
}

ImplementationRegistry::PatternRegistration<MockMapping> CreatePatternRegistration() {
  return [](RewritePatternSet &, const PatternBuildContext &, const ImplementationKey &,
            std::shared_ptr<const MockMapping>) {};
}

LogicalResult Register(ImplementationRegistry &registry, llvm::StringRef id, MockMapping mapping = {true}) {
  return registry.registerImplementation<MockAddOp, MockMapping, MockProvider>(
      ImplementationKind::AscendC, {id.str()}, std::move(mapping), CreatePatternRegistration());
}

TEST(ImplementationRegistryTest, EmptyFrozenRegistryReturnsEmptyCandidates) {
  MLIRContext context;
  ImplementationRegistry registry(context);
  ASSERT_TRUE(succeeded(registry.freeze()));
  CapabilityQuery query(registry);
  TargetProfile target = CreateTargetProfile();
  OwningOperation op = CreateOperation(context);

  auto candidates = query.queryLowering(op.get(), QueryContext{target});

  ASSERT_TRUE(succeeded(candidates));
  EXPECT_TRUE(candidates->getAll().empty());
}

TEST(ImplementationRegistryTest, QueryReturnsSupportedMockCandidate) {
  MLIRContext context;
  ImplementationRegistry registry(context);
  MockMapping mapping{true};
  ASSERT_TRUE(succeeded(Register(registry, "add.tensor", mapping)));
  mapping.supported = false;
  ASSERT_TRUE(succeeded(registry.freeze()));
  CapabilityQuery query(registry);
  TargetProfile target = CreateTargetProfile();
  OwningOperation op = CreateOperation(context);

  auto candidates = query.queryLowering(op.get(), QueryContext{target});

  ASSERT_TRUE(succeeded(candidates));
  ASSERT_EQ(candidates->getAll().size(), 1U);
  EXPECT_EQ(candidates->getAll().front().key.id.value, "add.tensor");
  EXPECT_EQ(candidates->getAll().front().support.getState(), SupportState::Supported);
  EXPECT_TRUE(registry.contains(candidates->getAll().front().key));
}

TEST(ImplementationRegistryTest, QuerySortsCandidatesIndependentOfRegistrationOrder) {
  MLIRContext context;
  ImplementationRegistry registry(context);
  ASSERT_TRUE(succeeded(Register(registry, "z.tensor")));
  ASSERT_TRUE(succeeded(Register(registry, "a.scalar")));
  ASSERT_TRUE(succeeded(registry.freeze()));
  CapabilityQuery query(registry);
  TargetProfile target = CreateTargetProfile();
  OwningOperation op = CreateOperation(context);

  auto candidates = query.queryLowering(op.get(), QueryContext{target});

  ASSERT_TRUE(succeeded(candidates));
  ASSERT_EQ(candidates->getAll().size(), 2U);
  EXPECT_EQ(candidates->getAll()[0].key.id.value, "a.scalar");
  EXPECT_EQ(candidates->getAll()[1].key.id.value, "z.tensor");
}

TEST(ImplementationRegistryTest, FreezeRejectsDuplicateKey) {
  MLIRContext context;
  ImplementationRegistry registry(context);
  ASSERT_TRUE(succeeded(Register(registry, "duplicate")));
  ASSERT_TRUE(succeeded(Register(registry, "duplicate")));

  EXPECT_TRUE(failed(registry.freeze()));
  EXPECT_FALSE(registry.isFrozen());
}

TEST(ImplementationRegistryTest, RegistrationRejectsIncompleteEntryAndMutationAfterFreeze) {
  MLIRContext context;
  ImplementationRegistry registry(context);
  ImplementationRegistry::PatternRegistration<MockMapping> emptyPattern;

  EXPECT_TRUE(failed(registry.registerImplementation<MockAddOp, MockMapping, MockProvider>(
      ImplementationKind::AscendC, {""}, MockMapping{true}, CreatePatternRegistration())));
  EXPECT_TRUE(failed(registry.registerImplementation<MockAddOp, MockMapping, MockProvider>(
      ImplementationKind::AscendC, {"missing.pattern"}, MockMapping{true}, std::move(emptyPattern))));
  ASSERT_TRUE(succeeded(Register(registry, "valid")));
  ASSERT_TRUE(succeeded(registry.freeze()));
  EXPECT_TRUE(failed(Register(registry, "late")));
}

TEST(ImplementationRegistryTest, QueryRejectsInvalidRegistryOperationAndContext) {
  MLIRContext registryContext;
  MLIRContext otherContext;
  ImplementationRegistry registry(registryContext);
  ASSERT_TRUE(succeeded(Register(registry, "valid")));
  CapabilityQuery unfrozenQuery(registry);
  TargetProfile target = CreateTargetProfile();
  OwningOperation registryOp = CreateOperation(registryContext);
  OwningOperation otherOp = CreateOperation(otherContext);

  EXPECT_TRUE(failed(unfrozenQuery.queryLowering(registryOp.get(), QueryContext{target})));
  EXPECT_TRUE(failed(unfrozenQuery.queryLowering(nullptr, QueryContext{target})));
  ASSERT_TRUE(succeeded(registry.freeze()));
  CapabilityQuery frozenQuery(registry);
  EXPECT_TRUE(failed(frozenQuery.queryLowering(otherOp.get(), QueryContext{target})));
}

TEST(ImplementationRegistryTest, KeysMatchAcrossMLIRContexts) {
  MLIRContext registryContext;
  MLIRContext otherContext;
  ImplementationRegistry registry(registryContext);
  ASSERT_TRUE(succeeded(Register(registry, "add.tensor")));
  ASSERT_TRUE(succeeded(registry.freeze()));

  CapabilityQuery query(registry);
  TargetProfile target = CreateTargetProfile();
  OwningOperation op = CreateOperation(registryContext);
  auto candidates = query.queryLowering(op.get(), QueryContext{target});
  ASSERT_TRUE(succeeded(candidates));
  ASSERT_EQ(candidates->getAll().size(), 1U);

  ImplementationKey keyFromOtherContext{
      OperationName("afir.add", &otherContext), ImplementationKind::AscendC, {"add.tensor"}};
  EXPECT_TRUE(registry.contains(keyFromOtherContext));
  EXPECT_TRUE(candidates->getAll().front().key == keyFromOtherContext);
}

TEST(ImplementationRegistryTest, KeysRemainStableAcrossDialectRegistration) {
  MLIRContext context;
  context.allowUnregisteredDialects();
  OperationName beforeDialectRegistration("afir.add", &context);
  EXPECT_EQ(beforeDialectRegistration.getStringRef(), "afir.add");
  ImplementationRegistry registry(context);
  ASSERT_TRUE(succeeded(Register(registry, "add.tensor")));
  ASSERT_TRUE(succeeded(registry.freeze()));

  context.getOrLoadDialect<mlir::afir::AFIRDialect>();
  OperationName afterDialectRegistration("afir.add", &context);
  ImplementationKey afterKey{afterDialectRegistration, ImplementationKind::AscendC, {"add.tensor"}};

  EXPECT_TRUE(registry.contains(afterKey));
}

TEST(ImplementationRegistryTest, ProviderCanReturnUndeterminedFromTargetFacts) {
  MLIRContext context;
  ImplementationRegistry registry(context);
  ASSERT_TRUE(succeeded(Register(registry, "target.sensitive")));
  ASSERT_TRUE(succeeded(registry.freeze()));
  CapabilityQuery query(registry);
  TargetProfile target = CreateTargetProfile(FeatureState::Unknown);
  OwningOperation op = CreateOperation(context);

  auto candidates = query.queryLowering(op.get(), QueryContext{target});

  ASSERT_TRUE(succeeded(candidates));
  ASSERT_EQ(candidates->getAll().size(), 1U);
  EXPECT_EQ(candidates->getAll().front().support.getState(), SupportState::Undetermined);
  EXPECT_EQ(candidates->getAll().front().support.getReason(), CapabilityReason::MissingInformation);
}

}  // namespace
}  // namespace mlir::ascend::instruction
