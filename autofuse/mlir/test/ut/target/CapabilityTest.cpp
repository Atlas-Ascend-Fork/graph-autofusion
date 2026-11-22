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
#include "target/Ascend/Capability/CapabilityProvider.h"

#include <string>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "gtest/gtest.h"

namespace mlir::ascend::instruction {
namespace {

TargetProfile CreateTargetProfile() {
  return TargetProfile(
      {{"AscendSynthetic", "AscendTest", NpuArch::Unknown}, {8, 4, 4, CubeVectorTopology::Split}, {}, {}, {}, {}, {}},
      "fingerprint");
}

ImplementationKey CreateKey(MLIRContext &context, ImplementationKind kind, llvm::StringRef id) {
  return {OperationName("afir.add", &context), kind, {id.str()}};
}

class OwningResultProvider final : public CapabilityProvider {
 public:
  CapabilitySupport queryLowering(Operation *, const QueryContext &) const override {
    llvm::SmallVector<CapabilityConstraint, 4> constraints;
    constraints.emplace_back(RankConstraint{{ValuePositionKind::Operand, 0}, 1, 8});
    return CapabilitySupport::supported(std::move(constraints));
  }
};

TEST(CapabilityTest, SupportResultOwnsConstraintStorage) {
  llvm::SmallVector<uint32_t, 2> operandIndices = {0, 1};
  llvm::SmallVector<CapabilityConstraint, 4> constraints;
  constraints.emplace_back(ScalarOperandConstraint{operandIndices, false});

  CapabilitySupport support = CapabilitySupport::supported(constraints);
  operandIndices.clear();
  constraints.clear();

  ASSERT_EQ(support.getConstraints().size(), 1U);
  const auto &scalar = std::get<ScalarOperandConstraint>(support.getConstraints().front());
  EXPECT_EQ(scalar.allowedOperandIndices, (llvm::SmallVector<uint32_t, 2>{0, 1}));
}

TEST(CapabilityTest, ProviderResultDoesNotBorrowTemporaryContainers) {
  TargetProfile target = CreateTargetProfile();
  QueryContext context{target};
  OwningResultProvider provider;

  CapabilitySupport support = provider.queryLowering(nullptr, context);

  ASSERT_EQ(support.getState(), SupportState::Supported);
  ASSERT_EQ(support.getConstraints().size(), 1U);
  const auto &rank = std::get<RankConstraint>(support.getConstraints().front());
  EXPECT_EQ(rank.value.kind, ValuePositionKind::Operand);
  EXPECT_EQ(rank.value.index, 0U);
  EXPECT_EQ(rank.minimumRank, 1);
  EXPECT_EQ(rank.maximumRank, 8);
}

TEST(CapabilityTest, CandidatesSortByKindAndStableId) {
  MLIRContext context;
  llvm::SmallVector<Candidate, 2> values;
  values.push_back({CreateKey(context, ImplementationKind::AscendC, "z.tensor"), CapabilitySupport::supported()});
  values.push_back({CreateKey(context, ImplementationKind::AscendC, "a.scalar"), CapabilitySupport::supported()});
  values.push_back(
      {CreateKey(context, ImplementationKind::CCEIntrinsic, "a.intrinsic"), CapabilitySupport::supported()});

  Candidates candidates(std::move(values));

  ASSERT_EQ(candidates.getAll().size(), 3U);
  EXPECT_EQ(candidates.getAll()[0].key.id.value, "a.scalar");
  EXPECT_EQ(candidates.getAll()[1].key.id.value, "z.tensor");
  EXPECT_EQ(candidates.getAll()[2].key.id.value, "a.intrinsic");
}

TEST(CapabilityTest, GetSupportedReturnsOwningCopy) {
  MLIRContext context;
  llvm::SmallVector<Candidate, 2> values;
  values.push_back({CreateKey(context, ImplementationKind::AscendC, "supported"), CapabilitySupport::supported()});
  values.push_back({CreateKey(context, ImplementationKind::AscendC, "unsupported"),
                    CapabilitySupport::unsupported(CapabilityReason::UnsupportedType)});
  Candidates candidates(std::move(values));

  llvm::SmallVector<Candidate, 2> supported = candidates.getSupported();
  ASSERT_EQ(supported.size(), 1U);
  supported.front().key.id.value.assign("modified");

  ASSERT_EQ(candidates.getAll().size(), 2U);
  EXPECT_EQ(candidates.getAll().front().key.id.value, "supported");
}

TEST(CapabilityTest, DistinguishesUnsupportedAndUndetermined) {
  MLIRContext context;
  llvm::SmallVector<Candidate, 2> values;
  values.push_back({CreateKey(context, ImplementationKind::AscendC, "unsupported"),
                    CapabilitySupport::unsupported(CapabilityReason::UnsupportedTarget)});
  values.push_back({CreateKey(context, ImplementationKind::AscendC, "undetermined"),
                    CapabilitySupport::undetermined(CapabilityReason::TargetFeatureNotModeled)});
  Candidates candidates(std::move(values));

  EXPECT_TRUE(candidates.getSupported().empty());
  EXPECT_TRUE(candidates.hasUndetermined());
  EXPECT_EQ(candidates.getAll()[0].key.id.value, "undetermined");
  EXPECT_EQ(candidates.getAll()[0].support.getReason(), CapabilityReason::TargetFeatureNotModeled);
  EXPECT_EQ(candidates.getAll()[1].key.id.value, "unsupported");
  EXPECT_EQ(candidates.getAll()[1].support.getReason(), CapabilityReason::UnsupportedTarget);
}

}  // namespace
}  // namespace mlir::ascend::instruction
