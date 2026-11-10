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
#include "dialect/AFIR/Analysis/TransferPlan.h"
#include "conversion/AFIRToAscendC/Implementations/ElementwiseLocalView.h"
#include "conversion/AFIRToAscendC/RegisterAscendCImplementations.h"
#include "target/Ascend/Capability/CapabilityQuery.h"
#include "target/Ascend/Capability/ImplementationRegistry.h"
#include "target/Ascend/TargetProfileLoader.h"

#include <optional>
#include <string>

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

namespace mlir::afir {
using namespace mlir::ascend::kernel;
namespace {

struct BinaryOperationCase {
  llvm::StringLiteral operationName;
  llvm::StringLiteral stableId;
  llvm::StringLiteral scalarStableId;
};

constexpr BinaryOperationCase kBinaryOperationCases[] = {
    {"afir.add", "add.tensor_tensor", "add.tensor_scalar"},
};

ascend::TargetProfile CreateTargetProfile(int64_t vectorCoreCount = 4) {
  return ascend::TargetProfile({{"AscendSynthetic", "AscendTest", ascend::NpuArch::Unknown},
                                {8, 4, vectorCoreCount, ascend::CubeVectorTopology::Split},
                                {},
                                {},
                                {},
                                {},
                                {}},
                               "fingerprint");
}

// `CreateTargetProfile` above leaves every `TargetFeature` unmodeled (`Unknown`), which is
// fine for the `Any`-family candidates this file otherwise exercises -- `checkApiFamily`
// skips the feature check entirely for `Any`. The row-broadcast-inline candidate is
// `MemBased`, so it does ask `TargetFeature::RegBasedApiFamily`; on the synthetic profile
// that answers `Unknown` and the candidate comes back `Undetermined`, not the deterministic
// `Supported`/`Unsupported` these two tests want to compare against a fixed reason. A real
// built-in profile models the feature (`Ascend910B1` is mem-based), so tests that touch a
// `MemBased`/`RegBased`-gated candidate use this instead.
ascend::TargetProfile CreateMemBasedTargetProfile() {
  llvm::raw_null_ostream diagnostic;
  FailureOr<ascend::TargetProfile> profile = ascend::TargetProfileLoader::loadBuiltin("Ascend910B1", diagnostic);
  EXPECT_TRUE(succeeded(profile));
  return succeeded(profile) ? *profile : CreateTargetProfile();
}

void LoadDialects(MLIRContext &context) {
  context.getOrLoadDialect<afir::AFIRDialect>();
  context.getOrLoadDialect<ascendc::AscendCDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
}

OwningOpRef<ModuleOp> ParseBufferizedBinary(MLIRContext &context, llvm::StringRef operationName,
                                            llvm::StringRef lhsType, llvm::StringRef rhsType,
                                            llvm::StringRef destinationType, llvm::StringRef resultType) {
  std::string ir =
      "module {\n"
      "  func.func @test(%lhs: " +
      lhsType.str() + ", %rhs: " + rhsType.str() + ", %destination: " + destinationType.str() +
      ") {\n"
      "    %result = " +
      operationName.str() + " %lhs, %rhs outs(%destination) {" +
      "afir.operand_descs = [#afir.tensor_desc<role = <input>>, #afir.tensor_desc<role = <input>>], " +
      "afir.destination_descs = [#afir.tensor_desc<role = <compute>>], " +
      "afir.result_descs = [#afir.tensor_desc<role = <compute>>]" + "} : (" + lhsType.str() + ", " + rhsType.str() +
      "), " + destinationType.str() + " -> " + resultType.str() +
      "\n"
      "    return\n"
      "  }\n"
      "}\n";
  return parseSourceString<ModuleOp>(ir, &context);
}

Operation *FindOperation(ModuleOp module, llvm::StringRef operationName) {
  Operation *found = nullptr;
  module.walk([&](Operation *operation) {
    if (operation->getName().getStringRef() == operationName) {
      found = operation;
    }
  });
  return found;
}

// 按 stable id 查询 tensor-tensor 和 tensor-scalar 实现，分别检查其支持条件。
// 不能取候选 front()，否则注册顺序变化会使测试查询另一个不同形态的实现。
template <typename ResultT>
const auto *FindCandidate(const ResultT &result, llvm::StringRef stableId) {
  for (const auto &candidate : result.getAll()) {
    if (candidate.key.id.value == stableId) {
      return &candidate;
    }
  }
  return decltype(&result.getAll().front()){nullptr};
}

std::unique_ptr<ascend::instruction::ImplementationRegistry> CreateRegistry(MLIRContext &context) {
  auto registry = std::make_unique<ascend::instruction::ImplementationRegistry>(context);
  if (failed(registerAscendCImplementations(*registry)) || failed(registry->freeze())) {
    return nullptr;
  }
  return registry;
}

TEST(BinaryElementwiseL2CapabilityTest, SupportsDeclaredElementTypes) {
  MLIRContext context;
  LoadDialects(context);
  auto registry = CreateRegistry(context);
  ASSERT_TRUE(registry);
  ascend::instruction::CapabilityQuery query(*registry);
  ascend::TargetProfile target = CreateTargetProfile();
  constexpr llvm::StringLiteral elementTypes[] = {"f16", "f32", "i16", "i32"};

  for (const BinaryOperationCase &operationCase : kBinaryOperationCases) {
    for (llvm::StringRef elementType : elementTypes) {
      SCOPED_TRACE((operationCase.operationName + ":" + elementType).str());
      std::string tensorType = "!ascendc.local_tensor<?x" + elementType.str() + ">";
      OwningOpRef<ModuleOp> module =
          ParseBufferizedBinary(context, operationCase.operationName, tensorType, tensorType, tensorType, tensorType);
      ASSERT_TRUE(module);
      auto candidates = query.queryLowering(FindOperation(*module, operationCase.operationName),
                                            ascend::instruction::QueryContext{target});

      ASSERT_TRUE(succeeded(candidates));
      const auto *tensorTensor = FindCandidate(*candidates, operationCase.stableId);
      ASSERT_NE(tensorTensor, nullptr);
      EXPECT_EQ(tensorTensor->support.getState(), ascend::instruction::SupportState::Supported);
      // 右操作数是一块 tile，标量那一个必须拒——两条判据互斥，不靠优先级挑。
      const auto *tensorScalar = FindCandidate(*candidates, operationCase.scalarStableId);
      ASSERT_NE(tensorScalar, nullptr);
      EXPECT_EQ(tensorScalar->support.getState(), ascend::instruction::SupportState::Unsupported);
    }
  }
}

TEST(BinaryElementwiseL2CapabilityTest, RejectsUnsupportedElementType) {
  MLIRContext context;
  LoadDialects(context);
  auto registry = CreateRegistry(context);
  ASSERT_TRUE(registry);
  ascend::instruction::CapabilityQuery query(*registry);
  ascend::TargetProfile target = CreateTargetProfile();

  for (const BinaryOperationCase &operationCase : kBinaryOperationCases) {
    SCOPED_TRACE(operationCase.operationName.str());
    OwningOpRef<ModuleOp> module = ParseBufferizedBinary(
        context, operationCase.operationName, "!ascendc.local_tensor<?xf64>", "!ascendc.local_tensor<?xf64>",
        "!ascendc.local_tensor<?xf64>", "!ascendc.local_tensor<?xf64>");
    ASSERT_TRUE(module);
    auto candidates = query.queryLowering(FindOperation(*module, operationCase.operationName),
                                          ascend::instruction::QueryContext{target});

    ASSERT_TRUE(succeeded(candidates));
    const auto *candidate = FindCandidate(*candidates, operationCase.stableId);
    ASSERT_NE(candidate, nullptr);
    EXPECT_EQ(candidate->support.getReason(), ascend::instruction::CapabilityReason::UnsupportedType);
  }
}

TEST(BinaryElementwiseL2CapabilityTest, RejectsUnsupportedRankAndMismatchedShapes) {
  MLIRContext context;
  LoadDialects(context);
  auto registry = CreateRegistry(context);
  ASSERT_TRUE(registry);
  ascend::instruction::CapabilityQuery query(*registry);
  ascend::TargetProfile target = CreateTargetProfile();

  for (const BinaryOperationCase &operationCase : kBinaryOperationCases) {
    SCOPED_TRACE(operationCase.operationName.str());
    OwningOpRef<ModuleOp> rankTwo = ParseBufferizedBinary(
        context, operationCase.operationName, "!ascendc.local_tensor<2x8xf16>", "!ascendc.local_tensor<2x8xf16>",
        "!ascendc.local_tensor<2x8xf16>", "!ascendc.local_tensor<2x8xf16>");
    ASSERT_TRUE(rankTwo);
    OwningOpRef<ModuleOp> mismatchedShape = ParseBufferizedBinary(
        context, operationCase.operationName, "!ascendc.local_tensor<?xf16>", "!ascendc.local_tensor<16xf16>",
        "!ascendc.local_tensor<?xf16>", "!ascendc.local_tensor<?xf16>");
    ASSERT_TRUE(mismatchedShape);

    auto rankCandidates = query.queryLowering(FindOperation(*rankTwo, operationCase.operationName),
                                              ascend::instruction::QueryContext{target});
    auto shapeCandidates = query.queryLowering(FindOperation(*mismatchedShape, operationCase.operationName),
                                               ascend::instruction::QueryContext{target});

    ASSERT_TRUE(succeeded(rankCandidates));
    ASSERT_TRUE(succeeded(shapeCandidates));
    const auto *rankCandidate = FindCandidate(*rankCandidates, operationCase.stableId);
    const auto *shapeCandidate = FindCandidate(*shapeCandidates, operationCase.stableId);
    ASSERT_NE(rankCandidate, nullptr);
    ASSERT_NE(shapeCandidate, nullptr);
    EXPECT_EQ(rankCandidate->support.getReason(), ascend::instruction::CapabilityReason::UnsupportedShape);
    EXPECT_EQ(shapeCandidate->support.getReason(), ascend::instruction::CapabilityReason::UnsupportedShape);
  }
}

// 输入与结果的 dtype 一致性由 AllElementTypesMatch 校验，destination 类型也须与结果一致。
// 这种无效 IR 无法进入 capability 查询，负例由 smoke/afir_tmp_buffer_invalid.mlir 覆盖，
// 此处不重复测试不可达的 capability 分支。

TEST(BinaryElementwiseL2CapabilityTest, RejectsTargetWithoutVectorCores) {
  MLIRContext context;
  LoadDialects(context);
  auto registry = CreateRegistry(context);
  ASSERT_TRUE(registry);
  ascend::instruction::CapabilityQuery query(*registry);
  ascend::TargetProfile target = CreateTargetProfile(/*vectorCoreCount=*/0);

  for (const BinaryOperationCase &operationCase : kBinaryOperationCases) {
    SCOPED_TRACE(operationCase.operationName.str());
    OwningOpRef<ModuleOp> module = ParseBufferizedBinary(
        context, operationCase.operationName, "!ascendc.local_tensor<?xf16>", "!ascendc.local_tensor<?xf16>",
        "!ascendc.local_tensor<?xf16>", "!ascendc.local_tensor<?xf16>");
    ASSERT_TRUE(module);
    auto candidates = query.queryLowering(FindOperation(*module, operationCase.operationName),
                                          ascend::instruction::QueryContext{target});

    ASSERT_TRUE(succeeded(candidates));
    // 没有向量核这条判据在两个 implementation 上一样成立。
    ASSERT_FALSE(candidates->getAll().empty());
    for (const auto &candidate : candidates->getAll()) {
      EXPECT_EQ(candidate.support.getReason(), ascend::instruction::CapabilityReason::TargetFeatureUnavailable);
    }
  }
}

TEST(BinaryElementwiseL2CapabilityTest, QueriesInplaceCapabilityPerImplementation) {
  MLIRContext context;
  LoadDialects(context);
  auto registry = CreateRegistry(context);
  ASSERT_TRUE(registry);
  ascend::instruction::CapabilityQuery query(*registry);
  ascend::TargetProfile target = CreateTargetProfile();

  OwningOpRef<ModuleOp> module =
      ParseBufferizedBinary(context, "afir.add", "!ascendc.local_tensor<?xf16>", "!ascendc.local_tensor<?xf16>",
                            "!ascendc.local_tensor<?xf16>", "!ascendc.local_tensor<?xf16>");
  ASSERT_TRUE(module);
  Operation *operation = FindOperation(*module, "afir.add");

  ascend::instruction::CapabilityQueryRequest request{
      ascend::instruction::CapabilityKind::Inplace, operation, std::nullopt,
      ascend::instruction::InplaceQuery{0, 0, ascend::instruction::AliasAccessMode::ReadWriteInput}};
  auto result = query.queryCapability(request, ascend::instruction::QueryContext{target});
  ASSERT_TRUE(succeeded(result));
  ASSERT_EQ(result->getSupported().size(), 1U);
  EXPECT_EQ(result->getSupported().front().key.id.value, "add.tensor_tensor");

  OwningOpRef<ModuleOp> unsupported =
      ParseBufferizedBinary(context, "afir.add", "!ascendc.local_tensor<?xf16>", "!ascendc.local_tensor<?xf16>",
                            "!ascendc.local_tensor<?xf16>", "!ascendc.local_tensor<?xf16>");
  ASSERT_TRUE(unsupported);
  Operation *unsupportedOperation = FindOperation(*unsupported, "afir.add");

  auto unsupportedResult = query.queryCapability(
      {ascend::instruction::CapabilityKind::Inplace, unsupportedOperation, std::nullopt,
       ascend::instruction::InplaceQuery{0, 0, ascend::instruction::AliasAccessMode::ReadOnlyInput}},
      ascend::instruction::QueryContext{target});
  ASSERT_TRUE(succeeded(unsupportedResult));
  EXPECT_EQ(unsupportedResult->getSupported().size(), 0U);
  const auto &alias =
      std::get<ascend::instruction::AliasConstraint>(result->getSupported().front().support.getConstraints().front());
  EXPECT_EQ(alias.inputOperandIndex, 0U);
  EXPECT_EQ(alias.outputResultIndex, 0U);

  auto rhsResult = query.queryCapability(
      {ascend::instruction::CapabilityKind::Inplace, operation, std::nullopt,
       ascend::instruction::InplaceQuery{1, 0, ascend::instruction::AliasAccessMode::ReadWriteInput}},
      ascend::instruction::QueryContext{target});
  ASSERT_TRUE(succeeded(rhsResult));
  ASSERT_EQ(rhsResult->getSupported().size(), 1U);

  auto destinationResult = query.queryCapability(
      {ascend::instruction::CapabilityKind::Inplace, operation, std::nullopt,
       ascend::instruction::InplaceQuery{2, 0, ascend::instruction::AliasAccessMode::ReadWriteInput}},
      ascend::instruction::QueryContext{target});
  ASSERT_TRUE(succeeded(destinationResult));
  const auto *destinationCandidate = FindCandidate(*destinationResult, "add.tensor_tensor");
  ASSERT_NE(destinationCandidate, nullptr);
  EXPECT_EQ(destinationCandidate->support.getReason(), ascend::instruction::CapabilityReason::MissingInformation);
}

TEST(BinaryElementwiseL2CapabilityTest, RejectsMalformedCapabilityQuery) {
  MLIRContext context;
  LoadDialects(context);
  auto registry = CreateRegistry(context);
  ASSERT_TRUE(registry);
  ascend::instruction::CapabilityQuery query(*registry);
  ascend::TargetProfile target = CreateTargetProfile();
  OwningOpRef<ModuleOp> module =
      ParseBufferizedBinary(context, "afir.add", "!ascendc.local_tensor<?xf16>", "!ascendc.local_tensor<?xf16>",
                            "!ascendc.local_tensor<?xf16>", "!ascendc.local_tensor<?xf16>");
  ASSERT_TRUE(module);

  auto result = query.queryCapability({ascend::instruction::CapabilityKind::Inplace, FindOperation(*module, "afir.add"),
                                       std::nullopt, ascend::instruction::ScalarInputQuery{{0}, false}},
                                      ascend::instruction::QueryContext{target});
  EXPECT_TRUE(failed(result));
}

TEST(BinaryElementwiseL2CapabilityTest, RejectsInvalidCapabilityQueryState) {
  MLIRContext context;
  LoadDialects(context);
  ascend::instruction::ImplementationRegistry registry(context);
  ASSERT_TRUE(succeeded(registerAscendCImplementations(registry)));
  ascend::instruction::CapabilityQuery query(registry);
  ascend::TargetProfile target = CreateTargetProfile();
  OwningOpRef<ModuleOp> module =
      ParseBufferizedBinary(context, "afir.add", "!ascendc.local_tensor<?xf16>", "!ascendc.local_tensor<?xf16>",
                            "!ascendc.local_tensor<?xf16>", "!ascendc.local_tensor<?xf16>");
  ASSERT_TRUE(module);
  Operation *operation = FindOperation(*module, "afir.add");
  ascend::instruction::ImplementationKey missingKey{
      operation->getName(), ascend::instruction::ImplementationKind::AscendC, {"add.missing"}};
  ascend::instruction::CapabilityQueryRequest request{
      ascend::instruction::CapabilityKind::Inplace, operation, missingKey,
      ascend::instruction::InplaceQuery{0, 0, ascend::instruction::AliasAccessMode::ReadWriteInput}};

  EXPECT_TRUE(failed(query.queryCapability(request, ascend::instruction::QueryContext{target})));

  ASSERT_TRUE(succeeded(registry.freeze()));
  EXPECT_TRUE(failed(query.queryCapability(request, ascend::instruction::QueryContext{target})));

  MLIRContext otherContext;
  LoadDialects(otherContext);
  OwningOpRef<ModuleOp> otherModule =
      ParseBufferizedBinary(otherContext, "afir.add", "!ascendc.local_tensor<?xf16>", "!ascendc.local_tensor<?xf16>",
                            "!ascendc.local_tensor<?xf16>", "!ascendc.local_tensor<?xf16>");
  ASSERT_TRUE(otherModule);
  Operation *otherOperation = FindOperation(*otherModule, "afir.add");
  EXPECT_TRUE(failed(query.queryCapability(
      {ascend::instruction::CapabilityKind::Inplace, otherOperation, std::nullopt,
       ascend::instruction::InplaceQuery{0, 0, ascend::instruction::AliasAccessMode::ReadWriteInput}},
      ascend::instruction::QueryContext{target})));
}

TEST(BinaryElementwiseL2CapabilityTest, RejectsSpecifiedImplementationForUnregisteredOperation) {
  MLIRContext context;
  context.allowUnregisteredDialects();
  ascend::instruction::ImplementationRegistry registry(context);
  ASSERT_TRUE(succeeded(registry.freeze()));
  ascend::instruction::CapabilityQuery query(registry);
  ascend::TargetProfile target = CreateTargetProfile();

  OperationState state(UnknownLoc::get(&context), "afir.unknown");
  Operation *operation = Operation::create(state);
  ascend::instruction::ImplementationKey key{
      OperationName("afir.unknown", &context), ascend::instruction::ImplementationKind::AscendC, {"unknown"}};

  EXPECT_TRUE(failed(query.queryCapability({ascend::instruction::CapabilityKind::ScalarInput, operation, key,
                                            ascend::instruction::ScalarInputQuery{{}, true}},
                                           ascend::instruction::QueryContext{target})));
  operation->destroy();
}

TEST(BinaryElementwiseL2CapabilityTest, QueriesSpecifiedImplementationOnly) {
  MLIRContext context;
  LoadDialects(context);
  auto registry = CreateRegistry(context);
  ASSERT_TRUE(registry);
  ascend::instruction::CapabilityQuery query(*registry);
  ascend::TargetProfile target = CreateTargetProfile();
  OwningOpRef<ModuleOp> module =
      ParseBufferizedBinary(context, "afir.add", "!ascendc.local_tensor<?xf16>", "!ascendc.local_tensor<?xf16>",
                            "!ascendc.local_tensor<?xf16>", "!ascendc.local_tensor<?xf16>");
  ASSERT_TRUE(module);
  Operation *operation = FindOperation(*module, "afir.add");
  ascend::instruction::ImplementationKey key{
      operation->getName(), ascend::instruction::ImplementationKind::AscendC, {"add.tensor_tensor"}};

  auto result = query.queryCapability({ascend::instruction::CapabilityKind::ScalarInput, operation, key,
                                       ascend::instruction::ScalarInputQuery{{1}, false}},
                                      ascend::instruction::QueryContext{target});
  ASSERT_TRUE(succeeded(result));
  ASSERT_EQ(result->getAll().size(), 1U);
  EXPECT_EQ(result->getAll().front().key, key);
  EXPECT_EQ(result->getAll().front().support.getReason(), ascend::instruction::CapabilityReason::UnsupportedType);
}

TEST(BinaryElementwiseL2CapabilityTest, RejectsScalarInputCapability) {
  MLIRContext context;
  LoadDialects(context);
  auto registry = CreateRegistry(context);
  ASSERT_TRUE(registry);
  ascend::instruction::CapabilityQuery query(*registry);
  // `afir.add` 现在还有一个 `MemBased` 的 row-broadcast-inline implementation（见下面
  // `RejectsBroadcastInlineCapability` 的注释），所以这里也要用能把那个 feature 答
  // 确定的 profile，否则它答 `Undetermined` 而不是 `UnsupportedType`。
  ascend::TargetProfile target = CreateMemBasedTargetProfile();
  OwningOpRef<ModuleOp> module =
      ParseBufferizedBinary(context, "afir.add", "!ascendc.local_tensor<?xf16>", "!ascendc.local_tensor<?xf16>",
                            "!ascendc.local_tensor<?xf16>", "!ascendc.local_tensor<?xf16>");
  ASSERT_TRUE(module);
  Operation *operation = FindOperation(*module, "afir.add");

  for (ascend::instruction::ScalarInputQuery detail : {
           ascend::instruction::ScalarInputQuery{{0}, false},
           ascend::instruction::ScalarInputQuery{{}, true},
       }) {
    auto result =
        query.queryCapability({ascend::instruction::CapabilityKind::ScalarInput, operation, std::nullopt, detail},
                              ascend::instruction::QueryContext{target});
    ASSERT_TRUE(succeeded(result));
    ASSERT_FALSE(result->getAll().empty());
    for (const auto &candidate : result->getAll()) {
      EXPECT_EQ(candidate.support.getState(), ascend::instruction::SupportState::Unsupported);
      EXPECT_EQ(candidate.support.getReason(), ascend::instruction::CapabilityReason::UnsupportedType);
    }
  }
}

TEST(BinaryElementwiseL2CapabilityTest, RejectsBroadcastInlineCapability) {
  MLIRContext context;
  LoadDialects(context);
  auto registry = CreateRegistry(context);
  ASSERT_TRUE(registry);
  ascend::instruction::CapabilityQuery query(*registry);
  // 行广播（brc_inline）implementation 真的会答这个查询（见
  // `BinaryBrcInlineCapabilityTest.AcceptsRowBroadcastShape`），所以这里要用一张不满足
  // 那个形态的图才钉得住「其余 implementation 一律不支持」——两个输入的 descriptor 都是
  // 裸的 `role = <input>`，没有 `axis` / `vectorized_axis`，本来就不满足任何形态判据。
  ascend::TargetProfile target = CreateMemBasedTargetProfile();
  OwningOpRef<ModuleOp> module =
      ParseBufferizedBinary(context, "afir.add", "!ascendc.local_tensor<?xf16>", "!ascendc.local_tensor<?xf16>",
                            "!ascendc.local_tensor<?xf16>", "!ascendc.local_tensor<?xf16>");
  ASSERT_TRUE(module);
  Operation *operation = FindOperation(*module, "afir.add");

  auto result = query.queryCapability({ascend::instruction::CapabilityKind::BroadcastInline, operation, std::nullopt,
                                       ascend::instruction::BroadcastInlineQuery{0, 2, 0, {1}, {1}, false}},
                                      ascend::instruction::QueryContext{target});
  ASSERT_TRUE(succeeded(result));
  ASSERT_FALSE(result->getAll().empty());
  for (const auto &candidate : result->getAll()) {
    EXPECT_EQ(candidate.support.getState(), ascend::instruction::SupportState::Unsupported);
  }
}

TEST(BinaryElementwiseL2CapabilityTest, ProductRegistrationUsesSemanticStableIds) {
  MLIRContext context;
  LoadDialects(context);
  auto registry = CreateRegistry(context);
  ASSERT_TRUE(registry);
  ascend::instruction::CapabilityQuery query(*registry);
  ascend::TargetProfile target = CreateTargetProfile();

  for (const BinaryOperationCase &operationCase : kBinaryOperationCases) {
    SCOPED_TRACE(operationCase.operationName.str());
    OwningOpRef<ModuleOp> module = ParseBufferizedBinary(
        context, operationCase.operationName, "!ascendc.local_tensor<?xf16>", "!ascendc.local_tensor<?xf16>",
        "!ascendc.local_tensor<?xf16>", "!ascendc.local_tensor<?xf16>");
    ASSERT_TRUE(module);
    auto candidates = query.queryLowering(FindOperation(*module, operationCase.operationName),
                                          ascend::instruction::QueryContext{target});

    ASSERT_TRUE(succeeded(candidates));
    ASSERT_EQ(candidates->getSupported().size(), 1U);
    EXPECT_EQ(candidates->getSupported().front().key.id.value, operationCase.stableId);
  }
}

TEST(BinaryElementwiseL2CapabilityTest, ProductRegistrationRejectsDuplicateKeysAtFreeze) {
  MLIRContext context;
  LoadDialects(context);
  ascend::instruction::ImplementationRegistry registry(context);

  ASSERT_TRUE(succeeded(registerAscendCImplementations(registry)));
  ASSERT_TRUE(succeeded(registerAscendCImplementations(registry)));
  EXPECT_TRUE(failed(registry.freeze()));
  EXPECT_FALSE(registry.isFrozen());
}

TEST(BinaryElementwiseL2CapabilityTest, ProductRegistrationRejectsMutationAfterFreeze) {
  MLIRContext context;
  LoadDialects(context);
  ascend::instruction::ImplementationRegistry registry(context);

  ASSERT_TRUE(succeeded(registerAscendCImplementations(registry)));
  ASSERT_TRUE(succeeded(registry.freeze()));
  EXPECT_TRUE(failed(registerAscendCImplementations(registry)));
}

TEST(BinaryElementwiseL2CapabilityTest, SingletonScalarCannotAliasOutput) {
  MLIRContext context;
  LoadDialects(context);
  auto registry = CreateRegistry(context);
  ASSERT_TRUE(registry);
  ascend::instruction::CapabilityQuery query(*registry);
  ascend::TargetProfile target = CreateTargetProfile();
  for (unsigned scalarIndex : {0U, 1U}) {
    std::string ir = R"mlir(
#one = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"1">], strides = [#afir.expr<"0">], vectorized_axis = [0], vectorized_strides = [#afir.expr<"0">], role = <compute>>
#full = #afir.tensor_desc<axis = [0], repeats = [#afir.expr<"32">], strides = [#afir.expr<"1">], vectorized_axis = [0], vectorized_strides = [#afir.expr<"1">], role = <compute>>
func.func @test(%input: !ascendc.local_tensor<32xf32>, %dst: !ascendc.local_tensor<32xf32>) {
  %pipe = ascendc.pipe
  %buf = ascendc.tbuf : <veccalc>
  %one = ascendc.tbuf.get_tensor %buf : !ascendc.tbuf<veccalc>, !ascendc.local_tensor<1xf32>
)mlir";
    bool left = scalarIndex == 0;
    ir += left ? "  %r = afir.add %one, %input" : "  %r = afir.add %input, %one";
    ir += " outs(%dst) {afir.operand_descs = ";
    ir += left ? "[#one, #full]" : "[#full, #one]";
    ir += ", afir.destination_descs = [#full], afir.result_descs = [#full]} : (";
    ir += left ? "!ascendc.local_tensor<1xf32>, !ascendc.local_tensor<32xf32>"
               : "!ascendc.local_tensor<32xf32>, !ascendc.local_tensor<1xf32>";
    ir += "), !ascendc.local_tensor<32xf32> -> !ascendc.local_tensor<32xf32>\n return\n}";
    auto module = parseSourceString<ModuleOp>(ir, &context);
    ASSERT_TRUE(module);
    auto *operation = FindOperation(*module, "afir.add");
    for (unsigned aliasIndex : {0U, 1U}) {
      auto result = query.queryCapability(
          {ascend::instruction::CapabilityKind::Inplace, operation, std::nullopt,
           ascend::instruction::InplaceQuery{aliasIndex, 0, ascend::instruction::AliasAccessMode::ReadWriteInput}},
          ascend::instruction::QueryContext{target});
      ASSERT_TRUE(succeeded(result));
      EXPECT_EQ(result->getSupported().size(), aliasIndex == scalarIndex ? 0U : 1U);
      if (aliasIndex != scalarIndex) {
        EXPECT_EQ(result->getSupported().front().key.id.value, "add.tensor_scalar");
      }
    }
  }
}

// The GM rows differ while the UB tile is identical. The old transfer-plan
// comparison rejected the one-segment/two-segment pair before API selection.
TEST(BinaryElementwiseL2CapabilityTest, LocalCompatibilityIgnoresGmAxisMerging) {
  MLIRContext context;
  LoadDialects(context);
  const llvm::StringRef type = "!ascendc.local_tensor<2x4x8xf32>";
  auto module = ParseBufferizedBinary(context, "afir.add", type, type, type, type);
  ASSERT_TRUE(module);
  auto *op = FindOperation(*module, "afir.add");
  auto graph = parseAttribute(R"mlir(#afir.asc_graph<tiling_key = 0, axes = [
    <id = 0, name = "outer", axis_type = Original, size = <"2">>,
    <id = 1, name = "row", axis_type = Original, size = <"4">>,
    <id = 2, name = "col", axis_type = Original, size = <"8">>], type = Compute>)mlir",
                              &context);
  ASSERT_TRUE(graph);
  op->getParentOfType<func::FuncOp>()->setAttr(kAscGraphAttrName, graph);
  op->setAttr(kSchedAttrName, parseAttribute("#afir.sched<axis = [0, 1, 2], loop_axis = 0>", &context));
  auto descriptor = [&](llvm::StringRef gmRow, llvm::StringRef ubRow, llvm::StringRef rows) {
    auto parsed = parseAttribute("#afir.tensor_desc<axis = [0, 1, 2], repeats = [<\"2\">, <\"" + rows.str() +
                                     "\">, <\"8\">], strides = [<\"128\">, <\"" + gmRow.str() +
                                     "\">, <\"1\">], vectorized_axis = [1, 2], vectorized_strides = [<\"" +
                                     ubRow.str() + "\">, <\"1\">], role = <compute>>",
                                 &context);
    auto result = dyn_cast_or_null<TensorDescAttr>(parsed);
    EXPECT_TRUE(result) << "invalid local-view test descriptor";
    return result;
  };
  auto source = descriptor("8", "8", "4");
  auto destination = descriptor("32", "8", "4");
  auto sourcePlan = buildTransferPlanForDescriptor(op, source);
  auto destinationPlan = buildTransferPlanForDescriptor(op, destination);
  ASSERT_TRUE(succeeded(sourcePlan));
  ASSERT_TRUE(succeeded(destinationPlan));
  ASSERT_EQ(sourcePlan->kind, TransferPlanKind::Scheduled);
  ASSERT_EQ(sourcePlan->dims.size(), 1U);
  ASSERT_EQ(destinationPlan->dims.size(), 2U);

  using ascend::kernel::elementwise::LocalViewCompatibility;
  bool wholeTensor = true;
  EXPECT_EQ(ascend::kernel::elementwise::checkLocalViewCompatibility(op, {source, destination}, &wholeTensor),
            LocalViewCompatibility::Compatible);
  EXPECT_FALSE(wholeTensor);
  EXPECT_EQ(ascend::kernel::elementwise::checkLocalViewCompatibility(op, {destination, source}),
            LocalViewCompatibility::Compatible);
  EXPECT_EQ(ascend::kernel::elementwise::checkLocalViewCompatibility(op, {source, descriptor("0", "8", "4")}),
            LocalViewCompatibility::Compatible);
  EXPECT_EQ(ascend::kernel::elementwise::checkLocalViewCompatibility(op, {source, descriptor("32", "16", "4")}),
            LocalViewCompatibility::Compatible);
  EXPECT_EQ(ascend::kernel::elementwise::checkLocalViewCompatibility(op, {source, descriptor("32", "8", "3")}),
            LocalViewCompatibility::Compatible);
  EXPECT_EQ(ascend::kernel::elementwise::checkLocalViewCompatibility(op, {source, TensorDescAttr{}}),
            LocalViewCompatibility::MissingInformation);
}

TEST(BinaryElementwiseL2CapabilityTest, LocalViewChecksDomainWindowAndStaticStorage) {
  MLIRContext context;
  LoadDialects(context);
  const llvm::StringRef type = "!ascendc.local_tensor<2x4x8xf32>";
  auto module = ParseBufferizedBinary(context, "afir.add", type, type, type, type);
  ASSERT_TRUE(module);
  Operation *op = FindOperation(*module, "afir.add");
  auto desc = [&](StringRef row, StringRef col = "1", StringRef rows = "4") {
    // tensor_desc requires a GM array, but these unbound symbols must not be
    // interpreted by the local computation analysis.
    auto expr = [&](StringRef text) { return *makeExprAttr(&context, text); };
    return TensorDescAttr::get(&context, {0, 1, 2}, {expr("2"), expr(rows), expr("8")},
                               {expr("gm0"), expr("gm1"), expr("gm2")}, {1, 2}, {expr(row), expr(col)},
                               RoleAttr::get(&context, Role::Compute));
  };
  auto contiguous = desc("8");
  ASSERT_TRUE(contiguous);
  op->setAttr(kDestinationDescsAttrName, ArrayAttr::get(&context, {contiguous}));
  op->setAttr(kResultDescsAttrName, ArrayAttr::get(&context, {contiguous}));
  using ascend::kernel::elementwise::ElementwiseLocalView;
  using ascend::kernel::elementwise::LocalViewCompatibility;
  auto analyze = [&](TensorDescAttr input, ArrayRef<unsigned> scalars = {}) {
    op->setAttr(kOperandDescsAttrName, ArrayAttr::get(&context, {input, contiguous}));
    ElementwiseLocalView view;
    return ascend::kernel::elementwise::prepareElementwiseLocalView(op, scalars, false, view);
  };
  EXPECT_EQ(analyze(desc("16")), LocalViewCompatibility::Compatible);
  // No GM symbol bindings, schedule or graph required for a complete local view.
  ElementwiseLocalView view;
  ASSERT_EQ(ascend::kernel::elementwise::prepareElementwiseLocalView(op, {}, false, view),
            LocalViewCompatibility::Compatible);
  ASSERT_TRUE(view.vectorPlan);
  ASSERT_EQ(view.vectorPlan->segments.size(), 2U);
  EXPECT_EQ(view.extentSource, ExtentSource::Nominal);
  EXPECT_TRUE(provablyEqual(view.vectorPlan->segments[0].strides[0], *makeExprAttr(&context, "16")));
  EXPECT_TRUE(provablyEqual(view.vectorPlan->segments[0].strides[1], *makeExprAttr(&context, "8")));
  EXPECT_EQ(analyze(desc("32")), LocalViewCompatibility::UnsupportedShape);           // span exceeds capacity
  EXPECT_EQ(analyze(desc("4")), LocalViewCompatibility::UnsupportedShape);            // overlapping rows
  EXPECT_EQ(analyze(desc("9")), LocalViewCompatibility::UnsupportedShape);            // unaligned API starts
  EXPECT_EQ(analyze(desc("16", "2")), LocalViewCompatibility::UnsupportedShape);      // strided API window
  EXPECT_EQ(analyze(desc("8", "1", "1")), LocalViewCompatibility::UnsupportedShape);  // implicit broadcast
  // Scalar selection occurs before merging: the excluded descriptor cannot split
  // the continuous window or alter its domain, and original operand indices remain.
  op->setAttr(kOperandDescsAttrName, ArrayAttr::get(&context, {desc("0", "0", "1"), contiguous}));
  ASSERT_EQ(ascend::kernel::elementwise::prepareElementwiseLocalView(op, {0}, false, view),
            LocalViewCompatibility::Compatible);
  ASSERT_EQ(view.vectorPlan->tensors.size(), 2U);
  EXPECT_EQ(view.vectorPlan->tensors[0].index, 1U);
  EXPECT_EQ(view.vectorPlan->segments.size(), 1U);
  EXPECT_FALSE(ascend::kernel::elementwise::hasSameLocalLayout(op, 0));
  EXPECT_TRUE(ascend::kernel::elementwise::hasSameLocalLayout(op, 1));
  auto duplicate = TensorDescAttr::get(&context, contiguous.getAxis(), contiguous.getRepeats(),
                                       contiguous.getStrides(), {1, 1}, contiguous.getVectorizedStrides(),
                                       contiguous.getRole());
  EXPECT_EQ(analyze(duplicate), LocalViewCompatibility::MissingInformation);
  auto incomplete = TensorDescAttr::get(&context, {0, 1, 2}, {}, {}, {1, 2},
                                        {*makeExprAttr(&context, "8"), *makeExprAttr(&context, "1")},
                                        RoleAttr::get(&context, Role::Compute));
  EXPECT_EQ(analyze(incomplete), LocalViewCompatibility::MissingInformation);
  op->setAttr(kDestinationDescsAttrName, ArrayAttr::get(&context, {desc("32")}));
  op->setAttr(kOperandDescsAttrName, ArrayAttr::get(&context, {desc("32"), desc("32")}));
  EXPECT_EQ(ascend::kernel::elementwise::prepareElementwiseLocalView(op, {}, false, view),
            LocalViewCompatibility::UnsupportedShape);
  op->setAttr(kDestinationDescsAttrName, ArrayAttr::get(&context, {contiguous}));
  context.getOrLoadDialect<arith::ArithDialect>();
  OpBuilder builder(op);
  Value zero = builder.create<arith::ConstantIndexOp>(op->getLoc(), 0);
  Value alias = builder.create<ascendc::LocalTensorSubIndexOp>(op->getLoc(), op->getOperand(0).getType(),
                                                               op->getOperand(0), zero);
  op->setOperand(2, alias);
  EXPECT_EQ(analyze(desc("16")), LocalViewCompatibility::UnsupportedShape);
  op->setOperand(2, op->getOperand(0));
  EXPECT_EQ(analyze(desc("16")), LocalViewCompatibility::UnsupportedShape);  // incompatible in-place mapping
}

TEST(BinaryElementwiseL2CapabilityTest, SymbolicIndependentPitchesAndAxisOrder) {
  MLIRContext context;
  LoadDialects(context);
  const llvm::StringRef sourceType = "!ascendc.local_tensor<?x8x?xf32>";
  const llvm::StringRef destinationType = "!ascendc.local_tensor<?x?x8xf32>";
  auto module =
      ParseBufferizedBinary(context, "afir.add", sourceType, destinationType, destinationType, destinationType);
  ASSERT_TRUE(module);
  Operation *op = FindOperation(*module, "afir.add");
  auto expr = [&](StringRef text) { return *makeExprAttr(&context, text); };
  auto source = TensorDescAttr::get(
      &context, {0, 2, 1}, {expr("m"), expr("8"), expr("n")}, {expr("gm0"), expr("gm2"), expr("gm1")}, {1, 2},
      {expr("(16 * Ceiling((Rational(1 , 16) * n)))"), expr("1")}, RoleAttr::get(&context, Role::Compute));
  auto destination = TensorDescAttr::get(
      &context, {0, 1, 2}, {expr("m"), expr("n"), expr("8")}, {expr("gm0"), expr("gm1"), expr("gm2")}, {1, 2},
      {expr("(8 * Ceiling((Rational(1 , 8) * n)))"), expr("1")}, RoleAttr::get(&context, Role::Compute));
  op->setAttr(kOperandDescsAttrName, ArrayAttr::get(&context, {source, destination}));
  op->setAttr(kDestinationDescsAttrName, ArrayAttr::get(&context, {destination}));
  op->setAttr(kResultDescsAttrName, ArrayAttr::get(&context, {destination}));
  using namespace ascend::kernel::elementwise;
  ElementwiseLocalView view;
  EXPECT_EQ(prepareElementwiseLocalView(op, {}, false, view), LocalViewCompatibility::Compatible);
  ASSERT_TRUE(view.vectorPlan);
  EXPECT_EQ(view.vectorPlan->segments.size(), 2U);
  auto registry = CreateRegistry(context);
  ASSERT_TRUE(registry);
  ascend::instruction::CapabilityQuery query(*registry);
  auto target = CreateTargetProfile();
  auto candidates = query.queryLowering(op, ascend::instruction::QueryContext{target});
  ASSERT_TRUE(succeeded(candidates));
  const auto *candidate = FindCandidate(*candidates, "add.tensor_tensor");
  ASSERT_NE(candidate, nullptr);
  EXPECT_EQ(candidate->support.getState(), ascend::instruction::SupportState::Supported);
  // A scalar-looking dimension inside the vector domain is still broadcast, not an equal-domain Add.
  auto broadcast =
      TensorDescAttr::get(&context, {0, 1, 2}, {expr("m"), expr("1"), expr("8")}, {expr("gm0"), expr("0"), expr("gm2")},
                          {1, 2}, {expr("0"), expr("1")}, RoleAttr::get(&context, Role::Compute));
  op->setAttr(kOperandDescsAttrName, ArrayAttr::get(&context, {broadcast, destination}));
  EXPECT_EQ(analyzeElementwiseLocalView(op, {}, false, view), LocalViewCompatibility::Compatible);
  EXPECT_EQ(prepareElementwiseLocalView(op, {}, false, view), LocalViewCompatibility::UnsupportedShape);
}

}  // namespace
}  // namespace mlir::afir
