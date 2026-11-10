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
#include "conversion/AFIRToAscendC/RegisterAscendCImplementations.h"
#include "target/Ascend/Capability/CapabilityQuery.h"
#include "target/Ascend/Capability/ImplementationRegistry.h"

#include <optional>
#include <string>

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
  ascend::TargetProfile target = CreateTargetProfile();
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
  ascend::TargetProfile target = CreateTargetProfile();
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
    EXPECT_EQ(candidate.support.getReason(), ascend::instruction::CapabilityReason::UnsupportedShape);
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

}  // namespace
}  // namespace mlir::afir
