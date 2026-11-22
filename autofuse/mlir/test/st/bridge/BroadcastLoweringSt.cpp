/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/AFIRToAscendC/AFIRToAscendCPipeline.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFPlan/IR/AFPlan.h"
#include "target/Ascend/TargetProfileLoader.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"

namespace mlir::afir {
namespace {
struct BroadcastCase {
  const char *file;
  bool scheduled;
  const char *api;
};

class BroadcastLoweringSt : public testing::TestWithParam<BroadcastCase> {};

void LoadDialects(MLIRContext &context) {
  context.loadDialect<AFIRDialect, afplan::AFPlanDialect, ascendc::AscendCDialect, emitasc::EmitAscDialect,
                      arith::ArithDialect, bufferization::BufferizationDialect, emitc::EmitCDialect, func::FuncDialect,
                      memref::MemRefDialect, scf::SCFDialect>();
}

LogicalResult Lower(ModuleOp module, bool scheduled) {
  llvm::raw_null_ostream diagnostic;
  auto profile = ascend::TargetProfileLoader::loadBuiltin("Ascend910B1", diagnostic);
  if (failed(profile)) {
    return failure();
  }
  auto session = ascend::kernel::createAFIRToAscendCPipelineSession(
      *module.getContext(), std::make_shared<const ascend::TargetProfile>(std::move(*profile)));
  if (failed(session)) {
    return failure();
  }
  PassManager manager(module.getContext());
  auto added = scheduled ? ascend::kernel::addPlanCandidateLoweringPipeline(manager, *session)
                         : ascend::kernel::addAFIRToAscendCPipeline(manager, *session);
  return failed(added) ? failure() : manager.run(module);
}

TEST_P(BroadcastLoweringSt, PreservesLayoutAndEliminatesBroadcast) {
  MLIRContext context;
  LoadDialects(context);
  const auto &test = GetParam();
  auto module = parseSourceFile<ModuleOp>(std::string(AF_BROADCAST_FIXTURE_DIR) + "/" + test.file, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(Lower(*module, test.scheduled)));
  unsigned calls = 0;
  module->walk([&](Operation *op) {
    EXPECT_NE(op->getName().getStringRef(), "afir.broadcast");
    if (op->getName().getStringRef() != test.api) {
      return;
    }
    ++calls;
    if (StringRef(test.file).contains("duplicate-rows")) {
      auto loop = op->getParentOfType<scf::ForOp>();
      ASSERT_TRUE(loop);
      auto count = op->getOperand(2).getDefiningOp<arith::ConstantIndexOp>();
      ASSERT_TRUE(count);
      EXPECT_EQ(count.value(), 8);
      auto scalar = op->getOperand(1).getDefiningOp();
      ASSERT_NE(scalar, nullptr);
      EXPECT_EQ(scalar->getName().getStringRef(), "ascendc.local_tensor.get_value");
      EXPECT_FALSE(loop->isAncestor(scalar));
      auto subindex = op->getOperand(0).getDefiningOp();
      ASSERT_NE(subindex, nullptr);
      auto offset = subindex->getOperand(1).getDefiningOp<arith::MulIOp>();
      ASSERT_TRUE(offset);
      EXPECT_EQ(offset.getLhs(), loop.getInductionVar());
      EXPECT_EQ(offset.getRhs().getDefiningOp<arith::ConstantIndexOp>().value(), 16);
    }
    if (StringRef(test.file).contains("padded-suffix")) {
      EXPECT_EQ(op->getOperand(3).getDefiningOp<arith::ConstantIndexOp>().value(), 24);
      EXPECT_EQ(op->getOperand(5).getDefiningOp<arith::ConstantIndexOp>().value(), 4);
      EXPECT_EQ(op->getOperand(6), op->getOperand(3));
    }
  });
  EXPECT_GT(calls, 0U);
}

INSTANTIATE_TEST_SUITE_P(
    Layouts, BroadcastLoweringSt,
    testing::Values(BroadcastCase{"afir-broadcast-duplicate-rows.mlir", true, "ascendc.duplicate_l2"},
                    BroadcastCase{"afir-broadcast-duplicate.mlir", true, "ascendc.duplicate_l2"},
                    BroadcastCase{"afir-broadcast-duplicate-scalar.mlir", true, "ascendc.duplicate_l2"},
                    BroadcastCase{"afir-broadcast-padded-suffix.mlir", false, "ascendc.broadcast_dim3"},
                    BroadcastCase{"afir-broadcast-segmented.mlir", false, "ascendc.broadcast_dim3"},
                    BroadcastCase{"afir-broadcast-source-pitch.mlir", false, "ascendc.broadcast_dim3"},
                    BroadcastCase{"afir-broadcast-inner-pitch.mlir", false, "ascendc.broadcast_dim3"},
                    BroadcastCase{"afir-broadcast-tail-pitch-inner-brc.mlir", false, "ascendc.broadcast_dim3"}));

TEST(BroadcastRejectionSt, RejectsUnsafeDuplicateRows) {
  auto input =
      llvm::MemoryBuffer::getFile(std::string(AF_BROADCAST_FIXTURE_DIR) + "/afir-broadcast-duplicate-rows.mlir");
  ASSERT_TRUE(input);
  const std::pair<StringRef, StringRef> replacements[] = {{"f32", "ui8"}, {"<\"16\">", "<\"15\">"}};
  for (const auto &replacement : replacements) {
    std::string source = (*input)->getBuffer().str();
    size_t position = 0;
    while ((position = source.find(replacement.first.str(), position)) != std::string::npos) {
      source.replace(position, replacement.first.size(), replacement.second.str());
      position += replacement.second.size();
    }
    MLIRContext context;
    LoadDialects(context);
    auto module = parseSourceString<ModuleOp>(source, &context);
    ASSERT_TRUE(module);
    ScopedDiagnosticHandler handler(&context, [](Diagnostic &) { return success(); });
    EXPECT_TRUE(failed(Lower(*module, true)));
  }
}

TEST(BroadcastRejectionSt, RejectsIncompatibleWindowLayouts) {
  auto input = llvm::MemoryBuffer::getFile(std::string(AF_BROADCAST_FIXTURE_DIR) + "/afir-broadcast-unsupported.mlir");
  ASSERT_TRUE(input);
  SmallVector<StringRef> cases;
  (*input)->getBuffer().split(cases, "// -----");
  ASSERT_EQ(cases.size(), 5U);
  for (auto source : cases) {
    MLIRContext context;
    LoadDialects(context);
    auto module = parseSourceString<ModuleOp>(source, &context);
    ASSERT_TRUE(module);
    ScopedDiagnosticHandler handler(&context, [](Diagnostic &) { return success(); });
    EXPECT_TRUE(failed(Lower(*module, false)));
  }
}
}  // namespace
}  // namespace mlir::afir
