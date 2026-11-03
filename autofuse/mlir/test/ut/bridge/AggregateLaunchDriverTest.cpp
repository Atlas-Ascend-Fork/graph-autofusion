/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "artifact/KernelSource/AggregateLaunchDriver.h"
#include "artifact/KernelSource/AddKernelIncludes.h"

#include "common/tiling_schema/schedule_group_names.h"
#include "conversion/AFIRToAscendC/AscendCKernelABI.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFPlan/IR/AFPlan.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <gtest/gtest.h>

namespace {
// Minimal valid Plan module: same shape as IsolatedPlanEntryTest.cpp's makePlan
// helper, kept local to this file since it is only needed to build inputs for
// runAggregateLaunchDriver.
mlir::OwningOpRef<mlir::ModuleOp> makePlan(mlir::MLIRContext &context, int64_t id) {
  context.getOrLoadDialect<mlir::afplan::AFPlanDialect>();
  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc(), "plan" + std::to_string(id));
  module->setAttr("afir.module_kind", builder.getStringAttr("plan"));
  module->setAttr(mlir::afplan::kCandidateAttrName, mlir::afplan::CandidateAttr::get(&context, id, 0));
  module->setAttr(mlir::afplan::kStagesAttrName, mlir::afplan::StagesAttr::get(&context, {1}));
  auto stage = mlir::emitasc::PyStructType::get(&context, builder.getStringAttr("plan0_StageData"),
                                                builder.getArrayAttr({mlir::TypeAttr::get(builder.getI32Type())}),
                                                builder.getArrayAttr({builder.getStringAttr("tiling_key")}));
  auto schema = mlir::emitasc::PyStructType::get(
      &context, builder.getStringAttr("PlanData"), builder.getArrayAttr({mlir::TypeAttr::get(stage)}),
      builder.getArrayAttr({builder.getStringAttr(mlir::common::MakeScheduleGroupTilingFieldName(id, 0, 0))}));
  module->setAttr("ascendc.tiling_schema", mlir::TypeAttr::get(schema));
  return module;
}

// "plan0_main" with a real body, so GenAggregateKernelBody's clone step has a
// definition to clone (mirrors IsolatedPlanEntryTest.cpp's addRealBody helper).
void addRealBody(mlir::func::FuncOp function, mlir::func::FuncOp observer) {
  mlir::OpBuilder builder(function.getContext());
  mlir::Block *entry = function.addEntryBlock();
  builder.setInsertionPointToEnd(entry);
  auto first = builder.create<mlir::arith::ConstantIntOp>(function.getLoc(), 7, 32);
  auto second = builder.create<mlir::arith::ConstantIntOp>(function.getLoc(), 7, 32);
  builder.create<mlir::func::CallOp>(function.getLoc(), observer, mlir::ValueRange{first});
  builder.create<mlir::func::CallOp>(function.getLoc(), observer, mlir::ValueRange{second});
  builder.create<mlir::func::ReturnOp>(function.getLoc());
}

size_t countOccurrences(llvm::StringRef text, llvm::StringRef needle) {
  size_t count = 0;
  size_t offset = 0;
  while ((offset = text.find(needle, offset)) != llvm::StringRef::npos) {
    ++count;
    offset += needle.size();
  }
  return count;
}

mlir::OwningOpRef<mlir::ModuleOp> makePlanWithRealMain(mlir::MLIRContext &context, int64_t id) {
  auto plan = makePlan(context, id);
  auto schema = (*plan)->getAttrOfType<mlir::TypeAttr>("ascendc.tiling_schema");
  mlir::OpBuilder builder(plan->getBodyRegion());
  builder.setInsertionPointToEnd(plan->getBody());
  // AutofuseLaunchEmitter requires exactly one Workspace-kind argument among
  // the aggregate arguments, so "_main" needs a workspace argument alongside
  // the tiling struct argument. 身份由参数自己声明——`getKernelArgumentIdentity`
  // 不再按「类型恰好等于 workspace 类型」反推（设计文档 3.6）。
  mlir::Type workspaceType = mlir::ascend::kernel::getAscendCWorkspaceType(&context);
  auto observer = builder.create<mlir::func::FuncOp>(plan->getLoc(), plan->getSymName()->str() + "_observe",
                                                     mlir::FunctionType::get(&context, {builder.getI32Type()}, {}));
  mlir::Block *observerEntry = observer.addEntryBlock();
  builder.setInsertionPointToEnd(observerEntry);
  builder.create<mlir::func::ReturnOp>(observer.getLoc());
  builder.setInsertionPointToEnd(plan->getBody());
  // `_main` 一 stage 收一个 tiling 形参、序号即 stage；这里的 Plan 只有一个 stage。
  auto planSchema = mlir::cast<mlir::emitasc::PyStructType>(schema.getValue());
  auto stageType = mlir::cast<mlir::TypeAttr>(planSchema.getTypesAttr()[0]).getValue();
  auto realMain = builder.create<mlir::func::FuncOp>(plan->getLoc(), plan->getSymName()->str() + "_main",
                                                     mlir::FunctionType::get(&context, {workspaceType, stageType}, {}));
  mlir::afir::setKernelArgAttr(realMain, 0, mlir::afir::KernelArgRole::Workspace);
  mlir::afir::setKernelArgAttr(realMain, 1, mlir::afir::KernelArgRole::Tiling, 0);
  addRealBody(realMain, observer);
  return plan;
}

void loadDriverDialects(mlir::MLIRContext &context) {
  context.getOrLoadDialect<mlir::afir::AFIRDialect>();
  context.getOrLoadDialect<mlir::afplan::AFPlanDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::emitasc::EmitAscDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::emitc::EmitCDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
}

mlir::ModuleOp makeEntryWithIncludes(mlir::MLIRContext &context) {
  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc(), "entry");
  module->setAttr("afir.module_kind", builder.getStringAttr("entry"));
  auto schema = mlir::emitasc::PyStructType::get(&context, builder.getStringAttr("TilingData"),
                                                 builder.getArrayAttr({mlir::TypeAttr::get(builder.getI32Type())}),
                                                 builder.getArrayAttr({builder.getStringAttr("field")}));
  module->setAttr("ascendc.tiling_schema", mlir::TypeAttr::get(schema));

  mlir::OpBuilder moduleBuilder(module.getBodyRegion());
  moduleBuilder.setInsertionPointToStart(module.getBody());
  moduleBuilder.create<mlir::emitc::IncludeOp>(module.getLoc(), "kernel_operator.h");
  moduleBuilder.create<mlir::emitc::IncludeOp>(module.getLoc(), "kernel_operator.h");
  moduleBuilder.create<mlir::emitc::IncludeOp>(module.getLoc(), "cstdint");
  moduleBuilder.create<mlir::emitc::IncludeOp>(module.getLoc(), "custom_header.h");
  return module;
}
}  // namespace

TEST(AddKernelIncludesTest, RebuildsManagedIncludesAndPreservesOtherIncludes) {
  mlir::MLIRContext context;
  loadDriverDialects(context);
  auto module = makeEntryWithIncludes(context);

  ASSERT_TRUE(succeeded(mlir::ascend::kernel::runAddKernelIncludes(module)));

  auto collectIncludes = [&]() {
    llvm::SmallVector<std::string> includes;
    for (mlir::emitc::IncludeOp include : module.getOps<mlir::emitc::IncludeOp>()) {
      includes.push_back(include.getInclude().str());
    }
    return includes;
  };

  const auto firstIncludes = collectIncludes();
  ASSERT_EQ(firstIncludes.size(), 4U);
  EXPECT_EQ(firstIncludes[0], "kernel_operator.h");
  EXPECT_EQ(firstIncludes[1], "autofuse_tiling_data.h");
  EXPECT_EQ(firstIncludes[2], "cstdint");
  EXPECT_EQ(firstIncludes[3], "custom_header.h");

  ASSERT_TRUE(succeeded(mlir::ascend::kernel::runAddKernelIncludes(module)));
  EXPECT_EQ(collectIncludes(), firstIncludes);
}

TEST(AggregateLaunchDriverTest, ProducesNonEmptyArtifactsForValidPlan) {
  mlir::MLIRContext context;
  loadDriverDialects(context);
  auto plan0 = makePlanWithRealMain(context, 0);
  llvm::SmallVector<mlir::ModuleOp> modules = {*plan0};

  std::string diagnostic;
  auto artifacts = mlir::ascend::kernel::runAggregateLaunchDriver(modules, "FusedGraph", &diagnostic);
  ASSERT_TRUE(succeeded(artifacts)) << diagnostic;
  EXPECT_TRUE(diagnostic.empty());

  EXPECT_FALSE(artifacts->kernel_body.empty());
  EXPECT_NE(artifacts->kernel_body.find("fused_graph"), std::string::npos);
  EXPECT_EQ(countOccurrences(artifacts->kernel_body, " = 7;"), 1U);
  EXPECT_EQ(countOccurrences(artifacts->kernel_body, "observe("), 3U);  // One definition and two calls.

  EXPECT_FALSE(artifacts->tiling_data.empty());
  EXPECT_NE(artifacts->tiling_data.find("AutofuseTilingData"), std::string::npos);

  EXPECT_FALSE(artifacts->autofuse_launch.empty());
  EXPECT_NE(artifacts->autofuse_launch.find("AutofuseLaunch"), std::string::npos);
  EXPECT_NE(artifacts->autofuse_launch.find("fused_graph<<<blockDim"), std::string::npos);
}

TEST(AggregateLaunchDriverTest, ProducesArtifactsForMultiplePlans) {
  mlir::MLIRContext context;
  loadDriverDialects(context);
  auto plan0 = makePlanWithRealMain(context, 0);
  auto plan1 = makePlanWithRealMain(context, 1);
  llvm::SmallVector<mlir::ModuleOp> modules = {*plan0, *plan1};

  std::string diagnostic;
  auto artifacts = mlir::ascend::kernel::runAggregateLaunchDriver(modules, "Fused", &diagnostic);
  ASSERT_TRUE(succeeded(artifacts)) << diagnostic;
  EXPECT_NE(artifacts->kernel_body.find("plan0_main"), std::string::npos);
  EXPECT_NE(artifacts->kernel_body.find("plan1_main"), std::string::npos);
  EXPECT_NE(artifacts->tiling_data.find("graph0_result0_g0_tiling_data"), std::string::npos);
  EXPECT_NE(artifacts->tiling_data.find("graph1_result0_g0_tiling_data"), std::string::npos);
}

TEST(AggregateLaunchDriverTest, FailsWhenPlanContractIsInvalid) {
  mlir::MLIRContext context;
  loadDriverDialects(context);
  mlir::OpBuilder builder(&context);
  // A bare module with none of the required AFIR/AscendC plan metadata
  // attributes: GenAggregateTilingData's contract validation rejects it, so
  // the failure surfaces before kernel-entry/kernel-body generation runs.
  mlir::ModuleOp bareModule = mlir::ModuleOp::create(builder.getUnknownLoc(), "bad_plan");
  llvm::SmallVector<mlir::ModuleOp> modules = {bareModule};

  std::string diagnostic;
  auto artifacts = mlir::ascend::kernel::runAggregateLaunchDriver(modules, "Fused", &diagnostic);
  EXPECT_TRUE(failed(artifacts));
  EXPECT_EQ(diagnostic, "aggregate tiling data generation failed");
}

TEST(AggregateLaunchDriverTest, FailsWhenFusedGraphNameSanitizesToEmpty) {
  mlir::MLIRContext context;
  loadDriverDialects(context);
  auto plan0 = makePlanWithRealMain(context, 0);
  llvm::SmallVector<mlir::ModuleOp> modules = {*plan0};

  // A name made entirely of characters genValidName() treats as separators
  // collapses to the empty string, which GenAggregateKernelBody's
  // promoteAggregateMain rejects.
  std::string diagnostic;
  auto artifacts = mlir::ascend::kernel::runAggregateLaunchDriver(modules, "!!!", &diagnostic);
  EXPECT_TRUE(failed(artifacts));
  EXPECT_EQ(diagnostic, "aggregate kernel body generation failed");
}
