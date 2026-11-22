/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "conversion/AFIRToAscendC/AFIRToAscendCPasses.h"
#include "artifact/KernelSource/GenAggregateKernelBody.h"
#include "artifact/KernelSource/GenAggregateKernelEntry.h"
#include "artifact/KernelSource/GenAggregateTilingData.h"
#include "conversion/AFIRToAscendC/AscendCKernelABI.h"
#include "artifact/KernelSource/Emit/TilingDataHeaderEmitter.h"
#include "support/CxxIdentifier.h"

#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFPlan/IR/AFPlan.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "common/tiling_schema/schedule_group_names.h"
#include "llvm/ADT/SmallVector.h"

#include <gtest/gtest.h>

namespace {
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

// 两个 stage 的 Plan（RCore 的形状）：载体带两个 stage 结构体，字段名就是它们在最外层
// TilingData 里的成员名。
mlir::OwningOpRef<mlir::ModuleOp> makeTwoStagePlan(mlir::MLIRContext &context) {
  context.getOrLoadDialect<mlir::afplan::AFPlanDialect>();
  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc(), "plan0");
  module->setAttr("afir.module_kind", builder.getStringAttr("plan"));
  module->setAttr(mlir::afplan::kCandidateAttrName, mlir::afplan::CandidateAttr::get(&context, 0, 0));
  module->setAttr(mlir::afplan::kStagesAttrName, mlir::afplan::StagesAttr::get(&context, {1, 1}));
  module->setAttr("afir.workspace_tensor_ids", builder.getDenseI64ArrayAttr({2}));
  llvm::SmallVector<mlir::Attribute> stageTypes;
  llvm::SmallVector<mlir::Attribute> stageNames;
  for (int64_t stage = 0; stage < 2; ++stage) {
    // `workspace2` 不是凑数：生产路径上 `GenTilingData` 的 `collectStageFields` 会把模块上
    // 声明的每个 workspace 号都加进每个 stage 的结构体，聚合入口正是往这一份里写偏移。
    auto stageType = mlir::emitasc::PyStructType::get(
        &context, builder.getStringAttr(mlir::common::MakeScheduleGroupTilingTypeName(0, 0, stage)),
        builder.getArrayAttr({mlir::TypeAttr::get(mlir::IntegerType::get(&context, 64, mlir::IntegerType::Unsigned)),
                              mlir::TypeAttr::get(builder.getI32Type())}),
        builder.getArrayAttr({builder.getStringAttr("workspace2"), builder.getStringAttr("tiling_key")}));
    stageTypes.push_back(mlir::TypeAttr::get(stageType));
    stageNames.push_back(builder.getStringAttr(mlir::common::MakeScheduleGroupTilingFieldName(0, 0, stage)));
  }
  auto schema = mlir::emitasc::PyStructType::get(&context, builder.getStringAttr("PlanData"),
                                                 builder.getArrayAttr(stageTypes), builder.getArrayAttr(stageNames));
  module->setAttr("ascendc.tiling_schema", mlir::TypeAttr::get(schema));
  return module;
}

mlir::OwningOpRef<mlir::ModuleOp> makePlanWithTemplates(mlir::MLIRContext &context, int64_t id) {
  context.getOrLoadDialect<mlir::afplan::AFPlanDialect>();
  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc(), "plan" + std::to_string(id));
  module->setAttr("afir.module_kind", builder.getStringAttr("plan"));
  module->setAttr(mlir::afplan::kCandidateAttrName, mlir::afplan::CandidateAttr::get(&context, id, 0));
  module->setAttr(mlir::afplan::kStagesAttrName, mlir::afplan::StagesAttr::get(&context, {2}));

  auto sizeVar = mlir::afir::ExprAttr::get(&context, builder.getStringAttr("tile_size"));
  builder.setInsertionPointToEnd(module.getBody());
  for (int64_t templateId = 0; templateId < 2; ++templateId) {
    auto graph = mlir::afir::AscGraphAttrGroupsAttr::get(&context, 10 + templateId * 10, {},
                                                         mlir::afir::AscGraphType::Compute, {sizeVar});
    auto function =
        builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "stage0_template" + std::to_string(templateId),
                                           mlir::FunctionType::get(&context, {builder.getF16Type()}, {}));
    function.setPrivate();
    function->setAttr("afir.asc_graph", graph);
    function->setAttr("ascendc.num_inputs", builder.getI32IntegerAttr(1));
    function->setAttr(mlir::afplan::kVariantAttrName, mlir::afplan::VariantAttr::get(&context, 0, templateId));
    function.addEntryBlock();
    mlir::OpBuilder functionBuilder = mlir::OpBuilder::atBlockEnd(&function.front());
    functionBuilder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
  }
  return module;
}

mlir::OwningOpRef<mlir::ModuleOp> makePlanWithDifferentTemplateSignatures(mlir::MLIRContext &context) {
  mlir::OpBuilder builder(&context);
  auto module = makePlanWithTemplates(context, 0);
  module->getOperation()->setAttr(mlir::afplan::kStagesAttrName, mlir::afplan::StagesAttr::get(&context, {2}));
  module->getOperation()->setAttr("afir.workspace_tensor_ids", builder.getDenseI64ArrayAttr({2}));
  auto stage = mlir::emitasc::PyStructType::get(
      &context, builder.getStringAttr("StageData"),
      builder.getArrayAttr({mlir::TypeAttr::get(builder.getI32Type()),
                            mlir::TypeAttr::get(mlir::IntegerType::get(&context, 64, mlir::IntegerType::Unsigned))}),
      builder.getArrayAttr({builder.getStringAttr("tiling_key"), builder.getStringAttr("workspace2")}));
  auto schema = mlir::emitasc::PyStructType::get(
      &context, builder.getStringAttr("PlanData"), builder.getArrayAttr({mlir::TypeAttr::get(stage)}),
      builder.getArrayAttr({builder.getStringAttr(mlir::common::MakeScheduleGroupTilingFieldName(0, 0, 0))}));
  (*module)->setAttr("ascendc.tiling_schema", mlir::TypeAttr::get(schema));
  auto templates = module->getOps<mlir::func::FuncOp>();
  auto first = templates.begin();
  (void)(*first).insertArgument(1, stage, mlir::DictionaryAttr::get(&context), (*first).getLoc());
  (*first)->setAttr("ascendc.num_inputs", builder.getI32IntegerAttr(1));
  mlir::afir::setKernelArgAttr(*first, 0, mlir::afir::KernelArgRole::Input, 0);
  ++first;
  (void)(*first).insertArgument(1, builder.getF16Type(), mlir::DictionaryAttr::get(&context), (*first).getLoc());
  (void)(*first).insertArgument(2, stage, mlir::DictionaryAttr::get(&context), (*first).getLoc());
  (*first)->setAttr("ascendc.num_inputs", builder.getI32IntegerAttr(2));
  mlir::afir::setKernelArgAttr(*first, 0, mlir::afir::KernelArgRole::Input, 0);
  mlir::afir::setKernelArgAttr(*first, 1, mlir::afir::KernelArgRole::Input, 1);
  return module;
}
// Plan 的 `_main` 一 stage 收一个 tiling 形参、序号即 stage：ATT 的最外层 TilingData 里没有
// plan 那一层，各 schedule group 的结构体平铺在根上。下面的夹具都按这个 ABI 建 `_main`。
mlir::Type stageTilingType(mlir::TypeAttr planSchema, size_t stage = 0) {
  auto schema = mlir::cast<mlir::emitasc::PyStructType>(planSchema.getValue());
  return mlir::cast<mlir::TypeAttr>(schema.getTypesAttr()[stage]).getValue();
}

void markPlanMainTilingArguments(mlir::func::FuncOp main, size_t firstTilingArgument, size_t stageCount) {
  for (size_t stage = 0; stage < stageCount; ++stage) {
    mlir::afir::setKernelArgAttr(main, firstTilingArgument + stage, mlir::afir::KernelArgRole::Tiling,
                                 static_cast<int64_t>(stage));
  }
}
}  // namespace

TEST(IsolatedPlanEntryTest, CentralizesPyStructTypeValidation) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::emitasc::EmitAscDialect>();
  mlir::OpBuilder builder(&context);
  auto schema = mlir::emitasc::PyStructType::get(&context, builder.getStringAttr("TilingData"),
                                                 builder.getArrayAttr({mlir::TypeAttr::get(builder.getI32Type())}),
                                                 builder.getArrayAttr({builder.getStringAttr("tiling_key")}));

  EXPECT_TRUE(mlir::ascend::kernel::getTilingSchema(schema));
  EXPECT_FALSE(mlir::ascend::kernel::getTilingSchema(builder.getI32Type()));
  EXPECT_EQ(mlir::ascend::kernel::getTilingSchemaField(schema, "tiling_key"), builder.getI32Type());
  EXPECT_FALSE(mlir::ascend::kernel::getTilingSchemaField(schema, "missing"));

  // 结构体叫什么名字不参与判据。`TilingDataType` 曾经在这里额外要求名字等于 "TilingData"，
  // 而 `GenTilingData` 马上会把它改成 `<module>TilingData`——那条名字匹配是整条流水线的一条
  // 隐式排序约束，不是类型校验。参数身份由 `#afir.kernel_arg<role = tiling>` 回答。
  auto otherSchema = mlir::emitasc::PyStructType::get(&context, builder.getStringAttr("Stage0TilingData"),
                                                      builder.getArrayAttr({mlir::TypeAttr::get(builder.getI32Type())}),
                                                      builder.getArrayAttr({builder.getStringAttr("tiling_key")}));
  EXPECT_TRUE(mlir::ascend::kernel::getTilingSchema(otherSchema));
  EXPECT_EQ(mlir::ascend::kernel::getTilingSchemaField(otherSchema, "tiling_key"), builder.getI32Type());
}

TEST(IsolatedPlanEntryTest, AggregatesSchemasAndEmitsOuterMain) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::afir::AFIRDialect>();
  context.getOrLoadDialect<mlir::afplan::AFPlanDialect>();
  context.getOrLoadDialect<mlir::emitasc::EmitAscDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::emitc::EmitCDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  auto plan0 = makePlan(context, 0);
  auto plan1 = makePlan(context, 1);
  llvm::SmallVector<mlir::ModuleOp> modules = {*plan0, *plan1};
  auto entry = mlir::ascend::kernel::runGenAggregateTilingData(modules);
  ASSERT_TRUE(succeeded(entry));
  mlir::ModuleOp entryModule = **entry;
  auto schema = entryModule->getAttrOfType<mlir::TypeAttr>("ascendc.tiling_schema");
  ASSERT_TRUE(schema);
  auto aggregateSchema = mlir::dyn_cast<mlir::emitasc::PyStructType>(schema.getValue());
  ASSERT_TRUE(aggregateSchema);
  // 两个子图各一个 plan，不是扁平形态：根上自带 ATT 要读写的四个公共标量，再是每个子图的
  // plan 选择子，最后每个 schedule group 的结构体平铺在根上（没有 plan 那一层）。
  ASSERT_EQ(aggregateSchema.getNamesAttr().size(), 8U);
  EXPECT_EQ(mlir::cast<mlir::StringAttr>(aggregateSchema.getNamesAttr()[0]).getValue(), "block_dim");
  EXPECT_EQ(mlir::cast<mlir::StringAttr>(aggregateSchema.getNamesAttr()[1]).getValue(), "corenum");
  EXPECT_EQ(mlir::cast<mlir::StringAttr>(aggregateSchema.getNamesAttr()[2]).getValue(), "ub_size");
  EXPECT_EQ(mlir::cast<mlir::StringAttr>(aggregateSchema.getNamesAttr()[3]).getValue(), "hbm_size");
  EXPECT_EQ(mlir::cast<mlir::StringAttr>(aggregateSchema.getNamesAttr()[4]).getValue(), "graph0_tiling_key");
  EXPECT_EQ(mlir::cast<mlir::StringAttr>(aggregateSchema.getNamesAttr()[5]).getValue(), "graph1_tiling_key");
  EXPECT_EQ(mlir::cast<mlir::StringAttr>(aggregateSchema.getNamesAttr()[6]).getValue(),
            "graph0_result0_g0_tiling_data");
  EXPECT_EQ(mlir::cast<mlir::StringAttr>(aggregateSchema.getNamesAttr()[7]).getValue(),
            "graph1_result0_g0_tiling_data");
  auto contracts = entryModule->getAttrOfType<mlir::ArrayAttr>("ascendc.plan_entries");
  ASSERT_TRUE(contracts);
  EXPECT_EQ(contracts.size(), 2U);
  EXPECT_FALSE(entryModule.lookupSymbol<mlir::func::FuncOp>("main"));
}

TEST(IsolatedPlanEntryTest, AllowsDifferentTemplateArgumentCounts) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::afir::AFIRDialect>();
  context.getOrLoadDialect<mlir::afplan::AFPlanDialect>();
  context.getOrLoadDialect<mlir::emitasc::EmitAscDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  auto plan = makePlanWithDifferentTemplateSignatures(context);
  mlir::PassManager kernelEntryPassManager(&context);
  kernelEntryPassManager.addPass(mlir::ascend::kernel::createGenKernelEntryPass());
  ASSERT_TRUE(succeeded(kernelEntryPassManager.run(plan->getOperation())));
  auto main = plan->lookupSymbol<mlir::func::FuncOp>("plan0_main");
  ASSERT_TRUE(main);
  EXPECT_EQ(main.getNumArguments(), 3U);
  EXPECT_EQ(main.getFunctionType().getInput(0), mlir::Float16Type::get(&context));
  EXPECT_EQ(main.getFunctionType().getInput(1), mlir::Float16Type::get(&context));
}

TEST(IsolatedPlanEntryTest, EmitsOuterDispatchFromAggregateContract) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::afir::AFIRDialect>();
  context.getOrLoadDialect<mlir::emitasc::EmitAscDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::emitc::EmitCDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  auto plan0 = makePlan(context, 0);
  auto plan1 = makePlan(context, 1);
  llvm::SmallVector<mlir::ModuleOp> modules = {*plan0, *plan1};
  auto entry = mlir::ascend::kernel::runGenAggregateTilingData(modules);
  ASSERT_TRUE(succeeded(entry));
  mlir::ModuleOp entryModule = **entry;
  for (mlir::ModuleOp plan : modules) {
    auto schema = plan->getAttrOfType<mlir::TypeAttr>("ascendc.tiling_schema");
    ASSERT_TRUE(schema);
    mlir::OpBuilder builder(plan.getBodyRegion());
    builder.setInsertionPointToEnd(plan.getBody());
    auto declaration =
        builder.create<mlir::func::FuncOp>(plan.getLoc(), plan.getSymName()->str() + "_main",
                                           mlir::FunctionType::get(&context, {stageTilingType(schema)}, {}));
    markPlanMainTilingArguments(declaration, 0, 1);
    declaration.setPrivate();
  }
  ASSERT_TRUE(succeeded(mlir::ascend::kernel::runGenAggregateKernelEntry(modules, entryModule)));

  auto main = entryModule.lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main);
  EXPECT_TRUE(main->hasAttr("ascendc.fused_main"));
  EXPECT_TRUE(entryModule.lookupSymbol<mlir::func::FuncOp>("plan0_main"));
  EXPECT_TRUE(entryModule.lookupSymbol<mlir::func::FuncOp>("plan1_main"));
  size_t switchCount = 0;
  size_t callCount = 0;
  main.walk([&](mlir::scf::IndexSwitchOp) { ++switchCount; });
  main.walk([&](mlir::func::CallOp) { ++callCount; });
  EXPECT_EQ(switchCount, 2U);
  EXPECT_EQ(callCount, 2U);
}

TEST(IsolatedPlanEntryTest, EmitsSingleSwitchWithMultipleCasesForSameSubgraphCandidates) {
  // SelectDynamicCandidates can return several PlanSelections for
  // the SAME subgraph_id (different plan_id). The aggregate entry must
  // dispatch between them with one scf.index_switch per subgraph, not one
  // per candidate -- otherwise the runtime tiling_key selector for that
  // subgraph would have nothing to switch on.
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::afir::AFIRDialect>();
  context.getOrLoadDialect<mlir::afplan::AFPlanDialect>();
  context.getOrLoadDialect<mlir::emitasc::EmitAscDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::emitc::EmitCDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  auto plan0 = makePlan(context, 0);
  auto plan1 = makePlan(context, 1);
  // 两个候选落在同一个 subgraph 上，plan 编号不同。
  int64_t sharedSubgraph = mlir::afplan::getCandidate(*plan0).getSubgraph();
  (*plan1)->setAttr(mlir::afplan::kCandidateAttrName, mlir::afplan::CandidateAttr::get(&context, sharedSubgraph, 1));
  llvm::SmallVector<mlir::ModuleOp> modules = {*plan0, *plan1};
  auto entry = mlir::ascend::kernel::runGenAggregateTilingData(modules);
  ASSERT_TRUE(succeeded(entry));
  mlir::ModuleOp entryModule = **entry;
  for (mlir::ModuleOp plan : modules) {
    auto schema = plan->getAttrOfType<mlir::TypeAttr>("ascendc.tiling_schema");
    ASSERT_TRUE(schema);
    mlir::OpBuilder builder(plan.getBodyRegion());
    builder.setInsertionPointToEnd(plan.getBody());
    auto declaration =
        builder.create<mlir::func::FuncOp>(plan.getLoc(), plan.getSymName()->str() + "_main",
                                           mlir::FunctionType::get(&context, {stageTilingType(schema)}, {}));
    markPlanMainTilingArguments(declaration, 0, 1);
    declaration.setPrivate();
  }
  ASSERT_TRUE(succeeded(mlir::ascend::kernel::runGenAggregateKernelEntry(modules, entryModule)));

  auto main = entryModule.lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main);
  size_t switchCount = 0;
  size_t callCount = 0;
  main.walk([&](mlir::scf::IndexSwitchOp switchOp) {
    ++switchCount;
    EXPECT_EQ(switchOp.getCaseRegions().size(), 2U);
  });
  main.walk([&](mlir::func::CallOp) { ++callCount; });
  EXPECT_EQ(switchCount, 1U);
  EXPECT_EQ(callCount, 2U);

  // 缺省分支必须陷入而不是空转。`graph<N>_tiling_key` 由 host 侧 tiling 写，落不进已下降的
  // 候选编号就说明 host 与 kernel 对候选集的认知不一致。空 `scf.yield` 会让 kernel 什么都不做
  // 直接返回、输出保持未初始化——数值错且没有任何报错，是这条链路上唯一会静默出错的地方。
  size_t trapCount = 0;
  main.walk([&](mlir::scf::IndexSwitchOp switchOp) {
    switchOp.getDefaultRegion().walk([&](mlir::ascendc::TrapOp) { ++trapCount; });
  });
  EXPECT_EQ(trapCount, 1U);
}

// 同身份 GM 参数的 rank-0 与 1×? 视图可合并为同一个 ABI 参数，并在调用点恢复。
TEST(IsolatedPlanEntryTest, MergesSameArgumentAcrossPlansWithDifferentScheduledViews) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::afir::AFIRDialect>();
  context.getOrLoadDialect<mlir::afplan::AFPlanDialect>();
  context.getOrLoadDialect<mlir::emitasc::EmitAscDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::emitc::EmitCDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  auto plan0 = makePlan(context, 0);
  auto plan1 = makePlan(context, 1);
  const int64_t sharedSubgraph = mlir::afplan::getCandidate(*plan0).getSubgraph();
  (*plan1)->setAttr(mlir::afplan::kCandidateAttrName, mlir::afplan::CandidateAttr::get(&context, sharedSubgraph, 1));
  llvm::SmallVector<mlir::ModuleOp> modules = {*plan0, *plan1};
  auto entry = mlir::ascend::kernel::runGenAggregateTilingData(modules);
  ASSERT_TRUE(succeeded(entry));
  mlir::ModuleOp entryModule = **entry;

  const mlir::Attribute gm = mlir::ascend::kernel::getAscendCGlobalMemorySpaceAttr(&context);
  // 秩 0 与秩 2，正是实机上 plan 0 与 plan 1 的 `output:0`。
  const llvm::SmallVector<llvm::SmallVector<int64_t>> shapes = {{}, {1, mlir::ShapedType::kDynamic}};
  for (auto [index, plan] : llvm::enumerate(modules)) {
    auto schema = plan->getAttrOfType<mlir::TypeAttr>("ascendc.tiling_schema");
    ASSERT_TRUE(schema);
    auto output =
        mlir::MemRefType::get(shapes[index], mlir::Float32Type::get(&context), mlir::MemRefLayoutAttrInterface{}, gm);
    mlir::OpBuilder builder(plan.getBodyRegion());
    builder.setInsertionPointToEnd(plan.getBody());
    auto declaration =
        builder.create<mlir::func::FuncOp>(plan.getLoc(), plan.getSymName()->str() + "_main",
                                           mlir::FunctionType::get(&context, {output, stageTilingType(schema)}, {}));
    mlir::afir::setKernelArgAttr(declaration, 0, mlir::afir::KernelArgRole::Output, 0);
    markPlanMainTilingArguments(declaration, 1, 1);
    declaration.setPrivate();
  }

  auto layout = mlir::ascend::kernel::runGenAggregateKernelEntry(modules, entryModule);
  ASSERT_TRUE(succeeded(layout));
  auto main = entryModule.lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main);
  // 合并后的类型保留元素类型与地址空间，但不再声称任何一个 Plan 的秩与形状。
  const auto unifiedType = mlir::UnrankedMemRefType::get(mlir::Float32Type::get(&context), gm);
  EXPECT_EQ(main.getFunctionType().getInput(0), mlir::Type(unifiedType));
  // 调用点与被调声明的类型必须一致，否则 `func.call` 的验证器会拒；两个 Plan 各拿到一次 cast。
  size_t castCount = 0;
  main.walk([&](mlir::memref::CastOp) { ++castCount; });
  EXPECT_EQ(castCount, 2U);
  EXPECT_TRUE(succeeded(mlir::verify(entryModule)));
}

// 单候选也使用 unranked ABI，确保入口类型不依赖候选集。
TEST(IsolatedPlanEntryTest, UsesTheAbiArgumentTypeForASinglePlanToo) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::afir::AFIRDialect>();
  context.getOrLoadDialect<mlir::afplan::AFPlanDialect>();
  context.getOrLoadDialect<mlir::emitasc::EmitAscDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::emitc::EmitCDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  auto plan0 = makePlan(context, 0);
  llvm::SmallVector<mlir::ModuleOp> modules = {*plan0};
  auto entry = mlir::ascend::kernel::runGenAggregateTilingData(modules);
  ASSERT_TRUE(succeeded(entry));
  mlir::ModuleOp entryModule = **entry;

  const mlir::Attribute gm = mlir::ascend::kernel::getAscendCGlobalMemorySpaceAttr(&context);
  auto schema = (*plan0)->getAttrOfType<mlir::TypeAttr>("ascendc.tiling_schema");
  ASSERT_TRUE(schema);
  auto output = mlir::MemRefType::get({1, mlir::ShapedType::kDynamic}, mlir::Float32Type::get(&context),
                                      mlir::MemRefLayoutAttrInterface{}, gm);
  mlir::OpBuilder builder(plan0->getBodyRegion());
  builder.setInsertionPointToEnd(plan0->getBody());
  auto declaration = builder.create<mlir::func::FuncOp>(
      plan0->getLoc(), "plan0_main", mlir::FunctionType::get(&context, {output, stageTilingType(schema)}, {}));
  mlir::afir::setKernelArgAttr(declaration, 0, mlir::afir::KernelArgRole::Output, 0);
  markPlanMainTilingArguments(declaration, 1, 1);
  declaration.setPrivate();

  ASSERT_TRUE(succeeded(mlir::ascend::kernel::runGenAggregateKernelEntry(modules, entryModule)));
  auto main = entryModule.lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main);
  EXPECT_EQ(main.getFunctionType().getInput(0),
            mlir::Type(mlir::UnrankedMemRefType::get(mlir::Float32Type::get(&context), gm)));
  EXPECT_TRUE(succeeded(mlir::verify(entryModule)));
}

namespace {
void addRealBody(mlir::func::FuncOp function) {
  mlir::OpBuilder builder(function.getContext());
  mlir::Block *entry = function.addEntryBlock();
  builder.setInsertionPointToEnd(entry);
  builder.create<mlir::func::ReturnOp>(function.getLoc());
}
}  // namespace

TEST(IsolatedPlanEntryTest, ClonesPlanBodyIntoAggregateDeclaration) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::afir::AFIRDialect>();
  context.getOrLoadDialect<mlir::afplan::AFPlanDialect>();
  context.getOrLoadDialect<mlir::emitasc::EmitAscDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::emitc::EmitCDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  auto plan0 = makePlan(context, 0);
  llvm::SmallVector<mlir::ModuleOp> modules = {*plan0};
  auto entry = mlir::ascend::kernel::runGenAggregateTilingData(modules);
  ASSERT_TRUE(succeeded(entry));
  mlir::ModuleOp entryModule = **entry;

  auto schema = (*plan0)->getAttrOfType<mlir::TypeAttr>("ascendc.tiling_schema");
  ASSERT_TRUE(schema);
  mlir::OpBuilder builder(plan0->getBodyRegion());
  builder.setInsertionPointToEnd(plan0->getBody());
  auto realMain = builder.create<mlir::func::FuncOp>(plan0->getLoc(), "plan0_main",
                                                     mlir::FunctionType::get(&context, {stageTilingType(schema)}, {}));
  markPlanMainTilingArguments(realMain, 0, 1);
  addRealBody(realMain);

  ASSERT_TRUE(succeeded(mlir::ascend::kernel::runGenAggregateKernelEntry(modules, entryModule)));
  ASSERT_TRUE(succeeded(mlir::ascend::kernel::runGenAggregateKernelBody(modules, entryModule, "FusedGraph.0")));

  auto clonedMain = entryModule.lookupSymbol<mlir::func::FuncOp>("plan0_main");
  ASSERT_TRUE(clonedMain);
  EXPECT_FALSE(clonedMain.isExternal());
  EXPECT_TRUE(clonedMain.isPrivate());

  auto aggregateMain = entryModule.lookupSymbol<mlir::func::FuncOp>("fused_graph_0");
  ASSERT_TRUE(aggregateMain);
  EXPECT_TRUE(mlir::afir::hasFunctionRole(aggregateMain, mlir::afir::FunctionRole::KernelEntry));
}

TEST(IsolatedPlanEntryTest, ClonesEveryPlanBodyAcrossSubgraphs) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::afir::AFIRDialect>();
  context.getOrLoadDialect<mlir::afplan::AFPlanDialect>();
  context.getOrLoadDialect<mlir::emitasc::EmitAscDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::emitc::EmitCDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  auto plan0 = makePlan(context, 0);
  auto plan1 = makePlan(context, 1);
  llvm::SmallVector<mlir::ModuleOp> modules = {*plan0, *plan1};
  auto entry = mlir::ascend::kernel::runGenAggregateTilingData(modules);
  ASSERT_TRUE(succeeded(entry));
  mlir::ModuleOp entryModule = **entry;

  for (mlir::ModuleOp plan : modules) {
    auto schema = plan->getAttrOfType<mlir::TypeAttr>("ascendc.tiling_schema");
    ASSERT_TRUE(schema);
    mlir::OpBuilder builder(plan.getBodyRegion());
    builder.setInsertionPointToEnd(plan.getBody());
    auto realMain =
        builder.create<mlir::func::FuncOp>(plan.getLoc(), plan.getSymName()->str() + "_main",
                                           mlir::FunctionType::get(&context, {stageTilingType(schema)}, {}));
    markPlanMainTilingArguments(realMain, 0, 1);
    addRealBody(realMain);
  }

  ASSERT_TRUE(succeeded(mlir::ascend::kernel::runGenAggregateKernelEntry(modules, entryModule)));
  ASSERT_TRUE(succeeded(mlir::ascend::kernel::runGenAggregateKernelBody(modules, entryModule, "Fused")));

  auto main0 = entryModule.lookupSymbol<mlir::func::FuncOp>("plan0_main");
  auto main1 = entryModule.lookupSymbol<mlir::func::FuncOp>("plan1_main");
  ASSERT_TRUE(main0);
  ASSERT_TRUE(main1);
  EXPECT_FALSE(main0.isExternal());
  EXPECT_FALSE(main1.isExternal());
  EXPECT_TRUE(main0.isPrivate());
  EXPECT_TRUE(main1.isPrivate());

  auto aggregateMain = entryModule.lookupSymbol<mlir::func::FuncOp>("fused");
  ASSERT_TRUE(aggregateMain);
  EXPECT_TRUE(mlir::afir::hasFunctionRole(aggregateMain, mlir::afir::FunctionRole::KernelEntry));
}

TEST(IsolatedPlanEntryTest, PromotesAggregateMainWithAscendCNumInputs) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::afir::AFIRDialect>();
  context.getOrLoadDialect<mlir::afplan::AFPlanDialect>();
  context.getOrLoadDialect<mlir::emitasc::EmitAscDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::emitc::EmitCDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  auto plan0 = makePlan(context, 0);
  llvm::SmallVector<mlir::ModuleOp> modules = {*plan0};
  auto entry = mlir::ascend::kernel::runGenAggregateTilingData(modules);
  ASSERT_TRUE(succeeded(entry));
  mlir::ModuleOp entryModule = **entry;

  auto schema = (*plan0)->getAttrOfType<mlir::TypeAttr>("ascendc.tiling_schema");
  ASSERT_TRUE(schema);
  mlir::OpBuilder builder(plan0->getBodyRegion());
  builder.setInsertionPointToEnd(plan0->getBody());
  auto realMain = builder.create<mlir::func::FuncOp>(
      plan0->getLoc(), "plan0_main",
      mlir::FunctionType::get(&context, {builder.getF16Type(), stageTilingType(schema)}, {}));
  mlir::afir::setKernelArgAttr(realMain, 0, mlir::afir::KernelArgRole::Input, 0);
  markPlanMainTilingArguments(realMain, 1, 1);
  addRealBody(realMain);

  ASSERT_TRUE(succeeded(mlir::ascend::kernel::runGenAggregateKernelEntry(modules, entryModule)));
  ASSERT_TRUE(succeeded(mlir::ascend::kernel::runGenAggregateKernelBody(modules, entryModule, "Fused")));

  auto aggregateMain = entryModule.lookupSymbol<mlir::func::FuncOp>("fused");
  ASSERT_TRUE(aggregateMain);
  auto numInputs = aggregateMain->getAttrOfType<mlir::IntegerAttr>("ascendc.num_inputs");
  ASSERT_TRUE(numInputs);
  EXPECT_EQ(numInputs.getInt(), 1);
}

TEST(IsolatedPlanEntryTest, ReturnsAggregateArgumentLayoutMatchingMainArguments) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::afir::AFIRDialect>();
  context.getOrLoadDialect<mlir::afplan::AFPlanDialect>();
  context.getOrLoadDialect<mlir::emitasc::EmitAscDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::emitc::EmitCDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  auto plan0 = makePlan(context, 0);
  llvm::SmallVector<mlir::ModuleOp> modules = {*plan0};
  auto entry = mlir::ascend::kernel::runGenAggregateTilingData(modules);
  ASSERT_TRUE(succeeded(entry));
  mlir::ModuleOp entryModule = **entry;

  auto schema = (*plan0)->getAttrOfType<mlir::TypeAttr>("ascendc.tiling_schema");
  ASSERT_TRUE(schema);
  mlir::OpBuilder builder(plan0->getBodyRegion());
  builder.setInsertionPointToEnd(plan0->getBody());
  auto realMain = builder.create<mlir::func::FuncOp>(
      plan0->getLoc(), "plan0_main",
      mlir::FunctionType::get(&context, {builder.getF16Type(), stageTilingType(schema)}, {}));
  mlir::afir::setKernelArgAttr(realMain, 0, mlir::afir::KernelArgRole::Input, 0);
  markPlanMainTilingArguments(realMain, 1, 1);
  addRealBody(realMain);

  auto layout = mlir::ascend::kernel::runGenAggregateKernelEntry(modules, entryModule);
  ASSERT_TRUE(succeeded(layout));
  ASSERT_EQ(layout->arguments.size(), 1U);
  EXPECT_EQ(layout->arguments[0].kind, mlir::ascend::kernel::KernelArgumentKind::Input);
  EXPECT_EQ(layout->arguments[0].index, 0);
  EXPECT_EQ(layout->tiling_arg_index, 1U);

  auto aggregateMain = entryModule.lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(aggregateMain);
  EXPECT_EQ(aggregateMain.getNumArguments(), layout->tiling_arg_index + 1);
}

TEST(IsolatedPlanEntryTest, ValidateAggregateContractAcceptsMatchingPlanCount) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::emitasc::EmitAscDialect>();
  auto plan0 = makePlan(context, 0);
  auto plan1 = makePlan(context, 1);
  llvm::SmallVector<mlir::ModuleOp> modules = {*plan0, *plan1};
  auto entry = mlir::ascend::kernel::runGenAggregateTilingData(modules);
  ASSERT_TRUE(succeeded(entry));
  mlir::ModuleOp entryModule = **entry;

  EXPECT_TRUE(succeeded(mlir::ascend::kernel::validateAggregateContract(modules, entryModule)));
}

TEST(IsolatedPlanEntryTest, ValidateAggregateContractRejectsPlanCountMismatch) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::emitasc::EmitAscDialect>();
  auto plan0 = makePlan(context, 0);
  auto plan1 = makePlan(context, 1);
  llvm::SmallVector<mlir::ModuleOp> modules = {*plan0, *plan1};
  auto entry = mlir::ascend::kernel::runGenAggregateTilingData(modules);
  ASSERT_TRUE(succeeded(entry));
  mlir::ModuleOp entryModule = **entry;

  llvm::SmallVector<mlir::ModuleOp> mismatchedModules = {*plan0};
  EXPECT_TRUE(failed(mlir::ascend::kernel::validateAggregateContract(mismatchedModules, entryModule)));
}

TEST(IsolatedPlanEntryTest, ValidateAggregateContractRejectsNonEntryModule) {
  mlir::MLIRContext context;
  mlir::OpBuilder builder(&context);
  auto notEntry = mlir::ModuleOp::create(builder.getUnknownLoc());
  llvm::SmallVector<mlir::ModuleOp> modules;
  EXPECT_TRUE(failed(mlir::ascend::kernel::validateAggregateContract(modules, notEntry)));
  EXPECT_TRUE(failed(mlir::ascend::kernel::validateAggregateContract(modules, mlir::ModuleOp())));
}

TEST(IsolatedPlanEntryTest, RejectsAggregateContractMismatchedToPlanModules) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::afir::AFIRDialect>();
  context.getOrLoadDialect<mlir::afplan::AFPlanDialect>();
  context.getOrLoadDialect<mlir::emitasc::EmitAscDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::emitc::EmitCDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  auto plan0 = makePlan(context, 0);
  llvm::SmallVector<mlir::ModuleOp> modules = {*plan0};
  auto entry = mlir::ascend::kernel::runGenAggregateTilingData(modules);
  ASSERT_TRUE(succeeded(entry));
  mlir::ModuleOp entryModule = **entry;

  auto schema = (*plan0)->getAttrOfType<mlir::TypeAttr>("ascendc.tiling_schema");
  ASSERT_TRUE(schema);
  mlir::OpBuilder builder(plan0->getBodyRegion());
  builder.setInsertionPointToEnd(plan0->getBody());
  auto declaration = builder.create<mlir::func::FuncOp>(
      plan0->getLoc(), "plan0_main", mlir::FunctionType::get(&context, {stageTilingType(schema)}, {}));
  markPlanMainTilingArguments(declaration, 0, 1);
  declaration.setPrivate();
  ASSERT_TRUE(succeeded(mlir::ascend::kernel::runGenAggregateKernelEntry(modules, entryModule)));

  // Drop plan0 from the module list handed to runGenAggregateKernelBody so its
  // ascendc.plan_entries coordinates no longer resolve to any Plan module.
  llvm::SmallVector<mlir::ModuleOp> emptyPlanModules;
  EXPECT_TRUE(failed(mlir::ascend::kernel::runGenAggregateKernelBody(emptyPlanModules, entryModule, "Fused")));
}

namespace {
// A "_main" that calls a template function living in the same Plan module,
// mirroring GenKernelEntry's buildStage/buildSwitch output (scf.index_switch +
// func::CallOp to sibling "stage*_template*" functions). Every plan gets a
// template function with the SAME name ("shared_template") to exercise the
// cross-Plan symbol collision that a naive single-level clone would hit.
mlir::func::FuncOp addTemplateFunction(mlir::ModuleOp plan) {
  mlir::OpBuilder builder(plan.getBodyRegion());
  builder.setInsertionPointToEnd(plan.getBody());
  auto templateFunc = builder.create<mlir::func::FuncOp>(plan.getLoc(), "shared_template",
                                                         mlir::FunctionType::get(plan.getContext(), {}, {}));
  templateFunc.setPrivate();
  mlir::Block *entry = templateFunc.addEntryBlock();
  mlir::OpBuilder::atBlockEnd(entry).create<mlir::func::ReturnOp>(templateFunc.getLoc());
  return templateFunc;
}

void addRealBodyCallingTemplate(mlir::func::FuncOp function, mlir::func::FuncOp templateFunc) {
  mlir::OpBuilder builder(function.getContext());
  mlir::Block *entry = function.addEntryBlock();
  builder.setInsertionPointToEnd(entry);
  builder.create<mlir::func::CallOp>(function.getLoc(), templateFunc, mlir::ValueRange{});
  builder.create<mlir::func::ReturnOp>(function.getLoc());
}
}  // namespace

TEST(IsolatedPlanEntryTest, RelocatesSameNamedTemplateFunctionsAcrossSubgraphs) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::afir::AFIRDialect>();
  context.getOrLoadDialect<mlir::afplan::AFPlanDialect>();
  context.getOrLoadDialect<mlir::emitasc::EmitAscDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::emitc::EmitCDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  auto plan0 = makePlan(context, 0);
  auto plan1 = makePlan(context, 1);
  llvm::SmallVector<mlir::ModuleOp> modules = {*plan0, *plan1};
  auto entry = mlir::ascend::kernel::runGenAggregateTilingData(modules);
  ASSERT_TRUE(succeeded(entry));
  mlir::ModuleOp entryModule = **entry;

  for (mlir::ModuleOp plan : modules) {
    auto schema = plan->getAttrOfType<mlir::TypeAttr>("ascendc.tiling_schema");
    ASSERT_TRUE(schema);
    auto templateFunc = addTemplateFunction(plan);
    mlir::OpBuilder builder(plan.getBodyRegion());
    builder.setInsertionPointToEnd(plan.getBody());
    auto realMain =
        builder.create<mlir::func::FuncOp>(plan.getLoc(), plan.getSymName()->str() + "_main",
                                           mlir::FunctionType::get(&context, {stageTilingType(schema)}, {}));
    markPlanMainTilingArguments(realMain, 0, 1);
    addRealBodyCallingTemplate(realMain, templateFunc);
  }

  ASSERT_TRUE(succeeded(mlir::ascend::kernel::runGenAggregateKernelEntry(modules, entryModule)));
  ASSERT_TRUE(succeeded(mlir::ascend::kernel::runGenAggregateKernelBody(modules, entryModule, "Fused")));

  auto relocated0 = entryModule.lookupSymbol<mlir::func::FuncOp>("plan0_shared_template");
  auto relocated1 = entryModule.lookupSymbol<mlir::func::FuncOp>("plan1_shared_template");
  ASSERT_TRUE(relocated0);
  ASSERT_TRUE(relocated1);
  EXPECT_NE(relocated0, relocated1);

  auto main0 = entryModule.lookupSymbol<mlir::func::FuncOp>("plan0_main");
  auto main1 = entryModule.lookupSymbol<mlir::func::FuncOp>("plan1_main");
  ASSERT_TRUE(main0);
  ASSERT_TRUE(main1);
  auto calleeOf = [](mlir::func::FuncOp main) {
    mlir::func::CallOp call;
    main.walk([&](mlir::func::CallOp op) { call = op; });
    return call ? call.getCallee() : llvm::StringRef();
  };
  EXPECT_EQ(calleeOf(main0), "plan0_shared_template");
  EXPECT_EQ(calleeOf(main1), "plan1_shared_template");

  auto functionPosition = [&](llvm::StringRef name) {
    int64_t position = 0;
    for (mlir::func::FuncOp function : entryModule.getOps<mlir::func::FuncOp>()) {
      if (function.getName() == name) {
        return position;
      }
      ++position;
    }
    return int64_t{-1};
  };
  int64_t relocated0Position = functionPosition("plan0_shared_template");
  int64_t relocated1Position = functionPosition("plan1_shared_template");
  int64_t main0Position = functionPosition("plan0_main");
  int64_t main1Position = functionPosition("plan1_main");
  ASSERT_NE(relocated0Position, -1);
  ASSERT_NE(relocated1Position, -1);
  ASSERT_NE(main0Position, -1);
  ASSERT_NE(main1Position, -1);
  EXPECT_LT(relocated0Position, main0Position);
  EXPECT_LT(relocated1Position, main1Position);
  EXPECT_TRUE(succeeded(mlir::verify(entryModule)));
}

// 单子图 / 单 plan / 单 stage 是扁平形态：根上不自带公共标量，那些名字是转发到唯一 stage
// 结构体的访问器。这条正是转发这一半，`DoesNotForwardRootAccessorsForMultiplePlans` 是另一半。
TEST(IsolatedPlanEntryTest, EmitsNestedTilingDataHeader) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::emitasc::EmitAscDialect>();
  auto plan0 = makePlan(context, 0);
  llvm::SmallVector<mlir::ModuleOp> modules = {*plan0};
  auto entry = mlir::ascend::kernel::runGenAggregateTilingData(modules);
  ASSERT_TRUE(succeeded(entry));
  auto schema = (**entry)->getAttrOfType<mlir::TypeAttr>("ascendc.tiling_schema");
  ASSERT_TRUE(schema);
  std::string header;
  llvm::raw_string_ostream output(header);
  auto aggregateSchema = mlir::dyn_cast<mlir::emitasc::PyStructType>(schema.getValue());
  ASSERT_TRUE(aggregateSchema);
  llvm::SmallVector<mlir::ascend::kernel::TilingSchema> schemas = {
      mlir::ascend::kernel::getTilingSchema(aggregateSchema)};
  ASSERT_TRUE(llvm::succeeded(mlir::ascend::kernel::emitTilingDataHeader(schemas, output)));
  EXPECT_NE(header.find("AutofuseTilingData"), std::string::npos);
  EXPECT_NE(header.find("graph0_result0_g0_tiling_data"), std::string::npos);
  EXPECT_NE(header.find("struct plan0_StageData"), std::string::npos);
  EXPECT_NE(header.find("graph0_result0_g0_tiling_data.set_tiling_key(value)"), std::string::npos);
  // plan 级 schema 只是载体，不是根的字段，所以不会生成结构体。
  EXPECT_EQ(header.find("struct PlanData"), std::string::npos);
}

TEST(IsolatedPlanEntryTest, DoesNotForwardRootAccessorsForMultiplePlans) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::emitasc::EmitAscDialect>();
  auto plan0 = makePlan(context, 0);
  auto plan1 = makePlan(context, 1);
  llvm::SmallVector<mlir::ModuleOp> modules = {*plan0, *plan1};
  auto entry = mlir::ascend::kernel::runGenAggregateTilingData(modules);
  ASSERT_TRUE(succeeded(entry));
  auto schema = (**entry)->getAttrOfType<mlir::TypeAttr>("ascendc.tiling_schema");
  ASSERT_TRUE(schema);
  auto aggregateSchema = mlir::dyn_cast<mlir::emitasc::PyStructType>(schema.getValue());
  ASSERT_TRUE(aggregateSchema);
  std::string header;
  llvm::raw_string_ostream output(header);
  llvm::SmallVector<mlir::ascend::kernel::TilingSchema> schemas = {
      mlir::ascend::kernel::getTilingSchema(aggregateSchema)};
  ASSERT_TRUE(llvm::succeeded(mlir::ascend::kernel::emitTilingDataHeader(schemas, output)));
  EXPECT_EQ(header.find("graph0_result0_g0_tiling_data.set_tiling_key(value)"), std::string::npos);
  // 非扁平形态下根自带这四个量（ATT 在根上读写它们，再拷进各 group 的结构体）。
  EXPECT_NE(header.find("uint32_t block_dim;"), std::string::npos);
  EXPECT_NE(header.find("uint32_t ub_size;"), std::string::npos);
}

TEST(IsolatedPlanEntryTest, RejectsConflictingCxxStructNames) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::emitasc::EmitAscDialect>();
  mlir::OpBuilder builder(&context);
  auto first = mlir::emitasc::PyStructType::get(&context, builder.getStringAttr("Duplicate"),
                                                builder.getArrayAttr({mlir::TypeAttr::get(builder.getI32Type())}),
                                                builder.getArrayAttr({builder.getStringAttr("first")}));
  auto second = mlir::emitasc::PyStructType::get(&context, builder.getStringAttr("Duplicate"),
                                                 builder.getArrayAttr({mlir::TypeAttr::get(builder.getI64Type())}),
                                                 builder.getArrayAttr({builder.getStringAttr("second")}));
  std::string header;
  llvm::raw_string_ostream output(header);
  llvm::SmallVector<mlir::ascend::kernel::TilingSchema> schemas = {mlir::ascend::kernel::getTilingSchema(first),
                                                                   mlir::ascend::kernel::getTilingSchema(second)};
  ASSERT_TRUE(llvm::failed(mlir::ascend::kernel::emitTilingDataHeader(schemas, output)));
}

TEST(IsolatedPlanEntryTest, SanitizesCxxIdentifiers) {
  EXPECT_EQ(mlir::afir::makeCxxIdentifier("load.abs-store"), "load_x2eabs_x2dstore");
  EXPECT_EQ(mlir::afir::makeCxxIdentifier("123tensor"), "_123tensor");
  EXPECT_EQ(mlir::afir::makeCxxIdentifier("class"), "class_");
  EXPECT_EQ(mlir::afir::makeCxxIdentifier("a_b"), "a_b");
  EXPECT_EQ(mlir::afir::makeCxxIdentifier(""), "_x00");
  EXPECT_EQ(mlir::afir::makeCxxIdentifier("a_x2eb"), "a_x5fx2eb");
  EXPECT_NE(mlir::afir::makeCxxIdentifier("a.b"), mlir::afir::makeCxxIdentifier("a_x2eb"));
}

// RCore 的形状：一个 Plan 两个 stage。两个 group 的结构体必须平铺在最外层 TilingData 上
// （ATT 的 `ProcessMultiGroup` 就是这么访问的），`_main` 一 stage 收一个 tiling 形参、序号
// 即 stage，聚合入口按成员名把两者对上。
TEST(IsolatedPlanEntryTest, FlattensTwoStagePlanIntoAggregateRoot) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::afir::AFIRDialect>();
  context.getOrLoadDialect<mlir::emitasc::EmitAscDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  mlir::OpBuilder builder(&context);
  auto plan0 = makeTwoStagePlan(context);
  llvm::SmallVector<mlir::ModuleOp> modules = {*plan0};
  auto entry = mlir::ascend::kernel::runGenAggregateTilingData(modules);
  ASSERT_TRUE(succeeded(entry));
  mlir::ModuleOp entryModule = **entry;

  auto schemaAttr = entryModule->getAttrOfType<mlir::TypeAttr>("ascendc.tiling_schema");
  ASSERT_TRUE(schemaAttr);
  auto aggregate = mlir::dyn_cast<mlir::emitasc::PyStructType>(schemaAttr.getValue());
  ASSERT_TRUE(aggregate);
  // 单 plan 但两个 stage，不是扁平形态：根自带四个公共标量与 `graph0_tiling_key`，
  // 再是 workspace 偏移，最后两个 group 的结构体。
  llvm::SmallVector<llvm::StringRef> names;
  for (mlir::Attribute name : aggregate.getNamesAttr()) {
    names.push_back(mlir::cast<mlir::StringAttr>(name).getValue());
  }
  EXPECT_EQ(names, llvm::SmallVector<llvm::StringRef>(
                       {"block_dim", "corenum", "ub_size", "hbm_size", "graph0_tiling_key", "workspace2",
                        "graph0_result0_g0_tiling_data", "graph0_result0_g1_tiling_data"}));

  EXPECT_EQ(mlir::cast<mlir::TypeAttr>(aggregate.getTypesAttr()[5]).getValue(),
            mlir::IntegerType::get(&context, 64, mlir::IntegerType::Unsigned));

  // 契约里的 tiling 字段是一 stage 一个，按 stage 次序。
  auto contracts = entryModule->getAttrOfType<mlir::ArrayAttr>("ascendc.plan_entries");
  ASSERT_TRUE(contracts);
  ASSERT_EQ(contracts.size(), 1U);
  auto fields = mlir::cast<mlir::DictionaryAttr>(contracts[0]).getAs<mlir::ArrayAttr>("tiling_fields");
  ASSERT_TRUE(fields);
  ASSERT_EQ(fields.size(), 2U);
  EXPECT_EQ(mlir::cast<mlir::StringAttr>(fields[0]).getValue(), "graph0_result0_g0_tiling_data");
  EXPECT_EQ(mlir::cast<mlir::StringAttr>(fields[1]).getValue(), "graph0_result0_g1_tiling_data");

  // 聚合入口按这两个名字取成员，逐个传给 `_main` 的两个 tiling 形参。
  builder.setInsertionPointToEnd(plan0->getBody());
  llvm::SmallVector<mlir::Type> mainInputs;
  for (mlir::Attribute stage : mlir::cast<mlir::emitasc::PyStructType>(
                                   (*plan0)->getAttrOfType<mlir::TypeAttr>("ascendc.tiling_schema").getValue())
                                   .getTypesAttr()) {
    mainInputs.push_back(mlir::cast<mlir::TypeAttr>(stage).getValue());
  }
  auto declaration = builder.create<mlir::func::FuncOp>(plan0->getLoc(), "plan0_main",
                                                        mlir::FunctionType::get(&context, mainInputs, {}));
  markPlanMainTilingArguments(declaration, 0, 2);
  declaration.setPrivate();
  ASSERT_TRUE(succeeded(mlir::ascend::kernel::runGenAggregateKernelEntry(modules, entryModule)));
  auto main = entryModule.lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main);
  llvm::SmallVector<llvm::StringRef> members;
  main.walk([&](mlir::emitasc::MemberOp member) { members.push_back(member.getField()); });
  // 每取一个 stage 的结构体，紧跟着从根上读一次 workspace 偏移写进去：Plan 主函数读的是
  // stage 自己那一份，而 ATT 只往根上写（schema 把 workspace 声明在 `"fusion"` 作用域）。
  EXPECT_EQ(members, llvm::SmallVector<llvm::StringRef>({"graph0_result0_g0_tiling_data", "workspace2",
                                                         "graph0_result0_g1_tiling_data", "workspace2"}));
  llvm::SmallVector<llvm::StringRef> assigned;
  main.walk([&](mlir::emitasc::SetMemberOp member) { assigned.push_back(member.getField()); });
  EXPECT_EQ(assigned, llvm::SmallVector<llvm::StringRef>({"workspace2", "workspace2"}));
}
