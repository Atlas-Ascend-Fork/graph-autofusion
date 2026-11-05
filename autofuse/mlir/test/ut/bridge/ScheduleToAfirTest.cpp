/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bridge/ScheduleToAfir/ReduceScratchAdapter.h"
#include "test/Support/ReduceGraphBuilder.h"
#include "dialect/AFIR/Transforms/AFIRPasses.h"
#include <algorithm>

#include "bridge/ScheduleToAfir/ScheduleToAfir.h"

#include "bridge/AfirBuilder/AfirBuilder.h"
#include "bridge/AfirBuilder/WorkspaceViews.h"
#include "bridge/CodegenOrchestration/CodegenOrchestration.h"
#include "optimize/optimize.h"
#include "optimize/task_generator/reduce_schedule_case_generator.h"
#include "support/AfirElementTypes.h"
#include "test/Support/TestEnvironment.h"

#include "conversion/AFIRToAscendC/AscendCKernelABI.h"
#include "conversion/AFIRToAscendC/AFIRToAscendCPasses.h"
#include "artifact/KernelSource/GenAggregateKernelEntry.h"
#include "artifact/KernelSource/GenAggregateTilingData.h"
#include "artifact/TilingSource/TilingProgramSpec.h"
#include "artifact/TilingSource/TilingFuncEmitter.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/Support/SymbolicExpression.h"
#include "support/CapabilityShortfall.h"
#include "common/tiling_schema/schedule_group_names.h"
#include "dialect/AFPlan/IR/AFPlan.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "framework/easy_asc_graph/asc_graph_builder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <gtest/gtest.h>

namespace {

bool HasStructField(mlir::emitasc::PyStructType schema, llvm::StringRef name) {
  for (mlir::Attribute field : schema.getNamesAttr()) {
    if (mlir::cast<mlir::StringAttr>(field).getValue() == name) {
      return true;
    }
  }
  return false;
}

void ExpectCallSignaturesMatch(mlir::ModuleOp module) {
  module.walk([&](mlir::func::CallOp call) {
    auto callee = module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
    ASSERT_TRUE(callee);
    auto inputs = callee.getFunctionType().getInputs();
    ASSERT_EQ(call.getNumOperands(), inputs.size());
    for (auto [operand, type] : llvm::zip(call.getOperands(), inputs)) {
      EXPECT_EQ(operand.getType(), type);
    }
  });
}

void RunSrEntryPasses(mlir::MLIRContext &context, mlir::ModuleOp module) {
  mlir::PassManager signaturePassManager(&context);
  signaturePassManager.addPass(mlir::ascend::kernel::createAFIRKernelBodySignaturePass());
  ASSERT_TRUE(succeeded(signaturePassManager.run(module.getOperation())));

  mlir::PassManager tilingDataPassManager(&context);
  tilingDataPassManager.addPass(mlir::ascend::kernel::createGenTilingDataPass());
  ASSERT_TRUE(succeeded(tilingDataPassManager.run(module.getOperation())));

  mlir::PassManager kernelEntryPassManager(&context);
  kernelEntryPassManager.addPass(mlir::ascend::kernel::createGenKernelEntryPass());
  ASSERT_TRUE(succeeded(kernelEntryPassManager.run(module.getOperation())));
}

void AssignTensorIds(af::AscGraph &graph) {
  int64_t tensorId = 0;
  for (const auto &node : graph.GetAllNodes()) {
    for (const auto &output : node->outputs()) {
      output->attr.mem.tensor_id = tensorId++;
    }
  }
}

af::AscGraph MakeBinaryGraph(int64_t tilingKey) {
  auto graph = af::testing::AscGraphBuilder("elewise_brc")
                   .Loops({af::testing::Sym(8), af::testing::Sym(8)})
                   .Data("x", 0, af::DT_FLOAT16)
                   .Data("b", 1, af::DT_FLOAT16)
                   .Load("load_x", "x")
                   .Load("load_b", "b")
                   .Add("add", "load_x", "load_b")
                   .Store("store_y", "add")
                   .Output("y", "store_y", 0, af::DT_FLOAT16)
                   .Build();
  AssignTensorIds(graph);
  if (tilingKey >= 0) {
    graph.SetTilingKey(static_cast<uint32_t>(tilingKey));
  }
  return graph;
}

af::AscGraph MakeUnaryGraph(int64_t tilingKey) {
  auto graph = af::testing::AscGraphBuilder("elewise_unary")
                   .Loops({af::testing::Sym(8), af::testing::Sym(8)})
                   .Data("x", 0, af::DT_FLOAT16)
                   .Load("load_x", "x")
                   .Store("store_y", "load_x")
                   .Output("y", "store_y", 0, af::DT_FLOAT16)
                   .Build();
  AssignTensorIds(graph);
  if (tilingKey >= 0) {
    graph.SetTilingKey(static_cast<uint32_t>(tilingKey));
  }
  return graph;
}

// 两个 Store/Output 节点，io index 故意与遍历顺序相反：遍历先到的 "hi" 是 index 1,
// 后到的 "lo" 是 index 0。bridge 按遍历顺序收集，ABI 顺序只能由 index 决定。
// 含一个 `Scalar` 节点的图。它与 `MakeBinaryGraph` 的差别只有一处：第二个加数不是
// 又一个 `Data` + `Load`，而是一个编译期标量。
af::AscGraph MakeScalarGraph() {
  auto graph = af::testing::AscGraphBuilder("elewise_scalar")
                   .Loops({af::testing::Sym(8), af::testing::Sym(8)})
                   .Data("x", 0, af::DT_FLOAT16)
                   .Load("load_x", "x")
                   .Scalar("two", "2.5", af::DT_FLOAT16)
                   .Add("add", "load_x", "two")
                   .Store("store_y", "add")
                   .Output("y", "store_y", 0, af::DT_FLOAT16)
                   .Build();
  AssignTensorIds(graph);
  graph.SetTilingKey(0);
  return graph;
}

// 含一个 `ScalarData` 节点的图。与 `MakeScalarGraph` 只差一处：加数不是编译期字面量，
// 而是主机在运行期传进来的标量输入。**`scalarIndex` 与 `Data` 的 index 0 是同一个序号
// 空间**，所以传 0 就是一次真冲突（见 `RejectsScalarDataSharingAnIndexWithAData`）。
af::AscGraph MakeScalarDataGraph(int64_t scalarIndex) {
  auto graph = af::testing::AscGraphBuilder("elewise_scalar_data")
                   .Loops({af::testing::Sym(8), af::testing::Sym(8)})
                   .Data("x", 0, af::DT_FLOAT16)
                   .Load("load_x", "x")
                   .ScalarData("k", scalarIndex, af::DT_FLOAT16)
                   .Add("add", "load_x", "k")
                   .Store("store_y", "add")
                   .Output("y", "store_y", 0, af::DT_FLOAT16)
                   .Build();
  AssignTensorIds(graph);
  graph.SetTilingKey(0);
  return graph;
}

// 含一个 `IndexExpr` 节点的图。它是这一族的第三种标量源：`Scalar` 是编译期字面量、
// `ScalarData` 是 kernel 形参、`IndexExpr` 是一条按 size var 写的表达式，在 kernel 顶部
// 算成常量。**表达式带不带自由符号，走的是两条不同的路**，由参数决定。
af::AscGraph MakeIndexExprGraph(const char *expression, bool dynamicShape = false) {
  af::testing::AscGraphBuilder builder("elewise_index_expr");
  // 带自由符号的表达式要引用一个**真的** size var：名字不在图上登记的话，legacy 渲染时
  // 替换不掉它，发出来的 C++ 会引用一个不存在的变量。
  const auto extent = dynamicShape ? builder.SizeVar("s0") : af::testing::Sym(8);
  auto graph = builder.Loops({extent, af::testing::Sym(8)})
                   .Data("x", 0, af::DT_FLOAT16)
                   .Load("load_x", "x")
                   .IndexExpr("k", af::Expression::Parse(expression), af::DT_FLOAT16)
                   .Add("add", "load_x", "k")
                   .Store("store_y", "add")
                   .Output("y", "store_y", 0, af::DT_FLOAT16)
                   .Build();
  AssignTensorIds(graph);
  graph.SetTilingKey(0);
  return graph;
}

af::AscGraph MakeTwoOutputGraph() {
  auto graph = af::testing::AscGraphBuilder("two_outputs")
                   .Loops({af::testing::Sym(8), af::testing::Sym(8)})
                   .Data("x", 0, af::DT_FLOAT16)
                   .Data("b", 1, af::DT_FLOAT16)
                   .Load("load_x", "x")
                   .Load("load_b", "b")
                   .Add("add", "load_x", "load_b")
                   .Store("store_hi", "add")
                   .Output("hi", "store_hi", 1, af::DT_FLOAT16)
                   .Store("store_lo", "load_x")
                   .Output("lo", "store_lo", 0, af::DT_FLOAT16)
                   .Build();
  AssignTensorIds(graph);
  return graph;
}

ascir::ScheduledResult MakeTwoOutputScheduledResult() {
  ascir::ScheduleGroup group;
  group.impl_graphs.push_back(MakeTwoOutputGraph());
  ascir::ScheduledResult result;
  result.schedule_groups.push_back(std::move(group));
  return result;
}

ascir::ScheduledResult MakeScheduledResult(int64_t tilingKey) {
  ascir::ScheduleGroup group;
  group.impl_graphs.push_back(MakeBinaryGraph(tilingKey));
  group.impl_graphs.push_back(MakeUnaryGraph(tilingKey + 100));
  ascir::ScheduledResult result;
  result.schedule_groups.push_back(std::move(group));
  return result;
}

mlir::afir::TargetAttr MakeTarget(mlir::MLIRContext &context, int64_t aivNum = 48, int64_t ubSize = 196608) {
  return mlir::afir::TargetAttr::get(&context, mlir::StringAttr::get(&context, "Ascend910B1"), aivNum, ubSize);
}

void BuildTwoGraphModules(mlir::MLIRContext &context, llvm::SmallVector<mlir::ModuleOp> &modules,
                          llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>> &moduleRefs) {
  ascir::FusedScheduledResult fusedResult;
  fusedResult.fused_graph_name = "two_graphs";
  fusedResult.node_idx_to_scheduled_results = {
      {MakeScheduledResult(0), MakeScheduledResult(1)},
      {MakeScheduledResult(0), MakeScheduledResult(1)},
  };
  for (auto [subgraphId, scheduledResults] : llvm::enumerate(fusedResult.node_idx_to_scheduled_results)) {
    for (auto [planId, scheduledResult] : llvm::enumerate(scheduledResults)) {
      auto module = mlir::afir::bridge::PlanToAfir(context, fusedResult, scheduledResult, subgraphId, planId);
      ASSERT_TRUE(module);
      EXPECT_EQ(
          mlir::afplan::getCandidate(module->getOperation()),
          mlir::afplan::CandidateAttr::get(&context, static_cast<int64_t>(subgraphId), static_cast<int64_t>(planId)));
      module->getOperation()->setAttr(mlir::afir::kTargetAttrName, MakeTarget(context));
      llvm::SmallVector<unsigned> argumentCounts;
      for (mlir::func::FuncOp function : module->getOps<mlir::func::FuncOp>()) {
        argumentCounts.push_back(function.getNumArguments());
      }
      ASSERT_EQ(argumentCounts.size(), 2U);
      EXPECT_EQ(argumentCounts[0], 3U);
      EXPECT_EQ(argumentCounts[1], 2U);
      modules.push_back(*module);
      moduleRefs.push_back(std::move(module));
    }
  }
}

void RunAndVerifySrModules(mlir::MLIRContext &context, llvm::ArrayRef<mlir::ModuleOp> modules) {
  for (mlir::ModuleOp module : modules) {
    RunSrEntryPasses(context, module);
    ASSERT_TRUE(succeeded(mlir::verify(module.getOperation())));
    ExpectCallSignaturesMatch(module);
  }
}

void VerifyAggregateEntry(llvm::ArrayRef<mlir::ModuleOp> modules, mlir::ModuleOp entryModule) {
  auto schemaAttr = entryModule->getAttrOfType<mlir::TypeAttr>("ascendc.tiling_schema");
  ASSERT_TRUE(schemaAttr);
  auto schema = mlir::dyn_cast<mlir::emitasc::PyStructType>(schemaAttr.getValue());
  ASSERT_TRUE(schema);
  for (llvm::StringRef field : {"block_dim", "corenum", "ub_size", "hbm_size", "graph0_tiling_key", "graph1_tiling_key",
                                "graph0_result0_g0_tiling_data", "graph0_result1_g0_tiling_data",
                                "graph1_result0_g0_tiling_data", "graph1_result1_g0_tiling_data"}) {
    EXPECT_TRUE(HasStructField(schema, field));
  }
  ASSERT_TRUE(succeeded(mlir::ascend::kernel::runGenAggregateKernelEntry(modules, entryModule)));
  auto main = entryModule.lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main);
  size_t switchCount = 0;
  size_t callCount = 0;
  main.walk([&](mlir::scf::IndexSwitchOp) { ++switchCount; });
  main.walk([&](mlir::func::CallOp) { ++callCount; });
  EXPECT_EQ(switchCount, 2U);
  EXPECT_EQ(callCount, 4U);
  ASSERT_TRUE(succeeded(mlir::verify(entryModule.getOperation())));
  ExpectCallSignaturesMatch(entryModule);
}

mlir::OwningOpRef<mlir::ModuleOp> MakeTilingContractModule(mlir::MLIRContext &context, int64_t subgraphId,
                                                           int64_t planId,
                                                           llvm::ArrayRef<llvm::ArrayRef<llvm::StringRef>> stageFields,
                                                           llvm::ArrayRef<llvm::StringRef> shapeVars) {
  context.getOrLoadDialect<mlir::afir::AFIRDialect>();
  context.getOrLoadDialect<mlir::afplan::AFPlanDialect>();
  context.getOrLoadDialect<mlir::emitasc::EmitAscDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc(), "tiling_contract");
  module->setAttr(mlir::afplan::kCandidateAttrName, mlir::afplan::CandidateAttr::get(&context, subgraphId, planId));
  module->setAttr(mlir::afir::kTargetAttrName, MakeTarget(context));

  llvm::SmallVector<mlir::Attribute> shapeVarAttrs;
  for (llvm::StringRef shapeVar : shapeVars) {
    shapeVarAttrs.push_back(builder.getStringAttr(shapeVar));
  }
  if (!shapeVarAttrs.empty()) {
    module->setAttr("afir.frontend_shape_vars", builder.getArrayAttr(shapeVarAttrs));
  }

  llvm::SmallVector<mlir::Attribute> stageTypes;
  llvm::SmallVector<mlir::Attribute> stageNames;
  builder.setInsertionPointToEnd(module.getBody());
  for (auto [stageId, fields] : llvm::enumerate(stageFields)) {
    llvm::SmallVector<mlir::Attribute> fieldTypes;
    llvm::SmallVector<mlir::Attribute> fieldNames;
    for (llvm::StringRef field : fields) {
      fieldTypes.push_back(mlir::TypeAttr::get(mlir::IntegerType::get(&context, 32, mlir::IntegerType::Unsigned)));
      fieldNames.push_back(builder.getStringAttr(field));
    }
    auto stageType =
        mlir::emitasc::PyStructType::get(&context, builder.getStringAttr("Stage" + std::to_string(stageId)),
                                         builder.getArrayAttr(fieldTypes), builder.getArrayAttr(fieldNames));
    stageTypes.push_back(mlir::TypeAttr::get(stageType));
    // 载体的字段名就是这个 group 在最外层 TilingData 里的成员名，由共享公式生成
    // （`common/tiling_schema/schedule_group_names.h`），别在夹具里手写第二份。
    stageNames.push_back(builder.getStringAttr(
        mlir::common::MakeScheduleGroupTilingFieldName(subgraphId, planId, static_cast<int64_t>(stageId))));

    auto function = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "stage" + std::to_string(stageId) + "_template0", builder.getFunctionType({}, {}));
    function->setAttr(mlir::afplan::kVariantAttrName,
                      mlir::afplan::VariantAttr::get(&context, static_cast<int64_t>(stageId), 0));
    function->setAttr("afir.workspace_size", mlir::afir::ExprAttr::get(&context, builder.getStringAttr("0")));
  }
  auto schema = mlir::emitasc::PyStructType::get(&context, builder.getStringAttr("PlanTilingData"),
                                                 builder.getArrayAttr(stageTypes), builder.getArrayAttr(stageNames));
  module->setAttr("ascendc.tiling_schema", mlir::TypeAttr::get(schema));
  return module;
}

std::string CaptureTilingSpecDiagnostic(mlir::MLIRContext &context, llvm::ArrayRef<mlir::ModuleOp> modules) {
  std::string message;
  mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &diagnostic) {
    llvm::raw_string_ostream output(message);
    diagnostic.print(output);
  });
  EXPECT_TRUE(failed(mlir::bridge::buildAggregateTilingProgramSpec(modules)));
  return message;
}

TEST(ScheduleToAfirTest, RejectsUnsupportedDtype) {
  EXPECT_EQ(mlir::afir::bridge::detail::DTypeStr(ge::DT_FLOAT16), std::optional<std::string>("float16"));
  EXPECT_FALSE(mlir::afir::bridge::detail::DTypeStr(ge::DT_UNDEFINED).has_value());
}

// 暂缓名单上的每一行都必须还是供给表上的一位，而且必须真的还被挡着。
//
// 两个方向都要查，而且两个方向漏掉的后果都是静默的：
//
//   - 名单上写了一位，供给表里却没有它：这一行是句死话。它读起来像「这一位我们建模了、
//     只是先不放」，实际是「根本没建模」——排期的人会去找一个不存在的闸门。
//   - 供给表加了一位、名单上那一行忘了删：那一位永远进不来，而 lit 与 UT 全绿，因为它们
//     走的是 ODS 和 implementation，一条都不经过这道闸。bf16 就是这么被挡了一期：闸门一
//     开，reg 侧接管表上 28 格里 18 格是接管，而那期间 lit 与 UT 一直全绿。
TEST(ScheduleToAfirTest, HeldBackDtypesAreRealSupplyTablePositions) {
  const uint32_t supplied = mlir::afir::afirAllElementTypeBits();
  for (const mlir::afir::bridge::detail::HeldBackDType &held : mlir::afir::bridge::detail::HeldBackDTypes()) {
    EXPECT_NE(held.bit & supplied, 0U) << "暂缓名单上有一位不在供给表里（理由写的是「" << held.reason.str()
                                       << "」）：要么把它加进 AFIR_ELEMENT_TYPE_TABLE，要么删掉这一行——它现在是句死话";
  }
}

// 反向：供给表上被挡着的那几位，`IsAfirModeledDType` 必须答「不收」；没被挡的必须答「收」。
// 这条把那个函数与两张表钉在一起，所以「头文件说开了、函数返回 false」那种自相矛盾再也
// 写不出来。
TEST(ScheduleToAfirTest, ModeledDtypeGateFollowsTheSupplyTableMinusHeldBack) {
  uint32_t heldBackBits = 0U;
  for (const mlir::afir::bridge::detail::HeldBackDType &held : mlir::afir::bridge::detail::HeldBackDTypes()) {
    heldBackBits |= held.bit;
  }

#define AFIR_TEST_MODELED_DTYPE_GATE(NAME, BIT, CSV, BUILDER, BYTES, GE, KIND, WIDTH)                                \
  {                                                                                                                  \
    const bool held = (mlir::afir::kAfirBit##NAME & heldBackBits) != 0U;                                             \
    EXPECT_EQ(mlir::afir::bridge::detail::IsAfirModeledDType(ge::GE), !held)                                         \
        << CSV << " 在供给表上" << (held ? "、且在暂缓名单上" : "、不在暂缓名单上") << "，但这道闸答的是相反的结论"; \
  }
  AFIR_ELEMENT_TYPE_TABLE(AFIR_TEST_MODELED_DTYPE_GATE)
#undef AFIR_TEST_MODELED_DTYPE_GATE

  // 叫得出名字但还没建模的那几位，一律不收——它们走的是「带着类型名的那道闸」。
#define AFIR_TEST_NAMED_ONLY_GATE(CSV, BUILDER, BYTES, GE, KIND, WIDTH)                                       \
  EXPECT_FALSE(mlir::afir::bridge::detail::IsAfirModeledDType(ge::GE)) << CSV << " 还没建模，这道闸不该收它"; \
  EXPECT_EQ(mlir::afir::bridge::detail::DTypeStr(ge::GE), std::optional<std::string>(BUILDER))                \
      << CSV << " 必须叫得出名字，否则拒收诊断印不出类型名";
  AFIR_ELEMENT_TYPE_NAMED_ONLY(AFIR_TEST_NAMED_ONLY_GATE)
#undef AFIR_TEST_NAMED_ONLY_GATE
}

TEST(ScheduleToAfirTest, RejectsUnsupportedAxisType) {
  EXPECT_EQ(mlir::afir::bridge::detail::AxisTypeStr(af::Axis::Type::kAxisTypeOriginal),
            std::optional<std::string>("original"));
  EXPECT_FALSE(mlir::afir::bridge::detail::AxisTypeStr(af::Axis::Type::kAxisTypeInvalid).has_value());
}

// `exec_condition` 是上游那个枚举的**全部**取值，不是其中一个。
//
// `kConditionInvalid` 必须落到 `nullopt`：它在上游表示「还没定」，翻成 `no_cache` 会把
// 「没决定」说成「决定了不缓存」，在生成侧再也分不开。这条与上面两条同形——翻译表漏一格
// 的表现都是静默取到一个错值，不是报错。
TEST(ScheduleToAfirTest, RejectsUnsupportedExecCondition) {
  EXPECT_EQ(mlir::afir::bridge::detail::ExecConditionStr(af::ExecuteCondition::kNoCache),
            std::optional<std::string>("no_cache"));
  EXPECT_EQ(mlir::afir::bridge::detail::ExecConditionStr(af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis),
            std::optional<std::string>("cache_block_split_fused_brc_axis"));
  EXPECT_EQ(mlir::afir::bridge::detail::ExecConditionStr(af::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis),
            std::optional<std::string>("cache_block_split_origin_brc_axis"));
  EXPECT_FALSE(mlir::afir::bridge::detail::ExecConditionStr(af::ExecuteCondition::kConditionInvalid).has_value());
}

TEST(ScheduleToAfirTest, PreservesSizeExpressionPrecedence) {
  EXPECT_EQ(mlir::afir::bridge::detail::MakeSizeBytesExpr({"s0 + 1", "s1"}, 1), "(s0 + 1) * (s1)");
  EXPECT_EQ(mlir::afir::bridge::detail::MakeSizeBytesExpr({"Max(s0, 4)", "s1 * 2"}, 4),
            "(4 * (Max(s0, 4)) * (s1 * 2))");
}

TEST(ScheduleToAfirTest, ParsesNumericRepeatsAndRejectsInvalidNumbers) {
  EXPECT_EQ(mlir::afir::bridge::detail::MakeSizeBytesExpr({"2", "3"}, 1), "6");
  EXPECT_EQ(mlir::afir::bridge::detail::MakeSizeBytesExpr({"2x"}, 1), "(2x)");
  EXPECT_FALSE(mlir::afir::bridge::detail::MakeSizeBytesExpr({"9223372036854775808"}, 1).has_value());
  EXPECT_FALSE(mlir::afir::bridge::detail::MakeSizeBytesExpr({"0"}, 1).has_value());
  EXPECT_FALSE(mlir::afir::bridge::detail::MakeSizeBytesExpr({""}, 1).has_value());
}

// **这条断言换过一次说法。** 原先它靠一个反例立住：表里有一行是 `Tensor` 形态
// （最后是 `Cast`），于是「白名单是显式的」可以写成「这一行没有 buffer 形态」。
// `Cast` 接通之后反例没有了——今天 40 行全是 `Both`，再挑哪一行写 `EXPECT_FALSE` 都是
// 假的。
//
// 换成钉住**这张表本身是判据**：白名单只认表里写下的名字，一个拼错的、一个 legacy 有
// 而这里没登记的，都必须答 false。这条挡的是同一件事——「没登记过的节点被默许建出
// buffer 形态」——而且不依赖表里恰好有一行是 `Tensor`。
//
// 三个名字各挡一种走法：大小写、legacy 里真实存在但这一层还没接的节点、彻底没有的。
TEST(ScheduleToAfirTest, KeepsBufferFormAllowlistExplicit) {
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("Load"));
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("Store"));
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("Broadcast"));
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("Add"));
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("Sub"));
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("LogicalAnd"));
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("LogicalOr"));
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("Axpy"));
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("TrueDiv"));
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("Tanh"));
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("Erf"));
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("Pow"));
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("Cast"));
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("Rsqrt"));
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("LogicalNot"));
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("Eq"));
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("Ne"));
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("Lt"));
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("Le"));
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("Gt"));
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("Ge"));
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("LogicalXor"));
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("Log2"));
  // 第 7 批。`Ceil` 与 `Round` 挑出来是因为它们 legacy 走的是 fork 的包装，而
  // `Adv/Math.td` 里偏偏有同名的标准调用——照名字接会接到那一个去。
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("Ceil"));
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("Round"));
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("Igamma"));
  // 第 8 批。两个都走标准调用，而 `Log10` 还有一个同名的 `ascendc.log10` RegOp
  // （`Basic/OpReg.td` 的 `UnaryRegOp`，作用在 `RegTensor` 上）——那是 VF 那条路的供给，
  // 不是这里的；挑错了编得过而发出去的调用形参类型不对。
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("Log10"));
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("Hypot"));

  // 第 9 批。挑 `ChebyshevPolynomialU` 与 `ShiftedChebyshevPolynomialT` 两个：**两族各
  // 一个**，因为它们的前导头是两个不同的 utils，而那两个键是手工写的。
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("ChebyshevPolynomialU"));
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("ShiftedChebyshevPolynomialT"));

  // step 2a。挑 `FloorToInt` 是因为它**没有同名的 AscendC API**——CANN 全树里没有
  // `AscendC::FloorToInt`，legacy 的 `GetApiName()` 给的那个名字不是发出去的函数名。
  // 照那个名字去找供给会找不到，而照它去建一个新 op 则会发出一个不存在的调用。
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("Ceil2Int"));
  EXPECT_TRUE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("FloorToInt"));

  // 名字逐字匹配，不做大小写折叠——scheduler 侧的节点类型名是 `Cast` 不是 `cast`。
  EXPECT_FALSE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("cast"));
  // legacy 里真有这个节点类型（`ascir_builtin_ops_v2.cpp` 的 `REG_ASC_IR(Frexp)`，reg 侧
  // 独有），这一层还没接：没登记就是不支持，不是「像是 elementwise 就放行」。
  // **挑它不是随手挑的**：`Frexp` 与刚接通的 `hypot` 同一族供给——都在
  // `v35/ascendc/api_regbase/` 下、都走 `CalcVoidTmpSizeV2` 要一块尺寸为 0 的 tmp，
  // 站在这里的名字越像已接的那些，这条断言挡的东西才越实。接通它的时候这一行要换成别的
  // 还没接的名字——原先站这个位置的是 `Log10`（2026-10-27 接通），再之前是 `LogicalXor`、
  // `Eq`、`Rsqrt`，四个都接通了。
  //
  // 顺带记下它为什么还没接，而且这一条比 `Log10` 那条硬：`Frexp` 有**两个输出而且 dtype
  // 不同**（`.Output("mantissa", "T")` 是 f32、`.Output("exponent", "U")` 是 i32，
  // `FrexpAscIrCodegenImplV2::IsNodeValid` 还明写 `outputs().size() == 2`），调用形态是
  // `UnaryDoubleOutputApiTmpCall`。而这条路上从 `AFIR_BufferizableUnaryElementwiseOp` 到
  // 三段划分到 `ResultElementRule` 全部假定**一个**结果。接它不是加一行注册，是先开出
  // 「多输出」这根轴。
  EXPECT_FALSE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("Frexp"));
  EXPECT_FALSE(mlir::afir::bridge::isAFIRBufferFormOperationSupported("NoSuchScheduleNode"));
}

TEST(ScheduleToAfirTest, CanonicalizesGeneratedBufferSizeExpression) {
  mlir::MLIRContext context;
  mlir::afir::bridge::PlanInfo plan;
  plan.name = "generated_size_expr";

  mlir::afir::bridge::TemplateInfo tmpl;
  tmpl.name = "kernel";
  tmpl.original_name = "kernel";
  tmpl.workspace_size = "(4 * s0)";
  plan.templates.push_back(tmpl);
  plan.frontend_shape_vars = {"s0"};

  mlir::afir::bridge::BufferDescDTO workspace;
  workspace.sizeBytesExpr = "(2 * ((100 + s0)) * (s1) * (s2))";
  workspace.lifeTimeAxisId = -1;
  plan.workspace_plan.push_back(workspace);

  auto module = mlir::afir::bridge::BuildAfirModule(context, plan);
  ASSERT_TRUE(module);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("kernel");
  ASSERT_TRUE(function);
  auto frontendShapeVars = module->getOperation()->getAttrOfType<mlir::ArrayAttr>("afir.frontend_shape_vars");
  ASSERT_TRUE(frontendShapeVars);
  ASSERT_EQ(frontendShapeVars.size(), 1U);
  EXPECT_EQ(mlir::cast<mlir::StringAttr>(frontendShapeVars[0]).getValue(), "s0");
  auto workspaceSize = function->getAttrOfType<mlir::afir::ExprAttr>("afir.workspace_size");
  ASSERT_TRUE(workspaceSize);
  EXPECT_EQ(workspaceSize.getValue().getValue(), "(4 * s0)");
  auto workspacePlan = function->getAttrOfType<mlir::ArrayAttr>(mlir::afir::kWorkspacePlanAttrName);
  ASSERT_TRUE(workspacePlan);
  auto descriptor = mlir::cast<mlir::afir::BufferDescAttr>(workspacePlan[0]);
  ASSERT_TRUE(descriptor.getSizeBytesExpr());
  EXPECT_EQ(descriptor.getSizeBytesExpr().getValue().getValue().str(), "((2 * s0 * s1 * s2) + (200 * s1 * s2))");
}

TEST(ScheduleToAfirTest, TwoGraphsRunBridgeAndAllAggregateEntryPasses) {
  mlir::MLIRContext context;
  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>> moduleRefs;
  llvm::SmallVector<mlir::ModuleOp> modules;
  BuildTwoGraphModules(context, modules, moduleRefs);
  RunAndVerifySrModules(context, modules);
  auto tilingSpec = mlir::bridge::buildAggregateTilingProgramSpec(modules);
  ASSERT_TRUE(succeeded(tilingSpec));
  ASSERT_EQ(tilingSpec->plans.size(), 4U);
  EXPECT_EQ(tilingSpec->plans[0].stages[0].tiling_field_name, "graph0_result0_g0_tiling_data");

  auto entry = mlir::ascend::kernel::runGenAggregateTilingData(modules);
  ASSERT_TRUE(succeeded(entry));
  mlir::ModuleOp entryModule = **entry;
  VerifyAggregateEntry(modules, entryModule);
}

TEST(ScheduleToAfirTest, DefaultsUnsetTilingKeysToTemplateIds) {
  mlir::MLIRContext context;
  ascir::FusedScheduledResult fusedResult;
  fusedResult.fused_graph_name = "unset_tiling_keys";

  ascir::ScheduleGroup group;
  group.impl_graphs.push_back(MakeBinaryGraph(-1));
  group.impl_graphs.push_back(MakeUnaryGraph(-1));
  ascir::ScheduledResult scheduledResult;
  scheduledResult.schedule_groups.push_back(std::move(group));

  auto module = mlir::afir::bridge::PlanToAfir(context, fusedResult, scheduledResult, 0, 0);
  ASSERT_TRUE(module);
  for (mlir::func::FuncOp function : module->getOps<mlir::func::FuncOp>()) {
    mlir::afplan::VariantAttr variant = mlir::afplan::getVariant(function);
    auto graphAttr = function->getAttrOfType<mlir::afir::AscGraphAttrGroupsAttr>("afir.asc_graph");
    ASSERT_TRUE(variant);
    ASSERT_TRUE(graphAttr);
    EXPECT_EQ(graphAttr.getTilingKey(), variant.getTemplateId());
  }
  RunSrEntryPasses(context, *module);
}

// 图级 `AscGraph::SetTilingKey` + 一个 stage 多模板 = 所有模板拿到同一个 `tiling_key`，
// 而 host 就是靠它选模板。两边都分不开（ATT 用同一条规则，且只查「全设或全不设」），
// 所以在 bridge 就报错，而不是等 `GenKernelEntry` 以「key 重复」的形式冒出来。
//
// 把 `ScheduleToAfir` 里那条查重去掉，这条会变成「建得出来」而后在整条流水线更靠后处才红。
TEST(ScheduleToAfirTest, RejectsTemplatesSharingAGraphLevelTilingKey) {
  mlir::MLIRContext context;
  ascir::FusedScheduledResult fusedResult;
  fusedResult.fused_graph_name = "shared_tiling_key";

  ascir::ScheduleGroup group;
  group.impl_graphs.push_back(MakeBinaryGraph(0));
  group.impl_graphs.push_back(MakeUnaryGraph(0));
  ascir::ScheduledResult scheduledResult;
  scheduledResult.schedule_groups.push_back(std::move(group));

  std::string diagnostic;
  mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &entry) {
    diagnostic += entry.str();
    return mlir::success();
  });
  EXPECT_FALSE(mlir::afir::bridge::PlanToAfir(context, fusedResult, scheduledResult, 0, 0));
  EXPECT_NE(diagnostic.find("templates sharing tiling_key 0"), std::string::npos) << diagnostic;
}

TEST(ScheduleToAfirTest, TilingSpecReportsMissingTargetMetadata) {
  mlir::MLIRContext context;
  llvm::SmallVector<llvm::StringRef> fields = {"tiling_key", "s0"};
  auto module = MakeTilingContractModule(context, 0, 0, {fields}, {"s0"});
  module->getOperation()->removeAttr(mlir::afir::kTargetAttrName);
  llvm::SmallVector<mlir::ModuleOp> modules = {*module};

  std::string diagnostic = CaptureTilingSpecDiagnostic(context, modules);
  EXPECT_NE(diagnostic.find("afir.target"), std::string::npos);
  EXPECT_NE(diagnostic.find("subgraph 0 plan 0"), std::string::npos);
}

TEST(ScheduleToAfirTest, TilingSpecReportsMismatchedTargetMetadata) {
  mlir::MLIRContext context;
  llvm::SmallVector<llvm::StringRef> fields = {"tiling_key", "s0"};
  auto first = MakeTilingContractModule(context, 0, 0, {fields}, {"s0"});
  auto second = MakeTilingContractModule(context, 1, 0, {fields}, {"s0"});
  second->getOperation()->setAttr(mlir::afir::kTargetAttrName, MakeTarget(context, 48, 131072));
  llvm::SmallVector<mlir::ModuleOp> modules = {*first, *second};

  std::string diagnostic = CaptureTilingSpecDiagnostic(context, modules);
  EXPECT_NE(diagnostic.find("mismatched afir.target"), std::string::npos);
  EXPECT_NE(diagnostic.find("expected #afir.target<soc = \"Ascend910B1\", aiv_num = 48, ub_size = 196608>"),
            std::string::npos);
  EXPECT_NE(diagnostic.find("got #afir.target<soc = \"Ascend910B1\", aiv_num = 48, ub_size = 131072>"),
            std::string::npos);
}

// 资源上限的值域不再由本层检查——它已迁进 `#afir.target` 的 verifier，
// 负例见 test/lit/smoke/afir_target_invalid.mlir。这里只剩「有没有」与
// 「各候选是否一致」两条，因为它们是聚合这一层才成立的约束。

TEST(ScheduleToAfirTest, TilingSpecTracksDerivedShapeVariablesIndividually) {
  mlir::MLIRContext context;
  llvm::SmallVector<llvm::StringRef> sourceFields = {"tiling_key", "s0", "s1"};
  llvm::SmallVector<llvm::StringRef> destinationFields = {"tiling_key", "s0", "s1"};
  auto module = MakeTilingContractModule(context, 0, 0, {sourceFields, destinationFields}, {"s0", "s1"});
  mlir::OpBuilder builder(&context);
  auto sourceExpression = mlir::afir::ExprAttr::get(&context, builder.getStringAttr("s0"));
  module->getOperation()->setAttr(
      mlir::afir::kVarRelationsAttrName,
      builder.getArrayAttr({builder.getDictionaryAttr({
          builder.getNamedAttr(mlir::afir::kVarRelationDstStageKey, builder.getI64IntegerAttr(1)),
          builder.getNamedAttr(mlir::afir::kVarRelationSrcStageKey, builder.getI64IntegerAttr(0)),
          builder.getNamedAttr(mlir::afir::kVarRelationDstVarKey, builder.getStringAttr("s0")),
          builder.getNamedAttr(mlir::afir::kVarRelationSrcExprKey, sourceExpression),
      })}));
  llvm::SmallVector<mlir::ModuleOp> modules = {*module};

  auto spec = mlir::bridge::buildAggregateTilingProgramSpec(modules);
  ASSERT_TRUE(succeeded(spec));
  ASSERT_EQ(spec->plans[0].stages[1].derived_shape_vars, std::vector<std::string>({"s0"}));
  mlir::bridge::TilingCodeBundle bundle;
  bundle.tiling_functions["TilingHead"] = "// generated ATT declarations\n";
  auto source = mlir::bridge::emitInductorTilingFunc(*spec, bundle);
  ASSERT_TRUE(succeeded(source));
  EXPECT_NE(source->find("extern \"C\" bool AutofuseIsStaticShape() {\n  return false;"), std::string::npos);
  EXPECT_EQ(source->find("graph0_result0_g1_tiling_data.set_s0(s0)"), std::string::npos);
  EXPECT_NE(source->find("graph0_result0_g1_tiling_data.set_s1(s1)"), std::string::npos);
  EXPECT_NE(source->find("limit->aiv_num > g_no_limit_res.aiv_num"), std::string::npos);
  EXPECT_NE(source->find("limit->ub_size <= 256"), std::string::npos);
}

TEST(ScheduleToAfirTest, TilingFuncEmitterKeepsApiHeaderWithoutStateHeader) {
  mlir::MLIRContext context;
  llvm::SmallVector<llvm::StringRef> fields = {"tiling_key", "s0"};
  auto module = MakeTilingContractModule(context, 0, 0, {fields}, {"s0"});
  llvm::SmallVector<mlir::ModuleOp> modules = {*module};

  auto spec = mlir::bridge::buildAggregateTilingProgramSpec(modules);
  ASSERT_TRUE(succeeded(spec));
  mlir::bridge::TilingCodeBundle bundle;
  bundle.tiling_functions["TilingHead"] = "// generated ATT declarations\n";
  bundle.tiling_functions["TilingApiHeader"] = "bool ApiOnlyHeader();\n";
  auto source = mlir::bridge::emitInductorTilingFunc(*spec, bundle);
  ASSERT_TRUE(succeeded(source));
  EXPECT_NE(source->find("bool ApiOnlyHeader();"), std::string::npos);
}

TEST(ScheduleToAfirTest, TilingSpecEmitsWorkspaceFromTypedExpressionTree) {
  mlir::MLIRContext context;
  llvm::SmallVector<llvm::StringRef> fields = {"tiling_key", "s0", "t"};
  auto module = MakeTilingContractModule(context, 0, 0, {fields}, {"s0"});
  auto function = module->lookupSymbol<mlir::func::FuncOp>("stage0_template0");
  ASSERT_TRUE(function);
  mlir::OpBuilder builder(&context);
  function->setAttr("afir.workspace_size", mlir::afir::ExprAttr::get(&context, builder.getStringAttr("(s0 + t)")));
  llvm::SmallVector<mlir::ModuleOp> modules = {*module};

  auto spec = mlir::bridge::buildAggregateTilingProgramSpec(modules);
  ASSERT_TRUE(succeeded(spec));
  mlir::bridge::TilingCodeBundle bundle;
  bundle.tiling_functions["TilingHead"] = "// generated ATT declarations\n";
  auto source = mlir::bridge::emitInductorTilingFunc(*spec, bundle);
  ASSERT_TRUE(succeeded(source));
  EXPECT_NE(source->find("static_cast<uint64_t>(t.graph0_result0_g0_tiling_data.s0)"), std::string::npos);
  EXPECT_NE(source->find("static_cast<uint64_t>(t.graph0_result0_g0_tiling_data.t)"), std::string::npos);
  EXPECT_NE(source->find("const long double stage_ws_size"), std::string::npos);
  EXPECT_NE(source->find("bool CalculateWorkspaceSize"), std::string::npos);
  EXPECT_NE(source->find("bool GetWorkspaceSize(const AutofuseTilingData &t, uint64_t &workspace_size)"),
            std::string::npos);
}

// Compile and execute the emitted helper: text checks cannot detect truncation,
// stale upper output bits, or overflow at the uint64_t alignment boundary.
TEST(ScheduleToAfirTest, WorkspaceUint64CapacityAndOverflowExecuteGeneratedHelper) {
  mlir::MLIRContext context;
  llvm::SmallVector<llvm::StringRef> fields = {"tiling_key", "s0", "t"};
  auto module = MakeTilingContractModule(context, 0, 0, {fields, fields}, {});
  mlir::OpBuilder builder(&context);
  for (int stage = 0; stage < 2; ++stage) {
    auto function = module->lookupSymbol<mlir::func::FuncOp>("stage" + std::to_string(stage) + "_template0");
    function->setAttr("afir.workspace_size", mlir::afir::ExprAttr::get(&context, builder.getStringAttr("(s0 * t)")));
  }
  llvm::SmallVector<mlir::ModuleOp> modules = {*module};
  auto spec = mlir::bridge::buildAggregateTilingProgramSpec(modules);
  ASSERT_TRUE(succeeded(spec));
  mlir::bridge::TilingCodeBundle bundle;
  bundle.tiling_functions["TilingHead"] = "// ATT declarations supplied by the test harness\n";
  auto source = mlir::bridge::emitInductorTilingFunc(*spec, bundle);
  ASSERT_TRUE(succeeded(source));
  EXPECT_NE(source->find("uint64_t *workspaceSize"), std::string::npos);
  const auto begin = source->find("bool CalculateWorkspaceSize(");
  ASSERT_NE(begin, std::string::npos);
  auto compiler = llvm::sys::findProgramByName("c++");
  ASSERT_TRUE(static_cast<bool>(compiler));
  llvm::SmallString<128> directory;
  ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory("workspace-uint64", directory));
  auto cleanup = llvm::make_scope_exit([&] { llvm::sys::fs::remove_directories(directory); });
  const std::string input = directory.str().str() + "/test.cpp";
  const std::string executable = directory.str().str() + "/test";
  std::error_code error;
  llvm::raw_fd_ostream file(input, error);
  ASSERT_FALSE(error);
  file << "#include <cstdint>\n#include <limits>\n"
          "struct Stage { uint32_t tiling_key = 0, s0 = 0, t = 0; };\n"
          "struct AutofuseTilingData { Stage graph0_result0_g0_tiling_data, graph0_result0_g1_tiling_data;\n"
          " uint32_t block_dim = 0; void set_block_dim(uint32_t x) { block_dim = x; }\n"
          " uint32_t get_block_dim() const { return block_dim; } void set_ub_size(uint32_t) {} };\n"
          "namespace optiling { bool GetTiling(AutofuseTilingData &, int, void *) { return true; } }\n";
  file << source->substr(begin);
  file << R"cpp(
    int main() {
      AutofuseTilingData data{};
      uint64_t size = UINT64_MAX;
      uint32_t blocks = 0;
      if (AutofuseTiling(&data, &size, &blocks, nullptr) != 0 || size != 0) return 1;
      auto &a = data.graph0_result0_g0_tiling_data;
      auto &b = data.graph0_result0_g1_tiling_data;
      a.s0 = 65536;
      a.t = 65537;
      if (AutofuseTiling(&data, &size, &blocks, nullptr) != 0 || size != UINT64_C(4295032832)) return 2;
      a.s0 = UINT32_MAX;
      a.t = UINT32_MAX;
      if (!GetWorkspaceSize(data, size) || size != UINT64_C(0xfffffffe00000200)) return 3;
      b.s0 = 2;
      b.t = UINT32_MAX;
      if (GetWorkspaceSize(data, size)) return 4;  // total == UINT64_MAX: alignment overflow
      b.s0 = 3;
      if (GetWorkspaceSize(data, size)) return 5;  // total exceeds UINT64_MAX
      return 0;
    }
  )cpp";
  file.close();
  ASSERT_EQ(
      llvm::sys::ExecuteAndWait(*compiler, {*compiler, "-std=c++17", input, "-o", executable}, std::nullopt, {}, 30),
      0);
  EXPECT_EQ(llvm::sys::ExecuteAndWait(executable, {executable}, std::nullopt, {}, 30), 0);
}

TEST(ScheduleToAfirTest, TilingSpecEmitsNestedWorkspaceExpression) {
  mlir::MLIRContext context;
  llvm::SmallVector<llvm::StringRef> fields = {"tiling_key", "s0"};
  auto module = MakeTilingContractModule(context, 0, 0, {fields}, {"s0"});
  auto function = module->lookupSymbol<mlir::func::FuncOp>("stage0_template0");
  ASSERT_TRUE(function);
  mlir::OpBuilder builder(&context);
  function->setAttr("afir.workspace_size",
                    mlir::afir::ExprAttr::get(&context, builder.getStringAttr("Max(8192, Min(65536, (64 * s0)))")));
  llvm::SmallVector<mlir::ModuleOp> modules = {*module};

  auto spec = mlir::bridge::buildAggregateTilingProgramSpec(modules);
  ASSERT_TRUE(succeeded(spec));
  mlir::bridge::TilingCodeBundle bundle;
  bundle.tiling_functions["TilingHead"] = "// generated ATT declarations\n";
  auto source = mlir::bridge::emitInductorTilingFunc(*spec, bundle);
  ASSERT_TRUE(succeeded(source));
  // `Max` / `Min` 是 ATT 发在 `namespace optiling` 里的函数模板，而
  // `CalculateWorkspaceSize` 发在全局作用域，所以调用点必须带限定。
  EXPECT_NE(source->find("optiling::Max("), std::string::npos);
  EXPECT_NE(source->find("optiling::Min("), std::string::npos);
  EXPECT_NE(source->find("64 * static_cast<uint64_t>(t.graph0_result0_g0_tiling_data.s0)"), std::string::npos);
}

TEST(ScheduleToAfirTest, TilingSpecRejectsUnknownWorkspaceSymbol) {
  mlir::MLIRContext context;
  llvm::SmallVector<llvm::StringRef> fields = {"tiling_key", "s0"};
  auto module = MakeTilingContractModule(context, 0, 0, {fields}, {"s0"});
  auto function = module->lookupSymbol<mlir::func::FuncOp>("stage0_template0");
  ASSERT_TRUE(function);
  mlir::OpBuilder builder(&context);
  function->setAttr("afir.workspace_size",
                    mlir::afir::ExprAttr::get(&context, builder.getStringAttr("(missing + s0)")));
  llvm::SmallVector<mlir::ModuleOp> modules = {*module};

  std::string diagnostic = CaptureTilingSpecDiagnostic(context, modules);
  EXPECT_NE(diagnostic.find("afir.workspace_size references unknown stage field 'missing'"), std::string::npos);
}

TEST(ScheduleToAfirTest, TilingSpecRejectsUnknownRelationDestination) {
  mlir::MLIRContext context;
  llvm::SmallVector<llvm::StringRef> fields = {"tiling_key", "s0"};
  auto module = MakeTilingContractModule(context, 0, 0, {fields}, {"s0"});
  mlir::OpBuilder builder(&context);
  auto sourceExpression = mlir::afir::ExprAttr::get(&context, builder.getStringAttr("s0"));
  module->getOperation()->setAttr(
      mlir::afir::kVarRelationsAttrName,
      builder.getArrayAttr({builder.getDictionaryAttr({
          builder.getNamedAttr(mlir::afir::kVarRelationDstStageKey, builder.getI64IntegerAttr(0)),
          builder.getNamedAttr(mlir::afir::kVarRelationSrcStageKey, builder.getI64IntegerAttr(0)),
          builder.getNamedAttr(mlir::afir::kVarRelationDstVarKey, builder.getStringAttr("missing")),
          builder.getNamedAttr(mlir::afir::kVarRelationSrcExprKey, sourceExpression),
      })}));
  llvm::SmallVector<mlir::ModuleOp> modules = {*module};

  std::string diagnostic = CaptureTilingSpecDiagnostic(context, modules);
  EXPECT_NE(diagnostic.find("destination 'missing' is not a field of stage 0"), std::string::npos);
}

// 动态维能从同轴的输入直接问出来时，就发 `memref.dim`，不走符号序言。
//
// 这条用例此前断言 `dimCount == 0`，而它的 walk 体里那两句（源是 kernel 参数、结果有使用
// 者）在计数为 0 时一次都执行不到——写它的人预期的是有 `DimOp`，那个 0 是后来按当时的
// 实际行为改的。实际行为之所以是 0，是因为 `proveExpressionsEqual` 用 `Parse` 校验
// `ExpectEq(a, a)` 的规范性，而它的规范形式就是 `True`：唯一该返回 true 的分支必然解析
// 失败（设计文档 8.3.1，S5 债务）。改用 `ParseRelaxed` 之后快路径才真正命中。
//
// 数值不受影响——动态维只是 `memref` 类型上的尺寸，两条路给出的是同一个值；差别是符号
// 序言那一段算术还发不发。
TEST(ScheduleToAfirTest, MaterializesDynamicDimFromTheMatchingInput) {
  auto makeTensor = [](mlir::afir::Space space, int64_t bufferId) {
    mlir::afir::bridge::TensorValueDTO value;
    value.tensor.shape = {mlir::ShapedType::kDynamic};
    value.tensor.shape_exprs = {"s0"};
    value.tensor.dtype = "float16";
    value.tensor.axis = {0};
    value.tensor.repeats = {"s0"};
    value.tensor.strides = {"1"};
    value.buffer.space = space;
    value.buffer.bufferId = bufferId;
    value.buffer.slotCount = space == mlir::afir::Space::UB ? 2 : -1;
    value.buffer.queueDepth = space == mlir::afir::Space::UB ? 2 : -1;
    value.buffer.sizeBytesExpr = "(2 * s0)";
    value.buffer.lifeTimeAxisId = -1;
    return value;
  };

  mlir::MLIRContext context;
  mlir::afir::bridge::PlanInfo plan;
  plan.name = "dynamic_buffer_form";

  mlir::afir::bridge::TemplateInfo tmpl;
  tmpl.name = "dynamic_buffer_form";
  tmpl.original_name = tmpl.name;
  tmpl.buffer_form = true;
  tmpl.size_vars = {"s0"};
  tmpl.axes.push_back({0, "z", "original", "s0"});

  auto input0 = makeTensor(mlir::afir::Space::GM, 0);
  auto input1 = makeTensor(mlir::afir::Space::GM, 1);
  auto local0 = makeTensor(mlir::afir::Space::UB, 2);
  auto local1 = makeTensor(mlir::afir::Space::UB, 3);
  auto output = makeTensor(mlir::afir::Space::GM, 4);
  tmpl.inputs = {{"input0", input0, 0}, {"input1", input1, 1}};
  tmpl.outputs = {{"output", output, 0}};

  mlir::afir::bridge::NodeInfo load0;
  load0.name = "load0";
  load0.type = "Load";
  load0.input_names = {"input0"};
  load0.inputs = {input0};
  load0.output = local0;
  load0.sched_axis = {0};
  load0.loop_axis = 0;

  mlir::afir::bridge::NodeInfo load1 = load0;
  load1.name = "load1";
  load1.input_names = {"input1"};
  load1.inputs = {input1};
  load1.output = local1;

  mlir::afir::bridge::NodeInfo add;
  add.name = "add";
  add.type = "Add";
  add.input_names = {"load0", "load1"};
  add.inputs = {local0, local1};
  add.output = local1;
  add.sched_axis = {0};
  add.loop_axis = 0;

  mlir::afir::bridge::NodeInfo store;
  store.name = "output";
  store.type = "Store";
  store.input_names = {"add"};
  store.inputs = {local1};
  store.output = output;
  store.sched_axis = {0};
  store.loop_axis = 0;

  tmpl.nodes = {load0, load1, add, store};
  plan.templates.push_back(tmpl);

  auto module = mlir::afir::bridge::BuildAfirModule(context, plan);
  ASSERT_TRUE(module);

  // 每个 `memref.dim` 都必须问到 kernel 参数（那是唯一能在运行期给出真实尺寸的源），
  // 并且被真正用上；发一个没人用的 dim 与不发是一回事。
  unsigned dimCount = 0;
  module->walk([&](mlir::memref::DimOp dim) {
    ++dimCount;
    EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(dim.getSource()));
    EXPECT_FALSE(dim.getResult().use_empty());
  });
  // 本夹具里有两处动态维的表达式与某个输入同轴，因此两处都走快路径。数字本身不是判据，
  // 判据是「大于零」加上面两条——但写成精确值，快路径被改回去时这里会直接失败而不是
  // 悄悄少发几个。
  EXPECT_EQ(dimCount, 2U);
}

// 整数 `ir_attr` 走完 `BuildAfirModule` 之后建出来的属性，**类型要和 ODS 的约束对得上**。
//
// 这条是补的，而且是被咬了一次才补的：第 9 批（多项式族）第一版 ODS 写的是 `SI64Attr`
// （有符号 `si64`），而 `buildBufferNodeOp` 建的是 `getI64IntegerAttr`（无符号性 `i64`），
// 两者不是同一个类型，verifier 当场拒收——八格从干净回落变成**硬失败**。
//
// **lit 按原理抓不到它**：lit 从「IR 里属性已经写好了」开始，手写的是 `n = 3 : i64`，
// 走的是正确的那一种，bridge 的构造路径一行都没走到。能抓到的只有这条 UT 和接管表，
// 而接管表要二十分钟、还得占着那台 aarch64。
//
// 所以这里断言的是 `mlir::verify` 而不只是 `BuildAfirModule` 返回非空：build 本身不跑
// verifier，属性类型不对的模块照样建得出来，要到下游 pass 才红。
//
// `n` 取负数：包装里 `N < 0` 是一条真分支（`Duplicate(dst, 0)`），而无符号性的
// `IntegerAttr` 存的仍是有符号 `APInt`——取 0 或正数的话这一位验不出来。
TEST(ScheduleToAfirTest, BuildsIntIrAttrWithTheTypeOdsRequires) {
  auto makeTensor = [](mlir::afir::Space space, int64_t bufferId) {
    mlir::afir::bridge::TensorValueDTO value;
    value.tensor.shape = {mlir::ShapedType::kDynamic};
    value.tensor.shape_exprs = {"s0"};
    value.tensor.dtype = "float32";
    value.tensor.axis = {0};
    value.tensor.repeats = {"s0"};
    value.tensor.strides = {"1"};
    value.buffer.space = space;
    value.buffer.bufferId = bufferId;
    value.buffer.slotCount = space == mlir::afir::Space::UB ? 2 : -1;
    value.buffer.queueDepth = space == mlir::afir::Space::UB ? 2 : -1;
    value.buffer.sizeBytesExpr = "(4 * s0)";
    value.buffer.lifeTimeAxisId = -1;
    return value;
  };

  mlir::MLIRContext context;
  mlir::afir::bridge::PlanInfo plan;
  plan.name = "int_ir_attr";

  mlir::afir::bridge::TemplateInfo tmpl;
  tmpl.name = "int_ir_attr";
  tmpl.original_name = tmpl.name;
  tmpl.buffer_form = true;
  tmpl.size_vars = {"s0"};
  tmpl.axes.push_back({0, "z", "original", "s0"});

  auto input = makeTensor(mlir::afir::Space::GM, 0);
  auto local0 = makeTensor(mlir::afir::Space::UB, 1);
  auto local1 = makeTensor(mlir::afir::Space::UB, 2);
  auto output = makeTensor(mlir::afir::Space::GM, 3);
  tmpl.inputs = {{"input", input, 0}};
  tmpl.outputs = {{"output", output, 0}};

  mlir::afir::bridge::NodeInfo load;
  load.name = "load";
  load.type = "Load";
  load.input_names = {"input"};
  load.inputs = {input};
  load.output = local0;
  load.sched_axis = {0};
  load.loop_axis = 0;

  mlir::afir::bridge::NodeInfo poly;
  poly.name = "poly";
  poly.type = "ChebyshevPolynomialT";
  poly.input_names = {"load"};
  poly.inputs = {local0};
  poly.output = local1;
  poly.sched_axis = {0};
  poly.loop_axis = 0;
  poly.int_ir_attr = -3;

  mlir::afir::bridge::NodeInfo store;
  store.name = "output";
  store.type = "Store";
  store.input_names = {"poly"};
  store.inputs = {local1};
  store.output = output;
  store.sched_axis = {0};
  store.loop_axis = 0;

  tmpl.nodes = {load, poly, store};
  plan.templates.push_back(tmpl);

  auto module = mlir::afir::bridge::BuildAfirModule(context, plan);
  ASSERT_TRUE(module);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(module->getOperation())))
      << "整数 ir_attr 建出来的属性类型与 ODS 的约束对不上；对不上不会在这里返回空，"
         "而是让整条编译在下游红掉（第 9 批那八格就是这么从回落变成硬失败的）";

  unsigned seen = 0;
  module->walk([&](mlir::afir::ChebyshevPolynomialTOp op) {
    ++seen;
    auto attr = op->getAttrOfType<mlir::IntegerAttr>("n");
    ASSERT_TRUE(attr);
    EXPECT_EQ(attr.getValue().getSExtValue(), -3);
  });
  EXPECT_EQ(seen, 1U) << "没建出那个 op，上面那条 verify 断言就是恒真";
}

// `Scalar` 节点不产生 AFIR op，只产生一个有类型的常量——跟 `Data` / `Output` 同类。
// 折叠没做对的表现不是算错：要么 bridge 报「不支持的节点类型」（它不在 buffer 形态
// 白名单里，也不该在），要么给它建出一块 `memref.alloc` 当 tile 用。
//
// 这条只有 UT 能验。LIT 从「IR 里已经有一个标量 SSA 值」开始，走不到 bridge；而 CPU
// E2E 走不通——legacy 的 ATT tiling 对含 `Scalar` 节点的图直接返回失败，在 MLIR 这一
// 半跑起来之前（见方案 E3 段）。
// 从**真实的 AscGraph** 出发走完 ScheduleToAfir，不是手填 `NodeInfo`。
//
// `FoldsScalarNodeIntoATypedConstant` 钉的是 `AfirBuilder` 那一半；ScheduleToAfir 这一半
// （`ReadScalarValue` 把字面量取出来、`AppendTemplateNode` 让 `Scalar` 绕开 buffer 形态
// 白名单）它一行都没走到。而这一半独有的失败方式是：`Scalar` 的输出仍然要过
// `AppendNodeOutputs` -> `InitializeTensorValue`，那里对 position / alloc_type 都有要求，
// 而 `Scalar` 节点这两个字段是默认值。
TEST(ScheduleToAfirTest, BridgesAGraphContainingAScalarNode) {
  mlir::MLIRContext context;
  ascir::FusedScheduledResult fusedResult;
  fusedResult.fused_graph_name = "elewise_scalar";
  ascir::ScheduleGroup group;
  group.impl_graphs.push_back(MakeScalarGraph());
  ascir::ScheduledResult scheduled;
  scheduled.schedule_groups.push_back(std::move(group));
  fusedResult.node_idx_to_scheduled_results = {{scheduled}};

  auto module = mlir::afir::bridge::PlanToAfir(context, fusedResult, scheduled, 0, 0);
  ASSERT_TRUE(module);

  // 字面量到了 MLIR 这边是一个 f16 常量，不是一串文本、不是 0。
  unsigned constantCount = 0;
  module->walk([&](mlir::arith::ConstantOp constant) {
    auto attr = mlir::dyn_cast<mlir::FloatAttr>(constant.getValue());
    if (!attr || !attr.getType().isF16()) {
      return;
    }
    ++constantCount;
    EXPECT_DOUBLE_EQ(attr.getValueAsDouble(), 2.5);
  });
  EXPECT_EQ(constantCount, 1U);

  unsigned addCount = 0;
  module->walk([&](mlir::afir::AddOp add) {
    ++addCount;
    EXPECT_TRUE(mlir::isa<mlir::arith::ConstantOp>(add.getRhs().getDefiningOp()));
  });
  EXPECT_EQ(addCount, 1U);
}

// `ScalarData` 与 `Scalar` 只差一个字，落到 AFIR 上却是两样东西：`Scalar` 折成一个
// `arith.constant`，`ScalarData` 变成一个**按值传的 kernel 形参**。
//
// 这条钉的是 bridge 这一半——CPU E2E 只看得见最后发出来的 C++，中间这一步错了要到最后
// 才看得出来，而且看到的是别的症状（形参类型序列不对、或者干脆链不上）。
TEST(ScheduleToAfirTest, BridgesAGraphContainingAScalarDataNode) {
  mlir::MLIRContext context;
  ascir::FusedScheduledResult fusedResult;
  fusedResult.fused_graph_name = "elewise_scalar_data";
  ascir::ScheduleGroup group;
  group.impl_graphs.push_back(MakeScalarDataGraph(/*scalarIndex=*/1));
  ascir::ScheduledResult scheduled;
  scheduled.schedule_groups.push_back(std::move(group));
  fusedResult.node_idx_to_scheduled_results = {{scheduled}};

  auto module = mlir::afir::bridge::PlanToAfir(context, fusedResult, scheduled, 0, 0);
  ASSERT_TRUE(module);

  mlir::func::FuncOp kernel;
  for (mlir::func::FuncOp function : module->getOps<mlir::func::FuncOp>()) {
    if (mlir::afir::hasFunctionRole(function, mlir::afir::FunctionRole::KernelBody)) {
      ASSERT_FALSE(kernel) << "expected exactly one kernel body";
      kernel = function;
    }
  }
  ASSERT_TRUE(kernel);

  // buffer 形态下参数是 [输入…, 输出…]：x、k、y。
  ASSERT_EQ(kernel.getNumArguments(), 3U);

  // **标量形参的类型是元素类型本身，不是 memref。** 建成 memref 的话它就成了一块 GM
  // 存储，主机要传地址而不是值——签名与 legacy 对不上。
  EXPECT_TRUE(kernel.getArgument(1).getType().isF16());
  EXPECT_TRUE(mlir::isa<mlir::MemRefType>(kernel.getArgument(0).getType()));
  EXPECT_TRUE(mlir::isa<mlir::MemRefType>(kernel.getArgument(2).getType()));

  // 身份由参数自己声明。序号取自 AscGraph 的 `index`，与 `Data` 同一个空间。
  auto scalarRole = mlir::afir::getKernelArgAttr(kernel, 1);
  ASSERT_TRUE(scalarRole);
  EXPECT_EQ(scalarRole.getRole(), mlir::afir::KernelArgRole::ScalarInput);
  EXPECT_EQ(scalarRole.getIndex(), 1);
  auto dataRole = mlir::afir::getKernelArgAttr(kernel, 0);
  ASSERT_TRUE(dataRole);
  EXPECT_EQ(dataRole.getRole(), mlir::afir::KernelArgRole::Input);
  EXPECT_EQ(dataRole.getIndex(), 0);

  // 它没有存储，所以不挂 `afir.buffer_desc`；而另外两个参数挂。
  EXPECT_FALSE(kernel.getArgAttr(1, mlir::afir::kBufferDescAttrName));
  EXPECT_TRUE(kernel.getArgAttr(0, mlir::afir::kBufferDescAttrName));
  EXPECT_TRUE(kernel.getArgAttr(2, mlir::afir::kBufferDescAttrName));

  // **`afir.arg_descs` 仍然覆盖每一个数据参数**：标量那一格是退化视图——无轴、只留角色。
  // 少一格的话描述子下标就跟参数下标错开，错开之后每条检查都还在跑，只是查的是别的参数。
  auto argDescs = kernel->getAttrOfType<mlir::ArrayAttr>(mlir::afir::kArgDescsAttrName);
  ASSERT_TRUE(argDescs);
  ASSERT_EQ(argDescs.size(), 3U);
  auto scalarDesc = mlir::dyn_cast<mlir::afir::TensorDescAttr>(argDescs[1]);
  ASSERT_TRUE(scalarDesc);
  EXPECT_EQ(scalarDesc.getRole().getValue(), mlir::afir::Role::Input);
  EXPECT_TRUE(scalarDesc.getAxis().empty());
  EXPECT_TRUE(scalarDesc.getRepeats().empty());
  EXPECT_TRUE(scalarDesc.getStrides().empty());

  // 加数是那个块参数本身，不是常量——这正是它与 `Scalar` 分道的地方。
  unsigned addCount = 0;
  module->walk([&](mlir::afir::AddOp add) {
    ++addCount;
    EXPECT_TRUE(mlir::isa<mlir::BlockArgument>(add.getRhs()));
    EXPECT_EQ(add.getRhs(), kernel.getArgument(1));
  });
  EXPECT_EQ(addCount, 1U);
}

// `Data` 与 `ScalarData` 共用一个 index 序号空间，所以同一个 index 上出现两者是**冲突**，
// 不是两个各自合法的参数。不报的话主机侧两个实参会挤到同一个位置上。
TEST(ScheduleToAfirTest, RejectsScalarDataSharingAnIndexWithAData) {
  mlir::MLIRContext context;
  ascir::FusedScheduledResult fusedResult;
  fusedResult.fused_graph_name = "elewise_scalar_data";
  ascir::ScheduleGroup group;
  group.impl_graphs.push_back(MakeScalarDataGraph(/*scalarIndex=*/0));
  ascir::ScheduledResult scheduled;
  scheduled.schedule_groups.push_back(std::move(group));
  fusedResult.node_idx_to_scheduled_results = {{scheduled}};

  EXPECT_FALSE(mlir::afir::bridge::PlanToAfir(context, fusedResult, scheduled, 0, 0));
}

// `IndexExpr` 与 `Scalar` 落在同一处——一个有类型的常量，不产生 kernel 形参。
//
// **它不需要归纳变量。** legacy 把两者一起收进 `kernel.constant_tensors`，由
// `AppendConstTensorInit` 在 kernel 顶部物化，而那一处跑在 `root_loop.Generate` 之前。
TEST(ScheduleToAfirTest, FoldsAConstantIndexExprIntoATypedConstant) {
  mlir::MLIRContext context;
  ascir::FusedScheduledResult fusedResult;
  fusedResult.fused_graph_name = "elewise_index_expr";
  ascir::ScheduleGroup group;
  group.impl_graphs.push_back(MakeIndexExprGraph("2"));
  ascir::ScheduledResult scheduled;
  scheduled.schedule_groups.push_back(std::move(group));
  fusedResult.node_idx_to_scheduled_results = {{scheduled}};

  auto module = mlir::afir::bridge::PlanToAfir(context, fusedResult, scheduled, 0, 0);
  ASSERT_TRUE(module);

  mlir::func::FuncOp kernel;
  for (mlir::func::FuncOp function : module->getOps<mlir::func::FuncOp>()) {
    if (mlir::afir::hasFunctionRole(function, mlir::afir::FunctionRole::KernelBody)) {
      ASSERT_FALSE(kernel) << "expected exactly one kernel body";
      kernel = function;
    }
  }
  ASSERT_TRUE(kernel);

  // **它不占形参**：只有 x 和 y。`ScalarData` 那条恰好相反，那边标量占一格。
  EXPECT_EQ(kernel.getNumArguments(), 2U);

  unsigned constantCount = 0;
  module->walk([&](mlir::arith::ConstantOp constant) {
    auto attr = mlir::dyn_cast<mlir::FloatAttr>(constant.getValue());
    if (!attr || !attr.getType().isF16()) {
      return;
    }
    ++constantCount;
    EXPECT_DOUBLE_EQ(attr.getValueAsDouble(), 2.0);
  });
  EXPECT_EQ(constantCount, 1U);

  unsigned addCount = 0;
  module->walk([&](mlir::afir::AddOp add) {
    ++addCount;
    EXPECT_TRUE(mlir::isa<mlir::arith::ConstantOp>(add.getRhs().getDefiningOp()));
  });
  EXPECT_EQ(addCount, 1U);
}

// 带自由符号的表达式走另一条路：符号逐个取成 `afir.tiling_param`、表达式本身成为一个
// `afir.symbolic_eval`，两者都落在**入口块的尺寸序言**里——那正是 legacy 的落点
// （`AppendConstTensorInit` 跑在 `root_loop.Generate` 之前）。
//
// 算出来的是 `index`，而它要当元素类型的标量用，所以还跟着一次转换：浮点要先到 i64
// 再 `sitofp`，因为 `sitofp` 不收 `index`。
TEST(ScheduleToAfirTest, MaterializesASymbolicIndexExprInTheSizePrologue) {
  mlir::MLIRContext context;
  ascir::FusedScheduledResult fusedResult;
  fusedResult.fused_graph_name = "elewise_index_expr";
  ascir::ScheduleGroup group;
  group.impl_graphs.push_back(MakeIndexExprGraph("s0 + 2", /*dynamicShape=*/true));
  ascir::ScheduledResult scheduled;
  scheduled.schedule_groups.push_back(std::move(group));
  fusedResult.node_idx_to_scheduled_results = {{scheduled}};

  auto module = mlir::afir::bridge::PlanToAfir(context, fusedResult, scheduled, 0, 0);
  ASSERT_TRUE(module);

  // 自由符号取成了 `afir.tiling_param "s0"`，不是被当成常量折掉。
  unsigned tilingParams = 0;
  module->walk([&](mlir::afir::TilingParamOp param) {
    if (param.getSymbol() == "s0") {
      ++tilingParams;
    }
  });
  EXPECT_EQ(tilingParams, 1U);

  unsigned evals = 0;
  module->walk([&](mlir::afir::SymbolicEvalOp) { ++evals; });
  EXPECT_GE(evals, 1U);

  // **它们必须在入口块的序言里**：块首那一段连续的 tiling_param / symbolic_eval。落到
  // 循环体里的话支配不了后面的消费者，而那是一次静默的 IR 破坏。
  mlir::func::FuncOp kernel;
  for (mlir::func::FuncOp function : module->getOps<mlir::func::FuncOp>()) {
    if (mlir::afir::hasFunctionRole(function, mlir::afir::FunctionRole::KernelBody)) {
      kernel = function;
    }
  }
  ASSERT_TRUE(kernel);
  bool sawParamInPrologue = false;
  for (mlir::Operation &op : kernel.front()) {
    if (!mlir::isa<mlir::afir::TilingParamOp, mlir::afir::SymbolicEvalOp>(op)) {
      break;
    }
    sawParamInPrologue |= mlir::isa<mlir::afir::TilingParamOp>(op);
  }
  EXPECT_TRUE(sawParamInPrologue);

  // 加数不是常量了——它是序言里算出来的那个值，经过一次到 f16 的转换。
  module->walk([&](mlir::afir::AddOp add) {
    EXPECT_FALSE(mlir::isa<mlir::arith::ConstantOp>(add.getRhs().getDefiningOp()));
    EXPECT_TRUE(mlir::isa<mlir::arith::SIToFPOp>(add.getRhs().getDefiningOp()));
  });
}

// 字面量解析。这一族的失败方式全是**静默给出另一个值**，所以每一条都得单独钉。
//
// 反面教材是 `StringRef::getAsInteger(unsigned, APInt&)`：无符号重载 + 「按字符串估出
// 来的最小位宽」，"8" 会变成 4 位的 `0b1000`，sext 到 i32 就是 -8；负号它根本不收，
// 而 `.Scalar("k", "-1", DT_INT32)` 是仓库里的现存写法。
namespace {
// 造一份只含「一个 Scalar + 一个 add」的最小 buffer 形态 plan，标量的值与 dtype 由参数
// 给。解析不了时 `BuildAfirModule` 返回空。
mlir::OwningOpRef<mlir::ModuleOp> BuildScalarPlan(mlir::MLIRContext &context, llvm::StringRef value,
                                                  llvm::StringRef dtype) {
  auto makeTensor = [&](mlir::afir::Space space, int64_t bufferId) {
    mlir::afir::bridge::TensorValueDTO tensor;
    tensor.tensor.shape = {128};
    tensor.tensor.shape_exprs = {"128"};
    tensor.tensor.dtype = dtype.str();
    tensor.tensor.axis = {0};
    tensor.tensor.repeats = {"128"};
    tensor.tensor.strides = {"1"};
    tensor.buffer.space = space;
    tensor.buffer.bufferId = bufferId;
    tensor.buffer.slotCount = space == mlir::afir::Space::UB ? 2 : -1;
    tensor.buffer.queueDepth = space == mlir::afir::Space::UB ? 2 : -1;
    tensor.buffer.sizeBytesExpr = "512";
    tensor.buffer.lifeTimeAxisId = -1;
    return tensor;
  };

  mlir::afir::bridge::PlanInfo plan;
  plan.name = "scalar_literal";
  mlir::afir::bridge::TemplateInfo tmpl;
  tmpl.name = "scalar_literal";
  tmpl.original_name = tmpl.name;
  tmpl.buffer_form = true;
  tmpl.axes.push_back({0, "z", "original", "128"});

  auto input0 = makeTensor(mlir::afir::Space::GM, 0);
  auto local0 = makeTensor(mlir::afir::Space::UB, 1);
  auto local1 = makeTensor(mlir::afir::Space::UB, 2);
  auto output = makeTensor(mlir::afir::Space::GM, 3);
  tmpl.inputs = {{"input0", input0, 0}};
  tmpl.outputs = {{"output", output, 0}};

  mlir::afir::bridge::NodeInfo load0;
  load0.name = "load0";
  load0.type = "Load";
  load0.input_names = {"input0"};
  load0.inputs = {input0};
  load0.output = local0;
  load0.sched_axis = {0};
  load0.loop_axis = 0;

  mlir::afir::bridge::NodeInfo scalar;
  scalar.name = "k";
  scalar.type = "Scalar";
  scalar.output = local0;
  scalar.scalar_value = value.str();

  mlir::afir::bridge::NodeInfo add;
  add.name = "add";
  add.type = "Add";
  add.input_names = {"load0", "k"};
  add.inputs = {local0, local0};
  add.output = local1;
  add.sched_axis = {0};
  add.loop_axis = 0;

  mlir::afir::bridge::NodeInfo store;
  store.name = "output";
  store.type = "Store";
  store.input_names = {"add"};
  store.inputs = {local1};
  store.output = output;
  store.sched_axis = {0};
  store.loop_axis = 0;

  tmpl.nodes = {load0, scalar, add, store};
  plan.templates.push_back(tmpl);
  return mlir::afir::bridge::BuildAfirModule(context, plan);
}

// 模块里那个常量的数值。没有常量时返回 nullopt。
std::optional<double> ScalarConstantValue(mlir::ModuleOp module) {
  std::optional<double> found;
  module.walk([&](mlir::arith::ConstantOp constant) {
    if (auto attr = mlir::dyn_cast<mlir::FloatAttr>(constant.getValue())) {
      found = attr.getValueAsDouble();
    } else if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue())) {
      found = static_cast<double>(integer.getInt());
    }
  });
  return found;
}
}  // namespace

TEST(ScheduleToAfirTest, ParsesScalarLiteralsAtTheirDeclaredType) {
  struct Accepted {
    llvm::StringLiteral value;
    llvm::StringLiteral dtype;
    double expected;
  };
  // legacy 那边这些写法都是直接拼进 C++ 的，一个都不能拒。
  constexpr Accepted kAccepted[] = {
      {"2.5", "float32", 2.5},
      {"-1.0", "float32", -1.0},
      {"2.00000000000000000000e+00", "float32", 2.0},
      {"0.00000000000000000", "float32", 0.0},
      // `f` 后缀：仓库里的现存写法。期望值写成 `double(998.998F)` 而不是 998.998——
      // f32 存不下后者，按目标语义解析出来的就该是舍入后的那个数。
      {"998.998f", "float32", static_cast<double>(998.998F)},
      {"0", "float32", 0.0},  // 浮点类型收整数字面量
      {"8", "int32", 8.0},    // 不是 -8
      {"9", "int32", 9.0},
      {"-1", "int32", -1.0},  // 负号
      {"-2", "int16", -2.0},
      {"+5", "int32", 5.0},
      {"32767", "int16", 32767.0},
      {"-32768", "int16", -32768.0},
  };
  for (const Accepted &accepted : kAccepted) {
    SCOPED_TRACE((accepted.value + ":" + accepted.dtype).str());
    mlir::MLIRContext context;
    auto module = BuildScalarPlan(context, accepted.value, accepted.dtype);
    ASSERT_TRUE(module);
    std::optional<double> value = ScalarConstantValue(*module);
    ASSERT_TRUE(value.has_value());
    EXPECT_DOUBLE_EQ(*value, accepted.expected);
  }
}

TEST(ScheduleToAfirTest, RejectsScalarLiteralsItCannotRepresent) {
  struct Rejected {
    llvm::StringLiteral value;
    llvm::StringLiteral dtype;
  };
  constexpr Rejected kRejected[] = {
      {"", "float32"},                                                // 空
      {"  ", "float32"},                                              // 只有空白
      {"inf", "float32"},                                             // legacy 拼进 C++ 也编不过
      {"nan", "float32"},   {"MY_MACRO", "float32"},                  // 宏名：legacy 能拼，这里不认
      {"1e39", "float32"},                                            // 溢出，不能悄悄变成 +Inf
      {"70000", "float16"}, {"2.5", "int32"},                         // 整数类型收不下小数
      {"32768", "int16"},                                             // 越界，不能截断
      {"-32769", "int16"},  {"100000", "int16"},     {"2", "uint8"},  // arith.constant 只收 signless
  };
  for (const Rejected &rejected : kRejected) {
    SCOPED_TRACE((rejected.value + ":" + rejected.dtype).str());
    mlir::MLIRContext context;
    // 这些都该出诊断，而不是产出一个别的值。
    context.getDiagEngine().registerHandler([](mlir::Diagnostic &) { return mlir::success(); });
    auto module = BuildScalarPlan(context, rejected.value, rejected.dtype);
    EXPECT_FALSE(module);
  }
}

TEST(ScheduleToAfirTest, FoldsScalarNodeIntoATypedConstant) {
  auto makeTensor = [](mlir::afir::Space space, int64_t bufferId) {
    mlir::afir::bridge::TensorValueDTO value;
    value.tensor.shape = {128};
    value.tensor.shape_exprs = {"128"};
    value.tensor.dtype = "float32";
    value.tensor.axis = {0};
    value.tensor.repeats = {"128"};
    value.tensor.strides = {"1"};
    value.buffer.space = space;
    value.buffer.bufferId = bufferId;
    value.buffer.slotCount = space == mlir::afir::Space::UB ? 2 : -1;
    value.buffer.queueDepth = space == mlir::afir::Space::UB ? 2 : -1;
    value.buffer.sizeBytesExpr = "512";
    value.buffer.lifeTimeAxisId = -1;
    return value;
  };

  mlir::MLIRContext context;
  mlir::afir::bridge::PlanInfo plan;
  plan.name = "scalar_add";

  mlir::afir::bridge::TemplateInfo tmpl;
  tmpl.name = "scalar_add";
  tmpl.original_name = tmpl.name;
  tmpl.buffer_form = true;
  tmpl.axes.push_back({0, "z", "original", "128"});

  auto input0 = makeTensor(mlir::afir::Space::GM, 0);
  auto local0 = makeTensor(mlir::afir::Space::UB, 1);
  auto local1 = makeTensor(mlir::afir::Space::UB, 2);
  auto output = makeTensor(mlir::afir::Space::GM, 3);
  tmpl.inputs = {{"input0", input0, 0}};
  tmpl.outputs = {{"output", output, 0}};

  mlir::afir::bridge::NodeInfo load0;
  load0.name = "load0";
  load0.type = "Load";
  load0.input_names = {"input0"};
  load0.inputs = {input0};
  load0.output = local0;
  load0.sched_axis = {0};
  load0.loop_axis = 0;

  // 标量节点自己没有 tile：没有输入，值是一串字面文本。
  mlir::afir::bridge::NodeInfo scalar;
  scalar.name = "two";
  scalar.type = "Scalar";
  scalar.output = local0;
  scalar.output.tensor.dtype = "float32";
  scalar.scalar_value = "2.5";

  mlir::afir::bridge::NodeInfo add;
  add.name = "add";
  add.type = "Add";
  add.input_names = {"load0", "two"};
  add.inputs = {local0, local0};
  add.output = local1;
  add.sched_axis = {0};
  add.loop_axis = 0;

  mlir::afir::bridge::NodeInfo store;
  store.name = "output";
  store.type = "Store";
  store.input_names = {"add"};
  store.inputs = {local1};
  store.output = output;
  store.sched_axis = {0};
  store.loop_axis = 0;

  tmpl.nodes = {load0, scalar, add, store};
  plan.templates.push_back(tmpl);

  auto module = mlir::afir::bridge::BuildAfirModule(context, plan);
  ASSERT_TRUE(module);

  // 字面量解析成了一个 f32 常量，不是原样搬过来的一串文本，也不是 0。
  unsigned constantCount = 0;
  module->walk([&](mlir::arith::ConstantOp constant) {
    auto attr = mlir::dyn_cast<mlir::FloatAttr>(constant.getValue());
    if (!attr || !attr.getType().isF32()) {
      return;
    }
    ++constantCount;
    EXPECT_DOUBLE_EQ(attr.getValueAsDouble(), 2.5);
  });
  EXPECT_EQ(constantCount, 1U);

  // 它成了 `afir.add` 的第二个操作数，而不是又一块 tile。
  unsigned addCount = 0;
  module->walk([&](mlir::afir::AddOp add) {
    ++addCount;
    EXPECT_FALSE(mlir::isa<mlir::ShapedType>(add.getRhs().getType()));
    EXPECT_TRUE(mlir::isa<mlir::arith::ConstantOp>(add.getRhs().getDefiningOp()));
    EXPECT_TRUE(mlir::isa<mlir::ShapedType>(add.getLhs().getType()));
  });
  EXPECT_EQ(addCount, 1U);
}

TEST(ScheduleToAfirTest, AcceptsSchedulerTiledTransferWithDifferentLogicalAndLocalShapes) {
  auto makeExternalTensor = [](int64_t bufferId) {
    mlir::afir::bridge::TensorValueDTO value;
    value.tensor.dtype = "float32";
    value.buffer.space = mlir::afir::Space::GM;
    value.buffer.bufferId = bufferId;
    value.buffer.sizeBytesExpr = "25600";
    value.buffer.lifeTimeAxisId = -1;
    return value;
  };
  auto makeLocalTensor = [](int64_t bufferId) {
    mlir::afir::bridge::TensorValueDTO value;
    value.tensor.shape = {mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic};
    value.tensor.shape_exprs = {"(6400 / (a0Tb_size * a0t_size))", "a0Tb_size", "a0t_size"};
    value.tensor.dtype = "float32";
    value.tensor.axis = {3, 4, 2};
    value.tensor.repeats = value.tensor.shape_exprs;
    value.tensor.strides = {"(a0Tb_size * a0t_size)", "a0t_size", "1"};
    value.buffer.space = mlir::afir::Space::UB;
    value.buffer.bufferId = bufferId;
    value.buffer.slotCount = 2;
    value.buffer.queueDepth = 2;
    value.buffer.sizeBytesExpr = "(4 * a0t_size)";
    value.buffer.lifeTimeAxisId = -1;
    return value;
  };

  mlir::MLIRContext context;
  mlir::afir::bridge::PlanInfo plan;
  plan.name = "scheduler_tiled_transfer";

  mlir::afir::bridge::TemplateInfo tmpl;
  tmpl.name = plan.name;
  tmpl.original_name = tmpl.name;
  tmpl.buffer_form = true;
  tmpl.size_vars = {"a0t_size", "a0Tb_size"};
  tmpl.axes.push_back({2, "a0t", "tile_in", "a0t_size"});
  tmpl.axes.push_back({3, "a0TB", "block_out", "(6400 / (a0Tb_size * a0t_size))"});
  tmpl.axes.push_back({4, "a0Tb", "block_in", "a0Tb_size"});

  auto input = makeExternalTensor(0);
  auto local = makeLocalTensor(1);
  auto output = makeExternalTensor(2);
  tmpl.inputs = {{"input", input, 0}};
  tmpl.outputs = {{"output", output, 0}};

  mlir::afir::bridge::NodeInfo load;
  load.name = "load";
  load.type = "Load";
  load.input_names = {"input"};
  load.inputs = {input};
  load.output = local;
  load.sched_axis = {3, 4, 2};
  load.loop_axis = 4;

  mlir::afir::bridge::NodeInfo store;
  store.name = "output";
  store.type = "Store";
  store.input_names = {"load"};
  store.inputs = {local};
  store.output = output;
  store.sched_axis = {3, 4, 2};
  store.loop_axis = 4;

  tmpl.nodes = {load, store};
  plan.templates.push_back(tmpl);

  auto module = mlir::afir::bridge::BuildAfirModule(context, plan);
  ASSERT_TRUE(module);
  EXPECT_TRUE(succeeded(mlir::verify(module->getOperation())));
  EXPECT_TRUE(module->lookupSymbol<mlir::func::FuncOp>(plan.name));

  // 动态维的范围就是调度器给的那个 shape 表达式，不是占位常量。
  //
  // 这一条盯住的是一个不会让任何用例变红的错误：`memref.alloc` 的动态操作数在 AscendC
  // 物化路径上没有读点（队列字节数来自 `size_bytes_expr`，成员尺寸来自 tensor
  // descriptor），所以此前那个 `arith.constant 1` 生成的 kernel 完全正确——它只是让 IR
  // 里写着「这块缓冲运行期为 1」。判据落在结构上：三个动态维分别由三个 tiling 参数
  // 表达式算出，其中两个是单符号（直接取 `afir.tiling_param`），一个是复合表达式
  // （`afir.symbolic_eval`）。
  llvm::SmallVector<mlir::memref::AllocOp> allocations;
  module->walk([&](mlir::memref::AllocOp alloc) { allocations.push_back(alloc); });
  ASSERT_EQ(allocations.size(), 1U);
  EXPECT_EQ(allocations.front().getDynamicSizes().size(), 3U);

  llvm::StringSet<> tilingParams;
  module->walk([&](mlir::afir::TilingParamOp param) { tilingParams.insert(param.getSymbol()); });
  EXPECT_EQ(tilingParams.size(), 2U);
  EXPECT_TRUE(tilingParams.contains("a0t_size"));
  EXPECT_TRUE(tilingParams.contains("a0Tb_size"));

  for (mlir::Value size : allocations.front().getDynamicSizes()) {
    mlir::Operation *definition = size.getDefiningOp();
    ASSERT_NE(definition, nullptr);
    EXPECT_TRUE((mlir::isa<mlir::afir::TilingParamOp, mlir::afir::SymbolicEvalOp>(definition)))
        << "dynamic extent is defined by " << definition->getName().getStringRef().str();
    // 尺寸序言在入口块里，因此支配 alloc；否则 verifier 早就红了，这里钉的是「它没有
    // 被放进某个循环体」——那种形态只在循环生成之后才会暴露。
    EXPECT_EQ(definition->getBlock(), allocations.front()->getBlock());
  }
}

// `exec_condition` 从 DTO 到 `afir.sched` 的那一段。三格：认得的拼写落到属性上、认不得的
// 拼写报错、没有 `sched_axis` 时报错。
//
// **不认得的拼写必须报错**，因为 `no_cache` 是缺省值：静默落回缺省的表现是每个迭代都重算
// ——不错，但上游那次决策在 IR 里一点痕迹都不剩，往后既没有诊断也没有性能数字能发现它。
// 第三格同理：没有 `sched_axis` 就不会建 `afir.sched`，这个条件没有地方可挂。
TEST(ScheduleToAfirTest, CarriesExecConditionOntoTheScheduleAttribute) {
  auto buildPlan = [](mlir::MLIRContext &context, const std::string &execCondition, bool withSchedAxis) {
    mlir::afir::bridge::TensorValueDTO external;
    external.tensor.dtype = "float32";
    external.buffer.space = mlir::afir::Space::GM;
    external.buffer.sizeBytesExpr = "25600";
    external.buffer.lifeTimeAxisId = -1;

    mlir::afir::bridge::TensorValueDTO local;
    local.tensor.shape = {mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic};
    local.tensor.shape_exprs = {"(6400 / (a0Tb_size * a0t_size))", "a0Tb_size", "a0t_size"};
    local.tensor.dtype = "float32";
    local.tensor.axis = {3, 4, 2};
    local.tensor.repeats = local.tensor.shape_exprs;
    local.tensor.strides = {"(a0Tb_size * a0t_size)", "a0t_size", "1"};
    local.buffer.space = mlir::afir::Space::UB;
    local.buffer.bufferId = 1;
    local.buffer.slotCount = 1;
    local.buffer.queueDepth = 2;
    local.buffer.sizeBytesExpr = "(4 * a0t_size)";
    local.buffer.lifeTimeAxisId = -1;

    mlir::afir::bridge::PlanInfo plan;
    plan.name = "exec_condition_bridge";

    mlir::afir::bridge::TemplateInfo tmpl;
    tmpl.name = plan.name;
    tmpl.original_name = tmpl.name;
    tmpl.buffer_form = true;
    tmpl.size_vars = {"a0t_size", "a0Tb_size"};
    tmpl.axes.push_back({2, "a0t", "tile_in", "a0t_size"});
    tmpl.axes.push_back({3, "a0TB", "block_out", "(6400 / (a0Tb_size * a0t_size))"});
    tmpl.axes.push_back({4, "a0Tb", "block_in", "a0Tb_size"});

    mlir::afir::bridge::TensorValueDTO output = external;
    output.buffer.bufferId = 2;
    external.buffer.bufferId = 0;
    tmpl.inputs = {{"input", external, 0}};
    tmpl.outputs = {{"output", output, 0}};

    mlir::afir::bridge::NodeInfo load;
    load.name = "load";
    load.type = "Load";
    load.input_names = {"input"};
    load.inputs = {external};
    load.output = local;
    load.exec_condition = execCondition;
    if (withSchedAxis) {
      load.sched_axis = {3, 4, 2};
      load.loop_axis = 4;
    }

    mlir::afir::bridge::NodeInfo store;
    store.name = "output";
    store.type = "Store";
    store.input_names = {"load"};
    store.inputs = {local};
    store.output = output;
    store.sched_axis = {3, 4, 2};
    store.loop_axis = 4;

    tmpl.nodes = {load, store};
    plan.templates.push_back(tmpl);
    return mlir::afir::bridge::BuildAfirModule(context, plan);
  };

  {
    mlir::MLIRContext context;
    auto module = buildPlan(context, "cache_block_split_fused_brc_axis", /*withSchedAxis=*/true);
    ASSERT_TRUE(module);
    EXPECT_TRUE(succeeded(mlir::verify(module->getOperation())));
    mlir::afir::ScheduleAttr loadSchedule;
    module->walk([&](mlir::afir::LoadOp load) {
      loadSchedule = load->getAttrOfType<mlir::afir::ScheduleAttr>(mlir::afir::kSchedAttrName);
    });
    ASSERT_TRUE(loadSchedule);
    EXPECT_EQ(loadSchedule.getExecCondition(), mlir::afir::ExecCondition::CacheBlockSplitFusedBroadcastAxis);
    // Store 上游没有这个条件，它必须还是缺省值——守卫包错算子的表现是少写一片输出。
    mlir::afir::ScheduleAttr storeSchedule;
    module->walk([&](mlir::afir::StoreOp store) {
      storeSchedule = store->getAttrOfType<mlir::afir::ScheduleAttr>(mlir::afir::kSchedAttrName);
    });
    ASSERT_TRUE(storeSchedule);
    EXPECT_EQ(storeSchedule.getExecCondition(), mlir::afir::ExecCondition::NoCache);
  }

  {
    mlir::MLIRContext context;
    std::string message;
    mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &diagnostic) {
      llvm::raw_string_ostream output(message);
      diagnostic.print(output);
    });
    EXPECT_FALSE(buildPlan(context, "cache_block_split_fused_brc_axes", /*withSchedAxis=*/true));
    EXPECT_NE(message.find("unknown exec_condition"), std::string::npos) << message;
  }

  {
    mlir::MLIRContext context;
    std::string message;
    mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &diagnostic) {
      llvm::raw_string_ostream output(message);
      diagnostic.print(output);
    });
    EXPECT_FALSE(buildPlan(context, "cache_block_split_fused_brc_axis", /*withSchedAxis=*/false));
    EXPECT_NE(message.find("requires a scheduled axis list"), std::string::npos) << message;
  }
}

TEST(ScheduleToAfirTest, BufferFormStoreUsesInputTensorSchedulingAttributes) {
  mlir::MLIRContext context;
  mlir::afir::bridge::PlanInfo plan;
  plan.name = "buffer_store_schedule_attrs";

  mlir::afir::bridge::TemplateInfo tmpl;
  tmpl.name = plan.name;
  tmpl.original_name = tmpl.name;
  tmpl.buffer_form = true;

  auto makeTensor = [](mlir::afir::Space space, int64_t bufferId, int64_t vectorizedAxis) {
    mlir::afir::bridge::TensorValueDTO value;
    value.tensor.shape = {8};
    value.tensor.dtype = "float16";
    value.tensor.axis = {0};
    value.tensor.repeats = {"8"};
    value.tensor.strides = {"1"};
    value.tensor.vectorized_axis = {vectorizedAxis};
    value.tensor.vectorized_strides = {"1"};
    value.buffer.space = space;
    value.buffer.bufferId = bufferId;
    value.buffer.sizeBytesExpr = "16";
    value.buffer.lifeTimeAxisId = -1;
    return value;
  };

  auto input = makeTensor(mlir::afir::Space::UB, 0, 0);
  auto output = makeTensor(mlir::afir::Space::GM, 1, 1);
  tmpl.inputs = {{"input", input, 0}};
  tmpl.outputs = {{"output", output, 0}};

  mlir::afir::bridge::NodeInfo store;
  store.name = "output";
  store.type = "Store";
  store.input_names = {"input"};
  store.inputs = {input};
  store.output = output;
  tmpl.nodes = {store};
  plan.templates.push_back(tmpl);

  auto module = mlir::afir::bridge::BuildAfirModule(context, plan);
  ASSERT_TRUE(module);

  mlir::afir::StoreOp storeOp;
  module->walk([&](mlir::afir::StoreOp op) { storeOp = op; });
  ASSERT_TRUE(storeOp);
  // 步 3b 起不再产生 `afir.vectorized_axis` 这份影子拷贝：Store 的向量化轴就是它
  // 局部侧（操作数，UB）descriptor 的 vectorized_axis。这里断言的仍是同一件事——
  // 取的是输入的 0 而不是输出的 1——只是读的是唯一的那处事实。
  EXPECT_FALSE(storeOp->hasAttr("afir.vectorized_axis"));
  auto operandDescs = storeOp->getAttrOfType<mlir::ArrayAttr>(mlir::afir::kOperandDescsAttrName);
  ASSERT_TRUE(operandDescs);
  ASSERT_EQ(operandDescs.size(), 1U);
  auto operandDesc = mlir::dyn_cast<mlir::afir::TensorDescAttr>(operandDescs[0]);
  ASSERT_TRUE(operandDesc);
  ASSERT_EQ(operandDesc.getVectorizedAxis().size(), 1U);
  EXPECT_EQ(operandDesc.getVectorizedAxis()[0], 0);
}

// 多个 Store/Output 节点此前一个用例都没有：所有测试图都是单 Store 单 Output，
// `CodegenOrchestrationSt` 甚至写死 `output_nodes.size() == 1`。这里钉住两件事：
// bridge 把每个 Output 节点都带出来（而不是只带第一个），以及 kernel 参数的 ABI
// 顺序由 io index 决定、与图的遍历顺序无关。
TEST(ScheduleToAfirTest, BridgesEveryGraphOutputAndOrdersAbiByIoIndex) {
  mlir::MLIRContext context;
  ascir::FusedScheduledResult fusedResult;
  fusedResult.fused_graph_name = "two_outputs";
  fusedResult.node_idx_to_scheduled_results = {{MakeTwoOutputScheduledResult()}};

  auto module =
      mlir::afir::bridge::PlanToAfir(context, fusedResult, fusedResult.node_idx_to_scheduled_results[0][0], 0, 0);
  ASSERT_TRUE(module);

  mlir::func::FuncOp function;
  unsigned functionCount = 0;
  for (mlir::func::FuncOp candidate : module->getOps<mlir::func::FuncOp>()) {
    function = candidate;
    ++functionCount;
  }
  ASSERT_EQ(functionCount, 1U);
  ASSERT_TRUE(function);

  // buffer 形态下输入和输出都是参数。若 AppendNodeIo 只带出第一个 Output 节点，
  // 这里会是 3 而不是 4。
  ASSERT_EQ(function.getNumArguments(), 4U);
  EXPECT_EQ(function.getNumResults(), 0U);

  llvm::SmallVector<mlir::ascend::kernel::KernelArgumentIdentity> identities;
  for (unsigned index = 0; index < function.getNumArguments(); ++index) {
    auto identity = mlir::ascend::kernel::getKernelArgumentIdentity(function, index);
    ASSERT_TRUE(succeeded(identity));
    identities.push_back(*identity);
  }
  EXPECT_EQ(identities[0].kind, mlir::ascend::kernel::KernelArgumentKind::Input);
  EXPECT_EQ(identities[0].index, 0);
  EXPECT_EQ(identities[1].kind, mlir::ascend::kernel::KernelArgumentKind::Input);
  EXPECT_EQ(identities[1].index, 1);
  // bridge 不重排：遍历先到的 "hi" 留在前面，它的 io index 是 1。
  EXPECT_EQ(identities[2].kind, mlir::ascend::kernel::KernelArgumentKind::Output);
  EXPECT_EQ(identities[2].index, 1);
  EXPECT_EQ(identities[3].kind, mlir::ascend::kernel::KernelArgumentKind::Output);
  EXPECT_EQ(identities[3].index, 0);

  // ABI 层按 (kind, index) 重排，所以 index 0 的输出必须排到 index 1 之前。
  auto permutation = mlir::ascend::kernel::getKernelArgumentPermutation(identities);
  ASSERT_EQ(permutation.size(), 4U);
  EXPECT_EQ(permutation[0], 0U);
  EXPECT_EQ(permutation[1], 1U);
  EXPECT_EQ(permutation[2], 3U);
  EXPECT_EQ(permutation[3], 2U);
}

// 两个 Output 节点共用同一个 io index 必须被拦下，否则 ABI 排序的键不唯一，
// 两个输出会按发现顺序静默决定先后。
TEST(ScheduleToAfirTest, RejectsDuplicateOutputIoIndex) {
  auto graph = af::testing::AscGraphBuilder("duplicate_output_index")
                   .Loops({af::testing::Sym(8), af::testing::Sym(8)})
                   .Data("x", 0, af::DT_FLOAT16)
                   .Load("load_x", "x")
                   .Store("store_a", "load_x")
                   .Output("a", "store_a", 0, af::DT_FLOAT16)
                   .Store("store_b", "load_x")
                   .Output("b", "store_b", 0, af::DT_FLOAT16)
                   .Build();
  AssignTensorIds(graph);

  ascir::ScheduleGroup group;
  group.impl_graphs.push_back(std::move(graph));
  ascir::ScheduledResult scheduledResult;
  scheduledResult.schedule_groups.push_back(std::move(group));

  ascir::FusedScheduledResult fusedResult;
  fusedResult.fused_graph_name = "duplicate_output_index";
  fusedResult.node_idx_to_scheduled_results = {{std::move(scheduledResult)}};

  mlir::MLIRContext context;
  std::string message;
  mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &diagnostic) {
    llvm::raw_string_ostream output(message);
    diagnostic.print(output);
  });
  auto module =
      mlir::afir::bridge::PlanToAfir(context, fusedResult, fusedResult.node_idx_to_scheduled_results[0][0], 0, 0);
  EXPECT_FALSE(module);
  EXPECT_NE(message.find("duplicate output index"), std::string::npos);
}

// 一条队列上的两个成员：**元素类型不同是合法调度，descriptor 字段不一致才拒收。**
//
// 前一半是「一条队列上不能混 dtype」那条限制的去处。队列就是一块字节，legacy 的
// `oneBufSize` 取成员的 max，而片上资源的字节数由 ATT 填进 `q<id>_size`
// （`size_from_resource`），descriptor 上没有一处依赖 dtype。最常见的形状是 ui8 掩码与
// 两边的浮点一起搬入——E2E `load_mask_select_store` 钉的是它的数值。
//
// 后一半拿 `queue_depth` 做判据：它说的是这条队列轮转几块，两个成员各说一个数就是
// 矛盾。两半查的是同一份字段清单（`afir::canonicalResourceFactsAgree`）。
//
// 三遍跑同一个图，只差第二个成员的 dtype 与 depth——所以「拒收」不会是别的原因造成的：
// 前两遍必须建得出模块。图里两个 Load 各自直连一个 Store，谁都不喂同一个计算节点，
// 免得 dtype 一改就撞上算子自己的类型判据（那样拦住它的就不是这里要验的那条）。
TEST(ScheduleToAfirTest, SharesOneQueueAcrossDtypesButRejectsIncompatibleDescriptors) {
  auto buildResult = [](af::DataType secondDtype, int64_t secondQueueDepth, int64_t secondPitch = 16) {
    auto graph = af::testing::AscGraphBuilder("shared_queue_descriptors")
                     .Loops({af::testing::Sym(8), af::testing::Sym(8)})
                     .Data("x", 0, af::DT_FLOAT16)
                     .Data("b", 1, secondDtype)
                     .Load("load_x", "x")
                     .Load("load_b", "b")
                     .Store("store_x", "load_x")
                     .Output("y0", "store_x", 0, af::DT_FLOAT16)
                     .Store("store_b", "load_b")
                     .Output("y1", "store_b", 1, secondDtype)
                     .Build();
    AssignTensorIds(graph);
    graph.SetTilingKey(0);
    for (const auto &node : graph.GetAllNodes()) {
      const bool second = node->GetName() == "load_b";
      if (node->GetName() != "load_x" && !second) {
        continue;
      }
      for (const auto &output : node->outputs()) {
        auto &attr = output->attr;
        attr.mem.position = af::Position::kPositionVecIn;
        attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
        attr.mem.reuse_id = 0;
        attr.que.id = 0;
        attr.que.buf_num = 1;
        attr.que.depth = second ? secondQueueDepth : 1;
        attr.vectorized_axis = {0, 1};
        attr.vectorized_strides = {af::Symbol(second ? secondPitch : 16), af::Symbol(1)};
      }
    }

    ascir::ScheduleGroup group;
    group.impl_graphs.push_back(std::move(graph));
    ascir::ScheduledResult scheduledResult;
    scheduledResult.schedule_groups.push_back(std::move(group));
    ascir::FusedScheduledResult fusedResult;
    fusedResult.fused_graph_name = "shared_queue_descriptors";
    fusedResult.node_idx_to_scheduled_results = {{std::move(scheduledResult)}};
    return fusedResult;
  };

  mlir::MLIRContext context;
  auto sameDtype = buildResult(af::DT_FLOAT16, 1);
  EXPECT_TRUE(mlir::afir::bridge::PlanToAfir(context, sameDtype, sameDtype.node_idx_to_scheduled_results[0][0], 0, 0));

  auto mixedDtype = buildResult(af::DT_UINT8, 1);
  EXPECT_TRUE(
      mlir::afir::bridge::PlanToAfir(context, mixedDtype, mixedDtype.node_idx_to_scheduled_results[0][0], 0, 0));

  auto mixedPitch = buildResult(af::DT_UINT8, 1, 32);
  EXPECT_TRUE(
      mlir::afir::bridge::PlanToAfir(context, mixedPitch, mixedPitch.node_idx_to_scheduled_results[0][0], 0, 0));

  auto conflicting = buildResult(af::DT_UINT8, 2);
  std::string message;
  mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &diagnostic) {
    llvm::raw_string_ostream output(message);
    diagnostic.print(output);
  });
  EXPECT_FALSE(
      mlir::afir::bridge::PlanToAfir(context, conflicting, conflicting.node_idx_to_scheduled_results[0][0], 0, 0));
  EXPECT_NE(message.find("BindingConflict"), std::string::npos) << message;
  auto sameDtypeBadPitch = buildResult(af::DT_FLOAT16, 1, 32);
  message.clear();
  EXPECT_FALSE(mlir::afir::bridge::PlanToAfir(context, sameDtypeBadPitch,
                                              sameDtypeBadPitch.node_idx_to_scheduled_results[0][0], 0, 0));
  EXPECT_NE(message.find("BindingConflict"), std::string::npos) << message;
  EXPECT_NE(message.find("incompatible tensor layout"), std::string::npos) << message;
}

// 同一条队列上「广播来的 tile」与「整形状 tile」并存：收不了，但那是**能力不足**，
// 不是调度坏了。
//
// 上面那条用例拿 `queue_depth` 做判据，差的是资源字段；这一条差的是**张量视图**
// （`repeats` / `strides`），而视图对不上此前一律判成 `BindingConflict` 硬失败。
// 判据的前提是「同一条队列上的张量今天必然是同一个 tile」，而广播让它不成立：
// stride 为 0 的那根轴上，广播那一侧的 Load 不前进，tile 就比整形状那一侧小。
// 实测 `add(brc(x), y)` 在 `[1,8] -> [128,8]` 上就是这个形状（探针 T3，2026-10-12），
// 而硬失败不会回落 legacy，整条编译就断在这里。
//
// 两条断言成对：差别能由 stride 0 解释 → `UnsupportedOperation` 且带能力不足标记；
// 解释不了（这里把广播那一侧的 stride 改成非 0，于是两张视图是真的对不上）→ 仍旧
// `BindingConflict`。少了后一条，这条判据就退化成「视图对不上一律回落」，真 bug 会被藏起来。
TEST(ScheduleToAfirTest, BroadcastTileSharingOneQueueIsCapabilityShortfallNotConflict) {
  auto buildResult = [](int64_t broadcastStride) {
    auto graph = af::testing::AscGraphBuilder("shared_queue_broadcast_tile")
                     .Loops({af::testing::Sym(8), af::testing::Sym(8)})
                     .Data("x", 0, af::DT_FLOAT16)
                     .Data("b", 1, af::DT_FLOAT16)
                     .Load("load_x", "x")
                     .Load("load_b", "b")
                     .Store("store_x", "load_x")
                     .Output("y0", "store_x", 0, af::DT_FLOAT16)
                     .Store("store_b", "load_b")
                     .Output("y1", "store_b", 1, af::DT_FLOAT16)
                     .Build();
    AssignTensorIds(graph);
    graph.SetTilingKey(0);
    for (const auto &node : graph.GetAllNodes()) {
      const bool second = node->GetName() == "load_b";
      if (node->GetName() != "load_x" && !second) {
        continue;
      }
      for (const auto &output : node->outputs()) {
        auto &attr = output->attr;
        attr.mem.position = af::Position::kPositionVecIn;
        attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
        attr.mem.reuse_id = 0;
        attr.que.id = 0;
        attr.que.buf_num = 1;
        attr.que.depth = 1;
        if (!second) {
          continue;
        }
        // 第二个成员的最外轴收成一格：repeats 1、stride 取参数。stride 为 0 时这就是
        // 广播来的 tile，非 0 时就只是一张对不上的视图。
        if (!attr.repeats.empty()) {
          attr.repeats[0] = af::Symbol(1);
        }
        if (!attr.strides.empty()) {
          attr.strides[0] = af::Symbol(broadcastStride);
        }
      }
    }

    ascir::ScheduleGroup group;
    group.impl_graphs.push_back(std::move(graph));
    ascir::ScheduledResult scheduledResult;
    scheduledResult.schedule_groups.push_back(std::move(group));
    ascir::FusedScheduledResult fusedResult;
    fusedResult.fused_graph_name = "shared_queue_broadcast_tile";
    fusedResult.node_idx_to_scheduled_results = {{std::move(scheduledResult)}};
    return fusedResult;
  };

  auto run = [](ascir::FusedScheduledResult &result, std::string &message, bool &shortfall) {
    mlir::MLIRContext context;
    mlir::afir::CapabilityShortfallObserver observer(context);
    mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &diagnostic) {
      llvm::raw_string_ostream output(message);
      diagnostic.print(output);
      return mlir::failure();
    });
    const bool built = static_cast<bool>(
        mlir::afir::bridge::PlanToAfir(context, result, result.node_idx_to_scheduled_results[0][0], 0, 0));
    shortfall = observer.sawShortfall();
    return built;
  };

  // stride 0：广播能解释，报能力不足、带标记，于是上层回落 legacy。
  auto broadcastTile = buildResult(0);
  std::string broadcastMessage;
  bool broadcastShortfall = false;
  EXPECT_FALSE(run(broadcastTile, broadcastMessage, broadcastShortfall));
  EXPECT_TRUE(broadcastShortfall) << broadcastMessage;
  EXPECT_NE(broadcastMessage.find("broadcast tile and a full-shape tile"), std::string::npos) << broadcastMessage;

  // stride 非 0：解释不了，仍旧是硬失败，且**不带**能力不足标记。
  auto mismatchedTile = buildResult(3);
  std::string mismatchedMessage;
  bool mismatchedShortfall = false;
  EXPECT_FALSE(run(mismatchedTile, mismatchedMessage, mismatchedShortfall));
  EXPECT_FALSE(mismatchedShortfall) << mismatchedMessage;
  EXPECT_NE(mismatchedMessage.find("BindingConflict"), std::string::npos) << mismatchedMessage;
}

// 非队列 buffer 上「广播 tile + full-shape tile」是时间复用，不是资源冲突。
//
// 这与上一条队列用例只差 `alloc_type` / `buf.id`：`reuse_id = -1` 的两个成员都属于
// 同一个非队列 `b<id>`，materializer 现在按各自 `view.offset` 起算，偏移都是 0，也就是
// 先后写同一份存储。diamond 图 `brc -> add -> {abs, neg} -> add -> store` 在真实 Inductor
// 上正是这个形状；若这里仍拒收，strict MLIR route 会在编译期直接失败。
TEST(ScheduleToAfirTest, BroadcastTileSharingNonQueueBufferIsSupported) {
  auto graph = af::testing::AscGraphBuilder("shared_buffer_broadcast_tile")
                   .Loops({af::testing::Sym(8), af::testing::Sym(8)})
                   .Data("x", 0, af::DT_FLOAT16)
                   .Data("b", 1, af::DT_FLOAT16)
                   .Load("load_x", "x")
                   .Load("load_b", "b")
                   .Store("store_x", "load_x")
                   .Output("y0", "store_x", 0, af::DT_FLOAT16)
                   .Store("store_b", "load_b")
                   .Output("y1", "store_b", 1, af::DT_FLOAT16)
                   .Build();
  AssignTensorIds(graph);
  graph.SetTilingKey(0);
  for (const auto &node : graph.GetAllNodes()) {
    // 真实 Inductor 的 scalar `Data` 节点没有 GM 轴；保持这个事实，才能让下面的 Load
    // 成为 scheduler-tiled transfer，而不是被静态 verifier 误判成 GM 形状缺失。
    if (node->GetType() == "Data") {
      node->outputs()[0]->attr.axis.clear();
    }
    if (node->GetType() == "Store" || node->GetType() == "Output") {
      // GM 输出对齐各自输入 tile，避免这个 UT 变成在测 transfer verifier；它要钉的是：
      // 同一非队列 b0 上的一张 broadcast tile 与一张 full-shape tile 可以通过资源
      // compatibility。
      const bool broadcastOutput = node->GetName() == "store_b" || node->GetName() == "y1";
      auto &tensor = node->outputs()[0]->attr;
      tensor.axis = {0, 1};
      tensor.repeats = {af::Symbol(broadcastOutput ? 1 : 8), af::Symbol(8)};
      tensor.strides = {af::Symbol(broadcastOutput ? 0 : 8), af::Symbol(1)};
      continue;
    }

    const bool second = node->GetName() == "load_b";
    if (node->GetName() != "load_x" && !second) {
      continue;
    }
    // 这两个 Load 是调度切分出的 GM→UB tile；没有 sched 时，静态 load verifier 会把
    // scalar GM 与 1x8 UB tile 当成真正的元素数 mismatch。
    node->attr.sched.axis = {0, 1};
    node->attr.sched.loop_axis = 1;
    for (const auto &output : node->outputs()) {
      auto &attr = output->attr;
      attr.mem.position = af::Position::kPositionVecCalc;
      attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
      attr.mem.reuse_id = -1;
      attr.buf.id = 0;
      if (second) {
        // 广播侧 tile：最外轴只有一格，stride 为 0。
        if (!attr.repeats.empty()) {
          attr.repeats[0] = af::Symbol(1);
        }
        if (!attr.strides.empty()) {
          attr.strides[0] = af::Symbol(0);
        }
      }
    }
  }

  ascir::ScheduleGroup group;
  group.impl_graphs.push_back(std::move(graph));
  ascir::ScheduledResult scheduledResult;
  scheduledResult.schedule_groups.push_back(std::move(group));
  ascir::FusedScheduledResult fusedResult;
  fusedResult.fused_graph_name = "shared_buffer_broadcast_tile";
  fusedResult.node_idx_to_scheduled_results = {{std::move(scheduledResult)}};

  mlir::MLIRContext context;
  std::string message;
  mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &diagnostic) {
    llvm::raw_string_ostream output(message);
    diagnostic.print(output);
    return mlir::failure();
  });
  auto module =
      mlir::afir::bridge::PlanToAfir(context, fusedResult, fusedResult.node_idx_to_scheduled_results[0][0], 0, 0);
  ASSERT_TRUE(module) << message;
  EXPECT_EQ(message.find("BindingConflict"), std::string::npos) << message;
  EXPECT_EQ(message.find("broadcast tile and a full-shape tile"), std::string::npos) << message;
}

struct InnerBroadcastPlanOutcome {
  bool succeeded;
  std::string message;
};

InnerBroadcastPlanOutcome RunSharedBufferInnerBroadcastPlan(const std::string &name, int64_t fullOuterStride) {
  auto graph = af::testing::AscGraphBuilder(name)
                   .Loops({af::testing::Sym(8), af::testing::Sym(128), af::testing::Sym(1)})
                   .Data("x", 0, af::DT_FLOAT16)
                   .Data("b", 1, af::DT_FLOAT16)
                   .Load("load_x", "x")
                   .Load("load_b", "b")
                   .Store("store_x", "load_x")
                   .Output("y0", "store_x", 0, af::DT_FLOAT16)
                   .Store("store_b", "load_b")
                   .Output("y1", "store_b", 1, af::DT_FLOAT16)
                   .Build();
  AssignTensorIds(graph);
  graph.SetTilingKey(0);
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetType() == "Data") {
      node->outputs()[0]->attr.axis.clear();
    }
    if (node->GetType() == "Store" || node->GetType() == "Output") {
      const bool broadcastOutput = node->GetName() == "store_x" || node->GetName() == "y0";
      auto &tensor = node->outputs()[0]->attr;
      tensor.axis = {0, 1, 2};
      tensor.repeats = {af::Symbol(8), af::Symbol(broadcastOutput ? 1 : 128), af::Symbol(1)};
      tensor.strides = {af::Symbol(broadcastOutput ? 1 : fullOuterStride), af::Symbol(broadcastOutput ? 0 : 1),
                        af::Symbol(1)};
      continue;
    }

    const bool broadcastLoad = node->GetName() == "load_x";
    if (node->GetName() != "load_x" && node->GetName() != "load_b") {
      continue;
    }
    node->attr.sched.axis = {0, 1, 2};
    node->attr.sched.loop_axis = 2;
    for (const auto &output : node->outputs()) {
      auto &attr = output->attr;
      attr.mem.position = af::Position::kPositionVecCalc;
      attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
      attr.mem.reuse_id = -1;
      attr.buf.id = 0;
      attr.axis = {0, 1, 2};
      attr.repeats = {af::Symbol(8), af::Symbol(broadcastLoad ? 1 : 128), af::Symbol(1)};
      attr.strides = {af::Symbol(broadcastLoad ? 1 : fullOuterStride), af::Symbol(broadcastLoad ? 0 : 1),
                      af::Symbol(1)};
    }
  }

  ascir::ScheduleGroup group;
  group.impl_graphs.push_back(std::move(graph));
  ascir::ScheduledResult scheduledResult;
  scheduledResult.schedule_groups.push_back(std::move(group));
  ascir::FusedScheduledResult fusedResult;
  fusedResult.fused_graph_name = name.c_str();
  fusedResult.node_idx_to_scheduled_results = {{std::move(scheduledResult)}};

  mlir::MLIRContext context;
  std::string message;
  mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &diagnostic) {
    llvm::raw_string_ostream output(message);
    diagnostic.print(output);
    return mlir::failure();
  });
  auto module =
      mlir::afir::bridge::PlanToAfir(context, fusedResult, fusedResult.node_idx_to_scheduled_results[0][0], 0, 0);
  return {static_cast<bool>(module), message};
}

// 内层广播会让小 tile 的外层行距也变成 1，而 full-shape tile 的外层行距是 128。
// 这不是资源冲突：非队列 buffer 上的两个成员先后复用同一块存储，各自按自己的视图
// 读写。该用例钉住 `DifferenceIsExplainedByBroadcast` 对“非广播外轴 stride 差异 +
// 后续内层广播轴”的识别，避免 Form-B diamond 被误报 BindingConflict。
TEST(ScheduleToAfirTest, InnerBroadcastTileSharingNonQueueBufferIsSupported) {
  const auto outcome = RunSharedBufferInnerBroadcastPlan("shared_buffer_inner_broadcast_tile", 128);
  ASSERT_TRUE(outcome.succeeded) << outcome.message;
  EXPECT_EQ(outcome.message.find("BindingConflict"), std::string::npos) << outcome.message;
  EXPECT_EQ(outcome.message.find("broadcast tile and a full-shape tile"), std::string::npos) << outcome.message;
}

// 内层确实有广播轴，但 full-shape 一侧的外层行距不等于小 tile 行距乘广播 extent：
// 999 不是 1 × 128。这里必须报 BindingConflict，不能因为“后面有广播轴”任意放行。
TEST(ScheduleToAfirTest, InnerBroadcastWithWrongOuterPitchIsConflict) {
  const auto outcome = RunSharedBufferInnerBroadcastPlan("shared_buffer_inner_broadcast_bad_pitch", 999);
  EXPECT_FALSE(outcome.succeeded);
  EXPECT_NE(outcome.message.find("BindingConflict"), std::string::npos) << outcome.message;
}

// 非广播外轴的 stride 差异不能只靠“后面曾经有广播”解释；这里没有任何广播轴，
// 因此两张视图是真冲突，必须保持 BindingConflict。
TEST(ScheduleToAfirTest, OuterStrideDifferenceWithoutInnerBroadcastIsConflict) {
  auto graph = af::testing::AscGraphBuilder("shared_buffer_outer_stride_mismatch")
                   .Loops({af::testing::Sym(8), af::testing::Sym(8)})
                   .Data("x", 0, af::DT_FLOAT16)
                   .Data("b", 1, af::DT_FLOAT16)
                   .Load("load_x", "x")
                   .Load("load_b", "b")
                   .Store("store_x", "load_x")
                   .Output("y0", "store_x", 0, af::DT_FLOAT16)
                   .Store("store_b", "load_b")
                   .Output("y1", "store_b", 1, af::DT_FLOAT16)
                   .Build();
  AssignTensorIds(graph);
  graph.SetTilingKey(0);
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetType() == "Data") {
      node->outputs()[0]->attr.axis.clear();
    }
    if (node->GetType() == "Store" || node->GetType() == "Output") {
      auto &tensor = node->outputs()[0]->attr;
      tensor.axis = {0, 1};
      tensor.repeats = {af::Symbol(8), af::Symbol(8)};
      tensor.strides = {af::Symbol(8), af::Symbol(1)};
      continue;
    }
    if (node->GetName() != "load_x" && node->GetName() != "load_b") {
      continue;
    }
    node->attr.sched.axis = {0, 1};
    node->attr.sched.loop_axis = 1;
    for (const auto &output : node->outputs()) {
      auto &attr = output->attr;
      attr.mem.position = af::Position::kPositionVecCalc;
      attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
      attr.mem.reuse_id = -1;
      attr.buf.id = 0;
      attr.axis = {0, 1};
      attr.repeats = {af::Symbol(8), af::Symbol(8)};
      attr.strides = {af::Symbol(node->GetName() == "load_x" ? 1 : 2), af::Symbol(1)};
    }
  }

  ascir::ScheduleGroup group;
  group.impl_graphs.push_back(std::move(graph));
  ascir::ScheduledResult scheduledResult;
  scheduledResult.schedule_groups.push_back(std::move(group));
  ascir::FusedScheduledResult fusedResult;
  fusedResult.fused_graph_name = "shared_buffer_outer_stride_mismatch";
  fusedResult.node_idx_to_scheduled_results = {{std::move(scheduledResult)}};

  mlir::MLIRContext context;
  std::string message;
  mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &diagnostic) {
    llvm::raw_string_ostream output(message);
    diagnostic.print(output);
    return mlir::failure();
  });
  auto module =
      mlir::afir::bridge::PlanToAfir(context, fusedResult, fusedResult.node_idx_to_scheduled_results[0][0], 0, 0);
  EXPECT_FALSE(module);
  EXPECT_NE(message.find("BindingConflict"), std::string::npos) << message;
}

InnerBroadcastPlanOutcome RunSharedBufferVectorizedPitchPlan(const std::string &name,
                                                             int64_t secondLogicalOuterStride) {
  auto graph = af::testing::AscGraphBuilder(name)
                   .Loops({af::testing::Sym(8), af::testing::Sym(8)})
                   .Data("x", 0, af::DT_FLOAT16)
                   .Data("b", 1, af::DT_FLOAT16)
                   .Load("load_x", "x")
                   .Load("load_b", "b")
                   .Store("store_x", "load_x")
                   .Output("y0", "store_x", 0, af::DT_FLOAT16)
                   .Store("store_b", "load_b")
                   .Output("y1", "store_b", 1, af::DT_FLOAT16)
                   .Build();
  AssignTensorIds(graph);
  graph.SetTilingKey(0);
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetType() == "Data") {
      node->outputs()[0]->attr.axis.clear();
    }
    if (node->GetType() == "Store" || node->GetType() == "Output") {
      auto &tensor = node->outputs()[0]->attr;
      tensor.axis = {0, 1};
      tensor.repeats = {af::Symbol(8), af::Symbol(8)};
      tensor.strides = {af::Symbol(8), af::Symbol(1)};
      continue;
    }
    if (node->GetName() != "load_x" && node->GetName() != "load_b") {
      continue;
    }
    node->attr.sched.axis = {0, 1};
    node->attr.sched.loop_axis = 1;
    for (const auto &output : node->outputs()) {
      auto &attr = output->attr;
      attr.mem.position = af::Position::kPositionVecCalc;
      attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
      attr.mem.reuse_id = -1;
      attr.buf.id = 0;
      attr.axis = {0, 1};
      attr.repeats = {af::Symbol(8), af::Symbol(8)};
      attr.strides = {af::Symbol(node->GetName() == "load_x" ? 8 : secondLogicalOuterStride), af::Symbol(1)};
      attr.vectorized_axis = {1};
      attr.vectorized_strides = {af::Symbol(node->GetName() == "load_x" ? 1 : 8)};
    }
  }

  ascir::ScheduleGroup group;
  group.impl_graphs.push_back(std::move(graph));
  ascir::ScheduledResult scheduledResult;
  scheduledResult.schedule_groups.push_back(std::move(group));
  ascir::FusedScheduledResult fusedResult;
  fusedResult.fused_graph_name = name.c_str();
  fusedResult.node_idx_to_scheduled_results = {{std::move(scheduledResult)}};

  mlir::MLIRContext context;
  std::string message;
  mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &diagnostic) {
    llvm::raw_string_ostream output(message);
    diagnostic.print(output);
    return mlir::failure();
  });
  auto module =
      mlir::afir::bridge::PlanToAfir(context, fusedResult, fusedResult.node_idx_to_scheduled_results[0][0], 0, 0);
  return {static_cast<bool>(module), message};
}

// 同一逻辑 tile 的 padded / unpadded vectorized pitch 是独立于 broadcast 的合法时间复用：
// repeats 与 logical strides 完全一致，只有 vectorized_strides 不同。
TEST(ScheduleToAfirTest, SameLogicalTileWithDifferentVectorizedPitchIsSupported) {
  const auto outcome = RunSharedBufferVectorizedPitchPlan("shared_buffer_vectorized_pitch", 8);
  ASSERT_TRUE(outcome.succeeded) << outcome.message;
  EXPECT_EQ(outcome.message.find("BindingConflict"), std::string::npos) << outcome.message;
}

// vectorized pitch 不同本身不足以放行：logical stride 也不同时，这是两张不同的逻辑 tile，
// 不能按“padded / unpadded 视图”复用同一块非队列 buffer。
TEST(ScheduleToAfirTest, DifferentLogicalTileWithDifferentVectorizedPitchIsConflict) {
  const auto outcome = RunSharedBufferVectorizedPitchPlan("shared_buffer_different_logical_tile", 9);
  EXPECT_FALSE(outcome.succeeded);
  EXPECT_NE(outcome.message.find("BindingConflict"), std::string::npos) << outcome.message;
}

// RCore 的 `Workspace` 节点在一张图里出现的方向有两种，两条用例分别钉住：
//
// - phase 2 当**源**（`workspace -> Load -> Sum`），自身带完整 axis / repeats / strides；
// - phase 1 当**汇**（`Sum -> Store -> workspace`），自身退化到只剩 `mem`，形状要从写它的
//   Store 上取。
//
// 两边共同的判据是同一条：它变成函数**体内**一块带 GM `buffer_desc` 的存储，`buffer_id`
// 等于 tensor id，且**不占 kernel 形参**。最后这条是与「映射成 `Data`」方案的分水岭——
// 按 `Data` 走会凭空多一个入参并打乱既有入参编号。
namespace {
mlir::afir::bridge::TensorValueDTO MakeWorkspaceTestTensor(mlir::afir::Space space, int64_t bufferId) {
  mlir::afir::bridge::TensorValueDTO value;
  value.tensor.shape = {4, 8};
  value.tensor.shape_exprs = {"4", "8"};
  value.tensor.dtype = "float32";
  value.tensor.axis = {0, 1};
  value.tensor.repeats = {"4", "8"};
  value.tensor.strides = {"8", "1"};
  value.buffer.space = space;
  value.buffer.bufferId = bufferId;
  value.buffer.slotCount = space == mlir::afir::Space::UB ? 2 : -1;
  value.buffer.queueDepth = space == mlir::afir::Space::UB ? 2 : -1;
  value.buffer.sizeBytesExpr = "128";
  value.buffer.lifeTimeAxisId = -1;
  return value;
}

// 函数体里那块带 GM `buffer_desc`、`buffer_id` 为 `tensorId` 的 alloc。找不到返回空。
mlir::memref::AllocOp FindWorkspaceAlloc(mlir::func::FuncOp function, int64_t tensorId) {
  mlir::memref::AllocOp found;
  function.walk([&](mlir::memref::AllocOp alloc) {
    auto descriptor = alloc->getAttrOfType<mlir::afir::BufferDescAttr>(mlir::afir::kBufferDescAttrName);
    if (descriptor && descriptor.getSpace().getValue() == mlir::afir::Space::GM &&
        descriptor.getBufferId() == tensorId) {
      found = alloc;
    }
  });
  return found;
}

// `afir.store` 两端 dtype 不同是今天真会被 legacy 调度图违反的一条不变量——`Fma` 接 bf16、
// `round_to_int` / `trunc_to_int` 都踩过：legacy 对某些节点「升精度算、按原 dtype 存」
// （`dtype_consistency.cpp` 的 `GetConversionDtype`），存前插的 Cast 落回的是节点自己声明的
// dtype，而 GM 侧的函数签名可能是另一个。这条测试钉住 `AfirBuilder.cpp` 里那条干净拒收：
// 两端对不上要走 `CapabilityShortfall`、回落 legacy，不能绕到 `afir.store` 的 verifier
// 变成硬失败（那条诊断是 `transfer source and destination must have the same element
// type`，参见 `dialect/AFIR/IR/Ops/Data.cpp` 的 `verifyTransferIndexingMaps`）。
TEST(ScheduleToAfirTest, StoreElementTypeMismatchIsCapabilityShortfallNotHardFailure) {
  auto graph = af::testing::AscGraphBuilder("store_dtype_mismatch")
                   .Loops({af::testing::Sym(8), af::testing::Sym(8)})
                   .Data("x", 0, af::DT_BF16)
                   .Load("load_x", "x")
                   .Cast("up", "load_x", af::DT_FLOAT)
                   .Cast("down", "up", af::DT_BF16)
                   .Store("store_y", "down")
                   // **故意写错**：Store 的源（`down`）是 bf16，这里的 GM 签名是 f32——
                   // 与 `abs_fma_bf16_test` 实测的那张调度图（`%output: f32` 配
                   // `Store(Cast(Fma) -> bf16)`）逐字同形。
                   .Output("y", "store_y", 0, af::DT_FLOAT)
                   .Build();
  AssignTensorIds(graph);
  graph.SetTilingKey(0);

  ascir::ScheduleGroup group;
  group.impl_graphs.push_back(std::move(graph));
  ascir::ScheduledResult scheduledResult;
  scheduledResult.schedule_groups.push_back(std::move(group));
  ascir::FusedScheduledResult fusedResult;
  fusedResult.fused_graph_name = "store_dtype_mismatch";
  fusedResult.node_idx_to_scheduled_results = {{std::move(scheduledResult)}};

  mlir::MLIRContext context;
  mlir::afir::CapabilityShortfallObserver observer(context);
  std::string message;
  mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &diagnostic) {
    llvm::raw_string_ostream output(message);
    diagnostic.print(output);
    return mlir::failure();
  });
  const bool built = static_cast<bool>(
      mlir::afir::bridge::PlanToAfir(context, fusedResult, fusedResult.node_idx_to_scheduled_results[0][0], 0, 0));

  EXPECT_FALSE(built);
  EXPECT_TRUE(observer.sawShortfall()) << message;
  EXPECT_NE(message.find("transfers between"), std::string::npos) << message;
}

}  // namespace

TEST(ScheduleToAfirTest, WorkspaceSourceBecomesFunctionLocalGmStorage) {
  mlir::MLIRContext context;
  mlir::afir::bridge::PlanInfo plan;
  plan.name = "workspace_source";

  mlir::afir::bridge::TemplateInfo tmpl;
  tmpl.name = "workspace_source";
  tmpl.original_name = tmpl.name;
  tmpl.buffer_form = true;
  tmpl.axes.push_back({0, "r", "original", "4"});
  tmpl.axes.push_back({1, "a", "original", "8"});

  auto workspaceTensor = MakeWorkspaceTestTensor(mlir::afir::Space::GM, 2);
  auto local = MakeWorkspaceTestTensor(mlir::afir::Space::UB, 3);
  auto output = MakeWorkspaceTestTensor(mlir::afir::Space::GM, -1);
  tmpl.outputs = {{"output", output, 0}};

  mlir::afir::bridge::NodeInfo workspace;
  workspace.name = "ws";
  workspace.type = "Workspace";
  workspace.output = workspaceTensor;
  workspace.output_names = {"ws"};

  mlir::afir::bridge::NodeInfo load;
  load.name = "load";
  load.type = "Load";
  load.input_names = {"ws"};
  load.inputs = {workspaceTensor};
  load.output = local;
  load.sched_axis = {0, 1};
  load.loop_axis = 0;

  mlir::afir::bridge::NodeInfo store;
  store.name = "output";
  store.type = "Store";
  store.input_names = {"load"};
  store.inputs = {local};
  store.output = output;
  store.sched_axis = {0, 1};
  store.loop_axis = 0;

  tmpl.nodes = {workspace, load, store};
  plan.templates.push_back(tmpl);

  auto module = mlir::afir::bridge::BuildAfirModule(context, plan);
  ASSERT_TRUE(module);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("workspace_source");
  ASSERT_TRUE(function);

  // 形参只有那一个 output，workspace 没有挤进来。
  EXPECT_EQ(function.getNumArguments(), 1U);

  auto alloc = FindWorkspaceAlloc(function, 2);
  ASSERT_TRUE(alloc);
  auto type = mlir::cast<mlir::MemRefType>(alloc.getResult().getType());
  EXPECT_EQ(type.getShape(), (llvm::ArrayRef<int64_t>{4, 8}));
  // Load 读的就是它。
  EXPECT_FALSE(alloc.getResult().use_empty());
}

TEST(ScheduleToAfirTest, WorkspaceSinkTakesItsShapeFromTheStoreThatWritesIt) {
  mlir::MLIRContext context;
  mlir::afir::bridge::PlanInfo plan;
  plan.name = "workspace_sink";

  mlir::afir::bridge::TemplateInfo tmpl;
  tmpl.name = "workspace_sink";
  tmpl.original_name = tmpl.name;
  tmpl.buffer_form = true;
  tmpl.axes.push_back({0, "r", "original", "4"});
  tmpl.axes.push_back({1, "a", "original", "8"});

  auto input = MakeWorkspaceTestTensor(mlir::afir::Space::GM, -1);
  auto local = MakeWorkspaceTestTensor(mlir::afir::Space::UB, 3);
  auto storeTensor = MakeWorkspaceTestTensor(mlir::afir::Space::GM, -1);
  tmpl.inputs = {{"input0", input, 0}};

  mlir::afir::bridge::NodeInfo load;
  load.name = "load";
  load.type = "Load";
  load.input_names = {"input0"};
  load.inputs = {input};
  load.output = local;
  load.sched_axis = {0, 1};
  load.loop_axis = 0;

  mlir::afir::bridge::NodeInfo store;
  store.name = "store";
  store.type = "Store";
  store.input_names = {"load"};
  store.inputs = {local};
  store.output = storeTensor;
  store.output_names = {"store"};
  store.sched_axis = {0, 1};
  store.loop_axis = 0;

  // 调度器产出的汇：自身没有形状，只有 `mem`。
  mlir::afir::bridge::NodeInfo workspace;
  workspace.name = "ws";
  workspace.type = "Workspace";
  workspace.input_names = {"store"};
  workspace.inputs = {storeTensor};
  workspace.output.tensor.dtype = "float32";
  workspace.output.buffer.space = mlir::afir::Space::GM;
  workspace.output.buffer.bufferId = 2;
  workspace.output.buffer.slotCount = -1;
  workspace.output.buffer.queueDepth = -1;
  workspace.output.buffer.lifeTimeAxisId = -1;

  tmpl.nodes = {load, store, workspace};
  plan.templates.push_back(tmpl);

  auto module = mlir::afir::bridge::BuildAfirModule(context, plan);
  ASSERT_TRUE(module);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("workspace_sink");
  ASSERT_TRUE(function);

  // 这张图没有 `Output` 节点，形参里只有那个输入——Store 写的是体内的 workspace。
  EXPECT_EQ(function.getNumArguments(), 1U);

  auto alloc = FindWorkspaceAlloc(function, 2);
  ASSERT_TRUE(alloc);
  // 形状是从 Store 上取来的，不是退化描述子的 rank-0。
  auto type = mlir::cast<mlir::MemRefType>(alloc.getResult().getType());
  EXPECT_EQ(type.getShape(), (llvm::ArrayRef<int64_t>{4, 8}));

  // Store 的 destination 就是这块 workspace，而不是某个 kernel 参数。
  mlir::afir::StoreOp storeOp;
  function.walk([&](mlir::afir::StoreOp op) { storeOp = op; });
  ASSERT_TRUE(storeOp);
  EXPECT_EQ(storeOp.getDestination(), alloc.getResult());
}

// 上一条的两侧轴表相同，折叠按下标配对也能过。但归约轴分布在保留轴两侧时（`[R, A, R]`
// 归约 {0, 2}），调度器会把不相邻的那两根归约轴融成一根再拿去分核：目的建立在融合后的块轴
// 上，源还留在原始轴上，两张轴表既不同序也不同集合。按下标比会拿源的 `aT`（切块个数）去比
// 目的的 `at`（块内长度），报出一条两个数都不该放在一起的错——实测 `(8,8,128,128)` 归约
// {0,2,3} 就死在这里，而同一形状只归约 {0} / {2,3} / {0,1} 都能过。
TEST(ScheduleToAfirTest, WorkspaceStoreFoldsBlockScatterWhenAxisListsDiverge) {
  mlir::MLIRContext context;
  mlir::afir::bridge::PlanInfo plan;
  plan.name = "workspace_axis_diverge";

  mlir::afir::bridge::TemplateInfo tmpl;
  tmpl.name = "workspace_axis_diverge";
  tmpl.original_name = tmpl.name;
  tmpl.buffer_form = true;
  tmpl.axes.push_back({0, "r0", "original", "2"});
  tmpl.axes.push_back({1, "aT", "original", "4"});
  tmpl.axes.push_back({2, "at", "original", "2"});
  tmpl.axes.push_back({3, "rB", "original", "4"});

  // 源：留在原始轴 {r0, aT, at} 上，r0 已归约掉所以 repeat 为 1，保留轴被切成 aT×at。
  auto reduced = MakeWorkspaceTestTensor(mlir::afir::Space::UB, 3);
  reduced.tensor.axis = {0, 1, 2};
  reduced.tensor.shape = {1, 4, 2};
  reduced.tensor.shape_exprs = {"1", "4", "2"};
  reduced.tensor.repeats = {"1", "4", "2"};
  reduced.tensor.strides = {"0", "2", "1"};

  auto input = MakeWorkspaceTestTensor(mlir::afir::Space::GM, -1);
  input.tensor.axis = {0, 1, 2};
  input.tensor.shape = {1, 4, 2};
  input.tensor.shape_exprs = {"1", "4", "2"};
  input.tensor.repeats = {"1", "4", "2"};
  input.tensor.strides = {"0", "2", "1"};

  // 目的：轴表换成 {aT, at, rB}——保留轴在前，块轴 rB 是融合出来的、源上根本没有。
  // 注意 `at` 的下标在两侧不同（源 2、目的 1），按下标配对会把它和源的 `aT` 撞在一起。
  auto storeTensor = MakeWorkspaceTestTensor(mlir::afir::Space::GM, -1);
  storeTensor.tensor.axis = {1, 2, 3};
  storeTensor.tensor.shape = {4, 2, 4};
  storeTensor.tensor.shape_exprs = {"4", "2", "4"};
  storeTensor.tensor.repeats = {"4", "2", "4"};
  storeTensor.tensor.strides = {"2", "1", "8"};

  mlir::afir::bridge::NodeInfo load;
  load.name = "load";
  load.type = "Load";
  load.input_names = {"input0"};
  load.inputs = {input};
  load.output = reduced;
  load.sched_axis = {0, 1, 2};
  load.loop_axis = 0;
  tmpl.inputs = {{"input0", input, 0}};

  mlir::afir::bridge::NodeInfo store;
  store.name = "store";
  store.type = "Store";
  store.input_names = {"load"};
  store.inputs = {reduced};
  store.output = storeTensor;
  store.output_names = {"store"};
  store.sched_axis = {0, 1, 2};
  store.loop_axis = 0;

  mlir::afir::bridge::NodeInfo workspace;
  workspace.name = "ws";
  workspace.type = "Workspace";
  workspace.input_names = {"store"};
  workspace.inputs = {storeTensor};
  workspace.output.tensor.dtype = "float32";
  workspace.output.buffer.space = mlir::afir::Space::GM;
  workspace.output.buffer.bufferId = 2;
  workspace.output.buffer.slotCount = -1;
  workspace.output.buffer.queueDepth = -1;
  workspace.output.buffer.lifeTimeAxisId = -1;

  tmpl.nodes = {load, store, workspace};
  plan.templates.push_back(tmpl);

  auto module = mlir::afir::bridge::BuildAfirModule(context, plan);
  ASSERT_TRUE(module);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("workspace_axis_diverge");
  ASSERT_TRUE(function);

  mlir::afir::StoreOp storeOp;
  function.walk([&](mlir::afir::StoreOp op) { storeOp = op; });
  ASSERT_TRUE(storeOp);
  auto destinations = storeOp->getAttrOfType<mlir::ArrayAttr>(mlir::afir::kDestinationDescsAttrName);
  ASSERT_TRUE(destinations);
  ASSERT_EQ(destinations.size(), 1U);
  auto destination = mlir::cast<mlir::afir::TensorDescAttr>(destinations[0]);

  // 按轴 id 配上的两维原样保留，源上没有的块轴折成 1：元素数 4×2 与源相等。
  EXPECT_EQ(destination.getRepeats()[0].getValue().getValue(), "4");
  EXPECT_EQ(destination.getRepeats()[1].getValue().getValue(), "2");
  EXPECT_EQ(destination.getRepeats()[2].getValue().getValue(), "1");
  // 块轴的 stride 保持目的的，它是块偏移。
  EXPECT_EQ(destination.getStrides()[2].getValue().getValue(), "8");
}

// 多核 reduce 里，写 workspace 的那条 Store 两侧口径不同：源是本核归约出来的一片（块轴
// repeat 为 1），目的却是整块 workspace（块轴 repeat 为块数）。`afir.store` 的契约要求两侧
// 元素数相等，表达不了这种「按块索引散写」——块偏移由块号在下降时施加，不由描述子表达。
//
// 所以 destination 描述子要折一次：轴、步长、向量化保留目的的（那条块步长正是偏移所需），
// repeats 取源的。这条钉住折叠**只动 repeat，不动 stride**——把 stride 一起折掉的话，
// 每个核都会写到 workspace 的第 0 行。
TEST(ScheduleToAfirTest, WorkspaceStoreFoldsBlockScatteredRepeatsButKeepsStrides) {
  mlir::MLIRContext context;
  mlir::afir::bridge::PlanInfo plan;
  plan.name = "workspace_block_scatter";

  mlir::afir::bridge::TemplateInfo tmpl;
  tmpl.name = "workspace_block_scatter";
  tmpl.original_name = tmpl.name;
  tmpl.buffer_form = true;
  tmpl.axes.push_back({0, "r", "original", "4"});
  tmpl.axes.push_back({1, "a", "original", "8"});

  // 本核归约出来的一片：块轴 repeat 为 1、stride 为 0。夹具里不真做归约（`Sum` 还要求
  // scratch），直接让 Load 搬的就是这一片——关心的是 Store 两侧的口径差，不是谁产出它。
  auto reduced = MakeWorkspaceTestTensor(mlir::afir::Space::UB, 3);
  reduced.tensor.shape = {1, 8};
  reduced.tensor.shape_exprs = {"1", "8"};
  reduced.tensor.repeats = {"1", "8"};
  reduced.tensor.strides = {"0", "1"};

  auto input = MakeWorkspaceTestTensor(mlir::afir::Space::GM, -1);
  input.tensor.shape = {1, 8};
  input.tensor.shape_exprs = {"1", "8"};
  input.tensor.repeats = {"1", "8"};
  input.tensor.strides = {"0", "1"};

  // Store 的目的是整块 workspace：块轴 repeat 为块数 4、stride 为 8（每核一行）。
  auto storeTensor = MakeWorkspaceTestTensor(mlir::afir::Space::GM, -1);
  storeTensor.tensor.strides = {"8", "1"};

  mlir::afir::bridge::NodeInfo load;
  load.name = "load";
  load.type = "Load";
  load.input_names = {"input0"};
  load.inputs = {input};
  load.output = reduced;
  load.sched_axis = {0, 1};
  load.loop_axis = 0;
  tmpl.inputs = {{"input0", input, 0}};

  mlir::afir::bridge::NodeInfo store;
  store.name = "store";
  store.type = "Store";
  store.input_names = {"load"};
  store.inputs = {reduced};
  store.output = storeTensor;
  store.output_names = {"store"};
  store.sched_axis = {0, 1};
  store.loop_axis = 0;

  mlir::afir::bridge::NodeInfo workspace;
  workspace.name = "ws";
  workspace.type = "Workspace";
  workspace.input_names = {"store"};
  workspace.inputs = {storeTensor};
  workspace.output.tensor.dtype = "float32";
  workspace.output.buffer.space = mlir::afir::Space::GM;
  workspace.output.buffer.bufferId = 2;
  workspace.output.buffer.slotCount = -1;
  workspace.output.buffer.queueDepth = -1;
  workspace.output.buffer.lifeTimeAxisId = -1;

  tmpl.nodes = {load, store, workspace};
  plan.templates.push_back(tmpl);

  auto module = mlir::afir::bridge::BuildAfirModule(context, plan);
  ASSERT_TRUE(module);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("workspace_block_scatter");
  ASSERT_TRUE(function);

  mlir::afir::StoreOp storeOp;
  function.walk([&](mlir::afir::StoreOp op) { storeOp = op; });
  ASSERT_TRUE(storeOp);
  auto destinations = storeOp->getAttrOfType<mlir::ArrayAttr>(mlir::afir::kDestinationDescsAttrName);
  ASSERT_TRUE(destinations);
  ASSERT_EQ(destinations.size(), 1U);
  auto destination = mlir::cast<mlir::afir::TensorDescAttr>(destinations[0]);

  // repeat 折成源的 1，元素数与源相等，`afir.store` 的契约才成立。
  EXPECT_EQ(destination.getRepeats()[0].getValue().getValue(), "1");
  EXPECT_EQ(destination.getRepeats()[1].getValue().getValue(), "8");
  // stride 保持目的的——它是块偏移，折掉就所有核写同一行。
  EXPECT_EQ(destination.getStrides()[0].getValue().getValue(), "8");
  EXPECT_EQ(destination.getStrides()[1].getValue().getValue(), "1");

  // workspace 值的类型跟着折叠后的视图走，否则 destination 的类型与描述子对不上。
  auto alloc = FindWorkspaceAlloc(function, 2);
  ASSERT_TRUE(alloc);
  EXPECT_EQ(mlir::cast<mlir::MemRefType>(alloc.getResult().getType()).getShape(), (llvm::ArrayRef<int64_t>{1, 8}));
}

}  // namespace

namespace {
mlir::afir::bridge::PlanInfo MakeArgMaxPairPlan() {
  using namespace mlir::afir;
  using namespace mlir::afir::bridge;
  PlanInfo plan;
  plan.name = "argmax_pair";
  TemplateInfo tmpl;
  tmpl.name = tmpl.original_name = plan.name;
  tmpl.buffer_form = true;
  tmpl.axes = {{0, "r", "original", "4"}, {1, "a", "original", "8"}};
  auto input = MakeWorkspaceTestTensor(Space::GM, -1);
  auto local = MakeWorkspaceTestTensor(Space::UB, 0);
  local.tensor.vectorized_axis = {0, 1};
  local.tensor.vectorized_strides = {"8", "1"};
  auto value = MakeWorkspaceTestTensor(Space::UB, 1);
  value.tensor.shape = {1, 8};
  value.tensor.shape_exprs = {"1", "8"};
  value.tensor.repeats = {"1", "8"};
  value.tensor.strides = {"0", "1"};
  value.tensor.vectorized_axis = {0, 1};
  value.tensor.vectorized_strides = {"0", "1"};
  auto index = value;
  index.tensor.dtype = "int64";
  index.buffer.bufferId = 2;
  auto valueOut = value, indexOut = index;
  valueOut.buffer.space = indexOut.buffer.space = Space::GM;
  valueOut.buffer.slotCount = indexOut.buffer.slotCount = -1;
  valueOut.buffer.queueDepth = indexOut.buffer.queueDepth = -1;
  tmpl.inputs = {{"input", input, 0}};
  tmpl.outputs = {{"value_store", valueOut, 0}, {"index_store", indexOut, 1}};
  NodeInfo load;
  load.name = "load";
  load.type = "Load";
  load.input_names = {"input"};
  load.inputs = {input};
  load.output = local;
  load.sched_axis = {0, 1};
  load.loop_axis = 0;
  NodeInfo phase;
  phase.name = "phase";
  phase.type = "ArgMaxMultiRPhase1";
  phase.input_names = {"load"};
  phase.inputs = {local};
  phase.output = value;
  phase.outputs = {value, index};
  phase.output_names = {"phase:0", "phase:1"};
  phase.sched_axis = {0, 1};
  phase.loop_axis = 0;
  phase.tmp_buffers = {{3, -1}, {4, -1}, {5, -1}};
  NodeInfo valueStore;
  valueStore.name = "value_store";
  valueStore.type = "Store";
  valueStore.input_names = {"phase:0"};
  valueStore.inputs = {value};
  valueStore.output = valueOut;
  valueStore.sched_axis = {0, 1};
  valueStore.loop_axis = 0;
  NodeInfo indexStore = valueStore;
  indexStore.name = "index_store";
  indexStore.input_names = {"phase:1"};
  indexStore.inputs = {index};
  indexStore.output = indexOut;
  tmpl.nodes = {load, phase, valueStore, indexStore};
  plan.templates = {tmpl};
  return plan;
}
}  // namespace

TEST(ScheduleToAfirTest, ArgMaxPhase1IsUnsupported) {
  mlir::MLIRContext context;
  mlir::afir::CapabilityShortfallObserver observer(context);
  EXPECT_FALSE(mlir::afir::bridge::BuildAfirModule(context, MakeArgMaxPairPlan()));
  EXPECT_TRUE(observer.sawShortfall());
}

TEST(ScheduleToAfirTest, ArgMaxPhase2IsUnsupported) {
  using namespace mlir::afir;
  mlir::MLIRContext context;
  mlir::afir::CapabilityShortfallObserver observer(context);
  auto plan = MakeArgMaxPairPlan();
  auto &tmpl = plan.templates[0];
  auto indexInput = tmpl.inputs[0];
  indexInput.name = "input_index";
  indexInput.index = 1;
  indexInput.info.tensor.dtype = "int64";
  tmpl.inputs.push_back(indexInput);
  auto indexLoad = tmpl.nodes[0];
  indexLoad.name = "load_index";
  indexLoad.input_names = {"input_index"};
  indexLoad.inputs = {indexInput.info};
  indexLoad.output.tensor.dtype = "int64";
  indexLoad.output.buffer.bufferId = 6;
  auto phase = tmpl.nodes[1];
  phase.type = "ArgMaxMultiRPhase2";
  phase.input_names = {"load", "load_index"};
  phase.inputs.push_back(indexLoad.output);
  phase.output = phase.outputs[1];
  phase.outputs = {phase.output};
  phase.output_names = {"phase:1"};
  tmpl.nodes = {tmpl.nodes[0], indexLoad, phase, tmpl.nodes[3]};
  tmpl.outputs = {tmpl.outputs[1]};
  tmpl.outputs[0].index = 0;
  auto module = mlir::afir::bridge::BuildAfirModule(context, plan);
  EXPECT_FALSE(module);
  EXPECT_TRUE(observer.sawShortfall());
}

TEST(ScheduleToAfirTest, ArgMaxRCoreCandidatesRemainDisabled) {
  for (const char *route : {"none", "fallback", "mlir"}) {
    for (auto dtype : {af::DT_FLOAT, af::DT_INT32, af::DT_FLOAT16}) {
      SCOPED_TRACE(route);
      SCOPED_TRACE(static_cast<int>(dtype));
      mlir::test::ScopedEnvVar backend("AUTOFUSE_DFX_FLAGS", std::string("--mlir_codegen_config=") + route + ";");
      af::testing::AscGraphBuilder builder("argmax_rcore_gate");
      builder.Loops({af::testing::Sym(8), af::testing::Sym(8193)}).Data("x", 0, dtype).Load("load", "x");
      auto graph = mlir::test::AddReduce<af::ascir_op::ArgMax>(
                       builder, "reduce", "load", {af::testing::Sym(8), af::testing::Sym(8193)}, {1}, af::DT_INT64)
                       .Store("store", "reduce")
                       .Output("output", "store", 0, af::DT_INT64)
                       .Build();
      std::vector<optimize::ScheduleTask> tasks;
      optimize::ReducePartitionCaseGenerator generator;
      ASSERT_EQ(generator.GeneratorTask(graph, tasks, optimize::OptimizerOptions{}), af::SUCCESS);
      ascir::FusedScheduledResult result;
      result.node_idx_to_scheduled_results.resize(1);
      bool phase1 = false, phase2 = false;
      for (const auto &task : tasks) {
        ascir::ScheduledResult plan;
        for (const auto &impl : task.grouped_graphs) {
          for (const auto &node : impl.GetAllNodes()) {
            phase1 |= node->GetType() == "ArgMaxMultiRPhase1";
            phase2 |= node->GetType() == "ArgMaxMultiRPhase2";
          }
          ascir::ScheduleGroup group;
          group.impl_graphs.push_back(impl);
          plan.schedule_groups.push_back(std::move(group));
        }
        result.node_idx_to_scheduled_results[0].push_back(std::move(plan));
      }
      EXPECT_FALSE(phase1);
      EXPECT_FALSE(phase2);
    }
  }
}

TEST(ScheduleToAfirTest, PartitionWorkspaceLayoutRemainsUpstreamForBothRoutes) {
  for (const char *route : {"none", "mlir"}) {
    mlir::test::ScopedEnvVar backend("AUTOFUSE_DFX_FLAGS", std::string("--mlir_codegen_config=") + route + ";");
    // Four post-reduce ops plus Store exceed the post-fusion partition threshold.
    auto graph = af::testing::AscGraphBuilder("partition_workspace_route")
                     .Loops({af::testing::Sym(16), af::testing::Sym(32)})
                     .Data("x", 0)
                     .Load("load", "x")
                     .Sum("first", "load", {1})
                     .Abs("abs", "first")
                     .Relu("relu", "abs")
                     .Op<af::ascir_op::Exp>("exp", {"relu"})
                     .Op<af::ascir_op::Tanh>("tanh", {"exp"})
                     .Store("store", "tanh")
                     .Output("output", "store", 0)
                     .Build();
    std::vector<optimize::ScheduleTask> tasks;
    optimize::ReducePartitionCaseGenerator generator;
    ASSERT_EQ(generator.GeneratorTask(graph, tasks, optimize::OptimizerOptions{}), af::SUCCESS);
    size_t source_count = 0;
    for (const auto &task : tasks) {
      if (task.reduce_type != optimize::ReduceTemplateType::kCommon) {
        continue;
      }
      for (const auto &impl : task.grouped_graphs) {
        for (const auto &node : impl.GetAllNodes()) {
          if (node->GetType() != "Workspace" || node->GetName().find("first_Workspace") == std::string::npos ||
              !node->GetInDataNodes().empty()) {
            continue;
          }
          ++source_count;
          const auto &view = node->outputs[0].attr;
          EXPECT_TRUE(view.axis.empty());
          EXPECT_TRUE(view.repeats.empty());
          EXPECT_TRUE(view.strides.empty());
          EXPECT_EQ(view.dtype, af::DT_FLOAT);
        }
      }
    }
    EXPECT_GT(source_count, 0U);
  }
}

namespace {
mlir::afir::bridge::PlanInfo MakeCrossStageWorkspacePlan() {
  using namespace mlir::afir::bridge;
  PlanInfo plan;
  TemplateInfo writer, reader;
  writer.stage_id = 0;
  writer.template_id = 0;
  writer.buffer_form = true;
  reader.stage_id = 1;
  reader.template_id = 0;
  reader.buffer_form = true;
  NodeInfo store, sink, source, load;
  store.name = "store";
  store.type = "Store";
  store.output_names = {"stored"};
  store.output.tensor.dtype = "float32";
  store.output.tensor.repeats = {"4", "4"};
  store.output.tensor.strides = {"4", "1"};
  store.output.buffer.sizeBytesExpr = "64";
  sink.type = "Workspace";
  sink.name = "write_name";
  sink.input_names = {"stored"};
  sink.output.buffer.bufferId = 17;
  source.type = "Workspace";
  source.name = "different_read_name";
  source.output_names = {source.name};
  source.output.buffer.bufferId = 17;
  source.output.tensor.dtype = "float32";
  load.type = "Load";
  load.input_names = {source.name};
  load.inputs = {source.output};
  load.output.tensor.dtype = "float32";
  load.output.tensor.repeats = {"2", "8"};
  load.output.tensor.strides = {"8", "1"};
  // UB padding must not be confused with the GM row pitch.
  load.output.tensor.vectorized_strides = {"32", "1"};
  writer.nodes = {store, sink};
  reader.nodes = {source, load};
  plan.templates = {writer, reader};
  return plan;
}
}  // namespace

TEST(ScheduleToAfirTest, WorkspaceViewsJoinByIdentityAndPreserveLocalGeometry) {
  mlir::MLIRContext context;
  auto plan = MakeCrossStageWorkspacePlan();
  ASSERT_TRUE(succeeded(mlir::afir::bridge::resolveWorkspaceSourceViews(plan, mlir::UnknownLoc::get(&context))));
  const auto &source = plan.templates[1].nodes[0].output;
  EXPECT_EQ(source.tensor.shape, (std::vector<int64_t>{16}));
  EXPECT_TRUE(source.tensor.axis.empty());
  EXPECT_EQ(source.buffer.bufferId, 17);
  EXPECT_EQ(plan.templates[1].nodes[1].inputs[0].tensor.shape, source.tensor.shape);
  EXPECT_EQ(plan.templates[1].nodes[1].output.tensor.vectorized_strides[0], "32");
  ASSERT_TRUE(succeeded(mlir::afir::bridge::resolveWorkspaceSourceViews(plan, mlir::UnknownLoc::get(&context))));
}

TEST(ScheduleToAfirTest, WorkspaceViewsAcceptConsistentAlternativeTemplates) {
  mlir::MLIRContext context;
  auto plan = MakeCrossStageWorkspacePlan();
  auto alternative = plan.templates[0];
  alternative.template_id = 1;
  alternative.nodes[0].output.tensor.repeats = {"2", "8"};
  alternative.nodes[0].output.tensor.strides = {"8", "1"};
  plan.templates.push_back(alternative);
  EXPECT_TRUE(succeeded(mlir::afir::bridge::resolveWorkspaceSourceViews(plan, mlir::UnknownLoc::get(&context))));
}

TEST(ScheduleToAfirTest, WorkspaceViewsAcceptSharedOriginalSymbol) {
  mlir::MLIRContext context;
  auto plan = MakeCrossStageWorkspacePlan();
  for (auto &tmpl : plan.templates) {
    tmpl.axes.push_back({0, "a", "original", "s0"});
  }
  for (auto *view : {&plan.templates[0].nodes[0].output, &plan.templates[1].nodes[1].output}) {
    view->tensor.repeats = {"s0"};
    view->tensor.strides = {"1"};
    view->buffer.sizeBytesExpr = "(4 * s0)";
  }
  EXPECT_TRUE(succeeded(mlir::afir::bridge::resolveWorkspaceSourceViews(plan, mlir::UnknownLoc::get(&context))));
  EXPECT_EQ(plan.templates[1].nodes[0].output.tensor.shape_exprs, (std::vector<std::string>{"s0"}));
  EXPECT_EQ(plan.templates[1].nodes[0].output.tensor.shape, (std::vector<int64_t>{mlir::ShapedType::kDynamic}));
}

namespace {
mlir::afir::bridge::PlanInfo MakeRelatedWorkspacePlan() {
  auto plan = MakeCrossStageWorkspacePlan();
  auto &writer = plan.templates[0];
  writer.stage_id = 1;
  writer.axes.push_back({0, "a", "original", "A_org_size"});
  writer.nodes[0].output.tensor.repeats = {"A_org_size"};
  writer.nodes[0].output.tensor.strides = {"1"};
  writer.nodes[0].output.buffer.sizeBytesExpr = "(4 * A_org_size)";
  plan.templates[1].stage_id = 2;
  mlir::afir::bridge::TemplateInfo original;
  original.stage_id = 0;
  original.axes.push_back({0, "a", "original", "s0"});
  plan.templates.push_back(original);
  plan.var_relations.push_back({1, 0, "A_org_size", "16"});
  return plan;
}
}  // namespace

TEST(ScheduleToAfirTest, WorkspaceViewsResolveStageShapeRelations) {
  mlir::MLIRContext context;
  auto plan = MakeRelatedWorkspacePlan();
  ASSERT_TRUE(succeeded(mlir::afir::bridge::resolveWorkspaceSourceViews(plan, mlir::UnknownLoc::get(&context))));
  EXPECT_EQ(plan.templates[1].nodes[0].output.tensor.shape, (std::vector<int64_t>{16}));
  EXPECT_EQ(plan.templates[1].nodes[0].output.tensor.shape_exprs, (std::vector<std::string>{"16"}));
  EXPECT_EQ(plan.templates[1].nodes[0].output.buffer.sizeBytesExpr, "64");
  // Writer-local symbols stay in the writer descriptor, never in the read view.
  EXPECT_EQ(plan.templates[0].nodes[0].output.tensor.repeats, (std::vector<std::string>{"A_org_size"}));
}

TEST(ScheduleToAfirTest, WorkspaceViewsResolveChainedDynamicRelationsInStageScope) {
  mlir::MLIRContext context;
  auto plan = MakeRelatedWorkspacePlan();
  plan.var_relations = {{1, 0, "A_org_size", "(2 * s0)"}, {2, 1, "N", "A_org_size"}};
  auto &reader = plan.templates[1];
  reader.axes.push_back({0, "a", "original", "N"});
  reader.nodes[1].output.tensor.repeats = {"N"};
  reader.nodes[1].output.tensor.strides = {"1"};
  ASSERT_TRUE(succeeded(mlir::afir::bridge::resolveWorkspaceSourceViews(plan, mlir::UnknownLoc::get(&context))));
  EXPECT_TRUE(mlir::afir::ProveExpressionsEqual(reader.nodes[0].output.buffer.sizeBytesExpr, "(4 * N)"));
  EXPECT_EQ(reader.nodes[0].output.tensor.shape_exprs, (std::vector<std::string>{"N"}));
  EXPECT_EQ(reader.nodes[0].output.tensor.shape, (std::vector<int64_t>{mlir::ShapedType::kDynamic}));
}

TEST(ScheduleToAfirTest, WorkspaceViewsRejectInvalidOrUnequalStageRelations) {
  mlir::MLIRContext context;
  for (int fault = 0; fault < 7; ++fault) {
    SCOPED_TRACE(fault);
    auto plan = MakeRelatedWorkspacePlan();
    switch (fault) {
      case 0:
        plan.var_relations[0].src_expr = "17";
        break;
      case 1:
        plan.var_relations[0].src_stage = 1;
        break;
      case 2:
        plan.templates.pop_back();
        break;
      case 3:
        plan.var_relations.push_back({1, 0, "A_org_size", "17"});
        break;
      case 4:
        plan.var_relations[0].src_expr = "tile_size";
        break;
      case 5:
        plan.var_relations.clear();
        break;
      case 6:
        plan.templates[1].axes.push_back({0, "a", "original", "A_org_size"});
        plan.templates[1].nodes[1].output.tensor.repeats = {"A_org_size"};
        plan.templates[1].nodes[1].output.tensor.strides = {"1"};
        plan.var_relations.push_back({2, 0, "A_org_size", "17"});
        break;
    }
    mlir::ScopedDiagnosticHandler handler(&context, [](mlir::Diagnostic &) { return mlir::success(); });
    EXPECT_TRUE(failed(mlir::afir::bridge::resolveWorkspaceSourceViews(plan, mlir::UnknownLoc::get(&context))));
  }
}

TEST(ScheduleToAfirTest, WorkspaceViewsRejectUnprovenCrossStageBindings) {
  mlir::MLIRContext context;
  for (int fault = 0; fault < 9; ++fault) {
    SCOPED_TRACE(fault);
    auto plan = MakeCrossStageWorkspacePlan();
    switch (fault) {
      case 0:
        plan.templates[1].nodes[0].output.buffer.bufferId = 18;
        break;
      case 1:
        plan.templates[0].nodes[0].output.tensor.dtype = "int32";
        break;
      case 2:
        plan.templates[0].nodes[0].output.tensor.strides[0] = "8";
        break;
      case 3:
        plan.templates[1].nodes[1].output.tensor.repeats[0] = "3";
        break;
      case 4:
        plan.templates[0].stage_id = 1;
        break;
      case 5: {
        auto other = plan.templates[0];
        other.nodes[0].output.tensor.repeats[0] = "8";
        plan.templates.push_back(other);
        break;
      }
      case 6:
        plan.templates[0].nodes[0].output.tensor.repeats = {"tile_size"};
        plan.templates[0].nodes[0].output.tensor.strides = {"1"};
        break;
      case 7:
        plan.templates[0].nodes[0].type = "Sum";
        break;
      case 8: {
        auto alternative = plan.templates[0];
        alternative.nodes.clear();
        plan.templates.push_back(alternative);
        break;
      }
    }
    std::string diagnostic;
    mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &diag) {
      llvm::raw_string_ostream os(diagnostic);
      diag.print(os);
      return mlir::success();
    });
    EXPECT_TRUE(failed(mlir::afir::bridge::resolveWorkspaceSourceViews(plan, mlir::UnknownLoc::get(&context))));
    EXPECT_NE(diagnostic.find("InvalidWorkspace"), std::string::npos);
  }
}

TEST(ReduceScratchAdapterTest, DecodesLegacyRolesWithoutInventingAnAxis) {
  using namespace mlir::afir;
  auto api = bridge::normalizeLegacyReduceScratch(11, -1);
  auto round = bridge::normalizeLegacyReduceScratch(23, 0);
  ASSERT_TRUE(api);
  ASSERT_TRUE(round);
  EXPECT_EQ(api->id, 11);
  EXPECT_EQ(round->id, 23);
  EXPECT_EQ(api->life_time_axis_id, -1);
  EXPECT_EQ(round->life_time_axis_id, -1);
  EXPECT_EQ(api->reduce_role, ReduceScratchRole::ApiScratch);
  EXPECT_EQ(round->reduce_role, ReduceScratchRole::RoundResult);
  EXPECT_FALSE(bridge::normalizeLegacyReduceScratch(23, 7));
  EXPECT_FALSE(bridge::normalizeLegacyReduceScratch(23, -2));
  EXPECT_FALSE(bridge::normalizeLegacyReduceScratch(-1, 0));
}

TEST(ReduceScratchAdapterTest, SharedStorageUsesNormalizedLifetimesOnlyForReduce) {
  using namespace mlir::afir;
  auto reduce = bridge::translateSchedulerTmpBuffer("Sum", 23, 0);
  auto other = bridge::translateSchedulerTmpBuffer("Exp", 23, 0);
  ASSERT_TRUE(reduce);
  ASSERT_TRUE(other);
  EXPECT_EQ(reduce->life_time_axis_id, -1);
  EXPECT_EQ(reduce->reduce_role, ReduceScratchRole::RoundResult);
  EXPECT_EQ(other->life_time_axis_id, 0);
  EXPECT_FALSE(other->reduce_role);
  EXPECT_FALSE(bridge::translateSchedulerTmpBuffer("Sum", 23, 7));
}

namespace {
af::AscGraph MakeSharedReduceScratchGraph(bool ra) {
  const std::vector<af::Expression> shape = ra ? std::vector<af::Expression>{af::Symbol(1024), af::Symbol(8)}
                                               : std::vector<af::Expression>{af::Symbol(8), af::Symbol(1024)};
  const std::vector<size_t> axes = {ra ? 0U : 1U};
  af::testing::AscGraphBuilder builder(ra ? "shared_reduce_ra" : "shared_reduce_ar");
  builder.Loops(shape).Data("x", 0, af::DT_FLOAT).Load("load", "x").Sum("sum", "load", axes);
  mlir::test::AddReduce<af::ascir_op::Max>(builder, "max", "load", shape, axes, af::DT_FLOAT);
  auto graph = builder.Add("add", "sum", "max").Store("store", "add").Output("out", "store", 0).Build();
  auto &output = graph.FindNode("out")->outputs[0].attr;
  output.repeats = ra ? std::vector<af::Expression>{af::Symbol(1), af::Symbol(8)}
                      : std::vector<af::Expression>{af::Symbol(8), af::Symbol(1)};
  output.strides = ra ? std::vector<af::Expression>{af::Symbol(0), af::Symbol(1)}
                      : std::vector<af::Expression>{af::Symbol(1), af::Symbol(0)};
  return graph;
}

mlir::Value FindScratchTBuf(mlir::Value value) {
  while (auto *owner = value.getDefiningOp()) {
    if (owner->getName().getStringRef() == "ascendc.tbuf.get_tensor") {
      return owner->getOperand(0);
    }
    if (owner->getNumOperands() == 0) {
      break;
    }
    value = owner->getOperand(0);
  }
  return {};
}
// Snapshot the bridge's API field names against CANN kernel_tiling/kernel_tiling.h.
// Header emission uses the official external structs, so compiling generated C++
// alone cannot detect an unused/misspelled modeled member in this metadata.
TEST(ScheduleToAfirTest, TransposeAndPadApiFieldsMatchCannContract) {
  auto graph = af::testing::AscGraphBuilder("api_field_contract")
                   .Loops({2, 37, 23})
                   .Data("x", 0, af::DT_FLOAT)
                   .Load("load", "x")
                   .Transpose("transpose", "load", {0, 2, 1})
                   .Store("store", "transpose", {af::Symbol(2), af::Symbol(23), af::Symbol(37)}, {})
                   .Output("output", "store")
                   .Build();
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused;
  ASSERT_EQ(optimizer.Optimize(graph, fused), af::SUCCESS);
  const std::vector<std::string> padFields = {"srcHeight",
                                              "srcWidth",
                                              "srcOriWidth",
                                              "widthWithoutLastBlock",
                                              "blocksPerRow",
                                              "heightTiling",
                                              "heightFractal",
                                              "heightFractalTail",
                                              "mainLoopOffset",
                                              "tailBlockOffset",
                                              "tmpBuffer1BlockNum",
                                              "tmpBuffer1RowNum",
                                              "tmpBuffer2Offset",
                                              "widthTiling",
                                              "widthFractal",
                                              "widthFractalTail",
                                              "widthFractalTailAlingned",
                                              "brcbTiling",
                                              "brcbFractal",
                                              "brcbFractalTail",
                                              "maxRepeatTimes",
                                              "brcbTilingRepeatTimes",
                                              "brcbTilingRepeatTimesTail",
                                              "brcbFractalTailRepeatTimes",
                                              "brcbFractalTailRepeatTimesTail",
                                              "reserved"};
  size_t padCount = 0U, transposeCount = 0U;
  mlir::MLIRContext context;
  ASSERT_FALSE(fused.node_idx_to_scheduled_results.empty());
  for (size_t planId = 0; planId < fused.node_idx_to_scheduled_results[0].size(); ++planId) {
    auto module =
        mlir::afir::bridge::PlanToAfir(context, fused, fused.node_idx_to_scheduled_results[0][planId], 0, planId);
    ASSERT_TRUE(module);
    module->walk([&](mlir::func::FuncOp func) {
      auto fields = func->getAttrOfType<mlir::ArrayAttr>("afir.api_tiling_fields");
      if (!fields) {
        return;
      }
      for (auto entry : fields) {
        auto item = mlir::cast<mlir::DictionaryAttr>(entry);
        std::vector<std::string> actual;
        for (auto name : item.getAs<mlir::ArrayAttr>("fields")) {
          actual.push_back(mlir::cast<mlir::StringAttr>(name).getValue().str());
        }
        const auto type = item.getAs<mlir::StringAttr>("type").getValue();
        if (type == "PadTiling") {
          ++padCount;
          EXPECT_EQ(actual, padFields);
        }
        if (type == "ConfusionTransposeTiling") {
          ++transposeCount;
          std::vector<std::string> expected;
          for (size_t i = 0; i < 18; ++i) {
            expected.push_back("param" + std::to_string(i));
          }
          EXPECT_EQ(actual, expected);
        }
      }
    });
  }
  EXPECT_GT(padCount, 0U);
  EXPECT_GT(transposeCount, 0U);
}

}  // namespace

template <typename PlanT>
static bool CheckScheduledReduceScratchSharing(PlanT &plan, size_t &sharedApi, size_t &roundResults) {
  constexpr size_t kPairedReduceCount = 2;
  bool hasPair = false;
  for (auto &group : plan.schedule_groups) {
    for (auto &impl : group.impl_graphs) {
      std::map<int64_t, size_t> uses;
      for (const auto &node : impl.GetAllNodes()) {
        for (const auto &tmp : node->attr.tmp_buffers) {
          ++uses[tmp.id];
        }
        for (const auto &out : node->outputs()) {
          if (out->attr.mem.alloc_type == af::AllocType::kAllocTypeBuffer) {
            ++uses[out->attr.buf.id];
          }
        }
      }
      size_t reductions = 0;
      std::map<int64_t, size_t> apiUses;
      for (const auto &node : impl.GetAllNodes()) {
        if (!mlir::afir::bridge::isReduceNodeType(node->GetType())) {
          continue;
        }
        ++reductions;
        for (const auto &tmp : node->attr.tmp_buffers) {
          if (tmp.buf_desc.life_time_axis_id == 0) {
            ++roundResults;
            EXPECT_EQ(uses[tmp.id], 1U) << "round_result id=" << tmp.id;
          } else if (tmp.buf_desc.life_time_axis_id == -1) {
            ++apiUses[tmp.id];
          }
        }
      }
      for (const auto &[id, count] : apiUses) {
        if (count > 1) {
          ++sharedApi;
        }
      }
      hasPair |= reductions >= kPairedReduceCount;
    }
  }
  return hasPair;
}

TEST(ScheduleToAfirTest, AllocatorSharesApiScratchButKeepsRoundResultsDistinct) {
  for (bool ra : {false, true}) {
    SCOPED_TRACE(ra ? "RA" : "AR");
    auto graph = MakeSharedReduceScratchGraph(ra);
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    ascir::FusedScheduledResult result;
    ASSERT_EQ(optimizer.Optimize(graph, result), af::SUCCESS);
    size_t checked = 0;
    size_t sharedApi = 0;
    size_t roundResults = 0;
    for (auto &plans : result.node_idx_to_scheduled_results) {
      for (auto &plan : plans) {
        const bool hasPair = CheckScheduledReduceScratchSharing(plan, sharedApi, roundResults);
        mlir::MLIRContext context;
        auto module = mlir::afir::bridge::PlanToAfir(context, result, plan, 0, checked++);
        ASSERT_TRUE(module);
        mlir::PassManager pm(&context);
        pm.addPass(mlir::ascend::kernel::createAFIRKernelBodySignaturePass());
        pm.addPass(mlir::ascend::kernel::createAscendCLoopGeneratePass());
        pm.addPass(mlir::afir::createAFIRScopeReduceScratchPass());
        pm.addPass(mlir::ascend::kernel::createAFIRMemoryAllocatePass());
        ASSERT_TRUE(succeeded(pm.run(*module)));
        size_t pairedFunctions = 0;
        for (auto function : module->getOps<mlir::func::FuncOp>()) {
          llvm::SmallVector<mlir::Value> apis, rounds;
          function.walk([&](mlir::Operation *op) {
            auto reduce = mlir::dyn_cast<mlir::afir::SumOp>(op);
            auto maximum = mlir::dyn_cast<mlir::afir::MaxOp>(op);
            if (!reduce && !maximum) {
              return;
            }
            auto scratch = reduce ? reduce.getScratch() : maximum.getScratch();
            ASSERT_EQ(scratch.size(), 2U);
            apis.push_back(FindScratchTBuf(scratch[0]));
            rounds.push_back(FindScratchTBuf(scratch[1]));
          });
          if (apis.size() == 2) {
            ++pairedFunctions;
            ASSERT_TRUE(apis[0]);
            ASSERT_TRUE(apis[1]);
            ASSERT_TRUE(rounds[0]);
            ASSERT_TRUE(rounds[1]);
            EXPECT_EQ(apis[0], apis[1]);
            EXPECT_NE(rounds[0], rounds[1]);
            EXPECT_NE(apis[0], rounds[0]);
            EXPECT_NE(apis[0], rounds[1]);
          }
        }
        if (hasPair) {
          EXPECT_GT(pairedFunctions, 0U);
        }
      }
    }
    EXPECT_GT(checked, 0U);
    EXPECT_GT(roundResults, 0U);
    // RA scheduling may split the two reductions into separate implementation graphs.
    if (!ra) {
      EXPECT_GT(sharedApi, 0U);
    }
  }
}

TEST(ScheduleToAfirTest, RejectsArtificialRoundResultSharingWithNonReduceAxisLifetime) {
  auto graph = MakeSharedReduceScratchGraph(false);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult result;
  ASSERT_EQ(optimizer.Optimize(graph, result), af::SUCCESS);
  size_t checked = 0;
  for (auto &plans : result.node_idx_to_scheduled_results) {
    for (auto &plan : plans) {
      for (auto &group : plan.schedule_groups) {
        for (auto &impl : group.impl_graphs) {
          auto add = impl.FindNode("add");
          if (!add) {
            continue;
          }
          for (const auto &node : impl.GetAllNodes()) {
            if (node->GetType() != "Sum") {
              continue;
            }
            for (const auto &tmp : node->attr.tmp_buffers) {
              if (tmp.buf_desc.life_time_axis_id != 0) {
                continue;
              }
              mlir::MLIRContext context;
              ASSERT_TRUE(mlir::afir::bridge::PlanToAfir(context, result, plan, 0, checked));
              // Deliberately violate the allocator's dedicated round-result ID contract.
              add->attr.tmp_buffers.push_back(tmp);
              std::string message;
              mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &diagnostic) {
                llvm::raw_string_ostream os(message);
                diagnostic.print(os);
                return mlir::success();
              });
              EXPECT_FALSE(mlir::afir::bridge::PlanToAfir(context, result, plan, 0, checked++));
              EXPECT_NE(message.find("InvalidLifetime: tmp_buffer " + std::to_string(tmp.id) +
                                     " has conflicting life_time_axis_id values"),
                        std::string::npos)
                  << message;
              add->attr.tmp_buffers.pop_back();
            }
          }
        }
      }
    }
  }
  EXPECT_GT(checked, 0U);
}

TEST(ScheduleToAfirTest, RejectsReduceScratchDtoWithoutNormalizedRole) {
  using namespace mlir::afir;
  auto plan = MakeArgMaxPairPlan();
  auto &tmpl = plan.templates[0];
  tmpl.outputs.resize(1);
  tmpl.nodes.pop_back();
  auto &reduce = tmpl.nodes[1];
  reduce.type = "Sum";
  reduce.outputs.resize(1);
  reduce.output_names = {"phase"};
  reduce.tmp_buffers = {{3, -1, ReduceScratchRole::ApiScratch}, {4, -1, ReduceScratchRole::RoundResult}};
  tmpl.nodes[2].input_names = {"phase"};
  mlir::MLIRContext context;
  ASSERT_TRUE(bridge::BuildAfirModule(context, plan));
  reduce.tmp_buffers[1].reduce_role.reset();
  std::string message;
  mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &diagnostic) {
    llvm::raw_string_ostream os(message);
    diagnostic.print(os);
    return mlir::success();
  });
  EXPECT_FALSE(bridge::BuildAfirModule(context, plan));
  EXPECT_NE(message.find("InvalidBinding: reduce scratch 4 has no normalized role"), std::string::npos) << message;
}
