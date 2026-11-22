/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Migrated from TestCodegenTiling's four Inductor tests using the interleaved
// Transpose API-tiling fixture. TF backend and optimizer-only tests are excluded.
#include "bridge/AttSchemaAdapter/AttSchemaAdapter.h"
#include "bridge/CodegenOrchestration/CodegenOrchestration.h"
#include "optimize/optimize.h"
#include "framework/easy_asc_graph/asc_graph_builder.h"
#include "artifact/KernelSource/Emit/TilingDataHeaderEmitter.h"
#include "conversion/AFIRToAscendC/AFIRToAscendCPasses.h"
#include "conversion/AFIRToAscendC/AscendCKernelABI.h"
#include "conversion/AFIRToAscendC/AscendCAttributes.h"
#include "mlir/Pass/Pass.h"
#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFPlan/IR/AFPlan.h"
#include "ascir/Dialect/EmitAsc/IR/EmitAsc.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "test/Support/TestEnvironment.h"
#include "common/platform_context.h"
#include "common_utils.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "gen_tiling_impl.h"
#include <gtest/gtest.h>

namespace {
using mlir::succeeded;
static ascir::ScheduleGroup GenScheduleGroupWithInterleavedApiTilingField() {
  constexpr int64_t kSecondDimSize = 32;
  constexpr int64_t kThirdDimSize = 128;
  ascir::ImplGraph graph0("graph0");
  auto s0 = graph0.CreateSizeVar("s0");
  auto fixed_s1 = graph0.CreateSizeVar(kSecondDimSize);
  auto fixed_s2 = graph0.CreateSizeVar(kThirdDimSize);
  auto z0 = graph0.CreateAxis("z0", s0);
  auto z1 = graph0.CreateAxis("z1", fixed_s1);
  auto z2 = graph0.CreateAxis("z2", fixed_s2);

  af::ascir_op::Data data("data", graph0);
  data.y.dtype = ge::DT_FLOAT;
  *data.y.axis = {z0.id, z1.id, z2.id};
  *data.y.repeats = {s0, fixed_s1, fixed_s2};
  *data.y.strides = {fixed_s1 * fixed_s2, fixed_s2, af::ops::One};

  af::ascir_op::Load load("load");
  graph0.AddNode(load);
  load.x = data.y;
  load.y.dtype = ge::DT_FLOAT;
  *load.y.axis = {z0.id, z1.id, z2.id};
  *load.y.repeats = {s0, fixed_s1, fixed_s2};
  *load.y.strides = {fixed_s1 * fixed_s2, fixed_s2, af::ops::One};
  *load.y.vectorized_axis = {z0.id, z1.id, z2.id};
  *load.y.vectorized_strides = {fixed_s1 * fixed_s2, fixed_s2, af::ops::One};

  af::ascir_op::Transpose transpose("Transpose");
  graph0.AddNode(transpose);
  transpose.x = load.y;
  transpose.y.dtype = ge::DT_FLOAT;
  *transpose.y.axis = {z0.id, z2.id, z1.id};
  *transpose.y.repeats = {s0, fixed_s2, fixed_s1};
  *transpose.y.strides = {fixed_s1 * fixed_s2, fixed_s1, af::ops::One};
  *transpose.y.vectorized_axis = {z0.id, z2.id, z1.id};
  *transpose.y.vectorized_strides = {fixed_s1 * fixed_s2, fixed_s1, af::ops::One};
  graph0.FindNode("Transpose")->outputs[0].attr.que.id = 0;

  ascir::ImplGraph graph1("graph1");
  auto s1 = graph1.CreateSizeVar("s1");
  (void)graph1.CreateAxis("z1", s1);

  ascir::ScheduleGroup schedule_group;
  schedule_group.impl_graphs = {graph0, graph1};
  return schedule_group;
}

static ascir::FusedScheduledResult GenSingleGroupFusedScheduleResultWithApiTilingField() {
  ascir::ScheduledResult scheduled_result;
  scheduled_result.schedule_groups.push_back(GenScheduleGroupWithInterleavedApiTilingField());

  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.node_idx_to_scheduled_results.push_back({scheduled_result});
  return fused_schedule_result;
}

class LegacyInductorTransposeTest : public testing::Test {
 protected:
  mlir::MLIRContext context;
  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
    context.getOrLoadDialect<mlir::afir::AFIRDialect>();
    context.getOrLoadDialect<mlir::afplan::AFPlanDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::emitasc::EmitAscDialect>();
  }

  // Same declaration inputs as the legacy fixture: template 0 contributes s0
  // and Transpose, template 1 contributes s1; q0 belongs after both templates.
  mlir::OwningOpRef<mlir::ModuleOp> MakePlan(bool multiGroup) {
    using namespace mlir;
    OpBuilder b(&context);
    auto module = ModuleOp::create(b.getUnknownLoc(), "transpose");
    module->setAttr("afir.module_kind", b.getStringAttr("plan"));
    module->setAttr(afplan::kCandidateAttrName, afplan::CandidateAttr::get(&context, 0, 0));
    const unsigned stageCount = multiGroup ? 2U : 1U;
    for (unsigned stage = 0; stage < stageCount; ++stage) {
      for (unsigned tmpl = 0; tmpl < (stage == 0 ? 2U : 1U); ++tmpl) {
        auto base = ascend::kernel::TilingSchema::get(&context, b.getStringAttr("Base"), b.getArrayAttr({}),
                                                      b.getArrayAttr({}));
        auto fn = func::FuncOp::create(b.getUnknownLoc(),
                                       "stage" + std::to_string(stage) + "_template" + std::to_string(tmpl),
                                       b.getFunctionType({base.getType()}, {}));
        fn->setAttr(afplan::kVariantAttrName, afplan::VariantAttr::get(&context, stage, tmpl));
        SmallVector<afir::ExprAttr> vars;
        if (stage == 0) {
          vars.push_back(afir::ExprAttr::get(&context, b.getStringAttr(tmpl == 0 ? "s0" : "s1")));
        }
        fn->setAttr(afir::kAscGraphAttrName,
                    afir::AscGraphAttrGroupsAttr::get(&context, -1, {}, afir::AscGraphType::Compute, vars));
        afir::setKernelArgAttr(fn, 0, afir::KernelArgRole::Tiling);
        if (stage == 0 && tmpl == 0) {
          SmallVector<Attribute> members;
          constexpr unsigned kExpectedTransposeCases = 18;
          for (unsigned i = 0; i < kExpectedTransposeCases; ++i) {
            members.push_back(b.getStringAttr("param" + std::to_string(i)));
          }
          fn->setAttr(afir::kApiTilingFieldsAttrName,
                      b.getArrayAttr({b.getDictionaryAttr(
                          {b.getNamedAttr(afir::kApiTilingTypeKey, b.getStringAttr("ConfusionTransposeTiling")),
                           b.getNamedAttr(afir::kApiTilingFieldKey, b.getStringAttr("Transpose_tilingData_0")),
                           b.getNamedAttr(afir::kApiTilingFieldsKey, b.getArrayAttr(members))})}));
          fn->setAttr(afir::kQueueIdsAttrName, b.getDenseI64ArrayAttr({0}));
        }
        auto *block = fn.addEntryBlock();
        OpBuilder body(block, block->begin());
        body.create<func::ReturnOp>(fn.getLoc());
        module.push_back(fn);
      }
    }
    return module;
  }

  void ExpectReprOrder(bool multiGroup) {
    auto module = MakePlan(multiGroup);
    mlir::PassManager pm(&context);
    pm.addPass(mlir::ascend::kernel::createGenTilingDataPass());
    ASSERT_TRUE(succeeded(pm.run(*module)));
    auto attr = (*module)->getAttrOfType<mlir::TypeAttr>(mlir::ascend::kernel::kAscendCTilingSchemaAttrName);
    ASSERT_TRUE(attr);
    std::string source;
    llvm::raw_string_ostream os(source);
    ASSERT_TRUE(succeeded(
        mlir::ascend::kernel::emitTilingDataRepr(mlir::ascend::kernel::getTilingSchema(attr.getValue()), os)));
    os.flush();
    const auto s0 = source.find(".s0 = ");
    const auto api = source.find(".Transpose_tilingData_0 = {");
    const auto s1 = source.find(".s1 = ");
    const auto q0 = source.find(".q0_size = ");
    ASSERT_NE(s0, std::string::npos);
    ASSERT_NE(api, std::string::npos);
    ASSERT_NE(s1, std::string::npos);
    ASSERT_NE(q0, std::string::npos);
    EXPECT_LT(s0, api);
    EXPECT_LT(api, s1);
    EXPECT_LT(s1, q0);
    EXPECT_NE(source.find(".param17 = "), std::string::npos);
    if (multiGroup) {
      EXPECT_NE(source.find(".graph0_result0_g1_tiling_data = {"), std::string::npos);
    }
  }

  void GenerateAtt(mlir::bridge::TilingCodeBundle &bundle, bool disablePgo = false) {
    const auto fused = GenSingleGroupFusedScheduleResultWithApiTilingField();
    mlir::common::TilingFieldSchema schema;
    ASSERT_TRUE(mlir::common::BuildTilingFieldSchema(fused, schema));
    mlir::bridge::PlanSelection selection;
    selection.template_ids_per_stage = {{0, 1}};
    mlir::bridge::AttOptions options;
    if (disablePgo) {
      options.emplace(att::kInternalEnableAutofusePgo, "false");
    }
    std::string diagnostic;
    ASSERT_EQ(mlir::bridge::RunAttForCandidates({"transpose", fused, {selection}, schema, options, true}, bundle,
                                                &diagnostic),
              af::SUCCESS)
        << diagnostic;
    ASSERT_TRUE(bundle.schema_validated);
  }
};

TEST_F(LegacyInductorTransposeTest, GetTilingDataReprShouldKeepSingleGroupApiTilingFieldInDeclarationOrder) {
  ExpectReprOrder(false);
}
TEST_F(LegacyInductorTransposeTest, GetTilingDataReprShouldKeepMultiGroupApiTilingFieldInDeclarationOrder) {
  ExpectReprOrder(true);
}
TEST_F(LegacyInductorTransposeTest, SplitHeaderApiTilingSourceShouldIncludeApiHeaders) {
  mlir::bridge::TilingCodeBundle bundle;
  GenerateAtt(bundle);
  ASSERT_FALSE(HasFatalFailure());
  bool found = false;
  for (const auto &[key, source] : bundle.tiling_functions) {
    if (key == "TilingHead" || source.find("GetConfusionTranspose") == std::string::npos) {
      continue;
    }
    found = true;
    EXPECT_NE(source.find("#include \"graph/tensor.h\""), std::string::npos);
    EXPECT_NE(source.find("#ifndef AUTOFUSE_CONFUSION_TRANSPOSE_TILING_DEFS"), std::string::npos);
    EXPECT_NE(source.find("const uint32_t ONE_BLK_SIZE = 32;"), std::string::npos);
    EXPECT_NE(source.find("const uint32_t BLOCK_CUBE = 16;"), std::string::npos);
  }
  EXPECT_TRUE(found);
}
TEST_F(LegacyInductorTransposeTest, DisabledInductorPgoShouldOverrideAttPgoEnvironment) {
  mlir::test::ScopedEnvVar pgo("AUTOFUSE_FLAGS", "--autofuse_enable_pgo=true");
  mlir::bridge::TilingCodeBundle bundle;
  GenerateAtt(bundle, true);
  ASSERT_FALSE(HasFatalFailure());
  for (const auto &[key, source] : bundle.tiling_functions) {
    EXPECT_EQ(source.find("PGOProfileReuseGroup"), std::string::npos) << key;
    EXPECT_EQ(source.find("PGOByCoreNumSearchTilingKey"), std::string::npos) << key;
    EXPECT_NE(key, "PgoRunner");
  }
}
TEST_F(LegacyInductorTransposeTest, NativeInductorKeepsPgoDisabledForTranspose) {
  mlir::test::ScopedEnvVar pgo("AUTOFUSE_FLAGS", "--autofuse_enable_pgo=true");
  mlir::test::ScopedEnvVar strict("AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=mlir");
  auto graph = af::testing::AscGraphBuilder("inductor_transpose_pgo")
                   .Loops({2, 37, 23})
                   .Data("x", 0, af::DT_FLOAT)
                   .Load("load", "x")
                   .Transpose("transpose", "load", {0, 2, 1})
                   .Store("store", "transpose")
                   .Output("out", "store")
                   .Build();
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused;
  ASSERT_EQ(optimizer.Optimize(graph, fused), af::SUCCESS);
  codegen::CodegenResult result;
  ASSERT_EQ(mlir::bridge::RunMlirCodegen(fused, mlir::bridge::MlirCallerScene::kInductor, "Ascend910B1", result),
            mlir::bridge::MlirCodegenStatus::kSucceeded);
  EXPECT_NE(result.kernel.find("_subgraph0_plan"), std::string::npos);
  EXPECT_NE(result.tiling.find("GetTilingDataRepr"), std::string::npos);
  EXPECT_EQ(result.tiling.find("PGOProfileReuseGroup"), std::string::npos);
  EXPECT_EQ(result.tiling.find("PGOByCoreNumSearchTilingKey"), std::string::npos);
}
}  // namespace
