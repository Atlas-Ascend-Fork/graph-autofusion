/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cctype>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

#include <gtest/gtest.h>

#include "ascir_utils.h"
#include "codegen.h"
#include "common_utils.h"
#include "common/platform_context.h"
#include "optimize.h"
#include "test/Support/TestEnvironment.h"
#include "tests/framework/easy_asc_graph/asc_graph_builder.h"

namespace {
const std::filesystem::path DUMP_DIR_BASE = std::filesystem::path(AUTOFUSE_PROJECT_ROOT) / "dump";

af::AscGraph BuildLoadAddStoreGraph() {
  using af::DT_FLOAT16;
  using af::testing::AscGraphBuilder;

  return AscGraphBuilder("load_add_store")
      .Loops({af::testing::Sym("s0"), af::testing::Sym("s1"), af::testing::Sym("s2")})
      .Data("x", 0, DT_FLOAT16)
      .Data("bias", 1, DT_FLOAT16)
      .Load("load_x", "x")
      .Load("load_bias", "bias")
      .Add("add", "load_x", "load_bias")
      .Store("store", "add")
      .Output("y", "store", 0, DT_FLOAT16)
      .Build();
}

// Shared legacy E2E input shape, exercised through GenerateForInductor.
af::AscGraph BuildLegacyInductorTransposeGraph() {
  using af::testing::AscGraphBuilder;
  const auto s0 = af::Symbol(2);
  const auto s1 = af::Symbol(37);
  const auto s2 = af::Symbol(23);
  auto graph = AscGraphBuilder("legacy_inductor_transpose_stage")
                   .Loops({s0, s1, s2})
                   .Data("x", 0, af::DT_FLOAT)
                   .Load("load", "x")
                   .Transpose("transpose", "load", {0, 2, 1})
                   .Store("store", "transpose", {s0, s2, s1}, {s2 * s1, s1, af::sym::kSymbolOne})
                   .Output("y", "store", 0, af::DT_FLOAT)
                   .Build();
  auto &output = graph.FindNode("y")->outputs()[0]->attr;
  const auto &stored = graph.FindNode("store")->outputs()[0]->attr;
  output.axis = stored.axis;
  output.repeats = stored.repeats;
  output.strides = stored.strides;
  return graph;
}

void ExpectPassDumps(const std::filesystem::path &dump_dir) {
  ASSERT_TRUE(std::filesystem::exists(dump_dir));
  std::filesystem::path pass_dump_dir;
  for (const auto &entry : std::filesystem::recursive_directory_iterator(dump_dir)) {
    if (entry.is_directory() && entry.path().filename() == "mlir_subgraph_0_plan_0") {
      pass_dump_dir = entry.path();
      break;
    }
  }
  ASSERT_FALSE(pass_dump_dir.empty());
  const std::vector<std::string> expected_passes = {
      "afir-kernel-body-signature", "afir-loop-generate", "afir-memory-allocate", "ascend-select-implementation",
      "convert-afir-to-ascendc",    "gen-tiling-data",    "gen-kernel-entry"};
  for (const auto &pass : expected_passes) {
    bool found = false;
    for (const auto &entry : std::filesystem::recursive_directory_iterator(pass_dump_dir)) {
      if (!entry.is_regular_file() || entry.path().filename().string().find(pass) == std::string::npos) {
        continue;
      }
      const auto filename = entry.path().filename().string();
      ASSERT_GE(filename.size(), 3U);
      EXPECT_TRUE(std::isdigit(static_cast<unsigned char>(filename[0])) != 0);
      EXPECT_TRUE(std::isdigit(static_cast<unsigned char>(filename[1])) != 0);
      EXPECT_EQ(filename[2], '_');
      found = true;
      break;
    }
    EXPECT_TRUE(found) << "missing pass snapshot for " << pass;
  }
}

void DumpCodegenResult(const std::filesystem::path &artifact_dir, const std::string &graph_name,
                       const codegen::CodegenResult &result) {
  std::filesystem::create_directories(artifact_dir);
  std::ofstream kernel_file(artifact_dir / (graph_name + "_op_kernel.cpp"));
  std::ofstream tiling_data_file(artifact_dir / "autofuse_tiling_data.h");
  std::ofstream tiling_file(artifact_dir / (graph_name + "_tiling_func.cpp"));
  ASSERT_TRUE(kernel_file.is_open());
  ASSERT_TRUE(tiling_data_file.is_open());
  ASSERT_TRUE(tiling_file.is_open());
  kernel_file << result.kernel;
  tiling_data_file << result.tiling_data;
  tiling_file << result.tiling;
  ASSERT_TRUE(kernel_file.good());
  ASSERT_TRUE(tiling_data_file.good());
  ASSERT_TRUE(tiling_file.good());
}

}  // namespace

TEST(CodegenOrchestrationSt, ManualGraphScheduleAndMlirCodegen) {
  const std::filesystem::path dump_dir = DUMP_DIR_BASE / "ManualGraphScheduleAndMlirCodegen";
  mlir::test::ScopedEnvVar dfx_flags(
      "AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=mlir;--codegen_compile_debug=true;--debug_dir=" + dump_dir.string());

  auto graph = BuildLoadAddStoreGraph();
  graph.SetTilingKey(10);

  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused_schedule_result;
  ASSERT_EQ(optimizer.Optimize(graph, fused_schedule_result), af::SUCCESS);
  ASSERT_FALSE(fused_schedule_result.node_idx_to_scheduled_results.empty());
  ASSERT_FALSE(fused_schedule_result.node_idx_to_scheduled_results[0].empty());
  ASSERT_EQ(fused_schedule_result.input_nodes.size(), 2U);
  ASSERT_EQ(fused_schedule_result.output_nodes.size(), 1U);
  EXPECT_FALSE(ascgen_utils::IsCubeFusedScheduled(fused_schedule_result));

  codegen::Codegen codegen(codegen::CodegenOptions{.soc_version = "Ascend910B1"});
  codegen::CodegenResult result;
  ASSERT_EQ(codegen.GenerateForInductor(fused_schedule_result, result), af::SUCCESS);

  EXPECT_FALSE(result.kernel.empty());
  EXPECT_FALSE(result.tiling.empty());
  EXPECT_FALSE(result.tiling_data.empty());
  EXPECT_NE(result.tiling.find("extern \"C\" int64_t AutofuseTiling("), std::string::npos);
  EXPECT_NE(result.kernel.find("AutofuseLaunch"), std::string::npos);
  EXPECT_NE(result.tiling_data.find("AutofuseTilingData"), std::string::npos);
  const std::filesystem::path artifact_dir = dump_dir / "artifacts";
  DumpCodegenResult(artifact_dir, "mlir_st_load_add_store", result);
  EXPECT_TRUE(std::filesystem::exists(artifact_dir / "mlir_st_load_add_store_op_kernel.cpp"));
  EXPECT_TRUE(std::filesystem::exists(artifact_dir / "autofuse_tiling_data.h"));
  EXPECT_TRUE(std::filesystem::exists(artifact_dir / "mlir_st_load_add_store_tiling_func.cpp"));
  ExpectPassDumps(dump_dir);

  std::filesystem::remove_all(dump_dir);
}

TEST(CodegenOrchestrationSt, LegacyInductorTransposeStageContractsAndStableArtifacts) {
  ge::PlatformContext::GetInstance().Reset();
  const auto dump_dir = DUMP_DIR_BASE / "LegacyInductorTransposeStageContracts";
  mlir::test::ScopedEnvVar flags(
      "AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=mlir;--codegen_compile_debug=true;--debug_dir=" + dump_dir.string());
  ::ascir::utils::ResetDumpConfig();
  auto graph = BuildLegacyInductorTransposeGraph();
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused;
  ASSERT_EQ(optimizer.Optimize(graph, fused), af::SUCCESS);
  ASSERT_EQ(fused.input_nodes.size(), 1U);
  ASSERT_EQ(fused.output_nodes.size(), 1U);
  codegen::Codegen codegen(codegen::CodegenOptions{.soc_version = "Ascend910B1"});
  codegen::CodegenResult first;
  ASSERT_EQ(codegen.GenerateForInductor(fused, first), af::SUCCESS);
  EXPECT_NE(first.kernel.find("_subgraph0_plan"), std::string::npos);
  EXPECT_NE(first.kernel.find("codegen::ConfusionTranspose<"), std::string::npos);
  EXPECT_NE(first.kernel.find("AscendC::TQue"), std::string::npos);
  EXPECT_NE(first.kernel.find("InitBuffer("), std::string::npos);
  EXPECT_NE(first.tiling_data.find("ConfusionTransposeTiling"), std::string::npos);
  EXPECT_NE(first.tiling_data.find("q0_size"), std::string::npos);
  EXPECT_NE(first.tiling.find("GetTilingDataRepr"), std::string::npos);
  EXPECT_NE(first.tiling.find(".param17 = "), std::string::npos);
  ExpectPassDumps(dump_dir);
  // MLIR has different source from legacy; pin repeatable artifacts and semantic
  // contracts instead of comparing bytes to a legacy-generated C++ snapshot.
  codegen::CodegenResult repeated;
  ASSERT_EQ(codegen.GenerateForInductor(fused, repeated), af::SUCCESS);
  EXPECT_EQ(first.kernel, repeated.kernel);
  EXPECT_EQ(first.tiling_data, repeated.tiling_data);
  EXPECT_EQ(first.tiling, repeated.tiling);
  std::filesystem::remove_all(dump_dir);
  ::ascir::utils::ResetDumpConfig();
}

TEST(CodegenOrchestrationSt, LegacyInductorRejectsInvalidScheduledTransposeWithoutFallback) {
  ge::PlatformContext::GetInstance().Reset();
  mlir::test::ScopedEnvVar strict("AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=mlir");
  auto graph = BuildLegacyInductorTransposeGraph();
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused;
  ASSERT_EQ(optimizer.Optimize(graph, fused), af::SUCCESS);
  codegen::Codegen codegen(codegen::CodegenOptions{.soc_version = "Ascend910B1"});
  codegen::CodegenResult valid;
  ASSERT_EQ(codegen.GenerateForInductor(fused, valid), af::SUCCESS);
  // Exact corruption used by legacy LoadTransposeStoreInvalidTransposeCodegen.
  size_t corrupted = 0;
  for (auto &results : fused.node_idx_to_scheduled_results) {
    for (auto &result : results) {
      for (auto &group : result.schedule_groups) {
        for (auto &candidate : group.impl_graphs) {
          for (const auto &node : candidate.GetAllNodes()) {
            if (node->GetType() != "Transpose") {
              continue;
            }
            ASSERT_FALSE(node->inputs().empty());
            node->outputs()[0]->attr.vectorized_axis = node->inputs()[0]->attr.vectorized_axis;
            node->outputs()[0]->attr.vectorized_strides = node->inputs()[0]->attr.vectorized_strides;
            ++corrupted;
          }
        }
      }
    }
  }
  ASSERT_GT(corrupted, 0U);
  codegen::CodegenResult result;
  ASSERT_EQ(codegen.GenerateForInductor(fused, result), af::FAILED);
  EXPECT_TRUE(result.kernel.empty());
}
