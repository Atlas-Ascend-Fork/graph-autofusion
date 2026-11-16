/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <fstream>
#include <gtest/gtest.h>

#include "backend_common.h"
#include "codegen.h"
#include "common/common_utils.h"
#include "common/platform_context.h"
#include "optimize.h"
#include "runtime_stub.h"
#include "share_graph.h"
#if defined(ARGMAX_SINGLE_ROW) || defined(ARGMAX_ARA)
#include "asc_graph_builder.h"
#endif

class TestBackendArgmaxStoreE2e : public testing::Test {
 protected:
  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
    af::InstallV35SocPerfRuntimeStub();
  }
  void TearDown() override {
    ge::RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
  }
};

#ifdef ARGMAX_ARA
TEST_F(TestBackendArgmaxStoreE2e, GenerateAraCommon) {
#else
TEST_F(TestBackendArgmaxStoreE2e, GenerateRCoreOnly) {
#endif
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused_result;
  fused_result.node_idx_to_scheduled_results.emplace_back();
#if defined(ARGMAX_SINGLE_ROW) || defined(ARGMAX_ARA)
  af::testing::AscGraphBuilder builder("argmax_store_test");
#ifdef ARGMAX_ARA
  auto graph =
      builder.Loops({af::Symbol(4), af::Symbol(8), af::Symbol(2048), af::Symbol(32), af::Symbol(16)})
          .Data("data", 0, af::DT_FLOAT)
          .Load("load", "data")
          .Op<af::ascir_op::Argmax>(
              "argmax", {"load"}, {af::Symbol(4), af::Symbol(8), af::Symbol(2048), af::Symbol(1), af::Symbol(16)},
              {af::Symbol(262144), af::Symbol(32768), af::Symbol(16), af::Symbol(0), af::Symbol(1)}, af::DT_INT64)
          .Store("store", "argmax")
          .Output("output", "store", 0, af::DT_INT64)
          .Build();
#else
  const auto columns = builder.SizeVar("s1");
  auto graph = builder.Loops({af::Symbol(1), columns})
                   .Data("data", 0, af::DT_FLOAT)
                   .Load("load", "data")
                   .Op<af::ascir_op::Argmax>("argmax", {"load"}, {af::Symbol(1), af::Symbol(1)},
                                             {af::Symbol(1), af::Symbol(0)}, af::DT_INT64)
                   .Store("store", "argmax")
                   .Output("output", "store", 0, af::DT_INT64)
                   .Build();
#endif
  ASSERT_EQ(optimizer.Optimize(graph, fused_result), 0);
#else
  ASSERT_EQ(optimizer.Optimize(ascir::ShareGraph::ArgmaxStoreFusedGraph(2), fused_result), 0);
#endif
  ASSERT_EQ(fused_result.node_idx_to_scheduled_results.size(), 1U);
  auto &schedules = fused_result.node_idx_to_scheduled_results.front();
#ifdef ARGMAX_ARA
  // Keep all ordinary Common candidates; the generated tiling selects the real path later.
  schedules.erase(std::remove_if(schedules.begin(), schedules.end(),
                                 [](const ascir::ScheduledResult &schedule) {
                                   return schedule.semantic_name != ascgen_utils::template_semantic::kReduceCommon;
                                 }),
                  schedules.end());
  ASSERT_FALSE(schedules.empty());
#else
  // Keep both RCore phases in one result and remove all fallback templates.
  schedules.erase(std::remove_if(schedules.begin(), schedules.end(),
                                 [](const ascir::ScheduledResult &schedule) {
                                   return schedule.semantic_name != ascgen_utils::template_semantic::kReduceRCore;
                                 }),
                  schedules.end());
  ASSERT_EQ(schedules.size(), 1U);
  ASSERT_EQ(schedules.front().schedule_groups.size(), 2U);
#ifdef ARGMAX_SINGLE_ROW
  ASSERT_TRUE(schedules.front().enable_workspace_alignment);
#endif
#endif

  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen::CodegenResult result;
#ifdef ARGMAX_ARA
  ASSERT_EQ(codegen.Generate({}, fused_result, result), 0);
  ASSERT_NE(result.kernel.find("ArgMaxWithValueExtend<float, AscendC::Pattern::Reduce::RA>"), std::string::npos);
#else
  ASSERT_EQ(codegen.Generate({{"s0", "stub_s0"}, {"s1", "stub_s1"}}, fused_result, result), 0);
#endif
  const auto paths = splitString(KERNEL_SRC_LIST, ':');
  ASSERT_EQ(paths.size(), 3U);
  std::ofstream kernel(paths[0]);
  std::ofstream tiling(paths[1]);
  std::ofstream tiling_data(paths[2]);
  kernel << R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling) AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)" << RemoveSubDirInclude(result.kernel);
  tiling << result.tiling;
  tiling_data << result.tiling_data;
  kernel.close();
  tiling.close();
  tiling_data.close();
  ASSERT_TRUE(kernel && tiling && tiling_data);
}
