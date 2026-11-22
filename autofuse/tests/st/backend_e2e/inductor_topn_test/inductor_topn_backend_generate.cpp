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
#include <exception>
#include <fstream>
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "backend_common.h"
#include "codegen.h"
#include "optimize.h"
#include "share_graph.h"

namespace {
constexpr size_t kTilingDataFileIndex = 0;
constexpr size_t kHostFileIndex = 1;
constexpr size_t kDeviceFileIndex = 2;
constexpr size_t kExpectedKernelSrcCount = 3;

class TestBackendInductorTopnE2e : public testing::Test {};

TEST_F(TestBackendInductorTopnE2e, InductorTopnE2eCodegen) {
  auto graph = ascir::ShareGraph::BrcInlineFusedGraph(2);
  auto parts = splitString(KERNEL_SRC_LIST, ':');
  ASSERT_EQ(parts.size(), kExpectedKernelSrcCount);

  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    codegen::Codegen codegen(codegen::CodegenOptions{});
    ascir::FusedScheduledResult fused_schedule_result;
    std::vector<::ascir::ScheduledResult> schedule_results;
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    ASSERT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);

    codegen::CodegenResult result;
    ASSERT_EQ(codegen.GenerateForInductor(fused_schedule_result, result), 0);

    EXPECT_FALSE(result.tiling_data.empty());
    EXPECT_FALSE(result.tiling.empty());
    EXPECT_FALSE(result.kernel.empty());
    EXPECT_NE(result.tiling.find("extern \"C\" int64_t GenerateTopnSolutions("), std::string::npos);
    EXPECT_NE(result.tiling.find("std::string GetTilingDataRepr("), std::string::npos);
    EXPECT_NE(result.kernel.find("AutofuseLaunch"), std::string::npos);

    std::fstream tiling_data_file(parts[kTilingDataFileIndex], std::ios::out);
    std::fstream host_file(parts[kHostFileIndex], std::ios::out);
    std::fstream device_file(parts[kDeviceFileIndex], std::ios::out);
    ASSERT_TRUE(tiling_data_file.is_open());
    ASSERT_TRUE(host_file.is_open());
    ASSERT_TRUE(device_file.is_open());

    tiling_data_file << result.tiling_data;
    host_file << result.tiling;
    device_file << result.kernel;
  } catch (const std::exception &e) {
    FAIL() << e.what();
  } catch (...) {
    FAIL() << "inductor topn codegen failed";
  }
}

TEST_F(TestBackendInductorTopnE2e, EmptyTensorInductorTilingShouldSkipPerfEvaluation) {
  auto graph = ascir::ShareGraph::BrcInlineFusedGraph(2);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.node_idx_to_scheduled_results.emplace_back();
  ASSERT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);

  auto &impl_graph = fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  const auto store = impl_graph.FindNode("store");
  ASSERT_NE(store, nullptr);
  ASSERT_FALSE(store->outputs[0].attr.axis.empty());
  const auto owner_graph = store->GetOwnerComputeGraph();
  ASSERT_NE(owner_graph, nullptr);
  const auto graph_attr = owner_graph->GetAttrsGroup<af::AscGraphAttr>();
  ASSERT_NE(graph_attr, nullptr);
  graph_attr->axis[store->outputs[0].attr.axis[0]]->size = af::Symbol(0);

  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen::CodegenResult result;
  ASSERT_EQ(codegen.GenerateForInductor(fused_schedule_result, result), 0);
  const auto &tiling = result.tiling;
  EXPECT_NE(tiling.find("return DBL_MAX;"), std::string::npos);
  EXPECT_EQ(tiling.find("optiling::GetPerf"), std::string::npos);

  std::fstream empty_tiling_data_file(EMPTY_TILING_DEF_FILE, std::ios::out);
  std::fstream empty_host_file(EMPTY_HOST_CODE_FILE, std::ios::out);
  ASSERT_TRUE(empty_tiling_data_file.is_open());
  ASSERT_TRUE(empty_host_file.is_open());
  empty_tiling_data_file << result.tiling_data;
  empty_host_file << result.tiling;
}
}  // namespace
