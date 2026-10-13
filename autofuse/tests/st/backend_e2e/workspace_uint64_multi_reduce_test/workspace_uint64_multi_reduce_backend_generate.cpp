/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <algorithm>
#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "asc_graph_builder.h"
#include "backend_common.h"
#include "codegen.h"
#include "optimize.h"

namespace {
constexpr uint64_t kWorkspaceBytes = 0x90000000ULL;
constexpr int64_t kWorkspaceFloatCount = static_cast<int64_t>(kWorkspaceBytes / sizeof(float));
constexpr size_t kKernelFileIndex = 0;
constexpr size_t kTilingFileIndex = 1;
constexpr size_t kTilingDataFileIndex = 2;
const std::string kTilingStub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling) AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";

af::AscGraph MakeWorkspaceUint64MultiReduceGraph() {
  auto graph = af::testing::AscGraphBuilder("workspace_uint64_multi_reduce")
                   .Loops({af::testing::Sym(64), af::testing::Sym(1024)})
                   .Data("data", 0, {af::testing::Sym(64), af::testing::Sym(1024)},
                         {af::testing::Sym(1024), af::testing::Sym(1)})
                   .Load("load0", "data")
                   .Sum("sum0", "load0", {0})
                   .Load("load1", "data")
                   .Sum("sum1", "load1", {0})
                   .Load("load2", "data")
                   .Sum("sum2", "load2", {0})
                   .Add("sum01", "sum0", "sum1")
                   .Add("sum012", "sum01", "sum2")
                   .Store("store", "sum012")
                   .Output("output", "store", 0)
                   .Build();
  return graph;
}

bool CloneScheduledResult(const ascir::ScheduledResult &source, ascir::ScheduledResult &cloned) {
  cloned = source;
  for (auto &schedule_group : cloned.schedule_groups) {
    for (auto &impl_graph : schedule_group.impl_graphs) {
      af::AscGraph cloned_graph(impl_graph.GetName().c_str());
      if (!cloned_graph.CopyFrom(impl_graph)) {
        return false;
      }
      impl_graph = cloned_graph;
    }
  }
  return true;
}

void InjectLargeWorkspaces(af::AscGraph &impl_graph, std::vector<af::AscNodePtr> &workspace_nodes) {
  af::ascir_op::Scalar workspace_size_scalar("workspace_size_scalar", impl_graph);
  workspace_size_scalar.ir_attr.SetValue("0");
  auto scalar_node = impl_graph.FindNode("workspace_size_scalar");
  ASSERT_NE(scalar_node, nullptr);
  scalar_node->outputs[0].attr.dtype = af::DT_FLOAT;
  scalar_node->outputs[0].attr.repeats = {af::Symbol(kWorkspaceFloatCount)};
  scalar_node->outputs[0].attr.strides = {af::Symbol(1)};

  const std::vector<ascir::TensorId> workspace_ids = {7, 8, 9};
  for (size_t i = 0; i < workspace_ids.size(); ++i) {
    const std::string workspace_name = "workspace" + std::to_string(workspace_ids[i]);
    af::ascir_op::Workspace workspace_op(workspace_name.c_str());
    auto workspace = impl_graph.AddNode(workspace_op);
    ASSERT_NE(workspace, nullptr);
    workspace->outputs[0].attr.mem.tensor_id = workspace_ids[i];
    workspace->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
    ASSERT_EQ(af::GraphUtils::AddEdge(scalar_node->GetOutDataAnchor(0), workspace->GetInDataAnchor(0)),
              ge::GRAPH_SUCCESS);
    workspace_nodes.push_back(workspace);
  }
}

void WriteGeneratedFiles(const codegen::CodegenResult &workspace_result, const codegen::CodegenResult &reduce_result) {
  const auto parts = splitString(KERNEL_SRC_LIST, ':');
  ASSERT_EQ(parts.size(), 3U);
  std::fstream kernel_file(parts[kKernelFileIndex], std::ios::out);
  std::fstream tiling_file(parts[kTilingFileIndex], std::ios::out);
  std::fstream tiling_data_file(parts[kTilingDataFileIndex], std::ios::out);
  ASSERT_TRUE(kernel_file.is_open());
  ASSERT_TRUE(tiling_file.is_open());
  ASSERT_TRUE(tiling_data_file.is_open());
  kernel_file << kTilingStub << RemoveSubDirInclude(reduce_result.kernel);
  tiling_file << workspace_result.tiling;
  tiling_data_file << workspace_result.tiling_data;
}

}  // namespace

class TestBackendWorkspaceUint64MultiReduce : public testing::Test {};

TEST_F(TestBackendWorkspaceUint64MultiReduce, WorkspaceUint64MultiReduceCodegen) {
  auto graph = MakeWorkspaceUint64MultiReduceGraph();
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  codegen::Codegen codegen(codegen::CodegenOptions{});
  ascir::FusedScheduledResult fused_schedule_result;
  ASSERT_EQ(optimizer.Optimize(graph, fused_schedule_result), af::SUCCESS);
  auto &scheduled_results = fused_schedule_result.node_idx_to_scheduled_results[0];
  const auto selected_result =
      std::find_if(scheduled_results.begin(), scheduled_results.end(),
                   [](const ascir::ScheduledResult &result) { return result.schedule_groups.size() > 1U; });
  ASSERT_NE(selected_result, scheduled_results.end());
  ASSERT_FALSE(selected_result->schedule_groups.empty());
  ASSERT_FALSE(selected_result->schedule_groups.front().impl_graphs.empty());
  const auto original_scheduled_results = scheduled_results;
  const auto original_workspace_nodes = fused_schedule_result.workspace_nodes;
  ascir::ScheduledResult workspace_scheduled_result;
  ASSERT_TRUE(CloneScheduledResult(*selected_result, workspace_scheduled_result));
  auto &impl_graph = workspace_scheduled_result.schedule_groups.front().impl_graphs.front();
  fused_schedule_result.workspace_nodes = original_workspace_nodes;
  InjectLargeWorkspaces(impl_graph, fused_schedule_result.workspace_nodes);
  scheduled_results = {workspace_scheduled_result};
  ASSERT_EQ(fused_schedule_result.workspace_nodes.size(), original_workspace_nodes.size() + 3U);

  codegen::CodegenResult workspace_result;
  ASSERT_EQ(codegen.Generate({}, fused_schedule_result, workspace_result), af::SUCCESS);
  EXPECT_NE(workspace_result.tiling_data.find("TILING_DATA_FIELD_DEF_T(uint64_t, workspace7);"), std::string::npos);
  EXPECT_NE(workspace_result.tiling_data.find("TILING_DATA_FIELD_DEF_T(uint64_t, workspace8);"), std::string::npos);
  EXPECT_NE(workspace_result.tiling_data.find("TILING_DATA_FIELD_DEF_T(uint64_t, workspace9);"), std::string::npos);
  EXPECT_NE(workspace_result.tiling.find("uint64_t* workspaceSize"), std::string::npos);
  EXPECT_NE(workspace_result.tiling.find("bool GetWorkspaceSize("), std::string::npos);
  EXPECT_NE(workspace_result.tiling.find("__builtin_add_overflow"), std::string::npos);

  scheduled_results = original_scheduled_results;
  fused_schedule_result.workspace_nodes = original_workspace_nodes;
  codegen::CodegenResult reduce_result;
  ASSERT_EQ(codegen.Generate({}, fused_schedule_result, reduce_result), af::SUCCESS);
  WriteGeneratedFiles(workspace_result, reduce_result);
}
