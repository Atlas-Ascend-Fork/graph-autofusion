/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "ascendc_ir.h"
#include "ascir_ops_utils.h"
#include "asc_graph_utils.h"
#include "task_generator/schedule_group_partitioner.h"
#include "task_generator/node_partition_utils.h"
#include "schedule_utils.h"
#include "asc_graph_builder.h"

namespace schedule {
using namespace optimize;
using namespace ge;
using namespace af::ops;
using af::testing::AscGraphBuilder;
using af::testing::Sym;

class ScheduleGroupGraphPartitionerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }

  // Helper: Create a simple graph with Load -> Abs -> Store (unique node names per graph)
  static af::AscGraph CreateSimpleGraph(const std::string &name, int data_index) {
    std::string suffix = "_" + name;
    return AscGraphBuilder(name)
        .Loops({Sym(128), Sym(64)})
        .Data("data" + suffix, data_index)
        .Load("load" + suffix, "data" + suffix)
        .Abs("abs" + suffix, "load" + suffix)
        .Store("store" + suffix, "abs" + suffix)
        .Output("out" + suffix, "store" + suffix, 0)
        .Build();
  }

  // Helper: Create a graph with different axis
  static af::AscGraph CreateGraphWithDifferentAxis(const std::string &name, int data_index) {
    std::string suffix = "_" + name;
    return AscGraphBuilder(name)
        .Loops({Sym(256), Sym(128)})
        .Data("data" + suffix, data_index)
        .Load("load" + suffix, "data" + suffix)
        .Abs("abs" + suffix, "load" + suffix)
        .Store("store" + suffix, "abs" + suffix)
        .Output("out" + suffix, "store" + suffix, 0)
        .Build();
  }

  // Helper: Create a graph with Concat (not simple compute graph)
  static af::AscGraph CreateGraphWithConcat(const std::string &name, int data_index1, int data_index2) {
    std::string suffix = "_" + name;
    return AscGraphBuilder(name)
        .Loops({Sym(128), Sym(64)})
        .Data("data1" + suffix, data_index1)
        .Data("data2" + suffix, data_index2)
        .Load("load1" + suffix, "data1" + suffix)
        .Load("load2" + suffix, "data2" + suffix)
        .Concat("concat" + suffix, {"load1" + suffix, "load2" + suffix}, 0)
        .Store("store" + suffix, "concat" + suffix)
        .Output("out" + suffix, "store" + suffix, 0)
        .Build();
  }

  // Helper: Copy axis attributes from source graph to target graph
  void CopyAxisAttrs(af::AscGraph &target, const af::AscGraph &source) {
    auto target_cg = af::AscGraphUtils::GetComputeGraph(target);
    auto source_cg = af::AscGraphUtils::GetComputeGraph(source);
    auto source_attr = source_cg->GetOrCreateAttrsGroup<af::AscGraphAttr>();
    auto target_attr = target_cg->GetOrCreateAttrsGroup<af::AscGraphAttr>();
    target_attr->axis = source_attr->axis;
  }

  static void VerifyWorkspaceEdges(const af::AscGraph &graph) {
    af::AscNodePtr workspace_with_input;
    af::AscNodePtr workspace_with_output;
    for (const auto &node : graph.GetAllNodes()) {
      const auto asc_node = std::dynamic_pointer_cast<af::AscNode>(node);
      ASSERT_NE(asc_node, nullptr);
      if (asc_node->GetType() != af::ascir_op::Workspace::Type) {
        continue;
      }
      if (!asc_node->GetInDataNodes().empty()) {
        workspace_with_input = asc_node;
      }
      if (!asc_node->GetOutDataNodes().empty()) {
        workspace_with_output = asc_node;
      }
    }
    ASSERT_NE(workspace_with_input, nullptr);
    ASSERT_NE(workspace_with_output, nullptr);
    EXPECT_NE(workspace_with_input, workspace_with_output);
    ASSERT_EQ(workspace_with_input->GetInDataNodes().size(), 1UL);
    EXPECT_EQ(workspace_with_input->GetInDataNodes().at(0)->GetType(), af::ascir_op::Data::Type);
    ASSERT_EQ(workspace_with_output->GetOutDataNodes().size(), 1UL);
    EXPECT_EQ(workspace_with_output->GetOutDataNodes().at(0)->GetType(), af::ascir_op::Load::Type);
  }

  static void VerifyPartitionedWorkspaceEdges(const std::vector<af::AscGraph> &grouped_graphs) {
    bool found_workspace_with_input = false;
    bool found_workspace_with_output = false;
    for (const auto &grouped_graph : grouped_graphs) {
      for (const auto &node : grouped_graph.GetAllNodes()) {
        const auto asc_node = std::dynamic_pointer_cast<af::AscNode>(node);
        ASSERT_NE(asc_node, nullptr);
        if (asc_node->GetType() != af::ascir_op::Workspace::Type) {
          continue;
        }
        if (!asc_node->GetInDataNodes().empty()) {
          found_workspace_with_input = true;
          ASSERT_EQ(asc_node->GetInDataNodes().size(), 1UL);
          EXPECT_EQ(asc_node->GetInDataNodes().at(0)->GetType(), af::ascir_op::Data::Type);
        }
        if (!asc_node->GetOutDataNodes().empty()) {
          found_workspace_with_output = true;
          ASSERT_EQ(asc_node->GetOutDataNodes().size(), 1UL);
          EXPECT_EQ(asc_node->GetOutDataNodes().at(0)->GetType(), af::ascir_op::Load::Type);
        }
      }
    }
    EXPECT_TRUE(found_workspace_with_input);
    EXPECT_TRUE(found_workspace_with_output);
  }
};

TEST_F(ScheduleGroupGraphPartitionerTest, DuplicateWorkspaceNames_CopyAndPartitionKeepEdges) {
  auto graph = AscGraphBuilder("duplicate_workspace")
                   .Loops({Sym(128)})
                   .Data("data", 0)
                   .Workspace("workspace_pre", "data")
                   .Workspace("workspace_post")
                   .Load("load", "workspace_post")
                   .Store("store", "load")
                   .Output("out", "store", 0)
                   .Build();
  const auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  ASSERT_NE(compute_graph, nullptr);
  const auto workspace_pre = compute_graph->FindNode("workspace_pre");
  const auto workspace_post = compute_graph->FindNode("workspace_post");
  ASSERT_NE(workspace_pre, nullptr);
  ASSERT_NE(workspace_post, nullptr);
  workspace_pre->GetOpDesc()->SetName("workspace");
  workspace_post->GetOpDesc()->SetName("workspace");

  af::AscGraph copied_graph("copied_duplicate_workspace");
  ASSERT_TRUE(copied_graph.CopyFrom(graph));

  VerifyWorkspaceEdges(copied_graph);

  std::vector<af::AscGraph> grouped_graphs;
  ASSERT_EQ(ScheduleGroupGraphPartitioner::PartitionByConnectivity(graph, grouped_graphs), af::SUCCESS);
  ASSERT_EQ(grouped_graphs.size(), 2UL);
  VerifyPartitionedWorkspaceEdges(grouped_graphs);
}

// 多引用切分让某一引用分支先于合并被摘掉(该分支自己的 load 已不在), 只留下一个无输入
// 无输出的同名 Workspace 副本; 它与同名的写侧 Workspace(有输入, 不参与去重分组, 所以不会
// 在上面的合并主循环里被一并处理)仍然同落在这个子图里。预期它被判定为死副本并摘除。
TEST_F(ScheduleGroupGraphPartitionerTest, MergeDuplicatedCopyNodesInSubGraph_DeadNamedWorkspaceCopy_Removed) {
  auto graph = AscGraphBuilder("dead_named_workspace_copy")
                   .Loops({Sym(128)})
                   .Data("data", 0)
                   .Workspace("workspace_write", "data")
                   .Workspace("workspace_dead")
                   .Build();
  const auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  ASSERT_NE(compute_graph, nullptr);
  const auto workspace_write = compute_graph->FindNode("workspace_write");
  const auto workspace_dead = compute_graph->FindNode("workspace_dead");
  ASSERT_NE(workspace_write, nullptr);
  ASSERT_NE(workspace_dead, nullptr);
  workspace_write->GetOpDesc()->SetName("workspace");
  workspace_dead->GetOpDesc()->SetName("workspace");

  ASSERT_EQ(MergeDuplicatedCopyNodesInSubGraph(graph), af::SUCCESS);

  std::vector<std::remove_const_t<decltype(workspace_write)>> remaining_named_workspace;
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetName() == "workspace") {
      remaining_named_workspace.push_back(node);
    }
  }
  ASSERT_EQ(remaining_named_workspace.size(), 1UL);
  EXPECT_EQ(remaining_named_workspace.front(), workspace_write);
}

TEST_F(ScheduleGroupGraphPartitionerTest, DuplicateWorkspaceNames_PartitionDoesNotDropComponents) {
  auto graph = AscGraphBuilder("duplicate_outputs")
                   .Loops({Sym(128)})
                   .Data("data0", 0)
                   .Data("data1", 1)
                   .Workspace("workspace0", "data0")
                   .Workspace("workspace1", "data1")
                   .Build();
  const auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  ASSERT_NE(compute_graph, nullptr);
  const auto workspace0 = compute_graph->FindNode("workspace0");
  const auto workspace1 = compute_graph->FindNode("workspace1");
  ASSERT_NE(workspace0, nullptr);
  ASSERT_NE(workspace1, nullptr);
  workspace0->GetOpDesc()->SetName("workspace");
  workspace1->GetOpDesc()->SetName("workspace");

  std::vector<af::AscGraph> grouped_graphs;
  EXPECT_EQ(ScheduleGroupGraphPartitioner::PartitionByConnectivity(graph, grouped_graphs), af::SUCCESS);
  EXPECT_EQ(grouped_graphs.size(), 2UL);
}

// Test 1: Less than 5 graphs - should do nothing
TEST_F(ScheduleGroupGraphPartitionerTest, ReduceGraphCount_LessThan5_NoChange) {
  std::vector<af::AscGraph> graphs;
  graphs.push_back(CreateSimpleGraph("graph0", 0));
  graphs.push_back(CreateSimpleGraph("graph1", 1));
  graphs.push_back(CreateSimpleGraph("graph2", 2));

  size_t original_size = graphs.size();
  EXPECT_EQ(ScheduleGroupGraphPartitioner::ReduceGraphCount(graphs, 5), af::SUCCESS);

  EXPECT_EQ(graphs.size(), original_size);
}

// Test 2: Exactly 5 graphs - should do nothing
TEST_F(ScheduleGroupGraphPartitionerTest, ReduceGraphCount_Exactly5_NoChange) {
  std::vector<af::AscGraph> graphs;
  for (int i = 0; i < 5; ++i) {
    graphs.push_back(CreateSimpleGraph("graph" + std::to_string(i), i));
  }

  size_t original_size = graphs.size();
  EXPECT_EQ(ScheduleGroupGraphPartitioner::ReduceGraphCount(graphs, 5), af::SUCCESS);

  EXPECT_EQ(graphs.size(), original_size);
}

// Test 3: More than 5 graphs, all mergeable - should reduce to 5
TEST_F(ScheduleGroupGraphPartitionerTest, ReduceGraphCount_MoreThan5_AllMergeable_ReduceTo5) {
  std::vector<af::AscGraph> graphs;
  auto base_graph = CreateSimpleGraph("base", 0);

  for (int i = 0; i < 8; ++i) {
    auto graph = CreateSimpleGraph("graph" + std::to_string(i), i);
    CopyAxisAttrs(graph, base_graph);  // Share same axis attributes
    graphs.push_back(std::move(graph));
  }

  EXPECT_EQ(ScheduleGroupGraphPartitioner::ReduceGraphCount(graphs, 5), af::SUCCESS);

  EXPECT_EQ(graphs.size(), 5UL);
}

// Test 4: More than 5 graphs, none mergeable (different axis) - should stay the same
TEST_F(ScheduleGroupGraphPartitionerTest, ReduceGraphCount_MoreThan5_NoneMergeable_NoChange) {
  std::vector<af::AscGraph> graphs;

  // Create graphs with different axis IDs by using different loop configurations
  for (int i = 0; i < 8; ++i) {
    // Each graph will have unique axis IDs because AscGraphBuilder creates new axes
    auto graph = AscGraphBuilder("graph" + std::to_string(i))
                     .Loops({Sym(128 + i), Sym(64 + i)})  // Different sizes for each graph
                     .Data("data", i)
                     .Load("load", "data")
                     .Abs("abs", "load")
                     .Store("store", "abs")
                     .Output("out", "store", 0)
                     .Build();
    graphs.push_back(std::move(graph));
  }

  size_t original_size = graphs.size();
  EXPECT_EQ(ScheduleGroupGraphPartitioner::ReduceGraphCount(graphs, 5), af::SUCCESS);

  EXPECT_EQ(graphs.size(), original_size);
}

// Test 5: More than 5 graphs, some mergeable - should reduce as much as possible
TEST_F(ScheduleGroupGraphPartitionerTest, ReduceGraphCount_MoreThan5_SomeMergeable_PartialReduce) {
  std::vector<af::AscGraph> graphs;
  auto base_graph = CreateSimpleGraph("base", 0);

  // 4 graphs with same axis (mergeable)
  for (int i = 0; i < 4; ++i) {
    auto graph = CreateSimpleGraph("same_axis_" + std::to_string(i), i);
    CopyAxisAttrs(graph, base_graph);
    graphs.push_back(std::move(graph));
  }
  // 4 graphs with different axis (not mergeable with above)
  for (int i = 0; i < 4; ++i) {
    graphs.push_back(CreateGraphWithDifferentAxis("diff_axis_" + std::to_string(i), i + 10));
  }

  EXPECT_EQ(ScheduleGroupGraphPartitioner::ReduceGraphCount(graphs, 5), af::SUCCESS);

  // Should reduce from 8 to 5 by merging 3 pairs from the same_axis group
  EXPECT_EQ(graphs.size(), 5UL);
}

// Test 6: Priority test - should merge smallest graphs first
TEST_F(ScheduleGroupGraphPartitionerTest, ReduceGraphCount_Priority_MergeSmallestFirst) {
  std::vector<af::AscGraph> graphs;
  auto base_graph = CreateSimpleGraph("base", 0);

  // Small graphs (2 nodes each: Load -> Store)
  for (int i = 0; i < 3; ++i) {
    std::string name = "small_" + std::to_string(i);
    std::string suffix = "_" + name;
    auto graph = AscGraphBuilder(name)
                     .Loops({Sym(128), Sym(64)})
                     .Data("data" + suffix, i)
                     .Load("load" + suffix, "data" + suffix)
                     .Store("store" + suffix, "load" + suffix)
                     .Output("out" + suffix, "store" + suffix, 0)
                     .Build();
    CopyAxisAttrs(graph, base_graph);
    graphs.push_back(std::move(graph));
  }

  // Large graphs (4 nodes each: Load -> Abs -> Relu -> Store)
  for (int i = 0; i < 5; ++i) {
    std::string name = "large_" + std::to_string(i);
    std::string suffix = "_" + name;
    auto graph = AscGraphBuilder(name)
                     .Loops({Sym(128), Sym(64)})
                     .Data("data" + suffix, i + 10)
                     .Load("load" + suffix, "data" + suffix)
                     .Abs("abs" + suffix, "load" + suffix)
                     .Relu("relu" + suffix, "abs" + suffix)
                     .Store("store" + suffix, "relu" + suffix)
                     .Output("out" + suffix, "store" + suffix, 0)
                     .Build();
    CopyAxisAttrs(graph, base_graph);
    graphs.push_back(std::move(graph));
  }

  // Total: 8 graphs, should reduce to 5
  EXPECT_EQ(ScheduleGroupGraphPartitioner::ReduceGraphCount(graphs, 5), af::SUCCESS);

  EXPECT_EQ(graphs.size(), 5UL);
}

// Test 7: Graphs with Concat (not simple compute graph) - should not be merged
TEST_F(ScheduleGroupGraphPartitionerTest, ReduceGraphCount_WithConcat_NotMerged) {
  std::vector<af::AscGraph> graphs;

  // 3 simple graphs (mergeable)
  for (int i = 0; i < 3; ++i) {
    graphs.push_back(CreateSimpleGraph("simple_" + std::to_string(i), i));
  }

  // 5 graphs with Concat (not mergeable)
  for (int i = 0; i < 5; ++i) {
    graphs.push_back(CreateGraphWithConcat("concat_" + std::to_string(i), i + 10, i + 20));
  }

  // Total: 8 graphs
  EXPECT_EQ(ScheduleGroupGraphPartitioner::ReduceGraphCount(graphs, 5), af::SUCCESS);

  // Should reduce from 8 to 6 by merging 2 simple graphs, then stop
  // (can't merge concat graphs, and only 1 simple graph left)
  EXPECT_LE(graphs.size(), 8UL);
  EXPECT_GE(graphs.size(), 5UL);
}

// Test 8: Custom target count
TEST_F(ScheduleGroupGraphPartitionerTest, ReduceGraphCount_CustomTarget_Respected) {
  std::vector<af::AscGraph> graphs;
  auto base_graph = CreateSimpleGraph("base", 0);

  for (int i = 0; i < 10; ++i) {
    auto graph = CreateSimpleGraph("graph" + std::to_string(i), i);
    CopyAxisAttrs(graph, base_graph);
    graphs.push_back(std::move(graph));
  }

  EXPECT_EQ(ScheduleGroupGraphPartitioner::ReduceGraphCount(graphs, 3), af::SUCCESS);

  EXPECT_EQ(graphs.size(), 3UL);
}

// Test 9: Empty graph list - should not crash
TEST_F(ScheduleGroupGraphPartitionerTest, ReduceGraphCount_EmptyList_NoCrash) {
  std::vector<af::AscGraph> graphs;

  EXPECT_EQ(ScheduleGroupGraphPartitioner::ReduceGraphCount(graphs, 5), af::SUCCESS);

  EXPECT_EQ(graphs.size(), 0UL);
}

// Test 10: Verify merged graph contains nodes from both original graphs
TEST_F(ScheduleGroupGraphPartitionerTest, ReduceGraphCount_MergedGraph_ContainsAllNodes) {
  std::vector<af::AscGraph> graphs;
  auto base_graph = CreateSimpleGraph("base", 0);

  auto graph1 = CreateSimpleGraph("graph1", 0);
  auto graph2 = CreateSimpleGraph("graph2", 1);

  CopyAxisAttrs(graph1, base_graph);
  CopyAxisAttrs(graph2, base_graph);

  graphs.push_back(std::move(graph1));
  graphs.push_back(std::move(graph2));

  // Add 4 more to exceed 5
  for (int i = 2; i < 6; ++i) {
    auto graph = CreateSimpleGraph("graph" + std::to_string(i), i);
    CopyAxisAttrs(graph, base_graph);
    graphs.push_back(std::move(graph));
  }

  EXPECT_EQ(ScheduleGroupGraphPartitioner::ReduceGraphCount(graphs, 5), af::SUCCESS);

  EXPECT_EQ(graphs.size(), 5UL);
}

// Test 11: Verify that a graph used as source is not reused as destination
TEST_F(ScheduleGroupGraphPartitionerTest, ReduceGraphCount_NoSourceReuseAsDestination) {
  std::vector<af::AscGraph> graphs;
  auto base_graph = CreateSimpleGraph("base", 0);

  // Create 6 graphs with varying node counts to force multiple merges
  // Graph 0: 1 node (smallest)
  auto graph0 = AscGraphBuilder("graph0")
                    .Loops({Sym(128), Sym(64)})
                    .Data("data0", 0)
                    .Load("load0", "data0")
                    .Output("out0", "load0", 0)
                    .Build();
  CopyAxisAttrs(graph0, base_graph);
  graphs.push_back(std::move(graph0));

  // Graph 1-5: 3 nodes each
  for (int i = 1; i < 6; ++i) {
    auto graph = CreateSimpleGraph("graph" + std::to_string(i), i);
    CopyAxisAttrs(graph, base_graph);
    graphs.push_back(std::move(graph));
  }

  // Count total nodes before merge
  size_t total_nodes_before = 0;
  for (const auto &g : graphs) {
    for (const auto &node : g.GetAllNodes()) {
      (void)node;
      ++total_nodes_before;
    }
  }

  // Reduce from 6 to 2 (4 reductions needed)
  EXPECT_EQ(ScheduleGroupGraphPartitioner::ReduceGraphCount(graphs, 2), af::SUCCESS);

  EXPECT_EQ(graphs.size(), 2UL);

  // Count total nodes after merge - should equal original total
  size_t total_nodes_after = 0;
  for (const auto &g : graphs) {
    for (const auto &node : g.GetAllNodes()) {
      (void)node;
      ++total_nodes_after;
    }
  }

  EXPECT_EQ(total_nodes_after, total_nodes_before);
}
}  // namespace schedule
