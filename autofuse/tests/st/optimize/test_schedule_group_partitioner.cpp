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

#include "asc_graph_builder.h"
#include "task_generator/node_partition_utils.h"
#include "asc_graph_utils.h"

namespace {
using af::AscGraph;
using af::testing::AscGraphBuilder;
using af::testing::Sym;
}  // namespace

// 多引用切分让某一引用分支先于合并被摘掉(该分支自己的 load 已不在), 只留下一个无输入
// 无输出的同名 Workspace 副本; 它与同名的写侧 Workspace(有输入, 不参与去重分组, 所以不会
// 在合并主循环里被一并处理)仍然同落在这个子图里。预期它被判定为死副本并摘除。
TEST(ScheduleGroupPartitionerSt, MergeDuplicatedCopyNodesInSubGraph_DeadNamedWorkspaceCopy_Removed) {
  auto graph = AscGraphBuilder("dead_named_workspace_copy_st")
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

  ASSERT_EQ(optimize::MergeDuplicatedCopyNodesInSubGraph(graph), af::SUCCESS);

  std::vector<std::remove_const_t<decltype(workspace_write)>> remaining_named_workspace;
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetName() == "workspace") {
      remaining_named_workspace.push_back(node);
    }
  }
  ASSERT_EQ(remaining_named_workspace.size(), 1UL);
  EXPECT_EQ(remaining_named_workspace.front(), workspace_write);
}
