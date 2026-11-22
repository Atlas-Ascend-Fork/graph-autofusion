/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/workspace_readback/WorkspaceReadbackCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = workspace_readback::GetShapeConfig(workspace_readback::kOutputShapes, 0).shape;
  const auto input0_shape =
      ToExpressions(workspace_readback::GetShapeConfig(workspace_readback::kInputShapes, 0).shape);
  AscGraphBuilder builder(workspace_readback::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input0_shape, {}, af::DT_FLOAT);
  builder.Load("load0", "input0", input0_shape, {});
  builder.Add("add0", "load0", "load0");
  // 先将结果 Store 到 kernel 的 GM 输出参数，再读回同一存储继续计算。
  // 使用 role=output 的 memref，因为 buffer-form bridge 尚不支持专用 Workspace 节点。
  builder.Store("store0", "add0");
  builder.Output("output", "store0", 0, af::DT_FLOAT);
  builder.Load("load1", "output", input0_shape, {});
  builder.Add("add1", "load1", "load1");
  builder.Store("store1", "add1");
  auto graph = builder.Output("output2", "store1", 1, af::DT_FLOAT).Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(workspace_readback::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 回读边是 MTE3→MTE2 的显式事件对：record 侧占槽（AllocEventID）、wait 侧等待并归还
  // （WaitFlag + ReleaseEventID）。CPU debug 下 MTE3_MTE2 的 WaitFlag 非 no-op，时序
  // 在本用例即被真实执行。
  EXPECT_NE(kernel.find("AllocEventID<AscendC::HardEvent::MTE3_MTE2>"), std::string::npos);
  EXPECT_NE(kernel.find("WaitFlag<AscendC::HardEvent::MTE3_MTE2>"), std::string::npos);
  EXPECT_NE(kernel.find("ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>"), std::string::npos);
}
}  // namespace mlir::afir
