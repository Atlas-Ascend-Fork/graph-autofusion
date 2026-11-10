/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 验证 load -> broadcast -> abs -> store，覆盖广播结果继续参与计算的路径。
// 相比广播后直接 Store，此图还检查 scratch 复用、消费者 descriptor 一致性及 PIPE_V 同步。
// 当 [8,1] 广播到 [8,128] 时，B0Y1 模板可能将 Abs 调度到 Broadcast 前，
// 此时 Abs 的向量化 extent 全为 1，需要 materializeVectorizedSegments 的单元素窗口。
// 测试须确认 MLIR 接管，不能将回落 legacy 后数值正确当作该路径通过。

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_broadcast_abs_store/LoadBroadcastAbsStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_broadcast_abs_store;
  const BroadcastBinding binding = BindBroadcast(config::GetShapeConfig(config::kOutputShapes, 0).shape,
                                                 config::GetShapeConfig(config::kInputShapes, 0).shape);
  AscGraphBuilder builder(config::kKernelName);
  builder.Loops(binding.outputShape)
      .Data("input0", 0, binding.inputShape, {}, af::DT_FLOAT)
      .Load("load0", "input0", binding.inputShape, binding.loadStrides);
  return builder.Broadcast("brc", "load0", binding.brcAxes)
      .Abs("abs", "brc")
      .Store("store", "abs")
      .Output("output", "store", 0, af::DT_FLOAT)
      .Build();
}

void CheckGeneratedKernel(const std::string &kernel) {
  namespace config = load_broadcast_abs_store;
  // 走的是 MLIR 那条路而不是回落 legacy。少了这一条，整个用例在回落之后照样全绿。
  EXPECT_NE(kernel.find(std::string(config::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 两个算子都在，且次序是广播在前。钉次序而不只是「都在」：反了的话数值仍然对
  // （`|brc(x)| == brc(|x|)`），只有次序这一条分得开。
  const size_t broadcast = kernel.find("Broadcast(");
  ASSERT_NE(broadcast, std::string::npos);
  EXPECT_NE(kernel.find("Abs", broadcast), std::string::npos);
}
}  // namespace mlir::afir
