/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 验证广播结果与完整形状的 Load 结果共同参与 Add，例如偏置相加。
// 两份输入的来源和局部视图不同，广播源在广播轴上的 stride 为 0，不能沿用仅有一份
// 源和一份 destination 的分段假设。
// 用例只声明所注册形状的覆盖，不用其他形状的历史探针结果推断当前支持范围。

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_broadcast_add_store/LoadBroadcastAddStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_broadcast_add_store;
  // `input0` 是广播源（带 1 的那个形状），`input1` 与输出同形。
  const BroadcastBinding binding = BindBroadcast(config::GetShapeConfig(config::kOutputShapes, 0).shape,
                                                 config::GetShapeConfig(config::kInputShapes, 0).shape);
  const auto fullShape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 1).shape);
  AscGraphBuilder builder(config::kKernelName);
  builder.Loops(binding.outputShape)
      .Data("input0", 0, binding.inputShape, {}, af::DT_FLOAT)
      .Data("input1", 1, fullShape, {}, af::DT_FLOAT)
      .Load("load0", "input0", binding.inputShape, binding.loadStrides)
      .Load("load1", "input1", fullShape, {});
  return builder.Broadcast("brc", "load0", binding.brcAxes)
      .Add("add", "load1", "brc")
      .Store("store", "add")
      .Output("output", "store", 0, af::DT_FLOAT)
      .Build();
}

void CheckGeneratedKernel(const std::string &kernel) {
  namespace config = load_broadcast_add_store;
  // 走的是 MLIR 那条路而不是回落 legacy。少了这一条，整个用例在回落之后照样全绿。
  EXPECT_NE(kernel.find(std::string(config::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 广播在加法之前。两条搬入与一条广播都在，次序由这一条钉住。
  const size_t broadcast = kernel.find("Broadcast(");
  ASSERT_NE(broadcast, std::string::npos);
  EXPECT_NE(kernel.find("AscendC::Add", broadcast), std::string::npos);
}
}  // namespace mlir::afir
