/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 覆盖以 Abs 为分叉点的菱形图，与 load_add_diamond_store 的 Add 分叉对照。
// 调度将 unary 分叉结果放在非队列计算缓冲中，可能与 tmp buffer 共用同一 b<id>。
// 两者 alignment_bytes 分别为 -1 和 32，资源合并应兼容这两种要求，而不是将其判为
// conflicting descriptors。线性链的结果直接进队列，不能覆盖这种合并。
// 分叉点只接收一个输入，因此使用三个 Load；形状采用 [128, 8]。

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_abs_diamond_store/LoadAbsDiamondStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace cfg = load_abs_diamond_store;

  const auto output_shape = cfg::GetShapeConfig(cfg::kOutputShapes, 0).shape;
  const auto input0_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 0).shape);
  const auto input1_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 1).shape);
  const auto input2_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 2).shape);
  AscGraphBuilder builder(cfg::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_FLOAT)
      .Data("input1", 1, input1_shape, {}, af::DT_FLOAT)
      .Data("input2", 2, input2_shape, {}, af::DT_FLOAT)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {})
      .Load("load2", "input2", input2_shape, {});

  // 两个消费者都是 `Add`，与 `load_add_diamond_store` 一致——那条用例改的只有分叉点。
  // 消费者是什么不起作用（T5c 的两个消费者是 unary，结论相同），所以这里不另立一格。
  auto graph = builder.Abs("fork", "load0")
                   .Add("add1", "fork", "load1")
                   .Add("add2", "fork", "load2")
                   .Add("add3", "add1", "add2")
                   .Store("store", "add3")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  // 检查 MLIR 确实接管此图。数值结果本身无法区分 MLIR 修复成功与回落 legacy。
  EXPECT_NE(kernel.find(std::string(load_abs_diamond_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  EXPECT_NE(kernel.find("AbsExtend<float>("), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::Add"), std::string::npos);
  // 分叉结果与 Abs scratch 共用 b<id>，应生成 TBuf，验证两个 descriptor 已合并为同一资源。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
}
}  // namespace mlir::afir
