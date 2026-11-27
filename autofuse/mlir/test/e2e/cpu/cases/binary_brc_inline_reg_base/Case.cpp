/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "Case.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  namespace cfg = binary_brc_inline_reg_base;
  const auto out = cfg::GetShapeConfig(cfg::kOutputShapes, 0).shape;
  const auto lhs = BindBroadcast(out, cfg::GetShapeConfig(cfg::kInputShapes, 0).shape);
  const auto rhs = BindBroadcast(out, cfg::GetShapeConfig(cfg::kInputShapes, 1).shape);
  af::testing::AscGraphBuilder builder(cfg::kKernelName);
  // Model the graph after legacy removes Broadcast: inputs keep their own
  // repeats and zero broadcast strides; the result follows the full input.
  builder.Loops(out)
      .Data("input0", 0, lhs.inputShape, {}, af::DT_FLOAT)
      .Data("input1", 1, rhs.inputShape, {}, af::DT_FLOAT)
      .Load("load0", "input0", lhs.inputShape, lhs.loadStrides)
      .Load("load1", "input1", rhs.inputShape, rhs.loadStrides);
  const size_t full =
      cfg::GetShapeConfig(cfg::kInputShapes, 0).element_count > cfg::GetShapeConfig(cfg::kInputShapes, 1).element_count
          ? 0U
          : 1U;
  auto graph = builder.Op<af::ascir_op::AF_BRC_OP>("binary", {"load0", "load1"}, full)
                   .Store("store", "binary")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(binary_brc_inline_reg_base::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  EXPECT_NE(kernel.find(AF_BRC_API), std::string::npos);
  EXPECT_EQ(kernel.find("BinaryBrcInlineApiWithTwoVectorizedAxis"), std::string::npos);
  EXPECT_EQ(kernel.find("Broadcast("), std::string::npos);
}
}  // namespace mlir::afir
