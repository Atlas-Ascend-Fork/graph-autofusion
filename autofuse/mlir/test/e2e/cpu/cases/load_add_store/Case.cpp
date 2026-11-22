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

#include "test/e2e/cpu/cases/load_add_store/LoadAddStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_add_store::GetShapeConfig(load_add_store::kOutputShapes, 0).shape;
  const auto input0_shape = ToExpressions(load_add_store::GetShapeConfig(load_add_store::kInputShapes, 0).shape);
  const auto input1_shape = ToExpressions(load_add_store::GetShapeConfig(load_add_store::kInputShapes, 1).shape);
  AscGraphBuilder builder(load_add_store::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_FLOAT)
      .Data("input1", 1, input1_shape, {}, af::DT_FLOAT)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {});

  auto graph =
      builder.Add("add", "load0", "load1").Store("store", "add").Output("output", "store", 0, af::DT_FLOAT).Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_add_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::TPosition::VECIN"), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::TPosition::VECOUT"), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::Add"), std::string::npos);
  EXPECT_NE(kernel.find(".EnQue("), std::string::npos);
  EXPECT_NE(kernel.find(".DeQue<"), std::string::npos);
}
}  // namespace mlir::afir
