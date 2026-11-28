/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// floor_div 的 i64：和 legacy 同一条实现 `floor_div.tensor.extend_reg_base`，发同一个
// `FloorDivExtend` 调用，i64 那一支是包装里的 SIMT 实现。包装对 `INT64_MIN / -1` 没有溢出
// 保护，这里与 legacy 保持一致；数值比对只钉除 0 输出 0 和向下取整，不碰那一格。

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_floor_div_i64_reg_base_store/LoadFloorDivI64RegBaseStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_floor_div_i64_reg_base_store;

  const auto output_shape = config::GetShapeConfig(config::kOutputShapes, 0).shape;
  const auto input0_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 0).shape);
  const auto input1_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 1).shape);
  AscGraphBuilder builder(config::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_INT64)
      .Data("input1", 1, input1_shape, {}, af::DT_INT64)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {});

  auto graph = builder.FloorDiv("floor_div", "load0", "load1")
                   .Store("store", "floor_div")
                   .Output("output", "store", 0, af::DT_INT64)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_floor_div_i64_reg_base_store::kKernelName) + "_subgraph0_plan0"),
            std::string::npos);
  // 包装源码内联后 `FloorDivExtend(` 同时出现在定义与调用点。
  EXPECT_EQ(CountOccurrences(kernel, "FloorDivExtend("), 2U);
}
}  // namespace mlir::afir
