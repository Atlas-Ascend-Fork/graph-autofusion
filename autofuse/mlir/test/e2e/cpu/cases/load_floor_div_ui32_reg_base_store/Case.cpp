/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// dtype 第二波：floor_div 的 ui32。这是 FloorDivExtend 分派里 `FloorDivUintImplVF`
// 那一支——无符号除法没有 INT_MIN / -1 溢出格，第二波据此放行（i8 / i64 仍钉在
// kKnownNarrower）。被除数贴 UINT32_MAX、除数遍历 1..9，把商的量级铺满。

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_floor_div_ui32_reg_base_store/LoadFloorDivUi32RegBaseStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_floor_div_ui32_reg_base_store;

  const auto output_shape = config::GetShapeConfig(config::kOutputShapes, 0).shape;
  const auto input0_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 0).shape);
  const auto input1_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 1).shape);
  AscGraphBuilder builder(config::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_UINT32)
      .Data("input1", 1, input1_shape, {}, af::DT_UINT32)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {});

  auto graph = builder.FloorDiv("floor_div", "load0", "load1")
                   .Store("store", "floor_div")
                   .Output("output", "store", 0, af::DT_UINT32)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_floor_div_ui32_reg_base_store::kKernelName) + "_subgraph0_plan0"),
            std::string::npos);
  // 包装源码内联后 `FloorDivExtend(` 同时出现在定义与调用点。
  EXPECT_EQ(CountOccurrences(kernel, "FloorDivExtend("), 2U);
}
}  // namespace mlir::afir
