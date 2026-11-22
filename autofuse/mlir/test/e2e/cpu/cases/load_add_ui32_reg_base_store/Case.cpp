/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// dtype 第二波：ui32 的 add。这一位只在 reg 侧供给，走的是拆出来的
// `add.tensor_tensor.extend_reg_base` 行——发出的仍是标准 `AscendC::Add`。
// 数据取值贴近 UINT32_MAX：回绕语义是 ui32 加法的定义行为，任何截断到窄类型的
// 实现错法都会在这里现形。

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_add_ui32_reg_base_store/LoadAddUi32RegBaseStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_add_ui32_reg_base_store;

  const auto output_shape = config::GetShapeConfig(config::kOutputShapes, 0).shape;
  const auto input0_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 0).shape);
  const auto input1_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 1).shape);
  AscGraphBuilder builder(config::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_UINT32)
      .Data("input1", 1, input1_shape, {}, af::DT_UINT32)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {});

  auto graph =
      builder.Add("add", "load0", "load1").Store("store", "add").Output("output", "store", 0, af::DT_UINT32).Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_add_ui32_reg_base_store::kKernelName) + "_subgraph0_plan0"),
            std::string::npos);
  // reg 拆行发的仍是标准调用，一条；不存在 Add 包装。
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Add("), 1U);
  EXPECT_EQ(kernel.find("AddExtend"), std::string::npos);
}
}  // namespace mlir::afir
