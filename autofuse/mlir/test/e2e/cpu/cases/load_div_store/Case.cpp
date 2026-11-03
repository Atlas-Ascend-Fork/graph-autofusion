/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load x2 -> div -> store CPU case --------------*- C++ -*-===//
//
// `load_sub_store` 的另一半。两个 op 在 legacy 那边逐行同形，这里钉的也是同一件事
// （`TmpBufferUse::Unused`：调度分了一块 tmp buffer，而标准调用不收它），来由见那一份。
//
// 单独留一份而不是并进 `load_sub_store`，是因为**元素类型不同**：`div` 两侧都只收
// f16 / f32，`sub` 还收整数。两个 op 各自过一遍端到端，谁的注册写错了都只红自己那一份。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_div_store/Case.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_div_store::GetShapeConfig(load_div_store::kOutputShapes, 0).shape;
  const auto input0_shape = ToExpressions(load_div_store::GetShapeConfig(load_div_store::kInputShapes, 0).shape);
  const auto input1_shape = ToExpressions(load_div_store::GetShapeConfig(load_div_store::kInputShapes, 1).shape);
  AscGraphBuilder builder(load_div_store::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_FLOAT)
      .Data("input1", 1, input1_shape, {}, af::DT_FLOAT)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {});

  auto graph =
      builder.Div("div", "load0", "load1").Store("store", "div").Output("output", "store", 0, af::DT_FLOAT).Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_div_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::Div("), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
  EXPECT_EQ(CountCallArguments(kernel, "AscendC::Div("), 4U);
  EXPECT_EQ(kernel.find("__ASCENDC_REGBASE_API_DIV_H__"), std::string::npos);
}
}  // namespace mlir::afir
