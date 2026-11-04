/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> mul -> abs -> store CPU case (reg-base) -*- C++ -*-===//
//
// **M4 的 `afir.mul` 那一行**。reg 侧接管表上 `load_brc_test` 此前报的就是
// `'afir.mul' has no register-form lowering`，整图回落。算子表加一行要带一格用例，
// 这是那一格。
//
// 图取 `mul` 在前、`abs` 在后，与 `load_abs_neg_reg_base_store` 同一个考虑：这样两个算子
// 与它们的顺序都可观测。乘积会跨过 0（一个输入按下标变号），于是
// —— 漏发 `mul` 得到 `|x|`、漏发 `abs` 得到 `x*y`、顺序写反（`abs` 在前）得到 `|x|*y`
// —— 三种都在乘积为负的元素上错。
//
// **这一行的判据只能是数值**：表里填错一个 API（比如填 `Reg::Add`）在 IR 与 C++ 文本上
// 都看不出不对。
//
// 设计见 docs/development/2026-10-18-afir-vector-func-implementation-design.md 的 M4。
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_mul_abs_reg_base_store/LoadMulAbsRegBaseStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_mul_abs_reg_base_store;

  const auto output_shape = config::GetShapeConfig(config::kOutputShapes, 0).shape;
  const auto input0_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 0).shape);
  const auto input1_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 1).shape);
  AscGraphBuilder builder(config::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_FLOAT)
      .Data("input1", 1, input1_shape, {}, af::DT_FLOAT)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {});

  auto graph = builder.Mul("mul", "load0", "load1")
                   .Abs("abs", "mul")
                   .Store("store", "abs")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_mul_abs_reg_base_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 产物必须是**寄存器形态**：一个向量作用域、一层 lane 循环，中间结果不落 UB。
  EXPECT_NE(kernel.find("__VEC_SCOPE__"), std::string::npos);
  // 两个输入各取一次、结果存一次。**计数式而不是「存在一条」**：少发一条 Load 也能编过、
  // 也能算对（上一个 lane 的寄存器还在）。
  EXPECT_EQ(CountOccurrences(kernel, "LoadAlign"), 2U);
  EXPECT_EQ(CountOccurrences(kernel, "StoreAlign"), 1U);
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::Mul("), 1U);
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::Abs("), 1U);
  // 这两条是「真的走了 VF 这条路」的判据：tile 级的包装一条都不该剩。
  EXPECT_EQ(kernel.find("MulExtend"), std::string::npos);
  EXPECT_EQ(kernel.find("AbsExtend"), std::string::npos);
}
}  // namespace mlir::afir
