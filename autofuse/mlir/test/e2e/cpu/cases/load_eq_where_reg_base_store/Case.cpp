/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> eq -> where -> store CPU case (reg-base) -*- C++ -*-===//
//
// **M3 的那一格**：谓词寄存器作为值。`eq` 在寄存器形态下产出的不是一段 `ui8` 的值，
// 而是一条 `MaskReg`——一位一个 lane；`where` 直接拿它当 `Reg::Select` 的 mask。中间那条
// 谓词一次都不落 UB，也没有被物化成 0/1 的向量。
//
// 这张图与 reg 侧接管表上的 `load_compare_where_store_test` 同形（四路取数、`eq` 比前两
// 路、`where` 在后两路之间选），那一格此前报 `'afir.eq' has no register-form lowering`
// 整图回落。
//
// **实现前它是红的**：`eq` 不在算子表里，VF 的能力层报能力不足，而 CPU E2E 的生成器直接
// 调 `RunMlirCodegen`、不接 legacy 回落，所以红在生成那一步。
//
// 数值取法：`input0` / `input1` 按下标构造成**三分之一左右相等**，这样 `where` 两路都会
// 被取到——条件恒真或恒假的那种取法会让「两路接反」看不出来。两路候选值本身也必须处处
// 不等（见 KernelTest.cpp）。
//
// 设计见 docs/development/2026-10-18-afir-vector-func-implementation-design.md 的 M3。
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_eq_where_reg_base_store/Case.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_eq_where_reg_base_store;

  const auto output_shape = config::GetShapeConfig(config::kOutputShapes, 0).shape;
  const auto input0_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 0).shape);
  const auto input1_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 1).shape);
  const auto input2_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 2).shape);
  const auto input3_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 3).shape);
  AscGraphBuilder builder(config::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_FLOAT)
      .Data("input1", 1, input1_shape, {}, af::DT_FLOAT)
      .Data("input2", 2, input2_shape, {}, af::DT_FLOAT)
      .Data("input3", 3, input3_shape, {}, af::DT_FLOAT)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {})
      .Load("load2", "input2", input2_shape, {})
      .Load("load3", "input3", input3_shape, {});

  auto graph = builder.Eq("eq", "load0", "load1")
                   .Where("where", "eq", "load2", "load3")
                   .Store("store", "where")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_eq_where_reg_base_store::kKernelName) + "_subgraph0_plan0"),
            std::string::npos);
  // 产物必须是**寄存器形态**：一个向量作用域、一层 lane 循环，中间结果不落 UB。
  EXPECT_NE(kernel.find("__VEC_SCOPE__"), std::string::npos);
  // 四路各取一次、结果存一次。**计数式而不是「存在一条」**：少发一条 Load 也能编过、
  // 也能算对（上一个 lane 的寄存器还在），错的是别的形状才会暴露。
  EXPECT_EQ(CountOccurrences(kernel, "LoadAlign"), 4U);
  EXPECT_EQ(CountOccurrences(kernel, "StoreAlign"), 1U);
  // 这一期新增的那一对，各一条。
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::Compare<"), 1U);
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::Select("), 1U);
  // 谓词是一条 `MaskReg`，不是一段值：这一格里只该多出**一个** MaskReg 声明
  // （`UpdateMask` 那条 lane 活性掩码不走声明，它是带初值的赋值）。物化回值的那条
  // `Reg::Duplicate` 一条都不该有——有就说明走了 legacy 的另一条路。
  EXPECT_EQ(kernel.find("AscendC::Reg::Duplicate"), std::string::npos);
  // 这两条是「真的走了 VF 这条路」的判据：tile 级的包装一条都不该剩。
  EXPECT_EQ(kernel.find("CompareExtend"), std::string::npos);
  EXPECT_EQ(kernel.find("WhereExtend"), std::string::npos);
}
}  // namespace mlir::afir
