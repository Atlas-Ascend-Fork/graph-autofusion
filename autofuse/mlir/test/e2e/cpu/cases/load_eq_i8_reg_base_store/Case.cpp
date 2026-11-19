/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// dtype 第二波：i8 的比较。比较族 reg 行的集合常量整批收进三个新位，这一条钉
// `CompareExtend<int8_t, ..., CMPMODE::EQ>` 真的生成且数值正确。输入含 INT8_MIN
// 与成对相等/不等的模式；结果恒 ui8 的规则与 dtype 无关，这里一并钉住。

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_eq_i8_reg_base_store/LoadEqI8RegBaseStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_eq_i8_reg_base_store;

  const auto output_shape = config::GetShapeConfig(config::kOutputShapes, 0).shape;
  const auto input0_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 0).shape);
  const auto input1_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 1).shape);
  AscGraphBuilder builder(config::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_INT8)
      .Data("input1", 1, input1_shape, {}, af::DT_INT8)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {});

  auto graph =
      builder.Eq("eq", "load0", "load1").Store("store", "eq").Output("output", "store", 0, af::DT_UINT8).Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_eq_i8_reg_base_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // int8 模板实参与 EQ 模式参数双双出现才说明选中的是 reg 侧包装的那条路。
  EXPECT_NE(kernel.find("CompareExtend<int8_t"), std::string::npos);
  EXPECT_NE(kernel.find("CMPMODE::EQ"), std::string::npos);
}
}  // namespace mlir::afir
