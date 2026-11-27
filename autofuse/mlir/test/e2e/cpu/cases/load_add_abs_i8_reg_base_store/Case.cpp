/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load x2 -> add -> abs -> store CPU case (reg-base, int8) -*- C++ -*-===//
//
// int8 的 VF 边界，全程 int8 的那一种。bridge 收集子图时有一道 dtype 白名单
// （`ScheduleToAfir.cpp` 的 `AppendVectorFuncBody`），int8 这一位此前不在名单上：带 int8
// 边界的 VF 在那里报能力不足、整张回落。名单一放开，就得有人证明它真的降得下去——
// 那道白名单上方的注释写的就是这件事：放开一位 dtype，拦住它的就从「干净回落」变成
// 「ODS 与 conversion 的硬失败」，而硬失败只有真编一遍 kernel 才看得见。
//
// 图是 `load_add_abs_reg_base_store` 把 dtype 换成 int8。`add` 与 `abs` 在 legacy 那一侧都
// 不升精度，所以两个节点原样进 VF：源边界、中间结果、落点全是 int8。这一格压的是别的
// int8 用例压不到的两件事——lane 按 1 字节算（一段 lane 的元素数是 2 字节 lane 的两倍），
// 以及等宽的取与存不带 dist。`Reg::Add` 与 `Reg::Abs` 都收 int8（dav_3510 的 `AddImpl` /
// `AbsImpl`）。
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_add_abs_i8_reg_base_store/LoadAddAbsI8RegBaseStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_add_abs_i8_reg_base_store;

  const auto output_shape = config::GetShapeConfig(config::kOutputShapes, 0).shape;
  const auto input0_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 0).shape);
  const auto input1_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 1).shape);
  AscGraphBuilder builder(config::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_INT8)
      .Data("input1", 1, input1_shape, {}, af::DT_INT8)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {});

  auto graph = builder.Add("add", "load0", "load1")
                   .Abs("abs", "add")
                   .Store("store", "abs")
                   .Output("output", "store", 0, af::DT_INT8)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  namespace config = load_add_abs_i8_reg_base_store;
  EXPECT_NE(kernel.find(std::string(config::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 产物必须是寄存器形态：一个向量作用域、一层 lane 循环，中间结果不落 UB。
  EXPECT_NE(kernel.find("__VEC_SCOPE__"), std::string::npos);
  // 最宽的 dtype 就是 int8：lane 按 1 字节算。
  EXPECT_NE(kernel.find("AscendC::Reg::UpdateMask<int8_t>("), std::string::npos);
  // 两路取数、一路存回，都等宽、不带 dist。计数式而不是「存在一条」。
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::LoadAlign<int8_t>("), 2U);
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::StoreAlign<int8_t>("), 1U);
  EXPECT_EQ(CountOccurrences(kernel, "LoadAlign"), 2U);
  EXPECT_EQ(CountOccurrences(kernel, "StoreAlign"), 1U);
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::Add("), 1U);
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::Abs("), 1U);
  // 全程 int8，VF 里不该出现任何一次类型转换。
  EXPECT_EQ(kernel.find("AscendC::Reg::Cast<"), std::string::npos);
  // 真的走了 VF 这条路：tile 级的包装一条都不该剩。
  EXPECT_EQ(kernel.find("AbsExtend"), std::string::npos);
  EXPECT_EQ(kernel.find("AddExtend"), std::string::npos);
}
}  // namespace mlir::afir
