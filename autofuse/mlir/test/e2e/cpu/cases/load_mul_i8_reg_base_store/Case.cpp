/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load x2 -> mul -> store CPU case (reg-base, int8) -*- C++ -*-===//
//
// 两路原始 int8 输入相乘。图上只有 `load x2 -> mul -> store`，升精度不是用例写的：寄存器
// 形态的 `Reg::Mul` 没有 8 位分支，legacy 在调度之前就把 int8 的乘法升到 int16，于是落进
// `VectorFunc` 的是「两路各升一次 -> 乘」。这一格把那条路固定下来：
//
//   1. 乘法确实在 i16 上算：`Reg::Mul` 的三个操作数都是 `RegTensor<int16_t>`。要是有人让
//      它直接在 i8 上算，kernel 编译期才 `static_assert`（`MulImpl` 的类型白名单）。
//   2. 乘积以 i16 存回 UB，降回 int8 是 tile 级的那一个 `CastExtend`。int16 -> int8 这一支
//      带饱和（`v35/ascendc/api_regbase/cast.h` 的 `CastExtendInt16ToInt8`）：超出 int8 的
//      乘积被钳到 [-128, 127]，而不是截掉高位。这一条只有数值钉得住，见 KernelTest.cpp。
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_mul_i8_reg_base_store/LoadMulI8RegBaseStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_mul_i8_reg_base_store;

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
      builder.Mul("mul", "load0", "load1").Store("store", "mul").Output("output", "store", 0, af::DT_INT8).Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  namespace config = load_mul_i8_reg_base_store;
  EXPECT_NE(kernel.find(std::string(config::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 产物里有一段寄存器形态：一个向量作用域、一层 lane 循环。
  EXPECT_NE(kernel.find("__VEC_SCOPE__"), std::string::npos);
  // 两路取数：1 字节的边界被 `DIST_UNPACK_B8` 撑进 2 字节的 lane 槽位。计数式而不是
  // 「存在一条」：少生成一条 Load 也能编过、也能算对（上一趟的寄存器还在）。
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::LoadAlign<int8_t, AscendC::Reg::LoadDist::DIST_UNPACK_B8>"), 2U);
  EXPECT_EQ(CountOccurrences(kernel, "LoadAlign"), 2U);
  EXPECT_EQ(CountOccurrences(kernel, "StoreAlign"), 1U);
  // 乘法在 i16 上算：先把两路操作数各升一次，再乘；lane 也按 i16 算。
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::Cast<int16_t, int8_t, cast_trait_bf16_2_float>("), 2U);
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::UpdateMask<int16_t>("), 1U);
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::Mul("), 1U);
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::RegTensor<int16_t>"), 3U);
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::RegTensor<int8_t>"), 2U);
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::StoreAlign<int16_t>("), 1U);
  // 降回 int8 是 tile 级的那一个 `CastExtend(dst, src, ...)`：目的/src 两个实参的类型说明它是
  // 从 UB 上的 int16 降到 int8。实参是 SSA 名字，随调度漂移，所以按名字回去查声明。
  const size_t narrow = kernel.find("\n    CastExtend(");
  ASSERT_NE(narrow, std::string::npos);
  const size_t open = kernel.find('(', narrow);
  const size_t comma = open == std::string::npos ? std::string::npos : kernel.find(',', open);
  const size_t end = comma == std::string::npos ? std::string::npos : kernel.find(',', comma + 1U);
  ASSERT_TRUE(open != std::string::npos && comma != std::string::npos && end != std::string::npos);
  const std::string destination = kernel.substr(open + 1U, comma - open - 1U);
  const std::string source = kernel.substr(comma + 2U, end - comma - 2U);
  EXPECT_NE(kernel.find("AscendC::LocalTensor<int8_t> " + destination), std::string::npos) << destination;
  EXPECT_NE(kernel.find("AscendC::LocalTensor<int16_t> " + source), std::string::npos) << source;
  // 真的走了 VF 这条路：tile 级的乘法包装一条都不该剩。
  EXPECT_EQ(kernel.find("MulExtend"), std::string::npos);
}
}  // namespace mlir::afir
