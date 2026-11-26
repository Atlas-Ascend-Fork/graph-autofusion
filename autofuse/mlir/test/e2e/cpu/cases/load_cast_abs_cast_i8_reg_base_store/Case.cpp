/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> cast -> abs -> cast -> store (reg-base, int8) -*- C++ -*-===//
//
// int8 的 VF 边界，跨 dtype 的那一种——`load_cast_abs_cast_reg_base_store` 把两个边界换成
// int8。int8 进、int8 出、中间走 half：lane 按 2 字节算，1 字节的边界取进来时由
// `DIST_UNPACK_B8` 撑进 lane 槽位、存回去时由 `DIST_PACK_B16` 压回。`cast_abs` 那一格
// 钉的是 half 进 float 出的同一组事实；这里是 1 字节与 2 字节之间的那一组，dist 的取法、
// 两个 cast trait 都是另一对，取错了编得过、跑得完、静默算错。
//
// 两个 Cast 是图上显式写的：legacy 的 VF 只收 int8 -> half 与 half -> int8 这一对
// （`CastAscIrCodegenImplV2::IsVectorFunctionSupported` 的表），优化器不会再插别的。
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_cast_abs_cast_i8_reg_base_store/LoadCastAbsCastI8RegBaseStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_cast_abs_cast_i8_reg_base_store;

  const auto output_shape = config::GetShapeConfig(config::kOutputShapes, 0).shape;
  const auto input0_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 0).shape);
  AscGraphBuilder builder(config::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_INT8)
      .Load("load0", "input0", input0_shape, {});

  auto graph = builder.Cast("cast_up", "load0", af::DT_FLOAT16)
                   .Abs("abs", "cast_up")
                   .Cast("cast_down", "abs", af::DT_INT8)
                   .Store("store", "cast_down")
                   .Output("output", "store", 0, af::DT_INT8)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  namespace config = load_cast_abs_cast_i8_reg_base_store;
  EXPECT_NE(kernel.find(std::string(config::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 产物必须是寄存器形态：一个向量作用域、一层 lane 循环，中间结果不落 UB。
  EXPECT_NE(kernel.find("__VEC_SCOPE__"), std::string::npos);
  // lane 的宽度按最宽的那个 dtype：中间那一趟是 half。
  EXPECT_NE(kernel.find("AscendC::Reg::UpdateMask<half>("), std::string::npos);
  // 两条 dist，连 dist 一起看：不带 dist 的 `LoadAlign<int8_t>` 同样是一条 `LoadAlign`。
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::LoadAlign<int8_t, AscendC::Reg::LoadDist::DIST_UNPACK_B8>"), 1U);
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::StoreDist::DIST_PACK_B16>"), 1U);
  EXPECT_EQ(CountOccurrences(kernel, "LoadAlign"), 1U);
  EXPECT_EQ(CountOccurrences(kernel, "StoreAlign"), 1U);
  // 一来一回两个 cast trait，加一条寄存器形态的 Abs。
  EXPECT_NE(kernel.find("AscendC::Reg::Cast<half, int8_t, cast_trait_bf16_2_float>"), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::Reg::Cast<int8_t, half, cast_trait_half_2_int8>"), std::string::npos);
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::Abs("), 1U);
  // 真的走了 VF 这条路：tile 级的包装一条都不该剩。
  EXPECT_EQ(kernel.find("AbsExtend"), std::string::npos);
  EXPECT_EQ(kernel.find("CastExtend"), std::string::npos);
}
}  // namespace mlir::afir
