/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> cast -> abs -> cast -> store (reg-base) -*- C++ -*-===//
//
// **第一个中途换 dtype 的 VF（M2）。** M1 那格（`load_add_abs_reg_base_store`）全程 f32，
// 所以 lane 的宽度与边界的宽度是同一个数，三件只在跨 dtype 下才成立的事一条都走不到：
//
//   1. lane 的宽度由**最宽**的那个元素类型定。这一格 half 进、half 出、中间走 float，
//      所以一段 lane 装 `VL/4` 个元素而不是 `VL/2` 个。取错（按 half 取）的后果是
//      **第一趟 lane 仍然全对、从第 65 个元素起才错**（实测），所以形状要够跑多趟，
//      见 CMakeLists。
//   2. 窄边界的搬运要带 `dist`：取进来 `DIST_UNPACK_B16`（把 half 撑进 4 字节的 lane
//      槽位）、存回去 `DIST_PACK_B32`（压回 half）。**不带就是 `DIST_NORM`，编得过、
//      跑得完、静默算错**——这正是 CPU 模拟器能当场逮住的那一类。
//   3. `Reg::Cast<OutT, InT, trait>` 的三个模板实参，而一来一回挑的是不同的两个 trait。
//
// 图是 legacy 的 `cast_abs_test`（`tests/v35/st/backend_e2e_v2/cast_abs_test/`）那一种：
// 三个节点全部 VF-capable 且相邻（`CastAscIrCodegenImplV2::IsVectorFunctionSupported` 里
// f16 -> f32 与 f32 -> f16 都在表上），所以调度在 buffer 分配阶段把它们聚成一个
// `VectorFunc`。
//
// **它在 M2 落地之前是红的**：M1 的能力判据里有「边界混了元素类型就报能力不足」那一条，
// 而 CPU E2E 的生成器直接调 `RunMlirCodegen`、不接 legacy 回落，所以失败在生成那一步。
//
// 设计见 docs/development/2026-10-18-afir-vector-func-implementation-design.md 的 M2。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_cast_abs_cast_reg_base_store/Case.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_cast_abs_cast_reg_base_store;

  const auto output_shape = config::GetShapeConfig(config::kOutputShapes, 0).shape;
  const auto input_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 0).shape);
  AscGraphBuilder builder(config::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input_shape, {}, af::DT_FLOAT16)
      .Load("load0", "input0", input_shape, {});

  auto graph = builder.Cast("cast_up", "load0", af::DT_FLOAT)
                   .Abs("abs", "cast_up")
                   .Cast("cast_down", "abs", af::DT_FLOAT16)
                   .Store("store", "cast_down")
                   .Output("output", "store", 0, af::DT_FLOAT16)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_cast_abs_cast_reg_base_store::kKernelName) + "_subgraph0_plan0"),
            std::string::npos);
  // 产物必须是寄存器形态：一个向量作用域、一层 lane 循环，中间结果不落 UB。
  EXPECT_NE(kernel.find("__VEC_SCOPE__"), std::string::npos);
  // **lane 的宽度按最宽的那个 dtype**。写成 `<half>` 也编得过、也跑得完，只有尾段会错。
  EXPECT_NE(kernel.find("AscendC::Reg::UpdateMask<float>("), std::string::npos);
  // 两条 dist。**计数式而不是「存在一条」**，而且连 dist 一起看：不带 dist 的
  // `LoadAlign<half>` 同样是一条 `LoadAlign`，数条数数不出区别。
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>"), 1U);
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::StoreAlign<half, AscendC::Reg::StoreDist::DIST_PACK_B32>"), 1U);
  EXPECT_EQ(CountOccurrences(kernel, "LoadAlign"), 1U);
  EXPECT_EQ(CountOccurrences(kernel, "StoreAlign"), 1U);
  // 一来一回挑的是**不同**的两个 trait（舍入方向不一样）。两条写成同一个也编得过。
  EXPECT_NE(kernel.find("AscendC::Reg::Cast<float, half, cast_trait_bf16_2_float>"), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::Reg::Cast<half, float, cast_trait_float_2_half>"), std::string::npos);
  // trait 的**定义**要被内联进来，否则编译器找不到那个名字。
  EXPECT_NE(kernel.find("cast_trait_float_2_half = {"), std::string::npos);
  // 这几条是「真的走了 VF 这条路」的判据：tile 级的包装一条都不该剩。
  EXPECT_EQ(kernel.find("CastExtend"), std::string::npos);
  EXPECT_EQ(kernel.find("AbsExtend"), std::string::npos);
}
}  // namespace mlir::afir
