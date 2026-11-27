/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> cast -> sqrt -> cast -> store (reg-base) -*- C++ -*-===//
//
// 算子表里 `afir.sqrt` 那一行（`AFIRVectorFuncToRegBase.cpp`）的用例。表头上写着「加一行要
// 带一格用例」——一行声明的是「这个 AscendC::Reg API 与这个 AFIR 算子同义」，而 lit 只钉
// 产物文本，`Reg::Sqrt` 的调用编不编得过、算得对不对，要真编一遍、真跑一遍才知道。
//
// 图上写的是 bf16 的 sqrt，与 legacy 的 `bf16_sqrt_test` 同形。`Reg::Sqrt` 只收 half /
// float，所以优化器在调度之前把它升到 f32、两头各插一个 Cast：
// cast(bf16 -> f32) -> sqrt(f32) -> cast(f32 -> bf16)。三个节点都 VF-capable 且相邻，
// 调度把它们聚成一个 `VectorFunc`，落到寄存器形态就是跨 dtype 的那一种：2 字节的边界被
// `DIST_UNPACK_B16` 撑进 4 字节的 lane，算完再由 `DIST_PACK_B32` 压回去。
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_sqrt_bf16_reg_base_store/LoadSqrtBf16RegBaseStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_sqrt_bf16_reg_base_store;

  const auto output_shape = config::GetShapeConfig(config::kOutputShapes, 0).shape;
  const auto input0_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 0).shape);
  AscGraphBuilder builder(config::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_BF16)
      .Load("load0", "input0", input0_shape, {});

  auto graph = builder.Sqrt("sqrt", "load0").Store("store", "sqrt").Output("output", "store", 0, af::DT_BF16).Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  namespace config = load_sqrt_bf16_reg_base_store;
  EXPECT_NE(kernel.find(std::string(config::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 产物必须是寄存器形态：一个向量作用域、一层 lane 循环，中间结果不落 UB。
  EXPECT_NE(kernel.find("__VEC_SCOPE__"), std::string::npos);
  // lane 的宽度按最宽的那个 dtype：中间那一趟是 f32。
  EXPECT_NE(kernel.find("AscendC::Reg::UpdateMask<float>("), std::string::npos);
  // 两条 dist，连 dist 一起看：不带 dist 的 `LoadAlign<bfloat16_t>` 同样是一条 `LoadAlign`。
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>"),
            1U);
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::StoreAlign<bfloat16_t, AscendC::Reg::StoreDist::DIST_PACK_B32>"),
            1U);
  EXPECT_EQ(CountOccurrences(kernel, "LoadAlign"), 1U);
  EXPECT_EQ(CountOccurrences(kernel, "StoreAlign"), 1U);
  // 一来一回两个 Cast，加这一格的主角一条寄存器形态的 Sqrt。
  EXPECT_NE(kernel.find("AscendC::Reg::Cast<float, bfloat16_t, cast_trait_bf16_2_float>"), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::Reg::Cast<bfloat16_t, float, cast_trait_float_2_bf16>"), std::string::npos);
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::Sqrt("), 1U);
  // 真的走了 VF 这条路：tile 级的包装一条都不该剩。
  EXPECT_EQ(kernel.find("SqrtExtend"), std::string::npos);
  EXPECT_EQ(kernel.find("CastExtend"), std::string::npos);
}
}  // namespace mlir::afir
