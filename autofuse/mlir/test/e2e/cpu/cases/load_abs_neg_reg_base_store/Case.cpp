/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 验证寄存器形态 afir.neg 对应 Reg::Neg，而不是仅检查生成的 API 名称。
// 采用 abs -> neg，使结果为 -|x|：负输入可发现遗漏 abs，非零输入可发现遗漏 neg，
// 交换算子顺序也会得到错误的正值。反向的 neg -> abs 会掩盖遗漏 neg 的问题。
// 两节点均支持 VF 且满足 kMinVfNodesNum = 2，会融合成 VectorFunc。
// 单节点 load_neg_reg_base_store 使用 tile 级 NegExtend，不能覆盖 Reg::Neg。
// 设计依据见 2026-10-18-afir-vector-func-implementation-design.md 功能需求 5。

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_abs_neg_reg_base_store/LoadAbsNegRegBaseStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_abs_neg_reg_base_store;

  const auto output_shape = config::GetShapeConfig(config::kOutputShapes, 0).shape;
  const auto input_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 0).shape);
  AscGraphBuilder builder(config::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  auto graph = builder.Abs("abs", "load0")
                   .Neg("neg", "abs")
                   .Store("store", "neg")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_abs_neg_reg_base_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 产物必须是寄存器形态：一个向量作用域、一层 lane 循环，中间结果不落 UB。
  EXPECT_NE(kernel.find("__VEC_SCOPE__"), std::string::npos);
  // 新加的那一行真的被用上了。顺序由数值钉（见文件头），这里只钉「生成了这条 API」。
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::Neg("), 1U);
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::Abs("), 1U);
  // 一个输入取一次、结果存一次。计数式而不是「存在一条」：少生成一条 Load 也能编过、
  // 也能算对（上一个 lane 的寄存器还在），错的是别的形状才会暴露。
  EXPECT_EQ(CountOccurrences(kernel, "LoadAlign"), 1U);
  EXPECT_EQ(CountOccurrences(kernel, "StoreAlign"), 1U);
  // 这两条是「真的走了 VF 这条路」的判据：tile 级的包装一条都不该剩。
  EXPECT_EQ(kernel.find("NegExtend"), std::string::npos);
  EXPECT_EQ(kernel.find("AbsExtend"), std::string::npos);
}
}  // namespace mlir::afir
