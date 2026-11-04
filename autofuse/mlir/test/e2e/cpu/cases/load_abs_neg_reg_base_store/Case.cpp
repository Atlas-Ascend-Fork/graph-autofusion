/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> abs -> neg -> store CPU case (reg-base) -*- C++ -*-===//
//
// **`afir.neg` 那一行的那一格用例。**
//
// 算子表加一行声明的是「`AscendC::Reg::Neg` 与 `afir.neg` 同义」，而同义是个会错的判断
// ——所以规矩是「加一行要带一格用例」，且那一格必须**验数值**：表里填错一个 API（比如填
// `Reg::Not`，位取反而不是取相反数）在 IR 和 C++ 文本上都看不出不对，只有算一遍才知道。
//
// 图取 `abs` 在前、`neg` 在后，不是反过来：
//
//   - `abs` → `neg` 是 `-|x|`，**两个算子都可观测**：漏发 `neg` 得到 `|x|`（x≠0 全错）、
//     漏发 `abs` 得到 `-x`（x<0 全错）、两者**顺序写反**（先 neg 再 abs）得到 `|x|`（x≠0
//     全错）。所以这一格连算子顺序一起钉住了，而顺序错的产物同样编得过。
//   - 反过来（`neg` → `abs`）是 `|x|`，`neg` 被 `abs` 吃掉，漏发它一个元素都不会错。
//
// 两个节点都无条件 VF-capable（`NegAscIrCodegenImplV2::IsVectorFunctionSupported` 恒为
// true），相邻且够 `kMinVfNodesNum = 2`，所以调度会把它们聚成一个 `VectorFunc`。
//
// 已有的 `load_neg_reg_base_store` 不能代替这一格：它只有一个计算节点，够不上融合门槛，
// 走的是 tile 级的 `NegExtend` 包装，与寄存器形态的 `Reg::Neg` 是两条路。
//
// 设计见 docs/development/2026-10-18-afir-vector-func-implementation-design.md 的功能需求 5。
//
//===----------------------------------------------------------------------===//

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
  // 新加的那一行真的被用上了。顺序由数值钉（见文件头），这里只钉「发了这条 API」。
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::Neg("), 1U);
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::Abs("), 1U);
  // 一个输入取一次、结果存一次。**计数式而不是「存在一条」**：少发一条 Load 也能编过、
  // 也能算对（上一个 lane 的寄存器还在），错的是别的形状才会暴露。
  EXPECT_EQ(CountOccurrences(kernel, "LoadAlign"), 1U);
  EXPECT_EQ(CountOccurrences(kernel, "StoreAlign"), 1U);
  // 这两条是「真的走了 VF 这条路」的判据：tile 级的包装一条都不该剩。
  EXPECT_EQ(kernel.find("NegExtend"), std::string::npos);
  EXPECT_EQ(kernel.find("AbsExtend"), std::string::npos);
}
}  // namespace mlir::afir
