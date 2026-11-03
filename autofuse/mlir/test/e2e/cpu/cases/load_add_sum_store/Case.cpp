/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load, load -> add -> sum -> store CPU case -----*- C++ -*-===//
//
// 归约**之前**融合一个**二元**算子，两侧都是张量输入。
//
// 与 `load_neg_sum_store`（同是前融合）并列而不是给它加一组形状：那条是一元，被归约的
// tile 只有一个生产者；这条有**两个**，于是第一次走到「同一个多维有效视图上，两个输入侧的
// 局部跨度要各自物化、再喂给同一条 `calCount`」这件事。`2026-09-19-ub-tile-addressing-
// model-design.md` 7.2 的探针表里 `load_add_sum_store` 一行记的就是它（当时的表现是
// `failed to legalize 'afir.add'`），从那以后一直没有注册。
//
// 为什么此前漏掉：reduce 侧的融合语料全是一元（`Neg`、`Relu`）或带 `Scalar`
// （`load_sum_mul_store`）。带 `Scalar` 的那条虽然是二元 op，但第二个操作数不是张量、不参与
// tile 寻址，所以它覆盖不到两份局部跨度这件事。
//
// 形状取 `[16, 37]`：R 轴 37 不是 32B 对齐，Add 要覆盖到 padding 之后 `ReduceInit` 仍要能把
// 补出来的位置恢复成 sum 的单位元——与 `load_neg_sum_store_16x37` 同一组形状，便于对账。
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <vector>

#include "test/e2e/cpu/cases/load_add_sum_store/Case.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

#include "ascgraph_info_complete.h"

namespace mlir::afir {
namespace {

// 与 `load_neg_sum_store` 同一份：归约后的 `Output` 视图要按被归约维 repeats=1 / stride=0
// 写全，否则完备化补出来的是未归约的形状。
void SetOutputView(af::AscGraph &graph, const char *name, std::initializer_list<int64_t> shape) {
  const auto output = graph.FindNode(name);
  assert(output != nullptr && output->outputs().size() == 1U);
  auto &view = output->outputs()[0]->attr;
  view.repeats = ToExpressions(shape);
  view.strides.clear();
  int64_t stride = 1;
  for (auto dimension = std::rbegin(shape); dimension != std::rend(shape); ++dimension) {
    view.strides.push_back(af::Symbol(*dimension == 1 ? 0 : stride));
    if (*dimension != 1) {
      stride *= *dimension;
    }
  }
  std::reverse(view.strides.begin(), view.strides.end());
}

}  // namespace

af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace cfg = load_add_sum_store;

  const auto input0_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 0).shape);
  const auto input1_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 1).shape);
  const auto input_shape = cfg::GetShapeConfig(cfg::kInputShapes, 0).shape;
  assert(input_shape.size() == 2U);

  AscGraphBuilder builder(cfg::kKernelName);
  builder.Loops(input_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_FLOAT)
      .Data("input1", 1, input1_shape, {}, af::DT_FLOAT)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {})
      .Add("add0", "load0", "load1")
      .Sum("sum0", "add0", {1U});

  auto graph = builder.Store("store0", "sum0").Output("output", "store0", 0, af::DT_FLOAT).Build();
  SetOutputView(graph, "output", cfg::GetShapeConfig(cfg::kOutputShapes, 0).shape);
  EXPECT_EQ(optimize::AscGraphInfoComplete::CompleteApiInfo(graph), af::SUCCESS);
  // 与 `load_neg_sum_store` 同一口径：单 plan 单模板，自带 key 不会撞到自增计数器。
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  namespace cfg = load_add_sum_store;
  EXPECT_NE(kernel.find(std::string(cfg::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 两份张量输入都要真的进 UB 再相加：`AscendC::Add` 的三个操作数全是 LocalTensor。
  EXPECT_NE(kernel.find("AscendC::Add"), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::ReduceSum"), std::string::npos);
  // 末维归约 -> AR；R 不对齐，补出来的位置必须被 ReduceInit 填成 0，否则会被加进和里。
  EXPECT_NE(kernel.find("AscendC::Pattern::Reduce::AR"), std::string::npos);
  EXPECT_EQ(kernel.find("ReduceInit<"), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::Duplicate"), std::string::npos);
}

}  // namespace mlir::afir
