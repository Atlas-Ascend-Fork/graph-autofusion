/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> sum -> add(tensor) -> store CPU case ---*- C++ -*-===//
//
// 归约**之后**融合一个**二元**算子，第二个操作数是**归约后形状**的张量（`sum(x, 1) + bias`）。
//
// 与 `load_add_sum_store`（前融合二元）成镜像。它覆盖的是**切图后的消费者子图带一个真实的
// 外部张量输入**这件事。
//
// 实测产物是**两个 stage**（stage 0 做 Sum 并 Store 进 workspace，stage 1 做 Load + Add +
// Store），而 `load_add_sum_store` 只有一个。这不是偶然——`ReducePartitionPostFusion` 的
// `IsNotPartitionReduce` 一旦发现后继链上某个节点有**不属于后继链**的输入就判要切图，而二元
// 后融合的第二个操作数天然就不属于后继链。**所以二元后融合必然切图**，留在同一张图、走
// `LoopGenerate` 的 `redirectAccumulatedOperand` 的只有一元那条（`load_sum_relu_store`）。
//
// 与带 `Scalar` 的 `load_sum_mul_store` 的区别就在这里：那条切出来的消费者子图只有 workspace
// 一个 GM 输入，`Scalar` 在 kernel 顶部物化、不参与 tile 寻址；这条的消费者子图有**两个** GM
// 输入——workspace 加上原图的 `input1`，后者要在归约后的 tile 上自己算一份局部跨度。
//
// `2026-09-19-ub-tile-addressing-model-design.md` 7.2 的探针表里 `load_sum_add_store` 一行记的
// 就是它（当时的表现是 `failed to legalize 'afir.sum'`），从那以后一直没有注册。
//
// 形状取 `[16, 37]` + `[16, 1]`：R 轴 37 不是 32B 对齐，补出来的位置由 `ReduceInit` 填成 sum
// 的单位元；bias 只有 A 这一维。与 `load_add_sum_store_16x37` 同一组形状，便于两支对账。
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <vector>

#include "test/e2e/cpu/cases/load_sum_add_store/LoadSumAddStoreCase.h"
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
  namespace cfg = load_sum_add_store;

  const auto input0_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 0).shape);
  // 第二份输入是**归约后**形状 `[A, 1]`：被归约维 repeats 1 / stride 0，与 `SetOutputView`
  // 给 `output` 写的同一口径。它不是 broadcast——Add 的两侧在同一个归约后的 tile 上。
  const auto input1_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 1).shape);
  const std::vector<af::Expression> input1_strides{af::Symbol(1), af::Symbol(0)};
  const auto input_shape = cfg::GetShapeConfig(cfg::kInputShapes, 0).shape;
  assert(input_shape.size() == 2U);

  AscGraphBuilder builder(cfg::kKernelName);
  builder.Loops(input_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_FLOAT)
      .Data("input1", 1, input1_shape, input1_strides, af::DT_FLOAT)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, input1_strides)
      .Sum("sum0", "load0", {1U})
      .Add("add0", "sum0", "load1");

  auto graph = builder.Store("store0", "add0").Output("output", "store0", 0, af::DT_FLOAT).Build();
  SetOutputView(graph, "output", cfg::GetShapeConfig(cfg::kOutputShapes, 0).shape);
  EXPECT_EQ(optimize::AscGraphInfoComplete::CompleteApiInfo(graph), af::SUCCESS);
  // 与 `load_neg_sum_store` 同一口径：单 plan 单模板，自带 key 不会撞到自增计数器。
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  namespace cfg = load_sum_add_store;
  EXPECT_NE(kernel.find(std::string(cfg::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 必须真的切成两个 stage：只有一个就说明没走切图那条路，测的是别的东西。
  EXPECT_NE(kernel.find("stage1_template0"), std::string::npos);
  // 两份张量输入都要真的进 UB 再相加：`AscendC::Add` 的三个操作数全是 LocalTensor。
  EXPECT_NE(kernel.find("AscendC::Add"), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::ReduceSum"), std::string::npos);
  // Add 必须排在 ReduceSum **之后**：顺序错了测的就是前融合。
  EXPECT_LT(kernel.find("AscendC::ReduceSum"), kernel.rfind("AscendC::Add"));
  // 末维归约 -> AR；R 不对齐，补出来的位置必须被 ReduceInit 填成 0，否则会被加进和里。
  EXPECT_NE(kernel.find("AscendC::Pattern::Reduce::AR"), std::string::npos);
  EXPECT_EQ(kernel.find("ReduceInit<"), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::Duplicate"), std::string::npos);
}

}  // namespace mlir::afir
