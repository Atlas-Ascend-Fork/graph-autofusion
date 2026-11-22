/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 覆盖双张量 Add 前融合后执行 Sum 的路径。两个输入都参与 tile 寻址，
// 需要分别物化多维有效视图的局部跨度，再使用相同 calCount。
// 一元前融合和带 Scalar 的二元算子都不能覆盖两份张量跨度。
// 形状 [16, 37] 的 R 轴未按 32B 对齐；Add 处理 padding 后，ReduceInit 必须将其
// 恢复为 Sum 单位元 0，与 load_neg_sum_store_16x37 对照。

#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <vector>

#include "test/e2e/cpu/cases/load_add_sum_store/LoadAddSumStoreCase.h"
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
  // 与 `load_neg_sum_store` 同一规则：单 plan 单模板，自带 key 不会撞到自增计数器。
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
