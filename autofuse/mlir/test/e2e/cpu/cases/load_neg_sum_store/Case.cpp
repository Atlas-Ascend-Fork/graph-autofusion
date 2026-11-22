/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED.
 * See the License for the specific language governing permissions and limitations under the License.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <vector>

#include "test/e2e/cpu/cases/load_neg_sum_store/LoadNegSumStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

#include "ascgraph_info_complete.h"

namespace mlir::afir {
namespace {

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

  const auto input_shape = load_neg_sum_store::GetShapeConfig(load_neg_sum_store::kInputShapes, 0).shape;
  const auto input_exprs = ToExpressions(input_shape);
  std::vector<size_t> reduce_axes;
  if (input_shape.size() == 2U) {
    reduce_axes = {1U};
  } else {
    // 与 `load_sum_store` 的四维注册同一张 ARAR 图：保留 z0/z2、归约 z1/z3。这里多一个
    // Neg，用来验证多项有效视图上的 elementwise 与 reduce 能落在同一个 tile 上。
    assert(input_shape.size() == 4U);
    reduce_axes = {1U, 3U};
  }
  AscGraphBuilder builder(load_neg_sum_store::kKernelName);
  builder.Loops(input_shape)
      .Data("input0", 0, input_exprs, {}, af::DT_FLOAT)
      .Load("load0", "input0", input_exprs, {})
      .Neg("neg0", "load0")
      .Sum("sum0", "neg0", reduce_axes);

  auto graph = builder.Store("store0", "sum0").Output("output", "store0", 0, af::DT_FLOAT).Build();
  SetOutputView(graph, "output", GetShapeConfig(load_neg_sum_store::kOutputShapes, 0).shape);
  EXPECT_EQ(optimize::AscGraphInfoComplete::CompleteApiInfo(graph), af::SUCCESS);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_neg_sum_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  EXPECT_NE(kernel.find("NegExtend<float>("), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::ReduceSum"), std::string::npos);
}

}  // namespace mlir::afir
