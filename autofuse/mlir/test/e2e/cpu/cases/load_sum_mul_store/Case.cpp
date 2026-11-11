/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED.
 * See the License for the specific language governing permissions and limitations under the License.
 */

// IsNotPartitionReduce 检测到归约后继节点还使用链外 Scalar 输入，因此触发切图。
// 两段分别为 Sum -> Store -> Workspace 和 Workspace -> Load -> Mul -> Store。
// 此处由链外输入判据触发，与节点数阈值 4 无关。

#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <vector>

#include "test/e2e/cpu/cases/load_sum_mul_store/LoadSumMulStoreCase.h"
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
  namespace cfg = load_sum_mul_store;

  const auto input_shape = cfg::GetShapeConfig(cfg::kInputShapes, 0).shape;
  const auto input_exprs = ToExpressions(input_shape);
  assert(input_shape.size() == 2U);

  AscGraphBuilder builder(cfg::kKernelName);
  builder.Loops(input_shape)
      .Data("input0", 0, input_exprs, {}, af::DT_FLOAT)
      .Load("load0", "input0", input_exprs, {})
      .Sum("sum0", "load0", {1U});

  auto graph = builder.Scalar("two", "2.0", af::DT_FLOAT)
                   .Mul("mul0", "sum0", "two")
                   .Store("store0", "mul0")
                   .Output("output", "store0", 0, af::DT_FLOAT)
                   .Build();
  SetOutputView(graph, "output", cfg::GetShapeConfig(cfg::kOutputShapes, 0).shape);
  EXPECT_EQ(optimize::AscGraphInfoComplete::CompleteApiInfo(graph), af::SUCCESS);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_sum_mul_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::ReduceSum"), std::string::npos);
  // 切图之后乘法落在第二张图里，读的是 workspace。
  EXPECT_NE(kernel.find("Muls<float, 1>("), std::string::npos);
}

}  // namespace mlir::afir
