/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED.
 * See the License for the specific language governing permissions and limitations under the License.
 */

//===- Case.cpp - load -> sum -> relu -> store CPU case ---------*- C++ -*-===//
//
// 归约**之后**融合一个一元算子，**不切图**的那一支。
//
// 与 `load_sum_mul_store` 并列而不是合并：那一条的后继链带外部输入（`Scalar`），
// `IsNotPartitionReduce` 因此判要切图；这一条是一元、后继链上只有 Relu 与 Store 两个节点，
// 既没有外部输入、也没超过阈值 4，于是**留在同一张图里**。两支走的是完全不同的代码路径，
// 现场也完全不同——切图那支挂在 workspace 的形状推导上，这一支挂在归约结果的资源绑定上。
//
// legacy 支持这种后融合（`reduce_api_call.cpp` 的目的地寻址不看容器），所以这是 MLIR 相对
// legacy 的能力退化，不是双方都没有的新场景；设计文档
// `2026-09-19-ub-tile-addressing-model-design.md` 的「Reduce 后融合」一行记的就是它。
//
// 形状取 A 小 R 大（`A=8 < corenum`），让 RCore 两相切分成为候选：现场报的正是二相图里那个
// 归约节点的绑定冲突，钉不到这个候选就钉不到这条缺陷。**所以这条注册不钉 plan**。
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <vector>

#include "test/e2e/cpu/cases/load_sum_relu_store/Case.h"
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
  namespace cfg = load_sum_relu_store;

  const auto input_shape = cfg::GetShapeConfig(cfg::kInputShapes, 0).shape;
  const auto input_exprs = ToExpressions(input_shape);
  assert(input_shape.size() == 2U);

  AscGraphBuilder builder(cfg::kKernelName);
  auto graph = builder.Loops(input_shape)
                   .Data("input0", 0, input_exprs, {}, af::DT_FLOAT)
                   .Load("load0", "input0", input_exprs, {})
                   .Sum("sum0", "load0", {1U})
                   .Relu("relu0", "sum0")
                   .Store("store0", "relu0")
                   .Output("output", "store0", 0, af::DT_FLOAT)
                   .Build();
  SetOutputView(graph, "output", cfg::GetShapeConfig(cfg::kOutputShapes, 0).shape);
  EXPECT_EQ(optimize::AscGraphInfoComplete::CompleteApiInfo(graph), af::SUCCESS);
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_sum_relu_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::ReduceSum"), std::string::npos);
  // 不带模板实参：`Relu` 走的是模板实参可推导的那条重载，与 `LeakyRelu<float, 1>` 不同。
  EXPECT_NE(kernel.find("AscendC::Relu("), std::string::npos);
}

}  // namespace mlir::afir
