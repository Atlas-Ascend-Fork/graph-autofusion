/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED.
 * See the License for the specific language governing permissions and limitations under the License.
 */

//===- Case.cpp - load -> sum(dim 0) -> mul(scalar) -> store CPU case -*- C++ -*-===//
//
// 与 `load_sum_mul_store` 只差一处：归约的是**首轴**（RA 布局）而不是末轴。
//
// 分开一条而不是给那份加一组形状：真机上这两支的结果不同——末轴那支全绿，首轴那支在
// `[512, 512]` 上**算错值**（`sum(x, 0) * 2` 与 `mean(x, 0)` 的误差逐位相同，所以不是 mean
// 的问题），而 `[256, 512]`、`[64, 64]` 又是对的。首轴归约**不切图**（`relu(sum(x, 0))`）
// 也是对的，所以被测的是「首轴 + 切图」这个组合。
//
// 归约**之后**再接一个带外部输入的节点。这一类此前没有用例，而它走的是一条与
// `load_neg_sum_store`（归约之前融合）完全不同的路：
//
// `ReducePartitionPostFusion` 的 `IsNotPartitionReduce` 一旦发现归约的后继链上某个节点还
// 有一个**不属于后继链**的输入（这里是 `Scalar`），就判「要切图」，于是
// `PartitionReduceNode` 把图切成 `... -> Sum -> Store -> Workspace` 与
// `Workspace -> Load -> Mul -> Store -> ...` 两张。节点数那条阈值（4）在这里没起作用——
// 两个节点也照样切。
//
// 这正是 `torch.mean` 走的路：inductor 不发 Mean 节点，它拆成 `Sum` + `Broadcast(Scalar)`
// + `TrueDiv`，那个 `Scalar` 就是外部输入。所以这条用例钉的是 mean 这一大类，而不是一个
// 边角形状。
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <vector>

#include "test/e2e/cpu/cases/load_sum0_mul_store/LoadSum0MulStoreCase.h"
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
  namespace cfg = load_sum0_mul_store;

  const auto input_shape = cfg::GetShapeConfig(cfg::kInputShapes, 0).shape;
  const auto input_exprs = ToExpressions(input_shape);
  assert(input_shape.size() == 2U);

  AscGraphBuilder builder(cfg::kKernelName);
  builder.Loops(input_shape)
      .Data("input0", 0, input_exprs, {}, af::DT_FLOAT)
      .Load("load0", "input0", input_exprs, {})
      .Sum("sum0", "load0", {0U});

  auto graph = builder.Scalar("two", "2.0", af::DT_FLOAT)
                   .Mul("mul0", "sum0", "two")
                   .Store("store0", "mul0")
                   .Output("output", "store0", 0, af::DT_FLOAT)
                   .Build();
  SetOutputView(graph, "output", cfg::GetShapeConfig(cfg::kOutputShapes, 0).shape);
  EXPECT_EQ(optimize::AscGraphInfoComplete::CompleteApiInfo(graph), af::SUCCESS);
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_sum0_mul_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::ReduceSum"), std::string::npos);
  // 切图之后乘法落在第二张图里，读的是 workspace。
  EXPECT_NE(kernel.find("Muls<float, 1>("), std::string::npos);
  // 这张图有两块 workspace，而 Plan 主函数读的是**每个 stage 自己那份** TilingData 上的
  // `workspace<ID>`；ATT 只往根上写偏移（schema 把 workspace 声明在 `"fusion"` 作用域），
  // 所以聚合入口必须逐个拷下去。少了这一步两块 workspace 会同用 0 号基址、互相覆盖，而且
  // 不报任何错——这条断言就是钉住那次拷贝。
  EXPECT_NE(kernel.find(".workspace1 = "), std::string::npos);
  EXPECT_NE(kernel.find(".workspace3 = "), std::string::npos);
}

}  // namespace mlir::afir
