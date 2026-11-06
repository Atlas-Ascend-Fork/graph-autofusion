/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "test/e2e/cpu/cases/transpose_generalization/TransposeGeneralizationCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"
#include <gtest/gtest.h>

namespace mlir::afir {
af::AscGraph BuildGraph() {
  namespace cfg = transpose_generalization;
  const auto input = cfg::GetShapeConfig(cfg::kInputShapes, 0).shape;
  const auto output = cfg::GetShapeConfig(cfg::kOutputShapes, 0).shape;
  af::testing::AscGraphBuilder builder(cfg::kKernelName);
#if AF_TRANSPOSE_CASE == 4
  builder.Loops({8, 37, 23});
#else
  builder.Loops(input);
#endif
#if AF_TRANSPOSE_CASE == 1
  builder.Data("input0", 0, ToExpressions(input), {}, af::DT_FLOAT16)
      .Load("load0", "input0")
      .Cast("cast", "load0", af::DT_FLOAT)
      .Transpose("transpose", "cast", {1, 0});
#else
  builder.Data("input0", 0, ToExpressions(input), {}, af::DT_FLOAT).Load("load0", "input0", ToExpressions(input), {});
#if AF_TRANSPOSE_CASE == 3
  builder.Transpose("transpose0", "load0", {0, 2, 1, 3}).Transpose("transpose", "transpose0", {0, 3, 1, 2});
#elif AF_TRANSPOSE_CASE == 4
  builder.Broadcast("broadcast", "load0", {0}).Transpose("transpose", "broadcast", {0, 2, 1});
#else
  builder.Transpose("transpose", "load0", {0, 2, 1});
#endif
#endif
  std::string last = "transpose";
#if AF_TRANSPOSE_CASE == 2
  builder.Data("input1", 1, ToExpressions(output), {}, af::DT_FLOAT)
      .Load("load1", "input1", ToExpressions(output), {})
      .Add("add", "transpose", "load1");
  last = "add";
#elif AF_TRANSPOSE_CASE == 5
  builder.Sum("sum", "transpose", {2});
  last = "sum";
#endif
  auto graph = builder.Store("store", last, ToExpressions(output), {}).Output("output", "store").Build();
#if AF_TRANSPOSE_CASE == 3
  // The builder interprets a perm against loop axes; the second perm is instead
  // relative to the first result. Supply the composed logical axis identity.
  graph.FindNode("transpose")->outputs()[0]->attr.axis = {0, 3, 2, 1};
  graph.FindNode("store")->outputs()[0]->attr.axis = {0, 3, 2, 1};
#elif AF_TRANSPOSE_CASE == 2
  for (const auto *name : {"input1", "load1"}) {
    graph.FindNode(name)->outputs()[0]->attr.axis = {0, 2, 1};
  }
#endif
  auto &view = graph.FindNode("output")->outputs()[0]->attr;
  const auto &stored = graph.FindNode("store")->outputs()[0]->attr;
  view.axis = stored.axis;
  view.repeats = stored.repeats;
  view.strides = stored.strides;
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(transpose_generalization::kKernelName) + "_subgraph0_plan"), std::string::npos);
  EXPECT_EQ(kernel.find("#include \"\""), std::string::npos);
}
}  // namespace mlir::afir
