/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "test/e2e/cpu/cases/sub_transpose_abs/Case.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"
#include "gtest/gtest.h"

namespace mlir::afir {
// Legacy SubTransposeAbsFusedGraph: perm [1,0,2], default shape [32,64,512].
af::AscGraph BuildGraph() {
  namespace config = sub_transpose_abs;
  auto shape = config::GetShapeConfig(config::kInputShapes, 0).shape;
  auto dims = std::vector<int64_t>(shape);
  auto graph = af::testing::AscGraphBuilder(config::kKernelName)
                   .Loops(shape)
                   .Data("input0", 0, af::DT_FLOAT)
                   .Data("input1", 1, af::DT_FLOAT)
                   .Load("load0", "input0")
                   .Load("load1", "input1")
                   .Sub("sub", "load0", "load1")
                   .Transpose("transpose", "sub", {1, 0, 2})
                   .Abs("abs", "transpose")
                   .Store("store", "abs", {af::Symbol(dims[1]), af::Symbol(dims[0]), af::Symbol(dims[2])},
                          {af::Symbol(dims[0] * dims[2]), af::Symbol(dims[2]), af::sym::kSymbolOne})
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  // Output must describe the same permuted view that Store writes.
  auto &output = graph.FindNode("output")->outputs()[0]->attr;
  const auto &stored = graph.FindNode("store")->outputs()[0]->attr;
  output.axis = stored.axis;
  output.repeats = stored.repeats;
  output.strides = stored.strides;
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(sub_transpose_abs::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::Sub("), std::string::npos);
  EXPECT_NE(kernel.find("Abs"), std::string::npos);
}
}  // namespace mlir::afir
