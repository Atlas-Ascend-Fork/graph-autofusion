/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <vector>

#include "test/e2e/cpu/cases/load_transpose_store_2d/Case.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"
namespace mlir::afir {
// Part 2 的最简整图目标：2D `(A,B)→(B,A)` 全对齐 transpose（f16 `(64,64)`，设计文档
// 批次 1/2 的形状）。既有的 3D 用例把分块轴留在最外、perm 落内层；这一格 **两根轴都进
// perm**（`[1, 0]`），是 load/store 合轴后段数最少（预期 ≤2 段）的形态。
// mem 与 reg 共用图和逐元素数值校验，分别检查各自的目标 API。
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_transpose_store_2d;
  const auto input_config = config::GetShapeConfig(config::kInputShapes, 0);
  const auto input_shape = ToExpressions(input_config.shape);
  const std::vector<int64_t> output_dims(config::GetShapeConfig(config::kOutputShapes, 0).shape);
  // `perm [1, 0]`：输出轴 `(z1, z0)`，输入 `(z0, z1)`——全转置。Store 按输出轴行主序落
  // GM（`{s1, 1}`）。
  std::vector<af::Expression> store_strides = {af::Symbol(output_dims[1]), af::sym::kSymbolOne};
  AscGraphBuilder builder(config::kKernelName);
  builder.Loops(input_config.shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT16).Load("load0", "input0");
  auto graph = builder.Transpose("transpose", "load0", {1, 0})
                   .Store("store", "transpose", {af::Symbol(output_dims[0]), af::Symbol(output_dims[1])}, store_strides)
                   .Output("output", "store", 0, af::DT_FLOAT16)
                   .Build();
  // Output must describe the same permuted view that Store writes.
  auto &output = graph.FindNode("output")->outputs()[0]->attr;
  const auto &stored = graph.FindNode("store")->outputs()[0]->attr;
  output.axis = stored.axis;
  output.repeats = stored.repeats;
  output.strides = stored.strides;
  return graph;
}
}  // namespace mlir::afir
