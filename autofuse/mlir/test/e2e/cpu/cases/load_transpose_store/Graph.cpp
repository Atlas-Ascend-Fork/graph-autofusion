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

#include "test/e2e/cpu/cases/load_transpose_store/LoadTransposeStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"
namespace mlir::afir {
// mem 侧（`CaseMem.cpp`）与 reg 侧（`CaseRegBase.cpp`）两个实例共用这一张图——同一张图
// 两个家族各走各的下降，差异只在实现选择，图本身没有家族属性。
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_transpose_store;
  const auto input_config = config::GetShapeConfig(config::kInputShapes, 0);
  const auto input_shape = ToExpressions(input_config.shape);
  // `ShapeConfig::shape` 是 initializer_list（没有 operator[]），物化成 vector 再用。
  const std::vector<int64_t> output_dims(config::GetShapeConfig(config::kOutputShapes, 0).shape);
  // perm `[0, 2, 1]`：输出轴 `(z0, z2, z1)`。分块轴（`z0`）两侧都在最外、perm 落在内层
  // 轴。调度对这张图枚举**两类候选**：保留转置 op 的（mem 侧发 `ConfusionTranspose`、reg
  // 侧发 `TransposeExtend`）与把转置折进搬入布局的（reg 侧 `LoadToNddmaTemplate` 直转
  // ND-DMA）；胜出的 plan 按 tiling 在多 template 间派发，两类都可能出现——reg 侧的
  // kernel 文本断言（`CaseRegBase.cpp`）两类都钉。Store 按 Transpose 输出轴的行主序落
  // GM（`{s2*s1, s1, 1}`）。
  std::vector<af::Expression> store_strides = {af::Symbol(output_dims[1]) * af::Symbol(output_dims[2]),
                                               af::Symbol(output_dims[2]), af::sym::kSymbolOne};
  AscGraphBuilder builder(config::kKernelName);
  builder.Loops(input_config.shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0");
  auto graph =
      builder.Transpose("transpose", "load0", {0, 2, 1})
          .Store("store", "transpose",
                 {af::Symbol(output_dims[0]), af::Symbol(output_dims[1]), af::Symbol(output_dims[2])}, store_strides)
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
}  // namespace mlir::afir
