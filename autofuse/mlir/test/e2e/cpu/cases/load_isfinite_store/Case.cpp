/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 谓词族共用 lowering，但每个 op 内联不同扩展头。此用例编译 isfinite.h，
// 检查该头自身的语法、前导依赖及与其他头合并后的重名问题；isnan.h 的用例不能替代。

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_isfinite_store/LoadIsfiniteStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_isfinite_store::GetShapeConfig(load_isfinite_store::kOutputShapes, 0).shape;
  const auto input_shape =
      ToExpressions(load_isfinite_store::GetShapeConfig(load_isfinite_store::kInputShapes, 0).shape);
  AscGraphBuilder builder(load_isfinite_store::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  auto graph = builder.IsFinite("isfinite", "load0")
                   .Store("store", "isfinite")
                   .Output("output", "store", 0, af::DT_UINT8)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_isfinite_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 模板实参是源的 float，不是目的地的 uint8。
  EXPECT_NE(kernel.find("IsFiniteExtend<float>("), std::string::npos);
  EXPECT_EQ(kernel.find("IsFiniteExtend<uint8_t>("), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_ISFINITE_H__"), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_UTILS_H__"), std::string::npos);
  EXPECT_EQ(kernel.find("ReinterpretCast<bool>"), std::string::npos);
}
}  // namespace mlir::afir
