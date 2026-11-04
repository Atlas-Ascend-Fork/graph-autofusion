/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> isfinite -> store CPU case ------------*- C++ -*-===//
//
// 谓词族的三个 op 共用一条 lowering 路径，但**内联进 kernel 的是三份互不相同的头文
// 本**。`load_isnan_store` 编的是 `isnan.h`，`isfinite.h` 里的代码在它那里一行都没被
// 编译器看过——头自身的语法、它依赖而前导没给够的名字、它和别的扩展头拼在同一个 TU
// 里时的重名，都只有把这一份也编出来才会暴露。
//
//===----------------------------------------------------------------------===//

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
