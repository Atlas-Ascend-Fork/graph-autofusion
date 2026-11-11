/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 验证两类 API 支持表共用的 AscendC::Gelu 调用需要 tmp buffer，但不引入扩展头。
// 不能沿用 Exp / Ln / Relu / Sqrt 的无 scratch 注册，否则调度分配的缓冲会使
// checkTmpBuffer 拒绝该实现。TBuf 断言检查 scratch 已被实际接入。
// 另一目标的同调用映射由 afir-gelu-stock-api.mlir 覆盖。

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_gelu_store/LoadGeluStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_gelu_store::GetShapeConfig(load_gelu_store::kOutputShapes, 0).shape;
  const auto input_shape = ToExpressions(load_gelu_store::GetShapeConfig(load_gelu_store::kInputShapes, 0).shape);
  AscGraphBuilder builder(load_gelu_store::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  auto graph = builder.Gelu("gelu", "load0").Store("store", "gelu").Output("output", "store", 0, af::DT_FLOAT).Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_gelu_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 标准调用，带 `AscendC::` 限定——没有包装，也就没有扩展头被内联进来。
  EXPECT_NE(kernel.find("AscendC::Gelu("), std::string::npos);
  // 调度给了一块 tmp buffer，而且这条调用收了它：四个实参里最后一个是 count。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
  EXPECT_EQ(CountCallArguments(kernel, "AscendC::Gelu("), 4U);
}
}  // namespace mlir::afir
