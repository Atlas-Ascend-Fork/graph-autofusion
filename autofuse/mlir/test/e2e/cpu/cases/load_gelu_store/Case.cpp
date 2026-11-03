/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> gelu -> store CPU case ----------------*- C++ -*-===//
//
// `afir.gelu` 只有一个 implementation：两张供给表给的是同一个 `AscendC::Gelu`，也不引
// 扩展头。所以这份用例不分两侧——reg 侧那半由 afir-gelu-stock-api.mlir 的第二组 RUN 钉。
//
// 它要钉的是另一件事：**这个「同一个调用」是要 tmp buffer 的**。exp / ln / relu / sqrt
// 那一档也是两侧同调用，但都不要 tmp；照着它们把 gelu 注册成不要 tmp 的话，调度分的那块
// 会被 `checkTmpBuffer` 判成多给，整个 op 选不出实现。下面那条 TBuf 断言就是这条的现场。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_gelu_store/Case.h"
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
