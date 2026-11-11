/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// mem 侧 Tanh 的 isReuseSource 由 arith.constant false 物化为 constexpr bool，
// 作为 AscendC::Tanh 的模板实参。本 E2E 实际编译该 kernel，补充仅匹配文本的 LIT，
// 验证布尔常量确实可用于该模板参数。

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_tanh_store/LoadTanhStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_tanh_store::GetShapeConfig(load_tanh_store::kOutputShapes, 0).shape;
  const auto input_shape = ToExpressions(load_tanh_store::GetShapeConfig(load_tanh_store::kInputShapes, 0).shape);
  AscGraphBuilder builder(load_tanh_store::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  auto graph = builder.Tanh("tanh", "load0").Store("store", "tanh").Output("output", "store", 0, af::DT_FLOAT).Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_tanh_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 那个常量真的被物化成了 `constexpr bool`，而且真的落在模板实参位上。两条都写死，
  // 少哪一条这份用例都会退化成「tanh 算得对」。
  EXPECT_NE(kernel.find("constexpr bool "), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::Tanh<float, "), std::string::npos);
  // mem 侧是标准调用，没有包装，所以扩展头一个都不该被内联进来。
  EXPECT_EQ(kernel.find("TanhExtend"), std::string::npos);
  // 调度给了一块 tmp buffer，而且这条调用收了它：四个实参里最后一个是 count。
  // 这里的标记不以 `(` 收尾——常量名由 emitter 自己编，写不死。`CountCallArguments`
  // 是从标记末尾往后扫到第一个 `(` 才开始记深度的，中间只剩 `c0_i1>`，不影响计数。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
  EXPECT_EQ(CountCallArguments(kernel, "AscendC::Tanh<float, "), 4U);
}
}  // namespace mlir::afir
