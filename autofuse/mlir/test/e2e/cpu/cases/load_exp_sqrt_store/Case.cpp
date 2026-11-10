/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 验证两个 buffer 形态的标准 elementwise 调用串联：Sqrt 必须读取 Exp 的结果，
// 以保留流水同步依赖；两者不需要 tmp buffer 或扩展头。
// CPU 用例仅编译一个目标，跨 profile 复用同一实现由 afir-unary-d0-both-targets.mlir 覆盖。

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_exp_sqrt_store/LoadExpSqrtStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_exp_sqrt_store::GetShapeConfig(load_exp_sqrt_store::kOutputShapes, 0).shape;
  const auto input_shape =
      ToExpressions(load_exp_sqrt_store::GetShapeConfig(load_exp_sqrt_store::kInputShapes, 0).shape);
  AscGraphBuilder builder(load_exp_sqrt_store::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  auto graph = builder.Exp("exp", "load0")
                   .Sqrt("sqrt", "exp")
                   .Store("store", "sqrt")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_exp_sqrt_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 标准调用，两套家族同一个。计算侧没有扩展 wrapper，也没有它带进来的扩展头。
  EXPECT_NE(kernel.find("AscendC::Exp("), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::Sqrt("), std::string::npos);
  // 这两条原先是通查（`Extend<` 与 `__ASCENDC_API_` 一个都不许有），2026-09-23 起不成立：
  // Load / Store 一律生成 `DataCopyPadExtend`，`datacopy.h` 因此恒被内联进来。改成数个数，
  // 强度不降——多出任何一个计算侧的 wrapper 或扩展头，这两条照样红。
  EXPECT_EQ(CountOccurrences(kernel, "Extend<"), 2U);  // load 一处、store 一处
  EXPECT_EQ(CountOccurrences(kernel, "#define __ASCENDC_API_"), 1U);
  EXPECT_NE(kernel.find("#define __ASCENDC_API_DATACOPY_H__"), std::string::npos);
  // D0 不要 tmp buffer。TBuf 只会因为 tmp buffer 出现——队列走的是 TQue。
  EXPECT_EQ(kernel.find("AscendC::TBuf<"), std::string::npos);
}
}  // namespace mlir::afir
