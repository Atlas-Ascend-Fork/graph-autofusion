/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> exp -> sqrt -> store CPU case ---------*- C++ -*-===//
//
// D0 波次的端到端用例，两件事：一条链上串两个 buffer 形态的 elementwise 节点（第二个
// 节点读的必须是第一个的**结果值**，否则流水同步少一条边），以及 D0 走标准调用、不要
// tmp buffer、不引扩展头。
//
// 「同一份 AFIR 在两套 profile 下走同一个 implementation」这一半在这里验不了——CPU
// E2E 只按单个 soc 编译——那是 afir-unary-d0-both-targets.mlir 的事。
//
//===----------------------------------------------------------------------===//

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
  // **这两条原先是通查**（`Extend<` 与 `__ASCENDC_API_` 一个都不许有），2026-09-23 起不成立：
  // Load / Store 一律发 `DataCopyPadExtend`，`datacopy.h` 因此恒被内联进来。改成数个数，
  // 强度不降——多出任何一个计算侧的 wrapper 或扩展头，这两条照样红。
  EXPECT_EQ(CountOccurrences(kernel, "Extend<"), 2U);  // load 一处、store 一处
  EXPECT_EQ(CountOccurrences(kernel, "#define __ASCENDC_API_"), 1U);
  EXPECT_NE(kernel.find("#define __ASCENDC_API_DATACOPY_H__"), std::string::npos);
  // D0 不要 tmp buffer。TBuf 只会因为 tmp buffer 出现——队列走的是 TQue。
  EXPECT_EQ(kernel.find("AscendC::TBuf<"), std::string::npos);
}
}  // namespace mlir::afir
