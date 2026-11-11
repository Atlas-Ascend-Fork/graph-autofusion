/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 验证 mem-based NegExtend 使用扩展头但不需要 tmp buffer。
// 生成 kernel 应包含 NegExtend 且不含 TBuf，避免将扩展 API 与 scratch 需求绑定。
// 另一支持表的头名选择由 afir-neg-api-family.mlir 覆盖。

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_neg_store/LoadNegStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_neg_store::GetShapeConfig(load_neg_store::kOutputShapes, 0).shape;
  const auto input_shape = ToExpressions(load_neg_store::GetShapeConfig(load_neg_store::kInputShapes, 0).shape);
  AscGraphBuilder builder(load_neg_store::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  auto graph = builder.Neg("neg", "load0").Store("store", "neg").Output("output", "store", 0, af::DT_FLOAT).Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_neg_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 包装是全局函数，元素类型是显式模板实参。
  EXPECT_NE(kernel.find("NegExtend<float>("), std::string::npos);
  // 头文件被整段内联进来了，而且是 mem-based 那张表的那一份：两张表的包装同名同签名，
  // 只有 include guard 分得开它们。
  EXPECT_NE(kernel.find("__ASCENDC_API_NEG_H__"), std::string::npos);
  EXPECT_EQ(kernel.find("__ASCENDC_API_REGBASE_NEG_H__"), std::string::npos);
  // 走了包装，却一块 tmp buffer 都不要。TBuf 只会因为 tmp buffer 出现——队列走的是 TQue。
  EXPECT_EQ(kernel.find("AscendC::TBuf<"), std::string::npos);
}
}  // namespace mlir::afir
