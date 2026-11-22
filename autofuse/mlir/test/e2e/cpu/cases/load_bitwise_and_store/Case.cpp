/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 验证 int16 输入、输出贯穿 tiling、GM 搬运和 kernel 签名的路径。
// 默认 CPU 目标走 mem-based BitwiseAndExtend，包含 bitwise_and.h 并传入 tmp buffer；
// 包装虽不使用 scratch，签名仍要求该参数。reg 侧 stock API 由对应 LIT 覆盖。
// int16 直接转发 AscendC::And，不经过 ReinterpretCast；ui8 转换路径另由 LIT 覆盖。

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_bitwise_and_store/LoadBitwiseAndStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_bitwise_and_store::GetShapeConfig(load_bitwise_and_store::kOutputShapes, 0).shape;
  const auto input0_shape =
      ToExpressions(load_bitwise_and_store::GetShapeConfig(load_bitwise_and_store::kInputShapes, 0).shape);
  const auto input1_shape =
      ToExpressions(load_bitwise_and_store::GetShapeConfig(load_bitwise_and_store::kInputShapes, 1).shape);
  AscGraphBuilder builder(load_bitwise_and_store::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_INT16)
      .Data("input1", 1, input1_shape, {}, af::DT_INT16)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {});

  auto graph = builder.BitwiseAnd("bitwise_and", "load0", "load1")
                   .Store("store", "bitwise_and")
                   .Output("output", "store", 0, af::DT_INT16)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_bitwise_and_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 包装，不是标准调用：全局函数、没有 `AscendC::` 限定。
  EXPECT_NE(kernel.find("BitwiseAndExtend("), std::string::npos);
  EXPECT_EQ(kernel.find("AscendC::BitwiseAnd("), std::string::npos);
  // 扩展头的文本被内联进来了，连同它的前导 `using namespace AscendC;`。
  EXPECT_NE(kernel.find("using namespace AscendC;"), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_BITWISE_AND_H__"), std::string::npos);
}
}  // namespace mlir::afir
