/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 验证 mem-based FloorDivExtend 的 tmp buffer、floor_div.h 及 utils.h 前导。
// floor_div.h 使用 utils.h 定义的 MAX_REPEAT_TIME；仅匹配调用文本的 LIT 不能发现
// 缺少前导，必须编译内联后的 kernel。
// 使用 mem 侧注册支持的 f32；reg 侧无 tmp buffer 的包装由对应 LIT 覆盖。

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_floor_div_store/LoadFloorDivStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_floor_div_store::GetShapeConfig(load_floor_div_store::kOutputShapes, 0).shape;
  const auto input0_shape =
      ToExpressions(load_floor_div_store::GetShapeConfig(load_floor_div_store::kInputShapes, 0).shape);
  const auto input1_shape =
      ToExpressions(load_floor_div_store::GetShapeConfig(load_floor_div_store::kInputShapes, 1).shape);
  AscGraphBuilder builder(load_floor_div_store::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_FLOAT)
      .Data("input1", 1, input1_shape, {}, af::DT_FLOAT)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {});

  auto graph = builder.FloorDiv("floor_div", "load0", "load1")
                   .Store("store", "floor_div")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_floor_div_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 包装，不是标准调用：全局函数、没有 `AscendC::` 限定。
  EXPECT_NE(kernel.find("FloorDivExtend("), std::string::npos);
  // 扩展头与它的前导都被内联进来了。少了 `utils.h` 这份 kernel 编不过（MAX_REPEAT_TIME）。
  EXPECT_NE(kernel.find("__ASCENDC_API_FLOOR_DIV_H__"), std::string::npos);
  EXPECT_NE(kernel.find("MAX_REPEAT_TIME"), std::string::npos);
  EXPECT_NE(kernel.find("using namespace AscendC;"), std::string::npos);
}
}  // namespace mlir::afir
