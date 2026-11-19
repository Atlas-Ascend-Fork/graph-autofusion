/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// dtype 第二波：ui64 的 sub。`SubExtend` 包装的 reg 行随第二波收进 ui64，
// 8 字节元素是这条路最容易在尾块对齐上出错的位宽。数据贴近 UINT64_MAX 并让
// 被减数小于减数：无符号回绕同样是定义行为。

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_sub_ui64_reg_base_store/LoadSubUi64RegBaseStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_sub_ui64_reg_base_store;

  const auto output_shape = config::GetShapeConfig(config::kOutputShapes, 0).shape;
  const auto input0_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 0).shape);
  const auto input1_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 1).shape);
  AscGraphBuilder builder(config::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_UINT64)
      .Data("input1", 1, input1_shape, {}, af::DT_UINT64)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {});

  auto graph =
      builder.Sub("sub", "load0", "load1").Store("store", "sub").Output("output", "store", 0, af::DT_UINT64).Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_sub_ui64_reg_base_store::kKernelName) + "_subgraph0_plan0"),
            std::string::npos);
  // reg 侧走 sub_reg_base.h 的包装 SubExtend。e2e 的 kernel 文本会把包装源码内联进来，
  // 所以 `SubExtend(` 出现两次：一次是内联的函数定义、一次是调用点；包装体里转发给
  // `AscendC::Sub(` 是预期，不能当 mem 侧调用来断言不存在。
  EXPECT_EQ(CountOccurrences(kernel, "SubExtend("), 2U);
}
}  // namespace mlir::afir
