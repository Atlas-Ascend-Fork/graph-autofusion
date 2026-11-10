/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 使用 i32 输入和 f32 输出，覆盖 TrueDivExtend 区别于普通 Div 的转换分支。
// 该分支先按 CAST_RINT 将两个输入转换为 f32，再相除；f32 输入只会转发 AscendC::Div，
// 不能验证 TrueDiv 独有的类型与 scratch 行为。

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_truediv_store/LoadTruedivStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_truediv_store::GetShapeConfig(load_truediv_store::kOutputShapes, 0).shape;
  const auto input0_shape =
      ToExpressions(load_truediv_store::GetShapeConfig(load_truediv_store::kInputShapes, 0).shape);
  const auto input1_shape =
      ToExpressions(load_truediv_store::GetShapeConfig(load_truediv_store::kInputShapes, 1).shape);
  AscGraphBuilder builder(load_truediv_store::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_INT32)
      .Data("input1", 1, input1_shape, {}, af::DT_INT32)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {});

  // 输出是 `DT_FLOAT`，两个输入是 `DT_INT32`——`kTrueDivTypePairs` 的 i32 那一行。
  // builder 里的 `TrueDiv` 就是按这条对表把继承来的 dtype 改掉的。
  auto graph = builder.TrueDiv("truediv", "load0", "load1")
                   .Store("store", "truediv")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_truediv_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 裸名，不带 `AscendC::` 限定——`true_div.h` 里的包装是全局函数。
  EXPECT_NE(kernel.find("TrueDivExtend("), std::string::npos);
  // 两个业务头都被内联进来了。
  EXPECT_NE(kernel.find("__ASCENDC_API_TRUE_DIV_H__"), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_SCALAR_DIV_H__"), std::string::npos);
  // 前导头也被内联进来了。 把 td 上那个 `["utils.h"]` 删掉，lit 一条都不红，
  // 这里先编不过、再红在这一行。
  EXPECT_NE(kernel.find("__ASCENDC_API_UTILS_H__"), std::string::npos);
  // 包装要一块 tmp buffer，i32 这一支真的用它（切两段装两份 cast 结果）。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
  // 进来的是 mem 那一份：reg 侧生成的是 `DivExtend`（`div_reg_base.h`），与 `afir.div`
  // 的 reg 侧同一个调用。不能按 `DivExtend(` 去找——`TrueDivExtend(` 含着它。
  EXPECT_EQ(kernel.find("__ASCENDC_REGBASE_API_DIV_H__"), std::string::npos);
}
}  // namespace mlir::afir
