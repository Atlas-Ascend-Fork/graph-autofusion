/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> leaky_relu -> store CPU case ----------*- C++ -*-===//
//
// 第一个**带自有属性**的节点走完整条流水。斜率在 AscGraph 上是节点的 `ir_attr`
// （`negative_slope`），在 AFIR 上是 op 属性（`negative_slop`），在生成的调用里是第三个
// 实参——三种形态，中间两段各有一处新代码：`ScheduleToAfir` 读 `ir_attr`、`AfirBuilder`
// 挂成 op 属性。在此之前 bridge 只有 `Scalar` 节点的 `value` 走过这条路，而它变成的是
// 一个 SSA 常量、不是属性。
//
// 用例钉的是「那个数原样到得了 kernel」：把 bridge 那两处任何一处去掉，斜率就丢了，
// 负半轴的期望值立刻对不上。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_leaky_relu_store/Case.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_leaky_relu_store::GetShapeConfig(load_leaky_relu_store::kOutputShapes, 0).shape;
  const auto input_shape =
      ToExpressions(load_leaky_relu_store::GetShapeConfig(load_leaky_relu_store::kInputShapes, 0).shape);
  AscGraphBuilder builder(load_leaky_relu_store::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  auto graph = builder.LeakyRelu("leaky_relu", "load0", load_leaky_relu_store::kNegativeSlope)
                   .Store("store", "leaky_relu")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_leaky_relu_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // `<float, 1>` 里的 1 是 `isSetMask`，与 legacy 不带模板实参时用的默认值同义：API
  // 自己设 mask。取 `0` 表示调用方已经设好了，今天没有 pass 这么做——见 `Base.td` 的
  // `VecScalarL2Op`。**这个形状探得到那条**（2026-09-23 起）：一趟切出来的 count 超过
  // 一个 repeat（f32 是 64 个），取 `0` 时 API 按 `repeatTimes = 1` 只发一条指令，
  // 逐元素断言会红。形状还是 `[128]` 的那阵子它探不到，见清单 L-2。
  EXPECT_NE(kernel.find("AscendC::LeakyRelu<float, 1>("), std::string::npos);
  // 斜率作为一个**按元素类型建的**常量出现在 kernel 里（f32 这一份是 `(float)`）。
  // 丢了属性的话它不会退化成 0，而是根本建不出这个常量——bridge 那边会先报错，
  // 这里是第二道。
  EXPECT_NE(kernel.find("(float)1.25"), std::string::npos);
  // 两张供给表给的是同一个标准调用，没有扩展头也没有 tmp buffer。
  EXPECT_EQ(kernel.find("LeakyReluExtend"), std::string::npos);
}
}  // namespace mlir::afir
