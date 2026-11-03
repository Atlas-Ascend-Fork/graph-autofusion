/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> tanh -> store CPU case ----------------*- C++ -*-===//
//
// **这份用例存在的理由是「`constexpr bool` 当模板实参真的编得过」，不是「tanh 算得对」。**
//
// `afir.tanh` 的 mem 侧发的是 `AscendC::Tanh<float, c0_i1>(dst, src, tmp, count)`，
// 其中 `c0_i1` 是 lowering 物化的 `arith.constant false` 被 emit 成的
// `constexpr bool c0_i1 = false;`。整条链上只有这里会真的把那份 kernel 编一遍——
// lit 那两份只比文本，`constexpr bool` 填不进模板实参位的话它们照样绿。
//
// legacy 的 `UnaryApiTmpCall` 发的是不带模板实参的 `Tanh(y[off], x[off], tmp, size);`，
// 靠 AscendC 的默认值 `isReuseSource = false`。我们打出来的与它语义相同、文本不同，
// 所以下面按 `AscendC::Tanh<` 去找，别按 legacy 那个形状找。
//
// f32:mem 侧的 `REG_ASC_IR(Tanh)` 只收 f16 / f32，E2E harness 的 soc 钉在 Ascend910B1,
// 走的就是 mem 这一侧。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_tanh_store/Case.h"
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
  // 这里的标记**不以 `(` 收尾**——常量名由 emitter 自己编，写不死。`CountCallArguments`
  // 是从标记末尾往后扫到第一个 `(` 才开始记深度的，中间只剩 `c0_i1>`，不影响计数。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
  EXPECT_EQ(CountCallArguments(kernel, "AscendC::Tanh<float, "), 4U);
}
}  // namespace mlir::afir
