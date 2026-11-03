/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> sigmoid -> store CPU case -------------*- C++ -*-===//
//
// **这份用例存在的理由是「`sigmoid.h` 那段头文本编得过」。**
//
// mem 侧发的是包装 `SigmoidExtend<float>(dst, src, tmp, count)`，头 `sigmoid.h` 连同它
// 声明的前导 `utils.h` 被**整段内联**进 kernel。而 `sigmoid.h` 正是当初把「扩展头不是
// 自足的翻译单元」这件事暴露出来的那个 op——它用了只有 `utils.h` 里才有的 `KernelUtils`
// 与 `MAX_REPEAT_TIME`，E2 那一轮就是在它身上发现前导头这条通路整条缺失的。
//
// 结果是：**整条链上最先撞上这个问题的头，自己反倒一直没有编译型用例**。lit 那两份只
// 比文本，前导头声明被删掉它们照样绿，失败要一路走到 device 编译器上，报在内联进来的
// 那段头文本的行号里。这是清单 L-13 里 mem 侧剩下的最后一个洞。
//
// f32：mem 侧的 `REG_ASC_IR(Sigmoid)` 只收 f16 / f32，E2E harness 的 soc 钉在
// Ascend910B1，走的就是 mem 这一侧。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_sigmoid_store/Case.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_sigmoid_store::GetShapeConfig(load_sigmoid_store::kOutputShapes, 0).shape;
  const auto input_shape = ToExpressions(load_sigmoid_store::GetShapeConfig(load_sigmoid_store::kInputShapes, 0).shape);
  AscGraphBuilder builder(load_sigmoid_store::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  auto graph =
      builder.Sigmoid("sigmoid", "load0").Store("store", "sigmoid").Output("output", "store", 0, af::DT_FLOAT).Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_sigmoid_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 裸名加显式模板实参，不带 `AscendC::` 限定——包装是全局函数。
  EXPECT_NE(kernel.find("SigmoidExtend<float>("), std::string::npos);
  // 业务头与**前导头**都被内联进来了。后一条是这份用例的要害：`sigmoid.h` 用了
  // `KernelUtils` / `MAX_REPEAT_TIME`，只有 `utils.h` 声明它们。把 ODS 上那句
  // `ApiExtensionHeader<["sigmoid.h"], ["utils.h"]>` 的前导去掉，这两条里的第二条先红，
  // 紧接着 device 编译也过不去。
  EXPECT_NE(kernel.find("__ASCENDC_API_SIGMOID_H__"), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_UTILS_H__"), std::string::npos);
  // 前导头要排在业务头**前面**——它们是当作文本内联的，彼此不 include。
  EXPECT_LT(kernel.find("__ASCENDC_API_UTILS_H__"), kernel.find("__ASCENDC_API_SIGMOID_H__"));
  // 包装要一块调用方给的 tmp buffer：四个实参里最后一个是 count。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
  EXPECT_EQ(CountCallArguments(kernel, "SigmoidExtend<float>("), 4U);
  // 走的是 mem 这一侧，reg-base 的头不该出现。
  EXPECT_EQ(kernel.find("sigmoid_reg_base.h"), std::string::npos);
}
}  // namespace mlir::afir
