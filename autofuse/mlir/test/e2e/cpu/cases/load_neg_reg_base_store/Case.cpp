/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> neg -> store CPU case (reg-base) ------*- C++ -*-===//
//
// **这是第一个跑在 reg-base 一侧的 CPU E2E。**
//
// 在它之前，整套 CPU E2E 的 soc 钉死在 Ascend910B1，也就是 mem-based 一侧；15 个
// `*_reg_base.h` 的内联文本**一行都没有被任何编译器看过**，前导头要不要声明全靠读代码
// 判断。这正是 E2 那一轮吃过两次亏的形状：include 照发、key 照解析、文本照内联，最后
// 炸在 device 编译器上，报的是内联进来那段头文本的行号。清单里这一条是 L-13。
//
// 挑 `neg` 打头阵，是因为它是 reg 侧最短的一条路：包装 `NegExtend` 只是转发，不要
// tmp buffer，也不声明前导头。它验的是**通路本身**——换 soc 之后 optimizer / ATT 选
// v2 的 impl、MLIR 选 reg 侧的 implementation、`neg_reg_base.h` 内联进来并编得过。
// 前导头、micro-API、brace-init 数组那几种形状由同一批里的其他用例各验一条。
//
// 与 `load_neg_store`（mem 侧）是同一张图、同一份数值断言，**差别只有 soc**：
// 两份都绿才说明这个 op 两侧都通。`afir-neg-api-family.mlir` 比的是文本，这一对比的是
// 「编得过、算得对」。
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_neg_reg_base_store/LoadNegRegBaseStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_neg_reg_base_store::GetShapeConfig(load_neg_reg_base_store::kOutputShapes, 0).shape;
  const auto input_shape =
      ToExpressions(load_neg_reg_base_store::GetShapeConfig(load_neg_reg_base_store::kInputShapes, 0).shape);
  AscGraphBuilder builder(load_neg_reg_base_store::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  auto graph = builder.Neg("neg", "load0").Store("store", "neg").Output("output", "store", 0, af::DT_FLOAT).Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_neg_reg_base_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 包装是全局函数，元素类型是显式模板实参。两张表的 `NegExtend` 同名同签名，所以这一条
  // 两侧都成立——**分得开它们的只有 include guard**，也就是下面那两条。
  EXPECT_NE(kernel.find("NegExtend<float>("), std::string::npos);
  // 内联进来的必须是 **reg-base** 那一份。这两条反过来就是 `load_neg_store` 里写的那两条；
  // 换 soc 只改了第 1 处（`CodegenOptions::soc_version`）而漏了 NpuArch 的话，MLIR 这边
  // 照样选 reg 侧的 implementation，这一对仍然会过——真正会先炸的是调度那一侧算出来的
  // tmp buffer 与所选实现对不上。
  EXPECT_NE(kernel.find("__ASCENDC_API_REGBASE_NEG_H__"), std::string::npos);
  EXPECT_EQ(kernel.find("__ASCENDC_API_NEG_H__"), std::string::npos);
  // reg 侧这份包装同样不要 tmp buffer，也不声明前导头。
  EXPECT_EQ(kernel.find("AscendC::TBuf<"), std::string::npos);
  EXPECT_EQ(kernel.find("__ASCENDC_API_UTILS_H__"), std::string::npos);
}
}  // namespace mlir::afir
