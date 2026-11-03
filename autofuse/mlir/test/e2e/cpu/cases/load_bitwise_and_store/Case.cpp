/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load x2 -> bitwise_and -> store CPU case ------*- C++ -*-===//
//
// **第一个非浮点的 E2E。** 到 LeakyRelu 为止，这条 CPU harness 上跑过的每一个 tensor
// 都是 f32（谓词族的 ui8 只在图中间出现，两端仍是 f32）。这里两个输入、目的地、GM 两端
// 全是 int16，所以「元素类型不是 f32」这件事在 tiling、搬运、kernel 签名上到底成不成立，
// 到这个用例才第一次被端到端验过。
//
// harness 把 soc 钉在 Ascend910B1（cpu/CMakeLists.txt），走 V1 也就是 mem-based 一侧：
// 包装 `BitwiseAndExtend(dst, x1, x2, tmp_buf, size)`，扩展头 `bitwise_and.h`，要一块
// tmp buffer——**包装其实没用它**，但形参在签名里。reg 侧的 `AscendC::BitwiseAnd` 在这里
// 到不了，由 afir-bitwise-and-stock-api.mlir 钉。
//
// 取 int16 而不是 uint8：包装对 16 位宽的 dtype 直接转发 `AscendC::And`，不走
// `ReinterpretCast<int16_t>` 那一支，失败时少一层间接。另一支由 lit 的 ui8 那份覆盖。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_bitwise_and_store/Case.h"
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
