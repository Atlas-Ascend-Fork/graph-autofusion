/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load x2 -> remainder -> store CPU case --------*- C++ -*-===//
//
// **第一个 ODS 上原本没有、本切片现建的 op 的端到端用例。** 在此之前接通的每一个 op，
// `afir.*` 的定义都已经在了，改的是「换个 base 类」；`afir.remainder` 是从零写的，所以
// 从 pyasc 的 `Remainder` 节点一路到 kernel 里那行调用，中间每一段都是第一次走。
//
// harness 把 soc 钉在 Ascend910B1（cpu/CMakeLists.txt），走 V1 也就是 mem-based 一侧：
// `RemainderExtend(dst, x1, x2, tmp_buf, size)`，扩展头 `remainder.h`，要 tmp buffer。
// reg 侧那份同名、同样收 tmp，只有头不同，在这里到不了，由
// afir-remainder-reg-base-extension-api.mlir 钉。
//
// 取 f32：mem 侧 legacy 的 `REG_ASC_IR(Remainder)` 收 f32 / i32，f16 只有 reg 侧收。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_remainder_store/LoadRemainderStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_remainder_store::GetShapeConfig(load_remainder_store::kOutputShapes, 0).shape;
  const auto input0_shape =
      ToExpressions(load_remainder_store::GetShapeConfig(load_remainder_store::kInputShapes, 0).shape);
  const auto input1_shape =
      ToExpressions(load_remainder_store::GetShapeConfig(load_remainder_store::kInputShapes, 1).shape);
  AscGraphBuilder builder(load_remainder_store::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_FLOAT)
      .Data("input1", 1, input1_shape, {}, af::DT_FLOAT)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {});

  auto graph = builder.Remainder("remainder", "load0", "load1")
                   .Store("store", "remainder")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_remainder_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 包装，不是标准调用：全局函数、没有 `AscendC::` 限定。
  EXPECT_NE(kernel.find("RemainderExtend("), std::string::npos);
  // 内联进来的是 mem 侧那份头，不是 reg 侧那份。
  EXPECT_NE(kernel.find("__ASCENDC_API_REMAINDER_H__"), std::string::npos);
  EXPECT_EQ(kernel.find("__ASCENDC_API_REGBASE_REMAINDER_H"), std::string::npos);
}
}  // namespace mlir::afir
