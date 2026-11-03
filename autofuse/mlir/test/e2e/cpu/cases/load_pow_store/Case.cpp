/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load x2 -> pow -> store CPU case --------------*- C++ -*-===//
//
// harness 把 soc 钉在 Ascend910B1（cpu/CMakeLists.txt），走 V1 也就是 mem-based 一侧：
// `Pow(dst, x1, x2, tmp_buf, size)`，扩展头 `pow.h`，要 tmp buffer。reg 侧那份同名、
// 同样收 tmp，只有头不同，在这里到不了，由 afir-pow-reg-base-extension-api.mlir 钉。
//
// 取 f32：mem 侧 legacy 的 `REG_ASC_IR(Pow)` 收 {i32, f16, f32}，ui8 / i16 / ui16
// 只有 reg 侧收。
//
// **这个用例真正拦的是前导头 `utils.h`。** 两份 `pow.h` 都用了 `KernelUtils::BlkSize`,
// reg 那份还用了 `KernelUtils::BlkAlign` 与 `IsIntegerWithinUint32Magnitude`——少声明
// 一个前导，IR 照建、`ascir-translate` 照打那行调用，两份 lit 全绿，只有这里真去编
// kernel 才炸。`where.h` / `sign.h` / `logical.h` / `floor_div.h` / `scalar_div.h`
// 之后这已经是第六次。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_pow_store/Case.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_pow_store::GetShapeConfig(load_pow_store::kOutputShapes, 0).shape;
  const auto input0_shape = ToExpressions(load_pow_store::GetShapeConfig(load_pow_store::kInputShapes, 0).shape);
  const auto input1_shape = ToExpressions(load_pow_store::GetShapeConfig(load_pow_store::kInputShapes, 1).shape);
  AscGraphBuilder builder(load_pow_store::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_FLOAT)
      .Data("input1", 1, input1_shape, {}, af::DT_FLOAT)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {});

  auto graph =
      builder.Pow("pow", "load0", "load1").Store("store", "pow").Output("output", "store", 0, af::DT_FLOAT).Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_pow_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 包装，不是标准调用：全局函数、没有 `AscendC::` 限定。标准调用叫 `Power`——
  // `AscendC::Pow` 这个名字不存在，所以这条断言找的就是裸名。
  EXPECT_NE(kernel.find("Pow("), std::string::npos);
  // 内联进来的是 mem 侧那份头，不是 reg 侧那份。两份的 include guard 撞名
  // （都是 `__ASCENDC_API_POW_H__`），所以 reg 那一侧改钉它独有的 `pow_config`。
  EXPECT_NE(kernel.find("__ASCENDC_API_POW_H__"), std::string::npos);
  EXPECT_EQ(kernel.find("PowerAlgo::DOUBLE_FLOAT_TECH"), std::string::npos);
  // 前导头：`pow.h` 用的 `KernelUtils::` 只有它声明。
  EXPECT_NE(kernel.find("__ASCENDC_API_UTILS_H__"), std::string::npos);
}
}  // namespace mlir::afir
