/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> pow(scalar, tile) -> store (reg-base) --*- C++ -*-===//
//
// reg-base 一侧的规则三（位置原样保留）：`y = 2.0 ^ x`，标量在**左**。与
// `load_scalar_pow_store` 同一张图、同一份数值断言，差别只有 soc。
//
// 两侧用的是**同一个 AscendC def**（`AscendC_PowRegBaseOp` 与 `AscendC_PowOp` 各自那份
// 都是同名重载），发出来的调用文本也逐字相同——**分得开它们的只有内联进来的那份头**。
// 而两份 `pow.h` 的 include guard 还撞名（都是 `__ASCENDC_API_POW_H__`），所以下面钉的
// 是 reg 那份独有的 `pow_config` / `PowerAlgo`：`load_pow_store` 与
// `load_scalar_pow_store` 反过来钉它不出现。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_scalar_pow_reg_base_store/LoadScalarPowRegBaseStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace cfg = load_scalar_pow_reg_base_store;

  const auto output_shape = cfg::GetShapeConfig(cfg::kOutputShapes, 0).shape;
  const auto input_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 0).shape);
  AscGraphBuilder builder(cfg::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  // 标量是底数、在前；tile 是指数、在后。输出属性跟第二个输入（follow_index = 1）。
  auto graph = builder.Scalar("two", "2.0", af::DT_FLOAT)
                   .Op<af::ascir_op::Pow>("pow", {"two", "load0"}, 1U)
                   .Store("store", "pow")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  namespace cfg = load_scalar_pow_reg_base_store;
  EXPECT_NE(kernel.find(std::string(cfg::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // **字面量写全**：emitter 打的是定长科学计数。
  EXPECT_NE(kernel.find("constexpr float c2_f32 = (float)2.000000000e+00;"), std::string::npos);
  // **标量在第二个实参位**，也就是 AFIR 上的原次序。从最后一处 `Pow(` 起算：`pow_reg_base.h`
  // 整段内联进 kernel，前面那些是各重载的定义，调用点排在它们之后（内部转发的是
  // `Power(`，不含 `Pow(`，数不进来）。
  const size_t call = kernel.rfind("Pow(");
  ASSERT_NE(call, std::string::npos);
  const size_t firstComma = kernel.find(',', call);
  ASSERT_NE(firstComma, std::string::npos);
  EXPECT_EQ(kernel.compare(firstComma + 1, std::string(" c2_f32,").size(), " c2_f32,"), 0)
      << kernel.substr(call, kernel.find(';', call) - call);
  // 内联进来的是 **reg** 那一份。两份头的 include guard 撞名，所以钉 reg 独有的那两个
  // 名字——`load_scalar_pow_store` 钉的是它们不出现。
  EXPECT_NE(kernel.find("__ASCENDC_API_POW_H__"), std::string::npos);
  EXPECT_NE(kernel.find("PowerAlgo::DOUBLE_FLOAT_TECH"), std::string::npos);
  // 两侧都要 tmp buffer，前导 `utils.h` 两侧也都要。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_UTILS_H__"), std::string::npos);
}
}  // namespace mlir::afir
