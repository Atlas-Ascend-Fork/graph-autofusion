/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_unsigned_scalar_arithmetic_reg_base_store/UnsignedScalarArithmeticCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {

namespace config = load_unsigned_scalar_arithmetic_reg_base_store;

af::AscGraph BuildGraph() {
  constexpr auto dtype = AFIR_SCALAR_WIDTH == config::kUint16Width ? af::DT_UINT16 : af::DT_UINT32;
  const auto shape = config::GetShapeConfig(config::kOutputShapes, 0).shape;
  const auto inputShape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 0).shape);
  af::testing::AscGraphBuilder builder(config::kKernelName);
  builder.Loops(shape)
      .Data("input0", 0, inputShape, {}, dtype)
      .Load("load0", "input0", inputShape, {})
      .Scalar("value", std::to_string(config::kScalar), dtype);
#if defined(AFIR_SCALAR_Add)
  builder.Add("result", "load0", "value");
#elif defined(AFIR_SCALAR_Maximum)
  builder.Op<af::ascir_op::Maximum>("result", {"value", "load0"}, 1);
#else
  builder.Minimum("result", "load0", "value");
#endif
  auto graph = builder.Store("store", "result").Output("output", "store", 0, dtype).Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
#if defined(AFIR_SCALAR_Add)
  constexpr const char *wrapper = "AddsExtend(";
  constexpr const char *microApi = "AscendC::MicroAPI::Adds(";
  constexpr const char *standardCall = "AscendC::Adds(";
#elif defined(AFIR_SCALAR_Maximum)
  constexpr const char *wrapper = "MaxsExtend(";
  constexpr const char *microApi = "AscendC::MicroAPI::Maxs(";
  constexpr const char *standardCall = "AscendC::Maxs(";
#else
  constexpr const char *wrapper = "MinsExtend(";
  constexpr const char *microApi = "AscendC::MicroAPI::Mins(";
  constexpr const char *standardCall = "AscendC::Mins(";
#endif
  // 包装源码内联后 `AddsExtend(` 同时出现在定义与调用点。
  EXPECT_EQ(CountOccurrences(kernel, wrapper), 2U);
  // 包装的循环体调一次寄存器标量接口。连续 LocalTensor 接口不收 ui16 / ui32，选不中包装时
  // 才会出现，不该有。
  EXPECT_EQ(CountOccurrences(kernel, microApi), 1U);
  EXPECT_EQ(kernel.find(standardCall), std::string::npos);
  // 包装的循环形状，逐项计数：一层循环、归纳变量是 uint16_t（设备编译器对 `__VEC_SCOPE__`
  // 内的循环有这条要求）、每轮一次掩码更新、两个寄存器、一进一出。
  const std::string loopHead = "for (uint16_t i = 0; i < repeat_time; i++) {";
  const std::string regTensor = "AscendC::MicroAPI::RegTensor<T>";
  EXPECT_EQ(CountOccurrences(kernel, loopHead), 1U);
  EXPECT_EQ(CountOccurrences(kernel, "mask = AscendC::MicroAPI::UpdateMask<T>(count);"), 1U);
  EXPECT_EQ(CountOccurrences(kernel, regTensor), 2U);
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::MicroAPI::DataCopy("), 2U);
  // 寄存器声明在循环之前，掩码更新在循环之后（即循环体内）。
  const size_t loopAt = kernel.find(loopHead);
  EXPECT_LT(kernel.find(regTensor), loopAt);
  EXPECT_GT(kernel.find("mask = AscendC::MicroAPI::UpdateMask<T>(count);"), loopAt);
}

}  // namespace mlir::afir
