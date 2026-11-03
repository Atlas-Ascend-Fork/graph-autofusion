/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> cast -> store CPU case (reg-base) -----*- C++ -*-===//
//
// reg-base 一侧，钉的是 **brace-init 数组那条发射路径真的编得过**。
//
// `CastExtend(dst, src, {count}, {1U}, {1U})` 是这条路上唯一一个用 brace-init 数组传参
// 的调用：三个数组收 `const uint32_t (&)[dim]`（**与比较族的 `uint16_t` 不是同一个**），
// `dim` 只能从 brace-init 的长度推出来，所以一个显式模板实参都不能写——写了就得把 `dim`
// 也写出来，而它排在 `InT` / `OutT` 中间。这套约束此前只有 lit 的文本比对管着。
//
// 另一件 mem 侧钉不到的：**reg 侧这份包装不要 tmp buffer**，而 mem 侧那份要（中间要转
// 两道的 dtype 对拿它装中间结果）。这是 `afir.cast` 两侧最大的差别，生成的 kernel 里
// 一个 `TBuf` 都不该有。
//
// f32 -> i32：这一对在 reg 侧的对表里（`ascir_builtin_ops_v2.cpp` 的 `kCastTypePairs`）。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_cast_reg_base_store/Case.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_cast_reg_base_store::GetShapeConfig(load_cast_reg_base_store::kOutputShapes, 0).shape;
  const auto input_shape =
      ToExpressions(load_cast_reg_base_store::GetShapeConfig(load_cast_reg_base_store::kInputShapes, 0).shape);
  AscGraphBuilder builder(load_cast_reg_base_store::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  auto graph = builder.Cast("cast", "load0", af::DT_INT32)
                   .Store("store", "cast")
                   .Output("output", "store", 0, af::DT_INT32)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_cast_reg_base_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // **brace-init 数组那条路第一次被真编一遍。** 三个数组实参收的是
  // `const uint32_t (&)[dim]`（注意与比较族的 `uint16_t` 不同），`dim` 只能从
  // brace-init 的长度推——所以这条调用一个显式模板实参都不能写，写了就得把 `dim` 也
  // 写出来，而它排在 `InT` / `OutT` 中间。照实际输出写死。
  EXPECT_NE(kernel.find("CastExtend("), std::string::npos);
  EXPECT_NE(kernel.find("}, {1U}, {1U});"), std::string::npos);
  // 内联进来的是 reg 那一份。两份 `cast.h` 的 guard 不撞名（mem 是
  // `__ASCENDC_API_CAST_H__`，reg 那份只有一个裸的 `CAST_H`），所以这里比得了 guard；
  // 正面证据挑 reg 独有的 micro-API 入口。
  EXPECT_NE(kernel.find("CastExtendCommon"), std::string::npos);
  EXPECT_EQ(kernel.find("__ASCENDC_API_CAST_H__"), std::string::npos);
  // reg 侧不要 tmp buffer——mem 侧那份要（中间要转两道的 dtype 对拿它装中间结果）。
  // 这一对是 `afir.cast` 两侧最大的差别。
  EXPECT_EQ(kernel.find("AscendC::TBuf<"), std::string::npos);
  EXPECT_EQ(kernel.find("__ASCENDC_API_UTILS_H__"), std::string::npos);
}
}  // namespace mlir::afir
