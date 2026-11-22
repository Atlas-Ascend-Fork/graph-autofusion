/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> sign -> store CPU case ----------------*- C++ -*-===//
//
// **第一个声明两个业务头的 op 的端到端用例，也是前置 A（`00cda3a0` 把扩展头改成列表）
// 唯一的用处兑现的地方。** 包装按 dtype 分流：f16 / f32 直接转发 `AscendC::Sign`,
// 其余先 `Cast`。所以 `cast.h` 与 `sign.h` 两个都要内联进 kernel——少声明一个，少的那份
// 的函数在 kernel 里没有定义，这份用例在编译期就红，不用等断言。
//
// harness 把 soc 钉在 Ascend910B1，走 mem 侧：`SignExtend<T>(dst, src, tmp_buf, size)`,
// 扩展头 `cast.h` + `sign.h`，要 tmp buffer。reg 侧同名同签名，只有头不同，由
// afir-sign-reg-base-extension-api.mlir 钉。
//
// 取 f32:mem 侧收 f16 / f32 / i32,ui8 只有 reg 侧收。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_sign_store/LoadSignStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_sign_store::GetShapeConfig(load_sign_store::kOutputShapes, 0).shape;
  const auto input_shape = ToExpressions(load_sign_store::GetShapeConfig(load_sign_store::kInputShapes, 0).shape);
  AscGraphBuilder builder(load_sign_store::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  auto graph = builder.Sign("sign", "load0").Store("store", "sign").Output("output", "store", 0, af::DT_FLOAT).Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_sign_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 包装是全局函数，元素类型是显式模板实参。
  EXPECT_NE(kernel.find("SignExtend<float>("), std::string::npos);
  // 两个头都被内联进来了。 只声明 `sign.h` 的话这一条会红——而且在它之前，kernel
  // 先编不过。
  EXPECT_NE(kernel.find("__ASCENDC_API_SIGN_H__"), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_CAST_H__"), std::string::npos);
  // 进来的是 mem 那一份，不是 reg 那一份。两份 sign.h 的 include guard 是同一个
  // （都叫 `__ASCENDC_API_SIGN_H__`），所以这里不能照 neg 那样比 guard，只能挑一处
  // 只有 reg 那份才有的文本：它的直通分支多认一个 int64。
  EXPECT_EQ(kernel.find("std::is_same_v<T, int64_t>"), std::string::npos);
  // 包装要 tmp buffer。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
}
}  // namespace mlir::afir
