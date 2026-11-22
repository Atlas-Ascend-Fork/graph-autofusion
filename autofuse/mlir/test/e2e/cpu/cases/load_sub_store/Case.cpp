/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load x2 -> sub -> store CPU case --------------*- C++ -*-===//
//
// 这份用例存在的理由不是「减法算得对」，是钉住 `TmpBufferUse::Unused`。
//
// harness 把 soc 钉在 Ascend910B1，走 mem-based 一侧。那一侧生成的是标准的
// `AscendC::Sub(dst, x1, x2, size)`——四个实参，没有 tmp 的位置——而 legacy 的
// `SubAscIrCodegenImpl::CalcTmpBufSize` 照样给这个节点分了一块（`CalcBinaryApiTmpSize`
// 的 tensor 一支，32 字节）。那块是给标量形态的 `Subs` 用的，tensor-tensor 这条路只是
// 收下它。
//
// 所以下面两条断言是成对的，缺一条就说明不了问题：kernel 里既有 TBuf（调度真的分了
// 一块），那行调用又只有四个实参（这条路真的没往下传）。
//
// reg 侧的 `SubExtend` 在这里到不了，由 afir-sub-div-reg-base-extension-api.mlir 钉。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_sub_store/LoadSubStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_sub_store::GetShapeConfig(load_sub_store::kOutputShapes, 0).shape;
  const auto input0_shape = ToExpressions(load_sub_store::GetShapeConfig(load_sub_store::kInputShapes, 0).shape);
  const auto input1_shape = ToExpressions(load_sub_store::GetShapeConfig(load_sub_store::kInputShapes, 1).shape);
  AscGraphBuilder builder(load_sub_store::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_FLOAT)
      .Data("input1", 1, input1_shape, {}, af::DT_FLOAT)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {});

  auto graph =
      builder.Sub("sub", "load0", "load1").Store("store", "sub").Output("output", "store", 0, af::DT_FLOAT).Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_sub_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 标准调用，带 `AscendC::` 限定——与走包装的那些正好相反。
  EXPECT_NE(kernel.find("AscendC::Sub("), std::string::npos);
  // 调度分了一块 tmp buffer：TBuf 只会因为 tmp buffer 出现，队列走的是 TQue。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
  // 而这条调用没有收它。注册成 `TmpBufferUse::Passed` 的话这里就是 5。
  EXPECT_EQ(CountCallArguments(kernel, "AscendC::Sub("), 4U);
  // reg 侧那份包装的头没有被内联进来。
  EXPECT_EQ(kernel.find("__ASCENDC_REGBASE_API_SUB_H__"), std::string::npos);
}
}  // namespace mlir::afir
