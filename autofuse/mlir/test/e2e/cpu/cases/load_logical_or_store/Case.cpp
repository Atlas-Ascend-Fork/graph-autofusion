/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load x2 -> logical_or -> store CPU case -------*- C++ -*-===//
//
// 与 load_logical_and_store 同形，两处不同，都是有意的：
//
//   - 建的是 `LogicalOr`。两个 op 只差名字，而「哪个 AFIR op 配哪个包装」是注册处一张
//     手写的对照表——只验 `logical_and` 的话，把两行的 `TargetOp` 互换，整套还是全绿。
//   - 输入取 `DT_INT32` 而不是 `DT_FLOAT`。`logical.h` 里 int32 走的是
//     `Int64LogicalCommon`（先 `Cast` 到 float 再到 half，tmp buffer 要切两段），
//     f32 走的是 `OtherTypeLogicalCommon`——两条不同的分支。两份用例各走一条，
//     tmp buffer 分得不够大的话这一条先炸。
//
// 其余（`utils.h` 前导、输入与输出不同 dtype）与那份一样，理由见那边的文件头。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_logical_or_store/LoadLogicalOrStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_logical_or_store::GetShapeConfig(load_logical_or_store::kOutputShapes, 0).shape;
  const auto input0_shape =
      ToExpressions(load_logical_or_store::GetShapeConfig(load_logical_or_store::kInputShapes, 0).shape);
  const auto input1_shape =
      ToExpressions(load_logical_or_store::GetShapeConfig(load_logical_or_store::kInputShapes, 1).shape);
  AscGraphBuilder builder(load_logical_or_store::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_INT32)
      .Data("input1", 1, input1_shape, {}, af::DT_INT32)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {});

  auto graph = builder.LogicalOr("logical_or", "load0", "load1")
                   .Store("store", "logical_or")
                   .Output("output", "store", 0, af::DT_UINT8)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_logical_or_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  EXPECT_NE(kernel.find("LogicalOr("), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_LOGICAL_H__"), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_UTILS_H__"), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
  EXPECT_EQ(kernel.find("__ASCENDC_API_REGBASE_LOGICAL_H__"), std::string::npos);
  EXPECT_EQ(kernel.find("LogicalOrExtend("), std::string::npos);
}
}  // namespace mlir::afir
