/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load x2 -> logical_and(i64) -> store CPU case --*- C++ -*-===//
//
// 逻辑族 i64 的 e2e。mem 侧 `logical.h` 对 int64 专门有 `Int64LogicalCommon`——它把
// i64 先 cast 成 float 再折 half 计算，tmp 要拆成四份，与 f16/f32 那条直算的路不是
// 同一条实现，f32 那份 `load_logical_and_store` 盖不住，必须自己钉一遍数值。
// i64 建模之前这张图同样死在 bridge 的供给表，这里同时钉「bridge 收 i64 输入」与
// 「int64 专用包装分支算得对」两件事。
//
// harness 把 soc 钉在 Ascend910B1，走 mem 侧：`LogicalAnd(dst, x1, x2, tmp, size)`。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_logical_and_i64_store/LoadLogicalAndI64StoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace cfg = load_logical_and_i64_store;

  const auto output_shape = cfg::GetShapeConfig(cfg::kOutputShapes, 0).shape;
  const auto input0_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 0).shape);
  const auto input1_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 1).shape);
  AscGraphBuilder builder(cfg::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_INT64)
      .Data("input1", 1, input1_shape, {}, af::DT_INT64)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {});

  // 两个输入是 `DT_INT64`，输出是 `DT_UINT8`。
  auto graph = builder.LogicalAnd("logical_and", "load0", "load1")
                   .Store("store", "logical_and")
                   .Output("output", "store", 0, af::DT_UINT8)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_logical_and_i64_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  EXPECT_NE(kernel.find("LogicalAnd("), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_LOGICAL_H__"), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_UTILS_H__"), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
  EXPECT_EQ(kernel.find("__ASCENDC_API_REGBASE_LOGICAL_H__"), std::string::npos);
  EXPECT_EQ(kernel.find("LogicalAndExtend("), std::string::npos);
}
}  // namespace mlir::afir
