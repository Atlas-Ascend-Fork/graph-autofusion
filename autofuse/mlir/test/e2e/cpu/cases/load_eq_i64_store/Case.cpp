/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load x2 -> eq(i64) -> store CPU case -----------*- C++ -*-===//
//
// 比较族 i64 的 e2e。i64 建模进 `AFIR_ELEMENT_TYPE_TABLE` 之前，i64 的比较图死在
// bridge 的供给表（`element type 'int64' is not in the AFIR supply table`）；建模之后
// 这条路第一次真正走得通，这里钉的就是它。mem 侧（harness 钉 Ascend910B1）的
// `compare.h` 对 int64 的 tensor 形态走「Sub 后 cast 到 float 再比较」的那一支，
// 与 f32 不是同一条实现——所以 f32 那份 `load_eq_store` 盖不住这里，必须有自己
// 的数值对照。
//
// 喂值沿用 `load_eq_store` 的原则：相等与不相等各占一半、不相等的两个方向都有，
// 让六种 CMPMODE 的答案互相分得开。i64 的取值带正带负，顺带盖住 int64 字节序下
// 高低位都不为 0 的情形——`EQ` 错生成任何带符号比较的模式都至少错一半元素。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_eq_i64_store/LoadEqI64StoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace cfg = load_eq_i64_store;

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
  auto graph =
      builder.Eq("eq", "load0", "load1").Store("store", "eq").Output("output", "store", 0, af::DT_UINT8).Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_eq_i64_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  EXPECT_NE(kernel.find("CMPMODE::EQ, static_cast<uint32_t>("), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_COMPARE_H__"), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_UTILS_H__"), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
  EXPECT_EQ(kernel.find("__ASCENDC_API_REGBASE_COMPARE_H__"), std::string::npos);
}
}  // namespace mlir::afir
