/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> neg -> store CPU case -----------------*- C++ -*-===//
//
// `afir.neg` 是第一个「走 Autofuse 包装、却不要 tmp buffer」的 op。在它之前这两件事是
// 同一件事——凡是走包装的都要一块 tmp buffer——所以这个用例钉的是那个耦合已经拆开：
// 生成的 kernel 里既有 `NegExtend`，又一个 `TBuf` 都没有。
//
// CPU E2E 只按单个 soc 编译，所以这里只验 mem-based 那一侧；「同一份 AFIR 在另一张
// 供给表上换成 `neg_reg_base.h`」是 afir-neg-api-family.mlir 的事。
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_neg_store/Case.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_neg_store::GetShapeConfig(load_neg_store::kOutputShapes, 0).shape;
  const auto input_shape = ToExpressions(load_neg_store::GetShapeConfig(load_neg_store::kInputShapes, 0).shape);
  AscGraphBuilder builder(load_neg_store::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  auto graph = builder.Neg("neg", "load0").Store("store", "neg").Output("output", "store", 0, af::DT_FLOAT).Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_neg_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 包装是全局函数，元素类型是显式模板实参。
  EXPECT_NE(kernel.find("NegExtend<float>("), std::string::npos);
  // 头文件被整段内联进来了，而且是 mem-based 那张表的那一份：两张表的包装同名同签名，
  // 只有 include guard 分得开它们。
  EXPECT_NE(kernel.find("__ASCENDC_API_NEG_H__"), std::string::npos);
  EXPECT_EQ(kernel.find("__ASCENDC_API_REGBASE_NEG_H__"), std::string::npos);
  // 走了包装，却一块 tmp buffer 都不要。TBuf 只会因为 tmp buffer 出现——队列走的是 TQue。
  EXPECT_EQ(kernel.find("AscendC::TBuf<"), std::string::npos);
}
}  // namespace mlir::afir
