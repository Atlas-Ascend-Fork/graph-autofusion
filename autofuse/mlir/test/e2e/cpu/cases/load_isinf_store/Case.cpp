/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 编译 is_inf.h 并验证 isinf 到该文件名的 ODS ApiExtensionHeader 映射。
// 文件名与 op 名不同，映射错误需通过内联查表和实际 kernel 编译发现。

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_isinf_store/LoadIsinfStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_isinf_store::GetShapeConfig(load_isinf_store::kOutputShapes, 0).shape;
  const auto input_shape = ToExpressions(load_isinf_store::GetShapeConfig(load_isinf_store::kInputShapes, 0).shape);
  AscGraphBuilder builder(load_isinf_store::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  auto graph =
      builder.IsInf("isinf", "load0").Store("store", "isinf").Output("output", "store", 0, af::DT_UINT8).Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_isinf_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 模板实参是源的 float，不是目的地的 uint8。
  EXPECT_NE(kernel.find("IsInfExtend<float>("), std::string::npos);
  EXPECT_EQ(kernel.find("IsInfExtend<uint8_t>("), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_IS_INF_H__"), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_UTILS_H__"), std::string::npos);
  EXPECT_EQ(kernel.find("ReinterpretCast<bool>"), std::string::npos);
}
}  // namespace mlir::afir
