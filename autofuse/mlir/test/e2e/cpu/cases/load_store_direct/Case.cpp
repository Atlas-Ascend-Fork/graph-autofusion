/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_store_direct/Case.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_store_direct::GetShapeConfig(load_store_direct::kOutputShapes, 0).shape;
  const auto input0_shape = ToExpressions(load_store_direct::GetShapeConfig(load_store_direct::kInputShapes, 0).shape);
  AscGraphBuilder builder(load_store_direct::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input0_shape, {}, af::DT_FLOAT);
  builder.Load("load0", "input0", input0_shape, {});

  // load 直连 store：store 的输入就是 load 的输出，数据不经矢量计算直接搬出。
  auto graph = builder.Store("store", "load0").Output("output", "store", 0, af::DT_FLOAT).Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_store_direct::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 物化为双端队列：一条声明同时携带 VECIN 与 VECOUT。
  EXPECT_NE(kernel.find("TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT"), std::string::npos);
  EXPECT_NE(kernel.find(".EnQue("), std::string::npos);
  EXPECT_NE(kernel.find(".DeQue<"), std::string::npos);
  // 队列已含同步，不得叠加显式事件。
  EXPECT_EQ(kernel.find("SetFlag"), std::string::npos);
  EXPECT_EQ(kernel.find("WaitFlag"), std::string::npos);
}
}  // namespace mlir::afir
