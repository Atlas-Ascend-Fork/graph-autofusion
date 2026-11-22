/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - dynamic multi-axis broadcast --------*- C++ -*-===//

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_broadcast_multi_axis_dynamic_store/LoadBroadcastMultiAxisDynamicStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_broadcast_multi_axis_dynamic_store;

  const std::vector<int64_t> output(config::GetShapeConfig(config::kOutputShapes, 0).shape);
  AscGraphBuilder builder(config::kKernelName);

  const af::Expression ks0 = builder.SizeVar("ks0");
  const af::Expression ks1 = builder.SizeVar("ks1");
  const std::vector<af::Expression> outputShape = {ks0, ks1, ks0, ks1};
  const std::vector<af::Expression> sourceShape(4U, af::sym::kSymbolOne);
  const std::vector<af::Expression> sourceStrides(4U, af::sym::kSymbolZero);

  g_e2eRuntimeShapeArgs = {static_cast<uint32_t>(output[0]), static_cast<uint32_t>(output[1])};
  builder.Loops(outputShape)
      .Data("input0", 0, {}, {}, af::DT_FLOAT)
      .Load("load0", "input0", sourceShape, sourceStrides);
  auto graph = builder.Broadcast("brc", "load0", {0, 1, 2, 3})
                   .Store("store", "brc")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();

  // Inductor 的外层 Data/Output 可能只保留 dtype/index，GM 侧 descriptor 为空；
  // Load/Store 的局部 tile 由各自 op 描述。这里显式构造同一个退化形态。
  for (const auto &node : graph.GetAllNodes()) {
    const bool isExternalData = node->GetType() == "Data";
    const bool isOutput = node->GetType() == "Output" && node->GetName() == "output";
    if (isExternalData || isOutput) {
      auto &tensor = node->outputs()[0]->attr;
      tensor.axis.clear();
      tensor.repeats.clear();
      tensor.strides.clear();
    }
  }
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  namespace config = load_broadcast_multi_axis_dynamic_store;
  EXPECT_NE(kernel.find(std::string(config::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::Duplicate("), std::string::npos);
}
}  // namespace mlir::afir
