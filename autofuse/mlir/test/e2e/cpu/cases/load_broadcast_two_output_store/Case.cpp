/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - brc -> {store, store} ------------------------*- C++ -*-===//
//
// 级联拓扑 T7：**一个广播的结果写到两个输出**。
//
// 它动到的那件事是 T1 碰不到的：广播输出那块片上张量有**两个消费者**，而两条 Store 各自
// 要算自己的 GM 偏移。偏移逐张量各取各的 stride 这条（`materializeTensorOffset`）此前只有
// 单输出的图在走，两条 Store 共用一个源是第一次。
//
// 两个输出同形、数值相同——**这不是冗余**：要测的不是「两个结果不一样」，而是两条搬运边各
// 自写对了地方。写错的后果是其中一个输出不对，而它只有在两个都比过之后才看得见。
//
// 形状取自探针报 ok 的格（2026-10-12 实测）。
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_broadcast_two_output_store/LoadBroadcastTwoOutputStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_broadcast_two_output_store;
  const BroadcastBinding binding = BindBroadcast(config::GetShapeConfig(config::kOutputShapes, 0).shape,
                                                 config::GetShapeConfig(config::kInputShapes, 0).shape);
  AscGraphBuilder builder(config::kKernelName);
  builder.Loops(binding.outputShape)
      .Data("input0", 0, binding.inputShape, {}, af::DT_FLOAT)
      .Load("load0", "input0", binding.inputShape, binding.loadStrides);
  return builder.Broadcast("brc", "load0", binding.brcAxes)
      .Store("store0", "brc")
      .Output("output0", "store0", 0, af::DT_FLOAT)
      .Store("store1", "brc")
      .Output("output1", "store1", 1, af::DT_FLOAT)
      .Build();
}

void CheckGeneratedKernel(const std::string &kernel) {
  namespace config = load_broadcast_two_output_store;
  // 走的是 MLIR 那条路而不是回落 legacy。**少了这一条，整个用例在回落之后照样全绿。**
  EXPECT_NE(kernel.find(std::string(config::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 两条搬出都发出来了。只有一条的话另一个输出留着初值，数值那一头会红，这里钉的是结构。
  EXPECT_GE(CountOccurrences(kernel, "DataCopyPad"), 2U);
}
}  // namespace mlir::afir
