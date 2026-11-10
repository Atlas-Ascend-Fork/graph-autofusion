/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> brc -> brc -> store -------------------*- C++ -*-===//
//
// 级联拓扑 T4：两步广播，一根广播轴一趟。
//
// 它与「一条 `afir.broadcast` 带两根广播轴」在数值上等价（同一个结果），但在结构上差一件
// T1 碰不到的事：第二趟广播的源是片上张量，而不是 Load 的结果。广播的源此前总是刚搬进
// 来的那一块，于是「源在 GM 上长什么样」一直答得出来；这一格答不出来——源根本不在 GM 上。
//
// 形状只有一格，而且必须是两根广播轴的（一根的话第二趟没东西可广播），取自探针报 ok
// 的格（2026-10-12 实测）。同一张图在 f16 上实测报不支持、回落 legacy，所以这里只有 f32。
//
// 等价性本身也是这一格在测的东西：期望值按一次广播算（`BroadcastSourceIndex`），
// 两趟拆开之后算出来必须和它逐元素相等。
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_broadcast_chain_store/LoadBroadcastChainStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_broadcast_chain_store;
  const BroadcastBinding binding = BindBroadcast(config::GetShapeConfig(config::kOutputShapes, 0).shape,
                                                 config::GetShapeConfig(config::kInputShapes, 0).shape);
  // 内侧那一半先广播，其余交给第二趟。`brcAxes` 是内 → 外，所以前半是内侧那些。
  // 只有一根广播轴时第二趟是空集，图就退化成 T1——CMakeLists 里那条注释说明了为什么形状
  // 必须带两个 1。
  const size_t split = binding.brcAxes.size() / 2U;
  const std::vector<int64_t> inner(binding.brcAxes.begin(), binding.brcAxes.begin() + split);
  const std::vector<int64_t> outer(binding.brcAxes.begin() + split, binding.brcAxes.end());
  AscGraphBuilder builder(config::kKernelName);
  builder.Loops(binding.outputShape)
      .Data("input0", 0, binding.inputShape, {}, af::DT_FLOAT)
      .Load("load0", "input0", binding.inputShape, binding.loadStrides);
  return builder.Broadcast("brc0", "load0", inner)
      .Broadcast("brc1", "brc0", outer)
      .Store("store", "brc1")
      .Output("output", "store", 0, af::DT_FLOAT)
      .Build();
}

void CheckGeneratedKernel(const std::string &kernel) {
  namespace config = load_broadcast_chain_store;
  // 走的是 MLIR 那条路而不是回落 legacy。少了这一条，整个用例在回落之后照样全绿。
  EXPECT_NE(kernel.find(std::string(config::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 两趟广播都生成了。只数「有没有 Broadcast」数不出这件事——一趟也有。
  EXPECT_GE(CountOccurrences(kernel, "Broadcast("), 2U);
}
}  // namespace mlir::afir
