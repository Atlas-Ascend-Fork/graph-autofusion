/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - add(load(full), brc(load(small))) -------------*- C++ -*-===//
//
// 级联拓扑 T3：**广播作二元算子的一个 operand**，另一个 operand 是整形状的 Load。
//
// 这是真实模型里最常见的广播形态（偏置相加、按行缩放都长这样），而既有用例一个都没有：
// 广播此前只出现在「自己写回 GM」那种图里。它动到的那件事是 T1 结构上碰不到的——
// **一个算子身上同时挂着两张来路不同的片上张量**：一张来自广播（源在广播轴上 stride 为 0），
// 一张来自普通搬入。`classifySegments` 的「源与目的地各恰好一个」正是按 T1 那种形状写的。
//
// 形状取自 `test/probe/CapabilityBoundaryProbe.cpp` 的拓扑切片里**报 ok 的格**
// （2026-10-15 在 `9660fef8` 实测）。T3 的 10 格拓扑表中，当前为 6 格 ok、2 格
// FAILED（f32 `[1,8] -> [128,8]` 与 `[1,1,8] -> [4,4,8]`，未捕获首条 error）、
// 2 格 unsupported（f32/f16 `[2,3,5,1] -> [2,3,5,7]`，报
// `afir.broadcast has no supported Ascend implementation`）；T3c 对照在 10 格全过。
// 因此本用例只声明选定格，不用其他格的旧诊断反推当前边界。
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_broadcast_add_store/LoadBroadcastAddStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_broadcast_add_store;
  // `input0` 是广播源（带 1 的那个形状），`input1` 与输出同形。
  const BroadcastBinding binding = BindBroadcast(config::GetShapeConfig(config::kOutputShapes, 0).shape,
                                                 config::GetShapeConfig(config::kInputShapes, 0).shape);
  const auto fullShape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 1).shape);
  AscGraphBuilder builder(config::kKernelName);
  builder.Loops(binding.outputShape)
      .Data("input0", 0, binding.inputShape, {}, af::DT_FLOAT)
      .Data("input1", 1, fullShape, {}, af::DT_FLOAT)
      .Load("load0", "input0", binding.inputShape, binding.loadStrides)
      .Load("load1", "input1", fullShape, {});
  return builder.Broadcast("brc", "load0", binding.brcAxes)
      .Add("add", "load1", "brc")
      .Store("store", "add")
      .Output("output", "store", 0, af::DT_FLOAT)
      .Build();
}

void CheckGeneratedKernel(const std::string &kernel) {
  namespace config = load_broadcast_add_store;
  // 走的是 MLIR 那条路而不是回落 legacy。**少了这一条，整个用例在回落之后照样全绿。**
  EXPECT_NE(kernel.find(std::string(config::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 广播在加法之前。两条搬入与一条广播都在，次序由这一条钉住。
  const size_t broadcast = kernel.find("Broadcast(");
  ASSERT_NE(broadcast, std::string::npos);
  EXPECT_NE(kernel.find("AscendC::Add", broadcast), std::string::npos);
}
}  // namespace mlir::afir
