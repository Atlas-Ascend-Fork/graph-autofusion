/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load(f16) -> brc -> cast(f32) -> store --------*- C++ -*-===//
//
// 级联拓扑 T6：**dtype 在图中途变**。
//
// 它动到的那件事是 T1 碰不到的：**一张图里「32B 一档装多少个元素」有两个值**。广播那一层
// 的判据正是按这个数算的（「目的地在窗口里稠密」查的是 `Align(Z, 32/sizeof(T))` 与 `Z` 相
// 不相等），广播之后换 dtype，等于让同一串轴在两种档位下各走一遍。
//
// **方向是 f16 → f32，不是反过来**：探针实测 f32 输入那几格整图回落，不是这条路发的调用，
// f16 输入这三格报 ok（2026-10-12 实测）。所以这里收窄到 f16 源。
//
// `[8,1] -> [8,128]` 那一格原先是硬失败（`failed to legalize operation 'afir.cast'`）：
// 那一格的 `B0Y1` 模板把 `cast` 排到 `afir.broadcast` **之前**，于是它作用在广播前那块
// 向量化轴 extent 全为 1 的 tile 上，而计算侧当时不收零段。`materializeVectorizedSegments`
// 的单元素窗口那一档收下之后它报 ok。
//
// 取值全部在 half 里精确可表示（见 KernelTest），所以 `Cast` 之后可以逐元素比相等而不是比
// 容差——容差比法会把「源下标取错一位」这类错误放过去。
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_broadcast_cast_store/LoadBroadcastCastStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_broadcast_cast_store;
  const BroadcastBinding binding = BindBroadcast(config::GetShapeConfig(config::kOutputShapes, 0).shape,
                                                 config::GetShapeConfig(config::kInputShapes, 0).shape);
  AscGraphBuilder builder(config::kKernelName);
  builder.Loops(binding.outputShape)
      .Data("input0", 0, binding.inputShape, {}, af::DT_FLOAT16)
      .Load("load0", "input0", binding.inputShape, binding.loadStrides);
  return builder.Broadcast("brc", "load0", binding.brcAxes)
      .Cast("cast", "brc", af::DT_FLOAT)
      .Store("store", "cast")
      .Output("output", "store", 0, af::DT_FLOAT)
      .Build();
}

void CheckGeneratedKernel(const std::string &kernel) {
  namespace config = load_broadcast_cast_store;
  // 走的是 MLIR 那条路而不是回落 legacy。**少了这一条，整个用例在回落之后照样全绿。**
  EXPECT_NE(kernel.find(std::string(config::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 广播在 Cast 之前。次序反了的话数值也对（广播与逐元素转换可交换），只有次序分得开；
  // 而次序决定了广播那一层按哪个 dtype 的档位算，这正是这一格要测的。
  const size_t broadcast = kernel.find("Broadcast(");
  ASSERT_NE(broadcast, std::string::npos);
  EXPECT_NE(kernel.find("Cast", broadcast), std::string::npos);
}
}  // namespace mlir::afir
