/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - 菱形，分叉点是 afir.broadcast ------------------*- C++ -*-===//
//
// 级联拓扑 T5：**广播的结果有两个消费者**，于是它要活到第二个消费者算完。
//
// 它跟着 `load_abs_diamond_store` 一起解锁，原因与广播无关：`afir.broadcast` 也是单输入
// 单输出那一档，调度把它的结果放进非队列的计算缓冲，与下游 unary 的 tmp buffer 共用一个
// `b<id>`，两份 descriptor 只差 `alignment_bytes`（见那条用例的文件头）。
// 探针上 T5 与 T5c / T5e 逐格同步：改动前十格全是
// `InvalidBinding: conflicting descriptors for one canonical resource`。
//
// 它要两条改动才成立，两条都在 `MemoryAllocate`：
//   1. `alignment_bytes` 退出资源身份字段（否则绑定层就报 InvalidBinding）；
//   2. 非队列资源的成员不并排、各自从自己的 `view.offset` 起算（否则第二个成员越界，
//      运行期 abort 且输出一字节不写——这一条在改之前**不报错**，所以这一格是它的
//      唯一数值证据）。
//
// 形状取探针报 ok 的两格（见 CMakeLists.txt）。其余格是 `afir.broadcast` 自己的能力
// 缺口，与本条无关。f16 的 `[1,8] -> [128,8]` 尾轴只有 16B，碰的是「广播尾轴要 32B
// 对齐」那条不变量。
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_broadcast_diamond_store/Case.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_broadcast_diamond_store;
  const BroadcastBinding binding = BindBroadcast(config::GetShapeConfig(config::kOutputShapes, 0).shape,
                                                 config::GetShapeConfig(config::kInputShapes, 0).shape);
  AscGraphBuilder builder(config::kKernelName);
  builder.Loops(binding.outputShape)
      .Data("input0", 0, binding.inputShape, {}, af::DT_FLOAT)
      .Load("load0", "input0", binding.inputShape, binding.loadStrides);
  return builder.Broadcast("brc", "load0", binding.brcAxes)
      .Abs("abs", "brc")
      .Neg("neg", "brc")
      .Add("add", "abs", "neg")
      .Store("store", "add")
      .Output("output", "store", 0, af::DT_FLOAT)
      .Build();
}

void CheckGeneratedKernel(const std::string &kernel) {
  namespace config = load_broadcast_diamond_store;
  // 走的是 MLIR 那条路而不是回落 legacy。**少了这一条，整个用例在回落之后照样全绿**，
  // 而「回落」恰好就是改动前这张图的行为。
  EXPECT_NE(kernel.find(std::string(config::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 两个消费者都在广播之后：钉次序而不只是「都在」。
  const size_t broadcast = kernel.find("Broadcast(");
  ASSERT_NE(broadcast, std::string::npos);
  EXPECT_NE(kernel.find("AbsExtend<float>(", broadcast), std::string::npos);
  EXPECT_NE(kernel.find("NegExtend<float>(", broadcast), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::Add", broadcast), std::string::npos);
}
}  // namespace mlir::afir
