/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> brc -> abs -> store ------------------*- C++ -*-===//
//
// 级联拓扑 T2：**广播的结果进计算**，而不是直接写回 GM。
//
// 既有的广播用例全是 `load -> brc -> store`，也就是广播之后立刻搬出去。真实模型里广播几乎
// 总是中间结果，而「广播之后还有人用它」会动到三件 T1 碰不到的事：下游算子与 broadcast 争
// tmp buffer、广播输出那块片上张量多了一个消费者（descriptor 必须一致）、以及两者之间的
// `PIPE_V` 同步。
//
// 形状取自 `test/probe/CapabilityBoundaryProbe.cpp` 的拓扑切片里**返回 ok 的格**
// （2026-10-12 实测）。这条顺序不能反：探针返回 unsupported 的格写成 E2E 是**假绿**——
// `GenerateForInductor` 在 MLIR 报不支持时会接着跑 legacy 并返回成功，数值照样全对。
//
// 同一张图在 `[8,1] -> [8,128]`（最内轴是广播轴）上原先是**硬失败**
// （`failed to legalize operation 'afir.abs'`，能力层答支持而 pattern 失败）：那一格的
// `B0Y1` 模板把 `abs` 排到 `afir.broadcast` **之前**，于是它作用在广播前那块向量化轴
// extent 全为 1 的 tile 上，而计算侧当时不收零段。`materializeVectorizedSegments` 的
// 单元素窗口那一档收下之后这一格返回 ok，所以它现在也在这里。
//
// `[1, 65536] -> [33, 65536]`：内轴超过 UB，调度把广播轴与内轴合轴再按块切分
// （`z0z1T`/`z0z1Tb`），不走「物化一份广播结果、复用」的路径，而是每次块迭代各自重新
// `Load` 同一段输入、`Abs` 一遍、按块号算出的 GM 偏移各写各的——语义等价但产物里不出现
// `Broadcast(`/`Duplicate(` 字面调用。这一档原先在 `getGlobalDescriptor`
// （`TransferPlan.cpp`）里把 Store 的 GM 偏移错取成生产者 UB 侧的步长，广播轴那一项的
// 步长读成 0 被整项剔掉，于是全部块写到同一个 GM 地址——静默产生错误结果（2026-11-10 剩余工作
// 清单 H.1）。`KernelTest.cpp` 的逐元素数值比较钉住这一档；本函数不对这一档重复钉文本
// 次序（见下方按形状分流）。
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_broadcast_abs_store/LoadBroadcastAbsStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_broadcast_abs_store;
  const BroadcastBinding binding = BindBroadcast(config::GetShapeConfig(config::kOutputShapes, 0).shape,
                                                 config::GetShapeConfig(config::kInputShapes, 0).shape);
  AscGraphBuilder builder(config::kKernelName);
  builder.Loops(binding.outputShape)
      .Data("input0", 0, binding.inputShape, {}, af::DT_FLOAT)
      .Load("load0", "input0", binding.inputShape, binding.loadStrides);
  return builder.Broadcast("brc", "load0", binding.brcAxes)
      .Abs("abs", "brc")
      .Store("store", "abs")
      .Output("output", "store", 0, af::DT_FLOAT)
      .Build();
}

void CheckGeneratedKernel(const std::string &kernel) {
  namespace config = load_broadcast_abs_store;
  // 走的是 MLIR 那条路而不是回落 legacy。少了这一条，整个用例在回落之后照样全绿。
  EXPECT_NE(kernel.find(std::string(config::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  const size_t broadcast = kernel.find("Broadcast(");
  if (broadcast != std::string::npos) {
    // 两个算子都在，且次序是广播在前。钉次序而不只是「都在」：反了的话数值仍然对
    // （`|brc(x)| == brc(|x|)`），只有次序这一条分得开。
    EXPECT_NE(kernel.find("Abs", broadcast), std::string::npos);
    return;
  }
  // `[1, 65536] -> [33, 65536]`（见文件头注释）：这一档的广播靠按块重新 Load + 各自算
  // GM 偏移实现，不发 `Broadcast(`/`Duplicate(` 字面调用，产物里只有 Abs。数值正确性由
  // `KernelTest.cpp` 的逐元素比较钉住（这正是 H.1 会漏的那一环：当年"声称接管"但产物没
  // 有广播调用、数值大面积为 0，文本断言本身看不出来）。
  EXPECT_NE(kernel.find("Abs"), std::string::npos);
}
}  // namespace mlir::afir
