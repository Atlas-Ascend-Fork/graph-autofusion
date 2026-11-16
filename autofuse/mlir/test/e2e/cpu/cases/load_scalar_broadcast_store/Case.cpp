/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===----------------------------------------------------------------------===//
//
// `Scalar` → `Broadcast` → 两个消费者。对应 legacy `BroadcastScalar` 的第一支
// （`in.is_constant`，`codegen/api_call/broadcast/broadcast_api_call.cpp:249`）：标量没有
// tile，常量名直接拼进 `Duplicate(dst, 标量, count)`。
//
// 两件事决定了这张图的形状：
//
// 一、广播必须有**两个**消费者。legacy 的 `ScalarBroadcastOptimizationPass` 会把
//     「出入度都是 1 且输入 repeats 全为 1」的 scalar broadcast 整个消掉
//     （`optimize/schedule_utils.cpp:1100-1111`）。单消费者的图根本到不了 MLIR 这边，
//     用例就白写了。线上那张图（`mean` 拆成 `sum` + `truediv`，除数被两处除法共用）
//     正是多消费者这一档。
//
// 二、标量源的广播此前连 IR 都建不起来：`afir.broadcast` 的 operand #0 只收 shaped
//     type，bridge 给的是裸标量，于是 module 校验失败、整个 codegen 硬失败（不回落）。
//     结构由 `test/lit/conversion/afir-broadcast-duplicate-scalar.mlir` 钉，这里钉的是
//     另一件事：算出来是对的。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_scalar_broadcast_store/LoadScalarBroadcastStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_scalar_broadcast_store;

  const auto output_shape = config::GetShapeConfig(config::kOutputShapes, 0).shape;
  const auto input_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 0).shape);

  AscGraphBuilder builder(config::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  auto graph = builder.Scalar("two", "2.5", af::DT_FLOAT)
                   .BroadcastFromScalar("brc", "two")
                   .Add("add", "load0", "brc")
                   .Mul("mul", "add", "brc")
                   .Store("store", "mul")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  namespace config = load_scalar_broadcast_store;
  EXPECT_NE(kernel.find(std::string(config::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 标量常量仍然是一条 `constexpr`，与 `load_add_scalar_store` 同一个物化方式。
  EXPECT_NE(kernel.find("constexpr float c2_5_f32 = (float)2.500000000e+00;"), std::string::npos);
  // 标量直接进 `Duplicate`：发的是标准调用，且常量名就在这条调用里。
  const size_t duplicate = kernel.find("AscendC::Duplicate(");
  ASSERT_NE(duplicate, std::string::npos);
  EXPECT_NE(kernel.find("c2_5_f32", duplicate), std::string::npos);
  // 这两条是本用例相对 `load_broadcast_duplicate_store` 的差别：标量源不从 tile 读值，
  // 也就没有为读值准备的事件对。少了这两条断言，把标量当 tile 读的写法也能全绿。
  EXPECT_EQ(kernel.find("AscendC::SetFlag<AscendC::HardEvent::MTE2_S>"), std::string::npos);
  EXPECT_EQ(kernel.find(".GetValue("), std::string::npos);
  // 广播出来的是一块 tile，两个消费者都是 tensor_tensor 形态（四个实参、最后一个是
  // count），不是 `Adds` / `Muls` 那种 tensor_scalar 形态。
  // `Add` 是 `AscendC::` 限定名、`Mul` 是包装头里的非限定名——两者的限定不同不是笔误，
  // 写断言前 dump 过实际输出。
  EXPECT_NE(kernel.find("AscendC::Add("), std::string::npos);
  EXPECT_NE(kernel.find("Mul("), std::string::npos);
  EXPECT_EQ(kernel.find("AscendC::Adds<"), std::string::npos);
  EXPECT_EQ(kernel.find("AscendC::Muls<"), std::string::npos);
  EXPECT_EQ(kernel.find("AscendC::Broadcast"), std::string::npos);
}
}  // namespace mlir::afir
