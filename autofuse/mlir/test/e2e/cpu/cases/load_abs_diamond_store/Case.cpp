/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - 菱形，分叉点是 unary --------------------------*- C++ -*-===//
//
// 与 `load_add_diamond_store/Case.cpp` **只差一件事**：分叉点是 `Abs`（单输入单输出）
// 而不是 `Add`。那条用例是此前唯一的菱形 E2E，于是「分叉点是什么」这一维在用例集里
// 是盲的——而它正是唯一起作用的那一维（探针 T5c / T5d / T5e，2026-10-12）。
//
// 分叉点是 unary 时，调度会把它的结果放进**非队列的计算缓冲**，而 unary 算子的
// tmp buffer 也在同一个 id 空间里（`ScheduleToAfir` 里那段注释），于是两者可以落到
// 同一个 `b<id>`。此前 `registerResource` 要求同一块存储上的 descriptor 逐字一致，
// 两者只差 `alignment_bytes`（tmp buffer 固定 32、计算缓冲是 -1），整图就报
// `InvalidBinding: conflicting descriptors for one canonical resource`。
// 线性链上看不见：那里 unary 的结果直接进队列，非队列那一侧只有 tmp buffer 一家。
//
// 形状取自探针报 ok 的格。三个 Load 而不是四个：分叉点只吃一个输入。
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_abs_diamond_store/LoadAbsDiamondStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace cfg = load_abs_diamond_store;

  const auto output_shape = cfg::GetShapeConfig(cfg::kOutputShapes, 0).shape;
  const auto input0_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 0).shape);
  const auto input1_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 1).shape);
  const auto input2_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 2).shape);
  AscGraphBuilder builder(cfg::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_FLOAT)
      .Data("input1", 1, input1_shape, {}, af::DT_FLOAT)
      .Data("input2", 2, input2_shape, {}, af::DT_FLOAT)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {})
      .Load("load2", "input2", input2_shape, {});

  // 两个消费者都是 `Add`，与 `load_add_diamond_store` 一致——那条用例改的只有分叉点。
  // 消费者是什么不起作用（T5c 的两个消费者是 unary，结论相同），所以这里不另立一格。
  auto graph = builder.Abs("fork", "load0")
                   .Add("add1", "fork", "load1")
                   .Add("add2", "fork", "load2")
                   .Add("add3", "add1", "add2")
                   .Store("store", "add3")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  // 走的是 MLIR 那条路而不是回落 legacy。**少了这一条，整个用例在回落之后照样全绿。**
  // 改动前这张图是硬失败（不是回落），所以这一条更要钉住：真修好了与「悄悄交给 legacy」
  // 在数值上一模一样。
  EXPECT_NE(kernel.find(std::string(load_abs_diamond_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  EXPECT_NE(kernel.find("AbsExtend<float>("), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::Add"), std::string::npos);
  // 分叉点的结果与 `Abs` 的 tmp buffer 共用一个 `b<id>`，所以这条路上必然有一块 TBuf。
  // 改动前正是这两份 descriptor 对不上才整图硬失败，钉住它就钉住了「真的合到了一块」。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
}
}  // namespace mlir::afir
