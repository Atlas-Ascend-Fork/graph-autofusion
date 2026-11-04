/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> add -> abs -> store CPU case (reg-base) -*- C++ -*-===//
//
// **这是第一个会产出 `VectorFunc` 的 CPU E2E。**
//
// 在它之前，10 个 `*_reg_base_*` 用例的计算节点数全是 0 或 1（2026-10-18 数过），够不上
// A5 二次融合的门槛 `kMinVfNodesNum = 2`（`v35/optimize/partition/vector_func_partitioner.cpp`），
// 所以整套 reg 侧 CPU E2E **一张带 VF 的图都没产过**——那一侧全绿说明不了 VF 这条路通不通。
//
// 这张图是最小的会被融的那一种：`add` 与 `abs` 相邻且都无条件 VF-capable，于是调度在
// buffer 分配阶段把它们聚成一个 `VectorFunc` 节点，节点体是一张嵌套 AscGraph。
//
// **它在功能需求 1~4 落地之前是红的**，而且红的方式要看清楚：MLIR 的 bridge 在节点类型
// 那道闸门上报能力不足（`ScheduleToAfir.cpp` 的 `does not support node ... of type
// 'VectorFunc'`），而 CPU E2E 的生成器直接调 `RunMlirCodegen`、不接 legacy 回落，所以
// 失败在生成那一步、不是数值不对。这正是 M1 要的那个「会变红的东西」。
//
// 数值取法与 `load_add_store` / `load_neg_reg_base_store` 同一套：和要跨过 0（这样 `abs`
// 不是恒等变换），输入里不留 0（全零区分不出「算了」和「没算」）。
//
// 设计见 docs/development/2026-10-18-afir-vector-func-implementation-design.md 的 M1。
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_add_abs_reg_base_store/LoadAddAbsRegBaseStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_add_abs_reg_base_store;

  const auto output_shape = config::GetShapeConfig(config::kOutputShapes, 0).shape;
  const auto input0_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 0).shape);
  const auto input1_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 1).shape);
  AscGraphBuilder builder(config::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_FLOAT)
      .Data("input1", 1, input1_shape, {}, af::DT_FLOAT)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {});

  auto graph = builder.Add("add", "load0", "load1")
                   .Abs("abs", "add")
                   .Store("store", "abs")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_add_abs_reg_base_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 产物必须是**寄存器形态**：一个向量作用域、一层 lane 循环，中间结果不落 UB。
  EXPECT_NE(kernel.find("__VEC_SCOPE__"), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::Reg::"), std::string::npos);
  // 两个输入各取一次、结果存一次。**计数式而不是「存在一条」**：少发一条 Load 也能编过、
  // 也能算对（上一个 lane 的寄存器还在），错的是别的形状才会暴露。
  EXPECT_EQ(CountOccurrences(kernel, "LoadAlign"), 2U);
  EXPECT_EQ(CountOccurrences(kernel, "StoreAlign"), 1U);
  // 这两条是「真的走了 VF 这条路」的判据：tile 级的包装一条都不该剩。
  EXPECT_EQ(kernel.find("AddExtend"), std::string::npos);
  EXPECT_EQ(kernel.find("AbsExtend"), std::string::npos);
}
}  // namespace mlir::afir
