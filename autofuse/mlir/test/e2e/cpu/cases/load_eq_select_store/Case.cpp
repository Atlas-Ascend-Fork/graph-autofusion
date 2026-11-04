/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load x4 -> eq -> select -> store CPU case -----*- C++ -*-===//
//
// 比较族最自然的下游用法：掩码由 `Eq` 现算，再喂给 `Select`。此前没有任何用例走到
// 「比较族的结果接一个计算节点」——`load_eq_store` 的下游直接是 Store。
//
// **这个图在改动之前就是绿的，记下来**：遗留问题清单把它写成「一条队列上不能混
// dtype」那条限制的复现图（比较族产 ui8，与两边的 f32 落在同一条队列上，当场
// `BindingConflict`），而实测它一路绿到逐元素断言。原因在调度：`Eq` 的 ui8 结果是
// 计算中间量，落在 `TBuf<VECCALC>` 上，四块搬入分在两条 `TQue<VECIN>` 上——掩码
// 与浮点从来没有共用过一条队列。真正混 dtype 的形状是**掩码自己从 GM 搬进来**，
// 由 `load_mask_select_store` 复现，那一份改动之前是红的。
//
// 所以这个用例钉的是另一件事：比较族的结果作为操作数被下一个 op 读到时，dtype 不同
// 的两条链（ui8 掩码 / f32 值）在同一个 kernel 里各自发对了调用。
//
// harness 把 soc 钉在 Ascend910B1，走 mem 侧：`CompareExtend` + `Where`。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_eq_select_store/LoadEqSelectStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_eq_select_store::GetShapeConfig(load_eq_select_store::kOutputShapes, 0).shape;
  const auto input0_shape =
      ToExpressions(load_eq_select_store::GetShapeConfig(load_eq_select_store::kInputShapes, 0).shape);
  const auto input1_shape =
      ToExpressions(load_eq_select_store::GetShapeConfig(load_eq_select_store::kInputShapes, 1).shape);
  const auto input2_shape =
      ToExpressions(load_eq_select_store::GetShapeConfig(load_eq_select_store::kInputShapes, 2).shape);
  const auto input3_shape =
      ToExpressions(load_eq_select_store::GetShapeConfig(load_eq_select_store::kInputShapes, 3).shape);
  AscGraphBuilder builder(load_eq_select_store::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_FLOAT)
      .Data("input1", 1, input1_shape, {}, af::DT_FLOAT)
      .Data("input2", 2, input2_shape, {}, af::DT_FLOAT)
      .Data("input3", 3, input3_shape, {}, af::DT_FLOAT)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {})
      .Load("load2", "input2", input2_shape, {})
      .Load("load3", "input3", input3_shape, {});

  auto graph = builder.Eq("mask", "load0", "load1")
                   .Select("select", "mask", "load2", "load3")
                   .Store("store", "select")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_eq_select_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 两个调用都在同一个 kernel 里。`CMPMODE::EQ` 后面那个 `static_cast<uint32_t>` 是
  // emitter 自己打的（`compare.h` 的头文本里没有），`Where(` 是全局包装、不带
  // `AscendC::` 限定。
  EXPECT_NE(kernel.find("CMPMODE::EQ, static_cast<uint32_t>("), std::string::npos);
  EXPECT_NE(kernel.find("Where("), std::string::npos);
  // 掩码是计算中间量，落在 VECCALC 的 TBuf 上——这就是它没有与浮点共用队列的原因，
  // 也是这个图在放宽队列 dtype 判据之前就绿的原因。
  EXPECT_NE(kernel.find("AscendC::TBuf<AscendC::TPosition::VECCALC>"), std::string::npos);
}
}  // namespace mlir::afir
