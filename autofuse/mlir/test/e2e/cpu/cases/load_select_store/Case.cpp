/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load x3 -> select -> store CPU case -----------*- C++ -*-===//
//
// 选择族唯一的编译型用例。E5 接通 select / where / masked_fill 时只写了 lit——lit 比对的
// 是生成的调用文本，不编译内联进来的头，于是 `where.h` 漏声明 `utils.h` 前导那条一路
// 绿到底：`where.h` 与 `duplicate.h` 都用 `KernelUtils::` 和 `MAX_REPEAT_TIME`，只有
// `utils.h` 声明它们。那个 bug 是后来加 clip 的 E2E 时在 `clipbyvalue.h` 上撞出来的，
// 两处一起修；这个用例是把 `where.h` 那一半也钉住，删掉前导声明它就红。
//
// 与 clip 的用例互补：clip 的三个输入同 dtype，这一份的第一个是 ui8 掩码，而
// 「第一个输入的元素类型与另外两个不同」正是选择族与 clip 族在 ODS 与 capability 上
// 分成两支的原因。两类各有一个 E2E，那条分支才两边都被走到。
//
// 掩码由图里的 `Isnan` 现算，而不是从 GM 当第三个输入 load 进来。当初后者跑不通，卡点
// 不在选择族：三个 Load 会被调度到同一条队列上，而 bridge 的
// `CheckResourceCompatibility` 要求同一个 `(domain, buffer_id, reuse_id)` 下的成员 dtype
// 逐字相同，ui8 掩码与 f32 值立刻判 `BindingConflict`。这条限制已解（队列就是一块
// 字节，legacy 的 `oneBufSize` 取成员的 max），从 GM 搬掩码的那一份是
// `load_mask_select_store`。这个用例照旧只走 `Isnan` 现算那一种形状——两种形状分别落在
// `TBuf<VECCALC>` 与共用队列上，各有一个用例。
//
// harness 把 soc 钉在 Ascend910B1，走 V1，也就是 mem-based 一侧：
// `Where(dst, mask, src0, src1, calCount, tmp_buf)`，扩展头 `duplicate.h` + `where.h`。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_select_store/LoadSelectStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_select_store::GetShapeConfig(load_select_store::kOutputShapes, 0).shape;
  const auto probe_shape = ToExpressions(load_select_store::GetShapeConfig(load_select_store::kInputShapes, 0).shape);
  const auto lhs_shape = ToExpressions(load_select_store::GetShapeConfig(load_select_store::kInputShapes, 1).shape);
  const auto rhs_shape = ToExpressions(load_select_store::GetShapeConfig(load_select_store::kInputShapes, 2).shape);
  AscGraphBuilder builder(load_select_store::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, probe_shape, {}, af::DT_FLOAT)
      .Data("input1", 1, lhs_shape, {}, af::DT_FLOAT)
      .Data("input2", 2, rhs_shape, {}, af::DT_FLOAT)
      .Load("load0", "input0", probe_shape, {})
      .Load("load1", "input1", lhs_shape, {})
      .Load("load2", "input2", rhs_shape, {});

  auto graph = builder.Isnan("mask", "load0")
                   .Select("select", "mask", "load1", "load2")
                   .Store("store", "select")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_select_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 包装是全局函数，没有 `AscendC::` 限定，也没有显式模板实参。
  EXPECT_NE(kernel.find("Where("), std::string::npos);
  EXPECT_EQ(kernel.find("AscendC::Where"), std::string::npos);
  EXPECT_EQ(kernel.find("Where<"), std::string::npos);
  // 两个扩展头都内联进来了，前导 `utils.h` 也在——`where.h` / `duplicate.h` 用的
  // `KernelUtils::` 只有它声明，缺了这一份 kernel 编不过。
  EXPECT_NE(kernel.find("__ASCENDC_API_WHERE_H__"), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_DUPLICATE_H__"), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_UTILS_H__"), std::string::npos);
  // mem 一侧要 tmp buffer。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
}
}  // namespace mlir::afir
