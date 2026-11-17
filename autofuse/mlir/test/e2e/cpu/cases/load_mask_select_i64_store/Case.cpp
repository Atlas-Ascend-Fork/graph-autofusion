/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load mask + x2 -> select(i64) -> store CPU case --*- C++ -*-===//
//
// 选择族 i64 的 e2e。mem 侧 `where.h` 对 int64 专门有 `CastAndSelect`（int64 的 mask
// 移位与 cast 分支）——与 f32 那条 `Where(dst, mask, src0, src1, count, tmp)` 不是同
// 一条实现，f32 那份 `load_mask_select_store` 盖不住。i64 建模之前这张图死在 bridge
// 的供给表；这里同时钉「bridge 收 i64 的 select」与「int64 分支的数值」两件事。
//
// harness 把 soc 钉在 Ascend910B1，走 mem 侧。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_mask_select_i64_store/LoadMaskSelectI64StoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace cfg = load_mask_select_i64_store;

  const auto output_shape = cfg::GetShapeConfig(cfg::kOutputShapes, 0).shape;
  const auto input0_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 0).shape);
  const auto input1_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 1).shape);
  const auto input2_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 2).shape);
  AscGraphBuilder builder(cfg::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_UINT8)
      .Data("input1", 1, input1_shape, {}, af::DT_INT64)
      .Data("input2", 2, input2_shape, {}, af::DT_INT64)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {})
      .Load("load2", "input2", input2_shape, {});

  // 掩杀是 `DT_UINT8`，两个数据输入与输出都是 `DT_INT64`。
  auto graph = builder.Select("select", "load0", "load1", "load2")
                   .Store("store", "select")
                   .Output("output", "store", 0, af::DT_INT64)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_mask_select_i64_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  EXPECT_NE(kernel.find("Where("), std::string::npos);
  // 三块搬入共用一条队列：与 `load_mask_select_store` 同一判据——dtype 不参与资源身份。
  EXPECT_NE(kernel.find("AscendC::TQue<AscendC::TPosition::VECIN, 2>"), std::string::npos);
  EXPECT_EQ(kernel.find("AscendC::TQue<AscendC::TPosition::VECIN, 2>",
                        kernel.find("AscendC::TQue<AscendC::TPosition::VECIN, 2>") + 1),
            std::string::npos);
}
}  // namespace mlir::afir
