/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load x3(ui8+f32) -> select -> store CPU case --*- C++ -*-===//
//
// **一条队列上混 dtype。** 三块搬入——一个 ui8 掩码加两块 f32——被调度到同一条
// `TQue<VECIN>` 上，此前 bridge 的 `CheckResourceCompatibility` 要求同一个
// `(domain, buffer_id, reuse_id)` 下的成员 dtype 逐字相同，这张图当场
// `BindingConflict`、整条 MLIR codegen bail 掉。
//
// 队列就是一块字节：legacy 的 `oneBufSize` 取成员的 max，混 dtype 是合法调度。片上
// 资源的字节数由 ATT 填进 `q<id>_size`（descriptor 上是 `size_from_resource`），所以
// descriptor 没有一处依赖 dtype——判据去掉它就够，字段清单现在只有一份，在
// `afir::canonicalResourceFactsAgree`。
//
// 判据是数值：三个成员在同一块字节上各占一段，偏移按各自的元素宽度算。掩码 1 字节、
// 两个值各 4 字节，偏移错一段的话读到的是另一个成员的内容，逐元素断言当场红。
//
// 与 `load_select_store` 互补：那一份的掩码由图里的 `Isnan` 现算（计算中间量落在
// `TBuf<VECCALC>` 上，绕开了这条限制，理由记在那个用例的文件头），这一份是掩码自己
// 从 GM 搬进来。比较族那条链由 `load_eq_select_store` 钉——它走的是前一种形状。
//
// harness 把 soc 钉在 Ascend910B1，走 mem 侧：`Where(dst, mask, src0, src1, count, tmp)`。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_mask_select_store/Case.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace cfg = load_mask_select_store;

  const auto output_shape = cfg::GetShapeConfig(cfg::kOutputShapes, 0).shape;
  const auto input0_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 0).shape);
  const auto input1_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 1).shape);
  const auto input2_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 2).shape);
  AscGraphBuilder builder(cfg::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_UINT8)
      .Data("input1", 1, input1_shape, {}, af::DT_FLOAT)
      .Data("input2", 2, input2_shape, {}, af::DT_FLOAT)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {})
      .Load("load2", "input2", input2_shape, {});

  auto graph = builder.Select("select", "load0", "load1", "load2")
                   .Store("store", "select")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_mask_select_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  EXPECT_NE(kernel.find("Where("), std::string::npos);
  // **三块搬入共用一条队列**：`TQue<VECIN>` 只有一个。这是这个用例存在的理由——判据
  // 要是把 dtype 算回资源身份里，不是分成三条队列，而是整张图发不出来。
  EXPECT_NE(kernel.find("AscendC::TQue<AscendC::TPosition::VECIN, 2>"), std::string::npos);
  EXPECT_EQ(kernel.find("AscendC::TQue<AscendC::TPosition::VECIN, 2>",
                        kernel.find("AscendC::TQue<AscendC::TPosition::VECIN, 2>") + 1),
            std::string::npos);
}
}  // namespace mlir::afir
