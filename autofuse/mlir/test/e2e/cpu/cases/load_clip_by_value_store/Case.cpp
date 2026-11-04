/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load x3 -> clip_by_value -> store CPU case ----*- C++ -*-===//
//
// **第一个走完整条流水的三元节点。** 选择族（E5）只有 lit 与 bridge UT 覆盖：它的三个
// 输入里第一个是 ui8 掩码，而这条 CPU E2E 的 harness 一路都是 f32 张量。clip 的三个输入
// 同 dtype，能原样跑完，所以「三个输入 + 一块 tmp buffer」这个操作数形态到这里才第一次
// 被端到端验过——三段划分（inputs | destinations | tmpBuffers）在 arity 3 上是否还对，
// 靠的就是这个用例。
//
// harness 把 soc 钉在 Ascend910B1（cpu/CMakeLists.txt），走的是 V1，也就是 mem-based 一侧：
// `ClipByValue(dst, x, min, max, calCount, tmp_buf)`，扩展头 `clipbyvalue.h`，要 tmp buffer。
// reg-base 一侧在这里到不了，由 afir-clip-by-value-reg-base-extension-api.mlir 钉。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_clip_by_value_store/LoadClipByValueStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"
#include "ascir_ops.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_clip_by_value_store::GetShapeConfig(load_clip_by_value_store::kOutputShapes, 0).shape;
  const auto x_shape =
      ToExpressions(load_clip_by_value_store::GetShapeConfig(load_clip_by_value_store::kInputShapes, 0).shape);
  const auto lower_shape =
      ToExpressions(load_clip_by_value_store::GetShapeConfig(load_clip_by_value_store::kInputShapes, 1).shape);
  const auto upper_shape =
      ToExpressions(load_clip_by_value_store::GetShapeConfig(load_clip_by_value_store::kInputShapes, 2).shape);
  AscGraphBuilder builder(load_clip_by_value_store::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, x_shape, {}, af::DT_FLOAT)
      .Data("input1", 1, lower_shape, {}, af::DT_FLOAT)
      .Data("input2", 2, upper_shape, {}, af::DT_FLOAT)
      .Load("load0", "input0", x_shape, {})
      .Load("load1", "input1", lower_shape, {})
      .Load("load2", "input2", upper_shape, {});

  auto graph = builder.Op<af::ascir_op::ClipByValue>("clip", {"load0", "load1", "load2"})
                   .Store("store", "clip")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_clip_by_value_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 包装是全局函数，没有 `AscendC::` 限定，也没有显式模板实参（元素类型自己推导）。
  EXPECT_NE(kernel.find("ClipByValue("), std::string::npos);
  EXPECT_EQ(kernel.find("AscendC::ClipByValue"), std::string::npos);
  EXPECT_EQ(kernel.find("ClipByValue<"), std::string::npos);
  // 扩展头整段内联进来了。
  EXPECT_NE(kernel.find("__ASCENDC_API_CLIPBYVALUE_H__"), std::string::npos);
  // mem 一侧要 tmp buffer：它作为最后一个实参出现，而调度真的给了一块 TBuf。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
}
}  // namespace mlir::afir
