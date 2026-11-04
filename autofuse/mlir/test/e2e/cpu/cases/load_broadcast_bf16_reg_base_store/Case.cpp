/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_broadcast_bf16_reg_base_store/LoadBroadcastBf16RegBaseStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"
namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_broadcast_bf16_reg_base_store;
  const auto output_shape = config::GetShapeConfig(config::kOutputShapes, 0).shape;
  const auto input_config = config::GetShapeConfig(config::kInputShapes, 0);
  const auto input_shape = ToExpressions(input_config.shape);
  // 建图写法与 `load_broadcast_segmented_store/Case.cpp` 逐字同源，只换 dtype：广播轴由
  // 「输入该维是 1、输出不是」定出来，Load 在广播轴上 stride 取 0。
  std::vector<int64_t> in_dims(input_config.shape.begin(), input_config.shape.end());
  std::vector<int64_t> out_dims(output_shape.begin(), output_shape.end());
  std::vector<int64_t> brc_axes;
  std::vector<af::Expression> load_strides(in_dims.size(), af::sym::kSymbolZero);
  int64_t stride = 1;
  for (size_t r = in_dims.size(); r > 0U; --r) {
    const size_t axis = r - 1U;
    if (in_dims[axis] == 1 && out_dims[axis] != 1) {
      brc_axes.push_back(static_cast<int64_t>(axis));
      continue;
    }
    load_strides[axis] = af::Symbol(stride);
    stride *= in_dims[axis];
  }
  AscGraphBuilder builder(config::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input_shape, {}, af::DT_BF16)
      .Load("load0", "input0", input_shape, load_strides);
  auto graph = builder.Broadcast("brc", "load0", brc_axes)
                   .Store("store", "brc")
                   .Output("output", "store", 0, af::DT_BF16)
                   .Build();
  return graph;
}
void CheckGeneratedKernel(const std::string &kernel) {
  namespace config = load_broadcast_bf16_reg_base_store;
  EXPECT_NE(kernel.find(std::string(config::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // **`<bfloat16_t,` 这一段是这个用例的全部理由。** `broadcast.tensor.extend_reg_base` 这一档
  // 的 bf16 位（`Implementations/Broadcast.cpp` 的 `kExtendElementTypes`）与 ODS 的
  // `AFIR_BroadcastElementTypes` 是 2026-10-26 一起开的；开之前这张图失败在 verifier 上
  // （硬失败），开之后要确认它真的选到了这一档、而不是落到别处或者悄悄回落。
  EXPECT_NE(kernel.find("BroadcastExtend<bfloat16_t,"), std::string::npos);
}
}  // namespace mlir::afir
