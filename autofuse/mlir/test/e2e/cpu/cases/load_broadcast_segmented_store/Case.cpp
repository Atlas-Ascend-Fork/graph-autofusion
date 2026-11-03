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

#include "test/e2e/cpu/cases/load_broadcast_segmented_store/Case.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"
namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_broadcast_segmented_store;
  const auto output_shape = config::GetShapeConfig(config::kOutputShapes, 0).shape;
  const auto input_config = config::GetShapeConfig(config::kInputShapes, 0);
  const auto input_shape = ToExpressions(input_config.shape);
  // 广播轴由「输入该维是 1、输出不是」定出来。Load 在广播轴上 stride 取 0，其余维按输入自己的
  // 行主序算——与 legacy 的 nddma 用例（`optimize_st_v2.cpp` 的 `NddmaCase*`）同一写法。
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
      .Data("input0", 0, input_shape, {}, af::DT_FLOAT)
      .Load("load0", "input0", input_shape, load_strides);
  auto graph = builder.Broadcast("brc", "load0", brc_axes)
                   .Store("store", "brc")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  return graph;
}
void CheckGeneratedKernel(const std::string &kernel) {
  namespace config = load_broadcast_segmented_store;
  EXPECT_NE(kernel.find(std::string(config::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 这张图的 stage 有两个模板（广播轴在 tile 内 / 排到 tile 外），计划选择代不出常数，
  // 两个都留给运行期选（见 `SelectStaticWinner` 的回落）。两份都要在 kernel 里。
  EXPECT_NE(kernel.find("stage0_template0"), std::string::npos);
  EXPECT_NE(kernel.find("stage0_template1"), std::string::npos);
  // 缓存守卫：广播轴排在 tile 内的那个模板上，上游给 `load0` / `brc` 标了
  // `exec_condition = cache_block_split_fused_brc_axis`，生成侧要把这两个算子包进一个
  // `if`。谓词的形状是两个 bool 相或（本核第一趟 ‖ 复用周期起点），整份 kernel 里只有它
  // 长这样——改动前这个文件里一处都没有。
  //
  // **结构由 `test/lit/conversion/afir-exec-condition-cache-guard.mlir` 钉**。这里钉的是
  // 这条真实路径上确实生成了守卫，以及跳过那些迭代之后算出来还是对的。
  const size_t predicate = kernel.find(" | v");
  ASSERT_NE(predicate, std::string::npos);
  EXPECT_LT(kernel.find("if (", predicate), kernel.find("Broadcast(", predicate));
}
}  // namespace mlir::afir
