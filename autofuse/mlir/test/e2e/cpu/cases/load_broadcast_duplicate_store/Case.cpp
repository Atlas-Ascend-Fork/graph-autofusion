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

#include "test/e2e/cpu/cases/load_broadcast_duplicate_store/Case.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_broadcast_duplicate_store;

  // 源只有一个元素，所以每一根轴都是广播轴、Load 在每根轴上的 stride 都是 0——这正是
  // `Duplicate` 那一档（`AscendC::Duplicate(dst, src.GetValue(0), count)`）的判据。
  // 分段那一档（源在一部分轴上前进）今天到不了这一步，见
  // `docs/development/2026-09-28-afir-vectorized-axis-collapse-design.md` 的「测试设计」。
  const auto output_shape = config::GetShapeConfig(config::kOutputShapes, 0).shape;
  const auto input_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 0).shape);
  const std::vector<af::Expression> load_strides(input_shape.size(), af::sym::kSymbolZero);
  std::vector<int64_t> broadcast_axes(input_shape.size());
  for (size_t axis = 0U; axis < broadcast_axes.size(); ++axis) {
    broadcast_axes[axis] = static_cast<int64_t>(axis);
  }

  AscGraphBuilder builder(config::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input_shape, {}, af::DT_FLOAT)
      .Load("load0", "input0", input_shape, load_strides);

  auto graph = builder.Broadcast("brc", "load0", broadcast_axes)
                   .Store("store", "brc")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  namespace config = load_broadcast_duplicate_store;
  EXPECT_NE(kernel.find(std::string(config::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::TPosition::VECIN"), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::TPosition::VECOUT"), std::string::npos);
  // 标量读事件对 + `GetValue(0)` + `Duplicate`：与 `test/lit/conversion/afir-broadcast-duplicate.mlir`
  // 钉的是同一条调用，这里多钉一层「算出来是对的」。
  EXPECT_NE(kernel.find("AscendC::SetFlag<AscendC::HardEvent::MTE2_S>"), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>"), std::string::npos);
  EXPECT_NE(kernel.find(".GetValue("), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::Duplicate("), std::string::npos);
  EXPECT_EQ(kernel.find("AscendC::Broadcast"), std::string::npos);
  // 缓存守卫：上游给 `load0` / `brc` 标了 `exec_condition = cache_block_split_fused_brc_axis`，
  // 生成侧要把这两个算子包进一个 `if`。谓词的形状是两个 bool 相或（本核第一趟 ‖ 复用周期
  // 起点），整份 kernel 里只有它长这样——改动前这个文件里一处都没有。
  //
  // **结构由 `test/lit/conversion/afir-exec-condition-cache-guard.mlir` 钉**（哪些算子在
  // `if` 里、哪些在外面）。这里钉的是另外两件事：这条真实路径上确实生成了守卫，以及跳过
  // 那些迭代之后**算出来还是对的**——后者是本用例的主断言，不在这段文本里。
  const size_t predicate = kernel.find(" | v");
  ASSERT_NE(predicate, std::string::npos);
  EXPECT_LT(kernel.find("if (", predicate), kernel.find("AscendC::Duplicate(", predicate));
}
}  // namespace mlir::afir
