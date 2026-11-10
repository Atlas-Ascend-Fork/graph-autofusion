/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 验证两个需要 tmp buffer 的节点顺序复用同一资源。
// legacy mem_reuse_manager 可跨节点复用 id；ATT 在节点内求和、节点间取 max，
// 因此 b{id}_size 只覆盖单节点需求，不能把两个 alloc 按累加偏移并排放入同一 TBuf。
// CanonicalGroupKey::exclusiveMemberOrdinal 应使两者独立分组并从偏移 0 开始。
// CheckGeneratedKernel 检查此约束，避免第二块从 tbuf.Get()[b0_size] 开始而越界。
// 选择 abs -> reciprocal，使 mem-based 两节点都需要 scratch，并保留可检查的数值结果。

#include <cstddef>
#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_abs_reciprocal_store/LoadAbsReciprocalStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
// 取 `call(` 之后第 `index` 个实参（0 起）。实参里没有嵌套括号和逗号，按逗号切就够。
std::string ArgumentOf(const std::string &kernel, const std::string &call, size_t index) {
  const size_t open = kernel.find(call);
  if (open == std::string::npos) {
    return {};
  }
  const size_t begin = open + call.size();
  const size_t end = kernel.find(')', begin);
  if (end == std::string::npos) {
    return {};
  }
  std::string arguments = kernel.substr(begin, end - begin);
  size_t at = 0;
  for (size_t seen = 0; seen < index; ++seen) {
    at = arguments.find(',', at);
    if (at == std::string::npos) {
      return {};
    }
    at = arguments.find_first_not_of(" ", at + 1);
  }
  const size_t stop = arguments.find(',', at);
  return arguments.substr(at, stop == std::string::npos ? std::string::npos : stop - at);
}

af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape =
      load_abs_reciprocal_store::GetShapeConfig(load_abs_reciprocal_store::kOutputShapes, 0).shape;
  const auto input_shape =
      ToExpressions(load_abs_reciprocal_store::GetShapeConfig(load_abs_reciprocal_store::kInputShapes, 0).shape);
  AscGraphBuilder builder(load_abs_reciprocal_store::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  auto graph = builder.Abs("abs", "load0")
                   .Reciprocal("reciprocal", "abs")
                   .Store("store", "reciprocal")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_abs_reciprocal_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  EXPECT_NE(kernel.find("AbsExtend<float>("), std::string::npos);
  EXPECT_NE(kernel.find("ReciprocalExtend<float>("), std::string::npos);
  // 两个包装各自的头都内联进来了。
  EXPECT_NE(kernel.find("__ASCENDC_API_ABS_H__"), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_RECIPROCAL_H__"), std::string::npos);
  // 前导只生成一份。`reciprocal.h` 声明了 `utils.h`（`abs.h` 自足、不声明），而扩展头
  // 与前导头都按名字去重——数 `#define` 那一行，因为 `#ifndef` / `#define` 一次内联
  // 就出现两次，找子串数不出重复内联。
  EXPECT_EQ(CountOccurrences(kernel, "#define __ASCENDC_API_UTILS_H__"), 1U);
  // 调度给这两个节点同一个 buffer id，所以只该有一块 tmp buffer，而且**两个调用拿到的
  // 是同一个张量**。并排放会让第二个拿到一个偏移在资源之外的别名——两个实参不同、
  // 用例当场红。这是本用例存在的主要理由。
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::TBuf<"), 1U);
  const std::string absTmpBuffer = ArgumentOf(kernel, "AbsExtend<float>(", 2);
  const std::string reciprocalTmpBuffer = ArgumentOf(kernel, "ReciprocalExtend<float>(", 2);
  EXPECT_FALSE(absTmpBuffer.empty());
  EXPECT_EQ(absTmpBuffer, reciprocalTmpBuffer);
}
}  // namespace mlir::afir
