/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> abs -> reciprocal -> store CPU case ---*- C++ -*-===//
//
// **两个都要 tmp buffer 的节点串在一条链上。** 在此之前没有任何用例是这个形状：带
// tmp buffer 的 E2E 全是单计算节点，唯一的多节点用例（load_exp_sqrt_store）两个节点都是
// D0、都不要 tmp buffer。这个洞藏着一个越界写，加这个用例时当场就炸出来了。
//
// 调度给两个节点的是**同一个** tmp buffer id——legacy 的 `mem_reuse_manager` 允许跨节点
// 复用，而 ATT 的尺寸聚合正是跨节点取 max（`buf_occupy_expr.cpp`：各节点内 Σ，节点之间
// max），也就是说 `b{id}_size` 只按一个节点的需要算过一次。MLIR 这边却把两块 alloc 当成
// 同一块资源里**并排**的两个成员，依次累加偏移，于是第二块整个落在资源之外：
// `InitBuffer(tbuf, b0_size)` 之后取 `tbuf.Get()[b0_size]`。没有任何断言拦得住——
// 「resource member offset must not overflow」只查算术溢出，不查偏移小于尺寸。表现是
// 结果全 0，不是报错。
//
// 修法在 `CanonicalGroupKey::exclusiveMemberOrdinal`：tmp buffer 各自成组，各自从 0 开始，
// 与 legacy 的「先后复用同一块」逐字对上。下面 CheckGeneratedKernel 钉的就是这一条。
//
// 链选 abs -> reciprocal 是因为 mem-based 一侧两个都走包装、两个都要 tmp
// （`CalcAbsTmpSize` 对所有 dtype 都给，`CalcDefaultTmpSize` 同）。
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_abs_reciprocal_store/Case.h"
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
  // 前导只发一份。`reciprocal.h` 声明了 `utils.h`（`abs.h` 自足、不声明），而扩展头
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
