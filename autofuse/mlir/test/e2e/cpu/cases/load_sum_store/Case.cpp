/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <iterator>
#include <regex>

#include "test/e2e/cpu/cases/load_sum_store/LoadSumStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
namespace {
// `AscGraphBuilder::Output` 只设 index 与 dtype，`Build()` 再把没设形状的输出兜底成循环范围——
// 对逐元素图那正好是对的，但归约之后输出比循环范围小，兜底值就成了归约前的形状。这里按真实输出
// 形状补上视图，步长与构图器的 `ComputeStrides` 同一规则：长度为 1 的维步长记 0。
void SetOutputView(af::AscGraph &graph, const char *name, std::initializer_list<int64_t> shape) {
  const auto output = graph.FindNode(name);
  assert(output != nullptr && output->outputs().size() == 1U);
  auto &view = output->outputs()[0]->attr;
  view.repeats = ToExpressions(shape);
  view.strides.clear();
  int64_t stride = 1;
  for (auto dimension = std::rbegin(shape); dimension != std::rend(shape); ++dimension) {
    view.strides.push_back(af::Symbol(*dimension == 1 ? 0 : stride));
    if (*dimension != 1) {
      stride *= *dimension;
    }
  }
  std::reverse(view.strides.begin(), view.strides.end());
}
}  // namespace

af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto input_shape = GetShapeConfig(load_sum_store::kInputShapes, 0).shape;
  const auto input_exprs = ToExpressions(input_shape);
  const auto output_shape = GetShapeConfig(load_sum_store::kOutputShapes, 0).shape;
  std::vector<size_t> reduce_axes;
  if (input_shape.size() == 2U) {
    // 二维用例按输出形状选归约轴，而不是再加一个注册参数：输出 `[M, 1]` 是末维被归约
    // （tile 布局 `[A, R]`，走 AR），输出 `[1, N]` 是首维被归约（布局 `[R, A]`，走 RA）。
    // 两者的输出形状本来就得分别登记给 harness 分配缓冲，让它同时当判据就不会各说各话。
    reduce_axes = {output_shape.begin()[0] == 1 ? 0U : 1U};
  } else if (input_shape.size() == 3U) {
    // 三维 ARA 注册保留 z0/z2、归约 z1，用于覆盖 AllLoad 的 RA 布局和 A 组多轴乘积。
    // 二维 RA 不满足 IsLegacyFullLoadReduce 或 IsTailReduceFullLoad，不会生成 AllLoad 候选。
    // ARA 的 vectorized_axis 为 [R, A_inner, A_outer]，归类为 RA，first.axes 包含两根 A 轴。
    // 详见 2026-10-01-afir-reduce-allload-template-design.md 2.2、2.3。
    reduce_axes = {1U};
  } else {
    // The four-dimensional registration is the real ARAR graph from the
    // scheduler probe: keep z0/z2 and reduce z1/z3.
    assert(input_shape.size() == 4U);
    reduce_axes = {1U, 3U};
  }
  AscGraphBuilder builder(load_sum_store::kKernelName);
  builder.Loops(input_shape)
      .Data("input0", 0, input_exprs, {}, af::DT_FLOAT)
      .Load("load0", "input0", input_exprs, {})
      .Sum("sum", "load0", reduce_axes);

#ifdef AF_SUM_MAX_TEST
  builder.Max("max", "load0", reduce_axes).Add("add", "sum", "max");
  auto graph = builder.Store("store", "add").Output("output", "store", 0, af::DT_FLOAT).Build();
#elif defined(AF_REDUCE_PARTITION)
  builder.Neg("neg0", "sum").Neg("neg1", "neg0").Neg("neg2", "neg1").Neg("neg3", "neg2");
  auto graph = builder.Store("store", "neg3").Output("output", "store", 0, af::DT_FLOAT).Build();
#else
  auto graph = builder.Store("store", "sum").Output("output", "store", 0, af::DT_FLOAT).Build();
#endif
  // 输出形状取 harness 分配缓冲用的同一份 `kOutputShapes`，构图与分配不会各说各话；bridge 会
  // 核对它与 Store 实际写入的范围是否一致。
  SetOutputView(graph, "output", output_shape);
  // 这里不设 tiling key。ATT 的 `GenerateModelInfo` 用一个自增计数器给每个 impl graph 编
  // `tiling_case_id`，但图上自带的 key 会把计数器顶掉（`graph.GetTilingKey() >= 0 ? ... : tiling_key`）。
  // 默认值是 -1，设成 0 就等于宣称「这几个模板都是第 0 号」——ATT 那边 4 份 `TilingCase0Impl`
  // 撞成重定义，MLIR 这边 `InitializeTemplateInfo` 同一处三元式也退不到 templateId，报
  // `duplicate tiling_key in stage 0`。两个看起来无关的错，同一个根因。
  return graph;
}

#ifdef AF_SUM_MAX_TEST
namespace {
// Extract one emitted function definition, excluding declarations and call sites.
// Generated function bodies use balanced braces; callers must reject an empty result.
std::string GeneratedFunctionBody(const std::string &kernel, const std::string &symbol) {
  const std::regex definition("\\bvoid " + symbol + R"(\([^;{}]*\)\s*\{)");
  std::smatch match;
  if (!std::regex_search(kernel, match, definition)) {
    return {};
  }
  const size_t begin = match.position() + match.length();
  size_t depth = 1;
  for (size_t at = begin; at < kernel.size(); ++at) {
    if (kernel[at] == '{') {
      ++depth;
    } else if (kernel[at] == '}' && --depth == 0) {
      return kernel.substr(begin, at - begin);
    }
  }
  return {};
}

}  // namespace
#endif

void CheckGeneratedKernel(const std::string &kernel) {
#ifdef AF_REDUCE_PARTITION
  EXPECT_NE(kernel.find("stage1_template"), std::string::npos);
#endif
  EXPECT_NE(kernel.find("AscendC::ReduceSum"), std::string::npos);
#ifdef AF_SUM_MAX_TEST
  // This registration executes plan 2. Compare both APIs inside that same template.
  const auto body =
      GeneratedFunctionBody(kernel, std::string(load_sum_store::kKernelName) + "_subgraph0_plan2_stage0_template0");
  ASSERT_FALSE(body.empty());
  std::smatch sum, maximum;
  ASSERT_TRUE(std::regex_search(body, sum, std::regex(R"(AscendC::ReduceSum<[^\n]+>\([^,]+, [^,]+, ([^,]+),)")));
  ASSERT_TRUE(std::regex_search(body, maximum, std::regex(R"(AscendC::ReduceMax<[^\n]+>\([^,]+, [^,]+, ([^,]+),)")));
  EXPECT_EQ(sum[1].str(), maximum[1].str()) << "the two APIs must share scratch in the executed plan";
  EXPECT_NE(body.find("AscendC::Add"), std::string::npos);
#endif

  // AR / RA 的模板参数必须与注册的形状对上，否则这条用例的参考实现算的是另一件事。
  // 二维按输出形状判（与 `BuildGraph` 同一条判据）；三维 ARA 恒为 RA——归约的 z1 在
  // `vectorized_axis` 里排首位（实测 `[R, A_inner, A_outer]`），与输出首维是不是 1 无关。
  const auto &checked_input_shape = GetShapeConfig(load_sum_store::kInputShapes, 0).shape;
  const bool reduces_leading_axis =
      checked_input_shape.size() == 3U || GetShapeConfig(load_sum_store::kOutputShapes, 0).shape.begin()[0] == 1;
  EXPECT_NE(kernel.find(reduces_leading_axis ? "AscendC::Pattern::Reduce::RA" : "AscendC::Pattern::Reduce::AR"),
            std::string::npos);

  // Padding is expanded into native instructions, never an embedded helper.
  EXPECT_EQ(kernel.find("ReduceInit<"), std::string::npos);
  EXPECT_EQ(kernel.find("AUTOFUSE_MLIR_REDUCE_INIT_H"), std::string::npos);
  if (!reduces_leading_axis) {
    EXPECT_NE(kernel.find("AscendC::Duplicate"), std::string::npos);
  }
}
}  // namespace mlir::afir
