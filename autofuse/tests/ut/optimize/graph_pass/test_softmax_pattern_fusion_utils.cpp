/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License for the specific language governing permissions and limitations under the License.
 */

#include "gtest/gtest.h"

#include <string>
#include <vector>

#include "ascir_ops.h"
#include "graph/utils/graph_utils.h"
#include "indirect_load_utils.h"
#include "optimize/graph_pass/softmax_pattern_fusion_utils.h"

namespace {
// softmax pattern 原始输入的来源形态：
// - kDirect：pattern 输入直接是 IndirectLoad 输出；
// - kMultiInputAdd：IndirectLoad 输出经 +bias / +bmm 两条双输入 Add 链后进入 pattern；
// - kUnaryChain：IndirectLoad 输出经单输入 Elementwise 链后进入 pattern；
// - kBroadcastChain：中间存在 Broadcast（非 Elementwise），不回溯；
// - kTransposeChain：中间存在 Transpose（非 Elementwise compute_type），不回溯；
// - kUnrelated：pattern 输入来自独立 Load，与 IndirectLoad 无关。
enum class SoftmaxSourceKind {
  kDirect,
  kMultiInputAdd,
  kUnaryChain,
  kUnaryThenBroadcastChain,
  kBroadcastChain,
  kTransposeChain,
  kUnrelated
};

struct SoftmaxGraphHandle {
  af::AscNodePtr indirect_load;
  af::AscNodePtr true_div;
  af::AscNodePtr sub_node;
  std::vector<af::AxisId> output_axes;
  std::vector<af::Expression> output_repeats;
  std::vector<af::Expression> output_strides;
};

template <typename Op>
void SetSoftmaxNodeView(Op &op, af::DataType dtype, const std::vector<af::AxisId> &axes,
                        const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides) {
  op.y.dtype = dtype;
  op.attr.sched.axis = axes;
  *op.y.axis = axes;
  *op.y.repeats = repeats;
  *op.y.strides = strides;
}

template <typename Op>
void SetElewiseApi(Op &op) {
  op.attr.api.compute_type = af::ComputeType::kComputeElewise;
  op.attr.api.type = af::ApiType::kAPITypeCompute;
}

// 在 graph 上构建 IndirectLoad -> (来源链) -> 稳定 softmax pattern -> Store -> Output。
// use_div_node 控制除法尾节点形态：false 为 TrueDiv，true 为 Div（同 dtype 浮点除法）。
void BuildIndirectLoadSoftmaxGraph(af::AscGraph &graph, SoftmaxSourceKind kind, SoftmaxGraphHandle &handle,
                                   bool use_div_node = false) {
  const af::Expression s0 = graph.CreateSizeVar(2);
  const af::Expression s1 = graph.CreateSizeVar(3);
  const af::Expression s2 = graph.CreateSizeVar(5);
  const af::Expression in0 = graph.CreateSizeVar(2);
  const af::Expression in1 = graph.CreateSizeVar(4);
  const af::Expression in2 = graph.CreateSizeVar(5);
  const auto y0 = graph.CreateAxis("y0", s0);
  const auto y1 = graph.CreateAxis("y1", s1);
  const auto y2 = graph.CreateAxis("y2", s2);
  const auto x0 = graph.CreateAxis("x0", in0);
  const auto x1 = graph.CreateAxis("x1", in1);
  const auto x2 = graph.CreateAxis("x2", in2);
  handle.output_axes = {y0.id, y1.id, y2.id};
  handle.output_repeats = {s0, s1, s2};
  handle.output_strides = {s1 * s2, s2, af::sym::kSymbolOne};
  const std::vector<af::AxisId> input_axes = {x0.id, x1.id, x2.id};
  const std::vector<af::Expression> input_repeats = {in0, in1, in2};
  const std::vector<af::Expression> input_strides = {in1 * in2, in2, af::sym::kSymbolOne};
  // 尾轴归约视图：尾轴 repeat=1、stride=0，其余轴保持。
  const std::vector<af::Expression> reduce_repeats = {s0, s1, af::sym::kSymbolOne};
  const std::vector<af::Expression> reduce_strides = {s1 * s2, s2, af::sym::kSymbolZero};

  af::ascir_op::Data input_data("input_data", graph);
  input_data.ir_attr.SetIndex(0);
  SetSoftmaxNodeView(input_data, af::DT_FLOAT16, input_axes, input_repeats, input_strides);
  af::ascir_op::Load input_load("input_load");
  input_load.x = input_data.y;
  SetSoftmaxNodeView(input_load, af::DT_FLOAT16, input_axes, input_repeats, input_strides);
  af::ascir_op::Data index_data("index_data", graph);
  index_data.ir_attr.SetIndex(1);
  SetSoftmaxNodeView(index_data, af::DT_INT32, handle.output_axes, handle.output_repeats, handle.output_strides);
  af::ascir_op::Load index_load("index_load");
  index_load.x = index_data.y;
  SetSoftmaxNodeView(index_load, af::DT_INT32, handle.output_axes, handle.output_repeats, handle.output_strides);

  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  indirect_load.x1 = input_load.y;
  indirect_load.x2 = index_load.y;
  indirect_load.ir_attr.SetAxis(1);
  SetSoftmaxNodeView(indirect_load, af::DT_FLOAT16, handle.output_axes, handle.output_repeats, handle.output_strides);

  // pattern 原始输入（回流点）按场景构造。
  af::AscOpOutput *pattern_source = &indirect_load.y;
  if (kind == SoftmaxSourceKind::kMultiInputAdd) {
    af::ascir_op::Data bias_data("bias_data", graph);
    bias_data.ir_attr.SetIndex(2);
    SetSoftmaxNodeView(bias_data, af::DT_FLOAT16, handle.output_axes, handle.output_repeats, handle.output_strides);
    af::ascir_op::Load bias_load("bias_load");
    bias_load.x = bias_data.y;
    SetSoftmaxNodeView(bias_load, af::DT_FLOAT16, handle.output_axes, handle.output_repeats, handle.output_strides);
    af::ascir_op::Data bmm_data("bmm_data", graph);
    bmm_data.ir_attr.SetIndex(3);
    SetSoftmaxNodeView(bmm_data, af::DT_FLOAT16, handle.output_axes, handle.output_repeats, handle.output_strides);
    af::ascir_op::Load bmm_load("bmm_load");
    bmm_load.x = bmm_data.y;
    SetSoftmaxNodeView(bmm_load, af::DT_FLOAT16, handle.output_axes, handle.output_repeats, handle.output_strides);
    af::ascir_op::Add bias_add("bias_add");
    bias_add.x1 = indirect_load.y;
    bias_add.x2 = bias_load.y;
    SetElewiseApi(bias_add);
    SetSoftmaxNodeView(bias_add, af::DT_FLOAT16, handle.output_axes, handle.output_repeats, handle.output_strides);
    af::ascir_op::Add bmm_add("bmm_add");
    bmm_add.x1 = bias_add.y;
    bmm_add.x2 = bmm_load.y;
    SetElewiseApi(bmm_add);
    SetSoftmaxNodeView(bmm_add, af::DT_FLOAT16, handle.output_axes, handle.output_repeats, handle.output_strides);
    pattern_source = &bmm_add.y;
  } else if (kind == SoftmaxSourceKind::kUnaryChain) {
    af::ascir_op::Abs chain_abs("chain_abs");
    chain_abs.x = indirect_load.y;
    SetElewiseApi(chain_abs);
    SetSoftmaxNodeView(chain_abs, af::DT_FLOAT16, handle.output_axes, handle.output_repeats, handle.output_strides);
    pattern_source = &chain_abs.y;
  } else if (kind == SoftmaxSourceKind::kUnaryThenBroadcastChain) {
    // IndirectLoad -> Broadcast(链中间，非 Elementwise，回溯终止) -> Abs -> pattern。
    af::ascir_op::Broadcast mid_broadcast("mid_broadcast");
    mid_broadcast.x = indirect_load.y;
    SetSoftmaxNodeView(mid_broadcast, af::DT_FLOAT16, handle.output_axes, handle.output_repeats, handle.output_strides);
    af::ascir_op::Abs chain_abs("chain_abs");
    chain_abs.x = mid_broadcast.y;
    SetElewiseApi(chain_abs);
    SetSoftmaxNodeView(chain_abs, af::DT_FLOAT16, handle.output_axes, handle.output_repeats, handle.output_strides);
    pattern_source = &chain_abs.y;
  } else if (kind == SoftmaxSourceKind::kBroadcastChain) {
    af::ascir_op::Broadcast chain_broadcast("chain_broadcast");
    chain_broadcast.x = indirect_load.y;
    SetSoftmaxNodeView(chain_broadcast, af::DT_FLOAT16, handle.output_axes, handle.output_repeats,
                       handle.output_strides);
    pattern_source = &chain_broadcast.y;
  } else if (kind == SoftmaxSourceKind::kTransposeChain) {
    af::ascir_op::Transpose chain_transpose("chain_transpose");
    chain_transpose.x = indirect_load.y;
    // Transpose 的 compute_type 不是 kComputeElewise，回溯必须终止。
    chain_transpose.attr.api.compute_type = af::ComputeType::kComputeTranspose;
    chain_transpose.attr.api.type = af::ApiType::kAPITypeCompute;
    SetSoftmaxNodeView(chain_transpose, af::DT_FLOAT16, handle.output_axes, handle.output_repeats,
                       handle.output_strides);
    pattern_source = &chain_transpose.y;
  } else if (kind == SoftmaxSourceKind::kUnrelated) {
    af::ascir_op::Data plain_data("plain_data", graph);
    plain_data.ir_attr.SetIndex(4);
    SetSoftmaxNodeView(plain_data, af::DT_FLOAT16, handle.output_axes, handle.output_repeats, handle.output_strides);
    af::ascir_op::Load plain_load("plain_load");
    plain_load.x = plain_data.y;
    SetSoftmaxNodeView(plain_load, af::DT_FLOAT16, handle.output_axes, handle.output_repeats, handle.output_strides);
    pattern_source = &plain_load.y;
  }

  // 稳定 softmax pattern：max -> brc -> sub -> exp -> sum -> brc -> truediv。
  af::ascir_op::Max max_op("max");
  max_op.x = *pattern_source;
  max_op.attr.api.compute_type = af::ComputeType::kComputeReduce;
  SetSoftmaxNodeView(max_op, af::DT_FLOAT16, handle.output_axes, reduce_repeats, reduce_strides);
  af::ascir_op::Broadcast max_broadcast("max_broadcast");
  max_broadcast.x = max_op.y;
  SetSoftmaxNodeView(max_broadcast, af::DT_FLOAT16, handle.output_axes, handle.output_repeats, handle.output_strides);
  af::ascir_op::Sub sub_op("sub");
  sub_op.x1 = *pattern_source;
  sub_op.x2 = max_broadcast.y;
  SetElewiseApi(sub_op);
  SetSoftmaxNodeView(sub_op, af::DT_FLOAT16, handle.output_axes, handle.output_repeats, handle.output_strides);
  af::ascir_op::Exp exp_op("exp");
  exp_op.x = sub_op.y;
  SetElewiseApi(exp_op);
  SetSoftmaxNodeView(exp_op, af::DT_FLOAT16, handle.output_axes, handle.output_repeats, handle.output_strides);
  af::ascir_op::Sum sum_op("sum");
  sum_op.x = exp_op.y;
  sum_op.attr.api.compute_type = af::ComputeType::kComputeReduce;
  SetSoftmaxNodeView(sum_op, af::DT_FLOAT16, handle.output_axes, reduce_repeats, reduce_strides);
  af::ascir_op::Broadcast sum_broadcast("sum_broadcast");
  sum_broadcast.x = sum_op.y;
  SetSoftmaxNodeView(sum_broadcast, af::DT_FLOAT16, handle.output_axes, handle.output_repeats, handle.output_strides);
  af::AscOpOutput *div_result = nullptr;
  if (use_div_node) {
    af::ascir_op::Div div_op("true_div");
    div_op.x1 = exp_op.y;
    div_op.x2 = sum_broadcast.y;
    SetElewiseApi(div_op);
    SetSoftmaxNodeView(div_op, af::DT_FLOAT16, handle.output_axes, handle.output_repeats, handle.output_strides);
    div_result = &div_op.y;
  } else {
    af::ascir_op::TrueDiv true_div("true_div");
    true_div.x1 = exp_op.y;
    true_div.x2 = sum_broadcast.y;
    SetElewiseApi(true_div);
    SetSoftmaxNodeView(true_div, af::DT_FLOAT16, handle.output_axes, handle.output_repeats, handle.output_strides);
    div_result = &true_div.y;
  }
  af::ascir_op::Store store("store");
  store.x = *div_result;
  SetSoftmaxNodeView(store, af::DT_FLOAT16, handle.output_axes, handle.output_repeats, handle.output_strides);
  af::ascir_op::Output output("output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  SetSoftmaxNodeView(output, af::DT_FLOAT16, handle.output_axes, handle.output_repeats, handle.output_strides);

  if (kind == SoftmaxSourceKind::kUnrelated) {
    // IndirectLoad 输出接一个独立 Store，保证图合法且与 pattern 无关。
    af::ascir_op::Store unrelated_store("unrelated_store");
    unrelated_store.x = indirect_load.y;
    SetSoftmaxNodeView(unrelated_store, af::DT_FLOAT16, handle.output_axes, handle.output_repeats,
                       handle.output_strides);
    af::ascir_op::Output unrelated_output("unrelated_output");
    unrelated_output.x = unrelated_store.y;
    unrelated_output.ir_attr.SetIndex(1);
    SetSoftmaxNodeView(unrelated_output, af::DT_FLOAT16, handle.output_axes, handle.output_repeats,
                       handle.output_strides);
  }
  handle.indirect_load = graph.FindNode("indirect_load");
  handle.true_div = graph.FindNode("true_div");
  handle.sub_node = graph.FindNode("sub");
}

bool GraphHasNode(const af::AscGraph &graph, const std::string &name) {
  return graph.FindNode(name.c_str()) != nullptr;
}
}  // namespace

TEST(SoftmaxPatternFusionUtilsTest, MatchStableStructureHitsCompletePattern) {
  af::AscGraph graph("softmax_pattern_utils_ut_graph");
  SoftmaxGraphHandle handle;
  BuildIndirectLoadSoftmaxGraph(graph, SoftmaxSourceKind::kDirect, handle);
  ASSERT_NE(handle.true_div, nullptr);
  optimize::softmax_pattern::MatchResult pattern;
  EXPECT_TRUE(optimize::softmax_pattern::MatchStable(handle.true_div, pattern));
  EXPECT_EQ(pattern.max_node->GetName(), "max");
  EXPECT_EQ(pattern.sub_node->GetName(), "sub");
  EXPECT_EQ(pattern.exp_node->GetName(), "exp");
  EXPECT_EQ(pattern.sum_node->GetName(), "sum");
  EXPECT_EQ(pattern.true_div_node->GetName(), "true_div");
}

TEST(SoftmaxPatternFusionUtilsTest, MatchStableStructureHitsCompletePatternWithDiv) {
  af::AscGraph graph("softmax_pattern_utils_ut_graph_div");
  SoftmaxGraphHandle handle;
  BuildIndirectLoadSoftmaxGraph(graph, SoftmaxSourceKind::kDirect, handle, true);
  ASSERT_NE(handle.true_div, nullptr);
  ASSERT_STREQ(handle.true_div->GetTypePtr(), "Div");
  optimize::softmax_pattern::MatchResult pattern;
  EXPECT_TRUE(optimize::softmax_pattern::MatchStable(handle.true_div, pattern));
  EXPECT_EQ(pattern.max_node->GetName(), "max");
  EXPECT_EQ(pattern.sub_node->GetName(), "sub");
  EXPECT_EQ(pattern.exp_node->GetName(), "exp");
  EXPECT_EQ(pattern.sum_node->GetName(), "sum");
  EXPECT_EQ(pattern.true_div_node->GetName(), "true_div");
}

TEST(SoftmaxPatternFusionUtilsTest, MatchStableRejectsWhenSumInputIsNotExp) {
  af::AscGraph graph("softmax_pattern_utils_ut_graph");
  SoftmaxGraphHandle handle;
  BuildIndirectLoadSoftmaxGraph(graph, SoftmaxSourceKind::kDirect, handle);
  // 断开 sum <- exp 的回流，改为 sum 直接消费 sub 的输出。
  const auto sum_node = graph.FindNode("sum");
  const auto exp_node = graph.FindNode("exp");
  const auto sub_node = graph.FindNode("sub");
  ASSERT_NE(sum_node, nullptr);
  ASSERT_NE(exp_node, nullptr);
  ASSERT_NE(sub_node, nullptr);
  EXPECT_EQ(af::GraphUtils::ReplaceEdgeSrc(exp_node->GetOutDataAnchor(0), sum_node->GetInDataAnchor(0),
                                           sub_node->GetOutDataAnchor(0)),
            af::GRAPH_SUCCESS);
  optimize::softmax_pattern::MatchResult pattern;
  EXPECT_FALSE(optimize::softmax_pattern::MatchStable(handle.true_div, pattern));
}

TEST(SoftmaxPatternFusionUtilsTest, MatchStableDedicatedRequiresTailAxisReduce) {
  af::AscGraph graph("softmax_pattern_utils_ut_graph");
  SoftmaxGraphHandle handle;
  BuildIndirectLoadSoftmaxGraph(graph, SoftmaxSourceKind::kDirect, handle);
  ASSERT_NE(handle.true_div, nullptr);
  optimize::softmax_pattern::MatchResult pattern;
  EXPECT_TRUE(optimize::softmax_pattern::MatchStableDedicated(handle.true_div, pattern));

  // 把 max/sum 的归约轴改为首轴（非尾轴），专用匹配必须失败。
  for (const char *name : {"max", "sum"}) {
    const auto node = graph.FindNode(name);
    ASSERT_NE(node, nullptr);
    auto *output = node->outputs()[0];
    output->attr.repeats = {af::sym::kSymbolOne, handle.output_repeats[1], handle.output_repeats[2]};
    output->attr.strides = {af::sym::kSymbolZero, handle.output_strides[1], handle.output_strides[2]};
  }
  optimize::softmax_pattern::MatchResult rejected;
  EXPECT_FALSE(optimize::softmax_pattern::MatchStableDedicated(handle.true_div, rejected));
}

TEST(SoftmaxPatternFusionUtilsTest, ReplaceWithSoftmaxSwapsNodesAndKeepsViews) {
  af::AscGraph graph("softmax_pattern_utils_ut_graph");
  SoftmaxGraphHandle handle;
  BuildIndirectLoadSoftmaxGraph(graph, SoftmaxSourceKind::kDirect, handle);
  ASSERT_NE(handle.true_div, nullptr);
  ASSERT_NE(handle.sub_node, nullptr);
  optimize::softmax_pattern::MatchResult pattern;
  ASSERT_TRUE(optimize::softmax_pattern::MatchStable(handle.true_div, pattern));
  const auto expected_output_axis = handle.true_div->outputs()[0]->attr.axis;
  const auto expected_input_axis = handle.sub_node->inputs()[0]->attr.axis;

  ASSERT_EQ(optimize::softmax_pattern::ReplaceWithSoftmax(graph, pattern), af::SUCCESS);
  const auto softmax_node = graph.FindNode("true_div_softmax");
  ASSERT_NE(softmax_node, nullptr);
  EXPECT_EQ(softmax_node->attr.api.compute_type, af::ComputeType::kComputeReduce);
  // Softmax 继承 pattern 输入/输出视图。
  EXPECT_EQ(softmax_node->inputs()[0]->attr.axis, expected_input_axis);
  EXPECT_EQ(softmax_node->outputs()[0]->attr.axis, expected_output_axis);
  // 原 pattern 节点全部移除。
  for (const char *name : {"max", "max_broadcast", "sub", "exp", "sum", "sum_broadcast", "true_div"}) {
    EXPECT_FALSE(GraphHasNode(graph, name)) << "node " << name << " should be removed";
  }
}

TEST(SoftmaxPatternFusionUtilsTest, NormalizeDirectPostSoftmaxReplacesDirectInput) {
  af::AscGraph graph("softmax_pattern_utils_ut_graph");
  SoftmaxGraphHandle handle;
  BuildIndirectLoadSoftmaxGraph(graph, SoftmaxSourceKind::kDirect, handle);
  ASSERT_NE(handle.indirect_load, nullptr);
  bool changed = false;
  ASSERT_EQ(optimize::softmax_pattern::NormalizeDirectPostSoftmax(graph, handle.indirect_load, changed), af::SUCCESS);
  EXPECT_TRUE(changed);
  EXPECT_TRUE(GraphHasNode(graph, "true_div_softmax"));
  EXPECT_FALSE(GraphHasNode(graph, "true_div"));
}

TEST(SoftmaxPatternFusionUtilsTest, NormalizeDirectPostSoftmaxTracesMultiInputElementwiseChain) {
  // 回归（本需求核心场景）：gather 输出经 +bias / +bmm 双输入 Add 链后再进入 softmax
  // pattern。多输入 Elementwise 回溯必须命中替换。
  af::AscGraph graph("softmax_pattern_utils_ut_graph");
  SoftmaxGraphHandle handle;
  BuildIndirectLoadSoftmaxGraph(graph, SoftmaxSourceKind::kMultiInputAdd, handle);
  ASSERT_NE(handle.indirect_load, nullptr);
  bool changed = false;
  ASSERT_EQ(optimize::softmax_pattern::NormalizeDirectPostSoftmax(graph, handle.indirect_load, changed), af::SUCCESS);
  EXPECT_TRUE(changed);
  EXPECT_TRUE(GraphHasNode(graph, "true_div_softmax"));
  // 中间 Add 链保留在 Softmax 之前。
  EXPECT_TRUE(GraphHasNode(graph, "bias_add"));
  EXPECT_TRUE(GraphHasNode(graph, "bmm_add"));
  for (const char *name : {"max", "max_broadcast", "sub", "exp", "sum", "sum_broadcast", "true_div"}) {
    EXPECT_FALSE(GraphHasNode(graph, name)) << "node " << name << " should be removed";
  }
}

TEST(SoftmaxPatternFusionUtilsTest, NormalizeDirectPostSoftmaxTracesUnaryChain) {
  af::AscGraph graph("softmax_pattern_utils_ut_graph");
  SoftmaxGraphHandle handle;
  BuildIndirectLoadSoftmaxGraph(graph, SoftmaxSourceKind::kUnaryChain, handle);
  ASSERT_NE(handle.indirect_load, nullptr);
  bool changed = false;
  ASSERT_EQ(optimize::softmax_pattern::NormalizeDirectPostSoftmax(graph, handle.indirect_load, changed), af::SUCCESS);
  EXPECT_TRUE(changed);
  EXPECT_TRUE(GraphHasNode(graph, "true_div_softmax"));
  EXPECT_TRUE(GraphHasNode(graph, "chain_abs"));
}

TEST(SoftmaxPatternFusionUtilsTest, NormalizeDirectPostSoftmaxTracesDirectNonElementwiseProducer) {
  // 回溯的 depth=0（pattern 直接生产者）豁免 Elementwise 检查：Broadcast/Transpose 作为
  // 直接生产者时替换仍命中（替换不移动该节点，语义不变）。
  for (const auto kind : {SoftmaxSourceKind::kBroadcastChain, SoftmaxSourceKind::kTransposeChain}) {
    af::AscGraph graph("softmax_pattern_utils_ut_graph");
    SoftmaxGraphHandle handle;
    BuildIndirectLoadSoftmaxGraph(graph, kind, handle);
    ASSERT_NE(handle.indirect_load, nullptr);
    bool changed = false;
    ASSERT_EQ(optimize::softmax_pattern::NormalizeDirectPostSoftmax(graph, handle.indirect_load, changed), af::SUCCESS);
    EXPECT_TRUE(changed);
    EXPECT_TRUE(GraphHasNode(graph, "true_div_softmax"));
  }
}

TEST(SoftmaxPatternFusionUtilsTest, NormalizeDirectPostSoftmaxKeepsNonElementwiseMidChain) {
  // 中间链上的 Broadcast（depth>0，非 Elementwise）终止回溯，不做替换。
  af::AscGraph graph("softmax_pattern_utils_ut_graph");
  SoftmaxGraphHandle handle;
  BuildIndirectLoadSoftmaxGraph(graph, SoftmaxSourceKind::kUnaryThenBroadcastChain, handle);
  ASSERT_NE(handle.indirect_load, nullptr);
  bool changed = true;
  ASSERT_EQ(optimize::softmax_pattern::NormalizeDirectPostSoftmax(graph, handle.indirect_load, changed), af::SUCCESS);
  EXPECT_FALSE(changed);
  EXPECT_FALSE(GraphHasNode(graph, "true_div_softmax"));
  EXPECT_TRUE(GraphHasNode(graph, "true_div"));
  EXPECT_TRUE(GraphHasNode(graph, "mid_broadcast"));
}

TEST(SoftmaxPatternFusionUtilsTest, NormalizeDirectPostSoftmaxIgnoresUnrelatedSource) {
  af::AscGraph graph("softmax_pattern_utils_ut_graph");
  SoftmaxGraphHandle handle;
  BuildIndirectLoadSoftmaxGraph(graph, SoftmaxSourceKind::kUnrelated, handle);
  ASSERT_NE(handle.indirect_load, nullptr);
  bool changed = true;
  ASSERT_EQ(optimize::softmax_pattern::NormalizeDirectPostSoftmax(graph, handle.indirect_load, changed), af::SUCCESS);
  EXPECT_FALSE(changed);
  EXPECT_FALSE(GraphHasNode(graph, "true_div_softmax"));
  EXPECT_TRUE(GraphHasNode(graph, "true_div"));
}
