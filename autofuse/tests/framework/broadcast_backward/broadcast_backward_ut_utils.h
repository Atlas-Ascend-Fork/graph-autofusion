/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AUTOFUSE_TESTS_FRAMEWORK_BROADCAST_BACKWARD_UT_UTILS_H_
#define AUTOFUSE_TESTS_FRAMEWORK_BROADCAST_BACKWARD_UT_UTILS_H_

#include "gtest/gtest.h"

#include <string>
#include <vector>

#include "asc_graph_builder.h"
#include "graph_utils.h"
#include "optimize/graph_pass/broadcast_backward_pass.h"
#include "tests/framework/broadcast_backward/broadcast_backward_test_utils.h"

namespace broadcast_backward_test {

inline const std::vector<af::Expression> kCompactRepeats = {af::testing::Sym("s0"), af::sym::kSymbolOne};
inline const std::vector<af::Expression> kCompactStrides = {af::sym::kSymbolOne, af::sym::kSymbolZero};

inline bool IsConnected(af::AscGraph &graph, const char *src_name, const char *dst_name) {
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetName() != src_name) {
      continue;
    }
    const auto out_anchor = node->GetOutDataAnchor(0);
    if (out_anchor == nullptr) {
      return false;
    }
    for (const auto &peer : out_anchor->GetPeerInDataAnchors()) {
      if (peer != nullptr && peer->GetOwnerNode()->GetName() == dst_name) {
        return true;
      }
    }
  }
  return false;
}

inline bool AreExpressionVectorsEqual(const std::vector<af::Expression> &lhs, const std::vector<af::Expression> &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t index = 0U; index < lhs.size(); ++index) {
    if (af::SymbolicUtils::StaticCheckEq(lhs[index], rhs[index]) != af::TriBool::kTrue) {
      return false;
    }
  }
  return true;
}

inline bool AreConnectedTensorAttrsEqual(const af::AscNodePtr &source_node, const af::AscNodePtr &destination_node,
                                         size_t destination_input_index) {
  if (source_node == nullptr || destination_node == nullptr || source_node->GetOpDesc() == nullptr ||
      destination_node->GetOpDesc() == nullptr ||
      destination_input_index >= destination_node->GetAllInDataAnchorsSize()) {
    return false;
  }
  const auto source_anchor = source_node->GetOutDataAnchor(0);
  const auto destination_anchor = destination_node->GetInDataAnchor(static_cast<int32_t>(destination_input_index));
  return source_anchor != nullptr && destination_anchor != nullptr &&
         destination_anchor->GetPeerOutAnchor() == source_anchor &&
         source_node->GetOpDesc()->GetOutputDesc(0U).GetDataType() ==
             destination_node->GetOpDesc()->GetInputDesc(static_cast<uint32_t>(destination_input_index)).GetDataType();
}

inline bool IsEdgeAttrConsistent(af::AscGraph &graph, const char *src_name, const char *dst_name) {
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetName() != src_name || node->GetOutDataAnchor(0) == nullptr) {
      continue;
    }
    for (const auto &peer : node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      if (peer == nullptr || peer->GetOwnerNode()->GetName() != dst_name) {
        continue;
      }
      const auto destination_node = std::dynamic_pointer_cast<af::AscNode>(peer->GetOwnerNode());
      const auto source_node = std::dynamic_pointer_cast<af::AscNode>(node);
      return AreConnectedTensorAttrsEqual(source_node, destination_node, static_cast<size_t>(peer->GetIdx()));
    }
  }
  return false;
}

inline bool HasNode(af::AscGraph &graph, const char *node_name) {
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetName() == node_name) {
      return true;
    }
  }
  return false;
}

inline void ExpectStaticEq(const std::vector<af::Expression> &actual, const std::vector<af::Expression> &expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t index = 0U; index < actual.size(); ++index) {
    EXPECT_EQ(af::SymbolicUtils::StaticCheckEq(actual[index], expected[index]), af::TriBool::kTrue);
  }
}

inline void ExpectResidualConnections(af::AscGraph &graph) {
  EXPECT_TRUE(IsConnected(graph, "load0", "broadcast0_residual_0"));
  EXPECT_TRUE(IsConnected(graph, "broadcast0_residual_0", "merge"));
  EXPECT_TRUE(IsConnected(graph, "load1", "broadcast1_residual_1"));
  EXPECT_TRUE(IsConnected(graph, "broadcast1_residual_1", "merge"));
}

inline af::AscGraph BuildUnaryGraph(const std::string &op_name) {
  af::testing::AscGraphBuilder builder("broadcast_backward_unary_" + op_name);
  builder.Loops({af::testing::Sym("s0"), af::testing::Sym("s1")})
      .Data("data", 0)
      .Load("load", "data", kCompactRepeats, kCompactStrides)
      .Broadcast("broadcast", "load", {af::testing::Sym("s0"), af::testing::Sym("s1")});
  if (op_name == "Abs") {
    builder.Abs("compute", "broadcast");
  } else if (op_name == "Neg") {
    builder.Neg("compute", "broadcast");
  } else if (op_name == "Exp") {
    builder.Exp("compute", "broadcast");
  } else if (op_name == "Sqrt") {
    builder.Sqrt("compute", "broadcast");
  } else if (op_name == "Relu") {
    builder.Relu("compute", "broadcast");
  } else if (op_name == "Rsqrt") {
    builder.Op<af::ascir_op::Rsqrt>("compute", {"broadcast"});
  } else if (op_name == "Reciprocal") {
    builder.Op<af::ascir_op::Reciprocal>("compute", {"broadcast"});
  } else if (op_name == "Erf") {
    builder.Op<af::ascir_op::Erf>("compute", {"broadcast"});
  } else if (op_name == "Sign") {
    builder.Op<af::ascir_op::Sign>("compute", {"broadcast"});
  } else if (op_name == "Tanh") {
    builder.Op<af::ascir_op::Tanh>("compute", {"broadcast"});
  } else if (op_name == "Ln") {
    builder.Op<af::ascir_op::Ln>("compute", {"broadcast"});
  } else {
    ADD_FAILURE() << "Unsupported unary test operator: " << op_name;
  }
  return builder.Store("store", "compute").Output("output", "store").Build();
}

inline af::AscGraph BuildBinaryGraph(const std::string &op_name) {
  af::testing::AscGraphBuilder builder("broadcast_backward_binary_" + op_name);
  builder.Loops({af::testing::Sym("s0"), af::testing::Sym("s1")})
      .Data("data0", 0)
      .Data("data1", 1)
      .Load("load0", "data0", kCompactRepeats, kCompactStrides)
      .Load("load1", "data1", kCompactRepeats, kCompactStrides)
      .Broadcast("broadcast0", "load0", {af::testing::Sym("s0"), af::testing::Sym("s1")})
      .Broadcast("broadcast1", "load1", {af::testing::Sym("s0"), af::testing::Sym("s1")});
  if (op_name == "Add") {
    builder.Add("compute", "broadcast0", "broadcast1");
  } else if (op_name == "Sub") {
    builder.Sub("compute", "broadcast0", "broadcast1");
  } else if (op_name == "Mul") {
    builder.Mul("compute", "broadcast0", "broadcast1");
  } else if (op_name == "Div") {
    builder.Div("compute", "broadcast0", "broadcast1");
  } else if (op_name == "Minimum") {
    builder.Minimum("compute", "broadcast0", "broadcast1");
  } else if (op_name == "Maximum") {
    builder.Maximum("compute", "broadcast0", "broadcast1");
  } else {
    ADD_FAILURE() << "Unsupported binary test operator: " << op_name;
  }
  return builder.Store("store", "compute").Output("output", "store").Build();
}

inline af::AscGraph BuildDtypeAwareBinaryGraph(const std::string &op_name) {
  af::testing::AscGraphBuilder builder("broadcast_backward_dtype_aware_" + op_name);
  builder.Loops({af::testing::Sym("s0"), af::testing::Sym("s1")})
      .Data("data0", 0)
      .Data("data1", 1)
      .Load("load0", "data0", kCompactRepeats, kCompactStrides)
      .Load("load1", "data1", kCompactRepeats, kCompactStrides)
      .Broadcast("broadcast0", "load0", {af::testing::Sym("s0"), af::testing::Sym("s1")})
      .Broadcast("broadcast1", "load1", {af::testing::Sym("s0"), af::testing::Sym("s1")});
  if (op_name == "Eq") {
    builder.Op<af::ascir_op::Eq>("compute", {"broadcast0", "broadcast1"});
  } else if (op_name == "TrueDiv") {
    builder.Op<af::ascir_op::TrueDiv>("compute", {"broadcast0", "broadcast1"});
  } else {
    ADD_FAILURE() << "Unsupported dtype-aware binary test operator: " << op_name;
  }
  return builder.Store("store", "compute").Output("output", "store").Build();
}

inline void ExpectBinaryBroadcastMove(af::AscGraph &graph) {
  EXPECT_TRUE(IsConnected(graph, "load0", "compute"));
  EXPECT_TRUE(IsConnected(graph, "load1", "compute"));
  EXPECT_TRUE(IsConnected(graph, "compute", "broadcast0"));
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "store"));
  EXPECT_FALSE(HasNode(graph, "broadcast1"));
}

inline void ExpectDtypeAwareBinaryMove(af::AscGraph &graph, const std::string &op_name) {
  CompleteApiInfo(graph);
  optimize::BroadcastBackwardPass pass;
  ASSERT_EQ(pass.RunPass(graph), af::SUCCESS) << op_name;
  EXPECT_TRUE(IsConnected(graph, "load0", "compute"));
  EXPECT_TRUE(IsConnected(graph, "load1", "compute"));
  EXPECT_TRUE(IsConnected(graph, "compute", "broadcast0"));
  EXPECT_TRUE(IsConnected(graph, "broadcast0", "store"));
  EXPECT_FALSE(HasNode(graph, "broadcast1"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "load0", "compute"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "load1", "compute"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "compute", "broadcast0"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "broadcast0", "store"));
}

inline void ExpectCommonAxisLayouts(af::AscGraph &graph, const std::vector<af::Expression> &compact_repeats,
                                    const std::vector<af::Expression> &expanded_repeats,
                                    const std::vector<af::Expression> &compact_strides,
                                    const std::vector<af::Expression> &expanded_strides) {
  const auto merge = FindNode(graph, "merge");
  const auto common = FindNode(graph, "merge_broadcast_backward_common");
  const auto residual0 = FindNode(graph, "broadcast0_residual_0");
  const auto residual1 = FindNode(graph, "broadcast1_residual_1");
  ASSERT_NE(merge, nullptr);
  ASSERT_NE(common, nullptr);
  ASSERT_NE(residual0, nullptr);
  ASSERT_NE(residual1, nullptr);
  ExpectStaticEq(merge->inputs[0].attr.repeats, compact_repeats);
  ExpectStaticEq(merge->inputs[1].attr.repeats, compact_repeats);
  ExpectStaticEq(merge->outputs[0].attr.repeats, compact_repeats);
  ExpectStaticEq(common->outputs[0].attr.repeats, expanded_repeats);
  ExpectStaticEq(residual0->outputs[0].attr.strides, compact_strides);
  ExpectStaticEq(residual1->outputs[0].attr.strides, compact_strides);
  ExpectStaticEq(merge->outputs[0].attr.strides, compact_strides);
  ExpectStaticEq(common->outputs[0].attr.strides, expanded_strides);
}

inline af::AscGraph BuildDirectFanOutGraph(const std::string &name) {
  const auto s0 = af::testing::Sym("s0");
  const auto s1 = af::testing::Sym("s1");
  return af::testing::AscGraphBuilder(name)
      .Loops({s0, s1})
      .Data("data", 0)
      .Load("load", "data", kCompactRepeats, kCompactStrides)
      .Broadcast("broadcast", "load", {1})
      .Abs("branch0", "broadcast")
      .Neg("branch1", "broadcast")
      .Add("merge", "branch0", "branch1")
      .Store("store", "merge")
      .Output("output", "store")
      .Build();
}

inline af::AscGraph BuildMultiNodeFanOutGraph(const std::string &name) {
  const auto s0 = af::testing::Sym("s0");
  const auto s1 = af::testing::Sym("s1");
  return af::testing::AscGraphBuilder(name)
      .Loops({s0, s1})
      .Data("data", 0)
      .Load("load", "data", kCompactRepeats, kCompactStrides)
      .Broadcast("broadcast", "load", {1})
      .Abs("branch0_head", "broadcast")
      .Relu("branch0_tail", "branch0_head")
      .Neg("branch1_head", "broadcast")
      .Exp("branch1_tail", "branch1_head")
      .Add("merge", "branch0_tail", "branch1_tail")
      .Store("store", "merge")
      .Output("output", "store")
      .Build();
}

inline af::AscGraph BuildScalarForkJoinGraph(const std::string &name) {
  const auto s0 = af::testing::Sym("s0");
  const auto s1 = af::testing::Sym("s1");
  return af::testing::AscGraphBuilder(name)
      .Loops({s0, s1})
      .Scalar("scalar", "1.0")
      .Broadcast("broadcast", "scalar", {s0, s1})
      .Abs("left", "broadcast")
      .Neg("right", "broadcast")
      .Add("merge", "left", "right")
      .Store("store", "merge")
      .Output("output", "store")
      .Build();
}

inline af::AscGraph BuildSharedDtypeAwareFanOutGraph(const std::string &name) {
  const auto s0 = af::testing::Sym("s0");
  const auto s1 = af::testing::Sym("s1");
  const auto s2 = af::testing::Sym("s2");
  const std::vector<af::Expression> compact = {s0, af::sym::kSymbolOne, s2};
  const std::vector<af::Expression> strides = {s2, af::sym::kSymbolZero, af::sym::kSymbolOne};
  return af::testing::AscGraphBuilder(name)
      .Loops({s0, s1, s2})
      .Data("data", 0, af::DT_INT32)
      .Load("load", "data", compact, strides)
      .Broadcast("broadcast", "load", {1})
      .Abs("abs", "broadcast")
      .Cast("left_cast", "abs", af::DT_FLOAT)
      .Cast("right_cast", "broadcast", af::DT_FLOAT)
      .Relu("relu", "right_cast")
      .Add("add", "relu", "left_cast")
      .Sqrt("sqrt", "add")
      .Op<af::ascir_op::Sigmoid>("sigmoid", {"sqrt"})
      .Store("store", "sigmoid")
      .Output("output", "store")
      .Build();
}

inline void CompleteSharedDtypeAwareFanOutGraph(af::AscGraph &graph) {
  CompleteApiInfo(graph);
  for (const auto *node_name : {"load", "broadcast", "abs"}) {
    SetNodeDtype(graph, node_name, af::DT_INT32);
  }
  for (const auto *node_name : {"left_cast", "right_cast"}) {
    const auto cast = FindNode(graph, node_name);
    ASSERT_NE(cast, nullptr);
    cast->GetOpDesc()->MutableInputDesc(0U)->SetDataType(af::DT_INT32);
  }
  for (const auto *node_name : {"relu", "add", "sqrt", "sigmoid", "store"}) {
    SetNodeDtype(graph, node_name, af::DT_FLOAT);
  }
}

inline void ExpectDirectFanOutCandidate(af::AscGraph &graph) {
  const auto broadcast = FindNode(graph, "broadcast");
  const auto branch0 = FindNode(graph, "branch0");
  const auto branch1 = FindNode(graph, "branch1");
  const auto merge = FindNode(graph, "merge");
  ASSERT_NE(broadcast, nullptr);
  ASSERT_NE(branch0, nullptr);
  ASSERT_NE(branch1, nullptr);
  ASSERT_NE(merge, nullptr);
  ASSERT_EQ(broadcast->GetOutDataAnchor(0)->GetPeerInDataAnchors().size(), 2U);
  ASSERT_EQ(branch0->GetAllInDataAnchorsSize(), 1U);
  ASSERT_EQ(branch0->GetOutDataNodesSize(), 1U);
  ASSERT_EQ(branch1->GetAllInDataAnchorsSize(), 1U);
  ASSERT_EQ(branch1->GetOutDataNodesSize(), 1U);
  ASSERT_EQ(merge->GetAllInDataAnchorsSize(), 2U);
  ASSERT_EQ(merge->GetOutDataNodesSize(), 1U);
  ASSERT_TRUE(IsEdgeAttrConsistent(graph, "broadcast", "branch0"));
  ASSERT_TRUE(IsEdgeAttrConsistent(graph, "broadcast", "branch1"));
  ASSERT_TRUE(IsEdgeAttrConsistent(graph, "branch0", "merge"));
  ASSERT_TRUE(IsEdgeAttrConsistent(graph, "branch1", "merge"));
  ASSERT_TRUE(IsEdgeAttrConsistent(graph, "merge", "store"));
  ASSERT_TRUE(AreExpressionVectorsEqual(merge->outputs[0].attr.repeats, broadcast->outputs[0].attr.repeats));
  ASSERT_EQ(merge->outputs[0].attr.axis, broadcast->outputs[0].attr.axis);
  ASSERT_EQ(merge->outputs[0].attr.dtype, broadcast->outputs[0].attr.dtype);
  ASSERT_TRUE(AreExpressionVectorsEqual(merge->outputs[0].attr.strides, broadcast->outputs[0].attr.strides));
}

inline void ExpectDirectFanOutMoved(af::AscGraph &graph) {
  EXPECT_TRUE(IsConnected(graph, "load", "branch0"));
  EXPECT_TRUE(IsConnected(graph, "load", "branch1"));
  EXPECT_TRUE(IsConnected(graph, "branch0", "merge"));
  EXPECT_TRUE(IsConnected(graph, "branch1", "merge"));
  EXPECT_TRUE(IsConnected(graph, "merge", "broadcast"));
  EXPECT_TRUE(IsConnected(graph, "broadcast", "store"));
  const auto branch0 = FindNode(graph, "branch0");
  const auto branch1 = FindNode(graph, "branch1");
  const auto merge = FindNode(graph, "merge");
  ASSERT_NE(branch0, nullptr);
  ASSERT_NE(branch1, nullptr);
  ASSERT_NE(merge, nullptr);
  ExpectStaticEq(branch0->inputs[0].attr.repeats, kCompactRepeats);
  ExpectStaticEq(branch1->inputs[0].attr.repeats, kCompactRepeats);
  ExpectStaticEq(merge->outputs[0].attr.repeats, kCompactRepeats);
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "load", "branch0"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "load", "branch1"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "branch0", "merge"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "branch1", "merge"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "merge", "broadcast"));
  EXPECT_TRUE(IsEdgeAttrConsistent(graph, "broadcast", "store"));
}

// 多级广播链（load0(1,s1,1) → b1 扩轴0 → b2 扩轴2）+ 两个多输入消费者；消费者另一路为仅扩
// 轴 1 的广播（与链扩维轴 {0,2} 无公共轴，拆分后无公共轴部分后移），用于观测纯拆分形态。
inline af::AscGraph BuildMultiLevelChainFanOutGraph(const std::string &name) {
  const auto s0 = af::testing::Sym("s0");
  const auto s1 = af::testing::Sym("s1");
  const auto s2 = af::testing::Sym("s2");
  const auto one = af::sym::kSymbolOne;
  const auto zero = af::sym::kSymbolZero;
  return af::testing::AscGraphBuilder(name)
      .Loops({s0, s1, s2})
      .Data("data0", 0)
      .Data("dataX", 1)
      .Data("dataY", 2)
      .Load("load0", "data0", {one, s1, one}, {zero, one, zero})
      .Broadcast("b1", "load0", {0})
      .Broadcast("b2", "b1", {2})
      .Load("loadX", "dataX", {s0, one, s2}, {s2, zero, one})
      .Broadcast("bx", "loadX", {1})
      .Load("loadY", "dataY", {s0, one, s2}, {s2, zero, one})
      .Broadcast("by", "loadY", {1})
      .Add("add", "b2", "bx")
      .Sub("sub", "b2", "by")
      .Store("store1", "add")
      .Output("output1", "store1")
      .Store("store2", "sub")
      .Output("output2", "store2")
      .Build();
}

// 多级广播链（扩轴 {0,2}）+ 两个多输入消费者；消费者另一路为逐节点同轴同构广播链，
// 拆分后主循环可执行同轴兄弟删除并将整链沉过消费者。
inline af::AscGraph BuildMultiLevelChainSiblingGraph(const std::string &name) {
  const auto s0 = af::testing::Sym("s0");
  const auto s1 = af::testing::Sym("s1");
  const auto s2 = af::testing::Sym("s2");
  const auto one = af::sym::kSymbolOne;
  const auto zero = af::sym::kSymbolZero;
  const std::vector<af::Expression> compact = {one, s1, one};
  const std::vector<af::Expression> strides = {zero, one, zero};
  return af::testing::AscGraphBuilder(name)
      .Loops({s0, s1, s2})
      .Data("data0", 0)
      .Data("dataX", 1)
      .Data("dataY", 2)
      .Load("load0", "data0", compact, strides)
      .Broadcast("b1", "load0", {0})
      .Broadcast("b2", "b1", {2})
      .Load("loadX", "dataX", compact, strides)
      .Broadcast("x1", "loadX", {0})
      .Broadcast("x2", "x1", {2})
      .Load("loadY", "dataY", compact, strides)
      .Broadcast("y1", "loadY", {0})
      .Broadcast("y2", "y1", {2})
      .Add("add", "b2", "x2")
      .Mul("mul", "b2", "y2")
      .Store("store1", "add")
      .Output("output1", "store1")
      .Store("store2", "mul")
      .Output("output2", "store2")
      .Build();
}

}  // namespace broadcast_backward_test

#endif  // AUTOFUSE_TESTS_FRAMEWORK_BROADCAST_BACKWARD_UT_UTILS_H_
