/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"

#include "graph/operator_reg_af.h"
#include "graph_utils_ex.h"
#include "node_utils.h"
#include "op_desc_utils.h"

#include "ascir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"

#include "reduce_reuse_utils.h"

namespace af {
namespace ascir {
extern std::vector<std::unique_ptr<af::TmpBufDesc>> CalcReduceTmpSizeV2(const af::AscNode &node);

using namespace testing;
using namespace af::ascir_op;

class CalcReduceTmpSizeV2Test : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

template <af::DataType T>
void CreateGraphReduceMultiBranchBroadcast(af::AscGraph &graph, Expression &s1, Expression &s2,
                                           bool second_branch_has_broadcast) {
  af::Expression One = af::Symbol(1);
  af::Expression Zero = af::Symbol(0);
  auto s0 = graph.CreateSizeVar("s0");
  s1 = graph.CreateSizeVar("s1");
  s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x1("x1", graph);
  Data x2("x2", graph);
  Load load1("load1");
  Load load2("load2");
  af::ascir_op::Broadcast broadcast1("broadcast1");
  af::ascir_op::Broadcast broadcast2("broadcast2");
  af::ascir_op::Add add0("add0");
  af::ascir_op::Max max0("max0");
  Store store("store");
  Output y("y");

  graph.AddNode(load1);
  graph.AddNode(load2);
  graph.AddNode(broadcast1);
  if (second_branch_has_broadcast) {
    graph.AddNode(broadcast2);
  }
  graph.AddNode(add0);
  graph.AddNode(max0);
  graph.AddNode(store);

  x1.attr.sched.axis = {z0.id, z1.id, z2.id};
  x1.y.dtype = T;
  *x1.y.axis = {z0.id, z1.id, z2.id};
  *x1.y.repeats = {s0, s1, s2};
  *x1.y.strides = {s1 * s2, s2, One};

  x2.attr.sched.axis = {z0.id, z1.id, z2.id};
  x2.y.dtype = T;
  *x2.y.axis = {z0.id, z1.id, z2.id};
  *x2.y.repeats = {s0, s1, s2};
  *x2.y.strides = {s1 * s2, s2, One};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.y.dtype = T;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, One};
  *load1.y.vectorized_axis = {z1.id, z2.id};
  *load1.y.vectorized_strides = {s2, Zero};

  broadcast1.x = {load1.y};
  broadcast1.attr.sched.axis = {z0.id, z1.id, z2.id};
  broadcast1.y.dtype = T;
  *broadcast1.y.axis = {z0.id, z1.id, z2.id};
  *broadcast1.y.repeats = {s0, s1, s2};
  *broadcast1.y.strides = {s1 * s2, s2, One};
  *broadcast1.y.vectorized_axis = {z1.id, z2.id};
  *broadcast1.y.vectorized_strides = {s2, One};

  load2.x = x2.y;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  load2.y.dtype = T;
  *load2.y.axis = {z0.id, z1.id, z2.id};
  *load2.y.repeats = {s0, s1, s2};
  *load2.y.strides = {s1 * s2, s2, One};
  *load2.y.vectorized_axis = {z1.id, z2.id};
  *load2.y.vectorized_strides = {s2, second_branch_has_broadcast ? Zero : One};

  if (second_branch_has_broadcast) {
    broadcast2.x = {load2.y};
    broadcast2.attr.sched.axis = {z0.id, z1.id, z2.id};
    broadcast2.y.dtype = T;
    *broadcast2.y.axis = {z0.id, z1.id, z2.id};
    *broadcast2.y.repeats = {s0, s1, s2};
    *broadcast2.y.strides = {s1 * s2, s2, One};
    *broadcast2.y.vectorized_axis = {z1.id, z2.id};
    *broadcast2.y.vectorized_strides = {s2, One};
    add0.x2 = broadcast2.y;
  } else {
    add0.x2 = load2.y;
  }

  add0.x1 = broadcast1.y;
  add0.attr.sched.axis = {z0.id, z1.id, z2.id};
  add0.y.dtype = T;
  *add0.y.axis = {z0.id, z1.id, z2.id};
  *add0.y.repeats = {s0, s1, s2};
  *add0.y.strides = {s1 * s2, s2, One};
  *add0.y.vectorized_axis = {z1.id, z2.id};
  *add0.y.vectorized_strides = {s2, One};

  max0.x = add0.y;
  max0.attr.sched.axis = {z0.id, z1.id, z2.id};
  max0.attr.sched.loop_axis = {z0.id};
  max0.y.dtype = T;
  *max0.y.axis = {z0.id, z1.id, z2.id};
  *max0.y.repeats = {s0, s1, One};
  *max0.y.strides = {s2, One, Zero};
  *max0.y.vectorized_axis = {z1.id, z2.id};
  *max0.y.vectorized_strides = {One, Zero};

  store.x = max0.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.y.dtype = T;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1 * s2, s2, One};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id};
  y.y.dtype = T;
  *y.y.axis = {z0.id, z1.id, z2.id};
  *y.y.repeats = {s0, s1, s2};
  *y.y.strides = {s1 * s2, s2, One};
}
template <af::DataType T>
void CreateGraphReduceWithInputMultiRefs(af::AscGraph &graph, Expression &s1, Expression &s2) {
  af::Expression One = af::Symbol(1);
  af::Expression Zero = af::Symbol(0);
  auto s0 = graph.CreateSizeVar("s0");
  s1 = graph.CreateSizeVar("s1");
  s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x1("x1", graph);
  Load load1("load1");
  af::ascir_op::Max max0("max0");
  Store store("store");
  Output y("y");
  Store store1("store1");
  Output y1("y1");

  x1.attr.sched.axis = {z0.id, z1.id, z2.id};
  x1.y.dtype = T;
  *x1.y.axis = {z0.id, z1.id, z2.id};
  *x1.y.repeats = {s0, s1, s2};
  *x1.y.strides = {s1 * s2, s2, One};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.y.dtype = T;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, One};
  *load1.y.vectorized_axis = {z1.id, z2.id};

  max0.x = load1.y;
  max0.attr.sched.axis = {z0.id, z1.id, z2.id};
  max0.attr.sched.loop_axis = {z0.id};
  max0.y.dtype = T;
  *max0.y.axis = {z0.id, z1.id, z2.id};
  *max0.y.repeats = {s0, s1, One};
  *max0.y.strides = {s2, One, Zero};
  *max0.y.vectorized_axis = {z1.id, z2.id};

  store.x = max0.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.y.dtype = T;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1 * s2, s2, One};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id};
  y.y.dtype = T;
  *y.y.axis = {z0.id, z1.id, z2.id};
  *y.y.repeats = {s0, s1, s2};
  *y.y.strides = {s1 * s2, s2, One};

  store1.x = load1.y;
  store1.attr.sched.axis = {z0.id, z1.id, z2.id};
  store1.y.dtype = T;
  *store1.y.axis = {z0.id, z1.id, z2.id};
  *store1.y.repeats = {s0, s1, s2};
  *store1.y.strides = {s1 * s2, s2, One};

  y1.x = store1.y;
  y1.attr.sched.axis = {z0.id, z1.id, z2.id};
  y1.y.dtype = T;
  *y1.y.axis = {z0.id, z1.id, z2.id};
  *y1.y.repeats = {s0, s1, s2};
  *y1.y.strides = {s1 * s2, s2, One};
}

template <af::DataType T>
void CreateGraphReduceAccWithInputMultiRefs(af::AscGraph &graph, Expression &s1, Expression &s2) {
  af::Expression One = af::Symbol(1);
  af::Expression Zero = af::Symbol(0);
  auto s0 = graph.CreateSizeVar("s0");
  s1 = graph.CreateSizeVar("s1");
  s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x1("x1", graph);
  Load load1("load1");
  af::ascir_op::Prod prod0("prod0");
  Store store("store");
  Output y("y");
  Store store1("store1");
  Output y1("y1");

  x1.attr.sched.axis = {z0.id, z1.id, z2.id};
  x1.y.dtype = T;
  *x1.y.axis = {z0.id, z1.id, z2.id};
  *x1.y.repeats = {s0, s1, s2};
  *x1.y.strides = {s1 * s2, s2, One};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.y.dtype = T;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, One};
  *load1.y.vectorized_axis = {z1.id, z2.id};

  prod0.x = load1.y;
  prod0.attr.sched.axis = {z0.id, z1.id, z2.id};
  prod0.attr.sched.loop_axis = {z0.id};
  prod0.y.dtype = T;
  *prod0.y.axis = {z0.id, z1.id, z2.id};
  *prod0.y.repeats = {s0, s1, One};
  *prod0.y.strides = {s2, One, Zero};
  *prod0.y.vectorized_axis = {z1.id, z2.id};

  store.x = prod0.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.y.dtype = T;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1 * s2, s2, One};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id};
  y.y.dtype = T;
  *y.y.axis = {z0.id, z1.id, z2.id};
  *y.y.repeats = {s0, s1, s2};
  *y.y.strides = {s1 * s2, s2, One};

  store1.x = load1.y;
  store1.attr.sched.axis = {z0.id, z1.id, z2.id};
  store1.y.dtype = T;
  *store1.y.axis = {z0.id, z1.id, z2.id};
  *store1.y.repeats = {s0, s1, s2};
  *store1.y.strides = {s1 * s2, s2, One};

  y1.x = store1.y;
  y1.attr.sched.axis = {z0.id, z1.id, z2.id};
  y1.y.dtype = T;
  *y1.y.axis = {z0.id, z1.id, z2.id};
  *y1.y.repeats = {s0, s1, s2};
  *y1.y.strides = {s1 * s2, s2, One};
}

TEST_F(CalcReduceTmpSizeV2Test, CalcReduceTmpSizeV2_test_0) {
  af::AscGraph graph("testx");
  af::Expression One = af::Symbol(1);
  af::Expression Zero = af::Symbol(0);
  Expression s1;
  Expression s2;
  CreateGraphReduceWithInputMultiRefs<af::DT_FLOAT>(graph, s1, s2);
  std::shared_ptr<af::AscNode> node = graph.FindNode("max0");
  node->inputs[0].attr.vectorized_strides = {s2, One};
  node->outputs[0].attr.vectorized_strides = {One, Zero};
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcReduceTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 2);
}

TEST_F(CalcReduceTmpSizeV2Test, CalcReduceTmpSize_test_1) {
  af::AscGraph graph("testx");
  af::Expression One = af::Symbol(1);
  af::Expression Zero = af::Symbol(0);
  Expression s1;
  Expression s2;
  CreateGraphReduceAccWithInputMultiRefs<af::DT_FLOAT>(graph, s1, s2);
  std::shared_ptr<af::AscNode> node = graph.FindNode("prod0");
  node->inputs[0].attr.vectorized_strides = {s2, One};
  node->outputs[0].attr.vectorized_strides = {One, Zero};
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcReduceTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 2);
}

TEST_F(CalcReduceTmpSizeV2Test, ArgmaxInt64ValueBuffersUseEightBytes) {
  for (const std::string type : {"ArgmaxWithValue", "ArgMaxMultiRPhase1"}) {
    af::AscGraph graph("argmax_int64_tmp");
    Expression s1;
    Expression s2;
    CreateGraphReduceWithInputMultiRefs<af::DT_INT64>(graph, s1, s2);
    auto node = graph.FindNode("max0");
    node->GetOpDesc()->SetType(type);
    node->inputs[0].attr.vectorized_strides = {s2, af::Symbol(1)};
    node->outputs[0].attr.vectorized_strides = {af::Symbol(1), af::Symbol(0)};
    const auto result = CalcReduceTmpSizeV2(*node);
    const af::TmpBufDesc *index_tmp = nullptr;
    const af::TmpBufDesc *value_tmp = nullptr;
    for (const auto &desc : result) {
      if (desc->life_time_axis_id == 0) {
        index_tmp = desc.get();
      } else if (desc->life_time_axis_id == 1) {
        value_tmp = desc.get();
      }
    }
    ASSERT_NE(index_tmp, nullptr);
    ASSERT_NE(value_tmp, nullptr);
    EXPECT_EQ(value_tmp->size, index_tmp->size * af::Symbol(2));
  }
}

TEST_F(CalcReduceTmpSizeV2Test, ProdInt64TmpBuffersUseInputDtypeSize) {
  af::AscGraph graph("prod_int64_tmp");
  Expression s1;
  Expression s2;
  CreateGraphReduceAccWithInputMultiRefs<af::DT_INT64>(graph, s1, s2);
  auto node = graph.FindNode("prod0");
  node->inputs[0].attr.vectorized_strides = {s2, af::Symbol(1)};
  node->outputs[0].attr.vectorized_strides = {af::Symbol(1), af::Symbol(0)};
  const auto result = CalcReduceTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 2U);

  const auto expected_api_size = sym::Align(s2, 8) * s1 * af::Symbol(8) + af::Symbol(256) + af::Symbol(32);
  const auto expected_a_size = sym::Align(s1, 8) * af::Symbol(8);
  EXPECT_TRUE(sym::AssertSymbolEq(result[0]->size, expected_api_size, __FILE__, __LINE__));
  EXPECT_TRUE(sym::AssertSymbolEq(result[1]->size, expected_a_size, __FILE__, __LINE__));
}

TEST_F(CalcReduceTmpSizeV2Test, HasUpstreamBroadcastOnReduceAxis_ReturnsTrue_WhenAllBranchesHaveBroadcast) {
  af::AscGraph graph("test_all_broadcast");
  Expression s1;
  Expression s2;
  CreateGraphReduceMultiBranchBroadcast<af::DT_FLOAT>(graph, s1, s2, true);
  std::shared_ptr<af::AscNode> node = graph.FindNode("max0");
  node->inputs[0].attr.vectorized_axis = {node->inputs[0].attr.axis[1], node->inputs[0].attr.axis[2]};
  node->inputs[0].attr.vectorized_strides = {s2, af::Symbol(1)};
  node->outputs[0].attr.vectorized_axis = {node->outputs[0].attr.axis[1], node->outputs[0].attr.axis[2]};
  node->outputs[0].attr.vectorized_strides = {af::Symbol(1), af::Symbol(0)};
  EXPECT_TRUE(HasUpstreamBroadcastOnReduceAxis(*node));
}

TEST_F(CalcReduceTmpSizeV2Test, HasUpstreamBroadcastOnReduceAxis_ReturnsFalse_WhenAnyBranchMissesBroadcast) {
  af::AscGraph graph("test_missing_broadcast");
  Expression s1;
  Expression s2;
  CreateGraphReduceMultiBranchBroadcast<af::DT_FLOAT>(graph, s1, s2, false);
  std::shared_ptr<af::AscNode> node = graph.FindNode("max0");
  node->inputs[0].attr.vectorized_axis = {node->inputs[0].attr.axis[1], node->inputs[0].attr.axis[2]};
  node->inputs[0].attr.vectorized_strides = {s2, af::Symbol(1)};
  node->outputs[0].attr.vectorized_axis = {node->outputs[0].attr.axis[1], node->outputs[0].attr.axis[2]};
  node->outputs[0].attr.vectorized_strides = {af::Symbol(1), af::Symbol(0)};
  EXPECT_FALSE(HasUpstreamBroadcastOnReduceAxis(*node));
}

}  // namespace ascir
}  // namespace af
