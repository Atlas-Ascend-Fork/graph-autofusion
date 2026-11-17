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

#include "../test_util.h"
namespace af {
namespace ascir {
extern std::vector<std::unique_ptr<af::TmpBufDesc>> CalcReduceTmpSize(const af::AscNode &node);

using namespace testing;
using namespace af::ascir_op;

class CalcReduceTmpSizeTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};
template <af::DataType T>
void CreateGraphReduce(af::AscGraph &graph, Expression &s1, Expression &s2) {
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
}

template <af::DataType T>
void CreateGraphReduceAcc(af::AscGraph &graph, Expression &s1, Expression &s2) {
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
}
/**
 * @tc.name: CalcReduceTmpSize_test_0
 * @tc.number: CalcReduceTmpSize_Test_001
 * @tc.desc: Test when node is valid then CalcReduceTmpSize returns correct size
 */
TEST_F(CalcReduceTmpSizeTest, CalcReduceTmpSize_test_0) {
  af::AscGraph graph("testx");
  af::Expression One = af::Symbol(1);
  af::Expression Zero = af::Symbol(0);
  Expression s1;
  Expression s2;
  CreateGraphReduce<af::DT_FLOAT>(graph, s1, s2);
  std::shared_ptr<af::AscNode> node = graph.FindNode("max0");
  node->inputs[0].attr.vectorized_strides = {s2, One};
  node->outputs[0].attr.vectorized_strides = {One, Zero};
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcReduceTmpSize(*node);
  ASSERT_EQ(result.size(), 2);
}

/**
 * @tc.name: CalcReduceTmpSize_test_1
 * @tc.number: CalcReduceTmpSize_Test_002
 * @tc.desc: Test when node is valid then CalcReduceTmpSize returns correct size
 */
TEST_F(CalcReduceTmpSizeTest, CalcReduceTmpSize_test_1) {
  af::AscGraph graph("testx");
  af::Expression One = af::Symbol(1);
  af::Expression Zero = af::Symbol(0);
  Expression s1;
  Expression s2;
  CreateGraphReduceAcc<af::DT_FLOAT>(graph, s1, s2);
  std::shared_ptr<af::AscNode> node = graph.FindNode("prod0");
  node->inputs[0].attr.vectorized_strides = {s2, One};
  node->outputs[0].attr.vectorized_strides = {One, Zero};
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcReduceTmpSize(*node);
  ASSERT_EQ(result.size(), 2);
}

// 复制自 CreateGraphReduce，算子换成 Argmax（覆盖 CalcReduceTmpSize 的 Argmax 分支）
template <af::DataType T>
void CreateGraphArgmax(af::AscGraph &graph, Expression &s1, Expression &s2) {
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
  af::ascir_op::Argmax argmax0("argmax0");
  Store store("store");
  Output y("y");

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

  argmax0.x = load1.y;
  argmax0.attr.sched.axis = {z0.id, z1.id, z2.id};
  argmax0.attr.sched.loop_axis = {z0.id};
  argmax0.y.dtype = T;
  *argmax0.y.axis = {z0.id, z1.id, z2.id};
  *argmax0.y.repeats = {s0, s1, One};
  *argmax0.y.strides = {s2, One, Zero};
  *argmax0.y.vectorized_axis = {z1.id, z2.id};

  store.x = argmax0.y;
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

// 复制自 CreateGraphReduce，算子换成 ArgMaxMultiRPhase1（覆盖对应分支）
template <af::DataType T>
void CreateGraphArgMaxPhase1(af::AscGraph &graph, Expression &s1, Expression &s2) {
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
  af::ascir_op::ArgMaxMultiRPhase1 argmaxp1("argmaxp1");
  Store store("store");
  Output y("y");

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

  argmaxp1.x = load1.y;
  argmaxp1.attr.sched.axis = {z0.id, z1.id, z2.id};
  argmaxp1.attr.sched.loop_axis = {z0.id};
  argmaxp1.value.dtype = T;
  *argmaxp1.value.axis = {z0.id, z1.id, z2.id};
  *argmaxp1.value.repeats = {s0, s1, One};
  *argmaxp1.value.strides = {s2, One, Zero};
  *argmaxp1.value.vectorized_axis = {z1.id, z2.id};

  store.x = argmaxp1.value;
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

// Argmax 型：返回 4 个 tmp buf，生命周期序列 -1,0,1,2 是该分支独有指纹
TEST_F(CalcReduceTmpSizeTest, CalcReduceTmpSize_WhenNodeTypeIsArgmax) {
  af::AscGraph graph("test_argmax");
  af::Expression One = af::Symbol(1);
  af::Expression Zero = af::Symbol(0);
  Expression s1;
  Expression s2;
  CreateGraphArgmax<af::DT_FLOAT>(graph, s1, s2);
  std::shared_ptr<af::AscNode> node = graph.FindNode("argmax0");
  node->inputs[0].attr.vectorized_strides = {s2, One};
  node->outputs[0].attr.vectorized_strides = {One, Zero};
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcReduceTmpSize(*node);
  ASSERT_EQ(result.size(), 4U);
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
  ASSERT_EQ(result[1]->life_time_axis_id, 0);
  ASSERT_EQ(result[2]->life_time_axis_id, 1);
  ASSERT_EQ(result[3]->life_time_axis_id, 2);
}

// ArgMaxMultiRPhase1 型：返回 3 个 tmp buf，生命周期序列 -1,0,1
TEST_F(CalcReduceTmpSizeTest, CalcReduceTmpSize_WhenNodeTypeIsArgMaxMultiRPhase1) {
  af::AscGraph graph("test_argmax_p1");
  af::Expression One = af::Symbol(1);
  af::Expression Zero = af::Symbol(0);
  Expression s1;
  Expression s2;
  CreateGraphArgMaxPhase1<af::DT_FLOAT>(graph, s1, s2);
  std::shared_ptr<af::AscNode> node = graph.FindNode("argmaxp1");
  node->inputs[0].attr.vectorized_strides = {s2, One};
  node->outputs[0].attr.vectorized_strides = {One, Zero};
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcReduceTmpSize(*node);
  ASSERT_EQ(result.size(), 3U);
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
  ASSERT_EQ(result[1]->life_time_axis_id, 0);
  ASSERT_EQ(result[2]->life_time_axis_id, 1);
}
}  // namespace ascir
}  // namespace af
