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
extern std::vector<std::unique_ptr<af::TmpBufDesc>> CalcArgmaxTmpSize(const af::AscNode &node);

using namespace testing;
using namespace af::ascir_op;

class CalcArgmaxTmpSizeTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

template <ge::DataType T>
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
  argmax0.y.dtype = ge::DT_INT64;
  *argmax0.y.axis = {z0.id, z1.id, z2.id};
  *argmax0.y.repeats = {s0, s1, One};
  *argmax0.y.strides = {s2, One, Zero};
  *argmax0.y.vectorized_axis = {z1.id, z2.id};

  store.x = argmax0.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.y.dtype = ge::DT_INT64;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1 * s2, s2, One};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id};
  y.y.dtype = ge::DT_INT64;
  *y.y.axis = {z0.id, z1.id, z2.id};
  *y.y.repeats = {s0, s1, s2};
  *y.y.strides = {s1 * s2, s2, One};
}

template <ge::DataType T>
void CreateGraphArgmaxRA(af::AscGraph &graph, Expression &s1, Expression &s2) {
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
  argmax0.attr.sched.loop_axis = {z2.id};
  argmax0.y.dtype = ge::DT_INT64;
  *argmax0.y.axis = {z0.id, z1.id, z2.id};
  *argmax0.y.repeats = {One, s1, s2};
  *argmax0.y.strides = {Zero, s2, One};
  *argmax0.y.vectorized_axis = {z1.id, z2.id};

  store.x = argmax0.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.y.dtype = ge::DT_INT64;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1 * s2, s2, One};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id};
  y.y.dtype = ge::DT_INT64;
  *y.y.axis = {z0.id, z1.id, z2.id};
  *y.y.repeats = {s0, s1, s2};
  *y.y.strides = {s1 * s2, s2, One};
}

/**
 * @tc.name: CalcArgmaxTmpSize_test_0
 * @tc.number: CalcArgmaxTmpSize_Test_001
 * @tc.desc: Test when node is valid then CalcArgmaxTmpSize returns correct size for float input
 */
TEST_F(CalcArgmaxTmpSizeTest, CalcArgmaxTmpSize_test_0) {
  af::AscGraph graph("testx");
  af::Expression One = af::Symbol(1);
  af::Expression Zero = af::Symbol(0);
  Expression s1;
  Expression s2;
  CreateGraphArgmax<ge::DT_FLOAT>(graph, s1, s2);
  std::shared_ptr<af::AscNode> node = graph.FindNode("argmax0");
  node->inputs[0].attr.vectorized_strides = {s2, One};
  node->outputs[0].attr.vectorized_strides = {One, Zero};
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcArgmaxTmpSize(*node);
  ASSERT_GE(result.size(), 1);
}

/**
 * @tc.name: CalcArgmaxTmpSize_test_1
 * @tc.number: CalcArgmaxTmpSize_Test_002
 * @tc.desc: Test when node is valid then CalcArgmaxTmpSize returns correct size for int32 input
 */
TEST_F(CalcArgmaxTmpSizeTest, CalcArgmaxTmpSize_test_1) {
  af::AscGraph graph("testx");
  af::Expression One = af::Symbol(1);
  af::Expression Zero = af::Symbol(0);
  Expression s1;
  Expression s2;
  CreateGraphArgmax<ge::DT_INT32>(graph, s1, s2);
  std::shared_ptr<af::AscNode> node = graph.FindNode("argmax0");
  node->inputs[0].attr.vectorized_strides = {s2, One};
  node->outputs[0].attr.vectorized_strides = {One, Zero};
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcArgmaxTmpSize(*node);
  ASSERT_GE(result.size(), 1);
}

/**
 * @tc.name: CalcArgmaxTmpSize_test_2
 * @tc.number: CalcArgmaxTmpSize_Test_003
 * @tc.desc: Test when node is valid then CalcArgmaxTmpSize returns correct size for RA pattern with float input
 */
TEST_F(CalcArgmaxTmpSizeTest, CalcArgmaxTmpSize_test_2) {
  af::AscGraph graph("testx");
  af::Expression One = af::Symbol(1);
  af::Expression Zero = af::Symbol(0);
  Expression s1;
  Expression s2;
  CreateGraphArgmaxRA<ge::DT_FLOAT>(graph, s1, s2);
  std::shared_ptr<af::AscNode> node = graph.FindNode("argmax0");
  node->inputs[0].attr.vectorized_strides = {s2, One};
  node->outputs[0].attr.vectorized_strides = {Zero, One};
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcArgmaxTmpSize(*node);
  ASSERT_GE(result.size(), 1);
}

/**
 * @tc.name: CalcArgmaxTmpSize_test_3
 * @tc.number: CalcArgmaxTmpSize_Test_004
 * @tc.desc: Test when node is valid then CalcArgmaxTmpSize returns correct size for RA pattern with int32 input
 */
TEST_F(CalcArgmaxTmpSizeTest, CalcArgmaxTmpSize_test_3) {
  af::AscGraph graph("testx");
  af::Expression One = af::Symbol(1);
  af::Expression Zero = af::Symbol(0);
  Expression s1;
  Expression s2;
  CreateGraphArgmaxRA<ge::DT_INT32>(graph, s1, s2);
  std::shared_ptr<af::AscNode> node = graph.FindNode("argmax0");
  node->inputs[0].attr.vectorized_strides = {s2, One};
  node->outputs[0].attr.vectorized_strides = {Zero, One};
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcArgmaxTmpSize(*node);
  ASSERT_GE(result.size(), 1);
}

}  // namespace ascir
}  // namespace af
