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
extern std::vector<std::unique_ptr<af::TmpBufDesc>> CalcBroadCastTmpSize(const af::AscNode &node);

using namespace testing;
using namespace af::ascir_op;

class CalcBroadCastTmpSizeTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
  template <af::DataType T>
  void CreateGraph(af::AscGraph &graph, Expression &s0, Expression &s1, Expression &s2) {
    af::Expression One = af::Symbol(1);
    s0 = graph.CreateSizeVar("s0");
    s1 = graph.CreateSizeVar("s1");
    s2 = graph.CreateSizeVar("s2");

    auto z0 = graph.CreateAxis("z0", s0);
    auto zo = graph.CreateAxis("zo", s1 + s2);
    auto zo_s_0 = graph.CreateAxis("zo_s_0", Axis::Type::kAxisTypeOriginal, s1, {zo.id}, af::kIdNone);
    auto zo_s_1 = graph.CreateAxis("zo_s_1", Axis::Type::kAxisTypeOriginal, s2, {zo.id}, af::kIdNone);

    Data x1("x1", graph);
    Load load1("load1");
    af::ascir_op::Broadcast broadcast("broadcast");
    Store store("store");
    Output y("y");

    x1.attr.sched.axis = {z0.id, zo_s_0.id};
    x1.y.dtype = T;
    *x1.y.axis = {z0.id, zo_s_0.id};
    *x1.y.repeats = {s0, s1};
    *x1.y.strides = {s1, One};

    load1.x = x1.y;
    load1.attr.sched.axis = {z0.id, zo_s_0.id};
    load1.y.dtype = T;
    *load1.y.axis = {z0.id, zo_s_0.id};
    *load1.y.repeats = {s0, s1};
    *load1.y.strides = {s1, One};
    *load1.y.vectorized_axis = {z0.id, zo.id};

    broadcast.x = {load1.y};
    broadcast.attr.sched.axis = {z0.id, zo.id};
    broadcast.y.dtype = T;
    *broadcast.y.axis = {z0.id, zo.id};
    *broadcast.y.repeats = {s0, s1 + s2};
    *broadcast.y.strides = {s1 + s2, One};
    *broadcast.y.vectorized_axis = {z0.id, zo.id};

    store.x = broadcast.y;
    store.attr.sched.axis = {z0.id, zo.id};
    store.y.dtype = T;
    *store.y.axis = {z0.id, zo.id};
    *store.y.repeats = {s0, s1 + s2};
    *store.y.strides = {s1 + s2, One};

    y.x = store.y;
    y.attr.sched.axis = {z0.id, zo.id};
    y.y.dtype = T;
    *y.y.axis = {z0.id, zo.id};
    *y.y.repeats = {s0, s1 + s2};
    *y.y.strides = {s1 + s2, One};
    af::Expression Zero = af::Symbol(0);
    for (auto node : graph.GetAllNodes()) {
      if (node->GetType() == "Data" || node->GetType() == "Output") {
        continue;
      }
      if (node->GetType() == "Load") {
        node->outputs[0].attr.vectorized_strides = {s1, Zero};
      } else {
        node->outputs[0].attr.vectorized_strides = {s1, One};
      }
    }
  }

  template <af::DataType T>
  void CreateGraphMultiAxis(af::AscGraph &graph, Expression &s1, Expression &s2, Expression &s3, Expression &s4) {
    af::Expression One = af::Symbol(1);
    af::Expression Zero = af::Symbol(0);
    auto s0 = graph.CreateSizeVar("s0");
    s1 = graph.CreateSizeVar("s1");
    s2 = graph.CreateSizeVar("s2");
    s3 = graph.CreateSizeVar("s3");
    s4 = graph.CreateSizeVar("s4");

    auto z0 = graph.CreateAxis("z0", s0).id;
    auto z1 = graph.CreateAxis("z1", s1).id;
    auto z2 = graph.CreateAxis("z2", s2).id;
    auto z3 = graph.CreateAxis("z3", s3).id;
    auto z4 = graph.CreateAxis("z4", s4).id;

    Data x_op("x");
    graph.AddNode(x_op);
    x_op.y.dtype = T;

    Load load_op("load");
    graph.AddNode(load_op);
    load_op.x = x_op.y;
    load_op.attr.sched.axis = {z0, z1, z2, z3, z4};
    *load_op.y.axis = {z0, z1, z2, z3, z4};
    load_op.y.dtype = T;
    *load_op.y.repeats = {s0, One, s2, s3, One};
    *load_op.y.strides = {s1 * s3, Zero, s3, One, Zero};
    load_op.attr.sched.loop_axis = z0;
    *load_op.y.vectorized_axis = {z1, z2, z3, z4};

    Broadcast broadcast_op("broadcast");
    graph.AddNode(broadcast_op);
    broadcast_op.x = load_op.y;
    broadcast_op.attr.sched.axis = {z0, z1, z2, z3, z4};
    broadcast_op.attr.sched.loop_axis = z0;
    broadcast_op.y.dtype = T;
    *broadcast_op.y.axis = {z0, z1, z2, z3, z4};
    *broadcast_op.y.repeats = {s0, s1, s2, s3, s4};
    *broadcast_op.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};
    *broadcast_op.y.vectorized_axis = {z1, z2, z3, z4};

    Store store_op("store");
    graph.AddNode(store_op);
    store_op.x = broadcast_op.y;
    store_op.y.dtype = T;
    store_op.attr.sched.axis = {z0, z1, z2, z3, z4};
    store_op.attr.sched.loop_axis = z0;
    *store_op.y.axis = {z0, z1, z2, z3, z4};
    *store_op.y.repeats = {s0, s1, s2, s3, s4};
    *store_op.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};
    *store_op.y.vectorized_axis = {z1, z2, z3, z4};

    Output y_op("y");
    graph.AddNode(y_op);
    y_op.x = store_op.y;
    y_op.y.dtype = T;

    for (auto node : graph.GetAllNodes()) {
      if (node->GetType() == "Data" || node->GetType() == "Output") {
        continue;
      }
      if (node->GetType() == "Load") {
        node->outputs[0].attr.vectorized_strides = {Zero, s3, One, Zero};
      } else {
        node->outputs[0].attr.vectorized_strides = {s2 * s3 * s4, s3 * s4, s4, One};
      }
    }
  }
};

/**
 * @tc.name:CalcBroadCastTmpSize_NeedExtrabufErr
 * @tc.number: CalcBroadCastTmpSize_Test_001
 * @tc.desc: Test when node is valid then CalcBroadCastTmpSize returns correct size
 */
TEST_F(CalcBroadCastTmpSizeTest, CalcBroadCastTmpSize_NeedExtrabufFalse) {
  af::AscGraph graph("test");
  Expression s0;
  Expression s1;
  Expression s2;
  CreateGraph<af::DT_FLOAT>(graph, s0, s1, s2);
  std::shared_ptr<af::AscNode> node = graph.FindNode("broadcast");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcBroadCastTmpSize(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(8192));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

/**
 * @tc.name:CalcBroadCastTmpSize_NeedExtrabufU8
 * @tc.number: CalcBroadCastTmpSize_Test_002
 * @tc.desc: Test when node is valid then CalcBroadCastTmpSize returns correct size
 */
TEST_F(CalcBroadCastTmpSizeTest, CalcBroadCastTmpSize_NeedExtrabufU8) {
  af::AscGraph graph("test");
  Expression s0;
  Expression s1;
  Expression s2;
  CreateGraph<af::DT_UINT8>(graph, s0, s1, s2);
  std::shared_ptr<af::AscNode> node = graph.FindNode("broadcast");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcBroadCastTmpSize(*node);
  ASSERT_EQ(result.size(), 1);
  Expression input_size = af::Symbol(32) * sym::Ceiling(sym::Rational(1, 16) * s0);
  Expression output_size =
      af::Symbol(32) * sym::Ceiling(af::Symbol(2) * sym::Ceiling((s1 + s2) * sym::Rational(1, 32)) * s0);
  Expression tmp_size = input_size + output_size + af::Symbol(8192);
  ASSERT_EQ(result[0]->size, tmp_size);
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

/**
 * @tc.name:CalcBroadCastTmpSize_NeedExtra
 * @tc.number: CalcBroadCastTmpSize_Test_003
 * @tc.desc: Test when node is valid then CalcBroadCastTmpSize returns correct size
 */
TEST_F(CalcBroadCastTmpSizeTest, CalcBroadCastTmpSize_NeedExtrabufTrue) {
  af::AscGraph graph("test");
  Expression s1;
  Expression s2;
  Expression s3;
  Expression s4;
  CreateGraphMultiAxis<af::DT_FLOAT>(graph, s1, s2, s3, s4);
  std::shared_ptr<af::AscNode> node = graph.FindNode("broadcast");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcBroadCastTmpSize(*node);
  ASSERT_EQ(result.size(), 1);
  Expression tmp_size = sym::Align(s4, 8) * af::Symbol(4) * s1 * s2 * s3 + af::Symbol(8192);
  ASSERT_EQ(result[0]->size, tmp_size);
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

/**
 * @tc.name: BroadcastInferDataType_SupportsBf16
 * @tc.number: BroadcastInferDataType_Test_001
 * @tc.desc: Test Broadcast accepts DT_BF16 on both v1(2201) and v2 platforms
 */
TEST_F(CalcBroadCastTmpSizeTest, BroadcastInferDataType_SupportsBf16) {
  for (const auto *npu_arch : {"2201", "3510", "5102", "9201", "9202"}) {
    std::vector<af::DataType> input_dtypes{af::DT_BF16};
    std::vector<af::DataType> output_dtypes{af::DT_BF16};
    EXPECT_EQ(af::ascir_op::Broadcast::InferDataType(input_dtypes, output_dtypes, npu_arch), af::SUCCESS)
        << "npu_arch=" << npu_arch;
  }
}
}  // namespace ascir
}  // namespace af
