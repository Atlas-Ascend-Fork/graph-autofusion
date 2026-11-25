/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "default_reg_func_v2.h"

namespace af {
namespace ascir {

using namespace testing;
using namespace af::ascir_op;

class CalcIgammacTmpSizeV2Test : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}

  std::shared_ptr<AscNode> BuildGraph(af::AscGraph &graph, const af::DataType dtype) {
    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");
    auto s2 = graph.CreateSizeVar("s2");

    auto z0 = graph.CreateAxis("z0", s0);
    auto zo = graph.CreateAxis("zo", s1 + s2);
    auto zo_s_0 = graph.CreateAxis("zo_s_0", Axis::Type::kAxisTypeOriginal, s1, {zo.id}, af::kIdNone);

    af::ascir_op::Data x1("x1", graph);
    x1.attr.sched.axis = {z0.id, zo_s_0.id};
    x1.y.dtype = dtype;
    *x1.y.axis = {z0.id, zo_s_0.id};
    *x1.y.repeats = {s0, s1};
    *x1.y.strides = {s1, Symbol(1)};

    af::ascir_op::Load load1("load1");
    load1.x = x1.y;
    load1.attr.sched.axis = {z0.id, zo_s_0.id};
    load1.y.dtype = dtype;
    *load1.y.axis = {z0.id, zo_s_0.id};
    *load1.y.repeats = {s0, s1};
    *load1.y.strides = {s1, Symbol(1)};
    *load1.y.vectorized_axis = {z0.id, zo_s_0.id};

    af::ascir_op::Igammac op_node("Igammac");
    op_node.x1 = load1.y;
    op_node.x2 = load1.y;
    op_node.attr.sched.axis = {z0.id, zo_s_0.id};
    op_node.y.dtype = dtype;
    *op_node.y.axis = {z0.id, zo_s_0.id};
    *op_node.y.repeats = {s0, s1};
    *op_node.y.strides = {s1, Symbol(1)};

    af::ascir_op::Store store("store");
    store.x = op_node.y;
    store.attr.sched.axis = {z0.id, zo_s_0.id};
    store.y.dtype = dtype;
    *store.y.axis = {z0.id, zo_s_0.id};
    *store.y.repeats = {s0, s1};
    *store.y.strides = {s1, Symbol(1)};

    af::ascir_op::Output y("y");
    y.x = store.y;
    y.attr.sched.axis = {z0.id, zo_s_0.id};
    y.y.dtype = dtype;
    *y.y.axis = {z0.id, zo_s_0.id};
    *y.y.repeats = {s0, s1};
    *y.y.strides = {s1, Symbol(1)};

    auto node = graph.FindNode("Igammac");
    node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
    return node;
  }
};

/**
 * @tc.name: CalcIgammacTmpSizeV2_ShouldReturnCorrectSize_WhenInputsIsFLOAT
 * @tc.number: CalcIgammacTmpSizeV2_Test_001
 * @tc.desc: Test CalcIgammacTmpSizeV2 returns expected tmp buffer size for DT_FLOAT
 */
TEST_F(CalcIgammacTmpSizeV2Test, CalcIgammacTmpSizeV2_ShouldReturnCorrectSize_WhenInputsIsFLOAT) {
  af::AscGraph graph("test");
  auto node = BuildGraph(graph, af::DT_FLOAT);
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcIgammacTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, Symbol(2048));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

/**
 * @tc.name: CalcIgammacTmpSizeV2_ShouldReturnCorrectSize_WhenInputsIsFLOAT16
 * @tc.number: CalcIgammacTmpSizeV2_Test_002
 * @tc.desc: Test CalcIgammacTmpSizeV2 returns expected tmp buffer size for DT_FLOAT16
 */
TEST_F(CalcIgammacTmpSizeV2Test, CalcIgammacTmpSizeV2_ShouldReturnCorrectSize_WhenInputsIsFLOAT16) {
  af::AscGraph graph("test");
  auto node = BuildGraph(graph, af::DT_FLOAT16);
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcIgammacTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, Symbol(2048));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}
}  // namespace ascir
}  // namespace af
