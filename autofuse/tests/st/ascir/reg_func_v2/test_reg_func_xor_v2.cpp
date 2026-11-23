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
#include "v35/ascir/reg_func/default_reg_func_v2.h"

namespace af {
namespace ascir {
using namespace testing;
using namespace af::ascir_op;

class CalcXorTmpSizeV2Test : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Helper function to create the xor test node with the whole graph flow.
static std::shared_ptr<af::AscNode> CreateXorTestNode(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto zo = graph.CreateAxis("zo", s1 + s2);
  auto zo_s_0 = graph.CreateAxis("zo_s_0", Axis::Type::kAxisTypeOriginal, s1, {zo.id}, af::kIdNone);
  auto zo_s_1 = graph.CreateAxis("zo_s_1", Axis::Type::kAxisTypeOriginal, s2, {zo.id}, af::kIdNone);

  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Load load1("load1");
  af::ascir_op::Xor xor_op("xor");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  x1.attr.sched.axis = {z0.id, zo_s_0.id};
  x1.y.dtype = ge::DT_UINT8;
  *x1.y.axis = {z0.id, zo_s_0.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, Symbol(1)};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, zo_s_0.id};
  load1.y.dtype = ge::DT_UINT8;
  *load1.y.axis = {z0.id, zo_s_0.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, zo_s_0.id};

  xor_op.x1 = load1.y;
  xor_op.x2 = load1.y;
  xor_op.attr.sched.axis = {z0.id, zo_s_0.id};
  xor_op.y.dtype = ge::DT_UINT8;
  *xor_op.y.axis = {z0.id, zo_s_0.id};
  *xor_op.y.repeats = {s0, s1};
  *xor_op.y.strides = {s1, Symbol(1)};

  store.x = xor_op.y;
  store.attr.sched.axis = {z0.id, zo_s_0.id};
  store.y.dtype = ge::DT_UINT8;
  *store.y.axis = {z0.id, zo_s_0.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, Symbol(1)};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, zo_s_0.id};
  y.y.dtype = ge::DT_UINT8;
  *y.y.axis = {z0.id, zo_s_0.id};
  *y.y.repeats = {s0, s1};
  *y.y.strides = {s1, Symbol(1)};

  std::shared_ptr<af::AscNode> node = graph.FindNode("xor");
  node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
  return node;
}

TEST_F(CalcXorTmpSizeV2Test, CalcXorTmpSizeV2_ShouldReturnCorrectSize_WhenInputsIsUINT8) {
  af::AscGraph graph("test");
  std::shared_ptr<af::AscNode> node = CreateXorTestNode(graph);
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcXorTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, Symbol(256));
  // 256 * 1 = 256 bytes
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

}  // namespace ascir
}  // namespace af
