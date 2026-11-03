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

namespace af {
namespace ascir {
extern std::vector<std::unique_ptr<af::TmpBufDesc>> CalcPowTmpSizeV2(const af::AscNode &node);
const Expression MAX_TMP_BUFFER_SIZE = af::Symbol(255 * 256 + 32);

using namespace testing;

class CalcPowTmpSizeV2Test : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(CalcPowTmpSizeV2Test, CalcPowTmpSizeV2_ShouldReturnCorrectSize_Int32) {
  af::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("zo", s1);

  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Load load1("load1");
  af::ascir_op::Pow pow("pow");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  const auto data_type = af::DT_INT32;

  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = data_type;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, Symbol(1)};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = data_type;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  pow.x1 = load1.y;
  pow.x2 = load1.y;
  pow.attr.sched.axis = {z0.id, z1.id};
  pow.y.dtype = data_type;
  *pow.y.axis = {z0.id, z1.id};
  *pow.y.repeats = {s0, s1};
  *pow.y.strides = {s1, Symbol(1)};

  store.x = pow.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = data_type;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, Symbol(1)};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = data_type;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {s0, s1};
  *y.y.strides = {s1, Symbol(1)};

  std::shared_ptr<af::AscNode> node = graph.FindNode("pow");
  node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcPowTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, 0);
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcPowTmpSizeV2Test, CalcPowTmpSizeV2_ShouldReturnCorrectSize_FLoat32) {
  af::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("zo", s1);

  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Load load1("load1");
  af::ascir_op::Pow pow("pow");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  const auto data_type = af::DT_FLOAT;

  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = data_type;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, Symbol(1)};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = data_type;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  pow.x1 = load1.y;
  pow.x2 = load1.y;
  pow.attr.sched.axis = {z0.id, z1.id};
  pow.y.dtype = data_type;
  *pow.y.axis = {z0.id, z1.id};
  *pow.y.repeats = {s0, s1};
  *pow.y.strides = {s1, Symbol(1)};

  store.x = pow.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = data_type;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, Symbol(1)};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = data_type;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {s0, s1};
  *y.y.strides = {s1, Symbol(1)};

  std::shared_ptr<af::AscNode> node = graph.FindNode("pow");
  node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcPowTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, sym::Min(sym::Align(af::Symbol(4) * s1 * s0, 32) * af::Symbol(3), MAX_TMP_BUFFER_SIZE));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcPowTmpSizeV2Test, CalcPowTmpSizeV2_ShouldReturnCorrectSize_Float16) {
  af::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("zo", s1);

  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Load load1("load1");
  af::ascir_op::Pow pow("pow");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  const auto data_type = af::DT_FLOAT16;

  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = data_type;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, Symbol(1)};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = data_type;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, Symbol(1)};
  *load1.y.vectorized_axis = {z0.id, z1.id};

  pow.x1 = load1.y;
  pow.x2 = load1.y;
  pow.attr.sched.axis = {z0.id, z1.id};
  pow.y.dtype = data_type;
  *pow.y.axis = {z0.id, z1.id};
  *pow.y.repeats = {s0, s1};
  *pow.y.strides = {s1, Symbol(1)};

  store.x = pow.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = data_type;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, Symbol(1)};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = data_type;
  *y.y.axis = {z0.id, z1.id};
  *y.y.repeats = {s0, s1};
  *y.y.strides = {s1, Symbol(1)};

  std::shared_ptr<af::AscNode> node = graph.FindNode("pow");
  node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcPowTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, sym::Min(sym::Align(af::Symbol(2) * s1 * s0, 32) * af::Symbol(6), MAX_TMP_BUFFER_SIZE));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcPowTmpSizeV2Test, CalcPowTmpSize_WhenInputsAreAllScalarOrUbScalar) {
  af::AscGraph graph("test");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto zo = graph.CreateAxis("zo", s1 + s2);
  auto zo_s_0 = graph.CreateAxis("zo_s_0", Axis::Type::kAxisTypeOriginal, s1, {zo.id}, af::kIdNone);
  auto zo_s_1 = graph.CreateAxis("zo_s_1", Axis::Type::kAxisTypeOriginal, s2, {zo.id}, af::kIdNone);

  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Load load1("load1");
  af::ascir_op::Pow pow("pow");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  x1.attr.sched.axis = {z0.id, zo_s_0.id};
  x1.y.dtype = af::DT_FLOAT;
  *x1.y.axis = {z0.id, zo_s_0.id};
  *x1.y.repeats = {Symbol(1), Symbol(1)};
  *x1.y.strides = {Symbol(0), Symbol(0)};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, zo_s_0.id};
  load1.y.dtype = af::DT_FLOAT;
  *load1.y.axis = {z0.id, zo_s_0.id};
  *load1.y.repeats = {Symbol(1), Symbol(1)};
  *load1.y.strides = {Symbol(0), Symbol(0)};
  *load1.y.vectorized_axis = {z0.id, zo_s_0.id};

  pow.x1 = load1.y;
  pow.x2 = load1.y;
  pow.attr.sched.axis = {z0.id, zo_s_0.id};
  pow.y.dtype = af::DT_FLOAT;
  *pow.y.axis = {z0.id, zo_s_0.id};
  *pow.y.repeats = {s0, s1};
  *pow.y.strides = {s1, Symbol(1)};

  store.x = pow.y;
  store.attr.sched.axis = {z0.id, zo_s_0.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id, zo_s_0.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, Symbol(1)};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, zo_s_0.id};
  y.y.dtype = af::DT_FLOAT;
  *y.y.axis = {z0.id, zo_s_0.id};
  *y.y.repeats = {s0, s1};
  *y.y.strides = {s1, Symbol(1)};

  std::shared_ptr<af::AscNode> node = graph.FindNode("pow");
  node->inputs[0].attr.vectorized_strides = {s1, Symbol(1)};
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcPowTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  // 全标量分支为f32/bf16额外扩容Power DOUBLE_FLOAT_TECH scratch（3段*4B*alignCount=384B，pow.h left_tmp_buf）
  ASSERT_EQ(result[0]->size, sym::Min(af::Symbol(4) * s1 + af::Symbol(32 * 2) + af::Symbol(384), MAX_TMP_BUFFER_SIZE));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}
}  // namespace ascir
}  // namespace af
