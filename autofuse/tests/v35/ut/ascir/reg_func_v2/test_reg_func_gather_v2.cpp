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
extern std::vector<std::unique_ptr<af::TmpBufDesc>> CalcGatherTmpSizeV2(const af::AscNode &node);

using namespace testing;
using namespace af::ascir_op;

class CalcGatherTmpSizeV2Test : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

template <af::DataType T1, af::DataType T2>
void CreateGraph(af::AscGraph &graph, af::Expression &s0, af::Expression &s2) {
  af::Expression One = af::Symbol(1);
  s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Data x2("x2", graph);
  af::ascir_op::Gather gather("gather");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  x1.attr.sched.axis = {z0.id};
  x1.y.dtype = T1;
  *x1.y.axis = {z0.id};
  *x1.y.repeats = {s0};
  *x1.y.strides = {One};

  x2.attr.sched.axis = {z1.id, z2.id};
  x2.y.dtype = T2;
  *x2.y.axis = {z1.id, z2.id};
  *x2.y.repeats = {s1, s2};
  *x2.y.strides = {s2, One};

  gather.x1 = x1.y;
  gather.x2 = x2.y;
  gather.attr.sched.axis = {z1.id, z2.id};
  gather.y.dtype = T1;
  *gather.y.axis = {z1.id, z2.id};
  *gather.y.repeats = {s1, s2};
  *gather.y.strides = {s2, One};

  store.x = gather.y;
  store.attr.sched.axis = {z1.id, z2.id};
  store.y.dtype = T1;
  *store.y.axis = {z1.id, z2.id};
  *store.y.repeats = {s1, s2};
  *store.y.strides = {s2, One};

  y.x = store.y;
  y.attr.sched.axis = {z1.id, z2.id};
  y.y.dtype = T1;
  *y.y.axis = {z1.id, z2.id};
  *y.y.repeats = {s1, s2};
  *y.y.strides = {s2, One};
}

template <af::DataType T1, af::DataType T2>
void CreateGraphStaticOneAxis(af::AscGraph &graph, int32_t param_axis, int32_t index_axis_1, int32_t index_axis_2) {
  Expression One = af::Symbol(1);
  Expression s0 = af::Symbol(param_axis);
  Expression s1 = af::Symbol(index_axis_1);
  Expression s2 = af::Symbol(index_axis_2);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Data x2("x2", graph);
  af::ascir_op::Gather gather("gather");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  x1.attr.sched.axis = {z0.id};
  x1.y.dtype = T1;
  *x1.y.axis = {z0.id};
  *x1.y.repeats = {s0};
  *x1.y.strides = {One};

  x2.attr.sched.axis = {z1.id, z2.id};
  x2.y.dtype = T2;
  *x2.y.axis = {z1.id, z2.id};
  *x2.y.repeats = {s1, s2};
  *x2.y.strides = {s2, One};

  gather.x1 = x1.y;
  gather.x2 = x2.y;
  gather.attr.sched.axis = {z1.id, z2.id};
  gather.y.dtype = T1;
  *gather.y.axis = {z1.id, z2.id};
  *gather.y.repeats = {s1, s2};
  *gather.y.strides = {s2, One};

  store.x = gather.y;
  store.attr.sched.axis = {z1.id, z2.id};
  store.y.dtype = T1;
  *store.y.axis = {z1.id, z2.id};
  *store.y.repeats = {s1, s2};
  *store.y.strides = {s2, One};

  y.x = store.y;
  y.attr.sched.axis = {z1.id, z2.id};
  y.y.dtype = T1;
  *y.y.axis = {z1.id, z2.id};
  *y.y.repeats = {s1, s2};
  *y.y.strides = {s2, One};
}

template <af::DataType T1, af::DataType T2>
void CreateGraphStaticMiddleAxis(af::AscGraph &graph, int32_t param_axis_1, int32_t param_axis_2, int32_t index_axis_1,
                                 int32_t index_axis_2) {
  Expression One = af::Symbol(1);
  Expression s0 = af::Symbol(param_axis_1);
  Expression s1 = af::Symbol(param_axis_2);
  Expression s2 = af::Symbol(index_axis_1);
  Expression s3 = af::Symbol(index_axis_2);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Data x2("x2", graph);
  af::ascir_op::Gather gather("gather");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = T1;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, One};

  x2.attr.sched.axis = {z2.id, z3.id};
  x2.y.dtype = T2;
  *x2.y.axis = {z2.id, z3.id};
  *x2.y.repeats = {s2, s3};
  *x2.y.strides = {s3, One};

  gather.x1 = x1.y;
  gather.x2 = x2.y;
  gather.attr.sched.axis = {z1.id, z2.id};
  gather.y.dtype = T1;
  gather.ir_attr.SetAxis(0);
  gather.ir_attr.SetNegative_index_support(false);
  *gather.y.axis = {z1.id, z2.id};
  *gather.y.repeats = {s1, s2};
  *gather.y.strides = {s2, One};

  store.x = gather.y;
  store.attr.sched.axis = {z2.id, z3.id, z1.id};
  store.y.dtype = T1;
  *store.y.axis = {z2.id, z3.id, z1.id};
  *store.y.repeats = {s2, s3, s1};
  *store.y.strides = {s3 * s1, s1, One};

  y.x = store.y;
  y.attr.sched.axis = {z2.id, z3.id, z1.id};
  y.y.dtype = T1;
  *y.y.axis = {z2.id, z3.id, z1.id};
  *y.y.repeats = {s2, s3, s1};
  *y.y.strides = {s3 * s1, s1, One};
}

template <af::DataType T1, af::DataType T2>
void CreateGraphStaticTailAxis(af::AscGraph &graph, int32_t param_axis_1, int32_t param_axis_2, int32_t index_axis_1,
                               int32_t index_axis_2) {
  Expression One = af::Symbol(1);
  Expression s0 = af::Symbol(param_axis_1);
  Expression s1 = af::Symbol(param_axis_2);
  Expression s2 = af::Symbol(index_axis_1);
  Expression s3 = af::Symbol(index_axis_2);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Data x2("x2", graph);
  af::ascir_op::Gather gather("gather");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = T1;
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, One};

  x2.attr.sched.axis = {z2.id, z3.id};
  x2.y.dtype = T2;
  *x2.y.axis = {z2.id, z3.id};
  *x2.y.repeats = {s2, s3};
  *x2.y.strides = {s3, One};

  gather.x1 = x1.y;
  gather.x2 = x2.y;
  gather.attr.sched.axis = {z1.id, z2.id};
  gather.y.dtype = T1;
  gather.ir_attr.SetAxis(1);
  gather.ir_attr.SetNegative_index_support(false);
  *gather.y.axis = {z1.id, z2.id};
  *gather.y.repeats = {s1, s2};
  *gather.y.strides = {s2, One};

  store.x = gather.y;
  store.attr.sched.axis = {z0.id, z2.id, z3.id};
  store.y.dtype = T1;
  *store.y.axis = {z0.id, z2.id, z3.id};
  *store.y.repeats = {s0, s2, s3};
  *store.y.strides = {s2 * s3, s3, One};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, z2.id, z3.id};
  y.y.dtype = T1;
  *y.y.axis = {z0.id, z2.id, z3.id};
  *y.y.repeats = {s0, s2, s3};
  *y.y.strides = {s2 * s3, s3, One};
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size) {
  af::AscGraph graph("test");
  Expression s0;
  Expression s2;
  CreateGraph<af::DT_FLOAT, af::DT_INT64>(graph, s0, s2);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(32));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size2) {
  af::AscGraph graph("test");
  Expression s0;
  Expression s2;
  CreateGraph<af::DT_FLOAT, af::DT_INT32>(graph, s0, s2);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(32));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size3) {
  af::AscGraph graph("test");
  Expression s0;
  Expression s2;
  CreateGraph<af::DT_FLOAT16, af::DT_INT64>(graph, s0, s2);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(32));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size4) {
  af::AscGraph graph("test");
  Expression s0;
  Expression s2;
  CreateGraph<af::DT_FLOAT16, af::DT_INT32>(graph, s0, s2);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(32));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size5) {
  af::AscGraph graph("test");
  CreateGraphStaticOneAxis<af::DT_FLOAT, af::DT_INT32>(graph, 30000, 100, 100);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(int32_t(0.9 * 4 * 30000) + 24 * 1024));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size6) {
  af::AscGraph graph("test");
  CreateGraphStaticOneAxis<af::DT_FLOAT16, af::DT_INT32>(graph, 30000, 100, 100);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(int32_t(0.9 * 2 * 30000) + 24 * 1024));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size7) {
  af::AscGraph graph("test");
  CreateGraphStaticOneAxis<af::DT_FLOAT, af::DT_INT32>(graph, 30001, 100, 100);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(32));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size8) {
  af::AscGraph graph("test");
  CreateGraphStaticOneAxis<af::DT_FLOAT, af::DT_INT32>(graph, 1000, 27, 100);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(5124));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size9) {
  af::AscGraph graph("test");
  CreateGraphStaticOneAxis<af::DT_FLOAT, af::DT_INT64>(graph, 1000, 6, 100);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(5124));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size10) {
  af::AscGraph graph("test");
  CreateGraphStaticOneAxis<af::DT_FLOAT, af::DT_INT64>(graph, 30000, 100, 100);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(141056));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size11) {
  af::AscGraph graph("test");
  CreateGraphStaticOneAxis<af::DT_FLOAT16, af::DT_INT64>(graph, 30000, 100, 100);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(int32_t(0.8 * 2 * 30000 + 44 * 1024)));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size12) {
  af::AscGraph graph("test");
  CreateGraphStaticOneAxis<af::DT_FLOAT, af::DT_INT64>(graph, 30001, 100, 100);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(32));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size13) {
  af::AscGraph graph("test");
  CreateGraphStaticOneAxis<af::DT_FLOAT16, af::DT_INT64>(graph, 30000, 10000, 10000);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(int32_t(0.8 * 2 * 30000 + 45056)));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size14) {
  af::AscGraph graph("test");
  CreateGraphStaticMiddleAxis<af::DT_FLOAT, af::DT_INT64>(graph, 200, 100, 750, 1);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(4 * 20000));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size15) {
  af::AscGraph graph("test");
  CreateGraphStaticMiddleAxis<af::DT_FLOAT16, af::DT_INT64>(graph, 200, 100, 750, 1);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(2 * 20000));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size16) {
  af::AscGraph graph("test");
  CreateGraphStaticMiddleAxis<af::DT_FLOAT, af::DT_INT32>(graph, 200, 100, 750, 1);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(4 * 20000));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size17) {
  af::AscGraph graph("test");
  CreateGraphStaticMiddleAxis<af::DT_FLOAT, af::DT_INT32>(graph, 200, 100, 749, 1);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(32));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size18) {
  af::AscGraph graph("test");
  CreateGraphStaticMiddleAxis<af::DT_FLOAT, af::DT_INT32>(graph, 200, 101, 750, 1);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(32));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size19) {
  af::AscGraph graph("test");
  CreateGraphStaticMiddleAxis<af::DT_FLOAT16, af::DT_INT32>(graph, 200, 100, 750, 1);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(2 * 20000));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size20) {
  af::AscGraph graph("test");
  CreateGraphStaticMiddleAxis<af::DT_FLOAT16, af::DT_INT32>(graph, 200, 101, 750, 1);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(32));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size21) {
  af::AscGraph graph("test");
  CreateGraphStaticTailAxis<af::DT_FLOAT, af::DT_INT32>(graph, 200, 100, 750, 1);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(32));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenNodelsValid_Size22) {
  af::AscGraph graph("test");
  CreateGraphStaticTailAxis<af::DT_FLOAT16, af::DT_INT32>(graph, 200, 100, 750, 1);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(32));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenValueIsInt64AndParamIsSmall) {
  af::AscGraph graph("test");
  CreateGraphStaticOneAxis<af::DT_INT64, af::DT_INT32>(graph, 2, 3, 9);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(1041));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenValueIsInt64AndParamIsLarge) {
  af::AscGraph graph("test");
  CreateGraphStaticOneAxis<af::DT_INT64, af::DT_INT32>(graph, 30000, 100, 100);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(32));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}

TEST_F(CalcGatherTmpSizeV2Test, Int64ValuesAndIndicesDoNotReserveAnUndersizedFullLoadBuffer) {
  for (int32_t param_size : {27540, 28160, 28200, 30000}) {
    SCOPED_TRACE(param_size);
    af::AscGraph graph("test");
    CreateGraphStaticOneAxis<af::DT_INT64, af::DT_INT64>(graph, param_size, 100, 100);
    auto node = graph.FindNode("gather");
    const auto result = CalcGatherTmpSizeV2(*node);
    ASSERT_EQ(result.size(), 1);
    // SIMT gathers directly from GM and needs no full params copy in UB.
    EXPECT_EQ(result[0]->size, af::Symbol(32));
  }
}

TEST_F(CalcGatherTmpSizeV2Test, Int64ValuesAndIndicesKeepFullLoadWhenWorkspaceFits) {
  af::AscGraph graph("test");
  CreateGraphStaticOneAxis<af::DT_INT64, af::DT_INT64>(graph, 27520, 100, 100);
  auto node = graph.FindNode("gather");
  const auto result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0]->size, af::Symbol(27520 * 8 + 1024));
}

TEST_F(CalcGatherTmpSizeV2Test, CalcGatherTmpSize_ShouldReturnCorrectSize_WhenValueIsInt64AndGatherMiddleAxis) {
  af::AscGraph graph("test");
  CreateGraphStaticMiddleAxis<af::DT_INT64, af::DT_INT64>(graph, 200, 100, 750, 1);
  auto node = graph.FindNode("gather");
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcGatherTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, af::Symbol(8 * 20000));
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
}
}  // namespace ascir
}  // namespace af
