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

#include "ascendc_ir.h"
#include "ascendc_ir_def.h"
#include "ascir_ops.h"
#define private public
#include "optimize.h"
#include "autoschedule/autoschedule.h"
#undef private
#include "ascir_ops_utils.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/compute_graph.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "attr_utils.h"
#include "graph/debug/ge_op_types.h"
#include "autoschedule/axis_group.h"
#include "schedule_utils.h"
#include "fused_graph/fused_graph_unfolder.h"
#include "platform_context.h"
#include "platform/v1/platformv1.h"

#include "asc_graph_builder.h"

using namespace std;
using namespace af;
using namespace af::ops;
using namespace af::ascir_op;
using af::testing::AscGraphBuilder;
using af::testing::Sym;
using optimize::autoschedule::AxisGroup;

AscGraph Construct_Reduce_RARA(const std::string &name) {
  return AscGraphBuilder(name)
      .Loops({Sym("s0"), Sym("s1"), Sym("s2"), Sym("s3")})
      .Data("arg4_1", 0)
      .Load("b0_load", "arg4_1")
      .Abs("abs", "b0_load")
      .Max("b0_max", "abs", {0, 2})
      .Store("b3_store", "b0_max")
      .Output("buf3", "b3_store", 0)
      .Build();
}

AscGraph Construct_Reduce_ARAR(const std::string &name) {
  return AscGraphBuilder(name)
      .Loops({Sym("s0"), Sym("s1"), Sym("s2"), Sym("s3")})
      .Data("arg4_1", 0)
      .Load("b0_load", "arg4_1")
      .Abs("abs", "b0_load")
      .Max("b0_max", "abs", {1, 3})
      .Store("b3_store", "b0_max")
      .Output("buf3", "b3_store", 0)
      .Build();
}

void Construct_Reduce_RR(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data("data", graph);
  data.y.dtype = af::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = af::DT_FLOAT;
  *load.y.strides = {s1, af::ops::One};
  *load.y.repeats = {s1, s0};

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.x = load.y;
  *sum.y.axis = {z0.id, z1.id};
  sum.y.dtype = af::DT_FLOAT;
  *sum.y.strides = {af::ops::One, af::ops::One};
  *sum.y.repeats = {af::ops::Zero, af::ops::Zero};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.x = sum.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = af::DT_FLOAT;
  *store_op1.y.axis = {z0.id, z1.id};
  *store_op1.y.strides = {af::ops::One, af::ops::One};
  *store_op1.y.repeats = {af::ops::Zero, af::ops::Zero};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = af::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

AscGraph Construct_Mul_Consumer_Struct(const std::string &name) {
  auto s0 = Sym("s0");
  auto s1 = Sym("s1");
  auto s2 = Sym("s2");
  auto s3 = Sym("s3");
  return AscGraphBuilder(name)
      .Loops({s0 * s1 * s2, s3})
      .Data("arg4_1", 0, {s0 * s1 * s2, s3}, {s3, af::ops::One}, af::DT_FLOAT16)
      .Load("b0_load", "arg4_1", {s0 * s1 * s2, s3}, {s3, af::ops::One})
      .Exp("b1_exp", "b0_load")
      .Abs("b0_abs", "b1_exp")
      .Max("b0_max", "b0_abs", {1})
      .Broadcast("b1_broadcast", "b0_max", {s0 * s1 * s2, s3})
      .Store("b0_store", "b1_broadcast")
      .Output("buf0", "b0_store", 1, af::DT_FLOAT)
      .template Op<af::ascir_op::Relu>("b0_relu", {"b1_exp"})
      .Store("b1_store", "b0_relu")
      .Output("buf1", "b1_store", 2, af::DT_FLOAT)
      .Build();
}

AscGraph ConstructNormStruct(const std::string &name) {
  return AscGraphBuilder(name)
      .Loops({Sym(128), Sym(64)})
      .Data("data", 0)
      .Load("load", "data")
      .Exp("exp", "load")
      .Sum("sum", "exp", {0, 1})
      .Broadcast("broadcast", "sum", {Sym(128), Sym(64)})
      .Sub("sub", "broadcast", "exp")
      .Store("store1", "sub")
      .Output("output", "store1", 0)
      .Build();
}

AscGraph ConstructNormStruct3Elewise(const std::string &name) {
  return AscGraphBuilder(name)
      .Loops({Sym(128), Sym(64)})
      .Data("data", 0)
      .Load("load", "data")
      .Sum("sum", "load", {0, 1})
      .Abs("abs", "sum")
      .Exp("exp", "abs")
      .Relu("b0_relu", "exp")
      .Store("store1", "b0_relu")
      .Output("output", "store1", 0)
      .Build();
}

AscGraph ConstructNormStruct1Elewise(const std::string &name) {
  return AscGraphBuilder(name)
      .Loops({Sym(128), Sym(64)})
      .Data("data", 0)
      .Load("load", "data")
      .Sum("sum", "load", {0, 1})
      .Abs("abs", "sum")
      .Store("store1", "abs")
      .Output("output", "store1", 0)
      .Build();
}

AscGraph ConstructNormStruct4Elewise(const std::string &name) {
  return AscGraphBuilder(name)
      .Loops({Sym(128), Sym(64)})
      .Data("data", 0)
      .Load("load", "data")
      .Sum("sum", "load", {0, 1})
      .Abs("abs", "sum")
      .Op<af::ascir_op::Tanh>("tanh", {"abs"})
      .Exp("exp", "tanh")
      .Relu("b0_relu", "exp")
      .Store("store1", "b0_relu")
      .Output("output", "store1", 0)
      .Build();
}

AscGraph ConstructNormStruct4Elewise4ReduceMultipleCitations(const std::string &name) {
  return AscGraphBuilder(name)
      .Loops({Sym(128), Sym(64)})
      .Data("data", 0)
      .Load("load", "data")
      .Sum("sum", "load", {0, 1})
      .Abs("abs", "sum")
      .Op<af::ascir_op::Tanh>("tanh", {"sum"})
      .Add("add", "abs", "tanh")
      .Relu("b0_relu", "add")
      .Store("store1", "b0_relu")
      .Output("output", "store1", 0)
      .Build();
}

AscGraph ConstructNormStruct4Elewise3ReduceMultipleCitations(const std::string &name) {
  return AscGraphBuilder(name)
      .Loops({Sym(128), Sym(64)})
      .Data("data", 0)
      .Load("load", "data")
      .Sum("sum", "load", {0, 1})
      .Abs("abs", "sum")
      .Op<af::ascir_op::Tanh>("tanh", {"sum"})
      .Add("add", "abs", "tanh")
      .Store("store1", "add")
      .Output("output", "store1", 0)
      .Build();
}

AscGraph ConstructNormStruct4Elewise3(const std::string &name) {
  return AscGraphBuilder(name)
      .Loops({Sym(128), Sym(64)})
      .Data("data0", 0)
      .Load("load0", "data0")
      .Data("data1", 1)
      .Load("load1", "data1")
      .Mul("mul", "load0", "load1")
      .Sum("sum", "mul", {0, 1})
      .Relu("b0_relu", "sum")
      .Op<af::ascir_op::Tanh>("tanh", {"b0_relu"})
      .Add("add", "tanh", "b0_relu")
      .Abs("abs", "add")
      .Store("store1", "abs")
      .Output("output", "store1", 0)
      .Build();
}

void Construct_Reduce_Cast_RR(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data("data", graph);
  data.y.dtype = af::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = af::DT_FLOAT;
  *load.y.strides = {s1, af::ops::One};
  *load.y.repeats = {s1, s0};

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.x = load.y;
  *sum.y.axis = {z0.id, z1.id};
  sum.y.dtype = af::DT_FLOAT;
  *sum.y.strides = {af::ops::One, af::ops::One};
  *sum.y.repeats = {af::ops::Zero, af::ops::Zero};

  Cast cast("cast");
  cast.attr.sched.axis = {z0.id, z1.id};
  cast.x = sum.y;
  *cast.y.axis = {z0.id, z1.id};
  cast.y.dtype = af::DT_FLOAT16;
  *cast.y.axis = {z0.id, z1.id};
  *cast.y.strides = {af::ops::One, af::ops::One};
  *cast.y.repeats = {af::ops::Zero, af::ops::Zero};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.x = cast.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = af::DT_FLOAT16;
  *store_op1.y.axis = {z0.id, z1.id};
  *store_op1.y.strides = {af::ops::One, af::ops::One};
  *store_op1.y.repeats = {af::ops::Zero, af::ops::Zero};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = af::DT_FLOAT16;
  output_op.ir_attr.SetIndex(0);
}

AscGraph ConstructNormStruct4MulReduce(const std::string &name) {
  auto s0 = Sym(256);
  auto s1 = Sym(39);
  auto s2 = Sym(80);
  return AscGraphBuilder(name)
      .Loops({s0, s1, s2})
      .Data("data0", 0, {s0, s1, af::ops::One}, {s1, af::ops::One, af::ops::Zero}, af::DT_FLOAT)
      .Load("load0", "data0", {s0, s1, af::ops::One}, {s1, af::ops::One, af::ops::Zero})
      .Broadcast("brc", "load0", {s0, s1, s2})
      .Data("data1", 1, {s0, s1, s2}, {s1 * s2, s2, af::ops::One}, af::DT_FLOAT)
      .Load("load1", "data1")
      .Mul("mul", "brc", "load1")
      .Store("store1", "mul")
      .Output("output", "store1", 0, af::DT_FLOAT)
      .Sum("sum", "mul", {1})
      .Store("store2", "sum")
      .Output("output2", "store2", 1, af::DT_FLOAT)
      .Mul("mul1", "mul", "mul")
      .Sum("sum1", "mul1", {1})
      .Store("store3", "sum1")
      .Output("output3", "store3", 2, af::DT_FLOAT)
      .Build();
}

AscGraph ConstructNormStruct3ElemwiseReducePostMulInput(const std::string &name) {
  return AscGraphBuilder(name)
      .Loops({Sym(128), Sym(64)})
      .Data("data0", 0)
      .Load("load0", "data0")
      .Data("data1", 1)
      .Load("load1", "data1")
      .Mul("mul", "load0", "load1")
      .Sum("sum", "mul", {0, 1})
      .Relu("b0_relu", "sum")
      .Op<af::ascir_op::Tanh>("tanh", {"b0_relu"})
      .Add("add", "tanh", "b0_relu")
      .Store("store1", "add")
      .Output("output", "store1", 0)
      .Build();
}

AscGraph ConstructNormStruct3ElemwiseReducePostMulInputV2(const std::string &name) {
  return AscGraphBuilder(name)
      .Loops({Sym(128), Sym(64)})
      .Data("data0", 0)
      .Load("load0", "data0")
      .Data("data1", 1)
      .Load("load1", "data1")
      .Mul("mul", "load0", "load1")
      .Sum("sum", "mul", {0, 1})
      .Relu("b0_relu", "sum")
      .Op<af::ascir_op::Tanh>("tanh", {"b0_relu"})
      .Add("add", "tanh", "mul")
      .Store("store1", "add")
      .Output("output", "store1", 0)
      .Build();
}

AscGraph ConstructNormStruct4Elewise4ReduceMultipleCitationsMulOut(const std::string &name) {
  return AscGraphBuilder(name)
      .Loops({Sym(128), Sym(64)})
      .Data("data", 0)
      .Load("load", "data")
      .Sum("sum", "load", {0, 1})
      .Abs("abs", "sum")
      .Op<af::ascir_op::Tanh>("tanh", {"sum"})
      .Add("add", "sum", "tanh")
      .Relu("b0_relu", "sum")
      .Store("store1", "b0_relu")
      .Output("output", "store1", 0)
      .Store("store2", "add")
      .Output("output1", "store2", 1)
      .Store("store3", "abs")
      .Output("output2", "store3", 2)
      .Build();
}

// 双reduce水平融合 + 多引用结构（load0同输入喂abs0/abs1两路，两路reduce规约轴相同）：
// 走全流程Optimize验证loop group排序路径（BufQueAllocator在AutoScheduler后以v2模式重排序）
AscGraph ConstructTwoReduceSameAxisHorizontalFusion(const std::string &name) {
  return AscGraphBuilder(name)
      .Loops({Sym(128), Sym(64)})
      .Data("data0", 0)
      .Load("load0", "data0")
      .Abs("abs0", "load0")
      .Sum("sum0", "abs0", {1})
      .Relu("relu0", "sum0")
      .Store("store0", "relu0")
      .Output("output0", "store0", 0, af::DT_FLOAT)
      .Abs("abs1", "load0")
      .Sum("sum1", "abs1", {1})
      .Relu("relu1", "sum1")
      .Store("store1", "relu1")
      .Output("output1", "store1", 1, af::DT_FLOAT)
      .Build();
}

namespace optimize {
class OptimizerReduceSt : public ::testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }

  optimize::Optimizer optimizer;

  OptimizerReduceSt() : optimizer(optimize::OptimizerOptions{}) {}

  static std::string ExpressToStr(std::vector<af::Expression> exprs) {
    std::stringstream ss;
    for (auto &size_expr : exprs) {
      ss << std::string(size_expr.Str().get()) << ", ";
    }
    return ss.str();
  }

  static std::string RepeatsToStr(const af::AscGraph &graph, const char *node_name) {
    auto node = graph.FindNode(node_name);
    if (node == nullptr) {
      return "";
    }
    return ExpressToStr(node->outputs[0].attr.repeats);
  }

  static std::string StridesToStr(const af::AscGraph &graph, const char *node_name) {
    auto node = graph.FindNode(node_name);
    if (node == nullptr) {
      return "";
    }
    return ExpressToStr(node->outputs[0].attr.strides);
  }

  static std::string AxisToStr(af::AscGraph &graph, const char *node_name) {
    auto node = graph.FindNode(node_name);
    if (node == nullptr) {
      return "";
    }
    std::stringstream ss;
    for (auto axis_id : node->outputs[0].attr.axis) {
      ss << graph.FindAxis(axis_id)->name << ", ";
    }
    return ss.str();
  }
};

TEST_F(OptimizerReduceSt, TestReduce_RARA) {
  auto graph = Construct_Reduce_RARA("REDUCE_RARA");
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  // RARA 单 reduce 双 R 轴: phase-1 各 tiling case 切分结构不同(relation 不一致), AppendReduceFirstStageGroup
  // 回退逐 case 展开, 每个展开结果携带该 case 自身的 relation; 6 = Common 1 + RCore 展开 4 + AllLoad 1
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 6UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 4UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups.size(), 2UL);
}

TEST_F(OptimizerReduceSt, TestReduce_MUL_CONSUMER) {
  auto graph = Construct_Mul_Consumer_Struct("REDUCE_MUL_CONSUMER");
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 2UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups.size(), 3UL);
}

TEST_F(OptimizerReduceSt, TestReduce_ARAR) {
  auto graph = Construct_Reduce_ARAR("REDUCE_ARAR");
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  // ARAR 同 RARA: 单 reduce 双 R 轴 relation 不一致, 走逐 case 展开回退路径,
  // 6 = Common 1 + RCore 展开 4 + AllLoad 1
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 6UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 4UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups.size(), 2UL);
}

TEST_F(OptimizerReduceSt, TestReduce_RR) {
  af::AscGraph graph("REDUCE_RR");
  Construct_Reduce_RR(graph);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  // R-core phase-1 两个 tiling case 收敛为同 group 多 impl, RCore 任务的结果数 2 -> 1, 调度结果由 4 减为 3
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 3UL);
}

TEST_F(OptimizerReduceSt, TestReduce_Cast_RR) {
  af::AscGraph graph("REDUCE_Cast_RR");
  Construct_Reduce_Cast_RR(graph);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  // 同 TestReduce_RR: R-core phase-1 多 case 收敛为同 group 多 impl, 调度结果由 4 减为 3
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 3UL);
}

TEST_F(OptimizerReduceSt, TestReduce_PatitionNorm) {
  auto graph = ConstructNormStruct("reduce_patition_norm");
  ::ascir::FusedScheduledResult fused_scheduled_result;
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  optimizer.Optimize(graph, fused_scheduled_result);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL].size(), 3UL);
}

TEST_F(OptimizerReduceSt, TestReduce_Three_Elewise_Store) {
  auto graph = ConstructNormStruct3Elewise("reduce_three_elewise_store");
  ::ascir::FusedScheduledResult fused_scheduled_result;
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  optimizer.Optimize(graph, fused_scheduled_result);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL].size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][0UL].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][1UL].schedule_groups.size(), 2UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][2UL].schedule_groups.size(), 1UL);
}

TEST_F(OptimizerReduceSt, TestReduce_One_Elewise_Store) {
  auto graph = ConstructNormStruct1Elewise("reduce_one_elewise_store");
  ::ascir::FusedScheduledResult fused_scheduled_result;
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  optimizer.Optimize(graph, fused_scheduled_result);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][0UL].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][1UL].schedule_groups.size(), 2UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][2UL].schedule_groups.size(), 1UL);
}

TEST_F(OptimizerReduceSt, TestReduce_Four_Elewise_Store) {
  auto graph = ConstructNormStruct4Elewise("reduce_four_elewise_store");
  ::ascir::FusedScheduledResult fused_scheduled_result;
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  optimizer.Optimize(graph, fused_scheduled_result);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL].size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][0UL].schedule_groups.size(), 2UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][1UL].schedule_groups.size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][2UL].schedule_groups.size(), 1UL);
}

TEST_F(OptimizerReduceSt, TestReduce_Four_Elewise_Store_V2) {
  auto graph = ConstructNormStruct4Elewise4ReduceMultipleCitations("reduce_four_elewise_store");
  ::ascir::FusedScheduledResult fused_scheduled_result;
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  optimizer.Optimize(graph, fused_scheduled_result);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL].size(), 3UL);
}

TEST_F(OptimizerReduceSt, TestReduce_Three_Elewise_Store_Multi_Citation) {
  auto graph = ConstructNormStruct4Elewise3ReduceMultipleCitations("reduce_three_elewise_store");
  ::ascir::FusedScheduledResult fused_scheduled_result;
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  optimizer.Optimize(graph, fused_scheduled_result);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL].size(), 3UL);
}

TEST_F(OptimizerReduceSt, TestReduce_Four_Elewise_Store_V3) {
  auto graph = ConstructNormStruct4Elewise3("reduce_four_elewise_store");
  ::ascir::FusedScheduledResult fused_scheduled_result;
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  optimizer.Optimize(graph, fused_scheduled_result);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL].size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][0UL].schedule_groups.size(), 2UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][1UL].schedule_groups.size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][2UL].schedule_groups.size(), 1UL);
}

TEST_F(OptimizerReduceSt, TestReduce_Elewise_Store_MulReduce) {
  auto graph = ConstructNormStruct4MulReduce("reduce_four_elewise_store");
  ::ascir::FusedScheduledResult fused_scheduled_result;
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), af::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  // 双 reduce 链放开 R 轴分核后新增 RCore 调度结果, phase-1 多 tiling case 收敛为同 group 多
  // impl, 共 3 个结果(Common/RCore/AllLoad 各一)
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL].size(), 3UL);
}

TEST_F(OptimizerReduceSt, TestReduce_Three_Elewise_Reduce_Post_Node_Multi_Input_V1) {
  auto graph = ConstructNormStruct3ElemwiseReducePostMulInput("reduce_three_elewise_store");
  ::ascir::FusedScheduledResult fused_scheduled_result;
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  optimizer.Optimize(graph, fused_scheduled_result);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL].size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][0UL].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][1UL].schedule_groups.size(), 2UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL][2UL].schedule_groups.size(), 1UL);
}

TEST_F(OptimizerReduceSt, TestReduce_Three_Elewise_Reduce_Post_Node_Multi_Input_V2) {
  auto graph = ConstructNormStruct3ElemwiseReducePostMulInputV2("reduce_three_elewise_store");
  ::ascir::FusedScheduledResult fused_scheduled_result;
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  optimizer.Optimize(graph, fused_scheduled_result);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL].size(), 3UL);
}

TEST_F(OptimizerReduceSt, TestReduce_Three_Elewise_Store_Multi_Citation_Multi_Out) {
  auto graph = ConstructNormStruct4Elewise4ReduceMultipleCitationsMulOut("reduce_three_elewise_store");
  ::ascir::FusedScheduledResult fused_scheduled_result;
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  optimizer.Optimize(graph, fused_scheduled_result);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0UL].size(), 3UL);
}

// 双reduce水平融合（同输入不同输出、规约轴相同）场景走全流程Optimize，
// 验证loop group排序分支与调度/内存分配流程兼容
TEST_F(OptimizerReduceSt, TestReduce_LoopGroupSortTwoReduceSameAxis) {
  auto graph = ConstructTwoReduceSameAxisHorizontalFusion("reduce_loop_group_sort");
  ::ascir::FusedScheduledResult fused_scheduled_result;
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_FALSE(fused_scheduled_result.node_idx_to_scheduled_results[0].empty());
  ASSERT_FALSE(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.empty());
  ASSERT_FALSE(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.empty());
}
}  // namespace optimize
