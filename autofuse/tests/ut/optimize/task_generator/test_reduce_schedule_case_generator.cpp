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
#include "ascir_ops_utils.h"
#include "ascir_utils.h"
#include "asc_graph_utils.h"
#include "ascir_ops.h"
#include "task_generator/schedule_case_generator.h"
#include "task_generator/reduce_schedule_case_generator.h"
#include "asc_graph_builder.h"

namespace schedule {
using namespace optimize;
using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;
using af::testing::AscGraphBuilder;
using af::testing::Sym;

class ReduceScheduleCaseGeneratorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }

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

bool HasReduceTemplateType(const std::vector<ScheduleTask> &tasks, ReduceTemplateType reduce_type) {
  for (const auto &task : tasks) {
    if (task.reduce_type == reduce_type) {
      return true;
    }
  }
  return false;
}

// 统计公共任务(kCommon)各子图内指定名字的节点个数: 子图划分与子图内合并动作都作用在
// 公共任务上, RCore/AllLoad 任务由其派生, 不计入避免重复统计
size_t CountCommonTaskNodesByName(const std::vector<ScheduleTask> &tasks, const std::string &node_name) {
  size_t count = 0UL;
  for (const auto &task : tasks) {
    if (task.reduce_type != ReduceTemplateType::kCommon) {
      continue;
    }
    for (const auto &grouped_graph : task.grouped_graphs) {
      for (const auto &node : grouped_graph.GetAllNodes()) {
        if (node->GetName() == node_name) {
          ++count;
        }
      }
    }
  }
  return count;
}

void ConstructFourAxisReduce(af::AscGraph &graph, bool tail_reduce) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto s2 = graph.CreateSizeVar(32);
  auto s3 = graph.CreateSizeVar(16);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  std::vector<af::AxisId> axes = {z0.id, z1.id, z2.id, z3.id};

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = axes;
  load.x = data.y;
  *load.y.axis = axes;
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1 * s2 * s3, s2 * s3, s3, af::ops::One};
  *load.y.repeats = {s0, s1, s2, s3};

  Sum sum("sum");
  sum.attr.sched.axis = axes;
  sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
  sum.x = load.y;
  *sum.y.axis = axes;
  sum.y.dtype = ge::DT_FLOAT;
  if (tail_reduce) {
    *sum.y.repeats = {s0, s1, s2, af::ops::One};
    *sum.y.strides = {s1 * s2, s2, af::ops::One, af::ops::Zero};
  } else {
    *sum.y.repeats = {s0, s1, s2, s3};
    *sum.y.strides = {s1 * s2 * s3, s2 * s3, s3, af::ops::One};
  }

  Store store_op("store");
  store_op.attr.sched.axis = axes;
  store_op.attr.api.compute_type = af::ComputeType::kComputeStore;
  store_op.x = sum.y;
  *store_op.y.axis = *sum.y.axis;
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = *sum.y.strides;
  *store_op.y.repeats = *sum.y.repeats;

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

void ConstructEmptyRepeatReduce(af::AscGraph &graph) {
  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.x = data.y;
  load.y.dtype = ge::DT_FLOAT;

  Sum sum("sum");
  sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
  sum.x = load.y;
  sum.y.dtype = ge::DT_FLOAT;

  Store store_op("store");
  store_op.attr.api.compute_type = af::ComputeType::kComputeStore;
  store_op.x = sum.y;
  store_op.y.dtype = ge::DT_FLOAT;

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

void ConstructFourAxisSoftmax(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto s2 = graph.CreateSizeVar(32);
  auto s3 = graph.CreateSizeVar(16);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  std::vector<af::AxisId> axes = {z0.id, z1.id, z2.id, z3.id};

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = axes;
  load.x = data.y;
  *load.y.axis = axes;
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1 * s2 * s3, s2 * s3, s3, af::ops::One};
  *load.y.repeats = {s0, s1, s2, s3};

  Softmax softmax("softmax");
  softmax.attr.sched.axis = axes;
  softmax.attr.api.compute_type = af::ComputeType::kComputeReduce;
  softmax.x = load.y;
  *softmax.y.axis = axes;
  softmax.y.dtype = ge::DT_FLOAT;
  *softmax.y.repeats = {s0, s1, s2, s3};
  *softmax.y.strides = {s1 * s2 * s3, s2 * s3, s3, af::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = axes;
  store_op.attr.api.compute_type = af::ComputeType::kComputeStore;
  store_op.x = softmax.y;
  *store_op.y.axis = *softmax.y.axis;
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = *softmax.y.strides;
  *store_op.y.repeats = *softmax.y.repeats;

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

void ConstructNormStruct3Elewise(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s0);

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1, af::ops::One};
  *load.y.repeats = {s0, s1};

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
  sum.x = load.y;
  *sum.y.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.repeats = {af::ops::One, af::ops::One};
  *sum.y.strides = {af::ops::Zero, af::ops::Zero};

  Abs abs("abs");
  abs.x = sum.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  *abs.y.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.strides = {af::ops::Zero, af::ops::Zero};
  *abs.y.repeats = {af::ops::One, af::ops::One};

  Exp exp("exp");
  exp.x = abs.y;
  exp.attr.sched.axis = {z0.id, z1.id};
  exp.attr.api.compute_type = af::ComputeType::kComputeElewise;
  *exp.y.axis = {z0.id, z1.id};
  exp.y.dtype = ge::DT_FLOAT;
  *exp.y.strides = {af::ops::Zero, af::ops::Zero};
  *exp.y.repeats = {af::ops::One, af::ops::One};

  af::ascir_op::Relu b0_relu("b0_relu");
  b0_relu.x = exp.y;
  b0_relu.attr.sched.axis = {z0.id, z1.id};
  b0_relu.attr.api.compute_type = af::ComputeType::kComputeElewise;
  b0_relu.y.dtype = ge::DT_FLOAT;
  *b0_relu.y.axis = {z0.id, z1.id};
  *b0_relu.y.repeats = {af::ops::One, af::ops::One};
  *b0_relu.y.strides = {af::ops::Zero, af::ops::Zero};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.attr.api.compute_type = af::ComputeType::kComputeStore;
  store_op1.x = b0_relu.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_FLOAT;
  *store_op1.y.strides = {af::ops::Zero, af::ops::Zero};
  *store_op1.y.repeats = {af::ops::One, af::ops::One};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

void ConstructNormStruct1Elewise(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s0);

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1, af::ops::One};
  *load.y.repeats = {s0, s1};

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
  sum.x = load.y;
  *sum.y.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.repeats = {af::ops::One, af::ops::One};
  *sum.y.strides = {af::ops::Zero, af::ops::Zero};

  Abs abs("abs");
  abs.x = sum.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  *abs.y.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.strides = {af::ops::Zero, af::ops::Zero};
  *abs.y.repeats = {af::ops::One, af::ops::One};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.attr.api.compute_type = af::ComputeType::kComputeStore;
  store_op1.x = abs.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_FLOAT;
  *store_op1.y.strides = {af::ops::Zero, af::ops::Zero};
  *store_op1.y.repeats = {af::ops::One, af::ops::One};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

void ConstructNormStructMultiplyCitations(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s0);

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1, af::ops::One};
  *load.y.repeats = {s0, s1};

  Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.ir_attr.SetIndex(1);

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.x = data.y;
  *load1.y.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.strides = {s1, af::ops::One};
  *load1.y.repeats = {s0, s1};

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
  sum.x = load.y;
  *sum.y.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.repeats = {af::ops::One, af::ops::One};
  *sum.y.strides = {af::ops::Zero, af::ops::Zero};

  Add add("add");
  add.x1 = sum.y;
  add.x2 = load1.y;
  add.attr.sched.axis = {z0.id, z1.id};
  add.attr.api.compute_type = af::ComputeType::kComputeElewise;
  *add.y.axis = {z0.id, z1.id};
  add.y.dtype = ge::DT_FLOAT;
  *add.y.strides = {s1, One};
  *add.y.repeats = {s0, s1};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.attr.api.compute_type = af::ComputeType::kComputeStore;
  store_op1.x = add.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_FLOAT;
  *store_op1.y.strides = {s1, One};
  *store_op1.y.repeats = {s0, s1};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

void ConstructNormStruct4Elewise(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s0);

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1, af::ops::One};
  *load.y.repeats = {s0, s1};

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
  sum.x = load.y;
  *sum.y.axis = {z0.id, z1.id};
  sum.y.dtype = ge::DT_FLOAT;
  *sum.y.repeats = {af::ops::One, af::ops::One};
  *sum.y.strides = {af::ops::Zero, af::ops::Zero};

  Abs abs("abs");
  abs.x = sum.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  *abs.y.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.strides = {af::ops::Zero, af::ops::Zero};
  *abs.y.repeats = {af::ops::One, af::ops::One};

  Tanh tanh("tanh");
  tanh.x = abs.y;
  tanh.attr.sched.axis = {z0.id, z1.id};
  tanh.attr.api.compute_type = af::ComputeType::kComputeElewise;
  *tanh.y.axis = {z0.id, z1.id};
  tanh.y.dtype = ge::DT_FLOAT;
  *tanh.y.strides = {af::ops::Zero, af::ops::Zero};
  *tanh.y.repeats = {af::ops::One, af::ops::One};

  Exp exp("exp");
  exp.x = tanh.y;
  exp.attr.sched.axis = {z0.id, z1.id};
  exp.attr.api.compute_type = af::ComputeType::kComputeElewise;
  *exp.y.axis = {z0.id, z1.id};
  exp.y.dtype = ge::DT_FLOAT;
  *exp.y.strides = {af::ops::Zero, af::ops::Zero};
  *exp.y.repeats = {af::ops::One, af::ops::One};

  af::ascir_op::Relu b0_relu("b0_relu");
  b0_relu.x = exp.y;
  b0_relu.attr.sched.axis = {z0.id, z1.id};
  b0_relu.attr.api.compute_type = af::ComputeType::kComputeElewise;
  b0_relu.y.dtype = ge::DT_FLOAT;
  *b0_relu.y.axis = {z0.id, z1.id};
  *b0_relu.y.repeats = {af::ops::One, af::ops::One};
  *b0_relu.y.strides = {af::ops::Zero, af::ops::Zero};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.attr.api.compute_type = af::ComputeType::kComputeStore;
  store_op1.x = b0_relu.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_FLOAT;
  *store_op1.y.strides = {af::ops::Zero, af::ops::Zero};
  *store_op1.y.repeats = {af::ops::One, af::ops::One};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

TEST_F(ReduceScheduleCaseGeneratorTest, TestReduce_Three_Elewise_Store) {
  af::AscGraph graph("reduce_Three_elewise_store");
  ConstructNormStruct3Elewise(graph);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 2UL);
}

TEST_F(ReduceScheduleCaseGeneratorTest, TestReduce_One_Elewise_Store) {
  af::AscGraph graph("reduce_one_elewise_store");
  ConstructNormStruct1Elewise(graph);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  generator.GeneratorTask(graph, tasks, options);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 2UL);
}

TEST_F(ReduceScheduleCaseGeneratorTest, TestReduce_Four_Elewise_Store) {
  af::AscGraph graph("reduce_four_elewise_store");
  ConstructNormStruct4Elewise(graph);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  generator.GeneratorTask(graph, tasks, options);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 2UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 3UL);
}

TEST_F(ReduceScheduleCaseGeneratorTest, TestReduce_AllLoadTailReduceFourAxis) {
  af::AscGraph graph("reduce_all_load_tail_reduce_four_axis");
  ConstructFourAxisReduce(graph, true);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  EXPECT_TRUE(HasReduceTemplateType(tasks, ReduceTemplateType::kAllLoad));
}

TEST_F(ReduceScheduleCaseGeneratorTest, TestReduce_NoAllLoadForFourAxisNonTailReduce) {
  af::AscGraph graph("reduce_no_all_load_for_four_axis_non_tail_reduce");
  ConstructFourAxisReduce(graph, false);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  EXPECT_FALSE(HasReduceTemplateType(tasks, ReduceTemplateType::kAllLoad));
}

TEST_F(ReduceScheduleCaseGeneratorTest, TestReduce_NoAllLoadForEmptyRepeats) {
  af::AscGraph graph("reduce_no_all_load_for_empty_repeats");
  ConstructEmptyRepeatReduce(graph);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  EXPECT_FALSE(HasReduceTemplateType(tasks, ReduceTemplateType::kAllLoad));
}

TEST_F(ReduceScheduleCaseGeneratorTest, TestReduce_AllLoadForSoftmaxWithUnchangedTailRepeat) {
  af::AscGraph graph("reduce_all_load_for_softmax_with_unchanged_tail_repeat");
  ConstructFourAxisSoftmax(graph);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  EXPECT_TRUE(HasReduceTemplateType(tasks, ReduceTemplateType::kAllLoad));
}

TEST_F(ReduceScheduleCaseGeneratorTest, TestReduce_Multi_Cita_Store) {
  af::AscGraph graph("reduce_multi_citation_store");
  ConstructNormStructMultiplyCitations(graph);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  generator.GeneratorTask(graph, tasks, options);
  EXPECT_TRUE(HasReduceTemplateType(tasks, ReduceTemplateType::kCommon));
  EXPECT_FALSE(HasReduceTemplateType(tasks, ReduceTemplateType::kAllLoad));
}

TEST_F(ReduceScheduleCaseGeneratorTest, TestReduce_Multi_Cita_Multi_Out_NoDependencyCycle) {
  // 双 source 多输出交叉链：shared0/shared1 各自的 citation group 共享 sum0（经 merge0 汇聚），
  // 触发并查集合并；sum0 为最小 id 的 anchor。非 anchor citation（branch01→sum2、branch11→sum1）
  // 被切分，且断边后 source 与 citation 不再连通，满足 CheckDuplicateWorkspaceNodes 对
  // workspace 读写端必须分属不同子图的约束（PR !1914 引入）。
  auto graph = AscGraphBuilder("reduce_multi_citation_multi_out")
                   .Loops({Sym(128), Sym(64)})
                   .Data("data0", 0)
                   .Load("load0", "data0")
                   .Data("data1", 1)
                   .Load("load1", "data1")
                   .Abs("shared0", "load0")
                   .Relu("branch00", "shared0")
                   .Op<af::ascir_op::Tanh>("branch01", {"shared0"})
                   .Abs("shared1", "load1")
                   .Relu("branch10", "shared1")
                   .Op<af::ascir_op::Tanh>("branch11", {"shared1"})
                   .Add("merge0", "branch00", "branch10")
                   .Sum("sum0", "merge0", {0, 1})
                   .Sum("sum1", "branch11", {0, 1})
                   .Sum("sum2", "branch01", {0, 1})
                   .Store("store0", "sum0")
                   .Output("output0", "store0", 0)
                   .Store("store1", "sum1")
                   .Output("output1", "store1", 1)
                   .Store("store2", "sum2")
                   .Output("output2", "store2", 2)
                   .Build();
  std::vector<ScheduleTask> tasks;
  ReducePartitionCaseGenerator generator;
  OptimizerOptions options;

  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  EXPECT_TRUE(HasReduceTemplateType(tasks, ReduceTemplateType::kCommon));

  size_t shared0_workspace_count = 0UL;
  size_t shared1_workspace_count = 0UL;
  for (const auto &task : tasks) {
    for (const auto &grouped_graph : task.grouped_graphs) {
      for (const auto &node : grouped_graph.GetAllNodes()) {
        if (!af::ops::IsOps<af::ascir_op::Workspace>(node)) {
          continue;
        }
        if (node->GetName().find("shared0_to_branch") != std::string::npos) {
          ++shared0_workspace_count;
        }
        if (node->GetName().find("shared1_to_branch") != std::string::npos) {
          ++shared1_workspace_count;
        }
      }
    }
  }
  // Both citation groups share sum0 and must be merged; sum0 is the lowest-id anchor.
  // Each non-anchor source/reduce pair is partitioned once, producing one workspace pair.
  EXPECT_EQ(shared0_workspace_count, 2UL);
  EXPECT_EQ(shared1_workspace_count, 2UL);
}

TEST_F(ReduceScheduleCaseGeneratorTest, TestReduce_Multi_Cita_SameReduce_NoDuplicatePartition) {
  auto graph = AscGraphBuilder("reduce_multi_citation_same_reduce")
                   .Loops({Sym(128), Sym(64)})
                   .Data("data", 0)
                   .Load("load", "data")
                   .Abs("shared", "load")
                   .Relu("branch0", "shared")
                   .Op<af::ascir_op::Tanh>("branch1", {"shared"})
                   .Add("merge", "branch0", "branch1")
                   .Sum("sum0", "merge", {0, 1})
                   .Relu("branch2", "shared")
                   .Sum("sum1", "branch2", {0, 1})
                   .Store("store0", "sum0")
                   .Output("output0", "store0", 0)
                   .Store("store1", "sum1")
                   .Output("output1", "store1", 1)
                   .Build();
  std::vector<ScheduleTask> tasks;
  ReducePartitionCaseGenerator generator;
  OptimizerOptions options;

  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);

  size_t shared_workspace_count = 0UL;
  for (const auto &task : tasks) {
    for (const auto &grouped_graph : task.grouped_graphs) {
      for (const auto &node : grouped_graph.GetAllNodes()) {
        if (af::ops::IsOps<af::ascir_op::Workspace>(node) &&
            node->GetName().find("shared_to_branch") != std::string::npos) {
          ++shared_workspace_count;
        }
      }
    }
  }
  // One partition creates a workspace pair; duplicate citation partitioning would create two pairs.
  EXPECT_EQ(shared_workspace_count, 2UL);
}

TEST_F(ReduceScheduleCaseGeneratorTest, TestReduce_Multi_Cita_Data_Source) {
  // 单 Data 多引用：data 同时被 load0/load1 引用，两路分别到达 sum0/sum1 两个不同 reduce，
  // 非锚点 citation（load1→sum1）触发对 Data 源的切分。Data 是 GM 输入占位符（kAPITypeBuffer），
  // 不能走通用路径物化（Data 直连 Store 会生成非法的 DataCopyPadExtend(GlobalTensor, ...)），
  // 应复制 copy_from_ Data 并重定向 citation 边，两个子图各自重读同一 GM 输入。
  auto graph = AscGraphBuilder("reduce_multi_citation_data_source")
                   .Loops({Sym(128), Sym(64)})
                   .Data("data", 0)
                   .Load("load0", "data")
                   .Load("load1", "data")
                   .Sum("sum0", "load0", {0, 1})
                   .Sum("sum1", "load1", {0, 1})
                   .Store("store0", "sum0")
                   .Output("output0", "store0", 0)
                   .Store("store1", "sum1")
                   .Output("output1", "store1", 1)
                   .Build();
  std::vector<ScheduleTask> tasks;
  ReducePartitionCaseGenerator generator;
  OptimizerOptions options;

  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);

  bool has_copy_from_data = false;
  bool has_data_to_store = false;
  bool load1_redirected = false;
  for (const auto &task : tasks) {
    for (const auto &grouped_graph : task.grouped_graphs) {
      for (const auto &node : grouped_graph.GetAllNodes()) {
        if (af::ops::IsOps<af::ascir_op::Data>(node) && node->GetName() == "copy_from_data") {
          has_copy_from_data = true;
        }
        if (af::ops::IsOps<af::ascir_op::Store>(node)) {
          for (const auto &in_node : node->GetInDataNodes()) {
            has_data_to_store = has_data_to_store || af::ops::IsOps<af::ascir_op::Data>(in_node);
          }
        }
      }
      auto load1_node = grouped_graph.FindNode("load1");
      if (load1_node != nullptr && load1_node->GetInDataNodesSize() == 1UL &&
          std::string(load1_node->GetInDataNodes().at(0UL)->GetName()) == "copy_from_data") {
        load1_redirected = true;
      }
    }
  }
  EXPECT_TRUE(has_copy_from_data);
  EXPECT_FALSE(has_data_to_store);
  EXPECT_TRUE(load1_redirected);
}

void ConstructReduceWithScalarData(AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s0);

  Data data("data", graph);
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  *load.y.strides = {s1, af::ops::One};
  *load.y.repeats = {s0, s1};

  Sum sum("sum");
  sum.attr.sched.axis = {z0.id, z1.id};
  sum.x = load.y;
  *sum.y.axis = {z0.id, z1.id};
  *sum.y.repeats = {af::ops::One, af::ops::One};
  *sum.y.strides = {af::ops::Zero, af::ops::Zero};

  af::ascir_op::ScalarData scalar_data("scalar_data", graph);
  scalar_data.ir_attr.SetIndex(1);

  Abs abs("abs");
  abs.x = sum.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.strides = {af::ops::Zero, af::ops::Zero};
  *abs.y.repeats = {af::ops::One, af::ops::One};

  Add add("add");
  add.x1 = abs.y;
  add.x2 = scalar_data.y;
  add.attr.sched.axis = {z0.id, z1.id};
  *add.y.axis = {z0.id, z1.id};
  *add.y.strides = {af::ops::Zero, af::ops::Zero};
  *add.y.repeats = {af::ops::One, af::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = add.y;
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.strides = {af::ops::Zero, af::ops::Zero};
  *store_op.y.repeats = {af::ops::One, af::ops::One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.ir_attr.SetIndex(0);
}

TEST_F(ReduceScheduleCaseGeneratorTest, TestReduce_ScalarData_Input) {
  af::AscGraph graph("reduce_scalar_data_input");
  ConstructReduceWithScalarData(graph);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  ASSERT_EQ(tasks.size(), 2UL);
  // ScalarData直连算子，reduce拆分时ScalarData走PartitionScalar分支，保持ScalarData类型
  for (const auto &task : tasks) {
    for (const auto &grouped_graph : task.grouped_graphs) {
      auto scalar_data_node = grouped_graph.FindNode("copy_from_scalar_data");
      if (scalar_data_node != nullptr) {
        EXPECT_EQ(std::string(scalar_data_node->GetTypePtr()), "ScalarData");
      }
    }
  }
}

TEST_F(ReduceScheduleCaseGeneratorTest, TestReduce_Multi_Cita_MergeCopiesInSubGraph) {
  // 同一 source(load) 的两条非锚点引用(c1/c2) 经 join 汇聚到同一个 reduce(sum2), 两次切分
  // 产生的同名 copy_from_load/copy_from_data 副本落在同一子图, 应合并为一份
  auto graph = AscGraphBuilder("reduce_multi_citation_merge_copies")
                   .Loops({Sym(128), Sym(64)})
                   .Data("data", 0)
                   .Load("load", "data")
                   .Abs("c0", "load")
                   .Sum("sum0", "c0", {0, 1})
                   .Relu("c1", "load")
                   .Op<af::ascir_op::Tanh>("c2", {"load"})
                   .Add("join", "c1", "c2")
                   .Sum("sum2", "join", {0, 1})
                   .Store("store0", "sum0")
                   .Output("output0", "store0", 0)
                   .Store("store2", "sum2")
                   .Output("output2", "store2", 1)
                   .Build();
  std::vector<ScheduleTask> tasks;
  ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  EXPECT_TRUE(HasReduceTemplateType(tasks, ReduceTemplateType::kCommon));

  // 合并后公共任务内同名副本各只剩一份
  EXPECT_EQ(CountCommonTaskNodesByName(tasks, "copy_from_load"), 1UL);
  EXPECT_EQ(CountCommonTaskNodesByName(tasks, "copy_from_data"), 1UL);

  // 保留下来的副本同时服务 c1 与 c2
  bool feeds_both = false;
  for (const auto &task : tasks) {
    if (task.reduce_type != ReduceTemplateType::kCommon) {
      continue;
    }
    for (const auto &grouped_graph : task.grouped_graphs) {
      auto copy_node = grouped_graph.FindNode("copy_from_load");
      if (copy_node == nullptr) {
        continue;
      }
      bool has_c1 = false;
      bool has_c2 = false;
      for (const auto &out_node : copy_node->GetOutDataNodes()) {
        has_c1 = has_c1 || (out_node->GetName() == "c1");
        has_c2 = has_c2 || (out_node->GetName() == "c2");
      }
      feeds_both = feeds_both || (has_c1 && has_c2);
    }
  }
  EXPECT_TRUE(feeds_both);
}

TEST_F(ReduceScheduleCaseGeneratorTest, TestReduce_Multi_Cita_RemoveOrphanSourceKeepCrossSubGraphCopies) {
  // src0 的两条引用 c1/c2 都是非锚点(并查集合并后锚点取自 src1 的 sum_small), 两条边都被切走后
  // src0 失去全部消费者: 整图清扫应摘除 src0 及其失去消费者的输入 d0, 并剔除切分根序列中的
  // src0(否则会被重建成孤立子图); src0 的两份副本分属不同子图, 不做合并
  auto graph = AscGraphBuilder("reduce_multi_citation_orphan_source")
                   .Loops({Sym(128), Sym(64)})
                   .Data("d0", 0)
                   .Load("src0", "d0")
                   .Data("d1", 1)
                   .Load("src1", "d1")
                   .Relu("c1", "src0")
                   .Op<af::ascir_op::Tanh>("c2", {"src0"})
                   .Relu("t1", "src1")
                   .Op<af::ascir_op::Tanh>("t2", {"src1"})
                   .Sum("sum_small", "t1", {0, 1})
                   .Add("shared_in", "c1", "t2")
                   .Sum("sum_shared", "shared_in", {0, 1})
                   .Sum("sum_other", "c2", {0, 1})
                   .Store("store0", "sum_small")
                   .Output("output0", "store0", 0)
                   .Store("store1", "sum_shared")
                   .Output("output1", "store1", 1)
                   .Store("store2", "sum_other")
                   .Output("output2", "store2", 2)
                   .Build();
  std::vector<ScheduleTask> tasks;
  ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  EXPECT_TRUE(HasReduceTemplateType(tasks, ReduceTemplateType::kCommon));

  // 悬空源节点及其已失去消费者的输入被摘除, 且不残留在切分根序列中(否则会重建出孤立子图)
  EXPECT_EQ(CountCommonTaskNodesByName(tasks, "src0"), 0UL);
  EXPECT_EQ(CountCommonTaskNodesByName(tasks, "d0"), 0UL);

  // src1 保留锚点引用(t1->sum_small), 原节点与副本正常保留
  EXPECT_EQ(CountCommonTaskNodesByName(tasks, "src1"), 1UL);
  EXPECT_EQ(CountCommonTaskNodesByName(tasks, "copy_from_src1"), 1UL);
  EXPECT_EQ(CountCommonTaskNodesByName(tasks, "copy_from_d1"), 1UL);

  // src0 的两份副本分属不同子图, 各自保留(不做跨子图合并)
  EXPECT_EQ(CountCommonTaskNodesByName(tasks, "copy_from_src0"), 2UL);
  EXPECT_EQ(CountCommonTaskNodesByName(tasks, "copy_from_d0"), 2UL);
  size_t subgraphs_with_src0_copy = 0UL;
  for (const auto &task : tasks) {
    if (task.reduce_type != ReduceTemplateType::kCommon) {
      continue;
    }
    for (const auto &grouped_graph : task.grouped_graphs) {
      if (grouped_graph.FindNode("copy_from_src0") != nullptr) {
        ++subgraphs_with_src0_copy;
      }
    }
  }
  EXPECT_EQ(subgraphs_with_src0_copy, 2UL);
}
}  // namespace schedule
