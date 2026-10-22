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

namespace schedule {
using namespace optimize;
using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;

class ArgMaxScheduleCaseGeneratorTest : public ::testing::Test {
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

void ConstructNormStruct3ElewiseArgMax(af::AscGraph &graph) {
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

  // ArgMax outputs int32 indices
  af::ascir_op::ArgMax argmax("argmax");
  argmax.attr.sched.axis = {z0.id, z1.id};
  argmax.attr.api.compute_type = af::ComputeType::kComputeReduce;
  argmax.x = load.y;
  *argmax.y.axis = {z0.id, z1.id};
  argmax.y.dtype = ge::DT_INT32;
  *argmax.y.repeats = {af::ops::One, af::ops::One};
  *argmax.y.strides = {af::ops::Zero, af::ops::Zero};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.attr.api.compute_type = af::ComputeType::kComputeStore;
  store_op1.x = argmax.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_INT32;
  *store_op1.y.strides = {af::ops::Zero, af::ops::Zero};
  *store_op1.y.repeats = {af::ops::One, af::ops::One};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_INT32;
  output_op.ir_attr.SetIndex(0);
}

void ConstructNormStruct1ElewiseArgMax(af::AscGraph &graph) {
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

  // ArgMax outputs int32 indices
  af::ascir_op::ArgMax argmax("argmax");
  argmax.attr.sched.axis = {z0.id, z1.id};
  argmax.attr.api.compute_type = af::ComputeType::kComputeReduce;
  argmax.x = load.y;
  *argmax.y.axis = {z0.id, z1.id};
  argmax.y.dtype = ge::DT_INT32;
  *argmax.y.repeats = {af::ops::One, af::ops::One};
  *argmax.y.strides = {af::ops::Zero, af::ops::Zero};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.attr.api.compute_type = af::ComputeType::kComputeStore;
  store_op1.x = argmax.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_INT32;
  *store_op1.y.strides = {af::ops::Zero, af::ops::Zero};
  *store_op1.y.repeats = {af::ops::One, af::ops::One};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_INT32;
  output_op.ir_attr.SetIndex(0);
}

void ConstructNormStructMultiplyCitationsArgMax(af::AscGraph &graph) {
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

  // ArgMax outputs int32 indices
  af::ascir_op::ArgMax argmax("argmax");
  argmax.attr.sched.axis = {z0.id, z1.id};
  argmax.attr.api.compute_type = af::ComputeType::kComputeReduce;
  argmax.x = load.y;
  *argmax.y.axis = {z0.id, z1.id};
  argmax.y.dtype = ge::DT_INT32;
  *argmax.y.repeats = {af::ops::One, af::ops::One};
  *argmax.y.strides = {af::ops::Zero, af::ops::Zero};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.attr.api.compute_type = af::ComputeType::kComputeStore;
  store_op1.x = argmax.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_INT32;
  *store_op1.y.strides = {af::ops::Zero, af::ops::Zero};
  *store_op1.y.repeats = {af::ops::One, af::ops::One};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_INT32;
  output_op.ir_attr.SetIndex(0);
}

void ConstructNormStruct4ElewiseArgMax(af::AscGraph &graph) {
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

  // ArgMax outputs int32 indices
  af::ascir_op::ArgMax argmax("argmax");
  argmax.attr.sched.axis = {z0.id, z1.id};
  argmax.attr.api.compute_type = af::ComputeType::kComputeReduce;
  argmax.x = load.y;
  *argmax.y.axis = {z0.id, z1.id};
  argmax.y.dtype = ge::DT_INT32;
  *argmax.y.repeats = {af::ops::One, af::ops::One};
  *argmax.y.strides = {af::ops::Zero, af::ops::Zero};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.attr.api.compute_type = af::ComputeType::kComputeStore;
  store_op1.x = argmax.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_INT32;
  *store_op1.y.strides = {af::ops::Zero, af::ops::Zero};
  *store_op1.y.repeats = {af::ops::One, af::ops::One};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_INT32;
  output_op.ir_attr.SetIndex(0);
}

TEST_F(ArgMaxScheduleCaseGeneratorTest, TestArgMax_Three_Elewise_Store) {
  af::AscGraph graph("argmax_three_elewise_store");
  ConstructNormStruct3ElewiseArgMax(graph);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 1UL);
}

TEST_F(ArgMaxScheduleCaseGeneratorTest, TestArgMax_One_Elewise_Store) {
  af::AscGraph graph("argmax_one_elewise_store");
  ConstructNormStruct1ElewiseArgMax(graph);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  generator.GeneratorTask(graph, tasks, options);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 1UL);
}

TEST_F(ArgMaxScheduleCaseGeneratorTest, TestArgMax_Four_Elewise_Store) {
  af::AscGraph graph("argmax_four_elewise_store");
  ConstructNormStruct4ElewiseArgMax(graph);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  generator.GeneratorTask(graph, tasks, options);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
  ASSERT_EQ(tasks[1].grouped_graphs.size(), 1UL);
}

TEST_F(ArgMaxScheduleCaseGeneratorTest, TestArgMax_Multi_Cita_Store) {
  af::AscGraph graph("argmax_multi_citation_store");
  ConstructNormStructMultiplyCitationsArgMax(graph);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  generator.GeneratorTask(graph, tasks, options);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
}
}  // namespace schedule
