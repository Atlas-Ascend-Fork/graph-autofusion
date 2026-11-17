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
#include "common/platform_context.h"

namespace schedule {
using namespace optimize;
using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;

class ArgmaxScheduleCaseGeneratorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    // Argmax 的 R 轴分核（Phase1/Phase2）依赖 v2 平台注册与 Reg 指令实现；
    // 本测试默认在 v2 平台（5102）下验证门控放开。
    SetPlatform("5102");
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::PlatformContext::GetInstance().Reset();
  }

  static void SetPlatform(const std::string &soc_version) {
    ge::PlatformContext::GetInstance().Reset();
    ge::PlatformInfo platform_info;
    platform_info.soc_ver = soc_version;
    platform_info.aiv_num = 48;
    platform_info.ub_size = 245760;
    ASSERT_EQ(ge::PlatformContext::GetInstance().SetPlatformTarget(platform_info, soc_version), af::SUCCESS);
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

void ConstructNormStruct1ElewiseArgmax(af::AscGraph &graph) {
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

  // Argmax outputs int32 indices
  af::ascir_op::Argmax argmax("argmax");
  argmax.attr.sched.axis = {z0.id, z1.id};
  argmax.attr.api.compute_type = af::ComputeType::kComputeReduce;
  argmax.x = load.y;
  *argmax.y.axis = {z0.id, z1.id};
  argmax.y.dtype = ge::DT_INT32;
  *argmax.y.repeats = {s0, af::ops::One};
  *argmax.y.strides = {af::ops::One, af::ops::Zero};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.attr.api.compute_type = af::ComputeType::kComputeStore;
  store_op1.x = argmax.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = ge::DT_INT32;
  *store_op1.y.strides = {af::ops::One, af::ops::Zero};
  *store_op1.y.repeats = {s0, af::ops::One};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = ge::DT_INT32;
  output_op.ir_attr.SetIndex(0);
}

// 前端算子 ArgmaxWithValue（1 入 2 出）：x(FLOAT) -> value(FLOAT) + index(INT64)
void ConstructNormStruct1ElewiseArgmaxWithValue(af::AscGraph &graph) {
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

  // ArgmaxWithValue outputs value(FLOAT) + index(INT64)
  af::ascir_op::ArgmaxWithValue argmax_with_value("argmax_with_value");
  argmax_with_value.attr.sched.axis = {z0.id, z1.id};
  argmax_with_value.attr.api.compute_type = af::ComputeType::kComputeReduce;
  argmax_with_value.x = load.y;
  *argmax_with_value.value.axis = {z0.id, z1.id};
  argmax_with_value.value.dtype = ge::DT_FLOAT;
  *argmax_with_value.value.repeats = {s0, af::ops::One};
  *argmax_with_value.value.strides = {af::ops::One, af::ops::Zero};
  *argmax_with_value.index.axis = {z0.id, z1.id};
  argmax_with_value.index.dtype = ge::DT_INT64;
  *argmax_with_value.index.repeats = {s0, af::ops::One};
  *argmax_with_value.index.strides = {af::ops::One, af::ops::Zero};

  Store store_value("store_value");
  store_value.attr.sched.axis = {z0.id, z1.id};
  store_value.attr.api.compute_type = af::ComputeType::kComputeStore;
  store_value.x = argmax_with_value.value;
  *store_value.y.axis = {z0.id, z1.id};
  store_value.y.dtype = ge::DT_FLOAT;
  *store_value.y.strides = {af::ops::One, af::ops::Zero};
  *store_value.y.repeats = {s0, af::ops::One};

  Store store_index("store_index");
  store_index.attr.sched.axis = {z0.id, z1.id};
  store_index.attr.api.compute_type = af::ComputeType::kComputeStore;
  store_index.x = argmax_with_value.index;
  *store_index.y.axis = {z0.id, z1.id};
  store_index.y.dtype = ge::DT_INT64;
  *store_index.y.strides = {af::ops::One, af::ops::Zero};
  *store_index.y.repeats = {s0, af::ops::One};

  Output output_op("output");
  output_op.x = store_value.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

// FR13：ArgmaxWithValue int64 输入（x(INT64) -> value(INT64) + index(INT64)）
void ConstructNormStruct1ElewiseArgmaxWithValueInt64(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s0);

  Data data("data", graph);
  data.y.dtype = ge::DT_INT64;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_INT64;
  *load.y.strides = {s1, af::ops::One};
  *load.y.repeats = {s0, s1};

  // ArgmaxWithValue outputs value(INT64) + index(INT64)
  af::ascir_op::ArgmaxWithValue argmax_with_value("argmax_with_value");
  argmax_with_value.attr.sched.axis = {z0.id, z1.id};
  argmax_with_value.attr.api.compute_type = af::ComputeType::kComputeReduce;
  argmax_with_value.x = load.y;
  *argmax_with_value.value.axis = {z0.id, z1.id};
  argmax_with_value.value.dtype = ge::DT_INT64;
  *argmax_with_value.value.repeats = {s0, af::ops::One};
  *argmax_with_value.value.strides = {af::ops::One, af::ops::Zero};
  *argmax_with_value.index.axis = {z0.id, z1.id};
  argmax_with_value.index.dtype = ge::DT_INT64;
  *argmax_with_value.index.repeats = {s0, af::ops::One};
  *argmax_with_value.index.strides = {af::ops::One, af::ops::Zero};

  Store store_value("store_value");
  store_value.attr.sched.axis = {z0.id, z1.id};
  store_value.attr.api.compute_type = af::ComputeType::kComputeStore;
  store_value.x = argmax_with_value.value;
  *store_value.y.axis = {z0.id, z1.id};
  store_value.y.dtype = ge::DT_INT64;
  *store_value.y.strides = {af::ops::One, af::ops::Zero};
  *store_value.y.repeats = {s0, af::ops::One};

  Store store_index("store_index");
  store_index.attr.sched.axis = {z0.id, z1.id};
  store_index.attr.api.compute_type = af::ComputeType::kComputeStore;
  store_index.x = argmax_with_value.index;
  *store_index.y.axis = {z0.id, z1.id};
  store_index.y.dtype = ge::DT_INT64;
  *store_index.y.strides = {af::ops::One, af::ops::Zero};
  *store_index.y.repeats = {s0, af::ops::One};

  Output output_op("output");
  output_op.x = store_value.y;
  output_op.y.dtype = ge::DT_INT64;
  output_op.ir_attr.SetIndex(0);
}

TEST_F(ArgmaxScheduleCaseGeneratorTest, TestArgmax_One_Elewise_Store) {
  af::AscGraph graph("argmax_one_elewise_store");
  ConstructNormStruct1ElewiseArgmax(graph);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  // 门控已剔除：含 Argmax 的 common task 进入 RCore 模板生成，
  // 产出 phase1/phase2 两阶段分核任务（kRCore，grouped_graphs=2，map 串联）；
  // 全载模板（AllLoad）作为独立候选追加，故 tasks 总量为 3。
  ASSERT_EQ(tasks.size(), 3UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
  bool has_rcore_task = false;
  for (const auto &task : tasks) {
    if (task.reduce_type == ReduceTemplateType::kRCore) {
      has_rcore_task = true;
      ASSERT_EQ(task.grouped_graphs.size(), 2UL);
      ASSERT_EQ(task.groups_relations_in.size(), 1UL);
      ASSERT_EQ(task.groups_relations_in.begin()->second.size(), 1UL);
      // Phase1 必须使用 ArgMaxMultiRPhase1（codegen 走 RegArgMaxPhase1ApiCall，index 全局化），
      // 不能保留前端 ArgmaxWithValue（其 codegen 默认 indexBase=0，只写局部下标）。
      bool has_phase1_op = false;
      bool has_front_argmax_with_value = false;
      for (const auto &node : task.grouped_graphs[0].GetAllNodes()) {
        has_phase1_op = has_phase1_op || node->GetType() == "ArgMaxMultiRPhase1";
        has_front_argmax_with_value = has_front_argmax_with_value || node->GetType() == "ArgmaxWithValue";
      }
      EXPECT_TRUE(has_phase1_op) << "Phase1 graph should contain ArgMaxMultiRPhase1";
      EXPECT_FALSE(has_front_argmax_with_value) << "Phase1 graph should not keep front-end ArgmaxWithValue";
    }
  }
  EXPECT_TRUE(has_rcore_task);
}

// v2 平台：含 ArgmaxWithValue 的 common task 进入 RCore 模板生成，
// Phase2 用 ArgMaxMultiRPhase2WithValue 一次归约输出全局 value+index，
// 产出 phase1/phase2 两阶段分核任务（kRCore，grouped_graphs=2，map 串联）。
TEST_F(ArgmaxScheduleCaseGeneratorTest, TestArgmaxWithValue_V2Platform_RCorePartition) {
  af::AscGraph graph("argmax_with_value_v2_rcore");
  ConstructNormStruct1ElewiseArgmaxWithValue(graph);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  ASSERT_EQ(tasks.size(), 3UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
  bool has_rcore_task = false;
  for (const auto &task : tasks) {
    if (task.reduce_type == ReduceTemplateType::kRCore) {
      has_rcore_task = true;
      ASSERT_EQ(task.grouped_graphs.size(), 2UL);
      ASSERT_EQ(task.groups_relations_in.size(), 1UL);
      ASSERT_EQ(task.groups_relations_in.begin()->second.size(), 1UL);
      // Phase1 必须使用 ArgMaxMultiRPhase1（codegen 走 RegArgMaxPhase1ApiCall，index 全局化），
      // 不能保留前端 ArgmaxWithValue（其 codegen 默认 indexBase=0，只写局部下标）。
      bool has_phase1_op = false;
      bool has_front_argmax_with_value = false;
      for (const auto &node : task.grouped_graphs[0].GetAllNodes()) {
        has_phase1_op = has_phase1_op || node->GetType() == "ArgMaxMultiRPhase1";
        has_front_argmax_with_value = has_front_argmax_with_value || node->GetType() == "ArgmaxWithValue";
      }
      EXPECT_TRUE(has_phase1_op) << "Phase1 graph should contain ArgMaxMultiRPhase1";
      EXPECT_FALSE(has_front_argmax_with_value) << "Phase1 graph should not keep front-end ArgmaxWithValue";
    }
  }
  EXPECT_TRUE(has_rcore_task);
}

// v1 平台（2201）：HasArgmaxReduce 匹配集合扩展后，ArgmaxWithValue 与 Argmax 一致被拦截，
// 不进入 RCore 模板生成；common 任务之外全载模板（AllLoad）作为独立候选追加，tasks 总量为 2。
TEST_F(ArgmaxScheduleCaseGeneratorTest, TestArgmaxWithValue_V1Platform_KeepGate) {
  SetPlatform("2201");
  af::AscGraph graph("argmax_with_value_v1_keep_gate");
  ConstructNormStruct1ElewiseArgmaxWithValue(graph);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
  for (const auto &task : tasks) {
    EXPECT_NE(task.reduce_type, ReduceTemplateType::kRCore);
  }
}

// FR13：int64 输入的 ArgmaxWithValue 在 v2 平台进入 RCore 分核（Phase1/Phase2 value 通路 int64）
TEST_F(ArgmaxScheduleCaseGeneratorTest, TestArgmaxWithValueInt64_V2Platform_RCorePartition) {
  af::AscGraph graph("argmax_with_value_int64_v2_rcore");
  ConstructNormStruct1ElewiseArgmaxWithValueInt64(graph);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  ASSERT_EQ(tasks.size(), 3UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
  bool has_rcore_task = false;
  for (const auto &task : tasks) {
    if (task.reduce_type == ReduceTemplateType::kRCore) {
      has_rcore_task = true;
      ASSERT_EQ(task.grouped_graphs.size(), 2UL);
      ASSERT_EQ(task.groups_relations_in.size(), 1UL);
      ASSERT_EQ(task.groups_relations_in.begin()->second.size(), 1UL);
      // Phase1 必须使用 ArgMaxMultiRPhase1（codegen 走 RegArgMaxPhase1ApiCall，index 全局化），
      // 不能保留前端 ArgmaxWithValue（其 codegen 默认 indexBase=0，只写局部下标）。
      bool has_phase1_op = false;
      bool has_front_argmax_with_value = false;
      for (const auto &node : task.grouped_graphs[0].GetAllNodes()) {
        has_phase1_op = has_phase1_op || node->GetType() == "ArgMaxMultiRPhase1";
        has_front_argmax_with_value = has_front_argmax_with_value || node->GetType() == "ArgmaxWithValue";
      }
      EXPECT_TRUE(has_phase1_op) << "Phase1 graph should contain ArgMaxMultiRPhase1";
      EXPECT_FALSE(has_front_argmax_with_value) << "Phase1 graph should not keep front-end ArgmaxWithValue";
    }
  }
  EXPECT_TRUE(has_rcore_task);
}

// FR13：int64 输入的 ArgmaxWithValue 在 v1 平台与 float 版一致被拦截
TEST_F(ArgmaxScheduleCaseGeneratorTest, TestArgmaxWithValueInt64_V1Platform_KeepGate) {
  SetPlatform("2201");
  af::AscGraph graph("argmax_with_value_int64_v1_keep_gate");
  ConstructNormStruct1ElewiseArgmaxWithValueInt64(graph);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
  for (const auto &task : tasks) {
    EXPECT_NE(task.reduce_type, ReduceTemplateType::kRCore);
  }
}

// v1 平台（2201）继续保留 HasArgmaxReduce 门控拦截：
// 含 Argmax 的图不进入 RCore 模板生成；common 任务之外全载模板（AllLoad）作为独立候选追加，tasks 总量为 2。
TEST_F(ArgmaxScheduleCaseGeneratorTest, TestArgmax_V1Platform_KeepGate) {
  SetPlatform("2201");
  af::AscGraph graph("argmax_v1_keep_gate");
  ConstructNormStruct1ElewiseArgmax(graph);
  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  ASSERT_EQ(tasks.size(), 2UL);
  ASSERT_EQ(tasks[0].grouped_graphs.size(), 1UL);
  for (const auto &task : tasks) {
    EXPECT_NE(task.reduce_type, ReduceTemplateType::kRCore);
  }
}

// Argmax 同样受 HasMultiRAxisReduce 约束：多 R 轴（z0/z1 均归约）时跳过 RCore 模板，
// 仅保留 Common + AllLoad，tasks 总量为 2。
TEST_F(ArgmaxScheduleCaseGeneratorTest, TestArgmax_MultiRAxis_SkipRCore) {
  af::AscGraph graph("argmax_multi_r_axis");
  ConstructNormStruct1ElewiseArgmax(graph);
  auto argmax = graph.FindNode("argmax");
  ASSERT_NE(argmax, nullptr);
  argmax->outputs[0].attr.repeats = {af::ops::One, af::ops::One};
  argmax->outputs[0].attr.strides = {af::ops::Zero, af::ops::Zero};
  auto store = graph.FindNode("store1");
  ASSERT_NE(store, nullptr);
  store->outputs[0].attr.repeats = {af::ops::One, af::ops::One};
  store->outputs[0].attr.strides = {af::ops::Zero, af::ops::Zero};

  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  ASSERT_EQ(tasks.size(), 2UL);
  for (const auto &task : tasks) {
    EXPECT_NE(task.reduce_type, ReduceTemplateType::kRCore);
  }
}

// ArgmaxWithValue 同样受 HasMultiRAxisReduce 约束：多 R 轴时跳过 RCore 模板。
TEST_F(ArgmaxScheduleCaseGeneratorTest, TestArgmaxWithValue_MultiRAxis_SkipRCore) {
  af::AscGraph graph("argmax_with_value_multi_r_axis");
  ConstructNormStruct1ElewiseArgmaxWithValue(graph);
  auto argmax = graph.FindNode("argmax_with_value");
  ASSERT_NE(argmax, nullptr);
  argmax->outputs[0].attr.repeats = {af::ops::One, af::ops::One};
  argmax->outputs[0].attr.strides = {af::ops::Zero, af::ops::Zero};
  argmax->outputs[1].attr.repeats = {af::ops::One, af::ops::One};
  argmax->outputs[1].attr.strides = {af::ops::Zero, af::ops::Zero};
  for (const char *store_name : {"store_value", "store_index"}) {
    auto store = graph.FindNode(store_name);
    ASSERT_NE(store, nullptr);
    store->outputs[0].attr.repeats = {af::ops::One, af::ops::One};
    store->outputs[0].attr.strides = {af::ops::Zero, af::ops::Zero};
  }

  std::vector<ScheduleTask> tasks;
  optimize::ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  EXPECT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  ASSERT_EQ(tasks.size(), 2UL);
  for (const auto &task : tasks) {
    EXPECT_NE(task.reduce_type, ReduceTemplateType::kRCore);
  }
}
}  // namespace schedule
