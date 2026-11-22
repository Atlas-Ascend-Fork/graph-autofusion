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
#include "common_utils.h"
#include "arg_reduce_utils.h"

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

  static void CheckWithValuePhase2(af::AscGraph &graph, ge::DataType value_dtype,
                                   const std::string &phase_type = "ArgMaxMultiRPhase2WithValue") {
    af::AscNodePtr reduce;
    for (const auto &node : graph.GetAllNodes()) {
      if (node->GetType() == phase_type) {
        ASSERT_EQ(reduce, nullptr);
        reduce = node;
      }
    }
    ASSERT_NE(reduce, nullptr);
    ASSERT_EQ(reduce->outputs().size(), 2UL);
    const std::vector<int64_t> axes{0, 1};
    ASSERT_NE(graph.FindAxis(axes[0]), nullptr);
    ASSERT_NE(graph.FindAxis(axes[1]), nullptr);
    EXPECT_EQ(graph.FindAxis(axes[0])->name, "Rm");
    EXPECT_EQ(graph.FindAxis(axes[1])->name, "A");
    for (size_t i = 0; i < 2UL; ++i) {
      const auto &attr = reduce->outputs[i].attr;
      EXPECT_EQ(attr.axis, axes);
      EXPECT_EQ(ExpressToStr(attr.repeats), ExpressToStr({One, graph.FindAxis(axes[1])->size}));
      EXPECT_EQ(ExpressToStr(attr.strides), ExpressToStr({Zero, One}));
      EXPECT_EQ(attr.dtype, i == 0UL ? value_dtype : ge::DT_INT64);
      auto store = graph.FindNode(i == 0UL ? "store_value" : "store_index");
      ASSERT_NE(store, nullptr);
      EXPECT_EQ(store->GetInDataAnchor(0)->GetPeerOutAnchor(), reduce->GetOutDataAnchor(i));
      EXPECT_EQ(store->inputs[0].attr.dtype, attr.dtype);
      auto output = graph.FindNode(i == 0UL ? "output" : "output_index");
      ASSERT_NE(output, nullptr);
      EXPECT_EQ(output->GetInDataAnchor(0)->GetPeerOutAnchor(), store->GetOutDataAnchor(0));
    }
  }
  static void CheckMinRCoreTask(ScheduleTask &task, ge::DataType value_dtype, bool with_value) {
    ASSERT_EQ(task.grouped_graphs.size(), 2UL);
    bool phase1 = false, phase2 = false;
    for (const auto &node : task.grouped_graphs[0].GetAllNodes()) {
      EXPECT_FALSE(ascgen_utils::IsArgMaxType(node->GetType()));
      if (node->GetType() == "ArgMinMultiRPhase1") {
        phase1 = true;
        ASSERT_EQ(node->outputs().size(), 2UL);
        EXPECT_EQ(node->outputs[0].attr.dtype, value_dtype);
        EXPECT_EQ(node->outputs[1].attr.dtype, ge::DT_INT32);
      }
    }
    for (const auto &node : task.grouped_graphs[1].GetAllNodes()) {
      EXPECT_FALSE(ascgen_utils::IsArgMaxType(node->GetType()));
      phase2 = phase2 || node->GetType() == (with_value ? "ArgMinMultiRPhase2WithValue" : "ArgMinMultiRPhase2");
    }
    EXPECT_TRUE(phase1);
    EXPECT_TRUE(phase2);
    for (size_t i = 0; i < 2UL; ++i) {
      const auto name = ascgen_utils::NormalizeTemplateSemanticName(ascgen_utils::RefineTemplateSemanticName(
          ascgen_utils::template_semantic::kReduceRCore, task.grouped_graphs[i]));
      EXPECT_NE(name.find(i == 0UL ? "@Phase1" : "@Phase2"), std::string::npos) << name;
    }
    if (with_value) {
      ASSERT_NO_FATAL_FAILURE(CheckWithValuePhase2(task.grouped_graphs[1], value_dtype, "ArgMinMultiRPhase2WithValue"));
    }
  }
};

template <typename Op = af::ascir_op::Argmax>
void ConstructNormStruct1ElewiseArgmax(af::AscGraph &graph, ge::DataType index_dtype = ge::DT_INT32) {
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
  Op argmax("argmax");
  argmax.attr.sched.axis = {z0.id, z1.id};
  argmax.attr.api.compute_type = af::ComputeType::kComputeReduce;
  argmax.x = load.y;
  *argmax.y.axis = {z0.id, z1.id};
  argmax.y.dtype = index_dtype;
  *argmax.y.repeats = {s0, af::ops::One};
  *argmax.y.strides = {af::ops::One, af::ops::Zero};

  Store store_op1("store1");
  store_op1.attr.sched.axis = {z0.id, z1.id};
  store_op1.attr.api.compute_type = af::ComputeType::kComputeStore;
  store_op1.x = argmax.y;
  *store_op1.y.axis = {z0.id, z1.id};
  store_op1.y.dtype = index_dtype;
  *store_op1.y.strides = {af::ops::One, af::ops::Zero};
  *store_op1.y.repeats = {s0, af::ops::One};

  Output output_op("output");
  output_op.x = store_op1.y;
  output_op.y.dtype = index_dtype;
  output_op.ir_attr.SetIndex(0);
}

static void AddArgmaxIndexOutput(Store &store_index) {
  Output output_index("output_index");
  output_index.x = store_index.y;
  output_index.y.dtype = ge::DT_INT64;
  output_index.ir_attr.SetIndex(1);
}

static void AddArgReduceValueOutput(Store &store, ge::DataType dtype) {
  Output output("output");
  output.x = store.y;
  output.y.dtype = dtype;
  output.ir_attr.SetIndex(0);
}

// Both frontend directions have the same value/index topology.
template <typename Op = af::ascir_op::ArgmaxWithValue>
void ConstructNormStruct1ElewiseArgmaxWithValue(af::AscGraph &graph, ge::DataType dtype = ge::DT_FLOAT) {
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s0);

  Data data("data", graph);
  data.y.dtype = dtype;
  data.ir_attr.SetIndex(0);

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = dtype;
  *load.y.strides = {s1, af::ops::One};
  *load.y.repeats = {s0, s1};

  // ArgmaxWithValue outputs value(FLOAT) + index(INT64)
  Op argmax_with_value("argmax_with_value");
  argmax_with_value.attr.sched.axis = {z0.id, z1.id};
  argmax_with_value.attr.api.compute_type = af::ComputeType::kComputeReduce;
  argmax_with_value.x = load.y;
  *argmax_with_value.value.axis = {z0.id, z1.id};
  argmax_with_value.value.dtype = dtype;
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
  store_value.y.dtype = dtype;
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

  AddArgReduceValueOutput(store_value, dtype);
  AddArgmaxIndexOutput(store_index);
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

  AddArgmaxIndexOutput(store_index);
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
  for (auto &task : tasks) {
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
      ASSERT_NO_FATAL_FAILURE(CheckWithValuePhase2(task.grouped_graphs[1], ge::DT_FLOAT));
    }
  }
  EXPECT_TRUE(has_rcore_task);
}

// v1 平台（2201）：Argmax/ArgmaxWithValue 均不进入 RCore 模板生成。
TEST_F(ArgmaxScheduleCaseGeneratorTest, TestArgmaxWithValue_V1Platform_KeepGate) {
  SetPlatform("2201");
  for (const auto value_dtype : {ge::DT_FLOAT, ge::DT_INT64}) {
    SCOPED_TRACE(static_cast<int>(value_dtype));
    af::AscGraph graph("argmax_with_value_v1_keep_gate");
    if (value_dtype == ge::DT_FLOAT) {
      ConstructNormStruct1ElewiseArgmaxWithValue(graph);
    } else {
      ConstructNormStruct1ElewiseArgmaxWithValueInt64(graph);
    }
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
  for (auto &task : tasks) {
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
      ASSERT_NO_FATAL_FAILURE(CheckWithValuePhase2(task.grouped_graphs[1], ge::DT_INT64));
    }
  }
  EXPECT_TRUE(has_rcore_task);
}

TEST_F(ArgmaxScheduleCaseGeneratorTest, TestArgmaxWithValueRA_V2Platform_RCorePartition) {
  for (const auto value_dtype : {ge::DT_FLOAT, ge::DT_INT64}) {
    SCOPED_TRACE(static_cast<int>(value_dtype));
    af::AscGraph graph("argmax_with_value_ra_v2_rcore");
    if (value_dtype == ge::DT_FLOAT) {
      ConstructNormStruct1ElewiseArgmaxWithValue(graph);
    } else {
      ConstructNormStruct1ElewiseArgmaxWithValueInt64(graph);
    }
    const auto a_size = graph.FindNode("load")->outputs[0].attr.repeats[1];
    for (const char *name : {"argmax_with_value", "store_value", "store_index"}) {
      auto node = graph.FindNode(name);
      ASSERT_NE(node, nullptr);
      for (auto output : node->outputs()) {
        output->attr.repeats = {One, a_size};
        output->attr.strides = {Zero, One};
      }
    }
    std::vector<ScheduleTask> tasks;
    optimize::ReducePartitionCaseGenerator generator;
    OptimizerOptions options;
    ASSERT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
    bool has_rcore_task = false;
    for (auto &task : tasks) {
      if (task.reduce_type == ReduceTemplateType::kRCore) {
        has_rcore_task = true;
        ASSERT_EQ(task.grouped_graphs.size(), 2UL);
        ASSERT_NO_FATAL_FAILURE(CheckWithValuePhase2(task.grouped_graphs[1], value_dtype));
      }
    }
    EXPECT_TRUE(has_rcore_task);
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

// Argmax/ArgmaxWithValue 均受 HasMultiRAxisReduce 约束：多 R 轴时跳过 RCore 模板。
TEST_F(ArgmaxScheduleCaseGeneratorTest, TestArgmax_MultiRAxis_SkipRCore) {
  for (const bool with_value : {false, true}) {
    SCOPED_TRACE(with_value ? "ArgmaxWithValue" : "Argmax");
    af::AscGraph graph(with_value ? "argmax_with_value_multi_r_axis" : "argmax_multi_r_axis");
    if (with_value) {
      ConstructNormStruct1ElewiseArgmaxWithValue(graph);
    } else {
      ConstructNormStruct1ElewiseArgmax(graph);
    }
    const char *reduce_name = with_value ? "argmax_with_value" : "argmax";
    auto reduce = graph.FindNode(reduce_name);
    ASSERT_NE(reduce, nullptr);
    for (auto output : reduce->outputs()) {
      output->attr.repeats = {af::ops::One, af::ops::One};
      output->attr.strides = {af::ops::Zero, af::ops::Zero};
    }
    const std::vector<const char *> store_names =
        with_value ? std::vector<const char *>{"store_value", "store_index"} : std::vector<const char *>{"store1"};
    for (const char *store_name : store_names) {
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
}
TEST_F(ArgmaxScheduleCaseGeneratorTest, ArgminRCorePreservesDirectionAndPhaseRoles) {
  af::AscGraph graph("argmin_rcore");
  ConstructNormStruct1ElewiseArgmax<af::ascir_op::Argmin>(graph, ge::DT_INT64);
  std::vector<ScheduleTask> tasks;
  ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  ASSERT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  bool found = false;
  for (auto &task : tasks) {
    if (task.reduce_type == ReduceTemplateType::kRCore) {
      found = true;
      ASSERT_NO_FATAL_FAILURE(CheckMinRCoreTask(task, ge::DT_FLOAT, false));
    }
  }
  EXPECT_TRUE(found);
}

TEST_F(ArgmaxScheduleCaseGeneratorTest, ArgminWithValueRCorePreservesAllDtypesBothLayoutsAndTwoOutputs) {
  for (const auto dtype : {ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64}) {
    for (const bool ra : {false, true}) {
      SCOPED_TRACE(static_cast<int>(dtype));
      SCOPED_TRACE(ra);
      af::AscGraph graph("argmin_value_rcore");
      ConstructNormStruct1ElewiseArgmaxWithValue<af::ascir_op::ArgminWithValue>(graph, dtype);
      if (ra) {
        const auto a_size = graph.FindNode("load")->outputs[0].attr.repeats[1];
        for (const char *name : {"argmax_with_value", "store_value", "store_index"}) {
          for (auto output : graph.FindNode(name)->outputs()) {
            output->attr.repeats = {One, a_size};
            output->attr.strides = {Zero, One};
          }
        }
      }
      std::vector<ScheduleTask> tasks;
      ReducePartitionCaseGenerator generator;
      OptimizerOptions options;
      ASSERT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
      bool found = false;
      for (auto &task : tasks) {
        if (task.reduce_type == ReduceTemplateType::kRCore) {
          found = true;
          ASSERT_NO_FATAL_FAILURE(CheckMinRCoreTask(task, dtype, true));
        }
      }
      EXPECT_TRUE(found);
    }
  }
}

TEST_F(ArgmaxScheduleCaseGeneratorTest, ArgminWithValueKeepsV1RCoreGate) {
  SetPlatform("2201");
  af::AscGraph graph("argmin_v1_gate");
  ConstructNormStruct1ElewiseArgmaxWithValue<af::ascir_op::ArgminWithValue>(graph, ge::DT_INT64);
  std::vector<ScheduleTask> tasks;
  ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  ASSERT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  for (const auto &task : tasks) {
    EXPECT_NE(task.reduce_type, ReduceTemplateType::kRCore);
  }
}

TEST_F(ArgmaxScheduleCaseGeneratorTest, ArgminMultiRAxisKeepsExistingAllLoadFallback) {
  af::AscGraph graph("argmin_multi_r");
  ConstructNormStruct1ElewiseArgmaxWithValue<af::ascir_op::ArgminWithValue>(graph, ge::DT_INT64);
  for (const char *name : {"argmax_with_value", "store_value", "store_index"}) {
    for (auto output : graph.FindNode(name)->outputs()) {
      output->attr.repeats = {One, One};
      output->attr.strides = {Zero, Zero};
    }
  }
  std::vector<ScheduleTask> tasks;
  ReducePartitionCaseGenerator generator;
  OptimizerOptions options;
  ASSERT_EQ(generator.GeneratorTask(graph, tasks, options), af::SUCCESS);
  ASSERT_EQ(tasks.size(), 2UL);
  for (const auto &task : tasks) {
    EXPECT_NE(task.reduce_type, ReduceTemplateType::kRCore);
  }
}
}  // namespace schedule
