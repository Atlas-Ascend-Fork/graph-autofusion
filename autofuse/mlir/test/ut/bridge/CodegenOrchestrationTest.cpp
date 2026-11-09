/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bridge/CodegenOrchestration/CodegenOrchestration.h"
#include "bridge/ScheduleToAfir/ScheduleToAfir.h"
#include "optimize/optimize.h"
#include "common/autofuse_config/auto_fuse_config.h"
#include "llvm/ADT/ScopeExit.h"
#include "test/Support/ReduceGraphBuilder.h"

#include <cctype>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <unistd.h>
#include <vector>

#include "ascgen_log.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "ascir_utils.h"
#include "ascendc_api_registry.h"
#include "codegen.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir_def.h"
#include "test/ut/schedule/ScheduleTestUtils.h"
#include "test/Support/TestEnvironment.h"
#include "depends/runtime/src/runtime_stub.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "support/CapabilityShortfall.h"

#include "llvm/ADT/STLExtras.h"

#include <gtest/gtest.h>

namespace {

void SetupLoadAttrs(af::AscNode &load, uint64_t z0_id, const af::Expression &z0_size) {
  auto &attr = load.outputs[0].attr;
  attr.axis = {static_cast<int64_t>(z0_id)};
  attr.vectorized_axis = {static_cast<int64_t>(z0_id)};
  attr.vectorized_strides = {af::ops::One};
  attr.repeats = {z0_size};
  attr.strides = {af::ops::One};
  attr.mem.position = af::Position::kPositionVecIn;
  attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  attr.mem.tensor_id = 1;
  attr.que.id = 0;
  attr.mem.reuse_id = 0;
  attr.que.depth = 2;
  attr.que.buf_num = 2;
  attr.opt.merge_scope = af::kIdNone;
}

void SetupStoreAttrs(af::AscNode &store, uint64_t z0_id, const af::Expression &z0_size) {
  auto &attr = store.outputs[0].attr;
  attr.mem.position = af::Position::kPositionGM;
  attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  attr.mem.tensor_id = 2;
  attr.axis = {static_cast<int64_t>(z0_id)};
  attr.vectorized_axis = {static_cast<int64_t>(z0_id)};
  attr.vectorized_strides = {af::ops::One};
  attr.repeats = {z0_size};
  attr.strides = {af::ops::One};
}

void SetupAddAttrs(af::AscNode &add, uint64_t z0_id, const af::Expression &z0_size) {
  auto &attr = add.outputs[0].attr;
  attr.dtype = ge::DT_FLOAT16;
  attr.axis = {static_cast<int64_t>(z0_id)};
  attr.vectorized_axis = {static_cast<int64_t>(z0_id)};
  attr.vectorized_strides = {af::ops::One};
  attr.repeats = {z0_size};
  attr.strides = {af::ops::One};
  attr.mem.position = af::Position::kPositionVecOut;
  attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  attr.mem.tensor_id = 4;
  attr.mem.reuse_id = 0;
  attr.que.id = 1;
  attr.que.depth = 2;
  attr.que.buf_num = 2;
  add.attr.sched.axis = {static_cast<int64_t>(z0_id)};
  add.attr.sched.loop_axis = z0_id;
}

// Minimal load -> add -> store -> output fused schedule result that survives
// the full addPlanCandidateLoweringPipeline (AFIRLoopGenerate, memory allocate,
// implementation selection, AscendC conversion). Ported from
// test_codegen_tiling.cpp's GenSingleDynamicPlanFusedScheduleResult, which is
// the only fixture in the codebase already proven to clear that pipeline;
// see that file's comments for why each of these fields is required (e.g.
// AFIRLoopGenerate rejects a templated node unless afir.sched carries a
// loop_axis, which ScheduleToAfir only emits when node->attr.sched.loop_axis
// differs from its unset default).
// `break_binding` 把这张图改成**永久**不可下降：`mem.tensor_id` 为负会被
// `ValidateScheduleBinding` 当场判 `InvalidBinding`。故意不拿「某个 op 没有 Ascend 实现」
// 之类的能力缺口做锚点——缺口总会被补上，那天这条用例就静默失去意义了。
//
// 每次调用都新建一张图，不是拷一份再改：`af::AscGraph` 是共享句柄（现有 fixture 把局部
// `graph` 里的节点指针存进 fused result 之后还能用，就是这个缘故），拷贝改一份会把原图
// 一起改坏。
af::AscGraph BuildScheduledTestGraph(bool break_binding, bool static_shape = false) {
  af::AscGraph graph("test_graph");
  graph.SetTilingKey(10);
  if (!static_shape) {
    graph.CreateSizeVar("s0");
  }
  auto z0 = graph.CreateAxis("z0", af::ops::One);

  af::ascir_op::Data x_op("x", graph);
  x_op.ir_attr.SetIndex(0);
  af::ascir_op::Load load_op("load");
  af::ascir_op::Add add_op("add");
  af::ascir_op::Store store_op("store");
  af::ascir_op::Output y_op("y");
  y_op.ir_attr.SetIndex(0);
  graph.AddNode(load_op);
  graph.AddNode(add_op);
  graph.AddNode(store_op);
  graph.AddNode(y_op);

  load_op.x = x_op.y;
  load_op.y.dtype = ge::DT_FLOAT16;
  add_op.x1 = load_op.y;
  add_op.x2 = load_op.y;
  add_op.y.dtype = ge::DT_FLOAT16;
  store_op.x = add_op.y;
  y_op.x = store_op.y;

  auto x = graph.FindNode("x");
  auto load = graph.FindNode("load");
  auto add = graph.FindNode("add");
  auto store = graph.FindNode("store");
  auto y = graph.FindNode("y");

  x->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load->outputs[0].attr.dtype = ge::DT_FLOAT16;
  add->outputs[0].attr.dtype = ge::DT_FLOAT16;
  store->outputs[0].attr.dtype = ge::DT_FLOAT16;
  y->outputs[0].attr.dtype = ge::DT_FLOAT16;
  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.tensor_id = 0;
  x->attr.api.unit = af::ComputeUnit::kUnitNone;
  y->attr.api.unit = af::ComputeUnit::kUnitNone;
  x->outputs[0].attr.axis = {static_cast<int64_t>(z0.id)};
  x->outputs[0].attr.vectorized_axis = {static_cast<int64_t>(z0.id)};
  x->outputs[0].attr.vectorized_strides = {af::ops::One};
  x->outputs[0].attr.repeats = {z0.size};
  x->outputs[0].attr.strides = {af::ops::One};
  y->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  y->outputs[0].attr.mem.tensor_id = 3;
  y->outputs[0].attr.axis = {static_cast<int64_t>(z0.id)};
  y->outputs[0].attr.vectorized_axis = {static_cast<int64_t>(z0.id)};
  y->outputs[0].attr.vectorized_strides = {af::ops::One};
  y->outputs[0].attr.repeats = {z0.size};
  y->outputs[0].attr.strides = {af::ops::One};

  SetupLoadAttrs(*load, z0.id, z0.size);
  SetupAddAttrs(*add, z0.id, z0.size);
  SetupStoreAttrs(*store, z0.id, z0.size);
  load->attr.sched.axis = {static_cast<int64_t>(z0.id)};
  load->attr.sched.loop_axis = z0.id;
  store->attr.sched.axis = {static_cast<int64_t>(z0.id)};
  store->attr.sched.loop_axis = z0.id;

  if (break_binding) {
    add->outputs[0].attr.mem.tensor_id = -1;
  }
  return graph;
}

ascir::FusedScheduledResult BuildValidFusedScheduleResult(bool static_shape = false) {
  af::AscGraph graph = BuildScheduledTestGraph(false, static_shape);
  auto x = graph.FindNode("x");
  auto y = graph.FindNode("y");

  ::ascir::ScheduledResult schedule_result;
  schedule_result.schedule_groups.resize(1);
  for (auto &schedule_group : schedule_result.schedule_groups) {
    schedule_group.impl_graphs.emplace_back(graph);
  }
  std::vector<ascir::ScheduledResult> schedule_results;
  schedule_results.push_back(schedule_result);

  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = af::AscendString(graph.GetName().c_str());
  fused_schedule_result.input_nodes.push_back(x);
  fused_schedule_result.output_nodes.push_back(y);
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  fused_schedule_result.origin_vars = {static_shape ? af::ops::One : af::Symbol("s0")};
  return fused_schedule_result;
}

// 同一个子图两个计划：plan 0 不可下降、plan 1 正常。不可下降的那个放在**前面**，这样
// 「丢弃 + 重编号」必须真的发生：幸存者会从 plan 1 重排成 result0。
ascir::FusedScheduledResult BuildFusedScheduleResultWithUnlowerableFirstPlan() {
  af::AscGraph broken_graph = BuildScheduledTestGraph(true);
  ascir::FusedScheduledResult fused_schedule_result = BuildValidFusedScheduleResult();
  ::ascir::ScheduledResult broken_plan;
  broken_plan.schedule_groups.resize(1);
  broken_plan.schedule_groups[0].impl_graphs.emplace_back(broken_graph);
  auto &plans = fused_schedule_result.node_idx_to_scheduled_results[0];
  plans.insert(plans.begin(), std::move(broken_plan));
  return fused_schedule_result;
}

// 同一个子图两个都能下降的计划，支持静态和动态 shape。
ascir::FusedScheduledResult BuildFusedScheduleResultWithTwoValidPlans(bool static_shape = false) {
  af::AscGraph second_graph = BuildScheduledTestGraph(false, static_shape);
  ascir::FusedScheduledResult fused_schedule_result = BuildValidFusedScheduleResult(static_shape);
  ::ascir::ScheduledResult second_plan;
  second_plan.schedule_groups.resize(1);
  second_plan.schedule_groups[0].impl_graphs.emplace_back(second_graph);
  fused_schedule_result.node_idx_to_scheduled_results[0].push_back(std::move(second_plan));
  return fused_schedule_result;
}

// 只有一个计划、且它不可下降。
ascir::FusedScheduledResult BuildFusedScheduleResultWithOnlyUnlowerablePlan() {
  af::AscGraph broken_graph = BuildScheduledTestGraph(true);
  ascir::FusedScheduledResult fused_schedule_result = BuildValidFusedScheduleResult();
  auto &plans = fused_schedule_result.node_idx_to_scheduled_results[0];
  plans.clear();
  ::ascir::ScheduledResult broken_plan;
  broken_plan.schedule_groups.resize(1);
  broken_plan.schedule_groups[0].impl_graphs.emplace_back(broken_graph);
  plans.push_back(std::move(broken_plan));
  return fused_schedule_result;
}

}  // namespace

TEST(CodegenOrchestrationTest, ReadsConfigAndSharedDfxDumpRoot) {
  const std::string dump_dir = "/tmp/autofuse_mlir_options_" + std::to_string(static_cast<long long>(getpid()));
  std::filesystem::remove_all(dump_dir);
  auto scoped_dfx_env = mlir::test::ScopedEnvVar::Unset("AUTOFUSE_DFX_FLAGS");
  ::ascir::utils::ResetDumpConfig();

  auto options = mlir::bridge::ReadMlirCodegenOptions();
  EXPECT_EQ(options.config, mlir::bridge::MlirCodegenConfig::kNone);
  EXPECT_TRUE(options.dump_root.empty());

  {
    mlir::test::ScopedEnvVar dfx_flags(
        "AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=mlir;--codegen_compile_debug=true;--debug_dir=" + dump_dir);
    ::ascir::utils::ResetDumpConfig();
    options = mlir::bridge::ReadMlirCodegenOptions();
    EXPECT_EQ(options.config, mlir::bridge::MlirCodegenConfig::kMlir);
    EXPECT_EQ(options.dump_root.rfind(dump_dir, 0), 0U);
  }

  {
    mlir::test::ScopedEnvVar dfx_flags("AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=invalid;");
    EXPECT_EQ(mlir::bridge::ReadMlirCodegenOptions().config, mlir::bridge::MlirCodegenConfig::kInvalid);
  }

  std::filesystem::remove_all(dump_dir);
  ::ascir::utils::ResetDumpConfig();
}

TEST(CodegenOrchestrationTest, ReadsConfigValues) {
  auto unset = mlir::test::ScopedEnvVar::Unset("AUTOFUSE_DFX_FLAGS");
  EXPECT_EQ(mlir::bridge::ReadMlirCodegenOptions().config, mlir::bridge::MlirCodegenConfig::kNone);

  const std::map<std::string, mlir::bridge::MlirCodegenConfig> values = {
      {"none", mlir::bridge::MlirCodegenConfig::kNone},
      {"mlir", mlir::bridge::MlirCodegenConfig::kMlir},
      {"fallback", mlir::bridge::MlirCodegenConfig::kFallback},
      {"invalid", mlir::bridge::MlirCodegenConfig::kInvalid},
  };
  for (const auto &[value, expected] : values) {
    SCOPED_TRACE(value);
    mlir::test::ScopedEnvVar flag("AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=" + value + ";");
    EXPECT_EQ(mlir::bridge::ReadMlirCodegenOptions().config, expected);
  }
}

TEST(CodegenOrchestrationTest, InvalidConfigFails) {
  mlir::test::ScopedEnvVar flag("AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=invalid;");
  auto fused_schedule_result = mlir::test::MakeFusedResult();
  codegen::CodegenResult result;
  EXPECT_EQ(mlir::bridge::RunMlirCodegen(fused_schedule_result, mlir::bridge::MlirCallerScene::kInductor, "Ascend910B1",
                                         result),
            mlir::bridge::MlirCodegenStatus::kFailed);
  EXPECT_TRUE(result.kernel.empty());
  EXPECT_TRUE(result.tiling.empty());
  EXPECT_TRUE(result.tiling_data.empty());
}

TEST(CodegenOrchestrationTest, ConfigPolicyCoversCubeAndCapabilityShortfall) {
  for (auto scene : {mlir::bridge::MlirCallerScene::kInductor, mlir::bridge::MlirCallerScene::kTensorFlow}) {
    for (bool cube : {false, true}) {
      auto scheduled = BuildValidFusedScheduleResult();
      if (cube) {
        scheduled.node_idx_to_scheduled_results[0][0].cube_type = ascir::CubeTemplateType::kFixpip;
      } else {
        auto &graph = scheduled.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
        auto add = graph.FindNode("add");
        ASSERT_NE(add, nullptr);
        add->attr.sched.exec_condition = af::ExecuteCondition::kConditionInvalid;
      }
      for (const char *value : {"fallback", "mlir"}) {
        SCOPED_TRACE(value);
        mlir::test::ScopedEnvVar flag("AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=" + std::string(value) + ";");
        codegen::CodegenResult result;
        const auto status = mlir::bridge::RunMlirCodegen(scheduled, scene, "Ascend910B1", result);
        const auto expected = cube ? mlir::bridge::MlirCodegenStatus::kNotSupported
                                   : (std::string(value) == "fallback" ? mlir::bridge::MlirCodegenStatus::kNotSupported
                                                                       : mlir::bridge::MlirCodegenStatus::kFailed);
        EXPECT_EQ(status, expected);
        EXPECT_TRUE(result.kernel.empty());
        EXPECT_TRUE(result.tiling.empty());
        EXPECT_TRUE(result.tiling_data.empty());
      }
    }
  }
}

TEST(CodegenOrchestrationTest, ConfigUnsetReturnsNotSupported) {
  auto scoped_dfx_env = mlir::test::ScopedEnvVar::Unset("AUTOFUSE_DFX_FLAGS");
  auto fused_schedule_result = mlir::test::MakeFusedResult();
  codegen::CodegenResult result;
  auto status = mlir::bridge::RunMlirCodegen(fused_schedule_result, mlir::bridge::MlirCallerScene::kInductor,
                                             "Ascend910B1", result);
  EXPECT_EQ(status, mlir::bridge::MlirCodegenStatus::kNotSupported);
  EXPECT_TRUE(result.kernel.empty());
  EXPECT_TRUE(result.tiling.empty());
  EXPECT_TRUE(result.tiling_data.empty());
}

TEST(CodegenOrchestrationTest, CubeFusedFailsInMlirConfig) {
  mlir::test::ScopedEnvVar scoped_mlir_env("AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=mlir;");
  auto fused_schedule_result = mlir::test::MakeFusedResult();
  fused_schedule_result.node_idx_to_scheduled_results[0][0].cube_type = ascir::CubeTemplateType::kFixpip;
  codegen::CodegenResult result;
  auto status = mlir::bridge::RunMlirCodegen(fused_schedule_result, mlir::bridge::MlirCallerScene::kInductor,
                                             "Ascend910B1", result);
  EXPECT_EQ(status, mlir::bridge::MlirCodegenStatus::kNotSupported);
}

TEST(CodegenOrchestrationTest, ValidFixtureSucceeds) {
  mlir::test::ScopedEnvVar denied("AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=mlir;");
  dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_DEBUG, 0);
  mlir::test::ScopedEnvVar scoped_mlir_env("AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=mlir;");
  auto fused_schedule_result = BuildValidFusedScheduleResult();
  codegen::CodegenResult result;
  auto status = mlir::bridge::RunMlirCodegen(fused_schedule_result, mlir::bridge::MlirCallerScene::kInductor,
                                             "Ascend910B1", result);
  ASSERT_EQ(status, mlir::bridge::MlirCodegenStatus::kSucceeded);
  EXPECT_FALSE(result.kernel.empty());
  EXPECT_NE(result.tiling.find("#include \"autofuse_tiling_data.h\""), std::string::npos);
  EXPECT_NE(result.tiling.find("extern \"C\" int64_t AutofuseTiling("), std::string::npos);
  EXPECT_NE(result.tiling.find("tiling->graph0_result0_g0_tiling_data.set_s0(s0);"), std::string::npos);
  EXPECT_FALSE(result.tiling_data.empty());
  EXPECT_NE(result.tiling_data.find("struct AutofuseTilingDataPerf"), std::string::npos);
  EXPECT_NE(result.tiling_data.find("inline void set_s0(uint32_t value) { "
                                    "graph0_result0_g0_tiling_data.set_s0(value); }"),
            std::string::npos);
}

TEST(CodegenOrchestrationTest, ValidFixtureDumpsEachLoweringPassResult) {
  const std::string dump_dir = "/tmp/autofuse_mlir_pass_dump_" + std::to_string(static_cast<long long>(getpid()));
  std::filesystem::remove_all(dump_dir);
  mlir::test::ScopedEnvVar scoped_dump_dir(
      "AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=mlir;--codegen_compile_debug=true;--debug_dir=" + dump_dir);
  ::ascir::utils::ResetDumpConfig();

  auto fused_schedule_result = BuildValidFusedScheduleResult();
  codegen::CodegenResult result;
  ASSERT_EQ(mlir::bridge::RunMlirCodegen(fused_schedule_result, mlir::bridge::MlirCallerScene::kInductor, "Ascend910B1",
                                         result),
            mlir::bridge::MlirCodegenStatus::kSucceeded);
  ASSERT_TRUE(std::filesystem::exists(dump_dir));
  bool found_mlir_dump_dir = false;
  for (const auto &entry : std::filesystem::recursive_directory_iterator(dump_dir)) {
    if (entry.is_directory() && entry.path().filename().string() == "mlir_subgraph_0_plan_0") {
      found_mlir_dump_dir = true;
      break;
    }
  }
  EXPECT_TRUE(found_mlir_dump_dir);

  const std::vector<std::string> expected_passes = {
      "afir-kernel-body-signature", "afir-loop-generate", "afir-memory-allocate", "ascend-select-implementation",
      "convert-afir-to-ascendc",    "gen-tiling-data",    "gen-kernel-entry"};
  for (const auto &pass : expected_passes) {
    bool found = false;
    for (const auto &entry : std::filesystem::recursive_directory_iterator(dump_dir)) {
      if (entry.is_regular_file() && entry.path().filename().string().find(pass) != std::string::npos) {
        const auto filename = entry.path().filename().string();
        ASSERT_GE(filename.size(), 3U);
        EXPECT_TRUE(std::isdigit(static_cast<unsigned char>(filename[0])) != 0);
        EXPECT_TRUE(std::isdigit(static_cast<unsigned char>(filename[1])) != 0);
        EXPECT_EQ(filename[2], '_');
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found) << "missing pass snapshot for " << pass;
  }
  std::filesystem::remove_all(dump_dir);
  ::ascir::utils::ResetDumpConfig();
}

// 「能力不足」与「坏了」的分流（`support/CapabilityShortfall.h`）。标记只走诊断的
// metadata，所以这一条先单独钉住机制本身：没标记的错误不算能力不足。
//
// 反过来写不了——`InFlightDiagnostic` 的报告动作在析构时发生，所以两条都必须真的发出去。
TEST(CodegenOrchestrationTest, OnlyMarkedDiagnosticsCountAsCapabilityShortfall) {
  mlir::MLIRContext context;
  {
    mlir::afir::CapabilityShortfallObserver observer(context);
    EXPECT_FALSE(observer.sawShortfall());
    mlir::emitError(mlir::UnknownLoc::get(&context)) << "an ordinary failure";
    EXPECT_FALSE(observer.sawShortfall());
    mlir::afir::emitCapabilityShortfall(mlir::UnknownLoc::get(&context)) << "a capability shortfall";
    EXPECT_TRUE(observer.sawShortfall());
  }
  // 观察者是作用域内的：出了作用域，下一次下降从干净的状态开始。
  mlir::afir::CapabilityShortfallObserver fresh(context);
  EXPECT_FALSE(fresh.sawShortfall());
}

// 严格 MLIR 配置下，能力不足也必须返回 `kFailed`——`kNotSupported` 会触发 legacy 回退，
// 测试就失去了证明 MLIR 路径行为的能力。
//
// 走的是 bridge 那条通路（上游给了一个这一层没建模的 `exec_condition` 取值）。能力层那条
// 通路（`has no supported Ascend implementation`）收束在同一个标记点上，形态由
// `test/lit/conversion/afir-transfer-multi-segment-unsupported.mlir` 钉。
//
// **先跑一遍没改的 fixture**：不先确认它本来成功，`kFailed` 可能来自别的原因——这条用例
// 就什么都没证明。
TEST(CodegenOrchestrationTest, CapabilityShortfallFailsInMlirConfig) {
  mlir::test::ScopedEnvVar scoped_mlir_env("AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=mlir;");
  auto fused_schedule_result = BuildFusedScheduleResultWithTwoValidPlans();
  codegen::CodegenResult baseline;
  ASSERT_EQ(mlir::bridge::RunMlirCodegen(fused_schedule_result, mlir::bridge::MlirCallerScene::kInductor, "Ascend910B1",
                                         baseline),
            mlir::bridge::MlirCodegenStatus::kSucceeded);

  auto &graph = fused_schedule_result.node_idx_to_scheduled_results[0][1].schedule_groups[0].impl_graphs[0];
  auto add = graph.FindNode("add");
  ASSERT_NE(add, nullptr);
  add->attr.sched.exec_condition = af::ExecuteCondition::kConditionInvalid;

  codegen::CodegenResult result;
  EXPECT_EQ(mlir::bridge::RunMlirCodegen(fused_schedule_result, mlir::bridge::MlirCallerScene::kInductor, "Ascend910B1",
                                         result),
            mlir::bridge::MlirCodegenStatus::kFailed);
  // 失败的前提是 `result` 没被动过，避免调用方拿到半成品。
  EXPECT_TRUE(result.kernel.empty());
  EXPECT_TRUE(result.tiling.empty());
  EXPECT_TRUE(result.tiling_data.empty());
}

// Changing an Add's type tests bridge dispatch only; it does not create a valid
// schedule for ATT or legacy codegen.
TEST(CodegenOrchestrationTest, UnsupportedOperationTypesEmitBridgeShortfall) {
  for (const char *type : {"UnregisteredOp", "ArgMax", "ArgMaxMultiRPhase1", "ArgMaxMultiRPhase2"}) {
    SCOPED_TRACE(type);
    auto scheduled = BuildValidFusedScheduleResult();
    auto &plan = scheduled.node_idx_to_scheduled_results[0][0];
    auto node = plan.schedule_groups[0].impl_graphs[0].FindNode("add");
    ASSERT_NE(node, nullptr);
    node->GetOpDesc()->SetType(type);
    mlir::MLIRContext context;
    mlir::afir::CapabilityShortfallObserver observer(context);
    EXPECT_FALSE(mlir::afir::bridge::PlanToAfir(context, scheduled, plan, 0, 0));
    EXPECT_TRUE(observer.sawShortfall());
  }
}

TEST(CodegenOrchestrationTest, ValidArgMaxSchedulePassesAttAndFailsInMlirConfig) {
  mlir::test::ScopedEnvVar enabled("AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=mlir;");
  af::testing::AscGraphBuilder builder("argmax_fallback");
  builder.Loops({af::testing::Sym(16), af::testing::Sym(32)}).Data("x", 0).Load("load", "x");
  auto graph = mlir::test::AddReduce<af::ascir_op::ArgMax>(
                   builder, "reduce", "load", {af::testing::Sym(16), af::testing::Sym(32)}, {1}, af::DT_INT64)
                   .Store("store", "reduce")
                   .Output("output", "store", 0, af::DT_INT64)
                   .Build();
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult scheduled;
  ASSERT_EQ(optimizer.Optimize(graph, scheduled), af::SUCCESS);
  codegen::CodegenResult result;
  EXPECT_EQ(mlir::bridge::RunMlirCodegen(scheduled, mlir::bridge::MlirCallerScene::kInductor, "Ascend910B1", result),
            mlir::bridge::MlirCodegenStatus::kFailed);
  EXPECT_TRUE(result.kernel.empty());
  EXPECT_TRUE(result.tiling.empty());
  EXPECT_TRUE(result.tiling_data.empty());
}

// 严格模式的正例：**同一张图、同一处能力不足**，缺省回落（上一条用例）而严格变红。
//
// 两条用例共用 `exec_condition = kConditionInvalid` 这一处埋点，为的是让差别只落在取值上。
// 换一张图写这一条，变红可能来自别的原因，那就什么都没证明。
//
// 同时钉住两件事：
//   `result` 仍然没被动过——严格模式只改结论，不改「失败时不许留半成品」这条；
//   成功的图在严格模式下**照旧成功**，否则这台量具量出来的全是红的，没有信息。
TEST(CodegenOrchestrationTest, StrictModeTurnsCapabilityShortfallIntoFailure) {
  mlir::test::ScopedEnvVar scoped_mlir_env("AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=mlir;");
  auto fused_schedule_result = BuildValidFusedScheduleResult();
  codegen::CodegenResult baseline;
  ASSERT_EQ(mlir::bridge::RunMlirCodegen(fused_schedule_result, mlir::bridge::MlirCallerScene::kInductor, "Ascend910B1",
                                         baseline),
            mlir::bridge::MlirCodegenStatus::kSucceeded);

  auto &graph = fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  auto add = graph.FindNode("add");
  ASSERT_NE(add, nullptr);
  add->attr.sched.exec_condition = af::ExecuteCondition::kConditionInvalid;

  codegen::CodegenResult result;
  EXPECT_EQ(mlir::bridge::RunMlirCodegen(fused_schedule_result, mlir::bridge::MlirCallerScene::kInductor, "Ascend910B1",
                                         result),
            mlir::bridge::MlirCodegenStatus::kFailed);
  EXPECT_TRUE(result.kernel.empty());
  EXPECT_TRUE(result.tiling.empty());
  EXPECT_TRUE(result.tiling_data.empty());
}

// 严格模式**不收紧** Cube 融合那道闸门：它明确不在这条路的范围内，收紧只会让 legacy 的
// Cube 用例整片变红。于是 `kNotSupported` 在严格模式下只剩「没开路 / 不在范围」两个来路，
// 「回落」与它们因此分得开——这正是那张接管表不出现「状态未知」行的依据。
TEST(CodegenOrchestrationTest, StrictModeStillFallsBackForCubeFusion) {
  mlir::test::ScopedEnvVar scoped_mlir_env("AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=mlir;");
  auto fused_schedule_result = mlir::test::MakeFusedResult();
  fused_schedule_result.node_idx_to_scheduled_results[0][0].cube_type = ascir::CubeTemplateType::kFixpip;
  codegen::CodegenResult result;
  EXPECT_EQ(mlir::bridge::RunMlirCodegen(fused_schedule_result, mlir::bridge::MlirCallerScene::kInductor, "Ascend910B1",
                                         result),
            mlir::bridge::MlirCodegenStatus::kNotSupported);
}

// 闸门标签的归属：**每一档取值各走一条路，各自落在自己那道闸门上**。
//
// 这一条是整个严格模式里最容易答错的地方——插错位置不会让任何既有用例变红，只会让接管表把
// 一行归到别的桶里，而那张表是后面每一项的验收手段。所以这里逐格钉住 `gate=` 与
// `shortfall=` 两列，并且**要求标签恰好一行**：少一行说明有出口没插，多一行说明某个出口
// 被插了两次。
//
// `shortfall` 这一列单独钉，是因为它与 `gate` 不能互推：同一道 `lowering` 闸门既可能是能力
// 不足（缺省那一档会回落）也可能是坏了（缺省那一档也失败），表里那两态全靠它分。
TEST(CodegenOrchestrationTest, StrictModeLabelsTheGateThatStoppedTheGraph) {
  mlir::test::ScopedEnvVar scoped_mlir_env("AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=mlir;");
  const std::string marker = mlir::bridge::kMlirCodegenStrictMarker.str();

  const auto count_markers = [&marker](const std::string &text) {
    size_t count = 0;
    for (size_t at = text.find(marker); at != std::string::npos; at = text.find(marker, at + marker.size())) {
      ++count;
    }
    return count;
  };

  // 接管：一道都没拦下。
  {
    auto fused_schedule_result = BuildValidFusedScheduleResult();
    codegen::CodegenResult result;
    mlir::test::ScopedStderrCapture capture;
    const auto status = mlir::bridge::RunMlirCodegen(fused_schedule_result, mlir::bridge::MlirCallerScene::kInductor,
                                                     "Ascend910B1", result);
    const std::string text = capture.Text();
    ASSERT_EQ(status, mlir::bridge::MlirCodegenStatus::kSucceeded) << text;
    EXPECT_EQ(count_markers(text), 1U) << text;
    EXPECT_NE(text.find(marker + " gate=none shortfall=no"), std::string::npos) << text;
  }

  // Cube 融合：不在范围，结论仍是回落。
  {
    auto fused_schedule_result = mlir::test::MakeFusedResult();
    fused_schedule_result.node_idx_to_scheduled_results[0][0].cube_type = ascir::CubeTemplateType::kFixpip;
    codegen::CodegenResult result;
    mlir::test::ScopedStderrCapture capture;
    const auto status = mlir::bridge::RunMlirCodegen(fused_schedule_result, mlir::bridge::MlirCallerScene::kInductor,
                                                     "Ascend910B1", result);
    const std::string text = capture.Text();
    ASSERT_EQ(status, mlir::bridge::MlirCodegenStatus::kNotSupported) << text;
    EXPECT_EQ(count_markers(text), 1U) << text;
    EXPECT_NE(text.find(marker + " gate=cube-fusion shortfall=no"), std::string::npos) << text;
  }

  // 能力不足：拦在下降那一道，且 `shortfall=yes` —— 缺省那一档会回落。
  {
    auto fused_schedule_result = BuildValidFusedScheduleResult();
    auto &graph = fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
    auto add = graph.FindNode("add");
    ASSERT_NE(add, nullptr);
    add->attr.sched.exec_condition = af::ExecuteCondition::kConditionInvalid;
    codegen::CodegenResult result;
    mlir::test::ScopedStderrCapture capture;
    const auto status = mlir::bridge::RunMlirCodegen(fused_schedule_result, mlir::bridge::MlirCallerScene::kInductor,
                                                     "Ascend910B1", result);
    const std::string text = capture.Text();
    ASSERT_EQ(status, mlir::bridge::MlirCodegenStatus::kFailed) << text;
    EXPECT_EQ(count_markers(text), 1U) << text;
    EXPECT_NE(text.find(marker + " gate=lowering shortfall=yes"), std::string::npos) << text;
  }

  // 坏了：芯片解析不出来，拦在 soc-version 那一道，且 `shortfall=no`。
  {
    auto fused_schedule_result = BuildValidFusedScheduleResult();
    codegen::CodegenResult result;
    mlir::test::ScopedStderrCapture capture;
    const auto status = mlir::bridge::RunMlirCodegen(fused_schedule_result, mlir::bridge::MlirCallerScene::kInductor,
                                                     "NotARealSoc123", result);
    const std::string text = capture.Text();
    ASSERT_EQ(status, mlir::bridge::MlirCodegenStatus::kFailed) << text;
    EXPECT_EQ(count_markers(text), 1U) << text;
    EXPECT_NE(text.find(marker + " gate=pipeline-session shortfall=no"), std::string::npos) << text;
  }
}

// 缺省那一档**一个字节都不多输出**：标签只在严格模式下打。
// 没有这一条，产品部署会多出一行谁也没要的 stderr。
TEST(CodegenOrchestrationTest, DefaultModeEmitsNoGateLabel) {
  mlir::test::ScopedEnvVar scoped_mlir_env("AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=fallback;");
  auto fused_schedule_result = BuildValidFusedScheduleResult();
  codegen::CodegenResult result;
  mlir::test::ScopedStderrCapture capture;
  const auto status = mlir::bridge::RunMlirCodegen(fused_schedule_result, mlir::bridge::MlirCallerScene::kInductor,
                                                   "Ascend910B1", result);
  const std::string text = capture.Text();
  ASSERT_EQ(status, mlir::bridge::MlirCodegenStatus::kSucceeded) << text;
  EXPECT_EQ(text.find(mlir::bridge::kMlirCodegenStrictMarker.str()), std::string::npos) << text;
}

// 严格模式的负例：**硬失败在两档取值下都是 `kFailed`**。
// 严格模式是把一类回落升级成失败，不是把失败降级——如果它顺手改了这一类的结论，那它就不只
// 是一台量具了。
TEST(CodegenOrchestrationTest, StrictModeKeepsHardFailureFailing) {
  mlir::test::ScopedEnvVar scoped_mlir_env("AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=mlir;");
  auto fused_schedule_result = BuildValidFusedScheduleResult();
  codegen::CodegenResult result;
  EXPECT_EQ(mlir::bridge::RunMlirCodegen(fused_schedule_result, mlir::bridge::MlirCallerScene::kInductor,
                                         "NotARealSoc123", result),
            mlir::bridge::MlirCodegenStatus::kFailed);
}

TEST(CodegenOrchestrationTest, InvalidSocVersionFails) {
  mlir::test::ScopedEnvVar scoped_mlir_env("AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=mlir;");
  auto fused_schedule_result = BuildValidFusedScheduleResult();
  codegen::CodegenResult result;
  auto status = mlir::bridge::RunMlirCodegen(fused_schedule_result, mlir::bridge::MlirCallerScene::kInductor,
                                             "NotARealSoc123", result);
  EXPECT_EQ(status, mlir::bridge::MlirCodegenStatus::kFailed);
}

TEST(CodegenOrchestrationTest, EmptySocVersionUsesRuntimeSocName) {
  mlir::test::ScopedEnvVar scoped_mlir_env("AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=mlir;");
  auto fused_schedule_result = BuildValidFusedScheduleResult();
  codegen::CodegenResult result;
  auto status =
      mlir::bridge::RunMlirCodegen(fused_schedule_result, mlir::bridge::MlirCallerScene::kInductor, "", result);
  ASSERT_EQ(status, mlir::bridge::MlirCodegenStatus::kSucceeded);
  EXPECT_FALSE(result.kernel.empty());
  EXPECT_FALSE(result.tiling.empty());
  EXPECT_FALSE(result.tiling_data.empty());
}

TEST(CodegenOrchestrationTest, EmptySocVersionFailsWhenRuntimeSocNameIsUnavailable) {
  class EmptySocNameRuntimeStub final : public ge::RuntimeStub {
   public:
    const char *aclrtGetSocName() override {
      return nullptr;
    }
  };

  ge::RuntimeStub::SetInstance(std::make_shared<EmptySocNameRuntimeStub>());
  mlir::test::ScopedEnvVar scoped_mlir_env("AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=mlir;");
  auto fused_schedule_result = BuildValidFusedScheduleResult();
  codegen::CodegenResult result;
  auto status =
      mlir::bridge::RunMlirCodegen(fused_schedule_result, mlir::bridge::MlirCallerScene::kInductor, "", result);
  ge::RuntimeStub::Reset();

  EXPECT_EQ(status, mlir::bridge::MlirCodegenStatus::kFailed);
  EXPECT_TRUE(result.kernel.empty());
  EXPECT_TRUE(result.tiling.empty());
  EXPECT_TRUE(result.tiling_data.empty());
}

// Compare the exported launch declaration. Internal kernel calls use typed GM
// pointers in the MLIR path; AutofuseLaunchEmitterTest checks that body separately.
TEST(CodegenOrchestrationTest, AutofuseLaunchSignatureMatchesLegacy) {
  auto fused_schedule_result = BuildValidFusedScheduleResult();

  codegen::Codegen legacy_codegen(codegen::CodegenOptions{});
  codegen::CodegenResult legacy_result;
  ASSERT_EQ(legacy_codegen.GenerateForInductor(fused_schedule_result, legacy_result), af::SUCCESS);

  mlir::test::ScopedEnvVar scoped_mlir_env("AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=mlir;");
  codegen::CodegenResult mlir_result;
  auto status = mlir::bridge::RunMlirCodegen(fused_schedule_result, mlir::bridge::MlirCallerScene::kInductor,
                                             "Ascend910B1", mlir_result);
  ASSERT_EQ(status, mlir::bridge::MlirCodegenStatus::kSucceeded);

  const std::string marker = "extern \"C\" int64_t AutofuseLaunch(";
  auto legacy_pos = legacy_result.kernel.find(marker);
  auto mlir_pos = mlir_result.kernel.find(marker);
  ASSERT_NE(legacy_pos, std::string::npos);
  ASSERT_NE(mlir_pos, std::string::npos);
  auto legacy_end = legacy_result.kernel.find('\n', legacy_pos);
  auto mlir_end = mlir_result.kernel.find('\n', mlir_pos);
  ASSERT_NE(legacy_end, std::string::npos);
  ASSERT_NE(mlir_end, std::string::npos);
  EXPECT_EQ(legacy_result.kernel.substr(legacy_pos, legacy_end - legacy_pos),
            mlir_result.kernel.substr(mlir_pos, mlir_end - mlir_pos));
}

// 扩展头的内容由 `AscendCApiRegistry` 提供，而三个 registrar 是靠静态初始化注册进去的
// ——一个没被引用的 TU 有可能被 linker 丢掉，表现是 provider 返回空、`#include` 原样
// 留在 kernel 里、device 编译才报找不到头。mem 侧那张表由 load_abs_store 的 E2E 覆盖
// （`abs.h`），reg-base 那张表在 MLIR 路径上还没有用例走到，所以这里直接问它要一条。
TEST(CodegenOrchestrationTest, RegBaseApiHeadersReachTheRegistry) {
  auto &registry = codegen::AscendCApiRegistry::GetInstance();
  EXPECT_FALSE(registry.GetFileContent("cast_reg_base.h").empty());
  EXPECT_FALSE(registry.GetFileContent("abs.h").empty());
  EXPECT_FALSE(registry.GetFileContent("reduce_init.h").empty());
  // 未注册的名字返回空串，这是 provider 「取不到就把 include 原样留下」的依据。
  EXPECT_TRUE(registry.GetFileContent("no_such_extension_header.h").empty());
}

// 每个扩展 op 上写死的头文件名都要在供给表里取得到。取不到不会在 MLIR 这一层报错：
// include 会被原样留在 kernel 源码里，最后炸在编译器的「找不到这个头」上，离写错的
// 那个字符串已经隔了几层。名单是手抄的——新加扩展 op 时忘了加进来只是少一条覆盖，
// 不会让这条断言说谎。
// A valid alternative must not hide a lowering error in another scheduled plan.
TEST(CodegenOrchestrationTest, UnlowerableCandidateRejectsTheCompleteSchedule) {
  mlir::test::ScopedEnvVar scoped_mlir_env("AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=mlir;");
  auto fused_schedule_result = BuildFusedScheduleResultWithUnlowerableFirstPlan();
  codegen::CodegenResult result;
  auto status = mlir::bridge::RunMlirCodegen(fused_schedule_result, mlir::bridge::MlirCallerScene::kInductor,
                                             "Ascend910B1", result);
  EXPECT_EQ(status, mlir::bridge::MlirCodegenStatus::kFailed);
  EXPECT_TRUE(result.kernel.empty());
  EXPECT_TRUE(result.tiling.empty());
}

// Failed lowering must not produce an empty successful kernel.
TEST(CodegenOrchestrationTest, AllCandidatesUnlowerableFails) {
  mlir::test::ScopedEnvVar scoped_mlir_env("AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=mlir;");
  auto fused_schedule_result = BuildFusedScheduleResultWithOnlyUnlowerablePlan();
  codegen::CodegenResult result;
  auto status = mlir::bridge::RunMlirCodegen(fused_schedule_result, mlir::bridge::MlirCallerScene::kInductor,
                                             "Ascend910B1", result);
  EXPECT_EQ(status, mlir::bridge::MlirCodegenStatus::kFailed);
}

TEST(CodegenOrchestrationTest, FallbackConfigCoversUnlowerableCandidates) {
  mlir::test::ScopedEnvVar scoped_mlir_env("AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=fallback;");
  auto fused_schedule_result = BuildFusedScheduleResultWithOnlyUnlowerablePlan();
  codegen::CodegenResult result;
  auto status = mlir::bridge::RunMlirCodegen(fused_schedule_result, mlir::bridge::MlirCallerScene::kInductor,
                                             "Ascend910B1", result);
  EXPECT_EQ(status, mlir::bridge::MlirCodegenStatus::kNotSupported);
}

// Dynamic shapes retain every candidate for ATT runtime selection.
TEST(CodegenOrchestrationTest, DynamicShapeKeepsEveryCandidate) {
  mlir::test::ScopedEnvVar scoped_mlir_env("AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=mlir;");
  auto fused_schedule_result = BuildFusedScheduleResultWithTwoValidPlans();
  codegen::CodegenResult result;
  auto status = mlir::bridge::RunMlirCodegen(fused_schedule_result, mlir::bridge::MlirCallerScene::kInductor,
                                             "Ascend910B1", result);
  ASSERT_EQ(status, mlir::bridge::MlirCodegenStatus::kSucceeded);
  EXPECT_NE(result.tiling.find("tiling->graph0_result0_g0_tiling_data.set_s0(s0);"), std::string::npos);
  EXPECT_NE(result.tiling.find("tiling->graph0_result1_g0_tiling_data.set_s0(s0);"), std::string::npos);
}

// Even this fixed-size graph has solver-bound ATT costs (for example block_dim).
TEST(CodegenOrchestrationTest, StaticShapeKeepsBothPlansForRuntimeTiling) {
  mlir::test::ScopedEnvVar scoped_mlir_env("AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=mlir;");
  auto fused_schedule_result = BuildFusedScheduleResultWithTwoValidPlans(true);
  codegen::CodegenResult result;
  auto status = mlir::bridge::RunMlirCodegen(fused_schedule_result, mlir::bridge::MlirCallerScene::kInductor,
                                             "Ascend910B1", result);
  ASSERT_EQ(status, mlir::bridge::MlirCodegenStatus::kSucceeded);
  EXPECT_NE(result.tiling.find("graph0_result0_g0_tiling_data"), std::string::npos);
  EXPECT_NE(result.tiling.find("graph0_result1_g0_tiling_data"), std::string::npos);
  EXPECT_NE(result.kernel.find("subgraph0_plan0"), std::string::npos);
  EXPECT_NE(result.kernel.find("subgraph0_plan1"), std::string::npos);
  EXPECT_NE(result.tiling.find("optiling::GetTiling(*tiling"), std::string::npos);
  // Runtime ATT sources use the normal split-file compilation contract.
  EXPECT_NE(result.tiling.find("AUTOFUSE_SPLIT_FILE_BEGIN"), std::string::npos);
  EXPECT_NE(result.tiling_data.find("graph0_tiling_key"), std::string::npos);
  EXPECT_EQ(fused_schedule_result.node_idx_to_scheduled_results[0].size(), 2U);
}

TEST(CodegenOrchestrationTest, EmptyExtensionHeadersAreEmptyCollections) {
  EXPECT_TRUE(mlir::ascendc::PadOp::getExtensionHeaders().empty());
  EXPECT_TRUE(mlir::ascendc::DataCopyBlkAlignedOp::getExtensionHeaders().empty());
  const auto official = mlir::ascendc::PadOp::getExtensionOfficialIncludes();
  ASSERT_EQ(official.size(), 1U);
  EXPECT_EQ(official.front(), "lib/pad/pad.h");
  const auto prelude = mlir::ascendc::DataCopyBlkAlignedOp::getExtensionPreludeHeaders();
  ASSERT_EQ(prelude.size(), 1U);
  EXPECT_EQ(prelude.front(), "utils.h");
}

TEST(CodegenOrchestrationTest, StaticCodegenDoesNotInvokeHostCompiler) {
  mlir::test::ScopedEnvVar compiler("CXX", "/nonexistent/autofuse-test-cxx");
  const auto schedule = BuildFusedScheduleResultWithTwoValidPlans(true);
  for (const char *mode : {"mlir", "fallback"}) {
    mlir::test::ScopedEnvVar config("AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=" + std::string(mode) + ";");
    codegen::CodegenResult result;
    const auto status =
        mlir::bridge::RunMlirCodegen(schedule, mlir::bridge::MlirCallerScene::kInductor, "Ascend910B1", result);
    EXPECT_EQ(status, mlir::bridge::MlirCodegenStatus::kSucceeded);
    EXPECT_FALSE(result.kernel.empty());
    EXPECT_FALSE(result.tiling.empty());
    EXPECT_FALSE(result.tiling_data.empty());
  }
}

TEST(CodegenOrchestrationTest, StaticPgoRequestRetainsRuntimeTiling) {
  auto &pgo = att::AutoFuseConfig::MutablePgoStrategyConfig();
  const auto previous = pgo;
  auto restore = llvm::make_scope_exit([&] { pgo = previous; });
  pgo.is_first_init = false;
  pgo.set_env_enable_autofuse_pgo = true;
  pgo.enable_autofuse_pgo = "true";
  const auto schedule = BuildFusedScheduleResultWithTwoValidPlans(true);
  for (const char *mode : {"mlir", "fallback"}) {
    mlir::test::ScopedEnvVar config("AUTOFUSE_DFX_FLAGS", "--mlir_codegen_config=" + std::string(mode) + ";");
    codegen::CodegenResult result;
    testing::internal::CaptureStderr();
    const auto status =
        mlir::bridge::RunMlirCodegen(schedule, mlir::bridge::MlirCallerScene::kInductor, "Ascend910B1", result);
    const auto diagnostic = testing::internal::GetCapturedStderr();
    EXPECT_EQ(status, mlir::bridge::MlirCodegenStatus::kSucceeded) << diagnostic;
    EXPECT_NE(result.tiling.find("optiling::GetTiling(*tiling"), std::string::npos);
    EXPECT_NE(result.kernel.find("subgraph0_plan0"), std::string::npos);
    EXPECT_NE(result.kernel.find("subgraph0_plan1"), std::string::npos);
  }
}

TEST(CodegenOrchestrationTest, DeclaredExtensionHeadersResolveInTheRegistry) {
  auto &registry = codegen::AscendCApiRegistry::GetInstance();
  std::vector<llvm::StringRef> headers;
  for (llvm::ArrayRef<llvm::StringRef> declared : {
           mlir::ascendc::AbsExtendOp::getExtensionHeaders(),
           mlir::ascendc::NegExtendOp::getExtensionHeaders(),
           mlir::ascendc::NegExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::ReciprocalExtendOp::getExtensionHeaders(),
           mlir::ascendc::SigmoidExtendOp::getExtensionHeaders(),
           mlir::ascendc::DataCopyPadExtendOp::getExtensionHeaders(),
           mlir::ascendc::DataCopyPadExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::DataCopyNddmaRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::IsnanExtendOp::getExtensionHeaders(),
           mlir::ascendc::IsFiniteExtendOp::getExtensionHeaders(),
           mlir::ascendc::IsInfExtendOp::getExtensionHeaders(),
           // 特殊函数一元族的第一个（2026-10-20）。它正是上面那段说的形态：头名是
           // 生成名 `ndtr_reg_base.h`，源文件叫 `ndtr.h`，写错那一边这条才看得出来。
           mlir::ascendc::NdtrExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::NdtriExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::LogNdtrExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::I0ExtendRegBaseOp::getExtensionHeaders(),
           // 本族第一个声明两个头的：上面那条「列表里每一项都要取得到」正是为它写的。
           mlir::ascendc::AiryAiExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::BesselJ0ExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::BesselJ1ExtendRegBaseOp::getExtensionHeaders(),
           // 第 4 批的两个是本族第一个声明**两个扩展头**的，所以「列表里每一项都要取得到」
           // 这条对它们才是真有内容的：漏注册 `modified_bessel_i0_reg_base.h` 与漏注册
           // `i0e_reg_base.h` 的表现一样。
           mlir::ascendc::I0eExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::I1eExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::I0eExtendRegBaseOp::getExtensionPreludeHeaders(),
           mlir::ascendc::I1eExtendRegBaseOp::getExtensionPreludeHeaders(),
           // 第 5 批。`K0` / `K1` 同样声明两个扩展头，依赖序与字典序一致。
           mlir::ascendc::ModifiedBesselI0ExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::ModifiedBesselI1ExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::ModifiedBesselK0ExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::ModifiedBesselK1ExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::ModifiedBesselI0ExtendRegBaseOp::getExtensionPreludeHeaders(),
           mlir::ascendc::ModifiedBesselK0ExtendRegBaseOp::getExtensionPreludeHeaders(),
           // 第 6 批。九个有扩展头的（`log2` / `floor` / `lgamma` 走标准调用，没有头）。
           mlir::ascendc::LogicalXorExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::Exp2ExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::ErfinvExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::FmodExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::NextAfterExtendRegBaseOp::getExtensionHeaders(),
           // 第 7 批。27 个有扩展头的（`digamma` / `erfc` / `trunc` 走标准调用，没有头）。
           mlir::ascendc::AcosExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::AcoshExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::AsinExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::AsinhExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::AtanExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::AtanhExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::CeilExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::CosExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::CoshExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::ErfcxExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::RoundExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::SinExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::SinhExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::TanExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::BesselY0ExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::BesselY1ExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::SphericalBesselJ0ExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::ScaledModifiedBesselK0ExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::ScaledModifiedBesselK1ExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::Expm1ExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::Log1pExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::CopySignExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::TruncDivExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::Atan2ExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::ZetaExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::IgammaExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::IgammacExtendRegBaseOp::getExtensionHeaders(),
           // 第 9 批（多项式族八个）。**前导头也要逐个取**：两族各有一个 utils，它们走
           // preludes 而不是 headers——理由见 `Autofuse/OpVecUnary.td` 那段（字典序会把
           // `t` / `u` 排到 `utils` 前面，而依赖序要求 utils 先内联）。
           mlir::ascendc::ChebyshevPolynomialTExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::ChebyshevPolynomialUExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::ChebyshevPolynomialVExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::ChebyshevPolynomialWExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::ShiftedChebyshevPolynomialTExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::ShiftedChebyshevPolynomialUExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::ShiftedChebyshevPolynomialVExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::ShiftedChebyshevPolynomialWExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::ChebyshevPolynomialTExtendRegBaseOp::getExtensionPreludeHeaders(),
           mlir::ascendc::ChebyshevPolynomialUExtendRegBaseOp::getExtensionPreludeHeaders(),
           mlir::ascendc::ChebyshevPolynomialVExtendRegBaseOp::getExtensionPreludeHeaders(),
           mlir::ascendc::ChebyshevPolynomialWExtendRegBaseOp::getExtensionPreludeHeaders(),
           mlir::ascendc::ShiftedChebyshevPolynomialTExtendRegBaseOp::getExtensionPreludeHeaders(),
           mlir::ascendc::ShiftedChebyshevPolynomialUExtendRegBaseOp::getExtensionPreludeHeaders(),
           mlir::ascendc::ShiftedChebyshevPolynomialVExtendRegBaseOp::getExtensionPreludeHeaders(),
           mlir::ascendc::ShiftedChebyshevPolynomialWExtendRegBaseOp::getExtensionPreludeHeaders(),
           // step 2a：`Ceil2IntExtend` 与 `CastExtend` 共用 cast_reg_base.h，没有前导头。
           mlir::ascendc::Ceil2IntExtendRegBaseOp::getExtensionHeaders(),
           // step 4：三个都没有前导头——包装各自只转发给自己的标准调用，没有引别的名字。
           mlir::ascendc::BitwiseNotExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::BitwiseOrExtendRegBaseOp::getExtensionHeaders(),
           mlir::ascendc::BitwiseXorExtendRegBaseOp::getExtensionHeaders(),
           // 前导头：Bessel 那三个与已接通的 `bessel_j0` 同一对 utils；`igamma` / `igammac`
           // 的四个 helper 住在 `igammac_helper/` 子目录里，注册表的键带着那一段路径,
           // 写丢它会在这条断言上红——这正是这张名单要手工维护的理由。
           mlir::ascendc::BesselY0ExtendRegBaseOp::getExtensionPreludeHeaders(),
           mlir::ascendc::BesselY1ExtendRegBaseOp::getExtensionPreludeHeaders(),
           mlir::ascendc::SphericalBesselJ0ExtendRegBaseOp::getExtensionPreludeHeaders(),
           mlir::ascendc::ScaledModifiedBesselK0ExtendRegBaseOp::getExtensionPreludeHeaders(),
           mlir::ascendc::ScaledModifiedBesselK1ExtendRegBaseOp::getExtensionPreludeHeaders(),
           mlir::ascendc::IgammaExtendRegBaseOp::getExtensionPreludeHeaders(),
           mlir::ascendc::IgammacExtendRegBaseOp::getExtensionPreludeHeaders(),
           // 前导头也要逐项取得到：这一族的 utils 都走那张表，漏注册的表现与漏注册
           // 扩展头一样，是 kernel 里留下一条解析不了的 include。
           mlir::ascendc::AiryAiExtendRegBaseOp::getExtensionPreludeHeaders(),
           mlir::ascendc::BesselJ0ExtendRegBaseOp::getExtensionPreludeHeaders(),
           mlir::ascendc::BesselJ1ExtendRegBaseOp::getExtensionPreludeHeaders(),
           // 前导头走的是同一张表，而且它是后补进去的——legacy 从不查这张表取 utils.h，
           // 所以漏注册不会让 legacy 出问题，只会让 MLIR 这边把 include 原样留下。
           mlir::ascendc::ReciprocalExtendOp::getExtensionPreludeHeaders(),
           mlir::ascendc::SigmoidExtendOp::getExtensionPreludeHeaders(),
       }) {
    // 声明的是列表，**列表里每一项**都要取得到：漏注册第二个头的表现与漏注册第一个
    // 完全一样，而只查第一项的断言会放它过去。
    //
    // **上面那张 op 名单是手工维护的，加了新的扩展 op 就要在这里加一行。** 漏加的后果不是
    // 这条用例变弱而已：`datacopy_nddma_reg_base.h` 那一行就是补上的——它原先写成源文件名
    // `datacopy_nddma.h`，而注册表的键是 `api_regbase/CMakeLists.txt` 生成出来的
    // `x_reg_base.h`，查不到，于是 kernel 里留下一条解析不了的 include，一路炸到设备编译器。
    // 那时这张名单里没有这个 op，所以这条用例是绿的。
    EXPECT_FALSE(declared.empty());
    // 这里原先查的是「声明顺序必须已经是字典序」——`AddKernelIncludes` 当时按字典序发 include
    // 块，对声明顺序不是字典序的 op 直接报错。那条断言是 2026-10-21 由 `AiryAiExtend` 逼出来
    // 的：它要 `trigonometric_function_utils_reg_base.h` 先内联（`airy_ai.h` 调里面的
    // `AutofuseSin` / `AutofuseCos`），而那个名字的字典序排在 `airy_ai_reg_base.h` **后面**。
    // 当时 lit、UT、CPU 模拟三样全绿，只有 200 格接管表照出来——那一格从干净回落变成硬失败。
    // 那一课仍然成立，而且是另一条规则：utils 一类放**前导头**那一格（它排在所有扩展头前面），
    // 而不是塞进这个列表。前导表达不了的只是两个扩展头**之间**的先后。
    //
    // 现在发射层照声明次序排（`ApiExtensionOrder` 拓扑排，同层字典序），所以字典序不再是要求：
    // `I0e` 那种反字典序的依赖是合法声明——`i0e.h` 用的 `I0_A` / `I0_B` 只有
    // `modified_bessel_i0.h` 定义，而 `i0e_reg_base.h` 的字典序恰好在它前面。
    //
    // 留给这里查的是重名：同一个列表里重复一个名字在排序那边是个环，而环是**硬失败**——这个
    // op 一旦出现在图里就在 artifact-assembly 闸门上红，缺省档也不回落。跨 op 的次序冲突和
    // 成环归 `ApiExtensionOrderTest` 盯（那是排序本身的用例），这里只盯每个 op 自己那一份。
    std::vector<llvm::StringRef> names(declared.begin(), declared.end());
    llvm::sort(names);
    EXPECT_TRUE(llvm::unique(names) == names.end())
        << "扩展头声明列表里有重名，排序那边是个环，发不出来（第一个是 " << declared.front().str() << "）";
    headers.insert(headers.end(), declared.begin(), declared.end());
  }
  for (llvm::StringRef header : headers) {
    EXPECT_FALSE(registry.GetFileContent(header.str()).empty()) << header.str();
  }
  // 自足的包装不声明前导——空列表不会变成一次「查不到」的查询。
  EXPECT_TRUE(mlir::ascendc::NegExtendOp::getExtensionPreludeHeaders().empty());
  // 同一个包装的两张表各是各的键，不能有一边悄悄回落到另一边。
  EXPECT_NE(mlir::ascendc::NegExtendOp::getExtensionHeaders(),
            mlir::ascendc::NegExtendRegBaseOp::getExtensionHeaders());
}
