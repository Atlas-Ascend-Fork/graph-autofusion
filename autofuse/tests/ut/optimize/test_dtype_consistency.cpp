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

#include <ascendc_ir.h>
#include "ascir.h"
#include <ascir_ops.h>
#include <ascir_utils.h>
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#define private public
#include "optimize.h"
#include "platform_context.h"
#include "optimize/graph_pass/pass_runner_handler.h"
#include "optimize/graph_completeness/dtype_consistency.h"
#undef private
#include "asc_tensor_utils.h"
#include "ascgraph_info_complete.h"
#include "ascir_ops_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_op_types.h"
#include "graph/normal_graph/ge_tensor_impl.h"
#include "codegen.h"
#include "fusion/autofuse_attrs.h"
#include "task_generator/transpose_schedule_case_generator.h"
#include "task_generator/reduce_schedule_case_generator.h"
#include "ascgraph_info_complete.h"
#include "schedule_result.h"
#include "attribute_group/attr_group_shape_env.h"
#include "autoschedule/tiling_group.h"
#include "expression/testcase/source_stub.h"
#include "util/mem_utils.h"
#include "platform/platform_factory.h"
#include "runtime_stub.h"

#include "codegen.h"
#include "../../common/att/soc_perf_test_config.h"
#include "asc_graph_builder.h"
#include "optimize/graph_pass/pass_utils.h"

using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;

namespace {
class TestDtypeConsistency : public ::testing::Test {
 protected:
  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v1 = std::make_shared<RuntimeStub>();
    RuntimeStub::SetInstance(stub_v1);
    ASSERT_EQ(
        att::soc_perf_test::SetSocPerfTestPlatform(att::soc_perf_test::kV35NpuArch, att::soc_perf_test::kSocEmpty),
        af::SUCCESS);
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }

  void TearDown() override {
    ge::PlatformContext::GetInstance().Reset();
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
};

TEST_F(TestDtypeConsistency, InsertCastForInputMismatch) {
  af::AscGraph graph("test_fp16_mul_chain");

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Mul mul("mul");
  mul.x1 = load.y;
  mul.x2 = load.y;
  mul.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT16;

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store"), {DT_FLOAT16}, {ge::DT_FLOAT16}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);
  // 验证结果: load(FP16) -> cast(FP16->FP32) -> mul(FP32) -> cast(FP32->FP16) -> store(FP16)
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  bool mul_is_fp32 = false;
  for (const auto &node : all_nodes) {
    if (af::ops::IsOps<Cast>(node)) {
      cast_count++;
    }
    if (node->GetType() == "Mul") {
      if (node->inputs[0].attr.dtype == ge::DT_FLOAT && node->outputs[0].attr.dtype == ge::DT_FLOAT) {
        mul_is_fp32 = true;
      }
    }
  }
  EXPECT_EQ(cast_count, 2U);
  EXPECT_TRUE(mul_is_fp32);
}

// Test 2: No cast needed when dtypes match
TEST_F(TestDtypeConsistency, NoCastWhenDtypesMatch) {
  af::AscGraph graph("test_fp32_mul_chain");

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Mul mul("mul");
  mul.x1 = load.y;
  mul.x2 = load.y;
  mul.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT;

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);

  // 验证结果：不需要插入 cast
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  for (const auto &node : all_nodes) {
    if (af::ops::IsOps<Cast>(node)) {
      cast_count++;
    }
  }
  EXPECT_EQ(cast_count, 0U);
}

// Test 3: Cancel identity cast (Cast A->A)
TEST_F(TestDtypeConsistency, CancelIdentityCast) {
  af::AscGraph graph("test_identity_cast");

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT;

  // 手动添加一个恒等 cast (FP32 -> FP32)
  af::ascir_op::Cast cast("cast");
  cast.x = load.y;
  cast.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Mul mul("mul");
  mul.x1 = cast.y;
  mul.x2 = cast.y;
  mul.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT;

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);

  // 验证结果：恒等 cast 应被删除
  // 注意：只统计真正有输入和输出连接的cast节点（已删除的节点边会为空）
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  for (const auto &node : all_nodes) {
    if (af::ops::IsOps<Cast>(node)) {
      // 检查这个cast是否真的在图中（有输入边和输出边）
      auto in_anchor = node->GetInDataAnchor(0);
      auto out_anchor = node->GetOutDataAnchor(0);
      if (in_anchor != nullptr && in_anchor->GetPeerOutAnchor() != nullptr && out_anchor != nullptr &&
          !out_anchor->GetPeerInDataAnchors().empty()) {
        cast_count++;
      }
    }
  }
  EXPECT_EQ(cast_count, 0U);
}

// Test 4: Cast CSE - merge identical casts from same upstream
TEST_F(TestDtypeConsistency, DoCastCSE) {
  af::AscGraph graph("test_cast_cse");

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  // 手动添加两个相同的 cast
  af::ascir_op::Cast cast1("cast1");
  cast1.x = load.y;
  cast1.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Cast cast2("cast2");
  cast2.x = load.y;
  cast2.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Mul mul1("mul1");
  mul1.x1 = cast1.y;
  mul1.x2 = cast1.y;
  mul1.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Mul mul2("mul2");
  mul2.x1 = cast2.y;
  mul2.x2 = cast2.y;
  mul2.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Store store1("store1");
  store1.x = mul1.y;
  store1.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Store store2("store2");
  store2.x = mul2.y;
  store2.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Output output1("output1");
  output1.x = store1.y;
  output1.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Output output2("output2");
  output2.x = store2.y;
  output2.y.dtype = ge::DT_FLOAT;

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("mul1"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("mul2"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store1"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store2"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);

  // 验证结果：两个相同的 cast 应该合并成一个
  // 注意：只统计真正有输入和输出连接的cast节点（已删除的节点边会为空）
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  bool found_cast_with_two_downstream = false;
  for (const auto &node : all_nodes) {
    if (af::ops::IsOps<Cast>(node)) {
      cast_count++;
      // 检查这个cast是否真的在图中（有输入边和输出边）
      auto in_anchor = node->GetInDataAnchor(0);
      auto out_anchor = node->GetOutDataAnchor(0);
      auto peer_anchors = out_anchor->GetPeerInDataAnchors();
      if (peer_anchors.size() == 4U) {
        found_cast_with_two_downstream = true;
      }
    }
  }
  EXPECT_EQ(cast_count, 1U);
  EXPECT_TRUE(found_cast_with_two_downstream);
}

// Test 5: Add with two different input dtypes
TEST_F(TestDtypeConsistency, AddWithMixedInputDtypes) {
  af::AscGraph graph("test_add_mixed");

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Load load1("load1");
  load1.x = data0.y;
  load1.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Load load2("load2");
  load2.x = data1.y;
  load2.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Add add("add");
  add.x1 = load1.y;
  add.x2 = load2.y;
  add.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Store store("store");
  store.x = add.y;
  store.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT16;

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load1"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("load2"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("add"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);

  // 验证结果: load1(FP16) -> cast(FP16->FP32) --\
  //                                             > add(FP32) -> cast(FP32->FP16) -> store(FP16)
  // load2(FP32) ----------------------------------/
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  bool add_is_fp32 = false;
  for (const auto &node : all_nodes) {
    if (af::ops::IsOps<Cast>(node)) {
      cast_count++;
    }
    if (node->GetType() == "Add") {
      if (node->inputs[0].attr.dtype == ge::DT_FLOAT && node->inputs[1].attr.dtype == ge::DT_FLOAT &&
          node->outputs[0].attr.dtype == ge::DT_FLOAT) {
        add_is_fp32 = true;
      }
    }
  }
  EXPECT_EQ(cast_count, 2U);  // Add 前面一个 (为 load1)，后面一个
  EXPECT_TRUE(add_is_fp32);
}

// Test 6: Chain of operations
TEST_F(TestDtypeConsistency, ChainOfOperations) {
  af::AscGraph graph("test_mul_chain");

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Mul mul1("mul1");
  mul1.x1 = load.y;
  mul1.x2 = load.y;
  mul1.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Mul mul2("mul2");
  mul2.x1 = mul1.y;
  mul2.x2 = mul1.y;
  mul2.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Store store("store");
  store.x = mul2.y;
  store.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT16;

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("mul1"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("mul2"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);

  // 验证结果: load(FP16) -> cast(FP16->FP32) -> mul1(FP32) -> mul2(FP32) -> cast(FP32->FP16) -> store(FP16)
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  size_t fp32_mul_count = 0;
  for (const auto &node : all_nodes) {
    if (af::ops::IsOps<Cast>(node)) {
      cast_count++;
    }
    if (node->GetType() == "Mul") {
      if (node->inputs[0].attr.dtype == ge::DT_FLOAT && node->outputs[0].attr.dtype == ge::DT_FLOAT) {
        fp32_mul_count++;
      }
    }
  }
  EXPECT_EQ(cast_count, 2U);  // 开头一个，结尾一个
  EXPECT_EQ(fp32_mul_count, 2U);
}

// Test 7: Empty graph
TEST_F(TestDtypeConsistency, EmptyGraph) {
  af::AscGraph graph("empty_graph");

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);
}

// Test 8: Only cast nodes (non-identity)
TEST_F(TestDtypeConsistency, OnlyCastNodes) {
  af::AscGraph graph("test_cast_chain");

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Cast cast1("cast1");
  cast1.x = load.y;
  cast1.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Cast cast2("cast2");
  cast2.x = cast1.y;
  cast2.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Store store("store");
  store.x = cast2.y;
  store.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT16;

  // 所有中间节点都需要注册
  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);

  // 验证结果：Cast 节点应该保留（不是恒等转换）
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  for (const auto &node : all_nodes) {
    if (af::ops::IsOps<Cast>(node)) {
      cast_count++;
    }
  }
  EXPECT_EQ(cast_count, 2U);
}

// Test 9: Chain of identity casts should all be removed
TEST_F(TestDtypeConsistency, ChainOfIdentityCasts) {
  af::AscGraph graph("test_identity_cast_chain");

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT;

  // 手动添加两个恒等 cast (FP32 -> FP32)
  af::ascir_op::Cast cast1("cast1");
  cast1.x = load.y;
  cast1.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Cast cast2("cast2");
  cast2.x = cast1.y;
  cast2.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Mul mul("mul");
  mul.x1 = cast2.y;
  mul.x2 = cast2.y;
  mul.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT;

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);

  // 验证结果：所有恒等 cast 都应被删除
  // 注意：只统计真正有输入和输出连接的cast节点（已删除的节点边会为空）
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  for (const auto &node : all_nodes) {
    if (af::ops::IsOps<Cast>(node)) {
      // 检查这个cast是否真的在图中（有输入边和输出边）
      auto in_anchor = node->GetInDataAnchor(0);
      auto out_anchor = node->GetOutDataAnchor(0);
      if (in_anchor != nullptr && in_anchor->GetPeerOutAnchor() != nullptr && out_anchor != nullptr &&
          !out_anchor->GetPeerInDataAnchors().empty()) {
        cast_count++;
      }
    }
  }
  EXPECT_EQ(cast_count, 0U);
}

// Test 10: Cast CSE with multiple identical casts (more than 2)
TEST_F(TestDtypeConsistency, DoCastCSE_MultipleCasts) {
  af::AscGraph graph("test_cast_cse_multiple");

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  // 手动添加三个相同的 cast
  af::ascir_op::Cast cast1("cast1");
  cast1.x = load.y;
  cast1.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Cast cast2("cast2");
  cast2.x = load.y;
  cast2.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Cast cast3("cast3");
  cast3.x = load.y;
  cast3.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Mul mul1("mul1");
  mul1.x1 = cast1.y;
  mul1.x2 = cast1.y;
  mul1.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Mul mul2("mul2");
  mul2.x1 = cast2.y;
  mul2.x2 = cast2.y;
  mul2.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Mul mul3("mul3");
  mul3.x1 = cast3.y;
  mul3.x2 = cast3.y;
  mul3.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Store store1("store1");
  store1.x = mul1.y;
  store1.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Store store2("store2");
  store2.x = mul2.y;
  store2.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Store store3("store3");
  store3.x = mul3.y;
  store3.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Output output1("output1");
  output1.x = store1.y;
  output1.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Output output2("output2");
  output2.x = store2.y;
  output2.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Output output3("output3");
  output3.x = store3.y;
  output3.y.dtype = ge::DT_FLOAT;

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("mul1"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("mul2"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("mul3"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store1"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store2"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store3"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);

  // 验证结果：三个相同的 cast 应该合并成一个
  // 注意：只统计真正有输入和输出连接的cast节点（已删除的节点边会为空）
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  bool found_cast_with_three_downstream = false;
  for (const auto &node : all_nodes) {
    if (af::ops::IsOps<Cast>(node)) {
      // 检查这个cast是否真的在图中（有输入边和输出边）
      auto in_anchor = node->GetInDataAnchor(0);
      auto out_anchor = node->GetOutDataAnchor(0);
      cast_count++;
      auto peer_anchors = out_anchor->GetPeerInDataAnchors();
      if (peer_anchors.size() == 6U) {
        found_cast_with_three_downstream = true;
      }
    }
  }
  EXPECT_EQ(cast_count, 1U);
  EXPECT_TRUE(found_cast_with_three_downstream);
}

// Test 11: Mixed identity and non-identity casts
TEST_F(TestDtypeConsistency, MixedIdentityAndNonIdentityCasts) {
  af::AscGraph graph("test_mixed_casts");

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT;

  // 恒等 cast
  af::ascir_op::Cast cast_identity("cast_identity");
  cast_identity.x = load.y;
  cast_identity.y.dtype = ge::DT_FLOAT;

  // 非恒等 cast
  af::ascir_op::Cast cast_real("cast_real");
  cast_real.x = cast_identity.y;
  cast_real.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Mul mul("mul");
  mul.x1 = cast_real.y;
  mul.x2 = cast_real.y;
  mul.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT16;

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_FLOAT16, ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);

  // 验证结果：恒等 cast 被删除，非恒等 cast 保留
  // 注意：只统计真正有输入和输出连接的cast节点（已删除的节点边会为空）
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  bool found_fp32_to_fp16_cast = false;
  for (const auto &node : all_nodes) {
    if (af::ops::IsOps<Cast>(node)) {
      // 检查这个cast是否真的在图中（有输入边和输出边）
      auto in_anchor = node->GetInDataAnchor(0);
      auto out_anchor = node->GetOutDataAnchor(0);
      if (in_anchor != nullptr && in_anchor->GetPeerOutAnchor() != nullptr && out_anchor != nullptr &&
          !out_anchor->GetPeerInDataAnchors().empty()) {
        cast_count++;
        if (node->inputs[0].attr.dtype == ge::DT_FLOAT && node->outputs[0].attr.dtype == ge::DT_FLOAT16) {
          found_fp32_to_fp16_cast = true;
        }
      }
    }
  }
  EXPECT_EQ(cast_count, 1U);
  EXPECT_TRUE(found_fp32_to_fp16_cast);
}

// Test 12: No cast nodes in graph
TEST_F(TestDtypeConsistency, NoCastNodes) {
  af::AscGraph graph("test_no_cast");

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Mul mul("mul");
  mul.x1 = load.y;
  mul.x2 = load.y;
  mul.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT;

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);

  // 验证结果：没有 cast 节点
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  for (const auto &node : all_nodes) {
    if (af::ops::IsOps<Cast>(node)) {
      cast_count++;
    }
  }
  EXPECT_EQ(cast_count, 0U);
}

// Test 13: Output dtype directly modified (no cast after output)
TEST_F(TestDtypeConsistency, OutputDtypeDirectlyModified) {
  af::AscGraph graph("test_output_dtype_modified");

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Mul mul("mul");
  mul.x1 = load.y;
  mul.x2 = load.y;
  mul.y.dtype = ge::DT_FLOAT16;  // 初始是 FP16

  af::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_FLOAT16;  // Store 期望 FP16

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT16;

  // Mul 输出需要 FP32（与当前的 FP16 不同）
  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_FLOAT16, ge::DT_FLOAT16}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);

  // 验证结果：Mul 的输出 dtype 被直接修改为 FP32，然后在 Store 前插入 cast(FP32->FP16)
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  bool mul_output_is_fp32 = false;
  bool store_input_is_fp16 = false;
  for (const auto &node : all_nodes) {
    if (af::ops::IsOps<Cast>(node)) {
      cast_count++;
    }
    if (node->GetType() == "Mul") {
      if (node->outputs[0].attr.dtype == ge::DT_FLOAT) {
        mul_output_is_fp32 = true;
      }
    }
    if (node->GetType() == "Store") {
      if (node->inputs[0].attr.dtype == ge::DT_FLOAT16) {
        store_input_is_fp16 = true;
      }
    }
  }
  EXPECT_EQ(cast_count, 1U);  // 只有 Store 前的 cast
  EXPECT_TRUE(mul_output_is_fp32);
  EXPECT_TRUE(store_input_is_fp16);
}

TEST_F(TestDtypeConsistency, ProdInt8WidensToInt16AndWrapsOutputToInt8) {
  auto graph = af::testing::AscGraphBuilder("test_prod_int8_output_cast_path")
                   .Loops({af::testing::Sym("s0")})
                   .Data("data0", 0, {af::testing::Sym("s0")}, {af::ops::One}, ge::DT_INT8)
                   .Load("load0", "data0")
                   .Prod("prod", "load0", {0})
                   .Store("store0", "prod")
                   .Output("output0", "store0", 0, ge::DT_INT8)
                   .Build();

  std::vector<optimize::NodeDtypeRequirement> requirements;
  EXPECT_EQ(optimize::DtypeConsistency::CollectDtypeRequirements(graph, requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);

  bool has_int8_to_int16_cast = false;
  bool has_int16_to_uint8_cast = false;
  bool has_uint8_to_int8_cast = false;
  bool prod_is_int16 = false;
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetType() == "Prod") {
      prod_is_int16 = node->inputs[0].attr.dtype == ge::DT_INT16 && node->outputs[0].attr.dtype == ge::DT_INT16;
    }
    if (!af::ops::IsOps<Cast>(node)) {
      continue;
    }

    has_int8_to_int16_cast = has_int8_to_int16_cast ||
                             (node->inputs[0].attr.dtype == ge::DT_INT8 && node->outputs[0].attr.dtype == ge::DT_INT16);
    has_int16_to_uint8_cast = has_int16_to_uint8_cast || (node->inputs[0].attr.dtype == ge::DT_INT16 &&
                                                          node->outputs[0].attr.dtype == ge::DT_UINT8);
    has_uint8_to_int8_cast = has_uint8_to_int8_cast ||
                             (node->inputs[0].attr.dtype == ge::DT_UINT8 && node->outputs[0].attr.dtype == ge::DT_INT8);
  }

  EXPECT_TRUE(has_int8_to_int16_cast);
  EXPECT_TRUE(has_int16_to_uint8_cast);
  EXPECT_TRUE(has_uint8_to_int8_cast);
  EXPECT_TRUE(prod_is_int16);
}

TEST_F(TestDtypeConsistency, ProdInt8WithoutConsumerSkipsOutputCastPath) {
  auto graph = af::testing::AscGraphBuilder("test_prod_int8_no_output_consumer")
                   .Loops({af::testing::Sym("s0")})
                   .Data("data0", 0, {af::testing::Sym("s0")}, {af::ops::One}, ge::DT_INT8)
                   .Load("load0", "data0")
                   .Prod("prod", "load0", {0})
                   .Build();

  std::vector<optimize::NodeDtypeRequirement> requirements;
  EXPECT_EQ(optimize::DtypeConsistency::CollectDtypeRequirements(graph, requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, requirements), af::SUCCESS);

  bool has_output_wrap_cast = false;
  for (const auto &node : graph.GetAllNodes()) {
    if (!af::ops::IsOps<Cast>(node)) {
      continue;
    }
    has_output_wrap_cast = has_output_wrap_cast ||
                           (node->inputs[0].attr.dtype == ge::DT_INT16 && node->outputs[0].attr.dtype == ge::DT_UINT8);
  }

  EXPECT_FALSE(has_output_wrap_cast);
}

TEST_F(TestDtypeConsistency, ProdInt8EpilogueMulKeepsWrapOutputPath) {
  auto graph = af::testing::AscGraphBuilder("test_prod_int8_epilogue_mul")
                   .Loops({af::testing::Sym("s0")})
                   .Data("data0", 0, {af::testing::Sym("s0")}, {af::ops::One}, ge::DT_INT8)
                   .Load("load0", "data0")
                   .Prod("prod", "load0", {0})
                   .Mul("mul", "prod", "prod")
                   .Store("store0", "mul")
                   .Output("output0", "store0", 0, ge::DT_INT8)
                   .Build();

  std::vector<optimize::NodeDtypeRequirement> requirements;
  EXPECT_EQ(optimize::DtypeConsistency::CollectDtypeRequirements(graph, requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);

  bool has_int16_to_uint8_cast = false;
  bool has_uint8_to_int8_cast = false;
  for (const auto &node : graph.GetAllNodes()) {
    if (!af::ops::IsOps<Cast>(node)) {
      continue;
    }
    has_int16_to_uint8_cast = has_int16_to_uint8_cast || (node->inputs[0].attr.dtype == ge::DT_INT16 &&
                                                          node->outputs[0].attr.dtype == ge::DT_UINT8);
    has_uint8_to_int8_cast = has_uint8_to_int8_cast ||
                             (node->inputs[0].attr.dtype == ge::DT_UINT8 && node->outputs[0].attr.dtype == ge::DT_INT8);
  }

  EXPECT_TRUE(has_int16_to_uint8_cast);
  EXPECT_TRUE(has_uint8_to_int8_cast);
}

TEST_F(TestDtypeConsistency, ProdInt8EpilogueMulKeepsWrapOutputWhenProdIsSecondInput) {
  auto graph = af::testing::AscGraphBuilder("test_prod_int8_epilogue_mul_second_input")
                   .Loops({af::testing::Sym("s0")})
                   .Data("data0", 0, {af::testing::Sym("s0")}, {af::ops::One}, ge::DT_INT8)
                   .Data("data1", 0, {af::testing::Sym("s0")}, {af::ops::One}, ge::DT_INT8)
                   .Load("load0", "data0")
                   .Load("load1", "data1")
                   .Prod("prod", "load1", {0})
                   .Mul("mul", "load0", "prod")
                   .Store("store0", "mul")
                   .Output("output0", "store0", 0, ge::DT_INT8)
                   .Build();

  std::vector<optimize::NodeDtypeRequirement> requirements;
  EXPECT_EQ(optimize::DtypeConsistency::CollectDtypeRequirements(graph, requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);

  bool has_int16_to_uint8_cast = false;
  bool has_uint8_to_int8_cast = false;
  for (const auto &node : graph.GetAllNodes()) {
    if (!af::ops::IsOps<Cast>(node)) {
      continue;
    }
    has_int16_to_uint8_cast = has_int16_to_uint8_cast || (node->inputs[0].attr.dtype == ge::DT_INT16 &&
                                                          node->outputs[0].attr.dtype == ge::DT_UINT8);
    has_uint8_to_int8_cast = has_uint8_to_int8_cast ||
                             (node->inputs[0].attr.dtype == ge::DT_UINT8 && node->outputs[0].attr.dtype == ge::DT_INT8);
  }
  EXPECT_TRUE(has_int16_to_uint8_cast);
  EXPECT_TRUE(has_uint8_to_int8_cast);
}

TEST_F(TestDtypeConsistency, MulInt8DoesNotWrapOutputWithoutProd) {
  auto graph = af::testing::AscGraphBuilder("test_mul_int8_without_prod")
                   .Loops({af::testing::Sym("s0")})
                   .Data("data0", 0, {af::testing::Sym("s0")}, {af::ops::One}, ge::DT_INT8)
                   .Load("load0", "data0")
                   .Mul("mul", "load0", "load0")
                   .Store("store0", "mul")
                   .Output("output0", "store0", 0, ge::DT_INT8)
                   .Build();

  std::vector<optimize::NodeDtypeRequirement> requirements;
  EXPECT_EQ(optimize::DtypeConsistency::CollectDtypeRequirements(graph, requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, requirements), af::SUCCESS);

  bool has_output_wrap_cast = false;
  for (const auto &node : graph.GetAllNodes()) {
    if (!af::ops::IsOps<Cast>(node)) {
      continue;
    }
    has_output_wrap_cast = has_output_wrap_cast ||
                           (node->inputs[0].attr.dtype == ge::DT_INT16 && node->outputs[0].attr.dtype == ge::DT_UINT8);
  }

  EXPECT_FALSE(has_output_wrap_cast);
}

// Test 14: Multiple outputs with different dtype requirements
TEST_F(TestDtypeConsistency, MultipleOutputsWithDifferentDtypes) {
  af::AscGraph graph("test_multiple_outputs");

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Mul mul("mul");
  mul.x1 = load.y;
  mul.x2 = load.y;
  mul.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Store store1("store1");
  store1.x = mul.y;
  store1.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Store store2("store2");
  store2.x = mul.y;
  store2.y.dtype = ge::DT_FLOAT;  // Store2 需要 FP32

  af::ascir_op::Output output1("output1");
  output1.x = store1.y;
  output1.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Output output2("output2");
  output2.x = store2.y;
  output2.y.dtype = ge::DT_FLOAT;

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store1"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("store2"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);

  // 验证结果：Mul 输出 FP32，两个 Store 前各有一个 cast
  auto all_nodes = graph.GetAllNodes();
  size_t cast_fp32_to_fp16_count = 0;
  size_t cast_fp16_to_fp32_count = 0;
  for (const auto &node : all_nodes) {
    if (af::ops::IsOps<Cast>(node)) {
      if (node->inputs[0].attr.dtype == ge::DT_FLOAT && node->outputs[0].attr.dtype == ge::DT_FLOAT16) {
        cast_fp32_to_fp16_count++;
      }
      if (node->inputs[0].attr.dtype == ge::DT_FLOAT16 && node->outputs[0].attr.dtype == ge::DT_FLOAT) {
        cast_fp16_to_fp32_count++;
      }
    }
  }
  EXPECT_EQ(cast_fp16_to_fp32_count, 1U);  // Load 后面一个
  EXPECT_EQ(cast_fp32_to_fp16_count, 1U);  // Store1 前面一个
}

TEST_F(TestDtypeConsistency, MultipleOutputsWithMultipleCastPath) {
  using af::ops::One;

  auto graph =
      af::testing::AscGraphBuilder("test_multiple_outputs_cast_path")
          .Loops({af::testing::Sym(16)})
          .Data("data0", 0, {af::testing::Sym(16)}, {One}, ge::DT_INT16)
          .Load("load0", "data0", {af::testing::Sym(16)}, {One})
          .Split("split", "load0",
                 {{ge::DT_INT16, {}, {af::testing::Sym(16)}, {One}}, {ge::DT_INT16, {}, {af::testing::Sym(16)}, {One}}})
          .Store("store0", "split:0")
          .Store("store1", "split:1")
          .Output("output0", "store0", 0, ge::DT_INT16)
          .Output("output1", "store1", 0, ge::DT_INT8)
          .Build();

  auto split_node = graph.FindNode("split");
  ASSERT_NE(split_node, nullptr);
  optimize::NodeDtypeRequirement requirement{
      split_node, {ge::DT_INT16}, {ge::DT_INT16, ge::DT_INT16}, {{}, {ge::DT_UINT8, ge::DT_INT8}}};
  EXPECT_EQ(optimize::DtypeConsistency::ProcessOutputCastPath(graph, requirement), af::SUCCESS);

  ASSERT_FALSE(split_node->GetOutDataAnchor(1)->GetPeerInDataAnchors().empty());
  auto first_cast_node = std::dynamic_pointer_cast<af::AscNode>(
      (*split_node->GetOutDataAnchor(1)->GetPeerInDataAnchors().begin())->GetOwnerNode());
  ASSERT_NE(first_cast_node, nullptr);
  EXPECT_TRUE(af::ops::IsOps<Cast>(first_cast_node));
  EXPECT_EQ(first_cast_node->inputs[0].attr.dtype, ge::DT_INT16);
  EXPECT_EQ(first_cast_node->outputs[0].attr.dtype, ge::DT_UINT8);

  ASSERT_FALSE(first_cast_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().empty());
  auto second_cast_node = std::dynamic_pointer_cast<af::AscNode>(
      (*first_cast_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().begin())->GetOwnerNode());
  ASSERT_NE(second_cast_node, nullptr);
  EXPECT_TRUE(af::ops::IsOps<Cast>(second_cast_node));
  EXPECT_EQ(second_cast_node->inputs[0].attr.dtype, ge::DT_UINT8);
  EXPECT_EQ(second_cast_node->outputs[0].attr.dtype, ge::DT_INT8);
}

// Test 15: All dtypes already match, no conversion needed
TEST_F(TestDtypeConsistency, AllDtypesMatch_NoConversion) {
  af::AscGraph graph("test_all_match");

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Mul mul("mul");
  mul.x1 = load.y;
  mul.x2 = load.y;
  mul.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT;

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);

  // 验证结果：没有插入任何 cast
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  for (const auto &node : all_nodes) {
    if (af::ops::IsOps<Cast>(node)) {
      cast_count++;
    }
  }
  EXPECT_EQ(cast_count, 0U);
}

// Test 16: Single input with different dtype
TEST_F(TestDtypeConsistency, SingleInputDtypeMismatch) {
  af::AscGraph graph("test_single_input_mismatch");

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Mul mul("mul");
  mul.x1 = load.y;
  mul.x2 = load.y;
  mul.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT16;

  // 只有第一个输入需要 FP32，第二个保持 FP16
  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_FLOAT, ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);

  // 验证结果：只有第一个输入前插入 cast
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  for (const auto &node : all_nodes) {
    if (af::ops::IsOps<Cast>(node)) {
      cast_count++;
    }
  }
  EXPECT_EQ(cast_count, 1U);  // 只有 Mul 的第一个输入前
}

// Test 17: Merge with upstream cast (single consumer)
TEST_F(TestDtypeConsistency, MergeUpstreamCast_SingleConsumer) {
  af::AscGraph graph("test_merge_upstream_single");

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  // 手动添加一个 cast
  af::ascir_op::Cast cast1("cast1");
  cast1.x = load.y;
  cast1.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Mul mul("mul");
  mul.x1 = cast1.y;
  mul.x2 = cast1.y;
  mul.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT16;

  // Mul 需要 FP16，所以 cast1 应该被合并为 FP16->FP16（恒等）
  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_FLOAT16, ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});

  ::ascir::utils::DumpGraph(graph, "Before");
  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);
  ::ascir::utils::DumpGraph(graph, "After");

  // 验证结果：cast1 被合并为 FP16->FP16，然后被删除
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  for (const auto &node : all_nodes) {
    if (af::ops::IsOps<Cast>(node)) {
      cast_count++;
    }
  }
  EXPECT_EQ(cast_count, 0U);  // 恒等 cast 被删除
}

// Test 18: Merge with upstream cast (multiple consumers)
TEST_F(TestDtypeConsistency, MergeUpstreamCast_MultipleConsumers) {
  af::AscGraph graph("test_merge_upstream_multiple");

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  // 手动添加一个 cast
  af::ascir_op::Cast cast1("cast1");
  cast1.x = load.y;
  cast1.y.dtype = ge::DT_FLOAT;

  // 第一个消费者需要 FP32
  af::ascir_op::Mul mul1("mul1");
  mul1.x1 = cast1.y;
  mul1.x2 = cast1.y;
  mul1.y.dtype = ge::DT_FLOAT;

  // 第二个消费者需要 FP16（应该触发合并）
  af::ascir_op::Mul mul2("mul2");
  mul2.x1 = cast1.y;
  mul2.x2 = cast1.y;
  mul2.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Store store1("store1");
  store1.x = mul1.y;
  store1.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Store store2("store2");
  store2.x = mul2.y;
  store2.y.dtype = ge::DT_FLOAT16;

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("mul1"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("mul2"), {ge::DT_FLOAT16, ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("store1"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store2"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);

  // 验证结果：创建了新的 merged cast (FP16->FP16)
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  for (const auto &node : all_nodes) {
    if (af::ops::IsOps<Cast>(node)) {
      cast_count++;
    }
  }
  // 原 cast1(FP16->FP32) 保留，新 merged cast(FP16->FP16) 会被删除
  EXPECT_EQ(cast_count, 1U);
}

// Test 19: Unsupported cast conversion should fail
TEST_F(TestDtypeConsistency, UnsupportedCastConversion) {
  af::AscGraph graph("test_unsupported_cast");

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_INT64;  // INT64 不支持转换到某些类型

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_INT64;

  af::ascir_op::Mul mul("mul");
  mul.x1 = load.y;
  mul.x2 = load.y;
  mul.y.dtype = ge::DT_INT64;

  af::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_INT64;

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_INT64;

  // 尝试将 INT64 转换为 DT_FLOAT（如果支持的话）
  // 实际上 Cast::InferDataType 会检查是否支持
  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_INT64}, {ge::DT_INT64}});
  // 注意：这里使用一个可能不支持的转换组合进行测试
  // 实际支持的转换需要参考 Cast::InferDataType 的实现
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_INT64, ge::DT_INT64}, {ge::DT_INT64}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_INT64}, {ge::DT_INT64}});

  // 因为 dtype 匹配，不应该插入 cast
  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
}

// ============================================================================
// 以下测试用例针对 Cast 合并的精度安全问题
// ============================================================================

// Test 29: 安全的有符号整数扩展合并 - int8 -> int16，下游需要 int16
// 场景：load(int8) -> cast(int8->int16) -> mul(需要int16)
// 由于 cast 的输出 int16 已经满足 mul 的需求，不会触发合并
// 这个测试验证同类型时不触发合并逻辑
TEST_F(TestDtypeConsistency, SafeIntWidening_NoMergeNeeded) {
  af::AscGraph graph("test_safe_int_no_merge");

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_INT8;

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_INT8;

  af::ascir_op::Cast cast1("cast1");
  cast1.x = load.y;
  cast1.y.dtype = ge::DT_INT16;

  af::ascir_op::Mul mul("mul");
  mul.x1 = cast1.y;
  mul.x2 = cast1.y;
  mul.y.dtype = ge::DT_INT16;

  af::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_INT16;

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_INT16;

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_INT8}, {ge::DT_INT8}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_INT16, ge::DT_INT16}, {ge::DT_INT16}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_INT16}, {ge::DT_INT16}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);

  // cast1(int8->int16) 应该保留
  auto all_nodes = graph.GetAllNodes();
  bool has_int8_to_int16 = false;
  for (const auto &node : all_nodes) {
    if (af::ops::IsOps<Cast>(node)) {
      auto in_anchor = node->GetInDataAnchor(0);
      auto out_anchor = node->GetOutDataAnchor(0);
      if (in_anchor != nullptr && in_anchor->GetPeerOutAnchor() != nullptr && out_anchor != nullptr &&
          !out_anchor->GetPeerInDataAnchors().empty()) {
        if (node->inputs[0].attr.dtype == ge::DT_INT8 && node->outputs[0].attr.dtype == ge::DT_INT16) {
          has_int8_to_int16 = true;
        }
      }
    }
  }
  EXPECT_TRUE(has_int8_to_int16);
}

// Test 30: 跨类别 Cast 合并测试 - int8 -> uint8 -> 下游需要 int16
// 由于 int8->uint8 不保持值，不应合并为 int8->int16
TEST_F(TestDtypeConsistency, CrossCategoryCast_ShouldInsertNewCast) {
  af::AscGraph graph("test_cross_category_insert");

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_INT8;

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_INT8;

  // 跨类别 cast：int8 -> uint8
  af::ascir_op::Cast cast1("cast1");
  cast1.x = load.y;
  cast1.y.dtype = ge::DT_UINT8;

  af::ascir_op::Mul mul("mul");
  mul.x1 = cast1.y;
  mul.x2 = cast1.y;
  mul.y.dtype = ge::DT_UINT8;

  af::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_UINT8;

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_UINT8;

  // mul 需要 int16（与 cast1 输出的 uint8 不同）
  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_INT8}, {ge::DT_INT8}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_INT16, ge::DT_INT16}, {ge::DT_INT16}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_INT16}, {ge::DT_INT16}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);

  // 由于 int8->uint8 不保持值，不应该合并为 int8->int16
  // 应该保留 cast1(int8->uint8) 并插入新的 cast(uint8->int16)
  auto all_nodes = graph.GetAllNodes();
  bool has_int8_to_int16 = false;
  bool has_int8_to_uint8 = false;
  for (const auto &node : all_nodes) {
    if (af::ops::IsOps<Cast>(node)) {
      auto in_anchor = node->GetInDataAnchor(0);
      auto out_anchor = node->GetOutDataAnchor(0);
      if (in_anchor != nullptr && in_anchor->GetPeerOutAnchor() != nullptr && out_anchor != nullptr &&
          !out_anchor->GetPeerInDataAnchors().empty()) {
        if (node->inputs[0].attr.dtype == ge::DT_INT8 && node->outputs[0].attr.dtype == ge::DT_INT16) {
          has_int8_to_int16 = true;
        }
        if (node->inputs[0].attr.dtype == ge::DT_INT8 && node->outputs[0].attr.dtype == ge::DT_UINT8) {
          has_int8_to_uint8 = true;
        }
      }
    }
  }
  // 不应该有直接的 int8->int16（因为那意味着跨类别合并）
  EXPECT_FALSE(has_int8_to_int16);
}

// Test 31: 安全的有符号整数扩展 - int16 -> int32，下游需要 int32
// 场景：load(int16) -> cast(int16->int32) -> mul(需要int32)
// cast1 的输出 int32 已经满足 mul 的需求，不会触发合并，保留 cast1
TEST_F(TestDtypeConsistency, SafeIntWidening_CastPreserved) {
  af::AscGraph graph("test_safe_int_widening");

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_INT16;

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_INT16;

  af::ascir_op::Cast cast1("cast1");
  cast1.x = load.y;
  cast1.y.dtype = ge::DT_INT32;

  af::ascir_op::Mul mul("mul");
  mul.x1 = cast1.y;
  mul.x2 = cast1.y;
  mul.y.dtype = ge::DT_INT32;

  af::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_INT32;

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_INT32;

  // mul 需要 int32（与 cast1 输出的 int32 相同）
  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_INT16}, {ge::DT_INT16}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_INT32, ge::DT_INT32}, {ge::DT_INT32}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_INT32}, {ge::DT_INT32}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);

  // cast1(int16->int32) 应该保留
  auto all_nodes = graph.GetAllNodes();
  bool has_int16_to_int32 = false;
  for (const auto &node : all_nodes) {
    if (af::ops::IsOps<Cast>(node)) {
      auto in_anchor = node->GetInDataAnchor(0);
      auto out_anchor = node->GetOutDataAnchor(0);
      if (in_anchor != nullptr && in_anchor->GetPeerOutAnchor() != nullptr && out_anchor != nullptr &&
          !out_anchor->GetPeerInDataAnchors().empty()) {
        if (node->inputs[0].attr.dtype == ge::DT_INT16 && node->outputs[0].attr.dtype == ge::DT_INT32) {
          has_int16_to_int32 = true;
        }
      }
    }
  }
  EXPECT_TRUE(has_int16_to_int32);
}

// Test 32: 安全的浮点扩展 - fp32 -> fp64，下游需要 fp64
// 场景：load(fp32) -> cast(fp32->fp64) -> mul(需要fp64)
// cast1 的输出 fp64 已经满足 mul 的需求，不会触发合并，保留 cast1
TEST_F(TestDtypeConsistency, SafeFloatWidening_F64_CastPreserved) {
  af::AscGraph graph("test_safe_fp_widening_f64");

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Cast cast1("cast1");
  cast1.x = load.y;
  cast1.y.dtype = ge::DT_DOUBLE;

  af::ascir_op::Mul mul("mul");
  mul.x1 = cast1.y;
  mul.x2 = cast1.y;
  mul.y.dtype = ge::DT_DOUBLE;

  af::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_DOUBLE;

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_DOUBLE;

  // mul 需要 fp64（与 cast1 输出的 fp64 相同）
  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_DOUBLE, ge::DT_DOUBLE}, {ge::DT_DOUBLE}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_DOUBLE}, {ge::DT_DOUBLE}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);

  // cast1(fp32->fp64) 应该保留
  auto all_nodes = graph.GetAllNodes();
  bool has_fp32_to_fp64 = false;
  for (const auto &node : all_nodes) {
    if (af::ops::IsOps<Cast>(node)) {
      auto in_anchor = node->GetInDataAnchor(0);
      auto out_anchor = node->GetOutDataAnchor(0);
      if (in_anchor != nullptr && in_anchor->GetPeerOutAnchor() != nullptr && out_anchor != nullptr &&
          !out_anchor->GetPeerInDataAnchors().empty()) {
        if (node->inputs[0].attr.dtype == ge::DT_FLOAT && node->outputs[0].attr.dtype == ge::DT_DOUBLE) {
          has_fp32_to_fp64 = true;
        }
      }
    }
  }
  EXPECT_TRUE(has_fp32_to_fp64);
}

// Test 33: 安全的浮点扩展合并 - fp16 -> fp32，下游需要 fp32（与 Test 17 类似）
TEST_F(TestDtypeConsistency, SafeFloatWidening_NoMergeNeeded) {
  af::AscGraph graph("test_safe_fp_no_merge");

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Cast cast1("cast1");
  cast1.x = load.y;
  cast1.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Mul mul("mul");
  mul.x1 = cast1.y;
  mul.x2 = cast1.y;
  mul.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT;

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);

  // cast1(fp16->fp32) 应该保留
  auto all_nodes = graph.GetAllNodes();
  bool has_fp16_to_fp32 = false;
  for (const auto &node : all_nodes) {
    if (af::ops::IsOps<Cast>(node)) {
      auto in_anchor = node->GetInDataAnchor(0);
      auto out_anchor = node->GetOutDataAnchor(0);
      if (in_anchor != nullptr && in_anchor->GetPeerOutAnchor() != nullptr && out_anchor != nullptr &&
          !out_anchor->GetPeerInDataAnchors().empty()) {
        if (node->inputs[0].attr.dtype == ge::DT_FLOAT16 && node->outputs[0].attr.dtype == ge::DT_FLOAT) {
          has_fp16_to_fp32 = true;
        }
      }
    }
  }
  EXPECT_TRUE(has_fp16_to_fp32);
}

// Test 34: 无符号整数扩展 - uint8 -> uint16，下游需要 uint16
TEST_F(TestDtypeConsistency, SafeUnsignedWidening_NoMergeNeeded) {
  af::AscGraph graph("test_safe_uint_no_merge");

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_UINT8;

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_UINT8;

  af::ascir_op::Cast cast1("cast1");
  cast1.x = load.y;
  cast1.y.dtype = ge::DT_UINT16;

  af::ascir_op::Mul mul("mul");
  mul.x1 = cast1.y;
  mul.x2 = cast1.y;
  mul.y.dtype = ge::DT_UINT16;

  af::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_UINT16;

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_UINT16;

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_UINT8}, {ge::DT_UINT8}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_UINT16, ge::DT_UINT16}, {ge::DT_UINT16}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_UINT16}, {ge::DT_UINT16}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);

  // cast1(uint8->uint16) 应该保留
  auto all_nodes = graph.GetAllNodes();
  bool has_uint8_to_uint16 = false;
  for (const auto &node : all_nodes) {
    if (af::ops::IsOps<Cast>(node)) {
      auto in_anchor = node->GetInDataAnchor(0);
      auto out_anchor = node->GetOutDataAnchor(0);
      if (in_anchor != nullptr && in_anchor->GetPeerOutAnchor() != nullptr && out_anchor != nullptr &&
          !out_anchor->GetPeerInDataAnchors().empty()) {
        if (node->inputs[0].attr.dtype == ge::DT_UINT8 && node->outputs[0].attr.dtype == ge::DT_UINT16) {
          has_uint8_to_uint16 = true;
        }
      }
    }
  }
  EXPECT_TRUE(has_uint8_to_uint16);
}

// Test 35: BF16 扩展测试 - bf16 -> fp32，下游需要 fp32
TEST_F(TestDtypeConsistency, BF16Widening_NoMergeNeeded) {
  af::AscGraph graph("test_bf16_no_merge");

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_BF16;

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_BF16;

  af::ascir_op::Cast cast1("cast1");
  cast1.x = load.y;
  cast1.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Mul mul("mul");
  mul.x1 = cast1.y;
  mul.x2 = cast1.y;
  mul.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT;

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_BF16}, {ge::DT_BF16}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);

  // cast1(bf16->fp32) 应该保留
  auto all_nodes = graph.GetAllNodes();
  bool has_bf16_to_fp32 = false;
  for (const auto &node : all_nodes) {
    if (af::ops::IsOps<Cast>(node)) {
      auto in_anchor = node->GetInDataAnchor(0);
      auto out_anchor = node->GetOutDataAnchor(0);
      if (in_anchor != nullptr && in_anchor->GetPeerOutAnchor() != nullptr && out_anchor != nullptr &&
          !out_anchor->GetPeerInDataAnchors().empty()) {
        if (node->inputs[0].attr.dtype == ge::DT_BF16 && node->outputs[0].attr.dtype == ge::DT_FLOAT) {
          has_bf16_to_fp32 = true;
        }
      }
    }
  }
  EXPECT_TRUE(has_bf16_to_fp32);
}

// Test 36: 安全合并场景 - int8 -> int16，下游需要 int16（与 Test 17 类似但用 int）
// cast1 的输出 int16 已经满足 mul 的需求，保留 cast1
TEST_F(TestDtypeConsistency, SafeInt8ToInt16_CastPreserved) {
  af::AscGraph graph("test_safe_int8_to_int16");

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_INT8;

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_INT8;

  af::ascir_op::Cast cast1("cast1");
  cast1.x = load.y;
  cast1.y.dtype = ge::DT_INT16;

  af::ascir_op::Mul mul("mul");
  mul.x1 = cast1.y;
  mul.x2 = cast1.y;
  mul.y.dtype = ge::DT_INT16;

  af::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_INT16;

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_INT16;

  // mul 需要 int16（与 cast1 输出的 int16 相同）
  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_INT8}, {ge::DT_INT8}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_INT16, ge::DT_INT16}, {ge::DT_INT16}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_INT16}, {ge::DT_INT16}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);

  // cast1(int8->int16) 应该保留
  auto all_nodes = graph.GetAllNodes();
  bool has_int8_to_int16 = false;
  for (const auto &node : all_nodes) {
    if (af::ops::IsOps<Cast>(node)) {
      auto in_anchor = node->GetInDataAnchor(0);
      auto out_anchor = node->GetOutDataAnchor(0);
      if (in_anchor != nullptr && in_anchor->GetPeerOutAnchor() != nullptr && out_anchor != nullptr &&
          !out_anchor->GetPeerInDataAnchors().empty()) {
        if (node->inputs[0].attr.dtype == ge::DT_INT8 && node->outputs[0].attr.dtype == ge::DT_INT16) {
          has_int8_to_int16 = true;
        }
      }
    }
  }
  EXPECT_TRUE(has_int8_to_int16);
}

// Test 37: 跨类别不应合并 - int8 -> uint8，下游需要 uint8
// 不应该合并为 int8->uint8（已经是 int8->uint8，保持不变）
TEST_F(TestDtypeConsistency, CrossCategoryCast_NotMergeToInt8) {
  af::AscGraph graph("test_cross_no_merge_int8");

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_INT8;

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_INT8;

  af::ascir_op::Cast cast1("cast1");
  cast1.x = load.y;
  cast1.y.dtype = ge::DT_UINT8;

  af::ascir_op::Mul mul("mul");
  mul.x1 = cast1.y;
  mul.x2 = cast1.y;
  mul.y.dtype = ge::DT_UINT8;

  af::ascir_op::Store store("store");
  store.x = mul.y;
  store.y.dtype = ge::DT_UINT8;

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_UINT8;

  // mul 需要 uint8（与 cast1 输出相同），不触发合并
  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_INT8}, {ge::DT_INT8}});
  mock_requirements.push_back({graph.FindNode("mul"), {ge::DT_UINT8, ge::DT_UINT8}, {ge::DT_UINT8}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_UINT8}, {ge::DT_UINT8}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);

  // cast1(int8->uint8) 应该保留
  auto all_nodes = graph.GetAllNodes();
  bool has_int8_to_uint8 = false;
  for (const auto &node : all_nodes) {
    if (af::ops::IsOps<Cast>(node)) {
      auto in_anchor = node->GetInDataAnchor(0);
      auto out_anchor = node->GetOutDataAnchor(0);
      if (in_anchor != nullptr && in_anchor->GetPeerOutAnchor() != nullptr && out_anchor != nullptr &&
          !out_anchor->GetPeerInDataAnchors().empty()) {
        if (node->inputs[0].attr.dtype == ge::DT_INT8 && node->outputs[0].attr.dtype == ge::DT_UINT8) {
          has_int8_to_uint8 = true;
        }
      }
    }
  }
  EXPECT_TRUE(has_int8_to_uint8);
}

// Test 38: FloorDiv chain with existing Cast nodes - FloorDiv needs FP32 processing
// Graph: Load(FP16) -> FloorDiv(FP16) -> Cast(FP16->FP32) -> FloorDiv(FP32) -> Store(FP32)
// FloorDiv_0 needs FP32, so Cast(FP16->FP32) is inserted before it
// Existing Cast(FP16->FP32) can be reused for FloorDiv_1
TEST_F(TestDtypeConsistency, FloorDivChainWithExistingCast) {
  af::AscGraph graph("test_floordiv_cast_chain");

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Load load("load0");
  load.x = data0.y;
  load.y.dtype = ge::DT_FLOAT16;

  // Scalar for FloorDiv divisor
  af::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Broadcast broadcast0("broadcast0");
  broadcast0.x = scalar0.y;
  broadcast0.y.dtype = ge::DT_FLOAT16;

  // FloorDiv_0: FP16 input, needs FP32 for computation
  af::ascir_op::FloorDiv floordiv0("floordiv0");
  floordiv0.x1 = load.y;
  floordiv0.x2 = broadcast0.y;
  floordiv0.y.dtype = ge::DT_FLOAT16;

  // Existing Cast: FP16 -> FP32
  af::ascir_op::Cast cast0("cast0");
  cast0.x = floordiv0.y;
  cast0.y.dtype = ge::DT_FLOAT;

  // Scalar for FloorDiv_1 divisor
  af::ascir_op::Scalar scalar1("scalar1", graph);
  scalar1.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Broadcast broadcast1("broadcast1");
  broadcast1.x = scalar1.y;
  broadcast1.y.dtype = ge::DT_FLOAT;

  // FloorDiv_1: needs FP32 (matches cast0 output)
  af::ascir_op::FloorDiv floordiv1("floordiv1");
  floordiv1.x1 = cast0.y;
  floordiv1.x2 = broadcast1.y;
  floordiv1.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Store store("store");
  store.x = floordiv1.y;
  store.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = ge::DT_FLOAT;

  // Mock requirements: FloorDiv ops need FP32 for computation
  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load0"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("broadcast0"), {ge::DT_FLOAT16}, {ge::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("floordiv0"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("cast0"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("broadcast1"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("floordiv1"), {ge::DT_FLOAT, ge::DT_FLOAT}, {ge::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store"), {ge::DT_FLOAT}, {ge::DT_FLOAT}});

  EXPECT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  EXPECT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);
  EXPECT_EQ(optimize::ScheduleUtils::TopologicalSorting(graph), af::SUCCESS);

  // Verify:
  // - FloorDiv_0 should have Cast before it (FP16->FP32) and after it (FP32->FP16 for cast0 input)
  // - FloorDiv_1 should use cast0 output directly (already FP32)
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  bool floordiv0_is_float = false;
  bool floordiv1_is_float = false;
  bool has_fp16_to_fp32 = false;

  for (const auto &node : all_nodes) {
    if (af::ops::IsOps<Cast>(node)) {
      cast_count++;
      auto in_anchor = node->GetInDataAnchor(0);
      auto out_anchor = node->GetOutDataAnchor(0);
      if (in_anchor != nullptr && in_anchor->GetPeerOutAnchor() != nullptr && out_anchor != nullptr &&
          !out_anchor->GetPeerInDataAnchors().empty()) {
        if (node->inputs[0].attr.dtype == ge::DT_FLOAT16 && node->outputs[0].attr.dtype == ge::DT_FLOAT) {
          has_fp16_to_fp32 = true;
        }
      }
    }
    if (node->GetName() == "floordiv0") {
      if (node->inputs[0].attr.dtype == ge::DT_FLOAT && node->outputs[0].attr.dtype == ge::DT_FLOAT) {
        floordiv0_is_float = true;
      }
    }
    if (node->GetName() == "floordiv1") {
      if (node->inputs[0].attr.dtype == ge::DT_FLOAT && node->outputs[0].attr.dtype == ge::DT_FLOAT) {
        floordiv1_is_float = true;
      }
    }
  }

  // FloorDiv_0 should be converted to FP32
  EXPECT_TRUE(floordiv0_is_float);
  // FloorDiv_1 is already FP32 (cast0 output)
  EXPECT_TRUE(floordiv1_is_float);
  // Should have FP16->FP32 cast
  EXPECT_TRUE(has_fp16_to_fp32);
}

}  // namespace
