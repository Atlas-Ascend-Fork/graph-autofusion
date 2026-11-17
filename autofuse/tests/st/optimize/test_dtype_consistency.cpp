/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "gtest/gtest.h"

#include "ascir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "schedule_utils.h"
#include "../../common/att/soc_perf_test_config.h"
#define private public
#include "ascir_utils.h"
#include "optimize/graph_completeness/dtype_consistency.h"
#undef private

#include "asc_graph_builder.h"

using namespace af;
using namespace af::ascir_op;
using af::ops::IsOps;
using af::ops::One;
using af::testing::AscGraphBuilder;
using af::testing::Sym;

namespace {

class TestDtypeConsistencyST : public ::testing::Test {
 protected:
  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
    ASSERT_EQ(
        att::soc_perf_test::SetSocPerfTestPlatform(att::soc_perf_test::kV35NpuArch, att::soc_perf_test::kSocEmpty),
        af::SUCCESS);
  }

  void TearDown() override {
    ge::PlatformContext::GetInstance().Reset();
  }
};

// ST Test 1: Complex FP16 -> FP32 -> FP16 conversion chain with multiple operators
TEST_F(TestDtypeConsistencyST, ComplexFp16ToFp32Chain) {
  // Create nodes: Data -> Load -> Add -> Mul -> Store -> Output
  auto graph = AscGraphBuilder("test_complex_fp16_fp32_chain")
                   .Loops({Sym("s0")})
                   .Data("x1", 0, {Sym("s0")}, {One}, af::DT_FLOAT16)
                   .Load("load", "x1")
                   .Add("add", "load", "load")
                   .Mul("mul", "add", "add")
                   .Store("store", "mul")
                   .Output("y", "store", 0, af::DT_FLOAT16)
                   .Build();

  // Mock requirements: Add and Mul need FP32 for computation
  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load"), {af::DT_FLOAT16}, {af::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("add"), {af::DT_FLOAT, af::DT_FLOAT}, {af::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("mul"), {af::DT_FLOAT, af::DT_FLOAT}, {af::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store"), {af::DT_FLOAT16}, {af::DT_FLOAT16}});

  ASSERT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  ASSERT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);
  ASSERT_EQ(optimize::ScheduleUtils::TopologicalSorting(graph), af::SUCCESS);

  // Verify the result:
  // Expected: Data(FP16) -> Load(FP16) -> Cast(FP16->FP32) -> Add(FP32) -> Mul(FP32) -> Cast(FP32->FP16) -> Store(FP16)
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  bool add_is_fp32 = false;
  bool mul_is_fp32 = false;

  for (const auto &node : all_nodes) {
    if (IsOps<Cast>(node)) {
      cast_count++;
    }
    if (node->GetType() == "Add") {
      if (node->inputs[0].attr.dtype == af::DT_FLOAT && node->outputs[0].attr.dtype == af::DT_FLOAT) {
        add_is_fp32 = true;
      }
    }
    if (node->GetType() == "Mul") {
      if (node->inputs[0].attr.dtype == af::DT_FLOAT && node->outputs[0].attr.dtype == af::DT_FLOAT) {
        mul_is_fp32 = true;
      }
    }
  }

  // Should have 2 Cast nodes: one before Add (FP16->FP32), one after Mul (FP32->FP16)
  EXPECT_EQ(cast_count, 2U);
  EXPECT_TRUE(add_is_fp32);
  EXPECT_TRUE(mul_is_fp32);
}

TEST_F(TestDtypeConsistencyST, ProdInt8WidensToInt16AndWrapsOutputToInt8) {
  auto graph = AscGraphBuilder("test_prod_int8_dtype")
                   .Loops({Sym("s0")})
                   .Data("data1", 0, {Sym("s0")}, {One}, af::DT_INT8)
                   .Load("load", "data1")
                   .Prod("prod", "load", {0})
                   .Store("store", "prod")
                   .Build();

  std::vector<optimize::NodeDtypeRequirement> requirements;
  ASSERT_EQ(optimize::DtypeConsistency::CollectDtypeRequirements(graph, requirements), af::SUCCESS);
  ASSERT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, requirements), af::SUCCESS);
  ASSERT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);
  ASSERT_EQ(optimize::ScheduleUtils::TopologicalSorting(graph), af::SUCCESS);

  size_t cast_count = 0;
  bool has_int8_to_int16_cast = false;
  bool has_int16_to_uint8_cast = false;
  bool has_uint8_to_int8_cast = false;
  bool prod_is_int16 = false;
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetType() == "Prod") {
      prod_is_int16 = node->inputs[0].attr.dtype == af::DT_INT16 && node->outputs[0].attr.dtype == af::DT_INT16;
    }
    if (!IsOps<Cast>(node)) {
      continue;
    }

    ++cast_count;
    has_int8_to_int16_cast = has_int8_to_int16_cast ||
                             (node->inputs[0].attr.dtype == af::DT_INT8 && node->outputs[0].attr.dtype == af::DT_INT16);
    has_int16_to_uint8_cast = has_int16_to_uint8_cast || (node->inputs[0].attr.dtype == af::DT_INT16 &&
                                                          node->outputs[0].attr.dtype == af::DT_UINT8);
    has_uint8_to_int8_cast = has_uint8_to_int8_cast ||
                             (node->inputs[0].attr.dtype == af::DT_UINT8 && node->outputs[0].attr.dtype == af::DT_INT8);
  }

  EXPECT_EQ(cast_count, 3U);
  EXPECT_TRUE(has_int8_to_int16_cast);
  EXPECT_TRUE(has_int16_to_uint8_cast);
  EXPECT_TRUE(has_uint8_to_int8_cast);
  EXPECT_TRUE(prod_is_int16);
}

TEST_F(TestDtypeConsistencyST, ProdInt8WithoutConsumerDoesNotInsertOutputCast) {
  auto graph = AscGraphBuilder("test_prod_int8_no_consumer")
                   .Loops({Sym("s0")})
                   .Data("data1", 0, {Sym("s0")}, {One}, af::DT_INT8)
                   .Load("load", "data1")
                   .Prod("prod", "load", {0})
                   .Build();

  std::vector<optimize::NodeDtypeRequirement> requirements;
  ASSERT_EQ(optimize::DtypeConsistency::CollectDtypeRequirements(graph, requirements), af::SUCCESS);
  ASSERT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, requirements), af::SUCCESS);

  bool has_int16_to_uint8_cast = false;
  bool has_uint8_to_int8_cast = false;
  for (const auto &node : graph.GetAllNodes()) {
    if (!IsOps<Cast>(node)) {
      continue;
    }
    has_int16_to_uint8_cast = has_int16_to_uint8_cast || (node->inputs[0].attr.dtype == af::DT_INT16 &&
                                                          node->outputs[0].attr.dtype == af::DT_UINT8);
    has_uint8_to_int8_cast = has_uint8_to_int8_cast ||
                             (node->inputs[0].attr.dtype == af::DT_UINT8 && node->outputs[0].attr.dtype == af::DT_INT8);
  }

  EXPECT_FALSE(has_int16_to_uint8_cast);
  EXPECT_FALSE(has_uint8_to_int8_cast);
}

// ST Test 2: Multi-input with different dtypes, all need FP32
TEST_F(TestDtypeConsistencyST, MultiInputDifferentDtypes) {
  // Create two data inputs: one FP16, one FP32
  auto graph = AscGraphBuilder("test_multi_input_diff_dtypes")
                   .Loops({Sym("s0")})
                   .Data("data1", 0, {Sym("s0")}, {One}, af::DT_FLOAT16)
                   .Data("data2", 1, {Sym("s0")}, {One}, af::DT_FLOAT16)
                   .Load("load1", "data1")
                   .Load("load2", "data2")
                   .Add("add", "load1", "load2")
                   .Store("store", "add")
                   .Build();

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load1"), {af::DT_FLOAT16}, {af::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("load2"), {af::DT_FLOAT16}, {af::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("add"), {af::DT_FLOAT, af::DT_FLOAT}, {af::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store"), {af::DT_FLOAT16}, {af::DT_FLOAT16}});

  ASSERT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  ASSERT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);
  ASSERT_EQ(optimize::ScheduleUtils::TopologicalSorting(graph), af::SUCCESS);

  // Verify both inputs are cast to FP32
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  bool add_is_fp32 = false;

  for (const auto &node : all_nodes) {
    if (IsOps<Cast>(node)) {
      cast_count++;
    }
    if (node->GetType() == "Add") {
      if (node->inputs[0].attr.dtype == af::DT_FLOAT && node->inputs[1].attr.dtype == af::DT_FLOAT &&
          node->outputs[0].attr.dtype == af::DT_FLOAT) {
        add_is_fp32 = true;
      }
    }
  }

  EXPECT_EQ(cast_count, 3U);  // Two cast nodes before add, one after add
  EXPECT_TRUE(add_is_fp32);
}

// ST Test 3: Merge with upstream cast optimization
TEST_F(TestDtypeConsistencyST, MergeWithUpstreamCast) {
  auto graph = AscGraphBuilder("test_merge_upstream_cast")
                   .Loops({Sym("s0")})
                   .Data("data1", 0, {Sym("s0")}, {One}, af::DT_FLOAT16)
                   .Load("load", "data1")
                   .Cast("existing_cast", "load", af::DT_FLOAT)
                   .Add("add", "existing_cast", "existing_cast")
                   .Store("store", "add")
                   .Build();

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load"), {af::DT_FLOAT16}, {af::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("add"), {af::DT_FLOAT, af::DT_FLOAT}, {af::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store"), {af::DT_FLOAT16}, {af::DT_FLOAT16}});

  ASSERT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  ASSERT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);
  ASSERT_EQ(optimize::ScheduleUtils::TopologicalSorting(graph), af::SUCCESS);

  // Verify upstream cast is reused, no new cast inserted before add
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  bool add_is_fp32 = false;

  for (const auto &node : all_nodes) {
    if (IsOps<Cast>(node)) {
      cast_count++;
    }
    if (node->GetType() == "Add") {
      if (node->inputs[0].attr.dtype == af::DT_FLOAT && node->outputs[0].attr.dtype == af::DT_FLOAT) {
        add_is_fp32 = true;
      }
    }
  }

  // Should have 2 Cast: existing upstream one and one after add (FP32->FP16)
  EXPECT_EQ(cast_count, 2U);
  EXPECT_TRUE(add_is_fp32);
}

// ST Test 4: Cast CSE - multiple identical casts from same upstream
TEST_F(TestDtypeConsistencyST, CastCSEComplex) {
  // Multiple operators need FP32 from same FP16 source
  auto graph = AscGraphBuilder("test_cast_cse_complex")
                   .Loops({Sym("s0")})
                   .Data("data1", 0, {Sym("s0")}, {One}, af::DT_FLOAT16)
                   .Load("load", "data1")
                   .Add("add", "load", "load")
                   .Mul("mul", "load", "load")
                   .Sub("sub", "load", "load")
                   .Build();

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load"), {af::DT_FLOAT16}, {af::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("add"), {af::DT_FLOAT, af::DT_FLOAT}, {af::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("mul"), {af::DT_FLOAT, af::DT_FLOAT}, {af::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("sub"), {af::DT_FLOAT, af::DT_FLOAT}, {af::DT_FLOAT}});

  ASSERT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  ASSERT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);
  ASSERT_EQ(optimize::ScheduleUtils::TopologicalSorting(graph), af::SUCCESS);

  // Verify CSE merged identical casts
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  size_t cast_with_multiple_downstream = 0;

  for (const auto &node : all_nodes) {
    if (IsOps<Cast>(node)) {
      cast_count++;
      auto out_anchor = node->GetOutDataAnchor(0);
      if (out_anchor != nullptr && out_anchor->GetPeerInDataAnchors().size() >= 6) {
        cast_with_multiple_downstream++;
      }
    }
  }

  // Should have only 1 Cast node (merged) serving all 3 operators
  EXPECT_EQ(cast_count, 1U);
  EXPECT_EQ(cast_with_multiple_downstream, 1U);
}

// ST Test 5: Cancel identity cast (Cast FP32->FP32)
TEST_F(TestDtypeConsistencyST, CancelIdentityCastComplex) {
  // Add identity cast (FP32 -> FP32)
  auto graph = AscGraphBuilder("test_identity_cast_complex")
                   .Loops({Sym("s0")})
                   .Data("data1", 0, {Sym("s0")}, {One}, af::DT_FLOAT)
                   .Load("load", "data1")
                   .Cast("identity_cast", "load", af::DT_FLOAT)
                   .Add("add", "identity_cast", "identity_cast")
                   .Store("store", "add")
                   .Build();

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load"), {af::DT_FLOAT}, {af::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("add"), {af::DT_FLOAT, af::DT_FLOAT}, {af::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store"), {af::DT_FLOAT}, {af::DT_FLOAT}});

  ASSERT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  ASSERT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);
  ASSERT_EQ(optimize::ScheduleUtils::TopologicalSorting(graph), af::SUCCESS);

  // Verify identity cast is removed
  auto all_nodes = graph.GetAllNodes();
  size_t active_cast_count = 0;

  for (const auto &node : all_nodes) {
    if (IsOps<Cast>(node)) {
      auto in_anchor = node->GetInDataAnchor(0);
      auto out_anchor = node->GetOutDataAnchor(0);
      if (in_anchor != nullptr && in_anchor->GetPeerOutAnchor() != nullptr && out_anchor != nullptr &&
          !out_anchor->GetPeerInDataAnchors().empty()) {
        active_cast_count++;
      }
    }
  }

  EXPECT_EQ(active_cast_count, 0U);
}

// ST Test 6: Multiple data type conversions in a deep chain
TEST_F(TestDtypeConsistencyST, DeepChainWithMultipleConversions) {
  // Chain: Add -> Mul -> Add -> Mul, all need FP32
  auto graph = AscGraphBuilder("test_deep_chain")
                   .Loops({Sym("s0")})
                   .Data("data1", 0, {Sym("s0")}, {One}, af::DT_FLOAT16)
                   .Load("load", "data1")
                   .Add("add1", "load", "load")
                   .Mul("mul1", "add1", "add1")
                   .Add("add2", "mul1", "mul1")
                   .Mul("mul2", "add2", "add2")
                   .Store("store", "mul2")
                   .Build();

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load"), {af::DT_FLOAT16}, {af::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("add1"), {af::DT_FLOAT, af::DT_FLOAT}, {af::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("mul1"), {af::DT_FLOAT, af::DT_FLOAT}, {af::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("add2"), {af::DT_FLOAT, af::DT_FLOAT}, {af::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("mul2"), {af::DT_FLOAT, af::DT_FLOAT}, {af::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store"), {af::DT_FLOAT16}, {af::DT_FLOAT16}});

  ASSERT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  ASSERT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);
  ASSERT_EQ(optimize::ScheduleUtils::TopologicalSorting(graph), af::SUCCESS);

  // Verify only 2 casts: one at beginning, one at end
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;

  for (const auto &node : all_nodes) {
    if (IsOps<Cast>(node)) {
      cast_count++;
    }
  }

  EXPECT_EQ(cast_count, 2U);
}

// ST Test 8: TryMergeWithUpstreamCast - multiple consumers case (create new merged cast)
// Safe scenario: FP16 -> FP32 (widening), multiple consumers need same dtype
// mul1 needs FP32 (same as cast output), abs needs FP32 (same as cast output)
TEST_F(TestDtypeConsistencyST, MergeUpstreamCastMultipleConsumers) {
  // Upstream cast: FP16 -> FP32 (safe widening)
  auto graph = AscGraphBuilder("test_merge_multiple_consumers")
                   .Loops({Sym("s0")})
                   .Data("data1", 0, {Sym("s0")}, {One}, af::DT_FLOAT16)
                   .Load("load", "data1")
                   .Cast("cast", "load", af::DT_FLOAT)
                   .Mul("mul1", "cast", "cast")
                   .Abs("abs", "cast")
                   .Add("add", "abs", "mul1")
                   .Store("store1", "add")
                   .Build();

  std::vector<optimize::NodeDtypeRequirement> mock_requirements;
  mock_requirements.push_back({graph.FindNode("load"), {af::DT_FLOAT16}, {af::DT_FLOAT16}});
  mock_requirements.push_back({graph.FindNode("mul1"), {af::DT_FLOAT, af::DT_FLOAT}, {af::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("abs"), {af::DT_FLOAT}, {af::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("add"), {af::DT_FLOAT, af::DT_FLOAT}, {af::DT_FLOAT}});
  mock_requirements.push_back({graph.FindNode("store1"), {af::DT_FLOAT}, {af::DT_FLOAT}});

  ASSERT_EQ(optimize::DtypeConsistency::ApplyDtypeConversions(graph, mock_requirements), af::SUCCESS);
  ASSERT_EQ(optimize::DtypeConsistency::CancelRedundantCast(graph), af::SUCCESS);
  ASSERT_EQ(optimize::ScheduleUtils::TopologicalSorting(graph), af::SUCCESS);

  // Verify: single FP16->FP32 cast serves all consumers (mul1, abs)
  auto all_nodes = graph.GetAllNodes();
  size_t cast_count = 0;
  bool mul1_is_fp32 = false;
  bool abs_is_fp32 = false;
  bool has_fp16_to_fp32_cast = false;

  for (const auto &node : all_nodes) {
    if (IsOps<Cast>(node)) {
      cast_count++;
      auto in_anchor = node->GetInDataAnchor(0);
      auto out_anchor = node->GetOutDataAnchor(0);
      if (in_anchor != nullptr && in_anchor->GetPeerOutAnchor() != nullptr && out_anchor != nullptr &&
          !out_anchor->GetPeerInDataAnchors().empty()) {
        if (node->inputs[0].attr.dtype == af::DT_FLOAT16 && node->outputs[0].attr.dtype == af::DT_FLOAT) {
          has_fp16_to_fp32_cast = true;
        }
      }
    }
    if (node->GetName() == "mul1") {
      if (node->inputs[0].attr.dtype == af::DT_FLOAT && node->outputs[0].attr.dtype == af::DT_FLOAT) {
        mul1_is_fp32 = true;
      }
    }
    if (node->GetName() == "abs") {
      if (node->inputs[0].attr.dtype == af::DT_FLOAT && node->outputs[0].attr.dtype == af::DT_FLOAT) {
        abs_is_fp32 = true;
      }
    }
  }

  // Should have 1 Cast: FP16->FP32 serving all consumers
  EXPECT_EQ(cast_count, 1U);
  EXPECT_TRUE(mul1_is_fp32);
  EXPECT_TRUE(abs_is_fp32);
  EXPECT_TRUE(has_fp16_to_fp32_cast);
}

// ST Test 9: TryMergeWithUpstreamCast - safe widening merge scenario with single consumer
// Safe scenario: FP16 -> FP32, downstream node needs FP32 (same as cast output)
TEST_F(TestDtypeConsistencyST, TryMergeWithUpstreamCast) {
  // Upstream cast: FP16 -> FP32 (safe widening)
  auto graph = AscGraphBuilder("test_merge_upstream_multiple")
                   .Loops({Sym("s0")})
                   .Data("data0", 0, {Sym("s0")}, {One}, af::DT_FLOAT16)
                   .Load("load", "data0")
                   .Cast("cast0", "load", af::DT_FLOAT)
                   .Add("add", "cast0", "cast0")
                   .Build();

  auto upstream_cast = graph.FindNode("cast0");
  auto downstream_node = graph.FindNode("add");

  // FP16->FP32 preserves values, target is FP32 (same as cast output), merge should succeed
  // This will modify cast0's output dtype to FP32 (no change needed) and update add's input dtype
  EXPECT_TRUE(optimize::DtypeConsistency::TryMergeWithUpstreamCast(graph, upstream_cast, downstream_node, 0, DT_FLOAT));
}
}  // namespace
