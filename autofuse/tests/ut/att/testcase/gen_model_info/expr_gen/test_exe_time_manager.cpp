/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <iostream>
#include <limits>
#include "gtest/gtest.h"
#include "test_brc_buf_graph.h"
#include "parser/ascend_graph_parser.h"
#include "expr_gen/arg_list_manager.h"
#include "expr_gen/exe_time_pass.h"
#include "expr_gen/cache_guard_count.h"
#include "graph_construct_utils.h"

namespace att {
namespace {
Expr Symbol(const char *name) {
  return af::Expression::Parse(name);
}

CacheGuardInfo MakeOriginGuard(const Expr &guard_axis, const Expr &loop_extent) {
  return {CacheGuardKind::kOriginBroadcast, guard_axis, Expr(), Expr(), loop_extent, true, true};
}

CacheGuardInfo MakeFusedGuard(const Expr &guard_axis, const Expr &period, const Expr &block_inner_extent,
                              const Expr &loop_extent) {
  return {CacheGuardKind::kFusedBroadcast, guard_axis, period, block_inner_extent, loop_extent, true, true};
}

uint64_t EvaluateConst(const Expr &expr) {
  uint64_t value = 0;
  EXPECT_TRUE(expr.GetConstValue(value));
  return value;
}

TuningSpacePtr MakeGuardTuningSpace(const Expr &period, const Expr &inner_extent, const Expr &block_dim) {
  auto tuning_space = std::make_shared<TuningSpace>();
  auto origin = std::make_unique<SubAxis>();
  origin->id = 1;
  origin->name = "guarded";
  origin->repeat = inner_extent;
  auto outer = std::make_unique<SubAxis>();
  outer->id = 2;
  outer->name = "outer";
  outer->axis_type = AxisPosition::OUTER;
  outer->repeat = period;
  auto inner = std::make_unique<SubAxis>();
  inner->id = 3;
  inner->name = "block_inner";
  inner->axis_type = AxisPosition::INNER;
  inner->is_bind_multi_core = true;
  inner->repeat = inner_extent;
  inner->orig_axis = {origin.get()};
  // Parser split outputs reference their common source axis through parent_axis.
  inner->parent_axis = {origin.get()};
  outer->parent_axis = {origin.get()};
  auto block = std::make_unique<SubAxis>();
  block->id = 4;
  block->name = "block";
  block->axis_type = AxisPosition::OUTER;
  block->repeat = block_dim;
  tuning_space->sub_axes.emplace_back(std::move(origin));
  tuning_space->sub_axes.emplace_back(std::move(outer));
  tuning_space->sub_axes.emplace_back(std::move(inner));
  tuning_space->sub_axes.emplace_back(std::move(block));
  tuning_space->block_dims = {{tuning_space->sub_axes[3].get()}};
  NodeInfo node;
  node.name = "cast0";
  node.exec_condition = af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis;
  node.loop_axes = {tuning_space->sub_axes[2].get()};
  node.cache_executions.emplace_back();
  node.cache_executions.front().loops = {{3, Symbol("block_inner"), inner_extent}};
  node.cache_executions.front().mode = ascir_param::CacheExecutionMode::kFusedBroadcast;
  node.cache_executions.front().call_loop_axis_id = 3;
  node.cache_executions.front().guard = MakeFusedGuard(Symbol("block_inner"), period, inner_extent, inner_extent);
  tuning_space->node_infos.emplace_back(node);
  return tuning_space;
}
}  // namespace

TEST(CacheGuardInfoTest, OriginGuardCountsOneHit) {
  CacheGuardInfo info = MakeOriginGuard(Symbol("i"), Symbol("19"));
  EXPECT_EQ(EvaluateConst(CountGuardHits(info, Symbol("0"))), 1);
}

TEST(CacheGuardInfoTest, FusedGuardCountsPeriodHitsPerBlock) {
  CacheGuardInfo info = MakeFusedGuard(Symbol("i"), Symbol("4"), Symbol("2"), Symbol("6"));
  EXPECT_EQ(EvaluateConst(CountGuardHits(info, Symbol("0"))), 1);
  EXPECT_EQ(EvaluateConst(CountGuardHits(info, Symbol("1"))), 1);
}

TEST(CacheGuardInfoTest, FusedGuardReturnsZeroBeyondPeriod) {
  CacheGuardInfo info = MakeFusedGuard(Symbol("i"), Symbol("4"), Symbol("2"), Symbol("2"));
  EXPECT_EQ(EvaluateConst(CountGuardHits(info, Symbol("3"))), 0);
}

TEST(CacheGuardInfoTest, InvalidPeriodDoesNotBuildModulo) {
  CacheGuardInfo info = MakeFusedGuard(Symbol("i"), Symbol("0"), Symbol("2"), Symbol("6"));
  EXPECT_FALSE(ValidateCacheGuardInfo(info));
}

TEST(CacheGuardInfoTest, BlockDependentCountFallsBackUnlessUniform) {
  CacheGuardInfo info = MakeFusedGuard(Symbol("i"), Symbol("5"), Symbol("2"), Symbol("6"));
  EXPECT_FALSE(IsBlockCountUniform(info, Symbol("54")));
}

TEST(CacheGuardInfoTest, SymbolicCanonicalBlockSplitProvesUniformity) {
  const auto period = Symbol("period");
  const auto inner = Symbol("inner");
  CacheGuardInfo info = MakeFusedGuard(Symbol("i"), period, inner, inner);
  const auto block_dim = af::sym::Ceiling(af::sym::Div(period, inner));
  EXPECT_TRUE(IsBlockCountUniform(info, block_dim));
}

TEST(CacheGuardInfoTest, ArbitrarySymbolicBlockDimFallsBack) {
  CacheGuardInfo info = MakeFusedGuard(Symbol("period"), Symbol("inner"), Symbol("inner"), Symbol("inner"));
  EXPECT_FALSE(IsBlockCountUniform(info, Symbol("block_dim")));
}

TEST(CacheGuardInfoTest, UnitExtentTailMakesBlockCountNonUniform) {
  CacheGuardInfo info = MakeFusedGuard(Symbol("i"), Symbol("5"), Symbol("2"), Symbol("1"));
  EXPECT_FALSE(IsBlockCountUniform(info, Symbol("4")));
}

TEST(CacheGuardInfoTest, UnitPeriodTailMakesBlockCountNonUniform) {
  CacheGuardInfo info = MakeFusedGuard(Symbol("i"), Symbol("1"), Symbol("2"), Symbol("2"));
  EXPECT_FALSE(IsBlockCountUniform(info, Symbol("2")));
}

TEST(CacheGuardInfoTest, UnitPeriodUnitExtentCountsSingleHit) {
  CacheGuardInfo info = MakeFusedGuard(Symbol("i"), Symbol("1"), Symbol("2"), Symbol("1"));
  EXPECT_EQ(EvaluateConst(CountGuardHits(info, Symbol("0"))), 1);
}

TEST(CacheGuardInfoTest, RejectsNonPositiveLoopExtent) {
  EXPECT_FALSE(ValidateCacheGuardInfo(MakeOriginGuard(Symbol("i"), Symbol("0"))));
  EXPECT_FALSE(ValidateCacheGuardInfo(MakeOriginGuard(Symbol("i"), Symbol("-1"))));
}

TEST(CacheGuardInfoTest, AllowsMixedSymbolicAndConstantMetadata) {
  CacheGuardInfo info = MakeFusedGuard(Symbol("i"), Symbol("period"), Symbol("2"), Symbol("extent"));
  EXPECT_TRUE(ValidateCacheGuardInfo(info));
}

TEST(CacheGuardInfoTest, RejectsUnprovenSymbolicMetadata) {
  CacheGuardInfo info = MakeFusedGuard(Symbol("i"), Symbol("period"), Symbol("2"), Symbol("extent"));
  info.positive_range_proven = false;
  EXPECT_FALSE(ValidateCacheGuardInfo(info));
}

TEST(CacheGuardInfoTest, RejectsNonIntegerConstantMetadata) {
  CacheGuardInfo info = MakeFusedGuard(Symbol("i"), Symbol("2.5"), Symbol("2"), Symbol("6"));
  EXPECT_FALSE(ValidateCacheGuardInfo(info));
}

TEST(CacheGuardInfoTest, NoneGuardSafelyReturnsZero) {
  CacheGuardInfo info;
  EXPECT_EQ(EvaluateConst(CountGuardHits(info, Symbol("0"))), 0);
}

TEST(CacheGuardInfoTest, OverflowingBlockRangeIsNotUniform) {
  CacheGuardInfo info = MakeFusedGuard(Symbol("i"), CreateExpr(1U),
                                       CreateExpr(std::numeric_limits<uint64_t>::max() / 2U), CreateExpr(2U));
  EXPECT_FALSE(IsBlockCountUniform(info, CreateExpr(3U)));
}

TEST(CacheGuardInfoTest, LargeLoopExtentUsesConstantTimeCounting) {
  CacheGuardInfo info = MakeFusedGuard(Symbol("i"), CreateExpr(4U), CreateExpr(2U), CreateExpr(1000000000000UL));
  EXPECT_EQ(EvaluateConst(CountGuardHits(info, CreateExpr(0U))), 1);
}

TEST(CacheGuardInfoTest, RejectsRealExtentBeyondUint64) {
  EXPECT_FALSE(ValidateCacheGuardInfo(MakeOriginGuard(Symbol("i"), Symbol("18446744073709551616.0"))));
}

size_t CountSubstring(const std::string &text, const std::string &needle) {
  size_t count = 0U;
  for (size_t pos = text.find(needle); pos != std::string::npos; pos = text.find(needle, pos + needle.size())) {
    ++count;
  }
  return count;
}

TEST(CacheGuardInfoTest, BlockZeroHitCountStaysSingleCeiling) {
  // Block 0 starts at phase 0, so the general formula's offset and trailing-tail
  // terms are constant there.  They must not reach the emitted expression,
  // otherwise every symbolic guard carries two dead factors into downstream cost.
  auto info = MakeFusedGuard(Symbol("i"), Symbol("period"), Symbol("period"), Symbol("bytes"));
  // DoubleTile records never clip to the period, so the closed form is the
  // plain Ceiling(loop_extent / period) without a Min.
  info.clip_to_period = false;
  const auto hits = CountGuardHits(info, CreateExpr(0U));
  const std::string rendered = Str(hits);
  EXPECT_EQ(rendered.find("Floor"), std::string::npos) << rendered;
  EXPECT_EQ(rendered.find("Mod"), std::string::npos) << rendered;
  EXPECT_EQ(rendered.find("Min"), std::string::npos) << rendered;
  // A single Ceiling is the whole formula; the general path would emit three.
  EXPECT_EQ(CountSubstring(rendered, "Ceiling"), 1) << rendered;
}

TEST(CacheGuardInfoTest, BlockDependentAGuardUsesBoundedCount) {
  auto space = MakeGuardTuningSpace(CreateExpr(4), CreateExpr(1), Symbol("block_dim"));
  auto &node = space->node_infos.front();
  auto &record = node.cache_executions.front();
  record.mode = ascir_param::CacheExecutionMode::kReduceDoubleTile;
  record.loops = {{3, Symbol("block_inner"), CreateExpr(4)}};
  // block_inner_extent = 1 and period = 4 come from different axes, so the phase
  // differs per block and no exact closed form exists.
  auto guard = MakeFusedGuard(Symbol("block_inner"), CreateExpr(4), CreateExpr(1), CreateExpr(4));
  guard.clip_to_period = false;
  record.reduce_guards[ascir_param::ReduceCacheCondition::kA] = guard;
  ExeTimePassManager manager(space);
  // Enumerating the emitted predicate gives 1, 2, 2, 2 hits for blocks 0..3, so
  // the count must land on the tightest phase independent bound, not on 0.
  EXPECT_EQ(manager.UpdateNodeExeTime(node, CreateExpr(4)).GetTernaryOpStr(), "2");
}

TEST(CacheGuardInfoTest, RGuardIsPhaseIndependent) {
  auto space = MakeGuardTuningSpace(CreateExpr(8), CreateExpr(8), Symbol("block_dim"));
  auto &node = space->node_infos.front();
  auto &record = node.cache_executions.front();
  record.mode = ascir_param::CacheExecutionMode::kReduceDoubleTile;
  record.loops = {{3, Symbol("block_inner"), CreateExpr(8)}};
  // R does not reference block_dim, so it stays exact even for a symbolic one.
  auto guard = MakeFusedGuard(Symbol("block_inner"), CreateExpr(4), CreateExpr(1), CreateExpr(8));
  guard.clip_to_period = false;
  record.reduce_guards[ascir_param::ReduceCacheCondition::kR] = guard;
  ExeTimePassManager manager(space);
  EXPECT_EQ(manager.UpdateNodeExeTime(node, CreateExpr(8)).GetTernaryOpStr(), "2");
}

TEST(CacheGuardInfoTest, EmittedGuardInAnotherScopeKeepsFullCount) {
  auto space = MakeGuardTuningSpace(Symbol("20"), Symbol("20"), Symbol("1"));
  auto &node = space->node_infos.front();
  ExeTimePassManager manager(space);
  ASSERT_EQ(manager.UpdateNodeExeTime(node, Symbol("20")).GetTernaryOpStr(), "1");
  // The record claims a call site on axis 99, which is not the recorded loop axis.
  node.cache_executions.front().call_loop_axis_id = 99;
  EXPECT_EQ(manager.UpdateNodeExeTime(node, Symbol("20")).GetTernaryOpStr(), "20");
  // A guard whose extent disagrees with the recorded loop is not used either.
  node.cache_executions.front().call_loop_axis_id = 3;
  node.cache_executions.front().guard.loop_extent = Symbol("19");
  EXPECT_EQ(manager.UpdateNodeExeTime(node, Symbol("20")).GetTernaryOpStr(), "20");
}

TEST(CacheGuardInfoTest, NodeWithoutRecordKeepsFullCountForNddma) {
  auto space = MakeGuardTuningSpace(Symbol("1116"), Symbol("20"), Symbol("56"));
  auto &node = space->node_infos.front();
  node.node_type = "Nddma";
  node.cache_executions.clear();
  ExeTimePassManager manager(space);
  EXPECT_EQ(manager.UpdateNodeExeTime(node, Symbol("20")).GetTernaryOpStr(), "20");
}

class TestExeTimePass : public ::testing::Test {
 public:
  static void TearDownTestCase() {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase() {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(TestExeTimePass, FusedGuardUsesHitCount) {
  auto tuning_space = MakeGuardTuningSpace(Symbol("4"), Symbol("2"), Symbol("2"));
  const auto &node = tuning_space->node_infos.front();
  ExeTimePassManager exe_mgr(tuning_space);
  auto exe = exe_mgr.UpdateNodeExeTime(node, Symbol("2"));
  EXPECT_EQ(exe.GetTernaryOpStr(), "1");
}

TEST_F(TestExeTimePass, SymbolicCanonicalBlockSplitUsesHitCount) {
  const auto period = Symbol("period");
  const auto inner = Symbol("inner");
  auto tuning_space = MakeGuardTuningSpace(period, inner, af::sym::Ceiling(af::sym::Div(period, inner)));
  const auto &node = tuning_space->node_infos.front();
  ExeTimePassManager exe_mgr(tuning_space);
  const auto exe = exe_mgr.UpdateNodeExeTime(node, inner);
  EXPECT_NE(exe.GetTernaryOpStr(), Str(inner));
}

TEST_F(TestExeTimePass, OriginGuardUsesSingleHit) {
  auto tuning_space = MakeGuardTuningSpace(Symbol("4"), Symbol("2"), Symbol("2"));
  auto &node = tuning_space->node_infos.front();
  node.exec_condition = af::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis;
  node.cache_executions.front().mode = ascir_param::CacheExecutionMode::kOriginBroadcast;
  node.cache_executions.front().guard = MakeOriginGuard(Symbol("block_inner"), Symbol("2"));
  ExeTimePassManager exe_mgr(tuning_space);
  auto exe = exe_mgr.UpdateNodeExeTime(node, Symbol("2"));
  EXPECT_EQ(exe.GetTernaryOpStr(), "1");
}

TEST_F(TestExeTimePass, TailBlockUsesSingleGuardHit) {
  auto tuning_space = MakeGuardTuningSpace(Symbol("1024"), Symbol("19"), Symbol("54"));
  const auto &node = tuning_space->node_infos.front();
  ExeTimePassManager exe_mgr(tuning_space);
  auto exe = exe_mgr.UpdateNodeExeTime(node, Symbol("19"));
  EXPECT_EQ(exe.GetTernaryOpStr(), "1");
}

TEST_F(TestExeTimePass, NonUniformGuardFallsBackToLegacyRepeat) {
  auto tuning_space = MakeGuardTuningSpace(Symbol("5"), Symbol("6"), Symbol("54"));
  const auto &node = tuning_space->node_infos.front();
  ExeTimePassManager exe_mgr(tuning_space);
  auto exe = exe_mgr.UpdateNodeExeTime(node, Symbol("19"));
  EXPECT_EQ(exe.GetTernaryOpStr(), "19");
}

TEST_F(TestExeTimePass, MissingMetadataFallsBackToLegacyRepeat) {
  auto tuning_space = MakeGuardTuningSpace(Symbol("4"), Symbol("2"), Symbol("54"));
  auto &node = tuning_space->node_infos.front();
  node.exec_condition = af::ExecuteCondition::kConditionInvalid;
  node.cache_executions.clear();
  ExeTimePassManager exe_mgr(tuning_space);
  auto exe = exe_mgr.UpdateNodeExeTime(node, Symbol("19"));
  EXPECT_EQ(exe.GetTernaryOpStr(), "19");
}

TEST_F(TestExeTimePass, MissingAndDisabledGuardsKeepTwentyCallsForAllMovementKinds) {
  auto space = MakeGuardTuningSpace(Symbol("1116"), Symbol("20"), Symbol("56"));
  auto &node = space->node_infos.front();
  ExeTimePassManager manager(space);
  for (const auto *type : {"Load", "Nddma", "Broadcast"}) {
    SCOPED_TRACE(type);
    node.node_type = type;
    for (auto mode : {ascir_param::CacheExecutionMode::kUnknown, ascir_param::CacheExecutionMode::kNoGuard}) {
      node.cache_executions.front().mode = mode;
      EXPECT_EQ(manager.UpdateNodeExeTime(node, Symbol("20")).GetTernaryOpStr(), "20");
    }
  }
}

TEST_F(TestExeTimePass, OriginGuardRetainsOtherLoopCalls) {
  auto space = MakeGuardTuningSpace(Symbol("20"), Symbol("20"), Symbol("1"));
  auto &node = space->node_infos.front();
  SubAxis other;
  other.id = 7;
  other.repeat = Symbol("3");
  node.loop_axes.emplace_back(&other);
  node.cache_executions.front().loops.push_back({7, Symbol("other"), Symbol("3")});
  node.cache_executions.front().mode = ascir_param::CacheExecutionMode::kOriginBroadcast;
  node.cache_executions.front().guard = MakeOriginGuard(Symbol("block_inner"), Symbol("20"));
  ExeTimePassManager manager(space);
  EXPECT_EQ(manager.UpdateNodeExeTime(node, Symbol("60")).GetTernaryOpStr(), "3");
  // A caller-provided base that cannot be matched to the loop factors is not compressed.
  EXPECT_EQ(manager.UpdateNodeExeTime(node, Symbol("19")).GetTernaryOpStr(), "19");
}

TEST_F(TestExeTimePass, InvalidGuardAndEmptyWorkNeverCreateFalseZeroCost) {
  auto space = MakeGuardTuningSpace(Symbol("1116"), Symbol("20"), Symbol("56"));
  auto &node = space->node_infos.front();
  ExeTimePassManager manager(space);
  for (const auto *period : {"0", "-1", "2.5"}) {
    node.cache_executions.front().guard.period = Symbol(period);
    EXPECT_EQ(manager.UpdateNodeExeTime(node, Symbol("20")).GetTernaryOpStr(), "20");
  }
  EXPECT_EQ(manager.UpdateNodeExeTime(node, Symbol("0")).GetTernaryOpStr(), "0");
}

TEST_F(TestExeTimePass, MismatchedGuardKindKeepsFullCost) {
  auto space = MakeGuardTuningSpace(Symbol("1116"), Symbol("20"), Symbol("56"));
  auto &node = space->node_infos.front();
  node.cache_executions.front().guard.kind = CacheGuardKind::kNone;
  ExeTimePassManager manager(space);
  EXPECT_EQ(manager.UpdateNodeExeTime(node, Symbol("20")).GetTernaryOpStr(), "20");
}

TEST(CacheGuardInfoTest, FusedCountMatchesEnumeratedGeneratedPredicate) {
  for (uint64_t period = 1; period <= 9; ++period) {
    for (uint64_t inner = 1; inner <= 4; ++inner) {
      auto info = MakeFusedGuard(Symbol("i"), CreateExpr(period), CreateExpr(inner), CreateExpr(inner));
      for (uint64_t block = 0; block < 5; ++block) {
        uint64_t expected = 0;
        const auto extent = block * inner >= period ? 0U : std::min(inner, period - block * inner);
        for (uint64_t i = 0; i < extent; ++i) {
          expected += (i < 1 || (block * inner + i) % period < 1) ? 1U : 0U;
        }
        EXPECT_EQ(EvaluateConst(CountGuardHits(info, CreateExpr(block))), expected);
      }
    }
  }
}

void GetExeTime(const TuningSpacePtr &tuning_space, TernaryOp &exe_cast0, TernaryOp &exe_store) {
  ExeTimePassManager exe_mgr(tuning_space);
  Expr exe_time;
  for (const auto &node : tuning_space->node_infos) {
    exe_time = CreateExpr(1U);
    for (auto &loop_axis : node.loop_axes) {
      exe_time = af::sym::Mul(exe_time, loop_axis->repeat);
    }
    if (node.name == "cast0") {
      exe_cast0 = exe_mgr.UpdateNodeExeTime(node, exe_time);
    } else if (node.name == "store") {
      exe_store = exe_mgr.UpdateNodeExeTime(node, exe_time);
    }
  }
}

TEST_F(TestExeTimePass, case1) {
  af::AscGraph graph("graph");
  att::BrcBufBeforeAutoFuse1(graph);
  att::BrcBufAfterScheduler1(graph);
  att::BrcBufAfterQueBufAlloc1(graph);
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  TuningSpacePtr tuning_space = std::make_shared<TuningSpace>();
  att::AscendGraphParser ascend_graph_parser(tuning_space);
  auto ret = ascend_graph_parser.GraphParser(graph);
  TernaryOp exe_time_cast0;
  TernaryOp exe_time_store;
  GetExeTime(tuning_space, exe_time_cast0, exe_time_store);
  EXPECT_EQ(exe_time_cast0.GetTernaryOpStr(), "(Ceiling((Z1 / (z1t_size))) * z0z2Tb_size)");
  EXPECT_EQ(exe_time_store.GetTernaryOpStr(), "(Ceiling((Z1 / (z1t_size))) * z0z2Tb_size)");
}

TEST_F(TestExeTimePass, case2) {
  af::AscGraph graph("graph");
  att::BrcBufBeforeAutoFuse2(graph);
  att::BrcBufAfterScheduler1(graph);
  att::BrcBufAfterQueBufAlloc1(graph);
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  TuningSpacePtr tuning_space = std::make_shared<TuningSpace>();
  att::AscendGraphParser ascend_graph_parser(tuning_space);
  auto ret = ascend_graph_parser.GraphParser(graph);
  TernaryOp exe_time_cast0;
  TernaryOp exe_time_store;
  GetExeTime(tuning_space, exe_time_cast0, exe_time_store);
  EXPECT_EQ(exe_time_cast0.GetTernaryOpStr(), "(Ceiling((Z1 / (z1t_size))) * z0z2Tb_size)");
  EXPECT_EQ(exe_time_store.GetTernaryOpStr(), "(Ceiling((Z1 / (z1t_size))) * z0z2Tb_size)");
}

TEST_F(TestExeTimePass, case3) {
  af::AscGraph graph("graph");
  att::BrcBufBeforeAutoFuse3(graph);
  att::BrcBufAfterScheduler3(graph);
  att::BrcBufAfterQueBufAlloc3(graph);
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  TuningSpacePtr tuning_space = std::make_shared<TuningSpace>();
  att::AscendGraphParser ascend_graph_parser(tuning_space);
  auto ret = ascend_graph_parser.GraphParser(graph);
  TernaryOp exe_time_cast0;
  TernaryOp exe_time_store;
  GetExeTime(tuning_space, exe_time_cast0, exe_time_store);
  EXPECT_EQ(exe_time_cast0.GetTernaryOpStr(), "(Ceiling((Z1 / (z1t_size))) * z0z2Tb_size)");
  EXPECT_EQ(exe_time_store.GetTernaryOpStr(), "(Ceiling((Z1 / (z1t_size))) * z0z2Tb_size)");
}

TEST_F(TestExeTimePass, case4) {
  af::AscGraph graph("graph");
  att::BrcBufBeforeAutoFuse4(graph);
  att::BrcBufAfterScheduler4(graph);
  att::BrcBufAfterQueBufAlloc4(graph);
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  TuningSpacePtr tuning_space = std::make_shared<TuningSpace>();
  att::AscendGraphParser ascend_graph_parser(tuning_space);
  auto ret = ascend_graph_parser.GraphParser(graph);
  TernaryOp exe_time_cast0;
  TernaryOp exe_time_store;
  GetExeTime(tuning_space, exe_time_cast0, exe_time_store);
  EXPECT_EQ(exe_time_cast0.GetTernaryOpStr(), "(Ceiling((Z1 / (z1t_size))) * z0z2Tb_size)");
  EXPECT_EQ(exe_time_store.GetTernaryOpStr(), "(Ceiling((Z1 / (z1t_size))) * z0z2Tb_size)");
}

TEST_F(TestExeTimePass, exec_condition) {
  af::AscGraph graph("graph");
  att::BrcBufBeforeAutoFuse3(graph);
  att::BrcBufAfterScheduler3(graph);
  att::BrcBufAfterQueBufAlloc3(graph);
  auto load = graph.FindNode("cast0");
  load->attr.sched.exec_condition = af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis;
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  TuningSpacePtr tuning_space = std::make_shared<TuningSpace>();
  att::AscendGraphParser ascend_graph_parser(tuning_space);
  auto ret = ascend_graph_parser.GraphParser(graph);
  TernaryOp exe_time_cast0;
  TernaryOp exe_time_store;
  GetExeTime(tuning_space, exe_time_cast0, exe_time_store);
  EXPECT_EQ(exe_time_cast0.GetTernaryOpStr(), "(Ceiling((Z1 / (z1t_size))) * z0z2Tb_size)");
  EXPECT_EQ(exe_time_store.GetTernaryOpStr(), "(Ceiling((Z1 / (z1t_size))) * z0z2Tb_size)");
}
}  // namespace att

namespace att {
TEST_F(TestExeTimePass, ReduceConditionsMatchEnumeratedEmittedPredicates) {
  using Condition = ascir_param::ReduceCacheCondition;
  auto space = MakeGuardTuningSpace(CreateExpr(1), CreateExpr(1), CreateExpr(1));
  auto &node = space->node_infos.front();
  auto &record = node.cache_executions.front();
  record.mode = ascir_param::CacheExecutionMode::kReduceDoubleTile;
  ExeTimePassManager manager(space);
  for (uint64_t a = 1; a <= 4; ++a) {
    for (uint64_t r = 1; r <= 5; ++r) {
      record.loops = {{10, Symbol("a"), CreateExpr(a)}, {11, Symbol("r"), CreateExpr(r)}};
      for (uint64_t period_a = 1; period_a <= 3; ++period_a) {
        for (uint64_t period_r = 1; period_r <= 3; ++period_r) {
          for (uint64_t control = 1; control <= 2; ++control) {
            for (uint32_t mask = 1; mask <= 7; ++mask) {
              record.reduce_guards.clear();
              if ((mask & 1U) != 0U) {
                auto guard = MakeFusedGuard(Symbol("a"), CreateExpr(period_a), CreateExpr(a), CreateExpr(a));
                guard.clip_to_period = false;
                record.reduce_guards[Condition::kA] = guard;
              }
              if ((mask & 2U) != 0U) {
                auto guard = MakeFusedGuard(Symbol("r"), CreateExpr(period_r), CreateExpr(1), CreateExpr(r));
                guard.clip_to_period = false;
                record.reduce_guards[Condition::kR] = guard;
              }
              if ((mask & 4U) != 0U) {
                record.reduce_guards[Condition::kControl].loop_extent = CreateExpr(control);
              }
              uint64_t expected = 0;
              for (uint64_t i = 0; i < a; ++i) {
                for (uint64_t j = 0; j < r; ++j) {
                  expected +=
                      (((mask & 1U) != 0U && (i == 0 || i % period_a == 0)) ||
                       ((mask & 2U) != 0U && (j == 0 || j % period_r == 0)) || ((mask & 4U) != 0U && control != 1))
                          ? 1U
                          : 0U;
                }
              }
              const auto result = manager.UpdateNodeExeTime(node, CreateExpr(a * r)).DeepCopyIfCase();
              ASSERT_TRUE(result->IsLeaf());
              EXPECT_EQ(EvaluateConst(result->GetExpr().Simplify()), expected)
                  << a << ',' << r << ',' << period_a << ',' << period_r << ',' << control << ',' << mask;
            }
          }
        }
      }
    }
  }
}

TEST_F(TestExeTimePass, MultipleCallsKeepTheirOwnLoopFactors) {
  auto space = MakeGuardTuningSpace(CreateExpr(20), CreateExpr(20), CreateExpr(1));
  auto &node = space->node_infos.front();
  auto &first = node.cache_executions.front();
  first.mode = ascir_param::CacheExecutionMode::kOriginBroadcast;
  first.guard = MakeOriginGuard(Symbol("block_inner"), CreateExpr(20));
  auto second = first;
  second.mode = ascir_param::CacheExecutionMode::kNoGuard;
  second.loops = {{9, Symbol("other"), CreateExpr(6)}};
  node.cache_executions.push_back(second);
  ExeTimePassManager manager(space);
  EXPECT_EQ(manager.UpdateNodeExeTime(node, CreateExpr(20)).GetTernaryOpStr(), "7");
}

TEST_F(TestExeTimePass, IncompleteReduceConditionsKeepFullCount) {
  auto space = MakeGuardTuningSpace(CreateExpr(20), CreateExpr(20), CreateExpr(1));
  auto &node = space->node_infos.front();
  auto &record = node.cache_executions.front();
  record.mode = ascir_param::CacheExecutionMode::kReduceDoubleTile;
  ExeTimePassManager manager(space);
  EXPECT_EQ(manager.UpdateNodeExeTime(node, CreateExpr(20)).GetTernaryOpStr(), "20");
  record.reduce_guards[ascir_param::ReduceCacheCondition::kR] = {};
  EXPECT_EQ(manager.UpdateNodeExeTime(node, CreateExpr(20)).GetTernaryOpStr(), "20");
}

}  // namespace att
