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
#include "ascir_ops.h"
#define private public
#include "optimize.h"
#undef private
#include "ascir_ops_utils.h"
#include "platform_context.h"

#include "asc_graph_builder.h"

using namespace af;
using namespace af::ops;
using namespace af::ascir_op;
using af::testing::AscGraphBuilder;
using af::testing::Sym;

namespace {

class RemoveZeroStrideAxisSt : public ::testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }

  optimize::Optimizer optimizer;

  RemoveZeroStrideAxisSt() : optimizer(optimize::OptimizerOptions{}) {}
};

}  // namespace

// 删轴后退化全归约的端到端调度：z1 为端到端广播轴（Load 输出沿其 stride=0，Sum/Store 均
// 折叠），Optimize 全流程中 RemoveAllZeroStrideLoopAxis 删除 z1 后图退化为纯 R（剩余 z0
// 全为归约轴），就地补 axis_1d 保证 y_group 非空，调度正常产出模板。
// 修复前：GraphPass 期判定因广播轴 src == dst == 0 不成立不补轴，纯 R 图在
// PrepareTilingCases 断言 "No valid tiling cases" 失败。
TEST_F(RemoveZeroStrideAxisSt, DegeneratedAllReduceAfterAxisRemove) {
  auto graph = AscGraphBuilder("degen_all_reduce_after_remove_st")
                   .Loops({Sym("s0"), Sym("s1")})
                   .Data("data", 0, DT_FLOAT)
                   .Load("load", "data", {Sym("s0"), Sym("s1")}, {One, Zero})
                   .Sum("sum", "load", {0, 1})
                   .Store("store", "sum", {One, One}, {Zero, Zero})
                   .Output("output", "store", 0, DT_FLOAT)
                   .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  const Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  EXPECT_FALSE(fused_scheduled_result.node_idx_to_scheduled_results.empty());
  EXPECT_GE(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
}

// 删轴发生但非退化的端到端调度：z2 为端到端广播轴被删，剩余 {z0, z1} 中 z1 为真实保留轴
// （Sum 仅归约 z0/z2，输出沿 z1 非零），不满足全归约判定、不补 axis_1d，调度按既有行为
// 正常产出模板。
TEST_F(RemoveZeroStrideAxisSt, PartialReduceAfterAxisRemove) {
  auto graph = AscGraphBuilder("partial_reduce_after_remove_st")
                   .Loops({Sym("s0"), Sym("s1"), Sym("s2")})
                   .Data("data", 0, DT_FLOAT)
                   .Load("load", "data", {Sym("s0"), Sym("s1"), Sym("s2")}, {Sym("s1") * Sym("s2"), Sym("s2"), Zero})
                   .Sum("sum", "load", {0, 2})
                   .Store("store", "sum", {One, Sym("s1"), One}, {Zero, One, Zero})
                   .Output("output", "store", 0, DT_FLOAT)
                   .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  const Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  EXPECT_FALSE(fused_scheduled_result.node_idx_to_scheduled_results.empty());
  EXPECT_GE(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
}
