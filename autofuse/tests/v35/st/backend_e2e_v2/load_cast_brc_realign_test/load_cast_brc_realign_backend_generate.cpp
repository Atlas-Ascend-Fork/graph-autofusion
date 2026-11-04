/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <gtest/gtest.h>
#include "optimize.h"
#include "share_graph.h"
#include "backend_common.h"

#include <iostream>
#include <vector>
#include <string>
#include "runtime_stub.h"
#include "common/platform_context.h"

// load-cast-brc 调序生成 nddma 的优化阶段验证: 分别以常量shape(稀疏store, brc单根向量化轴且
// stride为常量, 走按两侧dtype的常量实判与重排)和动态shape(brc次尾轴向量化stride为非常量表达式,
// 走保守触发重排)两种构图跑完整 Optimize 流程, 验证 swap 及次尾轴/单轴重排在全链路下不异常。
// 本用例不生成真实kernel, KERNEL_SRC槽位写入占位源文件, 仅保持与既有e2e用例相同的构建形态。
class TestBackendLoadCastBrcRealignE2e : public testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::PlatformContext::GetInstance().Reset();
    af::InstallV35SocPerfRuntimeStub();
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::RuntimeStub::Reset();
  }
};

namespace {
void OptimizeLoadCastBrcGraph(const af::ComputeGraphPtr &graph) {
  ASSERT_NE(graph, nullptr);
  std::vector<::ascir::ScheduledResult> schedule_results;
  ::ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);
}
}  // namespace

TEST_F(TestBackendLoadCastBrcRealignE2e, LoadCastBrcRealignConstOptimize) {
  OptimizeLoadCastBrcGraph(ascir::ShareGraph::LoadCastBrcRealignConstFusedGraph());
}

TEST_F(TestBackendLoadCastBrcRealignE2e, LoadCastBrcRealignDynamicOptimize) {
  OptimizeLoadCastBrcGraph(ascir::ShareGraph::LoadCastBrcRealignDynamicFusedGraph());
}

TEST_F(TestBackendLoadCastBrcRealignE2e, WriteKernelSrcPlaceholder) {
  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
  ASSERT_FALSE(parts.empty());
  std::fstream kernel_file(parts[0], std::ios::out);
  kernel_file << "// placeholder: optimization-stage-only e2e case, no real kernel is generated.\n";
}
