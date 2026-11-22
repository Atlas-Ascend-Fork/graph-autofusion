/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <fstream>
#include <gtest/gtest.h>
#include "backend_common.h"
#include "codegen.h"
#include "common/common_utils.h"
#include "common/platform_context.h"
#include "optimize.h"
#include "runtime_stub.h"
#include "share_graph.h"

namespace {
#if defined(ARGMIN_I64)
constexpr af::DataType kInputType = af::DT_INT64;
const char *const kCppType = "int64_t";
#elif defined(ARGMIN_I32)
constexpr af::DataType kInputType = af::DT_INT32;
const char *const kCppType = "int32_t";
#else
constexpr af::DataType kInputType = af::DT_FLOAT;
const char *const kCppType = "float";
#endif
#ifdef ARGMIN_RA
constexpr bool kRa = true;
#else
constexpr bool kRa = false;
#endif

class BackendArgminTest : public testing::Test {
 protected:
  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
    af::InstallV35SocPerfRuntimeStub();
  }
  void TearDown() override {
    ge::RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
  }
};

void OptimizeMinGraph(ascir::FusedScheduledResult &result) {
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
#ifdef ARGMIN_WITH_VALUE
  const auto graph = ascir::ShareGraph::ArgminWithValueStoreFusedGraph(2, kInputType, kRa);
#else
  const auto graph = ascir::ShareGraph::ArgminStoreFusedGraph(2, kInputType, kRa);
#endif
  ASSERT_NE(graph, nullptr);
  result.node_idx_to_scheduled_results.emplace_back();
  ASSERT_EQ(optimizer.Optimize(graph, result), af::SUCCESS);
}

void SelectMinSchedule(ascir::FusedScheduledResult &result) {
  ASSERT_EQ(result.node_idx_to_scheduled_results.size(), 1UL);
  auto &schedules = result.node_idx_to_scheduled_results.front();
#ifdef ARGMIN_RCORE
  const std::string wanted = ascgen_utils::template_semantic::kReduceRCore;
#elif defined(ARGMIN_ALLLOAD)
  const std::string wanted = ascgen_utils::template_semantic::kReduceAllLoad;
#else
  const std::string wanted = ascgen_utils::template_semantic::kReduceCommon;
#endif
  schedules.erase(
      std::remove_if(schedules.begin(), schedules.end(),
                     [&](const ascir::ScheduledResult &schedule) { return schedule.semantic_name != wanted; }),
      schedules.end());
  ASSERT_FALSE(schedules.empty()) << wanted;
#ifdef ARGMIN_RCORE
  ASSERT_EQ(schedules.front().schedule_groups.size(), 2UL);
#endif
}

void CheckMinCalls(const std::string &source) {
  const std::string type(kCppType);
  const std::string pattern = kRa ? "AscendC::Pattern::Reduce::RA" : "AscendC::Pattern::Reduce::AR";
  const bool ordinary = source.find("ArgMinExtend<" + type + ", " + pattern + ">") != std::string::npos;
  const bool value = source.find("ArgMinWithValueExtend<" + type + ", " + pattern + ">") != std::string::npos;
  const bool accumulate =
      source.find("ArgMinWithValueAccumulateExtend<" + type + ", " + pattern + ">") != std::string::npos;
  EXPECT_TRUE(ordinary || value || accumulate);
  EXPECT_EQ(source.find("ArgMaxExtend<" + type + ","), std::string::npos);
  EXPECT_EQ(source.find("ArgMaxWithValueExtend<" + type + ","), std::string::npos);
#ifdef ARGMIN_RCORE
#ifdef ARGMIN_WITH_VALUE
  EXPECT_NE(source.find("ArgMinMultiRPhase2WithValueVectorExtend<" + type + ">"), std::string::npos);
#else
  EXPECT_NE(source.find("ArgMinMultiRPhase2VectorExtend<" + type + ">"), std::string::npos);
#endif
#endif
}

void EmitMinSources(const ascir::FusedScheduledResult &schedules) {
  codegen::Codegen generator(codegen::CodegenOptions{});
  codegen::CodegenResult generated;
  ASSERT_EQ(generator.Generate({{"s0", "stub_s0"}, {"s1", "stub_s1"}}, schedules, generated), af::SUCCESS);
  ASSERT_NO_FATAL_FAILURE(CheckMinCalls(generated.kernel));
  const auto paths = splitString(KERNEL_SRC_LIST, ':');
  ASSERT_EQ(paths.size(), 3UL);
  std::ofstream kernel(paths[0]), tiling(paths[1]), data(paths[2]);
  ASSERT_TRUE(kernel && tiling && data);
  kernel << R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling) AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)" << RemoveSubDirInclude(generated.kernel);
  tiling << generated.tiling;
  data << generated.tiling_data;
  kernel.close();
  tiling.close();
  data.close();
  ASSERT_TRUE(kernel && tiling && data);
}

TEST_F(BackendArgminTest, RegisteredOpReachesRequestedScheduleAndMinApi) {
  ascir::FusedScheduledResult schedules;
  ASSERT_NO_FATAL_FAILURE(OptimizeMinGraph(schedules));
  ASSERT_NO_FATAL_FAILURE(SelectMinSchedule(schedules));
  ASSERT_NO_FATAL_FAILURE(EmitMinSources(schedules));
}
}  // namespace
