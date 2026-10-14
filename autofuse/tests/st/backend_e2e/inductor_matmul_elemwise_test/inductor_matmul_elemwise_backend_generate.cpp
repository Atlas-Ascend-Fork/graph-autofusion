/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <exception>
#include <fstream>
#include <gtest/gtest.h>

#include <map>
#include <string>
#include <vector>

#include "backend_common.h"
#include "codegen.h"
#include "common_utils.h"
#include "common/platform_context.h"
#include "../../../common/att/soc_perf_test_config.h"
#include "optimize.h"
#include "runtime_stub.h"
#include "share_graph.h"

namespace {
constexpr size_t kTilingDataFileIndex = 0;
constexpr size_t kHostFileIndex = 1;
constexpr size_t kDeviceFileIndex = 2;
constexpr size_t kExpectedKernelSrcCount = 3;

class TestBackendInductorMatmulElemwiseE2e : public testing::Test {
 protected:
  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v2 = std::make_shared<ge::RuntimeStubV2Common>();
    ge::RuntimeStub::SetInstance(stub_v2);
    ASSERT_EQ(
        att::soc_perf_test::SetSocPerfTestPlatform(att::soc_perf_test::kV35AltNpuArch, att::soc_perf_test::kSocEmpty),
        af::SUCCESS);
  }

  void TearDown() override {
    ge::RuntimeStub::Reset();
  }
};

TEST_F(TestBackendInductorMatmulElemwiseE2e, InductorMatmulElemwiseE2eCodegen) {
  auto graph = ascir::ShareGraph::LoadMatmulElewiseBrcFusedGraph();
  auto parts = splitString(KERNEL_SRC_LIST, ':');
  ASSERT_EQ(parts.size(), kExpectedKernelSrcCount);

  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    codegen::Codegen codegen(codegen::CodegenOptions{});
    ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.node_idx_to_scheduled_results.push_back({});
    ASSERT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);
    ASSERT_TRUE(ascgen_utils::IsCubeFusedScheduled(fused_schedule_result));

    codegen::CodegenResult result;
    ASSERT_EQ(codegen.GenerateForInductor(fused_schedule_result, result), 0);

    EXPECT_FALSE(result.tiling_data.empty());
    EXPECT_FALSE(result.tiling.empty());
    EXPECT_FALSE(result.kernel.empty());
    EXPECT_NE(result.tiling_data.find("CVAutofuseTilingData"), std::string::npos);
    EXPECT_NE(result.tiling.find("CVAutofuseTilingData"), std::string::npos);
    EXPECT_NE(result.tiling.find("CallCubeTiling"), std::string::npos);
    EXPECT_EQ(result.tiling.find("GenerateTopnSolutions"), std::string::npos);
    EXPECT_EQ(result.tiling.find("AscirCompileAndLaunch"), std::string::npos);
    EXPECT_NE(result.kernel.find("AutofuseLaunch"), std::string::npos);

    std::fstream tiling_data_file(parts[kTilingDataFileIndex], std::ios::out);
    std::fstream host_file(parts[kHostFileIndex], std::ios::out);
    std::fstream device_file(parts[kDeviceFileIndex], std::ios::out);
    ASSERT_TRUE(tiling_data_file.is_open());
    ASSERT_TRUE(host_file.is_open());
    ASSERT_TRUE(device_file.is_open());

    tiling_data_file << result.tiling_data;
    host_file << result.tiling;
    device_file << result.kernel;
  } catch (const std::exception &e) {
    FAIL() << e.what();
  } catch (...) {
    FAIL() << "inductor matmul elemwise codegen failed";
  }
}

TEST_F(TestBackendInductorMatmulElemwiseE2e, InductorBatchMatmulElemwiseE2eCodegen) {
  auto graph = ascir::ShareGraph::LoadBatchMatmulElewiseBrcFusedGraph();
  auto parts = splitString(KERNEL_SRC_LIST, ':');
  ASSERT_EQ(parts.size(), kExpectedKernelSrcCount);

  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    codegen::Codegen codegen(codegen::CodegenOptions{});
    ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.node_idx_to_scheduled_results.push_back({});
    ASSERT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);
    ASSERT_TRUE(ascgen_utils::IsCubeFusedScheduled(fused_schedule_result));

    codegen::CodegenResult result;
    ASSERT_EQ(codegen.GenerateForInductor(fused_schedule_result, result), 0);

    EXPECT_NE(result.kernel.find("BATCH_API_LEVEL"), std::string::npos);
    EXPECT_NE(result.kernel.find("_DISPATCH_BATCH_MATMUL"), std::string::npos);
    EXPECT_NE(result.kernel.find("BatchMatMulV3TilingData"), std::string::npos);
    EXPECT_EQ(result.kernel.find("int8_t API_LEVEL = (tiling_key >> 0) & 0xF;"), std::string::npos);
  } catch (const std::exception &e) {
    FAIL() << e.what();
  } catch (...) {
    FAIL() << "inductor batch matmul elemwise codegen failed";
  }
}

TEST_F(TestBackendInductorMatmulElemwiseE2e, CvTilingUsesDirectLogHeaderDependency) {
  auto graph = ascir::ShareGraph::LoadMatmulElewiseBrcFusedGraph(true);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.node_idx_to_scheduled_results.push_back({});
  ASSERT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);
  ASSERT_TRUE(ascgen_utils::IsCubeFusedScheduled(fused_schedule_result));

  codegen::Codegen codegen(codegen::CodegenOptions{});
  const std::map<std::string, std::string> shape_info = {{"s0", "stub_s0"}, {"s1", "stub_s1"}};
  const auto tiling_files = codegen.GenerateTiling(fused_schedule_result, shape_info, "", "0");
  EXPECT_EQ(tiling_files.find("TilingDataLog"), tiling_files.end());
  ASSERT_NE(tiling_files.find("BCubeKernelTilingWrapperCpp"), tiling_files.end());
  const auto &wrapper_cpp = tiling_files.at("BCubeKernelTilingWrapperCpp");
  EXPECT_NE(wrapper_cpp.find("#include \"autofuse_tiling_func_log.h\""), std::string::npos);
  EXPECT_EQ(wrapper_cpp.find("autofuse_tiling_data_log.h"), std::string::npos);
}
}  // namespace
