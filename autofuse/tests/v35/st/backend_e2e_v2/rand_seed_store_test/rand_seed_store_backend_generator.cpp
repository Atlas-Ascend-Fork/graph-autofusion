/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
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
#include "common/platform_context.h"
#include "optimize.h"
#include "runtime_stub.h"
#include "share_graph.h"

class TestBackendRandSeedStoreE2e : public testing::Test {
 protected:
  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
    af::InstallV35SocPerfRuntimeStub();
  }
  void TearDown() override {
    ge::RuntimeStub::Reset();
  }
};

TEST_F(TestBackendRandSeedStoreE2e, RandSeedStoreE2eCodegen) {
  bool gen_success = true;
  std::string rand_tiling_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";
  std::map<std::string, std::string> shape_info({{"s0", "stub_s0"}});
  const af::DataType dtype =
#if defined(RAND_E2E_DTYPE_U32)
      af::DT_UINT32;
#elif defined(RAND_E2E_DTYPE_I32)
      af::DT_INT32;
#elif defined(RAND_E2E_DTYPE_FP16)
      af::DT_FLOAT16;
#elif defined(RAND_E2E_DTYPE_BF16)
      af::DT_BF16;
#else
      af::DT_FLOAT;
#endif
  auto graph = ascir::ShareGraph::RandSeedStoreFusedGraph(1, dtype);
  std::vector<std::string> source_parts = splitString(KERNEL_SRC_LIST, ':');
  const std::string &kernel_file = source_parts[0];
  const std::string &tiling_file = source_parts[1];
  const std::string &tiling_data_file = source_parts[2];

  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    codegen::Codegen code_generator(codegen::CodegenOptions{});
    std::fstream kernel_stream(kernel_file, std::ios::out);
    std::fstream tiling_stream(tiling_file, std::ios::out);
    std::fstream tiling_data_stream(tiling_data_file, std::ios::out);
    std::vector<::ascir::ScheduledResult> schedules;
    ascir::FusedScheduledResult fused_result;
    fused_result.node_idx_to_scheduled_results.push_back(schedules);
    EXPECT_EQ(optimizer.Optimize(graph, fused_result), 0);
    codegen::CodegenResult result;
    EXPECT_EQ(code_generator.Generate(shape_info, fused_result, result), 0);
    EXPECT_NE(result.kernel.find("RandSeedOffset("), std::string::npos);
    EXPECT_NE(result.kernel.find("simt_api/cpp/kernel_simt_intf.h"), std::string::npos);
    EXPECT_NE(result.kernel.find("tmp_buf_"), std::string::npos);
    kernel_stream << rand_tiling_stub << RemoveSubDirInclude(result.kernel);
    tiling_stream << result.tiling;
    tiling_data_stream << result.tiling_data;
  } catch (...) {
    gen_success = false;
  }
  EXPECT_EQ(gen_success, true);
}

TEST_F(TestBackendRandSeedStoreE2e, RandSeedStoreElementOffsetsCodegen) {
  std::map<std::string, std::string> shape_info({{"s0", "stub_s0"}, {"s1", "stub_s1"}});
  auto graph = ascir::ShareGraph::RandSeedStoreFusedGraph(2, af::DT_FLOAT);

  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    codegen::Codegen code_generator(codegen::CodegenOptions{});
    std::vector<::ascir::ScheduledResult> schedules;
    ascir::FusedScheduledResult fused_result;
    fused_result.node_idx_to_scheduled_results.push_back(schedules);
    ASSERT_EQ(optimizer.Optimize(graph, fused_result), 0);
    codegen::CodegenResult result;
    ASSERT_EQ(code_generator.Generate(shape_info, fused_result, result), 0);
    EXPECT_NE(result.kernel.find("AscendC::Reg::Arange"), std::string::npos);
    EXPECT_NE(result.kernel.find("RandSeedOffset("), std::string::npos);
  } catch (...) {
    FAIL() << "rand element offset codegen failed";
  }
}
