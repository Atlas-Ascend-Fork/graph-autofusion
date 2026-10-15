/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#include <exception>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <map>
#include <iostream>

#include "backend_common.h"
#include "codegen.h"
#include "common/platform_context.h"
#include "optimize.h"
#include "runtime_stub.h"
#include "share_graph.h"

class TestBackendLoadAiryAiStoreE2e : public testing::Test {
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

TEST_F(TestBackendLoadAiryAiStoreE2e, LoadAiryAiStoreE2eCodegen) {
  bool gen_success = true;
  std::string airy_ai_tiling_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";

  std::map<std::string, std::string> airy_ai_shape_info({{"s0", "stub_s0"}, {"s1", "stub_s1"}});
  auto graph = ascir::ShareGraph::LoadAiryAiStoreFusedGraph(2);
  std::vector<std::string> airy_ai_parts = splitString(KERNEL_SRC_LIST, ':');
  const std::string &kernel_src_file_name = airy_ai_parts[0];
  const std::string &tiling_src_file_name = airy_ai_parts[1];
  const std::string &tiling_data_src_file_name = airy_ai_parts[2];

  try {
    optimize::Optimizer airy_ai_optimizer(optimize::OptimizerOptions{});
    codegen::Codegen airy_ai_codegen(codegen::CodegenOptions{});
    std::fstream airy_ai_kernel_stream(kernel_src_file_name, std::ios::out);
    std::fstream airy_ai_tiling_stream(tiling_src_file_name, std::ios::out);
    std::fstream airy_ai_data_stream(tiling_data_src_file_name, std::ios::out);
    std::vector<::ascir::ScheduledResult> airy_ai_schedules;
    ascir::FusedScheduledResult airy_ai_fused_result;
    airy_ai_fused_result.node_idx_to_scheduled_results.push_back(airy_ai_schedules);
    EXPECT_EQ(airy_ai_optimizer.Optimize(graph, airy_ai_fused_result), 0);
    codegen::CodegenResult airy_ai_result;
    EXPECT_EQ(airy_ai_codegen.Generate(airy_ai_shape_info, airy_ai_fused_result, airy_ai_result), 0);
    EXPECT_NE(airy_ai_result.kernel.find("AiryAiExtend"), std::string::npos);
    EXPECT_NE(airy_ai_result.kernel.find("tmp_buf_"), std::string::npos);
    airy_ai_kernel_stream << airy_ai_tiling_stub << RemoveSubDirInclude(airy_ai_result.kernel);
    airy_ai_tiling_stream << airy_ai_result.tiling;
    airy_ai_data_stream << airy_ai_result.tiling_data;
  } catch (...) {
    gen_success = false;
  }
  EXPECT_EQ(gen_success, true);
}
