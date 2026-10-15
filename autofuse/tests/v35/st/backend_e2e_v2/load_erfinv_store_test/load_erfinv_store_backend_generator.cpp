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

#include <fstream>
#include <exception>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <iostream>
#include <map>

#include "backend_common.h"
#include "codegen.h"
#include "common/platform_context.h"
#include "optimize.h"
#include "runtime_stub.h"
#include "share_graph.h"

class TestBackendLoadErfinvStoreE2e : public testing::Test {
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

TEST_F(TestBackendLoadErfinvStoreE2e, LoadErfinvStoreE2eCodegen) {
  bool gen_success = true;
  std::string erfinv_tiling_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";

  std::map<std::string, std::string> erfinv_shape_info({{"s0", "stub_s0"}, {"s1", "stub_s1"}});
  auto graph = ascir::ShareGraph::LoadErfinvStoreFusedGraph(2);
  std::vector<std::string> erfinv_parts = splitString(KERNEL_SRC_LIST, ':');
  const std::string &kernel_src_file_name = erfinv_parts[0];
  const std::string &tiling_src_file_name = erfinv_parts[1];
  const std::string &tiling_data_src_file_name = erfinv_parts[2];

  try {
    optimize::Optimizer erfinv_optimizer(optimize::OptimizerOptions{});
    codegen::Codegen erfinv_codegen(codegen::CodegenOptions{});
    std::fstream erfinv_kernel_stream(kernel_src_file_name, std::ios::out);
    std::fstream erfinv_tiling_stream(tiling_src_file_name, std::ios::out);
    std::fstream erfinv_data_stream(tiling_data_src_file_name, std::ios::out);
    std::vector<::ascir::ScheduledResult> erfinv_schedules;
    ascir::FusedScheduledResult erfinv_fused_result;
    erfinv_fused_result.node_idx_to_scheduled_results.push_back(erfinv_schedules);
    EXPECT_EQ(erfinv_optimizer.Optimize(graph, erfinv_fused_result), 0);
    codegen::CodegenResult erfinv_result;
    EXPECT_EQ(erfinv_codegen.Generate(erfinv_shape_info, erfinv_fused_result, erfinv_result), 0);
    EXPECT_NE(erfinv_result.kernel.find("Erfinv"), std::string::npos);
    const auto erfinv_call_begin = erfinv_result.kernel.find("Erfinv(local_");
    ASSERT_NE(erfinv_call_begin, std::string::npos);
    const auto erfinv_call_end = erfinv_result.kernel.find(';', erfinv_call_begin);
    ASSERT_NE(erfinv_call_end, std::string::npos);
    // Store may need scratch even though the Erfinv call does not.
    EXPECT_EQ(erfinv_result.kernel.substr(erfinv_call_begin, erfinv_call_end - erfinv_call_begin).find("tmp_buf_"),
              std::string::npos);
    erfinv_kernel_stream << erfinv_tiling_stub << RemoveSubDirInclude(erfinv_result.kernel);
    erfinv_tiling_stream << erfinv_result.tiling;
    erfinv_data_stream << erfinv_result.tiling_data;
  } catch (...) {
    gen_success = false;
  }
  EXPECT_EQ(gen_success, true);
}
