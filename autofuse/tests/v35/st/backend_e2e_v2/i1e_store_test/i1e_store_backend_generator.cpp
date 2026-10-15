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
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "backend_common.h"
#include "codegen.h"
#include "common/platform_context.h"
#include "optimize.h"
#include "runtime_stub.h"
#include "share_graph.h"

class TestBackendI1eStoreE2e : public testing::Test {
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

TEST_F(TestBackendI1eStoreE2e, I1eStoreE2eCodegen) {
  bool gen_success = true;
  std::string i1e_tiling_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";

  std::map<std::string, std::string> i1e_shape_info({{"s0", "stub_s0"}, {"s1", "stub_s1"}});
  auto graph = ascir::ShareGraph::I1eStoreFusedGraph(2);
  std::vector<std::string> i1e_parts = splitString(KERNEL_SRC_LIST, ':');
  const std::string &kernel_src_file_name = i1e_parts[0];
  const std::string &tiling_src_file_name = i1e_parts[1];
  const std::string &tiling_data_src_file_name = i1e_parts[2];

  try {
    optimize::Optimizer i1e_optimizer(optimize::OptimizerOptions{});
    codegen::Codegen i1e_codegen(codegen::CodegenOptions{});

    std::fstream i1e_kernel_stream(kernel_src_file_name, std::ios::out);
    std::fstream i1e_tiling_stream(tiling_src_file_name, std::ios::out);
    std::fstream i1e_data_stream(tiling_data_src_file_name, std::ios::out);

    std::vector<::ascir::ScheduledResult> i1e_schedules;
    ascir::FusedScheduledResult i1e_fused_result;
    i1e_fused_result.node_idx_to_scheduled_results.push_back(i1e_schedules);
    EXPECT_EQ(i1e_optimizer.Optimize(graph, i1e_fused_result), 0);
    codegen::CodegenResult i1e_result;
    EXPECT_EQ(i1e_codegen.Generate(i1e_shape_info, i1e_fused_result, i1e_result), 0);
    EXPECT_NE(i1e_result.kernel.find("I1eExtend"), std::string::npos);
    i1e_kernel_stream << i1e_tiling_stub << RemoveSubDirInclude(i1e_result.kernel);
    i1e_tiling_stream << i1e_result.tiling;
    i1e_data_stream << i1e_result.tiling_data;
  } catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}
