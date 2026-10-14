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

class TestBackendNextAfterStoreE2e : public testing::Test {
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

TEST_F(TestBackendNextAfterStoreE2e, NextAfterStoreE2eCodegen) {
  bool gen_success = true;
  std::string next_after_tiling_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";

  std::map<std::string, std::string> next_after_shape_info({{"s0", "stub_s0"}, {"s1", "stub_s1"}});
  auto graph = ascir::ShareGraph::NextAfterStoreFusedGraph(2);
  std::vector<std::string> next_after_parts = splitString(KERNEL_SRC_LIST, ':');
  const std::string &kernel_src_file_name = next_after_parts[0];
  const std::string &tiling_src_file_name = next_after_parts[1];
  const std::string &tiling_data_src_file_name = next_after_parts[2];

  try {
    optimize::Optimizer next_after_optimizer(optimize::OptimizerOptions{});
    codegen::Codegen next_after_codegen(codegen::CodegenOptions{});

    std::fstream next_after_kernel_stream(kernel_src_file_name, std::ios::out);
    std::fstream next_after_tiling_stream(tiling_src_file_name, std::ios::out);
    std::fstream next_after_data_stream(tiling_data_src_file_name, std::ios::out);

    std::vector<::ascir::ScheduledResult> next_after_schedules;
    ascir::FusedScheduledResult next_after_fused_result;
    next_after_fused_result.node_idx_to_scheduled_results.push_back(next_after_schedules);
    EXPECT_EQ(next_after_optimizer.Optimize(graph, next_after_fused_result), 0);
    codegen::CodegenResult next_after_result;
    EXPECT_EQ(next_after_codegen.Generate(next_after_shape_info, next_after_fused_result, next_after_result), 0);
    EXPECT_NE(next_after_result.kernel.find("NextAfterExtend"), std::string::npos);
    next_after_kernel_stream << next_after_tiling_stub << RemoveSubDirInclude(next_after_result.kernel);
    next_after_tiling_stream << next_after_result.tiling;
    next_after_data_stream << next_after_result.tiling_data;
  } catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}
