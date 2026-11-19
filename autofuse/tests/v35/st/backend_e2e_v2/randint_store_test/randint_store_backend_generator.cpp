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

class TestBackendRandIntStoreE2e : public testing::Test {
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

TEST_F(TestBackendRandIntStoreE2e, RandIntStoreE2eCodegen) {
  bool gen_success = true;
  constexpr int64_t kRandIntSimtDcacheSize = 40 * 1024;
  std::string rand_int_tiling_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";

  std::map<std::string, std::string> rand_int_shape_info({{"s0", "stub_s0"}, {"s1", "stub_s1"}});
  auto graph = ascir::ShareGraph::RandIntStoreFusedGraph(2);
  std::vector<std::string> rand_int_parts = splitString(KERNEL_SRC_LIST, ':');
  const std::string &kernel_src_file_name = rand_int_parts[0];
  const std::string &tiling_src_file_name = rand_int_parts[1];
  const std::string &tiling_data_src_file_name = rand_int_parts[2];

  try {
    optimize::Optimizer rand_int_optimizer(optimize::OptimizerOptions{});
    codegen::Codegen rand_int_codegen(codegen::CodegenOptions{});
    std::fstream rand_int_kernel_stream(kernel_src_file_name, std::ios::out);
    std::fstream rand_int_tiling_stream(tiling_src_file_name, std::ios::out);
    std::fstream rand_int_data_stream(tiling_data_src_file_name, std::ios::out);
    std::vector<::ascir::ScheduledResult> rand_int_schedules;
    ascir::FusedScheduledResult rand_int_fused_result;
    rand_int_fused_result.node_idx_to_scheduled_results.push_back(rand_int_schedules);
    EXPECT_EQ(rand_int_optimizer.Optimize(graph, rand_int_fused_result), 0);
    codegen::CodegenResult rand_int_codegen_result;
    EXPECT_EQ(rand_int_codegen.Generate(rand_int_shape_info, rand_int_fused_result, rand_int_codegen_result), 0);
    EXPECT_NE(rand_int_codegen_result.kernel.find("RandInt("), std::string::npos);
    EXPECT_NE(rand_int_codegen_result.kernel.find("RandIntSimtCompute"), std::string::npos);
    EXPECT_NE(rand_int_codegen_result.kernel.find("simt_api/cpp/kernel_simt_intf.h"), std::string::npos);
    EXPECT_NE(rand_int_codegen_result.kernel.find("tmp_buf_"), std::string::npos);
    EXPECT_NE(rand_int_codegen_result.tiling.find(std::to_string(kRandIntSimtDcacheSize)), std::string::npos);
    // seed 标量路径指纹：ub_scalar 取值行存在，标量变量喂给 RandInt 调用行
    EXPECT_NE(rand_int_codegen_result.kernel.find("GetValue(0)"), std::string::npos);
    EXPECT_NE(rand_int_codegen_result.kernel.find("_ub_scalar"), std::string::npos);
    rand_int_kernel_stream << rand_int_tiling_stub << RemoveSubDirInclude(rand_int_codegen_result.kernel);
    rand_int_tiling_stream << rand_int_codegen_result.tiling;
    rand_int_data_stream << rand_int_codegen_result.tiling_data;
  } catch (...) {
    gen_success = false;
  }
  EXPECT_EQ(gen_success, true);
}
