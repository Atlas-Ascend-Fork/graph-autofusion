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
#include <gtest/gtest.h>
#include <fstream>
#include <map>
#include <iostream>
#include <vector>
#include <string>

#include "backend_common.h"
#include "codegen.h"
#include "common/platform_context.h"
#include "optimize.h"
#include "runtime_stub.h"
#include "share_graph.h"

class TestBackendRemainderInt32StoreE2e : public testing::Test {
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

TEST_F(TestBackendRemainderInt32StoreE2e, RemainderInt32StoreE2eCodegen) {
  bool gen_success = true;
  constexpr int64_t kRemainderSimtDcacheSize = 40 * 1024;
  std::string remainder_tiling_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";

  std::map<std::string, std::string> remainder_shape_info({{"s0", "stub_s0"}, {"s1", "stub_s1"}});
  auto graph = ascir::ShareGraph::RemainderInt32StoreFusedGraph(2);
  std::vector<std::string> remainder_parts = splitString(KERNEL_SRC_LIST, ':');
  const std::string &kernel_src_file_name = remainder_parts[0];
  const std::string &tiling_src_file_name = remainder_parts[1];
  const std::string &tiling_data_src_file_name = remainder_parts[2];

  try {
    optimize::Optimizer remainder_optimizer(optimize::OptimizerOptions{});
    codegen::Codegen remainder_codegen(codegen::CodegenOptions{});
    std::fstream remainder_kernel_stream(kernel_src_file_name, std::ios::out);
    std::fstream remainder_tiling_stream(tiling_src_file_name, std::ios::out);
    std::fstream remainder_data_stream(tiling_data_src_file_name, std::ios::out);
    std::vector<::ascir::ScheduledResult> remainder_schedules;
    ascir::FusedScheduledResult remainder_fused_result;
    remainder_fused_result.node_idx_to_scheduled_results.push_back(remainder_schedules);
    EXPECT_EQ(remainder_optimizer.Optimize(graph, remainder_fused_result), 0);
    codegen::CodegenResult remainder_codegen_result;
    EXPECT_EQ(remainder_codegen.Generate(remainder_shape_info, remainder_fused_result, remainder_codegen_result), 0);
    EXPECT_NE(remainder_codegen_result.kernel.find("RemainderExtend"), std::string::npos);
    EXPECT_NE(remainder_codegen_result.kernel.find("RemainderIntSimtCompute"), std::string::npos);
    EXPECT_NE(remainder_codegen_result.kernel.find("simt_api/cpp/kernel_simt_intf.h"), std::string::npos);
    EXPECT_NE(remainder_codegen_result.kernel.find("tmp_buf_"), std::string::npos);
    EXPECT_NE(remainder_codegen_result.tiling.find(std::to_string(kRemainderSimtDcacheSize)), std::string::npos);
    remainder_kernel_stream << remainder_tiling_stub << RemoveSubDirInclude(remainder_codegen_result.kernel);
    remainder_tiling_stream << remainder_codegen_result.tiling;
    remainder_data_stream << remainder_codegen_result.tiling_data;
  } catch (...) {
    gen_success = false;
  }
  EXPECT_EQ(gen_success, true);
}
