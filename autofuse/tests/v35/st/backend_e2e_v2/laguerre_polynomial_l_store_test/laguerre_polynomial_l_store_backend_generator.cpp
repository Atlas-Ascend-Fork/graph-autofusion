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
#include <iostream>
#include <string>
#include <map>
#include <vector>

#include "backend_common.h"
#include "codegen.h"
#include "common/platform_context.h"
#include "optimize.h"
#include "runtime_stub.h"
#include "share_graph.h"

class TestBackendLaguerrePolynomialLStoreE2e : public testing::Test {
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

TEST_F(TestBackendLaguerrePolynomialLStoreE2e, LaguerrePolynomialLStoreE2eCodegen) {
  bool gen_success = true;
  std::string laguerre_tiling_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";

  std::map<std::string, std::string> laguerre_shape_info({{"s0", "stub_s0"}, {"s1", "stub_s1"}});
  auto graph = ascir::ShareGraph::LaguerrePolynomialLStoreFusedGraph(2);
  std::vector<std::string> laguerre_parts = splitString(KERNEL_SRC_LIST, ':');
  const std::string &kernel_src_file_name = laguerre_parts[0];
  const std::string &tiling_src_file_name = laguerre_parts[1];
  const std::string &tiling_data_src_file_name = laguerre_parts[2];

  try {
    optimize::Optimizer laguerre_optimizer(optimize::OptimizerOptions{});
    codegen::Codegen laguerre_codegen(codegen::CodegenOptions{});
    std::fstream laguerre_kernel_stream(kernel_src_file_name, std::ios::out);
    std::fstream laguerre_tiling_stream(tiling_src_file_name, std::ios::out);
    std::fstream laguerre_data_stream(tiling_data_src_file_name, std::ios::out);
    std::vector<::ascir::ScheduledResult> laguerre_schedules;
    ascir::FusedScheduledResult laguerre_fused_result;
    laguerre_fused_result.node_idx_to_scheduled_results.push_back(laguerre_schedules);
    EXPECT_EQ(laguerre_optimizer.Optimize(graph, laguerre_fused_result), 0);
    codegen::CodegenResult laguerre_result;
    EXPECT_EQ(laguerre_codegen.Generate(laguerre_shape_info, laguerre_fused_result, laguerre_result), 0);
    EXPECT_NE(laguerre_result.kernel.find("LaguerrePolynomialLExtend"), std::string::npos);
    laguerre_kernel_stream << laguerre_tiling_stub << RemoveSubDirInclude(laguerre_result.kernel);
    laguerre_tiling_stream << laguerre_result.tiling;
    laguerre_data_stream << laguerre_result.tiling_data;
  } catch (...) {
    gen_success = false;
  }
  EXPECT_EQ(gen_success, true);
}
