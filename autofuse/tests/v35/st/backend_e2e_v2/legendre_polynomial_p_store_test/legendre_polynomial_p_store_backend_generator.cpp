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

#include <gtest/gtest.h>
#include <exception>
#include <fstream>
#include <map>
#include <string>
#include <iostream>
#include <vector>

#include "backend_common.h"
#include "codegen.h"
#include "common/platform_context.h"
#include "optimize.h"
#include "runtime_stub.h"
#include "share_graph.h"

class TestBackendLegendrePolynomialPStoreE2e : public testing::Test {
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

TEST_F(TestBackendLegendrePolynomialPStoreE2e, LegendrePolynomialPStoreE2eCodegen) {
  bool gen_success = true;
  std::string legendre_tiling_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";

  std::map<std::string, std::string> legendre_shape_info({{"s0", "stub_s0"}, {"s1", "stub_s1"}});
  auto graph = ascir::ShareGraph::LegendrePolynomialPStoreFusedGraph(2);
  std::vector<std::string> legendre_parts = splitString(KERNEL_SRC_LIST, ':');
  const std::string &kernel_src_file_name = legendre_parts[0];
  const std::string &tiling_src_file_name = legendre_parts[1];
  const std::string &tiling_data_src_file_name = legendre_parts[2];

  try {
    optimize::Optimizer legendre_optimizer(optimize::OptimizerOptions{});
    codegen::Codegen legendre_codegen(codegen::CodegenOptions{});
    std::fstream legendre_kernel_stream(kernel_src_file_name, std::ios::out);
    std::fstream legendre_tiling_stream(tiling_src_file_name, std::ios::out);
    std::fstream legendre_data_stream(tiling_data_src_file_name, std::ios::out);
    std::vector<::ascir::ScheduledResult> legendre_schedules;
    ascir::FusedScheduledResult legendre_fused_result;
    legendre_fused_result.node_idx_to_scheduled_results.push_back(legendre_schedules);
    EXPECT_EQ(legendre_optimizer.Optimize(graph, legendre_fused_result), 0);
    codegen::CodegenResult legendre_result;
    EXPECT_EQ(legendre_codegen.Generate(legendre_shape_info, legendre_fused_result, legendre_result), 0);
    EXPECT_NE(legendre_result.kernel.find("LegendrePolynomialPExtend"), std::string::npos);
    legendre_kernel_stream << legendre_tiling_stub << RemoveSubDirInclude(legendre_result.kernel);
    legendre_tiling_stream << legendre_result.tiling;
    legendre_data_stream << legendre_result.tiling_data;
  } catch (...) {
    gen_success = false;
  }
  EXPECT_EQ(gen_success, true);
}
