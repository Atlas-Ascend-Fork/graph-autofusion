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
#include <fstream>
#include <exception>
#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "backend_common.h"
#include "codegen.h"
#include "common/platform_context.h"
#include "optimize.h"
#include "runtime_stub.h"
#include "share_graph.h"

class TestBackendFloorDivInt32StoreE2e : public testing::Test {
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

TEST_F(TestBackendFloorDivInt32StoreE2e, FloorDivInt32StoreE2eCodegen) {
  bool gen_success = true;
  std::string floor_div_tiling_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";

  std::map<std::string, std::string> floor_div_shape_info({{"s0", "stub_s0"}, {"s1", "stub_s1"}});
  auto graph = ascir::ShareGraph::FloorDivInt32StoreFusedGraph(2);
  std::vector<std::string> floor_div_parts = splitString(KERNEL_SRC_LIST, ':');
  const std::string &kernel_src_file_name = floor_div_parts[0];
  const std::string &tiling_src_file_name = floor_div_parts[1];
  const std::string &tiling_data_src_file_name = floor_div_parts[2];

  try {
    optimize::Optimizer floor_div_optimizer(optimize::OptimizerOptions{});
    codegen::Codegen floor_div_codegen(codegen::CodegenOptions{});
    std::fstream floor_div_kernel_stream(kernel_src_file_name, std::ios::out);
    std::fstream floor_div_tiling_stream(tiling_src_file_name, std::ios::out);
    std::fstream floor_div_data_stream(tiling_data_src_file_name, std::ios::out);
    std::vector<::ascir::ScheduledResult> floor_div_schedules;
    ascir::FusedScheduledResult floor_div_fused_result;
    floor_div_fused_result.node_idx_to_scheduled_results.push_back(floor_div_schedules);
    EXPECT_EQ(floor_div_optimizer.Optimize(graph, floor_div_fused_result), 0);
    codegen::CodegenResult floor_div_result;
    EXPECT_EQ(floor_div_codegen.Generate(floor_div_shape_info, floor_div_fused_result, floor_div_result), 0);
    EXPECT_NE(floor_div_result.kernel.find("FloorDivExtend"), std::string::npos);
    floor_div_kernel_stream << floor_div_tiling_stub << RemoveSubDirInclude(floor_div_result.kernel);
    floor_div_tiling_stream << floor_div_result.tiling;
    floor_div_data_stream << floor_div_result.tiling_data;
  } catch (...) {
    gen_success = false;
  }
  EXPECT_EQ(gen_success, true);
}
