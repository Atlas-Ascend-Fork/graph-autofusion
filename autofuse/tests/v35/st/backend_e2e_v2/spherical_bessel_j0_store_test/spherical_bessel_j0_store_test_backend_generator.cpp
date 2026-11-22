/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>

#include "common/platform_context.h"
#include "runtime_stub.h"
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "backend_common.h"
#include "codegen.h"
#include "optimize.h"
#include "share_graph.h"

using GraphFactory = af::ComputeGraphPtr (*)(size_t);

inline bool GenerateSpecialStoreKernel(GraphFactory graph_factory, const std::string &api_name,
                                       bool has_tmp_buf = false) {
  std::string tiling_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";
  std::map<std::string, std::string> shape_info({{"s0", "stub_s0"}, {"s1", "stub_s1"}});
  auto graph = graph_factory(2);
  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');

  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    codegen::Codegen codegen(codegen::CodegenOptions{});
    std::vector<::ascir::ScheduledResult> schedules;
    ascir::FusedScheduledResult fused_result;
    fused_result.node_idx_to_scheduled_results.push_back(schedules);
    EXPECT_EQ(optimizer.Optimize(graph, fused_result), 0);
    codegen::CodegenResult result;
    EXPECT_EQ(codegen.Generate(shape_info, fused_result, result), 0);
    EXPECT_NE(result.kernel.find(api_name), std::string::npos);
    if (has_tmp_buf) {
      EXPECT_NE(result.kernel.find("tmp_buf_"), std::string::npos);
    }

    std::fstream kernel_stream(parts[0], std::ios::out);
    std::fstream tiling_stream(parts[1], std::ios::out);
    std::fstream data_stream(parts[2], std::ios::out);
    kernel_stream << tiling_stub << RemoveSubDirInclude(result.kernel);
    tiling_stream << result.tiling;
    data_stream << result.tiling_data;
  } catch (...) {
    return false;
  }
  return true;
}

class TestBackendSphericalBesselJ0StoreE2e : public testing::Test {
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

TEST_F(TestBackendSphericalBesselJ0StoreE2e, SphericalBesselJ0StoreE2eCodegen) {
  EXPECT_TRUE(
      GenerateSpecialStoreKernel(ascir::ShareGraph::SphericalBesselJ0StoreFusedGraph, "SphericalBesselJ0Extend"));
}
