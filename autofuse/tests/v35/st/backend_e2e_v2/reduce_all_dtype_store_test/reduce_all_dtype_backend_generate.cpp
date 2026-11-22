/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <map>
#include <string>
#include <vector>

#include "backend_common.h"
#include "codegen.h"
#include "common/platform_context.h"
#include "optimize.h"
#include "runtime_stub.h"
#include "share_graph.h"

namespace {
void GenerateReduceAllStoreCodegen(af::DataType dtype, const std::string &exec_dtype) {
  const std::string tiling_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";
  const std::map<std::string, std::string> shape_info({{"s0", "stub_s0"}, {"s1", "stub_s1"}});

  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  codegen::Codegen codegen(codegen::CodegenOptions{});
  const auto graph = ascir::ShareGraph::LoadAllReduceStoreFusedGraph(dtype);
  std::vector<::ascir::ScheduledResult> schedule_results;
  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  ASSERT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);

  codegen::CodegenResult result;
  ASSERT_EQ(codegen.Generate(shape_info, fused_schedule_result, result), 0);
  const std::string kernel = RemoveSubDirInclude(result.kernel);
  EXPECT_NE(kernel.find("ReduceAllExtend<" + exec_dtype + ", "), std::string::npos) << kernel;
  EXPECT_NE(kernel.find("ReduceInit<" + exec_dtype), std::string::npos) << kernel;
  if (dtype == af::DT_BOOL) {
    EXPECT_NE(kernel.find(".template ReinterpretCast<uint8_t>()"), std::string::npos) << kernel;
  }
  EXPECT_EQ(kernel.find("ReduceAllExtend<bool"), std::string::npos) << kernel;

  const std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
  ASSERT_EQ(parts.size(), 3U);
  std::fstream kernel_file(parts[0], std::ios::out);
  std::fstream tiling_file(parts[1], std::ios::out);
  std::fstream tiling_data_file(parts[2], std::ios::out);
  ASSERT_TRUE(kernel_file.is_open() && tiling_file.is_open() && tiling_data_file.is_open());
  kernel_file << tiling_stub << kernel;
  tiling_file << result.tiling;
  tiling_data_file << result.tiling_data;
  EXPECT_TRUE(kernel_file.good() && tiling_file.good() && tiling_data_file.good());
}
}  // namespace

class TestBackendReduceAllDtypeE2e : public testing::Test {
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

TEST_F(TestBackendReduceAllDtypeE2e, ReduceDtypeCodegen) {
  GenerateReduceAllStoreCodegen(REDUCE_DTYPE, REDUCE_EXEC_DTYPE);
}
