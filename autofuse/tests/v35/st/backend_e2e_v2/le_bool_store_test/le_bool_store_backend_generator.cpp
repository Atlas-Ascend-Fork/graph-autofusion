/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <gtest/gtest.h>
#include <exception>
#include <filesystem>
#include "codegen.h"
#include "optimize.h"
#include "share_graph.h"
#include "backend_common.h"

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include "runtime_stub.h"
#include "common/platform_context.h"

class TestBackendLoadLeBoolStoreE2e : public testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v2 = std::make_shared<af::RuntimeStubV2>();
    ge::RuntimeStub::SetInstance(stub_v2);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::RuntimeStub::Reset();
  }
};

TEST_F(TestBackendLoadLeBoolStoreE2e, LoadLeBoolStoreE2eCodegen) {
  bool gen_success = true;
  std::string tiling_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";

  // shape_info 和 LoadLeBoolStoreFusedGraph 入参 dims_size 匹配（个数相同，命名规则为 s 开头、编号从 0 开始）
  std::map<std::string, std::string> shape_info({{"s0", "stub_s0"}, {"s1", "stub_s1"}});
  auto graph = ascir::ShareGraph::LoadLeBoolStoreFusedGraph(2);
  std::cout << "KERNEL_SRC_LIST=" << KERNEL_SRC_LIST << std::endl;
  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
  const std::string &kernel_src_file_name = parts[0];
  const std::string &tiling_src_file_name = parts[1];
  const std::string &tiling_data_src_file_name = parts[2];

  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    codegen::Codegen codegen(codegen::CodegenOptions{});

    std::fstream kernel_file(kernel_src_file_name, std::ios::out);
    std::fstream tiling_file(tiling_src_file_name, std::ios::out);
    std::fstream tiling_data_file(tiling_data_src_file_name, std::ios::out);

    std::vector<::ascir::ScheduledResult> schedule_results;
    ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);
    codegen::CodegenResult result;
    EXPECT_EQ(codegen.Generate(shape_info, fused_schedule_result, result), 0);
    // BOOL 输入：双 tensor 分支生成 CompareExtend<bool>，scalar 分支生成 CompareScalarExtend<bool> 或
    // CompareExtend<bool>（blk tensor 形态）；调用点均无 ReinterpretCast（regbase 内部按 uint8 执行）
    EXPECT_NE(result.kernel.find("CompareExtend<bool"), std::string::npos);
    EXPECT_TRUE(result.kernel.find("CompareScalarExtend<bool") != std::string::npos ||
                result.kernel.find("CompareExtend<bool") != result.kernel.rfind("CompareExtend<bool"));
    EXPECT_EQ(result.kernel.find(".ReinterpretCast<uint8_t>("), std::string::npos);
    kernel_file << tiling_stub << RemoveSubDirInclude(result.kernel);
    tiling_file << result.tiling;
    tiling_data_file << result.tiling_data;
  } catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}
