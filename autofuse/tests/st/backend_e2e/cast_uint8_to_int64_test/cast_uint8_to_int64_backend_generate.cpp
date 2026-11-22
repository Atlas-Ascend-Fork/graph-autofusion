/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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

#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include "common/platform_context.h"

class TestBackendCastUint8ToInt64E2e : public testing::Test {
 protected:
  void SetUp() override {
    // A3(2201) 的 dtype 白名单和 3510/v35 不同，必须显式指定平台才能覆盖 2201 表
    ge::PlatformContext::GetInstance().Reset();
    ge::PlatformContext::GetInstance().SetPlatform("2201");
  }
  void TearDown() override {
    ge::PlatformContext::GetInstance().Reset();
  }
};

TEST_F(TestBackendCastUint8ToInt64E2e, CastUint8ToInt64E2eCodegen) {
  bool gen_success = true;
  std::string tilig_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";

  // shape_info 和 CastUint8ToInt64FusedGraph入参dims_size匹配（个数相同，命名规则为s开头、编号从0开始）
  std::map<std::string, std::string> shape_info({{"s0", "stub_s0"}, {"s1", "stub_s1"}});
  auto graph = ascir::ShareGraph::CastUint8ToInt64FusedGraph(2);
  std::cout << "KERNEL_SRC_LIST=" << KERNEL_SRC_LIST << std::endl;
  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
  std::string kernel_src_file_name = parts[0];       // cast_uint8_to_int64_test_kernel.cpp
  std::string tiling_src_file_name = parts[1];       // cast_uint8_to_int64_test_tiling.cpp
  std::string tiling_data_src_file_name = parts[2];  // autofuse_tiling_data.h

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
    // uint8 -> int64 不在 EnableCastMaskModeOptimize 白名单，必须走非 mask 的一维
    // CastExtend(dst, src, tmp_buf, size) 分支。mask 形态实参更多（带外部 repeat/stride 表达式），
    // 只判断 "CastExtend(" 出现无法区分两者，因此固定调用点为 4 实参。
    const std::string call_prefix = "CastExtend(local_";
    const size_t call_pos = result.kernel.find(call_prefix);
    EXPECT_NE(call_pos, std::string::npos);
    if (call_pos != std::string::npos) {
      const size_t call_end = result.kernel.find(");", call_pos);
      EXPECT_NE(call_end, std::string::npos);
      if (call_end != std::string::npos) {
        const std::string cast_call = result.kernel.substr(call_pos, call_end - call_pos);
        EXPECT_EQ(std::count(cast_call.begin(), cast_call.end(), ','), 3) << "unexpected cast call: " << cast_call;
      }
    }
    kernel_file << tilig_stub << RemoveSubDirInclude(result.kernel);
    tiling_file << result.tiling;
    tiling_data_file << result.tiling_data;
  } catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}
