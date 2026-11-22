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

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include "autofuse_config/auto_fuse_config.h"

class TestBackendAddAbsE2e : public testing::Test {};

TEST_F(TestBackendAddAbsE2e, BrcInlineE2eCodegen) {
  setenv("AUTOFUSE_DFX_FLAGS", "--codegen_compile_debug=true", 1);
  bool gen_success = true;
  std::string tilig_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";

  // shape_info 和 AddAbsFusedGraph入参dims_size匹配（个数相同，命名规则为s开头、编号从0开始）
  std::map<std::string, std::string> shape_info({{"s0", "stub_s0"}, {"s1", "stub_s1"}});
  auto graph = ascir::ShareGraph::BrcInlineFusedGraph(2);
  std::cout << "KERNEL_SRC_LIST=" << KERNEL_SRC_LIST << std::endl;
  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
  std::string kernel_src_file_name = parts[0];       // add_abs_test_tiling.cpp
  std::string tiling_src_file_name = parts[1];       // add_abs_test_kernel.cpp
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
    EXPECT_NE(result.tiling.find("Broadcast/Inline"), std::string::npos);
    kernel_file << tilig_stub << RemoveSubDirInclude(result.kernel);
    tiling_file << result.tiling;
    tiling_data_file << result.tiling_data;
  } catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}

TEST_F(TestBackendAddAbsE2e, BrcInlineLoadOrderE2eCodegen) {
  setenv("AUTOFUSE_DFX_FLAGS", "--codegen_compile_debug=true", 1);
  bool gen_success = true;
  std::string tilig_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";

  // load1(大数据行, z0不变, 被外提)与load0(小数据列, z0依赖)——验证拓扑序中load0先于load1时的codegen行为
  std::map<std::string, std::string> shape_info({{"s0", "stub_s0"}, {"s1", "stub_s1"}});
  auto graph = ascir::ShareGraph::BrcInlineLoadOrderFusedGraph(2);

  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    codegen::Codegen codegen(codegen::CodegenOptions{});

    std::fstream kernel_file("brc_inline_load_order_test_kernel.cpp", std::ios::out);
    std::fstream tiling_file("brc_inline_load_order_test_tiling.cpp", std::ios::out);
    std::fstream tiling_data_file("autofuse_load_order_tiling_data.h", std::ios::out);

    std::vector<::ascir::ScheduledResult> schedule_results;
    ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);
    codegen::CodegenResult result;
    EXPECT_EQ(codegen.Generate(shape_info, fused_schedule_result, result), 0);

    // 校验未拆出重复循环：外提Load未前置（SortHoistedLoadsFirst失效）时会生成两个z0Tb循环及重复变量定义
    size_t loop_cnt = 0;
    for (size_t pos = 0; (pos = result.kernel.find("for (int z0Tb", pos)) != std::string::npos; pos++) {
      loop_cnt++;
    }
    EXPECT_EQ(loop_cnt, 1UL);

    kernel_file << tilig_stub << RemoveSubDirInclude(result.kernel);
    tiling_file << result.tiling;
    tiling_data_file << result.tiling_data;
  } catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}
