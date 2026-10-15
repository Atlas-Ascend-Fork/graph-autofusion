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
#include <vector>
#include <string>
#include <gtest/gtest.h>

#include "codegen.h"
#include "optimize.h"
#include "backend_common.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "platform_context.h"
#include "runtime_stub.h"
#include "asc_graph_builder.h"

using af::testing::AscGraphBuilder;

class TestBackendConcatE2e : public testing::Test {
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

  static af::Expression ParseDim(const std::string &dim_str) {
    if (dim_str[0] == 's') {
      return af::Symbol(dim_str.c_str());
    }
    return af::Symbol(std::atoi(dim_str.c_str()));
    ;
  }

  static af::AscGraph CreateConcatAscGraph(const std::vector<std::string> &dims, af::DataType dtype) {
    auto s0 = ParseDim(dims[0]);
    auto s1 = ParseDim(dims[1]);
    auto s2 = s1 + s1;
    auto graph = af::testing::AscGraphBuilder("test_graph")
                     .Loops({s0, s2})
                     .Data("data0", 0, dtype)
                     .Load("load0", "data0", {s0, s1}, {s1, af::sym::kSymbolOne})
                     .Data("data1", 1, dtype)
                     .Load("load1", "data1", {s0, s1}, {s1, af::sym::kSymbolOne})
                     .Concat("concat", {"load0", "load1"})
                     .Store("store", "concat")
                     .Output("out", "store")
                     .Build();
    return graph;
  }

  static af::AscGraph CreateConcatAscGraphInputReverted(const std::vector<std::string> &dims, af::DataType dtype) {
    auto s0 = ParseDim(dims[0]);
    auto s1 = ParseDim(dims[1]);
    auto s2 = s1 + s1;
    auto graph = af::testing::AscGraphBuilder("test_graph")
                     .Loops({s0, s2})
                     .Data("data1", 1, dtype)
                     .Load("load1", "data1", {s0, s1}, {s1, af::sym::kSymbolOne})
                     .Data("data0", 0, dtype)
                     .Load("load0", "data0", {s0, s1}, {s1, af::sym::kSymbolOne})
                     .Concat("concat", {"load0", "load1"})
                     .Store("store", "concat")
                     .Output("out", "store")
                     .Build();
    return graph;
  }

  static af::AscGraph CreateOneAxisConcatAscGraph(const std::vector<std::string> &dims, af::DataType dtype) {
    auto s0 = ParseDim(dims[0]);
    auto s1 = ParseDim(dims[1]);
    auto s2 = s1 + s1;
    auto graph = af::testing::AscGraphBuilder("test_graph")
                     .Loops({s2})
                     .Data("data0", 0, dtype)
                     .Load("load0", "data0", {s0}, {af::sym::kSymbolOne})
                     .Data("data1", 1, dtype)
                     .Load("load1", "data1", {s1}, {af::sym::kSymbolOne})
                     .Concat("concat", {"load0", "load1"})
                     .Store("store", "concat")
                     .Output("out", "store")
                     .Build();
    return graph;
  }

  static af::AscGraph CreateConcatAscGraphDiffSymbol(const std::string &name) {
    auto s0 = af::Symbol("s0");
    auto s1 = af::Symbol("s1");
    auto s2 = af::Symbol("s2");
    auto s3 = s1 + s2;
    auto graph = af::testing::AscGraphBuilder(name)
                     .Loops({s0, s3})
                     .Data("data0", 0, af::DT_INT32)
                     .Load("load0", "data0", {s0, s1}, {s1, af::sym::kSymbolOne})
                     .Data("data1", 1, af::DT_INT32)
                     .Load("load1", "data1", {s0, s2}, {s2, af::sym::kSymbolOne})
                     .Neg("neg0", "load0")
                     .Neg("neg1", "load1")
                     .Concat("concat", {"neg0", "neg1"})
                     .Store("store", "concat")
                     .Output("out", "store")
                     .Build();
    return graph;
  }
};

TEST_F(TestBackendConcatE2e, ConcatNotAllAligned) {
  bool gen_success = true;
  std::string tilig_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";

  auto graph = CreateConcatAscGraphDiffSymbol("concat_v2_test");
  std::map<std::string, std::string> shape_info({{"s0", "stub_s0"}, {"s1", "stub_s1"}, {"s2", "stub_s2"}});
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
    std::cout << result.kernel << std::endl;

    kernel_file << tilig_stub << RemoveSubDirInclude(result.kernel);
    tiling_file << result.tiling;
    tiling_data_file << result.tiling_data;
  } catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}

TEST_F(TestBackendConcatE2e, ConcatNotAllAligned_B64) {
  bool gen_success = true;
  af::AscGraph graph = CreateConcatAscGraph({"s0", "s1"}, af::DT_INT64);
  std::map<std::string, std::string> shape_info({{"s0", "stub_s0"}, {"s1", "stub_s1"}});

  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    codegen::Codegen codegen(codegen::CodegenOptions{});

    std::vector<::ascir::ScheduledResult> schedule_results;
    ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);
    codegen::CodegenResult result;
    EXPECT_EQ(codegen.Generate(shape_info, fused_schedule_result, result), 0);
    const auto &kernel = RemoveSubDirInclude(result.kernel);

    std::string expected = "const concat::ConcatTiling<2> concat_tiling {\n";
    EXPECT_TRUE(kernel.find(expected) != std::string::npos);
    expected = "concat::ConcatExtendDyn<uint32_t, 2>((__ubuf__ uint32_t *)";
    EXPECT_TRUE(kernel.find(expected) != std::string::npos);
  } catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}

TEST_F(TestBackendConcatE2e, ConcatNotAllAligned_B8) {
  bool gen_success = true;
  af::AscGraph graph = CreateConcatAscGraphInputReverted({"s0", "s1"}, af::DT_INT8);
  std::map<std::string, std::string> shape_info({{"s0", "stub_s0"}, {"s1", "stub_s1"}});
  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    codegen::Codegen codegen(codegen::CodegenOptions{});

    std::vector<::ascir::ScheduledResult> schedule_results;
    ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);
    codegen::CodegenResult result;
    EXPECT_EQ(codegen.Generate(shape_info, fused_schedule_result, result), 0);
    const auto &kernel = RemoveSubDirInclude(result.kernel);
    std::string expected = "const concat::ConcatTiling<2> concat_tiling {\n";
    EXPECT_TRUE(kernel.find(expected) != std::string::npos);
    expected = "concat::ConcatExtendDyn<uint8_t, 2>((__ubuf__ uint8_t *)";
    EXPECT_TRUE(kernel.find(expected) != std::string::npos);
  } catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}

TEST_F(TestBackendConcatE2e, ConcatNotAllAligned_B8ToB16) {
  bool gen_success = true;
  af::AscGraph graph = CreateConcatAscGraph({"s0", "130"}, af::DT_INT8);
  std::map<std::string, std::string> shape_info({{"s0", "stub_s0"}});
  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    codegen::Codegen codegen(codegen::CodegenOptions{});

    std::vector<::ascir::ScheduledResult> schedule_results;
    ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);
    codegen::CodegenResult result;
    EXPECT_EQ(codegen.Generate(shape_info, fused_schedule_result, result), 0);
    const auto &kernel = RemoveSubDirInclude(result.kernel);

    std::string expected =
        "    const concat::ConcatTiling<2> concat_tiling {\n"
        "      .num_rows = static_cast<uint32_t>(z0t_actual_size),\n"
        "      .num_dst_cols = 130,\n"
        "      .num_srcs_cols = {65, 65, },\n"
        "    };\n";
    EXPECT_TRUE(kernel.find(expected) != std::string::npos);
    expected = "concat::ConcatExtend<uint16_t, 2>((__ubuf__ uint16_t *)";
    EXPECT_TRUE(kernel.find(expected) != std::string::npos);
  } catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}

TEST_F(TestBackendConcatE2e, ConcatAllAligned) {
  bool gen_success = true;
  af::AscGraph graph = CreateConcatAscGraph({"s0", "32"}, af::DT_INT8);
  std::map<std::string, std::string> shape_info({{"s0", "stub_s0"}});
  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    codegen::Codegen codegen(codegen::CodegenOptions{});

    std::vector<::ascir::ScheduledResult> schedule_results;
    ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);
    codegen::CodegenResult result;
    EXPECT_EQ(codegen.Generate(shape_info, fused_schedule_result, result), 0);
    const auto &kernel = RemoveSubDirInclude(result.kernel);
    std::string expected =
        "    constexpr ConcatTilingAllAligned<2> concat_tiling {\n"
        "      .dst_col_size = 64,\n"
        "      .src_col_sizes = { 32, 32, },\n"
        "      .dst_offsets = { 0, 32, },\n"
        "    };\n";
    EXPECT_TRUE(kernel.find(expected) != std::string::npos);
    expected = "ConcatAllAligned<int8_t, 2>(";
    EXPECT_TRUE(kernel.find(expected) != std::string::npos);
  } catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}

TEST_F(TestBackendConcatE2e, ConcatOneAxis) {
  bool gen_success = true;
  af::AscGraph graph = CreateOneAxisConcatAscGraph({"1", "2"}, af::DT_FLOAT);
  std::map<std::string, std::string> shape_info({{"s0", "stub_s0"}});
  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    codegen::Codegen codegen(codegen::CodegenOptions{});

    std::vector<::ascir::ScheduledResult> schedule_results;
    ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);
    codegen::CodegenResult result;
    EXPECT_EQ(codegen.Generate(shape_info, fused_schedule_result, result), 0);
    const auto &kernel = RemoveSubDirInclude(result.kernel);

    std::string expected = "concat::ConcatOneAxis";
    EXPECT_TRUE(kernel.find(expected) != std::string::npos);
  } catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}
