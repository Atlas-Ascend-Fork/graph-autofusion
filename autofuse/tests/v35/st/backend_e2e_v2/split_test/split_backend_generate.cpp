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
#include "platform_context.h"
#include "runtime_stub.h"
#include "graph/utils/graph_utils.h"

class TestBackendSplitE2e : public testing::Test {
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

  static void CreateSplitAscGraph(af::AscGraph &graph, const std::vector<std::string> &dim_sizes,
                                  af::DataType dtype = af::DT_INT32) {
    af::ascir_op::Data x0("split_data0", graph);
    x0.ir_attr.SetIndex(0);
    x0.y.dtype = dtype;

    af::ascir_op::Load load_0("load0");
    load_0.x = x0.y;
    load_0.y.dtype = dtype;

    af::ascir_op::Split split("split");
    split.InstanceOutputy(2U);  // 需要先指定输出个数
    split.x = load_0.y;

    split.y[0].dtype = dtype;
    split.y[1].dtype = dtype;

    af::ascir_op::Store x_out("store");
    x_out.x = split.y[0];
    x_out.y.dtype = dtype;

    af::ascir_op::Store x_out2("store1");
    x_out2.x = split.y[1];
    x_out2.y.dtype = dtype;

    af::ascir_op::Output y("output");
    y.x = x_out.y;
    y.ir_attr.SetIndex(0);
    y.y.dtype = dtype;

    af::ascir_op::Output y2("output1");
    y2.x = x_out2.y;
    y2.ir_attr.SetIndex(1);
    y2.y.dtype = dtype;

    ConstructVVAscGraphAxisInfo(graph, dim_sizes);
    auto split_node = graph.FindNode("split");
    auto size = split_node->attr.sched.axis.size();
    auto repeats = split_node->outputs()[0]->attr.repeats;
    repeats[size - 1] = repeats[size - 1] + repeats[size - 1];
    auto strides = split_node->outputs()[0]->attr.strides;
    for (int i = dim_sizes.size() - 2; i >= 0; i--) {
      strides[i] = af::sym::Mul(repeats[i + 1], strides[i + 1]);
    }
    auto repeat_string = af::ToString(repeats).c_str();

    auto store_node = graph.FindNode("load0");
    store_node->outputs()[0]->attr.strides = strides;
    store_node->outputs()[0]->attr.repeats = repeats;

    auto data_node = graph.FindNode("split_data0");
    data_node->outputs()[0]->attr.strides = strides;
    data_node->outputs()[0]->attr.repeats = repeats;
  }

  static void ConstructVVAscGraphAxisInfo(af::AscGraph &graph, const std::vector<std::string> &dim_sizes) {
    std::vector<int64_t> axis;
    std::vector<af::Expression> repeats;
    std::vector<af::Expression> strides;
    auto ONE = af::Symbol(1);

    // 构造符号、轴信息
    for (size_t i = 0; i < dim_sizes.size(); i++) {
      af::Expression sym_s;
      if (dim_sizes[i][0] == 's') {
        sym_s = graph.CreateSizeVar(dim_sizes[i]);
      } else {
        sym_s = af::Symbol(std::atoi(dim_sizes[i].c_str()));
      }
      std::string sym_str = "s" + std::to_string(i);
      std::string axis_str = "z" + std::to_string(i);
      auto aixs_z = graph.CreateAxis(axis_str.c_str(), sym_s);
      axis.push_back(aixs_z.id);
      repeats.push_back(sym_s);
      strides.push_back(ONE);
    }
    // 计算每个轴的stride
    for (int32_t i = static_cast<int32_t>(dim_sizes.size()) - 2; i >= 0; i--) {
      strides[i] = af::sym::Mul(repeats[i + 1], strides[i + 1]);
    }
    // 将原始轴信息设置到图中所有节点上
    for (auto node : graph.GetAllNodes()) {
      node->attr.sched.axis = axis;
      printf("node name is %s\n", node->GetNamePtr());
      auto name_string = node->GetNamePtr();
      auto node_output_size = node->outputs().size();
      for (auto output_attr : node->outputs()) {
        output_attr->attr.axis = axis;
        output_attr->attr.repeats = repeats;
        output_attr->attr.strides = strides;
      }
    }
  }
};

TEST_F(TestBackendSplitE2e, SplitAllAligned) {
  bool gen_success = true;
  af::AscGraph graph("split_v2_test");
  CreateSplitAscGraph(graph, {"s0", "32"}, af::DT_INT8);
  std::map<std::string, std::string> shape_info({{"s0", "stub_s0"}});
  std::cout << "KERNEL_SRC_LIST=" << KERNEL_SRC_LIST << std::endl;
  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
  // dlog_setlevel(0, 0, 1);
  // setenv("DUMP_GE_GRAPH", "1", 1);
  // setenv("DUMP_GRAPH_LEVEL", "1", 1);
  // setenv("DUMP_GRAPH_PATH", "./", 1);

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
        "constexpr SplitTilingAllAligned<2> split_tiling {\n"
        "  .src_col_size = 64,\n"
        "  .dst_col_sizes = { 32, 32, },\n"
        "  .src_offsets = { 0, 32, },\n"
        "};\n";
    EXPECT_TRUE(kernel.find(expected) != std::string::npos);
    expected = "SplitAllAligned<int8_t, 2>(";
    EXPECT_TRUE(kernel.find(expected) != std::string::npos);
  } catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}

TEST_F(TestBackendSplitE2e, SplitNotAllAligned_B64) {
  //   dlog_setlevel(0, 0, 1);
  //   setenv("DUMP_GE_GRAPH", "1", 1);
  //   setenv("DUMP_GRAPH_LEVEL", "1", 1);
  //   setenv("DUMP_GRAPH_PATH", "./", 1);
  bool gen_success = true;
  af::AscGraph graph("split_v2_test");
  CreateSplitAscGraph(graph, {"s0", "s1"}, af::DT_INT64);
  std::map<std::string, std::string> shape_info({{"s0", "stub_s0"}, {"s1", "stub_s1"}});
  std::cout << "KERNEL_SRC_LIST=" << KERNEL_SRC_LIST << std::endl;
  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');

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

    std::string expected = "const split::SplitTiling<2> split_tiling {\n";
    EXPECT_TRUE(kernel.find(expected) != std::string::npos);
    expected = "split::SplitExtend<uint32_t, 2>((__ubuf__ uint32_t *)";
    EXPECT_TRUE(kernel.find(expected) != std::string::npos);
  } catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}

TEST_F(TestBackendSplitE2e, SplitNotAllAligned_B8) {
  bool gen_success = true;
  af::AscGraph graph("split_v2_test");
  CreateSplitAscGraph(graph, {"s0", "s1"}, af::DT_INT8);
  std::map<std::string, std::string> shape_info({{"s0", "stub_s0"}, {"s1", "stub_s1"}});
  std::cout << "KERNEL_SRC_LIST=" << KERNEL_SRC_LIST << std::endl;
  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');

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
    std::string expected = "const split::SplitTiling<2> split_tiling {\n";
    EXPECT_TRUE(kernel.find(expected) != std::string::npos);
    expected = "split::SplitExtend<int8_t, 2>((__ubuf__ int8_t *)";
    EXPECT_TRUE(kernel.find(expected) != std::string::npos);
  } catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}

TEST_F(TestBackendSplitE2e, SplitNotAllAligned_B8ToB16) {
  bool gen_success = true;
  af::AscGraph graph("split_v2_test");
  CreateSplitAscGraph(graph, {"s0", "14"}, af::DT_INT8);
  std::map<std::string, std::string> shape_info({{"s0", "stub_s0"}});
  std::cout << "KERNEL_SRC_LIST=" << KERNEL_SRC_LIST << std::endl;
  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');

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
        "const split::SplitTiling<2> split_tiling {\n"
        "  .num_rows = static_cast<uint32_t>(z0t_actual_size), \n"
        "  .num_src_cols = 14, \n"
        "  .num_dsts_cols = {7, 7, }\n"
        "};\n";
    EXPECT_TRUE(kernel.find(expected) != std::string::npos);
    expected = "split::SplitExtend<uint16_t, 2>((__ubuf__ uint16_t *)";
    EXPECT_TRUE(kernel.find(expected) != std::string::npos);
  } catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}

TEST_F(TestBackendSplitE2e, SplitNotAllAligned) {
  bool gen_success = true;
  std::string tilig_stub = R"(
 #define REGISTER_TILING_DEFAULT(tiling)
 #define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
 )";

  af::AscGraph graph("split_v2_test");
  CreateSplitAscGraph(graph, {"s0", "s1"});
  std::map<std::string, std::string> shape_info({{"s0", "stub_s0"}, {"s1", "stub_s1"}});
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
    kernel_file << tilig_stub << RemoveSubDirInclude(result.kernel);
    tiling_file << result.tiling;
    tiling_data_file << result.tiling_data;
  } catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}
