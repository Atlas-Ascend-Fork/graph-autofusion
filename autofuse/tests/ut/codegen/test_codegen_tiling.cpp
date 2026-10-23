/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "autofuse_config/auto_fuse_config.h"
#include "gen_tiling_impl.h"
#define private public
#include "codegen.h"
#include "codegen_tiling.h"
#include "codegen_tiling_cube_wrapper.h"
#include "common_utils.h"
#include "../../att/generator/final_tiling_log_gen.h"
#define protected public
#include "../../att/generator/high_perf_tiling_code_gen_impl.h"
#undef protected
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "schedule_result.h"
#include "runtime_stub.h"
#include "platform_context.h"
#include "../../common/att/soc_perf_test_config.h"
#include "ascgraph_info_complete.h"
#if defined(FALSE)
#pragma push_macro("FALSE")
#undef FALSE
#define AUTOFUSE_TEST_PUSHED_FALSE 1
#endif
#if defined(TRUE)
#pragma push_macro("TRUE")
#undef TRUE
#define AUTOFUSE_TEST_PUSHED_TRUE 1
#endif
#if defined(FAILED)
#pragma push_macro("FAILED")
#undef FAILED
#define AUTOFUSE_TEST_PUSHED_FAILED 1
#endif
#include "transpose/transpose_api_call.h"
#ifdef AUTOFUSE_TEST_PUSHED_FAILED
#pragma pop_macro("FAILED")
#undef AUTOFUSE_TEST_PUSHED_FAILED
#endif
#ifdef AUTOFUSE_TEST_PUSHED_TRUE
#pragma pop_macro("TRUE")
#undef AUTOFUSE_TEST_PUSHED_TRUE
#endif
#ifdef AUTOFUSE_TEST_PUSHED_FALSE
#pragma pop_macro("FALSE")
#undef AUTOFUSE_TEST_PUSHED_FALSE
#endif
#include "optimize/optimize.h"
#include "share_graph.h"
#include "tests/common/inductor_pgo_codegen_test_utils.h"
#include <fstream>
#include <filesystem>

namespace {
std::pair<int, std::string> execute_command(const std::string &command) {
  std::array<char, 128> buffer;
  std::string output;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);

  if (!pipe) {
    throw std::runtime_error("Failed to open pipe");
  }

  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    output += buffer.data();
  }

  return {WEXITSTATUS(pclose(pipe.release())), output};
}

bool CompileCode(const std::string &code, bool append_main = true) {
  std::string cmake_dir = CMAKE_BINARY_DIR;
  // 临时目录
  std::string temp_dir = cmake_dir + "/tests/ut/temp_compile_codegen_tiling";
  std::filesystem::remove_all(temp_dir);
  std::filesystem::create_directories(temp_dir);
  // 生成的 C++ 文件路径
  std::string source_file = temp_dir + "/temp_codegen_infershape.cpp";
  // 生成 C++ 代码
  std::ofstream source_stream(source_file);
  source_stream << code;
  if (append_main) {
    source_stream << R"(
        int main() {
            return 0;
        }
    )";
  }
  source_stream.close();
  // 头文件路径
  std::string ascend_install_path = ASCEND_INSTALL_PATH;
  std::string include_path = "-I" + ascend_install_path + "/include/ ";
  std::string link_path = "-L" + ascend_install_path + "/lib64";
  // 编译代码
  std::string compile_command = "g++ -std=c++17 " + include_path + " " + link_path + " " + source_file + " -lc_sec";
  auto [compile_exit_code, compile_output] = execute_command(compile_command);
  // 清理临时目录
  std::filesystem::remove_all(temp_dir);
  return compile_exit_code == 0;
}

bool CompileAndRunCode(const std::string &code) {
  const std::string temp_dir = std::string(CMAKE_BINARY_DIR) + "/tests/ut/temp_run_codegen_tiling";
  std::filesystem::remove_all(temp_dir);
  std::filesystem::create_directories(temp_dir);
  const std::string source_file = temp_dir + "/temp_codegen_tiling.cpp";
  const std::string binary_file = temp_dir + "/temp_codegen_tiling";

  std::ofstream source_stream(source_file);
  source_stream << code;
  source_stream.close();

  const std::string compile_command = "g++ -std=c++17 " + source_file + " -o " + binary_file;
  const auto [compile_exit_code, compile_output] = execute_command(compile_command);
  if (compile_exit_code != 0) {
    std::filesystem::remove_all(temp_dir);
    return false;
  }

  const auto [run_exit_code, run_output] = execute_command(binary_file);
  std::filesystem::remove_all(temp_dir);
  return run_exit_code == 0;
}

void ExpectSystemHeaders(const std::string &source, const std::vector<std::string> &required,
                         const std::vector<std::string> &forbidden) {
  for (const auto &header : required) {
    EXPECT_NE(source.find("#include <" + header + ">"), std::string::npos) << header;
  }
  for (const auto &header : forbidden) {
    EXPECT_EQ(source.find("#include <" + header + ">"), std::string::npos) << header;
  }
}

std::string GetSplitContent(const std::string &combined, const std::string &key) {
  const std::string begin_marker = "// AUTOFUSE_SPLIT_FILE_BEGIN: " + key + "\n";
  const std::string end_marker = "// AUTOFUSE_SPLIT_FILE_END: " + key + "\n";
  const size_t begin = combined.find(begin_marker);
  if (begin == std::string::npos) {
    return {};
  }
  const size_t content_begin = begin + begin_marker.size();
  const size_t end = combined.find(end_marker, content_begin);
  if (end == std::string::npos) {
    return {};
  }
  return combined.substr(content_begin, end - content_begin);
}

uint64_t StableSourceHash(const std::string &source) {
  uint64_t hash = 1469598103934665603ULL;
  for (const unsigned char ch : source) {
    hash = (hash ^ ch) * 1099511628211ULL;
  }
  return hash;
}

using autofuse::tests::ScopedAutofusePgoFlag;

bool g_soc_codegen_func_called = false;

bool SuccessSocCodegenFunc(const std::string &op_name, const ascir::FusedScheduledResult &fused_schedule_result,
                           std::map<std::string, std::string> &options,
                           std::map<std::string, std::string> &tiling_file_name_to_content, bool is_inductor_scene) {
  (void)op_name;
  (void)fused_schedule_result;
  (void)options;
  (void)tiling_file_name_to_content;
  (void)is_inductor_scene;
  g_soc_codegen_func_called = true;
  return true;
}

class ScopedSocPlatformState {
 public:
  ScopedSocPlatformState() {
    has_state_ = ge::PlatformContext::GetInstance().GetPlatformInfo(saved_info_) == af::SUCCESS;
    std::string npu_arch;
    has_state_ =
        has_state_ && ge::PlatformContext::GetInstance().GetPlatformTarget(npu_arch, saved_soc_) == af::SUCCESS;
  }

  ~ScopedSocPlatformState() {
    if (has_state_) {
      (void)ge::PlatformContext::GetInstance().SetPlatformTarget(saved_info_, saved_soc_);
      return;
    }
    ge::PlatformContext::GetInstance().Reset();
  }

 private:
  bool has_state_{false};
  ge::PlatformInfo saved_info_{};
  std::string saved_soc_;
};
}  // namespace

namespace {

static void CreateElemwiseGraphWithRelu(af::AscGraph &graph) {
  auto n = graph.CreateSizeVar(1);
  auto c = graph.CreateSizeVar(64);
  auto h = graph.CreateSizeVar(56);
  auto w = graph.CreateSizeVar(56);

  auto z_n = graph.CreateAxis("z_n", n);
  auto z_c = graph.CreateAxis("z_c", c);
  auto z_h = graph.CreateAxis("z_h", h);
  auto z_w = graph.CreateAxis("z_w", w);

  af::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0.y.strides = {c * h * w, h * w, w, af::ops::One};
  *data0.y.repeats = {n, c, h, w};
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  load0.x = data0.y;
  *load0.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {c * h * w, h * w, w, af::ops::One};
  *load0.y.repeats = {n, c, h, w};

  af::ascir_op::Relu relu("relu");
  graph.AddNode(relu);
  relu.x = load0.y;
  relu.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  relu.y.dtype = ge::DT_FLOAT;
  *relu.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  *relu.y.repeats = {n, c, h, w};
  *relu.y.strides = {c * h * w, h * w, w, af::ops::One};
  relu.attr.api.compute_type = af::ComputeType::kComputeElewise;

  af::ascir_op::Store store_op("store");
  store_op.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  store_op.x = relu.y;
  *store_op.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {c * h * w, h * w, w, af::ops::One};
  *store_op.y.repeats = {n, c, h, w};

  af::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  auto x1Local = graph.FindNode("data0");
  x1Local->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  x1Local->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  x1Local->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
}

static void VerifyTilingCodeBasic(const std::map<std::string, std::string> &res) {
  auto pos = res.at("tiling_def_and_tiling_const").find("extern \"C\" int64_t FindBestTilingKey");
  ASSERT_NE(pos, std::string::npos);
  auto static_shape_pos =
      res.at("tiling_def_and_tiling_const").find("extern \"C\" bool AutofuseIsStaticShape() {\n  return true;");
  ASSERT_NE(static_shape_pos, std::string::npos);
  auto tiling_func_pos = res.at("tiling_def_and_tiling_const").find("extern \"C\" ge::graphStatus TilingFunc");
  ASSERT_NE(tiling_func_pos, std::string::npos);
  auto get_size_pos = res.at("tiling_def_and_tiling_const").find("extern \"C\" size_t GetTilingDataSize()");
  ASSERT_NE(get_size_pos, std::string::npos);
  auto tiling_data_pos = res.at("tiling_def_and_tiling_const").find("AutofuseTilingData");
  ASSERT_NE(tiling_data_pos, std::string::npos);
}

static void CreateConv2DOffsetBiasGraph(af::AscGraph &conv2d_offset_bias_graph) {
  auto n_ob = conv2d_offset_bias_graph.CreateSizeVar(1);
  auto c_ob = conv2d_offset_bias_graph.CreateSizeVar(64);
  auto h_ob = conv2d_offset_bias_graph.CreateSizeVar(56);
  auto w_ob = conv2d_offset_bias_graph.CreateSizeVar(56);

  auto z_n_ob = conv2d_offset_bias_graph.CreateAxis("z_n", n_ob);
  auto z_c_ob = conv2d_offset_bias_graph.CreateAxis("z_c", c_ob);
  auto z_h_ob = conv2d_offset_bias_graph.CreateAxis("z_h", h_ob);
  auto z_w_ob = conv2d_offset_bias_graph.CreateAxis("z_w", w_ob);

  af::ascir_op::Data data0_ob("data0", conv2d_offset_bias_graph);
  data0_ob.attr.sched.axis = {z_n_ob.id, z_c_ob.id, z_h_ob.id, z_w_ob.id};
  data0_ob.y.dtype = ge::DT_FLOAT16;
  *data0_ob.y.axis = {z_n_ob.id, z_c_ob.id, z_h_ob.id, z_w_ob.id};
  data0_ob.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0_ob.y.strides = {c_ob * h_ob * w_ob, h_ob * w_ob, w_ob, af::ops::One};
  *data0_ob.y.repeats = {n_ob, c_ob, h_ob, w_ob};
  data0_ob.ir_attr.SetIndex(0);

  af::ascir_op::Load load0_ob("load0");
  load0_ob.attr.sched.axis = {z_n_ob.id, z_c_ob.id, z_h_ob.id, z_w_ob.id};
  load0_ob.x = data0_ob.y;
  *load0_ob.y.axis = {z_n_ob.id, z_c_ob.id, z_h_ob.id, z_w_ob.id};
  load0_ob.y.dtype = ge::DT_FLOAT16;
  *load0_ob.y.strides = {c_ob * h_ob * w_ob, h_ob * w_ob, w_ob, af::ops::One};
  *load0_ob.y.repeats = {n_ob, c_ob, h_ob, w_ob};

  af::ascir_op::Data data1_ob("data1", conv2d_offset_bias_graph);
  data1_ob.y.dtype = ge::DT_FLOAT16;
  data1_ob.attr.sched.axis = {z_n_ob.id, z_c_ob.id, z_h_ob.id, z_w_ob.id};
  *data1_ob.y.axis = {z_n_ob.id, z_c_ob.id, z_h_ob.id, z_w_ob.id};
  data1_ob.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data1_ob.y.repeats = {af::ops::One, af::ops::One, af::ops::One, af::ops::One};
  *data1_ob.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero, af::ops::Zero};
  data1_ob.ir_attr.SetIndex(1);

  af::ascir_op::Load load1_ob("load1");
  load1_ob.x = data1_ob.y;
  load1_ob.attr.sched.axis = {z_n_ob.id, z_c_ob.id, z_h_ob.id, z_w_ob.id};
  load1_ob.y.dtype = ge::DT_FLOAT16;
  *load1_ob.y.axis = {z_n_ob.id, z_c_ob.id, z_h_ob.id, z_w_ob.id};
  *load1_ob.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero, af::ops::Zero};
  *load1_ob.y.repeats = {af::ops::One, af::ops::One, af::ops::One, af::ops::One};

  af::ascir_op::Data data2_ob("data2", conv2d_offset_bias_graph);
  data2_ob.y.dtype = ge::DT_FLOAT;
  data2_ob.attr.sched.axis = {z_c_ob.id};
  *data2_ob.y.axis = {z_c_ob.id};
  data2_ob.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data2_ob.y.repeats = {c_ob};
  *data2_ob.y.strides = {af::ops::One};
  data2_ob.ir_attr.SetIndex(2);

  af::ascir_op::Load load2_ob("load2");
  load2_ob.x = data2_ob.y;
  load2_ob.attr.sched.axis = {z_c_ob.id};
  load2_ob.y.dtype = ge::DT_FLOAT;
  *load2_ob.y.axis = {z_c_ob.id};
  *load2_ob.y.strides = {af::ops::One};
  *load2_ob.y.repeats = {c_ob};

  af::ascir_op::Data data3_ob("data3", conv2d_offset_bias_graph);
  data3_ob.y.dtype = ge::DT_FLOAT16;
  data3_ob.attr.sched.axis = {z_c_ob.id};
  *data3_ob.y.axis = {z_c_ob.id};
  data3_ob.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data3_ob.y.repeats = {c_ob};
  *data3_ob.y.strides = {af::ops::One};
  data3_ob.ir_attr.SetIndex(3);

  af::ascir_op::Load load3_ob("load3");
  load3_ob.x = data3_ob.y;
  load3_ob.attr.sched.axis = {z_c_ob.id};
  load3_ob.y.dtype = ge::DT_FLOAT16;
  *load3_ob.y.axis = {z_c_ob.id};
  *load3_ob.y.strides = {af::ops::One};
  *load3_ob.y.repeats = {c_ob};

  af::ascir_op::Conv2DOffsetBias conv2d_offset_bias("conv2d_offset_bias");
  conv2d_offset_bias.attr.sched.axis = {z_n_ob.id, z_c_ob.id, z_h_ob.id, z_w_ob.id};
  conv2d_offset_bias.x = load0_ob.y;
  conv2d_offset_bias.filter = load1_ob.y;
  conv2d_offset_bias.bias = load2_ob.y;
  conv2d_offset_bias.offset_w = load3_ob.y;
  conv2d_offset_bias.y.dtype = ge::DT_FLOAT;
  *conv2d_offset_bias.y.axis = {z_n_ob.id, z_c_ob.id, z_h_ob.id, z_w_ob.id};
  *conv2d_offset_bias.y.repeats = {n_ob, c_ob, h_ob, w_ob};
  *conv2d_offset_bias.y.strides = {c_ob * h_ob * w_ob, h_ob * w_ob, w_ob, af::ops::One};
  conv2d_offset_bias.attr.api.compute_type = af::ComputeType::kComputeCube;
  conv2d_offset_bias.ir_attr.SetStrides({1, 1});
  conv2d_offset_bias.ir_attr.SetPads({1, 1, 1, 1});
  conv2d_offset_bias.ir_attr.SetDilations({1, 1});
  conv2d_offset_bias.ir_attr.SetGroups(1);
  conv2d_offset_bias.ir_attr.SetData_format("NCHW");
  conv2d_offset_bias.ir_attr.SetOffset_x(0);
  conv2d_offset_bias.ir_attr.SetEnable_hf32(false);

  af::ascir_op::Store store_ob("store");
  store_ob.attr.sched.axis = {z_n_ob.id, z_c_ob.id, z_h_ob.id, z_w_ob.id};
  store_ob.x = conv2d_offset_bias.y;
  *store_ob.y.axis = {z_n_ob.id, z_c_ob.id, z_h_ob.id, z_w_ob.id};
  store_ob.y.dtype = ge::DT_FLOAT;
  *store_ob.y.strides = {c_ob * h_ob * w_ob, h_ob * w_ob, w_ob, af::ops::One};
  *store_ob.y.repeats = {n_ob, c_ob, h_ob, w_ob};

  af::ascir_op::Output output_ob("output");
  output_ob.x = store_ob.y;
  output_ob.y.dtype = ge::DT_FLOAT;
  output_ob.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(conv2d_offset_bias_graph);
}

static void CreateElemwiseGraphWithReluDynamic(af::AscGraph &graph) {
  auto n = graph.CreateSizeVar("n");
  auto c = graph.CreateSizeVar("c");
  auto h = graph.CreateSizeVar("h");
  auto w = graph.CreateSizeVar("w");

  auto z_n = graph.CreateAxis("z_n", n);
  auto z_c = graph.CreateAxis("z_c", c);
  auto z_h = graph.CreateAxis("z_h", h);
  auto z_w = graph.CreateAxis("z_w", w);

  af::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0.y.strides = {c * h * w, h * w, w, af::ops::One};
  *data0.y.repeats = {n, c, h, w};
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  load0.x = data0.y;
  *load0.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {c * h * w, h * w, w, af::ops::One};
  *load0.y.repeats = {n, c, h, w};

  af::ascir_op::Relu relu("relu");
  graph.AddNode(relu);
  relu.x = load0.y;
  relu.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  relu.y.dtype = ge::DT_FLOAT;
  *relu.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  *relu.y.repeats = {n, c, h, w};
  *relu.y.strides = {c * h * w, h * w, w, af::ops::One};
  relu.attr.api.compute_type = af::ComputeType::kComputeElewise;

  af::ascir_op::Store store_op("store");
  store_op.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  store_op.x = relu.y;
  *store_op.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {c * h * w, h * w, w, af::ops::One};
  *store_op.y.repeats = {n, c, h, w};

  af::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  auto x1Local = graph.FindNode("data0");
  x1Local->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  x1Local->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  x1Local->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
}

static void CreateMatmulElemwiseDynamicGraph(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0.y.strides = {s1, af::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1, af::ops::One};
  *load0.y.repeats = {s0, s1};

  af::ascir_op::Abs abs("abs");
  graph.AddNode(abs);
  abs.x = load0.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {s0, s1};
  *abs.y.strides = {s1, af::ops::One};
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;

  af::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.attr.sched.axis = {z0.id, z1.id};
  scalar0.ir_attr.SetValue("0");
  scalar0.y.dtype = ge::DT_FLOAT;
  *scalar0.y.axis = {z0.id, z1.id};
  *scalar0.y.repeats = {af::ops::One, af::ops::One};
  *scalar0.y.strides = {af::ops::Zero, af::ops::Zero};

  af::ascir_op::Broadcast broadcast0("broadcast0");
  broadcast0.x = scalar0.y;
  broadcast0.attr.sched.axis = {z0.id, z1.id};
  *broadcast0.y.axis = {z0.id, z1.id};
  broadcast0.y.dtype = ge::DT_FLOAT;
  *broadcast0.y.repeats = {af::ops::One, s1};
  *broadcast0.y.strides = {af::ops::Zero, af::ops::One};

  af::ascir_op::Broadcast broadcast1("broadcast1");
  broadcast1.x = broadcast0.y;
  broadcast1.attr.sched.axis = {z0.id, z1.id};
  *broadcast1.y.axis = {z0.id, z1.id};
  broadcast1.y.dtype = ge::DT_FLOAT;
  *broadcast1.y.repeats = {s0, s1};
  *broadcast1.y.strides = {s1, af::ops::One};

  af::ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id};
  add_op.x1 = abs.y;
  add_op.x2 = broadcast1.y;
  add_op.y.dtype = ge::DT_FLOAT;
  *add_op.y.axis = {z0.id, z1.id};
  *add_op.y.repeats = {s0, s1};
  *add_op.y.strides = {s1, af::ops::One};

  af::ascir_op::Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data1.y.repeats = {af::ops::One, af::ops::One};
  *data1.y.strides = {af::ops::Zero, af::ops::Zero};
  data1.ir_attr.SetIndex(1);

  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {af::ops::Zero, af::ops::Zero};
  *load1.y.repeats = {af::ops::One, af::ops::One};

  af::ascir_op::Broadcast broadcast2("broadcast2");
  broadcast2.x = load1.y;
  broadcast2.attr.sched.axis = {z0.id, z1.id};
  *broadcast2.y.axis = {z0.id, z1.id};
  broadcast2.y.dtype = ge::DT_FLOAT;
  *broadcast2.y.repeats = {af::ops::One, s1};
  *broadcast2.y.strides = {af::ops::Zero, af::ops::One};

  af::ascir_op::Broadcast broadcast3("broadcast3");
  broadcast3.x = broadcast2.y;
  broadcast3.attr.sched.axis = {z0.id, z1.id};
  *broadcast3.y.axis = {z0.id, z1.id};
  broadcast3.y.dtype = ge::DT_FLOAT;
  *broadcast3.y.repeats = {s0, s1};
  *broadcast3.y.strides = {s1, af::ops::One};

  af::ascir_op::Mul mul("mul");
  mul.attr.sched.axis = {z0.id, z1.id};
  mul.x1 = add_op.y;
  mul.x2 = broadcast3.y;
  mul.y.dtype = ge::DT_FLOAT;
  *mul.y.axis = {z0.id, z1.id};
  *mul.y.repeats = {s0, s1};
  *mul.y.strides = {s1, af::ops::One};

  af::ascir_op::Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = mul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1, af::ops::One};
  *store_op.y.repeats = {s0, s1};

  af::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  auto x1Local = graph.FindNode("data0");
  x1Local->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  x1Local->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  x1Local->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
}

static void CreateElemwiseGraphWithMulDynamic(af::AscGraph &graph) {
  auto n = graph.CreateSizeVar("n");
  auto c = graph.CreateSizeVar("c");
  auto h = graph.CreateSizeVar("h");
  auto w = graph.CreateSizeVar("w");

  auto z_n = graph.CreateAxis("z_n", n);
  auto z_c = graph.CreateAxis("z_c", c);
  auto z_h = graph.CreateAxis("z_h", h);
  auto z_w = graph.CreateAxis("z_w", w);

  af::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0.y.strides = {c * h * w, h * w, w, af::ops::One};
  *data0.y.repeats = {n, c, h, w};
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  load0.x = data0.y;
  *load0.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {c * h * w, h * w, w, af::ops::One};
  *load0.y.repeats = {n, c, h, w};

  af::ascir_op::Scalar scalar("scalar", graph);
  scalar.ir_attr.SetValue("2.0");
  scalar.y.dtype = ge::DT_FLOAT;
  *scalar.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  *scalar.y.repeats = {af::ops::One, af::ops::One, af::ops::One, af::ops::One};
  *scalar.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero, af::ops::Zero};

  af::ascir_op::Broadcast broadcast("broadcast");
  broadcast.x = scalar.y;
  broadcast.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  *broadcast.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  broadcast.y.dtype = ge::DT_FLOAT;
  *broadcast.y.repeats = {n, c, h, w};
  *broadcast.y.strides = {c * h * w, h * w, w, af::ops::One};

  af::ascir_op::Mul mul("mul");
  mul.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  mul.x1 = load0.y;
  mul.x2 = broadcast.y;
  mul.y.dtype = ge::DT_FLOAT;
  *mul.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  *mul.y.repeats = {n, c, h, w};
  *mul.y.strides = {c * h * w, h * w, w, af::ops::One};
  mul.attr.api.compute_type = af::ComputeType::kComputeElewise;

  af::ascir_op::Store store_op("store");
  store_op.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  store_op.x = mul.y;
  *store_op.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {c * h * w, h * w, w, af::ops::One};
  *store_op.y.repeats = {n, c, h, w};

  af::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  auto x1Local = graph.FindNode("data0");
  x1Local->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  x1Local->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  x1Local->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
}

static void CreateElemwiseGraphWithAbsAndAddStatic(af::AscGraph &graph) {
  auto n = graph.CreateSizeVar(1);
  auto c = graph.CreateSizeVar(64);
  auto h = graph.CreateSizeVar(56);
  auto w = graph.CreateSizeVar(56);

  auto z_n = graph.CreateAxis("z_n", n);
  auto z_c = graph.CreateAxis("z_c", c);
  auto z_h = graph.CreateAxis("z_h", h);
  auto z_w = graph.CreateAxis("z_w", w);

  af::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0.y.strides = {c * h * w, h * w, w, af::ops::One};
  *data0.y.repeats = {n, c, h, w};
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  load0.x = data0.y;
  *load0.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {c * h * w, h * w, w, af::ops::One};
  *load0.y.repeats = {n, c, h, w};

  af::ascir_op::Abs abs("abs");
  graph.AddNode(abs);
  abs.x = load0.y;
  abs.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  *abs.y.repeats = {n, c, h, w};
  *abs.y.strides = {c * h * w, h * w, w, af::ops::One};
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;

  af::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  scalar0.ir_attr.SetValue("0.1");
  scalar0.y.dtype = ge::DT_FLOAT;
  *scalar0.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  *scalar0.y.repeats = {af::ops::One, af::ops::One, af::ops::One, af::ops::One};
  *scalar0.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero, af::ops::Zero};

  af::ascir_op::Broadcast broadcast0("broadcast0");
  broadcast0.x = scalar0.y;
  broadcast0.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  *broadcast0.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  broadcast0.y.dtype = ge::DT_FLOAT;
  *broadcast0.y.repeats = {n, c, h, w};
  *broadcast0.y.strides = {c * h * w, h * w, w, af::ops::One};

  af::ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  add_op.x1 = abs.y;
  add_op.x2 = broadcast0.y;
  add_op.y.dtype = ge::DT_FLOAT;
  *add_op.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  *add_op.y.repeats = {n, c, h, w};
  *add_op.y.strides = {c * h * w, h * w, w, af::ops::One};

  af::ascir_op::Store store_op("store");
  store_op.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  store_op.x = add_op.y;
  *store_op.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {c * h * w, h * w, w, af::ops::One};
  *store_op.y.repeats = {n, c, h, w};

  af::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  auto x1Local = graph.FindNode("data0");
  x1Local->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  x1Local->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  x1Local->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
}

static void CreateConv2DOffsetGraph(af::AscGraph &conv2d_offset_graph) {
  auto n_o = conv2d_offset_graph.CreateSizeVar(1);
  auto c_o = conv2d_offset_graph.CreateSizeVar(64);
  auto h_o = conv2d_offset_graph.CreateSizeVar(56);
  auto w_o = conv2d_offset_graph.CreateSizeVar(56);

  auto z_n_o = conv2d_offset_graph.CreateAxis("z_n", n_o);
  auto z_c_o = conv2d_offset_graph.CreateAxis("z_c", c_o);
  auto z_h_o = conv2d_offset_graph.CreateAxis("z_h", h_o);
  auto z_w_o = conv2d_offset_graph.CreateAxis("z_w", w_o);

  af::ascir_op::Data data0_o("data0", conv2d_offset_graph);
  data0_o.attr.sched.axis = {z_n_o.id, z_c_o.id, z_h_o.id, z_w_o.id};
  data0_o.y.dtype = ge::DT_FLOAT16;
  *data0_o.y.axis = {z_n_o.id, z_c_o.id, z_h_o.id, z_w_o.id};
  data0_o.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0_o.y.strides = {c_o * h_o * w_o, h_o * w_o, w_o, af::ops::One};
  *data0_o.y.repeats = {n_o, c_o, h_o, w_o};
  data0_o.ir_attr.SetIndex(0);

  af::ascir_op::Load load0_o("load0");
  load0_o.attr.sched.axis = {z_n_o.id, z_c_o.id, z_h_o.id, z_w_o.id};
  load0_o.x = data0_o.y;
  *load0_o.y.axis = {z_n_o.id, z_c_o.id, z_h_o.id, z_w_o.id};
  load0_o.y.dtype = ge::DT_FLOAT16;
  *load0_o.y.strides = {c_o * h_o * w_o, h_o * w_o, w_o, af::ops::One};
  *load0_o.y.repeats = {n_o, c_o, h_o, w_o};

  af::ascir_op::Data data1_o("data1", conv2d_offset_graph);
  data1_o.y.dtype = ge::DT_FLOAT16;
  data1_o.attr.sched.axis = {z_n_o.id, z_c_o.id, z_h_o.id, z_w_o.id};
  *data1_o.y.axis = {z_n_o.id, z_c_o.id, z_h_o.id, z_w_o.id};
  data1_o.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data1_o.y.repeats = {af::ops::One, af::ops::One, af::ops::One, af::ops::One};
  *data1_o.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero, af::ops::Zero};
  data1_o.ir_attr.SetIndex(1);

  af::ascir_op::Load load1_o("load1");
  load1_o.x = data1_o.y;
  load1_o.attr.sched.axis = {z_n_o.id, z_c_o.id, z_h_o.id, z_w_o.id};
  load1_o.y.dtype = ge::DT_FLOAT16;
  *load1_o.y.axis = {z_n_o.id, z_c_o.id, z_h_o.id, z_w_o.id};
  *load1_o.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero, af::ops::Zero};
  *load1_o.y.repeats = {af::ops::One, af::ops::One, af::ops::One, af::ops::One};

  af::ascir_op::Data data2_o("data2", conv2d_offset_graph);
  data2_o.y.dtype = ge::DT_FLOAT16;
  data2_o.attr.sched.axis = {z_c_o.id};
  *data2_o.y.axis = {z_c_o.id};
  data2_o.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data2_o.y.repeats = {c_o};
  *data2_o.y.strides = {af::ops::One};
  data2_o.ir_attr.SetIndex(2);

  af::ascir_op::Load load2_o("load2");
  load2_o.x = data2_o.y;
  load2_o.attr.sched.axis = {z_c_o.id};
  load2_o.y.dtype = ge::DT_FLOAT16;
  *load2_o.y.axis = {z_c_o.id};
  *load2_o.y.strides = {af::ops::One};
  *load2_o.y.repeats = {c_o};

  af::ascir_op::Conv2DOffset conv2d_offset("conv2d_offset");
  conv2d_offset.attr.sched.axis = {z_n_o.id, z_c_o.id, z_h_o.id, z_w_o.id};
  conv2d_offset.x = load0_o.y;
  conv2d_offset.filter = load1_o.y;
  conv2d_offset.offset_w = load2_o.y;
  conv2d_offset.y.dtype = ge::DT_FLOAT;
  *conv2d_offset.y.axis = {z_n_o.id, z_c_o.id, z_h_o.id, z_w_o.id};
  *conv2d_offset.y.repeats = {n_o, c_o, h_o, w_o};
  *conv2d_offset.y.strides = {c_o * h_o * w_o, h_o * w_o, w_o, af::ops::One};
  conv2d_offset.attr.api.compute_type = af::ComputeType::kComputeCube;
  conv2d_offset.ir_attr.SetStrides({1, 1});
  conv2d_offset.ir_attr.SetPads({1, 1, 1, 1});
  conv2d_offset.ir_attr.SetDilations({1, 1});
  conv2d_offset.ir_attr.SetGroups(1);
  conv2d_offset.ir_attr.SetData_format("NCHW");
  conv2d_offset.ir_attr.SetOffset_x(0);
  conv2d_offset.ir_attr.SetEnable_hf32(false);

  af::ascir_op::Store store_o("store");
  store_o.attr.sched.axis = {z_n_o.id, z_c_o.id, z_h_o.id, z_w_o.id};
  store_o.x = conv2d_offset.y;
  *store_o.y.axis = {z_n_o.id, z_c_o.id, z_h_o.id, z_w_o.id};
  store_o.y.dtype = ge::DT_FLOAT;
  *store_o.y.strides = {c_o * h_o * w_o, h_o * w_o, w_o, af::ops::One};
  *store_o.y.repeats = {n_o, c_o, h_o, w_o};

  af::ascir_op::Output output_o("output");
  output_o.x = store_o.y;
  output_o.y.dtype = ge::DT_FLOAT;
  output_o.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(conv2d_offset_graph);
}

static void CreateConv2DGraphWithGroups(af::AscGraph &conv2d_graph) {
  auto n_g = conv2d_graph.CreateSizeVar(1);
  auto c_g = conv2d_graph.CreateSizeVar(64);
  auto h_g = conv2d_graph.CreateSizeVar(56);
  auto w_g = conv2d_graph.CreateSizeVar(56);

  auto z_n_g = conv2d_graph.CreateAxis("z_n", n_g);
  auto z_c_g = conv2d_graph.CreateAxis("z_c", c_g);
  auto z_h_g = conv2d_graph.CreateAxis("z_h", h_g);
  auto z_w_g = conv2d_graph.CreateAxis("z_w", w_g);

  af::ascir_op::Data data0_g("data0", conv2d_graph);
  data0_g.attr.sched.axis = {z_n_g.id, z_c_g.id, z_h_g.id, z_w_g.id};
  data0_g.y.dtype = ge::DT_FLOAT16;
  *data0_g.y.axis = {z_n_g.id, z_c_g.id, z_h_g.id, z_w_g.id};
  data0_g.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0_g.y.strides = {c_g * h_g * w_g, h_g * w_g, w_g, af::ops::One};
  *data0_g.y.repeats = {n_g, c_g, h_g, w_g};
  data0_g.ir_attr.SetIndex(0);

  af::ascir_op::Load load0_g("load0");
  load0_g.attr.sched.axis = {z_n_g.id, z_c_g.id, z_h_g.id, z_w_g.id};
  load0_g.x = data0_g.y;
  *load0_g.y.axis = {z_n_g.id, z_c_g.id, z_h_g.id, z_w_g.id};
  load0_g.y.dtype = ge::DT_FLOAT16;
  *load0_g.y.strides = {c_g * h_g * w_g, h_g * w_g, w_g, af::ops::One};
  *load0_g.y.repeats = {n_g, c_g, h_g, w_g};

  af::ascir_op::Data data1_g("data1", conv2d_graph);
  data1_g.y.dtype = ge::DT_FLOAT16;
  data1_g.attr.sched.axis = {z_n_g.id, z_c_g.id, z_h_g.id, z_w_g.id};
  *data1_g.y.axis = {z_n_g.id, z_c_g.id, z_h_g.id, z_w_g.id};
  data1_g.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data1_g.y.repeats = {af::ops::One, af::ops::One, af::ops::One, af::ops::One};
  *data1_g.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero, af::ops::Zero};
  data1_g.ir_attr.SetIndex(1);

  af::ascir_op::Load load1_g("load1");
  load1_g.x = data1_g.y;
  load1_g.attr.sched.axis = {z_n_g.id, z_c_g.id, z_h_g.id, z_w_g.id};
  load1_g.y.dtype = ge::DT_FLOAT16;
  *load1_g.y.axis = {z_n_g.id, z_c_g.id, z_h_g.id, z_w_g.id};
  *load1_g.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero, af::ops::Zero};
  *load1_g.y.repeats = {af::ops::One, af::ops::One, af::ops::One, af::ops::One};

  af::ascir_op::Conv2D conv2d_g("conv2d");
  conv2d_g.attr.sched.axis = {z_n_g.id, z_c_g.id, z_h_g.id, z_w_g.id};
  conv2d_g.x = load0_g.y;
  conv2d_g.filter = load1_g.y;
  conv2d_g.y.dtype = ge::DT_FLOAT;
  *conv2d_g.y.axis = {z_n_g.id, z_c_g.id, z_h_g.id, z_w_g.id};
  *conv2d_g.y.repeats = {n_g, c_g, h_g, w_g};
  *conv2d_g.y.strides = {c_g * h_g * w_g, h_g * w_g, w_g, af::ops::One};
  conv2d_g.attr.api.compute_type = af::ComputeType::kComputeCube;
  conv2d_g.ir_attr.SetStrides({2, 2});
  conv2d_g.ir_attr.SetPads({1, 1, 1, 1});
  conv2d_g.ir_attr.SetDilations({1, 1});
  conv2d_g.ir_attr.SetGroups(4);
  conv2d_g.ir_attr.SetData_format("NCHW");
  conv2d_g.ir_attr.SetOffset_x(0);
  conv2d_g.ir_attr.SetEnable_hf32(false);

  af::ascir_op::Store store_g("store");
  store_g.attr.sched.axis = {z_n_g.id, z_c_g.id, z_h_g.id, z_w_g.id};
  store_g.x = conv2d_g.y;
  *store_g.y.axis = {z_n_g.id, z_c_g.id, z_h_g.id, z_w_g.id};
  store_g.y.dtype = ge::DT_FLOAT;
  *store_g.y.strides = {c_g * h_g * w_g, h_g * w_g, w_g, af::ops::One};
  *store_g.y.repeats = {n_g, c_g, h_g, w_g};

  af::ascir_op::Output output_g("output");
  output_g.x = store_g.y;
  output_g.y.dtype = ge::DT_FLOAT;
  output_g.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(conv2d_graph);
}

static void CreateConv2DGraphWithDilation(af::AscGraph &conv2d_graph) {
  auto n_d = conv2d_graph.CreateSizeVar(1);
  auto c_d = conv2d_graph.CreateSizeVar(64);
  auto h_d = conv2d_graph.CreateSizeVar(56);
  auto w_d = conv2d_graph.CreateSizeVar(56);

  auto z_n_d = conv2d_graph.CreateAxis("z_n", n_d);
  auto z_c_d = conv2d_graph.CreateAxis("z_c", c_d);
  auto z_h_d = conv2d_graph.CreateAxis("z_h", h_d);
  auto z_w_d = conv2d_graph.CreateAxis("z_w", w_d);

  af::ascir_op::Data data0_d("data0", conv2d_graph);
  data0_d.attr.sched.axis = {z_n_d.id, z_c_d.id, z_h_d.id, z_w_d.id};
  data0_d.y.dtype = ge::DT_FLOAT16;
  *data0_d.y.axis = {z_n_d.id, z_c_d.id, z_h_d.id, z_w_d.id};
  data0_d.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0_d.y.strides = {c_d * h_d * w_d, h_d * w_d, w_d, af::ops::One};
  *data0_d.y.repeats = {n_d, c_d, h_d, w_d};
  data0_d.ir_attr.SetIndex(0);

  af::ascir_op::Load load0_d("load0");
  load0_d.attr.sched.axis = {z_n_d.id, z_c_d.id, z_h_d.id, z_w_d.id};
  load0_d.x = data0_d.y;
  *load0_d.y.axis = {z_n_d.id, z_c_d.id, z_h_d.id, z_w_d.id};
  load0_d.y.dtype = ge::DT_FLOAT16;
  *load0_d.y.strides = {c_d * h_d * w_d, h_d * w_d, w_d, af::ops::One};
  *load0_d.y.repeats = {n_d, c_d, h_d, w_d};

  af::ascir_op::Data data1_d("data1", conv2d_graph);
  data1_d.y.dtype = ge::DT_FLOAT16;
  data1_d.attr.sched.axis = {z_n_d.id, z_c_d.id, z_h_d.id, z_w_d.id};
  *data1_d.y.axis = {z_n_d.id, z_c_d.id, z_h_d.id, z_w_d.id};
  data1_d.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data1_d.y.repeats = {af::ops::One, af::ops::One, af::ops::One, af::ops::One};
  *data1_d.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero, af::ops::Zero};
  data1_d.ir_attr.SetIndex(1);

  af::ascir_op::Load load1_d("load1");
  load1_d.x = data1_d.y;
  load1_d.attr.sched.axis = {z_n_d.id, z_c_d.id, z_h_d.id, z_w_d.id};
  load1_d.y.dtype = ge::DT_FLOAT16;
  *load1_d.y.axis = {z_n_d.id, z_c_d.id, z_h_d.id, z_w_d.id};
  *load1_d.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero, af::ops::Zero};
  *load1_d.y.repeats = {af::ops::One, af::ops::One, af::ops::One, af::ops::One};

  af::ascir_op::Conv2D conv2d_d("conv2d");
  conv2d_d.attr.sched.axis = {z_n_d.id, z_c_d.id, z_h_d.id, z_w_d.id};
  conv2d_d.x = load0_d.y;
  conv2d_d.filter = load1_d.y;
  conv2d_d.y.dtype = ge::DT_FLOAT;
  *conv2d_d.y.axis = {z_n_d.id, z_c_d.id, z_h_d.id, z_w_d.id};
  *conv2d_d.y.repeats = {n_d, c_d, h_d, w_d};
  *conv2d_d.y.strides = {c_d * h_d * w_d, h_d * w_d, w_d, af::ops::One};
  conv2d_d.attr.api.compute_type = af::ComputeType::kComputeCube;
  conv2d_d.ir_attr.SetStrides({1, 1});
  conv2d_d.ir_attr.SetPads({2, 2, 2, 2});
  conv2d_d.ir_attr.SetDilations({2, 2});
  conv2d_d.ir_attr.SetGroups(1);
  conv2d_d.ir_attr.SetData_format("NCHW");
  conv2d_d.ir_attr.SetOffset_x(0);
  conv2d_d.ir_attr.SetEnable_hf32(false);

  af::ascir_op::Store store_d("store");
  store_d.attr.sched.axis = {z_n_d.id, z_c_d.id, z_h_d.id, z_w_d.id};
  store_d.x = conv2d_d.y;
  *store_d.y.axis = {z_n_d.id, z_c_d.id, z_h_d.id, z_w_d.id};
  store_d.y.dtype = ge::DT_FLOAT;
  *store_d.y.strides = {c_d * h_d * w_d, h_d * w_d, w_d, af::ops::One};
  *store_d.y.repeats = {n_d, c_d, h_d, w_d};

  af::ascir_op::Output output_d("output");
  output_d.x = store_d.y;
  output_d.y.dtype = ge::DT_FLOAT;
  output_d.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(conv2d_graph);
}

static void CreateBatchMatmulElemwiseDynamicGraph(af::AscGraph &graph) {
  auto batch = graph.CreateSizeVar("batch");
  auto m = graph.CreateSizeVar("m");
  auto n = graph.CreateSizeVar("n");

  auto z_batch = graph.CreateAxis("z_batch", batch);
  auto z_m = graph.CreateAxis("z_m", m);
  auto z_n = graph.CreateAxis("z_n", n);

  af::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z_batch.id, z_m.id, z_n.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z_batch.id, z_m.id, z_n.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0.y.strides = {m * n, n, af::ops::One};
  *data0.y.repeats = {batch, m, n};
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z_batch.id, z_m.id, z_n.id};
  load0.x = data0.y;
  *load0.y.axis = {z_batch.id, z_m.id, z_n.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {m * n, n, af::ops::One};
  *load0.y.repeats = {batch, m, n};

  af::ascir_op::Relu relu("relu");
  graph.AddNode(relu);
  relu.x = load0.y;
  relu.attr.sched.axis = {z_batch.id, z_m.id, z_n.id};
  relu.y.dtype = ge::DT_FLOAT;
  *relu.y.axis = {z_batch.id, z_m.id, z_n.id};
  *relu.y.repeats = {batch, m, n};
  *relu.y.strides = {m * n, n, af::ops::One};
  relu.attr.api.compute_type = af::ComputeType::kComputeElewise;

  af::ascir_op::Store store_op("store");
  store_op.attr.sched.axis = {z_batch.id, z_m.id, z_n.id};
  store_op.x = relu.y;
  *store_op.y.axis = {z_batch.id, z_m.id, z_n.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {m * n, n, af::ops::One};
  *store_op.y.repeats = {batch, m, n};

  af::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  auto x1Local = graph.FindNode("data0");
  x1Local->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  x1Local->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  x1Local->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
}

// 创建 BatchMatMul 融合图的一路输入（Data+Load），并在 load 存活期间绑定到 matmul 的指定输入
// 注意：不可将 load.y 以返回值形式带出本函数，AscOpOutput 绑定的存储随局部 Load 析构而失效
static void CreateBatchMatmulOperand(af::AscGraph &graph, af::ascir_op::BatchMatMul &matmul, bool bind_x1,
                                     const char *data_name, const char *load_name, int64_t index,
                                     const std::vector<int64_t> &axes, const std::vector<af::Expression> &strides,
                                     const std::vector<af::Expression> &repeats) {
  af::ascir_op::Data data(data_name, graph);
  data.attr.sched.axis = axes;
  data.y.dtype = ge::DT_FLOAT16;
  *data.y.axis = axes;
  data.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data.y.strides = strides;
  *data.y.repeats = repeats;
  data.ir_attr.SetIndex(index);

  af::ascir_op::Load load(load_name);
  load.attr.sched.axis = axes;
  load.x = data.y;
  *load.y.axis = axes;
  load.y.dtype = ge::DT_FLOAT16;
  *load.y.strides = strides;
  *load.y.repeats = repeats;
  if (bind_x1) {
    matmul.x1 = load.y;
  } else {
    matmul.x2 = load.y;
  }
}

// 组装 BatchMatMul 节点及其两路输入，输出布局由后续 elementwise 节点复用
static void CreateBatchMatmulFusedPrefix(af::AscGraph &graph, af::ascir_op::BatchMatMul &matmul) {
  auto batch = graph.CreateSizeVar(2);
  auto m = graph.CreateSizeVar(32);
  auto n = graph.CreateSizeVar(32);
  auto k = graph.CreateSizeVar(32);
  auto z_batch = graph.CreateAxis("z_batch", batch);
  auto z_m = graph.CreateAxis("z_m", m);
  auto z_n = graph.CreateAxis("z_n", n);
  auto z_k = graph.CreateAxis("z_k", k);

  CreateBatchMatmulOperand(graph, matmul, true, "data0", "load0", 0, {z_batch.id, z_m.id, z_k.id},
                           {m * k, k, af::ops::One}, {batch, m, k});
  CreateBatchMatmulOperand(graph, matmul, false, "data1", "load1", 1, {z_batch.id, z_k.id, z_n.id},
                           {k * n, n, af::ops::One}, {batch, k, n});
  matmul.attr.sched.axis = {z_batch.id, z_m.id, z_n.id};
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z_batch.id, z_m.id, z_n.id};
  *matmul.y.repeats = {batch, m, n};
  *matmul.y.strides = {m * n, n, af::ops::One};
  matmul.attr.api.compute_type = af::ComputeType::kComputeCube;
  matmul.ir_attr.SetAdj_x1(0);
  matmul.ir_attr.SetAdj_x2(0);
  matmul.ir_attr.SetHas_relu(0);
  matmul.ir_attr.SetEnable_hf32(0);
  matmul.ir_attr.SetOffset_x(0);
}

// BatchMatMul + elementwise 完整融合图：供 Optimizer 自动做 CV 调度（切图并插入 Workspace）
static void CreateBatchMatmulElemwiseFusedGraph(af::AscGraph &graph) {
  af::ascir_op::BatchMatMul batch_matmul("batch_matmul");
  CreateBatchMatmulFusedPrefix(graph, batch_matmul);

  af::ascir_op::Relu relu("relu");
  relu.attr.sched.axis = batch_matmul.attr.sched.axis;
  relu.x = batch_matmul.y;
  *relu.y.axis = *batch_matmul.y.axis;
  relu.y.dtype = ge::DT_FLOAT;
  *relu.y.repeats = *batch_matmul.y.repeats;
  *relu.y.strides = *batch_matmul.y.strides;
  relu.attr.api.compute_type = af::ComputeType::kComputeElewise;

  af::ascir_op::Store store_op("store");
  store_op.attr.sched.axis = batch_matmul.attr.sched.axis;
  store_op.x = relu.y;
  *store_op.y.axis = *batch_matmul.y.axis;
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = *batch_matmul.y.strides;
  *store_op.y.repeats = *batch_matmul.y.repeats;

  af::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
}

static void VerifyDynamicShapeTiling(const std::map<std::string, std::string> &res) {
  auto pos = res.at("tiling_def_and_tiling_const").find("extern \"C\" int64_t FindBestTilingKey");
  ASSERT_NE(pos, std::string::npos);
  auto dynamic_shape_pos =
      res.at("tiling_def_and_tiling_const").find("extern \"C\" bool AutofuseIsStaticShape() {\n  return false;");
  ASSERT_NE(dynamic_shape_pos, std::string::npos);
  auto tiling_func_pos = res.at("tiling_def_and_tiling_const")
                             .find("extern \"C\" ge::graphStatus TilingFunc(gert::TilingSymbolEvalContext *context)");
  ASSERT_NE(tiling_func_pos, std::string::npos);
  auto tiling_call_pos = res.at("tiling_def_and_tiling_const").find("AutofuseTilingWithConfig");
  ASSERT_NE(tiling_call_pos, std::string::npos);
  auto cache_key_pos =
      res.at("tiling_def_and_tiling_const").find("extern \"C\" ge::graphStatus GetSymbolTilingCacheKey");
  ASSERT_NE(cache_key_pos, std::string::npos);
  auto tiling_data_pos = res.at("tiling_def_and_tiling_const").find("AutofuseTilingData");
  ASSERT_NE(tiling_data_pos, std::string::npos);
}

static void VerifyConv2dElemwiseTiling(const std::map<std::string, std::string> &res) {
  auto pos = res.at("tiling_def_and_tiling_const").find("extern \"C\" int64_t FindBestTilingKey");
  ASSERT_NE(pos, std::string::npos);
  auto static_shape_pos =
      res.at("tiling_def_and_tiling_const").find("extern \"C\" bool AutofuseIsStaticShape() {\n  return true;");
  ASSERT_NE(static_shape_pos, std::string::npos);
  auto tiling_func_pos = res.at("tiling_def_and_tiling_const").find("extern \"C\" ge::graphStatus TilingFunc");
  ASSERT_NE(tiling_func_pos, std::string::npos);
  auto tiling_parse_pos = res.at("tiling_def_and_tiling_const").find("extern \"C\" ge::graphStatus TilingParse");
  ASSERT_NE(tiling_parse_pos, std::string::npos);
  auto get_size_pos = res.at("tiling_def_and_tiling_const").find("extern \"C\" size_t GetTilingDataSize()");
  ASSERT_NE(get_size_pos, std::string::npos);
  auto workspace_pos = res.at("tiling_def_and_tiling_const").find("*context->GetWorkspaceSizes(1) = 16 * 1024 * 1024");
  ASSERT_NE(workspace_pos, std::string::npos);
  auto tiling_data_pos = res.at("tiling_def_and_tiling_const").find("AutofuseTilingData");
  ASSERT_NE(tiling_data_pos, std::string::npos);
  auto block_dim_pos = res.at("tiling_def_and_tiling_const").find("set_block_dim");
  ASSERT_NE(block_dim_pos, std::string::npos);
}

static void VerifyConv2DBiasElemwiseTiling(const std::map<std::string, std::string> &res) {
  auto pos = res.at("tiling_def_and_tiling_const").find("extern \"C\" int64_t FindBestTilingKey");
  ASSERT_NE(pos, std::string::npos);
  auto static_shape_pos =
      res.at("tiling_def_and_tiling_const").find("extern \"C\" bool AutofuseIsStaticShape() {\n  return true;");
  ASSERT_NE(static_shape_pos, std::string::npos);
  auto tiling_func_pos = res.at("tiling_def_and_tiling_const").find("extern \"C\" ge::graphStatus TilingFunc");
  ASSERT_NE(tiling_func_pos, std::string::npos);
  auto get_size_pos = res.at("tiling_def_and_tiling_const").find("extern \"C\" size_t GetTilingDataSize()");
  ASSERT_NE(get_size_pos, std::string::npos);
  auto tiling_data_pos = res.at("tiling_def_and_tiling_const").find("AutofuseTilingData");
  ASSERT_NE(tiling_data_pos, std::string::npos);
  auto workspace_pos = res.at("tiling_def_and_tiling_const").find("*context->GetWorkspaceSizes(1) = 16 * 1024 * 1024");
  ASSERT_NE(workspace_pos, std::string::npos);
}

static void VerifyConv2DOffsetTiling(const std::map<std::string, std::string> &res) {
  auto pos = res.at("tiling_def_and_tiling_const").find("extern \"C\" int64_t FindBestTilingKey");
  ASSERT_NE(pos, std::string::npos);
  auto static_shape_pos =
      res.at("tiling_def_and_tiling_const").find("extern \"C\" bool AutofuseIsStaticShape() {\n  return true;");
  ASSERT_NE(static_shape_pos, std::string::npos);
  auto tiling_func_pos = res.at("tiling_def_and_tiling_const").find("extern \"C\" ge::graphStatus TilingFunc");
  ASSERT_NE(tiling_func_pos, std::string::npos);
  auto get_size_pos = res.at("tiling_def_and_tiling_const").find("extern \"C\" size_t GetTilingDataSize()");
  ASSERT_NE(get_size_pos, std::string::npos);
  auto tiling_data_pos = res.at("tiling_def_and_tiling_const").find("AutofuseTilingData");
  ASSERT_NE(tiling_data_pos, std::string::npos);
}

}  // namespace

void CreateMatmulGraph(af::AscGraph &graph, bool is_dynamic);

class TestCodegenTiling : public testing::Test, public codegen::TilingLib {
 public:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_DEBUG, 0);
    ge::PlatformContext::GetInstance().Reset();
    ASSERT_EQ(
        att::soc_perf_test::SetSocPerfTestPlatform(att::soc_perf_test::kV35NpuArch, att::soc_perf_test::kSocEmpty),
        af::SUCCESS);
    ge::RuntimeStub::SetInstance(std::make_shared<ge::RuntimeStubV2Common>());
  }
  void TearDown() override {
    ge::PlatformContext::GetInstance().Reset();
    ge::RuntimeStub::Reset();
  }

  std::string GenerateMatmulTilingForSoc(const std::shared_ptr<ge::RuntimeStub> &stub, const std::string &platform) {
    ge::PlatformContext::GetInstance().SetPlatform(platform);
    ge::RuntimeStub::SetInstance(stub);

    af::AscGraph graph("matmul_elemwise_pro");
    CreateMatmulElemwiseDynamicGraph(graph);

    af::AscGraph mm_graph("mutmul");
    CreateMatmulGraph(mm_graph, true);
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    ascir::FusedScheduledResult fused_schedule_result;
    EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);

    ascir::ScheduleGroup schedule_group;
    schedule_group.impl_graphs.push_back(mm_graph);
    fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups.push_back(schedule_group);
    fused_schedule_result.node_idx_to_scheduled_results[0][0].cube_type = ascir::CubeTemplateType::kUBFuse;

    std::map<std::string, std::string> shape_info{{"s0", "64"}, {"s1", "64"}};
    return Generate(fused_schedule_result, shape_info, "", "0").at("tiling_def_and_tiling_const");
  }
  void SetupLoadAttrs(af::AscNode &load, uint64_t z0_id, const af::Expression &z0_size) {
    auto &attr = load.outputs[0].attr;
    attr.axis = {static_cast<int64_t>(z0_id)};
    attr.vectorized_axis = {static_cast<int64_t>(z0_id)};
    attr.vectorized_strides = {af::ops::One};
    attr.repeats = {z0_size};
    attr.strides = {af::ops::One};
    attr.mem.position = af::Position::kPositionVecIn;
    attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
    attr.mem.tensor_id = 1;
    attr.que.id = 0;
    attr.mem.reuse_id = 0;
    attr.que.depth = 2;
    attr.que.buf_num = 2;
    attr.opt.merge_scope = af::kIdNone;
  }

  void SetupStoreAttrs(af::AscNode &store, uint64_t z0_id, const af::Expression &z0_size) {
    auto &attr = store.outputs[0].attr;
    attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
    attr.mem.tensor_id = 2;
    attr.axis = {static_cast<int64_t>(z0_id)};
    attr.vectorized_axis = {static_cast<int64_t>(z0_id)};
    attr.vectorized_strides = {af::ops::One};
    attr.repeats = {z0_size};
    attr.strides = {af::ops::One};
  }

  ascir::FusedScheduledResult GenBasicFusedScheduleResult(const std::vector<af::Expression> &origin_vars = {},
                                                          const af::Expression &axis_size = af::ops::Zero) {
    af::AscGraph graph("test_graph");
    auto s0 = graph.CreateSizeVar("s0");
    auto z0 = graph.CreateAxis("z0", axis_size);

    af::ascir_op::Data x_op("x", graph);
    x_op.ir_attr.SetIndex(0);
    af::ascir_op::Load load_op("load");
    af::ascir_op::Store store_op("store");
    af::ascir_op::Output y_op("y");
    y_op.ir_attr.SetIndex(0);
    graph.AddNode(load_op);
    graph.AddNode(store_op);
    graph.AddNode(y_op);

    load_op.x = x_op.y;
    load_op.y.dtype = ge::DT_FLOAT16;
    store_op.x = load_op.y;
    y_op.x = store_op.y;

    auto x = graph.FindNode("x");
    auto load = graph.FindNode("load");
    auto store = graph.FindNode("store");
    auto y = graph.FindNode("y");

    x->outputs[0].attr.dtype = ge::DT_FLOAT16;
    load->outputs[0].attr.dtype = ge::DT_FLOAT16;
    store->outputs[0].attr.dtype = ge::DT_FLOAT16;
    y->outputs[0].attr.dtype = ge::DT_FLOAT16;
    x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
    x->outputs[0].attr.mem.tensor_id = 0;
    x->attr.api.unit = af::ComputeUnit::kUnitNone;
    y->attr.api.unit = af::ComputeUnit::kUnitNone;

    SetupLoadAttrs(*load, z0.id, z0.size);
    SetupStoreAttrs(*store, z0.id, z0.size);

    ::ascir::ScheduledResult schedule_result;
    schedule_result.schedule_groups.resize(1);
    for (auto &schedule_group : schedule_result.schedule_groups) {
      schedule_group.impl_graphs.emplace_back(graph);
    }
    std::vector<ascir::ScheduledResult> schedule_results;
    schedule_results.push_back(schedule_result);
    schedule_results.push_back(schedule_result);

    ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.fused_graph_name = af::AscendString(graph.GetName().c_str());
    fused_schedule_result.input_nodes.push_back(x);
    fused_schedule_result.output_nodes.push_back(y);
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    fused_schedule_result.origin_vars = origin_vars;
    return fused_schedule_result;
  }

  std::map<std::string, std::string> GenTilingCode(const std::vector<af::Expression> &origin_vars = {},
                                                   const std::map<std::string, std::string> &shape_info = {}) {
    auto fused_schedule_result = GenBasicFusedScheduleResult(origin_vars);
    return this->Generate(fused_schedule_result, shape_info, "", "0");
  }

  std::map<std::string, std::string> GenTilingCodeForInductor(const std::vector<af::Expression> &origin_vars = {}) {
    auto fused_schedule_result = GenBasicFusedScheduleResult(origin_vars);
    return this->GenerateForInductor(fused_schedule_result);
  }

  ascir::FusedScheduledResult GenTilingKeyCountResult(const std::vector<std::vector<size_t>> &result_impl_counts,
                                                      const af::Expression &axis_size = af::ops::Zero) {
    auto fused_schedule_result = GenBasicFusedScheduleResult({}, axis_size);
    const auto graph = fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
    std::vector<ascir::ScheduledResult> scheduled_results;
    for (const auto &impl_counts : result_impl_counts) {
      ascir::ScheduledResult scheduled_result;
      for (const auto impl_count : impl_counts) {
        ascir::ScheduleGroup schedule_group;
        schedule_group.impl_graphs.assign(impl_count, graph);
        scheduled_result.schedule_groups.emplace_back(std::move(schedule_group));
      }
      scheduled_results.emplace_back(std::move(scheduled_result));
    }
    fused_schedule_result.node_idx_to_scheduled_results = {std::move(scheduled_results)};
    return fused_schedule_result;
  }

 protected:
  TestCodegenTiling() : codegen::TilingLib("test", "test") {}
};

TEST_F(TestCodegenTiling, DynamicShapeCacheKeyGuardsCapacityAndPreservesSymbolOrder) {
  const auto result = this->GenTilingCode({af::Symbol("s0"), af::Symbol("s1")}, {{"s0", "32"}, {"s1", "64"}});
  const auto &source = result.at("tiling_def_and_tiling_const");

  const auto cache_fn = source.find("extern \"C\" ge::graphStatus GetSymbolTilingCacheKey");
  ASSERT_NE(cache_fn, std::string::npos);
  const auto cache_end = source.find("extern \"C\" ge::graphStatus DfxInputSymbolInfo", cache_fn);
  ASSERT_NE(cache_end, std::string::npos);
  const auto cache_body = source.substr(cache_fn, cache_end - cache_fn);

  const auto capacity_guard = cache_body.find("symbol_src_vec->GetCapacity() < 2");
  const auto first_symbol = cache_body.find("MutableData()[0] = s0");
  const auto second_symbol = cache_body.find("MutableData()[1] = s1");
  const auto set_size = cache_body.find("symbol_src_vec->SetSize(2)");
  ASSERT_NE(capacity_guard, std::string::npos);
  ASSERT_NE(first_symbol, std::string::npos);
  ASSERT_NE(second_symbol, std::string::npos);
  ASSERT_NE(set_size, std::string::npos);
  EXPECT_LT(capacity_guard, first_symbol);
  EXPECT_LT(first_symbol, second_symbol);
  EXPECT_LT(second_symbol, set_size);
}

TEST_F(TestCodegenTiling, GenerateForInductorValidatesSocTargetBeforeTilingCodegen) {
  const ScopedSocPlatformState platform_state_guard;
  ge::PlatformContext::GetInstance().SetPlatform("UnknownArch");
  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen.tiling_lib_.codegen_func_ = SuccessSocCodegenFunc;
  const auto fused_schedule_result = this->GenBasicFusedScheduleResult({}, af::Symbol(64));

  codegen::CodegenResult result;
  g_soc_codegen_func_called = false;
  EXPECT_NE(codegen.GenerateForInductor(fused_schedule_result, result), af::SUCCESS);
  EXPECT_FALSE(g_soc_codegen_func_called);
  EXPECT_TRUE(result.tiling_data.empty());
  EXPECT_TRUE(result.tiling.empty());
  EXPECT_TRUE(result.kernel.empty());

  ge::PlatformInfo platform_info;
  platform_info.soc_ver = "5102";
  platform_info.aiv_num = 1;
  platform_info.ub_size = 262144;
  ASSERT_EQ(ge::PlatformContext::GetInstance().SetPlatformTarget(platform_info, "SOC_EMPTY"), af::SUCCESS);
  result = codegen::CodegenResult{};
  g_soc_codegen_func_called = false;
  EXPECT_EQ(codegen.GenerateForInductor(fused_schedule_result, result), af::SUCCESS);
  EXPECT_TRUE(g_soc_codegen_func_called);
}

TEST_F(TestCodegenTiling, GenerateTilingAcceptsV2ArchitecturesUnknownSocAndLegacyTarget) {
  const ScopedSocPlatformState platform_state_guard;
  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen.tiling_lib_.codegen_func_ = SuccessSocCodegenFunc;
  const auto schedule_result = this->GenBasicFusedScheduleResult({}, af::Symbol(64));
  for (const std::string &npu_arch : {"3510", "5102", "9201", "9202"}) {
    for (const std::string &soc : {std::string("SOC_UNREGISTERED"), std::string()}) {
      ASSERT_EQ(att::soc_perf_test::SetSocPerfTestPlatform(npu_arch, soc), af::SUCCESS);
      if (soc.empty()) {
        ge::PlatformContext::GetInstance().SetPlatform(npu_arch);
      }
      std::map<std::string, std::string> tiling_files;
      g_soc_codegen_func_called = false;
      EXPECT_EQ(codegen.GenerateTiling(schedule_result, {}, "", "0", tiling_files), af::SUCCESS);
      EXPECT_TRUE(g_soc_codegen_func_called);
    }
  }
}

class RuntimeStubWithFullSocName : public ge::RuntimeStubV2Common {
 public:
  const char *aclrtGetSocName() override {
    return "Ascend910_9591";
  }
};

class RuntimeStubWithNullSocName : public ge::RuntimeStubV2Common {
 public:
  const char *aclrtGetSocName() override {
    return nullptr;
  }
};

TEST_F(TestCodegenTiling, NoWorkspaceTest) {
  ascir::ImplGraph graph0("test_graph0");
  graph0.CreateSizeVar("s0");
  graph0.CreateSizeVar("s1");

  std::vector<ascir::ImplGraph> impl_graphs;
  impl_graphs.push_back(graph0);

  std::vector<ascir::ScheduledResult> schedule_results;
  ascir::ScheduledResult schedule_result;
  ascir::ScheduleGroup schedule_group;
  schedule_group.impl_graphs = impl_graphs;
  schedule_result.schedule_groups.push_back(schedule_group);
  schedule_results.push_back(schedule_result);
  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);

  const auto code = this->GenGetWorkspaceSizeFunc("AutofuseTilingData", fused_schedule_result);
  EXPECT_NE(code.find("bool GetWorkspaceSize(const AutofuseTilingData &t, uint64_t &workspace_size) {"),
            std::string::npos);
  EXPECT_NE(code.find("if (!CheckedWorkspaceAdd(ws_size, 0, ws_size))"), std::string::npos);
  EXPECT_NE(code.find("if (!CheckedWorkspaceAlign(ws_size, 512U, ws_size))"), std::string::npos);
  EXPECT_NE(code.find("return true;"), std::string::npos);
}

TEST_F(TestCodegenTiling, GetWorkspaceSizeUsesUint64AndCheckedHelpers) {
  ascir::ImplGraph graph0("test_graph0");
  ascir::ScheduledResult schedule_result;
  ascir::ScheduleGroup schedule_group;
  schedule_group.impl_graphs.push_back(graph0);
  schedule_result.schedule_groups.push_back(schedule_group);
  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.node_idx_to_scheduled_results.push_back({schedule_result});

  const auto code = this->GenGetWorkspaceSizeFunc("AutofuseTilingData", fused_schedule_result);

  EXPECT_NE(code.find("bool GetWorkspaceSize(const AutofuseTilingData &t, uint64_t &workspace_size) {"),
            std::string::npos);
  EXPECT_NE(code.find("uint64_t ws_size = 0;"), std::string::npos);
  EXPECT_NE(code.find("inline bool CheckedWorkspaceAdd("), std::string::npos);
  EXPECT_NE(code.find("inline bool CheckedWorkspaceAlign("), std::string::npos);
  EXPECT_NE(code.find("if (!CheckedWorkspaceAlign(ws_size, 512U, ws_size)) {"), std::string::npos);
  EXPECT_NE(code.find("workspace_size = ws_size;"), std::string::npos);
  EXPECT_EQ(code.find("uint32_t GetWorkspaceSize("), std::string::npos);
  EXPECT_EQ(code.find("uint32_t ws_size"), std::string::npos);
}

TEST_F(TestCodegenTiling, DirectTilingAbiUsesUint64WorkspaceSize) {
  ascir::ImplGraph graph0("test_graph0");
  ascir::ScheduledResult schedule_result;
  ascir::ScheduleGroup schedule_group;
  schedule_group.impl_graphs.push_back(graph0);
  schedule_result.schedule_groups.push_back(schedule_group);
  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.node_idx_to_scheduled_results.push_back({schedule_result});

  const auto inductor_code = this->GenTilingFuncForInductor(fused_schedule_result, fused_schedule_result,
                                                            "AutofuseTiling", "AutofuseTilingData");
  const auto direct_code = this->GenTilingFunc({}, fused_schedule_result, "AutofuseTiling", "AutofuseTilingData", "48");

  EXPECT_NE(inductor_code.find("uint64_t* workspaceSize"), std::string::npos);
  EXPECT_EQ(inductor_code.find("uint32_t* workspaceSize"), std::string::npos);
  EXPECT_NE(direct_code.find("uint64_t* workspaceSize"), std::string::npos);
  EXPECT_EQ(direct_code.find("uint32_t* workspaceSize"), std::string::npos);
}

TEST_F(TestCodegenTiling, CubeCommonTilingAbiUsesUint64WorkspaceSize) {
  const auto cube_code = this->GenCVTilingFunc();
  const auto common_code = this->GenTilingDataBlockDimAndWss();

  EXPECT_NE(cube_code.find("uint64_t workspace_size;"), std::string::npos);
  EXPECT_NE(common_code.find("uint64_t* workspace_size"), std::string::npos);
  EXPECT_EQ(cube_code.find("uint32_t workspace_size;"), std::string::npos);
  EXPECT_EQ(common_code.find("uint32_t* workspace_size"), std::string::npos);
}

TEST_F(TestCodegenTiling, PrepareMatMulAttrsShouldUseDefaultOpImplModeForMatMulV3) {
  codegen::MatMulCubeInfo cube_info;
  cube_info.is_batch = false;
  cube_info.enable_hf32 = 0;

  std::vector<codegen::AttrInfo> attrs;
  PrepareMatMulAttrs(cube_info, attrs);

  ASSERT_GT(attrs.size(), 3U);
  EXPECT_EQ(attrs[3].name, "opImplMode");
  EXPECT_EQ(attrs[3].dtype, "int");
  EXPECT_EQ(attrs[3].value_int, 0);
}

TEST_F(TestCodegenTiling, SingleGroupWorkspaceSymbolTest) {
  ascir::ImplGraph graph0("test_graph0");
  auto s0 = graph0.CreateSizeVar("s0");
  auto s1 = graph0.CreateSizeVar("s1");

  auto z0 = graph0.CreateAxis("z0", s0);
  auto z1 = graph0.CreateAxis("z1", s1);

  af::ascir_op::Workspace workspace("workspace");
  graph0.AddNode(workspace);
  workspace.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Load load("load");
  graph0.AddNode(load);
  load.x = workspace.y;
  load.attr.sched.axis = {z0.id, z1.id};
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, s1};
  *load.y.strides = {s1, af::ops::One};

  auto load_node = graph0.FindNode("load");
  auto workspace_node = graph0.FindNode("workspace");

  workspace_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  workspace_node->outputs[0].attr.mem.tensor_id = 0;

  load_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load_node->outputs[0].attr.mem.tensor_id = 1;

  std::vector<ascir::ImplGraph> impl_graphs;
  impl_graphs.push_back(graph0);

  std::vector<ascir::ScheduledResult> schedule_results;
  ascir::ScheduledResult schedule_result;
  ascir::ScheduleGroup schedule_group;
  schedule_group.impl_graphs = impl_graphs;
  schedule_result.schedule_groups.push_back(schedule_group);
  schedule_results.push_back(schedule_result);

  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.workspace_nodes.push_back(workspace_node);
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  const auto code = this->GenGetWorkspaceSizeFunc("AutofuseTilingData", fused_schedule_result);
  EXPECT_NE(code.find("uint64_t ws_size = 0;"), std::string::npos);
  EXPECT_NE(code.find("const uint64_t workspace_size_symbol_0 = static_cast<uint64_t>(t.s0);"), std::string::npos);
  EXPECT_NE(code.find("const uint64_t workspace_size_symbol_1 = static_cast<uint64_t>(t.s1);"), std::string::npos);
  EXPECT_NE(code.find("Max(0, (2 * Max(Max(1, workspace_size_symbol_1), (workspace_size_symbol_0 * "
                      "workspace_size_symbol_1))))"),
            std::string::npos);
  EXPECT_NE(code.find("if (!CheckedWorkspaceAlign(ws_size, 512U, ws_size))"), std::string::npos);
}

TEST_F(TestCodegenTiling, GetWorkspaceSizeEvaluatesSymbolsInUint64Domain) {
  ascir::ImplGraph graph0("test_graph0");
  auto s0 = graph0.CreateSizeVar("s0");
  auto s1 = graph0.CreateSizeVar("s1");

  auto z0 = graph0.CreateAxis("z0", s0);
  auto z1 = graph0.CreateAxis("z1", s1);

  af::ascir_op::Workspace workspace("workspace");
  graph0.AddNode(workspace);
  workspace.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Load load("load");
  graph0.AddNode(load);
  load.x = workspace.y;
  load.attr.sched.axis = {z0.id, z1.id};
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, s1};
  *load.y.strides = {s1, af::ops::One};

  auto load_node = graph0.FindNode("load");
  auto workspace_node = graph0.FindNode("workspace");
  workspace_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  workspace_node->outputs[0].attr.mem.tensor_id = 0;
  load_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load_node->outputs[0].attr.mem.tensor_id = 1;

  ascir::ScheduledResult schedule_result;
  ascir::ScheduleGroup schedule_group;
  schedule_group.impl_graphs.push_back(graph0);
  schedule_result.schedule_groups.push_back(schedule_group);
  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.workspace_nodes.push_back(workspace_node);
  fused_schedule_result.node_idx_to_scheduled_results.push_back({schedule_result});

  const auto code = this->GenGetWorkspaceSizeFunc("AutofuseTilingData", fused_schedule_result);
  const std::string source = R"(
#include <cstdint>

struct AutofuseTilingData {
  uint32_t tiling_key;
  uint32_t s0;
  uint32_t s1;
};

namespace optiling {
template <typename T, typename U>
inline auto Max(T a, U b) {
  return static_cast<double>(a) > static_cast<double>(b) ? a : b;
}
}  // namespace optiling

#define OP_LOGE(...) ((void)0)
#define OP_LOGW(...) ((void)0)
#define OP_NAME "workspace-size-test"

)" + code + R"(

int main() {
  AutofuseTilingData tiling_data{0U, 707981472U, 4U};
  uint64_t workspace_size = 0U;
  if (!GetWorkspaceSize(tiling_data, workspace_size)) {
    return 2;
  }
  return workspace_size == 5663852032ULL ? 0 : 3;
}
)";

  EXPECT_TRUE(CompileAndRunCode(source));
}

TEST_F(TestCodegenTiling, SingleGroupWorkspaceValueTest) {
  ascir::ImplGraph graph0("test_graph0");
  auto s0 = graph0.CreateSizeVar(150);
  auto s1 = graph0.CreateSizeVar(2);

  auto z0 = graph0.CreateAxis("z0", s0);
  auto z1 = graph0.CreateAxis("z1", s1);

  af::ascir_op::Workspace workspace("workspace");
  graph0.AddNode(workspace);
  workspace.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Load load("load");
  graph0.AddNode(load);
  load.x = workspace.y;
  load.attr.sched.axis = {z0.id, z1.id};
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, s1};
  *load.y.strides = {s1, af::ops::One};

  auto load_node = graph0.FindNode("load");
  auto workspace_node = graph0.FindNode("workspace");

  workspace_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  workspace_node->outputs[0].attr.mem.tensor_id = 0;

  load_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load_node->outputs[0].attr.mem.tensor_id = 1;

  std::vector<ascir::ImplGraph> impl_graphs;
  impl_graphs.push_back(graph0);

  std::vector<ascir::ScheduledResult> schedule_results;
  ascir::ScheduledResult schedule_result;
  ascir::ScheduleGroup schedule_group;
  schedule_group.impl_graphs = impl_graphs;
  schedule_result.schedule_groups.push_back(schedule_group);
  schedule_results.push_back(schedule_result);

  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.workspace_nodes.push_back(workspace_node);
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  const auto code = this->GenGetWorkspaceSizeFunc("AutofuseTilingData", fused_schedule_result);
  EXPECT_NE(code.find("uint64_t ws_size = 0;"), std::string::npos);
  EXPECT_NE(code.find("if (!CheckedWorkspaceAdd(ws_size, 600, ws_size))"), std::string::npos);
  EXPECT_NE(code.find("if (!CheckedWorkspaceAlign(ws_size, 512U, ws_size))"), std::string::npos);
}

TEST_F(TestCodegenTiling, TfTilingWithConfigShouldUseParsedUbSizeDirectly) {
  af::AscGraph graph("relu_graph");
  CreateElemwiseGraphWithRelu(graph);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused_schedule_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);
  const std::map<std::string, std::string> shape_info;
  auto res = this->Generate(fused_schedule_result, shape_info, "", "0");
  ASSERT_TRUE(res.find(codegen::kTilingDefAndConstIdentify) != res.end());
  const auto &tiling_impl = res.at(codegen::kTilingDefAndConstIdentify);

  EXPECT_NE(tiling_impl.find("(*tiling_parse_data)->ub_size = ub_size;"), std::string::npos);
  EXPECT_NE(tiling_impl.find("limit.ub_size = (uint32_t)parse->ub_size;"), std::string::npos);
  EXPECT_NE(tiling_impl.find("tiling->set_ub_size(limit->ub_size);"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("tiling->set_ub_size(limit->ub_size - 256);"), std::string::npos);
}

TEST_F(TestCodegenTiling, TilingParseShouldReserveUbExceptAscend910And910B) {
  af::AscGraph graph("relu_graph");
  CreateElemwiseGraphWithRelu(graph);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused_schedule_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);
  const std::map<std::string, std::string> shape_info;
  auto res = this->Generate(fused_schedule_result, shape_info, "", "0");
  ASSERT_TRUE(res.find(codegen::kTilingDefAndConstIdentify) != res.end());
  const auto &tiling_impl = res.at(codegen::kTilingDefAndConstIdentify);

  EXPECT_NE(tiling_impl.find("ascendc_platform.GetSocVersion() != platform_ascendc::SocVersion::ASCEND910"),
            std::string::npos);
  EXPECT_NE(tiling_impl.find("ascendc_platform.GetSocVersion() != platform_ascendc::SocVersion::ASCEND910B"),
            std::string::npos);
  EXPECT_EQ(tiling_impl.find("ascendc_platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950 && "
                             "ub_size % 1024 == 0"),
            std::string::npos);
}

TEST_F(TestCodegenTiling, GetWorkspaceSizeGuardsDynamicDenominator) {
  ascir::ImplGraph graph0("test_graph0");
  auto a1t_size = graph0.CreateSizeVar("a1t_size");
  auto z0 = graph0.CreateAxis("z0", af::ops::One);

  af::ascir_op::Workspace workspace("workspace");
  graph0.AddNode(workspace);
  workspace.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Load load("load");
  graph0.AddNode(load);
  load.x = workspace.y;
  load.attr.sched.axis = {z0.id};
  *load.y.axis = {z0.id};
  *load.y.repeats = {af::sym::Ceiling(af::Symbol(512) / a1t_size)};
  *load.y.strides = {af::ops::One};

  auto load_node = graph0.FindNode("load");
  auto workspace_node = graph0.FindNode("workspace");
  workspace_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  workspace_node->outputs[0].attr.mem.tensor_id = 0;
  load_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load_node->outputs[0].attr.mem.tensor_id = 1;

  ascir::ScheduledResult schedule_result;
  ascir::ScheduleGroup schedule_group;
  schedule_group.impl_graphs.push_back(graph0);
  schedule_result.schedule_groups.push_back(schedule_group);

  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.workspace_nodes.push_back(workspace_node);
  fused_schedule_result.node_idx_to_scheduled_results.push_back({schedule_result});

  const auto code = this->GenGetWorkspaceSizeFunc("AutofuseTilingData", fused_schedule_result);
  EXPECT_NE(code.find("const uint64_t workspace_size_symbol_0 = static_cast<uint64_t>(t.a1t_size);"),
            std::string::npos);
  EXPECT_NE(code.find("if (workspace_size_symbol_0 <= 0) {"), std::string::npos);
  EXPECT_NE(code.find("return false;"), std::string::npos);
  EXPECT_LT(code.find("if (workspace_size_symbol_0 <= 0) {"), code.find("CheckedWorkspaceAdd(ws_size"));
}

TEST_F(TestCodegenTiling, MultiGroupWorkspaceSymbolTest) {
  ascir::ImplGraph graph0("test_graph0");
  auto s0 = graph0.CreateSizeVar("s0");
  auto s1 = graph0.CreateSizeVar("s1");

  auto z0 = graph0.CreateAxis("z0", s0);
  auto z1 = graph0.CreateAxis("z1", s1);

  af::ascir_op::Workspace workspace("workspace");
  graph0.AddNode(workspace);
  workspace.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Load load("load");
  graph0.AddNode(load);
  load.x = workspace.y;
  load.attr.sched.axis = {z0.id, z1.id};
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, s1};
  *load.y.strides = {s1, af::ops::One};

  auto load_node = graph0.FindNode("load");
  auto workspace_node = graph0.FindNode("workspace");

  workspace_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  workspace_node->outputs[0].attr.mem.tensor_id = 0;

  load_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load_node->outputs[0].attr.mem.tensor_id = 2;

  ascir::ImplGraph graph1("test_graph1");
  s0 = graph1.CreateSizeVar("s0");
  s1 = graph1.CreateSizeVar("s1");

  z0 = graph1.CreateAxis("z0", s0);
  z1 = graph1.CreateAxis("z1", s1);

  af::ascir_op::Workspace workspace1("workspace1");
  graph1.AddNode(workspace1);
  workspace1.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Load load1("load1");
  graph1.AddNode(load1);
  load1.x = workspace1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, af::ops::One};

  auto load1_node = graph1.FindNode("load1");
  auto workspace1_node = graph1.FindNode("workspace1");

  workspace1_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  workspace1_node->outputs[0].attr.mem.tensor_id = 1;

  load1_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load1_node->outputs[0].attr.mem.tensor_id = 3;

  std::vector<ascir::ScheduledResult> schedule_results;
  ascir::ScheduledResult schedule_result;
  ascir::ScheduleGroup sch_groups0;
  ascir::ScheduleGroup sch_groups1;
  sch_groups0.impl_graphs = {graph0};
  sch_groups1.impl_graphs = {graph1};
  schedule_result.schedule_groups.push_back(sch_groups0);
  schedule_result.schedule_groups.push_back(sch_groups1);
  schedule_results.push_back(schedule_result);

  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.workspace_nodes.push_back(workspace_node);
  fused_schedule_result.workspace_nodes.push_back(workspace1_node);
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  const auto code = this->GenGetWorkspaceSizeFunc("AutofuseTilingData", fused_schedule_result);
  EXPECT_NE(code.find("uint64_t ws_size = 0;"), std::string::npos);
  EXPECT_NE(code.find("const uint64_t workspace_size_symbol_0 = static_cast<uint64_t>("
                      "t.graph0_result0_g0_tiling_data.s0);"),
            std::string::npos);
  EXPECT_NE(code.find("Max(0, (2 * Max(Max(1, workspace_size_symbol_1), (workspace_size_symbol_0 * "
                      "workspace_size_symbol_1))))"),
            std::string::npos);
  EXPECT_NE(code.find("const uint64_t workspace_size_symbol_0 = static_cast<uint64_t>("
                      "t.graph0_result0_g1_tiling_data.s0);"),
            std::string::npos);
  EXPECT_NE(code.find("if (!CheckedWorkspaceAlign(ws_size, 512U, ws_size))"), std::string::npos);
}

TEST_F(TestCodegenTiling, MultiGroupWorkspaceValueTest) {
  ascir::ImplGraph graph0("test_graph0");
  auto s0 = graph0.CreateSizeVar(16);
  auto s1 = graph0.CreateSizeVar(32);

  auto z0 = graph0.CreateAxis("z0", s0);
  auto z1 = graph0.CreateAxis("z1", s1);

  af::ascir_op::Workspace workspace("workspace");
  graph0.AddNode(workspace);
  workspace.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Load load("load");
  graph0.AddNode(load);
  load.x = workspace.y;
  load.attr.sched.axis = {z0.id, z1.id};
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, s1};
  *load.y.strides = {s1, af::ops::One};

  auto load_node = graph0.FindNode("load");
  auto workspace_node = graph0.FindNode("workspace");

  workspace_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  workspace_node->outputs[0].attr.mem.tensor_id = 0;

  load_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load_node->outputs[0].attr.mem.tensor_id = 2;

  ascir::ImplGraph graph1("test_graph1");
  s0 = graph1.CreateSizeVar(5);
  s1 = graph1.CreateSizeVar(100);

  z0 = graph1.CreateAxis("z0", s0);
  z1 = graph1.CreateAxis("z1", s1);

  af::ascir_op::Workspace workspace1("workspace1");
  graph1.AddNode(workspace1);
  workspace1.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Load load1("load1");
  graph1.AddNode(load1);
  load1.x = workspace1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, af::ops::One};

  auto load1_node = graph1.FindNode("load1");
  auto workspace1_node = graph1.FindNode("workspace1");

  workspace1_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  workspace1_node->outputs[0].attr.mem.tensor_id = 1;

  load1_node->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load1_node->outputs[0].attr.mem.tensor_id = 3;

  std::vector<ascir::ScheduledResult> schedule_results;
  ascir::ScheduledResult schedule_result;
  ascir::ScheduleGroup sch_groups0;
  ascir::ScheduleGroup sch_groups1;
  sch_groups0.impl_graphs = {graph0};
  sch_groups1.impl_graphs = {graph1};
  schedule_result.schedule_groups.push_back(sch_groups0);
  schedule_result.schedule_groups.push_back(sch_groups1);
  schedule_results.push_back(schedule_result);

  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.workspace_nodes.push_back(workspace_node);
  fused_schedule_result.workspace_nodes.push_back(workspace1_node);
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  const auto code = this->GenGetWorkspaceSizeFunc("AutofuseTilingData", fused_schedule_result);
  EXPECT_NE(code.find("uint64_t ws_size = 0;"), std::string::npos);
  EXPECT_NE(code.find("if (!CheckedWorkspaceAdd(ws_size, 1024, ws_size))"), std::string::npos);
  EXPECT_NE(code.find("if (!CheckedWorkspaceAdd(ws_size, 1000, ws_size))"), std::string::npos);
  EXPECT_NE(code.find("if (!CheckedWorkspaceAlign(ws_size, 512U, ws_size))"), std::string::npos);
}

TEST_F(TestCodegenTiling, EmptyTensorKernel) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", af::ops::Zero);

  af::ascir_op::Data x_op("x", graph);
  x_op.ir_attr.SetIndex(0);
  af::ascir_op::Load load_op("load");
  af::ascir_op::Store store_op("store");
  af::ascir_op::Output y_op("y");
  y_op.ir_attr.SetIndex(0);
  graph.AddNode(load_op);
  graph.AddNode(store_op);
  graph.AddNode(y_op);

  load_op.x = x_op.y;
  load_op.y.dtype = ge::DT_FLOAT16;
  store_op.x = load_op.y;
  y_op.x = store_op.y;

  auto x = graph.FindNode("x");
  auto load = graph.FindNode("load");
  auto store = graph.FindNode("store");
  auto y = graph.FindNode("y");

  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.tensor_id = 0;
  x->attr.api.unit = af::ComputeUnit::kUnitNone;
  y->attr.api.unit = af::ComputeUnit::kUnitNone;

  load->outputs[0].attr.axis = {z0.id};
  load->outputs[0].attr.vectorized_axis = {z0.id};
  load->outputs[0].attr.vectorized_strides = {af::ops::One};
  load->outputs[0].attr.repeats = {z0.size};
  load->outputs[0].attr.strides = {af::ops::One};
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.mem.tensor_id = 1;
  load->outputs[0].attr.que.id = 0;
  load->outputs[0].attr.mem.reuse_id = 0;
  load->outputs[0].attr.que.depth = 2;
  load->outputs[0].attr.que.buf_num = 2;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.tensor_id = 2;
  store->outputs[0].attr.axis = {z0.id};
  store->outputs[0].attr.vectorized_axis = {z0.id};
  store->outputs[0].attr.vectorized_strides = {af::ops::One};
  store->outputs[0].attr.repeats = {z0.size};
  store->outputs[0].attr.strides = {af::ops::One};

  ::ascir::ScheduledResult schedule_result;
  schedule_result.schedule_groups.resize(1);
  for (auto &schedule_group : schedule_result.schedule_groups) {
    schedule_group.impl_graphs.emplace_back(graph);
  }
  std::vector<ascir::ScheduledResult> schedule_results;
  schedule_results.push_back(schedule_result);
  schedule_results.push_back(schedule_result);

  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = af::AscendString(graph.GetName().c_str());
  fused_schedule_result.input_nodes.push_back(x);
  fused_schedule_result.output_nodes.push_back(y);
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  const std::map<std::string, std::string> shape_info;
  auto res = this->Generate(fused_schedule_result, shape_info, "", "0");
  std::string tiling_func_declare{"TilingFunc(gert::TilingSymbolEvalContext *context)\n{\n"};
  auto pos = res["tiling_def_and_tiling_const"].find(tiling_func_declare) + tiling_func_declare.size();
  std::string expect_str{
      "  context->SetBlockDim(1);\n  *context->GetWorkspaceSizes(1) = 0;\n  return ge::GRAPH_SUCCESS;\n"};
  std::string tiling_func_content = res["tiling_def_and_tiling_const"].substr(pos, expect_str.size());
  EXPECT_EQ(expect_str, tiling_func_content);
}

TEST_F(TestCodegenTiling, EmptyTensorInductorModeledPerfShouldNotReferenceAttTiling) {
  auto tiling_files = this->GenTilingCodeForInductor();
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  const size_t perf_func_pos = tiling_impl.find("static double EvaluateModeledPerf");
  ASSERT_NE(perf_func_pos, std::string::npos);
  const size_t topn_func_pos = tiling_impl.find("extern \"C\" int64_t GenerateTopnSolutions", perf_func_pos);
  ASSERT_NE(topn_func_pos, std::string::npos);
  const std::string perf_func = tiling_impl.substr(perf_func_pos, topn_func_pos - perf_func_pos);

  EXPECT_NE(perf_func.find("return DBL_MAX;"), std::string::npos);
  EXPECT_EQ(perf_func.find("optiling::GetPerf"), std::string::npos);
  EXPECT_EQ(perf_func.find("AscGraph"), std::string::npos);

  EXPECT_EQ(tiling_impl.find("TemplateSemanticInfo"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("GetTemplateSemanticInfos"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("GetTilingCore(default_tiling, -1)"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("EmitFinalTilingByCase"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("ShouldEmitFinalTiling()"), std::string::npos);
}

TEST_F(TestCodegenTiling, TestGenDfxInputSymbolInfo) {
  std::map<std::string, std::string> shape_info;
  shape_info["s0"] = R"([&]() -> int64_t {
    const auto *tensor = context->GetInputTensor(0);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(0);
  }())";
  shape_info["s1"] = R"([&]() -> int64_t {
    const auto *tensor = context->GetInputTensor(0);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(1);
  }())";
  shape_info["s2"] = R"([&]() -> int64_t {
    const auto *tensor = context->GetInputTensor(1);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(0);
  }())";

  ascir::FusedScheduledResult fused_schedule_result;
  std::vector<af::Expression> origin_vars{af::Symbol("s0"), af::Symbol("s1"), af::Symbol("s2")};
  fused_schedule_result.origin_vars = origin_vars;
  auto gen_func = this->GenDfxInputSymbolInfo(fused_schedule_result, shape_info);
  auto expect_func =
      R"(extern "C" ge::graphStatus DfxInputSymbolInfo(gert::TilingSymbolEvalContext *context, char *out_symbol_info, size_t size)
{
  if (out_symbol_info == nullptr || size == 0) {
    return ge::GRAPH_SUCCESS;
  }
  std::string symbol_info;
  auto s0 = [&]() -> int64_t {
    const auto *tensor = context->GetInputTensor(0);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(0);
  }();
  symbol_info += ("s0: " + std::to_string(s0));

  auto s1 = [&]() -> int64_t {
    const auto *tensor = context->GetInputTensor(0);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(1);
  }();
  symbol_info += (", s1: " + std::to_string(s1));

  auto s2 = [&]() -> int64_t {
    const auto *tensor = context->GetInputTensor(1);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(0);
  }();
  symbol_info += (", s2: " + std::to_string(s2));


  if (symbol_info.empty()) {
    out_symbol_info[0] = '\0';
    return ge::GRAPH_SUCCESS;
  }
  symbol_info += ".";
  if (strncpy_s(out_symbol_info, size, symbol_info.c_str(), std::min(symbol_info.size(), size - 1)) != 0) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}
)";
  EXPECT_EQ(gen_func, expect_func);
}

TEST_F(TestCodegenTiling, TestCompileSuccess) {
  std::stringstream ss;

  ss << "#include <stdexcept>" << std::endl;
  ss << "#include <sstream>" << std::endl;
  ss << "#include <cmath>" << std::endl;
  ss << "#ifndef __CCE_KT_TEST__" << std::endl;
  ss << "#include \"register/op_def_registry.h\"" << std::endl;
  ss << "#include \"exe_graph/runtime/infer_shape_context.h\"" << std::endl;
  ss << "#include \"exe_graph/runtime/kernel_context.h\"" << std::endl;
  ss << "#include \"exe_graph/runtime/continuous_vector.h\"" << std::endl;
  ss << "#endif" << std::endl;

  ss << "#define Max(a, b) ((double)(a) > (double)(b) ? (a) : (b))" << std::endl;
  ss << "#define Min(a, b) ((double)(a) < (double)(b) ? (a) : (b))" << std::endl;
  ss << "#define Log(a) (log((double)(a)))" << std::endl;
  ss << "#define Pow(a, b) pow(a, b)" << std::endl;
  ss << "#define Rational(a, b) ((double)(a) / (double)(b))" << std::endl;

  std::string tiling_context = R"(
namespace gert {
  class TilingSymbolEvalContext : public TilingContext {
    public:
      const gert::Tensor *GetGraphInputTensor(size_t data_index) const {
        auto *tensor = GetInputPointer<gert::Tensor>(data_index + 1);
        if (tensor == nullptr) {
          return nullptr;
        }
        return tensor;
      }
  };
})";

  ss << tiling_context << std::endl;

  std::map<std::string, std::string> shape_info;
  shape_info["s0"] = R"([&]() -> int64_t {
    const auto *tensor = context->GetInputTensor(0);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(0);
  }())";
  shape_info["s1"] = R"([&]() -> int64_t {
    const auto *tensor = context->GetInputTensor(0);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(1);
  }())";
  shape_info["s2"] = R"([&]() -> int64_t {
    const auto *tensor = context->GetInputTensor(1);
    if (tensor == nullptr) {
      return -1;
    }
    return tensor->GetOriginShape().GetDim(0);
  }())";
  ascir::FusedScheduledResult fused_schedule_result;
  std::vector<af::Expression> origin_vars{af::Symbol("s0"), af::Symbol("s1"), af::Symbol("s2")};
  fused_schedule_result.origin_vars = origin_vars;
  auto dfx_func = this->GenDfxInputSymbolInfo(fused_schedule_result, shape_info);
  ss << dfx_func << std::endl;
  ASSERT_TRUE(CompileCode(ss.str()));
}

/*
 * Codegen FindBestTilingKey测试
 * 1、单graph，单result单group
 * 2、多graph，仅在inductor场景下有，本轮暂不支持
 * 3、单graph，多result组合场景
 *    result1：单group，单graph
 *    result2：单group，多graph
 *    result3：多group场景组合
 *             group1：单graph
 *             group2：多graph
 * 4、enable_group_parallel场景， 不支持生成
 */

TEST_F(TestCodegenTiling, TestGenFindBestTilingKeyFuncFor1Group) {
  af::AscGraph graph1("graph1");
  af::ascir_op::Workspace workspace("workspace");
  graph1.AddNode(workspace);

  af::AscGraph graph2("graph2");
  af::AscGraph graph3("graph3");

  ascir::ScheduleGroup schedule_group;
  schedule_group.impl_graphs.push_back(graph1);
  schedule_group.impl_graphs.push_back(graph2);
  schedule_group.impl_graphs.push_back(graph3);

  ascir::ScheduledResult schedule_result;
  schedule_result.schedule_groups.push_back(schedule_group);

  ascir::FusedScheduledResult fused_schedule_result;
  std::vector<ascir::ScheduledResult> graph0_results = {schedule_result};
  fused_schedule_result.node_idx_to_scheduled_results.emplace_back(std::move(graph0_results));

  const std::map<std::string, std::string> shape_info;
  auto res = this->Generate(fused_schedule_result, shape_info, "", "0");

  std::string expect = R"(extern "C" int64_t FindBestTilingKey(AutofuseTilingData &t)
{
  if (t.tiling_key == 0) {
    return 0;
  } else if (t.tiling_key == 1) {
    return 1;
  } else if (t.tiling_key == 2) {
    return 2;
  }
  return -1;
}
)";
  auto pos = res["tiling_def_and_tiling_const"].find("extern \"C\" int64_t FindBestTilingKey(AutofuseTilingData &t)");
  auto func = res["tiling_def_and_tiling_const"].substr(pos, expect.size());
  ASSERT_EQ(func, expect);
}

constexpr char kExpectedMultiResultFindBestTilingKey[] = R"(extern "C" int64_t FindBestTilingKey(AutofuseTilingData &t)
{
  if (t.graph0_tiling_key == 0) {
    int64_t local_tiling_key = 0;
    if (t.graph0_result0_g0_tiling_data.tiling_key >= 1) {
      return -1;
    }
    local_tiling_key = local_tiling_key * 1 + t.graph0_result0_g0_tiling_data.tiling_key;
    return 0 + local_tiling_key;
  }  else if (t.graph0_tiling_key == 1) {
    int64_t local_tiling_key = 0;
    if (t.graph0_result1_g0_tiling_data.tiling_key >= 2) {
      return -1;
    }
    local_tiling_key = local_tiling_key * 2 + t.graph0_result1_g0_tiling_data.tiling_key;
    return 1 + local_tiling_key;
  }  else if (t.graph0_tiling_key == 2) {
    int64_t local_tiling_key = 0;
    if (t.graph0_result2_g0_tiling_data.tiling_key >= 1) {
      return -1;
    }
    local_tiling_key = local_tiling_key * 1 + t.graph0_result2_g0_tiling_data.tiling_key;
    if (t.graph0_result2_g1_tiling_data.tiling_key >= 2) {
      return -1;
    }
    local_tiling_key = local_tiling_key * 2 + t.graph0_result2_g1_tiling_data.tiling_key;
    return 3 + local_tiling_key;
  }
  return -1;
}
)";

TEST_F(TestCodegenTiling, TestGenFindBestTilingKeyFuncForMultiResult) {
  af::AscGraph graph1("graph1");
  af::AscGraph graph2("graph2");

  ascir::ScheduleGroup schedule_group1;
  schedule_group1.impl_graphs.push_back(graph1);

  ascir::ScheduleGroup schedule_group2;
  schedule_group2.impl_graphs.push_back(graph1);
  schedule_group2.impl_graphs.push_back(graph2);

  // 单group，单graph
  ascir::ScheduledResult schedule_result1;
  schedule_result1.schedule_groups.push_back(schedule_group1);

  // 单group，多graph
  ascir::ScheduledResult schedule_result2;
  schedule_result2.schedule_groups.push_back(schedule_group2);

  // 多group
  // group1单graph
  // group2多graph
  ascir::ScheduledResult schedule_result3;
  schedule_result3.schedule_groups.push_back(schedule_group1);
  schedule_result3.schedule_groups.push_back(schedule_group2);

  ascir::FusedScheduledResult fused_schedule_result;
  std::vector<ascir::ScheduledResult> graph0_results = {schedule_result1, schedule_result2, schedule_result3};
  fused_schedule_result.node_idx_to_scheduled_results.emplace_back(std::move(graph0_results));

  const std::map<std::string, std::string> shape_info;
  auto res = this->Generate(fused_schedule_result, shape_info, "", "0");

  const std::string expect = kExpectedMultiResultFindBestTilingKey;
  auto pos = res["tiling_def_and_tiling_const"].find("extern \"C\" int64_t FindBestTilingKey(AutofuseTilingData &t)");
  auto func = res["tiling_def_and_tiling_const"].substr(pos, expect.size());
  ASSERT_EQ(func, expect);
}

TEST_F(TestCodegenTiling, TestGenFindBestTilingKeyFuncForManyGroupsShouldBeLinear) {
  af::AscGraph graph1("graph1");
  af::AscGraph graph2("graph2");
  af::AscGraph graph3("graph3");
  ascir::ScheduleGroup schedule_group;
  schedule_group.impl_graphs = {graph1, graph2, graph3};

  ascir::ScheduledResult schedule_result;
  for (size_t i = 0; i < 8U; ++i) {
    schedule_result.schedule_groups.push_back(schedule_group);
  }
  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.node_idx_to_scheduled_results = {{schedule_result}};

  const std::map<std::string, std::string> shape_info;
  const auto res = this->Generate(fused_schedule_result, shape_info, "", "0");
  const auto &source = res.at("tiling_def_and_tiling_const");
  const auto begin = source.find("extern \"C\" int64_t FindBestTilingKey(AutofuseTilingData &t)");
  const std::string end_marker = "\n  return -1;\n}\n";
  ASSERT_NE(begin, std::string::npos);
  const auto end = source.find(end_marker, begin);
  ASSERT_NE(end, std::string::npos);
  const auto func = source.substr(begin, end + end_marker.size() - begin);

  EXPECT_LT(func.size(), 4096U);
  EXPECT_NE(func.find("int64_t local_tiling_key = 0;"), std::string::npos);
  EXPECT_NE(func.find("if (t.graph0_result0_g0_tiling_data.tiling_key >= 3)"), std::string::npos);
  EXPECT_NE(func.find("local_tiling_key = local_tiling_key * 3 + t.graph0_result0_g0_tiling_data.tiling_key;"),
            std::string::npos);
  EXPECT_NE(func.find("local_tiling_key = local_tiling_key * 3 + t.graph0_result0_g7_tiling_data.tiling_key;"),
            std::string::npos);
  EXPECT_NE(func.find("return 0 + local_tiling_key;"), std::string::npos);
  EXPECT_NE(this->GenGetTilingKeyCount(fused_schedule_result).find("return 6561;"), std::string::npos);
}

TEST_F(TestCodegenTiling, PgoTilingKeyCountShouldRespectLimitAndEmptyGroup) {
  enable_autofuse_pgo_ = true;

  EXPECT_FALSE(ShouldFallbackPgo(GenTilingKeyCountResult({{10U, 10U, 10U, 10U}})));
  EXPECT_TRUE(ShouldFallbackPgo(GenTilingKeyCountResult({{10U, 10U, 10U, 11U}})));
  EXPECT_TRUE(ShouldFallbackPgo(GenTilingKeyCountResult({{10U, 10U, 10U, 10U}, {1U}})));
  EXPECT_FALSE(ShouldFallbackPgo(GenTilingKeyCountResult({{11U, 10U, 10U, 10U, 0U}})));
  EXPECT_TRUE(ShouldFallbackPgo(GenTilingKeyCountResult({{10U, 10U, 10U, 10U}, {}})));
  const auto empty_group = GenTilingKeyCountResult({{0U, 1U}});
  EXPECT_EQ(GenFindBestTilingKeyFunc(empty_group, "AutofuseTilingData").find("local_tiling_key"), std::string::npos);
}

TEST_F(TestCodegenTiling, PgoTilingKeyCountOverflowShouldFallbackTfAndPgoRunner) {
  enable_autofuse_pgo_ = true;
  const auto fused_schedule_result = GenTilingKeyCountResult({{10U, 10U, 10U, 11U}});
  const auto files = Generate(fused_schedule_result, {}, "/tmp", "10");
  const auto &entry = files.at(codegen::kTilingDefAndConstIdentify);

  EXPECT_EQ(entry.find("#include \"autofuse_tiling_func_pgo.h\""), std::string::npos);
  EXPECT_NE(entry.find("extern \"C\" int64_t FindBestTilingKey"), std::string::npos);
  const auto pgo_source = GenerateForPgo(fused_schedule_result, "/tmp");
  EXPECT_EQ(pgo_source, "int main() { return 1; }\n");
  EXPECT_EQ(pgo_source.find("PGOGetProfiling"), std::string::npos);
  EXPECT_TRUE(CompileCode(pgo_source, false));
}

TEST_F(TestCodegenTiling, PgoTilingKeyCountOverflowShouldFallbackInductor) {
  ScopedAutofusePgoFlag pgo_flag(true);
  auto fused_schedule_result = GenTilingKeyCountResult({{10U, 10U, 10U, 11U}});
  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen::CodegenResult result;

  ASSERT_EQ(codegen.GenerateForInductor(fused_schedule_result, result), af::SUCCESS);
  EXPECT_EQ(result.tiling.find("AUTOFUSE_SPLIT_FILE_BEGIN: PgoRunner"), std::string::npos);
  EXPECT_NE(result.tiling.find("Topn selector helpers: default-first"), std::string::npos);
}

TEST_F(TestCodegenTiling, PgoTilingKeyCountShouldRespectInt64Capacity) {
  const auto representable = GenTilingKeyCountResult({std::vector<size_t>(63U, 2U)});
  auto unrepresentable = representable;
  unrepresentable.node_idx_to_scheduled_results[0][0].schedule_groups.push_back(
      representable.node_idx_to_scheduled_results[0][0].schedule_groups[0]);

  const auto representable_func = GenFindBestTilingKeyFunc(representable, "AutofuseTilingData");
  const auto unrepresentable_func = GenFindBestTilingKeyFunc(unrepresentable, "AutofuseTilingData");
  EXPECT_NE(representable_func.find("graph0_result0_g62_tiling_data.tiling_key"), std::string::npos);
  EXPECT_NE(GenGetTilingKeyCount(representable).find("return 9223372036854775808ULL;"), std::string::npos);
  EXPECT_NE(GenGetTilingKeyCount(unrepresentable).find("return 18446744073709551615ULL;"), std::string::npos);
  EXPECT_TRUE(CompileCode("#include <cstdint>\n" + GenGetTilingKeyCount(unrepresentable)));
  EXPECT_EQ(unrepresentable_func, R"(extern "C" int64_t FindBestTilingKey(AutofuseTilingData &t)
{
  return -1;
}
)");
}

TEST_F(TestCodegenTiling, TestGenFindBestTilingKeyFuncForEnableParallel) {
  af::AscGraph graph1("graph1");
  af::AscGraph graph2("graph2");

  ascir::ScheduleGroup schedule_group1;
  schedule_group1.impl_graphs.push_back(graph1);

  ascir::ScheduleGroup schedule_group2;
  schedule_group2.impl_graphs.push_back(graph1);
  schedule_group2.impl_graphs.push_back(graph2);

  // 单group，单graph
  ascir::ScheduledResult schedule_result1;
  schedule_result1.schedule_groups.push_back(schedule_group1);

  // 单group，多graph
  ascir::ScheduledResult schedule_result2;
  schedule_result2.schedule_groups.push_back(schedule_group2);

  // 多group
  // group1单graph
  // group2多graph
  ascir::ScheduledResult schedule_result3;
  schedule_result3.enable_group_parallel = true;
  schedule_result3.schedule_groups.push_back(schedule_group1);
  schedule_result3.schedule_groups.push_back(schedule_group2);

  ascir::FusedScheduledResult fused_schedule_result;
  std::vector<ascir::ScheduledResult> graph0_results = {schedule_result1, schedule_result2, schedule_result3};
  fused_schedule_result.node_idx_to_scheduled_results.emplace_back(std::move(graph0_results));

  const std::map<std::string, std::string> shape_info;
  auto res = this->Generate(fused_schedule_result, shape_info, "", "0");

  auto pos = res["tiling_def_and_tiling_const"].find("extern \"C\" int64_t FindBestTilingKey(AutofuseTilingData &t)");
  ASSERT_EQ(pos, std::string::npos);
}

TEST_F(TestCodegenTiling, TestGenExternTilingFunc) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<ge::RuntimeStubV2Common>();
  ge::RuntimeStub::SetInstance(stub_v2);
  af::AscGraph graph1("graph1");
  af::AscGraph graph2("graph2");

  ascir::ScheduleGroup schedule_group1;
  schedule_group1.impl_graphs.push_back(graph1);

  ascir::ScheduleGroup schedule_group2;
  schedule_group2.impl_graphs.push_back(graph1);
  schedule_group2.impl_graphs.push_back(graph2);

  // 单group，单graph
  ascir::ScheduledResult schedule_result1;
  schedule_result1.schedule_groups.push_back(schedule_group1);

  // 单group，多graph
  ascir::ScheduledResult schedule_result2;
  schedule_result2.schedule_groups.push_back(schedule_group2);

  // 多group
  // group1单graph
  // group2多graph
  ascir::ScheduledResult schedule_result3;
  schedule_result3.enable_group_parallel = true;
  schedule_result3.schedule_groups.push_back(schedule_group1);
  schedule_result3.schedule_groups.push_back(schedule_group2);

  ascir::FusedScheduledResult fused_schedule_result;
  std::vector<ascir::ScheduledResult> graph0_results = {schedule_result1, schedule_result2, schedule_result3};
  fused_schedule_result.node_idx_to_scheduled_results.emplace_back(std::move(graph0_results));

  const std::map<std::string, std::string> shape_info;
  auto res = this->Generate(fused_schedule_result, shape_info, "", "0");

  auto pos = res["tiling_def_and_tiling_const"].find("extern \"C\" int64_t FindBestTilingKey(AutofuseTilingData &t)");
  ASSERT_EQ(pos, std::string::npos);
  ge::RuntimeStub::Reset();
  ge::PlatformContext::GetInstance().Reset();
}

TEST_F(TestCodegenTiling, TestPGOSearchTensorMallocDef) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", af::ops::One);

  af::ascir_op::Data x_op("x", graph);
  x_op.ir_attr.SetIndex(0);
  af::ascir_op::Load load_op("load");
  af::ascir_op::Store store_op("store");
  af::ascir_op::Output y_op("y");
  y_op.ir_attr.SetIndex(0);
  graph.AddNode(load_op);
  graph.AddNode(store_op);
  graph.AddNode(y_op);

  load_op.x = x_op.y;
  load_op.y.dtype = ge::DT_FLOAT16;
  store_op.x = load_op.y;
  y_op.x = store_op.y;

  auto x = graph.FindNode("x");
  auto load = graph.FindNode("load");
  auto store = graph.FindNode("store");
  auto y = graph.FindNode("y");

  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.tensor_id = 0;
  x->attr.api.unit = af::ComputeUnit::kUnitNone;
  y->attr.api.unit = af::ComputeUnit::kUnitNone;

  load->outputs[0].attr.axis = {z0.id};
  load->outputs[0].attr.vectorized_axis = {z0.id};
  load->outputs[0].attr.vectorized_strides = {af::ops::One};
  load->outputs[0].attr.repeats = {z0.size};
  load->outputs[0].attr.strides = {af::ops::One};
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.mem.tensor_id = 1;
  load->outputs[0].attr.que.id = 0;
  load->outputs[0].attr.mem.reuse_id = 0;
  load->outputs[0].attr.que.depth = 2;
  load->outputs[0].attr.que.buf_num = 2;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  store->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.tensor_id = 2;
  store->outputs[0].attr.axis = {z0.id};
  store->outputs[0].attr.vectorized_axis = {z0.id};
  store->outputs[0].attr.vectorized_strides = {af::ops::One};
  store->outputs[0].attr.repeats = {z0.size};
  store->outputs[0].attr.strides = {af::ops::One};

  y->inputs[0].attr.repeats = {z0.size};
  y->inputs[0].attr.strides = {af::ops::One};
  y->inputs[0].attr.dtype = ge::DT_FLOAT16;

  ::ascir::ScheduledResult schedule_result;
  schedule_result.schedule_groups.resize(1);
  for (auto &schedule_group : schedule_result.schedule_groups) {
    schedule_group.impl_graphs.emplace_back(graph);
  }
  std::vector<ascir::ScheduledResult> schedule_results;
  schedule_results.push_back(schedule_result);
  schedule_results.push_back(schedule_result);

  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = af::AscendString(graph.GetName().c_str());
  fused_schedule_result.input_nodes.push_back(x);
  fused_schedule_result.output_nodes.push_back(y);
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);

  std::string mallocdef = this->PGOSearchTensorMallocDef(fused_schedule_result);
  const std::string expect = R"(  size_t input0_size = 2;
  ret = aclrtMalloc(&input0, input0_size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    DLOGE("aclrtMalloc input0 failed. ERROR: %d", ret);
    return FAILED;
  }
  size_t output0_size = 2;
  ret = aclrtMalloc(&output0, output0_size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    DLOGE("aclrtMalloc output0 failed. ERROR: %d", ret);
    return FAILED;
  }
)";
  ASSERT_EQ(mallocdef, expect);
}

TEST_F(TestCodegenTiling, TestPGOSearchTensorMallocDefUsesLargestCandidateOutput) {
  af::AscGraph small_graph("small_graph");
  af::ascir_op::Store small_store_op("small_store");
  small_store_op.ir_attr.SetOffset(af::ops::Zero);
  af::ascir_op::Output small_output_op("small_output");
  small_output_op.ir_attr.SetIndex(0);
  small_graph.AddNode(small_store_op);
  small_graph.AddNode(small_output_op);
  small_output_op.x = small_store_op.y;
  auto small_output = small_graph.FindNode("small_output");
  small_output->inputs[0].attr.repeats = {af::ops::One};
  small_output->inputs[0].attr.strides = {af::ops::One};
  small_output->inputs[0].attr.dtype = ge::DT_FLOAT16;

  af::AscGraph large_graph("large_graph");
  af::ascir_op::Store large_store_op("large_store");
  large_store_op.ir_attr.SetOffset(af::Expression::Parse("12288"));
  af::ascir_op::Output large_output_op("large_output");
  large_output_op.ir_attr.SetIndex(0);
  large_graph.AddNode(large_store_op);
  large_graph.AddNode(large_output_op);
  large_output_op.x = large_store_op.y;
  auto large_output = large_graph.FindNode("large_output");
  large_output->inputs[0].attr.repeats = {af::Expression::Parse("4096")};
  large_output->inputs[0].attr.strides = {af::ops::One};
  large_output->inputs[0].attr.dtype = ge::DT_FLOAT16;

  ascir::ScheduledResult schedule_result;
  schedule_result.schedule_groups.resize(1);
  schedule_result.schedule_groups[0].impl_graphs = {small_graph, large_graph};

  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.output_nodes.push_back(small_output);
  fused_schedule_result.node_idx_to_scheduled_results.push_back({schedule_result});

  const std::string malloc_def = this->PGOSearchTensorMallocDef(fused_schedule_result);
  EXPECT_NE(malloc_def.find("  size_t output0_size = 2;\n"), std::string::npos);
  EXPECT_NE(malloc_def.find("  output0_size = std::max(output0_size, static_cast<size_t>(32768));\n"),
            std::string::npos);
}

TEST_F(TestCodegenTiling, TestPGOSearchTensorMallocDefUsesLargestCandidateInput) {
  af::AscGraph small_graph("small_graph");
  af::ascir_op::Data small_input_op("small_input", small_graph);
  small_input_op.ir_attr.SetIndex(0);
  af::ascir_op::Load small_load_op("small_load");
  small_load_op.ir_attr.SetOffset(af::ops::Zero);
  small_graph.AddNode(small_load_op);
  small_load_op.x = small_input_op.y;
  auto small_input = small_graph.FindNode("small_input");
  auto small_load = small_graph.FindNode("small_load");
  small_load->outputs[0].attr.repeats = {af::ops::One};
  small_load->outputs[0].attr.strides = {af::ops::One};
  small_load->outputs[0].attr.dtype = ge::DT_FLOAT16;

  af::AscGraph large_graph("large_graph");
  af::ascir_op::Data large_input_op("large_input", large_graph);
  large_input_op.ir_attr.SetIndex(0);
  af::ascir_op::Load large_load_op("large_load");
  large_load_op.ir_attr.SetOffset(af::Expression::Parse("12288"));
  large_graph.AddNode(large_load_op);
  large_load_op.x = large_input_op.y;
  auto large_load = large_graph.FindNode("large_load");
  large_load->outputs[0].attr.repeats = {af::Expression::Parse("4096")};
  large_load->outputs[0].attr.strides = {af::ops::One};
  large_load->outputs[0].attr.dtype = ge::DT_FLOAT16;

  ascir::ScheduledResult schedule_result;
  schedule_result.schedule_groups.resize(1);
  schedule_result.schedule_groups[0].impl_graphs = {small_graph, large_graph};

  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.input_nodes.push_back(small_input);
  fused_schedule_result.node_idx_to_scheduled_results.push_back({schedule_result});

  const std::string malloc_def = this->PGOSearchTensorMallocDef(fused_schedule_result);
  EXPECT_NE(malloc_def.find("  size_t input0_size = 2;\n"), std::string::npos);
  EXPECT_NE(malloc_def.find("  input0_size = std::max(input0_size, static_cast<size_t>(32768));\n"), std::string::npos);
}

TEST_F(TestCodegenTiling, TestPGOSearchTensorMallocDefRejectsSymbolicCandidateSize) {
  af::AscGraph fallback_graph("fallback_graph");
  af::ascir_op::Store fallback_store_op("fallback_store");
  af::ascir_op::Output fallback_output_op("fallback_output");
  fallback_output_op.ir_attr.SetIndex(0);
  fallback_graph.AddNode(fallback_store_op);
  fallback_graph.AddNode(fallback_output_op);
  fallback_output_op.x = fallback_store_op.y;
  auto fallback_output = fallback_graph.FindNode("fallback_output");
  fallback_output->inputs[0].attr.repeats = {af::Expression::Parse("1024")};
  fallback_output->inputs[0].attr.strides = {af::ops::One};
  fallback_output->inputs[0].attr.dtype = ge::DT_FLOAT16;

  af::AscGraph graph("symbolic_graph");
  af::ascir_op::Store store_op("store");
  store_op.ir_attr.SetOffset(af::Expression::Parse("A_org_size"));
  af::ascir_op::Output output_op("output");
  output_op.ir_attr.SetIndex(0);
  graph.AddNode(store_op);
  graph.AddNode(output_op);
  output_op.x = store_op.y;
  auto output = graph.FindNode("output");
  output->inputs[0].attr.repeats = {af::Expression::Parse("A_org_size")};
  output->inputs[0].attr.strides = {af::ops::One};
  output->inputs[0].attr.dtype = ge::DT_FLOAT16;

  ascir::ScheduledResult schedule_result;
  schedule_result.schedule_groups.resize(1);
  schedule_result.schedule_groups[0].impl_graphs = {graph};
  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.output_nodes.push_back(fallback_output);
  fused_schedule_result.node_idx_to_scheduled_results.push_back({schedule_result});

  const std::string malloc_def = this->PGOSearchTensorMallocDef(fused_schedule_result);
  EXPECT_NE(malloc_def.find("Invalid or symbolic PGO output0 memory size"), std::string::npos);
  EXPECT_NE(malloc_def.find("return FAILED;"), std::string::npos);
  EXPECT_EQ(malloc_def.find("A_org_size"), std::string::npos);
}

TEST_F(TestCodegenTiling, TestPGOSearchTensorMallocDefResolvesReducePhase2OutputSize) {
  af::AscGraph fallback_graph("fallback_graph");
  af::ascir_op::Store fallback_store_op("fallback_store");
  af::ascir_op::Output fallback_output_op("fallback_output");
  fallback_output_op.ir_attr.SetIndex(0);
  fallback_graph.AddNode(fallback_store_op);
  fallback_graph.AddNode(fallback_output_op);
  fallback_output_op.x = fallback_store_op.y;
  auto fallback_output = fallback_graph.FindNode("fallback_output");
  fallback_output->inputs[0].attr.repeats = {af::Expression::Parse("24")};
  fallback_output->inputs[0].attr.strides = {af::ops::One};
  fallback_output->inputs[0].attr.dtype = ge::DT_FLOAT16;

  af::AscGraph phase1_graph("phase1_graph");
  af::AscGraph phase2_graph("phase2_graph");
  const auto a_org_size = phase2_graph.CreateSizeVar("A_org_size");
  af::ascir_op::Store phase2_store_op("phase2_store");
  phase2_store_op.ir_attr.SetOffset(af::ops::Zero);
  af::ascir_op::Output phase2_output_op("phase2_output");
  phase2_output_op.ir_attr.SetIndex(0);
  phase2_graph.AddNode(phase2_store_op);
  phase2_graph.AddNode(phase2_output_op);
  phase2_output_op.x = phase2_store_op.y;
  auto phase2_output = phase2_graph.FindNode("phase2_output");
  phase2_output->inputs[0].attr.repeats = {af::ops::One, a_org_size};
  phase2_output->inputs[0].attr.strides = {af::ops::Zero, af::ops::One};
  phase2_output->inputs[0].attr.dtype = ge::DT_FLOAT16;

  ascir::ScheduledResult schedule_result;
  schedule_result.schedule_groups.resize(2);
  schedule_result.schedule_groups[0].impl_graphs = {phase1_graph};
  schedule_result.schedule_groups[1].impl_graphs = {phase2_graph};
  schedule_result.var_relations[1][0]["A_org_size"] = af::Expression::Parse("24");
  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.output_nodes.push_back(fallback_output);
  fused_schedule_result.node_idx_to_scheduled_results.push_back({schedule_result});

  const std::string malloc_def = this->PGOSearchTensorMallocDef(fused_schedule_result);
  EXPECT_NE(malloc_def.find("size_t output0_size = 48;"), std::string::npos);
  EXPECT_EQ(malloc_def.find("Invalid or symbolic PGO output0 memory size"), std::string::npos);
}

TEST_F(TestCodegenTiling, TestCalculateTensorMemorySizeStrWithNoRepeatsOrStrides) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", af::ops::One);

  af::ascir_op::Data x_op("x", graph);
  x_op.ir_attr.SetIndex(0);

  auto x = graph.FindNode("x");

  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.tensor_id = 0;
  x->attr.api.unit = af::ComputeUnit::kUnitNone;
  x->outputs[0].attr.repeats = {};
  x->outputs[0].attr.strides = {};

  std::string memory_size = this->CalculateTensorMemorySizeStr(x->outputs[0]);
  const std::string expect = "0";
  ASSERT_EQ(memory_size, expect);
}

TEST_F(TestCodegenTiling, TestCalculateTensorMemorySizeStrWithZeroFirstStride) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data x_op("x", graph);
  x_op.ir_attr.SetIndex(0);

  auto x = graph.FindNode("x");

  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.tensor_id = 0;
  x->attr.api.unit = af::ComputeUnit::kUnitNone;
  x->outputs[0].attr.dtype = ge::DT_FLOAT16;
  x->outputs[0].attr.axis = {z0.id, z1.id};
  x->outputs[0].attr.repeats = {s0, s1};
  x->outputs[0].attr.strides = {af::ops::Zero, af::ops::One};

  std::string memory_size = this->CalculateTensorMemorySizeStr(x->outputs[0]);
  const std::string expect = std::string(af::sym::Mul(s1, af::Expression::Parse("2")).Simplify().Str().get());
  ASSERT_EQ(memory_size, expect);
}

TEST_F(TestCodegenTiling, TestCalculateTensorMemorySizeStrWithBfloat16) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  af::ascir_op::Data x_op("x", graph);
  x_op.ir_attr.SetIndex(0);

  auto x = graph.FindNode("x");

  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.tensor_id = 0;
  x->attr.api.unit = af::ComputeUnit::kUnitNone;
  x->outputs[0].attr.dtype = ge::DT_BF16;
  x->outputs[0].attr.axis = {z0.id};
  x->outputs[0].attr.repeats = {s0};
  x->outputs[0].attr.strides = {af::ops::One};

  std::string memory_size = this->CalculateTensorMemorySizeStr(x->outputs[0]);
  const std::string expect = std::string(af::sym::Mul(s0, af::Expression::Parse("2")).Simplify().Str().get());
  ASSERT_EQ(memory_size, expect);
}

TEST_F(TestCodegenTiling, TestCalculateTensorMemorySizeStrWithOnlyZeroStride) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  af::ascir_op::Data x_op("x", graph);
  x_op.ir_attr.SetIndex(0);

  auto x = graph.FindNode("x");

  x->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  x->outputs[0].attr.mem.tensor_id = 0;
  x->attr.api.unit = af::ComputeUnit::kUnitNone;
  x->outputs[0].attr.dtype = ge::DT_FLOAT16;
  x->outputs[0].attr.axis = {z0.id};
  x->outputs[0].attr.repeats = {s0};
  x->outputs[0].attr.strides = {af::ops::Zero};

  std::string memory_size = this->CalculateTensorMemorySizeStr(x->outputs[0]);
  const std::string expect = std::string(af::sym::Mul(af::ops::One, af::Expression::Parse("2")).Simplify().Str().get());
  ASSERT_EQ(memory_size, expect);
}

TEST_F(TestCodegenTiling, TestCalculateTensorMemorySizeStrWithNonContiguousStrides) {
  af::AscGraph graph("test_graph");
  af::ascir_op::Data x_op("x", graph);
  auto x = graph.FindNode("x");

  x->outputs[0].attr.dtype = ge::DT_FLOAT16;
  x->outputs[0].attr.repeats = {af::Expression::Parse("1024"), af::Expression::Parse("2")};
  x->outputs[0].attr.strides = {af::Expression::Parse("1"), af::Expression::Parse("1024")};

  EXPECT_EQ(this->CalculateTensorMemorySizeStr(x->outputs[0]), "4096");
}

TEST_F(TestCodegenTiling, TestCalculateTensorMemorySizeStrRejectsRankMismatch) {
  af::AscGraph graph("test_graph");
  af::ascir_op::Data x_op("x", graph);
  auto x = graph.FindNode("x");

  x->outputs[0].attr.dtype = ge::DT_FLOAT16;
  x->outputs[0].attr.repeats = {af::Expression::Parse("16"), af::Expression::Parse("8")};
  x->outputs[0].attr.strides = {af::Expression::Parse("8")};

  EXPECT_EQ(this->CalculateTensorMemorySizeStr(x->outputs[0]), "0");
}

TEST_F(TestCodegenTiling, TestCalculateTensorMemorySizeStrRejectsNegativeStrideAndOffset) {
  af::AscGraph graph("test_graph");
  af::ascir_op::Data x_op("x", graph);
  auto x = graph.FindNode("x");

  x->outputs[0].attr.dtype = ge::DT_FLOAT16;
  x->outputs[0].attr.repeats = {af::Expression::Parse("16")};
  x->outputs[0].attr.strides = {af::Expression::Parse("-1")};
  EXPECT_EQ(this->CalculateTensorMemorySizeStr(x->outputs[0]), "0");

  x->outputs[0].attr.strides = {af::ops::One};
  EXPECT_EQ(this->CalculateTensorMemorySizeStr(x->outputs[0], af::Expression::Parse("-1")), "0");
}

TEST_F(TestCodegenTiling, TestCalculateTensorMemorySizeStrRejectsOverflow) {
  af::AscGraph graph("test_graph");
  af::ascir_op::Data x_op("x", graph);
  auto x = graph.FindNode("x");

  x->outputs[0].attr.dtype = ge::DT_FLOAT16;
  x->outputs[0].attr.repeats = {af::Expression::Parse("9223372036854775807")};
  x->outputs[0].attr.strides = {af::ops::One};

  EXPECT_EQ(this->CalculateTensorMemorySizeStr(x->outputs[0]), "0");
}

// 创建 BatchMatMul+Store+Output 尾链, 通过 AscOpOutput 引用连接避免 Load 按值传递
void CreateMatmulTailChain(af::AscGraph &graph, const af::AscOpOutput &x1, const af::AscOpOutput &x2,
                           const af::Axis &z0, const af::Axis &z1, const af::Expression &s0, const af::Expression &s1,
                           ge::DataType dtype, bool adj_x1, bool enable_hf32) {
  af::ascir_op::BatchMatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = x1;
  matmul.x2 = x2;
  matmul.y.dtype = dtype;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, af::ops::One};
  matmul.attr.api.compute_type = af::ComputeType::kComputeCube;
  matmul.ir_attr.SetAdj_x1(adj_x1 ? 1 : 0);
  matmul.ir_attr.SetAdj_x2(0);
  matmul.ir_attr.SetHas_relu(0);
  matmul.ir_attr.SetEnable_hf32(enable_hf32 ? 1 : 0);
  matmul.ir_attr.SetOffset_x(0);

  af::ascir_op::Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = matmul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = dtype;
  *store_op.y.strides = {s1, af::ops::One};
  *store_op.y.repeats = {s0, s1};
  store_op.ir_attr.SetOffset(af::ops::One);

  af::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = dtype;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
}

// 构造 matmul graph, 可自定义 K 值/动态/enable_hf32/transpose/dtype
// adj_x1=true: input0 shape=[K, M], K 在 shape[-2]
// adj_x1=false: input0 shape=[M, K], K 在 shape[-1]
void CreateMatmulGraphForFp32LargeK(af::AscGraph &graph, int64_t k_value, bool is_dynamic, bool enable_hf32,
                                    bool adj_x1, ge::DataType dtype = ge::DT_FLOAT) {
  af::Expression k_expr = is_dynamic ? graph.CreateSizeVar("k_var") : graph.CreateSizeVar(k_value);
  af::Expression m_expr = graph.CreateSizeVar(1);
  af::Expression s0 = adj_x1 ? k_expr : m_expr;  // K 在 shape[0] (transpose) 或 M
  af::Expression s1 = adj_x1 ? m_expr : k_expr;  // M 或 K 在 shape[1] (非transpose)
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  // input0: Data -> Load
  af::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = dtype;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0.y.strides = {s1, af::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);
  af::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = dtype;
  *load0.y.strides = {s1, af::ops::One};
  *load0.y.repeats = {s0, s1};

  // input1: Data -> Load
  af::ascir_op::Data data1("data1", graph);
  data1.y.dtype = dtype;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data1.y.strides = {af::ops::Zero, af::ops::Zero};
  *data1.y.repeats = {af::ops::One, af::ops::One};
  data1.ir_attr.SetIndex(1);
  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = dtype;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {af::ops::Zero, af::ops::Zero};
  *load1.y.repeats = {af::ops::One, af::ops::One};

  CreateMatmulTailChain(graph, load0.y, load1.y, z0, z1, s0, s1, dtype, adj_x1, enable_hf32);
}

// 从 graph 提取 MatMulCubeInfo
#define EXTRACT_CUBE_INFO(graph)                           \
  ([this, &graph]() -> codegen::MatMulCubeInfo {           \
    codegen::MatMulCubeInfo info;                          \
    this->ExtractMatMulCubeInfoFromImplGraph(graph, info); \
    return info;                                           \
  }())

// ==================== GenFp32LargeKCondition 测试 ====================
TEST_F(TestCodegenTiling, GenFp32LargeKConditionShouldReturnFalseWhenTypeSizeNotFp32) {
  af::AscGraph graph("fp16_matmul");
  CreateMatmulGraphForFp32LargeK(graph, 4096, false, false, true, ge::DT_FLOAT16);
  auto cube_info = EXTRACT_CUBE_INFO(graph);
  EXPECT_EQ(this->GenFp32LargeKCondition(cube_info), "false");
}

TEST_F(TestCodegenTiling, GenFp32LargeKConditionShouldReturnFalseWhenEnableHf32) {
  af::AscGraph graph("hf32_matmul");
  CreateMatmulGraphForFp32LargeK(graph, 4096, false, true, true, ge::DT_FLOAT);
  auto cube_info = EXTRACT_CUBE_INFO(graph);
  EXPECT_EQ(this->GenFp32LargeKCondition(cube_info), "false");
}

TEST_F(TestCodegenTiling, GenFp32LargeKConditionShouldReturnFalseWhenStaticKLe2048) {
  af::AscGraph graph("small_k_matmul");
  CreateMatmulGraphForFp32LargeK(graph, 1024, false, false, true, ge::DT_FLOAT);
  auto cube_info = EXTRACT_CUBE_INFO(graph);
  EXPECT_EQ(this->GenFp32LargeKCondition(cube_info), "false");
}

TEST_F(TestCodegenTiling, GenFp32LargeKConditionShouldReturnTrueWhenStaticKGT2048) {
  af::AscGraph graph("large_k_matmul");
  CreateMatmulGraphForFp32LargeK(graph, 4096, false, false, true, ge::DT_FLOAT);
  auto cube_info = EXTRACT_CUBE_INFO(graph);
  EXPECT_EQ(this->GenFp32LargeKCondition(cube_info), "true");
}

TEST_F(TestCodegenTiling, GenFp32LargeKConditionShouldReturnRuntimeExprWhenDynamicK) {
  af::AscGraph graph("dynamic_k_matmul");
  CreateMatmulGraphForFp32LargeK(graph, 0, true, false, true, ge::DT_FLOAT);
  auto cube_info = EXTRACT_CUBE_INFO(graph);
  EXPECT_EQ(this->GenFp32LargeKCondition(cube_info), "(static_cast<int64_t>(k_var) >= 2048)");
}

TEST_F(TestCodegenTiling, GenFp32LargeKConditionShouldReturnTrueWhenNoTransposeStaticKGT2048) {
  af::AscGraph graph("no_transpose_large_k_matmul");
  CreateMatmulGraphForFp32LargeK(graph, 4096, false, false, false, ge::DT_FLOAT);
  auto cube_info = EXTRACT_CUBE_INFO(graph);
  EXPECT_EQ(this->GenFp32LargeKCondition(cube_info), "true");
}

TEST_F(TestCodegenTiling, GenFp32LargeKConditionShouldReturnFalseWhenMatmulNodeIsNull) {
  codegen::MatMulCubeInfo cube_info;
  cube_info.type_size = 4U;
  cube_info.enable_hf32 = 0;
  EXPECT_EQ(this->GenFp32LargeKCondition(cube_info), "false");
}

void CreateMatmulGraph(af::AscGraph &graph, bool is_dynamic = false) {
  af::Expression s0;
  af::Expression s1;
  if (is_dynamic) {
    s0 = graph.CreateSizeVar("s0");
    s1 = graph.CreateSizeVar("s1");
  } else {
    s0 = graph.CreateSizeVar(31);
    s1 = graph.CreateSizeVar(1);
  }
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0.y.strides = {s1, af::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {s1, af::ops::One};
  *load0.y.repeats = {s0, s1};

  af::ascir_op::Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data1.y.repeats = {af::ops::One, af::ops::One};
  *data1.y.strides = {af::ops::Zero, af::ops::Zero};
  data1.ir_attr.SetIndex(1);

  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {af::ops::Zero, af::ops::Zero};
  *load1.y.repeats = {af::ops::One, af::ops::One};

  af::ascir_op::BatchMatMul matmul("matmul");
  matmul.attr.sched.axis = {z0.id, z1.id};
  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  matmul.y.dtype = ge::DT_FLOAT;
  *matmul.y.axis = {z0.id, z1.id};
  *matmul.y.repeats = {s0, s1};
  *matmul.y.strides = {s1, af::ops::One};
  matmul.attr.api.compute_type = af::ComputeType::kComputeCube;
  matmul.ir_attr.SetAdj_x1(1);
  matmul.ir_attr.SetAdj_x2(0);
  matmul.ir_attr.SetHas_relu(1);
  matmul.ir_attr.SetEnable_hf32(1);
  matmul.ir_attr.SetOffset_x(6);

  af::ascir_op::Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = matmul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1, af::ops::One};
  *store_op.y.repeats = {s0, s1};
  store_op.ir_attr.SetOffset(af::ops::One);

  af::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
}

TEST_F(TestCodegenTiling, TestMatmulElemwiseFuse) {
  af::AscGraph graph("matmul_elemwise_pro");

  auto s0 = graph.CreateSizeVar(64);
  auto s1 = graph.CreateSizeVar(64);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = ge::DT_FLOAT;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0.y.strides = {s1, af::ops::One};
  *data0.y.repeats = {s0, s1};
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  load0.y.dtype = ge::DT_FLOAT;
  *load0.y.strides = {s1, af::ops::One};
  *load0.y.repeats = {s0, s1};

  af::ascir_op::Abs abs("abs");
  graph.AddNode(abs);
  abs.x = load0.y;
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.y.dtype = ge::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {s0, s1};
  *abs.y.strides = {s1, af::ops::One};
  abs.attr.api.compute_type = af::ComputeType::kComputeElewise;

  af::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.attr.sched.axis = {z0.id, z1.id};
  scalar0.ir_attr.SetValue("0");
  scalar0.y.dtype = ge::DT_FLOAT;
  *scalar0.y.axis = {z0.id, z1.id};
  *scalar0.y.repeats = {af::ops::One, af::ops::One};
  *scalar0.y.strides = {af::ops::Zero, af::ops::Zero};

  af::ascir_op::Broadcast broadcast0("broadcast0");
  broadcast0.x = scalar0.y;
  broadcast0.attr.sched.axis = {z0.id, z1.id};
  *broadcast0.y.axis = {z0.id, z1.id};
  broadcast0.y.dtype = ge::DT_FLOAT;
  *broadcast0.y.repeats = {af::ops::One, s1};
  *broadcast0.y.strides = {af::ops::Zero, af::ops::One};

  af::ascir_op::Broadcast broadcast1("broadcast1");
  broadcast1.x = broadcast0.y;
  broadcast1.attr.sched.axis = {z0.id, z1.id};
  *broadcast1.y.axis = {z0.id, z1.id};
  broadcast1.y.dtype = ge::DT_FLOAT;
  *broadcast1.y.repeats = {s0, s1};
  *broadcast1.y.strides = {s1, af::ops::One};

  af::ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z0.id, z1.id};
  add_op.x1 = abs.y;
  add_op.x2 = broadcast1.y;
  add_op.y.dtype = ge::DT_FLOAT;
  *add_op.y.axis = {z0.id, z1.id};
  *add_op.y.repeats = {s0, s1};
  *add_op.y.strides = {s1, af::ops::One};

  af::ascir_op::Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT;
  data1.attr.sched.axis = {z0.id, z1.id};
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data1.y.repeats = {af::ops::One, af::ops::One};
  *data1.y.strides = {af::ops::Zero, af::ops::Zero};
  data1.ir_attr.SetIndex(1);

  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.strides = {af::ops::Zero, af::ops::Zero};
  *load1.y.repeats = {af::ops::One, af::ops::One};

  af::ascir_op::Broadcast broadcast2("broadcast2");
  broadcast2.x = load1.y;
  broadcast2.attr.sched.axis = {z0.id, z1.id};
  *broadcast2.y.axis = {z0.id, z1.id};
  broadcast2.y.dtype = ge::DT_FLOAT;
  *broadcast2.y.repeats = {af::ops::One, s1};
  *broadcast2.y.strides = {af::ops::Zero, af::ops::One};

  af::ascir_op::Broadcast broadcast3("broadcast3");
  broadcast3.x = broadcast2.y;
  broadcast3.attr.sched.axis = {z0.id, z1.id};
  *broadcast3.y.axis = {z0.id, z1.id};
  broadcast3.y.dtype = ge::DT_FLOAT;
  *broadcast3.y.repeats = {s0, s1};
  *broadcast3.y.strides = {s1, af::ops::One};

  af::ascir_op::Mul mul("mul");
  mul.attr.sched.axis = {z0.id, z1.id};
  mul.x1 = add_op.y;
  mul.x2 = broadcast3.y;
  mul.y.dtype = ge::DT_FLOAT;
  *mul.y.axis = {z0.id, z1.id};
  *mul.y.repeats = {s0, s1};
  *mul.y.strides = {s1, af::ops::One};

  af::ascir_op::Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = mul.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {s1, af::ops::One};
  *store_op.y.repeats = {s0, s1};

  af::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  auto x1Local = graph.FindNode("data0");
  x1Local->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  x1Local->outputs[0].attr.mem.hardware = af::MemHardware::kMemHardwareUB;
  x1Local->outputs[0].attr.mem.position = af::Position::kPositionVecIn;

  af::AscGraph mm_graph("mutmul");
  CreateMatmulGraph(mm_graph);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused_schedule_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);

  ascir::ScheduleGroup schedule_group2;
  schedule_group2.impl_graphs.push_back(mm_graph);
  fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups.push_back(schedule_group2);
  fused_schedule_result.node_idx_to_scheduled_results[0][0].cube_type = ascir::CubeTemplateType::kUBFuse;
  ;
  const std::map<std::string, std::string> shape_info;
  auto res = this->Generate(fused_schedule_result, shape_info, "", "0");

  std::fstream tiling_func("Mutmul_fuse_tiling_func.cpp", std::ios::out);
  tiling_func << res["tiling_def_and_tiling_const"];

  auto pos = res["tiling_def_and_tiling_const"].find("extern \"C\" int64_t FindBestTilingKey");
  ASSERT_NE(pos, std::string::npos);
}

// ==================== Conv2DOffset 融合测试 ====================

TEST_F(TestCodegenTiling, TestConv2DOffsetFuse) {
  af::AscGraph graph("conv2d_offset_elemwise_pro");
  CreateElemwiseGraphWithRelu(graph);

  af::AscGraph conv2d_offset_graph("conv2d_offset");
  CreateConv2DOffsetGraph(conv2d_offset_graph);

  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused_schedule_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);

  ascir::ScheduleGroup schedule_group2;
  schedule_group2.impl_graphs.push_back(conv2d_offset_graph);
  fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups.push_back(schedule_group2);
  fused_schedule_result.node_idx_to_scheduled_results[0][0].cube_type = ascir::CubeTemplateType::kUBFuse;

  const std::map<std::string, std::string> shape_info;
  auto res = this->Generate(fused_schedule_result, shape_info, "", "0");

  std::fstream tiling_func("Conv2d_offset_fuse_tiling_func.cpp", std::ios::out);
  tiling_func << res["tiling_def_and_tiling_const"];

  VerifyConv2DOffsetTiling(res);
}

// ==================== Conv2DOffsetBias 融合测试 ====================

TEST_F(TestCodegenTiling, TestConv2DOffsetBiasFuse) {
  af::AscGraph graph("conv2d_offset_bias_elemwise_pro");
  CreateElemwiseGraphWithRelu(graph);

  af::AscGraph conv2d_offset_bias_graph("conv2d_offset_bias");
  CreateConv2DOffsetBiasGraph(conv2d_offset_bias_graph);

  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused_schedule_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);

  ascir::ScheduleGroup schedule_group2;
  schedule_group2.impl_graphs.push_back(conv2d_offset_bias_graph);
  fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups.push_back(schedule_group2);
  fused_schedule_result.node_idx_to_scheduled_results[0][0].cube_type = ascir::CubeTemplateType::kUBFuse;

  const std::map<std::string, std::string> shape_info;
  auto res = this->Generate(fused_schedule_result, shape_info, "", "0");

  std::fstream tiling_func("Conv2d_offset_bias_fuse_tiling_func.cpp", std::ios::out);
  tiling_func << res["tiling_def_and_tiling_const"];

  VerifyTilingCodeBasic(res);
}

namespace {
static ascir::FusedScheduledResult GenMultiGroupFusedScheduleResult() {
  af::AscGraph graph1("graph1");
  af::AscGraph graph2("graph2");

  ascir::ScheduleGroup schedule_group1;
  schedule_group1.impl_graphs.push_back(graph1);

  ascir::ScheduleGroup schedule_group2;
  schedule_group2.impl_graphs.push_back(graph1);
  schedule_group2.impl_graphs.push_back(graph2);

  ascir::ScheduledResult schedule_result;
  schedule_result.schedule_groups.push_back(schedule_group1);
  schedule_result.schedule_groups.push_back(schedule_group2);

  ascir::FusedScheduledResult fused_schedule_result;
  std::vector<ascir::ScheduledResult> graph0_results = {schedule_result};
  fused_schedule_result.node_idx_to_scheduled_results.emplace_back(std::move(graph0_results));
  return fused_schedule_result;
}

static ascir::ImplGraph GenGraphWithSizeVar(const std::string &graph_name, const std::string &var_name) {
  ascir::ImplGraph graph(graph_name.c_str());
  auto size = graph.CreateSizeVar(var_name.c_str());
  (void)graph.CreateAxis("z0", size);
  return graph;
}

static ascir::ScheduleGroup GenScheduleGroupWithInterleavedApiTilingField() {
  constexpr int64_t kSecondDimSize = 32;
  constexpr int64_t kThirdDimSize = 128;
  ascir::ImplGraph graph0("graph0");
  auto s0 = graph0.CreateSizeVar("s0");
  auto fixed_s1 = graph0.CreateSizeVar(kSecondDimSize);
  auto fixed_s2 = graph0.CreateSizeVar(kThirdDimSize);
  auto z0 = graph0.CreateAxis("z0", s0);
  auto z1 = graph0.CreateAxis("z1", fixed_s1);
  auto z2 = graph0.CreateAxis("z2", fixed_s2);

  af::ascir_op::Data data("data", graph0);
  data.y.dtype = ge::DT_FLOAT;
  *data.y.axis = {z0.id, z1.id, z2.id};
  *data.y.repeats = {s0, fixed_s1, fixed_s2};
  *data.y.strides = {fixed_s1 * fixed_s2, fixed_s2, af::ops::One};

  af::ascir_op::Load load("load");
  graph0.AddNode(load);
  load.x = data.y;
  load.y.dtype = ge::DT_FLOAT;
  *load.y.axis = {z0.id, z1.id, z2.id};
  *load.y.repeats = {s0, fixed_s1, fixed_s2};
  *load.y.strides = {fixed_s1 * fixed_s2, fixed_s2, af::ops::One};
  *load.y.vectorized_axis = {z0.id, z1.id, z2.id};
  *load.y.vectorized_strides = {fixed_s1 * fixed_s2, fixed_s2, af::ops::One};

  af::ascir_op::Transpose transpose("Transpose");
  graph0.AddNode(transpose);
  transpose.x = load.y;
  transpose.y.dtype = ge::DT_FLOAT;
  *transpose.y.axis = {z0.id, z2.id, z1.id};
  *transpose.y.repeats = {s0, fixed_s2, fixed_s1};
  *transpose.y.strides = {fixed_s1 * fixed_s2, fixed_s1, af::ops::One};
  *transpose.y.vectorized_axis = {z0.id, z2.id, z1.id};
  *transpose.y.vectorized_strides = {fixed_s1 * fixed_s2, fixed_s1, af::ops::One};
  graph0.FindNode("Transpose")->outputs[0].attr.que.id = 0;

  ascir::ImplGraph graph1("graph1");
  auto s1 = graph1.CreateSizeVar("s1");
  (void)graph1.CreateAxis("z1", s1);

  ascir::ScheduleGroup schedule_group;
  schedule_group.impl_graphs = {graph0, graph1};
  return schedule_group;
}

static ascir::FusedScheduledResult GenSingleGroupFusedScheduleResultWithApiTilingField() {
  ascir::ScheduledResult scheduled_result;
  scheduled_result.schedule_groups.push_back(GenScheduleGroupWithInterleavedApiTilingField());

  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.node_idx_to_scheduled_results.push_back({scheduled_result});
  return fused_schedule_result;
}

static ascir::FusedScheduledResult GenMultiGroupFusedScheduleResultWithApiTilingField() {
  ascir::ScheduledResult scheduled_result;
  scheduled_result.schedule_groups.push_back(GenScheduleGroupWithInterleavedApiTilingField());
  scheduled_result.schedule_groups.emplace_back();

  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.node_idx_to_scheduled_results.push_back({scheduled_result});
  return fused_schedule_result;
}

static ascir::FusedScheduledResult GenMultiGroupFusedScheduleResultWithSizeVar(const std::string &var_name) {
  ascir::ScheduleGroup schedule_group1;
  schedule_group1.impl_graphs.push_back(GenGraphWithSizeVar("graph1", var_name));

  ascir::ScheduleGroup schedule_group2;
  schedule_group2.impl_graphs.push_back(GenGraphWithSizeVar("graph2", var_name));

  ascir::ScheduledResult schedule_result;
  schedule_result.schedule_groups.push_back(schedule_group1);
  schedule_result.schedule_groups.push_back(schedule_group2);

  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.origin_vars.push_back(af::Symbol(var_name.c_str()));
  fused_schedule_result.node_idx_to_scheduled_results.push_back({schedule_result});
  return fused_schedule_result;
}
}  // namespace

TEST_F(TestCodegenTiling, GenerateForInductorGetTilingDataReprShouldContainStableFields) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_NE(tiling_impl.find("GetTilingDataRepr returns a valid C++ designated initializer string"), std::string::npos);
  EXPECT_NE(tiling_impl.find("emit_field(\"block_dim\", tiling_data->get_block_dim()"), std::string::npos);
  EXPECT_NE(tiling_impl.find("emit_field(\"corenum\", tiling_data->get_corenum()"), std::string::npos);
  EXPECT_NE(tiling_impl.find("emit_field(\"ub_size\", tiling_data->get_ub_size()"), std::string::npos);
  EXPECT_NE(tiling_impl.find("emit_field(\"hbm_size\", tiling_data->get_hbm_size()"), std::string::npos);
  EXPECT_TRUE(tiling_impl.find("emit_field(\"tiling_key\"") != std::string::npos ||
              tiling_impl.find("emit_field(\"graph0_tiling_key\"") != std::string::npos);
}

TEST_F(TestCodegenTiling, GenerateForInductorGetTilingDataReprShouldKeepWorkspaceBeforeSymbols) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s1"), af::Symbol("s0")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  size_t tiling_key_pos = tiling_impl.find("emit_field(\"tiling_key\"");
  if (tiling_key_pos == std::string::npos) {
    tiling_key_pos = tiling_impl.find("emit_field(\"graph0_tiling_key\"");
  }
  const auto s0_pos = tiling_impl.find("emit_field(\"s0\"");
  const auto s1_pos = tiling_impl.find("emit_field(\"s1\"");
  ASSERT_NE(tiling_key_pos, std::string::npos);
  if (s0_pos != std::string::npos) {
    EXPECT_LT(tiling_key_pos, s0_pos);
  }
  if (s1_pos != std::string::npos) {
    EXPECT_LT(tiling_key_pos, s1_pos);
  }
}

TEST_F(TestCodegenTiling, GenerateForInductorGetTilingDataReprShouldUseGraphLevelTilingKeysForMultiGroup) {
  auto fused_schedule_result = GenMultiGroupFusedScheduleResult();
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_NE(tiling_impl.find("emit_field(\"graph0_tiling_key\""), std::string::npos);
  EXPECT_EQ(tiling_impl.find("emit_field(\"tiling_key\""), std::string::npos);
}

TEST_F(TestCodegenTiling, GenerateForInductorGetTilingDataReprShouldKeepZeroValuedStableFields) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_NE(tiling_impl.find("emit_field(\"block_dim\", tiling_data->get_block_dim()"), std::string::npos);
  EXPECT_NE(tiling_impl.find("emit_field(\"corenum\", tiling_data->get_corenum()"), std::string::npos);
  EXPECT_NE(tiling_impl.find("emit_field(\"ub_size\", tiling_data->get_ub_size()"), std::string::npos);
  EXPECT_NE(tiling_impl.find("emit_field(\"hbm_size\", tiling_data->get_hbm_size()"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("if (tiling_data->get_block_dim() != 0)"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("if (tiling_data->get_corenum() != 0)"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("if (tiling_data->get_ub_size() != 0)"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("if (tiling_data->get_hbm_size() != 0)"), std::string::npos);
}

TEST_F(TestCodegenTiling, GetTilingDataReprShouldKeepSingleGroupApiTilingFieldInDeclarationOrder) {
  ge::PlatformContext::GetInstance().SetPlatform("2201");
  const auto fused_schedule_result = GenSingleGroupFusedScheduleResultWithApiTilingField();
  const auto repr = this->GenGetTilingDataReprFuncForInductor(fused_schedule_result, "AutofuseTilingData");

  const auto s0_pos = repr.find("emit_field(\"s0\"");
  const auto api_tiling_pos = repr.find(".Transpose_tilingData_0 = {");
  const auto s1_pos = repr.find("emit_field(\"s1\"");
  const auto q0_pos = repr.find("emit_field(\"q0_size\"");
  ASSERT_NE(s0_pos, std::string::npos);
  ASSERT_NE(api_tiling_pos, std::string::npos);
  ASSERT_NE(s1_pos, std::string::npos);
  ASSERT_NE(q0_pos, std::string::npos);
  EXPECT_LT(s0_pos, api_tiling_pos);
  EXPECT_LT(api_tiling_pos, s1_pos);
  EXPECT_LT(s1_pos, q0_pos);
}

TEST_F(TestCodegenTiling, GetTilingDataReprShouldKeepMultiGroupApiTilingFieldInDeclarationOrder) {
  ge::PlatformContext::GetInstance().SetPlatform("2201");
  const auto fused_schedule_result = GenMultiGroupFusedScheduleResultWithApiTilingField();
  const auto repr = this->GenGetTilingDataReprFuncForInductor(fused_schedule_result, "AutofuseTilingData");

  const auto s0_pos = repr.find("emit_sub(\"s0\"");
  const auto api_tiling_pos = repr.find(".Transpose_tilingData_0 = {");
  const auto s1_pos = repr.find("emit_sub(\"s1\"");
  const auto q0_pos = repr.find("emit_sub(\"q0_size\"");
  ASSERT_NE(s0_pos, std::string::npos);
  ASSERT_NE(api_tiling_pos, std::string::npos);
  ASSERT_NE(s1_pos, std::string::npos);
  ASSERT_NE(q0_pos, std::string::npos);
  EXPECT_LT(s0_pos, api_tiling_pos);
  EXPECT_LT(api_tiling_pos, s1_pos);
  EXPECT_LT(s1_pos, q0_pos);
}

TEST_F(TestCodegenTiling, GenerateForInductorShouldContainTopnMainOutputAbi) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_NE(tiling_impl.find("extern \"C\" int64_t GenerateTopnSolutions("), std::string::npos);
  EXPECT_NE(tiling_impl.find("GetTilingDataRepr("), std::string::npos);
}

TEST_F(TestCodegenTiling, GenerateForInductorShouldUseGetTilingDataReprAsTilingDataValidationAid) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_NE(tiling_impl.find("GetTilingDataRepr("), std::string::npos);
}

// FinalTilingContext metadata is not available at the repr helper boundary.  Keep
// candidate diagnostics there-free until the final entry supplies the complete
// graph/result/group identity and pipe estimate.
TEST_F(TestCodegenTiling, FinalTilingEntriesDoNotEmitRuntimeSourceForCandidateRepr) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")}, af::Symbol(1));
  codegen_func_ = att::GenTilingImplAutoFuseV3;
  const auto inductor_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(inductor_files.find(codegen::kTilingDefAndConstIdentify) != inductor_files.end());
  const auto &inductor_source = inductor_files.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_NE(inductor_source.find("extern \"C\" int64_t GenerateTopnSolutions("), std::string::npos);
  EXPECT_NE(inductor_source.find("GetTilingDataRepr(&raw_candidate.tiling_data)"), std::string::npos);
  EXPECT_EQ(inductor_source.find("source=runtime"), std::string::npos);

  const auto tensorflow_files = this->Generate(fused_schedule_result, {}, "", "0");
  ASSERT_TRUE(tensorflow_files.find(codegen::kTilingDefAndConstIdentify) != tensorflow_files.end());
  const auto &tensorflow_source = tensorflow_files.at(codegen::kTilingDefAndConstIdentify);
  bool has_tensorflow_tiling_call = false;
  for (const auto &[name, source] : tensorflow_files) {
    (void)name;
    const size_t tensorflow_entry = source.find("extern \"C\" ge::graphStatus TilingFunc");
    if (tensorflow_entry == std::string::npos) {
      continue;
    }
    const size_t tensorflow_tiling_call = source.find("AutofuseTilingWithConfig", tensorflow_entry);
    has_tensorflow_tiling_call = has_tensorflow_tiling_call || tensorflow_tiling_call != std::string::npos;
    if (tensorflow_tiling_call != std::string::npos) {
      EXPECT_LT(tensorflow_entry, tensorflow_tiling_call);
    }
  }
  EXPECT_TRUE(has_tensorflow_tiling_call);
  EXPECT_EQ(tensorflow_source.find("source=runtime"), std::string::npos);
}

TEST_F(TestCodegenTiling, FinalTilingUsesSharedLogHeaderAndSelectionProducer) {
  const auto fused_schedule_result =
      this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")}, af::Symbol(1));
  codegen_func_ = att::GenTilingImplAutoFuseV3;
  const auto tensorflow_files = this->Generate(fused_schedule_result, {}, "", "0");
  ASSERT_NE(tensorflow_files.find(codegen::kTilingLogHeaderIdentify), tensorflow_files.end());
  const auto &log_header = tensorflow_files.at(codegen::kTilingLogHeaderIdentify);
  EXPECT_NE(log_header.find("struct FinalTilingContext"), std::string::npos);
  EXPECT_NE(log_header.find("bool ShouldEmitFinalTiling() noexcept"), std::string::npos);
  EXPECT_NE(log_header.find("FormatFinalTilingRecord"), std::string::npos);
  EXPECT_NE(log_header.find("FinalTilingChecksum"), std::string::npos);
  EXPECT_NE(log_header.find("int32_t score;"), std::string::npos);
  EXPECT_NE(log_header.find("std::to_string(context.score)"), std::string::npos);

  bool has_tensorflow_selection_producer = false;
  for (const auto &[name, source] : tensorflow_files) {
    (void)name;
    has_tensorflow_selection_producer =
        has_tensorflow_selection_producer || source.find("ShouldEmitFinalTiling()") != std::string::npos;
    has_tensorflow_selection_producer =
        has_tensorflow_selection_producer ||
        source.find("EmitFinalTilingLines(FormatFinalTilingRecord") != std::string::npos;
  }
  EXPECT_TRUE(has_tensorflow_selection_producer);
  bool has_pipe_estimate = false;
  bool has_summary_producer = false;
  for (const auto &[name, source] : tensorflow_files) {
    (void)name;
    has_pipe_estimate =
        has_pipe_estimate ||
        source.find("CollectPipeEstimates(tiling_data, final_context.pipe_estimates)") != std::string::npos;
    has_summary_producer =
        has_summary_producer || source.find("EmitFinalTilingSummary(final_summary)") != std::string::npos;
  }
  EXPECT_TRUE(has_pipe_estimate);
  bool has_score = false;
  for (const auto &[name, source] : tensorflow_files) {
    (void)name;
    has_score = has_score || source.find("final_context.score = final_case_impl->CalcScore") != std::string::npos;
  }
  EXPECT_TRUE(has_score);
  EXPECT_TRUE(has_summary_producer);

  const auto inductor_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_NE(inductor_files.find(codegen::kTilingLogHeaderIdentify), inductor_files.end());
  EXPECT_NE(inductor_files.at(codegen::kTilingLogHeaderIdentify).find("ShouldEmitFinalTiling()"), std::string::npos);
  bool has_inductor_repr_call = false;
  for (const auto &[name, source] : inductor_files) {
    (void)name;
    has_inductor_repr_call = has_inductor_repr_call || source.find("GetTilingDataRepr") != std::string::npos;
  }
  EXPECT_TRUE(has_inductor_repr_call);
}

TEST_F(TestCodegenTiling, FinalTilingContextInitializerKeepsSemanticNameOrder) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")}, af::Symbol(1));
  codegen_func_ = att::GenTilingImplAutoFuseV3;
  const auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  std::string all_tiling_code;
  for (const auto &tiling_file : tiling_files) {
    all_tiling_code += tiling_file.second;
  }
  EXPECT_NE(all_tiling_code.find("final_context.template_name = final_case_impl->GetTemplateName();"),
            std::string::npos);
  EXPECT_NE(all_tiling_code.find("final_context.semantic_name = final_case_impl->GetSemanticName();"),
            std::string::npos);
  EXPECT_NE(all_tiling_code.find("final_context.semantic_names = {final_context.semantic_name};"), std::string::npos);
  EXPECT_NE(all_tiling_code.find("EmitTemplateSemanticRecord(final_context, ambiguous_sub_case);"), std::string::npos);
}

TEST_F(TestCodegenTiling, FinalTilingObservationContextIsolatesSlotsAndReentry) {
  const std::string source = R"(
#include <string>
#include <vector>
#define CheckCachedLogLevel(level) true
#define ATT_FINAL_TILING_LOGI(op, fmt, value) (void)0
#define OP_NAME "test"
)" + att::final_tiling::GenFinalTilingLogHelpers(false) +
                             R"(
int main() {
  FinalTilingObservationContext outer_context;
  FinalTilingObservationCallGuard outer_guard(outer_context, true, 2U);
  {
    FinalTilingObservationSlotGuard slot_guard(0U);
    CommitCurrentFinalTilingSelection({true, 2U, 3U, 0U, 11U, 11, 1});
  }
  {
    FinalTilingObservationSlotGuard slot_guard(1U);
    CommitCurrentFinalTilingSelection({true, 2U, 3U, 1U, 12U, 12, 2});
  }
  FinalTilingSelectionSnapshot first{};
  FinalTilingSelectionSnapshot second{};
  if (!GetFinalTilingSelectionSnapshot(0U, first) || !GetFinalTilingSelectionSnapshot(1U, second)) { return 1; }
  if (!first.valid || first.sub_case_id != 1 || first.tiling_key != 11U) { return 2; }
  if (!second.valid || second.sub_case_id != 2 || second.tiling_key != 12U) { return 3; }

  {
    FinalTilingObservationContext inner_context;
    FinalTilingObservationCallGuard inner_guard(inner_context, true, 1U);
    FinalTilingObservationSlotGuard slot_guard(0U);
    CommitCurrentFinalTilingSelection({true, 9U, 9U, 9U, 99U, 99, 9});
    FinalTilingSelectionSnapshot inner{};
    if (!GetFinalTilingSelectionSnapshot(0U, inner) || inner.tiling_key != 99U) { return 4; }
  }
  if (!GetFinalTilingSelectionSnapshot(0U, first) || first.tiling_key != 11U) { return 5; }

  FinalTilingObservationContext disabled_context;
  FinalTilingObservationCallGuard disabled_guard(disabled_context, false, 1U);
  FinalTilingObservationSlotGuard slot_guard(0U);
  CommitCurrentFinalTilingSelection({true, 1U, 1U, 1U, 1U, 1, 1});
  FinalTilingSelectionSnapshot disabled{};
  if (GetFinalTilingSelectionSnapshot(0U, disabled) && disabled.valid) { return 6; }
  if (!disabled_context.selections.empty()) { return 7; }
  return 0;
}
)";
  ASSERT_TRUE(CompileAndRunCode(source));
}

TEST_F(TestCodegenTiling, FinalTilingSelectionUsesCallPrivateSlotSnapshots) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")}, af::Symbol(1));
  codegen_func_ = att::GenTilingImplAutoFuseV3;
  const auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  std::string all_tiling_code;
  for (const auto &tiling_file : tiling_files) {
    all_tiling_code += tiling_file.second;
  }
  EXPECT_EQ(all_tiling_code.find("g_final_selected_sub_case_tag"), std::string::npos);
  EXPECT_NE(all_tiling_code.find("FinalTilingObservationCallGuard"), std::string::npos);
  EXPECT_NE(all_tiling_code.find("FinalTilingObservationSlotGuard"), std::string::npos);
  EXPECT_NE(all_tiling_code.find("CommitCurrentFinalTilingSelection"), std::string::npos);
  EXPECT_NE(all_tiling_code.find("GetFinalTilingSubCaseId"), std::string::npos);
}

TEST_F(TestCodegenTiling, InductorMultiGroupFinalTilingUsesFullRepresentation) {
  codegen_func_ = att::GenTilingImplAutoFuseV3;
  // A zero-size axis marks the fixture as an empty-tensor scene, which falls back to the stub
  // tiling path without ATT codegen, so use a non-zero axis to exercise the ATT generator.
  const auto tiling_files = this->GenerateForInductor(this->GenTilingKeyCountResult({{1, 1}}, af::Symbol(1)));
  bool has_full_repr = false;
  bool has_key_only_repr = false;
  for (const auto &[name, source] : tiling_files) {
    (void)name;
    has_full_repr =
        has_full_repr || source.find("final_repr_kind = final_tiling_repr.empty() ? \"unavailable\" : \"full_json\"") !=
                             std::string::npos;
    has_key_only_repr = has_key_only_repr || source.find("final_repr_kind = \"key_only\"") != std::string::npos;
  }
  EXPECT_TRUE(has_full_repr);
  EXPECT_FALSE(has_key_only_repr);
}

TEST_F(TestCodegenTiling, FinalTilingKeepsCoreSelectionFreeOfObservabilityWork) {
  const auto fused_schedule_result =
      this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")}, af::Symbol(1));
  codegen_func_ = att::GenTilingImplAutoFuseV3;
  const auto tensorflow_files = this->Generate(fused_schedule_result, {}, "", "0");
  const auto core_file = std::find_if(tensorflow_files.cbegin(), tensorflow_files.cend(), [](const auto &file) {
    return file.second.find("bool GetTilingCore(") != std::string::npos;
  });
  ASSERT_NE(core_file, tensorflow_files.cend());
  const auto wrapper_file = std::find_if(tensorflow_files.cbegin(), tensorflow_files.cend(), [](const auto &file) {
    return file.second.find("bool GetTiling(") != std::string::npos &&
           file.second.find("ShouldEmitFinalTiling") != std::string::npos &&
           file.second.find("if (!final_summary.groups.empty())") != std::string::npos;
  });
  ASSERT_NE(wrapper_file, tensorflow_files.cend());
  const size_t core_begin = core_file->second.find("bool GetTilingCore(");
  ASSERT_NE(core_begin, std::string::npos);
  const size_t core_end = core_file->second.find("\nbool GetTiling(", core_begin + 1U);
  const std::string core_source = core_file->second.substr(core_begin, core_end - core_begin);
  EXPECT_EQ(core_source.find("ShouldEmitFinalTiling"), std::string::npos);
  EXPECT_EQ(core_source.find("EmitFinalTiling"), std::string::npos);
  EXPECT_EQ(core_source.find("FinalTilingSummary"), std::string::npos);

  const std::string &wrapper_source = wrapper_file->second;
  EXPECT_NE(wrapper_source.find("ShouldEmitFinalTiling"), std::string::npos);
  EXPECT_NE(wrapper_source.find("EmitFinalTilingSummary"), std::string::npos);
  EXPECT_NE(wrapper_source.find("FinalTilingObservationCallGuard"), std::string::npos);
  const size_t reset_pos = wrapper_source.find("FinalTilingObservationCallGuard final_tiling_observation_guard");
  const size_t core_call_pos = wrapper_source.find("const bool ret = GetTilingCore(");
  ASSERT_NE(reset_pos, std::string::npos);
  ASSERT_NE(core_call_pos, std::string::npos);
  EXPECT_LT(reset_pos, core_call_pos);
  EXPECT_NE(core_file->second.find("bool GetTilingCore("), std::string::npos);
  EXPECT_NE(wrapper_source.find("if (!final_summary.groups.empty())"), std::string::npos);
  EXPECT_NE(core_file->second.find("void EmitFinalTilingByCase("), std::string::npos);
}

TEST_F(TestCodegenTiling, SingleGroupFinalTilingEmitsRuntimeObservation) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")}, af::Symbol(1));
  fused_schedule_result.node_idx_to_scheduled_results[0].resize(1);
  codegen_func_ = att::GenTilingImplAutoFuseV3;
  const auto tensorflow_files = this->Generate(fused_schedule_result, {}, "", "0");

  bool has_observe_begin = false;
  bool has_observe_reset = false;
  bool has_runtime_emitter_call = false;
  for (const auto &[name, source] : tensorflow_files) {
    (void)name;
    has_observe_begin = has_observe_begin ||
                        source.find("const bool emit_final_tiling = ShouldEmitFinalTiling();") != std::string::npos;
    has_observe_reset =
        has_observe_reset ||
        source.find("FinalTilingObservationCallGuard final_tiling_observation_guard") != std::string::npos;
    has_runtime_emitter_call =
        has_runtime_emitter_call ||
        source.find("EmitFinalTilingByCase(tiling_data, tiling_data.get_block_dim(), \"runtime\"") != std::string::npos;
  }
  EXPECT_TRUE(has_observe_begin);
  EXPECT_TRUE(has_observe_reset);
  EXPECT_TRUE(has_runtime_emitter_call);
}

TEST_F(TestCodegenTiling, FinalTilingUsesSelectedCaseAndStructuredPipeInterface) {
  const auto fused_schedule_result =
      this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")}, af::Symbol(1));
  codegen_func_ = att::GenTilingImplAutoFuseV3;
  const auto tensorflow_files = this->Generate(fused_schedule_result, {}, "", "0");
  bool has_selected_case_factory = false;
  bool has_selection_case_factory = false;
  bool has_selected_sub_case_state = false;
  bool has_selected_sub_case_reset = false;
  bool has_structured_pipe_interface = false;
  bool has_axes_pipe_estimate = false;
  bool has_raw_case_pointer_pipe_call = false;
  for (const auto &[name, source] : tensorflow_files) {
    (void)name;
    has_selected_case_factory =
        has_selected_case_factory ||
        source.find("GetSelectedTilingImplPtr(static_cast<uint32_t>(final_case_id), corenum") != std::string::npos;
    has_selection_case_factory =
        has_selection_case_factory ||
        source.find("GetSelectedTilingImplPtr(static_cast<uint32_t>(snapshot.case_id), tiling_data.get_block_dim()") !=
            std::string::npos;
    has_selected_sub_case_state =
        has_selected_sub_case_state || source.find("CommitCurrentFinalTilingSelection(") != std::string::npos;
    has_selected_sub_case_reset =
        has_selected_sub_case_reset || source.find("ResetCurrentFinalTilingSelection()") != std::string::npos;
    has_structured_pipe_interface =
        has_structured_pipe_interface || source.find("CollectPipeEstimates(tiling_data") != std::string::npos;
    has_axes_pipe_estimate =
        has_axes_pipe_estimate ||
        source.find("GetTilingDataPerfStatic(PipeType::AIV_MTE2, tiling_data)") != std::string::npos;
    has_raw_case_pointer_pipe_call =
        has_raw_case_pointer_pipe_call || source.find("final_case_impl->GetAIV_") != std::string::npos;
  }
  EXPECT_TRUE(has_selected_case_factory);
  EXPECT_TRUE(has_selection_case_factory);
  EXPECT_TRUE(has_selected_sub_case_state);
  EXPECT_TRUE(has_selected_sub_case_reset);
  EXPECT_TRUE(has_structured_pipe_interface);
  EXPECT_TRUE(has_axes_pipe_estimate);
  EXPECT_FALSE(has_raw_case_pointer_pipe_call);
}

TEST_F(TestCodegenTiling, SplitHeaderGenerateForInductorShouldEmitHeaderKeysAndCppIncludes) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);

  EXPECT_NE(tiling_files.find("TilingHead"), tiling_files.end());
  EXPECT_NE(tiling_files.find("TilingStateHeader"), tiling_files.end());
  EXPECT_NE(tiling_files.find("TilingLogHeader"), tiling_files.end());
  EXPECT_NE(tiling_files.find("TilingPgoHeader"), tiling_files.end());
  EXPECT_NE(tiling_files.find("TilingSolverHeader"), tiling_files.end());
  EXPECT_NE(tiling_files.find("TilingApiHeader"), tiling_files.end());
  EXPECT_EQ(tiling_files.find("TilingBaseHeader"), tiling_files.end());
  EXPECT_EQ(tiling_files.find("TilingEntryHeader"), tiling_files.end());
  EXPECT_EQ(tiling_files.find("TilingTailHeader"), tiling_files.end());

  const auto &state_header = tiling_files.at("TilingStateHeader");
  const auto &solver_header = tiling_files.at("TilingSolverHeader");
  const auto &api_header = tiling_files.at("TilingApiHeader");
  EXPECT_EQ(state_header.find("#include \"autofuse_tiling_func_"), std::string::npos);
  EXPECT_EQ(solver_header.find("#include \"autofuse_tiling_func_"), std::string::npos);
  EXPECT_EQ(api_header.find("get_g_basen_basem_align"), std::string::npos);
  EXPECT_EQ(api_header.find("set_g_basen_basem_align"), std::string::npos);

  ASSERT_NE(tiling_files.find(codegen::kTilingDefAndConstIdentify), tiling_files.end());
  const auto &entry = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_NE(entry.find("#include \"autofuse_tiling_data.h\""), std::string::npos);
  EXPECT_NE(entry.find("#include \"autofuse_tiling_func_log.h\""), std::string::npos);
  EXPECT_NE(entry.find("#include \"autofuse_tiling_func_pgo.h\""), std::string::npos);
  EXPECT_NE(entry.find("#include \"autofuse_tiling_func_api.h\""), std::string::npos);
  EXPECT_EQ(entry.find("#include \"autofuse_tiling_func_solver.h\""), std::string::npos);
  EXPECT_EQ(entry.find("#include \"autofuse_tiling_func_common.h\""), std::string::npos);
  EXPECT_EQ(entry.find("#include \"exe_graph/runtime/tiling_context.h\""), std::string::npos);
  EXPECT_EQ(entry.find("exe_graph/runtime/infer_shape_context.h"), std::string::npos);
  EXPECT_EQ(entry.find("acl/acl.h"), std::string::npos);
  EXPECT_NE(entry.find("#include \"autofuse_tiling_func_api.h\"\n\nusing namespace optiling;"), std::string::npos);
  EXPECT_EQ(entry.find("IsEqual(kept.modeled_perf"), std::string::npos);
  ExpectSystemHeaders(entry,
                      {"algorithm", "cfloat", "cmath", "cstddef", "cstdint", "map", "ostream", "sstream", "string",
                       "unordered_map", "utility", "vector"},
                      {"array", "cstdlib", "fstream", "functional", "memory", "securec.h", "unordered_set"});

  const auto solver_iter = tiling_files.find("solver_func");
  if (solver_iter != tiling_files.end()) {
    const auto &solver = solver_iter->second;
    EXPECT_NE(solver.find("#include \"autofuse_tiling_func_log.h\""), std::string::npos);
    EXPECT_NE(solver.find("#include \"autofuse_tiling_func_solver.h\""), std::string::npos);
    EXPECT_EQ(solver.find("#include \"autofuse_tiling_func_common.h\""), std::string::npos);
  }
}

TEST_F(TestCodegenTiling, SplitHeaderApiTilingSourceShouldIncludeApiHeaders) {
  ge::PlatformContext::GetInstance().SetPlatform("2201");
  codegen_func_ = att::GenTilingImplAutoFuseV3;
  auto fused_schedule_result = GenSingleGroupFusedScheduleResultWithApiTilingField();
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);

  bool found_api_source = false;
  for (const auto &[key, source] : tiling_files) {
    if (key == "TilingHead" || source.find("GetConfusionTranspose") == std::string::npos) {
      continue;
    }
    found_api_source = true;
    EXPECT_NE(source.find("#include \"graph/tensor.h\""), std::string::npos);
    EXPECT_NE(source.find("#ifndef AUTOFUSE_CONFUSION_TRANSPOSE_TILING_DEFS"), std::string::npos);
    EXPECT_NE(source.find("const uint32_t ONE_BLK_SIZE = 32;"), std::string::npos);
    EXPECT_NE(source.find("const uint32_t BLOCK_CUBE = 16;"), std::string::npos);
  }
  EXPECT_TRUE(found_api_source);
}

TEST_F(TestCodegenTiling, SplitHeaderGenerateForTfShouldGuardRuntimeHeadersForCceKtTest) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")}, af::Symbol(1));
  const std::map<std::string, std::string> shape_info;
  auto tiling_files = this->Generate(fused_schedule_result, shape_info, ".", "10");

  ASSERT_NE(tiling_files.find(codegen::kTilingDefAndConstIdentify), tiling_files.end());
  const auto &entry = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  const std::string guarded_headers =
      "#ifndef __CCE_KT_TEST__\n"
      "#include \"exe_graph/runtime/tiling_context.h\"\n"
      "#include \"tiling/platform/platform_ascendc.h\"\n"
      "#endif\n";
  EXPECT_NE(entry.find(guarded_headers), std::string::npos);
  EXPECT_EQ(entry.find("#include \"platform_ascendc.h\""), std::string::npos);

  const auto tiling_call = entry.find("auto ret = AutofuseTilingWithConfig");
  const auto failure_check = entry.find("if (ret != 0) {", tiling_call);
  const auto set_block_dim = entry.find("context->SetBlockDim(block_dim);", tiling_call);
  ASSERT_NE(tiling_call, std::string::npos);
  ASSERT_NE(failure_check, std::string::npos);
  ASSERT_NE(set_block_dim, std::string::npos);
  EXPECT_LT(failure_check, set_block_dim);
}

TEST_F(TestCodegenTiling, SplitHeaderGenerateForPgoShouldIncludeDirectEntryDependencies) {
  enable_autofuse_pgo_ = true;
  codegen_func_ = att::GenTilingImplAutoFuseV3;
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")}, af::Symbol(1));
  const std::map<std::string, std::string> shape_info;
  auto tiling_files = this->Generate(fused_schedule_result, shape_info, ".", "10");

  ASSERT_NE(tiling_files.find(codegen::kTilingDefAndConstIdentify), tiling_files.end());
  const auto &entry = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_NE(entry.find("#include <fstream>"), std::string::npos);
  EXPECT_NE(entry.find("#include <securec.h>"), std::string::npos);
  EXPECT_NE(entry.find("#include <unordered_set>"), std::string::npos);
  EXPECT_NE(entry.find("#include \"exe_graph/runtime/tiling_context.h\""), std::string::npos);
  EXPECT_NE(entry.find("#include \"autofuse_tiling_func_pgo.h\""), std::string::npos);
  EXPECT_NE(entry.find("#include \"autofuse_tiling_func_solver.h\""), std::string::npos);

  bool found_pgo_core_budget_check = false;
  bool found_pgo_config_validation = false;
  bool found_pgo_tiling_isolation = false;
  bool found_pgo_final_tiling = false;
  for (const auto &[name, source] : tiling_files) {
    (void)name;
    if (source.find("Loaded PGO block_dim %u is outside core budget [1, %u]") != std::string::npos) {
      found_pgo_core_budget_check = true;
    }
    if (source.find("tiling_i32.size() != expect_num") != std::string::npos &&
        source.find("memcpy_s(&tiling_data, sizeof(tiling_data), tiling_i32.data(), sizeof(tiling_data)) != EOK") !=
            std::string::npos) {
      found_pgo_config_validation = true;
    }
    if (source.find("auto pgo_tiling = *tiling;") != std::string::npos &&
        source.find("PGOGetTilingKey(config_file, pgo_tiling)") != std::string::npos &&
        source.find("*tiling = pgo_tiling;") != std::string::npos) {
      found_pgo_tiling_isolation = true;
    }
    found_pgo_final_tiling = found_pgo_final_tiling ||
                             source.find("EmitFinalTilingByCase(*tiling") != std::string::npos ||
                             source.find("EmitFinalTilingForPgo(*tiling") != std::string::npos;
  }
  EXPECT_TRUE(found_pgo_core_budget_check);
  EXPECT_TRUE(found_pgo_config_validation);
  EXPECT_TRUE(found_pgo_tiling_isolation);
  EXPECT_TRUE(found_pgo_final_tiling);
}

TEST_F(TestCodegenTiling, SplitHeaderGenerateForInductorPgoShouldIncludeDirectEntryDependencies) {
  enable_autofuse_pgo_ = true;
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol(64), af::Symbol(128)});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);

  ASSERT_NE(tiling_files.find(codegen::kTilingDefAndConstIdentify), tiling_files.end());
  const auto &entry = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_NE(entry.find("optiling::IsEqual"), std::string::npos);
  EXPECT_NE(entry.find("#include <fstream>"), std::string::npos);
  EXPECT_NE(entry.find("#include <unordered_set>"), std::string::npos);
  EXPECT_NE(entry.find("#include <securec.h>"), std::string::npos);
  EXPECT_NE(entry.find("#include \"autofuse_tiling_func_solver.h\""), std::string::npos);
}

TEST_F(TestCodegenTiling, SplitHeaderFallbackShouldEmitUsableApiAndPgoHeaders) {
  codegen_func_ = nullptr;
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);

  ASSERT_NE(tiling_files.find(codegen::kTilingApiHeaderIdentify), tiling_files.end());
  ASSERT_NE(tiling_files.find(codegen::kTilingPgoHeaderIdentify), tiling_files.end());
  const auto &api_header = tiling_files.at(codegen::kTilingApiHeaderIdentify);
  const auto &pgo_header = tiling_files.at(codegen::kTilingPgoHeaderIdentify);
  EXPECT_NE(api_header.find("inline bool GetTiling("), std::string::npos);
  EXPECT_NE(api_header.find("inline bool PGOSearchTilingKey("), std::string::npos);
  EXPECT_NE(api_header.find("inline bool PGOByCoreNumSearchTilingKey("), std::string::npos);
  EXPECT_NE(api_header.find("struct FinalTilingGroupSelection;"), std::string::npos);
  EXPECT_NE(pgo_header.find("class PgoConfig"), std::string::npos);
  EXPECT_NE(pgo_header.find("struct SearchConfig"), std::string::npos);
  EXPECT_EQ(api_header.find("#include \"autofuse_tiling_func_"), std::string::npos);
  EXPECT_EQ(pgo_header.find("#include \"autofuse_tiling_func_"), std::string::npos);
  EXPECT_TRUE(CompileCode(api_header));
  EXPECT_TRUE(CompileCode(pgo_header));
}

TEST_F(TestCodegenTiling, SplitHeaderFallbackSolverShouldCompileWorkspaceHelpers) {
  codegen_func_ = nullptr;
  auto graph = ascir::ShareGraph::TailBrcTailReduceFusedGraph(3);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused_schedule_result;

  ASSERT_EQ(optimizer.Optimize(graph, fused_schedule_result), af::SUCCESS);
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_NE(tiling_files.find(codegen::kTilingDefAndConstIdentify), tiling_files.end());
  ASSERT_NE(tiling_files.find(codegen::kTilingSolverHeaderIdentify), tiling_files.end());
  const auto &entry = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  const auto &solver_header = tiling_files.at(codegen::kTilingSolverHeaderIdentify);
  EXPECT_NE(entry.find("#include \"autofuse_tiling_func_solver.h\""), std::string::npos);
  EXPECT_TRUE(CompileCode(solver_header + R"(
using namespace optiling;
double WorkspaceSize(double z0z1t_size, double z2t_size, double z2Tt_size) {
  return Max(0, 4 * Max(Max(32, z0z1t_size), 32 * Ceiling(Ceiling(7 / z2t_size) / z2Tt_size)));
}
)"));
}

TEST_F(TestCodegenTiling, GenerateForInductorTopnAbiShouldNotEmitOutputConfigsMetadataLogic) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_EQ(tiling_impl.find("std::map<std::string, std::string> solution_config;"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("configs.push_back(solution_config);"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("solution_config[\"canonical_repr\"]"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("solution_config[\"topn_status\"]"), std::string::npos);
}

TEST_F(TestCodegenTiling, GenerateForInductorNonCubeShouldNotDuplicateImplGraphs) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  ASSERT_FALSE(ascgen_utils::IsCubeFusedScheduled(fused_schedule_result));

  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);

  auto func_pos = tiling_impl.find("extern \"C\" uint64_t GetTilingKeyCount()");
  ASSERT_NE(func_pos, std::string::npos);
  auto func_end_pos = tiling_impl.find("}\n", func_pos);
  ASSERT_NE(func_end_pos, std::string::npos);
  auto return_two_pos = tiling_impl.find("  return 2;", func_pos);
  auto return_four_pos = tiling_impl.find("  return 4;", func_pos);

  EXPECT_LT(return_two_pos, func_end_pos);
  EXPECT_TRUE(return_four_pos == std::string::npos || return_four_pos > func_end_pos);
}

TEST_F(TestCodegenTiling, GenerateForInductorCvFusionShouldEmitCvTilingAndCubeWrapper) {
  auto graph = ascir::ShareGraph::LoadMatmulElewiseBrcFusedGraph();
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused_schedule_result;
  ASSERT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);
  ASSERT_TRUE(ascgen_utils::IsCubeFusedScheduled(fused_schedule_result));

  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  ASSERT_TRUE(tiling_files.find(codegen::kTilingApiHeaderIdentify) != tiling_files.end());
  ASSERT_TRUE(tiling_files.find(codegen::kTilingHeadIdentify) != tiling_files.end());
  ASSERT_TRUE(tiling_files.find(codegen::kCubeKernelTilingWrapperHpp) != tiling_files.end());
  ASSERT_TRUE(tiling_files.find(codegen::kCubeKernelTilingWrapperCpp) != tiling_files.end());
  EXPECT_TRUE(tiling_files.find("TilingDataLog") == tiling_files.end());

  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  const auto &api_header = tiling_files.at(codegen::kTilingApiHeaderIdentify);
  EXPECT_NE(api_header.find("int32_t get_g_basen_basem_align();"), std::string::npos);
  EXPECT_NE(api_header.find("void set_g_basen_basem_align(int32_t value);"), std::string::npos);
  const std::string base_align_defs = R"(
static int32_t g_basen_basem_align = 0;

int32_t get_g_basen_basem_align() {
  return g_basen_basem_align;
}

void set_g_basen_basem_align(int32_t value) {
  g_basen_basem_align = value;
}
)";
  const auto base_align_defs_pos = tiling_impl.find(base_align_defs);
  ASSERT_NE(base_align_defs_pos, std::string::npos);
  EXPECT_EQ(tiling_impl.find(base_align_defs, base_align_defs_pos + base_align_defs.size()), std::string::npos);
  EXPECT_NE(tiling_impl.find("CVAutofuseTilingData"), std::string::npos);
  EXPECT_NE(tiling_impl.find("CVTilingData"), std::string::npos);
  EXPECT_NE(tiling_impl.find("CallCubeTiling"), std::string::npos);
  EXPECT_NE(tiling_impl.find("AutofuseTiling("), std::string::npos);
  EXPECT_NE(tiling_impl.find("GenConstTilingData"), std::string::npos);
  EXPECT_NE(tiling_impl.find("autofuse_has_bias"), std::string::npos);
  EXPECT_NE(tiling_impl.find("autofuse_has_offset_w"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("GenerateTopnSolutions"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("GetModeledPerfForTesting"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("AscirCompileAndLaunch"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("GenAscirTilingAndLaunchFunc"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("#include \"exe_graph/runtime/tiling_context.h\""), std::string::npos);
  EXPECT_EQ(tiling_impl.find("#include \"autofuse_cube_tiling_data.h\""), std::string::npos);
  EXPECT_NE(tiling_impl.find("#include \"cube_kernel_tiling_wrapper.h\""), std::string::npos);
  ExpectSystemHeaders(
      tiling_impl,
      {"algorithm", "cfloat", "cstddef", "cstdint", "cstring", "iomanip", "ostream", "sstream", "string", "vector"},
      {"array", "cmath", "cstdlib", "functional", "map", "memory", "unordered_map", "utility"});
}

TEST_F(TestCodegenTiling, CubeWrapperShouldPreserveTilingDataBytes) {
  const auto &wrapper_hpp = kCubeKernelTilingWrapperHppValue;
  const auto &wrapper_cpp = kCubeKernelTilingWrapperCppValue;
  const auto matmul_tiling_header_pos = wrapper_hpp.find("#include \"arch35/mat_mul_tiling_data.h\"");
  const auto autofuse_namespace_pos = wrapper_hpp.find("namespace autofuse {");
  EXPECT_NE(wrapper_hpp.find("std::vector<uint8_t> tiling_data;"), std::string::npos);
  ASSERT_NE(matmul_tiling_header_pos, std::string::npos);
  ASSERT_NE(autofuse_namespace_pos, std::string::npos);
  EXPECT_LT(matmul_tiling_header_pos, autofuse_namespace_pos);
  EXPECT_NE(wrapper_cpp.find("result.tiling_data.assign"), std::string::npos);
  EXPECT_NE(wrapper_cpp.find("raw_tiling_data->GetDataSize()"), std::string::npos);
}

TEST_F(TestCodegenTiling, CubeWrapperShouldSupportBiasAndOffsetInputs) {
  const auto &wrapper_cpp = kCubeKernelTilingWrapperCppValue;
  EXPECT_NE(wrapper_cpp.find("MakeMatMulInputInstanceNum(request.matmul_attrs.has_bias"), std::string::npos);
  EXPECT_NE(wrapper_cpp.find("key.input2_shape = GetRuntimeShape(request.inputs[2]);"), std::string::npos);
  EXPECT_NE(wrapper_cpp.find("key.input3_shape = GetRuntimeShape(request.inputs[3]);"), std::string::npos);
  EXPECT_NE(wrapper_cpp.find("key.input_num = request.inputs.size();"), std::string::npos);
  EXPECT_NE(wrapper_cpp.find("inputs->size() >= 2 && inputs->size() <= 4"), std::string::npos);
  EXPECT_NE(wrapper_cpp.find("has_bias = AttrAsBool(attr);"), std::string::npos);
  EXPECT_NE(wrapper_cpp.find("has_offset_w = AttrAsBool(attr);"), std::string::npos);
  EXPECT_NE(wrapper_cpp.find("if (attrs.has_offset_w && input_index < inputs.size())"), std::string::npos);
  EXPECT_NE(wrapper_cpp.find("BuildMatMulInputSlots(request.inputs, request.matmul_attrs)"), std::string::npos);
  EXPECT_NE(wrapper_cpp.find("if (input == nullptr)"), std::string::npos);
  EXPECT_NE(wrapper_cpp.find("input2_dtype = inputs[2U].dtype;"), std::string::npos);
  EXPECT_NE(wrapper_cpp.find("input3_format = inputs[3U].format;"), std::string::npos);
  EXPECT_NE(wrapper_cpp.find("input_tensors_storage.reserve(input_slots.size());"), std::string::npos);
  EXPECT_NE(wrapper_cpp.find("for (const auto *input : input_slots)"), std::string::npos);
  EXPECT_NE(wrapper_cpp.find("DtypeToGeDataType(input.dtype) == ge::DT_UNDEFINED"), std::string::npos);
  EXPECT_NE(wrapper_cpp.find("ge::Format input_format = FormatToGeFormat(input->format);"), std::string::npos);
  EXPECT_NE(wrapper_cpp.find("size_t input_desc_index = 2U;"), std::string::npos);
  EXPECT_NE(wrapper_cpp.find("if (input_slots[2U] != nullptr)"), std::string::npos);
  EXPECT_NE(wrapper_cpp.find("if (input_slots[3U] != nullptr)"), std::string::npos);
  EXPECT_EQ(wrapper_cpp.find(".InputTensorDesc(2, input_slots[2U]"), std::string::npos);
  EXPECT_EQ(wrapper_cpp.find(".InputTensorDesc(3,"), std::string::npos);
}

TEST_F(TestCodegenTiling, MultiGroupInductorShouldContainTopnMainOutputAbi) {
  auto fused_schedule_result = GenMultiGroupFusedScheduleResult();
  auto res = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(res.find(codegen::kTilingDefAndConstIdentify) != res.end());
  const auto &tiling_impl = res.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_NE(tiling_impl.find("extern \"C\" int64_t GenerateTopnSolutions("), std::string::npos);
  EXPECT_NE(tiling_impl.find("const std::vector<std::map<std::string, std::string>> &input_configs"),
            std::string::npos);
  EXPECT_NE(tiling_impl.find("std::vector<AutofuseTilingData> &tiling_datas"), std::string::npos);
  EXPECT_NE(tiling_impl.find("std::vector<int64_t> &workspaces"), std::string::npos);
  EXPECT_NE(tiling_impl.find("std::vector<int64_t> &block_dims"), std::string::npos);
}

TEST_F(TestCodegenTiling, MultiGroupTopnShouldSetShapeDimOnGroupTilingData) {
  auto fused_schedule_result = GenMultiGroupFusedScheduleResultWithSizeVar("ks0");
  auto res = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(res.find(codegen::kTilingDefAndConstIdentify) != res.end());
  const auto &tiling_impl = res.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_EQ(tiling_impl.find("search_tiling.set_ks0("), std::string::npos);
  EXPECT_NE(tiling_impl.find("search_tiling.graph0_result0_g0_tiling_data.set_ks0(ks0);"), std::string::npos);
  EXPECT_NE(tiling_impl.find("search_tiling.graph0_result0_g1_tiling_data.set_ks0(ks0);"), std::string::npos);
}

TEST_F(TestCodegenTiling, MultiGroupInductorShouldContainReprAbi) {
  auto fused_schedule_result = GenMultiGroupFusedScheduleResult();
  auto res = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(res.find(codegen::kTilingDefAndConstIdentify) != res.end());
  const auto &tiling_impl = res.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_NE(tiling_impl.find("std::string GetTilingDataRepr(const AutofuseTilingData *tiling_data)"),
            std::string::npos);
}

TEST_F(TestCodegenTiling, TestMatmulElemwiseDynamicShapeFuse) {
  af::AscGraph graph("matmul_elemwise_pro");
  CreateMatmulElemwiseDynamicGraph(graph);

  af::AscGraph mm_graph("mutmul");
  CreateMatmulGraph(mm_graph, true);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused_schedule_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);

  ascir::ScheduleGroup schedule_group2;
  schedule_group2.impl_graphs.push_back(mm_graph);
  fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups.push_back(schedule_group2);
  fused_schedule_result.node_idx_to_scheduled_results[0][0].cube_type = ascir::CubeTemplateType::kUBFuse;

  std::map<std::string, std::string> shape_info;
  shape_info["s0"] = "64";
  shape_info["s1"] = "64";
  auto res = this->Generate(fused_schedule_result, shape_info, "", "0");

  std::fstream tiling_func("Mutmul_fuse_tiling_func.cpp", std::ios::out);
  tiling_func << res["tiling_def_and_tiling_const"];

  auto pos = res["tiling_def_and_tiling_const"].find("TilingResult result = wrapper.DoMatMulTiling(");
  ASSERT_NE(pos, std::string::npos);
}

TEST_F(TestCodegenTiling, CubeTilingUsesAclrtSocName) {
  const auto tiling_code = GenerateMatmulTilingForSoc(std::make_shared<RuntimeStubWithFullSocName>(), "3510");

  EXPECT_NE(tiling_code.find("compile_info.soc_version = \"Ascend910_9591\";"), std::string::npos);
}

TEST_F(TestCodegenTiling, CubeTilingFallsBackToPlatformNpuArchWhenAclrtSocNameIsNull) {
  const auto tiling_code = GenerateMatmulTilingForSoc(std::make_shared<RuntimeStubWithNullSocName>(), "3510");

  EXPECT_NE(tiling_code.find("compile_info.soc_version = \"3510\";"), std::string::npos);
}

// ==================== Conv2D 相关辅助函数 ====================

void CreateConv2dGraph(af::AscGraph &graph, bool is_dynamic = false) {
  af::Expression n, c, h, w;
  if (is_dynamic) {
    n = graph.CreateSizeVar("n");
    c = graph.CreateSizeVar("c");
    h = graph.CreateSizeVar("h");
    w = graph.CreateSizeVar("w");
  } else {
    n = graph.CreateSizeVar(1);
    c = graph.CreateSizeVar(64);
    h = graph.CreateSizeVar(56);
    w = graph.CreateSizeVar(56);
  }

  auto z_n = graph.CreateAxis("z_n", n);
  auto z_c = graph.CreateAxis("z_c", c);
  auto z_h = graph.CreateAxis("z_h", h);
  auto z_w = graph.CreateAxis("z_w", w);

  af::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0.y.strides = {c * h * w, h * w, w, af::ops::One};
  *data0.y.repeats = {n, c, h, w};
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  load0.x = data0.y;
  *load0.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {c * h * w, h * w, w, af::ops::One};
  *load0.y.repeats = {n, c, h, w};

  af::ascir_op::Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  *data1.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data1.y.repeats = {af::ops::One, af::ops::One, af::ops::One, af::ops::One};
  *data1.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero, af::ops::Zero};
  data1.ir_attr.SetIndex(1);

  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  *load1.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero, af::ops::Zero};
  *load1.y.repeats = {af::ops::One, af::ops::One, af::ops::One, af::ops::One};

  af::ascir_op::Conv2D conv2d("conv2d");
  conv2d.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  conv2d.x = load0.y;
  conv2d.filter = load1.y;
  conv2d.y.dtype = ge::DT_FLOAT;
  *conv2d.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  *conv2d.y.repeats = {n, c, h, w};
  *conv2d.y.strides = {c * h * w, h * w, w, af::ops::One};
  conv2d.attr.api.compute_type = af::ComputeType::kComputeCube;
  conv2d.ir_attr.SetStrides({1, 1});
  conv2d.ir_attr.SetPads({1, 1, 1, 1});
  conv2d.ir_attr.SetDilations({1, 1});
  conv2d.ir_attr.SetGroups(1);
  conv2d.ir_attr.SetData_format("NCHW");
  conv2d.ir_attr.SetOffset_x(0);
  conv2d.ir_attr.SetEnable_hf32(false);

  af::ascir_op::Store store_op("store");
  store_op.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  store_op.x = conv2d.y;
  *store_op.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {c * h * w, h * w, w, af::ops::One};
  *store_op.y.repeats = {n, c, h, w};

  af::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
}

// ==================== Conv2D + Elemwise 融合测试用例 ====================

TEST_F(TestCodegenTiling, TestConv2dElemwiseFuse) {
  af::AscGraph graph("conv2d_elemwise_pro");
  CreateElemwiseGraphWithAbsAndAddStatic(graph);

  af::AscGraph conv2d_graph("conv2d");
  CreateConv2dGraph(conv2d_graph, false);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused_schedule_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);

  ascir::ScheduleGroup schedule_group2;
  schedule_group2.impl_graphs.push_back(conv2d_graph);
  fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups.push_back(schedule_group2);
  fused_schedule_result.node_idx_to_scheduled_results[0][0].cube_type = ascir::CubeTemplateType::kUBFuse;

  const std::map<std::string, std::string> shape_info;
  auto res = this->Generate(fused_schedule_result, shape_info, "", "0");

  std::fstream tiling_func("Conv2d_fuse_tiling_func.cpp", std::ios::out);
  tiling_func << res["tiling_def_and_tiling_const"];

  VerifyConv2dElemwiseTiling(res);
}

// ==================== Conv2D + Elemwise 动态 Shape 融合测试用例 ====================

TEST_F(TestCodegenTiling, TestConv2dElemwiseDynamicShapeFuse) {
  af::AscGraph graph("conv2d_elemwise_dynamic_pro");
  CreateElemwiseGraphWithReluDynamic(graph);

  af::AscGraph conv2d_graph("conv2d_dynamic");
  CreateConv2dGraph(conv2d_graph, true);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused_schedule_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);

  ascir::ScheduleGroup schedule_group2;
  schedule_group2.impl_graphs.push_back(conv2d_graph);
  fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups.push_back(schedule_group2);
  fused_schedule_result.node_idx_to_scheduled_results[0][0].cube_type = ascir::CubeTemplateType::kUBFuse;

  std::map<std::string, std::string> shape_info;
  shape_info["n"] = "1";
  shape_info["c"] = "64";
  shape_info["h"] = "56";
  shape_info["w"] = "56";
  auto res = this->Generate(fused_schedule_result, shape_info, "", "0");

  std::fstream tiling_func("Conv2d_dynamic_fuse_tiling_func.cpp", std::ios::out);
  tiling_func << res["tiling_def_and_tiling_const"];

  VerifyDynamicShapeTiling(res);

  auto n_val_pos = res["tiling_def_and_tiling_const"].find("auto n = 1;");
  ASSERT_NE(n_val_pos, std::string::npos);

  auto c_val_pos = res["tiling_def_and_tiling_const"].find("auto c = 64;");
  ASSERT_NE(c_val_pos, std::string::npos);

  auto h_val_pos = res["tiling_def_and_tiling_const"].find("auto h = 56;");
  ASSERT_NE(h_val_pos, std::string::npos);

  auto w_val_pos = res["tiling_def_and_tiling_const"].find("auto w = 56;");
  ASSERT_NE(w_val_pos, std::string::npos);

  auto dfx_pos = res["tiling_def_and_tiling_const"].find("extern \"C\" ge::graphStatus DfxInputSymbolInfo");
  ASSERT_NE(dfx_pos, std::string::npos);
}

// ==================== Conv2DBias 辅助函数 ====================

void CreateConv2dBiasGraph(af::AscGraph &graph, bool is_dynamic = false) {
  af::Expression n, c, h, w;
  if (is_dynamic) {
    n = graph.CreateSizeVar("n");
    c = graph.CreateSizeVar("c");
    h = graph.CreateSizeVar("h");
    w = graph.CreateSizeVar("w");
  } else {
    n = graph.CreateSizeVar(1);
    c = graph.CreateSizeVar(64);
    h = graph.CreateSizeVar(56);
    w = graph.CreateSizeVar(56);
  }

  auto z_n = graph.CreateAxis("z_n", n);
  auto z_c = graph.CreateAxis("z_c", c);
  auto z_h = graph.CreateAxis("z_h", h);
  auto z_w = graph.CreateAxis("z_w", w);

  af::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0.y.strides = {c * h * w, h * w, w, af::ops::One};
  *data0.y.repeats = {n, c, h, w};
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  load0.x = data0.y;
  *load0.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {c * h * w, h * w, w, af::ops::One};
  *load0.y.repeats = {n, c, h, w};

  af::ascir_op::Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  *data1.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data1.y.repeats = {af::ops::One, af::ops::One, af::ops::One, af::ops::One};
  *data1.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero, af::ops::Zero};
  data1.ir_attr.SetIndex(1);

  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  *load1.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero, af::ops::Zero};
  *load1.y.repeats = {af::ops::One, af::ops::One, af::ops::One, af::ops::One};

  af::ascir_op::Data data2("data2", graph);
  data2.y.dtype = ge::DT_FLOAT;
  data2.attr.sched.axis = {z_c.id};
  *data2.y.axis = {z_c.id};
  data2.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data2.y.repeats = {c};
  *data2.y.strides = {af::ops::One};
  data2.ir_attr.SetIndex(2);

  af::ascir_op::Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z_c.id};
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.axis = {z_c.id};
  *load2.y.strides = {af::ops::One};
  *load2.y.repeats = {c};

  af::ascir_op::Conv2DBias conv2d_bias("conv2d_bias");
  conv2d_bias.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  conv2d_bias.x = load0.y;
  conv2d_bias.filter = load1.y;
  conv2d_bias.bias = load2.y;
  conv2d_bias.y.dtype = ge::DT_FLOAT;
  *conv2d_bias.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  *conv2d_bias.y.repeats = {n, c, h, w};
  *conv2d_bias.y.strides = {c * h * w, h * w, w, af::ops::One};
  conv2d_bias.attr.api.compute_type = af::ComputeType::kComputeCube;
  conv2d_bias.ir_attr.SetStrides({1, 1});
  conv2d_bias.ir_attr.SetPads({1, 1, 1, 1});
  conv2d_bias.ir_attr.SetDilations({1, 1});
  conv2d_bias.ir_attr.SetGroups(1);
  conv2d_bias.ir_attr.SetData_format("NCHW");
  conv2d_bias.ir_attr.SetOffset_x(0);
  conv2d_bias.ir_attr.SetEnable_hf32(false);

  af::ascir_op::Store store_op("store");
  store_op.attr.sched.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  store_op.x = conv2d_bias.y;
  *store_op.y.axis = {z_n.id, z_c.id, z_h.id, z_w.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {c * h * w, h * w, w, af::ops::One};
  *store_op.y.repeats = {n, c, h, w};

  af::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
}

// ==================== Conv2DBias + Elemwise 静态 Shape 融合测试 ====================

TEST_F(TestCodegenTiling, TestConv2dBiasElemwiseFuse) {
  af::AscGraph graph("conv2d_bias_elemwise_pro");
  CreateElemwiseGraphWithRelu(graph);

  af::AscGraph conv2d_bias_graph("conv2d_bias");
  CreateConv2dBiasGraph(conv2d_bias_graph, false);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused_schedule_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);

  ascir::ScheduleGroup schedule_group2;
  schedule_group2.impl_graphs.push_back(conv2d_bias_graph);
  fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups.push_back(schedule_group2);
  fused_schedule_result.node_idx_to_scheduled_results[0][0].cube_type = ascir::CubeTemplateType::kUBFuse;

  const std::map<std::string, std::string> shape_info;
  auto res = this->Generate(fused_schedule_result, shape_info, "", "0");

  std::fstream tiling_func("Conv2d_bias_fuse_tiling_func.cpp", std::ios::out);
  tiling_func << res["tiling_def_and_tiling_const"];

  VerifyConv2DBiasElemwiseTiling(res);
}

// ==================== Conv2DBias + Elemwise 动态 Shape 融合测试 ====================

TEST_F(TestCodegenTiling, TestConv2dBiasElemwiseDynamicShapeFuse) {
  af::AscGraph graph("conv2d_bias_elemwise_dynamic_pro");
  CreateElemwiseGraphWithMulDynamic(graph);

  af::AscGraph conv2d_bias_graph("conv2d_bias_dynamic");
  CreateConv2dBiasGraph(conv2d_bias_graph, true);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused_schedule_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);

  ascir::ScheduleGroup schedule_group2;
  schedule_group2.impl_graphs.push_back(conv2d_bias_graph);
  fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups.push_back(schedule_group2);
  fused_schedule_result.node_idx_to_scheduled_results[0][0].cube_type = ascir::CubeTemplateType::kUBFuse;

  std::map<std::string, std::string> shape_info;
  shape_info["n"] = "1";
  shape_info["c"] = "64";
  shape_info["h"] = "56";
  shape_info["w"] = "56";
  auto res = this->Generate(fused_schedule_result, shape_info, "", "0");

  std::fstream tiling_func("Conv2d_bias_dynamic_fuse_tiling_func.cpp", std::ios::out);
  tiling_func << res["tiling_def_and_tiling_const"];

  VerifyDynamicShapeTiling(res);

  auto n_val_pos = res["tiling_def_and_tiling_const"].find("auto n = 1;");
  ASSERT_NE(n_val_pos, std::string::npos);

  auto c_val_pos = res["tiling_def_and_tiling_const"].find("auto c = 64;");
  ASSERT_NE(c_val_pos, std::string::npos);

  auto h_val_pos = res["tiling_def_and_tiling_const"].find("auto h = 56;");
  ASSERT_NE(h_val_pos, std::string::npos);

  auto w_val_pos = res["tiling_def_and_tiling_const"].find("auto w = 56;");
  ASSERT_NE(w_val_pos, std::string::npos);
}

// ==================== BatchMatmul Dynamic Shape 融合测试 ====================

void CreateBatchMatmulDynamicGraph(af::AscGraph &graph) {
  auto batch = graph.CreateSizeVar("batch");
  auto m = graph.CreateSizeVar("m");
  auto n = graph.CreateSizeVar("n");
  auto k = graph.CreateSizeVar("k");

  auto z_batch = graph.CreateAxis("z_batch", batch);
  auto z_m = graph.CreateAxis("z_m", m);
  auto z_n = graph.CreateAxis("z_n", n);
  auto z_k = graph.CreateAxis("z_k", k);

  af::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z_batch.id, z_m.id, z_k.id};
  data0.y.dtype = ge::DT_FLOAT16;
  *data0.y.axis = {z_batch.id, z_m.id, z_k.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0.y.strides = {m * k, k, af::ops::One};
  *data0.y.repeats = {batch, m, k};
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z_batch.id, z_m.id, z_k.id};
  load0.x = data0.y;
  *load0.y.axis = {z_batch.id, z_m.id, z_k.id};
  load0.y.dtype = ge::DT_FLOAT16;
  *load0.y.strides = {m * k, k, af::ops::One};
  *load0.y.repeats = {batch, m, k};

  af::ascir_op::Data data1("data1", graph);
  data1.y.dtype = ge::DT_FLOAT16;
  data1.attr.sched.axis = {z_batch.id, z_k.id, z_n.id};
  *data1.y.axis = {z_batch.id, z_k.id, z_n.id};
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data1.y.repeats = {batch, k, n};
  *data1.y.strides = {k * n, n, af::ops::One};
  data1.ir_attr.SetIndex(1);

  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z_batch.id, z_k.id, z_n.id};
  load1.y.dtype = ge::DT_FLOAT16;
  *load1.y.axis = {z_batch.id, z_k.id, z_n.id};
  *load1.y.strides = {k * n, n, af::ops::One};
  *load1.y.repeats = {batch, k, n};

  af::ascir_op::BatchMatMul batch_matmul("batch_matmul");
  batch_matmul.attr.sched.axis = {z_batch.id, z_m.id, z_n.id};
  batch_matmul.x1 = load0.y;
  batch_matmul.x2 = load1.y;
  batch_matmul.y.dtype = ge::DT_FLOAT;
  *batch_matmul.y.axis = {z_batch.id, z_m.id, z_n.id};
  *batch_matmul.y.repeats = {batch, m, n};
  *batch_matmul.y.strides = {m * n, n, af::ops::One};
  batch_matmul.attr.api.compute_type = af::ComputeType::kComputeCube;
  batch_matmul.ir_attr.SetAdj_x1(0);
  batch_matmul.ir_attr.SetAdj_x2(0);
  batch_matmul.ir_attr.SetHas_relu(1);
  batch_matmul.ir_attr.SetEnable_hf32(true);
  batch_matmul.ir_attr.SetOffset_x(6);

  af::ascir_op::Store store_op("store");
  store_op.attr.sched.axis = {z_batch.id, z_m.id, z_n.id};
  store_op.x = batch_matmul.y;
  *store_op.y.axis = {z_batch.id, z_m.id, z_n.id};
  store_op.y.dtype = ge::DT_FLOAT;
  *store_op.y.strides = {m * n, n, af::ops::One};
  *store_op.y.repeats = {batch, m, n};

  af::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);
}

TEST_F(TestCodegenTiling, TestBatchMatmulDynamicShapeFuse) {
  af::AscGraph graph("batch_matmul_dynamic_pro");
  CreateBatchMatmulElemwiseDynamicGraph(graph);

  af::AscGraph mm_graph("batch_matmul_dynamic");
  CreateBatchMatmulDynamicGraph(mm_graph);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused_schedule_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);

  ascir::ScheduleGroup schedule_group2;
  schedule_group2.impl_graphs.push_back(mm_graph);
  fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups.push_back(schedule_group2);
  fused_schedule_result.node_idx_to_scheduled_results[0][0].cube_type = ascir::CubeTemplateType::kUBFuse;

  std::map<std::string, std::string> shape_info;
  shape_info["batch"] = "16";
  shape_info["m"] = "64";
  shape_info["n"] = "64";
  shape_info["k"] = "64";
  auto res = this->Generate(fused_schedule_result, shape_info, "", "0");

  std::fstream tiling_func("Batch_matmul_dynamic_fuse_tiling_func.cpp", std::ios::out);
  tiling_func << res["tiling_def_and_tiling_const"];

  auto pos = res["tiling_def_and_tiling_const"].find("TilingResult result = wrapper.DoMatMulTiling(");
  ASSERT_NE(pos, std::string::npos);

  auto dynamic_pos = res["tiling_def_and_tiling_const"].find("AutofuseIsStaticShape() {\n  return false;");
  ASSERT_NE(dynamic_pos, std::string::npos);
}

// ==================== Conv2D with Groups 测试 ====================

TEST_F(TestCodegenTiling, TestConv2dWithGroups) {
  af::AscGraph graph("conv2d_groups_pro");
  CreateElemwiseGraphWithRelu(graph);

  af::AscGraph conv2d_graph("conv2d_groups");
  CreateConv2DGraphWithGroups(conv2d_graph);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused_schedule_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);

  ascir::ScheduleGroup schedule_group2;
  schedule_group2.impl_graphs.push_back(conv2d_graph);
  fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups.push_back(schedule_group2);
  fused_schedule_result.node_idx_to_scheduled_results[0][0].cube_type = ascir::CubeTemplateType::kUBFuse;

  const std::map<std::string, std::string> shape_info;
  auto res = this->Generate(fused_schedule_result, shape_info, "", "0");

  std::fstream tiling_func("Conv2d_groups_tiling_func.cpp", std::ios::out);
  tiling_func << res["tiling_def_and_tiling_const"];

  VerifyTilingCodeBasic(res);
}

// ==================== Conv2D with Dilation 测试 ====================

TEST_F(TestCodegenTiling, TestConv2dWithDilation) {
  af::AscGraph graph("conv2d_dilation_pro");
  CreateElemwiseGraphWithRelu(graph);

  af::AscGraph conv2d_graph("conv2d_dilation");
  CreateConv2DGraphWithDilation(conv2d_graph);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused_schedule_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);

  ascir::ScheduleGroup schedule_group2;
  schedule_group2.impl_graphs.push_back(conv2d_graph);
  fused_schedule_result.node_idx_to_scheduled_results[0][0].schedule_groups.push_back(schedule_group2);
  fused_schedule_result.node_idx_to_scheduled_results[0][0].cube_type = ascir::CubeTemplateType::kUBFuse;

  const std::map<std::string, std::string> shape_info;
  auto res = this->Generate(fused_schedule_result, shape_info, "", "0");

  std::fstream tiling_func("Conv2d_dilation_tiling_func.cpp", std::ios::out);
  tiling_func << res["tiling_def_and_tiling_const"];

  VerifyTilingCodeBasic(res);
}
// --- Task 1: spec 协议最小字段集红灯测试 ---

TEST_F(TestCodegenTiling, ProtocolHeaderShouldContainMinimalRequestResponse) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_NE(tiling_impl.find("struct GetTilingRequest"), std::string::npos);
  EXPECT_NE(tiling_impl.find("struct CandidateSolution"), std::string::npos);
  EXPECT_NE(tiling_impl.find("struct GetTilingResponse"), std::string::npos);
}

TEST_F(TestCodegenTiling, ProtocolRequestShouldContainMinimalFields) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_NE(tiling_impl.find("const std::vector<std::map<std::string, std::string>> *input_configs = nullptr;"),
            std::string::npos);
  EXPECT_NE(tiling_impl.find("ResLimit *res_limit = nullptr;"), std::string::npos);
  EXPECT_NE(tiling_impl.find("int64_t topn = 1;"), std::string::npos);
}

TEST_F(TestCodegenTiling, ResLimitShouldUseOnlyPositiveOverrides) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_NE(tiling_impl.find("inline ResLimit GetResLimit(const ResLimit *res_limit)"), std::string::npos);
  EXPECT_NE(tiling_impl.find("if (res_limit->valid_num > 0U)"), std::string::npos);
  EXPECT_NE(tiling_impl.find("if (res_limit->aiv_num > 0U)"), std::string::npos);
  EXPECT_NE(tiling_impl.find("if (res_limit->aic_num > 0U)"), std::string::npos);
  EXPECT_NE(tiling_impl.find("if (res_limit->ub_size > 0U)"), std::string::npos);
  EXPECT_NE(tiling_impl.find("if (res_limit->resv[i] > 0U)"), std::string::npos);
  EXPECT_NE(tiling_impl.find("const ResLimit effective_res_limit = GetResLimit(res_limit);"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("res_limit == nullptr || res_limit->aiv_num == 0"), std::string::npos);
}

TEST_F(TestCodegenTiling, CandidateSolutionShouldContainOnlyMinimalFields) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_NE(tiling_impl.find("double modeled_perf = 0.0;"), std::string::npos);
  EXPECT_NE(tiling_impl.find("bool is_default = false;"), std::string::npos);
  EXPECT_NE(tiling_impl.find("std::string canonical_repr;"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("candidate.workspace ="), std::string::npos);
  EXPECT_EQ(tiling_impl.find("candidate.block_dim ="), std::string::npos);
}

TEST_F(TestCodegenTiling, ProtocolShouldNotContainBannedFields) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_EQ(tiling_impl.find("schedule_result_key"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("group_case_ids"), std::string::npos);
}

TEST_F(TestCodegenTiling, GetTilingShouldEnterMainSearchNotForKey) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_NE(tiling_impl.find(
                "int64_t GetTopnCandidateSolutions(const GetTilingRequest &request, GetTilingResponse &response)"),
            std::string::npos);
  // GetTiling must enter ATT main search skeleton via PGOSearchTilingKey (which internally calls
  // SearchAllTilingbyCaseId / ExecutePGOSolver). In codegen UT (no ATT callback), verify PGOSearchTilingKey
  // is used. In integrated environments (solver_func key present), also verify the internal functions.
  EXPECT_NE(tiling_impl.find("optiling::PGOSearchTilingKey("), std::string::npos);
  const std::string kSolverFunc = "solver_func";
  if (tiling_files.find(kSolverFunc) != tiling_files.end()) {
    std::string all_tiling_code = tiling_impl + tiling_files.at(kSolverFunc);
    EXPECT_NE(all_tiling_code.find("SearchAllTilingbyCaseId("), std::string::npos);
    EXPECT_NE(all_tiling_code.find("ExecutePGOSolver("), std::string::npos);
  }
  EXPECT_EQ(tiling_impl.find("for (int64_t key = 0; key < GetTilingKeyCount(); ++key)"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("if (AutofuseTiling(&default_tiling_data"), std::string::npos);
}

TEST_F(TestCodegenTiling, NoEarlyStopByTopnOrDefault) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_EQ(tiling_impl.find("if (request.topn == 1) { return"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("response.candidate_solutions.resize(topn)"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("partial_sort"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("current_candidate_num >= request.topn"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("if (found_default) break"), std::string::npos);
}

TEST_F(TestCodegenTiling, GenerateForPgoShouldUseTensorArgsForProfilingSignatures) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  fused_schedule_result.input_nodes.push_back(fused_schedule_result.input_nodes.front());
  fused_schedule_result.input_nodes.push_back(fused_schedule_result.input_nodes.front());
  fused_schedule_result.output_nodes.push_back(fused_schedule_result.output_nodes.front());
  auto tiling_code = this->GenerateForPgo(fused_schedule_result, "/tmp");

  EXPECT_NE(tiling_code.find("struct PgoTensorArgs"), std::string::npos);
  EXPECT_NE(tiling_code.find("PgoTensorArgs *tensor_args"), std::string::npos);
  EXPECT_EQ(tiling_code.find("void* input1,"), std::string::npos);
  EXPECT_EQ(tiling_code.find("void* input2,"), std::string::npos);
  EXPECT_EQ(tiling_code.find("void* output1,"), std::string::npos);
  EXPECT_NE(tiling_code.find("int WrapperOnlyLaunch(uint64_t workspace_size, AutofuseTilingData *tiling_data)"),
            std::string::npos);
  EXPECT_NE(tiling_code.find("int ProfilingBatchProcess(uint64_t workspace_size, "
                             "std::vector<AutofuseTilingDataPerf>::iterator begin"),
            std::string::npos);
  EXPECT_NE(tiling_code.find("DLOGE(\"malloc workspace failed, size: %\" PRIu64 \", ERROR: %d\""), std::string::npos);
  EXPECT_NE(tiling_code.find("if (workspace_size > SIZE_MAX)"), std::string::npos);
  codegen::PgoShapeStringStream pgo_shape_dim;
  const auto pgo_tiling_code =
      this->GenPgoAutofuseTiling(fused_schedule_result, pgo_shape_dim, "AutofuseTilingData", false);
  EXPECT_NE(pgo_tiling_code.find("CheckedWorkspaceAdd(*workspaceSize, 16ULL * 1024ULL * 1024ULL, *workspaceSize)"),
            std::string::npos);
  EXPECT_EQ(tiling_code.find("WrapperOnlyLaunch(PgoTensorArgs *tensor_args"), std::string::npos);
  EXPECT_EQ(tiling_code.find("ProfilingBatchProcess(PgoTensorArgs *tensor_args"), std::string::npos);
  EXPECT_NE(tiling_code.find("uint64_t input1;"), std::string::npos);
  EXPECT_NE(tiling_code.find("uint64_t output1;"), std::string::npos);
  EXPECT_NE(tiling_code.find("uint64_t tiling_addr;"), std::string::npos);
  EXPECT_NE(tiling_code.find("g_kernel_name + \"_\""), std::string::npos);
  EXPECT_EQ(tiling_code.find("AutofuseTilingData tiling_data;"), std::string::npos);
}

TEST_F(TestCodegenTiling, TfAndInductorPgoShouldSortAclDurationsConsistently) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol(64), af::Symbol(128)});

  const auto tf_source = GenerateForPgo(fused_schedule_result, "/tmp");
  const auto inductor_source = GenInductorPgoRunner(fused_schedule_result);

  // Duration ranking moved into libaihac_codegen (pgo_task_record.cpp); the
  // generated sources only delegate measurement to the collector ABI.
  EXPECT_EQ(tf_source.find("std::greater<int>()"), std::string::npos);
  EXPECT_EQ(tf_source.find("std::greater<double>()"), std::string::npos);
  EXPECT_EQ(inductor_source.find("std::greater<int>()"), std::string::npos);
  EXPECT_EQ(inductor_source.find("std::greater<double>()"), std::string::npos);
  EXPECT_NE(tf_source.find("AutofusePgoCollectorGetCandidateDurationNs"), std::string::npos);
  EXPECT_NE(inductor_source.find("AutofusePgoCollectorGetCandidateDurationNs"), std::string::npos);
  EXPECT_EQ(tf_source.find("mspti"), std::string::npos);
  EXPECT_EQ(inductor_source.find("mspti"), std::string::npos);
}

TEST_F(TestCodegenTiling, TfPgoGeneratedSourceContractShouldRemainStable) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol(64), af::Symbol(128)});
  fused_schedule_result.node_idx_to_scheduled_results[0].resize(1);

  const auto source = GenerateForPgo(fused_schedule_result, "/tmp/autofuse_pgo_source_contract");

  EXPECT_GT(source.size(), 0U);
  EXPECT_NE(source.find("PGOGetProfilingBatch"), std::string::npos);
  EXPECT_NE(source.find("const char *pgo_dir"), std::string::npos);
  EXPECT_NE(source.find("PgoTilingSearch"), std::string::npos);
  EXPECT_NE(source.find("static_pgo("), std::string::npos);
  EXPECT_NE(source.find("AutofusePgoCollectorBegin(collector, PGO_GRAPH_NAME)"), std::string::npos);
  EXPECT_NE(source.find("int PgoBeginCandidate(void *collector)"), std::string::npos);
  EXPECT_NE(source.find("const int begin_status = PgoBeginCandidate(g_pgo_collector);"), std::string::npos);
  EXPECT_EQ(source.find("int PgoBeginCandidate(void *collector, AutofuseTilingData *tiling_data, "
                        "uint32_t workspace_size)"),
            std::string::npos);
  EXPECT_NE(source.find("DLOGI(\"InitAclPgoEvents failed, status: %d\", status)"), std::string::npos);
  EXPECT_EQ(source.find("get_tiling_data_repr_fn(tiling_data)"), std::string::npos);
  EXPECT_EQ(source.find("kInductorPgoRunnerAbi"), std::string::npos);
  EXPECT_EQ(source.find("kPgoTopnMagic"), std::string::npos);
}

TEST_F(TestCodegenTiling, InductorPgoRunnerGeneratedSourceContractShouldRemainStable) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol(64), af::Symbol(128)});
  fused_schedule_result.node_idx_to_scheduled_results[0].resize(1);

  const auto source = GenInductorPgoRunner(fused_schedule_result);

  EXPECT_GT(source.size(), 0U);
  EXPECT_NE(source.find("PGOGetProfilingBatch"), std::string::npos);
  EXPECT_EQ(source.find("kInductorPgoRunnerAbi"), std::string::npos);
  EXPECT_NE(source.find("GenerateMeasuredTopnSolutions"), std::string::npos);
  EXPECT_NE(source.find("int PgoBeginCandidate(void *collector, AutofuseTilingData *tiling_data, "
                        "uint64_t workspace_size)"),
            std::string::npos);
  EXPECT_NE(source.find("const int begin_status = PgoBeginCandidate(g_pgo_collector, tiling_data, workspace_size);"),
            std::string::npos);
  EXPECT_NE(source.find("get_tiling_data_repr_fn(tiling_data)"), std::string::npos);
  EXPECT_EQ(source.find("__attribute__((weak))"), std::string::npos);
  EXPECT_NE(source.find("kPgoTopnMagic"), std::string::npos);
  EXPECT_EQ(source.find("const char *pgo_dir"), std::string::npos);
  EXPECT_EQ(source.find("static_pgo("), std::string::npos);
}

TEST_F(TestCodegenTiling, PgoConfigShouldKeepCurrentTensorArgs) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto pgo_config_code = this->PGOProfilingCallbackDef(fused_schedule_result, "AutofuseTilingData");

  EXPECT_NE(pgo_config_code.find("PgoTensorArgs *tensor_args = nullptr;"), std::string::npos);
}

TEST_F(TestCodegenTiling, ExecutePGOSolverShouldUseTensorArgsInsteadOfExpandedInputs) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  fused_schedule_result.input_nodes.push_back(fused_schedule_result.input_nodes.front());
  fused_schedule_result.input_nodes.push_back(fused_schedule_result.input_nodes.front());
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);

  std::string all_tiling_code;
  for (const auto &tiling_file : tiling_files) {
    all_tiling_code += tiling_file.second;
  }
  EXPECT_NE(all_tiling_code.find("PgoTensorArgs *tensor_args"), std::string::npos);
  EXPECT_EQ(all_tiling_code.find("void* input1,"), std::string::npos);
  EXPECT_EQ(all_tiling_code.find("void* input2,"), std::string::npos);
  if (all_tiling_code.find("ExecutePGOSolver(") != std::string::npos) {
    EXPECT_EQ(all_tiling_code.find("(void)input1"), std::string::npos);
  }
}

TEST_F(TestCodegenTiling, TopnSearchTilingKeyShouldPassSingleTensorArgs) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  fused_schedule_result.input_nodes.push_back(fused_schedule_result.input_nodes.front());
  fused_schedule_result.input_nodes.push_back(fused_schedule_result.input_nodes.front());
  fused_schedule_result.output_nodes.push_back(fused_schedule_result.output_nodes.front());
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);

  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_NE(tiling_impl.find("PGOSearchTilingKey(raw_candidates, cur_search_tiling, -1, &cur_search_tiling, "
                             "nullptr, nullptr, 0, best_perf"),
            std::string::npos);
  EXPECT_EQ(tiling_impl.find("PGOSearchTilingKey(raw_candidates, cur_search_tiling, -1, &cur_search_tiling, "
                             "nullptr, nullptr, nullptr,"),
            std::string::npos);
}

// Task 5: multi-group must carry workspace_map / block_dim_vec

TEST_F(TestCodegenTiling, MultiGroupDoesNotCarryWorkspaceMap) {
  auto fused_schedule_result = GenMultiGroupFusedScheduleResult();
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  // Multi-group GetTiling does not declare workspace_map (single-group only)
  EXPECT_EQ(tiling_impl.find("std::unordered_map<int64_t, uint64_t> workspace_map"), std::string::npos);
}

// Task 6: Bridge layer — raw-candidate to CandidateSolution mapping

TEST_F(TestCodegenTiling, BridgeMapsModeledPerfFromFinalComparablePerf) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_NE(tiling_impl.find("double final_modeled_perf ="), std::string::npos);
  EXPECT_NE(tiling_impl.find("solution.modeled_perf = final_modeled_perf;"), std::string::npos);
}

TEST_F(TestCodegenTiling, BridgePreservesSingleGroupComparablePerfSemantics) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_NE(tiling_impl.find("CandidateSolution solution;"), std::string::npos);
  EXPECT_NE(tiling_impl.find("solution.modeled_perf = final_modeled_perf;"), std::string::npos);
  EXPECT_NE(tiling_impl.find("std::isfinite(final_modeled_perf)"), std::string::npos);
}

TEST_F(TestCodegenTiling, BridgeAlwaysPreservesDefaultCandidateBeforeTopnSearch) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  const auto default_tiling_pos = tiling_impl.find("if (GetTiling(default_tiling, -1))");
  const auto config_path_pos =
      tiling_impl.find("const bool internal_no_config_path = (request.input_configs == nullptr);");
  const auto search_loop_pos = tiling_impl.find("for (const auto *cfg : config_ptrs)");
  ASSERT_NE(default_tiling_pos, std::string::npos);
  ASSERT_NE(config_path_pos, std::string::npos);
  ASSERT_NE(search_loop_pos, std::string::npos);
  EXPECT_LT(default_tiling_pos, config_path_pos);
  EXPECT_LT(default_tiling_pos, search_loop_pos);
  EXPECT_NE(tiling_impl.find("default_repr = GetTilingDataRepr(&default_tiling)"), std::string::npos);
  EXPECT_NE(tiling_impl.find("solution.is_default = !default_repr.empty() && "
                             "(solution.canonical_repr == default_repr);"),
            std::string::npos);
  EXPECT_NE(tiling_impl.find("if (solution.is_default) { found_default_candidate = true; }"), std::string::npos);
}

TEST_F(TestCodegenTiling, BridgeDoesNotWriteWorkspaceOrBlockDim) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  // Bridge layer must not write workspace into protocol object
  EXPECT_EQ(tiling_impl.find("solution.workspace ="), std::string::npos);
  EXPECT_EQ(tiling_impl.find("solution.block_dim ="), std::string::npos);
}

// Task 7: Selector + ABI backfill

TEST_F(TestCodegenTiling, WrapperUsesSelectorForTopn) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  // Wrapper must call SelectTopnCandidateSolutions
  EXPECT_NE(tiling_impl.find("SelectTopnCandidateSolutions(response.candidate_solutions, topn)"), std::string::npos);
}

TEST_F(TestCodegenTiling, WrapperBackfillsWorkspaceAndBlockDim) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  // Workspace must be dynamically computed from tiling_data
  EXPECT_NE(tiling_impl.find("GetWorkspaceSize(sol.tiling_data, workspace_size)"), std::string::npos);
  // Block dim must be dynamically extracted from tiling_data
  EXPECT_NE(tiling_impl.find("sol.tiling_data.get_block_dim()"), std::string::npos);
}

TEST_F(TestCodegenTiling, TopnWrapperPrintsCompleteFinalTilingDataInChunks) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);

  EXPECT_NE(tiling_impl.find("GenerateTopnSolutions enter: topn=%ld, input_configs.size=%zu operator=%s"),
            std::string::npos);
  EXPECT_NE(tiling_impl.find("output[%zu]: perf=%.6f is_default=%d block_dim=%ld"), std::string::npos);
  EXPECT_NE(tiling_impl.find("output[%zu] repr[%zu..%zu]: %.*s"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("block_dim=%ld repr=%s"), std::string::npos);
}

TEST_F(TestCodegenTiling, TemplateSemanticTopnCapturesCandidateIdentity) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")}, af::ops::One);
  fused_schedule_result.node_idx_to_scheduled_results[0][0].semantic_name = "Pointwise/Test";
  this->codegen_func_ = att::GenTilingImplAutoFuseV3;
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  ASSERT_TRUE(tiling_files.find(codegen::kTilingApiHeaderIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  const auto &api_header = tiling_files.at(codegen::kTilingApiHeaderIdentify);

  EXPECT_NE(tiling_impl.find("std::vector<TemplateSemanticInfo> semantic_infos{};"), std::string::npos);
  EXPECT_NE(api_header.find("struct TemplateSemanticInfo;"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("optiling::GetTemplateSemanticInfos(response.candidate_solutions.back().tiling_data)"),
            std::string::npos);
  std::string all_tiling_code;
  for (const auto &tiling_file : tiling_files) {
    all_tiling_code += tiling_file.second;
  }
  EXPECT_NE(all_tiling_code.find("const char *GetSemanticName() const override { return \"Pointwise/Test\"; }"),
            std::string::npos);
  EXPECT_NE(tiling_impl.find("record.tiling_key = info.tiling_key;"), std::string::npos);
  EXPECT_NE(tiling_impl.find("if (ShouldEmitFinalTiling()) {"), std::string::npos);
  EXPECT_NE(tiling_impl.find("EmitTemplateSemanticRecord(record);"), std::string::npos);
}

TEST_F(TestCodegenTiling, TemplateSemanticInductorSingleGroupEmitsAttReturn) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")}, af::Symbol(1));
  fused_schedule_result.node_idx_to_scheduled_results[0].resize(1);
  this->codegen_func_ = att::GenTilingImplAutoFuseV3;
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  std::string all_tiling_code;
  for (const auto &tiling_file : tiling_files) {
    all_tiling_code += tiling_file.second;
  }

  EXPECT_NE(tiling_impl.find("optiling::EmitFinalTilingByCase(*tiling, tiling->get_block_dim(), "
                             "\"runtime\", \"default\", 0U, -1);"),
            std::string::npos);
  EXPECT_NE(all_tiling_code.find("record.tiling_key = context.tiling_key;"), std::string::npos);
  EXPECT_NE(all_tiling_code.find("HasAmbiguousFinalTilingCase(final_tiling_key, nullptr)"), std::string::npos);
  EXPECT_NE(all_tiling_code.find("set_semantic_fallback();"), std::string::npos);
}

TEST_F(TestCodegenTiling, TemplateSemanticTopnAggregatesMultiGroupInfos) {
  auto fused_schedule_result = this->GenTilingKeyCountResult({{1, 1}}, af::Symbol(1));
  this->codegen_func_ = att::GenTilingImplAutoFuseV3;
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);

  EXPECT_NE(tiling_impl.find("std::vector<TemplateSemanticInfo> "
                             "GetTemplateSemanticInfos(AutofuseTilingData &tiling_data)"),
            std::string::npos);
  EXPECT_NE(tiling_impl.find("if (tiling_data.get_graph0_tiling_key() == 0U) {"), std::string::npos);
  EXPECT_NE(tiling_impl.find("optiling::AscGraph0ScheduleResult0G0::GetTemplateSemanticInfos("
                             "tiling_data.graph0_result0_g0_tiling_data)"),
            std::string::npos);
  EXPECT_NE(tiling_impl.find("optiling::AscGraph0ScheduleResult0G1::GetTemplateSemanticInfos("
                             "tiling_data.graph0_result0_g1_tiling_data)"),
            std::string::npos);
  std::string all_tiling_code;
  for (const auto &tiling_file : tiling_files) {
    all_tiling_code += tiling_file.second;
  }
  EXPECT_EQ(all_tiling_code.find("tiling_data.graph0_result0_g0_tiling_data.get_tiling_key()"), std::string::npos);
  EXPECT_EQ(all_tiling_code.find("tiling_data.graph0_result0_g1_tiling_data.get_tiling_key()"), std::string::npos);
}

TEST_F(TestCodegenTiling, TemplateSemanticTopnLogsOnlyAfterSuccessfulSwap) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")}, af::ops::One);
  this->codegen_func_ = att::GenTilingImplAutoFuseV3;
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);

  const auto tiling_swap_pos = tiling_impl.find("tiling_datas.swap(export_tilings);");
  const auto workspace_swap_pos = tiling_impl.find("workspaces.swap(export_workspaces);");
  const auto block_dim_swap_pos = tiling_impl.find("block_dims.swap(export_block_dims);");
  const auto semantic_log_pos = tiling_impl.find("EmitTemplateSemanticRecord(record);");
  ASSERT_NE(tiling_swap_pos, std::string::npos);
  ASSERT_NE(workspace_swap_pos, std::string::npos);
  ASSERT_NE(block_dim_swap_pos, std::string::npos);
  ASSERT_NE(semantic_log_pos, std::string::npos);
  const auto topn_entry_pos = tiling_impl.find("extern \"C\" int64_t GenerateTopnSolutions");
  ASSERT_NE(topn_entry_pos, std::string::npos);
  const auto topn_success_return_pos = tiling_impl.find("return 0;", topn_entry_pos);
  ASSERT_NE(topn_success_return_pos, std::string::npos);
  const auto topn_entry_body = tiling_impl.substr(topn_entry_pos, topn_success_return_pos - topn_entry_pos);
  EXPECT_EQ(topn_entry_body.find("return;"), std::string::npos);
  EXPECT_NE(topn_entry_body.find("if (ShouldEmitFinalTiling()) {"), std::string::npos);
  EXPECT_LT(tiling_swap_pos, semantic_log_pos);
  EXPECT_LT(workspace_swap_pos, semantic_log_pos);
  EXPECT_LT(block_dim_swap_pos, semantic_log_pos);
  std::string all_tiling_code;
  for (const auto &tiling_file : tiling_files) {
    all_tiling_code += tiling_file.second;
  }
  EXPECT_NE(all_tiling_code.find("record.stage = \"topn_output\";"), std::string::npos);
  EXPECT_NE(all_tiling_code.find("unavailable_record.name_status = \"unavailable\";"), std::string::npos);
  EXPECT_NE(all_tiling_code.find("std::vector<std::string> semantic_names = info.semantic_names;"), std::string::npos);
  EXPECT_NE(all_tiling_code.find("semantic_names.size() > 1U ? \"ambiguous\""), std::string::npos);
}

TEST_F(TestCodegenTiling, TemplateSemanticLogUsesBoundedFormatterAndEscapeHelper) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")}, af::ops::One);
  this->codegen_func_ = att::GenTilingImplAutoFuseV3;
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  std::string all_tiling_code;
  for (const auto &tiling_file : tiling_files) {
    all_tiling_code += tiling_file.second;
  }

  EXPECT_NE(all_tiling_code.find("FormatTemplateSemanticRecord"), std::string::npos);
  EXPECT_NE(all_tiling_code.find("kTemplateSemanticLineBudget = 700U"), std::string::npos);
  EXPECT_NE(all_tiling_code.find("FinalTilingShellQuote(semantic_name)"), std::string::npos);
  EXPECT_NE(all_tiling_code.find("ParseTemplateSemanticName"), std::string::npos);
  EXPECT_EQ(all_tiling_code.find("IsTemplateSemanticRole"), std::string::npos);
  EXPECT_NE(all_tiling_code.find("FinalTilingJsonQuote(group.semantic_name)"), std::string::npos);
  EXPECT_NE(all_tiling_code.find("semantic_name\\\":\""), std::string::npos);
  EXPECT_NE(all_tiling_code.find("semantic_name="), std::string::npos);
  EXPECT_NE(all_tiling_code.find("semantic_schema=1"), std::string::npos);
  EXPECT_NE(all_tiling_code.find("family="), std::string::npos);
  EXPECT_NE(all_tiling_code.find("strategy="), std::string::npos);
  EXPECT_NE(all_tiling_code.find("role="), std::string::npos);
  EXPECT_NE(all_tiling_code.find("api_features="), std::string::npos);
  EXPECT_NE(all_tiling_code.find("transforms="), std::string::npos);
  EXPECT_NE(all_tiling_code.find("semantic_index="), std::string::npos);
  EXPECT_NE(all_tiling_code.find("semantic_name_count="), std::string::npos);
  EXPECT_NE(all_tiling_code.find("TemplateSemanticDetailLines"), std::string::npos);
  EXPECT_NE(all_tiling_code.find("TemplateSemanticFieldSummary"), std::string::npos);
  EXPECT_NE(all_tiling_code.find("TemplateSemanticLineSummary"), std::string::npos);
  EXPECT_NE(all_tiling_code.find("line.size() > kTemplateSemanticLineBudget"), std::string::npos);
  EXPECT_EQ(all_tiling_code.find("TemplateSemanticShellQuote"), std::string::npos);
}

TEST_F(TestCodegenTiling, TemplateSemanticFormatterBoundsEveryGeneratedLine) {
  const std::string helpers = att::final_tiling::GenFinalTilingLogHelpers(false);
  std::string source = R"(
#define CheckCachedLogLevel(level) true
#define ATT_FINAL_TILING_LOGI(op, fmt, ...) (void)0
#define OP_NAME "test"
)";
  source += helpers;
  source += R"(
int main() {
  const auto bounded = [](const std::vector<std::string> &lines) {
    for (const auto &line : lines) {
      if (line.size() > 700U) { return false; }
    }
    return true;
  };
  TemplateSemanticRecord core;
  core.stage = "topn_output";
  core.has_output = true;
  core.has_group = true;
  core.semantic_name.assign(800U, 'x');
  if (!bounded(FormatTemplateSemanticRecord(core))) { return 1; }

  TemplateSemanticRecord detail = core;
  detail.semantic_name = "Pointwise/Test";
  detail.has_identity = true;
  detail.template_name.assign(800U, 'x');
  const auto lines = FormatTemplateSemanticRecord(detail);
  bool has_template_summary = false;
  for (const auto &line : lines) {
    if (line.find("template_hash=") != std::string::npos) { has_template_summary = true; }
  }
  return bounded(lines) && has_template_summary ? 0 : 1;
}
)";
  ASSERT_TRUE(CompileAndRunCode(source));
}

TEST_F(TestCodegenTiling, TemplateSemanticFormatterReadsCanonicalFields) {
  std::string source = R"(
#define CheckCachedLogLevel(level) true
#define ATT_FINAL_TILING_LOGI(op, fmt, ...) (void)0
#define OP_NAME "test"
)";
  source += att::final_tiling::GenFinalTilingLogHelpers(false);
  source += R"(
int main() {
  const std::string name = "Reduce/Common@Norm#AR|NoReuseSrc+NDDMA.Cast";
  const auto fields = ParseTemplateSemanticName(name);
  if (fields.family != "Reduce" || fields.strategy != "Common" || fields.role != "Norm" ||
      fields.api_features != std::vector<std::string>{"AR", "NoReuseSrc"} ||
      fields.transforms != std::vector<std::string>{"NDDMA.Cast"}) { return 1; }
  const auto parent = ParseTemplateSemanticName("Reduce/Common@Norm");
  if (parent.strategy != "Common" || parent.role != "Norm" || !parent.api_features.empty()) { return 3; }
  const auto api_only = ParseTemplateSemanticName("Reduce/Common#AR|ReuseSrc");
  if (api_only.strategy != "Common" || !api_only.role.empty() || api_only.api_features.size() != 2U) { return 4; }
  const auto family = ParseTemplateSemanticName("Reduce");
  if (family.family != "Reduce" || !family.strategy.empty()) { return 5; }
  FinalTilingGroupSelection group{};
  group.semantic_name = name;
  const std::string summary = FinalTilingGroupsJson({group});
  return summary.find("\"semantic_name\":\"" + name + "\"") != std::string::npos ? 0 : 6;
}
)";
  ASSERT_TRUE(CompileAndRunCode(source));
}

TEST_F(TestCodegenTiling, CachedTemplateSemanticFallbackExecutesKeyMapping) {
  att::TilingModelInfo models(3);
  for (size_t index = 0; index < models.size(); ++index) {
    models[index].tiling_case_id = index == 0 ? 0U : 1U;
    models[index].sub_case_tag = std::to_string(index);
    models[index].semantic_name =
        index == 0 ? "Pointwise/Common" : (index == 1 ? "Reduce/Common@Norm#AR" : "Reduce/Common@Norm#RA");
  }
  att::TilingCodeGenConfig config;
  config.tiling_data_type_name = "TestTilingData";
  att::HighPerfTilingCodeGenImpl generator("test", config, models, {}, false);
  ASSERT_EQ(generator.GenTemplateSemanticInfos(), af::SUCCESS);
  const std::string mapping = generator.tiling_func_.GetOutputStr();
  generator.GenFinalTilingSemanticFallback();
  std::string source = R"(
#include <string>
#include <vector>
std::vector<std::string> emitted_lines;
#define CheckCachedLogLevel(level) true
#define ATT_FINAL_TILING_LOGI(op, fmt, value) emitted_lines.emplace_back(value)
#define OP_NAME "test"
)" + att::final_tiling::GenFinalTilingLogHelpers(false) +
                       R"(
struct TestTilingData { uint32_t key; uint32_t get_tiling_key() { return key; } };
int32_t GetFinalTilingCaseId(uint32_t key) { return key < 2U ? key : -1; }
)" + mapping + R"(
int main() {
  const char *source = "runtime", *selection_mode = "default";
  const uint32_t group_id = 0U;
  const bool ambiguous_sub_case = false;
  for (uint32_t key : {0U, 1U, 99U}) {
    TestTilingData tiling_data{key};
    FinalTilingContext final_context{};
    final_context.tiling_key = key;
)" + generator.tiling_func_.GetOutputStr().substr(mapping.size()) +
                       R"(
    emitted_lines.clear();
    set_semantic_fallback();
    EmitTemplateSemanticRecord(final_context, key == 1U);
    if (emitted_lines.empty()) { return 1; }
    const std::string &line = emitted_lines.front();
    const std::string name = key == 0U ? "Pointwise/Common" : (key == 1U ? "Reduce/Common@Norm" : "Unknown");
    const std::string status = key == 0U ? "" : (key == 1U ? "ambiguous" : "unavailable");
    if (line.find("semantic_name=\"" + name + "\"") == std::string::npos ||
        (!status.empty() && line.find("name_status=" + status) == std::string::npos)) { return 2; }
  }
  return 0;
}
)";
  ASSERT_TRUE(CompileAndRunCode(source));
}

TEST_F(TestCodegenTiling, ReuseFinalTilingLogsCurrentIdentityForSelectedAndCachedCases) {
  att::TilingModelInfo models(3);
  att::ReuseScheduleGroupInfo primary, current;
  primary.tiling_keys = {0U, 1U, 1U};
  current.tiling_keys = {7U, 8U, 8U};
  for (size_t index = 0UL; index < models.size(); ++index) {
    models[index].schedule_group_ident = {2UL, 3UL, 4UL};
    models[index].reuse_schedule_group = std::make_shared<att::ReuseScheduleGroup>();
    models[index].tiling_case_id = current.tiling_keys[index];
    models[index].sub_case_tag = std::to_string(index);
    models[index].semantic_name =
        index == 0UL ? "Pointwise/Common" : (index == 1UL ? "Reduce/Common@Norm#AR" : "Reduce/Common@Norm#RA");
  }
  att::TilingCodeGenConfig config;
  config.tiling_data_type_name = "Data";
  att::HighPerfTilingCodeGenImpl generator("test", config, models, {}, false);
  std::map<att::ScheduleGroupIdent, att::ReuseScheduleGroupInfo> infos{{{2UL, 3UL, 4UL}, current}};
  ASSERT_EQ(generator.GenTemplateSemanticInfos(), af::SUCCESS);
  ASSERT_EQ(generator.GenReuseGroupTilingWrapperFinalEmitter("Cur", "Primary", primary, infos.cbegin()), af::SUCCESS);
  std::string source = R"(
#include <string>
#include <vector>
std::vector<std::string> emitted_lines;
bool log_enabled = true;
#define CheckCachedLogLevel(level) log_enabled
#define ATT_FINAL_TILING_LOGI(op, fmt, value) emitted_lines.emplace_back(value)
#define OP_NAME "test"
)" + att::final_tiling::GenFinalTilingLogHelpers(false) +
                       R"(
struct Data { uint32_t key; uint32_t get_tiling_key() const { return key; } void set_tiling_key(uint32_t v) { key = v; } };
using CurTilingData = Data;
using PrimaryTilingData = Data;
template<class TI, class TO> TO &RefToRef(TI &v) { return reinterpret_cast<TO &>(v); }
namespace Primary {
bool exact = true; bool fail_context = false;
int calls = 0;
void FillContext(Data &data, int32_t id, FinalTilingContext &context) {
  ++calls; if (fail_context) { throw std::bad_alloc(); }
  if (id >= 0 && id != static_cast<int32_t>(data.key)) { std::abort(); }
  context.case_id = id; context.tiling_key = data.key; context.score = 17;
  context.sub_case_tag = data.key == 0U ? "primary_tag" : "1";
  context.template_name = "graph0_result0_g0_0";
  context.semantic_name = data.key == 0U ? "Pointwise/Common" : "Reduce/Common@Norm#AR";
  context.semantic_names = {context.semantic_name};
}
bool GetFinalTilingContextFromSnapshot(Data &data, uint32_t, int32_t id,
                                       const FinalTilingSelectionSnapshot &snapshot, FinalTilingContext &context) {
  if (!snapshot.valid || snapshot.graph != 0U || snapshot.result != 0U || snapshot.group != 0U ||
      snapshot.tiling_key != data.key) { return false; }
  FillContext(data, id, context);
  return exact;
}
bool GetFinalTilingSelectionFromSnapshot(Data &data, const FinalTilingSelectionSnapshot &snapshot,
                                         FinalTilingGroupSelection &selection) {
  selection.case_id = snapshot.case_id;
  selection.tiling_key = data.key;
  selection.score = 17;
  selection.sub_case_tag = data.key == 0U ? "primary_tag" : "1";
  selection.template_name = "graph0_result0_g0_0";
  selection.semantic_name = data.key == 0U ? "Pointwise/Common" : "Reduce/Common@Norm#AR";
  return exact;
}
}
)" + generator.tiling_func_.GetOutputStr() +
                       R"(
int main() {
  log_enabled = false;
  Data disabled{7U};
  EmitFinalTilingByCase(disabled, 32U, "runtime", "default", 4U, 7);
  if (Primary::calls != 0 || !emitted_lines.empty()) { return 3; }
  log_enabled = true;
  FinalTilingObservationContext observation;
  FinalTilingObservationCallGuard observation_guard(observation, true, 1U);
  FinalTilingObservationSlotGuard observation_slot_guard(0U);
  for (bool exact : {true, false}) {
    Primary::exact = exact;
    for (uint32_t key : {7U, 8U}) {
      Data data{key}; emitted_lines.clear();
      CommitCurrentFinalTilingSelection(
          {true, 2U, 3U, 4U, key, static_cast<int32_t>(key), key == 7U ? 0 : 1});
      EmitFinalTilingByCase(data, 32U, "runtime", "default", 4U, static_cast<int32_t>(key));
      const std::string name = key == 7U ? "Pointwise/Common" :
          (exact ? "Reduce/Common@Norm#AR" : "Reduce/Common@Norm");
      const std::string templ = key == 7U ? "graph2_result3_g4_07" :
          (exact ? "graph2_result3_g4_18" : "");
      std::string logs;
      for (const auto &line : emitted_lines) { logs += line + "\n"; }
      if (data.key != key || logs.find("semantic_name=\"" + name + "\"") == std::string::npos ||
          logs.find("detail graph=2 result=3 tiling_key=" + std::to_string(key) + " template=\"" + templ + "\"") ==
              std::string::npos) { return 1; }
      if (exact && logs.find("score=17") == std::string::npos) { return 4; }
      if (!exact && key == 8U && logs.find("name_status=ambiguous") == std::string::npos) { return 2; }
    }
  }
  Primary::fail_context = true;
  CommitCurrentFinalTilingSelection({true, 2U, 3U, 4U, 7U, 7, 0});
  Data unchanged{7U};
  try { EmitFinalTilingByCase(unchanged, 32U, "runtime", "default", 4U, 7); } catch (...) { return 5; }
  return unchanged.key == 7U && !observation.enabled ? 0 : 6;
}
)";
  ASSERT_TRUE(CompileAndRunCode(source));
}

TEST_F(TestCodegenTiling, TemplateSemanticMetadataIsCanonicalBeforeLogging) {
  att::TilingModelInfo models(3);
  const std::vector<std::string> names = {"Reduce/Common/Norm/AR/NoReuseSrc+NDDMA.Cast", "Concat/UB/Extend/Copy",
                                          "Reduce/Common@Norm#RA"};
  for (size_t index = 0UL; index < models.size(); ++index) {
    models[index].tiling_case_id = index;
    models[index].semantic_name = ascgen_utils::NormalizeTemplateSemanticName(names[index]);
  }
  att::TilingCodeGenConfig config;
  config.tiling_data_type_name = "Data";
  att::HighPerfTilingCodeGenImpl generator("test", config, models, {}, true);
  ASSERT_EQ(generator.GenTemplateSemanticInfos(), af::SUCCESS);
  std::string source = R"(
#define CheckCachedLogLevel(level) true
#define ATT_FINAL_TILING_LOGI(op, fmt, value) (void)0
#define OP_NAME "test"
)" + att::final_tiling::GenFinalTilingLogHelpers(false) +
                       R"(
struct Data { uint32_t key; uint32_t get_tiling_key() const { return key; } };
)" + generator.tiling_func_.GetOutputStr() +
                       R"(
int main() {
  const std::vector<std::string> expected = {"Reduce/Common@Norm#AR|NoReuseSrc+NDDMA.Cast",
                                           "Concat/UB#Extend|Copy", "Reduce/Common@Norm#RA"};
  for (uint32_t key = 0U; key < expected.size(); ++key) {
    Data data{key}; const auto infos = GetTemplateSemanticInfos(data);
    if (infos.size() != 1U || infos[0].semantic_name != expected[key]) { return 1; }
  }
  return 0;
}
)";
  ASSERT_TRUE(CompileAndRunCode(source));
}

TEST_F(TestCodegenTiling, ReuseObservationStoresLogicalGroupIdentity) {
  att::TilingModelInfo models(1);
  models[0].schedule_group_ident = {2UL, 3UL, 4UL};
  att::TilingCodeGenConfig config;
  config.tiling_data_type_name = "Data";
  att::HighPerfTilingCodeGenImpl generator("test", config, models, {}, false);
  generator.SetFinalTilingObservationMetadata(1UL, 2UL, {{"", 0}});
  att::ReuseScheduleGroupInfo primary;
  primary.tiling_keys = {0U};
  std::map<att::ScheduleGroupIdent, att::ReuseScheduleGroupInfo> infos{{{2UL, 3UL, 4UL}, primary}};
  ASSERT_EQ(generator.GenReuseGroupTilingWrapperGetTiling("Cur", "Primary", primary, infos.cbegin()), af::SUCCESS);
  std::string source = R"(
#include <unordered_map>
#define CheckCachedLogLevel(level) true
#define ATT_FINAL_TILING_LOGI(op, fmt, value) (void)0
#define OP_NAME "test"
)" + att::final_tiling::GenFinalTilingLogHelpers(false) +
                       R"(
struct Data { uint32_t key{0}; uint32_t get_tiling_key() const { return key; }
  void set_tiling_key(uint32_t value) { key = value; } };
using CurTilingData = Data;
using PrimaryTilingData = Data;
template<class TI, class TO> TO &RefToRef(TI &value) { return value; }
namespace Primary {
bool GetTiling(Data &, std::unordered_map<int64_t, uint64_t> &, int32_t) {
  FinalTilingObservationSlotGuard guard(0U);
  CommitCurrentFinalTilingSelection({true, 0U, 0U, 0U, 0U, 0, 1}); return true;
}
}
)" + generator.tiling_func_.GetOutputStr() +
                       R"(
int main() {
  FinalTilingObservationContext context;
  FinalTilingObservationCallGuard guard(context, true, 2U);
  Data data; std::unordered_map<int64_t, uint64_t> workspace;
  if (!GetTiling(data, workspace, -1)) { return 1; }
  const auto &snapshot = context.selections[1];
  return snapshot.valid && snapshot.graph == 2U && snapshot.result == 3U && snapshot.group == 4U ? 0 : 2;
}
)";
  ASSERT_TRUE(CompileAndRunCode(source));
}

TEST_F(TestCodegenTiling, PgoObservationCapturesCurrentCaseInsteadOfDefault) {
  att::TilingModelInfo models(1);
  models[0].tiling_case_id = 7U;
  att::TilingCodeGenConfig config;
  config.tiling_data_type_name = "Data";
  att::HighPerfTilingCodeGenImpl generator("test", config, models, {}, true);
  generator.GenInductorExecutePGOSolver(models[0]);
  std::string source = R"(
#include <unordered_map>
#define CheckCachedLogLevel(level) true
#define ATT_FINAL_TILING_LOGI(op, fmt, value) (void)0
#define OP_NAME "test"
)" + att::final_tiling::GenFinalTilingLogHelpers(false) +
                       R"(
struct Data { uint32_t key{0}; uint32_t get_tiling_key() const { return key; } };
using AutofuseTilingData = Data;
struct AutofuseTilingDataPerf { Data tiling_data; double best_perf;
  std::vector<FinalTilingSelectionSnapshot> final_tiling_selections; };
struct SearchConfig {};
struct Base { virtual bool ExecutePGOSolver(Data &, std::vector<AutofuseTilingDataPerf> &, Data *, void *,
  std::unordered_map<int64_t, uint64_t> &, std::vector<uint32_t *>, const SearchConfig *) = 0; };
struct Case : Base {
  const SearchConfig *pending_search_cfg_{nullptr};
  bool GetTiling(Data &data, double &) { data.key = 7U; return true; }
  double GetPerf(Data &) { return 1.0; }
)" + generator.tiling_func_.GetOutputStr() +
                       R"(
};
int main() {
  FinalTilingObservationContext context;
  FinalTilingObservationCallGuard guard(context, true, 2U);
  { FinalTilingObservationSlotGuard slot(0U);
    CommitCurrentFinalTilingSelection({true, 0U, 0U, 0U, 14U, 14, 0}); }
  Data data; std::vector<AutofuseTilingDataPerf> candidates;
  std::unordered_map<int64_t, uint64_t> workspace;
  Case impl;
  if (!impl.ExecutePGOSolver(data, candidates, nullptr, nullptr, workspace)) { return 1; }
  const auto &snapshots = candidates[0].final_tiling_selections;
  if (snapshots[0].tiling_key != 7U || snapshots[1].valid || context.selections[0].tiling_key != 14U) { return 2; }
  RestoreFinalTilingSelectionSnapshots(snapshots);
  return context.selections[0].tiling_key == 7U && !context.selections[1].valid ? 0 : 3;
}
)";
  ASSERT_TRUE(CompileAndRunCode(source));
}

TEST_F(TestCodegenTiling, TemplateSemanticTopnDedupKeepsWinningObservationWithoutCopy) {
  for (bool measured : {false, true}) {
    std::stringstream helpers;
    if (measured) {
      GenDeduplicateMeasuredCandidateSolutions(helpers);
    } else {
      GenDeduplicateCandidateSolutions(helpers);
    }
    std::string source = R"(
#include <cmath>
#include <map>
#include <unordered_map>
#define OP_LOGI(...) (void)0
#define OP_LOGW(...) (void)0
#define CheckCachedLogLevel(level) true
#define ATT_FINAL_TILING_LOGI(op, fmt, value) (void)0
#define OP_NAME "test"
)" + att::final_tiling::GenFinalTilingLogHelpers(false) +
                         R"(
struct Data {}; struct ResLimit {};
)" + GenCandidateSolutionProtocolForInductor("Data", true) +
                         helpers.str() + "constexpr bool kMeasured = " + (measured ? "true;" : "false;") + R"(
int main() {
  std::vector<CandidateSolution> solutions(3U);
  for (size_t index = 0UL; index < solutions.size(); ++index) {
    auto &solution = solutions[index];
    solution.canonical_repr = index < 2UL ? "same" : "other";
    solution.modeled_perf = kMeasured ? 3.0 - index : 3.0;
    solution.is_default = !kMeasured && index == 1UL;
    solution.semantic_infos.resize(1U);
    solution.semantic_infos[0].semantic_name.assign(128U, static_cast<char>('A' + index));
  }
  const char *kept_name = solutions[1].semantic_infos[0].semantic_name.data();
  DeduplicateCandidateSolutions(solutions);
  return solutions.size() == 2U && solutions[0].semantic_infos[0].semantic_name.data() == kept_name &&
      solutions[0].semantic_infos[0].semantic_name == std::string(128U, 'B') ? 0 : 1;
}
)";
    ASSERT_TRUE(CompileAndRunCode(source));
  }
}

TEST_F(TestCodegenTiling, FinalTilingObservationAllocationFailuresPreserveSolverAndReentry) {
  std::string source = R"(
#include <cstdlib>
#include <new>
bool fail_allocation = false;
void *operator new(std::size_t size) {
  if (fail_allocation) { fail_allocation = false; throw std::bad_alloc(); }
  if (auto *value = std::malloc(size)) { return value; }
  throw std::bad_alloc();
}
void operator delete(void *value) noexcept { std::free(value); }
void operator delete(void *value, std::size_t) noexcept { std::free(value); }
#define CheckCachedLogLevel(level) true
#define ATT_FINAL_TILING_LOGI(op, fmt, value) (void)0
#define OP_NAME "test"
)" + att::final_tiling::GenFinalTilingLogHelpers(false) +
                       R"(
int main() {
  FinalTilingObservationContext outer;
  FinalTilingObservationCallGuard outer_guard(outer, true, 1U);
  FinalTilingObservationSlotGuard outer_slot(0U);
  CommitCurrentFinalTilingSelection({true, 0U, 0U, 0U, 7U, 7, 0});
  for (size_t count : {1UL, 64UL, 4096UL}) {
    FinalTilingObservationContext inner;
    fail_allocation = true;
    try {
      FinalTilingObservationCallGuard inner_guard(inner, true, count);
      if (inner.enabled || IsFinalTilingObservationEnabled()) { return 1; }
    } catch (...) { return 2; }
    if (!IsFinalTilingObservationEnabled() || outer.selections[0].tiling_key != 7U) { return 3; }
    for (bool capture : {false, true}) {
      FinalTilingObservationContext trial;
      FinalTilingObservationCallGuard trial_guard(trial, true, count);
      fail_allocation = true;
      const auto snapshots = capture ? CaptureFinalTilingSelectionSnapshot(0U, {}) : GetFinalTilingSelectionSnapshots();
      if (!snapshots.empty() || trial.enabled) { return 4; }
    }
  }
  return outer.selections[0].tiling_key == 7U ? 0 : 5;
}
)";
  ASSERT_TRUE(CompileAndRunCode(source));
}

TEST_F(TestCodegenTiling, TopnObservationVisitsOnlySelectedResultWithPerGroupFallback) {
  auto fused = GenBasicFusedScheduleResult({}, af::ops::One);
  const auto result = fused.node_idx_to_scheduled_results[0][0];
  fused.node_idx_to_scheduled_results[0].assign(2U, result);
  codegen_func_ = att::GenTilingImplAutoFuseV3;
  std::string source = R"(
#define CheckCachedLogLevel(level) true
#define ATT_FINAL_TILING_LOGI(op, fmt, value) (void)0
#define OP_NAME "test"
)" + att::final_tiling::GenFinalTilingLogHelpers(false) +
                       R"(
int fallback_calls = 0; bool fail_names = false;
struct Data { uint32_t get_tiling_key() const { return 0U; } };
struct AutofuseTilingData { Data graph0_result0_g0_tiling_data, graph0_result1_g0_tiling_data;
  uint32_t get_graph0_tiling_key() const { return 1U; } };
template<uint32_t Result> std::vector<TemplateSemanticInfo> Fallback(Data &) {
  ++fallback_calls; if (fail_names) { throw std::bad_alloc(); }
  TemplateSemanticInfo info; info.result = Result; info.semantic_name = "fallback"; return {info}; }
bool Selected(Data &, const FinalTilingSelectionSnapshot &snapshot, FinalTilingGroupSelection &selection) {
  selection.semantic_name = "selected"; return snapshot.valid; }
namespace optiling {
namespace AscGraph0ScheduleResult0G0 { auto GetTemplateSemanticInfos(Data &data) { return Fallback<0>(data); }
  auto GetFinalTilingSelectionFromSnapshot(Data &d, const FinalTilingSelectionSnapshot &s,
    FinalTilingGroupSelection &v) { return Selected(d, s, v); } }
namespace AscGraph0ScheduleResult1G0 { auto GetTemplateSemanticInfos(Data &data) { return Fallback<1>(data); }
  auto GetFinalTilingSelectionFromSnapshot(Data &d, const FinalTilingSelectionSnapshot &s,
    FinalTilingGroupSelection &v) { return Selected(d, s, v); } }
}
)" + GenTemplateSemanticInfosForInductor(fused) +
                       R"(
int main() {
  AutofuseTilingData data;
  std::vector<FinalTilingSelectionSnapshot> snapshots(2U);
  snapshots[1] = {true, 0U, 1U, 0U, 0U, 0, 0};
  FinalTilingObservationContext context;
  {
    FinalTilingObservationCallGuard disabled(context, false, 2U);
    for (size_t count = 0UL; count < 4096UL; ++count) {
      if (!optiling::GetTemplateSemanticInfos(data, snapshots).empty() || fallback_calls != 0) { return 3; }
    }
  }
  FinalTilingObservationCallGuard enabled(context, true, 2U);
  auto infos = optiling::GetTemplateSemanticInfos(data, snapshots);
  if (infos.size() != 1U || infos[0].result != 1U || infos[0].semantic_name != "selected") { return 1; }
  snapshots[1].valid = false;
  infos = optiling::GetTemplateSemanticInfos(data, snapshots);
  if (infos.size() != 1U || infos[0].result != 1U || infos[0].semantic_name != "fallback") { return 2; }
  fail_names = true;
  if (!optiling::GetTemplateSemanticInfos(data, snapshots).empty() || context.enabled) { return 4; }
  const auto calls = fallback_calls;
  return optiling::GetTemplateSemanticInfos(data, snapshots).empty() && fallback_calls == calls ? 0 : 5;
}
)";
  ASSERT_TRUE(CompileAndRunCode(source));
}

// Task 2: Config truth table — request construction & config semantics

TEST_F(TestCodegenTiling, TopnWrapperMapsEmptyConfigsToInternalNoConfigPath) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_NE(tiling_impl.find("if (input_configs.empty()) {"), std::string::npos);
  EXPECT_NE(tiling_impl.find("request.input_configs = nullptr;"), std::string::npos);
  EXPECT_NE(tiling_impl.find("request.input_configs = &input_configs;"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("normalized_configs"), std::string::npos);
}

TEST_F(TestCodegenTiling, TopnWrapperConstructsRequestAndInvokesGetTiling) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  // Wrapper should construct request/response objects and invoke GetTiling with them.
  EXPECT_NE(tiling_impl.find("GetTilingRequest request;"), std::string::npos);
  EXPECT_NE(tiling_impl.find("GetTilingResponse response;"), std::string::npos);
  EXPECT_NE(tiling_impl.find("GetTopnCandidateSolutions(request, response)"), std::string::npos);
}

TEST_F(TestCodegenTiling, GetTilingOriginalConfigDetectionIncludesInternalPath) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_NE(tiling_impl.find("const bool internal_no_config_path = (request.input_configs == nullptr);"),
            std::string::npos);
  EXPECT_NE(tiling_impl.find("const bool explicit_no_config_path = request.input_configs != nullptr && "),
            std::string::npos);
  EXPECT_NE(tiling_impl.find("const bool original_config_path = internal_no_config_path || explicit_no_config_path;"),
            std::string::npos);
  EXPECT_EQ(tiling_impl.find("is_default_config_request"), std::string::npos);
}

TEST_F(TestCodegenTiling, GetTilingIteratesConfigsInOrder) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  // Multi-config must iterate in input order
  EXPECT_NE(tiling_impl.find("for (const auto *cfg : config_ptrs)"), std::string::npos);
  // Each config feeds PGOSearchTilingKey within the loop
  EXPECT_NE(tiling_impl.find("PGOSearchTilingKey("), std::string::npos);
}

TEST_F(TestCodegenTiling, GetTilingInternalPathOnlyForNullptr) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  // nullptr comparison exists in GetTiling for internal path detection
  EXPECT_NE(tiling_impl.find("request.input_configs == nullptr"), std::string::npos);
  // Internal path keeps search_cfg nullptr so each Result/TilingCase uses its original generated config.
  EXPECT_NE(tiling_impl.find("internal_no_config_path"), std::string::npos);
  EXPECT_NE(tiling_impl.find("config_ptrs.push_back(nullptr)"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("configs.push_back(SearchConfig())"), std::string::npos);
  const auto search_call_pos =
      tiling_impl.find("PGOSearchTilingKey(raw_candidates, cur_search_tiling, -1, &cur_search_tiling, ");
  ASSERT_NE(search_call_pos, std::string::npos);
  const auto search_call_end = tiling_impl.find(");", search_call_pos);
  ASSERT_NE(search_call_end, std::string::npos);
  EXPECT_NE(tiling_impl.substr(search_call_pos, search_call_end - search_call_pos).find("cfg"), std::string::npos);
}

TEST_F(TestCodegenTiling, ExplicitConfigsPassSearchConfigPointerToTopnSearch) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_NE(tiling_impl.find("for (const auto &cfg : configs) { config_ptrs.push_back(&cfg); }"), std::string::npos);
  EXPECT_NE(tiling_impl.find("cfg->ub_threshold"), std::string::npos);
  const auto search_call_pos =
      tiling_impl.find("PGOSearchTilingKey(raw_candidates, cur_search_tiling, -1, &cur_search_tiling, ");
  ASSERT_NE(search_call_pos, std::string::npos);
  const auto search_call_end = tiling_impl.find(");", search_call_pos);
  ASSERT_NE(search_call_end, std::string::npos);
  EXPECT_NE(tiling_impl.substr(search_call_pos, search_call_end - search_call_pos).find("cfg"), std::string::npos);
}

TEST_F(TestCodegenTiling, ParseSearchConfigsParsesExplicitConfigsOnly) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_EQ(tiling_impl.find("if (raws.empty()) {"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("return {SearchConfig()};"), std::string::npos);
}

TEST_F(TestCodegenTiling, ParseSearchConfigShouldNormalizeZeroUbThreshold) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_NE(tiling_impl.find("constexpr double kMinUbThreshold = 0.001;"), std::string::npos);
  EXPECT_NE(tiling_impl.find("if (std::fabs(out.ub_threshold) < 1e-8) { out.ub_threshold = kMinUbThreshold; }"),
            std::string::npos);
}

TEST_F(TestCodegenTiling, TopnCandidatesKeepDefaultFirst) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_NE(tiling_impl.find("Topn selector helpers: default-first"), std::string::npos);
  EXPECT_NE(tiling_impl.find("if (lhs.is_default != rhs.is_default) { return lhs.is_default; }"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("solution.is_default = false;"), std::string::npos);
}

TEST_F(TestCodegenTiling, AppendsDefaultCandidateWhenPgoMissesDefault) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_NE(tiling_impl.find("bool found_default_candidate = false;"), std::string::npos);
  EXPECT_NE(tiling_impl.find("if (!default_repr.empty() && !found_default_candidate) {"), std::string::npos);
  EXPECT_NE(tiling_impl.find("default_solution.tiling_data = default_tiling;"), std::string::npos);
  EXPECT_NE(tiling_impl.find("default_solution.is_default = true;"), std::string::npos);
  EXPECT_NE(tiling_impl.find("if (!found_default_candidate) {"), std::string::npos);
  EXPECT_NE(tiling_impl.find("default topn candidate not found"), std::string::npos);
}

// Task 1 Step 2: multi-group semantic annotations — must match TF PGO real implementation

// Verify multi-group generates graph-level tiling keys and perf aggregation via UpdateCurPerfAndBlockByGroup
TEST_F(TestCodegenTiling, MultiGroupUsesGraphLevelTilingKeysAndPerfAggregation) {
  auto fused_schedule_result = GenMultiGroupFusedScheduleResult();
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  // Multi-group must generate PGOSearchTilingKey with best_perf pointer (not ref)
  EXPECT_NE(tiling_impl.find("PGOSearchTilingKey"), std::string::npos);
  // Must have config iteration for SearchConfig
  EXPECT_NE(tiling_impl.find("for (const auto *cfg : config_ptrs)"), std::string::npos);
  // Multi-group repr must use graph-level tiling keys
  EXPECT_NE(tiling_impl.find("graph0_tiling_key"), std::string::npos);
  // Multi-group perf aggregation must use UpdateCurPerfAndBlockByGroup
  EXPECT_NE(tiling_impl.find("UpdateCurPerfAndBlockByGroup"), std::string::npos);
}

TEST_F(TestCodegenTiling, MultiGroupMustNotUseBasicPGOSearchTilingKeyOverload) {
  auto fused_schedule_result = GenMultiGroupFusedScheduleResult();
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  // Must not generate basic PGOSearchTilingKey that takes AutofuseTilingData & directly
  EXPECT_EQ(tiling_impl.find("PGOSearchTilingKey(raw_candidates, AutofuseTilingData &"), std::string::npos);
  // Must not generate basic PGOSearchTilingKey with search_tiling output at -1 case id
  EXPECT_EQ(tiling_impl.find("PGOSearchTilingKey(raw_candidates, search_tiling, -1, &search_tiling"),
            std::string::npos);
}

TEST_F(TestCodegenTiling, CodegenGenerateForInductorShouldEmitSplitMarkers) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen::CodegenResult result;

  ASSERT_EQ(codegen.GenerateForInductor(fused_schedule_result, result), af::SUCCESS);

  EXPECT_EQ(result.tiling.find("// AUTOFUSE_SPLIT_FILE_BEGIN: TilingHead"), std::string::npos);
  EXPECT_EQ(result.tiling.find("// AUTOFUSE_SPLIT_FILE_END: TilingHead"), std::string::npos);
  EXPECT_NE(result.tiling.find("// AUTOFUSE_SPLIT_FILE_BEGIN: TilingStateHeader"), std::string::npos);
  EXPECT_NE(result.tiling.find("// AUTOFUSE_SPLIT_FILE_BEGIN: TilingLogHeader"), std::string::npos);
  EXPECT_NE(result.tiling.find("// AUTOFUSE_SPLIT_FILE_BEGIN: TilingPgoHeader"), std::string::npos);
  EXPECT_NE(result.tiling.find("// AUTOFUSE_SPLIT_FILE_BEGIN: TilingSolverHeader"), std::string::npos);
  EXPECT_NE(result.tiling.find("// AUTOFUSE_SPLIT_FILE_BEGIN: TilingApiHeader"), std::string::npos);
  EXPECT_EQ(result.tiling.find("// AUTOFUSE_SPLIT_FILE_BEGIN: TilingBaseHeader"), std::string::npos);
  EXPECT_EQ(result.tiling.find("// AUTOFUSE_SPLIT_FILE_BEGIN: TilingEntryHeader"), std::string::npos);
  EXPECT_EQ(result.tiling.find("// AUTOFUSE_SPLIT_FILE_BEGIN: TilingTailHeader"), std::string::npos);
  EXPECT_NE(result.tiling.find("#include \"autofuse_tiling_func_api.h\""), std::string::npos);
  const auto get_marker_content = [&result](const std::string &key) {
    const std::string begin = "// AUTOFUSE_SPLIT_FILE_BEGIN: " + key;
    const std::string end = "// AUTOFUSE_SPLIT_FILE_END: " + key;
    const size_t begin_pos = result.tiling.find(begin);
    EXPECT_NE(begin_pos, std::string::npos);
    const size_t content_pos = result.tiling.find('\n', begin_pos);
    EXPECT_NE(content_pos, std::string::npos);
    const size_t end_pos = result.tiling.find(end, content_pos);
    EXPECT_NE(end_pos, std::string::npos);
    return result.tiling.substr(content_pos + 1U, end_pos - content_pos - 1U);
  };
  const std::string state_header = get_marker_content("TilingStateHeader");
  const std::string solver_header = get_marker_content("TilingSolverHeader");
  const std::string api_header = get_marker_content("TilingApiHeader");
  EXPECT_EQ(state_header.find("#include \"autofuse_tiling_func_"), std::string::npos);
  EXPECT_EQ(solver_header.find("#include \"autofuse_tiling_func_"), std::string::npos);
  EXPECT_EQ(api_header.find("#include \"autofuse_tiling_func_"), std::string::npos);
  EXPECT_NE(result.tiling.find("// AUTOFUSE_SPLIT_FILE_BEGIN: tiling_def_and_tiling_const"), std::string::npos);
  EXPECT_NE(result.tiling.find("extern \"C\" int64_t AutofuseTiling"), std::string::npos);
  EXPECT_NE(result.tiling.find("extern \"C\" int64_t GenerateTopnSolutions"), std::string::npos);
  EXPECT_NE(result.tiling.find("GetTilingDataRepr("), std::string::npos);
}

TEST_F(TestCodegenTiling, InductorEntryUsingSolverHelpersShouldIncludeSolverHeader) {
  auto graph = ascir::ShareGraph::TailBrcTailReduceFusedGraph(3);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused_schedule_result;

  ASSERT_EQ(optimizer.Optimize(graph, fused_schedule_result), af::SUCCESS);
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_NE(tiling_files.find(codegen::kTilingDefAndConstIdentify), tiling_files.end());
  const auto &entry = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  ASSERT_NE(entry.find("Max("), std::string::npos);
  ASSERT_NE(entry.find("Ceiling("), std::string::npos);
  EXPECT_NE(entry.find("#include \"autofuse_tiling_func_solver.h\""), std::string::npos);
}

TEST_F(TestCodegenTiling, GenerateForInductorPgoFalseShouldKeepModeledTopn) {
  ScopedAutofusePgoFlag pgo_flag(false);
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen::CodegenResult result;

  ASSERT_EQ(codegen.GenerateForInductor(fused_schedule_result, result), af::SUCCESS);

  EXPECT_EQ(result.tiling.find("AUTOFUSE_SPLIT_FILE_BEGIN: PgoRunner"), std::string::npos);
  EXPECT_EQ(result.tiling.find("AUTOFUSE_SPLIT_FILE_BEGIN: PgoDeviceSource"), std::string::npos);
  EXPECT_EQ(result.tiling.find("GenerateMeasuredTopnSolutions"), std::string::npos);
  EXPECT_NE(result.tiling.find("Topn selector helpers: default-first"), std::string::npos);
  EXPECT_NE(result.tiling.find("EvaluateModeledPerf(raw_candidate.tiling_data)"), std::string::npos);
  EXPECT_NE(result.tiling.find("ParseSearchConfigs("), std::string::npos);
  const size_t modeled_search = result.tiling.find("static int64_t GetTopnCandidateSolutions");
  const size_t modeled_entry = result.tiling.find("extern \"C\" int64_t GenerateTopnSolutions", modeled_search);
  ASSERT_NE(modeled_search, std::string::npos);
  ASSERT_NE(modeled_entry, std::string::npos);
  const std::string modeled_body = result.tiling.substr(modeled_search, modeled_entry - modeled_search);
  EXPECT_NE(modeled_body.find("const uint32_t available_aiv_num = std::min(limit->aiv_num, g_no_limit_res.aiv_num)"),
            std::string::npos);
  EXPECT_EQ(modeled_body.find("PGOByCoreNumSearchTilingKey"), std::string::npos);
}

TEST_F(TestCodegenTiling, DisabledInductorPgoShouldOverrideAttPgoEnvironment) {
  ScopedAutofusePgoFlag pgo_flag(true);
  codegen_func_ = att::GenTilingImplAutoFuseV3;
  DisableInductorPgo();
  auto fused_schedule_result = GenSingleGroupFusedScheduleResultWithApiTilingField();

  const auto tiling_files = this->GenerateForInductor(fused_schedule_result);

  for (const auto &[key, source] : tiling_files) {
    EXPECT_EQ(source.find("PGOProfileReuseGroup"), std::string::npos) << key;
    EXPECT_EQ(source.find("PGOByCoreNumSearchTilingKey"), std::string::npos) << key;
  }
  EXPECT_EQ(tiling_files.find(codegen::kPgoRunnerIdentify), tiling_files.end());
}

void AssertSchemeAContract(const codegen::CodegenResult &result) {
  const std::string runner = GetSplitContent(result.tiling, "PgoRunner");
  const std::string device_source = GetSplitContent(result.tiling, "PgoDeviceSource");
  ASSERT_FALSE(runner.empty());
  ASSERT_FALSE(device_source.empty());
  EXPECT_EQ(runner.find("kInductorPgoRunnerAbi"), std::string::npos);
  EXPECT_EQ(device_source.find("AUTOFUSE_PGO_DEVICE_SOURCE_ABI"), std::string::npos);
  EXPECT_NE(device_source.find(result.kernel), std::string::npos);
  EXPECT_NE(result.tiling.find("extern \"C\" int64_t GenerateMeasuredTopnSolutions("), std::string::npos);
  EXPECT_NE(result.tiling.find("if (callback_ret == kPgoProfileUnsupported)"), std::string::npos);
  EXPECT_NE(result.tiling.find("return callback_ret;"), std::string::npos);
  EXPECT_EQ(result.tiling.find("kInductorPgoProxyAbi"), std::string::npos);
  EXPECT_NE(result.tiling.find("return RunInductorPgoProxy(input_configs, topn, tiling_datas, workspaces, block_dims,"),
            std::string::npos);
  EXPECT_EQ(result.tiling.find("FallbackToInductorPgoDefaultSolution"), std::string::npos);
  EXPECT_NE(result.tiling.find("FallbackToInductorModeledTopn"), std::string::npos);
  EXPECT_NE(result.tiling.find("PGOByCoreNumSearchTilingKey(measured_tiling_datas"), std::string::npos);
  EXPECT_EQ(result.tiling.find("GetBuiltinTfPgoConfigs()"), std::string::npos);
}

void AssertSchemeAFallback(const std::string &tiling) {
  EXPECT_NE(tiling.find("if (!ResolveInductorPgoArtifacts(artifacts)) {\n"
                        "    return FallbackToInductorModeledTopn(input_configs, topn,"),
            std::string::npos);
  EXPECT_NE(tiling.find("if (!parsed) {\n"
                        "    OP_LOGW(OP_NAME, \"Inductor PGO runner or result parsing failed, "
                        "runner_ret=%d\", runner_ret);\n"
                        "    return FallbackToInductorModeledTopn(input_configs, topn,"),
            std::string::npos);
  EXPECT_NE(tiling.find("mkdtemp(result_dir_template)"), std::string::npos);
  EXPECT_NE(tiling.find("path = std::string(result_dir_template) + \"/result.bin\""), std::string::npos);
  EXPECT_NE(tiling.find("rmdir(PgoParentPath(path).c_str())"), std::string::npos);
  EXPECT_EQ(tiling.find("mkstemp(result_template)"), std::string::npos);
  EXPECT_NE(tiling.find("raw_candidate.best_perf"), std::string::npos);
  EXPECT_NE(tiling.find("PgoConfig::Instance().tensor_args"), std::string::npos);
  EXPECT_NE(tiling.find("PgoConfig::Instance().stream"), std::string::npos);
  EXPECT_NE(tiling.find("kept.modeled_perf > solution.modeled_perf"), std::string::npos);
  EXPECT_NE(tiling.find("single_callback(PgoConfig::Instance().tensor_args"), std::string::npos);
}

void AssertSchemeASelectors(const std::string &tiling) {
  const size_t measured_search = tiling.find("static int64_t GetTopnCandidateSolutions");
  const size_t proxy_start = tiling.find("#include <spawn.h>", measured_search);
  ASSERT_NE(measured_search, std::string::npos);
  ASSERT_NE(proxy_start, std::string::npos);
  const std::string measured_body = tiling.substr(measured_search, proxy_start - measured_search);
  EXPECT_EQ(measured_body.find("ParseSearchConfigs("), std::string::npos);
  EXPECT_EQ(measured_body.find("EvaluateModeledPerf(raw_candidate.tiling_data)"), std::string::npos);
  EXPECT_EQ(measured_body.find("if (lhs.is_default != rhs.is_default)"), std::string::npos);
  const size_t fallback_start = tiling.rfind("namespace inductor_pgo_fallback {");
  const size_t fallback_end = tiling.find("}  // namespace inductor_pgo_fallback", fallback_start);
  ASSERT_NE(fallback_start, std::string::npos);
  ASSERT_NE(fallback_end, std::string::npos);
  const std::string fallback_body = tiling.substr(fallback_start, fallback_end - fallback_start);
  EXPECT_NE(fallback_body.find("ParseSearchConfigs("), std::string::npos);
  EXPECT_NE(fallback_body.find("EvaluateModeledPerf(raw_candidate.tiling_data)"), std::string::npos);
  EXPECT_NE(fallback_body.find("if (lhs.is_default != rhs.is_default)"), std::string::npos);
  const auto append_default_pos =
      tiling.find("response.candidate_solutions.push_back({default_tiling, default_perf, true, default_repr})");
  const auto select_topn_pos = tiling.find("SelectTopnCandidateSolutions(response.candidate_solutions, topn)");
  const auto export_measured_pos = tiling.find("measured_candidates->push_back(");
  const auto truncate_pos = tiling.find("solutions.resize(static_cast<size_t>(topn))", export_measured_pos);
  ASSERT_NE(append_default_pos, std::string::npos);
  ASSERT_NE(select_topn_pos, std::string::npos);
  ASSERT_NE(export_measured_pos, std::string::npos);
  ASSERT_NE(truncate_pos, std::string::npos);
  EXPECT_LT(append_default_pos, select_topn_pos);
  EXPECT_LT(export_measured_pos, truncate_pos);
}

TEST_F(TestCodegenTiling, GenerateForInductorPgoTrueShouldEmitSchemeAContract) {
  ScopedAutofusePgoFlag pgo_flag(true);
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol(64), af::Symbol(128)});
  fused_schedule_result.node_idx_to_scheduled_results[0].resize(1);
  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen::CodegenResult result;

  ASSERT_EQ(codegen.GenerateForInductor(fused_schedule_result, result), af::SUCCESS);

  AssertSchemeAContract(result);
  AssertSchemeAFallback(result.tiling);
  AssertSchemeASelectors(result.tiling);
  const size_t first_unused_callback = result.tiling.find("  (void)prof_callback;\n  (void)prof_batch_callback;");
  ASSERT_NE(first_unused_callback, std::string::npos);
  EXPECT_NE(result.tiling.find("  (void)prof_callback;\n  (void)prof_batch_callback;", first_unused_callback + 1U),
            std::string::npos);
}

TEST_F(TestCodegenTiling, GenerateForInductorPgoTopnShouldPreserveDefaultCandidateWhenTopnGreaterThanOne) {
  ScopedAutofusePgoFlag pgo_flag(true);
  // A zero-size axis marks the fixture as an empty-tensor scene, which falls back to the stub tiling path
  // without ATT codegen, so use a non-zero axis to exercise the ATT generator.
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol(64), af::Symbol(128)}, af::Symbol(1));
  fused_schedule_result.node_idx_to_scheduled_results[0].resize(1);
  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen::CodegenResult result;

  ASSERT_EQ(codegen.GenerateForInductor(fused_schedule_result, result), af::SUCCESS);

  EXPECT_NE(result.tiling.find("if (topn > 1 && solutions.size() > 1U)"), std::string::npos);
  EXPECT_NE(result.tiling.find("const auto default_solution = std::find_if(solutions.begin(), solutions.end()"),
            std::string::npos);
  EXPECT_NE(result.tiling.find("std::rotate(solutions.begin() + 1, default_solution, default_solution + 1);"),
            std::string::npos);
  EXPECT_EQ(result.tiling.find("TORCHINDUCTOR_NPU_EXT_AUTOTUNE_TOPN"), std::string::npos);
  EXPECT_NE(result.tiling.find("GetTilingCore(default_tiling, -1)"), std::string::npos);
  EXPECT_NE(result.tiling.find("bool GetTilingCore(AutofuseTilingData &tiling_data, int32_t tiling_case_id, "
                               "double *perf) {"),
            std::string::npos);
  EXPECT_NE(result.tiling.find("EmitFinalTilingByCase(*tiling"), std::string::npos);
  EXPECT_EQ(result.tiling.find("EmitFinalTiling(*tiling, tiling->get_block_dim(), \"runtime\", \"default\""),
            std::string::npos);
}

TEST_F(TestCodegenTiling, GenerateForInductorWithoutAttKeepsStubTilingEntry) {
  ScopedAutofusePgoFlag pgo_flag(false);
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol(64), af::Symbol(128)});
  const auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_NE(tiling_files.find(codegen::kTilingDefAndConstIdentify), tiling_files.end());
  const auto &source = tiling_files.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_NE(source.find("if (GetTiling(default_tiling, -1))"), std::string::npos);
  EXPECT_EQ(source.find("GetTilingCore(default_tiling, -1)"), std::string::npos);
}

TEST_F(TestCodegenTiling, GenerateForInductorPgoShouldExcludeCandidatesNotFasterThanDefault) {
  ScopedAutofusePgoFlag pgo_flag(true);
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol(64), af::Symbol(128)});
  fused_schedule_result.node_idx_to_scheduled_results[0].resize(1);
  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen::CodegenResult result;

  ASSERT_EQ(codegen.GenerateForInductor(fused_schedule_result, result), af::SUCCESS);

  const auto deduplicate_pos = result.tiling.find("DeduplicateCandidateSolutions(solutions);");
  const auto filter_pos = result.tiling.find("FilterMeasuredCandidatesByDefault(solutions);");
  const auto export_pos = result.tiling.find("measured_candidates->push_back(");
  const auto truncate_pos = result.tiling.find("solutions.resize(static_cast<size_t>(topn))", export_pos);
  ASSERT_NE(deduplicate_pos, std::string::npos);
  ASSERT_NE(export_pos, std::string::npos);
  ASSERT_NE(filter_pos, std::string::npos);
  ASSERT_NE(truncate_pos, std::string::npos);
  EXPECT_LT(deduplicate_pos, filter_pos);
  EXPECT_LT(filter_pos, export_pos);
  EXPECT_LT(export_pos, truncate_pos);
  EXPECT_NE(result.tiling.find("!solution.is_default && !(solution.modeled_perf < default_perf)"), std::string::npos);
}

TEST_F(TestCodegenTiling, GenerateForInductorPgoFailureShouldFallbackToModeledTopn) {
  ScopedAutofusePgoFlag pgo_flag(true);
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol(64), af::Symbol(128)});
  fused_schedule_result.node_idx_to_scheduled_results[0].resize(1);
  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen::CodegenResult result;

  ASSERT_EQ(codegen.GenerateForInductor(fused_schedule_result, result), af::SUCCESS);

  EXPECT_NE(result.tiling.find("namespace inductor_pgo_fallback"), std::string::npos);
  EXPECT_NE(result.tiling.find("static int64_t GenerateModeledFallbackTopnSolutions"), std::string::npos);
  EXPECT_EQ(result.tiling.find("extern \"C\" int64_t GenerateModeledFallbackTopnSolutions"), std::string::npos);
  EXPECT_NE(result.tiling.find("ResLimit modeled_limit = limit"), std::string::npos);
  EXPECT_EQ(result.tiling.find("const_cast<ResLimit *>(&limit)"), std::string::npos);
  EXPECT_NE(result.tiling.find("FallbackToInductorModeledTopn(input_configs, topn"), std::string::npos);
  EXPECT_NE(result.tiling.find("SelectTopnCandidateSolutions(response.candidate_solutions, topn)"), std::string::npos);
}

TEST_F(TestCodegenTiling, GenerateForInductorPgoShouldValidateSidecarsOncePerProxy) {
  ScopedAutofusePgoFlag pgo_flag(true);
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol(64), af::Symbol(128)});
  fused_schedule_result.node_idx_to_scheduled_results[0].resize(1);
  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen::CodegenResult result;

  ASSERT_EQ(codegen.GenerateForInductor(fused_schedule_result, result), af::SUCCESS);

  EXPECT_NE(result.tiling.find("#include <mutex>"), std::string::npos);
  EXPECT_NE(result.tiling.find("std::call_once(validation_once"), std::string::npos);
  EXPECT_NE(result.tiling.find("ResolveInductorPgoArtifactsUncached"), std::string::npos);
}

TEST_F(TestCodegenTiling, GenerateForInductorPgoTrueShouldReuseTfAllCoreSearch) {
  ScopedAutofusePgoFlag pgo_flag(true);
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol(64), af::Symbol(128)});
  fused_schedule_result.node_idx_to_scheduled_results[0].resize(1);
  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen::CodegenResult result;

  ASSERT_EQ(codegen.GenerateForInductor(fused_schedule_result, result), af::SUCCESS);

  const size_t measured_search = result.tiling.find("static int64_t GetTopnCandidateSolutions");
  const size_t measured_entry =
      result.tiling.find("extern \"C\" int64_t GenerateMeasuredTopnSolutions", measured_search);
  ASSERT_NE(measured_search, std::string::npos);
  ASSERT_NE(measured_entry, std::string::npos);
  const std::string measured_body = result.tiling.substr(measured_search, measured_entry - measured_search);
  EXPECT_NE(measured_body.find("const uint32_t available_aiv_num = std::min(limit->aiv_num, g_no_limit_res.aiv_num)"),
            std::string::npos);
  EXPECT_NE(measured_body.find("const uint32_t measured_aiv_num = available_aiv_num"), std::string::npos);
  EXPECT_NE(measured_body.find("optiling::PGOByCoreNumSearchTilingKey(measured_tiling_datas, &cur_search_tiling, "
                               "measured_aiv_num)"),
            std::string::npos);
  EXPECT_NE(measured_body.find("PgoConfig::Instance().pgo_threshold_index"), std::string::npos);
  EXPECT_NE(measured_body.find("PgoConfig::Instance().batch_callback"), std::string::npos);
  EXPECT_EQ(measured_body.find("aiv_num exceeds platform limit"), std::string::npos);
  EXPECT_EQ(measured_body.find("helper_ret = optiling::PGOSearchTilingKey("), std::string::npos);
}

TEST_F(TestCodegenTiling, GenerateForPgoShouldEmitSharedMeasuredCandidateNormalization) {
  ScopedAutofusePgoFlag pgo_flag(true);
  codegen::TilingLib tiling_lib("", "");
  const auto tf_fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  const auto tf_files =
      tiling_lib.Generate(tf_fused_schedule_result, {{"s0", "1"}, {"s1", "64"}}, "/tmp/autofuse_pgo_stage2", "0");
  const auto &tf_source = tf_files.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_EQ(tf_source.find("struct PgoMeasuredSearchRequest"), std::string::npos);
  EXPECT_EQ(tf_source.find("struct PgoMeasuredSearchResult"), std::string::npos);
  EXPECT_NE(tf_source.find("NormalizePgoMeasuredCandidates"), std::string::npos);
  EXPECT_NE(tf_source.find("std::vector<AutofuseTilingDataPerf> raw_search_candidates;"), std::string::npos);
  EXPECT_NE(tf_source.find("raw_search_candidates.push_back(tiling_data_perf);"), std::string::npos);
  EXPECT_NE(tf_source.find("NormalizePgoMeasuredCandidates(std::move(raw_search_candidates))"), std::string::npos);
  EXPECT_EQ(tf_source.find("NormalizePgoMeasuredCandidates(std::move(tiling_data_perf_list))"), std::string::npos);

  const auto inductor_fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol(64), af::Symbol(128)});
  const auto inductor_files = tiling_lib.GenerateForInductor(inductor_fused_schedule_result);
  const auto &inductor_source = inductor_files.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_EQ(inductor_source.find("struct PgoMeasuredSearchRequest"), std::string::npos);
  EXPECT_EQ(inductor_source.find("struct PgoMeasuredSearchResult"), std::string::npos);
  EXPECT_NE(inductor_source.find("NormalizePgoMeasuredCandidates"), std::string::npos);
  EXPECT_NE(inductor_source.find("NormalizePgoMeasuredCandidates(std::move("), std::string::npos);
}

TEST_F(TestCodegenTiling, GenerateForInductorWithoutPgoShouldNotEmitSharedMeasuredSearchModel) {
  ScopedAutofusePgoFlag pgo_flag(false);
  codegen::TilingLib tiling_lib("", "");
  const auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol(64), af::Symbol(128)});
  const auto files = tiling_lib.GenerateForInductor(fused_schedule_result);
  const auto &source = files.at(codegen::kTilingDefAndConstIdentify);
  EXPECT_EQ(source.find("struct PgoMeasuredSearchRequest"), std::string::npos);
  EXPECT_EQ(source.find("NormalizePgoMeasuredCandidates"), std::string::npos);
}

void AssertDlopenRunnerLoading(const std::string &runner) {
  EXPECT_NE(runner.find("int main(int argc, char *argv[])"), std::string::npos);
  EXPECT_NE(runner.find("\"GenerateMeasuredTopnSolutions\""), std::string::npos);
  EXPECT_NE(runner.find("RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE"), std::string::npos);
  EXPECT_NE(runner.find("std::atomic<uint32_t> g_pgo_active_calls"), std::string::npos);
  EXPECT_NE(runner.find("std::atomic<bool> g_pgo_closing"), std::string::npos);
  EXPECT_NE(runner.find("class PgoDsoCallGuard"), std::string::npos);
  EXPECT_NE(runner.find("g_pgo_closing.store(true"), std::string::npos);
  EXPECT_NE(runner.find("g_pgo_dso_cv.wait"), std::string::npos);
  EXPECT_NE(runner.find("dlsym(g_pgo_tiling_handle, name)"), std::string::npos);
  EXPECT_NE(runner.find("LoadInductorPgoSymbol(generate_measured_topn_solutions_fn, "
                        "\"GenerateMeasuredTopnSolutions\")"),
            std::string::npos);
  EXPECT_NE(runner.find("LoadInductorPgoSymbol(set_topn_pgo_context_fn, \"SetTopnPgoContext\")"), std::string::npos);
  EXPECT_NE(runner.find("LoadInductorPgoSymbol(clear_topn_pgo_context_fn, \"ClearTopnPgoContext\")"),
            std::string::npos);
  EXPECT_NE(runner.find("LoadInductorPgoSymbol(get_tiling_data_repr_fn, \"GetTilingDataRepr\")"), std::string::npos);
  EXPECT_NE(runner.find("set_topn_pgo_context_fn(&g_pgo_tensor_args, g_stream"), std::string::npos);
  EXPECT_NE(runner.find("aclrtBinaryLoadFromFile(g_kernel_o_file.c_str()"), std::string::npos);
  EXPECT_NE(runner.find("aclrtBinaryUnLoad"), std::string::npos);
}

void AssertCommonPgoLoaderLifetime(const std::string &source) {
  EXPECT_NE(source.find("RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE"), std::string::npos);
  EXPECT_NE(source.find("std::atomic<uint32_t> active_calls"), std::string::npos);
  EXPECT_NE(source.find("std::atomic<bool> closing"), std::string::npos);
  EXPECT_NE(source.find("class PgoDsoCallGuard"), std::string::npos);
  EXPECT_NE(source.find("closing.store(true"), std::string::npos);
  EXPECT_NE(source.find("dso_cv.wait"), std::string::npos);
  EXPECT_NE(source.find("if (!kPgoDlopenNodelete)"), std::string::npos);
  const auto close_pos = source.find("if (!kPgoDlopenNodelete) { dlclose(handle); }");
  const auto wait_pos = source.find("dso_cv.wait");
  ASSERT_NE(close_pos, std::string::npos);
  ASSERT_NE(wait_pos, std::string::npos);
  EXPECT_LT(wait_pos, close_pos);
  EXPECT_NE(source.find("PgoDsoCallGuard dso_guard;"), std::string::npos);
}

TEST_F(TestCodegenTiling, GenerateForPgoShouldProtectDsoLifetime) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol(64), af::Symbol(128)});
  AssertCommonPgoLoaderLifetime(GenerateForPgo(fused_schedule_result, "/tmp"));
}

TEST_F(TestCodegenTiling, GenerateForPgoShouldLoadFindBestTilingKeyOnlyWhenSupported) {
  auto supported_result = this->GenBasicFusedScheduleResult({af::Symbol(64), af::Symbol(128)});
  auto unsupported_result = supported_result;
  unsupported_result.node_idx_to_scheduled_results[0][0].enable_group_parallel = true;

  const auto supported_source = GenerateForPgo(supported_result, "/tmp");
  EXPECT_NE(supported_source.find("GetFunc(\"FindBestTilingKey\")"), std::string::npos);
  EXPECT_NE(supported_source.find("find_best_tiling_key_fn(*tiling_data)"), std::string::npos);

  const auto unsupported_source = GenerateForPgo(unsupported_result, "/tmp");
  EXPECT_EQ(unsupported_source.find("GetFunc(\"FindBestTilingKey\")"), std::string::npos);
  EXPECT_EQ(unsupported_source.find("find_best_tiling_key_fn(*tiling_data)"), std::string::npos);
  EXPECT_NE(unsupported_source.find("GetFunc(\"GetTilingKeyCount\")"), std::string::npos);
}

void AssertDlopenRunnerProfiling(const std::string &runner) {
  EXPECT_NE(runner.find("PGOGetProfilingBatch"), std::string::npos);
  EXPECT_EQ(runner.find("mspti.h"), std::string::npos);
  EXPECT_EQ(runner.find("msptiActivity"), std::string::npos);
  EXPECT_EQ(runner.find("msptiSubscriberHandle"), std::string::npos);
  // Timing no longer uses ACL events; the collector owns measurement via
  // profapi/STARS, so no event creation is generated.
  EXPECT_EQ(runner.find("aclrtCreateEvent"), std::string::npos);
  EXPECT_NE(runner.find("AutofusePgoCollectorBeginCandidate"), std::string::npos);
  EXPECT_NE(runner.find("get_tiling_data_repr_fn(tiling_data)"), std::string::npos);
  EXPECT_NE(runner.find("AutofusePgoCollectorFinalizeBatch"), std::string::npos);
  EXPECT_NE(runner.find("AutofusePgoCollectorGetCandidateDurationNs"), std::string::npos);
  EXPECT_EQ(runner.find("AutofusePgoCollectorGetDurationNs"), std::string::npos);
  EXPECT_EQ(runner.find("aclrtRecordEvent"), std::string::npos);
  EXPECT_EQ(runner.find("aclrtEventGetTimestamp"), std::string::npos);
  EXPECT_NE(runner.find("aclrtSynchronizeStream"), std::string::npos);
  EXPECT_NE(runner.find("kPgoMeasureLoop = 20U"), std::string::npos);
  const auto retry_definition = runner.find("kPgoRetryCount = 5");
  const auto retry_loop = runner.find("for (int i = 0; i < kPgoRetryCount; i++)");
  ASSERT_NE(retry_definition, std::string::npos);
  ASSERT_NE(retry_loop, std::string::npos);
  EXPECT_LT(retry_definition, retry_loop);
  EXPECT_NE(runner.find("kPgoBatchGroupSize = 1000U"), std::string::npos);
  EXPECT_NE(runner.find("DestroyAclPgoEvents"), std::string::npos);
  EXPECT_NE(runner.find("AutofusePgoCollectorRecordLaunch"), std::string::npos);
  EXPECT_NE(runner.find("AutofusePgoCollectorFinalizeBatch"), std::string::npos);
  EXPECT_NE(runner.find("AutofusePgoCollectorGetCandidateDurationNs"), std::string::npos);
  EXPECT_NE(runner.find("kPgoProfileUnsupported = -13"), std::string::npos);
  const auto unsupported_warning = runner.find("PGO profiling unsupported on this device; fallback to non-PGO path");
  const auto unsupported_branch = runner.find("if (init_status == kPgoProfileUnsupported)");
  ASSERT_NE(unsupported_warning, std::string::npos);
  ASSERT_NE(unsupported_branch, std::string::npos);
  EXPECT_LT(unsupported_branch, unsupported_warning);
  EXPECT_NE(runner.find("result = kPgoProfileUnsupported;"), std::string::npos);
  EXPECT_NE(runner.find("DLOGI(\"InitAclPgoEvents failed, status: %d\", status)"), std::string::npos);
  EXPECT_NE(runner.find("DLOGI(\"PgoBeginCandidate failed, status: %d\", begin_status)"), std::string::npos);
  EXPECT_NE(runner.find("DLOGI(\"AutofusePgoCollectorRecordLaunch failed, status: %d\", record_status)"),
            std::string::npos);
  EXPECT_NE(runner.find("DLOGI(\"WrapperOnlyLaunch failed, ERROR: %d\", launch_status)"), std::string::npos);
  EXPECT_NE(runner.find("DLOGI(\"AutofusePgoCollectorEnd failed, status: %d\", end_status)"), std::string::npos);
  EXPECT_NE(runner.find("DLOGI(\"AutofusePgoCollectorFinalizeBatch failed, status: %d\", finalize_status)"),
            std::string::npos);
  EXPECT_EQ(runner.find("AutofusePgoCollectorGetDurationNs"), std::string::npos);
  const auto batch_loop = runner.find("while (it != profiles->end())");
  const auto batch_init = runner.find("auto init_status = InitAclPgoEvents();");
  const auto batch_process = runner.find("result = ProfilingBatchProcess(workspace_size, it, end_it);");
  const auto batch_finalize = runner.find("AutofusePgoCollectorFinalizeBatch(g_pgo_collector)");
  ASSERT_NE(batch_loop, std::string::npos);
  ASSERT_NE(batch_init, std::string::npos);
  ASSERT_NE(batch_process, std::string::npos);
  ASSERT_NE(batch_finalize, std::string::npos);
  EXPECT_LT(batch_loop, batch_init);
  EXPECT_LT(batch_init, batch_process);
  // GenPgoBatchProcess (which contains the first FinalizeBatch call) is emitted
  // before GenPgoGetProfilingBatch, so finalize text precedes the batch loop.
  EXPECT_GT(batch_process, batch_finalize);
  EXPECT_EQ(runner.find("aclrtEventElapsedTime"), std::string::npos);
  EXPECT_EQ(runner.find("kPgoMinMeasureRepeats"), std::string::npos);
  EXPECT_NE(runner.find("for (uint64_t launch_count = 0U; launch_count < kPgoMeasureLoop; ++launch_count)"),
            std::string::npos);
  EXPECT_EQ(runner.find("static_cast<double>(launch_count)"), std::string::npos);
}

void AssertDlopenRunnerProtocol(const std::string &runner) {
  EXPECT_NE(runner.find("AUTOFUSE_PGO_TOPN_V2"), std::string::npos);
  EXPECT_NE(runner.find("constexpr size_t kPgoTopnMagicSize = 20U"), std::string::npos);
  EXPECT_NE(runner.find("constexpr uint32_t kPgoTopnProtocolVersion = 2U"), std::string::npos);
  EXPECT_NE(runner.find("sizeof(AutofuseTilingData)"), std::string::npos);
  EXPECT_NE(runner.find("std::is_trivially_copyable<AutofuseTilingData>::value"), std::string::npos);
  EXPECT_NE(runner.find("uint64_t workspace, int64_t block_dim"), std::string::npos);
  EXPECT_NE(runner.find("workspace > static_cast<uint64_t>(INT64_MAX)"), std::string::npos);
  EXPECT_NE(runner.find("WritePgoValue(out, workspace)"), std::string::npos);
  EXPECT_NE(runner.find("WritePgoValue(out, tiling_hash)"), std::string::npos);
  EXPECT_NE(runner.find("void PgoSaveTilingKey(const AutofuseTilingData &tiling_data, double best_perf"),
            std::string::npos);
  EXPECT_NE(runner.find("int WritePgoSearchResult(const InductorPgoRunnerArgs &args"), std::string::npos);
  EXPECT_NE(runner.find("std::string(PGO_GRAPH_NAME) + \"_search.txt\""), std::string::npos);
  EXPECT_NE(runner.find("std::vector<AutofuseTilingDataPerf> measured_candidates"), std::string::npos);
  EXPECT_NE(runner.find("measured_candidates) == 0"), std::string::npos);
  EXPECT_EQ(runner.find("GetWorkspaceSize(candidate.tiling_data)"), std::string::npos);
  EXPECT_NE(runner.find("using InductorPgoProfilingCallback = long int (*)("), std::string::npos);
  EXPECT_NE(runner.find("using InductorPgoProfilingBatchCallback = long int (*)("), std::string::npos);
  const auto write_topn_pos = runner.find("WritePgoTopnResult(args.result_file");
  const auto write_search_pos = runner.find("WritePgoSearchResult(args, measured_candidates)");
  ASSERT_NE(write_topn_pos, std::string::npos);
  ASSERT_NE(write_search_pos, std::string::npos);
  EXPECT_LT(write_topn_pos, write_search_pos);
  const auto run_end_pos = runner.find("\n}\n\nint main", write_search_pos);
  ASSERT_NE(run_end_pos, std::string::npos);
  const auto write_search_body = runner.substr(write_search_pos, run_end_pos - write_search_pos);
  EXPECT_NE(write_search_body.find("DLOGW(\"Write PGO search result failed\")"), std::string::npos);
  EXPECT_EQ(write_search_body.find("return FAILED"), std::string::npos);
}

void AssertDlopenRunnerLaunch(const std::string &runner, const std::string &kernel_name) {
  EXPECT_NE(runner.find("struct ResLimit {"), std::string::npos);
  EXPECT_NE(runner.find("constexpr char kInductorPgoKernelName[] = \"" + kernel_name + "\";"), std::string::npos);
  EXPECT_NE(runner.find("aclrtBinaryGetFunction(g_pgo_bin_handle, kInductorPgoKernelName"), std::string::npos);
  EXPECT_NE(runner.find("AutofuseTilingData tiling_data;"), std::string::npos);
  EXPECT_NE(runner.find("g_launch_params.aiv_args.tiling_data = tiling_data;"), std::string::npos);
  EXPECT_NE(runner.find("const auto update_status = UpdateLaunchParam(it->tiling_data);"), std::string::npos);
  EXPECT_NE(runner.find("const auto update_status = UpdateLaunchParam(*tiling_data);"), std::string::npos);
  EXPECT_NE(runner.find("WrapperOnlyLaunch(uint64_t workspace_size, AutofuseTilingData *tiling_data) {\n"
                        "  (void)workspace_size;"),
            std::string::npos);
  EXPECT_NE(runner.find("PGOGetProfilingBatch(PgoTensorArgs *tensor_args, void* stream, uint64_t workspace_size, "
                        "std::vector<AutofuseTilingDataPerf> *profiles) {\n"
                        "  (void)tensor_args;\n  (void)stream;"),
            std::string::npos);
  EXPECT_NE(runner.find("PGOGetProfiling(PgoTensorArgs *tensor_args, void *stream, uint64_t workspace_size, "
                        "AutofuseTilingData *tiling_data, double *outCostTime) {\n"
                        "  (void)tensor_args;\n  (void)stream;"),
            std::string::npos);
  EXPECT_NE(runner.find("tiling_key < 0 || static_cast<uint64_t>(tiling_key) >= tiling_key_count"), std::string::npos);
  EXPECT_NE(runner.find("clear_topn_pgo_context_fn()"), std::string::npos);
  EXPECT_NE(runner.find("g_workspace = nullptr"), std::string::npos);
  EXPECT_EQ(runner.find("g_kernel_name + \"_\""), std::string::npos);
  EXPECT_EQ(runner.find("g_launch_params.aiv_args.tiling_addr"), std::string::npos);
  EXPECT_EQ(runner.find("g_tiling_device_addr"), std::string::npos);
  EXPECT_NE(runner.find("args.tiling_file = argv[5]"), std::string::npos);
  EXPECT_NE(runner.find("args.kernel_file = argv[6]"), std::string::npos);
  EXPECT_NE(runner.find("args.result_file = argv[7]"), std::string::npos);
  EXPECT_EQ(runner.find("static_pgo("), std::string::npos);
}

TEST_F(TestCodegenTiling, GenerateForInductorPgoTrueShouldGenerateDlopenRunner) {
  ScopedAutofusePgoFlag pgo_flag(true);
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol(64), af::Symbol(128)});
  fused_schedule_result.node_idx_to_scheduled_results[0].resize(1);
  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen::CodegenResult result;

  ASSERT_EQ(codegen.GenerateForInductor(fused_schedule_result, result), af::SUCCESS);
  const std::string runner = GetSplitContent(result.tiling, "PgoRunner");
  const std::string kernel_name =
      ascgen_utils::CamelToLowerSneak(ascgen_utils::GenValidName(fused_schedule_result.fused_graph_name.GetString()));

  ASSERT_FALSE(runner.empty());
  AssertDlopenRunnerLoading(runner);
  AssertDlopenRunnerProfiling(runner);
  AssertDlopenRunnerProtocol(runner);
  AssertDlopenRunnerLaunch(runner, kernel_name);
  EXPECT_NE(result.tiling.find("extern \"C\" int64_t FindBestTilingKey"), std::string::npos);
}

TEST_F(TestCodegenTiling, GenerateForInductorPgoTopnShouldGuardUint64WorkspaceExport) {
  ScopedAutofusePgoFlag pgo_flag(true);
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol(64), af::Symbol(128)});
  fused_schedule_result.node_idx_to_scheduled_results[0].resize(1);
  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen::CodegenResult result;

  ASSERT_EQ(codegen.GenerateForInductor(fused_schedule_result, result), af::SUCCESS);

  EXPECT_NE(result.tiling.find("uint64_t workspace_size = 0U;"), std::string::npos);
  EXPECT_NE(result.tiling.find("if (!GetWorkspaceSize(tiling_data, candidate_workspace_size))"), std::string::npos);
  EXPECT_NE(result.tiling.find("uint64_t default_workspace = 0U;"), std::string::npos);
  EXPECT_NE(result.tiling.find("if (!GetWorkspaceSize(default_tiling, default_workspace))"), std::string::npos);
  EXPECT_NE(result.tiling.find("std::vector<int64_t> export_workspaces;"), std::string::npos);
  EXPECT_NE(result.tiling.find("workspace_size > static_cast<uint64_t>(INT64_MAX)"), std::string::npos);
  EXPECT_NE(result.tiling.find("workspaces.swap(export_workspaces);"), std::string::npos);
  EXPECT_EQ(result.tiling.find("    workspaces.push_back("), std::string::npos);
  EXPECT_EQ(result.tiling.find("static_cast<int64_t>(GetWorkspaceSize(sol.tiling_data))"), std::string::npos);
}

TEST_F(TestCodegenTiling, GenerateForInductorPgoTrueShouldGenerateValidatedSpawnProxy) {
  ScopedAutofusePgoFlag pgo_flag(true);
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol(64), af::Symbol(128)});
  fused_schedule_result.node_idx_to_scheduled_results[0].resize(1);
  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen::CodegenResult result;

  ASSERT_EQ(codegen.GenerateForInductor(fused_schedule_result, result), af::SUCCESS);

  EXPECT_NE(result.tiling.find("AUTOFUSE_PGO_GENERATION"), std::string::npos);
  EXPECT_NE(result.tiling.find("bundle_schema_version"), std::string::npos);
  EXPECT_NE(result.tiling.find("result_protocol_version"), std::string::npos);
  EXPECT_NE(result.tiling.find("profiling_backend"), std::string::npos);
  EXPECT_NE(result.tiling.find("requires_mspti"), std::string::npos);
  EXPECT_NE(result.tiling.find("profiling_backend == \"mspti_equivalent\""), std::string::npos);
  EXPECT_NE(result.tiling.find("kPgoCacheRootName"), std::string::npos);
  EXPECT_EQ(result.tiling.find("runner_abi"), std::string::npos);
  EXPECT_EQ(result.tiling.find("proxy_abi"), std::string::npos);
  EXPECT_EQ(result.tiling.find("device_source_abi"), std::string::npos);
  EXPECT_NE(result.tiling.find("aicore_binary_elf_v1"), std::string::npos);
  EXPECT_NE(result.tiling.find("ValidateInductorPgoManifest"), std::string::npos);
  EXPECT_NE(result.tiling.find("ComputeFileSha256"), std::string::npos);
  EXPECT_NE(result.tiling.find("dladdr(reinterpret_cast<const void *>(&ResolveInductorPgoArtifactsUncached)"),
            std::string::npos);
  EXPECT_NE(result.tiling.find("artifacts.tiling_so = artifacts.generation_dir + \"/\" + base"), std::string::npos);
  EXPECT_EQ(result.tiling.find("artifacts.tiling_so = real_path"), std::string::npos);
  EXPECT_NE(result.tiling.find("aclrtGetDevice(&device_id)"), std::string::npos);
  EXPECT_NE(result.tiling.find("posix_spawn(&pid"), std::string::npos);
  EXPECT_NE(result.tiling.find("waitpid(pid, &status, WNOHANG)"), std::string::npos);
  EXPECT_NE(result.tiling.find("kill(pid, SIGKILL)"), std::string::npos);
  EXPECT_NE(result.tiling.find("artifacts.ld_preload"), std::string::npos);
  EXPECT_NE(result.tiling.find("BuildInductorPgoRunnerEnv"), std::string::npos);
  EXPECT_NE(result.tiling.find("\"LD_PRELOAD=\" + ld_preload"), std::string::npos);
  EXPECT_NE(result.tiling.find("envp.data()"), std::string::npos);
  EXPECT_NE(result.tiling.find("AUTOFUSE_PGO_RUNNER_TIMEOUT_SECONDS"), std::string::npos);
  EXPECT_NE(result.tiling.find("constexpr int64_t kMaxPgoTopn = 1024"), std::string::npos);
  EXPECT_NE(result.tiling.find("ParseInductorPgoResult"), std::string::npos);
  EXPECT_NE(result.tiling.find("GetTilingDataRepr(&tiling_data) != repr"), std::string::npos);
  EXPECT_EQ(result.tiling.find("AUTOFUSE_PGO_TOPN_V1"), std::string::npos);
  EXPECT_NE(result.tiling.find("AUTOFUSE_PGO_TOPN_V2"), std::string::npos);
  EXPECT_NE(result.tiling.find("uint64_t workspace = 0;"), std::string::npos);
  EXPECT_NE(result.tiling.find("workspace > static_cast<uint64_t>(INT64_MAX)"), std::string::npos);
  EXPECT_NE(result.tiling.find("std::vector<uint64_t> parsed_workspaces;"), std::string::npos);
  const auto version_check_pos = result.tiling.find("version != kPgoTopnProtocolVersion");
  const auto record_loop_pos = result.tiling.find("for (uint32_t i = 0; i < count; ++i)");
  ASSERT_NE(version_check_pos, std::string::npos);
  ASSERT_NE(record_loop_pos, std::string::npos);
  EXPECT_LT(version_check_pos, record_loop_pos);
  const size_t measured_search = result.tiling.find("static int64_t GetTopnCandidateSolutions");
  const size_t proxy_start = result.tiling.find("#include <spawn.h>", measured_search);
  ASSERT_NE(measured_search, std::string::npos);
  ASSERT_NE(proxy_start, std::string::npos);
  EXPECT_EQ(result.tiling.substr(measured_search, proxy_start - measured_search)
                .find("EvaluateModeledPerf(raw_candidate.tiling_data)"),
            std::string::npos);
}

TEST_F(TestCodegenTiling, GenerateForInductorPgoTrueShouldExportPrivateContextAbi) {
  ScopedAutofusePgoFlag pgo_flag(true);
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol(64), af::Symbol(128)});
  fused_schedule_result.node_idx_to_scheduled_results[0].resize(1);
  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen::CodegenResult result;

  ASSERT_EQ(codegen.GenerateForInductor(fused_schedule_result, result), af::SUCCESS);

  EXPECT_NE(result.tiling.find("extern \"C\" int64_t SetTopnPgoContext("), std::string::npos);
  EXPECT_NE(result.tiling.find("extern \"C\" void ClearTopnPgoContext()"), std::string::npos);
  EXPECT_NE(result.tiling.find("PgoConfig::Instance().single_callback = single_callback"), std::string::npos);
  EXPECT_NE(result.tiling.find("PgoConfig::Instance().batch_callback = batch_callback"), std::string::npos);
}

TEST_F(TestCodegenTiling, GenerateForInductorPgoTrueShouldRejectDynamicShape) {
  ScopedAutofusePgoFlag pgo_flag(true);
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen::CodegenResult result;

  EXPECT_EQ(codegen.GenerateForInductor(fused_schedule_result, result), af::FAILED);
  EXPECT_EQ(result.tiling.find("AUTOFUSE_SPLIT_FILE_BEGIN: PgoRunner"), std::string::npos);
}

TEST_F(TestCodegenTiling, GenerateForInductorPgoTrueShouldAcceptStaticMultiGroup) {
  ScopedAutofusePgoFlag pgo_flag(true);
  auto graph = ascir::ShareGraph::TailBrcTailReduceFusedGraph(3);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused_schedule_result;
  ASSERT_EQ(optimizer.Optimize(graph, fused_schedule_result), af::SUCCESS);
  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen::CodegenResult result;

  ASSERT_FALSE(ascgen_utils::IsSingleGroup(fused_schedule_result));
  ASSERT_TRUE(ascgen_utils::IsStaticSchedResult(fused_schedule_result));
  ASSERT_TRUE(ascgen_utils::CanUseTilingKey(fused_schedule_result));
  ASSERT_EQ(codegen.GenerateForInductor(fused_schedule_result, result), af::SUCCESS);
  EXPECT_FALSE(GetSplitContent(result.tiling, "PgoRunner").empty());
  EXPECT_FALSE(GetSplitContent(result.tiling, "PgoDeviceSource").empty());
  EXPECT_NE(result.tiling.find("graph0_tiling_key"), std::string::npos);
}

TEST_F(TestCodegenTiling, GenerateForInductorPgoTrueShouldAcceptReduceRCore) {
  ScopedAutofusePgoFlag pgo_flag(true);
  auto graph = ascir::ShareGraph::TailBrcTailReduceFusedGraph(3);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused_schedule_result;
  ASSERT_EQ(optimizer.Optimize(graph, fused_schedule_result), af::SUCCESS);
  ASSERT_TRUE(ascgen_utils::IsStaticSchedResult(fused_schedule_result));
  ASSERT_FALSE(fused_schedule_result.workspace_nodes.empty());
  ASSERT_TRUE(ascgen_utils::CanUseTilingKey(fused_schedule_result));

  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen::CodegenResult result;
  ASSERT_EQ(codegen.GenerateForInductor(fused_schedule_result, result), af::SUCCESS);
  EXPECT_FALSE(GetSplitContent(result.tiling, "PgoRunner").empty());
  EXPECT_FALSE(GetSplitContent(result.tiling, "PgoDeviceSource").empty());
  EXPECT_NE(result.kernel.find("SyncAll();"), std::string::npos);
}

TEST_F(TestCodegenTiling, GenerateForInductorPgoTrueShouldAcceptGroupParallelWithoutTilingKeyMapping) {
  ScopedAutofusePgoFlag pgo_flag(true);
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol(64), af::Symbol(128)});
  fused_schedule_result.node_idx_to_scheduled_results[0].resize(1);
  fused_schedule_result.node_idx_to_scheduled_results[0][0].enable_group_parallel = true;
  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen::CodegenResult result;

  ASSERT_FALSE(ascgen_utils::CanUseTilingKey(fused_schedule_result));
  ASSERT_EQ(codegen.GenerateForInductor(fused_schedule_result, result), af::SUCCESS);
  EXPECT_NE(result.tiling.find("AUTOFUSE_SPLIT_FILE_BEGIN: PgoRunner"), std::string::npos);
  EXPECT_NE(result.tiling.find("AUTOFUSE_SPLIT_FILE_BEGIN: PgoDeviceSource"), std::string::npos);
  EXPECT_NE(result.tiling.find("typedef int64_t (*FindBestTilingKeyType)"), std::string::npos);
  EXPECT_NE(result.tiling.find("std::fill(func_handles.begin(), func_handles.end(), func_handle)"), std::string::npos);
}

TEST_F(TestCodegenTiling, GenerateForInductorPgoTrueShouldRejectCvFusion) {
  ScopedAutofusePgoFlag pgo_flag(true);
  auto graph = ascir::ShareGraph::LoadMatmulElewiseBrcFusedGraph();
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused_schedule_result;
  ASSERT_EQ(optimizer.Optimize(graph, fused_schedule_result), af::SUCCESS);
  ASSERT_TRUE(ascgen_utils::IsCubeFusedScheduled(fused_schedule_result));
  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen::CodegenResult result;

  EXPECT_EQ(codegen.GenerateForInductor(fused_schedule_result, result), af::FAILED);
  EXPECT_EQ(result.tiling.find("AUTOFUSE_SPLIT_FILE_BEGIN: PgoRunner"), std::string::npos);
}

TEST_F(TestCodegenTiling, SingleGroupEvaluateModeledPerfShouldUsePublicGetPerf) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")}, af::ops::One);
  fused_schedule_result.node_idx_to_scheduled_results[0].resize(1);
  ASSERT_TRUE(ascgen_utils::IsSingleGroup(fused_schedule_result));
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);
  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  const auto &tiling_impl = tiling_files.at(codegen::kTilingDefAndConstIdentify);

  const size_t perf_func_pos = tiling_impl.find("static double EvaluateModeledPerf");
  ASSERT_NE(perf_func_pos, std::string::npos);
  const size_t topn_func_pos = tiling_impl.find("extern \"C\" int64_t GenerateTopnSolutions", perf_func_pos);
  ASSERT_NE(topn_func_pos, std::string::npos);
  const std::string perf_func = tiling_impl.substr(perf_func_pos, topn_func_pos - perf_func_pos);
  EXPECT_NE(perf_func.find("return optiling::GetPerf(tmp);"), std::string::npos);
  EXPECT_EQ(perf_func.find("TilingCaseImplPtr impl = GetTilingImplPtr"), std::string::npos);
  EXPECT_EQ(tiling_impl.find("GetModeledPerfForTesting"), std::string::npos);
}

TEST_F(TestCodegenTiling, CodegenGenerateForInductorCvFusionShouldKeepCubeWrapperHeaderInTilingHead) {
  auto graph = ascir::ShareGraph::LoadMatmulElewiseBrcFusedGraph();
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused_schedule_result;
  ASSERT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);
  ASSERT_TRUE(ascgen_utils::IsCubeFusedScheduled(fused_schedule_result));
  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen::CodegenResult result;

  ASSERT_EQ(codegen.GenerateForInductor(fused_schedule_result, result), af::SUCCESS);

  EXPECT_NE(result.tiling.find("// AUTOFUSE_SPLIT_FILE_BEGIN: ACubeKernelTilingWrapperHpp"), std::string::npos);
  EXPECT_NE(result.tiling.find("class CubeKernelTilingWrapper"), std::string::npos);
  EXPECT_EQ(result.tiling.find("// AUTOFUSE_SPLIT_FILE_BEGIN: TilingDataLog"), std::string::npos);
  EXPECT_NE(result.tiling.find("// AUTOFUSE_SPLIT_FILE_BEGIN: BCubeKernelTilingWrapperCpp"), std::string::npos);
  const size_t wrapper_cpp_pos = result.tiling.find("// AUTOFUSE_SPLIT_FILE_BEGIN: BCubeKernelTilingWrapperCpp");
  ASSERT_NE(wrapper_cpp_pos, std::string::npos);
  const size_t wrapper_cpp_end_pos = result.tiling.find("// AUTOFUSE_SPLIT_FILE_END: BCubeKernelTilingWrapperCpp");
  ASSERT_NE(wrapper_cpp_end_pos, std::string::npos);
  const std::string wrapper_cpp = result.tiling.substr(wrapper_cpp_pos, wrapper_cpp_end_pos - wrapper_cpp_pos);
  EXPECT_NE(wrapper_cpp.find("#include \"cube_kernel_tiling_wrapper.h\""), std::string::npos);
  EXPECT_NE(wrapper_cpp.find("#include \"autofuse_tiling_func_log.h\""), std::string::npos);
}

TEST_F(TestCodegenTiling, CodegenGenerateForInductorBatchMatmulShouldEmitBatchDispatch) {
  af::AscGraph graph("batch_matmul_inductor_fuse");
  CreateBatchMatmulElemwiseFusedGraph(graph);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused_schedule_result;
  ASSERT_EQ(optimizer.Optimize(graph, fused_schedule_result), af::SUCCESS);
  ASSERT_TRUE(ascgen_utils::IsCubeFusedScheduled(fused_schedule_result));

  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen::CodegenResult result;
  ASSERT_EQ(codegen.GenerateForInductor(fused_schedule_result, result), af::SUCCESS);

  EXPECT_NE(result.kernel.find("int8_t BATCH_API_LEVEL = (tiling_key >> 0) & 0xF;"), std::string::npos);
  EXPECT_NE(result.kernel.find("int8_t BATCH_ITER_MODEL = (tiling_key >> 8) & 0xF;"), std::string::npos);
  EXPECT_NE(result.kernel.find("int8_t BMODEL = (tiling_key >> 12) & 0xF;"), std::string::npos);
  EXPECT_NE(result.kernel.find("int8_t BATCH_FULL_LOAD = (tiling_key >> 16) & 0xF;"), std::string::npos);
  EXPECT_NE(result.kernel.find("int8_t BATCH_L0C2OUT_MODEL = (tiling_key >> 20) & 0xF;"), std::string::npos);
  EXPECT_NE(result.kernel.find("_DISPATCH_BATCH_MATMUL"), std::string::npos);
  EXPECT_NE(result.kernel.find("BatchMatMulV3TilingData"), std::string::npos);
  EXPECT_EQ(result.kernel.find("int8_t API_LEVEL = (tiling_key >> 0) & 0xF;"), std::string::npos);
}

TEST_F(TestCodegenTiling, CodegenGenerateForInductorCvFusionShouldFallbackToSafetyWhenUbCasesFail) {
  auto graph = ascir::ShareGraph::LoadMatmulElewiseBrcFusedGraph();
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ascir::FusedScheduledResult fused_schedule_result;
  ASSERT_EQ(optimizer.Optimize(graph, fused_schedule_result), af::SUCCESS);
  ASSERT_TRUE(ascgen_utils::IsCubeFusedScheduled(fused_schedule_result));
  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen::CodegenResult result;

  ASSERT_EQ(codegen.GenerateForInductor(fused_schedule_result, result), af::SUCCESS);

  EXPECT_EQ(result.tiling.find("if (!optiling::GetTiling(tiling->tiling_data, 1)) {\n      return -1;"),
            std::string::npos);
  const std::string tiling_entry =
      result.tiling.find("if (!optiling::GetTilingCore(tiling->tiling_data, 1)) {") != std::string::npos
          ? "GetTilingCore"
          : "GetTiling";
  const size_t ub_case1_fail_pos = result.tiling.find("if (!optiling::" + tiling_entry + "(tiling->tiling_data, 1)) {");
  ASSERT_NE(ub_case1_fail_pos, std::string::npos);
  const size_t ub_case1_else_pos = result.tiling.find("    } else {", ub_case1_fail_pos);
  ASSERT_NE(ub_case1_else_pos, std::string::npos);
  const std::string ub_case1_fail_body = result.tiling.substr(ub_case1_fail_pos, ub_case1_else_pos - ub_case1_fail_pos);
  EXPECT_NE(ub_case1_fail_body.find("set_g_basen_basem_align(1);"), std::string::npos);
  EXPECT_NE(ub_case1_fail_body.find("for (size_t i = 2U;"), std::string::npos);
  EXPECT_NE(ub_case1_fail_body.find("tiling->cv_tiling_data.fusion_mode = 1;"), std::string::npos);
  EXPECT_NE(ub_case1_fail_body.find("tiling->cv_tiling_data.ub_mode = 0;"), std::string::npos);
  EXPECT_NE(ub_case1_fail_body.find("return 0;"), std::string::npos);
}

TEST_F(TestCodegenTiling, CodegenGenerateShouldNotEmitSplitMarkers) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen::CodegenResult result;

  ASSERT_EQ(codegen.Generate(fused_schedule_result, result), af::SUCCESS);

  EXPECT_EQ(result.tiling.find("AUTOFUSE_SPLIT_FILE_BEGIN"), std::string::npos);
  EXPECT_EQ(result.tiling.find("AUTOFUSE_SPLIT_FILE_END"), std::string::npos);
  EXPECT_EQ(result.tiling.find("#include \"autofuse_tiling_func_base.h\""), std::string::npos);
  EXPECT_EQ(result.tiling.find("#include \"autofuse_tiling_func_solver.h\""), std::string::npos);
}

TEST_F(TestCodegenTiling, TilingLibGenerateForInductorShouldNotEmitSplitMarkers) {
  auto fused_schedule_result = this->GenBasicFusedScheduleResult({af::Symbol("s0"), af::Symbol("s1")});
  auto tiling_files = this->GenerateForInductor(fused_schedule_result);

  ASSERT_TRUE(tiling_files.find(codegen::kTilingDefAndConstIdentify) != tiling_files.end());
  for (const auto &[key, content] : tiling_files) {
    EXPECT_EQ(content.find("AUTOFUSE_SPLIT_FILE_BEGIN"), std::string::npos) << key;
    EXPECT_EQ(content.find("AUTOFUSE_SPLIT_FILE_END"), std::string::npos) << key;
  }
}

TEST_F(TestCodegenTiling, TransposeApiCallGetTransposeTypeInvalid) {
  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data x("x", graph);
  af::ascir_op::Data y("y", graph);
  auto node0 = graph.FindNode("x");
  auto node1 = graph.FindNode("y");
  af::AscTensor input_tensor = node0->outputs[0];
  af::AscTensor out_tensor = node1->outputs[0];

  input_tensor.attr.vectorized_axis = {z0.id, z1.id, z2.id};
  out_tensor.attr.vectorized_axis = {z0.id, z1.id, z2.id};

  std::string dtype_name;
  codegen::Tensor::DtypeName(input_tensor.attr.dtype, dtype_name);
  codegen::Tensor input(input_tensor, dtype_name);
  codegen::Tensor output(out_tensor, dtype_name);

  codegen::TransposeApiCall call("Transpose");
  AutoFuseTransposeType transpose_type;
  call.CodeGenGetTransposeType(input, output, transpose_type);
  EXPECT_EQ(transpose_type, AutoFuseTransposeType::TRANSPOSE_INVALID);
}
