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
#include "node_utils_ex.h"
#include "graph_utils.h"
#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "codegen_kernel.h"
#include "utils/api_call_factory.h"
#include "split_reg_api_call.h"

namespace codegen {
using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;
class SplitRegApiCallUTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}

  static void BuildSplitGraph(const std::vector<af::Expression> &expressions, af::AscGraph &graph,
                              codegen::TPipe &tpipe, codegen::Tiler &tiler, DataType data_type = af::DT_FLOAT16) {
    auto s0 = expressions[0];
    auto s1 = expressions[1];
    auto s2_1 = expressions[2];
    auto s2_2 = expressions[3];
    auto s3 = expressions[4];
    // dlog_setlevel(0, 0, 1);
    auto z0 = graph.CreateAxis("z0", s0);
    auto z1 = graph.CreateAxis("z1", s1);
    auto z2 = graph.CreateAxis("z2", s2_1 + s2_2);
    auto z3 = graph.CreateAxis("z3", s3);

    Data x_op("x", graph);
    Load load_op("load");
    // Load load_op2("load2");
    af::ascir_op::Split split_op("split");

    graph.AddNode(load_op);
    // graph.AddNode(load_op2);
    // graph.AddNode(split_op);

    load_op.x = x_op.y;
    load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
    *load_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
    *load_op.y.repeats = {s0, s1, s2_1 + s2_2, s3};
    *load_op.y.strides = {(s2_1 + s2_2) * s3, Zero, s3, One};

    split_op.InstanceOutputy(2U);  // 需要先指定输出个数
    // split_op.x1 = load_op.y;
    // split_op.x2 = load_op2.y;
    split_op.x = {load_op.y};
    split_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
    *split_op.y[0].axis = {z0.id, z1.id, z2.id, z3.id};
    *split_op.y[0].repeats = {s0, s1, s2_1, s3};
    *split_op.y[0].strides = {s2_1 * s3, Zero, s3, One};

    *split_op.y[1].axis = {z0.id, z1.id, z2.id, z3.id};
    *split_op.y[1].repeats = {s0, s1, s2_2, s3};
    *split_op.y[1].strides = {s2_2 * s3, Zero, s3, One};

    auto load = graph.FindNode("load");
    load->attr.api.compute_type = af::ComputeType::kComputeLoad;
    load->attr.api.type = af::ApiType::kAPITypeCompute;
    load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
    load->attr.sched.loop_axis = z0.id;
    load->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id};
    load->outputs[0].attr.vectorized_strides = {(s2_1 + s2_2) * s3, Zero, s3, One};
    load->outputs[0].attr.dtype = data_type;
    load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
    load->outputs[0].attr.mem.tensor_id = 0;
    load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
    load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
    load->outputs[0].attr.que.id = 1;
    load->outputs[0].attr.opt.merge_scope = af::kIdNone;

    auto split = graph.FindNode("split");
    split->attr.api.unit = af::ComputeUnit::kUnitVector;
    split->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id};
    split->outputs[0].attr.vectorized_strides = {s2_1 * s3, Zero, s3, One};
    split->outputs[0].attr.dtype = data_type;
    split->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
    split->outputs[0].attr.mem.tensor_id = 2;
    split->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
    split->outputs[0].attr.que.id = 2;
    split->outputs[0].attr.opt.merge_scope = af::kIdNone;

    split->outputs[1].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id};
    split->outputs[1].attr.vectorized_strides = {s2_2 * s3, Zero, s3, One};
    split->outputs[1].attr.dtype = data_type;
    split->outputs[1].attr.mem.position = af::Position::kPositionVecOut;
    split->outputs[1].attr.mem.tensor_id = 3;
    split->outputs[1].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
    split->outputs[1].attr.que.id = 3;
    split->outputs[1].attr.opt.merge_scope = af::kIdNone;

    tiler.AddAxis(z0);
    tiler.AddAxis(z1);
    tiler.AddAxis(z2);
    tiler.AddAxis(z3);
    tiler.AddSizeVar(af::SizeVar(s0));
    tiler.AddSizeVar(af::SizeVar(s1));
    tiler.AddSizeVar(af::SizeVar(s2_1));
    tiler.AddSizeVar(af::SizeVar(s2_2));
    tiler.AddSizeVar(af::SizeVar(s3));
    tpipe.CollectQues(graph);
    // add load1 tensor
    EXPECT_EQ(tpipe.AddTensor(load->outputs[0]), 0);

    // add add tensor
    EXPECT_EQ(tpipe.AddTensor(split->outputs[0]), 0);
    EXPECT_EQ(tpipe.AddTensor(split->outputs[1]), 0);
    // EXPECT_EQ(tpipe.AddTensor(load2->outputs[1]), 0);

    split->attr.tmp_buffers.emplace_back(af::TmpBuffer{af::TmpBufDesc{af::Symbol(1024), -1}, {}});
  }
};

TEST_F(SplitRegApiCallUTest, AllAligned) {
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar(1);
  auto s2_1 = graph.CreateSizeVar(8);
  auto s2_2 = graph.CreateSizeVar(16);
  auto s3 = graph.CreateSizeVar(2);

  BuildSplitGraph({s0, s1, s2_1, s2_2, s3}, graph, tpipe, tiler);

  auto load = graph.FindNode("load");
  auto split = graph.FindNode("split");

  codegen::SplitRegApiCall call("Split");
  EXPECT_EQ(call.Init(split), 0);
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  call.inputs.push_back(&x1);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), af::SUCCESS);
  const auto api_param = CodegenApiParam::GetNodeApiParam(split);
  ASSERT_NE(api_param, nullptr);
  EXPECT_EQ(api_param->semantic_feature, "UB/AllAligned");
  EXPECT_EQ(result,
            "constexpr SplitTilingAllAligned<2> split_tiling {\n"
            "  .src_col_size = 48,\n"
            "  .dst_col_sizes = { 16, 32, },\n"
            "  .src_offsets = { 0, 16, },\n"
            "};\n"
            "LocalTensor<half> split_dst_tensors[] { local_2, local_3, };\n"
            "SplitAllAligned<half, 2>(t->s0, split_tiling, local_0, split_dst_tensors);\n");
}

TEST_F(SplitRegApiCallUTest, Unaligned_B8) {
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar(1);
  auto s2_1 = graph.CreateSizeVar("s2_1");
  auto s2_2 = graph.CreateSizeVar(16);
  auto s3 = graph.CreateSizeVar(3);

  BuildSplitGraph({s0, s1, s2_1, s2_2, s3}, graph, tpipe, tiler, DT_INT8);

  auto load = graph.FindNode("load");
  auto split = graph.FindNode("split");
  split->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};

  codegen::SplitRegApiCall call("Split");
  EXPECT_EQ(call.Init(split), 0);
  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  call.inputs.push_back(&x1);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), af::SUCCESS);
  EXPECT_EQ(
      result,
      "const split::SplitTiling<2> split_tiling {\n  .num_rows = static_cast<uint32_t>(t->s0), \n  .num_src_cols = "
      "static_cast<uint32_t>((((16 + t->s2_1) * 3))/(1)), \n  .num_dsts_cols = {static_cast<uint32_t>(((3 * "
      "t->s2_1))/(1)), 48, }\n};\n__ubuf__ int8_t "
      "*split_dst_addrs[] { (__ubuf__ int8_t *)local_2.GetPhyAddr(), (__ubuf__ int8_t *)local_3.GetPhyAddr(), "
      "};\nsplit::SplitExtend<int8_t, 2>((__ubuf__ int8_t *)local_0.GetPhyAddr(), split_dst_addrs, tmp_buf_0, "
      "split_tiling);\n");
}

TEST_F(SplitRegApiCallUTest, Unaligned_B8ToB16) {
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar(1);
  auto s2_1 = graph.CreateSizeVar("s2_1");
  auto s2_2 = graph.CreateSizeVar(16);
  auto s3 = graph.CreateSizeVar(2);

  BuildSplitGraph({s0, s1, s2_1, s2_2, s3}, graph, tpipe, tiler, DT_INT8);

  auto load = graph.FindNode("load");
  auto split = graph.FindNode("split");
  split->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};

  codegen::SplitRegApiCall call("Split");
  EXPECT_EQ(call.Init(split), 0);
  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  call.inputs.push_back(&x1);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), af::SUCCESS);
  EXPECT_EQ(
      result,
      "const split::SplitTiling<2> split_tiling {\n  .num_rows = static_cast<uint32_t>(t->s0), \n  .num_src_cols "
      "= static_cast<uint32_t>(((16 + t->s2_1))/(1)), \n  .num_dsts_cols = {static_cast<uint32_t>((t->s2_1)/(1)), "
      "16, }\n};\n__ubuf__ uint16_t "
      "*split_dst_addrs[] { (__ubuf__ uint16_t *)local_2.GetPhyAddr(), (__ubuf__ uint16_t *)local_3.GetPhyAddr(), "
      "};\nsplit::SplitExtend<uint16_t, 2>((__ubuf__ uint16_t *)local_0.GetPhyAddr(), split_dst_addrs, "
      "tmp_buf_0, split_tiling);\n");
}

TEST_F(SplitRegApiCallUTest, Unaligned_B16) {
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar(1);
  auto s2_1 = graph.CreateSizeVar("s2_1");
  auto s2_2 = graph.CreateSizeVar(16);
  auto s3 = graph.CreateSizeVar(2);

  BuildSplitGraph({s0, s1, s2_1, s2_2, s3}, graph, tpipe, tiler);

  auto load = graph.FindNode("load");
  auto split = graph.FindNode("split");
  split->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};

  codegen::SplitRegApiCall call("Split");
  EXPECT_EQ(call.Init(split), 0);
  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  call.inputs.push_back(&x1);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), af::SUCCESS);
  EXPECT_EQ(
      result,
      "const split::SplitTiling<2> split_tiling {\n  .num_rows = static_cast<uint32_t>(t->s0), \n  .num_src_cols = "
      "static_cast<uint32_t>((((16 + t->s2_1) * 2))/(1)), \n  .num_dsts_cols = {static_cast<uint32_t>(((2 * "
      "t->s2_1))/(1)), 32, }\n};\n__ubuf__ half "
      "*split_dst_addrs[] { (__ubuf__ half *)local_2.GetPhyAddr(), (__ubuf__ half *)local_3.GetPhyAddr(), "
      "};\nsplit::SplitExtend<half, 2>((__ubuf__ half *)local_0.GetPhyAddr(), split_dst_addrs, tmp_buf_0, "
      "split_tiling);\n");
}

TEST_F(SplitRegApiCallUTest, Unaligned_B32) {
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar(1);
  auto s2_1 = graph.CreateSizeVar("s2_1");
  auto s2_2 = graph.CreateSizeVar(16);
  auto s3 = graph.CreateSizeVar(2);

  BuildSplitGraph({s0, s1, s2_1, s2_2, s3}, graph, tpipe, tiler, DT_INT32);

  auto load = graph.FindNode("load");
  auto split = graph.FindNode("split");
  split->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};

  codegen::SplitRegApiCall call("Split");
  EXPECT_EQ(call.Init(split), 0);
  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  call.inputs.push_back(&x1);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), af::SUCCESS);
  EXPECT_EQ(
      result,
      "const split::SplitTiling<2> split_tiling {\n  .num_rows = static_cast<uint32_t>(t->s0), \n  .num_src_cols = "
      "static_cast<uint32_t>((((16 + t->s2_1) * 2))/(1)), \n  .num_dsts_cols = {static_cast<uint32_t>(((2 * "
      "t->s2_1))/(1)), 32, }\n};\n__ubuf__ int32_t "
      "*split_dst_addrs[] { (__ubuf__ int32_t *)local_2.GetPhyAddr(), (__ubuf__ int32_t *)local_3.GetPhyAddr(), "
      "};\nsplit::SplitExtend<int32_t, 2>((__ubuf__ int32_t *)local_0.GetPhyAddr(), split_dst_addrs, tmp_buf_0, "
      "split_tiling);\n");
}

TEST_F(SplitRegApiCallUTest, Unalign_B64) {
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar(1);
  auto s2_1 = graph.CreateSizeVar("s2_1");
  auto s2_2 = graph.CreateSizeVar(16);
  auto s3 = graph.CreateSizeVar(2);

  BuildSplitGraph({s0, s1, s2_1, s2_2, s3}, graph, tpipe, tiler, DT_INT64);

  auto load = graph.FindNode("load");
  auto split = graph.FindNode("split");
  split->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};

  codegen::SplitRegApiCall call("Split");
  EXPECT_EQ(call.Init(split), 0);
  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  call.inputs.push_back(&x1);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), af::SUCCESS);
  EXPECT_EQ(
      result,
      "const split::SplitTiling<2> split_tiling {\n  .num_rows = static_cast<uint32_t>(t->s0), \n  .num_src_cols = "
      "static_cast<uint32_t>((((16 + t->s2_1) * 4))/(1)), \n  .num_dsts_cols = {static_cast<uint32_t>(((4 * "
      "t->s2_1))/(1)), 64, }\n};\n__ubuf__ uint32_t "
      "*split_dst_addrs[] { (__ubuf__ uint32_t *)local_2.GetPhyAddr(), (__ubuf__ uint32_t *)local_3.GetPhyAddr(), "
      "};\nsplit::SplitExtend<uint32_t, 2>((__ubuf__ uint32_t *)local_0.GetPhyAddr(), split_dst_addrs, tmp_buf_0, "
      "split_tiling);\n");
}
}  // namespace codegen
