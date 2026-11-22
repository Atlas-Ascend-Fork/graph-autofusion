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
#include "concat_reg_api_call.h"

namespace codegen {
using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;
class ConcatRegApiCallUTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}

  static void BuildConcatGraph(const std::vector<af::Expression> &expressions, af::AscGraph &graph,
                               codegen::TPipe &tpipe, codegen::Tiler &tiler, DataType data_type = af::DT_FLOAT16,
                               uint32_t align_size = 0) {
    auto s0 = expressions[0];
    auto s1 = expressions[1];
    auto s2_1 = expressions[2];
    auto s2_2 = expressions[3];
    auto s3 = expressions[4];

    auto z0 = graph.CreateAxis("z0", s0);
    auto z1 = graph.CreateAxis("z1", s1);
    auto z2 = graph.CreateAxis("z2", s2_1 + s2_2);
    auto z3 = graph.CreateAxis("z3", s3);

    Data x_op("x", graph);
    Load load_op("load");
    Load load_op2("load2");
    af::ascir_op::Concat concat_op("concat");

    graph.AddNode(load_op);
    graph.AddNode(load_op2);
    // graph.AddNode(concat_op);

    load_op.x = x_op.y;
    load_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
    *load_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
    *load_op.y.repeats = {s0, s1, s2_1, s3};
    *load_op.y.strides = {s2_1 * s3, Zero, s3, One};

    load_op2.x = x_op.y;
    load_op2.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
    *load_op2.y.axis = {z0.id, z1.id, z2.id, z3.id};
    *load_op2.y.repeats = {s0, s1, s2_2, s3};
    *load_op2.y.strides = {s2_2 * s3, Zero, s3, One};

    // concat_op.x1 = load_op.y;
    // concat_op.x2 = load_op2.y;
    concat_op.x = {load_op.y, load_op2.y};
    concat_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
    *concat_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
    *concat_op.y.repeats = {s0, s1, s2_1 + s2_2, s3};
    *concat_op.y.strides = {(s2_1 + s2_2) * s3, Zero, s3, One};

    auto load = graph.FindNode("load");
    load->attr.api.compute_type = af::ComputeType::kComputeLoad;
    load->attr.api.type = af::ApiType::kAPITypeCompute;
    load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
    load->attr.sched.loop_axis = z0.id;
    load->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id};
    auto actual_s3 = align_size != 0 ? af::sym::Align(s3, align_size) : s3;
    load->outputs[0].attr.vectorized_strides = {s2_1 * actual_s3, Zero, actual_s3, One};
    load->outputs[0].attr.dtype = data_type;
    load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
    load->outputs[0].attr.mem.tensor_id = 0;
    load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
    load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
    load->outputs[0].attr.que.id = 1;
    load->outputs[0].attr.opt.merge_scope = af::kIdNone;

    auto load2 = graph.FindNode("load2");
    load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
    load2->attr.api.type = af::ApiType::kAPITypeCompute;
    load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
    load2->attr.sched.loop_axis = z0.id;
    load2->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id};
    load2->outputs[0].attr.vectorized_strides = {s2_2 * actual_s3, Zero, actual_s3, One};
    load2->outputs[0].attr.dtype = data_type;
    load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
    load2->outputs[0].attr.mem.tensor_id = 2;
    load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
    load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
    load2->outputs[0].attr.que.id = 2;
    load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

    auto concat = graph.FindNode("concat");
    concat->attr.api.unit = af::ComputeUnit::kUnitVector;
    concat->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id};
    concat->outputs[0].attr.vectorized_strides = {(s2_1 + s2_2) * actual_s3, Zero, actual_s3, One};
    concat->outputs[0].attr.dtype = data_type;
    concat->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
    concat->outputs[0].attr.mem.tensor_id = 3;
    concat->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
    concat->outputs[0].attr.que.id = 3;
    concat->outputs[0].attr.opt.merge_scope = af::kIdNone;

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
    EXPECT_EQ(tpipe.AddTensor(load2->outputs[0]), 0);

    // add add tensor
    EXPECT_EQ(tpipe.AddTensor(concat->outputs[0]), 0);

    concat->attr.tmp_buffers.emplace_back(af::TmpBuffer{af::TmpBufDesc{af::Symbol(1024), -1}, {}});
  }

  static void Build1DConcatGraph(const std::vector<af::Expression> &expressions, af::AscGraph &graph,
                                 codegen::TPipe &tpipe, codegen::Tiler &tiler, DataType data_type = af::DT_FLOAT16) {
    auto s0 = One;
    auto s1_1 = expressions[0];
    auto s1_2 = expressions[1];

    auto z0 = graph.CreateAxis("z0", s0);
    auto z1 = graph.CreateAxis("z1", s1_1 + s1_2);

    Data x_op("x", graph);
    Load load_op("load");
    Load load_op2("load2");
    af::ascir_op::Concat concat_op("concat");

    graph.AddNode(load_op);
    graph.AddNode(load_op2);

    load_op.x = x_op.y;
    load_op.attr.sched.axis = {z0.id, z1.id};
    *load_op.y.axis = {z0.id, z1.id};
    *load_op.y.repeats = {s0, s1_1};
    *load_op.y.strides = {Zero, One};

    load_op2.x = x_op.y;
    load_op2.attr.sched.axis = {z0.id, z1.id};
    *load_op2.y.axis = {z0.id, z1.id};
    *load_op2.y.repeats = {s0, s1_2};
    *load_op2.y.strides = {Zero, One};

    // concat_op.x1 = load_op.y;
    // concat_op.x2 = load_op2.y;
    concat_op.x = {load_op.y, load_op2.y};
    concat_op.attr.sched.axis = {z0.id, z1.id};
    *concat_op.y.axis = {z0.id, z1.id};
    *concat_op.y.repeats = {s0, s1_1 + s1_2};
    *concat_op.y.strides = {Zero, One};

    auto load = graph.FindNode("load");
    load->attr.api.compute_type = af::ComputeType::kComputeLoad;
    load->attr.api.type = af::ApiType::kAPITypeCompute;
    load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
    load->attr.sched.loop_axis = z0.id;
    load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
    load->outputs[0].attr.vectorized_strides = {Zero, One};
    load->outputs[0].attr.dtype = data_type;
    load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
    load->outputs[0].attr.mem.tensor_id = 0;
    load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
    load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
    load->outputs[0].attr.que.id = 1;
    load->outputs[0].attr.opt.merge_scope = af::kIdNone;

    auto load2 = graph.FindNode("load2");
    load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
    load2->attr.api.type = af::ApiType::kAPITypeCompute;
    load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
    load2->attr.sched.loop_axis = z0.id;
    load2->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
    load2->outputs[0].attr.vectorized_strides = {Zero, One};
    load2->outputs[0].attr.dtype = data_type;
    load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
    load2->outputs[0].attr.mem.tensor_id = 2;
    load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
    load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
    load2->outputs[0].attr.que.id = 2;
    load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

    auto concat = graph.FindNode("concat");
    concat->attr.api.unit = af::ComputeUnit::kUnitVector;
    concat->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
    concat->outputs[0].attr.vectorized_strides = {Zero, One};
    concat->outputs[0].attr.dtype = data_type;
    concat->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
    concat->outputs[0].attr.mem.tensor_id = 3;
    concat->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
    concat->outputs[0].attr.que.id = 3;
    concat->outputs[0].attr.opt.merge_scope = af::kIdNone;

    tiler.AddAxis(z0);
    tiler.AddAxis(z1);
    tiler.AddSizeVar(af::SizeVar(s0));
    tiler.AddSizeVar(af::SizeVar(s1_1));
    tiler.AddSizeVar(af::SizeVar(s1_2));
    tpipe.CollectQues(graph);
    // add load1 tensor
    EXPECT_EQ(tpipe.AddTensor(load->outputs[0]), 0);
    EXPECT_EQ(tpipe.AddTensor(load2->outputs[0]), 0);

    // add add tensor
    EXPECT_EQ(tpipe.AddTensor(concat->outputs[0]), 0);

    concat->attr.tmp_buffers.emplace_back(af::TmpBuffer{af::TmpBufDesc{af::Symbol(1024), -1}, {}});
  }

  static void BuildConcat3DGraph(const std::vector<af::Expression> &expressions, af::AscGraph &graph,
                                 codegen::TPipe &tpipe, codegen::Tiler &tiler, DataType data_type = af::DT_FLOAT16,
                                 uint32_t align_size = 0) {
    auto s0 = expressions[0];
    auto s1 = expressions[1];
    auto s2_1 = expressions[2];
    auto s2_2 = expressions[3];

    auto z0 = graph.CreateAxis("z0", s0);
    auto z1 = graph.CreateAxis("z1", s1);
    auto z2 = graph.CreateAxis("z2", s2_1 + s2_2);

    Data x_op("x", graph);
    Load load_op("load");
    Load load_op2("load2");
    af::ascir_op::Concat concat_op("concat");

    graph.AddNode(load_op);
    graph.AddNode(load_op2);

    load_op.x = x_op.y;
    load_op.attr.sched.axis = {z0.id, z1.id, z2.id};
    *load_op.y.axis = {z0.id, z1.id, z2.id};
    *load_op.y.repeats = {s0, s1, s2_1};
    *load_op.y.strides = {s2_1, Zero, One};

    load_op2.x = x_op.y;
    load_op2.attr.sched.axis = {z0.id, z1.id, z2.id};
    *load_op2.y.axis = {z0.id, z1.id, z2.id};
    *load_op2.y.repeats = {s0, s1, s2_2};
    *load_op2.y.strides = {s2_2, Zero, One};

    // concat_op.x1 = load_op.y;
    // concat_op.x2 = load_op2.y;
    concat_op.x = {load_op.y, load_op2.y};
    concat_op.attr.sched.axis = {z0.id, z1.id, z2.id};
    *concat_op.y.axis = {z0.id, z1.id, z2.id};
    *concat_op.y.repeats = {s0, s1, s2_1 + s2_2};
    *concat_op.y.strides = {(s2_1 + s2_2), Zero, One};

    auto load = graph.FindNode("load");
    load->attr.api.compute_type = af::ComputeType::kComputeLoad;
    load->attr.api.type = af::ApiType::kAPITypeCompute;
    load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
    load->attr.sched.loop_axis = z0.id;
    load->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
    auto actual_s2_1 = align_size != 0 ? af::sym::Align(s2_1, align_size) : s2_1;
    auto actual_s2_2 = align_size != 0 ? af::sym::Align(s2_2, align_size) : s2_2;
    auto actual_s2_1_s2_2 = align_size != 0 ? af::sym::Align(s2_1 + s2_2, align_size) : (s2_1 + s2_2);
    load->outputs[0].attr.vectorized_strides = {actual_s2_1, Zero, One};
    load->outputs[0].attr.dtype = data_type;
    load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
    load->outputs[0].attr.mem.tensor_id = 0;
    load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
    load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
    load->outputs[0].attr.que.id = 1;
    load->outputs[0].attr.opt.merge_scope = af::kIdNone;

    auto load2 = graph.FindNode("load2");
    load2->attr.api.compute_type = af::ComputeType::kComputeLoad;
    load2->attr.api.type = af::ApiType::kAPITypeCompute;
    load2->attr.api.unit = af::ComputeUnit::kUnitMTE2;
    load2->attr.sched.loop_axis = z0.id;
    load2->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
    load2->outputs[0].attr.vectorized_strides = {actual_s2_2, Zero, One};
    load2->outputs[0].attr.dtype = data_type;
    load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
    load2->outputs[0].attr.mem.tensor_id = 2;
    load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
    load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
    load2->outputs[0].attr.que.id = 2;
    load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

    auto concat = graph.FindNode("concat");
    concat->attr.api.unit = af::ComputeUnit::kUnitVector;
    concat->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
    concat->outputs[0].attr.vectorized_strides = {actual_s2_1_s2_2, Zero, One};
    concat->outputs[0].attr.dtype = data_type;
    concat->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
    concat->outputs[0].attr.mem.tensor_id = 3;
    concat->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
    concat->outputs[0].attr.que.id = 3;
    concat->outputs[0].attr.opt.merge_scope = af::kIdNone;

    tiler.AddAxis(z0);
    tiler.AddAxis(z1);
    tiler.AddAxis(z2);
    tiler.AddSizeVar(af::SizeVar(s0));
    tiler.AddSizeVar(af::SizeVar(s1));
    tiler.AddSizeVar(af::SizeVar(s2_1));
    tiler.AddSizeVar(af::SizeVar(s2_2));
    tpipe.CollectQues(graph);
    // add load1 tensor
    EXPECT_EQ(tpipe.AddTensor(load->outputs[0]), 0);
    EXPECT_EQ(tpipe.AddTensor(load2->outputs[0]), 0);

    // add add tensor
    EXPECT_EQ(tpipe.AddTensor(concat->outputs[0]), 0);

    concat->attr.tmp_buffers.emplace_back(af::TmpBuffer{af::TmpBufDesc{af::Symbol(1024), -1}, {}});
  }
};

TEST_F(ConcatRegApiCallUTest, AllAligned) {
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar(1);
  auto s2_1 = graph.CreateSizeVar(8);
  auto s2_2 = graph.CreateSizeVar(16);
  auto s3 = graph.CreateSizeVar(2);

  BuildConcatGraph({s0, s1, s2_1, s2_2, s3}, graph, tpipe, tiler);

  auto load = graph.FindNode("load");
  auto load2 = graph.FindNode("load2");
  auto concat = graph.FindNode("concat");

  codegen::ConcatRegApiCall call("Concat");
  EXPECT_EQ(call.Init(concat), 0);
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), af::SUCCESS);
  const auto api_param = CodegenApiParam::GetNodeApiParam(concat);
  ASSERT_NE(api_param, nullptr);
  EXPECT_EQ(api_param->semantic_feature, "UB/AllAligned");
  EXPECT_EQ(result,
            "constexpr ConcatTilingAllAligned<2> concat_tiling {\n"
            "  .dst_col_size = 48,\n"
            "  .src_col_sizes = { 16, 32, },\n"
            "  .dst_offsets = { 0, 16, },\n"
            "};\n"
            "LocalTensor<half> concat_src_tensors[] { local_0, local_2, };\n"
            "ConcatAllAligned<half, 2>(t->s0, concat_tiling, local_3, concat_src_tensors);\n");
}

TEST_F(ConcatRegApiCallUTest, Unaligned_B8) {
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar(1);
  auto s2_1 = graph.CreateSizeVar("s2_1");
  auto s2_2 = graph.CreateSizeVar(16);
  auto s3 = graph.CreateSizeVar(3);

  BuildConcatGraph({s0, s1, s2_1, s2_2, s3}, graph, tpipe, tiler, DT_INT8);

  auto load = graph.FindNode("load");
  auto load2 = graph.FindNode("load2");
  auto concat = graph.FindNode("concat");
  concat->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};

  codegen::ConcatRegApiCall call("Concat");
  EXPECT_EQ(call.Init(concat), 0);
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  load->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.que.id = 1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  call.is_input_tbuf_contiguous = true;

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), af::SUCCESS);
  EXPECT_EQ(
      result,
      "const concat::ConcatTiling<2> concat_tiling {\n  .num_rows = static_cast<uint32_t>(t->s0),\n  .num_dst_cols = "
      "static_cast<uint32_t>((((16 + t->s2_1) * 3))/(1)),\n  .num_srcs_cols = {static_cast<uint32_t>(((3 * "
      "t->s2_1))/(1)), 48, },\n};\n__ubuf__ uint8_t "
      "*concat_src_addrs[] { (__ubuf__ uint8_t *)local_0.GetPhyAddr(), (__ubuf__ uint8_t *)local_2.GetPhyAddr(), "
      "};\nconcat::ConcatExtendDyn<uint8_t, 2, true>((__ubuf__ uint8_t *)local_3.GetPhyAddr(), concat_src_addrs, "
      "tmp_buf_0, concat_tiling);\n");
}

TEST_F(ConcatRegApiCallUTest, Unaligned_B8ToB16) {
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar(1);
  auto s2_1 = graph.CreateSizeVar("s2_1");
  auto s2_2 = graph.CreateSizeVar(16);
  auto s3 = graph.CreateSizeVar(2);

  BuildConcatGraph({s0, s1, s2_1, s2_2, s3}, graph, tpipe, tiler, DT_INT8);

  auto load = graph.FindNode("load");
  auto load2 = graph.FindNode("load2");
  auto concat = graph.FindNode("concat");
  concat->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};

  codegen::ConcatRegApiCall call("Concat");
  EXPECT_EQ(call.Init(concat), 0);
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  load->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.que.id = 1;
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  call.is_input_tbuf_contiguous = true;

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), af::SUCCESS);
  EXPECT_EQ(
      result,
      "const concat::ConcatTiling<2> concat_tiling {\n  .num_rows = static_cast<uint32_t>(t->s0),\n  .num_dst_cols = "
      "static_cast<uint32_t>(((16 + t->s2_1))/(1)),\n  .num_srcs_cols = {static_cast<uint32_t>((t->s2_1)/(1)), "
      "16, },\n};\n__ubuf__ uint16_t *concat_src_addrs[] { "
      "(__ubuf__ uint16_t *)local_0.GetPhyAddr(), (__ubuf__ uint16_t *)local_2.GetPhyAddr(), "
      "};\nconcat::ConcatExtendDyn<uint16_t, 2, true>((__ubuf__ uint16_t *)local_3.GetPhyAddr(), concat_src_addrs, "
      "tmp_buf_0, concat_tiling);\n");
}

TEST_F(ConcatRegApiCallUTest, Unaligned_B16) {
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar(1);
  auto s2_1 = graph.CreateSizeVar("s2_1");
  auto s2_2 = graph.CreateSizeVar(16);
  auto s3 = graph.CreateSizeVar(2);

  BuildConcatGraph({s0, s1, s2_1, s2_2, s3}, graph, tpipe, tiler);

  auto load = graph.FindNode("load");
  auto load2 = graph.FindNode("load2");
  auto concat = graph.FindNode("concat");
  concat->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};

  codegen::ConcatRegApiCall call("Concat");
  EXPECT_EQ(call.Init(concat), 0);
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  load->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.que.id = 1;
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  call.is_input_tbuf_contiguous = true;

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), af::SUCCESS);
  EXPECT_EQ(
      result,
      "const concat::ConcatTiling<2> concat_tiling {\n  .num_rows = static_cast<uint32_t>(t->s0),\n  .num_dst_cols = "
      "static_cast<uint32_t>((((16 + t->s2_1) * 2))/(1)),\n  .num_srcs_cols = {static_cast<uint32_t>(((2 * "
      "t->s2_1))/(1)), 32, },\n};\n__ubuf__ half "
      "*concat_src_addrs[] { (__ubuf__ half *)local_0.GetPhyAddr(), (__ubuf__ half *)local_2.GetPhyAddr(), "
      "};\nconcat::ConcatExtendDyn<half, 2, true>((__ubuf__ half *)local_3.GetPhyAddr(), concat_src_addrs, tmp_buf_0, "
      "concat_tiling);\n");
}

TEST_F(ConcatRegApiCallUTest, Unaligned_B32) {
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar(1);
  auto s2_1 = graph.CreateSizeVar("s2_1");
  auto s2_2 = graph.CreateSizeVar(16);
  auto s3 = graph.CreateSizeVar(2);

  BuildConcatGraph({s0, s1, s2_1, s2_2, s3}, graph, tpipe, tiler, DT_INT32);

  auto load = graph.FindNode("load");
  auto load2 = graph.FindNode("load2");
  auto concat = graph.FindNode("concat");
  concat->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};

  codegen::ConcatRegApiCall call("Concat");
  EXPECT_EQ(call.Init(concat), 0);
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  load->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.que.id = 1;
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  call.is_input_tbuf_contiguous = true;

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), af::SUCCESS);
  EXPECT_EQ(
      result,
      "const concat::ConcatTiling<2> concat_tiling {\n  .num_rows = static_cast<uint32_t>(t->s0),\n  .num_dst_cols = "
      "static_cast<uint32_t>((((16 + t->s2_1) * 2))/(1)),\n  .num_srcs_cols = {static_cast<uint32_t>(((2 * "
      "t->s2_1))/(1)), 32, },\n};\n__ubuf__ int32_t "
      "*concat_src_addrs[] { (__ubuf__ int32_t *)local_0.GetPhyAddr(), (__ubuf__ int32_t *)local_2.GetPhyAddr(), "
      "};\nconcat::ConcatExtendDyn<int32_t, 2, true>((__ubuf__ int32_t *)local_3.GetPhyAddr(), concat_src_addrs, "
      "tmp_buf_0, concat_tiling);\n");
}

TEST_F(ConcatRegApiCallUTest, Unalign_B64) {
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar(1);
  auto s2_1 = graph.CreateSizeVar("s2_1");
  auto s2_2 = graph.CreateSizeVar(16);
  auto s3 = graph.CreateSizeVar(2);

  BuildConcatGraph({s0, s1, s2_1, s2_2, s3}, graph, tpipe, tiler, DT_INT64);

  auto load = graph.FindNode("load");
  auto load2 = graph.FindNode("load2");
  auto concat = graph.FindNode("concat");
  concat->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};

  codegen::ConcatRegApiCall call("Concat");
  EXPECT_EQ(call.Init(concat), 0);
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  load->outputs[0].attr.que.id = 1;
  load2->outputs[0].attr.que.id = 1;
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  call.is_input_tbuf_contiguous = true;

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), af::SUCCESS);
  EXPECT_EQ(
      result,
      "const concat::ConcatTiling<2> concat_tiling {\n  .num_rows = static_cast<uint32_t>(t->s0),\n  .num_dst_cols = "
      "static_cast<uint32_t>((((16 + t->s2_1) * 4))/(1)),\n  .num_srcs_cols = {static_cast<uint32_t>(((4 * "
      "t->s2_1))/(1)), 64, },\n};\n__ubuf__ uint32_t "
      "*concat_src_addrs[] { (__ubuf__ uint32_t *)local_0.GetPhyAddr(), (__ubuf__ uint32_t *)local_2.GetPhyAddr(), "
      "};\nconcat::ConcatExtendDyn<uint32_t, 2, true>((__ubuf__ uint32_t *)local_3.GetPhyAddr(), concat_src_addrs, "
      "tmp_buf_0, concat_tiling);\n");
}

TEST_F(ConcatRegApiCallUTest, Unaligned_B32_padded) {
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar(1);
  auto s2_1 = graph.CreateSizeVar("s2_1");
  auto s2_2 = graph.CreateSizeVar(16);
  auto s3 = graph.CreateSizeVar(2);

  BuildConcatGraph({s0, s1, s2_1, s2_2, s3}, graph, tpipe, tiler, DT_INT32, 8);

  auto load = graph.FindNode("load");
  auto load2 = graph.FindNode("load2");
  auto concat = graph.FindNode("concat");
  concat->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};

  codegen::ConcatRegApiCall call("Concat");
  EXPECT_EQ(call.Init(concat), 0);
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);
  call.is_input_tbuf_contiguous = true;

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), af::SUCCESS);
  const auto api_param = CodegenApiParam::GetNodeApiParam(concat);
  ASSERT_NE(api_param, nullptr);
  EXPECT_EQ(api_param->semantic_feature, "UB/AllAligned");
  std::cout << result << std::endl;
  EXPECT_EQ(result,
            "const ConcatTilingAllAligned<2> concat_tiling {\n  .dst_col_size = static_cast<uint32_t>((((16 + "
            "t->s2_1) * 8))/(1)),\n  "
            ".src_col_sizes = { static_cast<uint32_t>(((8 * t->s2_1))/(1)), 128, },\n  "
            ".dst_offsets = { 0, static_cast<uint32_t>(((8 * t->s2_1))/(1)), "
            "},\n};\nLocalTensor<int32_t> concat_src_tensors[] { local_0, local_2, };\nConcatAllAligned<int32_t, "
            "2>(t->s0, concat_tiling, local_3, concat_src_tensors);\n");
}

TEST_F(ConcatRegApiCallUTest, Unaligned_B32_padded_concat_last_dim) {
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar(1);
  auto s2_1 = graph.CreateSizeVar("s2_1");
  auto s2_2 = graph.CreateSizeVar(16);

  BuildConcat3DGraph({s0, s1, s2_1, s2_2}, graph, tpipe, tiler, DT_INT32, 8);

  auto load = graph.FindNode("load");
  auto load2 = graph.FindNode("load2");
  auto concat = graph.FindNode("concat");
  concat->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};

  codegen::ConcatRegApiCall call("Concat");
  EXPECT_EQ(call.Init(concat), 0);
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), af::SUCCESS);
  EXPECT_EQ(
      result,
      "const concat::ConcatTilingPadded<2> concat_tiling {\n  .num_rows = static_cast<uint32_t>(t->s0),\n  "
      ".num_dst_cols = static_cast<uint32_t>(((16 + t->s2_1))/(1)),\n  .num_srcs_cols = "
      "{static_cast<uint32_t>((t->s2_1)/(1)), 16, },\n  .src_row_strides = "
      "{static_cast<uint32_t>(((8 * Ceiling((Rational(1 , 8) * t->s2_1))))/(1)), 16, "
      "},\n  .src_second_last_dim_strides = {static_cast<uint32_t>(((8 * Ceiling((Rational(1 "
      ", 8) * t->s2_1))))/(1)), 0, },\n  .gather_mask_dim_sizes = {static_cast<uint32_t>(t->s2_1), 0, },\n};\n__ubuf__ "
      "int32_t "
      "*concat_src_addrs[] { (__ubuf__ int32_t *)local_0.GetPhyAddr(), (__ubuf__ int32_t *)local_2.GetPhyAddr(), "
      "};\nconcat::ConcatExtend<int32_t, 2>((__ubuf__ int32_t *)local_3.GetPhyAddr(), concat_src_addrs, tmp_buf_0, "
      "concat_tiling);\n");
}

TEST_F(ConcatRegApiCallUTest, OneAxis) {
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test_graph");
  auto s1_1 = graph.CreateSizeVar(1);
  auto s1_2 = graph.CreateSizeVar(1);

  Build1DConcatGraph({s1_1, s1_2}, graph, tpipe, tiler, DT_INT32);

  auto load = graph.FindNode("load");
  auto load2 = graph.FindNode("load2");
  auto concat = graph.FindNode("concat");

  codegen::ConcatRegApiCall call("Concat");
  EXPECT_EQ(call.Init(concat), 0);
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), af::SUCCESS);
  std::cout << result << std::endl;
  EXPECT_TRUE(result.find("concat::ConcatOneAxis") != std::string::npos);
}

TEST_F(ConcatRegApiCallUTest, ConcatByGather) {
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);

  af::AscGraph graph("test_graph");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar(1);
  auto s2_1 = graph.CreateSizeVar(63);
  auto s2_2 = graph.CreateSizeVar(63);
  auto s3 = graph.CreateSizeVar(1);

  BuildConcatGraph({s0, s1, s2_1, s2_2, s3}, graph, tpipe, tiler);

  auto load = graph.FindNode("load");
  auto load2 = graph.FindNode("load2");
  auto concat = graph.FindNode("concat");
  concat->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};

  codegen::ConcatRegApiCall call("Concat");
  EXPECT_EQ(call.Init(concat), 0);
  call.is_input_tbuf_contiguous = true;
  codegen::ApiTensor x1;
  codegen::ApiTensor x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), af::SUCCESS);
  const auto api_param = CodegenApiParam::GetNodeApiParam(concat);
  ASSERT_NE(api_param, nullptr);
  EXPECT_EQ(api_param->semantic_feature, "UB/ByGather");
  std::cout << result;
  EXPECT_TRUE(result.find("concat::ConcatExtendByGather") != std::string::npos);
}
}  // namespace codegen
