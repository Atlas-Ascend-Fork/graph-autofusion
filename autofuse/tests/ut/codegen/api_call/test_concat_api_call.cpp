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
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common_utils.h"
#include "utils/api_call_factory.h"
#include "concat/concat_api_call.h"

using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;
using namespace codegen;

TEST(ConcatApiCallTest, ConcatApicallWithConcatDimIsInerAxis) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1_1 = graph.CreateSizeVar("s1_1");
  auto s1_2 = graph.CreateSizeVar("s1_2");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1_1 + s1_2);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x_op("x", graph);
  Load load_op("load");
  Load load_op2("load2");
  af::ascir_op::Concat concat_op("concat");

  graph.AddNode(load_op);
  graph.AddNode(load_op2);
  // graph.AddNode(concat_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op.y.axis = {z0.id, z1.id, z2.id};
  *load_op.y.repeats = {s0, s1_1, s2};
  *load_op.y.strides = {s1_1 * s2, s2, One};

  load_op2.x = x_op.y;
  load_op2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.axis = {z0.id, z1.id, z2.id};
  *load_op2.y.repeats = {s0, s1_2, s2};
  *load_op2.y.strides = {s1_2 * s2, s2, One};

  // concat_op.x1 = load_op.y;
  // concat_op.x2 = load_op2.y;
  concat_op.x = {load_op.y, load_op2.y};
  concat_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  *concat_op.y.axis = {z0.id, z1.id, z2.id};
  *concat_op.y.repeats = {s0, s1_1 + s1_2, s2};
  *concat_op.y.strides = {(s1_1 + s1_2) * s2, s2, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  load->outputs[0].attr.vectorized_strides = {s1_1 * s2, s2, One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT16;
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
  load2->outputs[0].attr.vectorized_strides = {s1_2 * s2, s2, One};
  load2->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 2;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 2;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto concat = graph.FindNode("concat");
  concat->attr.api.unit = af::ComputeUnit::kUnitVector;
  concat->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  concat->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id};
  concat->outputs[0].attr.vectorized_strides = {(s1_1 + s1_2) * s2, s2, One};
  concat->outputs[0].attr.dtype = ge::DT_FLOAT16;
  concat->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  concat->outputs[0].attr.mem.tensor_id = 3;
  concat->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  concat->outputs[0].attr.que.id = 3;
  concat->outputs[0].attr.opt.merge_scope = af::kIdNone;

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  // add load1 tensor
  EXPECT_EQ(tpipe.AddTensor(load->outputs[0]), 0);
  EXPECT_EQ(tpipe.AddTensor(load2->outputs[0]), 0);

  // add add tensor
  EXPECT_EQ(tpipe.AddTensor(concat->outputs[0]), 0);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1_1));
  tiler.AddSizeVar(af::SizeVar(s1_2));
  tiler.AddSizeVar(af::SizeVar(s2));

  codegen::ApiTensor x1, x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;

  codegen::ConcatApiCall call("Concat");
  EXPECT_EQ(call.Init(concat), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), 0);
  std::cout << result << std::endl;
  EXPECT_EQ(
      result,
      std::string{"uint32_t concat_dim = 1;\nconst ConcatParams<half, 3> dst_params = {\n.shape  = "
                  "{static_cast<uint32_t>((t->s0)/(1)), static_cast<uint32_t>(((t->s1_1 + t->s1_2))/(1)), "
                  "static_cast<uint32_t>((t->s2)/(1))},\n.stride = {static_cast<uint32_t>((((t->s1_1 + t->s1_2) * "
                  "t->s2))/(1)), static_cast<uint32_t>((t->s2)/(1)), 1},\n.tensor = "
                  "&local_3,\n};\nconst ConcatParams<half, 3> srcs_params[2] = "
                  "{\n{\n.shape  = {static_cast<uint32_t>((t->s0)/(1)), static_cast<uint32_t>((t->s1_1)/(1)), "
                  "static_cast<uint32_t>((t->s2)/(1)), },\n.stride = {static_cast<uint32_t>(((t->s1_1 * t->s2))/(1)), "
                  "static_cast<uint32_t>((t->s2)/(1)), 1},\n.tensor = &local_0,\n},\n{\n.shape  "
                  "= {static_cast<uint32_t>((t->s0)/(1)), static_cast<uint32_t>((t->s1_2)/(1)), "
                  "static_cast<uint32_t>((t->s2)/(1)), "
                  "},\n.stride = {static_cast<uint32_t>(((t->s1_2 * t->s2))/(1)), static_cast<uint32_t>((t->s2)/(1)), "
                  "1},\n.tensor = "
                  "&local_2,\n},\n};\nConcatExtend<half, 3, 2>(dst_params, srcs_params, concat_dim, tmp_buf_0);\n"});
}

TEST(ConcatApiCallTest, ConcatApicallWithSmallTail) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1_1 = graph.CreateSizeVar(1);
  auto s1_2 = graph.CreateSizeVar(2);

  af::Expression pad = af::Symbol(16);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1_1 + s1_2);

  Data x_op("x", graph);
  Load load_op("load");
  Load load_op2("load2");
  af::ascir_op::Concat concat_op("concat");

  graph.AddNode(load_op);
  graph.AddNode(load_op2);
  // graph.AddNode(concat_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1_1};
  *load_op.y.strides = {s1_1, One};

  load_op2.x = x_op.y;
  load_op2.attr.sched.axis = {z0.id, z1.id};
  *load_op2.y.axis = {z0.id, z1.id};
  *load_op2.y.repeats = {s0, s1_2};
  *load_op2.y.strides = {s1_2, One};

  // concat_op.x1 = load_op.y;
  // concat_op.x2 = load_op2.y;
  concat_op.x = {load_op.y, load_op2.y};
  concat_op.attr.sched.axis = {z0.id, z1.id};
  *concat_op.y.axis = {z0.id, z1.id};
  *concat_op.y.repeats = {s0, s1_1 + s1_2};
  *concat_op.y.strides = {(s1_1 + s1_2), One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.vectorized_strides = {pad, One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT16;
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
  load2->outputs[0].attr.vectorized_axis = {
      z0.id,
      z1.id,
  };
  load2->outputs[0].attr.vectorized_strides = {pad, One};
  load2->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 2;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 2;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto concat = graph.FindNode("concat");
  concat->attr.api.unit = af::ComputeUnit::kUnitVector;
  concat->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  concat->outputs[0].attr.vectorized_strides = {(s1_1 + s1_2), One};
  concat->outputs[0].attr.dtype = ge::DT_FLOAT16;
  concat->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  concat->outputs[0].attr.mem.tensor_id = 3;
  concat->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  concat->outputs[0].attr.que.id = 3;
  concat->outputs[0].attr.opt.merge_scope = af::kIdNone;
  af::AttrUtils::SetBool(concat->GetOpDesc(), "_concat_small_tail", true);
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  // add load1 tensor
  EXPECT_EQ(tpipe.AddTensor(load->outputs[0]), 0);
  EXPECT_EQ(tpipe.AddTensor(load2->outputs[0]), 0);

  // add add tensor
  EXPECT_EQ(tpipe.AddTensor(concat->outputs[0]), 0);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1_1));
  tiler.AddSizeVar(af::SizeVar(s1_2));

  codegen::ApiTensor x1, x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;
  concat->attr.tmp_buffers.emplace_back(af::TmpBuffer{af::TmpBufDesc{af::Symbol(16 * 1024), -1}, {}, 0});
  codegen::ConcatApiCall call("Concat");
  EXPECT_EQ(call.Init(concat), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), 0);
  std::cout << result << std::endl;
  EXPECT_EQ(result,
            "ConcatInputList<half, 2> input_list {\n"
            "  .src_tensor_base_addrs = {(half *)local_0.GetPhyAddr(), (half *)local_2.GetPhyAddr(), },\n"
            "  .src_tensors = {&local_0, &local_2, },\n"
            "};\n"
            "ConcatContextDiffDimPadded<half, 2> concat_context;\n"
            "concat_context.total_row_num = t->s0;\n"
            "concat_context.input_list = &input_list;\n"
            "constexpr ConcatTiling<2> tiling {\n"
            "  .gcd = 1, \n"
            "  .tmp_buf_size = 16384, \n"
            "  .dst_dim_size = 3, \n"
            "  .dst_row_num_unit = 3, \n"
            "  .max_repeat_times = 5, \n"
            "  .max_element_num = 3840, \n"
            "  .max_orig_row_num = 1280, \n"
            "  .per_repeat_size = 768, \n"
            "  .first_copy_repeat_times = 80, \n"
            "  .last_trans_repeat_times = 15, \n"
            "  .src_dim_sizes = {1, 2, },\n"
            "  .src_strides = {20480, 20480, },\n"
            "  .src_buffer_offsets = {0, 1280, },\n"
            "  .gather_mask_repeat_strides = {1, 1, },\n"
            "  .gather_mask_dim_sizes = {1, 2, }\n"
            "};\n"
            "ConcatExtendV2(concat_context, tiling, local_3, tmp_buf_0);\n");
}

TEST(ConcatApiCallTest, ConcatApicallWithSameTail) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1_1 = graph.CreateSizeVar(1);
  auto s1_2 = graph.CreateSizeVar(1);

  af::Expression pad = af::Symbol(16);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1_1 + s1_2);

  Data x_op("x", graph);
  Load load_op("load");
  Load load_op2("load2");
  af::ascir_op::Concat concat_op("concat");

  graph.AddNode(load_op);
  graph.AddNode(load_op2);
  // graph.AddNode(concat_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1_1};
  *load_op.y.strides = {s1_1, One};

  load_op2.x = x_op.y;
  load_op2.attr.sched.axis = {z0.id, z1.id};
  *load_op2.y.axis = {z0.id, z1.id};
  *load_op2.y.repeats = {s0, s1_2};
  *load_op2.y.strides = {s1_2, One};

  // concat_op.x1 = load_op.y;
  // concat_op.x2 = load_op2.y;
  concat_op.x = {load_op.y, load_op2.y};
  concat_op.attr.sched.axis = {z0.id, z1.id};
  *concat_op.y.axis = {z0.id, z1.id};
  *concat_op.y.repeats = {s0, s1_1 + s1_2};
  *concat_op.y.strides = {(s1_1 + s1_2), One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.vectorized_strides = {pad, One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT16;
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
  load2->outputs[0].attr.vectorized_axis = {
      z0.id,
      z1.id,
  };
  load2->outputs[0].attr.vectorized_strides = {pad, One};
  load2->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 2;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 2;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto concat = graph.FindNode("concat");
  concat->attr.api.unit = af::ComputeUnit::kUnitVector;
  concat->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  concat->outputs[0].attr.vectorized_strides = {af::sym::Align(s1_1 + s1_2, 16), One};
  concat->outputs[0].attr.dtype = ge::DT_FLOAT16;
  concat->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  concat->outputs[0].attr.mem.tensor_id = 3;
  concat->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  concat->outputs[0].attr.que.id = 3;
  concat->outputs[0].attr.opt.merge_scope = af::kIdNone;
  af::AttrUtils::SetBool(concat->GetOpDesc(), "_concat_small_tail", true);
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  // add load1 tensor
  EXPECT_EQ(tpipe.AddTensor(load->outputs[0]), 0);
  EXPECT_EQ(tpipe.AddTensor(load2->outputs[0]), 0);

  // add add tensor
  EXPECT_EQ(tpipe.AddTensor(concat->outputs[0]), 0);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1_1));
  tiler.AddSizeVar(af::SizeVar(s1_2));

  codegen::ApiTensor x1, x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;

  concat->attr.tmp_buffers.emplace_back(af::TmpBuffer{af::TmpBufDesc{af::Symbol(16 * 1024), -1}, {}, 0});
  codegen::ConcatApiCall call("Concat");
  EXPECT_EQ(call.Init(concat), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), 0);
  std::cout << result << std::endl;
  EXPECT_EQ(result,
            "ConcatInputList<half, 2> input_list {\n"
            "  .src_tensor_base_addrs = {(half *)local_0.GetPhyAddr(), (half *)local_2.GetPhyAddr(), },\n"
            "  .src_tensors = {&local_0, &local_2, },\n"
            "};\n"
            "ConcatContextDiffDimPadded<half, 2> concat_context;\n"
            "concat_context.total_row_num = t->s0;\n"
            "concat_context.input_list = &input_list;\n"
            "constexpr ConcatTiling<2> tiling {\n"
            "  .gcd = 1, \n"
            "  .tmp_buf_size = 16384, \n"
            "  .dst_dim_size = 16, \n"
            "  .dst_row_num_unit = 16, \n"
            "  .max_repeat_times = 1, \n"
            "  .max_element_num = 4096, \n"
            "  .max_orig_row_num = 256, \n"
            "  .per_repeat_size = 4096, \n"
            "  .first_copy_repeat_times = 16, \n"
            "  .last_trans_repeat_times = 16, \n"
            "  .src_dim_sizes = {1, 1, },\n"
            "  .src_strides = {4096, 4096, },\n"
            "  .src_buffer_offsets = {0, 256, },\n"
            "  .gather_mask_repeat_strides = {1, 1, },\n"
            "  .gather_mask_dim_sizes = {1, 1, }\n"
            "};\n"
            "ConcatExtendV2(concat_context, tiling, local_3, tmp_buf_0);\n");
}

TEST(ConcatApiCallTest, ConcatApicallWithSmallTailUnpaded) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1_1 = graph.CreateSizeVar(1);
  auto s1_2 = graph.CreateSizeVar(2);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1_1 + s1_2);

  Data x_op("x", graph);
  Load load_op("load");
  Load load_op2("load2");
  af::ascir_op::Concat concat_op("concat");

  graph.AddNode(load_op);
  graph.AddNode(load_op2);
  // graph.AddNode(concat_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1_1};
  *load_op.y.strides = {s1_1, One};

  load_op2.x = x_op.y;
  load_op2.attr.sched.axis = {z0.id, z1.id};
  *load_op2.y.axis = {z0.id, z1.id};
  *load_op2.y.repeats = {s0, s1_2};
  *load_op2.y.strides = {s1_2, One};

  // concat_op.x1 = load_op.y;
  // concat_op.x2 = load_op2.y;
  concat_op.x = {load_op.y, load_op2.y};
  concat_op.attr.sched.axis = {z0.id, z1.id};
  *concat_op.y.axis = {z0.id, z1.id};
  *concat_op.y.repeats = {s0, s1_1 + s1_2};
  *concat_op.y.strides = {(s1_1 + s1_2), One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.vectorized_strides = {s1_1, One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT16;
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
  load2->outputs[0].attr.vectorized_axis = {
      z0.id,
      z1.id,
  };
  load2->outputs[0].attr.vectorized_strides = {s1_2, One};
  load2->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 2;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 2;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto concat = graph.FindNode("concat");
  concat->attr.api.unit = af::ComputeUnit::kUnitVector;
  concat->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  concat->outputs[0].attr.vectorized_strides = {(s1_1 + s1_2), One};
  concat->outputs[0].attr.dtype = ge::DT_FLOAT16;
  concat->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  concat->outputs[0].attr.mem.tensor_id = 3;
  concat->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  concat->outputs[0].attr.que.id = 3;
  concat->outputs[0].attr.opt.merge_scope = af::kIdNone;
  af::AttrUtils::SetBool(concat->GetOpDesc(), "_concat_small_tail", true);
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  // add load1 tensor
  EXPECT_EQ(tpipe.AddTensor(load->outputs[0]), 0);
  EXPECT_EQ(tpipe.AddTensor(load2->outputs[0]), 0);

  // add add tensor
  EXPECT_EQ(tpipe.AddTensor(concat->outputs[0]), 0);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1_1));
  tiler.AddSizeVar(af::SizeVar(s1_2));

  codegen::ApiTensor x1, x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;

  codegen::ConcatApiCall call("Concat");
  concat->attr.tmp_buffers.emplace_back(af::TmpBuffer{af::TmpBufDesc{af::Symbol(16 * 1024), -1}, {}, 0});
  EXPECT_EQ(call.Init(concat), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), 0);
  std::cout << result << std::endl;
  EXPECT_TRUE(result.find("ConcatContextDiffDim<") != std::string::npos);
}

TEST(ConcatApiCallTest, ConcatApicallWithSmallTailUnpadedDynamic) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1_1 = graph.CreateSizeVar("s1_1");
  auto s1_2 = graph.CreateSizeVar("s1_2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1_1 + s1_2);

  Data x_op("x", graph);
  Load load_op("load");
  Load load_op2("load2");
  af::ascir_op::Concat concat_op("concat");

  graph.AddNode(load_op);
  graph.AddNode(load_op2);
  // graph.AddNode(concat_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1_1};
  *load_op.y.strides = {s1_1, One};

  load_op2.x = x_op.y;
  load_op2.attr.sched.axis = {z0.id, z1.id};
  *load_op2.y.axis = {z0.id, z1.id};
  *load_op2.y.repeats = {s0, s1_2};
  *load_op2.y.strides = {s1_2, One};

  // concat_op.x1 = load_op.y;
  // concat_op.x2 = load_op2.y;
  concat_op.x = {load_op.y, load_op2.y};
  concat_op.attr.sched.axis = {z0.id, z1.id};
  *concat_op.y.axis = {z0.id, z1.id};
  *concat_op.y.repeats = {s0, s1_1 + s1_2};
  *concat_op.y.strides = {(s1_1 + s1_2), One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.vectorized_strides = {s1_1, One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT16;
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
  load2->outputs[0].attr.vectorized_axis = {
      z0.id,
      z1.id,
  };
  load2->outputs[0].attr.vectorized_strides = {s1_2, One};
  load2->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 2;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 2;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto concat = graph.FindNode("concat");
  concat->attr.api.unit = af::ComputeUnit::kUnitVector;
  concat->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  concat->outputs[0].attr.vectorized_strides = {(s1_1 + s1_2), One};
  concat->outputs[0].attr.dtype = ge::DT_FLOAT16;
  concat->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  concat->outputs[0].attr.mem.tensor_id = 3;
  concat->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  concat->outputs[0].attr.que.id = 3;
  concat->outputs[0].attr.opt.merge_scope = af::kIdNone;
  af::AttrUtils::SetBool(concat->GetOpDesc(), "_concat_small_tail", true);
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  // add load1 tensor
  EXPECT_EQ(tpipe.AddTensor(load->outputs[0]), 0);
  EXPECT_EQ(tpipe.AddTensor(load2->outputs[0]), 0);

  // add add tensor
  EXPECT_EQ(tpipe.AddTensor(concat->outputs[0]), 0);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1_1));
  tiler.AddSizeVar(af::SizeVar(s1_2));

  codegen::ApiTensor x1, x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;

  codegen::ConcatApiCall call("Concat");
  concat->attr.tmp_buffers.emplace_back(af::TmpBuffer{af::TmpBufDesc{af::Symbol(16 * 1024), -1}, {}, 0});
  EXPECT_EQ(call.Init(concat), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), 0);
  std::cout << result << std::endl;
  EXPECT_TRUE(result.find("ConcatShape<2>") != std::string::npos);
  EXPECT_TRUE(result.find("ConcatContextDiffDim<") != std::string::npos);
  EXPECT_TRUE(result.find("ConcatExtendV2Dyn") != std::string::npos);
}

TEST(ConcatApiCallTest, ConcatApicallWithSmallTailPadedDynamic) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1_1 = graph.CreateSizeVar("s1_1");
  auto s1_2 = graph.CreateSizeVar("s1_2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1_1 + s1_2);

  Data x_op("x", graph);
  Load load_op("load");
  Load load_op2("load2");
  af::ascir_op::Concat concat_op("concat");

  graph.AddNode(load_op);
  graph.AddNode(load_op2);
  // graph.AddNode(concat_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1_1};
  *load_op.y.strides = {s1_1, One};

  load_op2.x = x_op.y;
  load_op2.attr.sched.axis = {z0.id, z1.id};
  *load_op2.y.axis = {z0.id, z1.id};
  *load_op2.y.repeats = {s0, s1_2};
  *load_op2.y.strides = {s1_2, One};

  // concat_op.x1 = load_op.y;
  // concat_op.x2 = load_op2.y;
  concat_op.x = {load_op.y, load_op2.y};
  concat_op.attr.sched.axis = {z0.id, z1.id};
  *concat_op.y.axis = {z0.id, z1.id};
  *concat_op.y.repeats = {s0, s1_1 + s1_2};
  *concat_op.y.strides = {(s1_1 + s1_2), One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  load->outputs[0].attr.vectorized_strides = {af::sym::Align(s1_1, 16), One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT16;
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
  load2->outputs[0].attr.vectorized_axis = {
      z0.id,
      z1.id,
  };
  load2->outputs[0].attr.vectorized_strides = {s1_2, One};
  load2->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 2;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 2;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto concat = graph.FindNode("concat");
  concat->attr.api.unit = af::ComputeUnit::kUnitVector;
  concat->outputs[0].attr.vectorized_axis = {z0.id, z1.id};
  concat->outputs[0].attr.vectorized_strides = {s1_1 + s1_2, One};
  concat->outputs[0].attr.dtype = ge::DT_FLOAT16;
  concat->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  concat->outputs[0].attr.mem.tensor_id = 3;
  concat->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  concat->outputs[0].attr.que.id = 3;
  concat->outputs[0].attr.opt.merge_scope = af::kIdNone;
  af::AttrUtils::SetBool(concat->GetOpDesc(), "_concat_small_tail", true);
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  // add load1 tensor
  EXPECT_EQ(tpipe.AddTensor(load->outputs[0]), 0);
  EXPECT_EQ(tpipe.AddTensor(load2->outputs[0]), 0);

  // add add tensor
  EXPECT_EQ(tpipe.AddTensor(concat->outputs[0]), 0);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1_1));
  tiler.AddSizeVar(af::SizeVar(s1_2));

  codegen::ApiTensor x1, x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;

  codegen::ConcatApiCall call("Concat");
  concat->attr.tmp_buffers.emplace_back(af::TmpBuffer{af::TmpBufDesc{af::Symbol(16 * 1024), -1}, {}, 0});
  EXPECT_EQ(call.Init(concat), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), 0);
  std::cout << result << std::endl;
  EXPECT_TRUE(result.find("ConcatShape<2>") != std::string::npos);
  EXPECT_TRUE(result.find("ConcatContextDiffDimPadded<") != std::string::npos);
  EXPECT_TRUE(result.find("ConcatExtendV2Dyn") != std::string::npos);
}

TEST(ConcatApiCallTest, ConcatAllAligned) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar(1);
  auto s2_1 = graph.CreateSizeVar(8);
  auto s2_2 = graph.CreateSizeVar(16);
  auto s3 = graph.CreateSizeVar(2);

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
  load->outputs[0].attr.vectorized_strides = {s2_1 * s3, Zero, s3, One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT16;
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
  load2->outputs[0].attr.vectorized_strides = {s2_2 * s3, Zero, s3, One};
  load2->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 2;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 2;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto concat = graph.FindNode("concat");
  concat->attr.api.unit = af::ComputeUnit::kUnitVector;
  concat->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id};
  concat->outputs[0].attr.vectorized_strides = {(s2_1 + s2_2) * s3, Zero, s3, One};
  concat->outputs[0].attr.dtype = ge::DT_FLOAT16;
  concat->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  concat->outputs[0].attr.mem.tensor_id = 3;
  concat->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  concat->outputs[0].attr.que.id = 3;
  concat->outputs[0].attr.opt.merge_scope = af::kIdNone;
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  // add load1 tensor
  EXPECT_EQ(tpipe.AddTensor(load->outputs[0]), 0);
  EXPECT_EQ(tpipe.AddTensor(load2->outputs[0]), 0);

  // add add tensor
  EXPECT_EQ(tpipe.AddTensor(concat->outputs[0]), 0);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddAxis(z3);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2_1));
  tiler.AddSizeVar(af::SizeVar(s2_2));
  tiler.AddSizeVar(af::SizeVar(s3));

  codegen::ApiTensor x1, x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;

  codegen::ConcatApiCall call("Concat");
  concat->attr.tmp_buffers.emplace_back(af::TmpBuffer{af::TmpBufDesc{af::Symbol(16 * 1024), -1}, {}, 0});
  EXPECT_EQ(call.Init(concat), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), 0);
  EXPECT_EQ(result,
            "constexpr ConcatTilingAllAligned<2> concat_tiling {\n"
            "  .dst_col_size = 48,\n"
            "  .src_col_sizes = { 16, 32, },\n"
            "  .dst_offsets = { 0, 16, },\n"
            "};\n"
            "LocalTensor<half> concat_src_tensors[] { local_0, local_2, };\n"
            "ConcatAllAligned<half, 2>(t->s0, concat_tiling, local_3, concat_src_tensors);\n");
}

TEST(ConcatApiCallTest, ConcatAllAligned_Padded) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar(1);
  auto s2_1 = graph.CreateSizeVar(8);
  auto s2_2 = graph.CreateSizeVar(16);
  auto s3 = graph.CreateSizeVar(2);

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
  *load_op.y.strides = {s2_1 * af::Symbol(16), Zero, af::Symbol(16), One};

  load_op2.x = x_op.y;
  load_op2.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op2.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op2.y.repeats = {s0, s1, s2_2, s3};
  *load_op2.y.strides = {s2_2 * af::Symbol(16), Zero, af::Symbol(16), One};

  // concat_op.x1 = load_op.y;
  // concat_op.x2 = load_op2.y;
  concat_op.x = {load_op.y, load_op2.y};
  concat_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *concat_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *concat_op.y.repeats = {s0, s1, s2_1 + s2_2, s3};
  *concat_op.y.strides = {(s2_1 + s2_2) * af::Symbol(16), Zero, af::Symbol(16), One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id};
  load->outputs[0].attr.vectorized_strides = {s2_1 * af::Symbol(16), Zero, af::Symbol(16), One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT16;
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
  load2->outputs[0].attr.vectorized_strides = {s2_2 * af::Symbol(16), Zero, af::Symbol(16), One};
  load2->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 2;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 2;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto concat = graph.FindNode("concat");
  concat->attr.api.unit = af::ComputeUnit::kUnitVector;
  concat->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id};
  concat->outputs[0].attr.vectorized_strides = {(s2_1 + s2_2) * af::Symbol(16), Zero, af::Symbol(16), One};
  concat->outputs[0].attr.dtype = ge::DT_FLOAT16;
  concat->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  concat->outputs[0].attr.mem.tensor_id = 3;
  concat->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  concat->outputs[0].attr.que.id = 3;
  concat->outputs[0].attr.opt.merge_scope = af::kIdNone;
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  // add load1 tensor
  EXPECT_EQ(tpipe.AddTensor(load->outputs[0]), 0);
  EXPECT_EQ(tpipe.AddTensor(load2->outputs[0]), 0);

  // add add tensor
  EXPECT_EQ(tpipe.AddTensor(concat->outputs[0]), 0);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddAxis(z3);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2_1));
  tiler.AddSizeVar(af::SizeVar(s2_2));
  tiler.AddSizeVar(af::SizeVar(s3));

  codegen::ApiTensor x1, x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;

  codegen::ConcatApiCall call("Concat");
  concat->attr.tmp_buffers.emplace_back(af::TmpBuffer{af::TmpBufDesc{af::Symbol(16 * 1024), -1}, {}, 0});
  EXPECT_EQ(call.Init(concat), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), 0);
  EXPECT_EQ(result,
            "constexpr ConcatTilingAllAligned<2> concat_tiling {\n"
            "  .dst_col_size = 384,\n"
            "  .src_col_sizes = { 128, 256, },\n"
            "  .dst_offsets = { 0, 128, },\n"
            "};\n"
            "LocalTensor<half> concat_src_tensors[] { local_0, local_2, };\n"
            "ConcatAllAligned<half, 2>(t->s0, concat_tiling, local_3, concat_src_tensors);\n");
}

TEST(ConcatApiCallTest, ConcatAllAligned_Pack_Padded) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar(1);
  auto s2_1 = graph.CreateSizeVar(1);
  auto s2_2 = graph.CreateSizeVar(1);
  auto s3 = graph.CreateSizeVar(2);

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
  *load_op.y.strides = {s2_1 * af::Symbol(16), Zero, One};

  load_op2.x = x_op.y;
  load_op2.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op2.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_op2.y.repeats = {s0, s1, s2_2, s3};
  *load_op2.y.strides = {s2_2 * af::Symbol(16), Zero, Zero, One};

  // concat_op.x1 = load_op.y;
  // concat_op.x2 = load_op2.y;
  concat_op.x = {load_op.y, load_op2.y};
  concat_op.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *concat_op.y.axis = {z0.id, z1.id, z2.id, z3.id};
  *concat_op.y.repeats = {s0, s1, s2_1 + s2_2, s3};
  *concat_op.y.strides = {(s2_1 + s2_2) * af::Symbol(16), Zero, Zero, One};

  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id};
  load->outputs[0].attr.vectorized_strides = {s2_1 * af::Symbol(16), Zero, Zero, One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT16;
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
  load2->outputs[0].attr.vectorized_strides = {s2_2 * af::Symbol(16), Zero, Zero, One};
  load2->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 2;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 2;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto concat = graph.FindNode("concat");
  concat->attr.api.unit = af::ComputeUnit::kUnitVector;
  concat->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id};
  concat->outputs[0].attr.vectorized_strides = {(s2_1 + s2_2) * af::Symbol(16), Zero, af::Symbol(16), One};
  concat->outputs[0].attr.dtype = ge::DT_FLOAT16;
  concat->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  concat->outputs[0].attr.mem.tensor_id = 3;
  concat->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  concat->outputs[0].attr.que.id = 3;
  concat->outputs[0].attr.opt.merge_scope = af::kIdNone;
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  // add load1 tensor
  EXPECT_EQ(tpipe.AddTensor(load->outputs[0]), 0);
  EXPECT_EQ(tpipe.AddTensor(load2->outputs[0]), 0);

  // add add tensor
  EXPECT_EQ(tpipe.AddTensor(concat->outputs[0]), 0);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddAxis(z3);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2_1));
  tiler.AddSizeVar(af::SizeVar(s2_2));
  tiler.AddSizeVar(af::SizeVar(s3));

  codegen::ApiTensor x1, x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;

  codegen::ConcatApiCall call("Concat");
  concat->attr.tmp_buffers.emplace_back(af::TmpBuffer{af::TmpBufDesc{af::Symbol(16 * 1024), -1}, {}, 0});
  EXPECT_EQ(call.Init(concat), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), 0);
  EXPECT_EQ(result,
            "constexpr ConcatTilingAllAligned<2> concat_tiling {\n"
            "  .dst_col_size = 32,\n"
            "  .src_col_sizes = { 16, 16, },\n"
            "  .dst_offsets = { 0, 16, },\n"
            "};\n"
            "LocalTensor<half> concat_src_tensors[] { local_0, local_2, };\n"
            "ConcatAllAligned<half, 2>(t->s0, concat_tiling, local_3, concat_src_tensors);\n");
}

TEST(ConcatApiCallTest, Padded_SecondLastDim) {
  af::AscGraph graph("test_graph");

  // [s0, 1, 2, 2] [s0, 1, 4, 2]
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar(1);
  auto s2_1 = graph.CreateSizeVar(2);
  auto s2_2 = graph.CreateSizeVar(4);
  auto s3 = graph.CreateSizeVar(2);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2_1 + s2_2);
  auto z3 = graph.CreateAxis("z3", s3);

  Data x_op("x", graph);
  Load load_op("load");
  Load load_op2("load2");
  af::ascir_op::Concat concat_op("concat");

  auto kAlignment = af::Symbol(16);

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
  load->outputs[0].attr.vectorized_strides = {s2_1 * kAlignment, Zero, kAlignment, One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT16;
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
  load2->outputs[0].attr.vectorized_strides = {s2_2 * kAlignment, Zero, kAlignment, One};
  load2->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 2;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 2;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto concat = graph.FindNode("concat");
  concat->attr.api.unit = af::ComputeUnit::kUnitVector;
  concat->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id};
  concat->outputs[0].attr.vectorized_strides = {(s2_1 + s2_2) * s3, Zero, s3, One};
  concat->outputs[0].attr.dtype = ge::DT_FLOAT16;
  concat->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  concat->outputs[0].attr.mem.tensor_id = 3;
  concat->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  concat->outputs[0].attr.que.id = 3;
  concat->outputs[0].attr.opt.merge_scope = af::kIdNone;
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  // add load1 tensor
  EXPECT_EQ(tpipe.AddTensor(load->outputs[0]), 0);
  EXPECT_EQ(tpipe.AddTensor(load2->outputs[0]), 0);

  // add add tensor
  EXPECT_EQ(tpipe.AddTensor(concat->outputs[0]), 0);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddAxis(z3);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2_1));
  tiler.AddSizeVar(af::SizeVar(s2_2));
  tiler.AddSizeVar(af::SizeVar(s3));

  codegen::ApiTensor x1, x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;

  af::AttrUtils::SetBool(concat->GetOpDesc(), "_concat_small_tail", true);

  codegen::ConcatApiCall call("Concat");
  concat->attr.tmp_buffers.emplace_back(af::TmpBuffer{af::TmpBufDesc{af::Symbol(16 * 1024), -1}, {}, 0});
  EXPECT_EQ(call.Init(concat), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), 0);
  std::cout << result << std::endl;
  // concat ([1, 2, 2], [1, 4, 2]) -> [1, 6, 2]
  EXPECT_EQ(result,
            "ConcatInputList<half, 2> input_list {\n"
            "  .src_tensor_base_addrs = {(half *)local_0.GetPhyAddr(), (half *)local_2.GetPhyAddr(), },\n"
            "  .src_tensors = {&local_0, &local_2, },\n"
            "};\n"
            "ConcatContextDiffDimPadded<half, 2> concat_context;\n"
            "concat_context.total_row_num = t->s0;\n"
            "concat_context.input_list = &input_list;\n"
            "constexpr ConcatTiling<2> tiling {\n"
            "  .gcd = 4, \n"
            "  .tmp_buf_size = 16384, \n"
            "  .dst_dim_size = 12, \n"
            "  .dst_row_num_unit = 12, \n"
            "  .max_repeat_times = 5, \n"
            "  .max_element_num = 3840, \n"
            "  .max_orig_row_num = 320, \n"
            "  .per_repeat_size = 768, \n"
            "  .first_copy_repeat_times = 20, \n"
            "  .last_trans_repeat_times = 15, \n"
            "  .src_dim_sizes = {4, 8, },\n"
            "  .src_strides = {10240, 20480, },\n"
            "  .src_buffer_offsets = {0, 1280, },\n"
            "  .gather_mask_repeat_strides = {1, 1, },\n"
            "  .gather_mask_dim_sizes = {2, 2, }\n"
            "};\n"
            "ConcatExtendV2(concat_context, tiling, local_3, tmp_buf_0);\n");
}

TEST(ConcatApiCallTest, NotPadded_SecondLastDim) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar(1);
  auto s2_1 = graph.CreateSizeVar(2);
  auto s2_2 = graph.CreateSizeVar(4);
  auto s3 = graph.CreateSizeVar(2);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2_1 + s2_2);
  auto z3 = graph.CreateAxis("z3", s3);

  Data x_op("x", graph);
  Load load_op("load");
  Load load_op2("load2");
  af::ascir_op::Concat concat_op("concat");

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
  load->outputs[0].attr.vectorized_strides = {s2_1 * s3, Zero, s3, One};
  load->outputs[0].attr.dtype = ge::DT_FLOAT16;
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
  load2->outputs[0].attr.vectorized_strides = {s2_2 * s3, Zero, s3, One};
  load2->outputs[0].attr.dtype = ge::DT_FLOAT16;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.tensor_id = 2;
  load2->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load2->outputs[0].attr.que.id = 2;
  load2->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto concat = graph.FindNode("concat");
  concat->attr.api.unit = af::ComputeUnit::kUnitVector;
  concat->outputs[0].attr.vectorized_axis = {z0.id, z1.id, z2.id, z3.id};
  concat->outputs[0].attr.vectorized_strides = {(s2_1 + s2_2) * s3, Zero, s3, One};
  concat->outputs[0].attr.dtype = ge::DT_FLOAT16;
  concat->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  concat->outputs[0].attr.mem.tensor_id = 3;
  concat->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  concat->outputs[0].attr.que.id = 3;
  concat->outputs[0].attr.opt.merge_scope = af::kIdNone;
  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.CollectQues(graph);
  // add load1 tensor
  EXPECT_EQ(tpipe.AddTensor(load->outputs[0]), 0);
  EXPECT_EQ(tpipe.AddTensor(load2->outputs[0]), 0);

  // add add tensor
  EXPECT_EQ(tpipe.AddTensor(concat->outputs[0]), 0);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddAxis(z3);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2_1));
  tiler.AddSizeVar(af::SizeVar(s2_2));
  tiler.AddSizeVar(af::SizeVar(s3));

  codegen::ApiTensor x1, x2;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  x2.id = load2->outputs[0].attr.mem.tensor_id;

  af::AttrUtils::SetBool(concat->GetOpDesc(), "_concat_small_tail", true);

  codegen::ConcatApiCall call("Concat");
  concat->attr.tmp_buffers.emplace_back(af::TmpBuffer{af::TmpBufDesc{af::Symbol(16 * 1024), -1}, {}, 0});
  EXPECT_EQ(call.Init(concat), 0);
  call.inputs.push_back(&x1);
  call.inputs.push_back(&x2);

  std::string result;
  EXPECT_EQ(call.Generate(tpipe, vector<af::AxisId>{}, result), 0);
  std::cout << result << std::endl;
  // concat ([1, 2, 2], [1, 4, 2]) -> [1, 6, 2]
  EXPECT_EQ(result,
            "ConcatInputList<half, 2> input_list {\n"
            "  .src_tensor_base_addrs = {(half *)local_0.GetPhyAddr(), (half *)local_2.GetPhyAddr(), },\n"
            "};\n"
            "ConcatContextDiffDim<half, 2> concat_context;\n"
            "concat_context.total_row_num = t->s0;\n"
            "concat_context.input_list = &input_list;\n"
            "constexpr ConcatTiling<2> tiling {\n"
            "  .gcd = 4, \n"
            "  .tmp_buf_size = 16384, \n"
            "  .dst_dim_size = 12, \n"
            "  .dst_row_num_unit = 12, \n"
            "  .max_repeat_times = 5, \n"
            "  .max_element_num = 3840, \n"
            "  .max_orig_row_num = 320, \n"
            "  .per_repeat_size = 768, \n"
            "  .first_copy_repeat_times = 20, \n"
            "  .last_trans_repeat_times = 15, \n"
            "  .src_dim_sizes = {4, 8, },\n"
            "  .src_strides = {1280, 2560, },\n"
            "  .src_buffer_offsets = {0, 1280, },\n"
            "  .gather_mask_repeat_strides = {0, 0, },\n"
            "  .gather_mask_dim_sizes = {2, 2, }\n"
            "};\n"
            "ConcatExtendV2(concat_context, tiling, local_3, tmp_buf_0);\n");
}
