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
#include "elewise/unary_api_tmp_call.h"

using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;
using namespace codegen;

namespace {
void BuildLogicalNotCvStageGraph(af::AscGraph &graph, const af::Expression &s0, const af::Expression &s1,
                                 const af::Axis &z0, const af::Axis &z1);
void InitLogicalNotCvStageAttrs(af::AscGraph &graph, const af::Axis &z0, const af::Axis &z1);
}  // namespace

TEST(CodegenKernel, RegLogicalNotApiCall) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  BuildLogicalNotCvStageGraph(graph, s0, s1, z0, z1);
  InitLogicalNotCvStageAttrs(graph, z0, z1);

  auto load = graph.FindNode("load");
  auto logical_not = graph.FindNode("logical_not");

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(logical_not->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::UnaryApiTmpCall call("LogicalNotExtend");
  EXPECT_EQ(call.Init(logical_not), 0);
  call.inputs.push_back(&x1);
  std::string result;
  call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(result, std::string{"LogicalNotExtend(local_1[0], local_0[0], tmp_buf_0, local_0_actual_size);\n"});
}

namespace {
void BuildLogicalNotCvStageGraph(af::AscGraph &graph, const af::Expression &s0, const af::Expression &s1,
                                 const af::Axis &z0, const af::Axis &z1) {
  Data x_op("x", graph);
  Load load_op("load");
  af::ascir_op::LogicalNot logical_not_op("logical_not");
  graph.AddNode(load_op);
  graph.AddNode(logical_not_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id, z1.id};
  *load_op.y.axis = {z0.id, z1.id};
  *load_op.y.repeats = {s0, s1};
  *load_op.y.strides = {s1, One};
  logical_not_op.x = load_op.y;
  *logical_not_op.y.axis = {z0.id, z1.id};
  *logical_not_op.y.repeats = {s0, s1};
  *logical_not_op.y.strides = {s1, One};
}

void InitLogicalNotCvStageAttrs(af::AscGraph &graph, const af::Axis &z0, const af::Axis &z1) {
  auto load = graph.FindNode("load");
  load->attr.api.compute_type = af::ComputeType::kComputeLoad;
  load->attr.api.type = af::ApiType::kAPITypeCompute;
  load->attr.api.unit = af::ComputeUnit::kUnitMTE2;
  load->attr.sched.loop_axis = z0.id;
  load->outputs[0].attr.vectorized_axis = {z1.id};
  load->outputs[0].attr.vectorized_strides = {One};
  load->outputs[0].attr.dtype = af::DT_FLOAT;
  load->outputs[0].attr.mem.position = af::Position::kPositionVecIn;
  load->outputs[0].attr.mem.tensor_id = 0;
  load->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  load->outputs[0].attr.que.id = 1;
  load->outputs[0].attr.opt.merge_scope = af::kIdNone;

  auto logical_not = graph.FindNode("logical_not");
  logical_not->attr.api.compute_type = af::ComputeType::kComputeElewise;
  logical_not->attr.api.type = af::ApiType::kAPITypeCompute;
  logical_not->attr.api.unit = af::ComputeUnit::kUnitVector;
  logical_not->attr.sched.loop_axis = z0.id;
  logical_not->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  logical_not->outputs[0].attr.vectorized_axis = {z1.id};
  logical_not->outputs[0].attr.vectorized_strides = {One};
  logical_not->outputs[0].attr.dtype = af::DT_INT16;
  logical_not->outputs[0].attr.mem.position = af::Position::kPositionVecOut;
  logical_not->outputs[0].attr.mem.tensor_id = 1;
  logical_not->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
  logical_not->outputs[0].attr.que.id = 2;
  logical_not->outputs[0].attr.opt.merge_scope = af::kIdNone;
}

void InitLogicalNotCvStageTpipe(codegen::TPipe &tpipe, codegen::Tiler &tiler, const af::AscGraph &graph,
                                const af::AscNodePtr &load, const af::AscNodePtr &logical_not, const af::Expression &s0,
                                const af::Expression &s1, const af::Axis &z0, const af::Axis &z1) {
  tpipe.cv_fusion_type = ascir::CubeTemplateType::kUBFuse;
  tpipe.is_inductor = false;
  tpipe.CollectQues(graph);
  tpipe.AddTensor(load->outputs[0]);
  tpipe.AddTensor(logical_not->outputs[0]);

  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
}
}  // namespace

TEST(CodegenKernel, RegLogicalNotApiCallCVStage) {
  af::AscGraph graph("test_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  BuildLogicalNotCvStageGraph(graph, s0, s1, z0, z1);
  InitLogicalNotCvStageAttrs(graph, z0, z1);

  auto load = graph.FindNode("load");
  auto logical_not = graph.FindNode("logical_not");

  codegen::Tiler tiler;
  codegen::TPipe tpipe("tpipe", tiler);
  InitLogicalNotCvStageTpipe(tpipe, tiler, graph, load, logical_not, s0, s1, z0, z1);
  std::vector<af::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::ApiTensor x1;
  x1.id = load->outputs[0].attr.mem.tensor_id;
  codegen::UnaryApiTmpCall call("LogicalNotExtend");
  EXPECT_EQ(call.Init(logical_not), 0);
  call.api_call_context.stage = codegen::ComputeStage::kCVFuseStage1;
  call.inputs.push_back(&x1);
  std::string result;
  call.Generate(tpipe, current_axis, result);
  // CV rows have different physical pitches for FP32 input and INT16 output.
  EXPECT_EQ(result, std::string{"local_1_actual_size = curAivM * ((curAivN + 16 - 1) / 16 * 16);\n"
                                "for (uint32_t cv_row = 0; cv_row < ConvertToUint32(curAivM); cv_row++) {\n"
                                "  LogicalNotExtend(local_1[cv_row * ConvertToUint32(((curAivN + 16 - 1) / 16 * 16))], "
                                "local_0[cv_row * ConvertToUint32(((curAivN + 8 - 1) / 8 * 8))], "
                                "tmp_buf_0, ConvertToUint32(curAivN));\n"
                                "}\n"});
}
