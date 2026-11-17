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
#include <algorithm>
#include "ascir_ops.h"
#include "ascir_node_param/ascir_node_param.h"
#include "ascir_node_param/ascir_param_builder.h"
#include "codegen_kernel.h"
#include "utils/api_call_factory.h"
#include "reduce/reduce_api_call.h"
#include "reduce/reduce_api_call_base.h"

using namespace af::ops;
using namespace codegen;
using namespace af::ascir_op;
using namespace reduce_base;
using namespace testing;

namespace {
class ReduceApicallTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

ReduceSpecificParams BuildReduceParamForShadowTest() {
  ReduceSpecificParams params;
  params.valid = true;
  params.reduce_type = "Max";
  params.pattern = ReducePattern::kAR;
  params.merged_dims = {true, af::Symbol(3U), af::Symbol(5U)};
  params.merge_mode = ReduceMergeMode::kCopy;
  params.merge_size = af::Symbol(4U);
  params.merge_times = af::Symbol(2U);
  params.reuse = {true, true};
  return params;
}

ascir_param::ReduceNodeParams BuildReduceNodeParamForShadowTest() {
  ascir_param::ReduceNodeParams params;
  params.canonical_params = BuildReduceParamForShadowTest();
  params.exprs.merge_size = {true, {{params.canonical_params.merge_size, ascir_param::ParamExprRole::kSemantic}}};
  params.exprs.merge_times = {true, {{params.canonical_params.merge_times, ascir_param::ParamExprRole::kSemantic}}};
  return params;
}

ReduceCodegenShadowCheckInput MakeNonFatalShadowInput(af::AscNodePtr node) {
  ReduceCodegenShadowCheckInput input;
  input.api_name = "ReduceMax";
  input.node = node;
  return input;
}

void AddAllAxesToTiler(const af::AscGraph &graph, codegen::Tiler &tiler) {
  for (const auto &axis : graph.GetAllAxis()) {
    if (axis != nullptr) {
      EXPECT_EQ(tiler.AddAxis(*axis), af::SUCCESS);
    }
  }
}

void BuildRminBaseGraph(af::AscGraph &graph, const af::Axis &z0, const af::Axis &z1, const af::Expression &s0,
                        const af::Expression &s1) {
  af::ascir_op::Data x("x");
  graph.AddNode(x);
  x.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Load load("load");
  graph.AddNode(load);
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id};
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, s1};
  *load.y.strides = {s1, af::sym::kSymbolOne};

  af::ascir_op::Min rmin("rmin");
  graph.AddNode(rmin);
  rmin.x = load.y;
  rmin.attr.sched.axis = {z0.id, z1.id};
  *rmin.y.axis = {z0.id, z1.id};
  *rmin.y.repeats = {s0, af::sym::kSymbolOne};
  *rmin.y.strides = {af::sym::kSymbolOne, af::sym::kSymbolZero};
}

void ApplyRminTileSplit(af::AscGraph &graph, const af::Axis &z0, const af::Axis &z1, const af::AscNodePtr &load_node,
                        const af::AscNodePtr &rmin_node) {
  auto z0_split = graph.TileSplit(z0.id);
  auto z0_t = z0_split.first;
  auto z0_inner = z0_split.second;
  auto z0_block_split = graph.BlockSplit(z0_t->id);
  auto z0_block_outer = z0_block_split.first;
  auto z0_block_inner = z0_block_split.second;
  auto z1_split = graph.TileSplit(z1.id);
  auto z1_t = z1_split.first;
  auto z1_inner = z1_split.second;

  for (auto node : graph.GetAllNodes()) {
    if (IsOps<af::ascir_op::Data>(node)) {
      continue;
    }
    graph.ApplySplit(node, z0_t->id, z0_inner->id);
    graph.ApplySplit(node, z0_block_outer->id, z0_block_inner->id);
    graph.ApplySplit(node, z1_t->id, z1_inner->id);
    graph.ApplyReorder(node, {z0_block_outer->id, z0_block_inner->id, z1_t->id, z0_inner->id, z1_inner->id});
  }

  load_node->attr.sched.loop_axis = z1_t->id;
  load_node->outputs[0].attr.vectorized_axis = {z0_inner->id, z1_inner->id};
  load_node->outputs[0].attr.vectorized_strides = {graph.FindAxis(z1_inner->id)->size, af::sym::kSymbolOne};

  rmin_node->attr.sched.loop_axis = z1_t->id;
  rmin_node->outputs[0].attr.vectorized_axis = {z0_inner->id, z1_inner->id};
  rmin_node->outputs[0].attr.vectorized_strides = {af::sym::kSymbolOne, af::sym::kSymbolZero};
}

void BuildRminGraph(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  BuildRminBaseGraph(graph, z0, z1, s0, s1);

  auto load_node = graph.FindNode("load");
  auto rmin_node = graph.FindNode("rmin");
  ASSERT_NE(load_node, nullptr);
  ASSERT_NE(rmin_node, nullptr);
  load_node->outputs[0].attr.dtype = ge::DT_FLOAT;
  rmin_node->outputs[0].attr.dtype = ge::DT_FLOAT;
  ApplyRminTileSplit(graph, z0, z1, load_node, rmin_node);
}

void BuildOriginalParentAxisGraph(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  auto alias = graph.CreateAxis("z0_alias", af::Axis::kAxisTypeOriginal, s0, {z0.id}, af::kIdNone);
  auto loop_axis = graph.CreateAxis("z0_loop", af::Axis::kAxisTypeMerged, s0, {alias.id}, af::kIdNone);

  af::ascir_op::Data x("x");
  graph.AddNode(x);
  x.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Load load("load");
  graph.AddNode(load);
  load.x = x.y;
  load.attr.sched.axis = {loop_axis.id};
  load.attr.sched.loop_axis = loop_axis.id;
  *load.y.axis = {z0.id};
  *load.y.repeats = {s0};
  *load.y.strides = {af::sym::kSymbolOne};

  af::ascir_op::Min rmin("rmin");
  graph.AddNode(rmin);
  rmin.x = load.y;
  rmin.attr.sched.axis = {loop_axis.id};
  rmin.attr.sched.loop_axis = loop_axis.id;
  *rmin.y.axis = {z0.id};
  *rmin.y.repeats = {s0};
  *rmin.y.strides = {af::sym::kSymbolOne};

  auto load_node = graph.FindNode("load");
  auto rmin_node = graph.FindNode("rmin");
  ASSERT_NE(load_node, nullptr);
  ASSERT_NE(rmin_node, nullptr);
  load_node->outputs[0].attr.dtype = ge::DT_FLOAT;
  load_node->outputs[0].attr.vectorized_axis = {z0.id};
  load_node->outputs[0].attr.vectorized_strides = {af::sym::kSymbolOne};
  rmin_node->outputs[0].attr.dtype = ge::DT_FLOAT;
  rmin_node->outputs[0].attr.vectorized_axis = {z0.id};
  rmin_node->outputs[0].attr.vectorized_strides = {af::sym::kSymbolOne};
}

const af::AscTensor &GetPeerOutputTensor(const af::AscNodePtr &node, size_t input_index) {
  auto in_anchor = node->GetInDataAnchor(static_cast<int32_t>(input_index));
  EXPECT_NE(in_anchor, nullptr);
  auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
  EXPECT_NE(peer_out_anchor, nullptr);
  auto peer_node = std::dynamic_pointer_cast<af::AscNode>(peer_out_anchor->GetOwnerNode());
  EXPECT_NE(peer_node, nullptr);
  return peer_node->outputs[peer_out_anchor->GetIdx()];
}

void ExpectReduceShadowParamsMatch(const af::AscGraph &graph, const af::AscNodePtr &node, const std::string &api_name) {
  ASSERT_EQ(ascir_param::EnrichAscirGraphNodeParams(graph), af::SUCCESS);
  const auto params = ascir_param::GetAscirNodeParams(node);
  ASSERT_NE(params, nullptr);
  const auto *parser_params = ascir_param::GetSpecificParams<ascir_param::ReduceNodeParams>(*params);
  ASSERT_NE(parser_params, nullptr);

  codegen::Tiler tiler;
  AddAllAxesToTiler(graph, tiler);
  tiler.AddAxisSplitBAttr();
  codegen::TPipe tpipe("pipe", tiler);
  std::string input_dtype;
  const auto &input_tensor = GetPeerOutputTensor(node, 0U);
  ASSERT_EQ(codegen::Tensor::DtypeName(input_tensor.attr.dtype, input_dtype), af::SUCCESS);
  std::string output_dtype;
  ASSERT_EQ(codegen::Tensor::DtypeName(node->outputs[0].attr.dtype, output_dtype), af::SUCCESS);
  codegen::Tensor x(input_tensor, input_dtype);
  codegen::Tensor y(node->outputs[0], output_dtype);
  ASSERT_EQ(x.Init(), af::SUCCESS);
  ASSERT_EQ(y.Init(), af::SUCCESS);

  EXPECT_TRUE(parser_params->canonical_params.valid);
  CheckReduceSpecificParamsForCodegen({node, api_name, &tpipe, &x, &y, node->attr.sched.loop_axis});
}

void SetAscirNodeParamsForTest(const af::AscNodePtr &node, const ascir_param::AscirNodeParamsPtr &params) {
  ASSERT_NE(node, nullptr);
  ASSERT_NE(params, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  ASSERT_TRUE(op_desc->SetExtAttr("AscirNodeParams", params));
}

void SetupReduceTailAxisOneTiler(codegen::Tiler &tiler, std::vector<ascir::AxisId> &current_axis, af::Axis &z0,
                                 af::Axis &z1, af::Axis &z2) {
  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  z0 = af::Axis{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileInner, .size = s0.expr};
  z1 = af::Axis{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s1.expr};
  z2 = af::Axis{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeTileInner, .size = s2.expr};

  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  current_axis.push_back(z1.id);
}

void BuildReduceTailAxisOneGraph(af::AscGraph &graph, const af::Axis &z0, const af::Axis &z1, const af::Axis &z2) {
  af::ascir_op::Data x("x", graph);
  af::ascir_op::Data y("y", graph);
  af::ascir_op::Max reduce("reduce");
  graph.AddNode(reduce);

  auto nodex = graph.FindNode("x");
  nodex->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  af::AscTensor tensorx = nodex->outputs[0];
  auto nodey = graph.FindNode("y");
  af::AscTensor tensory = nodey->outputs[0];

  tensorx.attr.axis = {z0.id, z1.id, z2.id};
  tensorx.attr.vectorized_axis = {z0.id, z2.id};
  tensorx.attr.repeats = {z0.size, z1.size, z2.size};
  tensorx.attr.strides = {z1.size * z2.size, z2.size, One};
  tensorx.attr.mem.tensor_id = 1;
  tensorx.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensorx.attr.mem.position = af::Position::kPositionVecIn;
  tensorx.attr.opt.merge_scope = af::kIdNone;
  tensorx.attr.buf.id = 2;
  vector<af::Expression> vectorized_stride_x{One, Zero};
  tensorx.attr.vectorized_strides = vectorized_stride_x;

  tensory.attr.axis = {z0.id, z1.id, z2.id};
  tensory.attr.vectorized_axis = {z0.id, z2.id};
  tensory.attr.repeats = {One, z1.size, One};
  tensory.attr.strides = {Zero, One, Zero};
  tensory.attr.mem.tensor_id = 3;
  tensory.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensory.attr.mem.position = af::Position::kPositionVecOut;
  tensory.attr.opt.merge_scope = af::kIdNone;
  tensory.attr.buf.id = 4;
  vector<af::Expression> vectorized_stride_y{Zero, Zero};
  tensory.attr.vectorized_strides = vectorized_stride_y;
}

void SetupReduceTailAxisOneTensors(codegen::TPipe &tpipe, const af::AscGraph &graph, codegen::ReduceApiCall &call,
                                   codegen::ApiTensor &x_tensor) {
  auto nodex = graph.FindNode("x");
  af::AscTensor tensorx = nodex->outputs[0];
  auto nodey = graph.FindNode("y");
  auto reduce_node = graph.FindNode("reduce");
  af::AscTensor tensory = nodey->outputs[0];

  std::string dtype_name;
  Tensor::DtypeName(tensory.attr.dtype, dtype_name);
  // Setup inputs
  std::vector<std::reference_wrapper<const Tensor>> inputs;
  Tensor tensor1(tensorx, dtype_name);
  tensor1.is_constant = true;
  inputs.push_back(std::ref(tensor1));

  // Setup outputs
  std::vector<std::reference_wrapper<const Tensor>> outputs;
  Tensor output_tensor(tensory, dtype_name);
  outputs.push_back(std::ref(output_tensor));

  codegen::ApiTensor y_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  y_tensor.id = tensory.attr.mem.tensor_id;
  y_tensor.reuse_id = tensory.attr.mem.reuse_id;

  call.unit = af::ComputeUnit::kUnitVector;
  call.type = "Max";
  call.node = reduce_node;
  call.node_name = reduce_node->GetName();
  y_tensor.write = &call;
  call.inputs.push_back(&x_tensor);
  call.outputs.push_back(y_tensor);

  EXPECT_EQ(tpipe.AddTensor(tensor1), 0);
  EXPECT_EQ(tpipe.AddTensor(output_tensor), 0);
}
}  // namespace

TEST_F(ReduceApicallTest, ReduceApi_Test_001) {
  std::string api_name = "ReduceMax";

  std::vector<ascir::AxisId> current_axis;
  std::vector<std::reference_wrapper<const Tensor>> inputs;
  std::vector<std::reference_wrapper<const Tensor>> outputs;

  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileInner, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeTileInner, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  current_axis.push_back(z1.id);

  codegen::TPipe tpipe("tpipe", tiler);
  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  af::ascir_op::Data y("y", graph);
  af::ascir_op::Max reduce("reduce");
  graph.AddNode(reduce);

  auto nodex = graph.FindNode("x");
  af::AscTensor tensorx = nodex->outputs[0];
  auto nodey = graph.FindNode("y");
  auto reduce_node = graph.FindNode("reduce");
  af::AscTensor tensory = nodey->outputs[0];

  tensorx.attr.axis = {z0.id, z1.id, z2.id};
  tensorx.attr.vectorized_axis = {z0.id, z2.id};
  tensorx.attr.repeats = {z0.size, z1.size, z2.size};
  tensorx.attr.strides = {z1.size * z2.size, z2.size, One};
  tensorx.attr.mem.tensor_id = 1;
  tensorx.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensorx.attr.mem.position = af::Position::kPositionVecIn;
  tensorx.attr.opt.merge_scope = af::kIdNone;
  tensorx.attr.buf.id = 2;
  vector<af::Expression> vectorized_stride_x{One, One};
  tensorx.attr.vectorized_strides = vectorized_stride_x;

  tensory.attr.axis = {z0.id, z1.id, z2.id};
  tensory.attr.vectorized_axis = {z0.id, z2.id};
  tensory.attr.repeats = {z0.size, One, One};
  tensory.attr.strides = {One, Zero, Zero};
  tensory.attr.mem.tensor_id = 3;
  tensory.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensory.attr.mem.position = af::Position::kPositionVecOut;
  tensory.attr.opt.merge_scope = af::kIdNone;
  tensory.attr.buf.id = 4;
  vector<af::Expression> vectorized_stride_y{One, Zero};
  tensory.attr.vectorized_strides = vectorized_stride_y;

  std::string dtype_name;
  Tensor::DtypeName(tensory.attr.dtype, dtype_name);
  // Setup inputs
  Tensor tensor1(tensorx, dtype_name);
  tensor1.is_constant = true;
  Tensor tensor2(tensorx, dtype_name);
  tensor2.is_constant = true;
  Tensor tensor3(tensorx, dtype_name);
  tensor3.is_constant = true;
  inputs.push_back(std::ref(tensor1));
  inputs.push_back(std::ref(tensor2));
  inputs.push_back(std::ref(tensor3));

  // Setup outputs
  Tensor output_tensor(tensory, dtype_name);
  ;
  outputs.push_back(std::ref(output_tensor));

  codegen::ApiTensor x_tensor, y_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  y_tensor.id = tensory.attr.mem.tensor_id;
  y_tensor.reuse_id = tensory.attr.mem.reuse_id;

  codegen::ReduceApiCall call(api_name);
  call.unit = af::ComputeUnit::kUnitVector;
  call.type = "Max";
  call.node = reduce_node;
  call.node_name = reduce_node->GetName();
  y_tensor.write = &call;
  call.inputs.push_back(&x_tensor);
  call.outputs.push_back(y_tensor);

  EXPECT_EQ(tpipe.AddTensor(tensor1), 0);
  EXPECT_EQ(tpipe.AddTensor(output_tensor), 0);

  std::string result;
  call.tmp_buf_id[-1] = 0;
  call.tmp_buf_id[0] = 1;
  Status status = call.Generate(tpipe, current_axis, result);
  // Check the result
  EXPECT_EQ(status, af::SUCCESS);
}

TEST_F(ReduceApicallTest, ReduceApi_Test_002) {
  std::string api_name = "ReduceMax";

  std::vector<ascir::AxisId> current_axis;
  std::vector<std::reference_wrapper<const Tensor>> inputs;
  std::vector<std::reference_wrapper<const Tensor>> outputs;

  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileInner, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileInner, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeTileInner, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  current_axis.push_back(z1.id);

  codegen::TPipe tpipe("tpipe", tiler);
  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  af::ascir_op::Data y("y", graph);
  af::ascir_op::Max reduce("reduce");
  graph.AddNode(reduce);

  auto nodex = graph.FindNode("x");
  af::AscTensor tensorx = nodex->outputs[0];
  auto nodey = graph.FindNode("y");
  auto reduce_node = graph.FindNode("reduce");
  af::AscTensor tensory = nodey->outputs[0];

  tensorx.attr.axis = {z0.id, z1.id, z2.id};
  tensorx.attr.vectorized_axis = {z0.id, z2.id};
  tensorx.attr.repeats = {z0.size, z1.size, z2.size};
  tensorx.attr.strides = {z1.size * z2.size, z2.size, One};
  tensorx.attr.mem.tensor_id = 1;
  tensorx.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensorx.attr.mem.position = af::Position::kPositionVecIn;
  tensorx.attr.opt.merge_scope = af::kIdNone;
  tensorx.attr.buf.id = 2;
  vector<af::Expression> vectorized_stride_x{One, One};
  tensorx.attr.vectorized_strides = vectorized_stride_x;

  tensory.attr.axis = {z0.id, z1.id, z2.id};
  tensory.attr.vectorized_axis = {z0.id, z2.id};
  tensory.attr.repeats = {z0.size, One, One};
  tensory.attr.strides = {One, Zero, Zero};
  tensory.attr.mem.tensor_id = 3;
  tensory.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensory.attr.mem.position = af::Position::kPositionVecOut;
  tensory.attr.opt.merge_scope = af::kIdNone;
  tensory.attr.buf.id = 4;
  vector<af::Expression> vectorized_stride_y{One, Zero};
  tensory.attr.vectorized_strides = vectorized_stride_y;

  std::string dtype_name;
  Tensor::DtypeName(tensory.attr.dtype, dtype_name);
  // Setup inputs
  Tensor tensor1(tensorx, dtype_name);
  tensor1.is_constant = true;
  Tensor tensor2(tensorx, dtype_name);
  tensor2.is_constant = true;
  Tensor tensor3(tensorx, dtype_name);
  tensor3.is_constant = true;
  inputs.push_back(std::ref(tensor1));
  inputs.push_back(std::ref(tensor2));
  inputs.push_back(std::ref(tensor3));

  // Setup outputs
  Tensor output_tensor(tensory, dtype_name);
  outputs.push_back(std::ref(output_tensor));

  codegen::ApiTensor x_tensor, y_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  y_tensor.id = tensory.attr.mem.tensor_id;
  y_tensor.reuse_id = tensory.attr.mem.reuse_id;

  codegen::ReduceApiCall call(api_name);
  call.unit = af::ComputeUnit::kUnitVector;
  call.type = "Max";
  call.node = reduce_node;
  call.node_name = reduce_node->GetName();
  y_tensor.write = &call;
  call.inputs.push_back(&x_tensor);
  call.outputs.push_back(y_tensor);

  EXPECT_EQ(tpipe.AddTensor(tensor1), 0);
  EXPECT_EQ(tpipe.AddTensor(output_tensor), 0);

  std::string result;
  call.tmp_buf_id[-1] = 0;
  call.tmp_buf_id[0] = 1;
  Status status = call.Generate(tpipe, current_axis, result);
  // Check the result
  EXPECT_EQ(status, af::SUCCESS);
}

TEST_F(ReduceApicallTest, ReduceApi_Test_003) {
  std::string api_name = "ReduceMax";

  std::vector<ascir::AxisId> current_axis;
  std::vector<std::reference_wrapper<const Tensor>> inputs;
  std::vector<std::reference_wrapper<const Tensor>> outputs;

  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));
  af::SizeVar s3(af::Symbol("s3"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileInner, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeTileInner, .size = s2.expr};
  af::Axis z3{.id = 3, .name = "z3", .type = af::Axis::Type::kAxisTypeOriginal, .size = s3.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  tiler.AddSizeVar(af::SizeVar(s3));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  tiler.AddAxis(z3);
  current_axis.push_back(z1.id);

  codegen::TPipe tpipe("tpipe", tiler);
  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  af::ascir_op::Data y("y", graph);
  af::ascir_op::Max reduce("reduce");
  graph.AddNode(reduce);

  auto nodex = graph.FindNode("x");
  af::AscTensor tensorx = nodex->outputs[0];
  auto nodey = graph.FindNode("y");
  auto reduce_node = graph.FindNode("reduce");
  af::AscTensor tensory = nodey->outputs[0];

  tensorx.attr.axis = {z0.id, z1.id, z2.id, z3.id};
  tensorx.attr.vectorized_axis = {z0.id, z2.id, z3.id};
  tensorx.attr.repeats = {z0.size, z1.size, z2.size, z3.size};
  tensorx.attr.strides = {z1.size * z2.size * z3.size, z2.size * z3.size, z3.size, One};
  tensorx.attr.mem.tensor_id = 1;
  tensorx.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensorx.attr.mem.position = af::Position::kPositionVecIn;
  tensorx.attr.opt.merge_scope = af::kIdNone;
  tensorx.attr.buf.id = 2;
  vector<af::Expression> vectorized_stride_x{One, One, One};
  tensorx.attr.vectorized_strides = vectorized_stride_x;

  tensory.attr.axis = {z0.id, z1.id, z2.id, z3.id};
  tensory.attr.vectorized_axis = {z0.id, z2.id, z3.id};
  tensory.attr.repeats = {z0.size, One, One, One};
  tensory.attr.strides = {One, Zero, Zero, Zero};
  tensory.attr.mem.tensor_id = 3;
  tensory.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensory.attr.mem.position = af::Position::kPositionVecOut;
  tensory.attr.opt.merge_scope = af::kIdNone;
  tensory.attr.buf.id = 4;
  vector<af::Expression> vectorized_stride_y{One, Zero, Zero};
  tensory.attr.vectorized_strides = vectorized_stride_y;

  std::string dtype_name;
  Tensor::DtypeName(tensory.attr.dtype, dtype_name);
  // Setup inputs
  Tensor tensor1(tensorx, dtype_name);
  tensor1.is_constant = true;
  Tensor tensor2(tensorx, dtype_name);
  tensor2.is_constant = true;
  Tensor tensor3(tensorx, dtype_name);
  tensor3.is_constant = true;
  inputs.push_back(std::ref(tensor1));
  inputs.push_back(std::ref(tensor2));
  inputs.push_back(std::ref(tensor3));

  // Setup outputs
  Tensor output_tensor(tensory, dtype_name);
  ;
  outputs.push_back(std::ref(output_tensor));

  codegen::ApiTensor x_tensor, y_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  y_tensor.id = tensory.attr.mem.tensor_id;
  y_tensor.reuse_id = tensory.attr.mem.reuse_id;

  codegen::ReduceApiCall call(api_name);
  call.unit = af::ComputeUnit::kUnitVector;
  call.type = "Max";
  call.node = reduce_node;
  call.node_name = reduce_node->GetName();
  y_tensor.write = &call;
  call.inputs.push_back(&x_tensor);
  call.outputs.push_back(y_tensor);

  EXPECT_EQ(tpipe.AddTensor(tensor1), 0);
  EXPECT_EQ(tpipe.AddTensor(output_tensor), 0);

  std::string result;
  call.tmp_buf_id[-1] = 0;
  call.tmp_buf_id[0] = 1;
  Status status = call.Generate(tpipe, current_axis, result);
  // Check the result
  EXPECT_EQ(status, af::SUCCESS);
}

TEST_F(ReduceApicallTest, ReduceApi_Test_004) {
  std::string api_name = "ReduceMax";

  std::vector<ascir::AxisId> current_axis;
  std::vector<std::reference_wrapper<const Tensor>> inputs;
  std::vector<std::reference_wrapper<const Tensor>> outputs;

  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileInner, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeBlockInner, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeTileInner, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  current_axis.push_back(z1.id);

  codegen::TPipe tpipe("tpipe", tiler);
  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  af::ascir_op::Data y("y", graph);
  af::ascir_op::Max reduce("reduce");
  graph.AddNode(reduce);

  auto nodex = graph.FindNode("x");
  af::AscTensor tensorx = nodex->outputs[0];
  auto nodey = graph.FindNode("y");
  auto reduce_node = graph.FindNode("reduce");
  af::AscTensor tensory = nodey->outputs[0];

  tensorx.attr.axis = {z0.id, z1.id, z2.id};
  tensorx.attr.vectorized_axis = {z0.id, z2.id};
  tensorx.attr.repeats = {z0.size, z1.size, z2.size};
  tensorx.attr.strides = {z1.size * z2.size, z2.size, One};
  tensorx.attr.mem.tensor_id = 1;
  tensorx.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensorx.attr.mem.position = af::Position::kPositionVecIn;
  tensorx.attr.opt.merge_scope = af::kIdNone;
  tensorx.attr.buf.id = 2;
  vector<af::Expression> vectorized_stride_x{One, One};
  tensorx.attr.vectorized_strides = vectorized_stride_x;

  tensory.attr.axis = {z0.id, z1.id, z2.id};
  tensory.attr.vectorized_axis = {z0.id, z2.id};
  tensory.attr.repeats = {z0.size, One, One};
  tensory.attr.strides = {One, Zero, Zero};
  tensory.attr.mem.tensor_id = 3;
  tensory.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensory.attr.mem.position = af::Position::kPositionVecOut;
  tensory.attr.opt.merge_scope = af::kIdNone;
  tensory.attr.buf.id = 4;
  vector<af::Expression> vectorized_stride_y{One, Zero};
  tensory.attr.vectorized_strides = vectorized_stride_y;

  std::string dtype_name;
  Tensor::DtypeName(tensory.attr.dtype, dtype_name);
  // Setup inputs
  Tensor tensor1(tensorx, dtype_name);
  tensor1.is_constant = true;
  Tensor tensor2(tensorx, dtype_name);
  tensor2.is_constant = true;
  Tensor tensor3(tensorx, dtype_name);
  tensor3.is_constant = true;
  inputs.push_back(std::ref(tensor1));
  inputs.push_back(std::ref(tensor2));
  inputs.push_back(std::ref(tensor3));

  // Setup outputs
  Tensor output_tensor(tensory, dtype_name);
  ;
  outputs.push_back(std::ref(output_tensor));

  codegen::ApiTensor x_tensor, y_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  y_tensor.id = tensory.attr.mem.tensor_id;
  y_tensor.reuse_id = tensory.attr.mem.reuse_id;

  codegen::ReduceApiCall call(api_name);
  call.unit = af::ComputeUnit::kUnitVector;
  call.type = "Max";
  call.node = reduce_node;
  call.node_name = reduce_node->GetName();
  y_tensor.write = &call;
  call.inputs.push_back(&x_tensor);
  call.outputs.push_back(y_tensor);

  EXPECT_EQ(tpipe.AddTensor(tensor1), 0);
  EXPECT_EQ(tpipe.AddTensor(output_tensor), 0);

  std::string result;
  call.tmp_buf_id[-1] = 0;
  call.tmp_buf_id[0] = 1;
  Status status = call.Generate(tpipe, current_axis, result);
  // Check the result
  EXPECT_EQ(status, af::SUCCESS);
}

TEST_F(ReduceApicallTest, ReduceApi_Test_All_Reduce) {
  std::string api_name = "ReduceMax";

  std::vector<ascir::AxisId> current_axis;
  std::vector<std::reference_wrapper<const Tensor>> inputs;
  std::vector<std::reference_wrapper<const Tensor>> outputs;

  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileInner, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeTileInner, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  current_axis.push_back(z1.id);

  codegen::TPipe tpipe("tpipe", tiler);
  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  af::ascir_op::Data y("y", graph);
  af::ascir_op::Max reduce("reduce");
  graph.AddNode(reduce);

  auto nodex = graph.FindNode("x");
  nodex->attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};
  af::AscTensor tensorx = nodex->outputs[0];
  auto nodey = graph.FindNode("y");
  auto reduce_node = graph.FindNode("reduce");
  af::AscTensor tensory = nodey->outputs[0];

  tensorx.attr.axis = {z0.id, z1.id, z2.id};
  tensorx.attr.vectorized_axis = {z0.id, z2.id};
  tensorx.attr.repeats = {z0.size, z1.size, z2.size};
  tensorx.attr.strides = {z1.size * z2.size, z2.size, One};
  tensorx.attr.mem.tensor_id = 1;
  tensorx.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensorx.attr.mem.position = af::Position::kPositionVecIn;
  tensorx.attr.opt.merge_scope = af::kIdNone;
  tensorx.attr.buf.id = 2;
  vector<af::Expression> vectorized_stride_x{One, One};
  tensorx.attr.vectorized_strides = vectorized_stride_x;

  tensory.attr.axis = {z0.id, z1.id, z2.id};
  tensory.attr.vectorized_axis = {z0.id, z2.id};
  tensory.attr.repeats = {One, z1.size, One};
  tensory.attr.strides = {Zero, One, Zero};
  tensory.attr.mem.tensor_id = 3;
  tensory.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensory.attr.mem.position = af::Position::kPositionVecOut;
  tensory.attr.opt.merge_scope = af::kIdNone;
  tensory.attr.buf.id = 4;
  vector<af::Expression> vectorized_stride_y{Zero, Zero};
  tensory.attr.vectorized_strides = vectorized_stride_y;

  std::string dtype_name;
  Tensor::DtypeName(tensory.attr.dtype, dtype_name);
  // Setup inputs
  Tensor tensor1(tensorx, dtype_name);
  tensor1.is_constant = true;
  Tensor tensor2(tensorx, dtype_name);
  tensor2.is_constant = true;
  Tensor tensor3(tensorx, dtype_name);
  tensor3.is_constant = true;
  inputs.push_back(std::ref(tensor1));
  inputs.push_back(std::ref(tensor2));
  inputs.push_back(std::ref(tensor3));

  // Setup outputs
  Tensor output_tensor(tensory, dtype_name);
  ;
  outputs.push_back(std::ref(output_tensor));

  codegen::ApiTensor x_tensor, y_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  y_tensor.id = tensory.attr.mem.tensor_id;
  y_tensor.reuse_id = tensory.attr.mem.reuse_id;

  codegen::ReduceApiCall call(api_name);
  call.unit = af::ComputeUnit::kUnitVector;
  call.type = "Max";
  call.node = reduce_node;
  call.node_name = reduce_node->GetName();
  y_tensor.write = &call;
  call.inputs.push_back(&x_tensor);
  call.outputs.push_back(y_tensor);

  EXPECT_EQ(tpipe.AddTensor(tensor1), 0);
  EXPECT_EQ(tpipe.AddTensor(output_tensor), 0);

  std::string result;
  call.tmp_buf_id[-1] = 0;
  call.tmp_buf_id[0] = 1;
  Status status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(result,
            std::string{
                "{\nuint32_t last = 1 * t->s0 * KernelUtils::SizeAlign(t->s2, 32/sizeof(float));\nuint32_t last_actual "
                "= 1 * z0_actual_size * KernelUtils::SizeAlign(z2_actual_size, 32/sizeof(float));\nuint32_t first = "
                "1;\nuint32_t first_actual = 1;\nReduceInit<float, 1, false>(local_1, first_actual, last, last_actual, "
                "z2_actual_size);\nAscendC::PipeBarrier<PIPE_V>();\nuint32_t tmp_reduce_shape[] = {first_actual, "
                "last};\nReduceMax<float, AscendC::Pattern::Reduce::AR, false>(local_3[0], local_1[0], tmp_buf_0, "
                "tmp_reduce_shape, true);\n}\n"});
  // Check the result
  EXPECT_EQ(status, af::SUCCESS);
}

TEST_F(ReduceApicallTest, ReduceApi_Test_All_Reduce_TailAxisOne) {
  std::string api_name = "ReduceMax";
  std::vector<ascir::AxisId> current_axis;
  af::Axis z0;
  af::Axis z1;
  af::Axis z2;
  codegen::Tiler tiler;
  SetupReduceTailAxisOneTiler(tiler, current_axis, z0, z1, z2);

  codegen::TPipe tpipe("tpipe", tiler);
  af::AscGraph graph("test");
  BuildReduceTailAxisOneGraph(graph, z0, z1, z2);

  codegen::ApiTensor x_tensor;
  codegen::ReduceApiCall call(api_name);
  SetupReduceTailAxisOneTensors(tpipe, graph, call, x_tensor);

  std::string result;
  call.tmp_buf_id[-1] = 0;
  call.tmp_buf_id[0] = 1;
  Status status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(result,
            std::string{"{\nuint32_t first = 1 * t->s0;\nuint32_t first_actual = 1 * z0_actual_size;\nuint32_t last = "
                        "1 * KernelUtils::SizeAlign(1, 32/sizeof(float));\nuint32_t last_actual = 1 * "
                        "KernelUtils::SizeAlign(1, 32/sizeof(float));\nuint32_t tmp_reduce_shape[] = {first_actual, "
                        "last};\nReduceMax<float, AscendC::Pattern::Reduce::RA, false>(local_3[0], local_1[0], "
                        "tmp_buf_0, tmp_reduce_shape, true);\n}\n"});
  // Check the result
  EXPECT_EQ(status, af::SUCCESS);
}

TEST_F(ReduceApicallTest, ReduceApicallTest_Int32_Inner) {
  std::string api_name = "ReduceSum";

  std::vector<ascir::AxisId> current_axis;
  std::vector<std::reference_wrapper<const Tensor>> inputs;
  std::vector<std::reference_wrapper<const Tensor>> outputs;

  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileInner, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeBlockInner, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeTileInner, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  current_axis.push_back(z1.id);

  codegen::TPipe tpipe("tpipe", tiler);
  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  af::ascir_op::Data y("y", graph);
  af::ascir_op::Sum reduce("reduce");
  graph.AddNode(reduce);

  auto nodex = graph.FindNode("x");
  af::AscTensor tensorx = nodex->outputs[0];
  auto nodey = graph.FindNode("y");
  auto reduce_node = graph.FindNode("reduce");
  af::AscTensor tensory = nodey->outputs[0];

  tensorx.attr.axis = {z0.id, z1.id, z2.id};
  tensorx.attr.vectorized_axis = {z0.id, z2.id};
  tensorx.attr.repeats = {z0.size, z1.size, z2.size};
  tensorx.attr.strides = {z1.size * z2.size, z2.size, One};
  tensorx.attr.mem.tensor_id = 1;
  tensorx.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensorx.attr.mem.position = af::Position::kPositionVecIn;
  tensorx.attr.opt.merge_scope = af::kIdNone;
  tensorx.attr.buf.id = 2;
  tensorx.attr.dtype = ge::DT_INT32;
  vector<af::Expression> vectorized_stride_x{One, One};
  tensorx.attr.vectorized_strides = vectorized_stride_x;

  tensory.attr.axis = {z0.id, z1.id, z2.id};
  tensory.attr.vectorized_axis = {z0.id, z2.id};
  tensory.attr.repeats = {z0.size, One, One};
  tensory.attr.strides = {One, Zero, Zero};
  tensory.attr.mem.tensor_id = 3;
  tensory.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensory.attr.mem.position = af::Position::kPositionVecOut;
  tensory.attr.opt.merge_scope = af::kIdNone;
  tensory.attr.buf.id = 4;
  tensory.attr.dtype = ge::DT_INT32;
  vector<af::Expression> vectorized_stride_y{One, Zero};
  tensory.attr.vectorized_strides = vectorized_stride_y;

  std::string dtype_name;
  Tensor::DtypeName(tensory.attr.dtype, dtype_name);
  // Setup inputs
  Tensor tensor1(tensorx, dtype_name);
  tensor1.is_constant = true;
  Tensor tensor2(tensorx, dtype_name);
  tensor2.is_constant = true;
  Tensor tensor3(tensorx, dtype_name);
  tensor3.is_constant = true;
  inputs.push_back(std::ref(tensor1));
  inputs.push_back(std::ref(tensor2));
  inputs.push_back(std::ref(tensor3));

  // Setup outputs
  Tensor output_tensor(tensory, dtype_name);
  ;
  outputs.push_back(std::ref(output_tensor));

  codegen::ApiTensor x_tensor, y_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  y_tensor.id = tensory.attr.mem.tensor_id;
  y_tensor.reuse_id = tensory.attr.mem.reuse_id;

  codegen::ReduceApiCall call(api_name);
  call.unit = af::ComputeUnit::kUnitVector;
  call.type = "Sum";
  call.node = reduce_node;
  call.node_name = reduce_node->GetName();
  y_tensor.write = &call;
  call.inputs.push_back(&x_tensor);
  call.outputs.push_back(y_tensor);

  EXPECT_EQ(tpipe.AddTensor(tensor1), 0);
  EXPECT_EQ(tpipe.AddTensor(output_tensor), 0);

  std::string result;
  call.tmp_buf_id[-1] = 0;
  call.tmp_buf_id[0] = 1;
  Status status = call.Generate(tpipe, current_axis, result);
  // Check the result
  EXPECT_EQ(status, af::SUCCESS);
}

TEST_F(ReduceApicallTest, ReduceApicallTest_Int32_Outer) {
  std::string api_name = "ReduceSum";

  std::vector<ascir::AxisId> current_axis;
  std::vector<std::reference_wrapper<const Tensor>> inputs;
  std::vector<std::reference_wrapper<const Tensor>> outputs;

  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileInner, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeTileInner, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  current_axis.push_back(z1.id);

  codegen::TPipe tpipe("tpipe", tiler);
  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  af::ascir_op::Data y("y", graph);
  af::ascir_op::Sum reduce("reduce");
  graph.AddNode(reduce);

  auto nodex = graph.FindNode("x");
  af::AscTensor tensorx = nodex->outputs[0];
  auto nodey = graph.FindNode("y");
  auto reduce_node = graph.FindNode("reduce");
  af::AscTensor tensory = nodey->outputs[0];

  tensorx.attr.axis = {z0.id, z1.id, z2.id};
  tensorx.attr.vectorized_axis = {z0.id, z2.id};
  tensorx.attr.repeats = {z0.size, z1.size, z2.size};
  tensorx.attr.strides = {z1.size * z2.size, z2.size, One};
  tensorx.attr.mem.tensor_id = 1;
  tensorx.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensorx.attr.mem.position = af::Position::kPositionVecIn;
  tensorx.attr.opt.merge_scope = af::kIdNone;
  tensorx.attr.buf.id = 2;
  tensorx.attr.dtype = ge::DT_INT32;
  vector<af::Expression> vectorized_stride_x{One, One};
  tensorx.attr.vectorized_strides = vectorized_stride_x;

  tensory.attr.axis = {z0.id, z1.id, z2.id};
  tensory.attr.vectorized_axis = {z0.id, z2.id};
  tensory.attr.repeats = {z0.size, One, One};
  tensory.attr.strides = {One, Zero, Zero};
  tensory.attr.mem.tensor_id = 3;
  tensory.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensory.attr.mem.position = af::Position::kPositionVecOut;
  tensory.attr.opt.merge_scope = af::kIdNone;
  tensory.attr.buf.id = 4;
  tensory.attr.dtype = ge::DT_INT32;
  vector<af::Expression> vectorized_stride_y{One, Zero};
  tensory.attr.vectorized_strides = vectorized_stride_y;

  std::string dtype_name;
  Tensor::DtypeName(tensory.attr.dtype, dtype_name);
  // Setup inputs
  Tensor tensor1(tensorx, dtype_name);
  tensor1.is_constant = true;
  Tensor tensor2(tensorx, dtype_name);
  tensor2.is_constant = true;
  Tensor tensor3(tensorx, dtype_name);
  tensor3.is_constant = true;
  inputs.push_back(std::ref(tensor1));
  inputs.push_back(std::ref(tensor2));
  inputs.push_back(std::ref(tensor3));

  // Setup outputs
  Tensor output_tensor(tensory, dtype_name);
  ;
  outputs.push_back(std::ref(output_tensor));

  codegen::ApiTensor x_tensor, y_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  y_tensor.id = tensory.attr.mem.tensor_id;
  y_tensor.reuse_id = tensory.attr.mem.reuse_id;

  codegen::ReduceApiCall call(api_name);
  call.unit = af::ComputeUnit::kUnitVector;
  call.type = "Sum";
  call.node = reduce_node;
  call.node_name = reduce_node->GetName();
  y_tensor.write = &call;
  call.inputs.push_back(&x_tensor);
  call.outputs.push_back(y_tensor);

  EXPECT_EQ(tpipe.AddTensor(tensor1), 0);
  EXPECT_EQ(tpipe.AddTensor(output_tensor), 0);

  std::string result;
  call.tmp_buf_id[-1] = 0;
  call.tmp_buf_id[0] = 1;
  Status status = call.Generate(tpipe, current_axis, result);
  // Check the result
  EXPECT_EQ(status, af::SUCCESS);
}

TEST_F(ReduceApicallTest, ReduceApicallTest_ReduceMean_NoNeed_MultiReduce_Int32) {
  std::string api_name = "ReduceMean";

  std::vector<ascir::AxisId> current_axis;
  std::vector<std::reference_wrapper<const Tensor>> inputs;
  std::vector<std::reference_wrapper<const Tensor>> outputs;

  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileInner, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  current_axis.push_back(z0.id);

  codegen::TPipe tpipe("tpipe", tiler);
  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  af::ascir_op::Data y("y", graph);
  af::ascir_op::Sum reduce("reduce");
  graph.AddNode(reduce);

  auto nodex = graph.FindNode("x");
  af::AscTensor tensorx = nodex->outputs[0];
  auto nodey = graph.FindNode("y");
  auto reduce_node = graph.FindNode("reduce");
  af::AscTensor tensory = nodey->outputs[0];

  tensorx.attr.axis = {z0.id, z1.id, z2.id};
  tensorx.attr.vectorized_axis = {z1.id, z2.id};
  tensorx.attr.repeats = {z0.size, z1.size, z2.size};
  tensorx.attr.strides = {z1.size * z2.size, z2.size, One};
  tensorx.attr.mem.tensor_id = 1;
  tensorx.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensorx.attr.mem.position = af::Position::kPositionVecIn;
  tensorx.attr.opt.merge_scope = af::kIdNone;
  tensorx.attr.buf.id = 2;
  tensorx.attr.dtype = ge::DT_INT32;
  vector<af::Expression> vectorized_stride_x{z2.size, One};
  tensorx.attr.vectorized_strides = vectorized_stride_x;

  tensory.attr.axis = {z0.id, z1.id, z2.id};
  tensory.attr.vectorized_axis = {z1.id, z2.id};
  tensory.attr.repeats = {z0.size, z1.size, One};
  tensory.attr.strides = {z1.size, One, Zero};
  tensory.attr.mem.tensor_id = 2;
  tensory.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensory.attr.mem.position = af::Position::kPositionVecOut;
  tensory.attr.opt.merge_scope = af::kIdNone;
  tensory.attr.buf.id = 4;
  tensory.attr.dtype = ge::DT_INT32;
  vector<af::Expression> vectorized_stride_y{One, Zero};
  tensory.attr.vectorized_strides = vectorized_stride_y;

  std::string dtype_name;
  Tensor::DtypeName(tensory.attr.dtype, dtype_name);
  // Setup inputs
  Tensor tensor1(tensorx, dtype_name);
  tensor1.is_constant = true;
  Tensor tensor2(tensorx, dtype_name);
  tensor2.is_constant = true;
  Tensor tensor3(tensorx, dtype_name);
  tensor3.is_constant = true;
  inputs.push_back(std::ref(tensor1));
  inputs.push_back(std::ref(tensor2));
  inputs.push_back(std::ref(tensor3));

  // Setup outputs
  Tensor output_tensor(tensory, dtype_name);
  ;
  outputs.push_back(std::ref(output_tensor));

  codegen::ApiTensor x_tensor, y_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  y_tensor.id = tensory.attr.mem.tensor_id;
  y_tensor.reuse_id = tensory.attr.mem.reuse_id;

  codegen::ReduceApiCall call(api_name);
  call.unit = af::ComputeUnit::kUnitVector;
  call.type = "Mean";
  call.node = reduce_node;
  call.node_name = reduce_node->GetName();
  y_tensor.write = &call;
  call.inputs.push_back(&x_tensor);
  call.outputs.push_back(y_tensor);

  EXPECT_EQ(tpipe.AddTensor(tensor1), 0);
  EXPECT_EQ(tpipe.AddTensor(output_tensor), 0);

  std::string result;
  call.tmp_buf_id[-1] = 0;
  call.tmp_buf_id[0] = 1;
  Status status = call.Generate(tpipe, current_axis, result);
  // Check the result
  EXPECT_EQ(status, af::SUCCESS);
}

TEST_F(ReduceApicallTest, ReduceApicallTest_ReduceMean_NoNeed_MultiReduce_Float) {
  std::string api_name = "ReduceMean";

  std::vector<ascir::AxisId> current_axis;
  std::vector<std::reference_wrapper<const Tensor>> inputs;
  std::vector<std::reference_wrapper<const Tensor>> outputs;

  af::SizeVar s0(af::Symbol("s0"));
  af::SizeVar s1(af::Symbol("s1"));
  af::SizeVar s2(af::Symbol("s2"));

  af::Axis z0{.id = 0, .name = "z0", .type = af::Axis::Type::kAxisTypeTileInner, .size = s0.expr};
  af::Axis z1{.id = 1, .name = "z1", .type = af::Axis::Type::kAxisTypeTileOuter, .size = s1.expr};
  af::Axis z2{.id = 2, .name = "z2", .type = af::Axis::Type::kAxisTypeOriginal, .size = s2.expr};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s2));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z2);
  current_axis.push_back(z0.id);

  codegen::TPipe tpipe("tpipe", tiler);
  af::AscGraph graph("test");
  af::ascir_op::Data x("x", graph);
  af::ascir_op::Data y("y", graph);
  af::ascir_op::Sum reduce("reduce");
  graph.AddNode(reduce);

  auto nodex = graph.FindNode("x");
  af::AscTensor tensorx = nodex->outputs[0];
  auto nodey = graph.FindNode("y");
  auto reduce_node = graph.FindNode("reduce");
  af::AscTensor tensory = nodey->outputs[0];

  tensorx.attr.axis = {z0.id, z1.id, z2.id};
  tensorx.attr.vectorized_axis = {z1.id, z2.id};
  tensorx.attr.repeats = {z0.size, z1.size, z2.size};
  tensorx.attr.strides = {z1.size * z2.size, z2.size, One};
  tensorx.attr.mem.tensor_id = 1;
  tensorx.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensorx.attr.mem.position = af::Position::kPositionVecIn;
  tensorx.attr.opt.merge_scope = af::kIdNone;
  tensorx.attr.buf.id = 2;
  tensorx.attr.dtype = ge::DT_FLOAT;
  vector<af::Expression> vectorized_stride_x{z2.size, One};
  tensorx.attr.vectorized_strides = vectorized_stride_x;

  tensory.attr.axis = {z0.id, z1.id, z2.id};
  tensory.attr.vectorized_axis = {z1.id, z2.id};
  tensory.attr.repeats = {z0.size, z1.size, One};
  tensory.attr.strides = {z1.size, One, Zero};
  tensory.attr.mem.tensor_id = 2;
  tensory.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensory.attr.mem.position = af::Position::kPositionVecOut;
  tensory.attr.opt.merge_scope = af::kIdNone;
  tensory.attr.buf.id = 4;
  tensory.attr.dtype = ge::DT_FLOAT;
  vector<af::Expression> vectorized_stride_y{One, Zero};
  tensory.attr.vectorized_strides = vectorized_stride_y;

  std::string dtype_name;
  Tensor::DtypeName(tensory.attr.dtype, dtype_name);
  // Setup inputs
  Tensor tensor1(tensorx, dtype_name);
  tensor1.is_constant = true;
  Tensor tensor2(tensorx, dtype_name);
  tensor2.is_constant = true;
  Tensor tensor3(tensorx, dtype_name);
  tensor3.is_constant = true;
  inputs.push_back(std::ref(tensor1));
  inputs.push_back(std::ref(tensor2));
  inputs.push_back(std::ref(tensor3));

  // Setup outputs
  Tensor output_tensor(tensory, dtype_name);
  ;
  outputs.push_back(std::ref(output_tensor));

  codegen::ApiTensor x_tensor, y_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  y_tensor.id = tensory.attr.mem.tensor_id;
  y_tensor.reuse_id = tensory.attr.mem.reuse_id;

  codegen::ReduceApiCall call(api_name);
  call.unit = af::ComputeUnit::kUnitVector;
  call.type = "Mean";
  call.node = reduce_node;
  call.node_name = reduce_node->GetName();
  y_tensor.write = &call;
  call.inputs.push_back(&x_tensor);
  call.outputs.push_back(y_tensor);

  EXPECT_EQ(tpipe.AddTensor(tensor1), 0);
  EXPECT_EQ(tpipe.AddTensor(output_tensor), 0);

  std::string result;
  call.tmp_buf_id[-1] = 0;
  call.tmp_buf_id[0] = 1;
  Status status = call.Generate(tpipe, current_axis, result);
  // Check the result
  EXPECT_EQ(status, af::SUCCESS);
}

TEST_F(ReduceApicallTest, ReduceShadowCheck_DoesNotFailWhenNodeParamsMissing) {
  CheckReduceSpecificParamsForCodegen(MakeNonFatalShadowInput(nullptr));
  SUCCEED();
}

TEST_F(ReduceApicallTest, ReduceShadowCheck_DoesNotFailWhenStatusNotBuilt) {
  af::AscGraph graph("shadow_status_graph");
  af::ascir_op::Add add_op("shadow_status_node");
  graph.AddNode(add_op);
  auto node = graph.FindNode("shadow_status_node");
  ASSERT_NE(node, nullptr);

  auto params = std::make_shared<ascir_param::AscirNodeParams>();
  params->api_name = "ReduceMax";
  params->status = ascir_param::ParamBuildStatus::kSkipped;
  SetAscirNodeParamsForTest(node, params);

  CheckReduceSpecificParamsForCodegen(MakeNonFatalShadowInput(node));
  SUCCEED();
}

TEST_F(ReduceApicallTest, ReduceShadowCheck_DoesNotFailWhenSpecificParamsTypeMismatch) {
  af::AscGraph graph("shadow_type_graph");
  af::ascir_op::Add add_op("shadow_type_node");
  graph.AddNode(add_op);
  auto node = graph.FindNode("shadow_type_node");
  ASSERT_NE(node, nullptr);

  auto params = std::make_shared<ascir_param::AscirNodeParams>();
  params->api_name = "ReduceMax";
  params->status = ascir_param::ParamBuildStatus::kBuilt;
  SetAscirNodeParamsForTest(node, params);

  CheckReduceSpecificParamsForCodegen(MakeNonFatalShadowInput(node));
  SUCCEED();
}

TEST_F(ReduceApicallTest, ReduceShadowCheck_DoesNotFailWhenShadowBuildFails) {
  af::AscGraph graph("shadow_build_graph");
  af::ascir_op::Add add_op("shadow_build_node");
  graph.AddNode(add_op);
  auto node = graph.FindNode("shadow_build_node");
  ASSERT_NE(node, nullptr);

  auto params = std::make_shared<ascir_param::AscirNodeParams>();
  params->api_name = "ReduceMax";
  params->status = ascir_param::ParamBuildStatus::kBuilt;
  params->specific_params = BuildReduceNodeParamForShadowTest();
  SetAscirNodeParamsForTest(node, params);

  CheckReduceSpecificParamsForCodegen(MakeNonFatalShadowInput(node));
  SUCCEED();
}

TEST_F(ReduceApicallTest, ReduceShadowCheck_RminTileSplitMatchesParserParams) {
  af::AscGraph graph("rmin_shadow_graph");
  BuildRminGraph(graph);
  auto rmin = graph.FindNode("rmin");
  ASSERT_NE(rmin, nullptr);
  ExpectReduceShadowParamsMatch(graph, rmin, "Min");
}

TEST_F(ReduceApicallTest, ReduceShadowCheck_OriginalParentAxisMatchesParserParams) {
  af::AscGraph graph("rmin_original_parent_shadow_graph");
  BuildOriginalParentAxisGraph(graph);
  auto rmin = graph.FindNode("rmin");
  ASSERT_NE(rmin, nullptr);
  ExpectReduceShadowParamsMatch(graph, rmin, "Min");
}
