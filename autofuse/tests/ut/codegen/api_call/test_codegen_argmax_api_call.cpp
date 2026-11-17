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
#include "ascir_ops.h"
#include "codegen_kernel.h"
#include "utils/api_call_factory.h"
#include "reduce/reduce_api_call.h"

using namespace af::ops;
using namespace codegen;
using namespace af::ascir_op;
using namespace testing;

namespace {
void CheckReduceTailBoolParam(const std::string &result) {
  const bool has_argmax_call =
      (result.find("ArgMaxExtend") != std::string::npos) || (result.find("ArgMaxWithValueExtend") != std::string::npos);
  const std::string expected_tail = has_argmax_call ? "tmp_reduce_shape, false);" : "tmp_reduce_shape, true);";
  const std::string forbidden_tail = has_argmax_call ? "tmp_reduce_shape, true);" : "tmp_reduce_shape, false);";
  EXPECT_NE(result.find(expected_tail), std::string::npos);
  EXPECT_EQ(result.find(forbidden_tail), std::string::npos);
}
}  // namespace

class ArgmaxApicallTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(ArgmaxApicallTest, ReduceApi_Test_Argmax) {
  std::string api_name = "Argmax";

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
  af::ascir_op::Max reduce("reduce");  // 创建Reduce类型的节点
  graph.AddNode(reduce);               // 将Reduce节点添加到图中

  auto nodex = graph.FindNode("x");
  af::AscTensor tensorx = nodex->outputs[0];
  auto nodey = graph.FindNode("y");
  auto reduce_node = graph.FindNode("reduce");  // 获取Reduce节点
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
  tensory.attr.dtype = ge::DT_INT32;

  std::string dtype_name;
  Tensor::DtypeName(tensorx.attr.dtype, dtype_name);
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
  std::string output_dtype_name;
  Tensor::DtypeName(tensory.attr.dtype, output_dtype_name);
  Tensor output_tensor(tensory, output_dtype_name);
  outputs.push_back(std::ref(output_tensor));

  codegen::ApiTensor x_tensor, y_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  y_tensor.id = tensory.attr.mem.tensor_id;
  y_tensor.reuse_id = tensory.attr.mem.reuse_id;

  codegen::ReduceApiCall call(api_name);
  call.unit = af::ComputeUnit::kUnitVector;
  call.type = "Argmax";
  call.node = reduce_node;  // 设置Reduce节点，BuildApiParam需要有效的node
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
  CheckReduceTailBoolParam(result);
}

TEST_F(ArgmaxApicallTest, ReduceApi_Test_ArgMaxMultiRPhase1) {
  std::string api_name = "ArgMaxMultiRPhase1";

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
  af::ascir_op::Max reduce("reduce");  // 创建Reduce类型的节点
  graph.AddNode(reduce);               // 将Reduce节点添加到图中

  auto nodex = graph.FindNode("x");
  af::AscTensor tensorx = nodex->outputs[0];
  auto nodey = graph.FindNode("y");
  auto reduce_node = graph.FindNode("reduce");  // 获取Reduce节点
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
  tensory.attr.dtype = ge::DT_INT32;

  // Setup second output (index) for ArgMaxMultiRPhase1
  af::ascir_op::Data z("z", graph);
  auto nodez = graph.FindNode("z");
  af::AscTensor tensorz = nodez->outputs[0];
  tensorz.attr.axis = {z0.id, z1.id, z2.id};
  tensorz.attr.vectorized_axis = {z0.id, z2.id};
  tensorz.attr.repeats = {z0.size, One, One};
  tensorz.attr.strides = {One, Zero, Zero};
  tensorz.attr.mem.tensor_id = 5;
  tensorz.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensorz.attr.mem.position = af::Position::kPositionVecOut;
  tensorz.attr.opt.merge_scope = af::kIdNone;
  tensorz.attr.buf.id = 6;
  tensorz.attr.vectorized_strides = vectorized_stride_y;
  tensorz.attr.dtype = ge::DT_INT64;

  std::string dtype_name;
  Tensor::DtypeName(tensorx.attr.dtype, dtype_name);
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
  std::string output_dtype_name;
  Tensor::DtypeName(tensory.attr.dtype, output_dtype_name);
  Tensor output_tensor(tensory, output_dtype_name);
  outputs.push_back(std::ref(output_tensor));

  // Setup second output (index tensor)
  std::string index_dtype_name;
  Tensor::DtypeName(tensorz.attr.dtype, index_dtype_name);
  Tensor index_tensor(tensorz, index_dtype_name);
  outputs.push_back(std::ref(index_tensor));

  codegen::ApiTensor x_tensor, y_tensor, z_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  y_tensor.id = tensory.attr.mem.tensor_id;
  y_tensor.reuse_id = tensory.attr.mem.reuse_id;
  z_tensor.id = tensorz.attr.mem.tensor_id;
  z_tensor.reuse_id = tensorz.attr.mem.reuse_id;

  codegen::ReduceApiCall call(api_name);
  call.unit = af::ComputeUnit::kUnitVector;
  call.type = "ArgMaxMultiRPhase1";
  call.node = reduce_node;  // 设置Reduce节点，BuildApiParam需要有效的node
  call.node_name = reduce_node->GetName();
  y_tensor.write = &call;
  z_tensor.write = &call;
  call.inputs.push_back(&x_tensor);
  call.outputs.push_back(y_tensor);
  call.outputs.push_back(z_tensor);

  EXPECT_EQ(tpipe.AddTensor(tensor1), 0);
  EXPECT_EQ(tpipe.AddTensor(output_tensor), 0);
  EXPECT_EQ(tpipe.AddTensor(index_tensor), 0);

  std::string result;
  call.tmp_buf_id[-1] = 0;
  call.tmp_buf_id[0] = 1;
  Status status = call.Generate(tpipe, current_axis, result);
  // Check the result
  EXPECT_EQ(status, af::SUCCESS);
  CheckReduceTailBoolParam(result);
}

TEST_F(ArgmaxApicallTest, ReduceApi_Test_ArgMaxMultiRPhase2) {
  std::string api_name = "ArgMaxMultiRPhase2";

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
  af::ascir_op::Max reduce("reduce");  // 创建Reduce类型的节点
  graph.AddNode(reduce);               // 将Reduce节点添加到图中

  auto nodex = graph.FindNode("x");
  af::AscTensor tensorx = nodex->outputs[0];
  auto nodey = graph.FindNode("y");
  auto reduce_node = graph.FindNode("reduce");  // 获取Reduce节点
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
  tensory.attr.dtype = ge::DT_INT32;

  // Setup second input (index) for ArgMaxMultiRPhase2
  af::ascir_op::Data z("z", graph);
  auto nodez = graph.FindNode("z");
  af::AscTensor tensorz = nodez->outputs[0];
  tensorz.attr.axis = {z0.id, z1.id, z2.id};
  tensorz.attr.vectorized_axis = {z0.id, z2.id};
  tensorz.attr.repeats = {z0.size, One, One};
  tensorz.attr.strides = {One, Zero, Zero};
  tensorz.attr.mem.tensor_id = 5;
  tensorz.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensorz.attr.mem.position = af::Position::kPositionVecIn;
  tensorz.attr.opt.merge_scope = af::kIdNone;
  tensorz.attr.buf.id = 6;
  tensorz.attr.vectorized_strides = vectorized_stride_y;
  tensorz.attr.dtype = ge::DT_INT64;

  std::string dtype_name;
  Tensor::DtypeName(tensorx.attr.dtype, dtype_name);
  // Setup inputs (value tensor)
  Tensor tensor1(tensorx, dtype_name);
  tensor1.is_constant = true;
  Tensor tensor2(tensorx, dtype_name);
  tensor2.is_constant = true;
  Tensor tensor3(tensorx, dtype_name);
  tensor3.is_constant = true;
  inputs.push_back(std::ref(tensor1));
  inputs.push_back(std::ref(tensor2));
  inputs.push_back(std::ref(tensor3));

  // Setup second input (index tensor)
  std::string index_dtype_name;
  Tensor::DtypeName(tensorz.attr.dtype, index_dtype_name);
  Tensor index_tensor(tensorz, index_dtype_name);
  index_tensor.is_constant = true;
  inputs.push_back(std::ref(index_tensor));

  // Setup outputs
  std::string output_dtype_name;
  Tensor::DtypeName(tensory.attr.dtype, output_dtype_name);
  Tensor output_tensor(tensory, output_dtype_name);
  outputs.push_back(std::ref(output_tensor));

  codegen::ApiTensor x_tensor, z_tensor, y_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  z_tensor.id = tensorz.attr.mem.tensor_id;
  y_tensor.id = tensory.attr.mem.tensor_id;
  y_tensor.reuse_id = tensory.attr.mem.reuse_id;

  codegen::ReduceApiCall call(api_name);
  call.unit = af::ComputeUnit::kUnitVector;
  call.type = "ArgMaxMultiRPhase2";
  call.node = reduce_node;  // 设置Reduce节点，BuildApiParam需要有效的node
  call.node_name = reduce_node->GetName();
  y_tensor.write = &call;
  call.inputs.push_back(&x_tensor);
  call.inputs.push_back(&z_tensor);
  call.outputs.push_back(y_tensor);

  EXPECT_EQ(tpipe.AddTensor(tensor1), 0);
  EXPECT_EQ(tpipe.AddTensor(index_tensor), 0);
  EXPECT_EQ(tpipe.AddTensor(output_tensor), 0);

  std::string result;
  call.tmp_buf_id[-1] = 0;
  call.tmp_buf_id[0] = 1;
  Status status = call.Generate(tpipe, current_axis, result);
  // Check the result
  EXPECT_EQ(status, af::SUCCESS);
  CheckReduceTailBoolParam(result);
}

TEST_F(ArgmaxApicallTest, ReduceApicallTest_Argmax_NoNeed_MultiReduce_Int32) {
  std::string api_name = "Argmax";

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
  af::ascir_op::Max reduce("reduce");  // 创建Reduce类型的节点
  graph.AddNode(reduce);               // 将Reduce节点添加到图中

  auto nodex = graph.FindNode("x");
  af::AscTensor tensorx = nodex->outputs[0];
  auto nodey = graph.FindNode("y");
  auto reduce_node = graph.FindNode("reduce");  // 获取Reduce节点
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
  Tensor::DtypeName(tensorx.attr.dtype, dtype_name);
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
  call.type = "Argmax";
  call.node = reduce_node;  // 设置Reduce节点，BuildApiParam需要有效的node
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
  CheckReduceTailBoolParam(result);
}

TEST_F(ArgmaxApicallTest, ReduceApicallTest_ArgMaxMultiRPhase1_NoNeed_MultiReduce_Int32) {
  std::string api_name = "ArgMaxMultiRPhase1";

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
  af::ascir_op::Max reduce("reduce");  // 创建Reduce类型的节点
  graph.AddNode(reduce);               // 将Reduce节点添加到图中

  auto nodex = graph.FindNode("x");
  af::AscTensor tensorx = nodex->outputs[0];
  auto nodey = graph.FindNode("y");
  auto reduce_node = graph.FindNode("reduce");  // 获取Reduce节点
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

  // Setup second output (index) for ArgMaxMultiRPhase1
  af::ascir_op::Data z("z", graph);
  auto nodez = graph.FindNode("z");
  af::AscTensor tensorz = nodez->outputs[0];
  tensorz.attr.axis = {z0.id, z1.id, z2.id};
  tensorz.attr.vectorized_axis = {z1.id, z2.id};
  tensorz.attr.repeats = {z0.size, One, One};
  tensorz.attr.strides = {One, Zero, Zero};
  tensorz.attr.mem.tensor_id = 3;
  tensorz.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensorz.attr.mem.position = af::Position::kPositionVecOut;
  tensorz.attr.opt.merge_scope = af::kIdNone;
  tensorz.attr.buf.id = 6;
  tensorz.attr.vectorized_strides = vectorized_stride_y;
  tensorz.attr.dtype = ge::DT_INT64;

  std::string dtype_name;
  Tensor::DtypeName(tensorx.attr.dtype, dtype_name);
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

  // Setup second output (index tensor)
  std::string index_dtype_name;
  Tensor::DtypeName(tensorz.attr.dtype, index_dtype_name);
  Tensor index_tensor(tensorz, index_dtype_name);
  outputs.push_back(std::ref(index_tensor));

  codegen::ApiTensor x_tensor, y_tensor, z_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  y_tensor.id = tensory.attr.mem.tensor_id;
  y_tensor.reuse_id = tensory.attr.mem.reuse_id;
  z_tensor.id = tensorz.attr.mem.tensor_id;
  z_tensor.reuse_id = tensorz.attr.mem.reuse_id;

  codegen::ReduceApiCall call(api_name);
  call.unit = af::ComputeUnit::kUnitVector;
  call.type = "ArgMaxMultiRPhase1";
  call.node = reduce_node;  // 设置Reduce节点，BuildApiParam需要有效的node
  call.node_name = reduce_node->GetName();
  y_tensor.write = &call;
  z_tensor.write = &call;
  call.inputs.push_back(&x_tensor);
  call.outputs.push_back(y_tensor);
  call.outputs.push_back(z_tensor);

  EXPECT_EQ(tpipe.AddTensor(tensor1), 0);
  EXPECT_EQ(tpipe.AddTensor(output_tensor), 0);
  EXPECT_EQ(tpipe.AddTensor(index_tensor), 0);

  std::string result;
  call.tmp_buf_id[-1] = 0;
  call.tmp_buf_id[0] = 1;
  Status status = call.Generate(tpipe, current_axis, result);
  // Check the result
  EXPECT_EQ(status, af::SUCCESS);
  CheckReduceTailBoolParam(result);
}

TEST_F(ArgmaxApicallTest, ReduceApicallTest_ArgMaxMultiRPhase2_NoNeed_MultiReduce_Int32) {
  std::string api_name = "ArgMaxMultiRPhase2";

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
  af::ascir_op::Max reduce("reduce");  // 创建Reduce类型的节点
  graph.AddNode(reduce);               // 将Reduce节点添加到图中

  auto nodex = graph.FindNode("x");
  af::AscTensor tensorx = nodex->outputs[0];
  auto nodey = graph.FindNode("y");
  auto reduce_node = graph.FindNode("reduce");  // 获取Reduce节点
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

  // Setup second input (index) for ArgMaxMultiRPhase2
  af::ascir_op::Data z("z", graph);
  auto nodez = graph.FindNode("z");
  af::AscTensor tensorz = nodez->outputs[0];
  tensorz.attr.axis = {z0.id, z1.id, z2.id};
  tensorz.attr.vectorized_axis = {z1.id, z2.id};
  tensorz.attr.repeats = {z0.size, One, One};
  tensorz.attr.strides = {One, Zero, Zero};
  tensorz.attr.mem.tensor_id = 3;
  tensorz.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensorz.attr.mem.position = af::Position::kPositionVecIn;
  tensorz.attr.opt.merge_scope = af::kIdNone;
  tensorz.attr.buf.id = 6;
  tensorz.attr.vectorized_strides = vectorized_stride_y;
  tensorz.attr.dtype = ge::DT_INT64;

  std::string dtype_name;
  Tensor::DtypeName(tensorx.attr.dtype, dtype_name);
  // Setup inputs (value tensor)
  Tensor tensor1(tensorx, dtype_name);
  tensor1.is_constant = true;
  Tensor tensor2(tensorx, dtype_name);
  tensor2.is_constant = true;
  Tensor tensor3(tensorx, dtype_name);
  tensor3.is_constant = true;
  inputs.push_back(std::ref(tensor1));
  inputs.push_back(std::ref(tensor2));
  inputs.push_back(std::ref(tensor3));

  // Setup second input (index tensor)
  std::string index_dtype_name;
  Tensor::DtypeName(tensorz.attr.dtype, index_dtype_name);
  Tensor index_tensor(tensorz, index_dtype_name);
  index_tensor.is_constant = true;
  inputs.push_back(std::ref(index_tensor));

  // Setup outputs
  Tensor output_tensor(tensory, dtype_name);
  outputs.push_back(std::ref(output_tensor));

  codegen::ApiTensor x_tensor, z_tensor, y_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  z_tensor.id = tensorz.attr.mem.tensor_id;
  y_tensor.id = tensory.attr.mem.tensor_id;
  y_tensor.reuse_id = tensory.attr.mem.reuse_id;

  codegen::ReduceApiCall call(api_name);
  call.unit = af::ComputeUnit::kUnitVector;
  call.type = "ArgMaxMultiRPhase2";
  call.node = reduce_node;  // 设置Reduce节点，BuildApiParam需要有效的node
  call.node_name = reduce_node->GetName();
  y_tensor.write = &call;
  call.inputs.push_back(&x_tensor);
  call.inputs.push_back(&z_tensor);
  call.outputs.push_back(y_tensor);

  EXPECT_EQ(tpipe.AddTensor(tensor1), 0);
  EXPECT_EQ(tpipe.AddTensor(index_tensor), 0);
  EXPECT_EQ(tpipe.AddTensor(output_tensor), 0);

  std::string result;
  call.tmp_buf_id[-1] = 0;
  call.tmp_buf_id[0] = 1;
  Status status = call.Generate(tpipe, current_axis, result);
  // Check the result
  EXPECT_EQ(status, af::SUCCESS);
  CheckReduceTailBoolParam(result);
}
