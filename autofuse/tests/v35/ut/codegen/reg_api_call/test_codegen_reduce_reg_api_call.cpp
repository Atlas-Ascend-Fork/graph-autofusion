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
#include <memory>
#include "ascir_ops.h"
#include "v35/ascir/ascir_codegen_v2.h"
#include "codegen_kernel.h"
#include "utils/api_call_factory.h"
#include "../reg_api_call/reg_reduce_api_call.h"

#include "runtime_stub.h"
#include "platform_context.h"
#include "common_utils.h"

using namespace af::ops;
using namespace codegen;
using namespace af::ascir_op;
using namespace testing;

class RegReduceApicallTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v2 = std::make_shared<ge::RuntimeStubV2Common>();
    ge::RuntimeStub::SetInstance(stub_v2);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
  }
};

TEST_F(RegReduceApicallTest, RegReduceApi_Test_001) {
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
  af::AscTensor tensory = nodey->outputs[0];
  auto reduce_node = graph.FindNode("reduce");

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

  codegen::RegReduceApiCall call(api_name);
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
  const auto api_param = CodegenApiParam::GetNodeApiParam(reduce_node);
  ASSERT_NE(api_param, nullptr);
  EXPECT_EQ(api_param->semantic_feature, "AR/NoReuseSrc/MultiRound");
  EXPECT_NE(result.find("first_actual"), std::string::npos);
  EXPECT_NE(result.find("tmp_reduce_shape"), std::string::npos);
  EXPECT_NE(result.find("ReduceInit<"), std::string::npos) << result;
  EXPECT_NE(result.find("AscendC::PipeBarrier<PIPE_V>()"), std::string::npos) << result;
  EXPECT_NE(result.find("ReduceMaxExtend<"), std::string::npos) << result;
  EXPECT_NE(result.find("AscendC::Pattern::Reduce::AR"), std::string::npos);
  EXPECT_EQ(result.find("AscirNodeParams"), std::string::npos);
}

namespace {
void TestBoolLogicalReduceUsesUint8ExecutionView(const std::string &api_name, bool multi_reduce = false) {
  const bool is_any = api_name == "ReduceAny";

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
  auto reduce = is_any ? std::unique_ptr<af::Operator>(std::make_unique<af::ascir_op::Any>("reduce"))
                       : std::make_unique<af::ascir_op::All>("reduce");
  auto reduce_node = graph.AddNode(*reduce);

  auto nodex = graph.FindNode("x");
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
  tensorx.attr.dtype = af::DT_BOOL;
  tensorx.attr.vectorized_strides = {One, One};

  tensory.attr.axis = {z0.id, z1.id, z2.id};
  tensory.attr.vectorized_axis = {z0.id, z2.id};
  tensory.attr.repeats =
      multi_reduce ? std::vector<af::Expression>{One, One, One} : std::vector<af::Expression>{z0.size, One, One};
  if (multi_reduce) {
    tensory.attr.strides = {s0.expr, Zero, Zero};
  } else {
    tensory.attr.strides = {One, Zero, Zero};
  }
  tensory.attr.mem.tensor_id = 3;
  tensory.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensory.attr.mem.position = af::Position::kPositionVecOut;
  tensory.attr.opt.merge_scope = af::kIdNone;
  tensory.attr.buf.id = 4;
  tensory.attr.dtype = af::DT_BOOL;
  tensory.attr.vectorized_strides = {One, Zero};

  std::string dtype_name;
  Tensor::DtypeName(tensory.attr.dtype, dtype_name);
  Tensor tensor1(tensorx, dtype_name);
  Tensor output_tensor(tensory, dtype_name);
  inputs.push_back(std::ref(tensor1));
  outputs.push_back(std::ref(output_tensor));

  codegen::ApiTensor x_tensor;
  codegen::ApiTensor y_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  y_tensor.id = tensory.attr.mem.tensor_id;
  y_tensor.reuse_id = tensory.attr.mem.reuse_id;

  codegen::RegReduceApiCall call(api_name);
  call.unit = af::ComputeUnit::kUnitVector;
  call.type = is_any ? "Max" : "Min";
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
  ASSERT_EQ(call.Generate(tpipe, current_axis, result), af::SUCCESS) << result;

  const auto api_param = CodegenApiParam::GetNodeApiParam(reduce_node);
  ASSERT_NE(api_param, nullptr);
  ASSERT_EQ(api_param->template_params.size(), 3U);
  const auto block_start = std::find(api_param->api_pre_process.begin(), api_param->api_pre_process.end(), "{\n");
  ASSERT_NE(block_start, api_param->api_pre_process.end());
  const auto input_view = std::find_if(block_start, api_param->api_pre_process.end(), [](const std::string &line) {
    return line.find("_exec = ") != std::string::npos;
  });
  ASSERT_NE(input_view, api_param->api_pre_process.end());
  EXPECT_EQ(std::count_if(input_view, api_param->api_pre_process.end(),
                          [](const std::string &line) { return line.find("_exec = ") != std::string::npos; }),
            2);
  EXPECT_EQ(api_param->template_params[0], "uint8_t");
  EXPECT_NE(result.find(".template ReinterpretCast<uint8_t>()"), std::string::npos) << result;
  EXPECT_NE(result.find("ReduceInit<uint8_t, " + std::to_string(is_any ? 4 : 5)), std::string::npos) << result;
  EXPECT_NE(result.find(api_name + "Extend<uint8_t"), std::string::npos) << result;
  EXPECT_EQ(result.find(api_name + "Extend<bool"), std::string::npos) << result;
}
}  // namespace

TEST_F(RegReduceApicallTest, RegReduceApi_BoolAnyUsesUint8ExecutionView) {
  TestBoolLogicalReduceUsesUint8ExecutionView("ReduceAny");
}

TEST_F(RegReduceApicallTest, RegReduceApi_BoolAllUsesUint8ExecutionView) {
  TestBoolLogicalReduceUsesUint8ExecutionView("ReduceAll");
}

TEST_F(RegReduceApicallTest, RegReduceApi_BoolAnyMultiReduceUsesUint8ExecutionView) {
  TestBoolLogicalReduceUsesUint8ExecutionView("ReduceAny", true);
}

TEST_F(RegReduceApicallTest, RegReduceApi_AnyDirectIntegerDtypes) {
  const std::vector<std::pair<ge::DataType, std::string>> cases = {
      {ge::DT_INT8, "int8_t"}, {ge::DT_INT16, "int16_t"}, {ge::DT_INT32, "int32_t"}, {ge::DT_INT64, "int64_t"}};
  for (const auto &dtype_case : cases) {
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
    auto reduce = std::make_unique<af::ascir_op::Any>("reduce");
    auto reduce_node = graph.AddNode(*reduce);

    auto nodex = graph.FindNode("x");
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
    tensorx.attr.dtype = dtype_case.first;
    tensorx.attr.vectorized_strides = {One, One};

    tensory.attr.axis = {z0.id, z1.id, z2.id};
    tensory.attr.vectorized_axis = {z0.id, z2.id};
    tensory.attr.repeats = {z0.size, One, One};
    tensory.attr.strides = {One, Zero, Zero};
    tensory.attr.mem.tensor_id = 3;
    tensory.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
    tensory.attr.mem.position = af::Position::kPositionVecOut;
    tensory.attr.opt.merge_scope = af::kIdNone;
    tensory.attr.buf.id = 4;
    tensory.attr.dtype = dtype_case.first;
    tensory.attr.vectorized_strides = {One, Zero};

    std::string dtype_name;
    Tensor::DtypeName(tensory.attr.dtype, dtype_name);
    Tensor tensor1(tensorx, dtype_name);
    Tensor output_tensor(tensory, dtype_name);
    inputs.push_back(std::ref(tensor1));
    outputs.push_back(std::ref(output_tensor));

    codegen::ApiTensor x_tensor;
    codegen::ApiTensor y_tensor;
    x_tensor.id = tensorx.attr.mem.tensor_id;
    y_tensor.id = tensory.attr.mem.tensor_id;
    y_tensor.reuse_id = tensory.attr.mem.reuse_id;

    codegen::RegReduceApiCall call("ReduceAny");
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
    ASSERT_EQ(call.Generate(tpipe, current_axis, result), af::SUCCESS) << result;
    const auto api_param = CodegenApiParam::GetNodeApiParam(reduce_node);
    ASSERT_NE(api_param, nullptr);
    ASSERT_EQ(api_param->template_params.size(), 3U);
    EXPECT_EQ(api_param->template_params[0], dtype_case.second);
    EXPECT_NE(result.find("ReduceAnyExtend<" + dtype_case.second), std::string::npos) << result;
    EXPECT_EQ(result.find("LocalTensor<uint8_t>"), std::string::npos) << result;
  }
}

TEST_F(RegReduceApicallTest, RegReduceApi_AllDirectIntegerDtypes) {
  const std::vector<std::pair<ge::DataType, std::string>> cases = {
      {ge::DT_INT8, "int8_t"}, {ge::DT_INT16, "int16_t"}, {ge::DT_INT32, "int32_t"}, {ge::DT_INT64, "int64_t"}};
  for (const auto &dtype_case : cases) {
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
    auto reduce = std::make_unique<af::ascir_op::All>("reduce");
    auto reduce_node = graph.AddNode(*reduce);

    auto nodex = graph.FindNode("x");
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
    tensorx.attr.dtype = dtype_case.first;
    tensorx.attr.vectorized_strides = {One, One};

    tensory.attr.axis = {z0.id, z1.id, z2.id};
    tensory.attr.vectorized_axis = {z0.id, z2.id};
    tensory.attr.repeats = {z0.size, One, One};
    tensory.attr.strides = {One, Zero, Zero};
    tensory.attr.mem.tensor_id = 3;
    tensory.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
    tensory.attr.mem.position = af::Position::kPositionVecOut;
    tensory.attr.opt.merge_scope = af::kIdNone;
    tensory.attr.buf.id = 4;
    tensory.attr.dtype = dtype_case.first;
    tensory.attr.vectorized_strides = {One, Zero};

    std::string dtype_name;
    Tensor::DtypeName(tensory.attr.dtype, dtype_name);
    Tensor tensor1(tensorx, dtype_name);
    Tensor output_tensor(tensory, dtype_name);
    inputs.push_back(std::ref(tensor1));
    outputs.push_back(std::ref(output_tensor));

    codegen::ApiTensor x_tensor;
    codegen::ApiTensor y_tensor;
    x_tensor.id = tensorx.attr.mem.tensor_id;
    y_tensor.id = tensory.attr.mem.tensor_id;
    y_tensor.reuse_id = tensory.attr.mem.reuse_id;

    codegen::RegReduceApiCall call("ReduceAll");
    call.unit = af::ComputeUnit::kUnitVector;
    call.type = "Min";
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
    ASSERT_EQ(call.Generate(tpipe, current_axis, result), af::SUCCESS) << result;
    const auto api_param = CodegenApiParam::GetNodeApiParam(reduce_node);
    ASSERT_NE(api_param, nullptr);
    ASSERT_EQ(api_param->template_params.size(), 3U);
    EXPECT_EQ(api_param->template_params[0], dtype_case.second);
    EXPECT_NE(result.find("ReduceAllExtend<" + dtype_case.second), std::string::npos) << result;
    EXPECT_EQ(result.find("LocalTensor<uint8_t>"), std::string::npos) << result;
  }
}

TEST_F(RegReduceApicallTest, RegReduceApi_Test_002) {
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
  af::AscTensor tensory = nodey->outputs[0];
  auto reduce_node = graph.FindNode("reduce");

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

  codegen::RegReduceApiCall call(api_name);
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

TEST_F(RegReduceApicallTest, RegReduceApi_Test_003) {
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
  af::AscTensor tensory = nodey->outputs[0];
  auto reduce_node = graph.FindNode("reduce");

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

  codegen::RegReduceApiCall call(api_name);
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

TEST_F(RegReduceApicallTest, RegReduceApi_Test_004) {
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
  af::AscTensor tensory = nodey->outputs[0];
  auto reduce_node = graph.FindNode("reduce");

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

  codegen::RegReduceApiCall call(api_name);
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

TEST_F(RegReduceApicallTest, RegReduceApicallTest_Int32_Inner) {
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
  af::AscTensor tensory = nodey->outputs[0];
  auto reduce_node = graph.FindNode("reduce");

  tensorx.attr.axis = {z0.id, z1.id, z2.id};
  tensorx.attr.vectorized_axis = {z0.id, z2.id};
  tensorx.attr.repeats = {z0.size, z1.size, z2.size};
  tensorx.attr.strides = {z1.size * z2.size, z2.size, One};
  tensorx.attr.mem.tensor_id = 1;
  tensorx.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensorx.attr.mem.position = af::Position::kPositionVecIn;
  tensorx.attr.opt.merge_scope = af::kIdNone;
  tensorx.attr.buf.id = 2;
  tensorx.attr.dtype = af::DT_INT32;
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
  tensory.attr.dtype = af::DT_INT32;
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

  codegen::RegReduceApiCall call(api_name);
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

TEST_F(RegReduceApicallTest, RegReduceApicallTest_Int32_Outer) {
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
  af::AscTensor tensory = nodey->outputs[0];
  auto reduce_node = graph.FindNode("reduce");

  tensorx.attr.axis = {z0.id, z1.id, z2.id};
  tensorx.attr.vectorized_axis = {z0.id, z2.id};
  tensorx.attr.repeats = {z0.size, z1.size, z2.size};
  tensorx.attr.strides = {z1.size * z2.size, z2.size, One};
  tensorx.attr.mem.tensor_id = 1;
  tensorx.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensorx.attr.mem.position = af::Position::kPositionVecIn;
  tensorx.attr.opt.merge_scope = af::kIdNone;
  tensorx.attr.buf.id = 2;
  tensorx.attr.dtype = af::DT_INT32;
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
  tensory.attr.dtype = af::DT_INT32;
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

  codegen::RegReduceApiCall call(api_name);
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

TEST_F(RegReduceApicallTest, RegReduceApicallTest_ReduceMean_NoNeed_MultiReduce_Int32) {
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
  af::AscTensor tensory = nodey->outputs[0];
  auto reduce_node = graph.FindNode("reduce");

  tensorx.attr.axis = {z0.id, z1.id, z2.id};
  tensorx.attr.vectorized_axis = {z1.id, z2.id};
  tensorx.attr.repeats = {z0.size, z1.size, z2.size};
  tensorx.attr.strides = {z1.size * z2.size, z2.size, One};
  tensorx.attr.mem.tensor_id = 1;
  tensorx.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensorx.attr.mem.position = af::Position::kPositionVecIn;
  tensorx.attr.opt.merge_scope = af::kIdNone;
  tensorx.attr.buf.id = 2;
  tensorx.attr.dtype = af::DT_INT32;
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
  tensory.attr.dtype = af::DT_INT32;
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

  codegen::RegReduceApiCall call(api_name);
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

TEST_F(RegReduceApicallTest, RegReduceApicallTest_ReduceMean_NoNeed_MultiReduce_Float) {
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
  af::AscTensor tensory = nodey->outputs[0];
  auto reduce_node = graph.FindNode("reduce");

  tensorx.attr.axis = {z0.id, z1.id, z2.id};
  tensorx.attr.vectorized_axis = {z1.id, z2.id};
  tensorx.attr.repeats = {z0.size, z1.size, z2.size};
  tensorx.attr.strides = {z1.size * z2.size, z2.size, One};
  tensorx.attr.mem.tensor_id = 1;
  tensorx.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensorx.attr.mem.position = af::Position::kPositionVecIn;
  tensorx.attr.opt.merge_scope = af::kIdNone;
  tensorx.attr.buf.id = 2;
  tensorx.attr.dtype = af::DT_FLOAT;
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
  tensory.attr.dtype = af::DT_FLOAT;
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

  codegen::RegReduceApiCall call(api_name);
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

TEST_F(RegReduceApicallTest, RegReduceApicallTest_ParseAttr) {
  af::AscGraph graph("test");
  af::Expression One = af::Symbol(1);
  af::Expression Zero = af::Symbol(0);
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data x1("x1", graph);
  Load load1("load1");
  Max max0("max0");
  Store store("store");
  Output y("y");

  x1.attr.sched.axis = {z0.id, z1.id, z2.id};
  x1.y.dtype = af::DT_FLOAT;
  *x1.y.axis = {z0.id, z1.id, z2.id};
  *x1.y.repeats = {s0, s1, s2};
  *x1.y.strides = {s1 * s2, s2, One};

  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.y.dtype = af::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, One};
  *load1.y.vectorized_axis = {z1.id, z2.id};

  max0.x = load1.y;
  max0.attr.sched.axis = {z0.id, z1.id, z2.id};
  max0.attr.sched.loop_axis = {z0.id};
  max0.y.dtype = af::DT_FLOAT;
  *max0.y.axis = {z0.id, z1.id, z2.id};
  *max0.y.repeats = {s0, s1, One};
  *max0.y.strides = {s2, One, Zero};
  *max0.y.vectorized_axis = {z1.id, z2.id};

  store.x = max0.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1 * s2, s2, One};

  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id};
  y.y.dtype = af::DT_FLOAT;
  *y.y.axis = {z0.id, z1.id, z2.id};
  *y.y.repeats = {s0, s1, s2};
  *y.y.strides = {s1 * s2, s2, One};

  std::string api_name = "Max";
  RegReduceApiCall call(api_name);
  auto node_max = graph.FindNode("max0");
  EXPECT_EQ(call.CallParseAttr(node_max), af::SUCCESS);
}

TEST_F(RegReduceApicallTest, ReduceSumInt8AndInt16DoNotRequireCast) {
  for (const auto dtype : {af::DT_INT8, af::DT_INT16}) {
    SCOPED_TRACE(::testing::Message() << "dtype=" << static_cast<int32_t>(dtype));
    af::AscGraph graph("reduce_sum_integer_dtype");

    af::ascir_op::Data data("data", graph);
    data.y.dtype = dtype;

    af::ascir_op::Sum sum("sum");
    sum.x = data.y;
    sum.y.dtype = dtype;

    auto sum_node = graph.FindNode("sum");
    ASSERT_NE(sum_node, nullptr);
    auto codegen_impl = ascgen_utils::GetAscIrCodegenImpl(sum_node->GetType());
    ASSERT_NE(codegen_impl, nullptr);

    const auto [input_dtypes, output_dtypes] = codegen_impl->GetConversionDtype(*sum_node);
    ASSERT_EQ(input_dtypes.size(), 1U);
    ASSERT_EQ(output_dtypes.size(), 1U);
    EXPECT_EQ(input_dtypes[0], dtype);
    EXPECT_EQ(output_dtypes[0], dtype);
  }
}

TEST_F(RegReduceApicallTest, ReduceProdIntegerDtypeConversions) {
  const std::vector<std::pair<af::DataType, af::DataType>> dtype_cases = {{af::DT_INT8, af::DT_INT16},
                                                                          {af::DT_INT16, af::DT_INT16},
                                                                          {af::DT_INT32, af::DT_INT32},
                                                                          {af::DT_INT64, af::DT_INT64},
                                                                          {af::DT_FLOAT, af::DT_FLOAT}};

  for (const auto &[dtype, exec_dtype] : dtype_cases) {
    SCOPED_TRACE(::testing::Message() << "dtype=" << static_cast<int32_t>(dtype));
    af::AscGraph graph("reduce_prod_integer_dtype");

    af::ascir_op::Data data("data", graph);
    data.y.dtype = dtype;

    af::ascir_op::Prod prod("prod");
    prod.x = data.y;
    prod.y.dtype = dtype;

    auto prod_node = graph.FindNode("prod");
    ASSERT_NE(prod_node, nullptr);
    auto codegen_impl = ascgen_utils::GetAscIrCodegenImpl(prod_node->GetType());
    ASSERT_NE(codegen_impl, nullptr);

    const auto [input_dtypes, output_dtypes] = codegen_impl->GetConversionDtype(*prod_node);
    auto *codegen_impl_v2 = dynamic_cast<af::ascir::AscIrCodegenV2 *>(codegen_impl.get());
    ASSERT_NE(codegen_impl_v2, nullptr);
    const auto output_cast_paths = codegen_impl_v2->GetOutputConversionPath(*prod_node);
    ASSERT_EQ(input_dtypes.size(), 1U);
    ASSERT_EQ(output_dtypes.size(), 1U);
    EXPECT_EQ(input_dtypes[0], exec_dtype);
    EXPECT_EQ(output_dtypes[0], exec_dtype);
    if (dtype == af::DT_INT8) {
      ASSERT_EQ(output_cast_paths.size(), 1U);
      ASSERT_EQ(output_cast_paths[0], (std::vector<af::DataType>({af::DT_UINT8, af::DT_INT8})));
    } else {
      EXPECT_TRUE(output_cast_paths.empty());
    }
  }
}

TEST_F(RegReduceApicallTest, ProdInt8EpilogueMulRecognizesTruncationCastChain) {
  af::AscGraph graph("prod_int8_mul_cast_chain");

  af::ascir_op::Data data("data", graph);
  data.y.dtype = af::DT_INT8;
  af::ascir_op::Cast cast_to_int16("cast_to_int16");
  cast_to_int16.x = data.y;
  cast_to_int16.y.dtype = af::DT_INT16;

  af::ascir_op::Prod prod("prod");
  prod.x = cast_to_int16.y;
  prod.y.dtype = af::DT_INT16;

  af::ascir_op::Cast cast_to_uint8("cast_to_uint8");
  cast_to_uint8.x = prod.y;
  cast_to_uint8.y.dtype = af::DT_UINT8;
  af::ascir_op::Cast cast_to_int8("cast_to_int8");
  cast_to_int8.x = cast_to_uint8.y;
  cast_to_int8.y.dtype = af::DT_INT8;

  af::ascir_op::Data other("other", graph);
  other.y.dtype = af::DT_INT8;
  af::ascir_op::Mul mul("mul");
  mul.x1 = cast_to_int8.y;
  mul.x2 = other.y;
  mul.y.dtype = af::DT_INT8;

  auto mul_node = graph.FindNode("mul");
  ASSERT_NE(mul_node, nullptr);
  auto codegen_impl = ascgen_utils::GetAscIrCodegenImpl(mul_node->GetType());
  ASSERT_NE(codegen_impl, nullptr);
  auto *codegen_impl_v2 = dynamic_cast<af::ascir::AscIrCodegenV2 *>(codegen_impl.get());
  ASSERT_NE(codegen_impl_v2, nullptr);
  const auto output_cast_paths = codegen_impl_v2->GetOutputConversionPath(*mul_node);
  ASSERT_EQ(output_cast_paths.size(), 1U);
  ASSERT_EQ(output_cast_paths[0], (std::vector<af::DataType>({af::DT_UINT8, af::DT_INT8})));
}

TEST_F(RegReduceApicallTest, ReduceProdUsesReuseSourceWhenSourceIsExclusive) {
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
  af::AscGraph graph("reduce_prod_no_reuse_source");

  af::ascir_op::Data x("x", graph);
  x.y.dtype = af::DT_INT16;

  af::ascir_op::Prod prod("prod");
  prod.x = x.y;
  prod.y.dtype = af::DT_INT16;

  auto prod_node = graph.FindNode("prod");
  ASSERT_NE(prod_node, nullptr);

  auto node_x = graph.FindNode("x");
  af::AscTensor tensor_x = node_x->outputs[0];
  tensor_x.attr.axis = {z0.id, z1.id, z2.id};
  tensor_x.attr.vectorized_axis = {z0.id, z2.id};
  tensor_x.attr.repeats = {z0.size, z1.size, z2.size};
  tensor_x.attr.strides = {z1.size * z2.size, z2.size, One};
  tensor_x.attr.mem.tensor_id = 1;
  tensor_x.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensor_x.attr.mem.position = af::Position::kPositionVecIn;
  tensor_x.attr.opt.merge_scope = af::kIdNone;
  tensor_x.attr.buf.id = 2;
  tensor_x.attr.dtype = af::DT_INT16;
  tensor_x.attr.vectorized_strides = {One, One};

  af::AscTensor tensor_y = prod_node->outputs[0];
  tensor_y.attr.axis = {z0.id, z1.id, z2.id};
  tensor_y.attr.vectorized_axis = {z0.id, z2.id};
  tensor_y.attr.repeats = {z0.size, One, One};
  tensor_y.attr.strides = {One, Zero, Zero};
  tensor_y.attr.mem.tensor_id = 3;
  tensor_y.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
  tensor_y.attr.mem.position = af::Position::kPositionVecOut;
  tensor_y.attr.opt.merge_scope = af::kIdNone;
  tensor_y.attr.buf.id = 4;
  tensor_y.attr.dtype = af::DT_INT16;
  tensor_y.attr.vectorized_strides = {One, Zero};

  RegReduceApiCall call("ReduceProd");
  std::string dtype_name;
  Tensor::DtypeName(tensor_y.attr.dtype, dtype_name);
  Tensor input_tensor(tensor_x, dtype_name);
  Tensor output_tensor(tensor_y, dtype_name);
  inputs.push_back(std::ref(input_tensor));
  outputs.push_back(std::ref(output_tensor));

  codegen::ApiTensor x_tensor;
  codegen::ApiTensor y_tensor;
  x_tensor.id = tensor_x.attr.mem.tensor_id;
  y_tensor.id = tensor_y.attr.mem.tensor_id;
  y_tensor.reuse_id = tensor_y.attr.mem.reuse_id;

  EXPECT_EQ(tpipe.AddTensor(input_tensor), 0);
  EXPECT_EQ(tpipe.AddTensor(output_tensor), 0);

  ASSERT_EQ(call.CallParseAttr(prod_node), af::SUCCESS);
  call.unit = af::ComputeUnit::kUnitVector;
  call.type = "Prod";
  call.node = prod_node;
  call.node_name = prod_node->GetName();
  y_tensor.write = &call;
  call.inputs.push_back(&x_tensor);
  call.outputs.push_back(y_tensor);

  std::string result;
  call.tmp_buf_id[-1] = 0;
  call.tmp_buf_id[0] = 1;
  ASSERT_EQ(call.Generate(tpipe, current_axis, result), af::SUCCESS) << result;
  const auto api_param = CodegenApiParam::GetNodeApiParam(prod_node);
  ASSERT_NE(api_param, nullptr);
  EXPECT_EQ(api_param->semantic_feature, "AR/ReuseSrc/MultiRound");
}
