/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "../reg_api_call/reg_argmax_api_call.h"

#include "runtime_stub.h"
#include "platform_context.h"
#include "common_utils.h"

using namespace af::ops;
using namespace codegen;
using namespace af::ascir_op;
using namespace testing;

class RegArgmaxApicallTest : public ::testing::Test {
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

  // 构造 AR 归约的 Argmax 图：x{z0, z1}，归约沿 z1（尾轴），输出 y{z0}
  static void BuildArgmaxGraph(af::AscGraph &graph, af::Expression &s0, af::Expression &s1, af::Axis &z0,
                               af::Axis &z1) {
    s0 = graph.CreateSizeVar("s0");
    s1 = graph.CreateSizeVar("s1");
    z0 = graph.CreateAxis("z0", s0);
    z1 = graph.CreateAxis("z1", s1);
    Data data("data", graph);
    data.y.dtype = ge::DT_FLOAT;
    data.ir_attr.SetIndex(0);
    Load load("load");
    load.attr.sched.axis = {z0.id, z1.id};
    load.x = data.y;
    *load.y.axis = {z0.id, z1.id};
    load.y.dtype = ge::DT_FLOAT;
    *load.y.strides = {s1, af::ops::One};
    *load.y.repeats = {s0, s1};
    af::ascir_op::Argmax argmax("argmax");
    argmax.attr.sched.axis = {z0.id, z1.id};
    argmax.attr.api.compute_type = af::ComputeType::kComputeReduce;
    argmax.x = load.y;
    *argmax.y.axis = {z0.id, z1.id};
    argmax.y.dtype = ge::DT_INT64;
    *argmax.y.repeats = {s0, af::ops::One};
    *argmax.y.strides = {af::ops::One, af::ops::Zero};
    Store store("store");
    store.attr.sched.axis = {z0.id, z1.id};
    store.x = argmax.y;
    *store.y.axis = {z0.id, z1.id};
    store.y.dtype = ge::DT_INT64;
    *store.y.repeats = {s0, af::ops::One};
    *store.y.strides = {af::ops::One, af::ops::Zero};
    Output output("output");
    output.x = store.y;
    output.y.dtype = ge::DT_INT64;
    output.ir_attr.SetIndex(0);
  }

  static void FillReduceTensorAttrs(af::AscTensor &tensorx, af::AscTensor &tensory, const af::Axis &z0,
                                    const af::Axis &z1, const af::Expression &s0, const af::Expression &s1,
                                    ge::DataType in_dtype, ge::DataType out_dtype) {
    tensorx.attr.axis = {z0.id, z1.id};
    tensorx.attr.vectorized_axis = {z0.id, z1.id};
    tensorx.attr.repeats = {s0, s1};
    tensorx.attr.strides = {s1, One};
    tensorx.attr.mem.tensor_id = 1;
    tensorx.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
    tensorx.attr.mem.position = af::Position::kPositionVecIn;
    tensorx.attr.opt.merge_scope = af::kIdNone;
    tensorx.attr.buf.id = 2;
    tensorx.attr.dtype = in_dtype;
    vector<af::Expression> vectorized_stride_x{One, One};
    tensorx.attr.vectorized_strides = vectorized_stride_x;

    tensory.attr.axis = {z0.id, z1.id};
    tensory.attr.vectorized_axis = {z0.id, z1.id};
    tensory.attr.repeats = {s0, One};
    tensory.attr.strides = {One, Zero};
    tensory.attr.mem.tensor_id = 3;
    tensory.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
    tensory.attr.mem.position = af::Position::kPositionVecOut;
    tensory.attr.opt.merge_scope = af::kIdNone;
    tensory.attr.buf.id = 4;
    tensory.attr.dtype = out_dtype;
    vector<af::Expression> vectorized_stride_y{One, Zero};
    tensory.attr.vectorized_strides = vectorized_stride_y;
  }
};

TEST_F(RegArgmaxApicallTest, RegArgmaxApi_Test_001) {
  af::AscGraph graph("argmax_graph");
  af::Expression s0;
  af::Expression s1;
  af::Axis z0;
  af::Axis z1;
  BuildArgmaxGraph(graph, s0, s1, z0, z1);

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  std::vector<ascir::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::TPipe tpipe("tpipe", tiler);
  auto argmax_node = graph.FindNode("argmax");

  af::AscTensor tensorx = graph.FindNode("load")->outputs[0];
  af::AscTensor tensory = argmax_node->outputs[0];
  FillReduceTensorAttrs(tensorx, tensory, z0, z1, s0, s1, ge::DT_FLOAT, ge::DT_INT64);

  std::string in_dtype_name;
  Tensor::DtypeName(tensorx.attr.dtype, in_dtype_name);
  std::string out_dtype_name;
  Tensor::DtypeName(tensory.attr.dtype, out_dtype_name);

  std::vector<std::reference_wrapper<const Tensor>> inputs;
  std::vector<std::reference_wrapper<const Tensor>> outputs;
  Tensor input_tensor(tensorx, in_dtype_name);
  Tensor output_tensor(tensory, out_dtype_name);
  inputs.push_back(std::ref(input_tensor));
  outputs.push_back(std::ref(output_tensor));

  codegen::ApiTensor x_tensor, y_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  y_tensor.id = tensory.attr.mem.tensor_id;
  y_tensor.reuse_id = tensory.attr.mem.reuse_id;

  codegen::RegArgmaxApiCall call("Argmax");
  call.unit = af::ComputeUnit::kUnitVector;
  call.type = "Argmax";
  call.node = argmax_node;
  call.node_name = argmax_node->GetName();
  y_tensor.write = &call;
  call.inputs.push_back(&x_tensor);
  call.outputs.push_back(y_tensor);
  call.tmp_buf_id[0] = 1;

  EXPECT_EQ(tpipe.AddTensor(input_tensor), 0);
  EXPECT_EQ(tpipe.AddTensor(output_tensor), 0);

  std::string result;
  Status status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(status, af::SUCCESS);
  EXPECT_NE(result.find("first_actual"), std::string::npos);
  EXPECT_NE(result.find("tmp_reduce_shape"), std::string::npos);
  EXPECT_NE(result.find("ArgMaxExtend<"), std::string::npos) << result;
  EXPECT_NE(result.find("tmp_reduce_shape, false, z1_actual_size);"), std::string::npos) << result;
  EXPECT_EQ(result.find("tmp_reduce_shape, false, last_actual);"), std::string::npos) << result;
  EXPECT_NE(result.find("int64_t"), std::string::npos) << result;
  EXPECT_NE(result.find("float"), std::string::npos) << result;
  EXPECT_NE(result.find("AscendC::Pattern::Reduce::AR"), std::string::npos);
  // 无 tmp buffer：不生成 tmp_reduce 临时变量（tmp_reduce_shape 是 srcShape 数组，应存在）
  EXPECT_EQ(result.find("tmp_reduce ="), std::string::npos);
  EXPECT_NE(result.find("CastExtend<int32_t, int64_t, 1>"), std::string::npos) << result;
}

TEST_F(RegArgmaxApicallTest, RegArgmaxApi_Test_002) {
  // int32 输入 + int64 索引：验证模板参数使用输入 dtype
  af::AscGraph graph("argmax_int32_graph");
  af::Expression s0;
  af::Expression s1;
  af::Axis z0;
  af::Axis z1;
  BuildArgmaxGraph(graph, s0, s1, z0, z1);

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  std::vector<ascir::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::TPipe tpipe("tpipe", tiler);
  auto argmax_node = graph.FindNode("argmax");

  af::AscTensor tensorx = graph.FindNode("load")->outputs[0];
  af::AscTensor tensory = argmax_node->outputs[0];
  FillReduceTensorAttrs(tensorx, tensory, z0, z1, s0, s1, ge::DT_INT32, ge::DT_INT64);

  std::string in_dtype_name;
  Tensor::DtypeName(tensorx.attr.dtype, in_dtype_name);
  std::string out_dtype_name;
  Tensor::DtypeName(tensory.attr.dtype, out_dtype_name);

  std::vector<std::reference_wrapper<const Tensor>> inputs;
  std::vector<std::reference_wrapper<const Tensor>> outputs;
  Tensor input_tensor(tensorx, in_dtype_name);
  Tensor output_tensor(tensory, out_dtype_name);
  inputs.push_back(std::ref(input_tensor));
  outputs.push_back(std::ref(output_tensor));

  codegen::ApiTensor x_tensor, y_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  y_tensor.id = tensory.attr.mem.tensor_id;
  y_tensor.reuse_id = tensory.attr.mem.reuse_id;

  codegen::RegArgmaxApiCall call("Argmax");
  call.unit = af::ComputeUnit::kUnitVector;
  call.type = "Argmax";
  call.node = argmax_node;
  call.node_name = argmax_node->GetName();
  y_tensor.write = &call;
  call.inputs.push_back(&x_tensor);
  call.outputs.push_back(y_tensor);
  call.tmp_buf_id[0] = 1;

  EXPECT_EQ(tpipe.AddTensor(input_tensor), 0);
  EXPECT_EQ(tpipe.AddTensor(output_tensor), 0);

  std::string result;
  Status status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(status, af::SUCCESS);
  EXPECT_NE(result.find("ArgMaxExtend<int32_t,"), std::string::npos) << result;
  EXPECT_EQ(result.find("tmp_reduce ="), std::string::npos);
}

TEST_F(RegArgmaxApicallTest, RegArgmaxWithValueApi_Test_001) {
  // 前端算子 ArgmaxWithValue（1 入 2 出）：float 输入 + int64 索引
  af::AscGraph graph("argmax_with_value_graph");
  af::Expression s0;
  af::Expression s1;
  af::Axis z0;
  af::Axis z1;
  s0 = graph.CreateSizeVar("s0");
  s1 = graph.CreateSizeVar("s1");
  z0 = graph.CreateAxis("z0", s0);
  z1 = graph.CreateAxis("z1", s1);
  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);
  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1, af::ops::One};
  *load.y.repeats = {s0, s1};
  af::ascir_op::ArgmaxWithValue argmax_with_value("argmax_with_value");
  argmax_with_value.attr.sched.axis = {z0.id, z1.id};
  argmax_with_value.attr.api.compute_type = af::ComputeType::kComputeReduce;
  argmax_with_value.x = load.y;
  *argmax_with_value.value.axis = {z0.id, z1.id};
  argmax_with_value.value.dtype = ge::DT_FLOAT;
  *argmax_with_value.value.repeats = {s0, af::ops::One};
  *argmax_with_value.value.strides = {af::ops::One, af::ops::Zero};
  *argmax_with_value.index.axis = {z0.id, z1.id};
  argmax_with_value.index.dtype = ge::DT_INT64;
  *argmax_with_value.index.repeats = {s0, af::ops::One};
  *argmax_with_value.index.strides = {af::ops::One, af::ops::Zero};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  std::vector<ascir::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::TPipe tpipe("tpipe", tiler);
  auto argmax_node = graph.FindNode("argmax_with_value");

  af::AscTensor tensorx = graph.FindNode("load")->outputs[0];
  af::AscTensor tensorv = argmax_node->outputs[0];  // value
  af::AscTensor tensori = argmax_node->outputs[1];  // index
  FillReduceTensorAttrs(tensorx, tensorv, z0, z1, s0, s1, ge::DT_FLOAT, ge::DT_FLOAT);
  tensori.attr.mem.tensor_id = 5;
  tensori.attr.mem.position = af::Position::kPositionVecOut;
  tensori.attr.buf.id = 6;
  tensori.attr.dtype = ge::DT_INT64;

  std::string in_dtype_name;
  Tensor::DtypeName(tensorx.attr.dtype, in_dtype_name);
  std::string value_dtype_name;
  Tensor::DtypeName(tensorv.attr.dtype, value_dtype_name);
  std::string index_dtype_name;
  Tensor::DtypeName(tensori.attr.dtype, index_dtype_name);

  std::vector<std::reference_wrapper<const Tensor>> inputs;
  std::vector<std::reference_wrapper<const Tensor>> outputs;
  Tensor input_tensor(tensorx, in_dtype_name);
  Tensor value_tensor(tensorv, value_dtype_name);
  Tensor index_tensor(tensori, index_dtype_name);
  inputs.push_back(std::ref(input_tensor));
  outputs.push_back(std::ref(value_tensor));
  outputs.push_back(std::ref(index_tensor));

  codegen::ApiTensor x_tensor, value_api_tensor, index_api_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  value_api_tensor.id = tensorv.attr.mem.tensor_id;
  value_api_tensor.reuse_id = tensorv.attr.mem.reuse_id;
  index_api_tensor.id = tensori.attr.mem.tensor_id;
  index_api_tensor.reuse_id = tensori.attr.mem.reuse_id;

  codegen::RegArgmaxWithValueApiCall call("ArgmaxWithValue");
  call.unit = af::ComputeUnit::kUnitVector;
  call.type = "ArgmaxWithValue";
  call.node = argmax_node;
  call.node_name = argmax_node->GetName();
  value_api_tensor.write = &call;
  index_api_tensor.write = &call;
  call.inputs.push_back(&x_tensor);
  call.outputs.push_back(value_api_tensor);
  call.outputs.push_back(index_api_tensor);
  call.tmp_buf_id[0] = 1;

  EXPECT_EQ(tpipe.AddTensor(input_tensor), 0);
  EXPECT_EQ(tpipe.AddTensor(value_tensor), 0);
  EXPECT_EQ(tpipe.AddTensor(index_tensor), 0);

  std::string result;
  Status status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(status, af::SUCCESS);
  EXPECT_NE(result.find("ArgMaxWithValueExtend<float, AscendC::Pattern::Reduce::AR>"), std::string::npos) << result;
  EXPECT_NE(result.find("tmp_reduce_shape"), std::string::npos) << result;
  // 无 tmp buffer：不生成 tmp_reduce 临时变量与 tmp_buf 访问
  EXPECT_EQ(result.find("tmp_reduce ="), std::string::npos);
  EXPECT_NE(result.find("CastExtend<int32_t, int64_t, 1>"), std::string::npos) << result;
}

TEST_F(RegArgmaxApicallTest, RegArgmaxWithValueApi_Test_002) {
  // 前端算子 ArgmaxWithValue（1 入 2 出）：int64 输入 + int64 索引（FR13）
  af::AscGraph graph("argmax_with_value_int64_graph");
  af::Expression s0;
  af::Expression s1;
  af::Axis z0;
  af::Axis z1;
  s0 = graph.CreateSizeVar("s0");
  s1 = graph.CreateSizeVar("s1");
  z0 = graph.CreateAxis("z0", s0);
  z1 = graph.CreateAxis("z1", s1);
  Data data("data", graph);
  data.y.dtype = ge::DT_INT64;
  data.ir_attr.SetIndex(0);
  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_INT64;
  *load.y.strides = {s1, af::ops::One};
  *load.y.repeats = {s0, s1};
  af::ascir_op::ArgmaxWithValue argmax_with_value("argmax_with_value");
  argmax_with_value.attr.sched.axis = {z0.id, z1.id};
  argmax_with_value.attr.api.compute_type = af::ComputeType::kComputeReduce;
  argmax_with_value.x = load.y;
  *argmax_with_value.value.axis = {z0.id, z1.id};
  argmax_with_value.value.dtype = ge::DT_INT64;
  *argmax_with_value.value.repeats = {s0, af::ops::One};
  *argmax_with_value.value.strides = {af::ops::One, af::ops::Zero};
  *argmax_with_value.index.axis = {z0.id, z1.id};
  argmax_with_value.index.dtype = ge::DT_INT64;
  *argmax_with_value.index.repeats = {s0, af::ops::One};
  *argmax_with_value.index.strides = {af::ops::One, af::ops::Zero};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  std::vector<ascir::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::TPipe tpipe("tpipe", tiler);
  auto argmax_node = graph.FindNode("argmax_with_value");

  af::AscTensor tensorx = graph.FindNode("load")->outputs[0];
  af::AscTensor tensorv = argmax_node->outputs[0];  // value
  af::AscTensor tensori = argmax_node->outputs[1];  // index
  FillReduceTensorAttrs(tensorx, tensorv, z0, z1, s0, s1, ge::DT_INT64, ge::DT_INT64);
  tensori.attr.mem.tensor_id = 5;
  tensori.attr.mem.position = af::Position::kPositionVecOut;
  tensori.attr.buf.id = 6;
  tensori.attr.dtype = ge::DT_INT64;

  std::string in_dtype_name;
  Tensor::DtypeName(tensorx.attr.dtype, in_dtype_name);
  std::string value_dtype_name;
  Tensor::DtypeName(tensorv.attr.dtype, value_dtype_name);
  std::string index_dtype_name;
  Tensor::DtypeName(tensori.attr.dtype, index_dtype_name);

  std::vector<std::reference_wrapper<const Tensor>> inputs;
  std::vector<std::reference_wrapper<const Tensor>> outputs;
  Tensor input_tensor(tensorx, in_dtype_name);
  Tensor value_tensor(tensorv, value_dtype_name);
  Tensor index_tensor(tensori, index_dtype_name);
  inputs.push_back(std::ref(input_tensor));
  outputs.push_back(std::ref(value_tensor));
  outputs.push_back(std::ref(index_tensor));

  codegen::ApiTensor x_tensor, value_api_tensor, index_api_tensor;
  x_tensor.id = tensorx.attr.mem.tensor_id;
  value_api_tensor.id = tensorv.attr.mem.tensor_id;
  value_api_tensor.reuse_id = tensorv.attr.mem.reuse_id;
  index_api_tensor.id = tensori.attr.mem.tensor_id;
  index_api_tensor.reuse_id = tensori.attr.mem.reuse_id;

  codegen::RegArgmaxWithValueApiCall call("ArgmaxWithValue");
  call.unit = af::ComputeUnit::kUnitVector;
  call.type = "ArgmaxWithValue";
  call.node = argmax_node;
  call.node_name = argmax_node->GetName();
  value_api_tensor.write = &call;
  index_api_tensor.write = &call;
  call.inputs.push_back(&x_tensor);
  call.outputs.push_back(value_api_tensor);
  call.outputs.push_back(index_api_tensor);
  call.tmp_buf_id[0] = 1;

  EXPECT_EQ(tpipe.AddTensor(input_tensor), 0);
  EXPECT_EQ(tpipe.AddTensor(value_tensor), 0);
  EXPECT_EQ(tpipe.AddTensor(index_tensor), 0);

  std::string result;
  Status status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(status, af::SUCCESS);
  EXPECT_NE(result.find("ArgMaxWithValueExtend<int64_t, AscendC::Pattern::Reduce::AR>"), std::string::npos) << result;
  EXPECT_NE(result.find("tmp_reduce_shape"), std::string::npos) << result;
  // 无 tmp buffer：不生成 tmp_reduce 临时变量与 tmp_buf 访问
  EXPECT_EQ(result.find("tmp_reduce ="), std::string::npos);
  EXPECT_NE(result.find("CastExtend<int32_t, int64_t, 1>"), std::string::npos) << result;
}

TEST_F(RegArgmaxApicallTest, RegArgMaxPhase2WithValueApi_Test_001) {
  // ArgmaxWithValue R 分核 Phase2（WithValue 变体，2 入 2 出）：
  // value(FLOAT) + index(INT64) -> value(FLOAT) + index(INT64)，一次归约双输出
  af::AscGraph graph("argmax_phase2_with_value_graph");
  af::Expression s0;
  af::Expression s1;
  af::Axis z0;
  af::Axis z1;
  s0 = graph.CreateSizeVar("s0");
  s1 = graph.CreateSizeVar("s1");
  z0 = graph.CreateAxis("z0", s0);
  z1 = graph.CreateAxis("z1", s1);
  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);
  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1, af::ops::One};
  *load.y.repeats = {s0, s1};
  af::ascir_op::ArgMaxMultiRPhase2WithValue phase2_with_value("argmax_phase2_with_value");
  phase2_with_value.attr.sched.axis = {z0.id, z1.id};
  phase2_with_value.attr.api.compute_type = af::ComputeType::kComputeReduce;
  phase2_with_value.value_src = load.y;
  *phase2_with_value.value.axis = {z0.id, z1.id};
  phase2_with_value.value.dtype = ge::DT_FLOAT;
  *phase2_with_value.value.repeats = {s0, af::ops::One};
  *phase2_with_value.value.strides = {af::ops::One, af::ops::Zero};
  *phase2_with_value.index.axis = {z0.id, z1.id};
  phase2_with_value.index.dtype = ge::DT_INT64;
  *phase2_with_value.index.repeats = {s0, af::ops::One};
  *phase2_with_value.index.strides = {af::ops::One, af::ops::Zero};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  std::vector<ascir::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::TPipe tpipe("tpipe", tiler);
  auto phase2_node = graph.FindNode("argmax_phase2_with_value");

  // index 输入 source：独立 anchor（AscTensor.attr 是引用，复制 value_in 会共享属性）
  af::ascir_op::Data index_data("index_data", graph);
  index_data.y.dtype = ge::DT_INT64;

  af::AscTensor value_in = graph.FindNode("load")->outputs[0];
  af::AscTensor index_in = graph.FindNode("index_data")->outputs[0];
  af::AscTensor value_out = phase2_node->outputs[0];  // value
  af::AscTensor index_out = phase2_node->outputs[1];  // index
  // value 通路：value_in / value_out 均为 FLOAT（{N,A} -> {A}）
  FillReduceTensorAttrs(value_in, value_out, z0, z1, s0, s1, ge::DT_FLOAT, ge::DT_FLOAT);
  // index 通路：index_in / index_out 均为 INT64；覆盖独立 tensor_id/buf.id 避免与 value 通路冲突
  FillReduceTensorAttrs(index_in, index_out, z0, z1, s0, s1, ge::DT_INT32, ge::DT_INT64);
  index_in.attr.mem.tensor_id = 7;
  index_in.attr.buf.id = 8;
  index_out.attr.mem.tensor_id = 9;
  index_out.attr.buf.id = 10;

  std::string value_dtype_name;
  Tensor::DtypeName(value_in.attr.dtype, value_dtype_name);
  std::string index_dtype_name;
  Tensor::DtypeName(index_in.attr.dtype, index_dtype_name);

  std::vector<std::reference_wrapper<const Tensor>> inputs;
  std::vector<std::reference_wrapper<const Tensor>> outputs;
  Tensor value_in_tensor(value_in, value_dtype_name);
  Tensor index_in_tensor(index_in, index_dtype_name);
  Tensor value_out_tensor(value_out, value_dtype_name);
  Tensor index_out_tensor(index_out, index_dtype_name);
  inputs.push_back(std::ref(value_in_tensor));
  inputs.push_back(std::ref(index_in_tensor));
  outputs.push_back(std::ref(value_out_tensor));
  outputs.push_back(std::ref(index_out_tensor));

  codegen::ApiTensor value_in_api_tensor, index_in_api_tensor, value_out_api_tensor, index_out_api_tensor;
  value_in_api_tensor.id = value_in.attr.mem.tensor_id;
  index_in_api_tensor.id = index_in.attr.mem.tensor_id;
  value_out_api_tensor.id = value_out.attr.mem.tensor_id;
  value_out_api_tensor.reuse_id = value_out.attr.mem.reuse_id;
  index_out_api_tensor.id = index_out.attr.mem.tensor_id;
  index_out_api_tensor.reuse_id = index_out.attr.mem.reuse_id;

  codegen::RegArgMaxPhase2WithValueApiCall call("ArgMaxMultiRPhase2WithValue");
  call.unit = af::ComputeUnit::kUnitVector;
  call.type = "ArgMaxMultiRPhase2WithValue";
  call.node = phase2_node;
  call.node_name = phase2_node->GetName();
  value_out_api_tensor.write = &call;
  index_out_api_tensor.write = &call;
  call.inputs.push_back(&value_in_api_tensor);
  call.inputs.push_back(&index_in_api_tensor);
  call.outputs.push_back(value_out_api_tensor);
  call.outputs.push_back(index_out_api_tensor);
  call.tmp_buf_id[0] = 1;

  EXPECT_EQ(tpipe.AddTensor(value_in_tensor), 0);
  EXPECT_EQ(tpipe.AddTensor(index_in_tensor), 0);
  EXPECT_EQ(tpipe.AddTensor(value_out_tensor), 0);
  EXPECT_EQ(tpipe.AddTensor(index_out_tensor), 0);

  std::string result;
  Status status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(status, af::SUCCESS);
  // 向量入口（内部同宽走向量 / 混宽回退标量，行为等价）
  EXPECT_NE(result.find("ArgMaxMultiRPhase2WithValueVectorExtend<float>"), std::string::npos) << result;
  EXPECT_NE(result.find("tmp_reduce_shape"), std::string::npos) << result;
  // validA 传入 A 轴实际长度，Phase2 只处理真实列、不碰 padding
  EXPECT_NE(result.find("tmp_reduce_shape, z1_actual_size);"), std::string::npos) << result;
  // value(index) 与 value(float) dtype 不同：需额外按 index dtype 生成行跨度 last_tail，
  // 并以三元组 {first_actual, last, last_tail} 传入 Phase2（value/index 分别寻址）。
  EXPECT_NE(result.find("last_tail"), std::string::npos) << result;
  EXPECT_NE(result.find("{first_actual, last, last_tail}"), std::string::npos) << result;
  // 无 tmp buffer：不生成 tmp_reduce 临时变量与 tmp_buf 访问
  EXPECT_EQ(result.find("tmp_reduce ="), std::string::npos);
  EXPECT_NE(result.find("CastExtend<int32_t, int64_t, 1>"), std::string::npos) << result;
}

TEST_F(RegArgmaxApicallTest, RegArgMaxPhase2Api_Test_001) {
  // Argmax R 分核 Phase2（单输出，2 入 1 出）：value(FLOAT) + index(INT64) -> index(INT64)
  // 向量入口 ArgMaxMultiRPhase2VectorExtend<T,U>（内部混宽回退标量）。
  af::AscGraph graph("argmax_phase2_graph");
  af::Expression s0;
  af::Expression s1;
  af::Axis z0;
  af::Axis z1;
  s0 = graph.CreateSizeVar("s0");
  s1 = graph.CreateSizeVar("s1");
  z0 = graph.CreateAxis("z0", s0);
  z1 = graph.CreateAxis("z1", s1);
  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);
  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1, af::ops::One};
  *load.y.repeats = {s0, s1};
  af::ascir_op::ArgMaxMultiRPhase2 phase2("argmax_phase2");
  phase2.attr.sched.axis = {z0.id, z1.id};
  phase2.attr.api.compute_type = af::ComputeType::kComputeReduce;
  phase2.value = load.y;
  // index 为输入端口（AscOpInput，无输出属性成员），其布局由下方 index_in（index_data 输出）设置
  *phase2.y.axis = {z0.id, z1.id};
  phase2.y.dtype = ge::DT_INT64;
  *phase2.y.repeats = {s0, af::ops::One};
  *phase2.y.strides = {af::ops::One, af::ops::Zero};

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  std::vector<ascir::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::TPipe tpipe("tpipe", tiler);
  auto phase2_node = graph.FindNode("argmax_phase2");

  af::ascir_op::Data index_data("index_data", graph);
  index_data.y.dtype = ge::DT_INT64;

  af::AscTensor value_in = graph.FindNode("load")->outputs[0];
  af::AscTensor index_in = graph.FindNode("index_data")->outputs[0];
  af::AscTensor index_out = phase2_node->outputs[0];
  // value/index 均为输入（{N,A} 每核一行）；index_out 为归约输出（{A}）。
  // FillReduceTensorAttrs(x, y, ...) 将 x 设为输入布局、y 设为输出布局：
  //   - index 通路：index_in 输入布局 / index_out 输出布局（同 WithValue UT）；
  //   - value 输入：单独触发输入布局填充（输出占位仅用于满足 helper 的 y 参数，dtype 会被第二次调用重置）。
  af::AscTensor value_out_placeholder = graph.FindNode("argmax_phase2")->outputs[0];
  FillReduceTensorAttrs(value_in, value_out_placeholder, z0, z1, s0, s1, ge::DT_FLOAT, ge::DT_FLOAT);
  FillReduceTensorAttrs(index_in, index_out, z0, z1, s0, s1, ge::DT_INT32, ge::DT_INT64);
  // 重置 index_out 属性（value 占位填充可能污染其 dtype/布局）
  index_out.attr.axis = {z0.id, z1.id};
  index_out.attr.repeats = {s0, af::ops::One};
  index_out.attr.strides = {af::ops::One, af::ops::Zero};
  index_out.attr.dtype = ge::DT_INT64;
  index_in.attr.mem.tensor_id = 7;
  index_in.attr.buf.id = 8;
  index_out.attr.mem.tensor_id = 9;
  index_out.attr.buf.id = 10;

  std::string value_dtype_name;
  Tensor::DtypeName(value_in.attr.dtype, value_dtype_name);
  std::string index_dtype_name;
  Tensor::DtypeName(index_in.attr.dtype, index_dtype_name);

  std::vector<std::reference_wrapper<const Tensor>> inputs;
  std::vector<std::reference_wrapper<const Tensor>> outputs;
  Tensor value_in_tensor(value_in, value_dtype_name);
  Tensor index_in_tensor(index_in, index_dtype_name);
  Tensor index_out_tensor(index_out, index_dtype_name);
  inputs.push_back(std::ref(value_in_tensor));
  inputs.push_back(std::ref(index_in_tensor));
  outputs.push_back(std::ref(index_out_tensor));

  codegen::ApiTensor value_in_api_tensor, index_in_api_tensor, index_out_api_tensor;
  value_in_api_tensor.id = value_in.attr.mem.tensor_id;
  index_in_api_tensor.id = index_in.attr.mem.tensor_id;
  index_out_api_tensor.id = index_out.attr.mem.tensor_id;
  index_out_api_tensor.reuse_id = index_out.attr.mem.reuse_id;

  codegen::RegArgMaxPhase2ApiCall call("ArgMaxMultiRPhase2");
  call.unit = af::ComputeUnit::kUnitVector;
  call.type = "ArgMaxMultiRPhase2";
  call.node = phase2_node;
  call.node_name = phase2_node->GetName();
  index_out_api_tensor.write = &call;
  call.inputs.push_back(&value_in_api_tensor);
  call.inputs.push_back(&index_in_api_tensor);
  call.outputs.push_back(index_out_api_tensor);
  call.tmp_buf_id[0] = 1;

  EXPECT_EQ(tpipe.AddTensor(value_in_tensor), 0);
  EXPECT_EQ(tpipe.AddTensor(index_in_tensor), 0);
  EXPECT_EQ(tpipe.AddTensor(index_out_tensor), 0);

  std::string result;
  Status status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(status, af::SUCCESS);
  // 向量入口（内部混宽回退标量），模板参数 index/value dtype
  EXPECT_NE(result.find("ArgMaxMultiRPhase2VectorExtend<float>"), std::string::npos) << result;
  EXPECT_NE(result.find("tmp_reduce_shape"), std::string::npos) << result;
  EXPECT_NE(result.find("{first_actual, last, last_tail}"), std::string::npos) << result;
  // validA 传入 A 轴实际长度，Phase2 只处理真实列、不碰 padding
  EXPECT_NE(result.find("tmp_reduce_shape, z1_actual_size);"), std::string::npos) << result;
  // 无 tmp buffer
  EXPECT_EQ(result.find("tmp_reduce ="), std::string::npos);
  EXPECT_NE(result.find("CastExtend<int32_t, int64_t, 1>"), std::string::npos) << result;
}

TEST_F(RegArgmaxApicallTest, RegArgMaxPhase1Api_IndexBaseFromRAxis_Test) {
  // Argmax R 分核 Phase1（1 入 2 出）：x(FLOAT) -> value(FLOAT) + index(INT64)。
  // 回归点：核偏移必须取 R 轴自身的分块序号（R 轴 block-inner 循环轴的上级轴 z1T），
  // 而非全局核号 block_dim —— 后者是 A 轴与 R 轴合轴后的核序号，混入了 A 轴的核偏移。
  af::AscGraph graph("argmax_phase1_graph");
  af::Expression s0 = graph.CreateSizeVar("s0");
  af::Expression s1 = graph.CreateSizeVar("s1");
  af::Expression s1t = graph.CreateSizeVar("s1t");
  af::Axis z0 = graph.CreateAxis("z0", s0);
  af::Axis z1 = graph.CreateAxis("z1", s1);

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);
  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1, af::ops::One};
  *load.y.repeats = {s0, s1};

  af::ascir_op::ArgMaxMultiRPhase1 phase1("argmax_phase1");
  phase1.attr.sched.axis = {z0.id, z1.id};
  phase1.attr.api.compute_type = af::ComputeType::kComputeReduce;
  phase1.x = load.y;
  *phase1.value.axis = {z0.id, z1.id};
  phase1.value.dtype = ge::DT_FLOAT;
  *phase1.value.repeats = {s0, af::ops::One};
  *phase1.value.strides = {af::ops::One, af::ops::Zero};
  *phase1.index.axis = {z0.id, z1.id};
  phase1.index.dtype = ge::DT_INT64;
  *phase1.index.repeats = {s0, af::ops::One};
  *phase1.index.strides = {af::ops::One, af::ops::Zero};

  // R 轴分核切分：z1 -> (z1T tile-outer, z1t tile-inner)；z1T -> (z1TT block-outer, z1Ttb block-inner)。
  // 这里只构造 block-inner 循环轴 z1Ttb 与其上级轴 z1T，模拟 Phase1 调度后的轴结构。
  af::Axis z1T;
  z1T.id = 100;
  z1T.name = "z1T";
  z1T.type = ascir::Axis::Type::kAxisTypeTileOuter;
  z1T.from = {z1.id};
  z1T.size = s1t;
  af::Axis z1Ttb;
  z1Ttb.id = 101;
  z1Ttb.name = "z1Ttb";
  z1Ttb.type = ascir::Axis::Type::kAxisTypeBlockInner;
  z1Ttb.from = {z1T.id};
  z1Ttb.size = s1t;

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s1t));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z1T);
  tiler.AddAxis(z1Ttb);
  std::vector<ascir::AxisId> current_axis;
  current_axis.push_back(z0.id);
  current_axis.push_back(z1Ttb.id);  // 最内层 loop 轴 = R 轴 block-inner

  codegen::TPipe tpipe("tpipe", tiler);
  auto phase1_node = graph.FindNode("argmax_phase1");

  af::AscTensor x_in = graph.FindNode("load")->outputs[0];
  af::AscTensor value_out = phase1_node->outputs[0];
  af::AscTensor index_out = phase1_node->outputs[1];
  af::AscTensor value_placeholder = phase1_node->outputs[0];
  FillReduceTensorAttrs(x_in, value_placeholder, z0, z1, s0, s1, ge::DT_FLOAT, ge::DT_FLOAT);
  value_out.attr.axis = {z0.id, z1.id};
  value_out.attr.vectorized_axis = {z0.id, z1.id};
  value_out.attr.repeats = {s0, af::ops::One};
  value_out.attr.strides = {af::ops::One, af::ops::Zero};
  value_out.attr.vectorized_strides = {af::ops::One, af::ops::Zero};
  value_out.attr.dtype = ge::DT_FLOAT;
  index_out.attr.axis = {z0.id, z1.id};
  index_out.attr.vectorized_axis = {z0.id, z1.id};
  index_out.attr.repeats = {s0, af::ops::One};
  index_out.attr.strides = {af::ops::One, af::ops::Zero};
  index_out.attr.vectorized_strides = {af::ops::One, af::ops::Zero};
  index_out.attr.dtype = ge::DT_INT32;
  x_in.attr.mem.tensor_id = 1;
  value_out.attr.mem.tensor_id = 3;
  index_out.attr.mem.tensor_id = 5;
  value_out.attr.buf.id = 4;
  index_out.attr.buf.id = 6;

  std::string in_dtype_name;
  std::string value_dtype_name;
  std::string index_dtype_name;
  Tensor::DtypeName(x_in.attr.dtype, in_dtype_name);
  Tensor::DtypeName(value_out.attr.dtype, value_dtype_name);
  Tensor::DtypeName(index_out.attr.dtype, index_dtype_name);

  Tensor x_tensor(x_in, in_dtype_name);
  Tensor value_tensor(value_out, value_dtype_name);
  Tensor index_tensor(index_out, index_dtype_name);

  std::vector<std::reference_wrapper<const Tensor>> inputs{std::ref(x_tensor)};
  std::vector<std::reference_wrapper<const Tensor>> outputs{std::ref(value_tensor), std::ref(index_tensor)};

  codegen::ApiTensor x_api, value_api, index_api;
  x_api.id = x_in.attr.mem.tensor_id;
  value_api.id = value_out.attr.mem.tensor_id;
  index_api.id = index_out.attr.mem.tensor_id;

  codegen::RegArgMaxPhase1ApiCall call("ArgMaxMultiRPhase1");
  call.unit = af::ComputeUnit::kUnitVector;
  call.type = "ArgMaxMultiRPhase1";
  call.node = phase1_node;
  call.node_name = phase1_node->GetName();
  value_api.write = &call;
  call.inputs.push_back(&x_api);
  call.outputs.push_back(value_api);
  call.outputs.push_back(index_api);
  // Phase1 跨 tile 合并需要 index/value 两个 loop 级 tmp buffer（lifetime 0/1）
  call.tmp_buf_id[0] = 1;
  call.tmp_buf_id[1] = 2;

  EXPECT_EQ(tpipe.AddTensor(x_tensor), 0);
  EXPECT_EQ(tpipe.AddTensor(value_tensor), 0);
  EXPECT_EQ(tpipe.AddTensor(index_tensor), 0);

  std::string result;
  Status status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(status, af::SUCCESS) << result;
  EXPECT_NE(result.find("ArgMaxWithValueExtend<float, AscendC::Pattern::Reduce::AR>"), std::string::npos) << result;
  // 跨 tile 合并：首 tile 初始化输出累加器，后续 tile 与输出做 ArgmaxMergeRound
  EXPECT_NE(result.find("ArgmaxMergeRound<float>"), std::string::npos) << result;
  EXPECT_NE(result.find("tmp_argmax_index"), std::string::npos) << result;
  // 首 tile 拷贝必须用实际长度（不套 SizeAlign），避免依赖输出 padding
  EXPECT_NE(result.find("ArgmaxCopy(global_5[0], tmp_argmax_index[0], global_5_actual_size);"), std::string::npos)
      << result;
  EXPECT_EQ(result.find("DataCopyExtend(global_5[0], tmp_argmax_index[0], KernelUtils::SizeAlign"), std::string::npos)
      << result;
  // 核偏移取 R 轴分块序号 z1T（= block_dim_offset + 核内 tile 序号），而非全局核号 block_dim；
  // 步长必须用未对齐的 R tile size（t->s1），不能用对齐后的 last，否则非 8 对齐 tile 会漂移。
  EXPECT_NE(result.find("uint32_t argmax_index_base = z1T * t->s1;"), std::string::npos) << result;
  EXPECT_EQ(result.find("argmax_index_base = z1T * last;"), std::string::npos) << result;
  EXPECT_EQ(result.find("block_dim * "), std::string::npos) << result;
}

TEST_F(RegArgmaxApicallTest, RegArgMaxPhase1Api_RVectorizedAxisIsTileInner_Test) {
  // 回归：真实调度后 x.vectorized_axis 的最内 R 轴是 TileInner（z1t），不是原始轴。
  // TileInner 的 from[0] 是原始轴，配对的 TileOuter 在 split_pair_other_id 上；
  // 必须通过 split_pair_other_id 找到 z1T，才能定位 block-inner z1Ttb。
  af::AscGraph graph("argmax_phase1_tile_inner_axis");
  af::Expression s0 = graph.CreateSizeVar("s0");
  af::Expression s1 = graph.CreateSizeVar("s1");
  af::Expression s1t = graph.CreateSizeVar("s1t");
  af::Axis z0 = graph.CreateAxis("z0", s0);
  af::Axis z1 = graph.CreateAxis("z1", s1);

  Data data("data", graph);
  data.y.dtype = ge::DT_FLOAT;
  data.ir_attr.SetIndex(0);
  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id};
  load.y.dtype = ge::DT_FLOAT;
  *load.y.strides = {s1, af::ops::One};
  *load.y.repeats = {s0, s1};

  af::ascir_op::ArgMaxMultiRPhase1 phase1("argmax_phase1");
  phase1.attr.sched.axis = {z0.id, z1.id};
  phase1.attr.api.compute_type = af::ComputeType::kComputeReduce;
  phase1.x = load.y;
  *phase1.value.axis = {z0.id, z1.id};
  phase1.value.dtype = ge::DT_FLOAT;
  *phase1.value.repeats = {s0, af::ops::One};
  *phase1.value.strides = {af::ops::One, af::ops::Zero};
  *phase1.index.axis = {z0.id, z1.id};
  phase1.index.dtype = ge::DT_INT64;
  *phase1.index.repeats = {s0, af::ops::One};
  *phase1.index.strides = {af::ops::One, af::ops::Zero};

  // 真实调度结构：z1 -> (z1T TileOuter, z1t TileInner)；
  // z1T -> (z1TT BlockOuter, z1Ttb BlockInner)。TileInner 通过 split_pair 指向 TileOuter。
  af::Axis z1T;
  z1T.id = 100;
  z1T.name = "z1T";
  z1T.type = ascir::Axis::Type::kAxisTypeTileOuter;
  z1T.from = {z1.id};
  z1T.size = s1t;
  z1T.split_pair_other_id = 102;
  af::Axis z1t;
  z1t.id = 102;
  z1t.name = "z1t";
  z1t.type = ascir::Axis::Type::kAxisTypeTileInner;
  z1t.from = {z1.id};
  z1t.size = s1t;
  z1t.split_pair_other_id = z1T.id;
  af::Axis z1TT;
  z1TT.id = 101;
  z1TT.name = "z1TT";
  z1TT.type = ascir::Axis::Type::kAxisTypeBlockOuter;
  z1TT.from = {z1T.id};
  z1TT.size = s1t;
  z1TT.split_pair_other_id = 103;
  af::Axis z1Ttb;
  z1Ttb.id = 103;
  z1Ttb.name = "z1Ttb";
  z1Ttb.type = ascir::Axis::Type::kAxisTypeBlockInner;
  z1Ttb.from = {z1T.id};
  z1Ttb.size = s1t;
  z1Ttb.split_pair_other_id = z1TT.id;

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s1t));
  tiler.AddAxis(z0);
  tiler.AddAxis(z1);
  tiler.AddAxis(z1T);
  tiler.AddAxis(z1t);
  tiler.AddAxis(z1TT);
  tiler.AddAxis(z1Ttb);
  std::vector<ascir::AxisId> current_axis;
  current_axis.push_back(z0.id);
  current_axis.push_back(z1Ttb.id);

  codegen::TPipe tpipe("tpipe", tiler);
  auto phase1_node = graph.FindNode("argmax_phase1");

  af::AscTensor x_in = graph.FindNode("load")->outputs[0];
  af::AscTensor value_out = phase1_node->outputs[0];
  af::AscTensor index_out = phase1_node->outputs[1];
  af::AscTensor value_placeholder = phase1_node->outputs[0];
  FillReduceTensorAttrs(x_in, value_placeholder, z0, z1, s0, s1, ge::DT_FLOAT, ge::DT_FLOAT);
  // 模拟真实调度：x 的 vectorized_axis 是 TileInner（z1t），而不是原始轴 z1。
  x_in.attr.vectorized_axis = {z0.id, z1t.id};
  x_in.attr.vectorized_strides = {af::ops::One, af::ops::One};
  value_out.attr.axis = {z0.id, z1.id};
  value_out.attr.vectorized_axis = {z0.id, z1.id};
  value_out.attr.repeats = {s0, af::ops::One};
  value_out.attr.strides = {af::ops::One, af::ops::Zero};
  value_out.attr.vectorized_strides = {af::ops::One, af::ops::Zero};
  value_out.attr.dtype = ge::DT_FLOAT;
  index_out.attr.axis = {z0.id, z1.id};
  index_out.attr.vectorized_axis = {z0.id, z1.id};
  index_out.attr.repeats = {s0, af::ops::One};
  index_out.attr.strides = {af::ops::One, af::ops::Zero};
  index_out.attr.vectorized_strides = {af::ops::One, af::ops::Zero};
  index_out.attr.dtype = ge::DT_INT32;
  x_in.attr.mem.tensor_id = 1;
  value_out.attr.mem.tensor_id = 3;
  index_out.attr.mem.tensor_id = 5;
  value_out.attr.buf.id = 4;
  index_out.attr.buf.id = 6;

  std::string in_dtype_name;
  std::string value_dtype_name;
  std::string index_dtype_name;
  Tensor::DtypeName(x_in.attr.dtype, in_dtype_name);
  Tensor::DtypeName(value_out.attr.dtype, value_dtype_name);
  Tensor::DtypeName(index_out.attr.dtype, index_dtype_name);

  Tensor x_tensor(x_in, in_dtype_name);
  Tensor value_tensor(value_out, value_dtype_name);
  Tensor index_tensor(index_out, index_dtype_name);

  std::vector<std::reference_wrapper<const Tensor>> inputs{std::ref(x_tensor)};
  std::vector<std::reference_wrapper<const Tensor>> outputs{std::ref(value_tensor), std::ref(index_tensor)};

  codegen::ApiTensor x_api, value_api, index_api;
  x_api.id = x_in.attr.mem.tensor_id;
  value_api.id = value_out.attr.mem.tensor_id;
  index_api.id = index_out.attr.mem.tensor_id;

  codegen::RegArgMaxPhase1ApiCall call("ArgMaxMultiRPhase1");
  call.unit = af::ComputeUnit::kUnitVector;
  call.type = "ArgMaxMultiRPhase1";
  call.node = phase1_node;
  call.node_name = phase1_node->GetName();
  value_api.write = &call;
  call.inputs.push_back(&x_api);
  call.outputs.push_back(value_api);
  call.outputs.push_back(index_api);
  call.tmp_buf_id[0] = 1;
  call.tmp_buf_id[1] = 2;

  EXPECT_EQ(tpipe.AddTensor(x_tensor), 0);
  EXPECT_EQ(tpipe.AddTensor(value_tensor), 0);
  EXPECT_EQ(tpipe.AddTensor(index_tensor), 0);

  std::string result;
  Status status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(status, af::SUCCESS) << result;
  EXPECT_NE(result.find("uint32_t argmax_index_base = z1T * t->s1t;"), std::string::npos) << result;
}
