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
#include <tuple>
#include <memory>
#include <utility>
#include "ascir_ops.h"
#include "codegen_kernel.h"
#include "ascendc_api_registry.h"
#include "utils/api_call_factory.h"
#include "../reg_api_call/reg_arg_reduce_api_call.h"

#include "runtime_stub.h"
#include "platform_context.h"
#include "common_utils.h"
#include "ascir_node_param/ascir_node_param.h"
#include "ascir_node_param/ascir_param_builder.h"
#include "parser/specific_params_builder.h"

using namespace af::ops;
using namespace codegen;
using namespace af::ascir_op;
using namespace testing;

struct SplitTileChain;

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
  template <typename Op = af::ascir_op::Argmax>
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
    Op argmax("argmax");
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

  static void AddBasicAxes(codegen::Tiler &tiler, const af::Expression &s0, const af::Expression &s1,
                           const af::Axis &z0, const af::Axis &z1) {
    tiler.AddSizeVar(af::SizeVar(s0));
    tiler.AddSizeVar(af::SizeVar(s1));
    tiler.AddAxis(z0);
    tiler.AddAxis(z1);
  }

  static void AddAxesToTiler(codegen::Tiler &tiler, std::initializer_list<af::Axis> axes) {
    for (const auto &axis : axes) {
      ASSERT_EQ(tiler.AddAxis(axis), af::SUCCESS);
    }
  }

  static Tensor MakeArgmaxTensor(const af::AscTensor &tensor, ge::DataType name_dtype = ge::DT_UNDEFINED) {
    std::string dtype_name;
    EXPECT_EQ(Tensor::DtypeName(name_dtype == ge::DT_UNDEFINED ? tensor.attr.dtype : name_dtype, dtype_name),
              af::SUCCESS);
    return Tensor(tensor, dtype_name);
  }

  static void InitArgmaxCall(ApiCall &call, const af::AscNodePtr &node) {
    call.unit = af::ComputeUnit::kUnitVector;
    call.type = node->GetType();
    call.node = node;
    call.node_name = node->GetName();
  }

  static void CreateBasicAxes(af::AscGraph &graph, af::Expression &s0, af::Expression &s1, af::Axis &z0, af::Axis &z1) {
    s0 = graph.CreateSizeVar("s0");
    s1 = graph.CreateSizeVar("s1");
    z0 = graph.CreateAxis("z0", s0);
    z1 = graph.CreateAxis("z1", s1);
  }

  static void BindArgmaxInput(ApiCall &call, ApiTensor &api, const Tensor &tensor, TPipe &pipe) {
    api.id = tensor.id;
    call.inputs.push_back(&api);
    EXPECT_EQ(pipe.AddTensor(tensor), af::SUCCESS);
  }

  static void BindArgmaxOutput(ApiCall &call, ApiTensor &api, const Tensor &tensor, TPipe &pipe,
                               bool mark_write = true) {
    api.id = tensor.id;
    api.reuse_id = tensor.reuse_id;
    if (mark_write) {
      api.write = &call;
    }
    call.outputs.push_back(api);
    EXPECT_EQ(pipe.AddTensor(tensor), af::SUCCESS);
  }

  static af::AscTensor MakePhase2IndexInput(af::AscGraph &graph) {
    Data index_data("index_data", graph);
    index_data.y.dtype = ge::DT_INT64;
    return graph.FindNode("index_data")->outputs[0];
  }

  static void FillPhase2IndexAttrs(af::AscTensor &input, af::AscTensor &output, const af::Axis &z0, const af::Axis &z1,
                                   const af::Expression &s0, const af::Expression &s1) {
    FillReduceTensorAttrs(input, output, z0, z1, s0, s1, ge::DT_INT32, ge::DT_INT64);
    input.attr.mem.tensor_id = 7;
    input.attr.buf.id = 8;
    output.attr.mem.tensor_id = 9;
    output.attr.buf.id = 10;
  }

  static void VerifyPhase2WithValueParams(const af::AscNodePtr &node, const af::Expression &s0,
                                          const af::Expression &s1, att::NodeInfo &info) {
    info.node_type = "ArgMaxMultiRPhase2WithValue";
    ASSERT_EQ(att::FillSpecificParams(node, info), af::SUCCESS);
    EXPECT_TRUE(info.argmax_node_params.valid);
    EXPECT_TRUE(info.argmax_node_params.is_phase2);
    EXPECT_TRUE(info.argmax_node_params.with_value);
    ASSERT_EQ(info.argmax_node_params.src_shape.size(), 3U);
    EXPECT_EQ(info.argmax_node_params.src_shape[0], s0);
    EXPECT_EQ(info.argmax_node_params.valid_extent, s1);
  }

  static void VerifyPhase2MixedPitch(ApiCall &call, TPipe &pipe, const std::vector<ascir::AxisId> &current_axis,
                                     const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                     const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                     Tensor &value_input, const af::Expression &s1, att::NodeInfo &info) {
    value_input.dtype = ge::DT_INT64;
    ASSERT_EQ(call.BuildApiParam(pipe, current_axis, inputs, outputs), af::SUCCESS);
    ASSERT_EQ(att::FillSpecificParams(call.node, info), af::SUCCESS);
    EXPECT_EQ(info.argmax_node_params.src_shape[1], af::sym::Ceiling(s1 / af::Symbol(4)) * af::Symbol(4));
    EXPECT_EQ(info.argmax_node_params.src_shape[2], af::sym::Ceiling(s1 / af::Symbol(8)) * af::Symbol(8));
  }

  static void FillPhase1IndexAttrs(af::AscTensor &index, const af::Axis &z0, const af::Axis &z1,
                                   const af::Expression &s0) {
    index.attr.axis = {z0.id, z1.id};
    index.attr.vectorized_axis = {z0.id, z1.id};
    index.attr.repeats = {s0, One};
    index.attr.strides = {One, Zero};
    index.attr.vectorized_strides = {One, Zero};
    index.attr.dtype = ge::DT_INT32;
    index.attr.mem.tensor_id = 5;
    index.attr.buf.id = 6;
  }

  static void VerifyPhase1ArResult(const af::AscNodePtr &node, const std::string &result) {
    EXPECT_NE(result.find("ArgMaxWithValueAccumulateExtend<float, AscendC::Pattern::Reduce::AR>(global_5[0], "
                          "local_3[0], local_1["),
              std::string::npos)
        << result;
    EXPECT_NE(result.find("tmp_arg_reduce_index[0], tmp_arg_reduce_value[0], tmp_reduce_shape, arg_reduce_index_base, "
                          "z1_actual_size, arg_reduce_first_tile);"),
              std::string::npos)
        << result;
    EXPECT_NE(result.find("bool arg_reduce_first_tile = (z1Ttb == 0);"), std::string::npos) << result;
    EXPECT_EQ(result.find("ArgReduceMergeRound"), std::string::npos) << result;
    att::NodeInfo info;
    info.node_type = "ArgMaxMultiRPhase1";
    ASSERT_EQ(att::FillSpecificParams(node, info), af::SUCCESS);
    EXPECT_TRUE(info.argmax_node_params.valid);
    EXPECT_TRUE(info.argmax_node_params.with_value);
    EXPECT_TRUE(info.argmax_node_params.has_tile_merge);
    EXPECT_TRUE(info.argmax_node_params.has_accumulate);
    EXPECT_NE(result.find("tmp_arg_reduce_index"), std::string::npos) << result;
    EXPECT_EQ(result.find("ArgReduceCopy("), std::string::npos) << result;
    const auto api = CodegenApiParam::TryGetNodeApiParam(node);
    ASSERT_NE(api, nullptr);
    EXPECT_EQ(api->semantic_feature, "AR/MultiRound");
    EXPECT_EQ(result.find("DataCopyExtend(global_5[0], tmp_arg_reduce_index[0], KernelUtils::SizeAlign"),
              std::string::npos)
        << result;
    EXPECT_NE(result.find("uint32_t arg_reduce_index_base = (int64_t)z1T * (int64_t)t->s1;"), std::string::npos)
        << result;
    EXPECT_EQ(result.find("arg_reduce_index_base = z1T * last;"), std::string::npos) << result;
    EXPECT_EQ(result.find("block_dim * "), std::string::npos) << result;
  }

  template <typename Op = af::ascir_op::ArgmaxWithValue>
  static void BuildArgmaxWithValueGraph(af::AscGraph &graph, ge::DataType in_dtype, af::Expression &s0,
                                        af::Expression &s1, af::Axis &z0, af::Axis &z1) {
    Data data("data", graph);
    data.y.dtype = in_dtype;
    data.ir_attr.SetIndex(0);
    Load load("load");
    load.attr.sched.axis = {z0.id, z1.id};
    load.x = data.y;
    *load.y.axis = {z0.id, z1.id};
    load.y.dtype = in_dtype;
    *load.y.strides = {s1, af::ops::One};
    *load.y.repeats = {s0, s1};
    Op argmax_with_value("argmax_with_value");
    argmax_with_value.attr.sched.axis = {z0.id, z1.id};
    argmax_with_value.attr.api.compute_type = af::ComputeType::kComputeReduce;
    argmax_with_value.x = load.y;
    *argmax_with_value.value.axis = {z0.id, z1.id};
    argmax_with_value.value.dtype = in_dtype;
    *argmax_with_value.value.repeats = {s0, af::ops::One};
    *argmax_with_value.value.strides = {af::ops::One, af::ops::Zero};
    *argmax_with_value.index.axis = {z0.id, z1.id};
    argmax_with_value.index.dtype = ge::DT_INT64;
    *argmax_with_value.index.repeats = {s0, af::ops::One};
    *argmax_with_value.index.strides = {af::ops::One, af::ops::Zero};
  }

  static void BuildPhase2WithValueGraph(af::AscGraph &graph, af::Expression &s0, af::Expression &s1, af::Axis &z0,
                                        af::Axis &z1) {
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
  }

  static void BuildPhase2Graph(af::AscGraph &graph, af::Expression &s0, af::Expression &s1, af::Axis &z0,
                               af::Axis &z1) {
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
  }

  static void BuildPhase1Graph(af::AscGraph &graph, af::Expression &s0, af::Expression &s1, af::Axis &z0,
                               af::Axis &z1) {
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
  }

  static void VerifyEmptyPreregisteredParams(const af::AscNodePtr &argmax_node) {
    const auto preregistered = ascir_param::GetAscirNodeParams(argmax_node);
    ASSERT_NE(preregistered, nullptr);
    const auto *empty_params = ascir_param::GetSpecificParams<ascir_param::ArgmaxNodeParams>(*preregistered);
    ASSERT_NE(empty_params, nullptr);
    EXPECT_FALSE(empty_params->valid);
    EXPECT_FALSE(empty_params->has_accumulate);
  }

  static void VerifyPhase1RaCase(af::AscGraph &graph, ApiCall &call, Tiler &tiler, const SplitTileChain &chain,
                                 const af::Axis &z0);

  template <typename Op, typename Call>
  static void CheckValueFrontend(ge::DataType dtype, const char *api_name, const char *entry, const char *cpp_type) {
    af::AscGraph graph(api_name);
    af::Expression s0, s1;
    af::Axis z0, z1;
    CreateBasicAxes(graph, s0, s1, z0, z1);
    BuildArgmaxWithValueGraph<Op>(graph, dtype, s0, s1, z0, z1);
    Tiler tiler;
    AddBasicAxes(tiler, s0, s1, z0, z1);
    TPipe pipe("pipe", tiler);
    auto node = graph.FindNode("argmax_with_value");
    auto x = graph.FindNode("load")->outputs[0];
    auto value = node->outputs[0];
    auto index = node->outputs[1];
    FillReduceTensorAttrs(x, value, z0, z1, s0, s1, dtype, dtype);
    index.attr.mem.tensor_id = 5;
    index.attr.mem.position = af::Position::kPositionVecOut;
    index.attr.buf.id = 6;
    index.attr.dtype = ge::DT_INT64;
    auto tx = MakeArgmaxTensor(x), tv = MakeArgmaxTensor(value), ti = MakeArgmaxTensor(index);
    ApiTensor ix, ov, oi;
    Call call(api_name);
    InitArgmaxCall(call, node);
    BindArgmaxInput(call, ix, tx, pipe);
    BindArgmaxOutput(call, ov, tv, pipe);
    BindArgmaxOutput(call, oi, ti, pipe);
    call.tmp_buf_id[0] = 1;
    ASSERT_EQ(ascir_param::EnrichAscirGraphNodeParams(graph), af::SUCCESS);
    ASSERT_NO_FATAL_FAILURE(VerifyEmptyPreregisteredParams(node));
    std::string result;
    ASSERT_EQ(call.Generate(pipe, {z0.id}, result), af::SUCCESS);
    EXPECT_NE(result.find(std::string(entry) + "<" + cpp_type + ", AscendC::Pattern::Reduce::AR>"), std::string::npos);
    EXPECT_NE(result.find("(tmp_arg_reduce_index[0], local_3["), std::string::npos) << result;
    EXPECT_NE(result.find("CastExtend<int32_t, int64_t, 1>"), std::string::npos);
    const auto params = ascir_param::GetAscirNodeParams(node);
    ASSERT_NE(params, nullptr);
    EXPECT_EQ(params->api_name, api_name);
    const auto *specific = ascir_param::GetSpecificParams<ascir_param::ArgReduceNodeParams>(*params);
    ASSERT_NE(specific, nullptr);
    EXPECT_TRUE(specific->with_value);
    const auto api = CodegenApiParam::TryGetNodeApiParam(node);
    ASSERT_NE(api, nullptr);
    EXPECT_EQ(api->semantic_feature, "AR");
  }

  static void VerifyBasicArgmaxParams(const af::AscNodePtr &argmax_node, const af::Expression &s0,
                                      const af::Expression &s1) {
    att::NodeInfo info;
    info.node_type = argmax_node->GetType();
    ASSERT_EQ(att::FillSpecificParams(argmax_node, info), af::SUCCESS);
    const auto &params = info.argmax_node_params;
    ASSERT_TRUE(params.valid);
    EXPECT_TRUE(params.is_ar);
    EXPECT_FALSE(params.is_phase2);
    EXPECT_FALSE(params.has_tile_merge);
    EXPECT_TRUE(params.has_index_cast);
    ASSERT_EQ(params.src_shape.size(), 2U);
    EXPECT_EQ(params.src_shape[0], s0);
    EXPECT_EQ(params.src_shape[1], af::sym::Ceiling(s1 / af::Symbol(8)) * af::Symbol(8));
    EXPECT_EQ(params.valid_extent, s1);
  }
};

struct SplitTileChain {
  af::Axis tile_outer;
  af::Axis tile_inner;
  af::Axis block_outer;
  af::Axis block_inner;
};

SplitTileChain BuildSplitTileChain(const af::Axis &source, af::Expression size) {
  SplitTileChain chain;
  chain.tile_outer.id = 100;
  chain.tile_outer.name = "z1T";
  chain.tile_outer.type = ascir::Axis::Type::kAxisTypeTileOuter;
  chain.tile_outer.from = {source.id};
  chain.tile_outer.size = size;
  chain.block_inner.id = 101;
  chain.block_inner.name = "z1Ttb";
  chain.block_inner.type = ascir::Axis::Type::kAxisTypeBlockInner;
  chain.block_inner.from = {chain.tile_outer.id};
  chain.block_inner.size = size;
  return chain;
}

SplitTileChain BuildFullSplitTileChain(const af::Axis &source, af::Expression size) {
  SplitTileChain chain;
  chain.tile_outer.id = 100;
  chain.tile_outer.name = "z1T";
  chain.tile_outer.type = ascir::Axis::Type::kAxisTypeTileOuter;
  chain.tile_outer.from = {source.id};
  chain.tile_outer.size = size;
  chain.tile_outer.split_pair_other_id = 102;
  chain.tile_inner.id = 102;
  chain.tile_inner.name = "z1t";
  chain.tile_inner.type = ascir::Axis::Type::kAxisTypeTileInner;
  chain.tile_inner.from = {source.id};
  chain.tile_inner.size = size;
  chain.tile_inner.split_pair_other_id = chain.tile_outer.id;
  chain.block_outer.id = 101;
  chain.block_outer.name = "z1TT";
  chain.block_outer.type = ascir::Axis::Type::kAxisTypeBlockOuter;
  chain.block_outer.from = {chain.tile_outer.id};
  chain.block_outer.size = size;
  chain.block_outer.split_pair_other_id = 103;
  chain.block_inner.id = 103;
  chain.block_inner.name = "z1Ttb";
  chain.block_inner.type = ascir::Axis::Type::kAxisTypeBlockInner;
  chain.block_inner.from = {chain.tile_outer.id};
  chain.block_inner.size = size;
  chain.block_inner.split_pair_other_id = chain.block_outer.id;
  return chain;
}

void RegArgmaxApicallTest::VerifyPhase1RaCase(af::AscGraph &graph, ApiCall &call, Tiler &tiler,
                                              const SplitTileChain &chain, const af::Axis &z0) {
  auto node = graph.FindNode("argmax_phase1");
  auto x = graph.FindNode("load")->outputs[0];
  auto value = node->outputs[0];
  auto index = node->outputs[1];
  x.attr.vectorized_axis = {chain.tile_inner.id, z0.id};
  x.attr.vectorized_strides = {z0.size, One};
  value.attr.vectorized_axis = index.attr.vectorized_axis = {chain.tile_inner.id, z0.id};
  value.attr.vectorized_strides = index.attr.vectorized_strides = {Zero, One};
  auto ra_x = MakeArgmaxTensor(x);
  auto ra_value = MakeArgmaxTensor(value);
  auto ra_index = MakeArgmaxTensor(index);
  ASSERT_EQ(ascir_param::EnrichAscirGraphNodeParams(graph), af::SUCCESS);
  ASSERT_NO_FATAL_FAILURE(VerifyEmptyPreregisteredParams(node));
  TPipe pipe("tpipe", tiler);
  ASSERT_EQ(pipe.AddTensor(ra_x), af::SUCCESS);
  ASSERT_EQ(pipe.AddTensor(ra_value), af::SUCCESS);
  ASSERT_EQ(pipe.AddTensor(ra_index), af::SUCCESS);
  std::string result;
  ASSERT_EQ(call.Generate(pipe, {chain.block_inner.id, z0.id}, result), af::SUCCESS);
  EXPECT_NE(result.find("ArgMaxWithValueAccumulateExtend<float, AscendC::Pattern::Reduce::RA>(global_5[0], "
                        "local_3[0], local_1["),
            std::string::npos)
      << result;
  EXPECT_NE(result.find("tmp_arg_reduce_index[0], tmp_arg_reduce_value[0], tmp_reduce_shape, arg_reduce_index_base, "
                        "first_actual, arg_reduce_first_tile);"),
            std::string::npos)
      << result;
  EXPECT_NE(result.find("bool arg_reduce_first_tile = (z1Ttb == 0);"), std::string::npos) << result;
  EXPECT_EQ(result.find("ArgReduceCopy("), std::string::npos) << result;
  EXPECT_EQ(result.find("ArgReduceMergeRound"), std::string::npos) << result;
  att::NodeInfo info;
  info.node_type = "ArgMaxMultiRPhase1";
  ASSERT_EQ(att::FillSpecificParams(node, info), af::SUCCESS);
  const auto &params = info.argmax_node_params;
  EXPECT_TRUE(params.valid);
  EXPECT_TRUE(params.has_accumulate);
  EXPECT_FALSE(params.is_ar);
  ASSERT_EQ(params.src_shape.size(), 2U);
  EXPECT_EQ(params.src_shape[0], chain.tile_inner.size);
  EXPECT_EQ(params.valid_extent, chain.tile_inner.size);
  const auto api = CodegenApiParam::TryGetNodeApiParam(node);
  ASSERT_NE(api, nullptr);
  EXPECT_EQ(api->api_name, "ArgMaxWithValueAccumulateExtend");
  EXPECT_EQ(api->semantic_feature, "RA/MultiRound");
}

TEST_F(RegArgmaxApicallTest, ArgminRegistryKeepsAllFiveApiIdentitiesAndDependencyOrder) {
  ge::PlatformInfo platform;
  platform.soc_ver = "5102";
  platform.aiv_num = 48;
  platform.ub_size = 245760;
  ASSERT_EQ(ge::PlatformContext::GetInstance().SetPlatformTarget(platform, platform.soc_ver), af::SUCCESS);
  const std::pair<const char *, const char *> cases[] = {
      {"Argmin", "RegArgminApiCall"},
      {"ArgminWithValue", "RegArgminWithValueApiCall"},
      {"ArgMinMultiRPhase1", "RegArgMinPhase1ApiCall"},
      {"ArgMinMultiRPhase2", "RegArgMinPhase2ApiCall"},
      {"ArgMinMultiRPhase2WithValue", "RegArgMinPhase2WithValueApiCall"}};
  const std::vector<std::string> headers{
      "arg_reduce_common_reg_base.h", "arg_reduce_core_reg_base.h", "arg_reduce_merge_reg_base.h",
      "arg_reduce_reg_base.h",        "argmin_reg_base.h",          "cast_reg_base.h"};
  for (const auto &entry : cases) {
    SCOPED_TRACE(entry.first);
    const auto impl = ascgen_utils::GetAscIrCodegenImpl(entry.first);
    ASSERT_NE(impl, nullptr);
    EXPECT_EQ(impl->GetApiName(), entry.first);
    EXPECT_EQ(impl->GetApiCallName(), entry.second);
    EXPECT_EQ(impl->LoadApiHeaderFiles(false), headers);
    EXPECT_EQ(impl->LoadApiHeaderFiles(true), headers);
    std::unique_ptr<ApiCall> call(ApiCallFactory::Instance().Create(entry.second, entry.first));
    ASSERT_NE(call, nullptr);
    for (const auto &header : headers) {
      EXPECT_FALSE(AscendCApiRegistry::GetInstance().GetFileContent(header).empty()) << header;
    }
  }
}

TEST_F(RegArgmaxApicallTest, ArgminFrontendGeneratesMinAndPreservesIndexConversion) {
  af::AscGraph graph("argmin_frontend");
  af::Expression s0, s1;
  af::Axis z0, z1;
  BuildArgmaxGraph<af::ascir_op::Argmin>(graph, s0, s1, z0, z1);
  Tiler tiler;
  AddBasicAxes(tiler, s0, s1, z0, z1);
  TPipe pipe("pipe", tiler);
  const auto node = graph.FindNode("argmax");
  auto x = graph.FindNode("load")->outputs[0];
  auto y = node->outputs[0];
  FillReduceTensorAttrs(x, y, z0, z1, s0, s1, ge::DT_FLOAT, ge::DT_INT64);
  auto tx = MakeArgmaxTensor(x), ty = MakeArgmaxTensor(y);
  RegArgminApiCall call("Argmin");
  InitArgmaxCall(call, node);
  ApiTensor input, output;
  BindArgmaxInput(call, input, tx, pipe);
  BindArgmaxOutput(call, output, ty, pipe);
  call.tmp_buf_id[0] = 1;
  ASSERT_EQ(ascir_param::EnrichAscirGraphNodeParams(graph), af::SUCCESS);
  ASSERT_NO_FATAL_FAILURE(VerifyEmptyPreregisteredParams(node));
  std::string result;
  ASSERT_EQ(call.Generate(pipe, {z0.id}, result), af::SUCCESS);
  EXPECT_NE(result.find("ArgMinExtend<float, AscendC::Pattern::Reduce::AR>"), std::string::npos) << result;
  EXPECT_EQ(result.find("ArgMaxExtend<"), std::string::npos);
  EXPECT_NE(result.find("CastExtend<int32_t, int64_t, 1>"), std::string::npos);
  ASSERT_NO_FATAL_FAILURE(VerifyBasicArgmaxParams(node, s0, s1));
}

TEST_F(RegArgmaxApicallTest, SharedWithValueFrontendKeepsDirectionDtypeAndOutputOrder) {
  const std::pair<ge::DataType, const char *> types[] = {
      {ge::DT_FLOAT, "float"}, {ge::DT_INT32, "int32_t"}, {ge::DT_INT64, "int64_t"}};
  for (const auto &type : types) {
    SCOPED_TRACE(type.second);
    ASSERT_NO_FATAL_FAILURE((CheckValueFrontend<af::ascir_op::ArgminWithValue, RegArgminWithValueApiCall>(
        type.first, "ArgminWithValue", "ArgMinWithValueExtend", type.second)));
    ASSERT_NO_FATAL_FAILURE((CheckValueFrontend<af::ascir_op::ArgmaxWithValue, RegArgmaxWithValueApiCall>(
        type.first, "ArgmaxWithValue", "ArgMaxWithValueExtend", type.second)));
  }
}

TEST_F(RegArgmaxApicallTest, RegArgmaxApi_Test_001) {
  af::AscGraph graph("argmax_graph");
  af::Expression s0;
  af::Expression s1;
  af::Axis z0;
  af::Axis z1;
  BuildArgmaxGraph(graph, s0, s1, z0, z1);

  codegen::Tiler tiler;
  AddBasicAxes(tiler, s0, s1, z0, z1);
  std::vector<ascir::AxisId> current_axis;
  current_axis.push_back(z0.id);

  codegen::TPipe tpipe("tpipe", tiler);
  auto argmax_node = graph.FindNode("argmax");

  af::AscTensor tensorx = graph.FindNode("load")->outputs[0];
  af::AscTensor tensory = argmax_node->outputs[0];
  FillReduceTensorAttrs(tensorx, tensory, z0, z1, s0, s1, ge::DT_FLOAT, ge::DT_INT64);
  ASSERT_EQ(ascir_param::EnrichAscirGraphNodeParams(graph), af::SUCCESS);
  VerifyEmptyPreregisteredParams(argmax_node);

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
  VerifyBasicArgmaxParams(argmax_node, s0, s1);
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
  CreateBasicAxes(graph, s0, s1, z0, z1);
  BuildArgmaxWithValueGraph(graph, ge::DT_FLOAT, s0, s1, z0, z1);
  codegen::Tiler tiler;
  AddBasicAxes(tiler, s0, s1, z0, z1);
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

  std::vector<std::reference_wrapper<const Tensor>> inputs;
  std::vector<std::reference_wrapper<const Tensor>> outputs;
  auto input_tensor = MakeArgmaxTensor(tensorx);
  auto value_tensor = MakeArgmaxTensor(tensorv);
  auto index_tensor = MakeArgmaxTensor(tensori);
  inputs.push_back(std::ref(input_tensor));
  outputs.push_back(std::ref(value_tensor));
  outputs.push_back(std::ref(index_tensor));

  codegen::ApiTensor x_tensor, value_api_tensor, index_api_tensor;
  codegen::RegArgmaxWithValueApiCall call("ArgmaxWithValue");
  InitArgmaxCall(call, argmax_node);
  BindArgmaxInput(call, x_tensor, input_tensor, tpipe);
  BindArgmaxOutput(call, value_api_tensor, value_tensor, tpipe);
  BindArgmaxOutput(call, index_api_tensor, index_tensor, tpipe);
  call.tmp_buf_id[0] = 1;

  std::string result;
  Status status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(status, af::SUCCESS);
  EXPECT_NE(result.find("ArgMaxWithValueExtend<float, AscendC::Pattern::Reduce::AR>"), std::string::npos) << result;
  EXPECT_NE(result.find("tmp_reduce_shape"), std::string::npos) << result;
  // 无 tmp buffer：不生成 tmp_reduce 临时变量与 tmp_buf 访问
  EXPECT_EQ(result.find("tmp_reduce ="), std::string::npos);
  EXPECT_NE(result.find("CastExtend<int32_t, int64_t, 1>"), std::string::npos) << result;
  const auto params = ascir_param::GetAscirNodeParams(argmax_node);
  ASSERT_NE(params, nullptr);
  const auto *argmax = ascir_param::GetSpecificParams<ascir_param::ArgmaxNodeParams>(*params);
  ASSERT_NE(argmax, nullptr);
  EXPECT_TRUE(argmax->with_value);
}

TEST_F(RegArgmaxApicallTest, RegArgmaxWithValueApi_Test_002) {
  // 前端算子 ArgmaxWithValue（1 入 2 出）：int64 输入 + int64 索引（FR13）
  af::AscGraph graph("argmax_with_value_int64_graph");
  af::Expression s0;
  af::Expression s1;
  af::Axis z0;
  af::Axis z1;
  CreateBasicAxes(graph, s0, s1, z0, z1);
  BuildArgmaxWithValueGraph(graph, ge::DT_INT64, s0, s1, z0, z1);
  codegen::Tiler tiler;
  AddBasicAxes(tiler, s0, s1, z0, z1);
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

  std::vector<std::reference_wrapper<const Tensor>> inputs;
  std::vector<std::reference_wrapper<const Tensor>> outputs;
  auto input_tensor = MakeArgmaxTensor(tensorx);
  auto value_tensor = MakeArgmaxTensor(tensorv);
  auto index_tensor = MakeArgmaxTensor(tensori);
  inputs.push_back(std::ref(input_tensor));
  outputs.push_back(std::ref(value_tensor));
  outputs.push_back(std::ref(index_tensor));

  codegen::ApiTensor x_tensor, value_api_tensor, index_api_tensor;
  codegen::RegArgmaxWithValueApiCall call("ArgmaxWithValue");
  InitArgmaxCall(call, argmax_node);
  BindArgmaxInput(call, x_tensor, input_tensor, tpipe);
  BindArgmaxOutput(call, value_api_tensor, value_tensor, tpipe);
  BindArgmaxOutput(call, index_api_tensor, index_tensor, tpipe);
  call.tmp_buf_id[0] = 1;

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
  CreateBasicAxes(graph, s0, s1, z0, z1);
  BuildPhase2WithValueGraph(graph, s0, s1, z0, z1);
  codegen::Tiler tiler;
  AddBasicAxes(tiler, s0, s1, z0, z1);
  std::vector<ascir::AxisId> current_axis{z0.id};

  codegen::TPipe tpipe("tpipe", tiler);
  auto phase2_node = graph.FindNode("argmax_phase2_with_value");

  // index 输入 source：独立 anchor（AscTensor.attr 是引用，复制 value_in 会共享属性）
  auto index_in = MakePhase2IndexInput(graph);
  af::AscTensor value_in = graph.FindNode("load")->outputs[0];
  af::AscTensor value_out = phase2_node->outputs[0];  // value
  af::AscTensor index_out = phase2_node->outputs[1];  // index
  // value 通路：value_in / value_out 均为 FLOAT（{N,A} -> {A}）
  FillReduceTensorAttrs(value_in, value_out, z0, z1, s0, s1, ge::DT_FLOAT, ge::DT_FLOAT);
  // index 通路：index_in / index_out 均为 INT64；覆盖独立 tensor_id/buf.id 避免与 value 通路冲突
  FillPhase2IndexAttrs(index_in, index_out, z0, z1, s0, s1);
  auto value_in_tensor = MakeArgmaxTensor(value_in);
  auto index_in_tensor = MakeArgmaxTensor(index_in);
  auto value_out_tensor = MakeArgmaxTensor(value_out);
  auto index_out_tensor = MakeArgmaxTensor(index_out, index_in.attr.dtype);
  std::vector<std::reference_wrapper<const Tensor>> inputs{std::cref(value_in_tensor), std::cref(index_in_tensor)};
  std::vector<std::reference_wrapper<const Tensor>> outputs{std::cref(value_out_tensor), std::cref(index_out_tensor)};

  codegen::ApiTensor value_in_api_tensor, index_in_api_tensor, value_out_api_tensor, index_out_api_tensor;
  codegen::RegArgMaxPhase2WithValueApiCall call("ArgMaxMultiRPhase2WithValue");
  InitArgmaxCall(call, phase2_node);
  BindArgmaxInput(call, value_in_api_tensor, value_in_tensor, tpipe);
  BindArgmaxInput(call, index_in_api_tensor, index_in_tensor, tpipe);
  BindArgmaxOutput(call, value_out_api_tensor, value_out_tensor, tpipe);
  BindArgmaxOutput(call, index_out_api_tensor, index_out_tensor, tpipe);
  call.tmp_buf_id[0] = 1;

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
  att::NodeInfo info;
  ASSERT_NO_FATAL_FAILURE(VerifyPhase2WithValueParams(phase2_node, s0, s1, info));
  // 无 tmp buffer：不生成 tmp_reduce 临时变量与 tmp_buf 访问
  EXPECT_EQ(result.find("tmp_reduce ="), std::string::npos);
  EXPECT_NE(result.find("CastExtend<int32_t, int64_t, 1>"), std::string::npos) << result;
  // Rebuilding the same descriptor must replace previous candidates and keep mixed-width row pitches separate.
  ASSERT_NO_FATAL_FAILURE(
      VerifyPhase2MixedPitch(call, tpipe, current_axis, inputs, outputs, value_in_tensor, s1, info));
}

TEST_F(RegArgmaxApicallTest, RegArgMaxPhase2Api_Test_001) {
  // Argmax R 分核 Phase2（单输出，2 入 1 出）：value(FLOAT) + index(INT64) -> index(INT64)
  // 向量入口 ArgMaxMultiRPhase2VectorExtend<T,U>（内部混宽回退标量）。
  af::AscGraph graph("argmax_phase2_graph");
  af::Expression s0;
  af::Expression s1;
  af::Axis z0;
  af::Axis z1;
  CreateBasicAxes(graph, s0, s1, z0, z1);
  BuildPhase2Graph(graph, s0, s1, z0, z1);
  codegen::Tiler tiler;
  AddBasicAxes(tiler, s0, s1, z0, z1);
  std::vector<ascir::AxisId> current_axis{z0.id};

  codegen::TPipe tpipe("tpipe", tiler);
  auto phase2_node = graph.FindNode("argmax_phase2");

  auto index_in = MakePhase2IndexInput(graph);
  af::AscTensor value_in = graph.FindNode("load")->outputs[0];
  af::AscTensor index_out = phase2_node->outputs[0];
  // value/index 均为输入（{N,A} 每核一行）；index_out 为归约输出（{A}）。
  // FillReduceTensorAttrs(x, y, ...) 将 x 设为输入布局、y 设为输出布局：
  //   - index 通路：index_in 输入布局 / index_out 输出布局（同 WithValue UT）；
  //   - value 输入：单独触发输入布局填充（输出占位仅用于满足 helper 的 y 参数，dtype 会被第二次调用重置）。
  af::AscTensor value_out_placeholder = graph.FindNode("argmax_phase2")->outputs[0];
  FillReduceTensorAttrs(value_in, value_out_placeholder, z0, z1, s0, s1, ge::DT_FLOAT, ge::DT_FLOAT);
  FillPhase2IndexAttrs(index_in, index_out, z0, z1, s0, s1);

  std::vector<std::reference_wrapper<const Tensor>> inputs;
  std::vector<std::reference_wrapper<const Tensor>> outputs;
  auto value_in_tensor = MakeArgmaxTensor(value_in);
  auto index_in_tensor = MakeArgmaxTensor(index_in);
  auto index_out_tensor = MakeArgmaxTensor(index_out, index_in.attr.dtype);
  inputs.push_back(std::ref(value_in_tensor));
  inputs.push_back(std::ref(index_in_tensor));
  outputs.push_back(std::ref(index_out_tensor));

  codegen::ApiTensor value_in_api_tensor, index_in_api_tensor, index_out_api_tensor;
  codegen::RegArgMaxPhase2ApiCall call("ArgMaxMultiRPhase2");
  InitArgmaxCall(call, phase2_node);
  BindArgmaxInput(call, value_in_api_tensor, value_in_tensor, tpipe);
  BindArgmaxInput(call, index_in_api_tensor, index_in_tensor, tpipe);
  BindArgmaxOutput(call, index_out_api_tensor, index_out_tensor, tpipe);
  call.tmp_buf_id[0] = 1;

  std::string result;
  Status status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(status, af::SUCCESS);
  // 向量入口（内部混宽回退标量），模板参数 index/value dtype
  EXPECT_NE(result.find("ArgMaxMultiRPhase2VectorExtend<float>"), std::string::npos) << result;
  att::NodeInfo info;
  info.node_type = "ArgMaxMultiRPhase2";
  ASSERT_EQ(att::FillSpecificParams(phase2_node, info), af::SUCCESS);
  EXPECT_TRUE(info.argmax_node_params.valid);
  EXPECT_TRUE(info.argmax_node_params.is_phase2);
  EXPECT_FALSE(info.argmax_node_params.with_value);
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

  BuildPhase1Graph(graph, s0, s1, z0, z1);
  const SplitTileChain chain = BuildSplitTileChain(z1, s1t);
  const auto &z1T = chain.tile_outer;
  const auto &z1Ttb = chain.block_inner;

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s1t));
  AddAxesToTiler(tiler, {z0, z1, z1T, z1Ttb});
  std::vector<ascir::AxisId> current_axis;
  current_axis.push_back(z0.id);
  current_axis.push_back(z1Ttb.id);  // 最内层 loop 轴 = R 轴 block-inner

  codegen::TPipe tpipe("tpipe", tiler);
  auto phase1_node = graph.FindNode("argmax_phase1");

  af::AscTensor x_in = graph.FindNode("load")->outputs[0];
  af::AscTensor value_out = phase1_node->outputs[0];
  af::AscTensor index_out = phase1_node->outputs[1];
  FillReduceTensorAttrs(x_in, value_out, z0, z1, s0, s1, ge::DT_FLOAT, ge::DT_FLOAT);
  FillPhase1IndexAttrs(index_out, z0, z1, s0);
  auto x_tensor = MakeArgmaxTensor(x_in);
  auto value_tensor = MakeArgmaxTensor(value_out);
  auto index_tensor = MakeArgmaxTensor(index_out);

  std::vector<std::reference_wrapper<const Tensor>> inputs{std::ref(x_tensor)};
  std::vector<std::reference_wrapper<const Tensor>> outputs{std::ref(value_tensor), std::ref(index_tensor)};

  codegen::ApiTensor x_api, value_api, index_api;
  codegen::RegArgMaxPhase1ApiCall call("ArgMaxMultiRPhase1");
  InitArgmaxCall(call, phase1_node);
  BindArgmaxInput(call, x_api, x_tensor, tpipe);
  BindArgmaxOutput(call, value_api, value_tensor, tpipe);
  BindArgmaxOutput(call, index_api, index_tensor, tpipe, false);
  // Phase1 跨 tile 合并需要 index/value 两个 loop 级 tmp buffer（lifetime 0/1）
  call.tmp_buf_id[0] = 1;
  call.tmp_buf_id[1] = 2;

  std::string result;
  Status status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(status, af::SUCCESS) << result;
  ASSERT_NO_FATAL_FAILURE(VerifyPhase1ArResult(phase1_node, result));
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

  BuildPhase1Graph(graph, s0, s1, z0, z1);
  const SplitTileChain chain = BuildFullSplitTileChain(z1, s1t);
  const auto &z1T = chain.tile_outer;
  const auto &z1t = chain.tile_inner;
  const auto &z1TT = chain.block_outer;
  const auto &z1Ttb = chain.block_inner;

  codegen::Tiler tiler;
  tiler.AddSizeVar(af::SizeVar(s0));
  tiler.AddSizeVar(af::SizeVar(s1));
  tiler.AddSizeVar(af::SizeVar(s1t));
  AddAxesToTiler(tiler, {z0, z1, z1T, z1t, z1TT, z1Ttb});
  std::vector<ascir::AxisId> current_axis;
  current_axis.push_back(z0.id);
  current_axis.push_back(z1Ttb.id);

  codegen::TPipe tpipe("tpipe", tiler);
  auto phase1_node = graph.FindNode("argmax_phase1");

  af::AscTensor x_in = graph.FindNode("load")->outputs[0];
  af::AscTensor value_out = phase1_node->outputs[0];
  af::AscTensor index_out = phase1_node->outputs[1];
  FillReduceTensorAttrs(x_in, value_out, z0, z1, s0, s1, ge::DT_FLOAT, ge::DT_FLOAT);
  // 模拟真实调度：x 的 vectorized_axis 是 TileInner（z1t），而不是原始轴 z1。
  x_in.attr.vectorized_axis = {z0.id, z1t.id};
  x_in.attr.vectorized_strides = {af::ops::One, af::ops::One};
  FillPhase1IndexAttrs(index_out, z0, z1, s0);
  auto x_tensor = MakeArgmaxTensor(x_in);
  auto value_tensor = MakeArgmaxTensor(value_out);
  auto index_tensor = MakeArgmaxTensor(index_out);

  std::vector<std::reference_wrapper<const Tensor>> inputs{std::ref(x_tensor)};
  std::vector<std::reference_wrapper<const Tensor>> outputs{std::ref(value_tensor), std::ref(index_tensor)};

  codegen::ApiTensor x_api, value_api, index_api;
  codegen::RegArgMaxPhase1ApiCall call("ArgMaxMultiRPhase1");
  InitArgmaxCall(call, phase1_node);
  BindArgmaxInput(call, x_api, x_tensor, tpipe);
  BindArgmaxOutput(call, value_api, value_tensor, tpipe);
  BindArgmaxOutput(call, index_api, index_tensor, tpipe, false);
  call.tmp_buf_id[0] = 1;
  call.tmp_buf_id[1] = 2;

  std::string result;
  Status status = call.Generate(tpipe, current_axis, result);
  EXPECT_EQ(status, af::SUCCESS) << result;
  EXPECT_NE(result.find("uint32_t arg_reduce_index_base = (int64_t)z1T * (int64_t)t->s1t;"), std::string::npos)
      << result;
  EXPECT_NE(result.find("bool arg_reduce_first_tile = (z1Ttb == 0);"), std::string::npos) << result;

  // RA: the inner current axis is A, but firstTile and indexBase must still use R.
  ASSERT_NO_FATAL_FAILURE(VerifyPhase1RaCase(graph, call, tiler, chain, z0));
}

TEST_F(RegArgmaxApicallTest, ArgmaxRaFiveDimensionalDescriptorUsesMergedColumns) {
  af::AscGraph graph("argmax_ra_descriptor");
  af::Expression s0, s1;
  af::Axis z0, z1;
  BuildArgmaxGraph(graph, s0, s1, z0, z1);
  const auto r = graph.CreateAxis("r", af::Symbol(2048));
  const auto c0 = graph.CreateAxis("c0", af::Symbol(32));
  const auto c1 = graph.CreateAxis("c1", af::Symbol(16));
  codegen::Tiler tiler;
  for (const auto &axis : {z0, z1, r, c0, c1}) {
    ASSERT_EQ(tiler.AddAxis(axis), af::SUCCESS);
  }
  codegen::TPipe tpipe("tpipe", tiler);
  auto node = graph.FindNode("argmax");
  auto x = graph.FindNode("load")->outputs[0];
  auto y = node->outputs[0];
  FillReduceTensorAttrs(x, y, z0, z1, s0, s1, ge::DT_FLOAT, ge::DT_INT32);
  x.attr.axis = y.attr.axis = {z0.id, z1.id, r.id, c0.id, c1.id};
  x.attr.vectorized_axis = y.attr.vectorized_axis = {r.id, c0.id, c1.id};
  x.attr.repeats = {s0, s1, r.size, c0.size, c1.size};
  y.attr.repeats = {s0, s1, One, c0.size, c1.size};
  x.attr.strides = {s1 * r.size * c0.size * c1.size, r.size * c0.size * c1.size, c0.size * c1.size, c1.size, One};
  y.attr.strides = {s1 * c0.size * c1.size, c0.size * c1.size, Zero, c1.size, One};
  x.attr.vectorized_strides = {c0.size * c1.size, c1.size, One};
  y.attr.vectorized_strides = {Zero, c1.size, One};
  std::string value_dtype = "float";
  std::string index_dtype = "int32_t";
  const Tensor input(x, value_dtype);
  const Tensor output(y, index_dtype);
  RegArgmaxApiCall call("Argmax");
  call.node = node;
  ASSERT_EQ(call.BuildApiParam(tpipe, {z1.id}, {std::cref(input)}, {std::cref(output)}), af::SUCCESS);
  att::NodeInfo info;
  info.node_type = "Argmax";
  ASSERT_EQ(att::FillSpecificParams(node, info), af::SUCCESS);
  const auto &params = info.argmax_node_params;
  EXPECT_TRUE(params.valid);
  EXPECT_FALSE(params.is_ar);
  EXPECT_FALSE(params.has_tile_merge);
  ASSERT_EQ(params.src_shape.size(), 2U);
  EXPECT_EQ(params.src_shape[0], af::Symbol(2048));
  EXPECT_EQ(params.src_shape[1], af::Symbol(512));
  EXPECT_EQ(params.valid_extent, af::Symbol(2048));
}

class RegArgmaxDegenerateAxisTest : public RegArgmaxApicallTest,
                                    public ::testing::WithParamInterface<std::tuple<bool, int, ge::DataType>> {
 protected:
  struct DegenerateAxes {
    af::Axis r;
    af::Axis a;
    af::Axis rt;
    af::Axis ri;
    af::Axis rb;
    af::Axis rl;
  };

  static DegenerateAxes BuildAxes(af::AscGraph &graph, bool unit_r, bool all_unit) {
    DegenerateAxes axes;
    axes.r = graph.CreateAxis("r", af::Symbol(unit_r ? 1 : 19));
    axes.a = graph.CreateAxis("a", af::Symbol(unit_r && !all_unit ? 17 : 1));
    axes.rt = graph.CreateAxis("rt", af::Symbol(unit_r ? 1 : 3));
    axes.ri = graph.CreateAxis("ri", af::Symbol(unit_r ? 1 : 7));
    axes.rb = graph.CreateAxis("rb", af::Symbol(unit_r ? 1 : 2));
    axes.rl = graph.CreateAxis("rl", af::Symbol(unit_r ? 1 : 2));
    axes.rt.type = af::Axis::Type::kAxisTypeTileOuter;
    axes.rt.from = {axes.r.id};
    axes.rt.split_pair_other_id = axes.ri.id;
    axes.ri.type = af::Axis::Type::kAxisTypeTileInner;
    axes.ri.from = {axes.r.id};
    axes.ri.split_pair_other_id = axes.rt.id;
    axes.rb.type = af::Axis::Type::kAxisTypeBlockOuter;
    axes.rb.from = {axes.rt.id};
    axes.rb.split_pair_other_id = axes.rl.id;
    axes.rl.type = af::Axis::Type::kAxisTypeBlockInner;
    axes.rl.from = {axes.rt.id};
    axes.rl.split_pair_other_id = axes.rb.id;
    return axes;
  }

  static void FillTensor(af::AscTensor tensor, int64_t id, ge::DataType dtype, bool input, uint32_t physical_pitch,
                         bool unit_r, bool all_unit, const DegenerateAxes &axes) {
    tensor.attr.axis = {axes.rt.id, axes.ri.id, axes.a.id};
    tensor.attr.vectorized_axis =
        unit_r ? std::vector<ascir::AxisId>{axes.a.id, axes.ri.id} : std::vector<ascir::AxisId>{axes.ri.id, axes.a.id};
    tensor.attr.repeats = input ? std::vector<af::Expression>{axes.rt.size, axes.ri.size, axes.a.size}
                                : std::vector<af::Expression>{One, One, axes.a.size};
    tensor.attr.strides = input ? std::vector<af::Expression>{axes.ri.size * axes.a.size, axes.a.size, One}
                                : std::vector<af::Expression>{Zero, Zero, One};
    tensor.attr.vectorized_strides = {input ? af::Symbol(physical_pitch) : (unit_r && !all_unit ? One : Zero), Zero};
    tensor.attr.dtype = dtype;
    tensor.attr.mem.tensor_id = id;
    tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
    tensor.attr.mem.position = af::Position::kPositionVecCalc;
    tensor.attr.buf.id = id;
  }

  static af::AscNodePtr CreateReduceNode(const std::string &kind, af::AscGraph &graph, const Data &data) {
    if (kind == "ArgMaxMultiRPhase1") {
      af::ascir_op::ArgMaxMultiRPhase1 reduce("reduce");
      reduce.x = data.y;
    } else if (kind == "ArgmaxWithValue") {
      af::ascir_op::ArgmaxWithValue reduce("reduce");
      reduce.x = data.y;
    } else {
      af::ascir_op::Argmax reduce("reduce");
      reduce.x = data.y;
    }
    return graph.FindNode("reduce");
  }

  static std::unique_ptr<ApiCall> MakeCall(const std::string &kind, bool phase1, bool with_value) {
    if (phase1) {
      return std::make_unique<RegArgMaxPhase1ApiCall>(kind);
    }
    if (with_value) {
      return std::make_unique<RegArgmaxWithValueApiCall>(kind);
    }
    return std::make_unique<RegArgmaxApiCall>(kind);
  }

  static std::string JoinLines(const std::vector<std::string> &lines) {
    std::string joined;
    for (const auto &line : lines) {
      joined += line;
    }
    return joined;
  }

  static void VerifyPattern(const CodegenApiParam &api, const ApiCall &call, const Tiler &tiler, bool phase1,
                            bool unit_r, bool is_ar, const std::string &float_name, const std::string &pre,
                            const std::string &dimension) {
    ASSERT_EQ(api.template_params.size(), 2U);
    EXPECT_EQ(api.template_params[0], float_name);
    EXPECT_EQ(api.template_params[1], is_ar ? "AscendC::Pattern::Reduce::AR" : "AscendC::Pattern::Reduce::RA");
    if (phase1) {
      EXPECT_NE(pre.find(unit_r ? "arg_reduce_index_base = 0;" : "arg_reduce_index_base = (int64_t)rt * (int64_t)7;"),
                std::string::npos)
          << pre;
      EXPECT_NE(pre.find(unit_r ? "arg_reduce_first_tile = (0 == 0);" : "arg_reduce_first_tile = (rl == 0);"),
                std::string::npos)
          << pre;
      EXPECT_EQ(api.semantic_feature, is_ar ? "AR/MultiRound" : "RA/MultiRound");
    } else {
      EXPECT_EQ(api.semantic_feature, is_ar ? "AR" : "RA");
    }
    if (unit_r) {
      EXPECT_EQ(pre.find("rt *"), std::string::npos) << pre;
      EXPECT_EQ(pre.find("rl =="), std::string::npos) << pre;
    }
    std::stringstream dim;
    ASSERT_EQ(call.GenDimensionParam(api, tiler, dim), af::SUCCESS);
    (void)dimension;
  }

  static void VerifyParams(af::AscNodePtr node, const Tensor &x, const Tensor &y, bool phase1, bool unit_r, bool is_ar,
                           bool all_unit, uint32_t physical_pitch, const std::string &pre,
                           const std::string &dimension) {
    const auto params = ascir_param::GetAscirNodeParams(node);
    ASSERT_NE(params, nullptr);
    const auto *argmax = ascir_param::GetSpecificParams<ascir_param::ArgmaxNodeParams>(*params);
    ASSERT_NE(argmax, nullptr);
    EXPECT_EQ(argmax->is_ar, is_ar);
    EXPECT_EQ(argmax->valid_extent, unit_r ? One : GetDegenerateRiSize(unit_r));
    EXPECT_EQ(argmax->valid_a, !unit_r && !is_ar ? One : Zero);
    att::NodeInfo info;
    info.node_type = GetDegenerateKind(node);
    ASSERT_EQ(att::FillSpecificParams(node, info), af::SUCCESS);
    EXPECT_EQ(info.argmax_node_params.valid_a, argmax->valid_a);
    EXPECT_NE(dimension.find(unit_r ? (phase1 ? ", 1, arg_reduce_first_tile);" : ", 1);")
                                    : (is_ar ? "ri_actual_size" : "first_actual")),
              std::string::npos)
        << dimension;
    const auto valid_a_suffix = phase1 ? "arg_reduce_first_tile, 1);" : "first_actual, 1);";
    if (!unit_r && !is_ar) {
      EXPECT_NE(dimension.find(valid_a_suffix), std::string::npos) << dimension;
    } else {
      EXPECT_EQ(dimension.find(valid_a_suffix), std::string::npos) << dimension;
    }
    ASSERT_EQ(argmax->src_shape.size(), 2U);
    if (unit_r) {
      EXPECT_EQ(argmax->src_shape[0], GetDegenerateASize(unit_r, all_unit));
      EXPECT_EQ(argmax->src_shape[1], af::Symbol(all_unit ? 1 : physical_pitch));
      EXPECT_NE(pre.find(all_unit ? "uint32_t first_actual = 1;" : "uint32_t first_actual = 1 * a_actual_size;"),
                std::string::npos)
          << pre;
      EXPECT_NE(pre.find("last = " + std::to_string(all_unit ? 1 : physical_pitch) + ";"), std::string::npos) << pre;
    } else if (!is_ar) {
      EXPECT_EQ(argmax->src_shape[0], af::Symbol(7));
      EXPECT_EQ(argmax->src_shape[1], af::Symbol(physical_pitch));
      EXPECT_NE(pre.find("uint32_t first_actual = 1 * ri_actual_size;"), std::string::npos) << pre;
      EXPECT_NE(pre.find("last = " + std::to_string(physical_pitch) + ";"), std::string::npos) << pre;
      EXPECT_NE(pre.find("last_actual = last;"), std::string::npos) << pre;
    } else {
      EXPECT_EQ(argmax->src_shape[0], One);
      EXPECT_EQ(argmax->src_shape[1], af::Symbol(8));  // Seven elements aligned to a block for all three dtypes.
      EXPECT_EQ(pre.find("last_actual = last;"), std::string::npos) << pre;
    }
    // Canonicalization is a local API view; neither logical tensor shape nor shared strides may change.
    EXPECT_EQ(x.vectorized_strides.front(), af::Symbol(physical_pitch));
    EXPECT_EQ(x.vectorized_strides.back(), Zero);
    EXPECT_EQ(y.vectorized_strides.back(), Zero);
    EXPECT_EQ(y.axis_size.back(), GetDegenerateASize(unit_r, all_unit));
    if (unit_r) {
      EXPECT_EQ(dimension.find("ri_actual_size"), std::string::npos) << dimension;
    }
  }

  static af::Expression GetDegenerateRiSize(bool unit_r) {
    return af::Symbol(unit_r ? 1 : 7);
  }

  static af::Expression GetDegenerateASize(bool unit_r, bool all_unit) {
    return af::Symbol(unit_r && !all_unit ? 17 : 1);
  }

  static std::string GetDegenerateKind(const af::AscNodePtr &node) {
    return node->GetType();
  }

  static void VerifyUnitTileCase(TPipe &pipe, ApiCall &call, const Tensor &x, af::AscNodePtr node,
                                 const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                 const std::vector<ascir::AxisId> &current, const DegenerateAxes &axes) {
    EXPECT_NE(call.BuildApiParam(pipe, {axes.a.id}, {std::cref(x)}, outputs), af::SUCCESS);
    // A tile of length one still needs the R loop when the original reduction is non-unit.
    Tiler unit_tile_tiler;
    for (auto axis : {axes.r, axes.a, axes.rt, axes.ri, axes.rb, axes.rl}) {
      if (axis.id == axes.ri.id) {
        axis.size = One;
      }
      ASSERT_EQ(unit_tile_tiler.AddAxis(axis), af::SUCCESS);
    }
    TPipe unit_tile_pipe("tpipe", unit_tile_tiler);
    ASSERT_EQ(call.BuildApiParam(unit_tile_pipe, current, {std::cref(x)}, outputs), af::SUCCESS);
    const auto unit_tile_api = CodegenApiParam::TryGetNodeApiParam(node);
    ASSERT_NE(unit_tile_api, nullptr);
    const std::string unit_tile_pre = JoinLines(unit_tile_api->api_pre_process);
    EXPECT_NE(unit_tile_pre.find("arg_reduce_index_base = (int64_t)rt;"), std::string::npos) << unit_tile_pre;
    EXPECT_NE(unit_tile_pre.find("arg_reduce_first_tile = (rl == 0);"), std::string::npos) << unit_tile_pre;
  }

  static void VerifyDegenerateKind(const std::string &kind, bool unit_r, int physical_pitch, ge::DataType dtype) {
    const bool all_unit = unit_r && physical_pitch == 0;
    const bool is_ar = unit_r || physical_pitch == 1;
    SCOPED_TRACE(std::string(kind) + (unit_r ? "/unit_r" : "/scalar_output"));
    const bool phase1 = std::string(kind) == "ArgMaxMultiRPhase1";
    const bool with_value = std::string(kind) != "Argmax";
    af::AscGraph graph("argmax_degenerate_axis");
    const auto axes = BuildAxes(graph, unit_r, all_unit);
    Data data("data", graph);
    auto node = CreateReduceNode(kind, graph, data);
    ASSERT_NE(node, nullptr);
    FillTensor(graph.FindNode("data")->outputs[0], 1, dtype, true, physical_pitch, unit_r, all_unit, axes);
    if (with_value) {
      FillTensor(node->outputs[0], 2, dtype, false, physical_pitch, unit_r, all_unit, axes);
    }
    FillTensor(node->outputs[with_value ? 1 : 0], 3, phase1 ? ge::DT_INT32 : ge::DT_INT64, false, physical_pitch,
               unit_r, all_unit, axes);
    std::string float_name;
    ASSERT_EQ(Tensor::DtypeName(dtype, float_name), af::SUCCESS);
    std::string index_name = phase1 ? "int32_t" : "int64_t";
    Tensor x(graph.FindNode("data")->outputs[0], float_name);
    Tensor y(node->outputs[0], with_value ? float_name : index_name);
    Tensor index(node->outputs[with_value ? 1 : 0], index_name);
    Tiler tiler;
    AddAxesToTiler(tiler, {axes.r, axes.a, axes.rt, axes.ri, axes.rb, axes.rl});
    TPipe pipe("tpipe", tiler);
    auto call = MakeCall(kind, phase1, with_value);
    call->node = node;
    call->tmp_buf_id = {{0, 0}, {1, 1}, {2, 2}, {3, 3}};
    const std::vector<ascir::AxisId> current =
        phase1 && !unit_r ? std::vector<ascir::AxisId>{axes.rl.id, axes.a.id} : std::vector<ascir::AxisId>{axes.a.id};
    std::vector<std::reference_wrapper<const Tensor>> outputs{std::cref(y)};
    if (with_value) {
      outputs.emplace_back(std::cref(index));
    }
    ASSERT_EQ(call->BuildApiParam(pipe, current, {std::cref(x)}, outputs), af::SUCCESS);
    const auto api = CodegenApiParam::TryGetNodeApiParam(node);
    ASSERT_NE(api, nullptr);
    const std::string pre = JoinLines(api->api_pre_process);
    std::stringstream dimension;
    ASSERT_EQ(call->GenDimensionParam(*api, tiler, dimension), af::SUCCESS);
    VerifyPattern(*api, *call, tiler, phase1, unit_r, is_ar, float_name, pre, dimension.str());
    VerifyParams(node, x, y, phase1, unit_r, is_ar, all_unit, static_cast<uint32_t>(physical_pitch), pre,
                 dimension.str());
    if (phase1 && !unit_r) {
      VerifyUnitTileCase(pipe, *call, x, node, outputs, current, axes);
    }
  }
};

TEST_P(RegArgmaxDegenerateAxisTest, EffectivePatternAndEnclosingReductionLoop) {
  for (const auto &kind : {"Argmax", "ArgmaxWithValue", "ArgMaxMultiRPhase1"}) {
    ASSERT_NO_FATAL_FAILURE(
        VerifyDegenerateKind(kind, std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam())));
  }
}

INSTANTIATE_TEST_SUITE_P(ScalarOutputAndUnitReduction, RegArgmaxDegenerateAxisTest,
                         ::testing::Combine(::testing::Values(false, true), ::testing::Values(1, 4, 8, 16, 64),
                                            ::testing::Values(ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64)));

INSTANTIATE_TEST_SUITE_P(AllUnitZeroStrides, RegArgmaxDegenerateAxisTest,
                         ::testing::Values(std::make_tuple(true, 0, ge::DT_FLOAT),
                                           std::make_tuple(true, 0, ge::DT_INT32),
                                           std::make_tuple(true, 0, ge::DT_INT64)));
