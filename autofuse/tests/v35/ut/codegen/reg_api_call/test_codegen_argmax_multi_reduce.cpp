/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"
#include "ascir_ops.h"
#include "codegen_kernel.h"
#include "platform_context.h"
#include "reg_arg_reduce_api_call.h"
#include "reg_store_api_call.h"
#include "runtime_stub.h"
#include "ascir_node_param/ascir_node_param.h"

namespace {
using namespace af::ops;
using namespace codegen;

enum class ConsumerPosition { kOuter, kSame, kInner, kMixed };
using ArgmaxMultiReduceParam = std::tuple<bool, ge::DataType, ConsumerPosition, bool, bool>;

size_t ClosingBrace(const std::string &source, size_t opening) {
  size_t depth = 0;
  for (size_t i = opening; i < source.size(); ++i) {
    if (source[i] == '{') {
      ++depth;
    } else if (source[i] == '}' && --depth == 0) {
      return i;
    }
  }
  return std::string::npos;
}

class ArgmaxMultiReduceLoopTest : public testing::TestWithParam<ArgmaxMultiReduceParam> {
 protected:
  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
    ge::RuntimeStub::SetInstance(std::make_shared<ge::RuntimeStubV2Common>());
  }
  void TearDown() override {
    ge::RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
  }
};

struct MultiReduceAxes {
  af::Axis a;
  af::Axis av;
  af::Axis original_r;
  af::Axis r_tile;
  af::Axis r_inner;
  af::Axis r_block;
  af::Axis r;
  af::Axis inner;
};

MultiReduceAxes CreateMultiReduceAxes(af::AscGraph &graph, bool scalar_output) {
  MultiReduceAxes axes;
  axes.a = graph.CreateAxis("a", af::Symbol(scalar_output ? 1 : 3));
  axes.av = graph.CreateAxis("av", af::Symbol(scalar_output ? 1 : 16));
  axes.original_r = graph.CreateAxis("original_r", af::Symbol(19));
  axes.r_tile = graph.CreateAxis("r_tile", af::Symbol(3));
  axes.r_inner = graph.CreateAxis("r_inner", af::Symbol(8));
  axes.r_block = graph.CreateAxis("r_block", af::Symbol(2));
  axes.r = graph.CreateAxis("r", af::Symbol(2));
  axes.inner = graph.CreateAxis("consumer_inner", af::Symbol(2));
  axes.r_tile.type = af::Axis::Type::kAxisTypeTileOuter;
  axes.r_tile.from = {axes.original_r.id};
  axes.r_tile.split_pair_other_id = axes.r_inner.id;
  axes.r_inner.type = af::Axis::Type::kAxisTypeTileInner;
  axes.r_inner.from = {axes.original_r.id};
  axes.r_inner.split_pair_other_id = axes.r_tile.id;
  axes.r_block.type = af::Axis::Type::kAxisTypeBlockOuter;
  axes.r_block.from = {axes.r_tile.id};
  axes.r_block.split_pair_other_id = axes.r.id;
  axes.r.type = af::Axis::Type::kAxisTypeBlockInner;
  axes.r.from = {axes.r_tile.id};
  axes.r.split_pair_other_id = axes.r_block.id;
  return axes;
}

struct MultiReduceSourcePositions {
  size_t a_for;
  size_t r_for;
  size_t a_end;
  size_t r_end;
  size_t merge;
  size_t increment;
  size_t materialize;
};

// Own all objects referenced by the loop tree and ApiTensor readers for the whole test.
struct MultiReduceCase {
  bool with_value;
  ge::DataType index_dtype;
  ConsumerPosition position;
  bool scalar_output;
  bool is_min;
  af::AscGraph graph;
  MultiReduceAxes axes;
  Kernel kernel;
  af::AscNodePtr reduce_node;
  size_t index_output;
  int64_t index_id;
  std::unique_ptr<ApiCall> reduce_call;
  ApiTensor input_api;
  Loop a_loop;
  Loop block_loop;
  Loop r_loop;
  Loop inner_loop;
  StoreRegApiCall index_call;
  StoreRegApiCall other_call;
  ApiCall index_output_call;
  ApiCall other_output_call;
  Loop *index_loop{nullptr};
  Loop *other_loop{nullptr};

  explicit MultiReduceCase(const ArgmaxMultiReduceParam &param)
      : with_value(std::get<0>(param)),
        index_dtype(std::get<1>(param)),
        position(std::get<2>(param)),
        scalar_output(std::get<3>(param)),
        is_min(std::get<4>(param)),
        graph(is_min ? "argmin_multi_reduce_loop" : "argmax_multi_reduce_loop"),
        axes(CreateMultiReduceAxes(graph, scalar_output)),
        kernel(graph.GetName()),
        index_output(with_value ? 1U : 0U),
        index_id(with_value ? 5 : 3),
        a_loop(axes.a.id),
        block_loop(axes.r_block.id),
        r_loop(axes.r.id),
        inner_loop(axes.inner.id),
        index_call("DataCopyPadExtend"),
        other_call("DataCopyPadExtend"),
        index_output_call("Output"),
        other_output_call("Output") {}

  template <typename Op, bool withValue>
  void ConnectReduce(af::ascir_op::Data &input, af::ascir_op::Store &index_store, af::ascir_op::Store &other_store) {
    Op reduce("reduce");
    graph.AddNode(reduce);
    reduce.x = input.y;
    if constexpr (withValue) {
      index_store.x = reduce.index;
      other_store.x = reduce.value;
    } else {
      index_store.x = reduce.y;
      other_store.x = reduce.y;
    }
  }

  void BuildGraph() {
    for (const auto &axis :
         {axes.a, axes.av, axes.original_r, axes.r_tile, axes.r_inner, axes.r_block, axes.r, axes.inner}) {
      ASSERT_EQ(kernel.tiler.AddAxis(axis), af::SUCCESS);
    }
    af::ascir_op::Data input("input", graph);
    af::ascir_op::Store index_store("index_store");
    af::ascir_op::Store other_store("other_store");
    graph.AddNode(index_store);
    graph.AddNode(other_store);
    if (with_value) {
      if (is_min) {
        ConnectReduce<af::ascir_op::ArgminWithValue, true>(input, index_store, other_store);
      } else {
        ConnectReduce<af::ascir_op::ArgmaxWithValue, true>(input, index_store, other_store);
      }
    } else {
      if (is_min) {
        ConnectReduce<af::ascir_op::Argmin, false>(input, index_store, other_store);
      } else {
        ConnectReduce<af::ascir_op::Argmax, false>(input, index_store, other_store);
      }
    }
    index_store.ir_attr.SetOffset(Zero);
    other_store.ir_attr.SetOffset(Zero);
    reduce_node = graph.FindNode("reduce");
    ASSERT_NE(reduce_node, nullptr);
  }

  void FillTensor(af::AscTensor tensor, int64_t id, ge::DataType dtype, bool is_input, bool gm) {
    const auto &a = axes.a;
    const auto &av = axes.av;
    const auto &rt = axes.r_tile;
    const auto &ri = axes.r_inner;
    tensor.attr.axis = {a.id, av.id, rt.id, ri.id};
    tensor.attr.vectorized_axis = {av.id, ri.id};
    tensor.attr.repeats = is_input ? std::vector<af::Expression>{a.size, av.size, rt.size, ri.size}
                                   : std::vector<af::Expression>{a.size, av.size, One, One};
    tensor.attr.strides =
        is_input ? std::vector<af::Expression>{av.size * rt.size * ri.size, rt.size * ri.size, ri.size, One}
                 : std::vector<af::Expression>{av.size, One, Zero, Zero};
    tensor.attr.vectorized_strides =
        is_input ? std::vector<af::Expression>{ri.size, One} : std::vector<af::Expression>{One, Zero};
    tensor.attr.dtype = dtype;
    tensor.attr.mem.tensor_id = id;
    tensor.attr.mem.alloc_type = gm ? af::AllocType::kAllocTypeGlobal : af::AllocType::kAllocTypeBuffer;
    tensor.attr.mem.position = gm ? af::Position::kPositionGM : af::Position::kPositionVecCalc;
    tensor.attr.buf.id = id;
    tensor.attr.opt.merge_scope = af::kIdNone;
    ASSERT_EQ(kernel.tpipe.AddTensor(tensor), af::SUCCESS);
  }

  void FillTensors() {
    ASSERT_NO_FATAL_FAILURE(FillTensor(graph.FindNode("input")->outputs[0], 1, ge::DT_FLOAT, true, false));
    if (with_value) {
      ASSERT_NO_FATAL_FAILURE(FillTensor(reduce_node->outputs[0], 3, ge::DT_FLOAT, false, false));
    }
    ASSERT_NO_FATAL_FAILURE(FillTensor(reduce_node->outputs[index_output], index_id, index_dtype, false, false));
    for (auto output : reduce_node->outputs()) {
      auto *tensor = kernel.tpipe.GetTensor(output->attr.mem.tensor_id);
      ASSERT_NE(tensor, nullptr);
      EXPECT_EQ(tensor->is_ub_scalar, scalar_output);
      ASSERT_EQ(kernel.ParseOptimizeInfo(reduce_node, *output), af::SUCCESS);
      EXPECT_FALSE(tensor->is_ub_scalar);
    }
    ASSERT_NO_FATAL_FAILURE(FillTensor(graph.FindNode("index_store")->outputs[0], 10, index_dtype, false, true));
    if (with_value || position == ConsumerPosition::kMixed) {
      ASSERT_NO_FATAL_FAILURE(FillTensor(graph.FindNode("other_store")->outputs[0], 11,
                                         with_value ? ge::DT_FLOAT : index_dtype, false, true));
    }
    ASSERT_FALSE(testing::Test::HasFatalFailure());
  }

  void InitReduceCall() {
    if (with_value) {
      reduce_call = std::make_unique<RegArgReduceWithValueApiCall>(is_min ? "ArgminWithValue" : "ArgmaxWithValue");
    } else {
      reduce_call = std::make_unique<RegArgReduceApiCall>(is_min ? "Argmin" : "Argmax");
    }
    reduce_node->attr.api.unit = af::ComputeUnit::kUnitVector;
    reduce_node->attr.api.compute_type = af::ComputeType::kComputeReduce;
    ASSERT_EQ(reduce_call->Init(reduce_node), af::SUCCESS);
    reduce_call->axis = axes.r.id;
    reduce_call->depth = 3;
    reduce_call->tmp_buf_id = {{0, 20}, {1, 21}, {2, 22}, {3, 23}};
    input_api.id = 1;
    input_api.reads = {reduce_call.get()};
    reduce_call->inputs = {&input_api};
  }

  void ConnectStore(StoreRegApiCall &store, const std::string &name, Loop *loop, ApiTensor &source, ApiCall &output) {
    auto node = graph.FindNode(name.c_str());
    node->attr.api.unit = af::ComputeUnit::kUnitVector;
    node->attr.api.compute_type = af::ComputeType::kComputeStore;
    ASSERT_EQ(store.Init(node), af::SUCCESS);
    store.axis = loop->axis_id;
    store.depth = loop == &block_loop ? 2 : loop == &r_loop ? 3 : 4;
    store.inputs = {&source};
    source.reads.push_back(&store);
    output.axis = store.axis;
    output.depth = store.depth;
    output.unit = af::ComputeUnit::kUnitNone;
    store.outputs[0].reads = {&output};
    loop->AddCall(&store);
  }

  void SetupLoops() {
    a_loop.AddLoop(&block_loop);
    block_loop.AddLoop(&r_loop);
    r_loop.AddCall(reduce_call.get());
    if (position == ConsumerPosition::kInner) {
      r_loop.AddLoop(&inner_loop);
    }
    index_loop = position == ConsumerPosition::kSame    ? &r_loop
                 : position == ConsumerPosition::kInner ? &inner_loop
                                                        : &block_loop;
    other_loop = position == ConsumerPosition::kMixed ? &r_loop : index_loop;
    if (with_value || position == ConsumerPosition::kMixed) {
      ASSERT_NO_FATAL_FAILURE(
          ConnectStore(other_call, "other_store", other_loop, reduce_call->outputs[0], other_output_call));
    }
    ASSERT_NO_FATAL_FAILURE(
        ConnectStore(index_call, "index_store", index_loop, reduce_call->outputs[index_output], index_output_call));
    ASSERT_FALSE(testing::Test::HasFatalFailure());
  }

  void VerifyNodeParams() const {
    const auto node_params = ascir_param::GetAscirNodeParams(reduce_node);
    ASSERT_NE(node_params, nullptr);
    const auto *params = ascir_param::GetSpecificParams<ascir_param::ArgmaxNodeParams>(*node_params);
    ASSERT_NE(params, nullptr);
    EXPECT_TRUE(params->valid);
    EXPECT_TRUE(params->has_tile_merge);
    EXPECT_TRUE(params->with_value);
    EXPECT_EQ(params->has_index_cast, index_dtype != ge::DT_INT32);
    EXPECT_EQ(node_params->api_name, reduce_node->GetType());
  }

  void VerifyLoopBounds(const std::string &source, MultiReduceSourcePositions &p) const {
    p.a_for = source.find("for (int a = 0; a < a_loop_size; a++) {");
    p.r_for = source.find("for (int r = 0; r < r_loop_size; r++) {");
    ASSERT_NE(p.a_for, std::string::npos);
    ASSERT_NE(p.r_for, std::string::npos);
    p.a_end = ClosingBrace(source, source.find('{', p.a_for));
    p.r_end = ClosingBrace(source, source.find('{', p.r_for));
    ASSERT_NE(p.a_end, std::string::npos);
    ASSERT_NE(p.r_end, std::string::npos);
    EXPECT_LT(p.a_for, p.r_for);
    EXPECT_LT(p.r_end, p.a_end);
    EXPECT_EQ(source.find("for (int r =", p.r_for + 1), std::string::npos);
    const auto actual = source.find("r_actual_size = r_block < r_block_axis_size ? r_axis_size : r_tail_size;");
    const auto extent = source.find("r_loop_size = r_actual_size;");
    ASSERT_NE(actual, std::string::npos);
    ASSERT_NE(extent, std::string::npos);
    EXPECT_LT(p.a_for, actual);
    EXPECT_LT(actual, extent);
    EXPECT_LT(extent, p.r_for);
    EXPECT_NE(source.find("r_inner_actual_size = r_tile < r_tile_axis_size ? r_inner_axis_size : r_inner_tail_size;"),
              std::string::npos);
  }

  void VerifyReductionSteps(const std::string &source, MultiReduceSourcePositions &p) const {
    const auto reset = source.find("if (r == 0) {\n  accumulated_offset = 0;\n}");
    EXPECT_NE(source.find("static int64_t accumulated_offset = 0;"), std::string::npos);
    const auto reduce =
        source.find(std::string(is_min ? "ArgMin" : "ArgMax") + "WithValueExtend<float, AscendC::Pattern::Reduce::AR>");
    const auto first_copy = source.find("ArgReduceCopy(tmp_arg_reduce_index_saved[0], tmp_arg_reduce_index[0], local_" +
                                        std::to_string(index_id) + "_actual_size);");
    const std::string merge_call = std::string("ArgReduceMergeRound<float, ") + (is_min ? "false>" : "true>");
    p.merge = source.find(merge_call);
    p.increment = source.find("accumulated_offset += r_inner_actual_size;");
    for (const auto pos : {reset, reduce, first_copy, p.merge, p.increment}) {
      ASSERT_NE(pos, std::string::npos);
      EXPECT_LT(p.r_for, pos);
      EXPECT_LT(pos, p.r_end);
    }
    EXPECT_LT(reset, reduce);
    EXPECT_LT(reduce, first_copy);
    EXPECT_LT(first_copy, p.merge);
    EXPECT_NE(source.find("tmp_reduce_shape, false, accumulated_offset, r_inner_actual_size);"), std::string::npos);
    if (with_value) {
      const auto value_copy = source.find("ArgReduceCopy(local_3[0], tmp_arg_reduce_value[0], local_3_actual_size);");
      ASSERT_NE(value_copy, std::string::npos);
      EXPECT_LT(first_copy, value_copy);
      EXPECT_LT(value_copy, p.merge);
      EXPECT_NE(source.find(merge_call + "(local_3[0], tmp_arg_reduce_index_saved[0],"), std::string::npos);
    }
  }

  void VerifyIndexMaterialization(const std::string &source, MultiReduceSourcePositions &p) const {
    const std::string local = "local_" + std::to_string(index_id);
    const bool cast = index_dtype == ge::DT_INT64;
    p.materialize =
        source.find(cast ? "CastExtend<int32_t, int64_t, 1>(" + local + "[0], tmp_arg_reduce_index_saved[0],"
                         : "ArgReduceCopy(" + local + "[0], tmp_arg_reduce_index_saved[0],");
    ASSERT_NE(p.materialize, std::string::npos);
    EXPECT_LT(p.merge, p.materialize);
    EXPECT_LT(p.materialize, p.increment);
    EXPECT_EQ(source.find("CastExtend<int32_t, int64_t, 1>"), cast ? p.materialize : std::string::npos);
    if (cast) {
      EXPECT_EQ(source.find("CastExtend<int32_t, int64_t, 1>", p.materialize + 1), std::string::npos);
    } else {
      EXPECT_EQ(source.find("ArgReduceCopy(" + local + "[0],", p.materialize + 1), std::string::npos);
    }
    const auto guard = source.find("if (r + 1 == r_loop_size) {");
    if (position == ConsumerPosition::kOuter) {
      ASSERT_NE(guard, std::string::npos);
      const auto guard_end = ClosingBrace(source, source.find('{', guard));
      ASSERT_NE(guard_end, std::string::npos);
      EXPECT_LT(p.merge, guard);
      EXPECT_LT(guard, p.materialize);
      EXPECT_LT(p.materialize, guard_end);
      EXPECT_LT(guard_end, p.increment);
      EXPECT_EQ(source.find("if (r + 1 ==", guard + 1), std::string::npos);
    } else {
      EXPECT_EQ(source.find("if (r + 1 =="), std::string::npos);
    }
  }

  void VerifyConsumers(const std::string &source, const MultiReduceSourcePositions &p) const {
    const auto consumer = source.find("(global_10[");
    ASSERT_NE(consumer, std::string::npos);
    EXPECT_LT(p.materialize, consumer);
    EXPECT_LT(consumer, p.a_end);
    if (position == ConsumerPosition::kOuter || position == ConsumerPosition::kMixed) {
      EXPECT_LT(p.r_end, consumer);
    } else {
      EXPECT_LT(consumer, p.r_end);
    }
    if (position == ConsumerPosition::kInner) {
      const auto consumer_for = source.find("for (int consumer_inner =");
      ASSERT_NE(consumer_for, std::string::npos);
      EXPECT_LT(p.materialize, consumer_for);
      EXPECT_LT(consumer_for, consumer);
      EXPECT_LT(consumer, ClosingBrace(source, source.find('{', consumer_for)));
    }
    if (with_value || position == ConsumerPosition::kMixed) {
      const auto other = source.find("(global_11[");
      ASSERT_NE(other, std::string::npos);
      EXPECT_LT(p.materialize, other);
      EXPECT_LT(other, p.a_end);
      if (other_loop == &block_loop) {
        EXPECT_LT(p.r_end, other);
      } else {
        EXPECT_LT(other, p.r_end);
      }
    }
    EXPECT_EQ(std::count(source.begin(), source.end(), '{'), std::count(source.begin(), source.end(), '}'));
    for (const auto *first_round : {"if (r < 1)", "if (r < 2)"}) {
      const auto pos = source.find(first_round);
      if (pos != std::string::npos) {
        EXPECT_GT(pos, p.increment);
      }
    }
  }
};

TEST_P(ArgmaxMultiReduceLoopTest, MaterializeIndexAtConsumerRequiredIteration) {
  MultiReduceCase state(GetParam());
  ASSERT_NO_FATAL_FAILURE(state.BuildGraph());
  ASSERT_NO_FATAL_FAILURE(state.FillTensors());
  ASSERT_NO_FATAL_FAILURE(state.InitReduceCall());
  ASSERT_NO_FATAL_FAILURE(state.SetupLoops());
  std::string source;
  ASSERT_EQ(state.a_loop.Generate(state.kernel.tiler, state.kernel.tpipe, source), af::SUCCESS);
  ASSERT_NO_FATAL_FAILURE(state.VerifyNodeParams());
  SCOPED_TRACE(source);
  MultiReduceSourcePositions positions{};
  ASSERT_NO_FATAL_FAILURE(state.VerifyLoopBounds(source, positions));
  ASSERT_NO_FATAL_FAILURE(state.VerifyReductionSteps(source, positions));
  ASSERT_NO_FATAL_FAILURE(state.VerifyIndexMaterialization(source, positions));
  ASSERT_NO_FATAL_FAILURE(state.VerifyConsumers(source, positions));
}

INSTANTIATE_TEST_SUITE_P(OrdinaryAndWithValue, ArgmaxMultiReduceLoopTest,
                         testing::Combine(testing::Bool(), testing::Values(ge::DT_INT64, ge::DT_INT32),
                                          testing::Values(ConsumerPosition::kOuter, ConsumerPosition::kSame,
                                                          ConsumerPosition::kInner, ConsumerPosition::kMixed),
                                          testing::Bool(), testing::Bool()));
}  // namespace
