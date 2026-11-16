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
#include "reg_argmax_api_call.h"
#include "reg_store_api_call.h"
#include "runtime_stub.h"

namespace {
using namespace af::ops;
using namespace codegen;

enum class ConsumerPosition { kOuter, kSame, kInner, kMixed };
using ArgmaxMultiReduceParam = std::tuple<bool, ge::DataType, ConsumerPosition, bool>;

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

TEST_P(ArgmaxMultiReduceLoopTest, MaterializeIndexAtConsumerRequiredIteration) {
  const auto with_value = std::get<0>(GetParam());
  const auto index_dtype = std::get<1>(GetParam());
  const auto position = std::get<2>(GetParam());
  const bool scalar_output = std::get<3>(GetParam());
  const bool deferred = position == ConsumerPosition::kOuter;
  af::AscGraph graph("argmax_multi_reduce_loop");
  auto a = graph.CreateAxis("a", af::Symbol(scalar_output ? 1 : 3));
  auto av = graph.CreateAxis("av", af::Symbol(scalar_output ? 1 : 16));
  auto original_r = graph.CreateAxis("original_r", af::Symbol(19));
  auto r_tile = graph.CreateAxis("r_tile", af::Symbol(3));
  auto r_inner = graph.CreateAxis("r_inner", af::Symbol(8));
  auto r_block = graph.CreateAxis("r_block", af::Symbol(2));
  auto r = graph.CreateAxis("r", af::Symbol(2));
  auto inner = graph.CreateAxis("consumer_inner", af::Symbol(2));
  r_tile.type = af::Axis::Type::kAxisTypeTileOuter;
  r_tile.from = {original_r.id};
  r_tile.split_pair_other_id = r_inner.id;
  r_inner.type = af::Axis::Type::kAxisTypeTileInner;
  r_inner.from = {original_r.id};
  r_inner.split_pair_other_id = r_tile.id;
  r_block.type = af::Axis::Type::kAxisTypeBlockOuter;
  r_block.from = {r_tile.id};
  r_block.split_pair_other_id = r.id;
  r.type = af::Axis::Type::kAxisTypeBlockInner;
  r.from = {r_tile.id};
  r.split_pair_other_id = r_block.id;

  Kernel kernel(graph.GetName());
  auto &tiler = kernel.tiler;
  for (const auto &axis : {a, av, original_r, r_tile, r_inner, r_block, r, inner}) {
    ASSERT_EQ(tiler.AddAxis(axis), af::SUCCESS);
  }
  auto &tpipe = kernel.tpipe;
  af::ascir_op::Data input("input", graph);
  af::ascir_op::Store index_store("index_store");
  af::ascir_op::Store other_store("other_store");
  graph.AddNode(index_store);
  graph.AddNode(other_store);
  if (with_value) {
    af::ascir_op::ArgmaxWithValue reduce("reduce");
    graph.AddNode(reduce);
    reduce.x = input.y;
    index_store.x = reduce.index;
    other_store.x = reduce.value;
  } else {
    af::ascir_op::Argmax reduce("reduce");
    graph.AddNode(reduce);
    reduce.x = input.y;
    index_store.x = reduce.y;
    other_store.x = reduce.y;
  }
  index_store.ir_attr.SetOffset(Zero);
  other_store.ir_attr.SetOffset(Zero);

  auto reduce_node = graph.FindNode("reduce");
  ASSERT_NE(reduce_node, nullptr);
  auto fill_tensor = [&](af::AscTensor tensor, int64_t id, ge::DataType dtype, bool is_input, bool gm) {
    tensor.attr.axis = {a.id, av.id, r_tile.id, r_inner.id};
    tensor.attr.vectorized_axis = {av.id, r_inner.id};
    tensor.attr.repeats = is_input ? std::vector<af::Expression>{a.size, av.size, r_tile.size, r_inner.size}
                                   : std::vector<af::Expression>{a.size, av.size, One, One};
    tensor.attr.strides = is_input ? std::vector<af::Expression>{av.size * r_tile.size * r_inner.size,
                                                                 r_tile.size * r_inner.size, r_inner.size, One}
                                   : std::vector<af::Expression>{av.size, One, Zero, Zero};
    tensor.attr.vectorized_strides =
        is_input ? std::vector<af::Expression>{r_inner.size, One} : std::vector<af::Expression>{One, Zero};
    tensor.attr.dtype = dtype;
    tensor.attr.mem.tensor_id = id;
    tensor.attr.mem.alloc_type = gm ? af::AllocType::kAllocTypeGlobal : af::AllocType::kAllocTypeBuffer;
    tensor.attr.mem.position = gm ? af::Position::kPositionGM : af::Position::kPositionVecCalc;
    tensor.attr.buf.id = id;
    tensor.attr.opt.merge_scope = af::kIdNone;
    ASSERT_EQ(tpipe.AddTensor(tensor), af::SUCCESS);
  };
  fill_tensor(graph.FindNode("input")->outputs[0], 1, ge::DT_FLOAT, true, false);
  const size_t index_output = with_value ? 1U : 0U;
  const int64_t index_id = with_value ? 5 : 3;
  if (with_value) {
    fill_tensor(reduce_node->outputs[0], 3, ge::DT_FLOAT, false, false);
  }
  fill_tensor(reduce_node->outputs[index_output], index_id, index_dtype, false, false);
  for (auto output : reduce_node->outputs()) {
    auto *tensor = tpipe.GetTensor(output->attr.mem.tensor_id);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->is_ub_scalar, scalar_output);
    ASSERT_EQ(kernel.ParseOptimizeInfo(reduce_node, *output), af::SUCCESS);
    EXPECT_FALSE(tensor->is_ub_scalar);
  }
  fill_tensor(graph.FindNode("index_store")->outputs[0], 10, index_dtype, false, true);
  if (with_value || position == ConsumerPosition::kMixed) {
    fill_tensor(graph.FindNode("other_store")->outputs[0], 11, with_value ? ge::DT_FLOAT : index_dtype, false, true);
  }
  ASSERT_FALSE(testing::Test::HasFatalFailure());

  std::unique_ptr<ApiCall> reduce_call;
  if (with_value) {
    reduce_call = std::make_unique<RegArgmaxWithValueApiCall>("ArgmaxWithValue");
  } else {
    reduce_call = std::make_unique<RegArgmaxApiCall>("Argmax");
  }
  reduce_node->attr.api.unit = af::ComputeUnit::kUnitVector;
  reduce_node->attr.api.compute_type = af::ComputeType::kComputeReduce;
  ASSERT_EQ(reduce_call->Init(reduce_node), af::SUCCESS);
  reduce_call->axis = r.id;
  reduce_call->depth = 3;
  reduce_call->tmp_buf_id = {{0, 20}, {1, 21}, {2, 22}, {3, 23}};
  ApiTensor input_api;
  input_api.id = 1;
  input_api.reads = {reduce_call.get()};
  reduce_call->inputs = {&input_api};

  Loop a_loop(a.id), block_loop(r_block.id), r_loop(r.id), inner_loop(inner.id);
  a_loop.AddLoop(&block_loop);
  block_loop.AddLoop(&r_loop);
  r_loop.AddCall(reduce_call.get());
  if (position == ConsumerPosition::kInner) {
    r_loop.AddLoop(&inner_loop);
  }
  Loop *index_loop = position == ConsumerPosition::kSame    ? &r_loop
                     : position == ConsumerPosition::kInner ? &inner_loop
                                                            : &block_loop;
  Loop *other_loop = position == ConsumerPosition::kMixed ? &r_loop : index_loop;
  StoreRegApiCall index_call("DataCopyPadExtend"), other_call("DataCopyPadExtend");
  // Model the terminal Output readers too, as in the normal Store -> Output dependency chain.
  ApiCall index_output_call("Output"), other_output_call("Output");
  auto connect_store = [&](StoreRegApiCall &store, const std::string &name, Loop *loop, ApiTensor &source,
                           ApiCall &output) {
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
  };
  // Add same/inner readers before outer readers, matching execution order for lifetime handling.
  if (with_value || position == ConsumerPosition::kMixed) {
    connect_store(other_call, "other_store", other_loop, reduce_call->outputs[0], other_output_call);
  }
  connect_store(index_call, "index_store", index_loop, reduce_call->outputs[index_output], index_output_call);
  ASSERT_FALSE(testing::Test::HasFatalFailure());

  std::string source;
  ASSERT_EQ(a_loop.Generate(tiler, tpipe, source), af::SUCCESS);
  SCOPED_TRACE(source);
  const auto a_for = source.find("for (int a = 0; a < a_loop_size; a++) {");
  const auto r_for = source.find("for (int r = 0; r < r_loop_size; r++) {");
  ASSERT_NE(a_for, std::string::npos);
  ASSERT_NE(r_for, std::string::npos);
  const auto a_end = ClosingBrace(source, source.find('{', a_for));
  const auto r_end = ClosingBrace(source, source.find('{', r_for));
  ASSERT_NE(a_end, std::string::npos);
  ASSERT_NE(r_end, std::string::npos);
  EXPECT_LT(a_for, r_for);
  EXPECT_LT(r_end, a_end);
  EXPECT_EQ(source.find("for (int r =", r_for + 1), std::string::npos);

  // A partial final core has fewer R tiles; both the loop and final-round guard must use this runtime extent.
  const auto actual = source.find("r_actual_size = r_block < r_block_axis_size ? r_axis_size : r_tail_size;");
  const auto extent = source.find("r_loop_size = r_actual_size;");
  ASSERT_NE(actual, std::string::npos);
  ASSERT_NE(extent, std::string::npos);
  EXPECT_LT(a_for, actual);
  EXPECT_LT(actual, extent);
  EXPECT_LT(extent, r_for);
  EXPECT_NE(source.find("r_inner_actual_size = r_tile < r_tile_axis_size ? r_inner_axis_size : r_inner_tail_size;"),
            std::string::npos);

  const auto reset = source.find("if (r == 0) {\n  accumulated_offset = 0;\n}");
  EXPECT_NE(source.find("static int64_t accumulated_offset = 0;"), std::string::npos);
  const auto reduce = source.find("ArgMaxWithValueExtend<float, AscendC::Pattern::Reduce::AR>");
  const auto first_copy = source.find("ArgmaxCopy(tmp_argmax_index_saved[0], tmp_argmax_index[0], local_" +
                                      std::to_string(index_id) + "_actual_size);");
  const auto merge = source.find("ArgmaxMergeRound<float>");
  const auto increment = source.find("accumulated_offset += r_inner_actual_size;");
  for (const auto pos : {reset, reduce, first_copy, merge, increment}) {
    ASSERT_NE(pos, std::string::npos);
    EXPECT_LT(r_for, pos);
    EXPECT_LT(pos, r_end);
  }
  EXPECT_LT(reset, reduce);
  EXPECT_LT(reduce, first_copy);
  EXPECT_LT(first_copy, merge);
  EXPECT_NE(source.find("tmp_reduce_shape, false, accumulated_offset, r_inner_actual_size);"), std::string::npos);
  if (with_value) {
    const auto value_copy = source.find("ArgmaxCopy(local_3[0], tmp_argmax_value[0], local_3_actual_size);");
    ASSERT_NE(value_copy, std::string::npos);
    EXPECT_LT(first_copy, value_copy);
    EXPECT_LT(value_copy, merge);
    EXPECT_NE(source.find("ArgmaxMergeRound<float>(local_3[0], tmp_argmax_index_saved[0],"), std::string::npos);
  }

  const std::string local_index = "local_" + std::to_string(index_id);
  const bool cast = index_dtype == ge::DT_INT64;
  const auto materialize =
      source.find(cast ? "CastExtend<int32_t, int64_t, 1>(" + local_index + "[0], tmp_argmax_index_saved[0],"
                       : "ArgmaxCopy(" + local_index + "[0], tmp_argmax_index_saved[0],");
  ASSERT_NE(materialize, std::string::npos);
  EXPECT_LT(merge, materialize);
  EXPECT_LT(materialize, increment);
  EXPECT_EQ(source.find("CastExtend<int32_t, int64_t, 1>"), cast ? materialize : std::string::npos);
  if (cast) {
    EXPECT_EQ(source.find("CastExtend<int32_t, int64_t, 1>", materialize + 1), std::string::npos);
  } else {
    EXPECT_EQ(source.find("ArgmaxCopy(" + local_index + "[0],", materialize + 1), std::string::npos);
  }
  const auto guard = source.find("if (r + 1 == r_loop_size) {");
  if (deferred) {
    ASSERT_NE(guard, std::string::npos);
    const auto guard_end = ClosingBrace(source, source.find('{', guard));
    ASSERT_NE(guard_end, std::string::npos);
    EXPECT_LT(merge, guard);
    EXPECT_LT(guard, materialize);
    EXPECT_LT(materialize, guard_end);
    EXPECT_LT(guard_end, increment);
    EXPECT_EQ(source.find("if (r + 1 ==", guard + 1), std::string::npos);
  } else {
    EXPECT_EQ(source.find("if (r + 1 =="), std::string::npos);
  }

  const auto consumer = source.find("(global_10[");
  ASSERT_NE(consumer, std::string::npos);
  EXPECT_LT(materialize, consumer);
  EXPECT_LT(consumer, a_end);
  if (position == ConsumerPosition::kOuter || position == ConsumerPosition::kMixed) {
    EXPECT_LT(r_end, consumer);
  } else {
    EXPECT_LT(consumer, r_end);
  }
  if (position == ConsumerPosition::kInner) {
    const auto consumer_for = source.find("for (int consumer_inner =");
    ASSERT_NE(consumer_for, std::string::npos);
    EXPECT_LT(materialize, consumer_for);
    EXPECT_LT(consumer_for, consumer);
    EXPECT_LT(consumer, ClosingBrace(source, source.find('{', consumer_for)));
  }
  if (with_value || position == ConsumerPosition::kMixed) {
    const auto other_consumer = source.find("(global_11[");
    ASSERT_NE(other_consumer, std::string::npos);
    EXPECT_LT(materialize, other_consumer);
    EXPECT_LT(other_consumer, a_end);
    if (other_loop == &block_loop) {
      EXPECT_LT(r_end, other_consumer);
    } else {
      EXPECT_LT(other_consumer, r_end);
    }
  }
  EXPECT_EQ(std::count(source.begin(), source.end(), '{'), std::count(source.begin(), source.end(), '}'));
  // Scalar consumers may have their own first-round guard; the reduction must not be inside one.
  for (const auto *first_round : {"if (r < 1)", "if (r < 2)"}) {
    const auto first_round_pos = source.find(first_round);
    if (first_round_pos != std::string::npos) {
      EXPECT_GT(first_round_pos, increment);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(OrdinaryAndWithValue, ArgmaxMultiReduceLoopTest,
                         testing::Combine(testing::Bool(), testing::Values(ge::DT_INT64, ge::DT_INT32),
                                          testing::Values(ConsumerPosition::kOuter, ConsumerPosition::kSame,
                                                          ConsumerPosition::kInner, ConsumerPosition::kMixed),
                                          testing::Bool()));
}  // namespace
