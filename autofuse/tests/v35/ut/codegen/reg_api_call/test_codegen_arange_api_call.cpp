/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <string>

#include "gtest/gtest.h"

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "codegen_kernel.h"
#include "graph/utils/graph_utils.h"
#include "platform_context.h"
#include "runtime_stub.h"
#include "utils/api_call_factory.h"
#include "v35/ascir/generator/v2_ascir_codegen_impl.h"

namespace codegen {
namespace {

class ArangeApiCallTest : public testing::Test {
 protected:
  ArangeApiCallTest() {
    ge::PlatformContext::GetInstance().Reset();
    ge::PlatformContext::GetInstance().SetPlatform("3510");
    ge::RuntimeStub::SetInstance(std::make_shared<ge::RuntimeStubV2Common>());
  }

  ~ArangeApiCallTest() override {
    ge::RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
  }
};

struct ArangeGraph {
  af::AscGraph graph{"arange_graph"};
  af::AscNodePtr node;
  Tiler tiler;
  TPipe tpipe{"tpipe", tiler};
  ascir::AxisId axis_id{af::kIdNone};

  ArangeGraph(const af::Expression &base, const af::Expression &step, ge::DataType dtype,
              bool nonvectorized_outer_axis = false, bool vectorized_outer_axis = false,
              const af::Expression &vec_stride = af::ops::One) {
    const auto size = graph.CreateSizeVar("size");
    const auto axis = graph.CreateAxis("axis", size);
    axis_id = axis.id;
    af::ascir_op::Arange op("arange");
    graph.AddNode(op);
    op.ir_attr.SetBase(base);
    op.ir_attr.SetStep(step);
    *op.y.axis = {axis.id};
    *op.y.repeats = {size};
    *op.y.strides = {af::ops::One};
    if (nonvectorized_outer_axis || vectorized_outer_axis) {
      const auto outer_axis = graph.CreateAxis("outer_axis", af::Symbol(2));
      axis_id = outer_axis.id;
      *op.y.axis = {outer_axis.id, axis.id};
      *op.y.repeats = {af::Symbol(2), size};
      *op.y.strides = {size, af::ops::One};
      (void)tiler.AddAxis(outer_axis);
    }
    node = graph.FindNode("arange");
    if (node == nullptr) {
      return;
    }
    auto &out = node->outputs[0].attr;
    out.dtype = dtype;
    out.vectorized_axis = {axis.id};
    out.vectorized_strides = {vec_stride};
    if (vectorized_outer_axis) {
      out.vectorized_axis = *op.y.axis;
      out.vectorized_strides = *op.y.strides;
    }
    out.mem.tensor_id = 0;
    out.mem.position = af::Position::kPositionVecOut;
    out.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
    out.buf.id = 1;
    out.opt.merge_scope = af::kIdNone;
    node->attr.api.compute_type = af::ComputeType::kComputeElewise;
    node->attr.api.type = af::ApiType::kAPITypeCompute;
    node->attr.api.unit = af::ComputeUnit::kUnitVector;
    (void)tpipe.AddTensor(node->outputs[0]);
    (void)tiler.AddAxis(axis);
    tiler.AddSizeVar(af::SizeVar(size));
  }
};

std::unique_ptr<ApiCall> MakeInitedArangeCall(const ArangeGraph &g) {
  std::unique_ptr<ApiCall> call(CreateApiCallObject(g.node));
  if (call == nullptr) {
    return nullptr;
  }
  if (call->Init(g.node) != af::SUCCESS) {
    return nullptr;
  }
  return call;
}
}  // namespace

TEST_F(ArangeApiCallTest, FactoryCreatesRegisteredOrdinaryApiCall) {
  ArangeGraph g(af::Expression::Parse("base"), af::Expression::Parse("step"), ge::DT_INT32);
  ASSERT_NE(g.node, nullptr);
  std::unique_ptr<ApiCall> call(CreateApiCallObject(g.node));
  ASSERT_NE(call, nullptr);
  EXPECT_EQ(call->api_name_, "Arange");
  EXPECT_EQ(call->Init(g.node), af::SUCCESS);
}

TEST_F(ArangeApiCallTest, CodegenImplUsesArangeApiCall) {
  af::ascir::ArangeAscIrCodegenImplV2 impl;
  EXPECT_EQ(impl.GetApiCallName(), "ArangeApiCall");
  EXPECT_EQ(impl.GetApiName(), "Arange");
}

TEST_F(ArangeApiCallTest, GeneratesInt32LoopWithDynamicBaseAndStep) {
  ArangeGraph g(af::Expression::Parse("base"), af::Expression::Parse("step"), ge::DT_INT32);
  const auto *output = g.tpipe.GetTensor(g.node->outputs[0].attr.mem.tensor_id);
  ASSERT_NE(output, nullptr);
  auto call = MakeInitedArangeCall(g);
  ASSERT_NE(call, nullptr);

  std::string result;
  ASSERT_EQ(call->Generate(g.tpipe, {}, result), af::SUCCESS);
  RecordProperty("generated_code", result);
  EXPECT_NE(result.find("for (int64_t arange_i = 0; arange_i < " + output->actual_size.Str() + "; ++arange_i) {"),
            std::string::npos);
  EXPECT_NE(
      result.find(
          output->Str() +
          ".SetValue(static_cast<uint32_t>(0 + arange_i), static_cast<int32_t>((base) + (0 + arange_i) * (step)));"),
      std::string::npos);
}

// 对齐策略标记为 kDiscontinuous（如 concat 拆列场景 int64 32B 对齐步长=4）时，
// ArangeApiCall 应按 vectorized_strides 跳格物化，而非要求连续布局。
TEST_F(ArangeApiCallTest, GeneratesDiscontinuousVectorizedStrideLoop) {
  ArangeGraph g(af::Expression::Parse("0"), af::Expression::Parse("12"), ge::DT_INT64, false, false, af::Symbol(4));
  ASSERT_NE(g.node, nullptr);
  auto call = MakeInitedArangeCall(g);
  ASSERT_NE(call, nullptr);

  std::string result;
  ASSERT_EQ(call->Generate(g.tpipe, {}, result), af::SUCCESS);
  RecordProperty("generated_code", result);
  // 跳格物化：写地址按物理步长 4 推进
  EXPECT_NE(
      result.find(
          ".SetValue(static_cast<uint32_t>(0 + arange_i * (4)), static_cast<int64_t>((0) + (0 + arange_i) * (12)));"),
      std::string::npos);
  // 循环上界必须是有效元素数（size），而非物理跨度 (size-1)*4+1
  EXPECT_NE(result.find("for (int64_t arange_i = 0; arange_i < size; ++arange_i) {"), std::string::npos);
  EXPECT_EQ(result.find("(size - 1) * 4 + 1"), std::string::npos);
}

// 多轴布局：最内层 kDiscontinuous（stride=4）、外层物理紧凑（stride=size）时，
// phys_stride 应取最内层步长，循环上界为两轴有效元素数乘积。
TEST_F(ArangeApiCallTest, GeneratesDiscontinuousInnerWithCompactOuterAxis) {
  ArangeGraph g(af::Expression::Parse("0"), af::Expression::Parse("1"), ge::DT_INT64, false, true);
  ASSERT_NE(g.node, nullptr);
  ASSERT_EQ(g.tpipe.tensors.size(), 1UL);
  g.tpipe.tensors.at(0).vectorized_strides = {af::Symbol("size"), af::Symbol(4)};
  auto call = MakeInitedArangeCall(g);
  ASSERT_NE(call, nullptr);

  std::string result;
  ASSERT_EQ(call->Generate(g.tpipe, {}, result), af::SUCCESS);
  RecordProperty("generated_code", result);
  EXPECT_NE(result.find("arange_i * (4)"), std::string::npos);
  EXPECT_NE(result.find("for (int64_t arange_i = 0; arange_i < (2 * size); ++arange_i) {"), std::string::npos);
}

// 多轴布局：外层带空洞（stride 非紧凑）时，物化会静默错位，必须拒绝。
TEST_F(ArangeApiCallTest, RejectsDiscontinuousOuterHoleWithCompactInner) {
  ArangeGraph g(af::Expression::Parse("0"), af::Expression::Parse("1"), ge::DT_INT64, false, true);
  ASSERT_NE(g.node, nullptr);
  ASSERT_EQ(g.tpipe.tensors.size(), 1UL);
  g.tpipe.tensors.at(0).vectorized_strides = {af::Expression::Parse("size * 4"), af::Symbol(4)};
  auto call = MakeInitedArangeCall(g);
  ASSERT_NE(call, nullptr);

  std::string result;
  EXPECT_EQ(call->Generate(g.tpipe, {}, result), af::FAILED);
  EXPECT_TRUE(result.empty());
}

TEST_F(ArangeApiCallTest, GeneratesInt64LoopWithNegativeBaseAndStep) {
  ArangeGraph g(af::Symbol(-3), af::Symbol(-2), ge::DT_INT64);
  auto call = MakeInitedArangeCall(g);
  ASSERT_NE(call, nullptr);

  std::string result;
  ASSERT_EQ(call->Generate(g.tpipe, {}, result), af::SUCCESS);
  RecordProperty("generated_code", result);
  EXPECT_NE(result.find(
                ".SetValue(static_cast<uint32_t>(0 + arange_i), static_cast<int64_t>((-3) + (0 + arange_i) * (-2)));"),
            std::string::npos);
}

TEST_F(ArangeApiCallTest, GeneratesLoopWithZeroStep) {
  ArangeGraph g(af::Symbol(5), af::Symbol(0), ge::DT_INT32);
  auto call = MakeInitedArangeCall(g);
  ASSERT_NE(call, nullptr);

  std::string result;
  ASSERT_EQ(call->Generate(g.tpipe, {}, result), af::SUCCESS);
  RecordProperty("generated_code", result);
  EXPECT_NE(
      result.find(".SetValue(static_cast<uint32_t>(0 + arange_i), static_cast<int32_t>((5) + (0 + arange_i) * (0)));"),
      std::string::npos);
}

TEST_F(ArangeApiCallTest, GeneratesLoopWithUnitStep) {
  ArangeGraph g(af::Symbol(7), af::Symbol(1), ge::DT_INT32);
  auto call = MakeInitedArangeCall(g);
  ASSERT_NE(call, nullptr);

  std::string result;
  ASSERT_EQ(call->Generate(g.tpipe, {}, result), af::SUCCESS);
  RecordProperty("generated_code", result);
  EXPECT_NE(
      result.find(".SetValue(static_cast<uint32_t>(0 + arange_i), static_cast<int32_t>((7) + (0 + arange_i) * (1)));"),
      std::string::npos);
}

TEST_F(ArangeApiCallTest, GeneratesBoundedCurrentAxisScalarWrite) {
  ArangeGraph g(af::Expression::Parse("base"), af::Expression::Parse("step"), ge::DT_INT32);
  const auto *output = g.tpipe.GetTensor(g.node->outputs[0].attr.mem.tensor_id);
  ASSERT_NE(output, nullptr);
  auto call = MakeInitedArangeCall(g);
  ASSERT_NE(call, nullptr);

  std::string result;
  ASSERT_EQ(call->Generate(g.tpipe, {g.axis_id}, result), af::SUCCESS);
  RecordProperty("generated_code", result);
  EXPECT_NE(result.find("arange_i < 1;"), std::string::npos);
  EXPECT_NE(result.find(".SetValue(static_cast<uint32_t>((int64_t)axis + arange_i), "
                        "static_cast<int32_t>((base) + ((int64_t)axis + arange_i) * (step)));"),
            std::string::npos);
}

TEST_F(ArangeApiCallTest, GeneratesBoundedVectorizedOuterAxisSlice) {
  ArangeGraph g(af::Symbol(7), af::Symbol(1), ge::DT_INT32, false, true);
  const auto *output = g.tpipe.GetTensor(g.node->outputs[0].attr.mem.tensor_id);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(output->vectorized_axis.size(), 2U);
  EXPECT_EQ(g.tpipe.tiler.TensorActualSize(*output), "(2 - 1) * t->size + (axis_actual_size - 1) + 1");
  auto call = MakeInitedArangeCall(g);
  ASSERT_NE(call, nullptr);
  std::string result;
  ASSERT_EQ(call->Generate(g.tpipe, {g.axis_id}, result), af::SUCCESS);
  RecordProperty("generated_code", result);
  EXPECT_NE(result.find("arange_i < (axis_actual_size - 1) + 1;"), std::string::npos);
  EXPECT_NE(result.find(".SetValue(static_cast<uint32_t>((int64_t)outer_axis * (int64_t)t->size + arange_i), "
                        "static_cast<int32_t>((7) + ((int64_t)outer_axis * (int64_t)t->size + arange_i) * (1)));"),
            std::string::npos);
}

TEST_F(ArangeApiCallTest, GeneratesBoundedMultipleCurrentAxesScalarWrite) {
  ArangeGraph g(af::Symbol(-3), af::Symbol(-2), ge::DT_INT64, false, true);
  const auto *output = g.tpipe.GetTensor(g.node->outputs[0].attr.mem.tensor_id);
  ASSERT_NE(output, nullptr);
  auto call = MakeInitedArangeCall(g);
  ASSERT_NE(call, nullptr);
  std::string result;
  ASSERT_EQ(call->Generate(g.tpipe, output->vectorized_axis, result), af::SUCCESS);
  RecordProperty("generated_code", result);
  EXPECT_NE(result.find("arange_i < 1;"), std::string::npos);
  EXPECT_NE(result.find(".SetValue(static_cast<uint32_t>((int64_t)outer_axis * (int64_t)t->size + "
                        "(int64_t)axis + arange_i),"),
            std::string::npos);
}

TEST_F(ArangeApiCallTest, GeneratesBroadcastOuterAxisRepeatedSlice) {
  ArangeGraph g(af::Symbol(0), af::Symbol(1), ge::DT_INT32, false, true);
  // 广播行轴: 逻辑 stride 为 0, 物化布局为连续展开(与分区器 MaterializeArangeViewContiguous 对齐)。
  g.tpipe.tensors.at(0).axis_strides = {af::ops::Zero, af::ops::One};
  auto call = MakeInitedArangeCall(g);
  ASSERT_NE(call, nullptr);
  std::string result;
  ASSERT_EQ(call->Generate(g.tpipe, {}, result), af::SUCCESS);
  RecordProperty("generated_code", result);
  // 非固定广播轴在调用内部展开: 每帧物理偏移按连续布局前进, 取值按逻辑偏移(广播轴贡献 0)重复。
  EXPECT_NE(result.find("for (int64_t arange_b0 = 0; arange_b0 < 2; ++arange_b0) {"), std::string::npos);
  EXPECT_NE(result.find("arange_i < (axis_actual_size - 1) + 1;"), std::string::npos);
  EXPECT_NE(result.find(".SetValue(static_cast<uint32_t>(0 + arange_b0 * t->size + arange_i), "
                        "static_cast<int32_t>((0) + (0 + arange_i) * (1)));"),
            std::string::npos);
}

TEST_F(ArangeApiCallTest, RejectsLogicalStrideDifferentFromPhysicalStride) {
  ArangeGraph g(af::Symbol(7), af::Symbol(1), ge::DT_INT32);
  g.tpipe.tensors.at(0).axis_strides = {af::Symbol(2)};
  auto call = MakeInitedArangeCall(g);
  ASSERT_NE(call, nullptr);
  std::string result;
  EXPECT_EQ(call->Generate(g.tpipe, {}, result), af::FAILED);
  EXPECT_TRUE(result.empty());
}

TEST_F(ArangeApiCallTest, RejectsPhysicalHolesAtZeroOffset) {
  ArangeGraph g(af::Symbol(7), af::Symbol(1), ge::DT_INT32);
  g.tpipe.tensors.at(0).vectorized_strides = {af::Symbol(2)};
  auto call = MakeInitedArangeCall(g);
  ASSERT_NE(call, nullptr);
  std::string result;
  EXPECT_EQ(call->Generate(g.tpipe, {}, result), af::FAILED);
  EXPECT_TRUE(result.empty());
}

TEST_F(ArangeApiCallTest, RejectsLoopRangeLargerThanTensorExtent) {
  ArangeGraph g(af::Symbol(7), af::Symbol(1), ge::DT_INT32);
  g.tpipe.tensors.at(0).axis_size = {af::Symbol(1)};
  auto call = MakeInitedArangeCall(g);
  ASSERT_NE(call, nullptr);
  std::string result;
  EXPECT_EQ(call->Generate(g.tpipe, {g.axis_id}, result), af::FAILED);
  EXPECT_TRUE(result.empty());
}

TEST_F(ArangeApiCallTest, RejectsUnprovenActualSpanAtZeroOffset) {
  ArangeGraph g(af::Symbol(7), af::Symbol(1), ge::DT_INT32);
  g.tpipe.tensors.at(0).axis_size = {af::Symbol(1)};
  g.tiler.axis_map.at(g.axis_id).type = ascir::Axis::Type::kAxisTypeTileInner;
  auto call = MakeInitedArangeCall(g);
  ASSERT_NE(call, nullptr);
  std::string result;
  EXPECT_EQ(call->Generate(g.tpipe, {}, result), af::FAILED);
  EXPECT_TRUE(result.empty());
}

TEST_F(ArangeApiCallTest, NonvectorizedOuterAxisOnlyContributesToLogicalOffset) {
  ArangeGraph g(af::Expression::Parse("base"), af::Expression::Parse("step"), ge::DT_INT32, true);
  const auto *output = g.tpipe.GetTensor(g.node->outputs[0].attr.mem.tensor_id);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(output->axis.size(), 2U);
  ASSERT_EQ(output->vectorized_axis.size(), 1U);
  ASSERT_NE(output->vectorized_axis.front(), g.axis_id);
  auto call = MakeInitedArangeCall(g);
  ASSERT_NE(call, nullptr);

  std::string result;
  ASSERT_EQ(call->Generate(g.tpipe, {g.axis_id}, result), af::SUCCESS);
  RecordProperty("generated_code", result);
  EXPECT_NE(result.find(".SetValue(static_cast<uint32_t>(0 + arange_i), static_cast<int32_t>((base) + "
                        "((int64_t)outer_axis * (int64_t)t->size + arange_i) * (step)));"),
            std::string::npos);
}

TEST_F(ArangeApiCallTest, KernelGenerateOrdinaryFallbackWithLoopAndStore) {
  for (const bool full_local_view : {false, true}) {
    SCOPED_TRACE(full_local_view);
    ArangeGraph g(af::Symbol(7), af::Symbol(-2), ge::DT_INT32, !full_local_view, full_local_view);
    ASSERT_NE(g.node, nullptr);
    g.node->attr.sched.axis = g.node->outputs[0].attr.axis;
    g.node->attr.sched.loop_axis = g.axis_id;

    af::ascir_op::Store store("store");
    af::ascir_op::Output output("output");
    g.graph.AddNode(store);
    g.graph.AddNode(output);
    ASSERT_EQ(af::GraphUtils::AddEdge(g.node->GetOutDataAnchor(0), g.graph.FindNode("store")->GetInDataAnchor(0)),
              af::GRAPH_SUCCESS);
    output.x = store.y;
    output.ir_attr.SetIndex(0);
    const auto store_node = g.graph.FindNode("store");
    const auto output_node = g.graph.FindNode("output");
    ASSERT_NE(store_node, nullptr);
    ASSERT_NE(output_node, nullptr);
    store_node->outputs[0].attr = g.node->outputs[0].attr;
    store_node->outputs[0].attr.mem.tensor_id = 1;
    store_node->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
    store_node->outputs[0].attr.mem.position = af::Position::kPositionGM;
    store_node->attr.api.unit = af::ComputeUnit::kUnitMTE3;
    store_node->attr.api.type = af::ApiType::kAPITypeCompute;
    store_node->attr.api.compute_type = af::ComputeType::kComputeStore;
    store_node->attr.sched.axis = g.node->attr.sched.axis;
    store_node->attr.sched.loop_axis = full_local_view ? af::kIdNone : g.axis_id;

    ascir::FusedScheduledResult fused;
    fused.output_nodes.push_back(output_node);
    Kernel kernel(g.graph.GetName());
    ASSERT_EQ(Kernel::ParseGraph(g.graph, fused, kernel), af::SUCCESS);
    std::string result;
    ASSERT_EQ(kernel.Generate(g.graph.GetName(), "AutofuseTilingData", result, g.graph), af::SUCCESS);
    RecordProperty(full_local_view ? "full_view_kernel" : "row_view_kernel", result);
    EXPECT_EQ(result.find("VFCall"), std::string::npos);
    EXPECT_EQ(result.find("Reg::Arange"), std::string::npos);
    const auto loop_pos = result.find("for (int outer_axis = 0;");
    const auto alloc_pos = result.find("LocalTensor<int32_t> local_0 = b1.Get<int32_t>();");
    const auto actual_size_pos = result.find("const uint32_t local_0_actual_size =");
    const auto write_pos = result.find("local_0.SetValue(");
    const auto store_pos = result.find("DataCopy");
    ASSERT_NE(loop_pos, std::string::npos) << result;
    ASSERT_NE(alloc_pos, std::string::npos) << result;
    ASSERT_NE(actual_size_pos, std::string::npos) << result;
    ASSERT_NE(write_pos, std::string::npos) << result;
    ASSERT_NE(store_pos, std::string::npos) << result;
    EXPECT_LT(alloc_pos, write_pos);
    EXPECT_LT(actual_size_pos, write_pos);
    EXPECT_LT(loop_pos, write_pos);
    EXPECT_LT(write_pos, store_pos);
    EXPECT_NE(result.find("((int64_t)outer_axis * (int64_t)t->size + arange_i) * (-2)"), std::string::npos);
    if (full_local_view) {
      EXPECT_LT(actual_size_pos, loop_pos);
      EXPECT_NE(result.find("arange_i < (axis_actual_size - 1) + 1;"), std::string::npos);
      EXPECT_NE(result.find("static_cast<uint32_t>((int64_t)outer_axis * (int64_t)t->size + arange_i)"),
                std::string::npos);
      EXPECT_NE(result.find("KernelUtils::BlkAlign<int32_t>((2 - 1) * t->size + (t->size - 1) + 1)"),
                std::string::npos);
    } else {
      EXPECT_LT(loop_pos, actual_size_pos);
      EXPECT_NE(result.find("arange_i < local_0_actual_size;"), std::string::npos);
      EXPECT_NE(result.find("static_cast<uint32_t>(0 + arange_i)"), std::string::npos);
      EXPECT_NE(result.find("KernelUtils::BlkAlign<int32_t>((t->size - 1) + 1)"), std::string::npos);
    }
  }
}
}  // namespace codegen
