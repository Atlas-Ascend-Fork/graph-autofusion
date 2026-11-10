/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "schedule_graph_validator.h"

namespace {

TEST(ScheduleGraphValidator, ReportsTensorMetadataBeforeSchedule) {
  af::AscGraph graph("invalid_tensor_metadata");
  const auto size = graph.CreateSizeVar("size");
  const auto axis = graph.CreateAxis("axis", size);

  af::ascir_op::Abs abs("abs");
  abs.attr.sched.axis = {axis.id};
  abs.y.dtype = af::DT_FLOAT16;
  *abs.y.axis = {axis.id};
  *abs.y.repeats = {size};
  *abs.y.strides = {};
  ASSERT_NE(graph.AddNode(abs), nullptr);

  EXPECT_EQ(optimize::ValidateScheduleGraph(graph), af::FAILED);
  EXPECT_NE(optimize::GetScheduleValidationError().find("[AF][SCHEDULE GRAPH VALIDATION ERROR][TENSOR_METADATA]"),
            std::string::npos);
  EXPECT_NE(optimize::GetScheduleValidationError().find("node=abs"), std::string::npos);
}

TEST(ScheduleGraphValidator, ValidatesVectorizedMetadataBeforeStrideInference) {
  af::AscGraph graph("vectorized_stride_inference");
  const auto a = graph.CreateAxis("a", af::Symbol(4));
  const auto b = graph.CreateAxis("b", af::Symbol(8));
  const auto other = graph.CreateAxis("other", af::Symbol(2));
  af::ascir_op::Data data("data", graph);
  data.attr.api.type = af::ApiType::kAPITypeBuffer;
  af::ascir_op::Load load("load");
  load.x = data.y;
  load.attr.api.type = af::ApiType::kAPITypeCompute;
  load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load.attr.sched.axis = {a.id, b.id};
  *load.y.axis = {a.id, b.id};
  *load.y.repeats = {af::Symbol(4), af::Symbol(8)};
  *load.y.strides = {af::Symbol(8), af::ops::One};
  *load.y.vectorized_axis = {a.id, b.id};

  // Hint 图的向量化 stride 可以等待对齐阶段推导，校验器不能提前拒绝或改写它。
  EXPECT_EQ(optimize::ValidateScheduleGraph(graph), af::SUCCESS);
  EXPECT_TRUE(load.y.vectorized_strides->empty());
  *load.y.vectorized_strides = {af::Symbol(8), af::ops::One};
  EXPECT_EQ(optimize::ValidateScheduleGraph(graph), af::SUCCESS);

  // 已填写但长度不匹配的 stride 仍然是错误，不能当作待推导状态。
  *load.y.vectorized_strides = {af::ops::One};
  EXPECT_EQ(optimize::ValidateScheduleGraph(graph), af::FAILED);
  EXPECT_NE(optimize::GetScheduleValidationError().find("[TENSOR_METADATA]"), std::string::npos);
  *load.y.vectorized_axis = {};
  EXPECT_EQ(optimize::ValidateScheduleGraph(graph), af::FAILED);

  // stride 尚未填写也不能引用不属于该 Tensor 的向量化轴。
  *load.y.vectorized_axis = {other.id};
  *load.y.vectorized_strides = {};
  EXPECT_EQ(optimize::ValidateScheduleGraph(graph), af::FAILED);
  EXPECT_NE(optimize::GetScheduleValidationError().find("[AXIS_SET]"), std::string::npos);
}

TEST(ScheduleGraphValidator, AllowsCubeReductionAxesOutsideOutputView) {
  af::AscGraph graph("matmul_schedule_axes");
  const auto m = graph.CreateAxis("m", af::Symbol(4));
  const auto n = graph.CreateAxis("n", af::Symbol(8));
  const auto k = graph.CreateAxis("k", af::Symbol(16));
  af::ascir_op::Data lhs("lhs", graph);
  lhs.attr.api.type = af::ApiType::kAPITypeBuffer;
  *lhs.y.axis = {m.id, k.id};
  *lhs.y.repeats = {af::Symbol(4), af::Symbol(16)};
  *lhs.y.strides = {af::Symbol(16), af::ops::One};
  af::ascir_op::Data rhs("rhs", graph);
  rhs.attr.api.type = af::ApiType::kAPITypeBuffer;
  *rhs.y.axis = {k.id, n.id};
  *rhs.y.repeats = {af::Symbol(16), af::Symbol(8)};
  *rhs.y.strides = {af::Symbol(8), af::ops::One};
  af::ascir_op::MatMul matmul("matmul");
  matmul.x1 = lhs.y;
  matmul.x2 = rhs.y;
  matmul.attr.api.type = af::ApiType::kAPITypeCompute;
  matmul.attr.api.compute_type = af::ComputeType::kComputeCube;
  matmul.attr.sched.axis = {m.id, n.id, k.id};
  *matmul.y.axis = {m.id, n.id};
  *matmul.y.repeats = {af::Symbol(4), af::Symbol(8)};
  *matmul.y.strides = {af::Symbol(8), af::ops::One};
  EXPECT_EQ(optimize::ValidateScheduleGraph(graph), af::SUCCESS);

  // Cube 仍须遵守图轴引用、调度轴唯一性和输出元数据完整性约束。
  matmul.attr.sched.axis = {m.id, n.id, af::kIdNone};
  EXPECT_EQ(optimize::ValidateScheduleGraph(graph), af::FAILED);
  EXPECT_NE(optimize::GetScheduleValidationError().find("[AXIS_SET]"), std::string::npos);
  matmul.attr.sched.axis = {m.id, n.id, k.id, k.id};
  EXPECT_EQ(optimize::ValidateScheduleGraph(graph), af::FAILED);
  EXPECT_NE(optimize::GetScheduleValidationError().find("duplicate axis id"), std::string::npos);
  matmul.attr.sched.axis = {m.id, n.id, k.id};
  *matmul.y.repeats = {};
  EXPECT_EQ(optimize::ValidateScheduleGraph(graph), af::FAILED);
  EXPECT_NE(optimize::GetScheduleValidationError().find("[TENSOR_METADATA]"), std::string::npos);
}

TEST(ScheduleGraphValidator, ReportsTransposeWithoutRealInputAnchor) {
  af::AscGraph graph("invalid_transpose_input");
  const auto size = graph.CreateSizeVar("size");
  const auto axis = graph.CreateAxis("axis", size);

  af::ascir_op::Transpose transpose("transpose");
  transpose.attr.api.compute_type = af::ComputeType::kComputeTranspose;
  transpose.attr.api.type = af::ApiType::kAPITypeCompute;
  transpose.attr.sched.axis = {axis.id};
  transpose.y.dtype = af::DT_FLOAT16;
  *transpose.y.axis = {axis.id};
  *transpose.y.repeats = {size};
  *transpose.y.strides = {af::ops::One};
  ASSERT_NE(graph.AddNode(transpose), nullptr);

  EXPECT_EQ(optimize::ValidateScheduleGraph(graph), af::FAILED);
  EXPECT_NE(optimize::GetScheduleValidationError().find("[AF][SCHEDULE GRAPH VALIDATION ERROR][TRANSPOSE_AXIS]"),
            std::string::npos);
  EXPECT_NE(optimize::GetScheduleValidationError().find("input[0]"), std::string::npos);
}

TEST(ScheduleGraphValidator, AllowsBufferInputsWithDeferredViewMetadata) {
  af::AscGraph graph("buffer_input_views");
  const auto axis = graph.CreateAxis("a", af::Symbol(4));
  af::ascir_op::Data data("data", graph);
  data.attr.api.type = af::ApiType::kAPITypeBuffer;
  *data.y.axis = {axis.id};
  af::ascir_op::Load load("load");
  load.x = data.y;
  load.attr.api.type = af::ApiType::kAPITypeCompute;
  load.attr.sched.axis = {axis.id};
  *load.y.axis = {axis.id};
  *load.y.repeats = {af::Symbol(4)};
  *load.y.strides = {af::ops::One};

  af::ascir_op::Scalar scalar("scalar", graph);
  scalar.attr.api.type = af::ApiType::kAPITypeBuffer;
  scalar.ir_attr.SetValue("0");
  *scalar.y.axis = {axis.id};
  af::ascir_op::Broadcast broadcast("broadcast");
  broadcast.x = scalar.y;
  broadcast.attr.api.type = af::ApiType::kAPITypeCompute;
  broadcast.attr.sched.axis = {axis.id};
  *broadcast.y.axis = {axis.id};
  *broadcast.y.repeats = {af::Symbol(4)};
  *broadcast.y.strides = {af::ops::One};

  EXPECT_EQ(optimize::ValidateScheduleGraph(graph), af::SUCCESS);
  // Buffer 输入的延迟补齐不能放宽 Load 等计算节点的输出契约。
  *load.y.repeats = {};
  EXPECT_EQ(optimize::ValidateScheduleGraph(graph), af::FAILED);
  EXPECT_NE(optimize::GetScheduleValidationError().find("[AF][SCHEDULE GRAPH VALIDATION ERROR][TENSOR_METADATA]"),
            std::string::npos);
}

TEST(ScheduleGraphValidator, AllowsReduceBeforeScheduleAxesAreFilled) {
  af::AscGraph graph("reduce_without_schedule_axes");
  const auto size = graph.CreateSizeVar("size");
  const auto axis = graph.CreateAxis("axis", size);

  af::ascir_op::Data data("data", graph);
  data.attr.api.type = af::ApiType::kAPITypeBuffer;
  data.y.dtype = ge::DT_FLOAT16;
  *data.y.axis = {axis.id};
  *data.y.repeats = {size};
  *data.y.strides = {af::ops::One};
  af::ascir_op::Max reduce("reduce");
  reduce.x = data.y;
  reduce.attr.api.compute_type = af::ComputeType::kComputeReduce;
  reduce.attr.api.type = af::ApiType::kAPITypeCompute;
  reduce.y.dtype = ge::DT_FLOAT16;
  *reduce.y.axis = {axis.id};
  *reduce.y.repeats = {af::ops::One};
  *reduce.y.strides = {af::ops::Zero};
  const auto node = graph.FindNode("reduce");
  ASSERT_NE(node, nullptr);

  EXPECT_EQ(optimize::ValidateScheduleGraph(graph), af::SUCCESS);
  EXPECT_TRUE(optimize::GetScheduleValidationError().empty());
  // 即使调度轴尚未回填，非法的归约输出长度也必须被检出。
  *reduce.y.repeats = {af::Symbol(2)};
  EXPECT_EQ(optimize::ValidateScheduleGraph(graph), af::FAILED);
  EXPECT_NE(optimize::GetScheduleValidationError().find("expected=1"), std::string::npos);
  // 回填调度轴后使用相同的形状规则，不需要等待第二个校验入口才能发现错误。
  node->attr.sched.axis = {axis.id};
  EXPECT_EQ(optimize::ValidateScheduleGraph(graph), af::FAILED);
  *reduce.y.repeats = {af::ops::One};
  EXPECT_EQ(optimize::ValidateScheduleGraph(graph), af::SUCCESS);
}

TEST(ScheduleGraphValidator, ChecksReduceInputOutputAxisCounts) {
  // 覆盖轴减少、轴增加，以及轴数量不变但归约轴长度变成 1 的合法情况。
  for (const size_t input_rank : {1U, 2U}) {
    for (const size_t output_rank : {1U, 2U}) {
      SCOPED_TRACE("input_rank=" + std::to_string(input_rank) + " output_rank=" + std::to_string(output_rank));
      af::AscGraph graph("reduce_axis_counts");
      const auto size = graph.CreateSizeVar("size");
      const auto a = graph.CreateAxis("a", size);
      const auto r = graph.CreateAxis("r", size);
      const std::vector<af::AxisId> all_axes = {a.id, r.id};
      const std::vector<af::AxisId> input_axes(all_axes.begin(), all_axes.begin() + input_rank);
      const std::vector<af::AxisId> output_axes(all_axes.begin(), all_axes.begin() + output_rank);

      af::ascir_op::Data data("data", graph);
      data.attr.api.type = af::ApiType::kAPITypeBuffer;
      data.y.dtype = af::DT_FLOAT16;
      *data.y.axis = input_axes;
      *data.y.repeats = std::vector<af::Expression>(input_rank, size);
      *data.y.strides = std::vector<af::Expression>(input_rank, af::ops::One);

      af::ascir_op::Max reduce("reduce");
      reduce.x = data.y;
      reduce.attr.api.compute_type = af::ComputeType::kComputeReduce;
      reduce.attr.api.type = af::ApiType::kAPITypeCompute;
      reduce.attr.sched.axis = output_axes;
      reduce.y.dtype = af::DT_FLOAT16;
      *reduce.y.axis = output_axes;
      *reduce.y.repeats = std::vector<af::Expression>(output_rank, af::ops::One);
      *reduce.y.strides = std::vector<af::Expression>(output_rank, af::ops::Zero);
      ASSERT_NE(graph.FindNode("reduce"), nullptr);

      const auto status = optimize::ValidateScheduleGraph(graph);
      if (input_rank == output_rank) {
        EXPECT_EQ(status, af::SUCCESS);
        EXPECT_TRUE(optimize::GetScheduleValidationError().empty());
      } else {
        EXPECT_EQ(status, af::FAILED);
        const auto &diagnostic = optimize::GetScheduleValidationError();
        EXPECT_NE(diagnostic.find("[AF][SCHEDULE GRAPH VALIDATION ERROR][REDUCE_AXIS]"), std::string::npos);
        EXPECT_NE(diagnostic.find("node=reduce"), std::string::npos);
        EXPECT_NE(diagnostic.find("input[0] axes=" + std::to_string(input_rank)), std::string::npos);
        EXPECT_NE(diagnostic.find("output[0] axes=" + std::to_string(output_rank)), std::string::npos);
      }
    }
  }
}

TEST(ScheduleGraphValidator, ChecksReducedOutputRepeatIsOne) {
  // 第一轴保持长度 4，第二轴执行普通归约或长度为 1 的退化归约。
  for (const int64_t input_r : {1, 8}) {
    for (const int64_t output_r : {1, 2, 8}) {
      SCOPED_TRACE("input_r=" + std::to_string(input_r) + " output_r=" + std::to_string(output_r));
      af::AscGraph graph("reduce_output_repeat");
      const auto a = graph.CreateAxis("a", af::Symbol(4));
      const auto r = graph.CreateAxis("r", af::Symbol(input_r));
      const std::vector<af::AxisId> axes = {a.id, r.id};

      af::ascir_op::Data data("data", graph);
      data.attr.api.type = af::ApiType::kAPITypeBuffer;
      data.y.dtype = af::DT_FLOAT16;
      *data.y.axis = axes;
      *data.y.repeats = {af::Symbol(4), af::Symbol(input_r)};
      // 退化轴的输入 stride 也为零，覆盖 input repeat=1 且 output stride=0 的候选轴。
      *data.y.strides = {af::Symbol(input_r), input_r == 1 ? af::ops::Zero : af::ops::One};

      af::ascir_op::Max reduce("reduce");
      reduce.x = data.y;
      reduce.attr.api.compute_type = af::ComputeType::kComputeReduce;
      reduce.attr.api.type = af::ApiType::kAPITypeCompute;
      reduce.attr.sched.axis = axes;
      reduce.y.dtype = af::DT_FLOAT16;
      *reduce.y.axis = axes;
      *reduce.y.repeats = {af::Symbol(4), af::Symbol(output_r)};
      *reduce.y.strides = {af::ops::One, af::ops::Zero};
      ASSERT_NE(graph.FindNode("reduce"), nullptr);

      const auto status = optimize::ValidateScheduleGraph(graph);
      if (output_r == 1) {
        EXPECT_EQ(status, af::SUCCESS);
        EXPECT_TRUE(optimize::GetScheduleValidationError().empty());
      } else {
        EXPECT_EQ(status, af::FAILED);
        const auto &diagnostic = optimize::GetScheduleValidationError();
        EXPECT_NE(diagnostic.find("[AF][SCHEDULE GRAPH VALIDATION ERROR][REDUCE_AXIS]"), std::string::npos);
        EXPECT_NE(diagnostic.find("node=reduce"), std::string::npos);
        EXPECT_NE(diagnostic.find("axis=" + std::to_string(r.id)), std::string::npos);
        EXPECT_NE(diagnostic.find("input[0].repeats=" + std::to_string(input_r)), std::string::npos);
        EXPECT_NE(diagnostic.find("output[0].repeats=" + std::to_string(output_r)), std::string::npos);
        EXPECT_NE(diagnostic.find("expected=1"), std::string::npos);
      }
    }
  }
}

TEST(ScheduleGraphValidator, ChecksEveryReducedAxisAndDefersUnknownRepeat) {
  af::AscGraph graph("multiple_reduce_output_repeats");
  const auto a = graph.CreateAxis("a", af::Symbol(4));
  const auto r = graph.CreateAxis("r", af::Symbol(8));
  const std::vector<af::AxisId> axes = {a.id, r.id};
  af::ascir_op::Data data("data", graph);
  data.attr.api.type = af::ApiType::kAPITypeBuffer;
  data.y.dtype = af::DT_FLOAT16;
  *data.y.axis = axes;
  *data.y.repeats = {af::Symbol(4), af::Symbol(8)};
  *data.y.strides = {af::Symbol(8), af::ops::One};

  af::ascir_op::Max reduce("reduce");
  reduce.x = data.y;
  reduce.attr.api.compute_type = af::ComputeType::kComputeReduce;
  reduce.attr.api.type = af::ApiType::kAPITypeCompute;
  reduce.attr.sched.axis = axes;
  reduce.y.dtype = af::DT_FLOAT16;
  *reduce.y.axis = axes;
  *reduce.y.repeats = {af::ops::One, af::Symbol(2)};
  *reduce.y.strides = {af::ops::Zero, af::ops::Zero};
  ASSERT_NE(graph.FindNode("reduce"), nullptr);

  // 第一条 R 轴正确不能掩盖第二条 R 轴的错误。
  EXPECT_EQ(optimize::ValidateScheduleGraph(graph), af::FAILED);
  EXPECT_NE(optimize::GetScheduleValidationError().find("axis=" + std::to_string(r.id)), std::string::npos);
  *reduce.y.repeats = {af::ops::One, af::ops::One};
  EXPECT_EQ(optimize::ValidateScheduleGraph(graph), af::SUCCESS);

  // 无法静态判定的符号值沿用原有放行策略，不能宣称该值已经验证为 1。
  const auto unknown = graph.CreateSizeVar("unknown");
  *reduce.y.repeats = {af::ops::One, unknown};
  EXPECT_EQ(optimize::ValidateScheduleGraph(graph), af::SUCCESS);
  EXPECT_TRUE(optimize::GetScheduleValidationError().empty());
}

TEST(ScheduleGraphValidator, ChecksTransposeLengthsByAxisId) {
  af::AscGraph graph("transpose_axis_lengths");
  const auto a = graph.CreateAxis("a", af::Symbol(4));
  const auto b = graph.CreateAxis("b", af::Symbol(8));
  af::ascir_op::Data data("data", graph);
  data.attr.api.type = af::ApiType::kAPITypeBuffer;
  data.y.dtype = af::DT_FLOAT16;
  *data.y.axis = {a.id, b.id};
  *data.y.repeats = {af::Symbol(4), af::Symbol(8)};
  *data.y.strides = {af::Symbol(8), af::ops::One};

  af::ascir_op::Transpose transpose("transpose");
  transpose.x = data.y;
  transpose.attr.api.compute_type = af::ComputeType::kComputeTranspose;
  transpose.attr.api.type = af::ApiType::kAPITypeCompute;
  transpose.attr.sched.axis = {a.id, b.id};
  transpose.y.dtype = af::DT_FLOAT16;
  *transpose.y.axis = {b.id, a.id};
  *transpose.y.strides = {af::Symbol(4), af::ops::One};
  ASSERT_NE(graph.FindNode("transpose"), nullptr);

  // 正确交换、第一轴长度错误、第二轴长度错误、总元素数不变但长度没有跟随轴交换。
  const std::vector<std::vector<int64_t>> shapes = {{8, 4}, {6, 4}, {8, 3}, {4, 8}};
  for (const auto &shape : shapes) {
    SCOPED_TRACE("output=" + std::to_string(shape[0]) + "," + std::to_string(shape[1]));
    *transpose.y.repeats = {af::Symbol(shape[0]), af::Symbol(shape[1])};
    const auto status = optimize::ValidateScheduleGraph(graph);
    if (shape[0] == 8 && shape[1] == 4) {
      EXPECT_EQ(status, af::SUCCESS);
      EXPECT_TRUE(optimize::GetScheduleValidationError().empty());
    } else {
      EXPECT_EQ(status, af::FAILED);
      const size_t index = shape[0] != 8 ? 0U : 1U;
      const auto &diagnostic = optimize::GetScheduleValidationError();
      EXPECT_NE(diagnostic.find("[AF][SCHEDULE GRAPH VALIDATION ERROR][TRANSPOSE_AXIS]"), std::string::npos);
      EXPECT_NE(diagnostic.find("node=transpose"), std::string::npos);
      EXPECT_NE(diagnostic.find("axis=" + std::to_string(index == 0U ? b.id : a.id)), std::string::npos);
      EXPECT_NE(diagnostic.find("input[0].repeats=" + std::to_string(index == 0U ? 8 : 4)), std::string::npos);
      EXPECT_NE(diagnostic.find("output[0].repeats=" + std::to_string(shape[index])), std::string::npos);
    }
  }

  // 恒等置换沿用原有允许策略，但也必须保持轴长度不变。
  *transpose.y.axis = {a.id, b.id};
  *transpose.y.repeats = {af::Symbol(4), af::Symbol(8)};
  *transpose.y.strides = {af::Symbol(8), af::ops::One};
  EXPECT_EQ(optimize::ValidateScheduleGraph(graph), af::SUCCESS);
  *transpose.y.repeats = {af::Symbol(4), af::Symbol(6)};
  EXPECT_EQ(optimize::ValidateScheduleGraph(graph), af::FAILED);
  EXPECT_NE(optimize::GetScheduleValidationError().find("axis=" + std::to_string(b.id)), std::string::npos);
}

TEST(ScheduleGraphValidator, AllowsTransposeSymbolicLengthsAndRejectsMissingRepeats) {
  af::AscGraph graph("transpose_symbolic_lengths");
  const auto size = graph.CreateSizeVar("size");
  const auto unknown = graph.CreateSizeVar("unknown");
  const auto a = graph.CreateAxis("a", size);
  const auto b = graph.CreateAxis("b", af::Symbol(4));
  af::ascir_op::Data data("data", graph);
  data.attr.api.type = af::ApiType::kAPITypeBuffer;
  data.y.dtype = af::DT_FLOAT16;
  *data.y.axis = {a.id, b.id};
  *data.y.repeats = {size, af::Symbol(4)};
  *data.y.strides = {af::Symbol(4), af::ops::One};

  af::ascir_op::Transpose transpose("transpose");
  transpose.x = data.y;
  transpose.attr.api.compute_type = af::ComputeType::kComputeTranspose;
  transpose.attr.api.type = af::ApiType::kAPITypeCompute;
  transpose.attr.sched.axis = {a.id, b.id};
  transpose.y.dtype = af::DT_FLOAT16;
  *transpose.y.axis = {b.id, a.id};
  *transpose.y.repeats = {af::Symbol(4), size};
  *transpose.y.strides = {size, af::ops::One};
  ASSERT_NE(graph.FindNode("transpose"), nullptr);
  EXPECT_EQ(optimize::ValidateScheduleGraph(graph), af::SUCCESS);

  // 同一符号能证明长度相等；两个未知符号之间的关系无法确定时暂不拒绝。
  *transpose.y.repeats = {af::Symbol(4), unknown};
  EXPECT_EQ(optimize::ValidateScheduleGraph(graph), af::SUCCESS);
  EXPECT_TRUE(optimize::GetScheduleValidationError().empty());

  // 缺失 repeats 时应先报告元数据错误，不能在读取长度时越界。
  *data.y.repeats = {};
  EXPECT_EQ(optimize::ValidateScheduleGraph(graph), af::FAILED);
  EXPECT_NE(optimize::GetScheduleValidationError().find("[AF][SCHEDULE GRAPH VALIDATION ERROR][TENSOR_METADATA]"),
            std::string::npos);
}

// 使用真实数据边构造两轴 Reduce，便于分别修改轴身份、长度和 stride。
class ReduceShapeValidationTest : public testing::Test {
 protected:
  af::AscGraph graph_{"reduce_shape_contract"};
  af::AscNodePtr reduce_;
  af::AxisId a_;
  af::AxisId r_;

  void SetUp() override {
    a_ = graph_.CreateAxis("a", af::Symbol(4)).id;
    r_ = graph_.CreateAxis("r", af::Symbol(8)).id;
    af::ascir_op::Data data("data", graph_);
    data.attr.api.type = af::ApiType::kAPITypeBuffer;
    data.y.dtype = af::DT_FLOAT16;
    *data.y.axis = {a_, r_};
    *data.y.repeats = {af::Symbol(4), af::Symbol(8)};
    *data.y.strides = {af::Symbol(8), af::ops::One};
    af::ascir_op::Max reduce("reduce");
    reduce.x = data.y;
    reduce.attr.api.type = af::ApiType::kAPITypeCompute;
    reduce.attr.api.compute_type = af::ComputeType::kComputeReduce;
    reduce.y.dtype = af::DT_FLOAT16;
    *reduce.y.axis = {a_, r_};
    *reduce.y.repeats = {af::Symbol(4), af::ops::One};
    *reduce.y.strides = {af::ops::One, af::ops::Zero};
    reduce_ = graph_.FindNode("reduce");
    ASSERT_NE(reduce_, nullptr);
  }
};

TEST_F(ReduceShapeValidationTest, PreservesNonReducedLengthsWithOrWithoutScheduleAxes) {
  for (const bool filled : {false, true}) {
    SCOPED_TRACE(filled);
    reduce_->attr.sched.axis = filled ? std::vector<af::AxisId>{a_, r_} : std::vector<af::AxisId>{};
    reduce_->outputs[0].attr.repeats = {af::Symbol(5), af::ops::One};
    EXPECT_EQ(optimize::ValidateScheduleGraph(graph_), af::FAILED);
    EXPECT_NE(optimize::GetScheduleValidationError().find("non-reduced axis length changed"), std::string::npos);
    EXPECT_NE(optimize::GetScheduleValidationError().find("axis=" + std::to_string(a_)), std::string::npos);
    reduce_->outputs[0].attr.repeats = {af::Symbol(4), af::ops::One};
    EXPECT_EQ(optimize::ValidateScheduleGraph(graph_), af::SUCCESS);
  }
}

TEST_F(ReduceShapeValidationTest, RejectsReplacedOrReorderedAxes) {
  const auto other = graph_.CreateAxis("other", af::Symbol(8)).id;
  for (const auto &axes : {std::vector<af::AxisId>{a_, other}, std::vector<af::AxisId>{r_, a_}}) {
    reduce_->outputs[0].attr.axis = axes;
    EXPECT_EQ(optimize::ValidateScheduleGraph(graph_), af::FAILED);
    EXPECT_NE(optimize::GetScheduleValidationError().find("axis IDs and order must match"), std::string::npos);
  }
  reduce_->outputs[0].attr.axis = {a_, r_};
  EXPECT_EQ(optimize::ValidateScheduleGraph(graph_), af::SUCCESS);
}

TEST_F(ReduceShapeValidationTest, SingletonWithoutZeroStrideIsNotEvidenceOfReduction) {
  reduce_->inputs[0].attr.repeats = {af::ops::One, af::Symbol(8)};
  reduce_->outputs[0].attr.repeats = reduce_->inputs[0].attr.repeats;
  reduce_->outputs[0].attr.strides = reduce_->inputs[0].attr.strides;
  EXPECT_EQ(optimize::ValidateScheduleGraph(graph_), af::FAILED);
  EXPECT_NE(optimize::GetScheduleValidationError().find("cannot find an R axis"), std::string::npos);

  // 第一轴显式标为零 stride 后，允许对单元素轴做合法的退化归约。
  reduce_->outputs[0].attr.strides = {af::ops::Zero, af::ops::One};
  EXPECT_EQ(optimize::ValidateScheduleGraph(graph_), af::SUCCESS);
  // 已找到退化 R 轴也不能掩盖另一个轴的非法长度变化。
  reduce_->outputs[0].attr.repeats = {af::ops::One, af::Symbol(2)};
  EXPECT_EQ(optimize::ValidateScheduleGraph(graph_), af::FAILED);
  EXPECT_NE(optimize::GetScheduleValidationError().find("non-reduced axis length changed"), std::string::npos);
}

TEST_F(ReduceShapeValidationTest, DoesNotTreatUnknownStrideAsProvenReduction) {
  const auto unknown_stride = graph_.CreateSizeVar("unknown_stride");
  reduce_->inputs[0].attr.strides = {af::Symbol(8), unknown_stride};
  reduce_->outputs[0].attr.repeats = {af::Symbol(4), af::Symbol(8)};
  // 未知输入 stride 可能为零，此时第二轴也可能保持不变，不能静态断言输出必须为 1。
  EXPECT_EQ(optimize::ValidateScheduleGraph(graph_), af::SUCCESS);
  reduce_->inputs[0].attr.strides = {af::Symbol(8), af::ops::One};
  EXPECT_EQ(optimize::ValidateScheduleGraph(graph_), af::FAILED);
  EXPECT_NE(optimize::GetScheduleValidationError().find("expected=1"), std::string::npos);
}

TEST_F(ReduceShapeValidationTest, PreservesScalarViewCompatibility) {
  reduce_->inputs[0].attr.axis.clear();
  reduce_->inputs[0].attr.repeats.clear();
  reduce_->inputs[0].attr.strides.clear();
  reduce_->outputs[0].attr.axis.clear();
  reduce_->outputs[0].attr.repeats.clear();
  reduce_->outputs[0].attr.strides.clear();
  EXPECT_EQ(optimize::ValidateScheduleGraph(graph_), af::SUCCESS);
  EXPECT_TRUE(optimize::GetScheduleValidationError().empty());
}

TEST_F(ReduceShapeValidationTest, RequiresCompleteInputViewEvenFromBuffer) {
  reduce_->inputs[0].attr.repeats.clear();
  EXPECT_EQ(optimize::ValidateScheduleGraph(graph_), af::FAILED);
  EXPECT_NE(optimize::GetScheduleValidationError().find("input[0] metadata lengths differ"), std::string::npos);
}

}  // namespace
