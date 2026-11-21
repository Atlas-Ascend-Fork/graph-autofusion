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

#include "ascir_register.h"
#include "ascir_ops.h"
#include "v1_ascir_codegen_impl.h"

#ifndef CODEGEN_CALC_JOIN
#define CODEGEN_CALC_JOIN(a, b) a##b
#define JOIN(a, b) CODEGEN_CALC_JOIN(a, b)
#endif

namespace af {
namespace ascir {

class AscIrCodegenCalcTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Unary elementwise graph: Data -> Load -> Op -> Store -> Output
#define EXPECT_UNARY_CALC(ir_name)                                 \
  {                                                                \
    af::AscGraph graph("t_" #ir_name);                             \
    auto s0 = graph.CreateSizeVar("s0");                           \
    auto s1 = graph.CreateSizeVar("s1");                           \
    auto z0 = graph.CreateAxis("z0", s0);                          \
    auto z1 = graph.CreateAxis("z1", s1);                          \
    af::ascir_op::Data x1("x1", graph);                            \
    af::ascir_op::Load load1("load1");                             \
    af::ascir_op::ir_name op0("op0");                              \
    af::ascir_op::Store store("store");                            \
    af::ascir_op::Output y("y");                                   \
    x1.attr.sched.axis = {z0.id, z1.id};                           \
    x1.y.dtype = af::DT_FLOAT;                                     \
    *x1.y.axis = {z0.id, z1.id};                                   \
    *x1.y.repeats = {s0, s1};                                      \
    *x1.y.strides = {s1, af::Symbol(1)};                           \
    load1.x = x1.y;                                                \
    load1.attr.sched.axis = {z0.id, z1.id};                        \
    load1.y.dtype = af::DT_FLOAT;                                  \
    *load1.y.axis = {z0.id, z1.id};                                \
    *load1.y.repeats = {s0, s1};                                   \
    *load1.y.strides = {s1, af::Symbol(1)};                        \
    *load1.y.vectorized_axis = {z0.id, z1.id};                     \
    op0.x = load1.y;                                               \
    op0.attr.sched.axis = {z0.id, z1.id};                          \
    op0.y.dtype = af::DT_FLOAT;                                    \
    *op0.y.axis = {z0.id, z1.id};                                  \
    *op0.y.repeats = {s0, s1};                                     \
    *op0.y.strides = {s1, af::Symbol(1)};                          \
    *op0.y.vectorized_axis = {z0.id, z1.id};                       \
    store.x = op0.y;                                               \
    store.attr.sched.axis = {z0.id, z1.id};                        \
    store.y.dtype = af::DT_FLOAT;                                  \
    *store.y.axis = {z0.id, z1.id};                                \
    *store.y.repeats = {s0, s1};                                   \
    *store.y.strides = {s1, af::Symbol(1)};                        \
    y.x = store.y;                                                 \
    y.attr.sched.axis = {z0.id, z1.id};                            \
    y.y.dtype = af::DT_FLOAT;                                      \
    *y.y.axis = {z0.id, z1.id};                                    \
    *y.y.repeats = {s0, s1};                                       \
    *y.y.strides = {s1, af::Symbol(1)};                            \
    auto node = graph.FindNode("op0");                             \
    ASSERT_NE(node, nullptr);                                      \
    node->inputs[0].attr.vectorized_strides = {s1, af::Symbol(1)}; \
    JOIN(ir_name, AscIrCodegenImpl) impl;                          \
    const auto bufs = impl.CalcTmpBufSize(*node);                  \
    EXPECT_FALSE(bufs.empty());                                    \
    EXPECT_TRUE(impl.IsNodeValid(*node));                          \
    (void)impl.IsInplaceSupported(*node);                          \
    (void)impl.IsBrcInlineSupported(*node);                        \
  }

TEST_F(AscIrCodegenCalcTest, CalcTmpBufSize_UnaryOps) {
  EXPECT_UNARY_CALC(Erf);
  EXPECT_UNARY_CALC(Tanh);
  EXPECT_UNARY_CALC(Rsqrt);
  EXPECT_UNARY_CALC(Reciprocal);
  EXPECT_UNARY_CALC(Isnan);
  EXPECT_UNARY_CALC(IsInf);
  EXPECT_UNARY_CALC(IsFinite);
  EXPECT_UNARY_CALC(LogicalNot);
}

// Binary elementwise graph: both inputs are tensors
#define EXPECT_BINARY_CALC(ir_name)                                \
  {                                                                \
    af::AscGraph graph("t_" #ir_name);                             \
    auto s0 = graph.CreateSizeVar("s0");                           \
    auto s1 = graph.CreateSizeVar("s1");                           \
    auto z0 = graph.CreateAxis("z0", s0);                          \
    auto z1 = graph.CreateAxis("z1", s1);                          \
    af::ascir_op::Data x1("x1", graph);                            \
    af::ascir_op::Data x2("x2", graph);                            \
    af::ascir_op::Load load1("load1");                             \
    af::ascir_op::Load load2("load2");                             \
    af::ascir_op::ir_name op0("op0");                              \
    af::ascir_op::Store store("store");                            \
    af::ascir_op::Output y("y");                                   \
    x1.attr.sched.axis = {z0.id, z1.id};                           \
    x1.y.dtype = af::DT_FLOAT;                                     \
    *x1.y.axis = {z0.id, z1.id};                                   \
    *x1.y.repeats = {s0, s1};                                      \
    *x1.y.strides = {s1, af::Symbol(1)};                           \
    x2.attr.sched.axis = {z0.id, z1.id};                           \
    x2.y.dtype = af::DT_FLOAT;                                     \
    *x2.y.axis = {z0.id, z1.id};                                   \
    *x2.y.repeats = {s0, s1};                                      \
    *x2.y.strides = {s1, af::Symbol(1)};                           \
    load1.x = x1.y;                                                \
    load1.attr.sched.axis = {z0.id, z1.id};                        \
    load1.y.dtype = af::DT_FLOAT;                                  \
    *load1.y.axis = {z0.id, z1.id};                                \
    *load1.y.repeats = {s0, s1};                                   \
    *load1.y.strides = {s1, af::Symbol(1)};                        \
    *load1.y.vectorized_axis = {z0.id, z1.id};                     \
    load2.x = x2.y;                                                \
    load2.attr.sched.axis = {z0.id, z1.id};                        \
    load2.y.dtype = af::DT_FLOAT;                                  \
    *load2.y.axis = {z0.id, z1.id};                                \
    *load2.y.repeats = {s0, s1};                                   \
    *load2.y.strides = {s1, af::Symbol(1)};                        \
    *load2.y.vectorized_axis = {z0.id, z1.id};                     \
    op0.x1 = load1.y;                                              \
    op0.x2 = load2.y;                                              \
    op0.attr.sched.axis = {z0.id, z1.id};                          \
    op0.y.dtype = af::DT_FLOAT;                                    \
    *op0.y.axis = {z0.id, z1.id};                                  \
    *op0.y.repeats = {s0, s1};                                     \
    *op0.y.strides = {s1, af::Symbol(1)};                          \
    *op0.y.vectorized_axis = {z0.id, z1.id};                       \
    store.x = op0.y;                                               \
    store.attr.sched.axis = {z0.id, z1.id};                        \
    store.y.dtype = af::DT_FLOAT;                                  \
    *store.y.axis = {z0.id, z1.id};                                \
    *store.y.repeats = {s0, s1};                                   \
    *store.y.strides = {s1, af::Symbol(1)};                        \
    y.x = store.y;                                                 \
    y.attr.sched.axis = {z0.id, z1.id};                            \
    y.y.dtype = af::DT_FLOAT;                                      \
    *y.y.axis = {z0.id, z1.id};                                    \
    *y.y.repeats = {s0, s1};                                       \
    *y.y.strides = {s1, af::Symbol(1)};                            \
    auto node = graph.FindNode("op0");                             \
    ASSERT_NE(node, nullptr);                                      \
    node->inputs[0].attr.vectorized_strides = {s1, af::Symbol(1)}; \
    JOIN(ir_name, AscIrCodegenImpl) impl;                          \
    const auto bufs = impl.CalcTmpBufSize(*node);                  \
    EXPECT_FALSE(bufs.empty());                                    \
    EXPECT_TRUE(impl.IsNodeValid(*node));                          \
    (void)impl.IsInplaceSupported(*node);                          \
    (void)impl.IsBrcInlineSupported(*node);                        \
  }

TEST_F(AscIrCodegenCalcTest, CalcTmpBufSize_BinaryOps) {
  EXPECT_BINARY_CALC(Eq);
  EXPECT_BINARY_CALC(Ge);
  EXPECT_BINARY_CALC(Gt);
  EXPECT_BINARY_CALC(Le);
  EXPECT_BINARY_CALC(Lt);
  EXPECT_BINARY_CALC(Ne);
  EXPECT_BINARY_CALC(TrueDiv);
  EXPECT_BINARY_CALC(Remainder);
  EXPECT_BINARY_CALC(FloorDiv);
  EXPECT_BINARY_CALC(LogicalAnd);
  EXPECT_BINARY_CALC(LogicalOr);
  EXPECT_BINARY_CALC(BitwiseAnd);
}

// Extended coverage: node validity checks without the tmp-buffer assertion
// (some impls do not override CalcTmpBufSize and legitimately return an empty vector)
#define EXPECT_UNARY_VALID(ir_name)                                \
  {                                                                \
    af::AscGraph graph("t_" #ir_name);                             \
    auto s0 = graph.CreateSizeVar("s0");                           \
    auto s1 = graph.CreateSizeVar("s1");                           \
    auto z0 = graph.CreateAxis("z0", s0);                          \
    auto z1 = graph.CreateAxis("z1", s1);                          \
    af::ascir_op::Data x1("x1", graph);                            \
    af::ascir_op::Load load1("load1");                             \
    af::ascir_op::ir_name op0("op0");                              \
    af::ascir_op::Store store("store");                            \
    af::ascir_op::Output y("y");                                   \
    x1.attr.sched.axis = {z0.id, z1.id};                           \
    x1.y.dtype = af::DT_FLOAT;                                     \
    *x1.y.axis = {z0.id, z1.id};                                   \
    *x1.y.repeats = {s0, s1};                                      \
    *x1.y.strides = {s1, af::Symbol(1)};                           \
    load1.x = x1.y;                                                \
    load1.attr.sched.axis = {z0.id, z1.id};                        \
    load1.y.dtype = af::DT_FLOAT;                                  \
    *load1.y.axis = {z0.id, z1.id};                                \
    *load1.y.repeats = {s0, s1};                                   \
    *load1.y.strides = {s1, af::Symbol(1)};                        \
    *load1.y.vectorized_axis = {z0.id, z1.id};                     \
    op0.x = load1.y;                                               \
    op0.attr.sched.axis = {z0.id, z1.id};                          \
    op0.y.dtype = af::DT_FLOAT;                                    \
    *op0.y.axis = {z0.id, z1.id};                                  \
    *op0.y.repeats = {s0, s1};                                     \
    *op0.y.strides = {s1, af::Symbol(1)};                          \
    *op0.y.vectorized_axis = {z0.id, z1.id};                       \
    store.x = op0.y;                                               \
    store.attr.sched.axis = {z0.id, z1.id};                        \
    store.y.dtype = af::DT_FLOAT;                                  \
    *store.y.axis = {z0.id, z1.id};                                \
    *store.y.repeats = {s0, s1};                                   \
    *store.y.strides = {s1, af::Symbol(1)};                        \
    y.x = store.y;                                                 \
    y.attr.sched.axis = {z0.id, z1.id};                            \
    y.y.dtype = af::DT_FLOAT;                                      \
    *y.y.axis = {z0.id, z1.id};                                    \
    *y.y.repeats = {s0, s1};                                       \
    *y.y.strides = {s1, af::Symbol(1)};                            \
    auto node = graph.FindNode("op0");                             \
    ASSERT_NE(node, nullptr);                                      \
    node->inputs[0].attr.vectorized_strides = {s1, af::Symbol(1)}; \
    JOIN(ir_name, AscIrCodegenImpl) impl;                          \
    EXPECT_TRUE(impl.IsNodeValid(*node));                          \
    (void)impl.IsInplaceSupported(*node);                          \
    (void)impl.IsBrcInlineSupported(*node);                        \
    (void)impl.CalcTmpBufSize(*node);                              \
  }

TEST_F(AscIrCodegenCalcTest, IsNodeValid_UnaryExtended) {
  EXPECT_UNARY_VALID(Exp);
  EXPECT_UNARY_VALID(Sigmoid);
  EXPECT_UNARY_VALID(Sign);
  EXPECT_UNARY_VALID(Sqrt);
  EXPECT_UNARY_VALID(Neg);
  EXPECT_UNARY_VALID(Ln);
  EXPECT_UNARY_VALID(LeakyRelu);
  EXPECT_UNARY_VALID(RemovePad);
  EXPECT_UNARY_VALID(Cast);
  EXPECT_UNARY_VALID(Gelu);
  EXPECT_UNARY_VALID(Relu);
  EXPECT_UNARY_VALID(Transpose);
  EXPECT_UNARY_VALID(Ub2ub);
  EXPECT_UNARY_VALID(Pad);
}

// Reduce-family graph: 3-dim input with loop_axis on dim0 (Tier-1 proven layout)
#define EXPECT_REDUCE_VALID(ir_name)                        \
  {                                                         \
    af::AscGraph graph("t_" #ir_name);                      \
    af::Expression One = af::Symbol(1);                     \
    af::Expression Zero = af::Symbol(0);                    \
    auto s0 = graph.CreateSizeVar("s0");                    \
    auto s1 = graph.CreateSizeVar("s1");                    \
    auto s2 = graph.CreateSizeVar("s2");                    \
    auto z0 = graph.CreateAxis("z0", s0);                   \
    auto z1 = graph.CreateAxis("z1", s1);                   \
    auto z2 = graph.CreateAxis("z2", s2);                   \
    af::ascir_op::Data x1("x1", graph);                     \
    af::ascir_op::Load load1("load1");                      \
    af::ascir_op::ir_name op0("op0");                       \
    af::ascir_op::Store store("store");                     \
    af::ascir_op::Output y("y");                            \
    x1.attr.sched.axis = {z0.id, z1.id, z2.id};             \
    x1.y.dtype = af::DT_FLOAT;                              \
    *x1.y.axis = {z0.id, z1.id, z2.id};                     \
    *x1.y.repeats = {s0, s1, s2};                           \
    *x1.y.strides = {s1 * s2, s2, One};                     \
    load1.x = x1.y;                                         \
    load1.attr.sched.axis = {z0.id, z1.id, z2.id};          \
    load1.y.dtype = af::DT_FLOAT;                           \
    *load1.y.axis = {z0.id, z1.id, z2.id};                  \
    *load1.y.repeats = {s0, s1, s2};                        \
    *load1.y.strides = {s1 * s2, s2, One};                  \
    *load1.y.vectorized_axis = {z1.id, z2.id};              \
    op0.x = load1.y;                                        \
    op0.attr.sched.axis = {z0.id, z1.id, z2.id};            \
    op0.attr.sched.loop_axis = {z0.id};                     \
    op0.y.dtype = af::DT_FLOAT;                             \
    *op0.y.axis = {z0.id, z1.id, z2.id};                    \
    *op0.y.repeats = {s0, s1, One};                         \
    *op0.y.strides = {s2, One, Zero};                       \
    *op0.y.vectorized_axis = {z1.id, z2.id};                \
    store.x = op0.y;                                        \
    store.attr.sched.axis = {z0.id, z1.id, z2.id};          \
    store.y.dtype = af::DT_FLOAT;                           \
    *store.y.axis = {z0.id, z1.id, z2.id};                  \
    *store.y.repeats = {s0, s1, s2};                        \
    *store.y.strides = {s1 * s2, s2, One};                  \
    y.x = store.y;                                          \
    y.attr.sched.axis = {z0.id, z1.id, z2.id};              \
    y.y.dtype = af::DT_FLOAT;                               \
    *y.y.axis = {z0.id, z1.id, z2.id};                      \
    *y.y.repeats = {s0, s1, s2};                            \
    *y.y.strides = {s1 * s2, s2, One};                      \
    auto node = graph.FindNode("op0");                      \
    ASSERT_NE(node, nullptr);                               \
    node->inputs[0].attr.vectorized_strides = {s2, One};    \
    node->outputs[0].attr.vectorized_strides = {One, Zero}; \
    JOIN(ir_name, AscIrCodegenImpl) impl;                   \
    EXPECT_TRUE(impl.IsNodeValid(*node));                   \
    (void)impl.IsInplaceSupported(*node);                   \
    (void)impl.IsBrcInlineSupported(*node);                 \
    (void)impl.CalcTmpBufSize(*node);                       \
  }

TEST_F(AscIrCodegenCalcTest, ReduceFamily_ValidAndCalc) {
  EXPECT_REDUCE_VALID(Sum);
  EXPECT_REDUCE_VALID(Max);
  EXPECT_REDUCE_VALID(Mean);
  EXPECT_REDUCE_VALID(All);
  EXPECT_REDUCE_VALID(Any);
  EXPECT_REDUCE_VALID(Min);
  EXPECT_REDUCE_VALID(Prod);
  EXPECT_REDUCE_VALID(Argmax);
}

// ArgMaxMultiRPhase1: single input x, dual outputs value/index
TEST_F(AscIrCodegenCalcTest, ReduceFamily_ArgmaxMultiRPhase) {
  {
    af::AscGraph graph("t_p1");
    af::Expression One = af::Symbol(1);
    af::Expression Zero = af::Symbol(0);
    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");
    auto z0 = graph.CreateAxis("z0", s0);
    auto z1 = graph.CreateAxis("z1", s1);
    af::ascir_op::Data x1("x1", graph);
    af::ascir_op::Load load1("load1");
    af::ascir_op::ArgMaxMultiRPhase1 op0("op0");
    af::ascir_op::Store store("store");
    af::ascir_op::Output y("y");
    x1.attr.sched.axis = {z0.id, z1.id};
    x1.y.dtype = af::DT_FLOAT;
    *x1.y.axis = {z0.id, z1.id};
    *x1.y.repeats = {s0, s1};
    *x1.y.strides = {s1, One};
    load1.x = x1.y;
    load1.attr.sched.axis = {z0.id, z1.id};
    load1.y.dtype = af::DT_FLOAT;
    *load1.y.axis = {z0.id, z1.id};
    *load1.y.repeats = {s0, s1};
    *load1.y.strides = {s1, One};
    *load1.y.vectorized_axis = {z0.id, z1.id};
    op0.x = load1.y;
    op0.attr.sched.axis = {z0.id, z1.id};
    op0.attr.sched.loop_axis = {z0.id};
    op0.value.dtype = af::DT_FLOAT;
    *op0.value.axis = {z0.id, z1.id};
    *op0.value.repeats = {s0, One};
    *op0.value.strides = {One, Zero};
    store.x = op0.value;
    store.attr.sched.axis = {z0.id, z1.id};
    store.y.dtype = af::DT_FLOAT;
    *store.y.axis = {z0.id, z1.id};
    *store.y.repeats = {s0, One};
    *store.y.strides = {One, Zero};
    y.x = store.y;
    y.attr.sched.axis = {z0.id, z1.id};
    y.y.dtype = af::DT_FLOAT;
    *y.y.axis = {z0.id, z1.id};
    *y.y.repeats = {s0, One};
    *y.y.strides = {One, Zero};
    auto node = graph.FindNode("op0");
    ASSERT_NE(node, nullptr);
    node->inputs[0].attr.vectorized_strides = {s1, One};
    node->outputs[0].attr.vectorized_strides = {One, Zero};
    ArgMaxMultiRPhase1AscIrCodegenImpl impl;
    EXPECT_TRUE(impl.IsNodeValid(*node));
  }
}

// Binary validity-only: impls whose CalcTmpBufSize may fall back to the empty base default
#define EXPECT_BINARY_VALID(ir_name) \
  { EXPECT_BINARY_CALC_CORE(ir_name, /*with_calc_assert=*/false); }

#define EXPECT_BINARY_CALC_CORE(ir_name, with_calc_assert)         \
  {                                                                \
    af::AscGraph graph("t_" #ir_name);                             \
    auto s0 = graph.CreateSizeVar("s0");                           \
    auto s1 = graph.CreateSizeVar("s1");                           \
    auto z0 = graph.CreateAxis("z0", s0);                          \
    auto z1 = graph.CreateAxis("z1", s1);                          \
    af::ascir_op::Data x1("x1", graph);                            \
    af::ascir_op::Data x2("x2", graph);                            \
    af::ascir_op::Load load1("load1");                             \
    af::ascir_op::Load load2("load2");                             \
    af::ascir_op::ir_name op0("op0");                              \
    af::ascir_op::Store store("store");                            \
    af::ascir_op::Output y("y");                                   \
    x1.attr.sched.axis = {z0.id, z1.id};                           \
    x1.y.dtype = af::DT_FLOAT;                                     \
    *x1.y.axis = {z0.id, z1.id};                                   \
    *x1.y.repeats = {s0, s1};                                      \
    *x1.y.strides = {s1, af::Symbol(1)};                           \
    x2.attr.sched.axis = {z0.id, z1.id};                           \
    x2.y.dtype = af::DT_FLOAT;                                     \
    *x2.y.axis = {z0.id, z1.id};                                   \
    *x2.y.repeats = {s0, s1};                                      \
    *x2.y.strides = {s1, af::Symbol(1)};                           \
    load1.x = x1.y;                                                \
    load1.attr.sched.axis = {z0.id, z1.id};                        \
    load1.y.dtype = af::DT_FLOAT;                                  \
    *load1.y.axis = {z0.id, z1.id};                                \
    *load1.y.repeats = {s0, s1};                                   \
    *load1.y.strides = {s1, af::Symbol(1)};                        \
    *load1.y.vectorized_axis = {z0.id, z1.id};                     \
    load2.x = x2.y;                                                \
    load2.attr.sched.axis = {z0.id, z1.id};                        \
    load2.y.dtype = af::DT_FLOAT;                                  \
    *load2.y.axis = {z0.id, z1.id};                                \
    *load2.y.repeats = {s0, s1};                                   \
    *load2.y.strides = {s1, af::Symbol(1)};                        \
    *load2.y.vectorized_axis = {z0.id, z1.id};                     \
    op0.x1 = load1.y;                                              \
    op0.x2 = load2.y;                                              \
    op0.attr.sched.axis = {z0.id, z1.id};                          \
    op0.y.dtype = af::DT_FLOAT;                                    \
    *op0.y.axis = {z0.id, z1.id};                                  \
    *op0.y.repeats = {s0, s1};                                     \
    *op0.y.strides = {s1, af::Symbol(1)};                          \
    *op0.y.vectorized_axis = {z0.id, z1.id};                       \
    store.x = op0.y;                                               \
    store.attr.sched.axis = {z0.id, z1.id};                        \
    store.y.dtype = af::DT_FLOAT;                                  \
    *store.y.axis = {z0.id, z1.id};                                \
    *store.y.repeats = {s0, s1};                                   \
    *store.y.strides = {s1, af::Symbol(1)};                        \
    y.x = store.y;                                                 \
    y.attr.sched.axis = {z0.id, z1.id};                            \
    y.y.dtype = af::DT_FLOAT;                                      \
    *y.y.axis = {z0.id, z1.id};                                    \
    *y.y.repeats = {s0, s1};                                       \
    *y.y.strides = {s1, af::Symbol(1)};                            \
    auto node = graph.FindNode("op0");                             \
    ASSERT_NE(node, nullptr);                                      \
    node->inputs[0].attr.vectorized_strides = {s1, af::Symbol(1)}; \
    JOIN(ir_name, AscIrCodegenImpl) impl;                          \
    EXPECT_TRUE(impl.IsNodeValid(*node));                          \
    (void)impl.IsInplaceSupported(*node);                          \
    (void)impl.IsBrcInlineSupported(*node);                        \
    if (with_calc_assert) {                                        \
      const auto bufs = impl.CalcTmpBufSize(*node);                \
      EXPECT_FALSE(bufs.empty());                                  \
    } else {                                                       \
      (void)impl.CalcTmpBufSize(*node);                            \
    }                                                              \
  }

TEST_F(AscIrCodegenCalcTest, IsNodeValid_BinaryExtended) {
  EXPECT_BINARY_VALID(Div);
  EXPECT_BINARY_VALID(Sub);
  EXPECT_BINARY_VALID(Mul);
  EXPECT_BINARY_VALID(Minimum);
  EXPECT_BINARY_VALID(Maximum);
}

TEST_F(AscIrCodegenCalcTest, IsNodeValid_MatmulFamily) {
  EXPECT_BINARY_VALID(MatMul);
  EXPECT_BINARY_VALID(BatchMatMul);
  EXPECT_BINARY_VALID(Gather);
  EXPECT_BINARY_VALID(Axpy);
}

// Three-input ops (Where/Select/ClipByValue share the x1/x2/x3 layout; MaskedFill uses x/mask/value)
#define EXPECT_TRINARY_VALID(ir_name, in1, in2, in3)               \
  {                                                                \
    af::AscGraph graph("t_" #ir_name);                             \
    auto s0 = graph.CreateSizeVar("s0");                           \
    auto s1 = graph.CreateSizeVar("s1");                           \
    auto z0 = graph.CreateAxis("z0", s0);                          \
    auto z1 = graph.CreateAxis("z1", s1);                          \
    af::ascir_op::Data x1("x1", graph);                            \
    af::ascir_op::Data x2("x2", graph);                            \
    af::ascir_op::Data x3("x3", graph);                            \
    af::ascir_op::Load load1("load1");                             \
    af::ascir_op::Load load2("load2");                             \
    af::ascir_op::Load load3("load3");                             \
    af::ascir_op::ir_name op0("op0");                              \
    af::ascir_op::Store store("store");                            \
    af::ascir_op::Output y("y");                                   \
    x1.attr.sched.axis = {z0.id, z1.id};                           \
    x1.y.dtype = af::DT_FLOAT;                                     \
    *x1.y.axis = {z0.id, z1.id};                                   \
    *x1.y.repeats = {s0, s1};                                      \
    *x1.y.strides = {s1, af::Symbol(1)};                           \
    x2.attr.sched.axis = {z0.id, z1.id};                           \
    x2.y.dtype = af::DT_FLOAT;                                     \
    *x2.y.axis = {z0.id, z1.id};                                   \
    *x2.y.repeats = {s0, s1};                                      \
    *x2.y.strides = {s1, af::Symbol(1)};                           \
    x3.attr.sched.axis = {z0.id, z1.id};                           \
    x3.y.dtype = af::DT_FLOAT;                                     \
    *x3.y.axis = {z0.id, z1.id};                                   \
    *x3.y.repeats = {s0, s1};                                      \
    *x3.y.strides = {s1, af::Symbol(1)};                           \
    load1.x = x1.y;                                                \
    load1.attr.sched.axis = {z0.id, z1.id};                        \
    load1.y.dtype = af::DT_FLOAT;                                  \
    *load1.y.axis = {z0.id, z1.id};                                \
    *load1.y.repeats = {s0, s1};                                   \
    *load1.y.strides = {s1, af::Symbol(1)};                        \
    *load1.y.vectorized_axis = {z0.id, z1.id};                     \
    load2.x = x2.y;                                                \
    load2.attr.sched.axis = {z0.id, z1.id};                        \
    load2.y.dtype = af::DT_FLOAT;                                  \
    *load2.y.axis = {z0.id, z1.id};                                \
    *load2.y.repeats = {s0, s1};                                   \
    *load2.y.strides = {s1, af::Symbol(1)};                        \
    *load2.y.vectorized_axis = {z0.id, z1.id};                     \
    load3.x = x3.y;                                                \
    load3.attr.sched.axis = {z0.id, z1.id};                        \
    load3.y.dtype = af::DT_FLOAT;                                  \
    *load3.y.axis = {z0.id, z1.id};                                \
    *load3.y.repeats = {s0, s1};                                   \
    *load3.y.strides = {s1, af::Symbol(1)};                        \
    *load3.y.vectorized_axis = {z0.id, z1.id};                     \
    op0.in1 = load1.y;                                             \
    op0.in2 = load2.y;                                             \
    op0.in3 = load3.y;                                             \
    op0.attr.sched.axis = {z0.id, z1.id};                          \
    op0.y.dtype = af::DT_FLOAT;                                    \
    *op0.y.axis = {z0.id, z1.id};                                  \
    *op0.y.repeats = {s0, s1};                                     \
    *op0.y.strides = {s1, af::Symbol(1)};                          \
    *op0.y.vectorized_axis = {z0.id, z1.id};                       \
    store.x = op0.y;                                               \
    store.attr.sched.axis = {z0.id, z1.id};                        \
    store.y.dtype = af::DT_FLOAT;                                  \
    *store.y.axis = {z0.id, z1.id};                                \
    *store.y.repeats = {s0, s1};                                   \
    *store.y.strides = {s1, af::Symbol(1)};                        \
    y.x = store.y;                                                 \
    y.attr.sched.axis = {z0.id, z1.id};                            \
    y.y.dtype = af::DT_FLOAT;                                      \
    *y.y.axis = {z0.id, z1.id};                                    \
    *y.y.repeats = {s0, s1};                                       \
    *y.y.strides = {s1, af::Symbol(1)};                            \
    auto node = graph.FindNode("op0");                             \
    ASSERT_NE(node, nullptr);                                      \
    node->inputs[0].attr.vectorized_strides = {s1, af::Symbol(1)}; \
    JOIN(ir_name, AscIrCodegenImpl) impl;                          \
    EXPECT_TRUE(impl.IsNodeValid(*node));                          \
    (void)impl.CalcTmpBufSize(*node);                              \
    (void)impl.IsInplaceSupported(*node);                          \
    (void)impl.IsBrcInlineSupported(*node);                        \
  }

TEST_F(AscIrCodegenCalcTest, IsNodeValid_TrinaryOps) {
  EXPECT_TRINARY_VALID(Where, x1, x2, x3);
  EXPECT_TRINARY_VALID(Select, x1, x2, x3);
  EXPECT_TRINARY_VALID(ClipByValue, x1, x2, x3);
  EXPECT_TRINARY_VALID(MaskedFill, x, mask, value);
}

// 回归用例：int64 Compare 的 tmp 需求必须按 repeats 换算“一次调用处理的行数”。
// 现场问题：vectorized_axis 中把 degenerate 轴(repeat==1, 向量化 stride==0)放在索引 1 时，
// 旧公式 repeats[axis_ids[0]] * vectorized_strides[1] 会把整项乘成 0，tiling 于是给该算子
// 分配 0 字节 tmp；device 侧 CompareExtendInt64GtGeLe 按 repeat_times 切分 tmp 时越过 UB，
// 触发 aivec "VEC 指令读写 UB 越界"（tf_af_broadcast_support_int64_0002）。
TEST_F(AscIrCodegenCalcTest, CalcTmpBufSize_Int64CompareWithDegenerateVectorizedAxis) {
  af::AscGraph graph("t_le_int64_degenerate");
  const auto z0 = graph.CreateAxis("z0", af::Symbol(1));
  const auto z1 = graph.CreateAxis("z1", af::Symbol(10));
  const auto z2 = graph.CreateAxis("z2", af::Symbol(2));
  const auto z3 = graph.CreateAxis("z3", af::Symbol(23));
  const std::vector<af::AxisId> axes{z0.id, z1.id, z2.id, z3.id};
  const std::vector<af::Expression> repeats{af::Symbol(1), af::Symbol(10), af::Symbol(2), af::Symbol(23)};
  const std::vector<af::Expression> strides{af::Symbol(460), af::Symbol(46), af::Symbol(2), af::Symbol(1)};
  // vectorized_axis 取 axis 顺序，索引 1 落在 z1(repeat==1) 上，其向量化 stride 为 0
  const std::vector<af::Expression> vec_strides{af::Symbol(46), af::Symbol(0), af::Symbol(2), af::Symbol(1)};

  af::ascir_op::Data x1("x1", graph);
  af::ascir_op::Data x2("x2", graph);
  af::ascir_op::Load load1("load1");
  af::ascir_op::Load load2("load2");
  af::ascir_op::Le le("le");
  af::ascir_op::Store store("store");
  af::ascir_op::Output y("y");

  x1.attr.sched.axis = axes;
  x1.y.dtype = af::DT_INT64;
  *x1.y.axis = axes;
  *x1.y.repeats = repeats;
  *x1.y.strides = strides;
  x2.attr.sched.axis = axes;
  x2.y.dtype = af::DT_INT64;
  *x2.y.axis = axes;
  *x2.y.repeats = repeats;
  *x2.y.strides = strides;
  load1.x = x1.y;
  load1.attr.sched.axis = axes;
  load1.y.dtype = af::DT_INT64;
  *load1.y.axis = axes;
  *load1.y.repeats = repeats;
  *load1.y.strides = strides;
  *load1.y.vectorized_axis = axes;
  *load1.y.vectorized_strides = vec_strides;
  load2.x = x2.y;
  load2.attr.sched.axis = axes;
  load2.y.dtype = af::DT_INT64;
  *load2.y.axis = axes;
  *load2.y.repeats = repeats;
  *load2.y.strides = strides;
  *load2.y.vectorized_axis = axes;
  *load2.y.vectorized_strides = vec_strides;
  le.x1 = load1.y;
  le.x2 = load2.y;
  le.attr.sched.axis = axes;
  le.y.dtype = af::DT_UINT8;
  *le.y.axis = axes;
  *le.y.repeats = repeats;
  *le.y.strides = strides;
  *le.y.vectorized_axis = axes;
  store.x = le.y;
  store.attr.sched.axis = axes;
  store.y.dtype = af::DT_UINT8;
  *store.y.axis = axes;
  *store.y.repeats = repeats;
  *store.y.strides = strides;
  y.x = store.y;
  y.attr.sched.axis = axes;
  y.y.dtype = af::DT_UINT8;
  *y.y.axis = axes;
  *y.y.repeats = repeats;
  *y.y.strides = strides;

  auto node = graph.FindNode("le");
  ASSERT_NE(node, nullptr);
  af::ascir::LeAscIrCodegenImpl impl;
  const auto bufs = impl.CalcTmpBufSize(*node);
  ASSERT_EQ(bufs.size(), 1U);
  ASSERT_NE(bufs[0], nullptr);
  int64_t size_bytes = -1;
  ASSERT_TRUE(bufs[0]->size.Simplify().GetConstValue(size_bytes));
  // rows = repeats[z0] * repeats[z1] * repeats[z2] = 1 * 10 * 2 = 20；末轴 Align(23, 32) = 32
  // 需求 = 40 * rows * Align(last_axis, 32) = 40 * 20 * 32 = 25600，旧公式会得到 0
  EXPECT_EQ(size_bytes, 40 * 20 * 32);
}

}  // namespace ascir
}  // namespace af
