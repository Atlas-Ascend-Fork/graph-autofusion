/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include <string>
#include <vector>

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "ascir/generator/v2_ascir_codegen_impl.h"

using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;

namespace {

// 构造 Data -> Load -> Maximum/Minimum(load, scalar) 测试图，返回 min/max 节点
af::AscNodePtr BuildMinMaxGraph(af::AscGraph &graph, ge::DataType dtype, bool build_maximum) {
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);

  Data x_op("x", graph);
  Load load_op("load");
  Scalar constant_op("constant");
  constant_op.ir_attr.SetValue("0.89");
  graph.AddNode(load_op);
  graph.AddNode(constant_op);

  load_op.x = x_op.y;
  load_op.attr.sched.axis = {z0.id};
  *load_op.y.axis = {z0.id};
  *load_op.y.repeats = {s0};
  *load_op.y.strides = {One};

  if (build_maximum) {
    Maximum max_op("maximum");
    graph.AddNode(max_op);
    max_op.x1 = load_op.y;
    max_op.x2 = constant_op.y;
    *max_op.y.axis = {z0.id};
    *max_op.y.repeats = {s0};
    *max_op.y.strides = {One};
  } else {
    Minimum min_op("minimum");
    graph.AddNode(min_op);
    min_op.x1 = load_op.y;
    min_op.x2 = constant_op.y;
    *min_op.y.axis = {z0.id};
    *min_op.y.repeats = {s0};
    *min_op.y.strides = {One};
  }

  auto load = graph.FindNode("load");
  load->outputs[0].attr.dtype = dtype;
  auto constant = graph.FindNode("constant");
  constant->outputs[0].attr.dtype = dtype;

  auto min_max = graph.FindNode(build_maximum ? "maximum" : "minimum");
  min_max->inputs[0].attr.dtype = dtype;
  min_max->inputs[1].attr.dtype = dtype;
  min_max->outputs[0].attr.dtype = dtype;
  return min_max;
}

std::string GenSimtScalarExpr(const af::ascir::AscIrCodegenV2 &codegen, const af::AscNode &node) {
  std::string expr;
  const ge::graphStatus ret = codegen.GenerateSimtScalarExpr(node, {"v_0", "v_1"}, expr);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  return expr;
}

}  // namespace

// torch.maximum NaN 传播语义：浮点 SIMT 标量表达式需先判 (x != x) 保 NaN，
// 裸三目 (x > y ? x : y) 会把首输入的 NaN 静默替换为第二输入
TEST(CodegenKernel, TestMaximumFloatSimtExprPropagatesNaN) {
  af::AscGraph graph("test_maximum_float_simt");
  auto node = BuildMinMaxGraph(graph, DT_FLOAT, true);
  af::ascir::MaximumAscIrCodegenImplV2 codegen;
  EXPECT_TRUE(codegen.IsSimtScalarSupported(*node));
  EXPECT_EQ(GenSimtScalarExpr(codegen, *node), "((v_0 != v_0) || (v_0 > v_1)) ? v_0 : v_1");
}

// torch.minimum NaN 传播语义：与 maximum 对称，浮点先判 (x != x) 保 NaN
TEST(CodegenKernel, TestMinimumFloatSimtExprPropagatesNaN) {
  af::AscGraph graph("test_minimum_float_simt");
  auto node = BuildMinMaxGraph(graph, DT_FLOAT, false);
  af::ascir::MinimumAscIrCodegenImplV2 codegen;
  EXPECT_TRUE(codegen.IsSimtScalarSupported(*node));
  EXPECT_EQ(GenSimtScalarExpr(codegen, *node), "((v_0 != v_0) || (v_0 < v_1)) ? v_0 : v_1");
}

// float16/bfloat16 属浮点分支，与 float32 同样传播 NaN
TEST(CodegenKernel, TestMaximumFloat16AndMinimumBfloat16SimtExprPropagatesNaN) {
  af::AscGraph graph_fp16("test_maximum_fp16_simt");
  auto node_fp16 = BuildMinMaxGraph(graph_fp16, DT_FLOAT16, true);
  af::ascir::MaximumAscIrCodegenImplV2 max_codegen;
  EXPECT_EQ(GenSimtScalarExpr(max_codegen, *node_fp16), "((v_0 != v_0) || (v_0 > v_1)) ? v_0 : v_1");

  af::AscGraph graph_bf16("test_minimum_bf16_simt");
  auto node_bf16 = BuildMinMaxGraph(graph_bf16, DT_BF16, false);
  af::ascir::MinimumAscIrCodegenImplV2 min_codegen;
  EXPECT_EQ(GenSimtScalarExpr(min_codegen, *node_bf16), "((v_0 != v_0) || (v_0 < v_1)) ? v_0 : v_1");
}

// 整型无 NaN，保持原有三目 lowering 不变
TEST(CodegenKernel, TestMinMaxInt32SimtExprKeepsPlainTernary) {
  af::AscGraph graph_max("test_maximum_int32_simt");
  auto node_max = BuildMinMaxGraph(graph_max, DT_INT32, true);
  af::ascir::MaximumAscIrCodegenImplV2 max_codegen;
  EXPECT_TRUE(max_codegen.IsSimtScalarSupported(*node_max));
  EXPECT_EQ(GenSimtScalarExpr(max_codegen, *node_max), "(v_0 > v_1 ? v_0 : v_1)");

  af::AscGraph graph_min("test_minimum_int32_simt");
  auto node_min = BuildMinMaxGraph(graph_min, DT_INT32, false);
  af::ascir::MinimumAscIrCodegenImplV2 min_codegen;
  EXPECT_TRUE(min_codegen.IsSimtScalarSupported(*node_min));
  EXPECT_EQ(GenSimtScalarExpr(min_codegen, *node_min), "(v_0 < v_1 ? v_0 : v_1)");
}
