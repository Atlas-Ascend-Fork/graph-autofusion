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

#include "graph/operator_reg_af.h"
#include "graph_utils_ex.h"
#include "node_utils.h"
#include "op_desc_utils.h"

#include "ascir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"
#include "default_reg_func_v2.h"

namespace af {
namespace ascir {

using namespace testing;
using namespace af::ascir_op;

class CalcSoftmaxTmpSizeV2Test : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}

  // 构造 Data->Load->Softmax->Store->Output；vectorized_strides 非空才能进入 tmp 计算分支。
  std::shared_ptr<AscNode> BuildGraph(af::AscGraph &graph, const af::DataType dtype, Expression &out_s0,
                                      Expression &out_s1) {
    auto s0 = graph.CreateSizeVar("s0");
    auto s1 = graph.CreateSizeVar("s1");
    auto s2 = graph.CreateSizeVar("s2");
    out_s0 = s0;
    out_s1 = s1;

    auto z0 = graph.CreateAxis("z0", s0);
    auto zo = graph.CreateAxis("zo", s1 + s2);
    auto zo_s_0 = graph.CreateAxis("zo_s_0", Axis::Type::kAxisTypeOriginal, s1, {zo.id}, af::kIdNone);

    af::ascir_op::Data x1("x1", graph);
    x1.attr.sched.axis = {z0.id, zo_s_0.id};
    x1.y.dtype = dtype;
    *x1.y.axis = {z0.id, zo_s_0.id};
    *x1.y.repeats = {s0, s1};
    *x1.y.strides = {s1, Symbol(1)};

    af::ascir_op::Load load1("load1");
    load1.x = x1.y;
    load1.attr.sched.axis = {z0.id, zo_s_0.id};
    load1.y.dtype = dtype;
    *load1.y.axis = {z0.id, zo_s_0.id};
    *load1.y.repeats = {s0, s1};
    *load1.y.strides = {s1, Symbol(1)};
    *load1.y.vectorized_axis = {z0.id, zo_s_0.id};

    af::ascir_op::Softmax softmax("softmax");
    softmax.x = load1.y;
    softmax.attr.sched.axis = {z0.id, zo_s_0.id};
    softmax.y.dtype = dtype;
    *softmax.y.axis = {z0.id, zo_s_0.id};
    *softmax.y.repeats = {s0, s1};
    *softmax.y.strides = {s1, Symbol(1)};
    // CalcSoftmaxTmpSizeV2 以 vectorized_axis + vectorized_strides 推导 a/r 轴；
    // vectorized_strides 为空时直接返回空描述符，故此处必须显式赋值。
    *softmax.y.vectorized_axis = {z0.id, zo_s_0.id};
    *softmax.y.vectorized_strides = {s1, Symbol(1)};

    af::ascir_op::Store store("store");
    store.x = softmax.y;
    store.attr.sched.axis = {z0.id, zo_s_0.id};
    store.y.dtype = dtype;
    *store.y.axis = {z0.id, zo_s_0.id};
    *store.y.repeats = {s0, s1};
    *store.y.strides = {s1, Symbol(1)};

    af::ascir_op::Output y("y");
    y.x = store.y;
    y.attr.sched.axis = {z0.id, zo_s_0.id};
    y.y.dtype = dtype;
    *y.y.axis = {z0.id, zo_s_0.id};
    *y.y.repeats = {s0, s1};
    *y.y.strides = {s1, Symbol(1)};

    return graph.FindNode("softmax");
  }
};

/**
 * @tc.name: CalcSoftmaxTmpSizeV2_ShouldReturnOneDesc_WhenVectorizedStridesNotEmpty
 * @tc.number: CalcSoftmaxTmpSizeV2_Test_001
 * @tc.desc: vectorized_strides 非空时返回 1 个 TmpBufDesc，且 life_time_axis_id 为 -1
 */
TEST_F(CalcSoftmaxTmpSizeV2Test, CalcSoftmaxTmpSizeV2_ShouldReturnOneDesc_WhenVectorizedStridesNotEmpty) {
  af::AscGraph graph("test");
  Expression s0;
  Expression s1;
  auto node = BuildGraph(graph, af::DT_FLOAT, s0, s1);
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcSoftmaxTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
  // FLOAT: align_size = 32/4 = 8；末轴 r_exp = Align(s1, 8)，其余轴 a_exp = s0；
  // element = a*r + a*8；tmp = element * dtype_size(4)。
  ASSERT_EQ(result[0]->size, sym::Mul(sym::Add(sym::Mul(s0, sym::Align(s1, 8)), sym::Mul(s0, Symbol(8))), Symbol(4)));
}

/**
 * @tc.name: CalcSoftmaxTmpSizeV2_ShouldReturnEmpty_WhenVectorizedStridesEmpty
 * @tc.number: CalcSoftmaxTmpSizeV2_Test_002
 * @tc.desc: vectorized_strides 为空时直接返回空描述符集合（提前返回分支）
 */
TEST_F(CalcSoftmaxTmpSizeV2Test, CalcSoftmaxTmpSizeV2_ShouldReturnEmpty_WhenVectorizedStridesEmpty) {
  af::AscGraph graph("test");
  Expression s0;
  Expression s1;
  auto node = BuildGraph(graph, af::DT_FLOAT, s0, s1);
  node->outputs[0].attr.vectorized_strides.clear();
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcSoftmaxTmpSizeV2(*node);
  ASSERT_TRUE(result.empty());
}
}  // namespace ascir
}  // namespace af
