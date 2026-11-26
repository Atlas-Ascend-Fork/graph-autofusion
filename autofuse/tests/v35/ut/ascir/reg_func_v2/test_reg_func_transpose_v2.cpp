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

constexpr int32_t kTransposeBlockBytes = 32;

class CalcTransposeTmpSizeV2Test : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}

  // 构造 Data->Load->Transpose->Store->Output。
  // vectorized_strides 取连续布局 {s1, 1}，使 :42 的 "temp buffer size" 日志行被实际执行。
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

    af::ascir_op::Transpose transpose("transpose");
    transpose.x = load1.y;
    transpose.attr.sched.axis = {z0.id, zo_s_0.id};
    transpose.y.dtype = dtype;
    *transpose.y.axis = {z0.id, zo_s_0.id};
    *transpose.y.repeats = {s0, s1};
    *transpose.y.strides = {s1, Symbol(1)};
    *transpose.y.vectorized_axis = {z0.id, zo_s_0.id};
    *transpose.y.vectorized_strides = {s1, Symbol(1)};

    af::ascir_op::Store store("store");
    store.x = transpose.y;
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

    return graph.FindNode("transpose");
  }
};

/**
 * @tc.name: CalcTransposeTmpSizeV2_ShouldResolveEachVectorizedAxis_WhenAxesConsistent
 * @tc.number: CalcTransposeTmpSizeV2_Test_001
 * @tc.desc: vectorized_axis 均能在 attr.axis 中找到时，内部按各轴 repeats 正常求解；
 *           这里通过对外接口验证：非连续布局下只取最内轴，tmp = dtype_size * s1。
 */
TEST_F(CalcTransposeTmpSizeV2Test, CalcTransposeTmpSizeV2_ShouldResolveEachVectorizedAxis_WhenAxesConsistent) {
  af::AscGraph graph("test");
  Expression s0;
  Expression s1;
  auto node = BuildGraph(graph, af::DT_FLOAT, s0, s1);
  // 改造成非连续布局：外层步长 s1*1 与内层步长 1 不相等，两轴不合并，只取最内轴 repeats（s1）
  node->outputs[0].attr.vectorized_strides = {Symbol(1), Symbol(1)};
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcTransposeTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->size, sym::Mul(Symbol(4), s1));
}

/**
 * @tc.name: CalcTransposeTmpSizeV2_ShouldReturnOneDesc_WhenLayoutContiguous
 * @tc.number: CalcTransposeTmpSizeV2_Test_002
 * @tc.desc: 连续布局下 tmp = dtype_size * s0 * s1，覆盖 tmp buffer size 日志行
 */
TEST_F(CalcTransposeTmpSizeV2Test, CalcTransposeTmpSizeV2_ShouldReturnOneDesc_WhenLayoutContiguous) {
  af::AscGraph graph("test");
  Expression s0;
  Expression s1;
  auto node = BuildGraph(graph, af::DT_FLOAT, s0, s1);
  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcTransposeTmpSizeV2(*node);
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0]->life_time_axis_id, -1);
  // vectorized_strides = {s1, 1}，内层步长 s1*1 == 外层步长 s1 -> 两轴合并；
  // total_size = sizeof(float) * s0 * s1
  ASSERT_EQ(result[0]->size, sym::Mul(Symbol(4), sym::Mul(s0, s1)));
}

/**
 * @tc.name: CalcTransposeTmpSizeV2_ShouldReturnEmpty_WhenAxisIdInvalid
 * @tc.number: CalcTransposeTmpSizeV2_Test_003
 * @tc.desc: vectorized_axis 含 attr.axis 中不存在的轴 ID 时，内部解析 repeats 的断言分支被触发
 *           （"Incorrect axis ID" 日志），对外表现为返回空描述符集合。
 */
TEST_F(CalcTransposeTmpSizeV2Test, CalcTransposeTmpSizeV2_ShouldReturnEmpty_WhenAxisIdInvalid) {
  af::AscGraph graph("test");
  Expression s0;
  Expression s1;
  auto node = BuildGraph(graph, af::DT_FLOAT, s0, s1);
  auto &attr = node->outputs[0].attr;
  ASSERT_FALSE(attr.vectorized_axis.empty());

  // 取所有合法轴 ID 的最大值 + 1，保证构造出的 ID 一定不在 attr.axis 中
  AxisId invalid_id = *std::max_element(attr.axis.begin(), attr.axis.end());
  invalid_id += 1;
  ASSERT_EQ(std::find(attr.axis.begin(), attr.axis.end(), invalid_id), attr.axis.end());
  attr.vectorized_axis[0] = invalid_id;

  std::vector<std::unique_ptr<af::TmpBufDesc>> result = CalcTransposeTmpSizeV2(*node);
  ASSERT_TRUE(result.empty());
}
}  // namespace ascir
}  // namespace af
