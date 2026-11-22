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

#include <ascendc_ir.h>
#include "ascir.h"
#include <ascir_ops.h>
#include <ascir_utils.h>
#include "ascgraph_info_complete.h"
#include "ascir_ops_utils.h"
#include "codegen.h"
#include "platform_context.h"
#include "runtime_stub.h"

#define private public
#include "optimize.h"
#undef private

using namespace ge;
using namespace af::ops;
using namespace af::ascir_op;

namespace optimize {
namespace {

struct NodeViews {
  std::vector<af::AxisId> axes;
  std::vector<af::Expression> wide_shape;
  std::vector<af::Expression> wide_strides;
  std::vector<af::Expression> scalar_shape;
  std::vector<af::Expression> scalar_strides;
  std::vector<af::Expression> concat_shape;
  std::vector<af::Expression> concat_strides;
};

NodeViews SetupViews(af::AscGraph &graph) {
  const auto &a0 = graph.CreateAxis("a0", af::Symbol(241));
  const auto &a1 = graph.CreateAxis("a1", af::Symbol(4));
  NodeViews views;
  views.axes = {a0.id, a1.id};
  views.wide_shape = {af::Symbol(241), af::sym::kSymbolOne};
  views.wide_strides = {af::sym::kSymbolOne, af::sym::kSymbolZero};
  views.scalar_shape = {af::sym::kSymbolOne, af::sym::kSymbolOne};
  views.scalar_strides = {af::sym::kSymbolZero, af::sym::kSymbolZero};
  views.concat_shape = {af::Symbol(241), af::Symbol(4)};
  views.concat_strides = {af::Symbol(4), af::sym::kSymbolOne};
  return views;
}

void SetView(af::AscNodePtr node, const std::vector<af::AxisId> &axes, const std::vector<af::Expression> &repeats,
             const std::vector<af::Expression> &strides) {
  if (node == nullptr) {
    return;
  }
  node->attr.sched.axis = axes;
  node->outputs[0].attr.axis = axes;
  node->outputs[0].attr.repeats = repeats;
  node->outputs[0].attr.strides = strides;
}

af::AscNodePtr FindNode(af::AscGraph &graph, const std::string &name) {
  return graph.FindNode(name.c_str());
}

// 对齐 asc_graph.py 的形态：仅 1 个合法 Pattern B Arange -> Broadcast（输出尾轴退化），
// 4 组 Scalar 分支，concat -> store -> output。调度时 concat 被拆列成多个 store
// （GM 尾轴 stride=4 非连续），驱动对齐策略走 kDiscontinuous；本用例验证 kAligned
// 修复后完整 Optimize + codegen 链路通过。
af::AscGraph BuildAscGraphPyGraph() {
  af::AscGraph graph("asc_graph_py_arange");
  const NodeViews views = SetupViews(graph);

  af::ascir_op::Data data("data", graph);
  data.ir_attr.SetIndex(0);
  data.y.dtype = af::DT_INT64;

  af::ascir_op::Load load("load");
  load.x = data.y;
  load.y.dtype = af::DT_INT64;

  af::ascir_op::Arange arange("arange", graph);
  arange.ir_attr.SetBase(af::Symbol(0));
  arange.ir_attr.SetStep(af::Symbol(12));
  arange.y.dtype = af::DT_INT64;

  af::ascir_op::Broadcast broadcast("broadcast");
  broadcast.x = arange.y;
  broadcast.y.dtype = af::DT_INT64;
  af::ascir_op::Add add("add");
  add.x1 = broadcast.y;
  add.x2 = load.y;
  add.y.dtype = af::DT_INT64;

  std::vector<af::ascir_op::Add> branch_adds;
  for (int32_t i = 0; i < 4; ++i) {
    const std::string idx = std::to_string(i);
    af::ascir_op::Scalar scalar(("scalar" + idx).c_str(), graph);
    scalar.ir_attr.SetValue(std::to_string(i + 4));
    scalar.y.dtype = af::DT_INT64;
    af::ascir_op::Broadcast brc(("broadcast" + std::to_string(i + 1)).c_str());
    brc.x = scalar.y;
    brc.y.dtype = af::DT_INT64;
    af::ascir_op::Add addk(("add" + std::to_string(i + 1)).c_str());
    addk.x1 = add.y;
    addk.x2 = brc.y;
    addk.y.dtype = af::DT_INT64;
    branch_adds.push_back(addk);
  }

  af::ascir_op::Concat concat("concat");
  concat.x = {branch_adds[0].y, branch_adds[1].y, branch_adds[2].y, branch_adds[3].y};
  concat.y.dtype = af::DT_INT64;
  af::ascir_op::Store store("store");
  store.x = concat.y;
  store.y.dtype = af::DT_INT64;
  af::ascir_op::Output output("output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  output.y.dtype = af::DT_INT64;

  SetView(FindNode(graph, "load"), views.axes, views.wide_shape, views.wide_strides);
  SetView(FindNode(graph, "arange"), views.axes, views.wide_shape, views.wide_strides);
  SetView(FindNode(graph, "broadcast"), views.axes, views.wide_shape, views.wide_strides);
  SetView(FindNode(graph, "add"), views.axes, views.wide_shape, views.wide_strides);
  for (int32_t i = 0; i < 4; ++i) {
    SetView(FindNode(graph, "scalar" + std::to_string(i)), views.axes, views.scalar_shape, views.scalar_strides);
    SetView(FindNode(graph, "broadcast" + std::to_string(i + 1)), views.axes, views.wide_shape, views.wide_strides);
    SetView(FindNode(graph, "add" + std::to_string(i + 1)), views.axes, views.wide_shape, views.wide_strides);
  }
  SetView(FindNode(graph, "concat"), views.axes, views.concat_shape, views.concat_strides);
  SetView(FindNode(graph, "store"), views.axes, views.concat_shape, views.concat_strides);
  return graph;
}

}  // namespace

class ArangeConcatKalignedTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v2 = std::make_shared<af::RuntimeStubV2>();
    ge::RuntimeStub::SetInstance(stub_v2);
  }
  void TearDown() override {
    ge::PlatformContext::GetInstance().Reset();
  }
};

// asc_graph.py 形态（Arange + concat 拆列，非连续 GM store）：kAligned 修复后
// 完整 Optimize（schedule）+ codegen 必须成功。
TEST_F(ArangeConcatKalignedTest, FullOptimizeEndToEndSucceeds) {
  af::AscGraph graph = BuildAscGraphPyGraph();
  optimize::Optimizer optimizer(optimize::OptimizerOptions{.graph_type = optimize::GraphType::kFusedAscBackend});
  ascir::FusedScheduledResult fused_result;
  ASSERT_EQ(optimizer.Optimize(graph, fused_result), af::SUCCESS);
  codegen::Codegen codegen(codegen::CodegenOptions{});
  codegen::CodegenResult result;
  EXPECT_EQ(codegen.GenerateForInductor(fused_result, result), af::SUCCESS);
  EXPECT_FALSE(result.kernel.empty());
}

}  // namespace optimize
