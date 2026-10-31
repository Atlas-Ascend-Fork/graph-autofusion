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

#include "ascendc_ir.h"
#include "ascendc_ir_def.h"
#include "ascir_ops.h"
#define private public
#include "optimize.h"
#include "autoschedule/autoschedule.h"
#include "autoschedule/alignment_handler.h"
#include "platform_context.h"
#undef private
#include "ascir_ops_utils.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/compute_graph.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "attr_utils.h"
#include "graph/debug/ge_op_types.h"
#include "autoschedule/axis_group.h"
#include "schedule_utils.h"
#include "attribute_group/attr_group_shape_env.h"
#include "fusion/autofuse_attrs.h"
#include "fused_graph/fused_graph_unfolder.h"
#include "graph/debug/ge_attr_define.h"
#include "task_generator/concat_group_partitioner.h"
#include "expression/testcase/source_stub.h"
#include "util/mem_utils.h"
#include "platform/platform_factory.h"
#include "platform_context.h"
#include "platform/v1/platformv1.h"
#include "platform/v1/alignment_strategy.h"
#include "codegen.h"
#include "ascgraph_info_complete.h"
#include "asc_graph_builder.h"
#include "common/autofuse_backend_spec_api.h"
#include "optimize/graph_pass/softmax_pattern_fusion_pass.h"
#include "optimize/graph_pass/duplicate_elewise_cse_pass.h"

using namespace std;
using namespace af;
using namespace af::ops;
using namespace af::ascir_op;
using af::testing::AscGraphBuilder;
using af::testing::Sym;
using optimize::autoschedule::AxisGroup;

namespace {
std::string ToString(const Expression &e) {
  return std::string(e.Serialize().get());
}
class GraphBuilder {
 public:
  explicit GraphBuilder(const std::string &name) {
    graph_ = std::make_shared<ComputeGraph>(name);
  }

  GraphBuilder(const std::string &name, const std::string &node_type) {
    graph_ = std::make_shared<ComputeGraph>(name);
    node_type_ = node_type;
  }

  NodePtr AddNode(const std::string &name, const std::string &type, const int in_cnt, const int out_cnt,
                  const std::vector<int64_t> shape = {1, 1, 1, 1}) {
    auto tensor_desc = std::make_shared<GeTensorDesc>();
    tensor_desc->SetShape(GeShape(std::move(shape)));
    tensor_desc->SetFormat(FORMAT_NCHW);
    tensor_desc->SetDataType(DT_FLOAT);

    auto op_desc = std::make_shared<OpDesc>(name, (node_type_ == "") ? type : "AscGraph");
    for (std::int32_t i = 0; i < in_cnt; ++i) {
      op_desc->AddInputDesc(tensor_desc->Clone());
    }
    for (std::int32_t i = 0; i < out_cnt; ++i) {
      op_desc->AddOutputDesc(tensor_desc->Clone());
    }
    op_desc->AddInferFunc([](Operator &op) { return GRAPH_SUCCESS; });
    return graph_->AddNode(op_desc);
  }

  void AddDataEdge(const NodePtr &src_node, const std::int32_t src_idx, const NodePtr &dst_node,
                   const std::int32_t dst_idx) {
    GraphUtils::AddEdge(src_node->GetOutDataAnchor(src_idx), dst_node->GetInDataAnchor(dst_idx));
  }

  ComputeGraphPtr GetGraph() {
    graph_->TopologicalSorting();
    return graph_;
  }

 private:
  ComputeGraphPtr graph_;
  std::string node_type_;
};

/**
 *          NetOutput
 *            |
 *          AscBc4
 *            |
 *          AscBc3
 *        /       / \
 *      AscBc1    AscBc2
 *    /   \         /   \.
 * data0  data1   data2 data3
 */
ComputeGraphPtr BuildFusedGraph(const std::string node_type = "") {
  auto builder = GraphBuilder("test1", node_type);
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  af::AttrUtils::SetInt(data2->GetOpDescBarePtr(), "_parent_node_index", 2);
  auto data3 = builder.AddNode("data3", "Data", 0, 1);
  af::AttrUtils::SetInt(data3->GetOpDescBarePtr(), "_parent_node_index", 3);

  auto ascbc1 = builder.AddNode("ascbc1", "AscGraph", 2, 1);
  auto ascbc2 = builder.AddNode("ascbc2", "AscGraph", 2, 2);
  auto ascbc3 = builder.AddNode("ascbc3", "AscGraph", 3, 1);
  auto ascbc4 = builder.AddNode("ascbc4", "AscGraph", 1, 1);

  auto netoutput1 = builder.AddNode("netoutput1", af::NETOUTPUT, 2, 0);

  builder.AddDataEdge(data0, 0, ascbc1, 0);
  builder.AddDataEdge(data1, 0, ascbc1, 1);
  builder.AddDataEdge(data2, 0, ascbc2, 0);
  builder.AddDataEdge(data3, 0, ascbc2, 1);

  builder.AddDataEdge(ascbc1, 0, ascbc3, 0);
  builder.AddDataEdge(ascbc2, 0, ascbc3, 1);
  builder.AddDataEdge(ascbc2, 1, ascbc3, 2);

  builder.AddDataEdge(ascbc3, 0, ascbc4, 0);
  builder.AddDataEdge(ascbc4, 0, netoutput1, 0);

  return builder.GetGraph();
}
/**
 *         NetOutput
 *            |
 *          AscBc3
 *         /    \
 *     AscBc1   AscBc2
 *       |        |
 *     data0    data1
 */
static ComputeGraphPtr BuildFusedPackGraph(const std::string node_type = "") {
  auto builder = GraphBuilder("test2", node_type);
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);
  auto ascbc1 = builder.AddNode("ascbc1", "AscGraph", 1, 1);
  auto ascbc2 = builder.AddNode("ascbc2", "AscGraph", 1, 1);
  auto ascbc3 = builder.AddNode("ascbc3", "AscGraph", 2, 1);

  auto netoutput1 = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);
  builder.AddDataEdge(data0, 0, ascbc1, 0);
  builder.AddDataEdge(data1, 0, ascbc2, 0);
  builder.AddDataEdge(ascbc1, 0, ascbc3, 0);
  builder.AddDataEdge(ascbc2, 0, ascbc3, 1);
  builder.AddDataEdge(ascbc3, 0, netoutput1, 0);

  return builder.GetGraph();
}

/**
 *         NetOutput
 *            |
 *           AscBc2
 *            |   \
 *            |    \
 *            |     \
 *          AscBc1   \
 *         /    \     \
 *       |        |    \
 *     data0    data1  data2
 */
static ComputeGraphPtr BuildConcatBackwardFusion(const std::string node_type = "") {
  auto builder = GraphBuilder("test3", node_type);
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);
  af::AttrUtils::SetInt(data2->GetOpDescBarePtr(), "_parent_node_index", 2);
  auto ascbc1 = builder.AddNode("ascbc1", "AscGraph", 3, 1);
  auto ascbc2 = builder.AddNode("ascbc2", "AscGraph", 2, 1);

  auto netoutput1 = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);
  builder.AddDataEdge(data0, 0, ascbc1, 0);
  builder.AddDataEdge(data1, 0, ascbc1, 1);
  builder.AddDataEdge(data1, 0, ascbc1, 2);
  builder.AddDataEdge(ascbc1, 0, ascbc2, 0);
  builder.AddDataEdge(data2, 0, ascbc2, 1);
  builder.AddDataEdge(ascbc2, 0, netoutput1, 0);

  return builder.GetGraph();
}

static AscGraph BuildAscBackendGraph(const std::string &name, int64_t axis_num = 2) {
  std::vector<Expression> sizes;
  for (int64_t i = 0; i < axis_num; ++i) {
    sizes.push_back(Sym(("s" + std::to_string(i)).c_str()));
  }
  return AscGraphBuilder(name)
      .Loops(sizes)
      .Data(name + "_data", 0, af::DT_INT8)
      .Load(name + "_load", name + "_data")
      .Abs(name + "_abs", name + "_load")
      .Store(name + "_store", name + "_abs")
      .Output(name + "_out", name + "_store", 0, af::DT_FLOAT16)
      .Build();
}

static AscGraph BuildAscBackendGraphTwoInTwoOut(const std::string &name, int64_t axis_num = 2) {
  std::vector<Expression> sizes;
  for (int64_t i = 0; i < axis_num; ++i) {
    sizes.push_back(Sym(("s" + std::to_string(i)).c_str()));
  }
  return AscGraphBuilder(name)
      .Loops(sizes)
      .Data(name + "_data0", 0, af::DT_INT8)
      .Load(name + "_load0", name + "_data0")
      .Data(name + "_data1", 1, af::DT_INT8)
      .Load(name + "_load1", name + "_data1")
      .Add(name + "_add", name + "_load0", name + "_load1")
      .Store(name + "_store0", name + "_add")
      .Output(name + "_out0", name + "_store0", 0, af::DT_FLOAT16)
      .Store(name + "_store1", name + "_add")
      .Output(name + "_out1", name + "_store1", 1, af::DT_FLOAT16)
      .Build();
}

static AscGraph BuildAscBackendGraphTwoInOneOut(const std::string &name, int64_t axis_num = 2) {
  std::vector<Expression> sizes;
  for (int64_t i = 0; i < axis_num; ++i) {
    sizes.push_back(Sym(("s" + std::to_string(i)).c_str()));
  }
  return AscGraphBuilder(name)
      .Loops(sizes)
      .Data(name + "_data0", 0, af::DT_INT8)
      .Load(name + "_load0", name + "_data0")
      .Data(name + "_data1", 1, af::DT_INT8)
      .Load(name + "_load1", name + "_data1")
      .Add(name + "_add", name + "_load0", name + "_load1")
      .Store(name + "_store0", name + "_add")
      .Output(name + "_out0", name + "_store0", 0, af::DT_FLOAT16)
      .Build();
}

static AscGraph BuildOneNodeAscGraph(const std::string &name, const std::string &prefix = "g0") {
  return AscGraphBuilder(name)
      .Loops({Sym("s0"), Sym("s1")})
      .Data(prefix + "sub_data0", 0, {Sym("s0"), Sym("s1")}, {Sym("s1"), af::sym::kSymbolOne})
      .Load(prefix + "load0", prefix + "sub_data0", {Sym("s0"), af::sym::kSymbolOne})
      .Broadcast(prefix + "brc0", prefix + "load0", {Sym("s0"), Sym("s1")})
      .Abs(prefix + "abs0", prefix + "brc0")
      .Store(prefix + "store0", prefix + "abs0")
      .Output(prefix + "out0", prefix + "store0", 0, af::DT_FLOAT16)
      .Build();
}

static AscGraph BuildOneNodeWithReduceAscGraph(const std::string &name, const std::string &prefix = "g0") {
  return AscGraphBuilder(name)
      .Loops({Sym("s0"), Sym("s1")})
      .Data(prefix + "sub_data0", 0, {Sym("s0"), Sym("s1")}, {Sym("s1"), af::sym::kSymbolOne})
      .Load(prefix + "load0", prefix + "sub_data0", {Sym("s0"), Sym("s1")}, {Sym("s1"), af::sym::kSymbolOne})
      .Max(prefix + "max", prefix + "load0", {0})
      .Broadcast(prefix + "brc0", prefix + "max", {Sym("s0"), Sym("s1")})
      .Abs(prefix + "abs0", prefix + "brc0")
      .Store(prefix + "store0", prefix + "abs0")
      .Output(prefix + "out0", prefix + "store0", 0, af::DT_FLOAT16)
      .Build();
}

static AscGraph BuildMidPackAscGraph(const std::string &name) {
  const auto s0 = Sym("s0");
  const auto s2 = Symbol(2);
  const auto s1 = Sym("s1");
  const std::vector<Expression> data_strides = {s1, s1, af::sym::kSymbolOne};

  return AscGraphBuilder(name)
      .Loops({s0, s2, s1})
      .Data("data0", 0, {s0, af::sym::kSymbolOne, s1}, data_strides)
      .Load("load0", "data0", {s0, af::sym::kSymbolOne, s1}, data_strides, Sym("s88"))
      .Data("data1", 1, {s0, af::sym::kSymbolOne, s1}, data_strides)
      .Load("load1", "data1", {s0, af::sym::kSymbolOne, s1}, data_strides)
      .Concat("concat", {"load0", "load1"})
      .Store("store", "concat")
      .Output("out0", "store", 0, af::DT_FLOAT16)
      .Build();
}

static AscGraph BuildConcatPostGraph(const std::string &name) {
  return AscGraphBuilder(name)
      .Loops({Sym("s0")})
      .Data("data0", 0)
      .Load("load0", "data0")
      .Exp("exp0", "load0")
      .Store("store0", "exp0")
      .Output("out0", "store0", 0, af::DT_FLOAT16)
      .Build();
}

NodePtr CreateAscbcToAscGraph(const std::string &name, ComputeGraphPtr &compute_graph, int64_t in_num = 1,
                              int64_t out_num = 1) {
  OpDescBuilder op_desc_builder(name, "AscBackend");
  op_desc_builder.AddDynamicInput("x", in_num);
  op_desc_builder.AddDynamicOutput("y", out_num);
  const auto &op_desc = op_desc_builder.Build();
  auto node = compute_graph->AddNode(op_desc);
  node->SetOwnerComputeGraph(compute_graph);
  return node;
}

/**
 * Output0
 *    |
 *  AscBc3
 *    |
 *  AscBc2
 *    |
 *  AscBc1
 *    |
 *  data0
 */
ComputeGraphPtr BuildFusedAscbc1(const std::string node_type = "") {
  auto g0 = std::make_shared<AscGraph>(BuildAscBackendGraph("g0", 2));
  auto g1 = std::make_shared<AscGraph>(BuildAscBackendGraph("g1", 1));
  auto g2 = std::make_shared<AscGraph>(BuildAscBackendGraph("g2", 2));

  AscGraph fused_asc_graph("fused_graph");

  af::ascir_op::Data data0("data0", fused_asc_graph);
  auto ir_attr = data0.attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
  ir_attr->SetIndex(0);

  auto fused_graph = af::AscGraphUtils::GetComputeGraph(fused_asc_graph);
  auto data_node = fused_asc_graph.FindNode("data0");

  auto ascbc1 = CreateAscbcToAscGraph("ascbc1", fused_graph);
  auto ascbc2 = CreateAscbcToAscGraph("ascbc2", fused_graph);
  auto ascbc3 = CreateAscbcToAscGraph("ascbc3", fused_graph);
  af::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), ascbc1->GetInDataAnchor(0));
  af::GraphUtils::AddEdge(ascbc1->GetOutDataAnchor(0), ascbc2->GetInDataAnchor(0));
  af::GraphUtils::AddEdge(ascbc2->GetOutDataAnchor(0), ascbc3->GetInDataAnchor(0));

  af::ascir_op::Output output("output");
  auto out_ir_attr = output.attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
  out_ir_attr->SetIndex(0);
  auto out_desc = OpDescUtils::GetOpDescFromOperator(output);
  auto output_node = fused_graph->AddNode(out_desc);
  af::GraphUtils::AddEdge(ascbc3->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));

  auto fuse1_attrs = ascbc1->GetOpDesc()->GetOrCreateAttrsGroup<af::AutoFuseAttrs>();
  GE_ASSERT_NOTNULL(fuse1_attrs);
  fuse1_attrs->SetAscGraph(g0);

  auto fuse2_attrs = ascbc2->GetOpDesc()->GetOrCreateAttrsGroup<af::AutoFuseAttrs>();
  GE_ASSERT_NOTNULL(fuse2_attrs);
  fuse2_attrs->SetAscGraph(g1);

  auto fuse3_attrs = ascbc3->GetOpDesc()->GetOrCreateAttrsGroup<af::AutoFuseAttrs>();
  GE_ASSERT_NOTNULL(fuse3_attrs);
  fuse3_attrs->SetAscGraph(g2);
  fused_graph->TopologicalSorting();
  return fused_graph;
}

/**
 *
 *                  Output0
 *                    |
 *                  AscBc3
 *                /     |
 *           AscBc2    / ---Output1
 *        /    \     /
 *     data2  AscBc1
 *            /   \
 *         data0  data1
 */
ComputeGraphPtr BuildFusedAscbc2(const std::string node_type = "") {
  auto g0 = std::make_shared<AscGraph>(BuildAscBackendGraphTwoInTwoOut("g0", 2));
  auto g1 = std::make_shared<AscGraph>(BuildAscBackendGraphTwoInOneOut("g1", 1));
  auto g2 = std::make_shared<AscGraph>(BuildAscBackendGraphTwoInOneOut("g2", 2));

  AscGraph fused_asc_graph("fused_graph");
  af::ascir_op::Data data0("data0", fused_asc_graph);
  auto ir_attr0 = data0.attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
  ir_attr0->SetIndex(0);

  af::ascir_op::Data data1("data1", fused_asc_graph);
  auto ir_attr1 = data1.attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
  ir_attr1->SetIndex(1);

  af::ascir_op::Data data2("data2", fused_asc_graph);
  auto ir_attr2 = data2.attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
  ir_attr2->SetIndex(2);

  auto fused_graph = af::AscGraphUtils::GetComputeGraph(fused_asc_graph);
  auto data0_node = fused_asc_graph.FindNode("data0");
  auto data1_node = fused_asc_graph.FindNode("data1");
  auto data2_node = fused_asc_graph.FindNode("data2");

  auto ascbc1 = CreateAscbcToAscGraph("ascbc1", fused_graph, 2, 2);
  auto ascbc2 = CreateAscbcToAscGraph("ascbc2", fused_graph, 2, 1);
  auto ascbc3 = CreateAscbcToAscGraph("ascbc3", fused_graph, 2, 1);

  af::GraphUtils::AddEdge(data0_node->GetOutDataAnchor(0), ascbc1->GetInDataAnchor(0));
  af::GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), ascbc1->GetInDataAnchor(1));
  af::GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0), ascbc2->GetInDataAnchor(0));
  af::GraphUtils::AddEdge(ascbc1->GetOutDataAnchor(0), ascbc2->GetInDataAnchor(1));
  af::GraphUtils::AddEdge(ascbc2->GetOutDataAnchor(0), ascbc3->GetInDataAnchor(0));
  af::GraphUtils::AddEdge(ascbc1->GetOutDataAnchor(1), ascbc3->GetInDataAnchor(1));

  af::ascir_op::Output output0("output0");
  auto out0_ir_attr = output0.attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
  out0_ir_attr->SetIndex(0);
  auto out0_desc = OpDescUtils::GetOpDescFromOperator(output0);
  auto output0_node = fused_graph->AddNode(out0_desc);

  af::ascir_op::Output output1("output1");
  auto out1_ir_attr = output1.attr.ir_attr->DownCastTo<af::AscDataIrAttrDef>();
  out1_ir_attr->SetIndex(1);
  auto out1_desc = OpDescUtils::GetOpDescFromOperator(output1);
  auto output1_node = fused_graph->AddNode(out1_desc);
  af::GraphUtils::AddEdge(ascbc3->GetOutDataAnchor(0), output0_node->GetInDataAnchor(0));
  af::GraphUtils::AddEdge(ascbc1->GetOutDataAnchor(1), output1_node->GetInDataAnchor(0));

  auto fuse1_attrs = ascbc1->GetOpDesc()->GetOrCreateAttrsGroup<af::AutoFuseAttrs>();
  GE_ASSERT_NOTNULL(fuse1_attrs);
  fuse1_attrs->SetAscGraph(g0);

  auto fuse2_attrs = ascbc2->GetOpDesc()->GetOrCreateAttrsGroup<af::AutoFuseAttrs>();
  GE_ASSERT_NOTNULL(fuse2_attrs);
  fuse2_attrs->SetAscGraph(g1);

  auto fuse3_attrs = ascbc3->GetOpDesc()->GetOrCreateAttrsGroup<af::AutoFuseAttrs>();
  GE_ASSERT_NOTNULL(fuse3_attrs);
  fuse3_attrs->SetAscGraph(g2);
  fused_graph->TopologicalSorting();
  return fused_graph;
}

/**
 *         NetOutput
 *            |
 *          AscBc3
 *        /     /\
 *    AscBc1   AscBc2
 *    /   \   /    \
 * data0  data1   data2
 */
ComputeGraphPtr BuildFusedGraphWithSharedData(const std::string node_type = "") {
  auto builder = GraphBuilder("BuildFusedGraphWithSharedData", node_type);
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDesc(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDesc(), "_parent_node_index", 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  af::AttrUtils::SetInt(data2->GetOpDesc(), "_parent_node_index", 2);

  auto ascbc1 = builder.AddNode("ascbc1", "AscGraph", 2, 1);
  auto ascbc2 = builder.AddNode("ascbc2", "AscGraph", 2, 2);
  auto ascbc3 = builder.AddNode("ascbc3", "AscGraph", 3, 1);

  auto netoutput1 = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc1, 0);
  builder.AddDataEdge(data1, 0, ascbc1, 1);
  builder.AddDataEdge(data1, 0, ascbc2, 0);
  builder.AddDataEdge(data2, 0, ascbc2, 1);

  builder.AddDataEdge(ascbc1, 0, ascbc3, 0);
  builder.AddDataEdge(ascbc2, 0, ascbc3, 1);
  builder.AddDataEdge(ascbc2, 1, ascbc3, 2);

  builder.AddDataEdge(ascbc3, 0, netoutput1, 0);

  return builder.GetGraph();
}

static AscGraph BuildAddAscGraph(const std::string &name) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  return AscGraphBuilder(name)
      .Loops({s0, s1})
      .Data("sub_data0", 0, {s0, s1}, {s1, af::sym::kSymbolOne}, af::DT_INT8)
      .Load("load0", "sub_data0", {s0, s1}, {s1, af::sym::kSymbolOne}, Sym("s9999"))
      .Data("sub_data1", 1, {s0, s1}, {s1, af::sym::kSymbolOne})
      .Load("load1", "sub_data1")
      .Add("add", "load0", "load1")
      .Store("store0", "add")
      .Output("sub_out0", "store0", 0, af::DT_FLOAT16)
      .Build();
}

static AscGraph BuildAddAscGraph2(const std::string &name) {
  const auto s0 = Sym("s0");
  const auto s2 = Sym("s2");
  return AscGraphBuilder(name)
      .Loops({s0, s2})
      .Data("sub_data2", 0, {s0, s2}, {s2, af::sym::kSymbolOne})
      .Load("load2", "sub_data2")
      .Data("sub_data3", 1, {s0, s2}, {s2, af::sym::kSymbolOne})
      .Load("load3", "sub_data3")
      .Add("add1", "load2", "load3")
      .Store("store1", "add1")
      .Output("sub_out1", "store1", 0, af::DT_FLOAT16)
      .Store("store2", "add1")
      .Output("sub_out2", "store2", 1, af::DT_FLOAT16)
      .Build();
}

static AscGraph BuildAddAscGraph3(const std::string &name) {
  auto graph = AscGraphBuilder(name)
                   .Loops({Sym("s0"), Sym("s1")})
                   .Data("sub2_data0", 0)
                   .Load("sub2_load0", "sub2_data0")
                   .Data("sub2_data1", 1)
                   .Load("sub2_load1", "sub2_data1")
                   .Add("sub2_add0", "sub2_load0", "sub2_load1")
                   .Store("sub2_store0", "sub2_add0")
                   .Output("sub2_out0", "sub2_store0", 0, af::DT_FLOAT16)
                   .Store("sub2_store1", "sub2_add0")
                   .Output("sub2_out1", "sub2_store1", 1, af::DT_FLOAT16)
                   .Build();
  graph.SetGraphType(af::AscGraphType::kImplGraph);
  return graph;
}

static AscGraph BuildAddAscGraphAfterConcat(const std::string &name) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  auto graph = AscGraphBuilder(name)
                   .Loops({s0, s1 + s2 + s2})
                   .Data("sub2_data0", 0, {s0, s1 + s2 + s2}, {s1 + s2 + s2, af::sym::kSymbolOne})
                   .Load("sub2_load0", "sub2_data0")
                   .Data("sub2_data1", 1, {s0, s1 + s2 + s2}, {s1 + s2 + s2, af::sym::kSymbolOne})
                   .Load("sub2_load1", "sub2_data1")
                   .Add("sub2_add0", "sub2_load0", "sub2_load1")
                   .Store("sub2_store0", "sub2_add0")
                   .Output("sub2_out0", "sub2_store0", 0, af::DT_FLOAT16)
                   .Build();
  graph.SetGraphType(af::AscGraphType::kImplGraph);
  return graph;
}

static AscGraph BuildConcatAscGraph(const std::string &name, const bool same_data = false) {
  auto ONE = Symbol(1);
  const auto s0 = Symbol("s0");
  const auto s1 = Symbol("s1");
  const auto s2 = Symbol("s2");
  const auto input_dim = same_data ? s1 : s2;

  af::AscGraph graph(name.c_str());
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1 + input_dim + input_dim + input_dim);

  af::ascir_op::Data x1("concat_data0", graph);
  x1.attr.sched.axis = {z0.id, z1.id};
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, ONE};
  x1.ir_attr.SetIndex(0);

  af::ascir_op::Load x1Local("concat_load0");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {z0.id, z1.id};
  *x1Local.y.axis = {z0.id, z1.id};
  *x1Local.y.repeats = {s0, s1};
  *x1Local.y.strides = {s1, ONE};

  af::ascir_op::Data x2("concat_data1", graph);
  x2.attr.sched.axis = {z0.id, z1.id};
  *x2.y.axis = {z0.id, z1.id};
  *x2.y.repeats = {s0, input_dim};
  *x2.y.strides = {input_dim, ONE};
  x2.ir_attr.SetIndex(1);

  af::ascir_op::Load x2Local("concat_load1");
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {z0.id, z1.id};
  *x2Local.y.axis = {z0.id, z1.id};
  *x2Local.y.repeats = {s0, input_dim};
  *x2Local.y.strides = {input_dim, ONE};

  af::ascir_op::Data concat_data2("concat_data2", graph);
  concat_data2.attr.sched.axis = {z0.id, z1.id};
  *concat_data2.y.axis = {z0.id, z1.id};
  *concat_data2.y.repeats = {s0, input_dim};
  *concat_data2.y.strides = {input_dim, ONE};
  concat_data2.ir_attr.SetIndex(2);

  af::ascir_op::Load concat_load2("concat_load2");
  concat_load2.x = concat_data2.y;
  concat_load2.attr.sched.axis = {z0.id, z1.id};
  *concat_load2.y.axis = {z0.id, z1.id};
  *concat_load2.y.repeats = {s0, input_dim};
  *concat_load2.y.strides = {input_dim, ONE};

  af::ascir_op::Concat concat("concat");
  concat.x = {x1Local.y, x2Local.y, concat_load2.y};
  concat.attr.sched.axis = {z0.id, z1.id};
  *concat.y.axis = {z0.id, z1.id};
  *concat.y.repeats = {s0, s1 + input_dim + input_dim};
  *concat.y.strides = {s1 + input_dim + input_dim, ONE};

  af::ascir_op::Store x_out("concat_store");
  x_out.x = concat.y;
  x_out.attr.sched.axis = {z0.id, z1.id};
  *x_out.y.axis = {z0.id, z1.id};
  *x_out.y.repeats = {s0, s1 + input_dim + input_dim};
  *x_out.y.strides = {s1 + input_dim + input_dim, ONE};

  af::ascir_op::Output y("concat_out");
  y.x = x_out.y;
  y.y.dtype = af::DT_FLOAT16;
  y.ir_attr.SetIndex(0);

  return graph;
}

static AscGraph BuildSingleConcatAscGraph(const std::string &name) {
  const auto s0 = Symbol(32);
  const auto s1 = Symbol(3);
  return AscGraphBuilder(name)
      .Loops({s0, s1})
      .Data("concat_data0", 0, {s0, af::sym::kSymbolOne}, {af::sym::kSymbolOne, af::sym::kSymbolZero})
      .Load("concat_load0", "concat_data0", {s0, af::sym::kSymbolOne})
      .Data("concat_data1", 1, {s0, af::sym::kSymbolOne}, {af::sym::kSymbolOne, af::sym::kSymbolZero})
      .Load("concat_load1", "concat_data1", {s0, af::sym::kSymbolOne})
      .Data("concat_data2", 2, {s0, af::sym::kSymbolOne}, {af::sym::kSymbolOne, af::sym::kSymbolZero})
      .Load("concat_load2", "concat_data2", {s0, af::sym::kSymbolOne})
      .Concat("concat", {"concat_load0", "concat_load1", "concat_load2"})
      .Store("concat_store", "concat")
      .Output("concat_out", "concat_store", 0, af::DT_FLOAT16)
      .Build();
}

/**
 *                   store
 *                     |
 *                   mul0
 *                  /   \
 *               add0  exp1
 *              /    \ /
 *    (remove)brc1    \
 *             |      |
 *            exp0   brc0(remove)
 *              \   /
 *              abs0
 *               |
 *             load0
 *              |
 *            data0
 */
static AscGraph BuildRedundantBroadcastGraph(const std::string &name) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  return AscGraphBuilder(name)
      .Loops({s0, s1, s2})
      .Data("data0", 0, {af::sym::kSymbolOne, s1, s2}, {af::sym::kSymbolZero, s2, af::sym::kSymbolOne}, af::DT_FLOAT16)
      .Load("load0", "data0", {af::sym::kSymbolOne, s1, s2}, {af::sym::kSymbolZero, s2, af::sym::kSymbolOne})
      .Abs("abs0", "load0")
      .Exp("exp0", "abs0")
      .Broadcast("brc0", "abs0", {s0, s1, s2})
      .Broadcast("brc1", "exp0", {s0, s1, s2})
      .Add("add0", "brc0", "brc1")
      .Abs("exp1", "brc0")
      .Mul("mul0", "add0", "exp1")
      .Store("store", "mul0")
      .Output("y", "store", 0, af::DT_FLOAT16)
      .Build();
}

static bool CheckReservedTransposeInputSchedAxis(const ::ascir::ImplGraph &impl_graph) {
  const auto transpose_node = impl_graph.FindNode("transpose0");
  if (transpose_node == nullptr) {
    return false;
  }

  for (const auto &input_node : transpose_node->GetInDataNodes()) {
    const auto input_asc_node = std::dynamic_pointer_cast<af::AscNode>(input_node);
    EXPECT_NE(input_asc_node, nullptr);
    if (input_asc_node == nullptr) {
      continue;
    }
    EXPECT_EQ(input_asc_node->attr.sched.axis, transpose_node->attr.sched.axis)
        << "graph=" << impl_graph.GetName() << ", input=" << input_asc_node->GetName();
  }
  return true;
}

template <typename ImplGraphs>
static bool CheckReservedTransposeImplGraphs(const ImplGraphs &impl_graphs) {
  bool found_reserved_transpose = false;
  for (const auto &impl_graph : impl_graphs) {
    found_reserved_transpose = CheckReservedTransposeInputSchedAxis(impl_graph) || found_reserved_transpose;
  }
  return found_reserved_transpose;
}

}  // namespace

static void ConstructSoftmaxGraph(af::AscGraph &graph) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0 * s1 * s2);
  auto z1 = graph.CreateAxis("z1", s3);

  auto axis = {z0.id, z1.id};

  Data arg4_1("arg4_1", graph);
  arg4_1.attr.api.compute_type = ComputeType::kComputeInvalid;
  arg4_1.y.dtype = af::DT_FLOAT16;
  arg4_1.ir_attr.SetIndex(0);

  Load b0_load("b0_load");
  b0_load.x = arg4_1.y;
  b0_load.attr.sched.axis = axis;
  b0_load.y.dtype = af::DT_FLOAT16;
  *b0_load.y.axis = axis;
  *b0_load.y.repeats = {s0 * s1 * s2, s3};
  *b0_load.y.strides = {s3, One};

  af::ascir_op::Max b0_max("b0_max");
  b0_max.x = b0_load.y;
  b0_max.attr.sched.axis = axis;
  b0_max.y.dtype = af::DT_FLOAT16;
  *b0_max.y.axis = axis;
  *b0_max.y.repeats = {s0 * s1 * s2, One};
  *b0_max.y.strides = {One, Zero};

  Broadcast b1_broadcast("b1_broadcast");
  b1_broadcast.x = b0_max.y;
  b1_broadcast.attr.sched.axis = axis;
  b1_broadcast.y.dtype = af::DT_FLOAT16;
  *b1_broadcast.y.axis = axis;
  *b1_broadcast.y.repeats = {s0 * s1 * s2, s3};
  *b1_broadcast.y.strides = {s3, One};

  af::ascir_op::Sub b1_sub("b1_sub");
  b1_sub.x1 = b0_load.y;
  b1_sub.x2 = b1_broadcast.y;
  b1_sub.attr.sched.axis = axis;
  b1_sub.y.dtype = af::DT_FLOAT16;
  *b1_sub.y.axis = axis;
  *b1_sub.y.repeats = {s0 * s1 * s2, s3};
  *b1_sub.y.strides = {s3, One};

  Exp b1_exp("b1_exp");
  b1_exp.x = b1_sub.y;
  b1_exp.attr.sched.axis = axis;
  b1_exp.y.dtype = af::DT_FLOAT16;
  *b1_exp.y.axis = axis;
  *b1_exp.y.repeats = {s0 * s1 * s2, s3};
  *b1_exp.y.strides = {s3, One};

  Sum b2_sum("b2_sum");
  b2_sum.x = b1_exp.y;
  b2_sum.attr.sched.axis = axis;
  b2_sum.y.dtype = af::DT_FLOAT16;
  *b2_sum.y.axis = axis;
  *b2_sum.y.repeats = {s0 * s1 * s2, One};
  *b2_sum.y.strides = {One, Zero};

  Output buf3("buf3");
  buf3.ir_attr.SetIndex(2);

  Broadcast b3_broadcast("b3_broadcast");
  b3_broadcast.x = b2_sum.y;
  b3_broadcast.attr.sched.axis = axis;
  b3_broadcast.y.dtype = af::DT_FLOAT16;
  *b3_broadcast.y.axis = axis;
  *b3_broadcast.y.repeats = {s0 * s1 * s2, s3};
  *b3_broadcast.y.strides = {s3, One};

  af::ascir_op::Div b3_div("b3_div");
  b3_div.x1 = b1_exp.y;
  b3_div.x2 = b3_broadcast.y;
  b3_div.attr.sched.axis = axis;
  b3_div.y.dtype = af::DT_FLOAT16;
  *b3_div.y.axis = axis;
  *b3_div.y.repeats = {s0 * s1 * s2, s3};
  *b3_div.y.strides = {s3, One};

  Store b3_store("b3_store");
  b3_store.x = b3_div.y;
  b3_store.attr.sched.axis = axis;
  b3_store.y.dtype = af::DT_FLOAT16;
  *b3_store.y.axis = axis;
  *b3_store.y.repeats = {s0 * s1 * s2, s3};
  *b3_store.y.strides = {s3, One};

  buf3.x = b3_store.y;
  buf3.y.dtype = af::DT_FLOAT16;
}

class OptimizerSt : public ::testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::PlatformContext::GetInstance().Reset();
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::PlatformContext::GetInstance().Reset();
  }

  optimize::Optimizer optimizer;

  OptimizerSt() : optimizer(optimize::OptimizerOptions{}) {}

  static std::string ExpressToStr(std::vector<af::Expression> exprs) {
    std::stringstream ss;
    for (auto &size_expr : exprs) {
      ss << std::string(size_expr.Str().get()) << ", ";
    }
    return ss.str();
  }

  static std::string RepeatsToStr(const af::AscGraph &graph, const char *node_name) {
    auto node = graph.FindNode(node_name);
    if (node == nullptr) {
      return "";
    }
    return ExpressToStr(node->outputs[0].attr.repeats);
  }

  static std::string StridesToStr(const af::AscGraph &graph, const char *node_name) {
    auto node = graph.FindNode(node_name);
    if (node == nullptr) {
      return "";
    }
    return ExpressToStr(node->outputs[0].attr.strides);
  }

  static std::string AxisToStr(af::AscGraph &graph, const char *node_name) {
    auto node = graph.FindNode(node_name);
    if (node == nullptr) {
      return "";
    }
    std::stringstream ss;
    for (auto axis_id : node->outputs[0].attr.axis) {
      ss << graph.FindAxis(axis_id)->name << ", ";
    }
    return ss.str();
  }

  class AlignmentStrategyShadow : public optimize::AlignmentStrategy {
   public:
    AlignmentStrategyShadow() {
      AlignmentStrategy::InitAlignmentInferFunc();
    }

    af::Status AccessSetAlignWidth(const ::ascir::ImplGraph &impl_graph) {
      return SetAlignWidth(impl_graph);
    }

    af::Status AccessAddRemovePadForTailAxisDiscontinuousLoad(::ascir::ImplGraph &impl_graph) {
      return ForEachNode(impl_graph, &AlignmentStrategyShadow::AddRemovePadForOneNode);
    }
    af::Status AccessAddPadForAlignmentConflictNode(::ascir::ImplGraph &impl_graph) {
      return ForEachNode(impl_graph, &AlignmentStrategyShadow::AddPadForAlignmentConflictOneNode);
    }
    af::Status AccessInferAlignment(::ascir::ImplGraph &impl_graph) {
      return ForEachNode(impl_graph, &AlignmentStrategyShadow::InferAlignmentForOneNode);
    }
    // 当前tensor的对齐行为只会出现在尾轴,如果没有新的对齐行为或者类型,该函数不应该修改
    af::Status AccessSetVectorizedStrides(::ascir::ImplGraph &impl_graph) {
      return ForEachNode(impl_graph, &AlignmentStrategyShadow::SetVectorizedStridesForOneNode);
    }
    using optimize::BaseAlignmentStrategy::AlignmentTypeToString;
  };
};

namespace optimize {

TEST_F(OptimizerSt, TestSoftmaxGraph_OptimizeSuccess) {
  af::AscGraph graph("SoftmaxGraph");
  ConstructSoftmaxGraph(graph);
  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 3UL);
}

TEST_F(OptimizerSt, TestPackGraph_OptimizeSuccess) {
  ComputeGraphPtr compute_graph = BuildFusedPackGraph();
  ASSERT_NE(compute_graph, nullptr);
  auto ascbc1 = compute_graph->FindNode("ascbc1");
  ASSERT_NE(ascbc1, nullptr);
  auto ascbc2 = compute_graph->FindNode("ascbc2");
  ASSERT_NE(ascbc2, nullptr);
  auto ascbc3 = compute_graph->FindNode("ascbc3");
  ASSERT_NE(ascbc3, nullptr);
  auto subgraph1 = BuildOneNodeAscGraph("sub1", "g1");
  auto subgraph2 = BuildOneNodeAscGraph("sub2", "g2");
  auto subgraph3 = BuildMidPackAscGraph("sub3");

  std::string add_graph_str1;
  af::AscGraphUtils::SerializeToReadable(subgraph1, add_graph_str1);
  af::AttrUtils::SetStr(ascbc1->GetOpDescBarePtr(), "ascgraph", add_graph_str1);
  std::string add_graph_str2;
  af::AscGraphUtils::SerializeToReadable(subgraph2, add_graph_str2);
  af::AttrUtils::SetStr(ascbc2->GetOpDescBarePtr(), "ascgraph", add_graph_str2);
  std::string add_graph_str3;
  af::AscGraphUtils::SerializeToReadable(subgraph3, add_graph_str3);
  af::AttrUtils::SetStr(ascbc3->GetOpDescBarePtr(), "ascgraph", add_graph_str3);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  ASSERT_EQ(optimizer.Optimize(compute_graph, fused_scheduled_result), 0);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 3UL);

  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
  // 1 aligned + 1 not aligned
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups[0].impl_graphs.size(), 2UL);

  auto impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  auto ascbc_1 = impl_graph.FindNode("ascbc1");
  EXPECT_EQ(ascbc_1, nullptr);
  auto ascbc_2 = impl_graph.FindNode("ascbc2");
  EXPECT_EQ(ascbc_2, nullptr);
  auto ascbc_3 = impl_graph.FindNode("ascbc3");
  EXPECT_EQ(ascbc_3, nullptr);
}

TEST_F(OptimizerSt, TestPackGraph_OptimizeSuccessWithReduce) {
  ComputeGraphPtr compute_graph = BuildFusedPackGraph();
  ASSERT_NE(compute_graph, nullptr);
  auto ascbc1 = compute_graph->FindNode("ascbc1");
  ASSERT_NE(ascbc1, nullptr);
  auto ascbc2 = compute_graph->FindNode("ascbc2");
  ASSERT_NE(ascbc2, nullptr);
  auto ascbc3 = compute_graph->FindNode("ascbc3");
  ASSERT_NE(ascbc3, nullptr);
  auto subgraph1 = BuildOneNodeWithReduceAscGraph("sub1", "g1");
  auto subgraph2 = BuildOneNodeWithReduceAscGraph("sub2", "g2");
  auto subgraph3 = BuildMidPackAscGraph("sub3");

  std::string add_graph_str1;
  af::AscGraphUtils::SerializeToReadable(subgraph1, add_graph_str1);
  af::AttrUtils::SetStr(ascbc1->GetOpDescBarePtr(), "ascgraph", add_graph_str1);
  std::string add_graph_str2;
  af::AscGraphUtils::SerializeToReadable(subgraph2, add_graph_str2);
  af::AttrUtils::SetStr(ascbc2->GetOpDescBarePtr(), "ascgraph", add_graph_str2);
  std::string add_graph_str3;
  af::AscGraphUtils::SerializeToReadable(subgraph3, add_graph_str3);
  af::AttrUtils::SetStr(ascbc3->GetOpDescBarePtr(), "ascgraph", add_graph_str3);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  ASSERT_EQ(optimizer.Optimize(compute_graph, fused_scheduled_result), af::SUCCESS);
}

TEST_F(OptimizerSt, TestConcatGraph_OptimizeSuccess) {
  ComputeGraphPtr compute_graph = BuildFusedGraph();
  ASSERT_NE(compute_graph, nullptr);

  auto ascbc1 = compute_graph->FindNode("ascbc1");
  ASSERT_NE(ascbc1, nullptr);
  auto ascbc2 = compute_graph->FindNode("ascbc2");
  ASSERT_NE(ascbc2, nullptr);
  auto ascbc3 = compute_graph->FindNode("ascbc3");
  ASSERT_NE(ascbc3, nullptr);
  auto ascbc4 = compute_graph->FindNode("ascbc4");
  ASSERT_NE(ascbc4, nullptr);

  auto add_sub_graph1 = BuildAddAscGraph("add1");
  auto add_sub_graph2 = BuildAddAscGraph2("add2");
  auto concat_sub_graph = BuildConcatAscGraph("concat");
  auto concat_post_graph = BuildConcatPostGraph("concat_post");

  std::string add_graph_str1;
  af::AscGraphUtils::SerializeToReadable(add_sub_graph1, add_graph_str1);
  af::AttrUtils::SetStr(ascbc1->GetOpDescBarePtr(), "ascgraph", add_graph_str1);
  std::string add_graph_str2;
  af::AscGraphUtils::SerializeToReadable(add_sub_graph2, add_graph_str2);
  af::AttrUtils::SetStr(ascbc2->GetOpDescBarePtr(), "ascgraph", add_graph_str2);
  std::string concat_graph_str;
  af::AscGraphUtils::SerializeToReadable(concat_sub_graph, concat_graph_str);
  af::AttrUtils::SetStr(ascbc3->GetOpDescBarePtr(), "ascgraph", concat_graph_str);

  std::string concat_post_graph_str;
  af::AscGraphUtils::SerializeToReadable(concat_post_graph, concat_post_graph_str);
  af::AttrUtils::SetStr(ascbc4->GetOpDescBarePtr(), "ascgraph", concat_post_graph_str);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  ASSERT_EQ(optimizer.Optimize(compute_graph, fused_scheduled_result), 0);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.origin_vars.size(), 4UL);
  EXPECT_EQ(ToString(fused_scheduled_result.origin_vars[0]), "s0");
  EXPECT_EQ(ToString(fused_scheduled_result.origin_vars[1]), "s1");
  EXPECT_EQ(ToString(fused_scheduled_result.origin_vars[2]), "s2");
  EXPECT_EQ(ToString(fused_scheduled_result.origin_vars[3]), "s9999");

  auto impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  auto ascbc_1 = impl_graph.FindNode("ascbc1");
  EXPECT_EQ(ascbc_1, nullptr);
  auto ascbc_2 = impl_graph.FindNode("ascbc2");
  EXPECT_EQ(ascbc_2, nullptr);
  auto ascbc_3 = impl_graph.FindNode("ascbc3");
  EXPECT_EQ(ascbc_3, nullptr);

  // load0
  std::string load0_repeats = RepeatsToStr(impl_graph, "load0");
  std::string load0_strides = StridesToStr(impl_graph, "load0");
  std::string load0_axes = AxisToStr(impl_graph, "load0");
  EXPECT_EQ(load0_repeats, "(s0 / (z0Tb_size * z0t_size)), z0Tb_size, z0t_size, s1, ");
  EXPECT_EQ(load0_strides, "(s1 * z0Tb_size * z0t_size), (s1 * z0t_size), s1, 1, ");
  EXPECT_EQ(load0_axes, "z0TB, z0Tb, z0t, z1, ");

  // concat
  std::string max0_repeats = RepeatsToStr(impl_graph, "concat");
  std::string max0_strides = StridesToStr(impl_graph, "concat");
  std::string max0_axes = AxisToStr(impl_graph, "concat");
  EXPECT_EQ(max0_repeats, "(s0 / (z0Tb_size * z0t_size)), z0Tb_size, z0t_size, ((2 * s2) + s1), ");
  EXPECT_EQ(max0_strides,
            "(((2 * s2) + s1) * z0Tb_size * z0t_size), (((2 * s2) + s1) * z0t_size), ((2 * s2) + s1), 1, ");
  EXPECT_EQ(max0_axes, "z0TB, z0Tb, z0t, z1, ");

  // concat post node
  std::string broadcast1_repeats = RepeatsToStr(impl_graph, "exp0");
  std::string broadcast1_strides = StridesToStr(impl_graph, "exp0");
  std::string broadcast1_axes = AxisToStr(impl_graph, "exp0");
  EXPECT_EQ(broadcast1_repeats, "(s0 / (z0Tb_size * z0t_size)), z0Tb_size, z0t_size, 1, ");
  EXPECT_EQ(broadcast1_strides, "(z0Tb_size * z0t_size), z0t_size, 1, 0, ");
  EXPECT_EQ(broadcast1_axes, "z0TB, z0Tb, z0t, z1, ");
}

TEST_F(OptimizerSt, TestSingleConcatGraph_OptimizeSuccess) {
  auto builder = GraphBuilder("test_single_cat");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  af::AttrUtils::SetInt(data2->GetOpDescBarePtr(), "_parent_node_index", 2);
  auto ascg1 = builder.AddNode("ascbc1", "AscGraph", 3, 1);
  auto netoutput1 = builder.AddNode("netoutput1", af::NETOUTPUT, 2, 0);
  builder.AddDataEdge(data0, 0, ascg1, 0);
  builder.AddDataEdge(data1, 0, ascg1, 1);
  builder.AddDataEdge(data2, 0, ascg1, 2);
  builder.AddDataEdge(ascg1, 0, netoutput1, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  ASSERT_NE(compute_graph, nullptr);

  auto ascbc1 = compute_graph->FindNode("ascbc1");
  auto concat_sub_graph = BuildSingleConcatAscGraph("concat");
  std::string add_graph_str1;
  af::AscGraphUtils::SerializeToReadable(concat_sub_graph, add_graph_str1);
  af::AttrUtils::SetStr(ascbc1->GetOpDescBarePtr(), "ascgraph", add_graph_str1);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  ASSERT_EQ(optimizer.Optimize(compute_graph, fused_scheduled_result), 0);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.input_nodes.size(), 3UL);
  EXPECT_EQ(fused_scheduled_result.output_nodes.size(), 1UL);
  auto impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  auto ascbc_1 = impl_graph.FindNode("ascbc1");
  EXPECT_EQ(ascbc_1, nullptr);
  // load0
  std::string load0_repeats = RepeatsToStr(impl_graph, "concat_load0");
  std::string load0_strides = StridesToStr(impl_graph, "concat_load0");
  std::string load0_axes = AxisToStr(impl_graph, "concat_load0");
  EXPECT_EQ(load0_repeats, "(32 / (z0Tb_size * z0t_size)), z0Tb_size, z0t_size, 1, ");
  EXPECT_EQ(load0_strides, "(z0Tb_size * z0t_size), z0t_size, 1, 0, ");
  EXPECT_EQ(load0_axes, "z0TB, z0Tb, z0t, z1, ");

  // concat
  std::string max0_repeats = RepeatsToStr(impl_graph, "concat");
  std::string max0_strides = StridesToStr(impl_graph, "concat");
  std::string max0_axes = AxisToStr(impl_graph, "concat");
  EXPECT_EQ(max0_repeats, "(32 / (z0Tb_size * z0t_size)), z0Tb_size, z0t_size, 3, ");
  EXPECT_EQ(max0_strides, "(3 * z0Tb_size * z0t_size), (3 * z0t_size), 3, 1, ");
  EXPECT_EQ(max0_axes, "z0TB, z0Tb, z0t, z1, ");

  // store
  std::string broadcast1_repeats = RepeatsToStr(impl_graph, "concat_store");
  std::string broadcast1_strides = StridesToStr(impl_graph, "concat_store");
  std::string broadcast1_axes = AxisToStr(impl_graph, "concat_store");
  EXPECT_EQ(broadcast1_repeats, "(32 / (z0Tb_size * z0t_size)), z0Tb_size, z0t_size, 3, ");
  EXPECT_EQ(broadcast1_strides, "(3 * z0Tb_size * z0t_size), (3 * z0t_size), 3, 1, ");
  EXPECT_EQ(broadcast1_axes, "z0TB, z0Tb, z0t, z1, ");
}

TEST_F(OptimizerSt, TestFusedAscBackend_ReduceLike_OptimizeSuccess) {
  ComputeGraphPtr compute_graph = BuildFusedAscbc1();
  ASSERT_NE(compute_graph, nullptr);
  optimize::Optimizer opt(OptimizerOptions{.graph_type = GraphType::kFusedAscBackend});
  ::ascir::FusedScheduledResult fused_scheduled_result;
  ASSERT_EQ(opt.Optimize(compute_graph, fused_scheduled_result), 0);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.fused_graph_name.GetString(), compute_graph->GetName());
  ASSERT_EQ(fused_scheduled_result.origin_vars.size(), 2UL);
  EXPECT_EQ(ToString(fused_scheduled_result.origin_vars[0]), "s0");
  EXPECT_EQ(ToString(fused_scheduled_result.origin_vars[1]), "s1");
  ASSERT_EQ(fused_scheduled_result.input_nodes.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.output_nodes.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.workspace_nodes.size(), 2UL);

  // check workspace's tensor id
  auto ws0_0 = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].FindNode(
      "fused_workspace0");
  ASSERT_NE(ws0_0, nullptr);
  auto ws1_0 = fused_scheduled_result.node_idx_to_scheduled_results[1][0].schedule_groups[0].impl_graphs[0].FindNode(
      "fused_workspace0");
  ASSERT_NE(ws1_0, nullptr);
  auto ws1_1 = fused_scheduled_result.node_idx_to_scheduled_results[1][0].schedule_groups[0].impl_graphs[0].FindNode(
      "fused_workspace1");
  ASSERT_NE(ws1_1, nullptr);
  auto ws2_1 = fused_scheduled_result.node_idx_to_scheduled_results[2][0].schedule_groups[0].impl_graphs[0].FindNode(
      "fused_workspace1");
  ASSERT_NE(ws2_1, nullptr);
  EXPECT_EQ(ws0_0->inputs[0].attr.mem.tensor_id, ws1_0->outputs[0].attr.mem.tensor_id);
  EXPECT_NE(ws1_0->outputs[0].attr.mem.tensor_id, ws1_1->inputs[0].attr.mem.tensor_id);
  EXPECT_EQ(ws1_1->inputs[0].attr.mem.tensor_id, ws2_1->outputs[0].attr.mem.tensor_id);
}

TEST_F(OptimizerSt, TestFusedAscBackend_MultiIO_OptimizeSuccess) {
  ComputeGraphPtr compute_graph = BuildFusedAscbc2();
  ASSERT_NE(compute_graph, nullptr);
  optimize::Optimizer opt(OptimizerOptions{.graph_type = GraphType::kFusedAscBackend});
  ::ascir::FusedScheduledResult fused_scheduled_result;
  ASSERT_EQ(opt.Optimize(compute_graph, fused_scheduled_result), 0);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.fused_graph_name.GetString(), compute_graph->GetName());
  ASSERT_EQ(fused_scheduled_result.origin_vars.size(), 2UL);
  EXPECT_EQ(ToString(fused_scheduled_result.origin_vars[0]), "s0");
  EXPECT_EQ(ToString(fused_scheduled_result.origin_vars[1]), "s1");
  ASSERT_EQ(fused_scheduled_result.input_nodes.size(), 3UL);
  ASSERT_EQ(fused_scheduled_result.output_nodes.size(), 2UL);
  ASSERT_EQ(fused_scheduled_result.workspace_nodes.size(), 2UL);
  // check workspace's tensor id
  auto ws0_0 = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].FindNode(
      "fused_workspace0");
  ASSERT_NE(ws0_0, nullptr);
  auto ws1_0 = fused_scheduled_result.node_idx_to_scheduled_results[1][0].schedule_groups[0].impl_graphs[0].FindNode(
      "fused_workspace0");
  ASSERT_NE(ws1_0, nullptr);
  auto ws1_1 = fused_scheduled_result.node_idx_to_scheduled_results[1][0].schedule_groups[0].impl_graphs[0].FindNode(
      "fused_workspace1");
  ASSERT_NE(ws1_1, nullptr);
  auto ws2_1 = fused_scheduled_result.node_idx_to_scheduled_results[2][0].schedule_groups[0].impl_graphs[0].FindNode(
      "fused_workspace1");
  ASSERT_NE(ws2_1, nullptr);
  auto ws2_fused =
      fused_scheduled_result.node_idx_to_scheduled_results[2][0].schedule_groups[0].impl_graphs[0].FindNode(
          "fused_workspaceg2_data1");
  ASSERT_NE(ws2_fused, nullptr);
  EXPECT_EQ(ws0_0->inputs[0].attr.mem.tensor_id, ws1_0->outputs[0].attr.mem.tensor_id);
  EXPECT_NE(ws1_0->outputs[0].attr.mem.tensor_id, ws1_1->inputs[0].attr.mem.tensor_id);
  EXPECT_EQ(ws1_1->inputs[0].attr.mem.tensor_id, ws2_1->outputs[0].attr.mem.tensor_id);
  auto out1 =
      fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].FindNode("g0_out1");
  ASSERT_NE(out1, nullptr);
  EXPECT_EQ(ws2_fused->outputs[0].attr.mem.tensor_id, out1->outputs[0].attr.mem.tensor_id);
  auto store1 = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].FindNode(
      "g0_store1");
  ASSERT_NE(store1, nullptr);
  EXPECT_EQ(ws2_fused->outputs[0].attr.mem.tensor_id, store1->outputs[0].attr.mem.tensor_id);
}

// st无法覆盖到,待transpose相关代码全部合入后删除
TEST_F(OptimizerSt, RemoveDuplicatedAxisFromGroup) {
  af::AscGraph graph("reorder_vectorized_axes");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  AxisGroup axes_group;
  axes_group.x_group = {z0.id, z2.id, z4.id};
  axes_group.y_group = {z1.id, z3.id, z4.id};
  axes_group.axes_order = {0, 2, 4, 1, 3, 4};

  // remove x
  optimize::autoschedule::TilingCase case1;
  case1.ub_tiling_id_x = z0.id;
  case1.ub_tiling_id_y = z1.id;
  optimize::autoschedule::Scheduler sch1(graph, axes_group, case1);
  sch1.RemoveDuplicatedAxisFromGroup();
  std::vector<int64_t> golden_x1 = {z0.id, z2.id};
  std::vector<int64_t> golden_y1 = {z1.id, z3.id, z4.id};
  std::vector<size_t> golden_order1 = {0, 2, 1, 3, 4};
  EXPECT_EQ(sch1.axes_group_.x_group, golden_x1);
  EXPECT_EQ(sch1.axes_group_.y_group, golden_y1);
  EXPECT_EQ(sch1.axes_group_.axes_order, golden_order1);

  optimize::autoschedule::TilingCase case2;
  case2.ub_tiling_id_x = z0.id;
  case2.ub_tiling_id_y = z4.id;
  optimize::autoschedule::Scheduler sch2(graph, axes_group, case2);
  sch2.RemoveDuplicatedAxisFromGroup();
  std::vector<int64_t> golden_x2 = {z0.id, z2.id};
  std::vector<int64_t> golden_y2 = {z1.id, z3.id, z4.id};
  std::vector<size_t> golden_order2 = {0, 2, 1, 3, 4};
  EXPECT_EQ(sch2.axes_group_.x_group, golden_x2);
  EXPECT_EQ(sch2.axes_group_.y_group, golden_y2);
  EXPECT_EQ(sch2.axes_group_.axes_order, golden_order2);

  optimize::autoschedule::TilingCase case3;
  case3.ub_tiling_id_x = z4.id;
  case3.ub_tiling_id_y = z3.id;
  optimize::autoschedule::Scheduler sch3(graph, axes_group, case3);
  sch3.RemoveDuplicatedAxisFromGroup();
  std::vector<int64_t> golden_x3 = {z0.id, z2.id, z4.id};
  std::vector<int64_t> golden_y3 = {z1.id, z3.id};
  std::vector<size_t> golden_order3 = {0, 2, 4, 1, 3};
  EXPECT_EQ(sch3.axes_group_.x_group, golden_x3);
  EXPECT_EQ(sch3.axes_group_.y_group, golden_y3);
  EXPECT_EQ(sch3.axes_group_.axes_order, golden_order3);
}

TEST_F(OptimizerSt, ElewiseAndBrcCanMerge) {
  af::AscGraph graph1("graph1");
  graph1.SetGraphType(af::AscGraphType::kImplGraph);
  auto ONE = Symbol(1);
  const Expression s0 = graph1.CreateSizeVar("s0");
  const Expression s1 = graph1.CreateSizeVar("s1");
  auto z0 = graph1.CreateAxis("z0", s0);
  auto z1 = graph1.CreateAxis("z1", s1);
  af::ascir_op::Data data0("data0", graph1);
  data0.ir_attr.SetIndex(0);
  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id};
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, ONE};
  af::ascir_op::Output out0("out0");
  out0.x = load0.y;
  out0.y.dtype = af::DT_FLOAT16;
  out0.ir_attr.SetIndex(0);

  af::AscGraph graph2("graph2");
  graph2.SetGraphType(af::AscGraphType::kImplGraph);
  const Expression s1_0 = graph1.CreateSizeVar("s0");
  auto z1_0 = graph1.CreateAxis("z0", s1_0);
  af::ascir_op::Data data1_0("data1_0", graph2);
  data1_0.ir_attr.SetIndex(0);
  af::ascir_op::Load load1_0("load1_0");
  load1_0.x = data1_0.y;
  load1_0.attr.sched.axis = {z0.id};
  *load1_0.y.axis = {z0.id};
  *load1_0.y.repeats = {s0};
  *load1_0.y.strides = {ONE};
  af::ascir_op::Output out1_0("out1_0");
  out1_0.x = load1_0.y;
  out1_0.y.dtype = af::DT_FLOAT16;
  out1_0.ir_attr.SetIndex(0);

  AxisGroup lhs;
  EXPECT_EQ(GenAscGraphAxisGroup(graph1, lhs), 0);

  AxisGroup rhs;
  EXPECT_EQ(GenAscGraphAxisGroup(graph2, rhs), 0);
  // CanFuse do axis-mapping
  rhs.y_group.emplace_back(1);

  AxisGroup res;
  EXPECT_TRUE(autoschedule::CanMergeAxisGroup(lhs, rhs, res));

  EXPECT_EQ(res, lhs);
}

TEST_F(OptimizerSt, DoTilingOk) {
  // 当前ST构造不出xgroup、group，先手动构造，待后续支持reduce后删除
  af::AscGraph graph("apply_tiling_pk");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);
  af::ascir_op::Data data("data", graph);
  data.y.dtype = af::DT_FLOAT16;
  data.attr.api.compute_type = ComputeType::kComputeInvalid;
  data.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load("load_i");
  load.x = data.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load.y.repeats = {s0, s1, s2, s3, s4};
  *load.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, af::ops::One};
  load.attr.api.compute_type = ComputeType::kComputeLoad;

  af::ascir_op::Max max("max");
  max.x = load.y;
  max.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *max.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *max.y.repeats = {s0, af::ops::One, s2, One, s4};
  *max.y.strides = {s2 * s4, af::ops::Zero, s4, Zero, af::ops::One};
  max.attr.api.compute_type = ComputeType::kComputeReduce;

  af::ascir_op::Store store("store");
  store.x = max.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.y.dtype = af::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store.y.repeats = {s0, af::ops::One, s2, One, s4};
  *store.y.strides = {s2 * s4, af::ops::Zero, s4, Zero, af::ops::One};

  af::ascir_op::Output y("y");
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};

  std::vector<autoschedule::AutoScheduleOutput> schedule_outputs;
  optimize::autoschedule::AutoSchedule autoschedule(graph, schedule_outputs);
  autoschedule.axes_group_.x_group = {z0.id};
  autoschedule.axes_group_.y_group = {z4.id};
  autoschedule.axes_group_.r_group = {z1.id, z3.id};
  autoschedule.axes_group_.n_group = {z2.id};
  autoschedule.axes_group_.axes_order = {0, 4, 1, 3};

  std::vector<optimize::autoschedule::TilingCase> tiling_cases;
  autoschedule.GenTilingCase(tiling_cases);
  EXPECT_EQ(tiling_cases.size(), 2UL);

  optimize::autoschedule::Scheduler scheduler(graph, autoschedule.axes_group_, tiling_cases[0UL]);
  EXPECT_EQ(scheduler.DoScheduler(), 0);
}

TEST_F(OptimizerSt, ReduceCanMergeMock) {
  // 当前ST构造不出rgroup，先手动构造，待后续支持reduce后删除
  AxisGroup lhs;
  lhs.y_group = {0, 1, 2, 3};
  lhs.axes_order = {0, 1, 2, 3};
  AxisGroup rhs;
  rhs.y_group = {2, 1};
  rhs.r_group = {0, 3};
  rhs.axes_order = {2, 1, 0, 3};
  AxisGroup res1;
  ASSERT_TRUE(CanMergeAxisGroup(lhs, rhs, res1));

  AxisGroup res2;
  ASSERT_TRUE(CanMergeAxisGroup(rhs, lhs, res2));

  AxisGroup lhs1;
  lhs1.y_group = {2, 1};
  lhs1.r_group = {0, 3};
  lhs1.axes_order = {2, 1, 0, 3};
  AxisGroup rhs1;
  rhs1.y_group = {1, 2};
  rhs1.r_group = {3, 0};
  rhs1.axes_order = {0, 1, 2, 3};
  AxisGroup res3;
  ASSERT_TRUE(CanMergeAxisGroup(lhs1, rhs1, res3));
}

/**
 *         NetOutput
 *            |
 *          AscBc3
 *        /    /\
 *    AscBc1  AscBc2
 *    /   \   /    \
 * data0  data1   data2
 */
TEST_F(OptimizerSt, AscBcNodeUnfolder_With_Same_Data_Same_Load) {
  ComputeGraphPtr compute_graph = BuildFusedGraphWithSharedData();
  ASSERT_NE(compute_graph, nullptr);
  std::map<af::Node *, af::AscGraph> asc_backend_to_asc_graph;

  auto ascbc1 = compute_graph->FindNode("ascbc1");
  ASSERT_NE(ascbc1, nullptr);
  auto ascbc2 = compute_graph->FindNode("ascbc2");
  ASSERT_NE(ascbc2, nullptr);
  auto ascbc3 = compute_graph->FindNode("ascbc3");
  ASSERT_NE(ascbc3, nullptr);

  auto add_sub_graph1 = BuildAddAscGraph("sub1_add");
  auto add_sub_graph2 = BuildAddAscGraph3("sub2_add");
  auto concat_sub_graph = BuildConcatAscGraph("sub3_concat", true);

  asc_backend_to_asc_graph.emplace(ascbc1.get(), add_sub_graph1);
  asc_backend_to_asc_graph.emplace(ascbc2.get(), add_sub_graph2);
  asc_backend_to_asc_graph.emplace(ascbc3.get(), concat_sub_graph);

  AscGraph unfolded_asc_graph("unfolded_asc_graph");
  Status ret = FusedGraphUnfolder::UnfoldFusedGraph(compute_graph, asc_backend_to_asc_graph, unfolded_asc_graph);
  ASSERT_EQ(ret, af::SUCCESS);

  auto axis = unfolded_asc_graph.GetAllAxis();
  ASSERT_EQ(axis.size(), 2);
  EXPECT_EQ(axis[0]->size, concat_sub_graph.GetAllAxis()[0]->size);
  EXPECT_EQ(axis[1]->size, concat_sub_graph.GetAllAxis()[1]->size);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 11UL);
  auto data0 = unfolded_asc_graph.FindNode("data0");
  ASSERT_NE(data0, nullptr);
  EXPECT_EQ(data0->outputs[0].attr.dtype, af::DT_INT8);
  int64_t idx = -1;
  data0->attr.ir_attr->GetAttrValue("index", idx);
  EXPECT_EQ(idx, 0);
}

TEST_F(OptimizerSt, ScalarBroadcastOptimization_Multi_Out_Success) {
  const Expression s0 = af::Symbol(4);
  const Expression s1 = af::Symbol(5);
  const Expression s2 = af::Symbol(6);

  // Load with full padding: shape {1, 1, 1}, strides {0, 0, 0}
  std::vector<Expression> load_shape = {af::sym::kSymbolOne, af::sym::kSymbolOne, af::sym::kSymbolOne};
  std::vector<Expression> load_strides = {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolZero};

  auto graph = AscGraphBuilder("ScalarBroadcastOptimization_Multi_Out_Success")
                   .Loops({s0, s1, s2})
                   // Scalar chain: data -> load (all 1s) -> brc1 -> brc2 -> brc3
                   .Data("data", 0, af::DT_FLOAT)
                   .Load("load", "data", load_shape, load_strides)
                   .Broadcast("brc1", "load", {2})  // expand axis 2: {1,1,1} -> {1,1,s2}
                   .Broadcast("brc2", "brc1", {1})  // expand axis 1: {1,1,s2} -> {1,s1,s2}
                   .Broadcast("brc3", "brc2", {0})  // expand axis 0: {1,s1,s2} -> {s0,s1,s2}
                   // Normal data chain
                   .Data("data2", 1, af::DT_FLOAT)
                   .Load("load1", "data2")
                   .Add("add", "brc3", "load1")
                   .Mul("mul", "add", "brc3")
                   .Store("store", "mul")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);

  auto impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(impl_graph);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 8);
  EXPECT_EQ(compute_graph->FindNode("brc1"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc2"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc3"), nullptr);
  EXPECT_NE(compute_graph->FindNode("add"), nullptr);
  auto add_node = std::dynamic_pointer_cast<af::AscNode>(compute_graph->FindNode("add"));
  EXPECT_NE(add_node, nullptr);
  std::string add0_repeats = ExpressToStr(add_node->inputs[0].attr.repeats);
  EXPECT_EQ(add0_repeats, "(120 / (z0z1z2Tb_size * z0z1z2t_size)), z0z1z2Tb_size, z0z1z2t_size, ");
  std::string add1_repeats = ExpressToStr(add_node->inputs[1].attr.repeats);
  EXPECT_EQ(add1_repeats, ExpressToStr({af::ops::One, af::ops::One, af::ops::One}));

  EXPECT_NE(compute_graph->FindNode("mul"), nullptr);
  auto mul_node = std::dynamic_pointer_cast<af::AscNode>(compute_graph->FindNode("mul"));
  std::string mul0_repeats = ExpressToStr(mul_node->inputs[0].attr.repeats);
  EXPECT_EQ(mul0_repeats, "(120 / (z0z1z2Tb_size * z0z1z2t_size)), z0z1z2Tb_size, z0z1z2t_size, ");
  std::string mul1_repeats = ExpressToStr(mul_node->inputs[1].attr.repeats);
  EXPECT_EQ(mul1_repeats, ExpressToStr({af::ops::One, af::ops::One, af::ops::One}));
}

TEST_F(OptimizerSt, ScalarBroadcastOptimization_Api_Not_Support_Scalar) {
  af::AscGraph graph("ScalarBroadcastOptimization_Api_Not_Support_Scalar");

  auto s0 = graph.CreateSizeVar("4");
  auto s1 = graph.CreateSizeVar("5");
  auto s2 = graph.CreateSizeVar("6");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data("data", graph);
  data.ir_attr.SetIndex(0);
  data.y.dtype = af::DT_FLOAT;

  Load load("load");
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  load.x = data.y;
  *load.y.axis = {z0.id, z1.id, z2.id};
  load.y.dtype = af::DT_FLOAT;
  *load.y.repeats = {af::ops::One, af::ops::One, af::ops::One};
  *load.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::Zero};

  Broadcast brc1("brc1");
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc1.x = load.y;
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  brc1.y.dtype = af::DT_FLOAT;
  *brc1.y.repeats = {af::ops::One, af::ops::One, s2};
  *brc1.y.strides = {af::ops::Zero, af::ops::Zero, af::ops::One};

  Broadcast brc2("brc2");
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc2.x = brc1.y;
  *brc2.y.axis = {z0.id, z1.id, z2.id};
  brc2.y.dtype = af::DT_FLOAT;
  *brc2.y.repeats = {af::ops::One, s1, s2};
  *brc2.y.strides = {af::ops::Zero, s2, af::ops::One};

  Broadcast brc3("brc3");
  brc3.attr.sched.axis = {z0.id, z1.id, z2.id};
  brc3.x = brc2.y;
  *brc3.y.axis = {z0.id, z1.id, z2.id};
  brc3.y.dtype = af::DT_FLOAT;
  *brc3.y.repeats = {s0, s1, s2};
  *brc3.y.strides = {s1 * s2, s2, af::ops::One};

  Data data2("data2", graph);
  data2.ir_attr.SetIndex(1);
  data2.y.dtype = af::DT_FLOAT;

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.x = data2.y;
  load1.y.dtype = af::DT_FLOAT;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, af::ops::One};

  Gt gt("gt");
  gt.attr.sched.axis = {z0.id, z1.id, z2.id};
  gt.x1 = brc3.y;
  gt.x2 = load1.y;
  gt.y.dtype = af::DT_FLOAT;
  *gt.y.axis = {z0.id, z1.id, z2.id};
  *gt.y.repeats = {s0, s1, s2};
  *gt.y.strides = {s1 * s2, s2, af::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id, z2.id};
  store_op.x = gt.y;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  store_op.y.dtype = af::DT_FLOAT;
  *store_op.y.axis = {z0.id, z1.id, z2.id};
  *store_op.y.repeats = {s0, s1, s2};
  *store_op.y.strides = {s1 * s2, s2, af::ops::One};

  Output output_op("output");
  output_op.ir_attr.SetIndex(0);
  output_op.x = store_op.y;
  output_op.y.dtype = af::DT_FLOAT;

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 5UL);

  auto impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(impl_graph);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 8);
  EXPECT_EQ(compute_graph->FindNode("brc1"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc2"), nullptr);
  EXPECT_NE(compute_graph->FindNode("brc3"), nullptr);

  auto impl_graph_1 = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[1];
  auto compute_graph_1 = af::AscGraphUtils::GetComputeGraph(impl_graph_1);
  EXPECT_EQ(compute_graph_1->GetAllNodesSize(), 8);
  EXPECT_EQ(compute_graph_1->FindNode("brc1"), nullptr);
  EXPECT_NE(compute_graph_1->FindNode("brc2"), nullptr);
  EXPECT_EQ(compute_graph_1->FindNode("brc3"), nullptr);

  auto impl_graph_2 = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[2];
  auto compute_graph_2 = af::AscGraphUtils::GetComputeGraph(impl_graph_2);
  EXPECT_EQ(compute_graph_2->GetAllNodesSize(), 8);
  EXPECT_NE(compute_graph_2->FindNode("brc1"), nullptr);
  EXPECT_EQ(compute_graph_2->FindNode("brc2"), nullptr);
  EXPECT_EQ(compute_graph_2->FindNode("brc3"), nullptr);
}

TEST_F(OptimizerSt, ScalarBroadcastOptimization_Same_Input) {
  const Expression s0 = af::Symbol("4");
  const Expression s1 = af::Symbol("5");
  const Expression s2 = af::Symbol("6");

  // Load with full padding: shape {1, 1, 1}, strides {0, 0, 0}
  std::vector<Expression> load_shape = {af::sym::kSymbolOne, af::sym::kSymbolOne, af::sym::kSymbolOne};
  std::vector<Expression> load_strides = {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolZero};

  auto graph = AscGraphBuilder("ScalarBroadcastOptimization_Same_Input")
                   .Loops({s0, s1, s2})
                   // Scalar chain: data -> load (all 1s) -> brc1 -> brc2 -> brc3
                   .Data("data", 0, af::DT_FLOAT)
                   .Load("load", "data", load_shape, load_strides)
                   .Broadcast("brc1", "load", {2})  // expand axis 2: {1,1,1} -> {1,1,s2}
                   .Broadcast("brc2", "brc1", {1})  // expand axis 1: {1,1,s2} -> {1,s1,s2}
                   .Broadcast("brc3", "brc2", {0})  // expand axis 0: {1,s1,s2} -> {s0,s1,s2}
                   .Add("add", "brc3", "brc3")
                   .Store("store", "add")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 5UL);

  auto impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  auto compute_graph = af::AscGraphUtils::GetComputeGraph(impl_graph);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 6);
  EXPECT_EQ(compute_graph->FindNode("brc1"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc2"), nullptr);
  EXPECT_NE(compute_graph->FindNode("brc3"), nullptr);

  auto impl_graph_1 = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[1];
  auto compute_graph_1 = af::AscGraphUtils::GetComputeGraph(impl_graph_1);
  EXPECT_EQ(compute_graph_1->GetAllNodesSize(), 6);
  EXPECT_EQ(compute_graph_1->FindNode("brc1"), nullptr);
  EXPECT_NE(compute_graph_1->FindNode("brc2"), nullptr);
  EXPECT_EQ(compute_graph_1->FindNode("brc3"), nullptr);

  auto impl_graph_2 = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[2];
  auto compute_graph_2 = af::AscGraphUtils::GetComputeGraph(impl_graph_2);
  EXPECT_EQ(compute_graph_2->GetAllNodesSize(), 6);
  EXPECT_NE(compute_graph_2->FindNode("brc1"), nullptr);
  EXPECT_EQ(compute_graph_2->FindNode("brc2"), nullptr);
  EXPECT_EQ(compute_graph_2->FindNode("brc3"), nullptr);
}

TEST_F(OptimizerSt, ScalarBroadcastOptimization_Add_Ne_Common_Scalar_Success) {
  const Expression s0 = af::Symbol(4);
  const Expression s1 = af::Symbol(5);
  const Expression s2 = af::Symbol(6);

  auto graph = AscGraphBuilder("ScalarBroadcastOptimization_Add_Ne_Common_Scalar_Success")
                   .Loops({s0, s1, s2})
                   .Data("data", 0, af::DT_FLOAT)
                   .Load("load", "data", {af::sym::kSymbolOne, af::sym::kSymbolOne, af::sym::kSymbolOne},
                         {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolZero})
                   .Broadcast("brc1", "load", {2})
                   .Broadcast("brc2", "brc1", {1})
                   .Broadcast("brc3", "brc2", {0})
                   .Data("data2", 1, af::DT_FLOAT)
                   .Load("load1", "data2")
                   .Add("add", "brc3", "load1")
                   .template Op<af::ascir_op::Ne>("ne", {"brc3", "add"})
                   .Store("store", "ne")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);

  auto compute_graph = af::AscGraphUtils::GetComputeGraph(
      fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0]);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 8);
  EXPECT_EQ(compute_graph->FindNode("brc1"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc2"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc3"), nullptr);
  EXPECT_NE(compute_graph->FindNode("add"), nullptr);
  EXPECT_NE(compute_graph->FindNode("ne"), nullptr);
  auto add_node = std::dynamic_pointer_cast<af::AscNode>(compute_graph->FindNode("add"));
  std::string add0_repeats = ExpressToStr(add_node->inputs[0].attr.repeats);
  EXPECT_EQ(add0_repeats, "(120 / (z0z1z2Tb_size * z0z1z2t_size)), z0z1z2Tb_size, z0z1z2t_size, ");
  std::string add1_repeats = ExpressToStr(add_node->inputs[1].attr.repeats);
  std::string expected1_repeats = ExpressToStr({af::ops::One, af::ops::One, af::ops::One});
  EXPECT_EQ(add1_repeats, expected1_repeats);

  auto eq_node = std::dynamic_pointer_cast<af::AscNode>(compute_graph->FindNode("ne"));
  std::string ne0_repeats = ExpressToStr(eq_node->inputs[0].attr.repeats);
  EXPECT_EQ(ne0_repeats, "(120 / (z0z1z2Tb_size * z0z1z2t_size)), z0z1z2Tb_size, z0z1z2t_size, ");
  std::string ne1_repeats = ExpressToStr(eq_node->inputs[1].attr.repeats);
  EXPECT_EQ(ne1_repeats, expected1_repeats);
}

/**
 *                 select
 *               /0  \1  \2
 *              /     \    \
 *         not_equal   \     \
 *          /   \       \      \
 *         /      \      \       \
 *        /        \      \       \
 *       /       brc123  brc456  brc789
 *      /           |       |      |
 *    load0       load1   load2  load3
 *      |          |s      |s      |s
 *    data0      data1   data2   data3
 */
TEST_F(OptimizerSt, ScalarBroadcastOptimization_Select_2S_3S_Success) {
  const Expression s0 = af::Symbol(4);
  const Expression s1 = af::Symbol(5);
  const Expression s2 = af::Symbol(6);

  auto graph = AscGraphBuilder("ScalarBroadcastOptimization_Select_2S_3S_Success")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0, af::DT_FLOAT)
                   .Load("load0", "data0")
                   .Data("data1", 1, af::DT_FLOAT)
                   .Load("load1", "data1", {af::sym::kSymbolOne, af::sym::kSymbolOne, af::sym::kSymbolOne},
                         {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolZero})
                   .Broadcast("brc1", "load1", {2})
                   .Broadcast("brc2", "brc1", {1})
                   .Broadcast("brc3", "brc2", {0})
                   .Data("data2", 2, af::DT_FLOAT)
                   .Load("load2", "data2", {af::sym::kSymbolOne, af::sym::kSymbolOne, af::sym::kSymbolOne},
                         {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolZero})
                   .Broadcast("brc4", "load2", {2})
                   .Broadcast("brc5", "brc4", {1})
                   .Broadcast("brc6", "brc5", {0})
                   .template Op<af::ascir_op::Ne>("ne", {"brc3", "load0"})
                   .Data("data3", 3, af::DT_FLOAT)
                   .Load("load3", "data3", {af::sym::kSymbolOne, af::sym::kSymbolOne, af::sym::kSymbolOne},
                         {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolZero})
                   .Broadcast("brc7", "load3", {2})
                   .Broadcast("brc8", "brc7", {1})
                   .Broadcast("brc9", "brc8", {0})
                   .template Op<af::ascir_op::Select>("select", {"ne", "brc6", "brc9"})
                   .Store("store", "select")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);

  auto compute_graph = af::AscGraphUtils::GetComputeGraph(
      fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0]);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 12);
  EXPECT_EQ(compute_graph->FindNode("brc1"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc2"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc3"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc4"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc5"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc6"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc7"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc8"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc9"), nullptr);
  EXPECT_NE(compute_graph->FindNode("select"), nullptr);
  EXPECT_NE(compute_graph->FindNode("ne"), nullptr);

  auto ne_node = std::dynamic_pointer_cast<af::AscNode>(compute_graph->FindNode("ne"));
  std::string ne0_repeats = ExpressToStr(ne_node->inputs[0].attr.repeats);
  EXPECT_EQ(ne0_repeats, "(120 / (z0z1z2Tb_size * z0z1z2t_size)), z0z1z2Tb_size, z0z1z2t_size, ");
  std::string ne1_repeats = ExpressToStr(ne_node->inputs[1].attr.repeats);
  std::string expected1_repeats = ExpressToStr({af::ops::One, af::ops::One, af::ops::One});
  EXPECT_EQ(ne1_repeats, expected1_repeats);

  auto select_node = std::dynamic_pointer_cast<af::AscNode>(compute_graph->FindNode("select"));
  std::string select1_repeats = ExpressToStr(select_node->inputs[1].attr.repeats);
  EXPECT_EQ(select1_repeats, expected1_repeats);
  std::string select2_repeats = ExpressToStr(select_node->inputs[2].attr.repeats);
  EXPECT_EQ(select2_repeats, expected1_repeats);
}

TEST_F(OptimizerSt, ScalarBroadcastOptimization_Add_Le_Common_Scalar_Failed) {
  const Expression s0 = af::Symbol(4);
  const Expression s1 = af::Symbol(5);
  const Expression s2 = af::Symbol(6);

  auto graph = AscGraphBuilder("ScalarBroadcastOptimization_Add_Le_Common_Scalar_Failed")
                   .Loops({s0, s1, s2})
                   .Data("data", 0, af::DT_FLOAT)
                   .Load("load", "data", {af::sym::kSymbolOne, af::sym::kSymbolOne, af::sym::kSymbolOne},
                         {af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolZero})
                   .Broadcast("brc1", "load", {2})
                   .Broadcast("brc2", "brc1", {1})
                   .Broadcast("brc3", "brc2", {0})
                   .Data("data2", 1, af::DT_FLOAT)
                   .Load("load1", "data2")
                   .Add("add", "brc3", "load1")
                   .template Op<af::ascir_op::Le>("le", {"brc3", "add"})
                   .Store("store", "le")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 3UL);

  auto compute_graph = af::AscGraphUtils::GetComputeGraph(
      fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0]);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 9);
  EXPECT_NE(compute_graph->FindNode("brc1"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc2"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc3"), nullptr);
  EXPECT_NE(compute_graph->FindNode("add"), nullptr);
  EXPECT_NE(compute_graph->FindNode("le"), nullptr);
}

TEST_F(OptimizerSt, ScalarBroadcastOptimization_Scalar) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");

  auto graph =
      AscGraphBuilder("ScalarBroadcastOptimization_Scalar")
          .Loops({s0, s1, s2})
          .Scalar("data0", "0", af::DT_FLOAT16)
          .Broadcast("brc0", "data0", {af::sym::kSymbolOne, af::sym::kSymbolOne, s2})
          .Broadcast("brc1", "brc0", {s0, af::sym::kSymbolOne, s2})
          .Data("data1", 0, af::DT_FLOAT16)
          .Load("load1", "data1", {s0, af::sym::kSymbolOne, s2}, {s2, af::sym::kSymbolZero, af::sym::kSymbolOne})
          .Add("add", "brc1", "load1")
          .Store("store", "add")
          .Output("y", "store", 0, af::DT_FLOAT16)
          .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  ASSERT_TRUE(!fused_scheduled_result.node_idx_to_scheduled_results.empty());
  auto &schedule_results = fused_scheduled_result.node_idx_to_scheduled_results[0];
  EXPECT_EQ(schedule_results.size(), 1UL);
  EXPECT_EQ(schedule_results[0].schedule_groups.size(), 1UL);
  ASSERT_EQ(schedule_results[0].schedule_groups[0].impl_graphs.size(), 1UL);

  auto const &impl_graphs = schedule_results[0].schedule_groups[0].impl_graphs;
  EXPECT_EQ(impl_graphs[0].FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[0].FindNode("brc1"), nullptr);
}

TEST_F(OptimizerSt, MultiBroadcastCancellation_All_One) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  const auto s3 = Sym("s3");
  const auto s4 = Sym("s4");
  const auto all_one = std::vector<Expression>(5, af::sym::kSymbolOne);
  const auto all_zero = std::vector<Expression>(5, af::sym::kSymbolZero);

  auto graph = AscGraphBuilder("store_load")
                   .Loops({s0, s1, s2, s3, s4})
                   .Data("data0", 0, all_one, all_zero, af::DT_FLOAT16)
                   .Load("load0", "data0", all_one, all_zero)
                   .Abs("abs0", "load0")
                   .Broadcast("brc0", "abs0",
                              {af::sym::kSymbolOne, af::sym::kSymbolOne, af::sym::kSymbolOne, af::sym::kSymbolOne, s4})
                   .Broadcast("brc1", "brc0", {af::sym::kSymbolOne, af::sym::kSymbolOne, af::sym::kSymbolOne, s3, s4})
                   .Broadcast("brc2", "brc1", {af::sym::kSymbolOne, af::sym::kSymbolOne, s2, s3, s4})
                   .Broadcast("brc3", "brc2", {af::sym::kSymbolOne, s1, s2, s3, s4})
                   .Broadcast("brc4", "brc3", {s0, s1, s2, s3, s4})
                   .Exp("exp0", "brc4")
                   .Add("add0", "exp0", "brc4")
                   .Store("store", "add0")
                   .Output("y", "store", 0, af::DT_FLOAT16)
                   .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);

  auto impl_graphs = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs;
  EXPECT_EQ(impl_graphs.size(), 9);

  EXPECT_EQ(impl_graphs[0].FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[0].FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[0].FindNode("brc2"), nullptr);
  EXPECT_EQ(impl_graphs[0].FindNode("brc3"), nullptr);
  auto impl_grp_0_brc4 = impl_graphs[0].FindNode("brc4");
  EXPECT_NE(impl_grp_0_brc4, nullptr);
  EXPECT_EQ(impl_grp_0_brc4->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_0_brc4->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "add0");

  EXPECT_EQ(impl_graphs[1].FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[1].FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[1].FindNode("brc2"), nullptr);
  EXPECT_EQ(impl_graphs[1].FindNode("brc4"), nullptr);
  auto impl_grp_1_brc3 = impl_graphs[1].FindNode("brc3");
  EXPECT_NE(impl_grp_1_brc3, nullptr);
  EXPECT_EQ(impl_grp_1_brc3->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_1_brc3->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "add0");

  EXPECT_EQ(impl_graphs[2].FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[2].FindNode("brc3"), nullptr);
  EXPECT_EQ(impl_graphs[2].FindNode("brc4"), nullptr);
  auto impl_grp_2_brc2 = impl_graphs[2].FindNode("brc2");
  EXPECT_NE(impl_grp_2_brc2, nullptr);
  EXPECT_EQ(impl_grp_2_brc2->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_2_brc2->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "add0");

  EXPECT_EQ(impl_graphs[3].FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[3].FindNode("brc2"), nullptr);
  EXPECT_EQ(impl_graphs[3].FindNode("brc3"), nullptr);
  EXPECT_EQ(impl_graphs[3].FindNode("brc4"), nullptr);
  auto impl_grp_3_brc1 = impl_graphs[3].FindNode("brc1");
  EXPECT_NE(impl_grp_3_brc1, nullptr);
  EXPECT_EQ(impl_grp_3_brc1->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_3_brc1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "add0");

  EXPECT_EQ(impl_graphs[4].FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graphs[4].FindNode("brc2"), nullptr);
  EXPECT_EQ(impl_graphs[4].FindNode("brc3"), nullptr);
  EXPECT_EQ(impl_graphs[4].FindNode("brc4"), nullptr);
  auto impl_grp_4_exp0 = impl_graphs[4].FindNode("exp0");
  EXPECT_NE(impl_grp_4_exp0, nullptr);
  EXPECT_EQ(impl_grp_4_exp0->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_4_exp0->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");
}

TEST_F(OptimizerSt, ScalarBroadcastOptimization_Two_Scalar) {
  const auto s0 = Symbol(4);
  const auto s1 = Symbol(5);
  const auto s2 = Symbol(6);
  const auto all_one = std::vector<Expression>(3, af::sym::kSymbolOne);
  const auto all_zero = std::vector<Expression>(3, af::sym::kSymbolZero);

  auto graph = AscGraphBuilder("ScalarBroadcastOptimization_Two_Scalar")
                   .Loops({s0, s1, s2})
                   .Data("data", 0, af::DT_FLOAT)
                   .Load("load", "data", all_one, all_zero)
                   .Broadcast("brc1", "load", {af::sym::kSymbolOne, af::sym::kSymbolOne, s2})
                   .Broadcast("brc2", "brc1", {af::sym::kSymbolOne, s1, s2})
                   .Broadcast("brc3", "brc2", {s0, s1, s2})
                   .Data("data2", 1, af::DT_FLOAT)
                   .Load("load1", "data2", all_one, all_zero)
                   .Broadcast("brc4", "load1", {af::sym::kSymbolOne, af::sym::kSymbolOne, s2})
                   .Broadcast("brc5", "brc4", {af::sym::kSymbolOne, s1, s2})
                   .Broadcast("brc6", "brc5", {s0, s1, s2})
                   .Op<af::ascir_op::Pow>("pow", {"brc3", "brc6"})
                   .Store("store", "pow")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  auto impl_graphs = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs;
  EXPECT_EQ(impl_graphs.size(), 3);
  auto impl_graph0 = af::AscGraphUtils::GetComputeGraph(impl_graphs[0]);
  EXPECT_EQ(impl_graph0->GetAllNodesSize(), 8);
  EXPECT_NE(impl_graph0->FindNode("brc1"), nullptr);
  EXPECT_EQ(impl_graph0->FindNode("brc2"), nullptr);
  EXPECT_EQ(impl_graph0->FindNode("brc3"), nullptr);
  EXPECT_EQ(impl_graph0->FindNode("brc4"), nullptr);
  EXPECT_EQ(impl_graph0->FindNode("brc5"), nullptr);
  EXPECT_EQ(impl_graph0->FindNode("brc6"), nullptr);
}

/**
 *                   store
 *                     |
 *                   mul0
 *                  /   \
 *               add0  exp1
 *              /    \ /
 *    (remove)brc1    \
 *             |      |
 *            exp0   brc0(remove)
 *              \   /
 *              abs0
 *               |
 *             load0
 *              |
 *            data0
 */
TEST_F(OptimizerSt, RemoveRedundantBroadcast) {
  auto graph = BuildRedundantBroadcastGraph("RemoveRedundantBroadcast");

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  auto impl_graphs = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs;
  EXPECT_EQ(impl_graphs.size(), 4);
  // consumer split creates clone for exp1; common-axis backward removes brc0/brc1 from add0's chain
  auto impl_grp_0_exp1 = impl_graphs[0].FindNode("exp1");
  EXPECT_NE(impl_grp_0_exp1, nullptr);
  EXPECT_EQ(impl_grp_0_exp1->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_0_exp1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(),
            "brc0_consumer_split_1");

  auto impol_grp_0_add0 = impl_graphs[0].FindNode("add0");
  EXPECT_NE(impol_grp_0_add0, nullptr);
  EXPECT_EQ(impol_grp_0_add0->GetAllInDataAnchorsSize(), 2);
  EXPECT_EQ(impol_grp_0_add0->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");
  EXPECT_EQ(impol_grp_0_add0->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "exp0");

  EXPECT_NE(impl_graphs[0].FindNode("brc0_consumer_split_1"), nullptr);

  // check remove brc in unaligned template
  auto impl_grp_1_exp1 = impl_graphs[1].FindNode("exp1");
  EXPECT_NE(impl_grp_1_exp1, nullptr);
  EXPECT_EQ(impl_grp_1_exp1->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl_grp_1_exp1->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");

  auto impol_grp_1_add0 = impl_graphs[1].FindNode("add0");
  EXPECT_NE(impol_grp_1_add0, nullptr);
  EXPECT_EQ(impol_grp_1_add0->GetAllInDataAnchorsSize(), 2);
  EXPECT_EQ(impol_grp_1_add0->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");
  EXPECT_EQ(impol_grp_1_add0->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "exp0");

  EXPECT_EQ(impl_graphs[1].FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[1].FindNode("brc1"), nullptr);
}

/**
 *                     store
 *                       |
 *                     add0
 *                     /  \
 *                   /     \
 *                 /        \
 *               brc1        \
 *                |           \
 *              brc0         abs0
 *               |            |
 *             load0        load1
 *              |             |
 *            data0         data1
 */
TEST_F(OptimizerSt, RemoveContinuesBroadcast) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");
  const auto s3 = Sym("s3");
  const auto s4 = Sym("s4");

  auto graph = AscGraphBuilder("RemoveContinuesBroadcast")
                   .Loops({s0, s1, s2, s3, s4})
                   .Data("data0", 0, af::DT_FLOAT16)
                   .Load("load0", "data0")
                   .Abs("abs0", "load0")
                   .Data("data1", 1, {s0, s1, af::sym::kSymbolOne, af::sym::kSymbolOne, af::sym::kSymbolOne},
                         {s1, af::sym::kSymbolOne, af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolZero},
                         af::DT_FLOAT16)
                   .Load("load1", "data1", {s0, s1, af::sym::kSymbolOne, af::sym::kSymbolOne, af::sym::kSymbolOne},
                         {s1, af::sym::kSymbolOne, af::sym::kSymbolZero, af::sym::kSymbolZero, af::sym::kSymbolZero})
                   .Broadcast("brc0", "load1", {s0, s1, af::sym::kSymbolOne, af::sym::kSymbolOne, s4})
                   .Broadcast("brc1", "brc0", {s0, s1, af::sym::kSymbolOne, s3, s4})
                   .Broadcast("brc2", "brc1", {s0, s1, s2, s3, s4})
                   .Add("add0", "abs0", "brc2")
                   .Store("store", "add0")
                   .Output("y", "store", 0, af::DT_FLOAT16)
                   .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  auto impl_graphs = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs;
  EXPECT_EQ(impl_graphs.size(), 7);
  EXPECT_EQ(impl_graphs[0].FindNode("brc0"), nullptr);
  EXPECT_EQ(impl_graphs[0].FindNode("brc1"), nullptr);
  EXPECT_NE(impl_graphs[0].FindNode("brc2"), nullptr);
  auto impl_grp_0_brc2 = impl_graphs[0].FindNode("brc2");
  auto brc2_input_repeats = ExpressToStr(impl_grp_0_brc2->inputs[0].attr.repeats);
  EXPECT_EQ(brc2_input_repeats, "(s0 * s1 / (z0z1Tb_size * z0z1t_size)), z0z1Tb_size, z0z1t_size, 1, 1, 1, ");

  auto impl_grp_0_add0 = impl_graphs[0].FindNode("add0");
  EXPECT_NE(impl_grp_0_add0, nullptr);
  EXPECT_EQ(impl_grp_0_add0->GetAllInDataAnchorsSize(), 2);
  EXPECT_EQ(impl_grp_0_add0->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "abs0");
  EXPECT_EQ(impl_grp_0_add0->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "brc2");
}

TEST_F(OptimizerSt, RemovePad_not_align_broadcast) {
  const Expression s0 = af::Symbol("s0");
  const Expression s1 = af::Symbol("s1");
  const Expression s2 = af::Symbol("s2");

  // Load0 with padding: shape {1, s1, s2}, strides {0, s2, 1}
  std::vector<Expression> load0_shape = {af::sym::kSymbolOne, s1, s2};
  std::vector<Expression> load0_strides = {af::sym::kSymbolZero, s2, af::sym::kSymbolOne};

  // Load2 with padding: shape {1, s1, s2}, strides {0, s2, 1}
  std::vector<Expression> load2_shape = {af::sym::kSymbolOne, s1, s2};
  std::vector<Expression> load2_strides = {af::sym::kSymbolZero, s2, af::sym::kSymbolOne};

  auto graph = AscGraphBuilder("RemovePad_not_align_broadcast")
                   .Loops({s0, s1, s2})
                   // First chain: data0 -> load0 (with padding) -> brc0 (expand axis 0) -> add0 -> store0 -> y0
                   .Data("data0", 0, af::DT_FLOAT16)
                   .Load("load0", "data0", load0_shape, load0_strides)
                   .Broadcast("brc0", "load0", {0})  // broadcast on axis 0: {1, s1, s2} -> {s0, s1, s2}
                   // Second chain: data1 -> load1 (normal)
                   .Data("data1", 1, af::DT_FLOAT16)
                   .Load("load1", "data1")
                   .Add("add0", "brc0", "load1")
                   .Store("store0", "add0")
                   .Output("y0", "store0", 0, af::DT_FLOAT16)
                   // Third chain: data2 -> load2 (with padding) -> brc2 (expand axis 0) -> mul0 -> store1 -> y1
                   .Data("data2", 2, af::DT_FLOAT16)
                   .Load("load2", "data2", load2_shape, load2_strides)
                   .Broadcast("brc2", "load2", {0})  // broadcast on axis 0: {1, s1, s2} -> {s0, s1, s2}
                   .Mul("mul0", "load1", "brc2")
                   .Store("store1", "mul0")
                   .Output("y1", "store1", 1, af::DT_FLOAT16)
                   .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  auto impl_graphs = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs;
  EXPECT_EQ(impl_graphs.size(), 4);

  EXPECT_NE(impl_graphs[2].FindNode("brc0"), nullptr);
  EXPECT_NE(impl_graphs[2].FindNode("brc2"), nullptr);
  EXPECT_NE(impl_graphs[2].FindNode("brc0_remove_pad_0"), nullptr);
  EXPECT_NE(impl_graphs[2].FindNode("brc2_remove_pad_0"), nullptr);
  EXPECT_EQ(AscGraphUtils::GetComputeGraph(impl_graphs[2])->GetAllNodesSize(), 16);
  const auto &impl1_remove_pad0 = impl_graphs[2].FindNode("brc0_remove_pad_0");
  EXPECT_EQ(impl1_remove_pad0->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl1_remove_pad0->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "brc0");
  EXPECT_EQ(impl1_remove_pad0->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "add0");
  auto impl1_remove_pad0_in_strides = ExpressToStr(impl1_remove_pad0->inputs[0].attr.vectorized_strides);
  EXPECT_EQ(impl1_remove_pad0_in_strides, "(16 * Ceiling((Rational(1 , 16) * s1 * s2))), 1, ");
  auto impl1_remove_pad0_out_strides = ExpressToStr(impl1_remove_pad0->outputs[0].attr.vectorized_strides);
  EXPECT_EQ(impl1_remove_pad0_out_strides, "(s1 * s2), 1, ");

  const auto &impl1_remove_pad2 = impl_graphs[2].FindNode("brc2_remove_pad_0");
  EXPECT_EQ(impl1_remove_pad2->GetAllInDataAnchorsSize(), 1);
  EXPECT_EQ(impl1_remove_pad2->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "brc2");
  EXPECT_EQ(impl1_remove_pad2->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetName(), "mul0");
  auto impl1_remove_pad2_in_strides = ExpressToStr(impl1_remove_pad2->inputs[0].attr.vectorized_strides);
  EXPECT_EQ(impl1_remove_pad2_in_strides, "(16 * Ceiling((Rational(1 , 16) * s1 * s2))), 1, ");
  auto impl1_remove_pad2_out_strides = ExpressToStr(impl1_remove_pad2->outputs[0].attr.vectorized_strides);
  EXPECT_EQ(impl1_remove_pad2_out_strides, "(s1 * s2), 1, ");

  const auto &impl3 = impl_graphs[3];
  EXPECT_EQ("RemovePad_not_align_broadcast_0_B0Y0_inline_S0G0C3", impl3.GetName());
  EXPECT_EQ(impl3.FindNode("brc0"), nullptr);
  EXPECT_EQ(impl3.FindNode("brc2"), nullptr);
  EXPECT_EQ(impl3.FindNode("brc0_remove_pad_0"), nullptr);
  EXPECT_EQ(impl3.FindNode("brc2_remove_pad_0"), nullptr);
}

/**
 *                 store
 *                 |
 *               brc1 (s0, s1, s2)
 *                |
 *              brc0 (1, s1, s2)
 *               |
 *             load0 (1, s1, 1)
 *              |
 *            data0 (1, s1, 1)
 */
TEST_F(OptimizerSt, RemoveContinuesBroadcast_BAB) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");

  auto graph = AscGraphBuilder("RemoveContinuesBroadcast_BAB")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0, {af::sym::kSymbolOne, s1, af::sym::kSymbolOne},
                         {af::sym::kSymbolZero, af::sym::kSymbolOne, af::sym::kSymbolZero}, af::DT_FLOAT16)
                   .Load("load0", "data0", {af::sym::kSymbolOne, s1, af::sym::kSymbolOne},
                         {af::sym::kSymbolZero, af::sym::kSymbolOne, af::sym::kSymbolZero})
                   .Broadcast("brc0", "load0", {af::sym::kSymbolOne, s1, s2})
                   .Broadcast("brc1", "brc0", {s0, s1, s2})
                   .Store("store", "brc1")
                   .Output("y", "store", 0, af::DT_FLOAT16)
                   .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  auto impl_graphs = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs;
  EXPECT_EQ(impl_graphs.size(), 5);
  EXPECT_EQ(impl_graphs[0].FindNode("brc0"), nullptr);
  EXPECT_NE(impl_graphs[0].FindNode("brc1"), nullptr);
  auto impl_grp_0_brc1 = impl_graphs[0].FindNode("brc1");
  auto brc1_input_repeats = ExpressToStr(impl_grp_0_brc1->inputs[0].attr.repeats);
  EXPECT_EQ(brc1_input_repeats, "1, 1, 1, s1, 1, ");
}

TEST_F(OptimizerSt, BufQueAllocator_RemovePad_MemUnique) {
  af::AscGraph graph("BufQueAllocator_RemovePad_MemUnique");

  const Expression s0 = graph.CreateSizeVar(320);
  const Expression s1 = graph.CreateSizeVar(2889);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data x0("x", graph);
  x0.attr.api.compute_type = ComputeType::kComputeInvalid;
  x0.attr.api.type = af::ApiType::kAPITypeBuffer;
  x0.ir_attr.SetIndex(0);
  x0.y.dtype = af::DataType::DT_FLOAT;

  af::ascir_op::Load load0("load0");
  load0.x = x0.y;
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.attr.api.type = af::ApiType::kAPITypeCompute;
  load0.attr.sched.axis = {z0.id, z1.id};
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};
  load0.y.dtype = af::DataType::DT_FLOAT;
  load0.attr.api.unit = ComputeUnit::kUnitMTE2;

  af::ascir_op::Data x1("x1", graph);
  x1.attr.api.compute_type = ComputeType::kComputeInvalid;
  x1.attr.api.type = af::ApiType::kAPITypeBuffer;
  x1.y.dtype = af::DataType::DT_FLOAT;
  x1.ir_attr.SetIndex(1);

  af::ascir_op::Load load1("load1");
  load1.x = x1.y;
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.attr.api.type = af::ApiType::kAPITypeCompute;
  load1.attr.sched.axis = {z0.id, z1.id};
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {One, s1};
  *load1.y.strides = {Zero, One};
  load1.y.dtype = af::DataType::DT_FLOAT;
  load1.attr.api.unit = ComputeUnit::kUnitMTE2;

  af::ascir_op::Broadcast broadcast1("broadcast1");
  broadcast1.x = load1.y;
  broadcast1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  broadcast1.attr.api.type = af::ApiType::kAPITypeCompute;
  broadcast1.attr.sched.axis = {z0.id, z1.id};
  *broadcast1.y.axis = {z0.id, z1.id};
  *broadcast1.y.repeats = {s0, s1};
  *broadcast1.y.strides = {s1, One};
  broadcast1.y.dtype = af::DataType::DT_FLOAT;
  broadcast1.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Add add0("add0");
  add0.x1 = load0.y;
  add0.x2 = broadcast1.y;
  add0.attr.api.compute_type = ComputeType::kComputeElewise;
  add0.attr.api.type = af::ApiType::kAPITypeCompute;
  add0.attr.sched.axis = {z0.id, z1.id};
  *add0.y.axis = {z0.id, z1.id};
  *add0.y.repeats = {s0, s1};
  *add0.y.strides = {s1, One};
  add0.y.dtype = af::DataType::DT_FLOAT;
  add0.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Data x2("x2", graph);
  x2.attr.api.compute_type = ComputeType::kComputeInvalid;
  x2.attr.api.type = af::ApiType::kAPITypeBuffer;
  x2.ir_attr.SetIndex(2);
  x2.y.dtype = af::DataType::DT_FLOAT;

  af::ascir_op::Load load2("load2");
  load2.x = x2.y;
  load2.attr.api.compute_type = ComputeType::kComputeLoad;
  load2.attr.api.type = af::ApiType::kAPITypeCompute;
  load2.attr.sched.axis = {z0.id, z1.id};
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.repeats = {s0, s1};
  *load2.y.strides = {s1, One};
  load2.y.dtype = af::DataType::DT_FLOAT;
  load2.attr.api.unit = ComputeUnit::kUnitMTE2;

  af::ascir_op::Mul mul0("mul0");
  mul0.x1 = load2.y;
  mul0.x2 = add0.y;
  mul0.attr.api.compute_type = ComputeType::kComputeElewise;
  mul0.attr.api.type = af::ApiType::kAPITypeCompute;
  mul0.attr.sched.axis = {z0.id, z1.id};
  *mul0.y.axis = {z0.id, z1.id};
  *mul0.y.repeats = {s0, s1};
  *mul0.y.strides = {s1, One};
  mul0.y.dtype = af::DataType::DT_FLOAT;
  mul0.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Store store("store");
  store.x = mul0.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.attr.api.type = af::ApiType::kAPITypeCompute;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};
  store.y.dtype = af::DataType::DT_FLOAT;
  store.attr.api.unit = ComputeUnit::kUnitMTE3;

  af::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = af::DataType::DT_FLOAT;
  y.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 3UL);

  auto impl_graph1 = af::AscGraphUtils::GetComputeGraph(
      fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[1]);
  EXPECT_EQ(impl_graph1->GetAllNodesSize(), 12);
  EXPECT_NE(impl_graph1->FindNode("broadcast1"), nullptr);
  EXPECT_NE(impl_graph1->FindNode("broadcast1_remove_pad_0"), nullptr);
  EXPECT_NE(impl_graph1->FindNode("add0"), nullptr);
  const auto &impl_graph1_brc1 = std::dynamic_pointer_cast<af::AscNode>(impl_graph1->FindNode("broadcast1"));
  const auto &impl_graph1_rpd =
      std::dynamic_pointer_cast<af::AscNode>(impl_graph1->FindNode("broadcast1_remove_pad_0"));
  const auto &impl_graph1_add0 = std::dynamic_pointer_cast<af::AscNode>(impl_graph1->FindNode("add0"));
  const auto &impl_graph1_mul0 = std::dynamic_pointer_cast<af::AscNode>(impl_graph1->FindNode("mul0"));
  EXPECT_EQ(impl_graph1_brc1->outputs[0].attr.buf.id, 1);
  EXPECT_EQ(impl_graph1_rpd->outputs[0].attr.buf.id, 2);
  EXPECT_EQ(impl_graph1_add0->outputs[0].attr.que.id, impl_graph1_mul0->outputs[0].attr.que.id);
}

TEST_F(OptimizerSt, BufQueAllocator_Inplace) {
  af::AscGraph graph("BufQueAllocator_Inplace");
  af::ascir_op::Data x0("x0", graph);
  x0.attr.api.compute_type = ComputeType::kComputeInvalid;
  x0.attr.api.type = af::ApiType::kAPITypeBuffer;
  x0.ir_attr.SetIndex(0);

  af::ascir_op::Load load0("load0");
  load0.x = x0.y;
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.attr.api.unit = ComputeUnit::kUnitMTE2;

  af::ascir_op::Abs abs0("abs0");
  abs0.x = load0.y;
  abs0.attr.api.compute_type = ComputeType::kComputeElewise;
  abs0.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Abs abs1("abs1");
  abs1.x = abs0.y;
  abs1.attr.api.compute_type = ComputeType::kComputeElewise;
  abs1.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Abs abs2("abs2");
  abs2.x = abs1.y;
  abs2.attr.api.compute_type = ComputeType::kComputeElewise;
  abs2.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Data x1("x1", graph);
  x1.attr.api.compute_type = ComputeType::kComputeInvalid;
  x1.attr.api.type = af::ApiType::kAPITypeBuffer;
  x1.ir_attr.SetIndex(1);

  af::ascir_op::Load load1("load1");
  load1.x = x1.y;
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.attr.api.unit = ComputeUnit::kUnitMTE2;

  af::ascir_op::Abs abs4("abs4");
  abs4.x = load1.y;
  abs4.attr.api.compute_type = ComputeType::kComputeElewise;
  abs4.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Abs abs5("abs5");
  abs5.x = abs4.y;
  abs5.attr.api.compute_type = ComputeType::kComputeElewise;
  abs5.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Add add0("add0");
  add0.x1 = abs2.y;
  add0.x2 = abs5.y;
  add0.attr.api.compute_type = ComputeType::kComputeElewise;
  add0.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Add add1("add1");
  add1.x1 = abs2.y;
  add1.x2 = add0.y;
  add1.attr.api.compute_type = ComputeType::kComputeElewise;
  add1.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Abs abs3("abs3");
  abs3.x = add1.y;
  abs3.attr.api.compute_type = ComputeType::kComputeElewise;
  abs3.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Abs abs7("abs7");
  abs7.x = abs3.y;
  abs7.attr.api.compute_type = ComputeType::kComputeElewise;
  abs7.attr.api.unit = ComputeUnit::kUnitVector;
  abs7.y.dtype = DataType::DT_FLOAT16;

  af::ascir_op::Data x2("x2", graph);
  x2.attr.api.compute_type = ComputeType::kComputeInvalid;
  x2.attr.api.type = af::ApiType::kAPITypeBuffer;
  x2.ir_attr.SetIndex(2);

  af::ascir_op::Load load2("load2");
  load2.x = x2.y;
  load2.attr.api.compute_type = ComputeType::kComputeLoad;
  load2.attr.api.unit = ComputeUnit::kUnitMTE2;

  af::ascir_op::Add add2("add2");
  add2.x1 = abs7.y;
  add2.x2 = load2.y;
  add2.attr.api.compute_type = ComputeType::kComputeElewise;
  add2.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Abs abs9("abs9");
  abs9.x = add2.y;
  abs9.attr.api.compute_type = ComputeType::kComputeElewise;
  abs9.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Store store("store");
  store.x = abs9.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.attr.api.unit = ComputeUnit::kUnitMTE2;

  af::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.ir_attr.SetIndex(0);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);

  const auto &impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  auto load_result = impl_graph.FindNode("load0");  // vec in
  EXPECT_EQ(load_result->outputs[0].attr.que.id, 0);
  EXPECT_EQ(load_result->outputs[0].attr.mem.reuse_id, 0);
  auto load1_result = impl_graph.FindNode("load1");  // vec in
  EXPECT_EQ(load1_result->outputs[0].attr.que.id, 1);
  EXPECT_EQ(load1_result->outputs[0].attr.mem.reuse_id, 4);
  auto load2_result = impl_graph.FindNode("load2");  // vec in
  EXPECT_EQ(load2_result->outputs[0].attr.que.id, 0);
  EXPECT_EQ(load2_result->outputs[0].attr.mem.reuse_id, 11);

  auto abs0_result = impl_graph.FindNode("abs0");  // vec calc
  EXPECT_EQ(abs0_result->outputs[0].attr.buf.id, 1);
  EXPECT_EQ(abs0_result->outputs[0].attr.mem.reuse_id, af::kIdNone);
  auto abs1_result = impl_graph.FindNode("abs1");  // vec calc
  EXPECT_EQ(abs1_result->outputs[0].attr.buf.id, 2);
  EXPECT_EQ(abs1_result->outputs[0].attr.mem.reuse_id, af::kIdNone);
  auto abs2_result = impl_graph.FindNode("abs2");  // vec calc
  EXPECT_EQ(abs2_result->outputs[0].attr.buf.id, 3);
  EXPECT_EQ(abs2_result->outputs[0].attr.mem.reuse_id, af::kIdNone);

  auto abs4_result = impl_graph.FindNode("abs4");  // vec calc
  EXPECT_EQ(abs4_result->outputs[0].attr.buf.id, 4);
  EXPECT_EQ(abs4_result->outputs[0].attr.mem.reuse_id, af::kIdNone);
  auto abs5_result = impl_graph.FindNode("abs5");  // vec calc reuse que
  EXPECT_EQ(abs5_result->outputs[0].attr.que.id, 1);
  EXPECT_EQ(abs5_result->outputs[0].attr.mem.reuse_id, 6);

  auto add0_result = impl_graph.FindNode("add0");  // vec calc
  EXPECT_EQ(add0_result->outputs[0].attr.buf.id, 5);
  EXPECT_EQ(add0_result->outputs[0].attr.mem.reuse_id, af::kIdNone);
  auto add1_result = impl_graph.FindNode("add1");  // vec calc
  EXPECT_EQ(add1_result->outputs[0].attr.buf.id, 6);
  EXPECT_EQ(add1_result->outputs[0].attr.mem.reuse_id, af::kIdNone);
  auto abs3_result = impl_graph.FindNode("abs3");  // vec calc
  EXPECT_EQ(abs3_result->outputs[0].attr.buf.id, 7);
  EXPECT_EQ(abs3_result->outputs[0].attr.mem.reuse_id, af::kIdNone);
  auto abs7_result = impl_graph.FindNode("abs7");  // vec calc
  EXPECT_EQ(abs7_result->outputs[0].attr.buf.id, 8);
  EXPECT_EQ(abs7_result->outputs[0].attr.mem.reuse_id, af::kIdNone);
  auto add2_result = impl_graph.FindNode("add2");  // vec calc
  EXPECT_EQ(add2_result->outputs[0].attr.buf.id, 9);
  EXPECT_EQ(add2_result->outputs[0].attr.mem.reuse_id, af::kIdNone);

  auto abs9_result = impl_graph.FindNode("abs9");  // vecout
  EXPECT_EQ(abs9_result->outputs[0].attr.que.id, 2);
  EXPECT_EQ(abs9_result->outputs[0].attr.mem.reuse_id, 13);
}

TEST_F(OptimizerSt, concat_last1dim) {
  af::AscGraph graph("LoadAbsStore");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = af::Symbol(2);

  auto tmp = graph.CreateAxis("tmp", s0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data x("x", graph);
  x.attr.sched.axis = {z0.id, z1.id};
  x.y.dtype = af::DT_INT64;
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load load("load");
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id};
  load.y.dtype = af::DT_INT64;
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, One};
  *load.y.strides = {One, One};

  af::ascir_op::Data x1("x1", graph);
  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = af::DT_INT64;
  x1.ir_attr.SetIndex(1);

  af::ascir_op::Load load1("load1");
  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = af::DT_INT64;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, One};
  *load1.y.strides = {One, One};

  af::ascir_op::Concat concat("concat");
  concat.x = {load.y, load1.y};
  concat.attr.sched.axis = {z0.id, z1.id};
  concat.y.dtype = af::DT_INT64;
  *concat.y.axis = {z0.id, z1.id};
  *concat.y.repeats = {s0, s1};
  *concat.y.strides = {s1, One};

  af::ascir_op::Store store("store");
  store.x = concat.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = af::DT_INT64;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};

  af::ascir_op::Output output0("output0");
  output0.x = store.y;
  output0.attr.sched.axis = {z0.id, z1.id};
  output0.attr.api.type = af::ApiType::kAPITypeBuffer;
  output0.y.dtype = af::DT_INT64;
  output0.ir_attr.SetIndex(0);

  af::ascir_op::Data x2("x2", graph);
  x2.attr.sched.axis = {z0.id, z1.id};
  x2.y.dtype = af::DT_INT64;
  x2.ir_attr.SetIndex(2);

  af::ascir_op::Load load3("load3");
  load3.x = x2.y;
  load3.attr.sched.axis = {z0.id, z1.id};
  load3.y.dtype = af::DT_INT64;
  *load3.y.axis = {z0.id, z1.id};
  *load3.y.repeats = {s0, s1};
  *load3.y.strides = {s1, One};

  af::ascir_op::Store store1("store1");
  store1.x = load3.y;
  store1.attr.sched.axis = {z0.id, z1.id};
  store1.y.dtype = af::DT_INT64;
  *store1.y.axis = {z0.id, z1.id};
  *store1.y.repeats = {s0, s1};
  *store1.y.strides = {s1, One};

  af::ascir_op::Output y("y");
  y.x = store1.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = af::DT_INT64;
  y.ir_attr.SetIndex(0);

  auto axis = graph.GetAllAxis();
  axis.erase(axis.begin());
  const auto graph_attr = af::AscGraphUtils::GetComputeGraph(graph)->GetOrCreateAttrsGroup<af::AscGraphAttr>();
  graph_attr->axis = axis;

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 2UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 2UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);

  auto impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  auto res_axis = impl_graph.GetAllAxis();
  for (size_t i = 0UL; i < res_axis.size(); i++) {
    EXPECT_EQ(res_axis[i]->id, i);
  }

  auto load_node = impl_graph.FindNode("load");
  ASSERT_NE(nullptr, load_node);
  EXPECT_EQ(std::string(load_node->outputs[0].attr.vectorized_strides[0].Str().get()), "4");
  EXPECT_EQ(std::string(load_node->outputs[0].attr.vectorized_strides[1].Str().get()), "0");
  auto concat_node = impl_graph.FindNode("concat");
  ASSERT_NE(nullptr, concat_node);
  EXPECT_EQ(std::string(concat_node->outputs[0].attr.vectorized_strides[0].Str().get()), "4");
  EXPECT_EQ(std::string(concat_node->outputs[0].attr.vectorized_strides[1].Str().get()), "1");
}

TEST_F(OptimizerSt, concat_last1dim_small_tail_api) {
  af::AscGraph graph("LoadAbsStore");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = af::Symbol(2);

  auto tmp = graph.CreateAxis("tmp", s0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data x("x", graph);
  x.attr.sched.axis = {z0.id, z1.id};
  x.y.dtype = af::DT_FLOAT16;
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load load("load");
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id};
  load.y.dtype = af::DT_FLOAT16;
  *load.y.axis = {z0.id, z1.id};
  *load.y.repeats = {s0, One};
  *load.y.strides = {One, One};

  af::ascir_op::Data x1("x1", graph);
  x1.attr.sched.axis = {z0.id, z1.id};
  x1.y.dtype = af::DT_FLOAT16;
  x1.ir_attr.SetIndex(1);

  af::ascir_op::Load load1("load1");
  load1.x = x1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = af::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, One};
  *load1.y.strides = {One, One};

  af::ascir_op::Concat concat("concat");
  concat.x = {load.y, load1.y};
  concat.attr.sched.axis = {z0.id, z1.id};
  concat.y.dtype = af::DT_FLOAT16;
  *concat.y.axis = {z0.id, z1.id};
  *concat.y.repeats = {s0, s1};
  *concat.y.strides = {s1, One};

  af::ascir_op::Store store("store");
  store.x = concat.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = af::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};

  af::ascir_op::Output output0("output0");
  output0.x = store.y;
  output0.attr.sched.axis = {z0.id, z1.id};
  output0.attr.api.type = af::ApiType::kAPITypeBuffer;
  output0.y.dtype = af::DT_FLOAT16;
  output0.ir_attr.SetIndex(0);

  af::ascir_op::Load load3("load3");
  load3.x = output0.y;
  load3.attr.sched.axis = {z0.id, z1.id};
  load3.y.dtype = af::DT_FLOAT16;
  *load3.y.axis = {z0.id, z1.id};
  *load3.y.repeats = {s0, s1};
  *load3.y.strides = {s1, One};

  af::ascir_op::Store store1("store1");
  store1.x = load3.y;
  store1.attr.sched.axis = {z0.id, z1.id};
  store1.y.dtype = af::DT_FLOAT16;
  *store1.y.axis = {z0.id, z1.id};
  *store1.y.repeats = {s0, s1};
  *store1.y.strides = {s1, One};

  af::ascir_op::Output y("y");
  y.x = store1.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = af::DT_FLOAT16;
  y.ir_attr.SetIndex(0);

  auto axis = graph.GetAllAxis();
  axis.erase(axis.begin());
  const auto graph_attr = af::AscGraphUtils::GetComputeGraph(graph)->GetOrCreateAttrsGroup<af::AscGraphAttr>();
  graph_attr->axis = axis;

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);

  auto impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  auto res_axis = impl_graph.GetAllAxis();
  for (size_t i = 0; i < res_axis.size(); i++) {
    EXPECT_EQ(res_axis[i]->id, i);
  }

  auto load_node = impl_graph.FindNode("load");
  ASSERT_NE(nullptr, load_node);
  EXPECT_EQ(std::string(load_node->outputs[0].attr.vectorized_strides[0].Str().get()), "1");
  EXPECT_EQ(std::string(load_node->outputs[0].attr.vectorized_strides[1].Str().get()), "0");
  auto concat_node = impl_graph.FindNode("concat");
  ASSERT_NE(nullptr, concat_node);
  EXPECT_EQ(std::string(concat_node->outputs[0].attr.vectorized_strides[0].Str().get()), "2");
  EXPECT_EQ(std::string(concat_node->outputs[0].attr.vectorized_strides[1].Str().get()), "1");
}

TEST_F(OptimizerSt, transpose_axis_group) {
  // (0,1,2,3) -> (2,3,0,1) 会合并连续轴
  AscGraph graph("transpose_graph");
  graph.SetGraphType(af::AscGraphType::kImplGraph);
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);

  Data data_i("data_i", graph);
  data_i.ir_attr.SetIndex(0);
  data_i.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  data_i.y.dtype = af::DT_FLOAT16;
  *data_i.y.axis = {z0.id, z1.id, z2.id, z3.id};

  Load load_i("load_i");
  load_i.x = data_i.y;
  load_i.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  *load_i.y.axis = {z0.id, z1.id, z2.id, z3.id};
  data_i.y.dtype = af::DT_FLOAT16;
  *load_i.y.repeats = {s0, s1, s2, s3};
  *load_i.y.strides = {s1 * s2 * s3, s2 * s3, s3, af::ops::One};

  Transpose transpose("transpose");
  transpose.x = {load_i.y};
  transpose.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  transpose.y.dtype = af::DT_FLOAT16;
  *transpose.y.axis = {z2.id, z3.id, z0.id, z1.id};
  *transpose.y.repeats = {s2, s3, s0, s1};
  *transpose.y.strides = {s3 * s1 * s0, s1 * s0, s1, af::ops::One};

  Store store("store");
  store.x = transpose.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id};
  store.y.dtype = af::DT_FLOAT16;
  *store.y.axis = {z2.id, z3.id, z0.id, z1.id};
  *store.y.repeats = {s2, s3, s0, s1};
  *store.y.strides = {s3 * s1 * s0, s1 * s0, s1, af::ops::One};

  Output y("y");
  y.x = store.y;
  y.attr.sched.axis = {z2.id, z3.id, z0.id, z1.id};
  y.y.dtype = af::DT_FLOAT16;
  y.attr.api.type = af::ApiType::kAPITypeCompute;
  *y.y.axis = {z2.id, z3.id, z0.id, z1.id};
  y.ir_attr.SetIndex(0);
  auto transpose_node = graph.FindNode("transpose");

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 2);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups.size(), 1);
}

TEST_F(OptimizerSt, transpose_axis_group_2) {
  // (0,1) -> (1,0)
  AscGraph graph("transpose_graph");
  graph.SetGraphType(af::AscGraphType::kImplGraph);
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  Data data_i("data_i", graph);
  data_i.ir_attr.SetIndex(0);
  data_i.attr.sched.axis = {z0.id, z1.id};
  data_i.y.dtype = af::DT_FLOAT16;
  *data_i.y.axis = {z1.id, z0.id};

  Load load_i("load_i");
  load_i.x = data_i.y;
  load_i.attr.sched.axis = {z0.id, z1.id};
  load_i.y.dtype = af::DT_FLOAT16;
  *load_i.y.axis = {z1.id, z0.id};
  *load_i.y.repeats = {s1, s0};
  *load_i.y.strides = {s0, af::ops::One};

  Transpose transpose("transpose");
  transpose.x = {load_i.y};
  transpose.attr.sched.axis = {z0.id, z1.id};
  transpose.y.dtype = af::DT_FLOAT16;
  *transpose.y.axis = {z0.id, z1.id};
  *transpose.y.repeats = {s0, s1};
  *transpose.y.strides = {s1, af::ops::One};

  Abs abs("abs");
  abs.x = {transpose.y};
  abs.attr.sched.axis = {z0.id, z1.id};
  abs.y.dtype = af::DT_FLOAT16;
  *abs.y.axis = {z0.id, z1.id};
  *abs.y.repeats = {s0, s1};
  *abs.y.strides = {s1, af::ops::One};

  Store store("store");
  store.x = abs.y;
  store.attr.sched.axis = {z0.id, z1.id};
  store.y.dtype = af::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, af::ops::One};

  Output y("y");
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id};
  y.y.dtype = af::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id};
  y.attr.api.type = af::ApiType::kAPITypeCompute;
  y.ir_attr.SetIndex(0);
  auto transpose_node = graph.FindNode("transpose");

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 2);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups.size(), 1);
}

TEST_F(OptimizerSt, transpose_axis_group_3) {
  // (0,1,2) -> (1,0,2)
  AscGraph graph("transpose_graph");
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting(false, DynamicMode::kDynamic));
  SetCurShapeEnvContext(&shape_env);
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = shape_env.CreateSymbol(256, MakeShared<ge::GraphInputShapeSourceStub>(0, 0));
  graph.SetGraphType(af::AscGraphType::kImplGraph);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data_i("data_i", graph);
  data_i.ir_attr.SetIndex(0);
  data_i.attr.sched.axis = {z0.id, z1.id, z2.id};
  data_i.y.dtype = af::DT_FLOAT16;
  *data_i.y.axis = {z1.id, z0.id, z2.id};

  Load load_i("load_i");
  load_i.x = data_i.y;
  load_i.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_i.y.axis = {z1.id, z0.id, z2.id};
  *load_i.y.repeats = {s1, s0, s2};
  *load_i.y.strides = {s0 * s2, s2, af::ops::One};

  Transpose transpose("transpose");
  transpose.x = {load_i.y};
  transpose.attr.sched.axis = {z0.id, z1.id, z2.id};
  transpose.y.dtype = af::DT_FLOAT16;
  *transpose.y.axis = {z0.id, z1.id, z2.id};
  *transpose.y.repeats = {s0, s1, s2};
  *transpose.y.strides = {s1 * s2, s2, af::ops::One};

  Abs abs("abs");
  abs.x = {transpose.y};
  abs.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs.y.dtype = af::DT_FLOAT16;
  *abs.y.axis = {z0.id, z1.id, z2.id};
  *abs.y.repeats = {s0, s1, s2};
  *abs.y.strides = {s1 * s2, s2, af::ops::One};

  Store store("store");
  store.x = abs.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.y.dtype = af::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1 * s2, s2, af::ops::One};

  Output y("y");
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id};
  y.y.dtype = af::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, z2.id};
  y.attr.api.type = af::ApiType::kAPITypeCompute;
  y.ir_attr.SetIndex(0);
  auto transpose_node = graph.FindNode("transpose");

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 2);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups.size(), 1);
  SetCurShapeEnvContext(nullptr);
}

TEST_F(OptimizerSt, transpose_axis_group_4) {
  AscGraph graph("transpose_graph");
  graph.SetGraphType(af::AscGraphType::kImplGraph);
  auto s0 = graph.CreateSizeVar(16);
  auto s1 = graph.CreateSizeVar(86);
  auto s2 = graph.CreateSizeVar(36);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  Data data_0("data_0", graph);
  data_0.ir_attr.SetIndex(0);
  data_0.attr.sched.axis = {z0.id, z1.id, z2.id};
  data_0.y.dtype = af::DT_FLOAT16;
  *data_0.y.axis = {z0.id, z1.id, z2.id};

  Load load_0("load_0");
  load_0.x = data_0.y;
  load_0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_0.y.axis = {z0.id, z1.id, z2.id};
  *load_0.y.repeats = {s0, s1, s2};
  *load_0.y.strides = {s1 * s2, s2, af::ops::One};

  Data data_1("data_1", graph);
  data_1.ir_attr.SetIndex(1);
  data_1.attr.sched.axis = {z1.id, z0.id, z2.id};
  data_1.y.dtype = af::DT_FLOAT16;
  *data_1.y.axis = {z1.id, z0.id, z2.id};

  Load load_1("load_1");
  load_1.x = data_1.y;
  load_1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load_1.y.axis = {z1.id, z0.id, z2.id};
  *load_1.y.repeats = {s1, s0, s2};
  *load_1.y.strides = {s0 * s2, s2, af::ops::One};

  Transpose transpose("transpose");
  transpose.x = {load_1.y};
  transpose.attr.sched.axis = {z0.id, z1.id, z2.id};
  transpose.y.dtype = af::DT_FLOAT16;
  *transpose.y.axis = {z0.id, z1.id, z2.id};
  *transpose.y.repeats = {s0, s1, s2};
  *transpose.y.strides = {s1 * s2, s2, af::ops::One};

  Mul mul("mul");
  mul.x1 = {load_0.y};
  mul.x2 = {transpose.y};
  mul.attr.sched.axis = {z0.id, z1.id, z2.id};
  mul.y.dtype = af::DT_FLOAT16;
  *mul.y.axis = {z0.id, z1.id, z2.id};
  *mul.y.repeats = {s0, s1, s2};
  *mul.y.strides = {s1 * s2, s2, af::ops::One};

  Store store("store");
  store.x = mul.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.y.dtype = af::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1 * s2, s2, af::ops::One};

  Output y("y");
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id};
  y.y.dtype = af::DT_FLOAT16;
  *y.y.axis = {z0.id, z1.id, z2.id};
  y.attr.api.type = af::ApiType::kAPITypeCompute;
  y.ir_attr.SetIndex(0);
  auto transpose_node = graph.FindNode("transpose");

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 2);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups.size(), 1);
}

TEST_F(OptimizerSt, TestVecoutNotReusable) {
  const auto s0 = Sym("s0");
  const auto s1 = Sym("s1");
  const auto s2 = Sym("s2");

  auto graph = AscGraphBuilder("shorten_load")
                   .Loops({s0, s1})
                   .Data("x0", 0)
                   .Load("load0", "x0", {s0, s2}, {s2, af::sym::kSymbolOne})
                   .Abs("abs0", "load0")
                   .Store("store0", "abs0")
                   .Output("output0", "store0", 0)
                   .Load("load1", "x0", {s0, s2}, {s2, af::sym::kSymbolOne})
                   .Abs("abs1", "load1")
                   .Store("store1", "abs1")
                   .Output("output1", "store1", 1)
                   .Load("load2", "x0", {s0, s2}, {s2, af::sym::kSymbolOne})
                   .Abs("abs2", "load2")
                   .Store("store2", "abs2")
                   .Output("output2", "store2", 2)
                   .Load("load3", "x0", {s0, s2}, {s2, af::sym::kSymbolOne})
                   .Abs("abs3", "load3")
                   .Store("store3", "abs3")
                   .Output("output3", "store3", 3)
                   .Load("load4", "x0", {s0, s2}, {s2, af::sym::kSymbolOne})
                   .Abs("abs4", "load4")
                   .Store("store4", "abs4")
                   .Output("output4", "store4", 4)
                   .Concat("concat", {"abs0", "abs1", "abs2", "abs3", "abs4"})
                   .Store("store5", "concat")
                   .Output("output5", "store5", 5)
                   .Build();

  ::ascir::FusedScheduledResult fused_scheduled;
  int res = optimizer.Optimize(graph, fused_scheduled);
  EXPECT_EQ(res, 0);
  EXPECT_EQ(fused_scheduled.node_idx_to_scheduled_results[0].size(), 3);
  EXPECT_EQ(fused_scheduled.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1);
  // EXPECT_EQ(fused_scheduled.node_idx_to_scheduled_results[0][1].schedule_groups.size(), 5);
}

TEST_F(OptimizerSt, ConcatFirstDim) {
  af::AscGraph graph("concat_1st_dim_graph");

  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s0 + s1);

  Data x1_op("x1", graph);
  x1_op.ir_attr.SetIndex(0);
  Data x2_op("x2", graph);
  x2_op.ir_attr.SetIndex(1);
  Data x3_op("x3", graph);
  x3_op.ir_attr.SetIndex(2);

  Load load_op1("load1");
  Load load_op2("load2");
  Load load_op3("load3");

  std::vector<Data> all_data{x1_op, x2_op, x3_op};
  std::vector<Load> all_load{load_op1, load_op2, load_op3};

  for (size_t i = 0U; i < all_data.size(); ++i) {
    auto &x_op = all_data[i];
    auto &load_op = all_load[i];
    x_op.y.dtype = af::DT_FLOAT16;
    load_op.x = x_op.y;
    load_op.attr.sched.axis = {z3.id, z2.id};
    load_op.y.dtype = af::DT_FLOAT16;
    *load_op.y.axis = {z3.id, z2.id};
    load_op.y.dtype = af::DT_FLOAT16;
    *load_op.y.strides = {s2, af::ops::One};
    *load_op.y.repeats = {s0, s2};
  }
  load_op3.attr.sched.axis = {z3.id, z2.id};
  *load_op3.y.axis = {z3.id, z2.id};
  *load_op3.y.repeats = {s1, s2};

  af::ascir_op::Add add_op("add");
  add_op.attr.sched.axis = {z3.id, z2.id};
  add_op.x1 = load_op1.y;
  add_op.x2 = load_op2.y;
  add_op.y.dtype = af::DT_FLOAT16;
  *add_op.y.axis = {z3.id, z2.id};
  *add_op.y.strides = {s2, af::ops::One};
  *add_op.y.repeats = {s0, s2};

  af::ascir_op::Abs abs_op("abs");
  abs_op.attr.sched.axis = {z3.id, z2.id};
  abs_op.x = load_op3.y;
  abs_op.y.dtype = af::DT_FLOAT16;
  *abs_op.y.axis = {z3.id, z2.id};
  *abs_op.y.strides = {s2, af::ops::One};
  *abs_op.y.repeats = {s1, s2};

  af::ascir_op::Concat concat_op("concat");
  concat_op.attr.sched.axis = {z3.id, z2.id};
  concat_op.x = {add_op.y, abs_op.y};
  concat_op.y.dtype = af::DT_FLOAT16;
  *concat_op.y.axis = {z3.id, z2.id};
  *concat_op.y.repeats = {s0 + s1, s2};
  *concat_op.y.strides = {s2, af::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z3.id, z2.id};

  store_op.x = concat_op.y;
  store_op.y.dtype = af::DT_FLOAT16;
  *store_op.y.axis = {z3.id, z2.id};
  *store_op.y.repeats = {s0 + s1, s2};
  *store_op.y.strides = {s2, af::ops::One};

  Output y_op("y");
  y_op.x = store_op.y;
  y_op.ir_attr.SetIndex(0);
  auto store_node = graph.FindNode("store");

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1);
  auto &schedule_result = fused_scheduled_result.node_idx_to_scheduled_results[0][0];

  std::vector<Expression> offsets;
  std::vector<Expression> expect = {Symbol(0), (s0 * s2), Symbol(0), (s0 * s2)};
  for (const auto &schedule_group : schedule_result.schedule_groups) {
    for (auto &sub_impl_graph : schedule_group.impl_graphs) {
      for (const auto &sub_node : sub_impl_graph.GetAllNodes()) {
        if (sub_node->GetType() == "Store") {
          Expression offset;
          EXPECT_EQ(sub_node->attr.ir_attr->GetAttrValue("offset", offset), 0);
          offsets.emplace_back(offset);
        }
      }
    }
  }

  for (size_t i = 0; i < offsets.size(); ++i) {
    EXPECT_SYMBOL_EQ(offsets[i], expect[i]);
  }
  EXPECT_EQ(fused_scheduled_result.input_nodes.size(), 3);
  EXPECT_EQ(fused_scheduled_result.output_nodes.size(), 1);
  EXPECT_EQ(fused_scheduled_result.workspace_nodes.size(), 0);
  EXPECT_EQ(fused_scheduled_result.input_nodes[0]->GetName(), "x1");
  EXPECT_EQ(fused_scheduled_result.input_nodes[1]->GetName(), "x2");
  EXPECT_EQ(fused_scheduled_result.input_nodes[2]->GetName(), "x3");
  EXPECT_EQ(fused_scheduled_result.output_nodes[0]->GetName(), "y");

  std::set<std::set<std::string>> axis_name_groups;
  for (const auto &schedule_group : schedule_result.schedule_groups) {
    std::set<std::string> axis_names;
    for (const auto &axis : schedule_group.impl_graphs[0].GetAllAxis()) {
      axis_names.emplace(axis->name);
    }
    axis_name_groups.emplace(std::move(axis_names));
  }

  const std::set<std::set<std::string>> expected_axis_name_groups = {
      {"z3z2_0", "z3z2_0T", "z3z2_0TB", "z3z2_0Tb", "z3z2_0t"},
      {"z3z2_1", "z3z2_1T", "z3z2_1TB", "z3z2_1Tb", "z3z2_1t"},
  };
  EXPECT_EQ(axis_name_groups, expected_axis_name_groups);
}

TEST_F(OptimizerSt, gather_last1dim) {
  af::AscGraph graph("LoadAbsStore");
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  af::ascir_op::Data data0("data0", graph);
  data0.y.dtype = af::DT_FLOAT;
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data0.y.axis = {z0.id, z1.id, z2.id};
  *data0.y.repeats = {s0, s1, s2};
  *data0.y.strides = {s1 * s2, s2, One};

  af::ascir_op::Data data1("data1", graph);
  data1.y.dtype = af::DT_FLOAT;
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  data1.attr.api.type = af::ApiType::kAPITypeBuffer;
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z3.id, z4.id};
  *data1.y.axis = {z3.id, z4.id};
  *data1.y.repeats = {s3, s4};
  *data1.y.strides = {s4, One};

  af::ascir_op::Gather gather("gather");
  gather.attr.api.compute_type = ComputeType::kComputeGather;
  gather.x1 = data0.y;
  gather.x2 = data1.y;
  gather.ir_attr.SetAxis(2);
  gather.attr.sched.axis = {z0.id, z1.id, z3.id, z4.id};
  gather.y.dtype = af::DT_FLOAT;
  *gather.y.axis = {z0.id, z1.id, z3.id, z4.id};
  *gather.y.repeats = {s0, s1, s3, s4};
  *gather.y.strides = {s1 * s3 * s4, s3 * s4, s4, One};

  af::ascir_op::Abs abs("abs");
  abs.attr.api.compute_type = ComputeType::kComputeElewise;
  abs.x = gather.y;
  abs.attr.sched.axis = {z0.id, z1.id, z3.id, z4.id};
  abs.y.dtype = af::DT_FLOAT;
  *abs.y.axis = {z0.id, z1.id, z3.id, z4.id};
  *abs.y.repeats = {s0, s1, s3, s4};
  *abs.y.strides = {s1 * s3 * s4, s3 * s4, s4, One};

  af::ascir_op::Store store("store");
  store.attr.api.compute_type = ComputeType::kComputeElewise;
  store.x = abs.y;
  store.attr.sched.axis = {z0.id, z1.id, z3.id, z4.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z0.id, z1.id, z3.id, z4.id};
  *store.y.repeats = {s0, s1, s3, s4};
  *store.y.strides = {s1 * s3 * s4, s3 * s4, s4, One};

  af::ascir_op::Output y("y");
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z3.id, z4.id};
  y.y.dtype = af::DT_FLOAT;
  y.ir_attr.SetIndex(0);

  auto axis = graph.GetAllAxis();
  const auto graph_attr = af::AscGraphUtils::GetComputeGraph(graph)->GetOrCreateAttrsGroup<af::AscGraphAttr>();
  graph_attr->axis = axis;

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);

  auto impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  auto res_axis = impl_graph.GetAllAxis();
  EXPECT_EQ(res_axis[5]->size, s0 * s1);
  EXPECT_EQ(res_axis[6]->size, s3 * s4);

  auto store_node = impl_graph.FindNode("store");
  ASSERT_NE(nullptr, store_node);
  auto gather_node = impl_graph.FindNode("gather");
  ASSERT_NE(nullptr, gather_node);

  std::set<std::string> axis_names_0;
  for (const auto &axis :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllAxis()) {
    axis_names_0.emplace(axis->name);
  }

  std::set<std::string> expected_0{"z0", "z1", "z2", "z3", "z4", "z0z1", "z3z4", "z3z4T", "z3z4t", "z0z1B", "z0z1b"};
  EXPECT_EQ(axis_names_0, expected_0);
}
}  // namespace optimize

TEST_F(OptimizerSt, ConcatTailDim_SplitConcat) {
  af::AscGraph graph("concat_last_dim_graph");

  std::vector<std::string> concat_dim_sizes{"412", "1",  "6",  "6",  "6", "6", "16", "16", "33", "16", "32",
                                            "32",  "s1", "s2", "32", "1", "2", "3",  "16", "1",  "222"};
  auto s0 = graph.CreateSizeVar("s0");
  auto concat_size = af::Expression(af::Symbol(0));
  std::vector<std::shared_ptr<Data>> data_ops;
  std::vector<AscOpOutput> outputs;
  for (size_t i = 0; i < concat_dim_sizes.size(); ++i) {
    af::Expression s_i;
    if (concat_dim_sizes[i][0] == 's') {
      s_i = graph.CreateSizeVar(concat_dim_sizes[i]);
    } else {
      s_i = graph.CreateSizeVar(std::strtol(concat_dim_sizes[i].c_str(), nullptr, 10));
    }
    concat_size = (concat_size + s_i);
    auto data_op = std::make_shared<Data>(("Data" + std::to_string(i + 1)).c_str(), graph);
    data_op->y.dtype = af::DT_FLOAT16;
    *data_op->y.repeats = {s0, s_i};
    *data_op->y.strides = {s_i, af::ops::One};
    data_ops.emplace_back(data_op);
    outputs.emplace_back(data_ops.back()->y);
  }

  af::ascir_op::Concat concat_op("concat");
  concat_op.x = outputs;
  concat_op.y.dtype = af::DT_FLOAT16;
  *concat_op.y.repeats = {s0, concat_size};
  *concat_op.y.strides = {concat_size, af::ops::One};

  auto concat_node = graph.FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);

  optimize::ConcatGroupPartitioner partitioner(concat_node, 1);
  std::vector<optimize::ConcatGroupPartitioner::ConcatGroup> groups;
  ASSERT_EQ(partitioner.PartitionGroups(groups), af::SUCCESS);
  std::vector<std::vector<std::string>> results;
  for (const auto &group : groups) {
    std::cout << "start: " << group.start << ", end: " << group.end << ", type: " << group.group_type << std::endl;
    std::vector<std::string> dims(concat_dim_sizes.begin() + static_cast<int64_t>(group.start),
                                  concat_dim_sizes.begin() + static_cast<int64_t>(group.end));
    std::cout << "  " << af::ToString(dims) << ", size = " << group.size << std::endl;
    results.emplace_back(dims);
  }
  EXPECT_EQ(results.size(), 7);
  EXPECT_EQ(results[0], (std::vector<std::string>{"412"}));
  EXPECT_EQ(results[1], (std::vector<std::string>{"1", "6", "6", "6", "6", "16", "16", "33"}));
  EXPECT_EQ(results[2], (std::vector<std::string>{"16", "32", "32"}));
  EXPECT_EQ(results[3], (std::vector<std::string>{"s1"}));
  EXPECT_EQ(results[4], (std::vector<std::string>{"s2"}));
  EXPECT_EQ(results[5], (std::vector<std::string>{"32", "1", "2", "3", "16", "1"}));
  EXPECT_EQ(results[6], (std::vector<std::string>{"222"}));
}

TEST_F(OptimizerSt, ConcatTailDim_SplitConcat_AlignAndSmallTail) {
  af::AscGraph graph("concat_last_dim_graph");

  std::vector<std::string> concat_dim_sizes{"32", "32", "32", "32", "32", "32", "16", "16", "16", "16", "16", "17"};
  auto s0 = graph.CreateSizeVar("s0");
  auto concat_size = af::Expression(af::Symbol(0));
  std::vector<std::shared_ptr<Data>> data_ops;
  std::vector<AscOpOutput> outputs;
  for (size_t i = 0; i < concat_dim_sizes.size(); ++i) {
    af::Expression s_i;
    if (concat_dim_sizes[i][0] == 's') {
      s_i = graph.CreateSizeVar(concat_dim_sizes[i]);
    } else {
      s_i = graph.CreateSizeVar(std::strtol(concat_dim_sizes[i].c_str(), nullptr, 10));
    }
    concat_size = (concat_size + s_i);
    auto data_op = std::make_shared<Data>(("Data" + std::to_string(i + 1)).c_str(), graph);
    data_op->y.dtype = af::DT_FLOAT16;
    *data_op->y.repeats = {s0, s_i};
    *data_op->y.strides = {s_i, af::ops::One};
    data_ops.emplace_back(data_op);
    outputs.emplace_back(data_ops.back()->y);
  }

  af::ascir_op::Concat concat_op("concat");
  concat_op.x = outputs;
  concat_op.y.dtype = af::DT_FLOAT16;
  *concat_op.y.repeats = {s0, concat_size};
  *concat_op.y.strides = {concat_size, af::ops::One};

  auto concat_node = graph.FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);
  optimize::ConcatGroupPartitioner partitioner(concat_node, 1);
  std::vector<optimize::ConcatGroupPartitioner::ConcatGroup> groups;
  ASSERT_EQ(partitioner.PartitionGroups(groups), af::SUCCESS);
  std::vector<std::vector<std::string>> results;
  for (const auto &group : groups) {
    std::cout << "start: " << group.start << ", end: " << group.end << ", type: " << group.group_type << std::endl;
    std::vector<std::string> dims(concat_dim_sizes.begin() + static_cast<int64_t>(group.start),
                                  concat_dim_sizes.begin() + static_cast<int64_t>(group.end));
    std::cout << "  " << af::ToString(dims) << ", size = " << group.size << std::endl;
    results.emplace_back(dims);
  }
  EXPECT_EQ(results.size(), 2);
}

// codegen pad算子暂未支持,先在ut/st中模拟整个流程, 后续删除
TEST_F(OptimizerSt, removepad_and_add_pad) {
  af::AscGraph graph("LoadAbsStore");
  auto s0 = af::Symbol(2);
  auto s1 = af::Symbol(3);
  auto s2 = af::Symbol(10);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data x("x", graph);
  x.attr.api.compute_type = ComputeType::kComputeInvalid;
  x.attr.api.type = af::ApiType::kAPITypeBuffer;
  x.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Load load("load");
  load.x = x.y;
  load.attr.sched.axis = {z0.id, z1.id, z2.id};
  load.attr.api.compute_type = ComputeType::kComputeLoad;
  load.y.dtype = af::DT_FLOAT16;
  *load.y.axis = {z0.id, z1.id, z2.id};
  *load.y.repeats = {s0, s1, s2};
  *load.y.strides = {s1 * s2 * s2, s2 * s2, s2};

  af::ascir_op::Abs abs("abs");
  abs.x = load.y;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs.attr.api.compute_type = ComputeType::kComputeElewise;
  abs.y.dtype = af::DT_FLOAT16;
  *abs.y.axis = {z0.id, z1.id, z2.id};
  *abs.y.repeats = {s0, s1, s2};
  *abs.y.strides = {s1 * s2 * s2, s2 * s2, s2};

  af::ascir_op::Max max("max");
  max.x = abs.y;
  max.attr.sched.axis = {z0.id, z1.id, z2.id};
  max.attr.api.compute_type = ComputeType::kComputeReduce;
  max.y.dtype = af::DT_FLOAT16;
  *max.y.axis = {z0.id, z1.id, z2.id};
  *max.y.repeats = {s0, s1, One};
  *max.y.strides = {s1, One, Zero};

  af::ascir_op::Store store("store");
  store.x = max.y;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.y.dtype = af::DT_FLOAT16;
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, One};
  *store.y.strides = {s1, One, Zero};

  af::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.sched.axis = {z0.id, z1.id, z2.id};
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Load load1("load1");
  load1.x = x.y;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.y.dtype = af::DT_INT64;
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2 * s2, s2 * s2, s2};

  af::ascir_op::Abs abs1("abs1");
  abs1.x = load1.y;
  abs1.attr.sched.axis = {z0.id, z1.id, z2.id};
  abs1.attr.api.compute_type = ComputeType::kComputeElewise;
  abs1.y.dtype = af::DT_INT64;
  *abs1.y.axis = {z0.id, z1.id, z2.id};
  *abs1.y.repeats = {s0, s1, s2};
  *abs1.y.strides = {s1 * s2 * s2, s2 * s2, s2};

  af::ascir_op::Store store1("store1");
  store1.x = abs1.y;
  store1.attr.sched.axis = {z0.id, z1.id, z2.id};
  store1.attr.api.compute_type = ComputeType::kComputeStore;
  store1.y.dtype = af::DT_INT64;
  *store1.y.axis = {z0.id, z1.id, z2.id};
  *store1.y.repeats = {s0, s1, One};
  *store1.y.strides = {s1, One, Zero};

  af::ascir_op::Output y1("y1");
  y1.x = store1.y;
  y1.attr.sched.axis = {z0.id, z1.id, z2.id};
  y1.attr.api.compute_type = ComputeType::kComputeInvalid;
  y1.attr.api.type = af::ApiType::kAPITypeBuffer;
  y1.y.dtype = af::DT_INT64;

  for (auto n : graph.GetAllNodes()) {
    if (optimize::ScheduleUtils::IsBuffer(n)) {
      continue;
    }
    n->outputs[0].attr.vectorized_axis = {z1.id, z2.id};
  }
  // codegen pad算子暂未支持,先在ut/st中模拟整个流程
  AlignmentStrategyShadow handler;
  EXPECT_EQ(handler.AccessSetAlignWidth(graph), af::SUCCESS);
  EXPECT_EQ(handler.AccessAddRemovePadForTailAxisDiscontinuousLoad(graph), af::SUCCESS);
  EXPECT_EQ(handler.AccessInferAlignment(graph), af::SUCCESS);
  EXPECT_EQ(handler.AccessAddPadForAlignmentConflictNode(graph), af::SUCCESS);
  EXPECT_EQ(handler.AccessSetVectorizedStrides(graph), af::SUCCESS);

  std::vector<af::Expression> golden1 = {af::Symbol(160), af::Symbol(16)};
  std::vector<af::Expression> golden2 = {af::Symbol(16), af::Symbol(1)};
  auto load_node = graph.FindNode("load");
  ASSERT_NE(load_node, nullptr);
  EXPECT_EQ(load_node->outputs[0].attr.vectorized_strides, golden1);

  auto max_node = graph.FindNode("max");
  ASSERT_NE(max_node, nullptr);
  EXPECT_EQ(max_node->inputs[0].attr.vectorized_strides, golden2);

  std::vector<af::Expression> golden3 = {af::Symbol(40), af::Symbol(4)};
  auto load1_node = graph.FindNode("load1");
  ASSERT_NE(load1_node, nullptr);
  EXPECT_EQ(load1_node->outputs[0].attr.vectorized_strides, golden3);
}

TEST_F(OptimizerSt, ConcatTailDim_SplitConcat_412_1) {
  af::AscGraph graph("concat_last_dim_graph");

  std::vector<std::string> concat_dim_sizes(412, "1");
  concat_dim_sizes.emplace_back("16");
  concat_dim_sizes.emplace_back("16");
  concat_dim_sizes.emplace_back("1");
  concat_dim_sizes.emplace_back("2");
  auto s0 = graph.CreateSizeVar("s0");
  auto concat_size = af::Expression(af::Symbol(0));
  std::vector<std::shared_ptr<Data>> data_ops;
  std::vector<AscOpOutput> outputs;
  for (size_t i = 0; i < concat_dim_sizes.size(); ++i) {
    af::Expression s_i;
    if (concat_dim_sizes[i][0] == 's') {
      s_i = graph.CreateSizeVar(concat_dim_sizes[i]);
    } else {
      s_i = graph.CreateSizeVar(std::strtol(concat_dim_sizes[i].c_str(), nullptr, 10));
    }
    concat_size = (concat_size + s_i);
    auto data_op = std::make_shared<Data>(("Data" + std::to_string(i + 1)).c_str(), graph);
    data_op->y.dtype = af::DT_FLOAT;
    *data_op->y.repeats = {s0, s_i};
    *data_op->y.strides = {s_i, af::ops::One};
    data_ops.emplace_back(data_op);
    outputs.emplace_back(data_ops.back()->y);
  }

  af::ascir_op::Concat concat_op("concat");
  concat_op.x = outputs;
  concat_op.y.dtype = af::DT_FLOAT;
  *concat_op.y.repeats = {s0, concat_size};
  *concat_op.y.strides = {concat_size, af::ops::One};

  auto concat_node = graph.FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);

  optimize::ConcatGroupPartitioner partitioner(concat_node, 1);
  std::vector<optimize::ConcatGroupPartitioner::ConcatGroup> groups;
  ASSERT_EQ(partitioner.PartitionGroups(groups), af::SUCCESS);
  std::vector<std::vector<std::string>> results;
  for (const auto &group : groups) {
    std::cout << "start: " << group.start << ", end: " << group.end << ", type: " << group.group_type << std::endl;
    std::vector<std::string> dims(concat_dim_sizes.begin() + static_cast<int64_t>(group.start),
                                  concat_dim_sizes.begin() + static_cast<int64_t>(group.end));
    std::cout << "  " << af::ToString(dims) << ", size = " << group.size << std::endl;
    results.emplace_back(dims);
  }
  EXPECT_EQ(results.size(), 13);
  std::vector<std::string> expect = {28, "1"};
  expect.push_back("16");
  expect.push_back("16");
  expect.push_back("1");
  expect.push_back("2");
  EXPECT_EQ(results[12], expect);
}

TEST_F(OptimizerSt, ConcatTailDim_SplitConcat_ConvertSmallGroup) {
  af::AscGraph graph("concat_last_dim_graph");
  std::vector<int> concat_dim_sizes{64, 6, 28, 42};
  auto s0 = graph.CreateSizeVar(32 * 64);
  auto concat_size = af::Expression(af::Symbol(0));
  std::vector<std::shared_ptr<Data>> data_ops;
  std::vector<AscOpOutput> outputs;
  for (size_t i = 0; i < concat_dim_sizes.size(); ++i) {
    af::Expression s_i;
    s_i = graph.CreateSizeVar(concat_dim_sizes[i]);
    concat_size = (concat_size + s_i);
    auto data_op = std::make_shared<Data>(("Data" + std::to_string(i + 1)).c_str(), graph);
    data_op->y.dtype = af::DT_FLOAT;
    *data_op->y.repeats = {s0, s_i};
    *data_op->y.strides = {s_i, af::ops::One};
    data_ops.emplace_back(data_op);
    outputs.emplace_back(data_ops.back()->y);
  }

  af::ascir_op::Concat concat_op("concat");
  concat_op.x = outputs;
  concat_op.y.dtype = af::DT_FLOAT;
  *concat_op.y.repeats = {s0, concat_size};
  *concat_op.y.strides = {concat_size, af::ops::One};

  auto concat_node = graph.FindNode("concat");
  ASSERT_TRUE(concat_node != nullptr);
  ::optimize::ConcatGroupPartitioner partitioner(concat_node, 1);
  std::vector<::optimize::ConcatGroupPartitioner::ConcatGroup> groups;
  ASSERT_EQ(partitioner.PartitionGroups(groups), af::SUCCESS);
  size_t index = 0;
  size_t last_end = 0;
  for (const auto &group : groups) {
    std::cout << "index: " << index << ", start: " << group.start << ", end: " << group.end
              << ", type: " << group.group_type << std::endl;
    std::vector<int> dims(concat_dim_sizes.begin() + static_cast<int64_t>(group.start),
                          concat_dim_sizes.begin() + static_cast<int64_t>(group.end));
    std::cout << "  " << af::ToString(dims) << "count = " << group.end - group.start << ", size = " << group.size
              << std::endl;
    EXPECT_EQ(group.start, last_end);
    last_end = group.end;
    ++index;
  }
  EXPECT_EQ(groups.size(), 2);
}

TEST_F(OptimizerSt, LoadOpSequenceAdjustCase) {
  const auto s0 = af::Symbol(64);
  const auto s1 = af::Symbol(64);
  const auto all_one = std::vector<Expression>(2, af::sym::kSymbolOne);
  const auto all_zero = std::vector<Expression>(2, af::sym::kSymbolZero);

  auto graph = AscGraphBuilder("reorder_load_op")
                   .Loops({s0, s1})
                   .Data("data0", 0, all_one, all_zero, af::DT_FLOAT)
                   .Load("load0", "data0", all_one, all_zero)
                   .Broadcast("broadcast0", "load0", {s0, s1})
                   .Data("data1", 1, {s0, s1}, {s1, af::sym::kSymbolOne}, af::DT_FLOAT)
                   .Load("load1", "data1")
                   .Abs("abs", "load1")
                   .Data("data2", 2, all_one, all_zero, af::DT_FLOAT)
                   .Load("load2", "data2", all_one, all_zero)
                   .Broadcast("broadcast1", "load2", {s0, s1})
                   .Add("add", "abs", "broadcast1")
                   .Mul("mul", "broadcast0", "add")
                   .Store("store", "mul")
                   .Output("output", "store", 8, af::DT_FLOAT)
                   .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetOpDesc()->GetId() == 2) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Data");
    }
    if (node->GetOpDesc()->GetId() == 3) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Abs");
    }
    if (node->GetOpDesc()->GetId() == 4) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Load");
    }
    if (node->GetOpDesc()->GetId() == 5) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Add");
    }
    if (node->GetOpDesc()->GetId() == 6) {
      EXPECT_EQ(node->GetOpDesc()->GetType(), "Data");
    }
  }
}

TEST_F(OptimizerSt, platform_reg_test) {
  af::AscGraph graph("tmp");
  auto platform_v1 = optimize::PlatformFactory::GetInstance().GetPlatform();
  EXPECT_NE(platform_v1, nullptr);
  auto platform_v1_new = optimize::PlatformFactory::GetInstance().GetPlatform();
  EXPECT_EQ(platform_v1, platform_v1_new);

  EXPECT_EQ(platform_v1->PartitionSubFunctions(graph), af::SUCCESS);
}

TEST_F(OptimizerSt, platform_config_test) {
  auto platform_v1 = optimize::PlatformFactory::GetInstance().GetPlatform();
  ASSERT_NE(platform_v1, nullptr);
  const auto &config = platform_v1->GetPlatformConfig();
  EXPECT_FALSE(config.is_default_enabled);
  EXPECT_FALSE(config.is_support_compat_mode);
  EXPECT_EQ(config.max_que_num, 4UL);
}

TEST_F(OptimizerSt, ReduceNeedAlignment) {
  const Expression s0 = af::Symbol(7);
  const Expression s1 = af::Symbol(8);
  const Expression s2 = af::Symbol(9);
  const Expression s3 = af::Symbol(10);

  // Max output: shape {1, s1, 1, s3}, strides {0, s3, 0, 1}
  std::vector<Expression> max_shape = {af::sym::kSymbolOne, s1, af::sym::kSymbolOne, s3};
  std::vector<Expression> max_strides = {af::sym::kSymbolZero, s3, af::sym::kSymbolZero, af::sym::kSymbolOne};

  auto graph = AscGraphBuilder("ReduceNeedAlignment")
                   .Loops({s0, s1, s2, s3})
                   .Data("arg4_1", 0, af::DT_FLOAT)
                   .Load("b0_load", "arg4_1")
                   .Abs("abs", "b0_load")
                   .Max("b0_max", "abs", {0, 2})  // Max reduce on axes {0, 2}
                   .Store("b3_store", "b0_max", max_shape, max_strides)
                   .Output("buf3", "b3_store", 0, af::DT_FLOAT)
                   .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  // Max 对 {0,2} 两根轴归约, 单 reduce 双 R 轴: phase-1 各 tiling case 切分结构不同(relation 不一致),
  // AppendReduceFirstStageGroup 回退逐 case 展开, 6 = Common 1 + RCore 展开 4 + AllLoad 1
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 6UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 4UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups.size(), 2UL);

  const auto &impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[1];
  const auto &reduce_node = impl_graph.FindNode("b0_max");
  std::vector<Expression> golden_stride = {
      af::sym::kSymbolZero,
      Symbol(16),
      af::sym::kSymbolOne,
  };
  EXPECT_EQ(reduce_node->outputs[0].attr.vectorized_strides, golden_stride);
}

TEST_F(OptimizerSt, ConstantToStoreNeedBroadCast) {
  const Expression s0 = af::Symbol(128);

  auto graph = AscGraphBuilder("test_graph")
                   .Loops({s0})
                   .Scalar("const", "998.998f", af::DT_FLOAT)
                   .Store("store", "const")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();

  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
  auto optimize_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];

  auto cg = af::AscGraphUtils::GetComputeGraph(optimize_graph);
  auto found_broadcast = cg->FindFirstNodeMatchType(af::ascir_op::Broadcast::Type);
  ASSERT_NE(found_broadcast, nullptr);
  auto asc_broadcast = AscNode(found_broadcast->GetOpDesc(), nullptr);

  auto found_store = cg->FindFirstNodeMatchType(af::ascir_op::Store::Type);
  ASSERT_NE(found_store, nullptr);
  auto asc_store = AscNode(found_store->GetOpDesc(), nullptr);
}

TEST_F(OptimizerSt, ExpandDimsForAllReduce) {
  const Expression s0 = af::Symbol(128);
  const Expression s1 = af::Symbol(64);

  // Sum output: shape {1, 1}, strides {0, 0}
  std::vector<Expression> sum_shape = {af::sym::kSymbolOne, af::sym::kSymbolOne};
  std::vector<Expression> sum_strides = {af::sym::kSymbolZero, af::sym::kSymbolZero};

  auto graph = AscGraphBuilder("all_reduce")
                   .Loops({s0, s1})
                   .Data("data", 0, af::DT_FLOAT)
                   .Load("load", "data")
                   .Sum("sum", "load", {0, 1})  // Sum reduce on both axes
                   .Store("store1", "sum", sum_shape, sum_strides)
                   .Output("output", "store1", 0, af::DT_FLOAT)
                   .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, 0);
}

TEST_F(OptimizerSt, transpose_fp32) {
  af::AscGraph graph("Transpose");
  auto s0 = af::Symbol(10);
  auto s1 = af::Symbol(1);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.y.dtype = af::DT_FLOAT;
  data0.ir_attr.SetIndex(1);
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.api.compute_type = af::ComputeType::kComputeLoad;
  load0.attr.sched.axis = {z1.id, z0.id};
  load0.y.dtype = af::DT_FLOAT;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};
  *load0.y.vectorized_axis = {z0.id, z1.id};

  af::ascir_op::Transpose transpose("transpose");
  transpose.x = load0.y;
  transpose.attr.api.compute_type = af::ComputeType::kComputeTranspose;
  transpose.attr.sched.axis = {z1.id, z0.id};
  transpose.y.dtype = af::DT_FLOAT;
  *transpose.y.axis = {z1.id, z0.id};
  *transpose.y.repeats = {s1, s0};
  *transpose.y.strides = {s0, One};
  *transpose.y.vectorized_axis = {z1.id, z0.id};

  af::ascir_op::Store store("store");
  store.x = load0.y;
  store.attr.api.compute_type = af::ComputeType::kComputeStore;
  store.attr.sched.axis = {z1.id, z0.id};
  store.y.dtype = af::DT_FLOAT;
  *store.y.axis = {z1.id, z0.id};
  *store.y.repeats = {s1, s0};
  *store.y.strides = {s0, One};
  *store.y.vectorized_axis = {z0.id, z1.id};

  optimize::autoschedule::AlignmentHandler handler;
  ASSERT_EQ(handler.AlignVectorizedStrides(graph), af::SUCCESS);

  auto load0_node = graph.FindNode("load0");
  ASSERT_NE(load0_node, nullptr);
  auto stode_node = graph.FindNode("store");
  ASSERT_NE(stode_node, nullptr);

  std::vector<Expression> golden_strides = {Symbol(16), Symbol(0)};
  std::vector<Expression> golden_strides1 = {Symbol(16), Symbol(0)};
  EXPECT_EQ(load0_node->outputs[0].attr.vectorized_strides, golden_strides);
  EXPECT_EQ(stode_node->outputs[0].attr.vectorized_strides, golden_strides);
}

/**
 *                 add
 *              /      \
 *            /         \
 *          /            \
 *        /             brc1
 *       |(s0,s1)        |(s0,1)
 *      brc0           load1
 *       |               |
 *     scalar          data1
 */
TEST_F(OptimizerSt, NodeCacheMarkerBroadcast) {
  auto s0 = af::Symbol(20);
  auto s1 = af::Symbol(32);

  auto graph =
      AscGraphBuilder("NodeCacheMarkerBroadcast")
          .Loops({s0, s1})
          .Scalar("data0", "0", af::DT_FLOAT16)
          .Broadcast("brc0", "data0", {s0, s1})
          .Data("data1", 0, {s0, af::sym::kSymbolOne}, {af::sym::kSymbolOne, af::sym::kSymbolZero}, af::DT_FLOAT16)
          .Load("load1", "data1", {s0, af::sym::kSymbolOne}, {af::sym::kSymbolOne, af::sym::kSymbolZero})
          .Broadcast("brc1", "load1", {s0, s1})
          .Add("add0", "brc0", "brc1")
          .Store("store0", "add0", {af::sym::kSymbolOne, s1}, {af::sym::kSymbolZero, af::sym::kSymbolOne})
          .Output("y0", "store0", 0, af::DT_FLOAT16)
          // 验证防止重复判断
          .Add("add1", "brc0", "brc1")
          .Store("store1", "add1", {af::sym::kSymbolOne, s1}, {af::sym::kSymbolZero, af::sym::kSymbolOne})
          .Output("y1", "store1", 1, af::DT_FLOAT16)
          .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  const auto &impl_graphs = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs;

  EXPECT_EQ(impl_graphs.size(), 2);
  const auto &impl0 = impl_graphs[0];
  const auto &impl0_scalar_node = impl0.FindNode("data0");
  EXPECT_NE(impl0_scalar_node, nullptr);
  EXPECT_EQ(impl0_scalar_node->attr.sched.exec_condition, ExecuteCondition::kNoCache);
  const auto &impl0_brc0_node = impl0.FindNode("brc0");
  EXPECT_NE(impl0_brc0_node, nullptr);
  EXPECT_EQ(impl0_brc0_node->attr.sched.exec_condition, ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis);
  const auto &impl0_brc1_node = impl0.FindNode("brc1");
  EXPECT_NE(impl0_brc1_node, nullptr);
  EXPECT_EQ(impl0_brc1_node->attr.sched.exec_condition, ExecuteCondition::kNoCache);
  const auto &impl0_add0_node = impl0.FindNode("add0");
  EXPECT_NE(impl0_add0_node, nullptr);
  EXPECT_EQ(impl0_add0_node->attr.sched.exec_condition, ExecuteCondition::kNoCache);

  const auto &impl1 = impl_graphs[1];
  const auto &impl1_scalar_node = impl1.FindNode("data0");
  EXPECT_NE(impl1_scalar_node, nullptr);
  EXPECT_EQ(impl1_scalar_node->attr.sched.exec_condition, ExecuteCondition::kNoCache);
  const auto &impl1_brc0_node = impl1.FindNode("brc0");
  EXPECT_NE(impl1_brc0_node, nullptr);
  EXPECT_EQ(impl1_brc0_node->attr.sched.exec_condition, ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis);
  const auto &impl1_brc1_node = impl1.FindNode("brc1");
  EXPECT_NE(impl1_brc1_node, nullptr);
  EXPECT_EQ(impl1_brc1_node->attr.sched.exec_condition, ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis);
  const auto &impl1_load1_node = impl1.FindNode("load1");
  EXPECT_NE(impl1_load1_node, nullptr);
  EXPECT_EQ(impl1_load1_node->attr.sched.exec_condition, ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis);
  const auto &impl1_add0_node = impl1.FindNode("add0");
  EXPECT_NE(impl1_add0_node, nullptr);
  EXPECT_EQ(impl1_add0_node->attr.sched.exec_condition, ExecuteCondition::kNoCache);
}

/**
 *                 add
 *              /      \
 *            /      removepad
 *          /            \
 *        /             brc1
 *       |(s0,s1)        |(1,s1)
 *      load0          load1
 *       |              |
 *     data0          data1
 */
TEST_F(OptimizerSt, NodeCacheMarkerRemovepad) {
  af::AscGraph graph("NodeCacheMarkerRemovepad");
  auto s0 = af::Symbol(20);
  auto s1 = af::Symbol(32);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = ComputeType::kComputeInvalid;
  data0.attr.api.type = af::ApiType::kAPITypeBuffer;
  data0.y.dtype = af::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};

  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.y.dtype = af::DT_FLOAT16;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};

  af::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = ComputeType::kComputeInvalid;
  data1.attr.api.type = af::ApiType::kAPITypeBuffer;
  data1.y.dtype = af::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id};

  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.y.dtype = af::DT_FLOAT16;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {One, s1};
  *load1.y.strides = {Zero, One};

  af::ascir_op::Broadcast brc1("brc1");
  brc1.x = load1.y;
  brc1.attr.sched.axis = {z0.id, z1.id};
  brc1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc1.y.dtype = af::DT_FLOAT16;
  *brc1.y.axis = {z0.id, z1.id};
  *brc1.y.repeats = {s0, s1};
  *brc1.y.strides = {s1, One};

  af::ascir_op::RemovePad remove_pad("remove_pad");
  remove_pad.x = brc1.y;
  remove_pad.attr.sched.axis = {z0.id, z1.id};
  remove_pad.attr.api.compute_type = ComputeType::kComputeElewise;
  remove_pad.y.dtype = af::DT_FLOAT16;
  *remove_pad.y.axis = {z0.id, z1.id};
  *remove_pad.y.repeats = {s0, s1};
  *remove_pad.y.strides = {s1, One};

  af::ascir_op::Add add0("add0");
  add0.x1 = load0.y;
  add0.x2 = remove_pad.y;
  add0.attr.sched.axis = {z0.id, z1.id};
  add0.attr.api.compute_type = ComputeType::kComputeElewise;
  add0.y.dtype = af::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id};
  *add0.y.repeats = {s0, s1};
  *add0.y.strides = {s1, One};

  af::ascir_op::Store store0("store0");
  store0.x = add0.y;
  store0.attr.sched.axis = {z0.id, z1.id};
  store0.attr.api.compute_type = ComputeType::kComputeStore;
  store0.y.dtype = af::DT_FLOAT16;
  *store0.y.axis = {z0.id, z1.id};
  *store0.y.repeats = {s0, s1};
  *store0.y.strides = {s1, One};

  af::ascir_op::Output y0("y0");
  y0.ir_attr.SetIndex(0);
  y0.x = store0.y;
  y0.attr.sched.axis = {z0.id, z1.id};
  y0.attr.api.compute_type = ComputeType::kComputeInvalid;
  y0.attr.api.type = af::ApiType::kAPITypeBuffer;
  y0.y.dtype = af::DT_FLOAT16;
  *y0.y.axis = {z0.id, z1.id};
  *y0.y.repeats = {s0, s1};
  *y0.y.strides = {s1, One};

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  const auto &impl_graphs = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs;
  EXPECT_EQ(impl_graphs.size(), 2);
  const auto &impl0 = impl_graphs[0];
  const auto &impl0_scalar_node = impl0.FindNode("data0");
  EXPECT_NE(impl0_scalar_node, nullptr);
  EXPECT_EQ(impl0_scalar_node->attr.sched.exec_condition, ExecuteCondition::kNoCache);
  const auto &impl0_load1_node = impl0.FindNode("load1");
  EXPECT_NE(impl0_load1_node, nullptr);
  EXPECT_EQ(impl0_load1_node->attr.sched.exec_condition, ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis);
  const auto &impl0_brc1_node = impl0.FindNode("brc1");
  EXPECT_NE(impl0_brc1_node, nullptr);
  EXPECT_EQ(impl0_brc1_node->attr.sched.exec_condition, ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis);
  const auto &impl0_remove_pad_node = impl0.FindNode("remove_pad");
  EXPECT_NE(impl0_remove_pad_node, nullptr);
  EXPECT_EQ(impl0_remove_pad_node->attr.sched.exec_condition, ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis);
  const auto &impl0_add0_node = impl0.FindNode("add0");
  EXPECT_NE(impl0_add0_node, nullptr);
  EXPECT_EQ(impl0_add0_node->attr.sched.exec_condition, ExecuteCondition::kNoCache);

  const auto &impl1 = impl_graphs[1];
  for (const auto &node : impl1.GetAllNodes()) {
    EXPECT_NE(node, nullptr);
    EXPECT_EQ(node->attr.sched.exec_condition, ExecuteCondition::kNoCache);
  }
}

TEST_F(OptimizerSt, StaticGraphRecomputeSplit) {
  const Expression s0 = af::Symbol(2048);
  const Expression s1 = af::Symbol(126);

  // Load with padding: shape {1, s1}, strides {0, 1}
  std::vector<Expression> load0_shape = {af::sym::kSymbolOne, s1};
  std::vector<Expression> load0_strides = {af::sym::kSymbolZero, af::sym::kSymbolOne};

  auto graph = AscGraphBuilder("StaticGraphRecomputeSplit")
                   .Loops({s0, s1})
                   .Data("data0", 0, af::DT_FLOAT16)
                   .Load("load0", "data0", load0_shape, load0_strides)
                   .Abs("abs0", "load0")
                   .Abs("abs1", "abs0")
                   .Store("store0", "abs1")
                   .Output("out0", "store0", 0, af::DT_FLOAT16)
                   .Broadcast("brc1", "abs1", {0})  // broadcast on axis 0: {1, s1} -> {s0, s1}
                   .Abs("abs2", "brc1")
                   .Store("store1", "abs2")
                   .Output("y0", "store1", 0, af::DT_FLOAT16)
                   .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
}

/**
 *            reduce0
 *               |
 *             add0
 *            /   \
 *          /      \
 *        /       brc1
 *       |         |
 *      load0    load1
 *       |        |
 *     data0    data1
 */
TEST_F(OptimizerSt, NodeCacheMarkerReduce) {
  const Expression s0 = af::Symbol(128);
  const Expression s1 = af::Symbol(2889);
  const Expression s2 = af::Symbol(4);

  // Load1 with padding: shape {s0, 1, s2}, strides {s2, 0, 1}
  std::vector<Expression> load1_shape = {s0, af::sym::kSymbolOne, s2};
  std::vector<Expression> load1_strides = {s2, af::sym::kSymbolZero, af::sym::kSymbolOne};

  // Sum output: shape {s0, 1, s2}, strides {s2, 0, 1}
  std::vector<Expression> sum_shape = {s0, af::sym::kSymbolOne, s2};
  std::vector<Expression> sum_strides = {s2, af::sym::kSymbolZero, af::sym::kSymbolOne};

  auto graph = AscGraphBuilder("NodeCacheMarkerReduce")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0, af::DT_FLOAT16)
                   .Load("load0", "data0")
                   .Data("data1", 1, af::DT_FLOAT16)
                   .Load("load1", "data1", load1_shape, load1_strides)
                   .Broadcast("brc1", "load1", {1})  // broadcast on axis 1: {s0, 1, s2} -> {s0, s1, s2}
                   .Add("add0", "load0", "brc1")
                   .Sum("reduce0", "add0", {1})  // Sum reduce on axis 1
                   .Store("store0", "reduce0", sum_shape, sum_strides)
                   .Output("y0", "store0", 0, af::DT_FLOAT16)
                   .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  // R-core phase-1 两个 tiling case 收敛为同 group 多 impl, RCore 任务的结果数 2 -> 1, 调度结果由 3 减为 2
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 2UL);
  const auto &impl_graphs = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs;
  EXPECT_EQ(impl_graphs.size(), 2);

  const auto &impl0 = impl_graphs[0];
  for (const auto &node : impl0.GetAllNodes()) {
    EXPECT_NE(node, nullptr);
    if (node->GetName() == "load1" || node->GetName() == "brc1") {
      EXPECT_EQ(node->attr.sched.exec_condition, ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis);
    } else {
      EXPECT_EQ(node->attr.sched.exec_condition, ExecuteCondition::kNoCache);
    }
  }
}

TEST_F(OptimizerSt, BackendSpec) {
  auto spec = optimize::BackendSpec::GetInstance();
  ASSERT_TRUE(spec != nullptr);
  ASSERT_EQ(spec->concat_max_input_num, 63);
  ASSERT_FALSE(spec->is_default_enabled);
}

TEST_F(OptimizerSt, AutofuseBackendSpecTest) {
  auto spec = ge::GetAutofuseBackendSpec();
  ASSERT_NE(spec, nullptr);
  EXPECT_FALSE(spec->is_default_enabled);
}

TEST_F(OptimizerSt, TestConcatBackwardFusionGraph_OptimizeSuccess) {
  ComputeGraphPtr compute_graph = BuildConcatBackwardFusion();
  ASSERT_NE(compute_graph, nullptr);

  auto ascbc1 = compute_graph->FindNode("ascbc1");
  ASSERT_NE(ascbc1, nullptr);
  auto ascbc2 = compute_graph->FindNode("ascbc2");
  ASSERT_NE(ascbc2, nullptr);

  auto concat_sub_graph = BuildConcatAscGraph("concat");
  auto add_sub_graph1 = BuildAddAscGraphAfterConcat("add");

  std::string concat_graph_str;
  af::AscGraphUtils::SerializeToReadable(concat_sub_graph, concat_graph_str);
  af::AttrUtils::SetStr(ascbc1->GetOpDescBarePtr(), "ascgraph", concat_graph_str);

  std::string add_graph_str;
  af::AscGraphUtils::SerializeToReadable(add_sub_graph1, add_graph_str);
  af::AttrUtils::SetStr(ascbc2->GetOpDescBarePtr(), "ascgraph", add_graph_str);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  ASSERT_EQ(optimizer.Optimize(compute_graph, fused_scheduled_result), 0);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.origin_vars.size(), 3UL);
  EXPECT_EQ(ToString(fused_scheduled_result.origin_vars[0]), "s0");
  EXPECT_EQ(ToString(fused_scheduled_result.origin_vars[1]), "s1");
  EXPECT_EQ(ToString(fused_scheduled_result.origin_vars[2]), "s2");

  auto impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];
  auto ascbc_1 = impl_graph.FindNode("ascbc1");
  EXPECT_EQ(ascbc_1, nullptr);
  auto ascbc_2 = impl_graph.FindNode("ascbc2");
  EXPECT_EQ(ascbc_2, nullptr);
}

TEST_F(OptimizerSt, PowSubstitutionCase1) {
  const auto s0 = af::Symbol(64);

  auto graph = AscGraphBuilder("pow1")
                   .Loops({s0})
                   .Data("data0", 0)
                   .Load("load0", "data0")
                   .Scalar("scalar0", "-0.500000000001")
                   .Op<af::ascir_op::Pow>("pow0", {"load0", "scalar0"})
                   .Scalar("scalar1", "-1")
                   .Op<af::ascir_op::Pow>("pow1", {"pow0", "scalar1"})
                   .Scalar("scalar2", "-2")
                   .Op<af::ascir_op::Pow>("pow2", {"pow1", "scalar2"})
                   .Scalar("scalar3", "3")
                   .Op<af::ascir_op::Pow>("pow3", {"pow2", "scalar3"})
                   .Scalar("scalar4", "4")
                   .Op<af::ascir_op::Pow>("pow4", {"pow3", "scalar4"})
                   .Store("store", "pow4")
                   .Output("out0", "store", 0)
                   .Build();

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  size_t brc_num = 0UL;
  size_t pow_num = 0UL;
  size_t mul_num = 0UL;
  size_t div_num = 0UL;

  for (const auto &node :
       fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0].GetAllNodes()) {
    if (node->GetType() == Pow::Type) {
      ++pow_num;
    } else if (node->GetType() == Mul::Type) {
      ++mul_num;
    } else if (node->GetType() == Div::Type) {
      ++div_num;
    } else if (node->GetType() == Broadcast::Type) {
      ++brc_num;
    }
  }
  EXPECT_EQ(brc_num, 0UL);
  EXPECT_EQ(pow_num, 0UL);
  EXPECT_EQ(mul_num, 5UL);
  EXPECT_EQ(div_num, 3UL);
}

TEST_F(OptimizerSt, PowWithTwoScalar) {
  const auto s0 = af::Symbol(64);

  auto graph = AscGraphBuilder("pow1")
                   .Loops({s0})
                   .Scalar("scalar0", "0.0")
                   .Scalar("scalar1", "1.0")
                   .Op<af::ascir_op::Pow>("pow0", {"scalar0", "scalar1"})
                   .Abs("abs", "pow0")
                   .Store("store", "abs")
                   .Output("out0", "store", 0)
                   .Build();

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
}

TEST(OptimizeST, TransposeSkipPadTilingCase) {
  af::AscGraph graph("trans_int64");
  auto s0 = af::Symbol(3);
  auto s1 = af::Symbol(10);
  auto s2 = af::Symbol(4);

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  std::vector<int64_t> axis_ids = {z2.id, z1.id, z0.id};
  Data data0("data0", graph);
  data0.y.dtype = af::DT_FLOAT;
  data0.ir_attr.SetIndex(0);

  Load load0("load0");
  load0.attr.sched.axis = axis_ids;
  load0.x = data0.y;
  load0.y.dtype = af::DT_FLOAT;
  *load0.y.axis = axis_ids;
  *load0.y.repeats = {s2, s1, s0};
  *load0.y.strides = {s1 * s0, s0, af::ops::One};

  Transpose transpose0("transpose0");
  transpose0.attr.sched.axis = axis_ids;
  transpose0.x = load0.y;
  transpose0.y.dtype = af::DT_FLOAT;
  *transpose0.y.axis = {z0.id, z1.id, z2.id};
  *transpose0.y.repeats = {s0, s1, s2};
  *transpose0.y.strides = {s1 * s2, s2, af::ops::One};

  Cast cast0("cast0");
  cast0.attr.sched.axis = axis_ids;
  cast0.x = transpose0.y;
  cast0.y.dtype = af::DT_INT64;
  *cast0.y.axis = {z0.id, z1.id, z2.id};
  *cast0.y.repeats = {s0, s1, s2};
  *cast0.y.strides = {s1 * s2, s2, af::ops::One};

  Store store0("store0");
  store0.attr.sched.axis = axis_ids;
  store0.x = cast0.y;
  store0.y.dtype = af::DT_INT64;
  *store0.y.axis = {z0.id, z1.id, z2.id};
  *store0.y.repeats = {s0, s1, s2};
  *store0.y.strides = {s1 * s2, s2, af::ops::One};

  Output out0("out0");
  out0.x = store0.y;
  out0.y.dtype = af::DT_INT64;
  out0.ir_attr.SetIndex(0);

  ::optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  optimize::Optimizer optimizer(optimize::OptimizerOptions{});
  ::ascir::FusedScheduledResult fused_scheduled_result;
  ASSERT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 2UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][1].schedule_groups[0].impl_graphs.size(), 2UL);

  bool found_reserved_transpose = false;
  for (const auto &node_results : fused_scheduled_result.node_idx_to_scheduled_results) {
    for (const auto &scheduled_results : node_results) {
      for (const auto &schedule_group : scheduled_results.schedule_groups) {
        found_reserved_transpose =
            CheckReservedTransposeImplGraphs(schedule_group.impl_graphs) || found_reserved_transpose;
      }
    }
  }
  EXPECT_TRUE(found_reserved_transpose);
}

TEST_F(OptimizerSt, vecoutCanBeReuse) {
  const Expression s0 = af::Symbol(31);

  auto graph = AscGraphBuilder("reuse")
                   .Loops({s0})
                   .Data("data0", 0)
                   .Load("load0", "data0")
                   .Abs("abs0", "load0")
                   .Store("store0", "abs0")
                   .Output("out0", "store0", 0)
                   .Abs("abs1", "abs0")
                   .template Op<af::ascir_op::Sigmoid>("sigmoid0", {"abs1"})
                   .Abs("abs2", "sigmoid0")
                   .Abs("abs3", "abs2")
                   .Store("store1", "abs3")
                   .Output("out1", "store1", 1)
                   .Build();

  optimize::AscGraphInfoComplete::CompleteApiInfo(graph);

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);

  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 1UL);
  auto impl_graph = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs[0];

  // abs1 reuse load0
  auto load0_node = impl_graph.FindNode("load0");
  ASSERT_NE(load0_node, nullptr);
  auto abs1_node = impl_graph.FindNode("abs1");
  ASSERT_NE(abs1_node, nullptr);
  int64_t que_id = load0_node->outputs[0].attr.que.id;
  EXPECT_EQ(load0_node->outputs[0].attr.mem.position, af::Position::kPositionVecIn);
  EXPECT_EQ(abs1_node->outputs[0].attr.mem.alloc_type, af::AllocType::kAllocTypeQueue);
  EXPECT_EQ(abs1_node->outputs[0].attr.que.id, que_id);
  EXPECT_NE(abs1_node->outputs[0].attr.mem.reuse_id, af::kIdNone);

  // sigmoid0 reuse abs0, abs2 inplace reuse sigmoid0
  auto abs0_node = impl_graph.FindNode("abs0");
  ASSERT_NE(abs0_node, nullptr);
  auto sigmoid0_node = impl_graph.FindNode("sigmoid0");
  ASSERT_NE(sigmoid0_node, nullptr);
  auto abs2_node = impl_graph.FindNode("abs2");
  ASSERT_NE(abs2_node, nullptr);

  auto abs3_node = impl_graph.FindNode("abs3");
  ASSERT_NE(abs3_node, nullptr);
  int64_t que1_id = abs3_node->outputs[0].attr.que.id;
  EXPECT_EQ(abs0_node->outputs[0].attr.mem.position, af::Position::kPositionVecOut);
  EXPECT_EQ(sigmoid0_node->outputs[0].attr.mem.alloc_type, af::AllocType::kAllocTypeQueue);
  EXPECT_EQ(sigmoid0_node->outputs[0].attr.que.id, que1_id);
  EXPECT_NE(sigmoid0_node->outputs[0].attr.mem.reuse_id, af::kIdNone);
  EXPECT_EQ(abs2_node->outputs[0].attr.mem.alloc_type, af::AllocType::kAllocTypeQueue);
  EXPECT_EQ(abs2_node->outputs[0].attr.que.id, que1_id);
  EXPECT_NE(abs2_node->outputs[0].attr.mem.reuse_id, af::kIdNone);
}

TEST_F(OptimizerSt, EliminateSizeVar) {
  const Expression s0 = af::Symbol("s0");
  const Expression s1 = af::Symbol("s1");
  const Expression s2 = af::Symbol("s2");
  const Expression s3 = af::Symbol("s3");
  const Expression s4 = af::Symbol("s4");
  const Expression s5 = af::Symbol("s5");
  const Expression s6 = af::Symbol("s6");
  const Expression s7 = af::Symbol("s7");

  auto graph = AscGraphBuilder("EliminateSizeVar")
                   .Loops({s7})
                   .Data("data0", 0)
                   .Load("load0", "data0", {s0}, {af::sym::kSymbolOne})
                   .Data("data1", 1)
                   .Load("load1", "data1", {s1}, {af::sym::kSymbolOne})
                   .Data("data2", 2)
                   .Load("load2", "data2", {s2}, {af::sym::kSymbolOne})
                   .Data("data3", 3)
                   .Load("load3", "data3", {s3}, {af::sym::kSymbolOne})
                   .Data("data4", 4)
                   .Load("load4", "data4", {s4}, {af::sym::kSymbolOne})
                   .Data("data5", 5)
                   .Load("load5", "data5", {s5}, {af::sym::kSymbolOne})
                   .Data("data6", 6)
                   .Load("load6", "data6", {s6}, {af::sym::kSymbolOne})
                   .Concat("concat", {"load0", "load1", "load2", "load3", "load4", "load5", "load6"})
                   .Store("store0", "concat")
                   .Output("out0", "store0", 0)
                   .Build();
  graph.CreateSizeVar("s0");
  graph.CreateSizeVar("s1");
  graph.CreateSizeVar("s2");
  graph.CreateSizeVar("s3");
  graph.CreateSizeVar("s4");
  graph.CreateSizeVar("s5");
  graph.CreateSizeVar("s6");
  graph.CreateSizeVar("s7");

  ::ascir::FusedScheduledResult fused_scheduled_result;
  Status res = optimizer.Optimize(graph, fused_scheduled_result);
  EXPECT_EQ(res, af::SUCCESS);
  ASSERT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
  const auto &schedule_groups = fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups;
  ASSERT_EQ(schedule_groups.size(), 7UL);
  std::set<size_t> size_var_counts;
  for (const auto &schedule_group : schedule_groups) {
    ASSERT_EQ(schedule_group.impl_graphs.size(), 1UL);
    size_var_counts.emplace(schedule_group.impl_graphs[0].GetAllSizeVar().size());
  }
  const std::set<size_t> expected_size_var_counts{4UL, 5UL, 6UL, 7UL, 8UL, 9UL, 10UL};
  EXPECT_EQ(size_var_counts, expected_size_var_counts);
}

TEST_F(OptimizerSt, SliceSliceConcatD) {
  AscGraph graph("slice_concat");
  auto s0 = graph.CreateSizeVar(128);
  auto s1 = graph.CreateSizeVar(90);
  auto s2 = graph.CreateSizeVar(1);
  auto s1_0 = graph.CreateSizeVar(60);
  auto s1_1 = graph.CreateSizeVar(30);
  auto s3 = graph.CreateSizeVar(97);
  auto s4 = graph.CreateSizeVar(65);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z1_1 = graph.CreateAxis("z1_1", s1_1);
  auto z1_0 = graph.CreateAxis("z1_0", s1_0);

  Data data0("data0", graph);
  data0.y.dtype = af::DT_FLOAT;
  data0.ir_attr.SetIndex(0);
  data0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data0.y.axis = {z0.id, z1.id, z2.id};
  *data0.y.repeats = {s0, s1_1, One};
  *data0.y.strides = {s1_1, One, One};

  Data data1("data1", graph);
  data1.y.dtype = af::DT_FLOAT;
  data1.ir_attr.SetIndex(1);
  data1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *data1.y.axis = {z0.id, z1.id, z2.id};
  *data1.y.repeats = {s0, s1_0, One};
  *data1.y.strides = {s1_0, One, One};

  Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1_0.id};
  load0.x = data1.y;
  *load0.y.axis = {z0.id, z1_0.id};
  load0.y.dtype = af::DT_FLOAT;
  *load0.y.repeats = {s0, s1_0};
  *load0.y.strides = {s3 * s1_0, s3};

  Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1_1.id};
  load1.x = data0.y;
  *load1.y.axis = {z0.id, z1_1.id};
  load1.y.dtype = af::DT_FLOAT;
  *load1.y.repeats = {s0, s1_1};
  *load1.y.strides = {s4 * s1_1, s4};

  af::ascir_op::Concat concat_op("concat");
  concat_op.attr.sched.axis = {z0.id, z1.id};
  concat_op.x = {load0.y, load1.y};
  concat_op.y.dtype = af::DT_FLOAT;
  *concat_op.y.axis = {z0.id, z1.id};
  *concat_op.y.repeats = {s0, s1};
  *concat_op.y.strides = {s1, af::ops::One};

  Store store_op("store");
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.x = concat_op.y;
  *store_op.y.axis = {z0.id, z1.id};
  store_op.y.dtype = af::DT_FLOAT;
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1, af::ops::One};

  Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = af::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
  setenv("AUTOFUSE_DFX_FLAGS", "codegen_compile_debug=true;debug_dir=./TestDump", 1);
  ::ascir::utils::ResetDumpConfig();
  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), af::SUCCESS);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1UL);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs.size(), 2UL);
  for (auto impl_graph : fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups[0].impl_graphs) {
    auto load0_remove_pad_0 = impl_graph.FindNode("load0_remove_pad_0");
    EXPECT_NE(load0_remove_pad_0, nullptr);
    auto load1_remove_pad_0 = impl_graph.FindNode("load1_remove_pad_0");
    EXPECT_NE(load1_remove_pad_0, nullptr);
  }
  unsetenv("AUTOFUSE_DFX_FLAGS");
  ::ascir::utils::ResetDumpConfig();
}

// ============================================================================
// Multi-Transpose ST Tests (covering the multi-transpose-to-load feature)
// ============================================================================

TEST_F(OptimizerSt, multi_transpose_two_branch_dynamic) {
  auto s0 = Sym("s0"), s1 = Sym("s1"), s2 = Sym("s2");
  auto graph = AscGraphBuilder("multi_transpose_two_branch")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0")
                   .Transpose("transpose0", "load0", {1, 0, 2})
                   .Data("data1", 1, ge::DT_FLOAT16)
                   .Load("load1", "data1")
                   .Transpose("transpose1", "load1", {2, 1, 0})
                   .Mul("mul0", "transpose0", "transpose1")
                   .Store("store0", "mul0")
                   .Output("out0", "store0", 0)
                   .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1);
}

TEST_F(OptimizerSt, multi_transpose_two_branch_static) {
  auto s0 = Sym(16), s1 = Sym(86), s2 = Sym(36);
  auto graph = AscGraphBuilder("multi_transpose_two_branch_static")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0, ge::DT_FLOAT16)
                   .Load("load0", "data0")
                   .Transpose("transpose0", "load0", {1, 0, 2})
                   .Data("data1", 1, ge::DT_FLOAT16)
                   .Load("load1", "data1")
                   .Transpose("transpose1", "load1", {2, 1, 0})
                   .Mul("mul0", "transpose0", "transpose1")
                   .Store("store0", "mul0")
                   .Output("out0", "store0", 0)
                   .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1);
}

// ==================== Broadcast Reorder Tests ====================

TEST_F(OptimizerSt, broadcast_reorder_elementwise) {
  auto s0 = Sym(4);
  auto s1 = Sym(512 * 1024);  // > 256*1024
  auto graph = AscGraphBuilder("broadcast_reorder_elementwise")
                   .Loops({s0, s1})
                   .Data("data0", 0, {af::ops::One, s1}, {af::ops::Zero, af::ops::One}, ge::DT_FLOAT16)
                   .Load("load0", "data0", {af::ops::One, s1}, {af::ops::Zero, af::ops::One})
                   .Abs("abs0", "load0")
                   .Store("store0", "abs0")
                   .Output("y", "store0", 0, ge::DT_FLOAT16)
                   .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), 0);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1);
  EXPECT_EQ(fused_scheduled_result.node_idx_to_scheduled_results[0][0].schedule_groups.size(), 1);
}

// Load → Transpose → Sum(Reduce) → Store，图上有 Reduce 节点，
// Transpose 应被消除推到 Load 上，不生成保留模板，Reduce 正常调度
TEST_F(OptimizerSt, transpose_reduce_eliminate) {
  const Expression s0 = af::Symbol(4);
  const Expression s1 = af::Symbol(8);
  const Expression s2 = af::Symbol(16);

  auto graph = AscGraphBuilder("transpose_reduce")
                   .Loops({s0, s1, s2})
                   .Data("data0", 0)
                   .Load("load0", "data0")
                   .Transpose("transpose0", "load0", {1, 0, 2})
                   .Sum("sum0", "transpose0", {2UL})
                   .Store("store0", "sum0")
                   .Output("out0", "store0", 0)
                   .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), af::SUCCESS);
  // Transpose 被 eliminate，不生成额外的保留模板，只有 Reduce 的调度结果
  EXPECT_GE(fused_scheduled_result.node_idx_to_scheduled_results[0].size(), 1UL);
}

TEST_F(OptimizerSt, SoftmaxPatternFusion_StableSoftmax) {
  auto s0 = Sym("s0");
  auto s1 = Sym("s1");

  auto graph = AscGraphBuilder("StableSoftmax")
                   .Loops({s0, s1})
                   .Data("data0", 0, {s0, s1}, {s1, af::sym::kSymbolOne})
                   .Load("load0", "data0")
                   .Max("max", "load0", {1})
                   .Broadcast("brc_max", "max", {s0, s1})
                   .Sub("sub", "load0", "brc_max")
                   .Exp("exp", "sub")
                   .Sum("sum", "exp", {1})
                   .Broadcast("brc_sum", "sum", {s0, s1})
                   .template Op<TrueDiv>("truediv", {"exp", "brc_sum"})
                   .Store("store", "truediv")
                   .Output("output", "store", 0, af::DT_FLOAT16)
                   .Build();

  optimize::SoftmaxPatternFusionPass softmax_pass;
  EXPECT_EQ(softmax_pass.RunPass(graph), af::SUCCESS);

  auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  ASSERT_NE(compute_graph, nullptr);

  // Softmax node 应替换原有模式节点
  auto softmax_node = compute_graph->FindNode("truediv_softmax");
  ASSERT_NE(softmax_node, nullptr);
  EXPECT_EQ(softmax_node->GetType(), "Softmax");

  // 原有模式节点应被移除
  EXPECT_EQ(compute_graph->FindNode("max"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc_max"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("sub"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("exp"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("sum"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("brc_sum"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("truediv"), nullptr);
}

TEST_F(OptimizerSt, DuplicateElewiseCse_MergeSub) {
  auto s0 = Sym("s0");
  auto s1 = Sym("s1");

  auto graph = AscGraphBuilder("DuplicateElewiseCse")
                   .Loops({s0, s1})
                   .Data("data0", 0, {s0, s1}, {s1, af::sym::kSymbolOne})
                   .Load("load0", "data0")
                   .Abs("abs0", "load0")
                   .Data("data1", 1, {s0, s1}, {s1, af::sym::kSymbolOne})
                   .Load("load1", "data1")
                   .Relu("relu0", "load1")
                   .Sub("sub1", "abs0", "relu0")
                   .Store("store1", "sub1")
                   .Output("output1", "store1", 0, af::DT_FLOAT16)
                   .Sub("sub2", "abs0", "relu0")
                   .Store("store2", "sub2")
                   .Output("output2", "store2", 1, af::DT_FLOAT16)
                   .Build();

  ::ascir::FusedScheduledResult fused_scheduled_result;
  EXPECT_EQ(optimizer.Optimize(graph, fused_scheduled_result), af::SUCCESS);
}

TEST_F(OptimizerSt, AlignmentTypeToString_AllEnumsMapped) {
  EXPECT_STREQ(AlignmentStrategyShadow::AlignmentTypeToString(optimize::AlignmentType::kNotAligned), "NotAligned");
  EXPECT_STREQ(AlignmentStrategyShadow::AlignmentTypeToString(optimize::AlignmentType::kAligned), "Aligned");
  EXPECT_STREQ(AlignmentStrategyShadow::AlignmentTypeToString(optimize::AlignmentType::kDiscontinuous),
               "Discontinuous");
  EXPECT_STREQ(AlignmentStrategyShadow::AlignmentTypeToString(optimize::AlignmentType::kFixedNotAligned),
               "FixedNotAligned");
  EXPECT_STREQ(AlignmentStrategyShadow::AlignmentTypeToString(optimize::AlignmentType::kInvalid), "Invalid");
  EXPECT_STREQ(AlignmentStrategyShadow::AlignmentTypeToString(static_cast<optimize::AlignmentType>(0xFFFFFFFF)),
               "Invalid");
}
