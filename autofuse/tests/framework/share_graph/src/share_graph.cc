/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "share_graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_op_types.h"
#include "ascendc_ir.h"
#include "ascendc_ir_def.h"
#include "ascendc_ir/utils/asc_graph_utils.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "fusion/autofuse_attrs.h"

using namespace af;

namespace {
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
    tensor_desc->SetFormat(ge::FORMAT_NCHW);
    tensor_desc->SetDataType(ge::DT_FLOAT);

    auto op_desc = std::make_shared<OpDesc>(name, (node_type_ == "") ? type : "AscGraph");
    for (std::int32_t i = 0; i < in_cnt; ++i) {
      op_desc->AddInputDesc(tensor_desc->Clone());
    }
    for (std::int32_t i = 0; i < out_cnt; ++i) {
      op_desc->AddOutputDesc(tensor_desc->Clone());
    }
    op_desc->AddInferFunc([](Operator &op) { return ge::GRAPH_SUCCESS; });
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

static void ConstructVVAscGraphAxisInfo(af::AscGraph &graph, size_t dims_size) {
  std::vector<int64_t> axis;
  std::vector<af::Expression> repeats;
  std::vector<af::Expression> strides;
  auto ONE = Symbol(1);

  // 构造符号、轴信息
  for (size_t i = 0; i < dims_size; i++) {
    std::string sym_str = "s" + std::to_string(i);
    std::string axis_str = "z" + std::to_string(i);
    const auto sym_s = Symbol(sym_str.c_str());
    auto axis_z = graph.CreateAxis(axis_str.c_str(), sym_s);
    axis.push_back(axis_z.id);
    repeats.push_back(sym_s);
    strides.push_back(ONE);
  }
  // 计算每个轴的stride
  for (int i = dims_size - 2; i >= 0; i--) {
    strides[i] = af::sym::Mul(repeats[i + 1], strides[i + 1]);
  }
  // 将原始轴信息设置到图中所有节点上
  for (auto node : graph.GetAllNodes()) {
    if (af::ops::IsOps<af::ascir_op::Scalar>(node)) {
      continue;
    }
    node->attr.sched.axis = axis;
    for (auto output_attr : node->outputs()) {
      output_attr->attr.axis = axis;
      output_attr->attr.repeats = repeats;
      output_attr->attr.strides = strides;
    }
  }
}

static void ConstructVVConstAscGraphAxisInfo(af::AscGraph &graph, size_t dims_size, vector<int> dims) {
  std::vector<int64_t> axis;
  std::vector<af::Expression> repeats;
  std::vector<af::Expression> strides;
  auto ONE = Symbol(1);

  // 构造符号、轴信息
  for (size_t i = 0; i < dims_size; i++) {
    std::string sym_str = "s" + std::to_string(i);
    std::string axis_str = "z" + std::to_string(i);
    const auto sym_s = Symbol(dims[i], sym_str.c_str());
    auto axis_z = graph.CreateAxis(axis_str.c_str(), sym_s);
    axis.push_back(axis_z.id);
    repeats.push_back(sym_s);
    strides.push_back(ONE);
  }
  // 计算每个轴的stride
  for (int i = dims_size - 2; i >= 0; i--) {
    strides[i] = af::sym::Mul(repeats[i + 1], strides[i + 1]);
  }
  // 将原始轴信息设置到图中所有节点上
  for (auto node : graph.GetAllNodes()) {
    if (af::ops::IsOps<af::ascir_op::Scalar>(node)) {
      continue;
    }
    node->attr.sched.axis = axis;
    for (auto output_attr : node->outputs()) {
      output_attr->attr.axis = axis;
      output_attr->attr.repeats = repeats;
      output_attr->attr.strides = strides;
    }
  }
}

static void ConstructVVAscGraphAxisInfo(af::AscGraph &graph, size_t dims_size, size_t last_axis_strides) {
  std::vector<int64_t> axis;
  std::vector<af::Expression> repeats;
  std::vector<af::Expression> strides;
  auto ONE = Symbol(last_axis_strides);

  // 构造符号、轴信息
  for (size_t i = 0; i < dims_size; i++) {
    std::string sym_str = "s" + std::to_string(i);
    std::string axis_str = "z" + std::to_string(i);
    const auto sym_s = Symbol(sym_str.c_str());
    auto axis_z = graph.CreateAxis(axis_str.c_str(), sym_s);
    axis.push_back(axis_z.id);
    repeats.push_back(sym_s);
    strides.push_back(ONE);
  }
  // 计算每个轴的stride
  for (int i = dims_size - 2; i >= 0; i--) {
    strides[i] = af::sym::Mul(repeats[i + 1], strides[i + 1]);
  }
  // 将原始轴信息设置到图中所有节点上
  for (auto node : graph.GetAllNodes()) {
    if (af::ops::IsOps<af::ascir_op::Scalar>(node)) {
      continue;
    }
    node->attr.sched.axis = axis;
    for (auto output_attr : node->outputs()) {
      output_attr->attr.axis = axis;
      output_attr->attr.repeats = repeats;
      output_attr->attr.strides = strides;
    }
  }
}

static void ConstructVVAscGraphAxisInfo(af::AscGraph &graph, size_t dims_size, vector<size_t> perms) {
  // 原始轴信息
  std::vector<int64_t> axis;
  std::vector<af::Expression> repeats;
  std::vector<af::Expression> strides;
  // transpose后的轴信息
  std::vector<int64_t> t_axis;
  std::vector<af::Expression> t_repeats;
  std::vector<af::Expression> t_strides;
  auto ONE = Symbol(1);

  // 构造符号、轴信息
  for (size_t i = 0; i < dims_size; i++) {
    std::string sym_str = "s" + std::to_string(i);
    std::string axis_str = "z" + std::to_string(i);
    const auto sym_s = Symbol(sym_str.c_str());
    auto axis_z = graph.CreateAxis(axis_str.c_str(), sym_s);
    axis.push_back(axis_z.id);
    repeats.push_back(sym_s);
    strides.push_back(ONE);
  }
  for (size_t i = 0; i < dims_size; i++) {
    t_axis.push_back(axis[perms[i]]);
    t_repeats.push_back(repeats[perms[i]]);
    t_strides.push_back(ONE);
  }
  // 计算每个轴的stride
  for (int i = dims_size - 2; i >= 0; i--) {
    strides[i] = af::sym::Mul(repeats[i + 1], strides[i + 1]);
    t_strides[i] = af::sym::Mul(t_repeats[i + 1], t_strides[i + 1]);
  }
  // 将原始轴信息设置到图中所有节点上
  for (auto node : graph.GetAllNodes()) {
    if (af::ops::IsOps<af::ascir_op::Scalar>(node)) {
      continue;
    }
    // transpose及之后的节点需要设置成转置属性
    if (af::ops::IsOps<af::ascir_op::Transpose>(node)) {
      axis = t_axis;
      repeats = t_repeats;
      strides = t_strides;
    }
    node->attr.sched.axis = t_axis;
    for (auto output_attr : node->outputs()) {
      output_attr->attr.axis = axis;
      output_attr->attr.repeats = repeats;
      output_attr->attr.strides = strides;
    }
  }
}

static void ConstructVVAscGraphAxisInfo(af::AscGraph &graph, const std::vector<std::string> &dim_sizes, bool align) {
  std::vector<int64_t> axis;
  std::vector<af::Expression> repeats;
  std::vector<af::Expression> strides;
  auto ONE = af::Symbol(1);

  // 构造符号、轴信息
  for (size_t i = 0; i < dim_sizes.size(); i++) {
    af::Symbol sym_s;
    if (dim_sizes[i][0] == 's') {
      sym_s = af::Symbol(dim_sizes[i].c_str());
      graph.CreateSizeVar(dim_sizes[i]);
    } else {
      sym_s = af::Symbol(std::atoi(dim_sizes[i].c_str()));
    }
    std::string sym_str = "s" + std::to_string(i);
    std::string axis_str = "z" + std::to_string(i);
    auto axis_z = graph.CreateAxis(axis_str.c_str(), sym_s);
    axis.push_back(axis_z.id);
    repeats.push_back(sym_s);
    strides.push_back(ONE);
  }
  // 计算每个轴的stride
  // s0, s1
  if (align) {
    strides[dim_sizes.size() - 2] = af::sym::Align(repeats[dim_sizes.size() - 1], 32 / sizeof(int32_t));
  } else {
    strides[dim_sizes.size() - 2] = repeats[dim_sizes.size() - 1];
  }
  for (int32_t i = static_cast<int32_t>(dim_sizes.size()) - 3; i >= 0; i--) {
    strides[i] = af::sym::Mul(repeats[i + 1], strides[i + 1]);
  }
  // 将原始轴信息设置到图中所有节点上
  for (auto node : graph.GetAllNodes()) {
    node->attr.sched.axis = axis;
    for (auto output_attr : node->outputs()) {
      output_attr->attr.axis = axis;
      output_attr->attr.repeats = repeats;
      output_attr->attr.strides = strides;
    }
  }
}

static void ConstructVVConstAscGraphAxisInfo(af::AscGraph &graph, size_t dims_size, vector<size_t> perms,
                                             vector<int> dims) {
  // 原始轴信息
  std::vector<int64_t> axis;
  std::vector<af::Expression> repeats;
  std::vector<af::Expression> strides;
  // transpose后的轴信息
  std::vector<int64_t> t_axis;
  std::vector<af::Expression> t_repeats;
  std::vector<af::Expression> t_strides;
  auto ONE = Symbol(1);

  // 构造符号、轴信息
  for (size_t i = 0; i < dims_size; i++) {
    std::string sym_str = "s" + std::to_string(i);
    std::string axis_str = "z" + std::to_string(i);
    const auto sym_s = Symbol(dims[i], sym_str.c_str());
    auto axis_z = graph.CreateAxis(axis_str.c_str(), sym_s);
    axis.push_back(axis_z.id);
    repeats.push_back(sym_s);
    strides.push_back(ONE);
  }
  for (size_t i = 0; i < dims_size; i++) {
    t_axis.push_back(axis[perms[i]]);
    t_repeats.push_back(repeats[perms[i]]);
    t_strides.push_back(ONE);
  }
  // 计算每个轴的stride
  for (int i = dims_size - 2; i >= 0; i--) {
    strides[i] = af::sym::Mul(repeats[i + 1], strides[i + 1]);
    t_strides[i] = af::sym::Mul(t_repeats[i + 1], t_strides[i + 1]);
  }
  // 将原始轴信息设置到图中所有节点上
  for (auto node : graph.GetAllNodes()) {
    if (af::ops::IsOps<af::ascir_op::Scalar>(node)) {
      continue;
    }
    // transpose及之后的节点需要设置成转置属性
    if (af::ops::IsOps<af::ascir_op::Transpose>(node)) {
      axis = t_axis;
      repeats = t_repeats;
      strides = t_strides;
    }
    node->attr.sched.axis = t_axis;
    for (auto output_attr : node->outputs()) {
      output_attr->attr.axis = axis;
      output_attr->attr.repeats = repeats;
      output_attr->attr.strides = strides;
    }
  }
}
}  // namespace

namespace ascir {
/**
 *      output
 *         |
 *       store
 *         |
 *        abs
 *         |
 *        add
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateAddAbsAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  af::ascir_op::Add add("add");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;

  af::ascir_op::Abs abs("abs");
  abs.x = add.y;

  af::ascir_op::Store x_out("store");
  x_out.x = abs.y;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      output(bool)
 *         |
 *       store(bool)
 *         |
 *    transpose(bool)  [2D, perms={1,0}]
 *         |
 *      load(bool)
 *         |
 *      data(bool)
 */
static void CreateTransposeBoolAscGraph(af::AscGraph &graph) {
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);
  x.y.dtype = af::DT_BOOL;

  af::ascir_op::Load load("load0");
  load.x = x.y;
  load.y.dtype = af::DT_BOOL;

  af::ascir_op::Transpose transpose("transpose");
  transpose.x = load.y;
  transpose.y.dtype = af::DT_BOOL;
  transpose.attr.tmp_buffers = {{{af::Symbol(8192), -1}, af::MemAttr(), 0}};

  af::ascir_op::Store store("store");
  store.x = transpose.y;
  store.y.dtype = af::DT_BOOL;

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.y.dtype = af::DT_BOOL;
  output.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, 3, {1, 0, 2});
}

af::ComputeGraphPtr ShareGraph::TransposeBoolFusedGraph() {
  auto builder = GraphBuilder("transpose_bool_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("transpose_bool");
  CreateTransposeBoolAscGraph(sub_graph);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}
static void CreateBF16AddAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = af::DT_BF16;
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = af::DT_BF16;

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = af::DT_BF16;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = af::DT_BF16;

  af::ascir_op::Add add("add");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;
  add.y.dtype = af::DT_BF16;

  af::ascir_op::Store x_out("store");
  x_out.x = add.y;
  x_out.y.dtype = af::DT_BF16;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      output
 *         |
 *       store
 *         |
 *        add(bf16)
 *       /   \
 *   nddma0  nddma1
 *     |       |
 *   data0   data1
 */
static void CreateBF16NddmaAddAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = af::DT_BF16;
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = af::DT_BF16;

  af::ascir_op::Nddma x1Local("nddma0");
  x1Local.x = x1.y;
  x1Local.y.dtype = af::DT_BF16;

  af::ascir_op::Nddma x2Local("nddma1");
  x2Local.x = x2.y;
  x2Local.y.dtype = af::DT_BF16;

  af::ascir_op::Add add("add");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;
  add.y.dtype = af::DT_BF16;

  af::ascir_op::Store x_out("store");
  x_out.x = add.y;
  x_out.y.dtype = af::DT_BF16;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

static void CreateAddAbsConstAscGraph(af::AscGraph &graph, size_t dims_size, std::vector<int> dims) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  af::ascir_op::Add add("add");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;

  af::ascir_op::Abs abs("abs");
  abs.x = add.y;

  af::ascir_op::Store x_out("store");
  x_out.x = abs.y;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVConstAscGraphAxisInfo(graph, dims_size, dims);
}

/**
 *      output
 *         |
 *       store
 *         |
 *     logical_or
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateLoadLogicalOrStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  x1.y.dtype = af::DT_UINT8;
  x2.y.dtype = af::DT_UINT8;
  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  x1Local.y.dtype = af::DT_UINT8;
  x2Local.y.dtype = af::DT_UINT8;
  af::ascir_op::LogicalOr logical_or("logical_or");
  logical_or.x1 = x1Local.y;
  logical_or.x2 = x2Local.y;

  logical_or.y.dtype = af::DT_UINT8;
  af::ascir_op::Store x_out("store");
  x_out.x = logical_or.y;

  x_out.y.dtype = af::DT_UINT8;
  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      output
 *         |
 *       store
 *         |
 *    logical_and
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateLoadLogicalAndStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  x1.y.dtype = af::DT_UINT8;
  x2.y.dtype = af::DT_UINT8;
  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  x1Local.y.dtype = af::DT_UINT8;
  x2Local.y.dtype = af::DT_UINT8;
  af::ascir_op::LogicalAnd logical_and("logical_and");
  logical_and.x1 = x1Local.y;
  logical_and.x2 = x2Local.y;

  logical_and.y.dtype = af::DT_UINT8;
  af::ascir_op::Store x_out("store");
  x_out.x = logical_and.y;

  x_out.y.dtype = af::DT_UINT8;
  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      output
 *         |
 *       store
 *         |
 *        abs
 *         |
 *      floordiv
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateFloordivAbsAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = DT_INT8;
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = DT_INT8;

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = DT_INT8;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = DT_INT8;

  af::ascir_op::FloorDiv floordiv("floordiv");
  floordiv.x1 = x1Local.y;
  floordiv.x2 = x2Local.y;
  floordiv.y.dtype = DT_INT8;

  af::ascir_op::Abs abs("abs");
  abs.x = floordiv.y;
  abs.y.dtype = DT_INT8;

  af::ascir_op::Store x_out("store");
  x_out.x = abs.y;
  x_out.y.dtype = DT_INT8;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      output
 *         |
 *       store
 *         |
 *        exp2
 *         |
 *        add
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateAddExp2AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  af::ascir_op::Add add("add");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;

  af::ascir_op::Exp2 exp2("exp2");
  exp2.x = add.y;

  af::ascir_op::Store x_out("store");
  x_out.x = exp2.y;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      output
 *         |
 *       store
 *         |
 *       floor
 *         |
 *        add
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateAddFloorAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  af::ascir_op::Add add("add");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;

  af::ascir_op::Floor floor("floor");
  floor.x = add.y;

  af::ascir_op::Store x_out("store");
  x_out.x = floor.y;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**     (BF16)
 *      output
 *         |
 *       store
 *         |
 *       floor
 *         |
 *        add
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateAddFloorBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = af::DT_BF16;
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = af::DT_BF16;

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = af::DT_BF16;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = af::DT_BF16;

  af::ascir_op::Add add("add");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;
  add.y.dtype = af::DT_BF16;

  af::ascir_op::Floor floor("floor");
  floor.x = add.y;
  floor.y.dtype = af::DT_BF16;

  af::ascir_op::Store x_out("store");
  x_out.x = floor.y;
  x_out.y.dtype = af::DT_BF16;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**     (BF16)
 *      output
 *         |
 *       store
 *         |
 *        exp
 *         |
 *        add
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateAddExpBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = af::DT_BF16;
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = af::DT_BF16;

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = af::DT_BF16;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = af::DT_BF16;

  af::ascir_op::Add add("add");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;
  add.y.dtype = af::DT_BF16;

  af::ascir_op::Exp exp("exp");
  exp.x = add.y;
  exp.y.dtype = af::DT_BF16;

  af::ascir_op::Store x_out("store");
  x_out.x = exp.y;
  x_out.y.dtype = af::DT_BF16;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::AddExp2FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("add_exp2_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("add_exp2");
  CreateAddExp2AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::AddFloorFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("add_floor_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("add_floor");
  CreateAddFloorAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::AddFloorBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("add_floor_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("add_floor_bf16");
  CreateAddFloorBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::AddExpBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("add_exp_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("add_exp_bf16");
  CreateAddExpBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *        abs
 *         |
 *        axpy
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateAxpyAbsAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  af::ascir_op::Axpy axpy("axpy");
  axpy.x1 = x1Local.y;
  axpy.x2 = x2Local.y;
  axpy.ir_attr.SetAlpha(0.8);

  af::ascir_op::Abs abs("abs");
  abs.x = axpy.y;

  af::ascir_op::Store x_out("store");
  x_out.x = abs.y;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      output
 *         |
 *       store
 *         |
 *        abs
 *         |
 *        axpy
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateAxpyAbsHalfAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x1.y.dtype = af::DT_FLOAT16;
  x2.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  x1Local.y.dtype = af::DT_FLOAT16;
  x2Local.y.dtype = af::DT_FLOAT16;
  af::ascir_op::Axpy axpy("axpy");
  axpy.x1 = x1Local.y;
  axpy.x2 = x2Local.y;
  axpy.ir_attr.SetAlpha(0.8);
  axpy.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Abs abs("abs");
  abs.x = axpy.y;

  abs.y.dtype = af::DT_FLOAT16;
  af::ascir_op::Store x_out("store");
  x_out.x = abs.y;

  x_out.y.dtype = af::DT_FLOAT16;
  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *          output
 *             |
 *           store
 *             |
 *            add
 *           / |
 *        axpy |
 *       /   \ |
 *   load0  load1
 *     |      |
 *   data0  data1
 */
static void CreateAxpyAddAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  af::ascir_op::Axpy axpy("axpy");
  axpy.x1 = x1Local.y;
  axpy.x2 = x2Local.y;
  axpy.ir_attr.SetAlpha(0.8);

  af::ascir_op::Add add("add");
  add.x1 = axpy.y;
  add.x2 = x2Local.y;

  af::ascir_op::Store x_out("store");
  x_out.x = add.y;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

static void CreateAddAbsScalarAscGraph(af::AscGraph &graph, size_t dims_size, af::DataType dtype) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = dtype;

  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = dtype;

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = dtype;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = dtype;

  af::ascir_op::Add add("add");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;
  add.y.dtype = dtype;

  af::ascir_op::Abs abs("abs");
  abs.x = add.y;
  abs.y.dtype = dtype;

  af::ascir_op::Store x_out("store");
  x_out.x = abs.y;
  x_out.y.dtype = dtype;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = dtype;

  ConstructVVAscGraphAxisInfo(graph, dims_size, 0U);
}

/**
 *         data0
 *           |
 *         load0
 *         |    \
 *       store  abs
 *         |     |
 *        ouput0 output1
 */
static void CreateLoadToStoreAndAbsAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  af::ascir_op::Abs abs("abs");
  abs.x = x1Local.y;

  af::ascir_op::Store abs_store("store");
  abs_store.x = abs.y;

  af::ascir_op::Store load_2_store("store2");
  load_2_store.x = x1Local.y;

  af::ascir_op::Output y1("output");
  y1.x = abs_store.y;
  y1.ir_attr.SetIndex(0);

  af::ascir_op::Output y2("output2");
  y2.x = load_2_store.y;
  y2.ir_attr.SetIndex(1);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *         data0
 *           |
 *         load0
 *         |    \
 *       store  abs
 *         |     |
 *        ouput0 output1
 */
static void CreateLoadUnalignPadAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  af::ascir_op::Abs abs("abs");
  abs.x = x1Local.y;

  af::ascir_op::Store abs_store("store");
  abs_store.x = abs.y;

  af::ascir_op::Store load_2_store("store2");
  load_2_store.x = x1Local.y;

  af::ascir_op::Output y1("output");
  y1.x = abs_store.y;
  y1.ir_attr.SetIndex(0);

  af::ascir_op::Output y2("output2");
  y2.x = load_2_store.y;
  y2.ir_attr.SetIndex(1);

  ConstructVVAscGraphAxisInfo(graph, dims_size, 2UL);
}

/**
 *      output
 *         |
 *       store
 *         |
 *        add
 *        | \
 *       |  abs
 *      |    \
 *   load0   brc
 *     |      \
 *  data0   scalar0
 */
static void CreateAbsBrcAddAscGraph(af::AscGraph &graph, size_t dims_size) {
  const Expression s0 = graph.CreateSizeVar(2);
  const Expression s1 = graph.CreateSizeVar(8);
  auto One = Symbol(1);
  auto Zero = Symbol(0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data x0("data", graph);
  x0.attr.api.compute_type = ComputeType::kComputeInvalid;
  x0.attr.api.type = af::ApiType::kAPITypeBuffer;
  x0.ir_attr.SetIndex(0);
  x0.y.dtype = af::DataType::DT_FLOAT;

  af::ascir_op::Load load0("load");
  load0.x = x0.y;
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.attr.api.type = af::ApiType::kAPITypeCompute;
  load0.attr.sched.axis = {z0.id, z1.id};
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};
  load0.y.dtype = af::DataType::DT_FLOAT;
  load0.attr.api.unit = ComputeUnit::kUnitMTE2;

  af::ascir_op::Scalar x1("scalar", graph);
  x1.ir_attr.SetIndex(1);
  x1.ir_attr.SetValue("1.0");

  af::ascir_op::Broadcast brc0("brc");
  brc0.x = x1.y;
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.attr.api.type = af::ApiType::kAPITypeCompute;
  brc0.attr.sched.axis = {z0.id, z1.id};
  *brc0.y.axis = {z0.id, z1.id};
  *brc0.y.repeats = {s0, s1};
  *brc0.y.strides = {s1, One};
  brc0.y.dtype = af::DataType::DT_FLOAT;
  brc0.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Abs abs1("abs");
  abs1.x = brc0.y;
  abs1.attr.api.compute_type = ComputeType::kComputeLoad;
  abs1.attr.api.type = af::ApiType::kAPITypeCompute;
  abs1.attr.sched.axis = {z0.id, z1.id};
  *abs1.y.axis = {z0.id, z1.id};
  *abs1.y.repeats = {s0, s1};
  *abs1.y.strides = {s1, One};
  abs1.y.dtype = af::DataType::DT_FLOAT;
  abs1.attr.api.unit = ComputeUnit::kUnitMTE2;

  af::ascir_op::Add add("add");
  add.x1 = load0.y;
  add.x2 = abs1.y;
  add.attr.api.compute_type = ComputeType::kComputeLoad;
  add.attr.api.type = af::ApiType::kAPITypeCompute;
  add.attr.sched.axis = {z0.id, z1.id};
  *add.y.axis = {z0.id, z1.id};
  *add.y.repeats = {s0, s1};
  *add.y.strides = {s1, One};
  add.y.dtype = af::DataType::DT_FLOAT;
  add.attr.api.unit = ComputeUnit::kUnitMTE2;

  af::ascir_op::Store store("store");
  store.x = add.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.attr.api.type = af::ApiType::kAPITypeCompute;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};
  store.y.dtype = af::DataType::DT_FLOAT;
  store.attr.api.unit = ComputeUnit::kUnitMTE3;

  af::ascir_op::Output y("output");
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = af::DataType::DT_FLOAT;
  y.ir_attr.SetIndex(0);
}

/**
 *      output
 *         |
 *       store
 *         |
 *        add
 *        | \
 *       |  abs
 *      |    \
 *   load0   brc
 *     |      \
 *  data0   scalar0
 */
static void CreateUbScalerBrcAbsAddAscGraph(af::AscGraph &graph, size_t dims_size) {
  const Expression s0 = graph.CreateSizeVar(2);
  const Expression s1 = graph.CreateSizeVar(8);
  auto One = Symbol(1);
  auto Zero = Symbol(0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data x0("data", graph);
  x0.attr.api.compute_type = ComputeType::kComputeInvalid;
  x0.attr.api.type = af::ApiType::kAPITypeBuffer;
  x0.ir_attr.SetIndex(0);
  x0.y.dtype = af::DataType::DT_FLOAT;

  af::ascir_op::Load load0("load");
  load0.x = x0.y;
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.attr.api.type = af::ApiType::kAPITypeCompute;
  load0.attr.sched.axis = {z0.id, z1.id};
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, One};
  load0.y.dtype = af::DataType::DT_FLOAT;
  load0.attr.api.unit = ComputeUnit::kUnitMTE2;

  af::ascir_op::Data x1("scalar", graph);
  x1.ir_attr.SetIndex(1);

  af::ascir_op::Load load1("load1");
  load1.x = x1.y;
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.attr.api.type = af::ApiType::kAPITypeCompute;
  load1.attr.sched.axis = {z0.id, z1.id};
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {One, One};
  *load1.y.strides = {Zero, Zero};
  load1.y.dtype = af::DataType::DT_FLOAT;
  load1.attr.api.unit = ComputeUnit::kUnitMTE2;

  af::ascir_op::Broadcast brc0("brc");
  brc0.x = load1.y;
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.attr.api.type = af::ApiType::kAPITypeCompute;
  brc0.attr.sched.axis = {z0.id, z1.id};
  *brc0.y.axis = {z0.id, z1.id};
  *brc0.y.repeats = {s0, s1};
  *brc0.y.strides = {s1, One};
  brc0.y.dtype = af::DataType::DT_FLOAT;
  brc0.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Abs abs1("abs");
  abs1.x = brc0.y;
  abs1.attr.api.compute_type = ComputeType::kComputeLoad;
  abs1.attr.api.type = af::ApiType::kAPITypeCompute;
  abs1.attr.sched.axis = {z0.id, z1.id};
  *abs1.y.axis = {z0.id, z1.id};
  *abs1.y.repeats = {s0, s1};
  *abs1.y.strides = {s1, One};
  abs1.y.dtype = af::DataType::DT_FLOAT;
  abs1.attr.api.unit = ComputeUnit::kUnitMTE2;

  af::ascir_op::Add add("add");
  add.x1 = load0.y;
  add.x2 = abs1.y;
  add.attr.api.compute_type = ComputeType::kComputeLoad;
  add.attr.api.type = af::ApiType::kAPITypeCompute;
  add.attr.sched.axis = {z0.id, z1.id};
  *add.y.axis = {z0.id, z1.id};
  *add.y.repeats = {s0, s1};
  *add.y.strides = {s1, One};
  add.y.dtype = af::DataType::DT_FLOAT;
  add.attr.api.unit = ComputeUnit::kUnitMTE2;

  af::ascir_op::Store store("store");
  store.x = add.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.attr.api.type = af::ApiType::kAPITypeCompute;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};
  store.y.dtype = af::DataType::DT_FLOAT;
  store.attr.api.unit = ComputeUnit::kUnitMTE3;

  af::ascir_op::Output y("output");
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = af::DataType::DT_FLOAT;
  y.ir_attr.SetIndex(0);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::AddAbsFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("add_abs_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("add_abs");
  CreateAddAbsAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::FloordivAbsFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("floordiv_abs_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("floordiv_abs");
  CreateFloordivAbsAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::BF16AddFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("bf16_add_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("bf16_add");
  CreateBF16AddAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::BF16NddmaAddFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("bf16_nddma_add_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("bf16_nddma_add");
  CreateBF16NddmaAddAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateAbsUint8AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data", graph);
  x.y.dtype = af::DataType::DT_UINT8;
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load xLocal("load");
  xLocal.x = x.y;
  xLocal.y.dtype = af::DataType::DT_UINT8;

  af::ascir_op::Abs abs("abs");
  abs.x = xLocal.y;
  abs.y.dtype = af::DataType::DT_UINT8;

  af::ascir_op::Store x_out("store");
  x_out.x = abs.y;
  x_out.y.dtype = af::DataType::DT_UINT8;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::AbsUint8FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("abs_uint8_test");
  auto data = builder.AddNode("data", "Data", 0, 1);
  af::AttrUtils::SetInt(data->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("abs_uint8_test");
  CreateAbsUint8AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateAbsBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data", graph);
  x.y.dtype = af::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load xLocal("load");
  xLocal.x = x.y;
  xLocal.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Abs abs("abs");
  abs.x = xLocal.y;
  abs.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Store x_out("store");
  x_out.x = abs.y;
  x_out.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::AbsBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("abs_bf16_test");
  auto data = builder.AddNode("data", "Data", 0, 1);
  af::AttrUtils::SetInt(data->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("abs_bf16_test");
  CreateAbsBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateCeilBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data", graph);
  x.y.dtype = af::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load xLocal("load");
  xLocal.x = x.y;
  xLocal.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Ceil ceil("ceil");
  ceil.x = xLocal.y;
  ceil.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Store x_out("store");
  x_out.x = ceil.y;
  x_out.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::CeilBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("ceil_bf16_test");
  auto data = builder.AddNode("data", "Data", 0, 1);
  af::AttrUtils::SetInt(data->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("ceil_bf16_test");
  CreateCeilBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateTruncBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data", graph);
  x.y.dtype = af::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load xLocal("load");
  xLocal.x = x.y;
  xLocal.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Trunc trunc("trunc");
  trunc.x = xLocal.y;
  trunc.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Store x_out("store");
  x_out.x = trunc.y;
  x_out.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::TruncBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("trunc_bf16_test");
  auto data = builder.AddNode("data", "Data", 0, 1);
  af::AttrUtils::SetInt(data->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("trunc_bf16_test");
  CreateTruncBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateCosBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data", graph);
  x.y.dtype = af::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load xLocal("load");
  xLocal.x = x.y;
  xLocal.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Cos cos("cos");
  cos.x = xLocal.y;
  cos.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Store x_out("store");
  x_out.x = cos.y;
  x_out.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::CosBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("cos_bf16_test");
  auto data = builder.AddNode("data", "Data", 0, 1);
  af::AttrUtils::SetInt(data->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("cos_bf16_test");
  CreateCosBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateErfBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data", graph);
  x.y.dtype = af::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load xLocal("load");
  xLocal.x = x.y;
  xLocal.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Erf erf("erf");
  erf.x = xLocal.y;
  erf.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Store x_out("store");
  x_out.x = erf.y;
  x_out.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::ErfBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("erf_bf16_test");
  auto data = builder.AddNode("data", "Data", 0, 1);
  af::AttrUtils::SetInt(data->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("erf_bf16_test");
  CreateErfBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateLog1pBfloat16AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data0", graph);
  x.y.dtype = af::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load x1Local("load0");
  x1Local.y.dtype = af::DataType::DT_BF16;
  x1Local.x = x.y;

  af::ascir_op::Log1p log1p("log1p");
  log1p.x = x1Local.y;
  log1p.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Store log1p_store("store");
  log1p_store.x = log1p.y;
  log1p_store.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Output y("output");
  y.x = log1p_store.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::Log1pBfloat16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("log1p_bfloat16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("log1p_bfloat16_test");
  CreateLog1pBfloat16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateFloorToIntFloatAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load xLocal("load");
  xLocal.x = x.y;

  af::ascir_op::FloorToInt floortoint("floortoint");
  floortoint.x = xLocal.y;
  floortoint.y.dtype = af::DT_INT32;

  af::ascir_op::Store x_out("store");
  x_out.x = floortoint.y;
  x_out.y.dtype = af::DT_INT32;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.y.dtype = af::DT_INT32;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::FloorToIntFloatFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("floortoint_float_test");
  auto data = builder.AddNode("data", "Data", 0, 1);
  af::AttrUtils::SetInt(data->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("floortoint_float");
  CreateFloorToIntFloatAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateFmodFloatAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = af::DT_FLOAT;
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = af::DT_FLOAT;

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = af::DT_FLOAT;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = af::DT_FLOAT;

  af::ascir_op::Fmod fmod("fmod");
  fmod.x1 = x1Local.y;
  fmod.x2 = x2Local.y;
  fmod.y.dtype = af::DT_FLOAT;

  af::ascir_op::Store x_out("store");
  x_out.x = fmod.y;
  x_out.y.dtype = af::DT_FLOAT;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::FmodFloatFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("fmod_float_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("fmod_float");
  CreateFmodFloatAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *     logicalxor
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateLogicalXorFloatAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = af::DT_FLOAT;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = af::DT_FLOAT;

  af::ascir_op::LogicalXor logicalxor("logicalxor");
  logicalxor.x1 = x1Local.y;
  logicalxor.x2 = x2Local.y;
  logicalxor.y.dtype = af::DT_UINT8;

  af::ascir_op::Store x_out("store");
  x_out.x = logicalxor.y;
  x_out.y.dtype = af::DT_UINT8;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.y.dtype = af::DT_UINT8;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::LogicalXorFloatFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("logicalxor_float_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("logicalxor_float");
  CreateLogicalXorFloatAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *       hypot
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateHypotFloatAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  af::ascir_op::Hypot hypot("hypot");
  hypot.x1 = x1Local.y;
  hypot.x2 = x2Local.y;

  af::ascir_op::Store x_out("store");
  x_out.x = hypot.y;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::HypotFloatFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("hypot_float_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("hypot_float");
  CreateHypotFloatAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *       lgamma
 *         |
 *       load0
 *         |
 *       data0
 */
static void CreateLgammaFloatAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);
  x.y.dtype = af::DT_FLOAT;

  af::ascir_op::Load xLocal("load0");
  xLocal.x = x.y;
  xLocal.y.dtype = af::DT_FLOAT;

  af::ascir_op::Lgamma lgamma("lgamma");
  lgamma.x = xLocal.y;
  lgamma.y.dtype = af::DT_FLOAT;

  af::ascir_op::Store x_out("store");
  x_out.x = lgamma.y;
  x_out.y.dtype = af::DT_FLOAT;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = af::DT_FLOAT;

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::LgammaFloatFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("lgamma_float_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("lgamma_float");
  CreateLgammaFloatAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateLoadLog10StoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  af::ascir_op::Log10 log10("log10");
  log10.x = x1Local.y;

  af::ascir_op::Store log10_store("store");
  log10_store.x = log10.y;

  af::ascir_op::Output y("output");
  y.x = log10_store.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::LoadLog10StoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("log10_float_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("log10_float_test");
  CreateLoadLog10StoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateAbsClipAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  af::ascir_op::Data x3("data2", graph);
  x3.ir_attr.SetIndex(2);

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  af::ascir_op::Load x3Local("load2");
  x3Local.x = x3.y;

  af::ascir_op::Abs abs("abs");
  abs.x = x1Local.y;

  af::ascir_op::ClipByValue clipbyvalue("clipbyvalue");
  clipbyvalue.x1 = abs.y;
  clipbyvalue.x2 = x2Local.y;
  clipbyvalue.x3 = x3Local.y;

  af::ascir_op::Store x_out("store");
  x_out.x = clipbyvalue.y;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

static void CreateAbsFmaBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = af::DT_BF16;
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = af::DT_BF16;
  af::ascir_op::Data x3("data2", graph);
  x3.ir_attr.SetIndex(2);
  x3.y.dtype = af::DT_BF16;

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = af::DT_BF16;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = af::DT_BF16;

  af::ascir_op::Load x3Local("load2");
  x3Local.x = x3.y;
  x3Local.y.dtype = af::DT_BF16;

  af::ascir_op::Abs abs("abs");
  abs.x = x1Local.y;
  abs.y.dtype = af::DT_BF16;

  af::ascir_op::Fma fma("fma");
  fma.x1 = abs.y;
  fma.x2 = x2Local.y;
  fma.x3 = x3Local.y;
  fma.y.dtype = af::DT_BF16;

  af::ascir_op::Store x_out("store");
  x_out.x = fma.y;
  x_out.y.dtype = af::DT_BF16;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

static void CreateAbsFmaAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  af::ascir_op::Data x3("data2", graph);
  x3.ir_attr.SetIndex(2);

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  af::ascir_op::Load x3Local("load2");
  x3Local.x = x3.y;

  af::ascir_op::Abs abs("abs");
  abs.x = x1Local.y;

  af::ascir_op::Fma fma("fma");
  fma.x1 = abs.y;
  fma.x2 = x2Local.y;
  fma.x3 = x3Local.y;

  af::ascir_op::Store x_out("store");
  x_out.x = fma.y;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::AbsClipFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("abs_clip_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  af::AttrUtils::SetInt(data2->GetOpDescBarePtr(), "_parent_node_index", 2);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 3, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(data1, 0, ascbc, 2);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("abs_clip");
  CreateAbsClipAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

af::ComputeGraphPtr ShareGraph::AbsFmaFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("abs_fma_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  af::AttrUtils::SetInt(data2->GetOpDescBarePtr(), "_parent_node_index", 2);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 3, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(data1, 0, ascbc, 2);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("abs_fma");
  CreateAbsFmaAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

af::ComputeGraphPtr ShareGraph::AbsFmaBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("abs_fma_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  af::AttrUtils::SetInt(data2->GetOpDescBarePtr(), "_parent_node_index", 2);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 3, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(data1, 0, ascbc, 2);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("abs_fma_bf16");
  CreateAbsFmaBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::LoadLogicalOrStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_logicalor_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_logicalor_store");
  CreateLoadLogicalOrStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::LoadLogicalAndStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_logicaland_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_logicaland_store");
  CreateLoadLogicalAndStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::AxpyAbsFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("axpy_abs_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("axpy_abs");
  CreateAxpyAbsAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::AxpyAbsHalfFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("axpy_abs_half_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("axpy_abs_half");
  CreateAxpyAbsHalfAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::AxpyAddFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("axpy_add_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("axpy_add");
  CreateAxpyAddAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::AddAbsScalarFusedGraph(size_t dims_size, af::DataType dtype) {
  auto builder = GraphBuilder("add_abs_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("add_abs");
  CreateAddAbsScalarAscGraph(sub_graph, dims_size, dtype);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *        data0
 *         |
 *        load0
 *         |
 *        cast0
 *         |
 *        abs
 *         |
 *        cast1
 *         |
 *        store0
 *         |
 *        output0
 */
static void CreateCastAbsAscGraph(af::AscGraph &graph, size_t dims_size, af::DataType in_dtype,
                                  af::DataType out_dtype) {
  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = in_dtype;

  af::ascir_op::Load load("load");
  load.x = data0.y;
  load.y.dtype = in_dtype;

  af::ascir_op::Cast cast0("cast0");
  cast0.x = load.y;
  cast0.y.dtype = out_dtype;

  af::ascir_op::Abs abs("abs");
  abs.x = cast0.y;
  abs.y.dtype = out_dtype;

  af::ascir_op::Cast cast1("cast1");
  cast1.x = abs.y;
  cast1.y.dtype = in_dtype;

  af::ascir_op::Store store("store");
  store.x = cast1.y;
  store.y.dtype = in_dtype;

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  output.y.dtype = in_dtype;

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      output
 *         |
 *       store
 *         |
 *        add
 *       /   \
 *   load0    cast
 *     |       |
 *   data0   Scalar
 */
static void CreateScalarCastAddAscGraph(af::AscGraph &graph, size_t dims_size, af::DataType in_dtype,
                                        af::DataType out_dtype) {
  af::ascir_op::Data data("data0", graph);
  data.ir_attr.SetIndex(0);
  data.y.dtype = out_dtype;

  af::ascir_op::Load load("load");
  load.x = data.y;
  load.y.dtype = out_dtype;

  af::ascir_op::Scalar scalar("scalar", graph);
  scalar.ir_attr.SetValue("1.0");
  scalar.y.dtype = in_dtype;

  af::ascir_op::Cast cast("cast");
  cast.x = scalar.y;
  cast.y.dtype = out_dtype;

  af::ascir_op::Add add("add");
  add.x1 = load.y;
  add.x2 = cast.y;
  add.y.dtype = out_dtype;

  af::ascir_op::Store store("store");
  store.x = add.y;
  store.y.dtype = out_dtype;

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  output.y.dtype = out_dtype;

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *       data0
 *         |
 *       AscBc
 *        |
 *     NetOutput
 */
af::ComputeGraphPtr ShareGraph::CastCastFusedGraph(size_t dims_size, af::DataType in_dtype, af::DataType out_dtype) {
  auto builder = GraphBuilder("cast_abs_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("cast_abs");
  CreateCastAbsAscGraph(sub_graph, dims_size, in_dtype, out_dtype);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *       data0
 *         |
 *       AscBc
 *        |
 *     NetOutput
 */
af::ComputeGraphPtr ShareGraph::ScalarCastAddFusedGraph(size_t dims_size, af::DataType in_dtype,
                                                        af::DataType out_dtype) {
  auto builder = GraphBuilder("scalar_cast_add_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("scalar_cast_add");
  CreateScalarCastAddAscGraph(sub_graph, dims_size, in_dtype, out_dtype);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *        data0
 *         |
 *        load0
 *         |
 *        abs
 *         |
 *        store0
 *         |
 *        output0
 */
static void CreateCastReciprocalAscGraph(af::AscGraph &graph, size_t dims_size, af::DataType in_dtype,
                                         af::DataType out_dtype) {
  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = in_dtype;

  af::ascir_op::Load load("load");
  load.x = data0.y;
  load.y.dtype = in_dtype;

  af::ascir_op::Reciprocal reciprocal("reciprocal");
  reciprocal.x = load.y;
  reciprocal.y.dtype = out_dtype;

  af::ascir_op::Store store("store");
  store.x = reciprocal.y;
  store.y.dtype = out_dtype;

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  output.y.dtype = out_dtype;

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *       data0
 *         |
 *       AscBc
 *        |
 *     NetOutput
 */
af::ComputeGraphPtr ShareGraph::CastCastReciprocalFusedGraph(size_t dims_size, af::DataType in_dtype,
                                                             af::DataType out_dtype) {
  auto builder = GraphBuilder("cast_reciprocal_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("cast_reciprocal_test");
  CreateCastReciprocalAscGraph(sub_graph, dims_size, in_dtype, out_dtype);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *        data0
 *         |
 *        load0
 *         |
 *        IsNan
 *         |
 *        store0
 *         |
 *        output0
 */
static void CreateCastIsNanAscGraph(af::AscGraph &graph, size_t dims_size, af::DataType in_dtype,
                                    af::DataType out_dtype) {
  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = in_dtype;

  af::ascir_op::Load load("load");
  load.x = data0.y;
  load.y.dtype = in_dtype;

  af::ascir_op::Isnan isnan("IsNan");
  isnan.x = load.y;
  isnan.y.dtype = out_dtype;

  af::ascir_op::Store store("store");
  store.x = isnan.y;
  store.y.dtype = out_dtype;

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  output.y.dtype = out_dtype;

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *       data0
 *         |
 *       AscBc
 *        |
 *     NetOutput
 */
af::ComputeGraphPtr ShareGraph::CastCastNanFusedGraph(size_t dims_size, af::DataType in_dtype, af::DataType out_dtype) {
  auto builder = GraphBuilder("cast_nan_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("cast_nan");
  CreateCastIsNanAscGraph(sub_graph, dims_size, in_dtype, out_dtype);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *        data0
 *         |
 *        load0
 *         |
 *        IsFinite
 *         |
 *        store0
 *         |
 *        output0
 */
static void CreateCastIsFiniteAscGraph(af::AscGraph &graph, size_t dims_size, af::DataType in_dtype,
                                       af::DataType out_dtype) {
  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = in_dtype;

  af::ascir_op::Load load("load");
  load.x = data0.y;
  load.y.dtype = in_dtype;

  af::ascir_op::IsFinite Isfinite("IsFinite");
  Isfinite.x = load.y;
  Isfinite.y.dtype = out_dtype;

  af::ascir_op::Store store("store");
  store.x = Isfinite.y;
  store.y.dtype = out_dtype;

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  output.y.dtype = out_dtype;

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *       data0
 *         |
 *       AscBc
 *        |
 *     NetOutput
 */
af::ComputeGraphPtr ShareGraph::CastCastIsFiniteFusedGraph(size_t dims_size, af::DataType in_dtype,
                                                           af::DataType out_dtype) {
  auto builder = GraphBuilder("cast_isfinite_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("cast_isfinite_test");
  CreateCastIsFiniteAscGraph(sub_graph, dims_size, in_dtype, out_dtype);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      data
 *         |
 *      AscBc
 *         /
 *      netoutput
 */
af::ComputeGraphPtr ShareGraph::LoadToStoreAndAbsFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_to_store_and_abs_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 2);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 2, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  builder.AddDataEdge(ascbc, 1, netoutput, 1);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("add_abs");
  CreateLoadToStoreAndAbsAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *                |
 *              store
 *                |
 *              add0
 *              /   \
 *            load1  load0
 *             /      \
 *           data1   data0
 */
static void CreateLoadLoopModeFusedGraph(af::AscGraph &graph, size_t dims_size) {  // 4, 8, 16, 64, 32
  const Expression s0 = graph.CreateSizeVar(4);
  const Expression s1 = graph.CreateSizeVar(8);
  const Expression s2 = graph.CreateSizeVar(16);

  auto Three = Symbol(3);
  auto Two = Symbol(2);
  auto One = Symbol(1);
  auto Zero = Symbol(0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data x0("x0", graph);
  x0.attr.api.compute_type = ComputeType::kComputeInvalid;
  x0.attr.api.type = af::ApiType::kAPITypeBuffer;
  x0.ir_attr.SetIndex(0);
  x0.y.dtype = af::DataType::DT_FLOAT;

  af::ascir_op::Load load0("load0");
  load0.x = x0.y;
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.attr.api.type = af::ApiType::kAPITypeCompute;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load0.y.axis = {z0.id, z1.id, z2.id};
  *load0.y.repeats = {s0, s1, s2};
  *load0.y.strides = {s1 * s2 * Three, s2 * Two, One};
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
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, One};
  load1.y.dtype = af::DataType::DT_FLOAT;
  load1.attr.api.unit = ComputeUnit::kUnitMTE2;

  af::ascir_op::Add add0("add0");
  add0.x1 = load0.y;
  add0.x2 = load1.y;
  add0.attr.api.compute_type = ComputeType::kComputeElewise;
  add0.attr.api.type = af::ApiType::kAPITypeCompute;
  add0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *add0.y.axis = {z0.id, z1.id, z2.id};
  *add0.y.repeats = {s0, s1, s2};
  *add0.y.strides = {s1 * s2, s2, One};
  add0.y.dtype = af::DataType::DT_FLOAT;
  add0.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Store store("store");
  store.x = add0.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.attr.api.type = af::ApiType::kAPITypeCompute;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1 * s2, s2, One};
  store.y.dtype = af::DataType::DT_FLOAT;
  store.attr.api.unit = ComputeUnit::kUnitMTE3;

  af::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = af::DataType::DT_FLOAT;
  y.ir_attr.SetIndex(0);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::LoadNeedLoopModeFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_loop_mode_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_need_loop_mode");
  CreateLoadLoopModeFusedGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      data
 *         |
 *      AscBc
 *         /
 *      netoutput
 */
af::ComputeGraphPtr ShareGraph::LoadUnalignPadFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_unalign_pad_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 2);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 2, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  builder.AddDataEdge(ascbc, 1, netoutput, 1);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("add_abs");
  CreateLoadUnalignPadAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *        abs
 *         |
 *        add
 *       /   \
 *   load0   Scalar
 *     |
 *   data0
 */
static void CreateScalarInfAddAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);

  af::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.ir_attr.SetValue("inf");

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  af::ascir_op::Add add("add");
  add.x1 = x1Local.y;
  add.x2 = scalar0.y;

  af::ascir_op::Store x_out("store");
  x_out.x = add.y;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::ScalarInfAddFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("scalar_inf_add_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("scalar_inf_add");
  CreateScalarInfAddAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *        abs
 *         |
 *        div
 *       /   \
 *   Scalar  load0
 *             |
 *           data0
 */
static void CreateScalarInfDivAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);

  af::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.ir_attr.SetValue("inf");

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  af::ascir_op::Div div("div");
  div.x1 = scalar0.y;
  div.x2 = x1Local.y;

  af::ascir_op::Store x_out("store");
  x_out.x = div.y;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::ScalarDivInfFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("scalar_div_inf_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("scalar_inf_div");
  CreateScalarInfDivAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *    masked_fill
 *    /    |    \
 * load0 is_inf scalar0
 *   |     |
 * data0 load0
 */
static void CreateIsInfMaskedFillAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);
  x.y.dtype = af::DT_FLOAT;

  af::ascir_op::Load load("load0");
  load.x = x.y;
  load.y.dtype = af::DT_FLOAT;

  af::ascir_op::IsInf is_inf("is_inf");
  is_inf.x = load.y;
  is_inf.y.dtype = af::DT_UINT8;

  af::ascir_op::Scalar fill_value("scalar0", graph);
  fill_value.ir_attr.SetValue("-1.0");
  fill_value.y.dtype = af::DT_FLOAT;

  af::ascir_op::MaskedFill masked_fill("masked_fill");
  masked_fill.x = load.y;
  masked_fill.mask = is_inf.y;
  masked_fill.value = fill_value.y;
  masked_fill.y.dtype = af::DT_FLOAT;

  af::ascir_op::Store output_store("store");
  output_store.x = masked_fill.y;
  output_store.y.dtype = af::DT_FLOAT;

  af::ascir_op::Output output("output");
  output.x = output_store.y;
  output.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::IsInfMaskedFillFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("isinf_maskedfill_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("isinf_maskedfill");
  CreateIsInfMaskedFillAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *        gelu
 *         |
 *        add
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateAddGeluAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  af::ascir_op::Add add("add");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;

  af::ascir_op::Gelu gelu("gelu");
  gelu.x = add.y;

  af::ascir_op::Store x_out("store");
  x_out.x = gelu.y;
  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::AddGeluFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("add_gelu_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("add_gelu");
  CreateAddGeluAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *               output
 *                 |
 *               store
 *                 |
 *              compare
 *              /       \
 *         load0      scalar0/data1->load1
 *           |
 *         data0
 */
static void CreateCompareEqAscGraph(af::AscGraph &graph, size_t dims_size, bool is_second_input_tensor,
                                    af::DataType dtype) {
  af::ascir_op::Data x1("compare_data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = dtype;

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = dtype;

  af::ascir_op::Eq eq("eq");
  eq.x1 = x1Local.y;
  if (is_second_input_tensor) {
    af::ascir_op::Data x2("compare_data1", graph);
    x2.ir_attr.SetIndex(1);
    x2.y.dtype = dtype;
    af::ascir_op::Load x2Local("load2");
    x2Local.x = x2.y;
    x2Local.y.dtype = dtype;
    eq.x2 = x2Local.y;
  } else {
    af::ascir_op::Scalar scalar0("scalar0", graph);
    scalar0.ir_attr.SetValue("1");
    eq.x2 = scalar0.y;
  }

  eq.y.dtype = af::DT_UINT8;

  af::ascir_op::Store x_out("store");
  x_out.x = eq.y;
  x_out.y.dtype = af::DT_UINT8;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = af::DT_UINT8;

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

static void CreateCompareGtAscGraph(af::AscGraph &graph, size_t dims_size, bool is_second_input_tensor,
                                    af::DataType dtype) {
  af::ascir_op::Data x1("compare_data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = dtype;

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = dtype;

  af::ascir_op::Gt gt("gt");
  gt.x1 = x1Local.y;
  if (is_second_input_tensor) {
    af::ascir_op::Data x2("compare_data1", graph);
    x2.ir_attr.SetIndex(1);
    x2.y.dtype = dtype;
    af::ascir_op::Load x2Local("load2");
    x2Local.x = x2.y;
    x2Local.y.dtype = dtype;
    gt.x2 = x2Local.y;
  } else {
    af::ascir_op::Scalar scalar0("scalar0", graph);
    scalar0.ir_attr.SetValue("1");
    gt.x2 = scalar0.y;
  }

  gt.y.dtype = af::DT_UINT8;

  af::ascir_op::Store x_out("store");
  x_out.x = gt.y;
  x_out.y.dtype = af::DT_UINT8;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = af::DT_UINT8;

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *               output
 *                 |
 *               store
 *                 |
 *              concat
 *               /   \
 *           load0  load1
 *             |      |
 *           data0   data1
 */
static void CreateConcatAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("concat_data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = af::DT_UINT8;

  af::ascir_op::Data x2("concat_data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = af::DT_UINT8;

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = af::DT_UINT8;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = af::DT_UINT8;

  af::ascir_op::Concat concat("concat");
  concat.x = {x1Local.y, x2Local.y};
  concat.y.dtype = af::DT_UINT8;

  af::ascir_op::Store x_out("store");
  x_out.x = concat.y;
  x_out.y.dtype = af::DT_UINT8;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = af::DT_UINT8;

  ConstructVVAscGraphAxisInfo(graph, dims_size);
  auto concat_node = graph.FindNode("concat");
  auto size = concat_node->attr.sched.axis.size();
  auto repeats = concat_node->outputs()[0]->attr.repeats;
  repeats[size - 1] = repeats[size - 1] + repeats[size - 1];
  auto strides = concat_node->outputs()[0]->attr.strides;
  for (int i = dims_size - 2; i >= 0; i--) {
    strides[i] = af::sym::Mul(repeats[i + 1], strides[i + 1]);
  }
  concat_node->outputs()[0]->attr.strides = strides;
  concat_node->outputs()[0]->attr.repeats = repeats;
  auto store_node = graph.FindNode("store");
  store_node->outputs()[0]->attr.strides = strides;
  store_node->outputs()[0]->attr.repeats = repeats;
}

/**
 *      output
 *         |
 *       store
 *         |
 *        add
 *       /     \
 *   load0      sub
 *     |       /    \
 *   data0  scalar  scalar
 */
static void CreateSubAbsAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x0("data0", graph);
  x0.ir_attr.SetIndex(0);
  af::ascir_op::Load x1Local("load0");
  x1Local.x = x0.y;

  af::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.ir_attr.SetValue("2");
  af::ascir_op::Scalar scalar1("scalar1", graph);
  scalar1.ir_attr.SetValue("1");

  af::ascir_op::Sub sub("subs");
  sub.x1 = scalar0.y;
  sub.x2 = scalar1.y;

  af::ascir_op::Add add("add");
  add.x1 = sub.y;
  add.x2 = x1Local.y;

  af::ascir_op::Store x_out("store");
  x_out.x = add.y;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *     NetOutput
 *         |
 *       AscBc
 *         |
 *       data0
 */
af::ComputeGraphPtr ShareGraph::SubAbsFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("sub_abs_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("sub_abs");
  CreateSubAbsAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *       add
 *        |
 *    transpose
 *        |
 *       sub
 *      /   \
 *   data0  data1
 */
static void CreateSubTransposeAbsAscGraph(af::AscGraph &graph, size_t dims_size, vector<size_t> perms) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  af::ascir_op::Sub sub("subs");
  sub.x1 = x1Local.y;
  sub.x2 = x2Local.y;

  af::ascir_op::Transpose transpose("transpose");
  transpose.x = sub.y;

  af::ascir_op::Abs abs("abs");
  abs.x = transpose.y;

  af::ascir_op::Store x_out("store");
  x_out.x = abs.y;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  ConstructVVAscGraphAxisInfo(graph, dims_size, perms);
}

static void CreateSubTransposeAbsConstAscGraph(af::AscGraph &graph, size_t dims_size, vector<size_t> perms,
                                               std::vector<int> dims) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  af::ascir_op::Sub sub("subs");
  sub.x1 = x1Local.y;
  sub.x2 = x2Local.y;

  af::ascir_op::Transpose transpose("transpose");
  transpose.x = sub.y;

  af::ascir_op::Abs abs("abs");
  abs.x = transpose.y;

  af::ascir_op::Store x_out("store");
  x_out.x = abs.y;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  ConstructVVConstAscGraphAxisInfo(graph, dims_size, perms, dims);
}

/**
 *     NetOutput
 *         |
 *       AscBc
 *      |    |
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::SubTransposeAbsFusedGraph(size_t dims_size, vector<size_t> perms) {
  auto builder = GraphBuilder("sub_transpose_abs_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("sub_transpose_abs");
  CreateSubTransposeAbsAscGraph(sub_graph, dims_size, perms);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateFmaInt8AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = af::DT_INT8;
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = af::DT_INT8;
  af::ascir_op::Data x3("data2", graph);
  x3.ir_attr.SetIndex(2);
  x3.y.dtype = af::DT_INT8;

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = af::DT_INT8;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = af::DT_INT8;

  af::ascir_op::Load x3Local("load2");
  x3Local.x = x3.y;
  x3Local.y.dtype = af::DT_INT8;

  af::ascir_op::Fma fma("fma");
  fma.x1 = x1Local.y;
  fma.x2 = x2Local.y;
  fma.x3 = x3Local.y;
  fma.y.dtype = af::DT_INT8;

  af::ascir_op::Store x_out("store");
  x_out.x = fma.y;
  x_out.y.dtype = af::DT_INT8;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::FmaInt8FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("fma_int8_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  af::AttrUtils::SetInt(data2->GetOpDescBarePtr(), "_parent_node_index", 2);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 3, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(data2, 0, ascbc, 2);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("fma_int8");
  CreateFmaInt8AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateIsfiniteBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data", graph);
  x.y.dtype = af::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load xLocal("load");
  xLocal.x = x.y;
  xLocal.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::IsFinite isfinite("isfinite");
  isfinite.x = xLocal.y;
  isfinite.y.dtype = af::DataType::DT_UINT8;

  af::ascir_op::Store x_out("store");
  x_out.x = isfinite.y;
  x_out.y.dtype = af::DataType::DT_UINT8;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::IsfiniteBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("isfinite_bf16_test");
  auto data = builder.AddNode("data", "Data", 0, 1);
  af::AttrUtils::SetInt(data->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("isfinite_bf16_test");
  CreateIsfiniteBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateIsnanBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data", graph);
  x.y.dtype = af::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load xLocal("load");
  xLocal.x = x.y;
  xLocal.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Isnan isnan("isnan");
  isnan.x = xLocal.y;
  isnan.y.dtype = af::DataType::DT_UINT8;

  af::ascir_op::Store x_out("store");
  x_out.x = isnan.y;
  x_out.y.dtype = af::DataType::DT_UINT8;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::IsnanBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("isnan_bf16_test");
  auto data = builder.AddNode("data", "Data", 0, 1);
  af::AttrUtils::SetInt(data->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("isnan_bf16_test");
  CreateIsnanBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *     NetOutput
 *         |
 *       AscBc
 *      |    |
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::SubTransposeAbsFusedConstGraph(size_t dims_size, vector<size_t> perms,
                                                               std::vector<int> dims) {
  auto builder = GraphBuilder("sub_transpose_abs_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("sub_transpose_abs_const");
  CreateSubTransposeAbsConstAscGraph(sub_graph, dims_size, perms, dims);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *       NetOutput
 *           |
 *         asbc1
 *         /   \
 *     asbc0   data1
 *        |
 *      data0
 */
af::ComputeGraphPtr ShareGraph::CompareFusedGraph(size_t dims_size, bool is_second_input_tensor, af::DataType dtype,
                                                  std::string mode) {
  auto builder = GraphBuilder("compare_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);
  NodePtr data2;
  if (is_second_input_tensor) {
    data2 = builder.AddNode("data2", "Data", 0, 1);
    af::AttrUtils::SetInt(data2->GetOpDescBarePtr(), "_parent_node_index", 2);
  }

  int input_num = 1;
  if (is_second_input_tensor) {
    input_num = 2;
  }
  auto ascbc0 = builder.AddNode("ascbc0", "AscGraph", input_num, 1);
  auto ascbc1 = builder.AddNode("ascbc1", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc0, 0);
  if (is_second_input_tensor) {
    builder.AddDataEdge(data2, 0, ascbc0, 1);
  }
  builder.AddDataEdge(ascbc0, 0, ascbc1, 0);
  builder.AddDataEdge(data1, 0, ascbc1, 1);
  builder.AddDataEdge(ascbc1, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node0 = compute_graph->FindNode("ascbc0");
  af::AscGraph sub_graph0("compare");
  if (mode == "eq") {
    CreateCompareEqAscGraph(sub_graph0, dims_size, is_second_input_tensor, dtype);
  } else if (mode == "gt") {
    CreateCompareGtAscGraph(sub_graph0, dims_size, is_second_input_tensor, dtype);
  }

  std::string sub_graph_str0;
  af::AscGraphUtils::SerializeToReadable(sub_graph0, sub_graph_str0);
  af::AttrUtils::SetStr(ascbc_node0->GetOpDescBarePtr(), "ascgraph", sub_graph_str0);

  auto ascbc_node1 = compute_graph->FindNode("ascbc1");
  af::AscGraph sub_graph1("concat");
  CreateConcatAscGraph(sub_graph1, dims_size);

  std::string sub_graph_str1;
  af::AscGraphUtils::SerializeToReadable(sub_graph1, sub_graph_str1);
  af::AttrUtils::SetStr(ascbc_node1->GetOpDescBarePtr(), "ascgraph", sub_graph_str1);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *        neg
 *         |
 *        add
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateAddNegAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  af::ascir_op::Add add("add");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;

  af::ascir_op::Neg neg("neg");
  neg.x = add.y;

  af::ascir_op::Store x_out("store");
  x_out.x = neg.y;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::AddNegFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("add_neg_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("add_neg");
  CreateAddNegAscGraph(sub_graph, dims_size);
  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *                   output
 *                     |
 *                   store
 *                     |
 *                    mul
 *                /          \
 *              add0        load2
 *            /       \       |
 *       broadcast1  load0  data2
 *            |        |
 *          load1    data0
 *            |
 *          data1
 */
static void CreateBrcInlineAscGraph(af::AscGraph &graph, size_t dims_size) {
  const Expression s0 = graph.CreateSizeVar(320);
  const Expression s1 = graph.CreateSizeVar(32);
  auto One = Symbol(1);
  auto Zero = Symbol(0);
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
  x2.y.dtype = af::DataType::DT_FLOAT;
  x2.ir_attr.SetIndex(2);

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
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::BrcInlineFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("brc_inline_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("brc_inline");
  CreateBrcInlineAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *        where
 *     /   |    \
 *  load0 load1  load2
 *    |     |     |
 *  data0 data1  data2
 */
static void CreateLoadWhereStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = af::DT_UINT8;
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  af::ascir_op::Data x3("data2", graph);
  x3.ir_attr.SetIndex(2);

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = af::DT_UINT8;
  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  af::ascir_op::Load x3Local("load2");
  x3Local.x = x3.y;

  af::ascir_op::Where where("where");
  where.x1 = x1Local.y;
  where.x2 = x2Local.y;
  where.x3 = x3Local.y;

  af::ascir_op::Store x_out("store");
  x_out.x = where.y;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

static void CreateLoadWhereReduceStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = af::DT_UINT8;
  x1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x1.y.axis = {z0.id, z1.id, z2.id};
  *x1.y.repeats = {s0, s1, s2};
  *x1.y.strides = {s1 * s2, s2, af::ops::One};

  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x2.y.axis = {z0.id, z1.id, z2.id};
  *x2.y.repeats = {s0, s1, s2};
  *x2.y.strides = {s1 * s2, s2, af::ops::One};

  af::ascir_op::Data x3("data2", graph);
  x3.ir_attr.SetIndex(2);
  x3.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x3.y.axis = {z0.id, z1.id, z2.id};
  *x3.y.repeats = {s0, s1, s2};
  *x3.y.strides = {s1 * s2, s2, af::ops::One};

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = af::DT_UINT8;
  x1Local.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x1Local.y.axis = {z0.id, z1.id, z2.id};
  *x1Local.y.repeats = {s0, s1, s2};
  *x1Local.y.strides = {s1 * s2, s2, af::ops::One};

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x2Local.y.axis = {z0.id, z1.id, z2.id};
  *x2Local.y.repeats = {s0, s1, s2};
  *x2Local.y.strides = {s1 * s2, s2, af::ops::One};

  af::ascir_op::Load x3Local("load2");
  x3Local.x = x3.y;
  x3Local.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x3Local.y.axis = {z0.id, z1.id, z2.id};
  *x3Local.y.repeats = {s0, s1, s2};
  *x3Local.y.strides = {s1 * s2, s2, af::ops::One};

  af::ascir_op::Where where("where");
  where.x1 = x1Local.y;
  where.x2 = x2Local.y;
  where.x3 = x3Local.y;
  where.attr.sched.axis = {z0.id, z1.id, z2.id};
  *where.y.axis = {z0.id, z1.id, z2.id};
  *where.y.repeats = {s0, s1, s2};
  *where.y.strides = {s1 * s2, s2, af::ops::One};

  af::ascir_op::Max max("max");
  max.x = where.y;
  max.attr.sched.axis = {z0.id, z1.id, z2.id};
  *max.y.axis = {z0.id, z1.id, z2.id};
  *max.y.repeats = {s0, s1, af::ops::One};
  *max.y.strides = {s1, af::ops::One, af::ops::Zero};

  af::ascir_op::Store x_out("store");
  x_out.x = max.y;
  x_out.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x_out.y.axis = {z0.id, z1.id, z2.id};
  *x_out.y.repeats = {s0, s1, af::ops::One};
  *x_out.y.strides = {s1, af::ops::One, af::ops::Zero};

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
}

static void CreateLoadWhereReduceX2IsUbscalarStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = af::DT_UINT8;
  x1.attr.sched.axis = {z0.id, z1.id};
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, af::ops::One};

  af::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.ir_attr.SetValue("100");

  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x2.y.axis = {z0.id, z1.id, z2.id};
  *x2.y.repeats = {s0, s1, s2};
  *x2.y.strides = {s1 * s2, s2, af::ops::One};

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = af::DT_UINT8;
  x1Local.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x1Local.y.axis = {z0.id, z1.id, z2.id};
  *x1Local.y.repeats = {s0, s1, s2};
  *x1Local.y.strides = {s1 * s2, s2, af::ops::One};

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x2Local.y.axis = {z0.id, z1.id, z2.id};
  *x2Local.y.repeats = {s0, s1, s2};
  *x2Local.y.strides = {s1 * s2, s2, af::ops::One};

  af::ascir_op::Where where("where");
  where.x1 = x1Local.y;
  where.x2 = scalar0.y;
  where.x3 = x2Local.y;
  where.attr.sched.axis = {z0.id, z1.id, z2.id};
  *where.y.axis = {z0.id, z1.id, z2.id};
  *where.y.repeats = {s0, s1, s2};
  *where.y.strides = {s1 * s2, s2, af::ops::One};

  af::ascir_op::Max max("max");
  max.x = where.y;
  max.attr.sched.axis = {z0.id, z1.id, z2.id};
  *max.y.axis = {z0.id, z1.id, z2.id};
  *max.y.repeats = {s0, s1, af::ops::One};
  *max.y.strides = {s1, af::ops::One, af::ops::Zero};

  af::ascir_op::Store x_out("store");
  x_out.x = where.y;
  x_out.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x_out.y.axis = {z0.id, z1.id, z2.id};
  *x_out.y.repeats = {s0, s1, af::ops::One};
  *x_out.y.strides = {s1, af::ops::One, af::ops::Zero};

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
}

static void CreateLoadWhereReduceX3IsUbscalarStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = af::DT_UINT8;
  x1.attr.sched.axis = {z0.id, z1.id};
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, af::ops::One};

  af::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.ir_attr.SetValue("200");

  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x2.y.axis = {z0.id, z1.id, z2.id};
  *x2.y.repeats = {s0, s1, s2};
  *x2.y.strides = {s1 * s2, s2, af::ops::One};

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = af::DT_UINT8;
  x1Local.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x1Local.y.axis = {z0.id, z1.id, z2.id};
  *x1Local.y.repeats = {s0, s1, s2};
  *x1Local.y.strides = {s1 * s2, s2, af::ops::One};

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x2Local.y.axis = {z0.id, z1.id, z2.id};
  *x2Local.y.repeats = {s0, s1, s2};
  *x2Local.y.strides = {s1 * s2, s2, af::ops::One};

  af::ascir_op::Where where("where");
  where.x1 = x1Local.y;
  where.x2 = x2Local.y;
  where.x3 = scalar0.y;
  where.attr.sched.axis = {z0.id, z1.id, z2.id};
  *where.y.axis = {z0.id, z1.id, z2.id};
  *where.y.repeats = {s0, s1, s2};
  *where.y.strides = {s1 * s2, s2, af::ops::One};

  af::ascir_op::Max max("max");
  max.x = where.y;
  max.attr.sched.axis = {z0.id, z1.id, z2.id};
  *max.y.axis = {z0.id, z1.id, z2.id};
  *max.y.repeats = {s0, s1, af::ops::One};
  *max.y.strides = {s1, af::ops::One, af::ops::Zero};

  af::ascir_op::Store x_out("store");
  x_out.x = where.y;
  x_out.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x_out.y.axis = {z0.id, z1.id, z2.id};
  *x_out.y.repeats = {s0, s1, af::ops::One};
  *x_out.y.strides = {s1, af::ops::One, af::ops::Zero};

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
}

static void CreateLoadWhereReduceX2X3IsUbscalarStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = af::DT_UINT8;
  x1.attr.sched.axis = {z0.id, z1.id};
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, af::ops::One};

  af::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.ir_attr.SetValue("100");
  af::ascir_op::Scalar scalar1("scalar1", graph);
  scalar1.ir_attr.SetValue("200");

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = af::DT_UINT8;
  x1Local.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x1Local.y.axis = {z0.id, z1.id, z2.id};
  *x1Local.y.repeats = {s0, s1, s2};
  *x1Local.y.strides = {s1 * s2, s2, af::ops::One};

  af::ascir_op::Where where("where");
  where.x1 = x1Local.y;
  where.x2 = scalar0.y;
  where.x3 = scalar1.y;
  where.attr.sched.axis = {z0.id, z1.id, z2.id};
  *where.y.axis = {z0.id, z1.id, z2.id};
  *where.y.repeats = {s0, s1, s2};
  *where.y.strides = {s1 * s2, s2, af::ops::One};

  af::ascir_op::Max max("max");
  max.x = where.y;
  max.attr.sched.axis = {z0.id, z1.id, z2.id};
  *max.y.axis = {z0.id, z1.id, z2.id};
  *max.y.repeats = {s0, s1, af::ops::One};
  *max.y.strides = {s1, af::ops::One, af::ops::Zero};

  af::ascir_op::Store x_out("store");
  x_out.x = where.y;
  x_out.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x_out.y.axis = {z0.id, z1.id, z2.id};
  *x_out.y.repeats = {s0, s1, af::ops::One};
  *x_out.y.strides = {s1, af::ops::One, af::ops::Zero};

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
}

/**
 *      NetOutput
 *         |
 *        AscBc
 *     /   |    \
 *  load0 load1  load2
 *    |     |     |
 *  data0 data1  data2
 */
af::ComputeGraphPtr ShareGraph::LoadWhereStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_where_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  af::AttrUtils::SetInt(data2->GetOpDescBarePtr(), "_parent_node_index", 2);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 3, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(data2, 0, ascbc, 2);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_where_store");
  CreateLoadWhereStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

af::ComputeGraphPtr ShareGraph::LoadWhereReduceStoreFusedGraph(size_t dims_size, bool x2_scalar, bool x3_scalar) {
  auto builder = GraphBuilder("load_where_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  af::AttrUtils::SetInt(data2->GetOpDescBarePtr(), "_parent_node_index", 2);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 3, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(data2, 0, ascbc, 2);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_where_store");
  if (x2_scalar && x3_scalar) {
    CreateLoadWhereReduceX2X3IsUbscalarStoreAscGraph(sub_graph, dims_size);
  } else if (x2_scalar) {
    CreateLoadWhereReduceX2IsUbscalarStoreAscGraph(sub_graph, dims_size);
  } else if (x3_scalar) {
    CreateLoadWhereReduceX3IsUbscalarStoreAscGraph(sub_graph, dims_size);
  } else {
    CreateLoadWhereReduceStoreAscGraph(sub_graph, dims_size);
  }

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *        where
 *     /   |    \
 * load0 scalar0 scalar1
 *    |
 * data0
 */
static void CreateLoadWhereX2X3IsUbscalarStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = af::DT_UINT8;
  af::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.ir_attr.SetValue("100");
  af::ascir_op::Scalar scalar1("scalar1", graph);
  scalar1.ir_attr.SetValue("200");

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = af::DT_UINT8;

  af::ascir_op::Where where("where");
  where.x1 = x1Local.y;
  where.x2 = scalar0.y;
  where.x3 = scalar1.y;

  af::ascir_op::Store x_out("store");
  x_out.x = where.y;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *        AscBc
 *     /   |    \
 * load0 scalar0 scalar1
 *    |
 * data0
 */
af::ComputeGraphPtr ShareGraph::LoadWhereX2X3IsUbscalarStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_where_x2_x3_is_ubscalar_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_where_x2_x3_is_ubscalar_store");
  CreateLoadWhereX2X3IsUbscalarStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *        where
 *     /   |    \
 * load0 scalar0 load1
 *    |             |
 * data0          data1
 */
static void CreateLoadWhereX2IsUbscalarStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = af::DT_UINT8;
  af::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.ir_attr.SetValue("100");
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = af::DT_UINT8;
  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  af::ascir_op::Where where("where");
  where.x1 = x1Local.y;
  where.x2 = scalar0.y;
  where.x3 = x2Local.y;

  af::ascir_op::Store x_out("store");
  x_out.x = where.y;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *        AscBc
 *     /   |    \
 * load0 scalar0 load1
 *    |            |
 * data0          data1
 */
af::ComputeGraphPtr ShareGraph::LoadWhereX2IsUbscalarStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_where_x2_is_ubscalar_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_where_x2_is_ubscalar_store");
  CreateLoadWhereX2IsUbscalarStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *        where
 *     /   |    \
 * load0 load1  scalar0
 *    |    |
 * data0  data1
 */
static void CreateLoadWhereX3IsUbscalarStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = af::DT_UINT8;
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  af::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.ir_attr.SetValue("200");

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = af::DT_UINT8;
  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  af::ascir_op::Where where("where");
  where.x1 = x1Local.y;
  where.x2 = x2Local.y;
  where.x3 = scalar0.y;

  af::ascir_op::Store x_out("store");
  x_out.x = where.y;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *        AscBc
 *     /   |    \
 * load0 load1  scalar0
 *    |    |
 * data0  data1
 */
af::ComputeGraphPtr ShareGraph::LoadWhereX3IsUbscalarStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_where_x3_is_ubscalar_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_where_x3_is_ubscalar_store");
  CreateLoadWhereX3IsUbscalarStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *         data0
 *           |
 *         load0
 *           |
 *       logical_not
 *           |
 *         store
 *           |
 *        ouput0
 */
static void CreateLoadLogicalNotStoreAscGraph(af::AscGraph &graph, size_t dims_size, DataType dt_in = af::DT_FLOAT,
                                              DataType dt_out = af::DT_FLOAT) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = dt_in;

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = dt_in;

  af::ascir_op::LogicalNot logical_not("logical_not");
  logical_not.x = x1Local.y;
  logical_not.y.dtype = dt_out;

  af::ascir_op::Store logical_not_store("store");
  logical_not_store.x = logical_not.y;
  logical_not_store.y.dtype = dt_out;

  af::ascir_op::Output y("output");
  y.x = logical_not_store.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      data
 *         |
 *      AscBc
 *         /
 *      netoutput
 */
af::ComputeGraphPtr ShareGraph::LoadLogicalNotStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_logical_not_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_logical_not_store");
  CreateLoadLogicalNotStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

af::ComputeGraphPtr ShareGraph::LoadLogicalNotStoreFusedGraph(size_t dims_size, DataType dt_in, DataType dt_out) {
  auto builder = GraphBuilder("load_logical_not_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_logical_not_store");
  CreateLoadLogicalNotStoreAscGraph(sub_graph, dims_size, dt_in, dt_out);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *    bitwiseand
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateLoadBitwiseAndStoreAscGraph(af::AscGraph &graph, size_t dims_size, af::DataType in_dtype,
                                              af::DataType out_dtype) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = in_dtype;
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = in_dtype;

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = in_dtype;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = in_dtype;

  af::ascir_op::BitwiseAnd bitwiseand("bitwiseand");
  bitwiseand.x1 = x1Local.y;
  bitwiseand.x2 = x2Local.y;
  bitwiseand.y.dtype = out_dtype;

  af::ascir_op::Store x_out("store");
  x_out.x = bitwiseand.y;
  x_out.y.dtype = out_dtype;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::LoadBitwiseAndStoreFusedGraph(size_t dims_size, af::DataType in_dtype,
                                                              af::DataType out_dtype) {
  auto builder = GraphBuilder("load_bitwise_and_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_bitwise_and_store");
  CreateLoadBitwiseAndStoreAscGraph(sub_graph, dims_size, in_dtype, out_dtype);
  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *    bitwiseor
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateLoadBitwiseOrStoreAscGraph(af::AscGraph &graph, size_t dims_size, af::DataType in_dtype,
                                             af::DataType out_dtype) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = in_dtype;
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = in_dtype;

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = in_dtype;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = in_dtype;

  af::ascir_op::BitwiseOr bitwiseor("bitwiseor");
  bitwiseor.x1 = x1Local.y;
  bitwiseor.x2 = x2Local.y;
  bitwiseor.y.dtype = out_dtype;

  af::ascir_op::Store x_out("store");
  x_out.x = bitwiseor.y;
  x_out.y.dtype = out_dtype;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::LoadBitwiseOrStoreFusedGraph(size_t dims_size, af::DataType in_dtype,
                                                             af::DataType out_dtype) {
  auto builder = GraphBuilder("load_bitwise_or_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_bitwise_or_store");
  CreateLoadBitwiseOrStoreAscGraph(sub_graph, dims_size, in_dtype, out_dtype);
  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *    bitwisexor
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateLoadBitwiseXorStoreAscGraph(af::AscGraph &graph, size_t dims_size, af::DataType in_dtype,
                                              af::DataType out_dtype) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = in_dtype;
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = in_dtype;

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = in_dtype;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = in_dtype;

  af::ascir_op::BitwiseXor bitwisexor("bitwisexor");
  bitwisexor.x1 = x1Local.y;
  bitwisexor.x2 = x2Local.y;
  bitwisexor.y.dtype = out_dtype;

  af::ascir_op::Store x_out("store");
  x_out.x = bitwisexor.y;
  x_out.y.dtype = out_dtype;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::LoadBitwiseXorStoreFusedGraph(size_t dims_size, af::DataType in_dtype,
                                                              af::DataType out_dtype) {
  auto builder = GraphBuilder("load_bitwise_xor_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_bitwise_xor_store");
  CreateLoadBitwiseXorStoreAscGraph(sub_graph, dims_size, in_dtype, out_dtype);
  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *       output
 *         |
 *       store
 *         |
 *     bitwisenot
 *         |
 *       load0
 *         |
 *       data0
 */
static void CreateLoadBitwiseNotStoreAscGraph(af::AscGraph &graph, size_t dims_size, af::DataType in_dtype,
                                              af::DataType out_dtype) {
  af::ascir_op::Data x("data", graph);
  x.ir_attr.SetIndex(0);
  x.y.dtype = in_dtype;

  af::ascir_op::Load xLocal("load");
  xLocal.x = x.y;
  xLocal.y.dtype = in_dtype;

  af::ascir_op::BitwiseNot bitwisenot("bitwisenot");
  bitwisenot.x = xLocal.y;
  bitwisenot.y.dtype = out_dtype;

  af::ascir_op::Store x_out("store");
  x_out.x = bitwisenot.y;
  x_out.y.dtype = out_dtype;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::LoadBitwiseNotStoreFusedGraph(size_t dims_size, af::DataType in_dtype,
                                                              af::DataType out_dtype) {
  auto builder = GraphBuilder("load_bitwise_not_store_test");
  auto data = builder.AddNode("data", "Data", 0, 1);
  af::AttrUtils::SetInt(data->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_bitwise_not_store");
  CreateLoadBitwiseNotStoreAscGraph(sub_graph, dims_size, in_dtype, out_dtype);
  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *        Rsqrt
 *         |
 *        add
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateAddRsqrtAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  af::ascir_op::Add add("add");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;

  af::ascir_op::Rsqrt rsqrt("rsqrt");
  rsqrt.x = add.y;

  af::ascir_op::Store x_out("store");
  x_out.x = rsqrt.y;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::AddRsqrtFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("add_rsqrt_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("add_rsqrt");
  CreateAddRsqrtAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/** BABAB
 *              output
 *                |
 *              store
 *                |
 *              add0
 *              /   \
 *            brc2  load0
 *             /      \
 *           brc1   data0
 *            /
 *          brc0
 *            |
 *          load1
 *            |
 *          data1
 */
static void CreateContinuesBrcAscGraph(af::AscGraph &graph, size_t dims_size) {  // 4, 8, 16, 64, 32
  const Expression s0 = graph.CreateSizeVar(4);
  const Expression s1 = graph.CreateSizeVar(8);
  const Expression s2 = graph.CreateSizeVar(16);
  const Expression s3 = graph.CreateSizeVar(64);
  const Expression s4 = graph.CreateSizeVar(32);

  auto One = Symbol(1);
  auto Zero = Symbol(0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);

  af::ascir_op::Data x0("x0", graph);
  x0.attr.api.compute_type = ComputeType::kComputeInvalid;
  x0.attr.api.type = af::ApiType::kAPITypeBuffer;
  x0.ir_attr.SetIndex(0);
  x0.y.dtype = af::DataType::DT_FLOAT;

  af::ascir_op::Load load0("load0");
  load0.x = x0.y;
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.attr.api.type = af::ApiType::kAPITypeCompute;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load0.y.repeats = {s0, s1, s2, s3, s4};
  *load0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};
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
  load1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *load1.y.repeats = {One, s1, One, s3, One};
  *load1.y.strides = {Zero, s3, Zero, One, Zero};
  load1.y.dtype = af::DataType::DT_FLOAT;
  load1.attr.api.unit = ComputeUnit::kUnitMTE2;

  af::ascir_op::Broadcast brc0("brc0");
  brc0.x = load1.y;
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.attr.api.type = af::ApiType::kAPITypeCompute;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc0.y.repeats = {s0, s1, One, s3, One};
  *brc0.y.strides = {s1 * s3, s3, Zero, One, Zero};
  brc0.y.dtype = af::DataType::DT_FLOAT;
  brc0.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Broadcast brc1("brc1");
  brc1.x = brc0.y;
  brc1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc1.attr.api.type = af::ApiType::kAPITypeCompute;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc1.y.repeats = {s0, s1, s2, s3, One};
  *brc1.y.strides = {s1 * s2 * s3, s2 * s3, s3, One, Zero};
  brc1.y.dtype = af::DataType::DT_FLOAT;
  brc1.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Broadcast brc2("brc2");
  brc2.x = brc1.y;
  brc2.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc2.attr.api.type = af::ApiType::kAPITypeCompute;
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc2.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *brc2.y.repeats = {s0, s1, s2, s3, s4};
  *brc2.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};
  brc2.y.dtype = af::DataType::DT_FLOAT;
  brc2.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Store store1("store1");
  store1.x = brc2.y;
  store1.attr.api.compute_type = ComputeType::kComputeStore;
  store1.attr.api.type = af::ApiType::kAPITypeCompute;
  store1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store1.y.repeats = {s0, s1, s2, s3, s4};
  *store1.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};
  store1.y.dtype = af::DataType::DT_FLOAT;
  store1.attr.api.unit = ComputeUnit::kUnitMTE3;

  af::ascir_op::Output y1("y1");
  y1.x = store1.y;
  y1.attr.api.compute_type = ComputeType::kComputeInvalid;
  y1.attr.api.type = af::ApiType::kAPITypeBuffer;
  y1.y.dtype = af::DataType::DT_FLOAT;
  y1.ir_attr.SetIndex(1);

  af::ascir_op::Add add0("add0");
  add0.x1 = load0.y;
  add0.x2 = brc2.y;
  add0.attr.api.compute_type = ComputeType::kComputeElewise;
  add0.attr.api.type = af::ApiType::kAPITypeCompute;
  add0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *add0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *add0.y.repeats = {s0, s1, s2, s3, s4};
  *add0.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};
  add0.y.dtype = af::DataType::DT_FLOAT;
  add0.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Store store("store");
  store.x = add0.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.attr.api.type = af::ApiType::kAPITypeCompute;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *store.y.repeats = {s0, s1, s2, s3, s4};
  *store.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, One};
  store.y.dtype = af::DataType::DT_FLOAT;
  store.attr.api.unit = ComputeUnit::kUnitMTE3;

  af::ascir_op::Output y("y");
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = af::DataType::DT_FLOAT;
  y.ir_attr.SetIndex(0);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::ContinuesBrcFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("continues_brc_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("continues_brc");
  CreateContinuesBrcAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *                   output
 *                     |
 *                   store
 *                     |
 *                    mul
 *                /          \
 *              add0        load2
 *            /       \       |
 *       broadcast2  load0  data2
 *            |        |
 *       broadcast1    |
 *            |        |
 *       broadcast0    |
 *            |        |
 *          load1    data0
 *            |
 *          data1
 */
static void CreateLoadBrcAscGraphSevenDim(af::AscGraph &graph, size_t dims_size) {
  const Expression s0 = graph.CreateSizeVar(20);
  const Expression s1 = graph.CreateSizeVar(2);
  const Expression s2 = graph.CreateSizeVar(2);
  const Expression s3 = graph.CreateSizeVar(2);
  const Expression s4 = graph.CreateSizeVar(2);
  const Expression s5 = graph.CreateSizeVar(2);
  const Expression s6 = graph.CreateSizeVar(2);
  auto One = Symbol(1);
  auto Zero = Symbol(0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);
  auto z5 = graph.CreateAxis("z5", s5);
  auto z6 = graph.CreateAxis("z6", s6);
  std::vector<Expression> all_size_var = {s0, s1, s2, s3, s4, s5, s6};
  Expression tmp_stride = One;
  std::vector<bool> is_broadcast_axis = {false, true, false, true, false, true, false};

  ascir_op::Data x0("x", graph);
  x0.attr.api.compute_type = ComputeType::kComputeInvalid;
  x0.attr.api.type = ApiType::kAPITypeBuffer;
  x0.ir_attr.SetIndex(0);
  x0.y.dtype = DT_FLOAT;

  ascir_op::Load load0("load0");
  load0.x = x0.y;
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.attr.api.type = ApiType::kAPITypeCompute;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  *load0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  for (auto it = all_size_var.rbegin(); it != all_size_var.rend(); ++it) {
    load0.y.repeats->insert(load0.y.repeats->begin(), *it);
    load0.y.strides->insert(load0.y.strides->begin(), tmp_stride);
    tmp_stride = tmp_stride * (*it);
  }
  load0.y.dtype = DT_FLOAT;
  load0.attr.api.unit = ComputeUnit::kUnitMTE2;

  ascir_op::Data x1("x1", graph);
  x1.attr.api.compute_type = ComputeType::kComputeInvalid;
  x1.attr.api.type = ApiType::kAPITypeBuffer;
  x1.y.dtype = DT_FLOAT;
  x1.ir_attr.SetIndex(1);

  ascir_op::Load load1("load1");
  load1.x = x1.y;
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.attr.api.type = ApiType::kAPITypeCompute;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  *load1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  tmp_stride = One;
  for (int i = static_cast<int>(is_broadcast_axis.size()) - 1; i >= 0; --i) {
    if (is_broadcast_axis[i]) {
      load1.y.repeats->insert(load1.y.repeats->begin(), One);
      load1.y.strides->insert(load1.y.strides->begin(), Zero);
    } else {
      load1.y.repeats->insert(load1.y.repeats->begin(), all_size_var[i]);
      load1.y.strides->insert(load1.y.strides->begin(), tmp_stride);
      tmp_stride = tmp_stride * all_size_var[i];
    }
  }
  load1.y.dtype = DT_FLOAT;
  load1.attr.api.unit = ComputeUnit::kUnitMTE2;

  ascir_op::Broadcast broadcast0("broadcast0");
  broadcast0.x = load1.y;
  broadcast0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  broadcast0.attr.api.type = ApiType::kAPITypeCompute;
  broadcast0.attr.sched.axis = {
      z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id,
  };
  *broadcast0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  *broadcast0.y.repeats = {s0, One, s2, One, s4, s5, s6};
  *broadcast0.y.strides = {s2 * s4 * s5 * s6, Zero, s4 * s5 * s6, Zero, s5 * s6, s6, One};
  broadcast0.y.dtype = DT_FLOAT;
  broadcast0.attr.api.unit = ComputeUnit::kUnitVector;

  ascir_op::Broadcast broadcast1("broadcast1");
  broadcast1.x = broadcast0.y;
  broadcast1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  broadcast1.attr.api.type = ApiType::kAPITypeCompute;
  broadcast1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  *broadcast1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  *broadcast1.y.repeats = {s0, One, s2, s3, s4, s5, s6};
  *broadcast1.y.strides = {s2 * s3 * s4 * s5 * s6, Zero, s3 * s4 * s5 * s6, s4 * s5 * s6, s5 * s6, s6, One};
  broadcast1.y.dtype = DT_FLOAT;
  broadcast1.attr.api.unit = ComputeUnit::kUnitVector;

  ascir_op::Broadcast broadcast2("broadcast2");
  broadcast2.x = broadcast1.y;
  broadcast2.attr.api.compute_type = ComputeType::kComputeBroadcast;
  broadcast2.attr.api.type = ApiType::kAPITypeCompute;
  broadcast2.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  *broadcast2.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  *broadcast2.y.repeats = {s0, s1, s2, s3, s4, s5, s6};
  *broadcast2.y.strides = {
      s1 * s2 * s3 * s4 * s5 * s6, s2 * s3 * s4 * s5 * s6, s3 * s4 * s5 * s6, s4 * s5 * s6, s5 * s6, s6, One};
  broadcast2.y.dtype = DT_FLOAT;
  broadcast2.attr.api.unit = ComputeUnit::kUnitVector;

  ascir_op::Add add0("add0");
  add0.x1 = load0.y;
  add0.x2 = broadcast2.y;
  add0.attr.api.compute_type = ComputeType::kComputeElewise;
  add0.attr.api.type = ApiType::kAPITypeCompute;
  add0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  *add0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  *add0.y.repeats = {s0, s1, s2, s3, s4, s5, s6};
  *add0.y.strides = {
      s1 * s2 * s3 * s4 * s5 * s6, s2 * s3 * s4 * s5 * s6, s3 * s4 * s5 * s6, s4 * s5 * s6, s5 * s6, s6, One};
  add0.y.dtype = DT_FLOAT;
  add0.attr.api.unit = ComputeUnit::kUnitVector;

  ascir_op::Data x2("x2", graph);
  x2.attr.api.compute_type = ComputeType::kComputeInvalid;
  x2.attr.api.type = ApiType::kAPITypeBuffer;
  x2.y.dtype = DT_FLOAT;
  x2.ir_attr.SetIndex(2);

  ascir_op::Load load2("load2");
  load2.x = x2.y;
  load2.attr.api.compute_type = ComputeType::kComputeLoad;
  load2.attr.api.type = ApiType::kAPITypeCompute;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  *load2.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  *load2.y.repeats = {s0, s1, s2, s3, s4, s5, s6};
  *load2.y.strides = {
      s1 * s2 * s3 * s4 * s5 * s6, s2 * s3 * s4 * s5 * s6, s3 * s4 * s5 * s6, s4 * s5 * s6, s5 * s6, s6, One};
  load2.y.dtype = DT_FLOAT;
  load2.attr.api.unit = ComputeUnit::kUnitMTE2;

  ascir_op::Mul mul0("mul0");
  mul0.x1 = load2.y;
  mul0.x2 = add0.y;
  mul0.attr.api.compute_type = ComputeType::kComputeElewise;
  mul0.attr.api.type = ApiType::kAPITypeCompute;
  mul0.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  *mul0.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  *mul0.y.repeats = {s0, s1, s2, s3, s4, s5, s6};
  *mul0.y.strides = {
      s1 * s2 * s3 * s4 * s5 * s6, s2 * s3 * s4 * s5 * s6, s3 * s4 * s5 * s6, s4 * s5 * s6, s5 * s6, s6, One};
  mul0.y.dtype = DT_FLOAT;
  mul0.attr.api.unit = ComputeUnit::kUnitVector;

  ascir_op::Store store("store");
  store.x = mul0.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.attr.api.type = ApiType::kAPITypeCompute;
  store.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  *store.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id, z5.id, z6.id};
  *store.y.repeats = {s0, s1, s2, s3, s4, s5, s6};
  *store.y.strides = {
      s1 * s2 * s3 * s4 * s5 * s6, s2 * s3 * s4 * s5 * s6, s3 * s4 * s5 * s6, s4 * s5 * s6, s5 * s6, s6, One};
  store.y.dtype = DT_FLOAT;
  store.attr.api.unit = ComputeUnit::kUnitMTE3;

  ascir_op::Output y("y");
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = ApiType::kAPITypeBuffer;
  y.y.dtype = DT_FLOAT;
  y.ir_attr.SetIndex(0);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::LoadBrcFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_brc_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_brc");
  CreateLoadBrcAscGraphSevenDim(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateScalarBrcAscGraph(af::AscGraph &graph, size_t dims_size) {
  const Expression s0 = graph.CreateSizeVar(2);
  const Expression s1 = graph.CreateSizeVar(8);
  auto One = Symbol(1);
  auto Zero = Symbol(0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load load1("load1");
  load1.x = x.y;
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.attr.api.type = af::ApiType::kAPITypeCompute;
  load1.attr.sched.axis = {z0.id, z1.id};
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {One, One};
  *load1.y.strides = {Zero, Zero};
  load1.y.dtype = af::DataType::DT_FLOAT;
  load1.attr.api.unit = ComputeUnit::kUnitMTE2;

  af::ascir_op::Broadcast brc0("brc0");
  brc0.x = load1.y;
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.attr.api.type = af::ApiType::kAPITypeCompute;
  brc0.attr.sched.axis = {z0.id, z1.id};
  *brc0.y.axis = {z0.id, z1.id};
  *brc0.y.repeats = {s0, s1};
  *brc0.y.strides = {s1, One};
  brc0.y.dtype = af::DataType::DT_FLOAT;
  brc0.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Store store1("store1");
  store1.x = brc0.y;
  store1.attr.api.compute_type = ComputeType::kComputeStore;
  store1.attr.api.type = af::ApiType::kAPITypeCompute;
  store1.attr.sched.axis = {z0.id, z1.id};
  *store1.y.axis = {z0.id, z1.id};
  *store1.y.repeats = {s0, s1};
  *store1.y.strides = {s1, One};
  store1.y.dtype = af::DataType::DT_FLOAT;
  store1.attr.api.unit = ComputeUnit::kUnitMTE3;

  af::ascir_op::Output y1("y1");
  y1.x = store1.y;
  y1.attr.api.compute_type = ComputeType::kComputeInvalid;
  y1.attr.api.type = af::ApiType::kAPITypeBuffer;
  y1.y.dtype = af::DataType::DT_FLOAT;
  y1.ir_attr.SetIndex(0);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::ScalarBrcFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("scalar_brc_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("continues_brc");
  CreateScalarBrcAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  scalar0
 */
af::ComputeGraphPtr ShareGraph::AbsBrcAddFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("abs_brc_add_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("abs_brc_add");
  CreateAbsBrcAddAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  scalar0
 */
af::ComputeGraphPtr ShareGraph::UbScalarBrcAbsAddFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("ub_scalar_brc_abs_add_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("ub_scalar_brc_abs_add");
  CreateUbScalerBrcAbsAddAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *         data0
 *           |
 *         load0
 *           |
 *       LeakyRelu
 *           |
 *         store
 *           |
 *        ouput0
 */
static void CreateLoadLeakyReluStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  af::ascir_op::LeakyRelu leakyrelu0("leakyrelu0");
  leakyrelu0.x = x1Local.y;
  leakyrelu0.ir_attr.SetNegative_slope(0.8);

  af::ascir_op::Store leakyrelu0_store("store");
  leakyrelu0_store.x = leakyrelu0.y;

  af::ascir_op::Output y("output");
  y.x = leakyrelu0_store.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::LoadLeakyReluStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_leaky_relu_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_leaky_relu_store");
  CreateLoadLeakyReluStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *         data0
 *           |
 *         load0
 *           |
 *        Sigmoid
 *           |
 *         store
 *           |
 *        ouput0
 */
static void CreateLoadSigmoidStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  af::ascir_op::Sigmoid sigmoid0("sigmoid0");
  sigmoid0.x = x1Local.y;

  af::ascir_op::Store sigmoid0_store("store");
  sigmoid0_store.x = sigmoid0.y;

  af::ascir_op::Output y("output");
  y.x = sigmoid0_store.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::LoadSigmoidStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_sigmoid_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_sigmoid_store");
  CreateLoadSigmoidStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *         data0
 *           |
 *         load0
 *           |
 *          GE
 *           |
 *         Cast
 *           |
 *         store
 *           |
 *        ouput0
 */
static void CreateLoadCompareStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x1.y.dtype = af::DT_FLOAT;
  x2.y.dtype = af::DT_FLOAT;

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = af::DT_FLOAT;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = af::DT_FLOAT;

  af::ascir_op::Ge ge0("ge0");
  ge0.x1 = x1Local.y;
  ge0.x2 = x2Local.y;
  ge0.y.dtype = af::DT_UINT8;

  af::ascir_op::Cast cast0("cast0");
  cast0.x = ge0.y;
  cast0.y.dtype = af::DT_FLOAT;

  af::ascir_op::Store store0("store0");
  store0.x = cast0.y;
  store0.y.dtype = af::DT_FLOAT;
  af::ascir_op::Output y("output");
  y.x = store0.y;
  y.y.dtype = af::DT_FLOAT;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::LoadCompareStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_compare_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_compare_store");
  CreateLoadCompareStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateLoadLeBoolStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  af::ascir_op::Data x3("data2", graph);
  x3.ir_attr.SetIndex(2);
  x1.y.dtype = af::DT_BOOL;
  x2.y.dtype = af::DT_BOOL;
  x3.y.dtype = af::DT_BOOL;

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = af::DT_BOOL;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = af::DT_BOOL;

  af::ascir_op::Load x3Local("load2");
  x3Local.x = x3.y;
  x3Local.y.dtype = af::DT_BOOL;

  af::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.ir_attr.SetValue("0");
  scalar0.y.dtype = af::DT_BOOL;

  af::ascir_op::Le le0("le0");
  le0.x1 = x1Local.y;
  le0.x2 = x2Local.y;
  le0.y.dtype = af::DT_BOOL;

  af::ascir_op::Le le1("le1");
  le1.x1 = x3Local.y;
  le1.x2 = scalar0.y;
  le1.y.dtype = af::DT_BOOL;

  af::ascir_op::Store store0("store0");
  store0.x = le0.y;
  store0.y.dtype = af::DT_BOOL;
  af::ascir_op::Store store1("store1");
  store1.x = le1.y;
  store1.y.dtype = af::DT_BOOL;

  af::ascir_op::Output y0("output0");
  y0.x = store0.y;
  y0.y.dtype = af::DT_BOOL;
  y0.ir_attr.SetIndex(0);
  af::ascir_op::Output y1("output1");
  y1.x = store1.y;
  y1.y.dtype = af::DT_BOOL;
  y1.ir_attr.SetIndex(1);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput(2)
 *         |    |
 *       AscBc(3in, 2out)
 *      /  |    |
 *  data0 data1 data2
 */
af::ComputeGraphPtr ShareGraph::LoadLeBoolStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("le_bool_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  af::AttrUtils::SetInt(data2->GetOpDescBarePtr(), "_parent_node_index", 2);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 3, 2);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 2, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(data2, 0, ascbc, 2);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  builder.AddDataEdge(ascbc, 1, netoutput, 1);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("le_bool_store");
  CreateLoadLeBoolStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *         data0  data1
 *            \    /
 *          load0 load1
 *          /  \   /  \
 *        and  or  xor (load0,load1)
 *          |    |   |
 *        not0(load0)
 *          |    |   |
 *       store0-3
 *          |    |   |
 *      output0 output1..3
 */
static void CreateLoadBitwiseBoolStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = af::DT_BOOL;
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = af::DT_BOOL;

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = af::DT_BOOL;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = af::DT_BOOL;

  af::ascir_op::BitwiseAnd and0("and0");
  and0.x1 = x1Local.y;
  and0.x2 = x2Local.y;
  and0.y.dtype = af::DT_BOOL;

  af::ascir_op::BitwiseOr or0("or0");
  or0.x1 = x1Local.y;
  or0.x2 = x2Local.y;
  or0.y.dtype = af::DT_BOOL;

  af::ascir_op::BitwiseXor xor0("xor0");
  xor0.x1 = x1Local.y;
  xor0.x2 = x2Local.y;
  xor0.y.dtype = af::DT_BOOL;

  af::ascir_op::BitwiseNot not0("not0");
  not0.x = x1Local.y;
  not0.y.dtype = af::DT_BOOL;

  af::ascir_op::Store store0("store0");
  store0.x = and0.y;
  store0.y.dtype = af::DT_BOOL;
  af::ascir_op::Store store1("store1");
  store1.x = or0.y;
  store1.y.dtype = af::DT_BOOL;
  af::ascir_op::Store store2("store2");
  store2.x = xor0.y;
  store2.y.dtype = af::DT_BOOL;
  af::ascir_op::Store store3("store3");
  store3.x = not0.y;
  store3.y.dtype = af::DT_BOOL;

  af::ascir_op::Output y0("output0");
  y0.x = store0.y;
  y0.y.dtype = af::DT_BOOL;
  y0.ir_attr.SetIndex(0);
  af::ascir_op::Output y1("output1");
  y1.x = store1.y;
  y1.y.dtype = af::DT_BOOL;
  y1.ir_attr.SetIndex(1);
  af::ascir_op::Output y2("output2");
  y2.x = store2.y;
  y2.y.dtype = af::DT_BOOL;
  y2.ir_attr.SetIndex(2);
  af::ascir_op::Output y3("output3");
  y3.x = store3.y;
  y3.y.dtype = af::DT_BOOL;
  y3.ir_attr.SetIndex(3);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::LoadBitwiseBoolStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("bitwise_bool_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 4);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 4, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  builder.AddDataEdge(ascbc, 1, netoutput, 1);
  builder.AddDataEdge(ascbc, 2, netoutput, 2);
  builder.AddDataEdge(ascbc, 3, netoutput, 3);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("bitwise_bool_store");
  CreateLoadBitwiseBoolStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *         data0
 *           |
 *         load0
 *           |
 *          Erf
 *           |
 *         store
 *           |
 *        ouput0
 */
static void CreateLoadErfStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  af::ascir_op::Erf sigmoid0("sigmoid0");
  sigmoid0.x = x1Local.y;

  af::ascir_op::Store sigmoid0_store("store");
  sigmoid0_store.x = sigmoid0.y;

  af::ascir_op::Output y("output");
  y.x = sigmoid0_store.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::LoadErfStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_erf_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_erf_store");
  CreateLoadErfStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateBrcReduceAscGraph(af::AscGraph &graph, size_t dims_size) {
  const Expression s0 = graph.CreateSizeVar(2);
  const Expression s1 = graph.CreateSizeVar(8);
  const Expression s2 = graph.CreateSizeVar(7);
  auto One = Symbol(1);
  auto Zero = Symbol(0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load load1("load1");
  load1.x = x.y;
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.attr.api.type = af::ApiType::kAPITypeCompute;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, One, s2};
  *load1.y.strides = {s2, Zero, One};
  load1.y.dtype = af::DataType::DT_FLOAT;
  load1.attr.api.unit = ComputeUnit::kUnitMTE2;

  af::ascir_op::Abs abs("abs");
  abs.x = load1.y;
  abs.attr.api.compute_type = ComputeType::kComputeElewise;
  abs.attr.api.type = af::ApiType::kAPITypeCompute;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id};
  *abs.y.axis = {z0.id, z1.id, z2.id};
  *abs.y.repeats = {s0, One, s2};
  *abs.y.strides = {s2, Zero, One};
  abs.y.dtype = af::DataType::DT_FLOAT;
  abs.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Broadcast brc0("brc0");
  brc0.x = abs.y;
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.attr.api.type = af::ApiType::kAPITypeCompute;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *brc0.y.axis = {z0.id, z1.id, z2.id};
  *brc0.y.repeats = {s0, s1, s2};
  *brc0.y.strides = {s1 * s2, s2, One};
  brc0.y.dtype = af::DataType::DT_FLOAT;
  brc0.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Max max("max");
  max.attr.tmp_buffers = {{{af::Symbol(8192), -1}, MemAttr(), 0}, {{af::Symbol(8192), 0}, MemAttr(), 1}};
  max.x = brc0.y;
  max.attr.sched.axis = {z0.id, z1.id, z2.id};
  *max.y.axis = {z0.id, z1.id, z2.id};
  *max.y.repeats = {s0, One, s2};
  *max.y.strides = {s2, Zero, One};

  af::ascir_op::Store store1("store1");
  store1.x = max.y;
  store1.attr.api.compute_type = ComputeType::kComputeStore;
  store1.attr.api.type = af::ApiType::kAPITypeCompute;
  store1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *store1.y.axis = {z0.id, z1.id, z2.id};
  *store1.y.repeats = {s0, One, s2};
  *store1.y.strides = {s2, Zero, One};
  store1.y.dtype = af::DataType::DT_FLOAT;
  store1.attr.api.unit = ComputeUnit::kUnitMTE3;

  af::ascir_op::Output y1("y1");
  y1.x = store1.y;
  y1.attr.api.compute_type = ComputeType::kComputeInvalid;
  y1.attr.api.type = af::ApiType::kAPITypeBuffer;
  y1.y.dtype = af::DataType::DT_FLOAT;
  y1.ir_attr.SetIndex(0);
}

/**
 *         data0
 *           |
 *         load0
 *           |
 *          Tanh
 *           |
 *         store
 *           |
 *         ouput0
 */
static void CreateLoadTanhStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  af::ascir_op::Tanh sigmoid0("sigmoid0");
  sigmoid0.x = x1Local.y;

  af::ascir_op::Store sigmoid0_store("store");
  sigmoid0_store.x = sigmoid0.y;

  af::ascir_op::Output y("output");
  y.x = sigmoid0_store.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

static void CreateLoadTanhBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data", graph);
  x.y.dtype = af::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load xLocal("load");
  xLocal.x = x.y;
  xLocal.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Tanh tanh("tanh");
  tanh.x = xLocal.y;
  tanh.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Store x_out("store");
  x_out.x = tanh.y;
  x_out.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Output y("output");
  y.x = x_out.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::LoadTanhBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_tanh_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_tanh_bf16_test");
  CreateLoadTanhBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::LoadTanhStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_tanh_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_tanh_store");
  CreateLoadTanhStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static af::ComputeGraphPtr CreateOneInputAscGraphComputeGraph(const std::string &graph_name,
                                                              const std::string &sub_graph_name, size_t dims_size,
                                                              void (*create_sub_graph)(af::AscGraph &, size_t)) {
  auto builder = GraphBuilder(graph_name);
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  auto one_input_compute_graph = builder.GetGraph();
  if (one_input_compute_graph == nullptr) {
    return nullptr;
  }
  auto one_input_ascbc_node = one_input_compute_graph->FindNode("ascbc");
  af::AscGraph one_input_sub_graph(sub_graph_name.c_str());
  create_sub_graph(one_input_sub_graph, dims_size);

  std::string one_input_sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(one_input_sub_graph, one_input_sub_graph_str);
  af::AttrUtils::SetStr(one_input_ascbc_node->GetOpDescBarePtr(), "ascgraph", one_input_sub_graph_str);
  return one_input_compute_graph;
}

static af::ComputeGraphPtr CreateTwoInputAscGraphComputeGraph(const std::string &graph_name,
                                                              const std::string &sub_graph_name, size_t dims_size,
                                                              void (*create_sub_graph)(af::AscGraph &, size_t)) {
  auto builder = GraphBuilder(graph_name);
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  auto compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph(sub_graph_name.c_str());
  create_sub_graph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *         data0
 *           |
 *         load0
 *           |
 *    ModifiedBesselI0
 *           |
 *         store
 *           |
 *        output0
 */
static void CreateLoadModifiedBesselI0StoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  af::ascir_op::ModifiedBesselI0 i0("modified_bessel_i0");
  i0.x = x1Local.y;

  af::ascir_op::Store i0_store("store");
  i0_store.x = i0.y;

  af::ascir_op::Output y("output");
  y.x = i0_store.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::LoadModifiedBesselI0StoreFusedGraph(size_t dims_size) {
  return CreateOneInputAscGraphComputeGraph("load_modified_bessel_i0_store_test", "load_modified_bessel_i0_store",
                                            dims_size, CreateLoadModifiedBesselI0StoreAscGraph);
}

/**
 *         data0
 *           |
 *         load0
 *           |
 *    ModifiedBesselI1
 *           |
 *         store
 *           |
 *        output0
 */
static void CreateLoadModifiedBesselI1StoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load load("load0");
  load.x = x.y;

  af::ascir_op::ModifiedBesselI1 op("modified_bessel_i1");
  op.x = load.y;

  af::ascir_op::Store store("store");
  store.x = op.y;

  af::ascir_op::Output y("output");
  y.x = store.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::LoadModifiedBesselI1StoreFusedGraph(size_t dims_size) {
  return CreateOneInputAscGraphComputeGraph("load_modified_bessel_i1_store_test", "load_modified_bessel_i1_store",
                                            dims_size, CreateLoadModifiedBesselI1StoreAscGraph);
}

/**
 *         data0
 *           |
 *         load0
 *           |
 *    ModifiedBesselK0
 *           |
 *         store
 *           |
 *        output0
 */
static void CreateLoadModifiedBesselK0StoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load load("load0");
  load.x = x.y;

  af::ascir_op::ModifiedBesselK0 op("modified_bessel_k0");
  op.x = load.y;

  af::ascir_op::Store store("store");
  store.x = op.y;

  af::ascir_op::Output y("output");
  y.x = store.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::LoadModifiedBesselK0StoreFusedGraph(size_t dims_size) {
  return CreateOneInputAscGraphComputeGraph("load_modified_bessel_k0_store_test", "load_modified_bessel_k0_store",
                                            dims_size, CreateLoadModifiedBesselK0StoreAscGraph);
}

/**
 *         data0
 *           |
 *         load0
 *           |
 *    ModifiedBesselK1
 *           |
 *         store
 *           |
 *        output0
 */
static void CreateLoadModifiedBesselK1StoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load load("load0");
  load.x = x.y;

  af::ascir_op::ModifiedBesselK1 op("modified_bessel_k1");
  op.x = load.y;

  af::ascir_op::Store store("store");
  store.x = op.y;

  af::ascir_op::Output y("output");
  y.x = store.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::LoadModifiedBesselK1StoreFusedGraph(size_t dims_size) {
  return CreateOneInputAscGraphComputeGraph("load_modified_bessel_k1_store_test", "load_modified_bessel_k1_store",
                                            dims_size, CreateLoadModifiedBesselK1StoreAscGraph);
}

/**
 *      data0    data1
 *        |        |
 *      load0    load1
 *         \      /
 *   LaguerrePolynomialL
 *          |
 *        store
 *          |
 *       output0
 */
static void CreateLaguerrePolynomialLStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);

  af::ascir_op::Data n("data1", graph);
  n.ir_attr.SetIndex(1);
  n.y.dtype = af::DT_INT32;

  af::ascir_op::Load load_x("load0");
  load_x.x = x.y;

  af::ascir_op::Load load_n("load1");
  load_n.x = n.y;
  load_n.y.dtype = af::DT_INT32;

  af::ascir_op::LaguerrePolynomialL op("laguerre_polynomial_l");
  op.x = load_x.y;
  op.n = load_n.y;

  af::ascir_op::Store store("store");
  store.x = op.y;

  af::ascir_op::Output y("output");
  y.x = store.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::LaguerrePolynomialLStoreFusedGraph(size_t dims_size) {
  return CreateTwoInputAscGraphComputeGraph("laguerre_polynomial_l_store_test", "laguerre_polynomial_l_store",
                                            dims_size, CreateLaguerrePolynomialLStoreAscGraph);
}

/**
 *      data0    data1
 *        |        |
 *      load0    load1
 *         \      /
 *   LegendrePolynomialP
 *          |
 *        store
 *          |
 *       output0
 */
static void CreateLegendrePolynomialPStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data polynomial_x("data0", graph);
  polynomial_x.ir_attr.SetIndex(0);

  af::ascir_op::Data polynomial_n("data1", graph);
  polynomial_n.ir_attr.SetIndex(1);
  polynomial_n.y.dtype = af::DT_INT32;

  af::ascir_op::Load load_polynomial_x("load0");
  load_polynomial_x.x = polynomial_x.y;

  af::ascir_op::Load load_polynomial_n("load1");
  load_polynomial_n.x = polynomial_n.y;
  load_polynomial_n.y.dtype = af::DT_INT32;

  af::ascir_op::LegendrePolynomialP op("legendre_polynomial_p");
  op.x = load_polynomial_x.y;
  op.n = load_polynomial_n.y;

  af::ascir_op::Store store("store");
  store.x = op.y;

  af::ascir_op::Output y("output");
  y.x = store.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::LegendrePolynomialPStoreFusedGraph(size_t dims_size) {
  return CreateTwoInputAscGraphComputeGraph("legendre_polynomial_p_store_test", "legendre_polynomial_p_store",
                                            dims_size, CreateLegendrePolynomialPStoreAscGraph);
}

/**
 *         data0
 *           |
 *         load0
 *           |
 *        AiryAi
 *           |
 *         store
 *           |
 *        output0
 */
static void CreateLoadAiryAiStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load load("load0");
  load.x = x.y;

  af::ascir_op::AiryAi op("airy_ai");
  op.x = load.y;

  af::ascir_op::Store store("store");
  store.x = op.y;

  af::ascir_op::Output y("output");
  y.x = store.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::LoadAiryAiStoreFusedGraph(size_t dims_size) {
  return CreateOneInputAscGraphComputeGraph("load_airy_ai_store_test", "load_airy_ai_store", dims_size,
                                            CreateLoadAiryAiStoreAscGraph);
}

/**
 *         data0
 *           |
 *         load0
 *           |
 *        Erfinv
 *           |
 *         store
 *           |
 *        output0
 */
static void CreateLoadErfinvStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load load("load0");
  load.x = x.y;

  af::ascir_op::Erfinv op("erfinv");
  op.x = load.y;

  af::ascir_op::Store store("store");
  store.x = op.y;

  af::ascir_op::Output y("output");
  y.x = store.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::LoadErfinvStoreFusedGraph(size_t dims_size) {
  return CreateOneInputAscGraphComputeGraph("load_erfinv_store_test", "load_erfinv_store", dims_size,
                                            CreateLoadErfinvStoreAscGraph);
}

/*
 * Diagram for bessel_j0_store_test:
 *        data0
 *           |
 *         load
 *           |
 *       BesselJ0
 *           |
 *         store
 *           |
 *        output0
 */
static void CreateBesselJ0StoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load load("load0");
  load.x = x.y;

  af::ascir_op::BesselJ0 op("bessel_j0");
  op.x = load.y;

  af::ascir_op::Store store("store");
  store.x = op.y;

  af::ascir_op::Output y("output");
  y.x = store.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::BesselJ0StoreFusedGraph(size_t dims_size) {
  return CreateOneInputAscGraphComputeGraph("bessel_j0_store_test", "bessel_j0_store", dims_size,
                                            CreateBesselJ0StoreAscGraph);
}

template <typename Op>
static void CreateUnarySpecialStoreAscGraph(af::AscGraph &graph, size_t dims_size, const char *op_name) {
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load load("load0");
  load.x = x.y;

  Op op(op_name);
  op.x = load.y;

  af::ascir_op::Store store("store");
  store.x = op.y;

  af::ascir_op::Output y("output");
  y.x = store.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

static void CreateI1eStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  CreateUnarySpecialStoreAscGraph<af::ascir_op::I1e>(graph, dims_size, "i1e");
}

af::ComputeGraphPtr ShareGraph::I1eStoreFusedGraph(size_t dims_size) {
  return CreateOneInputAscGraphComputeGraph("i1e_store_test", "i1e_store", dims_size, CreateI1eStoreAscGraph);
}

static void CreateI0eStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  CreateUnarySpecialStoreAscGraph<af::ascir_op::I0e>(graph, dims_size, "i0e");
}

af::ComputeGraphPtr ShareGraph::I0eStoreFusedGraph(size_t dims_size) {
  return CreateOneInputAscGraphComputeGraph("i0e_store_test", "i0e_store", dims_size, CreateI0eStoreAscGraph);
}

static void CreateI0StoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  CreateUnarySpecialStoreAscGraph<af::ascir_op::I0>(graph, dims_size, "i0");
}

af::ComputeGraphPtr ShareGraph::I0StoreFusedGraph(size_t dims_size) {
  return CreateOneInputAscGraphComputeGraph("i0_store_test", "i0_store", dims_size, CreateI0StoreAscGraph);
}

static void CreateBesselJ1StoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  CreateUnarySpecialStoreAscGraph<af::ascir_op::BesselJ1>(graph, dims_size, "bessel_j1");
}

af::ComputeGraphPtr ShareGraph::BesselJ1StoreFusedGraph(size_t dims_size) {
  return CreateOneInputAscGraphComputeGraph("bessel_j1_store_test", "bessel_j1_store", dims_size,
                                            CreateBesselJ1StoreAscGraph);
}

static void CreateBesselY0StoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  CreateUnarySpecialStoreAscGraph<af::ascir_op::BesselY0>(graph, dims_size, "bessel_y0");
}

af::ComputeGraphPtr ShareGraph::BesselY0StoreFusedGraph(size_t dims_size) {
  return CreateOneInputAscGraphComputeGraph("bessel_y0_store_test", "bessel_y0_store", dims_size,
                                            CreateBesselY0StoreAscGraph);
}

static void CreateBesselY1StoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  CreateUnarySpecialStoreAscGraph<af::ascir_op::BesselY1>(graph, dims_size, "bessel_y1");
}

af::ComputeGraphPtr ShareGraph::BesselY1StoreFusedGraph(size_t dims_size) {
  return CreateOneInputAscGraphComputeGraph("bessel_y1_store_test", "bessel_y1_store", dims_size,
                                            CreateBesselY1StoreAscGraph);
}

static void CreateScaledModifiedBesselK0StoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  CreateUnarySpecialStoreAscGraph<af::ascir_op::ScaledModifiedBesselK0>(graph, dims_size, "scaled_modified_bessel_k0");
}

af::ComputeGraphPtr ShareGraph::ScaledModifiedBesselK0StoreFusedGraph(size_t dims_size) {
  return CreateOneInputAscGraphComputeGraph("scaled_modified_bessel_k0_store_test", "scaled_modified_bessel_k0_store",
                                            dims_size, CreateScaledModifiedBesselK0StoreAscGraph);
}

static void CreateScaledModifiedBesselK1StoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  CreateUnarySpecialStoreAscGraph<af::ascir_op::ScaledModifiedBesselK1>(graph, dims_size, "scaled_modified_bessel_k1");
}

af::ComputeGraphPtr ShareGraph::ScaledModifiedBesselK1StoreFusedGraph(size_t dims_size) {
  return CreateOneInputAscGraphComputeGraph("scaled_modified_bessel_k1_store_test", "scaled_modified_bessel_k1_store",
                                            dims_size, CreateScaledModifiedBesselK1StoreAscGraph);
}

static void CreateSphericalBesselJ0StoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  CreateUnarySpecialStoreAscGraph<af::ascir_op::SphericalBesselJ0>(graph, dims_size, "spherical_bessel_j0");
}

af::ComputeGraphPtr ShareGraph::SphericalBesselJ0StoreFusedGraph(size_t dims_size) {
  return CreateOneInputAscGraphComputeGraph("spherical_bessel_j0_store_test", "spherical_bessel_j0_store", dims_size,
                                            CreateSphericalBesselJ0StoreAscGraph);
}

static void CreateNdtrStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  CreateUnarySpecialStoreAscGraph<af::ascir_op::Ndtr>(graph, dims_size, "ndtr");
}

af::ComputeGraphPtr ShareGraph::NdtrStoreFusedGraph(size_t dims_size) {
  return CreateOneInputAscGraphComputeGraph("ndtr_store_test", "ndtr_store", dims_size, CreateNdtrStoreAscGraph);
}

static void CreateNdtriStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  CreateUnarySpecialStoreAscGraph<af::ascir_op::Ndtri>(graph, dims_size, "ndtri");
}

static void CreateLogNdtrStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  CreateUnarySpecialStoreAscGraph<af::ascir_op::LogNdtr>(graph, dims_size, "log_ndtr");
}

af::ComputeGraphPtr ShareGraph::NdtriStoreFusedGraph(size_t dims_size) {
  return CreateOneInputAscGraphComputeGraph("ndtri_store_test", "ndtri_store", dims_size, CreateNdtriStoreAscGraph);
}

af::ComputeGraphPtr ShareGraph::LogNdtrStoreFusedGraph(size_t dims_size) {
  return CreateOneInputAscGraphComputeGraph("log_ndtr_store_test", "log_ndtr_store", dims_size,
                                            CreateLogNdtrStoreAscGraph);
}

template <typename Op>
static void CreateBinarySpecialStoreAscGraph(af::AscGraph &graph, size_t dims_size, const char *op_name) {
  af::ascir_op::Data x0("data0", graph);
  x0.ir_attr.SetIndex(0);
  af::ascir_op::Data x1("data1", graph);
  x1.ir_attr.SetIndex(1);

  af::ascir_op::Load load0("load0");
  load0.x = x0.y;
  af::ascir_op::Load load1("load1");
  load1.x = x1.y;

  Op op(op_name);
  op.x1 = load0.y;
  op.x2 = load1.y;

  af::ascir_op::Store store("store");
  store.x = op.y;

  af::ascir_op::Output y("output");
  y.x = store.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

static void CreateZetaStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  CreateBinarySpecialStoreAscGraph<af::ascir_op::Zeta>(graph, dims_size, "zeta");
}

static void CreateNextAfterStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  CreateBinarySpecialStoreAscGraph<af::ascir_op::NextAfter>(graph, dims_size, "next_after");
}

static void CreatePolyGammaStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);

  af::ascir_op::Data n("data1", graph);
  n.ir_attr.SetIndex(1);
  n.y.dtype = af::DT_INT32;

  af::ascir_op::Load load_x("load0");
  load_x.x = x.y;

  af::ascir_op::Load load_n("load1");
  load_n.x = n.y;
  load_n.y.dtype = af::DT_INT32;

  af::ascir_op::PolyGamma op("poly_gamma");
  op.x = load_x.y;
  op.n = load_n.y;

  af::ascir_op::Store store("store");
  store.x = op.y;

  af::ascir_op::Output y("output");
  y.x = store.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::ZetaStoreFusedGraph(size_t dims_size) {
  return CreateTwoInputAscGraphComputeGraph("zeta_store_test", "zeta_store", dims_size, CreateZetaStoreAscGraph);
}

af::ComputeGraphPtr ShareGraph::NextAfterStoreFusedGraph(size_t dims_size) {
  return CreateTwoInputAscGraphComputeGraph("next_after_store_test", "next_after_store", dims_size,
                                            CreateNextAfterStoreAscGraph);
}

af::ComputeGraphPtr ShareGraph::PolyGammaStoreFusedGraph(size_t dims_size) {
  return CreateTwoInputAscGraphComputeGraph("poly_gamma_store_test", "poly_gamma_store", dims_size,
                                            CreatePolyGammaStoreAscGraph);
}

/**
 *         data0
 *           |
 *         load0
 *           |
 *       SignBit
 *           |
 *         store
 *           |
 *        output0
 */
static void CreateSignBitStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load load("load0");
  load.x = x.y;

  af::ascir_op::SignBit op("signbit");
  op.x = load.y;
  op.y.dtype = af::DT_UINT8;

  af::ascir_op::Store store("store");
  store.x = op.y;
  store.y.dtype = af::DT_UINT8;

  af::ascir_op::Output y("output");
  y.x = store.y;
  y.y.dtype = af::DT_UINT8;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::SignBitStoreFusedGraph(size_t dims_size) {
  return CreateOneInputAscGraphComputeGraph("signbit_store_test", "signbit_store", dims_size,
                                            CreateSignBitStoreAscGraph);
}

/**
 *         data0
 *           |
 *         load0
 *           |
 *       Frexp
 *         /    \
 *     store0  store1
 *         |      |
 *    output0  output1
 */
static void CreateFrexpStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load load("load0");
  load.x = x.y;

  af::ascir_op::Frexp op("frexp");
  op.x = load.y;
  op.exponent.dtype = af::DT_INT32;

  af::ascir_op::Store store0("store0");
  store0.x = op.mantissa;

  af::ascir_op::Store store1("store1");
  store1.x = op.exponent;
  store1.y.dtype = af::DT_INT32;

  af::ascir_op::Output y0("output0");
  y0.x = store0.y;
  y0.ir_attr.SetIndex(0);

  af::ascir_op::Output y1("output1");
  y1.x = store1.y;
  y1.y.dtype = af::DT_INT32;
  y1.ir_attr.SetIndex(1);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::FrexpStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("frexp_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 2);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 2, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  builder.AddDataEdge(ascbc, 1, netoutput, 1);
  auto compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("frexp_store");
  CreateFrexpStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *         data0
 *           |
 *         load0
 *           |
 *  ShiftedChebyshevPolynomialT (attr: n=3)
 *           |
 *         store
 *           |
 *        output0
 */
static void CreateShiftedChebyshevPolynomialTStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load load("load0");
  load.x = x.y;

  af::ascir_op::ShiftedChebyshevPolynomialT op("shifted_chebyshev_polynomial_t");
  op.x = load.y;
  op.ir_attr.SetN(3);

  af::ascir_op::Store store("store");
  store.x = op.y;

  af::ascir_op::Output y("output");
  y.x = store.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::ShiftedChebyshevPolynomialTStoreFusedGraph(size_t dims_size) {
  return CreateOneInputAscGraphComputeGraph("shifted_chebyshev_polynomial_t_store_test",
                                            "shifted_chebyshev_polynomial_t_store", dims_size,
                                            CreateShiftedChebyshevPolynomialTStoreAscGraph);
}

template <typename Op>
static void CreateShiftedChebyshevStoreAscGraph(af::AscGraph &graph, size_t dims_size, const char *op_name) {
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load load("load0");
  load.x = x.y;

  Op op(op_name);
  op.x = load.y;
  op.ir_attr.SetN(3);

  af::ascir_op::Store store("store");
  store.x = op.y;

  af::ascir_op::Output y("output");
  y.x = store.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

static void CreateShiftedChebyshevPolynomialUStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  CreateShiftedChebyshevStoreAscGraph<af::ascir_op::ShiftedChebyshevPolynomialU>(graph, dims_size,
                                                                                 "shifted_chebyshev_polynomial_u");
}

af::ComputeGraphPtr ShareGraph::ShiftedChebyshevPolynomialUStoreFusedGraph(size_t dims_size) {
  return CreateOneInputAscGraphComputeGraph("shifted_chebyshev_polynomial_u_store_test",
                                            "shifted_chebyshev_polynomial_u_store", dims_size,
                                            CreateShiftedChebyshevPolynomialUStoreAscGraph);
}

static void CreateShiftedChebyshevPolynomialVStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  CreateShiftedChebyshevStoreAscGraph<af::ascir_op::ShiftedChebyshevPolynomialV>(graph, dims_size,
                                                                                 "shifted_chebyshev_polynomial_v");
}

af::ComputeGraphPtr ShareGraph::ShiftedChebyshevPolynomialVStoreFusedGraph(size_t dims_size) {
  return CreateOneInputAscGraphComputeGraph("shifted_chebyshev_polynomial_v_store_test",
                                            "shifted_chebyshev_polynomial_v_store", dims_size,
                                            CreateShiftedChebyshevPolynomialVStoreAscGraph);
}

static void CreateShiftedChebyshevPolynomialWStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  CreateShiftedChebyshevStoreAscGraph<af::ascir_op::ShiftedChebyshevPolynomialW>(graph, dims_size,
                                                                                 "shifted_chebyshev_polynomial_w");
}

af::ComputeGraphPtr ShareGraph::ShiftedChebyshevPolynomialWStoreFusedGraph(size_t dims_size) {
  return CreateOneInputAscGraphComputeGraph("shifted_chebyshev_polynomial_w_store_test",
                                            "shifted_chebyshev_polynomial_w_store", dims_size,
                                            CreateShiftedChebyshevPolynomialWStoreAscGraph);
}

static void CreateChebyshevPolynomialTStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load load("load0");
  load.x = x.y;

  af::ascir_op::ChebyshevPolynomialT op("chebyshev_polynomial_t");
  op.x = load.y;
  op.ir_attr.SetN(3);

  af::ascir_op::Store store("store");
  store.x = op.y;

  af::ascir_op::Output y("output");
  y.x = store.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::ChebyshevPolynomialTStoreFusedGraph(size_t dims_size) {
  return CreateOneInputAscGraphComputeGraph("chebyshev_polynomial_t_store_test", "chebyshev_polynomial_t_store",
                                            dims_size, CreateChebyshevPolynomialTStoreAscGraph);
}

static void CreateChebyshevPolynomialUStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load load("load0");
  load.x = x.y;

  af::ascir_op::ChebyshevPolynomialU op("chebyshev_polynomial_u");
  op.x = load.y;
  op.ir_attr.SetN(3);

  af::ascir_op::Store store("store");
  store.x = op.y;

  af::ascir_op::Output y("output");
  y.x = store.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::ChebyshevPolynomialUStoreFusedGraph(size_t dims_size) {
  return CreateOneInputAscGraphComputeGraph("chebyshev_polynomial_u_store_test", "chebyshev_polynomial_u_store",
                                            dims_size, CreateChebyshevPolynomialUStoreAscGraph);
}

static void CreateChebyshevPolynomialVStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load load("load0");
  load.x = x.y;

  af::ascir_op::ChebyshevPolynomialV op("chebyshev_polynomial_v");
  op.x = load.y;
  op.ir_attr.SetN(3);

  af::ascir_op::Store store("store");
  store.x = op.y;

  af::ascir_op::Output y("output");
  y.x = store.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::ChebyshevPolynomialVStoreFusedGraph(size_t dims_size) {
  return CreateOneInputAscGraphComputeGraph("chebyshev_polynomial_v_store_test", "chebyshev_polynomial_v_store",
                                            dims_size, CreateChebyshevPolynomialVStoreAscGraph);
}

static void CreateChebyshevPolynomialWStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load load("load0");
  load.x = x.y;

  af::ascir_op::ChebyshevPolynomialW op("chebyshev_polynomial_w");
  op.x = load.y;
  op.ir_attr.SetN(3);

  af::ascir_op::Store store("store");
  store.x = op.y;

  af::ascir_op::Output y("output");
  y.x = store.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::ChebyshevPolynomialWStoreFusedGraph(size_t dims_size) {
  return CreateOneInputAscGraphComputeGraph("chebyshev_polynomial_w_store_test", "chebyshev_polynomial_w_store",
                                            dims_size, CreateChebyshevPolynomialWStoreAscGraph);
}

/**
 *      data0    data1
 *        |        |
 *      load0    load1
 *         \      /
 *   HermitePolynomialH
 *          |
 *        store
 *          |
 *       output0
 */
static void CreateHermitePolynomialHStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);

  af::ascir_op::Data n("data1", graph);
  n.ir_attr.SetIndex(1);
  n.y.dtype = af::DT_INT32;

  af::ascir_op::Load load_x("load0");
  load_x.x = x.y;

  af::ascir_op::Load load_n("load1");
  load_n.x = n.y;
  load_n.y.dtype = af::DT_INT32;

  af::ascir_op::HermitePolynomialH op("hermite_polynomial_h");
  op.x = load_x.y;
  op.n = load_n.y;

  af::ascir_op::Store store("store");
  store.x = op.y;

  af::ascir_op::Output y("output");
  y.x = store.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::HermitePolynomialHStoreFusedGraph(size_t dims_size) {
  return CreateTwoInputAscGraphComputeGraph("hermite_polynomial_h_store_test", "hermite_polynomial_h_store", dims_size,
                                            CreateHermitePolynomialHStoreAscGraph);
}

/**
 *      data0    data1
 *        |        |
 *      load0    load1
 *         \      /
 *   HermitePolynomialHe
 *          |
 *        store
 *          |
 *       output0
 */
static void CreateHermitePolynomialHeStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);

  af::ascir_op::Data n("data1", graph);
  n.ir_attr.SetIndex(1);
  n.y.dtype = af::DT_INT32;

  af::ascir_op::Load load_x("load0");
  load_x.x = x.y;

  af::ascir_op::Load load_n("load1");
  load_n.x = n.y;
  load_n.y.dtype = af::DT_INT32;

  af::ascir_op::HermitePolynomialHe op("hermite_polynomial_he");
  op.x = load_x.y;
  op.n = load_n.y;

  af::ascir_op::Store store("store");
  store.x = op.y;

  af::ascir_op::Output y("output");
  y.x = store.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::HermitePolynomialHeStoreFusedGraph(size_t dims_size) {
  return CreateTwoInputAscGraphComputeGraph("hermite_polynomial_he_store_test", "hermite_polynomial_he_store",
                                            dims_size, CreateHermitePolynomialHeStoreAscGraph);
}

/**
 *   data0  data1
 *     |      |
 *   load0  load1
 *     \    /
 *   Remainder
 *      |
 *    store
 *      |
 *   output0
 */
static void CreateRemainderInt32StoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = af::DT_INT32;

  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = af::DT_INT32;

  af::ascir_op::Load load_x1("load0");
  load_x1.x = x1.y;
  load_x1.y.dtype = af::DT_INT32;

  af::ascir_op::Load load_x2("load1");
  load_x2.x = x2.y;
  load_x2.y.dtype = af::DT_INT32;

  af::ascir_op::Remainder op("remainder");
  op.x1 = load_x1.y;
  op.x2 = load_x2.y;
  op.y.dtype = af::DT_INT32;

  af::ascir_op::Store store("store");
  store.x = op.y;
  store.y.dtype = af::DT_INT32;

  af::ascir_op::Output y("output");
  y.x = store.y;
  y.y.dtype = af::DT_INT32;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::RemainderInt32StoreFusedGraph(size_t dims_size) {
  return CreateTwoInputAscGraphComputeGraph("remainder_int32_store_test", "remainder_int32_store", dims_size,
                                            CreateRemainderInt32StoreAscGraph);
}

/**
 *   data0  data1
 *     |      |
 *   load0  load1
 *     \    /
 *   FloorDiv
 *      |
 *    store
 *      |
 *   output0
 */
static void CreateFloorDivInt32StoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data floor_div_x1("data0", graph);
  floor_div_x1.ir_attr.SetIndex(0);
  floor_div_x1.y.dtype = af::DT_INT32;

  af::ascir_op::Data floor_div_x2("data1", graph);
  floor_div_x2.ir_attr.SetIndex(1);
  floor_div_x2.y.dtype = af::DT_INT32;

  af::ascir_op::Load floor_div_load_x1("load0");
  floor_div_load_x1.x = floor_div_x1.y;
  floor_div_load_x1.y.dtype = af::DT_INT32;

  af::ascir_op::Load floor_div_load_x2("load1");
  floor_div_load_x2.x = floor_div_x2.y;
  floor_div_load_x2.y.dtype = af::DT_INT32;

  af::ascir_op::FloorDiv floor_div_op("floor_div");
  floor_div_op.x1 = floor_div_load_x1.y;
  floor_div_op.x2 = floor_div_load_x2.y;
  floor_div_op.y.dtype = af::DT_INT32;

  af::ascir_op::Store floor_div_store("store");
  floor_div_store.x = floor_div_op.y;
  floor_div_store.y.dtype = af::DT_INT32;

  af::ascir_op::Output floor_div_output("output");
  floor_div_output.x = floor_div_store.y;
  floor_div_output.y.dtype = af::DT_INT32;

  floor_div_output.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::FloorDivInt32StoreFusedGraph(size_t dims_size) {
  return CreateTwoInputAscGraphComputeGraph("floor_div_int32_store_test", "floor_div_int32_store", dims_size,
                                            CreateFloorDivInt32StoreAscGraph);
}

static void CreateSinBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data0", graph);
  x.y.dtype = af::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load x1Local("load0");
  x1Local.y.dtype = af::DataType::DT_BF16;
  x1Local.x = x.y;

  af::ascir_op::Sin sin("sin");
  sin.x = x1Local.y;
  sin.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Store sin_store("store");
  sin_store.x = sin.y;
  sin_store.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Output y("output");
  y.x = sin_store.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::BF16SinFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("sin_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("sin_bf16_test");
  CreateSinBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateSqrtBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data0", graph);
  x.y.dtype = af::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load x1Local("load0");
  x1Local.y.dtype = af::DataType::DT_BF16;
  x1Local.x = x.y;

  af::ascir_op::Sqrt sqrt("sqrt");
  sqrt.x = x1Local.y;
  sqrt.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Store sqrt_store("store");
  sqrt_store.x = sqrt.y;
  sqrt_store.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Output y("output");
  y.x = sqrt_store.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::BF16SqrtFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("sqrt_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("sqrt_bf16_test");
  CreateSqrtBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateRsqrtBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data0", graph);
  x.y.dtype = af::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load x1Local("load0");
  x1Local.y.dtype = af::DataType::DT_BF16;
  x1Local.x = x.y;

  af::ascir_op::Rsqrt rsqrt("rsqrt");
  rsqrt.x = x1Local.y;
  rsqrt.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Store rsqrt_store("store");
  rsqrt_store.x = rsqrt.y;
  rsqrt_store.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Output y("output");
  y.x = rsqrt_store.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::BF16RsqrtFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("rsqrt_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("rsqrt_bf16_test");
  CreateRsqrtBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateSigmoidBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data0", graph);
  x.y.dtype = af::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load x1Local("load0");
  x1Local.y.dtype = af::DataType::DT_BF16;
  x1Local.x = x.y;

  af::ascir_op::Sigmoid sigmoid("sigmoid");
  sigmoid.x = x1Local.y;
  sigmoid.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Store sigmoid_store("store");
  sigmoid_store.x = sigmoid.y;
  sigmoid_store.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Output y("output");
  y.x = sigmoid_store.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::BF16SigmoidFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("sigmoid_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("sigmoid_bf16_test");
  CreateSigmoidBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::BrcReduceFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("brc_reduce_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("brc_reduce");
  CreateBrcReduceAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateTailBrcTailReduceAscGraph(af::AscGraph &graph, size_t dims_size) {
  const Expression s0 = graph.CreateSizeVar(4);
  const Expression s1 = graph.CreateSizeVar(8);
  const Expression s2 = graph.CreateSizeVar(7);
  auto One = Symbol(1);
  auto Zero = Symbol(0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load load1("load1");
  load1.x = x.y;
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.attr.api.type = af::ApiType::kAPITypeCompute;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, One};
  *load1.y.strides = {s1, One, Zero};
  load1.y.dtype = af::DataType::DT_FLOAT;
  load1.attr.api.unit = ComputeUnit::kUnitMTE2;

  af::ascir_op::Abs abs("abs");
  abs.x = load1.y;
  abs.attr.api.compute_type = ComputeType::kComputeElewise;
  abs.attr.api.type = af::ApiType::kAPITypeCompute;
  abs.attr.sched.axis = {z0.id, z1.id, z2.id};
  *abs.y.axis = {z0.id, z1.id, z2.id};
  *abs.y.repeats = {s0, s1, One};
  *abs.y.strides = {s1, One, Zero};
  abs.y.dtype = af::DataType::DT_FLOAT;
  abs.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Broadcast brc0("brc0");
  brc0.x = abs.y;
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.attr.api.type = af::ApiType::kAPITypeCompute;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *brc0.y.axis = {z0.id, z1.id, z2.id};
  *brc0.y.repeats = {s0, s1, s2};
  *brc0.y.strides = {s1 * s2, s2, One};
  brc0.y.dtype = af::DataType::DT_FLOAT;
  brc0.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Max max("max");
  graph.AddNode(max);
  max.x = brc0.y;
  max.attr.sched.axis = {z0.id, z1.id, z2.id};
  *max.y.axis = {z0.id, z1.id, z2.id};
  *max.y.repeats = {s0, s1, One};
  *max.y.strides = {s1, One, Zero};
  max.attr.tmp_buffers = {{{af::Symbol(8192), -1}, MemAttr(), 0}, {{af::Symbol(8192), 0}, MemAttr(), 1}};

  af::ascir_op::Store store1("store1");
  store1.x = max.y;
  store1.attr.api.compute_type = ComputeType::kComputeStore;
  store1.attr.api.type = af::ApiType::kAPITypeCompute;
  store1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *store1.y.axis = {z0.id, z1.id, z2.id};
  *store1.y.repeats = {s0, s1, One};
  *store1.y.strides = {s1, One, Zero};
  store1.y.dtype = af::DataType::DT_FLOAT;
  store1.attr.api.unit = ComputeUnit::kUnitMTE3;

  af::ascir_op::Output y1("y1");
  y1.x = store1.y;
  y1.attr.api.compute_type = ComputeType::kComputeInvalid;
  y1.attr.api.type = af::ApiType::kAPITypeBuffer;
  y1.y.dtype = af::DataType::DT_FLOAT;
  y1.ir_attr.SetIndex(0);
}

/**
 *      output
 *         |
 *       store
 *         |
 *      matmul
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateMatMulAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  af::ascir_op::MatMul matmul("matmul");
  matmul.x1 = x1Local.y;
  matmul.x2 = x2Local.y;
  matmul.ir_attr.SetTranspose_x1(1);
  matmul.ir_attr.SetTranspose_x2(0);
  matmul.ir_attr.SetHas_relu(0);
  matmul.ir_attr.SetEnable_hf32(0);
  matmul.ir_attr.SetOffset_x(0);

  af::ascir_op::Store x_out("store");
  x_out.x = matmul.y;
  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::TailBrcTailReduceFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("tail_brc_tail_reduce_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("tail_brc_tail_reduce");
  CreateTailBrcTailReduceAscGraph(sub_graph, dims_size);
  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateFloorDivMulLessEqualSelectAscGraph(af::AscGraph &graph, size_t dims_size) {
  const Expression s0 = graph.CreateSizeVar(2);
  const Expression s1 = graph.CreateSizeVar(8);
  const Expression s2 = graph.CreateSizeVar(8);
  auto One = Symbol(1);
  auto Zero = Symbol(0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data x0("data0", graph);
  x0.attr.api.compute_type = ComputeType::kComputeInvalid;
  x0.attr.api.type = af::ApiType::kAPITypeBuffer;
  x0.ir_attr.SetIndex(0);
  x0.y.dtype = af::DataType::DT_FLOAT;

  af::ascir_op::Load load0("load0");
  load0.x = x0.y;
  load0.attr.api.compute_type = ComputeType::kComputeLoad;
  load0.attr.api.type = af::ApiType::kAPITypeCompute;
  load0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load0.y.axis = {z0.id, z1.id, z2.id};
  *load0.y.repeats = {s0, s1, s2};
  *load0.y.strides = {s1 * s2, s2, One};
  load0.y.dtype = af::DataType::DT_FLOAT;
  load0.attr.api.unit = ComputeUnit::kUnitMTE2;

  af::ascir_op::Data x1("data1", graph);
  x1.attr.api.compute_type = ComputeType::kComputeInvalid;
  x1.attr.api.type = af::ApiType::kAPITypeBuffer;
  x1.ir_attr.SetIndex(1);
  x1.y.dtype = af::DataType::DT_FLOAT;

  af::ascir_op::Load load1("load1");
  load1.x = x1.y;
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.attr.api.type = af::ApiType::kAPITypeCompute;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {One, One, s2};
  *load1.y.strides = {Zero, Zero, One};
  load1.y.dtype = af::DataType::DT_FLOAT;
  load1.attr.api.unit = ComputeUnit::kUnitMTE2;

  af::ascir_op::Broadcast brc0("brc0");
  brc0.x = load1.y;
  brc0.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc0.attr.api.type = af::ApiType::kAPITypeCompute;
  brc0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *brc0.y.axis = {z0.id, z1.id, z2.id};
  *brc0.y.repeats = {One, s1, s2};
  *brc0.y.strides = {Zero, s2, One};
  brc0.y.dtype = af::DataType::DT_FLOAT;
  brc0.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Broadcast brc1("brc1");
  brc1.x = brc0.y;
  brc1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc1.attr.api.type = af::ApiType::kAPITypeCompute;
  brc1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *brc1.y.axis = {z0.id, z1.id, z2.id};
  *brc1.y.repeats = {s0, s1, s2};
  *brc1.y.strides = {s0 * s1, s2, One};
  brc1.y.dtype = af::DataType::DT_FLOAT;
  brc1.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::FloorDiv floor_div0("floor_div0");
  floor_div0.x1 = load0.y;
  floor_div0.x2 = brc1.y;
  floor_div0.attr.api.compute_type = ComputeType::kComputeElewise;
  floor_div0.attr.api.type = af::ApiType::kAPITypeCompute;
  floor_div0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *floor_div0.y.axis = {z0.id, z1.id, z2.id};
  *floor_div0.y.repeats = {s0, s1, s2};
  *floor_div0.y.strides = {s0 * s1, s2, One};
  floor_div0.y.dtype = af::DataType::DT_FLOAT;
  floor_div0.attr.api.unit = ComputeUnit::kUnitMTE2;

  af::ascir_op::Data x2("data2", graph);
  x2.attr.api.compute_type = ComputeType::kComputeInvalid;
  x2.attr.api.type = af::ApiType::kAPITypeBuffer;
  x2.ir_attr.SetIndex(2);
  x2.y.dtype = af::DataType::DT_FLOAT;

  af::ascir_op::Load load2("load2");
  load2.x = x2.y;
  load2.attr.api.compute_type = ComputeType::kComputeLoad;
  load2.attr.api.type = af::ApiType::kAPITypeCompute;
  load2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load2.y.axis = {z0.id, z1.id, z2.id};
  *load2.y.repeats = {One, One, One};
  *load2.y.strides = {Zero, Zero, Zero};
  load2.y.dtype = af::DataType::DT_FLOAT;
  load2.attr.api.unit = ComputeUnit::kUnitMTE2;

  af::ascir_op::Exp exp0("exp0");
  exp0.x = load2.y;
  exp0.attr.api.compute_type = ComputeType::kComputeLoad;
  exp0.attr.api.type = af::ApiType::kAPITypeCompute;
  exp0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *exp0.y.axis = {z0.id, z1.id, z2.id};
  *exp0.y.repeats = {One, One, One};
  *exp0.y.strides = {Zero, Zero, Zero};
  exp0.y.dtype = af::DataType::DT_FLOAT;
  exp0.attr.api.unit = ComputeUnit::kUnitMTE2;

  af::ascir_op::Broadcast brc2("brc2");
  brc2.x = exp0.y;
  brc2.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc2.attr.api.type = af::ApiType::kAPITypeCompute;
  brc2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *brc2.y.axis = {z0.id, z1.id, z2.id};
  *brc2.y.repeats = {One, One, s2};
  *brc2.y.strides = {Zero, Zero, One};
  brc2.y.dtype = af::DataType::DT_FLOAT;
  brc2.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Broadcast brc3("brc3");
  brc3.x = brc2.y;
  brc3.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc3.attr.api.type = af::ApiType::kAPITypeCompute;
  brc3.attr.sched.axis = {z0.id, z1.id, z2.id};
  *brc3.y.axis = {z0.id, z1.id, z2.id};
  *brc3.y.repeats = {One, s1, s2};
  *brc3.y.strides = {Zero, s2, One};
  brc3.y.dtype = af::DataType::DT_FLOAT;
  brc3.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Broadcast brc4("brc4");
  brc4.x = brc3.y;
  brc4.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc4.attr.api.type = af::ApiType::kAPITypeCompute;
  brc4.attr.sched.axis = {z0.id, z1.id, z2.id};
  *brc4.y.axis = {z0.id, z1.id, z2.id};
  *brc4.y.repeats = {s0, s1, s2};
  *brc4.y.strides = {s1 * s2, s2, One};
  brc4.y.dtype = af::DataType::DT_FLOAT;
  brc4.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Mul mul("mul");
  mul.x1 = brc4.y;
  mul.x2 = floor_div0.y;
  mul.attr.api.compute_type = ComputeType::kComputeElewise;
  mul.attr.api.type = af::ApiType::kAPITypeCompute;
  mul.attr.sched.axis = {z0.id, z1.id, z2.id};
  *mul.y.axis = {z0.id, z1.id, z2.id};
  *mul.y.repeats = {s0, s1, s2};
  *mul.y.strides = {s1 * s2, s1, One};
  mul.y.dtype = af::DataType::DT_FLOAT;
  mul.attr.api.unit = ComputeUnit::kUnitMTE2;

  af::ascir_op::Scalar scalar("scalar", graph);
  scalar.ir_attr.SetIndex(3);
  scalar.ir_attr.SetValue("1.0");

  af::ascir_op::Broadcast brc5("brc5");
  brc5.x = scalar.y;
  brc5.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc5.attr.api.type = af::ApiType::kAPITypeCompute;
  brc5.attr.sched.axis = {z0.id, z1.id, z2.id};
  *brc5.y.axis = {z0.id, z1.id, z2.id};
  *brc5.y.repeats = {One, One, s2};
  *brc5.y.strides = {Zero, Zero, One};
  brc5.y.dtype = af::DataType::DT_FLOAT;
  brc5.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Broadcast brc6("brc6");
  brc6.x = brc5.y;
  brc6.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc6.attr.api.type = af::ApiType::kAPITypeCompute;
  brc6.attr.sched.axis = {z0.id, z1.id, z2.id};
  *brc6.y.axis = {z0.id, z1.id, z2.id};
  *brc6.y.repeats = {One, s1, s2};
  *brc6.y.strides = {Zero, s2, One};
  brc6.y.dtype = af::DataType::DT_FLOAT;
  brc6.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Broadcast brc7("brc7");
  brc7.x = brc6.y;
  brc7.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc7.attr.api.type = af::ApiType::kAPITypeCompute;
  brc7.attr.sched.axis = {z0.id, z1.id, z2.id};
  *brc7.y.axis = {z0.id, z1.id, z2.id};
  *brc7.y.repeats = {s0, s1, s2};
  *brc7.y.strides = {s1 * s2, s2, One};
  brc7.y.dtype = af::DataType::DT_FLOAT;
  brc7.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Le le0("le0");
  le0.x1 = mul.y;
  le0.x2 = brc7.y;
  le0.attr.api.compute_type = ComputeType::kComputeInvalid;
  le0.attr.api.type = af::ApiType::kAPITypeInvalid;
  le0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *le0.y.axis = {z0.id, z1.id, z2.id};
  *le0.y.repeats = {s0, s1, s2};
  *le0.y.strides = {s1 * s2, s2, One};
  le0.y.dtype = af::DataType::DT_UINT8;
  le0.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Where where0("where0");
  where0.x1 = le0.y;
  where0.x2 = floor_div0.y;
  where0.x3 = load0.y;
  where0.attr.api.compute_type = ComputeType::kComputeInvalid;
  where0.attr.api.type = af::ApiType::kAPITypeInvalid;
  where0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *where0.y.axis = {z0.id, z1.id, z2.id};
  *where0.y.repeats = {s0, s1, s2};
  *where0.y.strides = {s1 * s2, s2, One};
  where0.y.dtype = af::DataType::DT_FLOAT;
  where0.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Store store("store");
  store.x = where0.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.attr.api.type = af::ApiType::kAPITypeCompute;
  store.attr.sched.axis = {z0.id, z1.id, z2.id};
  *store.y.axis = {z0.id, z1.id, z2.id};
  *store.y.repeats = {s0, s1, s2};
  *store.y.strides = {s1 * s2, s1, One};
  store.y.dtype = af::DataType::DT_FLOAT;
  store.attr.api.unit = ComputeUnit::kUnitMTE3;

  af::ascir_op::Output y("output");
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = af::DataType::DT_FLOAT;
  y.ir_attr.SetIndex(0);
}

af::ComputeGraphPtr ShareGraph::MatMulFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("matmul_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("matmul");
  CreateMatMulAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

af::ComputeGraphPtr ShareGraph::FloorDivMulLessEqualSelectFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("floordiv_mul_le_select_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  af::AttrUtils::SetInt(data2->GetOpDescBarePtr(), "_parent_node_index", 2);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 3, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(data2, 0, ascbc, 2);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("floordiv_mul_le_select");
  CreateFloorDivMulLessEqualSelectAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *        Pow
 *     /      \
 *   brc1    brc2
 *    |       |
 *  load1  load2
 *    |      |
 * scalar0  scalar1
 */
static void CreateLoadPowAllInputIsScalarStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  const Expression s0 = graph.CreateSizeVar(2);
  const Expression s1 = graph.CreateSizeVar(8);
  auto One = Symbol(1);
  auto Zero = Symbol(0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);

  af::ascir_op::Data x1("scalar1", graph);
  x1.ir_attr.SetIndex(0);

  af::ascir_op::Load load1("load1");
  load1.x = x1.y;
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.attr.api.type = af::ApiType::kAPITypeCompute;
  load1.attr.sched.axis = {z0.id, z1.id};
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {One, One};
  *load1.y.strides = {Zero, Zero};
  load1.y.dtype = af::DataType::DT_FLOAT;
  load1.attr.api.unit = ComputeUnit::kUnitMTE2;

  af::ascir_op::Broadcast brc1("brc1");
  brc1.x = load1.y;
  brc1.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc1.attr.api.type = af::ApiType::kAPITypeCompute;
  brc1.attr.sched.axis = {z0.id, z1.id};
  *brc1.y.axis = {z0.id, z1.id};
  *brc1.y.repeats = {s0, s1};
  *brc1.y.strides = {s1, One};
  brc1.y.dtype = af::DataType::DT_FLOAT;
  brc1.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Data x2("scalar2", graph);
  x2.ir_attr.SetIndex(1);

  af::ascir_op::Load load2("load2");
  load2.x = x2.y;
  load2.attr.api.compute_type = ComputeType::kComputeLoad;
  load2.attr.api.type = af::ApiType::kAPITypeCompute;
  load2.attr.sched.axis = {z0.id, z1.id};
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.repeats = {One, One};
  *load2.y.strides = {Zero, Zero};
  load2.y.dtype = af::DataType::DT_FLOAT;
  load2.attr.api.unit = ComputeUnit::kUnitMTE2;

  af::ascir_op::Broadcast brc2("brc2");
  brc2.x = load2.y;
  brc2.attr.api.compute_type = ComputeType::kComputeBroadcast;
  brc2.attr.api.type = af::ApiType::kAPITypeCompute;
  brc2.attr.sched.axis = {z0.id, z1.id};
  *brc2.y.axis = {z0.id, z1.id};
  *brc2.y.repeats = {s0, s1};
  *brc2.y.strides = {s1, One};
  brc2.y.dtype = af::DataType::DT_FLOAT;
  brc2.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Pow pow("pow");
  pow.x1 = brc1.y;
  pow.x2 = brc2.y;
  pow.attr.api.compute_type = ComputeType::kComputeElewise;
  pow.attr.api.type = af::ApiType::kAPITypeCompute;
  pow.attr.sched.axis = {z0.id, z1.id};
  *pow.y.axis = {z0.id, z1.id};
  *pow.y.repeats = {s0, s1};
  *pow.y.strides = {s1, One};
  pow.y.dtype = af::DataType::DT_FLOAT;
  pow.attr.api.unit = ComputeUnit::kUnitVector;

  af::ascir_op::Store store("store");
  store.x = pow.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.attr.api.type = af::ApiType::kAPITypeCompute;
  store.attr.sched.axis = {z0.id, z1.id};
  *store.y.axis = {z0.id, z1.id};
  *store.y.repeats = {s0, s1};
  *store.y.strides = {s1, One};
  store.y.dtype = af::DataType::DT_FLOAT;
  store.attr.api.unit = ComputeUnit::kUnitMTE3;

  af::ascir_op::Output y("output");
  y.x = store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = af::DataType::DT_FLOAT;
  y.ir_attr.SetIndex(0);
}

/**
 *      NetOutput
 *         |
 *        AscBc
 *      /     \
 *  data0    data1
 */
af::ComputeGraphPtr ShareGraph::LoadPowAllInputIsScalarStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_pow_all_input_is_scalar_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_pow_all_input_is_scalar_store");
  CreateLoadPowAllInputIsScalarStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

af::ComputeGraphPtr ShareGraph::AddAbsFusedConstGraph(size_t dims_size, std::vector<int> dims) {
  auto builder = GraphBuilder("add_abs_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("add_abs");
  CreateAddAbsConstAscGraph(sub_graph, dims_size, dims);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

void ShareGraph::ConcatAscGraph(AscGraph &graph, const vector<std::string> &dim_sizes, bool align) {
  af::ascir_op::Data x1("concat_data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = af::DT_INT32;

  af::ascir_op::Data x2("concat_data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = af::DT_INT32;

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = af::DT_INT32;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = af::DT_INT32;

  af::ascir_op::Concat concat("concat");
  concat.x = {x1Local.y, x2Local.y};
  concat.y.dtype = af::DT_INT32;

  af::ascir_op::Store x_out("store");
  x_out.x = concat.y;
  x_out.y.dtype = af::DT_INT32;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = af::DT_INT32;

  ConstructVVAscGraphAxisInfo(graph, dim_sizes, align);
  auto concat_node = graph.FindNode("concat");
  auto size = concat_node->attr.sched.axis.size();
  auto repeats = concat_node->outputs()[0]->attr.repeats;
  repeats[size - 1] = repeats[size - 1] + repeats[size - 1];
  auto strides = concat_node->outputs()[0]->attr.strides;
  for (int i = dim_sizes.size() - 2; i >= 0; i--) {
    strides[i] = af::sym::Mul(repeats[i + 1], strides[i + 1]);
  }
  concat_node->outputs()[0]->attr.strides = strides;
  concat_node->outputs()[0]->attr.repeats = repeats;
  auto store_node = graph.FindNode("store");
  store_node->outputs()[0]->attr.strides = strides;
  store_node->outputs()[0]->attr.repeats = repeats;
}
static void LoadGatherAbsStore_BeforeAutofuse(af::AscGraph &graph, int64_t gather_axis, af::DataType data_type) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");
  auto s4 = graph.CreateSizeVar("s4");
  auto s5 = graph.CreateSizeVar("s5");
  auto s6 = graph.CreateSizeVar("s6");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  auto z4 = graph.CreateAxis("z4", s4);
  auto z5 = graph.CreateAxis("z5", s5);
  auto z6 = graph.CreateAxis("z6", s6);

  af::ascir_op::Data x1("x1");
  graph.AddNode(x1);
  x1.y.dtype = data_type;
  x1.attr.sched.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *x1.y.axis = {z0.id, z1.id, z2.id, z3.id, z4.id};
  *x1.y.repeats = {s0, s1, s2, s3, s4};
  *x1.y.strides = {s1 * s2 * s3 * s4, s2 * s3 * s4, s3 * s4, s4, af::ops::One};
  x1.ir_attr.SetIndex(0);

  af::ascir_op::Data x2("x2");
  graph.AddNode(x2);
  x2.y.dtype = af::DT_INT32;
  x2.attr.sched.axis = {z5.id, z6.id};
  *x2.y.axis = {z5.id, z6.id};
  *x2.y.repeats = {s5, s6};
  *x2.y.strides = {s6, af::ops::One};
  x2.ir_attr.SetIndex(1);

  af::ascir_op::Gather gather("gather");
  graph.AddNode(gather);
  gather.x1 = x1.y;
  gather.x2 = x2.y;
  gather.ir_attr.SetAxis(gather_axis);
  gather.ir_attr.SetNegative_index_support(false);
  gather.attr.sched.axis = {z0.id, z1.id, z5.id, z6.id, z3.id, z4.id};
  *gather.y.axis = {z0.id, z1.id, z5.id, z6.id, z3.id, z4.id};
  *gather.y.repeats = {s0, s1, s5, s6, s3, s4};
  *gather.y.strides = {s1 * s5 * s6 * s3 * s4, s5 * s6 * s3 * s4, s6 * s3 * s4, s3 * s4, s4, af::ops::One};

  af::ascir_op::Abs abs("abs");
  graph.AddNode(abs);
  abs.x = gather.y;
  abs.attr.sched.axis = {z0.id, z1.id, z5.id, z6.id, z3.id, z4.id};
  *abs.y.axis = {z0.id, z1.id, z5.id, z6.id, z3.id, z4.id};
  *abs.y.repeats = {s0, s1, s5, s6, s3, s4};
  *abs.y.strides = {s1 * s5 * s6 * s3 * s4, s5 * s6 * s3 * s4, s6 * s3 * s4, s3 * s4, s4, af::ops::One};

  af::ascir_op::Store store("store");
  graph.AddNode(store);
  store.x = abs.y;
  store.attr.sched.axis = {z0.id, z1.id, z5.id, z6.id, z3.id, z4.id};
  *store.y.axis = {z0.id, z1.id, z5.id, z6.id, z3.id, z4.id};
  *store.y.repeats = {s0, s1, s5, s6, s3, s4};
  *store.y.strides = {s1 * s5 * s6 * s3 * s4, s5 * s6 * s3 * s4, s6 * s3 * s4, s3 * s4, s4, af::ops::One};

  af::ascir_op::Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  y.y.dtype = data_type;
  y.ir_attr.SetIndex(0);
}

af::ComputeGraphPtr ShareGraph::LoadGatherAbsStore(int64_t gather_axis, af::DataType data_type) {
  auto builder = GraphBuilder("load_gather_abs_store_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_gather_abs_store_store_test");
  LoadGatherAbsStore_BeforeAutofuse(sub_graph, gather_axis, data_type);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void LoadGatherTailAbsStore_BeforeAutofuse(af::AscGraph &graph, int64_t gather_axis, af::DataType data_type) {
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

  af::ascir_op::Data x1("x1");
  graph.AddNode(x1);
  x1.y.dtype = data_type;
  x1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x1.y.axis = {z0.id, z1.id, z2.id};
  *x1.y.repeats = {s0, s1, s2};
  *x1.y.strides = {s1 * s2, s2, af::ops::One};
  x1.ir_attr.SetIndex(0);

  af::ascir_op::Data x2("x2");
  graph.AddNode(x2);
  x2.y.dtype = af::DT_INT32;
  x2.attr.sched.axis = {z3.id, z4.id};
  *x2.y.axis = {z3.id, z4.id};
  *x2.y.repeats = {s3, s4};
  *x2.y.strides = {s4, af::ops::One};
  x2.ir_attr.SetIndex(1);

  af::ascir_op::Gather gather("gather");
  graph.AddNode(gather);
  gather.x1 = x1.y;
  gather.x2 = x2.y;
  gather.ir_attr.SetAxis(gather_axis);
  gather.ir_attr.SetNegative_index_support(false);
  gather.attr.sched.axis = {z0.id, z1.id, z3.id, z4.id};
  *gather.y.axis = {z0.id, z1.id, z3.id, z4.id};
  *gather.y.repeats = {s0, s1, s3, s4};
  *gather.y.strides = {s1 * s3 * s4, s3 * s4, s4, af::ops::One};

  af::ascir_op::Abs abs("abs");
  graph.AddNode(abs);
  abs.x = gather.y;
  abs.attr.sched.axis = {z0.id, z1.id, z3.id, z4.id};
  *abs.y.axis = {z0.id, z1.id, z3.id, z4.id};
  *abs.y.repeats = {s0, s1, s3, s4};
  *abs.y.strides = {s1 * s3 * s4, s3 * s4, s4, af::ops::One};

  af::ascir_op::Store store("store");
  graph.AddNode(store);
  store.x = abs.y;
  store.attr.sched.axis = {z0.id, z1.id, z3.id, z4.id};
  *store.y.axis = {z0.id, z1.id, z3.id, z4.id};
  *store.y.repeats = {s0, s1, s3, s4};
  *store.y.strides = {s1 * s3 * s4, s3 * s4, s4, af::ops::One};

  af::ascir_op::Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  y.y.dtype = data_type;
  y.ir_attr.SetIndex(0);
}

af::ComputeGraphPtr ShareGraph::LoadGatherTailAbsStore(int64_t gather_axis, af::DataType data_type) {
  auto builder = GraphBuilder("load_gather_tail_abs_store_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_gather_tail_abs_store_store_test");
  LoadGatherTailAbsStore_BeforeAutofuse(sub_graph, gather_axis, data_type);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

template <typename Op>
void SetIndirectLoadNodeView(Op &op, const std::vector<int64_t> &axes, const std::vector<af::Expression> &repeats,
                             const std::vector<af::Expression> &strides, af::DataType data_type) {
  op.attr.sched.axis = axes;
  op.y.dtype = data_type;
  *op.y.axis = axes;
  *op.y.repeats = repeats;
  *op.y.strides = strides;
}

static void CreateIndirectLoadAxes(af::AscGraph &graph, size_t rank, size_t offset, std::vector<int64_t> &axes,
                                   std::vector<af::Expression> &repeats, std::vector<af::Expression> &strides,
                                   const std::vector<int64_t> &shape) {
  for (size_t i = 0UL; i < rank; ++i) {
    const size_t dim = offset + i;
    repeats.emplace_back(shape.empty() ? graph.CreateSizeVar("s" + std::to_string(dim))
                                       : graph.CreateSizeVar(shape[i]));
    axes.emplace_back(graph.CreateAxis("z" + std::to_string(dim), repeats.back()).id);
  }
  strides.assign(rank, af::ops::One);
  for (size_t i = rank - 1UL; i > 0UL; --i) {
    strides[i - 1UL] = repeats[i] * strides[i];
  }
}

static void AddIndirectLoadPrecisionCastDataChain(af::AscGraph &graph, const std::vector<int64_t> &axes,
                                                  const std::vector<af::Expression> &repeats,
                                                  const std::vector<af::Expression> &strides,
                                                  af::ascir_op::IndirectLoad &indirect_load) {
  af::ascir_op::Data x("x");
  graph.AddNode(x);
  x.y.dtype = af::DT_FLOAT16;
  x.ir_attr.SetIndex(0);
  af::ascir_op::Load input_load("input_load");
  graph.AddNode(input_load);
  input_load.x = x.y;
  SetIndirectLoadNodeView(input_load, axes, repeats, strides, af::DT_FLOAT16);
  af::ascir_op::Cast input_cast("input_cast");
  graph.AddNode(input_cast);
  input_cast.x = input_load.y;
  SetIndirectLoadNodeView(input_cast, axes, repeats, strides, af::DT_FLOAT);
  indirect_load.x1 = input_cast.y;
}

static void AddIndirectLoadDataChain(af::AscGraph &graph, const std::vector<int64_t> &axes,
                                     const std::vector<af::Expression> &repeats,
                                     const std::vector<af::Expression> &strides, af::DataType data_type,
                                     IndirectLoadInputPreType input_pre_type,
                                     af::ascir_op::IndirectLoad &indirect_load) {
  af::ascir_op::Data x("x");
  graph.AddNode(x);
  x.y.dtype = data_type;
  x.ir_attr.SetIndex(0);
  af::ascir_op::Load input_load("input_load");
  graph.AddNode(input_load);
  input_load.x = x.y;
  SetIndirectLoadNodeView(input_load, axes, repeats, strides, data_type);
  if (input_pre_type == IndirectLoadInputPreType::kNone) {
    indirect_load.x1 = input_load.y;
    return;
  }
  if (input_pre_type == IndirectLoadInputPreType::kExp2) {
    af::ascir_op::Exp2 input_exp2("input_exp2");
    graph.AddNode(input_exp2);
    input_exp2.x = input_load.y;
    SetIndirectLoadNodeView(input_exp2, axes, repeats, strides, data_type);
    indirect_load.x1 = input_exp2.y;
    return;
  }
  af::ascir_op::Relu input_relu("input_relu");
  graph.AddNode(input_relu);
  input_relu.x = input_load.y;
  SetIndirectLoadNodeView(input_relu, axes, repeats, strides, data_type);
  if (input_pre_type == IndirectLoadInputPreType::kRelu) {
    indirect_load.x1 = input_relu.y;
    return;
  }
  if (input_pre_type == IndirectLoadInputPreType::kReluRelu) {
    af::ascir_op::Relu input_relu2("input_relu2");
    graph.AddNode(input_relu2);
    input_relu2.x = input_relu.y;
    SetIndirectLoadNodeView(input_relu2, axes, repeats, strides, data_type);
    indirect_load.x1 = input_relu2.y;
    return;
  }
  af::ascir_op::Exp2 input_exp2("input_exp2");
  graph.AddNode(input_exp2);
  input_exp2.x = input_relu.y;
  SetIndirectLoadNodeView(input_exp2, axes, repeats, strides, data_type);
  indirect_load.x1 = input_exp2.y;
}

static void AddIndirectLoadIndexChain(af::AscGraph &graph, const std::vector<int64_t> &axes,
                                      const std::vector<af::Expression> &repeats,
                                      const std::vector<af::Expression> &strides, af::DataType index_type,
                                      bool mixed_index_pre, bool direct_index,
                                      af::ascir_op::IndirectLoad &indirect_load) {
  af::ascir_op::Data index("index");
  graph.AddNode(index);
  index.y.dtype = index_type;
  index.ir_attr.SetIndex(1);
  af::ascir_op::Load index_load("index_load");
  graph.AddNode(index_load);
  index_load.x = index.y;
  SetIndirectLoadNodeView(index_load, axes, repeats, strides, index_type);
  if (direct_index) {
    indirect_load.x2 = index_load.y;
    return;
  }
  af::ascir_op::Abs index_abs("index_abs");
  graph.AddNode(index_abs);
  index_abs.x = index_load.y;
  SetIndirectLoadNodeView(index_abs, axes, repeats, strides, index_type);
  if (!mixed_index_pre) {
    indirect_load.x2 = index_abs.y;
    return;
  }
  af::ascir_op::Abs index_abs2("index_abs2");
  graph.AddNode(index_abs2);
  index_abs2.x = index_abs.y;
  SetIndirectLoadNodeView(index_abs2, axes, repeats, strides, index_type);
  af::ascir_op::Cast index_cast("index_cast_float");
  graph.AddNode(index_cast);
  index_cast.x = index_abs2.y;
  SetIndirectLoadNodeView(index_cast, axes, repeats, strides, af::DT_FLOAT);
  af::ascir_op::Exp2 index_exp2("index_exp2");
  graph.AddNode(index_exp2);
  index_exp2.x = index_cast.y;
  SetIndirectLoadNodeView(index_exp2, axes, repeats, strides, af::DT_FLOAT);
  af::ascir_op::Log2 index_log2("index_log2");
  graph.AddNode(index_log2);
  index_log2.x = index_exp2.y;
  SetIndirectLoadNodeView(index_log2, axes, repeats, strides, af::DT_FLOAT);
  af::ascir_op::FloorToInt index_floor_to_int("index_floor_to_int");
  graph.AddNode(index_floor_to_int);
  index_floor_to_int.x = index_log2.y;
  SetIndirectLoadNodeView(index_floor_to_int, axes, repeats, strides, index_type);
  indirect_load.x2 = index_floor_to_int.y;
}

template <typename Op>
static void AddIndirectLoadCustomOutputChain(af::AscGraph &graph, const std::vector<int64_t> &axes,
                                             const std::vector<af::Expression> &repeats,
                                             const std::vector<af::Expression> &strides, af::DataType data_type,
                                             IndirectLoadOutputPostType output_post_type, Op &op) {
  af::AscOpOutput *output = &op.y;
  if (output_post_type == IndirectLoadOutputPostType::kModifiedBesselK0 ||
      output_post_type == IndirectLoadOutputPostType::kModifiedBesselK0Sum) {
    af::ascir_op::ModifiedBesselK0 modified_bessel_k0("output_modified_bessel_k0");
    graph.AddNode(modified_bessel_k0);
    modified_bessel_k0.x = *output;
    SetIndirectLoadNodeView(modified_bessel_k0, axes, repeats, strides, data_type);
    output = &modified_bessel_k0.y;
  }
  if (output_post_type == IndirectLoadOutputPostType::kExp2Sum) {
    af::ascir_op::Exp2 exp2("output_exp2");
    graph.AddNode(exp2);
    exp2.x = *output;
    SetIndirectLoadNodeView(exp2, axes, repeats, strides, data_type);
    output = &exp2.y;
  }
  if (output_post_type == IndirectLoadOutputPostType::kAbsExp2Sum) {
    af::ascir_op::Abs abs("output_abs");
    graph.AddNode(abs);
    abs.x = *output;
    SetIndirectLoadNodeView(abs, axes, repeats, strides, data_type);
    af::ascir_op::Exp2 exp2("output_exp2");
    graph.AddNode(exp2);
    exp2.x = abs.y;
    SetIndirectLoadNodeView(exp2, axes, repeats, strides, data_type);
    output = &exp2.y;
  }
  if (output_post_type == IndirectLoadOutputPostType::kCastSum) {
    af::ascir_op::Cast cast("output_cast");
    graph.AddNode(cast);
    cast.x = *output;
    SetIndirectLoadNodeView(cast, axes, repeats, strides, af::DT_FLOAT);
    output = &cast.y;
    data_type = af::DT_FLOAT;
  }
  if (output_post_type == IndirectLoadOutputPostType::kAddSum) {
    af::ascir_op::Data addend("output_addend");
    graph.AddNode(addend);
    addend.ir_attr.SetIndex(2);
    SetIndirectLoadNodeView(addend, axes, repeats, strides, data_type);
    af::ascir_op::Load addend_load("output_addend_load");
    graph.AddNode(addend_load);
    addend_load.x = addend.y;
    SetIndirectLoadNodeView(addend_load, axes, repeats, strides, data_type);
    af::ascir_op::Add add("output_add");
    graph.AddNode(add);
    add.x1 = *output;
    add.x2 = addend_load.y;
    SetIndirectLoadNodeView(add, axes, repeats, strides, data_type);
    output = &add.y;
  }
  if (output_post_type == IndirectLoadOutputPostType::kSum ||
      output_post_type == IndirectLoadOutputPostType::kSumKeepTail ||
      output_post_type == IndirectLoadOutputPostType::kSumLastAxis ||
      output_post_type == IndirectLoadOutputPostType::kExp2Sum ||
      output_post_type == IndirectLoadOutputPostType::kAbsExp2Sum ||
      output_post_type == IndirectLoadOutputPostType::kCastSum ||
      output_post_type == IndirectLoadOutputPostType::kAddSum ||
      output_post_type == IndirectLoadOutputPostType::kModifiedBesselK0Sum) {
    af::ascir_op::Sum sum("output_sum");
    graph.AddNode(sum);
    sum.x = *output;
    auto reduce_repeats = repeats;
    auto reduce_strides = strides;
    if (output_post_type == IndirectLoadOutputPostType::kSumLastAxis) {
      reduce_repeats[3] = af::ops::One;
      reduce_strides[0] = repeats[1] * repeats[2];
      reduce_strides[1] = repeats[2];
      reduce_strides[2] = af::ops::One;
      reduce_strides[3] = af::ops::Zero;
    } else if (output_post_type == IndirectLoadOutputPostType::kSumKeepTail) {
      reduce_repeats[2] = af::ops::One;
      reduce_strides[2] = af::ops::Zero;
      reduce_strides[0] = repeats[1] * repeats[3];
      reduce_strides[1] = repeats[3];
    } else {
      reduce_repeats[2] = af::ops::One;
      reduce_strides[2] = af::ops::Zero;
      reduce_repeats[3] = af::ops::One;
      reduce_strides[0] = repeats[1];
      reduce_strides[1] = af::ops::One;
      reduce_strides[3] = af::ops::Zero;
    }
    SetIndirectLoadNodeView(sum, axes, reduce_repeats, reduce_strides, data_type);

    af::ascir_op::Store store("store");
    graph.AddNode(store);
    store.x = sum.y;
    SetIndirectLoadNodeView(store, axes, reduce_repeats, reduce_strides, data_type);
    af::ascir_op::Output y("y");
    graph.AddNode(y);
    y.x = store.y;
    SetIndirectLoadNodeView(y, axes, reduce_repeats, reduce_strides, data_type);
    y.ir_attr.SetIndex(0);
    return;
  }
  af::ascir_op::Store store("store");
  graph.AddNode(store);
  store.x = *output;
  SetIndirectLoadNodeView(store, axes, repeats, strides, data_type);
  af::ascir_op::Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  SetIndirectLoadNodeView(y, axes, repeats, strides, data_type);
  y.ir_attr.SetIndex(0);
}

static void AddIndirectLoadPrecisionOutputChain(af::AscGraph &graph, const std::vector<int64_t> &axes,
                                                const std::vector<af::Expression> &repeats,
                                                const std::vector<af::Expression> &strides, bool use_exp2,
                                                af::ascir_op::IndirectLoad &indirect_load) {
  af::ascir_op::Exp exp("output_exp");
  af::ascir_op::Exp2 exp2("output_exp");
  af::AscOpOutput *math_output = nullptr;
  if (use_exp2) {
    graph.AddNode(exp2);
    exp2.x = indirect_load.y;
    SetIndirectLoadNodeView(exp2, axes, repeats, strides, af::DT_FLOAT);
    math_output = &exp2.y;
  } else {
    graph.AddNode(exp);
    exp.x = indirect_load.y;
    SetIndirectLoadNodeView(exp, axes, repeats, strides, af::DT_FLOAT);
    math_output = &exp.y;
  }
  af::ascir_op::Cast output_cast("output_cast");
  graph.AddNode(output_cast);
  output_cast.x = *math_output;
  SetIndirectLoadNodeView(output_cast, axes, repeats, strides, af::DT_FLOAT16);
  AddIndirectLoadCustomOutputChain(graph, axes, repeats, strides, af::DT_FLOAT16,
                                   IndirectLoadOutputPostType::kDirectStore, output_cast);
}

static void AddIndirectLoadUint32OutputChain(af::AscGraph &graph, const std::vector<int64_t> &axes,
                                             const std::vector<af::Expression> &repeats,
                                             const std::vector<af::Expression> &strides,
                                             af::ascir_op::IndirectLoad &indirect_load) {
  af::ascir_op::Cast cast_int32("output_cast_int32");
  graph.AddNode(cast_int32);
  cast_int32.x = indirect_load.y;
  SetIndirectLoadNodeView(cast_int32, axes, repeats, strides, af::DT_INT32);
  af::ascir_op::Cast cast_float("output_cast_float");
  graph.AddNode(cast_float);
  cast_float.x = cast_int32.y;
  SetIndirectLoadNodeView(cast_float, axes, repeats, strides, af::DT_FLOAT);
  af::ascir_op::Exp2 exp2("output_exp");
  graph.AddNode(exp2);
  exp2.x = cast_float.y;
  SetIndirectLoadNodeView(exp2, axes, repeats, strides, af::DT_FLOAT);
  af::ascir_op::Cast cast_back_int32("output_cast_back_int32");
  graph.AddNode(cast_back_int32);
  cast_back_int32.x = exp2.y;
  SetIndirectLoadNodeView(cast_back_int32, axes, repeats, strides, af::DT_INT32);
  af::ascir_op::Cast output_cast("output_cast");
  graph.AddNode(output_cast);
  output_cast.x = cast_back_int32.y;
  SetIndirectLoadNodeView(output_cast, axes, repeats, strides, af::DT_UINT32);
  AddIndirectLoadCustomOutputChain(graph, axes, repeats, strides, af::DT_UINT32,
                                   IndirectLoadOutputPostType::kDirectStore, output_cast);
}

static void AddIndirectLoadOutputChain(af::AscGraph &graph, const std::vector<int64_t> &axes,
                                       const std::vector<af::Expression> &repeats,
                                       const std::vector<af::Expression> &strides, af::DataType data_type,
                                       IndirectLoadInputPreType input_pre_type, bool use_exp2,
                                       IndirectLoadOutputPostType output_post_type,
                                       af::ascir_op::IndirectLoad &indirect_load) {
  if (input_pre_type == IndirectLoadInputPreType::kPrecisionCast) {
    AddIndirectLoadPrecisionOutputChain(graph, axes, repeats, strides, use_exp2, indirect_load);
    return;
  }
  if (output_post_type == IndirectLoadOutputPostType::kCastExp2CastBack) {
    AddIndirectLoadUint32OutputChain(graph, axes, repeats, strides, indirect_load);
    return;
  }
  if (output_post_type != IndirectLoadOutputPostType::kDefault) {
    AddIndirectLoadCustomOutputChain(graph, axes, repeats, strides, data_type, output_post_type, indirect_load);
    return;
  }
  af::ascir_op::Neg output_neg("output_neg");
  graph.AddNode(output_neg);
  if (input_pre_type == IndirectLoadInputPreType::kExp2 || input_pre_type == IndirectLoadInputPreType::kReluExp2) {
    output_neg.x = indirect_load.y;
  } else if (use_exp2) {
    af::ascir_op::Exp2 exp2("output_exp");
    graph.AddNode(exp2);
    exp2.x = indirect_load.y;
    SetIndirectLoadNodeView(exp2, axes, repeats, strides, data_type);
    output_neg.x = exp2.y;
  } else {
    af::ascir_op::Exp exp("output_exp");
    graph.AddNode(exp);
    exp.x = indirect_load.y;
    SetIndirectLoadNodeView(exp, axes, repeats, strides, data_type);
    output_neg.x = exp.y;
  }
  SetIndirectLoadNodeView(output_neg, axes, repeats, strides, data_type);

  af::ascir_op::Store store("store");
  graph.AddNode(store);
  store.x = output_neg.y;
  SetIndirectLoadNodeView(store, axes, repeats, strides, data_type);

  af::ascir_op::Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  SetIndirectLoadNodeView(y, axes, repeats, strides, data_type);
  y.ir_attr.SetIndex(0);
}

static void IndirectLoadStore_BeforeAutofuse(af::AscGraph &graph, size_t rank, int64_t axis, af::DataType data_type,
                                             af::DataType index_type, IndirectLoadInputPreType input_pre_type,
                                             bool use_exp2, IndirectLoadOutputPostType output_post_type,
                                             const std::vector<int64_t> &input_shape,
                                             const std::vector<int64_t> &output_shape, bool mixed_index_pre,
                                             bool direct_index) {
  std::vector<int64_t> input_axes;
  std::vector<af::Expression> input_repeats;
  std::vector<af::Expression> input_strides;
  CreateIndirectLoadAxes(graph, rank, 0UL, input_axes, input_repeats, input_strides, input_shape);
  std::vector<int64_t> output_axes;
  std::vector<af::Expression> output_repeats;
  std::vector<af::Expression> output_strides;
  CreateIndirectLoadAxes(graph, rank, rank, output_axes, output_repeats, output_strides, output_shape);
  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  graph.AddNode(indirect_load);
  if (input_pre_type == IndirectLoadInputPreType::kPrecisionCast) {
    AddIndirectLoadPrecisionCastDataChain(graph, input_axes, input_repeats, input_strides, indirect_load);
  } else {
    AddIndirectLoadDataChain(graph, input_axes, input_repeats, input_strides, data_type, input_pre_type, indirect_load);
  }
  AddIndirectLoadIndexChain(graph, output_axes, output_repeats, output_strides, index_type, mixed_index_pre,
                            direct_index, indirect_load);
  indirect_load.ir_attr.SetAxis(axis);
  const af::DataType indirect_load_type =
      input_pre_type == IndirectLoadInputPreType::kPrecisionCast ? af::DT_FLOAT : data_type;
  SetIndirectLoadNodeView(indirect_load, output_axes, output_repeats, output_strides, indirect_load_type);
  AddIndirectLoadOutputChain(graph, output_axes, output_repeats, output_strides, data_type, input_pre_type, use_exp2,
                             output_post_type, indirect_load);
}

af::ComputeGraphPtr ShareGraph::IndirectLoadStoreFusedGraph(
    size_t rank, int64_t axis, af::DataType data_type, af::DataType index_type, IndirectLoadInputPreType input_pre_type,
    bool use_exp2, IndirectLoadOutputPostType output_post_type, const std::vector<int64_t> &input_shape,
    const std::vector<int64_t> &output_shape, bool mixed_index_pre, bool direct_index) {
  if (rank < 2UL || rank > 4UL || axis < -static_cast<int64_t>(rank) || axis >= static_cast<int64_t>(rank) ||
      (!input_shape.empty() && input_shape.size() != rank) || (!output_shape.empty() && output_shape.size() != rank)) {
    return nullptr;
  }
  af::AscGraph fused_graph("indirect_load_store_test");
  af::ascir_op::Data data0("data0", fused_graph);
  data0.ir_attr.SetIndex(0);
  af::ascir_op::Data data1("data1", fused_graph);
  data1.ir_attr.SetIndex(1);
  if (output_post_type == IndirectLoadOutputPostType::kAddSum) {
    af::ascir_op::Data data2("data2", fused_graph);
    data2.ir_attr.SetIndex(2);
  }
  ComputeGraphPtr compute_graph = af::AscGraphUtils::GetComputeGraph(fused_graph);
  if (compute_graph == nullptr) {
    return nullptr;
  }

  auto data_desc = std::make_shared<GeTensorDesc>();
  data_desc->SetDataType(data_type);
  auto index_desc = std::make_shared<GeTensorDesc>();
  index_desc->SetDataType(index_type);
  auto backend_desc = std::make_shared<OpDesc>("asc_backend", "AscBackend");
  backend_desc->AddInputDesc(data_desc->Clone());
  backend_desc->AddInputDesc(index_desc->Clone());
  if (output_post_type == IndirectLoadOutputPostType::kAddSum) {
    backend_desc->AddInputDesc(data_desc->Clone());
  }
  auto output_desc = data_desc->Clone();
  if (output_post_type == IndirectLoadOutputPostType::kCastSum) {
    output_desc.SetDataType(af::DT_FLOAT);
  }
  backend_desc->AddOutputDesc(output_desc);
  auto backend = compute_graph->AddNode(backend_desc);
  if (backend == nullptr) {
    return nullptr;
  }

  auto sub_graph = std::make_shared<af::AscGraph>("indirect_load_store_test");
  IndirectLoadStore_BeforeAutofuse(*sub_graph, rank, axis, data_type, index_type, input_pre_type, use_exp2,
                                   output_post_type, input_shape, output_shape, mixed_index_pre, direct_index);
  auto fuse_attrs = backend->GetOpDesc()->GetOrCreateAttrsGroup<af::AutoFuseAttrs>();
  if (fuse_attrs == nullptr) {
    return nullptr;
  }
  fuse_attrs->SetAscGraph(sub_graph);

  af::ascir_op::Output output("output");
  output.ir_attr.SetIndex(0);
  auto output_node = compute_graph->AddNode(af::OpDescUtils::GetOpDescFromOperator(output));
  auto data0_node = fused_graph.FindNode("data0");
  auto data1_node = fused_graph.FindNode("data1");
  auto data2_node = fused_graph.FindNode("data2");
  if (data0_node == nullptr || data1_node == nullptr || output_node == nullptr ||
      (output_post_type == IndirectLoadOutputPostType::kAddSum && data2_node == nullptr)) {
    return nullptr;
  }
  if (af::GraphUtils::AddEdge(data0_node->GetOutDataAnchor(0), backend->GetInDataAnchor(0)) != ge::GRAPH_SUCCESS ||
      af::GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), backend->GetInDataAnchor(1)) != ge::GRAPH_SUCCESS ||
      (output_post_type == IndirectLoadOutputPostType::kAddSum &&
       af::GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0), backend->GetInDataAnchor(2)) != ge::GRAPH_SUCCESS) ||
      af::GraphUtils::AddEdge(backend->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)) != ge::GRAPH_SUCCESS ||
      compute_graph->TopologicalSorting() != ge::GRAPH_SUCCESS) {
    return nullptr;
  }
  return compute_graph;
}

template <typename Input>
static void AddIndirectLoadMixedInput(af::AscGraph &graph, const std::string &name, int64_t index, af::DataType dtype,
                                      const std::vector<int64_t> &axes, const std::vector<af::Expression> &repeats,
                                      const std::vector<af::Expression> &strides, Input &input) {
  af::ascir_op::Data data((name + "_data").c_str());
  graph.AddNode(data);
  data.y.dtype = dtype;
  data.ir_attr.SetIndex(index);
  af::ascir_op::Load load((name + "_load").c_str());
  graph.AddNode(load);
  load.x = data.y;
  SetIndirectLoadNodeView(load, axes, repeats, strides, dtype);
  input = load.y;
}

static void AddIndirectLoadMixedIndexChain(af::AscGraph &graph, const std::vector<int64_t> &axes,
                                           const std::vector<af::Expression> &repeats,
                                           const std::vector<af::Expression> &strides,
                                           af::ascir_op::IndirectLoad &indirect_load, int64_t input_index,
                                           bool is_simt) {
  af::ascir_op::Add add("index_add");
  graph.AddNode(add);
  AddIndirectLoadMixedInput(graph, "index0", input_index, af::DT_INT32, axes, repeats, strides, add.x1);
  AddIndirectLoadMixedInput(graph, "index1", input_index + 1, af::DT_INT32, axes, repeats, strides, add.x2);
  SetIndirectLoadNodeView(add, axes, repeats, strides, af::DT_INT32);
  af::ascir_op::Cast cast("index_cast");
  graph.AddNode(cast);
  cast.x = add.y;
  SetIndirectLoadNodeView(cast, axes, repeats, strides, af::DT_FLOAT);
  af::AscOpOutput *floor_input = &cast.y;
  af::ascir_op::Exp2 exp2("index_exp2");
  af::ascir_op::Log2 log2("index_log2");
  if (!is_simt) {
    graph.AddNode(exp2);
    exp2.x = cast.y;
    SetIndirectLoadNodeView(exp2, axes, repeats, strides, af::DT_FLOAT);
    graph.AddNode(log2);
    log2.x = exp2.y;
    SetIndirectLoadNodeView(log2, axes, repeats, strides, af::DT_FLOAT);
    floor_input = &log2.y;
  }
  af::ascir_op::FloorToInt floor("index_floor");
  graph.AddNode(floor);
  floor.x = *floor_input;
  SetIndirectLoadNodeView(floor, axes, repeats, strides, af::DT_INT32);
  indirect_load.x2 = floor.y;
}

static void AddIndirectLoadMixedOutputChain(af::AscGraph &graph, const std::vector<int64_t> &axes,
                                            const std::vector<af::Expression> &repeats,
                                            const std::vector<af::Expression> &strides,
                                            af::ascir_op::IndirectLoad &indirect_load, int64_t input_index) {
  af::ascir_op::Add add("output_add");
  graph.AddNode(add);
  add.x1 = indirect_load.y;
  AddIndirectLoadMixedInput(graph, "addend", input_index, af::DT_FLOAT16, axes, repeats, strides, add.x2);
  SetIndirectLoadNodeView(add, axes, repeats, strides, af::DT_FLOAT16);
  af::ascir_op::CopySign copy_sign("output_copy_sign");
  graph.AddNode(copy_sign);
  copy_sign.x1 = add.y;
  AddIndirectLoadMixedInput(graph, "sign", input_index + 1, af::DT_FLOAT16, axes, repeats, strides, copy_sign.x2);
  SetIndirectLoadNodeView(copy_sign, axes, repeats, strides, af::DT_FLOAT16);
  af::ascir_op::Mul mul("output_mul");
  graph.AddNode(mul);
  mul.x1 = copy_sign.y;
  AddIndirectLoadMixedInput(graph, "scale", input_index + 2, af::DT_FLOAT16, axes, repeats, strides, mul.x2);
  SetIndirectLoadNodeView(mul, axes, repeats, strides, af::DT_FLOAT16);
  af::ascir_op::Store store("store");
  graph.AddNode(store);
  store.x = mul.y;
  SetIndirectLoadNodeView(store, axes, repeats, strides, af::DT_FLOAT16);
  af::ascir_op::Output output("output");
  graph.AddNode(output);
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  SetIndirectLoadNodeView(output, axes, repeats, strides, af::DT_FLOAT16);
}

static void AddIndirectLoadSimtCoverageIndexChain(af::AscGraph &graph, const std::vector<int64_t> &axes,
                                                  const std::vector<af::Expression> &repeats,
                                                  const std::vector<af::Expression> &strides,
                                                  af::ascir_op::IndirectLoad &indirect_load) {
  af::ascir_op::Data index0_data("index0_data");
  graph.AddNode(index0_data);
  index0_data.y.dtype = af::DT_INT32;
  index0_data.ir_attr.SetIndex(1L);
  af::ascir_op::Load index0_load("index0_load");
  graph.AddNode(index0_load);
  index0_load.x = index0_data.y;
  SetIndirectLoadNodeView(index0_load, axes, repeats, strides, af::DT_INT32);
  af::ascir_op::Data index1_data("index1_data");
  graph.AddNode(index1_data);
  index1_data.y.dtype = af::DT_INT32;
  index1_data.ir_attr.SetIndex(2L);
  af::ascir_op::Load index1_load("index1_load");
  graph.AddNode(index1_load);
  index1_load.x = index1_data.y;
  SetIndirectLoadNodeView(index1_load, axes, repeats, strides, af::DT_INT32);
  af::ascir_op::Maximum maximum("index_maximum");
  graph.AddNode(maximum);
  maximum.x1 = index0_load.y;
  maximum.x2 = index1_load.y;
  SetIndirectLoadNodeView(maximum, axes, repeats, strides, af::DT_INT32);

  af::ascir_op::Minimum minimum("index_minimum");
  graph.AddNode(minimum);
  minimum.x1 = index0_load.y;
  minimum.x2 = index1_load.y;
  SetIndirectLoadNodeView(minimum, axes, repeats, strides, af::DT_INT32);
  af::ascir_op::Gt greater("index_gt");
  graph.AddNode(greater);
  greater.x1 = index0_load.y;
  greater.x2 = index1_load.y;
  SetIndirectLoadNodeView(greater, axes, repeats, strides, af::DT_UINT8);
  af::ascir_op::Lt less("index_lt");
  graph.AddNode(less);
  less.x1 = index0_load.y;
  less.x2 = index1_load.y;
  SetIndirectLoadNodeView(less, axes, repeats, strides, af::DT_UINT8);
  af::ascir_op::LogicalOr logical_or("index_logical_or");
  graph.AddNode(logical_or);
  logical_or.x1 = greater.y;
  logical_or.x2 = less.y;
  SetIndirectLoadNodeView(logical_or, axes, repeats, strides, af::DT_UINT8);
  af::ascir_op::Where where("index_where");
  graph.AddNode(where);
  where.x1 = logical_or.y;
  where.x2 = maximum.y;
  where.x3 = minimum.y;
  SetIndirectLoadNodeView(where, axes, repeats, strides, af::DT_INT32);
  indirect_load.x2 = where.y;
}

static void AddIndirectLoadSimtCoverageOutputChain(af::AscGraph &graph, const std::vector<int64_t> &axes,
                                                   const std::vector<af::Expression> &repeats,
                                                   const std::vector<af::Expression> &strides,
                                                   af::ascir_op::IndirectLoad &indirect_load) {
  af::ascir_op::Sub sub("output_sub");
  graph.AddNode(sub);
  sub.x1 = indirect_load.y;
  AddIndirectLoadMixedInput(graph, "addend", 3L, af::DT_FLOAT16, axes, repeats, strides, sub.x2);
  SetIndirectLoadNodeView(sub, axes, repeats, strides, af::DT_FLOAT16);
  af::ascir_op::Div div("output_div");
  graph.AddNode(div);
  div.x1 = sub.y;
  AddIndirectLoadMixedInput(graph, "scale", 5L, af::DT_FLOAT16, axes, repeats, strides, div.x2);
  SetIndirectLoadNodeView(div, axes, repeats, strides, af::DT_FLOAT16);
  af::ascir_op::CopySign copy_sign("output_copy_sign");
  graph.AddNode(copy_sign);
  copy_sign.x1 = div.y;
  AddIndirectLoadMixedInput(graph, "sign", 4L, af::DT_FLOAT16, axes, repeats, strides, copy_sign.x2);
  SetIndirectLoadNodeView(copy_sign, axes, repeats, strides, af::DT_FLOAT16);
  af::ascir_op::Store store("store");
  graph.AddNode(store);
  store.x = copy_sign.y;
  SetIndirectLoadNodeView(store, axes, repeats, strides, af::DT_FLOAT16);
  af::ascir_op::Output output("output");
  graph.AddNode(output);
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  SetIndirectLoadNodeView(output, axes, repeats, strides, af::DT_FLOAT16);
}

static std::shared_ptr<af::AscGraph> CreateIndirectLoadMixedElementwiseSubGraph(
    size_t rank, int64_t axis, bool is_simt, const std::vector<int64_t> &input_shape,
    const std::vector<int64_t> &output_shape, bool simt_coverage) {
  auto graph = std::make_shared<af::AscGraph>("indirect_load_mixed_elementwise_test");
  std::vector<int64_t> input_axes;
  std::vector<af::Expression> input_repeats;
  std::vector<af::Expression> input_strides;
  CreateIndirectLoadAxes(*graph, rank, 0UL, input_axes, input_repeats, input_strides, input_shape);
  std::vector<int64_t> output_axes;
  std::vector<af::Expression> output_repeats;
  std::vector<af::Expression> output_strides;
  CreateIndirectLoadAxes(*graph, rank, rank, output_axes, output_repeats, output_strides, output_shape);
  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  graph->AddNode(indirect_load);
  if (is_simt) {
    AddIndirectLoadMixedInput(*graph, "x", 0, af::DT_FLOAT16, input_axes, input_repeats, input_strides,
                              indirect_load.x1);
  } else {
    af::ascir_op::CopySign x_copy_sign("x_copy_sign");
    graph->AddNode(x_copy_sign);
    AddIndirectLoadMixedInput(*graph, "x0", 0, af::DT_FLOAT16, input_axes, input_repeats, input_strides,
                              x_copy_sign.x1);
    AddIndirectLoadMixedInput(*graph, "x1", 1, af::DT_FLOAT16, input_axes, input_repeats, input_strides,
                              x_copy_sign.x2);
    SetIndirectLoadNodeView(x_copy_sign, input_axes, input_repeats, input_strides, af::DT_FLOAT16);
    indirect_load.x1 = x_copy_sign.y;
  }
  const int64_t index_input = is_simt ? 1L : 2L;
  if (simt_coverage) {
    AddIndirectLoadSimtCoverageIndexChain(*graph, output_axes, output_repeats, output_strides, indirect_load);
  } else {
    AddIndirectLoadMixedIndexChain(*graph, output_axes, output_repeats, output_strides, indirect_load, index_input,
                                   is_simt);
  }
  indirect_load.ir_attr.SetAxis(axis);
  SetIndirectLoadNodeView(indirect_load, output_axes, output_repeats, output_strides, af::DT_FLOAT16);
  if (simt_coverage) {
    AddIndirectLoadSimtCoverageOutputChain(*graph, output_axes, output_repeats, output_strides, indirect_load);
  } else {
    AddIndirectLoadMixedOutputChain(*graph, output_axes, output_repeats, output_strides, indirect_load,
                                    index_input + 2L);
  }
  return graph;
}

static af::ComputeGraphPtr CreateIndirectLoadMixedElementwiseFusedGraph(size_t rank, int64_t axis, bool is_simt,
                                                                        const std::vector<int64_t> &input_shape,
                                                                        const std::vector<int64_t> &output_shape,
                                                                        bool simt_coverage) {
  af::AscGraph fused_graph("indirect_load_mixed_elementwise_test");
  std::vector<af::AscNodePtr> inputs;
  const int32_t input_count = is_simt ? 6 : 7;
  for (int32_t i = 0; i < input_count; ++i) {
    const std::string name = "data" + std::to_string(i);
    af::ascir_op::Data data(name.c_str(), fused_graph);
    data.ir_attr.SetIndex(i);
    inputs.emplace_back(fused_graph.FindNode(name.c_str()));
  }
  const af::ComputeGraphPtr compute_graph = af::AscGraphUtils::GetComputeGraph(fused_graph);
  if (compute_graph == nullptr || std::any_of(inputs.begin(), inputs.end(), [](const auto &node) { return !node; })) {
    return nullptr;
  }
  auto data_desc = std::make_shared<GeTensorDesc>();
  data_desc->SetDataType(af::DT_FLOAT16);
  auto index_desc = std::make_shared<GeTensorDesc>();
  index_desc->SetDataType(af::DT_INT32);
  auto backend_desc = std::make_shared<OpDesc>("asc_backend", "AscBackend");
  for (size_t i = 0UL; i < inputs.size(); ++i) {
    const size_t index_input = is_simt ? 1UL : 2UL;
    backend_desc->AddInputDesc((i == index_input || i == index_input + 1UL) ? index_desc->Clone() : data_desc->Clone());
  }
  backend_desc->AddOutputDesc(data_desc->Clone());
  const af::NodePtr backend = compute_graph->AddNode(backend_desc);
  af::ascir_op::Output output("output");
  output.ir_attr.SetIndex(0);
  const af::NodePtr output_node = compute_graph->AddNode(af::OpDescUtils::GetOpDescFromOperator(output));
  if (backend == nullptr || output_node == nullptr) {
    return nullptr;
  }
  auto fuse_attrs = backend->GetOpDesc()->GetOrCreateAttrsGroup<af::AutoFuseAttrs>();
  if (fuse_attrs == nullptr) {
    return nullptr;
  }
  fuse_attrs->SetAscGraph(
      CreateIndirectLoadMixedElementwiseSubGraph(rank, axis, is_simt, input_shape, output_shape, simt_coverage));
  for (size_t i = 0UL; i < inputs.size(); ++i) {
    if (af::GraphUtils::AddEdge(inputs[i]->GetOutDataAnchor(0), backend->GetInDataAnchor(i)) != ge::GRAPH_SUCCESS) {
      return nullptr;
    }
  }
  if (af::GraphUtils::AddEdge(backend->GetOutDataAnchor(0), output_node->GetInDataAnchor(0)) != ge::GRAPH_SUCCESS ||
      compute_graph->TopologicalSorting() != ge::GRAPH_SUCCESS) {
    return nullptr;
  }
  return compute_graph;
}

static bool IsIndirectLoadMixedShapeValid(size_t rank, int64_t axis, const std::vector<int64_t> &input_shape,
                                          const std::vector<int64_t> &output_shape) {
  const int64_t normalized_axis = axis < 0L ? axis + static_cast<int64_t>(rank) : axis;
  if (rank < 2UL || rank > 4UL || normalized_axis < 0L || normalized_axis >= static_cast<int64_t>(rank) ||
      input_shape.empty() != output_shape.empty() || (!input_shape.empty() && input_shape.size() != rank) ||
      (!output_shape.empty() && output_shape.size() != rank) || (!input_shape.empty() && input_shape == output_shape)) {
    return false;
  }
  if (!input_shape.empty()) {
    for (size_t i = 0UL; i < rank; ++i) {
      if (static_cast<int64_t>(i) != normalized_axis && input_shape[i] < output_shape[i]) {
        return false;
      }
    }
  }
  return true;
}

af::ComputeGraphPtr ShareGraph::IndirectLoadMixedElementwiseFusedGraph(size_t rank, int64_t axis, bool is_simt,
                                                                       const std::vector<int64_t> &input_shape,
                                                                       const std::vector<int64_t> &output_shape) {
  if (!IsIndirectLoadMixedShapeValid(rank, axis, input_shape, output_shape)) {
    return nullptr;
  }
  return CreateIndirectLoadMixedElementwiseFusedGraph(rank, axis, is_simt, input_shape, output_shape, false);
}

af::ComputeGraphPtr ShareGraph::IndirectLoadSimtElementwiseCoverageFusedGraph(
    size_t rank, int64_t axis, const std::vector<int64_t> &input_shape, const std::vector<int64_t> &output_shape) {
  if (!IsIndirectLoadMixedShapeValid(rank, axis, input_shape, output_shape)) {
    return nullptr;
  }
  return CreateIndirectLoadMixedElementwiseFusedGraph(rank, axis, true, input_shape, output_shape, true);
}

static void ConstructVVAscGraphAxisInfoForOneAxisGather(af::AscGraph &graph, size_t dims_size) {
  std::vector<int64_t> axis;
  std::vector<af::Expression> repeats;
  std::vector<af::Expression> strides;
  auto ONE = Symbol(1);

  // 构造符号、轴信息
  for (size_t i = 0; i < dims_size; i++) {
    std::string sym_str = "s" + std::to_string(i);
    std::string axis_str = "z" + std::to_string(i);
    const auto sym_s = Symbol(2);
    auto axis_z = graph.CreateAxis(axis_str.c_str(), sym_s);
    axis.push_back(axis_z.id);
    repeats.push_back(sym_s);
    strides.push_back(ONE);
  }
  // 计算每个轴的stride
  for (int i = dims_size - 2; i >= 0; i--) {
    strides[i] = af::sym::Mul(repeats[i + 1], strides[i + 1]);
  }
  // 将原始轴信息设置到图中所有节点上
  for (auto node : graph.GetAllNodes()) {
    if (af::ops::IsOps<af::ascir_op::Scalar>(node)) {
      continue;
    }
    node->attr.sched.axis = axis;
    for (auto output_attr : node->outputs()) {
      output_attr->attr.axis = axis;
      output_attr->attr.repeats = repeats;
      output_attr->attr.strides = strides;
    }
    if (node->GetName() == "data0") {
      auto node_output = node->outputs()[0];
      node_output->attr.dtype = DT_FLOAT;
    }
    if (node->GetName() == "data1") {
      auto node_output = node->outputs()[0];
      node_output->attr.dtype = DT_INT32;
    }
    if (node->GetType() == "Gather") {
      auto gather_output = node->outputs()[0];
      gather_output->attr.dtype = DT_FLOAT;
      auto gather_input0 = node->inputs()[0];
      gather_input0->attr.dtype = DT_FLOAT;
      auto gather_input1 = node->inputs()[1];
      gather_input1->attr.dtype = DT_INT32;
      const auto &op = node->GetOpDesc();
      const auto &attr = op->GetAttrsGroup<AscNodeAttr>();
      auto gather_ir_attr = dynamic_cast<ascir_op::Gather::AscGatherIrAttrDef *>(attr->ir_attr.get());
      gather_ir_attr->SetAxis(0);
      gather_ir_attr->SetNegative_index_support(false);
    }
  }
}

static void LoadGatherOneAxisAbsStore_BeforeAutofuse(af::AscGraph &graph, int64_t gather_axis, af::DataType data_type) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  af::ascir_op::Gather gather("gather");
  gather.x1 = x1.y;
  gather.x2 = x2.y;

  af::ascir_op::Abs abs("abs");
  abs.x = gather.y;

  af::ascir_op::Store x_out("store");
  x_out.x = gather.y;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfoForOneAxisGather(graph, 1);
}

af::ComputeGraphPtr ShareGraph::LoadGatherOneAxisAbsStore(int64_t gather_axis, af::DataType data_type) {
  auto builder = GraphBuilder("load_gather_one_axis_abs_store_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_gather_one_axis_abs_store_store_test");
  LoadGatherOneAxisAbsStore_BeforeAutofuse(sub_graph, gather_axis, data_type);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void GatherReduceStore_BeforeAutofuse(af::AscGraph &graph, int64_t gather_axis, af::DataType data_type) {
  auto s0 = graph.CreateSizeVar("s0");
  auto s1 = graph.CreateSizeVar("s1");
  auto s2 = graph.CreateSizeVar("s2");
  auto s3 = graph.CreateSizeVar("s3");

  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);
  auto z3 = graph.CreateAxis("z3", s3);
  af::ascir_op::Data x1("x1");
  graph.AddNode(x1);
  x1.y.dtype = data_type;
  x1.attr.sched.axis = {z0.id, z1.id};
  *x1.y.axis = {z0.id, z1.id};
  *x1.y.repeats = {s0, s1};
  *x1.y.strides = {s1, af::ops::One};
  x1.ir_attr.SetIndex(0);

  af::ascir_op::Data x2("x2");
  graph.AddNode(x2);
  x2.y.dtype = af::DT_INT32;
  x2.attr.sched.axis = {z2.id, z3.id};
  *x2.y.axis = {z2.id, z3.id};
  *x2.y.repeats = {s2, s3};
  *x2.y.strides = {s3, af::ops::One};
  x2.ir_attr.SetIndex(1);

  af::ascir_op::Gather gather("gather");
  graph.AddNode(gather);
  gather.x1 = x1.y;
  gather.x2 = x2.y;
  gather.ir_attr.SetAxis(0);
  gather.ir_attr.SetNegative_index_support(false);
  gather.attr.sched.axis = {z2.id, z3.id, z1.id};
  *gather.y.axis = {z2.id, z3.id, z1.id};
  *gather.y.repeats = {s2, s3, s1};
  *gather.y.strides = {s3 * s1, s1, af::ops::One};

  af::ascir_op::Max max("max");
  graph.AddNode(max);
  max.x = gather.y;
  max.attr.sched.axis = {z2.id, z3.id, z1.id};
  *max.y.axis = {z2.id, z3.id, z1.id};
  *max.y.repeats = {ops::One, s3, ops::One};
  *max.y.strides = {ops::Zero, af::ops::One, af::ops::Zero};

  af::ascir_op::Store store("store");
  graph.AddNode(store);
  store.x = max.y;
  store.attr.sched.axis = {z2.id, z3.id, z1.id};
  *store.y.axis = {z2.id, z3.id, z1.id};
  *store.y.repeats = {ops::One, s3, ops::One};
  *store.y.strides = {ops::Zero, af::ops::One, af::ops::Zero};

  af::ascir_op::Output y("y");
  graph.AddNode(y);
  y.x = store.y;
  y.y.dtype = data_type;
  y.ir_attr.SetIndex(0);
}

af::ComputeGraphPtr ShareGraph::GatherReduceStore(int64_t gather_axis, af::DataType data_type) {
  auto builder = GraphBuilder("gather_reduce_store_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("gather_reduce_store_store_test");
  GatherReduceStore_BeforeAutofuse(sub_graph, gather_axis, data_type);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *         data0  data1
 *           |      |
 *         load0  load1
 *           \     /
 *              GE
 *              |
 *             Cast
 *              |
 *             Sum
 *              |
 *            store
 *              |
 *            ouput0
 */
static void CreateLoadCompareCastSumStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  const Expression s0 = graph.CreateSizeVar(3);
  const Expression s1 = graph.CreateSizeVar(77);
  const Expression s2 = graph.CreateSizeVar(21);
  auto One = Symbol(1);
  auto Zero = Symbol(0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x1.y.dtype = af::DT_FLOAT;
  x2.y.dtype = af::DT_FLOAT;

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.attr.api.compute_type = ComputeType::kComputeLoad;
  x1Local.attr.api.type = af::ApiType::kAPITypeCompute;
  x1Local.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x1Local.y.axis = {z0.id, z1.id, z2.id};
  *x1Local.y.repeats = {s0, s1, s2};
  *x1Local.y.strides = {s1 * s2, s2, One};
  x1Local.attr.api.unit = ComputeUnit::kUnitMTE2;
  x1Local.y.dtype = af::DT_FLOAT;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.attr.api.compute_type = ComputeType::kComputeLoad;
  x2Local.attr.api.type = af::ApiType::kAPITypeCompute;
  x2Local.attr.sched.axis = {z0.id, z1.id, z2.id};
  *x2Local.y.axis = {z0.id, z1.id, z2.id};
  *x2Local.y.repeats = {s0, s1, s2};
  *x2Local.y.strides = {s1 * s2, s2, One};
  x2Local.attr.api.unit = ComputeUnit::kUnitMTE2;
  x2Local.y.dtype = af::DT_FLOAT;

  af::ascir_op::Ge ge0("ge0");
  ge0.x1 = x1Local.y;
  ge0.x2 = x2Local.y;
  ge0.attr.api.compute_type = ComputeType::kComputeElewise;
  ge0.attr.api.type = af::ApiType::kAPITypeCompute;
  ge0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *ge0.y.axis = {z0.id, z1.id, z2.id};
  *ge0.y.repeats = {s0, s1, s2};
  *ge0.y.strides = {s1 * s2, s2, One};
  ge0.attr.api.unit = ComputeUnit::kUnitVector;
  ge0.y.dtype = af::DT_UINT8;

  af::ascir_op::Cast cast0("cast0");
  cast0.x = ge0.y;
  cast0.attr.api.compute_type = ComputeType::kComputeElewise;
  cast0.attr.api.type = af::ApiType::kAPITypeCompute;
  cast0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *cast0.y.axis = {z0.id, z1.id, z2.id};
  *cast0.y.repeats = {s0, s1, s2};
  *cast0.y.strides = {s1 * s2, s2, One};
  cast0.attr.api.unit = ComputeUnit::kUnitVector;
  cast0.y.dtype = af::DT_INT8;

  af::ascir_op::Cast cast1("cast1");
  cast1.x = cast0.y;
  cast1.attr.api.compute_type = ComputeType::kComputeElewise;
  cast1.attr.api.type = af::ApiType::kAPITypeCompute;
  cast1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *cast1.y.axis = {z0.id, z1.id, z2.id};
  *cast1.y.repeats = {s0, s1, s2};
  *cast1.y.strides = {s1 * s2, s2, One};
  cast1.attr.api.unit = ComputeUnit::kUnitVector;
  cast1.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Cast cast2("cast2");
  cast2.x = cast1.y;
  cast2.attr.api.compute_type = ComputeType::kComputeElewise;
  cast2.attr.api.type = af::ApiType::kAPITypeCompute;
  cast2.attr.sched.axis = {z0.id, z1.id, z2.id};
  *cast2.y.axis = {z0.id, z1.id, z2.id};
  *cast2.y.repeats = {s0, s1, s2};
  *cast2.y.strides = {s1 * s2, s2, One};
  cast2.attr.api.unit = ComputeUnit::kUnitVector;
  cast2.y.dtype = af::DT_FLOAT;

  af::ascir_op::Sum sum0("sum0");
  sum0.x = cast2.y;
  sum0.attr.sched.axis = {z0.id, z1.id, z2.id};
  *sum0.y.axis = {z0.id, z1.id, z2.id};
  *sum0.y.repeats = {s0, One, s2};
  *sum0.y.strides = {s2, Zero, One};
  sum0.y.dtype = af::DT_FLOAT;

  af::ascir_op::Store sum0_store("store");
  sum0_store.x = sum0.y;
  sum0_store.attr.api.compute_type = ComputeType::kComputeStore;
  sum0_store.attr.api.type = af::ApiType::kAPITypeCompute;
  sum0_store.attr.sched.axis = {z0.id, z1.id, z2.id};
  *sum0_store.y.axis = {z0.id, z1.id, z2.id};
  *sum0_store.y.repeats = {s0, One, s2};
  *sum0_store.y.strides = {s2, Zero, One};
  sum0_store.attr.api.unit = ComputeUnit::kUnitMTE3;
  sum0_store.y.dtype = af::DT_FLOAT;

  af::ascir_op::Output y("output");
  y.x = sum0_store.y;
  y.attr.api.compute_type = ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.y.dtype = af::DataType::DT_FLOAT;
  y.ir_attr.SetIndex(0);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::LoadCompareCastSumStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_compare_cast_sum_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_compare_cast_sum_store");
  CreateLoadCompareCastSumStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *        log2
 *         |
 *       load0
 *         |
 *       data0
 */
static void CreateLoadLog2StoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  af::ascir_op::Log2 log2("log2");
  log2.x = x1Local.y;

  af::ascir_op::Store log2_store("store");
  log2_store.x = log2.y;

  af::ascir_op::Output y("output");
  y.x = log2_store.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::LoadLog2StoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("load_log2_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_log2_store");
  CreateLoadLog2StoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *        mod
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateModAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = af::DT_FLOAT;
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = af::DT_FLOAT;

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = af::DT_FLOAT;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = af::DT_FLOAT;

  af::ascir_op::Mod mod("mod");
  mod.x1 = x1Local.y;
  mod.x2 = x2Local.y;
  mod.y.dtype = af::DT_FLOAT;

  af::ascir_op::Store x_out("store");
  x_out.x = mod.y;
  x_out.y.dtype = af::DT_FLOAT;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::ModFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("mod_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("mod");
  CreateModAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *       LShift
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreateLoadLShiftStoreAscGraph(af::AscGraph &graph, size_t dims_size, af::DataType in_dtype,
                                          af::DataType out_dtype) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = in_dtype;
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = in_dtype;

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = in_dtype;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = in_dtype;

  af::ascir_op::LShift lshift("lshift");
  lshift.x1 = x1Local.y;
  lshift.x2 = x2Local.y;
  lshift.y.dtype = out_dtype;

  af::ascir_op::Store x_out("store");
  x_out.x = lshift.y;
  x_out.y.dtype = out_dtype;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::LoadLShiftStoreFusedGraph(size_t dims_size, af::DataType in_dtype,
                                                          af::DataType out_dtype) {
  auto builder = GraphBuilder("load_lshift_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_lshift_store");
  CreateLoadLShiftStoreAscGraph(sub_graph, dims_size, in_dtype, out_dtype);
  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

struct MatmulGraphContext {
  af::AscGraph &graph;
  af::Expression s0;
  af::Expression s1;
  af::AxisId z0;
  af::AxisId z1;
};

static MatmulGraphContext CreateMatmulGraphContext(af::AscGraph &graph, bool is_dynamic = false) {
  auto s0 = is_dynamic ? graph.CreateSizeVar("s0") : graph.CreateSizeVar(32);
  auto s1 = is_dynamic ? graph.CreateSizeVar("s1") : graph.CreateSizeVar(32);
  auto &z0 = graph.CreateAxis("z0", s0);
  auto &z1 = graph.CreateAxis("z1", s1);
  return {graph, s0, s1, z0.id, z1.id};
}

template <typename T>
static void SetMatmulGraphLayout(T &op, const MatmulGraphContext &context, af::DataType dtype,
                                 const std::vector<af::Expression> &repeats,
                                 const std::vector<af::Expression> &strides) {
  op.attr.sched.axis = {context.z0, context.z1};
  op.y.dtype = dtype;
  *op.y.axis = {context.z0, context.z1};
  *op.y.repeats = repeats;
  *op.y.strides = strides;
}

template <typename T>
static void SetFullMatmulGraphLayout(T &op, const MatmulGraphContext &context, af::DataType dtype) {
  SetMatmulGraphLayout(op, context, dtype, {context.s0, context.s1}, {context.s1, af::ops::One});
}

static void CreateMatmulPrefix(const MatmulGraphContext &context, af::ascir_op::MatMul &matmul) {
  af::ascir_op::Data data0("data0", context.graph);
  SetFullMatmulGraphLayout(data0, context, af::DT_FLOAT16);
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data0.ir_attr.SetIndex(0);
  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  SetFullMatmulGraphLayout(load0, context, af::DT_FLOAT16);

  af::ascir_op::Data data1("data1", context.graph);
  SetFullMatmulGraphLayout(data1, context, af::DT_FLOAT16);
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data1.ir_attr.SetIndex(1);
  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  SetFullMatmulGraphLayout(load1, context, af::DT_FLOAT16);

  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  SetFullMatmulGraphLayout(matmul, context, af::DT_FLOAT);
  matmul.ir_attr.SetTranspose_x1(1);
  matmul.ir_attr.SetTranspose_x2(0);
  matmul.ir_attr.SetHas_relu(0);
  matmul.ir_attr.SetEnable_hf32(0);
  matmul.ir_attr.SetOffset_x(0);
}

static void ConnectAddBroadcast(const MatmulGraphContext &context, af::ascir_op::Add &add_op) {
  af::ascir_op::Data data2("data2", context.graph);
  SetMatmulGraphLayout(data2, context, af::DT_FLOAT, {af::ops::One, af::ops::One}, {af::ops::Zero, af::ops::Zero});
  data2.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data2.ir_attr.SetIndex(2);
  af::ascir_op::Load load2("load2");
  load2.x = data2.y;
  SetMatmulGraphLayout(load2, context, af::DT_FLOAT, {af::ops::One, af::ops::One}, {af::ops::Zero, af::ops::Zero});

  af::ascir_op::Broadcast broadcast0("broadcast0");
  broadcast0.x = load2.y;
  SetMatmulGraphLayout(broadcast0, context, af::DT_FLOAT, {af::ops::One, context.s1}, {af::ops::Zero, af::ops::One});
  af::ascir_op::Broadcast broadcast1("broadcast1");
  broadcast1.x = broadcast0.y;
  SetFullMatmulGraphLayout(broadcast1, context, af::DT_FLOAT);
  add_op.x2 = broadcast1.y;
}

static void ConnectSigmoidBroadcast(const MatmulGraphContext &context, af::ascir_op::Mul &mul) {
  af::ascir_op::Data data3("data3", context.graph);
  SetMatmulGraphLayout(data3, context, af::DT_FLOAT, {context.s0, af::ops::One}, {af::ops::One, af::ops::Zero});
  data3.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data3.ir_attr.SetIndex(3);
  af::ascir_op::Load load3("load3");
  load3.x = data3.y;
  SetMatmulGraphLayout(load3, context, af::DT_FLOAT, {context.s0, af::ops::One}, {af::ops::One, af::ops::Zero});
  af::ascir_op::Broadcast broadcast2("broadcast2");
  broadcast2.x = load3.y;
  SetFullMatmulGraphLayout(broadcast2, context, af::DT_FLOAT);
  af::ascir_op::Sigmoid sigmoid0("sigmoid0");
  sigmoid0.x = broadcast2.y;
  SetFullMatmulGraphLayout(sigmoid0, context, af::DT_FLOAT);
  mul.x2 = sigmoid0.y;
}

static void ConnectRsqrtBroadcast(const MatmulGraphContext &context, af::ascir_op::Sub &sub) {
  af::ascir_op::Data data4("data4", context.graph);
  SetMatmulGraphLayout(data4, context, af::DT_FLOAT, {af::ops::One, context.s1}, {af::ops::Zero, af::ops::One});
  data4.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data4.ir_attr.SetIndex(4);
  af::ascir_op::Load load4("load4");
  load4.x = data4.y;
  SetMatmulGraphLayout(load4, context, af::DT_FLOAT, {af::ops::One, context.s1}, {af::ops::Zero, af::ops::One});
  af::ascir_op::Broadcast broadcast3("broadcast3");
  broadcast3.x = load4.y;
  SetFullMatmulGraphLayout(broadcast3, context, af::DT_FLOAT);
  af::ascir_op::Rsqrt rsqrt0("rsqrt0");
  rsqrt0.x = broadcast3.y;
  SetFullMatmulGraphLayout(rsqrt0, context, af::DT_FLOAT);
  sub.x2 = rsqrt0.y;
}

static void CreateMatmulGraphOutput(const MatmulGraphContext &context, const af::AscOpOutput &input) {
  af::ascir_op::Store store_op("store");
  store_op.x = input;
  SetFullMatmulGraphLayout(store_op, context, af::DT_FLOAT);
  af::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = af::DT_FLOAT;
  output_op.ir_attr.SetIndex(0);
}

static void CreateMatmulGraphOutput(const MatmulGraphContext &context, const af::AscOpOutput &input,
                                    af::DataType dtype) {
  af::ascir_op::Store store_op("store");
  store_op.x = input;
  SetFullMatmulGraphLayout(store_op, context, dtype);
  af::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = dtype;
  output_op.ir_attr.SetIndex(0);
}

static void ConnectCompareInputs(const MatmulGraphContext &context, af::ascir_op::Eq &eq0) {
  af::ascir_op::Data data2("data2", context.graph);
  SetFullMatmulGraphLayout(data2, context, af::DT_FLOAT);
  data2.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data2.ir_attr.SetIndex(2);
  af::ascir_op::Load load2("load2");
  load2.x = data2.y;
  SetFullMatmulGraphLayout(load2, context, af::DT_FLOAT);

  af::ascir_op::Data data3("data3", context.graph);
  SetMatmulGraphLayout(data3, context, af::DT_FLOAT, {af::ops::One, af::ops::One}, {af::ops::Zero, af::ops::Zero});
  data3.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data3.ir_attr.SetIndex(3);
  af::ascir_op::Load load3("load3");
  load3.x = data3.y;
  SetMatmulGraphLayout(load3, context, af::DT_FLOAT, {af::ops::One, af::ops::One}, {af::ops::Zero, af::ops::Zero});

  eq0.x1 = load2.y;
  eq0.x2 = load3.y;
  SetFullMatmulGraphLayout(eq0, context, af::DT_UINT8);
}

/**
 *                  sub
 *               mul    \
 *              /   \    \
 *           add sigmoid rsqrt
 *          /   \     \    \
 *       Matmul brc    brc  brc
 *      /     \   \     \    \
 *   data0 data1 data2 data3 data4
 */
static void CreateMatmulElewiseBrcGraph(af::AscGraph &graph, bool is_dynamic) {
  const auto context = CreateMatmulGraphContext(graph, is_dynamic);
  af::ascir_op::MatMul matmul("matmul");
  CreateMatmulPrefix(context, matmul);

  af::ascir_op::Add add_op("add");
  add_op.x1 = matmul.y;
  SetFullMatmulGraphLayout(add_op, context, af::DT_FLOAT);
  ConnectAddBroadcast(context, add_op);

  af::ascir_op::Mul mul("mul");
  mul.x1 = add_op.y;
  SetFullMatmulGraphLayout(mul, context, af::DT_FLOAT);
  ConnectSigmoidBroadcast(context, mul);

  af::ascir_op::Sub sub("sub");
  sub.x1 = mul.y;
  SetFullMatmulGraphLayout(sub, context, af::DT_FLOAT);
  ConnectRsqrtBroadcast(context, sub);
  CreateMatmulGraphOutput(context, sub.y);
}

af::ComputeGraphPtr ShareGraph::LoadMatmulElewiseBrcFusedGraph(bool is_dynamic) {
  auto builder = GraphBuilder("load_matmul_elewise_brc_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  af::AttrUtils::SetInt(data2->GetOpDescBarePtr(), "_parent_node_index", 2);
  auto data3 = builder.AddNode("data3", "Data", 0, 1);
  af::AttrUtils::SetInt(data3->GetOpDescBarePtr(), "_parent_node_index", 3);
  auto data4 = builder.AddNode("data4", "Data", 0, 1);
  af::AttrUtils::SetInt(data4->GetOpDescBarePtr(), "_parent_node_index", 4);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 5, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(data2, 0, ascbc, 2);
  builder.AddDataEdge(data3, 0, ascbc, 3);
  builder.AddDataEdge(data4, 0, ascbc, 4);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_matmul_elewise_brc_store");
  CreateMatmulElewiseBrcGraph(sub_graph, is_dynamic);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateBatchMatmulPrefix(const MatmulGraphContext &context, af::ascir_op::BatchMatMul &matmul) {
  af::ascir_op::Data data0("data0", context.graph);
  SetFullMatmulGraphLayout(data0, context, af::DT_FLOAT16);
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data0.ir_attr.SetIndex(0);
  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  SetFullMatmulGraphLayout(load0, context, af::DT_FLOAT16);

  af::ascir_op::Data data1("data1", context.graph);
  SetFullMatmulGraphLayout(data1, context, af::DT_FLOAT16);
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  data1.ir_attr.SetIndex(1);
  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  SetFullMatmulGraphLayout(load1, context, af::DT_FLOAT16);

  matmul.x1 = load0.y;
  matmul.x2 = load1.y;
  SetFullMatmulGraphLayout(matmul, context, af::DT_FLOAT);
  matmul.ir_attr.SetAdj_x1(1);
  matmul.ir_attr.SetAdj_x2(0);
  matmul.ir_attr.SetHas_relu(0);
  matmul.ir_attr.SetEnable_hf32(0);
  matmul.ir_attr.SetOffset_x(0);
}

static void CreateBatchMatmulElewiseBrcGraph(af::AscGraph &graph, bool is_dynamic) {
  const auto context = CreateMatmulGraphContext(graph, is_dynamic);
  af::ascir_op::BatchMatMul matmul("matmul");
  CreateBatchMatmulPrefix(context, matmul);

  af::ascir_op::Add add_op("add");
  add_op.x1 = matmul.y;
  SetFullMatmulGraphLayout(add_op, context, af::DT_FLOAT);
  ConnectAddBroadcast(context, add_op);

  af::ascir_op::Mul mul("mul");
  mul.x1 = add_op.y;
  SetFullMatmulGraphLayout(mul, context, af::DT_FLOAT);
  ConnectSigmoidBroadcast(context, mul);

  af::ascir_op::Sub sub("sub");
  sub.x1 = mul.y;
  SetFullMatmulGraphLayout(sub, context, af::DT_FLOAT);
  ConnectRsqrtBroadcast(context, sub);
  CreateMatmulGraphOutput(context, sub.y);
}

af::ComputeGraphPtr ShareGraph::LoadBatchMatmulElewiseBrcFusedGraph(bool is_dynamic) {
  auto builder = GraphBuilder("load_batch_matmul_elewise_brc_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  af::AttrUtils::SetInt(data2->GetOpDescBarePtr(), "_parent_node_index", 2);
  auto data3 = builder.AddNode("data3", "Data", 0, 1);
  af::AttrUtils::SetInt(data3->GetOpDescBarePtr(), "_parent_node_index", 3);
  auto data4 = builder.AddNode("data4", "Data", 0, 1);
  af::AttrUtils::SetInt(data4->GetOpDescBarePtr(), "_parent_node_index", 4);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 5, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(data2, 0, ascbc, 2);
  builder.AddDataEdge(data3, 0, ascbc, 3);
  builder.AddDataEdge(data4, 0, ascbc, 4);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_batch_matmul_elewise_brc_store");
  CreateBatchMatmulElewiseBrcGraph(sub_graph, is_dynamic);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *               where
 *          /        \     \
 *       Matmul      eq     \
 *      /    \     /    \    \
 *   data0 data1 data2 data3 scalar0
 */
static void CreateMatmulCompareScalarGraph(af::AscGraph &graph) {
  const auto context = CreateMatmulGraphContext(graph);
  af::ascir_op::MatMul matmul("matmul");
  CreateMatmulPrefix(context, matmul);
  af::ascir_op::Eq eq0("eq0");
  ConnectCompareInputs(context, eq0);

  af::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.ir_attr.SetValue("1");

  af::ascir_op::Where where0("where");
  where0.x1 = eq0.y;
  where0.x2 = matmul.y;
  where0.x3 = scalar0.y;
  SetFullMatmulGraphLayout(where0, context, af::DT_FLOAT);
  CreateMatmulGraphOutput(context, where0.y);
}

af::ComputeGraphPtr ShareGraph::LoadMatmulCompareScalarFusedGraph() {
  auto builder = GraphBuilder("load_matmul_compare_scalar_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  af::AttrUtils::SetInt(data2->GetOpDescBarePtr(), "_parent_node_index", 2);
  auto data3 = builder.AddNode("data3", "Data", 0, 1);
  af::AttrUtils::SetInt(data3->GetOpDescBarePtr(), "_parent_node_index", 3);
  auto data4 = builder.AddNode("data4", "Data", 0, 1);
  af::AttrUtils::SetInt(data4->GetOpDescBarePtr(), "_parent_node_index", 4);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 5, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(data2, 0, ascbc, 2);
  builder.AddDataEdge(data3, 0, ascbc, 3);
  builder.AddDataEdge(data4, 0, ascbc, 4);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_matmul_elewise_brc_store");
  CreateMatmulCompareScalarGraph(sub_graph);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateMatmulToIntCastGraph(af::AscGraph &graph) {
  const auto context = CreateMatmulGraphContext(graph);
  af::ascir_op::MatMul matmul("matmul");
  CreateMatmulPrefix(context, matmul);

  af::ascir_op::RoundToInt round_to_int("round_to_int");
  round_to_int.x = matmul.y;
  SetFullMatmulGraphLayout(round_to_int, context, af::DT_INT32);

  af::ascir_op::Cast cast0("cast0");
  cast0.x = round_to_int.y;
  SetFullMatmulGraphLayout(cast0, context, af::DT_FLOAT);

  af::ascir_op::TruncToInt trunc_to_int("trunc_to_int");
  trunc_to_int.x = cast0.y;
  SetFullMatmulGraphLayout(trunc_to_int, context, af::DT_INT32);

  af::ascir_op::Cast cast1("cast1");
  cast1.x = trunc_to_int.y;
  SetFullMatmulGraphLayout(cast1, context, af::DT_FLOAT);

  af::ascir_op::FloorToInt floor_to_int("floor_to_int");
  floor_to_int.x = cast1.y;
  SetFullMatmulGraphLayout(floor_to_int, context, af::DT_INT32);
  CreateMatmulGraphOutput(context, floor_to_int.y, af::DT_INT32);
}

af::ComputeGraphPtr ShareGraph::LoadMatmulToIntCastFusedGraph() {
  auto builder = GraphBuilder("load_matmul_to_int_cast_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("load_matmul_to_int_cast_store");
  CreateMatmulToIntCastGraph(sub_graph);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *        abs
 *         |
 *        div
 *       /   \
 *   load0  load1
 *      |     |
 *   data0  data1
 */
static void CreateDivAbsAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;

  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;

  af::ascir_op::Div div("div");
  div.x1 = x1Local.y;
  div.x2 = x2Local.y;

  af::ascir_op::Abs abs("abs");
  abs.x = div.y;

  af::ascir_op::Store x_out("store");
  x_out.x = abs.y;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::DivAbsFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("div_abs_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("div_abs");
  CreateDivAbsAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateTrueDivBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.y.dtype = af::DataType::DT_BF16;
  x1.ir_attr.SetIndex(0);

  af::ascir_op::Data x2("data1", graph);
  x2.y.dtype = af::DataType::DT_BF16;
  x2.ir_attr.SetIndex(1);

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::TrueDiv trueDiv("trueDiv");
  trueDiv.x1 = x1Local.y;
  trueDiv.x2 = x2Local.y;
  trueDiv.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Store x_out("store");
  x_out.x = trueDiv.y;
  x_out.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Output y("output");
  y.x = x_out.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

static void CreateTruedivAbsAscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.y.dtype = af::DataType::DT_FLOAT;
  x1.ir_attr.SetIndex(0);

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = af::DataType::DT_FLOAT;

  af::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.ir_attr.SetValue("1.2");

  af::ascir_op::Scalar scalar1("scalar1", graph);
  scalar1.ir_attr.SetValue("1.8");

  af::ascir_op::TrueDiv trueDiv0("trueDiv0");
  trueDiv0.x1 = x1Local.y;
  trueDiv0.x2 = scalar0.y;
  trueDiv0.y.dtype = af::DataType::DT_FLOAT;

  af::ascir_op::TrueDiv trueDiv1("trueDiv1");
  trueDiv1.x1 = scalar0.y;
  trueDiv1.x2 = scalar1.y;
  trueDiv1.y.dtype = af::DataType::DT_FLOAT;

  af::ascir_op::TrueDiv trueDiv2("trueDiv2");
  trueDiv2.x1 = trueDiv0.y;
  trueDiv2.x2 = trueDiv1.y;
  trueDiv2.y.dtype = af::DataType::DT_FLOAT;

  af::ascir_op::Store x_out("store");
  x_out.x = trueDiv2.y;
  x_out.y.dtype = af::DataType::DT_FLOAT;

  af::ascir_op::Output y("output");
  y.x = x_out.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::TrueDivBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("truediv_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("truediv_bf16");
  CreateTrueDivBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

af::ComputeGraphPtr ShareGraph::TruedivAbsFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("truediv_abs_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("truediv_abs");
  CreateTruedivAbsAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateTruncDivBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.y.dtype = af::DataType::DT_BF16;
  x1.ir_attr.SetIndex(0);

  af::ascir_op::Data x2("data1", graph);
  x2.y.dtype = af::DataType::DT_BF16;
  x2.ir_attr.SetIndex(1);

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::TruncDiv truncDiv("truncDiv");
  truncDiv.x1 = x1Local.y;
  truncDiv.x2 = x2Local.y;
  truncDiv.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Store x_out("store");
  x_out.x = truncDiv.y;
  x_out.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Output y("output");
  y.x = x_out.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::TruncDivBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("truncdiv_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("truncdiv_bf16");
  CreateTruncDivBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *                where
 *          /           \  \
 *        ge            add  \
 *      /     \      /      \  \
 *   data0 scalar0 scalar0  data1
 */
static void CreateCompareScalarWhereGraph(af::AscGraph &graph) {
  auto s0 = Symbol("s0");
  auto s1 = Symbol("s1");
  auto z0 = graph.CreateAxis("z0", Symbol("s0"));
  auto z1 = graph.CreateAxis("z1", Symbol("s1"));

  af::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = af::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0.y.repeats = {s0, s1};
  *data0.y.strides = {s1, af::ops::One};
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.y.dtype = af::DT_FLOAT16;
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, af::ops::One};

  af::ascir_op::Data data1("data1", graph);
  data1.attr.sched.axis = {z0.id, z1.id};
  data1.y.dtype = af::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data1.y.repeats = {s0, s1};
  *data1.y.strides = {s1, af::ops::One};
  data1.ir_attr.SetIndex(1);

  af::ascir_op::Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = af::DT_FLOAT16;
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, af::ops::One};

  af::ascir_op::Scalar scalar0("scalar0", graph);
  scalar0.ir_attr.SetValue("0.5");
  scalar0.y.dtype = af::DT_FLOAT16;

  af::ascir_op::Ge ge0("ge0");
  ge0.x1 = load0.y;
  ge0.x2 = scalar0.y;
  ge0.attr.sched.axis = {z0.id, z1.id};
  ge0.y.dtype = af::DT_UINT8;
  *ge0.y.axis = {z0.id, z1.id};
  *ge0.y.repeats = {s0, s1};
  *ge0.y.strides = {s1, af::ops::One};

  af::ascir_op::Add add0("add0");
  add0.x1 = load1.y;
  add0.x2 = scalar0.y;
  add0.attr.sched.axis = {z0.id, z1.id};
  add0.y.dtype = af::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id};
  *add0.y.repeats = {s0, s1};
  *add0.y.strides = {s1, af::ops::One};

  af::ascir_op::Where where0("where");
  where0.x1 = ge0.y;
  where0.x2 = add0.y;
  where0.x3 = load1.y;
  where0.attr.sched.axis = {z0.id, z1.id};
  where0.y.dtype = af::DT_FLOAT16;
  *where0.y.axis = {z0.id, z1.id};
  *where0.y.repeats = {s0, s1};
  *where0.y.strides = {s1, af::ops::One};

  af::ascir_op::Store store_op("store");
  store_op.x = where0.y;
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.y.dtype = af::DT_FLOAT16;
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1, af::ops::One};

  af::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = af::DT_FLOAT16;
  output_op.ir_attr.SetIndex(0);
}

af::ComputeGraphPtr ShareGraph::LoadCompareScalarWhereFusedGraph() {
  auto builder = GraphBuilder("load_compare_scalar_where_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("compare_scalar_where");
  CreateCompareScalarWhereGraph(sub_graph);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *            where
 *          /    \    \
 *        eq      \    \
 *      /    \     \    \
 *   data0 data1 data2 data3
 */
static void CreateCompareWhereGraph(af::AscGraph &graph) {
  auto s0 = Symbol("s0");
  auto s1 = Symbol("s1");
  auto z0 = graph.CreateAxis("z0", Symbol("s0"));
  auto z1 = graph.CreateAxis("z1", Symbol("s1"));

  af::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = af::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0.y.repeats = {s0, s1};
  *data0.y.strides = {s1, af::ops::One};
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.y.dtype = af::DT_FLOAT16;
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {s0, s1};
  *load0.y.strides = {s1, af::ops::One};

  af::ascir_op::Data data1("data1", graph);
  data1.attr.sched.axis = {z0.id, z1.id};
  data1.y.dtype = af::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data1.y.repeats = {s0, s1};
  *data1.y.strides = {s1, af::ops::One};
  data1.ir_attr.SetIndex(1);

  af::ascir_op::Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = af::DT_FLOAT16;
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {s0, s1};
  *load1.y.strides = {s1, af::ops::One};

  af::ascir_op::Data data2("data2", graph);
  data2.y.dtype = af::DT_FLOAT16;
  data2.attr.sched.axis = {z0.id, z1.id};
  *data2.y.axis = {z0.id, z1.id};
  data2.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data2.y.repeats = {s0, s1};
  *data2.y.strides = {s1, af::ops::One};
  data2.ir_attr.SetIndex(2);

  af::ascir_op::Load load2("load2");
  load2.x = data2.y;
  load2.attr.sched.axis = {z0.id, z1.id};
  load2.y.dtype = af::DT_FLOAT16;
  *load2.y.axis = {z0.id, z1.id};
  *load2.y.repeats = {s0, s1};
  *load2.y.strides = {s1, af::ops::One};

  af::ascir_op::Data data3("data3", graph);
  data3.attr.sched.axis = {z0.id, z1.id};
  data3.y.dtype = af::DT_FLOAT16;
  *data3.y.axis = {z0.id, z1.id};
  data3.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data3.y.repeats = {s0, s1};
  *data3.y.strides = {s1, af::ops::One};
  data3.ir_attr.SetIndex(3);

  af::ascir_op::Load load3("load3");
  load3.x = data3.y;
  load3.attr.sched.axis = {z0.id, z1.id};
  load3.y.dtype = af::DT_FLOAT16;
  *load3.y.axis = {z0.id, z1.id};
  *load3.y.repeats = {s0, s1};
  *load3.y.strides = {s1, af::ops::One};

  af::ascir_op::Eq eq0("eq0");
  eq0.x1 = load0.y;
  eq0.x2 = load1.y;
  eq0.attr.sched.axis = {z0.id, z1.id};
  eq0.y.dtype = af::DT_UINT8;
  *eq0.y.axis = {z0.id, z1.id};
  *eq0.y.repeats = {s0, s1};
  *eq0.y.strides = {s1, af::ops::One};

  af::ascir_op::Where where0("where");
  where0.x1 = eq0.y;
  where0.x2 = load2.y;
  where0.x3 = load3.y;
  where0.attr.sched.axis = {z0.id, z1.id};
  where0.y.dtype = af::DT_FLOAT16;
  *where0.y.axis = {z0.id, z1.id};
  *where0.y.repeats = {s0, s1};
  *where0.y.strides = {s1, af::ops::One};

  af::ascir_op::Store store_op("store");
  store_op.x = where0.y;
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.y.dtype = af::DT_FLOAT16;
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1, af::ops::One};

  af::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = af::DT_FLOAT16;
  output_op.ir_attr.SetIndex(0);
}

af::ComputeGraphPtr ShareGraph::LoadCompareWhereFusedGraph() {
  auto builder = GraphBuilder("load_compare_where_store_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);
  auto data2 = builder.AddNode("data2", "Data", 0, 1);
  af::AttrUtils::SetInt(data2->GetOpDescBarePtr(), "_parent_node_index", 2);
  auto data3 = builder.AddNode("data3", "Data", 0, 1);
  af::AttrUtils::SetInt(data3->GetOpDescBarePtr(), "_parent_node_index", 3);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 4, 1);
  af::AttrUtils::SetInt(ascbc->GetOpDescBarePtr(), "_parent_node_index", 4);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(data2, 0, ascbc, 2);
  builder.AddDataEdge(data3, 0, ascbc, 3);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("compare_where");
  CreateCompareWhereGraph(sub_graph);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

template <typename TensorT>
static void SetTensor2DInfo(TensorT &tensor, const std::vector<int64_t> &axis,
                            const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides,
                            af::DataType dtype) {
  tensor.dtype = dtype;
  *tensor.axis = axis;
  *tensor.repeats = repeats;
  *tensor.strides = strides;
}

template <typename NodeT>
static void SetNode2DInfo(NodeT &node, const std::vector<int64_t> &axis) {
  node.attr.sched.axis = axis;
}

struct AscGraph2DInfo {
  std::vector<int64_t> axis;
  std::vector<af::Expression> repeats;
  std::vector<af::Expression> strides;
};

static AscGraph2DInfo CreateAscGraph2DInfo(af::AscGraph &graph) {
  auto s0 = Symbol("s0");
  auto s1 = Symbol("s1");
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  return {{z0.id, z1.id}, {s0, s1}, {s1, af::ops::One}};
}

static void InitBoolMaskInput(af::ascir_op::Data &data, af::ascir_op::Load &load, const AscGraph2DInfo &info,
                              uint32_t index) {
  data.ir_attr.SetIndex(index);
  SetNode2DInfo(data, info.axis);
  SetTensor2DInfo(data.y, info.axis, info.repeats, info.strides, af::DT_FLOAT);
  load.x = data.y;
  SetNode2DInfo(load, info.axis);
  SetTensor2DInfo(load.y, info.axis, info.repeats, info.strides, af::DT_FLOAT);
}

static void CreateBoolMaskPredicates(af::AscGraph &graph, const AscGraph2DInfo &info, const af::ascir_op::Load &load0,
                                     const af::ascir_op::Load &load1, af::ascir_op::Gt &gt) {
  af::ascir_op::IsFinite isfinite("isfinite");
  isfinite.x = load0.y;
  SetNode2DInfo(isfinite, info.axis);
  SetTensor2DInfo(isfinite.y, info.axis, info.repeats, info.strides, af::DT_BOOL);

  af::ascir_op::Isnan isnan("isnan");
  isnan.x = load0.y;
  SetNode2DInfo(isnan, info.axis);
  SetTensor2DInfo(isnan.y, info.axis, info.repeats, info.strides, af::DT_BOOL);

  gt.x1 = load0.y;
  gt.x2 = load1.y;
  SetNode2DInfo(gt, info.axis);
  SetTensor2DInfo(gt.y, info.axis, info.repeats, info.strides, af::DT_BOOL);

  af::ascir_op::Cast cast("cast");
  cast.x = load0.y;
  SetNode2DInfo(cast, info.axis);
  SetTensor2DInfo(cast.y, info.axis, info.repeats, info.strides, af::DT_BOOL);

  af::ascir_op::Scalar scalar("scalar", graph);
  scalar.ir_attr.SetIndex(4);
  scalar.ir_attr.SetValue("1");
  scalar.y.dtype = af::DT_BOOL;

  af::ascir_op::Broadcast broadcast("broadcast");
  broadcast.x = scalar.y;
  SetNode2DInfo(broadcast, info.axis);
  SetTensor2DInfo(broadcast.y, info.axis, info.repeats, info.strides, af::DT_BOOL);
}

static void CreateBoolMaskOutput(const AscGraph2DInfo &info, const af::ascir_op::Gt &gt,
                                 const af::ascir_op::Load &load2, const af::ascir_op::Load &load3) {
  af::ascir_op::Where where("where");
  where.x1 = gt.y;
  where.x2 = load2.y;
  where.x3 = load3.y;
  SetNode2DInfo(where, info.axis);
  SetTensor2DInfo(where.y, info.axis, info.repeats, info.strides, af::DT_FLOAT);

  af::ascir_op::Store store("store");
  store.x = where.y;
  SetNode2DInfo(store, info.axis);
  SetTensor2DInfo(store.y, info.axis, info.repeats, info.strides, af::DT_FLOAT);

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  output.y.dtype = af::DT_FLOAT;
}

/**
 *               output
 *                  |
 *                store
 *                  |
 *                where
 *              /   |   \
 *            gt  load2 load3
 *           /  \   |     |
 *        load0 load1 data2 data3
 *          |
 *        data0
 *
 *   load0 also feeds IsFinite, Isnan and Cast. Scalar(bool) feeds Broadcast
 *   to cover DT_BOOL scalar/broadcast codegen paths in the same AscGraph.
 */
static void CreateBoolMaskFlowGraph(af::AscGraph &graph) {
  auto info = CreateAscGraph2DInfo(graph);
  af::ascir_op::Data data0("data0", graph), data1("data1", graph), data2("data2", graph), data3("data3", graph);
  af::ascir_op::Load load0("load0"), load1("load1"), load2("load2"), load3("load3");
  InitBoolMaskInput(data0, load0, info, 0);
  InitBoolMaskInput(data1, load1, info, 1);
  InitBoolMaskInput(data2, load2, info, 2);
  InitBoolMaskInput(data3, load3, info, 3);

  af::ascir_op::Gt gt("gt");
  CreateBoolMaskPredicates(graph, info, load0, load1, gt);
  CreateBoolMaskOutput(info, gt, load2, load3);
}

static af::ComputeGraphPtr BuildFourInputAscGraph(const std::string &graph_name, const std::string &sub_graph_name,
                                                  void (*create_sub_graph)(af::AscGraph &)) {
  auto builder = GraphBuilder(graph_name);
  std::vector<NodePtr> data_nodes;
  data_nodes.reserve(4);
  for (int32_t index = 0; index < 4; ++index) {
    auto data = builder.AddNode("data" + std::to_string(index), "Data", 0, 1);
    af::AttrUtils::SetInt(data->GetOpDescBarePtr(), "_parent_node_index", index);
    data_nodes.push_back(data);
  }

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 4, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);
  for (int32_t index = 0; index < 4; ++index) {
    builder.AddDataEdge(data_nodes[index], 0, ascbc, index);
  }
  builder.AddDataEdge(ascbc, 0, netoutput, 0);

  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }

  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph(sub_graph_name.c_str());
  create_sub_graph(sub_graph);
  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

af::ComputeGraphPtr ShareGraph::BoolMaskFlowFusedGraph() {
  return BuildFourInputAscGraph("bool_mask_flow_test", "bool_mask_flow", CreateBoolMaskFlowGraph);
}

/**
 *            output
 *               |
 *             store
 *               |
 *            select
 *          /   |   \
 *        gt  load2 load3
 *       /  \   |     |
 *    load0 load1 data2 data3
 *      |     |
 *    data0 data1
 *
 *   Gt produces DT_BOOL mask, Select uses it as condition.
 *   Select now shares WhereAscIrCodegenImplV2 codegen path.
 */
static void CreateSelectBoolGraph(af::AscGraph &graph) {
  auto info = CreateAscGraph2DInfo(graph);
  af::ascir_op::Data data0("data0", graph), data1("data1", graph), data2("data2", graph), data3("data3", graph);
  af::ascir_op::Load load0("load0"), load1("load1"), load2("load2"), load3("load3");
  InitBoolMaskInput(data0, load0, info, 0);
  InitBoolMaskInput(data1, load1, info, 1);
  InitBoolMaskInput(data2, load2, info, 2);
  InitBoolMaskInput(data3, load3, info, 3);

  af::ascir_op::Gt gt("gt");
  gt.x1 = load0.y;
  gt.x2 = load1.y;
  SetNode2DInfo(gt, info.axis);
  SetTensor2DInfo(gt.y, info.axis, info.repeats, info.strides, af::DT_BOOL);

  af::ascir_op::Select select("select");
  select.x1 = gt.y;
  select.x2 = load2.y;
  select.x3 = load3.y;
  SetNode2DInfo(select, info.axis);
  SetTensor2DInfo(select.y, info.axis, info.repeats, info.strides, af::DT_FLOAT);

  af::ascir_op::Store store("store");
  store.x = select.y;
  SetNode2DInfo(store, info.axis);
  SetTensor2DInfo(store.y, info.axis, info.repeats, info.strides, af::DT_FLOAT);

  af::ascir_op::Output output("output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  output.y.dtype = af::DT_FLOAT;
}

af::ComputeGraphPtr ShareGraph::SelectBoolFusedGraph() {
  return BuildFourInputAscGraph("select_bool_test", "select_bool", CreateSelectBoolGraph);
}

/**
 *      output
 *         |
 *       store
 *         |
 *       acos
 *         |
 *      load0
 *         |
 *      data0
 */
static void CreateAcosFloatAscGraph(af::AscGraph &graph, size_t dims_size) {
  // Data 节点
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);
  x.y.dtype = af::DT_FLOAT;

  // Load 节点
  af::ascir_op::Load xLocal("load0");
  xLocal.x = x.y;
  xLocal.y.dtype = af::DT_FLOAT;

  // Acos 操作
  af::ascir_op::Acos acos("acos");
  acos.x = xLocal.y;
  acos.y.dtype = af::DT_FLOAT;

  // Store 节点
  af::ascir_op::Store x_out("store");
  x_out.x = acos.y;
  x_out.y.dtype = af::DT_FLOAT;

  // Output 节点
  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = af::DT_FLOAT;

  // 设置维度信息
  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *     NetOutput
 *         |
 *       AscBc
 *         |
 *       data0
 */
af::ComputeGraphPtr ShareGraph::AcosFloatFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("acos_float_test");

  // 创建 data 节点
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  // 创建 AscGraph 节点
  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  // 连接边
  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);

  // 获取计算图
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }

  // 创建并序列化子图
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("acos_float");
  CreateAcosFloatAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);

  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *       acos
 *         |
 *      load0
 *         |
 *      data0
 */
static void CreateAcosBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  // Data 节点
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);
  x.y.dtype = af::DT_BF16;

  // Load 节点
  af::ascir_op::Load xLocal("load0");
  xLocal.x = x.y;
  xLocal.y.dtype = af::DT_BF16;

  // Acos 操作
  af::ascir_op::Acos acos("acos");
  acos.x = xLocal.y;
  acos.y.dtype = af::DT_BF16;

  // Store 节点
  af::ascir_op::Store x_out("store");
  x_out.x = acos.y;
  x_out.y.dtype = af::DT_BF16;

  // Output 节点
  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = af::DT_BF16;

  // 设置维度信息
  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *     NetOutput
 *         |
 *       AscBc
 *         |
 *       data0
 */
af::ComputeGraphPtr ShareGraph::AcosBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("acos_bf16_test");

  // 创建 data 节点
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  // 创建 AscGraph 节点
  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  // 连接边
  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);

  // 获取计算图
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }

  // 创建并序列化子图
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("acos_bf16");
  CreateAcosBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);

  return compute_graph;
}

/**
 *                                         add
 *                             /                         \
 *                          add                            \
 *                 /                   \                     \
 *              add                    add                    add
 *          /         \            /         \             /        \
 *        add        mul       maximum     minimum   logical_and   logical_and
 *      /    \     /     \     /     \     /     \     /     \      /     \
 *   data0 data1 data0 data1 data0 data1 data0 data1 data0  data1 data0  data1
 */
static void CreateBinaryApiScalarGraph(af::AscGraph &graph) {
  auto s0 = Symbol("s0");
  auto s1 = Symbol("s1");
  auto z0 = graph.CreateAxis("z0", Symbol("s0"));
  auto z1 = graph.CreateAxis("z1", Symbol("s1"));

  af::ascir_op::Data data0("data0", graph);
  data0.attr.sched.axis = {z0.id, z1.id};
  data0.y.dtype = af::DT_FLOAT16;
  *data0.y.axis = {z0.id, z1.id};
  data0.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data0.y.repeats = {af::ops::One, af::ops::One};
  *data0.y.strides = {af::ops::Zero, af::ops::Zero};
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load0("load0");
  load0.attr.sched.axis = {z0.id, z1.id};
  load0.y.dtype = af::DT_FLOAT16;
  load0.x = data0.y;
  *load0.y.axis = {z0.id, z1.id};
  *load0.y.repeats = {af::ops::One, af::ops::One};
  *load0.y.strides = {af::ops::Zero, af::ops::Zero};

  af::ascir_op::Data data1("data1", graph);
  data1.attr.sched.axis = {z0.id, z1.id};
  data1.y.dtype = af::DT_FLOAT16;
  *data1.y.axis = {z0.id, z1.id};
  data1.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  *data1.y.repeats = {af::ops::One, af::ops::One};
  *data1.y.strides = {af::ops::Zero, af::ops::Zero};
  data1.ir_attr.SetIndex(1);

  af::ascir_op::Load load1("load1");
  load1.attr.sched.axis = {z0.id, z1.id};
  load1.y.dtype = af::DT_FLOAT16;
  load1.x = data1.y;
  *load1.y.axis = {z0.id, z1.id};
  *load1.y.repeats = {af::ops::One, af::ops::One};
  *load1.y.strides = {af::ops::Zero, af::ops::Zero};

  af::ascir_op::Add add0("add0");
  add0.x1 = load0.y;
  add0.x2 = load1.y;
  add0.attr.sched.axis = {z0.id, z1.id};
  add0.y.dtype = af::DT_FLOAT16;
  *add0.y.axis = {z0.id, z1.id};
  *add0.y.repeats = {s0, s1};
  *add0.y.strides = {s1, af::ops::One};

  af::ascir_op::Mul mul0("mul0");
  mul0.x1 = load0.y;
  mul0.x2 = load1.y;
  mul0.attr.sched.axis = {z0.id, z1.id};
  mul0.y.dtype = af::DT_FLOAT16;
  *mul0.y.axis = {z0.id, z1.id};
  *mul0.y.repeats = {s0, s1};
  *mul0.y.strides = {s1, af::ops::One};

  af::ascir_op::Maximum maximum0("maximum0");
  maximum0.x1 = load0.y;
  maximum0.x2 = load1.y;
  maximum0.attr.sched.axis = {z0.id, z1.id};
  maximum0.y.dtype = af::DT_FLOAT16;
  *maximum0.y.axis = {z0.id, z1.id};
  *maximum0.y.repeats = {s0, s1};
  *maximum0.y.strides = {s1, af::ops::One};

  af::ascir_op::Minimum minimum0("minimum0");
  minimum0.x1 = load0.y;
  minimum0.x2 = load1.y;
  minimum0.attr.sched.axis = {z0.id, z1.id};
  minimum0.y.dtype = af::DT_FLOAT16;
  *minimum0.y.axis = {z0.id, z1.id};
  *minimum0.y.repeats = {s0, s1};
  *minimum0.y.strides = {s1, af::ops::One};

  af::ascir_op::LogicalAnd logical_and0("logical_and0");
  logical_and0.x1 = load0.y;
  logical_and0.x2 = load1.y;
  logical_and0.attr.sched.axis = {z0.id, z1.id};
  logical_and0.y.dtype = af::DT_UINT8;
  *logical_and0.y.axis = {z0.id, z1.id};
  *logical_and0.y.repeats = {s0, s1};
  *logical_and0.y.strides = {s1, af::ops::One};

  af::ascir_op::Cast cast0("cast0");
  cast0.x = logical_and0.y;
  cast0.attr.sched.axis = {z0.id, z1.id};
  cast0.y.dtype = af::DT_FLOAT16;
  *cast0.y.axis = {z0.id, z1.id};
  *cast0.y.repeats = {s0, s1};
  *cast0.y.strides = {s1, af::ops::One};

  af::ascir_op::LogicalOr logical_or0("logical_or0");
  logical_or0.x1 = load0.y;
  logical_or0.x2 = load1.y;
  logical_or0.attr.sched.axis = {z0.id, z1.id};
  logical_or0.y.dtype = af::DT_UINT8;
  *logical_or0.y.axis = {z0.id, z1.id};
  *logical_or0.y.repeats = {s0, s1};
  *logical_or0.y.strides = {s1, af::ops::One};

  af::ascir_op::Cast cast1("cast1");
  cast1.x = logical_or0.y;
  cast1.attr.sched.axis = {z0.id, z1.id};
  cast1.y.dtype = af::DT_FLOAT16;
  *cast1.y.axis = {z0.id, z1.id};
  *cast1.y.repeats = {s0, s1};
  *cast1.y.strides = {s1, af::ops::One};

  af::ascir_op::Add add1("add1");
  add1.x1 = add0.y;
  add1.x2 = mul0.y;
  add1.attr.sched.axis = {z0.id, z1.id};
  add1.y.dtype = af::DT_FLOAT16;
  *add1.y.axis = {z0.id, z1.id};
  *add1.y.repeats = {s0, s1};
  *add1.y.strides = {s1, af::ops::One};

  af::ascir_op::Add add2("add2");
  add2.x1 = maximum0.y;
  add2.x2 = minimum0.y;
  add2.attr.sched.axis = {z0.id, z1.id};
  add2.y.dtype = af::DT_FLOAT16;
  *add2.y.axis = {z0.id, z1.id};
  *add2.y.repeats = {s0, s1};
  *add2.y.strides = {s1, af::ops::One};

  af::ascir_op::Add add3("add3");
  add3.x1 = cast0.y;
  add3.x2 = cast1.y;
  add3.attr.sched.axis = {z0.id, z1.id};
  add3.y.dtype = af::DT_FLOAT16;
  *add3.y.axis = {z0.id, z1.id};
  *add3.y.repeats = {s0, s1};
  *add3.y.strides = {s1, af::ops::One};

  af::ascir_op::Add add4("add4");
  add4.x1 = add1.y;
  add4.x2 = add2.y;
  add4.attr.sched.axis = {z0.id, z1.id};
  add4.y.dtype = af::DT_FLOAT16;
  *add4.y.axis = {z0.id, z1.id};
  *add4.y.repeats = {s0, s1};
  *add4.y.strides = {s1, af::ops::One};

  af::ascir_op::Add add5("add5");
  add5.x1 = add3.y;
  add5.x2 = add4.y;
  add5.attr.sched.axis = {z0.id, z1.id};
  add5.y.dtype = af::DT_FLOAT16;
  *add5.y.axis = {z0.id, z1.id};
  *add5.y.repeats = {s0, s1};
  *add5.y.strides = {s1, af::ops::One};

  af::ascir_op::Store store_op("store");
  store_op.x = add5.y;
  store_op.attr.sched.axis = {z0.id, z1.id};
  store_op.y.dtype = af::DT_FLOAT16;
  *store_op.y.axis = {z0.id, z1.id};
  *store_op.y.repeats = {s0, s1};
  *store_op.y.strides = {s1, af::ops::One};

  af::ascir_op::Output output_op("output");
  output_op.x = store_op.y;
  output_op.y.dtype = af::DT_FLOAT16;
  output_op.ir_attr.SetIndex(0);
}

af::ComputeGraphPtr ShareGraph::BinaryApiScalarFusedGraph() {
  auto builder = GraphBuilder("binary_api_scalar_test");

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 0, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("binary_api_scalar");
  CreateBinaryApiScalarGraph(sub_graph);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateAcoshBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data0", graph);
  x.y.dtype = af::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load x1Local("load0");
  x1Local.y.dtype = af::DataType::DT_BF16;
  x1Local.x = x.y;

  // Acosh 操作
  af::ascir_op::Acosh acosh("acosh");
  acosh.x = x1Local.y;
  acosh.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Store acosh_store("store");
  acosh_store.x = acosh.y;
  acosh_store.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Output y("output");
  y.x = acosh_store.y;

  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::AcoshBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("acosh_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("acosh_bf16_test");
  CreateAcoshBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateAsinBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  // Data 节点
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);
  x.y.dtype = af::DataType::DT_BF16;

  // Load 节点
  af::ascir_op::Load xLocal("load0");
  xLocal.x = x.y;
  xLocal.y.dtype = af::DataType::DT_BF16;

  // Cast 节点 (BF16 -> FLOAT)
  af::ascir_op::Cast cast("cast");
  cast.x = xLocal.y;
  cast.y.dtype = af::DataType::DT_FLOAT;

  // Asin 操作
  af::ascir_op::Asin asin("asin");
  asin.x = cast.y;
  asin.y.dtype = af::DataType::DT_FLOAT;

  // Cast 节点 (FLOAT -> BF16)
  af::ascir_op::Cast castBack("cast_back");
  castBack.x = asin.y;
  castBack.y.dtype = af::DataType::DT_BF16;

  // Store 节点
  af::ascir_op::Store x_out("store");
  x_out.x = castBack.y;
  x_out.y.dtype = af::DataType::DT_BF16;

  // Output 节点
  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = af::DataType::DT_BF16;

  // 设置维度信息
  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::AsinBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("asin_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("asin_bf16_test");
  CreateAsinBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateAsinhBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  // Data 节点
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);
  x.y.dtype = af::DataType::DT_BF16;

  // Load 节点
  af::ascir_op::Load xLocal("load0");
  xLocal.x = x.y;
  xLocal.y.dtype = af::DataType::DT_BF16;

  // Cast 节点 (BF16 -> FLOAT)
  af::ascir_op::Cast cast("cast");
  cast.x = xLocal.y;
  cast.y.dtype = af::DataType::DT_FLOAT;

  // Asinh 操作
  af::ascir_op::Asinh asinh("asinh");
  asinh.x = cast.y;
  asinh.y.dtype = af::DataType::DT_FLOAT;

  // Cast 节点 (FLOAT -> BF16)
  af::ascir_op::Cast castBack("cast_back");
  castBack.x = asinh.y;
  castBack.y.dtype = af::DataType::DT_BF16;

  // Store 节点
  af::ascir_op::Store x_out("store");
  x_out.x = castBack.y;
  x_out.y.dtype = af::DataType::DT_BF16;

  // Output 节点
  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = af::DataType::DT_BF16;

  // 设置维度信息
  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::AsinhBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("asinh_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("asinh_bf16_test");
  CreateAsinhBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateAtanBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  // Data 节点
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);
  x.y.dtype = af::DataType::DT_BF16;

  // Load 节点
  af::ascir_op::Load xLocal("load0");
  xLocal.x = x.y;
  xLocal.y.dtype = af::DataType::DT_BF16;

  // Atan 操作
  af::ascir_op::Atan atan("atan");
  atan.x = xLocal.y;
  atan.y.dtype = af::DataType::DT_BF16;

  // Store 节点
  af::ascir_op::Store x_out("store");
  x_out.x = atan.y;
  x_out.y.dtype = af::DataType::DT_BF16;

  // Output 节点
  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = af::DataType::DT_BF16;

  // 设置维度信息
  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::AtanBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("atan_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("atan_bf16_test");
  CreateAtanBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateAtanhBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  // Data 节点
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);
  x.y.dtype = af::DataType::DT_BF16;

  // Load 节点
  af::ascir_op::Load xLocal("load0");
  xLocal.x = x.y;
  xLocal.y.dtype = af::DataType::DT_BF16;

  // Cast 节点 (BF16 -> FLOAT)
  af::ascir_op::Cast cast("cast");
  cast.x = xLocal.y;
  cast.y.dtype = af::DataType::DT_FLOAT;

  // Atanh 操作
  af::ascir_op::Atanh atanh("atanh");
  atanh.x = cast.y;
  atanh.y.dtype = af::DataType::DT_FLOAT;

  // Cast 节点 (FLOAT -> BF16)
  af::ascir_op::Cast castBack("cast_back");
  castBack.x = atanh.y;
  castBack.y.dtype = af::DataType::DT_BF16;

  // Store 节点
  af::ascir_op::Store x_out("store");
  x_out.x = castBack.y;
  x_out.y.dtype = af::DataType::DT_BF16;

  // Output 节点
  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = af::DataType::DT_BF16;

  // 设置维度信息
  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::AtanhBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("atanh_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("atanh_bf16_test");
  CreateAtanhBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateCoshBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  // Data 节点
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);
  x.y.dtype = af::DataType::DT_BF16;

  // Load 节点
  af::ascir_op::Load xLocal("load0");
  xLocal.x = x.y;
  xLocal.y.dtype = af::DataType::DT_BF16;

  // Cast 节点 (BF16 -> FLOAT)
  af::ascir_op::Cast cast("cast");
  cast.x = xLocal.y;
  cast.y.dtype = af::DataType::DT_FLOAT;

  // Cosh 操作
  af::ascir_op::Cosh cosh("cosh");
  cosh.x = cast.y;
  cosh.y.dtype = af::DataType::DT_FLOAT;

  // Cast 节点 (FLOAT -> BF16)
  af::ascir_op::Cast castBack("cast_back");
  castBack.x = cosh.y;
  castBack.y.dtype = af::DataType::DT_BF16;

  // Store 节点
  af::ascir_op::Store x_out("store");
  x_out.x = castBack.y;
  x_out.y.dtype = af::DataType::DT_BF16;

  // Output 节点
  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = af::DataType::DT_BF16;

  // 设置维度信息
  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::CoshBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("cosh_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("cosh_bf16_test");
  CreateCoshBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateDigammaBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  // Data 节点
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);
  x.y.dtype = af::DataType::DT_BF16;

  // Load 节点
  af::ascir_op::Load xLocal("load0");
  xLocal.x = x.y;
  xLocal.y.dtype = af::DataType::DT_BF16;

  // Cast 节点 (BF16 -> FLOAT)
  af::ascir_op::Cast cast("cast");
  cast.x = xLocal.y;
  cast.y.dtype = af::DataType::DT_FLOAT;

  // Digamma 操作
  af::ascir_op::Digamma digamma("digamma");
  digamma.x = cast.y;
  digamma.y.dtype = af::DataType::DT_FLOAT;

  // Cast 节点 (FLOAT -> BF16)
  af::ascir_op::Cast castBack("cast_back");
  castBack.x = digamma.y;
  castBack.y.dtype = af::DataType::DT_BF16;

  // Store 节点
  af::ascir_op::Store x_out("store");
  x_out.x = castBack.y;
  x_out.y.dtype = af::DataType::DT_BF16;

  // Output 节点
  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = af::DataType::DT_BF16;

  // 设置维度信息
  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::DigammaBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("digamma_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("digamma_bf16_test");
  CreateDigammaBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateErfcBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  // Data 节点
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);
  x.y.dtype = af::DataType::DT_BF16;

  // Load 节点
  af::ascir_op::Load xLocal("load0");
  xLocal.x = x.y;
  xLocal.y.dtype = af::DataType::DT_BF16;

  // Cast 节点 (BF16 -> FLOAT)
  af::ascir_op::Cast cast("cast");
  cast.x = xLocal.y;
  cast.y.dtype = af::DataType::DT_FLOAT;

  // Erfc 操作
  af::ascir_op::Erfc erfc("erfc");
  erfc.x = cast.y;
  erfc.y.dtype = af::DataType::DT_FLOAT;

  // Cast 节点 (FLOAT -> BF16)
  af::ascir_op::Cast castBack("cast_back");
  castBack.x = erfc.y;
  castBack.y.dtype = af::DataType::DT_BF16;

  // Store 节点
  af::ascir_op::Store x_out("store");
  x_out.x = castBack.y;
  x_out.y.dtype = af::DataType::DT_BF16;

  // Output 节点
  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = af::DataType::DT_BF16;

  // 设置维度信息
  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::ErfcBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("erfc_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("erfc_bf16_test");
  CreateErfcBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}
/**
 *      output
 *         |
 *       store
 *         |
 *        pow(bf16)
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
static void CreatePowBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = af::DT_BF16;
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = af::DT_BF16;

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = af::DT_BF16;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = af::DT_BF16;

  af::ascir_op::Pow pow("pow");
  pow.x1 = x1Local.y;
  pow.x2 = x2Local.y;
  pow.y.dtype = af::DT_BF16;

  af::ascir_op::Store x_out("store");
  x_out.x = pow.y;
  x_out.y.dtype = af::DT_BF16;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

static void CreateTanBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  // Data 节点
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);
  x.y.dtype = af::DataType::DT_BF16;

  // Load 节点
  af::ascir_op::Load xLocal("load0");
  xLocal.x = x.y;
  xLocal.y.dtype = af::DataType::DT_BF16;

  // Tan 操作
  af::ascir_op::Tan tan("tan");
  tan.x = xLocal.y;
  tan.y.dtype = af::DataType::DT_BF16;

  // Store 节点
  af::ascir_op::Store x_out("store");
  x_out.x = tan.y;
  x_out.y.dtype = af::DataType::DT_BF16;

  // Output 节点
  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = af::DataType::DT_BF16;

  // 设置维度信息
  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::TanBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("tan_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("tan_bf16_test");
  CreateTanBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateSquareUint8AscGraph(af::AscGraph &graph, size_t dims_size) {
  // Data 节点
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);
  x.y.dtype = af::DataType::DT_UINT8;

  // Load 节点
  af::ascir_op::Load xLocal("load0");
  xLocal.x = x.y;
  xLocal.y.dtype = af::DataType::DT_UINT8;

  // Square 操作
  af::ascir_op::Square square("square");
  square.x = xLocal.y;
  square.y.dtype = af::DataType::DT_UINT8;

  // Store 节点
  af::ascir_op::Store x_out("store");
  x_out.x = square.y;
  x_out.y.dtype = af::DataType::DT_UINT8;

  // Output 节点
  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = af::DataType::DT_UINT8;

  // 设置维度信息
  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::SquareUint8FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("square_uint8_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("square_uint8_test");
  CreateSquareUint8AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateXorUint8AscGraph(af::AscGraph &graph, size_t dims_size) {
  // Data 节点
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = af::DataType::DT_UINT8;

  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = af::DataType::DT_UINT8;

  // Load 节点
  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = af::DataType::DT_UINT8;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = af::DataType::DT_UINT8;

  // Xor 操作
  af::ascir_op::Xor xor_op("xor");
  xor_op.x1 = x1Local.y;
  xor_op.x2 = x2Local.y;
  xor_op.y.dtype = af::DataType::DT_UINT8;

  // Store 节点
  af::ascir_op::Store x_out("store");
  x_out.x = xor_op.y;
  x_out.y.dtype = af::DataType::DT_UINT8;

  // Output 节点
  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = af::DataType::DT_UINT8;

  // 设置维度信息
  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::XorUint8FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("xor_uint8_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("xor_uint8_test");
  CreateXorUint8AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateSinhBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  // Data 节点
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);
  x.y.dtype = af::DataType::DT_BF16;

  // Load 节点
  af::ascir_op::Load xLocal("load0");
  xLocal.x = x.y;
  xLocal.y.dtype = af::DataType::DT_BF16;

  // Sinh 操作
  af::ascir_op::Sinh sinh("sinh");
  sinh.x = xLocal.y;
  sinh.y.dtype = af::DataType::DT_BF16;

  // Store 节点
  af::ascir_op::Store x_out("store");
  x_out.x = sinh.y;
  x_out.y.dtype = af::DataType::DT_BF16;

  // Output 节点
  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = af::DataType::DT_BF16;

  // 设置维度信息
  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::SinhBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("sinh_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("sinh_bf16_test");
  CreateSinhBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::PowBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("pow_bf16_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("pow_bf16");
  CreatePowBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *        data
 *         |
 *        load
 *         |
 *       reciprocal
 *         |
 *       store
 *         |
 *      output
 */
static void CreateReciprocalBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data", graph);
  x.y.dtype = af::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load xLocal("load");
  xLocal.x = x.y;
  xLocal.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Reciprocal reciprocal("reciprocal");
  reciprocal.x = xLocal.y;
  reciprocal.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Store x_out("store");
  x_out.x = reciprocal.y;
  x_out.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *         data
 *          |
 *        load
 *          |
 *        Round
 *          |
 *        store
 *          |
 *       output
 */
static void CreateRoundBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data", graph);
  x.y.dtype = af::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load xLocal("load");
  xLocal.x = x.y;
  xLocal.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Round round("round");
  round.x = xLocal.y;
  round.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Store x_out("store");
  x_out.x = round.y;
  x_out.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0  data1
 */
af::ComputeGraphPtr ShareGraph::ReciprocalBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("reciprocal_bf16_test");
  auto data = builder.AddNode("data", "Data", 0, 1);
  af::AttrUtils::SetInt(data->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("reciprocal_bf16_test");
  CreateReciprocalBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *         |
 *       data
 */
af::ComputeGraphPtr ShareGraph::RoundBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("round_bf16_test");
  auto data = builder.AddNode("data", "Data", 0, 1);
  af::AttrUtils::SetInt(data->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("round_bf16_test");
  CreateRoundBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *         data
 *          |
 *        load
 *          |
 *        Relu
 *          |
 *        store
 *          |
 *       output
 */
static void CreateReluUint8AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data", graph);
  x.y.dtype = af::DataType::DT_UINT8;
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load xLocal("load");
  xLocal.x = x.y;
  xLocal.y.dtype = af::DataType::DT_UINT8;

  af::ascir_op::Relu relu("relu");
  relu.x = xLocal.y;
  relu.y.dtype = af::DataType::DT_UINT8;

  af::ascir_op::Store x_out("store");
  x_out.x = relu.y;
  x_out.y.dtype = af::DataType::DT_UINT8;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *         |
 *       data
 */
af::ComputeGraphPtr ShareGraph::ReluUint8FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("relu_uint8_test");
  auto data = builder.AddNode("data", "Data", 0, 1);
  af::AttrUtils::SetInt(data->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("relu_uint8_test");
  CreateReluUint8AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateRshiftUint8AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = af::DataType::DT_UINT8;
  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = af::DataType::DT_INT8;

  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = af::DataType::DT_UINT8;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = af::DataType::DT_INT8;

  af::ascir_op::RShift rshift("rshift");
  rshift.x1 = x1Local.y;
  rshift.x2 = x2Local.y;
  rshift.y.dtype = af::DataType::DT_UINT8;

  af::ascir_op::Store x_out("store");
  x_out.x = rshift.y;
  x_out.y.dtype = af::DataType::DT_UINT8;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::RshiftUint8FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("rshift_uint8_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("rshift_uint8");
  CreateRshiftUint8AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateSignUint8AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data", graph);
  x.y.dtype = af::DataType::DT_UINT8;
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load xLocal("load");
  xLocal.x = x.y;
  xLocal.y.dtype = af::DataType::DT_UINT8;

  af::ascir_op::Sign sign("sign");
  sign.x = xLocal.y;
  sign.y.dtype = af::DataType::DT_UINT8;

  af::ascir_op::Store x_out("store");
  x_out.x = sign.y;
  x_out.y.dtype = af::DataType::DT_UINT8;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::SignUint8FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("sign_uint8_test");
  auto data = builder.AddNode("data", "Data", 0, 1);
  af::AttrUtils::SetInt(data->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("sign_uint8_test");
  CreateSignUint8AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateSignBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data", graph);
  x.y.dtype = af::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load xLocal("load");
  xLocal.x = x.y;
  xLocal.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Sign sign("sign");
  sign.x = xLocal.y;
  sign.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Store x_out("store");
  x_out.x = sign.y;
  x_out.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::SignBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("sign_bf16_test");
  auto data = builder.AddNode("data", "Data", 0, 1);
  af::AttrUtils::SetInt(data->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("sign_bf16_test");
  CreateSignBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateAtan2Bf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  // Data 节点
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = af::DT_BF16;

  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = af::DT_BF16;

  // Load 节点
  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = af::DT_BF16;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = af::DT_BF16;

  // Atan2 操作
  af::ascir_op::Atan2 atan2("atan2");
  atan2.x1 = x1Local.y;
  atan2.x2 = x2Local.y;
  atan2.y.dtype = af::DT_BF16;

  // Store 节点
  af::ascir_op::Store x_out("store");
  x_out.x = atan2.y;
  x_out.y.dtype = af::DT_BF16;

  // Output 节点
  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = af::DT_BF16;

  // 设置维度信息
  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      output
 *         |
 *       store
 *         |
 *      atan2
 *       /   \
 *   load0   load1
 *     |       |
 *   data0   data1
 */
af::ComputeGraphPtr ShareGraph::Atan2Bf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("atan2_bf16_test");

  // 创建 data 节点
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  // 创建 AscGraph 节点
  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  // 连接边
  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);

  // 获取计算图
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  // 创建并序列化子图
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("atan2_bf16");
  CreateAtan2Bf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

/**
 *      output
 *         |
 *       store
 *         |
 *    copysign
 *      /     \
 *  load0   load1
 *    |        |
 *  data0   data1
 */
static void CreateCopysignBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  // Data 节点
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = af::DT_BF16;

  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = af::DT_BF16;

  // Load 节点
  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = af::DT_BF16;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = af::DT_BF16;

  // CopySign 操作
  af::ascir_op::CopySign copysign("copysign");
  copysign.x1 = x1Local.y;
  copysign.x2 = x2Local.y;
  copysign.y.dtype = af::DT_BF16;

  // Store 节点
  af::ascir_op::Store x_out("store");
  x_out.x = copysign.y;
  x_out.y.dtype = af::DT_BF16;

  // Output 节点
  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = af::DT_BF16;

  // 设置维度信息
  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::CopysignBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("copysign_bf16_test");

  // 创建 data 节点
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  // 创建 AscGraph 节点
  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  // 连接边
  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);

  // 获取计算图
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  // 创建并序列化子图
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("copysign_bf16");
  CreateCopysignBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateCeil2intBf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data", graph);
  x.y.dtype = af::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load xLocal("load");
  xLocal.x = x.y;
  xLocal.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Ceil2Int ceil2int("ceil2int");
  ceil2int.x = xLocal.y;
  ceil2int.y.dtype = af::DataType::DT_INT32;

  af::ascir_op::Store x_out("store");
  x_out.x = ceil2int.y;
  x_out.y.dtype = af::DataType::DT_INT32;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = af::DataType::DT_INT32;

  // 设置维度信息
  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::Ceil2intBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("ceil2int_bf16_test");

  // 创建 data 节点
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  // 创建 AscGraph 节点
  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  // 连接边
  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);

  // 获取计算图
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  // 创建并序列化子图
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("ceil2int_bf16");
  CreateCeil2intBf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateErfcxTestAscGraph(af::AscGraph &graph, size_t dims_size, af::DataType dtype) {
  // Data 节点
  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);
  x.y.dtype = dtype;

  // Load 节点
  af::ascir_op::Load xLocal("load0");
  xLocal.x = x.y;
  xLocal.y.dtype = dtype;

  // Erfcx 操作
  af::ascir_op::Erfcx erfcx("erfcx");
  erfcx.x = xLocal.y;
  erfcx.y.dtype = dtype;

  // Store 节点
  af::ascir_op::Store x_out("store");
  x_out.x = erfcx.y;
  x_out.y.dtype = dtype;

  // Output 节点
  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = dtype;

  // 设置维度信息
  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::ErfcxTestFusedGraph(size_t dims_size, af::DataType dtype) {
  auto builder = GraphBuilder("erfcx_bf16_test");

  // 创建 data 节点
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  // 创建 AscGraph 节点
  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  // 连接边
  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);

  // 获取计算图
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }

  // 创建并序列化子图
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("erfcx_test");
  CreateErfcxTestAscGraph(sub_graph, dims_size, dtype);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);

  return compute_graph;
}

static void CreateExpm1Bf16AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data", graph);
  x.y.dtype = af::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load xLocal("load");
  xLocal.x = x.y;
  xLocal.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Expm1 expm1("expm1");
  expm1.x = xLocal.y;
  expm1.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Store x_out("store");
  x_out.x = expm1.y;
  x_out.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = af::DataType::DT_BF16;

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

/**
 *      NetOutput
 *         |
 *       AscBc
 *      /     \
 *   data0
 */
af::ComputeGraphPtr ShareGraph::Expm1Bf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("expm1_bf16_test");
  auto data = builder.AddNode("data", "Data", 0, 1);
  af::AttrUtils::SetInt(data->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("expm1_bf16_test");
  CreateExpm1Bf16AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}
static void CreateRoundToIntFloatToInt32AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data", graph);
  x.y.dtype = af::DataType::DT_FLOAT;
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load xLocal("load");
  xLocal.x = x.y;
  xLocal.y.dtype = af::DataType::DT_FLOAT;

  af::ascir_op::RoundToInt round_to_int("round_to_int");
  round_to_int.x = xLocal.y;
  round_to_int.y.dtype = af::DataType::DT_INT32;

  af::ascir_op::Store x_out("store");
  x_out.x = round_to_int.y;
  x_out.y.dtype = af::DataType::DT_INT32;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

// ============================================================================
// MaskRegChain 测试图：链式 MaskReg 位宽转换
// 验证场景：
// 1. UINT8 Load -> Where (UINT8→INT64) [MaskUnPack] - load直接作为mask输入
// 2. INT64 Compare -> Where (INT64→INT16) [MaskPack]
// 3. INT16 Compare -> Store
// ============================================================================
static void CreateMaskRegChainAscGraph(af::AscGraph &graph, size_t dims_size) {
  // ========== Data0: Input A (UINT8) - mask来源 ==========
  af::ascir_op::Data data0("data0", graph);
  data0.y.dtype = ge::DT_UINT8;
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load0("load0");
  load0.y.dtype = ge::DT_UINT8;
  load0.x = data0.y;

  // ========== Data1: Input B (INT64) - Where0 x2 ==========
  af::ascir_op::Data data1("data1", graph);
  data1.y.dtype = ge::DT_INT64;
  data1.ir_attr.SetIndex(1);

  af::ascir_op::Load load1("load1");
  load1.y.dtype = ge::DT_INT64;
  load1.x = data1.y;

  // ========== Data2: Input C (INT64) - Where0 x3 ==========
  af::ascir_op::Data data2("data2", graph);
  data2.y.dtype = ge::DT_INT64;
  data2.ir_attr.SetIndex(2);

  af::ascir_op::Load load2("load2");
  load2.y.dtype = ge::DT_INT64;
  load2.x = data2.y;

  // ========== Where0: UINT8→INT64 [MaskUnPack] ==========
  // load0 (UINT8) 直接作为 mask 输入，无需 Compare 节点
  af::ascir_op::Where where0("where0");
  where0.x1 = load0.y;
  where0.x2 = load1.y;
  where0.x3 = load2.y;
  where0.y.dtype = ge::DT_INT64;

  // ========== Data3: Input D (INT64) - Ge0 x2 ==========
  af::ascir_op::Data data3("data3", graph);
  data3.y.dtype = ge::DT_INT64;
  data3.ir_attr.SetIndex(3);

  af::ascir_op::Load load3("load3");
  load3.y.dtype = ge::DT_INT64;
  load3.x = data3.y;

  // ========== Ge0: INT64 Compare → UINT8 ==========
  af::ascir_op::Ge ge0("ge0");
  ge0.x1 = where0.y;
  ge0.x2 = load3.y;
  ge0.y.dtype = ge::DT_UINT8;

  // ========== Data4: Input E (INT16) - Where1 x2 ==========
  af::ascir_op::Data data4("data4", graph);
  data4.y.dtype = ge::DT_INT16;
  data4.ir_attr.SetIndex(4);

  af::ascir_op::Load load4("load4");
  load4.y.dtype = ge::DT_INT16;
  load4.x = data4.y;

  // ========== Data5: Input F (INT16) - Where1 x3 ==========
  af::ascir_op::Data data5("data5", graph);
  data5.y.dtype = ge::DT_INT16;
  data5.ir_attr.SetIndex(5);

  af::ascir_op::Load load5("load5");
  load5.y.dtype = ge::DT_INT16;
  load5.x = data5.y;

  // ========== Where1: UINT8→INT16 [MaskPack] ==========
  af::ascir_op::Where where1("where1");
  where1.x1 = ge0.y;
  where1.x2 = load4.y;
  where1.x3 = load5.y;
  where1.y.dtype = ge::DT_INT16;

  // ========== Data6: Input G (INT16) - Ge1 x2 ==========
  af::ascir_op::Data data6("data6", graph);
  data6.y.dtype = ge::DT_INT16;
  data6.ir_attr.SetIndex(6);

  af::ascir_op::Load load6("load6");
  load6.y.dtype = ge::DT_INT16;
  load6.x = data6.y;

  // ========== Ge1: INT16 Compare → UINT8 ==========
  af::ascir_op::Ge ge1("ge1");
  ge1.x1 = where1.y;
  ge1.x2 = load6.y;
  ge1.y.dtype = ge::DT_UINT8;

  // ========== Store + Output ==========
  af::ascir_op::Store store0("store0");
  store0.x = ge1.y;
  store0.y.dtype = ge::DT_UINT8;

  af::ascir_op::Output output0("output0");
  output0.x = store0.y;
  output0.y.dtype = ge::DT_UINT8;
  output0.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::MaskRegChainFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("mask_reg_chain_test");

  // 创建7个Data节点作为输入
  for (int i = 0; i < 7; i++) {
    auto data = builder.AddNode("data" + std::to_string(i), "Data", 0, 1);
    af::AttrUtils::SetInt(data->GetOpDescBarePtr(), "_parent_node_index", i);
  }

  // 创建AscGraph节点，7个输入，1个输出
  auto ascbc = builder.AddNode("ascbc", "AscGraph", 7, 1);

  // 创建1个NetOutput节点
  auto netoutput = builder.AddNode("netoutput0", af::NETOUTPUT, 1, 0);

  // 连接边：Data -> AscGraph
  for (int i = 0; i < 7; i++) {
    builder.AddDataEdge(builder.GetGraph()->FindNode("data" + std::to_string(i)), 0, ascbc, i);
  }

  // 连接边：AscGraph -> NetOutput
  builder.AddDataEdge(ascbc, 0, netoutput, 0);

  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }

  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("mask_reg_chain");
  CreateMaskRegChainAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}
af::ComputeGraphPtr ShareGraph::RoundToIntFloatToInt32FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("round_to_int_float_to_int32_test");
  auto data = builder.AddNode("data", "Data", 0, 1);
  af::AttrUtils::SetInt(data->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("round_to_int_float_to_int32_test");
  CreateRoundToIntFloatToInt32AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateTruncToIntBf16ToInt32AscGraph(af::AscGraph &graph, size_t dims_size) {
  af::ascir_op::Data x("data", graph);
  x.y.dtype = af::DataType::DT_BF16;
  x.ir_attr.SetIndex(0);

  af::ascir_op::Load xLocal("load");
  xLocal.x = x.y;
  xLocal.y.dtype = af::DataType::DT_BF16;

  af::ascir_op::TruncToInt trunc_to_int("trunc_to_int");
  trunc_to_int.x = xLocal.y;
  trunc_to_int.y.dtype = af::DataType::DT_INT32;

  af::ascir_op::Store x_out("store");
  x_out.x = trunc_to_int.y;
  x_out.y.dtype = af::DataType::DT_INT32;

  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);

  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::TruncToIntBf16ToInt32FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("trunc_to_int_bf16_to_int32_test");
  auto data = builder.AddNode("data", "Data", 0, 1);
  af::AttrUtils::SetInt(data->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("trunc_to_int_bf16_to_int32_test");
  CreateTruncToIntBf16ToInt32AscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}
/**
 * 串行链接所有支持scalar输入的vf融合算子，包含ScalarBroadcast场景
 *
 *           output
 *              |
 *            store
 *              |
 *          maximum5
 *              |
 *          minimum4
 *              |
 *            add3
 *              |
 *     /--------|--------\
 *     |        |        |
 *  maximum0  minimum1   add2
 *     |        |        |
 *   scalar0       load1
 *                   |
 *                 data0 (tensor source)
 */
static void CreateVfScalarFusionComprehensiveGraph(af::AscGraph &graph) {
  // ========== 数据源（tensor输入） ==========
  af::ascir_op::Data data0("data0", graph);
  data0.y.dtype = ge::DT_FLOAT16;
  data0.ir_attr.SetIndex(0);

  af::ascir_op::Load load1("load1");
  load1.x = data0.y;
  load1.y.dtype = ge::DT_FLOAT16;

  // ========== UBscalar节点（UbScalar输入） ==========
  af::ascir_op::Data data_scalar("data_scalar", graph);
  data_scalar.y.dtype = ge::DT_FLOAT16;
  data_scalar.ir_attr.SetIndex(1);

  af::ascir_op::Load load_ubscalar("load_ubscalar");
  load_ubscalar.x = data_scalar.y;
  load_ubscalar.y.dtype = ge::DT_FLOAT16;
  *load_ubscalar.y.repeats = {af::sym::kSymbolOne, af::sym::kSymbolOne};
  *load_ubscalar.y.strides = {af::sym::kSymbolZero, af::sym::kSymbolZero};

  // ========== Scalar节点（真正的Scalar） ==========
  af::ascir_op::Scalar scalar_const("scalar_const", graph);
  scalar_const.ir_attr.SetValue("2.0");
  scalar_const.y.dtype = ge::DT_FLOAT16;

  // ========== 计算节点 - UbScalar分支 ==========
  // 注意：根据OnlySecondInputSupportScalar约束，只有第二个输入可以是scalar，第一个必须是tensor
  // 1. Maximum (UbScalar分支: load1 + load_ubscalar)
  af::ascir_op::Maximum maximum0("maximum0");
  maximum0.x1 = load1.y;          // tensor输入（第一个）
  maximum0.x2 = load_ubscalar.y;  // UbScalar输入（第二个）
  maximum0.y.dtype = ge::DT_FLOAT16;

  // 2. Minimum (UbScalar分支: load1 + load_ubscalar)
  af::ascir_op::Minimum minimum1("minimum1");
  minimum1.x1 = load1.y;          // tensor输入（第一个）
  minimum1.x2 = load_ubscalar.y;  // UbScalar输入（第二个）
  minimum1.y.dtype = ge::DT_FLOAT16;

  // 3. Add (UbScalar分支: load1 + load_ubscalar)
  af::ascir_op::Add add2("add2");
  add2.x1 = load1.y;          // tensor输入（第一个）
  add2.x2 = load_ubscalar.y;  // UbScalar输入（第二个）
  add2.y.dtype = ge::DT_FLOAT16;

  // ========== 计算节点 - Scalar分支 ==========
  // 4. Add (Scalar分支: load1 + scalar_const)
  af::ascir_op::Add add_scalar("add_scalar");
  add_scalar.x1 = load1.y;         // tensor输入（第一个）
  add_scalar.x2 = scalar_const.y;  // Scalar输入（第二个）
  add_scalar.y.dtype = ge::DT_FLOAT16;

  // ========== 计算节点 - 串联融合 ==========
  af::ascir_op::Add add3("add3");
  add3.x1 = maximum0.y;
  add3.x2 = minimum1.y;
  add3.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Minimum minimum4("minimum4");
  minimum4.x1 = add3.y;
  minimum4.x2 = add2.y;
  minimum4.y.dtype = ge::DT_FLOAT16;

  af::ascir_op::Maximum maximum5("maximum5");
  maximum5.x1 = minimum4.y;
  maximum5.x2 = add_scalar.y;  // 使用Scalar分支的结果
  maximum5.y.dtype = ge::DT_FLOAT16;

  // Store
  af::ascir_op::Store store_op("store");
  store_op.x = maximum5.y;
  store_op.y.dtype = ge::DT_FLOAT16;

  // Output
  af::ascir_op::Output output_op("output");
  output_op.ir_attr.SetIndex(0);
  output_op.x = store_op.y;
  output_op.y.dtype = ge::DT_FLOAT16;

  // 使用标准的轴构造函数，创建2D轴（与测试用例一致）
  ConstructVVAscGraphAxisInfo(graph, 2);
}

af::ComputeGraphPtr ShareGraph::VfScalarFusionComprehensiveFusedGraph() {
  auto builder = GraphBuilder("vf_scalar_fusion_comprehensive_test");

  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("vf_scalar_fusion_comprehensive");
  CreateVfScalarFusionComprehensiveGraph(sub_graph);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateRemainderBf16AscGraph(af::AscGraph &graph, size_t dims_size, ge::DataType dtype) {
  // Data 节点
  af::ascir_op::Data x1("data0", graph);
  x1.ir_attr.SetIndex(0);
  x1.y.dtype = dtype;

  af::ascir_op::Data x2("data1", graph);
  x2.ir_attr.SetIndex(1);
  x2.y.dtype = dtype;

  // Load 节点
  af::ascir_op::Load x1Local("load0");
  x1Local.x = x1.y;
  x1Local.y.dtype = dtype;

  af::ascir_op::Load x2Local("load1");
  x2Local.x = x2.y;
  x2Local.y.dtype = dtype;

  // Remainder 操作
  af::ascir_op::Remainder remainder("remainder");
  remainder.x1 = x1Local.y;
  remainder.x2 = x2Local.y;
  remainder.y.dtype = dtype;

  // Store 节点
  af::ascir_op::Store x_out("store");
  x_out.x = remainder.y;
  x_out.y.dtype = dtype;

  // Output 节点
  af::ascir_op::Output y("output");
  y.x = x_out.y;
  y.ir_attr.SetIndex(0);
  y.y.dtype = dtype;

  // 设置维度信息
  ConstructVVAscGraphAxisInfo(graph, dims_size);
}

af::ComputeGraphPtr ShareGraph::RemainderBf16FusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("remainder_bf16_test");

  // 创建 data 节点
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  af::AttrUtils::SetInt(data1->GetOpDescBarePtr(), "_parent_node_index", 1);

  // 创建 AscGraph 节点
  auto ascbc = builder.AddNode("ascbc", "AscGraph", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  // 连接边
  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(data1, 0, ascbc, 1);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);

  // 获取计算图
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }

  // 创建并序列化子图
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("remainder_bf16");
  CreateRemainderBf16AscGraph(sub_graph, dims_size, af::DataType::DT_BF16);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);

  return compute_graph;
}

static void CreateArgMaxAscGraph(af::AscGraph &graph, size_t dims_size) {
  const Expression s0 = graph.CreateSizeVar(32);
  const Expression s1 = graph.CreateSizeVar(16);
  const Expression s2 = graph.CreateSizeVar(16);
  auto One = Symbol(1);
  auto Zero = Symbol(0);
  auto z0 = graph.CreateAxis("z0", s0);
  auto z1 = graph.CreateAxis("z1", s1);
  auto z2 = graph.CreateAxis("z2", s2);

  af::ascir_op::Data x("data0", graph);
  x.ir_attr.SetIndex(0);
  x.y.dtype = af::DataType::DT_FLOAT;

  af::ascir_op::Load load1("load1");
  load1.x = x.y;
  load1.attr.api.compute_type = ComputeType::kComputeLoad;
  load1.attr.api.type = af::ApiType::kAPITypeCompute;
  load1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *load1.y.axis = {z0.id, z1.id, z2.id};
  *load1.y.repeats = {s0, s1, s2};
  *load1.y.strides = {s1 * s2, s2, One};
  load1.y.dtype = af::DataType::DT_FLOAT;
  load1.attr.api.unit = ComputeUnit::kUnitMTE2;

  af::ascir_op::ArgMax argmax("argmax");
  argmax.x = load1.y;
  argmax.attr.sched.axis = {z0.id, z1.id, z2.id};
  *argmax.y.axis = {z0.id, z1.id, z2.id};
  *argmax.y.repeats = {s0, s1, One};
  *argmax.y.strides = {s1, One, Zero};
  argmax.y.dtype = af::DataType::DT_INT64;
  argmax.attr.tmp_buffers = {{{af::Symbol(8192), -1}, MemAttr(), 0}, {{af::Symbol(8192), 0}, MemAttr(), 1}};

  af::ascir_op::Store store1("store1");
  store1.x = argmax.y;
  store1.attr.api.compute_type = ComputeType::kComputeStore;
  store1.attr.api.type = af::ApiType::kAPITypeCompute;
  store1.attr.sched.axis = {z0.id, z1.id, z2.id};
  *store1.y.axis = {z0.id, z1.id, z2.id};
  *store1.y.repeats = {s0, s1, One};
  *store1.y.strides = {s1, One, Zero};
  store1.y.dtype = af::DataType::DT_INT64;
  store1.attr.api.unit = ComputeUnit::kUnitMTE3;

  af::ascir_op::Output y1("y1");
  y1.x = store1.y;
  y1.attr.api.compute_type = ComputeType::kComputeInvalid;
  y1.attr.api.type = af::ApiType::kAPITypeBuffer;
  y1.y.dtype = af::DataType::DT_INT64;
  y1.ir_attr.SetIndex(0);
}

af::ComputeGraphPtr ShareGraph::ArgMaxFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("argmax_test");
  auto data0 = builder.AddNode("data0", "Data", 0, 1);
  af::AttrUtils::SetInt(data0->GetOpDescBarePtr(), "_parent_node_index", 0);

  auto ascbc = builder.AddNode("ascbc", "AscGraph", 1, 1);
  auto netoutput = builder.AddNode("netoutput1", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(data0, 0, ascbc, 0);
  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("argmax");
  CreateArgMaxAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateRandStoreAscGraph(af::AscGraph &graph, size_t dims_size) {
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  auto One = Symbol(1);

  af::ascir_op::Rand rand_op("rand");
  graph.AddNode(rand_op);
  *rand_op.y.axis = {z0.id};
  *rand_op.y.repeats = {s0};
  *rand_op.y.strides = {One};

  rand_op.attr.sched.axis = {z0.id};

  af::ascir_op::Abs abs_op("abs");
  graph.AddNode(abs_op);
  abs_op.x = rand_op.y;
  *abs_op.y.axis = {z0.id};
  *abs_op.y.repeats = {s0};
  *abs_op.y.strides = {One};

  abs_op.attr.api.compute_type = af::ComputeType::kComputeElewise;
  abs_op.attr.api.type = af::ApiType::kAPITypeCompute;
  abs_op.attr.api.unit = af::ComputeUnit::kUnitVector;
  abs_op.attr.sched.axis = {z0.id};

  af::ascir_op::Store store("store");
  store.x = abs_op.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.attr.api.type = af::ApiType::kAPITypeCompute;
  store.attr.api.unit = ComputeUnit::kUnitMTE3;
  store.attr.sched.axis = {z0.id};
  *store.y.axis = {z0.id};
  *store.y.repeats = {s0};
  *store.y.strides = {One};
  store.y.dtype = af::DataType::DT_FLOAT;

  af::ascir_op::Output y("output");
  y.x = store.y;
  y.ir_attr.SetIndex(0);
}

af::ComputeGraphPtr ShareGraph::RandStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("rand_store_test");
  auto ascbc = builder.AddNode("ascbc", "AscGraph", 0, 1);
  auto netoutput = builder.AddNode("netoutput", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("rand_store");
  CreateRandStoreAscGraph(sub_graph, dims_size);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateRandnStoreAscGraph(af::AscGraph &graph, size_t dims_size, ge::DataType dtype) {
  auto s0 = graph.CreateSizeVar("s0");
  auto z0 = graph.CreateAxis("z0", s0);
  auto One = Symbol(1);

  af::ascir_op::Randn randn_op("randn");
  graph.AddNode(randn_op);
  *randn_op.y.axis = {z0.id};
  *randn_op.y.repeats = {s0};
  *randn_op.y.strides = {One};
  randn_op.y.dtype = dtype;

  randn_op.attr.api.compute_type = af::ComputeType::kComputeElewise;
  randn_op.attr.api.type = af::ApiType::kAPITypeCompute;
  randn_op.attr.api.unit = af::ComputeUnit::kUnitVector;
  randn_op.attr.sched.axis = {z0.id};

  af::ascir_op::Store store("store");
  store.x = randn_op.y;
  store.attr.api.compute_type = ComputeType::kComputeStore;
  store.attr.api.type = af::ApiType::kAPITypeCompute;
  store.attr.api.unit = ComputeUnit::kUnitMTE3;
  store.attr.sched.axis = {z0.id};
  *store.y.axis = {z0.id};
  *store.y.repeats = {s0};
  *store.y.strides = {One};
  store.y.dtype = dtype;

  af::ascir_op::Output y("output");
  y.x = store.y;
  y.ir_attr.SetIndex(0);
}

af::ComputeGraphPtr ShareGraph::RandnStoreFusedGraph(size_t dims_size) {
  auto builder = GraphBuilder("randn_store_test");
  auto ascbc = builder.AddNode("ascbc", "AscGraph", 0, 1);
  auto netoutput = builder.AddNode("netoutput", af::NETOUTPUT, 1, 0);

  builder.AddDataEdge(ascbc, 0, netoutput, 0);
  ComputeGraphPtr compute_graph = builder.GetGraph();
  if (compute_graph == nullptr) {
    return nullptr;
  }
  auto ascbc_node = compute_graph->FindNode("ascbc");
  af::AscGraph sub_graph("randn_store");
  CreateRandnStoreAscGraph(sub_graph, dims_size, ge::DT_UINT32);

  std::string sub_graph_str;
  af::AscGraphUtils::SerializeToReadable(sub_graph, sub_graph_str);
  af::AttrUtils::SetStr(ascbc_node->GetOpDescBarePtr(), "ascgraph", sub_graph_str);
  return compute_graph;
}

static void CreateAscBackendGraphTwoInTwoOut(std::shared_ptr<af::AscGraph> &graph, const std::string &prefix,
                                             int64_t axis_num = 2) {
  auto ONE = af::Symbol(1);
  std::vector<int64_t> axis_ids;
  std::vector<af::Expression> repeats;
  for (int64_t i = 0; i < axis_num; ++i) {
    const af::Expression exp = graph->CreateSizeVar("s" + std::to_string(i));
    auto axis = graph->CreateAxis("z" + std::to_string(i), exp);
    axis_ids.push_back(i);
    repeats.push_back(exp);
  }

  std::vector<af::Expression> strides(repeats.size(), af::ops::One);
  if (axis_num > 1) {
    for (int64_t i = axis_num - 2; i >= 0; --i) {
      strides[i] = repeats[i + 1] * strides[i + 1];
    }
  }

  af::ascir_op::Data data0(std::string(prefix + "_data0").c_str(), *graph);
  data0.attr.sched.axis = axis_ids;
  *data0.y.axis = axis_ids;
  *data0.y.repeats = repeats;
  *data0.y.strides = strides;
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Load load0(std::string(prefix + "_load0").c_str());
  load0.x = data0.y;
  load0.attr.sched.axis = axis_ids;
  *load0.y.axis = axis_ids;
  *load0.y.repeats = repeats;
  *load0.y.strides = strides;

  af::ascir_op::Data data1(std::string(prefix + "_data1").c_str(), *graph);
  data1.attr.sched.axis = axis_ids;
  *data1.y.axis = axis_ids;
  *data1.y.repeats = repeats;
  *data1.y.strides = strides;
  data1.ir_attr.SetIndex(1);
  data1.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Load load1(std::string(prefix + "_load1").c_str());
  load1.x = data1.y;
  load1.attr.sched.axis = axis_ids;
  *load1.y.axis = axis_ids;
  *load1.y.repeats = repeats;
  *load1.y.strides = strides;

  af::ascir_op::Add add(std::string(prefix + "_add").c_str());
  add.x1 = load0.y;
  add.x2 = load1.y;
  add.attr.sched.axis = axis_ids;
  *add.y.axis = axis_ids;
  *add.y.repeats = repeats;
  *add.y.strides = strides;

  af::ascir_op::Store store0(std::string(prefix + "_store0").c_str());
  store0.x = add.y;
  store0.attr.sched.axis = axis_ids;
  *store0.y.axis = axis_ids;
  *store0.y.repeats = repeats;
  *store0.y.strides = strides;

  af::ascir_op::Output y0(std::string(prefix + "_out0").c_str());
  y0.x = store0.y;
  y0.ir_attr.SetIndex(0);
  y0.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Store store1(std::string(prefix + "_store1").c_str());
  store1.x = add.y;
  store1.attr.sched.axis = axis_ids;
  *store1.y.axis = axis_ids;
  *store1.y.repeats = repeats;
  *store1.y.strides = strides;

  af::ascir_op::Output y1(std::string(prefix + "_out1").c_str());
  y1.x = store1.y;
  y1.ir_attr.SetIndex(1);
  y1.y.dtype = ge::DT_FLOAT;
}

static void CreateAscBackendGraphTwoInOneOut(std::shared_ptr<af::AscGraph> &graph, const std::string &prefix,
                                             int64_t axis_num = 2) {
  auto ONE = af::Symbol(1);
  std::vector<int64_t> axis_ids;
  std::vector<af::Expression> repeats;
  for (int64_t i = 0; i < axis_num; ++i) {
    const af::Expression exp = graph->CreateSizeVar("s" + std::to_string(i));
    auto axis = graph->CreateAxis("z" + std::to_string(i), exp);
    axis_ids.push_back(i);
    repeats.push_back(exp);
  }

  std::vector<af::Expression> strides(repeats.size(), af::ops::One);
  if (axis_num > 1) {
    for (int64_t i = axis_num - 2; i >= 0; --i) {
      strides[i] = repeats[i + 1] * strides[i + 1];
    }
  }

  af::ascir_op::Data data0(std::string(prefix + "_data0").c_str(), *graph);
  data0.attr.sched.axis = axis_ids;
  *data0.y.axis = axis_ids;
  *data0.y.repeats = repeats;
  *data0.y.strides = strides;
  data0.ir_attr.SetIndex(0);
  data0.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Load load0(std::string(prefix + "_load0").c_str());
  load0.x = data0.y;
  load0.attr.sched.axis = axis_ids;
  *load0.y.axis = axis_ids;
  *load0.y.repeats = repeats;
  *load0.y.strides = strides;

  af::ascir_op::Data data1(std::string(prefix + "_data1").c_str(), *graph);
  data1.attr.sched.axis = axis_ids;
  *data1.y.axis = axis_ids;
  *data1.y.repeats = repeats;
  *data1.y.strides = strides;
  data1.ir_attr.SetIndex(1);
  data1.y.dtype = ge::DT_FLOAT;

  af::ascir_op::Load load1(std::string(prefix + "_load1").c_str());
  load1.x = data1.y;
  load1.attr.sched.axis = axis_ids;
  *load1.y.axis = axis_ids;
  *load1.y.repeats = repeats;
  *load1.y.strides = strides;

  af::ascir_op::Add add(std::string(prefix + "_add").c_str());
  add.x1 = load0.y;
  add.x2 = load1.y;
  add.attr.sched.axis = axis_ids;
  *add.y.axis = axis_ids;
  *add.y.repeats = repeats;
  *add.y.strides = strides;

  af::ascir_op::Store store0(std::string(prefix + "_store0").c_str());
  store0.x = add.y;
  store0.attr.sched.axis = axis_ids;
  *store0.y.axis = axis_ids;
  *store0.y.repeats = repeats;
  *store0.y.strides = strides;

  af::ascir_op::Output y0(std::string(prefix + "_out0").c_str());
  y0.x = store0.y;
  y0.ir_attr.SetIndex(0);
  y0.y.dtype = ge::DT_FLOAT;
}

static NodePtr CreateAscbcToAscGraph(const std::string &name, ComputeGraphPtr &compute_graph, int64_t in_num = 1,
                                     int64_t out_num = 1) {
  OpDescBuilder op_desc_builder(name, "AscBackend");
  op_desc_builder.AddDynamicInput("x", in_num);
  op_desc_builder.AddDynamicOutput("y", out_num);
  const auto &op_desc = op_desc_builder.Build();
  auto node = compute_graph->AddNode(op_desc);
  node->SetOwnerComputeGraph(compute_graph);
  return node;
}

af::ComputeGraphPtr ShareGraph::FusedBackendElewiseGraph(size_t dims_size) {
  std::shared_ptr<af::AscGraph> g0 = std::make_shared<af::AscGraph>("g0");
  CreateAscBackendGraphTwoInTwoOut(g0, "g0", dims_size);
  std::shared_ptr<af::AscGraph> g1 = std::make_shared<af::AscGraph>("g1");
  CreateAscBackendGraphTwoInOneOut(g1, "g1", dims_size);
  std::shared_ptr<af::AscGraph> g2 = std::make_shared<af::AscGraph>("g2");
  CreateAscBackendGraphTwoInOneOut(g2, "g2", dims_size);

  af::AscGraph fused_asc_graph("fused_backend_elewise_test");
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

  auto fuse1_attrs = ascbc1->GetOpDesc()->GetOrCreateAttrsGroup<ge::AutoFuseAttrs>();
  fuse1_attrs->SetAscGraph(g0);
  auto fuse2_attrs = ascbc2->GetOpDesc()->GetOrCreateAttrsGroup<ge::AutoFuseAttrs>();
  fuse2_attrs->SetAscGraph(g1);
  auto fuse3_attrs = ascbc3->GetOpDesc()->GetOrCreateAttrsGroup<ge::AutoFuseAttrs>();
  fuse3_attrs->SetAscGraph(g2);
  fused_graph->TopologicalSorting();
  return fused_graph;
}

}  // namespace ascir
