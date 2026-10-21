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

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "ascir_ops.h"
#include "common/platform_context.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/utils/graph_utils.h"
#include "indirect_load_utils.h"
#include "norm_utils.h"
#include "schedule_result.h"
#include "task_generator/indirect_load_schedule_case_generator.h"

namespace {
constexpr int64_t kSimtDcacheSize = 32 * 1024;

class PlatformContextReset {
 public:
  ~PlatformContextReset() {
    ge::PlatformContext::GetInstance().Reset();
  }
};

std::vector<std::string> AxisNames(af::AscGraph &graph, const std::vector<af::AxisId> &axis_ids) {
  std::vector<std::string> names;
  names.reserve(axis_ids.size());
  for (af::AxisId axis_id : axis_ids) {
    const auto *axis = graph.FindAxis(axis_id);
    names.push_back(axis == nullptr ? "" : axis->name);
  }
  return names;
}

af::AxisId FindAxisByName(af::AscGraph &graph, const std::string &name) {
  for (const auto &axis : graph.GetAllAxis()) {
    if (axis != nullptr && axis->name == name) {
      return axis->id;
    }
  }
  return af::kIdNone;
}

void ExpectAxisNames(af::AscGraph &graph, const std::vector<af::AxisId> &axis_ids,
                     const std::vector<std::string> &expected) {
  EXPECT_EQ(AxisNames(graph, axis_ids), expected);
}

void ExpectMergedFrom(af::AscGraph &graph, const std::string &merged_name,
                      const std::vector<std::string> &expected_from) {
  const af::AxisId merged_axis = FindAxisByName(graph, merged_name);
  ASSERT_NE(merged_axis, af::kIdNone);
  const auto *axis = graph.FindAxis(merged_axis);
  ASSERT_NE(axis, nullptr);
  EXPECT_EQ(axis->type, af::Axis::Type::kAxisTypeMerged);
  ExpectAxisNames(graph, axis->from, expected_from);
}

void ExpectFixedTileSplit(af::AscGraph &graph, af::AxisId outer_axis_id) {
  const auto *source_axis = graph.FindAxis(outer_axis_id);
  ASSERT_NE(source_axis, nullptr);
  const auto all_axes = graph.GetAllAxis();
  const auto tile_outer_axis = std::find_if(all_axes.begin(), all_axes.end(), [outer_axis_id](const auto &axis) {
    return axis != nullptr && axis->type == af::Axis::Type::kAxisTypeTileOuter &&
           axis->from == std::vector<af::AxisId>{outer_axis_id};
  });
  const auto tile_inner_axis = std::find_if(all_axes.begin(), all_axes.end(), [outer_axis_id](const auto &axis) {
    return axis != nullptr && axis->type == af::Axis::Type::kAxisTypeTileInner &&
           axis->from == std::vector<af::AxisId>{outer_axis_id};
  });
  ASSERT_NE(tile_outer_axis, all_axes.end());
  ASSERT_NE(tile_inner_axis, all_axes.end());
  EXPECT_EQ((*tile_outer_axis)->type, af::Axis::Type::kAxisTypeTileOuter);
  EXPECT_EQ((*tile_inner_axis)->type, af::Axis::Type::kAxisTypeTileInner);
  EXPECT_EQ((*tile_outer_axis)->size, source_axis->size);
  EXPECT_EQ(af::SymbolicUtils::StaticCheckEq((*tile_inner_axis)->size, af::ops::One), af::TriBool::kTrue);
  EXPECT_EQ((*tile_outer_axis)->split_pair_other_id, (*tile_inner_axis)->id);
  EXPECT_EQ((*tile_inner_axis)->split_pair_other_id, (*tile_outer_axis)->id);
}

void CollectAxisOrigins(af::AscGraph &graph, af::AxisId axis_id, std::vector<af::AxisId> &origins) {
  const auto *axis = graph.FindAxis(axis_id);
  ASSERT_NE(axis, nullptr);
  if (axis->from.empty()) {
    origins.emplace_back(axis_id);
    return;
  }
  for (const af::AxisId from : axis->from) {
    CollectAxisOrigins(graph, from, origins);
  }
}

void ExpectAxisOrigins(af::AscGraph &graph, af::AxisId axis_id, const std::vector<af::AxisId> &expected) {
  std::vector<af::AxisId> origins;
  CollectAxisOrigins(graph, axis_id, origins);
  EXPECT_EQ(origins, expected);
}

std::vector<af::Expression> DenseStrides(af::AscGraph &graph, const std::vector<af::AxisId> &axis_ids) {
  std::vector<af::Expression> strides(axis_ids.size(), af::sym::kSymbolOne);
  for (size_t i = 0UL; i + 1UL < axis_ids.size(); ++i) {
    strides[i] = graph.FindAxis(axis_ids[i + 1UL])->size;
    for (size_t j = i + 2UL; j < axis_ids.size(); ++j) {
      strides[i] = strides[i] * graph.FindAxis(axis_ids[j])->size;
    }
  }
  return strides;
}

template <typename Op>
void SetNodeView(Op &op, af::DataType dtype, const std::vector<af::AxisId> &axes,
                 const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides) {
  op.y.dtype = dtype;
  op.attr.sched.axis = axes;
  *op.y.axis = axes;
  *op.y.repeats = repeats;
  *op.y.strides = strides;
}

template <typename Op>
void SetVectorApi(Op &op) {
  op.attr.api.compute_type = af::ComputeType::kComputeElewise;
  op.attr.api.type = af::ApiType::kAPITypeCompute;
  op.attr.api.unit = af::ComputeUnit::kUnitVector;
}

void BuildInputPreChain(const af::AscOpOutput &input_load_output, const std::vector<af::AxisId> &axes,
                        const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides,
                        bool enabled, bool full_prefix, af::ascir_op::IndirectLoad &indirect_load) {
  if (!enabled) {
    indirect_load.x1 = input_load_output;
    return;
  }
  if (!full_prefix) {
    af::ascir_op::Abs pre_abs("pre_abs");
    pre_abs.x = input_load_output;
    SetNodeView(pre_abs, af::DT_FLOAT16, axes, repeats, strides);
    indirect_load.x1 = pre_abs.y;
    return;
  }
  af::ascir_op::Relu input_relu("input_relu");
  input_relu.x = input_load_output;
  SetNodeView(input_relu, af::DT_FLOAT16, axes, repeats, strides);
  af::ascir_op::Exp2 input_exp2("input_exp2");
  input_exp2.x = input_relu.y;
  SetNodeView(input_exp2, af::DT_FLOAT16, axes, repeats, strides);
  indirect_load.x1 = input_exp2.y;
}

void BuildIndexPreChain(const af::AscOpOutput &index_load_output, const std::vector<af::AxisId> &axes,
                        const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides,
                        bool full_prefix, af::ascir_op::IndirectLoad &indirect_load) {
  if (!full_prefix) {
    indirect_load.x2 = index_load_output;
    return;
  }
  af::ascir_op::Abs index_abs("index_abs");
  index_abs.x = index_load_output;
  SetNodeView(index_abs, af::DT_INT32, axes, repeats, strides);
  af::ascir_op::Abs index_abs2("index_abs2");
  index_abs2.x = index_abs.y;
  SetNodeView(index_abs2, af::DT_INT32, axes, repeats, strides);
  af::ascir_op::Cast index_cast("index_cast_float");
  index_cast.x = index_abs2.y;
  SetNodeView(index_cast, af::DT_FLOAT, axes, repeats, strides);
  af::ascir_op::Exp2 index_exp2("index_exp2");
  index_exp2.x = index_cast.y;
  SetNodeView(index_exp2, af::DT_FLOAT, axes, repeats, strides);
  af::ascir_op::Log2 index_log2("index_log2");
  index_log2.x = index_exp2.y;
  SetNodeView(index_log2, af::DT_FLOAT, axes, repeats, strides);
  af::ascir_op::FloorToInt index_floor_to_int("index_floor_to_int");
  index_floor_to_int.x = index_log2.y;
  SetNodeView(index_floor_to_int, af::DT_INT32, axes, repeats, strides);
  indirect_load.x2 = index_floor_to_int.y;
}

// [Bug A] index 链尾部额外外部消费者的图形态：kExternalStore 为合法的直连 store
// （纳入融合区域、收为空节点输出链），kIllegalConsumer 为非法的 Abs→store 消费者
// （SIMT 候选必须整候选淘汰）。
enum class IndexChainExtraConsumer { kNone, kExternalStore, kIllegalConsumer };

af::AscGraph BuildIndirectLoadGraph(int64_t axis, bool has_input_pre_node = false, bool full_prefix = false,
                                    IndexChainExtraConsumer index_extra = IndexChainExtraConsumer::kNone) {
  af::AscGraph graph("indirect_load_ut_graph");
  const af::Expression s0 = graph.CreateSizeVar("s0");
  const af::Expression s1 = graph.CreateSizeVar("s1");
  const af::Expression s2 = graph.CreateSizeVar("s2");
  const af::Expression s3 = graph.CreateSizeVar("s3");
  const af::Expression s4 = graph.CreateSizeVar("s4");
  const af::Expression s5 = graph.CreateSizeVar("s5");
  const af::Expression s6 = graph.CreateSizeVar("s6");
  const af::Expression s7 = graph.CreateSizeVar("s7");
  const auto z0 = graph.CreateAxis("z0", s0);
  const auto z1 = graph.CreateAxis("z1", s1);
  const auto z2 = graph.CreateAxis("z2", s2);
  const auto z3 = graph.CreateAxis("z3", s3);
  const auto z4 = graph.CreateAxis("z4", s4);
  const auto z5 = graph.CreateAxis("z5", s5);
  const auto z6 = graph.CreateAxis("z6", s6);
  const auto z7 = graph.CreateAxis("z7", s7);
  const std::vector<af::AxisId> input_axes = {z0.id, z1.id, z2.id, z3.id};
  const std::vector<af::AxisId> output_axes = {z4.id, z5.id, z6.id, z7.id};
  const std::vector<af::Expression> input_repeats = {s0, s1, s2, s3};
  const std::vector<af::Expression> output_repeats = {s4, s5, s6, s7};
  const std::vector<af::Expression> input_strides = {s1 * s2 * s3, s2 * s3, s3, af::sym::kSymbolOne};
  const std::vector<af::Expression> output_strides = {s5 * s6 * s7, s6 * s7, s7, af::sym::kSymbolOne};

  af::ascir_op::Data x("x", graph);
  x.y.dtype = ge::DT_FLOAT16;
  x.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  x.attr.api.type = af::ApiType::kAPITypeBuffer;
  x.ir_attr.SetIndex(0);
  x.attr.sched.axis = input_axes;
  *x.y.axis = input_axes;
  *x.y.repeats = input_repeats;
  *x.y.strides = input_strides;
  af::ascir_op::Load input_load("input_load");
  input_load.x = x.y;
  SetNodeView(input_load, af::DT_FLOAT16, input_axes, input_repeats, input_strides);

  af::ascir_op::Data index("index", graph);
  index.y.dtype = ge::DT_INT32;
  index.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  index.attr.api.type = af::ApiType::kAPITypeBuffer;
  index.ir_attr.SetIndex(1);
  index.attr.sched.axis = output_axes;
  *index.y.axis = output_axes;
  *index.y.repeats = output_repeats;
  *index.y.strides = output_strides;
  af::ascir_op::Load index_load("index_load");
  index_load.x = index.y;
  SetNodeView(index_load, af::DT_INT32, output_axes, output_repeats, output_strides);
  if (index_extra == IndexChainExtraConsumer::kExternalStore) {
    af::ascir_op::Store index_store("index_store");
    index_store.x = index_load.y;
    SetNodeView(index_store, af::DT_INT32, output_axes, output_repeats, output_strides);
    af::ascir_op::Output index_y("index_y");
    index_y.x = index_store.y;
    index_y.y.dtype = af::DT_INT32;
    index_y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
    index_y.attr.api.type = af::ApiType::kAPITypeBuffer;
    index_y.ir_attr.SetIndex(1);
    index_y.attr.sched.axis = output_axes;
    *index_y.y.axis = output_axes;
    *index_y.y.repeats = output_repeats;
    *index_y.y.strides = output_strides;
  } else if (index_extra == IndexChainExtraConsumer::kIllegalConsumer) {
    af::ascir_op::Abs index_abs("index_abs");
    index_abs.x = index_load.y;
    SetNodeView(index_abs, af::DT_INT32, output_axes, output_repeats, output_strides);
    SetVectorApi(index_abs);
    af::ascir_op::Store index_store("index_store");
    index_store.x = index_abs.y;
    SetNodeView(index_store, af::DT_INT32, output_axes, output_repeats, output_strides);
  }

  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  BuildInputPreChain(input_load.y, input_axes, input_repeats, input_strides, has_input_pre_node, full_prefix,
                     indirect_load);
  BuildIndexPreChain(index_load.y, output_axes, output_repeats, output_strides, full_prefix, indirect_load);
  indirect_load.y.dtype = ge::DT_FLOAT16;
  indirect_load.attr.sched.axis = output_axes;
  indirect_load.ir_attr.SetAxis(axis);
  *indirect_load.y.axis = output_axes;
  *indirect_load.y.repeats = output_repeats;
  *indirect_load.y.strides = output_strides;

  af::ascir_op::Store store("store");
  store.x = indirect_load.y;
  SetNodeView(store, af::DT_FLOAT16, output_axes, output_repeats, output_strides);
  af::ascir_op::Output y("y");
  y.x = store.y;
  y.y.dtype = ge::DT_FLOAT16;
  y.attr.api.compute_type = af::ComputeType::kComputeInvalid;
  y.attr.api.type = af::ApiType::kAPITypeBuffer;
  y.ir_attr.SetIndex(0);
  y.attr.sched.axis = output_axes;
  *y.y.axis = output_axes;
  *y.y.repeats = output_repeats;
  *y.y.strides = output_strides;

  return graph;
}

struct BroadcastGraphView {
  std::vector<af::AxisId> source_axes;
  std::vector<af::AxisId> input_axes;
  std::vector<af::AxisId> output_axes;
  std::vector<af::Expression> source_repeats;
  std::vector<af::Expression> input_repeats;
  std::vector<af::Expression> output_repeats;
};

BroadcastGraphView CreateBroadcastGraphView(af::AscGraph &graph) {
  BroadcastGraphView view;
  const std::vector<int64_t> shape = {2L, 3L, 4L};
  for (size_t dim = 0UL; dim < shape.size(); ++dim) {
    const af::Expression input_size = graph.CreateSizeVar(shape[dim]);
    const af::Expression output_size = graph.CreateSizeVar(shape[dim]);
    view.input_repeats.emplace_back(input_size);
    view.output_repeats.emplace_back(output_size);
    view.input_axes.emplace_back(graph.CreateAxis(("broadcast_i" + std::to_string(dim)).c_str(), input_size).id);
    view.output_axes.emplace_back(graph.CreateAxis(("broadcast_o" + std::to_string(dim)).c_str(), output_size).id);
  }
  view.source_axes = view.input_axes;
  view.source_repeats = view.input_repeats;
  view.source_axes[1] = graph.CreateAxis("broadcast_source", af::ops::One).id;
  view.source_repeats[1] = af::ops::One;
  return view;
}

void BuildBroadcastInputPreChain(const af::AscOpOutput &broadcast_output, const BroadcastGraphView &view,
                                 const std::vector<af::Expression> &input_strides, bool with_input_element,
                                 af::ascir_op::IndirectLoad &indirect_load) {
  if (!with_input_element) {
    indirect_load.x1 = broadcast_output;
    return;
  }
  af::ascir_op::Abs input_abs("broadcast_input_abs");
  input_abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  input_abs.x = broadcast_output;
  SetNodeView(input_abs, af::DT_FLOAT16, view.input_axes, view.input_repeats, input_strides);
  indirect_load.x1 = input_abs.y;
}

af::AscGraph BuildIndirectLoadBroadcastGraph(bool with_input_element = true, bool with_index_broadcast = false,
                                             int64_t axis = 1L, bool transpose_input = false) {
  af::AscGraph graph("indirect_load_broadcast_ut_graph");
  const BroadcastGraphView view = CreateBroadcastGraphView(graph);
  const std::vector<af::Expression> source_strides = {view.source_repeats[2], view.source_repeats[2], af::ops::One};
  const std::vector<af::Expression> input_strides = {view.input_repeats[2], af::ops::Zero, af::ops::One};
  const std::vector<af::Expression> output_strides = {view.output_repeats[1] * view.output_repeats[2],
                                                      view.output_repeats[2], af::ops::One};
  const af::Expression index_axis_stride = view.output_repeats[2] * (af::ops::One + af::ops::One);
  const std::vector<af::Expression> index_strides = {view.output_repeats[1] * index_axis_stride, index_axis_stride,
                                                     af::ops::One};
  af::ascir_op::Data x("broadcast_x", graph);
  x.ir_attr.SetIndex(0);
  SetNodeView(x, af::DT_FLOAT16, view.source_axes, view.source_repeats, source_strides);
  af::ascir_op::Load input_load("broadcast_input_load");
  input_load.x = x.y;
  SetNodeView(input_load, af::DT_FLOAT16, view.source_axes, view.source_repeats, source_strides);
  af::ascir_op::Transpose transpose("broadcast_input_transpose");
  if (transpose_input) {
    auto physical_axes = view.source_axes;
    auto physical_repeats = view.source_repeats;
    std::swap(physical_axes[0], physical_axes[2]);
    std::swap(physical_repeats[0], physical_repeats[2]);
    SetNodeView(input_load, af::DT_FLOAT16, physical_axes, physical_repeats,
                {af::Symbol(2), af::Symbol(2), af::ops::One});
    transpose.x = input_load.y;
    SetNodeView(transpose, af::DT_FLOAT16, view.source_axes, view.source_repeats, source_strides);
  }
  af::ascir_op::Broadcast broadcast("input_broadcast");
  broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  broadcast.x = transpose_input ? transpose.y : input_load.y;
  SetNodeView(broadcast, af::DT_FLOAT16, view.input_axes, view.input_repeats, input_strides);
  auto index_source_repeats = view.output_repeats;
  auto index_source_strides = index_strides;
  if (with_index_broadcast) {
    index_source_repeats[0] = af::ops::One;
    index_source_strides = {view.output_repeats[1] * view.output_repeats[2], view.output_repeats[2], af::ops::One};
  }
  af::ascir_op::Data index("broadcast_index", graph);
  index.ir_attr.SetIndex(1);
  SetNodeView(index, af::DT_INT32, view.output_axes, index_source_repeats, index_source_strides);
  af::ascir_op::Load index_load("broadcast_index_load");
  index_load.x = index.y;
  SetNodeView(index_load, af::DT_INT32, view.output_axes, index_source_repeats, index_source_strides);
  af::ascir_op::Abs index_abs("broadcast_index_abs");
  index_abs.attr.api.compute_type = af::ComputeType::kComputeElewise;
  index_abs.x = index_load.y;
  SetNodeView(index_abs, af::DT_INT32, view.output_axes, index_source_repeats, index_source_strides);
  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  BuildBroadcastInputPreChain(broadcast.y, view, input_strides, with_input_element, indirect_load);
  if (with_index_broadcast) {
    af::ascir_op::Broadcast index_broadcast("index_outer_broadcast");
    index_broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
    index_broadcast.x = index_abs.y;
    SetNodeView(index_broadcast, af::DT_INT32, view.output_axes, view.output_repeats, output_strides);
    indirect_load.x2 = index_broadcast.y;
  } else {
    indirect_load.x2 = index_abs.y;
  }
  indirect_load.ir_attr.SetAxis(axis);
  SetNodeView(indirect_load, af::DT_FLOAT16, view.output_axes, view.output_repeats, output_strides);
  af::ascir_op::Store store("broadcast_store");
  store.x = indirect_load.y;
  SetNodeView(store, af::DT_FLOAT16, view.output_axes, view.output_repeats, output_strides);
  af::ascir_op::Output output("broadcast_output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  SetNodeView(output, af::DT_FLOAT16, view.output_axes, view.output_repeats, output_strides);
  return graph;
}

af::AscGraph BuildIndirectLoadIndexBroadcastGraph() {
  af::AscGraph graph("indirect_load_index_broadcast_ut_graph");
  const af::Expression input_rows = graph.CreateSizeVar(100000);
  const af::Expression rows = graph.CreateSizeVar(1024);
  const af::Expression columns = graph.CreateSizeVar(1024);
  const af::AxisId input_axis0 = graph.CreateAxis("index_broadcast_x0", input_rows).id;
  const af::AxisId input_axis1 = graph.CreateAxis("index_broadcast_x1", columns).id;
  const af::AxisId output_axis0 = graph.CreateAxis("index_broadcast_y0", rows).id;
  const af::AxisId output_axis1 = graph.CreateAxis("index_broadcast_y1", columns).id;
  const std::vector<af::AxisId> input_axes = {input_axis0, input_axis1};
  const std::vector<af::AxisId> output_axes = {output_axis0, output_axis1};
  const std::vector<af::Expression> input_sizes = {input_rows, columns};
  const std::vector<af::Expression> output_sizes = {rows, columns};
  const std::vector<af::Expression> dense_strides = {columns, af::ops::One};
  af::ascir_op::Data input("index_broadcast_input", graph);
  input.ir_attr.SetIndex(0);
  SetNodeView(input, af::DT_FLOAT, input_axes, input_sizes, dense_strides);
  af::ascir_op::Load input_load("index_broadcast_input_load");
  input_load.x = input.y;
  SetNodeView(input_load, af::DT_FLOAT, input_axes, input_sizes, dense_strides);
  af::ascir_op::Data index("index_broadcast_index", graph);
  index.ir_attr.SetIndex(1);
  SetNodeView(index, af::DT_INT64, output_axes, {rows, af::ops::One}, {af::ops::One, af::ops::Zero});
  af::ascir_op::Load index_load("index_broadcast_index_load");
  index_load.x = index.y;
  SetNodeView(index_load, af::DT_INT64, output_axes, {rows, af::ops::One}, {af::ops::One, af::ops::Zero});
  af::ascir_op::Broadcast broadcast("index_broadcast");
  broadcast.attr.api.compute_type = af::ComputeType::kComputeBroadcast;
  broadcast.x = index_load.y;
  SetNodeView(broadcast, af::DT_INT64, output_axes, output_sizes, dense_strides);
  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  indirect_load.x1 = input_load.y;
  indirect_load.x2 = broadcast.y;
  indirect_load.ir_attr.SetAxis(0L);
  SetNodeView(indirect_load, af::DT_FLOAT, output_axes, output_sizes, dense_strides);
  af::ascir_op::Store store("index_broadcast_store");
  store.x = indirect_load.y;
  SetNodeView(store, af::DT_FLOAT, output_axes, output_sizes, dense_strides);
  af::ascir_op::Output output("index_broadcast_output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  SetNodeView(output, af::DT_FLOAT, output_axes, output_sizes, dense_strides);
  return graph;
}

bool AddSideConsumer(af::AscGraph &graph, const char *producer_name) {
  af::ascir_op::Abs side_consumer("coverage_side_consumer");
  graph.AddNode(side_consumer);
  const auto producer = graph.FindNode(producer_name);
  const auto consumer = graph.FindNode("coverage_side_consumer");
  return producer != nullptr && consumer != nullptr &&
         af::GraphUtils::AddEdge(producer->GetOutDataAnchor(0), consumer->GetInDataAnchor(0)) == ge::GRAPH_SUCCESS;
}

bool AddSimtFanoutTopology(af::AscGraph &graph) {
  const auto indirect_load = graph.FindNode("indirect_load");
  const auto store = graph.FindNode("store");
  if (indirect_load == nullptr || store == nullptr || indirect_load->outputs().empty()) {
    return false;
  }
  const auto &output_attr = indirect_load->outputs()[0]->attr;

  af::ascir_op::Abs main_transform("coverage_main_transform");
  SetNodeView(main_transform, af::DT_FLOAT16, output_attr.axis, output_attr.repeats, output_attr.strides);
  SetVectorApi(main_transform);
  graph.AddNode(main_transform);

  af::ascir_op::Data side_input("coverage_side_input", graph);
  side_input.ir_attr.SetIndex(2);
  SetNodeView(side_input, af::DT_FLOAT16, output_attr.axis, output_attr.repeats, output_attr.strides);
  af::ascir_op::Load side_load("coverage_side_load");
  SetNodeView(side_load, af::DT_FLOAT16, output_attr.axis, output_attr.repeats, output_attr.strides);
  side_load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  graph.AddNode(side_load);
  af::ascir_op::Abs side_pre("coverage_side_pre");
  SetNodeView(side_pre, af::DT_FLOAT16, output_attr.axis, output_attr.repeats, output_attr.strides);
  SetVectorApi(side_pre);
  graph.AddNode(side_pre);
  af::ascir_op::Add side_branch("coverage_side_branch");
  SetNodeView(side_branch, af::DT_FLOAT16, output_attr.axis, output_attr.repeats, output_attr.strides);
  SetVectorApi(side_branch);
  graph.AddNode(side_branch);
  af::ascir_op::Store side_store("coverage_side_store");
  SetNodeView(side_store, af::DT_FLOAT16, output_attr.axis, output_attr.repeats, output_attr.strides);
  side_store.attr.api.compute_type = af::ComputeType::kComputeStore;
  graph.AddNode(side_store);
  af::ascir_op::Output side_output("coverage_side_output");
  side_output.ir_attr.SetIndex(1);
  SetNodeView(side_output, af::DT_FLOAT16, output_attr.axis, output_attr.repeats, output_attr.strides);
  graph.AddNode(side_output);

  const auto main_transform_node = graph.FindNode("coverage_main_transform");
  const auto side_input_node = graph.FindNode("coverage_side_input");
  const auto side_load_node = graph.FindNode("coverage_side_load");
  const auto side_pre_node = graph.FindNode("coverage_side_pre");
  const auto side_branch_node = graph.FindNode("coverage_side_branch");
  const auto side_store_node = graph.FindNode("coverage_side_store");
  const auto side_output_node = graph.FindNode("coverage_side_output");
  if (main_transform_node == nullptr || side_input_node == nullptr || side_load_node == nullptr ||
      side_pre_node == nullptr || side_branch_node == nullptr || side_store_node == nullptr ||
      side_output_node == nullptr) {
    return false;
  }
  return af::GraphUtils::AddEdge(indirect_load->GetOutDataAnchor(0), main_transform_node->GetInDataAnchor(0)) ==
             ge::GRAPH_SUCCESS &&
         af::GraphUtils::ReplaceEdgeSrc(indirect_load->GetOutDataAnchor(0), store->GetInDataAnchor(0),
                                        main_transform_node->GetOutDataAnchor(0)) == ge::GRAPH_SUCCESS &&
         af::GraphUtils::AddEdge(side_input_node->GetOutDataAnchor(0), side_load_node->GetInDataAnchor(0)) ==
             ge::GRAPH_SUCCESS &&
         af::GraphUtils::AddEdge(side_load_node->GetOutDataAnchor(0), side_pre_node->GetInDataAnchor(0)) ==
             ge::GRAPH_SUCCESS &&
         af::GraphUtils::AddEdge(indirect_load->GetOutDataAnchor(0), side_branch_node->GetInDataAnchor(0)) ==
             ge::GRAPH_SUCCESS &&
         af::GraphUtils::AddEdge(side_pre_node->GetOutDataAnchor(0), side_branch_node->GetInDataAnchor(1)) ==
             ge::GRAPH_SUCCESS &&
         af::GraphUtils::AddEdge(side_branch_node->GetOutDataAnchor(0), side_store_node->GetInDataAnchor(0)) ==
             ge::GRAPH_SUCCESS &&
         af::GraphUtils::AddEdge(side_store_node->GetOutDataAnchor(0), side_output_node->GetInDataAnchor(0)) ==
             ge::GRAPH_SUCCESS;
}

void SetStridedInputPath(af::AscGraph &graph) {
  const std::vector<af::Expression> strides = {af::Symbol(64), af::Symbol(2)};
  for (const char *name : {"data0", "load0", "data1", "load1", "x_copy_sign"}) {
    const auto node = graph.FindNode(name);
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->outputs().size(), 1UL);
    node->outputs()[0]->attr.strides = strides;
  }
}

af::AscGraph BuildIndirectLoadPrecisionCastGraph() {
  af::AscGraph graph("indirect_load_precision_cast_ut_graph");
  const af::Expression s0 = graph.CreateSizeVar("s0");
  const af::Expression s1 = graph.CreateSizeVar("s1");
  const af::Expression s2 = graph.CreateSizeVar("s2");
  const af::Expression s3 = graph.CreateSizeVar("s3");
  const auto z0 = graph.CreateAxis("z0", s0);
  const auto z1 = graph.CreateAxis("z1", s1);
  const auto z2 = graph.CreateAxis("z2", s2);
  const auto z3 = graph.CreateAxis("z3", s3);
  const std::vector<af::AxisId> input_axes = {z0.id, z1.id};
  const std::vector<af::AxisId> output_axes = {z2.id, z3.id};
  const std::vector<af::Expression> input_repeats = {s0, s1};
  const std::vector<af::Expression> output_repeats = {s2, s3};
  const std::vector<af::Expression> input_strides = {s1, af::sym::kSymbolOne};
  const std::vector<af::Expression> output_strides = {s3, af::sym::kSymbolOne};

  af::ascir_op::Data x("x", graph);
  x.ir_attr.SetIndex(0);
  SetNodeView(x, af::DT_FLOAT16, input_axes, input_repeats, input_strides);
  af::ascir_op::Load input_load("input_load");
  input_load.x = x.y;
  SetNodeView(input_load, af::DT_FLOAT16, input_axes, input_repeats, input_strides);
  af::ascir_op::Cast input_cast("input_cast");
  input_cast.x = input_load.y;
  SetNodeView(input_cast, af::DT_FLOAT, input_axes, input_repeats, input_strides);

  af::ascir_op::Data index("index", graph);
  index.ir_attr.SetIndex(1);
  SetNodeView(index, af::DT_INT32, output_axes, output_repeats, output_strides);
  af::ascir_op::Load index_load("index_load");
  index_load.x = index.y;
  SetNodeView(index_load, af::DT_INT32, output_axes, output_repeats, output_strides);

  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  indirect_load.x1 = input_cast.y;
  indirect_load.x2 = index_load.y;
  indirect_load.ir_attr.SetAxis(1);
  SetNodeView(indirect_load, af::DT_FLOAT, output_axes, output_repeats, output_strides);
  af::ascir_op::Exp output_exp("output_exp");
  output_exp.x = indirect_load.y;
  SetNodeView(output_exp, af::DT_FLOAT, output_axes, output_repeats, output_strides);
  af::ascir_op::Cast output_cast("output_cast");
  output_cast.x = output_exp.y;
  SetNodeView(output_cast, af::DT_FLOAT16, output_axes, output_repeats, output_strides);
  af::ascir_op::Store store("store");
  store.x = output_cast.y;
  SetNodeView(store, af::DT_FLOAT16, output_axes, output_repeats, output_strides);
  af::ascir_op::Output y("y");
  y.x = store.y;
  y.ir_attr.SetIndex(0);
  SetNodeView(y, af::DT_FLOAT16, output_axes, output_repeats, output_strides);
  return graph;
}

af::AscGraph BuildExpandedMultiInputGraph(bool add_unary = false, int64_t output_dim0 = 2L, int64_t output_dim1 = 17L) {
  af::AscGraph graph("indirect_load_expanded_multi_input_ut_graph");
  const af::Expression x_s0 = graph.CreateSizeVar(2);
  const af::Expression x_s1 = graph.CreateSizeVar(16);
  const af::Expression y_s0 = graph.CreateSizeVar(output_dim0);
  const af::Expression y_s1 = graph.CreateSizeVar(output_dim1);
  const auto x0 = graph.CreateAxis("x0", x_s0);
  const auto x1 = graph.CreateAxis("x1", x_s1);
  const auto y0 = graph.CreateAxis("y0", y_s0);
  const auto y1 = graph.CreateAxis("y1", y_s1);
  const std::vector<af::AxisId> input_axes = {x0.id, x1.id};
  const std::vector<af::AxisId> output_axes = {y0.id, y1.id};
  const std::vector<af::Expression> input_repeats = {x_s0, x_s1};
  const std::vector<af::Expression> output_repeats = {y_s0, y_s1};
  const std::vector<af::Expression> input_strides = {x_s1, af::sym::kSymbolOne};
  const std::vector<af::Expression> output_strides = {y_s1, af::sym::kSymbolOne};

  af::ascir_op::Data data0("data0", graph);
  data0.ir_attr.SetIndex(0);
  SetNodeView(data0, af::DT_FLOAT16, input_axes, input_repeats, input_strides);
  af::ascir_op::Load load0("load0");
  load0.x = data0.y;
  SetNodeView(load0, af::DT_FLOAT16, input_axes, input_repeats, input_strides);
  af::ascir_op::Data data1("data1", graph);
  data1.ir_attr.SetIndex(1);
  SetNodeView(data1, af::DT_FLOAT16, input_axes, input_repeats, input_strides);
  af::ascir_op::Load load1("load1");
  load1.x = data1.y;
  SetNodeView(load1, af::DT_FLOAT16, input_axes, input_repeats, input_strides);
  af::ascir_op::CopySign copy_sign("x_copy_sign");
  copy_sign.x1 = load0.y;
  copy_sign.x2 = load1.y;
  SetNodeView(copy_sign, af::DT_FLOAT16, input_axes, input_repeats, input_strides);
  af::ascir_op::Data index("index", graph);
  index.ir_attr.SetIndex(2);
  SetNodeView(index, af::DT_INT32, output_axes, output_repeats, output_strides);
  af::ascir_op::Load index_load("index_load");
  index_load.x = index.y;
  SetNodeView(index_load, af::DT_INT32, output_axes, output_repeats, output_strides);
  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  indirect_load.x1 = copy_sign.y;
  indirect_load.x2 = index_load.y;
  indirect_load.ir_attr.SetAxis(1);
  SetNodeView(indirect_load, af::DT_FLOAT16, output_axes, output_repeats, output_strides);
  af::ascir_op::Store store("store");
  if (add_unary) {
    af::ascir_op::Abs pre_abs("pre_abs");
    pre_abs.x = indirect_load.y;
    SetNodeView(pre_abs, af::DT_FLOAT16, output_axes, output_repeats, output_strides);
    store.x = pre_abs.y;
  } else {
    store.x = indirect_load.y;
  }
  SetNodeView(store, af::DT_FLOAT16, output_axes, output_repeats, output_strides);
  af::ascir_op::Output output("output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  SetNodeView(output, af::DT_FLOAT16, output_axes, output_repeats, output_strides);
  return graph;
}

enum class OutputPostTopology { kSum, kExp2Sum, kAbsExp2Sum, kCastSum, kAddSum, kBesselK0Sum, kBesselK0 };

struct PostTopologyCase {
  OutputPostTopology topology;
  std::vector<const char *> chain;
  bool expect_simt;
  af::DataType post_dtype;
  bool direct_output;
};

void ExpectNodeView(const af::AscNodePtr &node, af::DataType dtype, const std::vector<af::AxisId> &axes,
                    const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides);

void ExpectPostTerminalGraph(af::AscGraph &graph, const PostTopologyCase &test_case, const af::AscNodePtr &producer) {
  const auto indirect_load = graph.FindNode("indirect_load");
  const auto store = graph.FindNode("store");
  const auto output = graph.FindNode("output");
  ASSERT_NE(indirect_load, nullptr);
  ASSERT_NE(store, nullptr);
  ASSERT_NE(output, nullptr);
  const auto &post_view = indirect_load->outputs()[0]->attr;
  const auto &terminal_view = store->outputs()[0]->attr;
  const auto sum = graph.FindNode("sum");
  if (test_case.topology == OutputPostTopology::kBesselK0) {
    ASSERT_TRUE(test_case.direct_output);
    ASSERT_EQ(sum, nullptr);
    const auto store_input = ascgen_utils::indirect_load::GetInputProducer(store, 0UL);
    ASSERT_NE(store_input, nullptr);
    EXPECT_EQ(store_input->GetName(), producer->GetName());
  } else {
    ASSERT_FALSE(test_case.direct_output);
    ASSERT_NE(sum, nullptr);
    ExpectNodeView(sum, test_case.post_dtype, terminal_view.axis, terminal_view.repeats, terminal_view.strides);
    const auto sum_input = ascgen_utils::indirect_load::GetInputProducer(sum, 0UL);
    const auto store_input = ascgen_utils::indirect_load::GetInputProducer(store, 0UL);
    ASSERT_NE(sum_input, nullptr);
    ASSERT_NE(store_input, nullptr);
    EXPECT_EQ(sum_input->GetName(), producer->GetName());
    EXPECT_EQ(store_input->GetName(), sum->GetName());
  }
  if (test_case.topology == OutputPostTopology::kAddSum) {
    const auto add = graph.FindNode("output_add");
    const auto addend_load = graph.FindNode("output_addend_load");
    ASSERT_NE(add, nullptr);
    ASSERT_NE(addend_load, nullptr);
    const auto addend_input = ascgen_utils::indirect_load::GetInputProducer(add, 1UL);
    ASSERT_NE(addend_input, nullptr);
    EXPECT_EQ(addend_input->GetName(), addend_load->GetName());
    ExpectNodeView(addend_load, af::DT_FLOAT16, post_view.axis, post_view.repeats, post_view.strides);
  }
  ExpectNodeView(store, test_case.post_dtype, terminal_view.axis, terminal_view.repeats, terminal_view.strides);
  ExpectNodeView(output, test_case.post_dtype, terminal_view.axis, terminal_view.repeats, terminal_view.strides);
  const auto output_input = ascgen_utils::indirect_load::GetInputProducer(output, 0UL);
  ASSERT_NE(output_input, nullptr);
  EXPECT_EQ(output_input->GetName(), store->GetName());
}

void ExpectNodeView(const af::AscNodePtr &node, af::DataType dtype, const std::vector<af::AxisId> &axes,
                    const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides) {
  ASSERT_NE(node, nullptr);
  ASSERT_FALSE(node->outputs().empty());
  const auto &view = node->outputs()[0]->attr;
  EXPECT_EQ(view.dtype, dtype) << node->GetName();
  EXPECT_EQ(view.axis, axes) << node->GetName();
  EXPECT_EQ(view.repeats, repeats) << node->GetName();
  EXPECT_EQ(view.strides, strides) << node->GetName();
}

void ExpectPostTopologyGraph(af::AscGraph &graph, const PostTopologyCase &test_case) {
  const auto indirect_load = graph.FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  const auto &post_view = indirect_load->outputs()[0]->attr;
  ExpectNodeView(indirect_load, af::DT_FLOAT16, post_view.axis, post_view.repeats, post_view.strides);
  af::AscNodePtr producer = indirect_load;
  for (const char *name : test_case.chain) {
    const auto node = graph.FindNode(name);
    ASSERT_NE(node, nullptr) << name;
    ExpectNodeView(node, test_case.post_dtype, post_view.axis, post_view.repeats, post_view.strides);
    const auto actual_producer = ascgen_utils::indirect_load::GetInputProducer(node, 0UL);
    ASSERT_NE(actual_producer, nullptr) << name;
    EXPECT_EQ(actual_producer->GetName(), producer->GetName()) << name;
    producer = node;
  }
  ExpectPostTerminalGraph(graph, test_case, producer);
}

void BuildAbsExp2Chain(const af::AscOpOutput &input, const std::vector<af::AxisId> &axes,
                       const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides,
                       af::ascir_op::Sum &sum) {
  af::ascir_op::Abs abs("output_abs");
  abs.x = input;
  SetNodeView(abs, af::DT_FLOAT16, axes, repeats, strides);
  SetVectorApi(abs);
  af::ascir_op::Exp2 exp2("output_exp2");
  exp2.x = abs.y;
  SetNodeView(exp2, af::DT_FLOAT16, axes, repeats, strides);
  SetVectorApi(exp2);
  sum.x = exp2.y;
}

void BuildAddChain(af::AscGraph &graph, const af::AscOpOutput &input, const std::vector<af::AxisId> &axes,
                   const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides,
                   af::ascir_op::Sum &sum) {
  af::ascir_op::Data addend("output_addend", graph);
  addend.ir_attr.SetIndex(2);
  SetNodeView(addend, af::DT_FLOAT16, axes, repeats, strides);
  af::ascir_op::Load addend_load("output_addend_load");
  addend_load.x = addend.y;
  SetNodeView(addend_load, af::DT_FLOAT16, axes, repeats, strides);
  addend_load.attr.api.compute_type = af::ComputeType::kComputeLoad;
  addend_load.attr.api.type = af::ApiType::kAPITypeCompute;
  addend_load.attr.api.unit = af::ComputeUnit::kUnitMTE2;
  af::ascir_op::Add add("output_add");
  add.x1 = input;
  add.x2 = addend_load.y;
  SetNodeView(add, af::DT_FLOAT16, axes, repeats, strides);
  SetVectorApi(add);
  sum.x = add.y;
}

void BuildOutputPostChain(af::AscGraph &graph, OutputPostTopology topology, const af::AscOpOutput &input,
                          const std::vector<af::AxisId> &axes, const std::vector<af::Expression> &repeats,
                          const std::vector<af::Expression> &strides, af::ascir_op::Sum &sum,
                          af::ascir_op::Store &store, af::DataType &post_dtype, bool &direct_output) {
  if (topology == OutputPostTopology::kAbsExp2Sum) {
    BuildAbsExp2Chain(input, axes, repeats, strides, sum);
    return;
  }
  if (topology == OutputPostTopology::kAddSum) {
    BuildAddChain(graph, input, axes, repeats, strides, sum);
    return;
  }
  if (topology == OutputPostTopology::kExp2Sum) {
    af::ascir_op::Exp2 exp2("output_exp2");
    exp2.x = input;
    SetNodeView(exp2, af::DT_FLOAT16, axes, repeats, strides);
    SetVectorApi(exp2);
    sum.x = exp2.y;
    return;
  }
  if (topology == OutputPostTopology::kCastSum) {
    af::ascir_op::Cast cast("output_cast");
    cast.x = input;
    SetNodeView(cast, af::DT_FLOAT, axes, repeats, strides);
    SetVectorApi(cast);
    post_dtype = af::DT_FLOAT;
    sum.x = cast.y;
    return;
  }
  if (topology == OutputPostTopology::kBesselK0Sum || topology == OutputPostTopology::kBesselK0) {
    af::ascir_op::ModifiedBesselK0 bessel("output_modified_bessel_k0");
    bessel.x = input;
    SetNodeView(bessel, af::DT_FLOAT16, axes, repeats, strides);
    SetVectorApi(bessel);
    if (topology == OutputPostTopology::kBesselK0) {
      store.x = bessel.y;
      direct_output = true;
    } else {
      sum.x = bessel.y;
    }
    return;
  }
  sum.x = input;
}

af::AscGraph BuildPostReduceGraph(const std::string &suffix, bool reduce_outer = false,
                                  OutputPostTopology topology = OutputPostTopology::kSum,
                                  bool cast_reduce_output = false, bool invalid_reduce_successor = false) {
  af::AscGraph graph("indirect_load_post_reduce_ut_graph");
  const af::Expression s0 = graph.CreateSizeVar(2);
  const af::Expression s1 = suffix[0] == 'B' ? af::Expression(af::sym::kSymbolOne) : graph.CreateSizeVar(3);
  const af::Expression s2 = suffix[1] == 'B' ? af::Expression(af::sym::kSymbolOne) : graph.CreateSizeVar(4);
  const af::Expression s3 = suffix[2] == 'B' ? af::Expression(af::sym::kSymbolOne) : graph.CreateSizeVar(5);
  const auto x0 = graph.CreateAxis("x0", s0);
  const auto x1 = graph.CreateAxis("x1", s1);
  const auto x2 = graph.CreateAxis("x2", s2);
  const auto x3 = graph.CreateAxis("x3", s3);
  const auto y0 = graph.CreateAxis("y0", s0);
  const auto y1 = graph.CreateAxis("y1", s1);
  const auto y2 = graph.CreateAxis("y2", s2);
  const auto y3 = graph.CreateAxis("y3", s3);
  const std::vector<af::AxisId> input_axes = {x0.id, x1.id, x2.id, x3.id};
  const std::vector<af::AxisId> output_axes = {y0.id, y1.id, y2.id, y3.id};
  const std::vector<af::Expression> input_repeats = {s0, s1, s2, s3};
  const std::vector<af::Expression> output_repeats = {s0, s1, s2, s3};
  std::vector<af::Expression> reduce_repeats = output_repeats;
  const std::vector<af::Expression> input_strides = {s1 * s2 * s3, s2 * s3, s3, af::sym::kSymbolOne};
  const std::vector<af::Expression> output_strides = {s1 * s2 * s3, s2 * s3, s3, af::sym::kSymbolOne};
  std::vector<af::Expression> reduce_input_strides = output_strides;
  std::vector<af::Expression> reduce_strides = output_strides;
  if (reduce_outer) {
    reduce_repeats[0] = af::sym::kSymbolOne;
    reduce_strides[0] = af::sym::kSymbolZero;
  }
  for (size_t i = 1UL; i < reduce_strides.size(); ++i) {
    if (suffix[i - 1UL] == 'R') {
      reduce_repeats[i] = af::sym::kSymbolOne;
      reduce_strides[i] = af::sym::kSymbolZero;
    } else if (suffix[i - 1UL] == 'B') {
      reduce_input_strides[i] = af::sym::kSymbolZero;
      reduce_strides[i] = af::sym::kSymbolZero;
    }
  }

  af::ascir_op::Data x("x", graph);
  x.ir_attr.SetIndex(0);
  SetNodeView(x, af::DT_FLOAT16, input_axes, input_repeats, input_strides);
  af::ascir_op::Load input_load("input_load");
  input_load.x = x.y;
  SetNodeView(input_load, af::DT_FLOAT16, input_axes, input_repeats, input_strides);
  af::ascir_op::Data index("index", graph);
  index.ir_attr.SetIndex(1);
  SetNodeView(index, af::DT_INT32, output_axes, output_repeats, output_strides);
  af::ascir_op::Load index_load("index_load");
  index_load.x = index.y;
  SetNodeView(index_load, af::DT_INT32, output_axes, output_repeats, output_strides);
  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  indirect_load.x1 = input_load.y;
  indirect_load.x2 = index_load.y;
  indirect_load.ir_attr.SetAxis(1);
  SetNodeView(indirect_load, af::DT_FLOAT16, output_axes, output_repeats, output_strides);
  af::ascir_op::Sum sum("sum");
  af::ascir_op::Store store("store");
  af::DataType post_dtype = af::DT_FLOAT16;
  bool direct_output = false;
  if (suffix.find('B') == std::string::npos) {
    BuildOutputPostChain(graph, topology, indirect_load.y, output_axes, output_repeats, reduce_input_strides, sum,
                         store, post_dtype, direct_output);
  } else {
    af::ascir_op::Abs common_zero_view("common_zero_view");
    common_zero_view.x = indirect_load.y;
    SetNodeView(common_zero_view, af::DT_FLOAT16, output_axes, output_repeats, reduce_input_strides);
    BuildOutputPostChain(graph, topology, common_zero_view.y, output_axes, output_repeats, reduce_input_strides, sum,
                         store, post_dtype, direct_output);
  }
  if (!direct_output) {
    sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
    sum.attr.sched.axis = output_axes;
    SetNodeView(sum, post_dtype, output_axes, reduce_repeats, reduce_strides);
    if (cast_reduce_output) {
      // Model the Cast inserted by DtypeConsistency between Reduce and Store.
      af::ascir_op::Cast reduce_cast("reduce_cast");
      reduce_cast.x = sum.y;
      SetNodeView(reduce_cast, post_dtype, output_axes, reduce_repeats, reduce_strides);
      store.x = reduce_cast.y;
    } else {
      store.x = sum.y;
    }
  }
  const auto &store_repeats = direct_output ? output_repeats : reduce_repeats;
  const auto &store_strides = direct_output ? output_strides : reduce_strides;
  SetNodeView(store, post_dtype, output_axes, store_repeats, store_strides);
  af::ascir_op::Output output("output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  SetNodeView(output, post_dtype, output_axes, store_repeats, store_strides);
  if (invalid_reduce_successor) {
    af::ascir_op::Abs invalid_successor("invalid_reduce_successor");
    invalid_successor.x = sum.y;
    SetNodeView(invalid_successor, post_dtype, output_axes, reduce_repeats, reduce_strides);
    const auto sum_node = graph.FindNode("sum");
    const auto store_node = graph.FindNode("store");
    const auto invalid_node = graph.FindNode("invalid_reduce_successor");
    EXPECT_NE(sum_node, nullptr);
    EXPECT_NE(store_node, nullptr);
    EXPECT_NE(invalid_node, nullptr);
    if (sum_node != nullptr && store_node != nullptr && invalid_node != nullptr) {
      EXPECT_EQ(af::GraphUtils::ReplaceEdgeSrc(sum_node->GetOutDataAnchor(0), store_node->GetInDataAnchor(0),
                                               invalid_node->GetOutDataAnchor(0)),
                af::GRAPH_SUCCESS);
    }
  }
  return graph;
}

af::AscGraph BuildShapeEnlargingInputPreGraph() {
  af::AscGraph graph("indirect_load_shape_enlarging_input_pre_ut_graph");
  const af::Expression x_s0 = graph.CreateSizeVar(2);
  const af::Expression x_s1 = graph.CreateSizeVar(16);
  const af::Expression y_s0 = graph.CreateSizeVar(2);
  const af::Expression y_s1 = graph.CreateSizeVar(17);
  const auto x0 = graph.CreateAxis("x0", x_s0);
  const auto x1 = graph.CreateAxis("x1", x_s1);
  const auto y0 = graph.CreateAxis("y0", y_s0);
  const auto y1 = graph.CreateAxis("y1", y_s1);
  const std::vector<af::AxisId> input_axes = {x0.id, x1.id};
  const std::vector<af::AxisId> output_axes = {y0.id, y1.id};
  const std::vector<af::Expression> input_repeats = {x_s0, x_s1};
  const std::vector<af::Expression> output_repeats = {y_s0, y_s1};
  const std::vector<af::Expression> input_strides = {x_s1, af::sym::kSymbolOne};
  const std::vector<af::Expression> output_strides = {y_s1, af::sym::kSymbolOne};

  af::ascir_op::Data data("data", graph);
  data.ir_attr.SetIndex(0);
  SetNodeView(data, af::DT_FLOAT16, input_axes, input_repeats, input_strides);
  af::ascir_op::Load load("input_load");
  load.x = data.y;
  SetNodeView(load, af::DT_FLOAT16, input_axes, input_repeats, input_strides);
  af::ascir_op::Cast pre_cast("pre_cast");
  pre_cast.x = load.y;
  SetNodeView(pre_cast, af::DT_FLOAT, input_axes, input_repeats, input_strides);
  af::ascir_op::Data index("index", graph);
  index.ir_attr.SetIndex(1);
  SetNodeView(index, af::DT_INT32, output_axes, output_repeats, output_strides);
  af::ascir_op::Load index_load("index_load");
  index_load.x = index.y;
  SetNodeView(index_load, af::DT_INT32, output_axes, output_repeats, output_strides);
  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  indirect_load.x1 = pre_cast.y;
  indirect_load.x2 = index_load.y;
  indirect_load.ir_attr.SetAxis(1);
  SetNodeView(indirect_load, af::DT_FLOAT, output_axes, output_repeats, output_strides);
  af::ascir_op::Store store("store");
  store.x = indirect_load.y;
  SetNodeView(store, af::DT_FLOAT, output_axes, output_repeats, output_strides);
  af::ascir_op::Output output("output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  SetNodeView(output, af::DT_FLOAT, output_axes, output_repeats, output_strides);
  return graph;
}

int32_t ParseScore(const std::string &score_function) {
  const std::string key = "return ";
  const size_t pos = score_function.find(key);
  EXPECT_NE(pos, std::string::npos);
  return pos == std::string::npos ? -1 : std::stoi(score_function.substr(pos + key.size()));
}

std::vector<af::AscGraph> GenerateIndirectLoadCases(int64_t axis) {
  auto graph = BuildIndirectLoadGraph(axis);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  EXPECT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  EXPECT_EQ(graphs.size(), 4UL);
  EXPECT_EQ(score_functions.size(), 4UL);
  // 动态 shape 下 SIMD 候选只能退化到逐元素 strided facade，SIMT 候选得分严格更高；
  // 两个 SIMD 候选与 SK 候选维持同分。
  EXPECT_EQ(score_functions[0], score_functions[1]);
  EXPECT_EQ(score_functions[0], score_functions[3]);
  EXPECT_GT(ParseScore(score_functions[2]), ParseScore(score_functions[0]));
  return graphs;
}

std::vector<af::AscGraph>::iterator FindGeneratedGraphByTemplate(std::vector<af::AscGraph> &graphs,
                                                                 ascir::TemplateId template_id) {
  return std::find_if(graphs.begin(), graphs.end(), [template_id](const af::AscGraph &graph) {
    const auto node = graph.FindNode("indirect_load");
    return node != nullptr && ascir::GetTemplateIdOrDefault(*node) == template_id;
  });
}

TEST(IndirectLoadScheduleCaseGeneratorTest, SimtSetsDcacheAndUsesUnifiedVectorizedAxisWithoutReduce) {
  auto graphs = GenerateIndirectLoadCases(2);
  const auto simt_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt);
  ASSERT_NE(simt_iter, graphs.end());
  auto &simt_graph = *simt_iter;
  const auto indirect_load = simt_graph.FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);

  EXPECT_EQ(ascir::GetDcacheSize(*indirect_load), kSimtDcacheSize);
  ExpectMergedFrom(simt_graph, "indirect_load_outer", {"z4", "z5", "z6", "z7"});
  ascgen_utils::indirect_load::TemplateAxes axes;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, axes), af::SUCCESS);
  ExpectAxisNames(simt_graph, indirect_load->attr.sched.axis, {"z4", "z5", "z6", "z7"});
  ExpectAxisNames(simt_graph, {axes.outer_axis}, {"indirect_load_outer"});
  EXPECT_TRUE(indirect_load->outputs()[0]->attr.vectorized_axis.empty());
  EXPECT_TRUE(indirect_load->outputs()[0]->attr.vectorized_strides.empty());

  const auto behavior = ascgen_utils::indirect_load::GetTemplateBehavior(indirect_load);
  EXPECT_FALSE(behavior.skips_main_schedule_tiling);
  EXPECT_FALSE(behavior.skips_api_emit);
  EXPECT_TRUE(behavior.uses_direct_gm_pipeline);
  EXPECT_TRUE(behavior.skips_ub_lifecycle);
  EXPECT_TRUE(behavior.preserves_vectorized_axis);
  EXPECT_FALSE(ascgen_utils::indirect_load::ShouldApplyInputInnerVectorization(indirect_load));
}

// [Bug A] index 链尾部的区域外直连 store：SIMT 候选将其纳入融合区域并获得
// kSimtDirectGmBoundary 角色（跳过普通调度发射）；lowering 收为空节点输出链，
// 链值由 codegen 侧 Index() 求值器结果注入。
TEST(IndirectLoadScheduleCaseGeneratorTest, SimtFoldsIndexChainExternalStoreIntoRegion) {
  auto graph = BuildIndirectLoadGraph(2, false, false, IndexChainExtraConsumer::kExternalStore);
  // 链去重按 target tensor id 判定，图构造需保证关键边/目标张量 id 互异。
  const auto index_load_node = graph.FindNode("index_load");
  ASSERT_NE(index_load_node, nullptr);
  index_load_node->outputs()[0]->attr.mem.tensor_id = 100;
  const auto index_store_node = graph.FindNode("index_store");
  ASSERT_NE(index_store_node, nullptr);
  index_store_node->outputs()[0]->attr.mem.tensor_id = 101;
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  EXPECT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  const auto simt_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt);
  ASSERT_NE(simt_iter, graphs.end());
  const auto index_store = simt_iter->FindNode("index_store");
  ASSERT_NE(index_store, nullptr);
  EXPECT_EQ(ascgen_utils::indirect_load::GetTemplateRole(index_store),
            ascgen_utils::indirect_load::TemplateRole::kSimtDirectGmBoundary);
  const auto indirect_load = simt_iter->FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  ascgen_utils::indirect_load::IndirectLoadLoweringMetadata metadata;
  ASSERT_EQ(ascgen_utils::indirect_load::GetLoweringMetadata(indirect_load, metadata), af::SUCCESS);
  ASSERT_EQ(metadata.simt.output_chains.size(), 2UL);
  const auto &index_chain = metadata.simt.output_chains.back();
  EXPECT_TRUE(index_chain.node_names.empty());
  EXPECT_EQ(index_chain.result_tensor_id, metadata.simt.index_result_tensor_id);
  EXPECT_EQ(index_chain.result_tensor_id, 100);
  EXPECT_EQ(index_chain.target_tensor_id, 101);
  EXPECT_EQ(index_chain.dtype, af::DT_INT32);
}

// [Bug A 非法形态] index 链尾部喂给非 store 的外部消费者（Abs→store）：该形态
// 无法由 SIMT 标量求值器供值，静默丢弃会导致目标 buffer 永不写入，SIMT 候选
// 必须整候选淘汰。
TEST(IndirectLoadScheduleCaseGeneratorTest, SimtRejectsIndexChainIllegalExternalConsumer) {
  auto graph = BuildIndirectLoadGraph(2, false, false, IndexChainExtraConsumer::kIllegalConsumer);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  EXPECT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  EXPECT_EQ(FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt), graphs.end());
}

TEST(IndirectLoadScheduleCaseGeneratorTest, ClassifiesDenseLayout) {
  const ascgen_utils::indirect_load::LogicalTensorView view = {
      {0, 1, 2}, {af::Symbol(2), af::Symbol(3), af::Symbol(4)}, {af::Symbol(12), af::Symbol(4), af::Symbol(1)}};
  ascgen_utils::indirect_load::IndirectLoadTensorLayout layout;
  ASSERT_EQ(ascgen_utils::indirect_load::ClassifyIndirectLoadLayout(view, layout), af::SUCCESS);
  EXPECT_EQ(layout.kind, ascgen_utils::indirect_load::IndirectLoadLayoutKind::kDense);
  EXPECT_EQ(layout.physical_repeats, view.sizes);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, ClassifiesZeroStrideCompactLayout) {
  const ascgen_utils::indirect_load::LogicalTensorView view = {
      {0, 1, 2}, {af::Symbol(2), af::Symbol(3), af::Symbol(4)}, {af::Symbol(4), af::Symbol(0), af::Symbol(1)}};
  ascgen_utils::indirect_load::IndirectLoadTensorLayout layout;
  ASSERT_EQ(ascgen_utils::indirect_load::ClassifyIndirectLoadLayout(view, layout), af::SUCCESS);
  EXPECT_EQ(layout.kind, ascgen_utils::indirect_load::IndirectLoadLayoutKind::kZeroStrideCompact);
  EXPECT_EQ(layout.physical_repeats, (std::vector<af::Expression>{af::Symbol(2), af::Symbol(1), af::Symbol(4)}));
}

TEST(IndirectLoadScheduleCaseGeneratorTest, ClassifiesNonOverlappingStridedLayout) {
  const ascgen_utils::indirect_load::LogicalTensorView view = {
      {0, 1, 2}, {af::Symbol(2), af::Symbol(3), af::Symbol(4)}, {af::Symbol(20), af::Symbol(4), af::Symbol(1)}};
  ascgen_utils::indirect_load::IndirectLoadTensorLayout layout;
  ASSERT_EQ(ascgen_utils::indirect_load::ClassifyIndirectLoadLayout(view, layout), af::SUCCESS);
  EXPECT_EQ(layout.kind, ascgen_utils::indirect_load::IndirectLoadLayoutKind::kStrided);
  EXPECT_EQ(layout.physical_repeats, view.sizes);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, ClassifiesDynamicShapeAsStridedLayout) {
  const af::Expression s0 = af::Symbol("s0");
  const af::Expression s1 = af::Symbol("s1");
  const ascgen_utils::indirect_load::LogicalTensorView view = {{0, 1}, {s0, s1}, {s1, af::Symbol(1)}};
  ascgen_utils::indirect_load::IndirectLoadTensorLayout layout;
  ASSERT_EQ(ascgen_utils::indirect_load::ClassifyIndirectLoadLayout(view, layout), af::SUCCESS);
  EXPECT_EQ(layout.kind, ascgen_utils::indirect_load::IndirectLoadLayoutKind::kStrided);
  EXPECT_EQ(layout.physical_repeats, view.sizes);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, RejectsOverlappingStridedLayout) {
  const ascgen_utils::indirect_load::LogicalTensorView view = {
      {0, 1, 2}, {af::Symbol(2), af::Symbol(3), af::Symbol(4)}, {af::Symbol(10), af::Symbol(4), af::Symbol(1)}};
  ascgen_utils::indirect_load::IndirectLoadTensorLayout layout;
  ASSERT_EQ(ascgen_utils::indirect_load::ClassifyIndirectLoadLayout(view, layout), af::SUCCESS);
  EXPECT_EQ(layout.kind, ascgen_utils::indirect_load::IndirectLoadLayoutKind::kUnsupported);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, RejectsMixedZeroStrideAndPhysicalGapLayout) {
  const ascgen_utils::indirect_load::LogicalTensorView view = {
      {0, 1, 2}, {af::Symbol(2), af::Symbol(3), af::Symbol(4)}, {af::Symbol(8), af::Symbol(0), af::Symbol(1)}};
  ascgen_utils::indirect_load::IndirectLoadTensorLayout layout;
  ASSERT_EQ(ascgen_utils::indirect_load::ClassifyIndirectLoadLayout(view, layout), af::SUCCESS);
  EXPECT_EQ(layout.kind, ascgen_utils::indirect_load::IndirectLoadLayoutKind::kUnsupported);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, ClassifiesMixedZeroStrideAndPhysicalGapLayoutForSimt) {
  const ascgen_utils::indirect_load::LogicalTensorView view = {
      {0, 1, 2}, {af::Symbol(2), af::Symbol(3), af::Symbol(4)}, {af::Symbol(8), af::Symbol(0), af::Symbol(1)}};
  ascgen_utils::indirect_load::IndirectLoadTensorLayout layout;
  ASSERT_EQ(ascgen_utils::indirect_load::ClassifyIndirectLoadLayout(view, layout, true), af::SUCCESS);
  EXPECT_EQ(layout.kind, ascgen_utils::indirect_load::IndirectLoadLayoutKind::kStrided);
  EXPECT_EQ(layout.physical_repeats, view.sizes);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, RejectsInvalidLogicalLayout) {
  ascgen_utils::indirect_load::IndirectLoadTensorLayout layout;
  const ascgen_utils::indirect_load::LogicalTensorView rank_mismatch = {
      {0, 1}, {af::Symbol(2)}, {af::Symbol(1), af::Symbol(1)}};
  EXPECT_NE(ascgen_utils::indirect_load::ClassifyIndirectLoadLayout(rank_mismatch, layout), af::SUCCESS);

  const ascgen_utils::indirect_load::LogicalTensorView negative_stride = {
      {0, 1}, {af::Symbol(2), af::Symbol(3)}, {af::Symbol(-3), af::Symbol(1)}};
  ASSERT_EQ(ascgen_utils::indirect_load::ClassifyIndirectLoadLayout(negative_stride, layout), af::SUCCESS);
  EXPECT_EQ(layout.kind, ascgen_utils::indirect_load::IndirectLoadLayoutKind::kUnsupported);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, KeepsBroadcastElementPathForGeneralInline) {
  PlatformContextReset platform_reset;
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformInfo platform_info;
  platform_info.soc_ver = "3510";
  platform_info.ub_size = 256 * 1024;
  platform_info.aiv_num = 48;
  ge::PlatformContext::GetInstance().SetPlatformInfo(platform_info);
  auto graph = BuildIndirectLoadBroadcastGraph();
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  ASSERT_EQ(score_functions.size(), graphs.size());
  const auto simd = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(simd, graphs.end());
  EXPECT_NE(simd->FindNode("input_broadcast"), nullptr);
  EXPECT_NE(simd->FindNode("broadcast_input_abs"), nullptr);
  const auto indirect_load = simd->FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  ascgen_utils::indirect_load::TemplateLogicalView logical_view;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateLogicalView(indirect_load, logical_view), af::SUCCESS);
  EXPECT_EQ(logical_view.input.kind, ascgen_utils::indirect_load::IndirectLoadLayoutKind::kZeroStrideCompact);

  for (const ascir::TemplateId template_id :
       {ascir::TemplateId::kIndirectLoadSimt, ascir::TemplateId::kIndirectLoadSK}) {
    const auto candidate = FindGeneratedGraphByTemplate(graphs, template_id);
    ASSERT_NE(candidate, graphs.end());
    EXPECT_EQ(candidate->FindNode("input_broadcast"), nullptr);
  }
}

TEST(IndirectLoadScheduleCaseGeneratorTest, AcceptsOuterInputBroadcastForSimd) {
  auto graph = BuildIndirectLoadBroadcastGraph(false, false, 2L);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);

  const auto simd = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(simd, graphs.end());
  EXPECT_NE(simd->FindNode("input_broadcast"), nullptr);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, AcceptsOuterIndexBroadcastForSimd) {
  auto graph = BuildIndirectLoadBroadcastGraph(false, true, 2L);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);

  const auto simd = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(simd, graphs.end());
  EXPECT_NE(simd->FindNode("index_outer_broadcast"), nullptr);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, SimtRewritesInputBroadcastAndSetsTemplateMetadata) {
  PlatformContextReset platform_reset;
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformInfo platform_info;
  platform_info.soc_ver = "3510";
  platform_info.ub_size = 256 * 1024;
  platform_info.aiv_num = 48;
  ge::PlatformContext::GetInstance().SetPlatformInfo(platform_info);
  auto graph = BuildIndirectLoadBroadcastGraph();
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);

  const auto simt = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt);
  ASSERT_NE(simt, graphs.end());
  EXPECT_EQ(simt->FindNode("input_broadcast"), nullptr);
  const auto indirect_load = simt->FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  EXPECT_EQ(ascir::GetDcacheSize(*indirect_load), kSimtDcacheSize);
  EXPECT_TRUE(indirect_load->outputs()[0]->attr.vectorized_axis.empty());
  EXPECT_TRUE(indirect_load->outputs()[0]->attr.vectorized_strides.empty());
}

TEST(IndirectLoadScheduleCaseGeneratorTest, BroadcastAfterTransposePreservesSourcePhysicalView) {
  PlatformContextReset platform_reset;
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformInfo platform_info;
  platform_info.soc_ver = "3510";
  platform_info.ub_size = 256 * 1024;
  platform_info.aiv_num = 48;
  ge::PlatformContext::GetInstance().SetPlatformInfo(platform_info);
  auto graph = BuildIndirectLoadBroadcastGraph(false, false, 2L, true);
  const auto source = graph.FindNode("broadcast_input_load")->outputs[0].attr;
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  ASSERT_EQ(graphs.size(), 3UL);
  for (auto &candidate : graphs) {
    const auto il = candidate.FindNode("indirect_load");
    ASSERT_NE(il, nullptr);
    EXPECT_NE(ascir::GetTemplateIdOrDefault(*il), ascir::TemplateId::kIndirectLoadSK);
    const auto load = candidate.FindNode("broadcast_input_load");
    ASSERT_NE(load, nullptr);
    const auto transpose = candidate.FindNode("broadcast_input_transpose");
    ascgen_utils::indirect_load::TemplateLogicalView view;
    ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateLogicalView(il, view), af::SUCCESS);
    // The scheduler must publish the final GM address strides for SIMT as well as SIMD.
    EXPECT_EQ(view.input.strides, (std::vector<af::Expression>{af::ops::One, af::ops::Zero, af::Symbol(2)}));
    if (ascir::GetTemplateIdOrDefault(*il) == ascir::TemplateId::kIndirectLoadSimd) {
      EXPECT_EQ(transpose, nullptr);
      // 折叠输入侧 Broadcast/Transpose 后，Load 节点视图被重写为中间形态（bf5d926c）；
      // 最终 GM 地址 strides 由 TemplateLogicalView 统一发布（上方 view.input 断言）。
      continue;
    }
    ASSERT_NE(transpose, nullptr);
    EXPECT_EQ(load->outputs[0].attr.axis.front(), transpose->outputs[0].attr.axis.back());
    EXPECT_EQ(load->outputs[0].attr.axis.back(), transpose->outputs[0].attr.axis.front());
    EXPECT_EQ(load->outputs[0].attr.repeats, source.repeats);
    // SIMT 保留 Transpose 表达 permutation；Load 视图同样被重写为中间形态，
    // 最终 GM 地址 strides 由 TemplateLogicalView 的 {1,0,2} 提供（上方断言）。
  }
}

TEST(IndirectLoadScheduleCaseGeneratorTest, PreservesSharedOrControlledTransposeLoadBoundary) {
  PlatformContextReset platform_reset;
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformInfo platform_info;
  platform_info.soc_ver = "3510";
  platform_info.ub_size = 256 * 1024;
  platform_info.aiv_num = 48;
  ge::PlatformContext::GetInstance().SetPlatformInfo(platform_info);
  for (const bool control_edge : {false, true}) {
    SCOPED_TRACE(control_edge);
    auto graph = BuildIndirectLoadBroadcastGraph(false, false, 2L, true);
    const auto load = graph.FindNode("broadcast_input_load");
    ASSERT_NE(load, nullptr);
    const auto source = load->outputs[0].attr;
    if (control_edge) {
      const auto index = graph.FindNode("broadcast_index_load");
      ASSERT_NE(index, nullptr);
      ASSERT_EQ(af::GraphUtils::AddEdge(load->GetOutControlAnchor(), index->GetInControlAnchor()), af::SUCCESS);
    } else {
      af::ascir_op::Store shared_store("shared_store");
      graph.AddNode(shared_store);
      SetNodeView(shared_store, source.dtype, source.axis, source.repeats, source.strides);
      ASSERT_EQ(af::GraphUtils::AddEdge(load->GetOutDataAnchor(0), graph.FindNode("shared_store")->GetInDataAnchor(0)),
                af::SUCCESS);
      af::ascir_op::Output shared_output("shared_output");
      graph.AddNode(shared_output);
      shared_output.ir_attr.SetIndex(1);
      shared_output.x = shared_store.y;
      SetNodeView(shared_output, source.dtype, source.axis, source.repeats, source.strides);
    }
    optimize::IndirectLoadScheduleCaseGenerator generator;
    std::vector<af::AscGraph> graphs;
    std::vector<std::string> score_functions;
    ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
    ASSERT_EQ(graphs.size(), 2UL);
    for (auto &candidate : graphs) {
      const auto input_load = candidate.FindNode("broadcast_input_load");
      ASSERT_NE(input_load, nullptr);
      EXPECT_NE(candidate.FindNode("broadcast_input_transpose"), nullptr);
      EXPECT_EQ(input_load->outputs[0].attr.repeats, source.repeats);
      EXPECT_EQ(input_load->outputs[0].attr.strides, source.strides);
      EXPECT_TRUE(ascgen_utils::indirect_load::ShouldApplyInputInnerVectorization(input_load));
    }
  }
}

TEST(IndirectLoadScheduleCaseGeneratorTest, BroadcastDirectPathUsesPhysicalViewForAllTemplates) {
  PlatformContextReset platform_reset;
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformInfo platform_info;
  platform_info.soc_ver = "3510";
  platform_info.ub_size = 256 * 1024;
  platform_info.aiv_num = 48;
  ge::PlatformContext::GetInstance().SetPlatformInfo(platform_info);
  auto graph = BuildIndirectLoadBroadcastGraph(false);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  EXPECT_EQ(graphs.size(), 4UL);
  ASSERT_EQ(score_functions.size(), graphs.size());
  for (auto &candidate : graphs) {
    const auto indirect_load = candidate.FindNode("indirect_load");
    ASSERT_NE(indirect_load, nullptr);
    const auto template_id = ascir::GetTemplateIdOrDefault(*indirect_load);
    if (template_id == ascir::TemplateId::kIndirectLoadSimd) {
      EXPECT_NE(candidate.FindNode("input_broadcast"), nullptr);
    } else {
      EXPECT_EQ(candidate.FindNode("input_broadcast"), nullptr);
    }
    EXPECT_EQ(candidate.FindNode("broadcast_input_abs"), nullptr);
    ascgen_utils::indirect_load::TemplateLogicalView logical_view;
    ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateLogicalView(indirect_load, logical_view), af::SUCCESS);
    EXPECT_EQ(logical_view.input.kind, ascgen_utils::indirect_load::IndirectLoadLayoutKind::kZeroStrideCompact);
  }

  auto inner_graph = BuildIndirectLoadBroadcastGraph(false, false, 0L);
  std::vector<af::AscGraph> inner_graphs;
  std::vector<std::string> inner_score_functions;
  ASSERT_EQ(generator.Generate(inner_graph, inner_graphs, inner_score_functions), af::SUCCESS);
  const auto inner_simd = FindGeneratedGraphByTemplate(inner_graphs, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(inner_simd, inner_graphs.end());
  EXPECT_NE(inner_simd->FindNode("input_broadcast"), nullptr);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, SimdDirectBroadcastUsesUnserializedTileSplit) {
  auto graph = BuildIndirectLoadBroadcastGraph(false, false, 1L);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  const auto simd = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(simd, graphs.end());
  const auto indirect_load = simd->FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  ascgen_utils::indirect_load::TemplateAxes axes;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, axes), af::SUCCESS);
  // 该图 index 视图带行间 padding（步长为稠密的 2 倍）：多行 staging 会剥离 padding 而
  // 模板按逻辑步长寻址，两者不一致，保持固定单行 tile。
  ExpectFixedTileSplit(*simd, axes.outer_axis);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, IndexBroadcastUsesFinalPhysicalView) {
  PlatformContextReset platform_reset;
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformInfo platform_info;
  platform_info.soc_ver = "3510";
  platform_info.ub_size = 256 * 1024;
  platform_info.aiv_num = 48;
  ge::PlatformContext::GetInstance().SetPlatformInfo(platform_info);
  auto graph = BuildIndirectLoadIndexBroadcastGraph();
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  const auto simt = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt);
  ASSERT_NE(simt, graphs.end());
  EXPECT_NE(simt->FindNode("index_broadcast"), nullptr);
  const auto indirect_load = simt->FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  ascgen_utils::indirect_load::TemplateLogicalView logical_view;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateLogicalView(indirect_load, logical_view), af::SUCCESS);
  EXPECT_EQ(logical_view.index.kind, ascgen_utils::indirect_load::IndirectLoadLayoutKind::kZeroStrideCompact);
  EXPECT_EQ(logical_view.index.sizes, (std::vector<af::Expression>{af::Symbol(1024), af::Symbol(1024)}));
  EXPECT_EQ(logical_view.index.strides, (std::vector<af::Expression>{af::ops::One, af::ops::Zero}));
  EXPECT_EQ(logical_view.index.physical_repeats, (std::vector<af::Expression>{af::Symbol(1024), af::ops::One}));
}

TEST(IndirectLoadScheduleCaseGeneratorTest, CompletesMissingDataViewAfterBroadcastRewrite) {
  auto graph = BuildIndirectLoadBroadcastGraph(false);
  const auto input = graph.FindNode("broadcast_x");
  ASSERT_NE(input, nullptr);
  ASSERT_EQ(input->outputs().size(), 1UL);
  input->outputs()[0]->attr.axis.clear();
  input->outputs()[0]->attr.repeats.clear();
  input->outputs()[0]->attr.strides.clear();

  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  const auto simd = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(simd, graphs.end());
  const auto completed_input = simd->FindNode("broadcast_x");
  const auto input_load = simd->FindNode("broadcast_input_load");
  ASSERT_NE(completed_input, nullptr);
  ASSERT_NE(input_load, nullptr);
  EXPECT_EQ(completed_input->outputs()[0]->attr.axis.size(), 3UL);
  EXPECT_EQ(completed_input->outputs()[0]->attr.repeats, input_load->outputs()[0]->attr.repeats);
  EXPECT_EQ(completed_input->outputs()[0]->attr.strides, input_load->outputs()[0]->attr.strides);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, BranchedBroadcastPathsKeepSimdCandidate) {
  // SIMD 不折叠 Broadcast，因此 Broadcast 或其源节点存在旁路消费者时仍可保留 SIMD candidate。
  for (const char *producer_name : {"input_broadcast", "broadcast_input_load"}) {
    auto graph = BuildIndirectLoadBroadcastGraph(false);
    ASSERT_TRUE(AddSideConsumer(graph, producer_name));
    optimize::IndirectLoadScheduleCaseGenerator generator;
    std::vector<af::AscGraph> graphs;
    std::vector<std::string> score_functions;
    EXPECT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS) << producer_name;
    EXPECT_NE(FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd), graphs.end())
        << producer_name;
    EXPECT_EQ(score_functions.size(), graphs.size()) << producer_name;
  }
}

TEST(IndirectLoadScheduleCaseGeneratorTest, SimdAcceptsStridedInputPath) {
  auto graph = BuildExpandedMultiInputGraph(false, 2L, 16L);
  SetStridedInputPath(graph);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  const auto simd = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(simd, graphs.end());
  const auto indirect_load = simd->FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  ascgen_utils::indirect_load::TemplateLogicalView logical_view;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateLogicalView(indirect_load, logical_view), af::SUCCESS);
  EXPECT_EQ(logical_view.input.strides, (std::vector<af::Expression>{af::Symbol(64), af::Symbol(2)}));
  for (const char *name : {"load0", "load1", "x_copy_sign"}) {
    const auto node = simd->FindNode(name);
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(ascgen_utils::indirect_load::GetTemplateRole(node),
              ascgen_utils::indirect_load::TemplateRole::kSimdInputPreStridedUbPath);
  }
}

TEST(IndirectLoadScheduleCaseGeneratorTest, StoresTemplateAxesWithoutOverwritingSchedAxis) {
  auto graphs = GenerateIndirectLoadCases(2);
  const auto simd_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(simd_iter, graphs.end());
  auto &simd_graph = *simd_iter;
  const auto indirect_load = simd_graph.FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);

  ascgen_utils::indirect_load::TemplateAxes axes;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, axes), af::SUCCESS);
  ExpectAxisNames(simd_graph, indirect_load->attr.sched.axis, {"z4", "z5", "z6", "z7"});
  ExpectAxisNames(simd_graph, {axes.outer_axis}, {"indirect_load_outer"});
  ExpectAxisNames(simd_graph, {axes.inner_axis}, {"indirect_load_inner"});
  // rank4/axis2（行维跨 z0/z1 两维，由 merged outer 承载）：多行 staging 无法映射为
  // 单一批量 DMA，保持生成器期固定 tile pair（可求解 tile 仅限 axis_index==1 形态）。
  ExpectFixedTileSplit(simd_graph, axes.outer_axis);

  ascgen_utils::indirect_load::TemplateLogicalView logical_view;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateLogicalView(indirect_load, logical_view), af::SUCCESS);
  ExpectAxisNames(simd_graph, logical_view.input.axis_ids, {"z0", "z1", "z2", "z3"});
  ExpectAxisNames(simd_graph, logical_view.index.axis_ids, {"z4", "z5", "z6", "z7"});
  ExpectAxisNames(simd_graph, logical_view.output.axis_ids, {"z4", "z5", "z6", "z7"});
  EXPECT_EQ(logical_view.input.sizes.size(), 4UL);
  EXPECT_EQ(logical_view.input.strides.size(), 4UL);
  EXPECT_EQ(logical_view.index.sizes.size(), 4UL);
  EXPECT_EQ(logical_view.index.strides.size(), 4UL);
  EXPECT_EQ(logical_view.output.sizes.size(), 4UL);
  EXPECT_EQ(logical_view.output.strides.size(), 4UL);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, StoresFixedTileAxesAsTemplateMetadata) {
  auto graphs = GenerateIndirectLoadCases(2);
  const auto simd_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(simd_iter, graphs.end());
  auto &simd_graph = *simd_iter;
  const auto indirect_load = simd_graph.FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  const auto op_desc = indirect_load->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  ascgen_utils::indirect_load::TemplateAxes axes;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, axes), af::SUCCESS);
  EXPECT_EQ(op_desc->TryGetExtAttr("af.internal.indirect_load.tile_outer_axis", static_cast<int64_t>(af::kIdNone)),
            axes.tile_outer_axis);
  EXPECT_EQ(op_desc->TryGetExtAttr("af.internal.indirect_load.tile_inner_axis", static_cast<int64_t>(af::kIdNone)),
            axes.tile_inner_axis);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, SkBuildsInputInnerAxisFromInputBoundary) {
  auto graph = BuildIndirectLoadGraph(2, true);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  const auto sk_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSK);
  ASSERT_NE(sk_iter, graphs.end());
  auto &sk_graph = *sk_iter;
  const auto indirect_load = sk_graph.FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  const auto input_boundary = ascgen_utils::indirect_load::GetInputProducer(indirect_load, 0UL);
  ASSERT_NE(input_boundary, nullptr);

  ascgen_utils::indirect_load::TemplateAxes axes;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, axes), af::SUCCESS);
  const auto *input_inner_axis = sk_graph.FindAxis(axes.input_inner_axis);
  ASSERT_NE(input_inner_axis, nullptr);
  EXPECT_EQ(ascgen_utils::indirect_load::GetTemplateRole(input_boundary),
            ascgen_utils::indirect_load::TemplateRole::kSkInputBoundary);
  ExpectAxisNames(sk_graph, input_boundary->outputs()[0]->attr.axis, {"z0", "z1", "z2", "z3"});
  ExpectAxisNames(sk_graph, input_inner_axis->from, {"z2", "z3"});
}

TEST(IndirectLoadScheduleCaseGeneratorTest, GeneratorTaskKeepsSkPartitionOrder) {
  auto graph = BuildIndirectLoadGraph(2);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<optimize::ScheduleTask> tasks;
  ASSERT_EQ(generator.GeneratorTask(graph, tasks, {}), af::SUCCESS);
  const auto sk_task = std::find_if(tasks.begin(), tasks.end(), [](const optimize::ScheduleTask &task) {
    const auto node = task.optimize_graph.FindNode("indirect_load");
    return node != nullptr && ascir::GetTemplateIdOrDefault(*node) == ascir::TemplateId::kIndirectLoadSK;
  });
  ASSERT_NE(sk_task, tasks.end());

  auto &grouped_graphs = sk_task->grouped_graphs;
  ASSERT_EQ(grouped_graphs.size(), 4UL);
  EXPECT_NE(grouped_graphs[0].FindNode("x"), nullptr);
  EXPECT_NE(grouped_graphs[1].FindNode("index"), nullptr);
  const auto indirect_load = grouped_graphs[2].FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  EXPECT_NE(grouped_graphs[3].FindNode("y"), nullptr);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, GeneratorTaskRestoresSkLogicalViewAfterPartition) {
  auto graph = BuildIndirectLoadGraph(2);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<optimize::ScheduleTask> tasks;
  ASSERT_EQ(generator.GeneratorTask(graph, tasks, {}), af::SUCCESS);
  const auto sk_task = std::find_if(tasks.begin(), tasks.end(), [](const optimize::ScheduleTask &task) {
    const auto node = task.optimize_graph.FindNode("indirect_load");
    return node != nullptr && ascir::GetTemplateIdOrDefault(*node) == ascir::TemplateId::kIndirectLoadSK;
  });
  ASSERT_NE(sk_task, tasks.end());

  const auto grouped_graph =
      std::find_if(sk_task->grouped_graphs.begin(), sk_task->grouped_graphs.end(),
                   [](const af::AscGraph &candidate) { return candidate.FindNode("indirect_load") != nullptr; });
  ASSERT_NE(grouped_graph, sk_task->grouped_graphs.end());
  const auto indirect_load = grouped_graph->FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);

  ascgen_utils::indirect_load::TemplateLogicalView logical_view;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateLogicalView(indirect_load, logical_view), af::SUCCESS);
  EXPECT_EQ(logical_view.input.axis_ids, indirect_load->inputs()[0]->attr.axis);
  EXPECT_EQ(logical_view.input.sizes, indirect_load->inputs()[0]->attr.repeats);
  EXPECT_EQ(logical_view.input.strides, indirect_load->inputs()[0]->attr.strides);
  EXPECT_EQ(logical_view.index.axis_ids, indirect_load->inputs()[1]->attr.axis);
  EXPECT_EQ(logical_view.index.sizes, indirect_load->inputs()[1]->attr.repeats);
  EXPECT_EQ(logical_view.index.strides, indirect_load->inputs()[1]->attr.strides);
  EXPECT_EQ(logical_view.output.axis_ids, indirect_load->outputs()[0]->attr.axis);
  EXPECT_EQ(logical_view.output.sizes, indirect_load->outputs()[0]->attr.repeats);
  EXPECT_EQ(logical_view.output.strides, indirect_load->outputs()[0]->attr.strides);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, GeneratorTaskPopulatesCandidateMetadata) {
  auto graph = BuildIndirectLoadGraph(2);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<optimize::ScheduleTask> tasks;
  ASSERT_EQ(generator.GeneratorTask(graph, tasks, {}), af::SUCCESS);

  ASSERT_EQ(tasks.size(), 4UL);
  for (const auto &task : tasks) {
    ASSERT_FALSE(task.score_func.empty());
    ASSERT_FALSE(task.grouped_graphs.empty());
    const auto indirect_load = task.optimize_graph.FindNode("indirect_load");
    ASSERT_NE(indirect_load, nullptr);
    const auto template_id = ascir::GetTemplateIdOrDefault(*indirect_load);
    ASSERT_TRUE(template_id == ascir::TemplateId::kIndirectLoadSimd ||
                template_id == ascir::TemplateId::kIndirectLoadSimt ||
                template_id == ascir::TemplateId::kIndirectLoadSK);
  }
}

TEST(IndirectLoadScheduleCaseGeneratorTest, SimtFanoutAnnotatesMainPathAndSideInputClosure) {
  PlatformContextReset platform_reset;
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformContext::GetInstance().SetPlatform("3510");
  auto graph = BuildIndirectLoadGraph(2);
  ASSERT_TRUE(AddSimtFanoutTopology(graph));

  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  const auto simt = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt);
  ASSERT_NE(simt, graphs.end());
  const auto main_transform = simt->FindNode("coverage_main_transform");
  const auto side_load = simt->FindNode("coverage_side_load");
  const auto side_pre = simt->FindNode("coverage_side_pre");
  const auto side_branch = simt->FindNode("coverage_side_branch");
  ASSERT_NE(main_transform, nullptr);
  ASSERT_NE(side_load, nullptr);
  ASSERT_NE(side_pre, nullptr);
  ASSERT_NE(side_branch, nullptr);
  EXPECT_EQ(ascgen_utils::indirect_load::GetTemplateRole(main_transform),
            ascgen_utils::indirect_load::TemplateRole::kSimtInlineTransform);
  EXPECT_EQ(ascgen_utils::indirect_load::GetTemplateRole(side_load),
            ascgen_utils::indirect_load::TemplateRole::kSimtDirectGmBoundary);
  EXPECT_EQ(ascgen_utils::indirect_load::GetTemplateRole(side_pre),
            ascgen_utils::indirect_load::TemplateRole::kSimtFanoutBranch);
  EXPECT_EQ(ascgen_utils::indirect_load::GetTemplateRole(side_branch),
            ascgen_utils::indirect_load::TemplateRole::kSimtFanoutBranch);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, SimdNormalizesNonExpandedGraph) {
  auto graph = BuildIndirectLoadGraph(2, true);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  const auto simd_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(simd_iter, graphs.end());
  auto &simd_graph = *simd_iter;
  const auto indirect_load = simd_graph.FindNode("indirect_load");
  const auto input_load = simd_graph.FindNode("input_load");
  const auto pre_abs = simd_graph.FindNode("pre_abs");
  ASSERT_NE(indirect_load, nullptr);
  ASSERT_NE(input_load, nullptr);
  ASSERT_NE(pre_abs, nullptr);

  EXPECT_EQ(
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kInputTensorIndex),
      input_load);
  EXPECT_EQ(ascgen_utils::indirect_load::GetTemplateRole(input_load),
            ascgen_utils::indirect_load::TemplateRole::kSimdInputPre);
  ExpectAxisNames(simd_graph, input_load->attr.sched.axis, {"z4", "z5", "z2", "z3"});
  ExpectAxisNames(simd_graph, input_load->outputs()[0]->attr.axis, {"z4", "z5", "z2", "z3"});
  EXPECT_EQ(ascgen_utils::indirect_load::GetInputProducer(pre_abs, 0UL), indirect_load);
  EXPECT_EQ(ascgen_utils::indirect_load::GetOnlyOutputConsumer(indirect_load), pre_abs);
  ExpectAxisNames(simd_graph, pre_abs->attr.sched.axis, {"z4", "z5", "z6", "z7"});
  ExpectAxisNames(simd_graph, pre_abs->outputs()[0]->attr.axis, {"z4", "z5", "z6", "z7"});

  ascgen_utils::indirect_load::TemplateAxes axes;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, axes), af::SUCCESS);
  ExpectAxisNames(simd_graph, indirect_load->attr.sched.axis, {"z4", "z5", "z6", "z7"});
  ExpectAxisNames(simd_graph, {axes.outer_axis}, {"indirect_load_outer"});
  ExpectAxisNames(simd_graph, {axes.inner_axis}, {"indirect_load_inner"});
  ExpectAxisNames(simd_graph, {axes.input_inner_axis}, {"indirect_load_input_inner"});
  ExpectMergedFrom(simd_graph, "indirect_load_input_inner", {"z2", "z3"});
  // 同 StoresTemplateAxesWithoutOverwritingSchedAxis：rank4/axis2 保持固定 tile pair。
  ExpectFixedTileSplit(simd_graph, axes.outer_axis);

  ascgen_utils::indirect_load::TemplateLogicalView logical_view;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateLogicalView(indirect_load, logical_view), af::SUCCESS);
  ExpectAxisNames(simd_graph, logical_view.input.axis_ids, {"z0", "z1", "z2", "z3"});
  ExpectAxisNames(simd_graph, logical_view.index.axis_ids, {"z4", "z5", "z6", "z7"});
  ExpectAxisNames(simd_graph, logical_view.output.axis_ids, {"z4", "z5", "z6", "z7"});
  EXPECT_EQ(logical_view.input.strides.size(), 4UL);
  EXPECT_EQ(logical_view.index.strides.size(), 4UL);
  EXPECT_EQ(logical_view.output.strides.size(), 4UL);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, SimdNormalizesNegativeAxis) {
  auto graph = BuildIndirectLoadGraph(-2);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  const auto simd_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(simd_iter, graphs.end());
  const auto indirect_load = simd_iter->FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  ascgen_utils::indirect_load::TemplateAxes axes;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, axes), af::SUCCESS);
  ExpectAxisNames(*simd_iter, {axes.outer_axis}, {"indirect_load_outer"});
  ExpectAxisNames(*simd_iter, {axes.inner_axis}, {"indirect_load_inner"});
  ExpectAxisNames(*simd_iter, {axes.input_inner_axis}, {"indirect_load_input_inner"});
  ExpectMergedFrom(*simd_iter, "indirect_load_input_inner", {"z2", "z3"});
  ascgen_utils::indirect_load::TemplateLogicalView logical_view;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateLogicalView(indirect_load, logical_view), af::SUCCESS);
  ExpectAxisNames(*simd_iter, logical_view.input.axis_ids, {"z0", "z1", "z2", "z3"});
  ExpectAxisNames(*simd_iter, logical_view.index.axis_ids, {"z4", "z5", "z6", "z7"});
  ExpectAxisNames(*simd_iter, logical_view.output.axis_ids, {"z4", "z5", "z6", "z7"});
}

TEST(IndirectLoadScheduleCaseGeneratorTest, SimdRegionMetadataAndSimtRejectsMultiInputRegion) {
  auto graph = BuildExpandedMultiInputGraph(true, 2L, 16L);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  ASSERT_EQ(graphs.size(), 3UL);
  const auto simd_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(simd_iter, graphs.end());
  const auto simt_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt);
  ASSERT_EQ(simt_iter, graphs.end());
  auto &simd_graph = *simd_iter;
  const auto indirect_load = simd_graph.FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);

  const auto copy_sign = simd_graph.FindNode("x_copy_sign");
  ASSERT_NE(copy_sign, nullptr);
  EXPECT_EQ(
      ascgen_utils::indirect_load::GetInputProducer(indirect_load, ascgen_utils::indirect_load::kInputTensorIndex),
      copy_sign);
  for (const char *name : {"load0", "load1", "x_copy_sign"}) {
    const auto node = simd_graph.FindNode(name);
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(ascgen_utils::indirect_load::GetTemplateRole(node),
              ascgen_utils::indirect_load::TemplateRole::kSimdInputPre);
    ExpectAxisNames(simd_graph, node->attr.sched.axis, {"y0", "x1"});
    ExpectAxisNames(simd_graph, node->outputs()[0]->attr.axis, {"y0", "x1"});
  }
  ascgen_utils::indirect_load::TemplateAxes axes;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, axes), af::SUCCESS);
  ExpectAxisNames(simd_graph, {axes.outer_axis, axes.inner_axis, axes.input_inner_axis}, {"y0", "y1", "x1"});
  ascgen_utils::indirect_load::TemplateLogicalView logical_view;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateLogicalView(indirect_load, logical_view), af::SUCCESS);
  ExpectAxisNames(simd_graph, logical_view.input.axis_ids, {"x0", "x1"});
  ExpectAxisNames(simd_graph, logical_view.index.axis_ids, {"y0", "y1"});
  ExpectAxisNames(simd_graph, logical_view.output.axis_ids, {"y0", "y1"});
}

TEST(IndirectLoadScheduleCaseGeneratorTest, PostReduceMetadataCoversReduceAxisLayouts) {
  const std::vector<std::pair<std::string, std::string>> layouts = {
      {"R", "RRR"}, {"AR", "ARR"}, {"RA", "RRA"}, {"ARA", "ARA"}};
  for (const auto &layout : layouts) {
    const auto &name = layout.first;
    const auto &suffix = layout.second;
    auto graph = BuildPostReduceGraph(suffix);
    const auto input_axes = graph.FindNode("x")->outputs()[0]->attr.axis;
    const auto output_axes = graph.FindNode("indirect_load")->outputs()[0]->attr.axis;
    const auto expected_input_strides = DenseStrides(graph, input_axes);
    const auto expected_output_strides = DenseStrides(graph, output_axes);
    optimize::IndirectLoadScheduleCaseGenerator generator;
    std::vector<af::AscGraph> graphs;
    std::vector<std::string> score_functions;
    ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS) << "layout=" << name;
    const size_t expected_candidates = name == "ARA" ? 2UL : 4UL;
    ASSERT_EQ(graphs.size(), expected_candidates) << "layout=" << name;

    const auto simd_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
    const auto simt_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt);
    ASSERT_NE(simt_iter, graphs.end()) << "layout=" << name;
    auto &simt_graph = *simt_iter;
    const auto simt_indirect_load = simt_graph.FindNode("indirect_load");
    ASSERT_NE(simt_indirect_load, nullptr);
    ascgen_utils::indirect_load::TemplateAxes simt_axes;
    ascgen_utils::indirect_load::TemplateLogicalView simt_view;
    ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(simt_indirect_load, simt_axes), af::SUCCESS);
    ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateLogicalView(simt_indirect_load, simt_view), af::SUCCESS);
    const size_t first_reduce = suffix.find('R') + 1UL;
    const std::vector<af::AxisId> simt_outer(output_axes.begin(),
                                             output_axes.begin() + static_cast<int64_t>(first_reduce));
    const std::vector<af::AxisId> simt_inner(output_axes.begin() + static_cast<int64_t>(first_reduce),
                                             output_axes.end());
    ExpectAxisOrigins(simt_graph, simt_axes.outer_axis, simt_outer);
    ExpectAxisOrigins(simt_graph, simt_axes.inner_axis, simt_inner);
    EXPECT_EQ(simt_axes.input_inner_axis, af::kIdNone);
    const bool is_ar = suffix.find('A', first_reduce) == std::string::npos;
    if (is_ar) {
      EXPECT_EQ(simt_axes.tile_outer_axis, af::kIdNone);
      EXPECT_EQ(simt_axes.tile_inner_axis, af::kIdNone);
    } else {
      EXPECT_NE(simt_axes.tile_outer_axis, af::kIdNone);
      EXPECT_NE(simt_axes.tile_inner_axis, af::kIdNone);
    }
    EXPECT_EQ(simt_view.input.axis_ids, input_axes);
    EXPECT_EQ(simt_view.index.axis_ids, output_axes);
    EXPECT_EQ(simt_view.output.axis_ids, output_axes);
    EXPECT_EQ(simt_view.input.strides, expected_input_strides);
    EXPECT_EQ(simt_view.index.strides, expected_output_strides);
    EXPECT_EQ(simt_view.output.strides, expected_output_strides);

    if (name == "ARA") {
      EXPECT_EQ(simd_iter, graphs.end());
      continue;
    }
    ASSERT_NE(simd_iter, graphs.end()) << "layout=" << name;
    auto &simd_graph = *simd_iter;
    const auto simd_indirect_load = simd_graph.FindNode("indirect_load");
    ASSERT_NE(simd_indirect_load, nullptr);
    ascgen_utils::indirect_load::TemplateAxes simd_axes;
    ascgen_utils::indirect_load::TemplateLogicalView simd_view;
    ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(simd_indirect_load, simd_axes), af::SUCCESS);
    ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateLogicalView(simd_indirect_load, simd_view), af::SUCCESS);
    const std::vector<af::AxisId> simd_outer(output_axes.begin(), output_axes.begin() + 1L);
    const std::vector<af::AxisId> simd_inner(output_axes.begin() + 1L, output_axes.end());
    const std::vector<af::AxisId> simd_input_inner(input_axes.begin() + 1L, input_axes.end());
    ExpectAxisOrigins(simd_graph, simd_axes.outer_axis, simd_outer);
    ExpectAxisOrigins(simd_graph, simd_axes.inner_axis, simd_inner);
    ExpectAxisOrigins(simd_graph, simd_axes.input_inner_axis, simd_input_inner);
    ExpectFixedTileSplit(simd_graph, simd_axes.outer_axis);
    EXPECT_EQ(simd_view.input.axis_ids, input_axes);
    EXPECT_EQ(simd_view.index.axis_ids, output_axes);
    EXPECT_EQ(simd_view.output.axis_ids, output_axes);
    EXPECT_EQ(simd_view.input.strides, expected_input_strides);
    EXPECT_EQ(simd_view.index.strides, expected_output_strides);
    EXPECT_EQ(simd_view.output.strides, expected_output_strides);
  }
}

TEST(IndirectLoadScheduleCaseGeneratorTest, PostReduceUsesSkWhenOuterAxisIsReducedBeforeGather) {
  auto graph = BuildPostReduceGraph("RRR", true);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);

  EXPECT_EQ(FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd), graphs.end());
  EXPECT_EQ(FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt), graphs.end());
  EXPECT_NE(FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSK), graphs.end());
}

TEST(IndirectLoadScheduleCaseGeneratorTest, PostReduceRejectsMultipleReduceSegmentsForBothTemplates) {
  auto graph = BuildPostReduceGraph("RAR");
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);

  EXPECT_EQ(graphs.size(), 1UL);
  EXPECT_EQ(FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd), graphs.end());
  EXPECT_EQ(FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt), graphs.end());
  EXPECT_NE(FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSK), graphs.end());
}

TEST(IndirectLoadScheduleCaseGeneratorTest, PostReduceAllowsDtypeCastBeforeStore) {
  auto graph = BuildPostReduceGraph("RAA", false, OutputPostTopology::kSum, true);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;

  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  EXPECT_NE(FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt), graphs.end());
}

TEST(IndirectLoadScheduleCaseGeneratorTest, PostReduceRejectsNonCastSuccessor) {
  auto graph = BuildPostReduceGraph("RAA", false, OutputPostTopology::kSum, false, true);

  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  // 非法 Reduce 后继属于候选级缺陷：四个候选（SIMD×2/SIMT/SK）全部被淘汰，
  // Generate 本身成功返回（bf5d926c 起不再以断言失败终止）。
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  EXPECT_TRUE(graphs.empty());
  EXPECT_TRUE(score_functions.empty());
}

TEST(IndirectLoadScheduleCaseGeneratorTest, PostReduceSkipsCommonZeroStrideAxes) {
  for (const std::string suffix : {"RBR", "BRA"}) {
    auto graph = BuildPostReduceGraph(suffix);
    const auto indirect_load = graph.FindNode("indirect_load");
    const auto reduce = graph.FindNode("sum");
    const auto common_zero_view = graph.FindNode("common_zero_view");
    ASSERT_NE(indirect_load, nullptr);
    ASSERT_NE(reduce, nullptr);
    ASSERT_NE(common_zero_view, nullptr);
    EXPECT_EQ(common_zero_view->inputs()[0]->attr.dtype, af::DT_FLOAT16);
    EXPECT_EQ(common_zero_view->outputs()[0]->attr.dtype, af::DT_FLOAT16);
    const auto output_axes = indirect_load->outputs()[0]->attr.axis;
    const size_t common_zero = suffix.find('B') + 1UL;
    EXPECT_EQ(indirect_load->outputs()[0]->attr.strides, DenseStrides(graph, output_axes));
    EXPECT_EQ(reduce->inputs()[0]->attr.strides[common_zero], af::sym::kSymbolZero);
    EXPECT_EQ(reduce->outputs()[0]->attr.strides[common_zero], af::sym::kSymbolZero);
    optimize::IndirectLoadScheduleCaseGenerator generator;
    std::vector<af::AscGraph> graphs;
    std::vector<std::string> score_functions;
    ge::PlatformContext::GetInstance().SetPlatform("3510");
    const auto status = generator.Generate(graph, graphs, score_functions);
    ge::PlatformContext::GetInstance().Reset();
    ASSERT_EQ(status, af::SUCCESS) << "suffix=" << suffix;
    EXPECT_NE(FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd), graphs.end())
        << "suffix=" << suffix;
    EXPECT_NE(FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt), graphs.end())
        << "suffix=" << suffix;
    ASSERT_EQ(graphs.size(), 4UL) << "suffix=" << suffix;

    const auto simt_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt);
    ASSERT_NE(simt_iter, graphs.end()) << "suffix=" << suffix;
    const auto simt_indirect_load = simt_iter->FindNode("indirect_load");
    ASSERT_NE(simt_indirect_load, nullptr);
    ascgen_utils::indirect_load::TemplateAxes axes;
    ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(simt_indirect_load, axes), af::SUCCESS);
    const size_t first_reduce = suffix.find('R') + 1UL;
    ExpectAxisOrigins(*simt_iter, axes.outer_axis,
                      std::vector<af::AxisId>(output_axes.begin(), output_axes.begin() + first_reduce));
  }
}

TEST(IndirectLoadScheduleCaseGeneratorTest, PostReduceHandlesUnknownOrIllegalInputStride) {
  struct StrideCase {
    const char *name;
    const char *suffix;
    size_t stride_index;
    bool use_unknown;
    bool mutate_output;
    // mutate_output 为 true 时有效：保留轴的符号 stride（动态 shape）允许生成
    // SIMD/SIMT；归约轴（输出 repeat=1）的输出 stride 无法确认时必须被拒绝。
    bool expect_templates_present;
  };
  const std::vector<StrideCase> cases = {{"outer_input_unknown", "RRR", 0UL, true, false, false},
                                         {"outer_output_unknown", "RRR", 0UL, true, true, true},
                                         {"outer_zero_output_nonzero", "RRR", 0UL, false, false, false},
                                         {"inner_input_unknown", "RRR", 1UL, true, false, false},
                                         {"inner_output_unknown", "RRR", 1UL, true, true, false},
                                         {"inner_zero_output_nonzero", "ARR", 1UL, false, false, false}};
  for (const auto &test_case : cases) {
    auto graph = BuildPostReduceGraph(test_case.suffix);
    const auto reduce = graph.FindNode("sum");
    ASSERT_NE(reduce, nullptr) << "case=" << test_case.name;
    ASSERT_FALSE(reduce->inputs().empty()) << "case=" << test_case.name;
    ASSERT_FALSE(reduce->outputs().empty()) << "case=" << test_case.name;
    const auto reduce_input = reduce->inputs()[0];
    const auto reduce_output = reduce->outputs()[0];
    ASSERT_NE(reduce_input, nullptr) << "case=" << test_case.name;
    ASSERT_NE(reduce_output, nullptr) << "case=" << test_case.name;
    const auto output_strides = reduce_output->attr.strides;
    af::Expression input_stride = af::ops::Zero;
    if (test_case.use_unknown) {
      input_stride = graph.CreateSizeVar("unknown_stride");
    }
    auto &strides = test_case.mutate_output ? reduce_output->attr.strides : reduce_input->attr.strides;
    strides[test_case.stride_index] = input_stride;

    EXPECT_EQ(strides[test_case.stride_index], input_stride) << "case=" << test_case.name;
    if (!test_case.mutate_output) {
      EXPECT_EQ(reduce_output->attr.strides, output_strides) << "case=" << test_case.name;
    }
    if (test_case.use_unknown) {
      EXPECT_EQ(af::SymbolicUtils::StaticCheckEq(input_stride, af::ops::Zero), af::TriBool::kUnknown)
          << "case=" << test_case.name;
    }

    optimize::IndirectLoadScheduleCaseGenerator generator;
    std::vector<af::AscGraph> graphs;
    std::vector<std::string> score_functions;
    if (!test_case.mutate_output) {
      // 改 reduce 输入 stride 会传导到 IL 输出视图使其非稠密，属于模板无关的不支持场景，直接报错。
      EXPECT_NE(generator.Generate(graph, graphs, score_functions), af::SUCCESS) << "case=" << test_case.name;
      EXPECT_TRUE(graphs.empty()) << "case=" << test_case.name;
      continue;
    }
    // 改 reduce 输出 stride 不影响 IL 输出视图：保留轴的符号 stride（动态 shape）允许
    // 生成 SIMD/SIMT；归约轴（输出 repeat=1）的输出 stride 无法确认为 0 时拒绝并回退 SK。
    ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS) << "case=" << test_case.name;
    if (test_case.expect_templates_present) {
      EXPECT_NE(FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd), graphs.end())
          << "case=" << test_case.name;
      EXPECT_NE(FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt), graphs.end())
          << "case=" << test_case.name;
    } else {
      EXPECT_EQ(FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd), graphs.end())
          << "case=" << test_case.name;
      EXPECT_EQ(FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt), graphs.end())
          << "case=" << test_case.name;
    }
  }
}

TEST(IndirectLoadScheduleCaseGeneratorTest, OutputPostTopologySelectsEligibleTemplatesAndPreservesChain) {
  const std::vector<PostTopologyCase> cases = {
      {OutputPostTopology::kSum, {}, true, af::DT_FLOAT16, false},
      {OutputPostTopology::kExp2Sum, {"output_exp2"}, false, af::DT_FLOAT16, false},
      {OutputPostTopology::kAbsExp2Sum, {"output_abs", "output_exp2"}, false, af::DT_FLOAT16, false},
      {OutputPostTopology::kCastSum, {"output_cast"}, false, af::DT_FLOAT, false},
      {OutputPostTopology::kAddSum, {"output_add"}, false, af::DT_FLOAT16, false},
      {OutputPostTopology::kBesselK0Sum, {"output_modified_bessel_k0"}, false, af::DT_FLOAT16, false},
      {OutputPostTopology::kBesselK0, {"output_modified_bessel_k0"}, false, af::DT_FLOAT16, true},
  };
  for (const auto &test_case : cases) {
    auto graph = BuildPostReduceGraph("RRR", false, test_case.topology);
    ExpectPostTopologyGraph(graph, test_case);

    optimize::IndirectLoadScheduleCaseGenerator generator;
    std::vector<af::AscGraph> graphs;
    std::vector<std::string> score_functions;
    ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
    const auto simd_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
    const auto simt_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt);
    ASSERT_NE(simd_iter, graphs.end());
    EXPECT_EQ(simt_iter != graphs.end(), test_case.expect_simt);
    ExpectPostTopologyGraph(*simd_iter, test_case);
  }
}

TEST(IndirectLoadScheduleCaseGeneratorTest, ShapeEnlargingIndirectLoadStopsSimdInputPreButNotSimt) {
  auto graph = BuildShapeEnlargingInputPreGraph();
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ge::PlatformContext::GetInstance().SetPlatform("3510");
  const auto status = generator.Generate(graph, graphs, score_functions);
  ge::PlatformContext::GetInstance().Reset();
  ASSERT_EQ(status, af::SUCCESS);
  ASSERT_EQ(graphs.size(), 4UL);
  ASSERT_EQ(score_functions.size(), graphs.size());
  // 静态稠密视图下 SIMD 候选走 dense facade，不需要让位给 SIMT，四个候选同分。
  for (const auto &score_function : score_functions) {
    EXPECT_EQ(ParseScore(score_function), ParseScore(score_functions[0]));
  }

  const auto simd_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(simd_iter, graphs.end());
  auto &simd_graph = *simd_iter;
  const auto simd_indirect_load = simd_graph.FindNode("indirect_load");
  const auto simd_load = simd_graph.FindNode("input_load");
  const auto simd_pre_cast = simd_graph.FindNode("pre_cast");
  ASSERT_NE(simd_indirect_load, nullptr);
  ASSERT_NE(simd_load, nullptr);
  ASSERT_NE(simd_pre_cast, nullptr);
  EXPECT_EQ(
      ascgen_utils::indirect_load::GetInputProducer(simd_indirect_load, ascgen_utils::indirect_load::kInputTensorIndex),
      simd_pre_cast);
  EXPECT_EQ(ascgen_utils::indirect_load::GetInputProducer(simd_pre_cast, 0UL), simd_load);

  const auto simt_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt);
  ASSERT_NE(simt_iter, graphs.end());
  auto &simt_graph = *simt_iter;
  const auto simt_indirect_load = simt_graph.FindNode("indirect_load");
  const auto simt_load = simt_graph.FindNode("input_load");
  const auto simt_pre_cast = simt_graph.FindNode("pre_cast");
  ASSERT_NE(simt_indirect_load, nullptr);
  ASSERT_NE(simt_load, nullptr);
  ASSERT_NE(simt_pre_cast, nullptr);
  EXPECT_EQ(
      ascgen_utils::indirect_load::GetInputProducer(simt_indirect_load, ascgen_utils::indirect_load::kInputTensorIndex),
      simt_load);
  EXPECT_EQ(ascgen_utils::indirect_load::GetInputProducer(simt_pre_cast, 0UL), simt_indirect_load);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, GenerateFailsWhenNonAxisInputDimensionIsSmallerThanIndex) {
  auto graph = BuildExpandedMultiInputGraph(false, 3L, 17L);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  EXPECT_NE(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  EXPECT_TRUE(graphs.empty());
}

TEST(IndirectLoadScheduleCaseGeneratorTest, GeneratedSkCandidateUsesSkBehavior) {
  auto graphs = GenerateIndirectLoadCases(2);
  const auto sk_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSK);
  ASSERT_NE(sk_iter, graphs.end());
  auto &sk_graph = *sk_iter;
  const auto indirect_load = sk_graph.FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);

  const auto behavior = ascgen_utils::indirect_load::GetTemplateBehavior(indirect_load);
  EXPECT_FALSE(behavior.skips_main_schedule_tiling);
  EXPECT_FALSE(behavior.skips_api_emit);
  EXPECT_FALSE(behavior.uses_direct_gm_pipeline);
  EXPECT_FALSE(behavior.skips_ub_lifecycle);
  EXPECT_FALSE(behavior.preserves_vectorized_axis);

  const auto input_boundary = ascgen_utils::indirect_load::GetInputProducer(indirect_load, 0UL);
  ASSERT_NE(input_boundary, nullptr);
  const auto input_behavior = ascgen_utils::indirect_load::GetTemplateBehavior(input_boundary);
  EXPECT_TRUE(input_behavior.skips_main_schedule_tiling);
  EXPECT_TRUE(input_behavior.skips_api_emit);
  EXPECT_FALSE(input_behavior.uses_direct_gm_pipeline);
  EXPECT_FALSE(input_behavior.skips_ub_lifecycle);
  EXPECT_TRUE(input_behavior.preserves_vectorized_axis);
  EXPECT_TRUE(ascgen_utils::indirect_load::ShouldApplyInputInnerVectorization(input_boundary));
}

TEST(IndirectLoadScheduleCaseGeneratorTest, SimdMovesInputPreAfterIndirectLoad) {
  auto graph = BuildIndirectLoadGraph(2, true);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  const auto simd_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(simd_iter, graphs.end());
  auto &simd_graph = *simd_iter;
  const auto indirect_load = simd_graph.FindNode("indirect_load");
  const auto pre_abs = simd_graph.FindNode("pre_abs");
  ASSERT_NE(indirect_load, nullptr);
  ASSERT_NE(pre_abs, nullptr);

  EXPECT_EQ(ascgen_utils::indirect_load::GetInputProducer(indirect_load, 0UL)->GetName(), "input_load");
  EXPECT_EQ(ascgen_utils::indirect_load::GetInputProducer(pre_abs, 0UL), indirect_load);
  EXPECT_EQ(ascgen_utils::indirect_load::GetOnlyOutputConsumer(indirect_load), pre_abs);
  ExpectAxisNames(simd_graph, pre_abs->attr.sched.axis, {"z4", "z5", "z6", "z7"});
  ExpectAxisNames(simd_graph, pre_abs->outputs()[0]->attr.axis, {"z4", "z5", "z6", "z7"});
}

struct FullPrefixNodeCase {
  const char *name;
  af::DataType dtype;
};

void ExpectFullPrefixChain(af::AscGraph &graph, const std::vector<std::pair<const char *, const char *>> &edges,
                           const std::vector<FullPrefixNodeCase> &nodes, const std::vector<af::AxisId> &axes,
                           const std::vector<af::Expression> &repeats, const std::vector<af::Expression> &strides) {
  for (const auto &edge : edges) {
    const auto producer = graph.FindNode(edge.first);
    const auto consumer = graph.FindNode(edge.second);
    ASSERT_NE(producer, nullptr) << edge.first;
    ASSERT_NE(consumer, nullptr) << edge.second;
    const size_t input_index =
        std::strcmp(edge.second, "indirect_load") == 0 && std::strcmp(edge.first, "index_floor_to_int") == 0 ? 1UL
                                                                                                             : 0UL;
    const auto actual_producer = ascgen_utils::indirect_load::GetInputProducer(consumer, input_index);
    ASSERT_NE(actual_producer, nullptr) << edge.second;
    EXPECT_EQ(actual_producer->GetName(), producer->GetName()) << edge.second;
  }
  for (const auto &test_case : nodes) {
    const auto node = graph.FindNode(test_case.name);
    ASSERT_NE(node, nullptr) << test_case.name;
    ASSERT_FALSE(node->outputs().empty());
    const auto &output = node->outputs()[0]->attr;
    EXPECT_EQ(output.dtype, test_case.dtype) << test_case.name;
    EXPECT_EQ(output.axis, axes) << test_case.name;
    EXPECT_EQ(output.repeats, repeats) << test_case.name;
    EXPECT_EQ(output.strides, strides) << test_case.name;
  }
}

void ExpectFullPrefixOriginalGraph(af::AscGraph &graph,
                                   const std::vector<std::pair<const char *, const char *>> &input_edges,
                                   const std::vector<std::pair<const char *, const char *>> &index_edges) {
  const auto x_output = graph.FindNode("x")->outputs()[0]->attr;
  const auto index_output = graph.FindNode("index")->outputs()[0]->attr;
  ExpectFullPrefixChain(graph, input_edges, {{"input_relu", af::DT_FLOAT16}, {"input_exp2", af::DT_FLOAT16}},
                        x_output.axis, x_output.repeats, x_output.strides);
  ExpectFullPrefixChain(graph, index_edges,
                        {{"index_abs", af::DT_INT32},
                         {"index_abs2", af::DT_INT32},
                         {"index_cast_float", af::DT_FLOAT},
                         {"index_exp2", af::DT_FLOAT},
                         {"index_log2", af::DT_FLOAT},
                         {"index_floor_to_int", af::DT_INT32}},
                        index_output.axis, index_output.repeats, index_output.strides);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, SimdFullPrefixPreservesInputAndIndexPreMetadata) {
  auto graph = BuildIndirectLoadGraph(2, true, true);
  const std::vector<std::pair<const char *, const char *>> input_edges = {
      {"input_load", "input_relu"}, {"input_relu", "input_exp2"}, {"input_exp2", "indirect_load"}};
  const std::vector<std::pair<const char *, const char *>> index_edges = {{"index_load", "index_abs"},
                                                                          {"index_abs", "index_abs2"},
                                                                          {"index_abs2", "index_cast_float"},
                                                                          {"index_cast_float", "index_exp2"},
                                                                          {"index_exp2", "index_log2"},
                                                                          {"index_log2", "index_floor_to_int"},
                                                                          {"index_floor_to_int", "indirect_load"}};
  ExpectFullPrefixOriginalGraph(graph, input_edges, index_edges);

  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  const auto simd_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimd);
  ASSERT_NE(simd_iter, graphs.end());
  const auto indirect_load = simd_iter->FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  const auto input_producer = ascgen_utils::indirect_load::GetInputProducer(indirect_load, 0UL);
  const auto index_producer = ascgen_utils::indirect_load::GetInputProducer(indirect_load, 1UL);
  ASSERT_NE(input_producer, nullptr);
  ASSERT_NE(index_producer, nullptr);
  EXPECT_EQ(input_producer->GetName(), "input_load");
  EXPECT_EQ(index_producer->GetName(), "index_floor_to_int");
  const std::vector<af::AxisId> moved_axes = indirect_load->outputs()[0]->attr.axis;
  const std::vector<af::Expression> moved_repeats = indirect_load->outputs()[0]->attr.repeats;
  const std::vector<af::Expression> moved_strides = indirect_load->outputs()[0]->attr.strides;
  ExpectFullPrefixChain(
      *simd_iter, {{"input_load", "indirect_load"}, {"indirect_load", "input_relu"}, {"input_relu", "input_exp2"}},
      {{"input_relu", af::DT_FLOAT16}, {"input_exp2", af::DT_FLOAT16}}, moved_axes, moved_repeats, moved_strides);
  const auto index_output = graph.FindNode("index")->outputs()[0]->attr;
  ExpectFullPrefixChain(*simd_iter, index_edges,
                        {{"index_abs", af::DT_INT32},
                         {"index_abs2", af::DT_INT32},
                         {"index_cast_float", af::DT_FLOAT},
                         {"index_exp2", af::DT_FLOAT},
                         {"index_log2", af::DT_FLOAT},
                         {"index_floor_to_int", af::DT_INT32}},
                        index_output.axis, index_output.repeats, index_output.strides);
  ascgen_utils::indirect_load::TemplateAxes axes;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(indirect_load, axes), af::SUCCESS);
  EXPECT_NE(axes.input_inner_axis, af::kIdNone);
  ascgen_utils::indirect_load::TemplateLogicalView logical_view;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateLogicalView(indirect_load, logical_view), af::SUCCESS);
  EXPECT_EQ(logical_view.input.axis_ids, graph.FindNode("x")->outputs()[0]->attr.axis);
  EXPECT_EQ(logical_view.index.axis_ids, graph.FindNode("index")->outputs()[0]->attr.axis);
  EXPECT_EQ(logical_view.output.axis_ids, graph.FindNode("indirect_load")->outputs()[0]->attr.axis);
  EXPECT_EQ(logical_view.input.strides, graph.FindNode("x")->outputs()[0]->attr.strides);
  EXPECT_EQ(logical_view.index.strides, graph.FindNode("index")->outputs()[0]->attr.strides);
  EXPECT_EQ(logical_view.output.strides, graph.FindNode("indirect_load")->outputs()[0]->attr.strides);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, SimtHandlesInputPrecisionCast) {
  auto graph = BuildIndirectLoadPrecisionCastGraph();
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ge::PlatformContext::GetInstance().SetPlatform("3510");
  const auto status = generator.Generate(graph, graphs, score_functions);
  ge::PlatformContext::GetInstance().Reset();
  ASSERT_EQ(status, af::SUCCESS);
  ASSERT_EQ(graphs.size(), 4UL);

  const auto simt_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt);
  ASSERT_NE(simt_iter, graphs.end());
  auto &simt_graph = *simt_iter;
  const auto input_cast = simt_graph.FindNode("input_cast");
  EXPECT_NE(input_cast, nullptr);
  EXPECT_NE(simt_graph.FindNode("output_cast"), nullptr);
  const auto simt_indirect_load = simt_graph.FindNode("indirect_load");
  const auto output_exp = simt_graph.FindNode("output_exp");
  ASSERT_NE(simt_indirect_load, nullptr);
  ASSERT_NE(output_exp, nullptr);
  EXPECT_EQ(simt_indirect_load->outputs()[0]->attr.dtype, af::DT_FLOAT16);
  EXPECT_EQ(output_exp->outputs()[0]->attr.dtype, af::DT_FLOAT);
  const auto input_producer =
      ascgen_utils::indirect_load::GetInputProducer(simt_indirect_load, ascgen_utils::indirect_load::kInputTensorIndex);
  const auto cast_producer = ascgen_utils::indirect_load::GetInputProducer(input_cast, 0UL);
  const auto cast_consumer = ascgen_utils::indirect_load::GetOnlyOutputConsumer(input_cast);
  ASSERT_NE(input_producer, nullptr);
  EXPECT_EQ(input_producer->GetName(), "input_load");
  EXPECT_EQ(cast_producer, simt_indirect_load);
  EXPECT_EQ(cast_consumer, output_exp);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, GenerateFailsWhenAxisOutOfRange) {
  for (int64_t axis : {-5L, 8L}) {
    auto graph = BuildIndirectLoadGraph(axis);
    optimize::IndirectLoadScheduleCaseGenerator generator;
    std::vector<af::AscGraph> graphs;
    std::vector<std::string> score_functions;
    EXPECT_NE(generator.Generate(graph, graphs, score_functions), af::SUCCESS) << "axis=" << axis;
    EXPECT_TRUE(graphs.empty());
  }
}

TEST(IndirectLoadScheduleCaseGeneratorTest, GenerateFailsWhenUnsupportedTopologyNodePresent) {
  auto graph = BuildIndirectLoadGraph(2L);

  af::ascir_op::Transpose transpose("unsupported_transpose");
  graph.AddNode(transpose);
  transpose.attr.api.compute_type = af::ComputeType::kComputeTranspose;
  transpose.attr.api.type = af::ApiType::kAPITypeCompute;
  transpose.y.dtype = ge::DT_FLOAT16;
  ASSERT_NE(graph.FindNode("unsupported_transpose"), nullptr);

  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  EXPECT_NE(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  EXPECT_TRUE(graphs.empty());
  EXPECT_TRUE(score_functions.empty());
}

// gather 输出经 +bias / +bmm 双输入 Add 链后进入尾轴归约 softmax pattern 的融合图。
af::AscGraph BuildIndirectLoadSoftmaxPostGraph() {
  af::AscGraph graph("indirect_load_softmax_post_ut_graph");
  const af::Expression s0 = graph.CreateSizeVar(2);
  const af::Expression s1 = graph.CreateSizeVar(3);
  const af::Expression s2 = graph.CreateSizeVar(5);
  const af::Expression in0 = graph.CreateSizeVar(2);
  const af::Expression in1 = graph.CreateSizeVar(4);
  const af::Expression in2 = graph.CreateSizeVar(5);
  const auto y0 = graph.CreateAxis("y0", s0);
  const auto y1 = graph.CreateAxis("y1", s1);
  const auto y2 = graph.CreateAxis("y2", s2);
  const auto x0 = graph.CreateAxis("x0", in0);
  const auto x1 = graph.CreateAxis("x1", in1);
  const auto x2 = graph.CreateAxis("x2", in2);
  const std::vector<af::AxisId> output_axes = {y0.id, y1.id, y2.id};
  const std::vector<af::Expression> output_repeats = {s0, s1, s2};
  const std::vector<af::Expression> output_strides = {s1 * s2, s2, af::sym::kSymbolOne};
  const std::vector<af::AxisId> input_axes = {x0.id, x1.id, x2.id};
  const std::vector<af::Expression> input_repeats = {in0, in1, in2};
  const std::vector<af::Expression> input_strides = {in1 * in2, in2, af::sym::kSymbolOne};
  const std::vector<af::Expression> reduce_repeats = {s0, s1, af::sym::kSymbolOne};
  const std::vector<af::Expression> reduce_strides = {s1 * s2, s2, af::sym::kSymbolZero};

  af::ascir_op::Data input_data("input_data", graph);
  input_data.ir_attr.SetIndex(0);
  SetNodeView(input_data, af::DT_FLOAT16, input_axes, input_repeats, input_strides);
  af::ascir_op::Load input_load("input_load");
  input_load.x = input_data.y;
  SetNodeView(input_load, af::DT_FLOAT16, input_axes, input_repeats, input_strides);
  af::ascir_op::Data index_data("index_data", graph);
  index_data.ir_attr.SetIndex(1);
  SetNodeView(index_data, af::DT_INT32, output_axes, output_repeats, output_strides);
  af::ascir_op::Load index_load("index_load");
  index_load.x = index_data.y;
  SetNodeView(index_load, af::DT_INT32, output_axes, output_repeats, output_strides);
  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  indirect_load.x1 = input_load.y;
  indirect_load.x2 = index_load.y;
  indirect_load.ir_attr.SetAxis(2);  // 尾轴gather：Softmax专用路径要求（非尾轴gather的SIMD输出物理布局有行对齐空洞）
  SetNodeView(indirect_load, af::DT_FLOAT16, output_axes, output_repeats, output_strides);

  af::ascir_op::Data bias_data("bias_data", graph);
  bias_data.ir_attr.SetIndex(2);
  SetNodeView(bias_data, af::DT_FLOAT16, output_axes, output_repeats, output_strides);
  af::ascir_op::Load bias_load("bias_load");
  bias_load.x = bias_data.y;
  SetNodeView(bias_load, af::DT_FLOAT16, output_axes, output_repeats, output_strides);
  af::ascir_op::Data bmm_data("bmm_data", graph);
  bmm_data.ir_attr.SetIndex(3);
  SetNodeView(bmm_data, af::DT_FLOAT16, output_axes, output_repeats, output_strides);
  af::ascir_op::Load bmm_load("bmm_load");
  bmm_load.x = bmm_data.y;
  SetNodeView(bmm_load, af::DT_FLOAT16, output_axes, output_repeats, output_strides);
  af::ascir_op::Add bias_add("bias_add");
  bias_add.x1 = indirect_load.y;
  bias_add.x2 = bias_load.y;
  SetVectorApi(bias_add);
  SetNodeView(bias_add, af::DT_FLOAT16, output_axes, output_repeats, output_strides);
  af::ascir_op::Add bmm_add("bmm_add");
  bmm_add.x1 = bias_add.y;
  bmm_add.x2 = bmm_load.y;
  SetVectorApi(bmm_add);
  SetNodeView(bmm_add, af::DT_FLOAT16, output_axes, output_repeats, output_strides);

  af::ascir_op::Max max_op("max");
  max_op.x = bmm_add.y;
  max_op.attr.api.compute_type = af::ComputeType::kComputeReduce;
  SetNodeView(max_op, af::DT_FLOAT16, output_axes, reduce_repeats, reduce_strides);
  af::ascir_op::Broadcast max_broadcast("max_broadcast");
  max_broadcast.x = max_op.y;
  SetNodeView(max_broadcast, af::DT_FLOAT16, output_axes, output_repeats, output_strides);
  af::ascir_op::Sub sub_op("sub");
  sub_op.x1 = bmm_add.y;
  sub_op.x2 = max_broadcast.y;
  SetVectorApi(sub_op);
  SetNodeView(sub_op, af::DT_FLOAT16, output_axes, output_repeats, output_strides);
  af::ascir_op::Exp exp_op("exp");
  exp_op.x = sub_op.y;
  SetVectorApi(exp_op);
  SetNodeView(exp_op, af::DT_FLOAT16, output_axes, output_repeats, output_strides);
  af::ascir_op::Sum sum_op("sum");
  sum_op.x = exp_op.y;
  sum_op.attr.api.compute_type = af::ComputeType::kComputeReduce;
  SetNodeView(sum_op, af::DT_FLOAT16, output_axes, reduce_repeats, reduce_strides);
  af::ascir_op::Broadcast sum_broadcast("sum_broadcast");
  sum_broadcast.x = sum_op.y;
  SetNodeView(sum_broadcast, af::DT_FLOAT16, output_axes, output_repeats, output_strides);
  af::ascir_op::TrueDiv true_div("true_div");
  true_div.x1 = exp_op.y;
  true_div.x2 = sum_broadcast.y;
  SetVectorApi(true_div);
  SetNodeView(true_div, af::DT_FLOAT16, output_axes, output_repeats, output_strides);
  af::ascir_op::Store store("store");
  store.x = true_div.y;
  SetNodeView(store, af::DT_FLOAT16, output_axes, output_repeats, output_strides);
  af::ascir_op::Output output("output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  SetNodeView(output, af::DT_FLOAT16, output_axes, output_repeats, output_strides);
  return graph;
}

// gather+softmax 专用路径：候选图内替换为 Softmax 节点，SIMT 候选的 boundary 覆写为
// 尾轴前一位（outer=[保留轴]，inner=[尾轴]），tile 行数不再预建固定轴（可求解）。
TEST(IndirectLoadScheduleCaseGeneratorTest, SoftmaxDedicatedCandidateSolvesTileSize) {
  auto graph = BuildIndirectLoadSoftmaxPostGraph();
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  const auto simt_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt);
  ASSERT_NE(simt_iter, graphs.end());
  auto &simt_graph = *simt_iter;

  // softmax 替换命中：多输入 Add 链保留在 Softmax 之前。
  EXPECT_NE(simt_graph.FindNode("true_div_softmax"), nullptr);
  EXPECT_EQ(simt_graph.FindNode("true_div"), nullptr);
  EXPECT_NE(simt_graph.FindNode("bmm_add"), nullptr);
  EXPECT_NE(simt_graph.FindNode("bias_add"), nullptr);

  const auto simt_indirect_load = simt_graph.FindNode("indirect_load");
  ASSERT_NE(simt_indirect_load, nullptr);
  ascgen_utils::norm::NormInfo norm_info;
  ASSERT_EQ(ascgen_utils::norm::TryGetNormInfo(simt_indirect_load, norm_info), af::SUCCESS);
  EXPECT_EQ(norm_info.kind, ascgen_utils::norm::NormInfo::Kind::kSoftmaxDedicated);
  const auto tail_axis = simt_graph.FindNode("true_div_softmax")->inputs()[0]->attr.axis.back();
  EXPECT_EQ(norm_info.softmax_reduce_axis, tail_axis);

  ascgen_utils::indirect_load::TemplateAxes axes;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(simt_indirect_load, axes), af::SUCCESS);
  // Softmax 专用节点与 AR 后置 Reduce 一致：tile 轴不再预建固定 1 行，
  // tile 行数交由调度期按 UB 容量求解（SoftmaxAR 的 A=tile 行数）。
  EXPECT_EQ(axes.tile_outer_axis, af::kIdNone);
  EXPECT_EQ(axes.tile_inner_axis, af::kIdNone);
  // outer=[y0,y1]（保留轴），inner=[y2]（Softmax 的 R 轴），向量化只沿尾轴。
  ExpectAxisNames(simt_graph, {axes.inner_axis}, {"y2"});
  EXPECT_EQ(axes.vectorized_axes, std::vector<af::AxisId>{tail_axis});
  const auto output_axes = simt_indirect_load->outputs()[0]->attr.axis;
  const std::vector<af::AxisId> expected_outer(output_axes.begin(), output_axes.begin() + 2);
  ExpectAxisOrigins(simt_graph, axes.outer_axis, expected_outer);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, SimtArPostReduceCandidateUsesSolvableTile) {
  auto graph = BuildPostReduceGraph("ARR");
  const auto output_axes = graph.FindNode("indirect_load")->outputs()[0]->attr.axis;
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  const auto simt_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt);
  ASSERT_NE(simt_iter, graphs.end());
  const auto simt_indirect_load = simt_iter->FindNode("indirect_load");
  ASSERT_NE(simt_indirect_load, nullptr);

  ascgen_utils::indirect_load::TemplateAxes axes;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(simt_indirect_load, axes), af::SUCCESS);
  // AR Reduce 的 retained prefix 由调度期按 UB 容量切分，Reduce suffix 保持完整。
  EXPECT_EQ(axes.tile_outer_axis, af::kIdNone);
  EXPECT_EQ(axes.tile_inner_axis, af::kIdNone);
  // ARR 布局：first_reduce 在第 2 轴，outer=[y0,y1]，inner=[y2,y3]。
  const size_t first_reduce = 2UL;
  const std::vector<af::AxisId> expected_outer(output_axes.begin(), output_axes.begin() + first_reduce);
  const std::vector<af::AxisId> expected_inner(output_axes.begin() + first_reduce, output_axes.end());
  ExpectAxisOrigins(*simt_iter, axes.outer_axis, expected_outer);
  ExpectAxisOrigins(*simt_iter, axes.inner_axis, expected_inner);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, SimtRaPostReduceCandidateKeepsFixedTile) {
  auto graph = BuildPostReduceGraph("RAA");
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  const auto simt_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSimt);
  ASSERT_NE(simt_iter, graphs.end());
  const auto simt_indirect_load = simt_iter->FindNode("indirect_load");
  ASSERT_NE(simt_indirect_load, nullptr);

  ascgen_utils::indirect_load::TemplateAxes axes;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(simt_indirect_load, axes), af::SUCCESS);
  EXPECT_NE(axes.tile_outer_axis, af::kIdNone);
  EXPECT_NE(axes.tile_inner_axis, af::kIdNone);
}

// ==================== 双 gather 水平融合 ====================
// rank2 图：input0(s0,s1)/input1(s0',s1')、index/out(s2,s3)、gather_dim=1；
// 两路 IndirectLoad 输出经 Add 归并，可选后置 Sum（reduce_dim=1，满足 G>=R）。
// 构图选项注释置于字段上方而非行尾：行尾注释对齐列随 clang-format 版本变化（v18 与 v19+
// 列不同），上置写法使格式化结果与版本无关。
struct DualGatherGraphOptions {
  // true: 两路 input 共用轴；false: 第二路独立轴
  bool same_input_axes = false;
  // true: 第二路 index 使用不同尺寸符号（非法组配对）
  bool index_shape_mismatch = false;
  // true: 第二路 IL 的 gather 轴不同（非法组配对）
  bool axis_mismatch = false;
  // true: 两路各自 store，不归并（非法组配对）
  bool independent_outputs = false;
  // true: 两路 input 各挂一个前置 Abs（验证逐路改图）
  bool with_input_pre = false;
  // true: 归并后接 Sum
  bool with_post_reduce = false;
  // true: 额外第三个 IL 独立输出（成员输出链发散，非法组）
  bool third_indirect_load = false;
  // true: 额外第三个 IL 经归并树汇入共享输出链（三路融合）
  bool third_indirect_load_merged = false;
  // true: 两路 input 值 dtype 为 int8（1 字节）——SIMD 档被 dtype 门控拒绝，仅 SIMT 档
  bool one_byte_value = false;
  // true: 第一路 input 值 dtype 为 int8（成员级 1 字节，混模板场景）
  bool first_value_one_byte = false;
  // true: 第二路 input 值 dtype 为 int8（成员级 1 字节，混模板场景；与第一路 dtype 不同
  // 时自动在第二路 IL 后插 Cast 对齐归并链 dtype）
  bool second_value_one_byte = false;
  // true: 第一路 index 链挂非法外部消费者（Abs→store，SIMT 候选否决、SIMD/SK 不受影响）
  bool first_index_illegal_consumer = false;
  // true: 第二路 index 链挂非法外部消费者（同上）
  bool second_index_illegal_consumer = false;
  // true: post Reduce 归约外轴（R>G 形态：SIMD/SIMT 不可行，SK 分段桥接承接）；仅与
  // with_post_reduce 组合生效
  bool reduce_outer = false;
};

af::AscGraph BuildDualIndirectLoadGraph(const DualGatherGraphOptions &options = {}) {
  af::AscGraph graph("indirect_load_dual_ut_graph");
  // BuildPostReduceLayout 对 Reduce 输入/输出 stride 逐轴做 StaticCheckEq(stride, 0) 静态判定，
  // 符号 stride 判定为 kUnknown 时布局直接非法；with_post_reduce 场景与单 gather 的
  // BuildPostReduceGraph 约定一致改用具体尺寸，其余场景保持符号尺寸。
  const af::DataType value0_dtype =
      (options.one_byte_value || options.first_value_one_byte) ? af::DT_INT8 : af::DT_FLOAT16;
  const af::DataType value1_dtype =
      (options.one_byte_value || options.second_value_one_byte) ? af::DT_INT8 : af::DT_FLOAT16;
  const bool mixed_value_dtype = value0_dtype != value1_dtype;
  const auto make_size = [&graph, &options](const char *name, int64_t value) {
    return options.with_post_reduce ? af::Expression(graph.CreateSizeVar(value)) : graph.CreateSizeVar(name);
  };
  const af::Expression s0 = make_size("s0", 32);
  const af::Expression s1 = make_size("s1", 16);
  const af::Expression s0b = make_size("s0b", 32);
  const af::Expression s1b = make_size("s1b", 16);
  const af::Expression s2 = make_size("s2", 8);
  const af::Expression s3 = make_size("s3", 16);
  const af::Expression s3b = make_size("s3b", 32);
  const auto z0 = graph.CreateAxis("z0", s0);
  const auto z1 = graph.CreateAxis("z1", s1);
  const auto w0 = graph.CreateAxis("w0", s0b);
  const auto w1 = graph.CreateAxis("w1", s1b);
  const auto z2 = graph.CreateAxis("z2", s2);
  const auto z3 = graph.CreateAxis("z3", s3);
  const auto z3b = graph.CreateAxis("z3b", s3b);
  const std::vector<af::AxisId> input0_axes = {z0.id, z1.id};
  const std::vector<af::AxisId> input1_axes =
      options.same_input_axes ? input0_axes : std::vector<af::AxisId>{w0.id, w1.id};
  const std::vector<af::AxisId> output_axes = {z2.id, z3.id};
  const std::vector<af::Expression> input0_repeats = {s0, s1};
  const std::vector<af::Expression> input1_repeats =
      options.same_input_axes ? input0_repeats : std::vector<af::Expression>{s0b, s1b};
  const std::vector<af::Expression> output_repeats = {s2, s3};
  const std::vector<af::Expression> input0_strides = {s1, af::sym::kSymbolOne};
  const std::vector<af::Expression> input1_strides =
      options.same_input_axes ? input0_strides : std::vector<af::Expression>{s1b, af::sym::kSymbolOne};
  const std::vector<af::Expression> output_strides = {s3, af::sym::kSymbolOne};
  const std::vector<af::Expression> index1_repeats =
      options.index_shape_mismatch ? std::vector<af::Expression>{s2, s3b} : output_repeats;
  const std::vector<af::Expression> index1_strides =
      options.index_shape_mismatch ? std::vector<af::Expression>{s3b, af::sym::kSymbolOne} : output_strides;

  af::ascir_op::Data x0("dual_x0", graph);
  x0.ir_attr.SetIndex(0);
  SetNodeView(x0, value0_dtype, input0_axes, input0_repeats, input0_strides);
  af::ascir_op::Load input_load0("dual_input_load0");
  input_load0.x = x0.y;
  SetNodeView(input_load0, value0_dtype, input0_axes, input0_repeats, input0_strides);
  af::ascir_op::Data index0("dual_index0", graph);
  index0.ir_attr.SetIndex(1);
  SetNodeView(index0, af::DT_INT32, output_axes, output_repeats, output_strides);
  af::ascir_op::Load index_load0("dual_index_load0");
  index_load0.x = index0.y;
  SetNodeView(index_load0, af::DT_INT32, output_axes, output_repeats, output_strides);
  if (options.first_index_illegal_consumer) {
    // 与单 gather kIllegalConsumer 形态一致：index 链尾部喂给非 store 的外部消费者，
    // SIMT 标量求值器无法供值（SIMT 候选否决）；SIMD/SK 按张量整体读 index，不受影响。
    af::ascir_op::Abs index_abs0("dual_index_abs0");
    index_abs0.x = index_load0.y;
    SetNodeView(index_abs0, af::DT_INT32, output_axes, output_repeats, output_strides);
    SetVectorApi(index_abs0);
    af::ascir_op::Store index_store0("dual_index_store0");
    index_store0.x = index_abs0.y;
    SetNodeView(index_store0, af::DT_INT32, output_axes, output_repeats, output_strides);
  }

  af::ascir_op::Data x1("dual_x1", graph);
  x1.ir_attr.SetIndex(2);
  SetNodeView(x1, value1_dtype, input1_axes, input1_repeats, input1_strides);
  af::ascir_op::Load input_load1("dual_input_load1");
  input_load1.x = x1.y;
  SetNodeView(input_load1, value1_dtype, input1_axes, input1_repeats, input1_strides);
  af::ascir_op::Data index1("dual_index1", graph);
  index1.ir_attr.SetIndex(3);
  SetNodeView(index1, af::DT_INT32, output_axes, index1_repeats, index1_strides);
  af::ascir_op::Load index_load1("dual_index_load1");
  index_load1.x = index1.y;
  SetNodeView(index_load1, af::DT_INT32, output_axes, index1_repeats, index1_strides);
  if (options.second_index_illegal_consumer) {
    af::ascir_op::Abs index_abs1("dual_index_abs1");
    index_abs1.x = index_load1.y;
    SetNodeView(index_abs1, af::DT_INT32, output_axes, index1_repeats, index1_strides);
    SetVectorApi(index_abs1);
    af::ascir_op::Store index_store1("dual_index_store1");
    index_store1.x = index_abs1.y;
    SetNodeView(index_store1, af::DT_INT32, output_axes, index1_repeats, index1_strides);
  }

  af::ascir_op::IndirectLoad indirect_load0("dual_indirect_load0");
  if (options.with_input_pre) {
    af::ascir_op::Abs pre_abs0("dual_pre_abs0");
    pre_abs0.x = input_load0.y;
    SetNodeView(pre_abs0, value0_dtype, input0_axes, input0_repeats, input0_strides);
    SetVectorApi(pre_abs0);
    indirect_load0.x1 = pre_abs0.y;
  } else {
    indirect_load0.x1 = input_load0.y;
  }
  indirect_load0.x2 = index_load0.y;
  indirect_load0.ir_attr.SetAxis(1);
  SetNodeView(indirect_load0, value0_dtype, output_axes, output_repeats, output_strides);

  af::ascir_op::IndirectLoad indirect_load1("dual_indirect_load1");
  if (options.with_input_pre) {
    af::ascir_op::Abs pre_abs1("dual_pre_abs1");
    pre_abs1.x = input_load1.y;
    SetNodeView(pre_abs1, value1_dtype, input1_axes, input1_repeats, input1_strides);
    SetVectorApi(pre_abs1);
    indirect_load1.x1 = pre_abs1.y;
  } else {
    indirect_load1.x1 = input_load1.y;
  }
  indirect_load1.x2 = index_load1.y;
  indirect_load1.ir_attr.SetAxis(options.axis_mismatch ? 0 : 1);
  SetNodeView(indirect_load1, value1_dtype, output_axes, output_repeats, output_strides);

  // 混模板场景（成员 dtype 不同）：第二路 IL 输出经 Cast 对齐归并链 dtype（模拟
  // DtypeConsistency 在归并树前插的 cast；IL 自身 input/output 保持 1 字节 dtype，
  // dtype 门控按 IL 输入判定）。
  af::ascir_op::Cast value_cast1("dual_cast1");
  if (mixed_value_dtype) {
    value_cast1.x = indirect_load1.y;
    SetNodeView(value_cast1, value0_dtype, output_axes, output_repeats, output_strides);
  }

  if (options.third_indirect_load) {
    af::ascir_op::IndirectLoad indirect_load2("dual_indirect_load2");
    indirect_load2.x1 = input_load0.y;
    indirect_load2.x2 = index_load0.y;
    indirect_load2.ir_attr.SetAxis(1);
    SetNodeView(indirect_load2, value0_dtype, output_axes, output_repeats, output_strides);
    af::ascir_op::Store store2("dual_store2");
    store2.x = indirect_load2.y;
    SetNodeView(store2, value0_dtype, output_axes, output_repeats, output_strides);
    af::ascir_op::Output output2("dual_output2");
    output2.x = store2.y;
    output2.ir_attr.SetIndex(4);
    SetNodeView(output2, value0_dtype, output_axes, output_repeats, output_strides);
  }

  if (options.independent_outputs) {
    af::ascir_op::Store store0("dual_store0");
    store0.x = indirect_load0.y;
    SetNodeView(store0, value0_dtype, output_axes, output_repeats, output_strides);
    af::ascir_op::Output output0("dual_output0");
    output0.x = store0.y;
    output0.ir_attr.SetIndex(0);
    SetNodeView(output0, value0_dtype, output_axes, output_repeats, output_strides);
    af::ascir_op::Store store1("dual_store1");
    store1.x = indirect_load1.y;
    SetNodeView(store1, value1_dtype, output_axes, output_repeats, output_strides);
    af::ascir_op::Output output1("dual_output1");
    output1.x = store1.y;
    output1.ir_attr.SetIndex(1);
    SetNodeView(output1, value1_dtype, output_axes, output_repeats, output_strides);
    return graph;
  }

  af::ascir_op::Add merge_add("dual_merge_add");
  merge_add.x1 = indirect_load0.y;
  merge_add.x2 = mixed_value_dtype ? value_cast1.y : indirect_load1.y;
  SetNodeView(merge_add, value0_dtype, output_axes, output_repeats, output_strides);
  SetVectorApi(merge_add);
  // reduce_outer：Sum 归约外轴（R>G，SIMD/SIMT 模板不可行）；默认归约内轴（G≥R）。
  const std::vector<af::Expression> terminal_repeats =
      options.with_post_reduce ? (options.reduce_outer ? std::vector<af::Expression>{af::sym::kSymbolOne, s3}
                                                       : std::vector<af::Expression>{s2, af::sym::kSymbolOne})
                               : output_repeats;
  const std::vector<af::Expression> terminal_strides =
      options.with_post_reduce
          ? (options.reduce_outer ? std::vector<af::Expression>{af::sym::kSymbolZero, af::sym::kSymbolOne}
                                  : std::vector<af::Expression>{af::sym::kSymbolOne, af::sym::kSymbolZero})
          : output_strides;
  af::ascir_op::Store store("dual_store");
  const auto wire_terminal = [&options, &output_axes, &terminal_repeats, &terminal_strides, &store,
                              &value0_dtype](const af::AscOpOutput &chain_head) {
    if (options.with_post_reduce) {
      af::ascir_op::Sum sum("dual_sum");
      sum.x = chain_head;
      sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
      sum.attr.sched.axis = output_axes;
      SetNodeView(sum, value0_dtype, output_axes, terminal_repeats, terminal_strides);
      store.x = sum.y;
    } else {
      store.x = chain_head;
    }
  };
  if (options.third_indirect_load_merged) {
    // 第三路 IL：独立 input/index（独立轴系），经第二个 Add 树形归并进共享输出链。
    const af::Expression s0c = make_size("s0c", 32);
    const af::Expression s1c = make_size("s1c", 16);
    const auto u0 = graph.CreateAxis("u0", s0c);
    const auto u1 = graph.CreateAxis("u1", s1c);
    const std::vector<af::AxisId> input2_axes = {u0.id, u1.id};
    const std::vector<af::Expression> input2_repeats = {s0c, s1c};
    const std::vector<af::Expression> input2_strides = {s1c, af::sym::kSymbolOne};
    af::ascir_op::Data x2("dual_x2", graph);
    x2.ir_attr.SetIndex(4);
    SetNodeView(x2, value0_dtype, input2_axes, input2_repeats, input2_strides);
    af::ascir_op::Load input_load2("dual_input_load2");
    input_load2.x = x2.y;
    SetNodeView(input_load2, value0_dtype, input2_axes, input2_repeats, input2_strides);
    af::ascir_op::Data index2("dual_index2", graph);
    index2.ir_attr.SetIndex(5);
    SetNodeView(index2, af::DT_INT32, output_axes, output_repeats, output_strides);
    af::ascir_op::Load index_load2("dual_index_load2");
    index_load2.x = index2.y;
    SetNodeView(index_load2, af::DT_INT32, output_axes, output_repeats, output_strides);
    af::ascir_op::IndirectLoad indirect_load2("dual_indirect_load2");
    indirect_load2.x1 = input_load2.y;
    indirect_load2.x2 = index_load2.y;
    indirect_load2.ir_attr.SetAxis(1);
    SetNodeView(indirect_load2, value0_dtype, output_axes, output_repeats, output_strides);
    af::ascir_op::Add merge_add2("dual_merge_add2");
    merge_add2.x1 = merge_add.y;
    merge_add2.x2 = indirect_load2.y;
    SetNodeView(merge_add2, value0_dtype, output_axes, output_repeats, output_strides);
    SetVectorApi(merge_add2);
    wire_terminal(merge_add2.y);
  } else {
    wire_terminal(merge_add.y);
  }
  SetNodeView(store, value0_dtype, output_axes, terminal_repeats, terminal_strides);
  af::ascir_op::Output output("dual_output");
  output.x = store.y;
  output.ir_attr.SetIndex(0);
  SetNodeView(output, value0_dtype, output_axes, terminal_repeats, terminal_strides);
  return graph;
}

std::vector<af::AscGraph> GenerateDualGatherCases(const DualGatherGraphOptions &options = {}) {
  auto graph = BuildDualIndirectLoadGraph(options);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  EXPECT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  return graphs;
}

TEST(IndirectLoadScheduleCaseGeneratorTest, DualGatherSimdCandidatesUseBothTiers) {
  const auto graphs = GenerateDualGatherCases();
  // 模板阶梯：双 gather 生成 SIMD 两档 + SIMT 一档共 3 个候选（SIMT 组员形态见
  // MultiGatherGeneratesSimdAndSimtCandidates）；此处验证候选内两 IL 模板一致与两档
  // SIMD 实现档位差异。
  ASSERT_EQ(graphs.size(), 3UL);
  for (const auto &candidate : graphs) {
    const auto il0 = candidate.FindNode("dual_indirect_load0");
    const auto il1 = candidate.FindNode("dual_indirect_load1");
    ASSERT_NE(il0, nullptr);
    ASSERT_NE(il1, nullptr);
    EXPECT_EQ(ascir::GetTemplateIdOrDefault(*il0), ascir::GetTemplateIdOrDefault(*il1));
  }
  ascgen_utils::indirect_load::Implementation implementation0 = ascgen_utils::indirect_load::Implementation::kDefault;
  ascgen_utils::indirect_load::Implementation implementation1 = ascgen_utils::indirect_load::Implementation::kDefault;
  ASSERT_EQ(ascgen_utils::indirect_load::GetImplementation(graphs[0].FindNode("dual_indirect_load0"), implementation0),
            af::SUCCESS);
  ASSERT_EQ(ascgen_utils::indirect_load::GetImplementation(graphs[1].FindNode("dual_indirect_load0"), implementation1),
            af::SUCCESS);
  EXPECT_NE(implementation0, implementation1);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, DualGatherSharesTemplateAxesAcrossPair) {
  // ExpectAxisNames/ExpectFixedTileSplit 需要非 const 的 AscGraph 引用（与单 gather 用例一致）。
  auto graphs = GenerateDualGatherCases();
  ASSERT_EQ(graphs.size(), 3UL);
  auto &simd_graph = graphs.front();
  const auto il0 = simd_graph.FindNode("dual_indirect_load0");
  const auto il1 = simd_graph.FindNode("dual_indirect_load1");
  ASSERT_NE(il0, nullptr);
  ASSERT_NE(il1, nullptr);
  ascgen_utils::indirect_load::TemplateAxes axes0;
  ascgen_utils::indirect_load::TemplateAxes axes1;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(il0, axes0), af::SUCCESS);
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(il1, axes1), af::SUCCESS);
  // primary 的切分规则与循环组织（outer/inner/tile/vectorized）对 secondary 生效。
  EXPECT_EQ(axes1.outer_axis, axes0.outer_axis);
  EXPECT_EQ(axes1.inner_axis, axes0.inner_axis);
  EXPECT_EQ(axes1.tile_outer_axis, axes0.tile_outer_axis);
  EXPECT_EQ(axes1.tile_inner_axis, axes0.tile_inner_axis);
  EXPECT_EQ(axes1.vectorized_axes, axes0.vectorized_axes);
  // 两路 input 轴不同：input 内轴独立构建。
  EXPECT_NE(axes1.input_inner_axis, axes0.input_inner_axis);
  ExpectAxisNames(simd_graph, {axes0.outer_axis}, {"z2"});
  ExpectAxisNames(simd_graph, {axes0.inner_axis}, {"z3"});
  ExpectAxisNames(simd_graph, {axes0.input_inner_axis}, {"z1"});
  ExpectAxisNames(simd_graph, {axes1.input_inner_axis}, {"w1"});
  ExpectFixedTileSplit(simd_graph, axes0.outer_axis);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, DualGatherReusesInputInnerAxisForSharedInputAxes) {
  DualGatherGraphOptions options;
  options.same_input_axes = true;
  const auto graphs = GenerateDualGatherCases(options);
  ASSERT_EQ(graphs.size(), 3UL);
  const auto &simd_graph = graphs.front();
  const auto il0 = simd_graph.FindNode("dual_indirect_load0");
  const auto il1 = simd_graph.FindNode("dual_indirect_load1");
  ASSERT_NE(il0, nullptr);
  ASSERT_NE(il1, nullptr);
  ascgen_utils::indirect_load::TemplateAxes axes0;
  ascgen_utils::indirect_load::TemplateAxes axes1;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(il0, axes0), af::SUCCESS);
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(il1, axes1), af::SUCCESS);
  // 两路 input 轴后缀一致时复用同一 input 内轴，避免重复建轴。
  EXPECT_EQ(axes1.input_inner_axis, axes0.input_inner_axis);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, DualGatherSetsPerIlInputInnerAxisHints) {
  const auto graphs = GenerateDualGatherCases();
  ASSERT_EQ(graphs.size(), 3UL);
  const auto &simd_graph = graphs.front();
  const auto input_load0 = simd_graph.FindNode("dual_input_load0");
  const auto input_load1 = simd_graph.FindNode("dual_input_load1");
  ASSERT_NE(input_load0, nullptr);
  ASSERT_NE(input_load1, nullptr);
  // 调度期按节点提示区分两路 input 边界的向量化内轴归属。
  const auto hint0 = ascgen_utils::indirect_load::GetSimdInputInnerAxis(input_load0);
  const auto hint1 = ascgen_utils::indirect_load::GetSimdInputInnerAxis(input_load1);
  EXPECT_NE(hint0, af::kIdNone);
  EXPECT_NE(hint1, af::kIdNone);
  EXPECT_NE(hint0, hint1);
  const auto il0 = simd_graph.FindNode("dual_indirect_load0");
  const auto il1 = simd_graph.FindNode("dual_indirect_load1");
  ASSERT_NE(il0, nullptr);
  ASSERT_NE(il1, nullptr);
  ascgen_utils::indirect_load::TemplateAxes axes0;
  ascgen_utils::indirect_load::TemplateAxes axes1;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(il0, axes0), af::SUCCESS);
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(il1, axes1), af::SUCCESS);
  EXPECT_EQ(hint0, axes0.input_inner_axis);
  EXPECT_EQ(hint1, axes1.input_inner_axis);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, DualGatherMovesInputPreNodesAfterEachIndirectLoad) {
  DualGatherGraphOptions options;
  options.with_input_pre = true;
  auto graphs = GenerateDualGatherCases(options);
  ASSERT_EQ(graphs.size(), 3UL);
  auto &simd_graph = graphs.front();
  const auto il0 = simd_graph.FindNode("dual_indirect_load0");
  const auto il1 = simd_graph.FindNode("dual_indirect_load1");
  const auto pre_abs0 = simd_graph.FindNode("dual_pre_abs0");
  const auto pre_abs1 = simd_graph.FindNode("dual_pre_abs1");
  ASSERT_NE(il0, nullptr);
  ASSERT_NE(il1, nullptr);
  ASSERT_NE(pre_abs0, nullptr);
  ASSERT_NE(pre_abs1, nullptr);
  // 改图逻辑逐 IL 独立执行：两路前置 Abs 均搬到各自 gather 之后。
  EXPECT_EQ(ascgen_utils::indirect_load::GetInputProducer(il0, 0UL)->GetName(), "dual_input_load0");
  EXPECT_EQ(ascgen_utils::indirect_load::GetInputProducer(il1, 0UL)->GetName(), "dual_input_load1");
  EXPECT_EQ(ascgen_utils::indirect_load::GetInputProducer(pre_abs0, 0UL), il0);
  EXPECT_EQ(ascgen_utils::indirect_load::GetInputProducer(pre_abs1, 0UL), il1);
  EXPECT_EQ(ascgen_utils::indirect_load::GetOnlyOutputConsumer(il0), pre_abs0);
  EXPECT_EQ(ascgen_utils::indirect_load::GetOnlyOutputConsumer(il1), pre_abs1);
  // 后移的前置 ele 与单 gather 语义一致：sched.axis 取各自 IL 输出轴空间（两路共享 z2/z3）。
  // per-IL 输入内轴归属由 DualGatherSetsPerIlInputInnerAxisHints 验证。
  ExpectAxisNames(simd_graph, pre_abs0->attr.sched.axis, {"z2", "z3"});
  ExpectAxisNames(simd_graph, pre_abs1->attr.sched.axis, {"z2", "z3"});
}

TEST(IndirectLoadScheduleCaseGeneratorTest, DualGatherFusesSharedPostReduce) {
  DualGatherGraphOptions options;
  options.with_post_reduce = true;
  const auto graphs = GenerateDualGatherCases(options);
  ASSERT_EQ(graphs.size(), 3UL);
  for (const auto &candidate : graphs) {
    const auto il0 = candidate.FindNode("dual_indirect_load0");
    const auto il1 = candidate.FindNode("dual_indirect_load1");
    const auto sum = candidate.FindNode("dual_sum");
    const auto store = candidate.FindNode("dual_store");
    ASSERT_NE(il0, nullptr);
    ASSERT_NE(il1, nullptr);
    ASSERT_NE(sum, nullptr);
    ASSERT_NE(store, nullptr);
    // 共享输出链：store 的生产者为 post Reduce，两路 lowering 元数据齐备。
    EXPECT_EQ(ascgen_utils::indirect_load::GetInputProducer(store, 0UL), sum);
    EXPECT_EQ(ascgen_utils::indirect_load::GetPostReduceConsumer(il0), sum);
    EXPECT_EQ(ascgen_utils::indirect_load::GetPostReduceConsumer(il1), sum);
    EXPECT_TRUE(ascgen_utils::indirect_load::HasLoweringMetadata(il0));
    EXPECT_TRUE(ascgen_utils::indirect_load::HasLoweringMetadata(il1));
    // L3（UB 归并，共享 Reduce 终端）：归并链为扇出分支、Reduce 后的 Store 由普通向量
    // 发射读取——不做 L2 的 kSimtDirectGmBoundary 跳过标注（仅寄存器归并直写 GM）。
    if (ascir::GetTemplateIdOrDefault(*il0) == ascir::TemplateId::kIndirectLoadSimt) {
      EXPECT_NE(ascgen_utils::indirect_load::GetTemplateRole(store),
                ascgen_utils::indirect_load::TemplateRole::kSimtDirectGmBoundary);
    }
  }
}

TEST(IndirectLoadScheduleCaseGeneratorTest, GeneratorTaskKeepsDualPairInSameGroupedGraph) {
  auto graph = BuildDualIndirectLoadGraph();
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<optimize::ScheduleTask> tasks;
  ASSERT_EQ(generator.GeneratorTask(graph, tasks, {}), af::SUCCESS);
  ASSERT_EQ(tasks.size(), 3UL);
  for (const auto &task : tasks) {
    ASSERT_FALSE(task.score_func.empty());
    ASSERT_FALSE(task.grouped_graphs.empty());
    // 两路经归并节点连通，必须落在同一 grouped graph（单 kernel）。
    const auto grouped_with_pair =
        std::find_if(task.grouped_graphs.begin(), task.grouped_graphs.end(), [](const af::AscGraph &grouped) {
          return grouped.FindNode("dual_indirect_load0") != nullptr &&
                 grouped.FindNode("dual_indirect_load1") != nullptr;
        });
    ASSERT_NE(grouped_with_pair, task.grouped_graphs.end());
  }
}

TEST(IndirectLoadScheduleCaseGeneratorTest, DualGatherFailsWhenIndexShapeMismatch) {
  DualGatherGraphOptions options;
  options.index_shape_mismatch = true;
  auto graph = BuildDualIndirectLoadGraph(options);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  // GE_ASSERT 失败返回自有错误码而非 af::FAILED，与既有负向用例一致以非 SUCCESS 判定。
  EXPECT_NE(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  EXPECT_TRUE(graphs.empty());
}

TEST(IndirectLoadScheduleCaseGeneratorTest, DualGatherFailsWhenAxisMismatch) {
  DualGatherGraphOptions options;
  options.axis_mismatch = true;
  auto graph = BuildDualIndirectLoadGraph(options);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  EXPECT_NE(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  EXPECT_TRUE(graphs.empty());
}

TEST(IndirectLoadScheduleCaseGeneratorTest, DualGatherFailsWhenOutputChainsDiverge) {
  DualGatherGraphOptions options;
  options.independent_outputs = true;
  auto graph = BuildDualIndirectLoadGraph(options);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  EXPECT_NE(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  EXPECT_TRUE(graphs.empty());
}

TEST(IndirectLoadScheduleCaseGeneratorTest, MultiGatherFailsWhenMemberOutputDiverges) {
  DualGatherGraphOptions options;
  options.third_indirect_load = true;
  auto graph = BuildDualIndirectLoadGraph(options);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  // 第三路 IL 独立输出，与 primary 的输出闭包无公共节点，组配对失败。
  EXPECT_NE(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  EXPECT_TRUE(graphs.empty());
}

TEST(IndirectLoadScheduleCaseGeneratorTest, MultiGatherFusesThreeGathersThroughMergeTree) {
  // 三路 gather 经归并树（add(add(g0,g1),g2)）汇入同一条输出链：数量不设上限，
  // 全组共享 primary 的切分轴与循环组织，仅生成 SIMD 两档候选。
  DualGatherGraphOptions options;
  options.third_indirect_load_merged = true;
  const auto graphs = GenerateDualGatherCases(options);
  ASSERT_EQ(graphs.size(), 3UL);
  for (const auto &candidate : graphs) {
    const auto il0 = candidate.FindNode("dual_indirect_load0");
    const auto il1 = candidate.FindNode("dual_indirect_load1");
    const auto il2 = candidate.FindNode("dual_indirect_load2");
    ASSERT_NE(il0, nullptr);
    ASSERT_NE(il1, nullptr);
    ASSERT_NE(il2, nullptr);
    ASSERT_EQ(ascir::GetTemplateIdOrDefault(*il0), ascir::GetTemplateIdOrDefault(*il1));
    ASSERT_EQ(ascir::GetTemplateIdOrDefault(*il0), ascir::GetTemplateIdOrDefault(*il2));
    if (ascir::GetTemplateIdOrDefault(*il0) != ascir::TemplateId::kIndirectLoadSimd) {
      // SIMT 组员候选：共享轴断言仅适用于 SIMD 候选（SIMT 组员形态见
      // MultiGatherGeneratesSimdAndSimtCandidates）。
      continue;
    }
    ascgen_utils::indirect_load::TemplateAxes axes0;
    ascgen_utils::indirect_load::TemplateAxes axes1;
    ascgen_utils::indirect_load::TemplateAxes axes2;
    ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(il0, axes0), af::SUCCESS);
    ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(il1, axes1), af::SUCCESS);
    ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(il2, axes2), af::SUCCESS);
    // 全组共享 outer/inner/tile/vectorized；三路 input 轴系互不相同，各自独立构建 input 内轴。
    EXPECT_EQ(axes1.outer_axis, axes0.outer_axis);
    EXPECT_EQ(axes2.outer_axis, axes0.outer_axis);
    EXPECT_EQ(axes1.inner_axis, axes0.inner_axis);
    EXPECT_EQ(axes2.inner_axis, axes0.inner_axis);
    EXPECT_EQ(axes1.tile_outer_axis, axes0.tile_outer_axis);
    EXPECT_EQ(axes2.tile_outer_axis, axes0.tile_outer_axis);
    EXPECT_EQ(axes1.tile_inner_axis, axes0.tile_inner_axis);
    EXPECT_EQ(axes2.tile_inner_axis, axes0.tile_inner_axis);
    EXPECT_EQ(axes1.vectorized_axes, axes0.vectorized_axes);
    EXPECT_EQ(axes2.vectorized_axes, axes0.vectorized_axes);
    EXPECT_NE(axes1.input_inner_axis, axes0.input_inner_axis);
    EXPECT_NE(axes2.input_inner_axis, axes0.input_inner_axis);
    EXPECT_NE(axes2.input_inner_axis, axes1.input_inner_axis);
  }
  ascgen_utils::indirect_load::Implementation implementation0 = ascgen_utils::indirect_load::Implementation::kDefault;
  ascgen_utils::indirect_load::Implementation implementation1 = ascgen_utils::indirect_load::Implementation::kDefault;
  ASSERT_NE(graphs[0].FindNode("dual_indirect_load0"), nullptr);
  ASSERT_NE(graphs[1].FindNode("dual_indirect_load0"), nullptr);
  EXPECT_EQ(ascgen_utils::indirect_load::GetImplementation(graphs[0].FindNode("dual_indirect_load0"), implementation0),
            af::SUCCESS);
  EXPECT_EQ(ascgen_utils::indirect_load::GetImplementation(graphs[1].FindNode("dual_indirect_load0"), implementation1),
            af::SUCCESS);
  EXPECT_NE(implementation0, implementation1);
}
TEST(IndirectLoadScheduleCaseGeneratorTest, MultiGatherGeneratesSimdAndSimtCandidates) {
  // 模板阶梯：双 gather 生成 SIMD 两档 + SIMT 一档共 3 个候选。SIMT 候选为组员形态——
  // 各路计划仅含本路 index 链，value 写 UB 局部目标（空变换链），共享归并/输出链
  // 不进入任何组员的 SIMT 计划。
  // L2 判定（CanEmitSimtScalar 逐节点查 V2 标量注册表）依赖平台上下文——与既有平台
  // 相关用例一致显式设置；未设置平台时 impl 解析失败会把 SIMT 候选整体降为 L3。
  PlatformContextReset platform_reset;
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformInfo platform_info;
  platform_info.soc_ver = "3510";
  platform_info.ub_size = 256 * 1024;
  platform_info.aiv_num = 48;
  ge::PlatformContext::GetInstance().SetPlatformInfo(platform_info);
  const auto graphs = GenerateDualGatherCases();
  ASSERT_EQ(graphs.size(), 3UL);
  size_t simd_count = 0UL;
  size_t simt_count = 0UL;
  for (const auto &candidate : graphs) {
    const auto il0 = candidate.FindNode("dual_indirect_load0");
    const auto il1 = candidate.FindNode("dual_indirect_load1");
    ASSERT_NE(il0, nullptr);
    ASSERT_NE(il1, nullptr);
    const auto template0 = ascir::GetTemplateIdOrDefault(*il0);
    ASSERT_EQ(template0, ascir::GetTemplateIdOrDefault(*il1));
    if (template0 == ascir::TemplateId::kIndirectLoadSimd) {
      ++simd_count;
      continue;
    }
    ASSERT_EQ(template0, ascir::TemplateId::kIndirectLoadSimt);
    ++simt_count;
    EXPECT_TRUE(ascgen_utils::indirect_load::IsSimtGroupMember(il0));
    EXPECT_TRUE(ascgen_utils::indirect_load::IsSimtGroupMember(il1));
    // L2 寄存器归并：evaluator 归并结果直写 Store 的 GM 目标（OutputTargets），Store 必须
    // 跳过独立发射（kSimtDirectGmBoundary，与单 gather output_store 语义一致——否则普通
    // 发射与直写冲突：重复写/读未产出的张量）。
    const auto fused_store = candidate.FindNode("dual_store");
    ASSERT_NE(fused_store, nullptr);
    EXPECT_EQ(ascgen_utils::indirect_load::GetTemplateRole(fused_store),
              ascgen_utils::indirect_load::TemplateRole::kSimtDirectGmBoundary);
    // seq-V1：primary 携带融合发射标记与全组成员名；非 primary 无标记（codegen 跳过其发射）。
    EXPECT_TRUE(ascgen_utils::indirect_load::IsSimtFusedPrimary(il0));
    EXPECT_FALSE(ascgen_utils::indirect_load::IsSimtFusedPrimary(il1));
    const auto fused_members = ascgen_utils::indirect_load::GetSimtFusedMemberNames(il0);
    ASSERT_EQ(fused_members.size(), 2UL);
    EXPECT_EQ(fused_members[0], "dual_indirect_load0");
    EXPECT_EQ(fused_members[1], "dual_indirect_load1");
    for (const af::AscNodePtr &member : {il0, il1}) {
      ascgen_utils::indirect_load::IndirectLoadLoweringMetadata metadata;
      ASSERT_EQ(ascgen_utils::indirect_load::GetLoweringMetadata(member, metadata), af::SUCCESS);
      EXPECT_FALSE(metadata.simt.has_post_reduce);
      ASSERT_EQ(metadata.simt.output_chains.size(), 1UL);
      EXPECT_TRUE(metadata.simt.output_chains[0].local_target);
      EXPECT_TRUE(metadata.simt.output_chains[0].node_names.empty());
      EXPECT_EQ(metadata.simt.output_chains[0].result_tensor_id, metadata.simt.value_tensor_id);
      EXPECT_EQ(metadata.simt.output_chains[0].target_tensor_id, metadata.simt.value_tensor_id);
      EXPECT_EQ(metadata.simt.output_result_tensor_id, metadata.simt.value_tensor_id);
    }
  }
  EXPECT_EQ(simd_count, 2UL);
  EXPECT_EQ(simt_count, 1UL);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, MultiGatherSimtLadderForOneByteValues) {
  // 模板阶梯（dtype 维度）：1 字节值（int8）仅 SIMT 标量路径支持——SIMD 两档被 dtype
  // 门控拒绝，仅生成 SIMT 一档候选。
  DualGatherGraphOptions options;
  options.one_byte_value = true;
  const auto graphs = GenerateDualGatherCases(options);
  ASSERT_EQ(graphs.size(), 1UL);
  const auto il0 = graphs[0].FindNode("dual_indirect_load0");
  const auto il1 = graphs[0].FindNode("dual_indirect_load1");
  ASSERT_NE(il0, nullptr);
  ASSERT_NE(il1, nullptr);
  EXPECT_EQ(ascir::GetTemplateIdOrDefault(*il0), ascir::TemplateId::kIndirectLoadSimt);
  EXPECT_EQ(ascir::GetTemplateIdOrDefault(*il1), ascir::TemplateId::kIndirectLoadSimt);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, MultiGatherMixedTemplateFallbackForHeterogeneousMembers) {
  // L5 混模板独立兜底：member0（fp16，index 链含非法外部消费者 → SIMT 不可行）与
  // member1（int8 值 → SIMD 被 dtype 门控拒绝）输入侧可行性互补——共模板候选全灭后，
  // 逐成员探测得到 {SIMD, SIMT} 异构选择，各成员以各自模板标注于同一 kernel。
  DualGatherGraphOptions options;
  options.first_index_illegal_consumer = true;
  options.second_value_one_byte = true;
  const auto graphs = GenerateDualGatherCases(options);
  ASSERT_EQ(graphs.size(), 1UL);
  const auto il0 = graphs[0].FindNode("dual_indirect_load0");
  const auto il1 = graphs[0].FindNode("dual_indirect_load1");
  ASSERT_NE(il0, nullptr);
  ASSERT_NE(il1, nullptr);
  EXPECT_EQ(ascir::GetTemplateIdOrDefault(*il0), ascir::TemplateId::kIndirectLoadSimd);
  EXPECT_EQ(ascir::GetTemplateIdOrDefault(*il1), ascir::TemplateId::kIndirectLoadSimt);
  // SIMT 成员组员形态：标记组员但不设融合发射标记（与共模板 SIMT 候选不同，各成员走
  // 标准单 gather invocation），value 写 UB 局部目标（空变换链）。
  EXPECT_TRUE(ascgen_utils::indirect_load::IsSimtGroupMember(il1));
  EXPECT_FALSE(ascgen_utils::indirect_load::IsSimtFusedPrimary(il1));
  EXPECT_FALSE(ascgen_utils::indirect_load::IsSimtFusedMemberEmitted(il1));
  ascgen_utils::indirect_load::IndirectLoadLoweringMetadata metadata;
  ASSERT_EQ(ascgen_utils::indirect_load::GetLoweringMetadata(il1, metadata), af::SUCCESS);
  EXPECT_FALSE(metadata.simt.has_post_reduce);
  ASSERT_EQ(metadata.simt.output_chains.size(), 1UL);
  EXPECT_TRUE(metadata.simt.output_chains[0].local_target);
  // M2 执行顺序（元数据级验证）：归并/输出链（含 int8→fp16 Cast）标为扇出分支，在全部
  // 成员发射之后作为普通向量代码执行——SIMD 成员输出 UB 张量与 SIMT 成员 UB 局部目标
  // 均由该段消费（执行顺序的端到端验证随 ST/E2E 补齐）。
  const auto merge_add = graphs[0].FindNode("dual_merge_add");
  const auto value_cast1 = graphs[0].FindNode("dual_cast1");
  ASSERT_NE(merge_add, nullptr);
  ASSERT_NE(value_cast1, nullptr);
  EXPECT_EQ(ascgen_utils::indirect_load::GetTemplateRole(merge_add),
            ascgen_utils::indirect_load::TemplateRole::kSimtFanoutBranch);
  EXPECT_EQ(ascgen_utils::indirect_load::GetTemplateRole(value_cast1),
            ascgen_utils::indirect_load::TemplateRole::kSimtFanoutBranch);
  // L5 混模板：SIMD 成员向量发射 + SIMT 成员标准单 gather invocation，归并链值由扇出
  // 分支向量代码产出——Store 正常发射，不做 L2 的 kSimtDirectGmBoundary 跳过标注。
  const auto mixed_store = graphs[0].FindNode("dual_store");
  ASSERT_NE(mixed_store, nullptr);
  EXPECT_NE(ascgen_utils::indirect_load::GetTemplateRole(mixed_store),
            ascgen_utils::indirect_load::TemplateRole::kSimtDirectGmBoundary);
  // SIMD 子集建轴，SIMT 成员复用同一 TemplateAxes（outer/inner/tile/vectorized 一致；
  // SIMT 仅消费 outer，含 SIMD 专用轴的整体拷贝见检视 M3）。
  ascgen_utils::indirect_load::TemplateAxes axes0;
  ascgen_utils::indirect_load::TemplateAxes axes1;
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(il0, axes0), af::SUCCESS);
  ASSERT_EQ(ascgen_utils::indirect_load::GetTemplateAxes(il1, axes1), af::SUCCESS);
  EXPECT_EQ(axes1.outer_axis, axes0.outer_axis);
  EXPECT_EQ(axes1.inner_axis, axes0.inner_axis);
  EXPECT_EQ(axes1.tile_outer_axis, axes0.tile_outer_axis);
  EXPECT_EQ(axes1.tile_inner_axis, axes0.tile_inner_axis);
  EXPECT_EQ(axes1.vectorized_axes, axes0.vectorized_axes);
}

TEST(IndirectLoadScheduleCaseGeneratorTest, MultiGatherSkBridgeFallbackSegmentsMembers) {
  // L6 SK 分段桥接（终态兜底）：双成员 index 链含非法外部消费者（SIMT 候选否决）且共享
  // post Reduce 归约外轴（R>G，SIMD 候选否决）——前五级全灭，逐成员复用单 gather SK
  // 流水：各成员独立 workspace 边界切断成员间及成员↔消费段连通性。
  DualGatherGraphOptions options;
  options.first_index_illegal_consumer = true;
  options.second_index_illegal_consumer = true;
  options.with_post_reduce = true;
  options.reduce_outer = true;
  const auto graphs = GenerateDualGatherCases(options);
  ASSERT_EQ(graphs.size(), 1UL);
  const auto &sk_graph = graphs[0];
  const auto il0 = sk_graph.FindNode("dual_indirect_load0");
  const auto il1 = sk_graph.FindNode("dual_indirect_load1");
  ASSERT_NE(il0, nullptr);
  ASSERT_NE(il1, nullptr);
  EXPECT_EQ(ascir::GetTemplateIdOrDefault(*il0), ascir::TemplateId::kIndirectLoadSK);
  EXPECT_EQ(ascir::GetTemplateIdOrDefault(*il1), ascir::TemplateId::kIndirectLoadSK);
  // 各成员独立的 output workspace 边界（成员名前缀，连通性切分依据）。
  EXPECT_NE(sk_graph.FindNode("dual_indirect_load0_sk_output_store"), nullptr);
  EXPECT_NE(sk_graph.FindNode("dual_indirect_load1_sk_output_store"), nullptr);

  // 分段语义：GeneratorTask 分区后各 IL 落入独立段（段内恒单 IL），归并/输出链作为
  // 消费段独立调度（读各路 workspace）。
  af::AscGraph graph = BuildDualIndirectLoadGraph(options);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<optimize::ScheduleTask> tasks;
  ASSERT_EQ(generator.GeneratorTask(graph, tasks, {}), af::SUCCESS);
  ASSERT_EQ(tasks.size(), 1UL);
  const auto &grouped = tasks[0].grouped_graphs;
  ASSERT_FALSE(grouped.empty());
  const auto find_group = [&grouped](const char *name) {
    for (size_t index = 0UL; index < grouped.size(); ++index) {
      if (grouped[index].FindNode(name) != nullptr) {
        return index;
      }
    }
    return grouped.size();
  };
  const size_t group_il0 = find_group("dual_indirect_load0");
  const size_t group_il1 = find_group("dual_indirect_load1");
  const size_t group_merge = find_group("dual_merge_add");
  ASSERT_LT(group_il0, grouped.size());
  ASSERT_LT(group_il1, grouped.size());
  ASSERT_LT(group_merge, grouped.size());
  EXPECT_NE(group_il0, group_il1);
  EXPECT_NE(group_il0, group_merge);
  EXPECT_NE(group_il1, group_merge);
  // 段内恒单 IL（H1 连通性分区正确性：段内单 IL 假设全部成立）。
  for (const auto &segment : grouped) {
    const bool has_il0 = segment.FindNode("dual_indirect_load0") != nullptr;
    const bool has_il1 = segment.FindNode("dual_indirect_load1") != nullptr;
    EXPECT_FALSE(has_il0 && has_il1);
  }
}

TEST(IndirectLoadScheduleCaseGeneratorTest, MultiGatherSkBridgeRejectsOneByteValueMembers) {
  // L6 dtype 门控（与单 gather ApplyGraphPass 对齐）：SK 的 AscendC::Gather 不支持 1 字节
  // 值。双成员 int8 + index 链非法外部消费者（SIMT 不可行）时六级阶梯全灭，不产生候选
  // （由前端 BackendSpec 预防兜底承接），而非生成 codegen 不可支持的 SK 候选。
  DualGatherGraphOptions options;
  options.first_index_illegal_consumer = true;
  options.second_index_illegal_consumer = true;
  options.first_value_one_byte = true;
  options.second_value_one_byte = true;
  af::AscGraph graph = BuildDualIndirectLoadGraph(options);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  EXPECT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  EXPECT_TRUE(graphs.empty());
}

}  // namespace

// ==================== SK分段SIMT化（R轴超限重定向）====================

// 构造带 Reduce 链的 gather 图：index 可切换收敛/非收敛，shape 可调以触发/不触发
// 窗口超限。
af::AscGraph BuildSkRedirectGraph(bool non_converged_index, int64_t input_rows, bool composite_norm,
                                  int64_t dim = 20000) {
  af::AscGraph graph("sk_redirect_ut_graph");
  const af::Expression s_rows = graph.CreateSizeVar(1);
  const af::Expression s_dim = graph.CreateSizeVar(dim);
  const af::Expression s_table = graph.CreateSizeVar(input_rows);
  const auto a0 = graph.CreateAxis("a0", s_rows);
  const auto a1 = graph.CreateAxis("a1", s_dim);
  const std::vector<af::AxisId> axes = {a0.id, a1.id};
  const std::vector<af::Expression> full = {s_rows, s_dim};
  const std::vector<af::Expression> full_strides = {s_dim, af::sym::kSymbolOne};
  const std::vector<af::Expression> row = {s_rows, af::sym::kSymbolOne};
  const std::vector<af::Expression> row_strides = {af::sym::kSymbolOne, af::sym::kSymbolZero};

  af::ascir_op::Data table("table", graph);
  table.ir_attr.SetIndex(0);
  table.y.dtype = af::DT_FLOAT;
  af::ascir_op::Load table_load("table_load");
  table_load.x = table.y;
  SetNodeView(table_load, af::DT_FLOAT, axes, {s_table, s_dim}, {s_dim, af::sym::kSymbolOne});

  af::ascir_op::Data index("index", graph);
  index.ir_attr.SetIndex(1);
  index.y.dtype = af::DT_INT64;
  af::ascir_op::Load index_load("index_load");
  index_load.x = index.y;
  if (non_converged_index) {
    // 非收敛：每元素独立行号（全轴 stride 非零）。
    SetNodeView(index_load, af::DT_INT64, axes, full, full_strides);
  } else {
    // 收敛：每行一个行号（尾轴零贡献）。
    SetNodeView(index_load, af::DT_INT64, axes, row, row_strides);
  }

  af::ascir_op::IndirectLoad indirect_load("indirect_load");
  indirect_load.x1 = table_load.y;
  if (non_converged_index) {
    indirect_load.x2 = index_load.y;
  } else {
    // 收敛形态经 Broadcast 补齐逻辑形状（IL 语义要求 index 与输出 repeats 匹配）：
    // 源头 [rows,1]/[1,0]（尾轴零贡献）广播到 [rows,dim]。
    af::ascir_op::Broadcast index_broadcast("index_broadcast");
    index_broadcast.x = index_load.y;
    SetNodeView(index_broadcast, af::DT_INT64, axes, full, full_strides);
    indirect_load.x2 = index_broadcast.y;
  }
  indirect_load.ir_attr.SetAxis(0);
  indirect_load.ir_attr.SetNegative_index_support(true);
  indirect_load.ir_attr.SetNeed_check_bound(true);
  indirect_load.ir_attr.SetMax(s_table);
  SetNodeView(indirect_load, af::DT_FLOAT, axes, full, full_strides);

  if (composite_norm) {
    // 双 Reduce 复合链（LayerNorm 形态）：mean → broadcast → diff（菱形）→ var。
    af::ascir_op::Sum mean_sum("mean_sum");
    mean_sum.x = indirect_load.y;
    mean_sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
    mean_sum.attr.sched.axis = axes;
    SetNodeView(mean_sum, af::DT_FLOAT, axes, row, row_strides);
    af::ascir_op::Broadcast mean_broadcast("mean_broadcast");
    mean_broadcast.x = mean_sum.y;
    SetNodeView(mean_broadcast, af::DT_FLOAT, axes, full, full_strides);
    af::ascir_op::Sub diff("diff");
    diff.x1 = indirect_load.y;
    diff.x2 = mean_broadcast.y;
    SetNodeView(diff, af::DT_FLOAT, axes, full, full_strides);
    af::ascir_op::Mul square("square");
    square.x1 = diff.y;
    square.x2 = diff.y;
    SetNodeView(square, af::DT_FLOAT, axes, full, full_strides);
    af::ascir_op::Sum var_sum("var_sum");
    var_sum.x = square.y;
    var_sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
    var_sum.attr.sched.axis = axes;
    SetNodeView(var_sum, af::DT_FLOAT, axes, row, row_strides);
    af::ascir_op::Broadcast var_broadcast("var_broadcast");
    var_broadcast.x = var_sum.y;
    SetNodeView(var_broadcast, af::DT_FLOAT, axes, full, full_strides);
    af::ascir_op::Mul norm_out("norm_out");
    norm_out.x1 = diff.y;
    norm_out.x2 = var_broadcast.y;
    SetNodeView(norm_out, af::DT_FLOAT, axes, full, full_strides);
    af::ascir_op::Store store("store");
    store.x = norm_out.y;
    SetNodeView(store, af::DT_FLOAT, axes, full, full_strides);
    af::ascir_op::Output output("output");
    output.ir_attr.SetIndex(0);
    output.x = store.y;
    SetNodeView(output, af::DT_FLOAT, axes, full, full_strides);
  } else {
    // 单 Reduce 链。
    af::ascir_op::Sum sum("sum");
    sum.x = indirect_load.y;
    sum.attr.api.compute_type = af::ComputeType::kComputeReduce;
    sum.attr.sched.axis = axes;
    SetNodeView(sum, af::DT_FLOAT, axes, row, row_strides);
    af::ascir_op::Store store("store");
    store.x = sum.y;
    SetNodeView(store, af::DT_FLOAT, axes, row, row_strides);
    af::ascir_op::Output output("output");
    output.ir_attr.SetIndex(0);
    output.x = store.y;
    SetNodeView(output, af::DT_FLOAT, axes, row, row_strides);
  }
  return graph;
}

void SetUpSkRedirectPlatform() {
  ge::PlatformContext::GetInstance().Reset();
  ge::PlatformInfo platform_info;
  platform_info.soc_ver = "3510";
  platform_info.ub_size = 245760;
  platform_info.aiv_num = 48;
  ge::PlatformContext::GetInstance().SetPlatformInfo(platform_info);
}

// 非收敛 index（逐元素独立行号）：index_source_converged 应为 false。
TEST(IndirectLoadScheduleCaseGeneratorTest, AccessInfoMarksNonConvergedIndexSource) {
  auto graph = BuildSkRedirectGraph(true, 4, false);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  ASSERT_FALSE(graphs.empty());
  // AccessInfo 在候选生成的 metadata 阶段构建，从任一候选图的 IL 节点读取。
  const auto indirect_load = graphs.front().FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  ascgen_utils::indirect_load::IndirectLoadAccessInfo access_info;
  ASSERT_EQ(ascgen_utils::indirect_load::GetIndirectLoadAccessInfo(indirect_load, access_info), af::SUCCESS);
  EXPECT_FALSE(access_info.index_source_converged);
}

// 收敛 index（尾轴零贡献，每行一个行号）：index_source_converged 应为 true。
TEST(IndirectLoadScheduleCaseGeneratorTest, AccessInfoMarksConvergedIndexSource) {
  auto graph = BuildSkRedirectGraph(false, 4, false);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  ASSERT_FALSE(graphs.empty());
  const auto indirect_load = graphs.front().FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  ascgen_utils::indirect_load::IndirectLoadAccessInfo access_info;
  ASSERT_EQ(ascgen_utils::indirect_load::GetIndirectLoadAccessInfo(indirect_load, access_info), af::SUCCESS);
  EXPECT_TRUE(access_info.index_source_converged);
}

// 窗口超限 + 单 Reduce：SK 候选重定向为 SIMT（分段标记 + output workspace 边界，
// input/index 无 workspace 边界）。
TEST(IndirectLoadScheduleCaseGeneratorTest, SkRedirectsToSimtWhenWindowExceedsUb) {
  PlatformContextReset platform_reset;
  SetUpSkRedirectPlatform();
  // input=[8192,256]×4B=8MB + index=[1,256]×8B > UB(245KB)。
  auto graph = BuildSkRedirectGraph(true, 8192, false);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  // 重定向后图列表含两个 SIMT template（原 SIMT 候选 + 重定向 SK），按分段标记定位。
  auto sk_iter = graphs.end();
  for (auto iter = graphs.begin(); iter != graphs.end(); ++iter) {
    const auto candidate = iter->FindNode("indirect_load");
    if (candidate != nullptr && ascir::GetTemplateIdOrDefault(*candidate) == ascir::TemplateId::kIndirectLoadSimt &&
        ascir::IsSkSegmentedSimt(*candidate)) {
      sk_iter = iter;
      break;
    }
  }
  ASSERT_NE(sk_iter, graphs.end());
  auto &sk_graph = *sk_iter;
  const auto indirect_load = sk_graph.FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  EXPECT_EQ(ascir::GetTemplateIdOrDefault(*indirect_load), ascir::TemplateId::kIndirectLoadSimt);
  EXPECT_EQ(ascir::GetTemplateIdOrDefault(*indirect_load), ascir::TemplateId::kIndirectLoadSimt);
  // output 侧 workspace 边界存在，input/index 侧不存在（IL 直读 GM）。
  bool has_output_boundary = false;
  bool has_input_boundary = false;
  for (const auto &node : sk_graph.GetAllNodes()) {
    const auto &name = node->GetName();
    if (name.find("indirect_load_sk_output") != std::string::npos && af::ops::IsOps<af::ascir_op::Workspace>(node)) {
      has_output_boundary = true;
    }
    if ((name.find("indirect_load_sk_input") != std::string::npos ||
         name.find("indirect_load_sk_index") != std::string::npos) &&
        af::ops::IsOps<af::ascir_op::Workspace>(node)) {
      has_input_boundary = true;
    }
  }
  EXPECT_TRUE(has_output_boundary);
  EXPECT_FALSE(has_input_boundary);
}

// 窗口不超限（小表）：SK 候选保持原行窗口路径（template=SK，无分段标记）。
TEST(IndirectLoadScheduleCaseGeneratorTest, SkKeepsOriginalPathWhenWindowFits) {
  PlatformContextReset platform_reset;
  SetUpSkRedirectPlatform();
  // input=[4,256]×4B=4KB < UB。
  auto graph = BuildSkRedirectGraph(true, 4, false, 256);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  const auto sk_iter = FindGeneratedGraphByTemplate(graphs, ascir::TemplateId::kIndirectLoadSK);
  ASSERT_NE(sk_iter, graphs.end());
  const auto indirect_load = sk_iter->FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  EXPECT_FALSE(ascir::IsSkSegmentedSimt(*indirect_load));
}

// 复合 Norm（双 Reduce 菱形）+ 窗口超限：重定向后 Norm 链的 Reduce/多消费者输出
// 均插入 workspace 边界（每段数据流自足）。
TEST(IndirectLoadScheduleCaseGeneratorTest, SkRedirectsCompositeNormChain) {
  PlatformContextReset platform_reset;
  SetUpSkRedirectPlatform();
  auto graph = BuildSkRedirectGraph(true, 8192, true);
  optimize::IndirectLoadScheduleCaseGenerator generator;
  std::vector<af::AscGraph> graphs;
  std::vector<std::string> score_functions;
  ASSERT_EQ(generator.Generate(graph, graphs, score_functions), af::SUCCESS);
  auto sk_iter = graphs.end();
  for (auto iter = graphs.begin(); iter != graphs.end(); ++iter) {
    const auto candidate = iter->FindNode("indirect_load");
    if (candidate != nullptr && ascir::GetTemplateIdOrDefault(*candidate) == ascir::TemplateId::kIndirectLoadSimt &&
        ascir::IsSkSegmentedSimt(*candidate)) {
      sk_iter = iter;
      break;
    }
  }
  ASSERT_NE(sk_iter, graphs.end());
  auto &sk_graph = *sk_iter;
  const auto indirect_load = sk_graph.FindNode("indirect_load");
  ASSERT_NE(indirect_load, nullptr);
  // mean/var 两个 Reduce 与菱形 diff 的输出边界均存在。
  int32_t norm_boundaries = 0;
  for (const auto &node : sk_graph.GetAllNodes()) {
    const auto &name = node->GetName();
    if ((name.find("mean_sum_sk_n") != std::string::npos || name.find("var_sum_sk_n") != std::string::npos ||
         name.find("diff_sk_n") != std::string::npos) &&
        af::ops::IsOps<af::ascir_op::Workspace>(node)) {
      ++norm_boundaries;
    }
  }
  // mean(1) + diff(2消费者) + var(1) = 4 个边界 workspace 节点（pre 形态）。
  EXPECT_GE(norm_boundaries, 3);
}
