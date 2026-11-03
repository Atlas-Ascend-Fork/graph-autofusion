/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir_node_param/ascir_param_builder.h"

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "common/checker.h"
#include "common/ge_common/debug/log.h"

namespace ascir_param {
namespace {
constexpr const char *kAscirNodeParams = "AscirNodeParams";
constexpr const char *kVectorFunc = "VectorFunc";
constexpr const char *kCast = "Cast";
constexpr const char *kBroadcast = "Broadcast";
constexpr const char *kCeil2Int = "Ceil2Int";

bool IsCompareParamSupported(const std::string &api_name) {
  static const std::set<std::string> kCompareTypes = {"Ge", "Eq", "Ne", "Gt", "Le", "Lt"};
  return kCompareTypes.count(api_name) != 0U;
}

struct AscirParamSourceContext {
  af::AscNodePtr node;
  const af::AscGraph *graph{nullptr};
};

struct TensorParamView {
  bool valid{false};
  std::string name;
  ge::DataType dtype{ge::DT_UNDEFINED};
  uint32_t dtype_size{0U};
  std::vector<int64_t> axis_ids;
  std::vector<ge::Expression> vectorized_repeats;
  std::vector<ge::Expression> vectorized_strides;
  std::vector<ge::Expression> tensor_repeats;
  std::vector<int64_t> tensor_axis_ids;
  std::vector<ge::Expression> tensor_strides;
};

struct AxisParamView {
  int64_t axis_id{-1L};
  ge::Expression codegen_size_expr{ge::Symbol(1U)};
  ge::Expression semantic_size{ge::Symbol(1U)};
  std::vector<int64_t> parent_axis_ids;
  std::vector<int64_t> orig_axis_ids;
  bool is_original_axis{false};
};

struct AscirParamBuildContext {
  af::AscNodePtr node;
  const af::AscGraph *graph{nullptr};
  std::string api_name;
  std::vector<TensorParamView> inputs;
  std::vector<TensorParamView> outputs;
  int64_t loop_axis_id{-1L};
  std::vector<AxisParamView> loop_axes;
};

struct ReduceAxisCount {
  std::set<int64_t> visited;
  int64_t total_count{0L};
  int64_t valid_count{0L};
};

af::Status RegisterAscirNodeParams(const af::AscNodePtr &node, const AscirNodeParamsPtr &params) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_NOTNULL(params);
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  const auto owner_graph = node->GetOwnerComputeGraph();
  if (params->status != ParamBuildStatus::kSkipped) {
    const auto node_name = node->GetName();
    const auto node_type = node->GetType();
    const auto graph_name = owner_graph == nullptr ? std::string("<null>") : owner_graph->GetName();
    GELOGI(
        "[ASCIR_PARAM_TRACE] Register api[%s], status[%d], node[%s], type[%s], graph[%s], node_ptr[%p], "
        "op_desc_ptr[%p], owner_graph_ptr[%p].",
        params->api_name.c_str(), static_cast<int32_t>(params->status), node_name.c_str(), node_type.c_str(),
        graph_name.c_str(), static_cast<void *>(node.get()), static_cast<void *>(op_desc.get()),
        static_cast<void *>(owner_graph.get()));
  }
  const std::string graph_name = owner_graph == nullptr ? std::string("<null>") : owner_graph->GetName();
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kAscirNodeParams, params), "Graph:%s, Node:%s SetExtAttr failed",
                 graph_name.c_str(), node->GetNamePtr());
  return af::SUCCESS;
}

af::Status RegisterSkippedAscirNodeParams(const af::AscNodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  auto params = std::make_shared<AscirNodeParams>();
  params->api_name = node->GetType();
  params->status = ParamBuildStatus::kSkipped;
  return RegisterAscirNodeParams(node, params);
}

af::Status RegisterVectorFuncAscirNodeParams(const af::AscNodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  auto params = std::make_shared<AscirNodeParams>();
  params->api_name = node->GetType();
  params->status = ParamBuildStatus::kBuilt;
  params->specific_params = VectorFuncNodeParams{};
  return RegisterAscirNodeParams(node, params);
}

af::Status RegisterCastAscirNodeParams(const af::AscNodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  auto params = std::make_shared<AscirNodeParams>();
  params->api_name = node->GetType();
  params->status = ParamBuildStatus::kBuilt;
  params->specific_params = CastNodeParams{};
  return RegisterAscirNodeParams(node, params);
}

af::Status RegisterBroadcastAscirNodeParams(const af::AscNodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  const auto existing_params = GetAscirNodeParams(node);
  if (existing_params != nullptr) {
    const auto *broadcast_params = GetSpecificParams<BroadcastNodeParams>(*existing_params);
    if (broadcast_params != nullptr && broadcast_params->valid) {
      // CopyFrom 可能共享已有 Broadcast 载荷；仅此复用分支需要隔离候选的执行记录。
      auto params = std::make_shared<AscirNodeParams>();
      params->api_name = existing_params->api_name;
      params->status = existing_params->status;
      params->specific_params = existing_params->specific_params;
      return RegisterAscirNodeParams(node, params);
    }
  }
  auto params = std::make_shared<AscirNodeParams>();
  params->api_name = node->GetType();
  params->status = ParamBuildStatus::kBuilt;
  params->specific_params = BroadcastNodeParams{};
  return RegisterAscirNodeParams(node, params);
}

af::Status RegisterCompareAscirNodeParams(const af::AscNodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  auto params = std::make_shared<AscirNodeParams>();
  params->api_name = node->GetType();
  params->status = ParamBuildStatus::kBuilt;
  params->specific_params = CompareNodeParams{};
  return RegisterAscirNodeParams(node, params);
}

af::Status RegisterUnaryBitWidthChangeAscirNodeParams(const af::AscNodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  auto params = std::make_shared<AscirNodeParams>();
  params->api_name = node->GetType();
  params->status = ParamBuildStatus::kBuilt;
  params->specific_params = UnaryBitWidthChangeNodeParams{};
  return RegisterAscirNodeParams(node, params);
}

af::Status RegisterTransposeAscirNodeParams(const af::AscNodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  auto params = std::make_shared<AscirNodeParams>();
  params->api_name = node->GetType();
  params->status = ParamBuildStatus::kBuilt;
  params->specific_params = TransposeNodeParams{};
  return RegisterAscirNodeParams(node, params);
}

std::map<int64_t, af::AxisPtr> BuildAxisMap(const af::AscGraph &graph) {
  std::map<int64_t, af::AxisPtr> axis_map;
  for (const auto &axis : graph.GetAllAxis()) {
    if (axis != nullptr) {
      axis_map[axis->id] = axis;
    }
  }
  return axis_map;
}

void AddOriginAxisIds(int64_t axis_id, const std::map<int64_t, af::AxisPtr> &axis_map, std::set<int64_t> &visited,
                      std::vector<int64_t> &origin_axis_ids) {
  if (!visited.insert(axis_id).second) {
    return;
  }
  const auto iter = axis_map.find(axis_id);
  if (iter == axis_map.end() || iter->second == nullptr) {
    return;
  }
  if (iter->second->type == af::Axis::kAxisTypeOriginal) {
    origin_axis_ids.emplace_back(axis_id);
    return;
  }
  for (const auto from_axis_id : iter->second->from) {
    AddOriginAxisIds(from_axis_id, axis_map, visited, origin_axis_ids);
  }
}

std::vector<int64_t> BuildOriginAxisIds(int64_t axis_id, const std::map<int64_t, af::AxisPtr> &axis_map) {
  std::vector<int64_t> origin_axis_ids;
  std::set<int64_t> visited;
  AddOriginAxisIds(axis_id, axis_map, visited, origin_axis_ids);
  return origin_axis_ids;
}

void AddAxisParamView(int64_t axis_id, const std::map<int64_t, af::AxisPtr> &axis_map, std::set<int64_t> &visited,
                      std::vector<AxisParamView> &views) {
  if (!visited.insert(axis_id).second) {
    return;
  }
  const auto iter = axis_map.find(axis_id);
  if (iter == axis_map.end() || iter->second == nullptr) {
    return;
  }
  const auto &axis = *iter->second;
  views.push_back({axis.id, ge::Symbol(axis.size.Str().get()), axis.size, axis.from,
                   BuildOriginAxisIds(axis.id, axis_map), axis.type == af::Axis::kAxisTypeOriginal});
  for (const auto from_axis_id : axis.from) {
    AddAxisParamView(from_axis_id, axis_map, visited, views);
  }
}

std::map<int64_t, size_t> BuildTensorAxisIndexMap(const af::AscTensorAttr &tensor_attr) {
  std::map<int64_t, size_t> axis_index_map;
  for (size_t i = 0U; i < tensor_attr.axis.size(); ++i) {
    axis_index_map[tensor_attr.axis[i]] = i;
  }
  return axis_index_map;
}

TensorParamView BuildTensorParamView(const std::string &name, const af::AscTensorAttr &tensor_attr) {
  TensorParamView view;
  view.name = name;
  view.dtype = static_cast<ge::DataType>(tensor_attr.dtype);
  view.dtype_size = static_cast<uint32_t>(ge::GetSizeByDataType(tensor_attr.dtype));
  view.axis_ids = tensor_attr.vectorized_axis;
  view.tensor_repeats = tensor_attr.repeats;
  view.tensor_axis_ids = tensor_attr.axis;
  view.tensor_strides = tensor_attr.strides;
  view.valid = !view.axis_ids.empty() && tensor_attr.vectorized_strides.size() == view.axis_ids.size() &&
               view.tensor_axis_ids.size() == view.tensor_repeats.size() &&
               view.tensor_axis_ids.size() == view.tensor_strides.size();
  view.vectorized_strides = tensor_attr.vectorized_strides;
  const auto axis_index_map = BuildTensorAxisIndexMap(tensor_attr);
  for (const auto axis_id : view.axis_ids) {
    const auto iter = axis_index_map.find(axis_id);
    if (iter == axis_index_map.end() || iter->second >= tensor_attr.repeats.size() ||
        iter->second >= tensor_attr.strides.size()) {
      view.valid = false;
      return view;
    }
    view.vectorized_repeats.emplace_back(tensor_attr.repeats[iter->second]);
  }
  return view;
}

std::vector<TensorParamView> BuildInputTensorParamViews(const af::AscNodePtr &node) {
  std::vector<TensorParamView> views;
  views.reserve(node->inputs.Size());
  for (uint32_t i = 0U; i < node->inputs.Size(); ++i) {
    views.emplace_back(BuildTensorParamView(node->GetName() + "_input_" + std::to_string(i), node->inputs[i].attr));
  }
  return views;
}

std::vector<TensorParamView> BuildOutputTensorParamViews(const af::AscNodePtr &node) {
  const auto outputs = node->outputs();
  std::vector<TensorParamView> views;
  views.reserve(outputs.size());
  for (size_t i = 0U; i < outputs.size(); ++i) {
    views.emplace_back(BuildTensorParamView(node->GetName() + "_output_" + std::to_string(i), outputs[i]->attr));
  }
  return views;
}

af::Status BuildLoopAxisParamViews(const AscirParamSourceContext &source, std::vector<AxisParamView> &views) {
  const auto axis_map = BuildAxisMap(*source.graph);
  std::set<int64_t> visited;
  bool loop_inside_flag = false;
  for (const auto axis_id : source.node->attr.sched.axis) {
    const auto iter = axis_map.find(axis_id);
    GE_ASSERT_TRUE(iter != axis_map.end() && iter->second != nullptr, "Invalid schedule axis id[%ld].", axis_id);
    if (iter->second->type == af::Axis::kAxisTypeBlockOuter) {
      if (axis_id == source.node->attr.sched.loop_axis) {
        AddAxisParamView(axis_id, axis_map, visited, views);
        loop_inside_flag = true;
      }
      continue;
    }
    if (!loop_inside_flag) {
      AddAxisParamView(axis_id, axis_map, visited, views);
    }
    if (axis_id == source.node->attr.sched.loop_axis) {
      loop_inside_flag = true;
    }
  }
  return af::SUCCESS;
}

const std::vector<int64_t> &GetMultiReduceAxisIds(const TensorParamView &tensor) {
  return tensor.tensor_axis_ids.empty() ? tensor.axis_ids : tensor.tensor_axis_ids;
}

const std::vector<ge::Expression> &GetMultiReduceStrides(const TensorParamView &tensor) {
  return tensor.tensor_strides.empty() ? tensor.vectorized_strides : tensor.tensor_strides;
}

bool FindAxis(const std::vector<int64_t> &axis_ids, int64_t axis_id, size_t &axis_index) {
  const auto iter = std::find(axis_ids.begin(), axis_ids.end(), axis_id);
  if (iter == axis_ids.end()) {
    return false;
  }
  axis_index = static_cast<size_t>(std::distance(axis_ids.begin(), iter));
  return true;
}

bool FindMultiReduceAxis(const TensorParamView &tensor, int64_t axis_id, size_t &axis_index) {
  return FindAxis(GetMultiReduceAxisIds(tensor), axis_id, axis_index);
}

bool IsValidTensorView(const TensorParamView &tensor) {
  return tensor.valid && tensor.dtype_size > 0U && !tensor.axis_ids.empty() &&
         tensor.axis_ids.size() == tensor.vectorized_repeats.size() &&
         tensor.vectorized_repeats.size() == tensor.vectorized_strides.size() &&
         (tensor.tensor_axis_ids.empty() || tensor.tensor_axis_ids.size() == tensor.tensor_strides.size());
}

std::map<int64_t, AxisParamView> BuildAxisMap(const std::vector<AxisParamView> &axes) {
  std::map<int64_t, AxisParamView> axis_map;
  for (const auto &axis : axes) {
    axis_map[axis.axis_id] = axis;
  }
  return axis_map;
}

bool HasAxisClosure(const std::map<int64_t, AxisParamView> &axis_map, int64_t axis_id, std::set<int64_t> &visited) {
  if (!visited.insert(axis_id).second) {
    return true;
  }
  const auto iter = axis_map.find(axis_id);
  if (iter == axis_map.end()) {
    return false;
  }
  for (const auto parent_axis_id : iter->second.parent_axis_ids) {
    if (!HasAxisClosure(axis_map, parent_axis_id, visited)) {
      return false;
    }
  }
  for (const auto orig_axis_id : iter->second.orig_axis_ids) {
    if (!HasAxisClosure(axis_map, orig_axis_id, visited)) {
      return false;
    }
  }
  return true;
}

bool IsReduceAxisValid(const TensorParamView &input, const TensorParamView &output, int64_t axis_id) {
  size_t axis_index = 0U;
  if (!FindMultiReduceAxis(output, axis_id, axis_index)) {
    return false;
  }
  const auto &input_strides = GetMultiReduceStrides(input);
  const auto &output_strides = GetMultiReduceStrides(output);
  if (axis_index >= output_strides.size() || axis_index >= input_strides.size()) {
    return false;
  }
  return output_strides[axis_index] == af::sym::kSymbolZero && input_strides[axis_index] != af::sym::kSymbolZero;
}

bool TryCountOutputAxis(const TensorParamView &input, const TensorParamView &output, int64_t axis_id,
                        ReduceAxisCount &count) {
  size_t output_axis_index = 0U;
  if (!FindMultiReduceAxis(output, axis_id, output_axis_index)) {
    return false;
  }
  ++count.total_count;
  count.valid_count += IsReduceAxisValid(input, output, axis_id) ? 1L : 0L;
  return true;
}

void CountReduceAxis(const TensorParamView &input, const TensorParamView &output,
                     const std::map<int64_t, AxisParamView> &axis_map, int64_t axis_id, ReduceAxisCount &count) {
  if (!count.visited.insert(axis_id).second) {
    return;
  }
  if (TryCountOutputAxis(input, output, axis_id, count)) {
    return;
  }
  const auto iter = axis_map.find(axis_id);
  if (iter == axis_map.end()) {
    return;
  }
  for (const auto parent_axis_id : iter->second.parent_axis_ids) {
    if (TryCountOutputAxis(input, output, parent_axis_id, count)) {
      return;
    }
    const auto parent_iter = axis_map.find(parent_axis_id);
    if (parent_iter == axis_map.end() || parent_iter->second.is_original_axis ||
        parent_iter->second.parent_axis_ids.empty()) {
      continue;
    }
    for (const auto grand_parent_axis_id : parent_iter->second.parent_axis_ids) {
      CountReduceAxis(input, output, axis_map, grand_parent_axis_id, count);
    }
  }
}

const AxisParamView *FindLoopAxis(const AscirParamBuildContext &ctx) {
  const auto iter = std::find_if(ctx.loop_axes.begin(), ctx.loop_axes.end(),
                                 [&ctx](const AxisParamView &axis) { return axis.axis_id == ctx.loop_axis_id; });
  return iter == ctx.loop_axes.end() ? nullptr : &(*iter);
}

bool IsNeedMultiReduce(const AscirParamBuildContext &ctx) {
  const auto *loop_axis = FindLoopAxis(ctx);
  if (loop_axis == nullptr || ctx.inputs.empty() || ctx.outputs.empty()) {
    return true;
  }
  ReduceAxisCount count;
  CountReduceAxis(ctx.inputs[0], ctx.outputs[0], BuildAxisMap(ctx.loop_axes), loop_axis->axis_id, count);
  return count.total_count == count.valid_count;
}

ge::Expression GetReduceMergeTimes(const AscirParamBuildContext &ctx) {
  const auto *loop_axis = FindLoopAxis(ctx);
  if (loop_axis == nullptr || !loop_axis->codegen_size_expr.IsValid()) {
    return ge::Symbol(1U);
  }
  return loop_axis->codegen_size_expr;
}

ge::Expression GetSemanticReduceMergeTimes(const AscirParamBuildContext &ctx) {
  const auto *loop_axis = FindLoopAxis(ctx);
  if (loop_axis == nullptr || !loop_axis->semantic_size.IsValid()) {
    return ge::Symbol(1U);
  }
  return loop_axis->semantic_size;
}

ParamExprProduct MakeSemanticProductExpr(const ge::Expression &value) {
  return {true, {{value, ParamExprRole::kSemantic}}};
}

ParamExprProduct MakeSemanticProductExpr(const std::vector<ge::Expression> &values) {
  ParamExprProduct expr{true, {}};
  for (const auto &value : values) {
    expr.factors.push_back({value, ParamExprRole::kSemantic});
  }
  return expr;
}

ParamExprProduct BuildMergeSizeExpr(const AscirParamBuildContext &ctx) {
  return MakeSemanticProductExpr(ctx.outputs[0].vectorized_repeats);
}

ParamExprProduct BuildMergeTimesExpr(const AscirParamBuildContext &ctx, bool need_multi_reduce) {
  return MakeSemanticProductExpr(need_multi_reduce ? GetSemanticReduceMergeTimes(ctx) : ge::Symbol(1U));
}

codegen::ReducePattern GetReducePattern(const AscirParamBuildContext &ctx) {
  if (ctx.outputs.empty() || ctx.outputs[0].vectorized_strides.empty()) {
    return codegen::ReducePattern::kUnknown;
  }
  return ctx.outputs[0].vectorized_strides.back() == af::sym::kSymbolZero ? codegen::ReducePattern::kAR
                                                                          : codegen::ReducePattern::kRA;
}

af::Status GetFirstInputOwnerNode(const af::AscNodePtr &node, af::AscNodePtr &input_node) {
  input_node = nullptr;
  GE_ASSERT_NOTNULL(node);
  auto node_in_anchor = node->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(node_in_anchor);
  auto peer_out_anchor = node_in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(peer_out_anchor);
  input_node = std::dynamic_pointer_cast<af::AscNode>(peer_out_anchor->GetOwnerNode());
  GE_ASSERT_NOTNULL(input_node);
  return af::SUCCESS;
}

af::Status FillReduceReuseSource(const af::AscNodePtr &node, codegen::ReduceReuseInfo &reuse) {
  af::AscNodePtr input_node;
  GE_ASSERT_SUCCESS(GetFirstInputOwnerNode(node, input_node));
  reuse.valid = true;
  reuse.is_reuse_source = input_node->GetOutAllNodes().size() == 1UL;
  return af::SUCCESS;
}

af::Status BuildReduceInput(const AscirParamBuildContext &ctx, codegen::ReduceSpecificParamBuildInput &input) {
  GE_ASSERT_TRUE(!ctx.inputs.empty(), "Reduce input tensor is empty, node[%s].", ctx.api_name.c_str());
  GE_ASSERT_TRUE(!ctx.outputs.empty(), "Reduce output tensor is empty, node[%s].", ctx.api_name.c_str());
  GE_ASSERT_TRUE(IsValidTensorView(ctx.inputs[0]), "Reduce input tensor is invalid, node[%s].", ctx.api_name.c_str());
  GE_ASSERT_TRUE(IsValidTensorView(ctx.outputs[0]), "Reduce output tensor is invalid, node[%s].", ctx.api_name.c_str());
  const auto axis_map = BuildAxisMap(ctx.loop_axes);
  std::set<int64_t> visited;
  GE_ASSERT_TRUE(HasAxisClosure(axis_map, ctx.loop_axis_id, visited), "Reduce loop axis closure is invalid, node[%s].",
                 ctx.api_name.c_str());
  input.node_name = ctx.node == nullptr ? ctx.api_name : ctx.node->GetName();
  input.reduce_type = ctx.api_name;
  input.input_repeats = ctx.inputs[0].vectorized_repeats;
  input.input_strides = ctx.inputs[0].vectorized_strides;
  input.output_dims = ctx.outputs[0].vectorized_repeats;
  input.output_strides = ctx.outputs[0].vectorized_strides;
  input.dtype_size = ctx.inputs[0].dtype_size;
  input.pattern = GetReducePattern(ctx);
  input.need_multi_reduce = IsNeedMultiReduce(ctx);
  input.merge_times = input.need_multi_reduce ? GetReduceMergeTimes(ctx) : ge::Symbol(1U);
  if (ctx.node != nullptr) {
    GE_ASSERT_SUCCESS(FillReduceReuseSource(ctx.node, input.reuse));
  }
  return af::SUCCESS;
}

af::Status BuildReduceNodeParams(const AscirParamBuildContext &ctx, AscirNodeParams &params) {
  params = AscirNodeParams{};
  params.api_name = ctx.api_name;
  codegen::ReduceSpecificParamBuildInput input;
  GE_ASSERT_SUCCESS(BuildReduceInput(ctx, input));
  ReduceNodeParams reduce_params;
  GE_ASSERT_SUCCESS(codegen::BuildReduceSpecificParams(input, reduce_params.canonical_params));
  reduce_params.exprs.merge_size = BuildMergeSizeExpr(ctx);
  reduce_params.exprs.merge_times = BuildMergeTimesExpr(ctx, input.need_multi_reduce);
  if (reduce_params.canonical_params.valid) {
    GE_ASSERT_SUCCESS(ValidateReduceNodeParams(reduce_params));
  }
  params.status = reduce_params.canonical_params.valid ? ParamBuildStatus::kBuilt : ParamBuildStatus::kInvalid;
  params.specific_params = reduce_params;
  return af::SUCCESS;
}

af::Status BuildAscirParamContext(const AscirParamSourceContext &source, AscirParamBuildContext &ctx) {
  GE_ASSERT_NOTNULL(source.node);
  GE_ASSERT_NOTNULL(source.graph);
  ctx = AscirParamBuildContext{};
  ctx.node = source.node;
  ctx.graph = source.graph;
  ctx.api_name = source.node->GetType();
  ctx.inputs = BuildInputTensorParamViews(source.node);
  ctx.outputs = BuildOutputTensorParamViews(source.node);
  ctx.loop_axis_id = source.node->attr.sched.loop_axis;
  GE_ASSERT_SUCCESS(BuildLoopAxisParamViews(source, ctx.loop_axes));
  return af::SUCCESS;
}

af::Status BuildAscirNodeParams(const AscirParamBuildContext &ctx, AscirNodeParams &params) {
  params = AscirNodeParams{};
  params.api_name = ctx.api_name;
  if (!IsReduceParamSupported(ctx.api_name)) {
    params.status = ParamBuildStatus::kSkipped;
    return af::SUCCESS;
  }
  return BuildReduceNodeParams(ctx, params);
}

af::Status BuildAndRegisterAscirNodeParams(const AscirParamBuildContext &ctx) {
  GE_ASSERT_NOTNULL(ctx.node);
  auto params = std::make_shared<AscirNodeParams>();
  GE_ASSERT_SUCCESS(BuildAscirNodeParams(ctx, *params));
  return RegisterAscirNodeParams(ctx.node, params);
}

af::Status EnrichAscirNodeParams(const AscirParamSourceContext &source) {
  GE_ASSERT_NOTNULL(source.node);
  if (source.node->GetType() == kVectorFunc) {
    return RegisterVectorFuncAscirNodeParams(source.node);
  }
  if (source.node->GetType() == kCast || source.node->GetType() == kCeil2Int) {
    return RegisterCastAscirNodeParams(source.node);
  }
  if (source.node->GetType() == kBroadcast) {
    return RegisterBroadcastAscirNodeParams(source.node);
  }
  if (IsCompareParamSupported(source.node->GetType())) {
    return RegisterCompareAscirNodeParams(source.node);
  }
  if (source.node->GetType() == "Isnan" || source.node->GetType() == "IsFinite" || source.node->GetType() == "IsInf") {
    return RegisterUnaryBitWidthChangeAscirNodeParams(source.node);
  }
  if (source.node->GetType() == "Transpose") {
    return RegisterTransposeAscirNodeParams(source.node);
  }
  if (!IsReduceParamSupported(source.node->GetType())) {
    return RegisterSkippedAscirNodeParams(source.node);
  }
  AscirParamBuildContext ctx;
  GE_ASSERT_SUCCESS(BuildAscirParamContext(source, ctx));
  return BuildAndRegisterAscirNodeParams(ctx);
}
}  // namespace

bool IsReduceParamSupported(const std::string &api_name) {
  static const std::set<std::string> kReduceTypes = {"Max",        "Min",       "Mean",      "Prod",       "Sum",
                                                     "Any",        "All",       "ReduceMax", "ReduceMean", "ReduceMin",
                                                     "ReduceProd", "ReduceSum", "ReduceAny", "ReduceAll"};
  return kReduceTypes.count(api_name) != 0U;
}

af::Status EnrichAscirGraphNodeParams(const af::AscGraph &graph) {
  for (const auto &node : graph.GetAllNodes()) {
    GE_ASSERT_SUCCESS(EnrichAscirNodeParams({node, &graph}));
  }
  return af::SUCCESS;
}

void ResetCacheExecutions(const af::AscNodePtr &node, uint32_t stage) {
  const auto params = GetAscirNodeParams(node);
  if (params == nullptr) {
    return;
  }
  auto &records = params->cache_executions;
  records.erase(
      std::remove_if(records.begin(), records.end(), [stage](const auto &record) { return record.stage == stage; }),
      records.end());
}

namespace {
CacheGuardInfo BuildCacheGuard(CacheGuardKind kind, const ascir::Axis &axis, const ascir::Axis &period,
                               const ge::Expression &block_inner_extent) {
  CacheGuardInfo guard;
  guard.kind = kind;
  guard.guard_axis = ge::Symbol(axis.name.c_str());
  guard.loop_extent = axis.size;
  if (kind == CacheGuardKind::kFusedBroadcast) {
    guard.period = period.size;
    guard.block_inner_extent = block_inner_extent;
  }
  guard.address_invariant = true;
  guard.positive_range_proven = true;
  return guard;
}
}  // namespace

void BindCacheCondition(CacheExecutionContext &context, ReduceCacheCondition condition, const ascir::Axis &axis,
                        const ascir::Axis &period, const ge::Expression &block_inner_extent) {
  auto guard = BuildCacheGuard(CacheGuardKind::kFusedBroadcast, axis, period, block_inner_extent);
  guard.clip_to_period = false;
  context.bindings[condition] = guard;
}

void BindCacheCondition(CacheExecutionContext &context, CacheExecutionMode mode, const ascir::Axis &axis,
                        const ascir::Axis &period, const ge::Expression &block_inner_extent) {
  const auto kind =
      mode == CacheExecutionMode::kOriginBroadcast ? CacheGuardKind::kOriginBroadcast : CacheGuardKind::kFusedBroadcast;
  context.broadcast_bindings[mode] = BuildCacheGuard(kind, axis, period, block_inner_extent);
}

af::Status CollectCacheExecution(const CacheExecutionContext &context, const af::AscNodePtr &node,
                                 const std::vector<ascir::AxisId> &loop_axes, ascir::AxisId call_axis,
                                 CacheExecutionMode mode, const std::vector<ReduceCacheCondition> &conditions) {
  if (node == nullptr) {
    return af::SUCCESS;
  }
  CacheExecutionParams execution;
  execution.mode = mode;
  execution.call_loop_axis_id = call_axis;
  execution.stage = context.stage;
  for (const auto id : loop_axes) {
    const auto axis = context.axes.find(id);
    GE_ASSERT_TRUE(axis != context.axes.end());
    if (axis->second->type != ascir::Axis::kAxisTypeBlockOuter) {
      execution.loops.push_back({id, ge::Symbol(axis->second->name.c_str()), axis->second->size});
    }
  }
  if (mode == CacheExecutionMode::kReduceDoubleTile) {
    for (const auto condition : conditions) {
      const auto binding = context.bindings.find(condition);
      if (binding == context.bindings.end()) {
        execution.mode = CacheExecutionMode::kUnknown;
      } else {
        execution.reduce_guards.emplace(*binding);
      }
    }
  } else if (mode == CacheExecutionMode::kOriginBroadcast || mode == CacheExecutionMode::kFusedBroadcast) {
    const auto binding = context.broadcast_bindings.find(mode);
    if (binding == context.broadcast_bindings.end()) {
      execution.mode = CacheExecutionMode::kUnknown;
    } else {
      execution.guard = binding->second;
    }
  }
  auto params = GetAscirNodeParams(node);
  if (params == nullptr) {
    params = GetOrCreateAscirNodeParams(node);
    GE_ASSERT_NOTNULL(params);
    // 单独生成 kernel 时可能没有 API 专用载荷，执行记录与其构建状态独立。
    params->status = ParamBuildStatus::kSkipped;
  }
  RecordCacheExecution(*params, execution);
  return af::SUCCESS;
}
}  // namespace ascir_param
