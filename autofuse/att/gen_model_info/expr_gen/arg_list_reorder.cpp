/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "arg_list_reorder.h"

#include <algorithm>
#include <limits>

namespace att {
namespace {
constexpr size_t kOrderIdStart = 1;
const std::vector<std::string> kDefaultNodeWhiteList = {"Data", "Output", "Constant", "Workspace", "TbufData"};

bool GetExprBytes(const Expr &repeat_expr, uint32_t data_type_size, uint64_t &axis_bytes) {
  if (data_type_size == 0U) {
    return false;
  }
  uint64_t repeat = 0UL;
  if (!repeat_expr.GetConstValue(repeat)) {
    return false;
  }
  if (repeat > std::numeric_limits<uint64_t>::max() / data_type_size) {
    return false;
  }
  axis_bytes = repeat * data_type_size;
  return true;
}

bool IsReduceOrigAxis(const SubAxis *axis, const std::set<std::string> &reduce_axis_ori_axes_set) {
  if (axis == nullptr) {
    return false;
  }
  if (reduce_axis_ori_axes_set.find(axis->name) != reduce_axis_ori_axes_set.end()) {
    return true;
  }
  for (const auto &orig_axis : axis->orig_axis_name) {
    if (reduce_axis_ori_axes_set.find(orig_axis) != reduce_axis_ori_axes_set.end()) {
      return true;
    }
  }
  for (const auto *orig_axis : axis->orig_axis) {
    if ((orig_axis != nullptr) && (reduce_axis_ori_axes_set.find(orig_axis->name) != reduce_axis_ori_axes_set.end())) {
      return true;
    }
  }
  return false;
}

bool IsSameOrRelatedAxis(const SubAxis *lhs, const SubAxis *rhs) {
  if ((lhs == nullptr) || (rhs == nullptr)) {
    return false;
  }
  if ((lhs == rhs) || (lhs->name == rhs->name)) {
    return true;
  }
  for (const auto &lhs_orig_axis : lhs->orig_axis_name) {
    if (std::find(rhs->orig_axis_name.begin(), rhs->orig_axis_name.end(), lhs_orig_axis) != rhs->orig_axis_name.end()) {
      return true;
    }
  }
  return false;
}

bool TryGetTensorDataTypeSize(const TensorPtr &tensor, uint32_t &data_type_size) {
  if ((tensor == nullptr) || (tensor->data_type_size == 0U)) {
    return false;
  }
  data_type_size = tensor->data_type_size;
  return true;
}

bool IsDynamicExprBytes(const Expr &repeat_expr, uint32_t data_type_size) {
  if (data_type_size == 0U) {
    return false;
  }
  uint64_t axis_bytes = 0UL;
  return !GetExprBytes(repeat_expr, data_type_size, axis_bytes);
}

bool IsConstExpr(const Expr &expr) {
  uint64_t value = 0UL;
  return expr.GetConstValue(value);
}

uint32_t CeilDiv(uint32_t dividend, uint32_t divisor) {
  if (divisor == 0U) {
    return 0U;
  }
  return dividend / divisor + ((dividend % divisor) == 0U ? 0U : 1U);
}

Expr GetOriginalAxisRepeat(const SubAxis *axis) {
  if (axis == nullptr) {
    return Expr();
  }
  Expr original_repeat;
  bool has_original_repeat = false;
  for (const auto *orig_axis : axis->orig_axis) {
    if ((orig_axis != nullptr) && orig_axis->repeat.IsValid()) {
      original_repeat = has_original_repeat ? af::sym::Mul(original_repeat, orig_axis->repeat) : orig_axis->repeat;
      has_original_repeat = true;
    }
  }
  return has_original_repeat ? original_repeat : axis->repeat;
}
}  // namespace

// 初始化arglist的优先级连边图
// 初始化排序原则：父轴的优先级必须大于子轴，举例说明:
// z0t->z0tt, z0t是z0tt的父轴，存在约束z0tt<=z0t，如果子轴优先级高于父轴，
// 轴排序算法可能会求解失败(子轴过大导致父轴无切分空间，容易超UB)
// 因此这是一个功能性的问题，即使子轴是reduce/broadcast/innermost轴，其优先级依然不能高于父轴(elementwise)
af::Status ArgListReorder::InitArgListPriorityGraph(const vector<AttAxisPtr> &arg_list) {
  for (size_t i = 0; i < arg_list.size(); i++) {
    auto arg = arg_list[i];
    for (const auto &from_axis : arg->from_axis) {
      GE_ASSERT_TRUE(axis_name_2_id_map_.find(from_axis->name) != axis_name_2_id_map_.end(),
                     "from axis cannot be found in arg[%s]", arg->name.c_str());
      size_t from_axis_id = axis_name_2_id_map_[from_axis->name];
      GE_ASSERT_TRUE(graph_->AddEdge(from_axis_id + kOrderIdStart, i + kOrderIdStart),
                     "add edge from axis[%s] to arg[%s] failed", from_axis->name.c_str(), arg->name.c_str());
      GELOGI("add edge from axis[%s] to arg[%s] successfully", from_axis->name.c_str(), arg->name.c_str());
    }
  }
  return af::SUCCESS;
}

bool ArgListReorder::HandleProperty(const SubAxis *dim, att::ArgListReorder::AxisProperty property, bool is_reduce,
                                    bool is_broadcast) {
  if ((property == AxisProperty::kReduce && is_reduce) || (property == AxisProperty::kBroadcast && is_broadcast)) {
    if (property == AxisProperty::kReduce) {
      reduce_map_.insert({dim->name, true});
    } else {
      broadcast_map_.insert({dim->name, true});
    }
    return true;
  }
  return false;
}

// 判断是否是reduce轴(不存在keep_dims=false(输出吞轴)，schedule会补轴)，规则如下：
// 1.node的输入tensor中某根轴的repeat不为1，输出tensor对应轴的repeat为1，那么该轴为reduce轴
// 2.node的输入tensor中某根轴的stride不为0，输出tensor对应轴的stride为0，那么该轴为reduce轴
bool ArgListReorder::CheckAxisProperty(const SubAxis *dim, const Expr &repeat, const Expr &stride,
                                       const std::vector<TensorPtr> &output_tensors, AxisProperty property) {
  for (const auto &tensor : output_tensors) {
    for (size_t i = 0; i < tensor->dim_info.size(); i++) {
      auto dim_info = tensor->dim_info[i];
      auto output_repeat = tensor->repeat[i];
      auto output_stride = tensor->stride[i];
      if (dim_info->name != dim->name) {
        continue;
      }
      bool is_reduce = (repeat != af::sym::kSymbolOne && output_repeat == af::sym::kSymbolOne) ||
                       (stride != af::sym::kSymbolZero && output_stride == af::sym::kSymbolZero);
      bool is_broadcast = (output_repeat != af::sym::kSymbolOne && repeat == af::sym::kSymbolOne) ||
                          (output_stride != af::sym::kSymbolZero && stride == af::sym::kSymbolZero);
      GELOGD("Begin to handle property is_reduce[%d], is_broadcast[%d], property[%u] of dim[%s]", is_reduce,
             is_broadcast, property, dim->name.c_str());
      if (HandleProperty(dim, property, is_reduce, is_broadcast)) {
        return true;
      }
    }
  }
  return false;
}

// 判断是否是broadcast轴(不存在隐式broadcast(输入吞轴)，schedule会补轴)，规则如下：
// 1.node的输出tensor中某根轴的repeat不为1，输入tensor对应轴的repeat为1，那么该轴为broadcast轴
// 2.node的输出tensor中某根轴的stride不为0，输入tensor对应轴的stride为0，那么该轴为broadcast轴
bool ArgListReorder::CheckBroadcast(const SubAxis *dim, const Expr &repeat, const Expr &stride,
                                    const std::vector<TensorPtr> &output_tensors) {
  return CheckAxisProperty(dim, repeat, stride, output_tensors, AxisProperty::kBroadcast);
}

bool ArgListReorder::CheckReduce(const SubAxis *dim, const Expr &repeat, const Expr &stride,
                                 const std::vector<TensorPtr> &output_tensors) {
  return CheckAxisProperty(dim, repeat, stride, output_tensors, AxisProperty::kReduce);
}

uint32_t ArgListReorder::GetCacheLineSize() const {
  if ((tuning_space_ != nullptr) && (tuning_space_->tiling_schedule_config_table != nullptr)) {
    return tuning_space_->tiling_schedule_config_table->GetCacheLineSize();
  }
  return arch_param::kDefaultCacheLineSize;
}

uint32_t ArgListReorder::GetVectorLenSize() const {
  if ((tuning_space_ != nullptr) && (tuning_space_->tiling_schedule_config_table != nullptr)) {
    return tuning_space_->tiling_schedule_config_table->GetVectorLenSize();
  }
  return arch_param::kDefaultVectorLenSize;
}

bool ArgListReorder::GetReduceAxisDataTypeSize(const NodeInfo &node, const SubAxis *axis,
                                               const std::set<std::string> &reduce_axis_ori_axes_set,
                                               uint32_t &data_type_size) const {
  std::vector<TensorPtr> tensors;
  tensors.insert(tensors.end(), node.inputs.begin(), node.inputs.end());
  tensors.insert(tensors.end(), node.outputs.begin(), node.outputs.end());
  for (const auto &tensor : tensors) {
    if (!TryGetTensorDataTypeSize(tensor, data_type_size)) {
      continue;
    }
    for (const auto *dim : tensor->dim_info) {
      if (IsSameOrRelatedAxis(dim, axis)) {
        return true;
      }
    }
  }
  for (const auto &tensor : tensors) {
    if (!TryGetTensorDataTypeSize(tensor, data_type_size)) {
      continue;
    }
    for (const auto *dim : tensor->dim_info) {
      if (IsReduceOrigAxis(dim, reduce_axis_ori_axes_set)) {
        return true;
      }
    }
  }
  return false;
}

const SubAxis *ArgListReorder::FindReduceTileAxis(const std::set<std::string> &reduce_axis_ori_axes_set) const {
  const SubAxis *reduce_axis = nullptr;
  for (const auto &axis : tuning_space_->sub_axes) {
    if ((axis->axis_type != AxisPosition::INNER) || axis->is_bind_multi_core ||
        !IsReduceOrigAxis(axis.get(), reduce_axis_ori_axes_set)) {
      continue;
    }
    if (reduce_axis != nullptr) {
      GELOGI("[ATT][ReduceTailBalance] Multiple Reduce tile axes found, keep the original order.");
      return nullptr;
    }
    reduce_axis = axis.get();
  }
  return reduce_axis;
}

bool ArgListReorder::IsReduceInputTensor(const TensorPtr &tensor, const SubAxis *reduce_axis,
                                         const std::set<std::string> &reduce_axis_ori_axes_set) const {
  if ((tensor == nullptr) || (tensor->data_type_size == 0U)) {
    return false;
  }
  return std::any_of(tensor->dim_info.begin(), tensor->dim_info.end(),
                     [this, reduce_axis, &reduce_axis_ori_axes_set](const SubAxis *dim) {
                       return IsSameOrRelatedAxis(dim, reduce_axis) || IsReduceOrigAxis(dim, reduce_axis_ori_axes_set);
                     });
}

bool ArgListReorder::CollectReduceTailAxis(const TensorPtr &tensor, const SubAxis *reduce_axis,
                                           const std::set<std::string> &reduce_axis_ori_axes_set,
                                           const SubAxis *&tail_axis, uint32_t &data_type_size) const {
  if (!IsReduceInputTensor(tensor, reduce_axis, reduce_axis_ori_axes_set)) {
    return true;
  }
  for (const auto *dim : tensor->dim_info) {
    if ((dim == nullptr) || !dim->is_node_innerest_dim || IsSameOrRelatedAxis(dim, reduce_axis)) {
      continue;
    }
    if ((tail_axis != nullptr) && !IsSameOrRelatedAxis(tail_axis, dim)) {
      GELOGI("[ATT][ReduceTailBalance] Multiple tail tile axes found, keep the original order.");
      return false;
    }
    if ((data_type_size != 0U) && (data_type_size != tensor->data_type_size)) {
      GELOGI("[ATT][ReduceTailBalance] Inconsistent input dtype sizes found, keep the original order.");
      return false;
    }
    tail_axis = dim;
    data_type_size = tensor->data_type_size;
  }
  return true;
}

bool ArgListReorder::IsReduceTailTileAxis(const SubAxis *tail_axis) const {
  if (tail_axis == nullptr) {
    return false;
  }
  return std::any_of(
      tuning_space_->sub_axes.begin(), tuning_space_->sub_axes.end(), [&tail_axis](const SubAxisPtr &axis) {
        return (axis->name == tail_axis->name) && (axis->axis_type == AxisPosition::INNER) && !axis->is_bind_multi_core;
      });
}

bool ArgListReorder::TryGetReduceTailTileInfo(const NodeInfo &node,
                                              const std::set<std::string> &reduce_axis_ori_axes_set,
                                              ReduceTailTileInfo &tile_info) const {
  const SubAxis *reduce_axis = FindReduceTileAxis(reduce_axis_ori_axes_set);
  if (reduce_axis == nullptr) {
    return false;
  }
  const SubAxis *tail_axis = nullptr;
  uint32_t data_type_size = 0U;
  for (const auto &tensor : node.inputs) {
    if (!CollectReduceTailAxis(tensor, reduce_axis, reduce_axis_ori_axes_set, tail_axis, data_type_size)) {
      return false;
    }
  }
  if ((tail_axis == nullptr) || (data_type_size == 0U) || !IsReduceTailTileAxis(tail_axis)) {
    return false;
  }

  tile_info.reduce_axis_name = reduce_axis->name;
  tile_info.tail_axis_name = tail_axis->name;
  tile_info.reduce_repeat = GetOriginalAxisRepeat(reduce_axis);
  tile_info.tail_repeat = GetOriginalAxisRepeat(tail_axis);
  tile_info.data_type_size = data_type_size;
  return true;
}

ArgListReorder::ReduceTailTilePolicy ArgListReorder::ClassifyReduceTailTilePolicy(
    const ReduceTailTileInfo &tile_info) const {
  uint64_t reduce_bytes = 0UL;
  uint64_t tail_bytes = 0UL;
  if (!GetExprBytes(tile_info.reduce_repeat, tile_info.data_type_size, reduce_bytes) ||
      !GetExprBytes(tile_info.tail_repeat, tile_info.data_type_size, tail_bytes)) {
    return ReduceTailTilePolicy::kFallback;
  }
  const uint32_t vector_len_size = GetVectorLenSize();
  const uint32_t cache_line_size = GetCacheLineSize();
  if ((vector_len_size == 0U) || (cache_line_size == 0U)) {
    return ReduceTailTilePolicy::kFallback;
  }
  if ((reduce_bytes > vector_len_size) && (tail_bytes < cache_line_size)) {
    return ReduceTailTilePolicy::kPreferTail;
  }
  if ((reduce_bytes < vector_len_size) && (tail_bytes > cache_line_size)) {
    return ReduceTailTilePolicy::kKeepDefault;
  }
  return ReduceTailTilePolicy::kEqual;
}

bool ArgListReorder::IsReduceAxisBlockSplit(const std::vector<SubAxisPtr> &all_axes,
                                            const std::set<std::string> &reduce_axis_ori_axes_set) const {
  for (const auto &axis : all_axes) {
    if (!axis->is_bind_multi_core || !IsReduceOrigAxis(axis.get(), reduce_axis_ori_axes_set)) {
      continue;
    }
    GELOGI("reduce axis[%s] bind multi core", axis->name.c_str());
    return true;
  }
  return false;
}

bool ArgListReorder::IsReduceAxisTileSplit(const std::set<std::string> &reduce_axis_ori_axes_set) const {
  for (const auto &axis : tuning_space_->sub_axes) {
    if ((axis->axis_type != AxisPosition::INNER) || axis->is_bind_multi_core ||
        !IsReduceOrigAxis(axis.get(), reduce_axis_ori_axes_set)) {
      continue;
    }
    return true;
  }
  return false;
}

bool ArgListReorder::TryBuildReduceTileRuntimeReorderRule(const NodeInfo &node,
                                                          const std::set<std::string> &reduce_axis_ori_axes_set,
                                                          RuntimeReorderRule &rule) const {
  for (const auto &axis : tuning_space_->sub_axes) {
    if ((axis->axis_type != AxisPosition::INNER) || axis->is_bind_multi_core ||
        !IsReduceOrigAxis(axis.get(), reduce_axis_ori_axes_set)) {
      continue;
    }
    uint32_t data_type_size = 0U;
    if (!GetReduceAxisDataTypeSize(node, axis.get(), reduce_axis_ori_axes_set, data_type_size)) {
      continue;
    }
    rule.compare_axis = GetOriginalAxisRepeat(axis.get());
    rule.compare_threshold = GetVectorLenSize() / data_type_size;
    return true;
  }
  return false;
}

bool ArgListReorder::HasSmallTailLargeReduceTile(const NodeInfo &node,
                                                 const std::set<std::string> &reduce_axis_ori_axes_set,
                                                 RuntimeReorderRule &rule) const {
  if (!TryBuildReduceTileRuntimeReorderRule(node, reduce_axis_ori_axes_set, rule)) {
    return false;
  }
  bool has_dynamic = false;
  for (const auto &tensor : node.inputs) {
    if (tensor == nullptr) {
      continue;
    }
    const size_t dim_size = std::min(tensor->dim_info.size(), tensor->repeat.size());
    for (size_t i = 0U; i < dim_size; ++i) {
      const auto *dim = tensor->dim_info[i];
      if ((dim == nullptr) || !dim->is_node_innerest_dim) {
        continue;
      }
      if (tensor->data_type_size == 0U) {
        continue;
      }
      rule.condition_axis = GetOriginalAxisRepeat(dim);
      rule.condition_threshold = CeilDiv(GetCacheLineSize(), tensor->data_type_size);
      uint64_t tail_bytes = 0UL;
      uint64_t reduce_bytes = 0UL;
      const bool has_static_tail = GetExprBytes(rule.condition_axis, tensor->data_type_size, tail_bytes);
      const bool has_static_reduce = GetExprBytes(rule.compare_axis, tensor->data_type_size, reduce_bytes);
      const bool small_tail = has_static_tail && (tail_bytes < GetCacheLineSize());
      const bool large_reduce = has_static_reduce && (reduce_bytes > GetVectorLenSize());
      if ((has_static_tail && !small_tail) || (has_static_reduce && !large_reduce)) {
        continue;
      }
      has_dynamic = has_dynamic || IsDynamicExprBytes(rule.condition_axis, tensor->data_type_size) ||
                    IsDynamicExprBytes(rule.compare_axis, tensor->data_type_size);
      if (small_tail && large_reduce) {
        return true;
      }
    }
  }
  return has_dynamic;
}

void ArgListReorder::RecordReduceTileTemplateSelection(const NodeInfo &node,
                                                       const std::set<std::string> &reduce_axis_ori_axes_set) {
  if (!IsReduceAxisTileSplit(reduce_axis_ori_axes_set)) {
    return;
  }
  ReduceTailTileInfo tile_info;
  if (TryGetReduceTailTileInfo(node, reduce_axis_ori_axes_set, tile_info)) {
    const ReduceTailTilePolicy policy = ClassifyReduceTailTilePolicy(tile_info);
    if (policy == ReduceTailTilePolicy::kEqual) {
      equal_order_reduce_tail_axes_.insert(tile_info.reduce_axis_name);
      equal_order_reduce_tail_axes_.insert(tile_info.tail_axis_name);
      GELOGI("[ATT][ReduceTailBalance] Set Reduce axis [%s] and tail axis [%s] to equal order.",
             tile_info.reduce_axis_name.c_str(), tile_info.tail_axis_name.c_str());
      return;
    }
    if (policy == ReduceTailTilePolicy::kPreferTail) {
      prefer_reduce_tile_ = true;
      GELOGI("[ATT][ReduceTailBalance] Prefer tail axis [%s] before Reduce axis [%s].",
             tile_info.tail_axis_name.c_str(), tile_info.reduce_axis_name.c_str());
      return;
    }
    if (policy == ReduceTailTilePolicy::kKeepDefault) {
      GELOGI("[ATT][ReduceTailBalance] Keep existing Reduce-first order for axis [%s] and tail axis [%s].",
             tile_info.reduce_axis_name.c_str(), tile_info.tail_axis_name.c_str());
      return;
    }
  }

  RuntimeReorderRule runtime_rule;
  if (!HasSmallTailLargeReduceTile(node, reduce_axis_ori_axes_set, runtime_rule)) {
    GELOGI(
        "[ATT][ReduceTileReorder] Static reduce tile selects fallback single template: keep reduce axis before tail "
        "axis, prefer splitting tail axis.");
    return;
  }
  const bool is_static = IsConstExpr(runtime_rule.condition_axis) && IsConstExpr(runtime_rule.compare_axis);
  if (is_static) {
    GELOGI(
        "[ATT][ReduceTileReorder] Static reduce tile selects preferred single template: keep tail axis before reduce "
        "axis, prefer splitting reduce axis.");
    prefer_reduce_tile_ = true;
    return;
  }
  GELOGI(
      "[ATT][ReduceTileReorder] Dynamic reduce tile selects runtime single template: condition axis %s < %u and "
      "compare axis %s > %u chooses preferred order at runtime.",
      Str(runtime_rule.condition_axis).c_str(), runtime_rule.condition_threshold,
      Str(runtime_rule.compare_axis).c_str(), runtime_rule.compare_threshold);
  has_dynamic_reduce_tile_reorder_ = true;
  dynamic_reduce_tile_reorder_rule_ = runtime_rule;
}

void ArgListReorder::SaveReduceAxisOrig(const SubAxis *reduce_axis,
                                        std::set<std::string> &reduce_axis_ori_axes_set) const {
  for (SubAxis *ori_axis : reduce_axis->orig_axis) {
    GELOGI("reduce axis ori set add: [%s]", ori_axis->name.c_str());
    reduce_axis_ori_axes_set.insert(ori_axis->name);
  }
}

void ArgListReorder::RecordSpecialArgs(const NodeInfo &node, const TensorPtr &tensor, size_t id,
                                       const std::vector<TensorPtr> &output_tensors,
                                       std::set<std::string> &reduce_axis_ori_axes_set) {
  auto dim = tensor->dim_info[id];
  auto repeat = tensor->repeat[id];
  auto stride = tensor->stride[id];
  if (CheckReduce(dim, repeat, stride, output_tensors)) {
    GELOGD("find reduce axis %s from node %s", dim->name.c_str(), node.name.c_str());
    SaveReduceAxisOrig(dim, reduce_axis_ori_axes_set);
    reduce_map_.insert({dim->name, true});
  }
  if (CheckBroadcast(dim, repeat, stride, output_tensors)) {
    GELOGD("find broadcast axis %s from node %s", dim->name.c_str(), node.name.c_str());
    broadcast_map_.insert({dim->name, true});
  }
  if (dim->is_node_innerest_dim) {
    GELOGD("find innermost dim axis %s from node %s", dim->name.c_str(), node.name.c_str());
    innermost_dim_map_.insert({dim->name, true});
  }
}

// 查找特殊轴，目前仅支持reduce/broadcast和最内轴，后续如果有其余特殊轴在这里扩展
void ArgListReorder::FindSpecialArgs() {
  for (const auto &node : tuning_space_->node_infos) {
    if (std::find(kDefaultNodeWhiteList.begin(), kDefaultNodeWhiteList.end(), node.node_type) !=
        kDefaultNodeWhiteList.end()) {
      continue;
    }

    const auto &input_tensors = node.inputs;
    const auto &output_tensors = node.outputs;
    std::set<std::string> reduce_axis_ori_axes_set;

    for (const auto &tensor : input_tensors) {
      for (size_t i = 0; i < tensor->dim_info.size(); i++) {
        RecordSpecialArgs(node, tensor, i, output_tensors, reduce_axis_ori_axes_set);
      }
      auto asc_node = node.node_ptr;
      auto graph = tuning_space_->asc_graph;
      std::string last_dim_name;
      if ((asc_node != nullptr) && (graph != nullptr) && AttUtils::IsLoadStoreNode(asc_node.get()) &&
          (AttUtils::GetLastTileSplitAxisName(*asc_node, *graph, last_dim_name))) {
        load_store_inner_most_dims_.insert(last_dim_name);
        GELOGD("[DFX]Found Tile split axis %s in load/store node", last_dim_name.c_str());
      }
    }
    const bool is_reduce_block_split = IsReduceAxisBlockSplit(tuning_space_->sub_axes, reduce_axis_ori_axes_set);
    tiling_R_ = tiling_R_ || is_reduce_block_split;
    if (is_reduce_block_split) {
      continue;
    }
    RecordReduceTileTemplateSelection(node, reduce_axis_ori_axes_set);
  }
}

// from_axes里的每根轴连边到to_axes里的每根轴
// 举例，假设from_axes里的轴都是reduce轴，to_axes里的轴都是非reduce轴，那么连边表示reduce轴优先级高于非reduce轴
af::Status ArgListReorder::AddEdgeGroups(const std::vector<std::string> &from_axes_group,
                                         const std::vector<std::string> &to_axes_group) {
  for (const auto &from_axis : from_axes_group) {
    GE_ASSERT_TRUE(axis_name_2_id_map_.find(from_axis) != axis_name_2_id_map_.end(),
                   "from axis[%s] cannot be found in arg", from_axis.c_str());
    size_t from_axis_id = axis_name_2_id_map_[from_axis];

    for (const auto &to_axis : to_axes_group) {
      GE_ASSERT_TRUE(axis_name_2_id_map_.find(to_axis) != axis_name_2_id_map_.end(),
                     "to axis[%s] cannot be found in arg", to_axis.c_str());
      size_t to_axis_id = axis_name_2_id_map_[to_axis];
      GE_ASSERT_TRUE(graph_->AddEdge(from_axis_id + kOrderIdStart, to_axis_id + kOrderIdStart),
                     "add edge from axis[%s] to axis[%s] failed", from_axis.c_str(), to_axis.c_str());
      GELOGI("add edge from axis[%s] to axis[%s] successfully", from_axis.c_str(), to_axis.c_str());
    }
  }
  return af::SUCCESS;
}

// 构建ArgList的优先级的连边图
// 优先级遵循的原则（依次排序）
// 1.父轴高于子轴
// 2.reduce轴高于非reduce轴
// 3.broadcast轴高于非broadcast轴（和reduce轴优先级孰高孰低可能存在争议，暂定reduce更高）
// 4.最内轴高于非最内轴
// 举例说明
// 假设[z0,z1].merge-> z0z1, z0z1.split->[z0z1T, z0z1t], z2.split->[z2T, z2t]
// 其中z0z1t是reduce轴，且z2t是最内轴，优先级图如下：
// z0  z1
//  \  /
//  z0z1
//   |
//  z0z1t  z2
//   \     /
//     z2t
// 拓扑排序结果为[z0, z1, z0z1, z0z1t, z2, z2t]
// 满足规则1,2,3,4
// 这里有一个冲突点，z0z1t是reduce轴，z2t是非reduce轴以及最内轴，但是连边的时候优先连的是reduce轴
// 即z0z1t->z2t, 在连最内轴的group到非最内轴的group的时候，会尝试将z2t->z0z1t，这里似乎冲突，但是图数据结构中
// 连边操作会判环，如果z0z1t->z2t, z2t->z0z1t会成环，因此不会连z2t->z0z1t，这样也是遵循reduce轴优先级高于最内轴这条原则
ArgListReorder::AxisCategories ArgListReorder::CategorizeAxesByProperty(const vector<AttAxisPtr> &arg_list) {
  AxisCategories categories;
  for (const auto &arg : arg_list) {
    if (reduce_map_.find(arg->name) != reduce_map_.end()) {
      categories.reduce_arg_names.push_back(arg->name);
    } else {
      categories.non_reduce_arg_names.push_back(arg->name);
    }

    if (broadcast_map_.find(arg->name) != broadcast_map_.end()) {
      categories.broadcast_arg_names.push_back(arg->name);
    } else {
      categories.non_broadcast_arg_names.push_back(arg->name);
    }

    if (innermost_dim_map_.find(arg->name) != innermost_dim_map_.end()) {
      categories.innermost_dim_arg_names.push_back(arg->name);
    } else {
      categories.non_innermost_dim_arg_names.push_back(arg->name);
    }
  }
  return categories;
}

af::Status ArgListReorder::ApplyPriorityRules(PriorityPolicy policy, const AxisCategories &categories) {
  if (policy == PriorityPolicy::kNonReduceFirst) {
    // 父子依赖已先连边；非 reduce 优先边必须早于 innermost 边，避免反向边成环后被忽略。
    GE_ASSERT_SUCCESS(AddEdgeGroups(categories.non_reduce_arg_names, categories.reduce_arg_names),
                      "add edge from non reduce to reduce failed");
  }
  if (policy == PriorityPolicy::kReduceFirst) {
    GE_ASSERT_SUCCESS(AddEdgeGroups(categories.reduce_arg_names, categories.non_reduce_arg_names),
                      "add edge from reduce to non reduce failed");
    GE_ASSERT_SUCCESS(AddEdgeGroups(categories.broadcast_arg_names, categories.non_broadcast_arg_names),
                      "add edge from broadcast to non broadcast failed");
    GE_ASSERT_SUCCESS(AddEdgeGroups(categories.innermost_dim_arg_names, categories.non_innermost_dim_arg_names),
                      "add edge from innermost dim to non innermost dim failed");
  } else {
    GE_ASSERT_SUCCESS(AddEdgeGroups(categories.innermost_dim_arg_names, categories.non_innermost_dim_arg_names),
                      "add edge from innermost dim to non innermost dim failed");
    GE_ASSERT_SUCCESS(AddEdgeGroups(categories.broadcast_arg_names, categories.non_broadcast_arg_names),
                      "add edge from broadcast to non broadcast failed");
    GE_ASSERT_SUCCESS(AddEdgeGroups(categories.reduce_arg_names, categories.non_reduce_arg_names),
                      "add edge from reduce to non reduce failed");
  }
  return af::SUCCESS;
}

// 构建ArgList的优先级的连边图
// 优先级遵循的原则（依次排序）
// 1.父轴高于子轴
// 2.reduce轴高于非reduce轴
// 3.broadcast轴高于非broadcast轴（和reduce轴优先级孰高孰低可能存在争议，暂定reduce更高）
// 4.最内轴高于非最内轴
// 举例说明
// 假设[z0,z1].merge-> z0z1, z0z1.split->[z0z1T, z0z1t], z2.split->[z2T, z2t]
// 其中z0z1t是reduce轴，且z2t是最内轴，优先级图如下：
// z0  z1
//  \  /
//  z0z1
//   |
//  z0z1t  z2
//   \     /
//     z2t
// 拓扑排序结果为[z0, z1, z0z1, z0z1t, z2, z2t]
// 满足规则1,2,3,4
// 这里有一个冲突点，z0z1t是reduce轴，z2t是非reduce轴以及最内轴，但是连边的时候优先连的是reduce轴
// 即z0z1t->z2t, 在连最内轴的group到非最内轴的group的时候，会尝试将z2t->z0z1t，这里似乎冲突，但是图数据结构中
// 连边操作会判环，如果z0z1t->z2t, z2t->z0z1t会成环，因此不会连z2t->z0z1t，这样也是遵循reduce轴优先级高于最内轴这条原则
af::Status ArgListReorder::BuildArgListPriorityGraph(const vector<AttAxisPtr> &arg_list, PriorityPolicy policy) {
  GE_ASSERT_SUCCESS(InitArgListPriorityGraph(arg_list), "init arg list graph failed");
  return ApplyPriorityRules(policy, CategorizeAxesByProperty(arg_list));
}

af::Status ArgListReorder::BuildTilingRArgList(const vector<AttAxisPtr> &arg_list,
                                               const std::vector<size_t> &canonical_order,
                                               vector<AttAxisPtr> &tiling_R_arg_list) {
  graph_ = af::MakeShared<ArgPriorityGraph>(arg_list.size());
  GE_ASSERT_NOTNULL(graph_, "Create tiling_R graph failed.");
  GE_ASSERT_SUCCESS(BuildArgListPriorityGraph(arg_list, PriorityPolicy::kInnermostFirst),
                    "build tiling_R arg list graph failed");
  const auto canonical_tile_order = GetTileSplitOrder(canonical_order, arg_list);
  auto tiling_r_order = graph_->TopologicalSort();
  // 原始轴等位置变化不代表 local buffer 求解顺序发生变化。
  if (GetTileSplitOrder(tiling_r_order, arg_list) == canonical_tile_order) {
    graph_ = af::MakeShared<ArgPriorityGraph>(arg_list.size());
    GE_ASSERT_NOTNULL(graph_, "Create non reduce first graph failed.");
    GE_ASSERT_SUCCESS(BuildArgListPriorityGraph(arg_list, PriorityPolicy::kNonReduceFirst),
                      "build non reduce first arg list graph failed");
    tiling_r_order = graph_->TopologicalSort();
  }
  tiling_R_arg_list.clear();
  if (GetTileSplitOrder(tiling_r_order, arg_list) != canonical_tile_order) {
    tiling_R_arg_list = GetNewArgList(tiling_r_order, arg_list);
  }
  return af::SUCCESS;
}

void ArgListReorder::SetAxesSameOrder(const std::vector<AttAxisPtr> &arg_list,
                                      const std::set<std::string> &axis_names) const {
  size_t min_order = SIZE_MAX;
  std::vector<AttAxisPtr> args_to_make_same_order;
  for (const auto &arg : arg_list) {
    const bool is_target = axis_names.find(arg->name) != axis_names.end();
    GELOGD("[DFX]arg name %s, axis_pos %d, order %d, bind_multicore %d, is_equal_order_target %d", arg->name.c_str(),
           arg->axis_pos, arg->order, arg->bind_multicore, is_target);
    if (is_target && !arg->bind_multicore && (arg->axis_pos == AxisPosition::INNER)) {
      args_to_make_same_order.emplace_back(arg);
      min_order = std::min(min_order, arg->order);
    }
  }
  for (auto &arg : args_to_make_same_order) {
    GELOGD("[DFX]Set arg %s order to %zu", arg->name.c_str(), min_order);
    arg->order = min_order;
  }
}

void ArgListReorder::MakeSureLoadStoreInnerestSameOrder(const std::vector<AttAxisPtr> &arg_list) const {
  // 处理 Load/Store 与 Reduce/tail 的同等切分优先级。
  constexpr size_t kReduceTailAxisCount = 2U;
  if (equal_order_reduce_tail_axes_.size() != kReduceTailAxisCount) {
    SetAxesSameOrder(arg_list, load_store_inner_most_dims_);
    return;
  }
  size_t reduce_tail_axis_count = 0U;
  for (const auto &arg : arg_list) {
    if ((equal_order_reduce_tail_axes_.find(arg->name) != equal_order_reduce_tail_axes_.end()) &&
        AttUtils::IsTileSplitAxis(arg)) {
      ++reduce_tail_axis_count;
    }
  }
  if (reduce_tail_axis_count != kReduceTailAxisCount) {
    GELOGI("[ATT][ReduceTailBalance] Reduce/tail tile axes are incomplete, keep the original order.");
    SetAxesSameOrder(arg_list, load_store_inner_most_dims_);
    return;
  }

  const bool overlaps_load_store = std::any_of(
      equal_order_reduce_tail_axes_.begin(), equal_order_reduce_tail_axes_.end(), [this](const std::string &name) {
        return load_store_inner_most_dims_.find(name) != load_store_inner_most_dims_.end();
      });
  if (overlaps_load_store) {
    std::set<std::string> combined_axes = load_store_inner_most_dims_;
    combined_axes.insert(equal_order_reduce_tail_axes_.begin(), equal_order_reduce_tail_axes_.end());
    if (combined_axes.size() > kReduceTailAxisCount) {
      GELOGI(
          "[ATT][ReduceTailBalance] Combined equal-order group has more than two axes, keep Transpose equal-order "
          "tiling and skip Reduce R/tail balance.");
      SetAxesSameOrder(arg_list, load_store_inner_most_dims_);
      return;
    }
    SetAxesSameOrder(arg_list, combined_axes);
  } else {
    SetAxesSameOrder(arg_list, load_store_inner_most_dims_);
    SetAxesSameOrder(arg_list, equal_order_reduce_tail_axes_);
  }
}

std::vector<AttAxisPtr> ArgListReorder::GetNewArgList(const std::vector<size_t> &topo_order,
                                                      const std::vector<AttAxisPtr> &arg_list) const {
  std::vector<AttAxisPtr> new_arg_list;
  for (auto &order_id : topo_order) {
    if ((order_id >= kOrderIdStart) && (order_id <= arg_list.size() + kOrderIdStart - 1)) {
      arg_list[order_id - kOrderIdStart]->order = order_id;
      new_arg_list.emplace_back(arg_list[order_id - kOrderIdStart]);
    }
  }

  GE_ASSERT_TRUE(new_arg_list.size() == arg_list.size(),
                 "arg list size is not equal after reorder, new_arg_list size[%zu] vs arg_list size[%zu]",
                 new_arg_list.size(), arg_list.size());
  MakeSureLoadStoreInnerestSameOrder(new_arg_list);
  GELOGI("After reorder, Arglist size is [%zu]", new_arg_list.size());
  for (size_t i = 0; i < new_arg_list.size(); i++) {
    auto &arg = new_arg_list[i];
    GELOGI("arg[%s], id is [%zu], order is [%zu]", arg->name.c_str(), i, arg->order);
  }
  return new_arg_list;
}

std::vector<Expr> ArgListReorder::GetTileSplitOrder(const std::vector<size_t> &topo_order,
                                                    const std::vector<AttAxisPtr> &arg_list) const {
  std::vector<Expr> order;
  for (const size_t order_id : topo_order) {
    if ((order_id < kOrderIdStart) || (order_id > arg_list.size() + kOrderIdStart - 1U)) {
      continue;
    }
    const auto &axis = arg_list[order_id - kOrderIdStart];
    if (AttUtils::IsTileSplitAxis(axis)) {
      order.emplace_back(axis->size->symbol_expr);
    }
  }
  return order;
}

bool ArgListReorder::SetRuntimePreferredOrder(const std::vector<AttAxisPtr> &canonical_arg_list,
                                              const std::vector<AttAxisPtr> &source_arg_list,
                                              const std::vector<size_t> &preferred_topo_order,
                                              RuntimeReorderRule &rule) const {
  std::vector<Expr> canonical_order;
  std::map<size_t, std::vector<Expr>> equal_order_groups;
  for (const auto &axis : canonical_arg_list) {
    if (!AttUtils::IsTileSplitAxis(axis)) {
      continue;
    }
    canonical_order.emplace_back(axis->size->symbol_expr);
    equal_order_groups[axis->order].emplace_back(axis->size->symbol_expr);
  }
  auto preferred_order = GetTileSplitOrder(preferred_topo_order, source_arg_list);
  for (const auto &group : equal_order_groups) {
    if (group.second.size() < 2U) {
      continue;
    }
    std::vector<size_t> positions;
    for (size_t i = 0U; i < preferred_order.size(); ++i) {
      if (std::find(group.second.begin(), group.second.end(), preferred_order[i]) != group.second.end()) {
        positions.emplace_back(i);
      }
    }
    if (positions.size() != group.second.size()) {
      return false;
    }
    for (size_t i = 0U; i < positions.size(); ++i) {
      preferred_order[positions[i]] = group.second[i];
    }
  }
  std::set<Expr, ExprCmp> canonical_set(canonical_order.begin(), canonical_order.end());
  std::set<Expr, ExprCmp> preferred_set(preferred_order.begin(), preferred_order.end());
  if ((canonical_order.size() != preferred_order.size()) || (canonical_set.size() != canonical_order.size()) ||
      (canonical_set != preferred_set)) {
    return false;
  }
  rule.preferred_order = std::move(preferred_order);
  return true;
}

// 排序的入口函数
af::Status ArgListReorder::SortArgList(vector<AttAxisPtr> &arg_list, vector<AttAxisPtr> &tiling_R_arg_list,
                                       std::vector<RuntimeReorderRule> *runtime_reorder_rules) {
  GE_ASSERT_TRUE(!arg_list.empty(), "arg list is empty");

  graph_ = af::MakeShared<ArgPriorityGraph>(arg_list.size());
  GE_ASSERT_NOTNULL(graph_, "Create graph failed");
  GELOGI("Before reorder ArgList:");
  for (size_t i = 0; i < arg_list.size(); i++) {
    auto &arg = arg_list[i];
    axis_name_2_id_map_[arg->name] = i;
    GELOGI("arg[%s], id is [%zu]", arg->name.c_str(), i);
  }
  FindSpecialArgs();
  // 构建最终优先级连边图
  const auto policy = prefer_reduce_tile_ ? PriorityPolicy::kInnermostFirst : PriorityPolicy::kReduceFirst;
  GE_ASSERT_SUCCESS(BuildArgListPriorityGraph(arg_list, policy), "build arg list graph failed");
  const auto canonical_order = graph_->TopologicalSort();
  std::vector<AttAxisPtr> new_arg_list = GetNewArgList(canonical_order, arg_list);
  if ((runtime_reorder_rules != nullptr) && has_dynamic_reduce_tile_reorder_) {
    graph_ = af::MakeShared<ArgPriorityGraph>(arg_list.size());
    GE_ASSERT_NOTNULL(graph_, "Create preferred graph failed");
    RuntimeReorderRule runtime_rule = dynamic_reduce_tile_reorder_rule_;
    if ((BuildArgListPriorityGraph(arg_list, PriorityPolicy::kInnermostFirst) == af::SUCCESS) &&
        SetRuntimePreferredOrder(new_arg_list, arg_list, graph_->TopologicalSort(), runtime_rule)) {
      runtime_reorder_rules->emplace_back(std::move(runtime_rule));
    } else {
      GELOGW("[ATT][ReduceTileReorder] Invalid preferred order, keep canonical order.");
    }
  }

  if (tiling_R_) {
    GELOGI("ReduceAxis BlockSplit, another reorder ArgList:");
    GE_ASSERT_SUCCESS(BuildTilingRArgList(arg_list, canonical_order, tiling_R_arg_list),
                      "build tiling_R arg list failed");
  }

  arg_list = new_arg_list;
  return af::SUCCESS;
}
}  // namespace att
