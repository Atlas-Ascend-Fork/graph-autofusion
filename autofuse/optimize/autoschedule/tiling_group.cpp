/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tiling_group.h"
#include "ascir_ops_utils.h"
#include "ascgraph_info_complete.h"
#include "ascir_utils.h"
#include "indirect_load_utils.h"
#include "schedule_utils.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace optimize::autoschedule {
namespace {
constexpr int64_t kDefaultGroup = -1;
void PrintGroup(std::stringstream &ss, const std::string &name, const std::vector<af::AxisId> &group) {
  ss << name << "[";
  for (auto axis_id : group) {
    ss << axis_id << ",";
  }
  ss << "],";
}
}  // namespace

std::string AxisGroup::ToString() const {
  std::stringstream ss;
  PrintGroup(ss, "XGroup:", x_group);
  PrintGroup(ss, "YGroup:", y_group);
  PrintGroup(ss, "RGroup:", r_group);
  PrintGroup(ss, "NGroup:", n_group);

  ss << "Order:[";
  for (auto &axis : axes_order) {
    ss << axis << ",";
  }
  ss << "]";
  return ss.str();
}

bool AxisGroup::IsEmpty() const {
  return x_group.empty() && y_group.empty() && r_group.empty() && n_group.empty();
}

extern "C" {
int32_t GenAscGraphAxisGroup(const af::AscGraph &graph, AxisGroup &axes_group) {
  GELOGD("Enter [GenAscGraphAxisGroup], graph name: %s, axis size: %zu", graph.GetName().c_str(),
         graph.GetAllAxis().size());
  for (auto &axis : graph.GetAllAxis()) {
    GE_CHECK_NOTNULL(axis);
    GELOGD("[GenAscGraphAxisGroup] AscGraph axis.name=%s, axis.id=%ld, axis.size=%s", axis->name.c_str(), axis->id,
           axis->size.Str().get());
  }
  GE_CHK_STATUS_RET(AscGraphInfoComplete::CompleteApiInfo(graph));

  GE_CHK_STATUS_RET(TilingGroup::GenTilingGroup(graph, axes_group), "Gen axis map failed, graph[%s] may be invalid.",
                    graph.GetName().c_str());
  GELOGD("Finish [GenAscGraphAxisGroup], graph name: %s", graph.GetName().c_str());
  return af::SUCCESS;
}

bool CanMergeAxisGroup(const AxisGroup &lhs, const AxisGroup &rhs, AxisGroup &merged_group, bool is_ge_call) {
  merged_group = lhs;
  AxisGroup new_rhs = rhs;
  auto ret = TilingGroup::MergeAxesGroup(merged_group, new_rhs, true, is_ge_call);
  GELOGI("Merged axis group: %s to target: %s, res:[%d].", rhs.ToString().c_str(), lhs.ToString().c_str(), ret);
  return ret;
}

static constexpr size_t kMaxFullLoadAxisSizeForNorm = 3UL;

static bool CalculateRAxisTotalSize(const af::AscTensorAttr &input_attr, const af::AscTensorAttr &output_attr,
                                    int64_t &r_axis_total_size, int64_t &a_axis_total_size, const char *node_name) {
  r_axis_total_size = 1;
  a_axis_total_size = 1;
  if (output_attr.repeats.empty() || output_attr.repeats.size() > kMaxFullLoadAxisSizeForNorm) {
    GELOGD("Output repeats size %zu exceeds max full load axis size %zu", output_attr.repeats.size(),
           kMaxFullLoadAxisSizeForNorm);
    return false;
  }

  std::string repeats_info;
  for (size_t i = 0; i < input_attr.repeats.size(); ++i) {
    int64_t in_val = 0;
    int64_t out_val = 0;
    (void)input_attr.repeats[i].GetConstValue(in_val);
    (void)output_attr.repeats[i].GetConstValue(out_val);
    repeats_info +=
        "axis[" + std::to_string(i) + "]: in=" + std::to_string(in_val) + " out=" + std::to_string(out_val) + "; ";
  }
  GELOGD("Node %s repeats: %s", node_name, repeats_info.c_str());

  bool first_valid_axis_checked = false;
  for (size_t i = 0; i < input_attr.repeats.size(); ++i) {
    int64_t input_size = 0;
    int64_t output_size = 0;
    (void)input_attr.repeats[i].GetConstValue(input_size);
    (void)output_attr.repeats[i].GetConstValue(output_size);

    if (!first_valid_axis_checked && (input_size > 1 || output_size > 1)) {
      first_valid_axis_checked = true;
      if (input_size != output_size) {
        GELOGD("First valid axis (index %zu) of input and output not equal, not AR/ARA pattern", i);
        return false;
      }
    }

    if (input_size > output_size) {
      r_axis_total_size *= input_size;
    } else {
      a_axis_total_size *= output_size;
    }
  }

  if (!first_valid_axis_checked) {
    GELOGD("No valid axis found (all repeats are 1), not AR/ARA pattern");
    return false;
  }

  return true;
}

static bool IsStaticShape(const af::AscNodePtr &node) {
  if (node == nullptr || node->outputs().empty()) {
    return false;
  }
  for (const auto &node_out : node->outputs()) {
    if (node_out->attr.repeats.empty()) {
      return false;
    }
    for (const auto &repeat : node_out->attr.repeats) {
      if (!repeat.IsConstExpr()) {
        return false;
      }
    }
  }
  return true;
}

static bool IsStaticGraph(const af::AscGraph &graph) {
  for (const auto &node : graph.GetAllNodes()) {
    if (!ScheduleUtils::IsLoad(node)) {
      continue;
    }
    if (!IsStaticShape(node)) {
      return false;
    }
  }
  GELOGD("Graph[%s] is static", graph.GetName().c_str());
  return true;
}

static bool CheckReduceNodeNormLike(const af::AscNodePtr &asc_node) {
  constexpr int64_t kThresholdTR = 32;
  constexpr int64_t kThresholdTA = 128;

  GELOGD("Found reduce node: %s", asc_node->GetNamePtr());

  if (asc_node->inputs().empty() || asc_node->outputs().empty()) {
    GELOGW("Reduce node %s has no inputs or outputs", asc_node->GetNamePtr());
    return false;
  }

  const af::AscTensorAttr *input_attr_ptr = nullptr;

  af::OutDataAnchorPtr current_out_anchor = nullptr;
  if (asc_node->GetAllInDataAnchorsSize() > 0 && asc_node->GetInDataAnchor(0) != nullptr) {
    current_out_anchor = asc_node->GetInDataAnchor(0)->GetPeerOutAnchor();
  }

  int32_t traverse_depth = 0;
  const int32_t max_traverse_depth = 100;

  while (current_out_anchor != nullptr && traverse_depth++ < max_traverse_depth) {
    auto current_tensor_attr = af::AscTensorAttr::GetTensorAttrPtr(*current_out_anchor);
    auto current_node = std::dynamic_pointer_cast<af::AscNode>(current_out_anchor->GetOwnerNode());

    bool is_load = (current_node != nullptr && ScheduleUtils::IsLoad(current_node));
    bool has_valid_attr = (current_tensor_attr != nullptr && !current_tensor_attr->repeats.empty());
    if (has_valid_attr || is_load) {
      if (has_valid_attr) {
        input_attr_ptr = current_tensor_attr;
      }
      break;
    }

    if (current_node == nullptr || current_node->GetAllInDataAnchorsSize() == 0 ||
        current_node->GetInDataAnchor(0) == nullptr) {
      return false;
    }

    auto next_in_anchor = current_node->GetInDataAnchor(0);
    current_out_anchor = next_in_anchor->GetPeerOutAnchor();
  }

  const auto &output = asc_node->outputs[0];
  if (input_attr_ptr == nullptr || input_attr_ptr->repeats.empty() ||
      input_attr_ptr->repeats.size() != output.attr.repeats.size()) {
    return false;
  }

  int64_t r_axis_total_size = 1;
  int64_t a_axis_total_size = 1;
  if (!CalculateRAxisTotalSize(*input_attr_ptr, output.attr, r_axis_total_size, a_axis_total_size,
                               asc_node->GetNamePtr())) {
    GELOGD("Reduce node %s: failed to calculate R/A axis size (non-const shape)", asc_node->GetNamePtr());
    return false;
  }

  if (r_axis_total_size > kThresholdTR || a_axis_total_size < kThresholdTA) {
    return false;
  }

  GELOGI("Reduce node %s passed check: R_axis=%ld (threshold=%ld), A_axis=%ld (threshold=%ld)", asc_node->GetNamePtr(),
         r_axis_total_size, kThresholdTR, a_axis_total_size, kThresholdTA);
  return true;
}

bool IsNormLikeReduceGraph(const af::AscGraph &graph) {
  if (!IsStaticGraph(graph)) {
    GELOGI("AscGraph is not static shape, return false for IsNormLikeReduceGraph");
    return false;
  }

  bool has_reduce = false;
  for (const auto &asc_node : graph.GetAllNodes()) {
    if (!ScheduleUtils::IsReduce(asc_node)) {
      continue;
    }

    has_reduce = true;
    if (!CheckReduceNodeNormLike(asc_node)) {
      return false;
    }
  }

  if (!has_reduce) {
    GELOGD("AscGraph has no reduce node, return false for IsNormLikeReduceGraph");
    return false;
  }

  GELOGD("AscGraph passed IsNormLikeReduceGraph check");
  return true;
}
}  // extern "C"

static bool CheckYAndYR(const std::vector<ascir::AxisId> &cur_y_group, const std::vector<ascir::AxisId> &new_y_group,
                        const std::vector<ascir::AxisId> &new_r_group, const std::vector<size_t> &new_axes_order,
                        const bool is_canfuse_call) {
  if (cur_y_group.size() != (new_y_group.size() + new_r_group.size())) {
    return false;
  }

  if (is_canfuse_call) {
    std::set<int64_t> l_g(cur_y_group.begin(), cur_y_group.end());
    std::set<int64_t> r_g(new_y_group.begin(), new_y_group.end());
    r_g.insert(new_r_group.begin(), new_r_group.end());
    return l_g == r_g;
  }

  for (size_t i = 0UL; i < new_y_group.size(); i++) {
    if (cur_y_group[new_axes_order[i]] != new_y_group[i]) {
      return false;
    }
  }
  size_t y_size = new_y_group.size();
  for (size_t i = 0UL; i < new_r_group.size(); i++) {
    if (cur_y_group[new_axes_order[y_size + i]] != new_r_group[i]) {
      return false;
    }
  }
  return true;
}

static bool CheckYAndR(const std::vector<ascir::AxisId> &cur_y_group, const std::vector<ascir::AxisId> &new_r_group,
                       const bool is_canfuse_call) {
  if (is_canfuse_call) {
    std::set<int64_t> l_g(cur_y_group.begin(), cur_y_group.end());
    std::set<int64_t> r_g(new_r_group.begin(), new_r_group.end());
    return l_g == r_g;
  }
  return cur_y_group == new_r_group;
}

static bool MergeYAndY(AxisGroup &lhs_group, AxisGroup &rhs_group, const bool is_canfuse_call, const bool is_ge_call) {
  (void)is_ge_call;
  if (lhs_group.y_group.size() != rhs_group.y_group.size()) {
    return false;
  }
  if (is_canfuse_call) {
    std::set<int64_t> lhs(lhs_group.y_group.begin(), lhs_group.y_group.end());
    std::set<int64_t> rhs(rhs_group.y_group.begin(), rhs_group.y_group.end());
    return lhs == rhs;
  }

  for (size_t i = 0UL; i < lhs_group.y_group.size(); i++) {
    if (lhs_group.y_group[i] != rhs_group.y_group[i]) {
      return false;
    }
  }
  return true;
}

static bool MergeYAndR(AxisGroup &lhs_group, AxisGroup &rhs_group, const bool is_canfuse_call, const bool is_ge_call) {
  (void)is_ge_call;
  if (!CheckYAndR(lhs_group.y_group, rhs_group.r_group, is_canfuse_call)) {
    return false;
  }
  lhs_group = rhs_group;
  return true;
}

static bool MergeRAndY(AxisGroup &lhs_group, AxisGroup &rhs_group, const bool is_canfuse_call, const bool is_ge_call) {
  (void)is_ge_call;
  return CheckYAndR(rhs_group.y_group, lhs_group.r_group, is_canfuse_call);
}

static bool MergeYAndXY(AxisGroup &lhs_group, AxisGroup &rhs_group, const bool is_canfuse_call, const bool is_ge_call) {
  (void)is_canfuse_call;
  (void)is_ge_call;
  std::set<int64_t> lhs(lhs_group.y_group.begin(), lhs_group.y_group.end());
  std::set<int64_t> rhs(rhs_group.y_group.begin(), rhs_group.y_group.end());
  rhs.insert(rhs_group.x_group.begin(), rhs_group.x_group.end());
  if (lhs != rhs) {
    return false;
  }
  lhs_group = rhs_group;
  return true;
}

static bool MergeXYAndY(AxisGroup &lhs_group, AxisGroup &rhs_group, const bool is_canfuse_call, const bool is_ge_call) {
  (void)is_canfuse_call;
  (void)is_ge_call;
  std::set<int64_t> lhs(lhs_group.y_group.begin(), lhs_group.y_group.end());
  lhs.insert(lhs_group.x_group.begin(), lhs_group.x_group.end());
  std::set<int64_t> rhs(rhs_group.y_group.begin(), rhs_group.y_group.end());
  if (lhs != rhs) {
    return false;
  }
  return true;
}

static bool MergeYAndYR(AxisGroup &lhs_group, AxisGroup &rhs_group, const bool is_canfuse_call, const bool is_ge_call) {
  (void)is_ge_call;
  if (!CheckYAndYR(lhs_group.y_group, rhs_group.y_group, rhs_group.r_group, rhs_group.axes_order, is_canfuse_call)) {
    return false;
  }
  lhs_group = rhs_group;
  return true;
}

static bool MergeYRAndY(AxisGroup &lhs_group, AxisGroup &rhs_group, const bool is_canfuse_call, const bool is_ge_call) {
  (void)is_ge_call;
  return CheckYAndYR(rhs_group.y_group, lhs_group.y_group, lhs_group.r_group, lhs_group.axes_order, is_canfuse_call);
}

static bool MergeYRAndYR(AxisGroup &lhs_group, AxisGroup &rhs_group, const bool is_canfuse_call,
                         const bool is_ge_call) {
  (void)is_ge_call;
  // y0 == y1, r0 == r1，可以做融合
  if (is_canfuse_call) {
    std::set<int64_t> l_y(lhs_group.y_group.begin(), lhs_group.y_group.end());
    std::set<int64_t> r_y(rhs_group.y_group.begin(), rhs_group.y_group.end());
    std::set<int64_t> l_r(lhs_group.r_group.begin(), lhs_group.r_group.end());
    std::set<int64_t> r_r(rhs_group.r_group.begin(), rhs_group.r_group.end());
    return (l_y == r_y) && (l_r == r_r);
  }
  return (lhs_group == rhs_group);
}

GroupType TilingGroup::GetGroupType(const AxisGroup &axes_group) {
  GroupType type = GroupType::GROUP_INVALID;
  if (!axes_group.x_group.empty()) {
    type = static_cast<GroupType>(type | GroupType::GROUP_X);
  }
  if (!axes_group.y_group.empty()) {
    type = static_cast<GroupType>(type | GroupType::GROUP_Y);
  }
  if (!axes_group.r_group.empty()) {
    type = static_cast<GroupType>(type | GroupType::GROUP_R);
  }
  return type;
}

static void RemoveNGroupAxisInXYRGroup(const std::set<af::AxisId> &n_groups, AxisGroup &single_node_axes_group) {
  if (n_groups.empty()) {
    return;
  }
  single_node_axes_group.n_group.assign(n_groups.begin(), n_groups.end());
  std::vector<size_t> new_axes_order;
  size_t idx = 0UL;
  for (auto iter = single_node_axes_group.x_group.begin(); iter != single_node_axes_group.x_group.end();) {
    if (n_groups.count(*iter) > 0UL) {
      iter = single_node_axes_group.x_group.erase(iter);
    } else {
      ++iter;
      new_axes_order.push_back(single_node_axes_group.axes_order[idx]);
    }
    ++idx;
  }

  for (auto iter = single_node_axes_group.y_group.begin(); iter != single_node_axes_group.y_group.end();) {
    if (n_groups.count(*iter) > 0UL) {
      iter = single_node_axes_group.y_group.erase(iter);
    } else {
      ++iter;
      new_axes_order.push_back(single_node_axes_group.axes_order[idx]);
    }
    ++idx;
  }

  for (auto iter = single_node_axes_group.r_group.begin(); iter != single_node_axes_group.r_group.end();) {
    if (n_groups.count(*iter) > 0UL) {
      iter = single_node_axes_group.r_group.erase(iter);
    } else {
      ++iter;
      new_axes_order.push_back(single_node_axes_group.axes_order[idx]);
    }
    ++idx;
  }

  single_node_axes_group.axes_order = new_axes_order;
}

static void MergeNGroup(AxisGroup &lhs_group, AxisGroup &rhs_group) {
  std::set<af::AxisId> n_groupset(lhs_group.n_group.begin(), lhs_group.n_group.end());
  n_groupset.insert(rhs_group.n_group.begin(), rhs_group.n_group.end());
  RemoveNGroupAxisInXYRGroup(n_groupset, lhs_group);
  RemoveNGroupAxisInXYRGroup(n_groupset, rhs_group);
}

// 在此之前不应该做任何的合轴，否则很难推断出轴关系
// 1. (1, y0, 1) merge (1, y1, 1) ==> (1, max(y0, y1), 1), 如果节点2是broadcast节点，要求节点1的tiling
// group与节点2的输入节点的tiling group一致，否则应该满足y0 == y1
// 2. (1, y0, 1) merge (1, y1, r1) ==> (1, y1, r1), 要求y0 == y1 U r1
// 3. (1, y0, 1) merge (x1, y1, 1) ==> (x1, y1, 1)
// 4. (1, y0, 1) merge (x1, y1, r1) ==> (x1, y1, r1)
// 5. (1, y0, r0) merge (1, y1, 1) ==> (1, y0, r0), 要求y1 == y0 U r0
// 6. (1, y0, r0) merge (1, y1, r1) ==> (1, y0, r0), 要求y1 == y0, r1 == r0
// 7. (1, y0, r0) merge (x1, y1, 1) ==> (x2, y2, r0)
// 8. (1, y0, r0) merge (x1, y1, r1) ==> (x1, y1, r1)
// 9. (x0, y0, 1) merge (1, y1, 1) ==> (x0, y0, 1)
// 10. (x0, y0, 1) merge (1, y1, r1) ==> (x2, y2, r1)
// 11. (x0, y0, 1) merge (x1, y1, 1) ==> (x0, y0, 1)
// 12. (x0, y0, 1) merge (x1, y1, r1) ==> (x1, y1, r1)
// 13. (x0, y0, r0) merge (1, y1, 1) ==> (x0, y0, r0)
// 14. (x0, y0, r0) merge (1, y1, r1) ==> (x0, y0, r0)
// 15. (x0, y0, r0) merge (x1, y1, 1) ==> (x0, y0, r0)
// 16. (x0, y0, r0) merge (x1, y1, r1) ==> (x0, y0, r0)
bool TilingGroup::MergeAxesGroup(AxisGroup &target, AxisGroup &src, const bool is_canfuse_call, const bool is_ge_call) {
  // 先单独处理Ngroup 再做其他group的合并, 否则当前多套轴下无法做merge
  MergeNGroup(target, src);
  if (target.x_group.empty() && target.y_group.empty() && target.r_group.empty()) {
    target = src;
    return true;
  }

  static std::map<std::pair<GroupType, GroupType>, AxisGroupMergeFunc> type_to_merge_func = {
      {{GroupType::GROUP_Y, GroupType::GROUP_Y}, MergeYAndY},
      {{GroupType::GROUP_Y, GroupType::GROUP_YR}, MergeYAndYR},
      {{GroupType::GROUP_YR, GroupType::GROUP_Y}, MergeYRAndY},
      {{GroupType::GROUP_YR, GroupType::GROUP_YR}, MergeYRAndYR},
      {{GroupType::GROUP_Y, GroupType::GROUP_XY}, MergeYAndXY},
      {{GroupType::GROUP_XY, GroupType::GROUP_Y}, MergeXYAndY},
      {{GroupType::GROUP_Y, GroupType::GROUP_R}, MergeYAndR},
      {{GroupType::GROUP_R, GroupType::GROUP_Y}, MergeRAndY},
  };

  auto merge_type = std::make_pair(GetGroupType(target), GetGroupType(src));
  auto iter = type_to_merge_func.find(merge_type);
  if (iter == type_to_merge_func.end()) {
    return false;
  }
  return iter->second(target, src, is_canfuse_call, is_ge_call);
}

Status TilingGroup::GenTilingGroup(const ascir::ImplGraph &impl_graph, AxisGroup &tiling_group,
                                   bool is_reduce_fullload) {
  std::vector<std::pair<std::string, AxisGroup>> node_name_to_tiling_group;
  std::set<af::AxisId> n_groupset;
  for (const auto &node : impl_graph.GetAllNodes()) {
    const auto behavior = ascgen_utils::indirect_load::GetTemplateBehavior(node);
    if (ScheduleUtils::IsBuffer(node) || behavior.skips_main_schedule_tiling || behavior.excludes_tiling_group) {
      continue;
    }
    AxisGroup single_node_axes_group;
    GE_CHK_STATUS_RET(GenAxisGroupForSingleNode(*node, single_node_axes_group, is_reduce_fullload));
    n_groupset.insert(single_node_axes_group.n_group.begin(), single_node_axes_group.n_group.end());
    node_name_to_tiling_group.emplace_back(node->GetName(), single_node_axes_group);
    GELOGD("GenTilingGroup node: %s, group: %s.", node->GetName().c_str(), single_node_axes_group.ToString().c_str());
  }
  tiling_group.n_group.assign(n_groupset.begin(), n_groupset.end());

  // merge xyr groups.
  for (auto &iter : node_name_to_tiling_group) {
    GE_ASSERT_TRUE(MergeAxesGroup(tiling_group, iter.second),
                   "Merged axis group: %s to target: %s failed, the graph [%s] cannot be fused.",
                   iter.second.ToString().c_str(), tiling_group.ToString().c_str(), impl_graph.GetName().c_str());
  }

  return af::SUCCESS;
}

Status TilingGroup::GenElewiseTilingGroup(af::AscNode &node, AxisGroup &axes_group) {
  GE_CHK_STATUS_RET(ScheduleUtils::GetLoopAxis(node, axes_group.y_group));
  axes_group.axes_order.reserve(axes_group.y_group.size());
  for (size_t i = 0UL; i < axes_group.y_group.size(); ++i) {
    axes_group.axes_order.push_back(i);
  }
  return af::SUCCESS;
}

Status TilingGroup::GenReduceTilingGroup(af::AscNode &node, AxisGroup &axes_group) {
  std::vector<ascir::AxisId> axes;
  GE_CHK_STATUS_RET(ScheduleUtils::GetLoopAxis(node, axes), "Get loop axis failed.");
  axes_group.axes_order.resize(axes.size());
  std::vector<ascir::SizeExpr> src_strides;
  GE_CHK_STATUS_RET(ScheduleUtils::GetReduceInputStrides(node, src_strides), "Get loop strides failed.");
  axes_group.r_group = ScheduleUtils::CalcReduceAxes(src_strides, node.outputs[0].attr.strides, axes);
  int64_t y_order_index = 0;
  int64_t r_order_index = axes.size() - axes_group.r_group.size();
  for (size_t i = 0; i < axes.size(); ++i) {
    if (std::find(axes_group.r_group.begin(), axes_group.r_group.end(), axes[i]) == axes_group.r_group.end()) {
      axes_group.y_group.push_back(axes[i]);
      axes_group.axes_order[y_order_index++] = i;
    } else {
      axes_group.axes_order[r_order_index++] = i;
    }
  }
  return af::SUCCESS;
}

Status TilingGroup::GenReduceTilingGroupFullLoad(af::AscNode &node, AxisGroup &axes_group) {
  std::vector<ascir::AxisId> axes;
  GE_CHK_STATUS_RET(ScheduleUtils::GetLoopAxis(node, axes), "Get loop axis failed.");
  axes_group.axes_order.resize(axes.size());

  if (node.GetType() == "Softmax") {
    if (!axes.empty()) {
      axes_group.n_group = {axes.back()};
    }
  } else {
    std::vector<ascir::SizeExpr> src_strides;
    GE_CHK_STATUS_RET(ScheduleUtils::GetReduceInputStrides(node, src_strides), "Get loop strides failed.");
    axes_group.n_group = ScheduleUtils::CalcReduceAxes(src_strides, node.outputs[0].attr.strides, axes);
  }

  int64_t y_order_index = 0;
  int64_t r_order_index = axes.size() - axes_group.n_group.size();
  for (size_t i = 0; i < axes.size(); ++i) {
    if (std::find(axes_group.n_group.begin(), axes_group.n_group.end(), axes[i]) == axes_group.n_group.end()) {
      axes_group.y_group.push_back(axes[i]);
      axes_group.axes_order[y_order_index++] = i;
    } else {
      axes_group.axes_order[r_order_index++] = i;
    }
  }
  return af::SUCCESS;
}

void PlaceRemainingAxis(const int64_t index, std::set<int64_t> &remaining_axis, AxisGroup &axes_group,
                        vector<ascir::AxisId> &output_axis) {
  for (int64_t j = index; j >= 0; --j) {
    if (remaining_axis.find(output_axis[j]) == remaining_axis.end()) {
      continue;
    }
    axes_group.y_group.emplace(axes_group.y_group.begin(), output_axis[j]);
  }
}

Status TilingGroup::GenTransposeTilingGroup(af::AscNode &node, AxisGroup &axes_group) {
  std::vector<ascir::AxisId> input_axis;
  GE_CHK_STATUS_RET(ScheduleUtils::GetInputForTranspose(node, input_axis), "Get Transpose loop axis failed.");
  std::vector<ascir::AxisId> &output_axis = node.outputs[0].attr.axis;
  GE_ASSERT_TRUE((input_axis.size() == output_axis.size()),
                 "The output dim cnt [%zu] of Transpose mismatch with input dim cnt [%zu].", output_axis.size(),
                 input_axis.size());
  std::set<int64_t> remaining_axis(input_axis.begin(), input_axis.end());
  GELOGD("GenTransposeTilingGroup input_axis %s, output_axis %s", af::ViewMemberToString(input_axis).c_str(),
         af::ViewMemberToString(output_axis).c_str());

  // 1. 从尾轴向前取轴，如果轴相同，分别放入x_group和y_group,直到轴不同为止
  int64_t i = static_cast<int64_t>(input_axis.size()) - 1;
  for (; i >= 0; --i) {
    // 如果input和output在该位置上的轴一致 => ngroup加入此轴, 当前att不能保证Xgroup和YGroup切同一根轴的结果一样
    if (input_axis[i] == output_axis[i]) {
      axes_group.n_group.emplace(axes_group.n_group.begin(), input_axis[i]);
      remaining_axis.erase(input_axis[i]);
    } else {
      break;
    }
  }

  // 2. 从第一个不同的轴开始处理
  for (; i >= 0; --i) {
    // 看输入的轴是否在y_group中，如果不在，则放入x_group
    if (std::find(axes_group.y_group.begin(), axes_group.y_group.end(), input_axis[i]) == axes_group.y_group.end()) {
      axes_group.x_group.emplace(axes_group.x_group.begin(), input_axis[i]);
      remaining_axis.erase(input_axis[i]);
    }
    // 看输出的轴是否在x_group中，如果不在则放入y_group中
    if (std::find(axes_group.x_group.begin(), axes_group.x_group.end(), output_axis[i]) == axes_group.x_group.end()) {
      axes_group.y_group.emplace(axes_group.y_group.begin(), output_axis[i]);
      remaining_axis.erase(output_axis[i]);
    }
    // 如果都在，则停止，将剩余的轴放入y_group
    if (std::find(axes_group.y_group.begin(), axes_group.y_group.end(), input_axis[i]) != axes_group.y_group.end() &&
        std::find(axes_group.x_group.begin(), axes_group.x_group.end(), output_axis[i]) != axes_group.x_group.end()) {
      // 将剩余的轴放入y_group
      PlaceRemainingAxis(i, remaining_axis, axes_group, output_axis);
      break;
    }
  }

  for (int64_t axis : axes_group.x_group) {
    axes_group.axes_order.emplace_back(axis);
  }

  for (int64_t axis : axes_group.y_group) {
    axes_group.axes_order.emplace_back(axis);
  }

  GELOGD("GenTransposeTilingGroup TilingGroup %s", axes_group.ToString().c_str());
  return af::SUCCESS;
}

Status TilingGroup::GenConcatTilingGroup(af::AscNode &node, AxisGroup &axes_group) {
  std::vector<ascir::AxisId> axes;
  GE_CHK_STATUS_RET(ScheduleUtils::GetLoopAxis(node, axes), "Get loop axis failed.");
  const std::vector<ascir::SizeExpr> &input_repeats = node.inputs[0].attr.repeats;    // 前端保证
  const std::vector<ascir::SizeExpr> &output_repeats = node.outputs[0].attr.repeats;  // 前端保证
  GE_ASSERT_TRUE((input_repeats.size() == output_repeats.size()),
                 "The output dim cnt [%zu] of concat mismatch with input dim cnt [%zu].", output_repeats.size(),
                 input_repeats.size());

  size_t concat_dim{0UL};
  for (size_t i = 0UL; i < input_repeats.size(); ++i) {
    if (af::SymbolicUtils::StaticCheckEq(input_repeats[i], output_repeats[i]) != af::TriBool::kTrue) {
      GELOGD("Concat node [%s], input_repeats[%zu]=%s, output_repeats[%zu]=%s", node.GetNamePtr(), i,
             input_repeats[i].Str().get(), i, output_repeats[i].Str().get());
      concat_dim = i;
      break;
    }
  }
  GELOGD("Concat node [%s], concat_dim is %zu", node.GetNamePtr(), concat_dim);
  axes_group.axes_order.reserve(axes.size());
  for (size_t i = 0UL; i < axes.size(); ++i) {
    if (i < concat_dim) {
      axes_group.y_group.push_back(axes[i]);
      axes_group.axes_order.push_back(i);
    } else {
      axes_group.n_group.push_back(axes[i]);
    }
  }
  // 由于concat的输入和输出大小不同，codegen需要输入输出在concat_dim上用不同符号来表达
  // 因此需要同时将输入和输出的concat_dim的axis_id加到中ngroup中
  for (auto &input_view : node.inputs()) {
    if (input_view->attr.axis.size() > concat_dim) {
      axes_group.n_group.push_back(input_view->attr.axis[concat_dim]);
    }
  }
  return af::SUCCESS;
}

Status TilingGroup::GenSplitTilingGroup(af::AscNode &node, AxisGroup &axes_group) {
  std::vector<ascir::AxisId> axes;
  GE_CHK_STATUS_RET(ScheduleUtils::GetLoopAxis(node, axes), "Get loop axis failed.");
  const std::vector<ascir::SizeExpr> &input_repeats = node.inputs[0].attr.repeats;    // 前端保证
  const std::vector<ascir::SizeExpr> &output_repeats = node.outputs[0].attr.repeats;  // 前端保证
  GE_ASSERT_TRUE((input_repeats.size() == output_repeats.size()),
                 "The output dim cnt [%zu] of split mismatch with input dim cnt [%zu].", output_repeats.size(),
                 input_repeats.size());

  ascir::SizeExpr pre_size = af::ops::One;
  size_t split_dim{0UL};
  for (size_t i = 0UL; i < input_repeats.size(); ++i) {
    if (af::SymbolicUtils::StaticCheckEq(input_repeats[i], output_repeats[i]) != af::TriBool::kTrue) {
      GELOGD("Split node [%s], input_repeats[%zu]=%s, output_repeats[%zu]=%s, pre_size=%s.", node.GetNamePtr(), i,
             input_repeats[i].Str().get(), i, output_repeats[i].Str().get(), pre_size.Str().get());
      split_dim = i;
      break;
    }
    pre_size = pre_size * input_repeats[i];
  }
  GELOGD("Split node [%s], split_dim is %zu", node.GetNamePtr(), split_dim);
  // 首轴Split在schedule前会被替换成store，因此canfuse会按照elewise生成AxisGroup
  if (split_dim == 0UL || (af::SymbolicUtils::StaticCheckEq(pre_size, af::ops::One) == af::TriBool::kTrue)) {
    GE_CHK_STATUS_RET(ScheduleUtils::GetLoopAxis(node, axes_group.y_group));
    axes_group.axes_order.reserve(axes_group.y_group.size());
    for (size_t i = 0UL; i < axes_group.y_group.size(); ++i) {
      axes_group.axes_order.push_back(i);
    }
    return af::SUCCESS;
  }

  axes_group.axes_order.reserve(axes.size());
  for (size_t i = 0UL; i < axes.size(); ++i) {
    if (i < split_dim) {
      axes_group.y_group.push_back(axes[i]);
      axes_group.axes_order.push_back(i);
    } else {
      axes_group.n_group.push_back(axes[i]);
    }
  }
  // 由于split的输入和输出大小不同，codegen需要输入输出在split_dim上用不同符号来表达
  // 因此需要同时将输入和输出的split_dim的axis_id加到中ngroup中
  for (auto &input_view : node.outputs()) {
    if (input_view->attr.axis.size() > split_dim) {
      axes_group.n_group.push_back(input_view->attr.axis[split_dim]);
    }
  }
  return af::SUCCESS;
}

void TilingGroup::NormGroup(AxisGroup &group) {
  if (group.x_group.empty()) {
    group.x_group.push_back(kDefaultGroup);
  }
  if (group.r_group.empty()) {
    group.r_group.push_back(kDefaultGroup);
  }
}

// tiling group的生成应该考虑api的能力，特别是当api有明确的轴不可切时，应当放入Ngroup
Status TilingGroup::GenAxisGroupForSingleNode(af::AscNode &node, AxisGroup &axes_group, bool is_reduce_ar_fullLoad) {
  static std::map<ascir::ComputeType, AxisGroupGenFunc> compute_type_to_group_gen_func = {
      {af::ComputeType::kComputeElewise, TilingGroup::GenElewiseTilingGroup},
      {af::ComputeType::kComputeBroadcast, TilingGroup::GenElewiseTilingGroup},
      {af::ComputeType::kComputeGather, TilingGroup::GenElewiseTilingGroup},
      {af::ComputeType::kComputeLoad, TilingGroup::GenElewiseTilingGroup},
      {af::ComputeType::kComputeStore, TilingGroup::GenElewiseTilingGroup},
      {af::ComputeType::kComputeReduce, TilingGroup::GenReduceTilingGroup},
      {af::ComputeType::kComputeConcat, TilingGroup::GenConcatTilingGroup},
      {af::ComputeType::kComputeTranspose, TilingGroup::GenTransposeTilingGroup},
      {af::ComputeType::kComputeSplit, TilingGroup::GenSplitTilingGroup},
      {af::ComputeType::kComputeCube, TilingGroup::GenElewiseTilingGroup},
  };

  if (node.attr.api.compute_type == af::ComputeType::kComputeReduce && is_reduce_ar_fullLoad) {
    GE_CHK_STATUS_RET(GenReduceTilingGroupFullLoad(node, axes_group), "Gen tiling case failed, compute type [%u].",
                      static_cast<uint32_t>(node.attr.api.compute_type));
    return af::SUCCESS;
  }

  auto iter = compute_type_to_group_gen_func.find(node.attr.api.compute_type);
  GE_ASSERT_TRUE(iter != compute_type_to_group_gen_func.end(), "Unsupported compute type [%u].",
                 node.attr.api.compute_type);
  GE_CHK_STATUS_RET(iter->second(node, axes_group), "Gen tiling case failed, compute type [%u].",
                    static_cast<uint32_t>(node.attr.api.compute_type));
  return af::SUCCESS;
}

}  // namespace optimize::autoschedule
