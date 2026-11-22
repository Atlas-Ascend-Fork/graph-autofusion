/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "node_cache_marker.h"
#include <algorithm>
#include <queue>
#include "schedule_utils.h"
#include "common_utils.h"
#include "indirect_load_utils.h"

namespace optimize::autoschedule {
// 获取对端节点的输出attr，作为当前节点的输入attr
af::Status NodeCacheMarker::GetAscNodeInputAttr(const af::NodePtr &node, int32_t idx, af::AscTensorAttr &attr) {
  const auto &asc_node = std::dynamic_pointer_cast<af::AscNode>(node);
  GE_ASSERT_NOTNULL(asc_node);
  GE_ASSERT_TRUE(static_cast<uint32_t>(idx) < asc_node->GetAllInDataAnchorsSize());
  GE_ASSERT_NOTNULL(asc_node->GetInDataAnchor(idx));
  const auto &pre_out_anchor = asc_node->GetInDataAnchor(idx)->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(pre_out_anchor);
  GE_ASSERT_NOTNULL(pre_out_anchor->GetOwnerNode());
  const auto &pre_asc_node = std::dynamic_pointer_cast<af::AscNode>(pre_out_anchor->GetOwnerNode());
  GE_ASSERT_NOTNULL(pre_asc_node);
  GE_ASSERT_TRUE(pre_asc_node->GetOutDataNodesSize() > static_cast<uint32_t>(pre_out_anchor->GetIdx()));
  attr = pre_asc_node->outputs[pre_out_anchor->GetIdx()].attr;
  return af::SUCCESS;
}

bool NodeCacheMarker::IsNodeVisited(const af::NodePtr &node) const {
  return visited_nodes_.find(node) != visited_nodes_.end();
}

void NodeCacheMarker::VisitNode(const af::NodePtr &node) {
  visited_nodes_.insert(node);
}

void NodeCacheMarker::AddToCacheStartSet(const af::NodePtr &node) {
  cache_start_nodes_.insert(node);
}

af::ExecuteCondition NodeCacheMarker::DoesNodeNeedCache(const std::vector<int64_t> &in_axis,
                                                        const std::vector<int64_t> &out_axis,
                                                        const std::vector<af::Expression> &in_repeats,
                                                        const std::vector<af::Expression> &out_repeats) const {
  if (in_axis.size() != out_axis.size() || in_repeats.size() != out_repeats.size() ||
      in_axis.size() != in_repeats.size() || out_axis.empty()) {
    GELOGD(
        "NodeCacheMarker skips cache analysis: in/out views are not positionally comparable (in rank %zu, out rank "
        "%zu).",
        in_axis.size(), out_axis.size());
    return af::ExecuteCondition::kNoCache;
  }

  // 构造 <轴id，是否广播轴> Map
  std::map<int64_t, bool> axis_id_brc_map;
  for (size_t i = 0UL; i < out_axis.size(); ++i) {
    axis_id_brc_map[out_axis[i]] = !ascgen_utils::ExpressEq(in_repeats[i], out_repeats[i]);  // 不相等，就是广播轴
  }

  // y.axis，不含Reduce场景：ORIGINAL(不一定有)、BLOCK_OUT、BLOCK_IN、TILE_OUT、TILE_IN、ORIGINAL(不一定有)
  // y.axis，含Reduce场景：ORIGINAL(不一定有)、BLOCK_OUT、TILE_OUT、BLOCK_IN、TILE_IN、TILE_OUT、TILE_IN、ORIGINAL(不一定有)
  std::vector<int64_t> tile_in_axes_idx;
  std::vector<int64_t> other_axes_idx;
  tile_in_axes_idx.reserve(out_axis.size());
  other_axes_idx.reserve(out_axis.size());
  // 倒序遍历输出轴，过滤掉ORIGINAL类型的轴
  bool skip_last_original_axis = true;
  for (auto it = out_axis.rbegin(); it != out_axis.rend(); ++it) {
    const auto &axis = *it;
    const auto &axis_ptr = this->graph_.FindAxis(axis);
    GE_ASSERT_NOTNULL(axis_ptr, "Not found axis=(%ld).", axis);
    // 跳过尾部的ORIGINAL轴
    if (axis_ptr->type != af::Axis::kAxisTypeOriginal) {
      skip_last_original_axis = false;
    }
    if (skip_last_original_axis) {
      continue;
    }
    // 从最后一个TILE_IN开始，倒序处理
    if (axis_ptr->type == af::Axis::kAxisTypeTileInner) {
      tile_in_axes_idx.push_back(axis);
    } else {
      other_axes_idx.push_back(axis);
    }
  }

  bool is_tile_in_brc = std::any_of(tile_in_axes_idx.begin(), tile_in_axes_idx.end(),
                                    [&axis_id_brc_map](const auto axis) { return axis_id_brc_map[axis]; });
  bool is_origin_axes_brc = std::all_of(other_axes_idx.begin(), other_axes_idx.end(),
                                        [&axis_id_brc_map](const auto axis) { return axis_id_brc_map[axis]; });
  // 若TILE_IN及以前所有轴都是广播轴，则属于 kCacheBlockSplitOriginBroadcastAxis
  if (is_origin_axes_brc && is_tile_in_brc) {
    return af::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis;
  }
  // 若只有TILE_IN轴是广播轴，则属于 kCacheBlockSplitFusedBroadcastAxis
  if (is_tile_in_brc) {
    return af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis;
  }
  // 其他情况，不支持缓存
  return af::ExecuteCondition::kNoCache;
}

af::ExecuteCondition NodeCacheMarker::DoesNodeNeedCache(const af::AscNodePtr &node) const {
  // 各种安全校验
  GE_ASSERT_TRUE(node->GetInDataNodesSize() > 0UL);
  GE_ASSERT_TRUE(node->GetOutDataNodesSize() > 0U);

  std::vector<int64_t> in_axis;
  std::vector<af::Expression> in_repeats;
  if (ScheduleUtils::IsBroadcast(node) && ScheduleUtils::IsScalarLikeNode(node->GetInDataNodes().at(0))) {
    // Scalar+Broadcast
    GELOGD("Graph [%s], find scalar broadcast [%s]", graph_.GetName().c_str(), node->GetNamePtr());
    in_axis = node->outputs[0].attr.axis;
    const std::vector<af::Expression> all_one_repeats(in_axis.size(), af::ops::One);
    in_repeats = all_one_repeats;
  } else {
    in_axis = node->inputs[0].attr.axis;
    in_repeats = node->inputs[0].attr.repeats;
  }
  const auto &out_axis = node->outputs[0].attr.axis;
  const auto &out_repeats = node->outputs[0].attr.repeats;
  return DoesNodeNeedCache(in_axis, out_axis, in_repeats, out_repeats);
}

af::ExecuteCondition NodeCacheMarker::DoesNodeNeedCache(const af::NodePtr &node) const {
  const auto &asc_node = std::dynamic_pointer_cast<af::AscNode>(node);
  GE_ASSERT_NOTNULL(asc_node);
  return DoesNodeNeedCache(asc_node);
}

af::ExecuteCondition NodeCacheMarker::DoesInlineNodeNeedCache(const af::NodePtr &node, int32_t brc_idx) const {
  af::AscTensorAttr in_attr;
  GE_ASSERT_SUCCESS(GetAscNodeInputAttr(node, brc_idx, in_attr));
  af::AscTensorAttr out_attr;
  GE_ASSERT_SUCCESS(GetAscNodeInputAttr(node, 1 - brc_idx, out_attr));

  const auto &in_axis = in_attr.axis;
  const auto &in_repeats = in_attr.repeats;
  const auto &out_axis = out_attr.axis;
  const auto &out_repeats = out_attr.repeats;
  return DoesNodeNeedCache(in_axis, out_axis, in_repeats, out_repeats);
}

void NodeCacheMarker::MarkNodeCacheable(const af::NodePtr &node) {
  const auto &asc_node = std::dynamic_pointer_cast<af::AscNode>(node);
  if (asc_node != nullptr) {
    // 由于引入精度问题，暂时关闭，待后续找到具体片段后再定位
    asc_node->attr.sched.exec_condition = af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis;
  }
}

/**
 * 把node及node的所有父节点标记为可缓存
 */
void NodeCacheMarker::MarkNodesCacheableBottomUp(const af::AscNodePtr &node,
                                                 [[maybe_unused]] const af::ExecuteCondition condition) {
  std::queue<af::NodePtr> queue;
  queue.push(node);
  while (!queue.empty()) {
    const auto &tmp_node = queue.front();
    VisitNode(tmp_node);
    queue.pop();

    // 遇到 Load 节点，说明已到达图的Data节点，为防止一个Data接多个load，将另一侧的load也标记缓存的情况此时直接截断。
    const auto &asc_tmp_node = std::dynamic_pointer_cast<af::AscNode>(tmp_node);

    if (asc_tmp_node != nullptr && ScheduleUtils::IsLoad(asc_tmp_node)) {
      GELOGD(
          "Graph [%s], find Load node [%s] as cache start boundary, truncate to avoid multi-output pollution from Data "
          "node.",
          graph_.GetName().c_str(), tmp_node->GetNamePtr());

      MarkNodeCacheable(tmp_node);   // 确保当前 Load 节点自身被正常标记缓存
      AddToCacheStartSet(tmp_node);  // 将 Load 作为正向刷新的 Start 节点
      continue;                      // 截断！
    }
    if (tmp_node->GetInDataNodesSize() == 0UL) {
      AddToCacheStartSet(tmp_node);
      continue;
    }
    MarkNodeCacheable(tmp_node);
    for (const auto &in_node : tmp_node->GetInDataNodes()) {
      queue.push(in_node);
    }
  }
}

void NodeCacheMarker::MarkNodesCacheableBottomUp(const af::NodePtr &node, const af::ExecuteCondition condition) {
  const auto &asc_node = std::dynamic_pointer_cast<af::AscNode>(node);
  if (asc_node != nullptr) {
    MarkNodesCacheableBottomUp(asc_node, condition);
  }
}

void NodeCacheMarker::MarkNodesCacheableUpBottom(const af::NodePtr &node) {
  std::queue<af::NodePtr> queue;
  queue.push(node);
  while (!queue.empty()) {
    const auto &tmp_node = queue.front();
    [[maybe_unused]] const auto exec_condition = ascgen_utils::GetNodeExecCondition(tmp_node);
    queue.pop();
    for (const auto &out_node : tmp_node->GetOutDataNodes()) {
      // 若子节点为多输入，则不会缓存
      // 若子节点是Store，则不会缓存
      if (out_node->GetInDataNodesSize() != 1UL || af::ops::IsOps<af::ascir_op::Store>(out_node)) {
        continue;
      }
      // 否则，若没有标记缓存，则将其标记为缓存，并加入队列继续遍历子节点
      if (!ascgen_utils::IsNodeCacheable(out_node)) {
        MarkNodeCacheable(out_node);
      }
      queue.push(out_node);
    }
  }
}

af::Status NodeCacheMarker::ReverseDfsCacheNode(const af::NodePtr &ge_node) {
  if (ScheduleUtils::IsIOBuffer(ge_node) || ge_node->GetInDataNodesSize() == 0UL) {
    return af::SUCCESS;
  }
  if (IsNodeVisited(ge_node)) {
    return af::SUCCESS;
  }
  VisitNode(ge_node);

  const auto &node = std::dynamic_pointer_cast<af::AscNode>(ge_node);
  GE_ASSERT_NOTNULL(node);
  // Stage1，step2: Broadcast场景，判断是否需要缓存
  if (ScheduleUtils::IsBroadcast(node)) {
    const auto condition = DoesNodeNeedCache(node);
    if (condition != af::ExecuteCondition::kNoCache) {
      MarkNodeCacheable(node);
      MarkNodesCacheableBottomUp(node, condition);
      GELOGD("Graph(%s) Broadcast(%s) supports brc cache.", graph_.GetName().c_str(), node->GetNamePtr());
      return af::SUCCESS;
    }
  }
  // Stage1，其他节点
  for (const auto &in_node : node->GetInDataNodes()) {
    GE_WARN_ASSERT(ReverseDfsCacheNode(in_node) == af::SUCCESS);
  }
  return af::SUCCESS;
}

/**
 * 标记每个节点是否需要缓存（输出），逻辑分为两个阶段：\n
 * 阶段一：倒序查找
 *   1. 倒序深度遍历每个模板的每个节点; \n
 *   2. 若遇到RemovePad节点，其输入只有1个且为Broadcast节点，且满足缓存条件，则将RemovePad及其所有父节点标记为可缓存，
 *      把RemovePad的root节点保存至StartNodesSet; \n
 *   3. 若遇到Broadcast节点，其输入只有1个且满足缓存条件，则将Broadcast及其所有父节点标记为可缓存，
 *      把Broadcast的root节点保存至StartNodesSet; \n
 *   4.
 * 若遇到两输入且其中一路存在隐式广播，则检查广播这一路是否满足缓存条件（记为A），若满足则将A及其所有父节点标记为可缓存，
 *      把A的root节点保存至StartNodesSet. \n
 * 阶段二：正向刷新
 *   1. 遍历StartNodesSet，若其子节点输入个数为1，且未被标记缓存，则标记为可缓存。
 */
af::Status NodeCacheMarker::MarkIfNodeNeedsCache() {
  // 若子图里有Transpose节点，则不缓存，否则后续transpose双切分会有精度问题
  if (ScheduleUtils::HasComputeType(graph_, af::ComputeType::kComputeTranspose)) {
    return af::SUCCESS;
  }
  // Stage1，step1: 倒序查找，先收集所有的Store节点，因为Output只有1个输入且必定是Store
  std::vector<af::NodePtr> store_nodes;
  for (const auto &node : graph_.GetAllNodes()) {
    if (ScheduleUtils::IsStore(node)) {
      store_nodes.push_back(node);
    }
  }
  if (store_nodes.empty()) {
    GELOGD("Graph(%s) has no store node, returning.", graph_.GetName().c_str());
    return af::SUCCESS;
  }
  visited_nodes_.clear();
  cache_start_nodes_.clear();

  // Gather+Norm 中 weight/bias 的 Broadcast 可能在调度前被判定为冗余并删除。
  // 此时参数 Load 仍是固定地址的跨循环输入，但已失去 Broadcast 反向遍历
  // 建立的缓存起点；显式恢复该缓存起点，保证参数只在缓存边界搬运一次。
  if (ascgen_utils::indirect_load::FindIndirectLoadNode(graph_) != nullptr) {
    for (const auto &node : graph_.GetAllNodes()) {
      if (!ScheduleUtils::IsLoad(node) || node->outputs().empty() || node->GetOutDataNodesSize() == 0UL) {
        continue;
      }
      const auto &output = node->outputs()[0]->attr;
      const bool is_broadcast_load = std::any_of(output.strides.begin(), output.strides.end(), [](const auto &stride) {
        return af::SymbolicUtils::StaticCheckEq(stride, af::sym::kSymbolZero) == af::TriBool::kTrue;
      });
      if (!is_broadcast_load) {
        continue;
      }
      const auto consumer = std::dynamic_pointer_cast<af::AscNode>(*node->GetOutDataNodes().begin());
      if (consumer == nullptr || !ScheduleUtils::IsElewise(consumer)) {
        continue;
      }
      MarkNodeCacheable(node);
      AddToCacheStartSet(node);
      GELOGD("[IndirectLoad] Mark fixed-parameter Load[%s] as cache start for consumer[%s].", node->GetNamePtr(),
             consumer->GetNamePtr());
    }
  }
  for (const auto &node : store_nodes) {
    GE_WARN_ASSERT(ReverseDfsCacheNode(node) == af::SUCCESS);
  }

  // Stage2，正向刷新
  for (const auto &node : cache_start_nodes_) {
    MarkNodesCacheableUpBottom(node);
  }
  return af::SUCCESS;
}

}  // namespace optimize::autoschedule
